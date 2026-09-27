"""Daily trading run on real market data. Schedule for ~08:45 America/New_York on trading days.

  shadow: computes targets and orders from live data; logs them and tracks a simulated book
  paper : sends market-on-open orders to the Alpaca PAPER account (virtual money)

Decisions use bars through the previous session's close; orders fill at today's open.
"""
import _path  # noqa: F401

import argparse
import json
from datetime import timedelta

import numpy as np
import pandas as pd

from trader import alpaca_io as io
from trader.config import ROOT, path, risk_limits, settings
from trader.model import load
from trader.pipeline import matrices, stock_block
from trader.risk import RiskEngine
from trader.shadow import SHADOW_BOOK, load_shadow_book, save_shadow_book
from trader.strategies import etf_trend_weights, stock_ml_weights


def latest_model():
    files = sorted((ROOT / "models").glob("stock_lgbm_*.txt"))
    if not files:
        raise SystemExit("no model found: run scripts/train_model.py first")
    return files[-1]


def flatten(tc, mode, why):
    print(f"FLATTEN: {why}")
    if mode == "paper":
        tc.close_all_positions(cancel_orders=True)
    else:
        SHADOW_BOOK.write_text("{}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["shadow", "paper"])
    args = ap.parse_args()
    S, L = settings(), risk_limits()
    mode = args.mode or S["mode"]
    cs, ce = S["stock_ml"], S["etf_trend"]

    tc, dc = io.trading_client(), io.data_client()
    now = pd.Timestamp.now(tz="America/New_York")
    today = now.normalize().tz_localize(None)
    days = io.trading_days(tc, (today - timedelta(days=15)).date(), today.date())
    if today not in days:
        print(f"{today.date()}: market closed, nothing to do")
        return
    prev_session = [d for d in days if d < today][-1]
    clock = tc.get_clock()
    minute = now.hour * 60 + now.minute
    if not clock.is_open and minute > 9 * 60 + 27:
        print("after 09:27 ET with market closed: too late for opening orders, run again before the open")
        return
    at_open = not clock.is_open
    logdir = path("logs", str(today.date()), "x").parent
    logdir.mkdir(parents=True, exist_ok=True)

    risk = RiskEngine(L)
    acct = tc.get_account()
    equity, last_equity = float(acct.equity), float(acct.last_equity)
    print(f"[{mode}] {now:%Y-%m-%d %H:%M} ET  equity ${equity:,.2f}  prev close ${last_equity:,.2f}")
    if risk.kill_requested():
        return flatten(tc, mode, f"KILL file present: {(ROOT / 'KILL').read_text().strip()}")
    reason = risk.check_equity(equity, last_equity)
    if reason:
        risk.halt(reason)
        return flatten(tc, mode, reason)

    # ---- real market data (enough history for 252-day features + 60-day universe window)
    assets = io.list_equities(tc)
    syms = sorted(set(assets.symbol) | set(ce["symbols"]))
    bars = io.fetch_daily_bars(dc, syms, (today - timedelta(days=620)).date(), feed=S["feed"])
    bars = bars[bars.date < today]
    last = bars.date.max()
    if last != prev_session:
        print(f"STALE DATA: last bar {last.date()} but previous session was {prev_session.date()}; no trading")
        return
    M = matrices(bars)
    px = M["close"].iloc[-1]

    # ---- stock ML sleeve
    mpath = latest_model()
    model, meta = load(mpath)
    age = (today - pd.Timestamp(meta["trained_on"])).days
    if age > 45:
        print(f"WARNING: model is {age} days old; run scripts/train_model.py --refresh")
    dates = M["close"].index[-15:]  # enough for the prediction EWMA
    panel, mask, _ = stock_block(M, assets, cs, ce["symbols"], dates=dates, with_label=False)
    pred = pd.Series(model.predict(panel[meta["features"]]), index=panel.index)
    Ws = stock_ml_weights(pred, mask, cs["long_short_frac"], cs["gross_leverage"],
                          cs["pred_smoothing_halflife"]).iloc[-1]

    # ---- cross-asset trend sleeve
    We = etf_trend_weights(M["close"][ce["symbols"]], ce["lookbacks"], ce["vol_target"],
                           ce["max_asset_weight"], ce["max_gross"], last_only=True).iloc[-1]

    a = S["sleeves"]
    tgt = (Ws * a["stock_ml"] * equity).add(We * a["etf_trend"] * equity, fill_value=0.0)
    tgt = tgt[tgt.abs() > 0]
    shortable = set(assets.loc[assets.shortable & assets.easy_to_borrow, "symbol"])
    dropped = [s for s, v in tgt.items() if v < 0 and s not in shortable]
    tgt = tgt.drop(dropped)
    targets = risk.cap_targets(tgt.to_dict(), equity)

    # ---- orders
    if mode == "paper":
        cur = io.positions(tc)
    else:
        book = load_shadow_book()
        cur = {s: v["qty"] for s, v in book.items()}
    min_trade = S["execution"]["min_trade_frac"] * equity
    rows, new_book = [], {}
    for s in sorted(set(targets) | set(cur)):
        p = float(px.get(s, np.nan))
        cq = float(cur.get(s, 0.0))
        tq = float(np.trunc(targets.get(s, 0.0) / p)) if np.isfinite(p) and p > 0 else 0.0
        note = ""
        if cq and tq and np.sign(cq) != np.sign(tq):
            tq, note = 0.0, "flip: close today, reopen next session"
        dq = tq - cq
        if tq:
            new_book[s] = tq
        if dq == 0:
            continue
        notional = dq * p if np.isfinite(p) else np.nan
        increasing = abs(tq) > abs(cq)
        skip = None
        if increasing and abs(notional) < min_trade:
            skip = "skip"
        elif increasing and not risk.order_ok(notional, equity):
            skip = "REJECTED fat-finger"
        if skip:
            new_book.pop(s, None)
            if cq:
                new_book[s] = cq
            if skip != "skip":
                rows.append(dict(symbol=s, qty=dq, price=p, notional=notional, status=skip))
            continue
        rows.append(dict(symbol=s, cur_qty=cq, target_qty=tq, qty=dq, price=p, notional=notional,
                         target_usd=targets.get(s, 0.0), note=note, status="shadow"))

    if mode == "paper":
        tc.cancel_orders()
        for r in rows:
            if r["status"] != "shadow":
                continue
            cid = f"{today:%Y%m%d}-{now:%H%M}-{r['symbol']}"
            try:
                o = io.submit_market(tc, r["symbol"], int(r["qty"]), at_open, cid)
                r["status"], r["order_id"] = str(o.status), str(o.id)
            except Exception as e:
                r["status"] = f"ERROR {e}"
    else:
        save_shadow_book(book, new_book, px)

    orders = pd.DataFrame(rows)
    pd.Series(targets, name="target_usd").to_csv(logdir / "targets.csv")
    orders.to_csv(logdir / "orders.csv", index=False)
    gross = sum(abs(v) for v in targets.values())
    net = sum(targets.values())
    summary = dict(mode=mode, time=str(now), equity=equity, last_equity=last_equity, data_through=str(last.date()),
                   model=mpath.name, n_targets=len(targets), gross_lev=gross / equity, net_exposure=net / equity,
                   n_orders=len(orders), order_notional=float(orders["notional"].abs().sum()) if len(orders) else 0.0,
                   dropped_unshortable=dropped, at_open=at_open)
    (logdir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str))
    if len(orders):
        print(orders.groupby("status").size().to_string())


if __name__ == "__main__":
    main()
