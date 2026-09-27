"""Faster trading test on real Binance 1-minute data: can we predict 1 / 5 / 15 / 60 minutes ahead,
and is the edge per trade bigger than the cost of the trade?

Features use only information available at the close of minute t (price, volume, taker-buy order flow,
BTC lead-lag, cross-sectional ranks). Label: sign of open(t+1) -> open(t+1+h). Walk-forward by month,
training on all earlier months, purged at the boundary.

Key output: average return per trade in basis points when trading the model's calls (all calls, and only
its most confident 10% / 1%), compared with round-trip costs:
  taker (market orders, retail): ~12 bp  (2 x 5 bp fee + ~2 bp spread/slippage)
  maker (limit orders, retail):  ~4 bp   (2 x 2 bp fee, if the order fills)
  top-tier market maker:         ~0 bp   (VIP maker fee 0 or rebate)
"""
import argparse
import math
import time
from datetime import date
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "crypto" / "klines_1m.parquet"
COSTS = {"taker (retail market orders)": 12.0, "maker (retail limit orders)": 4.0, "VIP market maker": 0.0}


def build(horizons):
    df = pd.read_parquet(DATA)
    W = {k: df.pivot(index="time", columns="symbol", values=k).sort_index()
         for k in ("open", "high", "low", "close", "volume", "trades", "taker_buy_base")}
    full = pd.date_range(W["close"].index[0], W["close"].index[-1], freq="min")
    W = {k: v.reindex(full) for k, v in W.items()}
    o, h, l, c, v, nt = W["open"], W["high"], W["low"], W["close"], W["volume"], W["trades"]
    syms, idx = list(c.columns), c.index
    bc = lambda s: pd.DataFrame(np.repeat(np.asarray(s, np.float32)[:, None], len(syms), 1), index=idx, columns=syms)
    roll = lambda x, n: x.rolling(n, min_periods=max(2, int(n * 0.8)))

    lr = np.log(c).diff()
    vol60 = roll(lr, 60).std()
    vol1440 = roll(lr, 1440).std()
    imb = W["taker_buy_base"] / v.replace(0, np.nan) - 0.5  # taker buy share: order-flow pressure
    F = {}
    for n in (1, 2, 3, 5, 10, 15, 30, 60, 240):
        F[f"r{n}"] = np.log(c / c.shift(n)) / (vol60 * np.sqrt(n))
    F["vol_ratio"] = np.log(vol60 / vol1440)
    F["lvol60"] = np.log(vol60)
    F["hl1"] = np.log(h / l) / vol60
    F["hl15"] = roll(np.log(h / l), 15).mean() / vol60
    for n in (1, 5, 15, 60):
        F[f"imb{n}"] = imb if n == 1 else (roll(imb * v, n).sum() / roll(v, n).sum())
    lv = np.log(v.replace(0, np.nan))
    F["vz"] = (lv - roll(lv, 60).mean()) / roll(lv, 60).std()
    F["vz15"] = (roll(lv, 15).mean() - roll(lv, 1440).mean()) / roll(lv, 1440).std()
    lt = np.log(nt.replace(0, np.nan))
    F["tz"] = (lt - roll(lt, 60).mean()) / roll(lt, 60).std()
    F["hi60"] = np.log(c / roll(c, 60).max()) / vol60
    F["lo60"] = np.log(c / roll(c, 60).min()) / vol60
    F["ma30"] = np.log(c / roll(c, 30).mean()) / vol60
    for n in (1, 3, 5, 15):
        F[f"btc_r{n}"] = bc(F[f"r{n}"]["BTC"])
    F["btc_imb5"] = bc(F["imb5"]["BTC"])
    F["mkt_r5"] = bc(F["r5"].mean(axis=1))
    F["mkt_imb5"] = bc(F["imb5"].mean(axis=1))
    for k in ("r5", "r15", "r60", "imb5", "vz"):
        F[f"xs_{k}"] = F[k].rank(axis=1, pct=True) - 0.5
    mins = idx.hour * 60 + idx.minute
    F["tod_sin"], F["tod_cos"] = bc(np.sin(2 * np.pi * mins / 1440)), bc(np.cos(2 * np.pi * mins / 1440))

    names = list(F)
    T, S = len(idx), len(syms)
    X = np.empty((T * S, len(names)), np.float32)
    for j, k in enumerate(names):
        X[:, j] = F[k].to_numpy(np.float32).reshape(-1)
    X[~np.isfinite(X)] = np.nan
    ok = ~np.isnan(X).any(axis=1)
    rets = {hz: np.log(o.shift(-1 - hz) / o.shift(-1)).to_numpy().reshape(-1) for hz in horizons}
    return dict(X=X, ok=ok, rets=rets, tidx=np.repeat(np.arange(T), S), times=idx, syms=syms, S=S, names=names)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizons", default="1,5,15,60")
    ap.add_argument("--test_months", type=int, default=6)
    ap.add_argument("--max_train", type=int, default=4_000_000, help="subsample training rows per fold")
    a = ap.parse_args()
    H = [int(x) for x in a.horizons.split(",")]
    t0 = time.time()
    D = build(H)
    X, ok, tidx, times, S = D["X"], D["ok"], D["tidx"], D["times"], D["S"]
    print(f"rows {len(X):,} ({ok.mean():.0%} usable) · {len(D['names'])} features · {times[0]} .. {times[-1]} "
          f"· built in {time.time() - t0:.0f}s", flush=True)
    months = pd.date_range(times[-1].normalize().replace(day=1) - pd.DateOffset(months=a.test_months - 1),
                           times[-1], freq="MS")
    rng = np.random.default_rng(0)
    lines = [f"# Fast-horizon test on real Binance 1-minute data: {date.today()}", "",
             f"{S} coins · {times[0]:%Y-%m-%d} .. {times[-1]:%Y-%m-%d %H:%M} UTC · walk-forward over the last "
             f"{a.test_months} months (monthly retrain, expanding window) · {len(D['names'])} features "
             "(price, volume, taker order flow, BTC lead-lag, cross-sectional ranks)", "",
             "Round-trip cost assumptions: " + " · ".join(f"{k}: {v:.0f} bp" for k, v in COSTS.items()), ""]
    summary = []
    for hz in H:
        ret = D["rets"][hz]
        y = (ret > 0).astype(np.float32)
        usable = ok & np.isfinite(ret) & (ret != 0)
        pred = np.full(len(y), np.nan)
        for mi, ms in enumerate(months):
            me = months[mi + 1] if mi + 1 < len(months) else times[-1] + pd.Timedelta(minutes=1)
            ts, te = times.searchsorted(ms), times.searchsorted(me)
            tr = np.flatnonzero(usable & (tidx < ts - hz - 1))
            if len(tr) > a.max_train:
                tr = np.sort(rng.choice(tr, a.max_train, replace=False))
            cut = tidx[tr[int(len(tr) * 0.9)]]
            fit, val = tr[tidx[tr] < cut - hz - 1], tr[tidx[tr] >= cut]
            m = lgb.train(dict(objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=5000,
                               feature_fraction=0.7, bagging_fraction=0.7, bagging_freq=1, lambda_l2=10,
                               num_threads=36, verbose=-1, seed=0),
                          lgb.Dataset(X[fit], y[fit]), 1000, valid_sets=[lgb.Dataset(X[val], y[val])],
                          callbacks=[lgb.early_stopping(50, verbose=False)])
            te_rows = np.flatnonzero(usable & (tidx >= ts) & (tidx < te))
            pred[te_rows] = m.predict(X[te_rows], num_iteration=m.best_iteration)
            print(f"  h={hz}m month {ms:%Y-%m}: train {len(fit):,} test {len(te_rows):,} trees {m.best_iteration} "
                  f"AUC {roc_auc_score(y[te_rows], pred[te_rows]):.4f}", flush=True)
            if mi == len(months) - 1:
                imp = pd.Series(m.feature_importance("gain"), index=D["names"]).sort_values(ascending=False)
        ev = np.flatnonzero(np.isfinite(pred))
        p, yy, rr = pred[ev], y[ev], ret[ev]
        acc = ((p > 0.5) == (yy == 1)).mean()
        conf = np.abs(p - 0.5)
        signed = np.sign(p - 0.5) * rr * 1e4  # bp earned per call if you trade the model's direction
        cut10, cut1 = np.quantile(conf, 0.9), np.quantile(conf, 0.99)
        top10, top1 = conf >= cut10, conf >= cut1
        acc10 = ((p[top10] > 0.5) == (yy[top10] == 1)).mean()
        acc1 = ((p[top1] > 0.5) == (yy[top1] == 1)).mean()
        rev = -np.sign(D["X"][ev, D["names"].index("r1")])  # simple rule: fade the last minute
        flow = np.sign(D["X"][ev, D["names"].index("imb5")])  # simple rule: follow taker order flow
        row = dict(h=hz, n=len(ev), up_rate=yy.mean(), acc=acc, auc=roc_auc_score(yy, p), acc_top10=acc10, acc_top1=acc1,
                   bp_all=signed.mean(), bp_top10=signed[top10].mean(), bp_top1=signed[top1].mean(),
                   bp_rule_reversal=(rev * rr * 1e4).mean(), bp_rule_orderflow=(flow * rr * 1e4).mean(),
                   median_abs_move_bp=np.median(np.abs(rr)) * 1e4, trades_top1_per_day=top1.sum() / (len(np.unique(tidx[ev])) / 1440))
        summary.append(row)
        lines += [f"## Horizon {hz} minute(s)", "",
                  f"- scored calls: {len(ev):,} · actual up-rate {yy.mean():.1%} · typical move |r| {row['median_abs_move_bp']:.1f} bp",
                  f"- accuracy **{acc:.1%}** (AUC {row['auc']:.3f}) · most confident 10%: **{acc10:.1%}** · most confident 1%: **{acc1:.1%}**",
                  f"- average edge per call: all {row['bp_all']:+.2f} bp · top 10% {row['bp_top10']:+.2f} bp · top 1% {row['bp_top1']:+.2f} bp",
                  f"- simple rules for comparison: fade last minute {row['bp_rule_reversal']:+.2f} bp · follow order flow {row['bp_rule_orderflow']:+.2f} bp",
                  "- profit per trade after round-trip costs (top 1% calls): " + " · ".join(
                      f"{k}: {row['bp_top1'] - c:+.2f} bp" for k, c in COSTS.items()),
                  f"- top features: {', '.join(imp.index[:8])}", ""]
    Sm = pd.DataFrame(summary).set_index("h")
    lines += ["## Summary", "", Sm.to_markdown(floatfmt=".3f"), ""]
    out = ROOT / "reports" / f"crypto_fast_horizons_{date.today()}.md"
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"report -> {out}  (total {(time.time() - t0) / 60:.0f} min)")


if __name__ == "__main__":
    main()
