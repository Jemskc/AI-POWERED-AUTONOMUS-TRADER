"""Live forward-test report: P&L of each virtual book + accuracy of every live prediction so far.

Writes reports/crypto_live_<date>.md and prints it.
"""
import _path  # noqa: F401

import json
import sqlite3
from datetime import date, datetime, timezone

import numpy as np
import pandas as pd

from trader.config import ROOT
from trader.crypto import market
from trader.crypto.features import SYMBOLS

SD = ROOT / "state" / "crypto"


def auc(p, y):
    if len(set(y)) < 2:
        return float("nan")
    r = pd.Series(p).rank().to_numpy()
    n1 = y.sum()
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(y) - n1))


def main():
    tk = market.tickers(SYMBOLS)
    now = datetime.now(timezone.utc)
    hb = json.loads((SD / "heartbeat.json").read_text()) if (SD / "heartbeat.json").exists() else {}
    age = (now - datetime.fromisoformat(hb["time"])).total_seconds() if hb else None
    L = [f"# Live crypto forward test: {now:%Y-%m-%d %H:%M} UTC", "",
         f"Engine heartbeat: {hb.get('time', 'never')} ({'LIVE' if age is not None and age < 120 else 'NOT RUNNING'}"
         f"{f', {age:.0f}s ago' if age is not None else ''}) · last decision candle {hb.get('last_decision')}", ""]

    L += ["## Virtual books ($100k each, real live prices, fees + spread + slippage charged)", "",
          "| book | started (UTC) | hours | equity | P&L | return | buy&hold 20 coins | BTC | trades | fees | positions (L/S) | gross | status |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    details = []
    for f in sorted(SD.glob("broker_*.json")):
        b = json.loads(f.read_text())
        pos = b["positions"]
        lv = sum(p["qty"] * tk[s]["last"] for s, p in pos.items() if p["qty"] > 0)
        sv = sum(p["qty"] * tk[s]["last"] for s, p in pos.items() if p["qty"] < 0)
        eq = b["cash"] + lv + sv
        bh = np.mean([tk[s]["last"] / p for s, p in b["start_prices"].items()]) - 1
        btc = tk["BTC"]["last"] / b["start_prices"]["BTC"] - 1
        hrs = (now - datetime.fromisoformat(b["start_time"])).total_seconds() / 3600
        nl, ns = sum(p["qty"] > 0 for p in pos.values()), sum(p["qty"] < 0 for p in pos.values())
        rule = "long-only rotation" if b.get("kind") == "rotation" else f"thr {b['threshold']}"
        L.append(f"| **{b['book']}** ({rule}) | {b['start_time'][:16]} | {hrs:.1f} | ${eq:,.2f} | "
                 f"{eq - b['start_capital']:+,.2f} | **{eq / b['start_capital'] - 1:+.2%}** | {bh:+.2%} | {btc:+.2%} | "
                 f"{b['n_trades']} | ${b['fees']:,.2f} | {nl}/{ns} | {(lv - sv) / eq:.2f}x | "
                 f"{'HALTED: ' + b['halted'] if b['halted'] else 'trading'} |")
        if pos:
            rows = sorted(((s, p["qty"], p["avg"], tk[s]["last"], (tk[s]["last"] - p["avg"]) * p["qty"])
                           for s, p in pos.items()), key=lambda r: -abs(r[1] * r[3]))
            details += ["", f"<details><summary>{b['book']} positions</summary>", "",
                  "| coin | side | value | entry | now | unrealized |", "|---|---|---|---|---|---|"]
            details += [f"| {s} | {'LONG' if q > 0 else 'SHORT'} | ${abs(q * px):,.0f} | {a:,.4f} | {px:,.4f} | {u:+,.2f} |"
                  for s, q, a, px, u in rows]
            details += ["", "</details>"]

    L += details
    con = sqlite3.connect(SD / "predictions.sqlite")
    P = pd.read_sql("SELECT * FROM preds", con)
    S = P.dropna(subset=["ret"])
    L += ["", "## Live prediction accuracy (each hour, all 20 coins: will it be up 4h later?)", "",
          f"Predictions made: **{len(P):,}** over {P.t.nunique()} hours · scored (4h outcome known): **{len(S):,}**", ""]
    if len(S):
        y = S.up.to_numpy()
        L += [f"Actual up-rate of scored predictions: {y.mean():.1%} (the 'always up' baseline)", "",
              "| model | accuracy | vs always-up | AUC |", "|---|---|---|---|",
              f"| always up | {y.mean():.1%} | 0.0 pts | 0.500 |"]
        for m in ("p_xgb", "p_lgbm", "p_ens"):
            acc = ((S[m] > 0.5) == (y == 1)).mean()
            L.append(f"| {m[2:]} | **{acc:.1%}** | {(acc - y.mean()) * 100:+.1f} pts | {auc(S[m].to_numpy(), y):.3f} |")
        for thr in (0.02, 0.04):
            c = S[(S.p_ens - 0.5).abs() > thr]
            if len(c):
                acc = ((c.p_ens > 0.5) == (c.up == 1)).mean()
                L.append(f"| ens, confident only (\\|p-0.5\\|>{thr}) | {acc:.1%} on {len(c)} calls | | |")
        n = len(S)
        L += ["", f"_With {n} scored predictions the 95% noise band around 50% is ±{1.96 * 0.5 / np.sqrt(n) * 100:.1f} pts "
                  f"(and predictions within the same hour are correlated, so the real band is wider). The backtest "
                  f"edge is ~+3 pts, so it needs thousands of live predictions to confirm._", "",
              "### By hour", "", "| decision candle | coins | up-rate | ens accuracy | mean return |", "|---|---|---|---|---|"]
        for t, g in S.groupby("t"):
            L.append(f"| {t[:16]} | {len(g)} | {g.up.mean():.0%} | {((g.p_ens > 0.5) == (g.up == 1)).mean():.0%} | "
                     f"{g.ret.mean() * 1e4:+.0f} bp |")
    if len(P):
        last = P[P.t == P.t.max()].sort_values("p_ens", ascending=False)
        L += ["", f"## Latest predictions (candle {last.t.iloc[0][:16]} UTC): P(up over next 4h)", "",
              " · ".join(f"{r.symbol} {r.p_ens:.3f}" for r in last.itertuples())]
    out = ROOT / "reports" / f"crypto_live_{date.today()}.md"
    out.write_text("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\nreport -> {out}")


if __name__ == "__main__":
    main()
