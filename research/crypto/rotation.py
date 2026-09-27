"""Backtest a long-only 'rotation' strategy on the real out-of-sample predictions (2022-01 .. 2026-09).

Every hour: rank the 20 coins by the model's P(up next 4h) (XGBoost+LightGBM average, the live ensemble).
  - SELL a held coin if the model turns bearish on it (p < exit_p) or it drops out of the top `keep` ranks
  - BUY the best-ranked coins that are in the top K with p > enter_p, until K coins are held
  - each held coin gets 1/K of equity; unused slots stay in cash (USDT). No shorting, no leverage.
Costs charged on every trade (turnover). Compared with buy & hold and with the SAME rules driven by
random scores (so we see whether the model adds anything beyond the trading mechanics).
"""
import itertools
import math
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PRIMARY = dict(K=5, keep=8, enter_p=0.52, exit_p=0.48, every=1)  # chosen before looking at results


def load():
    d = np.load(ROOT / "data" / "crypto" / "shootout_preds_H4.npz")
    df = pd.read_parquet(ROOT / "data" / "crypto" / "klines_1h.parquet")
    df = df[df.time <= pd.Timestamp("2026-09-22 23:00", tz="UTC")]
    o = df.pivot(index="time", columns="symbol", values="open").sort_index()  # alphabetical, as in the shoot-out
    o = o.reindex(pd.date_range(o.index[0], o.index[-1], freq="h"))
    T, S = o.shape
    assert T * S == len(d["xgboost"]), (T, S, len(d["xgboost"]))
    P = ((d["xgboost"] + d["lightgbm"]) / 2).reshape(T, S)
    r1 = (o.shift(-2) / o.shift(-1) - 1).to_numpy()  # return earned by a position decided at the close of t
    return P, r1, o.index, list(o.columns)


def simulate(P, r1, rows, K, keep, enter_p, exit_p, every, cost_bps, rng=None):
    S = P.shape[1]
    w = np.zeros(S)
    held = set()
    rets, turns, nheld = [], [], []
    for i, t in enumerate(rows):
        p = P[t].copy()
        if rng is not None:  # random-score control: same mechanics, no information
            p = np.where(np.isfinite(p), rng.permutation(np.where(np.isfinite(p), p, np.nan)), np.nan)
        target = w
        if i % every == 0:
            avail = np.isfinite(p)
            order = np.argsort(np.where(avail, -p, np.inf))
            rank = np.empty(S, int)
            rank[order] = np.arange(S)
            held = {s for s in held if avail[s] and p[s] >= exit_p and rank[s] < keep}
            for s in order[:K]:
                if len(held) >= K:
                    break
                if avail[s] and p[s] > enter_p:
                    held.add(s)
            target = np.zeros(S)
            for s in held:
                target[s] = 1 / K
            # don't churn tiny weight drifts of coins we keep holding
            keepw = (target > 0) & (w > 0) & (np.abs(target - w) < 0.25 / K)
            target[keepw] = w[keepw]
        turn = np.abs(target - w).sum()
        r = np.nan_to_num(r1[t])
        gross = float((target * r).sum())
        net = gross - turn * cost_bps / 1e4
        # drift weights with prices for the next hour
        grow = target * (1 + r)
        w = grow / (1 + gross) if 1 + gross > 0 else grow
        rets.append(net)
        turns.append(turn)
        nheld.append(len(held))
    return np.array(rets), np.array(turns), np.array(nheld)


def stats(r, turns=None, nheld=None, index=None):
    eq = np.cumprod(1 + r)
    yrs = len(r) / 8760
    out = dict(total=eq[-1] - 1, cagr=eq[-1] ** (1 / yrs) - 1, sharpe=r.mean() / r.std() * math.sqrt(8760) if r.std() else 0,
               max_dd=(eq / np.maximum.accumulate(eq) - 1).min())
    if turns is not None:
        out.update(turnover_per_day=turns.mean() * 24, avg_held=nheld.mean(), in_market=(nheld > 0).mean())
    if index is not None:
        s = pd.Series(r, index=index)
        out["yearly"] = s.groupby(s.index.year).apply(lambda x: (1 + x).prod() - 1)
    return out


def main():
    P, r1, times, syms = load()
    rows = np.flatnonzero(np.isfinite(P).any(axis=1) & np.isfinite(r1).any(axis=1))
    idx = times[rows]
    print(f"OOS hours: {len(rows):,}  ({idx[0]} .. {idx[-1]})", flush=True)

    res = {}
    for cost in (7.0, 3.0):
        r, tu, nh = simulate(P, r1, rows, cost_bps=cost, **PRIMARY)
        res[f"rotation (model) {cost:.0f}bp"] = stats(r, tu, nh, idx)
        rr = [simulate(P, r1, rows, cost_bps=cost, rng=np.random.default_rng(k), **PRIMARY) for k in range(3)]
        rnd = [stats(*x, index=idx) for x in rr]
        res[f"rotation (RANDOM scores) {cost:.0f}bp"] = {k: (np.mean([z[k] for z in rnd]) if k != "yearly" else
                                                            pd.concat([z[k] for z in rnd], axis=1).mean(axis=1)) for k in rnd[0]}
    ew = np.nanmean(np.where(np.isfinite(P[rows]), r1[rows], np.nan), axis=1)
    res["buy & hold, equal weight 20 coins"] = stats(np.nan_to_num(ew), index=idx)
    res["buy & hold BTC"] = stats(np.nan_to_num(r1[rows, syms.index("BTC")]), index=idx)

    L = [f"# Rotation strategy backtest (long-only, real out-of-sample predictions): {date.today()}", "",
         f"Out-of-sample {idx[0]:%Y-%m-%d} .. {idx[-1]:%Y-%m-%d} ({len(rows):,} hours). Model = XGBoost+LightGBM average "
         f"(the live ensemble). Primary rules fixed before testing: {PRIMARY}.", "",
         "| strategy | total | CAGR | Sharpe | max DD | turnover/day | avg coins held | time in market |",
         "|---|---|---|---|---|---|---|---|"]
    for k, s in res.items():
        L.append(f"| {k} | {s['total']:+.1%} | {s['cagr']:+.1%} | {s['sharpe']:.2f} | {s['max_dd']:.1%} | "
                 + (f"{s['turnover_per_day']:.2f}x | {s['avg_held']:.1f} | {s['in_market']:.0%} |" if "avg_held" in s else "| | |"))
    L += ["", "## Calendar-year returns", "", pd.DataFrame({k: v["yearly"] for k, v in res.items()}).to_markdown(floatfmt=".1%")]

    # sensitivity grid (7bp): shows whether the result depends on lucky parameter choice
    grid = []
    for K, keep_add, ep, xp, every in itertools.product((3, 5), (0, 3), (0.50, 0.52), (0.50, 0.48), (1, 4)):
        r, tu, nh = simulate(P, r1, rows, K=K, keep=K + keep_add, enter_p=ep, exit_p=xp, every=every, cost_bps=7.0)
        s = stats(r, tu, nh)
        grid.append(dict(K=K, keep=K + keep_add, enter_p=ep, exit_p=xp, every_h=every, total=s["total"],
                         sharpe=s["sharpe"], max_dd=s["max_dd"], turnover_day=s["turnover_per_day"]))
    G = pd.DataFrame(grid).sort_values("sharpe", ascending=False)
    L += ["", f"## Sensitivity: all {len(G)} rule variants at 7bp (sorted by Sharpe)", "",
          f"Variants with positive Sharpe: {(G.sharpe > 0).sum()}/{len(G)} · median Sharpe {G.sharpe.median():.2f} · "
          f"benchmark equal-weight Sharpe {res['buy & hold, equal weight 20 coins']['sharpe']:.2f}", "",
          G.to_markdown(index=False, floatfmt=".2f")]
    out = ROOT / "reports" / f"crypto_rotation_backtest_{date.today()}.md"
    out.write_text("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\nreport -> {out}")


if __name__ == "__main__":
    main()
