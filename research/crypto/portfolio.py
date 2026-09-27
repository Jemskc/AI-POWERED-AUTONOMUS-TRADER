"""Turn the same out-of-sample predictions into positions in different ways, to cut drawdowns.

Portfolio constructions (rebalanced every H hours on real Binance prices, costs on turnover):
  dir        directional: +1/-1 per coin on confident calls (|p-0.5|>0.04, else hold)  <- what we had
  mn         market-neutral: score = p - average p across coins; long above-average, short below-average,
             sized by 1/volatility, 50% of capital long + 50% short (net 0)
  mn_risk    mn + half-step rebalancing (less turnover) + 20% annual volatility target + drawdown brake
             (-10% from peak -> half size, -20% -> quarter size)                        <- PRIMARY, fixed in advance
  top5_risk  like mn_risk but only the 5 highest and 5 lowest scored coins
Also: the PRIMARY rules driven by random scores (control), and equal-weight buy & hold.
"""
import math
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from trader.crypto.features import wide  # noqa: E402

DC = ROOT / "data" / "crypto"
PRIMARY = ("p_oi", "mn_risk")


KIND = sys.argv[1] if len(sys.argv) > 1 else "futures"  # "futures" (up/down model) or "xs" (coin-ranking model)


def load(H):
    d = np.load(DC / f"preds_{KIND}_H{H}.npz", allow_pickle=True)
    times = pd.DatetimeIndex(d["times"]).tz_localize("UTC")
    T, S = len(times), len(d["symbols"])
    k1 = pd.read_parquet(DC / "klines_1h.parquet")
    W = wide(k1[k1.time >= times[0]])
    c = W["close"].reindex(times)
    vol = np.log(c).diff().rolling(168, min_periods=120).std().to_numpy()
    R = np.expm1(d["ret"].reshape(T, S))
    P = {k: d[k].reshape(T, S) for k in d.files if k not in ("ret", "tidx", "times", "symbols")}
    return P, R, vol, times


def simulate(p_all, R, vol, H, mode, cost_bps, rng=None):
    T, S = R.shape
    rows = [t for t in range(0, T, H) if np.isfinite(p_all[t]).sum() >= 10 and np.isfinite(R[t]).sum() >= 10]
    ppy = 8760 / H
    lookback = max(10, int(30 * 24 / H))
    risk = mode.endswith("_risk")
    lam = 0.5 if risk else 1.0
    w_prev = np.zeros(S)
    sig = np.zeros(S)
    eq, peak, rets, turns, mkt, dates = 1.0, 1.0, [], [], [], []
    for t in rows:
        p = p_all[t].copy()
        r = R[t]
        ok = np.isfinite(p) & np.isfinite(r) & np.isfinite(vol[t]) & (vol[t] > 0)
        if rng is not None:
            p[ok] = rng.permutation(p[ok])
        tgt = np.zeros(S)
        if mode == "dir":
            sig = np.where(ok & (p - 0.5 > 0.04), 1.0, np.where(ok & (0.5 - p > 0.04), -1.0, np.where(ok, sig, 0.0)))
            tgt = sig / max(ok.sum(), 1)
        else:
            s = np.where(ok, p - np.nanmean(np.where(ok, p, np.nan)), 0.0)
            if mode.startswith("top5"):
                order = np.argsort(np.where(ok, s, np.nan))
                n = ok.sum()
                keep = np.zeros(S, bool)
                keep[order[:5]] = True
                keep[order[n - 5:n]] = True
                s = np.where(keep & ok, s, 0.0)
            x = np.where(ok, s / np.where(ok, vol[t], 1.0), 0.0)
            pos, neg = x.clip(min=0), x.clip(max=0)
            if pos.sum() > 0 and neg.sum() < 0:
                tgt = 0.5 * pos / pos.sum() - 0.5 * neg / neg.sum()
        w = w_prev + lam * (tgt - w_prev)
        lev = 1.0
        if risk:
            if len(rets) >= lookback:
                rv = np.std(rets[-lookback:]) * math.sqrt(ppy)
                lev = float(np.clip(0.20 / rv, 0.25, 2.0)) if rv > 0 else 1.0
            dd = eq / peak - 1
            lev *= 0.25 if dd < -0.20 else 0.5 if dd < -0.10 else 1.0
        w = w * lev
        turn = np.abs(w - w_prev).sum()
        rr = np.nan_to_num(r)
        gross = float((w * rr).sum())
        net = gross - turn * cost_bps / 1e4
        eq *= 1 + net
        peak = max(peak, eq)
        w_prev = w * (1 + rr) / (1 + gross) if 1 + gross > 0 else w
        rets.append(net)
        turns.append(turn)
        mkt.append(np.nanmean(np.where(np.isfinite(r), r, np.nan)))
        dates.append(t)
    rets, mkt = np.array(rets), np.array(mkt)
    e = np.cumprod(1 + rets)
    beta = np.cov(rets, mkt)[0, 1] / np.var(mkt)
    return dict(sharpe=rets.mean() / rets.std() * math.sqrt(ppy), total=e[-1] - 1,
                cagr=e[-1] ** (ppy / len(rets)) - 1, max_dd=(e / np.maximum.accumulate(e) - 1).min(),
                turnover_day=np.mean(turns) * 24 / H, beta=beta, rets=rets, idx=dates)


def main():
    L = [f"# Portfolio construction test, {KIND} model (real out-of-sample predictions): {date.today()}", "",
         "Same predictions as the futures-data experiment (test 2022-07 .. 2026-09). Primary configuration fixed "
         f"before testing: features `{PRIMARY[0]}`, construction `{PRIMARY[1]}`.", ""]
    for H in (4, 24):
        P, R, vol, times = load(H)
        L += [f"## Horizon {H}h (rebalance every {H}h)", "",
              "| features | construction | Sharpe 7bp | total 7bp | CAGR | max DD | beta to market | turnover/day | Sharpe 3bp | total 3bp |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        yearly = {}
        for fs in ("base", "p_oi", "p_all"):
            for mode in ("dir", "mn", "mn_risk", "top5_risk"):
                a, b = simulate(P[fs], R, vol, H, mode, 7.0), simulate(P[fs], R, vol, H, mode, 3.0)
                star = " **(PRIMARY)**" if (fs, mode) == PRIMARY else ""
                L.append(f"| {fs}{star} | {mode} | {a['sharpe']:.2f} | {a['total']:+.1%} | {a['cagr']:+.1%} | "
                         f"{a['max_dd']:.1%} | {a['beta']:+.2f} | {a['turnover_day']:.2f}x | {b['sharpe']:.2f} | {b['total']:+.1%} |")
                if mode == "mn_risk":
                    s = pd.Series(a["rets"], index=times[a["idx"]])
                    yearly[f"{fs} mn_risk"] = s.groupby(s.index.year).apply(lambda x: (1 + x).prod() - 1)
                print(f"H={H} {fs} {mode}: Sharpe {a['sharpe']:.2f} total {a['total']:+.1%} DD {a['max_dd']:.1%}", flush=True)
        ctrl = [simulate(P[PRIMARY[0]], R, vol, H, PRIMARY[1], 7.0, rng=np.random.default_rng(k)) for k in range(5)]
        L.append(f"| RANDOM scores (5 runs) | {PRIMARY[1]} | {np.mean([c['sharpe'] for c in ctrl]):.2f} | "
                 f"{np.mean([c['total'] for c in ctrl]):+.1%} | | {np.mean([c['max_dd'] for c in ctrl]):.1%} | | | | |")
        mk = np.nanmean(np.where(np.isfinite(R), R, np.nan)[::H], axis=1)
        mk = mk[np.isfinite(mk)]
        e = np.cumprod(1 + mk)
        L.append(f"| buy & hold 20 coins | long only | {mk.mean() / mk.std() * math.sqrt(8760 / H):.2f} | {e[-1] - 1:+.1%} | | "
                 f"{(e / np.maximum.accumulate(e) - 1).min():.1%} | +1.00 | | | |")
        L += ["", "Calendar-year returns of the risk-managed market-neutral version (7bp):", "",
              pd.DataFrame(yearly).to_markdown(floatfmt=".1%"), ""]
    out = ROOT / "reports" / f"crypto_portfolio_{KIND}_{date.today()}.md"
    out.write_text("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"report -> {out}")


if __name__ == "__main__":
    main()
