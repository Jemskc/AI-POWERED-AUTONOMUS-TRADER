"""Strict checks of the 24h 'directional' lead: random-score control, market-hedged returns, yearly returns."""
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import portfolio as pf  # noqa: E402

pf.KIND = sys.argv[1] if len(sys.argv) > 1 else "xs"
H = 24
P, R, vol, times = pf.load(H)
ppy = 8760 / H


def st(r):
    e = np.cumprod(1 + r)
    return f"Sharpe {r.mean() / r.std() * math.sqrt(ppy):5.2f} | total {e[-1] - 1:+8.1%} | maxDD {(e / np.maximum.accumulate(e) - 1).min():6.1%}"


print(f"[{pf.KIND}] 24h coin-ranking model, directional (long rank>0.54, short <0.46, else hold), 7bp\n")
for fs in P:
    a = pf.simulate(P[fs], R, vol, H, "dir", 7.0)
    r, idx = a["rets"], a["idx"]
    mk = np.array([np.nanmean(np.where(np.isfinite(R[t]), R[t], np.nan)) for t in idx])
    beta = np.cov(r, mk)[0, 1] / np.var(mk)
    cs = [(lambda c: c.mean() / c.std() * math.sqrt(ppy))(pf.simulate(P[fs], R, vol, H, "dir", 7.0,
                                                                     rng=np.random.default_rng(k))["rets"]) for k in range(20)]
    real = r.mean() / r.std() * math.sqrt(ppy)
    s = pd.Series(r, index=times[idx])
    print(f"{fs:6s} strategy      : {st(r)} | beta {beta:+.2f} | period {times[idx[0]]:%Y-%m} .. {times[idx[-1]]:%Y-%m}")
    print(f"{'':6s} market-hedged : {st(r - beta * mk)}")
    print(f"{'':6s} RANDOM scores : Sharpe mean {np.mean(cs):.2f}, best of 20 {np.max(cs):.2f} | real beats {np.mean(np.array(cs) < real):.0%}")
    print(f"{'':6s} by year       : " + "  ".join(f"{y}:{v:+.0%}" for y, v in s.groupby(s.index.year).apply(lambda x: (1 + x).prod() - 1).items()))
    print()
first = pf.simulate(next(iter(P.values())), R, vol, H, "dir", 7.0)["idx"]
mk = np.array([np.nanmean(np.where(np.isfinite(R[t]), R[t], np.nan)) for t in first])
print(f"buy&hold 20 coins (same period): {st(mk)}")
s = pd.Series(mk, index=times[first])
print("   by year: " + "  ".join(f"{y}:{v:+.0%}" for y, v in s.groupby(s.index.year).apply(lambda x: (1 + x).prod() - 1).items()))
