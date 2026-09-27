"""100-coin, low-turnover, market-neutral coin-ranking strategy on real Binance data.

Universe: at every hour, the 100 coins with the highest trailing-30-day median trading volume (point in time,
from ~250 candidates) that have >= 60 days of history.
Model: LightGBM regression on the cross-sectional rank of the next-24h return (price/volume features, same
feature code as the live engine). Walk-forward: 6-month test blocks from 2021-07, expanding window, purged.
Portfolio (PRIMARY, fixed before testing): daily rebalance; 3-day EWMA of predictions; long when a coin's
rank enters [70%, 98%] and hold while >= 55%; short when it enters [2%, 30%] and hold while <= 45%;
inverse-volatility weights, 50% long / 50% short; 7bp per side on turnover.
"""
import math
import sys
import time
from datetime import date
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from trader.crypto.features import FIELDS, features  # noqa: E402

DC = ROOT / "data" / "crypto"
H = 24
START = pd.Timestamp("2020-01-01", tz="UTC")
PRIMARY = dict(entry=0.70, hold=0.55, extreme=0.02, ewm_days=3)


def build(n_universe=100, only=None):
    df = pd.read_parquet(DC / "klines_1h_wide_universe.parquet")
    if only is not None:
        df = df[df.symbol.isin(only)]
    W = {k: df.pivot(index="time", columns="symbol", values=k).sort_index() for k in FIELDS}
    full = pd.date_range(W["close"].index[0], W["close"].index[-1], freq="h")
    W = {k: v.reindex(full) for k, v in W.items()}
    c, qv = W["close"], W["quote_volume"]
    hist = c.notna().cumsum()
    liq = qv.rolling(720, min_periods=500).median().where(hist >= 60 * 24)
    U = (liq.rank(axis=1, ascending=False) <= n_universe) & c.notna()
    F = features(W)
    fwd = np.log(W["open"].shift(-1 - H) / W["open"].shift(-1))
    vol = np.log(c).diff().rolling(168, min_periods=120).std()
    return W, F, U, fwd, vol


def walk_forward(F, U, fwd):
    idx, cols = U.index, U.columns
    T, S = U.shape
    names = list(F)
    X = np.empty((T * S, len(names)), np.float32)
    for j, k in enumerate(names):
        X[:, j] = F[k].to_numpy(np.float32).reshape(-1)
    X[~np.isfinite(X)] = np.nan
    u = U.to_numpy().reshape(-1)
    ok = u & ~np.isnan(X).any(axis=1)
    y = (fwd.where(pd.DataFrame(ok.reshape(T, S), index=idx, columns=cols)).rank(axis=1, pct=True) - 0.5).to_numpy().reshape(-1)
    tidx = np.repeat(np.arange(T), S)
    use = ok & np.isfinite(y)
    pred = np.full(T * S, np.nan)
    starts = pd.date_range(pd.Timestamp("2021-07-01", tz="UTC"), idx[-1], freq="6MS")
    ics = []
    for fi, st in enumerate(starts):
        en = starts[fi + 1] if fi + 1 < len(starts) else idx[-1] + pd.Timedelta(hours=1)
        ts, te = idx.searchsorted(st), idx.searchsorted(en)
        tr = np.flatnonzero(use & (tidx < ts - H - 1) & (tidx % 4 == 0))  # every 4th hour: 24h labels overlap anyway
        cut = tidx[tr[int(len(tr) * 0.85)]]
        fit, val = tr[tidx[tr] < cut - H - 1], tr[tidx[tr] >= cut]
        m = lgb.train(dict(objective="regression", learning_rate=0.03, num_leaves=63, min_data_in_leaf=3000,
                           feature_fraction=0.7, bagging_fraction=0.7, bagging_freq=1, lambda_l2=10,
                           num_threads=36, verbose=-1, seed=0),
                      lgb.Dataset(X[fit], y[fit]), 2000, valid_sets=[lgb.Dataset(X[val], y[val])],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
        te_rows = np.flatnonzero(ok & (tidx >= ts) & (tidx < te))
        pred[te_rows] = m.predict(X[te_rows], num_iteration=m.best_iteration)
        pr = pd.DataFrame(pred.reshape(T, S)[ts:te]).rank(axis=1)
        yr = pd.DataFrame(y.reshape(T, S)[ts:te]).rank(axis=1)
        ic = pr.corrwith(yr, axis=1).mean()
        ics.append(ic)
        print(f"  fold {st:%Y-%m}: train {len(fit):,} test {len(te_rows):,} coins/hour {ok.reshape(T, S)[ts:te].sum(1).mean():.0f} "
              f"trees {m.best_iteration} Rank-IC {ic:+.4f}", flush=True)
    return pd.DataFrame(pred.reshape(T, S), index=idx, columns=cols), ics


def simulate(pred, U, fwd, vol, entry, hold, extreme, ewm_days, cost_bps, rng=None):
    """Daily rebalance at 00:00 UTC (decision on the 23:00 candle)."""
    days = pred.index[(pred.index.hour == 23) & pred.notna().any(axis=1).to_numpy()]
    P = pred.loc[days]
    if rng is not None:
        P = P.apply(lambda r: pd.Series(np.where(r.notna(), rng.permutation(r.to_numpy()), np.nan), index=r.index), axis=1)
    P = P.ewm(span=ewm_days, ignore_na=True).mean().where(U.loc[days] & P.notna())
    R = np.expm1(fwd.loc[days]).to_numpy()
    V = vol.loc[days].to_numpy()
    rank = P.rank(axis=1, pct=True).to_numpy()
    S = P.shape[1]
    side = np.zeros(S)
    w_prev = np.zeros(S)
    out, turns, nl, ns = [], [], [], []
    for i in range(len(days)):
        rk, r, v = rank[i], R[i], V[i]
        live = np.isfinite(rk) & np.isfinite(v) & (v > 0)
        new = np.zeros(S)
        new[live & (side == 1) & (rk >= hold)] = 1
        new[live & (side == -1) & (rk <= 1 - hold)] = -1
        new[live & (new == 0) & (rk >= entry) & (rk <= 1 - extreme)] = 1
        new[live & (new == 0) & (rk <= 1 - entry) & (rk >= extreme)] = -1
        side = new
        x = np.where(live, 1 / np.where(live, v, 1), 0)
        L, Sh = (side == 1), (side == -1)
        w = np.zeros(S)
        if L.any() and Sh.any():
            w[L] = 0.5 * x[L] / x[L].sum()
            w[Sh] = -0.5 * x[Sh] / x[Sh].sum()
        turn = np.abs(w - w_prev).sum()
        rr = np.nan_to_num(r)
        g = float((w * rr).sum())
        out.append(g - turn * cost_bps / 1e4)
        turns.append(turn)
        nl.append(L.sum())
        ns.append(Sh.sum())
        w_prev = w * (1 + rr) / (1 + g) if 1 + g > 0 else w
    r = np.array(out)
    e = np.cumprod(1 + r)
    s = pd.Series(r, index=days)
    return dict(sharpe=r.mean() / r.std() * math.sqrt(365), total=e[-1] - 1, cagr=e[-1] ** (365 / len(r)) - 1,
                max_dd=(e / np.maximum.accumulate(e) - 1).min(), turnover_day=np.mean(turns), n_long=np.mean(nl),
                n_short=np.mean(ns), yearly=s.groupby(s.index.year).apply(lambda x: (1 + x).prod() - 1), rets=s)


def row(name, a, b=None):
    return (f"| {name} | {a['sharpe']:.2f} | {a['total']:+.1%} | {a['cagr']:+.1%} | {a['max_dd']:.1%} | "
            f"{a['turnover_day']:.2f}x | {a['n_long']:.0f}/{a['n_short']:.0f} | "
            + (f"{b['sharpe']:.2f} | {b['total']:+.1%} |" if b else "| |"))


def main():
    t0 = time.time()
    L = [f"# 100-coin low-turnover market-neutral coin ranking: {date.today()}", "",
         "Real Binance hourly data. " + (__doc__.split("\n", 1)[1].strip().replace("\n", " ")), ""]
    results = {}
    for label, n, only in (("100 coins", 100, None), ("20 coins (original set)", 20,
                           ["BTC", "ETH", "BNB", "XRP", "ADA", "SOL", "DOGE", "DOT", "LTC", "LINK", "TRX", "AVAX",
                            "BCH", "ATOM", "ETC", "XLM", "NEAR", "UNI", "FIL", "AAVE"])):
        print(f"== {label}", flush=True)
        W, F, U, fwd, vol = build(n, only)
        pred, ics = walk_forward(F, U, fwd)
        results[label] = (pred, U, fwd, vol, ics)
        print(f"   built+trained in {(time.time() - t0) / 60:.1f} min", flush=True)
    L += ["## Signal quality (Rank-IC per 6-month test block)", "",
          "| universe | " + " | ".join(f"{i + 1}" for i in range(len(results['100 coins'][4]))) + " | mean |",
          "|---|" + "---|" * (len(results['100 coins'][4]) + 1)]
    for label, (_, _, _, _, ics) in results.items():
        L.append(f"| {label} | " + " | ".join(f"{x:+.3f}" for x in ics) + f" | **{np.mean(ics):+.3f}** |")
    L += ["", "## Portfolio results (daily rebalance, market-neutral)", "",
          "| strategy | Sharpe 7bp | total 7bp | CAGR 7bp | max DD | turnover/day | coins long/short | Sharpe 3bp | total 3bp |",
          "|---|---|---|---|---|---|---|---|---|"]
    pred, U, fwd, vol, _ = results["100 coins"]
    prim = simulate(pred, U, fwd, vol, cost_bps=7, **PRIMARY)
    L.append(row("**100 coins PRIMARY** (hysteresis, skip extremes, 3d smoothing)", prim, simulate(pred, U, fwd, vol, cost_bps=3, **PRIMARY)))
    variants = {"no hysteresis (enter=exit 70%)": dict(PRIMARY, hold=0.70),
                "no extreme exclusion": dict(PRIMARY, extreme=0.0),
                "no smoothing": dict(PRIMARY, ewm_days=1),
                "wider (enter top/bottom 20%)": dict(PRIMARY, entry=0.80, hold=0.60)}
    for k, v in variants.items():
        L.append(row(f"100 coins, {k}", simulate(pred, U, fwd, vol, cost_bps=7, **v), simulate(pred, U, fwd, vol, cost_bps=3, **v)))
    ctrl = [simulate(pred, U, fwd, vol, cost_bps=7, rng=np.random.default_rng(k), **PRIMARY) for k in range(10)]
    cs = [c["sharpe"] for c in ctrl]
    L.append(f"| 100 coins, RANDOM scores (10 runs) | {np.mean(cs):.2f} (best {np.max(cs):.2f}) | {np.mean([c['total'] for c in ctrl]):+.1%} | | "
             f"{np.mean([c['max_dd'] for c in ctrl]):.1%} | {np.mean([c['turnover_day'] for c in ctrl]):.2f}x | | | |")
    p20, U20, f20, v20, _ = results["20 coins (original set)"]
    L.append(row("20 coins, same PRIMARY rules", simulate(p20, U20, f20, v20, cost_bps=7, **PRIMARY),
                 simulate(p20, U20, f20, v20, cost_bps=3, **PRIMARY)))
    mk = np.expm1(fwd.loc[prim["rets"].index]).where(U.loc[prim["rets"].index]).mean(axis=1).fillna(0)
    e = np.cumprod(1 + mk.to_numpy())
    L.append(f"| buy & hold (equal-weight universe, daily) | {mk.mean() / mk.std() * math.sqrt(365):.2f} | {e[-1] - 1:+.1%} | | "
             f"{(e / np.maximum.accumulate(e) - 1).min():.1%} | | | | |")
    beta = np.cov(prim["rets"], mk)[0, 1] / np.var(mk)
    L += ["", f"PRIMARY beta to the market: {beta:+.2f} · real Sharpe beats {np.mean(np.array(cs) < prim['sharpe']):.0%} of random runs", "",
          "## PRIMARY calendar-year returns (7bp)", "", prim["yearly"].to_frame("return").to_markdown(floatfmt=".1%"), "",
          "Caveats: universe drawn from coins still listed today (delisted coins missing -> survivorship bias); "
          "short side assumes perpetual futures and ignores funding payments; delisting-day moves are not modelled."]
    prim["rets"].to_csv(DC / "xs100_primary_daily_returns.csv")
    out = ROOT / "reports" / f"crypto_xs100_{date.today()}.md"
    out.write_text("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"report -> {out}  ({(time.time() - t0) / 60:.0f} min)")


if __name__ == "__main__":
    main()
