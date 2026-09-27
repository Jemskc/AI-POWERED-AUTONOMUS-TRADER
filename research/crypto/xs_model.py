"""Coin-SELECTION model: predict each coin's return RELATIVE to the other coins (cross-sectional rank),
the way stat-arb funds do, then test it in market-neutral portfolios.

Target: rank of the forward H-hour return across the 20 coins at each hour, centred on 0 (-0.5 .. +0.5).
Model: LightGBM + XGBoost regression average. Same walk-forward, rows and features as shootout_futures.py.
Metric: Rank-IC = Spearman correlation between predicted and realized ranks, per hour, averaged.
"""
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from shootout_futures import START, extra_features  # noqa: E402
from trader.crypto.features import SYMBOLS, features, forward_return, wide  # noqa: E402

DC = ROOT / "data" / "crypto"


def main():
    t0 = time.time()
    k1 = pd.read_parquet(DC / "klines_1h.parquet")
    W = wide(k1[k1.time >= START - pd.Timedelta(days=60)])
    idx = W["close"].index
    F = features(W)
    E = extra_features(idx)
    sets = {"base": list(F), "p_oi": list(F) + list(E["oi"]),
            "p_all": list(F) + list(E["funding"]) + list(E["oi"]) + list(E["fng"])}
    allF = {**F, **E["funding"], **E["oi"], **E["fng"]}
    names = sets["p_all"]
    T, S = len(idx), len(SYMBOLS)
    X = np.empty((T * S, len(names)), np.float32)
    for j, k in enumerate(names):
        X[:, j] = allF[k].to_numpy(np.float32).reshape(-1)
    X[~np.isfinite(X)] = np.nan
    col = {k: j for j, k in enumerate(names)}
    tidx = np.repeat(np.arange(T), S)
    ok = ~np.isnan(X[:, :len(F)]).any(axis=1) & (np.repeat(idx, S) >= START)
    starts = pd.date_range(pd.Timestamp("2022-07-01", tz="UTC"), idx[-1], freq="6MS")
    for H in (4, 24):
        fwd = forward_return(W, H)
        rank = (fwd.where(ok.reshape(T, S)).rank(axis=1, pct=True) - 0.5)
        n_ok = rank.notna().sum(axis=1).to_numpy()
        y = rank.to_numpy().reshape(-1)
        y[np.repeat(n_ok < 10, S)] = np.nan  # need a real cross-section
        use = ok & np.isfinite(y)
        P = {s: np.full(T * S, np.nan) for s in sets}
        for fi, st in enumerate(starts):
            en = starts[fi + 1] if fi + 1 < len(starts) else idx[-1] + pd.Timedelta(hours=1)
            ts, te = idx.searchsorted(st), idx.searchsorted(en)
            tr = np.flatnonzero(use & (tidx < ts - H - 1))
            cut = tidx[tr[int(len(tr) * 0.85)]]
            fit, val = tr[tidx[tr] < cut - H - 1], tr[tidx[tr] >= cut]
            te_rows = np.flatnonzero(ok & (tidx >= ts) & (tidx < te))
            msg = []
            for s, fs in sets.items():
                c = [col[k] for k in fs]
                m = lgb.train(dict(objective="regression", learning_rate=0.03, num_leaves=31, min_data_in_leaf=2000,
                                   feature_fraction=0.7, bagging_fraction=0.7, bagging_freq=1, lambda_l2=10,
                                   num_threads=36, verbose=-1, seed=0),
                              lgb.Dataset(X[fit][:, c], y[fit]), 2000, valid_sets=[lgb.Dataset(X[val][:, c], y[val])],
                              callbacks=[lgb.early_stopping(100, verbose=False)])
                xm = xgb.XGBRegressor(n_estimators=2000, learning_rate=0.03, max_depth=5, subsample=0.7,
                                      colsample_bytree=0.7, min_child_weight=200, reg_lambda=10, tree_method="hist",
                                      n_jobs=36, random_state=0, early_stopping_rounds=100)
                xm.fit(X[fit][:, c], y[fit], eval_set=[(X[val][:, c], y[val])], verbose=False)
                P[s][te_rows] = (m.predict(X[te_rows][:, c], num_iteration=m.best_iteration) + xm.predict(X[te_rows][:, c])) / 2
                pr = pd.DataFrame(P[s].reshape(T, S)[ts:te]).rank(axis=1)
                yr = pd.DataFrame(y.reshape(T, S)[ts:te]).rank(axis=1)
                ic = pr.corrwith(yr, axis=1).mean()
                msg.append(f"{s}={ic:+.4f}")
            print(f"H={H} fold {st:%Y-%m}: Rank-IC " + " ".join(msg), flush=True)
        # stored as 'probability-like' scores centred on 0.5 so portfolio.py can reuse them unchanged
        np.savez_compressed(DC / f"preds_xs_H{H}.npz", **{k: v + 0.5 for k, v in P.items()},
                            ret=fwd.to_numpy().reshape(-1), tidx=tidx, times=idx.asi8, symbols=np.array(SYMBOLS))
    print(f"done in {(time.time() - t0) / 60:.0f} min")


if __name__ == "__main__":
    main()
