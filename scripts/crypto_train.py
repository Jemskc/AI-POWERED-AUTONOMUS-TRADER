"""Top up the Binance history to the current hour (keyless) and train the live models.

Models: XGBoost + LightGBM (the two best in the walk-forward shoot-out), predicting
P(return over the next H hours > 0). Number of trees chosen on the most recent 15% of
history, then refit on everything.
"""
import _path  # noqa: F401

import argparse
import json
from datetime import date

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb

from trader.config import ROOT
from trader.crypto import market
from trader.crypto.features import SYMBOLS, features, forward_return, matrix, standardize, wide

HIST = ROOT / "data" / "crypto" / "klines_1h.parquet"

LGB = dict(objective="binary", learning_rate=0.03, num_leaves=31, min_data_in_leaf=2000, feature_fraction=0.7,
           bagging_fraction=0.7, bagging_freq=1, lambda_l2=10, num_threads=32, verbose=-1, seed=0)
XGB = dict(learning_rate=0.03, max_depth=5, subsample=0.7, colsample_bytree=0.7, min_child_weight=200,
           reg_lambda=10, tree_method="hist", n_jobs=32, random_state=0, eval_metric="logloss")


def top_up() -> pd.DataFrame:
    df = pd.read_parquet(HIST)
    new = []
    for s in SYMBOLS:
        last = df.loc[df.symbol == s, "time"].max()
        add = market.history_since(s, last + pd.Timedelta(hours=1))
        if len(add):
            new.append(add)
    if new:
        df = pd.concat([df, *new]).drop_duplicates(["symbol", "time"], keep="last").sort_values(["symbol", "time"])
        df.to_parquet(HIST)
    print(f"history: {len(df):,} rows through {df.time.max()} UTC (+{sum(len(n) for n in new):,} new)", flush=True)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--H", type=int, default=4)
    a = ap.parse_args()
    df = top_up()
    W = wide(df)
    F = features(W)
    names = list(F)
    X = matrix(F, names)
    ret = forward_return(W, a.H).to_numpy().reshape(-1)
    T, S = len(W["close"]), len(SYMBOLS)
    tidx = np.repeat(np.arange(T), S)
    ok = ~np.isnan(X).any(axis=1) & np.isfinite(ret)
    y = (ret > 0).astype(np.float32)
    rows = np.flatnonzero(ok)
    cut = int(np.quantile(tidx[rows], 0.85))
    fit, val = rows[tidx[rows] < cut - a.H - 1], rows[tidx[rows] >= cut]
    mu, sd = np.nanmean(X[rows], 0), np.nanstd(X[rows], 0) + 1e-6
    Z = standardize(X, mu, sd)

    m = lgb.train(LGB, lgb.Dataset(Z[fit], y[fit]), 2000, valid_sets=[lgb.Dataset(Z[val], y[val])],
                  callbacks=[lgb.early_stopping(100, verbose=False)])
    n_lgb = max(50, int(m.best_iteration * 1.15))
    xm = xgb.XGBClassifier(n_estimators=2000, early_stopping_rounds=100, **XGB)
    xm.fit(Z[fit], y[fit], eval_set=[(Z[val], y[val])], verbose=False)
    n_xgb = max(50, int(xm.best_iteration * 1.15))
    print(f"validation picked trees: lgbm={m.best_iteration} xgb={xm.best_iteration}; refit on all {len(rows):,} rows",
          flush=True)

    m = lgb.train(LGB, lgb.Dataset(Z[rows], y[rows]), n_lgb)
    xm = xgb.XGBClassifier(n_estimators=n_xgb, **XGB).fit(Z[rows], y[rows], verbose=False)

    out = ROOT / "models" / "crypto"
    out.mkdir(parents=True, exist_ok=True)
    tag = f"H{a.H}_{date.today()}"
    m.save_model(str(out / f"lgbm_{tag}.txt"))
    xm.save_model(str(out / f"xgb_{tag}.json"))
    meta = dict(H=a.H, names=names, mu=mu.tolist(), sd=sd.tolist(), trained_on=str(date.today()),
                data_end=str(W["close"].index[-1]), rows=int(len(rows)), n_lgb=n_lgb, n_xgb=n_xgb,
                lgbm=f"lgbm_{tag}.txt", xgb=f"xgb_{tag}.json")
    (out / f"meta_{tag}.json").write_text(json.dumps(meta))
    (out / f"latest_H{a.H}.json").write_text(json.dumps(meta))
    print(f"saved models/crypto/*_{tag} (data through {meta['data_end']})")


if __name__ == "__main__":
    main()
