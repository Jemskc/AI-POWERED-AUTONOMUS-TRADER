"""Does futures-market data make the model better? Walk-forward ablation on real Binance data.

Feature sets (all point-in-time: only data published by the close of hour t):
  base      price/volume features used by the live engine today
  +funding  funding rate + futures premium index
  +oi       open interest + top-trader / global long-short ratios + futures taker ratio
  +all      base + funding + oi + Fear & Greed index
Model: XGBoost + LightGBM average (the live ensemble). 6-month test blocks from 2022-07, expanding training
window from 2022-01 (when the open-interest data starts), purged at the boundary.
"""
import argparse
import math
import sys
import time
from datetime import date
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from shootout import trading  # noqa: E402
from trader.crypto.features import SYMBOLS, features, forward_return, wide  # noqa: E402

DC = ROOT / "data" / "crypto"
START = pd.Timestamp("2022-01-01", tz="UTC")


def hourly_last(df, col, idx):
    """5-min snapshots -> last snapshot inside each hour [t, t+1h) -> known by the close of candle t."""
    d = df.assign(h=df.time.dt.floor("h")).groupby(["h", "symbol"])[col].last().unstack()
    return d.reindex(index=idx, columns=SYMBOLS)


def extra_features(idx):
    E = {"funding": {}, "oi": {}, "fng": {}}
    roll = lambda x, n: x.rolling(n, min_periods=max(2, int(n * 0.8)))
    # funding: last settled rate at or before the decision time (close of candle t = t + 1h)
    fu = pd.read_parquet(DC / "funding.parquet").dropna()
    fu = fu.assign(h=(fu.time - pd.Timedelta(hours=1)).dt.ceil("h")).groupby(["h", "symbol"]).funding.last().unstack()
    fu = fu.reindex(columns=SYMBOLS).reindex(fu.index.union(idx)).ffill().reindex(idx) * 1e4  # bp
    E["funding"]["fund"] = fu
    E["funding"]["fund_z"] = (fu - roll(fu, 720).mean()) / roll(fu, 720).std()
    E["funding"]["fund_chg72"] = fu - fu.shift(72)
    E["funding"]["xs_fund"] = fu.rank(axis=1, pct=True) - 0.5
    pr = pd.read_parquet(DC / "premium_1h.parquet").drop_duplicates(["symbol", "time"])
    pr = pr.pivot(index="time", columns="symbol", values="premium").reindex(index=idx, columns=SYMBOLS) * 1e4
    E["funding"]["prem"] = pr
    E["funding"]["prem24"] = roll(pr, 24).mean()
    E["funding"]["prem_z"] = (pr - roll(pr, 168).mean()) / roll(pr, 168).std()
    E["funding"]["xs_prem"] = pr.rank(axis=1, pct=True) - 0.5
    # open interest + positioning (5-min snapshots)
    m = pd.read_parquet(DC / "futures_metrics.parquet")
    oi = np.log(hourly_last(m, "oi", idx))
    for n in (1, 4, 24, 168):
        E["oi"][f"oi_chg{n}"] = oi - oi.shift(n)
    E["oi"]["oi_z"] = (oi - roll(oi, 720).mean()) / roll(oi, 720).std()
    for col, name in (("top_ls_pos", "top_ls"), ("ls_count", "crowd_ls"), ("taker_ls", "fut_taker")):
        x = np.log(hourly_last(m, col, idx))
        E["oi"][name] = x
        E["oi"][f"{name}_chg24"] = x - x.shift(24)
        E["oi"][f"{name}_z"] = (x - roll(x, 168).mean()) / roll(x, 168).std()
    E["oi"]["smart_vs_crowd"] = E["oi"]["top_ls"] - E["oi"]["crowd_ls"]
    for k in ("oi_chg24", "crowd_ls", "smart_vs_crowd"):
        E["oi"][f"xs_{k}"] = E["oi"][k].rank(axis=1, pct=True) - 0.5
    # Fear & Greed: daily value, used from the NEXT day on (no same-day lookahead)
    fg = pd.read_parquet(DC / "fear_greed.parquet").set_index("time").fng
    fg.index = fg.index + pd.Timedelta(days=1)
    fg = fg.reindex(fg.index.union(idx)).ffill().reindex(idx)
    bc = lambda s: pd.DataFrame(np.repeat(s.to_numpy(float)[:, None], len(SYMBOLS), 1), index=idx, columns=SYMBOLS)
    E["fng"]["fng"] = bc(fg / 100)
    E["fng"]["fng_chg7"] = bc((fg - fg.shift(168)) / 100)
    return E


def fit_predict(Xtr, ytr, Xva, yva, Xte):
    m = lgb.train(dict(objective="binary", learning_rate=0.03, num_leaves=31, min_data_in_leaf=2000,
                       feature_fraction=0.7, bagging_fraction=0.7, bagging_freq=1, lambda_l2=10, num_threads=36,
                       verbose=-1, seed=0), lgb.Dataset(Xtr, ytr), 2000, valid_sets=[lgb.Dataset(Xva, yva)],
                  callbacks=[lgb.early_stopping(100, verbose=False)])
    xm = xgb.XGBClassifier(n_estimators=2000, learning_rate=0.03, max_depth=5, subsample=0.7, colsample_bytree=0.7,
                           min_child_weight=200, reg_lambda=10, tree_method="hist", n_jobs=36, random_state=0,
                           early_stopping_rounds=100, eval_metric="logloss")
    xm.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    gain = pd.Series(m.feature_importance("gain"))
    return (m.predict(Xte, num_iteration=m.best_iteration) + xm.predict_proba(Xte)[:, 1]) / 2, gain


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizons", default="4,24")
    a = ap.parse_args()
    t0 = time.time()
    k1 = pd.read_parquet(DC / "klines_1h.parquet")
    W = wide(k1[k1.time >= START - pd.Timedelta(days=60)])
    idx = W["close"].index
    F = features(W)
    E = extra_features(idx)
    sets = {"base": list(F), "+funding": list(F) + list(E["funding"]), "+oi": list(F) + list(E["oi"]),
            "+all": list(F) + list(E["funding"]) + list(E["oi"]) + list(E["fng"])}
    allF = {**F, **E["funding"], **E["oi"], **E["fng"]}
    names = sets["+all"]
    T, S = len(idx), len(SYMBOLS)
    X = np.empty((T * S, len(names)), np.float32)
    for j, k in enumerate(names):
        X[:, j] = allF[k].to_numpy(np.float32).reshape(-1)
    X[~np.isfinite(X)] = np.nan
    col = {k: j for j, k in enumerate(names)}
    tidx = np.repeat(np.arange(T), S)
    # rows need complete PRICE features; new features may be missing (e.g. top-trader ratios before late 2022):
    # tree models handle missing values natively, and every feature set uses exactly the same rows
    ok = ~np.isnan(X[:, :len(F)]).any(axis=1) & (np.repeat(idx, S) >= START)
    print(f"{T * S:,} rows, {ok.mean():.0%} usable (complete price features); {len(names)} features "
          f"({len(F)} base + {len(names) - len(F)} new) · {idx[0]} .. {idx[-1]} · {time.time() - t0:.0f}s", flush=True)
    starts = pd.date_range(pd.Timestamp("2022-07-01", tz="UTC"), idx[-1], freq="6MS")
    L = [f"# Does futures-market data improve the model? Walk-forward ablation: {date.today()}", "",
         f"Real Binance data, {S} coins, test {starts[0]:%Y-%m} .. {idx[-1]:%Y-%m-%d}; all feature sets trained and "
         f"tested on the SAME rows. Model = XGBoost+LightGBM average. Trading: confident calls only "
         f"(|p-0.5|>0.04, else hold), every H hours.", ""]
    for H in [int(h) for h in a.horizons.split(",")]:
        ret = forward_return(W, H).to_numpy().reshape(-1)
        y = (ret > 0).astype(np.float32)
        use = ok & np.isfinite(ret)
        P = {s: np.full(len(y), np.nan) for s in sets}
        fold_auc = {s: [] for s in sets}
        gains = None
        for fi, st in enumerate(starts):
            en = starts[fi + 1] if fi + 1 < len(starts) else idx[-1] + pd.Timedelta(hours=1)
            ts, te = idx.searchsorted(st), idx.searchsorted(en)
            tr = np.flatnonzero(use & (tidx < ts - H - 1))
            cut = tidx[tr[int(len(tr) * 0.85)]]
            fit, val = tr[tidx[tr] < cut - H - 1], tr[tidx[tr] >= cut]
            te_rows = np.flatnonzero(use & (tidx >= ts) & (tidx < te))
            msg = []
            for s, fs in sets.items():
                c = [col[k] for k in fs]
                p, g = fit_predict(X[fit][:, c], y[fit], X[val][:, c], y[val], X[te_rows][:, c])
                P[s][te_rows] = p
                fold_auc[s].append(roc_auc_score(y[te_rows], p))
                msg.append(f"{s}={fold_auc[s][-1]:.4f}")
                if s == "+all":
                    g.index = fs
                    gains = g if gains is None else gains + g
            print(f"H={H} fold {st:%Y-%m}: train {len(fit):,} test {len(te_rows):,}  AUC " + " ".join(msg), flush=True)
        np.savez_compressed(DC / f"preds_futures_H{H}.npz", **{k.replace("+", "p_"): v for k, v in P.items()},
                            ret=ret, tidx=tidx, times=idx.asi8, symbols=np.array(SYMBOLS))
        ev = use & ~np.isnan(P["base"])
        D = dict(times=idx, S=S, ret=ret)
        L += [f"## Horizon {H}h", "", f"{ev.sum():,} out-of-sample predictions · actual up-rate {y[ev].mean():.1%}", "",
              "| features | accuracy | AUC | folds better than base | conf. accuracy (|p-.5|>.04) | conf. calls | Sharpe net 7bp | total net 7bp | Sharpe net 3bp | max DD |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for s in sets:
            p, yy = P[s][ev], y[ev]
            conf = np.abs(p - 0.5) > 0.04
            tr7, tr3 = trading(np.where(ev, P[s], np.nan), D, H, 7.0), trading(np.where(ev, P[s], np.nan), D, H, 3.0)
            better = sum(a_ > b_ for a_, b_ in zip(fold_auc[s], fold_auc["base"])) if s != "base" else "-"
            L.append(f"| {s} | {((p > .5) == (yy == 1)).mean():.2%} | {roc_auc_score(yy, p):.4f} | {better}/{len(starts)} | "
                     f"{((p[conf] > .5) == (yy[conf] == 1)).mean() if conf.any() else float('nan'):.2%} | {conf.sum():,} | "
                     f"{tr7['conf']['sharpe_net']:.2f} | {tr7['conf']['total_net']:+.1%} | {tr3['conf']['sharpe_net']:.2f} | "
                     f"{tr7['conf']['max_dd']:.1%} |")
        new = gains.drop(sets["base"]).sort_values(ascending=False)
        L += ["", f"Most useful NEW features (+all model, total gain): " + ", ".join(
            f"{k} ({v / gains.sum():.1%})" for k, v in new.head(10).items()),
            f"Share of total model gain from new features: {new.sum() / gains.sum():.1%}", ""]
    out = ROOT / "reports" / f"crypto_futures_features_{date.today()}.md"
    out.write_text("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"report -> {out}  ({(time.time() - t0) / 60:.0f} min)")


if __name__ == "__main__":
    main()
