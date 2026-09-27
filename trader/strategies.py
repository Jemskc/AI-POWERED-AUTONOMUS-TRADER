"""Turn forecasts into target weights (fraction of the sleeve's capital)."""
import numpy as np
import pandas as pd


def stock_ml_weights(pred: pd.Series, mask: pd.DataFrame, frac: float, gross: float,
                     halflife: float) -> pd.DataFrame:
    P = pred.unstack()
    m = mask.reindex(index=P.index, columns=P.columns).fillna(False).astype(bool)
    P = P.ewm(halflife=halflife, ignore_na=True).mean().where(m)
    r = P.rank(axis=1, pct=True)
    long, short = (r > 1 - frac).astype(float), (r <= frac).astype(float)
    W = long.div(long.sum(axis=1), axis=0) - short.div(short.sum(axis=1), axis=0)
    return (W * gross / 2).fillna(0.0)


def etf_trend_weights(c: pd.DataFrame, lookbacks, vol_target: float, max_w: float, max_gross: float,
                      last_only: bool = False) -> pd.DataFrame:
    lr = np.log(c).diff()
    dvol = lr.ewm(span=60, min_periods=40).std()
    sig = sum(np.clip(np.log(c / c.shift(n)) / (dvol * np.sqrt(n)), -2, 2) / 2 for n in lookbacks) / len(lookbacks)
    raw = sig / (dvol * np.sqrt(252))
    W = pd.DataFrame(0.0, index=c.index, columns=c.columns)
    rows = [len(c) - 1] if last_only else range(252, len(c))
    for i in rows:
        w = raw.iloc[i].dropna()
        if w.empty:
            continue
        cov = (lr.iloc[i - 125:i + 1][w.index].cov() * 252).fillna(0.0)
        pv = float(np.sqrt(w @ cov @ w))
        if not np.isfinite(pv) or pv <= 0:
            continue
        w = (w * vol_target / pv).clip(-max_w, max_w)
        g = w.abs().sum()
        if g > max_gross:
            w *= max_gross / g
        W.iloc[i, W.columns.get_indexer(w.index)] = w.values
    return W
