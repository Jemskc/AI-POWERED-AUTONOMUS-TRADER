"""Crypto hourly features: the SAME code for training and live prediction.

Identical to research/crypto/shootout.py (the walk-forward experiment), so live
predictions come from exactly the pipeline that was tested.
"""
import numpy as np
import pandas as pd

SYMBOLS = ["BTC", "ETH", "BNB", "XRP", "ADA", "SOL", "DOGE", "DOT", "LTC", "LINK",
           "TRX", "AVAX", "BCH", "ATOM", "ETC", "XLM", "NEAR", "UNI", "FIL", "AAVE"]
FIELDS = ("open", "high", "low", "close", "volume", "quote_volume", "trades", "taker_buy_base")


def wide(df: pd.DataFrame) -> dict:
    """long klines (symbol, time, fields...) -> {field: time x symbol} on a gap-free hourly grid."""
    W = {k: df.pivot(index="time", columns="symbol", values=k).sort_index() for k in FIELDS}
    full = pd.date_range(W["close"].index[0], W["close"].index[-1], freq="h")
    return {k: v.reindex(index=full, columns=SYMBOLS) for k, v in W.items()}


def _roll(x, n):
    return x.rolling(n, min_periods=max(2, int(n * 0.8)))


def features(W: dict) -> dict:
    o, h, l, c, qv, nt = W["open"], W["high"], W["low"], W["close"], W["quote_volume"], W["trades"]
    tr = W["taker_buy_base"] / W["volume"].replace(0, np.nan)
    syms, idx = list(c.columns), c.index
    bc = lambda s: pd.DataFrame(np.repeat(np.asarray(s, float)[:, None], len(syms), 1), index=idx, columns=syms)

    lr = np.log(c).diff()
    vol24 = lr.rolling(24, min_periods=20).std()
    vol168 = lr.rolling(168, min_periods=120).std()
    F = {}
    for n in (1, 2, 4, 8, 12, 24, 48, 72, 168, 336):
        F[f"r{n}"] = np.log(c / c.shift(n)) / (vol168 * np.sqrt(n))
    F["lvol24"], F["lvol168"] = np.log(vol24), np.log(vol168)
    F["volratio"] = np.log(vol24 / vol168)
    F["hl"] = np.log(h / l) / vol168
    F["hl24"] = _roll(np.log(h / l), 24).mean() / vol168
    lqv = np.log(qv.replace(0, np.nan))
    F["qv_z"] = (lqv - _roll(lqv, 168).mean()) / _roll(lqv, 168).std()
    F["qv24_z"] = (_roll(lqv, 24).mean() - _roll(lqv, 720).mean()) / _roll(lqv, 720).std()
    lnt = np.log(nt.replace(0, np.nan))
    F["trades_z"] = (lnt - _roll(lnt, 168).mean()) / _roll(lnt, 168).std()
    F["taker1"] = tr - 0.5
    F["taker24"] = _roll(tr, 24).mean() - 0.5
    F["taker_z"] = (_roll(tr, 24).mean() - _roll(tr, 720).mean()) / _roll(tr, 720).std()
    F["ma168"] = (c / _roll(c, 168).mean() - 1) / (vol168 * np.sqrt(168))
    F["hi720"] = np.log(c / _roll(c, 720).max())
    F["lo720"] = np.log(c / _roll(c, 720).min())
    for n in (1, 4, 24, 168):
        F[f"btc_r{n}"] = bc(F[f"r{n}"]["BTC"])
    F["btc_lvol24"] = bc(F["lvol24"]["BTC"])
    F["mkt_r24"] = bc(F["r24"].mean(axis=1))
    F["mkt_r168"] = bc(F["r168"].mean(axis=1))
    F["rel_r24"] = F["r24"] - F["btc_r24"]
    for k in ("r24", "r168", "qv_z", "taker24"):
        F[f"xs_{k}"] = F[k].rank(axis=1, pct=True) - 0.5
    F["hsin"], F["hcos"] = bc(np.sin(2 * np.pi * idx.hour / 24)), bc(np.cos(2 * np.pi * idx.hour / 24))
    F["dow"] = bc(idx.dayofweek / 6 - 0.5)
    return F


def matrix(F: dict, names: list) -> np.ndarray:
    """-> (T*S, n_features) float32, time-major (row = t*S + s); non-finite -> NaN."""
    X = np.stack([F[k].to_numpy(np.float32) for k in names], axis=-1).reshape(-1, len(names))
    X[~np.isfinite(X)] = np.nan
    return X


def forward_return(W: dict, H: int) -> pd.DataFrame:
    """log return open(t+1) -> open(t+1+H): what a decision at the close of hour t earns."""
    o = W["open"]
    return np.log(o.shift(-1 - H) / o.shift(-1))


def standardize(X, mu, sd):
    return np.nan_to_num(np.clip((X - mu) / sd, -5, 5)).astype(np.float32)
