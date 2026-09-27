"""Feature, universe and label code, shared by research, training and live trading.

Convention: everything indexed by date t uses only data up to the CLOSE of t.
Trades decided at t execute at the OPEN of t+1.
"""
import numpy as np
import pandas as pd


def wide(bars: pd.DataFrame, col: str) -> pd.DataFrame:
    return bars.pivot(index="date", columns="symbol", values=col).sort_index()


def universe_mask(c: pd.DataFrame, v: pd.DataFrame, size: int, min_price: float,
                  min_history: int = 252) -> pd.DataFrame:
    """Point-in-time liquid universe: top `size` by 60d median dollar volume on each date."""
    dv = (c * v).rolling(60, min_periods=40).median()
    history = c.notna().astype(int).rolling(min_history, min_periods=1).sum()
    dv = dv.where((c >= min_price) & (history >= 0.95 * min_history))
    return dv.rank(axis=1, ascending=False) <= size


def stock_features(o, h, l, c, v) -> dict:
    lr = np.log(c).diff()
    vol20 = lr.rolling(20, min_periods=15).std()
    vol60 = lr.rolling(60, min_periods=40).std()
    ldv = np.log((c * v).replace(0, np.nan))
    mkt = lr.mean(axis=1)
    mkt_var = mkt.rolling(60).var()
    beta = (lr.mul(mkt, axis=0).rolling(60).mean()
            - lr.rolling(60).mean().mul(mkt.rolling(60).mean(), axis=0)).div(mkt_var, axis=0)
    resid = lr - beta.mul(mkt, axis=0)

    F = {f"ret_{n}": np.log(c / c.shift(n)) / (vol60 * np.sqrt(n)) for n in (1, 5, 21, 63, 126, 252)}
    F.update({
        "mom_12_1": np.log(c.shift(21) / c.shift(252)) / (vol60 * np.sqrt(231)),
        "resid_mom_21": resid.rolling(21).sum() / (vol60 * np.sqrt(21)),
        "resid_rev_5": resid.rolling(5).sum() / (vol60 * np.sqrt(5)),
        "vol20": vol20,
        "vol60": vol60,
        "vol_ratio": vol20 / vol60,
        "range_vol": np.sqrt((np.log(h / l) ** 2).rolling(20).mean() / (4 * np.log(2))),
        "gap": np.log(o / c.shift(1)) / vol60,
        "intraday": np.log(c / o) / vol60,
        "beta": beta,
        "dv_z": (ldv - ldv.rolling(60).mean()) / ldv.rolling(60).std(),
        "log_dv": ldv.rolling(20).median(),
        "amihud": (lr.abs() / np.exp(ldv)).rolling(20).mean(),
        "dist_ma50": c / c.rolling(50).mean() - 1,
        "dist_hi252": c / c.rolling(252, min_periods=200).max() - 1,
        "skew60": lr.rolling(60).skew(),
    })
    return F


def to_panel(feats: dict, mask: pd.DataFrame, dates=None) -> pd.DataFrame:
    """Cross-sectionally rank every feature inside the universe -> long panel (date, symbol)."""
    if dates is not None:
        mask = mask.loc[dates]
    s = mask.stack(future_stack=True)
    idx = s[s.fillna(False).astype(bool)].index
    out = pd.DataFrame(index=idx)
    for k, m in feats.items():
        m = m.loc[mask.index].where(mask)
        out[k] = (m.rank(axis=1, pct=True) - 0.5).stack(future_stack=True).reindex(idx)
    return out


def forward_label(o: pd.DataFrame, mask: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Rank of the open(t+1) -> open(t+1+h) return inside the universe, centred on 0."""
    fwd = np.log(o.shift(-1 - horizon) / o.shift(-1)).where(mask)
    return fwd.rank(axis=1, pct=True) - 0.5


def next_open_returns(o: pd.DataFrame) -> pd.DataFrame:
    """Return earned by a position decided at t: open(t+1) -> open(t+2)."""
    return o.shift(-2) / o.shift(-1) - 1
