import numpy as np
import pandas as pd

from trader.features import next_open_returns


def run(W: pd.DataFrame, o: pd.DataFrame, cost_bps: float, borrow_bps_annual: float) -> pd.DataFrame:
    """W[t] = weights decided at close t, held open(t+1) -> open(t+2). Returns daily P&L frame."""
    W = W.reindex(columns=o.columns, fill_value=0.0).reindex(o.index).fillna(0.0)
    R = next_open_returns(o).reindex_like(W).fillna(0.0)
    gross = (W * R).sum(axis=1)
    turnover = W.diff().abs().sum(axis=1)
    turnover.iloc[0] = W.iloc[0].abs().sum()
    cost = turnover * cost_bps / 1e4 + W.clip(upper=0).abs().sum(axis=1) * borrow_bps_annual / 1e4 / 252
    df = pd.DataFrame({"gross": gross, "cost": cost, "net": gross - cost, "turnover": turnover})
    return df.iloc[:-2]  # last two rows have no realized return yet


def metrics(net: pd.Series) -> dict:
    net = net.dropna()
    eq = (1 + net).cumprod()
    ann_ret, ann_vol = net.mean() * 252, net.std() * np.sqrt(252)
    dd = (eq / eq.cummax() - 1).min()
    return {"ann_return": ann_ret, "ann_vol": ann_vol, "sharpe": ann_ret / ann_vol if ann_vol else np.nan,
            "max_drawdown": dd, "calmar": ann_ret / abs(dd) if dd else np.nan,
            "total_return": eq.iloc[-1] - 1, "days": len(net)}


def yearly(net: pd.Series) -> pd.Series:
    return net.groupby(net.index.year).apply(lambda x: (1 + x).prod() - 1)
