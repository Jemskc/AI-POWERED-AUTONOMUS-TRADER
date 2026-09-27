"""Glue shared by research, training and the live daily run, so all three compute identical features."""
import pandas as pd

from trader import alpaca_io as io
from trader.config import path
from trader.features import forward_label, stock_features, to_panel, universe_mask, wide

HISTORY = path("data", "bars_daily.parquet")
ASSETS = path("data", "assets.parquet")


def download(start, feed, symbols_extra=()) -> tuple:
    """-> (bars, assets)"""
    tc, dc = io.trading_client(), io.data_client()
    assets = io.list_equities(tc)
    assets.to_parquet(ASSETS)
    syms = sorted(set(assets.symbol) | set(symbols_extra))
    print(f"downloading {len(syms)} symbols from {start} (feed={feed})")
    try:
        return io.fetch_daily_bars(dc, syms, start, feed=feed), assets
    except Exception as e:
        if feed != "sip":
            raise
        print(f"  SIP feed failed ({e}); falling back to IEX")
        return io.fetch_daily_bars(dc, syms, start, feed="iex"), assets


def matrices(bars: pd.DataFrame) -> dict:
    return {k: wide(bars, k) for k in ("open", "high", "low", "close", "volume")}


def stock_block(M: dict, assets: pd.DataFrame, cfg: dict, etf_syms, dates=None, with_label=True):
    """-> (panel with ranked features [+ label 'y'], universe mask, feature names)."""
    excl = set(assets.loc[assets.is_fund, "symbol"]) | set(etf_syms)
    cols = [s for s in M["close"].columns if s not in excl]
    o, h, l, c, v = (M[k][cols] for k in ("open", "high", "low", "close", "volume"))
    mask = universe_mask(c, v, cfg["universe_size"], cfg["min_price"])
    ever = mask.any()
    o, h, l, c, v, mask = (x.loc[:, ever] for x in (o, h, l, c, v, mask))
    F = stock_features(o, h, l, c, v)
    panel = to_panel(F, mask, dates)
    if with_label:
        panel["y"] = forward_label(o, mask, cfg["horizon"]).stack(future_stack=True).reindex(panel.index)
    return panel, mask, list(F)
