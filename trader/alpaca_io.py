"""All Alpaca I/O: assets, real market data, account, positions, orders.

The trading client is hard-wired to the PAPER endpoint (virtual money, real market prices).
"""
import os
import re
import time
from datetime import datetime, timedelta, timezone

import pandas as pd
from alpaca.data.enums import Adjustment, DataFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import AssetClass, AssetStatus, OrderSide, TimeInForce
from alpaca.trading.requests import GetAssetsRequest, GetCalendarRequest, MarketOrderRequest

EXCHANGES = {"NYSE", "NASDAQ", "ARCA", "AMEX", "BATS"}
# Alpaca has no ETF flag; this name heuristic keeps funds/ETNs/leveraged products out of the stock sleeve
FUND_RE = re.compile(r"\bETF\b|\bETN\b|\bFund\b|\bTrust\b|iShares|ProShares|SPDR|Direxion|Vanguard"
                     r"|WisdomTree|Invesco|Global X|VanEck|Leveraged|Ultra|\b[23]X\b|Bull\b|Bear\b",
                     re.IGNORECASE)


def _keys():
    k, s = os.environ.get("ALPACA_API_KEY"), os.environ.get("ALPACA_SECRET_KEY")
    if not k or not s:
        raise SystemExit("Set ALPACA_API_KEY and ALPACA_SECRET_KEY in ai-trader/.env (see .env.example)")
    return k, s


def data_client() -> StockHistoricalDataClient:
    return StockHistoricalDataClient(*_keys())


def trading_client() -> TradingClient:
    return TradingClient(*_keys(), paper=True)


def list_equities(tc: TradingClient) -> pd.DataFrame:
    assets = tc.get_all_assets(GetAssetsRequest(status=AssetStatus.ACTIVE, asset_class=AssetClass.US_EQUITY))
    df = pd.DataFrame([dict(symbol=a.symbol, name=a.name or "",
                            exchange=getattr(a.exchange, "value", str(a.exchange)),
                            tradable=a.tradable, shortable=a.shortable,
                            easy_to_borrow=a.easy_to_borrow) for a in assets])
    df = df[df.tradable & df.exchange.isin(EXCHANGES)]
    df = df[~df.symbol.str.contains(r"[^A-Z]")]  # drop share classes, warrants, units (BRK.B, XYZ.WS ...)
    df["is_fund"] = (df.exchange == "ARCA") | df.name.str.contains(FUND_RE)
    return df.reset_index(drop=True)


def fetch_daily_bars(dc: StockHistoricalDataClient, symbols, start, feed="sip", batch=200) -> pd.DataFrame:
    """Split/dividend-adjusted daily bars. `date` is the US/Eastern session date."""
    end = datetime.now(timezone.utc) - timedelta(minutes=20)  # free plan: no SIP data from the last 15 min
    start = pd.Timestamp(start).tz_localize("UTC").to_pydatetime()
    out = []
    for i in range(0, len(symbols), batch):
        req = StockBarsRequest(symbol_or_symbols=list(symbols[i:i + batch]), timeframe=TimeFrame.Day,
                               start=start, end=end, adjustment=Adjustment.ALL, feed=DataFeed(feed))
        for attempt in range(6):
            try:
                df = dc.get_stock_bars(req).df
                break
            except Exception as e:  # rate limits / transient network errors
                err = e
                time.sleep(2 ** attempt)
        else:
            raise err
        if len(df):
            out.append(df.reset_index())
        print(f"  bars: {min(i + batch, len(symbols))}/{len(symbols)} symbols", flush=True)
    df = pd.concat(out, ignore_index=True)
    df["date"] = (pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert("America/New_York")
                  .dt.normalize().dt.tz_localize(None))
    df = df.drop_duplicates(["symbol", "date"], keep="last")
    return df[["symbol", "date", "open", "high", "low", "close", "volume", "vwap", "trade_count"]]


def trading_days(tc: TradingClient, start, end) -> list:
    cal = tc.get_calendar(GetCalendarRequest(start=start, end=end))
    return [pd.Timestamp(c.date) for c in cal]


def positions(tc: TradingClient) -> dict:
    """symbol -> signed share quantity (negative = short)."""
    out = {}
    for p in tc.get_all_positions():
        q = abs(float(p.qty))
        out[p.symbol] = -q if str(p.side).lower().endswith("short") else q
    return out


def submit_market(tc: TradingClient, symbol: str, qty: int, at_open: bool, client_id: str):
    return tc.submit_order(MarketOrderRequest(
        symbol=symbol, qty=abs(int(qty)), side=OrderSide.BUY if qty > 0 else OrderSide.SELL,
        time_in_force=TimeInForce.OPG if at_open else TimeInForce.DAY, client_order_id=client_id))
