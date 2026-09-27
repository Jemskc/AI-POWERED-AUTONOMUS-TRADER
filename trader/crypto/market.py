"""Keyless live market data from Binance's public data API (data-api.binance.vision)."""
import json
import time
from urllib.parse import quote

import pandas as pd
import requests

BASE = "https://data-api.binance.vision/api/v3"
_S = requests.Session()


def _get(path, params=None, tries=5):
    for i in range(tries):
        try:
            r = _S.get(f"{BASE}/{path}", params=params, timeout=20)
            if r.status_code == 200:
                return r.json()
            if r.status_code in (418, 429):
                time.sleep(int(r.headers.get("Retry-After", 10)))
                continue
            r.raise_for_status()
        except requests.RequestException:
            if i == tries - 1:
                raise
            time.sleep(2 ** i)
    raise RuntimeError(f"GET {path} failed")


def klines(sym: str, interval="1h", limit=1000, start_ms=None, closed_only=True) -> pd.DataFrame:
    params = {"symbol": f"{sym}USDT", "interval": interval, "limit": limit}
    if start_ms is not None:
        params["startTime"] = int(start_ms)
    rows = _get("klines", params)
    df = pd.DataFrame(rows, columns=["open_time", "open", "high", "low", "close", "volume", "close_time",
                                     "quote_volume", "trades", "taker_buy_base", "taker_buy_quote", "ignore"])
    if df.empty:
        return df
    if closed_only:
        df = df[df["close_time"] < time.time() * 1000]
    for c in ("open", "high", "low", "close", "volume", "quote_volume", "trades", "taker_buy_base"):
        df[c] = df[c].astype(float)
    df["time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["symbol"] = sym
    return df[["symbol", "time", "open", "high", "low", "close", "volume", "quote_volume", "trades", "taker_buy_base"]]


def history_since(sym: str, start: pd.Timestamp) -> pd.DataFrame:
    """All closed 1h candles from `start` to now (paginated)."""
    out, ms = [], int(start.timestamp() * 1000)
    while True:
        df = klines(sym, start_ms=ms)
        if df.empty:
            break
        out.append(df)
        if len(df) < 1000:
            break
        ms = int(df["time"].iloc[-1].timestamp() * 1000) + 3_600_000
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def recent(symbols, hours=1000) -> pd.DataFrame:
    return pd.concat([klines(s, limit=hours) for s in symbols], ignore_index=True)


def tickers(symbols) -> dict:
    """symbol -> live {last, bid, ask, chg24, high24, low24, qvol24}."""
    syms = json.dumps([f"{s}USDT" for s in symbols], separators=(",", ":"))
    rows = _get(f"ticker/24hr?symbols={quote(syms)}")
    return {r["symbol"][:-4]: dict(last=float(r["lastPrice"]), bid=float(r["bidPrice"]), ask=float(r["askPrice"]),
                                   chg24=float(r["priceChangePercent"]) / 100, high24=float(r["highPrice"]),
                                   low24=float(r["lowPrice"]), qvol24=float(r["quoteVolume"])) for r in rows}
