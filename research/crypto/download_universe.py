"""Hourly klines (2020 -> now) for a broad set of ~250 Binance USDT coins; the point-in-time top-100
universe is chosen later from trailing volume, so today's ranking does not leak into the past."""
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from download import COLS, get, urls  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "crypto" / "klines_1h_wide_universe.parquet"
STABLE = {"USDC", "FDUSD", "TUSD", "BUSD", "DAI", "USDP", "USDD", "EUR", "TRY", "BRL", "GBP", "AUD", "PAXG", "USD1",
          "RLUSD", "XUSD", "USDE", "AEUR", "EURI", "BFUSD", "UST", "USTC", "WBTC", "WBETH", "BETH", "PYUSD", "USDS",
          "FRAX", "LUSD", "GUSD", "SUSD", "XAUT", "EURR", "USDF"}

if __name__ == "__main__":
    t = pd.DataFrame(requests.get("https://data-api.binance.vision/api/v3/ticker/24hr", timeout=60).json())
    t = t[t.symbol.str.endswith("USDT")].assign(base=lambda d: d.symbol.str[:-4], qv=lambda d: d.quoteVolume.astype(float))
    t = t[t.base.str.fullmatch(r"[A-Z0-9]{2,10}") & ~t.base.isin(STABLE)
          & ~t.base.str.contains(r"(?:UP|DOWN|BULL|BEAR)$") & (t.qv > 0)]
    cands = t.sort_values("qv", ascending=False).base.head(250).tolist()
    for must in ("BTC", "ETH"):
        assert must in cands
    jobs = [(s, u) for s in cands for u in urls(s)]
    print(f"{len(cands)} candidate coins, {len(jobs)} monthly/daily files", flush=True)
    with ThreadPoolExecutor(32) as ex:
        res = list(ex.map(lambda j: (j[0], get(j[1])), jobs))
    frames = []
    for s, df in res:
        if df is None or df.empty:
            continue
        df = df.astype({c: "float64" for c in COLS if c not in ("open_time", "close_time", "ignore")})
        tt = df["open_time"].astype("int64")
        df["time"] = pd.to_datetime(tt.where(tt < 10**14, tt // 1000), unit="ms", utc=True)
        df["symbol"] = s
        frames.append(df[["symbol", "time", "open", "high", "low", "close", "volume", "quote_volume", "trades", "taker_buy_base"]])
    df = pd.concat(frames).drop_duplicates(["symbol", "time"]).sort_values(["symbol", "time"])
    df.to_parquet(OUT)
    first = df.groupby("symbol").time.min()
    print(f"{len(df):,} rows, {df.symbol.nunique()} coins with data; listed by 2021-01: {(first < '2021-01-01').sum()}, "
          f"by 2023-01: {(first < '2023-01-01').sum()} -> {OUT}")
