"""Real Binance spot 1h klines from data.binance.vision (official public archive, no key needed)."""
import io
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "crypto" / "klines_1h.parquet"
SYMBOLS = ["BTC", "ETH", "BNB", "XRP", "ADA", "SOL", "DOGE", "DOT", "LTC", "LINK",
           "TRX", "AVAX", "BCH", "ATOM", "ETC", "XLM", "NEAR", "UNI", "FIL", "AAVE"]
COLS = ["open_time", "open", "high", "low", "close", "volume", "close_time", "quote_volume",
        "trades", "taker_buy_base", "taker_buy_quote", "ignore"]
BASE = "https://data.binance.vision/data/spot"


def get(url):
    for _ in range(4):
        try:
            r = requests.get(url, timeout=60)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            with zipfile.ZipFile(io.BytesIO(r.content)) as z, z.open(z.namelist()[0]) as f:
                df = pd.read_csv(f, header=None, names=COLS)
            return df[pd.to_numeric(df["open_time"], errors="coerce").notna()]
        except requests.RequestException:
            continue
    raise RuntimeError(f"failed: {url}")


def urls(sym, start=date(2020, 1, 1)):
    today = date.today()
    first_of_month = today.replace(day=1)
    m = start
    while m < first_of_month:  # complete months
        yield f"{BASE}/monthly/klines/{sym}USDT/1h/{sym}USDT-1h-{m:%Y-%m}.zip"
        m = (m + timedelta(days=32)).replace(day=1)
    d = first_of_month
    while d < today:  # current month, day by day, through yesterday
        yield f"{BASE}/daily/klines/{sym}USDT/1h/{sym}USDT-1h-{d:%Y-%m-%d}.zip"
        d += timedelta(days=1)


if __name__ == "__main__":
    jobs = [(s, u) for s in SYMBOLS for u in urls(s)]
    print(f"{len(jobs)} files", flush=True)
    with ThreadPoolExecutor(16) as ex:
        res = list(ex.map(lambda j: (j[0], get(j[1])), jobs))
    frames = []
    for s, df in res:
        if df is None:
            continue
        df = df.astype({c: "float64" for c in COLS if c not in ("open_time", "close_time", "ignore")})
        t = df["open_time"].astype("int64")
        df["time"] = pd.to_datetime(t.where(t < 10**14, t // 1000), unit="ms", utc=True)  # 2025+ files are in µs
        df["symbol"] = s
        frames.append(df[["symbol", "time", "open", "high", "low", "close", "volume", "quote_volume",
                          "trades", "taker_buy_base"]])
    df = pd.concat(frames).drop_duplicates(["symbol", "time"]).sort_values(["symbol", "time"])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT)
    print(df.groupby("symbol")["time"].agg(["min", "max", "count"]).to_string())
    print(f"{len(df):,} rows -> {OUT}")
    sys.exit(0)
