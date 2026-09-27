"""12 months of real Binance 1-minute klines for the 20 coins (free archive, no key)."""
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from download import COLS, SYMBOLS, get  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "crypto" / "klines_1m.parquet"
BASE = "https://data.binance.vision/data/spot"


def urls(sym, start=date(2025, 9, 1)):
    today, m = date.today(), start
    first = today.replace(day=1)
    while m < first:
        yield f"{BASE}/monthly/klines/{sym}USDT/1m/{sym}USDT-1m-{m:%Y-%m}.zip"
        m = (m + timedelta(days=32)).replace(day=1)
    d = first
    while d < today:
        yield f"{BASE}/daily/klines/{sym}USDT/1m/{sym}USDT-1m-{d:%Y-%m-%d}.zip"
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
        df["time"] = pd.to_datetime(t.where(t < 10**14, t // 1000), unit="ms", utc=True)
        df["symbol"] = s
        frames.append(df[["symbol", "time", "open", "high", "low", "close", "volume", "quote_volume", "trades", "taker_buy_base"]])
    df = pd.concat(frames).drop_duplicates(["symbol", "time"]).sort_values(["symbol", "time"])
    df.to_parquet(OUT)
    print(df.groupby("symbol")["time"].agg(["min", "max", "count"]).to_string())
    print(f"{len(df):,} rows -> {OUT}")
