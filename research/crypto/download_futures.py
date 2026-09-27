"""Free Binance futures data (funding, premium index, open interest + long/short ratios) + Fear & Greed index."""
import io
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from download import SYMBOLS  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "crypto"
B = "https://data.binance.vision/data/futures/um"
S = requests.Session()


def get_csv(url, **kw):
    for _ in range(4):
        try:
            r = S.get(url, timeout=60)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            with zipfile.ZipFile(io.BytesIO(r.content)) as z, z.open(z.namelist()[0]) as f:
                return pd.read_csv(f, **kw)
        except requests.RequestException:
            continue
    return None


def months(start):
    m, first = start, date.today().replace(day=1)
    while m < first:
        yield m
        m = (m + timedelta(days=32)).replace(day=1)


def days(start):
    d = start
    while d < date.today():
        yield d
        d += timedelta(days=1)


def run(jobs, parse, name):
    print(f"{name}: {len(jobs)} files", flush=True)
    with ThreadPoolExecutor(24) as ex:
        res = list(ex.map(lambda j: parse(j[0], get_csv(j[1])), jobs))
    df = pd.concat([r for r in res if r is not None], ignore_index=True)
    df.to_parquet(OUT / f"{name}.parquet")
    print(f"  -> {len(df):,} rows, {df.symbol.nunique()} symbols", flush=True)


def p_funding(sym, df):
    if df is None:
        return None
    if "calc_time" not in df.columns:  # some old files have no header
        df.columns = ["calc_time", "funding_interval_hours", "last_funding_rate"][:len(df.columns)]
    return pd.DataFrame({"symbol": sym, "time": pd.to_datetime(pd.to_numeric(df.calc_time), unit="ms", utc=True),
                         "funding": pd.to_numeric(df.last_funding_rate, errors="coerce")})


def p_premium(sym, df):
    if df is None:
        return None
    df = df.iloc[:, :5]
    df.columns = ["open_time", "open", "high", "low", "close"]
    df = df[pd.to_numeric(df.open_time, errors="coerce").notna()]
    t = df.open_time.astype("int64")
    return pd.DataFrame({"symbol": sym, "time": pd.to_datetime(t.where(t < 10**14, t // 1000), unit="ms", utc=True),
                         "premium": pd.to_numeric(df.close, errors="coerce")})


def p_metrics(sym, df):
    if df is None or "create_time" not in df.columns:
        return None
    return pd.DataFrame({"symbol": sym, "time": pd.to_datetime(df.create_time, utc=True),
                         "oi": df.sum_open_interest, "oi_value": df.sum_open_interest_value,
                         "top_ls_count": df.count_toptrader_long_short_ratio,
                         "top_ls_pos": df.sum_toptrader_long_short_ratio,
                         "ls_count": df.count_long_short_ratio, "taker_ls": df.sum_taker_long_short_vol_ratio})


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    fng = requests.get("https://api.alternative.me/fng/?limit=0&format=json", timeout=60).json()["data"]
    pd.DataFrame({"time": pd.to_datetime([int(x["timestamp"]) for x in fng], unit="s", utc=True),
                  "fng": [int(x["value"]) for x in fng]}).sort_values("time").to_parquet(OUT / "fear_greed.parquet")
    print(f"fear&greed: {len(fng)} days", flush=True)
    run([(s, f"{B}/monthly/fundingRate/{s}USDT/{s}USDT-fundingRate-{m:%Y-%m}.zip") for s in SYMBOLS
         for m in months(date(2020, 1, 1))], p_funding, "funding")
    run([(s, f"{B}/monthly/premiumIndexKlines/{s}USDT/1h/{s}USDT-1h-{m:%Y-%m}.zip") for s in SYMBOLS
         for m in months(date(2020, 1, 1))], p_premium, "premium_1h")
    run([(s, f"{B}/daily/metrics/{s}USDT/{s}USDT-metrics-{d:%Y-%m-%d}.zip") for s in SYMBOLS
         for d in days(date(2021, 12, 1))], p_metrics, "futures_metrics")
