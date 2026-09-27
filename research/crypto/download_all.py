"""Survivorship-free data: EVERY USDT spot pair in Binance's archive (including delisted coins) + funding rates
for every USDT perpetual. Adds to the 250-coin file rather than re-downloading it."""
import re
import sys
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from download import COLS, get, urls  # noqa: E402
from download_futures import get_csv, months, p_funding  # noqa: E402
from download_universe import STABLE  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DC = ROOT / "data" / "crypto"
S3 = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
NS = {"s": "http://s3.amazonaws.com/doc/2006-03-01/"}


def list_prefixes(prefix):
    out, marker = [], ""
    while True:
        r = requests.get(S3, params={"delimiter": "/", "prefix": prefix, "marker": marker}, timeout=60)
        root = ET.fromstring(r.content)
        out += [p.find("s:Prefix", NS).text.split("/")[-2] for p in root.findall("s:CommonPrefixes", NS)]
        if root.find("s:IsTruncated", NS).text != "true":
            return out
        marker = root.find("s:NextMarker", NS).text


def keep(base):
    return (re.fullmatch(r"[A-Z0-9]{2,12}", base) and base not in STABLE
            and not re.search(r"(?:UP|DOWN|BULL|BEAR)$", base))


if __name__ == "__main__":
    from datetime import date
    spot = [s[:-4] for s in list_prefixes("data/spot/monthly/klines/") if s.endswith("USDT") and keep(s[:-4])]
    perp = [s[:-4] for s in list_prefixes("data/futures/um/monthly/fundingRate/") if s.endswith("USDT") and keep(s[:-4])]
    have = pd.read_parquet(DC / "klines_1h_wide_universe.parquet", columns=["symbol"]).symbol.unique()
    todo = sorted(set(spot) - set(have))
    print(f"archive: {len(spot)} USDT spot coins ever listed, {len(perp)} USDT perpetuals; "
          f"already have {len(have)}, downloading {len(todo)} more (incl. delisted)", flush=True)
    jobs = [(s, u) for s in todo for u in urls(s)]
    with ThreadPoolExecutor(32) as ex:
        res = list(ex.map(lambda j: (j[0], get(j[1])), jobs))
    frames = [pd.read_parquet(DC / "klines_1h_wide_universe.parquet")]
    for s, df in res:
        if df is None or df.empty:
            continue
        df = df.astype({c: "float64" for c in COLS if c not in ("open_time", "close_time", "ignore")})
        tt = df["open_time"].astype("int64")
        df["time"] = pd.to_datetime(tt.where(tt < 10**14, tt // 1000), unit="ms", utc=True)
        df["symbol"] = s
        frames.append(df[["symbol", "time", "open", "high", "low", "close", "volume", "quote_volume", "trades", "taker_buy_base"]])
    allk = pd.concat(frames).drop_duplicates(["symbol", "time"]).sort_values(["symbol", "time"])
    allk.to_parquet(DC / "klines_1h_all.parquet")
    last = allk.groupby("symbol").time.max()
    print(f"klines: {len(allk):,} rows, {allk.symbol.nunique()} coins; stopped trading before 2026-09: "
          f"{(last < pd.Timestamp('2026-09-01', tz='UTC')).sum()} (delisted)", flush=True)
    fjobs = [(s, f"https://data.binance.vision/data/futures/um/monthly/fundingRate/{s}USDT/{s}USDT-fundingRate-{m:%Y-%m}.zip")
             for s in perp for m in months(date(2020, 1, 1))]
    with ThreadPoolExecutor(32) as ex:
        fr = list(ex.map(lambda j: p_funding(j[0], get_csv(j[1])), fjobs))
    f = pd.concat([x for x in fr if x is not None], ignore_index=True).dropna()
    f.to_parquet(DC / "funding_all.parquet")
    print(f"funding: {len(f):,} rows for {f.symbol.nunique()} perpetuals", flush=True)
