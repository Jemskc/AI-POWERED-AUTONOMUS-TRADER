"""Download real split/dividend-adjusted daily history for every tradable US stock + the ETF sleeve."""
import _path  # noqa: F401

from trader.config import settings
from trader.pipeline import HISTORY, download

if __name__ == "__main__":
    S = settings()
    bars, assets = download(S["history_start"], S["feed"], S["etf_trend"]["symbols"])
    bars.to_parquet(HISTORY)
    print(f"saved {len(bars):,} bars, {bars.symbol.nunique():,} symbols, "
          f"{bars.date.min().date()} .. {bars.date.max().date()} -> {HISTORY}")
