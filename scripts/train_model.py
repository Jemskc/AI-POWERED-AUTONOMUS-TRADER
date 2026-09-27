"""Train the production stock model on the most recent `train_years` of real history."""
import _path  # noqa: F401

from datetime import date

import pandas as pd

from trader.config import path, settings
from trader.model import fit, save
from trader.pipeline import ASSETS, HISTORY, download, matrices, stock_block

if __name__ == "__main__":
    import sys
    S = settings()
    cs = S["stock_ml"]
    if "--refresh" in sys.argv:
        bars, assets = download(S["history_start"], S["feed"], S["etf_trend"]["symbols"])
        bars.to_parquet(HISTORY)
    bars, assets = pd.read_parquet(HISTORY), pd.read_parquet(ASSETS)
    panel, _, feats = stock_block(matrices(bars), assets, cs, S["etf_trend"]["symbols"])
    d = panel.index.get_level_values(0)
    tr = panel[d >= d.max() - pd.DateOffset(years=cs["train_years"])]
    model = fit(tr, feats, "y", cs["lgbm"], cs["num_boost_round"])
    out = path("models", f"stock_lgbm_{date.today()}.txt")
    save(model, feats, {"trained_on": str(date.today()), "data_end": str(d.max().date()),
                        "rows": int(tr["y"].notna().sum()), "horizon": cs["horizon"]}, out)
    print(f"model -> {out}  (data through {d.max().date()}, {len(tr):,} rows)")
