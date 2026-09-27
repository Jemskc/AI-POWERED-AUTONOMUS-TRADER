"""Walk-forward research on real history. Writes reports/research_YYYY-MM-DD.md.

Go/no-go gate before paper trading: stock-ML out-of-sample Rank-IC t-stat > 2,
net Sharpe > baselines, and combined net Sharpe > 0.8 with max drawdown < 20%.
"""
import _path  # noqa: F401

from datetime import date

import numpy as np
import pandas as pd

from trader import backtest as bt
from trader.config import path, settings
from trader.model import rank_ic, walk_forward
from trader.pipeline import ASSETS, HISTORY, matrices, stock_block
from trader.strategies import etf_trend_weights, stock_ml_weights


def fmt(m: dict) -> str:
    return (f"Sharpe {m['sharpe']:.2f} | ann ret {m['ann_return']:.1%} | ann vol {m['ann_vol']:.1%} | "
            f"maxDD {m['max_drawdown']:.1%} | total {m['total_return']:.1%} | {m['days']} days")


if __name__ == "__main__":
    S = settings()
    cs, ce, costs = S["stock_ml"], S["etf_trend"], S["costs"]
    bars, assets = pd.read_parquet(HISTORY), pd.read_parquet(ASSETS)
    M = matrices(bars)
    print("building stock features ...")
    panel, mask, feats = stock_block(M, assets, cs, ce["symbols"])
    print(f"panel: {len(panel):,} rows x {len(feats)} features")

    pred = walk_forward(panel, feats, "y", cs["lgbm"], cs["num_boost_round"], cs["train_years"],
                        cs["retrain_every_days"], cs["horizon"])
    ic = rank_ic(pred, panel["y"])
    o = M["open"]

    def sleeve(p):
        W = stock_ml_weights(p, mask, cs["long_short_frac"], cs["gross_leverage"], cs["pred_smoothing_halflife"])
        return bt.run(W, o, costs["cost_bps"], costs["borrow_bps_annual"])

    oos = pred.index.get_level_values(0).unique()
    base = panel.loc[panel.index.get_level_values(0).isin(oos)]
    res = {"stock_ml": sleeve(pred),
           "baseline_momentum_12_1": sleeve(base["mom_12_1"]),
           "baseline_reversal_5d": sleeve(-base["ret_5"])}

    We = etf_trend_weights(M["close"][ce["symbols"]], ce["lookbacks"], ce["vol_target"],
                           ce["max_asset_weight"], ce["max_gross"])
    res["etf_trend"] = bt.run(We.loc[oos[0]:], o[ce["symbols"]], costs["cost_bps"], costs["borrow_bps_annual"])
    a = S["sleeves"]
    combo = (a["stock_ml"] * res["stock_ml"]["net"] + a["etf_trend"] * res["etf_trend"]["net"]).dropna()

    lines = [f"# Research report {date.today()}", "",
             f"Out-of-sample period: {oos[0].date()} .. {oos[-1].date()} (walk-forward, purged, "
             f"retrain every {cs['retrain_every_days']} days on {cs['train_years']}y)", "",
             "## Stock ML signal quality",
             f"- mean Rank-IC: {ic.mean():.4f}   t-stat: {ic.mean() / ic.std() * np.sqrt(ic.count()):.2f}"
             f"   IC>0 on {(ic > 0).mean():.0%} of days", "",
             "## Net performance (after costs)"]
    for k, r in res.items():
        lines.append(f"- **{k}**: {fmt(bt.metrics(r['net']))} | gross Sharpe "
                     f"{bt.metrics(r['gross'])['sharpe']:.2f} | avg daily turnover {r['turnover'].mean():.2f}")
    lines += [f"- **COMBINED ({a})**: {fmt(bt.metrics(combo))}", "", "## Calendar-year returns (net)",
              pd.DataFrame({k: bt.yearly(r["net"]) for k, r in res.items()} | {"combined": bt.yearly(combo)})
              .to_markdown(floatfmt=".1%"), "",
              "## Rank-IC by year", ic.groupby(ic.index.year).mean().to_frame("rank_ic").to_markdown(floatfmt=".4f")]
    out = path("reports", f"research_{date.today()}.md")
    out.write_text("\n".join(lines) + "\n")
    pd.DataFrame({k: r["net"] for k, r in res.items()} | {"combined": combo}).to_csv(
        path("reports", f"research_daily_pnl_{date.today()}.csv"))
    print("\n".join(lines))
    print(f"\nreport -> {out}")
