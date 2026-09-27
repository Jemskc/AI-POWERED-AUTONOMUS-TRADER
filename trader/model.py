import json

import lightgbm as lgb
import numpy as np
import pandas as pd


def _dates(panel):
    return panel.index.get_level_values(0)


def fit(panel: pd.DataFrame, feats: list, target: str, params: dict, rounds: int) -> lgb.Booster:
    d = panel.dropna(subset=[target])
    return lgb.train(params, lgb.Dataset(d[feats], d[target]), num_boost_round=rounds)


def walk_forward(panel: pd.DataFrame, feats: list, target: str, params: dict, rounds: int,
                 train_years: int, retrain_every: int, horizon: int) -> pd.Series:
    """Out-of-sample predictions. Training labels that overlap the test window are purged."""
    dates = _dates(panel)
    ud = np.sort(dates.unique())
    gap = horizon + 2  # label at d spans open(d+1)..open(d+1+h)
    first = np.searchsorted(ud, ud[0] + pd.DateOffset(years=train_years))
    preds = []
    for i in range(first, len(ud), retrain_every):
        test = ud[i:i + retrain_every]
        tr = panel[(dates >= ud[i] - pd.DateOffset(years=train_years)) & (dates < ud[i - gap])]
        m = fit(tr, feats, target, params, rounds)
        te = panel[dates.isin(test)]
        preds.append(pd.Series(m.predict(te[feats]), index=te.index))
        print(f"  walk-forward {pd.Timestamp(test[0]).date()} .. {pd.Timestamp(test[-1]).date()}"
              f"  train_rows={len(tr):,}", flush=True)
    return pd.concat(preds).rename("pred")


def rank_ic(pred: pd.Series, y: pd.Series) -> pd.Series:
    j = pd.concat([pred, y.rename("y")], axis=1, join="inner").dropna()
    return j.groupby(level=0).apply(lambda g: g["pred"].rank().corr(g["y"].rank()) if len(g) > 20 else np.nan)


def save(model: lgb.Booster, feats: list, meta: dict, path):
    model.save_model(str(path))
    path.with_suffix(".json").write_text(json.dumps({"features": feats, **meta}, indent=2, default=str))


def load(path):
    return lgb.Booster(model_file=str(path)), json.loads(path.with_suffix(".json").read_text())
