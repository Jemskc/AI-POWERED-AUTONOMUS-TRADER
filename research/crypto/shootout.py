"""Model shoot-out on real Binance data: P(return over the next H hours > 0).

Models: always-up baseline, logistic regression, LightGBM, XGBoost, LSTM, Transformer, ensemble.
Walk-forward: expanding training window, 6-month test blocks from 2022-01-01, labels purged at the
boundary (no training label overlaps the test period). Decision at the close of hour t; the trade is
entered at open(t+1) and exited at open(t+1+H).
"""
import argparse
import math
import time
from datetime import date
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "crypto" / "klines_1h.parquet"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)
np.random.seed(0)


# ----------------------------------------------------------------------------- features
def build(H: int):
    df = pd.read_parquet(DATA)
    W = {k: df.pivot(index="time", columns="symbol", values=k).sort_index()
         for k in ("open", "high", "low", "close", "volume", "quote_volume", "trades", "taker_buy_base")}
    full = pd.date_range(W["close"].index[0], W["close"].index[-1], freq="h")
    W = {k: v.reindex(full) for k, v in W.items()}
    o, h, l, c, qv, nt = W["open"], W["high"], W["low"], W["close"], W["quote_volume"], W["trades"]
    tr = W["taker_buy_base"] / W["volume"].replace(0, np.nan)
    syms, idx = list(c.columns), c.index
    bc = lambda s: pd.DataFrame(np.repeat(np.asarray(s, float)[:, None], len(syms), 1), index=idx, columns=syms)

    lr = np.log(c).diff()
    vol24 = lr.rolling(24, min_periods=20).std()
    vol168 = lr.rolling(168, min_periods=120).std()
    F = {}
    for n in (1, 2, 4, 8, 12, 24, 48, 72, 168, 336):
        F[f"r{n}"] = np.log(c / c.shift(n)) / (vol168 * np.sqrt(n))
    F["lvol24"], F["lvol168"] = np.log(vol24), np.log(vol168)
    F["volratio"] = np.log(vol24 / vol168)
    F["hl"] = np.log(h / l) / vol168
    F["hl24"] = np.log(h / l).rolling(24, min_periods=19).mean() / vol168
    lqv = np.log(qv.replace(0, np.nan))
    F["qv_z"] = (lqv - lqv.rolling(168, min_periods=134).mean()) / lqv.rolling(168, min_periods=134).std()
    F["qv24_z"] = (lqv.rolling(24, min_periods=19).mean() - lqv.rolling(720, min_periods=576).mean()) / lqv.rolling(720, min_periods=576).std()
    lnt = np.log(nt.replace(0, np.nan))
    F["trades_z"] = (lnt - lnt.rolling(168, min_periods=134).mean()) / lnt.rolling(168, min_periods=134).std()
    F["taker1"] = tr - 0.5
    F["taker24"] = tr.rolling(24, min_periods=19).mean() - 0.5
    F["taker_z"] = (tr.rolling(24, min_periods=19).mean() - tr.rolling(720, min_periods=576).mean()) / tr.rolling(720, min_periods=576).std()
    F["ma168"] = (c / c.rolling(168, min_periods=134).mean() - 1) / (vol168 * np.sqrt(168))
    F["hi720"] = np.log(c / c.rolling(720, min_periods=576).max())
    F["lo720"] = np.log(c / c.rolling(720, min_periods=576).min())
    for n in (1, 4, 24, 168):
        F[f"btc_r{n}"] = bc(F[f"r{n}"]["BTC"])
    F["btc_lvol24"] = bc(F["lvol24"]["BTC"])
    F["mkt_r24"] = bc(F["r24"].mean(axis=1))
    F["mkt_r168"] = bc(F["r168"].mean(axis=1))
    F["rel_r24"] = F["r24"] - F["btc_r24"]
    for k in ("r24", "r168", "qv_z", "taker24"):
        F[f"xs_{k}"] = F[k].rank(axis=1, pct=True) - 0.5
    F["hsin"], F["hcos"] = bc(np.sin(2 * np.pi * idx.hour / 24)), bc(np.cos(2 * np.pi * idx.hour / 24))
    F["dow"] = bc(idx.dayofweek / 6 - 0.5)

    fwd = np.log(o.shift(-1 - H) / o.shift(-1))
    names = list(F)
    T, S = len(idx), len(syms)
    X = np.stack([F[k].to_numpy(np.float32) for k in names], axis=-1).reshape(T * S, len(names))
    X[~np.isfinite(X)] = np.nan
    ret = fwd.to_numpy().reshape(-1)
    y = np.where(np.isfinite(ret), (ret > 0).astype(np.float32), np.nan)
    tidx = np.repeat(np.arange(T), S)
    feat_ok = ~np.isnan(X).any(axis=1) & np.isfinite(c.to_numpy().reshape(-1))
    return dict(X=X, y=y, ret=ret, tidx=tidx, times=idx, syms=syms, S=S, names=names, ok=feat_ok)


# ----------------------------------------------------------------------------- sequence models
class LSTMNet(nn.Module):
    def __init__(self, f, hdim=64):
        super().__init__()
        self.rnn = nn.LSTM(f, hdim, num_layers=2, batch_first=True, dropout=0.2)
        self.head = nn.Sequential(nn.LayerNorm(hdim), nn.Linear(hdim, 1))

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.head(out[:, -1]).squeeze(-1)


class TransformerNet(nn.Module):
    def __init__(self, f, L, d=64):
        super().__init__()
        self.inp = nn.Linear(f, d)
        self.pos = nn.Parameter(torch.zeros(1, L, d))
        layer = nn.TransformerEncoderLayer(d, 4, 128, dropout=0.1, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, 2)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 1))

    def forward(self, x):
        z = self.enc(self.inp(x) + self.pos)
        return self.head(z[:, -1]).squeeze(-1)


def windows(rows, S, L):
    """rows: flat indices (t*S+s) -> (n, L) indices of the previous L hours of the same symbol."""
    return rows[:, None] - S * np.arange(L - 1, -1, -1)[None, :]


def train_seq(kind, Xg, S, L, fit_rows, val_rows, y_all, epochs=12, bs=4096, patience=2):
    f = Xg.shape[1]
    net = (LSTMNet(f) if kind == "lstm" else TransformerNet(f, L)).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    lossf = nn.BCEWithLogitsLoss()
    yg = torch.as_tensor(y_all, device=DEV)
    fit_w = torch.as_tensor(windows(fit_rows, S, L), device=DEV)
    val_w = torch.as_tensor(windows(val_rows, S, L), device=DEV)
    fit_r, val_r = torch.as_tensor(fit_rows, device=DEV), torch.as_tensor(val_rows, device=DEV)
    best, best_state, bad = 1e9, None, 0
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(len(fit_r), device=DEV)
        for i in range(0, len(perm), bs):
            b = perm[i:i + bs]
            loss = lossf(net(Xg[fit_w[b]]), yg[fit_r[b]])
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
        vl = lossf(torch.as_tensor(predict_seq(net, Xg, val_w, logits=True), device=DEV), yg[val_r]).item()
        if vl < best - 1e-5:
            best, bad = vl, 0
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad > patience:
                break
    net.load_state_dict(best_state)
    return net, ep + 1


@torch.no_grad()
def predict_seq(net, Xg, win, bs=16384, logits=False):
    net.eval()
    out = torch.cat([net(Xg[win[i:i + bs]]) for i in range(0, len(win), bs)])
    return (out if logits else torch.sigmoid(out)).cpu().numpy()


# ----------------------------------------------------------------------------- evaluation
def hold_until_confident(pm, thr):
    """+1/-1 when |p-0.5| > thr, else keep the previous position; flat until the first confident call."""
    sig = np.where(pm - 0.5 > thr, 1.0, np.where(0.5 - pm > thr, -1.0, np.nan))
    return pd.DataFrame(sig).ffill().fillna(0.0).to_numpy()


def trading(P, D, H, cost_bps):
    """Rebalance every H hours. Directional: sign(p-0.5) per coin, equal weight.
    Cross-sectional: long top 5 / short bottom 5 by p. Costs on turnover."""
    T, S = len(D["times"]), D["S"]
    ret = np.nan_to_num(np.expm1(D["ret"].reshape(T, S)))
    reb = np.arange(0, T, H)
    out = {}
    pm = P.reshape(T, S)[reb]
    avail = ~np.isnan(pm)
    n = np.maximum(avail.sum(1, keepdims=True), 1)
    books = {
        "dir": np.where(avail, np.sign(pm - 0.5), 0) / n,
        # only trade confident calls; hold the previous position otherwise (cuts turnover)
        "conf": hold_until_confident(np.where(avail, pm, 0.5), 0.04) / n,
    }
    rk = np.where(avail, pm, np.nan)
    order = np.argsort(np.where(avail, rk, np.inf), axis=1)  # ascending, NaN last
    xs = np.zeros_like(pm)
    for i in range(len(reb)):
        k = int(avail[i].sum())
        if k >= 10:
            xs[i, order[i, :5]] = -0.1
            xs[i, order[i, k - 5:k]] = 0.1
    books["xs"] = xs
    live = avail.any(1)
    for name, w in books.items():
        r = ret[reb]
        gross = (w * r).sum(1)
        turn = np.abs(np.diff(w, axis=0, prepend=0)).sum(1)
        net = gross - turn * cost_bps / 1e4
        gross, net, turn = gross[live], net[live], turn[live]
        ann = math.sqrt(8760 / H)
        eq = np.cumprod(1 + net)
        out[name] = dict(sharpe_net=net.mean() / net.std() * ann, sharpe_gross=gross.mean() / gross.std() * ann,
                         ann_ret_net=net.mean() * 8760 / H, max_dd=float((eq / np.maximum.accumulate(eq) - 1).min()),
                         total_net=float(eq[-1] - 1), turnover=turn.mean())
    return out


def benchmark(D, H, live_mask):
    T, S = len(D["times"]), D["S"]
    ret = np.expm1(D["ret"].reshape(T, S))[np.arange(0, T, H)]
    r = np.nanmean(np.where(live_mask, ret, np.nan), axis=1)
    r = r[~np.isnan(r)]
    eq = np.cumprod(1 + r)
    return dict(sharpe=r.mean() / r.std() * math.sqrt(8760 / H), total=float(eq[-1] - 1),
                max_dd=float((eq / np.maximum.accumulate(eq) - 1).min()))


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--H", type=int, default=4)
    ap.add_argument("--L", type=int, default=48, help="sequence length (hours) for LSTM/Transformer")
    ap.add_argument("--cost_bps", type=float, default=7.0, help="per side: 5bp perp taker fee + 2bp slippage")
    ap.add_argument("--test_start", default="2022-01-01")
    a = ap.parse_args()
    t0 = time.time()
    D = build(a.H)
    X, y, tidx, S, times = D["X"], D["y"], D["tidx"], D["S"], D["times"]
    print(f"H={a.H}h  rows={len(y):,}  features={len(D['names'])}  symbols={S}  "
          f"{times[0]} .. {times[-1]}  ({time.time() - t0:.0f}s)", flush=True)

    # rows usable as inputs (features + a full, valid L-hour history for the sequence models)
    ok = D["ok"].copy()
    ok_t = ok.reshape(-1, S)
    hist = pd.DataFrame(ok_t).rolling(a.L, min_periods=a.L).sum().to_numpy() == a.L
    ok = ok & np.nan_to_num(hist, nan=0).astype(bool).reshape(-1)
    has_y = ok & ~np.isnan(y)

    starts = pd.date_range(pd.Timestamp(a.test_start, tz="UTC"), times[-1], freq="6MS")
    models = ["always_up", "logreg", "lightgbm", "xgboost", "lstm", "transformer"]
    P = {m: np.full(len(y), np.nan) for m in models}
    fold_auc = {m: [] for m in models}
    purge = a.H + 1

    for fi, st in enumerate(starts):
        en = starts[fi + 1] if fi + 1 < len(starts) else times[-1] + pd.Timedelta(hours=1)
        ts, te = times.searchsorted(st), times.searchsorted(en)
        train = has_y & (tidx < ts - purge)
        test = ok & (tidx >= ts) & (tidx < te)
        tr_rows = np.flatnonzero(train)
        cut_t = int(np.quantile(tidx[tr_rows], 0.85))  # last 15% of training time = validation
        fit_rows = tr_rows[tidx[tr_rows] < cut_t - purge]
        val_rows = tr_rows[tidx[tr_rows] >= cut_t]
        te_rows = np.flatnonzero(test)

        mu = np.nanmean(X[fit_rows], 0)
        sd = np.nanstd(X[fit_rows], 0) + 1e-6
        Z = np.clip((X - mu) / sd, -5, 5)
        Z = np.nan_to_num(Z).astype(np.float32)
        info = []

        P["always_up"][te_rows] = y[fit_rows].mean()

        c0 = time.time()
        sub = np.random.choice(fit_rows, min(len(fit_rows), 400_000), replace=False)
        lrm = LogisticRegression(C=0.05, max_iter=300).fit(Z[sub], y[sub])
        P["logreg"][te_rows] = lrm.predict_proba(Z[te_rows])[:, 1]
        info.append(f"logreg {time.time() - c0:.0f}s")

        c0 = time.time()
        gb = lgb.train(dict(objective="binary", learning_rate=0.03, num_leaves=31, min_data_in_leaf=2000,
                            feature_fraction=0.7, bagging_fraction=0.7, bagging_freq=1, lambda_l2=10,
                            num_threads=16, verbose=-1, seed=0),
                       lgb.Dataset(Z[fit_rows], y[fit_rows]), 2000,
                       valid_sets=[lgb.Dataset(Z[val_rows], y[val_rows])],
                       callbacks=[lgb.early_stopping(100, verbose=False)])
        P["lightgbm"][te_rows] = gb.predict(Z[te_rows], num_iteration=gb.best_iteration)
        info.append(f"lgbm {gb.best_iteration}it {time.time() - c0:.0f}s")

        c0 = time.time()
        xm = xgb.XGBClassifier(n_estimators=2000, learning_rate=0.03, max_depth=5, subsample=0.7,
                               colsample_bytree=0.7, min_child_weight=200, reg_lambda=10, tree_method="hist",
                               device=DEV, early_stopping_rounds=100, eval_metric="logloss", random_state=0)
        xm.fit(Z[fit_rows], y[fit_rows], eval_set=[(Z[val_rows], y[val_rows])], verbose=False)
        P["xgboost"][te_rows] = xm.predict_proba(Z[te_rows])[:, 1]
        info.append(f"xgb {xm.best_iteration}it {time.time() - c0:.0f}s")

        Xg = torch.as_tensor(Z, device=DEV)
        y_all = np.nan_to_num(y).astype(np.float32)
        te_w = torch.as_tensor(windows(te_rows, S, a.L), device=DEV)
        for kind in ("lstm", "transformer"):
            c0 = time.time()
            net, eps = train_seq(kind, Xg, S, a.L, fit_rows, val_rows, y_all)
            P[kind][te_rows] = predict_seq(net, Xg, te_w)
            info.append(f"{kind} {eps}ep {time.time() - c0:.0f}s")
            del net
        del Xg, te_w
        torch.cuda.empty_cache()

        ev = te_rows[~np.isnan(y[te_rows])]
        for m in models:
            fold_auc[m].append(roc_auc_score(y[ev], P[m][ev]) if m != "always_up" else 0.5)
        print(f"fold {st.date()}..{(en - pd.Timedelta(hours=1)).date()}  train={len(fit_rows):,} test={len(te_rows):,}  "
              f"up-rate={y[ev].mean():.3f}  AUC " + " ".join(f"{m}={fold_auc[m][-1]:.4f}" for m in models[1:])
              + f"  [{', '.join(info)}]", flush=True)

    P["ensemble"] = np.nanmean(np.stack([P[m] for m in ("lightgbm", "xgboost", "lstm", "transformer")]), 0)
    fold_auc["ensemble"] = []
    models.append("ensemble")

    # ---------------------------------------------------------------- report
    oos = ok & (tidx >= times.searchsorted(starts[0]))
    ev = oos & ~np.isnan(y)
    yv = y[ev]
    base_up = yv.mean()
    # non-overlapping sample (every H-th hour) for an honest binomial z-test
    no = ev & (tidx % a.H == 0)
    rows, calib = [], {}
    for m in models:
        p = P[m][ev]
        acc = ((p > 0.5) == (yv == 1)).mean()
        pn, yn = P[m][no], y[no]
        acc_n = ((pn > 0.5) == (yn == 1)).mean()
        p0 = max(yn.mean(), 1 - yn.mean())
        z = (acc_n - p0) / math.sqrt(p0 * (1 - p0) / len(yn))
        fa = np.array([roc_auc_score(y[ev & (tidx >= times.searchsorted(s)) &
                                       (tidx < (times.searchsorted(starts[i + 1]) if i + 1 < len(starts) else len(times)))],
                                     P[m][ev & (tidx >= times.searchsorted(s)) &
                                          (tidx < (times.searchsorted(starts[i + 1]) if i + 1 < len(starts) else len(times)))])
                       if m != "always_up" else 0.5 for i, s in enumerate(starts)])
        tstat = (fa.mean() - 0.5) / (fa.std(ddof=1) / math.sqrt(len(fa))) if m != "always_up" else 0.0
        tr = trading(P[m], D, a.H, a.cost_bps)
        trm = trading(P[m], D, a.H, 3.0)  # maker scenario: 2bp fee + 1bp adverse selection
        rows.append(dict(model=m, mean_prob_up=p.mean(), accuracy=acc, acc_vs_always_up=acc - base_up,
                         z_vs_majority=z, auc=roc_auc_score(yv, p) if m != "always_up" else 0.5,
                         auc_folds_pos=f"{(fa > 0.5).sum()}/{len(fa)}", auc_tstat=tstat,
                         logloss=log_loss(yv, np.clip(p, 1e-6, 1 - 1e-6)), brier=brier_score_loss(yv, p),
                         dir_sharpe_net=tr["dir"]["sharpe_net"], dir_sharpe_gross=tr["dir"]["sharpe_gross"],
                         dir_total_net=tr["dir"]["total_net"], xs_sharpe_net=tr["xs"]["sharpe_net"],
                         xs_sharpe_gross=tr["xs"]["sharpe_gross"], xs_total_net=tr["xs"]["total_net"],
                         xs_maxdd=tr["xs"]["max_dd"], dir_sharpe_maker=trm["dir"]["sharpe_net"],
                         conf_sharpe_gross=tr["conf"]["sharpe_gross"], conf_sharpe_net=tr["conf"]["sharpe_net"],
                         conf_sharpe_maker=trm["conf"]["sharpe_net"], conf_total_net=tr["conf"]["total_net"],
                         conf_total_maker=trm["conf"]["total_net"], conf_maxdd=tr["conf"]["max_dd"],
                         conf_turnover=tr["conf"]["turnover"], dir_turnover=tr["dir"]["turnover"]))
        q = pd.qcut(p, 10, labels=False, duplicates="drop") if m != "always_up" else None
        if q is not None:
            calib[m] = pd.DataFrame({"p": p, "y": yv, "q": q}).groupby("q").agg(pred=("p", "mean"), actual=("y", "mean"))
    R = pd.DataFrame(rows).set_index("model")
    live_mask = (~np.isnan(P["lightgbm"])).reshape(len(times), S)[np.arange(0, len(times), a.H)]
    bh = benchmark(D, a.H, live_mask)

    last_t = int(tidx[ok].max())
    cur = {m: {D["syms"][s]: P[m][last_t * S + s] for s in range(S)} for m in models}
    cur_df = pd.DataFrame(cur).loc[["BTC", "ETH", "SOL", "XRP", "DOGE"]]

    out = ROOT / "reports" / f"crypto_model_shootout_H{a.H}_{date.today()}.md"
    out.parent.mkdir(exist_ok=True)
    fmt = {c: "{:.1%}" for c in ("mean_prob_up", "accuracy", "acc_vs_always_up", "dir_total_net", "xs_total_net",
                                 "xs_maxdd", "conf_total_net", "conf_total_maker", "conf_maxdd")}
    Rs = R.copy()
    for c_, f_ in fmt.items():
        Rs[c_] = Rs[c_].map(f_.format)
    lines = [
        f"# Crypto model shoot-out: P(next {a.H}h return > 0), real Binance data",
        f"_Generated {date.today()} · {S} coins · hourly bars {times[0].date()} .. {times[-1]} UTC · "
        f"out-of-sample {starts[0].date()} .. {times[-1].date()} · {len(yv):,} OOS predictions · "
        f"costs {a.cost_bps} bp/side · runtime {(time.time() - t0) / 60:.0f} min_",
        "", f"**Actual up-rate in the test period: {base_up:.1%}** (the 'always up' accuracy to beat)", "",
        "## Prediction quality", Rs[["mean_prob_up", "accuracy", "acc_vs_always_up", "z_vs_majority", "auc",
                                     "auc_folds_pos", "auc_tstat", "logloss", "brier"]].to_markdown(floatfmt=".4f"),
        "", "## Trading the predictions (rebalanced every H hours, net of costs)",
        "- `dir`: long if p>0.5 else short, every coin equally weighted", "- `xs`: long the 5 highest-p coins, "
        "short the 5 lowest (market-neutral)",
        f"- benchmark equal-weight buy & hold: Sharpe {bh['sharpe']:.2f}, total {bh['total']:.1%}, max DD {bh['max_dd']:.1%}",
        "", Rs[["dir_sharpe_gross", "dir_sharpe_net", "dir_total_net", "xs_sharpe_gross", "xs_sharpe_net",
                "xs_total_net", "xs_maxdd"]].to_markdown(floatfmt=".2f"),
        "", f"### Confidence-filtered (`conf`: trade only when |p-0.5|>0.04, else hold) and maker-cost (3bp) variants",
        Rs[["dir_turnover", "dir_sharpe_maker", "conf_turnover", "conf_sharpe_gross", "conf_sharpe_net",
            "conf_total_net", "conf_sharpe_maker", "conf_total_maker", "conf_maxdd"]].to_markdown(floatfmt=".2f"),
        "", f"## Latest out-of-sample estimate: P(up over next {a.H}h) at {times[last_t]} UTC",
        cur_df.to_markdown(floatfmt=".1%"), "", "## Calibration (predicted vs actual up-rate by decile)"]
    for m, cdf in calib.items():
        lines += [f"**{m}**: " + "  ".join(f"{r.pred:.3f}→{r.actual:.3f}" for r in cdf.itertuples())]
    out.write_text("\n".join(lines) + "\n")
    np.savez_compressed(ROOT / "data" / "crypto" / f"shootout_preds_H{a.H}.npz", **{m: P[m] for m in models},
                        y=y, tidx=tidx)
    print("\n".join(lines))
    print(f"\nreport -> {out}")


if __name__ == "__main__":
    main()
