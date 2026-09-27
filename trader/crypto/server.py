"""Live crypto dashboard backend: reads the engine's state files + keyless Binance market data."""
import json
import math
import os
import secrets
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from trader.config import ROOT
from trader.crypto import market
from trader.crypto.engine import BOOKS, CFG, DB, HEART, KILL, SD, paths
from trader.crypto.features import SYMBOLS

STATIC = Path(__file__).parent / "static"
TOKEN = os.environ.get("DASHBOARD_TOKEN", "")
app = FastAPI(docs_url=None, redoc_url=None)
app.mount("/static", StaticFiles(directory=ROOT / "trader" / "dashboard" / "static"), name="static")
app.mount("/cstatic", StaticFiles(directory=STATIC), name="cstatic")
_cache = {}


def cached(key, ttl, fn):
    hit = _cache.get(key)
    if hit and time.time() - hit[0] < ttl:
        return hit[1]
    v = fn()
    _cache[key] = (time.time(), v)
    return v


def clean(x):
    if isinstance(x, float) and not math.isfinite(x):
        return None
    return x


@app.middleware("http")
async def auth(request: Request, call_next):
    tok = request.query_params.get("token") or request.cookies.get("tok") or ""
    if not TOKEN or not secrets.compare_digest(tok, TOKEN):
        if request.url.path.startswith("/api"):
            return JSONResponse({"error": "unauthorized"}, status_code=401)
        return HTMLResponse("<h3 style='font-family:sans-serif'>Open the URL printed by scripts/crypto_dashboard.py</h3>", 401)
    resp = await call_next(request)
    if request.query_params.get("token"):
        resp.set_cookie("tok", tok, httponly=True, samesite="strict", max_age=30 * 86400)
    return resp


@app.exception_handler(Exception)
async def errors(request: Request, exc: Exception):
    return JSONResponse({"error": f"{type(exc).__name__}: {exc}"}, status_code=503)


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


def tickers():
    return cached("tk", 3, lambda: market.tickers(SYMBOLS))


def book(name):
    f = paths(name)[0]
    return json.loads(f.read_text()) if f.exists() else None


def preds_df():
    con = sqlite3.connect(DB)
    try:
        return pd.read_sql("SELECT * FROM preds", con)
    finally:
        con.close()


# ------------------------------------------------------------------------------- state
@app.get("/api/state")
def state():
    tk = tickers()
    hb = json.loads(HEART.read_text()) if HEART.exists() else {}
    now = datetime.now(timezone.utc)
    age = (now - datetime.fromisoformat(hb["time"])).total_seconds() if hb else None
    books = {}
    for n in BOOKS:
        b = book(n)
        if not b:
            continue
        pos = b["positions"]
        lv = sum(p["qty"] * tk[s]["last"] for s, p in pos.items() if p["qty"] > 0)
        sv = sum(p["qty"] * tk[s]["last"] for s, p in pos.items() if p["qty"] < 0)
        eq = b["cash"] + lv + sv
        bh = float(np.mean([tk[s]["last"] / p for s, p in b["start_prices"].items()]) - 1)
        books[n] = dict(threshold=b.get("threshold") if b.get("threshold") is not None else BOOKS[n].get("enter_p", 0.5) - 0.5,
                        kind=BOOKS[n].get("type", "tranche"), start=b["start_time"], equity=eq, start_capital=b["start_capital"],
                        pnl=eq - b["start_capital"], ret=eq / b["start_capital"] - 1, bh=bh,
                        btc=tk["BTC"]["last"] / b["start_prices"]["BTC"] - 1, day_pnl=eq - b["day_start_equity"],
                        trades=b["n_trades"], fees=b["fees"], realized=b["realized"],
                        unrealized=sum((tk[s]["last"] - p["avg"]) * p["qty"] for s, p in pos.items()),
                        n_long=sum(p["qty"] > 0 for p in pos.values()), n_short=sum(p["qty"] < 0 for p in pos.values()),
                        gross=(lv - sv) / eq, net=(lv + sv) / eq, drawdown=eq / max(b["peak"], eq) - 1,
                        halted=b["halted"], hours=(now - datetime.fromisoformat(b["start_time"])).total_seconds() / 3600)
    nxt = (pd.Timestamp(now).floor("h") + pd.Timedelta(hours=1, seconds=15)).isoformat()
    meta = json.loads((ROOT / "models" / "crypto" / f"latest_H{CFG['H']}.json").read_text())
    return dict(now=now.isoformat(), heartbeat=hb.get("time"), heartbeat_age=age, live=age is not None and age < 120,
                last_decision=hb.get("last_decision"), next_decision=nxt, books=books, kill=KILL.exists(),
                cfg=dict(H=CFG["H"], tranches=CFG["tranches"], fee_bps=CFG["fee_bps"], slippage_bps=CFG["slippage_bps"],
                         max_daily_loss=CFG["max_daily_loss"], max_drawdown=CFG["max_drawdown"]),
                model=dict(trained_on=meta["trained_on"], data_end=meta["data_end"], rows=meta["rows"]))


@app.get("/api/equity")
def equity():
    out = {}
    for n in BOOKS:
        f = paths(n)[1]
        if f.exists():
            df = pd.read_csv(f)
            out[n] = dict(time=[int(pd.Timestamp(t).timestamp()) for t in df.time], equity=df.equity.round(2).tolist(),
                          bh=df.bh_equity.round(2).tolist())
    return out


@app.get("/api/positions")
def positions(book_name: str = "active"):
    b, tk = book(book_name), tickers()
    if not b:
        return []
    eq = state()["books"][book_name]["equity"]
    rows = [dict(symbol=s, qty=p["qty"], side="LONG" if p["qty"] > 0 else "SHORT", avg=p["avg"], last=tk[s]["last"],
                 value=p["qty"] * tk[s]["last"], upl=(tk[s]["last"] - p["avg"]) * p["qty"],
                 uplpc=(tk[s]["last"] / p["avg"] - 1) * (1 if p["qty"] > 0 else -1) if p["avg"] else None,
                 chg24=tk[s]["chg24"], weight=p["qty"] * tk[s]["last"] / eq) for s, p in b["positions"].items()]
    return sorted(rows, key=lambda r: -abs(r["value"]))


@app.get("/api/trades")
def trades(book_name: str = "active", limit: int = 300):
    f = paths(book_name)[2]
    if not f.exists():
        return []
    df = pd.read_csv(f).tail(limit).iloc[::-1]
    return json.loads(df.to_json(orient="records"))


@app.get("/api/quotes")
def quotes():
    tk, P = tickers(), preds_df()
    last = P[P.t == P.t.max()].set_index("symbol") if len(P) else pd.DataFrame()
    sig = {}
    for n in BOOKS:
        b = book(n)
        sig[n] = {s: (1 if p["qty"] > 0 else -1) for s, p in (b or {"positions": {}})["positions"].items()}
    return [dict(symbol=s, **tk[s], p=clean(float(last.at[s, "p_ens"])) if s in last.index else None,
                 pos={n: sig[n].get(s, 0) for n in BOOKS}) for s in SYMBOLS]


@app.get("/api/bars")
def bars(symbol: str = "BTC", interval: str = "1h"):
    symbol = symbol.upper()
    df = cached(("bars", symbol, interval), 20, lambda: market.klines(symbol, interval=interval, limit=500, closed_only=False))
    candles = [dict(time=int(r.time.timestamp()), open=r.open, high=r.high, low=r.low, close=r.close, volume=r.volume)
               for r in df.itertuples()]
    P = preds_df()
    P = P[P.symbol == symbol]
    marks = [dict(time=int(pd.Timestamp(r.t).timestamp()) + 3600, p=r.p_ens, up=None if pd.isna(r.up) else int(r.up))
             for r in P.itertuples()]
    return dict(candles=candles, preds=marks)


@app.get("/api/accuracy")
def accuracy():
    P = preds_df()
    S = P.dropna(subset=["ret"])
    out = dict(n_preds=len(P), n_hours=int(P.t.nunique()) if len(P) else 0, n_scored=len(S))
    if not len(S):
        return out
    y = S.up.to_numpy()

    def auc(p):
        if len(set(y)) < 2:
            return None
        r = pd.Series(p).rank().to_numpy()
        n1 = y.sum()
        return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(y) - n1)))

    out["up_rate"] = float(y.mean())
    out["models"] = {m[2:]: dict(acc=float(((S[m] > 0.5) == (y == 1)).mean()), auc=auc(S[m].to_numpy()))
                     for m in ("p_xgb", "p_lgbm", "p_ens")}
    out["confident"] = {}
    for thr in (0.02, 0.04):
        c = S[(S.p_ens - 0.5).abs() > thr]
        out["confident"][str(thr)] = dict(n=len(c), acc=float(((c.p_ens > 0.5) == (c.up == 1)).mean()) if len(c) else None)
    out["noise_band"] = 1.96 * 0.5 / math.sqrt(len(S))
    g = S.assign(ok=((S.p_ens > 0.5) == (S.up == 1)).astype(int)).groupby("t")
    by = g.agg(n=("ok", "size"), acc=("ok", "mean"), up=("up", "mean"), ret=("ret", "mean")).reset_index()
    by["cum_acc"] = (by.acc * by.n).cumsum() / by.n.cumsum()
    out["by_hour"] = [dict(t=r.t, time=int(pd.Timestamp(r.t).timestamp()), n=int(r.n), acc=r.acc, up=r.up, ret=r.ret,
                           cum_acc=r.cum_acc) for r in by.itertuples()]
    q = pd.cut(S.p_ens, [0, 0.46, 0.48, 0.5, 0.52, 0.54, 1])
    cal = S.groupby(q, observed=True).agg(n=("up", "size"), pred=("p_ens", "mean"), actual=("up", "mean"))
    out["calibration"] = [dict(bin=str(k), n=int(r.n), pred=r.pred, actual=r.actual) for k, r in cal.iterrows()]
    return out


@app.get("/api/predictions")
def predictions(limit: int = 400):
    P = preds_df().sort_values(["t", "p_ens"], ascending=[False, False]).head(limit)
    return json.loads(P.to_json(orient="records"))


@app.get("/api/log")
def engine_log(lines: int = 200):
    f = ROOT / "logs" / "crypto_engine.log"
    if not f.exists():
        return []
    txt = [x for x in f.read_text().splitlines() if "Warning" not in x and "warn(" not in x]
    return txt[-lines:]


@app.get("/api/research")
def research():
    reps = sorted((ROOT / "reports").glob("crypto_model_shootout_*.md"))
    return [dict(name=r.name, text=r.read_text()) for r in reps[::-1]]


@app.post("/api/kill")
def kill():
    KILL.write_text(f"manual kill from dashboard {datetime.now(timezone.utc).isoformat(timespec='seconds')}\n")
    return {"kill": True}


@app.post("/api/resume")
def resume():
    KILL.unlink(missing_ok=True)
    (SD / "resume").touch()  # the engine clears the halt on its next tick
    return {"kill": False}
