"""Dashboard backend: live Alpaca account/market data + the trader's own state, logs and controls.

Binds to 127.0.0.1 and requires DASHBOARD_TOKEN (shared machine: other users can reach localhost).
"""
import json
import math
import os
import secrets
import subprocess
import sys
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
from alpaca.data.enums import Adjustment, DataFeed
from alpaca.data.requests import StockBarsRequest, StockSnapshotRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.trading.enums import QueryOrderStatus
from alpaca.trading.requests import GetOrdersRequest, GetPortfolioHistoryRequest
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from trader import alpaca_io as io
from trader.config import ROOT, path, risk_limits, settings
from trader.risk import KILL_FILE, STATE_FILE
from trader.shadow import load_shadow_book

STATIC = Path(__file__).parent / "static"
S = settings()
TOKEN = os.environ.get("DASHBOARD_TOKEN", "")
ETFS = set(S["etf_trend"]["symbols"])
app = FastAPI(title="ai-trader dashboard", docs_url=None, redoc_url=None)
app.mount("/static", StaticFiles(directory=STATIC), name="static")

_clients, _cache, _lock = {}, {}, threading.Lock()


def tc():
    if "tc" not in _clients:
        _clients["tc"] = io.trading_client()
    return _clients["tc"]


def dc():
    if "dc" not in _clients:
        _clients["dc"] = io.data_client()
    return _clients["dc"]


def cached(key, ttl, fn):
    now = time.time()
    with _lock:
        hit = _cache.get(key)
        if hit and now - hit[0] < ttl:
            return hit[1]
    val = fn()
    with _lock:
        _cache[key] = (now, val)
    return val


def num(x):
    try:
        x = float(x)
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def sleeve(sym):
    return "trend" if sym in ETFS else "ml"


# ---------------------------------------------------------------- auth
@app.middleware("http")
async def auth(request: Request, call_next):
    tok = request.query_params.get("token") or request.cookies.get("tok") or request.headers.get("x-token") or ""
    if not TOKEN or not secrets.compare_digest(tok, TOKEN):
        if request.url.path.startswith("/api"):
            return JSONResponse({"error": "unauthorized"}, status_code=401)
        return HTMLResponse("<h3 style='font-family:sans-serif'>Open the URL printed by scripts/dashboard.py "
                            "(it contains ?token=...)</h3>", status_code=401)
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


# ---------------------------------------------------------------- market data
def snapshots(symbols):
    symbols = sorted(set(symbols))
    if not symbols:
        return {}

    def fetch():
        out = {}
        for i in range(0, len(symbols), 200):
            snaps = dc().get_stock_snapshot(StockSnapshotRequest(symbol_or_symbols=symbols[i:i + 200],
                                                                 feed=DataFeed.IEX))
            for s, sn in snaps.items():
                last = num(sn.latest_trade.price) if sn.latest_trade else None
                prev = num(sn.previous_daily_bar.close) if sn.previous_daily_bar else None
                day = sn.daily_bar
                out[s] = dict(symbol=s, last=last, prev_close=prev,
                              chg=(last / prev - 1) if last and prev else None,
                              open=num(day.open) if day else None, high=num(day.high) if day else None,
                              low=num(day.low) if day else None, volume=num(day.volume) if day else None,
                              bid=num(sn.latest_quote.bid_price) if sn.latest_quote else None,
                              ask=num(sn.latest_quote.ask_price) if sn.latest_quote else None,
                              ts=sn.latest_trade.timestamp.isoformat() if sn.latest_trade else None)
        return out
    return cached(("snap", tuple(symbols)), 3, fetch)


@app.get("/api/quotes")
def quotes(symbols: str = ""):
    syms = [s.strip().upper() for s in symbols.split(",") if s.strip()] or sorted(ETFS)
    return list(snapshots(syms).values())


@app.get("/api/bars")
def bars(symbol: str, tf: str = "1D"):
    symbol = symbol.upper()

    def fetch():
        now = datetime.now(timezone.utc)
        if tf == "5Min":
            req = StockBarsRequest(symbol_or_symbols=symbol, timeframe=TimeFrame(5, TimeFrameUnit.Minute),
                                   start=now - timedelta(days=7), feed=DataFeed.IEX)
        else:
            req = StockBarsRequest(symbol_or_symbols=symbol, timeframe=TimeFrame.Day, start=now - timedelta(days=730),
                                   end=now - timedelta(minutes=20), adjustment=Adjustment.ALL, feed=DataFeed.SIP)
        df = dc().get_stock_bars(req).df
        if df.empty:
            return []
        df = df.reset_index()
        return [dict(time=int(r.timestamp.timestamp()), open=r.open, high=r.high, low=r.low, close=r.close,
                     volume=r.volume) for r in df.itertuples()]
    return cached(("bars", symbol, tf), 15 if tf == "5Min" else 300, fetch)


# ---------------------------------------------------------------- account / book
def account():
    def fetch():
        a = tc().get_account()
        return dict(equity=num(a.equity), last_equity=num(a.last_equity), cash=num(a.cash),
                    buying_power=num(a.buying_power), long_value=num(a.long_market_value),
                    short_value=num(a.short_market_value), status=str(a.status), number=a.account_number)
    return cached("acct", 4, fetch)


def paper_positions():
    def fetch():
        return [dict(symbol=p.symbol, qty=abs(num(p.qty)) * (-1 if str(p.side).lower().endswith("short") else 1),
                     avg=num(p.avg_entry_price), last=num(p.current_price), value=num(p.market_value),
                     upl=num(p.unrealized_pl), uplpc=num(p.unrealized_plpc), today=num(p.change_today),
                     upl_today=num(p.unrealized_intraday_pl)) for p in tc().get_all_positions()]
    return cached("pos", 4, fetch)


def shadow_positions():
    book = load_shadow_book()
    snap = snapshots(list(book))
    out = []
    for s, b in book.items():
        q, sn = b["qty"], snap.get(s, {})
        last = sn.get("last")
        avg = b.get("avg")
        val = q * last if last else None
        upl = (last - avg) * q if last and avg else None
        out.append(dict(symbol=s, qty=q, avg=avg, last=last, value=val, upl=upl,
                        uplpc=(upl / abs(avg * q)) if upl is not None and avg else None, today=sn.get("chg"),
                        upl_today=(last - sn["prev_close"]) * q if last and sn.get("prev_close") else None))
    return out


@app.get("/api/positions")
def positions(book: str = "paper"):
    rows = paper_positions() if book == "paper" else shadow_positions()
    eq = account()["equity"] or 1
    for r in rows:
        r["sleeve"] = sleeve(r["symbol"])
        r["side"] = "LONG" if r["qty"] > 0 else "SHORT"
        r["weight"] = (r["value"] or 0) / eq
    return sorted(rows, key=lambda r: -abs(r["value"] or 0))


@app.get("/api/overview")
def overview(book: str = "paper"):
    a = account()
    clock = cached("clock", 10, lambda: tc().get_clock())
    rows = positions(book)
    eq = a["equity"] or 1
    lv = sum(r["value"] or 0 for r in rows if r["qty"] > 0)
    sv = sum(r["value"] or 0 for r in rows if r["qty"] < 0)
    by_sleeve = {}
    for r in rows:
        d = by_sleeve.setdefault(r["sleeve"], {"long": 0.0, "short": 0.0, "n": 0, "upl_today": 0.0})
        d["long" if r["qty"] > 0 else "short"] += (r["value"] or 0) / eq
        d["n"] += 1
        d["upl_today"] += r["upl_today"] or 0
    state = json.loads(STATE_FILE.read_text()) if STATE_FILE.exists() else {}
    peak = max(state.get("peak_equity", eq), eq)
    L = risk_limits()
    return dict(
        mode=settings()["mode"], book=book, account=a,
        day_pnl=(a["equity"] - a["last_equity"]) if a["equity"] and a["last_equity"] else None,
        day_pnl_pct=(a["equity"] / a["last_equity"] - 1) if a["equity"] and a["last_equity"] else None,
        book_upl=sum(r["upl"] or 0 for r in rows), book_upl_today=sum(r["upl_today"] or 0 for r in rows),
        market=dict(is_open=clock.is_open, next_open=clock.next_open.isoformat(),
                    next_close=clock.next_close.isoformat(), now=clock.timestamp.isoformat()),
        exposure=dict(gross=(lv - sv) / eq, net=(lv + sv) / eq, long=lv / eq, short=sv / eq,
                      n_long=sum(r["qty"] > 0 for r in rows), n_short=sum(r["qty"] < 0 for r in rows),
                      max_name=max((abs(r["weight"]) for r in rows), default=0.0)),
        sleeves=by_sleeve, sleeve_alloc=settings()["sleeves"],
        risk=dict(limits=L, peak_equity=peak, drawdown=eq / peak - 1,
                  halted=KILL_FILE.exists(), halt_reason=KILL_FILE.read_text().strip() if KILL_FILE.exists() else None),
        job=job_status(tail=0), scheduler=SCHED, model=model_info(),
    )


@app.get("/api/equity")
def equity(period: str = "1M"):
    tf = {"1D": "5Min", "1W": "1H"}.get(period, "1D")

    def fetch():
        h = tc().get_portfolio_history(GetPortfolioHistoryRequest(period=period, timeframe=tf))
        pts = [dict(time=int(t), equity=num(e), pnl=num(p)) for t, e, p in
               zip(h.timestamp or [], h.equity or [], h.profit_loss or []) if num(e)]
        return dict(base=num(h.base_value), points=pts)
    return cached(("eq", period), 30, fetch)


@app.get("/api/orders")
def orders(book: str = "paper"):
    if book == "shadow":
        logs = sorted((ROOT / "logs").glob("*/orders.csv"))
        if not logs:
            return []
        df = pd.read_csv(logs[-1])
        return [dict(time=logs[-1].parent.name, symbol=r.get("symbol"), side="BUY" if r.get("qty", 0) > 0 else "SELL",
                     qty=abs(num(r.get("qty")) or 0), filled_qty=None, price=num(r.get("price")),
                     notional=num(r.get("notional")), status=r.get("status"), type="shadow", note=r.get("note"))
                for r in df.to_dict("records")]

    def fetch():
        req = GetOrdersRequest(status=QueryOrderStatus.ALL, after=datetime.now(timezone.utc) - timedelta(days=7),
                               limit=500, direction="desc")
        return [dict(time=(o.filled_at or o.submitted_at or o.created_at).isoformat(), symbol=o.symbol,
                     side=str(o.side.value).upper(), qty=num(o.qty), filled_qty=num(o.filled_qty),
                     price=num(o.filled_avg_price), notional=(num(o.filled_avg_price) or 0) * (num(o.filled_qty) or 0),
                     status=str(o.status.value), type=f"{o.type.value}/{o.time_in_force.value}",
                     note=o.client_order_id) for o in tc().get_orders(req)]
    return cached("orders", 8, fetch)


# ---------------------------------------------------------------- research / runs / model
def model_info():
    files = sorted((ROOT / "models").glob("stock_lgbm_*.json"))
    return json.loads(files[-1].read_text()) | {"file": files[-1].stem} if files else None


@app.get("/api/research")
def research():
    reps = sorted((ROOT / "reports").glob("research_2*.md"))
    pnls = sorted((ROOT / "reports").glob("research_daily_pnl_*.csv"))
    curves = {}
    if pnls:
        df = pd.read_csv(pnls[-1], index_col=0, parse_dates=True)
        eq = (1 + df.fillna(0)).cumprod()
        eq = eq.resample("W").last().dropna(how="all")
        curves = {c: [dict(time=int(t.timestamp()), value=float(v)) for t, v in eq[c].dropna().items()]
                  for c in eq.columns}
    return dict(report=reps[-1].read_text() if reps else None, name=reps[-1].name if reps else None, curves=curves)


@app.get("/api/runs")
def runs():
    out = []
    for f in sorted((ROOT / "logs").glob("*/summary.json"), reverse=True)[:60]:
        out.append(json.loads(f.read_text()) | {"date": f.parent.name})
    return out


@app.get("/api/targets")
def targets():
    files = sorted((ROOT / "logs").glob("*/targets.csv"))
    if not files:
        return []
    s = pd.read_csv(files[-1], index_col=0)["target_usd"]
    return [dict(symbol=k, target=float(v), sleeve=sleeve(k)) for k, v in s.sort_values().items()]


# ---------------------------------------------------------------- jobs + controls
JOBS = {"run_shadow": ["run_daily.py", "--mode", "shadow"], "run_paper": ["run_daily.py", "--mode", "paper"],
        "run": ["run_daily.py"], "train": ["train_model.py", "--refresh"], "research": ["research.py"],
        "download": ["download_history.py"]}
JOB = {"proc": None, "name": None, "log": None, "started": None}


def start_job(name: str):
    if JOB["proc"] is not None and JOB["proc"].poll() is None:
        raise HTTPException(409, f"job '{JOB['name']}' is still running")
    args = JOBS[name]
    log = path("logs", "jobs", f"{datetime.now():%Y-%m-%d_%H%M%S}_{name}.log")
    f = open(log, "w")
    JOB.update(proc=subprocess.Popen([sys.executable, "-u", str(ROOT / "scripts" / args[0]), *args[1:]],
                                     stdout=f, stderr=subprocess.STDOUT, cwd=ROOT),
               name=name, log=log, started=datetime.now().isoformat())
    return job_status()


def job_status(tail: int = 200):
    p = JOB["proc"]
    running = p is not None and p.poll() is None
    lines = JOB["log"].read_text().splitlines()[-tail:] if JOB["log"] and tail else []
    return dict(name=JOB["name"], running=running, started=JOB["started"],
                exit_code=None if p is None or running else p.returncode, log=lines)


@app.post("/api/job/{name}")
def job(name: str):
    if name not in JOBS:
        raise HTTPException(404, "unknown job")
    return start_job(name)


@app.get("/api/job")
def job_get():
    return job_status()


@app.post("/api/kill")
def kill():
    KILL_FILE.write_text(f"manual kill from dashboard at {datetime.now().isoformat(timespec='seconds')}\n")
    closed = False
    if paper_positions():
        tc().close_all_positions(cancel_orders=True)
        closed = True
    _cache.clear()
    return {"halted": True, "flattened_paper": closed}


@app.post("/api/resume")
def resume():
    KILL_FILE.unlink(missing_ok=True)
    return {"halted": False}


@app.post("/api/scheduler")
def scheduler_toggle(enabled: bool):
    SCHED["enabled"] = enabled
    return SCHED


# ---------------------------------------------------------------- scheduler
SCHED ={"enabled": S["scheduler"]["enabled"], "last_run": None, "last_retrain": None, "next": None}


def _scheduler():
    cfg = S["scheduler"]
    rh, rm = map(int, cfg["run_time_et"].split(":"))
    th, tm = map(int, cfg["retrain_time_et"].split(":"))
    while True:
        try:
            now = pd.Timestamp.now(tz="America/New_York")
            nxt = now.normalize() + pd.Timedelta(hours=rh, minutes=rm)
            if now >= nxt:
                nxt += pd.Timedelta(days=1)
            while nxt.weekday() >= 5:
                nxt += pd.Timedelta(days=1)
            SCHED["next"] = nxt.isoformat()
            mins = now.hour * 60 + now.minute
            today = str(now.date())
            if (SCHED["enabled"] and now.weekday() < 5 and rh * 60 + rm <= mins < 9 * 60 + 25
                    and SCHED["last_run"] != today and not job_status(tail=0)["running"]):
                start_job("run")  # run_daily itself skips market holidays
                SCHED["last_run"] = today
            m = model_info()
            stale = m is None or (now.tz_localize(None) - pd.Timestamp(m["trained_on"])).days > 25
            if (SCHED["enabled"] and now.weekday() == cfg["retrain_weekday"] and mins >= th * 60 + tm
                    and SCHED["last_retrain"] != today and stale and not job_status(tail=0)["running"]):
                start_job("train")
                SCHED["last_retrain"] = today
        except Exception as e:  # never let the scheduler thread die
            print(f"scheduler: {e}", flush=True)
        time.sleep(30)


@app.on_event("startup")
def _start():
    threading.Thread(target=_scheduler, daemon=True).start()
