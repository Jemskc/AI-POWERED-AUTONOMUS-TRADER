"""Live crypto paper-trading engine.

Every hour, right after a candle closes, it:
  1. pulls the last 1000 closed hourly candles for 20 coins (keyless Binance public data)
  2. computes the same features as training and predicts P(up next H hours) with XGBoost + LightGBM
  3. logs every prediction, and scores older ones once their real H-hour outcome is known
  4. rebalances one of the `tranches` staggered books (confident calls only) through the paper broker
Every `mark_every_s` seconds it marks the book to live prices, logs equity and enforces risk limits.
"""
import json
import math
import sqlite3
import time
import traceback
from datetime import datetime, timezone

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
import yaml

from trader.config import ROOT
from trader.crypto import market
from trader.crypto.features import SYMBOLS, features, matrix, standardize, wide

CFG = yaml.safe_load((ROOT / "config" / "crypto.yaml").read_text())
SD = ROOT / "state" / "crypto"
SD.mkdir(parents=True, exist_ok=True)
HEART, KILL, DB = SD / "heartbeat.json", ROOT / "KILL_CRYPTO", SD / "predictions.sqlite"
BOOKS = CFG["books"]


def paths(book):
    return SD / f"broker_{book}.json", SD / f"equity_{book}.csv", SD / f"trades_{book}.csv"


def log(msg):
    print(f"{datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S}Z {msg}", flush=True)


# ------------------------------------------------------------------------------- persistence
def db():
    con = sqlite3.connect(DB)
    con.execute("""CREATE TABLE IF NOT EXISTS preds (t TEXT, symbol TEXT, p_xgb REAL, p_lgbm REAL, p_ens REAL,
                   signal INTEGER, tranche INTEGER, price REAL, ret REAL, up INTEGER, PRIMARY KEY (t, symbol))""")
    return con


def load_broker(book):
    f = paths(book)[0]
    return json.loads(f.read_text()) if f.exists() else None


def save_broker(b):
    f = paths(b["book"])[0]
    tmp = f.with_suffix(".tmp")
    tmp.write_text(json.dumps(b, indent=1))
    tmp.replace(f)


def append_csv(path, row: dict):
    new = not path.exists()
    pd.DataFrame([row]).to_csv(path, mode="a", header=new, index=False)


# ------------------------------------------------------------------------------- model
class Model:
    def __init__(self):
        meta = json.loads((ROOT / "models" / "crypto" / f"latest_H{CFG['H']}.json").read_text())
        self.meta, self.names = meta, meta["names"]
        self.mu, self.sd = np.array(meta["mu"], np.float32), np.array(meta["sd"], np.float32)
        self.lgb = lgb.Booster(model_file=str(ROOT / "models" / "crypto" / meta["lgbm"]))
        self.xgb = xgb.XGBClassifier()
        self.xgb.load_model(str(ROOT / "models" / "crypto" / meta["xgb"]))

    def predict_latest(self, df):
        """-> (decision time t, {symbol: (p_xgb, p_lgbm, p_ens)}, wide frames)."""
        W = wide(df)
        F = features(W)
        X = matrix(F, self.names)
        S = len(SYMBOLS)
        last = X[-S:]
        ok = ~np.isnan(last).any(axis=1)
        Z = standardize(last, self.mu, self.sd)
        px = self.xgb.predict_proba(Z)[:, 1]
        pl = self.lgb.predict(Z)
        out = {SYMBOLS[i]: (float(px[i]), float(pl[i]), float((px[i] + pl[i]) / 2)) for i in range(S) if ok[i]}
        return W["close"].index[-1], out, W


# ------------------------------------------------------------------------------- broker
def mark(b, tk):
    long_ = sum(p["qty"] * tk[s]["last"] for s, p in b["positions"].items() if p["qty"] > 0 and s in tk)
    short = sum(p["qty"] * tk[s]["last"] for s, p in b["positions"].items() if p["qty"] < 0 and s in tk)
    return b["cash"] + long_ + short, long_, short


def fill(b, sym, dq, tk, reason, t, tranche, p):
    """Execute dq units at the live bid/ask plus slippage, pay the fee, update cash and average price."""
    if abs(dq) < 1e-12:
        return
    q = tk[sym]
    slip = CFG["slippage_bps"] / 1e4
    price = q["ask"] * (1 + slip) if dq > 0 else q["bid"] * (1 - slip)
    notional = abs(dq) * price
    fee = notional * CFG["fee_bps"] / 1e4
    pos = b["positions"].setdefault(sym, {"qty": 0.0, "avg": 0.0})
    old = pos["qty"]
    new = old + dq
    realized = 0.0
    if old != 0 and np.sign(old) != np.sign(dq):  # reducing / closing / flipping
        closed = min(abs(dq), abs(old))
        realized = closed * (price - pos["avg"]) * np.sign(old)
    if new == 0 or np.sign(new) != np.sign(old):
        pos["avg"] = price if new != 0 else 0.0
    elif abs(new) > abs(old):
        pos["avg"] = (pos["avg"] * abs(old) + price * abs(dq)) / abs(new)
    pos["qty"] = new
    if abs(new) < 1e-12:
        b["positions"].pop(sym)
    b["cash"] -= dq * price + fee
    b["fees"] += fee
    b["realized"] += realized
    b["n_trades"] += 1
    append_csv(paths(b["book"])[2], dict(time=datetime.now(timezone.utc).isoformat(timespec="seconds"), symbol=sym,
                            side="BUY" if dq > 0 else "SELL", qty=abs(dq), price=price, notional=notional, fee=fee,
                            realized=realized, reason=reason, decision_t=str(t), tranche=tranche, p_ens=p))


def flatten(b, tk, why):
    log(f"[{b['book']}] FLATTEN: {why}")
    for s, p in list(b["positions"].items()):
        fill(b, s, -p["qty"], tk, f"flatten: {why}", "", -1, None)
    b["tranche_qty"] = [{} for _ in range(CFG["tranches"])]
    b["tranche_sig"] = [{} for _ in range(CFG["tranches"])]


# ------------------------------------------------------------------------------- engine
def new_book(name, tk):
    cap = float(CFG["start_capital"])
    b = dict(book=name, threshold=BOOKS[name].get("threshold"), kind=BOOKS[name].get("type", "tranche"),
             start_time=datetime.now(timezone.utc).isoformat(timespec="seconds"), start_capital=cap,
             cash=cap, positions={}, fees=0.0, realized=0.0, n_trades=0, peak=cap,
             day=str(datetime.now(timezone.utc).date()), day_start_equity=cap, halted=None,
             tranche_qty=[{} for _ in range(CFG["tranches"])], tranche_sig=[{} for _ in range(CFG["tranches"])],
             start_prices={s: v["last"] for s, v in tk.items()})
    save_broker(b)
    log(f"[{name}] new paper account: ${cap:,.0f} virtual USD, rules {BOOKS[name]}")
    return b


class Engine:
    def __init__(self):
        self.model = Model()
        self.con = db()
        self.books = {n: load_broker(n) for n in BOOKS}
        self.last_eq_log = 0.0
        row = self.con.execute("SELECT MAX(t) FROM preds").fetchone()
        self.last_decision = row[0] if row else None
        log(f"model trained {self.model.meta['trained_on']} (data through {self.model.meta['data_end']}), "
            f"H={CFG['H']}h tranches={CFG['tranches']} books={BOOKS}")

    # ---- every hour
    def decide(self):
        df = market.recent(SYMBOLS, hours=1000)
        t, preds, W = self.model.predict_latest(df)
        if str(t) == self.last_decision:
            return False
        tk = market.tickers(SYMBOLS)
        k = (t + pd.Timedelta(hours=1)).hour % CFG["tranches"]
        self.con.executemany("INSERT OR REPLACE INTO preds VALUES (?,?,?,?,?,?,?,?,?,?)",
                             [(str(t), s, *v, int(np.sign(v[2] - 0.5)), k, tk[s]["last"], None, None)
                              for s, v in preds.items()])
        for name in BOOKS:
            if self.books[name] is None:
                self.books[name] = new_book(name, tk)
            if BOOKS[name].get("type") == "rotation":
                self.rotate(self.books[name], t, preds, tk)
            else:
                self.rebalance(self.books[name], t, k, preds, tk)
        self.score(W)
        self.con.commit()
        self.last_decision = str(t)
        return True

    def rebalance(self, b, t, k, preds, tk):
        thr = b["threshold"]
        eq, _, _ = mark(b, tk)
        sig_k, qty_k = b["tranche_sig"][k], b["tranche_qty"][k]
        notional = eq / CFG["tranches"] / len(SYMBOLS)
        for s, (_, _, p) in preds.items():
            if p - 0.5 > thr:
                sig_k[s] = 1
            elif 0.5 - p > thr:
                sig_k[s] = -1 if CFG["allow_short"] else 0
            # not confident -> this tranche keeps its previous signal
            sig = sig_k.get(s, 0)
            qty_k[s] = sig * notional / tk[s]["last"] if sig else 0.0
        if not b["halted"] and not KILL.exists():
            for s in SYMBOLS:
                target = sum(tq.get(s, 0.0) for tq in b["tranche_qty"])
                dq = target - b["positions"].get(s, {}).get("qty", 0.0)
                if abs(dq) * tk[s]["last"] > 1.0:  # ignore dust
                    fill(b, s, dq, tk, f"tranche {k}", t, k, preds.get(s, (None,) * 3)[2])
        b["last_decision_t"] = str(t)
        save_broker(b)
        n_conf = sum(abs(v[2] - 0.5) > thr for v in preds.values())
        eq, lv, sv = mark(b, tk)
        log(f"[{b['book']}] t={t} tranche={k}: {n_conf}/{len(preds)} confident | equity ${eq:,.2f} "
            f"gross {(lv - sv) / eq:.2f}x net {(lv + sv) / eq:+.2f}x | trades {b['n_trades']} fees ${b['fees']:,.2f}")

    def rotate(self, b, t, preds, tk):
        """Long-only rotation: keep/buy the model's top-ranked coins, sell the ones it turns bearish on."""
        c = BOOKS[b["book"]]
        last = b.get("last_rotation")
        if last and (pd.Timestamp(t) - pd.Timestamp(last)) < pd.Timedelta(hours=c["every_h"]):
            return
        K, keep, ep, xp = c["K"], c["keep"], c["enter_p"], c["exit_p"]
        ranked = sorted(preds, key=lambda s: -preds[s][2])
        rank = {s: i for i, s in enumerate(ranked)}
        held = [s for s, p in b["positions"].items() if p["qty"] > 0]
        picks = {s for s in held if s not in preds or (preds[s][2] >= xp and rank[s] < keep)}
        for s in ranked[:K]:
            if len(picks) >= K:
                break
            if preds[s][2] > ep:
                picks.add(s)
        eq = mark(b, tk)[0]
        slot = eq / K
        if b["halted"] or KILL.exists():
            return
        for s in held:  # sells first, so the cash is there for the buys
            cur = b["positions"][s]["qty"]
            if s not in picks:
                why = "model turned bearish" if s in preds and preds[s][2] < xp else f"dropped out of top {keep}"
                fill(b, s, -cur, tk, f"rotation SELL: {why}", t, -1, preds.get(s, (None,) * 3)[2])
            elif cur * tk[s]["last"] - slot > 0.25 * slot:
                fill(b, s, slot / tk[s]["last"] - cur, tk, "rotation trim to 1/K", t, -1, preds[s][2])
        for s in sorted(picks, key=lambda x: rank.get(x, 99)):
            cur = b["positions"].get(s, {}).get("qty", 0.0)
            need = slot - cur * tk[s]["last"]
            if need > 0.25 * slot or cur == 0:
                spend = min(need, b["cash"] * 0.995)
                if spend > 1:
                    fill(b, s, spend / (tk[s]["ask"] * (1 + CFG["slippage_bps"] / 1e4) * (1 + CFG["fee_bps"] / 1e4)), tk,
                         f"rotation BUY: rank {rank.get(s, -1) + 1}, p={preds[s][2]:.3f}", t, -1, preds[s][2])
        b["last_rotation"] = b["last_decision_t"] = str(t)
        save_broker(b)
        eq, lv, _ = mark(b, tk)
        log(f"[{b['book']}] t={t} rotation: holding {sorted(s for s, p in b['positions'].items() if p['qty'] > 0)} "
            f"| equity ${eq:,.2f} invested {lv / eq:.0%} cash ${b['cash']:,.2f} | trades {b['n_trades']} fees ${b['fees']:,.2f}")

    def score(self, W):
        """Fill in the realized H-hour outcome for predictions whose horizon has passed."""
        H, o = CFG["H"], W["open"]
        upd = []
        for t, s in self.con.execute("SELECT t, symbol FROM preds WHERE ret IS NULL").fetchall():
            ts = pd.Timestamp(t)
            a, z = ts + pd.Timedelta(hours=1), ts + pd.Timedelta(hours=1 + H)
            if z in o.index and a in o.index and s in o.columns:
                p0, p1 = o.at[a, s], o.at[z, s]
                if np.isfinite(p0) and np.isfinite(p1):
                    r = math.log(p1 / p0)
                    upd.append((r, int(r > 0), t, s))
        self.con.executemany("UPDATE preds SET ret=?, up=? WHERE t=? AND symbol=?", upd)
        if upd:
            log(f"scored {len(upd)} predictions whose {H}h outcome is now known")

    # ---- every few seconds
    def tick(self):
        tk = market.tickers(SYMBOLS)
        resume = SD / "resume"
        if resume.exists() and not KILL.exists():
            for b in self.books.values():
                if b and b["halted"]:
                    log(f"[{b['book']}] resumed (was: {b['halted']})")
                    b["halted"] = None
                    b["peak"] = mark(b, tk)[0]
            resume.unlink()
        now = time.time()
        log_eq = now - self.last_eq_log >= 60
        beat = {}
        for name, b in self.books.items():
            if b is None:
                continue
            eq, lv, sv = mark(b, tk)
            today = str(datetime.now(timezone.utc).date())
            if today != b["day"]:
                b["day"], b["day_start_equity"] = today, eq
            b["peak"] = max(b["peak"], eq)
            if not b["halted"]:
                why = None
                if KILL.exists():
                    why = "manual KILL_CRYPTO"
                elif eq < b["day_start_equity"] * (1 - CFG["max_daily_loss"]):
                    why = f"daily loss limit ({eq / b['day_start_equity'] - 1:.2%})"
                elif eq < b["peak"] * (1 - CFG["max_drawdown"]):
                    why = f"max drawdown ({eq / b['peak'] - 1:.2%})"
                if why:
                    flatten(b, tk, why)
                    b["halted"] = why
            save_broker(b)
            if log_eq:
                bh = np.mean([tk[s]["last"] / p for s, p in b["start_prices"].items() if s in tk])
                append_csv(paths(name)[1], dict(time=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                                equity=eq, cash=b["cash"], long=lv, short=sv,
                                                n_pos=len(b["positions"]), btc=tk["BTC"]["last"],
                                                bh_equity=b["start_capital"] * bh))
            beat[name] = dict(equity=eq, halted=b["halted"])
        if log_eq:
            self.last_eq_log = now
        HEART.write_text(json.dumps(dict(time=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                         last_decision=self.last_decision, books=beat)))

    def run(self):
        while True:
            try:
                now = datetime.now(timezone.utc)
                expected = str(pd.Timestamp(now).floor("h") - pd.Timedelta(hours=1))
                if self.last_decision != expected and now.minute * 60 + now.second >= 15:
                    if not self.decide():
                        time.sleep(10)  # newest candle not published yet
                self.tick()
            except Exception as e:
                log(f"ERROR {type(e).__name__}: {e}\n{traceback.format_exc(limit=3)}")
                time.sleep(20)
            time.sleep(CFG["mark_every_s"])


if __name__ == "__main__":
    Engine().run()
