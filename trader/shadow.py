"""Simulated book for shadow mode: symbol -> {qty, avg}. Fills assumed at the decision price (prev close)."""
import json
import math

from trader.config import path

SHADOW_BOOK = path("state", "shadow_positions.json")


def load_shadow_book() -> dict:
    if not SHADOW_BOOK.exists() or not SHADOW_BOOK.read_text().strip():
        return {}
    raw = json.loads(SHADOW_BOOK.read_text())
    return {s: v if isinstance(v, dict) else {"qty": float(v), "avg": None} for s, v in raw.items()}


def save_shadow_book(book: dict, new_qty: dict, px) -> None:
    out = {}
    for s, q in new_qty.items():
        old = book.get(s, {"qty": 0.0, "avg": None})
        oq, oa = old["qty"], old["avg"]
        p = float(px.get(s, math.nan))
        if oq and (oq > 0) == (q > 0) and oa is not None:
            avg = oa if abs(q) <= abs(oq) else (oa * abs(oq) + p * (abs(q) - abs(oq))) / abs(q)
        else:
            avg = p
        out[s] = {"qty": q, "avg": avg}
    SHADOW_BOOK.write_text(json.dumps(out))
