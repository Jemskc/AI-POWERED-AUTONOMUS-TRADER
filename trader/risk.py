"""Independent risk engine. It can veto or shrink anything the strategies ask for."""
import json

from trader.config import ROOT, path

KILL_FILE = ROOT / "KILL"          # create this file (touch ai-trader/KILL) to flatten and stop
STATE_FILE = path("state", "risk_state.json")


class RiskEngine:
    def __init__(self, limits: dict):
        self.L = limits
        self.state = json.loads(STATE_FILE.read_text()) if STATE_FILE.exists() else {}

    def kill_requested(self) -> bool:
        return KILL_FILE.exists()

    def check_equity(self, equity: float, last_equity: float):
        """Returns a reason string if trading must halt, else None."""
        peak = max(self.state.get("peak_equity", equity), equity)
        self.state["peak_equity"] = peak
        STATE_FILE.write_text(json.dumps(self.state))
        if last_equity and equity < last_equity * (1 - self.L["daily_loss_frac"]):
            return f"daily loss limit: equity {equity:,.0f} vs prev close {last_equity:,.0f}"
        if equity < peak * (1 - self.L["max_drawdown_frac"]):
            return f"max drawdown: equity {equity:,.0f} vs peak {peak:,.0f}"
        return None

    def halt(self, reason: str):
        KILL_FILE.write_text(reason + "\n")

    def cap_targets(self, targets: dict, equity: float) -> dict:
        """targets: symbol -> signed dollars. Applies per-name, gross and net caps."""
        cap = self.L["max_position_frac"] * equity
        t = {s: max(-cap, min(cap, v)) for s, v in targets.items()}
        gross = sum(abs(v) for v in t.values())
        if gross > self.L["max_gross_leverage"] * equity:
            k = self.L["max_gross_leverage"] * equity / gross
            t = {s: v * k for s, v in t.items()}
        net = sum(t.values())
        max_net = self.L["max_net_exposure_frac"] * equity
        if abs(net) > max_net:
            # shrink only the side that causes the excess
            side = 1 if net > 0 else -1
            side_sum = sum(v for v in t.values() if v * side > 0)
            k = (side_sum - (abs(net) - max_net) * side) / side_sum
            t = {s: v * k if v * side > 0 else v for s, v in t.items()}
        return t

    def order_ok(self, notional: float, equity: float) -> bool:
        return abs(notional) <= self.L["max_order_notional_frac"] * equity
