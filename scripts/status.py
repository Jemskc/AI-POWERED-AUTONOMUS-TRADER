"""Print the paper account, positions and the last run's summary."""
import _path  # noqa: F401

import pandas as pd

from trader import alpaca_io as io
from trader.config import ROOT, settings

if __name__ == "__main__":
    settings()
    tc = io.trading_client()
    a = tc.get_account()
    eq, last = float(a.equity), float(a.last_equity)
    print(f"equity ${eq:,.2f}  today {eq / last - 1:+.2%}  cash ${float(a.cash):,.2f}  "
          f"long ${float(a.long_market_value):,.0f}  short ${float(a.short_market_value):,.0f}")
    pos = tc.get_all_positions()
    if pos:
        df = pd.DataFrame([dict(symbol=p.symbol, qty=float(p.qty), value=float(p.market_value),
                                upl=float(p.unrealized_pl)) for p in pos]).sort_values("value")
        print(f"{len(df)} positions, unrealized P&L ${df.upl.sum():,.2f}")
        print(df.head(5).to_string(index=False)); print("..."); print(df.tail(5).to_string(index=False))
    if (ROOT / "KILL").exists():
        print(f"HALTED: {(ROOT / 'KILL').read_text().strip()}")
    logs = sorted((ROOT / "logs").glob("*/summary.json"))
    if logs:
        print(f"\nlast run {logs[-1].parent.name}:\n{logs[-1].read_text()}")
