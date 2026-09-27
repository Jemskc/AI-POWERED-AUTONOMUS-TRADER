# AI-Powered Autonomous Trader

An ML trading system built from scratch and tested honestly on **real market data**. It uses **free, keyless data** and trades **virtual money only**.

The research asks one question: *can the model predict well enough to make money after trading costs?* Each experiment is a walk-forward test (train on the past, test on the unseen future) with realistic fees and random-score controls. The reports in [`reports/`](reports/) are kept whether the results were good or bad.

## Headline results (crypto, Binance data)

| Experiment | Result | Report |
|---|---|---|
| 6 models predicting "up in 4h?" (LogReg, LightGBM, XGBoost, LSTM, Transformer, ensemble) | XGBoost is best at **52.7%** accuracy vs a 49.4% baseline. The skill is real (10/10 test windows) but **trading every signal loses** to fees. | `crypto_model_shootout_H4_*.md` |
| Fast horizons (1m / 5m / 15m / 60m) on 1-minute data | Up to **55%** accuracy on the most confident calls, but the edge is about 0.3 bp per trade against about 12 bp in costs. Speed doesn't pay without HFT infrastructure. | `crypto_fast_horizons_*.md` |
| Futures data (funding, open interest, long/short ratios, Fear & Greed) | Modest help (open interest better in 6–7 of 9 windows). The up/down model's skill turned out to be mostly market timing. | `crypto_futures_features_*.md` |
| Long-only rotation between coins | The model's picks beat random picks, but fees wipe out the gain. | `crypto_rotation_backtest_*.md` |
| **Cross-sectional ranking, 100 coins, low-turnover, market-neutral** | **Rank-IC +0.10 in 11/11 windows; Sharpe 1.17 after 7 bp fees, +128% (2021-07 → 2026-09), worst drop −15%, positive every year. Random scores: Sharpe −2.6. The same rules on 20 coins lose.** | `crypto_xs100_*.md` |

The 100-coin result is being re-checked for **survivorship bias** (adding delisted coins) and **funding costs** on the short side before it goes live. See [`docs/`](docs/) for the original research blueprint.

## Layout
```
trader/crypto/      live system: features (shared with training), keyless Binance data, paper broker engine,
                    dashboard backend + frontend (trader/crypto/static)
research/crypto/    every experiment: downloads, walk-forward shoot-outs, portfolio backtests
scripts/            crypto_train.py, crypto_run.sh (engine keep-alive), crypto_dashboard.py, crypto_status.py
config/             crypto.yaml (live books, costs, risk limits), stock settings
trader/ + scripts/  earlier US stock/ETF system on Alpaca paper trading (built, not used: needs an API key)
reports/            dated results of every experiment
```

## Quick start (crypto, no keys needed)
```bash
python -m venv .venv && .venv/bin/pip install pandas pyarrow numpy lightgbm xgboost scikit-learn pyyaml requests fastapi uvicorn tabulate
.venv/bin/python research/crypto/download.py            # 20 coins, hourly, 2020 → now (data.binance.vision)
.venv/bin/python scripts/crypto_train.py --H 4          # train the live ensemble
scripts/crypto_run.sh                                   # live paper engine (keep it running: screen/tmux)
.venv/bin/python scripts/crypto_dashboard.py            # dashboard on http://localhost:8766/?token=...
.venv/bin/python scripts/crypto_status.py               # P&L + live prediction accuracy report
```
Research: `research/crypto/download_universe.py` then `research/crypto/xs100.py` reproduces the 100-coin result.

## Risk and honesty
- **Virtual money only.** There is no real-money mode. Nothing here is financial advice.
- **Backtests are not live results.** Expect live performance below backtest. A single night proves nothing.
- **Built-in risk limits:** daily loss and drawdown halts, a kill switch (`touch KILL_CRYPTO`), and fees plus bid/ask spread charged on every simulated fill.
- **Data:** Binance's public archive (`data.binance.vision`) and public market-data API (`data-api.binance.vision`), which work without an account. Binance's futures API is blocked from US IPs.
