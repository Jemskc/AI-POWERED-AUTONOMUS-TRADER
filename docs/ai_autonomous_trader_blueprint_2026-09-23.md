# AI Autonomous Trader: Blueprint for Building One From Scratch
*Compiled 2026-09-23*

---

## 0. TL;DR: the honest version

1. **The model is rarely the edge.** Profitable quant firms win on **data, cost control, risk management, execution and research discipline** (not fooling themselves). ML is one part of a pipeline, and on noisy daily data it's often the simplest part.
2. **Financial data has a very low signal-to-noise ratio.** A predictive correlation (IC) of **0.02–0.05** between prediction and next-period return is *good*. Anything that backtests at Sharpe > 3 on daily bars is almost certainly a bug, lookahead or overfitting.
3. **Most "AI trading bots" sold to retail lose money after fees.** Public LLM-trading showdowns have shown the same thing: frontier LLMs placing trades directly produce high variance, mostly negative returns, and heavy over-trading. **An LLM is useful as an analyst/feature generator. It is bad as the thing that directly decides positions.**
4. **Where a small team can realistically win:** crypto (24/7, fragmented venues, public on-chain data, funding-rate and basis inefficiencies, lower competition than US equities HFT), medium-frequency (minutes → days) strategies, and niche or low-capacity edges that big funds ignore because they can't deploy billions there.
5. **Build order:** data → backtester with realistic costs → dumb baselines → ML → risk → paper trading → tiny live capital → scale. **Never skip paper trading or the go/no-go gates.**

---

## 1. The landscape: who builds autonomous traders

### 1.1 Institutional quant / systematic firms

| Firm | Style | What's known publicly about their approach |
|---|---|---|
| **Renaissance Technologies (Medallion)** | Short-horizon statistical arbitrage across many markets | Hires scientists, not finance people. Huge clean-data effort from day one. Thousands of small, weak signals combined. High leverage on low-vol, high-Sharpe returns. Medallion is closed to outsiders, ~5/44 fees, still widely reported as ~60%+ gross/yr over decades. Their key trait was relentless data cleaning plus *one* unified model instead of siloed strategies. |
| **Two Sigma** | ML/data-driven, equities plus macro | Big alt-data ingestion, large distributed compute, research platform engineering. |
| **D.E. Shaw** | Hybrid systematic and discretionary | Pioneer of computational stat arb (1988). |
| **Citadel / Citadel Securities** | Multi-manager hedge fund plus market maker | The hedge fund is a "pod shop". Citadel Securities is a massive market maker that earns the spread on retail flow (payment for order flow) and is heavily automated. |
| **Jane Street, HRT, Jump, Tower, Virtu, XTX, Optiver, IMC** | Market making / HFT / prop | Earn the bid-ask spread plus short-horizon prediction. XTX is known to be very ML-heavy, with a big GPU cluster. Latency plus inventory risk management. |
| **Man AHL, Winton, AQR, Aspect** | Trend-following (CTA) plus factor investing | Mostly transparent academic signals: momentum, carry, value. Money is made by diversifying across 100s of markets plus risk control. |
| **Bridgewater** | Systematic global macro | Economic "machine" rules, risk parity (All Weather). |
| **WorldQuant** | Alpha factory | Crowdsources millions of small formulaic "alphas" (see *101 Formulaic Alphas*), combines them. |
| **Millennium, Balyasny, Point72** | Multi-manager pods | Many independent teams with tight drawdown stops (roughly 5% → cut, 7.5% → fired). The platform takes a cut, and investors pay "pass-through" costs. |

**Crypto-native:** Wintermute, GSR, Amber, Cumberland (market making); Jump Crypto; many funding-rate/basis arb funds; MEV searchers (on-chain arbitrage and liquidations, which fits your DeFi background).

### 1.2 Open-source / retail frameworks (what you can study or use)

| Project | What it is | Use it for |
|---|---|---|
| **Freqtrade** | Python crypto bot with backtesting, hyperopt and **FreqAI** (ML module) | Fastest way to a working crypto bot. Good reference architecture. |
| **Hummingbot** | Open-source market-making / arb bot, many CEX/DEX connectors | Market-making and cross-exchange arb. |
| **NautilusTrader** | Rust-core, Python API, event-driven, institutional-grade backtest plus live with *identical code* | **Recommended core engine** if you want production quality. |
| **Jesse** | Clean Python crypto framework | Simple strategy research. |
| **vectorbt / vectorbt pro** | Ultra-fast vectorized backtesting (numpy/numba) | Rapid research, parameter sweeps. |
| **Backtrader, Zipline-reloaded, LEAN (QuantConnect)** | Classic backtesters. LEAN is C# with a big data library. | Equities research. |
| **Qlib (Microsoft)** | ML quant research platform: data, alpha models (LightGBM, LSTM, Transformer, etc.), backtest | **Best reference for an ML alpha pipeline.** |
| **FinRL** | Deep RL for trading (PPO, SAC, DDPG) | Learning RL. Treat its results skeptically. |
| **TradingAgents, FinGPT, FinRobot, FinMem** | LLM multi-agent trading frameworks (analyst, researcher, trader and risk-manager roles) | Design ideas for the LLM layer. Their papers' backtests are short and optimistic. |
| **ccxt** | Unified API for 100+ crypto exchanges | Order routing and data. |

### 1.3 LLM-agent traders (the 2024–2026 wave)

- **Architecture pattern:** several LLM "roles" (fundamental analyst, news/sentiment analyst, technical analyst, bull researcher, bear researcher, trader, risk manager) debate, then output a decision. Examples: TradingAgents, FinCon, FinMem (adds layered memory).
- **Reality:** LLMs are slow (seconds), expensive, non-deterministic, and they hallucinate numbers. They can't be backtested reliably because they may have **memorized the future** from their training data (lookahead contamination). Any backtest before the model's knowledge cutoff is suspect.
- **Where LLMs genuinely help:**
  - Turning unstructured text (news, filings, earnings calls, governance proposals, Discord/Twitter, exploit reports) into **structured numeric features** such as sentiment, event type and surprise.
  - Research assistant work: writing and debugging feature code, summarizing papers.
  - Monitoring and anomaly explanation: "why did PnL drop?"
  - Event detection: listings, hacks, depegs, regulatory news. Detection has to be *fast*, so use small local models.
- **Rule for our build:** the LLM produces **features and alerts**. A deterministic, backtested model plus a risk engine decides and sizes trades.

---

## 2. How hedge funds actually make money

### 2.1 The business model (fees)
- **"2 and 20":** 2% management fee on AUM plus 20% of profits above a high-water mark. Averages today are closer to ~1.3–1.5% / 16–18%.
- **Pass-through fees (multi-managers):** investors pay all costs, including trader pay. Often 5–10% of AUM per year, but net returns stay steady, so investors accept it.
- **Market makers / prop firms** have no outside investors. They trade their own capital and keep everything. This is the most profitable model when it works.
- The management fee means a big fund earns money even when flat. That is why capacity (how much money a strategy can absorb) matters so much to them and so little to you.

### 2.2 Where the returns (alpha) come from: 6 sources

| Source | Explanation | Examples |
|---|---|---|
| **1. Risk premia** | Getting paid to hold risk others don't want | Carry, value, momentum, selling volatility, the crypto funding rate (longs pay shorts during bull markets) |
| **2. Liquidity provision** | Getting paid to be the counterparty when others need to trade now | Market making, providing liquidity during liquidation cascades, index rebalancing |
| **3. Behavioral** | Humans under- or over-react | Post-earnings drift, momentum, overreaction mean-reversion |
| **4. Structural / flow** | Forced, predictable, price-insensitive trading | Index rebalances, option-hedging flows (gamma), token unlocks, ETF creations, stablecoin mints, liquidations |
| **5. Informational** | Knowing sooner or better | Alt data (satellite, card spending, web traffic), faster NLP on news, on-chain analytics |
| **6. Speed / technology** | Being first | HFT latency arb, cross-venue arb, MEV |

A durable strategy should have a clear answer to **"who is on the other side of my trade, and why are they willing to lose to me?"** If you can't answer that, the backtest is probably noise.

### 2.3 Strategy families in detail

1. **Statistical arbitrage (stat arb):** trade a large cross-section (e.g., 500–3000 stocks or 100+ crypto perps) market-neutral. Predict *relative* returns over the next minutes to days. Go long the top-ranked and short the bottom-ranked, and neutralize to market, sector and factor exposure. Profit comes from many tiny edges across many bets (the fundamental law: IR ≈ IC × √breadth). **This is the most ML-friendly strategy family.**
2. **Pairs / cointegration:** a simple ancestor of stat arb. Two related assets, trade their spread's mean-reversion (e.g., ETH/stETH, BTC spot/perp).
3. **Trend following / CTA:** time-series momentum across futures (indices, rates, FX, commodities, crypto). Low Sharpe (~0.5–1) but crisis alpha, and it works across 100 years of data. Mostly simple rules plus volatility targeting.
4. **Market making:** quote bid and ask, earn the spread, manage inventory (Avellaneda–Stoikov model), avoid adverse selection (being picked off by informed traders). ML predicts short-term price moves and toxicity to skew quotes.
5. **Basis / carry / funding arb (crypto favorite):** long spot plus short perp collects the funding rate at near-zero delta. Cash-and-carry uses dated futures. Cross-exchange funding differences. Returns are real but compressed, and the risks are exchange default (FTX), liquidation of one leg, and depegs.
6. **Cross-exchange / triangular arbitrage:** price discrepancies between venues. Competitive, and needs co-location plus inventory on every venue.
7. **Event-driven:** merger arb, earnings, index inclusion. Crypto versions: exchange listings, unlocks, governance votes, hacks/exploits.
8. **Volatility arbitrage:** implied vs realized vol, skew, dispersion. Deribit for crypto options.
9. **Global macro:** rates, FX and commodities based on economic views, systematic or discretionary.
10. **Fundamental long/short (quantamental):** ML on fundamentals plus alt data. Slow horizon.
11. **On-chain / MEV:** DEX-CEX arb, liquidations, backruns. Very competitive, but your DeFi skills are directly relevant.

### 2.4 Where the edge comes from in practice (the unglamorous part)
- **Transaction costs decide everything.** A 5 bps edge with 6 bps of costs loses money. Pro firms obsess over fees (maker rebates, VIP tiers), slippage, and market impact.
- **Portfolio construction and risk:** combining 50 weak, uncorrelated signals beats one "great" signal.
- **Capacity:** small money can harvest edges that big funds can't use. **This is your structural advantage.**
- **Research hygiene:** they kill most ideas. Out-of-sample discipline is the moat.

---

## 3. How much ML do they actually use?

| Area | ML intensity | Typical methods |
|---|---|---|
| Alpha / return prediction (mid-frequency) | **Medium–high** | Ridge/Lasso, **gradient-boosted trees (LightGBM/XGBoost/CatBoost)**, which win most tabular tasks, and small neural nets. Ensembles. |
| HFT / microstructure prediction | **High** | Order-book features plus GBMs, CNN/LSTM/Transformers on the limit order book (e.g., DeepLOB), online learning. Latency-optimized (C++/FPGA). |
| NLP / text | **High (growing fast)** | FinBERT → now LLM embeddings plus fine-tuned small LLMs for sentiment/event extraction. |
| Alt data processing | High | CV on satellite images, entity resolution, anomaly detection. |
| Execution | Medium | Supervised impact/slippage models, **RL for order placement** (one of the few places RL genuinely works in finance). |
| Portfolio construction | Low–medium | Mostly convex optimization (mean-variance, risk parity) plus shrinkage covariance (Ledoit-Wolf), HRP. |
| Risk | Low–medium | Factor models (Barra-style), PCA, stress tests. Mostly statistics, not deep learning. |
| Trend following / CTA | **Low** | Rules plus volatility scaling. ML adds little. |
| Direct end-to-end RL "agent trades everything" | **Very low in production** | Mostly academic. Overfits, unstable, hard to validate. |
| LLM "decides trades" | **Near zero in serious production** | Used for research and features, not direct execution. |

**Why is ML less dominant than people think?** Markets are **non-stationary** (patterns decay because others trade them away), **adversarial**, and **low signal-to-noise**, and there's **little independent data** (10 years of daily data is only ~2,500 points per asset). Deep nets need lots of data with a stable distribution, and finance gives neither at low frequency. At high frequency (tick data, millions of samples) deep learning shines, which is why HFT and market makers are the most ML-heavy.

**The key academic reference:** Gu, Kelly & Xiu (2020), *"Empirical Asset Pricing via Machine Learning"*. Trees and neural nets roughly doubled the Sharpe of linear models for predicting stock returns, driven by nonlinear interactions of momentum, liquidity and volatility features.

---

## 4. System architecture of an autonomous trader

```
┌──────────────────────────────────────────────────────────────────────────┐
│                          DATA LAYER                                       │
│  Market data (OHLCV, trades, L2 order book, funding, OI, liquidations)   │
│  On-chain (DEX flows, stablecoin mints, whale transfers, TVL)            │
│  Text (news, X/Twitter, governance, exchange announcements)              │
│  → Raw store (Parquet on disk / S3)  → cleaned, point-in-time store      │
└──────────────┬───────────────────────────────────────────────────────────┘
               ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  FEATURE LAYER (feature store): same code for backtest and live           │
│  Price/vol features · microstructure · cross-sectional ranks ·           │
│  funding/basis · on-chain · LLM-extracted text features                  │
└──────────────┬───────────────────────────────────────────────────────────┘
               ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  ALPHA / SIGNAL LAYER                                                     │
│  Many models → forecasts of return over horizon h (and uncertainty)       │
│  Baselines (momentum, mean-rev) + LightGBM + optional NN, ensembled      │
│  Meta-model: "should I trust this signal now?" (meta-labeling, regime)   │
└──────────────┬───────────────────────────────────────────────────────────┘
               ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  PORTFOLIO CONSTRUCTION                                                   │
│  Forecasts + covariance + costs → target positions                       │
│  Vol targeting · neutralization · turnover penalty · fractional Kelly    │
└──────────────┬───────────────────────────────────────────────────────────┘
               ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  RISK ENGINE (independent, has VETO power)                                │
│  Max position/notional/leverage · daily loss limit · max drawdown ·      │
│  exposure caps · exchange-health checks · KILL SWITCH                    │
└──────────────┬───────────────────────────────────────────────────────────┘
               ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  EXECUTION / OMS                                                          │
│  Target − current = orders · maker/taker logic · TWAP slicing ·          │
│  retries, idempotency, rate limits · reconcile with exchange state       │
└──────────────┬───────────────────────────────────────────────────────────┘
               ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  MONITORING & LEARNING LOOP                                               │
│  PnL attribution · live vs backtest drift · slippage tracking ·          │
│  alerts (Telegram/Discord) · model decay detection → retrain             │
└──────────────────────────────────────────────────────────────────────────┘
```

**Golden rule:** backtest and live must share **the same feature code and the same strategy code** (event-driven engines like NautilusTrader enforce this). Train/serve skew is the #1 cause of "worked in backtest, fails live".

---

## 5. The ML pipeline in detail

### 5.1 What to predict (labels / targets)
- **Forward return** over horizon h: `r = log(P[t+h]/P[t])`. Better: **volatility-normalized** (`r / σ`) or **cross-sectionally ranked** (removes market moves; this is the stat-arb target).
- **Triple-barrier label** (López de Prado): +1 if the take-profit is hit first, −1 if the stop-loss is hit first, 0 if time runs out. Matches how you actually trade.
- **Meta-labeling:** a primary model (even a simple rule) chooses the *side*, and a secondary ML model predicts *whether that trade will be profitable*, which sets the size. Very effective and cuts false positives.
- Pick h according to your costs: the shorter the horizon, the more trades and the more costs eat you.

### 5.2 Features (by category)

| Category | Examples |
|---|---|
| Returns / momentum | returns over 1, 4, 12, 24, 72, 168 bars. Skip the most recent bar for reversal. Distance from moving averages. |
| Volatility | realized vol (several windows), Parkinson/Garman-Klass range vol, vol-of-vol, vol ratio short/long |
| Volume / liquidity | volume z-score, dollar volume, Amihud illiquidity, taker buy ratio |
| Microstructure (intraday) | order-book imbalance, spread, depth, trade-flow imbalance (signed volume), VPIN, queue sizes |
| Crypto derivatives | **funding rate** (level and change), **open interest** change, perp–spot basis, long/short ratio, liquidation volume |
| Cross-sectional | rank of each feature across the universe, return relative to BTC/sector, beta, residual momentum |
| On-chain | exchange net flows, stablecoin supply change, whale transfers, DEX volume, active addresses |
| Calendar / regime | hour-of-day, day-of-week, funding timestamps, market-wide vol regime |
| Text (LLM-derived) | sentiment score, event type (hack, listing, unlock, regulation), novelty, source credibility |

**Tips:** make features **stationary** (returns, z-scores, ranks, not raw prices). Consider **fractional differencing** to keep memory while staying stationary. Everything must be **point-in-time**: only use data that was actually available at that timestamp.

### 5.3 Models (recommended progression)
1. **Baselines first:** momentum/mean-reversion rules, then linear/ridge regression on 10 features. If ML can't beat these out-of-sample after costs, stop there.
2. **LightGBM** (workhorse): shallow trees (num_leaves 15–63), strong regularization, low learning rate, feature subsampling, early stopping on the *validation fold*.
3. **Ensembles:** average several GBMs trained on different seeds/windows, plus a linear model. Averaging is the most robust trick in finance.
4. **Neural nets** (only with lots of data, e.g., intraday on many assets): MLP on tabular features, then temporal models (LSTM, TCN, Transformer / Temporal Fusion Transformer, PatchTST). DeepLOB-style CNNs on order-book snapshots for HFT.
5. **Time-series foundation models** (Chronos, TimesFM, Moirai, etc.): useful for volatility and volume forecasts. Weak for returns.
6. **RL:** use for **execution** (optimal order placement) or market-making quote skew, *not* for end-to-end portfolio decisions until everything else works.
7. **Online learning / periodic retraining:** retrain on a rolling window (e.g., weekly) because edges decay.

### 5.4 Validation: where most people fail
- **Never use random k-fold** on time series.
- **Walk-forward validation:** train on the past, test on the next block, roll forward.
- **Purging + embargo:** if a label looks h bars ahead, drop training samples whose label window overlaps the test period (purge) and add a gap after the test (embargo).
- **Combinatorial Purged CV (CPCV):** gives a *distribution* of backtest Sharpe ratios instead of a single lucky path.
- **Deflated Sharpe Ratio / Probability of Backtest Overfitting (PBO):** corrects for how many strategies you tried. **Log every experiment**, since the number of trials matters.
- **Hold out a final untouched test period** (e.g., the last 6–12 months) and look at it **once**.
- **Metrics:** IC / Rank-IC (Spearman of prediction vs realized) per period and its t-stat, hit rate, **Sharpe after costs**, max drawdown, turnover, Calmar, PnL per trade in bps vs cost per trade in bps.

### 5.5 Backtesting pitfalls checklist
- [ ] Lookahead bias (using the close of bar t to trade at the close of bar t → must trade at the **open of t+1** or later)
- [ ] Survivorship bias (include delisted coins/stocks)
- [ ] Fees + spread + slippage + market impact + funding payments included
- [ ] Realistic fills (limit orders don't always fill; in a backtest they fill only when price trades *through* them)
- [ ] Borrow/short availability, leverage limits, min order sizes
- [ ] Data errors (bad ticks, exchange outages, timestamp units ms vs µs)
- [ ] Overfitting via too many parameter tries
- [ ] Regime coverage (does the test include a crash? 2020-03, 2022 LUNA/FTX, 2025-10 liquidation cascade)

---

## 6. Portfolio construction and risk management

### 6.1 Position sizing
- **Volatility targeting:** `position = (target_vol / forecast_vol) × signal`. This is the single most important sizing rule. It stabilizes risk across regimes.
- **Kelly criterion:** `f* = μ / σ²`. Always use **fractional Kelly (¼–½)**, because estimation error in μ makes full Kelly ruinous.
- **Cross-sectional:** z-score or rank the forecasts → dollar-neutral weights → scale to target gross leverage.
- **Mean-variance with costs:** maximize `w·α − λ·wᵀΣw − κ·|w − w_prev|` (cvxpy). Use a shrinkage covariance (Ledoit-Wolf) or **Hierarchical Risk Parity** for robustness.

### 6.2 Risk limits (hard-coded, independent from the strategy)
- Max position per asset (e.g., 10% of equity), max gross leverage (e.g., 2×), max net exposure.
- **Daily loss limit** (e.g., −2% of equity → flatten and halt until a human reviews).
- **Max drawdown** from peak (e.g., −10% → kill switch).
- Stale data guard: if the last tick is older than X seconds, no new orders.
- Exchange health: API errors, withdrawal suspensions, abnormal spreads → reduce exposure.
- Fat-finger checks: order notional and price within X% of mid.
- **Counterparty risk:** spread capital across exchanges and keep only the needed margin on each (lesson from FTX).
- Signed-off config: limits live in a config that the strategy code **cannot modify**.

### 6.3 Execution
- Costs: **fees** (maker vs taker, e.g., 0.02% vs 0.05% on perps), **spread**, **slippage**, **impact** (≈ σ·√(Q/ADV), the square-root law).
- Prefer **post-only limit orders** when the signal horizon allows it. Use taker orders only when urgency beats cost.
- Slice large orders (TWAP/VWAP, Almgren–Chriss for optimal trade-off).
- **Idempotent order IDs**, handle partial fills, reconcile positions against the exchange every loop, respect rate limits, and reconnect websockets.
- Track **implementation shortfall** (decision price vs fill price) live. That tells you if the backtest cost model was honest.

---

## 7. Recommended tech stack (free / self-hosted, fits your setup)

| Layer | Choice | Notes |
|---|---|---|
| Language | **Python** (research + orchestration), **Rust** (only if you go HFT) | NautilusTrader gives Rust speed with a Python API |
| Data frames | **polars** (fast), pandas | |
| Storage | **Parquet files + DuckDB** to start; ClickHouse/TimescaleDB for ticks at scale | |
| Exchange connectivity | **ccxt / ccxt.pro** (websockets) | Hyperliquid / Binance / Bybit / OKX |
| Backtesting | **vectorbt** (fast research) + **NautilusTrader** (realistic event-driven + live) | or Freqtrade for a quick all-in-one |
| ML | **LightGBM**, scikit-learn, PyTorch | |
| Experiment tracking | MLflow or Weights & Biases (free tier) | Log every trial (needed for deflated Sharpe) |
| Optimization | cvxpy, PyPortfolioOpt, riskfolio-lib | |
| LLM layer | **Local Qwen via vLLM** (you already run this), structured JSON output | Text → features, alerts. Never direct order placement. |
| Scheduling | systemd / cron / Prefect | |
| Monitoring | Prometheus + Grafana, Telegram/Discord bot alerts | |
| Secrets | env vars / vault. **Exchange API keys with trade-only permission, NO withdrawal, IP-whitelisted** | |

### 7.1 Data sources

| Data | Free / cheap | Paid (better) |
|---|---|---|
| Crypto OHLCV + trades | **data.binance.vision** (bulk historical klines/trades/funding, free), ccxt REST, Hyperliquid API | Tardis.dev (tick + L2 book, all venues), Kaiko, CoinAPI |
| Funding / OI / liquidations | exchange APIs (Binance/Bybit/OKX), Coinalyze | CoinGlass API, Laevitas |
| On-chain | Dune (SQL), Flipside, your own node / RPC + The Graph, DefiLlama API | Glassnode, Nansen, Arkham, Artemis |
| US equities | Alpaca (free IEX data + paper trading), yfinance (unreliable, research only), SEC EDGAR | Polygon.io, Databento, Norgate (survivorship-free) |
| Macro | FRED | |
| News / text | RSS, exchange announcement pages, GDELT, Reddit API | CryptoPanic, Benzinga, RavenPack |

---

## 8. Build roadmap (phases with go/no-go gates)

### Phase 0: decisions (day 1)
- Market: **Recommend crypto perps (Binance/Bybit/Hyperliquid)** first: free data, 24/7, paper/testnet available, low minimums. Equities via Alpaca later.
- Frequency: **1h bars, holding hours to days.** That is low infrastructure burden and costs are manageable. Avoid HFT until later.
- Universe: top 30–50 liquid perps by volume (a cross-section enables stat-arb-style ML).

### Phase 1: data foundation (week 1–2)
- Downloader for klines + funding + OI → Parquet, partitioned by symbol/month.
- Data quality checks: gaps, duplicates, outliers, timestamp units.
- Universe builder that is **point-in-time** (only coins that existed and were liquid at that date).
- ✅ Gate: can reproduce any symbol's history with no gaps, plus a test suite.

### Phase 2: research backtester (week 2–3)
- Vectorized backtester: signals → positions (lagged 1 bar) → PnL − fees − slippage − funding.
- Implement **baselines**: time-series momentum, cross-sectional momentum, short-term reversal, funding-rate carry.
- ✅ Gate: baselines give sane, known-in-literature results (momentum works in trends, suffers in chop). Costs are verified.

### Phase 3: ML alpha (week 3–6)
- Feature library (section 5.2), labels (vol-normalized, cross-sectionally ranked forward returns).
- Walk-forward LightGBM with purge/embargo. Track IC, Rank-IC and Sharpe after costs. Ensemble.
- Feature importance (SHAP / permutation), with a stability check across folds.
- Meta-labeling on top of the best baseline.
- ✅ Gate: out-of-sample Rank-IC > 0.02 with a t-stat > 2, **net Sharpe > 1 after realistic costs** on the held-out period, and it beats baselines. Deflated Sharpe is still significant.

### Phase 4: portfolio + risk (week 6–7)
- Vol targeting, dollar-neutral construction, turnover penalty, all risk limits from section 6.2.
- ✅ Gate: stress tests on crash periods; the drawdown limits trigger correctly.

### Phase 5: event-driven engine + paper trading (week 7–10)
- Port the strategy to NautilusTrader (or a custom event loop) **reusing the exact feature code**.
- Run on the **exchange testnet / paper** for ≥ 4 weeks. Log every decision, order and fill.
- Compare live signals with the backtest replaying the same period. They must match.
- ✅ Gate: paper performance is within the expected range. Zero unhandled errors for 2 weeks. The kill switch has been tested.

### Phase 6: LLM layer (in parallel from week 6)
- Local Qwen via vLLM reads news/announcements → JSON `{asset, event_type, sentiment[-1,1], confidence, novelty}`.
- Store it as timestamped features. Evaluate them like any feature (IC out-of-sample, only on data after the model's knowledge cutoff).
- Hard-coded event overrides: e.g., "exchange hack / stablecoin depeg" → the risk engine reduces exposure.
- ✅ Gate: the text features add IC incrementally over price features.

### Phase 7: live with tiny capital (month 3+)
- Start with capital you can fully lose (e.g., $500–$2,000). Same code, real keys (trade-only, no withdrawal).
- Monitor slippage vs the model. Scale **only** after 2–3 months of live results match the backtest.
- Add strategies (funding carry, pairs, market making) as **uncorrelated** sleeves. Diversification is how funds get smooth returns.

### Phase 8: continuous improvement
- Weekly retrain, decay monitoring (rolling IC), new features, new venues, alert on drift.

---

## 9. Starter code (minimal, runnable skeleton)

### 9.1 Repo layout
```
ai-trader/
├── config/            # risk_limits.yaml (read-only to strategy), universe.yaml
├── data/              # parquet: raw/, clean/
├── trader/
│   ├── data/          # downloaders, cleaners, point-in-time universe
│   ├── features/      # feature functions (shared by backtest & live!)
│   ├── labels/        # forward returns, triple-barrier
│   ├── models/        # lgbm training, walk-forward CV, ensembles
│   ├── portfolio/     # sizing, vol targeting, optimizer
│   ├── risk/          # limits, kill switch
│   ├── execution/     # ccxt order manager, reconciliation
│   ├── llm/           # vLLM client, text→feature extraction
│   └── live/          # main loop
├── research/          # notebooks
├── tests/
└── mlruns/            # experiment tracking
```

### 9.2 Download free Binance futures data
```python
# trader/data/binance_download.py
import io, zipfile, requests, pandas as pd
from pathlib import Path

BASE = "https://data.binance.vision/data/futures/um/monthly/klines"
COLS = ["open_time","open","high","low","close","volume","close_time",
        "quote_volume","trades","taker_buy_base","taker_buy_quote","ignore"]

def download_klines(symbol: str, interval: str, months: list[str], out: Path):
    frames = []
    for m in months:  # e.g. "2024-01"
        url = f"{BASE}/{symbol}/{interval}/{symbol}-{interval}-{m}.zip"
        r = requests.get(url, timeout=60)
        if r.status_code != 200:
            continue
        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            with z.open(z.namelist()[0]) as f:
                df = pd.read_csv(f, header=None, names=COLS)
        # some files have a header row, drop it
        df = df[pd.to_numeric(df["open_time"], errors="coerce").notna()]
        frames.append(df)
    if not frames:
        return None
    df = pd.concat(frames).astype({c: float for c in COLS if c not in ("open_time","close_time")})
    ts = df["open_time"].astype("int64")
    unit = "us" if ts.max() > 10**14 else "ms"   # Binance moved some datasets to microseconds
    df.index = pd.to_datetime(ts, unit=unit, utc=True)
    df = df[~df.index.duplicated()].sort_index()
    out.mkdir(parents=True, exist_ok=True)
    df[["open","high","low","close","volume","quote_volume","taker_buy_base"]].to_parquet(
        out / f"{symbol}_{interval}.parquet")
    return df
```

### 9.3 Features and labels (shared by backtest and live)
```python
# trader/features/core.py
import numpy as np, pandas as pd

def make_features(df: pd.DataFrame) -> pd.DataFrame:
    """df: one symbol, columns open/high/low/close/volume/taker_buy_base. Uses only past data."""
    f = pd.DataFrame(index=df.index)
    lr = np.log(df["close"]).diff()
    for n in (1, 4, 12, 24, 72, 168):
        f[f"ret_{n}"] = np.log(df["close"] / df["close"].shift(n))
    for n in (24, 168):
        f[f"vol_{n}"] = lr.rolling(n).std()
    f["vol_ratio"] = f["vol_24"] / f["vol_168"]
    hl = np.log(df["high"] / df["low"])
    f["parkinson_24"] = np.sqrt((hl**2).rolling(24).mean() / (4*np.log(2)))
    f["volume_z"] = (df["volume"] - df["volume"].rolling(168).mean()) / df["volume"].rolling(168).std()
    f["taker_ratio"] = (df["taker_buy_base"] / df["volume"]).rolling(12).mean()
    f["dist_ma_72"] = df["close"] / df["close"].rolling(72).mean() - 1
    f["hour"] = df.index.hour
    # normalize return features by vol so they're comparable across assets/regimes
    for n in (1, 4, 12, 24, 72, 168):
        f[f"ret_{n}_z"] = f[f"ret_{n}"] / (f["vol_168"] * np.sqrt(n))
    return f

def make_label(df: pd.DataFrame, h: int = 24) -> pd.Series:
    """Vol-normalized forward return from next bar's open to open h bars later (no lookahead)."""
    entry = df["open"].shift(-1)
    exit_ = df["open"].shift(-1 - h)
    vol = np.log(df["close"]).diff().rolling(168).std() * np.sqrt(h)
    return np.log(exit_ / entry) / vol
```

Build a panel: loop over symbols, add a `symbol` column, concat. Then **cross-sectionally rank** the label per timestamp:
`panel["y"] = panel.groupby(level=0)["label"].rank(pct=True) - 0.5`.

### 9.4 Walk-forward LightGBM with purge + embargo
```python
# trader/models/walkforward.py
import lightgbm as lgb, numpy as np, pandas as pd
from scipy.stats import spearmanr

PARAMS = dict(objective="regression", learning_rate=0.02, num_leaves=31,
              min_data_in_leaf=500, feature_fraction=0.7, bagging_fraction=0.7,
              bagging_freq=1, lambda_l2=10.0, verbose=-1)

def walk_forward(panel: pd.DataFrame, feats: list[str], target="y",
                 train_days=365, test_days=30, horizon_bars=24, embargo_bars=24):
    """panel: MultiIndex-free, DatetimeIndex (UTC) + 'symbol' column, sorted by time."""
    times = panel.index.unique().sort_values()
    start = times[0] + pd.Timedelta(days=train_days)
    gap = pd.Timedelta(hours=horizon_bars + embargo_bars)  # purge label overlap + embargo
    preds, t = [], start
    while t < times[-1]:
        t_end = t + pd.Timedelta(days=test_days)
        tr = panel[(panel.index >= t - pd.Timedelta(days=train_days)) & (panel.index < t - gap)]
        te = panel[(panel.index >= t) & (panel.index < t_end)]
        tr = tr.dropna(subset=feats + [target]); te = te.dropna(subset=feats)
        if len(tr) and len(te):
            # inner validation = last 15% of train period (time-ordered) for early stopping
            cut = tr.index.unique().sort_values()[int(0.85 * tr.index.nunique())]
            fit, val = tr[tr.index < cut - gap], tr[tr.index >= cut]
            m = lgb.train(PARAMS, lgb.Dataset(fit[feats], fit[target]),
                          num_boost_round=2000,
                          valid_sets=[lgb.Dataset(val[feats], val[target])],
                          callbacks=[lgb.early_stopping(100, verbose=False)])
            out = te[["symbol"]].copy()
            out["pred"] = m.predict(te[feats], num_iteration=m.best_iteration)
            preds.append(out)
        t = t_end
    return pd.concat(preds)

def rank_ic(preds: pd.DataFrame, panel: pd.DataFrame, target="y") -> pd.Series:
    j = preds.set_index("symbol", append=True).join(
        panel.set_index("symbol", append=True)[[target]], how="inner").dropna()
    return j.groupby(level=0).apply(
        lambda g: spearmanr(g["pred"], g[target]).statistic if len(g) > 5 else np.nan)
```
(Look at the mean IC and `IC.mean()/IC.std()*sqrt(n)` as the t-stat. Plot the rolling IC to see decay.)

### 9.5 Cost-aware portfolio backtest
```python
# trader/portfolio/backtest.py
import numpy as np, pandas as pd

def backtest_xs(preds: pd.DataFrame, returns: pd.DataFrame, fee_bps=5.0, slip_bps=3.0,
                gross_leverage=1.0, rebalance_every=24):
    """preds: index=time, cols=symbols (forecast). returns: same shape, open-to-open return
    of the NEXT bar (i.e. realized after the decision). Dollar-neutral long/short."""
    z = preds.sub(preds.mean(axis=1), axis=0)
    w = z.div(z.abs().sum(axis=1), axis=0) * gross_leverage          # sum|w| = leverage
    w = w.iloc[::rebalance_every].reindex(preds.index).ffill()        # rebalance cadence
    turnover = w.diff().abs().sum(axis=1).fillna(w.abs().sum(axis=1))
    gross = (w * returns).sum(axis=1)
    cost = turnover * (fee_bps + slip_bps) / 1e4
    net = gross - cost
    ann = np.sqrt(24 * 365)  # hourly bars, crypto trades 24/7
    eq = (1 + net).cumprod()
    return {
        "sharpe_net": net.mean() / net.std() * ann,
        "sharpe_gross": gross.mean() / gross.std() * ann,
        "ann_return": net.mean() * 24 * 365,
        "max_dd": (eq / eq.cummax() - 1).min(),
        "avg_turnover_per_bar": turnover.mean(),
        "equity": eq,
    }
```
(Add funding payments for perps: `− w × funding_rate` at each funding timestamp.)

### 9.6 Risk engine with kill switch
```python
# trader/risk/engine.py
from dataclasses import dataclass
import time, yaml

@dataclass(frozen=True)
class Limits:
    max_pos_frac: float; max_gross_lev: float; daily_loss_frac: float
    max_dd_frac: float; max_data_age_s: float; max_px_dev_frac: float

class RiskEngine:
    def __init__(self, path="config/risk_limits.yaml"):
        self.L = Limits(**yaml.safe_load(open(path)))
        self.halted, self.peak, self.day_start_eq = False, None, None

    def update_equity(self, eq: float, new_day: bool):
        self.peak = max(self.peak or eq, eq)
        if new_day or self.day_start_eq is None:
            self.day_start_eq = eq
        if eq < self.day_start_eq * (1 - self.L.daily_loss_frac): self.halt("daily loss limit")
        if eq < self.peak * (1 - self.L.max_dd_frac): self.halt("max drawdown")

    def halt(self, why: str):
        self.halted = True
        print(f"[KILL SWITCH] {why}")  # also send Telegram alert + flatten positions

    def check_targets(self, targets: dict[str, float], equity: float,
                      last_tick_ts: float, mids: dict[str, float], limit_px: dict[str, float]):
        if self.halted: return {}
        if time.time() - last_tick_ts > self.L.max_data_age_s: return None  # stale: do nothing
        cap = self.L.max_pos_frac * equity
        t = {s: max(-cap, min(cap, v)) for s, v in targets.items()}
        gross = sum(abs(v) for v in t.values())
        if gross > self.L.max_gross_lev * equity:
            k = self.L.max_gross_lev * equity / gross
            t = {s: v * k for s, v in t.items()}
        for s, px in limit_px.items():
            if abs(px / mids[s] - 1) > self.L.max_px_dev_frac:
                t.pop(s, None)  # fat-finger guard
        return t
```

### 9.7 Live loop skeleton (testnet first)
```python
# trader/live/main.py
import ccxt, time, pandas as pd
from trader.features.core import make_features
from trader.risk.engine import RiskEngine

ex = ccxt.binanceusdm({"apiKey": "...", "secret": "...", "enableRateLimit": True})
ex.set_sandbox_mode(True)  # TESTNET. Remove only after passing the paper-trading gate.
risk = RiskEngine()

def loop(symbols, model, feats):
    while True:
        bars = {s: pd.DataFrame(ex.fetch_ohlcv(s, "1h", limit=400),
                   columns=["ts","open","high","low","close","volume"]) for s in symbols}
        # ...build same features as research (need taker_buy_base via raw endpoint or drop it)...
        # preds = model.predict(latest feature rows); targets = weights * equity
        bal = ex.fetch_balance(); equity = float(bal["total"]["USDT"])
        risk.update_equity(equity, new_day=(time.gmtime().tm_hour == 0))
        # targets = risk.check_targets(...)
        # positions = ex.fetch_positions(symbols); diff → create_order(..., params={"postOnly": True})
        # log everything: features, preds, targets, orders, fills
        time.sleep(3600 - time.time() % 3600 + 5)  # wake just after bar close
```

### 9.8 LLM text-to-feature extractor (local Qwen via vLLM)
```python
# trader/llm/extract.py
from openai import OpenAI   # vLLM exposes an OpenAI-compatible server
import json
client = OpenAI(base_url="http://localhost:8000/v1", api_key="none")
SCHEMA = {"type":"object","properties":{
  "assets":{"type":"array","items":{"type":"string"}},
  "event_type":{"enum":["hack","listing","delisting","unlock","regulation","partnership",
                        "macro","depeg","earnings","other"]},
  "sentiment":{"type":"number","minimum":-1,"maximum":1},
  "confidence":{"type":"number","minimum":0,"maximum":1}},
  "required":["assets","event_type","sentiment","confidence"]}

def extract(headline: str, body: str, model="Qwen/Qwen3-30B-A3B-Instruct-2507"):
    r = client.chat.completions.create(model=model, temperature=0,
        messages=[{"role":"system","content":"Extract trading-relevant facts. Output JSON only."},
                  {"role":"user","content":f"{headline}\n\n{body[:4000]}"}],
        extra_body={"guided_json": SCHEMA})
    return json.loads(r.choices[0].message.content)
# Store with the NEWS PUBLISH TIMESTAMP (not processing time) → becomes a point-in-time feature.
```

---

## 10. Legal, operational and security notes
- **Trading your own money** algorithmically is legal almost everywhere. **Managing other people's money** needs registration (US: RIA with the SEC/state; futures/crypto derivatives → CFTC CPO/CTA via NFA; other countries have their own regimes).
- Check **exchange ToS and jurisdiction**: some venues (e.g., Binance global, Bybit, Hyperliquid front-ends) restrict US persons. Don't use VPNs to evade restrictions, because accounts and funds can be frozen.
- **API key security:** trade-only, **withdrawals disabled**, IP whitelist, separate sub-account per strategy.
- Taxes: every trade may be a taxable event. Keep full trade logs.
- Market manipulation (spoofing, wash trading, pump coordination) is illegal even in crypto. Don't design strategies around it.

---

## 11. Reading list (priority order)

**Core (read these):**
1. Marcos López de Prado: *Advances in Financial Machine Learning* (labels, purged CV, meta-labeling, backtest overfitting). **The single most important book for this project.**
2. Stefan Jansen: *Machine Learning for Algorithmic Trading* (2nd ed.). Hands-on code, broad.
3. Grinold & Kahn: *Active Portfolio Management* (fundamental law, alpha combination, risk).
4. Ernest Chan: *Algorithmic Trading* and *Quantitative Trading* (practical, mean-reversion/momentum).
5. Zura Kakushadze & Juan Andrés Serur: *151 Trading Strategies*. Isichenko: *Quantitative Portfolio Management* (how a real stat-arb shop works).

**Microstructure / execution / market making:**
- Bouchaud et al.: *Trades, Quotes and Prices*
- Cartea, Jaimungal & Penalva: *Algorithmic and High-Frequency Trading*
- Avellaneda & Stoikov (2008): "High-frequency trading in a limit order book"
- Almgren & Chriss (2000): "Optimal execution of portfolio transactions"

**Key papers:**
- Gu, Kelly & Xiu (2020): "Empirical Asset Pricing via Machine Learning"
- Moskowitz, Ooi & Pedersen (2012): "Time Series Momentum"
- Kakushadze (2016): "101 Formulaic Alphas"
- Bailey & López de Prado (2014): "The Deflated Sharpe Ratio"
- Zhang, Zohren & Roberts (2019): "DeepLOB"
- Liu et al. (2018): "Crypto factor" papers (Liu, Tsyvinski & Wu, "Common Risk Factors in Cryptocurrency")
- Xiao et al. (2024): "TradingAgents: Multi-Agents LLM Financial Trading Framework"

**Story / culture:** Gregory Zuckerman, *The Man Who Solved the Market* (Renaissance); Scott Patterson, *The Quants* and *Dark Pools*.

---

## 12. Realistic expectations
- A good small systematic crypto strategy: **net Sharpe 1–2**, with drawdowns of 10–25% that will happen.
- Expect **most ideas to fail**, which is normal. Budget months of research before the first live dollar.
- Live performance usually comes in at **~50–70% of the backtest Sharpe** even when you did everything right.
- Edges decay. Plan for continuous research, not a "set and forget" bot.
- The biggest risks aren't bad predictions. They are **bugs, exchange failures, leverage and overconfidence.** The risk engine and kill switch matter more than the model.
