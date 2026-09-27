# Does futures-market data improve the model? Walk-forward ablation: 2026-09-26

Real Binance data, 20 coins, test 2022-07 .. 2026-09-27; all feature sets trained and tested on the SAME rows. Model = XGBoost+LightGBM average. Trading: confident calls only (|p-0.5|>0.04, else hold), every H hours.

## Horizon 4h

743,200 out-of-sample predictions · actual up-rate 49.5%

| features | accuracy | AUC | folds better than base | conf. accuracy (|p-.5|>.04) | conf. calls | Sharpe net 7bp | total net 7bp | Sharpe net 3bp | max DD |
|---|---|---|---|---|---|---|---|---|---|
| base | 52.39% | 0.5317 | -/9 | 54.55% | 201,774 | -0.39 | -81.9% | -0.17 | -92.1% |
| +funding | 52.38% | 0.5307 | 2/9 | 54.38% | 202,841 | -0.31 | -77.1% | -0.07 | -88.2% |
| +oi | 52.49% | 0.5328 | 6/9 | 54.47% | 218,492 | -0.01 | -54.0% | 0.24 | -90.4% |
| +all | 52.25% | 0.5312 | 5/9 | 54.40% | 222,115 | 0.24 | -8.0% | 0.50 | -71.3% |

Most useful NEW features (+all model, total gain): fng (8.0%), fng_chg7 (6.7%), fut_taker_chg24 (0.9%), fut_taker_z (0.9%), top_ls_chg24 (0.5%), top_ls_z (0.4%), crowd_ls_z (0.4%), prem24 (0.4%), fut_taker (0.4%), crowd_ls_chg24 (0.3%)
Share of total model gain from new features: 20.2%

## Horizon 24h

742,800 out-of-sample predictions · actual up-rate 49.4%

| features | accuracy | AUC | folds better than base | conf. accuracy (|p-.5|>.04) | conf. calls | Sharpe net 7bp | total net 7bp | Sharpe net 3bp | max DD |
|---|---|---|---|---|---|---|---|---|---|
| base | 50.29% | 0.5054 | -/9 | 52.66% | 141,144 | -0.19 | -70.7% | -0.18 | -90.5% |
| +funding | 50.41% | 0.5076 | 6/9 | 51.88% | 180,166 | 0.10 | -37.8% | 0.12 | -86.8% |
| +oi | 50.61% | 0.5102 | 7/9 | 51.85% | 165,424 | 0.30 | +4.6% | 0.32 | -80.6% |
| +all | 50.60% | 0.5126 | 6/9 | 52.93% | 185,073 | 0.00 | -49.6% | 0.04 | -83.0% |

Most useful NEW features (+all model, total gain): fng (18.0%), fng_chg7 (12.4%), fut_taker_z (1.3%), fut_taker_chg24 (1.2%), top_ls_chg24 (1.1%), top_ls (1.0%), top_ls_z (0.9%), prem24 (0.8%), smart_vs_crowd (0.6%), fut_taker (0.5%)
Share of total model gain from new features: 40.0%

