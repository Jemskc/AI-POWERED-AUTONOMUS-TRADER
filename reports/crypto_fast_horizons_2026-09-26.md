# Fast-horizon test on real Binance 1-minute data: 2026-09-26

20 coins · 2025-09-01 .. 2026-09-25 23:59 UTC · walk-forward over the last 6 months (monthly retrain, expanding window) · 37 features (price, volume, taker order flow, BTC lead-lag, cross-sectional ranks)

Round-trip cost assumptions: taker (retail market orders): 12 bp · maker (retail limit orders): 4 bp · VIP market maker: 0 bp

## Horizon 1 minute(s)

- scored calls: 3,753,917 · actual up-rate 50.2% · typical move |r| 5.3 bp
- accuracy **51.4%** (AUC 0.519) · most confident 10%: **53.5%** · most confident 1%: **54.9%**
- average edge per call: all +0.13 bp · top 10% +0.29 bp · top 1% +0.28 bp
- simple rules for comparison: fade last minute -0.01 bp · follow order flow -0.03 bp
- profit per trade after round-trip costs (top 1% calls): taker (retail market orders): -11.72 bp · maker (retail limit orders): -3.72 bp · VIP market maker: +0.28 bp
- top features: btc_r1, tod_cos, tod_sin, btc_imb5, mkt_imb5, mkt_r5, btc_r3, btc_r15

## Horizon 5 minute(s)

- scored calls: 4,244,369 · actual up-rate 49.9% · typical move |r| 10.2 bp
- accuracy **51.7%** (AUC 0.523) · most confident 10%: **53.3%** · most confident 1%: **55.4%**
- average edge per call: all +0.14 bp · top 10% +0.05 bp · top 1% +0.22 bp
- simple rules for comparison: fade last minute +0.02 bp · follow order flow -0.07 bp
- profit per trade after round-trip costs (top 1% calls): taker (retail market orders): -11.78 bp · maker (retail limit orders): -3.78 bp · VIP market maker: +0.22 bp
- top features: ma30, hi60, tod_sin, tod_cos, lo60, btc_r15, lvol60, mkt_r5

## Horizon 15 minute(s)

- scored calls: 4,434,150 · actual up-rate 50.0% · typical move |r| 15.8 bp
- accuracy **51.9%** (AUC 0.525) · most confident 10%: **53.3%** · most confident 1%: **53.2%**
- average edge per call: all +0.33 bp · top 10% +0.05 bp · top 1% -0.54 bp
- simple rules for comparison: fade last minute -0.02 bp · follow order flow -0.14 bp
- profit per trade after round-trip costs (top 1% calls): taker (retail market orders): -12.54 bp · maker (retail limit orders): -4.54 bp · VIP market maker: -0.54 bp
- top features: ma30, tod_sin, tod_cos, r30, btc_r15, lo60, hi60, r240

## Horizon 60 minute(s)

- scored calls: 4,575,777 · actual up-rate 50.2% · typical move |r| 30.7 bp
- accuracy **51.6%** (AUC 0.521) · most confident 10%: **52.1%** · most confident 1%: **49.4%**
- average edge per call: all +1.08 bp · top 10% +0.11 bp · top 1% -2.98 bp
- simple rules for comparison: fade last minute +0.09 bp · follow order flow -0.24 bp
- profit per trade after round-trip costs (top 1% calls): taker (retail market orders): -14.98 bp · maker (retail limit orders): -6.98 bp · VIP market maker: -2.98 bp
- top features: tod_cos, tod_sin, lvol60, r240, r60, vol_ratio, lo60, vz15

## Summary

|   h |           n |   up_rate |   acc |   auc |   acc_top10 |   acc_top1 |   bp_all |   bp_top10 |   bp_top1 |   bp_rule_reversal |   bp_rule_orderflow |   median_abs_move_bp |   trades_top1_per_day |
|----:|------------:|----------:|------:|------:|------------:|-----------:|---------:|-----------:|----------:|-------------------:|--------------------:|---------------------:|----------------------:|
|   1 | 3753917.000 |     0.502 | 0.514 | 0.519 |       0.535 |      0.549 |    0.135 |      0.294 |     0.277 |             -0.007 |              -0.026 |                5.276 |               210.901 |
|   5 | 4244369.000 |     0.499 | 0.517 | 0.523 |       0.533 |      0.554 |    0.144 |      0.049 |     0.217 |              0.016 |              -0.069 |               10.226 |               238.455 |
|  15 | 4434150.000 |     0.500 | 0.519 | 0.525 |       0.533 |      0.532 |    0.327 |      0.052 |    -0.542 |             -0.020 |              -0.135 |               15.835 |               249.128 |
|  60 | 4575777.000 |     0.502 | 0.516 | 0.521 |       0.521 |      0.494 |    1.075 |      0.113 |    -2.980 |              0.090 |              -0.239 |               30.713 |               257.129 |

