# Pre-warming simulation (test days 27-30, 385 eligible functions, 1152 five-minute buckets)

Total cold starts: **82,031** in 57,739 function-buckets; total cold-start latency: **81939 s** (22.8 h).

One pre-warm = one container kept ready for one 5-minute bucket; it avoids at most the first cold start in that bucket. Assumptions and limits are listed in `v2/prewarm_sim.py`. An upper bound for one container per function-bucket: avoiding every function-bucket's first cold start = 70.4% of cold starts and 60.2% of latency.

## Policies

| policy | pre-warms per bucket | cold starts avoided | % of cold starts | latency saved (s) | % of latency | wasted pre-warms | % wasted | idle container-hours |
|---|---|---|---|---|---|---|---|---|
| no_prewarm | 0.0 | 0 | 0.0% | 0 | 0.0% | 0 | 0.0% | 0 |
| keepalive_15min | 86.2 | 48,103 | 58.6% | 42,249 | 51.6% | 51,160 | 51.5% | 4,263 |
| keepalive_60min | 119.3 | 56,454 | 68.8% | 48,079 | 58.7% | 80,966 | 58.9% | 6,747 |
| rule_same_slot_yesterday | 50.2 | 52,373 | 63.8% | 44,171 | 53.9% | 5,442 | 9.4% | 454 |
| rule_majority_7d | 49.7 | 52,864 | 64.4% | 44,465 | 54.3% | 4,389 | 7.7% | 366 |
| xgb_full @ threshold 0.538 | 50.2 | 54,432 | 66.4% | 45,782 | 55.9% | 3,397 | 5.9% | 283 |
| abigru @ threshold 0.437 | 51.6 | 55,263 | 67.4% | 46,658 | 56.9% | 4,204 | 7.1% | 350 |
| mlp_static @ threshold 0.510 | 50.8 | 54,704 | 66.7% | 46,245 | 56.4% | 3,837 | 6.6% | 320 |
| hybrid_histogram (ATC'20, fallback 4 h) | 125.6 | 57,535 | 70.1% | 48,978 | 59.8% | 87,159 | 60.2% | 7,263 |
| hybrid_histogram (fallback 60 min) | 125.1 | 57,519 | 70.1% | 48,975 | 59.8% | 86,641 | 60.1% | 7,220 |

## Fixed budget: N pre-warmed containers per 5-minute bucket (top-N functions by score)

| N per bucket | xgb_full: % cold starts avoided / % wasted | abigru: % cold starts avoided / % wasted | rule votes (7-day): % cold starts avoided / % wasted |
|---|---|---|---|
| 5 | 7.0% / 0.0% | 7.0% / 0.0% | 7.0% / 0.8% |
| 10 | 14.0% / 0.0% | 14.0% / 0.0% | 13.8% / 1.7% |
| 20 | 27.9% / 0.5% | 28.0% / 0.2% | 27.4% / 2.5% |
| 30 | 40.7% / 3.4% | 40.8% / 3.3% | 40.0% / 5.1% |
| 40 | 50.8% / 9.5% | 50.8% / 9.5% | 49.8% / 11.3% |
| 50 | 58.4% / 16.8% | 58.4% / 16.9% | 57.5% / 18.1% |
| 60 | 63.8% / 24.3% | 63.8% / 24.3% | 62.9% / 25.3% |
| 80 | 68.6% / 38.9% | 68.6% / 38.9% | 67.8% / 39.6% |
| 100 | 70.2% / 50.0% | 70.3% / 50.0% | 69.3% / 50.7% |
| 125 | 70.3% / 59.9% | 70.4% / 59.9% | 69.3% / 60.5% |

## Policies as classifiers (test days 27-30)

| policy | MAIN all: precision / recall / F1 / MCC | MAIN reactivation: precision / recall / F1 / MCC |
|---|---|---|
| keepalive_15min | 0.485 / 0.833 / 0.613 / 0.566 | 0.000 / 0.000 / 0.000 / 0.000 |
| keepalive_60min | 0.411 / 0.978 / 0.579 / 0.559 | 0.000 / 0.000 / 0.000 / 0.000 |
| rule_same_slot_yesterday | 0.906 / 0.907 / 0.906 / 0.892 | 0.763 / 0.848 / 0.803 / 0.804 |
| rule_majority_7d | 0.923 / 0.916 / 0.919 / 0.907 | 0.900 / 0.842 / 0.870 / 0.870 |
| xgb_full @ threshold 0.538 | 0.941 / 0.943 / 0.942 / 0.933 | 0.981 / 0.837 / 0.903 / 0.906 |
| abigru @ threshold 0.437 | 0.929 / 0.957 / 0.943 / 0.934 | 0.971 / 0.831 / 0.896 / 0.898 |
| mlp_static @ threshold 0.510 | 0.934 / 0.947 / 0.941 / 0.932 | 0.985 / 0.823 / 0.897 / 0.900 |
| hybrid_histogram (ATC'20, fallback 4 h) | 0.398 / 0.996 / 0.568 / 0.553 | 0.023 / 0.880 / 0.045 / 0.127 |
| hybrid_histogram (fallback 60 min) | 0.399 / 0.996 / 0.570 / 0.554 | 0.023 / 0.868 / 0.045 / 0.126 |

Hybrid-histogram branch use (functions; a function can use several over time): fixed keep-alive 4, histogram 124, out-of-range forecaster 260.


Average number of functions with a cold start per bucket: 50.1.
