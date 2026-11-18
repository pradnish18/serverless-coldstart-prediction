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

Average number of functions with a cold start per bucket: 50.1.
