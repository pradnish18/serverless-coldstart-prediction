# Multi-seed results (test days 27-30)

Each seed's threshold is tuned on its own validation predictions. sd = standard deviation across seeds.


## MAIN all (eligible functions)

| model | seeds | F1 mean ± sd | F1 min–max | MCC mean ± sd | PR-AUC mean ± sd | Accuracy mean |
|---|---|---|---|---|---|---|
| xgb_no_static | 3 | 0.9427 ± 0.0002 | 0.9424–0.9430 | 0.9341 ± 0.0002 | 0.9881 ± 0.0001 | 0.9849 |
| xgb_full | 3 | 0.9426 ± 0.0005 | 0.9420–0.9431 | 0.9340 ± 0.0005 | 0.9877 ± 0.0002 | 0.9850 |
| abigru | 3 | 0.9423 ± 0.0006 | 0.9416–0.9430 | 0.9337 ± 0.0007 | 0.9884 ± 0.0004 | 0.9848 |
| bigru | 3 | 0.9414 ± 0.0007 | 0.9404–0.9420 | 0.9326 ± 0.0008 | 0.9880 ± 0.0004 | 0.9846 |
| gru | 3 | 0.9410 ± 0.0008 | 0.9399–0.9417 | 0.9322 ± 0.0009 | 0.9874 ± 0.0003 | 0.9844 |
| mlp_static | 3 | 0.9409 ± 0.0003 | 0.9405–0.9412 | 0.9320 ± 0.0003 | 0.9871 ± 0.0002 | 0.9845 |
| rule_majority_7d (rule, deterministic) | – | 0.9194 | – | 0.9074 | 0.8564 | 0.9791 |
| rule_same_bucket_1d (rule, deterministic) | – | 0.9065 | – | 0.8925 | 0.8338 | 0.9756 |

## MAIN reactivation (no event in prior hour)

| model | seeds | F1 mean ± sd | F1 min–max | MCC mean ± sd | PR-AUC mean ± sd | Accuracy mean |
|---|---|---|---|---|---|---|
| xgb_no_static | 3 | 0.8548 ± 0.0051 | 0.8487–0.8611 | 0.8543 ± 0.0050 | 0.9204 ± 0.0001 | 0.9988 |
| xgb_full | 3 | 0.8502 ± 0.0040 | 0.8448–0.8542 | 0.8497 ± 0.0039 | 0.9262 ± 0.0003 | 0.9987 |
| abigru | 3 | 0.8375 ± 0.0053 | 0.8311–0.8441 | 0.8371 ± 0.0052 | 0.9135 ± 0.0012 | 0.9986 |
| mlp_static | 3 | 0.8343 ± 0.0029 | 0.8315–0.8383 | 0.8339 ± 0.0028 | 0.9111 ± 0.0004 | 0.9986 |
| bigru | 3 | 0.8316 ± 0.0008 | 0.8308–0.8326 | 0.8312 ± 0.0007 | 0.9130 ± 0.0026 | 0.9985 |
| gru | 3 | 0.8314 ± 0.0042 | 0.8267–0.8369 | 0.8311 ± 0.0041 | 0.9115 ± 0.0009 | 0.9985 |
| rule_majority_7d (rule, deterministic) | – | 0.8701 | – | 0.8701 | 0.7586 | 0.9989 |
| rule_same_bucket_1d (rule, deterministic) | – | 0.8032 | – | 0.8035 | 0.6477 | 0.9983 |