# Robustness checks

## A. Rolling-origin evaluation (eligible functions, role 0)

Each fold trains on days 7..(val start − 1), tunes the threshold on the next 3 days and tests on the following 4 days. Fold 3 is the original split. Static function statistics always come from days 0–6, so no fold leaks.

| fold | train days | val days | test days | test positives | model | F1 | MCC | PR-AUC | best trees |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 7–15 | 16–18 | 19–22 | 57,287 | XGBoost full | 0.9550 | 0.9483 | 0.9933 | 281 |
| 1 | 7–15 | 16–18 | 19–22 | 57,287 | rule: 7-day majority | 0.9383 | 0.9293 | 0.8893 | – |
| 1 | 7–15 | 16–18 | 19–22 | 57,287 | rule: same slot yesterday | 0.9235 | 0.9122 | 0.8630 | – |
| 2 | 7–19 | 20–22 | 23–26 | 57,642 | XGBoost full | 0.9399 | 0.9309 | 0.9862 | 330 |
| 2 | 7–19 | 20–22 | 23–26 | 57,642 | rule: 7-day majority | 0.9127 | 0.8999 | 0.8455 | – |
| 2 | 7–19 | 20–22 | 23–26 | 57,642 | rule: same slot yesterday | 0.8923 | 0.8762 | 0.8103 | – |
| 3 | 7–23 | 24–26 | 27–30 | 57,739 | XGBoost full | 0.9420 | 0.9333 | 0.9875 | 356 |
| 3 | 7–23 | 24–26 | 27–30 | 57,739 | rule: 7-day majority | 0.9194 | 0.9074 | 0.8564 | – |
| 3 | 7–23 | 24–26 | 27–30 | 57,739 | rule: same slot yesterday | 0.9065 | 0.8925 | 0.8338 | – |

**Mean over folds:** XGBoost full F1 0.9456 (range 0.9399–0.9550); rule: 7-day majority F1 0.9235 (range 0.9127–0.9383); rule: same slot yesterday F1 0.9074 (range 0.8923–0.9235).
**Gap XGBoost − 7-day rule per fold:** +0.0166, +0.0271, +0.0226.

## B. XGBoost hyperparameter search (original split; selection on validation PR-AUC only)

Fixed: subsample 0.8, colsample_bytree 0.8, reg_lambda 1.0, up to 1,500 trees with early stopping (40). Grid: max_depth × learning_rate × min_child_weight = 16 configurations. The default used everywhere else is depth 8, learning rate 0.05, min_child_weight 5.

| max_depth | learning_rate | min_child_weight | trees | val PR-AUC | val F1 |
|---|---|---|---|---|---|
| 4 | 0.03 | 5 | 1363 | 0.9862 | 0.9393 |
| 4 | 0.03 | 20 | 1499 | 0.9862 | 0.9394 |
| 4 | 0.1 | 5 | 530 | 0.9863 | 0.9397 |
| 4 | 0.1 | 20 | 533 | 0.9862 | 0.9396 |
| 6 | 0.03 | 5 | 943 | 0.9866 | 0.9399 |
| 6 | 0.03 | 20 | 844 | 0.9863 | 0.9402 |
| 6 | 0.1 | 5 | 389 | 0.9866 | 0.9399 |
| 6 | 0.1 | 20 | 354 | 0.9862 | 0.9395 |
| 8 | 0.03 | 5 | 531 | 0.9865 | 0.9402 |
| 8 | 0.03 | 20 | 621 | 0.9863 | 0.9398 |
| 8 | 0.1 | 5 | 171 | 0.9865 | 0.9395 |
| 8 | 0.1 | 20 | 137 | 0.9863 | 0.9399 |
| 10 | 0.03 | 5 | 154 | 0.9864 | 0.9396 |
| 10 | 0.03 | 20 | 155 | 0.9862 | 0.9400 |
| 10 | 0.1 | 5 | 111 | 0.9864 | 0.9399 |
| 10 | 0.1 | 20 | 185 | 0.9863 | 0.9395 |

**Selected on validation:** max_depth 6, learning_rate 0.1, min_child_weight 5 (389 trees).
**Test (scored once):** tuned F1 0.9429, MCC 0.9343, PR-AUC 0.9879 — default F1 0.9420, MCC 0.9333, PR-AUC 0.9875. Difference +0.0009 F1 (seed-to-seed sd of XGBoost is about 0.0005).