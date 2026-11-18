# Extra evaluation (main seed of each model)

## 1. Confusion matrices per slice (test days 27-30; threshold tuned on the same slice of validation)


### MAIN all (n=443,520, positives=57,739)

| model | threshold | TP | FP | FN | TN | precision | recall | F1 |
|---|---|---|---|---|---|---|---|---|
| xgb_full | 0.538 | 54,432 | 3,397 | 3,307 | 382,384 | 0.941 | 0.943 | 0.942 |
| xgb_no_static | 0.478 | 54,940 | 3,847 | 2,799 | 381,934 | 0.935 | 0.952 | 0.943 |
| abigru | 0.437 | 55,263 | 4,204 | 2,476 | 381,577 | 0.929 | 0.957 | 0.943 |
| bigru | 0.518 | 54,795 | 3,805 | 2,944 | 381,976 | 0.935 | 0.949 | 0.942 |
| gru | 0.493 | 55,007 | 4,079 | 2,732 | 381,702 | 0.931 | 0.953 | 0.942 |
| mlp_static | 0.510 | 54,704 | 3,837 | 3,035 | 381,944 | 0.934 | 0.947 | 0.941 |
| rule_majority_7d | 0.500 | 52,864 | 4,389 | 4,875 | 381,392 | 0.923 | 0.916 | 0.919 |
| rule_same_bucket_1d | 0.500 | 52,373 | 5,442 | 5,366 | 380,339 | 0.906 | 0.907 | 0.906 |

### MAIN reactivation (n=306,100, positives=1,285)

| model | threshold | TP | FP | FN | TN | precision | recall | F1 |
|---|---|---|---|---|---|---|---|---|
| xgb_full | 0.157 | 1,113 | 237 | 172 | 304,578 | 0.824 | 0.866 | 0.845 |
| xgb_no_static | 0.142 | 1,116 | 191 | 169 | 304,624 | 0.854 | 0.868 | 0.861 |
| abigru | 0.131 | 1,110 | 256 | 175 | 304,559 | 0.813 | 0.864 | 0.837 |
| bigru | 0.129 | 1,101 | 263 | 184 | 304,552 | 0.807 | 0.857 | 0.831 |
| gru | 0.132 | 1,104 | 282 | 181 | 304,533 | 0.797 | 0.859 | 0.827 |
| mlp_static | 0.131 | 1,105 | 263 | 180 | 304,552 | 0.808 | 0.860 | 0.833 |
| rule_majority_7d | 0.500 | 1,082 | 120 | 203 | 304,695 | 0.900 | 0.842 | 0.870 |
| rule_same_bucket_1d | 0.500 | 1,090 | 339 | 195 | 304,476 | 0.763 | 0.848 | 0.803 |

### LOW-HISTORY (n=52,992, positives=91)

| model | threshold | TP | FP | FN | TN | precision | recall | F1 |
|---|---|---|---|---|---|---|---|---|
| xgb_full | 0.085 | 55 | 210 | 36 | 52,691 | 0.208 | 0.604 | 0.309 |
| xgb_no_static | 0.140 | 40 | 102 | 51 | 52,799 | 0.282 | 0.440 | 0.343 |
| abigru | 0.131 | 33 | 87 | 58 | 52,814 | 0.275 | 0.363 | 0.313 |
| bigru | 0.111 | 28 | 74 | 63 | 52,827 | 0.275 | 0.308 | 0.290 |
| gru | 0.069 | 37 | 105 | 54 | 52,796 | 0.261 | 0.407 | 0.318 |
| mlp_static | 0.043 | 25 | 70 | 66 | 52,831 | 0.263 | 0.275 | 0.269 |
| rule_majority_7d | 0.500 | 13 | 3 | 78 | 52,898 | 0.812 | 0.143 | 0.243 |
| rule_same_bucket_1d | 0.500 | 25 | 94 | 66 | 52,807 | 0.210 | 0.275 | 0.238 |

### HOLDOUT (unseen functions, own history) (n=110,592, positives=14,085)

| model | threshold | TP | FP | FN | TN | precision | recall | F1 |
|---|---|---|---|---|---|---|---|---|
| xgb_full | 0.454 | 13,637 | 436 | 448 | 96,071 | 0.969 | 0.968 | 0.969 |
| xgb_no_static | 0.491 | 13,647 | 444 | 438 | 96,063 | 0.968 | 0.969 | 0.969 |
| abigru | 0.358 | 13,768 | 574 | 317 | 95,933 | 0.960 | 0.977 | 0.969 |
| bigru | 0.480 | 13,690 | 477 | 395 | 96,030 | 0.966 | 0.972 | 0.969 |
| gru | 0.443 | 13,723 | 526 | 362 | 95,981 | 0.963 | 0.974 | 0.969 |
| mlp_static | 0.417 | 13,715 | 515 | 370 | 95,992 | 0.964 | 0.974 | 0.969 |
| rule_majority_7d | 0.500 | 13,520 | 923 | 565 | 95,584 | 0.936 | 0.960 | 0.948 |
| rule_same_bucket_1d | 0.500 | 13,387 | 906 | 698 | 95,601 | 0.937 | 0.950 | 0.943 |

## 2. Training vs validation vs test (MAIN all slice, eligible functions; threshold from validation)

| model | split | accuracy | precision | recall | F1 | MCC | PR-AUC |
|---|---|---|---|---|---|---|---|
| xgb_full | train (days 7-23) | 0.9910 | 0.9670 | 0.9643 | 0.9656 | 0.9605 | 0.9960 |
| xgb_full | validation (24-26) | 0.9842 | 0.9329 | 0.9469 | 0.9399 | 0.9308 | 0.9865 |
| xgb_full | test (27-30) | 0.9849 | 0.9413 | 0.9427 | 0.9420 | 0.9333 | 0.9875 |
| xgb_no_static | train (days 7-23) | 0.9910 | 0.9608 | 0.9709 | 0.9658 | 0.9606 | 0.9958 |
| xgb_no_static | validation (24-26) | 0.9840 | 0.9238 | 0.9554 | 0.9393 | 0.9303 | 0.9863 |
| xgb_no_static | test (27-30) | 0.9850 | 0.9346 | 0.9515 | 0.9430 | 0.9344 | 0.9881 |
| abigru | train (days 7-23) | 0.9892 | 0.9511 | 0.9675 | 0.9592 | 0.9531 | 0.9941 |
| abigru | validation (24-26) | 0.9839 | 0.9196 | 0.9599 | 0.9393 | 0.9303 | 0.9863 |
| abigru | test (27-30) | 0.9849 | 0.9293 | 0.9571 | 0.9430 | 0.9345 | 0.9889 |
| bigru | train (days 7-23) | 0.9893 | 0.9574 | 0.9613 | 0.9594 | 0.9532 | 0.9942 |
| bigru | validation (24-26) | 0.9841 | 0.9283 | 0.9510 | 0.9395 | 0.9304 | 0.9859 |
| bigru | test (27-30) | 0.9848 | 0.9351 | 0.9490 | 0.9420 | 0.9333 | 0.9885 |
| gru | train (days 7-23) | 0.9893 | 0.9547 | 0.9639 | 0.9593 | 0.9532 | 0.9940 |
| gru | validation (24-26) | 0.9840 | 0.9241 | 0.9557 | 0.9396 | 0.9306 | 0.9855 |
| gru | test (27-30) | 0.9846 | 0.9310 | 0.9527 | 0.9417 | 0.9329 | 0.9878 |
| mlp_static | train (days 7-23) | 0.9892 | 0.9570 | 0.9609 | 0.9590 | 0.9528 | 0.9938 |
| mlp_static | validation (24-26) | 0.9839 | 0.9277 | 0.9505 | 0.9390 | 0.9298 | 0.9853 |
| mlp_static | test (27-30) | 0.9845 | 0.9345 | 0.9474 | 0.9409 | 0.9320 | 0.9873 |
| rule_majority_7d | train (days 7-23) | 0.9838 | 0.9373 | 0.9386 | 0.9379 | 0.9286 | 0.8878 |
| rule_majority_7d | validation (24-26) | 0.9774 | 0.9197 | 0.9049 | 0.9123 | 0.8993 | 0.8447 |
| rule_majority_7d | test (27-30) | 0.9791 | 0.9233 | 0.9156 | 0.9194 | 0.9074 | 0.8564 |
| rule_same_bucket_1d | train (days 7-23) | 0.9810 | 0.9266 | 0.9282 | 0.9274 | 0.9165 | 0.8695 |
| rule_same_bucket_1d | validation (24-26) | 0.9708 | 0.8882 | 0.8867 | 0.8874 | 0.8706 | 0.8023 |
| rule_same_bucket_1d | test (27-30) | 0.9756 | 0.9059 | 0.9071 | 0.9065 | 0.8925 | 0.8338 |

## 3. Paired statistical tests on the test set (MAIN all slice)

Paired bootstrap: 1,000 resamples of whole functions; each model keeps its validation threshold. p = two-sided bootstrap p-value for 'no difference'. McNemar compares row-level disagreements; rows of one function are correlated, so its p-value is optimistic and the bootstrap is the primary test.

| model A | model B | F1(A) − F1(B) | 95% CI | bootstrap p | McNemar χ² | McNemar p |
|---|---|---|---|---|---|---|
| xgb_full | rule_majority_7d | +0.0226 | [+0.0121, +0.0351] | 0.000 | 1182.5 | 3.9e-259 |
| abigru | rule_majority_7d | +0.0236 | [+0.0127, +0.0366] | 0.000 | 1147.2 | 1.9e-251 |
| mlp_static | rule_majority_7d | +0.0215 | [+0.0109, +0.0340] | 0.000 | 1003.7 | 2.9e-220 |
| abigru | xgb_full | +0.0010 | [-0.0002, +0.0024] | 0.132 | 0.2 | 0.64 |
| xgb_no_static | xgb_full | +0.0010 | [-0.0000, +0.0020] | 0.052 | 2.2 | 0.13 |
| abigru | mlp_static | +0.0021 | [+0.0008, +0.0037] | 0.000 | 21.6 | 3.3e-06 |
| abigru | bigru | +0.0010 | [-0.0004, +0.0032] | 0.284 | 3.2 | 0.072 |
| abigru | gru | +0.0013 | [-0.0001, +0.0031] | 0.078 | 13.0 | 0.00031 |
| bigru | mlp_static | +0.0011 | [-0.0002, +0.0023] | 0.084 | 10.7 | 0.0011 |
| gru | mlp_static | +0.0008 | [+0.0000, +0.0016] | 0.038 | 2.9 | 0.089 |