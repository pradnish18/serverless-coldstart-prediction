# Cold-Start Prediction in Serverless Computing — Complete Technical Documentation

**Project:** Forecasting function cold-start events on the Huawei Cloud Public Trace 2025 (Region 5)
**Reference paper:** N. Saravana Kumar and S. Selvakumara Samy, *Cold Start Prediction and Provisioning Optimization in Serverless Computing Using Deep Learning*, Concurrency and Computation: Practice and Experience, 2025; 37:e8392 (the "ODL-CSP / ABiGRU" paper)
**Documentation basis:** the project folder at `coldstart-problem/`, inspected on 2026-11-18 and updated on 2026-11-19 (pre-warming simulator, calibration, statistical tests, robustness checks, all checkpoints saved) and on 2026-11-23 (hybrid-histogram baseline from Azure, ATC'20; region-aware data loading; IEEE report draft). **The v2 pipeline is the current project; `archive_v1/` is kept only for history.** Every number in this document was either read from a saved artifact or computed from the saved predictions by scripts in `v2/`. Anything that could not be verified is marked **[NOT VERIFIED]**; anything that was never computed is marked **[NOT CALCULATED]**.

> **How to read this document.** Sections 1–2 explain the problem and the data for a newcomer. Sections 3–7 explain the system. Section 8 holds all results. Section 9 is a glossary. Sections 10–13 are the code map, rationale, conclusions and reproduction guide. Section 14 collects the summary tables and diagrams. A one-page "status at a glance" is below so that nothing important is hidden deep in the text.

---

## 0. Status at a glance (read this first)

| Question | Honest answer |
|---|---|
| Is there a working pipeline? | Yes. The **v2 pipeline** (`v2/`) runs from raw CSVs to a benchmark table, plus additional evaluations and a **pre-warming simulator** that turns predictions into pre-warming decisions. |
| Which models were really trained? | In v2: 4 XGBoost variants and 4 neural models (MLP, GRU, BiGRU, ABiGRU). XGBoost full/no-static and all four neural models were trained with **3 seeds each** (Section 8.9). |
| Is there an older pipeline? | Yes, moved to `archive_v1/` (`main.py`, `src/`, `config/`, its outputs). It was only **smoke-tested** (1 epoch, 2,048 rows, 5 functions). It is not a result, and v2 does not depend on it. |
| Any pretrained weights? | **None.** Every model starts from random initialization or an empty tree. |
| Do the models talk to each other? | **No.** Every model is trained and evaluated independently. There is no ensemble or stacking (verified by search, Section 5). |
| Are trained v2 model files saved? | **Yes**, in `v2/models/` (about 40 MB): 8 XGBoost models (`.json`, plus 4 feature-list files), 12 neural checkpoints (4 models × 3 seeds) (`.pt`, weights + scaler + feature order), and `thresholds.json`. Reloading them reproduces the reported predictions (Section 13.8). |
| Best headline result | On test days 27–30, learned models reach F1 ≈ 0.941–0.943 (3-seed means), against 0.919 for the best simple rule (significant, p < 0.001). XGBoost and ABiGRU are tied at the top (difference not significant, p = 0.13); the spread among all learned models is about 0.002 F1. |
| Does it help in practice? | **Yes, in simulation.** Pre-warming the functions the models flag avoids about 67% of cold starts with 6–7% wasted containers, against 64% for the best rule, 69% with 59% waste for a 60-minute keep-alive and 70% with 60% waste for the hybrid-histogram policy of Azure Functions (both use about 20–25 times more idle container-hours) (Sections 7.10, 8.12.3). |
| Is there a comparison with an industry pre-warming policy? | Yes (added 2026-11-23). `v2/hh_policy.py` adapts the **hybrid-histogram** policy (Shahrad et al., USENIX ATC 2020) to this event-only trace; it avoids 70.1% of cold starts but pre-warms about 126 containers per bucket and wastes 60.2% (7,263 idle container-hours). The adaptation is described in Section 7.10. |
| Can it run on other Huawei regions? | The code can: `CS_REGION=<1-5>` and `v2/region_prep.py` aggregate a region's zip day by day (Section 2.14a). **No result for regions 1–4 exists yet**; all reported numbers are Region 5. |
| Are the results robust? | Yes: the advantage over the rule holds in 3 rolling test windows; XGBoost tuning changes F1 by only +0.0009; all models are well calibrated (ECE ≤ 0.003); overfitting is small (Section 8.12). |
| Where is it published? | GitHub: `pradnish18/serverless-coldstart-prediction` (code, saved models, results, documentation; the raw dataset is downloaded separately). |
| Biggest open problem | Predicting cold starts of **brand-new functions with no history** is not solved (F1 0.24, ROC-AUC 0.57). |

---

# 1. Project Overview and Research Background

## 1.1 Title and domain

**Title:** Cold-start event forecasting for serverless functions using machine learning on a production cloud trace.
**Domain:** Cloud computing → Function-as-a-Service (FaaS) → resource provisioning → time-series / tabular machine learning.

## 1.2 Background (a short story first)

Imagine a pizza shop with one oven. At dinner time the oven is hot and pizzas come out in minutes. At 3 a.m. nobody orders, so the owner switches the oven off to save electricity. When a customer walks in at 3 a.m., the cook must first heat the oven, and the customer waits much longer than usual.

Serverless platforms (AWS Lambda, Azure Functions, Google Cloud Functions, Huawei Cloud FunctionGraph) work the same way.
* A **function** is a small program that runs when triggered (an HTTP request, a timer, a file upload).
* The platform runs it inside a **container** (the "oven"). Keeping containers alive costs memory and CPU.
* When no container is ready, the platform has to create one. That delay is a **cold start**. When a container is ready, the call is a **warm start**.

## 1.3 What is the cold-start problem?

A **cold start** is the extra latency incurred when a function invocation arrives and no suitable, already-initialized execution environment exists. According to the reference paper, the platform must create the container, load the function code and dependencies, initialize the runtime, and only then run the handler. The Huawei data provider's documentation breaks this wait into four parts: pod allocation, code deployment, dependency loading, and scheduling/network overhead (Section 2.3).

## 1.3a Why it is important

* It adds latency (milliseconds to seconds) to user-facing requests.
* Providers cannot simply keep everything warm because idle containers cost money. The trade-off is **latency versus cost**.
* If the platform could **predict** cold starts, it could create containers shortly before they are needed ("pre-warming") and keep idle capacity small.

## 1.3b Real-world challenges

1. **Irregular demand.** Some functions fire on timers, others on user traffic, others rarely.
2. **No future information.** Predictions must use only the past.
3. **Class imbalance.** Cold starts are a minority event (about 13% of function-time-slots in our data).
4. **New functions.** A function with no history cannot be forecast from its own past.
5. **Different costs.** One cold start may cost 0.05 s, another 20 s, so counting events is not the same as saving time.
6. **Incomplete telemetry.** Public traces often contain only some of the events (this project's data contains cold starts only).

## 1.4 Problem statement (as implemented)

> For a function **f** and a 5-minute time bucket **t**, estimate the probability that **f** has at least one recorded cold-start event in bucket **t**, using only information from buckets before **t**, and convert that probability into a yes/no decision with a threshold chosen on validation data.

This is a **binary classification** task (Glossary, Section 9).

**What the problem statement deliberately does not claim:** that we can tell whether an individual request is cold or warm. The data has only cold-start events (Section 2), so "no event" means "no recorded cold start", not "ran warm".

## 1.5 Research objectives

1. Build a leakage-safe, reproducible pipeline for cold-start occurrence forecasting on the Huawei trace.
2. Establish the **baseline difficulty** of the task with simple seasonal rules.
3. Compare tree models (XGBoost) with neural models (MLP, GRU, BiGRU, ABiGRU) under identical inputs and splits.
4. Identify where learned models are actually needed: rare "reactivation" events, low-history functions, unseen functions.
5. Produce a trustworthy benchmark (confidence intervals, multiple seeds) that a research paper can build on.

## 1.6 Research questions and their status

| # | Question | Status |
|---|---|---|
| RQ1 | Do learned models beat strong seasonal rule baselines? | **Answered:** yes, by about 2.4 F1 points overall (Section 8.2). |
| RQ2 | Does a sequence encoder (GRU / BiGRU / ABiGRU) add value over tabular features? | **Answered (3 seeds):** practically no. ABiGRU beats the no-sequence MLP by 0.0014 F1 (consistent across seeds, but tiny) and only ties XGBoost; GRU and BiGRU do not beat the MLP (Section 8.9). |
| RQ3 | Do learned models help on hard "reactivation" events? | **Partly:** they rank better (PR-AUC 0.91–0.93 vs 0.76) but the 7-day majority rule has a higher F1 (0.870 vs 0.84–0.86). |
| RQ4 | Can cold starts of unseen functions be predicted from hardware and pool context alone? | **Answered: no** (F1 0.239, ROC-AUC 0.567). |
| RQ5 | Can predictions drive a cost-aware pre-warming policy? | **Answered with a simulator** (Section 8.12.3): at about 50 pre-warmed containers per 5-minute bucket, the models avoid 66–67% of cold starts with 6–7% waste, versus 64% / 8% for the best rule, 69% / 59% for a 60-minute keep-alive and 70% / 60% for the hybrid-histogram policy. |

## 1.7 Expected contributions (honest framing)

* A clean, leakage-audited benchmark and evaluation protocol on a public production trace.
* Evidence that, on this data, a simple "same time yesterday / last week" rule is already very strong, so published accuracy numbers on other data should be read with that baseline in mind.
* Evidence that model architecture (ABiGRU vs GRU vs XGBoost) matters little here compared with the seasonal features.
* An honest negative result on new-function prediction.
* A pre-warming simulation showing what the predictions are worth operationally: near keep-alive and hybrid-histogram benefit with about 20–25 times fewer idle container-hours.
* Supporting evidence: paired significance tests, calibration analysis, rolling-origin evaluation and a hyperparameter sensitivity check.

It is **not** a claim of a new state-of-the-art architecture.

## 1.8 Existing approaches and their limitations

The reference paper's literature review (read from the paper text) lists work on: traffic prediction (ASIP), reinforcement learning for proactive initialization, container caching (S-Cache, centralized cache controllers, HotC), SARIMA-based autoscaling (PBA), function fusion, language/memory studies, container-lifecycle-aware scheduling (CAS), application-level code loading (FaaSLight), schedulers (Hermes), TCN ensembles for instance arrival forecasting, and LSTM + RL warm-up. The paper itself notes limits such as dependence on workload patterns, scalability, and difficulty in setting ideal warm-up parameters.

**Limitations of the reference paper's own evaluation, as read from its text** (these are observations, not accusations, and some may reflect missing detail in the paper):
* Data: 10,000 rows of AWS Lambda data in a table whose sample (Table 3) shows one timestamp for every row; the collection method and the label's relationship to the features (for example historical/consecutive cold-start counts alongside the cold-start indicator) are not specified.
* The train/test split procedure is not described.
* Table 5 compares against *other publications' figures*, not models rerun on the same data.
* The paired t-test section (Section 4) mentions generating synthetic values "centered around the mean metric of the algorithm", which means it is not a test on independent runs.
* The reported test confusion matrix (Figure 7a) shows 241 warm-as-cold errors and zero cold-as-warm errors while the stated recall is 90.42%, which seems internally inconsistent **[NOT VERIFIED — read from the paper text only]**.

Because of these points we do **not** treat the paper's 90.36% test accuracy as a number to "beat". Different data and different definitions make that comparison invalid.

## 1.9 How this project addresses those limitations

| Limitation in prior evaluation | What this project does |
|---|---|
| Unclear split / leakage | Strict time-based split; all features use only earlier buckets; function statistics restricted to days 0–6; tested by erasing the future (Section 6.5). |
| No simple baselines | Six rule baselines run on the identical test rows. |
| Single run | Multiple seeds for the learned models (Section 8.9). |
| No uncertainty | Bootstrap 95% confidence intervals, resampling whole functions. |
| Averages hide hard cases | Separate slices: reactivation, low-history, unseen functions. |
| Accuracy-only reporting | F1, MCC, PR-AUC, ROC-AUC, precision, recall, confusion matrices. |

---

# 2. Complete Dataset Documentation

## 2.1 Datasets used

| Dataset | Used? | Purpose |
|---|---|---|
| **Huawei Cloud Public Trace 2025 — Region 5 cold starts** (31 daily CSV files) | **Yes, the only dataset used for modeling** | Source of all training, validation and test data |
| Reference paper's AWS Lambda data (10,000 rows) | **No** — not available in the project | Described in the paper only |
| An Azure dataset | **No** — no Azure dataset exists in the project or the Downloads folder (searched). Compatibility could not be assessed. | — |
| Huawei Regions 1–4 cold starts (`R1.zip`–`R4.zip` in the project root, git-ignored; each holds `day_XX.csv` files) | **No results yet** | The loader supports them (Section 2.14a); not trained or evaluated |
| Huawei requests table, time-series table, trigger-type table | **No** — described by the provider's README but not downloaded into this project | Possible future data (Section 12.6). **[NOT VERIFIED beyond the README text]** |

## 2.2 Source and location

* **Source:** the `sir-lab/data-release` repository (cloned into `_github_repo/`), described in `_github_repo/README_data_release_2025.md` and the paper "Serverless Cold Starts and Where to Find Them" (EuroSys 2025; PDF in `_github_repo/papers/`).
* **Local copy:** `dataset/raw/region5_cold_starts/R5/day_00.csv … day_30.csv` (31 files) plus the original archive `dataset/raw/region5_cold_starts.zip` (about 57 MB).
* **Timestamps:** normalized by the provider to start at 0 on day 0 (README).

## 2.3 Structure, size, and every column

Verified by loading all 31 files with pandas:

* **Rows:** 805,745 cold-start events. Per-day count: mean 25,992, minimum 25,097, maximum 26,962.
* **Columns:** 12. **Missing values: 0 in every column.**
* **Time span:** day 0 to day 30; `time` ranges from 0 to 2,678,389 seconds.

| Column | pandas type | Definition (provider README unless noted) | Example | Used in this project? |
|---|---|---|---|---|
| `day` | int64 | Day index of the event, 0–30 | `0` | Only to validate `time`; split is by bucket index |
| `time` | float64 | Cold-start timestamp in **seconds since the start of day 0** (does not reset daily) | `0.254` | **Yes** — defines the 5-minute bucket |
| `clusterName` | int64 | Cluster that processed the event; values 1–4 | `4` | Yes (per-function most common cluster, as a feature) |
| `funcName` | int64 | Anonymized function name | `210` | Yes (part of function identity) |
| `userID` | int64 | Anonymized owner of the function | `261` | Yes (part of function identity) |
| `requestID` | object | Hash identifying the request that triggered scaling. One request may trigger several cold starts (so rows can share a `requestID`: 792,641 unique IDs among 805,745 rows) | `cbe1483a…` | No |
| `totalCost_cold_start` | float64 | Total time spent on the cold start, **seconds**. The provider says this is the value to use as cold-start time | `0.089387` | Yes (sum/mean per bucket; cost features) |
| `podAllocationCost` | float64 | Seconds to start a pod, or select one from the pool | `0.000115` | Yes (function-level average, days 0–6) |
| `deployCodeCost` | float64 | Seconds to download, extract and deploy function code | `0.061` | Yes (function-level average, days 0–6) |
| `deployDependencyCost` | float64 | Seconds to fetch and load dependencies | `0.0` | Yes (fraction of events with dependency cost > 0, days 0–6) |
| `schedulingCost` | float64 | Seconds of networking, routing and scheduling overhead | `0.028272` | Yes (function-level average, days 0–6) |
| `podID` | object | Pod identifier; contains the pool name, e.g. `pool24-600-512-0000577863` = pool 24 whose pods have **CPU limit 600 millicores and memory limit 512 MB** | `pool24-600-512-0000577863` | Yes (pool name parsed into pool id, CPU, memory) |

Dataset size facts (verified): 521 distinct `funcName` values, 358 distinct `userID` values, 4 clusters, 802,828 distinct `podID` values, and **527 distinct function identities** when a function is defined as *(funcName, userID, pool name)* (Section 2.6). `totalCost_cold_start` has mean 0.915 s and maximum 58.9 s.

### Critical limitation of the data

Every row **is** a cold start. The files contain **no warm invocations, no request counts, and no CPU/memory utilization**. Therefore:
* "A cold start is identified" simply because a row exists for it.
* The prediction target is "will a cold-start row appear", not "is this request cold or warm".
* An empty bucket does **not** prove the function was idle.

## 2.4 Target variable

* **Name:** `y` (binary).
* **Definition:** `y(f,t) = 1` if function `f` has at least one cold-start row in bucket `t`, else `0`.
* **Prevalence:** 13.0% positives in the test main frame (57,739 of 443,520 rows).
* **Code:** `v2/data.py` builds the count matrix; `v2/frames.py` takes `y = ctx.ev[f, t]` where `ev = (cnt > 0)`.

## 2.5 Time buckets

Time is cut into **5-minute buckets** (300 seconds): `bucket = floor(time / 300)`.
* 12 buckets = 1 hour; **288 buckets = 1 day**; 31 days = **8,928 buckets**.
* Constants in `v2/data.py`: `B = 300`, `PD = 288`, `DAYS = 31`, `TOT = 8928`.

**Why 5 minutes?** It is long enough to contain events for most active functions, short enough to be actionable for pre-warming, and it matches the original project's choice (`archive_v1/config/config.py: TIME_BUCKET_SECONDS = 300`). It was not tuned. **[A different bucket size was not tested.]**

## 2.6 Function identity

A function is identified by *funcName + userID + pool name*. This follows the provider's definition (README: "A unique function is identified by funcID, which is a combination of funcName, userID, and poolName"). The pool name is `podID` with its last `-<number>` removed. `v2/data.py` joins them with `--` (the older `src/data/loader.py` joins them with `---`).

## 2.7 The two cached matrices

`v2/data.py` converts the CSVs into two matrices of shape **(527 functions × 8,928 buckets)**:
* `cnt[f, b]` = number of cold-start rows in that function-bucket.
* `cost[f, b]` = sum of `totalCost_cold_start` in that function-bucket.

Saved with metadata in `v2/cache/matrices.npz` (about 2.4 MB; other regions use `matrices_R<n>.npz`). Since 2026-11-23 the cache is built from per-day aggregates: `data.aggregate()` reduces each day's CSV to per-(function, bucket, cluster) sums (count, cost, dependency flag, code/scheduling/allocation cost), and `build()` combines them. The Region 5 cache rebuilt this way is identical to the earlier one (805,745 events; roles 385 / 96 / 46).

## 2.8 Function roles (how functions are grouped)

| Role | Meaning | Count | Used for |
|---|---|---|---|
| 0 — normal | ≥ 10 cold starts in days 0–23, not held out | 385 | Training, and the main test slice ("MAIN all") |
| 1 — holdout | A random 20% (seed 42) of the eligible functions, **excluded from training** | 96 | "Unseen function" evaluation |
| 2 — low-history | Fewer than 10 cold starts in days 0–23 | 46 | Included in training; reported as the "LOW-HISTORY" slice |

Role membership is decided from days 0–23 counts. It is used **only** to define groups and the holdout, never as an input feature.

## 2.9 Missing values and cleaning

* The CSVs have **no missing values** (verified), so no imputation of raw data is needed.
* `pool_id`, `pool_cpu`, `pool_mem` are **NaN for 2 function identities** whose pod name does not match `pool<id>-<cpu>-<mem>`. **[Which pod names these are was not inspected.]** XGBoost handles NaN natively. The neural models replace NaN by 0 after standardization (`v2/run_deep.py`).
* Empty buckets are filled with 0 events and 0 cost (the matrices start as zeros).
* No rows are removed or filtered from the raw data. (The only filtering is the role assignment.)
* The older loader (`archive_v1/src/data/loader.py`) additionally validates that `time` lies within the 24-hour window of its `day` and raises an error otherwise. v2 does not repeat that check.

## 2.10 Splits (time-based, never random)

| Split | Days | Target buckets | Function set | Purpose |
|---|---|---|---|---|
| Warm-up | 0–6 | — | — | History for lag features; static function statistics |
| **Train** | 7–23 | 2,016 … 6,911 | roles 0 and 2 (431 functions) → 2,110,176 rows | Fit models |
| **Validation** | 24–26 | 6,912 … 7,775 | roles 0 and 2 → 372,384 rows; holdout functions → 82,944 rows | Early stopping, threshold choice |
| **Test** | 27–30 | 7,776 … 8,927 | roles 0 and 2 → 496,512 rows (of which role 0: 443,520); holdout → 110,592 rows | Final scores only |

**Why time-based?** In real use the model predicts the future from the past. A random split would let the model "see" the day after a bucket, giving optimistic scores.

**Why skip days 0–6 as training targets?** The 7-day-ago features need 7 days of earlier history.

## 2.11 Feature engineering (the 57 features)

Built by `v2/features.py` (`make(ctx, fsel, t_lo, t_hi, kind)`), identical for every model so that comparisons are fair. For the row (function f, bucket t) every feature uses only buckets **< t** (or days 0–6 statistics).

| # | Group | Names | Count | Meaning and why |
|---|---|---|---|---|
| 1 | Clock | `hour_sin`, `hour_cos`, `slot` | 3 | Time of day. Sine and cosine put 23:55 next to 00:05 on a circle. `slot` = bucket index within the day (0–287) |
| 2 | Hardware | `pool_cpu`, `pool_mem`, `cluster` | 3 | Container size and cluster of the function |
| 3 | Pool profile | `pool_hour_rate` | 1 | Share of function-buckets of this pool with events at this hour of day, from days 0–6, computed only from non-holdout functions |
| 4 | Pool live load | `pool_last1`, `pool_last12`, `pool_yday` | 3 | Activity of *other* functions of the same pool in the last bucket, the last hour, and the same slot yesterday, each divided by pool size |
| 5 | Recent history | `lag1` … `lag12` | 12 | Was there an event 1 … 12 buckets ago (0/1) |
| 6 | Seasonal history | `d1_-1,d1_+0,d1_+1`, same for days 2, 3 and 7 | 12 | Event at the same time of day 1, 2, 3, 7 days earlier, at offsets −1, 0, +1 bucket (jitter tolerance) |
| 7 | Seasonal votes | `slot_hits_7d`, `slot_hits_7d_pm1` | 2 | Number of the last 7 days with an event in the same slot (and in the ±1 neighbours) |
| 8 | Rolling counts | `roll3`, `roll12`, `roll72`, `roll288`, `roll2016` | 5 | Events in the previous 15 min, 1 h, 6 h, 1 day, 7 days |
| 9 | Recency | `since_last` | 1 | Buckets since the last event, capped at 2,016 (and set to the cap if there is none) |
| 10 | Intensity | `cnt_lag1`, `cnt_yday`, `cost_lag1`, `cost_yday` | 4 | log(1+count) and log(1+cost) for the last bucket and for the same slot yesterday |
| 11 | Function profile (days 0–6 only) | `tr_events`, `tr_mean_cost`, `tr_dep_frac`, `tr_code`, `tr_sched`, `tr_alloc`, `tr_active_days`, `tr_slot_conc`, `tr_event_buckets` | 9 | Early behaviour of the function: events, mean cost, share of events that needed dependency loading, mean code/scheduling/allocation cost, number of active days, how concentrated its events are in one daily slot, number of event buckets |
| 12 | Function hourly rate | `fn_hour_rate` | 1 | Function's own event rate at this hour of day, days 0–6 |
| 13 | Age | `age_buckets` | 1 | Buckets since the function first appeared in days 0–6 (−1 if not yet seen) |

Total: 3+3+1+3+12+12+2+5+1+4+9+1+1 = **57**. The names are in this order in the saved XGBoost metadata.

**The "cold" feature set (10 features)** keeps only groups 1, 2, 3 and 4 (no own history). It is used for the new-function experiment (Section 4.6).

### Known small inconsistencies (disclosed)
* `pool_yday` includes the function's own event of the same slot yesterday (a legitimate past value), while `pool_last1` and `pool_last12` exclude the function itself. The inconsistency is small and affects only the pool features.
* The old comment in `v2/data.py` claiming static stats come from the training period was replaced (2026-11-23): the code uses days 0–6 only (`STAT_END = 7*PD`); the role (slice membership) uses days 0–23 counts but is never a feature.
* `data.py` has a `user` column in the metadata that is **not** used as a feature.

## 2.12 Normalization / transformations

* **Tree models and rules:** no scaling needed.
* **Neural models:** the 57 features are standardized with **training-set mean and standard deviation**, clipped to ±6, NaN replaced by 0 (`v2/nets.py`, function `standardize`). The scaler (`mu`, `sd`) is computed from the training rows and **saved inside each neural checkpoint** (Section 13.8).
* The sequence channels (0/1 events, log cost, sine/cosine, pool activity ratio) are used as they are.

## 2.13 Feature selection

No automated feature selection or dimensionality reduction was applied. Instead, **ablations** remove groups to measure their value (Section 8.6): `xgb_no_static` (removes groups 11–13), `xgb_no_pool` (removes groups 3–4 and the two `pool_` hardware columns of group 2; 51 features left), and `xgb_cold_newfunc` (keeps only groups 1–4).

## 2.14 Leakage controls (why each exists)

| Risk | Control |
|---|---|
| Using the future | Every feature for bucket t reads only buckets < t. Verified by zeroing a function's series from t onward and recomputing: maximum feature difference 0.0. |
| Static function statistics seeing target days | Restricted to days 0–6, before any training target. (A first version used days 0–23; it was found and fixed before any reported result.) |
| Pool statistics from held-out functions | Pool profile and live pool load are computed from non-holdout functions only. |
| Threshold tuned on test | Threshold is chosen on validation data (Section 8.1). |
| Scaler seeing test data | Standardization uses training rows only. |

## 2.14a Region-aware loading (added 2026-11-23)

* `CS_REGION` (environment variable, default `5`) selects the Huawei region. Region 5 keeps the cache name `matrices.npz`; other regions use `matrices_R<n>.npz` and a per-day folder `v2/cache/agg_R<n>/`.
* `python3 v2/region_prep.py <region> [zip]` reads a region's zip **without extracting it**, aggregates one day at a time (so large regions fit in memory), writes `agg_R<n>/day_XX.pkl`, skips days already done (resumable; `PREP_BUDGET` seconds limits one run) and then builds the region cache.
* Only the code path exists for regions 1–4; they have not been modeled, so every result in this document is Region 5. Function counts, roles and thresholds would differ per region.

---

# 3. Complete System Architecture

## 3.1 Two pipelines exist

| Pipeline | Folder | Status |
|---|---|---|
| **v2 (current, benchmark)** | `v2/` | Fully executed. All reported results come from it. |
| **v1 (original)** | `archive_v1/` (`main.py`, `src/`, `config/`) | Implemented; verified by a **smoke test only** (Section 8.8). 17 features, 12-bucket window, 6 models. Archived for reference; its paths assume it lives at the project root, so it does not run from the archive without editing `config.py`. |

## 3.2 Architecture diagram (v2)

```
                          RAW DATA
   dataset/raw/region5_cold_starts/R5/day_00.csv … day_30.csv
                (805,745 cold-start rows, 12 columns)
                               │
                               ▼
        ┌───────────────────────────────────────────┐
        │ v2/data.py                                │
        │  • build function id  (funcName,userID,pool)
        │  • 5-minute buckets   (time // 300)       │
        │  • cnt[f,b], cost[f,b]  (527 × 8,928)     │
        │  • static stats (days 0–6), pool profiles │
        │  • roles: normal / holdout / low-history  │
        └───────────────────────┬───────────────────┘
                                ▼   v2/cache/matrices.npz
        ┌───────────────────────────────────────────┐
        │ v2/features.py   (class Ctx + make())     │
        │  • lags, seasonal lags, rolling counts    │
        │  • pool live load, static profile         │
        │  → X: (rows × 57) "full"  or (rows × 10) "cold"
        └───────────────────────┬───────────────────┘
                                ▼
        ┌───────────────────────────────────────────┐
        │ v2/frames.py   splits by time             │
        │  train (d7–23) | val_main (d24–26)        │
        │  test_main (d27–30) | val_hold | test_hold│
        └───┬─────────────────┬───────────────┬─────┘
            │                 │               │
            ▼                 ▼               ▼
   ┌────────────────┐ ┌───────────────┐ ┌───────────────────────┐
   │ RULES          │ │ XGBoost ×4    │ │ Neural models ×4      │
   │ (no training)  │ │ run_gbdt.py   │ │ run_deep.py           │
   │ in run_gbdt.py │ │ run_xgb_seeds │ │ MLP | GRU | BiGRU |   │
   │                │ │               │ │ ABiGRU                │
   └───────┬────────┘ └──────┬────────┘ └──────────┬────────────┘
           │                 │                     │
           └──── probability / 0-1 scores per row ─┘
                             │       saved as results/preds_*.npz
                             ▼
        ┌───────────────────────────────────────────┐
        │ v2/evalx.py + report.py + final_table.py  │
        │  threshold on validation → metrics on test│
        │  slices, bootstrap CI, cost capture       │
        └───────────────────────┬───────────────────┘
                                ▼
          results/benchmark.md , final_test_table.txt
                                │
            ┌───────────────────┼─────────────────────────────┐
            ▼                   ▼                             ▼
┌──────────────────────┐ ┌───────────────────────┐ ┌──────────────────────────┐
│ extra_eval.py        │ │ prewarm_sim.py        │ │ robustness.py            │
│ calibration.py       │ │ PRE-WARMING SIMULATOR │ │ rolling-origin XGBoost,  │
│ slice confusion,     │ │ preds + thresholds +  │ │ XGBoost tuning           │
│ train metrics, tests,│ │ cache → policies →    │ │ (trains its own XGBoost  │
│ calibration          │ │ cold starts avoided,  │ │  from the cache)         │
│                      │ │ waste; fig7           │ │                          │
└──────────────────────┘ └───────────────────────┘ └──────────────────────────┘
   extra_eval.md,          prewarm_sim.md,            robustness.md
   calibration.md          figures/fig7
```

**Explanation of each connection**
1. *CSV → data.py:* rows become two matrices (counts and costs).
2. *data.py → matrices.npz → features.py:* the cached matrices are loaded into a context object `Ctx`, which also precomputes cumulative sums, "last event" indices and pool-level series for fast, vectorized feature building.
3. *features.py → frames.py:* `build()` calls `make()` once per split (train / val_main / test_main / val_hold / test_hold) and keeps the function index `f`, bucket index `t`, label `y`, and cost for each row.
4. *frames.py → models:* every model receives rows from the **same frames**. Rules read a few columns; XGBoost reads all or some of the 57 columns; neural models also read raw history from the matrices to build sequences.
5. *models → preds files:* each model writes a probability (or 0/1 score) per row of val/test frames.
6. *preds → evaluation:* thresholds are chosen on validation rows and applied to test rows; metrics are computed per slice.
7. *preds + saved models → additional evaluation:* `extra_eval.py` reloads the saved models to score the training rows and runs paired tests; `calibration.py` checks probabilities.
8. *preds + thresholds + cache → pre-warming simulator:* `prewarm_sim.py` converts each model's predictions into pre-warming decisions and measures avoided cold starts, saved latency and wasted containers (Section 7.10).
9. *cache → robustness:* `robustness.py` rebuilds features for other time windows and trains XGBoost there; it does not use the saved predictions.

## 3.3 Components and responsibilities

| Component | File | Responsibility | Input | Output |
|---|---|---|---|---|
| Data cache builder | `v2/data.py` | CSV → matrices, static stats, roles | 31 CSVs | `matrices.npz` |
| Feature builder | `v2/features.py` | Vectorized, leakage-safe features | matrices, row index (f,t) | `X` (n × 57 or n × 10) |
| Frame builder | `v2/frames.py` | Time splits and row sets | `Ctx` | dict of arrays per split |
| GBDT + rules runner | `v2/run_gbdt.py` | Computes rules, trains 4 XGBoost models, writes frames side-file | frames | `preds_gbdt.npz`, `info_*.json`, `frames.npz` |
| XGBoost seeds runner | `v2/run_xgb_seeds.py` | Seeds 1 and 2 of `xgb_full` and `xgb_no_static` | frames | `preds_xgbseeds.npz` |
| Neural runner | `v2/run_deep.py` | Trains MLP/GRU/BiGRU/ABiGRU for one seed | matrices, frames | `preds_<model>_s<seed>.npz`, `info_*.json`, `models/<model>_s<seed>.pt` |
| Neural model library | `v2/nets.py` | `Net` (all four architectures), `Attention`, sequence builder `seq`, `standardize`, `save_checkpoint`, `load_checkpoint` | — | — |
| Prediction tool | `v2/predict.py` | Loads any saved model and scores a function over a day | `models/*`, matrices | printed predictions |
| Thresholds | `v2/save_thresholds.py` | Writes each model's validation-tuned threshold | preds, frames | `models/thresholds.json` |
| Multi-seed summary | `v2/multiseed.py` | Mean/sd across seeds | preds, frames | `results/multiseed_summary.md` |
| Evaluator | `v2/evalx.py` | Threshold search, metrics, bootstrap, cost capture | y, scores | metric dicts |
| Report | `v2/report.py`, `v2/final_table.py` | Slice tables, final test table | preds, frames | `benchmark.md`, `final_test_table.txt` |
| Attention layer | `v2/nets.py: Attention` (copied from the v1 class so that v2 is self-contained) | Additive attention used by v2's ABiGRU | hidden states | context vector, weights |
| Extra evaluation | `v2/extra_eval.py` | Slice confusion matrices, train/val/test metrics, paired bootstrap and McNemar tests | preds, frames, saved models | `results/extra_eval.md` |
| Calibration | `v2/calibration.py` | Brier score, ECE, reliability table, isotonic check | preds, frames | `results/calibration.md` |
| **Hybrid-histogram policy** | `v2/hh_policy.py` | Causal adaptation of the Azure hybrid-histogram keep-alive/pre-warm policy (`hh_warm`); an industry baseline for the simulator | event matrix, (f, t) rows | boolean warm decision per row |
| **Region preparation** | `v2/region_prep.py` | Aggregates one region's zip day by day into `cache/agg_R<n>/` and builds its cache | `R<n>.zip` | `agg_R<n>/*.pkl`, `matrices_R<n>.npz` |
| **Pre-warming simulator** | `v2/prewarm_sim.py` | Turns predictions into pre-warming decisions; compares keep-alive, hybrid-histogram, rule, threshold and fixed-budget policies | preds, frames, cache, `models/thresholds.json` | `results/prewarm_sim.md`, `figures/fig7_prewarm_tradeoff.png` |
| Robustness | `v2/robustness.py` | Rolling-origin evaluation and XGBoost hyperparameter search | cache | `results/robustness.md` |
| Figures | `v2/make_figures.py` | Figures 1–6 for the paper (figure 7 comes from the simulator) | preds, frames, info files | `figures/fig1–6` |

## 3.4 How the final prediction is generated

```
for each (function f, bucket t) in the test set:
    build features X[f,t]   from buckets < t only
    p = model(X[f,t])       # probability in [0,1]  (rules output 0 or 1)
    decision = 1 if p >= threshold else 0
```
The `threshold` is chosen **once per model and slice on the validation rows** to maximize F1 (`evalx.best_thr`). The final output is a probability and a yes/no "a cold start is expected in this bucket". No live platform acts on this decision. What acting on it would achieve is estimated offline by the **pre-warming simulator** (Section 7.10): a "yes" means "keep one container ready for this function in this bucket".

---

# 4. Detailed Explanation of Every Machine Learning Model

## 4.0 Model inventory and verified status

| Pipeline | Model | Implemented | Trained | Pretrained | Notes |
|---|---|---|---|---|---|
| v2 | Rule: last bucket | Yes | No training needed | No | `lag1` |
| v2 | Rule: same slot yesterday | Yes | n/a | No | `d1_+0` |
| v2 | Rule: same slot 7 days ago | Yes | n/a | No | `d7_+0` |
| v2 | Rule: yesterday OR 7 days ago | Yes | n/a | No | |
| v2 | Rule: 7-day majority | Yes | n/a | No | `slot_hits_7d ≥ 4` |
| v2 | Rule: any event in last hour | Yes | n/a | No | `roll12 > 0` |
| v2 | Score: pool hour rate | Yes | n/a | No | used only for the unseen-function comparison |
| v2 | XGBoost full (57 features) | Yes | **Trained**, 3 seeds | No | |
| v2 | XGBoost no-static (46) | Yes | **Trained**, 3 seeds | No | ablation |
| v2 | XGBoost no-pool (51) | Yes | **Trained**, 1 seed | No | ablation |
| v2 | XGBoost cold-newfunc (10) | Yes | **Trained**, 1 seed | No | unseen-function test |
| v2 | MLP (static features only) | Yes | **Trained**, 3 seeds | No | |
| v2 | GRU | Yes | **Trained**, 3 seeds | No | |
| v2 | BiGRU | Yes | **Trained**, 3 seeds | No | |
| v2 | ABiGRU | Yes | **Trained**, 3 seeds | No | |
| v1 | Logistic Regression, Random Forest, XGBoost, GRU, BiGRU, ABiGRU | Yes | **Smoke-trained only** (partially trained) | No | 2,048 rows, 5 functions, 1 deep epoch |
| — | Residual model, multi-task model, new-function embedding model, TCN/Transformer, calibration, cost-aware policy | **No** | No | — | **Proposed only** |

"Seed 0 complete" for the deep models means the first of three planned seeds finished; see Section 8.9 for the multi-seed status.

## 4.1 Common settings of the v2 tabular models

* **Training rows:** `train` frame (2,110,176 rows; roles 0 and 2; days 7–23).
* **Validation rows:** `val_main` (372,384 rows) for early stopping.
* **Target:** `y` (0/1).
* **Preprocessing:** none (trees) or standardization (neural).
* **Metrics:** Section 8.

## 4.2 Rule baselines

**Algorithm.** Fixed if-then rules with no learned parameters. Example: "predict a cold start if there was one at this time yesterday".
**Why included.** They set the minimum standard. If a learned model cannot beat them, it adds no value. They are also the reasoning basis of the project: they showed that most of the task is periodicity.
**Inputs.** One or a few feature columns (Section 2.11).
**Output.** 0 or 1. (Binary outputs have no probability ranking, so their ROC-AUC/PR-AUC are computed from two score values only.)
**Hyperparameters.** The only constant is the vote count 4 in "majority of 7 days".
**Training.** None. **Learned parameters.** None.
**Saved artifacts.** Predictions in `results/preds_gbdt.npz` under names `rule_*`.

| Rule | Definition |
|---|---|
| `rule_last_bucket` | `lag1` (event in the previous bucket) |
| `rule_same_bucket_1d` | `d1_+0` (same slot, 1 day ago) |
| `rule_same_bucket_7d` | `d7_+0` (same slot, 7 days ago) |
| `rule_1d_or_7d` | max(`d1_+0`, `d7_+0`) |
| `rule_majority_7d` | `slot_hits_7d ≥ 4` (at least 4 of the last 7 days had an event in this slot) |
| `rule_any_last_hour` | `roll12 > 0` |

## 4.3 XGBoost (four variants)

**Definition (simple).** XGBoost builds many small decision trees one after another; each new tree tries to correct the mistakes of the trees before it ("gradient boosting"). A decision tree is a flowchart of yes/no questions on feature values (for example "event yesterday? yes → event 7 days ago? …") that ends in a number. Adding all the trees' numbers produces a probability.
**Why selected.** Fast, strong on tables of engineered features, handles missing values and different scales, and gives feature importances. It is the standard strong baseline for tabular data.
**Problem solved here.** Estimate `P(y=1)` for a function-bucket from the 57 clues.

**Variants**

| Name | Features | Count | Purpose |
|---|---|---|---|
| `xgb_full` | all | 57 | Main tabular model |
| `xgb_no_static` | without groups 11–13 (`tr_*`, `fn_hour_rate`, `age_buckets`) | 46 | Does the function profile matter? |
| `xgb_no_pool` | without every feature whose name starts with `pool_` | 51 | Does pool context matter? (The filter in `run_gbdt.py` removes `pool_cpu`, `pool_mem`, `pool_hour_rate`, `pool_last1`, `pool_last12`, `pool_yday`; `cluster` is kept.) |
| `xgb_cold_newfunc` | clock + hardware + pool features | 10 | Unseen-function test |

> Verified from `results/info_*.json`: `xgb_full` 57 features, `xgb_no_static` 46, `xgb_no_pool` 51, `xgb_cold_newfunc` 10.

**Hyperparameters (identical for all variants, from `run_gbdt.py` / `run_xgb_seeds.py`):**

| Parameter | Value | Meaning |
|---|---|---|
| `n_estimators` | up to 1,500 | Maximum number of trees |
| `early_stopping_rounds` | 40 | Stop if validation score does not improve for 40 trees |
| `eval_metric` | `aucpr` | Validation score used for early stopping (area under precision-recall curve) |
| `max_depth` | 8 | Depth of each tree |
| `learning_rate` | 0.05 | Size of each correction step |
| `subsample` | 0.8 | Fraction of rows per tree |
| `colsample_bytree` | 0.8 | Fraction of columns per tree |
| `min_child_weight` | 5 | Minimum weight in a leaf |
| `reg_lambda` | 1.0 | L2 regularization |
| `tree_method` | `hist` | Histogram-based fast training |
| `n_jobs` | 10 | CPU threads |
| `random_state` | 42 (main run); 1 and 2 (seed runs) | Randomness control |
| class weighting | **none** | Probabilities stay calibrated-ish; the threshold handles imbalance |

**Learned parameters.** The tree structures (which feature and split value at each node) and the leaf values.
**Loss / optimizer.** Logistic loss with gradient boosting (XGBoost default for binary classification). **[The objective was not set explicitly; this is the library default `binary:logistic`.]**
**Training results (verified from saved metadata).**

| Variant | Best iteration (trees kept) | Wall time (incl. prediction) |
|---|---|---|
| `xgb_full` | 356 | 29 s |
| `xgb_no_static` | 356 | 26 s |
| `xgb_no_pool` | 219 | 15 s |
| `xgb_cold_newfunc` | 110 | 8 s |
| `xgb_full` seeds 1, 2 | 373, 350 | not timed separately |
| `xgb_no_static` seeds 1, 2 | 412, 336 | not timed separately |

**Feature importance for `xgb_full`** (saved in `results/info_xgb_full.json`; "importance" is XGBoost's default gain-based share):

| Feature | Importance |
|---|---|
| `slot_hits_7d` | 0.495 |
| `d1_+0` | 0.433 |
| `lag6` | 0.012 |
| `roll288` | 0.004 |
| `d2_+0`, `tr_event_buckets` | 0.003 each |
| all other features | ≤ 0.003 each |

**Reading it:** about 93% of the model's attention goes to "how many of the last 7 days had an event in this slot" and "was there an event at this time yesterday". The model is primarily a *learned, smoother version of the seasonal rule*.
For `xgb_cold_newfunc` the top features are `pool_yday` (0.240), `pool_mem` (0.127), `pool_hour_rate` (0.126), `pool_cpu` (0.125), `pool_last12` (0.118).

**Saved artifacts.** Fitted models in `v2/models/xgb_<variant>_s<seed>.json` (XGBoost's portable native format; seed 42 is the main run, seeds 1 and 2 exist for `xgb_full` and `xgb_no_static`), the feature list each model expects in `v2/models/xgb_<variant>_features.json`, metadata in `results/info_xgb_*.json` (best iteration, parameters, importances), and predictions in `results/preds_gbdt.npz` and `results/preds_xgbseeds.npz`. XGBoost training here is deterministic: re-running produced identical predictions (difference 0.0).
**How output is used.** Written as a probability per row; read by the evaluator. No other model consumes it.

## 4.4 Neural models — shared design

All four neural models are built in **one class, `Net`, in `v2/nets.py`** (trained by `v2/run_deep.py`), parameterized by `kind ∈ {mlp_static, gru, bigru, abigru}`.

**Inputs (two parts).**
1. **Static/tabular branch:** the 57 standardized features → `Linear(57,128) → ReLU → Dropout(0.2) → Linear(128,64) → ReLU` → a 64-number summary.
2. **Sequence branch** (absent in `mlp_static`): a window of the **last 24 buckets** (2 hours), each step having 7 channels:

| Channel | Meaning |
|---|---|
| 1 | Event in that bucket (0/1) |
| 2 | log(1 + cost) in that bucket |
| 3, 4 | Sine and cosine of the time of day of that bucket |
| 5 | Event in the *same bucket one day earlier* (aligned to that step) |
| 6 | Event in the *same bucket seven days earlier* |
| 7 | Pool activity ratio at that bucket (events in the pool divided by pool size) |

Channels 5 and 6 let a 2-hour window see daily and weekly rhythm. The tensor has shape `(batch, 24, 7)` and is built on the Apple GPU directly from the matrices by `seq(f, t)`.

**Head.** The two branch outputs are concatenated and passed through `Dropout(0.3) → Linear(64 + enc, 64) → ReLU → Linear(64, 1)`, producing one **logit**; a sigmoid turns it into a probability.

**Training (all four):**

| Setting | Value |
|---|---|
| Loss | `binary_cross_entropy_with_logits` (no class weights) |
| Optimizer | RMSprop, learning rate 0.001, alpha 0.9, eps 1e-8 (the paper's optimizer and values) |
| Gradient clipping | max norm 1.0 |
| Batch size | 2,048 |
| Samples per epoch | 700,000 rows drawn at random from the 2,110,176 training rows |
| Max epochs | 14 |
| Early stopping | patience 3 epochs on validation PR-AUC; best weights are restored |
| LR scheduler | `ReduceLROnPlateau` (mode max, factor 0.5, patience 1) on validation PR-AUC |
| Dropout | 0.3 between GRU layers and in the head; 0.2 in the static branch |
| Device | NVIDIA GPU (CUDA) if available, else Apple GPU (MPS), else CPU; override with the environment variable `COLDSTART_DEVICE=cpu|cuda|mps`. The reported runs used MPS. **[CUDA path not tested: no NVIDIA GPU was available.]** |
| Seed | set by command-line argument |
| Pretrained weights | none |

**Learned parameters.** All weights and biases of the linear layers and GRU gates; for attention, its two linear maps `W` and `v`.
**Initialization.** PyTorch defaults (the archived v1 `ABiGRU` class uses Xavier/orthogonal initialization; v2 does not use that class).

### 4.4.1 MLP (`mlp_static`)
* **Definition.** A multi-layer perceptron: layers of weighted sums with non-linear activations. No notion of time order.
* **Why.** An **ablation**: it receives exactly the same 57 clues as XGBoost but no sequence. If the sequence models do not beat it, the sequence adds nothing.
* **Parameters:** 19,905. **Epochs run:** 11. **First-epoch validation PR-AUC** 0.9779, best 0.9846. Training time about 17 s.

### 4.4.2 GRU
* **Definition.** A *Gated Recurrent Unit* reads the sequence step by step and keeps a memory vector; "update" and "reset" gates decide what to remember and what to forget.
* **Why.** The standard sequence baseline; included in the paper's comparison family.
* **Architecture here.** 2-layer GRU, hidden size 64, unidirectional. The **final hidden state** (64 numbers) is concatenated with the static branch.
* **Parameters:** 62,977. **Epochs run:** 13. First-epoch validation PR-AUC 0.9782, best 0.9847. Training time about 392 s.

### 4.4.3 BiGRU
* **Definition.** A bidirectional GRU runs one GRU from oldest to newest and another from newest to oldest over the same window, then combines both.
* **Why.** Gives each position context from both directions *inside the observed window*. There is **no future leakage** because the window contains only buckets before t.
* **Architecture.** 2-layer bidirectional GRU, hidden 64 per direction; the final forward and backward states (128 numbers) are concatenated.
* **Parameters:** 130,625. **Epochs run:** 14 (reached the cap of 14, so training was not stopped by early stopping). First-epoch validation PR-AUC 0.9785, best 0.9852. Training time about 868 s.

### 4.4.4 ABiGRU (attention-based BiGRU; the paper's proposed architecture, adapted)
* **Definition.** A BiGRU plus (a) an **attention** layer that scores each of the 24 time steps and takes a weighted average, and (b) **max-pooling** that keeps the strongest value of each memory slot over the 24 steps.
* **Attention formula implemented** (`v2/nets.py: Attention`): `score_t = vᵀ·tanh(W·h_t + b)`, `α = softmax(score)`, `context = Σ α_t · h_t`, with attention dimension 32.
* **Architecture here.** BiGRU outputs `(batch,24,128)` → attention context `(batch,128)` and global max-pool over time `(batch,128)` → concatenated `(batch,256)` → joined with the 64-number static summary → head.
* **Parameters:** 142,977. **Epochs run:** 13. First-epoch validation PR-AUC 0.9790, best 0.9856. Training time about 908 s (through epoch 13).
* **Differences from the paper's equations (stated honestly).** The paper (Eqs. 7–12) *sums* forward and backward outputs element-wise, applies a different attention formulation (`u_t = tanh(W h + b)`, `α = softmax(u_tᵀ u_w)`), weights hidden states element-wise (`ν = α ⊙ h`), and max-pools over a sliding `k1×k2` window. This implementation *concatenates* directions, uses a standard additive attention weighted sum, and max-pools globally over time on the raw BiGRU outputs. It is an **adaptation, not an exact reproduction**. Several architectural details are not specified in the paper, so exact reproduction cannot be verified.

### 4.4.5 Evaluation, saved artifacts and use of outputs (neural models)
* **Validation/testing procedure:** the best epoch (by validation PR-AUC) is restored, predictions are made on `val_main`, `test_main`, `val_hold`, `test_hold`, and saved to `results/preds_<model>_s<seed>.npz`; the threshold is then chosen on validation rows by the evaluator.
* **Saved artifacts:** predictions (`preds_*.npz`), `info_<model>_s<seed>.json` (training history per epoch, parameter count, best validation PR-AUC) and a checkpoint `v2/models/<model>_s<seed>.pt` containing the weights, the standardization vectors `mu`/`sd`, the 57 feature names in order, the window length, seed, training history and torch version. Checkpoints exist for all three seeds of all four models (12 files). Runs that originally finished before saving was added were retrained on the same machine with the same seed; every retrained run reproduced its original predictions exactly (difference 0.0), so each saved checkpoint is exactly the model behind the reported numbers.
* **How output is used:** only by `report.py` / `final_table.py`.

## 4.5 The v1 models (smoke-trained only)

Archived in `archive_v1/src/models/` and trained by `archive_v1/main.py`:

| Model | Implementation | Key settings | Smoke-run training |
|---|---|---|---|
| Logistic Regression | `train_logistic_regression` | C=1.0, balanced class weights, lbfgs, max_iter 1000; 204 input features (12×17 flattened) | trained on 2,048 rows |
| Random Forest | `train_random_forest` | up to 200 trees (25 in smoke), depth 15, min_split 10, min_leaf 5, balanced | 25 trees |
| XGBoost | `train_xgboost` | up to 300 trees (15 in smoke), depth 8, lr 0.05, `scale_pos_weight`, early stopping 15 | 15 trees |
| GRU | `StandardGRU` | hidden 64, 2 layers, dropout 0.3; 40,961 params | 1 epoch |
| BiGRU | `BiGRU` | same, bidirectional; 106,497 params | 1 epoch |
| ABiGRU | `ABiGRU` | attention dim 32, FC 64, Xavier/orthogonal init; 127,041 params | 1 epoch |

All v1 deep models use RMSprop (lr 0.001, alpha 0.9), `BCEWithLogitsLoss` with `pos_weight = n_neg/n_pos`, batch 256, up to 50 epochs and patience 7 (in a full run; the smoke run used 1 epoch). Verified from the checkpoints: `best_epoch 0`, history length 1, for all three. Artifacts: `experiments/results/models/*.joblib`, `*.pt`, `*_history.json`. **These are not benchmark results.**

## 4.6 The unseen-function model (`xgb_cold_newfunc`)

* **Inputs:** 10 features (clock, hardware, pool profile, pool live load). **No own-history features.**
* **Training rows:** same training frame as other models (the model sees hardware and pool context of training functions).
* **Evaluation:** on the 96 **holdout functions** over test days.
* **Result:** F1 0.239, ROC-AUC 0.567, PR-AUC 0.149 (Section 8.5). The pool hour-rate baseline gets ROC-AUC 0.504. **The approach does not work.**
* **Why it is kept:** a negative result that bounds what hardware and pool context can explain.

---

# 5. Model Interconnection and Information Exchange

## 5.1 Verified result: the models are independent

I searched the source (v1 `main.py` and `src/`, now in `archive_v1/`, and `v2/*.py`) for stacking, ensembling, voting, blending and averaging of model outputs. **No model's output is used as input to any other model.** The only matches were `numpy.stack` / `torch.stack` calls for building arrays, and an import of `RandomForestClassifier`.

Therefore:
* The models run **in parallel in the logical sense (independently)**: each is trained alone on the same data and evaluated alone.
* There is **no ensemble**, no cascade, no meta-learner.
* In practice they were run **one after another** (compute reasons), not simultaneously.

## 5.2 What is shared (these are not "connections between models")

| Shared item | Producer | Consumers | Format |
|---|---|---|---|
| Cached matrices | `data.py` | `features.py`, `nets.py` (reads `ctx.ev`, `ctx.cost`, `ctx.P`) | NumPy arrays `(527, 8928)` |
| Row frames (`X`, `f`, `t`, `y`) | `frames.py`/`features.py` | all models and the evaluator | arrays |
| Standardization statistics | `run_deep.py` | the neural model of that run only | mean, std vectors (saved inside its checkpoint) |
| Validation scores | each model | `evalx.best_thr` for that model only | float32 arrays |

## 5.3 Information flow *inside* one neural model (the only "interconnection" that exists)

```
                 ┌──────────────── 57 standardized features ───────────────┐
                 │                                                         │
                 ▼                                                         │
        Static branch (MLP) ──► 64-number summary ──────────────┐          │
                                                                │          │
 24 × 7 sequence ─► GRU / BiGRU ─► hidden states (24 × H)       │          │
                                   │                            ▼          │
                                   ├─ final state (GRU,BiGRU) ─► concat ─► Head ─► logit ─► sigmoid ─► p
                                   └─ attention context + max-pool (ABiGRU) ─┘
```
* **Sender → receiver:** static branch → head; sequence encoder → head.
* **Exchanged information:** dense feature vectors (embeddings in the loose sense of learned numeric summaries), 64 numbers from the static branch and 64/128/256 numbers from the sequence encoder.
* **Effect:** the head combines "what is this function's tabular profile" with "what has been happening in the last 2 hours" to produce one logit.
* **Evidence of effect:** the MLP (static branch only) and the three sequence variants end within 0.002 F1 of each other, so the sequence branch contributes little measurable information beyond the static branch **in this data**.

## 5.4 Model interaction diagram

```
   Shared feature frames (57 features, same rows, same splits)
        │              │                 │                  │
        ▼              ▼                 ▼                  ▼
     RULES         XGBoost (×4)     MLP / GRU / BiGRU / ABiGRU
        │              │                 │
        ▼              ▼                 ▼
    0/1 score    probability       probability      (no arrows between the three)
        └──────────────┴─────────────────┘
                       ▼
          Evaluator: per-model threshold (validation) → test metrics
                       ▼
               side-by-side comparison tables
```
The comparison happens only in `report.py`, which reads the saved predictions of every model but never feeds one model's output to another.

---

# 6. Complete End-to-End Execution Workflow

The workflow is described for the v2 pipeline (the v1 pipeline is described in 6.7).

## 6.1 Chronological steps

| # | Step | What happens | Why | File / function | Input | Output |
|---|---|---|---|---|---|---|
| 1 | Dataset ingestion | The data is downloaded manually from the provider and unzipped to `dataset/raw/region5_cold_starts/R5/` | Obtain the trace | manual (the provider's link is in `_github_repo/README_data_release_2025.md`) | zip | 31 CSVs |
| 2 | Data loading | All 31 CSVs are concatenated | One table | `v2/data.py: build()` | CSVs | DataFrame of 805,745 rows |
| 3 | Validation | pandas loads types; v2 relies on prior checks (no missing values, verified). The v1 loader also checks time vs day. | Detect broken files | `archive_v1/src/data/loader.py` (v1) | | |
| 4 | Cleaning | Function identity built; pool name parsed from `podID`; no rows dropped | Define "the same function" | `data.py` | | `fid` column |
| 5 | Feature extraction | Rows are bucketed (`time // 300`) and aggregated to count and cost matrices | Time-series representation | `data.py` | | `cnt`, `cost` (527 × 8,928) |
| 6 | Feature engineering | Static stats (days 0–6), pool profiles, roles, holdout selection; at model time: lags, seasonal lags, rolling counts, pool load | Informative, leakage-safe clues | `data.py`, `features.py: Ctx, make` | matrices | `X` (n × 57) |
| 7 | Data transformation | Neural models standardize features and build 24×7 sequences | Stable training | `nets.py: standardize, seq` | `X`, matrices | tensors |
| 8 | Model training | XGBoost fits with early stopping; neural nets train with RMSprop | Learn the mapping | `run_gbdt.py`, `run_xgb_seeds.py`, `run_deep.py` | training frame | fitted model (in memory) |
| 9 | Model saving | Models, predictions and training info are saved | Reuse and reproducibility | `run_gbdt.py`, `run_xgb_seeds.py` (XGBoost `.json`), `nets.save_checkpoint` (neural `.pt`) | fitted model | `v2/models/*`, `preds_*.npz`, `info_*.json` |
| 10 | Model loading | A saved model is rebuilt from its file | Inference without retraining | `nets.load_checkpoint`, `predict.make_scorer` | `v2/models/*` | ready model |
| 11 | Input preparation | Test rows are built by `make()` and standardized with training statistics | Same format as training | `frames.py` | | `test_main` frame |
| 12 | Prediction generation | Each model produces a probability per row | | `predict_proba` / `predict(...)` | | arrays |
| 13 | Model-to-model exchange | **None** (Section 5) | | | | |
| 14 | Result processing | Thresholds chosen on validation, applied to test; slice masks; bootstrap; cost capture | Fair comparison | `evalx.py`, `report.py` | preds + frames | metric dicts |
| 15 | Final output | Markdown/text benchmark tables | Read by humans | `report.py`, `final_table.py` | | `benchmark.md`, `final_test_table.txt` |
| 16 | Evaluation | Metrics per slice (Section 8) | Answer the research questions | | | |
| 17 | Additional evaluation | Slice confusion matrices, train metrics, paired tests, calibration, rolling-origin evaluation, XGBoost tuning | Check that the conclusions are robust | `extra_eval.py`, `calibration.py`, `robustness.py` | preds, frames, saved models, cache | `extra_eval.md`, `calibration.md`, `robustness.md` |
| 18 | Pre-warming simulation | Predictions become per-bucket pre-warming decisions; avoided cold starts, saved latency and waste are counted | Measure the operational value of the forecasts | `prewarm_sim.py` | preds, thresholds, cache | `prewarm_sim.md`, `fig7` |

## 6.2 Pseudocode of the whole run

```
matrices, roles, static_stats = data.build(CSVs)             # data.py
ctx = Ctx(matrices)                                          # features.py
frames = {train, val_main, test_main, val_hold, test_hold}   # frames.py (time splits)
for each frame: X, y = make(ctx, functions, t_lo, t_hi)      # features.py

for rule in RULES:  preds[rule] = rule(X)                    # run_gbdt.py
for xgb_variant in (full, no_static, no_pool, cold):
    model = XGBClassifier(...).fit(X_train, y_train, eval_set=val)
    preds[xgb_variant] = model.predict_proba(X_val/test)
for kind in (mlp_static, gru, bigru, abigru):                # run_deep.py, per seed
    net = Net(kind);  train up to 14 epochs, keep best val PR-AUC
    preds[kind] = sigmoid(net(...))

for model in preds:                                          # report.py
    thr = argmax_F1(y_val, preds[model].val)
    metrics(y_test, preds[model].test >= thr)

for policy in (keep-alive, rules, model @ threshold, top-N per bucket):   # prewarm_sim.py
    warm[f,t] = policy(...)                     # one container for function f in bucket t
    avoided   = warm & (cold start happens in bucket t)
    wasted    = warm & (no cold start in bucket t)
    report % cold starts avoided, latency saved, % wasted, idle container-hours
```

## 6.3 Key code references

* Bucket creation: `v2/data.py` (`df["b"] = (df.time // B).astype(int)`).
* Feature builder: `v2/features.py: make`.
* Time splits: `v2/frames.py: frame_defs` (uses `T0 = 7*PD`, `TRAIN_END`, `VAL_END`).
* Sequence construction: `v2/nets.py: seq` (gathers `ev`, `lc`, `hs`, `hc`, aligned `ev` at −288 and −2016, and pool ratio).
* Threshold search: `v2/evalx.py: best_thr`.

## 6.4 Why this ordering

Data → features → split → train → threshold on validation → test last. The ordering guarantees that the test set influences nothing: not the scaler, not the early stopping, not the threshold.

## 6.5 Leakage test that was run

The future was zeroed for one sampled function from bucket t onward (events, counts, costs, cumulative sums, last-event indices), the features of that row were recomputed, and compared with the originals: **maximum absolute difference 0.0**.

## 6.6 Execution commands

See Section 13.5.

## 6.7 The v1 (original) workflow, for reference

`main.py` runs six phases: (1) load data, (2) EDA plots, (3) feature engineering → `dataset/processed/processed_data.npz` (17 features × 12 buckets, top 500 functions with ≥ 10 train events), (4) baselines, (5) deep models, (6) comparison and plots. Only the `--smoke-test` mode was executed to completion (`archive_v1/experiments/results/smoke_run.log`). An earlier full run stopped during feature building (at function 1 of 489; projected ~21 minutes); its log and the empty `dataset/processed/` folder were deleted during cleanup.

---

# 7. Cold-Start Detection and Prediction Mechanism

## 7.1 How a cold start is identified

In this data set **a row in the file is a cold start**. There is no detection algorithm; the provider's logging already labelled each event. Our project does not classify individual requests.

## 7.2 What is predicted

The *occurrence* of at least one cold-start row for a given function in a given future 5-minute bucket (Section 1.4).

## 7.3 What data identifies the prediction

* The function's own recent and seasonal cold-start history.
* The time of day.
* Pool-level activity and hardware context.
* Early-period function statistics.

## 7.4 How the models contribute

Each model independently maps the same inputs to a probability. XGBoost combines the features via boosted trees. The neural models combine the 57 features and the 24×7 history window. The rules apply fixed seasonal logic.

## 7.5 Final decision

`predicted_cold_start = (probability ≥ threshold)`, with the threshold tuned on validation rows to maximize F1 (for each model and each evaluation slice). **Implemented:** the decision, and an offline simulation of acting on it (Section 7.10). **Not implemented:** a live action in a real platform (pre-warming, provisioned-concurrency adjustment).

## 7.6 Insufficient history

* For buckets with no earlier events, lag features are 0 and `since_last` is set to the cap (2,016).
* Functions with fewer than 10 events in days 0–23 are the "low-history" group. They are **included in training** and reported separately.
* Result: low-history F1 about 0.27–0.34 (91 test positives, very wide confidence intervals), compared with about 0.24 for rules.

## 7.7 Unseen functions

* **Holdout functions (96) are completely absent from training.** When the model may use the function's own history at inference time, F1 is about 0.97 (the history is available, only the training data lacked them).
* **A function with no history at all** is represented by the 10-feature cold model: F1 0.239, ROC-AUC 0.567 — **not solved**.

## 7.8 Difference from conventional approaches

| Conventional (e.g. the reference paper) | This project |
|---|---|
| Predicts per invocation whether it is cold (needs invocation-level data) | Predicts per function per 5-minute bucket whether a cold start will occur (works with event-only data) |
| Reports one accuracy on one split | Reports slices, intervals, seeds, simple-rule baselines |
| Compares against other papers' numbers | Compares against rules and ablations on identical rows |

## 7.9 Implemented vs proposed

| Implemented | Proposed (not implemented) |
|---|---|
| Occurrence forecasting with 7 rules, 4 XGBoost variants, 4 neural models | Residual-on-seasonal-baseline model |
| Slice evaluation, bootstrap CIs | Multi-task head (event, time-to-next, cost) |
| Cost-captured proxy metric; pre-warming simulator with threshold and per-bucket budget policies (`v2/prewarm_sim.py`) | |
| Rolling-origin evaluation, XGBoost tuning, paired tests (`v2/robustness.py`, `v2/extra_eval.py`) | Shared function-embedding model for new functions |
| Probability calibration check (`v2/calibration.py`) | TCN / Transformer comparison |
| | Use of provider's request / time-series tables |

## 7.10 From prediction to action: the pre-warming simulator

**The story.** Go back to the pizza shop. The forecast says "a customer will probably arrive at 3:05". The simulator answers the next question: *if the cook heats the oven at 3:00 every time the forecast says so, how many customers avoid waiting, and how often is the oven heated for nobody?*

**What it does** (`v2/prewarm_sim.py`). It replays the test days 27–30 for the 385 eligible functions, bucket by bucket (1,152 five-minute buckets).
1. Before bucket *t*, a **policy** decides for each function whether to keep **one** container ready ("pre-warm") for bucket *t*.
2. If the function then has a cold start in bucket *t*, the pre-warm **avoids its first cold start**, and the latency saved is that bucket's average cold-start time (total cost ÷ number of cold starts).
3. Further cold starts in the same bucket are not avoided: one container serves one cold start.
4. If no cold start happens, the pre-warm is **wasted**: one container idles for 5 minutes.

**Policies compared.**

| Policy | Rule for pre-warming function *f* in bucket *t* |
|---|---|
| No pre-warming | never |
| Keep-alive 15 / 60 min | a cold start happened in the last 3 / 12 buckets (what platforms do today with longer keep-alive) |
| Hybrid histogram (Azure, ATC'20) | `L` = last cold-start bucket before *t*, gap `g = t − L`; warm iff `pw ≤ g ≤ ka_end`, where the window comes from the function's own gap histogram (see below) |
| Rule: same slot yesterday | a cold start in bucket *t* − 288 |
| Rule: 7-day majority | cold starts in this slot on at least 4 of the last 7 days |
| Model @ threshold | model probability ≥ its validation-tuned threshold (`models/thresholds.json`) |
| Fixed budget N | the N highest-scoring functions in every bucket |

**The hybrid-histogram baseline (`v2/hh_policy.py`).** Shahrad et al., *Serverless in the Wild* (USENIX ATC 2020) keep a histogram of idle times per application (1-minute bins, 4-hour range). If the histogram is representative (coefficient of variation of bin counts > 2) the container is unloaded after use, pre-warmed at the 5th percentile of idle times (minus 10%) and kept until the 99th percentile (plus 10%); if many idle times fall outside the range an ARIMA forecast sets the window (85% / 115% of the prediction); otherwise a fixed keep-alive of the full range is used. Adaptations for this trace, all stated in the code: (1) the trace holds only cold starts, so the "idle time" is the gap between consecutive buckets that contain a cold start, not between invocations; (2) bins are 5-minute buckets and the range is 48 buckets (4 h); (3) the ARIMA branch is replaced by the median of the last 5 out-of-range gaps (statsmodels is not in the pipeline); (4) strictly causal: parameters used after event *E[i]* are computed from gaps that ended at or before *E[i]*. Two variants are simulated: fallback keep-alive 4 h (as in the paper) and 60 min. Branch use over the test period: fixed keep-alive 4 functions, histogram 124, out-of-range forecaster 260. Because the window it opens is long, it behaves like a keep-alive: recall 0.996 but precision 0.398 (F1 0.568).

**How the models contribute.** Each model's saved test probabilities are the only input that differs between model policies, so the simulator compares models by their operational value, not just by F1.

**Main result** (full table in Section 8.12.3): at about 50 pre-warms per bucket, ABiGRU avoids 67.4% and XGBoost 66.4% of cold starts with 7.1% / 5.9% waste; the 7-day rule avoids 64.4% with 7.7% waste; a 60-minute keep-alive avoids 68.8% but wastes 58.9% (6,747 idle container-hours against about 300), and the hybrid histogram avoids 70.1% but wastes 60.2% (7,263 idle container-hours, about 126 containers per bucket). The ceiling for one container per function-bucket is 70.4% of cold starts.

**Assumptions and limits.** A container started at the end of bucket *t* − 1 is ready in bucket *t* (the mean cold start is about 0.9 s). Because the trace contains only cold starts, the simulator cannot model a pre-warmed container also serving later requests, or requests routed to another cluster, so the avoided counts are conservative. The platform's own keep-alive is not changed. These results are an offline estimate, not a measurement on a live platform.

---

# 8. Model Evaluation and Experimental Results

## 8.1 Evaluation protocol

* **Threshold:** for each model and slice, the F1-maximizing threshold on the *corresponding validation slice* (`evalx.best_thr` scans 400 quantile thresholds between the 50th and 99.95th percentile of validation scores; for binary rule outputs it uses 0.5; for slices with fewer than 5 validation positives it falls back to the whole validation set).
* **Test:** days 27–30, never used for tuning.
* **Confidence intervals:** cluster bootstrap over functions (whole functions are resampled, because rows of one function are correlated). 100 resamples for the main table.
* **Cost captured @2%:** fraction of the slice's total cold-start seconds that fall in the 2% highest-scored function-buckets. It is a proxy for "how much waiting time a warming budget of 2% of cells could cover". For rules, which only output 0/1, ties are broken arbitrarily, so this number is not directly comparable for rules.

## 8.1a What each metric means and why it matters

| Metric | Meaning | Why here |
|---|---|---|
| Accuracy | Share of correct yes/no answers | Misleading at 13% positives: "always no" scores 0.8698. Reported but not the headline |
| Precision | Of the buckets we flagged, the share that really had a cold start | Low precision = wasted pre-warming |
| Recall | Of the real cold-start buckets, the share we flagged | Low recall = missed cold starts |
| F1 | Harmonic mean of precision and recall | Single balance score |
| MCC | Correlation between predictions and truth, −1…1 | Fair under class imbalance |
| PR-AUC | Area under the precision–recall curve (threshold-free ranking quality) | Best summary for rare positives |
| ROC-AUC | Area under the true-positive vs false-positive curve | Ranking quality; can look optimistic when negatives dominate |
| Confusion matrix | Counts of TP, FP, FN, TN | Shows the type of error |

## 8.2 Main benchmark — test days 27–30, eligible functions (role 0)

Rows: 443,520. Positives: 57,739 (13.0%). "Always no" accuracy: 0.8698. Source: `v2/results/final_test_table.txt` (seed-0 deep models; XGBoost with its main seed 42).

| Model | Threshold | Accuracy | Precision | Recall | F1 | MCC | PR-AUC | ROC-AUC | Val F1 |
|---|---|---|---|---|---|---|---|---|---|
| ABiGRU | 0.437 | 0.9849 | 0.9293 | 0.9571 | **0.9430** | 0.9345 | 0.9889 | 0.9981 | 0.9393 |
| XGBoost no-static | 0.478 | 0.9850 | 0.9346 | 0.9515 | **0.9430** | 0.9344 | 0.9881 | 0.9980 | 0.9393 |
| XGBoost no-pool | 0.505 | 0.9849 | 0.9378 | 0.9469 | 0.9423 | 0.9336 | 0.9872 | 0.9978 | 0.9403 |
| XGBoost full | 0.538 | 0.9849 | 0.9413 | 0.9427 | 0.9420 | 0.9333 | 0.9875 | 0.9978 | 0.9399 |
| BiGRU | 0.518 | 0.9848 | 0.9351 | 0.9490 | 0.9420 | 0.9333 | 0.9885 | 0.9981 | 0.9395 |
| GRU | 0.493 | 0.9846 | 0.9310 | 0.9527 | 0.9417 | 0.9329 | 0.9878 | 0.9980 | 0.9396 |
| MLP (no sequence) | 0.510 | 0.9845 | 0.9345 | 0.9474 | 0.9409 | 0.9320 | 0.9873 | 0.9979 | 0.9390 |
| Rule: 7-day majority | 0.5 | 0.9791 | 0.9233 | 0.9156 | 0.9194 | 0.9074 | 0.8564 | 0.9521 | 0.9123 |
| Rule: yesterday OR 7 days ago | 0.5 | 0.9747 | 0.8678 | 0.9505 | 0.9073 | 0.8939 | 0.8313 | 0.9644 | 0.8966 |
| Rule: same slot yesterday | 0.5 | 0.9756 | 0.9059 | 0.9071 | 0.9065 | 0.8925 | 0.8338 | 0.9465 | 0.8874 |
| Rule: same slot 7 days ago | 0.5 | 0.9743 | 0.9036 | 0.8988 | 0.9012 | 0.8864 | 0.8253 | 0.9422 | 0.8987 |
| Rule: any event last hour | 0.5 | 0.8145 | 0.4108 | 0.9777 | 0.5785 | 0.5588 | 0.4046 | 0.8839 | 0.5747 |
| Rule: last bucket | 0.5 | 0.8636 | 0.4761 | 0.4761 | 0.4761 | 0.3977 | 0.2949 | 0.6989 | 0.4705 |

Bootstrap 95% F1 intervals (`v2/results/benchmark.md`): ABiGRU [0.926, 0.960], XGBoost full [0.924, 0.958], 7-day majority rule [0.891, 0.944].

## 8.3 Confusion matrices (test, role 0, seed-0 deep models)

| Model | TP | FP | FN | TN |
|---|---|---|---|---|
| ABiGRU | 55,263 | 4,204 | 2,476 | 381,577 |
| BiGRU | 54,795 | 3,805 | 2,944 | 381,976 |
| GRU | 55,007 | 4,079 | 2,732 | 381,702 |
| MLP | 54,704 | 3,837 | 3,035 | 381,944 |
| XGBoost full | 54,432 | 3,397 | 3,307 | 382,384 |
| XGBoost no-static | 54,940 | 3,847 | 2,799 | 381,934 |
| Rule 7-day majority | 52,864 | 4,389 | 4,875 | 381,392 |
| Rule same slot yesterday | 52,373 | 5,442 | 5,366 | 380,339 |

TP = predicted yes and a cold start occurred; FP = predicted yes but none occurred; FN = missed cold start; TN = correctly predicted none. Confusion matrices for the slices (reactivation, low-history, holdout) are in `v2/results/extra_eval.md`, Section 1 (see Section 8.12.1).

## 8.4 Hard slices (test days 27–30)

**Reactivation** — role-0 buckets with no event in the previous hour (306,100 rows, 1,285 positives; 0.4% positive):

| Model | F1 [95% CI] | Precision | Recall | PR-AUC |
|---|---|---|---|---|
| Rule: 7-day majority | 0.870 [0.820, 0.918] | 0.900 | 0.842 | 0.759 |
| XGBoost no-static | 0.861 [0.802, 0.909] | 0.854 | 0.868 | 0.920 |
| XGBoost no-pool | 0.853 [0.792, 0.902] | 0.837 | 0.870 | 0.926 |
| XGBoost full | 0.845 [0.778, 0.897] | 0.824 | 0.866 | 0.926 |
| ABiGRU | 0.837 [0.771, 0.901] | 0.813 | 0.864 | 0.915 |
| MLP | 0.833 [0.769, 0.896] | 0.808 | 0.860 | 0.912 |
| BiGRU | 0.831 [0.760, 0.897] | 0.807 | 0.857 | 0.916 |
| GRU | 0.827 [0.757, 0.897] | 0.797 | 0.859 | 0.912 |
| Rule: same slot yesterday | 0.803 [0.735, 0.879] | 0.763 | 0.848 | 0.648 |

**Reading:** the learned models rank events much better (PR-AUC 0.91–0.93 vs 0.76), but at the validation-chosen threshold the 7-day majority rule has the highest F1; the intervals overlap, so the difference is not conclusive.

**Low-history functions** (52,992 rows, **91 positives**): best F1 0.343 (XGBoost no-static, CI [0.250, 0.607]); GRU 0.318, ABiGRU 0.313, XGBoost full 0.309, BiGRU 0.290, MLP 0.269; best rule 0.243 (CI [0.025, 0.708]). With 91 positives the intervals are very wide.

## 8.5 Unseen (holdout) functions (96 functions; 110,592 rows; 14,085 positives)

| Model | Uses own history? | F1 [95% CI] | MCC | PR-AUC | ROC-AUC |
|---|---|---|---|---|---|
| XGBoost no-pool | yes | 0.970 [0.934, 0.994] | 0.966 | 0.996 | 0.999 |
| BiGRU, GRU, ABiGRU, MLP, XGBoost full (all) | yes | 0.969 | 0.964–0.965 | 0.996–0.997 | 0.999 |
| Rule: 7-day majority | yes | 0.948 [0.900, 0.987] | 0.940 | 0.904 | 0.975 |
| Rule: same slot yesterday | yes | 0.943 | 0.935 | 0.897 | 0.971 |
| **XGBoost cold (no own history)** | **no** | **0.239 [0.175, 0.300]** | 0.068 | 0.149 | 0.567 |
| Pool hour-rate score | no | 0.207 [0.131, 0.270] | 0.005 | 0.158 | 0.504 |

**Reading:** functions never seen in *training* are predicted well when their own recent history is available at inference (which is the normal situation after a few days of operation). Predicting a **new function with no history** is not solved.

## 8.6 Ablations

| Question | Comparison | Finding |
|---|---|---|
| Does the function profile help? | `xgb_full` vs `xgb_no_static` (3 seeds each) | No: F1 0.9426 ± 0.0005 vs 0.9427 ± 0.0002 |
| Does pool context help (with own history)? | `xgb_full` vs `xgb_no_pool` (1 seed) | No measurable change (F1 0.9420 vs 0.9423) |
| Does a sequence encoder help? | MLP vs GRU / BiGRU / ABiGRU (seed 0) | No measurable change (F1 0.9409 vs 0.9417–0.9430) |
| Does the seasonal history matter? | Feature importance and rule results | **Yes**: dominant (Section 4.3) |

## 8.7 Training and validation performance

* **Neural models** (`results/info_*_s0.json`): epochs run, parameters, first-epoch and best validation PR-AUC, final training loss.

| Model | Epochs | Parameters | Final train loss | First-epoch val PR-AUC | Best val PR-AUC |
|---|---|---|---|---|---|
| MLP | 11 | 19,905 | 0.0246 | 0.9779 | 0.9846 |
| GRU | 13 | 62,977 | 0.0240 | 0.9782 | 0.9847 |
| BiGRU | 14 | 130,625 | 0.0234 | 0.9785 | 0.9852 |
| ABiGRU | 13 | 142,977 | 0.0234 | 0.9790 | 0.9856 |

* **Training-set classification metrics** (accuracy/F1 on training rows) were added on 2026-11-19 by `v2/extra_eval.py` (Section 8.12.1): train F1 0.959–0.966, validation 0.939–0.940, test 0.941–0.943. The rule, which learns nothing, shows a similar train–test gap (0.019), so the gap reflects harder later days rather than overfitting.
* **Per-epoch curves** are stored in `results/info_*_s*.json` and plotted in `v2/figures/fig6_training_curves.png` (seed 0) and `fig8_training_all_seeds.png` (all 12 runs) (Section 8.11).
* **Training evidence for all 12 neural runs** (144 epochs in total) is collected in **`v2/results/training_report.md`**: epochs run, kept epoch, stopping reason, loss and validation curves, parameters, wall-clock training time, XGBoost tree counts, and SHA-256 fingerprints of every saved model. The console output printed during training is kept in `v2/results/logs/` (local paths removed). `v2/training_report.py` cross-checks the logs, the training records and the epoch history stored inside each checkpoint; all 12 runs agree epoch by epoch.

| Model | Epochs (seeds 0 / 1 / 2) | Kept epoch (seeds 0 / 1 / 2) |
|---|---|---|
| MLP | 11 / 8 / 14 | 8 / 5 / 12 |
| GRU | 13 / 14 / 8 | 10 / 13 / 5 |
| BiGRU | 14 / 9 / 13 | 13 / 6 / 10 |
| ABiGRU | 13 / 14 / 13 | 10 / 13 / 10 |
| **Total** | **144** | |

## 8.8 v1 smoke results (for the record; not benchmark results)

From `archive_v1/experiments/results/comparison_table.txt` (5 functions, 2,048 rows per split, 1 deep epoch, test on a 2,048-row subsample with ~76% positives): LR F1 0.8807, RF 0.8817, XGBoost 0.8776, GRU 0.8726, BiGRU 0.8807, ABiGRU 0.8605. **These verify that the code runs; they do not estimate performance** (different positive rate, tiny sample, one epoch).

## 8.9 Multi-seed confirmation

Purpose: show whether the ≈0.002 F1 differences between learned models are real or due to random initialization.

* **XGBoost (3 seeds, completed):** `xgb_full` F1 on main test = 0.9420, 0.9431, 0.9427 (mean 0.9426, sd 0.0005; PR-AUC 0.9875 / 0.9878 / 0.9878). `xgb_no_static` = 0.9430, 0.9424, 0.9427 (mean 0.9427, sd 0.0002; PR-AUC 0.9881 / 0.9880 / 0.9881). On the reactivation slice the XGBoost F1 spread across seeds is larger (full: mean 0.850, sd 0.004; no-static: mean 0.855, sd 0.005).
* **All learned models, 3 seeds each** (`v2/results/multiseed_summary.md`, produced by `v2/multiseed.py`; each seed's threshold is tuned on its own validation predictions; XGBoost seeds are 42, 1, 2; neural seeds 0, 1, 2):

**Main test slice (eligible functions, 443,520 rows)**

| Model | F1 mean ± sd | F1 min–max | MCC mean ± sd | PR-AUC mean ± sd | Accuracy |
|---|---|---|---|---|---|
| XGBoost no-static | 0.9427 ± 0.0002 | 0.9424–0.9430 | 0.9341 ± 0.0002 | 0.9881 ± 0.0001 | 0.9849 |
| XGBoost full | 0.9426 ± 0.0005 | 0.9420–0.9431 | 0.9340 ± 0.0005 | 0.9877 ± 0.0002 | 0.9850 |
| ABiGRU | 0.9423 ± 0.0006 | 0.9416–0.9430 | 0.9337 ± 0.0007 | 0.9884 ± 0.0004 | 0.9848 |
| BiGRU | 0.9414 ± 0.0007 | 0.9404–0.9420 | 0.9326 ± 0.0008 | 0.9880 ± 0.0004 | 0.9846 |
| GRU | 0.9410 ± 0.0008 | 0.9399–0.9417 | 0.9322 ± 0.0009 | 0.9874 ± 0.0003 | 0.9844 |
| MLP (no sequence) | 0.9409 ± 0.0003 | 0.9405–0.9412 | 0.9320 ± 0.0003 | 0.9871 ± 0.0002 | 0.9845 |
| Rule: 7-day majority (deterministic) | 0.9194 | – | 0.9074 | 0.8564 | 0.9791 |

**Reactivation slice (306,100 rows, 1,285 positives)**

| Model | F1 mean ± sd | F1 min–max | PR-AUC mean ± sd |
|---|---|---|---|
| XGBoost no-static | 0.8548 ± 0.0051 | 0.8487–0.8611 | 0.9204 ± 0.0001 |
| XGBoost full | 0.8502 ± 0.0040 | 0.8448–0.8542 | 0.9262 ± 0.0003 |
| ABiGRU | 0.8375 ± 0.0053 | 0.8311–0.8441 | 0.9135 ± 0.0012 |
| MLP | 0.8343 ± 0.0029 | 0.8315–0.8383 | 0.9111 ± 0.0004 |
| BiGRU | 0.8316 ± 0.0008 | 0.8308–0.8326 | 0.9130 ± 0.0026 |
| GRU | 0.8314 ± 0.0042 | 0.8267–0.8369 | 0.9115 ± 0.0009 |
| Rule: 7-day majority | 0.8701 | – | 0.7586 |

**Conclusions from the seeds**
1. **Learned models vs rules (main slice):** every learned model beats the best rule by 0.021–0.023 F1 in every seed. This difference is real.
2. **XGBoost vs ABiGRU:** 0.9426–0.9427 vs 0.9423, a gap smaller than one seed standard deviation. **They are tied.**
3. **ABiGRU vs the other neural models:** ABiGRU is ahead of BiGRU, GRU and MLP by 0.0009–0.0014 F1 on average (roughly 1.5–2 seed standard deviations). The order is consistent, but the size is about one tenth of an F1 point, far smaller than the function-level bootstrap interval (about ±0.017). It should be reported as "attention adds a very small, consistent gain over other neural variants, but does not beat gradient-boosted trees".
4. **GRU and BiGRU vs MLP:** no meaningful difference (0.0001–0.0005), so recurrent encoding of the 2-hour window adds nothing over the 57 engineered features.
5. **Reactivation slice:** XGBoost is about 0.013–0.023 F1 ahead of the neural models, and the 7-day majority rule still has the highest F1 (0.870), although every learned model ranks events much better (PR-AUC 0.91–0.93 vs 0.76).
6. **Reproducibility:** retraining seed 0 of all four neural models on the same machine gave predictions identical to the first run (difference 0.0), and XGBoost re-runs were also identical. Different hardware (CPU, NVIDIA GPU) may give slightly different neural results.

## 8.10 Computational performance

| Item | Value | Source |
|---|---|---|
| Hardware (this run) | Apple Silicon Mac, 10 CPU cores, 16 GB RAM, MPS GPU | `sysctl`, torch check |
| Feature building (57 features) | about 2 s for the 2.1 M-row training frame | timed once during development |
| XGBoost training | 8–29 s per variant | `run_gbdt.log` |
| Neural training time (seed 0) | MLP ≈ 17 s, GRU ≈ 392 s, BiGRU ≈ 868 s, ABiGRU ≈ 908 s (to epoch 13); up to about 20 min for ABiGRU when the machine was shared | `run_deep_s0.log`, `run_seeds.log`, `run_s0_resave.log` |
| Epoch time | MLP ≈ 1.5 s, GRU ≈ 30 s, BiGRU ≈ 62 s, ABiGRU ≈ 70 s | `run_deep_s0.log` |
| Additional evaluations | `extra_eval.py` 2 min 53 s (scoring 2.1 M training rows with every saved model); rolling-origin folds 21–36 s each; simulator under 1 min. The XGBoost grid search was not timed | timed during the 2026-11-19 runs |
| Inference latency, memory use | **[NOT CALCULATED]** | |

## 8.11 Figures (v2/figures/; figures 1–6 from `v2/make_figures.py`, figure 7 from `v2/prewarm_sim.py`, figure 8 from `v2/training_report.py`)

| File | Shows |
|---|---|
| `fig1_multiseed_f1.png` | F1 of each learned model (3 seeds: mean, each seed, min–max) against the 7-day rule, for all test buckets and for reactivation buckets |
| `fig2_precision_recall.png` | Precision–recall curves of XGBoost, ABiGRU and MLP, with the 7-day rule as a point; all buckets and reactivation buckets |
| `fig3_roc.png` | ROC curves (false-positive axis zoomed to 0–0.05) for the same models |
| `fig4_confusion_matrices.png` | Confusion matrices of XGBoost (seed 42) and ABiGRU (seed 0) at their validation thresholds, counts and row shares |
| `fig5_feature_importance.png` | Top 15 XGBoost features by gain share |
| `fig6_training_curves.png` | Validation PR-AUC per epoch for the four neural models (seed 0) |
| `fig7_prewarm_tradeoff.png` | Pre-warming simulator: cold starts avoided against containers per bucket, and against the share of wasted pre-warms; threshold and keep-alive policies as markers |
| `fig8_training_all_seeds.png` | Training loss and validation PR-AUC per epoch for all 12 neural runs (4 models × 3 seeds), kept epoch marked; produced by `v2/training_report.py` |

Static PNG, 200 dpi, light theme; colors from a colorblind-validated categorical palette; zoomed axes are stated on each figure. They need `v2/results/frames.npz` and `preds_*.npz`, so on a fresh clone run the training scripts first.

## 8.12 Additional evaluations (2026-11-19; no new models)

Full tables: `v2/results/extra_eval.md`, `calibration.md`, `prewarm_sim.md`, `robustness.md`. Main seed of each model (XGBoost seed 42, neural seed 0) unless stated.

### 8.12.1 Training vs validation vs test (overfitting check) and slice confusion matrices

| Model | Train F1 (days 7–23) | Validation F1 | Test F1 |
|---|---|---|---|
| XGBoost full | 0.9656 | 0.9399 | 0.9420 |
| XGBoost no-static | 0.9658 | 0.9393 | 0.9430 |
| ABiGRU | 0.9592 | 0.9393 | 0.9430 |
| BiGRU | 0.9594 | 0.9395 | 0.9420 |
| GRU | 0.9593 | 0.9396 | 0.9417 |
| MLP | 0.9590 | 0.9390 | 0.9409 |
| Rule: 7-day majority | 0.9379 | 0.9123 | 0.9194 |

The train–test gap is about 0.017–0.024 F1, and the rule (which learns nothing) shows a similar 0.019 gap, so most of the gap is the later days being slightly harder, not overfitting. Confusion matrices for every slice (main, reactivation, low-history, holdout) are in `extra_eval.md`, Section 1.

### 8.12.2 Paired statistical tests (test set, function-level paired bootstrap, 1,000 resamples)

| Comparison | F1 difference | 95% CI | p |
|---|---|---|---|
| XGBoost full − 7-day rule | +0.0226 | [+0.0121, +0.0351] | < 0.001 |
| ABiGRU − 7-day rule | +0.0236 | [+0.0127, +0.0366] | < 0.001 |
| MLP − 7-day rule | +0.0215 | [+0.0109, +0.0340] | < 0.001 |
| ABiGRU − XGBoost full | +0.0010 | [−0.0002, +0.0024] | 0.13 |
| XGBoost no-static − full | +0.0010 | [−0.0000, +0.0020] | 0.05 |
| ABiGRU − MLP | +0.0021 | [+0.0008, +0.0037] | < 0.001 |
| ABiGRU − BiGRU | +0.0010 | [−0.0004, +0.0032] | 0.28 |
| ABiGRU − GRU | +0.0013 | [−0.0001, +0.0031] | 0.08 |
| BiGRU − MLP | +0.0011 | [−0.0002, +0.0023] | 0.08 |
| GRU − MLP | +0.0008 | [+0.0000, +0.0016] | 0.04 |

With 10 comparisons, a Bonferroni-corrected level is 0.005. Only "learned model vs rule" and "ABiGRU vs MLP" pass it. **ABiGRU is not significantly different from XGBoost.** Row-level McNemar tests (also in `extra_eval.md`) are reported for completeness but are optimistic because rows of one function are correlated.

### 8.12.3 Pre-warming simulator (`v2/prewarm_sim.py`, figure `fig7_prewarm_tradeoff.png`)

Replays test days 27–30 for the 385 eligible functions. Before each 5-minute bucket a policy chooses which functions get **one** pre-warmed container; a pre-warm avoids the first cold start of that bucket (saving its latency) or, if no cold start happens, is wasted (one container idles 5 minutes). Assumptions: the container is ready within one bucket; counterfactual effects (a warm container serving later requests) cannot be simulated with event-only data, so avoided counts are conservative. Totals: 82,031 cold starts, 81,939 s of cold-start latency. Ceiling for one container per function-bucket: 70.4% of cold starts, 60.2% of latency.

| Policy | Pre-warms per bucket | Cold starts avoided | Latency saved | Wasted pre-warms | Idle container-hours |
|---|---|---|---|---|---|
| No pre-warming | 0 | 0% | 0% | – | 0 |
| Keep-alive 15 min | 86.2 | 58.6% | 51.6% | 51.5% | 4,263 |
| Keep-alive 60 min | 119.3 | 68.8% | 58.7% | 58.9% | 6,747 |
| Hybrid histogram (ATC'20, fallback 4 h) | 125.6 | 70.1% | 59.8% | 60.2% | 7,263 |
| Hybrid histogram (fallback 60 min) | 125.1 | 70.1% | 59.8% | 60.1% | 7,220 |
| Rule: same slot yesterday | 50.2 | 63.8% | 53.9% | 9.4% | 454 |
| Rule: 7-day majority | 49.7 | 64.4% | 54.3% | 7.7% | 366 |
| XGBoost full @ 0.538 | 50.2 | 66.4% | 55.9% | 5.9% | 283 |
| ABiGRU @ 0.437 | 51.6 | 67.4% | 56.9% | 7.1% | 350 |
| MLP @ 0.510 | 50.8 | 66.7% | 56.4% | 6.6% | 320 |

With a fixed budget of N containers per bucket (top-N functions by score), the models and the 7-day vote score are close: at N = 40 they avoid 50.8% (models) vs 49.8% (rule) of cold starts with 9.5% vs 11.3% waste. Threshold policies, which warm a different number of functions in each bucket, sit clearly above every fixed-N budget curve in `fig7`: a fixed N wastes containers in quiet buckets and runs short in busy ones. The fixed-budget table now goes up to N = 125, where the models reach 70.3–70.4% (the ceiling). **Practical reading:** prediction-driven pre-warming reaches nearly the same cold-start reduction as a long keep-alive or the hybrid-histogram policy (1.4–3.7 points fewer avoided) with about 20–25 times fewer idle container-hours (6,397–6,980 fewer), and the learned models add 2–3 percentage points over the best rule. As classifiers on the main slice, the 60-minute keep-alive and the hybrid histogram have recall 0.98–0.996 but precision 0.40–0.41 (F1 0.57–0.58) against 0.94 for the models (the 15-minute keep-alive: precision 0.485, recall 0.833, F1 0.613); on reactivation buckets the hybrid histogram's precision is 0.023 and keep-alive never fires (it needs an event in the last hour), so neither can target the rare events. The simulator uses the main seed of each model (XGBoost 42, neural seed 0).

### 8.12.4 Calibration (`v2/calibration.py`)

All learned models are well calibrated on test: expected calibration error 0.0018–0.0031, Brier score 0.0107–0.0113, mean predicted probability 0.130 against an observed rate of 0.130. Isotonic recalibration fitted on validation changes these by at most 0.0013 ECE, so the raw probabilities can be used directly (for example for budgeted pre-warming).

### 8.12.5 Robustness (`v2/robustness.py`)

**Rolling origin** (XGBoost full and rules; three 4-day test windows with expanding training windows):

| Fold | Train | Val | Test | XGBoost F1 | 7-day rule F1 | Gap |
|---|---|---|---|---|---|---|
| 1 | 7–15 | 16–18 | 19–22 | 0.9550 | 0.9383 | +0.0166 |
| 2 | 7–19 | 20–22 | 23–26 | 0.9399 | 0.9127 | +0.0271 |
| 3 (original) | 7–23 | 24–26 | 27–30 | 0.9420 | 0.9194 | +0.0226 |

The advantage over the rule holds in every window (mean F1 0.9456 vs 0.9235). Neural models were not rolled (about 15 minutes per model per fold).

**XGBoost hyperparameter search** (16 configurations of depth, learning rate and min_child_weight; selected on validation PR-AUC): validation PR-AUC ranged only 0.9862–0.9866. The selected setting (depth 6, learning rate 0.1, min_child_weight 5) scored test F1 0.9429 vs 0.9420 for the default, a +0.0009 difference comparable to seed noise. The default hyperparameters are therefore not a weakness of the results.

### 8.12.6 Saved checkpoints for every run

The five neural runs that had finished before model saving was added (GRU, BiGRU and MLP seed 1; ABiGRU seeds 1 and 2) were retrained with saving. Every retrained run reproduced its original predictions exactly (difference 0.0), so all 12 neural checkpoints in `v2/models/` are exactly the models behind the reported numbers.

## 8.13 Not calculated (listed so that nothing is assumed)

The paper's baselines rerun on this data; inference latency and memory; rolling-origin evaluation and tuning of the neural models (only XGBoost and the rules were rolled and tuned); results on other Huawei regions (the loader exists, Section 2.14a); the hybrid-histogram policy with a real ARIMA branch; a live (non-simulated) pre-warming test; any result on an Azure dataset. Training-set metrics, per-slice confusion matrices, calibration, paired significance tests and the pre-warming simulation were added on 2026-11-19 (Section 8.12).


---

# 9. Technical Definitions and Terminology (Glossary)

Format: **Term** — general definition. *In this project.* Example.

* **Cold start** — extra delay when a function must start a new container. *Each row of the dataset.* A function idle for hours is invoked and waits 0.9 s.
* **Warm start** — invocation served by an already-running container. *Not recorded in this data.*
* **Function (FaaS function)** — a small program run on demand. *Identified by funcName + userID + pool.*
* **Container / pod** — the isolated environment that runs a function. *`podID` is its identifier.*
* **Pool** — a group of pre-made pods of one size. *`pool24-600-512` = 600 millicores CPU, 512 MB.*
* **Invocation data** — records of calls to functions. *Not present here: only cold starts are listed.*
* **Bucket / slot** — a fixed time interval. *5 minutes; "slot" = position within the day (0–287).*
* **Time series** — values ordered in time. *Each function's event counts per bucket.*
* **Feature** — an input clue to a model. *One of the 57 columns.*
* **Feature engineering** — turning raw data into useful features. *Lags, seasonal lags, rolling counts.*
* **Lag** — a past value. *`lag3` = event 3 buckets ago.*
* **Seasonality / periodicity** — repetition at fixed intervals. *Same time of day, same weekday.*
* **Yesterday (in this project)** — the same time of day one day earlier = bucket t − 288. *Not "the previous calendar day in general".*
* **Target / label** — what the model predicts. *`y` = 1 if ≥ 1 cold start in bucket t.*
* **Classification** — predicting a category. *Binary: event / no event.*
* **Regression** — predicting a number. *Not used in v2 (proposed: time-to-next, cost).*
* **Prediction / inference** — applying a trained model. *Producing a probability for a test row.*
* **Probability score** — a number in [0,1] expressing confidence. *Output of XGBoost and neural models.*
* **Threshold** — cut-off turning a probability into yes/no. *Chosen on validation to maximize F1.*
* **Training / validation / test** — fitting / tuning / final scoring data. *Days 7–23 / 24–26 / 27–30.*
* **Hyperparameter** — a setting chosen by humans before training. *Tree depth 8, learning rate 0.001.*
* **Learned parameter** — a value found by training. *Network weights, tree splits.*
* **Model / algorithm** — a procedure with parameters that maps inputs to outputs.
* **Decision tree** — a flowchart of yes/no questions on features.
* **Gradient boosting (XGBoost)** — adding trees one by one, each correcting the previous errors.
* **Neural network** — layers of weighted sums with non-linear functions, trained by gradient descent.
* **MLP** — a basic fully connected network. *`mlp_static`.*
* **RNN / GRU** — a network that reads a sequence step by step and keeps a memory. *Gated Recurrent Unit.*
* **Bidirectional** — reading the window in both directions. *Inside a past-only window; no future leakage.*
* **Attention** — learning which steps matter more and averaging with those weights.
* **Max-pooling** — keeping the maximum value across steps.
* **Embedding** — a learned numeric vector summarizing something. *The 64-number static summary and the sequence encoder outputs are embedding-like vectors; no separate embedding table (for example per function ID) is used.*
* **Logit** — the raw network output before the sigmoid.
* **Sigmoid** — squashes a number to (0,1).
* **Loss function** — measures how wrong a prediction is. *Binary cross-entropy.*
* **Optimizer** — the method that adjusts parameters to reduce loss. *RMSprop.*
* **Epoch** — one training pass. *Here: 700,000 randomly drawn rows.*
* **Batch** — rows processed together. *2,048.*
* **Dropout** — randomly switching off neurons during training to reduce memorization.
* **Gradient clipping** — capping the size of an update.
* **Early stopping** — stopping when validation stops improving.
* **Overfitting** — learning the training data's quirks so that new data is predicted worse. *Checked by comparing validation and test scores.*
* **Underfitting** — a model too weak to capture the pattern.
* **Data leakage** — information from the future or from the test set entering training or features. *Controlled as in Section 2.14.*
* **Class imbalance** — one class much rarer. *13% positives.*
* **Precision, recall, F1, MCC, PR-AUC, ROC-AUC, accuracy** — see Section 8.1a.
* **Confusion matrix** — table of TP/FP/FN/TN.
* **Baseline** — a simple reference method. *The seven rules.*
* **Ablation** — removing a component to measure its effect. *no-static, no-pool, MLP.*
* **Ensemble learning** — combining several models' outputs. *Not used here (Section 5).* (Note: a "forest" or "boosted trees" combines many trees *inside* one model; that is not an ensemble of separate models.)
* **Seed** — the starting value of a random generator, so that runs can be repeated.
* **Bootstrap confidence interval** — a range estimated by resampling the data many times. *Resampling whole functions.*
* **Slice** — a subset of rows evaluated separately. *Reactivation, low-history, holdout.*
* **Reactivation event** — a cold start after at least an hour without any recorded event for that function. *`roll12 == 0` and target = 1.*
* **Holdout function** — a function excluded from training. *96 functions.*
* **Low-history function** — fewer than 10 cold starts in days 0–23. *46 functions.*
* **Cost captured @2%** — share of cold-start seconds in the top-scored 2% of cells.
* **Smoke test** — a tiny run that only checks that code executes. *v1 results.*
* **Pre-warming / provisioned concurrency** — creating containers before demand arrives. *The intended use of the forecast; simulated offline in `prewarm_sim.py`, not deployed on a live platform.*
* **Pre-warming simulator** — a program that replays historical data and applies a pre-warming policy. *`v2/prewarm_sim.py`; counts avoided cold starts, saved latency and wasted pre-warms.* Example: "ABiGRU @ threshold avoids 67.4% of cold starts".
* **Hybrid histogram policy** — Azure Functions' pre-warm/keep-alive policy (Shahrad et al., ATC 2020) that learns each application's idle-time histogram. *`v2/hh_policy.py`, adapted to cold-start gaps in 5-minute buckets.*
* **Keep-alive** — keeping a container alive for a fixed time after use. *Baseline policies keep-alive 15 min and 60 min.*
* **Pre-warming policy** — the rule deciding which functions get a container in each bucket. *Keep-alive, rules, model @ threshold, fixed budget.*
* **Threshold policy vs fixed budget** — a threshold policy warms every function whose probability is high enough (the number varies per bucket); a fixed budget warms exactly N functions per bucket. *Threshold policies dominate in `fig7`.*
* **Wasted pre-warm / idle container-hour** — a pre-warmed container that served no cold start; 12 wasted pre-warms = 1 idle container-hour. *Measures the cost side of pre-warming.*
* **Calibration** — whether predicted probabilities match observed frequencies. *If a model says 0.8 for many buckets, about 80% of them should have a cold start.*
* **Brier score** — mean squared difference between probability and outcome (lower is better). *0.0107–0.0113 here.*
* **ECE (expected calibration error)** — average gap between predicted probability and observed rate across probability bins. *0.0018–0.0031 here; near 0 = well calibrated.*
* **Isotonic regression (recalibration)** — a monotone mapping fitted on validation data to correct probabilities. *Tried; brought no gain because models were already calibrated.*
* **Reliability table** — observed outcome rate per predicted-probability bin. *In `calibration.md`.*
* **p-value** — the probability of seeing a difference at least this large if there were truly no difference. *p < 0.05 is the usual "significant" mark.*
* **Paired bootstrap test** — resampling the same units for two models and recomputing their difference many times. *Whole functions resampled 1,000 times.*
* **McNemar test** — compares where two classifiers disagree on the same rows. *Reported, but optimistic here because rows of a function are correlated.*
* **Bonferroni correction** — dividing the significance level by the number of tests to avoid lucky findings. *10 tests → 0.005.*
* **Rolling-origin evaluation** — repeating train/validate/test on several consecutive time windows. *Three 4-day test windows for XGBoost.*
* **Hyperparameter search (grid search)** — trying combinations of settings and choosing by validation score. *16 XGBoost settings.*
* **Fresh clone** — a new copy of the repository from GitHub, as a teammate would get it. *Used to test that the project works outside this laptop.*
* **MPS** — Apple's GPU backend for PyTorch.

---

# 10. Project Directory and Code Documentation

## 10.1 Directory tree

```
coldstart-problem/
├── dataset/
│   └── raw/
│       ├── region5_cold_starts/R5/day_00.csv … day_30.csv   (31 files, 123 MB)   ESSENTIAL
│       └── region5_cold_starts.zip                           (54 MB) backup of the same data
├── v2/                                  ★ CURRENT PIPELINE
│   ├── data.py  features.py  frames.py              data cache, features, splits
│   ├── run_gbdt.py  run_xgb_seeds.py                rules + XGBoost (train, save)
│   ├── nets.py  run_deep.py                         neural models (definitions; train, save)
│   ├── evalx.py  report.py  final_table.py  multiseed.py  save_thresholds.py
│   ├── predict.py                                   load a saved model and predict
│   ├── baseline_eval.py                             early rule-baseline analysis
│   ├── cache/matrices.npz, agg_R5/                  (git-ignored; matrices_R<n>.npz and agg_R<n>/ for other regions)
│   ├── make_figures.py                              paper figures -> figures/
│   ├── figures/  fig1–fig8 (.png)
│   ├── extra_eval.py  calibration.py  prewarm_sim.py  robustness.py   additional evaluations
│   ├── hh_policy.py                                 hybrid-histogram baseline (Azure, ATC'20)
│   ├── region_prep.py                               aggregate another region's zip, build its cache
│   ├── models/   xgb_*.json, *_features.json, *.pt, thresholds.json      (≈40 MB)
│   └── results/  preds_*.npz, info_*.json, frames.npz, benchmark.md, final_test_table.txt, multiseed_summary.md,
│                 extra_eval.md, calibration.md, prewarm_sim.md, robustness.md, training_report.md,
│                 logs/ (training console output, local paths removed)
├── archive_v1/                          OLD v1 PIPELINE (smoke-tested only; reference)
│   ├── main.py  config/  src/
│   ├── experiments/results/   (smoke models, 14 charts, tables, logs)
│   ├── IMPLEMENTATION_STATUS.md         v1 audit (2026-11-12)
│   └── PROJECT_PROGRESS_REPORT.md       superseded snapshot
├── research/   research_summary.md, algorithm_selection.md, literature_analysis.md  (paper analysis; parts describe v1)
├── _github_repo/                        provider's data-release repo (README, papers, demo notebooks)
├── docs/PROJECT_DOCUMENTATION.md        this document
├── requirements.txt                     pinned versions (v1's old list: archive_v1/requirements_v1.txt)
├── R1.zip … R4.zip                      raw data of Huawei regions 1–4 (git-ignored, not modeled yet)
├── report_overleaf/                     IEEE-style report: Overleaf project zip (main.tex + figures) and compiled PDF; not tracked in git
├── README.md                            quick start
└── .gitignore                           excludes raw data, caches, prediction arrays, logs, _github_repo/
```

Cleanup performed on 2026-11-19: v1 moved to `archive_v1/`; deleted `.DS_Store` files, all `__pycache__/` folders, `experiments/.mplconfig/`, the empty folders `notebooks/`, `dataset/processed/`, `experiments/baseline/`, `experiments/proposed_model/`, `src/preprocessing/`, and `pipeline_run.log` (an unfinished v1 run log).

## 10.2 v2 files

| File | Purpose | Main functions | Input → Output | Depends on | Essential? |
|---|---|---|---|---|---|
| `v2/data.py` | Build matrices, static stats, roles; per-day aggregation; `CS_REGION` switch | `aggregate()`, `raw_files()`, `build()`, `load()` | CSVs or `agg_R<n>/*.pkl` → `cache/matrices.npz` | pandas, numpy | **Yes** |
| `v2/features.py` | Features | `Ctx`, `make()` | matrices + (f,t) rows → X | `data.py` | **Yes** |
| `v2/frames.py` | Split definitions and frame builder | `get_ctx`, `frame_defs`, `build` | → dict of arrays | `data.py`, `features.py` | **Yes** |
| `v2/run_gbdt.py` | Rules + 4 XGBoost; writes `frames.npz` side-file | script | frames → `preds_gbdt.npz`, `info_xgb_*.json`, `frames.npz`, log | xgboost | **Yes** |
| `v2/run_xgb_seeds.py` | Extra XGBoost seeds | script | frames → `preds_xgbseeds.npz` | xgboost | Optional |
| `v2/nets.py` | Neural architectures, sequence builder, scaler, checkpoint save/load | `Net`, `Attention`, `make_tensors`, `seq`, `standardize`, `predict`, `save_checkpoint`, `load_checkpoint` | — | torch, `data.py` | **Yes** |
| `v2/run_deep.py` | Train one neural model and save it | script `python3 v2/run_deep.py <model> <seed>` | frames/matrices → `preds_<model>_s<seed>.npz`, `info_*.json`, `models/<model>_s<seed>.pt` | torch, `nets.py` | **Yes** |
| `v2/evalx.py` | Threshold, metrics, bootstrap, cost capture | `best_thr`, `metrics`, `cost_capture` | y, scores → metrics | sklearn | **Yes** |
| `v2/report.py` | Slice tables + seed aggregation | script | `preds_*.npz`, `frames.npz` → `benchmark.md` | `evalx.py` | **Yes** |
| `v2/final_table.py` | Final accuracy/F1 table | script | same → `final_test_table.txt` | `evalx.py` | Optional |
| `v2/multiseed.py` | Mean/sd/min–max across seeds | script | preds, frames → `results/multiseed_summary.md` | `evalx.py` | Optional |
| `v2/save_thresholds.py` | Validation-tuned threshold per saved model | script | preds, frames → `models/thresholds.json` | `evalx.py` | Recommended |
| `v2/predict.py` | Load a saved model; list models; score one function for one day | `make_scorer`, CLI | `models/*`, cache → printed table | xgboost, torch, `nets.py`, `features.py` | Recommended |
| `v2/make_figures.py` | Six paper figures | script | preds, frames, info files → `v2/figures/*.png` | matplotlib, `evalx.py` | Optional |
| `v2/extra_eval.py` | Slice confusion matrices, train/val/test metrics, paired tests | script | preds, frames, saved models → `results/extra_eval.md` | scipy, `nets.py`, xgboost | Optional |
| `v2/calibration.py` | Brier, ECE, reliability, isotonic check | script | preds, frames → `results/calibration.md` | sklearn | Optional |
| `v2/prewarm_sim.py` | Pre-warming simulator (policies and per-bucket budgets) | script | preds, frames, cache, thresholds → `results/prewarm_sim.md`, `figures/fig7_prewarm_tradeoff.png` | matplotlib (figure only) | Recommended |
| `v2/robustness.py` | Rolling-origin evaluation and XGBoost tuning | script | cache → `results/robustness.md` | xgboost | Optional |
| `v2/training_report.py` | Training evidence: all runs, epochs, times, fingerprints; cross-checks logs, records and checkpoints | script | `info_*.json`, `models/*`, `results/logs/` → `results/training_report.md`, `figures/fig8_training_all_seeds.png` | torch, matplotlib | Recommended |
| `v2/hh_policy.py` | Hybrid-histogram warm/cold decisions per (function, bucket) | `hh_warm`, `_params` | event matrix, query rows → bool warm, branch id | numpy | Recommended |
| `v2/region_prep.py` | Aggregate a region zip day by day, build `matrices_R<n>.npz` | script `python3 v2/region_prep.py <region> [zip]` | `R<n>.zip` → `cache/agg_R<n>/`, cache | `data.py`, pandas | Optional |
| `v2/baseline_eval.py` | First rule-baseline and slice analysis | script | CSVs → printed tables | sklearn | Optional |

## 10.3 v1 files (now under `archive_v1/`)

| File | Purpose | Key items | Status |
|---|---|---|---|
| `config/config.py` | Constants for v1 | splits, bucket size 300, window 12, 17 features, model sizes, RMSprop settings, device selection | used by v1 |
| `src/data/loader.py` | Load CSVs, validate schema/time, make funcID | `load_cold_start_data`, `extract_pool_name` | used by v1; v2 does not import it |
| `src/feature_engineering/features.py` | 17 features, 12-step windows, scaling | `process_dataset`, `build_function_time_series` | v1 (slow loop; run never finished) |
| `src/models/baselines.py` | LR, RF, XGBoost + save/load | `train_*`, `save_baseline`, `load_baseline` | v1 |
| `src/models/deep_models.py` | `Attention`, `StandardGRU`, `BiGRU`, `ABiGRU`, `create_model`, `load_deep_model` | | v1 (v2 has its own copy of `Attention` in `nets.py`) |
| `src/training/trainer.py` | RMSprop loop, early stopping, checkpointing | `train_model`, `predict_deep_model` | v1 |
| `src/evaluation/metrics.py` | Metrics and threshold search | `compute_all_metrics`, `find_optimal_threshold` | v1 |
| `src/visualization/plots.py` | EDA, ROC/PR, confusion matrices, curves | | v1 |
| `main.py` | v1 orchestrator (6 phases, `--smoke-test`) | `main()` | v1 |

Research notes in `research/` document the v1 design and the paper analysis; `_github_repo/` and `PROJECT_PROGRESS_REPORT.md` are background (the latter is a superseded snapshot; its "807,000+ events" figure differs from the verified 805,745).

## 10.4 Important execution commands

See Section 13.5.

---

# 11. Why This Architecture and Methodology Were Chosen

| Decision | Reasoning | Limitation |
|---|---|---|
| **Dataset: Huawei Region 5 cold-start trace** | Public, production-scale, 31 days, per-event cost breakdown; the only dataset available in the project | Cold-start events only (no warm invocations, no request counts); one region |
| **Task: next-bucket occurrence per function** | The only task supported by event-only data; actionable for pre-warming | Not equal to per-request cold/warm classification used in the paper |
| **5-minute buckets** | Reasonable resolution for pre-warming; carried over from v1 | Not tuned |
| **Time-based split** | Mimics deployment | Neural models on one split; XGBoost and rules also checked on 3 rolling windows |
| **Seasonal features (1, 2, 3, 7 days; ±1 bucket)** | Exploration showed periodicity dominates (rules F1 0.90–0.92) | Weekday names unknown (trace start date not documented here) |
| **Static statistics from days 0–6 only** | Prevent target leakage | Functions first appearing later have no static profile |
| **Pool context features** | Cheap, available for unseen functions | Added little |
| **Rules as baselines** | A learned model must prove its worth | — |
| **XGBoost** | Best default for tabular data; interpretable importances | — |
| **MLP / GRU / BiGRU / ABiGRU** | Isolate the contribution of sequence modeling; tie to the paper's architecture | ABiGRU adapted, not exactly reproduced |
| **Hybrid input (sequence + 57 features)** | Gives neural models the same information as XGBoost so the comparison is fair | Makes it harder for the sequence branch to show a unique contribution |
| **RMSprop lr 0.001, alpha 0.9** | Same as the paper | Not tuned; other optimizers not tested |
| **No class weights** | Keeps probabilities interpretable; threshold handles imbalance | Different training emphasis than v1 |
| **Threshold on validation by F1** | No test peeking | The best F1 threshold is not necessarily the best for cost saving |
| **Slices + cluster bootstrap** | Averages hide the hard cases; rows within a function are not independent | Small positive counts in some slices |
| **Model order** | Rules → trees → networks: increasing complexity, so each level must justify itself | — |
| **No model interconnection** | The goal was a fair comparison, not a combined system | An ensemble might add a little accuracy; not tried |
| **Hybrid histogram as the industry baseline** | The standard published pre-warming policy (Azure); makes the cost comparison meaningful | Adapted to event-only data (cold-start gaps, no ARIMA), so it is a faithful-in-spirit, not exact, reproduction |
| **Pre-warming simulator with one container per function-bucket** | Simplest policy that event-only data can score honestly; makes model comparisons operational | Counterfactual effects (a warm container serving later requests) cannot be simulated |
| **Function-level paired bootstrap for significance** | Rows of one function are correlated; resampling whole functions keeps that structure | Few functions in some slices → wide intervals |
| **Calibration check before using probabilities for budgets** | Budget policies rank by probability | — (models were already calibrated) |

**Why it suits the research problem:** it honestly measures how much is predictable, from which clues, and for which functions. **What remains limited:** single region, event-only data, offline (not live) simulation, neural models not rolled or tuned.

---

# 12. Final Outcomes and Research Contributions

## 12.1 What the project achieves
* A working, leakage-audited, vectorized pipeline that produces a full benchmark in minutes.
* Trained and evaluated 8 learned model variants plus 6 rule baselines on 443,520 test rows, with 3 seeds for the main models.
* Saved every trained model (8 XGBoost, 12 neural) with a loader (`predict.py`) that works on a fresh clone.
* Measured the operational value of the forecasts with a pre-warming simulator (including a hybrid-histogram baseline), and checked the conclusions with significance tests, calibration, rolling-origin evaluation and a tuning study.

## 12.2 What the trained models contribute
* They improve over the best rule on the main task by about 2.4 F1 points (0.942–0.943 vs 0.919), about 2.6 MCC points (0.933 vs 0.907), and a large PR-AUC margin (0.987–0.989 vs 0.856).
* Among themselves the differences are at most about 0.002 F1 (3-seed means): XGBoost and ABiGRU are tied at the top (paired test p = 0.13); ABiGRU is very slightly but consistently ahead of GRU, BiGRU and MLP (significant only against MLP).
* In the simulator, they avoid 2–3 percentage points more cold starts than the best rule with less waste.

## 12.3 How the system addresses cold-start challenges
* **Recurring cold starts (the majority):** well predicted.
* **Reactivation:** ranking improves (PR-AUC) but the F1 benefit over the best rule is not established.
* **Low-history functions:** a modest, noisy gain.
* **New functions without history:** **not addressed**; the attempted approach failed.
* **Operationally:** prediction-driven pre-warming reaches 66–67% avoided cold starts (ceiling 70.4%) with 6–7% waste, close to a 60-minute keep-alive's 68.8% but with about 20 times fewer idle container-hours.

## 12.4 Improvements over baselines
Main task: +0.023 F1 over the 7-day majority rule (0.9426 mean for XGBoost vs 0.9194; p < 0.001), consistent across 3 rolling windows (+0.017 to +0.027). Pre-warming: +2.0 to +3.0 points of avoided cold starts over the rule; against a 60-minute keep-alive, 1.4–2.4 points fewer avoided but 6,397–6,464 fewer idle container-hours; against the hybrid histogram, 2.7–3.7 points fewer avoided but 6,913–6,980 fewer idle container-hours. Slices: Sections 8.4–8.5.

## 12.5 Limitations
Single region modeled (Region 5; code for the others exists but is unused); event-only data; no warm-start ground truth; neural models evaluated on one time split (XGBoost was also rolled); ABiGRU only an adaptation; neural hyperparameters not tuned (XGBoost tuning showed little sensitivity); the pre-warming simulator cannot model counterfactual effects (a warm container serving later requests); cost-capture metric is approximate for rules.

## 12.6 Future improvements (proposals, not implemented)
1. Provider's other tables (request counts, per-function time series — the README lists them) could supply invocation information; **[their contents were not inspected beyond the README]**.
2. Residual model on top of the seasonal baseline; multi-task heads (occurrence, time to next, cost).
3. Cost-aware pre-warming policies that weight functions by expected latency saved (the simulator now exists, Section 8.12.3).
4. Better new-function modeling (function embeddings, more metadata).
5. Run regions 1–4 (`region_prep.py` is ready); rolling-origin evaluation and tuning of the neural models (done for XGBoost).
6. Done since: pinned `requirements.txt`, `README.md`, `.gitignore`; repository published at github.com/pradnish18/serverless-coldstart-prediction; hybrid-histogram baseline and region-aware loading (2026-11-23, branch `hybrid-histogram-baseline`); IEEE paper draft in `report_overleaf/`.

## 12.7 What remains incomplete
The v1 full-scale run (not needed for the benchmark). Multi-seed runs, model saving, calibration, statistical tests, rolling-origin evaluation (XGBoost) and the pre-warming simulator are complete.

---

# 13. Complete Project Reproduction Guide

## 13.1 Hardware used
Apple Silicon Mac, 10 CPU cores, 16 GB RAM, macOS (Darwin). The neural models used the Apple GPU (MPS). A CPU-only run is possible (the script falls back to CPU) but slower. Peak memory use was not measured; the 2.1 M × 57 float32 training matrix is about 480 MB.

## 13.2 Software versions verified in this environment

| Package | Version |
|---|---|
| Python | 3.8.18 |
| numpy | 1.24.4 |
| pandas | 2.0.3 |
| scikit-learn | 1.3.2 |
| xgboost | 2.1.4 |
| torch | 2.4.1 |
| joblib | 1.4.2 |

`requirements.txt` pins exactly these versions. The old v1 list of unpinned minimums (which also included matplotlib, seaborn and tqdm, not needed by v2) is kept as `archive_v1/requirements_v1.txt`. LightGBM is not used.

**What the git repository does not contain** (see `.gitignore`): the raw dataset, `v2/cache/`, the prediction arrays `v2/results/*.npz`, training logs and `_github_repo/`. On a fresh clone, `predict.py` works after `python3 v2/data.py`; `report.py` and `multiseed.py` need the prediction arrays, which are regenerated by retraining.

## 13.3 Environment setup
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```
(Python 3.8 matches the environment used here; newer Python versions need correspondingly newer package versions.)

## 13.4 Dataset preparation
1. Download "Region 5 cold starts" from the link in `_github_repo/README_data_release_2025.md`.
2. Unzip so the files are `dataset/raw/region5_cold_starts/R5/day_00.csv … day_30.csv`.
3. Verify 31 files, 12 columns, 805,745 rows in total.

## 13.5 Commands (run from the project root; scripts add their own folder to the import path)
```bash
python3 v2/data.py                  # builds v2/cache/matrices.npz (~3 s from per-day aggregates; first build reads the CSVs)
python3 v2/region_prep.py 1 R1.zip  # optional: another region (default zip search: dataset/raw/R<n>*.zip); then set CS_REGION=1 for the other scripts
python3 v2/run_gbdt.py              # rules + 4 XGBoost; writes frames.npz, preds_gbdt.npz (~1–2 min)
python3 v2/run_xgb_seeds.py         # optional: seeds 1 and 2
python3 v2/run_deep.py abigru 0     # one neural model, one seed; saves v2/models/abigru_s0.pt (MLP ≈ 20 s, GRU ≈ 7 min, BiGRU/ABiGRU ≈ 15 min on MPS)
python3 v2/run_deep.py gru 0
python3 v2/run_deep.py bigru 0
python3 v2/run_deep.py mlp_static 0
python3 v2/report.py                # slice tables → v2/results/benchmark.md
python3 v2/final_table.py           # final test table → v2/results/final_test_table.txt
python3 v2/multiseed.py             # multi-seed summary → v2/results/multiseed_summary.md
python3 v2/save_thresholds.py      # thresholds → v2/models/thresholds.json
python3 v2/predict.py --list        # list saved models
python3 v2/predict.py --model abigru_s0 --function 0 --day 28    # use a saved model
python3 v2/make_figures.py          # figures -> v2/figures/
python3 v2/extra_eval.py            # slice confusion matrices, train metrics, paired tests (~3 min)
python3 v2/calibration.py           # calibration check
python3 v2/prewarm_sim.py           # pre-warming simulator (incl. hybrid histogram) + fig7
python3 v2/robustness.py            # rolling-origin evaluation + XGBoost tuning (~5 min)
python3 v2/training_report.py       # training evidence report + fig8 (~10 s)
COLDSTART_SMOKE=1 python3 v2/run_deep.py abigru 0   # live training demo: 3 short epochs (~1 min), writes only to the system temp folder
```
`run_gbdt.py` must run before `report.py` because it writes `frames.npz`. Do **not** use `python3 -I` with these scripts (the isolated mode ignores the script folder and imports fail).

v1 is archived in `archive_v1/`; to run it, move `main.py`, `config/` and `src/` back to the project root (its paths are relative to the root).

To **use the saved models without training**, only `python3 v2/data.py` (3 s) is needed before `predict.py`.

## 13.6 Expected outputs
* `v2/cache/matrices.npz` with arrays `cnt, cost, fids, role, hour_rate, fn_hour_rate, meta, meta_cols`; roles: 385 normal, 96 holdout, 46 low-history.
* Frame sizes: train 2,110,176; val_main 372,384; test_main 496,512; val_hold 82,944; test_hold 110,592.
* `final_test_table.txt`: XGBoost and neural F1 ≈ 0.941–0.943, best rule 0.919 (small variation with seeds and hardware is expected).
* `prewarm_sim.md`: about 67% of cold starts avoided for the models at their thresholds, 64.4% for the 7-day rule, 68.8% for keep-alive 60 min, 70.1% for the hybrid histogram (7,263 idle container-hours); `figures/fig7_prewarm_tradeoff.png`.
* `extra_eval.md`, `calibration.md`, `robustness.md` with the numbers in Section 8.12.

## 13.7 Testing instructions
* Re-run the leakage check (Section 6.5): zero a function's future from bucket t and recompute `make(...)`; the row must not change.
* Check that `report.py` reproduces `final_test_table.txt` to three decimals for deterministic models (rules, XGBoost with `random_state=42`).
* Neural results reproduce exactly on the same machine and seed (verified), but can differ slightly across machines because GPU/CPU arithmetic is not bit-exact.
* Fresh-clone check (done 2026-11-19): clone from GitHub, add the dataset, run `python3 v2/data.py` and `python3 v2/predict.py --model abigru_s0 --function 0 --day 29`; it must run without training. This check found and fixed a crash in `data.py`.
* Reload check: `predict.make_scorer(<model>, ctx)` on `test_main` must match `results/preds_<model>.npz` (verified: XGBoost difference 0.0; neural ≤ 5e-7, CPU vs Apple GPU rounding).

## 13.8 Model artifacts, loading, and transferring to another computer

**What exists (v2/models/).**

| File | What it is | Needed with it |
|---|---|---|
| `xgb_full_s42.json`, `xgb_full_s1.json`, `xgb_full_s2.json` | XGBoost, all 57 features | `xgb_full_features.json` |
| `xgb_no_static_s42/s1/s2.json` | XGBoost, 46 features | `xgb_no_static_features.json` |
| `xgb_no_pool_s42.json` | XGBoost, 51 features | `xgb_no_pool_features.json` |
| `xgb_cold_newfunc_s42.json` | XGBoost, 10 "cold" features | `xgb_cold_newfunc_features.json` |
| `abigru_s0.pt`, `gru_s0.pt`, `bigru_s0.pt`, `mlp_static_s0.pt` | Neural checkpoints, seed 0 (exactly the models behind the seed-0 results) | self-contained (weights, `mu`, `sd`, feature order, window) |
| `*_s1.pt`, `*_s2.pt` for the same four models | Neural checkpoints, seeds 1 and 2 | self-contained |
| `thresholds.json` | Validation-tuned decision threshold per model (the cold model's threshold is tuned on held-out functions) | — |

All 12 neural runs (4 models × 3 seeds) have a checkpoint.

**Loading.**
```python
# XGBoost
import json, xgboost as xgb
m = xgb.XGBClassifier(); m.load_model("v2/models/xgb_full_s42.json")
feats = json.load(open("v2/models/xgb_full_features.json"))["features"]   # column order of X
# Neural (run from inside v2/ or add v2/ to sys.path)
import nets
net, ck = nets.load_checkpoint("v2/models/abigru_s0.pt")                  # ck["mu"], ck["sd"], ck["feature_names"]
```
`v2/predict.py` wraps both and also builds the features and the 24×7 sequence, so the simplest use is the command line in Section 13.5.

**Transferring to another computer.**
1. Copy (or `git clone`) the code and `v2/models/` (≈40 MB, small enough for normal git).
2. Install the same library versions (Section 13.2). The XGBoost `.json` format is portable across XGBoost versions; the `.pt` checkpoints need a compatible PyTorch (2.x).
3. Download the dataset (Section 13.4) and run `python3 v2/data.py` to rebuild the cache; this is required because features are computed from the data.
4. Use `predict.py`. Optional: copy `v2/results/` to regenerate tables without retraining.

Notes: the checkpoints load on CPU by default. `run_deep.py` picks the device automatically: NVIDIA GPU (CUDA) if available, else Apple GPU (MPS), else CPU; override with the environment variable `COLDSTART_DEVICE=cpu|cuda|mps`. The CUDA path was not tested here. `nets.load_checkpoint` passes `weights_only=False` because the checkpoint stores NumPy arrays; load only checkpoints you trust.

**Version compatibility (verified problem).** The v1 smoke-test models in `archive_v1/experiments/results/models/` were saved with scikit-learn **1.7.2**, and loading them here under scikit-learn 1.3.2 emits `InconsistentVersionWarning` ("results may be invalid"). They also show that the v1 artifacts were produced on another machine (log paths under a different user's folder) and with Python 3.13 (`__pycache__` files are `cpython-313`). Use identical library versions on both machines, or retrain. `torch.load` also warns about `weights_only` in torch 2.4; when loading your own files you may pass `weights_only=True` for state-dict-only files.

## 13.9 Common errors and solutions

| Error | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: No module named 'data'` | Running with `python3 -I`, or importing from another folder | Run `python3 v2/<script>.py` normally |
| `Input contains NaN` in `average_precision_score` | NaN pool fields for two functions in the neural pipeline | Already handled in `nets.standardize` (NaN → 0 after standardization) |
| `IndexError: cannot do a non-empty take` in `metrics` | `nboot=0` in older versions of `evalx.py` | Fixed: `nboot=0` now returns NaN intervals |
| `FileNotFoundError: ... v2/cache/matrices.npz` on a fresh clone | `cache/` is not in git and older `data.py` did not create it | Fixed (2026-11-19): `data.py` creates the folder; run `python3 v2/data.py` first |
| `FileNotFoundError: ... v2/results/frames.npz` when running `report.py` on a fresh clone | Prediction arrays are not in git | Run `python3 v2/run_gbdt.py` (and the neural runs) first |
| `sed -i` fails on macOS | BSD sed needs `-i ''` | `sed -i '' ...` |
| MPS device not available | Non-Apple hardware or old torch | The script falls back to CPU (slower) |
| `InconsistentVersionWarning` on old `.joblib` | Different scikit-learn | Match versions or retrain |
| Out-of-memory in v1 feature building | Dense per-function matrices | Use v2 |
| Wrong results if static stats use all train days | Leakage | v2 uses days 0–6 only |

---

# 14. Final Comprehensive System Summary

## 14.1 End-to-end architecture diagram

(Same as Section 3.2; reproduced in compact form.)
```
CSV (31 days) → data.py (per-day aggregates) → matrices.npz → features.py → frames.py
      → [Rules | XGBoost ×4 | MLP/GRU/BiGRU/ABiGRU] → preds_*.npz → evalx/report → benchmark tables
      preds_*.npz + thresholds (+ hh_policy.py baseline) → prewarm_sim.py (pre-warming simulator) → prewarm_sim.md, fig7
      preds + saved models → extra_eval.py, calibration.py;   cache → robustness.py
```

## 14.2 Data-flow diagram

```
 raw rows ──(group by function, time//300)──► cnt, cost matrices (527×8928)
 matrices ──(days 0–6)──► static function stats, pool profiles, roles
 matrices + (f,t) ──(only buckets < t)──► 57 features   |   24×7 sequence (neural only)
 train rows ──► fit      val rows ──► early stop + threshold      test rows ──► metrics
 test probabilities ≥ threshold ──► pre-warm decisions per bucket ──► avoided cold starts, saved latency, waste
```

## 14.3 Model interaction diagram

```
   features ─┬─► Rules ────────► 0/1 ────────┐
             ├─► XGBoost ×4 ───► probability ├─► Evaluator (independent comparisons)
             └─► MLP/GRU/BiGRU/ABiGRU ─► p ──┘
   (no model feeds another; the pre-warming simulator reads each model's
    probabilities separately and compares policies, it does not combine models)
```

## 14.4 Every model: role, input, output, connections

| Model | Role | Input | Output | Connected to |
|---|---|---|---|---|
| 6 rules | Baselines | 1–2 columns | 0/1 | Evaluator only |
| XGBoost full | Main tabular | 57 features | probability | Evaluator only |
| XGBoost no-static | Ablation | 46 | probability | Evaluator only |
| XGBoost no-pool | Ablation | 51 | probability | Evaluator only |
| XGBoost cold | New-function test | 10 | probability | Evaluator only |
| MLP | Ablation (no sequence) | 57 | probability | Evaluator only |
| GRU | Sequence baseline | 24×7 + 57 | probability | Evaluator only |
| BiGRU | Bidirectional | 24×7 + 57 | probability | Evaluator only |
| ABiGRU | Paper-style attention model | 24×7 + 57 | probability | Evaluator only |

Evaluation tools (not models):

| Tool | Role | Input | Output |
|---|---|---|---|
| Pre-warming simulator (`prewarm_sim.py`) | Operational value of the forecasts | saved probabilities, thresholds, cold-start counts and costs | avoided cold starts, saved latency, waste, fig7 |
| `extra_eval.py` | Slice confusion matrices, train metrics, paired tests | saved predictions and models | `extra_eval.md` |
| `calibration.py` | Probability quality | saved predictions | `calibration.md` |
| `robustness.py` | Rolling-origin evaluation, tuning | cache (trains its own XGBoost) | `robustness.md` |

## 14.5 Datasets and their purpose

| Dataset | Purpose | Used |
|---|---|---|
| Huawei Region 5 cold starts (31 days) | Train/validate/test everything | Yes |
| Huawei Regions 1–4 cold starts | Possible replication (loader ready) | No |
| AWS data of the reference paper | Paper's own evaluation | No (unavailable) |
| Azure dataset | Requested but absent | No |
| Huawei requests / time-series / trigger tables | Possible future invocation information | No (not downloaded) |

## 14.6 Important features

| Feature | Meaning | Importance (XGBoost full) |
|---|---|---|
| `slot_hits_7d` | Days (of last 7) with an event in this slot | 0.495 |
| `d1_+0` | Event at this slot yesterday | 0.433 |
| `lag6` | Event 30 minutes ago | 0.012 |
| `roll288` | Events in the last day | 0.004 |
| `d2_+0` | Event at this slot 2 days ago | 0.003 |
| `tr_event_buckets` | Early-period event buckets | 0.003 |
| `pool_*` | Pool context | ≤ 0.002 (own-history models); top for the cold model |

## 14.7 Training and evaluation status

| Model | Trained | Seeds | Evaluated on test | Weights saved |
|---|---|---|---|---|
| Rules | n/a | n/a | Yes | n/a |
| XGBoost full / no-static | Yes | 3 | Yes | Yes (all 3 seeds) |
| XGBoost no-pool / cold | Yes | 1 | Yes | Yes |
| MLP / GRU / BiGRU / ABiGRU | Yes | 3 | Yes (all seeds) | Yes (all 3 seeds) |
| v1 six models (archived) | Smoke only | 1 | Smoke subset only | Yes (smoke weights, in `archive_v1/`) |

| Evaluation | Status |
|---|---|
| Hybrid-histogram baseline | Done (adapted; `prewarm_sim.md`) |
| Other Huawei regions | Code ready, not run |
| Slice confusion matrices, train-set metrics | Done (`extra_eval.md`) |
| Paired significance tests | Done |
| Calibration | Done (`calibration.md`) |
| Rolling-origin evaluation | Done for XGBoost and rules; not for neural models |
| Hyperparameter search | Done for XGBoost; not for neural models |
| Pre-warming simulation | Done (offline; `prewarm_sim.md`, fig7) |
| Live platform test | Not done |

## 14.8 Execution sequence from raw data to final output

1. CSVs → 2. matrices and static stats → 3. roles and holdout → 4. 57 features per row → 5. time-based frames → 6. train rules/XGBoost/neural models → 7. save predictions → 8. choose thresholds on validation → 9. test metrics by slice with bootstrap → 10. tables (`benchmark.md`, `final_test_table.txt`, `multiseed_summary.md`) → 11. additional evaluations (`extra_eval.md`, `calibration.md`, `robustness.md`) → 12. pre-warming simulation (`prewarm_sim.md`, fig7).

## 14.9 One-paragraph summary

We built a leakage-safe pipeline that forecasts, for each serverless function and each 5-minute bucket, whether a cold start will occur, using the Huawei Cloud Region 5 trace. Daily and weekly repetition of cold starts explains most of the signal: a rule such as "event in this slot on most of the last seven days" already reaches F1 0.919. XGBoost and four neural models (MLP, GRU, BiGRU, ABiGRU) reach F1 about 0.942–0.943 and are statistically indistinguishable from each other, so a sequence architecture adds nothing measurable beyond the engineered features. The models are independent; there is no ensemble. Rare reactivation events are ranked better by learned models but not clearly better at the F1 operating point, and cold starts for brand-new functions with no history remain unpredictable from hardware and pool context. A pre-warming simulator shows the practical value: acting on the models' predictions avoids about 67% of cold starts with 6–7% wasted containers, close to what a 60-minute keep-alive or the Azure hybrid-histogram policy achieves (69–70%) but with about 20–25 times fewer idle container-hours, and 2–3 points better than the best rule. The conclusions hold under paired significance tests, three rolling test windows and an XGBoost tuning study, and the models' probabilities are well calibrated. The evidence supports a rigorous benchmark paper with an operational evaluation and a clear negative result; it does not support a claim of a new best architecture.
