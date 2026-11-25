> **Historical (v1, superseded).** This audit covers the archived v1 pipeline only. The current project is `v2/`; see [`../docs/PROJECT_DOCUMENTATION.md`](../docs/PROJECT_DOCUMENTATION.md) and [`../README.md`](../README.md).

# Implementation Status

**Audit date:** 2026-11-12  
**Scope:** Repository source, configuration, research notes, 31 Region 5 daily CSVs, prior run log, and a new end-to-end smoke run.  
**Current result:** All six estimators completed a smoke training/evaluation run on trace-derived data. This is functional verification, not full-scale training or publishable performance measurement.

`pipeline_run.log` is a prior incomplete run ending during feature preparation; it is not evidence that the integrated pipeline completed. The complete current run is recorded in `experiments/results/smoke_run.log` and `run_record.json`.

The timestamp audit found that CSV `time` values are already absolute seconds from day 0. The loader previously added `day * 86400` a second time, which misaligned splits and dropped later-day events from the series. That is fixed in `src/data/loader.py`. Any smoke metrics generated before this timestamp correction are superseded; the results below are from the corrected run.

## Component status

| Component | Implemented | Integrated | Tested | Full Training | Status |
|---|---|---|---|---|---|
| Logistic Regression | Yes | Yes | Yes, smoke train/predict/save/load/evaluate | No | Smoke passed |
| Random Forest | Yes | Yes | Yes, smoke train/predict/save/load/evaluate | No | Smoke passed |
| XGBoost | Yes | Yes | Yes, smoke train/predict/save/load/evaluate | No | Smoke passed; native thread count capped |
| Standard GRU | Yes | Yes | Yes, smoke train/backward/checkpoint reload/evaluate | No | Smoke passed |
| Bidirectional GRU | Yes | Yes | Yes, smoke train/backward/checkpoint reload/evaluate | No | Smoke passed |
| ABiGRU | Yes | Yes | Yes, smoke train/backward/attention/checkpoint reload/evaluate | No | Smoke passed |
| Data loading | Yes | Yes | Yes, all 31 CSVs loaded (805,745 rows); schema and missing-file checks added | N/A | Passed |
| Feature engineering and temporal splits | Yes | Yes | Yes, real-data smoke; shapes and target alignment inspected | No full artifact | Passed for smoke scope |
| Training and checkpointing | Yes | Yes | Yes, RMSProp/BCEWithLogits/class weights/clipping/early stopping exercised | No | Smoke passed |
| Metrics and evaluation | Yes | Yes | Yes, all requested metrics and a one-class undefined-AUC case | N/A | Passed |
| Visualizations and comparison export | Yes | Yes | Yes, smoke curves, ROC/PR, six confusion matrices, comparison chart/table | N/A | Passed |

## Dataset definition and split

The selected CSV files contain cold-start events only: `day`, `time`, cluster/function/user/request identifiers, cold-start cost components, and pod ID. They do not contain warm invocations, request counts, CPU/memory utilization, scaling events, or complete per-invocation telemetry. An empty bucket therefore means no recorded cold-start event, not no invocation or a verified warm start.

The CSV `time` is already seconds since day 0; it is validated against the `day` field and used directly as `abs_time`. The model task is a function/pool-level event forecast: 12 historical 5-minute buckets predict whether the **next** 5-minute bucket has one or more cold-start records. Inputs cover `[target_bucket - 12, target_bucket)`; the label comes from `target_bucket`. Days 0–23 are training targets, days 24–26 validation targets, and days 27–30 test targets. Historical context may cross a split boundary because it is prior to that sample's target. Function eligibility is determined from training-period events, and min/max scaling is fitted on training input values only.

## ABiGRU tensor shape verification

For a smoke batch with 17 features and 12 historical steps:

| Stage | Shape |
|---|---|
| Input | `(batch, 12, 17)` |
| 2-layer bidirectional GRU output | `(batch, 12, 128)` |
| Additive-attention scores/weights | `(batch, 12)` |
| Softmax attention weights | Sum to 1 across the 12 steps |
| Attention-weighted context | `(batch, 128)` |
| Temporal max pool | `(batch, 128)` |
| Context + pooled concatenation | `(batch, 256)` |
| Fully connected hidden representation | `(batch, 64)` |
| Output logit | `(batch, 1)` |

Attention has a nonzero gradient into the loss and its weights affect the concatenated representation. The model returns logits; `BCEWithLogitsLoss` applies the numerically stable sigmoid/BCE calculation, and inference explicitly applies sigmoid to produce probabilities. The numeric feature sequence is the input representation; there is no token embedding because the inputs are tabular measurements.

This is an architecture implementation of the project's requested ABiGRU components. It is not claimed as an exact reproduction: the original study uses AWS Lambda invocation data, while the local source is Huawei cold-start events only, and the publisher abstract/full material available to this audit does not define enough details to verify every layer and training setting.

## Smoke run results

Command: `python main.py --smoke-test`  
Source: actual Region 5 trace files; five functions selected by training-period cold-start count; deterministic 2,048-row sample per split; one epoch per deep model; 25 RF trees and 15 XGBoost trees. The metrics below verify the integrated code path only and must not be reported as full-dataset performance.

This execution environment selected CPU (`Device: cpu`), so it confirms the float32 fix and complete pipeline on CPU. The user's Mac MPS backend was not available to this run; the class-weight dtype error is fixed, but MPS-specific execution still needs a smoke run on that Mac to verify its device path.

| Model | Accuracy | Precision | Recall | F1 | MCC | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8027 | 0.8143 | 0.9588 | 0.8807 | 0.3739 | 0.8254 | 0.9313 |
| Random Forest | 0.8101 | 0.8362 | 0.9325 | 0.8817 | 0.4229 | 0.8059 | 0.9162 |
| XGBoost | 0.8086 | 0.8531 | 0.9035 | 0.8776 | 0.4446 | 0.8097 | 0.9185 |
| Standard GRU | 0.7910 | 0.8122 | 0.9428 | 0.8726 | 0.3373 | 0.7858 | 0.9050 |
| Bidirectional GRU | 0.8047 | 0.8210 | 0.9498 | 0.8807 | 0.3881 | 0.7938 | 0.9118 |
| ABiGRU | 0.7837 | 0.8432 | 0.8785 | 0.8605 | 0.3820 | 0.7877 | 0.9108 |

The smoke test also checked deep forward/backward passes, BCE loss and RMSProp optimizer updates through training, all six model round-trips, probability prediction after reload, attention normalization/gradient flow, and the undefined ROC-AUC/PR-AUC path when the evaluation labels contain one class.

## Corrected files

- `src/feature_engineering/features.py`: fixed target-bucket loop, split boundaries, training-only function selection, bucket count from `NUM_DAYS`, named the 7-day encoding as a collection-day cycle rather than a calendar weekday, and retained held-out sample keys/bucket IDs for demo reporting.
- `src/models/deep_models.py`: extract both terminal direction states for BiGRU variants; added deep model checkpoint loading.
- `src/models/baselines.py`: added estimator save/load and bounded native worker threads for XGBoost/RF.
- `src/training/trainer.py`: honor configured device, create MPS-compatible float32 class weights, make scheduler compatible with current PyTorch, validate nonempty data, save checkpoint and history, and handle empty prediction arrays.
- `src/evaluation/metrics.py`: report undefined ROC-AUC/PR-AUC honestly for one-class labels and keep confusion matrices at 2×2.
- `src/visualization/plots.py`: handle one-class curve inputs safely and include PR-AUC in the model comparison chart.
- `src/data/loader.py`: use the already absolute `time` value without adding the day offset; validate timestamp/day consistency; fail clearly if any requested CSV or schema is incomplete.
- `main.py`: added `--smoke-test`, seeds deep models before initialization, saves all six estimators/checkpoints and histories, exports one held-out input/prediction example, generates evaluation artifacts, and records run metadata.
- `config/config.py`, `research/research_summary.md`, `research/algorithm_selection.md`, `research/literature_analysis.md`: corrected the bucket/schema claims and documented the limits on paper reproduction.

## Generated smoke artifacts

- `experiments/results/experiment_results.json`
- `experiments/results/comparison_table.txt`
- `experiments/results/run_record.json`
- `experiments/results/smoke_run.log`
- `experiments/results/demo_input_sequence.csv` (normalized 12-step example input)
- `experiments/results/demo_sample_predictions.csv` and `demo_sample_summary.txt` (held-out label and all model outputs)
- `experiments/results/models/{logistic_regression,random_forest,xgboost}.joblib`
- `experiments/results/models/{gru,bigru,abigru}.pt` and matching `*_history.json`
- `experiments/results/visualizations/`: dataset EDA, train/validation curves, ROC and precision-recall curves, six confusion matrices, and the comparison chart.

The smoke run does not write its five-function subset to `dataset/processed/processed_data.npz`, avoiding accidental reuse as if it were the full processed dataset. A normal run creates and reuses the full configured processed data cache.

## Remaining work

- Run normal full-scale processing/training on the configured 500 training-eligible functions, with early stopping up to 50 epochs for each deep model and the full RF/XGBoost tree counts. No full training has been run.
- Review full-scale memory and runtime on the target machine; dense function-by-bucket sequences may require substantial RAM.
- Perform model ablations, hyperparameter experiments, calibration, and repeated-seed evaluation only after full training is available.
- For a claim about incoming invocations or verified warm/cold labels, obtain the corresponding invocation/request dataset and redesign the task/feature set. The current event-only release cannot support that claim.
- Resolve the original paper's underspecified or unavailable architectural details before describing the code as a faithful reproduction.

## Commands

```bash
python -u main.py --smoke-test > experiments/results/smoke_run.log 2>&1
python main.py
```
