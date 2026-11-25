> **Historical (v1 plan).** This summary describes the archived v1 design (12-bucket window, six models). The current pipeline is `v2/` (57 features, 24×7 sequence window, rules, XGBoost, MLP/GRU/BiGRU/ABiGRU, pre-warming simulator); see [`../docs/PROJECT_DOCUMENTATION.md`](../docs/PROJECT_DOCUMENTATION.md) and [`../README.md`](../README.md).

# Research Summary: Cold Start Prediction for Serverless Computing

## Project Overview

This project implements GRU, BiGRU, and an attention-based BiGRU variant alongside three classical baselines. It uses the Huawei 2025 Region 5 cold-start event release. The available CSVs contain cold-start events only; they are not invocation or complete function telemetry traces.

---

## Research Objectives

1. Evaluate a clearly specified adaptation of ABiGRU on cold-start event traces.
2. Predict whether a function has one or more cold-start events in the next five-minute bucket from the preceding 12 buckets.
3. Compare Logistic Regression, Random Forest, XGBoost, GRU, BiGRU, and ABiGRU with temporal splits.
4. Use only event fields that exist in the selected CSV schema and explicitly derived history.
5. Apply train-only feature scaling, temporal splitting, class imbalance handling, and reproducible seeds.

---

## Methodology

### Problem Statement

Given 12 past five-minute buckets (one hour) for a function/user/pod-pool key, predict whether its next five-minute bucket contains at least one recorded cold-start event. Since the release contains cold-start records only, empty buckets mean “no recorded cold start,” not “no invocation.”

### Dataset

**Huawei Cloud Production FaaS Traces 2025, Region 5 cold starts** — 31 daily CSV files with cold-start event time, function/user/cluster/pod identifiers, request ID, and cold-start cost components. The selected files do not provide request counts, invocation-level non-cold-start records, CPU/memory utilization, or scaling events.

### Models

| Model | Type | Role |
|-------|------|------|
| Logistic Regression | Linear | Simple baseline |
| Random Forest | Ensemble (trees) | Non-linear baseline |
| XGBoost | Gradient boosting | Strong tabular baseline |
| GRU | Recurrent neural network | Sequential baseline |
| BiGRU | Bidirectional RNN | Bidirectional baseline |
| **ABiGRU** | **Attention + BiGRU** | **Proposed model (from paper)** |

### Evaluation

- **Metrics**: Accuracy, Precision, Recall, F1-Score, MCC, ROC-AUC, PR-AUC
- **Buckets**: 300 seconds (five minutes); 12 input buckets; label is the following bucket
- **Split**: Temporal — Train days 0–23, validation days 24–26, test days 27–30
- **Seed**: 42 for all random operations

---

## Expected Outcomes

1. A validated comparison of ABiGRU against baselines on real production data
2. Quantified contribution of attention mechanism and bidirectionality through ablation
3. Insights into which event-derived historical features are predictive of subsequent recorded cold starts
4. A practical, reproducible prediction pipeline applicable to FaaS platforms

---

## Key Documents

| Document | Description |
|----------|-------------|
| [`literature_analysis.md`](literature_analysis.md) | Detailed analysis of the original paper, methodology, results, limitations, and gap analysis |
| [`algorithm_selection.md`](algorithm_selection.md) | Algorithm portfolio, feature mapping, problem reformulation, and experimental design |
