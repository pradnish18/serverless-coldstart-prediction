# Literature Analysis: Cold-Start Prediction in Serverless Computing

## 1. Problem Statement

Serverless computing (FaaS) allows developers to deploy functions without managing infrastructure. However, when a function is invoked without a pre-initialized execution environment, the platform must create one from scratch — this is a **cold start**. Cold starts introduce significant latency (often seconds) compared to warm starts (milliseconds).

### Cold Start Causes
- First-time function invocation
- Long periods of inactivity (container recycled)
- Execution environment termination by the platform
- Sudden traffic spikes (horizontal scaling)
- Increased concurrent invocation demand

### Research Objective
Predict whether an incoming serverless invocation will experience a cold start, enabling proactive resource provisioning to keep environments warm.

---

## 2. Original Paper Methodology

**Paper**: "Cold Start Prediction and Provisioning Optimization in Serverless Computing Using Deep Learning"  
**Authors**: N. Saravana Kumar, S. Selvakumara Samy (2025)  
**Publication**: Concurrency and Computation: Practice and Experience, Vol 37, Article e8392

The publisher abstract identifies an AWS Lambda, invocation-level prediction setting, lists input concepts, and reports the testing metrics below. The article's full text was not available in the project materials during this audit. Therefore the equations and layer choices below are standard implementation choices for this project's adaptation, not a verified exact transcription of every detail in the paper. See the [publisher record](https://doi.org/10.1002/cpe.8392).

### 2.1 Architecture: ABiGRU (Attention-Based Bidirectional GRU)

The proposed architecture combines:

1. **Input Layer** → Feature vector X = [I, T, L, D, HCS, ET, CSI, CCS, TLCS]
2. **Bidirectional GRU** → Captures temporal dependencies in both directions
3. **Attention Mechanism** → Weights important time steps
4. **Max Pooling** → Extracts strongest signals
5. **Fully Connected Layer** → Combines features for classification
6. **Sigmoid Output** → Cold start probability P(CSI=1|X)

### 2.2 GRU Gate Equations

**Update gate**: z_t = σ(W_z·x_t + U_z·h_{t-1})

**Reset gate**: r_t = σ(W_r·x_t + U_r·h_{t-1})

**Candidate state**: h̃_t = tanh(W_h·x_t + U_h·(r_t ⊙ h_{t-1}))

**Hidden state**: h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ h̃_t

### 2.3 Attention Mechanism

- e_t = score(h_t) = v^T · tanh(W_a·h_t + b_a)
- α_t = exp(e_t) / Σ_j exp(e_j)
- c = Σ_t α_t · h_t

### 2.4 Training Configuration
- The publisher abstract names RMSProp, but does not expose enough detail to establish the exact hyperparameters used by the code here.
- This repository's learning rate, decay, epochs, and architecture widths are initial implementation settings and must not be described as exact paper values.
- The original study is an AWS Lambda invocation-level setting; its data and sample-level availability are not reproduced in this repository.

---

## 3. Original Feature Set (9 Features)

| # | Feature | Symbol | Description |
|---|---------|--------|-------------|
| 1 | Invocation Frequency | I | How frequently a function is invoked |
| 2 | Timestamp | T | Temporal patterns in invocation |
| 3 | Latency | L | Execution latency |
| 4 | Duration | D | Execution duration |
| 5 | Historical Cold Starts | HCS | Past cold start occurrences |
| 6 | Event Type | ET | Trigger source (S3, CloudWatch, API Gateway) |
| 7 | Cold Start Indicator | CSI | **Target variable** (1=cold, 0=warm) |
| 8 | Consecutive Cold Starts | CCS | Consecutive cold start count |
| 9 | Time Since Last Cold Start | TLCS | Seconds since previous cold start |

---

## 4. Original Reported Results

| Metric | Testing (publisher abstract) |
|--------|-------------------------------|
| Accuracy | 90.36% |
| Precision | 91.87% |
| Recall | 90.42% |
| F1-Score | 90.28% |
| MCC | 82.28% |

---

## 5. Critical Limitations Identified

### 5.1 Dataset Limitations
- Dataset is not publicly available (available from authors upon request)
- Only ~10,000 records — very small for deep learning
- AWS Lambda specific — may not generalize to other platforms

### 5.2 Methodological Concerns
- **Data leakage risk**: Duration and Latency are post-invocation metrics that should not be used as input features for pre-invocation prediction
- **Feature CSI dual role**: CSI is both the target and listed as a feature — must be historical only
- **Bidirectional processing**: Must not access future events during prediction
- **Class imbalance**: Not addressed; accuracy alone is insufficient

### 5.3 Reproducibility Issues
- No public code or dataset
- Exact data preprocessing steps are underspecified
- Comparison baselines (ACSPRS-FP, HRAACSP, etc.) are not publicly implemented

---

## 6. Gap Analysis: Original Paper vs. New Dataset

### New Dataset: Huawei Cloud Production FaaS Traces 2025
- **Source**: sir-lab/data-release (GitHub)
- **Scale**: ~11.9 million cold start events across 5 regions, 31 days
- **Platform**: Huawei Cloud (not AWS Lambda)

### Key Differences

| Aspect | Original Paper | New Dataset |
|--------|---------------|-------------|
| Platform | AWS Lambda | Huawei Cloud |
| Scale | ~10,000 records | ~800,000 events (Region 5) |
| Duration | Not specified | 31 days |
| Granularity | Per-invocation | Per cold-start event |
| Warm starts | Explicitly labeled | Not present in selected event-only CSVs |
| Event types | S3, CloudWatch, etc. | Not available |
| Latency/Duration | Available | Available (cold start cost breakdown) |
| Request-level data | Yes (mixed warm+cold) | Not used; selected files are cold-start events only |

### Adaptation Strategy
Since the new dataset contains only cold start events (not all invocations), we reformulate the prediction problem:

**New formulation**: Given a function's historical cold-start event patterns aggregated into 5-minute buckets, predict whether a cold-start event appears in the next bucket. This is a bucket-level proxy forecast. Because the input release contains only cold-start events, an empty bucket does not establish that invocations occurred or that they were warm.

This preserves the core research hypothesis (temporal pattern recognition for cold start prediction) while adapting to the available data structure.

This is an adaptation, not an exact reproduction: the event-only Huawei data cannot support the source paper's incoming-invocation labels or its full invocation feature set, and the publisher abstract alone does not specify all architecture details needed for exact reproduction.
