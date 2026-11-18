"""
Configuration for Cold-Start Prediction System.
Adapted from: "Cold Start Prediction and Provisioning Optimization in Serverless Computing
Using Deep Learning" (Saravana Kumar & Selvakumara Samy, 2025)

New dataset: Huawei Cloud Production FaaS Traces 2025 (sir-lab/data-release)
"""

import os

# ============================================================================
# Project Paths
# ============================================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset")
RAW_DATA_DIR = os.path.join(DATASET_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATASET_DIR, "processed")
COLD_START_DATA_DIR = os.path.join(RAW_DATA_DIR, "region5_cold_starts", "R5")
EXPERIMENTS_DIR = os.path.join(PROJECT_ROOT, "experiments")
RESULTS_DIR = os.path.join(EXPERIMENTS_DIR, "results")
VISUALIZATION_DIR = os.path.join(RESULTS_DIR, "visualizations")

# Ensure directories exist
for d in [PROCESSED_DATA_DIR, RESULTS_DIR, VISUALIZATION_DIR]:
    os.makedirs(d, exist_ok=True)

# ============================================================================
# Data Configuration
# ============================================================================
# Number of days in the dataset (0-30)
NUM_DAYS = 31

# Temporal split: train/val/test by day
TRAIN_DAYS = list(range(0, 24))       # Days 0-23 (24 days)
VAL_DAYS = list(range(24, 27))        # Days 24-26 (3 days)
TEST_DAYS = list(range(27, 31))       # Days 27-30 (4 days)

# Time bucket size in seconds for aggregation
# We use 5-minute (300s) buckets to get meaningful patterns
TIME_BUCKET_SECONDS = 300  # 5 minutes

# Sequence window length (number of past time buckets for prediction)
SEQUENCE_LENGTH = 12  # Look back 12 * 5min = 1 hour

# Minimum number of cold start events for a function to be included
MIN_COLD_STARTS_PER_FUNCTION = 10

# Maximum number of functions to process (for memory management on laptop)
MAX_FUNCTIONS = 500

# ============================================================================
# Feature Configuration
# ============================================================================
# Features engineered per time bucket per function:
# 1. num_cold_starts: Number of cold starts in this bucket
# 2. avg_cold_start_cost: Average total cold start time
# 3. max_cold_start_cost: Max total cold start time
# 4. avg_pod_allocation_cost: Average pod allocation time
# 5. avg_deploy_code_cost: Average code deployment time
# 6. avg_deploy_dep_cost: Average dependency deployment time
# 7. avg_scheduling_cost: Average scheduling overhead
# 8. num_unique_clusters: Number of distinct clusters used
# 9. num_unique_pods: Number of distinct pods used
# 10. time_since_last_cs: Time since last cold start (seconds)
# 11. hour_of_day: Hour of day (cyclical encoded - sin)
# 12. hour_of_day_cos: Hour of day (cyclical encoded - cos)
# 13. day_cycle_sin: Collection-day cycle (0-6; not calendar weekday)
# 14. day_cycle_cos: Collection-day cycle (0-6; not calendar weekday)
# 15. cs_rate_1h: Cold start rate over past 1 hour
# 16. cs_rate_6h: Cold start rate over past 6 hours
# 17. consecutive_cs_buckets: Number of consecutive buckets with cold starts

NUM_FEATURES = 17

# Target: binary classification
# 1 = at least one cold start in the next time bucket
# 0 = no cold starts in the next time bucket

# ============================================================================
# Model Configuration
# ============================================================================
# ABiGRU Architecture (adapted from paper)
GRU_HIDDEN_SIZE = 64
GRU_NUM_LAYERS = 2
ATTENTION_DIM = 32
DROPOUT_RATE = 0.3
FC_HIDDEN_SIZE = 64

# Initial training configuration; these are implementation settings, not a claim
# that the published paper specifies/requires these exact values.
OPTIMIZER = "rmsprop"
LEARNING_RATE = 0.001
RMS_DECAY = 0.9
BATCH_SIZE = 256
NUM_EPOCHS = 50
EARLY_STOPPING_PATIENCE = 7

# Classification threshold (paper uses 0.5, we'll also optimize)
CLASSIFICATION_THRESHOLD = 0.5

# ============================================================================
# Reproducibility
# ============================================================================
RANDOM_SEED = 42

# ============================================================================
# Device
# ============================================================================
import torch
DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
