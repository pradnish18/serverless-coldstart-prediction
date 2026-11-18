"""
Feature engineering pipeline for cold-start prediction.

Transforms raw cold start event data into per-function time series features
suitable for sequence-based prediction models.

Approach:
    1. Bucket cold start events into fixed time windows (5-min buckets)
    2. For each function and time bucket, compute aggregate features
    3. Create sliding windows of past buckets as input sequences
    4. Target: whether the function will have a cold start in the next bucket
"""

import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from config.config import (
    TIME_BUCKET_SECONDS, SEQUENCE_LENGTH, NUM_FEATURES,
    MIN_COLD_STARTS_PER_FUNCTION, MAX_FUNCTIONS, PROCESSED_DATA_DIR,
    TRAIN_DAYS, VAL_DAYS, TEST_DAYS, RANDOM_SEED, NUM_DAYS
)


def create_time_buckets(df, bucket_size=TIME_BUCKET_SECONDS):
    """
    Assign each cold start event to a time bucket.
    
    Args:
        df: DataFrame with cold start events (must have 'abs_time' column)
        bucket_size: Size of each time bucket in seconds
    
    Returns:
        DataFrame with added 'time_bucket' column
    """
    df = df.copy()
    df['time_bucket'] = (df['abs_time'] // bucket_size).astype(int)
    return df


def compute_bucket_features(df, bucket_size=TIME_BUCKET_SECONDS):
    """
    Compute aggregate features per function per time bucket.
    
    For each (funcID, time_bucket) pair, we compute:
        - num_cold_starts: count of cold start events
        - avg_cold_start_cost: mean totalCost_cold_start
        - max_cold_start_cost: max totalCost_cold_start
        - avg_pod_allocation_cost: mean podAllocationCost
        - avg_deploy_code_cost: mean deployCodeCost
        - avg_deploy_dep_cost: mean deployDependencyCost
        - avg_scheduling_cost: mean schedulingCost
        - num_unique_clusters: number of distinct clusters
        - num_unique_pods: number of distinct pods
    """
    df = create_time_buckets(df, bucket_size)
    
    # Aggregate per function per bucket
    agg = df.groupby(['funcID', 'time_bucket']).agg(
        num_cold_starts=('totalCost_cold_start', 'count'),
        avg_cold_start_cost=('totalCost_cold_start', 'mean'),
        max_cold_start_cost=('totalCost_cold_start', 'max'),
        avg_pod_allocation_cost=('podAllocationCost', 'mean'),
        avg_deploy_code_cost=('deployCodeCost', 'mean'),
        avg_deploy_dep_cost=('deployDependencyCost', 'mean'),
        avg_scheduling_cost=('schedulingCost', 'mean'),
        num_unique_clusters=('clusterName', 'nunique'),
        num_unique_pods=('podID', 'nunique'),
    ).reset_index()
    
    return agg


def build_function_time_series(agg_df, func_id, total_buckets, bucket_size=TIME_BUCKET_SECONDS):
    """
    Build a complete time series for a single function.
    Fills in missing buckets with zeros (no cold starts in that bucket).
    Adds temporal features and historical rolling features.
    
    Returns:
        np.ndarray of shape (total_buckets, NUM_FEATURES)
    """
    func_data = agg_df[agg_df['funcID'] == func_id].copy()
    func_data = func_data.set_index('time_bucket')
    
    # Create full time range
    all_buckets = np.arange(total_buckets)
    ts = pd.DataFrame(index=all_buckets)
    
    # Join with aggregated data, fill missing buckets with 0
    feature_cols = [
        'num_cold_starts', 'avg_cold_start_cost', 'max_cold_start_cost',
        'avg_pod_allocation_cost', 'avg_deploy_code_cost', 'avg_deploy_dep_cost',
        'avg_scheduling_cost', 'num_unique_clusters', 'num_unique_pods'
    ]
    
    for col in feature_cols:
        ts[col] = 0.0
        if col in func_data.columns:
            valid = func_data.index.to_numpy(dtype=int)
            valid = valid[(valid >= 0) & (valid < total_buckets)]
            if len(valid):
                ts.loc[valid, col] = func_data.loc[valid, col].to_numpy()
    
    # Feature 10: time_since_last_cs (in number of buckets)
    cs_mask = ts['num_cold_starts'] > 0
    ts['time_since_last_cs'] = 0.0
    last_cs_bucket = -1
    for i in range(total_buckets):
        if cs_mask.iloc[i]:
            last_cs_bucket = i
            ts.iloc[i, ts.columns.get_loc('time_since_last_cs')] = 0.0
        elif last_cs_bucket >= 0:
            ts.iloc[i, ts.columns.get_loc('time_since_last_cs')] = float(i - last_cs_bucket)
        else:
            ts.iloc[i, ts.columns.get_loc('time_since_last_cs')] = float(i + 1)
    
    # Features 11-14: Temporal encoding (cyclical)
    bucket_abs_seconds = all_buckets * bucket_size
    hour_of_day = (bucket_abs_seconds % 86400) / 3600.0
    day_cycle = ((bucket_abs_seconds // 86400) % 7).astype(float)
    
    ts['hour_sin'] = np.sin(2 * np.pi * hour_of_day / 24.0)
    ts['hour_cos'] = np.cos(2 * np.pi * hour_of_day / 24.0)
    ts['day_cycle_sin'] = np.sin(2 * np.pi * day_cycle / 7.0)
    ts['day_cycle_cos'] = np.cos(2 * np.pi * day_cycle / 7.0)
    
    # Feature 15: cs_rate_1h (cold starts per bucket in past 1 hour = 12 buckets)
    buckets_1h = max(1, 3600 // bucket_size)
    ts['cs_rate_1h'] = ts['num_cold_starts'].rolling(window=buckets_1h, min_periods=1).mean().values
    
    # Feature 16: cs_rate_6h (cold starts per bucket in past 6 hours)
    buckets_6h = max(1, 6 * 3600 // bucket_size)
    ts['cs_rate_6h'] = ts['num_cold_starts'].rolling(window=buckets_6h, min_periods=1).mean().values
    
    # Feature 17: consecutive_cs_buckets
    ts['consecutive_cs'] = 0.0
    count = 0
    for i in range(total_buckets):
        if ts.iloc[i]['num_cold_starts'] > 0:
            count += 1
        else:
            count = 0
        ts.iloc[i, ts.columns.get_loc('consecutive_cs')] = float(count)
    
    # Select final features
    final_features = [
        'num_cold_starts', 'avg_cold_start_cost', 'max_cold_start_cost',
        'avg_pod_allocation_cost', 'avg_deploy_code_cost', 'avg_deploy_dep_cost',
        'avg_scheduling_cost', 'num_unique_clusters', 'num_unique_pods',
        'time_since_last_cs', 'hour_sin', 'hour_cos', 'day_cycle_sin', 'day_cycle_cos',
        'cs_rate_1h', 'cs_rate_6h', 'consecutive_cs'
    ]
    
    return ts[final_features].values.astype(np.float32)


def create_sequences(time_series, seq_length=SEQUENCE_LENGTH):
    """
    Create input-output pairs using sliding windows.
    
    Input: past seq_length buckets of features
    Target: whether there is a cold start in the next bucket
    
    Args:
        time_series: np.ndarray of shape (total_buckets, num_features)
        seq_length: number of past buckets to use as input
    
    Returns:
        X: np.ndarray of shape (num_samples, seq_length, num_features)
        y: np.ndarray of shape (num_samples,) - binary labels
        bucket_indices: np.ndarray - the target bucket index for each sample
    """
    total_buckets = len(time_series)
    X_list = []
    y_list = []
    idx_list = []
    
    for i in range(seq_length, total_buckets):
        # Input: features from buckets [i-seq_length, i)
        x = time_series[i - seq_length:i]
        # Target: whether bucket i has cold starts
        target = 1.0 if time_series[i, 0] > 0 else 0.0  # column 0 is num_cold_starts
        X_list.append(x)
        y_list.append(target)
        idx_list.append(i)
    
    if not X_list:
        return np.array([]), np.array([]), np.array([])
    
    return np.array(X_list), np.array(y_list), np.array(idx_list)


def process_dataset(df, bucket_size=TIME_BUCKET_SECONDS, seq_length=SEQUENCE_LENGTH,
                    save=True, max_functions=None):
    """
    Full feature engineering pipeline.
    
    Args:
        df: Raw cold start event DataFrame
        bucket_size: Time bucket size in seconds
        seq_length: Sequence length for sliding windows
        save: Whether to save processed data to disk
    
    Returns:
        dict with keys: 'X_train', 'y_train', 'X_val', 'y_val', 'X_test', 'y_test',
                        'feature_names', 'scaler_params'
    """
    np.random.seed(RANDOM_SEED)
    
    print("\n[1/5] Computing bucket aggregates...")
    agg_df = compute_bucket_features(df, bucket_size)
    
    # The trace represents 31 complete days, including empty buckets at the end.
    total_buckets = int(np.ceil((NUM_DAYS * 86400) / bucket_size))
    
    # Select functions with sufficient data
    train_cutoff = min(TRAIN_DAYS) * 86400 + (max(TRAIN_DAYS) - min(TRAIN_DAYS) + 1) * 86400
    train_cutoff_bucket = int(train_cutoff // bucket_size)
    train_agg = agg_df[agg_df['time_bucket'] < train_cutoff_bucket]
    func_cs_counts = train_agg.groupby('funcID')['num_cold_starts'].sum()
    active_functions = func_cs_counts[func_cs_counts >= MIN_COLD_STARTS_PER_FUNCTION].index.tolist()
    
    function_limit = MAX_FUNCTIONS if max_functions is None else max_functions
    if len(active_functions) > function_limit:
        # Take top MAX_FUNCTIONS by cold start count
        active_functions = func_cs_counts.loc[active_functions].nlargest(function_limit).index.tolist()
    
    print(f"  Selected {len(active_functions)} functions (from {agg_df['funcID'].nunique()} total)")
    
    print("\n[2/5] Building per-function time series...")
    all_X = []
    all_y = []
    all_buckets = []
    all_func_ids = []
    
    for func_id in tqdm(active_functions, desc="Processing functions"):
        ts = build_function_time_series(agg_df, func_id, total_buckets, bucket_size)
        X, y, bucket_indices = create_sequences(ts, seq_length)
        
        if len(X) > 0:
            all_X.append(X)
            all_y.append(y)
            all_buckets.append(bucket_indices)
            all_func_ids.extend([func_id] * len(X))
    
    X_all = np.concatenate(all_X, axis=0)
    y_all = np.concatenate(all_y, axis=0)
    buckets_all = np.concatenate(all_buckets, axis=0)
    func_ids_all = np.asarray(all_func_ids, dtype=str)
    
    print(f"  Total samples: {len(X_all):,}")
    print(f"  Positive (cold start): {int(y_all.sum()):,} ({100*y_all.mean():.1f}%)")
    print(f"  No cold-start event: {int((1-y_all).sum()):,} ({100*(1-y_all.mean()):.1f}%)")
    
    # Temporal split based on bucket index
    print("\n[3/5] Splitting data temporally...")
    train_end_bucket = int((max(TRAIN_DAYS) + 1) * 86400 / bucket_size)
    val_end_bucket = int((max(VAL_DAYS) + 1) * 86400 / bucket_size)

    train_mask = buckets_all < train_end_bucket
    val_mask = (buckets_all >= train_end_bucket) & (buckets_all < val_end_bucket)
    test_mask = (buckets_all >= val_end_bucket) & (buckets_all < total_buckets)
    
    X_train, y_train = X_all[train_mask], y_all[train_mask]
    X_val, y_val = X_all[val_mask], y_all[val_mask]
    X_test, y_test = X_all[test_mask], y_all[test_mask]
    func_ids_test = func_ids_all[test_mask]
    target_buckets_test = buckets_all[test_mask]
    
    print(f"  Train: {len(X_train):,} samples (pos: {int(y_train.sum()):,}, "
          f"{100*y_train.mean():.1f}%)")
    print(f"  Val:   {len(X_val):,} samples (pos: {int(y_val.sum()):,}, "
          f"{100*y_val.mean():.1f}%)")
    print(f"  Test:  {len(X_test):,} samples (pos: {int(y_test.sum()):,}, "
          f"{100*y_test.mean():.1f}%)")
    
    # Normalize features using training statistics (avoid data leakage)
    print("\n[4/5] Normalizing features (fit on train only)...")
    # Reshape for normalization: (samples * seq_len, features)
    if not len(X_train) or not len(X_val) or not len(X_test):
        raise ValueError("Temporal split produced an empty train, validation, or test set")
    train_flat = X_train.reshape(-1, X_train.shape[-1])
    
    feat_min = train_flat.min(axis=0)
    feat_max = train_flat.max(axis=0)
    feat_range = feat_max - feat_min
    feat_range[feat_range == 0] = 1.0  # Avoid division by zero
    
    # Apply Min-Max normalization (as per paper)
    X_train = (X_train - feat_min) / feat_range
    X_val = (X_val - feat_min) / feat_range
    X_test = (X_test - feat_min) / feat_range
    
    # Clip to handle potential out-of-range values in val/test
    X_train = np.clip(X_train, 0, 1)
    X_val = np.clip(X_val, 0, 1)
    X_test = np.clip(X_test, 0, 1)
    
    feature_names = [
        'num_cold_starts', 'avg_cold_start_cost', 'max_cold_start_cost',
        'avg_pod_allocation_cost', 'avg_deploy_code_cost', 'avg_deploy_dep_cost',
        'avg_scheduling_cost', 'num_unique_clusters', 'num_unique_pods',
        'time_since_last_cs', 'hour_sin', 'hour_cos', 'day_cycle_sin', 'day_cycle_cos',
        'cs_rate_1h', 'cs_rate_6h', 'consecutive_cs'
    ]
    
    result = {
        'X_train': X_train, 'y_train': y_train,
        'X_val': X_val, 'y_val': y_val,
        'X_test': X_test, 'y_test': y_test,
        'test_func_ids': func_ids_test,
        'test_target_buckets': target_buckets_test,
        'feature_names': feature_names,
        'scaler_params': {'min': feat_min, 'max': feat_max, 'range': feat_range},
    }
    
    if save:
        print("\n[5/5] Saving processed data...")
        save_path = os.path.join(PROCESSED_DATA_DIR, 'processed_data.npz')
        np.savez_compressed(
            save_path,
            X_train=X_train, y_train=y_train,
            X_val=X_val, y_val=y_val,
            X_test=X_test, y_test=y_test,
            test_func_ids=func_ids_test, test_target_buckets=target_buckets_test,
            feature_names=feature_names,
            scaler_min=feat_min, scaler_max=feat_max, scaler_range=feat_range,
        )
        print(f"  Saved to {save_path}")
    
    return result


def load_processed_data():
    """Load previously processed data from disk."""
    path = os.path.join(PROCESSED_DATA_DIR, 'processed_data.npz')
    if not os.path.exists(path):
        raise FileNotFoundError(f"Processed data not found at {path}. Run feature engineering first.")
    
    data = np.load(path, allow_pickle=True)
    return {
        'X_train': data['X_train'], 'y_train': data['y_train'],
        'X_val': data['X_val'], 'y_val': data['y_val'],
        'X_test': data['X_test'], 'y_test': data['y_test'],
        'test_func_ids': data['test_func_ids'] if 'test_func_ids' in data.files else None,
        'test_target_buckets': data['test_target_buckets'] if 'test_target_buckets' in data.files else None,
        'feature_names': data['feature_names'].tolist(),
        'scaler_params': {
            'min': data['scaler_min'],
            'max': data['scaler_max'],
            'range': data['scaler_range'],
        },
    }
