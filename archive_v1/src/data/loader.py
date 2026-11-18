"""
Data loader for Huawei Cloud cold start event data.

Loads daily CSV files from Region 5, combining them into a single DataFrame.
"""

import os
import pandas as pd
import numpy as np
from glob import glob
from tqdm import tqdm

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from config.config import COLD_START_DATA_DIR, NUM_DAYS, RAW_DATA_DIR


def load_cold_start_data(data_dir=None, days=None):
    """
    Load cold start event data from CSV files.
    
    Args:
        data_dir: Path to directory containing day_XX.csv files.
                  Defaults to COLD_START_DATA_DIR from config.
        days: List of day indices to load. Defaults to all days (0-30).
    
    Returns:
        pd.DataFrame: Combined cold start events with columns:
            day, time, clusterName, funcName, userID, requestID,
            totalCost_cold_start, podAllocationCost, deployCodeCost,
            deployDependencyCost, schedulingCost, podID
    """
    if data_dir is None:
        data_dir = COLD_START_DATA_DIR
    
    if days is None:
        days = list(range(NUM_DAYS))
    
    dfs = []
    missing_files = []
    for day in tqdm(days, desc="Loading cold start data"):
        filepath = os.path.join(data_dir, f"day_{day:02d}.csv")
        if os.path.exists(filepath):
            df = pd.read_csv(filepath)
            dfs.append(df)
        else:
            missing_files.append(filepath)

    if missing_files:
        raise FileNotFoundError(f"Missing {len(missing_files)} requested daily CSV file(s): {missing_files[:3]}")
    
    if not dfs:
        raise FileNotFoundError(f"No data files found in {data_dir}")
    
    df_all = pd.concat(dfs, ignore_index=True)

    required_columns = {
        'day', 'time', 'clusterName', 'funcName', 'userID', 'requestID',
        'totalCost_cold_start', 'podAllocationCost', 'deployCodeCost',
        'deployDependencyCost', 'schedulingCost', 'podID',
    }
    missing_columns = sorted(required_columns - set(df_all.columns))
    if missing_columns:
        raise ValueError(f"Cold-start CSV schema is missing required columns: {missing_columns}")
    
    # Sort by absolute time
    df_all = df_all.sort_values(['day', 'time']).reset_index(drop=True)
    
    # In this release, `time` is already seconds since day 0 (it does not reset
    # daily). Adding day*86400 here would shift days 1-30 a second time.
    df_all['abs_time'] = pd.to_numeric(df_all['time'], errors='raise')
    day_start = pd.to_numeric(df_all['day'], errors='raise') * 86400
    invalid_time = (df_all['abs_time'] < day_start) | (df_all['abs_time'] >= day_start + 86400)
    if invalid_time.any():
        raise ValueError(
            "CSV time values are expected to be absolute seconds from day 0 and fall "
            "within the day column's 24-hour interval"
        )
    
    # Create funcID (unique function identifier combining funcName, userID, podPool)
    df_all['poolName'] = df_all['podID'].apply(extract_pool_name)
    df_all['funcID'] = (df_all['funcName'].astype(str) + '---' + 
                        df_all['userID'].astype(str) + '---' + 
                        df_all['poolName'])
    
    return df_all


def extract_pool_name(pod_id):
    """Extract pool name from podID (e.g., pool22-300-128 from pool22-300-128-0000577863)."""
    if pd.isna(pod_id) or not isinstance(pod_id, str):
        return "unknown"
    parts = pod_id.rsplit('-', 1)
    return parts[0] if len(parts) == 2 else pod_id


def get_dataset_summary(df):
    """Print summary statistics of the loaded dataset."""
    print("\n" + "=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)
    print(f"Total cold start events: {len(df):,}")
    print(f"Date range: Day {df['day'].min()} to Day {df['day'].max()}")
    print(f"Time range: {df['abs_time'].min():.1f}s to {df['abs_time'].max():.1f}s")
    print(f"Unique functions (funcName): {df['funcName'].nunique()}")
    print(f"Unique users: {df['userID'].nunique()}")
    print(f"Unique funcIDs: {df['funcID'].nunique()}")
    print(f"Unique clusters: {df['clusterName'].nunique()}")
    print(f"Unique pods: {df['podID'].nunique()}")
    print(f"\nCold start cost statistics:")
    print(df['totalCost_cold_start'].describe())
    print(f"\nEvents per day:")
    print(df.groupby('day').size().describe())
    print("=" * 60)
    
    return {
        'total_events': len(df),
        'num_functions': df['funcName'].nunique(),
        'num_funcIDs': df['funcID'].nunique(),
        'num_users': df['userID'].nunique(),
        'num_clusters': df['clusterName'].nunique(),
        'avg_cs_cost': df['totalCost_cold_start'].mean(),
        'median_cs_cost': df['totalCost_cold_start'].median(),
    }


if __name__ == "__main__":
    df = load_cold_start_data()
    summary = get_dataset_summary(df)
    print(f"\nDataset loaded successfully with {len(df):,} records.")
