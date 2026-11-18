#!/usr/bin/env python3
"""
Cold-Start Prediction System - Main Entry Point

Adapted from: "Cold Start Prediction and Provisioning Optimization in
Serverless Computing Using Deep Learning" (Saravana Kumar & Selvakumara Samy, 2025)

Dataset: Huawei Cloud Production FaaS Traces 2025 (Region 5)
Source: https://github.com/sir-lab/data-release

Pipeline:
    1. Load raw cold start event data
    2. Exploratory data analysis & visualization
    3. Feature engineering (time-bucketed sequences)
    4. Train baseline models (LR, RF, XGBoost)
    5. Train deep learning models (GRU, BiGRU, ABiGRU)
    6. Evaluate all models
    7. Generate comparison results & visualizations
"""

import os
import sys
import json
import time
import warnings
import argparse
import csv

# The runtime exposes hundreds of logical CPUs; cap native thread pools so
# OpenMP/XGBoost do not attempt to create an invalid number of worker threads.
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('MPLCONFIGDIR', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'experiments', '.mplconfig'))
os.makedirs(os.environ['MPLCONFIGDIR'], exist_ok=True)
import numpy as np
from datetime import datetime, timezone

warnings.filterwarnings('ignore')

# Project root
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from config.config import (
    RESULTS_DIR, VISUALIZATION_DIR, PROCESSED_DATA_DIR,
    RANDOM_SEED, NUM_FEATURES, SEQUENCE_LENGTH, TIME_BUCKET_SECONDS,
    GRU_HIDDEN_SIZE, GRU_NUM_LAYERS, ATTENTION_DIM,
    DROPOUT_RATE, FC_HIDDEN_SIZE, DEVICE
)


def phase1_load_data():
    """Phase 1: Load raw cold start data."""
    print("\n" + "=" * 70)
    print("PHASE 1: LOADING RAW DATASET")
    print("=" * 70)
    
    from src.data.loader import load_cold_start_data, get_dataset_summary
    
    df = load_cold_start_data()
    summary = get_dataset_summary(df)
    
    return df, summary


def phase2_eda(df):
    """Phase 2: Exploratory Data Analysis & Visualizations."""
    print("\n" + "=" * 70)
    print("PHASE 2: EXPLORATORY DATA ANALYSIS")
    print("=" * 70)
    
    from src.visualization.plots import plot_cold_start_distribution, plot_hourly_pattern
    
    print("\nGenerating dataset visualizations...")
    plot_cold_start_distribution(df)
    plot_hourly_pattern(df)
    
    # Additional EDA statistics
    print("\n--- Feature Statistics ---")
    cost_cols = ['totalCost_cold_start', 'podAllocationCost', 'deployCodeCost',
                 'deployDependencyCost', 'schedulingCost']
    for col in cost_cols:
        print(f"  {col}:")
        print(f"    Mean: {df[col].mean():.4f}s, Median: {df[col].median():.4f}s, "
              f"Std: {df[col].std():.4f}s")
    
    print(f"\n--- Missing Values ---")
    missing = df.isnull().sum()
    if missing.sum() == 0:
        print("  No missing values found.")
    else:
        print(missing[missing > 0])
    
    print(f"\n--- Duplicate Records ---")
    n_dup = df.duplicated(subset=['day', 'time', 'funcName', 'userID']).sum()
    print(f"  Duplicates (same time+func+user): {n_dup}")
    
    # Cold start pattern analysis
    print(f"\n--- Cold Start Patterns ---")
    funcs_per_day = df.groupby('day')['funcName'].nunique()
    print(f"  Avg unique functions per day with cold starts: {funcs_per_day.mean():.0f}")
    print(f"  Functions with cold starts across all 31 days: "
          f"{(df.groupby('funcName')['day'].nunique() == 31).sum()}")


def phase3_feature_engineering(df, smoke_test=False):
    """Phase 3: Feature Engineering."""
    print("\n" + "=" * 70)
    print("PHASE 3: FEATURE ENGINEERING")
    print("=" * 70)
    
    from src.feature_engineering.features import process_dataset, load_processed_data
    
    processed_path = os.path.join(PROCESSED_DATA_DIR, 'processed_data.npz')
    
    if os.path.exists(processed_path) and not smoke_test:
        print("  Loading previously processed data...")
        data = load_processed_data()
    else:
        print("  Processing raw data into sequences...")
        data = process_dataset(df, save=not smoke_test, max_functions=5 if smoke_test else None)
    
    print(f"\n  Feature dimensions: {data['X_train'].shape}")
    print(f"  Features: {data['feature_names']}")
    
    return data


def phase4_train_baselines(data, smoke_test=False):
    """Phase 4: Train baseline ML models."""
    print("\n" + "=" * 70)
    print("PHASE 4: TRAINING BASELINE MODELS")
    print("=" * 70)
    
    from src.models.baselines import (
        train_logistic_regression, train_random_forest,
        train_xgboost, predict_baseline, save_baseline
    )
    from src.evaluation.metrics import compute_all_metrics, print_metrics, find_optimal_threshold
    from src.visualization.plots import plot_confusion_matrix
    
    X_train, y_train = data['X_train'], data['y_train']
    X_val, y_val = data['X_val'], data['y_val']
    X_test, y_test = data['X_test'], data['y_test']
    
    baseline_results = {}

    if smoke_test:
        # Deterministic, evenly spaced subsamples from the real temporal splits.
        test_indices = np.arange(len(data['y_test']))
        for key in ('X_train', 'y_train', 'X_val', 'y_val', 'X_test', 'y_test'):
            arr = data[key]
            if len(arr) > 2048:
                idx = np.linspace(0, len(arr) - 1, 2048, dtype=int)
                data[key] = arr[idx]
                if key == 'X_test':
                    test_indices = idx
        for key in ('test_func_ids', 'test_target_buckets'):
            if data.get(key) is not None:
                data[key] = data[key][test_indices]
        X_train, y_train = data['X_train'], data['y_train']
        X_val, y_val = data['X_val'], data['y_val']
        X_test, y_test = data['X_test'], data['y_test']
    
    # 1. Logistic Regression
    print("\n--- Training Logistic Regression ---")
    t0 = time.time()
    lr_model = train_logistic_regression(X_train, y_train, seed=RANDOM_SEED)
    save_baseline(lr_model, os.path.join(RESULTS_DIR, 'models', 'logistic_regression.joblib'))
    lr_time = time.time() - t0
    lr_probs_val = predict_baseline(lr_model, X_val)
    lr_probs_test = predict_baseline(lr_model, X_test)
    lr_thresh = find_optimal_threshold(y_val, lr_probs_val, 'f1')
    lr_metrics = compute_all_metrics(y_test, lr_probs_test, lr_thresh)
    print_metrics(lr_metrics, "Logistic Regression")
    plot_confusion_matrix(lr_metrics['confusion_matrix'], 'Logistic Regression')
    baseline_results['Logistic Regression'] = {
        'metrics': lr_metrics, 'y_prob': lr_probs_test,
        'threshold': lr_thresh, 'time': lr_time,
    }
    
    # 2. Random Forest
    print("\n--- Training Random Forest ---")
    t0 = time.time()
    rf_model = train_random_forest(X_train, y_train, seed=RANDOM_SEED,
                                   n_estimators=25 if smoke_test else 200)
    save_baseline(rf_model, os.path.join(RESULTS_DIR, 'models', 'random_forest.joblib'))
    rf_time = time.time() - t0
    rf_probs_val = predict_baseline(rf_model, X_val)
    rf_probs_test = predict_baseline(rf_model, X_test)
    rf_thresh = find_optimal_threshold(y_val, rf_probs_val, 'f1')
    rf_metrics = compute_all_metrics(y_test, rf_probs_test, rf_thresh)
    print_metrics(rf_metrics, "Random Forest")
    plot_confusion_matrix(rf_metrics['confusion_matrix'], 'Random Forest')
    baseline_results['Random Forest'] = {
        'metrics': rf_metrics, 'y_prob': rf_probs_test,
        'threshold': rf_thresh, 'time': rf_time,
    }
    
    # 3. XGBoost
    print("\n--- Training XGBoost ---")
    t0 = time.time()
    xgb_model = train_xgboost(X_train, y_train, X_val, y_val, seed=RANDOM_SEED,
                              n_estimators=15 if smoke_test else 300)
    save_baseline(xgb_model, os.path.join(RESULTS_DIR, 'models', 'xgboost.joblib'))
    xgb_time = time.time() - t0
    xgb_probs_val = predict_baseline(xgb_model, X_val)
    xgb_probs_test = predict_baseline(xgb_model, X_test)
    xgb_thresh = find_optimal_threshold(y_val, xgb_probs_val, 'f1')
    xgb_metrics = compute_all_metrics(y_test, xgb_probs_test, xgb_thresh)
    print_metrics(xgb_metrics, "XGBoost")
    plot_confusion_matrix(xgb_metrics['confusion_matrix'], 'XGBoost')
    baseline_results['XGBoost'] = {
        'metrics': xgb_metrics, 'y_prob': xgb_probs_test,
        'threshold': xgb_thresh, 'time': xgb_time,
    }
    
    return baseline_results


def phase5_train_deep_models(data, smoke_test=False):
    """Phase 5: Train deep learning models (GRU, BiGRU, ABiGRU)."""
    print("\n" + "=" * 70)
    print("PHASE 5: TRAINING DEEP LEARNING MODELS")
    print("=" * 70)
    
    from src.models.deep_models import create_model
    from src.training.trainer import train_model, predict_deep_model, set_seed
    from src.evaluation.metrics import compute_all_metrics, print_metrics, find_optimal_threshold
    from src.visualization.plots import plot_training_curves, plot_confusion_matrix
    
    X_train, y_train = data['X_train'], data['y_train']
    X_val, y_val = data['X_val'], data['y_val']
    X_test, y_test = data['X_test'], data['y_test']
    
    input_dim = X_train.shape[-1]  # Number of features
    
    model_config = {
        'hidden_dim': GRU_HIDDEN_SIZE,
        'num_layers': GRU_NUM_LAYERS,
        'attention_dim': ATTENTION_DIM,
        'dropout': DROPOUT_RATE,
        'fc_dim': FC_HIDDEN_SIZE,
    }
    
    deep_results = {}
    
    models_to_train = [
        ('gru', 'Standard GRU'),
        ('bigru', 'Bidirectional GRU'),
        ('abigru', 'ABiGRU (Proposed)'),
    ]
    
    for model_type, model_name in models_to_train:
        print(f"\n{'='*50}")
        print(f"Training: {model_name}")
        print(f"{'='*50}")
        
        # Seed before model construction so initial weights are reproducible.
        set_seed(RANDOM_SEED)
        model = create_model(model_type, input_dim, model_config)
        print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        checkpoint_path = os.path.join(RESULTS_DIR, 'models', f'{model_type}.pt')
        training_config = {'checkpoint_path': checkpoint_path}
        if smoke_test:
            training_config.update({'epochs': 1, 'batch_size': 256, 'patience': 1})
        result = train_model(model, X_train, y_train, X_val, y_val, config=training_config)
        trained_model = result['model']
        
        # Plot training curves
        plot_training_curves(result['history'], model_name)
        
        # Get predictions
        val_probs = predict_deep_model(trained_model, X_val)
        test_probs = predict_deep_model(trained_model, X_test)
        
        # Find optimal threshold on validation set
        optimal_thresh = find_optimal_threshold(y_val, val_probs, 'f1')
        
        # Evaluate on test set
        metrics = compute_all_metrics(y_test, test_probs, optimal_thresh)
        print_metrics(metrics, model_name)
        
        # Plot confusion matrix
        plot_confusion_matrix(metrics['confusion_matrix'], model_name)
        
        deep_results[model_name] = {
            'metrics': metrics,
            'y_prob': test_probs,
            'threshold': optimal_thresh,
            'time': result['training_time'],
            'history': result['history'],
            'best_epoch': result['best_epoch'],
        }
    
    return deep_results


def phase6_comparison(baseline_results, deep_results, y_test, smoke_test=False, data=None):
    """Phase 6: Compare all models and generate final results."""
    print("\n" + "=" * 70)
    print("PHASE 6: MODEL COMPARISON & FINAL RESULTS")
    print("=" * 70)
    
    from src.evaluation.metrics import create_comparison_table
    from src.visualization.plots import (
        plot_roc_curves, plot_pr_curves, plot_model_comparison_bars
    )
    
    # Combine all results
    all_results = {**baseline_results, **deep_results}

    # Save one deterministic held-out example so a reviewer can inspect the
    # exact model input, actual next-bucket label, and each prediction.
    if data is not None and len(y_test):
        example_index = 0
        example_x = data['X_test'][example_index]
        feature_names = data['feature_names']
        input_path = os.path.join(RESULTS_DIR, 'demo_input_sequence.csv')
        with open(input_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['history_step'] + feature_names)
            for step, feature_row in enumerate(example_x, start=-len(example_x)):
                writer.writerow([step] + [float(v) for v in feature_row])

        target_bucket = None
        func_id = None
        if data.get('test_target_buckets') is not None:
            target_bucket = int(data['test_target_buckets'][example_index])
        if data.get('test_func_ids') is not None:
            func_id = str(data['test_func_ids'][example_index])
        actual_label = int(y_test[example_index])
        target_start_seconds = (target_bucket * TIME_BUCKET_SECONDS) if target_bucket is not None else None
        target_day = int(target_start_seconds // 86400) if target_start_seconds is not None else None
        seconds_in_day = int(target_start_seconds % 86400) if target_start_seconds is not None else None
        sample_rows = []
        for name, model_data in all_results.items():
            probability = float(model_data['y_prob'][example_index])
            threshold = float(model_data.get('threshold', 0.5))
            predicted_label = int(probability >= threshold)
            sample_rows.append({
                'model': name,
                'probability_cold_start_next_bucket': probability,
                'threshold': threshold,
                'predicted_label': predicted_label,
                'actual_label': actual_label,
                'correct': predicted_label == actual_label,
            })
        prediction_path = os.path.join(RESULTS_DIR, 'demo_sample_predictions.csv')
        with open(prediction_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(sample_rows[0]))
            writer.writeheader()
            writer.writerows(sample_rows)

        summary_path = os.path.join(RESULTS_DIR, 'demo_sample_summary.txt')
        with open(summary_path, 'w') as f:
            f.write('Held-out smoke-test example (actual Region 5 trace data)\n')
            f.write('Input: 12 historical five-minute buckets, normalized with training-set statistics.\n')
            f.write('Target: whether the immediately following bucket contains a recorded cold-start event.\n')
            f.write('Label 1 = event recorded; label 0 = no cold-start event recorded (not proof of a warm invocation).\n')
            f.write(f'Test subset row: {example_index}\n')
            f.write(f'Function key: {func_id or "not available"}\n')
            f.write(f'Target bucket index: {target_bucket if target_bucket is not None else "not available"}\n')
            if target_start_seconds is not None:
                start_hour, start_minute = seconds_in_day // 3600, (seconds_in_day % 3600) // 60
                end_seconds = seconds_in_day + TIME_BUCKET_SECONDS
                end_hour, end_minute = (end_seconds // 3600) % 24, (end_seconds % 3600) // 60
                f.write(f'Target interval: day {target_day}, {start_hour:02d}:{start_minute:02d}-'
                        f'{end_hour:02d}:{end_minute:02d}\n')
            f.write(f'Actual label: {actual_label}\n\nPredictions:\n')
            for row in sample_rows:
                f.write(f"{row['model']}: probability={row['probability_cold_start_next_bucket']:.4f}, "
                        f"predicted={row['predicted_label']}, correct={row['correct']}\n")
        print(f"\nSaved one held-out example input to: {input_path}")
        print(f"Saved model predictions for that example to: {prediction_path}")
        print(f"Saved example explanation to: {summary_path}")
    
    # Print comparison table
    metrics_only = {name: data['metrics'] for name, data in all_results.items()}
    table = create_comparison_table(metrics_only)
    print(f"\n{table}")
    
    # Save comparison table
    table_path = os.path.join(RESULTS_DIR, 'comparison_table.txt')
    with open(table_path, 'w') as f:
        f.write("Cold-Start Prediction - Model Comparison Results\n")
        f.write("Dataset: Huawei Cloud Region 5 (31 days)\n")
        f.write(f"Test split: Days 27-30\n\n")
        if smoke_test:
            f.write("SMOKE TEST ONLY: five training-selected functions; 2,048 deterministic samples per split; deep models trained for one epoch. These metrics verify pipeline execution and are not full-scale results.\n\n")
        f.write(table)
    print(f"  Saved comparison table to {table_path}")
    
    # Generate comparison visualizations
    print("\nGenerating comparison visualizations...")
    plot_roc_curves(all_results, y_test)
    plot_pr_curves(all_results, y_test)
    plot_model_comparison_bars(all_results)
    
    # Save detailed results as JSON
    json_results = {}
    for name, data in all_results.items():
        json_results[name] = {
            'accuracy': float(data['metrics']['accuracy']),
            'precision': float(data['metrics']['precision']),
            'recall': float(data['metrics']['recall']),
            'f1_score': float(data['metrics']['f1_score']),
            'mcc': float(data['metrics']['mcc']),
            'roc_auc': (None if data['metrics']['roc_auc'] is None else float(data['metrics']['roc_auc'])),
            'pr_auc': (None if data['metrics']['pr_auc'] is None else float(data['metrics']['pr_auc'])),
            'threshold': float(data.get('threshold', 0.5)),
            'training_time_seconds': float(data.get('time', 0)),
        }
    
    json_path = os.path.join(RESULTS_DIR, 'experiment_results.json')
    with open(json_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    print(f"  Saved detailed results to {json_path}")
    
    # Determine best model
    best_model = max(all_results.items(), key=lambda x: x[1]['metrics']['f1_score'])
    print(f"\n  Best model by F1-Score: {best_model[0]} "
          f"(F1={best_model[1]['metrics']['f1_score']:.4f})")
    
    return all_results


def main(smoke_test=False):
    """Execute the complete cold-start prediction pipeline."""
    print("=" * 70)
    print("COLD-START PREDICTION IN SERVERLESS COMPUTING")
    print("Adapted from Saravana Kumar & Selvakumara Samy (2025)")
    print("Dataset: Huawei Cloud Production FaaS Traces - Region 5")
    print(f"Device: {DEVICE}")
    print(f"Random Seed: {RANDOM_SEED}")
    print("=" * 70)
    
    total_start = time.time()
    np.random.seed(RANDOM_SEED)
    
    # Phase 1: Load Data
    df, summary = phase1_load_data()
    
    # Phase 2: EDA
    phase2_eda(df)
    
    # Phase 3: Feature Engineering
    data = phase3_feature_engineering(df, smoke_test=smoke_test)
    
    # Phase 4: Baseline Models
    baseline_results = phase4_train_baselines(data, smoke_test=smoke_test)
    
    # Phase 5: Deep Learning Models
    deep_results = phase5_train_deep_models(data, smoke_test=smoke_test)
    
    # Phase 6: Comparison
    y_test = data['y_test']
    all_results = phase6_comparison(baseline_results, deep_results, y_test,
                                    smoke_test=smoke_test, data=data)

    run_record = {
        'run_type': 'smoke_test' if smoke_test else 'full_training',
        'completed_utc': datetime.now(timezone.utc).isoformat(),
        'models': list(all_results),
        'split_sizes': {name: int(len(data[f'X_{name}'])) for name in ('train', 'val', 'test')},
        'split_positive_counts': {name: int(np.asarray(data[f'y_{name}']).sum()) for name in ('train', 'val', 'test')},
        'note': 'Smoke test uses a deterministic subset of real trace-derived samples and one deep-learning epoch.' if smoke_test else 'Full configured training run.',
        'full_scale_results': not smoke_test,
    }
    with open(os.path.join(RESULTS_DIR, 'run_record.json'), 'w') as f:
        json.dump(run_record, f, indent=2)
    
    total_time = time.time() - total_start
    print(f"\n{'='*70}")
    print(f"PIPELINE COMPLETE - Total time: {total_time/60:.1f} minutes")
    print(f"Results saved to: {RESULTS_DIR}")
    print(f"Visualizations saved to: {VISUALIZATION_DIR}")
    print(f"{'='*70}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Cold-start prediction training pipeline')
    parser.add_argument('--smoke-test', action='store_true',
                        help='Run all six models on a deterministic small subset of the real dataset')
    args = parser.parse_args()
    main(smoke_test=args.smoke_test)
