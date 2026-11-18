"""
Visualization module for cold-start prediction experiments.

Generates:
    - Training/validation curves
    - ROC curves
    - Precision-Recall curves
    - Confusion matrix heatmaps
    - Model comparison charts
    - Feature importance plots
    - Dataset analysis plots
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from config.config import VISUALIZATION_DIR
from src.evaluation.metrics import get_roc_curve_data, get_pr_curve_data

sns.set_theme(style='whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['figure.dpi'] = 100


def save_fig(fig, filename):
    """Save figure to visualization directory."""
    os.makedirs(VISUALIZATION_DIR, exist_ok=True)
    path = os.path.join(VISUALIZATION_DIR, filename)
    fig.savefig(path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


def plot_training_curves(history, model_name="ABiGRU"):
    """Plot training and validation loss/accuracy curves."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss curves
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    axes[0].plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title(f'{model_name} - Training & Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy curves
    axes[1].plot(epochs, history['train_acc'], 'b-', label='Train Accuracy', linewidth=2)
    axes[1].plot(epochs, history['val_acc'], 'r-', label='Val Accuracy', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title(f'{model_name} - Training & Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    fig.suptitle(f'{model_name} Training Curves', fontsize=14, fontweight='bold')
    fig.tight_layout()
    return save_fig(fig, f'training_curves_{model_name.lower().replace(" ", "_")}.png')


def plot_confusion_matrix(cm, model_name="Model"):
    """Plot confusion matrix as a heatmap."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['No recorded cold start', 'Recorded cold start'],
                yticklabels=['No recorded cold start', 'Recorded cold start'])
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('Actual', fontsize=12)
    ax.set_title(f'{model_name} - Confusion Matrix', fontsize=14, fontweight='bold')
    
    fig.tight_layout()
    return save_fig(fig, f'confusion_matrix_{model_name.lower().replace(" ", "_")}.png')


def plot_roc_curves(results_dict, y_true):
    """Plot ROC curves for all models."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = plt.cm.Set1(np.linspace(0, 1, len(results_dict)))
    
    for (name, data), color in zip(results_dict.items(), colors):
        if len(np.unique(y_true)) < 2:
            continue
        y_prob = data['y_prob']
        fpr, tpr, _ = get_roc_curve_data(y_true, y_prob)
        auc = data['metrics']['roc_auc']
        label = 'undefined' if auc is None else f'{auc:.4f}'
        ax.plot(fpr, tpr, color=color, linewidth=2, label=f'{name} (AUC={label})')

    if len(np.unique(y_true)) < 2:
        ax.text(0.5, 0.5, 'ROC-AUC undefined: test set has one class',
                ha='center', va='center', transform=ax.transAxes)
    
    if len(np.unique(y_true)) > 1:
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='Random')
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curves - Model Comparison', fontsize=14, fontweight='bold')
    if len(np.unique(y_true)) > 1:
        ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    
    fig.tight_layout()
    return save_fig(fig, 'roc_curves_comparison.png')


def plot_pr_curves(results_dict, y_true):
    """Plot Precision-Recall curves for all models."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = plt.cm.Set1(np.linspace(0, 1, len(results_dict)))
    
    for (name, data), color in zip(results_dict.items(), colors):
        if len(np.unique(y_true)) < 2:
            continue
        y_prob = data['y_prob']
        precision, recall, _ = get_pr_curve_data(y_true, y_prob)
        pr_auc = data['metrics']['pr_auc']
        label = 'undefined' if pr_auc is None else f'{pr_auc:.4f}'
        ax.plot(recall, precision, color=color, linewidth=2,
                label=f'{name} (PR-AUC={label})')
    
    # Baseline = positive class ratio
    baseline = y_true.mean()
    if len(np.unique(y_true)) > 1:
        ax.axhline(y=baseline, color='k', linestyle='--', alpha=0.5,
                   label=f'Baseline ({baseline:.3f})')
    else:
        ax.text(0.5, 0.5, 'PR-AUC undefined: test set has one class',
                ha='center', va='center', transform=ax.transAxes)
    
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Precision-Recall Curves - Model Comparison', fontsize=14, fontweight='bold')
    if len(np.unique(y_true)) > 1:
        ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    return save_fig(fig, 'pr_curves_comparison.png')


def plot_model_comparison_bars(results_dict):
    """Plot bar chart comparing all models across metrics."""
    metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1_score', 'mcc', 'roc_auc', 'pr_auc']
    metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'MCC', 'ROC-AUC', 'PR-AUC']
    
    model_names = list(results_dict.keys())
    n_models = len(model_names)
    n_metrics = len(metrics_to_plot)
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    x = np.arange(n_metrics)
    width = 0.8 / n_models
    
    colors = plt.cm.Set2(np.linspace(0, 1, n_models))
    
    for i, (name, color) in enumerate(zip(model_names, colors)):
        values = [np.nan if results_dict[name]['metrics'][m] is None else results_dict[name]['metrics'][m]
                  for m in metrics_to_plot]
        offset = (i - n_models / 2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=name, color=color, edgecolor='gray')
        
        # Add value labels on bars
        for bar, val in zip(bars, values):
            if np.isnan(val):
                continue
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=7, rotation=45)
    
    ax.set_xlabel('Metric', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Model Comparison - All Metrics', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=11)
    ax.legend(loc='lower left', fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.grid(True, alpha=0.3, axis='y')
    
    fig.tight_layout()
    return save_fig(fig, 'model_comparison_bars.png')


def plot_cold_start_distribution(df):
    """Plot cold start event distribution over time."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Events per day
    events_per_day = df.groupby('day').size()
    axes[0, 0].bar(events_per_day.index, events_per_day.values, color='steelblue')
    axes[0, 0].set_xlabel('Day')
    axes[0, 0].set_ylabel('Cold Start Events')
    axes[0, 0].set_title('Cold Start Events per Day')
    
    # Cold start cost distribution
    axes[0, 1].hist(df['totalCost_cold_start'].clip(upper=10), bins=50,
                     color='coral', edgecolor='gray', alpha=0.8)
    axes[0, 1].set_xlabel('Total Cold Start Cost (seconds, clipped at 10)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Cold Start Cost Distribution')
    
    # Top functions by cold start count
    top_funcs = df['funcName'].value_counts().head(20)
    axes[1, 0].barh(range(len(top_funcs)), top_funcs.values, color='mediumseagreen')
    axes[1, 0].set_yticks(range(len(top_funcs)))
    axes[1, 0].set_yticklabels([f'Func {f}' for f in top_funcs.index], fontsize=8)
    axes[1, 0].set_xlabel('Cold Start Count')
    axes[1, 0].set_title('Top 20 Functions by Cold Start Count')
    axes[1, 0].invert_yaxis()
    
    # Events by cluster
    cluster_counts = df['clusterName'].value_counts().sort_index()
    axes[1, 1].bar(cluster_counts.index.astype(str), cluster_counts.values, color='orchid')
    axes[1, 1].set_xlabel('Cluster')
    axes[1, 1].set_ylabel('Cold Start Events')
    axes[1, 1].set_title('Cold Start Events by Cluster')
    
    fig.suptitle('Dataset: Huawei Cloud Region 5 - Cold Start Analysis',
                 fontsize=14, fontweight='bold')
    fig.tight_layout()
    return save_fig(fig, 'dataset_analysis.png')


def plot_hourly_pattern(df):
    """Plot hourly cold start patterns."""
    fig, ax = plt.subplots(figsize=(12, 5))
    
    # Calculate hour of day from absolute time
    hour = ((df['abs_time'] % 86400) / 3600).astype(int)
    hourly_counts = hour.value_counts().sort_index()
    
    ax.bar(hourly_counts.index, hourly_counts.values, color='steelblue', edgecolor='gray')
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Cold Start Events')
    ax.set_title('Cold Start Events by Hour of Day (Region 5)', fontweight='bold')
    ax.set_xticks(range(24))
    ax.grid(True, alpha=0.3, axis='y')
    
    fig.tight_layout()
    return save_fig(fig, 'hourly_pattern.png')
