"""
Evaluation metrics for cold-start prediction.

Implements all metrics from the original paper:
    - Accuracy, Precision, Recall, F1-Score
    - MCC (Matthews Correlation Coefficient)
    - ROC-AUC, PR-AUC
    - Confusion Matrix
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, roc_auc_score, average_precision_score,
    confusion_matrix, classification_report, roc_curve,
    precision_recall_curve
)


def compute_all_metrics(y_true, y_prob, threshold=0.5):
    """
    Compute all evaluation metrics.
    
    Args:
        y_true: Ground truth binary labels
        y_prob: Predicted probabilities (0-1)
        threshold: Classification threshold
    
    Returns:
        dict with all metrics
    """
    y_pred = (y_prob >= threshold).astype(int)
    y_true = y_true.astype(int)
    
    # Basic metrics
    acc = accuracy_score(y_true, y_pred)
    
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if prec + rec else 0.0
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = ((tp * tn - fp * fn) / denominator) if denominator else 0.0
    
    # Ranking metrics require both positive and negative examples.
    if len(np.unique(y_true)) < 2:
        roc = pr = None
    else:
        try:
            roc = roc_auc_score(y_true, y_prob)
        except ValueError:
            roc = None
        try:
            pr = average_precision_score(y_true, y_prob)
        except ValueError:
            pr = None

    cm = np.array([
        [tn, fp],
        [fn, tp],
    ], dtype=int)
    
    return {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1_score': f1,
        'mcc': mcc,
        'roc_auc': roc,
        'pr_auc': pr,
        'confusion_matrix': cm,
        'threshold': threshold,
    }


def print_metrics(metrics, model_name="Model"):
    """Pretty-print evaluation metrics."""
    print(f"\n{'='*50}")
    print(f"  {model_name} - Evaluation Results")
    print(f"{'='*50}")
    print(f"  Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"  Precision: {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)")
    print(f"  Recall:    {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)")
    print(f"  F1-Score:  {metrics['f1_score']:.4f} ({metrics['f1_score']*100:.2f}%)")
    print(f"  MCC:       {metrics['mcc']:.4f} ({metrics['mcc']*100:.2f}%)")
    fmt = lambda value: "undefined" if value is None else f"{value:.4f}"
    print(f"  ROC-AUC:   {fmt(metrics['roc_auc'])}")
    print(f"  PR-AUC:    {fmt(metrics['pr_auc'])}")
    print(f"  Threshold: {metrics['threshold']:.2f}")
    if 'confusion_matrix' in metrics:
        cm = metrics['confusion_matrix']
        print(f"\n  Confusion Matrix:")
        print(f"               Pred=0    Pred=1")
        print(f"  Actual=0  {cm[0,0]:8d}  {cm[0,1]:8d}")
        print(f"  Actual=1  {cm[1,0]:8d}  {cm[1,1]:8d}")
    print(f"{'='*50}")


def get_roc_curve_data(y_true, y_prob):
    """Compute ROC curve data."""
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    return fpr, tpr, thresholds


def get_pr_curve_data(y_true, y_prob):
    """Compute Precision-Recall curve data."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    return precision, recall, thresholds


def find_optimal_threshold(y_true, y_prob, metric='f1'):
    """
    Find the optimal classification threshold using validation data.
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        metric: Metric to optimize ('f1', 'mcc', 'accuracy')
    
    Returns:
        Optimal threshold value
    """
    thresholds = np.arange(0.1, 0.9, 0.05)
    best_score = -1
    best_thresh = 0.5
    
    for thresh in thresholds:
        y_pred = (y_prob >= thresh).astype(int)
        if metric == 'f1':
            score = f1_score(y_true, y_pred, zero_division=0)
        elif metric == 'mcc':
            score = matthews_corrcoef(y_true, y_pred)
        elif metric == 'accuracy':
            score = accuracy_score(y_true, y_pred)
        else:
            score = f1_score(y_true, y_pred, zero_division=0)
        
        if score > best_score:
            best_score = score
            best_thresh = thresh
    
    return best_thresh


def create_comparison_table(results_dict):
    """
    Create a comparison table from multiple model results.
    
    Args:
        results_dict: Dict of {model_name: metrics_dict}
    
    Returns:
        Formatted string table
    """
    header = f"{'Model':<25} {'Acc':>8} {'Prec':>8} {'Rec':>8} {'F1':>8} {'MCC':>8} {'ROC-AUC':>8} {'PR-AUC':>8}"
    separator = "-" * len(header)
    
    lines = [separator, header, separator]
    for name, metrics in results_dict.items():
        fmt = lambda value: "undefined" if value is None else f"{value:>7.4f}"
        line = (f"{name:<25} "
                f"{metrics['accuracy']:>7.4f} "
                f"{metrics['precision']:>7.4f} "
                f"{metrics['recall']:>7.4f} "
                f"{metrics['f1_score']:>7.4f} "
                f"{metrics['mcc']:>7.4f} "
                f"{fmt(metrics['roc_auc'])} "
                f"{fmt(metrics['pr_auc'])}")
        lines.append(line)
    lines.append(separator)
    
    return "\n".join(lines)
