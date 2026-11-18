"""
Baseline ML models for cold-start prediction comparison.

Implements:
    - Logistic Regression
    - Random Forest
    - XGBoost

These operate on flattened feature vectors (not sequential),
serving as non-temporal baselines.
"""

import os
import joblib
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False


def flatten_sequences(X):
    """
    Flatten sequential input (samples, seq_len, features) to (samples, seq_len*features)
    for use with non-sequential models.
    """
    if X.ndim == 3:
        return X.reshape(X.shape[0], -1)
    return X


def train_logistic_regression(X_train, y_train, seed=42):
    """Train a Logistic Regression baseline."""
    X_flat = flatten_sequences(X_train)
    model = LogisticRegression(
        max_iter=1000,
        C=1.0,
        class_weight='balanced',
        random_state=seed,
        solver='lbfgs',
    )
    model.fit(X_flat, y_train)
    return model


def train_random_forest(X_train, y_train, seed=42, n_estimators=200):
    """Train a Random Forest baseline."""
    X_flat = flatten_sequences(X_train)
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=5,
        class_weight='balanced',
        random_state=seed,
        n_jobs=1,
    )
    model.fit(X_flat, y_train)
    return model


def train_xgboost(X_train, y_train, X_val=None, y_val=None, seed=42, n_estimators=300):
    """Train an XGBoost baseline."""
    if not HAS_XGBOOST:
        raise ImportError("XGBoost is not installed. Install with: pip install xgboost")
    
    X_flat = flatten_sequences(X_train)
    
    # Calculate scale_pos_weight for imbalanced data
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    scale_pos_weight = n_neg / max(n_pos, 1)
    
    model = xgb.XGBClassifier(
        n_estimators=n_estimators,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=seed,
        eval_metric='logloss',
        early_stopping_rounds=15,
        use_label_encoder=False,
        tree_method='hist',
        n_jobs=1,
    )
    
    if X_val is not None and y_val is not None:
        X_val_flat = flatten_sequences(X_val)
        model.fit(
            X_flat, y_train,
            eval_set=[(X_val_flat, y_val)],
            verbose=False,
        )
    else:
        model.set_params(early_stopping_rounds=None)
        model.fit(X_flat, y_train, verbose=False)
    
    return model


def predict_baseline(model, X, return_proba=True):
    """
    Get predictions from a baseline model.
    
    Args:
        model: Trained sklearn/xgboost model
        X: Input data (may be 3D sequences)
        return_proba: If True, return probabilities; else return class labels
    
    Returns:
        Predictions array
    """
    X_flat = flatten_sequences(X)
    if return_proba and hasattr(model, 'predict_proba'):
        return model.predict_proba(X_flat)[:, 1]
    elif return_proba and hasattr(model, 'decision_function'):
        return model.decision_function(X_flat)
    else:
        return model.predict(X_flat)


def save_baseline(model, path):
    """Persist a fitted sklearn/XGBoost estimator."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    return path


def load_baseline(path):
    """Load an estimator saved with :func:`save_baseline`."""
    return joblib.load(path)


BASELINE_MODELS = {
    'logistic_regression': train_logistic_regression,
    'random_forest': train_random_forest,
    'xgboost': train_xgboost,
}
