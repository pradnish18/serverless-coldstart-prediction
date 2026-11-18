"""
Training pipeline for deep learning cold-start prediction models.

Supports:
    - RMSProp optimizer (as per original paper)
    - Early stopping
    - Class-weighted loss for imbalanced data
    - Training/validation curve logging
"""

import os
import sys
import time
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from copy import deepcopy

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from config.config import (
    LEARNING_RATE, RMS_DECAY, BATCH_SIZE, NUM_EPOCHS,
    EARLY_STOPPING_PATIENCE, DEVICE, RANDOM_SEED
)


def set_seed(seed=RANDOM_SEED):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_dataloader(X, y, batch_size=BATCH_SIZE, shuffle=True):
    """Create a PyTorch DataLoader from numpy arrays."""
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.FloatTensor(y)
    dataset = TensorDataset(X_tensor, y_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      drop_last=False, num_workers=0)


def compute_class_weights(y_train):
    """Compute class weights for imbalanced binary classification."""
    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 1.0
    # Weight for positive class = n_neg / n_pos
    return n_neg / n_pos


def train_model(model, X_train, y_train, X_val, y_val, config=None):
    """
    Train a deep learning model with early stopping.
    
    Args:
        model: PyTorch nn.Module
        X_train, y_train: Training data (numpy arrays)
        X_val, y_val: Validation data (numpy arrays)
        config: Optional dict with training hyperparameters
    
    Returns:
        dict with:
            - 'model': Best model state dict
            - 'history': Training history (losses, metrics per epoch)
            - 'best_epoch': Epoch of best validation performance
    """
    if config is None:
        config = {}
    
    set_seed(config.get('seed', RANDOM_SEED))
    
    lr = config.get('lr', LEARNING_RATE)
    decay = config.get('decay', RMS_DECAY)
    batch_size = config.get('batch_size', BATCH_SIZE)
    epochs = config.get('epochs', NUM_EPOCHS)
    patience = config.get('patience', EARLY_STOPPING_PATIENCE)
    device_name = config.get('device', DEVICE)
    
    device = torch.device(device_name)
    print(f"  Training on device: {device}")
    
    model = model.to(device)
    
    # Class-weighted BCE loss
    # NumPy-derived scalars default to float64, which MPS does not support for
    # this loss parameter. Keep it explicitly aligned with model/input dtype.
    pos_weight = torch.tensor(
        [compute_class_weights(y_train)], dtype=torch.float32, device=device
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    # RMSProp optimizer (as per paper)
    optimizer = torch.optim.RMSprop(
        model.parameters(),
        lr=lr,
        alpha=decay,  # decay factor ρ
        eps=1e-8,
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )
    
    # Data loaders
    train_loader = create_dataloader(X_train, y_train, batch_size, shuffle=True)
    if len(X_train) == 0 or len(X_val) == 0:
        raise ValueError("Training and validation datasets must both be non-empty")
    val_loader = create_dataloader(X_val, y_val, batch_size, shuffle=False)
    
    # Training loop
    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': [],
    }
    
    best_val_loss = float('inf')
    best_model_state = None
    best_epoch = 0
    patience_counter = 0
    
    print(f"  Training for up to {epochs} epochs (patience={patience})...")
    start_time = time.time()
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            optimizer.zero_grad()
            logits = model(X_batch).squeeze(-1)
            loss = criterion(logits, y_batch)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item() * len(X_batch)
            preds = (torch.sigmoid(logits) > 0.5).float()
            train_correct += (preds == y_batch).sum().item()
            train_total += len(y_batch)
        
        train_loss /= train_total
        train_acc = train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                
                logits = model(X_batch).squeeze(-1)
                loss = criterion(logits, y_batch)
                
                val_loss += loss.item() * len(X_batch)
                preds = (torch.sigmoid(logits) > 0.5).float()
                val_correct += (preds == y_batch).sum().item()
                val_total += len(y_batch)
        
        val_loss /= val_total
        val_acc = val_correct / val_total
        
        # Update scheduler
        scheduler.step(val_loss)
        
        # Record history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = deepcopy(model.state_dict())
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Print progress every 5 epochs
        if (epoch + 1) % 5 == 0 or epoch == 0:
            elapsed = time.time() - start_time
            print(f"  Epoch {epoch+1:3d}/{epochs} | "
                  f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
                  f"Time: {elapsed:.0f}s")
        
        if patience_counter >= patience:
            print(f"  Early stopping at epoch {epoch+1} (best epoch: {best_epoch+1})")
            break
    
    total_time = time.time() - start_time
    print(f"  Training complete in {total_time:.1f}s. Best epoch: {best_epoch+1}")
    
    # Load best model
    model.load_state_dict(best_model_state)

    checkpoint_path = config.get('checkpoint_path')
    if checkpoint_path:
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        torch.save({
            'state_dict': model.state_dict(),
            'model_class': model.__class__.__name__,
            'input_dim': int(X_train.shape[-1]),
            'best_epoch': int(best_epoch),
            'history': history,
            'seed': int(config.get('seed', RANDOM_SEED)),
        }, checkpoint_path)
        with open(os.path.splitext(checkpoint_path)[0] + '_history.json', 'w') as f:
            json.dump(history, f, indent=2)
    
    return {
        'model': model,
        'history': history,
        'best_epoch': best_epoch,
        'training_time': total_time,
    }


def predict_deep_model(model, X, batch_size=BATCH_SIZE):
    """
    Get predictions from a trained deep model.
    
    Returns:
        probabilities: np.ndarray of shape (n_samples,)
    """
    device = torch.device('cpu')
    model = model.to(device)
    model.eval()
    
    if len(X) == 0:
        return np.empty((0,), dtype=np.float32)
    loader = create_dataloader(X, np.zeros(len(X)), batch_size, shuffle=False)
    
    all_probs = []
    with torch.no_grad():
        for X_batch, _ in loader:
            X_batch = X_batch.to(device)
            logits = model(X_batch).squeeze(-1)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
    
    return np.concatenate(all_probs)
