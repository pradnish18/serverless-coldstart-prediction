"""
Deep Learning Models for Cold-Start Prediction.

Implements:
1. Standard GRU
2. Bidirectional GRU  
3. ABiGRU (Attention-Based Bidirectional GRU) - adapted from paper

Reference: Saravana Kumar & Selvakumara Samy (2025)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Attention(nn.Module):
    """
    Additive (Bahdanau-style) attention mechanism.
    
    Given hidden states H of shape (batch, seq_len, hidden_dim),
    computes attention weights and returns a context vector.
    
    As described in the paper:
        e_t = score(h_t) = v^T tanh(W_a h_t + b_a)
        α_t = softmax(e_t)
        c = Σ α_t * h_t
    """
    def __init__(self, hidden_dim, attention_dim):
        super().__init__()
        self.W = nn.Linear(hidden_dim, attention_dim, bias=True)
        self.v = nn.Linear(attention_dim, 1, bias=False)
    
    def forward(self, hidden_states):
        """
        Args:
            hidden_states: (batch, seq_len, hidden_dim)
        Returns:
            context: (batch, hidden_dim) - attention-weighted sum
            weights: (batch, seq_len) - attention weights
        """
        # (batch, seq_len, attention_dim)
        energy = torch.tanh(self.W(hidden_states))
        # (batch, seq_len, 1) -> (batch, seq_len)
        scores = self.v(energy).squeeze(-1)
        weights = F.softmax(scores, dim=1)
        # (batch, 1, seq_len) x (batch, seq_len, hidden_dim) -> (batch, 1, hidden_dim)
        context = torch.bmm(weights.unsqueeze(1), hidden_states).squeeze(1)
        return context, weights


class StandardGRU(nn.Module):
    """Standard (unidirectional) GRU for cold-start prediction."""
    
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout=0.3):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, input_dim)
        Returns:
            logits: (batch, 1)
        """
        _, hidden = self.gru(x)
        last_hidden = hidden[-1]  # final state of the last forward layer
        last_hidden = self.dropout(last_hidden)
        logits = self.fc(last_hidden)
        return logits


class BiGRU(nn.Module):
    """Bidirectional GRU for cold-start prediction."""
    
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout=0.3):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, 1)  # *2 for bidirectional
    
    def forward(self, x):
        _, hidden = self.gru(x)
        # h_n: (num_layers * 2, batch, hidden); combine both directions
        last_hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        last_hidden = self.dropout(last_hidden)
        logits = self.fc(last_hidden)
        return logits


class ABiGRU(nn.Module):
    """
    Attention-Based Bidirectional GRU (ABiGRU).
    
    Architecture (from paper):
        Input -> BiGRU -> Attention -> MaxPooling -> FC -> Sigmoid
    
    Adapted to work with the Huawei Cloud dataset features.
    The bidirectional processing operates on the historical observation window
    (no future information leakage since we only see past buckets).
    """
    
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, 
                 attention_dim=32, dropout=0.3, fc_dim=64, pool_size=2):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        # Bidirectional GRU (Section 6.2 of paper)
        self.bigru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        
        # Attention mechanism (Section 6.3 of paper)
        self.attention = Attention(hidden_dim * 2, attention_dim)
        
        # Max Pooling (Section 6.4 of paper)
        # Applied over the temporal dimension after attention weighting
        self.pool_size = pool_size
        
        # Fully connected layer (Section 6.5)
        self.dropout = nn.Dropout(dropout)
        # After attention, we get (batch, hidden_dim*2) 
        # We also use pooled features, concatenated
        self.fc1 = nn.Linear(hidden_dim * 2 * 2, fc_dim)  # attention + pooled
        self.fc2 = nn.Linear(fc_dim, 1)  # Sigmoid output (Section 6.6)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier initialization."""
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, input_dim) - sequence of feature vectors
        Returns:
            logits: (batch, 1) - cold start probability (before sigmoid)
        """
        batch_size = x.size(0)
        
        # BiGRU processing
        # output: (batch, seq_len, hidden_dim*2)
        gru_output, _ = self.bigru(x)
        
        # Attention mechanism
        # context: (batch, hidden_dim*2), weights: (batch, seq_len)
        context, attn_weights = self.attention(gru_output)
        
        # Max pooling over temporal dimension
        # (batch, seq_len, hidden_dim*2) -> pool over seq_len
        # Transpose to (batch, hidden_dim*2, seq_len) for 1D max pooling
        gru_t = gru_output.transpose(1, 2)
        pooled = F.adaptive_max_pool1d(gru_t, 1).squeeze(-1)  # (batch, hidden_dim*2)
        
        # Concatenate attention context and pooled features
        combined = torch.cat([context, pooled], dim=1)  # (batch, hidden_dim*4)
        
        # Fully connected layers
        combined = self.dropout(combined)
        hidden = F.relu(self.fc1(combined))
        hidden = self.dropout(hidden)
        logits = self.fc2(hidden)
        
        return logits
    
    def get_attention_weights(self, x):
        """Return attention weights for interpretability."""
        with torch.no_grad():
            gru_output, _ = self.bigru(x)
            _, attn_weights = self.attention(gru_output)
        return attn_weights


def create_model(model_name, input_dim, config=None):
    """
    Factory function to create models.
    
    Args:
        model_name: One of 'gru', 'bigru', 'abigru'
        input_dim: Number of input features
        config: Optional dict overriding default hyperparameters
    """
    if config is None:
        config = {}
    
    hidden_dim = config.get('hidden_dim', 64)
    num_layers = config.get('num_layers', 2)
    dropout = config.get('dropout', 0.3)
    attention_dim = config.get('attention_dim', 32)
    fc_dim = config.get('fc_dim', 64)
    
    if model_name == 'gru':
        return StandardGRU(input_dim, hidden_dim, num_layers, dropout)
    elif model_name == 'bigru':
        return BiGRU(input_dim, hidden_dim, num_layers, dropout)
    elif model_name == 'abigru':
        return ABiGRU(input_dim, hidden_dim, num_layers, attention_dim, dropout, fc_dim)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def load_deep_model(model_name, input_dim, checkpoint_path, config=None, map_location='cpu'):
    """Reconstruct a model and restore a checkpoint produced by ``train_model``."""
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    model = create_model(model_name, input_dim, config)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return model
