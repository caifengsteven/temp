"""
Embedding layers for Delphyne model including rotary positional embeddings
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class RotaryPositionalEmbedding(nn.Module):
    """
    Rotary Positional Embedding (RoPE) as used in the paper.
    Based on "RoFormer: Enhanced Transformer with Rotary Position Embedding"
    """
    
    def __init__(self, head_dim: int, max_position_embeddings: int = 16384, base: float = 10000.0):
        super().__init__()
        self.head_dim = head_dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        
        # Precompute the inverse frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        
        # Cache for cos and sin values
        self._cos_cached = None
        self._sin_cached = None
        self._seq_len_cached = 0
    
    def _update_cos_sin_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        """Update the cached cos and sin values if sequence length changed"""
        if seq_len > self._seq_len_cached or self._cos_cached is None:
            self._seq_len_cached = seq_len
            t = torch.arange(seq_len, device=device, dtype=dtype)
            freqs = torch.outer(t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            self._cos_cached = emb.cos()
            self._sin_cached = emb.sin()
    
    def forward(self, x: torch.Tensor, seq_len: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor of shape [batch_size, seq_len, num_heads, head_dim]
            seq_len: Sequence length (if None, inferred from x)
        
        Returns:
            Tuple of (cos, sin) tensors for rotary embedding
        """
        if seq_len is None:
            seq_len = x.shape[1]
        
        self._update_cos_sin_cache(seq_len, x.device, x.dtype)
        
        return (
            self._cos_cached[:seq_len].to(dtype=x.dtype),
            self._sin_cached[:seq_len].to(dtype=x.dtype)
        )


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary positional embedding to query and key tensors.
    
    Args:
        q: Query tensor [batch_size, seq_len, num_heads, head_dim]
        k: Key tensor [batch_size, seq_len, num_heads, head_dim]  
        cos: Cosine values [seq_len, head_dim]
        sin: Sine values [seq_len, head_dim]
    
    Returns:
        Tuple of rotated (q, k) tensors
    """
    # Expand cos and sin to match q and k dimensions
    cos = cos.unsqueeze(0).unsqueeze(2)  # [1, seq_len, 1, head_dim]
    sin = sin.unsqueeze(0).unsqueeze(2)  # [1, seq_len, 1, head_dim]
    
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    
    return q_embed, k_embed


class DelphyneEmbeddings(nn.Module):
    """
    Embedding layer for Delphyne that handles:
    1. Patch embeddings from time series data
    2. Variate ID embeddings
    3. Missing data and forecast masks
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.patch_size = config.patch_size
        self.hidden_size = config.hidden_size
        
        # Patch embedding - linear projection from patch_size to hidden_size
        self.patch_embedding = nn.Linear(config.patch_size, config.hidden_size)
        
        # Variate ID embeddings - to distinguish different time series variables
        # We'll use a reasonable default vocab size that can be adjusted
        self.max_variates = 1000  # Maximum number of different variates
        self.variate_embeddings = nn.Embedding(self.max_variates, config.hidden_size)
        
        # Special tokens for missing and forecast masks
        self.missing_token_embedding = nn.Parameter(torch.randn(config.hidden_size))
        self.forecast_token_embedding = nn.Parameter(torch.randn(config.hidden_size))
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.dropout_prob)
        
    def forward(
        self, 
        input_patches: torch.Tensor,
        variate_ids: torch.Tensor,
        missing_mask: Optional[torch.Tensor] = None,
        forecast_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            input_patches: Patched time series data [batch_size, seq_len, patch_size]
            variate_ids: Variate identifiers [batch_size, seq_len]
            missing_mask: Binary mask for missing data [batch_size, seq_len]
            forecast_mask: Binary mask for forecast positions [batch_size, seq_len]
        
        Returns:
            Embedded representations [batch_size, seq_len, hidden_size]
        """
        batch_size, seq_len = input_patches.shape[:2]
        
        # Get patch embeddings
        embeddings = self.patch_embedding(input_patches)  # [batch_size, seq_len, hidden_size]
        
        # Add variate embeddings
        variate_emb = self.variate_embeddings(variate_ids)  # [batch_size, seq_len, hidden_size]
        embeddings = embeddings + variate_emb
        
        # Handle missing data mask
        if missing_mask is not None:
            missing_positions = missing_mask.bool()
            embeddings[missing_positions] = self.missing_token_embedding
        
        # Handle forecast mask  
        if forecast_mask is not None:
            forecast_positions = forecast_mask.bool()
            embeddings[forecast_positions] = self.forecast_token_embedding
        
        # Apply layer norm and dropout
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        
        return embeddings
