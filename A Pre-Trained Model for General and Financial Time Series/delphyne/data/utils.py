"""
Data utilities for Delphyne model
"""

import torch
import numpy as np
from typing import Tuple, Optional, List, Union


def create_forecast_mask(
    batch_size: int,
    seq_len: int,
    forecast_length: int,
    device: Optional[torch.device] = None
) -> torch.Tensor:
    """
    Create forecast mask for the last `forecast_length` positions.
    
    Args:
        batch_size: Batch size
        seq_len: Sequence length
        forecast_length: Number of positions to forecast
        device: Device to create tensor on
        
    Returns:
        Binary mask [batch_size, seq_len] where 1 indicates forecast positions
    """
    mask = torch.zeros(batch_size, seq_len, device=device)
    mask[:, -forecast_length:] = 1.0
    return mask


def create_missing_mask(
    batch_size: int,
    seq_len: int,
    missing_prob: float,
    device: Optional[torch.device] = None,
    seed: Optional[int] = None
) -> torch.Tensor:
    """
    Create random missing data mask.
    
    Args:
        batch_size: Batch size
        seq_len: Sequence length
        missing_prob: Probability of missing values
        device: Device to create tensor on
        seed: Random seed for reproducibility
        
    Returns:
        Binary mask [batch_size, seq_len] where 1 indicates missing positions
    """
    if seed is not None:
        torch.manual_seed(seed)
    
    mask = torch.rand(batch_size, seq_len, device=device) < missing_prob
    return mask.float()


def create_beta_binomial_mask(
    batch_size: int,
    seq_len: int,
    alpha: float = 5.0,
    beta: float = 10.0,
    device: Optional[torch.device] = None,
    seed: Optional[int] = None
) -> torch.Tensor:
    """
    Create masking ratios using beta-binomial distribution as in the paper.
    
    Args:
        batch_size: Batch size
        seq_len: Sequence length
        alpha: Alpha parameter for beta distribution
        beta: Beta parameter for beta distribution
        device: Device to create tensor on
        seed: Random seed for reproducibility
        
    Returns:
        Binary mask [batch_size, seq_len] where 1 indicates masked positions
    """
    if seed is not None:
        np.random.seed(seed)
    
    masks = torch.zeros(batch_size, seq_len, device=device)
    
    for i in range(batch_size):
        # Sample masking ratio from beta distribution
        masking_ratio = np.random.beta(alpha, beta)
        
        # Create random mask with this ratio
        num_masked = int(masking_ratio * seq_len)
        indices = torch.randperm(seq_len)[:num_masked]
        masks[i, indices] = 1.0
    
    return masks


def create_variate_ids(
    batch_size: int,
    num_variates: int,
    seq_len: int,
    device: Optional[torch.device] = None
) -> torch.Tensor:
    """
    Create variate ID tensor for multivariate time series.
    
    Args:
        batch_size: Batch size
        num_variates: Number of variates
        seq_len: Sequence length per variate
        device: Device to create tensor on
        
    Returns:
        Variate IDs [batch_size, num_variates * seq_len]
    """
    # Create pattern: [0, 0, ..., 0, 1, 1, ..., 1, 2, 2, ..., 2, ...]
    variate_ids = torch.arange(num_variates, device=device)
    variate_ids = variate_ids.unsqueeze(1).repeat(1, seq_len)  # [num_variates, seq_len]
    variate_ids = variate_ids.flatten()  # [num_variates * seq_len]
    variate_ids = variate_ids.unsqueeze(0).repeat(batch_size, 1)  # [batch_size, num_variates * seq_len]
    
    return variate_ids


def normalize_time_series(
    time_series: torch.Tensor,
    method: str = "instance",
    eps: float = 1e-8
) -> Tuple[torch.Tensor, dict]:
    """
    Normalize time series data.
    
    Args:
        time_series: Input time series [batch_size, ...] 
        method: Normalization method ("instance", "batch", "global")
        eps: Small value for numerical stability
        
    Returns:
        Tuple of (normalized_data, normalization_stats)
    """
    if method == "instance":
        # Normalize each instance independently
        if time_series.dim() == 2:
            # [batch_size, seq_len]
            mean = time_series.mean(dim=-1, keepdim=True)
            std = time_series.std(dim=-1, keepdim=True) + eps
        elif time_series.dim() == 3:
            # [batch_size, num_variates, seq_len] - normalize each variate independently
            mean = time_series.mean(dim=-1, keepdim=True)
            std = time_series.std(dim=-1, keepdim=True) + eps
        else:
            raise ValueError(f"Unsupported tensor dimension: {time_series.dim()}")
    
    elif method == "batch":
        # Normalize across the batch
        mean = time_series.mean()
        std = time_series.std() + eps
    
    elif method == "global":
        # Normalize using global statistics (would need to be provided)
        raise NotImplementedError("Global normalization requires pre-computed statistics")
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    normalized = (time_series - mean) / std
    
    stats = {
        'mean': mean,
        'std': std,
        'method': method
    }
    
    return normalized, stats


def denormalize_time_series(
    normalized_data: torch.Tensor,
    stats: dict
) -> torch.Tensor:
    """
    Denormalize time series data using stored statistics.
    
    Args:
        normalized_data: Normalized time series data
        stats: Normalization statistics from normalize_time_series
        
    Returns:
        Denormalized data
    """
    mean = stats['mean']
    std = stats['std']
    
    return normalized_data * std + mean


def split_time_series(
    time_series: torch.Tensor,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Split time series data into train/val/test sets.
    
    Args:
        time_series: Input time series [batch_size, seq_len] or [batch_size, num_variates, seq_len]
        train_ratio: Ratio for training set
        val_ratio: Ratio for validation set  
        test_ratio: Ratio for test set
        
    Returns:
        Tuple of (train_data, val_data, test_data)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
    
    seq_len = time_series.shape[-1]
    
    train_end = int(seq_len * train_ratio)
    val_end = int(seq_len * (train_ratio + val_ratio))
    
    if time_series.dim() == 2:
        train_data = time_series[:, :train_end]
        val_data = time_series[:, train_end:val_end]
        test_data = time_series[:, val_end:]
    elif time_series.dim() == 3:
        train_data = time_series[:, :, :train_end]
        val_data = time_series[:, :, train_end:val_end]
        test_data = time_series[:, :, val_end:]
    else:
        raise ValueError(f"Unsupported tensor dimension: {time_series.dim()}")
    
    return train_data, val_data, test_data
