"""
Patching and preprocessing utilities for Delphyne model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, List, Dict, Any


class TimeSeriesPatcher(nn.Module):
    """
    Converts multivariate time series data into patches for transformer processing.
    
    Handles:
    1. Flattening multivariate data
    2. Patching into fixed-size chunks
    3. Right-padding for variable lengths
    4. Missing data and forecast masking
    """
    
    def __init__(self, patch_size: int = 32):
        super().__init__()
        self.patch_size = patch_size
        
    def forward(
        self, 
        time_series: torch.Tensor,
        variate_ids: Optional[torch.Tensor] = None,
        missing_mask: Optional[torch.Tensor] = None,
        forecast_mask: Optional[torch.Tensor] = None,
        max_length: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Convert multivariate time series to patches.
        
        Args:
            time_series: Input time series [batch_size, num_variates, seq_len] or [batch_size, seq_len]
            variate_ids: Variate identifiers for each time step
            missing_mask: Binary mask for missing values
            forecast_mask: Binary mask for forecast positions
            max_length: Maximum sequence length for padding
            
        Returns:
            Dictionary containing:
            - patches: Patched data [batch_size, num_patches, patch_size]
            - variate_ids: Variate IDs for each patch [batch_size, num_patches]
            - missing_mask: Missing mask for each patch [batch_size, num_patches]
            - forecast_mask: Forecast mask for each patch [batch_size, num_patches]
            - attention_mask: Valid positions mask [batch_size, num_patches]
        """
        batch_size = time_series.shape[0]
        
        # Handle different input shapes
        if time_series.dim() == 3:
            # Multivariate: [batch_size, num_variates, seq_len]
            batch_size, num_variates, seq_len = time_series.shape
            # Flatten to [batch_size, num_variates * seq_len]
            flattened = time_series.transpose(1, 2).contiguous().view(batch_size, -1)
            total_len = num_variates * seq_len
            
            # Create variate IDs if not provided
            if variate_ids is None:
                variate_ids = torch.arange(num_variates, device=time_series.device)
                variate_ids = variate_ids.unsqueeze(0).repeat(seq_len, 1).transpose(0, 1).flatten()
                variate_ids = variate_ids.unsqueeze(0).repeat(batch_size, 1)
                
        elif time_series.dim() == 2:
            # Univariate: [batch_size, seq_len]
            batch_size, seq_len = time_series.shape
            flattened = time_series
            total_len = seq_len
            
            # Create variate IDs (all zeros for univariate)
            if variate_ids is None:
                variate_ids = torch.zeros(batch_size, seq_len, dtype=torch.long, device=time_series.device)
        else:
            raise ValueError(f"Unsupported time_series shape: {time_series.shape}")
        
        # Apply right-padding if max_length is specified
        if max_length is not None and total_len < max_length:
            pad_length = max_length - total_len
            flattened = F.pad(flattened, (0, pad_length), value=0.0)
            variate_ids = F.pad(variate_ids, (0, pad_length), value=0)
            
            if missing_mask is not None:
                missing_mask = F.pad(missing_mask, (0, pad_length), value=1)  # Padded positions are "missing"
            if forecast_mask is not None:
                forecast_mask = F.pad(forecast_mask, (0, pad_length), value=0)
                
            total_len = max_length
        
        # Create patches
        num_patches = total_len // self.patch_size
        if total_len % self.patch_size != 0:
            # Pad to make divisible by patch_size
            pad_length = self.patch_size - (total_len % self.patch_size)
            flattened = F.pad(flattened, (0, pad_length), value=0.0)
            variate_ids = F.pad(variate_ids, (0, pad_length), value=0)
            
            if missing_mask is not None:
                missing_mask = F.pad(missing_mask, (0, pad_length), value=1)
            if forecast_mask is not None:
                forecast_mask = F.pad(forecast_mask, (0, pad_length), value=0)
                
            num_patches += 1
        
        # Reshape into patches
        patches = flattened.view(batch_size, num_patches, self.patch_size)
        
        # Create patch-level variate IDs (use the first variate ID in each patch)
        patch_variate_ids = variate_ids.view(batch_size, num_patches, self.patch_size)[:, :, 0]
        
        # Create patch-level masks
        patch_missing_mask = None
        patch_forecast_mask = None
        
        if missing_mask is not None:
            patch_missing_mask = missing_mask.view(batch_size, num_patches, self.patch_size)
            # A patch is considered missing if any element is missing
            patch_missing_mask = patch_missing_mask.any(dim=-1).float()
        
        if forecast_mask is not None:
            patch_forecast_mask = forecast_mask.view(batch_size, num_patches, self.patch_size)
            # A patch is considered forecast if any element is forecast
            patch_forecast_mask = patch_forecast_mask.any(dim=-1).float()
        
        # Create attention mask (all patches are valid by default)
        attention_mask = torch.ones(batch_size, num_patches, device=time_series.device)
        
        return {
            'patches': patches,
            'variate_ids': patch_variate_ids,
            'missing_mask': patch_missing_mask,
            'forecast_mask': patch_forecast_mask,
            'attention_mask': attention_mask
        }


class TimeSeriesNormalizer(nn.Module):
    """
    Normalizes time series data per variate as specified in the paper.
    Uses instance normalization independently for each variate.
    """
    
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        
    def forward(self, time_series: torch.Tensor, variate_ids: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Normalize time series data.
        
        Args:
            time_series: Input time series [batch_size, seq_len] or [batch_size, num_variates, seq_len]
            variate_ids: Variate identifiers
            
        Returns:
            Tuple of (normalized_data, normalization_stats)
        """
        if time_series.dim() == 3:
            # Multivariate case: normalize each variate independently
            batch_size, num_variates, seq_len = time_series.shape
            normalized = torch.zeros_like(time_series)
            means = torch.zeros(batch_size, num_variates, device=time_series.device)
            stds = torch.zeros(batch_size, num_variates, device=time_series.device)
            
            for v in range(num_variates):
                variate_data = time_series[:, v, :]  # [batch_size, seq_len]
                mean = variate_data.mean(dim=-1, keepdim=True)  # [batch_size, 1]
                std = variate_data.std(dim=-1, keepdim=True) + self.eps  # [batch_size, 1]
                
                normalized[:, v, :] = (variate_data - mean) / std
                means[:, v] = mean.squeeze(-1)
                stds[:, v] = std.squeeze(-1)
                
        else:
            # Univariate case
            mean = time_series.mean(dim=-1, keepdim=True)
            std = time_series.std(dim=-1, keepdim=True) + self.eps
            normalized = (time_series - mean) / std
            means = mean.squeeze(-1)
            stds = std.squeeze(-1)
        
        stats = {
            'mean': means,
            'std': stds
        }
        
        return normalized, stats
    
    def denormalize(self, normalized_data: torch.Tensor, stats: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Denormalize the data using stored statistics."""
        mean = stats['mean']
        std = stats['std']
        
        if normalized_data.dim() == 3 and mean.dim() == 2:
            # Multivariate case
            mean = mean.unsqueeze(-1)  # [batch_size, num_variates, 1]
            std = std.unsqueeze(-1)    # [batch_size, num_variates, 1]
        elif normalized_data.dim() == 2 and mean.dim() == 1:
            # Univariate case
            mean = mean.unsqueeze(-1)  # [batch_size, 1]
            std = std.unsqueeze(-1)    # [batch_size, 1]
        
        return normalized_data * std + mean
