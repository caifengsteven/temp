"""
Output distribution layers for Delphyne model using mixture of Student-T distributions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional, Dict, Any
from torch.distributions import StudentT, Categorical, MixtureSameFamily


class StudentTMixtureOutput(nn.Module):
    """
    Mixture of Student-T distributions for probabilistic forecasting.
    
    As specified in the paper, this is used instead of a single distribution
    or more complex mixtures to handle heavy-tailed financial data while
    maintaining simplicity (Occam's Razor).
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_components = config.num_mixture_components
        
        # Output projections for mixture parameters
        # Each component needs: location (mu), scale (sigma), degrees of freedom (nu), and mixture weight
        self.location_proj = nn.Linear(self.hidden_size, self.num_components)
        self.scale_proj = nn.Linear(self.hidden_size, self.num_components)
        self.df_proj = nn.Linear(self.hidden_size, self.num_components)
        self.weight_proj = nn.Linear(self.hidden_size, self.num_components)
        
        # Minimum values to ensure numerical stability
        self.min_scale = 1e-4
        self.min_df = 2.1  # Degrees of freedom must be > 2 for finite variance
        
    def forward(self, hidden_states: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute mixture parameters from hidden states.
        
        Args:
            hidden_states: Transformer output [batch_size, seq_len, hidden_size]
            
        Returns:
            Dictionary containing mixture parameters and distribution
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # Project to mixture parameters
        locations = self.location_proj(hidden_states)  # [batch_size, seq_len, num_components]
        scales = self.scale_proj(hidden_states)        # [batch_size, seq_len, num_components]
        dfs = self.df_proj(hidden_states)              # [batch_size, seq_len, num_components]
        logits = self.weight_proj(hidden_states)       # [batch_size, seq_len, num_components]
        
        # Apply constraints to ensure valid parameters
        scales = F.softplus(scales) + self.min_scale   # Ensure positive scale
        dfs = F.softplus(dfs) + self.min_df           # Ensure df > 2
        weights = F.softmax(logits, dim=-1)           # Ensure weights sum to 1
        
        # Create mixture distribution
        # Reshape for distribution creation
        locations = locations.view(-1, self.num_components)
        scales = scales.view(-1, self.num_components)
        dfs = dfs.view(-1, self.num_components)
        weights = weights.view(-1, self.num_components)
        
        # Create component distributions
        component_dist = StudentT(df=dfs, loc=locations, scale=scales)
        
        # Create mixture weights distribution
        mixture_dist = Categorical(probs=weights)
        
        # Create mixture distribution
        mixture = MixtureSameFamily(mixture_dist, component_dist)
        
        return {
            'distribution': mixture,
            'locations': locations.view(batch_size, seq_len, self.num_components),
            'scales': scales.view(batch_size, seq_len, self.num_components),
            'dfs': dfs.view(batch_size, seq_len, self.num_components),
            'weights': weights.view(batch_size, seq_len, self.num_components)
        }
    
    def compute_loss(self, hidden_states: torch.Tensor, targets: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute negative log-likelihood loss.

        Args:
            hidden_states: Transformer output [batch_size, num_patches, hidden_size]
            targets: Target values [batch_size, original_seq_len]
            mask: Optional mask for valid positions [batch_size, num_patches]

        Returns:
            Negative log-likelihood loss
        """
        output = self(hidden_states)
        distribution = output['distribution']

        batch_size, num_patches = hidden_states.shape[:2]

        # For now, we'll compute loss on a subset of targets that matches the patch structure
        # In a full implementation, you'd want to properly align targets with patches
        if targets.numel() == batch_size * num_patches:
            # Targets already match patch structure
            target_values = targets.view(-1)
        else:
            # Subsample targets to match patch structure
            original_seq_len = targets.shape[1]
            step_size = max(1, original_seq_len // num_patches)
            indices = torch.arange(0, original_seq_len, step_size, device=targets.device)[:num_patches]
            target_values = targets[:, indices].contiguous().view(-1)

        # Compute log probabilities
        log_probs = distribution.log_prob(target_values)  # [batch_size * num_patches]

        # Reshape back
        log_probs = log_probs.view(batch_size, -1)  # [batch_size, num_patches]

        # Apply mask if provided
        if mask is not None:
            log_probs = log_probs * mask
            # Compute mean over valid positions
            valid_positions = mask.sum()
            if valid_positions > 0:
                loss = -log_probs.sum() / valid_positions
            else:
                loss = torch.tensor(0.0, device=hidden_states.device, requires_grad=True)
        else:
            loss = -log_probs.mean()

        return loss
    
    def sample(self, hidden_states: torch.Tensor, num_samples: int = 1) -> torch.Tensor:
        """
        Sample from the mixture distribution.
        
        Args:
            hidden_states: Transformer output [batch_size, seq_len, hidden_size]
            num_samples: Number of samples to generate
            
        Returns:
            Samples [num_samples, batch_size, seq_len]
        """
        output = self(hidden_states)
        distribution = output['distribution']
        
        # Generate samples
        samples = distribution.sample((num_samples,))  # [num_samples, batch_size * seq_len]
        
        # Reshape to [num_samples, batch_size, seq_len]
        batch_size, seq_len = hidden_states.shape[:2]
        samples = samples.view(num_samples, batch_size, seq_len)
        
        return samples
    
    def quantile(self, hidden_states: torch.Tensor, quantiles: torch.Tensor) -> torch.Tensor:
        """
        Compute quantiles of the mixture distribution.
        
        Args:
            hidden_states: Transformer output [batch_size, seq_len, hidden_size]
            quantiles: Quantile levels [num_quantiles]
            
        Returns:
            Quantile values [batch_size, seq_len, num_quantiles]
        """
        output = self(hidden_states)
        distribution = output['distribution']
        
        # For mixture distributions, we need to approximate quantiles
        # We'll use sampling-based approximation
        num_samples = 10000
        samples = distribution.sample((num_samples,))  # [num_samples, batch_size * seq_len]
        
        # Compute quantiles
        quantile_values = torch.quantile(samples, quantiles, dim=0)  # [num_quantiles, batch_size * seq_len]
        
        # Reshape
        batch_size, seq_len = hidden_states.shape[:2]
        quantile_values = quantile_values.view(len(quantiles), batch_size, seq_len)
        quantile_values = quantile_values.permute(1, 2, 0)  # [batch_size, seq_len, num_quantiles]
        
        return quantile_values


class ForecastHead(nn.Module):
    """
    Forecast head that combines the transformer output with the mixture distribution.
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Output distribution
        self.output_dist = StudentTMixtureOutput(config)
        
        # Optional additional processing layers
        self.pre_output_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.dropout_prob)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        forecast_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """
        Forward pass through the forecast head.

        Args:
            hidden_states: Transformer output [batch_size, num_patches, hidden_size]
            targets: Target values for loss computation [batch_size, original_seq_len]
            forecast_mask: Mask indicating forecast positions [batch_size, num_patches]

        Returns:
            Dictionary containing outputs and optionally loss
        """
        # Apply normalization and dropout
        hidden_states = self.pre_output_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        # Get distribution parameters
        output = self.output_dist(hidden_states)

        result = {
            'distribution': output['distribution'],
            'mixture_params': {
                'locations': output['locations'],
                'scales': output['scales'],
                'dfs': output['dfs'],
                'weights': output['weights']
            }
        }

        # Compute loss if targets are provided
        if targets is not None:
            loss = self.output_dist.compute_loss(hidden_states, targets, forecast_mask)
            result['loss'] = loss

        return result
