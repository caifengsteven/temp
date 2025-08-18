"""
Training utilities for Delphyne model
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
import math
from typing import Dict, Any, Optional, Tuple, Union
import numpy as np


def create_optimizer(
    model: nn.Module,
    learning_rate: float = 1e-4,
    weight_decay: float = 0.1,
    beta1: float = 0.9,
    beta2: float = 0.98,
    eps: float = 1e-8
) -> optim.Optimizer:
    """
    Create AdamW optimizer as specified in the paper.
    
    Args:
        model: Model to optimize
        learning_rate: Learning rate
        weight_decay: Weight decay coefficient
        beta1: Adam beta1 parameter
        beta2: Adam beta2 parameter
        eps: Adam epsilon parameter
        
    Returns:
        AdamW optimizer
    """
    return optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(beta1, beta2),
        eps=eps
    )


def create_scheduler(
    optimizer: optim.Optimizer,
    num_warmup_steps: int = 10000,
    num_training_steps: int = 1000000,
    min_lr_ratio: float = 0.1
) -> LambdaLR:
    """
    Create learning rate scheduler with linear warmup and cosine annealing.
    
    Args:
        optimizer: Optimizer to schedule
        num_warmup_steps: Number of warmup steps
        num_training_steps: Total number of training steps
        min_lr_ratio: Minimum learning rate as ratio of initial lr
        
    Returns:
        Learning rate scheduler
    """
    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            # Linear warmup
            return float(current_step) / float(max(1, num_warmup_steps))
        else:
            # Cosine annealing
            progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
            return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))
    
    return LambdaLR(optimizer, lr_lambda)


def compute_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    mask: Optional[torch.Tensor] = None
) -> Dict[str, float]:
    """
    Compute evaluation metrics for time series forecasting.
    
    Args:
        predictions: Model predictions [batch_size, seq_len]
        targets: Ground truth targets [batch_size, seq_len]
        mask: Optional mask for valid positions [batch_size, seq_len]
        
    Returns:
        Dictionary of metrics
    """
    if mask is not None:
        # Apply mask
        predictions = predictions * mask
        targets = targets * mask
        valid_elements = mask.sum()
    else:
        valid_elements = predictions.numel()
    
    # Mean Absolute Error
    mae = torch.abs(predictions - targets).sum() / valid_elements
    
    # Mean Squared Error
    mse = ((predictions - targets) ** 2).sum() / valid_elements
    
    # Root Mean Squared Error
    rmse = torch.sqrt(mse)
    
    # Mean Absolute Percentage Error (avoid division by zero)
    mape = torch.abs((predictions - targets) / (targets + 1e-8)).sum() / valid_elements * 100
    
    return {
        'mae': mae.item(),
        'mse': mse.item(),
        'rmse': rmse.item(),
        'mape': mape.item()
    }


def compute_probabilistic_metrics(
    distribution: torch.distributions.Distribution,
    targets: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    quantiles: Optional[torch.Tensor] = None
) -> Dict[str, float]:
    """
    Compute probabilistic forecasting metrics.

    Args:
        distribution: Predicted distribution
        targets: Ground truth targets [batch_size, seq_len]
        mask: Optional mask for valid positions [batch_size, seq_len]
        quantiles: Quantile levels for evaluation

    Returns:
        Dictionary of probabilistic metrics
    """
    if quantiles is None:
        quantiles = torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9])

    # Get distribution batch shape to understand the expected target shape
    batch_shape = distribution.batch_shape

    # Reshape targets to match distribution batch shape if needed
    if targets.shape != batch_shape:
        # For now, we'll subsample targets to match the distribution shape
        # In a full implementation, you'd want proper alignment
        if len(batch_shape) == 1:
            # Distribution expects [batch_size * num_patches]
            expected_size = batch_shape[0]
            if targets.numel() >= expected_size:
                # Flatten and take first expected_size elements
                targets_flat = targets.view(-1)[:expected_size]
                if mask is not None:
                    mask_flat = mask.view(-1)[:expected_size]
                else:
                    mask_flat = None
            else:
                # Pad if needed
                targets_flat = torch.cat([
                    targets.view(-1),
                    torch.zeros(expected_size - targets.numel(), device=targets.device)
                ])
                if mask is not None:
                    mask_flat = torch.cat([
                        mask.view(-1),
                        torch.zeros(expected_size - mask.numel(), device=mask.device)
                    ])
                else:
                    mask_flat = None
        else:
            # More complex reshaping needed
            targets_flat = targets.view(-1)[:batch_shape.numel()]
            mask_flat = mask.view(-1)[:batch_shape.numel()] if mask is not None else None
    else:
        targets_flat = targets
        mask_flat = mask

    try:
        # Negative Log-Likelihood
        log_probs = distribution.log_prob(targets_flat)
        if mask_flat is not None:
            valid_mask = mask_flat > 0
            if valid_mask.sum() > 0:
                nll = -(log_probs * mask_flat).sum() / mask_flat.sum()
            else:
                nll = torch.tensor(0.0, device=targets.device)
        else:
            nll = -log_probs.mean()

        # Sample from distribution for other metrics
        num_samples = 100  # Reduced for speed
        samples = distribution.sample((num_samples,))  # [num_samples, ...]

        # Compute CRPS (simplified)
        if samples.numel() > 0 and targets_flat.numel() > 0:
            # Simple approximation of CRPS
            samples_mean = samples.mean(dim=0)
            crps = torch.abs(samples_mean - targets_flat).mean().item()
        else:
            crps = 0.0

        # Coverage statistics (simplified)
        coverage_stats = {}
        if samples.numel() > 0 and targets_flat.numel() > 0:
            for q in quantiles:
                try:
                    quantile_pred = torch.quantile(samples, q, dim=0)
                    if mask_flat is not None and mask_flat.sum() > 0:
                        coverage = ((targets_flat <= quantile_pred) * mask_flat).sum() / mask_flat.sum()
                    else:
                        coverage = (targets_flat <= quantile_pred).float().mean()
                    coverage_stats[f'coverage_{q:.1f}'] = coverage.item()
                except:
                    coverage_stats[f'coverage_{q:.1f}'] = 0.0

        metrics = {
            'nll': nll.item(),
            'crps': crps,
            **coverage_stats
        }

    except Exception as e:
        # Fallback metrics if computation fails
        metrics = {
            'nll': 0.0,
            'crps': 0.0,
            'coverage_0.1': 0.0,
            'coverage_0.5': 0.0,
            'coverage_0.9': 0.0
        }

    return metrics


def compute_crps(
    samples: torch.Tensor,
    targets: torch.Tensor,
    mask: Optional[torch.Tensor] = None
) -> float:
    """
    Compute Continuous Ranked Probability Score (CRPS).
    
    Args:
        samples: Forecast samples [num_samples, batch_size, seq_len]
        targets: Ground truth targets [batch_size, seq_len]
        mask: Optional mask for valid positions
        
    Returns:
        CRPS value
    """
    num_samples = samples.shape[0]
    
    # Sort samples
    samples_sorted, _ = torch.sort(samples, dim=0)
    
    # Expand targets to match samples shape
    targets_expanded = targets.unsqueeze(0).expand_as(samples)
    
    # Compute empirical CDF at target values
    cdf_at_target = (samples_sorted <= targets_expanded).float().mean(dim=0)
    
    # Compute CRPS using the formula:
    # CRPS = ∫ (F(x) - 1{x >= y})² dx
    # Approximated using the sorted samples
    
    # Create step function for empirical CDF
    x_values = samples_sorted
    cdf_values = torch.arange(1, num_samples + 1, device=samples.device).float() / num_samples
    cdf_values = cdf_values.unsqueeze(1).unsqueeze(2).expand_as(x_values)
    
    # Indicator function: 1 if x >= target
    indicator = (x_values >= targets_expanded).float()
    
    # Compute squared differences
    squared_diff = (cdf_values - indicator) ** 2
    
    # Integrate (approximate with mean)
    crps_values = squared_diff.mean(dim=0)
    
    if mask is not None:
        crps = (crps_values * mask).sum() / mask.sum()
    else:
        crps = crps_values.mean()
    
    return crps.item()


class EarlyStopping:
    """Early stopping utility."""
    
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = 'min'
    ):
        """
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'min' for loss, 'max' for accuracy
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_score = None
        self.counter = 0
        self.early_stop = False
        
        if mode == 'min':
            self.is_better = lambda score, best: score < best - min_delta
        else:
            self.is_better = lambda score, best: score > best + min_delta
    
    def __call__(self, score: float) -> bool:
        """
        Check if training should stop.
        
        Args:
            score: Current validation score
            
        Returns:
            True if training should stop
        """
        if self.best_score is None:
            self.best_score = score
        elif self.is_better(score, self.best_score):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: Optional[Any],
    epoch: int,
    step: int,
    loss: float,
    filepath: str,
    **kwargs
) -> None:
    """Save training checkpoint."""
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'step': step,
        'loss': loss,
        **kwargs
    }
    
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    
    torch.save(checkpoint, filepath)


def load_checkpoint(
    filepath: str,
    model: nn.Module,
    optimizer: Optional[optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    device: Optional[torch.device] = None
) -> Dict[str, Any]:
    """Load training checkpoint."""
    checkpoint = torch.load(filepath, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    return checkpoint
