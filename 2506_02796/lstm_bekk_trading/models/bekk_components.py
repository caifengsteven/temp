"""
BEKK Components

Implementation of BEKK model components for the LSTM-BEKK framework.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional
import logging

from .model_utils import ModelUtils, ConstrainedParameter


class BEKKLayer(nn.Module):
    """
    BEKK layer implementing the econometric component of LSTM-BEKK.
    
    Implements: H_t = CC' + a*r_{t-1}*r_{t-1}' + b*H_{t-1}
    """
    
    def __init__(self, n_assets: int, constraint_method: str = "sigmoid"):
        """
        Initialize BEKK layer.
        
        Args:
            n_assets: Number of assets
            constraint_method: Method for applying constraints
        """
        super().__init__()
        self.n_assets = n_assets
        self.constraint_method = constraint_method
        self.logger = logging.getLogger(__name__)
        
        # Initialize parameters
        params = ModelUtils.initialize_parameters(n_assets)
        
        # Static covariance matrix C (lower triangular)
        self.C = nn.Parameter(params['C'])
        
        # BEKK parameters with constraints
        if constraint_method == "sigmoid":
            self.a_raw = nn.Parameter(torch.tensor(0.0))  # Will be sigmoid transformed
            self.b_raw = nn.Parameter(torch.tensor(2.0))  # Will be sigmoid transformed
        else:
            self.a = ConstrainedParameter(params['a'], "positive")
            self.b = ConstrainedParameter(params['b'], "positive")
    
    def get_bekk_parameters(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get constrained BEKK parameters."""
        if self.constraint_method == "sigmoid":
            return ModelUtils.apply_constraints(self.a_raw, self.b_raw)
        else:
            return self.a(), self.b()
    
    def forward(self, returns: torch.Tensor, H_prev: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of BEKK layer.
        
        Args:
            returns: Return vectors (T x n)
            H_prev: Previous covariance matrix (n x n), if None uses unconditional
            
        Returns:
            Conditional covariance matrices (T x n x n)
        """
        T, n = returns.shape
        device = returns.device
        
        # Get constrained parameters
        a, b = self.get_bekk_parameters()
        
        # Static component: CC'
        C_lower = torch.tril(self.C)  # Ensure lower triangular
        # Ensure positive diagonal elements
        diag_elements = torch.diag(C_lower)
        diag_clamped = torch.clamp(diag_elements, min=1e-6)
        diag_correction = torch.diag(diag_clamped - diag_elements)
        C_lower = C_lower + diag_correction
        static_cov = C_lower @ C_lower.T
        
        # Initialize covariance matrices
        H = torch.zeros(T, n, n, device=device)
        
        # Initial covariance (unconditional or provided)
        if H_prev is None:
            # Use unconditional covariance as initial value
            H[0] = static_cov
        else:
            H[0] = H_prev
        
        # Iterate through time
        for t in range(1, T):
            # Previous return outer product
            r_prev = returns[t-1].unsqueeze(1)  # (n x 1)
            shock_component = a * (r_prev @ r_prev.T)  # (n x n)
            
            # Previous covariance
            persistence_component = b * H[t-1]  # (n x n)
            
            # Update covariance
            H[t] = static_cov + shock_component + persistence_component
            
            # Ensure positive definiteness
            H[t] = ModelUtils.ensure_positive_definite(H[t])
        
        return H
    
    def get_unconditional_covariance(self) -> torch.Tensor:
        """
        Calculate unconditional covariance matrix.
        
        Returns:
            Unconditional covariance matrix
        """
        a, b = self.get_bekk_parameters()
        
        # Static component
        C_lower = torch.tril(self.C)
        diag_elements = torch.diag(C_lower)
        diag_clamped = torch.clamp(diag_elements, min=1e-6)
        diag_correction = torch.diag(diag_clamped - diag_elements)
        C_lower = C_lower + diag_correction
        static_cov = C_lower @ C_lower.T
        
        # Unconditional covariance: Omega / (1 - a - b)
        # For scalar BEKK: H_unconditional = CC' / (1 - a - b)
        denominator = 1 - a - b
        if denominator <= 0:
            self.logger.warning("Non-stationary parameters detected")
            denominator = torch.clamp(denominator, min=1e-6)
        
        return static_cov / denominator


class ScalarBEKK(nn.Module):
    """
    Scalar BEKK model implementation.
    
    Simplified version where A = sqrt(a)*I and B = sqrt(b)*I
    """
    
    def __init__(self, n_assets: int):
        """
        Initialize Scalar BEKK model.
        
        Args:
            n_assets: Number of assets
        """
        super().__init__()
        self.n_assets = n_assets
        self.bekk_layer = BEKKLayer(n_assets)
        
    def forward(self, returns: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of Scalar BEKK.
        
        Args:
            returns: Return vectors (T x n)
            
        Returns:
            Tuple of (covariance_matrices, log_likelihood)
        """
        # Validate inputs
        ModelUtils.validate_inputs(returns)
        
        # Get covariance matrices
        H = self.bekk_layer(returns)
        
        # Calculate log-likelihood
        nll = ModelUtils.negative_log_likelihood(returns[1:], H[1:])  # Skip first period
        
        return H, -nll
    
    def predict_covariance(self, returns: torch.Tensor, steps: int = 1) -> torch.Tensor:
        """
        Predict future covariance matrices.
        
        Args:
            returns: Historical returns
            steps: Number of steps to forecast
            
        Returns:
            Predicted covariance matrices
        """
        with torch.no_grad():
            # Get current covariance
            H_current = self.bekk_layer(returns)[-1]  # Last covariance matrix
            
            # Get parameters
            a, b = self.bekk_layer.get_bekk_parameters()
            
            # Static component
            C_lower = torch.tril(self.bekk_layer.C)
            C_lower = C_lower + torch.diag(torch.clamp(torch.diag(C_lower), min=1e-6) - torch.diag(C_lower))
            static_cov = C_lower @ C_lower.T
            
            # Forecast
            H_forecast = torch.zeros(steps, self.n_assets, self.n_assets)
            H_prev = H_current
            
            for step in range(steps):
                # For multi-step forecasting, assume zero returns (unconditional forecast)
                H_forecast[step] = static_cov + b * H_prev
                H_prev = H_forecast[step]
            
            return H_forecast
    
    def get_parameters(self) -> dict:
        """Get model parameters."""
        a, b = self.bekk_layer.get_bekk_parameters()
        
        return {
            'a': a.item(),
            'b': b.item(),
            'C': self.bekk_layer.C.detach().numpy(),
            'unconditional_cov': self.bekk_layer.get_unconditional_covariance().detach().numpy()
        }
