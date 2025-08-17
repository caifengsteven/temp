"""
Model Utilities

Utility functions for the LSTM-BEKK model implementation.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional
import logging


class ModelUtils:
    """Utility functions for LSTM-BEKK model."""
    
    @staticmethod
    def ensure_positive_definite(matrix: torch.Tensor, min_eigenvalue: float = 1e-6) -> torch.Tensor:
        """
        Ensure a matrix is positive definite by adjusting eigenvalues.
        
        Args:
            matrix: Input matrix
            min_eigenvalue: Minimum eigenvalue threshold
            
        Returns:
            Positive definite matrix
        """
        # Eigenvalue decomposition
        eigenvalues, eigenvectors = torch.symeig(matrix, eigenvectors=True)
        
        # Clamp eigenvalues to minimum value
        eigenvalues = torch.clamp(eigenvalues, min=min_eigenvalue)
        
        # Reconstruct matrix
        return eigenvectors @ torch.diag(eigenvalues) @ eigenvectors.T
    
    @staticmethod
    def create_lower_triangular_matrix(vector: torch.Tensor, n_assets: int) -> torch.Tensor:
        """
        Create lower triangular matrix from vector.
        
        Args:
            vector: Input vector
            n_assets: Number of assets
            
        Returns:
            Lower triangular matrix
        """
        # Calculate expected vector length
        expected_length = n_assets * (n_assets + 1) // 2
        
        if len(vector) != expected_length:
            raise ValueError(f"Vector length {len(vector)} doesn't match expected {expected_length}")
        
        # Create lower triangular matrix
        L = torch.zeros(n_assets, n_assets, dtype=vector.dtype, device=vector.device)
        
        # Fill lower triangular part
        tril_indices = torch.tril_indices(n_assets, n_assets)
        L[tril_indices[0], tril_indices[1]] = vector
        
        return L
    
    @staticmethod
    def matrix_to_vector(matrix: torch.Tensor) -> torch.Tensor:
        """
        Convert lower triangular matrix to vector.
        
        Args:
            matrix: Lower triangular matrix
            
        Returns:
            Vector representation
        """
        tril_indices = torch.tril_indices(matrix.size(0), matrix.size(1))
        return matrix[tril_indices[0], tril_indices[1]]
    
    @staticmethod
    def swish_activation(x: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        """
        Swish activation function: x * sigmoid(beta * x)
        
        Args:
            x: Input tensor
            beta: Learnable parameter
            
        Returns:
            Activated tensor
        """
        return x * torch.sigmoid(beta * x)
    
    @staticmethod
    def check_constraints(a: torch.Tensor, b: torch.Tensor) -> bool:
        """
        Check BEKK constraints: a,b >= 0 and a+b < 1
        
        Args:
            a: BEKK parameter a
            b: BEKK parameter b
            
        Returns:
            True if constraints are satisfied
        """
        return (a >= 0).all() and (b >= 0).all() and (a + b < 1).all()
    
    @staticmethod
    def apply_constraints(a: torch.Tensor, b: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply BEKK constraints using sigmoid transformation.
        
        Args:
            a: Raw BEKK parameter a
            b: Raw BEKK parameter b
            
        Returns:
            Constrained parameters
        """
        # Apply sigmoid to ensure positivity
        a_constrained = torch.sigmoid(a)
        b_constrained = torch.sigmoid(b)
        
        # Ensure a + b < 1
        sum_params = a_constrained + b_constrained
        scale_factor = torch.where(sum_params >= 0.99, 0.99 / sum_params, torch.ones_like(sum_params))
        
        a_constrained = a_constrained * scale_factor
        b_constrained = b_constrained * scale_factor
        
        return a_constrained, b_constrained
    
    @staticmethod
    def negative_log_likelihood(returns: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
        """
        Calculate negative log-likelihood for multivariate normal distribution.
        
        Args:
            returns: Return vectors (T x n)
            H: Conditional covariance matrices (T x n x n)
            
        Returns:
            Negative log-likelihood
        """
        T, n = returns.shape
        
        # Calculate log determinant and inverse
        try:
            # Use Cholesky decomposition for numerical stability
            L = torch.cholesky(H)
            log_det = 2 * torch.sum(torch.log(torch.diagonal(L, dim1=-2, dim2=-1)), dim=-1)

            # Solve using Cholesky factor (manual triangular solve for compatibility)
            # Solve L * y = returns for y
            y = torch.zeros_like(returns.unsqueeze(-1))
            for i in range(T):
                y[i] = torch.triangular_solve(returns[i].unsqueeze(-1), L[i], upper=False)[0]
            quadratic_form = torch.sum(y.squeeze(-1) ** 2, dim=-1)

        except RuntimeError:
            # Fallback to regular computation if Cholesky fails
            log_det = torch.logdet(H)
            H_inv = torch.inverse(H)
            quadratic_form = torch.sum(returns.unsqueeze(-2) @ H_inv @ returns.unsqueeze(-1), dim=(-2, -1)).squeeze()
        
        # Calculate negative log-likelihood
        nll = 0.5 * (n * np.log(2 * np.pi) + log_det + quadratic_form)
        
        return torch.sum(nll)
    
    @staticmethod
    def initialize_parameters(n_assets: int, method: str = "xavier") -> dict:
        """
        Initialize model parameters.
        
        Args:
            n_assets: Number of assets
            method: Initialization method
            
        Returns:
            Dictionary of initialized parameters
        """
        if method == "xavier":
            # Xavier initialization
            C_init = torch.randn(n_assets, n_assets) * np.sqrt(2.0 / n_assets)
            # Make lower triangular
            C_init = torch.tril(C_init)
            # Ensure positive diagonal
            C_init.diagonal().fill_(0.1)
            
        elif method == "identity":
            # Identity-based initialization
            C_init = torch.eye(n_assets) * 0.1
            
        else:
            raise ValueError(f"Unknown initialization method: {method}")
        
        # BEKK parameters
        a_init = torch.tensor(0.05)  # Small positive value
        b_init = torch.tensor(0.90)  # High persistence
        
        # Swish parameter
        beta_init = torch.tensor(1.0)
        
        return {
            'C': C_init,
            'a': a_init,
            'b': b_init,
            'beta': beta_init
        }
    
    @staticmethod
    def validate_inputs(returns: torch.Tensor) -> None:
        """
        Validate input data for the model.
        
        Args:
            returns: Return data
        """
        if not isinstance(returns, torch.Tensor):
            raise TypeError("Returns must be a torch.Tensor")
        
        if returns.dim() != 2:
            raise ValueError("Returns must be 2-dimensional (T x n)")
        
        if torch.isnan(returns).any():
            raise ValueError("Returns contain NaN values")
        
        if torch.isinf(returns).any():
            raise ValueError("Returns contain infinite values")
        
        if returns.shape[0] < 10:
            raise ValueError("Insufficient time periods for estimation")
        
        if returns.shape[1] < 2:
            raise ValueError("Need at least 2 assets for multivariate modeling")


class ConstrainedParameter(nn.Module):
    """
    A parameter with constraints applied via transformation.
    """
    
    def __init__(self, initial_value: torch.Tensor, constraint_type: str = "positive"):
        """
        Initialize constrained parameter.
        
        Args:
            initial_value: Initial parameter value
            constraint_type: Type of constraint ('positive', 'unit_interval', 'bekk_sum')
        """
        super().__init__()
        self.constraint_type = constraint_type
        
        # Store unconstrained parameter
        if constraint_type == "positive":
            # Use log transformation for positivity
            self.unconstrained = nn.Parameter(torch.log(initial_value + 1e-8))
        elif constraint_type == "unit_interval":
            # Use logit transformation for [0,1] interval
            self.unconstrained = nn.Parameter(torch.logit(initial_value.clamp(1e-6, 1-1e-6)))
        else:
            self.unconstrained = nn.Parameter(initial_value)
    
    def forward(self) -> torch.Tensor:
        """Apply constraint transformation."""
        if self.constraint_type == "positive":
            return torch.exp(self.unconstrained)
        elif self.constraint_type == "unit_interval":
            return torch.sigmoid(self.unconstrained)
        else:
            return self.unconstrained
