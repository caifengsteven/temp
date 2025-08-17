"""
LSTM Components

Implementation of LSTM components for the LSTM-BEKK framework.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional
import logging

from .model_utils import ModelUtils


class LSTMComponent(nn.Module):
    """
    LSTM component for generating dynamic covariance matrices.
    
    Generates the dynamic component C_t in the LSTM-BEKK model.
    """
    
    def __init__(self, n_assets: int, hidden_size: Optional[int] = None, 
                 num_layers: int = 3, dropout: float = 0.2):
        """
        Initialize LSTM component.
        
        Args:
            n_assets: Number of assets
            hidden_size: LSTM hidden size (defaults to n_assets)
            num_layers: Number of LSTM layers
            dropout: Dropout rate
        """
        super().__init__()
        self.n_assets = n_assets
        self.hidden_size = hidden_size or n_assets
        self.num_layers = num_layers
        self.dropout = dropout
        self.logger = logging.getLogger(__name__)
        
        # Calculate output size for lower triangular matrix
        self.output_size = n_assets * (n_assets + 1) // 2
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=n_assets,
            hidden_size=self.hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Output projection layer
        self.output_layer = nn.Linear(self.hidden_size, self.output_size)
        
        # Swish activation parameter (learnable)
        self.beta = nn.Parameter(torch.tensor(1.0))
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize LSTM and linear layer weights."""
        # Initialize LSTM weights
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
                # Set forget gate bias to 1 (standard practice)
                n = param.size(0)
                param.data[n//4:n//2].fill_(1)
        
        # Initialize output layer
        nn.init.xavier_uniform_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)
    
    def forward(self, returns: torch.Tensor, 
                hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass of LSTM component.
        
        Args:
            returns: Return vectors (batch_size x seq_len x n_assets) or (seq_len x n_assets)
            hidden_state: Initial hidden state (h_0, c_0)
            
        Returns:
            Tuple of (dynamic_matrices, final_hidden_state)
        """
        # Handle 2D input (add batch dimension)
        if returns.dim() == 2:
            returns = returns.unsqueeze(0)  # (1 x seq_len x n_assets)
            squeeze_output = True
        else:
            squeeze_output = False
        
        batch_size, seq_len, n_assets = returns.shape
        
        # LSTM forward pass
        lstm_output, hidden_state = self.lstm(returns, hidden_state)
        
        # Project to output space
        C_tilde = self.output_layer(lstm_output)  # (batch_size x seq_len x output_size)
        
        # Reshape to lower triangular matrices
        C_matrices = torch.zeros(batch_size, seq_len, n_assets, n_assets, 
                                device=returns.device, dtype=returns.dtype)
        
        for b in range(batch_size):
            for t in range(seq_len):
                # Create lower triangular matrix
                C_t = ModelUtils.create_lower_triangular_matrix(C_tilde[b, t], n_assets)
                
                # Apply Swish activation to diagonal elements
                diag_indices = torch.arange(n_assets)
                diag_elements = C_t[diag_indices, diag_indices]
                activated_diag = ModelUtils.swish_activation(diag_elements, self.beta)
                C_t = C_t.clone()  # Avoid in-place operation
                C_t[diag_indices, diag_indices] = activated_diag
                
                C_matrices[b, t] = C_t
        
        if squeeze_output:
            C_matrices = C_matrices.squeeze(0)  # Remove batch dimension
        
        return C_matrices, hidden_state
    
    def init_hidden(self, batch_size: int = 1, device: torch.device = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Initialize hidden state.
        
        Args:
            batch_size: Batch size
            device: Device to place tensors on
            
        Returns:
            Initial hidden state (h_0, c_0)
        """
        if device is None:
            device = next(self.parameters()).device
        
        h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        c_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        
        return h_0, c_0
    
    def get_dynamic_covariance(self, returns: torch.Tensor) -> torch.Tensor:
        """
        Get dynamic covariance matrices C_t @ C_t'.
        
        Args:
            returns: Return vectors
            
        Returns:
            Dynamic covariance matrices
        """
        C_matrices, _ = self.forward(returns)
        
        # Calculate C_t @ C_t'
        if C_matrices.dim() == 3:  # (seq_len x n_assets x n_assets)
            seq_len = C_matrices.shape[0]
            dynamic_cov = torch.zeros_like(C_matrices)
            for t in range(seq_len):
                dynamic_cov[t] = C_matrices[t] @ C_matrices[t].T
        else:  # (batch_size x seq_len x n_assets x n_assets)
            dynamic_cov = torch.matmul(C_matrices, C_matrices.transpose(-2, -1))
        
        return dynamic_cov


class EnhancedLSTMComponent(LSTMComponent):
    """
    Enhanced LSTM component with attention mechanism and residual connections.
    """
    
    def __init__(self, n_assets: int, hidden_size: Optional[int] = None,
                 num_layers: int = 3, dropout: float = 0.2, use_attention: bool = True):
        """
        Initialize enhanced LSTM component.
        
        Args:
            n_assets: Number of assets
            hidden_size: LSTM hidden size
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            use_attention: Whether to use attention mechanism
        """
        super().__init__(n_assets, hidden_size, num_layers, dropout)
        self.use_attention = use_attention
        
        if use_attention:
            # Self-attention mechanism
            self.attention = nn.MultiheadAttention(
                embed_dim=self.hidden_size,
                num_heads=max(1, self.hidden_size // 64),
                dropout=dropout,
                batch_first=True
            )
            
            # Layer normalization
            self.layer_norm = nn.LayerNorm(self.hidden_size)
        
        # Residual connection for output
        self.residual_layer = nn.Linear(n_assets, self.output_size)
        
        # Gating mechanism for combining LSTM and residual
        self.gate = nn.Linear(self.hidden_size + n_assets, 1)
    
    def forward(self, returns: torch.Tensor,
                hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Enhanced forward pass with attention and residual connections."""
        # Handle 2D input
        if returns.dim() == 2:
            returns = returns.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        batch_size, seq_len, n_assets = returns.shape
        
        # LSTM forward pass
        lstm_output, hidden_state = self.lstm(returns, hidden_state)
        
        # Apply attention if enabled
        if self.use_attention:
            attended_output, _ = self.attention(lstm_output, lstm_output, lstm_output)
            lstm_output = self.layer_norm(lstm_output + attended_output)
        
        # LSTM-based output
        lstm_projection = self.output_layer(lstm_output)
        
        # Residual connection
        residual_projection = self.residual_layer(returns)
        
        # Gating mechanism
        gate_input = torch.cat([lstm_output, returns], dim=-1)
        gate_weights = torch.sigmoid(self.gate(gate_input))
        
        # Combine outputs
        C_tilde = gate_weights * lstm_projection + (1 - gate_weights) * residual_projection
        
        # Create matrices (same as parent class)
        C_matrices = torch.zeros(batch_size, seq_len, n_assets, n_assets,
                                device=returns.device, dtype=returns.dtype)
        
        for b in range(batch_size):
            for t in range(seq_len):
                C_t = ModelUtils.create_lower_triangular_matrix(C_tilde[b, t], n_assets)
                diag_indices = torch.arange(n_assets)
                diag_elements = C_t[diag_indices, diag_indices]
                activated_diag = ModelUtils.swish_activation(diag_elements, self.beta)
                C_t = C_t.clone()  # Avoid in-place operation
                C_t[diag_indices, diag_indices] = activated_diag
                C_matrices[b, t] = C_t
        
        if squeeze_output:
            C_matrices = C_matrices.squeeze(0)
        
        return C_matrices, hidden_state
