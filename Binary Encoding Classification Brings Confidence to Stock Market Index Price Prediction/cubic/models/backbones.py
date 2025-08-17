"""
Backbone architectures for CUBIC framework
Implements LSTM, Transformer, and MLP backbones
"""

import torch
import torch.nn as nn
import math
import logging
from typing import Tuple, Optional

logger = logging.getLogger(__name__)


class LSTMBackbone(nn.Module):
    """
    LSTM backbone for CUBIC framework
    """
    
    def __init__(self, input_dim: int, hidden_size: int = 128, num_layers: int = 2, 
                 dropout: float = 0.1, bidirectional: bool = False):
        """
        Initialize LSTM Backbone
        
        Args:
            input_dim: Input feature dimension
            hidden_size: Hidden state dimension
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            bidirectional: Whether to use bidirectional LSTM
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        # Output dimension
        self.output_dim = hidden_size * (2 if bidirectional else 1)
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
        logger.info(f"LSTM backbone initialized: input_dim={input_dim}, "
                   f"hidden_size={hidden_size}, num_layers={num_layers}, "
                   f"output_dim={self.output_dim}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for LSTM backbone
        
        Args:
            x: Input tensor (batch_size, sequence_length, input_dim)
            
        Returns:
            Output tensor (batch_size, output_dim)
        """
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Use the last hidden state
        if self.bidirectional:
            # Concatenate forward and backward hidden states
            output = torch.cat([hidden[-2], hidden[-1]], dim=1)
        else:
            output = hidden[-1]
        
        # Apply dropout
        output = self.dropout(output)
        
        return output


class TransformerBackbone(nn.Module):
    """
    Transformer backbone for CUBIC framework
    """
    
    def __init__(self, input_dim: int, d_model: int = 64, nhead: int = 8, 
                 num_layers: int = 1, dim_feedforward: int = 256, dropout: float = 0.1):
        """
        Initialize Transformer Backbone
        
        Args:
            input_dim: Input feature dimension
            d_model: Model dimension
            nhead: Number of attention heads
            num_layers: Number of transformer layers
            dim_feedforward: Feedforward network dimension
            dropout: Dropout rate
        """
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        
        # Input projection to d_model
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(d_model, dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(d_model)
        
        # Output dimension
        self.output_dim = d_model
        
        logger.info(f"Transformer backbone initialized: input_dim={input_dim}, "
                   f"d_model={d_model}, nhead={nhead}, num_layers={num_layers}, "
                   f"output_dim={self.output_dim}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for Transformer backbone
        
        Args:
            x: Input tensor (batch_size, sequence_length, input_dim)
            
        Returns:
            Output tensor (batch_size, output_dim)
        """
        # Project input to d_model
        x = self.input_projection(x)
        
        # Add positional encoding
        x = self.positional_encoding(x)
        
        # Transformer encoding
        encoded = self.transformer_encoder(x)
        
        # Global average pooling over sequence dimension
        output = torch.mean(encoded, dim=1)
        
        # Layer normalization
        output = self.layer_norm(output)
        
        return output


class MLPBackbone(nn.Module):
    """
    MLP backbone for CUBIC framework
    """
    
    def __init__(self, input_dim: int, hidden_layers: list = [256, 128, 64], 
                 dropout: float = 0.1, activation: str = "relu"):
        """
        Initialize MLP Backbone
        
        Args:
            input_dim: Input feature dimension
            hidden_layers: List of hidden layer dimensions
            dropout: Dropout rate
            activation: Activation function ("relu", "gelu", "tanh")
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.dropout_rate = dropout

        # Activation function
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # Build MLP layers
        layers = []
        # Note: input_dim here is the fusion output dim, but we need to account for sequence flattening
        # This will be handled in forward pass
        self.layers_config = []
        prev_dim = None  # Will be set dynamically in forward pass

        for hidden_dim in hidden_layers:
            self.layers_config.append(hidden_dim)

        # We'll build the actual layers in the first forward pass
        self.mlp = None
        self.built = False
        
        # Output dimension
        self.output_dim = hidden_layers[-1] if hidden_layers else input_dim
        
        logger.info(f"MLP backbone initialized: input_dim={input_dim}, "
                   f"hidden_layers={hidden_layers}, output_dim={self.output_dim}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for MLP backbone

        Args:
            x: Input tensor (batch_size, sequence_length, input_dim)

        Returns:
            Output tensor (batch_size, output_dim)
        """
        # Flatten sequence dimension
        batch_size, seq_len, input_dim = x.shape
        x_flat = x.view(batch_size, seq_len * input_dim)

        # Build MLP layers if not built yet
        if not self.built:
            self._build_mlp(seq_len * input_dim)

        # MLP forward pass
        output = self.mlp(x_flat)

        return output

    def _build_mlp(self, flattened_input_dim: int):
        """Build MLP layers with correct input dimension"""
        layers = []
        prev_dim = flattened_input_dim

        for hidden_dim in self.layers_config:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                self.activation,
                nn.Dropout(self.dropout_rate)
            ])
            prev_dim = hidden_dim

        self.mlp = nn.Sequential(*layers)
        self.output_dim = self.layers_config[-1] if self.layers_config else flattened_input_dim
        self.built = True


class PositionalEncoding(nn.Module):
    """
    Positional encoding for Transformer
    """
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        """
        Initialize Positional Encoding
        
        Args:
            d_model: Model dimension
            dropout: Dropout rate
            max_len: Maximum sequence length
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        # Register as buffer (not a parameter)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input
        
        Args:
            x: Input tensor (batch_size, sequence_length, d_model)
            
        Returns:
            Tensor with positional encoding added
        """
        x = x + self.pe[:x.size(1), :].transpose(0, 1)
        return self.dropout(x)
