"""
Fusion in Latent Space module for CUBIC framework
Implements stock embedding and multi-head pooling mechanism
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Tuple, Optional

logger = logging.getLogger(__name__)


class StockEmbedding(nn.Module):
    """
    Embedding module for individual stocks
    Projects stock indicators into latent space
    """
    
    def __init__(self, input_dim: int, embedding_dim: int = 32, dropout: float = 0.1):
        """
        Initialize Stock Embedding
        
        Args:
            input_dim: Number of input features per stock
            embedding_dim: Dimension of embedding space
            dropout: Dropout rate
        """
        super().__init__()
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        
        # MLP for embedding as mentioned in the paper
        self.embedding_mlp = nn.Sequential(
            nn.Linear(input_dim, embedding_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, embedding_dim)
        )
        
        # Layer normalization for stability
        self.layer_norm = nn.LayerNorm(embedding_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for stock embedding
        
        Args:
            x: Input features (batch_size, sequence_length, n_stocks, input_dim)
            
        Returns:
            Embedded features (batch_size, sequence_length, n_stocks, embedding_dim)
        """
        batch_size, seq_len, n_stocks, input_dim = x.shape
        
        # Reshape for processing
        x_reshaped = x.view(-1, input_dim)  # (batch_size * seq_len * n_stocks, input_dim)
        
        # Apply embedding MLP
        embedded = self.embedding_mlp(x_reshaped)  # (batch_size * seq_len * n_stocks, embedding_dim)
        
        # Apply layer normalization
        embedded = self.layer_norm(embedded)
        
        # Reshape back
        embedded = embedded.view(batch_size, seq_len, n_stocks, self.embedding_dim)
        
        return embedded


class MultiHeadPooling(nn.Module):
    """
    Multi-head pooling mechanism for aggregating stock embeddings
    Implements max, mean, and min pooling as described in the paper
    """
    
    def __init__(self, embedding_dim: int):
        """
        Initialize Multi-Head Pooling
        
        Args:
            embedding_dim: Dimension of stock embeddings
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        
        # Output dimension is 3 * embedding_dim (max + mean + min)
        self.output_dim = 3 * embedding_dim
        
    def forward(self, embedded_stocks: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for multi-head pooling
        
        Args:
            embedded_stocks: Stock embeddings (batch_size, sequence_length, n_stocks, embedding_dim)
            
        Returns:
            Pooled features (batch_size, sequence_length, 3 * embedding_dim)
        """
        # Pool across the stock dimension (dim=2)
        
        # Max pooling - captures extreme market movements and dominant patterns
        max_pooled = torch.max(embedded_stocks, dim=2)[0]  # (batch_size, seq_len, embedding_dim)
        
        # Mean pooling - preserves overall trend and collective behavior
        mean_pooled = torch.mean(embedded_stocks, dim=2)  # (batch_size, seq_len, embedding_dim)
        
        # Min pooling - captures lower bounds and potential risks
        min_pooled = torch.min(embedded_stocks, dim=2)[0]  # (batch_size, seq_len, embedding_dim)
        
        # Concatenate all pooling results
        pooled = torch.cat([max_pooled, mean_pooled, min_pooled], dim=-1)
        
        return pooled


class AttentionPooling(nn.Module):
    """
    Attention-based pooling as an alternative to multi-head pooling
    """
    
    def __init__(self, embedding_dim: int, num_heads: int = 4):
        """
        Initialize Attention Pooling
        
        Args:
            embedding_dim: Dimension of stock embeddings
            num_heads: Number of attention heads
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        
        assert embedding_dim % num_heads == 0, "embedding_dim must be divisible by num_heads"
        
        # Attention layers
        self.query = nn.Linear(embedding_dim, embedding_dim)
        self.key = nn.Linear(embedding_dim, embedding_dim)
        self.value = nn.Linear(embedding_dim, embedding_dim)
        
        self.output_projection = nn.Linear(embedding_dim, embedding_dim)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, embedded_stocks: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for attention pooling
        
        Args:
            embedded_stocks: Stock embeddings (batch_size, sequence_length, n_stocks, embedding_dim)
            
        Returns:
            Attention-pooled features (batch_size, sequence_length, embedding_dim)
        """
        batch_size, seq_len, n_stocks, embedding_dim = embedded_stocks.shape
        
        # Reshape for attention computation
        x = embedded_stocks.view(batch_size * seq_len, n_stocks, embedding_dim)
        
        # Compute queries, keys, values
        Q = self.query(x)  # (batch_size * seq_len, n_stocks, embedding_dim)
        K = self.key(x)
        V = self.value(x)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size * seq_len, n_stocks, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size * seq_len, n_stocks, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size * seq_len, n_stocks, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        attended = torch.matmul(attention_weights, V)
        
        # Reshape and concatenate heads
        attended = attended.transpose(1, 2).contiguous().view(
            batch_size * seq_len, n_stocks, embedding_dim
        )
        
        # Global pooling (mean across stocks)
        pooled = torch.mean(attended, dim=1)  # (batch_size * seq_len, embedding_dim)
        
        # Output projection
        output = self.output_projection(pooled)
        
        # Reshape back
        output = output.view(batch_size, seq_len, embedding_dim)
        
        return output


class FusionInLatentSpace(nn.Module):
    """
    Complete fusion in latent space module
    Combines stock embedding and pooling mechanisms
    """
    
    def __init__(self, input_dim: int, embedding_dim: int = 32, 
                 pooling_type: str = "multi_head", dropout: float = 0.1,
                 num_attention_heads: int = 4):
        """
        Initialize Fusion in Latent Space
        
        Args:
            input_dim: Number of input features per stock
            embedding_dim: Dimension of embedding space
            pooling_type: Type of pooling ("multi_head" or "attention")
            dropout: Dropout rate
            num_attention_heads: Number of attention heads (for attention pooling)
        """
        super().__init__()
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.pooling_type = pooling_type
        
        # Stock embedding module
        self.stock_embedding = StockEmbedding(input_dim, embedding_dim, dropout)
        
        # Pooling module
        if pooling_type == "multi_head":
            self.pooling = MultiHeadPooling(embedding_dim)
            self.output_dim = 3 * embedding_dim
        elif pooling_type == "attention":
            self.pooling = AttentionPooling(embedding_dim, num_attention_heads)
            self.output_dim = embedding_dim
        else:
            raise ValueError(f"Unknown pooling type: {pooling_type}")
        
        logger.info(f"Fusion module initialized with {pooling_type} pooling, "
                   f"input_dim={input_dim}, embedding_dim={embedding_dim}, "
                   f"output_dim={self.output_dim}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for fusion in latent space
        
        Args:
            x: Input features (batch_size, sequence_length, n_stocks, input_dim)
            
        Returns:
            Fused features (batch_size, sequence_length, output_dim)
        """
        # Embed individual stocks
        embedded_stocks = self.stock_embedding(x)
        
        # Apply pooling to aggregate information
        fused_features = self.pooling(embedded_stocks)
        
        return fused_features
    
    def get_stock_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get individual stock embeddings (for analysis purposes)
        
        Args:
            x: Input features
            
        Returns:
            Stock embeddings before pooling
        """
        return self.stock_embedding(x)
    
    def get_attention_weights(self, x: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Get attention weights (only available for attention pooling)
        
        Args:
            x: Input features
            
        Returns:
            Attention weights if using attention pooling, None otherwise
        """
        if self.pooling_type == "attention" and hasattr(self.pooling, 'get_attention_weights'):
            embedded_stocks = self.stock_embedding(x)
            return self.pooling.get_attention_weights(embedded_stocks)
        else:
            logger.warning("Attention weights not available for this pooling type")
            return None
