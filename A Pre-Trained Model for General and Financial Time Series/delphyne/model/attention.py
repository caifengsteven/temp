"""
Any-variate attention mechanism for Delphyne model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
from .embeddings import RotaryPositionalEmbedding, apply_rotary_pos_emb


class AnyVariateAttention(nn.Module):
    """
    Any-variate attention mechanism as described in the paper.
    
    This allows binary attention biases to encode variate indices for 
    flattened multi-variate time series. The attention score between 
    the (i, m)-th query and (j, n)-th key is calculated with special
    bias terms for variate relationships.
    
    From the paper's appendix:
    E_ij,mn = W_Q * x_i,m^T * R_i-j * W_K * x_j,n + B_mn
    
    Where B_mn is the variate bias term.
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                f"hidden_size ({self.hidden_size}) must be divisible by "
                f"num_attention_heads ({self.num_attention_heads})"
            )
        
        # Query, Key, Value projections
        self.query = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.key = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.value = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        
        # Output projection
        self.out_proj = nn.Linear(self.hidden_size, self.hidden_size)
        
        # Rotary positional embedding
        if config.rotary_embedding:
            self.rotary_emb = RotaryPositionalEmbedding(
                self.head_dim, 
                config.max_position_embeddings
            )
        else:
            self.rotary_emb = None
        
        # Any-variate bias parameters
        # We'll use a learnable bias matrix for variate relationships
        self.max_variates = 1000  # Should match embeddings
        self.variate_bias = nn.Parameter(
            torch.zeros(self.max_variates, self.max_variates)
        )
        
        # Dropout
        self.dropout = nn.Dropout(config.dropout_prob)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        variate_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            hidden_states: Input tensor [batch_size, seq_len, hidden_size]
            variate_ids: Variate identifiers [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len] or [batch_size, seq_len, seq_len]
            output_attentions: Whether to return attention weights
            
        Returns:
            Tuple of (output, attention_weights)
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # Linear projections and reshape for multi-head attention
        query_states = self.query(hidden_states)
        key_states = self.key(hidden_states)
        value_states = self.value(hidden_states)
        
        # Reshape to [batch_size, seq_len, num_heads, head_dim]
        query_states = query_states.view(batch_size, seq_len, self.num_attention_heads, self.head_dim)
        key_states = key_states.view(batch_size, seq_len, self.num_attention_heads, self.head_dim)
        value_states = value_states.view(batch_size, seq_len, self.num_attention_heads, self.head_dim)
        
        # Apply rotary positional embedding if enabled
        if self.rotary_emb is not None:
            cos, sin = self.rotary_emb(query_states, seq_len)
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        
        # Transpose for attention computation: [batch_size, num_heads, seq_len, head_dim]
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)
        
        # Compute attention scores
        attention_scores = torch.matmul(query_states, key_states.transpose(-2, -1)) * self.scale
        
        # Add any-variate bias
        # Create variate bias matrix for this batch
        variate_bias_matrix = self._compute_variate_bias(variate_ids, seq_len, hidden_states.device)
        
        # Add bias to attention scores
        # variate_bias_matrix: [batch_size, seq_len, seq_len]
        # attention_scores: [batch_size, num_heads, seq_len, seq_len]
        attention_scores = attention_scores + variate_bias_matrix.unsqueeze(1)
        
        # Apply attention mask if provided
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                # Convert to 4D mask [batch_size, 1, 1, seq_len]
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)
            elif attention_mask.dim() == 3:
                # Convert to 4D mask [batch_size, 1, seq_len, seq_len]
                attention_mask = attention_mask.unsqueeze(1)
            
            # Apply mask (set masked positions to large negative value)
            attention_scores = attention_scores.masked_fill(
                attention_mask == 0, torch.finfo(attention_scores.dtype).min
            )
        
        # Compute attention probabilities
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        
        # Apply attention to values
        context_layer = torch.matmul(attention_probs, value_states)
        
        # Reshape back to [batch_size, seq_len, hidden_size]
        context_layer = context_layer.transpose(1, 2).contiguous()
        context_layer = context_layer.view(batch_size, seq_len, self.hidden_size)
        
        # Final output projection
        output = self.out_proj(context_layer)
        
        outputs = (output,)
        if output_attentions:
            outputs = outputs + (attention_probs,)
        
        return outputs
    
    def _compute_variate_bias(self, variate_ids: torch.Tensor, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Compute the variate bias matrix for any-variate attention.
        
        Args:
            variate_ids: Variate identifiers [batch_size, seq_len]
            seq_len: Sequence length
            device: Device for computation
            
        Returns:
            Bias matrix [batch_size, seq_len, seq_len]
        """
        batch_size = variate_ids.shape[0]
        
        # Create bias matrix by indexing into the learned variate bias
        # variate_ids: [batch_size, seq_len]
        # We want to create a matrix where entry (i,j) contains the bias for 
        # the relationship between variate_ids[i] and variate_ids[j]
        
        # Expand variate_ids for broadcasting
        variate_i = variate_ids.unsqueeze(2)  # [batch_size, seq_len, 1]
        variate_j = variate_ids.unsqueeze(1)  # [batch_size, 1, seq_len]
        
        # Create bias matrix by indexing
        # This creates a [batch_size, seq_len, seq_len] matrix where each entry
        # contains the learned bias for the corresponding variate pair
        bias_matrix = self.variate_bias[variate_i, variate_j]  # [batch_size, seq_len, seq_len]
        
        return bias_matrix
