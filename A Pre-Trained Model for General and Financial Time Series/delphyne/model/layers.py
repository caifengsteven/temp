"""
Transformer layers for Delphyne model with GLU and SiLU activation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from .attention import AnyVariateAttention


class DelphyneGLU(nn.Module):
    """
    Gated Linear Unit (GLU) as specified in the paper.
    Replaces the standard FFN with gated activation.
    """
    
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        
        # Gate and up projections
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        
        # Down projection
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        
        # SiLU activation as specified in the paper
        self.act_fn = nn.SiLU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [batch_size, seq_len, hidden_size]
            
        Returns:
            Output tensor [batch_size, seq_len, hidden_size]
        """
        # GLU: gate * activation(up_proj(x)) 
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        
        # Apply SiLU activation to the up projection
        activated_up = self.act_fn(up)
        
        # Element-wise multiplication (gating)
        gated = gate * activated_up
        
        # Down projection
        output = self.down_proj(gated)
        
        return output


class DelphyneLayer(nn.Module):
    """
    Single transformer layer for Delphyne with:
    - Pre-normalization
    - Any-variate attention
    - GLU with SiLU activation
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Pre-normalization as specified in the paper
        self.input_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.post_attention_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # Any-variate attention
        self.self_attn = AnyVariateAttention(config)
        
        # GLU instead of standard FFN
        self.mlp = DelphyneGLU(config)
        
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
            attention_mask: Attention mask
            output_attentions: Whether to return attention weights
            
        Returns:
            Tuple of (output, attention_weights)
        """
        # Pre-normalization for attention
        normed_hidden_states = self.input_layernorm(hidden_states)
        
        # Self-attention with residual connection
        attn_outputs = self.self_attn(
            normed_hidden_states,
            variate_ids=variate_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions
        )
        
        attn_output = attn_outputs[0]
        hidden_states = hidden_states + attn_output  # Residual connection
        
        # Pre-normalization for MLP
        normed_hidden_states = self.post_attention_layernorm(hidden_states)
        
        # GLU with residual connection
        mlp_output = self.mlp(normed_hidden_states)
        hidden_states = hidden_states + mlp_output  # Residual connection
        
        outputs = (hidden_states,)
        if output_attentions:
            outputs = outputs + (attn_outputs[1],)
        
        return outputs


class DelphyneEncoder(nn.Module):
    """
    Multi-layer transformer encoder for Delphyne
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Stack of transformer layers
        self.layers = nn.ModuleList([
            DelphyneLayer(config) for _ in range(config.num_layers)
        ])
        
        # Final layer normalization
        self.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        variate_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False
    ) -> Tuple[torch.Tensor, ...]:
        """
        Args:
            hidden_states: Input embeddings [batch_size, seq_len, hidden_size]
            variate_ids: Variate identifiers [batch_size, seq_len]
            attention_mask: Attention mask
            output_attentions: Whether to return all attention weights
            output_hidden_states: Whether to return all hidden states
            
        Returns:
            Tuple containing:
            - last_hidden_state: Final layer output
            - hidden_states: All layer outputs (if output_hidden_states=True)
            - attentions: All attention weights (if output_attentions=True)
        """
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        
        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            
            layer_outputs = layer(
                hidden_states,
                variate_ids=variate_ids,
                attention_mask=attention_mask,
                output_attentions=output_attentions
            )
            
            hidden_states = layer_outputs[0]
            
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)
        
        # Apply final layer normalization
        hidden_states = self.final_layer_norm(hidden_states)
        
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        
        # Prepare outputs
        outputs = (hidden_states,)
        if output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if output_attentions:
            outputs = outputs + (all_attentions,)
        
        return outputs
