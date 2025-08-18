"""
Complete Delphyne model implementation
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Tuple, Union
import numpy as np

from ..config import DelphyneConfig
from .embeddings import DelphyneEmbeddings
from .layers import DelphyneEncoder
from .output import ForecastHead
from .patching import TimeSeriesPatcher, TimeSeriesNormalizer


class DelphyneModel(nn.Module):
    """
    Complete Delphyne model for time series forecasting.
    
    Architecture:
    1. Time series patching and normalization
    2. Embedding layer with variate IDs and masking
    3. Multi-layer transformer encoder with any-variate attention
    4. Mixture of Student-T output distribution
    
    Based on the paper: "DELPHYNE: A PRE-TRAINED MODEL FOR GENERAL AND FINANCIAL TIMESERIES"
    """
    
    def __init__(self, config: DelphyneConfig):
        super().__init__()
        self.config = config
        
        # Data preprocessing components
        self.patcher = TimeSeriesPatcher(patch_size=config.patch_size)
        self.normalizer = TimeSeriesNormalizer()
        
        # Core model components
        self.embeddings = DelphyneEmbeddings(config)
        self.encoder = DelphyneEncoder(config)
        self.forecast_head = ForecastHead(config)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """Initialize model weights according to the paper specifications."""
        if isinstance(module, nn.Linear):
            # Use normal initialization with std from config
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
    
    def forward(
        self,
        time_series: torch.Tensor,
        variate_ids: Optional[torch.Tensor] = None,
        missing_mask: Optional[torch.Tensor] = None,
        forecast_mask: Optional[torch.Tensor] = None,
        targets: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        normalize: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True
    ) -> Union[Dict[str, Any], Tuple]:
        """
        Forward pass through the Delphyne model.
        
        Args:
            time_series: Input time series data
                - Shape: [batch_size, seq_len] for univariate
                - Shape: [batch_size, num_variates, seq_len] for multivariate
            variate_ids: Variate identifiers [batch_size, seq_len] (optional)
            missing_mask: Binary mask for missing values [batch_size, seq_len] (optional)
            forecast_mask: Binary mask for forecast positions [batch_size, seq_len] (optional)
            targets: Target values for loss computation [batch_size, seq_len] (optional)
            attention_mask: Attention mask [batch_size, seq_len] (optional)
            normalize: Whether to apply normalization
            output_attentions: Whether to return attention weights
            output_hidden_states: Whether to return all hidden states
            return_dict: Whether to return a dictionary or tuple
            
        Returns:
            Dictionary or tuple containing model outputs
        """
        batch_size = time_series.shape[0]
        
        # Step 1: Normalize the time series data
        normalization_stats = None
        if normalize:
            time_series, normalization_stats = self.normalizer(time_series, variate_ids)
        
        # Step 2: Convert to patches
        patch_data = self.patcher(
            time_series=time_series,
            variate_ids=variate_ids,
            missing_mask=missing_mask,
            forecast_mask=forecast_mask,
            max_length=self.config.max_sequence_length
        )
        
        patches = patch_data['patches']
        patch_variate_ids = patch_data['variate_ids']
        patch_missing_mask = patch_data['missing_mask']
        patch_forecast_mask = patch_data['forecast_mask']
        patch_attention_mask = patch_data['attention_mask']
        
        # Use provided attention mask or default to patch attention mask
        if attention_mask is None:
            attention_mask = patch_attention_mask
        
        # Step 3: Embed the patches
        embeddings = self.embeddings(
            input_patches=patches,
            variate_ids=patch_variate_ids,
            missing_mask=patch_missing_mask,
            forecast_mask=patch_forecast_mask
        )
        
        # Step 4: Pass through transformer encoder
        encoder_outputs = self.encoder(
            hidden_states=embeddings,
            variate_ids=patch_variate_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states
        )
        
        last_hidden_state = encoder_outputs[0]
        
        # Step 5: Generate forecasts
        forecast_outputs = self.forecast_head(
            hidden_states=last_hidden_state,
            targets=targets,
            forecast_mask=patch_forecast_mask
        )
        
        # Prepare outputs
        if return_dict:
            outputs = {
                'distribution': forecast_outputs['distribution'],
                'mixture_params': forecast_outputs['mixture_params'],
                'last_hidden_state': last_hidden_state,
                'patch_data': patch_data,
                'normalization_stats': normalization_stats
            }
            
            if 'loss' in forecast_outputs:
                outputs['loss'] = forecast_outputs['loss']
            
            if output_hidden_states:
                outputs['hidden_states'] = encoder_outputs[1]
            
            if output_attentions:
                outputs['attentions'] = encoder_outputs[-1]
                
            return outputs
        else:
            # Return tuple format
            outputs = (forecast_outputs['distribution'], last_hidden_state)
            
            if 'loss' in forecast_outputs:
                outputs = (forecast_outputs['loss'],) + outputs
            
            if output_hidden_states:
                outputs = outputs + (encoder_outputs[1],)
            
            if output_attentions:
                outputs = outputs + (encoder_outputs[-1],)
            
            return outputs
    
    def generate_forecasts(
        self,
        time_series: torch.Tensor,
        forecast_length: int,
        num_samples: int = 1,
        variate_ids: Optional[torch.Tensor] = None,
        missing_mask: Optional[torch.Tensor] = None,
        temperature: float = 1.0
    ) -> Dict[str, torch.Tensor]:
        """
        Generate probabilistic forecasts.
        
        Args:
            time_series: Input time series data
            forecast_length: Number of steps to forecast
            num_samples: Number of samples to generate
            variate_ids: Variate identifiers (optional)
            missing_mask: Missing data mask (optional)
            temperature: Sampling temperature
            
        Returns:
            Dictionary containing forecast samples and statistics
        """
        self.eval()
        
        with torch.no_grad():
            # Create forecast mask for the desired forecast length
            batch_size = time_series.shape[0]
            seq_len = time_series.shape[-1]
            
            # For simplicity, we'll forecast the last `forecast_length` positions
            forecast_mask = torch.zeros(batch_size, seq_len, device=time_series.device)
            forecast_mask[:, -forecast_length:] = 1.0
            
            # Forward pass
            outputs = self(
                time_series=time_series,
                variate_ids=variate_ids,
                missing_mask=missing_mask,
                forecast_mask=forecast_mask,
                return_dict=True
            )
            
            distribution = outputs['distribution']
            
            # Generate samples
            if temperature != 1.0:
                # Adjust distribution for temperature sampling
                # This is a simplified approach - in practice, you might want
                # to adjust the scale parameters of the Student-T mixture
                pass
            
            # Sample from the distribution
            samples = distribution.sample((num_samples,))
            
            # Reshape samples to [num_samples, batch_size, seq_len]
            patch_data = outputs['patch_data']
            num_patches = patch_data['patches'].shape[1]
            samples = samples.view(num_samples, batch_size, num_patches)
            
            # Compute statistics
            mean_forecast = samples.mean(dim=0)
            std_forecast = samples.std(dim=0)
            
            # Compute quantiles
            quantiles = torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9], device=time_series.device)
            quantile_forecasts = torch.quantile(samples, quantiles, dim=0)
            
            return {
                'samples': samples,
                'mean': mean_forecast,
                'std': std_forecast,
                'quantiles': quantile_forecasts,
                'quantile_levels': quantiles,
                'normalization_stats': outputs['normalization_stats']
            }
    
    def get_num_parameters(self, only_trainable: bool = True) -> int:
        """Get the number of parameters in the model."""
        if only_trainable:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        else:
            return sum(p.numel() for p in self.parameters())
    
    @classmethod
    def from_config(cls, config: DelphyneConfig):
        """Create model from configuration."""
        return cls(config)
