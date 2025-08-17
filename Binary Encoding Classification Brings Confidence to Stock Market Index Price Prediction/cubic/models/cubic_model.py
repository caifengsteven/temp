"""
CUBIC Model - Complete implementation of the CUBIC framework
"""

import torch
import torch.nn as nn
import logging
from typing import Dict, Tuple, Optional, Union

from .fusion_module import FusionInLatentSpace
from .backbones import LSTMBackbone, TransformerBackbone, MLPBackbone
from ..utils.binary_encoder import BinaryEncoder
from ..utils.confidence_measures import ConfidenceMeasures, ConfidenceGuidedLoss

logger = logging.getLogger(__name__)


class BinaryClassificationHead(nn.Module):
    """
    Binary classification head for CUBIC framework
    Outputs probabilities for each binary digit
    """
    
    def __init__(self, input_dim: int, precision_bits: int = 15, dropout: float = 0.1):
        """
        Initialize Binary Classification Head
        
        Args:
            input_dim: Input feature dimension
            precision_bits: Number of binary digits
            dropout: Dropout rate
        """
        super().__init__()
        self.input_dim = input_dim
        self.precision_bits = precision_bits
        
        # Create classification layers for each binary digit
        self.classifiers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, input_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(input_dim // 2, 2)  # Binary classification
            )
            for _ in range(precision_bits)
        ])
        
        logger.info(f"Binary classification head initialized with {precision_bits} classifiers")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for binary classification head
        
        Args:
            x: Input features (batch_size, input_dim)
            
        Returns:
            Probabilities for each binary digit (batch_size, precision_bits, 2)
        """
        batch_size = x.shape[0]
        outputs = []
        
        for classifier in self.classifiers:
            logits = classifier(x)  # (batch_size, 2)
            probs = torch.softmax(logits, dim=-1)
            outputs.append(probs)
        
        # Stack outputs
        output = torch.stack(outputs, dim=1)  # (batch_size, precision_bits, 2)
        
        return output


class CUBICModel(nn.Module):
    """
    Complete CUBIC model implementation
    """
    
    def __init__(self, 
                 input_dim: int,
                 n_stocks: int,
                 backbone_type: str = "lstm",
                 backbone_config: Dict = None,
                 fusion_config: Dict = None,
                 binary_config: Dict = None,
                 confidence_config: Dict = None):
        """
        Initialize CUBIC Model
        
        Args:
            input_dim: Number of features per stock
            n_stocks: Number of constituent stocks
            backbone_type: Type of backbone ("lstm", "transformer", "mlp")
            backbone_config: Configuration for backbone model
            fusion_config: Configuration for fusion module
            binary_config: Configuration for binary encoding
            confidence_config: Configuration for confidence measures
        """
        super().__init__()
        
        # Default configurations
        backbone_config = backbone_config or {}
        fusion_config = fusion_config or {}
        binary_config = binary_config or {}
        confidence_config = confidence_config or {}
        
        self.input_dim = input_dim
        self.n_stocks = n_stocks
        self.backbone_type = backbone_type
        
        # Binary encoding configuration
        self.precision_bits = binary_config.get('precision_bits', 15)
        self.value_range = binary_config.get('value_range', (-1, 1))
        
        # Initialize binary encoder
        self.binary_encoder = BinaryEncoder(self.precision_bits, self.value_range)
        
        # Initialize fusion in latent space
        embedding_dim = fusion_config.get('embedding_dim', 32)
        pooling_type = fusion_config.get('pooling_type', 'multi_head')
        
        self.fusion_module = FusionInLatentSpace(
            input_dim=input_dim,
            embedding_dim=embedding_dim,
            pooling_type=pooling_type,
            dropout=fusion_config.get('dropout', 0.1)
        )
        
        # Initialize backbone model
        backbone_input_dim = self.fusion_module.output_dim
        
        if backbone_type == "lstm":
            self.backbone = LSTMBackbone(
                input_dim=backbone_input_dim,
                hidden_size=backbone_config.get('hidden_size', 128),
                num_layers=backbone_config.get('num_layers', 2),
                dropout=backbone_config.get('dropout', 0.1),
                bidirectional=backbone_config.get('bidirectional', False)
            )
        elif backbone_type == "transformer":
            self.backbone = TransformerBackbone(
                input_dim=backbone_input_dim,
                d_model=backbone_config.get('d_model', 64),
                nhead=backbone_config.get('nhead', 8),
                num_layers=backbone_config.get('num_layers', 1),
                dim_feedforward=backbone_config.get('dim_feedforward', 256),
                dropout=backbone_config.get('dropout', 0.1)
            )
        elif backbone_type == "mlp":
            self.backbone = MLPBackbone(
                input_dim=backbone_input_dim,
                hidden_layers=backbone_config.get('hidden_layers', [256, 128, 64]),
                dropout=backbone_config.get('dropout', 0.1),
                activation=backbone_config.get('activation', 'relu')
            )
        else:
            raise ValueError(f"Unknown backbone type: {backbone_type}")
        
        # Initialize binary classification head
        self.classification_head = BinaryClassificationHead(
            input_dim=self.backbone.output_dim,
            precision_bits=self.precision_bits,
            dropout=backbone_config.get('dropout', 0.1)
        )
        
        # Initialize confidence measures
        self.confidence_measures = ConfidenceMeasures(self.precision_bits)
        
        # Initialize confidence-guided loss
        confidence_weight = confidence_config.get('weight', 0.1)
        self.confidence_loss = ConfidenceGuidedLoss(self.precision_bits, confidence_weight)
        
        logger.info(f"CUBIC model initialized with {backbone_type} backbone, "
                   f"{self.precision_bits} binary digits, fusion pooling: {pooling_type}")
    
    def forward(self, x: torch.Tensor, return_confidence: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict]]:
        """
        Forward pass for CUBIC model
        
        Args:
            x: Input tensor (batch_size, sequence_length, n_stocks, input_dim)
            return_confidence: Whether to return confidence measures
            
        Returns:
            Binary classification probabilities or tuple with confidence measures
        """
        # Fusion in latent space
        fused_features = self.fusion_module(x)  # (batch_size, sequence_length, fusion_output_dim)
        
        # Backbone processing
        backbone_output = self.backbone(fused_features)  # (batch_size, backbone_output_dim)
        
        # Binary classification
        probabilities = self.classification_head(backbone_output)  # (batch_size, precision_bits, 2)
        
        if return_confidence:
            # Calculate confidence measures
            mean_confidence = self.confidence_measures.calculate_geometric_mean_confidence(probabilities)
            trend_confidence = self.confidence_measures.calculate_trend_confidence(probabilities)
            entropy_confidence = self.confidence_measures.calculate_entropy_based_confidence(probabilities)
            
            confidence_dict = {
                'mean_confidence': mean_confidence,
                'trend_confidence': trend_confidence,
                'entropy_confidence': entropy_confidence,
                'probabilities': probabilities
            }
            
            return probabilities, confidence_dict
        
        return probabilities
    
    def predict_values(self, x: torch.Tensor, use_argmax: bool = True) -> torch.Tensor:
        """
        Predict continuous values from input
        
        Args:
            x: Input tensor
            use_argmax: Whether to use argmax for binary decisions
            
        Returns:
            Predicted continuous values
        """
        probabilities = self.forward(x)
        predicted_values = self.binary_encoder.reconstruct_from_probabilities(probabilities, use_argmax)
        return predicted_values
    
    def calculate_loss(self, x: torch.Tensor, targets: torch.Tensor, 
                      confidence_type: str = "mean") -> Tuple[torch.Tensor, Dict]:
        """
        Calculate loss including confidence regularization
        
        Args:
            x: Input tensor
            targets: Target values (continuous)
            confidence_type: Type of confidence to use
            
        Returns:
            Tuple of (total_loss, loss_components)
        """
        # Forward pass
        probabilities = self.forward(x)
        
        # Convert targets to binary
        binary_targets = self.binary_encoder.create_binary_targets(targets)
        
        # Calculate confidence-guided loss
        total_loss, loss_components = self.confidence_loss(probabilities, binary_targets, confidence_type)
        
        # Add reconstruction error for monitoring
        predicted_values = self.binary_encoder.reconstruct_from_probabilities(probabilities)
        reconstruction_error = self.binary_encoder.calculate_reconstruction_error(targets, predicted_values)
        loss_components['reconstruction_error'] = reconstruction_error
        
        return total_loss, loss_components
    
    def get_model_info(self) -> Dict:
        """
        Get model information
        
        Returns:
            Dictionary with model information
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'backbone_type': self.backbone_type,
            'precision_bits': self.precision_bits,
            'value_range': self.value_range,
            'input_dim': self.input_dim,
            'n_stocks': self.n_stocks,
            'fusion_output_dim': self.fusion_module.output_dim,
            'backbone_output_dim': self.backbone.output_dim,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params
        }
    
    def freeze_backbone(self):
        """Freeze backbone parameters"""
        for param in self.backbone.parameters():
            param.requires_grad = False
        logger.info("Backbone parameters frozen")
    
    def unfreeze_backbone(self):
        """Unfreeze backbone parameters"""
        for param in self.backbone.parameters():
            param.requires_grad = True
        logger.info("Backbone parameters unfrozen")
    
    def freeze_fusion(self):
        """Freeze fusion module parameters"""
        for param in self.fusion_module.parameters():
            param.requires_grad = False
        logger.info("Fusion module parameters frozen")
    
    def unfreeze_fusion(self):
        """Unfreeze fusion module parameters"""
        for param in self.fusion_module.parameters():
            param.requires_grad = True
        logger.info("Fusion module parameters unfrozen")
