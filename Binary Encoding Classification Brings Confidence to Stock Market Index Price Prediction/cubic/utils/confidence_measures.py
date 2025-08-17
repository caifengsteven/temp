"""
Confidence Measures for CUBIC framework
Implements geometric confidence measures and confidence-guided mechanisms
"""

import torch
import torch.nn as nn
import numpy as np
import logging
from typing import Tuple, Dict, Optional

logger = logging.getLogger(__name__)


class ConfidenceMeasures:
    """
    Implements confidence measures for binary classification predictions
    """
    
    def __init__(self, precision_bits: int = 15):
        """
        Initialize Confidence Measures
        
        Args:
            precision_bits: Number of binary digits
        """
        self.precision_bits = precision_bits
        
    def calculate_geometric_mean_confidence(self, probabilities: torch.Tensor) -> torch.Tensor:
        """
        Calculate geometric mean confidence across all binary digits
        GC_mean = (∏(k=0 to K) p(γ̂_k))^(1/K)
        
        Args:
            probabilities: Probabilities for each binary digit (batch_size, precision_bits, 2)
            
        Returns:
            Geometric mean confidence for each sample
        """
        batch_size = probabilities.shape[0]
        
        # Get the probability of the predicted class for each bit
        predicted_bits = torch.argmax(probabilities, dim=-1)  # (batch_size, precision_bits)
        
        # Gather the probabilities of the predicted classes
        predicted_probs = torch.gather(probabilities, 2, predicted_bits.unsqueeze(-1)).squeeze(-1)
        
        # Calculate geometric mean
        # Use log-space for numerical stability: exp(mean(log(probs)))
        log_probs = torch.log(predicted_probs + 1e-8)  # Add small epsilon to avoid log(0)
        geometric_mean = torch.exp(torch.mean(log_probs, dim=1))
        
        return geometric_mean
    
    def calculate_trend_confidence(self, probabilities: torch.Tensor) -> torch.Tensor:
        """
        Calculate trend confidence based on the most significant bit (first bit)
        GC_trend = p(γ̂_0)
        
        Args:
            probabilities: Probabilities for each binary digit (batch_size, precision_bits, 2)
            
        Returns:
            Trend confidence for each sample
        """
        # Get the probability of the predicted class for the first bit (most significant)
        first_bit_probs = probabilities[:, 0, :]  # (batch_size, 2)
        predicted_first_bit = torch.argmax(first_bit_probs, dim=1)  # (batch_size,)
        
        # Get the probability of the predicted class
        trend_confidence = torch.gather(first_bit_probs, 1, predicted_first_bit.unsqueeze(-1)).squeeze(-1)
        
        return trend_confidence
    
    def calculate_position_weighted_confidence(self, probabilities: torch.Tensor, 
                                             weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Calculate position-weighted confidence
        
        Args:
            probabilities: Probabilities for each binary digit
            weights: Position weights (if None, uses exponential weights)
            
        Returns:
            Position-weighted confidence
        """
        if weights is None:
            # Use exponential weights based on bit significance
            weights = torch.tensor([2 ** (-k) for k in range(self.precision_bits)], 
                                 device=probabilities.device)
        
        # Get predicted probabilities
        predicted_bits = torch.argmax(probabilities, dim=-1)
        predicted_probs = torch.gather(probabilities, 2, predicted_bits.unsqueeze(-1)).squeeze(-1)
        
        # Calculate weighted confidence
        weighted_confidence = torch.sum(predicted_probs * weights.unsqueeze(0), dim=1) / torch.sum(weights)
        
        return weighted_confidence
    
    def calculate_entropy_based_confidence(self, probabilities: torch.Tensor) -> torch.Tensor:
        """
        Calculate confidence based on entropy (lower entropy = higher confidence)
        
        Args:
            probabilities: Probabilities for each binary digit
            
        Returns:
            Entropy-based confidence (1 - normalized_entropy)
        """
        # Calculate entropy for each bit
        entropy = -torch.sum(probabilities * torch.log(probabilities + 1e-8), dim=-1)
        
        # Average entropy across all bits
        avg_entropy = torch.mean(entropy, dim=1)
        
        # Normalize entropy (max entropy for binary classification is log(2))
        max_entropy = np.log(2)
        normalized_entropy = avg_entropy / max_entropy
        
        # Convert to confidence (higher entropy = lower confidence)
        confidence = 1 - normalized_entropy
        
        return confidence


class ConfidenceGuidedLoss(nn.Module):
    """
    Confidence-guided regularization loss for CUBIC framework
    """
    
    def __init__(self, precision_bits: int = 15, confidence_weight: float = 0.1):
        """
        Initialize Confidence-Guided Loss
        
        Args:
            precision_bits: Number of binary digits
            confidence_weight: Weight for confidence regularization term
        """
        super().__init__()
        self.precision_bits = precision_bits
        self.confidence_weight = confidence_weight
        self.confidence_measures = ConfidenceMeasures(precision_bits)
        
    def forward(self, probabilities: torch.Tensor, targets: torch.Tensor, 
                confidence_type: str = "mean") -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Calculate confidence-guided loss
        
        Args:
            probabilities: Model output probabilities (batch_size, precision_bits, 2)
            targets: Binary targets (batch_size, precision_bits)
            confidence_type: Type of confidence to use ("mean" or "trend")
            
        Returns:
            Tuple of (total_loss, loss_components)
        """
        batch_size = probabilities.shape[0]
        
        # Calculate cross-entropy loss for each bit
        ce_losses = []
        for k in range(self.precision_bits):
            ce_loss = nn.functional.cross_entropy(probabilities[:, k, :], targets[:, k], reduction='none')
            ce_losses.append(ce_loss)
        
        ce_losses = torch.stack(ce_losses, dim=1)  # (batch_size, precision_bits)
        
        # Calculate position weights (more weight to significant bits)
        position_weights = torch.tensor([2 ** (-k) for k in range(self.precision_bits)], 
                                      device=probabilities.device)
        
        # Weighted cross-entropy loss
        weighted_ce_loss = torch.sum(ce_losses * position_weights.unsqueeze(0), dim=1)
        mean_ce_loss = torch.mean(weighted_ce_loss)
        
        # Calculate confidence
        if confidence_type == "mean":
            confidence = self.confidence_measures.calculate_geometric_mean_confidence(probabilities)
        elif confidence_type == "trend":
            confidence = self.confidence_measures.calculate_trend_confidence(probabilities)
        else:
            raise ValueError(f"Unknown confidence type: {confidence_type}")
        
        # Calculate confidence regularization loss
        # L_conf = (1 - 2 * I[p(γ_0) > p(1-γ_0)]) * GC
        first_bit_probs = probabilities[:, 0, :]  # (batch_size, 2)
        correct_trend = (first_bit_probs[:, 1] > first_bit_probs[:, 0]).float()  # 1 if predicting positive trend
        
        # Indicator function: 1 if trend prediction is correct, -1 if incorrect
        trend_indicator = 2 * correct_trend - 1
        
        # Confidence regularization: encourage high confidence for correct trends, low for incorrect
        confidence_reg = torch.mean((1 - trend_indicator) * confidence)
        
        # Total loss
        total_loss = mean_ce_loss + self.confidence_weight * confidence_reg
        
        # Loss components for monitoring
        loss_components = {
            'cross_entropy': mean_ce_loss,
            'confidence_regularization': confidence_reg,
            'total_loss': total_loss,
            'mean_confidence': torch.mean(confidence),
            'trend_accuracy': torch.mean(correct_trend)
        }
        
        return total_loss, loss_components


class ConfidenceGuidedTrading:
    """
    Implements confidence-guided trading strategies
    """
    
    def __init__(self, confidence_thresholds: Dict[str, float] = None,
                 position_sizes: Dict[str, float] = None):
        """
        Initialize Confidence-Guided Trading
        
        Args:
            confidence_thresholds: Thresholds for different confidence levels
            position_sizes: Position sizes for different confidence levels
        """
        self.confidence_thresholds = confidence_thresholds or {
            'low': 0.5, 'medium': 0.7, 'high': 0.9
        }
        self.position_sizes = position_sizes or {
            'low_confidence': 0.5, 'medium_confidence': 0.75, 'high_confidence': 1.0
        }
        
        self.confidence_measures = ConfidenceMeasures()
    
    def get_position_size(self, confidence: float, predicted_direction: int) -> float:
        """
        Get position size based on confidence level
        
        Args:
            confidence: Confidence score [0, 1]
            predicted_direction: Predicted direction (1 for up, 0 for down)
            
        Returns:
            Position size (positive for long, negative for short)
        """
        # Determine confidence level
        if confidence >= self.confidence_thresholds['high']:
            position_size = self.position_sizes['high_confidence']
        elif confidence >= self.confidence_thresholds['medium']:
            position_size = self.position_sizes['medium_confidence']
        elif confidence >= self.confidence_thresholds['low']:
            position_size = self.position_sizes['low_confidence']
        else:
            position_size = 0.0  # No position for very low confidence
        
        # Apply direction (positive for long, negative for short)
        if predicted_direction == 0:  # Predicted down
            position_size = -position_size
        
        return position_size
    
    def generate_trading_signals(self, probabilities: torch.Tensor, 
                               confidence_type: str = "mean") -> Dict[str, torch.Tensor]:
        """
        Generate trading signals based on predictions and confidence
        
        Args:
            probabilities: Model output probabilities
            confidence_type: Type of confidence to use
            
        Returns:
            Dictionary with trading signals
        """
        batch_size = probabilities.shape[0]
        
        # Calculate confidence
        if confidence_type == "mean":
            confidence = self.confidence_measures.calculate_geometric_mean_confidence(probabilities)
        elif confidence_type == "trend":
            confidence = self.confidence_measures.calculate_trend_confidence(probabilities)
        else:
            raise ValueError(f"Unknown confidence type: {confidence_type}")
        
        # Get predicted direction from first bit (most significant)
        first_bit_probs = probabilities[:, 0, :]
        predicted_direction = torch.argmax(first_bit_probs, dim=1)
        
        # Generate position sizes
        position_sizes = torch.zeros(batch_size)
        for i in range(batch_size):
            position_sizes[i] = self.get_position_size(
                confidence[i].item(), 
                predicted_direction[i].item()
            )
        
        return {
            'confidence': confidence,
            'predicted_direction': predicted_direction,
            'position_sizes': position_sizes,
            'should_trade': torch.abs(position_sizes) > 0
        }
    
    def calculate_trading_performance(self, signals: Dict[str, torch.Tensor], 
                                    actual_returns: torch.Tensor,
                                    transaction_cost: float = 0.001) -> Dict[str, float]:
        """
        Calculate trading performance metrics
        
        Args:
            signals: Trading signals from generate_trading_signals
            actual_returns: Actual market returns
            transaction_cost: Transaction cost per trade
            
        Returns:
            Dictionary with performance metrics
        """
        position_sizes = signals['position_sizes']
        should_trade = signals['should_trade']
        
        # Calculate returns
        trading_returns = position_sizes * actual_returns
        
        # Apply transaction costs
        transaction_costs = should_trade.float() * transaction_cost
        net_returns = trading_returns - transaction_costs
        
        # Calculate performance metrics
        total_return = torch.sum(net_returns).item()
        mean_return = torch.mean(net_returns).item()
        std_return = torch.std(net_returns).item()
        sharpe_ratio = mean_return / (std_return + 1e-8) * np.sqrt(252)  # Annualized
        
        # Hit rate (percentage of profitable trades)
        profitable_trades = (net_returns > 0).float()
        hit_rate = torch.mean(profitable_trades).item()
        
        # Maximum drawdown
        cumulative_returns = torch.cumsum(net_returns, dim=0)
        running_max = torch.cummax(cumulative_returns, dim=0)[0]
        drawdown = running_max - cumulative_returns
        max_drawdown = torch.max(drawdown).item()
        
        return {
            'total_return': total_return,
            'mean_return': mean_return,
            'std_return': std_return,
            'sharpe_ratio': sharpe_ratio,
            'hit_rate': hit_rate,
            'max_drawdown': max_drawdown,
            'num_trades': torch.sum(should_trade).item()
        }
