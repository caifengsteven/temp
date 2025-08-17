"""
Evaluator for CUBIC framework
Comprehensive evaluation including financial metrics and confidence-guided trading
"""

import torch
import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

from ..models.cubic_model import CUBICModel
from ..utils.confidence_measures import ConfidenceGuidedTrading
from .metrics import FinancialMetrics
from ..utils.config_manager import ConfigManager

logger = logging.getLogger(__name__)


class CUBICEvaluator:
    """
    Comprehensive evaluator for CUBIC models
    """
    
    def __init__(self, model: CUBICModel, config_path: str = "config.yaml"):
        """
        Initialize CUBIC Evaluator
        
        Args:
            model: Trained CUBIC model
            config_path: Path to configuration file
        """
        self.model = model
        self.config = ConfigManager(config_path)
        self.evaluation_config = self.config.get('evaluation', {})
        self.trading_config = self.config.get('trading', {})
        
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Initialize metrics calculator
        transaction_cost = self.trading_config.get('transaction_cost', 0.001)
        self.metrics_calculator = FinancialMetrics(transaction_cost)
        
        # Initialize confidence-guided trading
        confidence_thresholds = self.trading_config.get('confidence_thresholds', {})
        position_sizes = self.trading_config.get('position_sizes', {})
        self.confidence_trading = ConfidenceGuidedTrading(confidence_thresholds, position_sizes)
        
        # Output directory
        self.results_dir = self.config.get('output.results_dir', 'results')
        self.plots_dir = self.config.get('output.plots_dir', 'plots')
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)
        
        logger.info("CUBIC Evaluator initialized")
    
    def evaluate_model(self, test_loader: DataLoader, 
                      confidence_type: str = "mean") -> Dict[str, any]:
        """
        Comprehensive model evaluation
        
        Args:
            test_loader: Test data loader
            confidence_type: Type of confidence to use ("mean" or "trend")
            
        Returns:
            Dictionary with evaluation results
        """
        logger.info("Starting comprehensive model evaluation...")
        
        self.model.eval()
        
        all_predictions = []
        all_targets = []
        all_probabilities = []
        all_confidence_scores = []
        all_trading_signals = []
        
        with torch.no_grad():
            for features, targets in test_loader:
                features = features.to(self.device)
                targets = targets.to(self.device)
                
                # Get predictions and confidence
                probabilities, confidence_dict = self.model.forward(features, return_confidence=True)
                predictions = self.model.predict_values(features)
                
                # Generate trading signals
                trading_signals = self.confidence_trading.generate_trading_signals(
                    probabilities, confidence_type
                )
                
                # Store results
                all_predictions.append(predictions.cpu().numpy())
                all_targets.append(targets.cpu().numpy())
                all_probabilities.append(probabilities.cpu().numpy())
                
                if confidence_type == "mean":
                    confidence_scores = confidence_dict['mean_confidence']
                else:
                    confidence_scores = confidence_dict['trend_confidence']
                
                all_confidence_scores.append(confidence_scores.cpu().numpy())
                all_trading_signals.append({
                    'position_sizes': trading_signals['position_sizes'].cpu().numpy(),
                    'should_trade': trading_signals['should_trade'].cpu().numpy()
                })
        
        # Concatenate all results
        predictions = np.concatenate(all_predictions)
        targets = np.concatenate(all_targets)
        confidence_scores = np.concatenate(all_confidence_scores)
        
        # Combine trading signals
        position_sizes = np.concatenate([signals['position_sizes'] for signals in all_trading_signals])
        should_trade = np.concatenate([signals['should_trade'] for signals in all_trading_signals])
        
        # Calculate basic metrics
        basic_metrics = self.metrics_calculator.calculate_all_metrics(predictions, targets)
        
        # Calculate confidence-guided trading metrics
        trading_metrics = self.metrics_calculator.calculate_all_metrics(
            predictions, targets, position_sizes
        )
        
        # Calculate confidence statistics
        confidence_stats = self._calculate_confidence_statistics(confidence_scores, targets, predictions)
        
        # Evaluation results
        results = {
            'basic_metrics': basic_metrics,
            'trading_metrics': trading_metrics,
            'confidence_stats': confidence_stats,
            'predictions': predictions,
            'targets': targets,
            'confidence_scores': confidence_scores,
            'position_sizes': position_sizes,
            'should_trade': should_trade,
            'num_samples': len(predictions),
            'num_trades': np.sum(should_trade)
        }
        
        logger.info("Model evaluation completed")
        return results
    
    def _calculate_confidence_statistics(self, confidence_scores: np.ndarray, 
                                       targets: np.ndarray, 
                                       predictions: np.ndarray) -> Dict[str, float]:
        """
        Calculate confidence-related statistics
        
        Args:
            confidence_scores: Array of confidence scores
            targets: Actual values
            predictions: Predicted values
            
        Returns:
            Dictionary with confidence statistics
        """
        # Basic confidence statistics
        stats = {
            'mean_confidence': np.mean(confidence_scores),
            'std_confidence': np.std(confidence_scores),
            'min_confidence': np.min(confidence_scores),
            'max_confidence': np.max(confidence_scores)
        }
        
        # Confidence vs accuracy correlation
        prediction_errors = np.abs(predictions - targets)
        confidence_error_corr = np.corrcoef(confidence_scores, prediction_errors)[0, 1]
        stats['confidence_error_correlation'] = confidence_error_corr
        
        # High confidence performance
        high_conf_threshold = np.percentile(confidence_scores, 75)
        high_conf_mask = confidence_scores >= high_conf_threshold
        
        if np.sum(high_conf_mask) > 0:
            high_conf_ic = self.metrics_calculator.calculate_ic(
                predictions[high_conf_mask], targets[high_conf_mask]
            )
            high_conf_da = self.metrics_calculator.calculate_direction_accuracy(
                predictions[high_conf_mask], targets[high_conf_mask]
            )
            stats['high_confidence_ic'] = high_conf_ic
            stats['high_confidence_da'] = high_conf_da
        
        return stats
