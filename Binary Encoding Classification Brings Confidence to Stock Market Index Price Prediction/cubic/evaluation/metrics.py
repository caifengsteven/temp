"""
Financial metrics for CUBIC framework evaluation
Implements IC, ICLR, DA, SR, AR as mentioned in the paper
"""

import numpy as np
import pandas as pd
import torch
from scipy import stats
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class FinancialMetrics:
    """
    Calculate financial performance metrics for CUBIC framework
    """
    
    def __init__(self, transaction_cost: float = 0.001):
        """
        Initialize Financial Metrics
        
        Args:
            transaction_cost: Transaction cost per trade (default: 0.1%)
        """
        self.transaction_cost = transaction_cost
    
    def calculate_ic(self, predictions: np.ndarray, actual: np.ndarray) -> float:
        """
        Calculate Information Coefficient (IC)
        Average daily Pearson correlation between predicted and actual returns
        
        Args:
            predictions: Predicted returns
            actual: Actual returns
            
        Returns:
            Information Coefficient
        """
        if len(predictions) != len(actual):
            raise ValueError("Predictions and actual values must have same length")
        
        # Remove NaN values
        mask = ~(np.isnan(predictions) | np.isnan(actual))
        if np.sum(mask) < 2:
            return 0.0
        
        correlation, _ = stats.pearsonr(predictions[mask], actual[mask])
        return correlation if not np.isnan(correlation) else 0.0
    
    def calculate_iclr(self, predictions: np.ndarray, actual: np.ndarray, 
                      window_size: int = 20) -> float:
        """
        Calculate Information Coefficient to Loss Ratio (ICLR)
        IC normalized by its standard deviation
        
        Args:
            predictions: Predicted returns
            actual: Actual returns
            window_size: Window size for rolling IC calculation
            
        Returns:
            Information Coefficient to Loss Ratio
        """
        if len(predictions) < window_size:
            return self.calculate_ic(predictions, actual)
        
        # Calculate rolling IC
        rolling_ics = []
        for i in range(window_size, len(predictions) + 1):
            window_pred = predictions[i-window_size:i]
            window_actual = actual[i-window_size:i]
            ic = self.calculate_ic(window_pred, window_actual)
            rolling_ics.append(ic)
        
        rolling_ics = np.array(rolling_ics)
        
        # Calculate ICLR
        mean_ic = np.mean(rolling_ics)
        std_ic = np.std(rolling_ics)
        
        if std_ic == 0:
            return mean_ic
        
        return mean_ic / std_ic
    
    def calculate_direction_accuracy(self, predictions: np.ndarray, actual: np.ndarray) -> float:
        """
        Calculate Direction Accuracy (DA)
        Percentage of correct directional predictions
        
        Args:
            predictions: Predicted returns
            actual: Actual returns
            
        Returns:
            Direction accuracy (0-1)
        """
        if len(predictions) != len(actual):
            raise ValueError("Predictions and actual values must have same length")
        
        # Remove NaN values
        mask = ~(np.isnan(predictions) | np.isnan(actual))
        if np.sum(mask) == 0:
            return 0.0
        
        pred_direction = np.sign(predictions[mask])
        actual_direction = np.sign(actual[mask])
        
        correct_predictions = (pred_direction == actual_direction)
        accuracy = np.mean(correct_predictions)
        
        return accuracy
    
    def calculate_returns(self, predictions: np.ndarray, actual: np.ndarray, 
                         position_sizes: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Calculate trading returns based on predictions
        
        Args:
            predictions: Predicted returns
            actual: Actual returns
            position_sizes: Position sizes for each prediction (optional)
            
        Returns:
            Array of trading returns
        """
        if position_sizes is None:
            # Simple long-short strategy
            position_sizes = np.sign(predictions)
        
        # Calculate gross returns
        gross_returns = position_sizes * actual
        
        # Calculate transaction costs
        position_changes = np.abs(np.diff(np.concatenate([[0], position_sizes])))
        transaction_costs = position_changes * self.transaction_cost
        
        # Net returns
        net_returns = gross_returns - transaction_costs
        
        return net_returns
    
    def calculate_sharpe_ratio(self, returns: np.ndarray, risk_free_rate: float = 0.0) -> float:
        """
        Calculate Sharpe Ratio (SR)
        Risk-adjusted return measure
        
        Args:
            returns: Array of returns
            risk_free_rate: Risk-free rate (annualized)
            
        Returns:
            Sharpe ratio
        """
        if len(returns) == 0:
            return 0.0
        
        # Remove NaN values
        returns = returns[~np.isnan(returns)]
        
        if len(returns) == 0 or np.std(returns) == 0:
            return 0.0
        
        # Calculate excess returns
        excess_returns = returns - risk_free_rate / 252  # Daily risk-free rate
        
        # Annualized Sharpe ratio
        sharpe = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
        
        return sharpe
    
    def calculate_annualized_return(self, returns: np.ndarray) -> float:
        """
        Calculate Annualized Return (AR)
        
        Args:
            returns: Array of returns
            
        Returns:
            Annualized return
        """
        if len(returns) == 0:
            return 0.0
        
        # Remove NaN values
        returns = returns[~np.isnan(returns)]
        
        if len(returns) == 0:
            return 0.0
        
        # Calculate cumulative return
        cumulative_return = np.prod(1 + returns) - 1
        
        # Annualize
        trading_days = len(returns)
        annualized_return = (1 + cumulative_return) ** (252 / trading_days) - 1
        
        return annualized_return
    
    def calculate_maximum_drawdown(self, returns: np.ndarray) -> float:
        """
        Calculate Maximum Drawdown
        
        Args:
            returns: Array of returns
            
        Returns:
            Maximum drawdown
        """
        if len(returns) == 0:
            return 0.0
        
        # Calculate cumulative returns
        cumulative_returns = np.cumprod(1 + returns)
        
        # Calculate running maximum
        running_max = np.maximum.accumulate(cumulative_returns)
        
        # Calculate drawdown
        drawdown = (cumulative_returns - running_max) / running_max
        
        # Maximum drawdown
        max_drawdown = np.min(drawdown)
        
        return abs(max_drawdown)
    
    def calculate_calmar_ratio(self, returns: np.ndarray) -> float:
        """
        Calculate Calmar Ratio (Annualized Return / Maximum Drawdown)
        
        Args:
            returns: Array of returns
            
        Returns:
            Calmar ratio
        """
        annualized_return = self.calculate_annualized_return(returns)
        max_drawdown = self.calculate_maximum_drawdown(returns)
        
        if max_drawdown == 0:
            return annualized_return
        
        return annualized_return / max_drawdown
    
    def calculate_hit_rate(self, returns: np.ndarray) -> float:
        """
        Calculate Hit Rate (percentage of profitable trades)
        
        Args:
            returns: Array of returns
            
        Returns:
            Hit rate (0-1)
        """
        if len(returns) == 0:
            return 0.0
        
        # Remove NaN values
        returns = returns[~np.isnan(returns)]
        
        if len(returns) == 0:
            return 0.0
        
        profitable_trades = returns > 0
        hit_rate = np.mean(profitable_trades)
        
        return hit_rate
    
    def calculate_all_metrics(self, predictions: np.ndarray, actual: np.ndarray,
                            position_sizes: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Calculate all financial metrics
        
        Args:
            predictions: Predicted returns
            actual: Actual returns
            position_sizes: Position sizes for each prediction (optional)
            
        Returns:
            Dictionary with all metrics
        """
        # Calculate trading returns
        trading_returns = self.calculate_returns(predictions, actual, position_sizes)
        
        # Calculate all metrics
        metrics = {
            'IC': self.calculate_ic(predictions, actual),
            'ICLR': self.calculate_iclr(predictions, actual),
            'DA': self.calculate_direction_accuracy(predictions, actual),
            'SR': self.calculate_sharpe_ratio(trading_returns),
            'AR': self.calculate_annualized_return(trading_returns),
            'Max_Drawdown': self.calculate_maximum_drawdown(trading_returns),
            'Calmar_Ratio': self.calculate_calmar_ratio(trading_returns),
            'Hit_Rate': self.calculate_hit_rate(trading_returns),
            'Total_Return': np.sum(trading_returns),
            'Volatility': np.std(trading_returns) * np.sqrt(252)  # Annualized
        }
        
        return metrics
    
    def create_performance_summary(self, metrics: Dict[str, float]) -> str:
        """
        Create a formatted performance summary
        
        Args:
            metrics: Dictionary of calculated metrics
            
        Returns:
            Formatted summary string
        """
        summary = f"""
Performance Summary:
==================
Information Coefficient (IC): {metrics['IC']:.4f}
IC to Loss Ratio (ICLR): {metrics['ICLR']:.4f}
Direction Accuracy (DA): {metrics['DA']:.4f}
Sharpe Ratio (SR): {metrics['SR']:.4f}
Annualized Return (AR): {metrics['AR']:.4f}
Maximum Drawdown: {metrics['Max_Drawdown']:.4f}
Calmar Ratio: {metrics['Calmar_Ratio']:.4f}
Hit Rate: {metrics['Hit_Rate']:.4f}
Total Return: {metrics['Total_Return']:.4f}
Annualized Volatility: {metrics['Volatility']:.4f}
"""
        return summary
