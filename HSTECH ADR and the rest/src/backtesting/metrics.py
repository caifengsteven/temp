"""
Backtesting Metrics and Utilities

This module provides utility functions for calculating and analyzing
backtesting metrics for the HSTECH estimation system.
"""

from typing import List, Dict, Tuple, Optional
import numpy as np
import pandas as pd
from decimal import Decimal
import logging

logger = logging.getLogger(__name__)


def calculate_mse(actual: List[float], predicted: List[float]) -> float:
    """Calculate Mean Squared Error."""
    if len(actual) != len(predicted):
        raise ValueError("Actual and predicted lists must have same length")
    
    return np.mean([(a - p) ** 2 for a, p in zip(actual, predicted)])


def calculate_mae(actual: List[float], predicted: List[float]) -> float:
    """Calculate Mean Absolute Error."""
    if len(actual) != len(predicted):
        raise ValueError("Actual and predicted lists must have same length")
    
    return np.mean([abs(a - p) for a, p in zip(actual, predicted)])


def calculate_mape(actual: List[float], predicted: List[float]) -> float:
    """Calculate Mean Absolute Percentage Error."""
    if len(actual) != len(predicted):
        raise ValueError("Actual and predicted lists must have same length")
    
    return np.mean([abs((a - p) / a) * 100 for a, p in zip(actual, predicted) if a != 0])


def calculate_rmse(actual: List[float], predicted: List[float]) -> float:
    """Calculate Root Mean Squared Error."""
    return np.sqrt(calculate_mse(actual, predicted))


def calculate_correlation(actual: List[float], predicted: List[float]) -> float:
    """Calculate Pearson correlation coefficient."""
    if len(actual) != len(predicted):
        raise ValueError("Actual and predicted lists must have same length")
    
    return np.corrcoef(actual, predicted)[0, 1]


def calculate_directional_accuracy(
    actual: List[float], 
    predicted: List[float],
    previous_actual: List[float]
) -> float:
    """
    Calculate directional accuracy (percentage of correct direction predictions).
    
    Args:
        actual: Actual values
        predicted: Predicted values  
        previous_actual: Previous actual values for direction calculation
        
    Returns:
        Directional accuracy as percentage (0-100)
    """
    if len(actual) != len(predicted) or len(actual) != len(previous_actual):
        raise ValueError("All lists must have same length")
    
    correct_directions = 0
    total_predictions = 0
    
    for i in range(len(actual)):
        if previous_actual[i] != 0:  # Avoid division by zero
            actual_direction = 1 if actual[i] > previous_actual[i] else -1
            predicted_direction = 1 if predicted[i] > previous_actual[i] else -1
            
            if actual_direction == predicted_direction:
                correct_directions += 1
            total_predictions += 1
    
    return (correct_directions / total_predictions * 100) if total_predictions > 0 else 0


def calculate_confidence_calibration(
    actual: List[float],
    predicted: List[float], 
    confidence_scores: List[float],
    num_bins: int = 10
) -> Dict[str, List[float]]:
    """
    Calculate confidence calibration metrics.
    
    Args:
        actual: Actual values
        predicted: Predicted values
        confidence_scores: Confidence scores (0-1)
        num_bins: Number of confidence bins
        
    Returns:
        Dict with bin_centers, bin_accuracies, and bin_counts
    """
    if len(actual) != len(predicted) or len(actual) != len(confidence_scores):
        raise ValueError("All lists must have same length")
    
    # Create confidence bins
    bin_edges = np.linspace(0, 1, num_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    bin_accuracies = []
    bin_counts = []
    
    for i in range(num_bins):
        # Find predictions in this confidence bin
        in_bin = [(conf >= bin_edges[i] and conf < bin_edges[i + 1]) 
                  for conf in confidence_scores]
        
        if i == num_bins - 1:  # Include upper bound in last bin
            in_bin = [(conf >= bin_edges[i] and conf <= bin_edges[i + 1]) 
                      for conf in confidence_scores]
        
        bin_actual = [actual[j] for j, in_b in enumerate(in_bin) if in_b]
        bin_predicted = [predicted[j] for j, in_b in enumerate(in_bin) if in_b]
        
        if bin_actual:
            # Calculate accuracy for this bin (using MAPE as accuracy metric)
            bin_mape = calculate_mape(bin_actual, bin_predicted)
            bin_accuracy = max(0, 100 - bin_mape)  # Convert MAPE to accuracy
            bin_accuracies.append(bin_accuracy)
            bin_counts.append(len(bin_actual))
        else:
            bin_accuracies.append(0)
            bin_counts.append(0)
    
    return {
        "bin_centers": bin_centers.tolist(),
        "bin_accuracies": bin_accuracies,
        "bin_counts": bin_counts
    }


def calculate_error_distribution(
    actual: List[float],
    predicted: List[float]
) -> Dict[str, float]:
    """
    Calculate error distribution statistics.
    
    Returns:
        Dict with error statistics
    """
    errors = [p - a for a, p in zip(actual, predicted)]
    percentage_errors = [abs((p - a) / a) * 100 for a, p in zip(actual, predicted) if a != 0]
    
    return {
        "mean_error": np.mean(errors),
        "std_error": np.std(errors),
        "min_error": np.min(errors),
        "max_error": np.max(errors),
        "median_error": np.median(errors),
        "q25_error": np.percentile(errors, 25),
        "q75_error": np.percentile(errors, 75),
        "mean_abs_error": np.mean([abs(e) for e in errors]),
        "mean_percentage_error": np.mean(percentage_errors),
        "std_percentage_error": np.std(percentage_errors)
    }


def analyze_method_performance(
    actual: List[float],
    predicted: List[float],
    method_weights_list: List[Dict[str, float]]
) -> Dict[str, Dict[str, float]]:
    """
    Analyze performance of different estimation methods.
    
    Args:
        actual: Actual values
        predicted: Predicted values
        method_weights_list: List of method weight dicts for each prediction
        
    Returns:
        Dict with performance metrics for each method
    """
    if len(actual) != len(predicted) or len(actual) != len(method_weights_list):
        raise ValueError("All lists must have same length")
    
    # Get all unique methods
    all_methods = set()
    for weights in method_weights_list:
        all_methods.update(weights.keys())
    
    method_performance = {}
    
    for method in all_methods:
        # Find predictions where this method had significant weight
        method_predictions = []
        method_actual = []
        
        for i, weights in enumerate(method_weights_list):
            if method in weights and weights[method] > 0.1:  # Significant contribution
                method_predictions.append(predicted[i])
                method_actual.append(actual[i])
        
        if method_actual:
            method_performance[method] = {
                "count": len(method_actual),
                "mse": calculate_mse(method_actual, method_predictions),
                "mae": calculate_mae(method_actual, method_predictions),
                "mape": calculate_mape(method_actual, method_predictions),
                "correlation": calculate_correlation(method_actual, method_predictions)
            }
        else:
            method_performance[method] = {
                "count": 0,
                "mse": None,
                "mae": None,
                "mape": None,
                "correlation": None
            }
    
    return method_performance


def create_performance_summary(
    actual: List[float],
    predicted: List[float],
    confidence_scores: Optional[List[float]] = None,
    method_weights_list: Optional[List[Dict[str, float]]] = None,
    previous_actual: Optional[List[float]] = None
) -> Dict[str, any]:
    """
    Create comprehensive performance summary.
    
    Returns:
        Dict with all performance metrics
    """
    summary = {
        "basic_metrics": {
            "count": len(actual),
            "mse": calculate_mse(actual, predicted),
            "mae": calculate_mae(actual, predicted),
            "rmse": calculate_rmse(actual, predicted),
            "mape": calculate_mape(actual, predicted),
            "correlation": calculate_correlation(actual, predicted)
        },
        "error_distribution": calculate_error_distribution(actual, predicted)
    }
    
    # Add directional accuracy if previous values provided
    if previous_actual:
        summary["directional_accuracy"] = calculate_directional_accuracy(
            actual, predicted, previous_actual
        )
    
    # Add confidence calibration if confidence scores provided
    if confidence_scores:
        summary["confidence_calibration"] = calculate_confidence_calibration(
            actual, predicted, confidence_scores
        )
    
    # Add method performance if method weights provided
    if method_weights_list:
        summary["method_performance"] = analyze_method_performance(
            actual, predicted, method_weights_list
        )
    
    return summary


def print_performance_summary(summary: Dict[str, any]):
    """Print formatted performance summary."""
    
    print("\n" + "="*50)
    print("HSTECH ESTIMATION PERFORMANCE SUMMARY")
    print("="*50)
    
    # Basic metrics
    metrics = summary["basic_metrics"]
    print(f"\nBasic Metrics ({metrics['count']} predictions):")
    print(f"  MSE:         {metrics['mse']:.2f}")
    print(f"  MAE:         {metrics['mae']:.2f}")
    print(f"  RMSE:        {metrics['rmse']:.2f}")
    print(f"  MAPE:        {metrics['mape']:.2f}%")
    print(f"  Correlation: {metrics['correlation']:.3f}")
    
    # Directional accuracy
    if "directional_accuracy" in summary:
        print(f"  Directional Accuracy: {summary['directional_accuracy']:.1f}%")
    
    # Error distribution
    if "error_distribution" in summary:
        err_dist = summary["error_distribution"]
        print(f"\nError Distribution:")
        print(f"  Mean Error:   {err_dist['mean_error']:.2f}")
        print(f"  Std Error:    {err_dist['std_error']:.2f}")
        print(f"  Median Error: {err_dist['median_error']:.2f}")
        print(f"  Q25-Q75:      {err_dist['q25_error']:.2f} to {err_dist['q75_error']:.2f}")
    
    # Method performance
    if "method_performance" in summary:
        print(f"\nMethod Performance:")
        for method, perf in summary["method_performance"].items():
            if perf["count"] > 0:
                print(f"  {method}:")
                print(f"    Count: {perf['count']}")
                print(f"    MAPE:  {perf['mape']:.2f}%")
                print(f"    Corr:  {perf['correlation']:.3f}")
    
    print("="*50 + "\n")
