"""
Backtesting Module for HSTECH Estimation System

This module provides comprehensive backtesting capabilities to validate
the accuracy and performance of the HSTECH index estimation methods.
"""

from .backtester import HSTECHBacktester, BacktestResult
from .metrics import (
    calculate_mse,
    calculate_mae,
    calculate_mape,
    calculate_rmse,
    calculate_correlation,
    calculate_directional_accuracy,
    calculate_confidence_calibration,
    create_performance_summary,
    print_performance_summary
)

__all__ = [
    "HSTECHBacktester",
    "BacktestResult",
    "calculate_mse",
    "calculate_mae",
    "calculate_mape",
    "calculate_rmse",
    "calculate_correlation",
    "calculate_directional_accuracy",
    "calculate_confidence_calibration",
    "create_performance_summary",
    "print_performance_summary"
]