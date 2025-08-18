"""
HSTECH Index Estimation System

A comprehensive system to estimate the HSTECH index price using US market data
when Hong Kong markets are closed.
"""

__version__ = "1.0.0"
__author__ = "HSTECH Estimation Team"

from .estimation.estimator import HSTECHEstimator

__all__ = ["HSTECHEstimator"]
