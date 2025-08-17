"""
LSTM-BEKK Trading System

A comprehensive implementation of the LSTM-BEKK model for multivariate volatility modeling
and trading strategy development based on the research paper:
"Deep Learning Enhanced Multivariate GARCH"

This package provides:
- Data infrastructure for financial time series
- LSTM-BEKK model implementation
- Trading strategies (GMV, volatility sizing, correlation breakdown)
- Risk management and performance evaluation
- Visualization and monitoring tools
"""

__version__ = "1.0.0"
__author__ = "LSTM-BEKK Trading System"

from .data import DataManager
from .models import LSTMBEKKModel
from .strategies import TradingStrategies
from .risk import RiskManager
from .backtesting import BacktestEngine
from .visualization import Visualizer

__all__ = [
    "DataManager",
    "LSTMBEKKModel", 
    "TradingStrategies",
    "RiskManager",
    "BacktestEngine",
    "Visualizer"
]
