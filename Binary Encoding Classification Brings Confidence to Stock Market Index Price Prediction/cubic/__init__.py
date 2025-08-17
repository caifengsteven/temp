"""
CUBIC: Component fUsion and Binary encoding classIfication with Confidence
for Stock Market Index Price Prediction

This package implements the CUBIC framework as described in the paper:
"Why Regression? Binary Encoding Classification Brings Confidence to Stock Market Index Price Prediction"
"""

__version__ = "1.0.0"
__author__ = "CUBIC Implementation Team"

from .data import BloombergDataFetcher, TechnicalIndicators
from .models import CUBICModel, LSTMBackbone, TransformerBackbone, MLPBackbone
from .utils import BinaryEncoder, ConfidenceMeasures, ConfigManager
from .training import CUBICTrainer
from .evaluation import CUBICEvaluator

__all__ = [
    "BloombergDataFetcher",
    "TechnicalIndicators", 
    "CUBICModel",
    "LSTMBackbone",
    "TransformerBackbone", 
    "MLPBackbone",
    "BinaryEncoder",
    "ConfidenceMeasures",
    "ConfigManager",
    "CUBICTrainer",
    "CUBICEvaluator"
]
