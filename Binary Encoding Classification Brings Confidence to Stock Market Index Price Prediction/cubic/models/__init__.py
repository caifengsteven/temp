"""
Models module for CUBIC framework
"""

from .fusion_module import FusionInLatentSpace
from .backbones import LSTMBackbone, TransformerBackbone, MLPBackbone
from .cubic_model import CUBICModel

__all__ = [
    "FusionInLatentSpace", 
    "LSTMBackbone", 
    "TransformerBackbone", 
    "MLPBackbone", 
    "CUBICModel"
]
