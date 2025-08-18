"""
Delphyne: A Pre-trained Model for General and Financial Time Series

This package implements the Delphyne model as described in the paper:
"DELPHYNE: A PRE-TRAINED MODEL FOR GENERAL AND FINANCIAL TIMESERIES"
"""

__version__ = "0.1.0"
__author__ = "Delphyne Implementation Team"

from .model.delphyne import DelphyneModel
from .config import DelphyneConfig, TrainingConfig, DataConfig

__all__ = ["DelphyneModel", "DelphyneConfig", "TrainingConfig", "DataConfig"]
