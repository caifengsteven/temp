"""JPMorgan Treasury Fair-Value modeling framework.

Public API:
    - FairValueModel: Base OLS regression class wrapping statsmodels.
    - get_model1_data / get_model2_data / get_model3_data: Data builders
      for the three downstream fair-value models.
"""

from .base_model import FairValueModel
from .data import (
    BLOOMBERG_TICKERS,
    get_model1_data,
    get_model2_data,
    get_model3_data,
)

__all__ = [
    "FairValueModel",
    "BLOOMBERG_TICKERS",
    "get_model1_data",
    "get_model2_data",
    "get_model3_data",
]
