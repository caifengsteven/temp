"""SUE (Standardized Unexpected Earnings) construction module.

Three SUE variants following Livnat & Mendenhall (2006):

- :func:`compute_sue1` — seasonal random walk (time-series).
- :func:`compute_sue2` — seasonal random walk excluding after-tax special items.
- :func:`compute_sue3` — analyst median-forecast based surprise.

Plus the Livnat-Mendenhall data-quality screen (:func:`apply_lm_filters`) and
look-ahead-free cross-sectional decile assignment (:func:`assign_deciles`).
"""

from __future__ import annotations

from pead.sue.analyst import compute_sue3
from pead.sue.time_series import compute_sue1, compute_sue2
from pead.sue.validate import apply_lm_filters, assign_deciles

__all__ = [
    "compute_sue1",
    "compute_sue2",
    "compute_sue3",
    "apply_lm_filters",
    "assign_deciles",
]
