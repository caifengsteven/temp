"""
PEAD asymmetry testing module.

Public API:
    - :func:`compute_asymmetry_ratio` — headline ratio point estimate
    - :func:`compute_asymmetry_by_sue_method` — ratio across SUE1/2/3
    - :func:`compute_asymmetry_by_window` — ratio across event windows
    - :func:`compute_amihud_by_event` — Amihud (2002) illiquidity
    - :func:`double_sort_asymmetry` — liquidity double-sort
    - :func:`double_sort_by_size` — size double-sort
    - :func:`bootstrap_asymmetry_ratio` — bootstrap CI for the ratio
    - :func:`clustered_difference_test` — firm-clustered t-test
    - :func:`full_asymmetry_report` — master report table
"""

from __future__ import annotations

from pead.asymmetry.double_sort import (
    compute_amihud_by_event,
    double_sort_asymmetry,
    double_sort_by_size,
)
from pead.asymmetry.inference import (
    bootstrap_asymmetry_ratio,
    clustered_difference_test,
    full_asymmetry_report,
)
from pead.asymmetry.ratio import (
    compute_asymmetry_by_sue_method,
    compute_asymmetry_by_window,
    compute_asymmetry_ratio,
)

__all__ = [
    "compute_asymmetry_ratio",
    "compute_asymmetry_by_sue_method",
    "compute_asymmetry_by_window",
    "compute_amihud_by_event",
    "double_sort_asymmetry",
    "double_sort_by_size",
    "bootstrap_asymmetry_ratio",
    "clustered_difference_test",
    "full_asymmetry_report",
]
