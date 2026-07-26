"""Time-series SUE variants (SUE1 and SUE2).

Both follow the seasonal random walk model of Livnat & Mendenhall (2006),
"Comparing the Post-Earnings Announcement Drift for Surprises Calculated
from Analyst and Time-Series Forecasts," JAR 44(1): 177-205.

- SUE1: Seasonal random walk on split-adjusted primary EPS.
        Forecast_t = EPS_{t-4};  SUE1 = (EPS_t - EPS_{t-4}) / P_t.

- SUE2: Seasonal random walk on EPS *excluding after-tax special items*
        (Livnat-Mendenhall fn. 6). The pre-tax Compustat special item
        (spiq) is converted to an after-tax per-share impact using the
        0.65 factor, then removed from reported EPS before differencing.

All EPS and price quantities are split-adjusted by ``adj_factor`` so the
surprise is expressed as a fraction of (split-adjusted) price — i.e. a
SUE of 0.01 means an earnings surprise equal to 1% of the stock price.
"""

from __future__ import annotations

import logging

import pandas as pd

from pead.schema import (
    Col,
    validate_dataframe,
)

logger = logging.getLogger(__name__)

# Columns required to construct SUE1.
_SUE1_REQUIRED: list[str] = [
    Col.TICKER,
    Col.FISCAL_QUARTER_END,
    Col.FISCAL_QUARTER,
    Col.EPS_PRIMARY,
    Col.PRICE_QUARTER_END,
    Col.ADJ_FACTOR,
]

# SUE2 additionally needs special items and basic shares.
_SUE2_REQUIRED: list[str] = _SUE1_REQUIRED + [Col.SPECIAL_ITEMS, Col.SHARES_BASIC]

# Number of quarters in the seasonal random walk lag.
_SEASONAL_LAG = 4


def _resolve_announce_date(df: pd.DataFrame) -> pd.Series:
    """Return the announcement date series, falling back to report date.

    The Compustat fundamentals table keys off ``report_date`` (rdq) while the
    event-study side keys off ``announce_date``. The synthetic generator
    carries ``announce_date`` through directly; for raw Compustat pulls where
    only ``report_date`` is present we fall back to it so the SUE table can
    still be joined to event-window returns downstream.
    """
    if Col.ANNOUNCE_DATE in df.columns:
        return pd.to_datetime(df[Col.ANNOUNCE_DATE])
    if Col.REPORT_DATE in df.columns:
        logger.info("announce_date not on fundamentals frame; falling back to report_date.")
        return pd.to_datetime(df[Col.REPORT_DATE])
    raise ValueError(
        f"fundamentals must contain {Col.ANNOUNCE_DATE!r} or {Col.REPORT_DATE!r} "
        "to label SUE rows with an announcement date."
    )


def _split_adjusted_eps(df: pd.DataFrame) -> pd.Series:
    """Split-adjusted primary EPS: eps_primary / adj_factor."""
    return df[Col.EPS_PRIMARY] / df[Col.ADJ_FACTOR]


def _split_adjusted_price(df: pd.DataFrame) -> pd.Series:
    """Split-adjusted quarter-end price: price_qe / adj_factor (the SUE deflator)."""
    return df[Col.PRICE_QUARTER_END] / df[Col.ADJ_FACTOR]


def compute_sue1(fundamentals: pd.DataFrame) -> pd.DataFrame:
    """SUE1: Seasonal Random Walk (Livnat-Mendenhall, RE1).

    Formula: ``SUE1 = (EPS_t - EPS_{t-4}) / P_t`` where ``EPS`` is the
    split-adjusted primary EPS (``eps_primary / adj_factor``) and ``P_t`` is
    the split-adjusted quarter-end price (``price_qe / adj_factor``).

    Rows are sorted by ticker + fiscal quarter end, then lagged four quarters
    *within each ticker* so firm histories never bleed into one another. The
    first four observations per firm have no lag and are dropped.

    Args:
        fundamentals: Quarterly fundamentals frame (see ``ExpectedSchema.FUNDAMENTALS_Q``)
            augmented with ``announce_date`` and ``adj_factor``.

    Returns:
        DataFrame with columns ``[ticker, announce_date, fiscal_quarter, sue1]``.
    """
    validate_dataframe(fundamentals, _SUE1_REQUIRED, name="fundamentals")

    df = fundamentals.copy()
    df[Col.FISCAL_QUARTER_END] = pd.to_datetime(df[Col.FISCAL_QUARTER_END])
    df = df.sort_values([Col.TICKER, Col.FISCAL_QUARTER_END]).reset_index(drop=True)

    eps_adj = _split_adjusted_eps(df)
    price_adj = _split_adjusted_price(df)

    eps_lag4 = eps_adj.groupby(df[Col.TICKER]).shift(_SEASONAL_LAG)
    df[Col.SUE1] = (eps_adj - eps_lag4) / price_adj
    df[Col.ANNOUNCE_DATE] = _resolve_announce_date(df)

    out = df[[Col.TICKER, Col.ANNOUNCE_DATE, Col.FISCAL_QUARTER, Col.SUE1]].copy()
    # First four quarters per firm have no seasonal lag → drop.
    out = out.dropna(subset=[Col.SUE1]).reset_index(drop=True)
    return out


def compute_sue2(fundamentals: pd.DataFrame, tax_adj: float = 0.65) -> pd.DataFrame:
    """SUE2: Random Walk excluding Special Items (Livnat-Mendenhall, RE2).

    The reported primary EPS is stripped of the after-tax per-share effect of
    Compustat special items (``spiq``) before forming the seasonal difference::

        after_tax_eps_impact = tax_adj * special_items / shares_basic
        actual2              = eps_primary - after_tax_eps_impact
        expected2            = actual2_{t-4}        (same seasonal random walk)
        SUE2                 = (actual2 - expected2) / P_t

    ``special_items`` is pre-tax in Compustat; the default ``tax_adj`` of 0.65
    follows Livnat & Mendenhall (2006, fn. 6) and approximates the average
    after-tax statutory rate. Both ``actual2`` and ``P_t`` are split-adjusted
    via ``adj_factor`` for consistency with :func:`compute_sue1`.

    Args:
        fundamentals: Quarterly fundamentals frame with special items and shares.
        tax_adj: After-tax conversion factor applied to pre-tax special items.

    Returns:
        DataFrame with columns ``[ticker, announce_date, fiscal_quarter, sue2]``.
    """
    validate_dataframe(fundamentals, _SUE2_REQUIRED, name="fundamentals")

    df = fundamentals.copy()
    df[Col.FISCAL_QUARTER_END] = pd.to_datetime(df[Col.FISCAL_QUARTER_END])
    df = df.sort_values([Col.TICKER, Col.FISCAL_QUARTER_END]).reset_index(drop=True)

    # After-tax per-share impact of pre-tax special items, then split-adjust.
    special_items_per_share = df[Col.SPECIAL_ITEMS] / df[Col.SHARES_BASIC]
    actual2_raw = df[Col.EPS_PRIMARY] - tax_adj * special_items_per_share
    actual2 = actual2_raw / df[Col.ADJ_FACTOR]

    price_adj = _split_adjusted_price(df)
    actual2_lag4 = actual2.groupby(df[Col.TICKER]).shift(_SEASONAL_LAG)
    df[Col.SUE2] = (actual2 - actual2_lag4) / price_adj
    df[Col.ANNOUNCE_DATE] = _resolve_announce_date(df)

    out = df[[Col.TICKER, Col.ANNOUNCE_DATE, Col.FISCAL_QUARTER, Col.SUE2]].copy()
    out = out.dropna(subset=[Col.SUE2]).reset_index(drop=True)
    return out
