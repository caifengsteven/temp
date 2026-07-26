"""Analyst-based SUE variant (SUE3).

SUE3 measures the earnings surprise relative to the *median* analyst
forecast, following Livnat & Mendenhall (2006), "Comparing the Post-Earnings
Announcement Drift for Surprises Calculated from Analyst and Time-Series
Forecasts," JAR 44(1): 177-205.

    SUE3 = (actual_eps - medest_eps) / price_qe

where ``medest_eps`` is the median of the most-recent individual analyst
forecasts inside a window of ``analyst_window_days`` ending on the
announcement date. The median (not the mean) is the Livnat-Mendenhall
standard because it is robust to outlier forecasts.

For the synthetic data the median forecast is pre-computed in
``pead.synthetic``, so :func:`compute_sue3` just applies the formula. For
real Bloomberg / I/B/E/S data, :func:`_filter_analyst_forecasts` performs the
standard deduplication — keep only the last forecast per analyst per event
within the window — before the median is taken.
"""

from __future__ import annotations

import logging

import pandas as pd

from pead.schema import Col, validate_dataframe

logger = logging.getLogger(__name__)

# Default analyst-forecast window (days before announcement).
_DEFAULT_ANALYST_WINDOW_DAYS = 90

_SUE3_REQUIRED: list[str] = [
    Col.TICKER,
    Col.ANNOUNCE_DATE,
    Col.FISCAL_QUARTER,
    Col.ACTUAL_EPS,
    Col.MEDEST_EPS,
    Col.PRICE_QUARTER_END,
]


def compute_sue3(earnings_events: pd.DataFrame) -> pd.DataFrame:
    """SUE3: Analyst-median forecast based surprise (Livnat-Mendenhall, RE3).

    Formula: ``SUE3 = (actual_eps - medest_eps) / price_qe``.

    The deflator is the quarter-end price, split-adjusted by ``adj_factor``
    when that column is present so SUE3 is denominated in the same (fraction
    of price) units as SUE1/SUE2. In the synthetic pipeline ``adj_factor``
    is 1.0 everywhere so this reduces to the literal formula above.

    Args:
        earnings_events: Event frame (see ``ExpectedSchema.EARNINGS_EVENTS``)
            augmented with ``price_qe`` (and optionally ``adj_factor``).
            ``medest_eps`` is assumed to already be the within-window median
            analyst forecast.

    Returns:
        DataFrame with columns ``[ticker, announce_date, fiscal_quarter, sue3]``.
    """
    validate_dataframe(earnings_events, _SUE3_REQUIRED, name="earnings_events")

    df = earnings_events.copy()
    df[Col.ANNOUNCE_DATE] = pd.to_datetime(df[Col.ANNOUNCE_DATE])

    deflator = df[Col.PRICE_QUARTER_END]
    if Col.ADJ_FACTOR in df.columns:
        deflator = deflator / df[Col.ADJ_FACTOR]

    df[Col.SUE3] = (df[Col.ACTUAL_EPS] - df[Col.MEDEST_EPS]) / deflator

    out = df[[Col.TICKER, Col.ANNOUNCE_DATE, Col.FISCAL_QUARTER, Col.SUE3]].copy()
    return out.reset_index(drop=True)


def _filter_analyst_forecasts(
    forecasts: pd.DataFrame,
    announce_date_col: str = Col.ANNOUNCE_DATE,
    forecast_date_col: str = "forecast_date",
    analyst_id_col: str = "analyst_id",
    window_days: int = _DEFAULT_ANALYST_WINDOW_DAYS,
) -> pd.DataFrame:
    """Deduplicate raw analyst forecasts to one (latest) estimate per analyst.

    Implements the standard I/B/E/S-style consensus construction used by
    Livnat & Mendenhall (2006): for each earnings event, restrict to forecasts
    made within ``window_days`` calendar days *before* the announcement, then
    keep only the most recent forecast from each analyst. The caller then
    takes the median across analysts of the surviving forecast value.

    An earnings event is identified by ``(ticker, announce_date)``; both
    columns are required on ``forecasts``.

    Args:
        forecasts: Long-format frame of individual analyst forecasts. Must
            contain ``ticker``, ``announce_date_col``, ``forecast_date_col``,
            ``analyst_id_col`` plus a forecast-value column.
        announce_date_col: Name of the announcement-date column.
        forecast_date_col: Name of the forecast (revision) date column.
        analyst_id_col: Name of the analyst identifier column.
        window_days: Length (calendar days) of the pre-announcement window.

    Returns:
        Filtered frame with at most one row per ``(ticker, announce_date,
        analyst)`` — the latest in-window forecast for each.
    """
    required = [Col.TICKER, announce_date_col, forecast_date_col, analyst_id_col]
    missing = [c for c in required if c not in forecasts.columns]
    if missing:
        raise ValueError(f"forecasts missing required columns: {missing}")

    fc = forecasts.copy()
    fc[announce_date_col] = pd.to_datetime(fc[announce_date_col])
    fc[forecast_date_col] = pd.to_datetime(fc[forecast_date_col])

    window_start = fc[announce_date_col] - pd.Timedelta(days=window_days)
    in_window = (fc[forecast_date_col] >= window_start) & (
        fc[forecast_date_col] <= fc[announce_date_col]
    )
    sub = fc.loc[in_window]

    if sub.empty:
        return sub

    group_keys = [Col.TICKER, announce_date_col, analyst_id_col]
    latest_idx = sub.groupby(group_keys, observed=True)[forecast_date_col].idxmax()
    deduped = sub.loc[latest_idx].sort_index()
    return deduped
