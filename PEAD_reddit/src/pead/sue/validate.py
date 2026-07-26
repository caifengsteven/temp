"""Livnat-Mendenhall data-quality filters and cross-sectional SUE deciles.

Reference: Livnat & Mendenhall (2006), "Comparing the Post-Earnings
Announcement Drift for Surprises Calculated from Analyst and Time-Series
Forecasts," JAR 44(1): 177-205.

Two responsibilities live here:

1. :func:`apply_lm_filters` — enforce the screen used by Livnat-Mendenhall
   (price, market cap, announcement/report-date agreement, finite SUE) so the
   estimation universe is clean and comparable to the published study.

2. :func:`assign_deciles` — form SUE deciles *independently within each
   cross-section* so portfolio breakpoints never use information from future
   periods (no look-ahead bias).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from pead.schema import Col

logger = logging.getLogger(__name__)

# All SUE columns that the NaN/inf screen should consider when present.
_SUE_COLS: list[str] = [Col.SUE1, Col.SUE2, Col.SUE3]


def _sue_columns_present(df: pd.DataFrame) -> list[str]:
    return [c for c in _SUE_COLS if c in df.columns]


def apply_lm_filters(
    sue_table: pd.DataFrame,
    fundamentals: pd.DataFrame,
    min_price: float = 1.0,
    min_mcap_millions: float = 5.0,
    max_date_diff_days: int = 1,
) -> pd.DataFrame:
    """Apply the Livnat-Mendenhall (2006) data-quality screen.

    Filters applied (a row must pass *all* to survive):

    1. Quarter-end price ``price_qe`` > ``min_price`` (drops penny stocks).
    2. Market cap ``price_qe * shares_basic`` > ``min_mcap_millions``
       (``shares_basic`` is in millions, so the product is in USD millions).
    3. ``|report_date - announce_date|`` <= ``max_date_diff_days`` (ensures the
       Compustat report date and the event announcement date agree).
    4. Drop rows where every present SUE column is NaN.
    5. Drop rows where any present SUE column is infinite.

    Args:
        sue_table: Frame produced by the SUE constructors, keyed by
            ``(ticker, fiscal_quarter)`` with an ``announce_date`` column.
        fundamentals: Quarterly fundamentals frame carrying ``price_qe``,
            ``shares_basic`` and ``report_date``.
        min_price: Minimum quarter-end price (USD).
        min_mcap_millions: Minimum market cap (USD millions).
        max_date_diff_days: Maximum tolerated |report - announce| lag in days.

    Returns:
        The input ``sue_table`` enriched with the fundamentals fields used by
        the screen plus two diagnostic columns:

        - ``filtered_out`` (bool): ``True`` if the row fails at least one
          filter and should be excluded from estimation.
        - ``filter_reason`` (str): semicolon-joined names of the filters that
          the row failed (empty for rows that pass).

        Obtain the surviving universe with ``result.loc[~result["filtered_out"]]``.
    """
    key = [Col.TICKER, Col.FISCAL_QUARTER]

    fund_cols = [
        Col.TICKER,
        Col.FISCAL_QUARTER,
        Col.PRICE_QUARTER_END,
        Col.SHARES_BASIC,
        Col.REPORT_DATE,
    ]
    # Carry announce_date over from fundamentals if the SUE table lacks it.
    if Col.ANNOUNCE_DATE not in sue_table.columns and Col.ANNOUNCE_DATE in fundamentals.columns:
        fund_cols.append(Col.ANNOUNCE_DATE)

    merged = sue_table.merge(
        fundamentals[fund_cols],
        on=key,
        how="left",
        validate="many_to_one",
    )

    if Col.ANNOUNCE_DATE not in merged.columns:
        raise ValueError(
            "announce_date must be present on sue_table or fundamentals to "
            "apply the report-date filter."
        )

    merged[Col.ANNOUNCE_DATE] = pd.to_datetime(merged[Col.ANNOUNCE_DATE])
    merged[Col.REPORT_DATE] = pd.to_datetime(merged[Col.REPORT_DATE])

    price = merged[Col.PRICE_QUARTER_END]
    shares = merged[Col.SHARES_BASIC]

    # (1) Price screen.
    fail_price = price.fillna(-np.inf) <= min_price

    # (2) Market-cap screen (shares in millions → product in $MM).
    mcap = price * shares
    fail_mcap = mcap.fillna(-np.inf) <= min_mcap_millions

    # (3) Announcement/report-date agreement.
    date_diff = (merged[Col.REPORT_DATE] - merged[Col.ANNOUNCE_DATE]).abs().dt.days
    fail_date = date_diff.isna() | (date_diff > max_date_diff_days)

    # (4) & (5) NaN / infinite SUE values.
    sue_cols = _sue_columns_present(merged)
    if sue_cols:
        sue_block = merged[sue_cols]
        fail_all_nan = sue_block.isna().all(axis=1)
        fail_inf = sue_block.apply(np.isinf).any(axis=1)
    else:
        fail_all_nan = pd.Series(False, index=merged.index)
        fail_inf = pd.Series(False, index=merged.index)

    reasons = pd.Series("", index=merged.index, dtype=object)
    reasons = reasons.where(~fail_price, other=reasons + "price;")
    reasons = reasons.where(~fail_mcap, other=reasons + "mcap;")
    reasons = reasons.where(~fail_date, other=reasons + "date;")
    reasons = reasons.where(~fail_all_nan, other=reasons + "sue_nan;")
    reasons = reasons.where(~fail_inf, other=reasons + "sue_inf;")

    filtered_out = fail_price | fail_mcap | fail_date | fail_all_nan | fail_inf
    merged["filtered_out"] = filtered_out.to_numpy()
    merged["filter_reason"] = reasons.str.rstrip(";").to_numpy()

    n_dropped = int(filtered_out.sum())
    n_total = len(merged)
    logger.info(
        "Livnat-Mendenhall screen: dropped %d / %d rows (%.1f%%).",
        n_dropped,
        n_total,
        100.0 * n_dropped / n_total if n_total else 0.0,
    )
    return merged


def assign_deciles(
    sue_table: pd.DataFrame,
    sue_col: str = Col.SUE3,
    n_deciles: int = 10,
    group_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Assign SUE deciles *per cross-section* (no look-ahead bias).

    Breakpoints are formed independently within each group defined by
    ``group_cols`` (typically ``["event_week"]`` or
    ``["event_week", "market_cap_bucket"]``). Decile 1 is the lowest-SUE
    bucket (large misses); decile ``n_deciles`` is the highest (large beats),
    matching the :data:`Col.IS_MISS` / :data:`Col.IS_BEAT` convention.

    Uses :func:`pandas.qcut` with ``duplicates="drop"`` so groups with many
    tied values safely collapse to fewer bins rather than raising. NaN SUE
    observations receive a NaN decile.

    .. note::
       **Simplification:** the academic standard (Fama-French, Bernard-Thomas)
       forms breakpoints from NYSE-listed stocks only. This pipeline uses all
       available stocks for simplicity — acceptable for a synthetic
       replication but worth revisiting for a publication-grade study.

    Args:
        sue_table: Frame containing the SUE column to bucket.
        sue_col: Name of the SUE column to bucket.
        n_deciles: Target number of buckets (1..n_deciles).
        group_cols: Columns defining a cross-section. If ``None``, deciles are
            formed globally and a warning is logged (this leaks information
            across periods and is discouraged for event studies).

    Returns:
        Copy of ``sue_table`` with an added ``f"{sue_col}_decile"`` column.
    """
    if sue_col not in sue_table.columns:
        raise ValueError(f"sue column {sue_col!r} not found on sue_table")
    if n_deciles < 2:
        raise ValueError("n_deciles must be >= 2")

    out_col = f"{sue_col}_decile"
    df = sue_table.copy()

    def _bucket(series: pd.Series) -> pd.Series:
        codes = pd.qcut(series, n_deciles, labels=False, duplicates="drop")
        return (codes + 1).astype("float64")

    if group_cols is None:
        logger.warning(
            "assign_deciles called with group_cols=None — forming deciles "
            "GLOBALLY across all periods. This introduces look-ahead bias and "
            "should only be used for ad-hoc inspection."
        )
        df[out_col] = _bucket(df[sue_col])
    else:
        missing = [c for c in group_cols if c not in df.columns]
        if missing:
            raise ValueError(f"group_cols missing from sue_table: {missing}")
        df[out_col] = df.groupby(group_cols, observed=True)[sue_col].transform(_bucket)

    return df
