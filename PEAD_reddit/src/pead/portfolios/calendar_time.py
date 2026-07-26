"""
Calendar-time portfolio (CTP) assembly.

The robust alternative to event-time CAR averaging. Each calendar date the
portfolio holds every firm currently inside its post-earnings holding window
[T+holding_start, T+holding_end]. This controls for overlapping event windows
and the resulting serial correlation in event-time returns.

References:
    Lyon, Barber, Tsai (1999), "Improved Methods for Tests of Long-Run
        Abnormal Stock Returns", Journal of Finance.
    Mitchell & Stafford (2000), "Managerial Decisions and Long-Term Stock
        Price Performance", Journal of Business.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pead.schema import Col


# ─── Internal helpers ───────────────────────────────────────────────────────


def _ensure_dates(df: pd.DataFrame, col: str) -> None:
    """In-place conversion of ``df[col]`` to datetime64[ns]."""
    if col in df.columns:
        df[col] = pd.to_datetime(df[col])


def _weights_per_row(df: pd.DataFrame, weight_scheme: str) -> np.ndarray:
    """Return a per-row weight vector (NOT normalized within a calendar date).

    - ``value_weight``  -> market cap (non-negative, NaN if invalid)
    - ``equal_weight``  -> ones
    """
    if weight_scheme == "equal_weight":
        return np.ones(len(df), dtype=float)
    if weight_scheme == "value_weight":
        if Col.MARKET_CAP not in df.columns:
            raise ValueError("weight_scheme='value_weight' requires a 'market_cap' column.")
        w = df[Col.MARKET_CAP].astype(float).to_numpy()
        return np.where(np.isfinite(w) & (w > 0), w, np.nan)
    raise ValueError(
        f"Unknown weight_scheme={weight_scheme!r}; expected 'value_weight' or 'equal_weight'."
    )


def _select_decile(df: pd.DataFrame, decile: int | str) -> pd.DataFrame:
    """Filter ``df`` to a single decile or the long-short legs.

    Returns:
        - For ``decile`` an int : rows with ``df.decile == decile``.
        - For ``decile`` == ``"long_short"`` : rows in the extreme deciles
          (top and bottom), with a ``leg`` column added
          (``'long'`` for top, ``'short'`` for bottom).
    """
    if Col.DECILE not in df.columns:
        raise ValueError("portfolio_assignments must contain a 'decile' column.")

    decile_vals = pd.Series(df[Col.DECILE].dropna().astype(int).unique())
    if decile_vals.empty:
        raise ValueError("portfolio_assignments has no decile assignments.")

    if isinstance(decile, (int, np.integer)):
        sub = df[df[Col.DECILE] == int(decile)].copy()
        if sub.empty:
            raise ValueError(
                f"Decile {decile} has no holdings; available deciles: "
                f"{sorted(decile_vals.tolist())}."
            )
        return sub

    if decile != "long_short":
        raise ValueError(f"decile must be an int or 'long_short', got {decile!r}.")

    top = int(decile_vals.max())
    bottom = int(decile_vals.min())
    long_leg = df[df[Col.DECILE] == top].copy()
    short_leg = df[df[Col.DECILE] == bottom].copy()
    long_leg["leg"] = "long"
    short_leg["leg"] = "short"
    return pd.concat([long_leg, short_leg], ignore_index=True)


# ─── Public API ─────────────────────────────────────────────────────────────


def build_calendar_time_portfolio(
    portfolio_assignments: pd.DataFrame,
    prices: pd.DataFrame,
    holding_start: int = 1,
    holding_end: int = 20,
    decile: int | str = "long_short",
    weight_scheme: str = "value_weight",
) -> pd.DataFrame:
    """Build a calendar-time portfolio (CTP).

    For each calendar trading date ``d`` the portfolio holds every firm that
    had an earnings announcement in the window
    ``[d - holding_end, d - holding_start]`` trading days ago. The portfolio
    return on date ``d`` is the value- (or equal-) weighted average of the
    firm-level returns realized on that date.

    Args:
        portfolio_assignments: Output of :func:`pead.portfolios.sort.assign_portfolio_deciles`.
            Must have ``ticker``, ``announce_date``, ``decile``; ``market_cap``
            is required when ``weight_scheme='value_weight'``.
        prices: Long-format daily prices with at least ``ticker``,
            ``trading_date``, ``ret``.
        holding_start: First post-announcement trading day to hold (T+1).
        holding_end:   Last post-announcement trading day to hold (T+20).
        decile:        Single decile int, or ``"long_short"`` for the long-top
                       minus short-bottom hedge portfolio.
        weight_scheme: ``'value_weight'`` or ``'equal_weight'``.

    Returns:
        DataFrame with columns
        ``[calendar_date, portfolio_ret_gross, n_holdings]`` (one row per
        calendar date). For ``decile='long_short'`` the return is the
        top-decile return minus the bottom-decile return, and ``n_holdings``
        is the count of long holdings (top decile) on that date.
    """
    # ── Validate & copy ──────────────────────────────────────────────────
    for key in (Col.TICKER, Col.ANNOUNCE_DATE, Col.DECILE):
        if key not in portfolio_assignments.columns:
            raise ValueError(f"portfolio_assignments missing column {key!r}.")
    for key in (Col.TICKER, Col.TRADING_DATE, Col.RET):
        if key not in prices.columns:
            raise ValueError(f"prices missing column {key!r}.")

    assign = portfolio_assignments.copy()
    prices = prices.copy()
    _ensure_dates(assign, Col.ANNOUNCE_DATE)
    _ensure_dates(prices, Col.TRADING_DATE)

    if holding_start < 1:
        raise ValueError(
            f"holding_start must be >= 1 (got {holding_start}); "
            "holding starts on T+1 to avoid the announcement-day reaction."
        )
    if holding_end < holding_start:
        raise ValueError(f"holding_end ({holding_end}) must be >= holding_start ({holding_start}).")

    # ── Build a (ticker -> sorted trading dates) index for offset lookup ─
    px_dates = (
        prices[[Col.TICKER, Col.TRADING_DATE]]
        .drop_duplicates()
        .sort_values([Col.TICKER, Col.TRADING_DATE])
        .reset_index(drop=True)
    )
    px_dates["row_idx"] = px_dates.groupby(Col.TICKER).cumcount()

    # Map each announce_date to its trading-day index per ticker.
    announce_idx = px_dates.merge(
        assign[[Col.TICKER, Col.ANNOUNCE_DATE]].drop_duplicates(),
        on=Col.TICKER,
        how="inner",
    )
    # For each (ticker, announce_date) find the row_idx of the announce_date
    # (or the first available trading day >= announce_date).
    announce_idx = announce_idx[announce_idx[Col.TRADING_DATE] >= announce_idx[Col.ANNOUNCE_DATE]]
    announce_idx = (
        announce_idx.sort_values([Col.TICKER, Col.ANNOUNCE_DATE, Col.TRADING_DATE])
        .drop_duplicates(subset=[Col.TICKER, Col.ANNOUNCE_DATE], keep="first")[
            [Col.TICKER, Col.ANNOUNCE_DATE, "row_idx"]
        ]
        .rename(columns={"row_idx": "announce_row_idx"})
    )

    # ── Assign base weights (per holding, independent of calendar date) ──
    assign["_base_w"] = _weights_per_row(assign, weight_scheme)

    # Attach announce_row_idx back to assignments.
    holding = assign.merge(announce_idx, on=[Col.TICKER, Col.ANNOUNCE_DATE], how="inner")
    if holding.empty:
        raise ValueError(
            "No (ticker, announce_date) in portfolio_assignments matched a "
            "trading date in prices. Check that prices span the announce dates."
        )

    # ── Decide which deciles we need to build ────────────────────────────
    long_short_mode = decile == "long_short"
    sub = _select_decile(holding, decile)
    decile_vals = pd.Series(sub[Col.DECILE].dropna().astype(int).unique())
    top_d = int(decile_vals.max()) if not decile_vals.empty else None
    bottom_d = int(decile_vals.min()) if not decile_vals.empty else None

    # ── For each ticker, enumerate the calendar dates inside its window ──
    # Build a long table of (ticker, announce_date, announce_row_idx,
    # calendar_row_idx, calendar_date, decile, weight).
    frames: list[pd.DataFrame] = []
    for offset in range(holding_start, holding_end + 1):
        offset_rows = sub.copy()
        offset_rows["calendar_row_idx"] = offset_rows["announce_row_idx"] + offset
        frames.append(offset_rows)
    expanded = pd.concat(frames, ignore_index=True)

    # Map (ticker, calendar_row_idx) -> calendar_date & return.
    date_lookup = px_dates.rename(
        columns={"row_idx": "calendar_row_idx", Col.TRADING_DATE: "calendar_date_dt"}
    )
    expanded = expanded.merge(
        date_lookup[[Col.TICKER, "calendar_row_idx", "calendar_date_dt"]],
        on=[Col.TICKER, "calendar_row_idx"],
        how="inner",
    )
    if expanded.empty:
        # No holding dates fell on actual trading days.
        return pd.DataFrame(columns=[Col.CALENDAR_DATE, Col.PORTFOLIO_RET_GROSS, Col.N_HOLDINGS])

    # Pull the realized return for each (ticker, calendar_date).
    ret_lookup = prices[[Col.TICKER, Col.TRADING_DATE, Col.RET]].rename(
        columns={Col.TRADING_DATE: "calendar_date_dt", Col.RET: "_ret"}
    )
    expanded = expanded.merge(ret_lookup, on=[Col.TICKER, "calendar_date_dt"], how="left")

    expanded.rename(columns={"calendar_date_dt": Col.CALENDAR_DATE}, inplace=True)

    # ── Aggregate per calendar date ──────────────────────────────────────
    def _weighted_ret(g: pd.DataFrame) -> tuple[float, int]:
        r = g["_ret"].astype(float).to_numpy()
        w = g["_base_w"].astype(float).to_numpy()
        mask = np.isfinite(r) & np.isfinite(w) & (w > 0)
        if mask.sum() == 0:
            return (np.nan, 0)
        rm, wm = r[mask], w[mask]
        wm = wm / wm.sum()
        return (float(np.sum(wm * rm)), int(mask.sum()))

    def _per_date_block(g: pd.DataFrame) -> pd.DataFrame:
        """Aggregate one (calendar_date) block, possibly splitting long/short."""
        if long_short_mode:
            long_g = g[g[Col.DECILE] == top_d]
            short_g = g[g[Col.DECILE] == bottom_d]
            ret_long, n_long = _weighted_ret(long_g)
            ret_short, _ = _weighted_ret(short_g)
            ret = (
                (ret_long - ret_short)
                if (np.isfinite(ret_long) and np.isfinite(ret_short))
                else np.nan
            )
            return pd.DataFrame(
                {
                    Col.PORTFOLIO_RET_GROSS: [ret],
                    Col.N_HOLDINGS: [n_long],
                }
            )
        ret, n = _weighted_ret(g)
        return pd.DataFrame(
            {
                Col.PORTFOLIO_RET_GROSS: [ret],
                Col.N_HOLDINGS: [n],
            }
        )

    agg = (
        expanded.groupby(Col.CALENDAR_DATE, as_index=False, sort=True)
        .apply(_per_date_block, include_groups=False)
        .reset_index(drop=True)
    )

    # Preserve calendar_date ordering from the groupby.
    dates_sorted = pd.Series(expanded[Col.CALENDAR_DATE].sort_values().unique())
    agg[Col.CALENDAR_DATE] = dates_sorted.values

    # Drop dates with zero holdings (shouldn't happen post-merge, but be safe).
    agg = agg[agg[Col.N_HOLDINGS] > 0].reset_index(drop=True)
    return agg[[Col.CALENDAR_DATE, Col.PORTFOLIO_RET_GROSS, Col.N_HOLDINGS]]


def compute_long_short_returns(
    ctp_top: pd.DataFrame,
    ctp_bottom: pd.DataFrame,
) -> pd.DataFrame:
    """Compute long-short returns: top decile return − bottom decile return.

    This is the PEAD hedge portfolio: long the highest-SUE decile and short
    the lowest-SUE decile.

    Args:
        ctp_top:    CTP returns for the top decile, with columns
                    ``[calendar_date, portfolio_ret_gross, n_holdings]``.
        ctp_bottom: CTP returns for the bottom decile, same schema.

    Returns:
        DataFrame with columns
        ``[calendar_date, portfolio_ret_gross, n_holdings_long]`` where
        ``portfolio_ret_gross`` is the long-short return (top − bottom) and
        ``n_holdings_long`` is the number of long-side holdings on that date
        (carried over from ``ctp_top``).
    """
    for key in (Col.CALENDAR_DATE, Col.PORTFOLIO_RET_GROSS, Col.N_HOLDINGS):
        if key not in ctp_top.columns:
            raise ValueError(f"ctp_top missing column {key!r}.")
        if key not in ctp_bottom.columns:
            raise ValueError(f"ctp_bottom missing column {key!r}.")

    t = ctp_top[[Col.CALENDAR_DATE, Col.PORTFOLIO_RET_GROSS, Col.N_HOLDINGS]].rename(
        columns={Col.PORTFOLIO_RET_GROSS: "_ret_top", Col.N_HOLDINGS: "n_holdings_long"}
    )
    b = ctp_bottom[[Col.CALENDAR_DATE, Col.PORTFOLIO_RET_GROSS]].rename(
        columns={Col.PORTFOLIO_RET_GROSS: "_ret_bot"}
    )

    out = t.merge(b, on=Col.CALENDAR_DATE, how="inner")
    out[Col.PORTFOLIO_RET_GROSS] = out["_ret_top"] - out["_ret_bot"]
    return out[[Col.CALENDAR_DATE, Col.PORTFOLIO_RET_GROSS, "n_holdings_long"]].reset_index(
        drop=True
    )
