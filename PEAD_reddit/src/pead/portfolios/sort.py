"""
Portfolio formation via decile/quintile sorting on SUE.

Forms decile portfolios from SUE rankings, independently within each
cross-section (event_week, optionally market_cap_bucket) to avoid look-ahead
bias. A global sort across the full sample would leak future information into
the breakpoints.

Reference: Bernard & Thomas (1989, 1990) for decile sort methodology.
           Livnat & Mendenhall (2006) for cross-sectional independence.
"""

from __future__ import annotations

import re
import warnings

import numpy as np
import pandas as pd

from pead.schema import Col


# ─── Internal helpers ───────────────────────────────────────────────────────


def _compute_event_week(announce_date: pd.Series) -> pd.Series:
    """Compute event_week as ISO year-week string for cross-section grouping.

    Using ISO calendar year + week ensures events in the same calendar week
    are sorted together, which is the relevant cross-section for point-in-time
    portfolio formation.
    """
    iso = announce_date.dt.isocalendar()
    return iso["year"].astype(str) + "-W" + iso["week"].astype(str).str.zfill(2)


def _qcut_safe(series: pd.Series, n_buckets: int) -> pd.Series:
    """Quantile-cut a series into 0..k-1 buckets, robust to ties / tiny groups.

    - Uses ``labels=False`` so the number of labels always matches the number
      of bins actually formed (handles ``duplicates='drop'`` producing fewer
      buckets than requested).
    - If qcut fails entirely (e.g. all-equal values or too few observations),
      every observation is placed in a single bucket (label 0).
    """
    try:
        return pd.qcut(series, n_buckets, labels=False, duplicates="drop")
    except (ValueError, IndexError):
        return pd.Series(0, index=series.index)


def _assign_bucket_within_group(
    df: pd.DataFrame,
    value_col: str,
    group_cols: list[str],
    n_buckets: int,
) -> pd.Series:
    """Assign 1..n bucket labels via qcut, independently within each group.

    Returns a Series aligned with ``df.index``. Bucket labels are 1-indexed
    (1 = lowest value of ``value_col`` within the group).
    """
    if group_cols:
        result = df.groupby(group_cols, observed=True)[value_col].transform(
            lambda s: _qcut_safe(s, n_buckets)
        )
    else:
        result = _qcut_safe(df[value_col], n_buckets)
    return result.astype(float) + 1  # 1-indexed (1 = lowest SUE)


def _detect_return_columns(df: pd.DataFrame) -> list[str]:
    """Detect return columns (CAR / BHAR / ret, optionally with window suffix).

    Matches e.g. ``car``, ``bhar``, ``ret``, ``car_0_1``, ``bhar_1_20``,
    ``ret_1_60``. Excludes non-return columns that happen to start with ret.
    """
    pattern = re.compile(r"^(car|bhar|ret)(_.+)?$")
    return [c for c in df.columns if pattern.match(c)]


def _compute_weights(df: pd.DataFrame, weight_scheme: str) -> np.ndarray:
    """Return a per-row weight vector (NOT yet normalized within decile).

    - ``value_weight``: uses ``market_cap`` (must be present, non-negative).
    - ``equal_weight``: all ones.
    """
    if weight_scheme == "equal_weight":
        return np.ones(len(df), dtype=float)
    if weight_scheme == "value_weight":
        if Col.MARKET_CAP not in df.columns:
            raise ValueError(
                "weight_scheme='value_weight' requires a 'market_cap' column "
                "in the assignments frame."
            )
        w = df[Col.MARKET_CAP].astype(float).to_numpy()
        # Sanitize: non-finite / non-positive weights cannot anchor value weights.
        w = np.where(np.isfinite(w) & (w > 0), w, np.nan)
        return w
    raise ValueError(
        f"Unknown weight_scheme={weight_scheme!r}; expected 'value_weight' or 'equal_weight'."
    )


# ─── Public API ─────────────────────────────────────────────────────────────


def assign_portfolio_deciles(
    sue_table: pd.DataFrame,
    event_returns: pd.DataFrame,
    sue_col: str = "sue3",
    n_portfolios: int = 10,
    sort_group_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Assign each event to a portfolio (1=lowest SUE, n=highest SUE).

    Merge ``sue_table`` with ``event_returns`` on ``[ticker, announce_date]``.
    If ``sort_group_cols`` is given (e.g. ``['event_week']``), sort WITHIN each
    group; otherwise sort globally (a look-ahead-bias warning is emitted).

    Added columns:
        - ``decile``              : 1..n (1 = lowest SUE)
        - ``event_week``          : ISO year-week of the announcement
        - ``market_cap_bucket``   : size quintile within event_week (if
                                    ``market_cap`` is available)

    Args:
        sue_table:       DataFrame with ``ticker``, ``announce_date`` and the
                         SUE column (``sue_col``).
        event_returns:   DataFrame with ``ticker``, ``announce_date`` and the
                         realised return columns (``car``, ``bhar``, ...). May
                         also carry ``market_cap`` used for value-weighting and
                         size buckets.
        sue_col:         Name of the SUE column to sort on.
        n_portfolios:    Number of portfolios (10 = deciles, 5 = quintiles).
        sort_group_cols: Columns defining the cross-section. Sorting within
                         these groups avoids look-ahead bias. ``['event_week']``
                         is the recommended minimum.

    Returns:
        Merged DataFrame with ``decile``, ``event_week`` and
        ``market_cap_bucket`` columns added.
    """
    # ── Validate inputs ──────────────────────────────────────────────────
    if sue_col not in sue_table.columns:
        raise ValueError(
            f"sue_table must contain the sort column {sue_col!r}; got {list(sue_table.columns)}."
        )
    for key in (Col.TICKER, Col.ANNOUNCE_DATE):
        if key not in sue_table.columns:
            raise ValueError(f"sue_table missing required column {key!r}.")
        if key not in event_returns.columns:
            raise ValueError(f"event_returns missing required column {key!r}.")

    sue_table = sue_table.copy()
    event_returns = event_returns.copy()
    sue_table[Col.ANNOUNCE_DATE] = pd.to_datetime(sue_table[Col.ANNOUNCE_DATE])
    event_returns[Col.ANNOUNCE_DATE] = pd.to_datetime(event_returns[Col.ANNOUNCE_DATE])

    # ── Merge ────────────────────────────────────────────────────────────
    # Prefer event_returns' market_cap if both frames carry one.
    df = sue_table.merge(
        event_returns,
        on=[Col.TICKER, Col.ANNOUNCE_DATE],
        how="inner",
        suffixes=("", "_er"),
    )

    if df.empty:
        raise ValueError(
            "Merge of sue_table and event_returns yielded no rows; check that "
            "(ticker, announce_date) keys align."
        )

    # ── event_week ───────────────────────────────────────────────────────
    df[Col.EVENT_WEEK] = _compute_event_week(df[Col.ANNOUNCE_DATE])

    # ── market_cap_bucket (size quintile within event_week) ──────────────
    if Col.MARKET_CAP in df.columns:
        df[Col.MARKET_CAP_BUCKET] = _assign_bucket_within_group(
            df, Col.MARKET_CAP, [Col.EVENT_WEEK], n_buckets=5
        ).astype("Int64")
    else:
        df[Col.MARKET_CAP_BUCKET] = pd.array([pd.NA] * len(df), dtype="Int64")

    # ── decile assignment ────────────────────────────────────────────────
    if sort_group_cols:
        # Validate group cols exist (event_week always exists now).
        missing_groups = [g for g in sort_group_cols if g not in df.columns]
        if missing_groups:
            raise ValueError(f"sort_group_cols not found in merged frame: {missing_groups}.")
        decile_raw = _assign_bucket_within_group(df, sue_col, list(sort_group_cols), n_portfolios)
    else:
        warnings.warn(
            "Deciles are being formed GLOBALLY across the entire sample. This "
            "introduces look-ahead bias because breakpoints use future data. "
            "Pass sort_group_cols=['event_week'] for proper point-in-time, "
            "cross-sectional sorting.",
            UserWarning,
            stacklevel=2,
        )
        decile_raw = _qcut_safe(df[sue_col], n_portfolios).astype(float) + 1

    df[Col.DECILE] = decile_raw.astype("Int64")
    return df


def compute_portfolio_returns(
    portfolio_assignments: pd.DataFrame,
    weight_scheme: str = "value_weight",
) -> pd.DataFrame:
    """Compute portfolio-level returns by decile.

    For each decile, compute the weighted average of every detected return
    column (CAR / BHAR / ret, with optional window suffix).

    Weight normalization happens WITHIN each decile so that each decile is a
    self-financing portfolio:
        - ``value_weight``  ->  w_i = mktcap_i / sum_j(mktcap_j)   (j in decile)
        - ``equal_weight``  ->  w_i = 1 / n_holdings

    Args:
        portfolio_assignments: Output of :func:`assign_portfolio_deciles`. Must
            contain ``decile`` plus at least one return column.
        weight_scheme: ``'value_weight'`` or ``'equal_weight'``.

    Returns:
        Long-format DataFrame with columns
        ``[decile, event_window, port_ret_gross, n_holdings]`` — one row per
        (decile, return-column) pair.
    """
    if Col.DECILE not in portfolio_assignments.columns:
        raise ValueError(
            "portfolio_assignments must contain a 'decile' column; "
            "run assign_portfolio_deciles first."
        )

    return_cols = _detect_return_columns(portfolio_assignments)
    if not return_cols:
        raise ValueError(
            "No return columns detected in portfolio_assignments. Expected a "
            "column matching ^(car|bhar|ret)(_.+)?$ (e.g. 'car', 'car_1_20')."
        )

    base_weights = _compute_weights(portfolio_assignments, weight_scheme)

    df = portfolio_assignments.copy()
    df["_base_w"] = base_weights

    rows: list[dict] = []
    decile_vals = df[Col.DECILE].dropna().astype(int).sort_values().unique()

    for rc in return_cols:
        for decile in decile_vals:
            grp = df[df[Col.DECILE] == decile]
            rets = grp[rc].astype(float).to_numpy()
            w = grp["_base_w"].astype(float).to_numpy()
            mask = np.isfinite(rets) & np.isfinite(w) & (w > 0)
            n = int(mask.sum())
            if n == 0:
                # Equal-weight fallback among rows that have a return only.
                ret_only = np.isfinite(rets)
                if ret_only.sum() == 0:
                    continue
                rm = rets[ret_only]
                wm = np.ones(len(rm))
            else:
                rm = rets[mask]
                wm = w[mask]
            wm = wm / wm.sum()
            weighted_ret = float(np.sum(wm * rm))
            rows.append(
                {
                    Col.DECILE: int(decile),
                    "event_window": rc,
                    Col.PORTFOLIO_RET_GROSS: weighted_ret,
                    Col.N_HOLDINGS: n,
                }
            )

    out = pd.DataFrame(rows)
    out = out.sort_values(["event_window", Col.DECILE]).reset_index(drop=True)
    return out
