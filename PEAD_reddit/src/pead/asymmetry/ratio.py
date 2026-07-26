"""
Core asymmetry ratio computation.

The asymmetry ratio is the headline number of the PEAD asymmetry analysis:
    ratio = |miss_drift| / |beat_drift

where `miss_drift` is the mean post-announcement drift of the bottom SUE decile
(large misses) and `beat_drift` is the mean post-announcement drift of the top
SUE decile (large beats). The Reddit claim is that this ratio is ~4.5x at T+20.

This module only computes point estimates. Statistical inference (CIs, t-tests)
lives in :mod:`pead.asymmetry.inference`.

Reference: Bird, Choi & Yeung (2011); Zhang, Gregoriou & Wu (2024);
           Narayanamoorthy (2006); Livnat & Mendenhall (2006).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pead.schema import Col

# Default SUE methods supported by the pipeline. ``sue2`` is included for
# completeness even though it is rarely the headline method.
_DEFAULT_SUE_METHODS: tuple[str, ...] = ("sue1", "sue2", "sue3")

# Default event-window return columns produced upstream by the event-study module.
_DEFAULT_WINDOWS: tuple[str, ...] = (
    "car_announcement_reaction",
    "car_short_drift",
    "car_medium_drift",
)


def compute_asymmetry_ratio(
    event_returns: pd.DataFrame,
    sue_method: str = "sue3",
    return_col: str = "car_short_drift",
    decile_col: str | None = None,
    extreme_decile: int = 1,
    n_deciles: int = 10,
) -> dict:
    """Compute the core asymmetry ratio: ``|miss drift| / |beat drift|``.

    Miss decile = ``extreme_decile`` (bottom, value 1 by default).
    Beat decile = ``n_deciles`` (top, value 10 by default).

    Args:
        event_returns: DataFrame with a decile column and a return column.
            Must contain at least ``[decile_col, return_col]``.
        sue_method: Label of the SUE method (``"sue1"``/``"sue2"``/``"sue3"``).
            Only used for labelling the returned dict and to resolve the
            default ``decile_col``.
        return_col: Name of the cumulative abnormal return column to average
            (e.g. ``"car_short_drift"``).
        decile_col: Name of the column holding 1..``n_deciles`` assignments.
            If ``None``, defaults to ``f"{sue_method}_decile"``.
        extreme_decile: Decile value that defines a "large miss" (bottom).
        n_deciles: Total number of deciles; the top decile (``n_deciles``)
            defines a "large beat".

    Returns:
        Dict with keys: ``sue_method``, ``return_col``, ``decile_col``,
        ``miss_car_mean``, ``miss_car_median``, ``miss_n``,
        ``beat_car_mean``, ``beat_car_median``, ``beat_n``,
        ``ratio`` (``|miss_mean| / |beat_mean|``), and
        ``difference`` (``miss_car_mean - abs(beat_car_mean)``,
        negative when misses drift more than beats).

    Notes:
        The ratio uses **absolute values** because misses have negative drift
        and beats have positive drift — the signed ratio would be negative
        and meaningless. Returns ``np.nan`` for the ratio when the beat mean
        is zero or when either group is empty.
    """
    if decile_col is None:
        decile_col = f"{sue_method}_decile"

    missing = [c for c in (decile_col, return_col) if c not in event_returns.columns]
    if missing:
        raise ValueError(
            f"event_returns missing required columns for sue_method={sue_method!r}: {missing}"
        )

    df = event_returns[[decile_col, return_col]].copy()
    df[decile_col] = pd.to_numeric(df[decile_col], errors="coerce")
    df[return_col] = pd.to_numeric(df[return_col], errors="coerce")
    df = df.dropna()

    miss_mask = df[decile_col] == extreme_decile
    beat_mask = df[decile_col] == n_deciles

    miss = df.loc[miss_mask, return_col]
    beat = df.loc[beat_mask, return_col]

    miss_mean = float(miss.mean()) if len(miss) else float("nan")
    miss_median = float(miss.median()) if len(miss) else float("nan")
    beat_mean = float(beat.mean()) if len(beat) else float("nan")
    beat_median = float(beat.median()) if len(beat) else float("nan")

    if np.isfinite(beat_mean) and beat_mean != 0.0 and np.isfinite(miss_mean):
        ratio = abs(miss_mean) / abs(beat_mean)
    else:
        ratio = float("nan")

    # ``difference`` follows the spec exactly: miss_mean - |beat_mean|.
    # Both groups drift in opposite directions, so when misses drift MORE
    # (more negative), this difference is more negative.
    if np.isfinite(miss_mean) and np.isfinite(beat_mean):
        difference = miss_mean - abs(beat_mean)
    else:
        difference = float("nan")

    return {
        "sue_method": sue_method,
        "return_col": return_col,
        "decile_col": decile_col,
        "extreme_decile": extreme_decile,
        "n_deciles": n_deciles,
        "miss_car_mean": miss_mean,
        "miss_car_median": miss_median,
        "miss_n": int(len(miss)),
        "beat_car_mean": beat_mean,
        "beat_car_median": beat_median,
        "beat_n": int(len(beat)),
        "ratio": float(ratio),
        "difference": float(difference),
    }


def compute_asymmetry_by_sue_method(
    event_returns: pd.DataFrame,
    return_col: str = "car_short_drift",
    sue_methods: list[str] | None = None,
    extreme_decile: int = 1,
    n_deciles: int = 10,
) -> pd.DataFrame:
    """Compute the asymmetry ratio separately for each SUE method.

    The Livnat-Mendenhall (2006) finding: drift magnitude differs by SUE
    construction method. If a 4.5x ratio appears only under one method, the
    result is likely methodological rather than behavioural.

    Args:
        event_returns: DataFrame with ``f"{method}_decile"`` columns and
            ``return_col``.
        return_col: CAR column to average.
        sue_methods: SUE methods to iterate. Defaults to sue1/sue2/sue3;
            methods whose decile column is absent are silently skipped.
        extreme_decile: Bottom decile value (large miss).
        n_deciles: Top decile value (large beat).

    Returns:
        DataFrame with one row per available SUE method and columns matching
        the keys of :func:`compute_asymmetry_ratio`.
    """
    methods = sue_methods or list(_DEFAULT_SUE_METHODS)
    rows: list[dict] = []
    for method in methods:
        decile_col = f"{method}_decile"
        if decile_col not in event_returns.columns:
            continue
        rows.append(
            compute_asymmetry_ratio(
                event_returns,
                sue_method=method,
                return_col=return_col,
                decile_col=decile_col,
                extreme_decile=extreme_decile,
                n_deciles=n_deciles,
            )
        )
    return pd.DataFrame(rows)


def compute_asymmetry_by_window(
    event_returns: pd.DataFrame,
    windows: list[str] | None = None,
    sue_method: str = "sue3",
    decile_col: str | None = None,
    extreme_decile: int = 1,
    n_deciles: int = 10,
) -> pd.DataFrame:
    """Compute the asymmetry ratio across multiple event windows.

    The Reddit claim is specifically about T+20 (``car_short_drift``). Testing
    additional windows — T+1 announcement reaction and T+60 medium drift —
    checks whether the asymmetry is a short-horizon or persistent phenomenon.

    Args:
        event_returns: DataFrame with the window return columns and the
            decile column for ``sue_method``.
        windows: List of return-column names. Defaults to the announcement
            reaction, short drift, and medium drift columns.
        sue_method: SUE method whose decile column is used.
        decile_col: Override for the decile column name.
        extreme_decile: Bottom decile value.
        n_deciles: Top decile value.

    Returns:
        DataFrame with one row per available window.
    """
    if decile_col is None:
        decile_col = f"{sue_method}_decile"
    if decile_col not in event_returns.columns:
        raise ValueError(
            f"event_returns missing decile column {decile_col!r} for sue_method={sue_method!r}"
        )

    win_list = windows if windows is not None else list(_DEFAULT_WINDOWS)
    rows: list[dict] = []
    for window in win_list:
        if window not in event_returns.columns:
            continue
        rows.append(
            compute_asymmetry_ratio(
                event_returns,
                sue_method=sue_method,
                return_col=window,
                decile_col=decile_col,
                extreme_decile=extreme_decile,
                n_deciles=n_deciles,
            )
        )
    return pd.DataFrame(rows)


# Re-export the canonical identifier for downstream imports.
__all__ = [
    "Col",
    "compute_asymmetry_ratio",
    "compute_asymmetry_by_sue_method",
    "compute_asymmetry_by_window",
]
