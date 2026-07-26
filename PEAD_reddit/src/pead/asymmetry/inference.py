"""
Statistical inference for the PEAD asymmetry ratio.

The asymmetry ratio ``|miss_drift| / |beat_drift`` is a ratio of two random
variables. When the denominator (the beat drift) is near zero, the ratio has
a fat-tailed, highly non-normal distribution. Analytic standard errors
(Delta method) systematically understate the uncertainty. We therefore use
the **percentile bootstrap** for the ratio CI.

For the simple *difference* ``miss_mean - beat_mean`` the distribution is
well-behaved, so we use a **clustered t-test** (clustered by firm) following
Cameron-Gelbach-Miller (2011, REStat) to account for serial correlation in
the same firm's earnings surprises.

Reference: Efron & Tibshirani (1993) for the bootstrap; Cameron, Gelbach &
           Miller (2011) for cluster-robust inference; Thompson (2011, JFE)
           for two-way clustering (the framework here extends naturally).
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm

from pead.asymmetry.ratio import compute_asymmetry_ratio
from pead.schema import Col

# Cap the per-batch array at ~50M float64 elements to bound memory.
# 50M float64 ≈ 400 MB which is comfortably below typical machine RAM.
_BOOTSTRAP_BATCH_ELEMENTS = 50_000_000


def _extract_extreme_groups(
    event_returns: pd.DataFrame,
    return_col: str,
    decile_col: str,
    extreme_decile: int = 1,
    n_deciles: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (miss_returns, beat_returns) as 1-D float arrays."""
    df = event_returns[[decile_col, return_col]].copy()
    df[decile_col] = pd.to_numeric(df[decile_col], errors="coerce")
    df[return_col] = pd.to_numeric(df[return_col], errors="coerce")
    df = df.dropna()
    miss = df.loc[df[decile_col] == extreme_decile, return_col].to_numpy(dtype=float)
    beat = df.loc[df[decile_col] == n_deciles, return_col].to_numpy(dtype=float)
    return miss, beat


def bootstrap_asymmetry_ratio(
    event_returns: pd.DataFrame,
    return_col: str = "car_short_drift",
    decile_col: str = "sue3_decile",
    n_bootstrap: int = 10000,
    confidence_level: float = 0.95,
    seed: int = 42,
    extreme_decile: int = 1,
    n_deciles: int = 10,
) -> dict:
    """Bootstrap the asymmetry ratio to obtain a percentile confidence interval.

    Process:
      1. Separate events into miss decile (bottom) and beat decile (top).
      2. Resample WITH replacement from each group, ``n_bootstrap`` times.
         Each bootstrap iteration draws ``n_miss`` events from the miss group
         and ``n_beat`` events from the beat group, then computes
         ``|miss_mean| / |beat_mean|``.
      3. Percentile CI at ``confidence_level``.

    Args:
        event_returns: DataFrame with ``[decile_col, return_col]``.
        return_col: CAR column (e.g. ``car_short_drift``).
        decile_col: Decile assignment column (1..``n_deciles``).
        n_bootstrap: Number of bootstrap resamples.
        confidence_level: CI coverage (e.g. 0.95 for a 95% CI).
        seed: RNG seed for reproducibility.
        extreme_decile: Bottom decile value (large miss).
        n_deciles: Top decile value (large beat).

    Returns:
        Dict with keys ``ratio_point_estimate``, ``ci_lower``, ``ci_upper``,
        ``bootstrap_std``, ``bootstrap_distribution`` (np.ndarray of all valid
        bootstrap ratios), ``n_miss``, ``n_beat``, ``n_bootstrap_requested``,
        ``n_bootstrap_valid`` (after dropping non-finite ratios).
    """
    if not 0 < confidence_level < 1:
        raise ValueError(f"confidence_level must be in (0, 1), got {confidence_level}")
    if n_bootstrap < 1:
        raise ValueError(f"n_bootstrap must be >= 1, got {n_bootstrap}")

    miss, beat = _extract_extreme_groups(
        event_returns, return_col, decile_col, extreme_decile, n_deciles
    )
    n_miss = int(len(miss))
    n_beat = int(len(beat))
    if n_miss == 0 or n_beat == 0:
        return {
            "ratio_point_estimate": float("nan"),
            "ci_lower": float("nan"),
            "ci_upper": float("nan"),
            "bootstrap_std": float("nan"),
            "bootstrap_distribution": np.array([], dtype=float),
            "n_miss": n_miss,
            "n_beat": n_beat,
            "n_bootstrap_requested": int(n_bootstrap),
            "n_bootstrap_valid": 0,
        }

    miss_mean = float(miss.mean())
    beat_mean = float(beat.mean())
    if beat_mean != 0.0:
        point_ratio = abs(miss_mean) / abs(beat_mean)
    else:
        point_ratio = float("nan")

    rng = np.random.default_rng(seed)

    # Chunk the bootstrap to bound peak memory. Each chunk produces arrays of
    # shape (chunk_size, n_obs); mean-reduce immediately.
    max_per_chunk = max(1, _BOOTSTRAP_BATCH_ELEMENTS // max(n_miss, n_beat))
    ratios_chunks: list[np.ndarray] = []
    remaining = int(n_bootstrap)
    while remaining > 0:
        chunk = min(max_per_chunk, remaining)
        miss_idx = rng.integers(0, n_miss, size=(chunk, n_miss))
        beat_idx = rng.integers(0, n_beat, size=(chunk, n_beat))
        miss_means = miss[miss_idx].mean(axis=1)
        beat_means = beat[beat_idx].mean(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.abs(miss_means) / np.abs(beat_means)
        r = r[np.isfinite(r)]
        ratios_chunks.append(r)
        remaining -= chunk

    ratios = np.concatenate(ratios_chunks) if ratios_chunks else np.array([], dtype=float)

    if len(ratios) == 0:
        return {
            "ratio_point_estimate": point_ratio,
            "ci_lower": float("nan"),
            "ci_upper": float("nan"),
            "bootstrap_std": float("nan"),
            "bootstrap_distribution": ratios,
            "n_miss": n_miss,
            "n_beat": n_beat,
            "n_bootstrap_requested": int(n_bootstrap),
            "n_bootstrap_valid": 0,
        }

    alpha = 1.0 - confidence_level
    ci_lower = float(np.percentile(ratios, 100.0 * (alpha / 2.0)))
    ci_upper = float(np.percentile(ratios, 100.0 * (1.0 - alpha / 2.0)))
    bootstrap_std = float(np.std(ratios, ddof=1)) if len(ratios) > 1 else float("nan")

    return {
        "ratio_point_estimate": float(point_ratio),
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "bootstrap_std": bootstrap_std,
        "bootstrap_distribution": ratios,
        "n_miss": n_miss,
        "n_beat": n_beat,
        "n_bootstrap_requested": int(n_bootstrap),
        "n_bootstrap_valid": int(len(ratios)),
    }


def clustered_difference_test(
    event_returns: pd.DataFrame,
    return_col: str = "car_short_drift",
    decile_col: str = "sue3_decile",
    cluster_col: str = Col.TICKER,
    extreme_decile: int = 1,
    n_deciles: int = 10,
) -> dict:
    """Formal difference test with cluster-robust standard errors.

    Tests ``H0: miss_drift_mean = beat_drift_mean`` against a two-sided
    alternative. Implemented as an OLS regression of the event return on a
    constant plus an ``is_miss`` indicator, with cluster-robust standard
    errors (cluster by firm to account for serial correlation in the same
    firm's earnings surprises).

    Coefficient on the indicator = ``miss_mean - beat_mean`` (negative when
    misses drift more negatively than beats drift positively).

    Args:
        event_returns: DataFrame with ``[decile_col, return_col, cluster_col]``.
        return_col: CAR column.
        decile_col: Decile column.
        cluster_col: Column to cluster by (default ``ticker``).
        extreme_decile: Bottom decile value.
        n_deciles: Top decile value.

    Returns:
        Dict with keys ``difference``, ``se`` (cluster-robust SE),
        ``t_stat``, ``p_value`` (two-sided), ``degrees_of_freedom``,
        ``n_clusters``, ``n_obs``, ``beat_mean``, ``miss_mean``.
    """
    required = [decile_col, return_col, cluster_col]
    missing = [c for c in required if c not in event_returns.columns]
    if missing:
        raise ValueError(f"event_returns missing required columns: {missing}")

    df = event_returns[required].copy()
    df[decile_col] = pd.to_numeric(df[decile_col], errors="coerce")
    df[return_col] = pd.to_numeric(df[return_col], errors="coerce")
    df = df.dropna()
    df = df.loc[df[decile_col].isin([extreme_decile, n_deciles])].copy()
    if df.empty:
        return {
            "difference": float("nan"),
            "se": float("nan"),
            "t_stat": float("nan"),
            "p_value": float("nan"),
            "degrees_of_freedom": 0,
            "n_clusters": 0,
            "n_obs": 0,
            "beat_mean": float("nan"),
            "miss_mean": float("nan"),
        }

    df["is_miss"] = (df[decile_col] == extreme_decile).astype(float)
    n_clusters = int(df[cluster_col].nunique())
    if n_clusters < 2:
        return {
            "difference": float(
                df.loc[df["is_miss"] == 1, return_col].mean()
                - df.loc[df["is_miss"] == 0, return_col].mean()
            ),
            "se": float("nan"),
            "t_stat": float("nan"),
            "p_value": float("nan"),
            "degrees_of_freedom": n_clusters - 1,
            "n_clusters": n_clusters,
            "n_obs": int(len(df)),
            "beat_mean": float(df.loc[df["is_miss"] == 0, return_col].mean()),
            "miss_mean": float(df.loc[df["is_miss"] == 1, return_col].mean()),
        }

    design = sm.add_constant(df["is_miss"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = sm.OLS(df[return_col].to_numpy(dtype=float), design).fit(
            cov_type="cluster",
            cov_kwds={"groups": df[cluster_col].to_numpy()},
            use_t=True,
        )

    # Cluster-robust inference uses (n_clusters - 1) dof, not the OLS residual dof.
    dof = max(n_clusters - 1, 1)
    diff = float(model.params["is_miss"])
    se = float(model.bse["is_miss"])
    t_stat = float(model.tvalues["is_miss"])

    # Recompute the p-value with the correct cluster dof for transparency.
    from scipy import stats as _stats

    p_value = float(2.0 * _stats.t.sf(abs(t_stat), df=dof))

    return {
        "difference": diff,
        "se": se,
        "t_stat": t_stat,
        "p_value": p_value,
        "degrees_of_freedom": int(dof),
        "n_clusters": n_clusters,
        "n_obs": int(len(df)),
        "beat_mean": float(df.loc[df["is_miss"] == 0, return_col].mean()),
        "miss_mean": float(df.loc[df["is_miss"] == 1, return_col].mean()),
    }


# Adjustment level -> column suffix convention used by ``full_asymmetry_report``.
_ADJUSTMENT_SUFFIXES: dict[str, str] = {
    "raw": "",
    "ff5_adjusted": "_ff5_adjusted",
    "dgtw_adjusted": "_dgtw_adjusted",
    "cost_adjusted": "_cost_adjusted",
}


def full_asymmetry_report(
    event_returns: pd.DataFrame,
    prices: pd.DataFrame | None = None,
    return_col: str = "car_short_drift",
    windows: list[str] | None = None,
    sue_methods: list[str] | None = None,
    adjustment_levels: list[str] | None = None,
    n_bootstrap: int = 10000,
    confidence_level: float = 0.95,
    seed: int = 42,
    cluster_col: str = Col.TICKER,
) -> pd.DataFrame:
    """Master asymmetry report table.

    Produces one row per ``(sue_method, window, adjustment_level)`` with
    point estimates, bootstrap CI, and cluster-robust difference test.

    Rows for adjustment levels whose columns are absent from ``event_returns``
    are silently dropped (so on raw synthetic data only the ``raw`` row per
    window appears).

    Args:
        event_returns: Event-return DataFrame produced upstream.
        prices: Daily prices, reserved for future liquidity-adjusted rows
            (currently unused; kept for API symmetry).
        return_col: Default CAR column; overridden per-window during iteration.
        windows: Event-window return columns to scan. Defaults to announcement
            reaction, short drift, medium drift.
        sue_methods: SUE methods to iterate; methods without a decile column
            in ``event_returns`` are skipped.
        adjustment_levels: Subset of ``["raw", "ff5_adjusted",
            "dgtw_adjusted", "cost_adjusted"]``; defaults to all four.
        n_bootstrap: Bootstrap iterations for the ratio CI.
        confidence_level: Bootstrap CI coverage.
        seed: Bootstrap RNG seed.
        cluster_col: Column to cluster the difference test by.

    Returns:
        DataFrame with columns ``[sue_method, window, adjustment_level,
        miss_car, beat_car, ratio, ci_lower, ci_upper, t_stat, p_value,
        n_miss, n_beat]``.
    """
    if windows is None:
        windows = ["car_announcement_reaction", "car_short_drift", "car_medium_drift"]
    if sue_methods is None:
        sue_methods = ["sue1", "sue2", "sue3"]
    if adjustment_levels is None:
        adjustment_levels = list(_ADJUSTMENT_SUFFIXES.keys())

    rows: list[dict[str, Any]] = []
    for sue_method in sue_methods:
        decile_col = f"{sue_method}_decile"
        if decile_col not in event_returns.columns:
            continue
        for window in windows:
            for adj in adjustment_levels:
                suffix = _ADJUSTMENT_SUFFIXES.get(adj, "")
                col = window if not suffix else f"{window}{suffix}"
                if col not in event_returns.columns:
                    continue

                ratio_res = compute_asymmetry_ratio(
                    event_returns,
                    sue_method=sue_method,
                    return_col=col,
                    decile_col=decile_col,
                )
                boot_res = bootstrap_asymmetry_ratio(
                    event_returns,
                    return_col=col,
                    decile_col=decile_col,
                    n_bootstrap=n_bootstrap,
                    confidence_level=confidence_level,
                    seed=seed,
                )
                diff_res = clustered_difference_test(
                    event_returns,
                    return_col=col,
                    decile_col=decile_col,
                    cluster_col=cluster_col,
                )

                rows.append(
                    {
                        "sue_method": sue_method,
                        "window": window,
                        "adjustment_level": adj,
                        "return_col": col,
                        "miss_car": ratio_res["miss_car_mean"],
                        "beat_car": ratio_res["beat_car_mean"],
                        "ratio": ratio_res["ratio"],
                        "ci_lower": boot_res["ci_lower"],
                        "ci_upper": boot_res["ci_upper"],
                        "t_stat": diff_res["t_stat"],
                        "p_value": diff_res["p_value"],
                        "n_miss": ratio_res["miss_n"],
                        "n_beat": ratio_res["beat_n"],
                    }
                )

    return pd.DataFrame(rows)


__all__ = [
    "bootstrap_asymmetry_ratio",
    "clustered_difference_test",
    "full_asymmetry_report",
]
