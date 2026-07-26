"""
Fama-French 5-factor (and 3-factor) time-series regression for risk adjustment.

Models:
    FF5: R_p - Rf = alpha + b1*MKT_RF + b2*SMB + b3*HML + b4*RMW + b5*CMA + eps
    FF3: R_p - Rf = alpha + b1*MKT_RF + b2*SMB + b3*HML             + eps

The intercept (alpha) is the factor-adjusted abnormal return. Comparing the raw
PEAD drift to the FF5-adjusted alpha separates behavioral underreaction from
factor exposure. If raw PEAD shows a 4.5x asymmetry but FF5-adjusted shows 1.5x,
the difference is factor exposure (SMB/HML/...), not behavioral asymmetry.

Standard errors: Newey-West HAC (6 lags) — required for valid inference on
overlapping / serially-correlated return data. Ordinary OLS SEs are forbidden.

Frequency handling: FF5 factor returns are monthly. Portfolio returns that
arrive at a higher frequency (e.g. daily CTP returns) are compounded to monthly
to align with the factor panel, so the reported alpha is a monthly abnormal
return. (A daily CAR alpha × TRADING_DAYS_PER_MONTH is the linear approximation
of this compounding; compounding is exact.)

Reference: Fama & French (2015), "A five-factor asset pricing model," JFE 116(1).
           Newey & Bartlett (1987) HAC covariance estimator.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm

from pead.schema import TRADING_DAYS_PER_MONTH, Col

# ─── Factor groups ───────────────────────────────────────────────────────────

FF5_FACTORS: list[str] = [Col.MKT_RF, Col.SMB, Col.HML, Col.RMW, Col.CMA]
FF3_FACTORS: list[str] = [Col.MKT_RF, Col.SMB, Col.HML]

# Newey-West HAC maximum lag (in periods). 6 monthly lags ~ ½ year of
# autocorrelation correction, standard for monthly factor regressions.
NEWEY_WEST_LAGS: int = 6

# Candidate date column names used to locate the time index on input frames.
_DATE_COLUMNS: tuple[str, ...] = (
    Col.CALENDAR_DATE,
    Col.TRADING_DATE,
    "calendar_date",
    "trading_date",
    "date",
)


# ─── Helpers ─────────────────────────────────────────────────────────────────


def _extract_date_column(df: pd.DataFrame, name: str) -> pd.Series:
    """Return a datetime Series for *df*, whether the dates live on the index
    or in a conventional column. Raises ValueError if nothing is found."""
    if isinstance(df.index, pd.DatetimeIndex):
        return pd.to_datetime(df.index)
    for col in _DATE_COLUMNS:
        if col in df.columns:
            return pd.to_datetime(df[col])
    raise ValueError(
        f"{name}: could not locate a date column. Looked for index or one of {_DATE_COLUMNS}."
    )


def _aggregate_to_monthly(
    portfolio_returns: pd.DataFrame,
    factors: pd.DataFrame,
    portfolio_col: str,
    factor_list: list[str],
) -> pd.DataFrame:
    """Align portfolio returns and factors to a common monthly Period grid.

    * Portfolio returns are compounded within each calendar month:
        monthly_ret = prod(1 + r) - 1
      This both aggregates daily CTP returns to monthly (the conversion the
      spec references) and is a no-op when the input is already monthly.
    * Factor returns are already monthly in the canonical panel; we take the
      last observation per month defensively.
    * The risk-free rate (``rf``) is carried through so the dependent variable
      is the portfolio's excess return.
    """
    pr = portfolio_returns.copy()
    pr["_period"] = _extract_date_column(pr, "portfolio_returns").dt.to_period("M")
    monthly_port = (
        pr.groupby("_period")[portfolio_col]
        .apply(lambda s: np.expm1(np.log1p(s).sum()))
        .rename("port_ret")
    )

    fc = factors.copy()
    fc["_period"] = _extract_date_column(fc, "factors").dt.to_period("M")
    carry_cols = list(factor_list)
    if Col.RF in fc.columns:
        carry_cols.append(Col.RF)
    monthly_factors = fc.groupby("_period")[carry_cols].last()

    merged = monthly_port.to_frame().join(monthly_factors, how="inner")
    merged = merged.replace([np.inf, -np.inf], np.nan).dropna()
    return merged


def _empty_result(factor_list: list[str], n_months: int = 0) -> dict[str, Any]:
    return {
        "alpha": float("nan"),
        "alpha_tstat": float("nan"),
        "alpha_pvalue": float("nan"),
        "betas": {f: float("nan") for f in factor_list},
        "r_squared": float("nan"),
        "n_months": int(n_months),
    }


def _run_regression(merged: pd.DataFrame, factor_list: list[str]) -> dict[str, Any]:
    """Run OLS of excess portfolio return on the factor list with Newey-West
    HAC (maxlags=NEWEY_WEST_LAGS) standard errors."""
    if Col.RF in merged.columns:
        rf = merged[Col.RF]
    else:
        rf = 0.0  # portfolio already in excess-return form
    y = merged["port_ret"] - rf
    design = sm.add_constant(merged[factor_list])

    n_obs = int(len(merged))
    # Need strictly more observations than parameters for a rank-deficient-safe fit.
    if n_obs < len(factor_list) + 2:
        return _empty_result(factor_list, n_months=n_obs)

    maxlags = min(NEWEY_WEST_LAGS, max(1, n_obs - 1))
    model = sm.OLS(y, design, hasconst=True)
    result = model.fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})

    const_name = "const" if "const" in result.params.index else result.params.index[0]
    return {
        "alpha": float(result.params[const_name]),
        "alpha_tstat": float(result.tvalues[const_name]),
        "alpha_pvalue": float(result.pvalues[const_name]),
        "betas": {f: float(result.params[f]) for f in factor_list},
        "r_squared": float(result.rsquared),
        "n_months": n_obs,
    }


# ─── Public API ──────────────────────────────────────────────────────────────


def run_ff5_regression(
    portfolio_returns: pd.DataFrame,
    factors: pd.DataFrame,
    portfolio_col: str = Col.PORTFOLIO_RET_GROSS,
) -> dict[str, Any]:
    """Run the Fama-French 5-factor time-series regression.

        R_portfolio - Rf = alpha + b1*MKT_RF + b2*SMB + b3*HML
                              + b4*RMW + b5*CMA + epsilon

    Uses statsmodels OLS with Newey-West HAC standard errors (6 lags). Portfolio
    returns are compounded to monthly frequency to match the (monthly) factor
    panel, so ``alpha`` is a monthly abnormal return.

    Args:
        portfolio_returns: time series of (gross) portfolio returns with a
            recognized date column or DatetimeIndex and column ``portfolio_col``.
        factors: monthly factor panel with columns MKT_RF, SMB, HML, RMW, CMA,
            and RF.
        portfolio_col: name of the return column in ``portfolio_returns``.

    Returns:
        Dict with ``alpha`` (monthly abnormal return), ``alpha_tstat``,
        ``alpha_pvalue``, ``betas`` (dict keyed by factor), ``r_squared`` and
        ``n_months``.

    CRITICAL: alpha is the risk-adjusted abnormal return. If raw PEAD shows a
    4.5x asymmetry but FF5-adjusted shows 1.5x, the difference is factor
    exposure, not behavioral asymmetry.
    """
    merged = _aggregate_to_monthly(portfolio_returns, factors, portfolio_col, FF5_FACTORS)
    return _run_regression(merged, FF5_FACTORS)


def run_ff3_regression(
    portfolio_returns: pd.DataFrame,
    factors: pd.DataFrame,
    portfolio_col: str = Col.PORTFOLIO_RET_GROSS,
) -> dict[str, Any]:
    """Run the Fama-French 3-factor (MKT_RF, SMB, HML) time-series regression.

    Same mechanics as :func:`run_ff5_regression` (monthly compounding, Newey-West
    HAC) but restricted to the three classic factors — useful as a comparison
    benchmark to the 5-factor specification.
    """
    merged = _aggregate_to_monthly(portfolio_returns, factors, portfolio_col, FF3_FACTORS)
    return _run_regression(merged, FF3_FACTORS)


def compare_factor_adjusted_vs_raw(
    raw_results: dict[Any, Any],
    ff5_results: dict[Any, Any],
) -> pd.DataFrame:
    """Produce a comparison table of raw vs FF5-adjusted alpha per decile.

    Args:
        raw_results: mapping ``{decile -> {"alpha": ...}}`` (or ``{decile -> alpha}``).
        ff5_results: same shape, FF5-adjusted alphas.

    Returns:
        DataFrame with one row per decile and columns: ``decile``,
        ``raw_alpha``, ``ff5_adjusted_alpha``, ``alpha_change`` (FF5 − raw),
        ``pct_factor_explained`` (share of the raw alpha absorbed by factors,
        in %). ``pct_factor_explained`` is NaN when the raw alpha is ~0.
    """
    keys = sorted(set(raw_results) | set(ff5_results), key=_sort_key)
    rows: list[dict[str, Any]] = []
    for key in keys:
        raw_alpha = _coerce_alpha(raw_results.get(key))
        ff5_alpha = _coerce_alpha(ff5_results.get(key))
        change = ff5_alpha - raw_alpha
        if not np.isclose(raw_alpha, 0.0, atol=1e-12):
            pct_explained = (raw_alpha - ff5_alpha) / raw_alpha * 100.0
        else:
            pct_explained = float("nan")
        rows.append(
            {
                "decile": key,
                "raw_alpha": raw_alpha,
                "ff5_adjusted_alpha": ff5_alpha,
                "alpha_change": change,
                "pct_factor_explained": pct_explained,
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "decile",
            "raw_alpha",
            "ff5_adjusted_alpha",
            "alpha_change",
            "pct_factor_explained",
        ],
    )


# ─── Small internal utilities ────────────────────────────────────────────────


def _coerce_alpha(value: Any) -> float:
    """Best-effort extraction of a scalar alpha from heterogeneous inputs."""
    if value is None:
        return float("nan")
    if isinstance(value, dict):
        if "alpha" in value:
            return float(value["alpha"])
        if "alpha_monthly" in value:
            return float(value["alpha_monthly"])
        # take the first numeric value as a fallback
        for v in value.values():
            try:
                return float(v)
            except (TypeError, ValueError):
                continue
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _sort_key(key: Any) -> tuple[int, Any]:
    """Sort helper that keeps numeric deciles ordered but tolerates str keys."""
    try:
        return (0, float(key))
    except (TypeError, ValueError):
        return (1, str(key))


__all__ = [
    "FF5_FACTORS",
    "FF3_FACTORS",
    "NEWEY_WEST_LAGS",
    "TRADING_DAYS_PER_MONTH",
    "run_ff5_regression",
    "run_ff3_regression",
    "compare_factor_adjusted_vs_raw",
]
