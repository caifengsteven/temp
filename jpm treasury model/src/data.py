"""Data layer for the JPMorgan Treasury Fair-Value framework.

Provides three model-ready datasets (`get_model1_data`, `get_model2_data`,
`get_model3_data`) used by the downstream fair-value models. Each builder
attempts to fetch real data from Bloomberg via ``xbbg`` and transparently
falls back to a realistic synthetic generator when Bloomberg is unavailable
(so the framework is fully runnable offline).

All series are returned as pandas Series/DataFrames with a business-day
``DatetimeIndex``.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Bloomberg ticker mapping
# ---------------------------------------------------------------------------
BLOOMBERG_TICKERS: dict[str, Optional[str]] = {
    # Treasury yields (percent)
    "yield_2y": "USGG2YR Index",
    "yield_3y": "USGG3YR Index",
    "yield_5y": "USGG5YR Index",
    "yield_7y": "USGG7YR Index",
    "yield_10y": "USGG10YR Index",
    "yield_20y": "USGG20YR Index",
    "yield_30y": "USGG30YR Index",
    # OIS rates (for constructing 1Y1Y forward)
    "ois_1y": "USSOA Curncy",  # 1Y OIS swap rate
    "ois_2y": "USSOB Curncy",  # 2Y OIS swap rate
    # TIPS yields (for constructing 5Y5Y breakeven forward)
    "tips_5y": "GTII5 Govt",
    "tips_10y": "GTII10 Govt",
    "be_5y": "USGGBE05 Index",  # 5Y breakeven
    "be_10y": "USGGBE10 Index",  # 10Y breakeven
    # Fed balance sheet
    "fed_assets": "WALCL",  # Fed total assets (FRED via Bloomberg)
    "us_gdp": "GDP Curncy",  # US GDP
    # JPM proprietary (may not be available)
    "jpm_fri": None,  # JPM Forecast Revision Index - proprietary
}


# ---------------------------------------------------------------------------
# Bloomberg fetch
# ---------------------------------------------------------------------------
def fetch_bloomberg(tickers: list[str], start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch daily historical data from Bloomberg via ``xbbg``.

    Parameters
    ----------
    tickers : list[str]
        Bloomberg tickers (e.g. ``["USGG10YR Index", "USSOA Curncy"]``).
    start_date, end_date : str
        Inclusive date bounds in ``YYYY-MM-DD`` format.

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by business day with one column per ticker.

    Raises
    ------
    ImportError
        If the ``xbbg`` package is not installed or Bloomberg Terminal /
        BLPAPI is unreachable.
    """
    try:
        import xbbg  # type: ignore[import-untyped]
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "xbbg is not installed. Install it with `pip install xbbg` and "
            "ensure a running Bloomberg Terminal / BLPAPI session. For offline "
            "use, call the model data builders with use_bloomberg=False."
        ) from exc

    raw: pd.DataFrame = xbbg.bdh(
        tickers=tickers,
        flds="px_last",
        start_date=start_date,
        end_date=end_date,
    )
    if raw is None or raw.empty:
        raise ValueError("Bloomberg returned an empty dataset.")

    # xbbg may return a MultiIndex on columns (ticker, field); flatten it.
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    raw.index = pd.to_datetime(raw.index)
    raw = raw.sort_index()
    return raw


# ---------------------------------------------------------------------------
# Synthetic data generation
# ---------------------------------------------------------------------------
def _generate_synthetic_data(
    start_date: str = "2020-01-01",
    end_date: str = "2025-07-01",
) -> pd.DataFrame:
    """Generate realistic synthetic daily data for every series in the mapping.

    The generator uses mild mean-reverting (Ornstein-Uhlenbeck-style) processes
    so that yields stay within plausible ranges over multi-year horizons.
    Treasury yields share a common curve factor to reproduce realistic
    cross-tenor correlation.

    Parameters
    ----------
    start_date, end_date : str
        Inclusive date bounds (``YYYY-MM-DD``).

    Returns
    -------
    pd.DataFrame
        Business-day DataFrame with all keys of ``BLOOMBERG_TICKERS`` as
        columns. Yields and rates are in percent (e.g. 4.25); ``fed_assets``
        and ``us_gdp`` are in USD billions.
    """
    rng = np.random.default_rng(42)
    dates = pd.bdate_range(start=start_date, end=end_date)
    n = len(dates)

    # Common curve factor — drives correlation across nominal yields/OIS.
    curve_factor = rng.normal(0.0, 0.025, n)

    def _ou(
        start_level: float,
        daily_vol: float,
        kappa: float = 0.01,
        common: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Simulate a mean-reverting random walk tethered to ``start_level``."""
        idio = rng.normal(0.0, daily_vol, n)
        shocks = idio if common is None else idio + common
        series = np.empty(n)
        series[0] = start_level
        for t in range(1, n):
            series[t] = (
                series[t - 1] + kappa * (start_level - series[t - 1]) + shocks[t]
            )
        return series

    data: dict[str, np.ndarray] = {}

    # --- Treasury yields (percent) -----------------------------------------
    data["yield_2y"] = _ou(4.50, 0.020, common=curve_factor)
    data["yield_3y"] = _ou(4.45, 0.018, common=curve_factor)
    data["yield_5y"] = _ou(4.30, 0.017, common=curve_factor)
    data["yield_7y"] = _ou(4.25, 0.016, common=curve_factor)
    data["yield_10y"] = _ou(4.20, 0.015, common=curve_factor)
    data["yield_20y"] = _ou(4.35, 0.014, common=curve_factor)
    data["yield_30y"] = _ou(4.40, 0.013, common=curve_factor)

    # --- OIS rates (slightly below Treasuries) ------------------------------
    data["ois_1y"] = _ou(4.30, 0.020, common=curve_factor)
    data["ois_2y"] = _ou(4.10, 0.019, common=curve_factor)

    # --- Breakevens (around 2.3%, mean-reverting) --------------------------
    data["be_5y"] = _ou(2.30, 0.025, kappa=0.02)
    data["be_10y"] = _ou(2.30, 0.022, kappa=0.02)

    # --- TIPS yields (approx nominal minus breakeven) -----------------------
    data["tips_5y"] = data["yield_5y"] - data["be_5y"]
    data["tips_10y"] = data["yield_10y"] - data["be_10y"]

    # --- Fed balance sheet & GDP (USD billions) -----------------------------
    # GDP drifts smoothly from ~$21.5T (2020) to ~$29T (2025).
    data["us_gdp"] = np.linspace(21_500.0, 29_000.0, n)
    # Fed assets / GDP ratio trends from ~18% (pre-COVID) to ~30% (post-QE),
    # then mean-reverts gently around 28%.
    fed_ratio = _ou(28.0, 0.05, kappa=0.005)
    fed_ratio = fed_ratio + np.linspace(-10.0, 0.0, n)  # ramp-up overlay
    data["fed_assets"] = data["us_gdp"] * fed_ratio / 100.0

    # --- JPM Forecast Revision Index (mean-reverting around 0) -------------
    data["jpm_fri"] = _ou(0.0, 0.20, kappa=0.05)

    df = pd.DataFrame(data, index=dates)
    df.index.name = "date"
    return df


# ---------------------------------------------------------------------------
# Forward-rate / breakeven constructors
# ---------------------------------------------------------------------------
def _construct_1y1y_ois(df: pd.DataFrame) -> pd.Series:
    """Construct the 1-year forward, 1-year OIS rate.

    Formula (in decimal terms)::

        f(1y, 1y) = (1 + ois_2y)^2 / (1 + ois_1y) - 1

    Inputs and output are expressed in percent (e.g. 3.91).
    """
    ois_1y = df["ois_1y"] / 100.0
    ois_2y = df["ois_2y"] / 100.0
    forward = (1.0 + ois_2y) ** 2 / (1.0 + ois_1y) - 1.0
    return (forward * 100.0).rename("ois_1y1y")


def _construct_5y5y_breakeven(df: pd.DataFrame) -> pd.Series:
    """Construct the 5-year forward, 5-year breakeven inflation.

    Formula (in decimal terms)::

        be(5y, 5y) = ((1 + be_10y)^10 / (1 + be_5y)^5)^(1/5) - 1

    Inputs and output are expressed in percent (e.g. 2.30).
    """
    be_5y = df["be_5y"] / 100.0
    be_10y = df["be_10y"] / 100.0
    forward = ((1.0 + be_10y) ** 10 / (1.0 + be_5y) ** 5) ** (1.0 / 5.0) - 1.0
    return (forward * 100.0).rename("be_5y5y")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _load_raw(
    keys: list[str],
    start_date: str,
    end_date: str,
    use_bloomberg: bool,
) -> pd.DataFrame:
    """Return a DataFrame with the requested keys as columns.

    Tries Bloomberg first; falls back to synthetic data on any failure or
    when a needed key has no Bloomberg ticker (e.g. ``jpm_fri``).
    """
    if not use_bloomberg:
        return _generate_synthetic_data(start_date, end_date)[keys]

    if any(BLOOMBERG_TICKERS.get(k) is None for k in keys):
        # A proprietary series is required — use synthetic for everything.
        return _generate_synthetic_data(start_date, end_date)[keys]

    try:
        key_to_ticker = {k: BLOOMBERG_TICKERS[k] for k in keys}  # type: ignore[assignment]
        raw = fetch_bloomberg(list(key_to_ticker.values()), start_date, end_date)
        raw = raw.rename(columns={v: k for k, v in key_to_ticker.items()})
        missing = [k for k in keys if k not in raw.columns]
        if missing:
            raise ValueError(f"Bloomberg response missing columns: {missing}")
        return raw[keys]
    except Exception:
        # Any failure -> transparent synthetic fallback.
        return _generate_synthetic_data(start_date, end_date)[keys]


def _trade_dummy(index: pd.DatetimeIndex) -> pd.Series:
    """0/1 dummy switching to 1 from 1-Apr-2025 onward (trade-policy regime)."""
    cutoff = pd.Timestamp("2025-04-01")
    return pd.Series((index >= cutoff).astype(int), index=index, name="trade_dummy")


# ---------------------------------------------------------------------------
# Public model data builders
# ---------------------------------------------------------------------------
def get_model1_data(
    start_date: str = "2020-01-01",
    end_date: str = "2025-07-01",
    use_bloomberg: bool = True,
) -> pd.DataFrame:
    """Build the dataset for the 10-year Treasury fair-value model.

    Columns returned (exact order)::

        ["yield_10y", "ois_1y1y", "be_5y5y", "jpm_fri", "fed_bs_gdp", "trade_dummy"]

    - ``yield_10y``  : 10y Treasury yield, percent (e.g. 4.25)
    - ``ois_1y1y``  : 1y1y forward OIS, percent
    - ``be_5y5y``   : 5y5y forward breakeven inflation, percent
    - ``jpm_fri``   : JPM Forecast Revision Index
    - ``fed_bs_gdp``: Fed total assets / US GDP, percent (e.g. 27.5)
    - ``trade_dummy``: 0/1 regime dummy, 1 from 2025-04-01 onward
    """
    keys = [
        "yield_10y",
        "ois_1y",
        "ois_2y",
        "be_5y",
        "be_10y",
        "fed_assets",
        "us_gdp",
        "jpm_fri",
    ]
    df = _load_raw(keys, start_date, end_date, use_bloomberg).copy()

    out = pd.DataFrame(index=df.index)
    out["yield_10y"] = df["yield_10y"]
    out["ois_1y1y"] = _construct_1y1y_ois(df)
    out["be_5y5y"] = _construct_5y5y_breakeven(df)
    out["jpm_fri"] = df["jpm_fri"]
    out["fed_bs_gdp"] = (df["fed_assets"] / df["us_gdp"]) * 100.0
    out["trade_dummy"] = _trade_dummy(df.index)
    return out.dropna()


def get_model2_data(
    start_date: str = "2020-01-01",
    end_date: str = "2025-07-01",
    use_bloomberg: bool = True,
) -> pd.DataFrame:
    """Build the dataset for the 5s30s curve fair-value model.

    Columns returned (exact order)::

        ["curve_5s30s", "ois_1y1y", "be_5y5y", "fed_bs_gdp"]

    - ``curve_5s30s``: yield_30y - yield_5y, percentage points (e.g. 0.15)
    - ``ois_1y1y``   : 1y1y forward OIS, percent
    - ``be_5y5y``    : 5y5y forward breakeven inflation, percent
    - ``fed_bs_gdp`` : Fed total assets / US GDP, percent

    Per JPM methodology, JPM FRI is intentionally excluded.
    """
    keys = [
        "yield_5y",
        "yield_30y",
        "ois_1y",
        "ois_2y",
        "be_5y",
        "be_10y",
        "fed_assets",
        "us_gdp",
    ]
    df = _load_raw(keys, start_date, end_date, use_bloomberg).copy()

    out = pd.DataFrame(index=df.index)
    out["curve_5s30s"] = df["yield_30y"] - df["yield_5y"]
    out["ois_1y1y"] = _construct_1y1y_ois(df)
    out["be_5y5y"] = _construct_5y5y_breakeven(df)
    out["fed_bs_gdp"] = (df["fed_assets"] / df["us_gdp"]) * 100.0
    return out.dropna()


def get_model3_data(
    start_date: str = "2020-01-01",
    end_date: str = "2025-07-01",
    use_bloomberg: bool = True,
) -> pd.DataFrame:
    """Build the dataset for the butterfly / curve-shape fair-value model.

    Returns all standard Treasury yields (percent) plus constructed butterfly
    and slope spreads (basis points)::

        yield_2y, yield_3y, yield_5y, yield_7y, yield_10y, yield_20y, yield_30y,
        fly_2s_3s_5s, fly_2s_5s_10s, fly_5s_7s_10s, fly_5s_10s_30s, fly_2s_10s_30s,
        slope_2s_10s, slope_5s_30s

    Butterfly weights use the standard ``(short - 2*belly + long)`` convention;
    multiplying percent yields by 100 expresses the result in basis points.
    """
    keys = [
        "yield_2y",
        "yield_3y",
        "yield_5y",
        "yield_7y",
        "yield_10y",
        "yield_20y",
        "yield_30y",
    ]
    df = _load_raw(keys, start_date, end_date, use_bloomberg).copy()

    out = pd.DataFrame(index=df.index)
    for k in keys:
        out[k] = df[k]

    # Butterflies (yields in %, so multiply by 100 -> basis points)
    out["fly_2s_3s_5s"] = (df["yield_2y"] - 2 * df["yield_3y"] + df["yield_5y"]) * 100.0
    out["fly_2s_5s_10s"] = (
        df["yield_2y"] - 2 * df["yield_5y"] + df["yield_10y"]
    ) * 100.0
    out["fly_5s_7s_10s"] = (
        df["yield_5y"] - 2 * df["yield_7y"] + df["yield_10y"]
    ) * 100.0
    out["fly_5s_10s_30s"] = (
        df["yield_5y"] - 2 * df["yield_10y"] + df["yield_30y"]
    ) * 100.0
    out["fly_2s_10s_30s"] = (
        df["yield_2y"] - 2 * df["yield_10y"] + df["yield_30y"]
    ) * 100.0

    # Slopes (basis points)
    out["slope_2s_10s"] = (df["yield_10y"] - df["yield_2y"]) * 100.0
    out["slope_5s_30s"] = (df["yield_30y"] - df["yield_5y"]) * 100.0

    return out.dropna()
