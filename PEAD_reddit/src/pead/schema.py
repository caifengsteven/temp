"""
Canonical schemas for the PEAD asymmetry pipeline.

Every module communicates via pandas DataFrames with standardized column names
defined here. This is the CONTRACT between modules — changing a column name
here without updating consumers will break the pipeline.

Data flow:
  Raw Bloomberg → EarningsEvents / FundamentalsQ / DailyPrices
                → SUE table (sue1, sue2, sue3)
                → EventReturns (CAR, BHAR, mid-quote)
                → Portfolios (decile assignments, CTP returns)
                → AsymmetryResults (ratio, CI, t-stat)

Reference: Livnat & Mendenhall (2006), Bernard & Thomas (1989/1990),
           Mitchell & Stafford (2000), Lyon Barber Tsai (1999).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import pandas as pd


# ─── Column name constants ──────────────────────────────────────────────────


class Col:
    """Standardized column names. Use these everywhere — no magic strings."""

    # Identifiers
    TICKER = "ticker"  # Bloomberg ticker, e.g. "AAPL US Equity"
    PERMNO = "permno"  # CRSP permno (if available)
    GVKEY = "gvkey"  # Compustat gvkey (if available)
    CUSIP = "cusip"
    COMPANY_NAME = "company_name"

    # Dates
    ANNOUNCE_DATE = "announce_date"  # Earnings announcement date (trading day)
    FISCAL_QUARTER = "fiscal_quarter"  # e.g. "2023Q1"
    FISCAL_QUARTER_END = "fq_end"  # Calendar end of fiscal quarter
    REPORT_DATE = "report_date"  # Compustat rdq (may differ from announce)
    TRADING_DATE = "trading_date"  # A CRSP/Bloomberg trading day

    # Earnings data
    ACTUAL_EPS = "actual_eps"  # Reported EPS (as-of announce, split-adjusted)
    MEDEST_EPS = "medest_eps"  # Median analyst forecast (SUE3)
    MEANEST_EPS = "meanest_eps"  # Mean analyst forecast
    N_ANALYSTS = "n_analysts"  # Number of analysts in consensus
    EPS_PRIMARY = "eps_primary"  # Compustat epspxq (primary)
    EPS_DILUTED = "eps_diluted"  # Compustat epsfxq (diluted)
    SPECIAL_ITEMS = "special_items"  # Compustat spiq
    SHARES_BASIC = "shares_basic"  # cshoq
    SHARES_DILUTED = "shares_diluted"  # cshfdq
    PRICE_QUARTER_END = "price_qe"  # prccq (unadjusted quarter-end)
    ADJ_FACTOR = "adj_factor"  # CRSP cfacshr / Bloomberg split adj

    # SUE
    SUE1 = "sue1"  # Random walk: (EPS_t - EPS_t-4) / P
    SUE2 = "sue2"  # Excluding special items
    SUE3 = "sue3"  # Analyst-median based
    SUE1_DECILE = "sue1_decile"  # 1-10, per cross-section
    SUE3_DECILE = "sue3_decile"  # 1-10, per cross-section
    SUE3_QUINTILE = "sue3_quintile"  # 1-5, per cross-section

    # Prices
    PX_OPEN = "px_open"
    PX_HIGH = "px_high"
    PX_LOW = "px_low"
    PX_CLOSE = "px_close"
    PX_BID = "px_bid"
    PX_ASK = "px_ask"
    PX_MIDQUOTE = "px_midquote"  # (bid + ask) / 2
    VOLUME = "volume"
    RET = "ret"  # Daily return (close-to-close)
    RET_MIDQUOTE = "ret_midquote"  # Daily return (midquote-to-midquote)
    RET_EXCESS = "ret_excess"  # Excess of risk-free

    # Market model
    ALPHA = "alpha"  # Market model intercept
    BETA = "beta"  # Market model slope
    AR = "ar"  # Abnormal return
    CAR = "car"  # Cumulative abnormal return
    BHAR = "bhar"  # Buy-and-hold abnormal return

    # Event classification
    IS_MISS = "is_miss"  # Bottom extreme decile (large miss)
    IS_BEAT = "is_beat"  # Top extreme decile (large beat)
    EVENT_WEEK = "event_week"  # Calendar week of announce_date
    MARKET_CAP_BUCKET = "mcap_bucket"  # Size quintile at announce_date
    AMIHUD_ILLIQUIDITY = "amihud"  # Amihud (2002) illiquidity measure

    # Portfolio
    DECILE = "decile"  # 1-10
    WEIGHT = "weight"  # Portfolio weight
    PORTFOLIO_RET_GROSS = "port_ret_gross"
    PORTFOLIO_RET_NET = "port_ret_net"  # After transaction costs
    CALENDAR_DATE = "calendar_date"  # CTP holding date
    N_HOLDINGS = "n_holdings"

    # Factors
    MKT_RF = "mkt_rf"
    SMB = "smb"
    HML = "hml"
    RMW = "rmw"
    CMA = "cma"
    RF = "rf"
    MOM = "mom"

    # Firm characteristics (for DGTW)
    MARKET_CAP = "market_cap"  # ME = price × shares
    BOOK_TO_MARKET = "btm"  # BE/ME
    MOMENTUM_12_2 = "mom_12_2"  # 12-2 month momentum
    DGTW_BUCKET = "dgtw_bucket"  # Assigned DGTW portfolio (0-124)

    # Delisting
    DELISTING_DATE = "delisting_date"
    DELISTING_RETURN = "delisting_return"

    # SUE method selector
    SUE_METHOD = "sue_method"  # "sue1" | "sue2" | "sue3"


# ─── Window spec ────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Window:
    """An event-study window in trading days relative to announcement (day 0)."""

    name: str
    start: int  # Relative to announce_date (0 = announce day)
    end: int  # Inclusive

    def __post_init__(self) -> None:
        if self.start > self.end:
            raise ValueError(f"Window '{self.name}': start ({self.start}) > end ({self.end})")

    @property
    def length(self) -> int:
        return self.end - self.start + 1

    def __repr__(self) -> str:
        return f"Window({self.name}, [{self.start}, {self.end}])"


# ─── Expected schemas (for validation) ──────────────────────────────────────


class ExpectedSchema:
    """
    Document the expected columns for each major DataFrame in the pipeline.
    Used by validate_dataframe() to catch schema violations early.
    """

    EARNINGS_EVENTS = [
        Col.TICKER,
        Col.ANNOUNCE_DATE,
        Col.FISCAL_QUARTER,
        Col.ACTUAL_EPS,
        Col.MEDEST_EPS,
        Col.N_ANALYSTS,
    ]

    FUNDAMENTALS_Q = [
        Col.TICKER,
        Col.FISCAL_QUARTER_END,
        Col.FISCAL_QUARTER,
        Col.EPS_PRIMARY,
        Col.EPS_DILUTED,
        Col.SPECIAL_ITEMS,
        Col.SHARES_BASIC,
        Col.PRICE_QUARTER_END,
        Col.REPORT_DATE,
    ]

    DAILY_PRICES = [
        Col.TICKER,
        Col.TRADING_DATE,
        Col.PX_OPEN,
        Col.PX_HIGH,
        Col.PX_LOW,
        Col.PX_CLOSE,
        Col.PX_BID,
        Col.PX_ASK,
        Col.VOLUME,
        Col.RET,
    ]

    SUE_TABLE = [
        Col.TICKER,
        Col.ANNOUNCE_DATE,
        Col.FISCAL_QUARTER,
        Col.SUE1,
        Col.SUE2,
        Col.SUE3,
    ]

    EVENT_RETURNS = [
        Col.TICKER,
        Col.ANNOUNCE_DATE,
        Col.SUE_METHOD,
        Col.DECILE,
    ]

    PORTFOLIO_RETURNS = [
        Col.CALENDAR_DATE,
        Col.SUE_METHOD,
        Col.DECILE,
        Col.PORTFOLIO_RET_GROSS,
        Col.N_HOLDINGS,
    ]

    ASYMMETRY_RESULT = [
        Col.SUE_METHOD,
        "window_name",
        "miss_car_mean",
        "beat_car_mean",
        "ratio",
        "ci_lower",
        "ci_upper",
        "t_stat",
        "p_value",
        "n_miss",
        "n_beat",
    ]


# ─── Validation ─────────────────────────────────────────────────────────────


def validate_dataframe(
    df: pd.DataFrame,
    required_cols: list[str],
    name: str = "DataFrame",
    strict_dates: list[str] | None = None,
) -> None:
    """
    Validate that a DataFrame has the required columns.

    Args:
        df: DataFrame to check.
        required_cols: Columns that must be present.
        name: Human-readable name for error messages.
        strict_dates: Columns that should be datetime64[ns].

    Raises:
        ValueError: If required columns are missing or dates are wrong dtype.
    """
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}")

    if strict_dates:
        for col in strict_dates:
            if col in df.columns and not pd.api.types.is_datetime64_any_dtype(df[col]):
                raise TypeError(f"{name} column '{col}' must be datetime64, got {df[col].dtype}")


# ─── Constants ──────────────────────────────────────────────────────────────

TRADING_DAYS_PER_YEAR = 252
TRADING_DAYS_PER_MONTH = 21


# SUE method labels (used throughout)
class SUEMethod(str, Enum):
    """SUE construction method selector."""

    SUE1 = "sue1"  # Seasonal random walk (Compustat only)
    SUE2 = "sue2"  # Random walk ex-special items
    SUE3 = "sue3"  # Analyst median forecast (I/B/E/S equivalent)


# ─── Config loader ──────────────────────────────────────────────────────────


@dataclass
class PipelineConfig:
    """Type-safe config loaded from config/config.yaml."""

    raw: dict[str, Any] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str = "config/config.yaml") -> PipelineConfig:
        import yaml

        with open(path) as f:
            data = yaml.safe_load(f)
        obj = cls(config=data)
        obj.raw = data
        return obj

    def get(self, *keys: str, default: Any = None) -> Any:
        """Nested dict access: config.get('event_study', 'use_mid_quote')."""
        result: Any = self.raw
        for key in keys:
            if not isinstance(result, dict):
                return default
            result = result.get(key, default)
        return result
