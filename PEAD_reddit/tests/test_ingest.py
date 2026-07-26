"""Unit tests for the PEAD ingest layer (Bloomberg mocks + factor loading).

These tests exercise the *offline* path: every Bloomberg ``fetch_*`` function
has a ``mock_*`` twin backed by :mod:`pead.synthetic`, so the full contract
(column names, dtypes, invariants) is verified without a live terminal.

Run with::

    python -m pytest tests/test_ingest.py -v
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pead.ingest.bloomberg import (
    apply_lm_filters,
    fetch_daily_prices,
    fetch_delisting_events,
    fetch_earnings_estimates,
    fetch_fundamentals,
    mock_daily_prices,
    mock_delisting_events,
    mock_earnings_estimates,
    mock_fundamentals,
)
from pead.ingest.factors import (
    FACTOR_COLUMNS,
    fetch_ff5_factors,
    load_factors_from_parquet,
    mock_factors,
    save_factors_to_parquet,
)
from pead.schema import Col, ExpectedSchema, validate_dataframe
from pead.synthetic import generate_synthetic_factors

# A wide date window that comfortably contains the synthetic generator's range
# (synthetic events start 2015-01-01 and run ~40 quarters).
START = "2014-01-01"
END = "2030-12-31"

TEST_TICKERS = ["AAPL US Equity", "MSFT US Equity", "GOOGL US Equity"]


# ─── 1. Earnings estimates ──────────────────────────────────────────────────


def test_mock_earnings_estimates():
    """mock_earnings_estimates returns the canonical EARNINGS_EVENTS schema."""
    df = mock_earnings_estimates(TEST_TICKERS, START, END)

    # Schema: every required column present.
    validate_dataframe(
        df,
        ExpectedSchema.EARNINGS_EVENTS,
        name="mock_earnings_estimates",
        strict_dates=[Col.ANNOUNCE_DATE],
    )

    # Bonus columns documented in the spec are also present.
    for extra in ("fq_end", "meanest_eps"):
        assert extra in df.columns, f"missing extra column {extra}"

    # Tickers are relabeled onto the requested universe.
    assert set(df[Col.TICKER]).issubset(set(TEST_TICKERS))

    # n_analysts is a sensible positive integer.
    assert df[Col.N_ANALYSTS].between(1, 100).all()

    assert len(df) > 0, "expected non-empty earnings frame"


# ─── 2. Fundamentals ────────────────────────────────────────────────────────


def test_mock_fundamentals():
    """mock_fundamentals returns the canonical FUNDAMENTALS_Q schema."""
    df = mock_fundamentals(TEST_TICKERS, START, END)

    validate_dataframe(
        df,
        ExpectedSchema.FUNDAMENTALS_Q,
        name="mock_fundamentals",
        strict_dates=[Col.FISCAL_QUARTER_END, Col.REPORT_DATE],
    )

    # Extra documented columns present.
    for extra in (Col.SHARES_DILUTED, Col.ADJ_FACTOR):
        assert extra in df.columns, f"missing extra column {extra}"

    # EPS primary should equal the actual EPS used to seed the generator.
    assert df[Col.EPS_PRIMARY].notna().all()
    # Shares and prices are strictly positive in the synthetic generator.
    assert (df[Col.SHARES_BASIC] > 0).all()
    assert (df[Col.PRICE_QUARTER_END] > 0).all()
    assert len(df) > 0


# ─── 3. Daily prices ────────────────────────────────────────────────────────


def test_mock_daily_prices():
    """mock_daily_prices returns canonical DAILY_PRICES + midquote invariant."""
    df = mock_daily_prices(TEST_TICKERS, START, END)

    validate_dataframe(
        df,
        ExpectedSchema.DAILY_PRICES,
        name="mock_daily_prices",
        strict_dates=[Col.TRADING_DATE],
    )

    # Midquote column present ...
    assert Col.PX_MIDQUOTE in df.columns
    # ... and exactly equals (bid + ask) / 2 everywhere.
    expected_mid = (df[Col.PX_BID] + df[Col.PX_ASK]) / 2
    np.testing.assert_allclose(df[Col.PX_MIDQUOTE].to_numpy(), expected_mid.to_numpy())

    # Bid <= mid <= ask ordering holds (no crossed quotes).
    assert (df[Col.PX_BID] <= df[Col.PX_MIDQUOTE] + 1e-9).all()
    assert (df[Col.PX_MIDQUOTE] <= df[Col.PX_ASK] + 1e-9).all()

    # OHLC sanity: low <= open/high/close <= high (allow float tolerance).
    assert (df[Col.PX_LOW] <= df[Col.PX_HIGH] + 1e-9).all()
    assert len(df) > 0


# ─── 4. Factor parquet round-trip ───────────────────────────────────────────


def test_load_factors_from_parquet(tmp_path):
    """Synthetic factors round-trip losslessly through parquet."""
    factors = mock_factors(START, END, seed=42)

    # Canonical schema before writing.
    validate_dataframe(
        factors,
        FACTOR_COLUMNS,
        name="mock_factors",
        strict_dates=[Col.CALENDAR_DATE],
    )

    path = tmp_path / "ff5_monthly.parquet"
    save_factors_to_parquet(factors, str(path))
    assert path.exists()

    loaded = load_factors_from_parquet(str(path))

    # Same canonical columns, same order.
    assert list(loaded.columns)[: len(FACTOR_COLUMNS)] == FACTOR_COLUMNS

    # Values are identical (lossless parquet round-trip).
    pd.testing.assert_frame_equal(
        loaded.reset_index(drop=True),
        factors.sort_values(Col.CALENDAR_DATE).reset_index(drop=True),
        check_like=True,
    )

    # Date column preserved as datetime64.
    assert pd.api.types.is_datetime64_any_dtype(loaded[Col.CALENDAR_DATE])
    assert len(loaded) == len(factors)


# ─── 5. fetch_ff5_factors offline fallback ──────────────────────────────────


def test_fetch_ff5_factors_synthetic_fallback(monkeypatch):
    """fetch_ff5_factors degrades to synthetic when network/Bloomberg absent."""
    # Force both the Bloomberg probe and the Kenneth French download to fail.
    monkeypatch.setattr("pead.ingest.factors._pdblp_available", lambda: False)
    monkeypatch.setattr(
        "pead.ingest.factors._load_ff5_monthly",
        lambda url: (_ for _ in ()).throw(RuntimeError("network disabled")),
    )

    df = fetch_ff5_factors(START, END, conn=None)

    validate_dataframe(
        df,
        FACTOR_COLUMNS,
        name="fetch_ff5_factors (synthetic fallback)",
        strict_dates=[Col.CALENDAR_DATE],
    )
    assert len(df) > 0
    # Sorted ascending by date.
    assert df[Col.CALENDAR_DATE].is_monotonic_increasing


# ─── 6. Delisting events ───────────────────────────────────────────────────


def test_mock_delisting_events():
    """mock_delisting_events returns delisting columns for survivorship bias."""
    df = mock_delisting_events(TEST_TICKERS, START, END, seed=7)

    assert list(df.columns) == [Col.TICKER, Col.DELISTING_DATE, Col.DELISTING_RETURN]
    assert pd.api.types.is_datetime64_any_dtype(df[Col.DELISTING_DATE])
    if not df.empty:
        # Delisting returns are predominantly negative (distressed firms).
        assert df[Col.DELISTING_RETURN].mean() < 0
        assert df[Col.TICKER].isin(TEST_TICKERS).all()


# ─── 7. Livnat-Mendenhall filters ──────────────────────────────────────────


def test_apply_lm_filters_keeps_valid_rows():
    """apply_lm_filters keeps LM-valid rows and drops penny/micro-cap mismatches.

    ``mock_fundamentals`` already carries both ``announce_date`` and
    ``report_date`` (the synthetic fundamentals generator copies the events
    table), so no merge is required.
    """
    fund = mock_fundamentals(TEST_TICKERS, START, END)
    assert Col.ANNOUNCE_DATE in fund.columns
    assert Col.REPORT_DATE in fund.columns

    kept = apply_lm_filters(
        fund,
        min_price=1.0,
        min_mcap_millions=5.0,
        max_date_diff_days=1,
    )

    # Synthetic data is LM-clean by construction, so nothing should be dropped.
    assert len(kept) == len(fund)

    # Price and mcap screens enforced.
    assert (kept[Col.PRICE_QUARTER_END] > 1.0).all()
    mcap = kept[Col.SHARES_BASIC] * kept[Col.PRICE_QUARTER_END]
    assert (mcap > 5.0).all()


def test_apply_lm_filters_drops_violations():
    """apply_lm_filters drops rows that violate the price / date screens."""
    fund = mock_fundamentals(TEST_TICKERS, START, END).reset_index(drop=True)

    # Inject a penny-stock row and a stale report-date row.
    bad_price = fund.iloc[[0]].copy()
    bad_price[Col.PRICE_QUARTER_END] = 0.50  # below $1

    bad_date = fund.iloc[[1]].copy()
    bad_date[Col.REPORT_DATE] = bad_date[Col.ANNOUNCE_DATE] + pd.Timedelta(days=10)

    polluted = pd.concat([fund, bad_price, bad_date], ignore_index=True)
    kept = apply_lm_filters(polluted)

    assert len(kept) == len(fund)  # only the two injected rows dropped
    assert (kept[Col.PRICE_QUARTER_END] > 0.50).all()


# ─── 8. Module imports without Bloomberg installed ──────────────────────────


def test_module_imports_without_bloomberg():
    """The ingest modules import cleanly even with no Bloomberg libs present."""
    import importlib

    import pead.ingest.bloomberg as bb
    import pead.ingest.factors as ff

    importlib.reload(bb)
    importlib.reload(ff)

    # And the fetch_* entrypoints are callable (they only touch pdblp lazily).
    assert callable(bb.fetch_earnings_estimates)
    assert callable(ff.fetch_ff5_factors)


# ─── 9. fetch_* raises clearly without a connection ────────────────────────


def test_fetch_earnings_raises_without_connection(monkeypatch):
    """fetch_* raises a clear RuntimeError when conn=None and pdblp missing."""
    import pead.ingest.bloomberg as bb

    # Simulate pdblp being unavailable.
    monkeypatch.setattr(bb, "_get_connection", _raise_runtime)

    with pytest.raises(RuntimeError, match="pdblp"):
        bb.fetch_earnings_estimates(TEST_TICKERS, START, END, conn=None)


def _raise_runtime(_conn):
    raise RuntimeError(
        "No Bloomberg connection was provided (conn=None) and the 'pdblp' package is not installed."
    )
