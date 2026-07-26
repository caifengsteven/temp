"""
Unit tests for the portfolio construction & transaction-cost module.

Covers:
    1. Decile assignment correctness (1..10, decile 1 = lowest SUE).
    2. Cross-sectional (within-group) sorting.
    3. Value-weight vs equal-weight divergence.
    4. Transaction costs reduce gross returns.
    5. Borrow cost applies only to the short (bottom) decile.
    6. Calendar-time portfolio smoke test.
    7. Long-short return computation.
    8. Amihud illiquidity monotonicity.
    9. Global-sort look-ahead warning.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from pead.portfolios.calendar_time import (
    build_calendar_time_portfolio,
    compute_long_short_returns,
)
from pead.portfolios.costs import (
    apply_transaction_costs,
    compute_amihud_illiquidity,
)
from pead.portfolios.sort import (
    assign_portfolio_deciles,
    compute_portfolio_returns,
)
from pead.schema import Col
from pead.synthetic import generate_all_synthetic_data


# ─── Fixtures ───────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def synthetic() -> dict[str, pd.DataFrame]:
    """Module-cached synthetic dataset (small enough for fast tests)."""
    return generate_all_synthetic_data(n_tickers=80, n_quarters=12, seed=42)


def _build_sue_table(events: pd.DataFrame, fundamentals: pd.DataFrame) -> pd.DataFrame:
    """Build a SUE3 table: (actual_eps - medest_eps) / price_qe."""
    fund = fundamentals[["ticker", "announce_date", "fq_end", "fiscal_quarter", "price_qe"]].copy()
    sue = events.merge(
        fund, on=["ticker", "announce_date", "fq_end", "fiscal_quarter"], how="inner"
    )
    sue["sue3"] = (sue["actual_eps"] - sue["medest_eps"]) / sue["price_qe"]
    # Add placeholder SUE1/SUE2 so the schema is complete.
    sue["sue1"] = sue["sue3"]
    sue["sue2"] = sue["sue3"]
    return sue


def _build_event_returns(
    events: pd.DataFrame,
    fundamentals: pd.DataFrame,
    prices: pd.DataFrame,
) -> pd.DataFrame:
    """Build event-level returns with a CAR column over [1, 20] and market cap.

    market_cap = price_qe * shares_basic (in $M).
    """
    px = prices[["ticker", "trading_date", "ret"]].copy()
    px["trading_date"] = pd.to_datetime(px["trading_date"])

    rows = []
    for _, ev in events.iterrows():
        tkr = ev["ticker"]
        announce = ev["announce_date"]
        sub = px[(px["ticker"] == tkr) & (px["trading_date"] > announce)].sort_values(
            "trading_date"
        )
        if len(sub) < 20:
            continue
        car = float(sub["ret"].iloc[0:20].sum())
        rows.append({"ticker": tkr, "announce_date": announce, "car": car})

    er = pd.DataFrame(rows)

    fund = fundamentals[["ticker", "announce_date", "price_qe", "shares_basic"]].copy()
    fund["market_cap"] = fund["price_qe"] * fund["shares_basic"]
    er = er.merge(fund, on=["ticker", "announce_date"], how="inner")
    return er


@pytest.fixture(scope="module")
def sue_table(synthetic) -> pd.DataFrame:
    return _build_sue_table(synthetic["earnings_events"], synthetic["fundamentals"])


@pytest.fixture(scope="module")
def event_returns(synthetic) -> pd.DataFrame:
    return _build_event_returns(
        synthetic["earnings_events"],
        synthetic["fundamentals"],
        synthetic["daily_prices"],
    )


@pytest.fixture(scope="module")
def prices(synthetic) -> pd.DataFrame:
    return synthetic["daily_prices"]


@pytest.fixture(scope="module")
def assignments(sue_table, event_returns) -> pd.DataFrame:
    """Standard decile assignments using within-event_week sorting."""
    return assign_portfolio_deciles(
        sue_table,
        event_returns,
        sue_col="sue3",
        n_portfolios=10,
        sort_group_cols=["event_week"],
    )


# ─── 1. Decile assignment ───────────────────────────────────────────────────


def test_decile_assignment(assignments):
    """Deciles are 1..10, decile 1 has the lowest mean SUE."""
    dec = assignments["decile"].dropna().astype(int)
    assert set(dec.unique()).issubset(set(range(1, 11))), "Deciles outside 1..10"
    assert dec.min() >= 1
    assert dec.max() <= 10

    by_decile = assignments.groupby("decile")["sue3"].mean()
    assert by_decile.loc[1] < by_decile.loc[10], (
        f"Decile 1 mean SUE ({by_decile.loc[1]:.4f}) should be < "
        f"decile 10 ({by_decile.loc[10]:.4f})"
    )


# ─── 2. Cross-sectional sort ────────────────────────────────────────────────


def test_cross_sectional_sort(assignments):
    """Within each event_week, decile 1 mean SUE < decile 10 mean SUE."""
    grouped = assignments.groupby(["event_week", "decile"])["sue3"].mean().unstack()
    diffs = grouped[10] - grouped[1]
    assert (diffs > 0).sum() >= 0.9 * diffs.notna().sum(), (
        "Decile 1 should have lower mean SUE than decile 10 in most event_weeks; "
        f"only {(diffs > 0).sum()} of {diffs.notna().sum()} weeks satisfied."
    )


# ─── 3. Value-weight vs equal-weight ────────────────────────────────────────


def test_value_vs_equal_weight(assignments):
    """Value- and equal-weighted returns differ when market caps are heterogeneous."""
    mcap_spread = assignments.groupby("decile")["market_cap"].std().mean()
    assert mcap_spread > 0, "Market caps are constant; test is not meaningful."

    vw = compute_portfolio_returns(assignments, weight_scheme="value_weight")
    ew = compute_portfolio_returns(assignments, weight_scheme="equal_weight")

    merged = vw.merge(ew, on=["decile", "event_window"], suffixes=("_vw", "_ew"))
    assert not merged.empty, "compute_portfolio_returns returned empty frames."

    diffs = (merged["port_ret_gross_vw"] - merged["port_ret_gross_ew"]).abs()
    assert diffs.max() > 1e-9, (
        "Value-weight and equal-weight returns are identical even though "
        "market caps vary; value weighting is not being applied."
    )


# ─── 4. Transaction costs reduce returns ────────────────────────────────────


def test_transaction_costs_reduce_returns(assignments, prices):
    """port_ret_net < port_ret_gross for every decile."""
    gross = compute_portfolio_returns(assignments, weight_scheme="value_weight")
    net = apply_transaction_costs(
        gross,
        assignments,
        prices,
        commission_bps=5.0,
        slippage_base_bps=5.0,
        slippage_amihud_scale=1.0,
        borrow_cost_annual_bps=50.0,
        holding_days=20,
    )
    assert "port_ret_net" in net.columns
    finite = net["port_ret_net"].notna() & net["port_ret_gross"].notna()
    assert (net.loc[finite, "port_ret_net"] < net.loc[finite, "port_ret_gross"]).all(), (
        "Transaction costs should strictly reduce gross returns for every decile."
    )


# ─── 5. Borrow cost only on shorts ──────────────────────────────────────────


def test_borrow_cost_only_on_shorts(assignments, prices):
    """Borrow cost lowers the bottom decile but not the top decile."""
    gross = compute_portfolio_returns(assignments, weight_scheme="value_weight")

    net_no_borrow = apply_transaction_costs(
        gross,
        assignments,
        prices,
        commission_bps=5.0,
        slippage_base_bps=5.0,
        slippage_amihud_scale=1.0,
        borrow_cost_annual_bps=0.0,
        holding_days=20,
    )
    net_with_borrow = apply_transaction_costs(
        gross,
        assignments,
        prices,
        commission_bps=5.0,
        slippage_base_bps=5.0,
        slippage_amihud_scale=1.0,
        borrow_cost_annual_bps=500.0,  # 5% / year — extreme, to make impact obvious
        holding_days=20,
    )

    bottom = int(gross["decile"].min())
    top = int(gross["decile"].max())

    def _delta(df, decile):
        g = df[df["decile"] == decile].iloc[0]
        return g["port_ret_gross"] - g["port_ret_net"]

    delta_bottom_no = _delta(net_no_borrow, bottom)
    delta_bottom_with = _delta(net_with_borrow, bottom)
    delta_top_no = _delta(net_no_borrow, top)
    delta_top_with = _delta(net_with_borrow, top)

    borrow_impact_bottom = delta_bottom_with - delta_bottom_no
    borrow_impact_top = delta_top_with - delta_top_no

    assert borrow_impact_bottom > 1e-8, "Borrow cost should reduce the bottom decile's net return."
    assert borrow_impact_top < 1e-8, (
        "Borrow cost should NOT affect the top decile's net return; "
        f"got delta_top_with - delta_top_no = {borrow_impact_top:.2e}."
    )
    assert borrow_impact_bottom > borrow_impact_top + 1e-6


# ─── 6. Calendar-time portfolio smoke test ──────────────────────────────────


def test_calendar_time_portfolio_smoke(assignments, prices):
    """CTP builds a non-empty frame with the right schema and plausible returns."""
    ctp_top = build_calendar_time_portfolio(
        assignments,
        prices,
        holding_start=1,
        holding_end=20,
        decile=int(assignments["decile"].max()),
        weight_scheme="value_weight",
    )
    assert {"calendar_date", Col.PORTFOLIO_RET_GROSS, "n_holdings"} <= set(ctp_top.columns)
    assert len(ctp_top) > 0, "CTP returned no rows; holding window never overlaps prices."
    assert (ctp_top["n_holdings"] > 0).all()
    assert ctp_top[Col.PORTFOLIO_RET_GROSS].notna().any()


# ─── 7. Long-short return computation ───────────────────────────────────────


def test_long_short_returns(assignments, prices):
    """Long-short = top decile − bottom decile, row by row."""
    top = int(assignments["decile"].max())
    bottom = int(assignments["decile"].min())

    ctp_top = build_calendar_time_portfolio(
        assignments, prices, holding_start=1, holding_end=20, decile=top
    )
    ctp_bot = build_calendar_time_portfolio(
        assignments, prices, holding_start=1, holding_end=20, decile=bottom
    )
    ls = compute_long_short_returns(ctp_top, ctp_bot)

    common = ctp_top.merge(ctp_bot, on="calendar_date", suffixes=("_top", "_bot"))
    assert not common.empty, "No overlapping calendar dates between top and bottom CTPs."
    sample = common.iloc[0]
    expected = sample[f"{Col.PORTFOLIO_RET_GROSS}_top"] - sample[f"{Col.PORTFOLIO_RET_GROSS}_bot"]
    got = ls.loc[ls["calendar_date"] == sample["calendar_date"], Col.PORTFOLIO_RET_GROSS].iloc[0]
    assert abs(got - expected) < 1e-12


# ─── 8. Amihud illiquidity monotonicity ─────────────────────────────────────


def test_amihud_illiquidity(prices):
    """Illiquid names (low volume) get higher Amihud than liquid names."""
    amihud = compute_amihud_illiquidity(prices, window=252)
    assert {"ticker", "amihud"} <= set(amihud.columns)
    assert (amihud["amihud"] > 0).all(), "Amihud must be positive"

    illiquid = prices.copy()
    illiquid["volume"] = illiquid["volume"] * 1e-3  # 1000x less liquid
    illiquid["ticker"] = illiquid["ticker"] + "_ILL"
    combined = pd.concat([prices, illiquid], ignore_index=True)
    amihud2 = compute_amihud_illiquidity(combined, window=252)

    base = amihud2[~amihud2["ticker"].str.endswith("_ILL")]["amihud"].median()
    ill = amihud2[amihud2["ticker"].str.endswith("_ILL")]["amihud"].median()
    assert ill > base, (
        f"Illiquid clone ({ill:.2e}) should have higher Amihud than baseline ({base:.2e})."
    )


# ─── 9. Global-sort look-ahead warning ──────────────────────────────────────


def test_global_sort_emits_warning(sue_table, event_returns):
    """A global sort (sort_group_cols=None) must emit a look-ahead warning."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        assign_portfolio_deciles(
            sue_table,
            event_returns,
            sue_col="sue3",
            n_portfolios=10,
            sort_group_cols=None,
        )
        assert any("look-ahead" in str(wi.message).lower() for wi in w), (
            "A global sort must emit a look-ahead-bias warning."
        )
