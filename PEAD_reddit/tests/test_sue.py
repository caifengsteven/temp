"""Unit tests for the SUE construction module.

Each test verifies a Livnat-Mendenhall (2006) formula against hand-computed
values or clearly-reasoned properties. Synthetic data (with adj_factor = 1.0)
is used for the analyst/decile end-to-end checks so arithmetic is exact.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pead.sue.analyst import _filter_analyst_forecasts, compute_sue3
from pead.sue.time_series import compute_sue1, compute_sue2
from pead.sue.validate import apply_lm_filters, assign_deciles
from pead.synthetic import (
    generate_synthetic_earnings_events,
    generate_synthetic_fundamentals,
)

# ─── Shared fixtures ────────────────────────────────────────────────────────


def _quarter_ends(n: int = 8, start: str = "2022-03-31") -> pd.DatetimeIndex:
    """``n`` consecutive fiscal-quarter end dates."""
    return pd.date_range(start, periods=n, freq="QE")


def _fiscal_labels(dates: pd.DatetimeIndex) -> list[str]:
    return [f"{d.year}Q{d.quarter}" for d in dates]


# ─── 1. SUE1: seasonal random walk ──────────────────────────────────────────


def test_sue1_known_value() -> None:
    """SUE1 = (EPS_t - EPS_{t-4}) / P_t with split-adjusted EPS and price."""
    fq_ends = _quarter_ends(8)
    eps = [1.0, 1.5, 2.0, 1.0, 2.0, 2.5, 3.0, 2.0]
    price = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]

    fund = pd.DataFrame(
        {
            "ticker": ["TEST US Equity"] * 8,
            "fq_end": fq_ends,
            "fiscal_quarter": _fiscal_labels(fq_ends),
            "announce_date": fq_ends + pd.Timedelta(days=30),
            "eps_primary": eps,
            "price_qe": price,
            "adj_factor": [1.0] * 8,
            "special_items": [0.0] * 8,
            "shares_basic": [100.0] * 8,
            "report_date": fq_ends + pd.Timedelta(days=30),
        }
    )

    out = compute_sue1(fund)

    # First four quarters have no seasonal lag → dropped.
    assert len(out) == 4
    assert set(out.columns) == {"ticker", "announce_date", "fiscal_quarter", "sue1"}

    expected = {
        f"{fq_ends[4].year}Q{fq_ends[4].quarter}": (2.0 - 1.0) / 50.0,
        f"{fq_ends[5].year}Q{fq_ends[5].quarter}": (2.5 - 1.5) / 60.0,
        f"{fq_ends[6].year}Q{fq_ends[6].quarter}": (3.0 - 2.0) / 70.0,
        f"{fq_ends[7].year}Q{fq_ends[7].quarter}": (2.0 - 1.0) / 80.0,
    }
    for _, row in out.iterrows():
        assert row["sue1"] == pytest.approx(expected[row["fiscal_quarter"]])


def test_sue1_group_isolation() -> None:
    """The seasonal lag must not leak across tickers."""
    fq = _quarter_ends(8)
    base = {
        "fq_end": list(fq) * 2,
        "fiscal_quarter": _fiscal_labels(fq) * 2,
        "announce_date": list(fq + pd.Timedelta(days=30)) * 2,
        "eps_primary": [1.0] * 8 + [5.0] * 8,
        "price_qe": [50.0] * 16,
        "adj_factor": [1.0] * 16,
        "special_items": [0.0] * 16,
        "shares_basic": [100.0] * 16,
        "report_date": list(fq + pd.Timedelta(days=30)) * 2,
    }
    fund = pd.DataFrame(base)
    fund.insert(0, "ticker", ["A"] * 8 + ["B"] * 8)

    out = compute_sue1(fund)
    # Firm A: constant EPS → SUE1 == 0 everywhere after the burn-in.
    a = out[out["ticker"] == "A"]
    assert np.allclose(a["sue1"].to_numpy(), 0.0)
    assert len(a) == 4
    # Firm B: also constant EPS → SUE1 == 0, and no leakage from A's lag.
    b = out[out["ticker"] == "B"]
    assert np.allclose(b["sue1"].to_numpy(), 0.0)
    assert len(b) == 4


# ─── 2. SUE2: random walk excluding special items ───────────────────────────


def test_sue2_special_items() -> None:
    """A one-off special gain inflates SUE1 but is removed from SUE2."""
    fq_ends = _quarter_ends(8)
    special_items = [0.0] * 8
    special_items[4] = 100.0  # pre-tax gain booked only in the 5th quarter

    fund = pd.DataFrame(
        {
            "ticker": ["TEST US Equity"] * 8,
            "fq_end": fq_ends,
            "fiscal_quarter": _fiscal_labels(fq_ends),
            "announce_date": fq_ends + pd.Timedelta(days=30),
            "eps_primary": [2.0] * 8,  # flat reported EPS
            "price_qe": [50.0] * 8,
            "adj_factor": [1.0] * 8,
            "special_items": special_items,
            "shares_basic": [100.0] * 8,
            "report_date": fq_ends + pd.Timedelta(days=30),
        }
    )

    s1 = compute_sue1(fund)
    s2 = compute_sue2(fund)

    # Flat EPS ⇒ SUE1 is zero at every post-burn-in quarter.
    assert np.allclose(s1["sue1"].to_numpy(), 0.0)

    target = f"{fq_ends[4].year}Q{fq_ends[4].quarter}"
    # actual2_t   = 2.0 - 0.65 * 100 / 100 = 1.35
    # actual2_t-4 = 2.0 - 0.65 *   0 / 100 = 2.00
    # SUE2 = (1.35 - 2.00) / 50 = -0.013
    sue2_val = s2.loc[s2["fiscal_quarter"] == target, "sue2"].iloc[0]
    assert sue2_val == pytest.approx(-0.013)

    # Removing the special gain lowers the surprise vs. SUE1.
    sue1_val = s1.loc[s1["fiscal_quarter"] == target, "sue1"].iloc[0]
    assert sue2_val < sue1_val

    # Other quarters have no special items ⇒ SUE2 also collapses to zero there.
    other = s2[s2["fiscal_quarter"] != target]
    assert np.allclose(other["sue2"].to_numpy(), 0.0)


def test_sue2_tax_factor_defaults_to_065() -> None:
    """The 0.65 after-tax factor is the Livnat-Mendenhall default."""
    fq_ends = _quarter_ends(8)
    fund = pd.DataFrame(
        {
            "ticker": ["X"] * 8,
            "fq_end": fq_ends,
            "fiscal_quarter": _fiscal_labels(fq_ends),
            "announce_date": fq_ends + pd.Timedelta(days=30),
            "eps_primary": [2.0] * 8,
            "price_qe": [50.0] * 8,
            "adj_factor": [1.0] * 8,
            "special_items": [200.0] + [0.0] * 7,  # gain in the *first* quarter
            "shares_basic": [100.0] * 8,
            "report_date": fq_ends + pd.Timedelta(days=30),
        }
    )
    # Quarter index 4: lag (index 0) has the special gain.
    s2_default = compute_sue2(fund)
    s2_custom = compute_sue2(fund, tax_adj=0.5)

    target = f"{fq_ends[4].year}Q{fq_ends[4].quarter}"
    # default:  actual2_t = 2.0; actual2_t-4 = 2.0 - 0.65*200/100 = 0.70  → (2.0-0.70)/50
    # custom:   actual2_t-4 = 2.0 - 0.50*200/100 = 1.00                    → (2.0-1.00)/50
    val_default = s2_default.loc[s2_default["fiscal_quarter"] == target, "sue2"].iloc[0]
    val_custom = s2_custom.loc[s2_custom["fiscal_quarter"] == target, "sue2"].iloc[0]
    assert val_default == pytest.approx((2.0 - (2.0 - 0.65 * 2.0)) / 50.0)
    assert val_custom == pytest.approx((2.0 - (2.0 - 0.50 * 2.0)) / 50.0)


# ─── 3. SUE3: analyst-median forecast ───────────────────────────────────────


def test_sue3_analyst_based() -> None:
    """SUE3 = (actual_eps - medest_eps) / price_qe."""
    events = pd.DataFrame(
        {
            "ticker": ["A", "B", "C"],
            "announce_date": pd.to_datetime(["2023-04-01"] * 3),
            "fiscal_quarter": ["2023Q1"] * 3,
            "actual_eps": [2.0, 1.0, 3.0],
            "medest_eps": [1.8, 1.2, 3.0],
            "price_qe": [50.0, 40.0, 30.0],
        }
    )

    out = compute_sue3(events)
    assert list(out.columns) == ["ticker", "announce_date", "fiscal_quarter", "sue3"]
    assert len(out) == 3

    got = out.set_index("ticker")["sue3"]
    assert got["A"] == pytest.approx((2.0 - 1.8) / 50.0)  # beat
    assert got["B"] == pytest.approx((1.0 - 1.2) / 40.0)  # miss
    assert got["C"] == pytest.approx((3.0 - 3.0) / 30.0)  # in-line


def test_sue3_uses_median_not_mean() -> None:
    """Sanity check that we wire the *median* forecast into the formula."""
    events = pd.DataFrame(
        {
            "ticker": ["A"],
            "announce_date": pd.to_datetime(["2023-04-01"]),
            "fiscal_quarter": ["2023Q1"],
            "actual_eps": [2.0],
            "medest_eps": [1.5],  # median
            "meanest_eps": [1.0],  # mean (must be ignored)
            "price_qe": [25.0],
        }
    )
    out = compute_sue3(events)
    # Median-based: (2.0 - 1.5)/25 = 0.02; mean-based would be (2.0-1.0)/25 = 0.04.
    assert out["sue3"].iloc[0] == pytest.approx(0.02)


def test_filter_analyst_forecasts_keeps_latest_per_analyst() -> None:
    """Within the window, only the most recent forecast per analyst survives."""
    announce = pd.Timestamp("2023-04-01")
    forecasts = pd.DataFrame(
        {
            "ticker": ["A"] * 4,
            "announce_date": [announce] * 4,
            "analyst_id": ["a1", "a1", "a2", "a2"],
            "forecast_date": [
                announce - pd.Timedelta(days=80),
                announce - pd.Timedelta(days=10),
                announce - pd.Timedelta(days=70),
                announce - pd.Timedelta(days=5),
            ],
            "forecast_eps": [1.0, 1.5, 2.0, 2.2],
        }
    )

    deduped = _filter_analyst_forecasts(forecasts, window_days=90)
    assert len(deduped) == 2
    # Latest in-window estimate per analyst is kept.
    assert sorted(deduped["forecast_eps"]) == [1.5, 2.2]


def test_filter_analyst_forecasts_drops_out_of_window() -> None:
    """Forecasts outside the pre-announcement window are discarded entirely."""
    announce = pd.Timestamp("2023-04-01")
    forecasts = pd.DataFrame(
        {
            "ticker": ["A", "A"],
            "announce_date": [announce, announce],
            "analyst_id": ["a1", "a2"],
            "forecast_date": [
                announce - pd.Timedelta(days=200),  # out of window
                announce - pd.Timedelta(days=10),  # in window
            ],
            "forecast_eps": [1.0, 2.0],
        }
    )
    deduped = _filter_analyst_forecasts(forecasts, window_days=90)
    assert len(deduped) == 1
    assert deduped.iloc[0]["analyst_id"] == "a2"


# ─── 4. Livnat-Mendenhall filters ──────────────────────────────────────────


def test_lm_filters() -> None:
    """Price, market-cap, date-agreement, NaN and inf screens each fire."""
    announce = pd.Timestamp("2023-04-01")
    sue = pd.DataFrame(
        {
            "ticker": ["A", "B", "C", "D", "E", "F"],
            "announce_date": [announce] * 6,
            "fiscal_quarter": ["2023Q1"] * 6,
            "sue3": [0.01, 0.02, 0.03, np.nan, np.inf, 0.04],
        }
    )
    fund = pd.DataFrame(
        {
            "ticker": ["A", "B", "C", "D", "E", "F"],
            "fiscal_quarter": ["2023Q1"] * 6,
            "price_qe": [50.0, 0.5, 2.0, 50.0, 50.0, 50.0],
            "shares_basic": [100.0, 100.0, 2.0, 100.0, 100.0, 100.0],
            "report_date": [
                announce,
                announce,
                announce,
                announce,
                announce,
                announce + pd.Timedelta(days=3),  # F: |report-announce| = 3 > 1
            ],
            "announce_date": [announce] * 6,
        }
    )

    out = apply_lm_filters(sue, fund, min_price=1.0, min_mcap_millions=5.0, max_date_diff_days=1)

    assert "filtered_out" in out.columns
    assert "filter_reason" in out.columns

    passing = out[~out["filtered_out"]]
    assert set(passing["ticker"]) == {"A"}

    reasons = out.set_index("ticker")["filter_reason"]
    assert "price" in reasons["B"]
    assert "mcap" in reasons["C"]  # 2.0 * 2.0 = 4.0 < 5.0
    assert "sue_nan" in reasons["D"]
    assert "sue_inf" in reasons["E"]
    assert "date" in reasons["F"]
    assert reasons["A"] == ""


def test_lm_filters_boundary_is_strict() -> None:
    """``>`` comparisons: a row exactly at the threshold is dropped."""
    announce = pd.Timestamp("2023-04-01")
    sue = pd.DataFrame(
        {
            "ticker": ["AT", "MCAP"],
            "announce_date": [announce] * 2,
            "fiscal_quarter": ["2023Q1"] * 2,
            "sue3": [0.01, 0.02],
        }
    )
    fund = pd.DataFrame(
        {
            "ticker": ["AT", "MCAP"],
            "fiscal_quarter": ["2023Q1"] * 2,
            "price_qe": [1.0, 5.0],  # AT exactly at min_price
            "shares_basic": [1.0, 1.0],  # MCAP: 5.0 * 1.0 = 5.0 exactly
            "report_date": [announce] * 2,
            "announce_date": [announce] * 2,
        }
    )
    out = apply_lm_filters(sue, fund, min_price=1.0, min_mcap_millions=5.0)
    # Both sit exactly on the boundary → both filtered out (strict inequality).
    assert out["filtered_out"].all()


# ─── 5. Cross-sectional deciles ─────────────────────────────────────────────


def test_deciles_per_cross_section() -> None:
    """Deciles must be formed within each group, not globally."""
    rows = []
    for week, base in [("W1", 1), ("W2", 11)]:
        for i in range(10):
            rows.append({"ticker": f"{week}_{i}", "event_week": week, "sue3": float(base + i)})
    df = pd.DataFrame(rows)

    out = assign_deciles(df, sue_col="sue3", n_deciles=10, group_cols=["event_week"])
    assert "sue3_decile" in out.columns

    for week in ("W1", "W2"):
        sub = out[out["event_week"] == week].sort_values("sue3")
        # Lowest SUE in the group → decile 1 (miss); highest → decile 10 (beat).
        assert sub["sue3_decile"].iloc[0] == 1
        assert sub["sue3_decile"].iloc[-1] == 10
        assert list(sub["sue3_decile"]) == list(range(1, 11))

    # The low value of W2 (11.0) is decile 1 per-group but would be ~decile 5
    # or higher under a global sort — confirming no look-ahead across groups.
    w2_lowest = out[(out["event_week"] == "W2") & (out["sue3"] == 11.0)]
    assert w2_lowest["sue3_decile"].iloc[0] == 1


def test_deciles_global_warns(caplog: pytest.LogCaptureFixture) -> None:
    """``group_cols=None`` produces a single global sort (and a warning)."""
    import logging

    df = pd.DataFrame({"ticker": list("abcdefghij"), "sue3": list(map(float, range(10)))})
    with caplog.at_level(logging.WARNING, logger="pead.sue.validate"):
        out = assign_deciles(df, sue_col="sue3", n_deciles=10, group_cols=None)
    assert list(out.sort_values("sue3")["sue3_decile"]) == list(range(1, 11))
    assert any("look-ahead" in rec.message for rec in caplog.records)


# ─── 6. SUE3 decile extremes: misses vs. beats ──────────────────────────────


def test_sue3_decile_extremes() -> None:
    """Bottom decile = large misses; top decile = large beats."""
    events = generate_synthetic_earnings_events(n_tickers=60, n_quarters=12, seed=11)
    fund = generate_synthetic_fundamentals(events, seed=11)

    # Bring quarter-end price onto the events frame so SUE3 can be computed.
    ev = events.merge(
        fund[["ticker", "fiscal_quarter", "price_qe"]],
        on=["ticker", "fiscal_quarter"],
        how="left",
        validate="one_to_one",
    )
    sue3 = compute_sue3(ev)

    # Deciles are formed within each fiscal quarter (the cross-section).
    bucketed = assign_deciles(
        sue3, sue_col="sue3", n_deciles=10, group_cols=["fiscal_quarter"]
    ).dropna(subset=["sue3_decile"])

    d1 = bucketed[bucketed["sue3_decile"] == 1]
    d10 = bucketed[bucketed["sue3_decile"] == 10]
    assert len(d1) > 0 and len(d10) > 0

    # Means are strictly ordered: misses (decile 1) ≪ beats (decile 10).
    assert d1["sue3"].mean() < d10["sue3"].mean()
    assert d1["sue3"].mean() < 0 < d10["sue3"].mean()

    # And the sign content lines up with the PEAD asymmetry buckets.
    assert (d1["sue3"] < 0).mean() > 0.5  # bottom decile is mostly misses
    assert (d10["sue3"] > 0).mean() > 0.5  # top decile is mostly beats
