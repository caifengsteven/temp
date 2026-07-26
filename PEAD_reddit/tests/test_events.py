"""
Unit tests for the event-study returns module.

Verifies, against hand-computed examples and the synthetic fixture with known
injected drift:
  1. trading-calendar snapping and event-time offsets,
  2. market-model OLS recovery of an injected beta,
  3. CAR = sum(AR_t) on a hand-constructed return series,
  4. mid-quote returns differ from close-to-close returns (bid-ask bounce purge),
  5. end-to-end recovery of the injected PEAD asymmetry (misses drift ~2x beats).

Run:  python -m pytest tests/test_events.py -v
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pead.events.calendar import (
    build_trading_calendar,
    get_window_dates,
    snap_to_trading_day,
    trading_day_offset,
)
from pead.events.returns import (
    compute_all_event_returns,
    compute_car,
    compute_market_returns,
    estimate_market_model,
)
from pead.synthetic import (
    INJECTED_ASYMMETRY_RATIO,
    generate_all_synthetic_data,
)

# ─── 1. Trading calendar ────────────────────────────────────────────────────


def test_trading_calendar() -> None:
    """Calendar extraction, snapping, offsets and window bounds."""
    raw_dates = pd.DatetimeIndex(
        [
            "2024-01-02",
            "2024-01-03",
            "2024-01-04",
            "2024-01-05",
            "2024-01-08",
            "2024-01-09",
        ]
    )
    # Panel with duplicate dates and out-of-order rows -> build must dedupe & sort.
    px = pd.DataFrame(
        {
            "trading_date": list(raw_dates[::-1]) + list(raw_dates),
            "ticker": "AAA",
        }
    )
    cal = build_trading_calendar(px)
    assert isinstance(cal, pd.DatetimeIndex)
    assert cal.equals(raw_dates)
    assert cal.is_monotonic_increasing

    # Saturday 2024-01-06 is between Fri 01-05 and Mon 01-08.
    assert snap_to_trading_day(pd.Timestamp("2024-01-06"), cal, "forward") == pd.Timestamp(
        "2024-01-08"
    )
    assert snap_to_trading_day(pd.Timestamp("2024-01-06"), cal, "backward") == pd.Timestamp(
        "2024-01-05"
    )
    # A trading day snaps to itself under both directions.
    assert snap_to_trading_day(pd.Timestamp("2024-01-03"), cal, "forward") == pd.Timestamp(
        "2024-01-03"
    )

    # Offsets: 0 -> same day, +2 -> skip the weekend, -2 -> two trading days back.
    assert trading_day_offset(pd.Timestamp("2024-01-04"), 0, cal) == pd.Timestamp("2024-01-04")
    assert trading_day_offset(pd.Timestamp("2024-01-04"), 2, cal) == pd.Timestamp("2024-01-08")
    assert trading_day_offset(pd.Timestamp("2024-01-04"), -2, cal) == pd.Timestamp("2024-01-02")

    # A non-trading announce date snaps forward before the offset is applied.
    assert trading_day_offset(pd.Timestamp("2024-01-06"), 0, cal) == pd.Timestamp("2024-01-08")
    assert trading_day_offset(pd.Timestamp("2024-01-06"), 1, cal) == pd.Timestamp("2024-01-09")

    # Window bounds (inclusive).
    assert get_window_dates(pd.Timestamp("2024-01-04"), -1, 2, cal) == (
        pd.Timestamp("2024-01-03"),
        pd.Timestamp("2024-01-08"),
    )

    # Boundary clamping: announce before the first trading day clamps to day 0.
    assert trading_day_offset(pd.Timestamp("2023-12-01"), -5, cal) == cal[0]

    with pytest.raises(ValueError):
        snap_to_trading_day(pd.Timestamp("2024-01-04"), cal, "sideways")
    with pytest.raises(ValueError):
        get_window_dates(pd.Timestamp("2024-01-04"), 2, 1, cal)


# ─── 2. Market-model estimation ─────────────────────────────────────────────


def _common_factor_panel(
    n_tickers: int = 40, n_days: int = 200, seed: int = 7
) -> tuple[pd.DataFrame, pd.DatetimeIndex, dict[str, float]]:
    """Build a clean panel with a *common* market factor and known per-firm betas.

    firm_ret = 0.0005 + beta_i * market + noise_i, so regressing on the
    cross-sectional average (which loads on the common market) recovers beta_i.
    """
    rng = np.random.default_rng(seed)
    market = rng.normal(0.0003, 0.01, n_days)
    dates = pd.bdate_range("2023-01-02", periods=n_days)
    frames: list[pd.DataFrame] = []
    true_betas: dict[str, float] = {}
    for i in range(n_tickers):
        beta = float(rng.uniform(0.6, 1.4))
        tkr = f"TKR{i:03d}"
        true_betas[tkr] = beta
        ret = 0.0005 + beta * market + rng.normal(0.0, 0.003, n_days)
        frames.append(
            pd.DataFrame(
                {
                    "ticker": tkr,
                    "trading_date": dates,
                    "ret": ret,
                    "px_close": np.cumprod(1.0 + ret) * 100.0,
                    "px_bid": np.nan,
                    "px_ask": np.nan,
                }
            )
        )
    return pd.concat(frames, ignore_index=True), dates, true_betas


def test_market_model_estimation() -> None:
    """OLS recovers the injected beta (and near-zero alpha) for a common factor."""
    px, dates, true_betas = _common_factor_panel()
    cal = build_trading_calendar(px)
    mkt = compute_market_returns(px, "ret")

    target = "TKR000"
    ab = estimate_market_model(
        px,
        announce_date=dates[-1],
        ticker=target,
        calendar=cal,
        est_start=-150,
        est_end=-5,
        market_returns=mkt,
    )
    assert ab is not None
    alpha, beta = ab
    assert np.isfinite(alpha) and np.isfinite(beta)
    # Beta recovered tightly to the injected value.
    assert abs(beta - true_betas[target]) < 0.10
    # Alpha is the intercept; with near-zero-mean residuals it is close to 0.0005.
    assert abs(alpha - 0.0005) < 0.001

    # Average |beta error| across the cross-section should be small.
    errors = []
    for tkr, true_beta in true_betas.items():
        res = estimate_market_model(
            px, dates[-1], tkr, cal, est_start=-150, est_end=-5, market_returns=mkt
        )
        assert res is not None
        errors.append(abs(res[1] - true_beta))
    assert float(np.mean(errors)) < 0.10


# ─── 3. CAR = sum(AR_t) on a hand-constructed series ────────────────────────


def test_car_computation() -> None:
    """CAR equals the hand-computed sum of abnormal returns."""
    dates = pd.bdate_range("2024-01-02", periods=5)
    firm = np.array([0.010, 0.020, -0.010, 0.005, 0.015])
    market = np.array([0.005, 0.010, 0.000, 0.002, 0.008])
    alpha, beta = 0.001, 0.8

    # Hand computation: AR_t = firm - (alpha + beta*market); CAR = sum.
    expected_car = float(np.sum(firm - (alpha + beta * market)))
    assert round(expected_car, 6) == 0.015

    px = pd.DataFrame(
        {
            "ticker": "AAA",
            "trading_date": dates,
            "ret": firm,
            "px_close": np.cumprod(1.0 + firm) * 100.0,
        }
    )
    cal = build_trading_calendar(px)
    mkt = pd.Series(market, index=dates)
    firm_s = pd.Series(firm, index=dates)

    car = compute_car(
        px,
        ticker="AAA",
        announce_date=dates[0],
        calendar=cal,
        window_start=0,
        window_end=4,
        alpha=alpha,
        beta=beta,
        market_returns=mkt,
        firm_series=firm_s,
    )
    assert car is not None
    assert car == pytest.approx(expected_car, abs=1e-12)

    # No firm observations inside the window -> None (insufficient data).
    sparse_px = pd.DataFrame(
        {"ticker": "AAA", "trading_date": [dates[0]], "ret": [0.01], "px_close": [100.0]}
    )
    car_none = compute_car(
        sparse_px,
        ticker="AAA",
        announce_date=dates[0],
        calendar=cal,
        window_start=1,
        window_end=3,
        alpha=alpha,
        beta=beta,
        market_returns=mkt,
    )
    assert car_none is None


# ─── 4. Mid-quote vs close-to-close (bid-ask bounce) ────────────────────────


def test_midquote_vs_close() -> None:
    """Mid-quote returns differ from close-to-close when closes bounce to bid/ask.

    With alpha=beta=0 the CAR reduces to the sum of returns, so this directly
    contrasts the two return constructions. The mid-quote path is smooth and
    upward; the close path is contaminated by alternating executions at the bid
    and ask (the bid-ask bounce that Zhang, Gregoriou & Wu (2024) show inflates
    short-side PEAD drift).
    """
    dates = pd.bdate_range("2024-01-02", periods=5)
    midquote = [100.0, 100.5, 101.0, 101.5, 102.0]  # smooth +0.5%/day
    # Closes alternate to the ask then the bid -> bouncy close-to-close returns.
    close = [100.0, 101.0, 100.5, 102.0, 101.5]
    bid = [m - 0.5 for m in midquote]
    ask = [m + 0.5 for m in midquote]

    px = pd.DataFrame(
        {
            "ticker": "X",
            "trading_date": dates,
            "px_close": close,
            "px_bid": bid,
            "px_ask": ask,
            "ret": pd.Series(close).pct_change().fillna(0.0).to_numpy(),
        }
    )
    cal = build_trading_calendar(px)

    car_close = compute_car(px, "X", dates[0], cal, 0, 4, alpha=0.0, beta=0.0, use_midquote=False)
    car_mid = compute_car(px, "X", dates[0], cal, 0, 4, alpha=0.0, beta=0.0, use_midquote=True)
    assert car_close is not None and car_mid is not None

    # The two constructions must diverge.
    assert not np.isclose(car_close, car_mid, atol=1e-6)

    # Cross-check against the directly computed return sums.
    close_ret = pd.Series(close).pct_change().dropna()
    mid_ret = pd.Series(midquote).pct_change().dropna()
    assert car_close == pytest.approx(float(close_ret.sum()), abs=1e-9)
    assert car_mid == pytest.approx(float(mid_ret.sum()), abs=1e-9)

    # Mid-quote returns are uniformly small and positive; closes are not.
    assert (mid_ret > 0).all()
    assert not (close_ret > 0).all()


# ─── 5. End-to-end drift recovery (the key validation) ──────────────────────


def test_drift_recovery() -> None:
    """Recover the injected PEAD asymmetry from the synthetic fixture.

    Synthetic ground truth (see pead.synthetic): beats drift +0.5% and misses
    drift -1.0% over [1, 20] trading days, i.e. misses drift ~2x beats. This
    test runs the full market-model event study and checks the average CAR
    recovers those signs, magnitudes and the asymmetry ratio.

    Drift windows deliberately EXCLUDE day 0 (the announcement reaction window
    [0, 1] carries no injected drift, which we assert as a sanity check).
    """
    data = generate_all_synthetic_data(n_tickers=200, n_quarters=24, seed=42)
    events = data["earnings_events"]
    prices = data["daily_prices"]

    windows = [
        ("announcement_reaction", 0, 1),
        ("short_drift", 1, 20),  # the Reddit-claim horizon; excludes day 0
        ("medium_drift", 1, 60),  # Bernard-Thomas canonical; excludes day 0
    ]
    out = compute_all_event_returns(
        events, prices, windows, use_midquote=True, estimation_start=-255, estimation_end=-46
    )

    # Almost every event must yield a market-model fit.
    assert out["alpha"].notna().sum() >= 0.95 * len(out)

    # Beat = positive earnings surprise, miss = negative (matches the injection).
    out = out.copy()
    out["surprise"] = out["actual_eps"] - out["medest_eps"]
    beats = out.loc[out["surprise"] > 0]
    misses = out.loc[out["surprise"] < 0]
    assert len(beats) > 1000 and len(misses) > 1000

    beat_car = float(beats["car_short_drift"].mean())
    miss_car = float(misses["car_short_drift"].mean())

    # Signs: beats drift up, misses drift down.
    assert beat_car > 0.0, f"expected beat CAR > 0, got {beat_car}"
    assert miss_car < 0.0, f"expected miss CAR < 0, got {miss_car}"

    # Magnitudes close to the injected +50bps / -100bps.
    assert 0.003 < beat_car < 0.011, f"beat CAR {beat_car} outside [0.3%, 1.1%]"
    assert -0.015 < miss_car < -0.006, f"miss CAR {miss_car} outside [-1.5%, -0.6%]"

    # Asymmetry: misses drift more than beats, ratio near the injected 2.0x.
    ratio = abs(miss_car) / beat_car
    assert 1.2 < ratio < 2.6, f"asymmetry ratio {ratio:.2f} not near {INJECTED_ASYMMETRY_RATIO}"
    assert miss_car < beat_car  # misses more negative than beats are positive

    # Sanity: the announcement-reaction window [0,1] carries no injected drift,
    # so average CARs there are economically negligible.
    assert abs(float(beats["car_announcement_reaction"].mean())) < 0.004
    assert abs(float(misses["car_announcement_reaction"].mean())) < 0.004

    # Mid-quote CAR column was produced and is finite for the bulk of events.
    assert "car_midquote_short_drift" in out.columns
    assert out["car_midquote_short_drift"].notna().sum() >= 0.9 * len(out)

    # BHAR is reported but right-skewed; it should still agree on sign.
    assert float(misses["bhar_short_drift"].mean()) < float(beats["bhar_short_drift"].mean())
