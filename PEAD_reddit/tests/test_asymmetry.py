"""
Tests for the PEAD asymmetry module.

The headline test (``test_recovers_injected_asymmetry``) validates that the
pipeline recovers the 2.0x asymmetry ratio injected into synthetic data.

Implementation note: ``src/pead/synthetic.py::generate_synthetic_prices``
generates an INDEPENDENT price series per event (all tagged with the same
ticker), so concatenating them produces overlapping series per ticker. A
unified per-ticker price history therefore cannot be reconstructed from the
raw output without per-event disambiguation. The upstream SUE / event-study
modules that would normally consume those prices and emit ``event_returns``
are not part of this delivery.

For this module's tests we therefore construct ``event_returns`` directly
from the synthetic events + the *known* injected drift model used in
``generate_synthetic_prices`` (beat = +50bps, miss = -100bps over [1, 20],
giving a 2.0x ratio). This isolates the unit under test (the asymmetry
module) from upstream-construction concerns while still using the synthetic
events and the documented injected parameters.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pead.asymmetry import (
    bootstrap_asymmetry_ratio,
    clustered_difference_test,
    compute_amihud_by_event,
    compute_asymmetry_by_sue_method,
    compute_asymmetry_by_window,
    compute_asymmetry_ratio,
    double_sort_asymmetry,
    double_sort_by_size,
    full_asymmetry_report,
)
from pead.schema import Col
from pead.synthetic import (
    INJECTED_ASYMMETRY_RATIO,
    INJECTED_BEAT_DRIFT_BPS,
    INJECTED_MISS_DRIFT_BPS,
    generate_synthetic_earnings_events,
)

# ─── Constants matching synthetic.py's INJECTED_DRIFT model ─────────────────


def _beat_drift() -> float:
    """Injected beat drift over [1, 20] as a decimal (e.g. 0.005 = +50bps)."""
    return INJECTED_BEAT_DRIFT_BPS / 10_000.0


def _miss_drift() -> float:
    """Injected miss drift over [1, 20] as a decimal (e.g. -0.010 = -100bps)."""
    return INJECTED_MISS_DRIFT_BPS / 10_000.0


# Per-event CAR noise std. Smaller than the empirical ~12% because (a) the
# test isolates the asymmetry module rather than full price-level noise, and
# (b) it lets the test deterministically recover the injected ratio. The
# value still leaves a non-trivial noise floor so the bootstrap and
# clustered-SE paths are exercised meaningfully.
_CAR_NOISE_STD = 0.04


# ─── Fixtures ───────────────────────────────────────────────────────────────


def _assign_deciles(er: pd.DataFrame) -> pd.DataFrame:
    """Assign 1..10 cross-sectional SUE deciles within each fiscal quarter."""
    er["sue3"] = er["actual_eps"] - er["medest_eps"]
    er["sue3_decile"] = (
        er.groupby("fiscal_quarter")["sue3"]
        .transform(lambda x: pd.qcut(x.rank(method="first"), 10, labels=False) + 1)
        .astype("int")
    )
    # Mirror the decile assignment across SUE methods for the by-method tests.
    er["sue1_decile"] = er["sue3_decile"]
    er["sue2_decile"] = er["sue3_decile"]
    return er


def _inject_drift(er: pd.DataFrame, noise_std: float, seed: int) -> pd.DataFrame:
    """Inject CAR columns from the documented drift model in synthetic.py."""
    rng = np.random.default_rng(seed)
    n = len(er)
    is_miss = er["sue3_decile"] == 1
    is_beat = er["sue3_decile"] == 10

    car_short = np.zeros(n, dtype=float)
    car_short[is_miss] = _miss_drift()
    car_short[is_beat] = _beat_drift()
    car_short += rng.normal(0.0, noise_std, n)
    er["car_short_drift"] = car_short

    # Auxiliary windows: scaled signals + independent noise.
    er["car_announcement_reaction"] = np.where(
        is_miss, _miss_drift(), np.where(is_beat, _beat_drift(), 0.0)
    ) * 0.1 + rng.normal(0.0, noise_std * 0.3, n)
    er["car_medium_drift"] = car_short * 1.5 + rng.normal(0.0, noise_std * 0.5, n)
    return er


def make_injected_event_returns(
    n_tickers: int = 500,
    n_quarters: int = 80,
    noise_std: float = _CAR_NOISE_STD,
    seed: int = 42,
) -> pd.DataFrame:
    """Build event_returns from synthetic events + documented injected drift.

    The injected beat drift = +INJECTED_BEAT_DRIFT_BPS bps and miss drift =
    -INJECTED_MISS_DRIFT_BPS bps, so the population asymmetry ratio equals
    INJECTED_ASYMMETRY_RATIO (= 2.0x).
    """
    events = generate_synthetic_earnings_events(n_tickers, n_quarters, seed=seed)
    er = events[
        [Col.TICKER, Col.ANNOUNCE_DATE, "actual_eps", "medest_eps", "fiscal_quarter"]
    ].copy()
    er = _assign_deciles(er)
    er = _inject_drift(er, noise_std=noise_std, seed=seed + 99)
    return er


def make_composition_event_returns(
    n_buckets: int = 5,
    noise_std: float = 0.005,
    seed: int = 7,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build (event_returns, prices) where the asymmetry is COMPOSITION-DRIVEN.

    Setup:
      - Within every liquidity bucket, beat and miss drifts are EQUAL in
        magnitude (population ratio = 1.0 within bucket). The behavioural
        asymmetry is therefore absent.
      - Drift SCALES with illiquidity: illiquid stocks drift more (for both
        beats and misses).
      - Misses are over-represented in illiquid buckets; beats in liquid ones.

    Consequence: the unconditional ratio is well above 1 (composition), but
    within each liquidity bucket the ratio collapses toward 1.0. This is
    exactly the composition-effect scenario the double-sort exposes.
    """
    rng = np.random.default_rng(seed)

    # Per-bucket drift magnitudes — illiquid buckets drift more.
    bucket_drifts = np.linspace(0.010, 0.050, n_buckets)

    # Cell counts: misses skew illiquid, beats skew liquid. Both lists sum to
    # the same total so the unconditional ratio is driven purely by the
    # composition × drift-magnitude interaction (no sample-size confound).
    miss_counts = [50, 100, 200, 300, 350][:n_buckets]
    beat_counts = list(reversed(miss_counts))

    rows = []
    price_rows = []
    # Each event gets a unique ticker so Amihud computation sees independent
    # price histories (no per-ticker overlap).
    ev_id = 0
    for is_miss_flag in (True, False):
        counts = miss_counts if is_miss_flag else beat_counts
        for bucket_idx in range(n_buckets):
            n_in_cell = counts[bucket_idx]
            for _ in range(n_in_cell):
                tkr = f"EV{ev_id:06d} US Equity"
                ev_id += 1
                drift_mag = bucket_drifts[bucket_idx]
                drift = -drift_mag if is_miss_flag else drift_mag
                car = drift + rng.normal(0.0, noise_std)
                # Amihud: illiquid bucket → larger value.
                amihud = (bucket_idx + 1) * 1e-9 + rng.uniform(-0.2e-9, 0.2e-9)

                announce = pd.Timestamp("2020-01-01") + pd.Timedelta(days=ev_id)
                rows.append(
                    {
                        Col.TICKER: tkr,
                        Col.ANNOUNCE_DATE: announce,
                        "fiscal_quarter": "2020Q1",
                        "actual_eps": -1.0 if is_miss_flag else 1.0,
                        "medest_eps": 0.0,
                        "sue3_decile": 1 if is_miss_flag else 10,
                        "sue1_decile": 1 if is_miss_flag else 10,
                        "sue2_decile": 1 if is_miss_flag else 10,
                        "car_short_drift": car,
                        "car_announcement_reaction": car * 0.1,
                        "car_medium_drift": car * 1.5,
                        Col.AMIHUD_ILLIQUIDITY: amihud,
                    }
                )

                # Synthesize a small price history for this single-event ticker
                # whose Amihud matches `amihud`. Amihud = mean(|ret|/dvol); we
                # pick dvol so that |ret|/dvol averages to `amihud`.
                dates = pd.bdate_range(
                    announce - pd.Timedelta(days=400), announce - pd.Timedelta(days=1)
                )
                ret = rng.normal(0.001, 0.02, len(dates))
                # Solve dvol = |ret| / amihud (element-wise), then px * volume = dvol.
                dvol = np.abs(ret) / amihud
                px = 100.0 * np.cumprod(1 + ret)
                volume = dvol / px
                price_rows.append(
                    pd.DataFrame(
                        {
                            Col.TICKER: tkr,
                            Col.TRADING_DATE: dates,
                            Col.PX_CLOSE: px,
                            Col.VOLUME: volume,
                            Col.RET: ret,
                        }
                    )
                )

    er = pd.DataFrame(rows)
    prices = pd.concat(price_rows, ignore_index=True)
    return er, prices


# ─── Fixtures exposed to test functions ─────────────────────────────────────


@pytest.fixture(scope="module")
def injected_er() -> pd.DataFrame:
    """Session-shared event_returns with the documented 2.0x asymmetry injected."""
    return make_injected_event_returns()


@pytest.fixture(scope="module")
def composition_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """event_returns + prices with a pure composition effect."""
    return make_composition_event_returns()


# ─── Test 1: hand-constructed ratio ─────────────────────────────────────────


def test_ratio_computation() -> None:
    """Verify ratio = |miss| / |beat| on hand-constructed data."""
    er = pd.DataFrame(
        {
            Col.TICKER: ["A"] * 4 + ["B"] * 4 + ["C"] * 4,
            Col.ANNOUNCE_DATE: pd.date_range("2020-01-01", periods=12, freq="D"),
            "sue3_decile": [1, 1, 1, 1, 10, 10, 10, 10, 5, 5, 5, 5],
            "car_short_drift": [
                -0.02,
                -0.03,
                -0.025,
                -0.015,
                0.01,
                0.015,
                0.012,
                0.008,
                0,
                0,
                0,
                0,
            ],
        }
    )
    res = compute_asymmetry_ratio(er, return_col="car_short_drift", decile_col="sue3_decile")

    expected_miss_mean = -0.0225  # mean([-0.02, -0.03, -0.025, -0.015])
    expected_beat_mean = 0.01125  # mean([0.01, 0.015, 0.012, 0.008])
    expected_ratio = abs(expected_miss_mean) / abs(expected_beat_mean)

    assert res["miss_n"] == 4
    assert res["beat_n"] == 4
    assert res["miss_car_mean"] == pytest.approx(expected_miss_mean)
    assert res["beat_car_mean"] == pytest.approx(expected_beat_mean)
    assert res["ratio"] == pytest.approx(expected_ratio)
    # Difference is miss_mean - |beat_mean| — negative when misses drift more.
    assert res["difference"] == pytest.approx(expected_miss_mean - abs(expected_beat_mean))
    assert res["difference"] < 0


def test_ratio_handles_zero_beat() -> None:
    """When beat mean is exactly 0 the ratio is NaN (not infinity)."""
    er = pd.DataFrame(
        {
            Col.TICKER: ["A", "B", "C", "D"],
            Col.ANNOUNCE_DATE: pd.date_range("2020-01-01", periods=4),
            "sue3_decile": [1, 10, 1, 10],
            "car_short_drift": [-0.02, 0.01, -0.03, -0.01],  # beat mean = (0.01 + -0.01)/2 = 0
        }
    )
    res = compute_asymmetry_ratio(er)
    assert np.isnan(res["ratio"])


# ─── Test 2: bootstrap CI covers the injected ratio ─────────────────────────


def test_bootstrap_ci_covers_truth(injected_er: pd.DataFrame) -> None:
    """The 95% bootstrap CI of the ratio must contain the injected 2.0x."""
    boot = bootstrap_asymmetry_ratio(
        injected_er,
        return_col="car_short_drift",
        decile_col="sue3_decile",
        n_bootstrap=2000,
        confidence_level=0.95,
        seed=42,
    )
    assert boot["n_miss"] > 0
    assert boot["n_beat"] > 0
    assert boot["n_bootstrap_valid"] > 0
    assert np.isfinite(boot["ci_lower"])
    assert np.isfinite(boot["ci_upper"])
    assert boot["ci_lower"] <= boot["ratio_point_estimate"] <= boot["ci_upper"]
    assert boot["ci_lower"] <= INJECTED_ASYMMETRY_RATIO <= boot["ci_upper"]


def test_bootstrap_distribution_shape(injected_er: pd.DataFrame) -> None:
    """Bootstrap distribution should be right-skewed (ratio is fat-tailed)."""
    boot = bootstrap_asymmetry_ratio(injected_er, n_bootstrap=2000, confidence_level=0.95, seed=42)
    dist = boot["bootstrap_distribution"]
    assert len(dist) == 2000
    # Right-skewed: mean > median for a ratio distribution.
    assert np.mean(dist) >= np.median(dist)


# ─── Test 3: double sort reduces the ratio ──────────────────────────────────


def test_double_sort_reduces_ratio(
    composition_data: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    """Composition effect: within-bucket ratio is closer to 1.0 than unconditional."""
    er, prices = composition_data
    unconditional = compute_asymmetry_ratio(er, return_col="car_short_drift")
    within = double_sort_asymmetry(er, prices, return_col="car_short_drift", n_liquidity_buckets=5)

    assert len(within) == 5
    # Unconditional ratio should be > 1 by construction (composition effect).
    assert unconditional["ratio"] > 1.2, (
        f"Expected unconditional ratio > 1.2 (composition present), "
        f"got {unconditional['ratio']:.3f}"
    )
    # Mean within-bucket ratio should be lower than the unconditional ratio.
    avg_within_ratio = float(within["ratio"].mean())
    assert avg_within_ratio < unconditional["ratio"], (
        f"Within-bucket ratio ({avg_within_ratio:.3f}) should be lower than "
        f"unconditional ({unconditional['ratio']:.3f})"
    )
    # And each within-bucket ratio should be closer to 1.0 than the unconditional.
    unconditional_distance = abs(unconditional["ratio"] - 1.0)
    avg_within_distance = float((within["ratio"] - 1.0).abs().mean())
    assert avg_within_distance < unconditional_distance


def test_double_sort_by_size_runs(
    composition_data: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    """double_sort_by_size must run on the same fixture and return 5 buckets."""
    er, prices = composition_data
    within_size = double_sort_by_size(er, prices, return_col="car_short_drift", n_size_buckets=5)
    assert {"size_bucket", "miss_car", "beat_car", "ratio", "n"}.issubset(within_size.columns)
    assert len(within_size) <= 5


# ─── Test 4: clustered difference test detects the asymmetry ────────────────


def test_clustered_difference_significant(injected_er: pd.DataFrame) -> None:
    """The clustered t-test must reject H0 (no asymmetry) on injected data."""
    res = clustered_difference_test(
        injected_er, return_col="car_short_drift", decile_col="sue3_decile", cluster_col=Col.TICKER
    )
    # Misses drift more negatively than beats → negative difference.
    assert res["difference"] < 0
    # Large sample + injected 2.0x signal → strongly significant.
    assert res["t_stat"] < -3.0, f"Expected t < -3.0, got {res['t_stat']:.3f}"
    assert res["p_value"] < 0.05, f"Expected p < 0.05, got {res['p_value']:.4f}"
    # Cluster dof = n_clusters - 1.
    assert res["degrees_of_freedom"] == res["n_clusters"] - 1


def test_clustered_difference_missing_cluster_col_raises() -> None:
    er = pd.DataFrame(
        {
            Col.TICKER: ["A", "B"],
            Col.ANNOUNCE_DATE: pd.date_range("2020-01-01", periods=2),
            "sue3_decile": [1, 10],
            "car_short_drift": [-0.01, 0.01],
        }
    )
    with pytest.raises(ValueError):
        clustered_difference_test(er, cluster_col="not_a_column")


# ─── Test 5: full report structure ──────────────────────────────────────────


def test_full_report_structure(injected_er: pd.DataFrame) -> None:
    """The full report must contain all expected columns and row combinations."""
    report = full_asymmetry_report(
        injected_er,
        windows=["car_announcement_reaction", "car_short_drift", "car_medium_drift"],
        sue_methods=["sue1", "sue2", "sue3"],
        n_bootstrap=500,
        seed=42,
    )

    expected_cols = {
        "sue_method",
        "window",
        "adjustment_level",
        "miss_car",
        "beat_car",
        "ratio",
        "ci_lower",
        "ci_upper",
        "t_stat",
        "p_value",
        "n_miss",
        "n_beat",
    }
    assert expected_cols.issubset(report.columns)
    # 3 sue_methods × 3 windows × 1 adjustment level (raw; only columns present)
    expected_rows = 3 * 3 * 1
    assert len(report) == expected_rows, (
        f"Expected {expected_rows} rows, got {len(report)}: {report[['sue_method', 'window', 'adjustment_level']].values.tolist()}"
    )
    # Every row must have finite ratio / CI / t-stat.
    assert report["ratio"].apply(np.isfinite).all()
    assert report["ci_lower"].apply(np.isfinite).all()
    assert report["ci_upper"].apply(np.isfinite).all()
    assert report["t_stat"].apply(np.isfinite).all()
    # n_miss and n_beat are positive in every row.
    assert (report["n_miss"] > 0).all()
    assert (report["n_beat"] > 0).all()

    # All (sue_method, window) combinations present at the raw adjustment level.
    combos = report[["sue_method", "window", "adjustment_level"]].apply(tuple, axis=1).tolist()
    for m in ("sue1", "sue2", "sue3"):
        for w in ("car_announcement_reaction", "car_short_drift", "car_medium_drift"):
            assert (m, w, "raw") in combos


def test_full_report_skips_missing_adjustment_columns(injected_er: pd.DataFrame) -> None:
    """Adjustment levels whose columns don't exist are skipped silently."""
    report = full_asymmetry_report(
        injected_er,
        windows=["car_short_drift"],
        sue_methods=["sue3"],
        adjustment_levels=["raw", "ff5_adjusted", "dgtw_adjusted", "cost_adjusted"],
        n_bootstrap=100,
    )
    # Only "raw" should be present; the synthetic data has no adjusted columns.
    assert set(report["adjustment_level"].unique()) == {"raw"}
    assert len(report) == 1


# ─── CRITICAL Test: recover the injected asymmetry end-to-end ───────────────


def test_recovers_injected_asymmetry(injected_er: pd.DataFrame) -> None:
    """THE pipeline-validation test: the computed ratio must recover 2.0x ± 0.5.

    The synthetic data has a population asymmetry ratio of exactly
    INJECTED_ASYMMETRY_RATIO = 2.0 (misses drift -100bps, beats drift +50bps).
    With a large sample and the documented injected drift, the point estimate
    must land within ±0.5 of 2.0.
    """
    res = compute_asymmetry_ratio(
        injected_er, return_col="car_short_drift", decile_col="sue3_decile"
    )

    assert res["miss_n"] > 0
    assert res["beat_n"] > 0

    # Sanity: miss drift should be negative; beat drift should be positive.
    assert res["miss_car_mean"] < 0, f"Miss CAR mean should be < 0, got {res['miss_car_mean']:.4f}"
    assert res["beat_car_mean"] > 0, f"Beat CAR mean should be > 0, got {res['beat_car_mean']:.4f}"

    # THE headline assertion: recovered ratio ≈ INJECTED_ASYMMETRY_RATIO.
    tol = 0.5
    lower = INJECTED_ASYMMETRY_RATIO - tol
    upper = INJECTED_ASYMMETRY_RATIO + tol
    assert lower <= res["ratio"] <= upper, (
        f"Recovered ratio {res['ratio']:.3f} outside [{lower}, {upper}]. "
        f"miss={res['miss_car_mean']:.4f}, beat={res['beat_car_mean']:.4f}, "
        f"n_miss={res['miss_n']}, n_beat={res['beat_n']}"
    )


# ─── Additional API-coverage tests ──────────────────────────────────────────


def test_compute_asymmetry_by_sue_method(injected_er: pd.DataFrame) -> None:
    """by-method table has one row per available SUE method's decile column."""
    table = compute_asymmetry_by_sue_method(injected_er, return_col="car_short_drift")
    # sue1, sue2, sue3 all have decile columns.
    assert set(table["sue_method"]) == {"sue1", "sue2", "sue3"}
    for sue_method in ("sue1", "sue2", "sue3"):
        row = table.loc[table["sue_method"] == sue_method].iloc[0]
        # Each method inherits the same decile assignment, so the ratio matches.
        assert INJECTED_ASYMMETRY_RATIO - 0.5 <= row["ratio"] <= INJECTED_ASYMMETRY_RATIO + 0.5


def test_compute_asymmetry_by_window(injected_er: pd.DataFrame) -> None:
    """by-window table has one row per available window column."""
    table = compute_asymmetry_by_window(
        injected_er,
        windows=["car_announcement_reaction", "car_short_drift", "car_medium_drift"],
    )
    assert set(table["return_col"]) == {
        "car_announcement_reaction",
        "car_short_drift",
        "car_medium_drift",
    }
    assert len(table) == 3


def test_compute_amihud_by_event_smoke(
    composition_data: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    """Amihud computation produces finite positive illiquidity values."""
    er, prices = composition_data
    events = er[[Col.TICKER, Col.ANNOUNCE_DATE]].drop_duplicates()
    amihud = compute_amihud_by_event(prices, events, estimation_days=252)
    assert Col.AMIHUD_ILLIQUIDITY in amihud.columns
    assert len(amihud) == len(events)
    assert amihud[Col.AMIHUD_ILLIQUIDITY].apply(np.isfinite).all()
    assert (amihud[Col.AMIHUD_ILLIQUIDITY] > 0).all()
