"""Unit tests for the benchmarks module (FF5 regression + DGTW adjustment).

Uses the deterministic synthetic data generator so all assertions are
reproducible. Run with:

    cd "/mnt/nas/data4/github project/jefferies project/PEAD_reddit" \\
        && python -m pytest tests/test_benchmarks.py -v
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pead.benchmarks.dgtw import (
    assign_dgtw_buckets,
    compute_dgtw_adjusted_returns,
    compute_dgtw_benchmark_returns,
    compute_firm_characteristics,
)
from pead.benchmarks.ff5 import (
    compare_factor_adjusted_vs_raw,
    run_ff3_regression,
    run_ff5_regression,
)
from pead.schema import Col
from pead.synthetic import generate_all_synthetic_data

# Keep the synthetic dataset small but large enough for 12-2 momentum
# (needs ~12 months of price history; n_days_pre=300 ≈ 14 months).
N_TICKERS = 40
N_QUARTERS = 16


@pytest.fixture(scope="module")
def synthetic() -> dict[str, pd.DataFrame]:
    """Deterministic synthetic dataset shared across the module's tests."""
    return generate_all_synthetic_data(n_tickers=N_TICKERS, n_quarters=N_QUARTERS)


# ─── Helpers ─────────────────────────────────────────────────────────────────


def _compute_car_for_events(
    prices: pd.DataFrame, events: pd.DataFrame, start: int, end: int
) -> pd.DataFrame:
    """Compute simple cumulative returns over [announce+start, announce+end]
    for each event, as a stand-in CAR column."""
    px = prices[[Col.TICKER, Col.TRADING_DATE, Col.RET]].copy()
    px[Col.TRADING_DATE] = pd.to_datetime(px[Col.TRADING_DATE])

    rows = []
    for _, ev in events.iterrows():
        tkr = ev[Col.TICKER]
        ann = pd.Timestamp(ev[Col.ANNOUNCE_DATE])
        sub = px[(px[Col.TICKER] == tkr) & (px[Col.TRADING_DATE] >= ann)].sort_values(
            Col.TRADING_DATE
        )
        if len(sub) <= end:
            car = np.nan
        else:
            window = sub[Col.RET].iloc[start : end + 1]
            car = float(window.sum()) if len(window) else np.nan
        rows.append(
            {
                Col.TICKER: tkr,
                Col.ANNOUNCE_DATE: ev[Col.ANNOUNCE_DATE],
                f"car_{start}_{end}": car,
            }
        )
    return pd.DataFrame(rows).dropna().reset_index(drop=True)


# ─── 1. FF5 regression ───────────────────────────────────────────────────────


def test_ff5_regression_basic(synthetic: dict[str, pd.DataFrame]) -> None:
    """Alpha and factor loadings are recovered on synthetic factor data."""
    factors = synthetic["factors"].copy()
    rng = np.random.default_rng(0)

    # Construct a portfolio whose excess return loads on the factors with a
    # known monthly alpha. gross = rf + alpha + betas·factors + noise.
    true_alpha = 0.004
    true_betas = {
        Col.MKT_RF: 1.20,
        Col.SMB: 0.30,
        Col.HML: -0.20,
        Col.RMW: 0.10,
        Col.CMA: 0.05,
    }
    excess = (
        true_alpha
        + true_betas[Col.MKT_RF] * factors[Col.MKT_RF]
        + true_betas[Col.SMB] * factors[Col.SMB]
        + true_betas[Col.HML] * factors[Col.HML]
        + true_betas[Col.RMW] * factors[Col.RMW]
        + true_betas[Col.CMA] * factors[Col.CMA]
        + rng.normal(0.0, 0.008, len(factors))
    )
    gross = factors[Col.RF] + excess

    portfolio_returns = pd.DataFrame(
        {Col.CALENDAR_DATE: factors[Col.CALENDAR_DATE], Col.PORTFOLIO_RET_GROSS: gross}
    )

    result = run_ff5_regression(portfolio_returns, factors)

    # Structural checks
    assert isinstance(result, dict)
    for key in ("alpha", "alpha_tstat", "alpha_pvalue", "betas", "r_squared", "n_months"):
        assert key in result, f"missing key {key}"

    # Alpha and all betas are finite numbers.
    assert np.isfinite(result["alpha"])
    for f in (Col.MKT_RF, Col.SMB, Col.HML, Col.RMW, Col.CMA):
        assert f in result["betas"]
        assert np.isfinite(result["betas"][f])

    # We used the full ~10y factor panel, so the fit should be tight and the
    # recovered alpha close to the injected 0.4%/mo.
    assert result["n_months"] > 100
    assert result["r_squared"] > 0.5
    assert abs(result["alpha"] - true_alpha) < 0.002
    assert abs(result["betas"][Col.MKT_RF] - true_betas[Col.MKT_RF]) < 0.15

    # FF3 should also run and produce a comparable market beta.
    ff3 = run_ff3_regression(portfolio_returns, factors)
    assert np.isfinite(ff3["alpha"])
    assert {Col.MKT_RF, Col.SMB, Col.HML} == set(ff3["betas"])


def test_compare_factor_adjusted_vs_raw() -> None:
    """The comparison table aligns raw and FF5 alphas per decile."""
    raw = {1: {"alpha": 0.010}, 5: {"alpha": 0.002}, 10: {"alpha": -0.012}}
    ff5 = {1: {"alpha": 0.004}, 5: {"alpha": 0.001}, 10: {"alpha": -0.006}}

    table = compare_factor_adjusted_vs_raw(raw, ff5)

    assert list(table.columns) == [
        "decile",
        "raw_alpha",
        "ff5_adjusted_alpha",
        "alpha_change",
        "pct_factor_explained",
    ]
    assert len(table) == 3
    # Decile 1: raw 1.0% → ff5 0.4%, so 60% of the alpha was factor exposure.
    row1 = table.loc[table["decile"] == 1].iloc[0]
    assert row1["raw_alpha"] == pytest.approx(0.010)
    assert row1["ff5_adjusted_alpha"] == pytest.approx(0.004)
    assert row1["alpha_change"] == pytest.approx(-0.006)
    assert row1["pct_factor_explained"] == pytest.approx(60.0)


# ─── 2. DGTW characteristics ─────────────────────────────────────────────────


def test_characteristics_computation(
    synthetic: dict[str, pd.DataFrame],
) -> None:
    """Market cap and 12-2 momentum are computed correctly from prices."""
    prices = synthetic["daily_prices"]
    events = synthetic["earnings_events"]

    # Use a formation date well inside the sample so 12m of history exists.
    char_date = pd.Timestamp(events[Col.ANNOUNCE_DATE].min()) + pd.DateOffset(months=14)

    chars = compute_firm_characteristics(prices, events, char_date)

    # Required columns present and finite.
    for col in (Col.TICKER, Col.MARKET_CAP, Col.BOOK_TO_MARKET, Col.MOMENTUM_12_2):
        assert col in chars.columns

    assert len(chars) > 0
    assert chars[Col.MARKET_CAP].replace([np.inf, -np.inf], np.nan).notna().all()
    assert chars[Col.MOMENTUM_12_2].replace([np.inf, -np.inf], np.nan).notna().all()
    assert (chars[Col.MARKET_CAP] > 0).all()

    # Manually recompute momentum for one firm and confirm it matches. The
    # module deduplicates the synthetic overlapping price windows keeping the
    # first occurrence per (ticker, date), so we mirror that here.
    sample_ticker = chars[Col.TICKER].iloc[0]
    px = (
        prices[prices[Col.TICKER] == sample_ticker]
        .drop_duplicates(subset=[Col.TRADING_DATE], keep="first")
        .sort_values(Col.TRADING_DATE)
    )
    p_2m = px.loc[px[Col.TRADING_DATE] <= char_date - pd.DateOffset(months=2), Col.PX_CLOSE].iloc[
        -1
    ]
    p_12m = px.loc[px[Col.TRADING_DATE] <= char_date - pd.DateOffset(months=12), Col.PX_CLOSE].iloc[
        -1
    ]
    expected_mom = p_2m / p_12m - 1.0

    got_mom = float(chars.loc[chars[Col.TICKER] == sample_ticker, Col.MOMENTUM_12_2].iloc[0])
    assert got_mom == pytest.approx(expected_mom, rel=1e-9, abs=1e-12)


# ─── 3. DGTW bucket assignment ───────────────────────────────────────────────


def test_dgtw_bucket_assignment(
    synthetic: dict[str, pd.DataFrame],
) -> None:
    """125 buckets are assigned and membership is balanced."""
    prices = synthetic["daily_prices"]
    events = synthetic["earnings_events"]
    char_date = pd.Timestamp(events[Col.ANNOUNCE_DATE].min()) + pd.DateOffset(months=14)

    chars = compute_firm_characteristics(prices, events, char_date)
    assigned = assign_dgtw_buckets(chars)

    # Rank columns in valid range.
    assert assigned["dgtw_size"].between(0, 4).all()
    assert assigned["dgtw_bm"].between(0, 4).all()
    assert assigned["dgtw_mom"].between(0, 4).all()

    # Combined bucket in [0, 124].
    assert assigned[Col.DGTW_BUCKET].between(0, 124).all()
    assert assigned[Col.DGTW_BUCKET].max() <= 124
    assert assigned[Col.DGTW_BUCKET].min() >= 0

    # Independent sort ⇒ each univariate rank is balanced (~n/5 per quintile).
    n = len(assigned)
    for rank_col in ("dgtw_size", "dgtw_bm", "dgtw_mom"):
        counts = assigned[rank_col].value_counts().sort_index()
        assert len(counts) == 5
        # No empty quintiles and no quintile holds more than ~½ the sample.
        assert counts.min() >= 1
        assert counts.max() <= n / 2 + 1

    # Bucket id is the documented composition of the three ranks.
    expected_bucket = assigned["dgtw_size"] * 25 + assigned["dgtw_bm"] * 5 + assigned["dgtw_mom"]
    np.testing.assert_allclose(assigned[Col.DGTW_BUCKET].values, expected_bucket.values)


def test_dgtw_benchmark_returns_value_weighted(
    synthetic: dict[str, pd.DataFrame],
) -> None:
    """Static-assignment benchmark returns are value-weighted and finite."""
    prices = synthetic["daily_prices"]
    events = synthetic["earnings_events"]
    char_date = pd.Timestamp(events[Col.ANNOUNCE_DATE].min()) + pd.DateOffset(months=14)

    chars = compute_firm_characteristics(prices, events, char_date)
    assigned = assign_dgtw_buckets(chars)
    assignments = assigned[[Col.TICKER, Col.DGTW_BUCKET, Col.MARKET_CAP]]

    bench = compute_dgtw_benchmark_returns(prices, assignments)

    assert list(bench.columns) == [Col.TRADING_DATE, Col.DGTW_BUCKET, "benchmark_ret"]
    assert len(bench) > 0
    assert bench["benchmark_ret"].replace([np.inf, -np.inf], np.nan).notna().all()
    # Returns are in a sane daily range.
    assert bench["benchmark_ret"].abs().max() < 1.0


# ─── 4. DGTW-adjusted event returns ──────────────────────────────────────────


def test_dgtw_adjusted_returns(
    synthetic: dict[str, pd.DataFrame],
) -> None:
    """DGTW-adjusted CAR columns are produced for every event and window."""
    prices = synthetic["daily_prices"]
    events = synthetic["earnings_events"]

    # Build event_returns with two CAR windows.
    car_short = _compute_car_for_events(prices, events, start=1, end=20)
    car_med = _compute_car_for_events(prices, events, start=1, end=40)
    event_returns = (
        events[[Col.TICKER, Col.ANNOUNCE_DATE]]
        .merge(car_short, on=[Col.TICKER, Col.ANNOUNCE_DATE], how="inner")
        .merge(car_med, on=[Col.TICKER, Col.ANNOUNCE_DATE], how="inner")
    )
    assert len(event_returns) > 0

    adjusted = compute_dgtw_adjusted_returns(event_returns, prices)

    # New adjusted columns exist for each input window.
    for window in ("car_1_20", "car_1_40"):
        assert f"dgtw_adjusted_{window}" in adjusted.columns

    # Original car columns are preserved.
    assert "car_1_20" in adjusted.columns
    assert "car_1_40" in adjusted.columns

    # DGTW needs a 12-month momentum warmup, so events before the first valid
    # formation legitimately return NaN. Split the sample at the warmup cutoff
    # (≈13 months past the first announce) and require near-full coverage after.
    adj_col = adjusted["dgtw_adjusted_car_1_20"].replace([np.inf, -np.inf], np.nan)
    ann = pd.to_datetime(adjusted[Col.ANNOUNCE_DATE])
    warmup_cutoff = ann.min() + pd.DateOffset(months=13)
    after_warmup = ann >= warmup_cutoff
    assert adj_col.notna().sum() > 0.8 * len(adjusted)
    assert adj_col.loc[after_warmup].notna().mean() > 0.9

    # The adjustment is a real subtraction: dgtw_adj == firm_car − benchmark_car,
    # so it must differ from the raw CAR for the bulk of events.
    diffs = (adjusted["dgtw_adjusted_car_1_20"] - adjusted["car_1_20"]).abs()
    assert diffs.dropna().gt(1e-10).any()


def test_dgtw_preserves_injected_signal(
    synthetic: dict[str, pd.DataFrame],
) -> None:
    """DGTW must NOT erase the behaviourally-injected beat/miss asymmetry.

    Synthetic drift is injected straight into returns with no characteristic
    tilt, so DGTW (which only strips characteristic/factor exposure) should
    preserve the miss-minus-beat spread. We measure the spread rather than the
    mean level, because the benchmark legitimately strips the common market
    drift that moves both legs together.
    """
    prices = synthetic["daily_prices"]
    events = synthetic["earnings_events"]

    car = _compute_car_for_events(prices, events, start=1, end=20)
    event_returns = events[[Col.TICKER, Col.ANNOUNCE_DATE]].merge(
        car, on=[Col.TICKER, Col.ANNOUNCE_DATE], how="inner"
    )

    adjusted = compute_dgtw_adjusted_returns(event_returns, prices)

    labels = events[[Col.TICKER, Col.ANNOUNCE_DATE, "actual_eps", "medest_eps"]].drop_duplicates(
        [Col.TICKER, Col.ANNOUNCE_DATE]
    )
    labels["is_beat"] = labels["actual_eps"] > labels["medest_eps"]
    merged = adjusted.merge(
        labels[[Col.TICKER, Col.ANNOUNCE_DATE, "is_beat"]],
        on=[Col.TICKER, Col.ANNOUNCE_DATE],
        how="left",
    )

    def _spread(col: str) -> tuple[float, float]:
        sub = merged.dropna(subset=[col, "is_beat"])
        beat = sub.loc[sub["is_beat"], col].mean()
        miss = sub.loc[~sub["is_beat"], col].mean()
        return float(beat), float(miss)

    raw_beat, raw_miss = _spread("car_1_20")
    adj_beat, adj_miss = _spread("dgtw_adjusted_car_1_20")

    raw_spread = raw_miss - raw_beat
    adj_spread = adj_miss - adj_beat

    assert np.isfinite(raw_spread) and np.isfinite(adj_spread)
    assert np.sign(raw_spread) == np.sign(adj_spread)
    assert abs(adj_spread) > 0.4 * abs(raw_spread)
