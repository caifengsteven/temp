"""
Synthetic data generator for testing the PEAD pipeline without Bloomberg.

Generates realistic-looking but fake data with known properties:
- Earnings events with analyst forecasts
- Quarterly fundamentals
- Daily prices with bid/ask spreads
- Factor returns

All data is deterministic (seeded) so tests are reproducible.
The generated data has a KNOWN asymmetry injected so we can verify
the pipeline detects it correctly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Deterministic seed for reproducibility
DEFAULT_SEED = 42


def generate_synthetic_earnings_events(
    n_tickers: int = 200,
    n_quarters: int = 40,
    seed: int = DEFAULT_SEED,
) -> pd.DataFrame:
    """
    Generate synthetic earnings events with analyst forecasts.

    Injects a KNOWN asymmetry: misses drift 2.0x more than beats
    (a realistic ratio, not the 4.5x Reddit claim). This lets us
    verify the pipeline recovers the injected signal.

    Returns DataFrame with schema matching ExpectedSchema.EARNINGS_EVENTS.
    """
    rng = np.random.default_rng(seed)

    tickers = [f"TKR{i:03d} US Equity" for i in range(n_tickers)]
    quarters = pd.date_range("2015-01-01", periods=n_quarters, freq="QS")

    rows = []
    for tkr in tickers:
        # Base EPS grows over time with firm-specific drift
        base_eps = rng.normal(2.0, 0.5)
        eps_drift = rng.normal(0.05, 0.02)
        eps_seasonal = rng.normal(0, 0.1, size=4)  # Q1-Q4 seasonal pattern

        for qi, qstart in enumerate(quarters):
            fq_end = qstart + pd.offsets.QuarterEnd(0)
            # Announce ~30-45 days after quarter end
            announce = fq_end + pd.Timedelta(days=int(rng.integers(28, 46)))

            # True EPS: seasonal random walk with drift + noise
            seasonal = eps_seasonal[qi % 4]
            noise = rng.normal(0, 0.3)
            actual_eps = base_eps + eps_drift * qi + seasonal + noise

            # Analyst median: biased forecast (creates the surprise)
            forecast_bias = rng.normal(0, 0.2)
            medest_eps = actual_eps + forecast_bias

            n_analysts = int(rng.integers(3, 25))

            rows.append(
                {
                    "ticker": tkr,
                    "announce_date": announce,
                    "fq_end": fq_end,
                    "fiscal_quarter": f"{fq_end.year}Q{fq_end.quarter}",
                    "actual_eps": round(actual_eps, 4),
                    "medest_eps": round(medest_eps, 4),
                    "meanest_eps": round(medest_eps + rng.normal(0, 0.05), 4),
                    "n_analysts": n_analysts,
                }
            )

            base_eps = actual_eps  # random walk

    df = pd.DataFrame(rows)
    df["announce_date"] = pd.to_datetime(df["announce_date"])
    df["fq_end"] = pd.to_datetime(df["fq_end"])
    return df


def generate_synthetic_fundamentals(
    events: pd.DataFrame,
    seed: int = DEFAULT_SEED,
) -> pd.DataFrame:
    """Generate Compustat-equivalent quarterly fundamentals."""
    rng = np.random.default_rng(seed + 1)
    df = events.copy()

    df["eps_primary"] = df["actual_eps"]
    df["eps_diluted"] = df["actual_eps"] - rng.normal(0, 0.02, len(df))
    df["special_items"] = rng.normal(0, 0.05, len(df)) * df["actual_eps"].abs()
    df["shares_basic"] = rng.uniform(100, 5000, len(df))  # millions
    df["shares_diluted"] = df["shares_basic"] * rng.uniform(1.0, 1.1, len(df))
    df["price_qe"] = rng.uniform(10, 200, len(df))
    df["report_date"] = df["announce_date"] + pd.to_timedelta(
        rng.integers(-1, 2, len(df)), unit="D"
    )
    df["adj_factor"] = 1.0  # No splits in synthetic data
    return df


def generate_synthetic_prices(
    events: pd.DataFrame,
    n_days_pre: int = 300,
    n_days_post: int = 90,
    injected_asymmetry_ratio: float = 2.0,
    seed: int = DEFAULT_SEED,
) -> pd.DataFrame:
    """
    Generate daily prices with bid/ask around each earnings event.

    INJECTS known PEAD asymmetry:
    - Beats (positive surprise): drift +0.5% over [1, 20]
    - Misses (negative surprise): drift -1.0% over [1, 20] (ratio = 2.0x)

    This is the ground truth the pipeline should recover.
    """
    rng = np.random.default_rng(seed + 2)
    all_prices = []

    for _, ev in events.iterrows():
        tkr = ev["ticker"]
        announce = ev["announce_date"]
        actual = ev["actual_eps"]
        medest = ev["medest_eps"]
        surprise = actual - medest

        # Price level around announce date
        px_base = rng.uniform(10, 200)
        vol_daily = rng.uniform(0.015, 0.035)  # 1.5-3.5% daily vol
        beta = rng.uniform(0.5, 1.5)

        # Compute SUE3 sign (will match pipeline's SUE3 closely since we used medest)
        is_beat = surprise > 0
        is_miss = surprise < 0

        dates = pd.bdate_range(
            start=announce - pd.Timedelta(days=n_days_pre),
            end=announce + pd.Timedelta(days=n_days_post),
        )

        n = len(dates)
        market_ret = rng.normal(0.0003, 0.01, n)  # market drift + noise

        # Firm-specific noise
        firm_ret = rng.normal(0, vol_daily, n)

        # INJECT PEAD: after announce_date, add drift based on surprise sign
        announce_mask = np.asarray(dates >= announce)
        announce_pos = int(announce_mask.argmax())  # first True index
        days_after = np.arange(n) - announce_pos

        # Post-announce drift: 0 to +20 trading days
        drift_mask = (days_after >= 1) & (days_after <= 20)
        drift_per_day = np.zeros(n)

        if is_beat:
            drift_per_day[drift_mask] = 0.00025  # +0.5% total over 20 days
        elif is_miss:
            drift_per_day[drift_mask] = -0.00050  # -1.0% total over 20 days

        # Total return each day
        total_ret = beta * market_ret + firm_ret + drift_per_day
        px_close = px_base * np.cumprod(1 + total_ret)

        # Bid/ask spread (in bps, varies with liquidity)
        spread_bps = rng.uniform(5, 50)
        half_spread = px_close * spread_bps / 20000
        px_bid = px_close - half_spread
        px_ask = px_close + half_spread
        px_midquote = (px_bid + px_ask) / 2

        df = pd.DataFrame(
            {
                "ticker": tkr,
                "trading_date": dates,
                "px_open": px_close * (1 + rng.normal(0, 0.005, n)),
                "px_high": px_close * (1 + np.abs(rng.normal(0, 0.008, n))),
                "px_low": px_close * (1 - np.abs(rng.normal(0, 0.008, n))),
                "px_close": px_close,
                "px_bid": px_bid,
                "px_ask": px_ask,
                "px_midquote": px_midquote,
                "volume": rng.lognormal(15, 1, n),
                "ret": total_ret,
            }
        )
        all_prices.append(df)

    return pd.concat(all_prices, ignore_index=True)


def generate_synthetic_factors(
    start: str = "2014-06-01",
    end: str = "2025-12-31",
    seed: int = DEFAULT_SEED,
) -> pd.DataFrame:
    """Generate synthetic FF5 + momentum monthly factor returns."""
    rng = np.random.default_rng(seed + 3)
    dates = pd.date_range(start, end, freq="ME")
    n = len(dates)

    # Realistic monthly means and vols for FF factors
    return pd.DataFrame(
        {
            "calendar_date": dates,
            "mkt_rf": rng.normal(0.005, 0.045, n),
            "smb": rng.normal(0.002, 0.030, n),
            "hml": rng.normal(-0.001, 0.025, n),
            "rmw": rng.normal(0.002, 0.020, n),
            "cma": rng.normal(0.001, 0.018, n),
            "mom": rng.normal(0.006, 0.040, n),
            "rf": rng.uniform(0.0001, 0.004, n),
        }
    )


def generate_all_synthetic_data(
    n_tickers: int = 200,
    n_quarters: int = 40,
    seed: int = DEFAULT_SEED,
) -> dict[str, pd.DataFrame]:
    """Generate the complete synthetic dataset for end-to-end pipeline testing."""
    events = generate_synthetic_earnings_events(n_tickers, n_quarters, seed)
    fundamentals = generate_synthetic_fundamentals(events, seed)
    prices = generate_synthetic_prices(events, seed=seed)
    factors = generate_synthetic_factors(seed=seed)

    return {
        "earnings_events": events,
        "fundamentals": fundamentals,
        "daily_prices": prices,
        "factors": factors,
    }


# ─── Known ground truth for test assertions ─────────────────────────────────

INJECTED_ASYMMETRY_RATIO = 2.0  # misses drift 2x more than beats in synthetic data
INJECTED_BEAT_DRIFT_BPS = 50  # +0.5% over T+1 to T+20
INJECTED_MISS_DRIFT_BPS = -100  # -1.0% over T+1 to T+20


if __name__ == "__main__":
    # Quick smoke test
    data = generate_all_synthetic_data(n_tickers=50, n_quarters=20)
    print(f"Earnings events: {len(data['earnings_events'])} rows")
    print(f"Fundamentals: {len(data['fundamentals'])} rows")
    print(f"Daily prices: {len(data['daily_prices'])} rows")
    print(f"Factor months: {len(data['factors'])} rows")
    print(
        f"\nGround truth: beats drift +{INJECTED_BEAT_DRIFT_BPS}bps, "
        f"misses drift {INJECTED_MISS_DRIFT_BPS}bps "
        f"(ratio = {INJECTED_ASYMMETRY_RATIO}x)"
    )
