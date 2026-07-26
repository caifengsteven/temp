"""
PEAD Asymmetry Pipeline — End-to-end orchestrator.

This module ties together all pipeline stages:
  Data → SUE → Event Returns → Portfolios → Benchmarks → Asymmetry Test → Report

Usage:
  from pead.pipeline import run_pipeline
  results = run_pipeline(use_synthetic=True)
  print(results["headline_table"].to_string())
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from pead.schema import Col, SUEMethod
from pead.synthetic import generate_all_synthetic_data

logger = logging.getLogger(__name__)


# ─── Pipeline stages ────────────────────────────────────────────────────────


def stage_load_data(
    use_synthetic: bool = True,
    n_tickers: int = 200,
    n_quarters: int = 40,
    seed: int = 42,
    bloomberg_config: dict | None = None,
) -> dict[str, pd.DataFrame]:
    """Stage 1: Load data from Bloomberg or synthetic generator."""
    if use_synthetic:
        logger.info("Loading synthetic data (n_tickers=%d, n_quarters=%d)", n_tickers, n_quarters)
        return generate_all_synthetic_data(n_tickers=n_tickers, n_quarters=n_quarters, seed=seed)
    else:
        from pead.ingest.bloomberg import (
            fetch_daily_prices,
            fetch_earnings_estimates,
            fetch_fundamentals,
        )
        from pead.ingest.factors import fetch_ff5_factors

        cfg = bloomberg_config or {}
        tickers = cfg.get("tickers", [])
        start = cfg.get("start_date", "1995-01-01")
        end = cfg.get("end_date", "2024-12-31")
        logger.info("Loading Bloomberg data (%d tickers, %s to %s)", len(tickers), start, end)
        return {
            "earnings_events": fetch_earnings_estimates(tickers, start, end),
            "fundamentals": fetch_fundamentals(tickers, start, end),
            "daily_prices": fetch_daily_prices(tickers, start, end),
            "factors": fetch_ff5_factors(start, end),
        }


def stage_compute_sue(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Stage 2: Compute SUE1, SUE2, SUE3 and merge into one table."""
    from pead.sue.time_series import compute_sue1, compute_sue2
    from pead.sue.analyst import compute_sue3

    events = data["earnings_events"]
    fundamentals = data["fundamentals"]

    _price_cols = [c for c in [Col.PRICE_QUARTER_END, Col.ADJ_FACTOR] if c in fundamentals.columns]
    if _price_cols and Col.PRICE_QUARTER_END not in events.columns:
        _merge_keys = [Col.TICKER, Col.ANNOUNCE_DATE]
        events = events.merge(
            fundamentals[_merge_keys + _price_cols],
            on=_merge_keys,
            how="left",
        )

    logger.info("Computing SUE1 (seasonal random walk)...")
    sue1 = compute_sue1(fundamentals)
    logger.info("  SUE1: %d rows", len(sue1))

    logger.info("Computing SUE2 (ex-special items)...")
    sue2 = compute_sue2(fundamentals)
    logger.info("  SUE2: %d rows", len(sue2))

    logger.info("Computing SUE3 (analyst-median)...")
    sue3 = compute_sue3(events)
    logger.info("  SUE3: %d rows", len(sue3))

    # Merge all three SUE variants
    sue_table = sue1.merge(
        sue2[[Col.TICKER, Col.ANNOUNCE_DATE, Col.SUE2]],
        on=[Col.TICKER, Col.ANNOUNCE_DATE],
        how="outer",
    ).merge(
        sue3[[Col.TICKER, Col.ANNOUNCE_DATE, Col.SUE3]],
        on=[Col.TICKER, Col.ANNOUNCE_DATE],
        how="outer",
    )
    logger.info("Merged SUE table: %d rows", len(sue_table))
    return sue_table


def stage_filter_and_decile(
    sue_table: pd.DataFrame,
    fundamentals: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    """Stage 3: Apply Livnat-Mendenhall filters and assign deciles."""
    from pead.sue.validate import apply_lm_filters, assign_deciles

    sue_cfg = config.get("sue", {})
    universe_cfg = config.get("universe", {})

    logger.info("Applying Livnat-Mendenhall filters...")
    filtered = apply_lm_filters(
        sue_table,
        fundamentals,
        min_price=universe_cfg.get("min_price", 1.0),
        min_mcap_millions=universe_cfg.get("min_mcap_millions", 5.0),
        max_date_diff_days=sue_cfg.get("max_date_diff_days", 1),
    )
    # Keep only non-filtered rows
    n_before = len(filtered)
    filtered = filtered[~filtered["filtered_out"]].copy()
    logger.info(
        "  Filtered: %d → %d rows (%d removed)", n_before, len(filtered), n_before - len(filtered)
    )

    # Assign deciles per cross-section for each SUE method
    for method in [Col.SUE1, Col.SUE3]:
        col = method
        if col not in filtered.columns:
            continue
        filtered = assign_deciles(
            filtered,
            sue_col=col,
            n_deciles=config.get("portfolios", {}).get("n_deciles", 10),
            group_cols=["event_week"] if "event_week" in filtered.columns else None,
        )

    return filtered


def stage_event_returns(
    events: pd.DataFrame,
    prices: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    """Stage 4: Compute CAR/BHAR for all event windows."""
    from pead.events.returns import compute_all_event_returns

    es_cfg = config.get("event_study", {})
    windows_cfg = es_cfg.get(
        "event_windows",
        {
            "announcement_reaction": [0, 1],
            "short_drift": [1, 20],
            "medium_drift": [1, 60],
        },
    )
    windows = [(name, start, end) for name, (start, end) in windows_cfg.items()]

    logger.info("Computing event returns for windows: %s", [w[0] for w in windows])
    returns = compute_all_event_returns(
        events=events,
        prices=prices,
        windows=windows,
        use_midquote=es_cfg.get("use_mid_quote", True),
        estimation_start=es_cfg.get("estimation_window", {}).get("start", -255),
        estimation_end=es_cfg.get("estimation_window", {}).get("end", -46),
    )
    logger.info("  Event returns: %d rows, %d columns", len(returns), len(returns.columns))
    return returns


def stage_asymmetry_test(
    merged: pd.DataFrame,
    prices: pd.DataFrame,
    config: dict,
) -> dict[str, Any]:
    """Stage 5: Run the full asymmetry analysis."""
    from pead.asymmetry.ratio import compute_asymmetry_ratio, compute_asymmetry_by_sue_method
    from pead.asymmetry.inference import (
        bootstrap_asymmetry_ratio,
        clustered_difference_test,
    )
    from pead.asymmetry.double_sort import double_sort_asymmetry, double_sort_by_size

    asym_cfg = config.get("asymmetry", {})
    n_bootstrap = asym_cfg.get("bootstrap_samples", 5000)

    results: dict[str, Any] = {}

    # ── Headline ratio: SUE3, CAR[1,20] ──────────────────────────────────
    logger.info("Computing headline asymmetry ratio (SUE3, CAR[1,20])...")
    ratio = compute_asymmetry_ratio(
        merged,
        sue_method="sue3",
        return_col="car_short_drift",
        decile_col="sue3_decile",
    )
    results["headline_ratio"] = ratio
    logger.info(
        "  Ratio: %.2fx (miss: %.4f%%, beat: %.4f%%)",
        ratio.get("ratio", 0),
        ratio.get("miss_car_mean", 0) * 100,
        ratio.get("beat_car_mean", 0) * 100,
    )

    # ── Bootstrap CI ─────────────────────────────────────────────────────
    logger.info("Bootstrapping asymmetry ratio (%d samples)...", n_bootstrap)
    ci = bootstrap_asymmetry_ratio(
        merged,
        return_col="car_short_drift",
        decile_col="sue3_decile",
        n_bootstrap=min(n_bootstrap, 5000),  # cap for speed
        confidence_level=asym_cfg.get("confidence_level", 0.95),
        seed=42,
    )
    results["bootstrap"] = ci
    logger.info("  95%% CI: [%.2f, %.2f]", ci.get("ci_lower", 0), ci.get("ci_upper", 0))

    # ── Clustered difference test ────────────────────────────────────────
    logger.info("Running clustered difference test...")
    diff_test = clustered_difference_test(
        merged,
        return_col="car_short_drift",
        decile_col="sue3_decile",
    )
    results["difference_test"] = diff_test
    logger.info(
        "  t-stat: %.2f, p-value: %.4f", diff_test.get("t_stat", 0), diff_test.get("p_value", 1)
    )

    # ── Double-sort: Amihud liquidity ────────────────────────────────────
    logger.info("Running liquidity double-sort...")
    liquidity_sort = double_sort_asymmetry(
        merged,
        prices,
        return_col="car_short_drift",
        decile_col="sue3_decile",
        n_liquidity_buckets=5,
    )
    results["liquidity_double_sort"] = liquidity_sort

    # ── Double-sort: Market cap ──────────────────────────────────────────
    logger.info("Running size double-sort...")
    size_sort = double_sort_by_size(
        merged,
        prices,
        return_col="car_short_drift",
        decile_col="sue3_decile",
        n_size_buckets=5,
    )
    results["size_double_sort"] = size_sort

    # ── By SUE method comparison ─────────────────────────────────────────
    logger.info("Comparing asymmetry across SUE methods...")
    by_method = compute_asymmetry_by_sue_method(
        merged,
        return_col="car_short_drift",
    )
    results["by_sue_method"] = by_method

    # ── Mid-quote vs close comparison ────────────────────────────────────
    if "car_midquote_short_drift" in merged.columns:
        logger.info("Computing mid-quote adjusted ratio...")
        mq_ratio = compute_asymmetry_ratio(
            merged,
            sue_method="sue3",
            return_col="car_midquote_short_drift",
            decile_col="sue3_decile",
        )
        results["midquote_ratio"] = mq_ratio

    return results


def stage_subperiod_analysis(
    merged: pd.DataFrame,
    prices: pd.DataFrame,
    config: dict,
) -> dict[str, dict]:
    """Stage 6: Sub-period analysis (Reg-FD, post-2010)."""
    splits = config.get("subperiods", {}).get("splits", [])
    results: dict[str, dict] = {}

    for split in splits:
        name = split["name"]
        start = split.get("start")
        end = split.get("end")

        mask = pd.Series(True, index=merged.index)
        if start:
            mask &= merged[Col.ANNOUNCE_DATE] >= pd.Timestamp(start)
        if end:
            mask &= merged[Col.ANNOUNCE_DATE] <= pd.Timestamp(end)

        subset = merged[mask].copy()
        if len(subset) < 100:
            logger.warning("Sub-period %s has too few events (%d), skipping", name, len(subset))
            continue

        logger.info("Sub-period %s: %d events", name, len(subset))

        from pead.asymmetry.ratio import compute_asymmetry_ratio

        ratio = compute_asymmetry_ratio(
            subset,
            sue_method="sue3",
            return_col="car_short_drift",
            decile_col="sue3_decile",
        )
        results[name] = ratio

    return results


def stage_transaction_cost_adjustment(
    merged: pd.DataFrame,
    prices: pd.DataFrame,
    config: dict,
) -> dict[str, Any]:
    """Stage 7: Compute transaction-cost-adjusted asymmetry ratio."""
    from pead.asymmetry.ratio import compute_asymmetry_ratio
    from pead.portfolios.costs import compute_amihud_illiquidity

    tc_cfg = config.get("transaction_costs", {})

    # Compute Amihud per ticker
    amihud = compute_amihud_illiquidity(prices)

    # Compute per-event cost
    merged_with_amihud = merged.merge(amihud, on=Col.TICKER, how="left")
    merged_with_amihud[Col.AMIHUD_ILLIQUIDITY] = merged_with_amihud[Col.AMIHUD_ILLIQUIDITY].fillna(
        0
    )

    # Amihud-based slippage in bps (documented scaling)
    AMIHUD_BPS_SCALE = 1e10
    amihud_bps = merged_with_amihud[Col.AMIHUD_ILLIQUIDITY] * AMIHUD_BPS_SCALE

    # Round-trip cost per event (both sides)
    commission = tc_cfg.get("commission_bps", 5.0)
    slippage = (
        tc_cfg.get("slippage_base_bps", 5.0) + tc_cfg.get("slippage_amihud_scale", 1.0) * amihud_bps
    )
    round_trip_cost_bps = 2 * (commission + slippage)

    # Borrow cost for shorts (bottom decile only)
    borrow_annual = tc_cfg.get("borrow_cost_annual_bps", 50.0)
    holding_days = 20
    borrow_bps = borrow_annual * (holding_days / 360)

    # Apply cost adjustment to CAR
    decile_col = "sue3_decile" if "sue3_decile" in merged_with_amihud.columns else "decile"
    min_decile = merged_with_amihud[decile_col].min()
    max_decile = merged_with_amihud[decile_col].max()

    cost_adjusted = merged_with_amihud.copy()
    cost_decimal = round_trip_cost_bps / 10000.0

    # Bottom decile (miss/short): subtract round-trip + borrow
    is_short = cost_adjusted[decile_col] == min_decile
    # Top decile (beat/long): subtract round-trip only
    is_long = cost_adjusted[decile_col] == max_decile

    cost_adjusted["car_short_drift_net"] = cost_adjusted["car_short_drift"].copy()
    cost_adjusted.loc[is_short, "car_short_drift_net"] -= (
        cost_decimal[is_short] + borrow_bps / 10000.0
    )
    cost_adjusted.loc[is_long, "car_short_drift_net"] -= cost_decimal[is_long]

    # Compute cost-adjusted ratio
    ratio_net = compute_asymmetry_ratio(
        cost_adjusted,
        sue_method="sue3",
        return_col="car_short_drift_net",
        decile_col=decile_col,
    )

    return {
        "ratio_net": ratio_net,
        "avg_round_trip_cost_bps": float(round_trip_cost_bps.mean()),
        "borrow_cost_bps": float(borrow_bps),
    }


# ─── Master orchestrator ────────────────────────────────────────────────────


@dataclass
class PipelineResults:
    """Container for all pipeline outputs."""

    headline_ratio: dict
    bootstrap: dict
    difference_test: dict
    liquidity_double_sort: pd.DataFrame
    size_double_sort: pd.DataFrame
    by_sue_method: pd.DataFrame
    subperiod_results: dict
    cost_adjusted: dict
    midquote_ratio: dict | None = None
    merged_data: pd.DataFrame | None = None
    headline_table: pd.DataFrame | None = None

    def summary(self) -> str:
        """Return a human-readable summary of results."""
        lines = []
        lines.append("=" * 70)
        lines.append("PEAD ASYMMETRY RESULTS")
        lines.append("=" * 70)
        lines.append("")

        r = self.headline_ratio
        lines.append(f"Headline ratio (SUE3, CAR[1,20]):")
        lines.append(
            f"  Miss drift:  {r.get('miss_car_mean', 0) * 100:+.2f}% (n={r.get('miss_n', 0)})"
        )
        lines.append(
            f"  Beat drift:  {r.get('beat_car_mean', 0) * 100:+.2f}% (n={r.get('beat_n', 0)})"
        )
        lines.append(f"  Ratio:       {r.get('ratio', 0):.2f}x")
        lines.append("")

        b = self.bootstrap
        lines.append(f"Bootstrap 95% CI: [{b.get('ci_lower', 0):.2f}, {b.get('ci_upper', 0):.2f}]")
        d = self.difference_test
        lines.append(f"Clustered t-stat: {d.get('t_stat', 0):.2f} (p={d.get('p_value', 1):.4f})")
        lines.append("")

        if self.cost_adjusted:
            c = self.cost_adjusted["ratio_net"]
            lines.append(f"After transaction costs:")
            lines.append(f"  Ratio:       {c.get('ratio', 0):.2f}x")
            lines.append(
                f"  Avg cost:    {self.cost_adjusted.get('avg_round_trip_cost_bps', 0):.1f} bps round-trip"
            )
            lines.append("")

        if self.midquote_ratio:
            m = self.midquote_ratio
            lines.append(f"Mid-quote adjusted ratio: {m.get('ratio', 0):.2f}x")
            lines.append("")

        lines.append("Liquidity double-sort (Amihud):")
        if self.liquidity_double_sort is not None and len(self.liquidity_double_sort) > 0:
            for _, row in self.liquidity_double_sort.iterrows():
                lines.append(
                    f"  Bucket {row.get('liquidity_bucket', '?')}: "
                    f"ratio={row.get('ratio', 0):.2f}x (n={row.get('n', 0)})"
                )
        lines.append("")

        lines.append("Sub-period analysis:")
        for name, sp in self.subperiod_results.items():
            lines.append(
                f"  {name}: ratio={sp.get('ratio', 0):.2f}x (n_miss={sp.get('miss_n', 0)}, n_beat={sp.get('beat_n', 0)})"
            )
        lines.append("")

        lines.append("=" * 70)
        return "\n".join(lines)


def run_pipeline(
    use_synthetic: bool = True,
    config_path: str = "config/config.yaml",
    n_tickers: int = 200,
    n_quarters: int = 40,
    seed: int = 42,
    bloomberg_config: dict | None = None,
) -> PipelineResults:
    """
    Run the full PEAD asymmetry pipeline end-to-end.

    Args:
        use_synthetic: If True, use synthetic data (for testing). If False, use Bloomberg.
        config_path: Path to config YAML.
        n_tickers: Number of synthetic tickers (ignored if use_synthetic=False).
        n_quarters: Number of synthetic quarters (ignored if use_synthetic=False).
        seed: Random seed for synthetic data.
        bloomberg_config: Dict with tickers/start_date/end_date for Bloomberg mode.

    Returns:
        PipelineResults with all analysis outputs.
    """
    import yaml

    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Stage 1: Load data
    data = stage_load_data(use_synthetic, n_tickers, n_quarters, seed, bloomberg_config)

    # Stage 2: Compute SUE
    sue_table = stage_compute_sue(data)

    # Add event_week for cross-sectional sorting
    sue_table[Col.EVENT_WEEK] = (
        sue_table[Col.ANNOUNCE_DATE].dt.isocalendar()["year"] * 100
        + sue_table[Col.ANNOUNCE_DATE].dt.isocalendar()["week"]
    )

    # Stage 3: Filter and assign deciles
    sue_table = stage_filter_and_decile(sue_table, data["fundamentals"], config)

    # Stage 4: Compute event returns
    events_for_returns = sue_table[
        [Col.TICKER, Col.ANNOUNCE_DATE, Col.FISCAL_QUARTER]
    ].drop_duplicates()
    event_returns = stage_event_returns(events_for_returns, data["daily_prices"], config)

    # Merge SUE table with event returns
    merge_cols = [Col.TICKER, Col.ANNOUNCE_DATE]
    # Avoid column conflicts
    sue_cols = [c for c in sue_table.columns if c not in event_returns.columns or c in merge_cols]
    merged = event_returns.merge(
        sue_table[sue_cols],
        on=merge_cols,
        how="inner",
    )
    logger.info("Merged dataset: %d events with returns and SUE", len(merged))

    # Stage 5: Asymmetry test
    asymmetry_results = stage_asymmetry_test(merged, data["daily_prices"], config)

    # Stage 6: Sub-period analysis
    subperiod_results = stage_subperiod_analysis(merged, data["daily_prices"], config)

    # Stage 7: Transaction cost adjustment
    cost_results = stage_transaction_cost_adjustment(merged, data["daily_prices"], config)

    # Build headline table
    headline_table = _build_headline_table(asymmetry_results, cost_results, subperiod_results)

    return PipelineResults(
        headline_ratio=asymmetry_results["headline_ratio"],
        bootstrap=asymmetry_results["bootstrap"],
        difference_test=asymmetry_results["difference_test"],
        liquidity_double_sort=asymmetry_results["liquidity_double_sort"],
        size_double_sort=asymmetry_results["size_double_sort"],
        by_sue_method=asymmetry_results["by_sue_method"],
        subperiod_results=subperiod_results,
        cost_adjusted=cost_results,
        midquote_ratio=asymmetry_results.get("midquote_ratio"),
        merged_data=merged,
        headline_table=headline_table,
    )


def _build_headline_table(
    asymmetry: dict,
    costs: dict,
    subperiods: dict,
) -> pd.DataFrame:
    """Build the headline comparison table."""
    rows = []

    # Raw ratio
    r = asymmetry["headline_ratio"]
    rows.append(
        {
            "adjustment": "Raw CAR[1,20]",
            "miss_drift_pct": r.get("miss_car_mean", 0) * 100,
            "beat_drift_pct": r.get("beat_car_mean", 0) * 100,
            "ratio": r.get("ratio", 0),
            "n_miss": r.get("miss_n", 0),
            "n_beat": r.get("beat_n", 0),
        }
    )

    # Mid-quote adjusted
    if "midquote_ratio" in asymmetry and asymmetry["midquote_ratio"]:
        m = asymmetry["midquote_ratio"]
        rows.append(
            {
                "adjustment": "Mid-quote adjusted",
                "miss_drift_pct": m.get("miss_car_mean", 0) * 100,
                "beat_drift_pct": m.get("beat_car_mean", 0) * 100,
                "ratio": m.get("ratio", 0),
                "n_miss": m.get("miss_n", 0),
                "n_beat": m.get("beat_n", 0),
            }
        )

    # Cost adjusted
    c = costs["ratio_net"]
    rows.append(
        {
            "adjustment": "After transaction costs",
            "miss_drift_pct": c.get("miss_car_mean", 0) * 100,
            "beat_drift_pct": c.get("beat_car_mean", 0) * 100,
            "ratio": c.get("ratio", 0),
            "n_miss": c.get("miss_n", 0),
            "n_beat": c.get("beat_n", 0),
        }
    )

    # Sub-periods
    for name, sp in subperiods.items():
        rows.append(
            {
                "adjustment": f"Sub-period: {name}",
                "miss_drift_pct": sp.get("miss_car_mean", 0) * 100,
                "beat_drift_pct": sp.get("beat_car_mean", 0) * 100,
                "ratio": sp.get("ratio", 0),
                "n_miss": sp.get("miss_n", 0),
                "n_beat": sp.get("beat_n", 0),
            }
        )

    return pd.DataFrame(rows)
