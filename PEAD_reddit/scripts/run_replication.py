#!/usr/bin/env python3
"""
PEAD Asymmetry Replication — CLI entry point.

Usage:
  # Synthetic data (testing/validation):
  python scripts/run_replication.py --synthetic --tickers 200 --quarters 40

  # Bloomberg data (production):
  python scripts/run_replication.py --bloomberg --tickers-file tickers.txt \
      --start 1995-01-01 --end 2024-12-31

  # Save results to CSV:
  python scripts/run_replication.py --synthetic --output data/processed/
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure src is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def main() -> int:
    parser = argparse.ArgumentParser(description="PEAD Asymmetry Replication Pipeline")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--synthetic", action="store_true", help="Use synthetic data for testing")
    mode.add_argument(
        "--bloomberg", action="store_true", help="Use Bloomberg data (requires terminal)"
    )

    parser.add_argument("--tickers", type=int, default=200, help="Number of synthetic tickers")
    parser.add_argument("--quarters", type=int, default=40, help="Number of synthetic quarters")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--config", default="config/config.yaml", help="Config file path")

    # Bloomberg options
    parser.add_argument("--tickers-file", help="File with one ticker per line (Bloomberg mode)")
    parser.add_argument("--start", default="1995-01-01", help="Start date (Bloomberg mode)")
    parser.add_argument("--end", default="2024-12-31", help="End date (Bloomberg mode)")

    parser.add_argument("--output", "-o", default=None, help="Output directory for results CSV")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    from pead.pipeline import run_pipeline

    bloomberg_config = None
    if args.bloomberg:
        tickers = []
        if args.tickers_file:
            with open(args.tickers_file) as f:
                tickers = [line.strip() for line in f if line.strip() and not line.startswith("#")]
        bloomberg_config = {
            "tickers": tickers,
            "start_date": args.start,
            "end_date": args.end,
        }

    print("\n" + "=" * 70)
    print("  PEAD ASYMMETRY REPLICATION PIPELINE")
    print("  Testing: Do earnings misses drift more than beats at T+20?")
    print("=" * 70 + "\n")

    results = run_pipeline(
        use_synthetic=args.synthetic,
        config_path=args.config,
        n_tickers=args.tickers,
        n_quarters=args.quarters,
        seed=args.seed,
        bloomberg_config=bloomberg_config,
    )

    # Print results
    print(results.summary())

    # Print headline table
    if results.headline_table is not None:
        print("\nHeadline Comparison Table:")
        print("-" * 70)
        print(results.headline_table.to_string(index=False, float_format="%.2f"))

    # Print bootstrap details
    b = results.bootstrap
    print(f"\nBootstrap details:")
    print(f"  Point estimate: {b.get('ratio_point_estimate', 0):.2f}x")
    print(f"  95% CI:         [{b.get('ci_lower', 0):.2f}, {b.get('ci_upper', 0):.2f}]")
    print(f"  Std error:      {b.get('bootstrap_std', 0):.2f}")

    # Save results if requested
    if args.output:
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)

        if results.headline_table is not None:
            results.headline_table.to_csv(output_dir / "headline_table.csv", index=False)

        if results.liquidity_double_sort is not None and len(results.liquidity_double_sort) > 0:
            results.liquidity_double_sort.to_csv(
                output_dir / "liquidity_double_sort.csv", index=False
            )

        if results.size_double_sort is not None and len(results.size_double_sort) > 0:
            results.size_double_sort.to_csv(output_dir / "size_double_sort.csv", index=False)

        if results.by_sue_method is not None and len(results.by_sue_method) > 0:
            results.by_sue_method.to_csv(output_dir / "by_sue_method.csv", index=False)

        print(f"\nResults saved to {output_dir}/")

    print("\n" + "=" * 70)
    print("  Pipeline complete.")
    print("=" * 70 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
