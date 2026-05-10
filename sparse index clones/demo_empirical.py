"""
demo_empirical.py
Demonstration of rolling-window SLOPE index tracking with synthetic data.

This script generates synthetic constituent/index returns and runs the
full rolling-window backtest comparing SLOPE, SLOPE-SLC, and LASSO.
"""

import numpy as np
import pandas as pd
from slope_empirical import rolling_window_backtest
from slope_solver import compute_lambda_sequence_bogdan


def generate_synthetic_equity_data(T=1500, K=50, n_factors=5, seed=42):
    """
    Generate synthetic daily returns for K constituents and 1 index.

    Model:
    - Factors ~ N(0, sigma_f^2)
    - Constituents = B' @ F + epsilon
    - Index = R @ w_true + nu
    """
    rng = np.random.default_rng(seed)

    # Factor returns
    F = rng.normal(0, 0.01, (T, n_factors))

    # Factor loadings (some grouping structure)
    B = rng.uniform(0.1, 0.5, (n_factors, K))
    # Create 5 groups with similar loadings
    for g in range(5):
        idx = slice(g * 10, (g + 1) * 10)
        B[:, idx] = B[:, idx].mean(axis=1, keepdims=True) + rng.normal(0, 0.02, (n_factors, 10))

    # Constituent returns
    epsilon = rng.normal(0, 0.015, (T, K))
    R = F @ B + epsilon

    # True index weights (sparse, 3 groups active)
    w_true = np.zeros(K)
    w_true[10:20] = 0.05   # group 2
    w_true[30:40] = 0.03   # group 4
    w_true[40:45] = 0.02   # group 5 partial
    w_true /= w_true.sum()

    nu = rng.normal(0, 0.005, T)
    Y = R @ w_true + nu

    # Build DataFrames
    dates = pd.date_range('2010-01-01', periods=T, freq='B')
    R_df = pd.DataFrame(R, index=dates, columns=[f'Asset_{i+1}' for i in range(K)])
    Y_s = pd.Series(Y, index=dates, name='Index')

    return R_df, Y_s, w_true


def main():
    print("="*80)
    print("SLOPE INDEX TRACKING - EMPIRICAL DEMO")
    print("="*80)

    # Generate data
    print("\nGenerating synthetic equity data (T=1500, K=50)...")
    R_df, Y_s, w_true = generate_synthetic_equity_data(T=1500, K=50, seed=42)
    print(f"True index weights: {int(np.sum(w_true > 0))} active assets, "
          f"{len(np.unique(np.round(w_true[w_true>0], 4)))} groups")

    R = R_df.values
    Y = Y_s.values
    T, K = R.shape

    # Lambda sequence
    lambda_alpha = 0.005
    lambda_seq = compute_lambda_sequence_bogdan(K, lambda_alpha, theta=0.1)
    print(f"Lambda sequence: alpha={lambda_alpha}, K={K}, "
          f"lambda_1={lambda_seq[0]:.4f}, lambda_K={lambda_seq[-1]:.6f}")

    # Backtest parameters
    window = 500
    step = 21  # monthly
    print(f"Rolling window: {window} days, rebalance every {step} days")

    strategies = ['slope', 'slope-slc', 'lasso']
    all_results = {}

    for strat in strategies:
        print(f"\n--- Running {strat.upper()} ---")
        res = rolling_window_backtest(
            R, Y, window_size=window, step_size=step,
            lambda_seq=lambda_seq, strategy=strat,
            constraint='simplex', percentile=75.0,
            warm_start=True, verbose=False
        )
        all_results[strat] = res
        print(f"  OOS steps:        {res['n_steps']}")
        print(f"  Mean sparsity:    {res['mean_sparsity']:.1f} / {K}")
        print(f"  Mean groups:      {res['mean_groups']:.1f}")
        print(f"  TE Volatility:    {res['te_vol']:.4f}")
        print(f"  RMSE:             {res['rmse']:.4f}")
        print(f"  Mean turnover:    {res['mean_turnover']:.4f}")

    # Summary comparison
    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)
    print(f"{'Strategy':>12} {'Sparsity':>10} {'Groups':>8} {'TE_Vol':>10} "
          f"{'RMSE':>10} {'Turnover':>10}")
    print("-"*60)
    for strat in strategies:
        r = all_results[strat]
        print(f"{strat:>12} {r['mean_sparsity']:>10.1f} {r['mean_groups']:>8.1f} "
              f"{r['te_vol']:>10.4f} {r['rmse']:>10.4f} {r['mean_turnover']:>10.4f}")

    # Plotting (optional, if matplotlib available)
    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        # Tracking error over time
        ax = axes[0, 0]
        for strat in strategies:
            r = all_results[strat]
            ax.plot(np.cumsum(r['oos_returns']), label=strat)
        ax.set_title('Cumulative Tracking Error')
        ax.set_xlabel('Out-of-sample period')
        ax.set_ylabel('Cumul. TE')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Sparsity over time
        ax = axes[0, 1]
        for strat in strategies:
            r = all_results[strat]
            ax.plot(r['sparsity'], label=strat, marker='o', markersize=2)
        ax.set_title('Portfolio Sparsity Over Time')
        ax.set_xlabel('Rebalance step')
        ax.set_ylabel('Number of active assets')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Turnover over time
        ax = axes[1, 0]
        for strat in strategies:
            r = all_results[strat]
            ax.plot(r['turnover'], label=strat, marker='o', markersize=2)
        ax.set_title('Turnover Over Time')
        ax.set_xlabel('Rebalance step')
        ax.set_ylabel('Turnover')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Final weights comparison
        ax = axes[1, 1]
        x = np.arange(K)
        width = 0.25
        for i, strat in enumerate(strategies):
            w_final = all_results[strat]['weights'][-1]
            ax.bar(x + i*width, w_final, width, label=strat)
        ax.set_title('Final Portfolio Weights')
        ax.set_xlabel('Asset')
        ax.set_ylabel('Weight')
        ax.legend()

        plt.tight_layout()
        plt.savefig('slope_demo_results.png', dpi=150)
        print("\nPlot saved to slope_demo_results.png")
    except ImportError:
        print("\n(matplotlib not available, skipping plots)")


if __name__ == '__main__':
    main()
