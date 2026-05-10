"""
test_real_data.py
Test SLOPE index tracking with real S&P 500 data from yfinance.
"""

import numpy as np
import pandas as pd
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

from slope_empirical import rolling_window_backtest
from slope_solver import compute_lambda_sequence_bogdan


# Top 80 S&P 500 constituents by approximate market cap (diverse sectors)
TICKERS = [
    'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'NVDA', 'META', 'BRK-B', 'TSLA', 'AVGO',
    'JPM', 'LLY', 'V', 'UNH', 'WMT', 'JNJ', 'XOM', 'MA', 'PG', 'HD',
    'CVX', 'MRK', 'ABBV', 'PEP', 'KO', 'BAC', 'COST', 'TMO', 'PFE', 'ABT',
    'MCD', 'ADBE', 'CSCO', 'DIS', 'ACN', 'CRM', 'VZ', 'NKE', 'TXN', 'QCOM',
    'NEE', 'PM', 'RTX', 'INTC', 'HON', 'AMGN', 'IBM', 'UNP', 'LOW', 'SPGI',
    'CAT', 'UPS', 'GS', 'SBUX', 'MS', 'BLK', 'MDT', 'GILD', 'AMAT', 'BKNG',
    'T', 'LMT', 'DE', 'SYK', 'ADP', 'ELV', 'C', 'CI', 'MDLZ', 'CB',
    'MMC', 'VRTX', 'SCHW', 'MO', 'SO', 'DUK', 'PNC', 'ISRG', 'USB', 'PGR',
]
INDEX_TICKER = '^GSPC'


def download_data(start='2015-01-01', end='2024-12-31'):
    """Download constituent and index price data."""
    print(f"Downloading {len(TICKERS)} constituents + {INDEX_TICKER}...")
    print("This may take 30-60 seconds...")

    all_tickers = TICKERS + [INDEX_TICKER]
    data = yf.download(
        all_tickers, start=start, end=end,
        auto_adjust=True, progress=False, threads=True
    )

    # Handle MultiIndex columns from yfinance
    if isinstance(data.columns, pd.MultiIndex):
        prices = data['Close'].copy()
        if isinstance(prices.columns, pd.MultiIndex):
            prices.columns = prices.columns.get_level_values(1)
    else:
        prices = data['Close'].copy()

    # Separate index
    index_prices = prices[INDEX_TICKER]
    constituent_prices = prices.drop(columns=[INDEX_TICKER], errors='ignore')

    print(f"Downloaded: {constituent_prices.shape[1]} constituents, "
          f"{len(constituent_prices)} days")
    return constituent_prices, index_prices


def preprocess_returns(prices_df, index_series, min_coverage=0.90):
    """
    Convert prices to returns and filter by coverage.
    """
    # Forward fill prices, then compute returns
    prices_df = prices_df.ffill()
    index_series = index_series.ffill()

    # Drop columns that are entirely NaN (delisted / unavailable)
    prices_df = prices_df.dropna(how='all', axis=1)

    # Coverage filter: keep tickers with >= min_coverage non-NaN
    coverage = prices_df.notna().sum() / len(prices_df)
    prices_df = prices_df.loc[:, coverage >= min_coverage]

    # Fill remaining NaNs
    prices_df = prices_df.ffill().bfill()

    # Compute returns
    R = prices_df.pct_change().dropna()
    Y = index_series.pct_change().dropna()

    # Align
    common_idx = R.index.intersection(Y.index)
    R = R.loc[common_idx]
    Y = Y.loc[common_idx]

    print(f"After filtering: {R.shape[1]} constituents, {len(R)} days")
    print(f"Date range: {R.index[0]} to {R.index[-1]}")
    return R, Y


def run_backtest(R, Y, window=750, step=21):
    """Run rolling window backtest with multiple strategies."""
    T, K = R.shape
    print(f"\nBacktest setup: T={T}, K={K}, window={window}, step={step}")

    # Lambda sequence - tune alpha to target reasonable sparsity
    # Try a few alpha values and pick one that gives ~10-30 active assets
    alpha_candidates = [0.0005, 0.001, 0.002, 0.005, 0.01]
    lambda_seqs = {a: compute_lambda_sequence_bogdan(K, a, theta=0.1) for a in alpha_candidates}

    strategies = ['slope', 'slope-slc', 'lasso']
    all_results = {}

    for alpha in alpha_candidates:
        print(f"\n--- Testing alpha={alpha} ---")
        lambda_seq = lambda_seqs[alpha]

        for strat in strategies:
            res = rolling_window_backtest(
                R.values, Y.values,
                window_size=window, step_size=step,
                lambda_seq=lambda_seq, strategy=strat,
                constraint='simplex', percentile=75.0,
                warm_start=True, verbose=False
            )
            key = f"{strat}_a{alpha}"
            all_results[key] = res
            print(f"  {strat:12s}: sparsity={res['mean_sparsity']:5.1f}, "
                  f"groups={res['mean_groups']:5.1f}, TE_vol={res['te_vol']:6.4f}, "
                  f"RMSE={res['rmse']:6.4f}, TO={res['mean_turnover']:6.4f}")

    return all_results


def print_best_results(all_results):
    """Print summary of best configurations."""
    print("\n" + "="*90)
    print("BEST CONFIGURATIONS SUMMARY")
    print("="*90)

    # Best by TE volatility
    print("\n--- Lowest Tracking Error Volatility ---")
    best_te = min(all_results.items(), key=lambda x: x[1]['te_vol'] if not np.isnan(x[1]['te_vol']) else 999)
    print(f"  {best_te[0]}: TE_vol={best_te[1]['te_vol']:.4f}")

    # Best by sparsity (most sparse with decent TE)
    print("\n--- Most Sparse SLOPE-SLC with TE_vol < 0.10 ---")
    candidates = {k: v for k, v in all_results.items()
                  if 'slc' in k and v['te_vol'] < 0.10}
    if candidates:
        best_sparse = min(candidates.items(), key=lambda x: x[1]['mean_sparsity'])
        print(f"  {best_sparse[0]}: sparsity={best_sparse[1]['mean_sparsity']:.1f}, "
              f"TE_vol={best_sparse[1]['te_vol']:.4f}")
    else:
        print("  No candidates with TE_vol < 0.10")

    # Full table
    print("\n" + "-"*90)
    print(f"{'Config':>20} {'Sparsity':>10} {'Groups':>8} {'TE_Vol':>10} "
          f"{'RMSE':>10} {'Turnover':>10}")
    print("-"*90)
    for key in sorted(all_results.keys()):
        r = all_results[key]
        print(f"{key:>20} {r['mean_sparsity']:>10.1f} {r['mean_groups']:>8.1f} "
              f"{r['te_vol']:>10.4f} {r['rmse']:>10.4f} {r['mean_turnover']:>10.4f}")


def plot_results(all_results, R, Y, window, step):
    """Plot cumulative tracking errors and weight evolution."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n(matplotlib not available, skipping plots)")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Cumulative tracking error for best alpha
    ax = axes[0, 0]
    alpha_plot = 0.002  # pick one alpha for comparison
    for strat in ['slope', 'slope-slc', 'lasso']:
        key = f"{strat}_a{alpha_plot}"
        if key in all_results:
            r = all_results[key]
            cumte = np.cumsum(r['oos_returns'])
            ax.plot(cumte, label=strat, linewidth=1.5)
    ax.axhline(0, color='black', linestyle='--', alpha=0.3)
    ax.set_title(f'Cumulative Tracking Error (alpha={alpha_plot})')
    ax.set_xlabel('Out-of-sample days')
    ax.set_ylabel('Cumulative TE')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Sparsity over time for best alpha
    ax = axes[0, 1]
    for strat in ['slope', 'slope-slc', 'lasso']:
        key = f"{strat}_a{alpha_plot}"
        if key in all_results:
            r = all_results[key]
            ax.plot(r['sparsity'], label=strat, marker='o', markersize=2, linewidth=1)
    ax.set_title(f'Portfolio Sparsity Over Time (alpha={alpha_plot})')
    ax.set_xlabel('Rebalance step')
    ax.set_ylabel('Active assets')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. TE volatility vs sparsity tradeoff
    ax = axes[1, 0]
    for strat in ['slope', 'slope-slc', 'lasso']:
        sparsities = []
        te_vols = []
        alphas = []
        for key, r in all_results.items():
            if key.startswith(strat):
                sparsities.append(r['mean_sparsity'])
                te_vols.append(r['te_vol'])
                # extract alpha from key
                try:
                    a = float(key.split('_a')[1])
                    alphas.append(a)
                except:
                    alphas.append(0)
        ax.scatter(sparsities, te_vols, label=strat, s=80, alpha=0.7)
        for i, a in enumerate(alphas):
            ax.annotate(f'{a:.4f}', (sparsities[i], te_vols[i]), fontsize=7, alpha=0.6)
    ax.set_title('Sparsity vs Tracking Error Tradeoff')
    ax.set_xlabel('Mean Sparsity (# active assets)')
    ax.set_ylabel('Tracking Error Volatility')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Turnover comparison
    ax = axes[1, 1]
    for strat in ['slope', 'slope-slc', 'lasso']:
        key = f"{strat}_a{alpha_plot}"
        if key in all_results:
            r = all_results[key]
            ax.plot(r['turnover'], label=strat, marker='o', markersize=2, linewidth=1)
    ax.set_title(f'Turnover Over Time (alpha={alpha_plot})')
    ax.set_xlabel('Rebalance step')
    ax.set_ylabel('Turnover')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('slope_real_data_results.png', dpi=150)
    print("\nPlot saved to slope_real_data_results.png")


def main():
    print("="*80)
    print("SLOPE INDEX TRACKING - REAL DATA TEST (S&P 500)")
    print("="*80)

    # Download
    constituent_prices, index_prices = download_data(
        start='2015-01-01', end='2024-12-31'
    )

    # Preprocess
    R, Y = preprocess_returns(constituent_prices, index_prices)

    if R.shape[1] < 10:
        print("ERROR: Too few constituents after filtering. Aborting.")
        return

    # Run backtest
    all_results = run_backtest(R, Y, window=750, step=21)

    # Print summary
    print_best_results(all_results)

    # Plot
    plot_results(all_results, R, Y, window=750, step=21)

    print("\nDone!")


if __name__ == '__main__':
    main()
