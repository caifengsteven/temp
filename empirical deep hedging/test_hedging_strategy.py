"""
Test Hedging Strategy

This script tests the hedging strategy with mock data.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Create a simple mock data generator
def generate_mock_data(start_date='2020-01-01', end_date='2023-12-31'):
    """Generate mock price data for testing"""
    # Create date range
    dates = pd.date_range(start=start_date, end=end_date, freq='B')

    # Generate asset price series with trend and volatility
    np.random.seed(42)  # For reproducibility
    asset_returns = np.random.normal(0.0005, 0.01, len(dates))
    asset_prices = 100 * np.cumprod(1 + asset_returns)

    # Generate hedge instrument price series (negatively correlated with asset)
    hedge_returns = -0.8 * asset_returns + np.random.normal(0, 0.005, len(dates))
    hedge_prices = 50 * np.cumprod(1 + hedge_returns)

    # Create DataFrame
    data = pd.DataFrame({
        'asset_price': asset_prices,
        'hedge_price': hedge_prices,
        'asset_return': asset_returns,
        'hedge_return': hedge_returns
    }, index=dates)

    return data

def simple_backtest(data, rebalance_freq='W', hedge_ratio_method='fixed', hedge_ratio=1.0):
    """
    Simple backtest of a hedging strategy

    Parameters:
    -----------
    data : pandas.DataFrame
        Price and return data
    rebalance_freq : str
        Rebalancing frequency ('D' for daily, 'W' for weekly, 'M' for monthly)
    hedge_ratio_method : str
        Method to calculate hedge ratio ('fixed', 'beta', 'rolling_beta')
    hedge_ratio : float
        Fixed hedge ratio (only used if method is 'fixed')

    Returns:
    --------
    pandas.DataFrame
        Backtest results
    """
    # Initialize variables
    initial_capital = 1000000
    capital = initial_capital
    asset_position = 0
    hedge_position = 0

    # Determine rebalance dates
    if rebalance_freq == 'D':
        rebalance_dates = data.index
    else:
        rebalance_dates = pd.date_range(start=data.index[0], end=data.index[-1], freq=rebalance_freq)

    # Initialize results
    results = []

    # Run backtest
    for date in data.index:
        # Check if we need to rebalance
        if date in rebalance_dates or asset_position == 0:
            # Calculate hedge ratio
            if hedge_ratio_method == 'fixed':
                current_hedge_ratio = hedge_ratio
            elif hedge_ratio_method == 'beta':
                # Calculate beta using all available data up to this point
                asset_rets = data.loc[:date, 'asset_return']
                hedge_rets = data.loc[:date, 'hedge_return']
                if len(asset_rets) > 10:  # Need enough data points
                    cov_matrix = np.cov(asset_rets, hedge_rets)
                    current_hedge_ratio = cov_matrix[0, 1] / cov_matrix[1, 1]
                else:
                    current_hedge_ratio = hedge_ratio
            elif hedge_ratio_method == 'rolling_beta':
                # Calculate beta using rolling window
                lookback = 60  # 60 business days (about 3 months)
                asset_rets = data.loc[:date, 'asset_return'].tail(lookback)
                hedge_rets = data.loc[:date, 'hedge_return'].tail(lookback)
                if len(asset_rets) > 10:  # Need enough data points
                    cov_matrix = np.cov(asset_rets, hedge_rets)
                    current_hedge_ratio = cov_matrix[0, 1] / cov_matrix[1, 1]
                else:
                    current_hedge_ratio = hedge_ratio

            # Calculate positions
            asset_position = capital * 0.5 / data.loc[date, 'asset_price']  # Invest half of capital in asset
            hedge_position = -asset_position * current_hedge_ratio  # Negative for short position

        # Calculate daily P&L
        if date > data.index[0]:
            asset_pnl = asset_position * data.loc[date, 'asset_price'] * data.loc[date, 'asset_return']
            hedge_pnl = hedge_position * data.loc[date, 'hedge_price'] * data.loc[date, 'hedge_return']
            daily_pnl = asset_pnl + hedge_pnl
            capital += daily_pnl

        # Record results
        results.append({
            'date': date,
            'capital': capital,
            'asset_position': asset_position,
            'hedge_position': hedge_position,
            'asset_price': data.loc[date, 'asset_price'],
            'hedge_price': data.loc[date, 'hedge_price'],
            'asset_return': data.loc[date, 'asset_return'],
            'hedge_return': data.loc[date, 'hedge_return'],
            'hedge_ratio': current_hedge_ratio if 'current_hedge_ratio' in locals() else hedge_ratio
        })

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    results_df.set_index('date', inplace=True)

    # Calculate performance metrics
    results_df['daily_return'] = results_df['capital'].pct_change().fillna(0)
    results_df['cumulative_return'] = (1 + results_df['daily_return']).cumprod() - 1

    # Calculate annualized metrics
    trading_days = 252
    results_df['annualized_return'] = results_df['cumulative_return'].iloc[-1] * (trading_days / len(results_df))
    results_df['annualized_volatility'] = results_df['daily_return'].std() * np.sqrt(trading_days)

    # Calculate Sharpe ratio
    ann_vol = results_df['annualized_volatility'].iloc[-1]
    if ann_vol > 0:
        results_df['sharpe_ratio'] = results_df['annualized_return'].iloc[-1] / ann_vol
    else:
        results_df['sharpe_ratio'] = 0

    return results_df

def plot_results(results, title="Hedging Strategy Results"):
    """Plot backtest results"""
    plt.figure(figsize=(12, 8))

    # Plot equity curve
    plt.subplot(2, 1, 1)
    plt.plot(results.index, results['capital'])
    plt.title(f'{title} - Equity Curve')
    plt.xlabel('Date')
    plt.ylabel('Capital')
    plt.grid(True)

    # Plot hedge ratio
    plt.subplot(2, 1, 2)
    plt.plot(results.index, results['hedge_ratio'])
    plt.title('Hedge Ratio')
    plt.xlabel('Date')
    plt.ylabel('Ratio')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(f'{title.lower().replace(" ", "_")}_results.png')
    plt.close()

def main():
    """Main function to test the hedging strategy"""
    print("Testing hedging strategy with mock data...")

    # Generate mock data
    data = generate_mock_data(start_date='2020-01-01', end_date='2023-12-31')

    # Test different hedging methods
    methods = ['fixed', 'beta', 'rolling_beta']

    for method in methods:
        print(f"\nTesting {method} hedging method:")

        # Backtest the strategy
        results = simple_backtest(
            data=data,
            rebalance_freq='W',  # Weekly rebalancing
            hedge_ratio_method=method,
            hedge_ratio=1.0  # Default hedge ratio
        )

        # Plot results
        plot_results(results, title=f"{method.replace('_', ' ').title()} Hedging")

        # Print summary statistics
        print("\nSummary Statistics:")
        print(f"Final Capital: ${results['capital'].iloc[-1]:,.2f}")
        print(f"Total Return: {results['cumulative_return'].iloc[-1]:.2%}")
        print(f"Annualized Return: {results['annualized_return'].iloc[-1]:.2%}")
        print(f"Annualized Volatility: {results['annualized_volatility'].iloc[-1]:.2%}")
        print(f"Sharpe Ratio: {results['sharpe_ratio'].iloc[-1]:.2f}")

        # Calculate drawdown
        peak = results['capital'].cummax()
        drawdown = (results['capital'] - peak) / peak
        max_drawdown = drawdown.min()

        print(f"Maximum Drawdown: {max_drawdown:.2%}")

if __name__ == "__main__":
    main()
