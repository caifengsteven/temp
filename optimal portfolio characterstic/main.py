"""
Main Script for Optimal Characteristic Portfolios

This script implements the optimal characteristic portfolios strategy
from the paper by Richard McGee and Jose Olmo using simulated data.
It uses simulated stock universe, characteristics, and returns data.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
from nonparametric import NonParametricEstimator
from portfolio import OptimalCharacteristicPortfolio
from stock_universe import generate_stock_universe, generate_characteristics


def run_strategy(start_date, end_date, characteristic_type='size',
                rebalance_freq='monthly', lookback_periods=12, gamma=1.0, num_stocks=100):
    """
    Run the optimal characteristic portfolio strategy

    Parameters:
    -----------
    start_date : str or datetime
        Start date for the backtest
    end_date : str or datetime
        End date for the backtest
    characteristic_type : str
        Type of characteristic to use ('size', 'value', 'momentum')
    rebalance_freq : str
        Rebalancing frequency ('monthly', 'quarterly', 'annual')
    lookback_periods : int
        Number of periods to use for estimating the relationship
    gamma : float
        Risk aversion parameter
    num_stocks : int
        Number of stocks in the simulated universe

    Returns:
    --------
    pandas.DataFrame
        DataFrame with portfolio weights over time
    pandas.Series
        Series with portfolio returns over time
    dict
        Dictionary with performance metrics
    """
    print(f"Running optimal characteristic portfolio strategy for {characteristic_type}...")

    # Convert dates to datetime if they're strings
    if isinstance(start_date, str):
        start_date = dt.datetime.strptime(start_date, '%Y-%m-%d')
    if isinstance(end_date, str):
        end_date = dt.datetime.strptime(end_date, '%Y-%m-%d')

    # Adjust start date to include lookback period
    if rebalance_freq == 'monthly':
        adjusted_start = start_date - dt.timedelta(days=30*lookback_periods)
    elif rebalance_freq == 'quarterly':
        adjusted_start = start_date - dt.timedelta(days=90*lookback_periods)
    elif rebalance_freq == 'annual':
        adjusted_start = start_date - dt.timedelta(days=365*lookback_periods)
    else:
        adjusted_start = start_date - dt.timedelta(days=30*lookback_periods)

    # Generate dates for simulation
    dates = pd.date_range(start=adjusted_start, end=end_date, freq='M')

    # Generate simulated stock universe
    print("Generating simulated stock universe...")
    universe = generate_stock_universe(
        adjusted_start,
        end_date,
        num_stocks=num_stocks,
        seed=42
    )

    # Generate simulated characteristics
    print(f"Generating simulated {characteristic_type} characteristics...")
    characteristics_history = generate_characteristics(
        universe,
        characteristic_type=characteristic_type,
        seed=42
    )

    # Get all unique tickers from the universe
    all_tickers = []
    for date in universe:
        all_tickers.extend(universe[date])
    all_tickers = list(set(all_tickers))

    # Generate simulated returns data
    print("Generating simulated returns data...")
    returns_data = {}

    for ticker in all_tickers:
        # Generate random returns with some correlation to the characteristic
        if characteristic_type == 'size':
            # Smaller stocks tend to have higher returns but more volatility
            mean_return = 0.01  # 1% monthly
            volatility = 0.05   # 5% monthly
        elif characteristic_type == 'value':
            # Value stocks tend to have higher returns
            mean_return = 0.008  # 0.8% monthly
            volatility = 0.04    # 4% monthly
        elif characteristic_type == 'momentum':
            # Momentum stocks tend to continue their trend
            mean_return = 0.012  # 1.2% monthly
            volatility = 0.06    # 6% monthly
        else:
            mean_return = 0.01
            volatility = 0.05

        returns_data[ticker] = pd.Series(
            np.random.normal(mean_return, volatility, len(dates)),
            index=dates
        )

    returns_history = pd.DataFrame(returns_data)

    # Generate simulated risk-free rate
    print("Generating simulated risk-free rate...")
    risk_free_rate = pd.Series(
        np.random.uniform(0.001, 0.003, len(dates)),  # 0.1% to 0.3% monthly
        index=dates
    )

    # Create portfolio constructor
    portfolio = OptimalCharacteristicPortfolio(gamma=gamma)

    # Backtest portfolio
    print("Backtesting portfolio...")
    weights_df, returns_series = portfolio.backtest_portfolio(
        characteristics_history,
        returns_history,
        rebalance_freq=rebalance_freq,
        lookback_periods=lookback_periods
    )

    # Calculate performance metrics
    metrics = portfolio.calculate_performance_metrics(returns_series, risk_free_rate)

    # No need to close Bloomberg connection since we're using simulated data

    return weights_df, returns_series, metrics


def plot_results(returns_series, metrics, title):
    """
    Plot the results of the strategy

    Parameters:
    -----------
    returns_series : pandas.Series
        Series with portfolio returns
    metrics : dict
        Dictionary with performance metrics
    title : str
        Title for the plot
    """
    # Create figure with two subplots
    _, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Plot cumulative returns
    cum_returns = (1 + returns_series).cumprod()
    ax1.plot(cum_returns.index, cum_returns, 'b-')
    ax1.set_title(f'{title} - Cumulative Returns')
    ax1.set_ylabel('Cumulative Return')
    ax1.grid(True)

    # Plot drawdowns
    running_max = cum_returns.cummax()
    drawdowns = (cum_returns / running_max) - 1
    ax2.fill_between(drawdowns.index, drawdowns, 0, color='r', alpha=0.3)
    ax2.set_title(f'{title} - Drawdowns')
    ax2.set_ylabel('Drawdown')
    ax2.grid(True)

    # Add performance metrics as text
    metrics_text = (
        f"Annualized Return: {metrics['mean_return']:.2%}\n"
        f"Annualized Volatility: {metrics['volatility']:.2%}\n"
        f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}\n"
        f"Maximum Drawdown: {metrics['max_drawdown']:.2%}\n"
        f"Win Rate: {metrics['win_rate']:.2%}\n"
        f"t-statistic: {metrics['t_statistic']:.2f}"
    )
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax1.text(0.05, 0.05, metrics_text, transform=ax1.transAxes, fontsize=10,
             verticalalignment='bottom', bbox=props)

    plt.tight_layout()
    plt.savefig(f"{title.replace(' ', '_').lower()}.png")
    plt.show()


def main():
    """Main function to run the strategy"""
    # Set parameters
    start_date = '2010-01-01'
    end_date = '2020-12-31'
    gamma = 2.0
    num_stocks = 50  # Number of stocks in the simulated universe

    print(f"Running optimal characteristic portfolios strategy with {num_stocks} stocks...")
    print(f"Time period: {start_date} to {end_date}")
    print(f"Risk aversion parameter (gamma): {gamma}")

    # Run strategy for size characteristic
    _, returns_size, metrics_size = run_strategy(
        start_date, end_date,
        characteristic_type='size',
        rebalance_freq='monthly',
        lookback_periods=12,
        gamma=gamma,
        num_stocks=num_stocks
    )

    # Print performance metrics for size
    print("\nSize Portfolio Performance Metrics:")
    for key, value in metrics_size.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")

    # Plot results for size
    plot_results(returns_size, metrics_size, "Size Characteristic Portfolio")

    # Run strategy for value characteristic
    _, returns_value, metrics_value = run_strategy(
        start_date, end_date,
        characteristic_type='value',
        rebalance_freq='monthly',
        lookback_periods=12,
        gamma=gamma,
        num_stocks=num_stocks
    )

    # Print performance metrics for value
    print("\nValue Portfolio Performance Metrics:")
    for key, value in metrics_value.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")

    # Plot results for value
    plot_results(returns_value, metrics_value, "Value Characteristic Portfolio")

    # Run strategy for momentum characteristic
    _, returns_momentum, metrics_momentum = run_strategy(
        start_date, end_date,
        characteristic_type='momentum',
        rebalance_freq='monthly',
        lookback_periods=12,
        gamma=gamma,
        num_stocks=num_stocks
    )

    # Print performance metrics for momentum
    print("\nMomentum Portfolio Performance Metrics:")
    for key, value in metrics_momentum.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")

    # Plot results for momentum
    plot_results(returns_momentum, metrics_momentum, "Momentum Characteristic Portfolio")

    # Compare strategies
    compare_strategies(
        returns_size, returns_value, returns_momentum,
        metrics_size, metrics_value, metrics_momentum
    )


def compare_strategies(returns_size, returns_value, returns_momentum,
                      metrics_size, metrics_value, metrics_momentum):
    """
    Compare the performance of different characteristic portfolios

    Parameters:
    -----------
    returns_size, returns_value, returns_momentum : pandas.Series
        Series with portfolio returns for each strategy
    metrics_size, metrics_value, metrics_momentum : dict
        Dictionaries with performance metrics for each strategy
    """
    # Align return series
    common_dates = returns_size.index.intersection(
        returns_value.index.intersection(returns_momentum.index)
    )

    size_returns = returns_size.loc[common_dates]
    value_returns = returns_value.loc[common_dates]
    momentum_returns = returns_momentum.loc[common_dates]

    # Calculate cumulative returns
    cum_size = (1 + size_returns).cumprod()
    cum_value = (1 + value_returns).cumprod()
    cum_momentum = (1 + momentum_returns).cumprod()

    # Plot comparison
    plt.figure(figsize=(12, 8))
    plt.plot(cum_size.index, cum_size, 'b-', label='Size')
    plt.plot(cum_value.index, cum_value, 'g-', label='Value')
    plt.plot(cum_momentum.index, cum_momentum, 'r-', label='Momentum')
    plt.title('Comparison of Characteristic Portfolios')
    plt.ylabel('Cumulative Return')
    plt.grid(True)
    plt.legend()

    # Add performance metrics as text
    metrics_text = (
        f"Sharpe Ratios:\n"
        f"Size: {metrics_size['sharpe_ratio']:.2f}\n"
        f"Value: {metrics_value['sharpe_ratio']:.2f}\n"
        f"Momentum: {metrics_momentum['sharpe_ratio']:.2f}\n\n"
        f"t-statistics:\n"
        f"Size: {metrics_size['t_statistic']:.2f}\n"
        f"Value: {metrics_value['t_statistic']:.2f}\n"
        f"Momentum: {metrics_momentum['t_statistic']:.2f}"
    )
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.text(0.05, 0.05, metrics_text, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='bottom', bbox=props)

    plt.tight_layout()
    plt.savefig("characteristic_portfolios_comparison.png")
    plt.show()

    # Create correlation matrix
    returns_df = pd.DataFrame({
        'Size': size_returns,
        'Value': value_returns,
        'Momentum': momentum_returns
    })
    corr_matrix = returns_df.corr()

    print("\nCorrelation Matrix:")
    print(corr_matrix)


if __name__ == "__main__":
    main()
