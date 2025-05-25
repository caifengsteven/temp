import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from scipy import integrate
import warnings
warnings.filterwarnings('ignore')

# Function to simulate stock returns
def simulate_stock_data(start_date, end_date, num_stocks=30, annual_mean_ret=0.10, annual_vol=0.20, corr=0.3):
    """
    Simulate daily stock returns for a set of stocks using geometric Brownian motion.
    Parameters:
    - start_date, end_date: Date range for simulation
    - num_stocks: Number of stocks to simulate (default 30 for DJIA)
    - annual_mean_ret: Annualized mean return (default 10%)
    - annual_vol: Annualized volatility (default 20%)
    - corr: Correlation between stocks (default 0.3 to mimic market correlation)
    """
    # Generate date range
    dates = pd.date_range(start=start_date, end=end_date, freq='B')  # Business days
    n_days = len(dates)
    
    # Simulate uncorrelated Brownian motion
    daily_mean_ret = annual_mean_ret / 252
    daily_vol = annual_vol / np.sqrt(252)
    uncorrelated_noise = np.random.normal(loc=daily_mean_ret, scale=daily_vol, size=(n_days, num_stocks))
    
    # Introduce correlation using Cholesky decomposition
    corr_matrix = np.full((num_stocks, num_stocks), corr)
    np.fill_diagonal(corr_matrix, 1.0)
    chol = np.linalg.cholesky(corr_matrix)
    correlated_noise = uncorrelated_noise @ chol
    
    # Convert to returns
    returns = pd.DataFrame(correlated_noise, index=dates, columns=[f'Stock_{i+1}' for i in range(num_stocks)])
    
    # Convert returns to prices (starting at 100 for each stock)
    prices = (1 + returns).cumprod() * 100
    return prices, returns

# Calculate Empirical Distribution Function (EDF)
def calculate_edf(returns):
    sorted_returns = np.sort(returns.dropna())
    n = len(sorted_returns)
    edf = np.arange(1, n + 1) / n
    return sorted_returns, edf

# Calculate CUAR (Cumulative Utility Area Ratio) for a pair of assets
def calculate_cuar(asset1_returns, asset2_returns, utility_type="DARA"):
    """
    Calculate CUAR as per paper (Section 2.1).
    Uses empirical distribution functions and utility function to compute incremental utility areas.
    """
    asset1_returns = asset1_returns.dropna()
    asset2_returns = asset2_returns.dropna()
    if len(asset1_returns) == 0 or len(asset2_returns) == 0:
        return 1.0  # Default to neutral if no data
    x1, F1 = calculate_edf(asset1_returns)
    x2, F2 = calculate_edf(asset2_returns)
    outcomes = np.unique(np.concatenate([x1, x2]))
    outcomes = np.sort(outcomes)
    F1_interp = np.interp(outcomes, x1, F1, left=0, right=1)
    F2_interp = np.interp(outcomes, x2, F2, left=0, right=1)
    if utility_type == "DARA":
        utility_deriv = lambda r: 1 / (1 + r) if r > -1 else 0
    elif utility_type == "CARA":
        utility_deriv = lambda r: np.exp(-r)
    else:  # Quadratic
        utility_deriv = lambda r: 1 - 2 * 0.5 * r
    diff_F1_over_F2 = np.maximum(F1_interp - F2_interp, 0)
    diff_F2_over_F1 = np.maximum(F2_interp - F1_interp, 0)
    u_deriv = np.array([utility_deriv(r) for r in outcomes])
    delta_F = integrate.trapz(diff_F1_over_F2 * u_deriv, outcomes)
    delta_G = integrate.trapz(diff_F2_over_F1 * u_deriv, outcomes)
    if delta_G == 0:
        return np.inf if delta_F > 0 else 0
    return delta_F / delta_G

# Construct AHP pairwise comparison matrix and derive weights
def ahp_weights(returns_df, utility_type="DARA"):
    """
    Construct pairwise comparison matrix using CUAR and apply AHP to get weights.
    """
    n = returns_df.shape[1]
    tickers = returns_df.columns
    comparison_matrix = np.ones((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            cuar = calculate_cuar(returns_df.iloc[:, i], returns_df.iloc[:, j], utility_type)
            comparison_matrix[i, j] = cuar
            comparison_matrix[j, i] = 1 / cuar if cuar != 0 else np.inf
    for i in range(n):
        if np.any(comparison_matrix[i, :] == np.inf) or np.any(comparison_matrix[:, i] == 0):
            comparison_matrix[i, :] = 0
            comparison_matrix[:, i] = np.inf
    eigenvalues, eigenvectors = np.linalg.eigh(comparison_matrix)
    max_eigen_idx = np.argmax(eigenvalues)
    weights = np.abs(eigenvectors[:, max_eigen_idx])
    weights = weights / weights.sum() if weights.sum() > 0 else np.ones(n) / n
    CI = (eigenvalues[max_eigen_idx] - n) / (n - 1) if n > 1 else 0
    RI = {1: 0, 2: 0, 3: 0.58, 4: 0.9, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49}.get(n, 1.5)
    CR = CI / RI if RI > 0 else 0
    if CR > 0.1:
        print(f"Warning: Consistency Ratio {CR:.3f} > 0.1, matrix may not be consistent.")
    return pd.Series(weights, index=tickers)

# Backtest strategy with monthly rebalancing
def backtest_strategy(data, rebalance_freq="M", utility_type="DARA"):
    """
    Backtest UETT strategy with full CUAR and AHP implementation.
    """
    returns = data.pct_change().dropna()
    portfolio_weights = pd.DataFrame(index=returns.index, columns=returns.columns)
    portfolio_returns = []
    benchmark_returns = returns.mean(axis=1)  # Equal-weight DJIA proxy for simplicity
    start_idx = returns.index[0]
    calculation_period = timedelta(days=365)
    for end_idx in returns.resample(rebalance_freq).last().index:
        if (end_idx - start_idx).days < calculation_period.days:
            continue
        window_start = end_idx - calculation_period
        window_data = returns.loc[window_start:end_idx]
        if len(window_data) > 0:
            weights = ahp_weights(window_data, utility_type)
            next_period_end = end_idx + timedelta(days=30)
            next_period = returns.loc[end_idx:next_period_end].index
            if len(next_period) > 0:
                portfolio_weights.loc[next_period[0]:next_period_end] = weights
        start_idx = end_idx
    portfolio_weights = portfolio_weights.fillna(method='ffill').dropna(how='all')
    portfolio_returns = (returns * portfolio_weights.shift(1)).sum(axis=1).dropna()
    return portfolio_returns, benchmark_returns.loc[portfolio_returns.index]

# Main execution
if __name__ == "__main__":
    # Define date range (same as paper for consistency)
    start_date = datetime(2010, 9, 30)
    end_date = datetime(2015, 9, 30)
    
    # Simulate data for 30 DJIA stocks
    print("Generating simulated data...")
    prices, returns = simulate_stock_data(start_date, end_date, num_stocks=30, annual_mean_ret=0.10, annual_vol=0.20, corr=0.3)
    
    # Run backtest
    print("Running backtest with full CUAR and AHP...")
    portfolio_ret, benchmark_ret = backtest_strategy(prices, rebalance_freq="M", utility_type="DARA")
    
    # Calculate cumulative returns
    portfolio_cum_ret = (1 + portfolio_ret).cumprod()
    benchmark_cum_ret = (1 + benchmark_ret).cumprod()
    
    # Performance metrics
    portfolio_annual_ret = portfolio_ret.mean() * 252 * 100
    benchmark_annual_ret = benchmark_ret.mean() * 252 * 100
    portfolio_vol = portfolio_ret.std() * np.sqrt(252) * 100
    benchmark_vol = benchmark_ret.std() * np.sqrt(252) * 100
    portfolio_sharpe = portfolio_annual_ret / portfolio_vol if portfolio_vol > 0 else 0
    benchmark_sharpe = benchmark_annual_ret / benchmark_vol if benchmark_vol > 0 else 0
    tracking_error = ((portfolio_ret - benchmark_ret).std() * np.sqrt(252)) * 100
    correlation = portfolio_ret.corr(benchmark_ret)
    # Approximate CUAR for portfolio vs benchmark (simplified)
    portfolio_cuar = calculate_cuar(portfolio_ret, benchmark_ret, "DARA")
    
    print(f"Portfolio Annual Return: {portfolio_annual_ret:.2f}%")
    print(f"Benchmark Annual Return: {benchmark_annual_ret:.2f}%")
    print(f"Portfolio Volatility: {portfolio_vol:.2f}%")
    print(f"Benchmark Volatility: {benchmark_vol:.2f}%")
    print(f"Portfolio Sharpe Ratio: {portfolio_sharpe:.2f}")
    print(f"Benchmark Sharpe Ratio: {benchmark_sharpe:.2f}")
    print(f"Tracking Error: {tracking_error:.2f}%")
    print(f"Correlation with Benchmark: {correlation:.2f}")
    print(f"Portfolio CUAR vs Benchmark: {portfolio_cuar:.2f}")
    
    # Plot results
    plt.figure(figsize=(12, 6))
    plt.plot(portfolio_cum_ret, label="Enhanced Portfolio (UETT)")
    plt.plot(benchmark_cum_ret, label="DJIA Benchmark (Equal-Weighted Proxy)")
    plt.title("Cumulative Returns: UETT vs DJIA Benchmark (Simulated Data)")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.legend()
    plt.grid()
    plt.show()