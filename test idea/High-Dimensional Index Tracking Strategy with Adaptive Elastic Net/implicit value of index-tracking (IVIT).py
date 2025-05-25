import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

class IVITTradingStrategy:
    """
    Trading strategy based on Implicit Value of Index Tracking (IVIT).
    
    The strategy dynamically switches between:
    - Mean-Variance Efficient Portfolio (MVEP) when estimation error is low
    - Mean-Enhanced Tracking Efficient Portfolio (MTEP) when estimation error is high
    """
    
    def __init__(self, 
                 lookback_window: int = 120,
                 rebalance_frequency: int = 20,
                 target_return: float = 0.01,
                 risk_aversion: float = 3.0,
                 ivit_threshold: float = 0.0):
        """
        Initialize the trading strategy.
        
        Parameters:
        -----------
        lookback_window : int
            Number of periods to use for parameter estimation
        rebalance_frequency : int
            Number of periods between rebalancing
        target_return : float
            Monthly target return
        risk_aversion : float
            Risk aversion parameter
        ivit_threshold : float
            Threshold for switching between MVEP and MTEP
        """
        self.lookback_window = lookback_window
        self.rebalance_frequency = rebalance_frequency
        self.target_return = target_return
        self.risk_aversion = risk_aversion
        self.ivit_threshold = ivit_threshold
        
        # Storage for results
        self.portfolio_weights = []
        self.portfolio_returns = []
        self.ivit_values = []
        self.strategy_choices = []
        self.performance_metrics = {}
        
    def estimate_parameters(self, returns: pd.DataFrame, benchmark_returns: pd.Series):
        """Estimate mean returns and covariance matrix."""
        mu = returns.mean().values
        Sigma = returns.cov().values
        
        # Benchmark parameters
        sigma_b_squared = benchmark_returns.var()
        c = np.array([returns.iloc[:, i].cov(benchmark_returns) 
                     for i in range(returns.shape[1])])
        beta = c / sigma_b_squared
        
        return mu, Sigma, c, beta, sigma_b_squared
    
    def compute_estimation_error_proxy(self, returns: pd.DataFrame) -> dict:
        """
        Compute proxies for estimation error level.
        
        Returns:
        --------
        dict
            Various estimation error metrics
        """
        T, d = returns.shape
        
        # Compute rolling window statistics
        rolling_means = returns.rolling(window=20).mean()
        rolling_stds = returns.rolling(window=20).std()
        
        # Estimation error proxies
        error_metrics = {
            'sample_ratio': d / T,  # Higher d/T means more estimation error
            'mean_instability': rolling_means.std().mean(),  # Instability in mean estimates
            'vol_instability': rolling_stds.std().mean(),  # Instability in volatility
            'cross_sectional_dispersion': returns.std(axis=1).mean(),  # Market dispersion
            'recent_volatility': returns.iloc[-20:].std().mean()  # Recent market volatility
        }
        
        # Composite estimation error score (0-1 scale)
        error_metrics['composite_score'] = (
            0.3 * min(error_metrics['sample_ratio'] * 10, 1) +
            0.2 * min(error_metrics['mean_instability'] / 0.01, 1) +
            0.2 * min(error_metrics['vol_instability'] / 0.01, 1) +
            0.15 * min(error_metrics['cross_sectional_dispersion'] / 0.05, 1) +
            0.15 * min(error_metrics['recent_volatility'] / 0.05, 1)
        )
        
        return error_metrics
    
    def compute_adaptive_ivit(self, returns: pd.DataFrame, benchmark_returns: pd.Series,
                            error_metrics: dict) -> float:
        """
        Compute an adaptive IVIT measure that considers current market conditions.
        """
        # Get parameters
        mu, Sigma, c, beta, sigma_b_squared = self.estimate_parameters(returns, benchmark_returns)
        
        T, d = returns.shape
        ones = np.ones(d)
        
        try:
            Sigma_inv = np.linalg.inv(Sigma)
        except:
            # Add small regularization if singular
            Sigma_inv = np.linalg.inv(Sigma + 1e-6 * np.eye(d))
        
        # Compute portfolio components
        alpha_0 = Sigma_inv @ ones / (ones.T @ Sigma_inv @ ones)
        eta_0 = mu.T @ alpha_0
        
        B = Sigma_inv - np.outer(alpha_0, ones.T @ Sigma_inv)
        alpha_1 = B @ mu
        eta_1 = mu.T @ alpha_1
        alpha_2 = B @ c
        eta_2 = mu.T @ alpha_2
        
        # Base IVIT calculation
        if eta_1 <= 0 or self.target_return <= eta_0:
            return -1.0  # Invalid case, prefer MVEP
        
        kappa_m = eta_1 / (self.target_return - eta_0)
        
        # Simplified IVIT approximation considering estimation error
        base_ivit = eta_2 * (1 - 2 * T / (T - d - 1))
        
        # Adjust IVIT based on market conditions
        volatility_adjustment = -0.5 * error_metrics['recent_volatility'] / 0.05
        dispersion_bonus = 0.3 * error_metrics['cross_sectional_dispersion'] / 0.05
        sample_ratio_bonus = 0.5 * error_metrics['sample_ratio']
        
        adaptive_ivit = base_ivit + volatility_adjustment + dispersion_bonus + sample_ratio_bonus
        
        return adaptive_ivit
    
    def select_portfolio(self, returns: pd.DataFrame, benchmark_returns: pd.Series) -> tuple:
        """
        Select between MVEP and MTEP based on IVIT and market conditions.
        
        Returns:
        --------
        tuple
            (weights, strategy_used, ivit_value)
        """
        # Compute estimation error metrics
        error_metrics = self.compute_estimation_error_proxy(returns)
        
        # Compute adaptive IVIT
        ivit = self.compute_adaptive_ivit(returns, benchmark_returns, error_metrics)
        
        # Get parameters
        mu, Sigma, c, beta, sigma_b_squared = self.estimate_parameters(returns, benchmark_returns)
        
        # Decision rule with multiple factors
        use_mtep = (
            (ivit > self.ivit_threshold) or 
            (error_metrics['composite_score'] > 0.7) or
            (error_metrics['sample_ratio'] > 0.3 and error_metrics['recent_volatility'] > 0.03)
        )
        
        if use_mtep:
            weights = self._compute_mtep_weights(mu, Sigma, c)
            strategy = 'MTEP'
        else:
            weights = self._compute_mvep_weights(mu, Sigma)
            strategy = 'MVEP'
        
        return weights, strategy, ivit
    
    def _compute_mvep_weights(self, mu: np.ndarray, Sigma: np.ndarray) -> np.ndarray:
        """Compute MVEP weights with constraints."""
        d = len(mu)
        
        # Objective: maximize mu'x - (kappa/2) x'Sigma x
        def objective(x):
            return -(mu @ x - self.risk_aversion/2 * x @ Sigma @ x)
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # Sum to 1
        ]
        
        # Bounds (allow short selling but limit to reasonable range)
        bounds = [(-0.5, 1.5) for _ in range(d)]
        
        # Initial guess
        x0 = np.ones(d) / d
        
        # Optimize
        result = minimize(objective, x0, method='SLSQP', 
                         bounds=bounds, constraints=constraints)
        
        return result.x
    
    def _compute_mtep_weights(self, mu: np.ndarray, Sigma: np.ndarray, 
                             c: np.ndarray) -> np.ndarray:
        """Compute MTEP weights with tracking error constraint."""
        d = len(mu)
        
        # Objective: maximize mu'x - (kappa/2) * tracking_error_variance
        def objective(x):
            tracking_var = x @ Sigma @ x - 2 * c @ x
            return -(mu @ x - self.risk_aversion/2 * tracking_var)
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # Sum to 1
        ]
        
        # Bounds
        bounds = [(-0.5, 1.5) for _ in range(d)]
        
        # Initial guess
        x0 = np.ones(d) / d
        
        # Optimize
        result = minimize(objective, x0, method='SLSQP', 
                         bounds=bounds, constraints=constraints)
        
        return result.x
    
    def backtest(self, prices: pd.DataFrame, benchmark_prices: pd.Series,
                 start_date: str = None, end_date: str = None) -> dict:
        """
        Backtest the trading strategy.
        
        Parameters:
        -----------
        prices : pd.DataFrame
            Asset prices (columns are assets, index is dates)
        benchmark_prices : pd.Series
            Benchmark index prices
        start_date : str
            Start date for backtest
        end_date : str
            End date for backtest
            
        Returns:
        --------
        dict
            Backtest results and performance metrics
        """
        # Calculate returns
        returns = prices.pct_change().dropna()
        benchmark_returns = benchmark_prices.pct_change().dropna()
        
        # Align dates
        common_dates = returns.index.intersection(benchmark_returns.index)
        returns = returns.loc[common_dates]
        benchmark_returns = benchmark_returns.loc[common_dates]
        
        # Filter by date range if specified
        if start_date:
            returns = returns[returns.index >= start_date]
            benchmark_returns = benchmark_returns[benchmark_returns.index >= start_date]
        if end_date:
            returns = returns[returns.index <= end_date]
            benchmark_returns = benchmark_returns[benchmark_returns.index <= end_date]
        
        # Initialize portfolio
        portfolio_value = 1.0
        portfolio_values = []
        dates = []
        
        # Backtest loop
        for i in range(self.lookback_window, len(returns), self.rebalance_frequency):
            # Get historical data
            hist_returns = returns.iloc[i-self.lookback_window:i]
            hist_benchmark = benchmark_returns.iloc[i-self.lookback_window:i]
            
            # Select portfolio
            weights, strategy, ivit = self.select_portfolio(hist_returns, hist_benchmark)
            
            # Store decision
            self.portfolio_weights.append(weights)
            self.strategy_choices.append(strategy)
            self.ivit_values.append(ivit)
            
            # Calculate returns for next period
            if i + self.rebalance_frequency < len(returns):
                period_returns = returns.iloc[i:i+self.rebalance_frequency]
                portfolio_period_returns = (period_returns @ weights).values
                
                # Update portfolio value
                for j, ret in enumerate(portfolio_period_returns):
                    portfolio_value *= (1 + ret)
                    portfolio_values.append(portfolio_value)
                    dates.append(period_returns.index[j])
        
        # Create results DataFrame
        if len(portfolio_values) > 0:
            results_df = pd.DataFrame({
                'portfolio_value': portfolio_values,
                'date': dates
            })
            results_df.set_index('date', inplace=True)
            
            # Calculate performance metrics
            self._calculate_performance_metrics(results_df, benchmark_returns)
        else:
            results_df = pd.DataFrame()
            self.performance_metrics = {}
        
        return {
            'portfolio_values': results_df,
            'weights_history': self.portfolio_weights,
            'strategy_history': self.strategy_choices,
            'ivit_history': self.ivit_values,
            'performance_metrics': self.performance_metrics
        }
    
    def _calculate_performance_metrics(self, portfolio_values: pd.DataFrame, 
                                     benchmark_returns: pd.Series):
        """Calculate various performance metrics."""
        if len(portfolio_values) < 2:
            self.performance_metrics = {
                'total_return': 0,
                'annual_return': 0,
                'annual_volatility': 0,
                'sharpe_ratio': 0,
                'sortino_ratio': 0,
                'max_drawdown': 0,
                'tracking_error': 0,
                'information_ratio': 0,
                'mtep_usage_pct': 0,
                'avg_ivit': 0
            }
            return
            
        # Portfolio returns
        portfolio_returns = portfolio_values['portfolio_value'].pct_change().dropna()
        
        # Align benchmark returns
        benchmark_returns = benchmark_returns.loc[portfolio_returns.index]
        
        # Basic metrics
        total_return = (portfolio_values['portfolio_value'].iloc[-1] / 
                       portfolio_values['portfolio_value'].iloc[0] - 1)
        annual_return = (1 + total_return) ** (252 / len(portfolio_returns)) - 1
        annual_vol = portfolio_returns.std() * np.sqrt(252)
        sharpe_ratio = annual_return / annual_vol if annual_vol > 0 else 0
        
        # Relative metrics
        excess_returns = portfolio_returns - benchmark_returns
        tracking_error = excess_returns.std() * np.sqrt(252)
        information_ratio = (excess_returns.mean() * 252) / tracking_error if tracking_error > 0 else 0
        
        # Downside metrics
        downside_returns = portfolio_returns[portfolio_returns < 0]
        downside_vol = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
        sortino_ratio = annual_return / downside_vol if downside_vol > 0 else 0
        
        # Maximum drawdown
        cum_returns = (1 + portfolio_returns).cumprod()
        running_max = cum_returns.expanding().max()
        drawdown = (cum_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Strategy usage statistics
        mtep_usage = sum(1 for s in self.strategy_choices if s == 'MTEP') / len(self.strategy_choices) if len(self.strategy_choices) > 0 else 0
        
        self.performance_metrics = {
            'total_return': total_return * 100,
            'annual_return': annual_return * 100,
            'annual_volatility': annual_vol * 100,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown * 100,
            'tracking_error': tracking_error * 100,
            'information_ratio': information_ratio,
            'mtep_usage_pct': mtep_usage * 100,
            'avg_ivit': np.mean(self.ivit_values) if len(self.ivit_values) > 0 else 0
        }


def simulate_market_data(n_assets: int = 10, n_periods: int = 2520, seed: int = 42):
    """
    Simulate realistic market data with varying market regimes.
    
    Parameters:
    -----------
    n_assets : int
        Number of assets to simulate
    n_periods : int
        Number of trading days (2520 = ~10 years)
    seed : int
        Random seed for reproducibility
        
    Returns:
    --------
    tuple
        (prices_df, benchmark_prices_series)
    """
    np.random.seed(seed)
    
    # Create date index
    dates = pd.date_range(start='2014-01-01', periods=n_periods, freq='B')
    
    # Market regimes (bull, bear, high vol, low vol)
    regime_lengths = [500, 300, 400, 320, 500, 500]
    regimes = []
    for i, length in enumerate(regime_lengths):
        if i % 2 == 0:  # Bull market
            market_mean = 0.0005
            market_vol = 0.01
        else:  # Bear/volatile market
            market_mean = -0.0002
            market_vol = 0.02
        regimes.extend([(market_mean, market_vol)] * length)
    
    # Ensure we have enough regime data
    while len(regimes) < n_periods:
        regimes.extend(regimes[:n_periods - len(regimes)])
    regimes = regimes[:n_periods]
    
    # Generate market returns with regime switching
    market_returns = []
    for mean, vol in regimes:
        market_returns.append(np.random.normal(mean, vol))
    market_returns = np.array(market_returns)
    
    # Generate asset-specific parameters
    asset_names = [f'Asset_{i+1}' for i in range(n_assets)]
    
    # Asset characteristics (diverse betas and alphas)
    betas = np.random.uniform(0.5, 1.5, n_assets)
    alphas = np.random.normal(0, 0.0002, n_assets)
    idio_vols = np.random.uniform(0.005, 0.02, n_assets)
    
    # Generate asset returns
    asset_returns = np.zeros((n_periods, n_assets))
    for i in range(n_assets):
        # Single factor model with time-varying volatility
        idio_returns = np.random.normal(0, idio_vols[i], n_periods)
        asset_returns[:, i] = alphas[i] + betas[i] * market_returns + idio_returns
    
    # Convert to prices
    asset_prices = np.zeros((n_periods, n_assets))
    benchmark_prices = np.zeros(n_periods)
    
    # Initial prices
    asset_prices[0, :] = 100
    benchmark_prices[0] = 1000
    
    # Compute prices from returns
    for t in range(1, n_periods):
        asset_prices[t, :] = asset_prices[t-1, :] * (1 + asset_returns[t, :])
        benchmark_prices[t] = benchmark_prices[t-1] * (1 + market_returns[t])
    
    # Create DataFrames
    prices_df = pd.DataFrame(asset_prices, index=dates, columns=asset_names)
    benchmark_series = pd.Series(benchmark_prices, index=dates, name='Benchmark')
    
    return prices_df, benchmark_series


def plot_backtest_results(results: dict, benchmark_prices: pd.Series):
    """Plot comprehensive backtest results."""
    if len(results['portfolio_values']) == 0:
        print("No results to plot.")
        return
        
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    
    # Portfolio value over time
    ax = axes[0, 0]
    portfolio_df = results['portfolio_values']
    ax.plot(portfolio_df.index, portfolio_df['portfolio_value'], 
            label='IVIT Strategy', linewidth=2)
    
    # Normalize benchmark to start at 1
    benchmark_norm = benchmark_prices / benchmark_prices.iloc[0]
    benchmark_norm = benchmark_norm.loc[portfolio_df.index[0]:]
    ax.plot(benchmark_norm.index, benchmark_norm.values, 
            label='Benchmark', linewidth=2, alpha=0.7)
    
    ax.set_ylabel('Portfolio Value')
    ax.set_title('Portfolio Performance')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Strategy usage over time
    ax = axes[0, 1]
    if len(results['strategy_history']) > 0:
        strategy_dates = portfolio_df.index[::len(portfolio_df)//len(results['strategy_history'])][:len(results['strategy_history'])]
        strategy_df = pd.DataFrame({
            'strategy': results['strategy_history'],
            'date': strategy_dates
        })
        
        # Calculate quarterly MTEP usage
        strategy_df['is_mtep'] = (strategy_df['strategy'] == 'MTEP').astype(int)
        quarterly = strategy_df.set_index('date').resample('Q')['is_mtep'].mean() * 100
        
        ax.bar(quarterly.index, quarterly.values, width=80, alpha=0.7)
        ax.set_ylabel('MTEP Usage %')
        ax.set_title('Quarterly MTEP Usage Percentage')
        ax.grid(True, alpha=0.3)
    
    # IVIT values over time
    ax = axes[1, 0]
    if len(results['ivit_history']) > 0:
        ivit_dates = portfolio_df.index[::len(portfolio_df)//len(results['ivit_history'])][:len(results['ivit_history'])]
        ax.plot(ivit_dates, results['ivit_history'], linewidth=1.5)
        ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        ax.set_ylabel('IVIT Value')
        ax.set_title('Implicit Value of Index Tracking Over Time')
        ax.grid(True, alpha=0.3)
    
    # Rolling Sharpe ratio
    ax = axes[1, 1]
    portfolio_returns = portfolio_df['portfolio_value'].pct_change().dropna()
    if len(portfolio_returns) > 252:
        rolling_sharpe = portfolio_returns.rolling(252).mean() / portfolio_returns.rolling(252).std() * np.sqrt(252)
        ax.plot(rolling_sharpe.index, rolling_sharpe.values, linewidth=1.5)
        ax.set_ylabel('Sharpe Ratio')
        ax.set_title('Rolling 1-Year Sharpe Ratio')
        ax.grid(True, alpha=0.3)
    
    # Drawdown
    ax = axes[2, 0]
    cum_returns = (1 + portfolio_returns).cumprod()
    running_max = cum_returns.expanding().max()
    drawdown = (cum_returns - running_max) / running_max * 100
    ax.fill_between(drawdown.index, drawdown.values, 0, alpha=0.3, color='red')
    ax.set_ylabel('Drawdown %')
    ax.set_title('Portfolio Drawdown')
    ax.grid(True, alpha=0.3)
    
    # Performance metrics table
    ax = axes[2, 1]
    ax.axis('off')
    metrics = results['performance_metrics']
    table_data = [
        ['Total Return', f"{metrics['total_return']:.2f}%"],
        ['Annual Return', f"{metrics['annual_return']:.2f}%"],
        ['Annual Volatility', f"{metrics['annual_volatility']:.2f}%"],
        ['Sharpe Ratio', f"{metrics['sharpe_ratio']:.2f}"],
        ['Sortino Ratio', f"{metrics['sortino_ratio']:.2f}"],
        ['Max Drawdown', f"{metrics['max_drawdown']:.2f}%"],
        ['Information Ratio', f"{metrics['information_ratio']:.2f}"],
        ['MTEP Usage', f"{metrics['mtep_usage_pct']:.1f}%"],
        ['Avg IVIT', f"{metrics['avg_ivit']:.3f}"]
    ]
    
    table = ax.table(cellText=table_data, 
                    colLabels=['Metric', 'Value'],
                    cellLoc='left',
                    loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    ax.set_title('Performance Metrics')
    
    plt.tight_layout()
    plt.show()


# Main execution with simulated data
if __name__ == "__main__":
    print("Generating simulated market data...")
    
    # Simulate 10 assets over 10 years
    prices, benchmark_prices = simulate_market_data(n_assets=10, n_periods=2520, seed=42)
    
    print(f"Generated data for {len(prices.columns)} assets over {len(prices)} trading days")
    print(f"Date range: {prices.index[0]} to {prices.index[-1]}")
    
    # Initialize and run strategy
    print("\nRunning IVIT Trading Strategy backtest...")
    strategy = IVITTradingStrategy(
        lookback_window=120,  # 120 days for estimation
        rebalance_frequency=20,  # Rebalance monthly
        target_return=0.01,  # 1% monthly target
        risk_aversion=3.0,
        ivit_threshold=0.0
    )
    
    # Run backtest
    results = strategy.backtest(prices, benchmark_prices)
    
    # Print performance summary
    print("\nPerformance Summary:")
    print("-" * 40)
    for metric, value in results['performance_metrics'].items():
        if isinstance(value, float):
            print(f"{metric.replace('_', ' ').title()}: {value:.2f}")
        else:
            print(f"{metric.replace('_', ' ').title()}: {value}")
    
    # Plot results
    plot_backtest_results(results, benchmark_prices)
    
    # Additional analysis: Compare with pure MVEP and pure MTEP
    print("\nComparing with pure strategies...")
    
    # Pure MVEP strategy
    strategy_mvep = IVITTradingStrategy(
        lookback_window=120,
        rebalance_frequency=20,
        target_return=0.01,
        risk_aversion=3.0,
        ivit_threshold=float('inf')  # Always use MVEP
    )
    results_mvep = strategy_mvep.backtest(prices, benchmark_prices)
    
    # Pure MTEP strategy
    strategy_mtep = IVITTradingStrategy(
        lookback_window=120,
        rebalance_frequency=20,
        target_return=0.01,
        risk_aversion=3.0,
        ivit_threshold=float('-inf')  # Always use MTEP
    )
    results_mtep = strategy_mtep.backtest(prices, benchmark_prices)
    
    # Compare strategies
    print("\nStrategy Comparison:")
    print("-" * 60)
    print(f"{'Metric':<20} {'IVIT Strategy':>15} {'Pure MVEP':>15} {'Pure MTEP':>15}")
    print("-" * 60)
    
    metrics_to_compare = ['annual_return', 'annual_volatility', 'sharpe_ratio', 
                         'max_drawdown', 'information_ratio']
    
    for metric in metrics_to_compare:
        ivit_val = results['performance_metrics'][metric]
        mvep_val = results_mvep['performance_metrics'][metric]
        mtep_val = results_mtep['performance_metrics'][metric]
        
        print(f"{metric.replace('_', ' ').title():<20} "
              f"{ivit_val:>15.2f} {mvep_val:>15.2f} {mtep_val:>15.2f}")
    
    # Test sensitivity to different market conditions
    print("\n\nTesting strategy under different market conditions...")
    
    # High volatility market
    print("\nHigh Volatility Market:")
    prices_vol, benchmark_vol = simulate_market_data(n_assets=10, n_periods=2520, seed=123)
    # Amplify volatility
    returns_vol = prices_vol.pct_change()
    returns_vol = returns_vol * 1.5  # 50% more volatile
    prices_vol = (1 + returns_vol).cumprod() * 100
    
    strategy_vol = IVITTradingStrategy(
        lookback_window=120,
        rebalance_frequency=20,
        target_return=0.01,
        risk_aversion=3.0,
        ivit_threshold=0.0
    )
    results_vol = strategy_vol.backtest(prices_vol, benchmark_vol)
    
    print(f"Annual Return: {results_vol['performance_metrics']['annual_return']:.2f}%")
    print(f"Sharpe Ratio: {results_vol['performance_metrics']['sharpe_ratio']:.2f}")
    print(f"MTEP Usage: {results_vol['performance_metrics']['mtep_usage_pct']:.1f}%")