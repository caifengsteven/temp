import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
from datetime import datetime, timedelta

# Set the random seed for reproducibility
np.random.seed(42)

class PortfolioBanditThompsonSampling:
    """
    Implementation of 'Adaptive Portfolio by Solving Multi-armed Bandit via Thompson Sampling' 
    as described in the paper.
    """
    
    def __init__(self, n_assets, top_c=0.6, sliding_window=120):
        """
        Initialize the Portfolio Bandit via Thompson Sampling (PBTS) model.
        
        Parameters:
        -----------
        n_assets : int
            Number of assets in the portfolio
        top_c : float
            Parameter for risk preference (higher c means lower risk tolerance)
        sliding_window : int
            Number of periods to consider for Sharpe ratio calculation
        """
        self.n_assets = n_assets
        self.top_c = top_c
        self.sliding_window = sliding_window
        
        # Strategic arms
        self.arms = ["BH", "SA", "EW", "VW", "MV"]
        self.n_arms = len(self.arms)
        
        # Beta distribution parameters for Thompson sampling
        self.alpha = np.ones(self.n_arms)
        self.beta = np.ones(self.n_arms)
        
        # Initialize portfolio weights
        self.weights_history = []
        self.arm_history = []
        self.returns_history = []
        
        # Initialize empty portfolio (equally weighted at start)
        self.current_weights = np.ones(n_assets) / n_assets
        self.weights_history.append(self.current_weights)
        
    def update_weights(self, returns, market_data=None):
        """
        Update portfolio weights based on the selected arm.
        
        Parameters:
        -----------
        returns : numpy.ndarray
            Asset returns for the current period
        market_data : pandas.DataFrame or None
            Additional market data needed for certain strategies (e.g., MV)
        
        Returns:
        --------
        numpy.ndarray
            Updated portfolio weights
        """
        # Store returns for historical tracking
        self.returns_history.append(returns)
        
        # Calculate weights for each strategy (arm)
        arm_weights = self.calculate_arm_weights(returns, market_data)
        
        # Sample from Beta distributions
        samples = np.array([np.random.beta(self.alpha[i], self.beta[i]) for i in range(self.n_arms)])
        
        # Select arm with highest sampled value
        selected_arm_idx = np.argmax(samples)
        selected_arm = self.arms[selected_arm_idx]
        self.arm_history.append(selected_arm)
        
        # Update portfolio weights based on selected arm
        self.current_weights = arm_weights[selected_arm_idx]
        self.weights_history.append(self.current_weights)
        
        # If we have enough history, update the Beta distributions
        if len(self.returns_history) > self.sliding_window:
            self.update_beta_distribution(selected_arm_idx, arm_weights)
        
        return self.current_weights
    
    def calculate_arm_weights(self, returns, market_data=None):
        """
        Calculate weights for each strategic arm.
        
        Parameters:
        -----------
        returns : numpy.ndarray
            Asset returns for the current period
        market_data : pandas.DataFrame or None
            Additional market data for strategies requiring historical data
            
        Returns:
        --------
        list of numpy.ndarray
            Weights for each arm
        """
        arm_weights = []
        
        # 1. Buy and Hold (BH)
        # Continue with current weights, adjusted for price changes
        bh_weights = self.current_weights * (1 + returns)
        bh_weights = bh_weights / np.sum(bh_weights) if np.sum(bh_weights) > 0 else np.zeros_like(bh_weights)
        arm_weights.append(bh_weights)
        
        # 2. Sold All (SA)
        # All cash position
        sa_weights = np.zeros(self.n_assets)
        arm_weights.append(sa_weights)
        
        # 3. Equally-weighted portfolio (EW)
        # Equal weights for all assets
        ew_weights = np.ones(self.n_assets) / self.n_assets
        arm_weights.append(ew_weights)
        
        # 4. Value-weighted portfolio (VW)
        # Weights proportional to market capitalization
        # If we don't have market caps, we'll simulate using current weights and returns
        vw_weights = self.current_weights * (1 + returns)
        vw_weights = vw_weights / np.sum(vw_weights) if np.sum(vw_weights) > 0 else np.zeros_like(vw_weights)
        arm_weights.append(vw_weights)
        
        # 5. Mean-variance portfolio (MV)
        # For simplicity, we'll use basic mean-variance optimization
        # In a real implementation, this would use robust estimation methods
        if len(self.returns_history) > self.sliding_window:
            # Use historical returns to estimate mean and covariance
            historical_returns = np.array(self.returns_history[-self.sliding_window:])
            mean_returns = np.mean(historical_returns, axis=0)
            cov_matrix = np.cov(historical_returns, rowvar=False)
            
            # Simple mean-variance optimization
            # Maximize Sharpe ratio (return/volatility)
            mv_weights = self.optimize_mean_variance(mean_returns, cov_matrix)
        else:
            # Not enough history, use EW weights
            mv_weights = ew_weights.copy()
        
        arm_weights.append(mv_weights)
        
        return arm_weights
    
    def optimize_mean_variance(self, mean_returns, cov_matrix, risk_aversion=1.0):
        """
        Perform mean-variance optimization.
        
        Parameters:
        -----------
        mean_returns : numpy.ndarray
            Expected returns for each asset
        cov_matrix : numpy.ndarray
            Covariance matrix of returns
        risk_aversion : float
            Risk aversion parameter
            
        Returns:
        --------
        numpy.ndarray
            Optimal portfolio weights
        """
        try:
            # Basic mean-variance optimization
            # w = Σ^(-1) * μ / λ where λ is risk aversion
            inv_cov = np.linalg.inv(cov_matrix + 1e-6 * np.eye(len(cov_matrix)))
            weights = inv_cov.dot(mean_returns) / risk_aversion
            
            # Normalize weights to sum to 1
            weights = weights / np.sum(np.abs(weights))
            
            # Ensure no extreme weights
            weights = np.clip(weights, -0.5, 1.0)
            weights = weights / np.sum(np.abs(weights))
            
            return weights
        except np.linalg.LinAlgError:
            # If matrix inversion fails, return equal weights
            return np.ones(self.n_assets) / self.n_assets
    
    def update_beta_distribution(self, selected_arm_idx, arm_weights):
        """
        Update Beta distribution parameters based on reward function.
        
        Parameters:
        -----------
        selected_arm_idx : int
            Index of the selected arm
        arm_weights : list of numpy.ndarray
            Weights for each arm
        """
        # Calculate Sharpe ratios for each arm using historical data
        sharpe_ratios = []
        for i in range(self.n_arms):
            # Skip arms without meaningful weights
            if np.sum(np.abs(arm_weights[i])) < 1e-6:
                sharpe_ratios.append(-np.inf)
                continue
            
            # Calculate portfolio returns for this arm's weights
            historical_returns = np.array(self.returns_history[-self.sliding_window:])
            port_returns = np.array([np.sum(arm_weights[i] * ret) for ret in historical_returns])
            
            # Calculate Sharpe ratio
            mean_return = np.mean(port_returns)
            std_return = np.std(port_returns)
            sharpe = mean_return / std_return if std_return > 0 else 0
            sharpe_ratios.append(sharpe)
        
        # Determine success based on top_c parameter
        selected_sharpe = sharpe_ratios[selected_arm_idx]
        success_count = sum([1 for sr in sharpe_ratios if selected_sharpe >= sr])
        success_ratio = success_count / self.n_arms
        
        # Update Beta parameters based on reward
        if success_ratio >= self.top_c:
            # Success: update alpha
            self.alpha[selected_arm_idx] += 1
        else:
            # Failure: update beta
            self.beta[selected_arm_idx] += 1
    
    def calculate_portfolio_performance(self, returns):
        """
        Calculate portfolio performance metrics.
        
        Parameters:
        -----------
        returns : numpy.ndarray
            Asset returns matrix (time x assets)
            
        Returns:
        --------
        dict
            Dictionary with performance metrics
        """
        weights = np.array(self.weights_history[:-1])  # Exclude the last weight which hasn't been applied yet
        
        # Calculate portfolio returns
        port_returns = np.sum(weights * returns, axis=1)
        
        # Calculate cumulative wealth
        cumulative_wealth = np.cumprod(1 + port_returns)
        
        # Calculate Sharpe ratio
        mean_return = np.mean(port_returns)
        std_return = np.std(port_returns)
        sharpe_ratio = mean_return / std_return * np.sqrt(252)  # Annualized
        
        # Calculate maximum drawdown
        peak = np.maximum.accumulate(cumulative_wealth)
        drawdown = (peak - cumulative_wealth) / peak
        max_drawdown = np.max(drawdown) * 100  # In percent
        
        # Calculate volatility
        volatility = std_return * np.sqrt(252) * 100  # Annualized, in percent
        
        return {
            'Sharpe Ratio': sharpe_ratio,
            'Cumulative Wealth': cumulative_wealth[-1],
            'Maximum Drawdown (%)': max_drawdown,
            'Volatility (%)': volatility,
            'Returns': port_returns,
            'Cumulative Wealth Series': cumulative_wealth
        }


class CompetingPortfolios:
    """
    Implementation of competing portfolio strategies for comparison.
    """
    
    def __init__(self, n_assets, sliding_window=120):
        """
        Initialize competing portfolio strategies.
        
        Parameters:
        -----------
        n_assets : int
            Number of assets in the portfolio
        sliding_window : int
            Number of periods for historical calculations
        """
        self.n_assets = n_assets
        self.sliding_window = sliding_window
        
        # Initialize portfolio weights for each strategy
        self.ew_weights = np.ones(n_assets) / n_assets
        self.vw_weights = np.ones(n_assets) / n_assets
        self.mv_weights = np.ones(n_assets) / n_assets
        
        # Historical weights for each strategy
        self.ew_history = [self.ew_weights]
        self.vw_history = [self.vw_weights]
        self.mv_history = [self.mv_weights]
        
        # Historical returns for MV calculation
        self.returns_history = []
    
    def update_weights(self, returns, market_data=None):
        """
        Update weights for each competing strategy.
        
        Parameters:
        -----------
        returns : numpy.ndarray
            Asset returns for the current period
        market_data : pandas.DataFrame or None
            Additional market data (not used in this implementation)
            
        Returns:
        --------
        dict
            Dictionary with updated weights for each strategy
        """
        # Store returns for historical tracking
        self.returns_history.append(returns)
        
        # 1. Equally-weighted portfolio (EW)
        self.ew_weights = np.ones(self.n_assets) / self.n_assets
        self.ew_history.append(self.ew_weights)
        
        # 2. Value-weighted portfolio (VW)
        self.vw_weights = self.vw_weights * (1 + returns)
        self.vw_weights = self.vw_weights / np.sum(self.vw_weights) if np.sum(self.vw_weights) > 0 else np.ones(self.n_assets) / self.n_assets
        self.vw_history.append(self.vw_weights)
        
        # 3. Mean-variance portfolio (MV)
        if len(self.returns_history) > self.sliding_window:
            # Use historical returns to estimate mean and covariance
            historical_returns = np.array(self.returns_history[-self.sliding_window:])
            mean_returns = np.mean(historical_returns, axis=0)
            cov_matrix = np.cov(historical_returns, rowvar=False)
            
            # Simple mean-variance optimization
            try:
                inv_cov = np.linalg.inv(cov_matrix + 1e-6 * np.eye(len(cov_matrix)))
                self.mv_weights = inv_cov.dot(mean_returns)
                self.mv_weights = self.mv_weights / np.sum(np.abs(self.mv_weights))
                self.mv_weights = np.clip(self.mv_weights, -0.5, 1.0)
                self.mv_weights = self.mv_weights / np.sum(np.abs(self.mv_weights))
            except np.linalg.LinAlgError:
                # If matrix inversion fails, keep current weights
                pass
        
        self.mv_history.append(self.mv_weights)
        
        return {
            'EW': self.ew_weights,
            'VW': self.vw_weights,
            'MV': self.mv_weights
        }
    
    def calculate_portfolio_performance(self, returns):
        """
        Calculate performance metrics for each competing strategy.
        
        Parameters:
        -----------
        returns : numpy.ndarray
            Asset returns matrix (time x assets)
            
        Returns:
        --------
        dict
            Dictionary with performance metrics for each strategy
        """
        performance = {}
        
        # Calculate performance for each strategy
        for name, weights_history in [
            ('EW', self.ew_history[:-1]),
            ('VW', self.vw_history[:-1]),
            ('MV', self.mv_history[:-1])
        ]:
            weights = np.array(weights_history)
            
            # Calculate portfolio returns
            port_returns = np.sum(weights * returns, axis=1)
            
            # Calculate cumulative wealth
            cumulative_wealth = np.cumprod(1 + port_returns)
            
            # Calculate Sharpe ratio
            mean_return = np.mean(port_returns)
            std_return = np.std(port_returns)
            sharpe_ratio = mean_return / std_return * np.sqrt(252)  # Annualized
            
            # Calculate maximum drawdown
            peak = np.maximum.accumulate(cumulative_wealth)
            drawdown = (peak - cumulative_wealth) / peak
            max_drawdown = np.max(drawdown) * 100  # In percent
            
            # Calculate volatility
            volatility = std_return * np.sqrt(252) * 100  # Annualized, in percent
            
            performance[name] = {
                'Sharpe Ratio': sharpe_ratio,
                'Cumulative Wealth': cumulative_wealth[-1],
                'Maximum Drawdown (%)': max_drawdown,
                'Volatility (%)': volatility,
                'Returns': port_returns,
                'Cumulative Wealth Series': cumulative_wealth
            }
        
        return performance


def generate_simulated_data(n_assets=25, n_periods=1000, freq='M'):
    """
    Generate simulated financial data.
    
    Parameters:
    -----------
    n_assets : int
        Number of assets to simulate
    n_periods : int
        Number of time periods to simulate
    freq : str
        Frequency of data ('D' for daily, 'M' for monthly)
        
    Returns:
    --------
    pandas.DataFrame
        Simulated returns data
    """
    # Define parameters for data generation
    if freq == 'D':
        # Daily data
        mean_returns = np.random.normal(0.0002, 0.0005, n_assets)  # Mean daily returns around 0.02% (5% annually)
        vol_annual = np.random.uniform(0.15, 0.40, n_assets)  # Annual volatility between 15% and 40%
        vol_daily = vol_annual / np.sqrt(252)  # Convert to daily volatility
        
        periods_per_year = 252
        date_range = pd.date_range(start='2010-01-01', periods=n_periods, freq='B')
    else:
        # Monthly data
        mean_returns = np.random.normal(0.005, 0.002, n_assets)  # Mean monthly returns around 0.5% (6% annually)
        vol_annual = np.random.uniform(0.15, 0.40, n_assets)  # Annual volatility between 15% and 40%
        vol_monthly = vol_annual / np.sqrt(12)  # Convert to monthly volatility
        vol_daily = vol_monthly
        
        periods_per_year = 12
        date_range = pd.date_range(start='2010-01-01', periods=n_periods, freq='M')
    
    # Generate correlated returns
    # Create a correlation matrix with some sector structure
    n_sectors = 5
    sector_size = n_assets // n_sectors
    remainder = n_assets % n_sectors
    
    correlation = np.zeros((n_assets, n_assets))
    
    # Assign sector correlations
    start_idx = 0
    for i in range(n_sectors):
        size = sector_size + (1 if i < remainder else 0)
        end_idx = start_idx + size
        
        # Intra-sector correlation (high)
        intra_corr = np.random.uniform(0.5, 0.8)
        correlation[start_idx:end_idx, start_idx:end_idx] = intra_corr
        
        start_idx = end_idx
    
    # Set diagonal to 1
    np.fill_diagonal(correlation, 1.0)
    
    # Ensure correlation matrix is positive semi-definite
    min_eig = np.min(np.linalg.eigvals(correlation))
    if min_eig < 0:
        correlation -= 1.1 * min_eig * np.eye(n_assets)
    
    # Generate multivariate normal returns
    rng = np.random.default_rng(42)
    returns = rng.multivariate_normal(mean_returns, correlation * np.outer(vol_daily, vol_daily), n_periods)
    
    # Add some market events (crashes, rallies)
    market_events = []
    
    # Crash events
    for _ in range(2):
        event_start = rng.integers(0, n_periods - 20)
        event_length = rng.integers(5, 20)
        event_severity = rng.uniform(0.03, 0.08)  # Daily drop of 3-8%
        
        for i in range(event_length):
            decay_factor = 1 - i/event_length  # Effect diminishes over time
            returns[event_start + i, :] -= event_severity * decay_factor
        
        market_events.append((event_start, event_start + event_length, 'Crash'))
    
    # Rally events
    for _ in range(2):
        event_start = rng.integers(0, n_periods - 20)
        event_length = rng.integers(5, 20)
        event_severity = rng.uniform(0.02, 0.05)  # Daily gain of 2-5%
        
        for i in range(event_length):
            decay_factor = 1 - i/event_length  # Effect diminishes over time
            returns[event_start + i, :] += event_severity * decay_factor
        
        market_events.append((event_start, event_start + event_length, 'Rally'))
    
    # Create DataFrame with asset names
    asset_names = [f'Asset_{i+1}' for i in range(n_assets)]
    returns_df = pd.DataFrame(returns, index=date_range, columns=asset_names)
    
    print("Simulated Data Generated:")
    print(f"- Number of assets: {n_assets}")
    print(f"- Time periods: {n_periods} ({freq})")
    print(f"- Mean annual return: {np.mean(mean_returns) * periods_per_year * 100:.2f}%")
    print(f"- Mean annual volatility: {np.mean(vol_annual) * 100:.2f}%")
    print(f"- Market events: {len(market_events)}")
    for start, end, event_type in market_events:
        print(f"  * {event_type} from period {start} to {end}")
    
    return returns_df

def run_backtest(returns_df, top_c=0.6, sliding_window=120):
    """
    Run a backtest of the PBTS strategy against competing strategies.
    
    Parameters:
    -----------
    returns_df : pandas.DataFrame
        Asset returns data
    top_c : float
        Parameter for risk preference in PBTS
    sliding_window : int
        Number of periods for calculation window
    
    Returns:
    --------
    dict
        Dictionary with backtest results
    """
    n_assets = returns_df.shape[1]
    returns_array = returns_df.values
    
    # Initialize strategies
    pbts = PortfolioBanditThompsonSampling(n_assets, top_c, sliding_window)
    competing = CompetingPortfolios(n_assets, sliding_window)
    
    # Run backtest
    for i in range(len(returns_df)):
        returns = returns_array[i]
        
        # Update portfolio weights
        pbts.update_weights(returns)
        competing.update_weights(returns)
    
    # Calculate performance
    pbts_performance = pbts.calculate_portfolio_performance(returns_array)
    competing_performance = competing.calculate_portfolio_performance(returns_array)
    
    # Combine results
    results = {
        'PBTS': pbts_performance,
        **competing_performance
    }
    
    # Print summary
    print("\nBacktest Results Summary:")
    print("-" * 70)
    print(f"{'Strategy':<15} {'Sharpe Ratio':<15} {'Cumulative Wealth':<20} {'Max Drawdown (%)':<20} {'Volatility (%)':<15}")
    print("-" * 70)
    
    for strategy, metrics in results.items():
        print(f"{strategy:<15} {metrics['Sharpe Ratio']:<15.2f} {metrics['Cumulative Wealth']:<20.2f} {metrics['Maximum Drawdown (%)']:<20.2f} {metrics['Volatility (%)']:<15.2f}")
    
    return results, pbts.arm_history

def plot_results(results, arm_history=None):
    """
    Plot backtest results.
    
    Parameters:
    -----------
    results : dict
        Dictionary with backtest results
    arm_history : list or None
        List of selected arms for PBTS strategy
    """
    # Plot cumulative wealth
    plt.figure(figsize=(12, 6))
    
    for strategy, metrics in results.items():
        plt.plot(metrics['Cumulative Wealth Series'], label=strategy)
    
    plt.title('Cumulative Wealth Comparison')
    plt.xlabel('Time Period')
    plt.ylabel('Cumulative Wealth')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # Plot PBTS arm selection history if available
    if arm_history:
        plt.figure(figsize=(12, 6))
        
        # Convert arm names to numeric values
        arm_map = {arm: i for i, arm in enumerate(["BH", "SA", "EW", "VW", "MV"])}
        arm_numeric = [arm_map[arm] for arm in arm_history]
        
        plt.plot(arm_numeric, marker='o', linestyle='-', alpha=0.5)
        plt.yticks(list(arm_map.values()), list(arm_map.keys()))
        plt.title('PBTS Arm Selection History')
        plt.xlabel('Time Period')
        plt.ylabel('Selected Arm')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

def parameter_sensitivity_analysis(returns_df, top_c_values=None, sliding_window_values=None):
    """
    Analyze sensitivity of the PBTS strategy to parameter changes.
    
    Parameters:
    -----------
    returns_df : pandas.DataFrame
        Asset returns data
    top_c_values : list or None
        List of top_c values to test
    sliding_window_values : list or None
        List of sliding window values to test
    """
    if top_c_values is None:
        top_c_values = [0.3, 0.5, 0.7, 0.9]
    
    if sliding_window_values is None:
        sliding_window_values = [60, 120, 180]
    
    # Test different top_c values
    results_top_c = {}
    for top_c in top_c_values:
        print(f"\nTesting PBTS with top_c = {top_c}")
        result, _ = run_backtest(returns_df, top_c=top_c)
        results_top_c[top_c] = result['PBTS']
    
    # Test different sliding window values
    results_window = {}
    for window in sliding_window_values:
        print(f"\nTesting PBTS with sliding_window = {window}")
        result, _ = run_backtest(returns_df, sliding_window=window)
        results_window[window] = result['PBTS']
    
    # Plot sensitivity to top_c
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.bar([str(c) for c in top_c_values], [results_top_c[c]['Sharpe Ratio'] for c in top_c_values])
    plt.title('Sharpe Ratio vs. top_c')
    plt.xlabel('top_c')
    plt.ylabel('Sharpe Ratio')
    
    plt.subplot(2, 2, 2)
    plt.bar([str(c) for c in top_c_values], [results_top_c[c]['Cumulative Wealth'] for c in top_c_values])
    plt.title('Cumulative Wealth vs. top_c')
    plt.xlabel('top_c')
    plt.ylabel('Cumulative Wealth')
    
    plt.subplot(2, 2, 3)
    plt.bar([str(c) for c in top_c_values], [results_top_c[c]['Maximum Drawdown (%)'] for c in top_c_values])
    plt.title('Maximum Drawdown vs. top_c')
    plt.xlabel('top_c')
    plt.ylabel('Maximum Drawdown (%)')
    
    plt.subplot(2, 2, 4)
    plt.bar([str(c) for c in top_c_values], [results_top_c[c]['Volatility (%)'] for c in top_c_values])
    plt.title('Volatility vs. top_c')
    plt.xlabel('top_c')
    plt.ylabel('Volatility (%)')
    
    plt.tight_layout()
    plt.show()
    
    # Plot sensitivity to sliding window
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.bar([str(w) for w in sliding_window_values], [results_window[w]['Sharpe Ratio'] for w in sliding_window_values])
    plt.title('Sharpe Ratio vs. Sliding Window')
    plt.xlabel('Sliding Window')
    plt.ylabel('Sharpe Ratio')
    
    plt.subplot(2, 2, 2)
    plt.bar([str(w) for w in sliding_window_values], [results_window[w]['Cumulative Wealth'] for w in sliding_window_values])
    plt.title('Cumulative Wealth vs. Sliding Window')
    plt.xlabel('Sliding Window')
    plt.ylabel('Cumulative Wealth')
    
    plt.subplot(2, 2, 3)
    plt.bar([str(w) for w in sliding_window_values], [results_window[w]['Maximum Drawdown (%)'] for w in sliding_window_values])
    plt.title('Maximum Drawdown vs. Sliding Window')
    plt.xlabel('Sliding Window')
    plt.ylabel('Maximum Drawdown (%)')
    
    plt.subplot(2, 2, 4)
    plt.bar([str(w) for w in sliding_window_values], [results_window[w]['Volatility (%)'] for w in sliding_window_values])
    plt.title('Volatility vs. Sliding Window')
    plt.xlabel('Sliding Window')
    plt.ylabel('Volatility (%)')
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Part 1: Generate simulated data (monthly frequency, 25 assets, 10 years of data)
    returns_df = generate_simulated_data(n_assets=25, n_periods=120, freq='M')
    
    # Part 2: Run backtest with default parameters
    results, arm_history = run_backtest(returns_df, top_c=0.6, sliding_window=12)
    
    # Part 3: Plot results
    plot_results(results, arm_history)
    
    # Part 4: Parameter sensitivity analysis
    parameter_sensitivity_analysis(
        returns_df, 
        top_c_values=[0.3, 0.5, 0.7, 0.9],
        sliding_window_values=[6, 12, 24]
    )
    
    # Part 5: Generate daily data and run another backtest
    daily_returns_df = generate_simulated_data(n_assets=25, n_periods=500, freq='D')
    
    daily_results, daily_arm_history = run_backtest(daily_returns_df, top_c=0.6, sliding_window=60)
    
    # Part 6: Plot daily results
    plot_results(daily_results, daily_arm_history)