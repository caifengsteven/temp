import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.stats.stattools import durbin_watson
import seaborn as sns
from tqdm import tqdm

# Set random seed for reproducibility
np.random.seed(42)

class LeadLagFXStrategy:
    """
    A trading strategy based on lead-lag relationships in forex markets
    """
    def __init__(self, alpha=0.01, min_correlation=0.1, bonferroni_correction=True):
        """
        Initialize the strategy parameters
        
        Parameters:
        -----------
        alpha : float
            Significance level for statistical tests
        min_correlation : float
            Minimum correlation coefficient to consider a lead-lag relationship
        bonferroni_correction : bool
            Whether to apply Bonferroni correction for multiple testing
        """
        self.alpha = alpha
        self.min_correlation = min_correlation
        self.bonferroni_correction = bonferroni_correction
        self.lead_lag_pairs = []
        self.performance = {}
        
    def generate_simulated_data(self, n_pairs=10, n_points=10000, mean_reversion=0.01, noise_level=0.001):
        """
        Generate simulated forex data with built-in lead-lag relationships
        
        Parameters:
        -----------
        n_pairs : int
            Number of currency pairs to simulate
        n_points : int
            Number of time points to generate
        mean_reversion : float
            Mean reversion parameter for the leading currency pairs
        noise_level : float
            Standard deviation of the noise component
            
        Returns:
        --------
        data : pd.DataFrame
            DataFrame containing simulated forex data
        """
        # Generate random correlation matrix for the leading pairs
        np.random.seed(42)
        corr_matrix = np.random.uniform(-0.3, 0.3, size=(n_pairs//2, n_pairs//2))
        np.fill_diagonal(corr_matrix, 1.0)
        corr_matrix = (corr_matrix + corr_matrix.T) / 2  # Make symmetric
        
        # Calculate Cholesky decomposition for generating correlated random variables
        L = np.linalg.cholesky(corr_matrix)
        
        # Generate random starting prices
        starting_prices = np.random.uniform(0.5, 2.0, size=n_pairs)
        
        # Initialize prices array
        prices = np.zeros((n_points, n_pairs))
        prices[0] = starting_prices
        
        # Initialize returns array
        returns = np.zeros((n_points, n_pairs))
        
        # Generate data for leading pairs (first half of pairs)
        for i in range(1, n_points):
            # Generate correlated random returns for leading pairs
            z = np.random.normal(0, noise_level, size=n_pairs//2)
            leading_noise = np.dot(L, z)
            
            # Incorporate mean reversion
            mean_reversion_component = mean_reversion * (np.log(starting_prices[:n_pairs//2]) - np.log(prices[i-1, :n_pairs//2]))
            
            # Calculate returns for leading pairs
            returns[i, :n_pairs//2] = mean_reversion_component + leading_noise
            
            # Calculate prices for leading pairs
            prices[i, :n_pairs//2] = prices[i-1, :n_pairs//2] * np.exp(returns[i, :n_pairs//2])
        
        # Generate data for lagging pairs (second half of pairs)
        for i in range(1, n_points):
            # For the first time step, use random returns
            if i == 1:
                returns[i, n_pairs//2:] = np.random.normal(0, noise_level, size=n_pairs//2)
            else:
                # Add lagging effect: each lagging pair follows corresponding leading pair with 1-minute lag
                # plus some independent noise
                lagging_effect = np.zeros(n_pairs//2)
                for j in range(n_pairs//2):
                    # Create a specific lead-lag relationship between pair j and j+n_pairs//2
                    # Different pairs will have different lead-lag strengths
                    lead_lag_strength = 0.4 + 0.4 * j / (n_pairs//2)  # Ranges from 0.4 to 0.8
                    lagging_effect[j] = returns[i-1, j] * lead_lag_strength
                
                returns[i, n_pairs//2:] = lagging_effect + np.random.normal(0, noise_level, size=n_pairs//2)
            
            # Calculate prices for lagging pairs
            prices[i, n_pairs//2:] = prices[i-1, n_pairs//2:] * np.exp(returns[i, n_pairs//2:])
        
        # Create a pandas DataFrame
        columns = [f'Pair_{i+1}' for i in range(n_pairs)]
        df = pd.DataFrame(prices, columns=columns)
        
        # Add a timestamp column
        df['timestamp'] = pd.date_range(start='2023-01-01', periods=n_points, freq='1min')
        df.set_index('timestamp', inplace=True)
        
        return df
    
    def calculate_returns(self, prices, log=True):
        """
        Calculate returns from price series
        
        Parameters:
        -----------
        prices : pd.DataFrame
            DataFrame containing price data
        log : bool
            Whether to calculate log returns or simple returns
            
        Returns:
        --------
        returns : pd.DataFrame
            DataFrame containing return data
        """
        if log:
            returns = np.log(prices / prices.shift(1))
        else:
            returns = prices / prices.shift(1) - 1
        
        return returns.dropna()
    
    def find_lead_lag_relationships(self, returns, lag=1):
        """
        Find statistically significant lead-lag relationships
        
        Parameters:
        -----------
        returns : pd.DataFrame
            DataFrame containing return data
        lag : int
            Lag period in minutes
            
        Returns:
        --------
        lead_lag_matrix : pd.DataFrame
            Matrix of lead-lag correlation coefficients
        p_values : pd.DataFrame
            Matrix of p-values for the correlations
        """
        n_pairs = returns.shape[1]
        pair_names = returns.columns
        
        # Initialize matrices for correlations and p-values
        lead_lag_matrix = pd.DataFrame(np.zeros((n_pairs, n_pairs)), 
                                      index=pair_names, columns=pair_names)
        p_values = pd.DataFrame(np.ones((n_pairs, n_pairs)), 
                               index=pair_names, columns=pair_names)
        
        # Calculate the critical p-value with Bonferroni correction if requested
        if self.bonferroni_correction:
            critical_p = self.alpha / (n_pairs * n_pairs)
        else:
            critical_p = self.alpha
        
        # Calculate lagged correlations for all pairs
        for i in range(n_pairs):
            for j in range(n_pairs):
                if i != j:  # Skip self-correlations
                    leader = returns.iloc[:-lag, i].values
                    lagger = returns.iloc[lag:, j].values
                    
                    # Skip pairs with insufficient data
                    if len(leader) < 30 or len(lagger) < 30:
                        continue
                    
                    # Calculate correlation and p-value
                    corr, p_val = pearsonr(leader, lagger)
                    
                    lead_lag_matrix.iloc[i, j] = corr
                    p_values.iloc[i, j] = p_val
        
        # Store significant lead-lag pairs
        self.lead_lag_pairs = []
        for i in range(n_pairs):
            for j in range(n_pairs):
                if p_values.iloc[i, j] < critical_p and abs(lead_lag_matrix.iloc[i, j]) > self.min_correlation:
                    self.lead_lag_pairs.append({
                        'leader': pair_names[i],
                        'lagger': pair_names[j],
                        'correlation': lead_lag_matrix.iloc[i, j],
                        'p_value': p_values.iloc[i, j]
                    })
        
        return lead_lag_matrix, p_values
    
    def find_granger_causality(self, returns, lag=1, max_lag=1):
        """
        Test for Granger causality between all pairs
        
        Parameters:
        -----------
        returns : pd.DataFrame
            DataFrame containing return data
        lag : int
            The lag to consider for causality
        max_lag : int
            Maximum lag to include in the test
            
        Returns:
        --------
        causality_matrix : pd.DataFrame
            Matrix with p-values for Granger causality tests
        """
        n_pairs = returns.shape[1]
        pair_names = returns.columns
        
        # Initialize matrix for p-values
        causality_matrix = pd.DataFrame(np.ones((n_pairs, n_pairs)), 
                                       index=pair_names, columns=pair_names)
        
        # Calculate the critical p-value with Bonferroni correction if requested
        if self.bonferroni_correction:
            critical_p = self.alpha / (n_pairs * n_pairs)
        else:
            critical_p = self.alpha
        
        # Calculate Granger causality for all pairs
        for i in tqdm(range(n_pairs), desc="Testing Granger causality"):
            for j in range(n_pairs):
                if i != j:  # Skip self-causality
                    # Get data for the two pairs
                    data = pd.DataFrame({
                        'x': returns.iloc[:, i],
                        'y': returns.iloc[:, j]
                    }).dropna()
                    
                    # Skip pairs with insufficient data
                    if len(data) < 30:
                        continue
                    
                    try:
                        # Test whether x Granger causes y
                        test_result = grangercausalitytests(data, maxlag=max_lag, verbose=False)
                        
                        # Get p-value for the test at the specified lag
                        p_value = test_result[lag][0]['ssr_ftest'][1]
                        causality_matrix.iloc[i, j] = p_value
                    except:
                        # In case of numerical issues, keep the default p-value of 1
                        pass
        
        # Store significant causal relationships
        self.causal_pairs = []
        for i in range(n_pairs):
            for j in range(n_pairs):
                if causality_matrix.iloc[i, j] < critical_p:
                    self.causal_pairs.append({
                        'leader': pair_names[i],
                        'lagger': pair_names[j],
                        'p_value': causality_matrix.iloc[i, j]
                    })
        
        return causality_matrix
    
    def backtest_strategy(self, prices, lag=1, holding_period=1, 
                         take_profit=0.005, stop_loss=0.005, 
                         transaction_cost=0.0001):
        """
        Backtest a trading strategy based on lead-lag relationships
        
        Parameters:
        -----------
        prices : pd.DataFrame
            DataFrame containing price data
        lag : int
            Lag period in minutes
        holding_period : int
            Number of periods to hold each position
        take_profit : float
            Take profit threshold as a percentage
        stop_loss : float
            Stop loss threshold as a percentage
        transaction_cost : float
            Transaction cost as a percentage
            
        Returns:
        --------
        performance : dict
            Dictionary containing performance metrics
        """
        if not self.lead_lag_pairs:
            print("No lead-lag relationships found. Run find_lead_lag_relationships first.")
            return
        
        # Calculate returns
        returns = self.calculate_returns(prices)
        
        # Initialize performance tracking
        total_trades = 0
        winning_trades = 0
        total_return = 0
        returns_list = []
        positions = pd.DataFrame(0, index=prices.index, columns=prices.columns)
        
        # Test each lead-lag pair
        for pair in self.lead_lag_pairs:
            leader = pair['leader']
            lagger = pair['lagger']
            correlation = pair['correlation']
            
            # Skip if correlation is too low
            if abs(correlation) < self.min_correlation:
                continue
                
            # Initialize tracking for this pair
            trades = 0
            wins = 0
            pair_return = 0
            pair_returns = []
            
            # Start from lag+1 to have enough history
            for t in range(lag+1, len(returns)):
                # Get the return of the leader at the previous period
                leader_return = returns[leader].iloc[t-lag]
                
                # Skip if the leader return is too small (noise)
                if abs(leader_return) < 0.0001:
                    continue
                
                # Determine position direction based on leader return and correlation
                if correlation > 0:
                    direction = 1 if leader_return > 0 else -1
                else:
                    direction = -1 if leader_return > 0 else 1
                
                # Take position in the lagger
                if positions.iloc[t, prices.columns.get_loc(lagger)] == 0:  # No existing position
                    positions.iloc[t, prices.columns.get_loc(lagger)] = direction
                    entry_price = prices[lagger].iloc[t]
                    entry_time = t
                    
                    # Track until exit
                    exit_time = min(t + holding_period, len(prices) - 1)
                    exit_price = prices[lagger].iloc[exit_time]
                    
                    # Calculate return (account for direction and transaction costs)
                    if direction == 1:
                        trade_return = (exit_price / entry_price - 1) - 2 * transaction_cost
                    else:
                        trade_return = (entry_price / exit_price - 1) - 2 * transaction_cost
                    
                    # Update statistics
                    trades += 1
                    if trade_return > 0:
                        wins += 1
                    pair_return += trade_return
                    pair_returns.append(trade_return)
            
            # Store performance for this pair
            if trades > 0:
                self.performance[f"{leader}->{lagger}"] = {
                    'trades': trades,
                    'win_rate': wins / trades,
                    'total_return': pair_return,
                    'avg_return': pair_return / trades if trades > 0 else 0,
                    'returns': pair_returns
                }
                
                # Update overall statistics
                total_trades += trades
                winning_trades += wins
                total_return += pair_return
                returns_list.extend(pair_returns)
        
        # Calculate overall performance metrics
        if total_trades > 0:
            sharpe_ratio = np.mean(returns_list) / (np.std(returns_list) + 1e-10) * np.sqrt(252 * 1440 / holding_period)
            win_rate = winning_trades / total_trades
            avg_return = total_return / total_trades
            
            # Calculate drawdown
            cumulative_returns = np.cumsum(returns_list)
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdown = running_max - cumulative_returns
            max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0
            
            performance_summary = {
                'total_trades': total_trades,
                'win_rate': win_rate,
                'total_return': total_return,
                'avg_return': avg_return,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'returns': returns_list
            }
            
            return performance_summary
        else:
            print("No trades executed during the backtest period.")
            return None

    def plot_lead_lag_matrix(self, lead_lag_matrix, title="Lead-Lag Correlation Matrix"):
        """
        Plot the lead-lag correlation matrix as a heatmap
        
        Parameters:
        -----------
        lead_lag_matrix : pd.DataFrame
            Matrix of lead-lag correlation coefficients
        title : str
            Title for the plot
        """
        plt.figure(figsize=(12, 10))
        sns.heatmap(lead_lag_matrix, cmap='coolwarm', center=0, 
                   annot=False, fmt='.2f', cbar=True)
        plt.title(title)
        plt.xlabel('Lagger')
        plt.ylabel('Leader')
        plt.tight_layout()
        plt.show()
    
    def plot_performance(self, performance):
        """
        Plot the performance of the strategy
        
        Parameters:
        -----------
        performance : dict
            Dictionary containing performance metrics
        """
        returns = performance['returns']
        
        plt.figure(figsize=(15, 10))
        
        # Plot cumulative returns
        plt.subplot(2, 2, 1)
        plt.plot(np.cumsum(returns), 'b-')
        plt.title('Cumulative Returns')
        plt.grid(True)
        
        # Plot return distribution - use basic matplotlib histogram instead of seaborn
        plt.subplot(2, 2, 2)
        plt.hist(returns, bins=30, alpha=0.7, color='blue', density=True)
        plt.axvline(x=0, color='r', linestyle='--')
        plt.title('Return Distribution')
        
        # Plot drawdown
        plt.subplot(2, 2, 3)
        cumulative_returns = np.cumsum(returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = running_max - cumulative_returns
        plt.plot(drawdown, 'r-')
        plt.title('Drawdown')
        plt.grid(True)
        
        # Plot performance metrics
        plt.subplot(2, 2, 4)
        metrics = [
            f"Total Trades: {performance['total_trades']}",
            f"Win Rate: {performance['win_rate']:.2%}",
            f"Total Return: {performance['total_return']:.2%}",
            f"Avg Return per Trade: {performance['avg_return']:.2%}",
            f"Sharpe Ratio: {performance['sharpe_ratio']:.2f}",
            f"Max Drawdown: {performance['max_drawdown']:.2%}"
        ]
        plt.axis('off')
        plt.text(0.1, 0.9, '\n'.join(metrics), fontsize=12)
        plt.title('Performance Metrics')
        
        plt.tight_layout()
        plt.show()
        
    def plot_pair_performance(self, pair_key):
        """
        Plot the performance of a specific lead-lag pair
        
        Parameters:
        -----------
        pair_key : str
            Key of the pair in the performance dictionary
        """
        if pair_key not in self.performance:
            print(f"Pair {pair_key} not found in performance results.")
            return
        
        pair_data = self.performance[pair_key]
        returns = pair_data['returns']
        
        plt.figure(figsize=(15, 5))
        
        # Plot cumulative returns
        plt.subplot(1, 2, 1)
        plt.plot(np.cumsum(returns), 'b-')
        plt.title(f'Cumulative Returns for {pair_key}')
        plt.grid(True)
        
        # Plot return distribution - use basic matplotlib histogram instead of seaborn
        plt.subplot(1, 2, 2)
        plt.hist(returns, bins=30, alpha=0.7, color='blue', density=True)
        plt.axvline(x=0, color='r', linestyle='--')
        plt.title(f'Return Distribution for {pair_key}')
        
        plt.tight_layout()
        plt.show()
        
        # Print performance metrics
        print(f"Performance for {pair_key}:")
        print(f"Trades: {pair_data['trades']}")
        print(f"Win Rate: {pair_data['win_rate']:.2%}")
        print(f"Total Return: {pair_data['total_return']:.2%}")
        print(f"Average Return per Trade: {pair_data['avg_return']:.2%}")


def run_simulation():
    """
    Run a simulation of the lead-lag strategy on simulated data
    """
    # Initialize strategy
    strategy = LeadLagFXStrategy(alpha=0.01, min_correlation=0.1, bonferroni_correction=True)
    
    # Generate simulated data
    print("Generating simulated data...")
    prices = strategy.generate_simulated_data(n_pairs=10, n_points=10000, 
                                             mean_reversion=0.01, noise_level=0.001)
    
    # Calculate returns
    returns = strategy.calculate_returns(prices)
    
    # Find lead-lag relationships
    print("Finding lead-lag relationships...")
    lead_lag_matrix, p_values = strategy.find_lead_lag_relationships(returns)
    
    # Plot lead-lag matrix
    strategy.plot_lead_lag_matrix(lead_lag_matrix)
    
    # Print lead-lag pairs
    print("\nSignificant lead-lag pairs:")
    for pair in strategy.lead_lag_pairs:
        print(f"{pair['leader']} -> {pair['lagger']}: corr = {pair['correlation']:.4f}, p = {pair['p_value']:.6f}")
    
    # Test for Granger causality
    print("\nTesting for Granger causality...")
    causality_matrix = strategy.find_granger_causality(returns)
    
    # Plot causality matrix
    strategy.plot_lead_lag_matrix(1 - causality_matrix, title="Granger Causality (1 - p-value)")
    
    # Print causal pairs
    print("\nSignificant Granger causal relationships:")
    for pair in strategy.causal_pairs:
        print(f"{pair['leader']} -> {pair['lagger']}: p = {pair['p_value']:.6f}")
    
    # Backtest the strategy
    print("\nBacktesting the strategy...")
    performance = strategy.backtest_strategy(prices, lag=1, holding_period=5)
    
    # Plot performance
    if performance:
        strategy.plot_performance(performance)
        
        # Plot performance for some individual pairs
        if strategy.performance:
            for pair_key in list(strategy.performance.keys())[:3]:  # Plot first 3 pairs
                strategy.plot_pair_performance(pair_key)
    
    return strategy, prices, returns, performance


if __name__ == "__main__":
    strategy, prices, returns, performance = run_simulation()