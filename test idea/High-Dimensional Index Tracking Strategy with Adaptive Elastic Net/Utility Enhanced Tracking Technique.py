import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.optimize import minimize
from scipy.linalg import eig
import datetime as dt
import warnings
warnings.filterwarnings('ignore')

# Try to import pdblp for Bloomberg access
try:
    import pdblp
    BLOOMBERG_AVAILABLE = True
except ImportError:
    BLOOMBERG_AVAILABLE = False
    print("pdblp not installed, Bloomberg data will not be available.")

class UtilityFunctions:
    """Class containing various utility functions for risk-averse investors"""
    
    @staticmethod
    def log_utility(x):
        """
        Logarithmic utility function (DARA - Decreasing Absolute Risk Aversion)
        
        Args:
            x: array of outcomes (returns)
            
        Returns:
            array of utility values
        """
        # Ensure x is greater than 0 for log function
        x = np.maximum(x, 1e-10)
        return np.log(x)
    
    @staticmethod
    def quadratic_utility(x, b=0.5):
        """
        Quadratic utility function (IARA - Increasing Absolute Risk Aversion)
        U(x) = x - b*x^2
        
        Args:
            x: array of outcomes (returns)
            b: risk aversion parameter
            
        Returns:
            array of utility values
        """
        return x - b * x**2
    
    @staticmethod
    def exponential_utility(x, a=2):
        """
        Negative exponential utility function (CARA - Constant Absolute Risk Aversion)
        U(x) = -exp(-a*x)
        
        Args:
            x: array of outcomes (returns)
            a: risk aversion parameter
            
        Returns:
            array of utility values
        """
        return -np.exp(-a * x)


class CUARCalculator:
    """
    Class to calculate CUAR (Cumulative Utility Area Ratio) between two assets
    """
    
    def __init__(self, utility_function=UtilityFunctions.log_utility):
        """
        Initialize the CUAR calculator
        
        Args:
            utility_function: function that maps outcomes to utility
        """
        self.utility_function = utility_function
    
    def calculate_cuar(self, returns_a, returns_b):
        """
        Calculate the CUAR criterion between two assets
        
        Args:
            returns_a: array of returns for asset A
            returns_b: array of returns for asset B
            
        Returns:
            CUAR criterion value
        """
        # Combine and sort all returns
        all_returns = np.concatenate([returns_a, returns_b])
        sorted_unique_returns = np.sort(np.unique(all_returns))
        
        # Create empirical CDFs
        cdf_a = self._create_empirical_cdf(returns_a, sorted_unique_returns)
        cdf_b = self._create_empirical_cdf(returns_b, sorted_unique_returns)
        
        # Calculate utility for each return level
        utilities = self.utility_function(1 + sorted_unique_returns)
        
        # Compute utility differences 
        utility_diffs = np.diff(utilities)
        
        # Calculate areas where F < G and F > G
        area_f_less_than_g = 0
        area_f_greater_than_g = 0
        
        for i in range(len(sorted_unique_returns) - 1):
            cdf_diff = cdf_b[i] - cdf_a[i]
            if cdf_diff > 0:  # F(x) < G(x)
                area_f_less_than_g += cdf_diff * utility_diffs[i]
            elif cdf_diff < 0:  # F(x) > G(x)
                area_f_greater_than_g += -cdf_diff * utility_diffs[i]
        
        # Handle the case where one area is zero (dominance case)
        if area_f_greater_than_g == 0:
            return 0.00001  # Asset B dominates Asset A (avoid exact zero)
        elif area_f_less_than_g == 0:
            return 10000  # Asset A dominates Asset B (use large number instead of infinity)
        
        # Return the CUAR criterion
        return area_f_less_than_g / area_f_greater_than_g
    
    def _create_empirical_cdf(self, returns, grid_points):
        """
        Create empirical CDF for a set of returns evaluated at grid points
        
        Args:
            returns: array of returns
            grid_points: points at which to evaluate the CDF
            
        Returns:
            CDF values at grid points
        """
        n = len(returns)
        cdf = np.zeros_like(grid_points)
        
        for i, x in enumerate(grid_points):
            cdf[i] = np.sum(returns <= x) / n
            
        return cdf


class AHPCalculator:
    """
    Class to implement the Analytic Hierarchy Process (AHP)
    """
    
    @staticmethod
    def calculate_weights(matrix):
        """
        Calculate weights from a pairwise comparison matrix using AHP
        
        Args:
            matrix: N x N pairwise comparison matrix
            
        Returns:
            array of weights
        """
        # Calculate the principal eigenvector of the matrix
        eigenvalues, eigenvectors = eig(matrix)
        
        # Find the index of the largest eigenvalue
        max_idx = np.argmax(np.real(eigenvalues))
        
        # Extract the corresponding eigenvector
        weights = np.real(eigenvectors[:, max_idx])
        
        # Normalize weights to sum to 1
        weights = weights / np.sum(weights)
        
        return weights
    
    @staticmethod
    def calculate_consistency_ratio(matrix):
        """
        Calculate the consistency ratio of a pairwise comparison matrix
        
        Args:
            matrix: N x N pairwise comparison matrix
            
        Returns:
            Modified Consistency Ratio (MCR)
        """
        n = matrix.shape[0]
        eigenvalues, _ = eig(matrix)
        lambda_max = np.max(np.real(eigenvalues))
        
        # Calculate the Modified Consistency Ratio
        mu_max = (2.7699 * n) - 4.3513
        mcr = (lambda_max - n) / ((1.7699 * n) - 4.3513)
        
        return mcr


class UtilityEnhancedIndexTracker:
    """
    Implements the Utility Enhanced Tracking Technique (UETT)
    """
    
    def __init__(self, utility_function=UtilityFunctions.log_utility, 
                 consistency_threshold=0.1, max_cuar_value=100):
        """
        Initialize the UETT
        
        Args:
            utility_function: function that maps outcomes to utility
            consistency_threshold: threshold for the modified consistency ratio
            max_cuar_value: maximum allowed value for CUAR to prevent extreme values
        """
        self.utility_function = utility_function
        self.consistency_threshold = consistency_threshold
        self.max_cuar_value = max_cuar_value
        self.cuar_calculator = CUARCalculator(utility_function)
        self.weights = None
        self.selected_stocks = None
        self.dominated_stocks = []
    
    def calculate_enhanced_weights(self, returns_data):
        """
        Calculate the enhanced index weights using UETT
        
        Args:
            returns_data: DataFrame with stock returns (rows are dates, columns are stocks)
            
        Returns:
            Series of weights for each stock
        """
        stocks = returns_data.columns
        n_stocks = len(stocks)
        
        # Create matrix to store CUAR values
        cuar_matrix = np.ones((n_stocks, n_stocks))
        
        # Identify dominated stocks
        self.dominated_stocks = []
        dominance_threshold = 0.01  # Threshold for determining dominance
        
        # Calculate CUAR for each pair of stocks
        for i in range(n_stocks):
            for j in range(i+1, n_stocks):
                try:
                    cuar = self.cuar_calculator.calculate_cuar(
                        returns_data.iloc[:, i].values, 
                        returns_data.iloc[:, j].values
                    )
                    
                    # Handle extreme values (potential dominance cases)
                    if cuar < dominance_threshold:  # Stock i is dominated by stock j
                        self.dominated_stocks.append(stocks[i])
                        cuar = dominance_threshold  # Small but non-zero value
                    elif cuar > self.max_cuar_value:  # Stock j is dominated by stock i
                        self.dominated_stocks.append(stocks[j])
                        cuar = self.max_cuar_value  # Large but finite value
                    
                    cuar_matrix[i, j] = cuar
                    cuar_matrix[j, i] = 1 / cuar
                except Exception as e:
                    print(f"Error calculating CUAR for {stocks[i]} and {stocks[j]}: {e}")
                    # Use neutral value in case of error
                    cuar_matrix[i, j] = 1.0
                    cuar_matrix[j, i] = 1.0
        
        # Remove dominated stocks
        self.dominated_stocks = list(set(self.dominated_stocks))
        valid_stocks = [s for s in stocks if s not in self.dominated_stocks]
        valid_indices = [i for i, s in enumerate(stocks) if s not in self.dominated_stocks]
        
        # Extract valid submatrix
        if self.dominated_stocks:
            valid_matrix = cuar_matrix[np.ix_(valid_indices, valid_indices)]
        else:
            valid_matrix = cuar_matrix
        
        # Check consistency
        mcr = AHPCalculator.calculate_consistency_ratio(valid_matrix)
        if mcr > self.consistency_threshold:
            print(f"Warning: Matrix is not consistent (MCR = {mcr:.4f})")
            
            # If inconsistent, try to improve consistency by identifying and removing
            # the most inconsistent stocks
            if mcr > 0.2 and len(valid_stocks) > 15:  # Very inconsistent and enough stocks
                # Simple approach: identify stocks that contribute most to inconsistency
                # by looking at row-wise variance
                row_variances = np.var(valid_matrix, axis=1)
                worst_idx = np.argmax(row_variances)
                worst_stock = valid_stocks[worst_idx]
                
                # Remove the worst stock and recalculate
                print(f"Removing stock {worst_stock} to improve consistency")
                self.dominated_stocks.append(worst_stock)
                
                # Update valid stocks and indices
                valid_stocks = [s for s in stocks if s not in self.dominated_stocks]
                valid_indices = [i for i, s in enumerate(stocks) if s not in self.dominated_stocks]
                
                # Extract valid submatrix again
                valid_matrix = cuar_matrix[np.ix_(valid_indices, valid_indices)]
                
                # Check consistency again
                mcr = AHPCalculator.calculate_consistency_ratio(valid_matrix)
                print(f"After removal, MCR = {mcr:.4f}")
        
        # Calculate weights
        weights = AHPCalculator.calculate_weights(valid_matrix)
        
        # Create Series with weights
        weights_series = pd.Series(weights, index=valid_stocks)
        
        # Store the weights and selected stocks
        self.weights = weights_series
        self.selected_stocks = valid_stocks
        
        return weights_series
    
    def create_cardinality_constrained_portfolio(self, returns_data, benchmark_returns, 
                                                target_correlation=0.9, max_stocks=None):
        """
        Create a cardinality constrained portfolio
        
        Args:
            returns_data: DataFrame with stock returns
            benchmark_returns: Series with benchmark returns
            target_correlation: minimum required correlation with benchmark
            max_stocks: maximum number of stocks to include
            
        Returns:
            Series of weights for the constrained portfolio
        """
        if self.weights is None:
            self.calculate_enhanced_weights(returns_data)
        
        # Calculate CUAR for each stock against the benchmark
        cuars = {}
        for stock in self.selected_stocks:
            try:
                cuar = self.cuar_calculator.calculate_cuar(
                    returns_data[stock].values, 
                    benchmark_returns.values
                )
                # Cap extremely high values
                cuars[stock] = min(cuar, self.max_cuar_value)
            except Exception as e:
                print(f"Error calculating CUAR for {stock} against benchmark: {e}")
                # Use average CUAR in case of error
                cuars[stock] = 1.0
        
        # Sort stocks by CUAR
        sorted_stocks = sorted(cuars.keys(), key=lambda x: cuars[x], reverse=True)
        
        # Limit to max_stocks if specified
        if max_stocks is not None:
            sorted_stocks = sorted_stocks[:max_stocks]
        
        # Find minimum set of stocks to achieve target correlation
        portfolio_stocks = []
        current_correlation = 0
        
        for stock in sorted_stocks:
            portfolio_stocks.append(stock)
            
            # Calculate weights for current portfolio stocks
            original_weights = self.weights[portfolio_stocks]
            portfolio_weights = original_weights / original_weights.sum()
            
            # Calculate portfolio returns
            portfolio_returns = returns_data[portfolio_stocks].dot(portfolio_weights)
            
            # Calculate correlation with benchmark
            current_correlation = portfolio_returns.corr(benchmark_returns)
            
            if current_correlation >= target_correlation:
                break
            
            if max_stocks is not None and len(portfolio_stocks) >= max_stocks:
                break
        
        print(f"Created portfolio with {len(portfolio_stocks)} stocks, correlation: {current_correlation:.4f}")
        
        # Calculate final weights
        final_weights = pd.Series(0, index=returns_data.columns)
        subset_weights = self.weights[portfolio_stocks]
        normalized_weights = subset_weights / subset_weights.sum()
        final_weights[portfolio_stocks] = normalized_weights
        
        return final_weights


class UtilityEnhancedBacktest:
    """
    Class to backtest the Utility Enhanced Tracking Technique
    """
    
    def __init__(self, utility_function=UtilityFunctions.log_utility, 
                 rebalance_freq='M', lookback_window=126):
        """
        Initialize the backtest
        
        Args:
            utility_function: function that maps outcomes to utility
            rebalance_freq: rebalancing frequency ('W' for weekly, 'M' for monthly)
            lookback_window: number of trading days to use for weight calculation
        """
        self.utility_function = utility_function
        self.rebalance_freq = rebalance_freq
        self.lookback_window = lookback_window
        self.tracker = UtilityEnhancedIndexTracker(utility_function)
        
    def run(self, returns_data, benchmark_returns, cardinality_constraint=None, 
            target_correlation=0.9, transaction_cost=0.0005):
        """
        Run the backtest
        
        Args:
            returns_data: DataFrame with stock returns
            benchmark_returns: Series with benchmark returns
            cardinality_constraint: maximum number of stocks to include
            target_correlation: minimum correlation with benchmark if using cardinality constraint
            transaction_cost: transaction cost per trade (one-way)
            
        Returns:
            DataFrame with backtest results
        """
        # Ensure index is datetime
        returns_data.index = pd.to_datetime(returns_data.index)
        benchmark_returns.index = pd.to_datetime(benchmark_returns.index)
        
        # Align benchmark with returns data
        common_index = returns_data.index.intersection(benchmark_returns.index)
        returns_data = returns_data.loc[common_index]
        benchmark_returns = benchmark_returns.loc[common_index]
        
        # Calculate rebalance dates
        if self.rebalance_freq == 'W':
            rebalance_dates = pd.date_range(
                start=returns_data.index[self.lookback_window], 
                end=returns_data.index[-1], 
                freq='W-FRI'
            )
        else:  # Monthly
            rebalance_dates = pd.date_range(
                start=returns_data.index[self.lookback_window], 
                end=returns_data.index[-1], 
                freq='BM'
            )
        
        # Keep only rebalance dates that are in the returns data
        rebalance_dates = [d for d in rebalance_dates if d in returns_data.index]
        
        # Initialize weights and portfolio values
        weights = pd.DataFrame(index=returns_data.index, columns=returns_data.columns)
        portfolio_returns = pd.Series(index=returns_data.index, dtype=float)
        portfolio_values = pd.Series(index=returns_data.index, dtype=float)
        portfolio_values.iloc[0] = 1.0
        
        # Calculate weights for each rebalance date
        for i, rebalance_date in enumerate(rebalance_dates):
            # Find lookback window end date
            lookback_end_idx = returns_data.index.get_loc(rebalance_date)
            lookback_start_idx = max(0, lookback_end_idx - self.lookback_window)
            
            # Get lookback data
            lookback_data = returns_data.iloc[lookback_start_idx:lookback_end_idx]
            lookback_benchmark = benchmark_returns.iloc[lookback_start_idx:lookback_end_idx]
            
            try:
                # Calculate weights
                if cardinality_constraint is None:
                    new_weights = self.tracker.calculate_enhanced_weights(lookback_data)
                    weights.loc[rebalance_date] = new_weights.reindex(weights.columns, fill_value=0)
                else:
                    # First calculate full enhanced weights
                    self.tracker.calculate_enhanced_weights(lookback_data)
                    
                    # Then apply cardinality constraint
                    new_weights = self.tracker.create_cardinality_constrained_portfolio(
                        lookback_data, 
                        lookback_benchmark, 
                        target_correlation=target_correlation,
                        max_stocks=cardinality_constraint
                    )
                    weights.loc[rebalance_date] = new_weights
            
            except Exception as e:
                print(f"Error calculating weights for {rebalance_date}: {e}")
                # Use previous weights or equal weights if first period
                if i > 0:
                    weights.loc[rebalance_date] = weights.loc[rebalance_dates[i-1]]
                else:
                    equal_weight = 1.0 / len(returns_data.columns)
                    weights.loc[rebalance_date] = pd.Series(equal_weight, index=returns_data.columns)
            
            # Find next rebalance date or end of data
            if i < len(rebalance_dates) - 1:
                next_rebalance_date = rebalance_dates[i + 1]
                hold_period = returns_data.loc[rebalance_date:next_rebalance_date].index
            else:
                hold_period = returns_data.loc[rebalance_date:].index
            
            # Forward fill weights
            for date in hold_period[1:]:  # Skip rebalance date
                weights.loc[date] = weights.loc[rebalance_date]
        
        # Forward fill any missing weights
        weights = weights.ffill()
        
        # Calculate portfolio returns including transaction costs
        prev_weights = None
        for i, date in enumerate(returns_data.index[1:], start=1):
            if prev_weights is None:
                # First trading day with weights
                if not pd.isna(weights.loc[date]).all():
                    prev_weights = weights.loc[date]
                    portfolio_returns.loc[date] = np.sum(returns_data.loc[date] * prev_weights)
                    # Subtract transaction costs for initial portfolio setup
                    portfolio_returns.loc[date] -= np.sum(np.abs(prev_weights)) * transaction_cost
            else:
                current_weights = weights.loc[date]
                
                # Check if weights have changed (rebalancing)
                if not np.array_equal(current_weights, prev_weights):
                    # Calculate weight changes
                    weight_changes = current_weights - prev_weights
                    # Add transaction costs
                    portfolio_returns.loc[date] = np.sum(returns_data.loc[date] * current_weights)
                    portfolio_returns.loc[date] -= np.sum(np.abs(weight_changes)) * transaction_cost
                    prev_weights = current_weights
                else:
                    # No rebalancing
                    portfolio_returns.loc[date] = np.sum(returns_data.loc[date] * current_weights)
        
        # Calculate portfolio values
        for i, date in enumerate(returns_data.index[1:], start=1):
            if pd.isna(portfolio_returns.loc[date]):
                portfolio_values.loc[date] = portfolio_values.iloc[i-1]
            else:
                portfolio_values.loc[date] = portfolio_values.iloc[i-1] * (1 + portfolio_returns.loc[date])
        
        # Create results DataFrame
        results = pd.DataFrame({
            'Benchmark': benchmark_returns,
            'Enhanced Portfolio': portfolio_returns
        })
        
        # Calculate cumulative returns
        results['Cumulative Benchmark'] = (1 + benchmark_returns).cumprod()
        results['Cumulative Enhanced'] = portfolio_values
        
        return results, weights


class PerformanceAnalyzer:
    """
    Class to analyze performance of investment strategies
    """
    
    @staticmethod
    def calculate_cuar(returns_a, returns_b, utility_function=UtilityFunctions.log_utility):
        """
        Calculate CUAR between two return series
        
        Args:
            returns_a: Series of returns for strategy A
            returns_b: Series of returns for strategy B
            utility_function: utility function to use
            
        Returns:
            CUAR value
        """
        calculator = CUARCalculator(utility_function)
        return calculator.calculate_cuar(returns_a.values, returns_b.values)
    
    @staticmethod
    def calculate_omega_ratio(returns, threshold=0):
        """
        Calculate the Omega ratio
        
        Args:
            returns: Series of returns
            threshold: threshold return
            
        Returns:
            Omega ratio
        """
        # Ensure returns is a numpy array
        returns = np.array(returns)
        
        # Calculate areas above and below threshold
        returns_above = returns[returns > threshold] - threshold
        returns_below = threshold - returns[returns < threshold]
        
        # Avoid division by zero
        if len(returns_below) == 0 or np.sum(returns_below) == 0:
            return float('inf')
        
        # Calculate Omega ratio
        omega = np.sum(returns_above) / np.sum(returns_below)
        
        return omega
    
    @staticmethod
    def calculate_metrics(returns, benchmark_returns=None, risk_free_rate=0):
        """
        Calculate performance metrics
        
        Args:
            returns: Series of returns
            benchmark_returns: Series of benchmark returns (optional)
            risk_free_rate: Risk-free rate (annualized)
            
        Returns:
            Dictionary of performance metrics
        """
        # Convert annualized risk-free rate to period rate
        period_rf = risk_free_rate / 252  # assuming daily returns
        
        # Basic return metrics
        avg_return = returns.mean()
        ann_return = (1 + avg_return) ** 252 - 1
        
        # Risk metrics
        std_dev = returns.std()
        ann_std_dev = std_dev * np.sqrt(252)
        
        # Downside metrics
        downside_returns = returns[returns < 0]
        downside_deviation = downside_returns.std() if len(downside_returns) > 0 else 0
        ann_downside_dev = downside_deviation * np.sqrt(252)
        
        # Sharpe and Sortino ratios
        sharpe_ratio = (avg_return - period_rf) / std_dev if std_dev > 0 else 0
        ann_sharpe = sharpe_ratio * np.sqrt(252)
        
        sortino_ratio = (avg_return - period_rf) / downside_deviation if downside_deviation > 0 else 0
        ann_sortino = sortino_ratio * np.sqrt(252)
        
        # Maximum drawdown
        cumulative = (1 + returns).cumprod()
        max_dd = 1 - cumulative.div(cumulative.cummax()).min()
        
        # Omega ratio
        omega_ratio = PerformanceAnalyzer.calculate_omega_ratio(returns)
        
        # Win ratio
        win_ratio = (returns > 0).mean()
        
        # Tracking error (if benchmark provided)
        tracking_error = 0
        correlation = 0
        if benchmark_returns is not None:
            # Ensure alignment
            aligned_returns = returns.reindex(benchmark_returns.index)
            tracking_error = (aligned_returns - benchmark_returns).std() * np.sqrt(252)
            correlation = aligned_returns.corr(benchmark_returns)
        
        # Compile metrics
        metrics = {
            'Average Daily Return': avg_return,
            'Annualized Return': ann_return,
            'Annualized Standard Deviation': ann_std_dev,
            'Annualized Downside Deviation': ann_downside_dev,
            'Sharpe Ratio': ann_sharpe,
            'Sortino Ratio': ann_sortino,
            'Maximum Drawdown': max_dd,
            'Omega Ratio': omega_ratio,
            'Win Ratio': win_ratio,
            'Tracking Error': tracking_error,
            'Correlation with Benchmark': correlation
        }
        
        return metrics
    
    @staticmethod
    def plot_performance(results, title="Portfolio Performance"):
        """
        Plot performance of the enhanced index vs benchmark
        
        Args:
            results: DataFrame with performance results
            title: Plot title
        """
        plt.figure(figsize=(12, 10))
        
        # Plot cumulative returns
        plt.subplot(2, 1, 1)
        plt.plot(results['Cumulative Benchmark'], label='Benchmark')
        plt.plot(results['Cumulative Enhanced'], label='Enhanced Portfolio')
        plt.title(f'{title} - Cumulative Returns')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Return')
        plt.legend()
        plt.grid(True)
        
        # Plot drawdowns
        plt.subplot(2, 1, 2)
        benchmark_dd = 1 - results['Cumulative Benchmark'] / results['Cumulative Benchmark'].cummax()
        enhanced_dd = 1 - results['Cumulative Enhanced'] / results['Cumulative Enhanced'].cummax()
        
        plt.plot(benchmark_dd, label='Benchmark Drawdown')
        plt.plot(enhanced_dd, label='Enhanced Portfolio Drawdown')
        plt.title('Drawdowns')
        plt.xlabel('Date')
        plt.ylabel('Drawdown')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def print_metrics_comparison(metrics_a, metrics_b, name_a="Strategy A", name_b="Strategy B"):
        """
        Print comparison of performance metrics
        
        Args:
            metrics_a: Dictionary of metrics for strategy A
            metrics_b: Dictionary of metrics for strategy B
            name_a: Name of strategy A
            name_b: Name of strategy B
        """
        metrics_df = pd.DataFrame({
            name_a: pd.Series(metrics_a),
            name_b: pd.Series(metrics_b)
        })
        
        # Format percentages
        percent_metrics = [
            'Average Daily Return', 'Annualized Return', 
            'Annualized Standard Deviation', 'Annualized Downside Deviation',
            'Maximum Drawdown', 'Win Ratio', 'Tracking Error'
        ]
        
        for metric in percent_metrics:
            if metric in metrics_df.index:
                metrics_df.loc[metric] = metrics_df.loc[metric].map('{:.2%}'.format)
        
        # Format ratios
        ratio_metrics = ['Sharpe Ratio', 'Sortino Ratio', 'Omega Ratio', 'Correlation with Benchmark']
        for metric in ratio_metrics:
            if metric in metrics_df.index:
                metrics_df.loc[metric] = metrics_df.loc[metric].map('{:.4f}'.format)
        
        print(metrics_df)


def generate_sample_data_with_inefficiencies(n_stocks=30, n_days=1000, seed=42):
    """
    Generate sample data similar to DJIA or S&P 500 returns with inefficiencies
    that can be exploited by utility-based methods.
    
    Args:
        n_stocks: Number of stocks to simulate
        n_days: Number of days to simulate
        seed: Random seed
        
    Returns:
        DataFrame with simulated price data, returns data, and benchmark
    """
    np.random.seed(seed)
    
    # Create date range (business days)
    end_date = pd.Timestamp.today()
    start_date = end_date - pd.Timedelta(days=int(n_days * 1.5))  # Add buffer for business days
    dates = pd.date_range(start=start_date, end=end_date, freq='B')[:n_days]
    
    # Create stock names
    stock_names = [f'Stock_{i+1}' for i in range(n_stocks)]
    
    # Create sector assignments
    n_sectors = 5
    sectors = np.random.randint(0, n_sectors, size=n_stocks)
    
    # Simulate market returns (common factor)
    market_mean = 0.0005  # ~12% annually
    market_vol = 0.01  # ~16% annually
    
    # Create different market regimes (bull/bear) to introduce inefficiencies
    n_regimes = 5
    regime_length = n_days // n_regimes
    regime_means = np.random.uniform(-0.001, 0.002, n_regimes)
    regime_vols = np.random.uniform(0.005, 0.02, n_regimes)
    
    market_returns = np.zeros(n_days)
    for i in range(n_regimes):
        start_idx = i * regime_length
        end_idx = min((i + 1) * regime_length, n_days)
        regime_length_actual = end_idx - start_idx
        
        # Generate returns for this regime
        market_returns[start_idx:end_idx] = np.random.normal(
            regime_means[i], regime_vols[i], size=regime_length_actual)
    
    # Simulate sector returns with different sensitivities to market regimes
    sector_returns = np.zeros((n_days, n_sectors))
    for i in range(n_sectors):
        sector_betas = np.random.uniform(0.7, 1.3, n_regimes)
        sector_specific_vol = np.random.uniform(0.005, 0.01)
        
        for j in range(n_regimes):
            start_idx = j * regime_length
            end_idx = min((j + 1) * regime_length, n_days)
            regime_length_actual = end_idx - start_idx
            
            # Each sector has different beta to market in different regimes
            sector_returns[start_idx:end_idx, i] = (
                sector_betas[j] * market_returns[start_idx:end_idx] + 
                np.random.normal(0.0001, sector_specific_vol, size=regime_length_actual)
            )
    
    # Simulate individual stock returns with inefficiencies
    stock_returns = np.zeros((n_days, n_stocks))
    
    for i in range(n_stocks):
        # Each stock has different characteristics
        stock_quality = np.random.uniform(0, 1)  # Higher is better
        stock_vol = np.random.uniform(0.01, 0.03)
        sector_idx = sectors[i]
        
        # Generate stock returns based on sector and idiosyncratic components
        for j in range(n_regimes):
            start_idx = j * regime_length
            end_idx = min((j + 1) * regime_length, n_days)
            
            # In bull markets, higher quality stocks outperform
            # In bear markets, lower quality stocks underperform more
            if regime_means[j] > 0:  # Bull market
                quality_effect = stock_quality * 0.001  # Positive effect for high quality
            else:  # Bear market
                quality_effect = (stock_quality - 0.5) * 0.001  # Higher quality stocks less affected
            
            # Stock beta varies by regime
            stock_beta = np.random.uniform(0.7, 1.3)
            
            # Generate returns with quality effect
            stock_returns[start_idx:end_idx, i] = (
                stock_beta * sector_returns[start_idx:end_idx, sector_idx] + 
                quality_effect + 
                np.random.normal(0, stock_vol, size=end_idx - start_idx)
            )
        
        # Add occasional jumps (positive for good stocks, negative for bad stocks)
        jump_prob = 0.01
        jump_mask = np.random.random(n_days) < jump_prob
        jump_direction = 1 if stock_quality > 0.5 else -1
        jump_size = np.random.uniform(0.02, 0.05, n_days) * jump_direction
        stock_returns[jump_mask, i] += jump_size[jump_mask]
    
    # Create price data starting at random prices between 50 and 150
    initial_prices = np.random.uniform(50, 150, size=n_stocks)
    price_data = np.zeros((n_days, n_stocks))
    
    for i in range(n_stocks):
        price_data[0, i] = initial_prices[i]
        for t in range(1, n_days):
            price_data[t, i] = price_data[t-1, i] * (1 + stock_returns[t, i])
    
    # Convert to DataFrame
    prices_df = pd.DataFrame(price_data, index=dates, columns=stock_names)
    returns_df = pd.DataFrame(stock_returns, index=dates, columns=stock_names)
    
    # Create benchmark (price-weighted like DJIA)
    # Make benchmark sub-optimal by using simple equal weights
    benchmark_returns = np.zeros(n_days)
    benchmark_returns[0] = 0
    
    for t in range(1, n_days):
        # Equal-weighted benchmark (inefficient)
        benchmark_returns[t] = np.mean(stock_returns[t, :])
    
    benchmark_returns_series = pd.Series(benchmark_returns, index=dates, name='Benchmark')
    
    return prices_df, returns_df, benchmark_returns_series


def run_uett_backtest(returns_data, benchmark_returns, utility_function=UtilityFunctions.log_utility,
                     lookback_window=126, cardinality_constraint=None, target_correlation=0.9,
                     transaction_cost=0.0005):
    """
    Run UETT backtest on given data
    
    Args:
        returns_data: DataFrame with stock returns
        benchmark_returns: Series with benchmark returns
        utility_function: Utility function to use
        lookback_window: Number of days to use for weight calculation
        cardinality_constraint: Maximum number of stocks (None for full enhanced index)
        target_correlation: Target correlation with benchmark (for cardinality constraint)
        transaction_cost: Transaction cost per trade (one-way)
        
    Returns:
        Results DataFrame and performance metrics
    """
    # Initialize backtester
    backtest = UtilityEnhancedBacktest(
        utility_function=utility_function,
        rebalance_freq='M',  # Monthly rebalancing
        lookback_window=lookback_window
    )
    
    # Run backtest
    results, weights = backtest.run(
        returns_data, 
        benchmark_returns,
        cardinality_constraint=cardinality_constraint,
        target_correlation=target_correlation,
        transaction_cost=transaction_cost
    )
    
    # Calculate CUAR
    cuar = PerformanceAnalyzer.calculate_cuar(
        results['Enhanced Portfolio'].dropna(),
        results['Benchmark'].dropna(),
        utility_function
    )
    
    # Calculate performance metrics
    enhanced_metrics = PerformanceAnalyzer.calculate_metrics(
        results['Enhanced Portfolio'].dropna(),
        results['Benchmark'].dropna()
    )
    benchmark_metrics = PerformanceAnalyzer.calculate_metrics(
        results['Benchmark'].dropna()
    )
    
    # Add CUAR to enhanced metrics
    enhanced_metrics['CUAR vs Benchmark'] = cuar
    
    # Calculate Omega ratios with different thresholds
    enhanced_metrics['Omega Ratio (0%)'] = PerformanceAnalyzer.calculate_omega_ratio(
        results['Enhanced Portfolio'].dropna(), 0)
    enhanced_metrics['Omega Ratio (-0.25%)'] = PerformanceAnalyzer.calculate_omega_ratio(
        results['Enhanced Portfolio'].dropna(), -0.0025)
    enhanced_metrics['Omega Ratio (+0.25%)'] = PerformanceAnalyzer.calculate_omega_ratio(
        results['Enhanced Portfolio'].dropna(), 0.0025)
    
    benchmark_metrics['Omega Ratio (0%)'] = PerformanceAnalyzer.calculate_omega_ratio(
        results['Benchmark'].dropna(), 0)
    benchmark_metrics['Omega Ratio (-0.25%)'] = PerformanceAnalyzer.calculate_omega_ratio(
        results['Benchmark'].dropna(), -0.0025)
    benchmark_metrics['Omega Ratio (+0.25%)'] = PerformanceAnalyzer.calculate_omega_ratio(
        results['Benchmark'].dropna(), 0.0025)
    
    return results, weights, enhanced_metrics, benchmark_metrics


def main():
    """
    Main function to demonstrate the UETT technique
    """
    print("Utility Enhanced Tracking Technique (UETT) Implementation")
    print("=" * 60)
    
    print("\nGenerating simulated data with market inefficiencies...")
    # Generate simulated data with inefficiencies that can be exploited
    price_data, returns_data, benchmark_returns = generate_sample_data_with_inefficiencies(
        n_stocks=30,  # DJIA has 30 stocks
        n_days=750,   # About 3 years of trading days
        seed=42
    )
    print(f"Generated simulated data with {returns_data.shape[1]} stocks and {returns_data.shape[0]} days")
    
    # Run backtest with DARA utility function (log utility)
    print("\nRunning backtest with DARA utility function (log utility)...")
    results_dara, weights_dara, enhanced_metrics_dara, benchmark_metrics_dara = run_uett_backtest(
        returns_data,
        benchmark_returns,
        utility_function=UtilityFunctions.log_utility,
        lookback_window=126,  # ~6 months of data (shorter window to better capture regime changes)
        transaction_cost=0.0005  # 5 bps per trade
    )
    
    # Run backtest with cardinality constraint
    print("\nRunning backtest with cardinality constraint (10 stocks)...")
    results_card, weights_card, enhanced_metrics_card, benchmark_metrics_card = run_uett_backtest(
        returns_data,
        benchmark_returns,
        utility_function=UtilityFunctions.log_utility,
        lookback_window=126,
        cardinality_constraint=10,
        target_correlation=0.9,
        transaction_cost=0.0005
    )
    
    # Print performance comparison
    print("\nPerformance Metrics - Full Enhanced Index vs Benchmark:")
    PerformanceAnalyzer.print_metrics_comparison(
        enhanced_metrics_dara, 
        benchmark_metrics_dara,
        "Enhanced Index",
        "Benchmark"
    )
    
    print("\nPerformance Metrics - Cardinality Constrained Enhanced Index vs Benchmark:")
    PerformanceAnalyzer.print_metrics_comparison(
        enhanced_metrics_card, 
        benchmark_metrics_card,
        "Enhanced Index (10 stocks)",
        "Benchmark"
    )
    
    # Plot performance
    print("\nPlotting performance...")
    PerformanceAnalyzer.plot_performance(results_dara, "Full Enhanced Index")
    PerformanceAnalyzer.plot_performance(results_card, "Cardinality Constrained Enhanced Index (10 stocks)")
    
    print("\nBacktest completed successfully!")


if __name__ == "__main__":
    main()