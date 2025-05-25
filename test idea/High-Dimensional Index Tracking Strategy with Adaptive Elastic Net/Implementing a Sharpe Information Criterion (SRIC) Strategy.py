import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.decomposition import PCA
import yfinance as yf
from datetime import datetime, timedelta
import warnings
from tqdm import tqdm
warnings.filterwarnings('ignore')

class SRICPortfolioOptimizer:
    """
    A class to implement the Sharpe Ratio Information Criterion (SRIC) 
    for portfolio optimization and model selection
    """
    
    def __init__(self):
        """Initialize the optimizer"""
        self.portfolio_weights = None
        self.in_sample_sharpe = None
        self.model_dimensions = None
        self.selected_model = None
        
    def calculate_sharpe_ratio(self, returns, annualization=252):
        """
        Calculate the Sharpe ratio of a return series
        
        Parameters:
        -----------
        returns : numpy.ndarray or pandas.Series
            Array of returns
        annualization : int
            Number of periods in a year for annualization
            
        Returns:
        --------
        sharpe : float
            Annualized Sharpe ratio
        """
        if len(returns) == 0:
            return 0
        
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        if std_return == 0:
            return 0
            
        sharpe = mean_return / std_return * np.sqrt(annualization)
        return sharpe
    
    def calculate_sric(self, in_sample_sharpe, k, T):
        """
        Calculate the Sharpe Ratio Information Criterion
        
        Parameters:
        -----------
        in_sample_sharpe : float
            In-sample Sharpe ratio
        k : int
            Number of parameters/dimensions
        T : float
            Number of years of in-sample data
            
        Returns:
        --------
        sric : float
            SRIC value (estimated out-of-sample Sharpe ratio)
        """
        if in_sample_sharpe <= 0:
            return -float('inf')
        sric = in_sample_sharpe - k / (T * in_sample_sharpe)
        return sric
    
    def calculate_aic(self, in_sample_sharpe, k, T):
        """
        Calculate the Akaike Information Criterion (AIC) for Sharpe ratio
        
        Parameters:
        -----------
        in_sample_sharpe : float
            In-sample Sharpe ratio
        k : int
            Number of parameters/dimensions
        T : float
            Number of years of in-sample data
            
        Returns:
        --------
        aic : float
            AIC value (transformed to be comparable with SRIC)
        """
        # AIC in terms of mean-variance utility or squared Sharpe ratio
        aic = in_sample_sharpe**2 - 2 * (k + 1) / T
        
        # Transform to be comparable with SRIC
        # This is an approximation - as the paper notes, SRIC and AIC converge for large T
        aic_transformed = np.sqrt(max(0, aic))
        
        return aic_transformed
    
    def optimize_portfolio(self, returns, method='markowitz'):
        """
        Optimize portfolio weights to maximize Sharpe ratio
        
        Parameters:
        -----------
        returns : numpy.ndarray
            Array of asset returns
        method : str
            Optimization method ('markowitz' or 'equal_weight')
            
        Returns:
        --------
        weights : numpy.ndarray
            Optimal portfolio weights
        sharpe : float
            In-sample Sharpe ratio
        """
        n_assets = returns.shape[1]
        
        if method == 'equal_weight':
            weights = np.ones(n_assets) / n_assets
        else:  # markowitz
            # Calculate mean returns and covariance
            mean_returns = np.mean(returns, axis=0)
            cov_matrix = np.cov(returns, rowvar=False)
            
            # Handle potential singularity
            try:
                # Calculate optimal weights (mean-variance optimization)
                weights = np.dot(np.linalg.inv(cov_matrix), mean_returns)
                # Normalize weights to sum to 1
                weights = weights / np.sum(np.abs(weights))
            except np.linalg.LinAlgError:
                # Fallback to equal weights if covariance matrix is singular
                weights = np.ones(n_assets) / n_assets
        
        # Calculate portfolio returns
        portfolio_returns = np.dot(returns, weights)
        
        # Calculate in-sample Sharpe ratio
        sharpe = self.calculate_sharpe_ratio(portfolio_returns)
        
        return weights, sharpe
    
    def select_model_dimension(self, returns, max_dim, T_years=None, method='sric'):
        """
        Select the optimal model dimension using SRIC or AIC
        
        Parameters:
        -----------
        returns : numpy.ndarray
            Array of asset returns
        max_dim : int
            Maximum model dimension to consider
        T_years : float or None
            Number of years of in-sample data (if None, inferred from returns)
        method : str
            Selection criterion ('sric' or 'aic')
            
        Returns:
        --------
        optimal_dim : int
            Optimal model dimension
        sharpes : list
            In-sample Sharpe ratios for each dimension
        criteria : list
            Selection criteria (SRIC or AIC) for each dimension
        """
        n_samples = returns.shape[0]
        
        # Infer T if not provided (assuming daily returns)
        if T_years is None:
            T_years = n_samples / 252
        
        # Store results for each dimension
        sharpes = []
        criteria = []
        weights_by_dim = []
        
        # Generate principal components to reduce dimensionality
        max_components = min(max_dim, returns.shape[1], returns.shape[0])
        pca = PCA(n_components=max_components)
        pca_returns = pca.fit_transform(returns)
        
        # Evaluate models of different dimensions
        for k in range(1, max_components + 1):
            # Use first k principal components
            reduced_returns = pca_returns[:, :k]
            
            # Optimize portfolio on reduced dimension
            weights, sharpe = self.optimize_portfolio(reduced_returns)
            sharpes.append(sharpe)
            
            # Calculate criterion (SRIC or AIC)
            if method == 'sric':
                criterion = self.calculate_sric(sharpe, k-1, T_years)  # k-1 because first component is like equal weight
            else:  # aic
                criterion = self.calculate_aic(sharpe, k-1, T_years)
                
            criteria.append(criterion)
            weights_by_dim.append(weights)
        
        # Select optimal dimension
        optimal_dim = np.argmax(criteria) + 1
        
        self.in_sample_sharpe = sharpes
        self.model_dimensions = list(range(1, max_components + 1))
        self.selected_model = optimal_dim
        self.portfolio_weights = weights_by_dim[optimal_dim - 1]
        
        return optimal_dim, sharpes, criteria
    
    def simulate_out_of_sample_performance(self, returns_in, returns_out, method='sric', max_dim=None):
        """
        Simulate out-of-sample performance after model selection
        
        Parameters:
        -----------
        returns_in : numpy.ndarray
            In-sample returns
        returns_out : numpy.ndarray
            Out-of-sample returns
        method : str
            Selection criterion ('sric', 'aic', 'markowitz', 'equal_weight')
        max_dim : int or None
            Maximum model dimension (if None, use all dimensions)
            
        Returns:
        --------
        out_of_sample_sharpe : float
            Out-of-sample Sharpe ratio
        selected_dim : int
            Selected model dimension
        """
        if max_dim is None:
            max_dim = min(returns_in.shape[1], returns_in.shape[0])
        else:
            max_dim = min(max_dim, returns_in.shape[1], returns_in.shape[0])
        
        T_years = returns_in.shape[0] / 252  # Assuming daily returns
        
        if method in ['sric', 'aic']:
            # Select model dimension using SRIC or AIC
            selected_dim, _, _ = self.select_model_dimension(returns_in, max_dim, T_years, method)
            
            # Apply PCA to in-sample and out-of-sample returns
            pca = PCA(n_components=max_dim)
            pca.fit(returns_in)
            
            returns_in_pca = pca.transform(returns_in)
            returns_out_pca = pca.transform(returns_out)
            
            # Optimize portfolio on selected dimension of in-sample PCA returns
            weights, _ = self.optimize_portfolio(returns_in_pca[:, :selected_dim])
            
            # Apply weights to out-of-sample PCA returns
            out_returns = np.dot(returns_out_pca[:, :selected_dim], weights)
            
        elif method == 'markowitz':
            # Full Markowitz portfolio without dimensional reduction
            weights, _ = self.optimize_portfolio(returns_in)
            out_returns = np.dot(returns_out, weights)
            selected_dim = returns_in.shape[1]
            
        elif method == 'equal_weight':
            # Equal weight portfolio
            n_assets = returns_in.shape[1]
            weights = np.ones(n_assets) / n_assets
            out_returns = np.dot(returns_out, weights)
            selected_dim = 1
        
        # Calculate out-of-sample Sharpe ratio
        out_of_sample_sharpe = self.calculate_sharpe_ratio(out_returns)
        
        return out_of_sample_sharpe, selected_dim


class SRICTradingStrategy:
    """
    A class to implement trading strategies using SRIC for model selection
    """
    
    def __init__(self, optimizer=None):
        """
        Initialize the trading strategy
        
        Parameters:
        -----------
        optimizer : SRICPortfolioOptimizer or None
            Optimizer to use for model selection (if None, a new one is created)
        """
        if optimizer is None:
            self.optimizer = SRICPortfolioOptimizer()
        else:
            self.optimizer = optimizer
        
        self.portfolio_history = []
        self.performance_history = {}
        self.selected_dims = []
    
    def download_data(self, tickers, start_date, end_date):
        """
        Download historical price data for a list of tickers
        
        Parameters:
        -----------
        tickers : list
            List of ticker symbols
        start_date : str
            Start date in 'YYYY-MM-DD' format
        end_date : str
            End date in 'YYYY-MM-DD' format
            
        Returns:
        --------
        prices : pandas.DataFrame
            DataFrame of adjusted close prices
        """
        print(f"Downloading data for {len(tickers)} assets...")
        try:
            prices = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
            
            # Handle single ticker case
            if isinstance(prices, pd.Series):
                prices = pd.DataFrame(prices, columns=[tickers[0]])
            
            # Handle missing values
            prices = prices.dropna(axis=1, how='all')
            prices = prices.fillna(method='ffill')
            
            return prices
        except Exception as e:
            print(f"Error downloading data: {e}")
            return pd.DataFrame()
    
    def calculate_returns(self, prices, period='daily'):
        """
        Calculate returns from price data
        
        Parameters:
        -----------
        prices : pandas.DataFrame
            DataFrame of prices
        period : str
            Return calculation period ('daily', 'weekly', or 'monthly')
            
        Returns:
        --------
        returns : pandas.DataFrame
            DataFrame of returns
        """
        if period == 'daily':
            returns = prices.pct_change().dropna()
        elif period == 'weekly':
            returns = prices.resample('W').last().pct_change().dropna()
        elif period == 'monthly':
            returns = prices.resample('M').last().pct_change().dropna()
        else:
            raise ValueError("period must be 'daily', 'weekly', or 'monthly'")
        
        return returns
    
    def create_features(self, prices, feature_list=None):
        """
        Create feature matrix from price data
        
        Parameters:
        -----------
        prices : pandas.DataFrame
            DataFrame of prices
        feature_list : list or None
            List of features to create (if None, uses default features)
            
        Returns:
        --------
        features : pandas.DataFrame
            DataFrame of features
        """
        if feature_list is None:
            # Default features: momentum at different horizons
            feature_list = ['mom_1m', 'mom_3m', 'mom_6m', 'mom_12m', 'vol_1m', 'vol_3m']
        
        features = pd.DataFrame(index=prices.index)
        
        # Calculate features for each asset
        for ticker in prices.columns:
            # Momentum features
            if 'mom_1m' in feature_list:
                features[f'{ticker}_mom_1m'] = prices[ticker].pct_change(20)
            if 'mom_3m' in feature_list:
                features[f'{ticker}_mom_3m'] = prices[ticker].pct_change(60)
            if 'mom_6m' in feature_list:
                features[f'{ticker}_mom_6m'] = prices[ticker].pct_change(125)
            if 'mom_12m' in feature_list:
                features[f'{ticker}_mom_12m'] = prices[ticker].pct_change(252)
            
            # Volatility features
            if 'vol_1m' in feature_list:
                features[f'{ticker}_vol_1m'] = prices[ticker].pct_change().rolling(20).std()
            if 'vol_3m' in feature_list:
                features[f'{ticker}_vol_3m'] = prices[ticker].pct_change().rolling(60).std()
            
            # Mean reversion features
            if 'rev_1m' in feature_list:
                features[f'{ticker}_rev_1m'] = -prices[ticker].pct_change(20)
            
            # Technical indicators
            if 'rsi' in feature_list:
                # Relative Strength Index
                delta = prices[ticker].diff()
                up = delta.clip(lower=0)
                down = -delta.clip(upper=0)
                ema_up = up.ewm(com=13, adjust=False).mean()
                ema_down = down.ewm(com=13, adjust=False).mean()
                rs = ema_up / ema_down
                features[f'{ticker}_rsi'] = 100 - (100 / (1 + rs))
            
            # Moving averages
            if 'ma_cross' in feature_list:
                ma_short = prices[ticker].rolling(50).mean()
                ma_long = prices[ticker].rolling(200).mean()
                features[f'{ticker}_ma_cross'] = (ma_short > ma_long).astype(int)
        
        # Fill NaN values with 0
        features = features.fillna(0)
        
        return features
    
    def backtest_strategy(self, prices, lookback_window=252, rebalance_freq='monthly', 
                        max_dim=10, method='sric', feature_list=None, use_features=True):
        """
        Backtest a trading strategy using SRIC for model selection
        
        Parameters:
        -----------
        prices : pandas.DataFrame
            DataFrame of prices
        lookback_window : int
            Number of trading days to use for model training
        rebalance_freq : str
            Rebalancing frequency ('daily', 'weekly', or 'monthly')
        max_dim : int
            Maximum model dimension to consider
        method : str
            Selection method ('sric', 'aic', 'markowitz', or 'equal_weight')
        feature_list : list or None
            List of features to create
        use_features : bool
            Whether to use features instead of returns for optimization
            
        Returns:
        --------
        performance : dict
            Dictionary of performance metrics
        portfolio_df : pandas.DataFrame
            DataFrame with portfolio performance
        weights_history : list
            History of portfolio weights
        selected_dims : list
            History of selected dimensions
        """
        # Calculate returns
        returns = self.calculate_returns(prices, period='daily')
        
        # Create features if requested
        if use_features:
            features = self.create_features(prices, feature_list)
            # Align dates between features and returns
            common_dates = features.index.intersection(returns.index)
            features = features.loc[common_dates]
            returns = returns.loc[common_dates]
            optimization_data = features.values
        else:
            optimization_data = returns.values
        
        # Convert to numpy arrays for faster computation
        returns_array = returns.values
        
        # Determine rebalancing dates
        if rebalance_freq == 'daily':
            rebalance_dates = returns.index[lookback_window:]
        elif rebalance_freq == 'weekly':
            weekly_dates = returns.resample('W').last().index
            rebalance_dates = returns.index[returns.index.isin(weekly_dates) & 
                                          (returns.index >= returns.index[lookback_window])]
        elif rebalance_freq == 'monthly':
            monthly_dates = returns.resample('M').last().index
            rebalance_dates = returns.index[returns.index.isin(monthly_dates) & 
                                          (returns.index >= returns.index[lookback_window])]
        else:
            raise ValueError("rebalance_freq must be 'daily', 'weekly', or 'monthly'")
        
        # Initialize portfolio
        portfolio_values = [1.0]
        portfolio_returns = []
        weights_history = []
        selected_dims = []
        
        print(f"Starting backtest with {len(rebalance_dates)} rebalancing periods...")
        
        # Use tqdm for progress bar
        for i, rebalance_date in enumerate(tqdm(rebalance_dates)):
            # Get index for current date
            current_idx = returns.index.get_loc(rebalance_date)
            
            # Define training window
            train_start_idx = max(0, current_idx - lookback_window)
            train_end_idx = current_idx
            
            # Get training data
            train_optimization_data = optimization_data[train_start_idx:train_end_idx]
            train_returns = returns_array[train_start_idx:train_end_idx]
            
            # Calculate years of training data
            T_years = (train_end_idx - train_start_idx) / 252
            
            try:
                # Select model dimension and optimize portfolio
                if method in ['sric', 'aic']:
                    selected_dim, _, _ = self.optimizer.select_model_dimension(
                        train_optimization_data, max_dim, T_years, method=method
                    )
                    
                    # Apply PCA to reduce dimensionality
                    pca = PCA(n_components=min(max_dim, train_optimization_data.shape[1], train_optimization_data.shape[0]))
                    pca.fit(train_optimization_data)
                    train_optimization_pca = pca.transform(train_optimization_data)
                    
                    # Optimize portfolio on reduced dimension
                    weights_pca, _ = self.optimizer.optimize_portfolio(train_optimization_pca[:, :selected_dim])
                    
                    # Transform back to asset weights
                    # For features, this is an approximation
                    if use_features:
                        # Create a mapping from PCA components to original assets
                        # Use the returns to judge how well each asset correlates with each component
                        pca_returns = pca.transform(train_optimization_data)
                        weights = np.zeros(returns.shape[1])
                        
                        for j in range(selected_dim):
                            # Calculate correlation between returns and PCA component
                            component_returns = pca_returns[:, j]
                            for asset_idx in range(returns.shape[1]):
                                asset_returns = train_returns[:, asset_idx]
                                # Use correlation as a weight
                                weights[asset_idx] += weights_pca[j] * np.corrcoef(component_returns, asset_returns)[0, 1]
                    else:
                        # For returns-based optimization, we can use PCA loadings directly
                        weights = np.zeros(returns.shape[1])
                        for j in range(selected_dim):
                            weights += weights_pca[j] * pca.components_[j]
                    
                elif method == 'markowitz':
                    weights, _ = self.optimizer.optimize_portfolio(train_returns)
                    selected_dim = returns.shape[1]
                    
                elif method == 'equal_weight':
                    weights = np.ones(returns.shape[1]) / returns.shape[1]
                    selected_dim = 1
                
                # Normalize weights to sum to 1
                weights = weights / np.sum(np.abs(weights))
                
                # Store weights and selected dimension
                weights_history.append(weights)
                selected_dims.append(selected_dim)
                
                # Calculate portfolio return until next rebalancing
                if i < len(rebalance_dates) - 1:
                    next_rebalance_idx = returns.index.get_loc(rebalance_dates[i + 1])
                    period_returns = returns_array[current_idx:next_rebalance_idx]
                else:
                    period_returns = returns_array[current_idx:]
                
                # Calculate portfolio returns
                period_portfolio_returns = np.dot(period_returns, weights)
                portfolio_returns.extend(period_portfolio_returns)
                
                # Update portfolio value
                for ret in period_portfolio_returns:
                    portfolio_values.append(portfolio_values[-1] * (1 + ret))
                
            except Exception as e:
                print(f"Error at date {rebalance_date}: {e}")
                # Use previous weights or equal weights if no previous weights
                if weights_history:
                    weights = weights_history[-1]
                else:
                    weights = np.ones(returns.shape[1]) / returns.shape[1]
                weights_history.append(weights)
                selected_dims.append(1 if not selected_dims else selected_dims[-1])
                
                # Handle returns similarly
                if i < len(rebalance_dates) - 1:
                    next_rebalance_idx = returns.index.get_loc(rebalance_dates[i + 1])
                    period_returns = returns_array[current_idx:next_rebalance_idx]
                else:
                    period_returns = returns_array[current_idx:]
                
                period_portfolio_returns = np.dot(period_returns, weights)
                portfolio_returns.extend(period_portfolio_returns)
                
                for ret in period_portfolio_returns:
                    portfolio_values.append(portfolio_values[-1] * (1 + ret))
        
        # Create DataFrame of portfolio values
        portfolio_dates = returns.index[lookback_window:lookback_window+len(portfolio_returns)]
        portfolio_df = pd.DataFrame({
            'Value': portfolio_values[:len(portfolio_dates)],
            'Return': [0] + portfolio_returns[:len(portfolio_dates)-1]
        }, index=portfolio_dates)
        
        # Calculate performance metrics
        performance = self.calculate_performance_metrics(portfolio_df)
        
        # Store portfolio history and selected dimensions
        self.portfolio_history.append(portfolio_df)
        self.performance_history[method] = performance
        self.selected_dims = selected_dims
        
        return performance, portfolio_df, weights_history, selected_dims
    
    def calculate_performance_metrics(self, portfolio_df):
        """
        Calculate performance metrics for a portfolio
        
        Parameters:
        -----------
        portfolio_df : pandas.DataFrame
            DataFrame of portfolio values and returns
            
        Returns:
        --------
        metrics : dict
            Dictionary of performance metrics
        """
        returns = portfolio_df['Return'].values
        
        # Calculate metrics
        total_return = portfolio_df['Value'].iloc[-1] / portfolio_df['Value'].iloc[0] - 1
        annualized_return = (1 + total_return) ** (252 / len(returns)) - 1
        
        volatility = np.std(returns) * np.sqrt(252)
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
        
        # Calculate drawdown
        drawdown = 1 - portfolio_df['Value'] / portfolio_df['Value'].cummax()
        max_drawdown = drawdown.max()
        
        # Calculate win rate
        win_rate = np.sum(returns > 0) / len(returns)
        
        # Calculate Sortino ratio (downside risk)
        downside_returns = returns[returns < 0]
        downside_deviation = np.std(downside_returns) * np.sqrt(252) if len(downside_returns) > 0 else 1e-6
        sortino_ratio = annualized_return / downside_deviation
        
        # Calculate Calmar ratio (return to max drawdown)
        calmar_ratio = annualized_return / max_drawdown if max_drawdown > 0 else float('inf')
        
        metrics = {
            'Total Return': total_return,
            'Annualized Return': annualized_return,
            'Annualized Volatility': volatility,
            'Sharpe Ratio': sharpe_ratio,
            'Sortino Ratio': sortino_ratio,
            'Calmar Ratio': calmar_ratio,
            'Max Drawdown': max_drawdown,
            'Win Rate': win_rate
        }
        
        return metrics
    
    def plot_performance(self, benchmark_df=None):
        """
        Plot performance of different strategies
        
        Parameters:
        -----------
        benchmark_df : pandas.DataFrame or None
            DataFrame of benchmark values (if None, no benchmark is plotted)
        """
        if not self.portfolio_history:
            print("No portfolio history to plot.")
            return
        
        plt.figure(figsize=(12, 8))
        
        # Plot each strategy
        for i, portfolio_df in enumerate(self.portfolio_history):
            label = f"Strategy {i+1}" if i >= len(self.performance_history) else list(self.performance_history.keys())[i]
            sharpe = self.performance_history.get(label, {}).get('Sharpe Ratio', 0)
            plt.plot(portfolio_df.index, portfolio_df['Value'], label=f"{label} (Sharpe: {sharpe:.2f})")
        
        # Plot benchmark if provided
        if benchmark_df is not None:
            benchmark_returns = benchmark_df['Return'].values
            benchmark_sharpe = np.mean(benchmark_returns) / np.std(benchmark_returns) * np.sqrt(252)
            plt.plot(benchmark_df.index, benchmark_df['Value'], 
                     label=f'Benchmark (Sharpe: {benchmark_sharpe:.2f})', 
                     color='black', linestyle='--')
        
        plt.title('Portfolio Performance')
        plt.xlabel('Date')
        plt.ylabel('Portfolio Value')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()
        
        # Plot selected dimensions over time if available
        if hasattr(self, 'selected_dims') and self.selected_dims:
            plt.figure(figsize=(12, 4))
            plt.plot(self.selected_dims)
            plt.title('Selected Model Dimensions Over Time')
            plt.xlabel('Rebalancing Period')
            plt.ylabel('Selected Dimension')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
    
    def compare_strategies(self):
        """
        Compare performance of different strategies
        """
        if not self.performance_history:
            print("No strategies to compare.")
            return
        
        # Create DataFrame with performance metrics
        metrics_df = pd.DataFrame.from_dict(self.performance_history, orient='index')
        
        # Print comparison table
        print("\nStrategy Performance Comparison:")
        print("-" * 100)
        pd.set_option('display.float_format', '{:.4f}'.format)
        print(metrics_df)
        print("-" * 100)
        
        # Plot key metrics
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        metrics_df['Sharpe Ratio'].plot(kind='bar', ax=axes[0, 0], title='Sharpe Ratio')
        axes[0, 0].grid(axis='y', alpha=0.3)
        
        metrics_df['Annualized Return'].plot(kind='bar', ax=axes[0, 1], title='Annualized Return')
        axes[0, 1].grid(axis='y', alpha=0.3)
        
        metrics_df['Max Drawdown'].plot(kind='bar', ax=axes[1, 0], title='Max Drawdown')
        axes[1, 0].grid(axis='y', alpha=0.3)
        
        metrics_df['Win Rate'].plot(kind='bar', ax=axes[1, 1], title='Win Rate')
        axes[1, 1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.show()


def generate_simulated_returns(n_samples=1260, n_assets=20, true_dim=5, 
                              daily_volatility=0.01, annual_sharpe=0.5, 
                              correlation=0.2, random_seed=None):
    """
    Generate simulated returns with controlled properties
    
    Parameters:
    -----------
    n_samples : int
        Number of samples (days)
    n_assets : int
        Number of assets
    true_dim : int
        Dimension of the true model (number of meaningful factors)
    daily_volatility : float
        Daily volatility of returns
    annual_sharpe : float
        Annual Sharpe ratio of true factors
    correlation : float
        Base correlation between assets
    random_seed : int or None
        Random seed for reproducibility
        
    Returns:
    --------
    returns : numpy.ndarray
        Simulated asset returns
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # Create correlation matrix
    corr_matrix = np.ones((n_assets, n_assets)) * correlation
    np.fill_diagonal(corr_matrix, 1.0)
    
    # Create covariance matrix from correlation and volatility
    cov_matrix = corr_matrix * (daily_volatility ** 2)
    
    # Generate random returns
    noise_returns = np.random.multivariate_normal(
        mean=np.zeros(n_assets), 
        cov=cov_matrix, 
        size=n_samples
    )
    
    # Generate factor structure
    factors = np.random.randn(true_dim, n_assets)
    
    # Normalize factors
    for i in range(true_dim):
        factors[i] = factors[i] / np.sqrt(np.sum(factors[i]**2))
    
    # Generate factor returns
    factor_returns = np.random.randn(n_samples, true_dim)
    
    # Add mean to factor returns to achieve desired Sharpe ratio
    daily_sharpe = annual_sharpe / np.sqrt(252)
    factor_returns = factor_returns + daily_sharpe * daily_volatility
    
    # Combine factor returns with loadings
    signal_returns = np.zeros((n_samples, n_assets))
    for i in range(true_dim):
        signal_returns += np.outer(factor_returns[:, i], factors[i])
    
    # Combine signal and noise
    returns = signal_returns + noise_returns
    
    return returns


def split_train_test(returns, train_ratio=0.6):
    """
    Split returns into training and test sets
    
    Parameters:
    -----------
    returns : numpy.ndarray
        Asset returns
    train_ratio : float
        Ratio of data to use for training
        
    Returns:
    --------
    train_returns, test_returns : numpy.ndarray, numpy.ndarray
        Training and test returns
    """
    n_samples = returns.shape[0]
    train_size = int(n_samples * train_ratio)
    
    train_returns = returns[:train_size]
    test_returns = returns[train_size:]
    
    return train_returns, test_returns


def analyze_sharpe_indifference_curves():
    """
    Visualize Sharpe Indifference Curves as shown in the paper
    (combinations of in-sample Sharpe and number of parameters
    that lead to the same estimated out-of-sample Sharpe)
    """
    # Parameters
    T = 10  # years of in-sample data
    k_values = np.arange(1, 51)  # number of parameters
    target_oos_sharpes = [0.25, 0.5, 0.75, 1.0]  # target out-of-sample Sharpe ratios
    
    plt.figure(figsize=(10, 7))
    
    for target_oos_sharpe in target_oos_sharpes:
        # Calculate required in-sample Sharpe ratios
        # From SRIC formula: oos_sharpe = is_sharpe - k/(T*is_sharpe)
        # Solving for is_sharpe: is_sharpe^2 - oos_sharpe*is_sharpe - k/T = 0
        # Using quadratic formula
        in_sample_sharpes = [(oos_sharpe + np.sqrt(oos_sharpe**2 + 4*k/T))/2 
                             for k, oos_sharpe in zip(k_values, [target_oos_sharpe]*len(k_values))]
        
        plt.plot(k_values, in_sample_sharpes, label=f'Out-of-Sample Sharpe = {target_oos_sharpe}')
    
    plt.title(f'Sharpe Indifference Curves (T = {T} years)')
    plt.xlabel('Number of Parameters (k)')
    plt.ylabel('Required In-Sample Sharpe Ratio')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def run_simulation(n_trials=50, n_samples=1260, n_assets=20, true_dims=[5, 10, 15],
                  train_ratio=0.6, methods=['sric', 'aic', 'markowitz', 'equal_weight']):
    """
    Run simulations to compare different model selection methods
    
    Parameters:
    -----------
    n_trials : int
        Number of simulation trials
    n_samples : int
        Number of samples per trial
    n_assets : int
        Number of assets
    true_dims : list
        List of true model dimensions to test
    train_ratio : float
        Ratio of data to use for training
    methods : list
        List of methods to test
        
    Returns:
    --------
    results : pandas.DataFrame
        Simulation results
    """
    results = []
    
    for true_dim in true_dims:
        print(f"Running simulations for true dimension {true_dim}...")
        
        for trial in tqdm(range(n_trials)):
            # Generate simulated returns
            returns = generate_simulated_returns(
                n_samples=n_samples,
                n_assets=n_assets,
                true_dim=true_dim,
                random_seed=trial
            )
            
            # Split into training and test sets
            train_returns, test_returns = split_train_test(returns, train_ratio)
            
            # Optimize portfolio using different methods
            optimizer = SRICPortfolioOptimizer()
            
            for method in methods:
                # Calculate out-of-sample performance
                oos_sharpe, selected_dim = optimizer.simulate_out_of_sample_performance(
                    train_returns, test_returns, method=method, max_dim=n_assets
                )
                
                # Store results
                results.append({
                    'Trial': trial,
                    'Method': method,
                    'True Dimension': true_dim,
                    'Selected Dimension': selected_dim,
                    'Out-of-Sample Sharpe': oos_sharpe,
                    'Train Samples': train_returns.shape[0],
                    'Test Samples': test_returns.shape[0]
                })
    
    return pd.DataFrame(results)


def plot_simulation_results(results):
    """
    Plot simulation results
    
    Parameters:
    -----------
    results : pandas.DataFrame
        Simulation results
    """
    plt.figure(figsize=(12, 8))
    
    # Plot out-of-sample Sharpe ratio by true dimension and method
    sns.barplot(
        data=results, 
        x='True Dimension', 
        y='Out-of-Sample Sharpe', 
        hue='Method',
        ci=95
    )
    
    plt.title('Out-of-Sample Sharpe Ratio by Method and True Dimension')
    plt.xlabel('True Model Dimension')
    plt.ylabel('Out-of-Sample Sharpe Ratio')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Plot selected dimension by true dimension and method
    plt.figure(figsize=(12, 8))
    
    # Filter out equal_weight which doesn't really select dimensions
    dimension_results = results[results['Method'] != 'equal_weight']
    
    sns.barplot(
        data=dimension_results, 
        x='True Dimension', 
        y='Selected Dimension', 
        hue='Method',
        ci=95
    )
    
    # Add line showing true dimension
    true_dims = sorted(dimension_results['True Dimension'].unique())
    plt.plot(range(len(true_dims)), true_dims, 'k--', label='True Dimension')
    
    plt.title('Selected Model Dimension by Method and True Dimension')
    plt.xlabel('True Model Dimension')
    plt.ylabel('Selected Model Dimension')
    plt.grid(axis='y', alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def simple_portfolio_optimization_example():
    """
    A simple example of portfolio optimization using SRIC
    """
    # Generate simulated returns
    n_samples = 1260  # 5 years of daily data
    n_assets = 20
    true_dim = 5
    
    returns = generate_simulated_returns(
        n_samples=n_samples,
        n_assets=n_assets,
        true_dim=true_dim
    )
    
    # Split into training and test sets
    train_returns, test_returns = split_train_test(returns, train_ratio=0.6)
    
    # Initialize optimizer
    optimizer = SRICPortfolioOptimizer()
    
    # Select model dimension using SRIC
    max_dim = min(n_assets, train_returns.shape[0])
    T_years = train_returns.shape[0] / 252
    
    optimal_dim, sharpes, sric_values = optimizer.select_model_dimension(
        train_returns, max_dim, T_years, method='sric'
    )
    
    # For comparison, also calculate AIC values
    _, _, aic_values = optimizer.select_model_dimension(
        train_returns, max_dim, T_years, method='aic'
    )
    
    # Plot results
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(range(1, len(sharpes)+1), sharpes, 'b-', label='In-Sample Sharpe')
    plt.axvline(x=true_dim, color='r', linestyle='--', label=f'True Dimension ({true_dim})')
    plt.title('In-Sample Sharpe Ratio vs. Model Dimension')
    plt.xlabel('Model Dimension')
    plt.ylabel('In-Sample Sharpe Ratio')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.subplot(2, 1, 2)
    plt.plot(range(1, len(sric_values)+1), sric_values, 'g-', label='SRIC')
    plt.plot(range(1, len(aic_values)+1), aic_values, 'm-', label='AIC')
    plt.axvline(x=optimal_dim, color='g', linestyle='--', 
                label=f'SRIC Selected Dimension ({optimal_dim})')
    plt.axvline(x=np.argmax(aic_values)+1, color='m', linestyle='--', 
                label=f'AIC Selected Dimension ({np.argmax(aic_values)+1})')
    plt.axvline(x=true_dim, color='r', linestyle='--', label=f'True Dimension ({true_dim})')
    plt.title('Model Selection Criteria vs. Model Dimension')
    plt.xlabel('Model Dimension')
    plt.ylabel('Selection Criterion Value')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Evaluate different methods
    methods = ['sric', 'aic', 'markowitz', 'equal_weight']
    performance = {}
    
    for method in methods:
        oos_sharpe, selected_dim = optimizer.simulate_out_of_sample_performance(
            train_returns, test_returns, method=method, max_dim=min(n_assets, train_returns.shape[0])
        )
        performance[method] = (oos_sharpe, selected_dim)
    
    # Print performance comparison
    print("\nOut-of-Sample Performance Comparison:")
    print("-" * 60)
    print(f"{'Method':<15} {'Out-of-Sample Sharpe':<25} {'Selected Dimension':<20}")
    print("-" * 60)
    for method, (sharpe, dim) in performance.items():
        print(f"{method:<15} {sharpe:<25.4f} {dim:<20}")
    print("-" * 60)
    
    # Calculate the true out-of-sample Sharpe for the true model dimension
    pca = PCA(n_components=min(n_assets, train_returns.shape[0]))
    pca.fit(train_returns)
    
    train_returns_pca = pca.transform(train_returns)
    test_returns_pca = pca.transform(test_returns)
    
    weights, _ = optimizer.optimize_portfolio(train_returns_pca[:, :true_dim])
    true_model_returns = np.dot(test_returns_pca[:, :true_dim], weights)
    true_model_sharpe = optimizer.calculate_sharpe_ratio(true_model_returns)
    
    print(f"\nTrue Model (dimension={true_dim}) Out-of-Sample Sharpe: {true_model_sharpe:.4f}")


def run_industry_portfolio_example():
    """
    Run a practical example using industry ETFs
    """
    # Define industry ETFs
    industry_etfs = [
        'XLK',  # Technology
        'XLF',  # Financials
        'XLE',  # Energy
        'XLV',  # Healthcare
        'XLI',  # Industrials
        'XLP',  # Consumer Staples
        'XLY',  # Consumer Discretionary
        'XLB',  # Materials
        'XLU',  # Utilities
        'XLRE'  # Real Estate
    ]
    
    # Define start and end dates
    start_date = '2010-01-01'
    end_date = '2023-01-01'
    
    # Initialize trading strategy
    strategy = SRICTradingStrategy()
    
    # Download data
    prices = strategy.download_data(industry_etfs, start_date, end_date)
    
    if prices.empty:
        print("Error downloading data. Exiting example.")
        return
    
    # Download benchmark (S&P 500)
    spy_prices = strategy.download_data(['SPY'], start_date, end_date)
    
    if spy_prices.empty:
        print("Error downloading benchmark data. Continuing without benchmark.")
        spy_portfolio = None
    else:
        spy_returns = strategy.calculate_returns(spy_prices)
        spy_portfolio = pd.DataFrame({
            'Value': (1 + spy_returns).cumprod(),
            'Return': spy_returns.values.flatten()
        }, index=spy_returns.index)
    
    # Backtest strategies
    print("\nTesting SRIC strategy...")
    _, portfolio_sric, _, _ = strategy.backtest_strategy(
        prices, lookback_window=252, rebalance_freq='monthly', 
        max_dim=10, method='sric'
    )
    
    print("\nTesting AIC strategy...")
    _, portfolio_aic, _, _ = strategy.backtest_strategy(
        prices, lookback_window=252, rebalance_freq='monthly', 
        max_dim=10, method='aic'
    )
    
    print("\nTesting Markowitz strategy...")
    _, portfolio_markowitz, _, _ = strategy.backtest_strategy(
        prices, lookback_window=252, rebalance_freq='monthly', 
        max_dim=10, method='markowitz'
    )
    
    print("\nTesting Equal Weight strategy...")
    _, portfolio_equal, _, _ = strategy.backtest_strategy(
        prices, lookback_window=252, rebalance_freq='monthly', 
        max_dim=10, method='equal_weight'
    )
    
    # Compare strategies
    strategy.compare_strategies()
    
    # Plot performance
    strategy.plot_performance(spy_portfolio)


def run_trading_strategy_with_signals():
    """
    Run a trading strategy using technical and fundamental signals
    """
    # Define tickers (e.g., S&P 500 sectors plus some individual stocks)
    tickers = [
        # Sector ETFs
        'XLK', 'XLF', 'XLE', 'XLV', 'XLI', 'XLP', 'XLY', 'XLB', 'XLU', 'XLRE',
        # Large cap stocks
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'BRK-B', 'JNJ', 'JPM', 'V',
        # Mid cap stocks
        'AMD', 'SQ', 'ROKU', 'ETSY', 'DKNG'
    ]
    
    # Define start and end dates
    start_date = '2015-01-01'
    end_date = '2023-01-01'
    
    # Initialize trading strategy
    strategy = SRICTradingStrategy()
    
    # Download data
    prices = strategy.download_data(tickers, start_date, end_date)
    
    if prices.empty:
        print("Error downloading data. Exiting example.")
        return
    
    # Handle any missing tickers
    tickers = prices.columns.tolist()
    
    # Download benchmark (S&P 500)
    spy_prices = strategy.download_data(['SPY'], start_date, end_date)
    
    if spy_prices.empty:
        print("Error downloading benchmark data. Continuing without benchmark.")
        spy_portfolio = None
    else:
        spy_returns = strategy.calculate_returns(spy_prices)
        spy_portfolio = pd.DataFrame({
            'Value': (1 + spy_returns).cumprod(),
            'Return': spy_returns.values.flatten()
        }, index=spy_returns.index)
    
    # Define feature list (trading signals)
    feature_list = ['mom_1m', 'mom_3m', 'mom_6m', 'mom_12m', 'vol_1m', 'vol_3m', 'rsi', 'rev_1m']
    
    # Backtest strategies
    print("\nTesting SRIC strategy with signals...")
    _, portfolio_sric, _, _ = strategy.backtest_strategy(
        prices, lookback_window=252, rebalance_freq='monthly', 
        max_dim=15, method='sric', feature_list=feature_list, use_features=True
    )
    
    print("\nTesting AIC strategy with signals...")
    _, portfolio_aic, _, _ = strategy.backtest_strategy(
        prices, lookback_window=252, rebalance_freq='monthly', 
        max_dim=15, method='aic', feature_list=feature_list, use_features=True
    )
    
    # Compare strategies
    strategy.compare_strategies()
    
    # Plot performance
    strategy.plot_performance(spy_portfolio)


def implement_etf_rotation_strategy():
    """
    Implement a sector rotation strategy using SRIC for signal selection
    """
    # Define sector ETFs
    sector_etfs = [
        'XLK',  # Technology
        'XLF',  # Financials
        'XLE',  # Energy
        'XLV',  # Healthcare
        'XLI',  # Industrials
        'XLP',  # Consumer Staples
        'XLY',  # Consumer Discretionary
        'XLB',  # Materials
        'XLU',  # Utilities
        'XLRE'  # Real Estate
    ]
    
    # Download data for the last 10 years
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=3650)).strftime('%Y-%m-%d')
    
    strategy = SRICTradingStrategy()
    prices = strategy.download_data(sector_etfs, start_date, end_date)
    
    if prices.empty:
        print("Error downloading data. Exiting strategy.")
        return None
    
    # Handle any missing tickers
    sector_etfs = prices.columns.tolist()
    
    # Create trading signals
    # 1. Momentum signals
    mom_1m = prices.pct_change(20)
    mom_3m = prices.pct_change(60)
    mom_6m = prices.pct_change(125)
    mom_12m = prices.pct_change(250)
    
    # 2. Volatility signals
    vol_1m = prices.pct_change().rolling(20).std()
    vol_3m = prices.pct_change().rolling(60).std()
    
    # 3. Moving average signals
    ma_50 = prices > prices.rolling(50).mean()
    ma_200 = prices > prices.rolling(200).mean()
    
    # 4. Relative strength signals
    spy_prices = strategy.download_data(['SPY'], start_date, end_date)
    
    if spy_prices.empty:
        print("Error downloading benchmark data. Skipping relative strength signals.")
        rs_mom = pd.DataFrame(0, index=prices.index, columns=prices.columns)
    else:
        relative_strength = prices.div(spy_prices['SPY'], axis=0)
        rs_mom = relative_strength.pct_change(60)
    
    # Combine signals into feature matrix
    features = pd.DataFrame(index=prices.index)
    
    for ticker in sector_etfs:
        features[f'{ticker}_mom_1m'] = mom_1m[ticker]
        features[f'{ticker}_mom_3m'] = mom_3m[ticker]
        features[f'{ticker}_mom_6m'] = mom_6m[ticker]
        features[f'{ticker}_mom_12m'] = mom_12m[ticker]
        features[f'{ticker}_vol_1m'] = vol_1m[ticker]
        features[f'{ticker}_vol_3m'] = vol_3m[ticker]
        features[f'{ticker}_ma_50'] = ma_50[ticker].astype(int)
        features[f'{ticker}_ma_200'] = ma_200[ticker].astype(int)
        features[f'{ticker}_rs_mom'] = rs_mom[ticker]
    
    # Fill missing values and align data
    features = features.fillna(0)
    
    # Define lookback window (1 year of data)
    lookback_window = 252
    
    # Get latest data for trading signals
    latest_features = features.iloc[-lookback_window:].values
    returns = prices.pct_change().dropna()
    latest_returns = returns.iloc[-lookback_window:].values
    
    # Initialize optimizer
    optimizer = SRICPortfolioOptimizer()
    
    # Select model dimension using SRIC
    T_years = lookback_window / 252
    selected_dim, sharpes, sric_values = optimizer.select_model_dimension(
        latest_features, 10, T_years, method='sric'
    )
    
    print(f"\nSelected model dimension: {selected_dim}")
    
    # Apply PCA to reduce dimensionality
    pca = PCA(n_components=min(10, latest_features.shape[1], latest_features.shape[0]))
    pca.fit(latest_features)
    features_pca = pca.transform(latest_features)
    
    # Optimize portfolio on selected dimension
    weights_pca, sharpe = optimizer.optimize_portfolio(features_pca[:, :selected_dim])
    
    print(f"In-sample Sharpe ratio: {sharpe:.4f}")
    print(f"Estimated out-of-sample Sharpe ratio: {optimizer.calculate_sric(sharpe, selected_dim-1, T_years):.4f}")
    
    # Map features to assets using correlations
    etf_weights = np.zeros(len(sector_etfs))
    
    for i in range(min(selected_dim, len(sector_etfs))):
        component = features_pca[:, i]
        
        # Calculate correlation between component and asset returns
        for j, ticker in enumerate(sector_etfs):
            # Get the correlation between PCA component and asset returns
            asset_returns = latest_returns[:, j]
            correlation = np.corrcoef(component, asset_returns)[0, 1]
            
            # Weight by correlation and PCA weight
            etf_weights[j] += weights_pca[i] * correlation
    
    # Normalize weights to sum to 1
    etf_weights = etf_weights / np.sum(np.abs(etf_weights))
    
    # Create portfolio recommendation
    portfolio = pd.DataFrame({
        'ETF': sector_etfs,
        'Weight': etf_weights,
        'Dollar Allocation': etf_weights * 10000  # Assuming $10,000 portfolio
    })
    
    # Sort by allocation (descending)
    portfolio = portfolio.sort_values('Weight', ascending=False)
    
    print("\nRecommended Portfolio Allocation:")
    print(portfolio)
    
    # Plot recommended allocation
    plt.figure(figsize=(10, 6))
    plt.bar(portfolio['ETF'], portfolio['Weight'])
    plt.title('Recommended Portfolio Allocation')
    plt.xlabel('Sector ETF')
    plt.ylabel('Portfolio Weight')
    plt.xticks(rotation=45)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Plot model selection metrics
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(sharpes)+1), sharpes, 'b-', label='In-Sample Sharpe')
    plt.plot(range(1, len(sric_values)+1), sric_values, 'g-', label='SRIC')
    plt.axvline(x=selected_dim, color='r', linestyle='--', 
                label=f'Selected Dimension ({selected_dim})')
    plt.title('Model Selection Metrics')
    plt.xlabel('Model Dimension')
    plt.ylabel('Sharpe Ratio / SRIC')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    return portfolio


def execute_trades(portfolio, api_key=None, account_id=None, total_portfolio_value=10000):
    """
    Execute trades based on portfolio allocation (example using a generic API)
    
    Parameters:
    -----------
    portfolio : pandas.DataFrame
        DataFrame with columns 'ETF' and 'Weight'
    api_key : str
        API key for broker
    account_id : str
        Account ID for broker
    total_portfolio_value : float
        Total portfolio value to allocate
    """
    if api_key is None:
        api_key = "YOUR_API_KEY"  # Replace with your actual API key
    
    if account_id is None:
        account_id = "YOUR_ACCOUNT_ID"  # Replace with your actual account ID
    
    try:
        # This is a placeholder for actual API implementation
        print(f"Connecting to broker API with key: {api_key[:5]}*** for account: {account_id}")
        
        # Get current positions (would come from API)
        current_positions = {etf: 0 for etf in portfolio['ETF']}
        
        # Get current prices (would come from API)
        current_prices = {}
        for etf in portfolio['ETF']:
            # Simulate getting current prices
            try:
                data = yf.download(etf, period="1d")
                if not data.empty:
                    current_prices[etf] = data['Close'].iloc[-1]
                else:
                    current_prices[etf] = 100  # Placeholder
            except:
                current_prices[etf] = 100  # Placeholder
        
        # Calculate target shares
        portfolio['Target Value'] = portfolio['Weight'] * total_portfolio_value
        portfolio['Target Shares'] = portfolio['Target Value'] / portfolio['ETF'].map(current_prices)
        portfolio['Target Shares'] = portfolio['Target Shares'].round().astype(int)
        
        # Calculate current position values
        portfolio['Current Shares'] = portfolio['ETF'].map(current_positions)
        portfolio['Current Value'] = portfolio['Current Shares'] * portfolio['ETF'].map(current_prices)
        
        # Calculate trades needed
        portfolio['Shares to Trade'] = portfolio['Target Shares'] - portfolio['Current Shares']
        
        # Execute trades
        for _, row in portfolio.iterrows():
            if row['Shares to Trade'] > 0:
                print(f"BUY {row['Shares to Trade']} shares of {row['ETF']} at ${current_prices[row['ETF']]}")
                # In actual implementation:
                # api.place_order(symbol=row['ETF'], qty=row['Shares to Trade'], side='buy')
            elif row['Shares to Trade'] < 0:
                print(f"SELL {-row['Shares to Trade']} shares of {row['ETF']} at ${current_prices[row['ETF']]}")
                # In actual implementation:
                # api.place_order(symbol=row['ETF'], qty=-row['Shares to Trade'], side='sell')
        
        # Print summary
        print("\nTrade Summary:")
        print(f"Total Portfolio Value: ${total_portfolio_value}")
        print(f"Total Buy Orders: {sum(portfolio['Shares to Trade'] > 0)}")
        print(f"Total Sell Orders: {sum(portfolio['Shares to Trade'] < 0)}")
        
        return True
    
    except Exception as e:
        print(f"Error executing trades: {e}")
        return False


# Main execution
def main():
    print("SRIC Trading Strategy - Main Menu")
    print("=" * 50)
    print("1. Visualize Sharpe Indifference Curves")
    print("2. Run Simple Portfolio Optimization Example")
    print("3. Run Industry Portfolio Example")
    print("4. Run Trading Strategy with Signals")
    print("5. Implement ETF Rotation Strategy (Current Allocation)")
    print("6. Execute Trades")
    print("7. Run Full Simulations")
    print("8. Exit")
    
    choice = input("\nEnter your choice (1-8): ")
    
    if choice == '1':
        analyze_sharpe_indifference_curves()
    elif choice == '2':
        simple_portfolio_optimization_example()
    elif choice == '3':
        run_industry_portfolio_example()
    elif choice == '4':
        run_trading_strategy_with_signals()
    elif choice == '5':
        portfolio = implement_etf_rotation_strategy()
    elif choice == '6':
        # First get the portfolio allocation
        portfolio = implement_etf_rotation_strategy()
        if portfolio is not None:
            # Then execute trades
            execute_trades(portfolio)
    elif choice == '7':
        results = run_simulation(
            n_trials=20,  # Reduced for faster execution
            n_samples=1260,
            n_assets=20,
            true_dims=[5, 10, 15],
            methods=['sric', 'aic', 'markowitz', 'equal_weight']
        )
        plot_simulation_results(results)
    elif choice == '8':
        print("Exiting program.")
        return
    else:
        print("Invalid choice. Please try again.")
    
    # Recursive call to show menu again
    main()


if __name__ == "__main__":
    main()