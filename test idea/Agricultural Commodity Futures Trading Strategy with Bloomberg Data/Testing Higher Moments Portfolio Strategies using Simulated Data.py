import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression
import datetime
from tqdm import tqdm
import warnings
import traceback
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

class HigherMomentsPortfolioStrategy:
    """
    Class for implementing and testing portfolio strategies that incorporate higher moments
    """
    
    def __init__(self):
        """Initialize the strategy"""
        pass
    
    def simulate_commodity_futures_data(self, n_assets=10, n_obs=1000, freq='daily'):
        """
        Simulate commodity futures price data with non-normal distributions
        
        Parameters:
        -----------
        n_assets : int
            Number of assets to simulate
        n_obs : int
            Number of observations
        freq : str
            Frequency of the data ('weekly', 'daily', or '30min')
            
        Returns:
        --------
        returns : pandas.DataFrame
            Simulated returns
        """
        # Set up parameters based on frequency
        if freq == 'weekly':
            # Lower volatility, less skewness and kurtosis for weekly data
            volatility_range = (0.01, 0.03)
            skewness_range = (-0.5, 0.5)
            kurtosis_add = 2
        elif freq == 'daily':
            # Medium volatility, skewness and kurtosis for daily data
            volatility_range = (0.005, 0.015)
            skewness_range = (-0.8, 0.8)
            kurtosis_add = 5
        elif freq == '30min':
            # Higher volatility, more skewness and kurtosis for high-frequency data
            volatility_range = (0.002, 0.006)
            skewness_range = (-1.2, 1.2)
            kurtosis_add = 10
        else:
            raise ValueError("freq must be 'weekly', 'daily', or '30min'")
        
        # Create date range
        if freq == 'weekly':
            dates = pd.date_range(start='2010-01-01', periods=n_obs, freq='W')
        elif freq == 'daily':
            dates = pd.date_range(start='2010-01-01', periods=n_obs, freq='B')
        else:  # 30min
            # Create business day dates first
            business_days = pd.date_range(start='2010-01-01', periods=n_obs//16, freq='B')
            # For each business day, create 16 30-minute intervals (assuming 8-hour trading day)
            all_times = []
            for day in business_days:
                for hour in range(9, 17):
                    for minute in [0, 30]:
                        all_times.append(pd.Timestamp(year=day.year, month=day.month, day=day.day, 
                                                 hour=hour, minute=minute))
            dates = pd.DatetimeIndex(all_times)[:n_obs]
        
        # Initialize returns DataFrame
        returns = pd.DataFrame(index=dates)
        
        # Simulate market returns with autocorrelation
        market_returns = np.zeros(n_obs)
        market_returns[0] = np.random.normal(0, volatility_range[1])
        
        # Parameters for market returns
        market_mean = -0.0005  # Slight negative mean as observed in the paper
        market_ar = 0.05  # Slight autocorrelation
        market_vol = volatility_range[1] * 0.8  # Market is generally less volatile than individual assets
        
        # Generate market returns with AR(1) process
        for i in range(1, n_obs):
            market_returns[i] = market_mean + market_ar * market_returns[i-1] + np.random.normal(0, market_vol)
        
        # Add non-normality to market returns
        skewness = np.random.uniform(skewness_range[0], skewness_range[0]/2)  # Negative skew more common
        market_returns = self.add_skewness_kurtosis(market_returns, skewness, kurtosis_add)
        
        # Add market returns to DataFrame
        returns['market'] = market_returns
        
        # Simulate individual asset returns
        for i in range(n_assets):
            # Parameters for individual asset
            beta = np.random.uniform(0.5, 1.5)  # Beta relative to market
            mean = np.random.uniform(-0.001, 0.001)  # Mean return
            vol = np.random.uniform(volatility_range[0], volatility_range[1])  # Volatility
            ar_coef = np.random.uniform(-0.1, 0.1)  # Autocorrelation
            
            # Initialize returns
            asset_returns = np.zeros(n_obs)
            asset_returns[0] = np.random.normal(mean, vol)
            
            # Generate returns with market exposure and autocorrelation
            for j in range(1, n_obs):
                # AR(1) process with market exposure
                asset_returns[j] = mean + beta * market_returns[j] + ar_coef * asset_returns[j-1] + np.random.normal(0, vol)
            
            # Add non-normality
            skewness = np.random.uniform(skewness_range[0], skewness_range[1])
            asset_returns = self.add_skewness_kurtosis(asset_returns, skewness, kurtosis_add)
            
            # Add to DataFrame
            returns[f'asset_{i+1}'] = asset_returns
        
        return returns
    
    def add_skewness_kurtosis(self, data, skew_param, kurt_add):
        """
        Add skewness and excess kurtosis to normally distributed data
        
        Parameters:
        -----------
        data : numpy.ndarray
            Input data
        skew_param : float
            Skewness parameter
        kurt_add : float
            Additional kurtosis to add
            
        Returns:
        --------
        transformed_data : numpy.ndarray
            Data with skewness and kurtosis
        """
        # Standardize data
        data_std = (data - np.mean(data)) / np.std(data)
        
        # Add skewness using the sinh-arcsinh transformation
        epsilon = 1e-10  # small number to avoid numerical issues
        delta = np.exp(skew_param)
        transformed_data = np.sinh(delta * np.arcsinh(data_std) + epsilon)
        
        # Add kurtosis by mixing with a t-distribution
        df = 5  # degrees of freedom for t-distribution
        t_dist = np.random.standard_t(df, size=len(data))
        
        # Mix normal and t-distribution to increase kurtosis
        mix_weight = kurt_add / (kurt_add + 10)  # Weight increases with kurt_add
        transformed_data = (1 - mix_weight) * transformed_data + mix_weight * t_dist
        
        # Rescale back to original mean and std
        transformed_data = transformed_data * np.std(data) + np.mean(data)
        
        return transformed_data
    
    def calculate_returns_moments(self, returns):
        """
        Calculate mean, variance, skewness, and kurtosis of returns
        
        Parameters:
        -----------
        returns : pandas.DataFrame
            Returns data
            
        Returns:
        --------
        moments : pandas.DataFrame
            Calculated moments for each asset
        """
        moments = pd.DataFrame(index=returns.columns)
        
        moments['mean'] = returns.mean()
        moments['variance'] = returns.var()
        moments['skewness'] = returns.skew()
        moments['kurtosis'] = returns.kurt()
        
        return moments
    
    def build_regression_models(self, returns, model_type='benchmark'):
        """
        Build regression models for each asset based on model type
        
        Parameters:
        -----------
        returns : pandas.DataFrame
            Returns data
        model_type : str
            Type of model to build. Options are:
            - 'benchmark': ri,t = β0 + β1 ri,t-1 + εit
            - 'systematic': ri,t = β0 + β1 ri,t-1 + β2 rm,t-1 + β3 r²m,t-1 + β4 r³m,t-1 + εit
            - 'individual': ri,t = β0 + β1 ri,t-1 + β2 r²i,t-1 + β3 r³i,t-1 + β4 r⁴i,t-1 + εit
            - 'all': ri,t = β0 + β1 ri,t-1 + β2 rm,t-1 + β3 r²m,t-1 + β4 r³m,t-1 + 
                     β5 r²i,t-1 + β6 r³i,t-1 + β7 r⁴i,t-1 + εit
            
        Returns:
        --------
        models : dict
            Dictionary of regression models for each asset
        """
        models = {}
        X_columns = []
        
        # Create lagged returns
        lagged_returns = returns.shift(1).copy()
        
        # Set up X columns based on model type
        if model_type == 'benchmark':
            # Only individual first moment
            X_columns = ['individual_lag1']
        
        elif model_type == 'systematic_1':
            # Individual first moment and market first moment
            X_columns = ['individual_lag1', 'market_lag1']
        
        elif model_type == 'systematic_2':
            # Individual first moment and market first and second moments
            X_columns = ['individual_lag1', 'market_lag1', 'market_lag1_squared']
        
        elif model_type == 'systematic_3':
            # Individual first moment and market first, second and third moments
            X_columns = ['individual_lag1', 'market_lag1', 'market_lag1_squared', 'market_lag1_cubed']
        
        elif model_type == 'individual_2':
            # Individual first and second moments
            X_columns = ['individual_lag1', 'individual_lag1_squared']
        
        elif model_type == 'individual_3':
            # Individual first, second and third moments
            X_columns = ['individual_lag1', 'individual_lag1_squared', 'individual_lag1_cubed']
        
        elif model_type == 'individual_4':
            # Individual first, second, third and fourth moments
            X_columns = ['individual_lag1', 'individual_lag1_squared', 'individual_lag1_cubed', 'individual_lag1_fourth']
        
        elif model_type == 'all':
            # All individual and systematic moments
            X_columns = ['individual_lag1', 'market_lag1', 'market_lag1_squared', 'market_lag1_cubed',
                        'individual_lag1_squared', 'individual_lag1_cubed', 'individual_lag1_fourth']
        
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Create predictors DataFrame
        X_data = pd.DataFrame(index=returns.index)
        
        # Add market predictors if needed
        if any('market' in col for col in X_columns):
            X_data['market_lag1'] = lagged_returns['market']
            X_data['market_lag1_squared'] = lagged_returns['market'] ** 2
            X_data['market_lag1_cubed'] = lagged_returns['market'] ** 3
        
        # For each asset, build a regression model
        for asset in returns.columns:
            if asset == 'market':
                continue
                
            # Add individual predictors
            X_data['individual_lag1'] = lagged_returns[asset]
            X_data['individual_lag1_squared'] = lagged_returns[asset] ** 2
            X_data['individual_lag1_cubed'] = lagged_returns[asset] ** 3
            X_data['individual_lag1_fourth'] = lagged_returns[asset] ** 4
            
            # Select only relevant columns and drop NAs
            X = X_data[X_columns].dropna()
            y = returns.loc[X.index, asset]
            
            # Fit model
            model = LinearRegression()
            model.fit(X, y)
            
            # Store model
            models[asset] = {
                'model': model,
                'X_columns': X_columns,
                'coefficients': dict(zip(X_columns, model.coef_)),
                'intercept': model.intercept_
            }
        
        return models, X_data
    
    def predict_returns_covariance(self, returns, models, X_data, estimation_window=250):
        """
        Predict expected returns and covariance matrix using regression models
        
        Parameters:
        -----------
        returns : pandas.DataFrame
            Returns data
        models : dict
            Dictionary of regression models for each asset
        X_data : pandas.DataFrame
            Predictors data
        estimation_window : int
            Number of observations to use for estimating covariance matrix
            
        Returns:
        --------
        expected_returns : pandas.Series
            Predicted expected returns
        covariance_matrix : pandas.DataFrame
            Estimated covariance matrix
        """
        # Get the latest data point for prediction
        latest_date = X_data.index[-1]
        X_latest = X_data.loc[latest_date]
        
        # Predict expected returns for each asset
        expected_returns = {}
        
        for asset, model_info in models.items():
            X_asset = X_latest[model_info['X_columns']]
            expected_returns[asset] = model_info['intercept'] + np.dot(X_asset, model_info['model'].coef_)
        
        # Convert to Series
        expected_returns = pd.Series(expected_returns)
        
        # Get residuals from the model for each asset
        residuals = pd.DataFrame(index=returns.index)
        
        for asset, model_info in models.items():
            # Select only dates where X_data is available
            valid_dates = X_data.dropna().index
            
            # Calculate fitted values
            X_asset = X_data.loc[valid_dates, model_info['X_columns']]
            y_hat = model_info['intercept'] + np.dot(X_asset, model_info['model'].coef_)
            
            # Calculate residuals
            residuals.loc[valid_dates, asset] = returns.loc[valid_dates, asset] - y_hat
        
        # Use last estimation_window observations for covariance estimation
        recent_residuals = residuals.tail(estimation_window).dropna(how='any')
        
        # If there are not enough observations, use all available
        if len(recent_residuals) < 10:
            recent_residuals = residuals.dropna(how='any')
        
        # Estimate covariance matrix from residuals
        covariance_matrix = recent_residuals.cov()
        
        return expected_returns, covariance_matrix
    
    def optimize_portfolio(self, expected_returns, covariance_matrix, risk_aversion=1.0):
        """
        Optimize portfolio weights using Lai et al. (2011) approach
        
        Parameters:
        -----------
        expected_returns : pandas.Series
            Expected returns for each asset
        covariance_matrix : pandas.DataFrame
            Covariance matrix
        risk_aversion : float
            Risk aversion parameter
            
        Returns:
        --------
        weights : pandas.Series
            Optimal portfolio weights
        """
        # For simplicity, we'll use a modified mean-variance optimization
        # that accounts for parameter uncertainty as described in the paper
        
        # Number of assets
        n = len(expected_returns)
        
        # Uncertainty adjustment factor based on estimation window
        # This simulates the parameter uncertainty consideration in Lai et al. (2011)
        adjustment_factor = 1.5
        
        # Adjusted covariance matrix to account for parameter uncertainty
        adjusted_covariance = covariance_matrix * adjustment_factor
        
        # Inverse of covariance matrix
        try:
            inv_cov = np.linalg.inv(adjusted_covariance.values)
        except np.linalg.LinAlgError:
            # If matrix is singular, use pseudoinverse
            inv_cov = np.linalg.pinv(adjusted_covariance.values)
        
        # Calculate weights
        weights_unnormalized = np.dot(inv_cov, expected_returns) / risk_aversion
        
        # Normalize weights to sum to 1
        weights = weights_unnormalized / np.sum(np.abs(weights_unnormalized))
        
        # Convert to Series
        weights = pd.Series(weights, index=expected_returns.index)
        
        # Ensure no short selling
        weights[weights < 0] = 0
        
        # Normalize again to sum to 1
        if weights.sum() > 0:
            weights = weights / weights.sum()
        else:
            # Equal weights if all weights are negative
            weights = pd.Series(1.0/n, index=expected_returns.index)
        
        return weights
    
    def backtest_strategy(self, returns, model_type, estimation_window=250, rebalance_freq=20):
        """
        Backtest portfolio strategy
        
        Parameters:
        -----------
        returns : pandas.DataFrame
            Returns data
        model_type : str
            Type of model to use for prediction
        estimation_window : int
            Number of observations to use for estimating models
        rebalance_freq : int
            Frequency of rebalancing in number of observations
            
        Returns:
        --------
        portfolio_returns : pandas.Series
            Portfolio returns
        weights_history : pandas.DataFrame
            History of portfolio weights
        """
        # Check if we have enough data
        if len(returns) <= estimation_window:
            print(f"Warning: Not enough data for {model_type} strategy. Need at least {estimation_window} observations.")
            return pd.Series(dtype=float), pd.DataFrame()
        
        # Initialize portfolio returns and weights
        portfolio_returns = pd.Series(index=returns.index[estimation_window:], dtype=float)
        weights_history = pd.DataFrame(index=returns.index[estimation_window:], columns=returns.columns.drop('market'))
        
        # Initialize with equal weights
        current_weights = pd.Series(1.0/len(returns.columns.drop('market')), index=returns.columns.drop('market'))
        
        # Backtest loop
        for i, date in tqdm(enumerate(returns.index[estimation_window:]), desc=f"Backtesting {model_type}", total=len(returns.index[estimation_window:])):
            # Rebalance portfolio at specified frequency
            if i % rebalance_freq == 0:
                try:
                    # Get data up to this date
                    data_up_to_date = returns.loc[:date]
                    
                    # Build regression models
                    models, X_data = self.build_regression_models(data_up_to_date.iloc[-estimation_window:], model_type)
                    
                    if not models or not X_data.any().any():
                        continue
                    
                    # Predict expected returns and covariance
                    expected_returns, covariance_matrix = self.predict_returns_covariance(
                        data_up_to_date.iloc[-estimation_window:], models, X_data, estimation_window)
                    
                    if expected_returns.empty or covariance_matrix.empty:
                        continue
                    
                    # Optimize portfolio
                    current_weights = self.optimize_portfolio(expected_returns, covariance_matrix)
                except Exception as e:
                    print(f"Error during rebalancing at {date}: {e}")
                    continue
            
            # Store current weights
            weights_history.loc[date] = current_weights
            
            # Calculate portfolio return
            if i < len(returns.index[estimation_window:]) - 1:
                try:
                    next_date = returns.index[estimation_window:][i+1]
                    next_returns = returns.loc[next_date, current_weights.index]
                    portfolio_returns.loc[date] = (current_weights * next_returns).sum()
                except Exception as e:
                    print(f"Error calculating return at {date}: {e}")
        
        # Remove dates with NaN returns
        portfolio_returns = portfolio_returns.dropna()
        
        # Check if we have any valid returns
        if portfolio_returns.empty:
            print(f"Warning: No valid returns for {model_type} strategy")
        
        return portfolio_returns, weights_history
    
    def calculate_performance_metrics(self, returns):
        """
        Calculate performance metrics for a return series
        
        Parameters:
        -----------
        returns : pandas.Series
            Return series
            
        Returns:
        --------
        metrics : dict
            Performance metrics
        """
        # Check if returns is empty
        if len(returns) == 0:
            return {
                'Frequency': "N/A",
                'Total Return': 0.0,
                'Annual Return': 0.0,
                'Volatility': 0.0,
                'Sharpe Ratio': 0.0,
                'Max Drawdown': 0.0,
                'Skewness': 0.0,
                'Kurtosis': 0.0,
                'Win Rate': 0.0
            }
        
        # Annualization factor
        if len(returns) >= 252:  # daily data
            annualization_factor = 252
            freq_str = "daily"
        elif len(returns) >= 52:  # weekly data
            annualization_factor = 52
            freq_str = "weekly"
        else:  # 30-minute data
            annualization_factor = 252 * 16  # Assuming 16 30-min periods per day
            freq_str = "30-minute"
        
        # Calculate metrics
        total_return = (1 + returns).prod() - 1
        annual_return = (1 + total_return) ** (annualization_factor / len(returns)) - 1
        volatility = returns.std() * np.sqrt(annualization_factor)
        sharpe_ratio = annual_return / volatility if volatility > 0 else 0
        
        # Maximum drawdown
        cumulative_returns = (1 + returns).cumprod()
        peak = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns / peak) - 1
        max_drawdown = drawdown.min()
        
        # Skewness and kurtosis
        skewness = returns.skew()
        kurtosis = returns.kurt()
        
        # Win rate
        win_rate = (returns > 0).mean()
        
        metrics = {
            'Frequency': freq_str,
            'Total Return': total_return,
            'Annual Return': annual_return,
            'Volatility': volatility,
            'Sharpe Ratio': sharpe_ratio,
            'Max Drawdown': max_drawdown,
            'Skewness': skewness,
            'Kurtosis': kurtosis,
            'Win Rate': win_rate
        }
        
        return metrics
    
    def compare_strategies(self, returns, model_types, estimation_window=250, rebalance_freq=20):
        """
        Compare different portfolio strategies
        
        Parameters:
        -----------
        returns : pandas.DataFrame
            Returns data
        model_types : list
            List of model types to compare
        estimation_window : int
            Number of observations to use for estimating models
        rebalance_freq : int
            Frequency of rebalancing in number of observations
            
        Returns:
        --------
        results : dict
            Dictionary of results for each strategy
        """
        results = {}
        
        for model_type in model_types:
            # Backtest strategy
            portfolio_returns, weights_history = self.backtest_strategy(
                returns, model_type, estimation_window, rebalance_freq)
            
            # Calculate performance metrics
            metrics = self.calculate_performance_metrics(portfolio_returns)
            
            # Store results
            results[model_type] = {
                'portfolio_returns': portfolio_returns,
                'weights_history': weights_history,
                'metrics': metrics
            }
        
        return results
    
    def plot_cumulative_returns(self, results):
        """
        Plot cumulative returns for different strategies
        
        Parameters:
        -----------
        results : dict
            Dictionary of results for each strategy
        """
        plt.figure(figsize=(12, 6))
        
        for model_type, result in results.items():
            returns = result['portfolio_returns']
            if not returns.empty:
                cumulative_returns = (1 + returns).cumprod()
                plt.plot(cumulative_returns.index, cumulative_returns, label=model_type)
        
        plt.title('Cumulative Returns')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Return')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    
    def plot_performance_comparison(self, results):
        """
        Plot performance metrics comparison for different strategies
        
        Parameters:
        -----------
        results : dict
            Dictionary of results for each strategy
        """
        # Extract metrics for comparison
        metrics_to_plot = ['Sharpe Ratio', 'Annual Return', 'Volatility', 'Max Drawdown']
        
        # Create DataFrame for plotting
        comparison_data = pd.DataFrame(index=results.keys(), columns=metrics_to_plot)
        
        for model_type, result in results.items():
            for metric in metrics_to_plot:
                comparison_data.loc[model_type, metric] = result['metrics'][metric]
        
        # Plot
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics_to_plot):
            ax = axes[i]
            comparison_data[metric].plot(kind='bar', ax=ax)
            ax.set_title(metric)
            ax.set_ylabel(metric)
            ax.grid(True, axis='y')
            
            # If plotting drawdown, invert y-axis
            if metric == 'Max Drawdown':
                ax.invert_yaxis()
        
        plt.tight_layout()
        plt.show()
    
    def print_performance_table(self, results):
        """
        Print performance metrics table for different strategies
        
        Parameters:
        -----------
        results : dict
            Dictionary of results for each strategy
        """
        # Create DataFrame for results
        metrics_df = pd.DataFrame()
        
        for model_type, result in results.items():
            metrics = result['metrics']
            metrics_df[model_type] = pd.Series(metrics)
        
        # Transpose for better display
        metrics_df = metrics_df.T
        
        # Reorder columns
        ordered_columns = ['Frequency', 'Annual Return', 'Volatility', 'Sharpe Ratio', 'Max Drawdown', 
                          'Skewness', 'Kurtosis', 'Win Rate', 'Total Return']
        
        # Filter only columns that exist
        available_columns = [col for col in ordered_columns if col in metrics_df.columns]
        metrics_df = metrics_df[available_columns]
        
        # Format values
        for col in ['Annual Return', 'Volatility', 'Max Drawdown', 'Total Return']:
            if col in metrics_df.columns:
                metrics_df[col] = metrics_df[col].map('{:.2%}'.format)
        
        for col in ['Sharpe Ratio', 'Skewness', 'Kurtosis', 'Win Rate']:
            if col in metrics_df.columns:
                metrics_df[col] = metrics_df[col].map('{:.2f}'.format)
        
        # Print table
        print("Performance Metrics:")
        print(metrics_df)
        
        return metrics_df
    
    def run_simulation(self, n_assets=10, n_obs=1000, freq='daily'):
        """
        Run a full simulation to test different portfolio strategies
        
        Parameters:
        -----------
        n_assets : int
            Number of assets to simulate
        n_obs : int
            Number of observations
        freq : str
            Frequency of the data ('weekly', 'daily', or '30min')
            
        Returns:
        --------
        results : dict
            Dictionary of results for each strategy
        """
        # Ensure we have enough observations
        if freq == 'weekly' and n_obs < 100:
            n_obs = 100
            print(f"Increasing n_obs to {n_obs} for weekly data to ensure enough observations for backtesting")
        elif freq == 'daily' and n_obs < 300:
            n_obs = 300
            print(f"Increasing n_obs to {n_obs} for daily data to ensure enough observations for backtesting")
        elif freq == '30min' and n_obs < 500:
            n_obs = 500
            print(f"Increasing n_obs to {n_obs} for 30-minute data to ensure enough observations for backtesting")
        
        # Simulate data
        print(f"Simulating {n_assets} assets with {n_obs} {freq} observations...")
        returns = self.simulate_commodity_futures_data(n_assets, n_obs, freq)
        
        # Calculate moments
        moments = self.calculate_returns_moments(returns)
        print("\nSimulated Data Statistics:")
        print(moments)
        
        # Plot return distributions
        plt.figure(figsize=(12, 6))
        plt.hist(returns['market'], bins=50, alpha=0.5, density=True, label='Market')
        plt.hist(returns['asset_1'], bins=50, alpha=0.5, density=True, label='Asset 1')
        plt.title(f'Return Distributions ({freq} frequency)')
        plt.xlabel('Return')
        plt.ylabel('Density')
        plt.legend()
        plt.show()
        
        # Define model types to compare
        model_types = [
            'benchmark',  # AR(1) only
            'systematic_1',  # AR(1) + market return
            'systematic_2',  # AR(1) + market return + market return squared
            'systematic_3',  # AR(1) + market return + market return squared + market return cubed
            'individual_2',  # AR(1) + individual return squared
            'individual_3',  # AR(1) + individual return squared + individual return cubed
            'individual_4',  # AR(1) + individual return squared + individual return cubed + individual return fourth
            'all'  # All higher moments
        ]
        
        # Set estimation window and rebalance frequency based on data frequency
        if freq == 'weekly':
            estimation_window = min(52, n_obs // 3)  # 1 year or less
            rebalance_freq = min(4, n_obs // 25)      # Monthly or less
        elif freq == 'daily':
            estimation_window = min(252, n_obs // 3)  # 1 year or less
            rebalance_freq = min(21, n_obs // 25)      # Monthly or less
        else:  # 30min
            estimation_window = min(252, n_obs // 3)  # About 1 year or less
            rebalance_freq = min(80, n_obs // 25)     # Weekly or less
        
        print(f"Using estimation window of {estimation_window} and rebalance frequency of {rebalance_freq}")
        
        # Compare strategies
        print(f"\nComparing {len(model_types)} portfolio strategies...")
        
        try:
            results = self.compare_strategies(returns, model_types, estimation_window, rebalance_freq)
            
            # Check if we have any valid results
            valid_results = False
            for model_type, result in results.items():
                if not result['portfolio_returns'].empty:
                    valid_results = True
                    break
            
            if not valid_results:
                print("Warning: No valid results for any strategy. Trying with simpler setup...")
                # Try with simpler setup
                model_types = ['benchmark', 'systematic_1', 'individual_2']
                estimation_window = max(20, n_obs // 10)
                rebalance_freq = max(5, n_obs // 100)
                print(f"Using simpler models with estimation window of {estimation_window} and rebalance frequency of {rebalance_freq}")
                results = self.compare_strategies(returns, model_types, estimation_window, rebalance_freq)
            
            # Plot results
            self.plot_cumulative_returns(results)
            self.plot_performance_comparison(results)
            metrics_df = self.print_performance_table(results)
            
            return results, metrics_df, returns
        
        except Exception as e:
            print(f"Error in simulation: {e}")
            traceback.print_exc()
            return {}, pd.DataFrame(), returns
    
    def run_multiple_simulations(self, n_simulations=5, n_assets=10, n_obs=500, frequencies=['weekly', 'daily', '30min']):
        """
        Run multiple simulations and aggregate results
        
        Parameters:
        -----------
        n_simulations : int
            Number of simulations to run
        n_assets : int
            Number of assets to simulate
        n_obs : int
            Number of observations
        frequencies : list
            List of frequencies to test
            
        Returns:
        --------
        aggregated_results : dict
            Dictionary of aggregated results
        """
        aggregated_results = {}
        
        for freq in frequencies:
            print(f"\n{'='*50}")
            print(f"Running {n_simulations} simulations for {freq} frequency")
            print(f"{'='*50}")
            
            # Initialize storage for aggregated metrics
            model_types = [
                'benchmark',
                'systematic_1',
                'systematic_2',
                'systematic_3',
                'individual_2',
                'individual_3',
                'individual_4',
                'all'
            ]
            
            metrics_to_aggregate = ['Sharpe Ratio', 'Annual Return', 'Volatility', 'Max Drawdown']
            aggregated_metrics = {model: {metric: [] for metric in metrics_to_aggregate} for model in model_types}
            
            # Run multiple simulations
            for sim in range(n_simulations):
                print(f"\nSimulation {sim+1}/{n_simulations}")
                
                # Run simulation
                results, metrics_df, _ = self.run_simulation(n_assets, n_obs, freq)
                
                # Store metrics
                for model in model_types:
                    for metric in metrics_to_aggregate:
                        if model in metrics_df.columns and metric in metrics_df.index:
                            try:
                                if metric in ['Annual Return', 'Volatility', 'Max Drawdown']:
                                    # Convert percentage string to float
                                    value = float(metrics_df.loc[metric, model].strip('%')) / 100
                                else:
                                    # Convert string to float
                                    value = float(metrics_df.loc[metric, model])
                                
                                aggregated_metrics[model][metric].append(value)
                            except Exception as e:
                                print(f"Error processing metric {metric} for model {model}: {e}")
                                continue
            
            # Calculate average metrics
            average_metrics = pd.DataFrame(index=model_types, columns=metrics_to_aggregate)
            
            for model in model_types:
                for metric in metrics_to_aggregate:
                    values = aggregated_metrics[model][metric]
                    average_metrics.loc[model, metric] = np.mean(values) if values else np.nan
            
            # Store aggregated results
            aggregated_results[freq] = average_metrics
            
            # Print and plot average results
            print(f"\nAverage results for {freq} frequency:")
            print(average_metrics)
            
            # Plot average results
            plt.figure(figsize=(12, 6))
            for i, metric in enumerate(metrics_to_aggregate):
                plt.subplot(2, 2, i + 1)
                average_metrics[metric].plot(kind='bar')
                plt.title(f'Average {metric}')
                plt.ylabel(metric)
                plt.grid(True, axis='y')
                
                # If plotting drawdown, invert y-axis
                if metric == 'Max Drawdown':
                    plt.gca().invert_yaxis()
            
            plt.tight_layout()
            plt.suptitle(f'Average Performance Metrics ({freq} frequency)')
            plt.subplots_adjust(top=0.9)
            plt.show()
        
        # Compare frequencies
        self.compare_frequencies(aggregated_results)
        
        return aggregated_results
    
    def compare_frequencies(self, aggregated_results):
        """
        Compare performance across different frequencies
        
        Parameters:
        -----------
        aggregated_results : dict
            Dictionary of aggregated results for each frequency
        """
        # Extract Sharpe ratios for comparison
        sharpe_ratios = pd.DataFrame({freq: results['Sharpe Ratio'] for freq, results in aggregated_results.items()})
        
        # Plot comparison
        plt.figure(figsize=(12, 6))
        sharpe_ratios.plot(kind='bar')
        plt.title('Sharpe Ratio Comparison Across Frequencies')
        plt.ylabel('Sharpe Ratio')
        plt.grid(True, axis='y')
        plt.legend(title='Frequency')
        plt.tight_layout()
        plt.show()
        
        # Print table
        print("\nSharpe Ratio Comparison Across Frequencies:")
        print(sharpe_ratios)
        
        return sharpe_ratios

# Run the strategy
if __name__ == "__main__":
    strategy = HigherMomentsPortfolioStrategy()
    
    # Run a single simulation for each frequency
    for freq in ['weekly', 'daily', '30min']:
        print(f"\n{'='*50}")
        print(f"Running simulation for {freq} frequency")
        print(f"{'='*50}")
        
        # Run simulation
        if freq == 'weekly':
            n_obs = 500
        elif freq == 'daily':
            n_obs = 1000
        else:  # 30min
            n_obs = 2000
            
        results, metrics_df, returns = strategy.run_simulation(n_assets=10, n_obs=n_obs, freq=freq)
    
    # Uncomment below to run multiple simulations
    # aggregated_results = strategy.run_multiple_simulations(n_simulations=3, n_assets=10, n_obs=500)