import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class RandomSubspaceTrader:
    """
    Trading strategy based on random subspace methods using simulated data
    """
    
    def __init__(self, 
                 k=10,                   # Subspace dimension 
                 n_draws=100,            # Number of random draws
                 method='rs',            # 'rs' for Random Subset, 'rp' for Random Projection
                 lookback_window=60,     # Days of historical data to use
                 rebalance_freq=5,       # Trading days between rebalancing
                 position_threshold=0.0, # Threshold for taking positions
                 max_position=1.0,       # Maximum position size
                 seed=42):               # Random seed
        """
        Initialize the Random Subspace Trading Strategy
        """
        self.k = k
        self.n_draws = n_draws
        self.method = method
        self.lookback_window = lookback_window
        self.rebalance_freq = rebalance_freq
        self.position_threshold = position_threshold
        self.max_position = max_position
        np.random.seed(seed)
        
        # Performance tracking
        self.positions = []
        self.returns = []
        self.benchmark_returns = []
        self.dates = []
        self.predictions = []
        self.selected_features = []
        
    def generate_random_matrix(self, px):
        """
        Generate a random matrix according to the chosen method
        """
        if self.method == 'rs':
            # Random Subset: select k columns out of px
            cols = np.random.choice(px, min(self.k, px), replace=False)
            R = np.zeros((px, min(self.k, px)))
            for i, col in enumerate(cols):
                R[col, i] = 1
        else:  # rp - Random Projection
            # Gaussian random weights
            R = np.random.normal(0, 1, (px, min(self.k, px)))
        
        return R
    
    def predict_returns(self, X_train, y_train, X_test, w_train=None, w_test=None):
        """
        Predict returns using random subspace method
        """
        # Get dimensions
        px = X_train.shape[1]
        
        # Generate predictions
        predictions = np.zeros(self.n_draws)
        selected_features_count = np.zeros(px)
        
        for i in range(self.n_draws):
            # Generate random matrix
            R = self.generate_random_matrix(px)
            
            # Track selected features (for RS method)
            if self.method == 'rs':
                selected = np.where(np.sum(R, axis=1) > 0)[0]
                selected_features_count[selected] += 1
            
            # Apply dimension reduction
            X_train_reduced = X_train @ R
            X_test_reduced = X_test @ R
            
            # Combine with essential predictors if provided
            if w_train is not None and w_test is not None:
                Z_train = np.hstack([w_train, X_train_reduced])
                Z_test = np.hstack([w_test, X_test_reduced])
            else:
                Z_train = X_train_reduced
                Z_test = X_test_reduced
            
            # Fit model and predict
            try:
                beta = np.linalg.lstsq(Z_train, y_train, rcond=None)[0]
                predictions[i] = Z_test @ beta
            except:
                # In case of numerical issues
                predictions[i] = 0.0
        
        # Average over all predictions
        y_pred = np.mean(predictions)
        
        # Return prediction and feature importance
        return y_pred, selected_features_count / self.n_draws

    def generate_simulated_data(self, T=1000, n_features=50, n_factors=3):
        """
        Generate simulated price data with factor structure
        
        Parameters:
        -----------
        T : int
            Number of time periods
        n_features : int
            Number of features to generate
        n_factors : int
            Number of underlying factors
            
        Returns:
        --------
        prices : pandas DataFrame
            Simulated price data
        features : pandas DataFrame
            Feature data
        macro_data : pandas DataFrame
            Macroeconomic indicator data
        """
        # Generate dates
        start_date = datetime(2010, 1, 1)
        dates = [start_date + timedelta(days=i) for i in range(T)]
        
        # Generate factor returns
        factor_returns = np.random.normal(0.0005, 0.01, (T, n_factors))  # Daily mean 0.05%, std 1%
        
        # Generate loadings for asset on factors
        asset_loadings = np.random.normal(0, 1, n_factors)
        asset_loadings = asset_loadings / np.sum(np.abs(asset_loadings))  # Normalize
        
        # Generate benchmark loadings (different from asset)
        benchmark_loadings = np.random.normal(0, 1, n_factors)
        benchmark_loadings = benchmark_loadings / np.sum(np.abs(benchmark_loadings))
        
        # Generate asset returns with factors plus idiosyncratic component
        asset_returns = factor_returns @ asset_loadings + np.random.normal(0, 0.01, T)
        
        # Generate benchmark returns
        benchmark_returns = factor_returns @ benchmark_loadings + np.random.normal(0, 0.01, T)
        
        # Generate prices from returns
        asset_prices = 100 * np.cumprod(1 + asset_returns)
        benchmark_prices = 100 * np.cumprod(1 + benchmark_returns)
        
        # Create DataFrames
        prices = pd.DataFrame({
            'asset': asset_prices,
            'benchmark': benchmark_prices
        }, index=dates)
        
        # Create features
        features = pd.DataFrame(index=dates)
        
        # 1. Technical indicators based on asset prices
        # Moving averages
        for window in [5, 10, 20, 50]:
            features[f'ma_{window}'] = prices['asset'].rolling(window).mean() / prices['asset'] - 1
        
        # Momentum
        for window in [1, 5, 10, 20]:
            features[f'mom_{window}'] = prices['asset'].pct_change(window)
        
        # Volatility
        for window in [5, 10, 20]:
            features[f'vol_{window}'] = prices['asset'].pct_change().rolling(window).std()
        
        # 2. Generate features with predictive power
        # Some features will be based on the factors (which drive returns)
        for i in range(n_factors):
            # Lagged factor
            features[f'factor_{i}'] = np.roll(factor_returns[:, i], 1)
            
            # Noisy version of factor
            features[f'noisy_factor_{i}'] = features[f'factor_{i}'] + np.random.normal(0, 0.005, T)
            
            # Squared factor (nonlinear relationship)
            features[f'factor_{i}_squared'] = features[f'factor_{i}']**2
        
        # 3. Add noise features with no predictive power
        for i in range(n_features - 3*n_factors - 3*4):
            features[f'noise_{i}'] = np.random.normal(0, 0.01, T)
        
        # Generate macro data
        macro_data = pd.DataFrame(index=dates)
        
        # Gold, Bonds, Oil proxies (related to factors)
        macro_data['gold'] = 100 * np.cumprod(1 + 0.0002 + 0.5*factor_returns[:, 0] + np.random.normal(0, 0.005, T))
        macro_data['bonds'] = 100 * np.cumprod(1 + 0.0001 - 0.3*factor_returns[:, 1] + np.random.normal(0, 0.003, T))
        macro_data['oil'] = 100 * np.cumprod(1 + 0.0003 + 0.7*factor_returns[:, 2] + np.random.normal(0, 0.015, T))
        
        # Drop NaN values from features
        features = features.fillna(0)
        
        # Calculate asset and benchmark returns
        asset_returns_series = pd.Series(asset_returns, index=dates)
        benchmark_returns_series = pd.Series(benchmark_returns, index=dates)
        
        return prices, features, macro_data, asset_returns_series, benchmark_returns_series
    
    def backtest_simulated(self, T=1000, n_features=50, n_factors=3):
        """
        Backtest the trading strategy on simulated data
        
        Parameters:
        -----------
        T : int
            Number of time periods
        n_features : int
            Number of features to generate
        n_factors : int
            Number of underlying factors
        """
        # Generate simulated data
        print(f"Generating simulated data with {n_features} features and {n_factors} factors...")
        prices, features, macro_data, asset_returns, benchmark_returns = self.generate_simulated_data(
            T=T, n_features=n_features, n_factors=n_factors
        )
        
        # Reserve first portion for training
        train_size = self.lookback_window
        
        # Standardize features
        scaler = StandardScaler()
        
        # Initialize tracking variables
        self.positions = []
        self.returns = []
        self.benchmark_returns = []
        self.dates = []
        self.predictions = []
        self.selected_features = []
        current_position = 0.0
        
        # Loop through trading days
        trading_days = features.index[train_size:].tolist()
        
        for i, current_date in enumerate(trading_days):
            # Skip if not a rebalance day and we already have a position
            if i % self.rebalance_freq != 0 and i > 0:
                # Record current position and return
                self.positions.append(current_position)
                self.returns.append(asset_returns.loc[current_date] * current_position)
                self.benchmark_returns.append(benchmark_returns.loc[current_date])
                self.dates.append(current_date)
                self.predictions.append(None)  # No new prediction
                self.selected_features.append(None)  # No new feature selection
                continue
            
            # Get training data (using lookback window)
            lookback_idx = features.index.get_loc(current_date) - self.lookback_window
            train_dates = features.index[lookback_idx:features.index.get_loc(current_date)]
            
            X_train = features.loc[train_dates].values
            y_train = asset_returns.loc[train_dates].values
            
            # If not enough training data, skip
            if len(y_train) < 30:
                continue
            
            # Standardize features
            X_train_std = scaler.fit_transform(X_train)
            X_test_std = scaler.transform(features.loc[current_date:current_date].values)
            
            # Predict return
            prediction, feature_importances = self.predict_returns(X_train_std, y_train, X_test_std)
            
            # Determine position size based on prediction
            if prediction > self.position_threshold:
                # Long position
                position_size = min(prediction / 0.01, self.max_position)  # Scale by expected return
            elif prediction < -self.position_threshold:
                # Short position
                position_size = max(prediction / 0.01, -self.max_position)  # Scale by expected return
            else:
                # No position
                position_size = 0.0
            
            # Update current position
            current_position = position_size
            
            # Record position, return, and date
            self.positions.append(current_position)
            self.returns.append(asset_returns.loc[current_date] * current_position)
            self.benchmark_returns.append(benchmark_returns.loc[current_date])
            self.dates.append(current_date)
            self.predictions.append(prediction)
            self.selected_features.append(feature_importances)
            
            # Print progress every 20% of the way
            if i % (len(trading_days) // 5) == 0:
                print(f"Progress: {i/len(trading_days)*100:.1f}% - Current date: {current_date.strftime('%Y-%m-%d')}")
        
        # Convert lists to pandas Series/DataFrames
        self.positions = pd.Series(self.positions, index=self.dates)
        self.returns = pd.Series(self.returns, index=self.dates)
        self.benchmark_returns = pd.Series(self.benchmark_returns, index=self.dates)
        self.predictions = pd.Series([p for p in self.predictions if p is not None], index=[d for i, d in enumerate(self.dates) if self.predictions[i] is not None])
        
        # Calculate cumulative returns
        self.cumulative_returns = (1 + self.returns).cumprod() - 1
        self.cumulative_benchmark = (1 + self.benchmark_returns).cumprod() - 1
        
        # Calculate performance metrics
        self.calculate_performance()
        
        return self.cumulative_returns, self.cumulative_benchmark
    
    def calculate_performance(self):
        """
        Calculate performance metrics
        """
        if len(self.dates) == 0:
            print("No trading dates available")
            return
            
        # Annualized return (assuming 252 trading days per year)
        days = (self.dates[-1] - self.dates[0]).days
        years = days / 365.25
        
        if years > 0:
            self.annual_return = ((1 + self.cumulative_returns.iloc[-1]) ** (1/years)) - 1
            self.annual_benchmark = ((1 + self.cumulative_benchmark.iloc[-1]) ** (1/years)) - 1
        else:
            self.annual_return = self.cumulative_returns.iloc[-1]
            self.annual_benchmark = self.cumulative_benchmark.iloc[-1]
        
        # Volatility
        self.volatility = self.returns.std() * np.sqrt(252)  # Annualized volatility
        self.benchmark_vol = self.benchmark_returns.std() * np.sqrt(252)
        
        # Sharpe ratio (assuming 0% risk-free rate for simplicity)
        self.sharpe_ratio = self.annual_return / self.volatility if self.volatility > 0 else 0
        self.benchmark_sharpe = self.annual_benchmark / self.benchmark_vol if self.benchmark_vol > 0 else 0
        
        # Maximum drawdown
        cum_returns = (1 + self.returns).cumprod()
        running_max = cum_returns.cummax()
        drawdown = (cum_returns / running_max) - 1
        self.max_drawdown = drawdown.min()
        
        # Win rate
        self.win_rate = (self.returns > 0).mean()
        
        # Print performance summary
        print("\nPerformance Summary:")
        print(f"Annualized Return: {self.annual_return*100:.2f}% (Benchmark: {self.annual_benchmark*100:.2f}%)")
        print(f"Volatility: {self.volatility*100:.2f}% (Benchmark: {self.benchmark_vol*100:.2f}%)")
        print(f"Sharpe Ratio: {self.sharpe_ratio:.2f} (Benchmark: {self.benchmark_sharpe:.2f})")
        print(f"Maximum Drawdown: {self.max_drawdown*100:.2f}%")
        print(f"Win Rate: {self.win_rate*100:.2f}%")
    
    def plot_performance(self):
        """
        Plot performance charts
        """
        if len(self.dates) == 0:
            print("No data to plot")
            return
            
        # Create figure with 3 subplots
        fig, axs = plt.subplots(3, 1, figsize=(12, 15), gridspec_kw={'height_ratios': [3, 1, 2]})
        
        # Plot cumulative returns
        axs[0].plot(self.cumulative_returns, label=f'Strategy ({self.method.upper()})')
        axs[0].plot(self.cumulative_benchmark, label='Benchmark', alpha=0.7)
        axs[0].set_title('Cumulative Returns', fontsize=14)
        axs[0].set_ylabel('Return (%)')
        axs[0].grid(True, alpha=0.3)
        axs[0].legend()
        
        # Plot positions
        axs[1].plot(self.positions, color='green' if np.mean(self.positions) >= 0 else 'red')
        axs[1].set_title('Position Size', fontsize=14)
        axs[1].set_ylabel('Position')
        axs[1].grid(True, alpha=0.3)
        
        # Plot predictions when available
        valid_predictions = self.predictions.dropna()
        if len(valid_predictions) > 0:
            axs[2].plot(valid_predictions, color='blue')
            axs[2].set_title('Return Predictions', fontsize=14)
            axs[2].set_ylabel('Predicted Return')
            axs[2].grid(True, alpha=0.3)
            
            # Add zero line
            axs[2].axhline(y=0, color='gray', linestyle='--', alpha=0.7)
            
            # Add threshold lines
            axs[2].axhline(y=self.position_threshold, color='green', linestyle='--', alpha=0.5)
            axs[2].axhline(y=-self.position_threshold, color='red', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig(f'rs_strategy_performance_{self.method}.png', dpi=300)
        plt.show()
    
    def plot_feature_importance(self, feature_names=None):
        """
        Plot feature importance
        """
        # Extract non-None feature importances
        feature_importances = [f for f in self.selected_features if f is not None]
        
        if not feature_importances or self.method != 'rs':
            print("Feature importance plot only available for RS method with at least one prediction.")
            return
        
        # Average feature importance
        avg_importance = np.mean(feature_importances, axis=0)
        
        # Get feature names
        if feature_names is None:
            feature_names = [f'Feature_{i}' for i in range(len(avg_importance))]
        
        # Sort by importance
        sorted_idx = np.argsort(avg_importance)[::-1]
        sorted_importance = avg_importance[sorted_idx]
        sorted_names = [feature_names[i] for i in sorted_idx]
        
        # Plot top 20 features
        plt.figure(figsize=(12, 8))
        plt.barh(range(min(20, len(sorted_names))), sorted_importance[:20], align='center')
        plt.yticks(range(min(20, len(sorted_names))), sorted_names[:20])
        plt.title('Feature Importance (Selection Frequency)', fontsize=14)
        plt.xlabel('Average Selection Frequency')
        plt.tight_layout()
        plt.savefig(f'rs_feature_importance_{self.method}.png', dpi=300)
        plt.show()
        
        return sorted_names[:20], sorted_importance[:20]

# Example usage with simulated data
if __name__ == "__main__":
    # Set parameters for simulation
    T = 1000              # Number of days
    n_features = 50       # Number of features
    n_factors = 3         # Number of underlying factors
    
    # Initialize strategies
    print("Testing Random Subset (RS) strategy...")
    rs_strategy = RandomSubspaceTrader(
        k=10,                    # Subspace dimension
        n_draws=100,             # Number of random draws
        method='rs',             # Random Subset
        lookback_window=60,      # Use 60 days of history
        rebalance_freq=5,        # Rebalance every 5 days
        position_threshold=0.001, # 0.1% threshold for taking positions
        max_position=1.0         # Maximum position size
    )
    
    # Run backtest on simulated data
    rs_strategy.backtest_simulated(T=T, n_features=n_features, n_factors=n_factors)
    
    # Plot results
    rs_strategy.plot_performance()
    
    # Generate feature names
    feature_names = []
    
    # Technical features
    for window in [5, 10, 20, 50]:
        feature_names.append(f'ma_{window}')
    
    for window in [1, 5, 10, 20]:
        feature_names.append(f'mom_{window}')
    
    for window in [5, 10, 20]:
        feature_names.append(f'vol_{window}')
    
    # Factor-based features
    for i in range(n_factors):
        feature_names.append(f'factor_{i}')
        feature_names.append(f'noisy_factor_{i}')
        feature_names.append(f'factor_{i}_squared')
    
    # Noise features
    for i in range(n_features - 3*n_factors - 3*4):
        feature_names.append(f'noise_{i}')
    
    # Plot feature importance
    top_features, importances = rs_strategy.plot_feature_importance(feature_names)
    
    # Print top features
    print("\nTop 10 Features by Selection Frequency:")
    for i in range(min(10, len(top_features))):
        print(f"{i+1}. {top_features[i]}: {importances[i]:.4f}")
    
    # Test Random Projection method
    print("\nTesting Random Projection (RP) strategy...")
    rp_strategy = RandomSubspaceTrader(
        k=10,                    # Subspace dimension
        n_draws=100,             # Number of random draws
        method='rp',             # Random Projection
        lookback_window=60,      # Use 60 days of history
        rebalance_freq=5,        # Rebalance every 5 days
        position_threshold=0.001, # 0.1% threshold for taking positions
        max_position=1.0         # Maximum position size
    )
    
    # Run backtest on simulated data
    rp_strategy.backtest_simulated(T=T, n_features=n_features, n_factors=n_factors)
    
    # Plot results
    rp_strategy.plot_performance()
    
    # Compare performances
    print("\nStrategy Comparison:")
    print(f"RS Sharpe Ratio: {rs_strategy.sharpe_ratio:.2f}")
    print(f"RP Sharpe Ratio: {rp_strategy.sharpe_ratio:.2f}")
    
    if rs_strategy.sharpe_ratio > 0 and rp_strategy.sharpe_ratio > 0:
        better = "RS" if rs_strategy.sharpe_ratio > rp_strategy.sharpe_ratio else "RP"
        ratio = max(rs_strategy.sharpe_ratio, rp_strategy.sharpe_ratio) / min(rs_strategy.sharpe_ratio, rp_strategy.sharpe_ratio)
        print(f"{better} outperforms by a factor of {ratio:.2f}x")
    
    # Test with different subspace dimensions
    print("\nTesting impact of subspace dimension k...")
    k_values = [5, 10, 15, 20, 25]
    rs_sharpes = []
    rp_sharpes = []
    
    for k in k_values:
        print(f"\nTesting with k = {k}")
        
        # Random Subset
        rs_k = RandomSubspaceTrader(
            k=k, 
            method='rs',
            lookback_window=60,
            rebalance_freq=5,
            position_threshold=0.001,
            max_position=1.0
        )
        rs_k.backtest_simulated(T=T, n_features=n_features, n_factors=n_factors)
        rs_sharpes.append(rs_k.sharpe_ratio)
        
        # Random Projection
        rp_k = RandomSubspaceTrader(
            k=k, 
            method='rp',
            lookback_window=60,
            rebalance_freq=5,
            position_threshold=0.001,
            max_position=1.0
        )
        rp_k.backtest_simulated(T=T, n_features=n_features, n_factors=n_factors)
        rp_sharpes.append(rp_k.sharpe_ratio)
    
    # Plot sharpe ratio vs k
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, rs_sharpes, 'o-', label='Random Subset (RS)')
    plt.plot(k_values, rp_sharpes, 's-', label='Random Projection (RP)')
    plt.xlabel('Subspace Dimension (k)')
    plt.ylabel('Sharpe Ratio')
    plt.title('Impact of Subspace Dimension on Performance')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('subspace_dimension_impact.png', dpi=300)
    plt.show()
    
    # Print optimal k values
    print("\nOptimal subspace dimensions:")
    print(f"RS: k = {k_values[np.argmax(rs_sharpes)]} (Sharpe: {max(rs_sharpes):.2f})")
    print(f"RP: k = {k_values[np.argmax(rp_sharpes)]} (Sharpe: {max(rp_sharpes):.2f})")