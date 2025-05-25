import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import warnings
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.model_selection import KFold
from tqdm import tqdm
warnings.filterwarnings('ignore')

class SparseSignalsStrategy:
    """
    Simplified implementation of the trading strategy from 'Sparse Signals in the Cross-Section of Returns'
    by Chinco, Clark-Joseph, and Ye (2019)
    """
    
    def __init__(self, n_stocks=40, n_minutes=120, estimation_window=30, n_lags=3, random_seed=42):
        """
        Initialize the strategy with synthetic data
        
        Parameters:
        -----------
        n_stocks : int
            Number of stocks to generate
        n_minutes : int
            Number of minutes to generate
        estimation_window : int
            Number of minutes in the estimation window
        n_lags : int
            Number of lags to use
        random_seed : int
            Random seed for reproducibility
        """
        self.n_stocks = n_stocks
        self.n_minutes = n_minutes
        self.estimation_window = estimation_window
        self.n_lags = n_lags
        self.random_seed = random_seed
        
        # Set random seed
        np.random.seed(random_seed)
        
        # Generate synthetic data
        self.generate_data()
        
        # Storage for results
        self.forecasts = {}
        self.selected_features = {}
        self.positions = pd.DataFrame()
        self.returns = pd.DataFrame()
    
    def generate_data(self):
        """
        Generate synthetic price and return data
        """
        print(f"Generating synthetic data for {self.n_stocks} stocks over {self.n_minutes} minutes...")
        
        # Create stock names
        self.stock_names = [f"STOCK{i:03d}" for i in range(self.n_stocks)]
        
        # Create timestamps (9:30 AM to end)
        base_time = dt.datetime.now().replace(hour=9, minute=30, second=0, microsecond=0)
        self.timestamps = [base_time + dt.timedelta(minutes=i) for i in range(self.n_minutes)]
        
        # Initialize return matrix with random noise
        # Shape: (n_minutes, n_stocks)
        returns = np.random.normal(0, 0.001, (self.n_minutes, self.n_stocks))
        
        # Add sparse predictive relationships as described in the paper
        for t in range(3, self.n_minutes):  # Start after minute 3 to allow for lags
            # Every few minutes, introduce new predictive relationships
            if t % 5 == 0:  # New relationships every 5 minutes
                # Select 5 random stocks to be predictors
                predictor_indices = np.random.choice(self.n_stocks, 5, replace=False)
                
                for predictor_idx in predictor_indices:
                    # Each predictor predicts 20% of other stocks
                    n_predicted = int(0.2 * self.n_stocks)
                    predicted_indices = np.random.choice(
                        [i for i in range(self.n_stocks) if i != predictor_idx],
                        n_predicted,
                        replace=False
                    )
                    
                    for predicted_idx in predicted_indices:
                        # Effect size +/- 0.19 as in paper
                        effect = np.random.choice([-0.19, 0.19])
                        
                        # Effect comes from lag 1, 2, or 3
                        lag = np.random.choice([1, 2, 3])
                        
                        # Add predictive effect
                        if t - lag >= 0:
                            returns[t, predicted_idx] += effect * returns[t-lag, predictor_idx]
        
        # Convert to DataFrame
        self.returns_df = pd.DataFrame(
            returns, 
            index=self.timestamps, 
            columns=self.stock_names
        )
        
        # Create price data from returns (starting at 100)
        prices = np.zeros((self.n_minutes, self.n_stocks))
        prices[0] = 100  # Starting prices
        
        for t in range(1, self.n_minutes):
            prices[t] = prices[t-1] * (1 + returns[t])
        
        self.prices_df = pd.DataFrame(
            prices,
            index=self.timestamps,
            columns=self.stock_names
        )
        
        # Set transaction costs (bid-ask spreads)
        self.transaction_costs = {stock: 0.001 for stock in self.stock_names}  # 10 bps
        
        # Select target stocks (stocks to predict)
        self.target_stocks = np.random.choice(
            self.stock_names,
            min(20, self.n_stocks),  # Select up to 20 stocks
            replace=False
        )
        
        print(f"Generated data for {len(self.target_stocks)} target stocks")
    
    def run_lasso(self, target_stock, time_idx):
        """
        Run LASSO for a single target stock at a specific time
        
        Parameters:
        -----------
        target_stock : str
            Name of target stock to predict
        time_idx : int
            Index of current time in the timestamp array
            
        Returns:
        --------
        forecast : float
            Return forecast
        selected : list
            List of selected predictors
        """
        # Check if we have enough data
        if time_idx < self.estimation_window + self.n_lags:
            return 0, []
            
        # Check if we have a next minute to predict
        if time_idx >= len(self.timestamps) - 1:
            return 0, []
        
        # Get estimation data
        start_idx = time_idx - self.estimation_window
        end_idx = time_idx
        
        # Target is the next minute's return for the target stock
        y = self.returns_df.iloc[time_idx + 1][target_stock]
        
        # Create features matrix
        X = np.zeros((self.estimation_window, self.n_stocks * self.n_lags))
        
        # Fill features matrix with lagged returns
        for i, lag in enumerate(range(1, self.n_lags + 1)):
            for j, stock in enumerate(self.stock_names):
                col_idx = j * self.n_lags + i
                
                # Get lagged returns for this stock
                lagged_returns = self.returns_df[stock].iloc[start_idx:end_idx].shift(lag).values
                
                # Fill features matrix
                X[:, col_idx] = np.pad(lagged_returns, (0, max(0, self.estimation_window - len(lagged_returns))), 'constant')
        
        # Remove any rows with NaN values (from the shifts)
        valid_rows = ~np.isnan(X).any(axis=1)
        X_valid = X[valid_rows]
        
        # If we don't have enough valid data, return empty
        if len(X_valid) < 10:  # Need at least 10 observations
            return 0, []
        
        # Standardize features
        X_mean = np.mean(X_valid, axis=0)
        X_std = np.std(X_valid, axis=0)
        X_std[X_std == 0] = 1.0  # Avoid division by zero
        
        X_valid_std = (X_valid - X_mean) / X_std
        
        # Create array of shape (len(X_valid),) filled with the target value
        y_valid = np.full(len(X_valid), y)
        
        # Try different alphas
        alphas = np.logspace(-5, 0, 10)
        best_alpha = 0.01  # Default
        
        # Simple cross-validation
        cv_errors = []
        for alpha in alphas:
            model = Lasso(alpha=alpha, max_iter=10000, fit_intercept=True)
            
            # 5-fold cross-validation
            kf = KFold(n_splits=min(5, len(X_valid)), shuffle=True, random_state=42)
            fold_errors = []
            
            for train_idx, val_idx in kf.split(X_valid_std):
                # Train model
                model.fit(X_valid_std[train_idx], y_valid[train_idx])
                
                # Predict and calculate error
                preds = model.predict(X_valid_std[val_idx])
                mse = np.mean((preds - y_valid[val_idx]) ** 2)
                fold_errors.append(mse)
            
            # Average error across folds
            cv_errors.append(np.mean(fold_errors))
        
        # Choose best alpha
        best_alpha_idx = np.argmin(cv_errors)
        best_alpha = alphas[best_alpha_idx]
        
        # Fit final model with best alpha
        final_model = Lasso(alpha=best_alpha, max_iter=10000, fit_intercept=True)
        final_model.fit(X_valid_std, y_valid)
        
        # Get current predictors for forecast
        X_current = np.zeros(self.n_stocks * self.n_lags)
        
        for i, lag in enumerate(range(1, self.n_lags + 1)):
            for j, stock in enumerate(self.stock_names):
                col_idx = j * self.n_lags + i
                
                # Get the appropriate lagged return
                if time_idx - lag >= 0:
                    X_current[col_idx] = self.returns_df[stock].iloc[time_idx - lag]
                else:
                    X_current[col_idx] = 0
        
        # Standardize current predictors
        X_current_std = (X_current - X_mean) / X_std
        
        # Make forecast
        forecast = final_model.predict([X_current_std])[0]
        
        # Get selected features
        selected = np.where(final_model.coef_ != 0)[0]
        
        return forecast, selected
    
    def run_benchmark(self, target_stock, time_idx):
        """
        Run AR(3) benchmark for a single target stock at a specific time
        
        Parameters:
        -----------
        target_stock : str
            Name of target stock to predict
        time_idx : int
            Index of current time in the timestamp array
            
        Returns:
        --------
        forecast : float
            Return forecast
        """
        # Check if we have enough data
        if time_idx < self.estimation_window + self.n_lags:
            return 0
            
        # Check if we have a next minute to predict
        if time_idx >= len(self.timestamps) - 1:
            return 0
        
        # Get estimation data
        start_idx = time_idx - self.estimation_window
        end_idx = time_idx
        
        # Target is the next minute's return for the target stock
        y = self.returns_df.iloc[time_idx + 1][target_stock]
        
        # Create features matrix with just the target stock's lagged returns
        X = np.zeros((self.estimation_window, self.n_lags))
        
        # Fill features matrix with lagged returns
        for i, lag in enumerate(range(1, self.n_lags + 1)):
            # Get lagged returns for target stock
            lagged_returns = self.returns_df[target_stock].iloc[start_idx:end_idx].shift(lag).values
            
            # Fill features matrix
            X[:, i] = np.pad(lagged_returns, (0, max(0, self.estimation_window - len(lagged_returns))), 'constant')
        
        # Remove any rows with NaN values (from the shifts)
        valid_rows = ~np.isnan(X).any(axis=1)
        X_valid = X[valid_rows]
        
        # If we don't have enough valid data, return empty
        if len(X_valid) < 10:  # Need at least 10 observations
            return 0
        
        # Standardize features
        X_mean = np.mean(X_valid, axis=0)
        X_std = np.std(X_valid, axis=0)
        X_std[X_std == 0] = 1.0  # Avoid division by zero
        
        X_valid_std = (X_valid - X_mean) / X_std
        
        # Create array of shape (len(X_valid),) filled with the target value
        y_valid = np.full(len(X_valid), y)
        
        # Fit linear regression model
        model = LinearRegression(fit_intercept=True)
        model.fit(X_valid_std, y_valid)
        
        # Get current predictors for forecast
        X_current = np.zeros(self.n_lags)
        
        for i, lag in enumerate(range(1, self.n_lags + 1)):
            # Get the appropriate lagged return
            if time_idx - lag >= 0:
                X_current[i] = self.returns_df[target_stock].iloc[time_idx - lag]
            else:
                X_current[i] = 0
        
        # Standardize current predictors
        X_current_std = (X_current - X_mean) / X_std
        
        # Make forecast
        forecast = model.predict([X_current_std])[0]
        
        return forecast
    
    def execute_strategy(self):
        """
        Execute the trading strategy
        """
        print("Executing strategy...")
        
        # Skip the first (estimation_window + n_lags) minutes
        start_idx = self.estimation_window + self.n_lags
        
        # Skip the last minute (need next minute's return)
        end_idx = len(self.timestamps) - 1
        
        # Initialize storage for positions and forecasts
        minute_positions = {}
        
        # Process each minute
        for idx in tqdm(range(start_idx, end_idx)):
            minute = self.timestamps[idx]
            minute_forecasts = {}
            minute_selected = {}
            
            # Process each target stock
            for stock in self.target_stocks:
                # Run LASSO
                forecast, selected = self.run_lasso(stock, idx)
                
                # Store results
                minute_forecasts[stock] = forecast
                minute_selected[stock] = selected
            
            # Store forecasts and selected features
            self.forecasts[minute] = minute_forecasts
            self.selected_features[minute] = minute_selected
            
            # Calculate positions
            positions = {}
            
            for stock, forecast in minute_forecasts.items():
                # Get transaction cost
                spread = self.transaction_costs[stock]
                
                # Only take position if forecast exceeds spread
                if abs(forecast) > spread:
                    # Position size based on forecast strength
                    positions[stock] = np.sign(forecast) * abs(forecast) / spread
            
            # Store positions
            minute_positions[minute] = positions
        
        # Convert positions to DataFrame
        self.positions = pd.DataFrame(minute_positions).T
        
        # Fill NaN values with 0 (no position)
        self.positions = self.positions.fillna(0)
        
        print(f"Executed strategy with positions for {len(self.positions.columns)} stocks")
    
    def calculate_returns(self):
        """
        Calculate strategy returns
        """
        print("Calculating returns...")
        
        # Initialize returns DataFrame
        returns_df = pd.DataFrame(index=self.positions.index, columns=['strategy_return'])
        returns_df['strategy_return'] = 0.0
        
        # For each minute with positions
        for i, minute in enumerate(self.positions.index):
            if i >= len(self.positions) - 1:
                continue  # Skip last minute (no next minute return)
                
            # Get positions for current minute
            positions = self.positions.loc[minute]
            
            # Get next minute
            next_minute = self.positions.index[i + 1]
            
            # Get actual returns for next minute
            actual_returns = {}
            
            for stock in positions.index:
                if stock in self.returns_df.columns:
                    # Get return for next minute
                    actual_returns[stock] = self.returns_df.loc[next_minute, stock]
            
            # Calculate strategy return
            strategy_return = 0.0
            total_position = 0.0
            
            for stock, position in positions.items():
                if position != 0 and stock in actual_returns:
                    # Get actual return
                    actual_return = actual_returns[stock]
                    
                    # Get transaction cost
                    spread = self.transaction_costs[stock]
                    
                    # Calculate net return
                    net_return = actual_return - spread * np.sign(position)
                    
                    # Add to strategy return
                    strategy_return += position * net_return
                    total_position += abs(position)
            
            # Normalize by total position
            if total_position > 0:
                strategy_return /= total_position
                returns_df.loc[minute, 'strategy_return'] = strategy_return
        
        # Calculate cumulative returns
        returns_df['cumulative_return'] = (1 + returns_df['strategy_return']).cumprod() - 1
        
        # Store returns
        self.returns = returns_df
        
        print("Returns calculated")
    
    def analyze_predictors(self):
        """
        Analyze the predictors selected by the LASSO
        """
        print("Analyzing predictors...")
        
        # Count predictors per forecast
        predictor_counts = {}
        
        for minute, selected in self.selected_features.items():
            for stock, features in selected.items():
                count = len(features)
                
                if count not in predictor_counts:
                    predictor_counts[count] = 0
                predictor_counts[count] += 1
        
        # Calculate average
        if predictor_counts:
            total_count = sum(predictor_counts.values())
            avg_predictors = sum(k * v for k, v in predictor_counts.items()) / total_count
        else:
            avg_predictors = 0
        
        # Track predictor duration
        predictor_tracking = {}
        duration_counts = {}
        
        # Get sorted minutes
        minutes = sorted(self.selected_features.keys())
        
        # For each minute
        for minute in minutes:
            selected = self.selected_features[minute]
            
            # For each stock
            for stock, features in selected.items():
                # For each selected feature
                for feature_idx in features:
                    # Create predictor ID
                    predictor_id = f"{stock}_{feature_idx}"
                    
                    # Initialize tracking if first appearance
                    if predictor_id not in predictor_tracking:
                        predictor_tracking[predictor_id] = {
                            'start_minute': minute,
                            'duration': 1,
                            'active': True
                        }
                    elif predictor_tracking[predictor_id]['active']:
                        # Increment duration if already active
                        predictor_tracking[predictor_id]['duration'] += 1
            
            # Check for predictors that are no longer active
            for pred_id, pred_data in predictor_tracking.items():
                if pred_data['active']:
                    # Check if this predictor is still active
                    stock, feature_idx = pred_id.split('_')
                    feature_idx = int(feature_idx)
                    
                    is_active = False
                    if stock in selected:
                        if feature_idx in selected[stock]:
                            is_active = True
                    
                    # If no longer active, update duration counts
                    if not is_active:
                        pred_data['active'] = False
                        duration = pred_data['duration']
                        
                        if duration not in duration_counts:
                            duration_counts[duration] = 0
                        duration_counts[duration] += 1
        
        # Return results
        results = {
            'avg_predictors': avg_predictors,
            'predictor_counts': predictor_counts,
            'duration_counts': duration_counts
        }
        
        print(f"Average number of predictors: {avg_predictors:.2f}")
        
        return results
    
    def plot_results(self, analysis_results):
        """
        Plot strategy results
        
        Parameters:
        -----------
        analysis_results : dict
            Dictionary with predictor analysis results
        """
        # Create figure
        plt.figure(figsize=(15, 12))
        
        # Plot cumulative returns
        plt.subplot(3, 1, 1)
        plt.plot(self.returns.index, self.returns['cumulative_return'] * 100)
        plt.title('Cumulative Strategy Returns (%)')
        plt.grid(True, alpha=0.3)
        
        # Plot histogram of number of predictors
        plt.subplot(3, 1, 2)
        predictor_counts = analysis_results['predictor_counts']
        
        if predictor_counts:
            x = list(predictor_counts.keys())
            y = list(predictor_counts.values())
            
            plt.bar(x, y)
            plt.axvline(
                analysis_results['avg_predictors'], 
                color='r', 
                linestyle='--',
                label=f"Average: {analysis_results['avg_predictors']:.2f}"
            )
            plt.title('Number of Predictors Selected by LASSO')
            plt.xlabel('Number of Predictors')
            plt.ylabel('Count')
            plt.legend()
        else:
            plt.title('No predictor data available')
        
        # Plot histogram of predictor duration
        plt.subplot(3, 1, 3)
        duration_counts = analysis_results['duration_counts']
        
        if duration_counts:
            x = list(duration_counts.keys())
            y = list(duration_counts.values())
            
            plt.bar(x, y)
            plt.title('Duration of Predictors (Minutes)')
            plt.xlabel('Duration (Minutes)')
            plt.ylabel('Count')
            
            if len(x) > 1:
                plt.xscale('log')
        else:
            plt.title('No duration data available')
        
        plt.tight_layout()
        plt.show()
    
    def run(self):
        """
        Run the full strategy pipeline
        """
        # Execute strategy
        self.execute_strategy()
        
        # Calculate returns
        self.calculate_returns()
        
        # Analyze predictors
        analysis_results = self.analyze_predictors()
        
        # Plot results
        self.plot_results(analysis_results)
        
        # Print performance summary
        if not self.returns.empty:
            total_return = self.returns['cumulative_return'].iloc[-1] * 100
            annualized_factor = 252 * 390 / len(self.returns)  # Assuming ~390 minutes per day
            annualized_return = ((1 + total_return/100) ** annualized_factor) - 1
            volatility = self.returns['strategy_return'].std() * 100
            annualized_vol = volatility * np.sqrt(annualized_factor)
            sharpe_ratio = annualized_return * 100 / annualized_vol if annualized_vol > 0 else 0
            
            print("\nStrategy Performance Summary:")
            print(f"Total Return: {total_return:.2f}%")
            print(f"Annualized Return: {annualized_return*100:.2f}%")
            print(f"Annualized Volatility: {annualized_vol:.2f}%")
            print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
            print(f"Average Number of Predictors: {analysis_results['avg_predictors']:.2f}")


# Run the strategy
if __name__ == "__main__":
    # Create and run strategy
    strategy = SparseSignalsStrategy(
        n_stocks=40,
        n_minutes=120,
        estimation_window=30,
        n_lags=3
    )
    
    strategy.run()