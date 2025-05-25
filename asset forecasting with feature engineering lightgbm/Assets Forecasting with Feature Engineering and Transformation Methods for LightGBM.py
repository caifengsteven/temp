import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
import time
from datetime import datetime, timedelta
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class StockForecaster:
    """
    A robust implementation for stock price forecasting using gradient boosting
    with features from the original paper and error handling.
    """
    
    def __init__(self, symbol, start_date, end_date, transformation_method='log_returns'):
        """
        Initialize the forecaster.
        
        Parameters:
        -----------
        symbol : str
            The ticker symbol to forecast
        start_date : str
            Start date in YYYY-MM-DD format
        end_date : str
            End date in YYYY-MM-DD format
        transformation_method : str
            The transformation method to use for the target variable.
            Options: 'log_returns', 'returns', 'ema_ratio', 'ema_diff_ratio'
        """
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.transformation_method = transformation_method
        
        # Data containers
        self.raw_data = None
        self.features = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        
        # Model
        self.model = None
        self.feature_importance = None
        
        # Performance metrics
        self.metrics = {}
        
        # Constants
        self.EMA_PERIOD = 14
    
    def load_bloomberg_current_data(self):
        """
        Load current data from Bloomberg API using BDP which works correctly
        """
        try:
            # Try to import blpapi
            try:
                import blpapi
            except ImportError:
                print("Installing Bloomberg API...")
                import subprocess
                subprocess.check_call(["pip", "install", "--index-url=https://bcms.bloomberg.com/pip/simple", "blpapi"])
                import blpapi
            
            print(f"Loading current Bloomberg data for {self.symbol}...")
            
            # Setup Bloomberg API session
            session_options = blpapi.SessionOptions()
            session_options.setServerHost("localhost")
            session_options.setServerPort(8194)
            
            session = blpapi.Session(session_options)
            if not session.start():
                print("Failed to start Bloomberg API session.")
                return False
            
            if not session.openService("//blp/refdata"):
                print("Failed to open //blp/refdata service.")
                session.stop()
                return False
            
            # Get reference data service
            refdata = session.getService("//blp/refdata")
            
            # Create request for reference data
            request = refdata.createRequest("ReferenceDataRequest")
            
            # Add securities
            securities = request.getElement("securities")
            securities.appendValue(self.symbol)
            
            # Add fields for current data
            fields = request.getElement("fields")
            fields.appendValue("PX_OPEN")
            fields.appendValue("PX_HIGH")
            fields.appendValue("PX_LOW")
            fields.appendValue("PX_LAST")
            fields.appendValue("PX_VOLUME")
            
            # Send request
            print("Sending Bloomberg reference data request...")
            session.sendRequest(request)
            
            # Process response
            current_data = None
            
            while True:
                event = session.nextEvent(500)
                
                for msg in event:
                    if msg.messageType() == blpapi.Name("ReferenceDataResponse"):
                        # Extract data
                        securityData = msg.getElement("securityData")
                        
                        for i in range(securityData.numValues()):
                            security = securityData.getValue(i)
                            ticker = security.getElementAsString("security")
                            fieldData = security.getElement("fieldData")
                            
                            # Extract current prices
                            current_data = {
                                'date': datetime.now().strftime('%Y-%m-%d'),
                                'open': fieldData.getElementAsFloat("PX_OPEN") if fieldData.hasElement("PX_OPEN") else None,
                                'high': fieldData.getElementAsFloat("PX_HIGH") if fieldData.hasElement("PX_HIGH") else None,
                                'low': fieldData.getElementAsFloat("PX_LOW") if fieldData.hasElement("PX_LOW") else None,
                                'close': fieldData.getElementAsFloat("PX_LAST") if fieldData.hasElement("PX_LAST") else None,
                                'volume': fieldData.getElementAsFloat("PX_VOLUME") if fieldData.hasElement("PX_VOLUME") else None
                            }
                            
                            print(f"Current data for {ticker}:")
                            for k, v in current_data.items():
                                print(f"  {k}: {v}")
                
                if event.eventType() == blpapi.Event.RESPONSE:
                    break
            
            # Close the session
            session.stop()
            
            if current_data is None:
                print("No data received from Bloomberg.")
                return False
            
            # For demonstration purposes, we'll generate a synthetic time series
            # starting with the real current data
            print(f"Building synthetic time series based on current {self.symbol} data...")
            
            # Convert dates to datetime
            start = pd.to_datetime(self.start_date)
            end = pd.to_datetime(self.end_date)
            
            # Generate date range
            dates = pd.date_range(start=start, end=end, freq='B')
            
            # Set a random seed for reproducibility
            np.random.seed(42)
            
            # Use the current close price as the starting point
            if current_data['close'] is not None:
                price = current_data['close']
            else:
                price = 100.0  # Fallback if current price is not available
                
            # Generate synthetic prices with a random walk backward from current price
            prices = [price]
            
            # Generate in reverse (from current to past)
            for i in range(1, len(dates)):
                # Add some mean reversion and momentum to make it more realistic
                momentum = 0.03 * (prices[-1] / prices[-2] - 1) if i > 1 else 0
                mean_reversion = -0.01 * (price / prices[0] - 1)
                daily_return = np.random.normal(0, 0.01) + momentum + mean_reversion
                price = price / (1 + daily_return)  # Going backward in time
                prices.append(price)
            
            # Reverse to get chronological order
            prices.reverse()
            
            # Create DataFrame
            df = pd.DataFrame({
                'open': prices,
                'high': [p * (1 + abs(np.random.normal(0, 0.005))) for p in prices],
                'low': [p * (1 - abs(np.random.normal(0, 0.005))) for p in prices],
                'close': [p * (1 + np.random.normal(0, 0.002)) for p in prices],
                'volume': [max(0, int(np.random.normal(1e6, 2e5))) for _ in prices]
            }, index=dates)
            
            # Ensure high >= open, close and low <= open, close
            df['high'] = df[['open', 'close', 'high']].max(axis=1)
            df['low'] = df[['open', 'close', 'low']].min(axis=1)
            
            # Store raw data
            self.raw_data = df
            
            print(f"Successfully created synthetic time series based on current {self.symbol} data.")
            return True
            
        except Exception as e:
            print(f"Error loading Bloomberg data: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _generate_synthetic_data(self):
        """
        Generate synthetic data for testing when Bloomberg is unavailable
        """
        print(f"Generating synthetic data for {self.symbol}...")
        
        # Convert dates to datetime
        start = pd.to_datetime(self.start_date)
        end = pd.to_datetime(self.end_date)
        
        # Generate date range
        dates = pd.date_range(start=start, end=end, freq='B')
        
        # Set a random seed for reproducibility
        np.random.seed(42)
        
        # Generate synthetic prices with a random walk
        price = 100.0
        prices = [price]
        
        for i in range(1, len(dates)):
            # Add some mean reversion and momentum to make it more realistic
            momentum = 0.05 * (prices[-1] / prices[-2] - 1) if i > 1 else 0
            mean_reversion = -0.02 * (price / 100 - 1)
            daily_return = np.random.normal(0, 0.01) + momentum + mean_reversion
            price *= (1 + daily_return)
            prices.append(price)
        
        # Create DataFrame
        df = pd.DataFrame({
            'open': prices,
            'high': [p * (1 + abs(np.random.normal(0, 0.005))) for p in prices],
            'low': [p * (1 - abs(np.random.normal(0, 0.005))) for p in prices],
            'close': [p * (1 + np.random.normal(0, 0.002)) for p in prices],
            'volume': [max(0, int(np.random.normal(1e6, 2e5))) for _ in prices]
        }, index=dates)
        
        # Ensure high >= open, close and low <= open, close
        df['high'] = df[['open', 'close', 'high']].max(axis=1)
        df['low'] = df[['open', 'close', 'low']].min(axis=1)
        
        # Store raw data
        self.raw_data = df
        
        print("Synthetic data generated successfully.")
        print(f"Data range: {df.index.min()} to {df.index.max()}")
        print(f"Data sample:\n{df.head()}")
    
    def engineer_features(self):
        """
        Create features as described in the paper
        """
        print("Creating features...")
        
        # Create a copy of the raw data
        df = self.raw_data.copy()
        
        # Add previous day's data
        df['open_prev'] = df['open'].shift(1)
        df['high_prev'] = df['high'].shift(1)
        df['low_prev'] = df['low'].shift(1)
        df['close_prev'] = df['close'].shift(1)
        df['volume_prev'] = df['volume'].shift(1)
        
        # Calculate typical price
        df['typical'] = (df['high'] + df['low'] + df['close']) / 3
        df['typical_prev'] = df['typical'].shift(1)
        
        # Lag features
        for period in [1, 5, 30]:
            df[f'open_lag_{period}'] = df['open'].shift(period)
            df[f'close_lag_{period}'] = df['close'].shift(period)
            df[f'typical_lag_{period}'] = df['typical'].shift(period)
        
        # Moving Averages
        for window in [5, 14, 21]:
            # Simple Moving Averages
            df[f'close_sma_{window}'] = df['close'].rolling(window=window).mean()
            df[f'volume_sma_{window}'] = df['volume'].rolling(window=window).mean()
            
            # Exponential Moving Averages
            df[f'close_ema_{window}'] = df['close'].ewm(span=window, adjust=False).mean()
            
            # Rolling volatility
            df[f'close_std_{window}'] = df['close'].rolling(window=window).std()
            df[f'high_low_range_{window}'] = (df['high'] - df['low']).rolling(window=window).mean()
        
        # Technical indicators
        
        # RSI - Relative Strength Index
        delta = df['close'].diff()
        gain = delta.mask(delta < 0, 0)
        loss = -delta.mask(delta > 0, 0)
        
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        
        rs = avg_gain / avg_loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD - Moving Average Convergence Divergence
        ema_12 = df['close'].ewm(span=12, adjust=False).mean()
        ema_26 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # ATR - Average True Range
        high_low = df['high'] - df['low']
        high_close = (df['high'] - df['close'].shift()).abs()
        low_close = (df['low'] - df['close'].shift()).abs()
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr'] = true_range.rolling(14).mean()
        
        # Price momentum
        df['momentum_5d'] = df['close'] / df['close'].shift(5) - 1
        df['momentum_14d'] = df['close'] / df['close'].shift(14) - 1
        
        # Overnight gap
        df['gap'] = df['open'] - df['close_prev']
        df['gap_pct'] = df['gap'] / df['close_prev']
        
        # Slope differences
        def calculate_slope(series, period=14):
            if len(series) < period:
                return np.nan
            
            x = np.arange(period)
            y = series.values[-period:]
            slope, _, _, _, _ = stats.linregress(x, y)
            return slope
        
        # Calculate slopes with rolling windows
        for window in [14, 21]:
            df[f'close_slope_{window}'] = df['close'].rolling(window).apply(
                lambda x: calculate_slope(x, window), raw=False)
            df[f'volume_slope_{window}'] = df['volume'].rolling(window).apply(
                lambda x: calculate_slope(x, window), raw=False)
            df[f'rsi_slope_{window}'] = df['rsi'].rolling(window).apply(
                lambda x: calculate_slope(x, window), raw=False)
        
        # Cross Features
        
        # ATR to price ratio
        df['atr_price_ratio'] = df['atr'] / df['close']
        
        # Slope differences
        df['slope_diff_14'] = df['close_slope_14'] - df['volume_slope_14']
        df['slope_diff_21'] = df['close_slope_21'] - df['volume_slope_21']
        
        # Create EMA difference ratios (from paper)
        df['close_ema_diff_ratio'] = (df['close'] - df['close_prev']) / df['close_ema_14']
        df['open_ema_diff_ratio'] = (df['open'] - df['open_prev']) / df['close_ema_14']
        
        # Timestamp features - Day of week, month, etc.
        # Convert to cyclical features to better represent their periodic nature
        df['day_of_week'] = df.index.dayofweek
        df['day_of_month'] = df.index.day
        df['month'] = df.index.month
        
        df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_of_month_sin'] = np.sin(2 * np.pi * df['day_of_month'] / 31)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        
        # Clean the data by replacing inf with NaN (will be handled later)
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # Store the engineered features
        self.features = df
        
        print("Features created successfully.")
        return df
    
    def transform_target(self):
        """
        Apply the selected transformation method to the target variable
        """
        print(f"Applying {self.transformation_method} transformation...")
        
        df = self.features.copy()
        
        if self.transformation_method == 'log_returns':
            df['target'] = np.log(df['close'] / df['close_prev'])
            
        elif self.transformation_method == 'returns':
            df['target'] = df['close'] / df['close_prev'] - 1
            
        elif self.transformation_method == 'ema_ratio':
            df['target'] = df['close'] / df['close_ema_14']
            
        elif self.transformation_method == 'ema_diff_ratio':
            df['target'] = (df['close'] - df['close_prev']) / df['close_ema_14']
            
        else:
            raise ValueError(f"Unknown transformation method: {self.transformation_method}")
        
        # Store the transformed data
        self.features = df
        
        print("Target transformation completed.")
        return df
    
    def prepare_data(self, test_size=0.2):
        """
        Prepare data for training and testing with robust cleaning
        """
        print("Preparing data for modeling...")
        
        # Create a clean copy of the data
        df = self.features.copy()
        
        # Get column names before dropping any rows
        all_columns = df.columns.tolist()
        non_feature_cols = ['open', 'high', 'low', 'close', 'volume', 
                            'day_of_week', 'day_of_month', 'month', 'target']
        
        # Final check for any remaining NaN/inf values
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna()
        
        # Verify we still have data after cleaning
        if len(df) == 0:
            raise ValueError("No data remaining after cleaning. All rows had NaN values.")
        
        # Identify feature columns
        feature_cols = [col for col in all_columns if col not in non_feature_cols]
        X = df[feature_cols]
        y = df['target']
        
        # Time-based train-test split
        split_idx = int(len(X) * (1 - test_size))
        
        self.X_train = X.iloc[:split_idx]
        self.y_train = y.iloc[:split_idx]
        self.X_test = X.iloc[split_idx:]
        self.y_test = y.iloc[split_idx:]
        
        # Store price data for evaluation
        self.close_train = df['close'].iloc[:split_idx]
        self.close_test = df['close'].iloc[split_idx:]
        self.close_prev_test = df['close_prev'].iloc[split_idx:]
        
        print(f"Data prepared: {len(self.X_train)} training samples, {len(self.X_test)} testing samples")
    
    def train_model(self):
        """
        Train the GradientBoostingRegressor with cross-validation
        """
        print("Training model...")
        start_time = time.time()
        
        # Define parameter grid for a smaller grid search
        param_grid = [
            {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 3},
            {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 5},
            {'n_estimators': 200, 'learning_rate': 0.05, 'max_depth': 3},
            {'n_estimators': 200, 'learning_rate': 0.05, 'max_depth': 5}
        ]
        
        # Create time series cross-validation folds
        tscv = TimeSeriesSplit(n_splits=3)
        
        # Find best parameters with cross-validation
        best_score = float('-inf')
        best_params = None
        
        for params in param_grid:
            model = GradientBoostingRegressor(
                n_estimators=params['n_estimators'],
                learning_rate=params['learning_rate'],
                max_depth=params['max_depth'],
                random_state=42
            )
            
            scores = []
            for train_idx, valid_idx in tscv.split(self.X_train):
                X_fold_train, X_fold_valid = self.X_train.iloc[train_idx], self.X_train.iloc[valid_idx]
                y_fold_train, y_fold_valid = self.y_train.iloc[train_idx], self.y_train.iloc[valid_idx]
                
                try:
                    model.fit(X_fold_train, y_fold_train)
                    score = -mean_squared_error(y_fold_valid, model.predict(X_fold_valid))
                    scores.append(score)
                except Exception as e:
                    print(f"Error in training fold: {e}")
                    scores.append(float('-inf'))
            
            avg_score = np.mean(scores) if scores else float('-inf')
            
            if avg_score > best_score:
                best_score = avg_score
                best_params = params
        
        if best_params:
            print(f"Best parameters found: {best_params}")
            
            # Train final model
            self.model = GradientBoostingRegressor(
                n_estimators=best_params['n_estimators'],
                learning_rate=best_params['learning_rate'],
                max_depth=best_params['max_depth'],
                random_state=42
            )
            self.model.fit(self.X_train, self.y_train)
            
            # Get feature importance
            self.feature_importance = pd.DataFrame({
                'Feature': self.X_train.columns,
                'Importance': self.model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
        else:
            print("No valid parameters found. Using default parameters.")
            self.model = GradientBoostingRegressor(random_state=42)
            self.model.fit(self.X_train, self.y_train)
            
            # Get feature importance
            self.feature_importance = pd.DataFrame({
                'Feature': self.X_train.columns,
                'Importance': self.model.feature_importances_
            }).sort_values('Importance', ascending=False)
        
        training_time = time.time() - start_time
        self.metrics['training_time'] = training_time
        
        print(f"Model training completed in {training_time:.2f} seconds")
    
    def evaluate_model(self):
        """
        Evaluate the model using metrics described in the paper
        """
        print("Evaluating model performance...")
        
        # Make predictions on test data
        predictions = self.model.predict(self.X_test)
        
        # Reverse transform predictions based on the transformation method
        if self.transformation_method == 'log_returns':
            pred_close = self.close_prev_test * np.exp(predictions)
        elif self.transformation_method == 'returns':
            pred_close = self.close_prev_test * (1 + predictions)
        elif self.transformation_method == 'ema_ratio':
            ema_test = self.features.loc[self.X_test.index, 'close_ema_14']
            pred_close = ema_test * predictions
        elif self.transformation_method == 'ema_diff_ratio':
            ema_test = self.features.loc[self.X_test.index, 'close_ema_14']
            pred_close = self.close_prev_test + (ema_test * predictions)
        
        # Calculate metrics
        mae = mean_absolute_error(self.close_test, pred_close)
        rmse = np.sqrt(mean_squared_error(self.close_test, pred_close))
        
        # Directional Accuracy
        actual_direction = np.sign(self.close_test.values - self.close_prev_test.values)
        pred_direction = np.sign(pred_close - self.close_prev_test.values)
        da = np.mean(actual_direction == pred_direction) * 100
        
        # Store metrics
        self.metrics['mae'] = mae
        self.metrics['rmse'] = rmse
        self.metrics['directional_accuracy'] = da
        
        print(f"Model Performance:")
        print(f"MAE: {mae:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"Directional Accuracy: {da:.2f}%")
        
        # Compare with random walk benchmark
        rw_pred = self.close_prev_test.values
        rw_mae = mean_absolute_error(self.close_test, rw_pred)
        rw_rmse = np.sqrt(mean_squared_error(self.close_test, rw_pred))
        
        # Directional Accuracy for random walk is 50% by definition
        rw_da = 50.0
        
        # Calculate relative improvement
        rel_mae_improvement = (rw_mae - mae) / rw_mae * 100
        rel_rmse_improvement = (rw_rmse - rmse) / rw_rmse * 100
        rel_da_improvement = da - rw_da
        
        print(f"\nRelative improvement over Random Walk:")
        print(f"MAE improvement: {rel_mae_improvement:.2f}%")
        print(f"RMSE improvement: {rel_rmse_improvement:.2f}%")
        print(f"DA improvement: {rel_da_improvement:.2f}%")
        
        # Store benchmark comparisons
        self.metrics['rw_mae'] = rw_mae
        self.metrics['rw_rmse'] = rw_rmse
        self.metrics['rw_da'] = rw_da
        self.metrics['rel_mae_improvement'] = rel_mae_improvement
        self.metrics['rel_rmse_improvement'] = rel_rmse_improvement
        self.metrics['rel_da_improvement'] = rel_da_improvement
        
        # Store predictions for plotting
        self.predictions = pred_close
        
        return self.metrics
    
    def plot_results(self):
        """
        Plot the forecasting results and feature importance
        """
        if not hasattr(self, 'predictions'):
            print("No predictions available. Run evaluate_model first.")
            return
        
        # Create figure
        plt.figure(figsize=(14, 16))
        
        # 1. Plot actual vs predicted prices
        plt.subplot(3, 1, 1)
        plt.plot(self.close_test.index, self.close_test.values, label='Actual Close', color='blue')
        plt.plot(self.close_test.index, self.predictions, label='Predicted Close', color='red', linestyle='--')
        plt.title(f'{self.symbol} Price Forecasting - {self.transformation_method}')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)
        
        # 2. Plot residuals
        plt.subplot(3, 1, 2)
        residuals = self.close_test.values - self.predictions
        plt.plot(self.close_test.index, residuals, color='green')
        plt.axhline(y=0, color='black', linestyle='-')
        plt.title('Forecasting Residuals')
        plt.xlabel('Date')
        plt.ylabel('Residual')
        plt.grid(True)
        
        # 3. Plot top feature importance
        plt.subplot(3, 1, 3)
        n_features = min(15, len(self.feature_importance))
        top_features = self.feature_importance.head(n_features)
        
        plt.barh(top_features['Feature'], top_features['Importance'])
        plt.title('Top 15 Feature Importance')
        plt.xlabel('Importance Score')
        plt.tight_layout()
        
        plt.show()
    
    def run_pipeline(self, plot=True):
        """
        Run the complete forecasting pipeline
        """
        # Try to load Bloomberg data first
        if not self.load_bloomberg_current_data():
            print("Warning: Unable to load Bloomberg data, using synthetic data instead.")
            self._generate_synthetic_data()
        
        # Create features and transform target
        self.engineer_features()
        self.transform_target()
        self.prepare_data()
        self.train_model()
        self.evaluate_model()
        
        if plot:
            self.plot_results()
        
        return self.metrics


# Main function to test multiple transformation methods
def test_forecasting_methods(symbol, start_date, end_date, methods):
    """
    Test multiple forecasting methods and compare their performance
    
    Parameters:
    -----------
    symbol : str
        Bloomberg ticker symbol
    start_date : str
        Start date in YYYY-MM-DD format
    end_date : str
        End date in YYYY-MM-DD format
    methods : list
        List of transformation methods to test
    """
    results = {}
    
    for method in methods:
        print(f"\n{'='*50}")
        print(f"Testing {method} transformation method on {symbol}")
        print(f"{'='*50}\n")
        
        forecaster = StockForecaster(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            transformation_method=method
        )
        
        metrics = forecaster.run_pipeline(plot=True)
        results[method] = metrics
    
    # Compare results
    if len(results) > 1:
        print("\n\nResults Comparison:")
        print("-----------------")
        comparison_df = pd.DataFrame({
            method: {
                'MAE': results[method]['mae'],
                'RMSE': results[method]['rmse'],
                'Directional Accuracy (%)': results[method]['directional_accuracy'],
                'Training Time (s)': results[method]['training_time'],
                'Relative MAE Improvement (%)': results[method]['rel_mae_improvement'],
                'Relative RMSE Improvement (%)': results[method]['rel_rmse_improvement']
            } for method in results.keys()
        })
        
        print(comparison_df.round(4))
    
    return results


# Main execution block
if __name__ == "__main__":
    try:
        # Define ticker, date range, and transformation methods
        ticker = "AAPL US Equity"  # Bloomberg ticker format
        start_date = "2018-01-01"
        end_date = "2023-01-01"
        
        # Test different transformation methods
        transformation_methods = [
            'log_returns',  # Run only one method for testing first
        ]
        
        # Run forecasting
        results = test_forecasting_methods(
            ticker, 
            start_date, 
            end_date, 
            transformation_methods
        )
        
    except Exception as e:
        import traceback
        print(f"Error occurred: {e}")
        traceback.print_exc()