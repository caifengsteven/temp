"""
Simplified LightGBM Model Implementation

This module provides a simplified version of the LightGBM model for stock price prediction
that avoids dependency issues with Dask and PyYAML.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from feature_engineering import DataTransformer

# Try to import LightGBM with a fallback to scikit-learn's GradientBoostingRegressor
try:
    # Directly import the core LightGBM C API to avoid Dask dependency issues
    import lightgbm.basic as lgb_basic
    import lightgbm.sklearn as lgb_sklearn
    LIGHTGBM_AVAILABLE = True
except ImportError:
    from sklearn.ensemble import GradientBoostingRegressor
    warnings.warn("LightGBM not available, using scikit-learn's GradientBoostingRegressor instead.")
    LIGHTGBM_AVAILABLE = False


class SimpleLightGBMForecaster:
    """Class for forecasting stock prices using LightGBM or GradientBoostingRegressor."""
    
    def __init__(self, target_transform='log_returns', ema_period=14):
        """
        Initialize the forecaster.
        
        Parameters:
        -----------
        target_transform : str
            Transformation method for the target variable.
            Options: 'returns', 'log_returns', 'ema_ratio', 'standardized_returns',
                    'standardized_log_returns', 'standardized_ema_ratio', 'ema_difference_ratio'
        ema_period : int
            Period for EMA calculation if using EMA-based transformations
        """
        self.target_transform = target_transform
        self.ema_period = ema_period
        self.model = None
        self.feature_importance = None
        self.transformer = DataTransformer()
        self.inverse_transform_func = None
    
    def _transform_target(self, data, column='close'):
        """
        Transform the target variable using the specified method.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            DataFrame containing the data
        column : str
            Column to transform
            
        Returns:
        --------
        pandas.Series
            Transformed target variable
        """
        if self.target_transform == 'returns':
            return self.transformer.returns(data, column)
        elif self.target_transform == 'log_returns':
            return self.transformer.log_returns(data, column)
        elif self.target_transform == 'ema_ratio':
            return self.transformer.ema_ratio(data, column, self.ema_period)
        elif self.target_transform == 'standardized_returns':
            return self.transformer.standardized_returns(data, column)
        elif self.target_transform == 'standardized_log_returns':
            return self.transformer.standardized_log_returns(data, column)
        elif self.target_transform == 'standardized_ema_ratio':
            return self.transformer.standardized_ema_ratio(data, column, self.ema_period)
        elif self.target_transform == 'ema_difference_ratio':
            return self.transformer.ema_difference_ratio(data, column, self.ema_period)
        else:
            raise ValueError(f"Unknown transformation method: {self.target_transform}")
    
    def _setup_inverse_transform(self, data, column='close'):
        """
        Set up the inverse transformation function based on the target transformation method.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            DataFrame containing the data
        column : str
            Column that was transformed
        """
        if self.target_transform == 'returns':
            # For returns: next_price = current_price * (1 + return)
            self.inverse_transform_func = lambda x, current: current * (1 + x)
        elif self.target_transform == 'log_returns':
            # For log returns: next_price = current_price * exp(log_return)
            self.inverse_transform_func = lambda x, current: current * np.exp(x)
        elif self.target_transform == 'ema_ratio':
            # For EMA ratio: next_price = ema * ratio
            ema = pd.Series(
                data=np.nan, 
                index=data.index, 
                name=f'ema_{self.ema_period}'
            )
            ema.loc[:] = self._calculate_ema(data[column], self.ema_period)
            self.inverse_transform_func = lambda x, idx: ema.loc[idx] * x
        elif self.target_transform == 'ema_difference_ratio':
            # For EMA difference ratio: next_price = ema * (1 + ratio)
            ema = pd.Series(
                data=np.nan, 
                index=data.index, 
                name=f'ema_{self.ema_period}'
            )
            ema.loc[:] = self._calculate_ema(data[column], self.ema_period)
            self.inverse_transform_func = lambda x, idx: ema.loc[idx] * (1 + x)
        elif self.target_transform.startswith('standardized_'):
            # For standardized transformations, we need the rolling mean and std
            # For simplicity, we'll just predict the direction
            self.inverse_transform_func = lambda x, current: current * (1 + np.sign(x) * 0.01)
    
    def _calculate_ema(self, series, period):
        """Calculate EMA for a series."""
        try:
            import talib
            return talib.EMA(series.values, timeperiod=period)
        except ImportError:
            # Fallback to pandas EMA if talib is not available
            return series.ewm(span=period, adjust=False).mean().values
    
    def train(self, data, target_column='close', test_size=0.2, random_state=42, 
              early_stopping_rounds=50, verbose=True):
        """
        Train the model.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            DataFrame containing features and target
        target_column : str
            Column to predict
        test_size : float
            Proportion of data to use for testing
        random_state : int
            Random seed for reproducibility
        early_stopping_rounds : int
            Number of rounds with no improvement to stop training
        verbose : bool
            Whether to print training progress
            
        Returns:
        --------
        dict
            Dictionary containing evaluation metrics
        """
        # Transform the target
        y = self._transform_target(data, target_column)
        
        # Set up inverse transform function
        self._setup_inverse_transform(data, target_column)
        
        # Remove the target column and any columns with NaN values
        X = data.drop(columns=[target_column])
        
        # Combine X and y and drop rows with NaN
        combined = pd.concat([X, y.rename('target')], axis=1)
        combined = combined.dropna()
        
        # Split back into X and y
        y = combined['target']
        X = combined.drop(columns=['target'])
        
        # Time series split
        n = len(X)
        train_size = int(n * (1 - test_size))
        
        X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
        y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]
        
        # Train the model
        if LIGHTGBM_AVAILABLE:
            # Use LightGBM's sklearn interface to avoid Dask dependency issues
            params = {
                'objective': 'regression',
                'metric': 'rmse',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.9,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbose': -1,
                'random_state': random_state
            }
            
            self.model = lgb_sklearn.LGBMRegressor(**params)
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_train, y_train), (X_test, y_test)],
                eval_names=['train', 'valid'],
                eval_metric='rmse',
                early_stopping_rounds=early_stopping_rounds,
                verbose=10 if verbose else 0
            )
            
            # Get feature importance
            self.feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
        else:
            # Use scikit-learn's GradientBoostingRegressor as a fallback
            params = {
                'n_estimators': 100,
                'learning_rate': 0.05,
                'max_depth': 5,
                'min_samples_split': 5,
                'random_state': random_state
            }
            
            self.model = GradientBoostingRegressor(**params)
            self.model.fit(X_train, y_train)
            
            # Get feature importance
            self.feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
        
        # Make predictions
        y_pred_train = self.model.predict(X_train)
        y_pred_test = self.model.predict(X_test)
        
        # Calculate metrics
        train_metrics = self._calculate_metrics(y_train, y_pred_train)
        test_metrics = self._calculate_metrics(y_test, y_pred_test)
        
        # Calculate directional accuracy
        train_dir_acc = self._directional_accuracy(y_train, y_pred_train)
        test_dir_acc = self._directional_accuracy(y_test, y_pred_test)
        
        train_metrics['directional_accuracy'] = train_dir_acc
        test_metrics['directional_accuracy'] = test_dir_acc
        
        if verbose:
            print(f"Train metrics: {train_metrics}")
            print(f"Test metrics: {test_metrics}")
            print(f"\nTop 10 important features:")
            print(self.feature_importance.head(10))
        
        return {
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'feature_importance': self.feature_importance
        }
    
    def predict(self, data, target_column='close'):
        """
        Make predictions using the trained model.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            DataFrame containing features
        target_column : str
            Column to predict
            
        Returns:
        --------
        pandas.Series
            Predicted values in the original scale
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Make a copy to avoid modifying the original data
        data_copy = data.copy()
        
        # Transform the target for reference (if needed for inverse transform)
        if target_column in data_copy.columns:
            y_transformed = self._transform_target(data_copy, target_column)
            
            # Set up inverse transform function if not already set
            if self.inverse_transform_func is None:
                self._setup_inverse_transform(data_copy, target_column)
        
        # Remove the target column if it exists
        if target_column in data_copy.columns:
            X = data_copy.drop(columns=[target_column])
        else:
            X = data_copy
        
        # Make predictions (in transformed space)
        predictions_transformed = self.model.predict(X)
        
        # Convert predictions back to original scale
        if self.target_transform in ['returns', 'log_returns']:
            # For returns and log returns, we need the current price
            current_prices = data_copy[target_column].shift(1)
            predictions_original = pd.Series(
                [self.inverse_transform_func(pred, curr) for pred, curr in zip(predictions_transformed, current_prices)],
                index=X.index
            )
        elif self.target_transform in ['ema_ratio', 'ema_difference_ratio']:
            # For EMA-based transformations, we need the index
            predictions_original = pd.Series(
                [self.inverse_transform_func(pred, idx) for pred, idx in zip(predictions_transformed, X.index)],
                index=X.index
            )
        else:
            # For standardized transformations, we use the sign-based approach
            current_prices = data_copy[target_column].shift(1)
            predictions_original = pd.Series(
                [self.inverse_transform_func(pred, curr) for pred, curr in zip(predictions_transformed, current_prices)],
                index=X.index
            )
        
        return predictions_original
    
    def _calculate_metrics(self, y_true, y_pred):
        """Calculate regression metrics."""
        return {
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred)
        }
    
    def _directional_accuracy(self, y_true, y_pred):
        """Calculate directional accuracy (percentage of correct direction predictions)."""
        # For transformed data, the sign indicates the direction
        correct_direction = np.sign(y_true) == np.sign(y_pred)
        return np.mean(correct_direction)
    
    def plot_feature_importance(self, top_n=20):
        """Plot feature importance."""
        if self.feature_importance is None:
            raise ValueError("Model not trained. Call train() first.")
        
        plt.figure(figsize=(10, 8))
        sns.barplot(
            x='importance',
            y='feature',
            data=self.feature_importance.head(top_n)
        )
        plt.title(f'Top {top_n} Feature Importance')
        plt.tight_layout()
        plt.show()
    
    def plot_predictions(self, data, target_column='close', test_size=0.2):
        """Plot actual vs predicted values."""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Make predictions
        predictions = self.predict(data, target_column)
        
        # Get actual values
        actuals = data[target_column]
        
        # Split into train and test for visualization
        n = len(data)
        train_size = int(n * (1 - test_size))
        
        train_actuals = actuals.iloc[:train_size]
        test_actuals = actuals.iloc[train_size:]
        
        train_preds = predictions.iloc[:train_size]
        test_preds = predictions.iloc[train_size:]
        
        # Plot
        plt.figure(figsize=(12, 6))
        plt.plot(train_actuals.index, train_actuals, label='Train Actual', color='blue')
        plt.plot(train_preds.index, train_preds, label='Train Predicted', color='lightblue', linestyle='--')
        plt.plot(test_actuals.index, test_actuals, label='Test Actual', color='green')
        plt.plot(test_preds.index, test_preds, label='Test Predicted', color='lightgreen', linestyle='--')
        
        plt.axvline(x=train_actuals.index[-1], color='red', linestyle='-', label='Train/Test Split')
        plt.title('Actual vs Predicted Values')
        plt.xlabel('Date')
        plt.ylabel(target_column)
        plt.legend()
        plt.tight_layout()
        plt.show()


# Example usage
if __name__ == "__main__":
    # Sample data
    dates = pd.date_range(start='2020-01-01', periods=100, freq='B')
    np.random.seed(42)
    
    # Create a trending series with some noise
    trend = np.linspace(100, 150, 100)
    noise = np.random.normal(0, 5, 100)
    close = trend + noise
    
    data = pd.DataFrame({
        'open': close - np.random.normal(0, 1, 100),
        'high': close + np.random.normal(2, 1, 100),
        'low': close - np.random.normal(2, 1, 100),
        'close': close,
        'volume': np.random.lognormal(15, 1, 100).astype(int),
        'feature1': np.random.normal(0, 1, 100),
        'feature2': np.random.normal(0, 1, 100),
        'feature3': np.random.normal(0, 1, 100)
    }, index=dates)
    
    # Ensure high >= open, close, low and low <= open, close
    data['high'] = data[['high', 'open', 'close']].max(axis=1)
    data['low'] = data[['low', 'open', 'close']].min(axis=1)
    
    # Train the model
    forecaster = SimpleLightGBMForecaster(target_transform='log_returns')
    results = forecaster.train(data, verbose=True)
    
    # Plot results
    forecaster.plot_feature_importance()
    forecaster.plot_predictions(data)
