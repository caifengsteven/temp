"""
Bloomberg Deep Hedging Strategy Implementation

This script implements a deep hedging strategy using Bloomberg data.
It demonstrates how to:
1. Connect to Bloomberg API
2. Fetch market data
3. Implement a basic hedging strategy
4. Backtest the strategy
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.optimizers import Adam

# Try to import Bloomberg API libraries
try:
    import blpapi
    from xbbg import blp
    BLOOMBERG_AVAILABLE = True
    print("Bloomberg API available")
except ImportError:
    BLOOMBERG_AVAILABLE = False
    print("Bloomberg API not available. Using mock data for testing.")

class BloombergDataFetcher:
    """Class to fetch data from Bloomberg"""
    
    def __init__(self):
        self.connected = BLOOMBERG_AVAILABLE
    
    def fetch_historical_data(self, tickers, fields, start_date, end_date):
        """
        Fetch historical data from Bloomberg
        
        Parameters:
        -----------
        tickers : list
            List of Bloomberg tickers
        fields : list
            List of Bloomberg fields
        start_date : str
            Start date in format 'YYYY-MM-DD'
        end_date : str
            End date in format 'YYYY-MM-DD'
            
        Returns:
        --------
        pandas.DataFrame
            Historical data
        """
        if not self.connected:
            # Generate mock data for testing
            dates = pd.date_range(start=start_date, end=end_date, freq='B')
            data = {}
            
            for ticker in tickers:
                ticker_data = {}
                for field in fields:
                    if field == 'PX_LAST':
                        # Generate random price series with trend and volatility
                        prices = 100 + np.cumsum(np.random.normal(0.0005, 0.01, len(dates)))
                        ticker_data[field] = prices
                    elif field == 'VOLATILITY_30D':
                        # Generate random volatility
                        vol = np.abs(np.random.normal(0.2, 0.05, len(dates)))
                        ticker_data[field] = vol
                    else:
                        # Generate random data for other fields
                        ticker_data[field] = np.random.normal(0, 1, len(dates))
                
                df = pd.DataFrame(ticker_data, index=dates)
                data[ticker] = df
            
            # Combine all tickers into a single DataFrame
            result = pd.concat(data, axis=1)
            return result
        
        try:
            # Use xbbg to fetch data
            data = blp.bdh(tickers=tickers, flds=fields, 
                          start_date=start_date, end_date=end_date)
            return data
        except Exception as e:
            print(f"Error fetching data from Bloomberg: {e}")
            return None

class DeepHedgingModel:
    """Deep Learning model for hedging"""
    
    def __init__(self, lookback_period=10, features=5, units=64, learning_rate=0.001):
        self.lookback_period = lookback_period
        self.features = features
        self.units = units
        self.learning_rate = learning_rate
        self.model = self._build_model()
        
    def _build_model(self):
        """Build the deep learning model"""
        model = Sequential([
            LSTM(self.units, return_sequences=True, input_shape=(self.lookback_period, self.features)),
            Dropout(0.2),
            LSTM(self.units, return_sequences=False),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(1)  # Output is the hedge ratio
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='mse'
        )
        
        return model
    
    def prepare_data(self, data, target_col, feature_cols):
        """
        Prepare data for LSTM model
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Input data
        target_col : str
            Target column name
        feature_cols : list
            List of feature column names
            
        Returns:
        --------
        tuple
            (X_train, y_train, X_test, y_test)
        """
        # Ensure all data is numeric
        for col in feature_cols + [target_col]:
            data[col] = pd.to_numeric(data[col], errors='coerce')
        
        # Drop rows with NaN values
        data = data.dropna()
        
        # Normalize data
        data_norm = data.copy()
        for col in feature_cols + [target_col]:
            mean = data[col].mean()
            std = data[col].std()
            data_norm[col] = (data[col] - mean) / std
        
        # Create sequences
        X, y = [], []
        for i in range(len(data_norm) - self.lookback_period):
            X.append(data_norm[feature_cols].iloc[i:i+self.lookback_period].values)
            y.append(data_norm[target_col].iloc[i+self.lookback_period])
        
        X = np.array(X)
        y = np.array(y)
        
        # Split into train and test sets (80% train, 20% test)
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        return X_train, y_train, X_test, y_test
    
    def train(self, X_train, y_train, epochs=50, batch_size=32, validation_split=0.2):
        """Train the model"""
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=1
        )
        return history
    
    def predict(self, X):
        """Make predictions"""
        return self.model.predict(X)
    
    def save_model(self, filepath):
        """Save the model"""
        self.model.save(filepath)
    
    def load_model(self, filepath):
        """Load the model"""
        self.model = tf.keras.models.load_model(filepath)

class HedgingStrategy:
    """Hedging strategy implementation"""
    
    def __init__(self, data_fetcher, model=None):
        self.data_fetcher = data_fetcher
        self.model = model
        self.positions = {}
        self.pnl_history = []
        
    def calculate_hedge_ratio(self, data, method='deep_learning', lookback=20):
        """
        Calculate hedge ratio
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Market data
        method : str
            Method to calculate hedge ratio ('deep_learning', 'beta', 'minimum_variance')
        lookback : int
            Lookback period for beta calculation
            
        Returns:
        --------
        float
            Hedge ratio
        """
        if method == 'deep_learning' and self.model is not None:
            # Use the deep learning model to predict hedge ratio
            # This assumes data is already prepared for the model
            return self.model.predict(data)[0][0]
        
        elif method == 'beta':
            # Calculate beta-based hedge ratio
            asset_returns = data['asset_returns'].iloc[-lookback:]
            hedge_returns = data['hedge_returns'].iloc[-lookback:]
            
            # Calculate beta using covariance and variance
            beta = np.cov(asset_returns, hedge_returns)[0, 1] / np.var(hedge_returns)
            return beta
        
        elif method == 'minimum_variance':
            # Calculate minimum variance hedge ratio
            asset_returns = data['asset_returns'].iloc[-lookback:]
            hedge_returns = data['hedge_returns'].iloc[-lookback:]
            
            # Calculate hedge ratio using OLS regression
            X = hedge_returns.values.reshape(-1, 1)
            y = asset_returns.values
            
            # Add constant for intercept
            X = np.hstack([np.ones((X.shape[0], 1)), X])
            
            # Calculate coefficients using normal equation
            beta = np.linalg.inv(X.T @ X) @ X.T @ y
            return beta[1]  # Return slope coefficient
        
        else:
            # Default to 1:1 hedge ratio
            return 1.0
    
    def backtest(self, asset_ticker, hedge_ticker, start_date, end_date, 
                 rebalance_freq='W', method='deep_learning', initial_capital=1000000):
        """
        Backtest the hedging strategy
        
        Parameters:
        -----------
        asset_ticker : str
            Bloomberg ticker for the asset to hedge
        hedge_ticker : str
            Bloomberg ticker for the hedging instrument
        start_date : str
            Start date in format 'YYYY-MM-DD'
        end_date : str
            End date in format 'YYYY-MM-DD'
        rebalance_freq : str
            Rebalancing frequency ('D' for daily, 'W' for weekly, 'M' for monthly)
        method : str
            Method to calculate hedge ratio
        initial_capital : float
            Initial capital
            
        Returns:
        --------
        pandas.DataFrame
            Backtest results
        """
        # Fetch historical data
        fields = ['PX_LAST', 'VOLATILITY_30D']
        data = self.data_fetcher.fetch_historical_data(
            [asset_ticker, hedge_ticker], 
            fields, 
            start_date, 
            end_date
        )
        
        if data is None or data.empty:
            print("No data available for backtesting")
            return None
        
        # Calculate returns
        asset_prices = data[(asset_ticker, 'PX_LAST')]
        hedge_prices = data[(hedge_ticker, 'PX_LAST')]
        
        asset_returns = asset_prices.pct_change().dropna()
        hedge_returns = hedge_prices.pct_change().dropna()
        
        # Align the return series
        returns_df = pd.DataFrame({
            'asset_returns': asset_returns,
            'hedge_returns': hedge_returns
        }).dropna()
        
        # Initialize backtest variables
        capital = initial_capital
        asset_position = 0
        hedge_position = 0
        
        results = []
        rebalance_dates = pd.date_range(start=returns_df.index[0], end=returns_df.index[-1], freq=rebalance_freq)
        
        for date in returns_df.index:
            # Check if we need to rebalance
            if date in rebalance_dates or asset_position == 0:
                # Calculate hedge ratio
                lookback_data = returns_df.loc[:date].tail(20)  # Use last 20 observations
                hedge_ratio = self.calculate_hedge_ratio(lookback_data, method=method)
                
                # Calculate positions
                asset_position = capital * 0.5 / asset_prices.loc[date]  # Invest half of capital in asset
                hedge_position = -asset_position * hedge_ratio  # Negative for short position
            
            # Calculate daily P&L
            if date > returns_df.index[0]:
                asset_pnl = asset_position * asset_prices.loc[date] * asset_returns.loc[date]
                hedge_pnl = hedge_position * hedge_prices.loc[date] * hedge_returns.loc[date]
                daily_pnl = asset_pnl + hedge_pnl
                capital += daily_pnl
            
            # Record results
            results.append({
                'date': date,
                'capital': capital,
                'asset_position': asset_position,
                'hedge_position': hedge_position,
                'asset_price': asset_prices.loc[date],
                'hedge_price': hedge_prices.loc[date],
                'asset_return': asset_returns.loc[date] if date in asset_returns.index else 0,
                'hedge_return': hedge_returns.loc[date] if date in hedge_returns.index else 0,
                'hedge_ratio': hedge_ratio if 'hedge_ratio' in locals() else 0
            })
        
        # Convert results to DataFrame
        results_df = pd.DataFrame(results)
        results_df.set_index('date', inplace=True)
        
        # Calculate performance metrics
        results_df['daily_return'] = results_df['capital'].pct_change()
        results_df['cumulative_return'] = (1 + results_df['daily_return']).cumprod() - 1
        
        # Calculate annualized metrics
        trading_days = 252
        results_df['annualized_return'] = results_df['cumulative_return'][-1] * (trading_days / len(results_df))
        results_df['annualized_volatility'] = results_df['daily_return'].std() * np.sqrt(trading_days)
        results_df['sharpe_ratio'] = results_df['annualized_return'] / results_df['annualized_volatility']
        
        return results_df
    
    def plot_results(self, results):
        """
        Plot backtest results
        
        Parameters:
        -----------
        results : pandas.DataFrame
            Backtest results
        """
        plt.figure(figsize=(12, 8))
        
        # Plot equity curve
        plt.subplot(2, 1, 1)
        plt.plot(results.index, results['capital'])
        plt.title('Equity Curve')
        plt.xlabel('Date')
        plt.ylabel('Capital')
        plt.grid(True)
        
        # Plot hedge ratio
        plt.subplot(2, 1, 2)
        plt.plot(results.index, results['hedge_ratio'])
        plt.title('Hedge Ratio')
        plt.xlabel('Date')
        plt.ylabel('Ratio')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('hedging_strategy_results.png')
        plt.close()
        
        # Print performance metrics
        print(f"Annualized Return: {results['annualized_return'].iloc[-1]:.2%}")
        print(f"Annualized Volatility: {results['annualized_volatility'].iloc[-1]:.2%}")
        print(f"Sharpe Ratio: {results['sharpe_ratio'].iloc[-1]:.2f}")

def main():
    """Main function to run the strategy"""
    # Initialize Bloomberg data fetcher
    data_fetcher = BloombergDataFetcher()
    
    # Define parameters
    asset_ticker = 'SPY US Equity'
    hedge_ticker = 'SH US Equity'  # ProShares Short S&P500 ETF
    start_date = '2020-01-01'
    end_date = '2023-12-31'
    
    # Initialize hedging strategy
    strategy = HedgingStrategy(data_fetcher)
    
    # Backtest the strategy using different methods
    methods = ['beta', 'minimum_variance']
    
    for method in methods:
        print(f"\nBacktesting with {method} method:")
        results = strategy.backtest(
            asset_ticker=asset_ticker,
            hedge_ticker=hedge_ticker,
            start_date=start_date,
            end_date=end_date,
            rebalance_freq='W',  # Weekly rebalancing
            method=method
        )
        
        if results is not None:
            strategy.plot_results(results)
    
    # If you want to use deep learning model
    if BLOOMBERG_AVAILABLE:
        print("\nTraining deep learning model...")
        # Fetch data for training
        fields = ['PX_LAST', 'VOLATILITY_30D', 'VOLUME', 'OPEN', 'HIGH', 'LOW']
        training_data = data_fetcher.fetch_historical_data(
            [asset_ticker, hedge_ticker], 
            fields, 
            '2018-01-01',  # Use longer history for training
            end_date
        )
        
        # Prepare features for deep learning
        # This is a simplified example - in practice, you would create more sophisticated features
        features = [
            (asset_ticker, 'PX_LAST'),
            (asset_ticker, 'VOLATILITY_30D'),
            (hedge_ticker, 'PX_LAST'),
            (hedge_ticker, 'VOLATILITY_30D'),
            (asset_ticker, 'VOLUME')
        ]
        
        # Create target variable (optimal hedge ratio using minimum variance method)
        # In practice, you might use a more sophisticated approach
        
        # Initialize and train the model
        model = DeepHedgingModel(lookback_period=10, features=len(features))
        
        # Use the trained model for hedging
        strategy_dl = HedgingStrategy(data_fetcher, model)
        
        print("\nBacktesting with deep learning method:")
        results_dl = strategy_dl.backtest(
            asset_ticker=asset_ticker,
            hedge_ticker=hedge_ticker,
            start_date=start_date,
            end_date=end_date,
            rebalance_freq='W',
            method='deep_learning'
        )
        
        if results_dl is not None:
            strategy_dl.plot_results(results_dl)

if __name__ == "__main__":
    main()
