"""
Bloomberg Deep Hedging Strategy

This script implements a deep hedging strategy using Bloomberg data.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import tensorflow as tf
from tensorflow.keras.models import load_model

# Import the deep hedging model
from deep_hedging_model import DeepHedgingModel

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

def prepare_data_for_model(data, asset_ticker, hedge_ticker, lookback=10, feature_dim=5):
    """
    Prepare data for the deep hedging model

    Parameters:
    -----------
    data : pandas.DataFrame
        Bloomberg data
    asset_ticker : str
        Bloomberg ticker for the asset
    hedge_ticker : str
        Bloomberg ticker for the hedging instrument
    lookback : int
        Lookback period for the model
    feature_dim : int
        Number of features to use (must match model's expected input)

    Returns:
    --------
    tuple
        (X, dates)
    """
    # Extract price data
    asset_prices = data[(asset_ticker, 'PX_LAST')]
    hedge_prices = data[(hedge_ticker, 'PX_LAST')]

    # Calculate returns
    asset_returns = asset_prices.pct_change().fillna(0)
    hedge_returns = hedge_prices.pct_change().fillna(0)

    # Extract volatility data
    if (asset_ticker, 'VOLATILITY_30D') in data.columns:
        asset_vol = data[(asset_ticker, 'VOLATILITY_30D')]
    else:
        # Calculate rolling volatility
        asset_vol = asset_returns.rolling(30).std() * np.sqrt(252)
        asset_vol = asset_vol.fillna(asset_returns.std() * np.sqrt(252))

    if (hedge_ticker, 'VOLATILITY_30D') in data.columns:
        hedge_vol = data[(hedge_ticker, 'VOLATILITY_30D')]
    else:
        # Calculate rolling volatility
        hedge_vol = hedge_returns.rolling(30).std() * np.sqrt(252)
        hedge_vol = hedge_vol.fillna(hedge_returns.std() * np.sqrt(252))

    # Create features DataFrame
    features_df = pd.DataFrame({
        'asset_return': asset_returns,
        'hedge_return': hedge_returns,
        'asset_vol': asset_vol,
        'hedge_vol': hedge_vol,
        'asset_price': asset_prices,
        'hedge_price': hedge_prices
    })

    # Normalize features
    features_norm = features_df.copy()
    for col in features_norm.columns:
        mean = features_norm[col].mean()
        std = features_norm[col].std()
        features_norm[col] = (features_norm[col] - mean) / std

    # Select only the required number of features
    # Make sure we use the most important features first
    feature_priority = ['asset_return', 'hedge_return', 'asset_vol', 'hedge_vol', 'asset_price', 'hedge_price']
    selected_features = feature_priority[:feature_dim]

    # Create sequences
    X = []
    dates = []

    for i in range(lookback, len(features_norm)):
        X.append(features_norm[selected_features].iloc[i-lookback:i].values)
        dates.append(features_norm.index[i])

    X = np.array(X)

    return X, dates

def backtest_strategy(model, data, asset_ticker, hedge_ticker, lookback=10,
                      rebalance_freq='W', initial_capital=1000000, feature_dim=5):
    """
    Backtest the deep hedging strategy

    Parameters:
    -----------
    model : DeepHedgingModel
        Trained deep hedging model
    data : pandas.DataFrame
        Bloomberg data
    asset_ticker : str
        Bloomberg ticker for the asset
    hedge_ticker : str
        Bloomberg ticker for the hedging instrument
    lookback : int
        Lookback period for the model
    rebalance_freq : str
        Rebalancing frequency ('D' for daily, 'W' for weekly, 'M' for monthly)
    initial_capital : float
        Initial capital
    feature_dim : int
        Number of features to use (must match model's expected input)

    Returns:
    --------
    pandas.DataFrame
        Backtest results
    """
    # Prepare data for the model
    X, dates = prepare_data_for_model(data, asset_ticker, hedge_ticker, lookback, feature_dim)

    # Extract price and return data
    asset_prices = data[(asset_ticker, 'PX_LAST')].loc[dates]
    hedge_prices = data[(hedge_ticker, 'PX_LAST')].loc[dates]

    asset_returns = asset_prices.pct_change().fillna(0)
    hedge_returns = hedge_prices.pct_change().fillna(0)

    # Determine rebalance dates
    if rebalance_freq == 'D':
        rebalance_dates = asset_prices.index
    else:
        rebalance_dates = pd.date_range(start=asset_prices.index[0], end=asset_prices.index[-1], freq=rebalance_freq)

    # Initialize backtest variables
    capital = initial_capital
    asset_position = 0
    hedge_position = 0

    # Initialize results
    results = []

    # Run backtest
    for i, date in enumerate(dates):
        # Check if we need to rebalance
        if date in rebalance_dates or asset_position == 0:
            # Predict hedge ratio
            hedge_ratio = model.predict(X[i:i+1])[0][0]

            # Calculate positions
            asset_position = capital * 0.5 / asset_prices.loc[date]  # Invest half of capital in asset
            hedge_position = -asset_position * hedge_ratio  # Negative for short position

        # Calculate daily P&L
        if i > 0:
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
            'asset_return': asset_returns.loc[date],
            'hedge_return': hedge_returns.loc[date],
            'hedge_ratio': hedge_ratio if 'hedge_ratio' in locals() else 0
        })

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    results_df.set_index('date', inplace=True)

    # Calculate performance metrics
    results_df['daily_return'] = results_df['capital'].pct_change().fillna(0)
    results_df['cumulative_return'] = (1 + results_df['daily_return']).cumprod() - 1

    # Calculate annualized metrics
    trading_days = 252
    results_df['annualized_return'] = results_df['cumulative_return'].iloc[-1] * (trading_days / len(results_df))
    results_df['annualized_volatility'] = results_df['daily_return'].std() * np.sqrt(trading_days)

    # Calculate Sharpe ratio
    ann_vol = results_df['annualized_volatility'].iloc[-1]
    if ann_vol > 0:
        results_df['sharpe_ratio'] = results_df['annualized_return'].iloc[-1] / ann_vol
    else:
        results_df['sharpe_ratio'] = 0

    return results_df

def plot_results(results, title="Deep Hedging Strategy Results"):
    """
    Plot backtest results

    Parameters:
    -----------
    results : pandas.DataFrame
        Backtest results
    title : str
        Plot title
    """
    plt.figure(figsize=(12, 10))

    # Plot equity curve
    plt.subplot(3, 1, 1)
    plt.plot(results.index, results['capital'])
    plt.title(f'{title} - Equity Curve')
    plt.xlabel('Date')
    plt.ylabel('Capital')
    plt.grid(True)

    # Plot hedge ratio
    plt.subplot(3, 1, 2)
    plt.plot(results.index, results['hedge_ratio'])
    plt.title('Hedge Ratio')
    plt.xlabel('Date')
    plt.ylabel('Ratio')
    plt.grid(True)

    # Plot asset and hedge positions
    plt.subplot(3, 1, 3)
    plt.plot(results.index, results['asset_position'], label='Asset Position')
    plt.plot(results.index, results['hedge_position'], label='Hedge Position')
    plt.title('Positions')
    plt.xlabel('Date')
    plt.ylabel('Position Size')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(f'{title.lower().replace(" ", "_")}.png')
    plt.close()

    # Plot additional performance metrics
    plt.figure(figsize=(12, 10))

    # Plot cumulative returns
    plt.subplot(3, 1, 1)
    plt.plot(results.index, results['cumulative_return'] * 100)
    plt.title('Cumulative Return (%)')
    plt.xlabel('Date')
    plt.ylabel('Return (%)')
    plt.grid(True)

    # Plot daily returns
    plt.subplot(3, 1, 2)
    plt.plot(results.index, results['daily_return'] * 100)
    plt.title('Daily Return (%)')
    plt.xlabel('Date')
    plt.ylabel('Return (%)')
    plt.grid(True)

    # Plot drawdown
    plt.subplot(3, 1, 3)
    peak = results['capital'].cummax()
    drawdown = (results['capital'] - peak) / peak * 100
    plt.plot(results.index, drawdown)
    plt.title('Drawdown (%)')
    plt.xlabel('Date')
    plt.ylabel('Drawdown (%)')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(f'{title.lower().replace(" ", "_")}_metrics.png')
    plt.close()

def main():
    """Main function"""
    print("Bloomberg Deep Hedging Strategy")

    # Define parameters
    asset_ticker = 'SPY US Equity'  # S&P 500 ETF
    hedge_ticker = 'SH US Equity'   # ProShares Short S&P500 ETF
    start_date = '2020-01-01'
    end_date = '2023-12-31'
    lookback = 10
    rebalance_freq = 'W'  # Weekly rebalancing

    # Initialize Bloomberg data fetcher
    data_fetcher = BloombergDataFetcher()

    # Fetch historical data
    print(f"\nFetching historical data for {asset_ticker} and {hedge_ticker}...")
    fields = ['PX_LAST', 'VOLATILITY_30D', 'VOLUME']
    data = data_fetcher.fetch_historical_data(
        [asset_ticker, hedge_ticker],
        fields,
        start_date,
        end_date
    )

    if data is None or data.empty:
        print("Failed to fetch data. Exiting.")
        return

    print(f"Data fetched successfully. Shape: {data.shape}")

    # Check if we have a pre-trained model
    model_path = 'deep_hedging_model.h5'
    if os.path.exists(model_path):
        print(f"\nLoading pre-trained model from {model_path}...")
        try:
            # Create a new model instance
            model = DeepHedgingModel(
                lookback_period=lookback,
                feature_dim=5,
                lstm_units=64,
                dense_units=32
            )
            # Load weights
            model.load_model(model_path)
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Training a new model...")
            model = None
    else:
        print("\nNo pre-trained model found. Training a new model...")
        model = None

    # If we don't have a model, train one
    if model is None:
        # Define feature dimension
        feature_dim = 5  # Must match the model's expected input dimension

        # Prepare data for training
        X, dates = prepare_data_for_model(data, asset_ticker, hedge_ticker, lookback, feature_dim)

        # Split into train and test sets
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]

        # Create target variable (simplified - using asset returns as target)
        asset_returns = data[(asset_ticker, 'PX_LAST')].loc[dates].pct_change().fillna(0)
        y_train = asset_returns.iloc[:train_size].values.reshape(-1, 1)
        y_test = asset_returns.iloc[train_size:].values.reshape(-1, 1)

        # Initialize model
        model = DeepHedgingModel(
            lookback_period=lookback,
            feature_dim=feature_dim,
            lstm_units=64,
            dense_units=32,
            learning_rate=0.001,
            lambda_reg=0.01,
            risk_aversion=1.0
        )

        # Train model
        print("\nTraining model...")
        history = model.train(
            X_train, y_train,
            epochs=50,
            batch_size=32,
            validation_split=0.2,
            verbose=1
        )

        # Plot training history
        model.plot_training_history(history)

        # Save model
        model.save_model(model_path)
        print(f"Model saved to {model_path}")

    # Backtest the strategy
    print("\nBacktesting the strategy...")
    results = backtest_strategy(
        model=model,
        data=data,
        asset_ticker=asset_ticker,
        hedge_ticker=hedge_ticker,
        lookback=lookback,
        rebalance_freq=rebalance_freq,
        feature_dim=5  # Must match the model's expected input dimension
    )

    # Plot results
    plot_results(results, title=f"Deep Hedging Strategy - {asset_ticker}")

    # Print summary statistics
    print("\nSummary Statistics:")
    print(f"Final Capital: ${results['capital'].iloc[-1]:,.2f}")
    print(f"Total Return: {results['cumulative_return'].iloc[-1]:.2%}")
    print(f"Annualized Return: {results['annualized_return'].iloc[-1]:.2%}")
    print(f"Annualized Volatility: {results['annualized_volatility'].iloc[-1]:.2%}")
    print(f"Sharpe Ratio: {results['sharpe_ratio'].iloc[-1]:.2f}")

    # Calculate drawdown
    peak = results['capital'].cummax()
    drawdown = (results['capital'] - peak) / peak
    max_drawdown = drawdown.min()

    print(f"Maximum Drawdown: {max_drawdown:.2%}")

    # Compare with buy-and-hold strategy
    print("\nComparing with buy-and-hold strategy...")

    # Calculate buy-and-hold returns
    asset_prices = data[(asset_ticker, 'PX_LAST')].loc[results.index]
    buy_hold_return = (asset_prices.iloc[-1] / asset_prices.iloc[0]) - 1
    buy_hold_annual_return = buy_hold_return * (252 / len(asset_prices))
    buy_hold_volatility = asset_prices.pct_change().std() * np.sqrt(252)
    buy_hold_sharpe = buy_hold_annual_return / buy_hold_volatility

    print(f"Buy-and-Hold {asset_ticker}:")
    print(f"Total Return: {buy_hold_return:.2%}")
    print(f"Annualized Return: {buy_hold_annual_return:.2%}")
    print(f"Annualized Volatility: {buy_hold_volatility:.2%}")
    print(f"Sharpe Ratio: {buy_hold_sharpe:.2f}")

    # Calculate outperformance
    outperformance = results['annualized_return'].iloc[-1] - buy_hold_annual_return
    print(f"\nStrategy Outperformance: {outperformance:.2%} per year")

if __name__ == "__main__":
    main()
