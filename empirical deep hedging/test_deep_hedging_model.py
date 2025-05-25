"""
Test Deep Hedging Model

This script tests the deep hedging model with synthetic data.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Import the deep hedging model
from deep_hedging_model import DeepHedgingModel, create_synthetic_data

def generate_realistic_market_data(start_date='2020-01-01', end_date='2023-12-31', seed=42):
    """
    Generate realistic market data for testing
    
    Parameters:
    -----------
    start_date : str
        Start date in format 'YYYY-MM-DD'
    end_date : str
        End date in format 'YYYY-MM-DD'
    seed : int
        Random seed
        
    Returns:
    --------
    pandas.DataFrame
        Market data
    """
    np.random.seed(seed)
    
    # Create date range
    dates = pd.date_range(start=start_date, end=end_date, freq='B')
    n_days = len(dates)
    
    # Generate asset price with realistic properties
    # - Positive drift (upward trend)
    # - Volatility clustering
    # - Fat tails
    
    # Parameters
    mu = 0.0005  # Daily drift (about 12% annual return)
    sigma = 0.01  # Base volatility (about 16% annual volatility)
    
    # Generate returns with volatility clustering
    z = np.random.normal(0, 1, n_days)  # Standard normal innovations
    
    # GARCH-like volatility process
    vol = np.zeros(n_days)
    vol[0] = sigma
    for t in range(1, n_days):
        vol[t] = 0.9 * vol[t-1] + 0.1 * abs(z[t-1]) * sigma
    
    # Generate returns
    returns = mu + vol * z
    
    # Generate prices
    asset_price = 100 * np.cumprod(1 + returns)
    
    # Generate correlated hedge instrument
    # - Negative correlation with asset
    # - Similar volatility characteristics
    
    # Generate correlated innovations
    rho = -0.8  # Correlation coefficient
    z_hedge = rho * z + np.sqrt(1 - rho**2) * np.random.normal(0, 1, n_days)
    
    # Generate hedge returns
    hedge_returns = -0.0002 + vol * z_hedge  # Slight negative drift
    
    # Generate hedge prices
    hedge_price = 50 * np.cumprod(1 + hedge_returns)
    
    # Generate additional features
    
    # Volume (with day-of-week effect)
    base_volume = 1000000
    day_of_week = np.array([d.weekday() for d in dates])
    day_factors = np.array([1.0, 1.1, 1.2, 1.1, 0.9])  # Mon-Fri factors
    volume = base_volume * day_factors[day_of_week] * (1 + 0.3 * np.random.normal(0, 1, n_days))
    
    # Volatility (20-day rolling standard deviation of returns)
    vol_20d = pd.Series(returns).rolling(20).std().fillna(sigma)
    
    # Momentum (20-day rolling mean of returns)
    mom_20d = pd.Series(returns).rolling(20).mean().fillna(mu)
    
    # Create DataFrame
    data = pd.DataFrame({
        'date': dates,
        'asset_price': asset_price,
        'asset_return': returns,
        'hedge_price': hedge_price,
        'hedge_return': hedge_returns,
        'volume': volume,
        'volatility_20d': vol_20d,
        'momentum_20d': mom_20d
    })
    
    data.set_index('date', inplace=True)
    
    return data

def backtest_deep_hedging(model, data, lookback=10, rebalance_freq='W', initial_capital=1000000):
    """
    Backtest the deep hedging model
    
    Parameters:
    -----------
    model : DeepHedgingModel
        Trained deep hedging model
    data : pandas.DataFrame
        Market data
    lookback : int
        Lookback period for the model
    rebalance_freq : str
        Rebalancing frequency ('D' for daily, 'W' for weekly, 'M' for monthly)
    initial_capital : float
        Initial capital
        
    Returns:
    --------
    pandas.DataFrame
        Backtest results
    """
    # Prepare features for the model
    feature_cols = ['asset_return', 'hedge_return', 'volatility_20d', 'momentum_20d', 'volume']
    
    # Normalize features
    data_norm = data.copy()
    for col in feature_cols:
        mean = data[col].mean()
        std = data[col].std()
        data_norm[col] = (data[col] - mean) / std
    
    # Initialize backtest variables
    capital = initial_capital
    asset_position = 0
    hedge_position = 0
    
    # Determine rebalance dates
    if rebalance_freq == 'D':
        rebalance_dates = data.index
    else:
        rebalance_dates = pd.date_range(start=data.index[0], end=data.index[-1], freq=rebalance_freq)
    
    # Initialize results
    results = []
    
    # Run backtest
    for i, date in enumerate(data.index[lookback:]):
        # Check if we need to rebalance
        if date in rebalance_dates or asset_position == 0:
            # Prepare input for the model
            X = np.array([data_norm[feature_cols].iloc[i:i+lookback].values])
            
            # Predict hedge ratio
            hedge_ratio = model.predict(X)[0][0]
            
            # Calculate positions
            asset_position = capital * 0.5 / data.loc[date, 'asset_price']  # Invest half of capital in asset
            hedge_position = -asset_position * hedge_ratio  # Negative for short position
        
        # Calculate daily P&L
        if i > 0:
            asset_pnl = asset_position * data.loc[date, 'asset_price'] * data.loc[date, 'asset_return']
            hedge_pnl = hedge_position * data.loc[date, 'hedge_price'] * data.loc[date, 'hedge_return']
            daily_pnl = asset_pnl + hedge_pnl
            capital += daily_pnl
        
        # Record results
        results.append({
            'date': date,
            'capital': capital,
            'asset_position': asset_position,
            'hedge_position': hedge_position,
            'asset_price': data.loc[date, 'asset_price'],
            'hedge_price': data.loc[date, 'hedge_price'],
            'asset_return': data.loc[date, 'asset_return'],
            'hedge_return': data.loc[date, 'hedge_return'],
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

def plot_backtest_results(results, title="Deep Hedging Backtest Results"):
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
    plt.savefig('deep_hedging_backtest_results.png')
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
    plt.savefig('deep_hedging_performance_metrics.png')
    plt.close()

def main():
    """Main function to test the deep hedging model"""
    print("Testing deep hedging model with realistic market data...")
    
    # Generate realistic market data
    data = generate_realistic_market_data(start_date='2020-01-01', end_date='2023-12-31')
    
    # Create synthetic data for training
    X_train, y_train, X_test, y_test = create_synthetic_data(n_samples=1000, n_features=5, lookback=10)
    
    # Initialize deep hedging model
    model = DeepHedgingModel(
        lookback_period=10,
        feature_dim=5,
        lstm_units=64,
        dense_units=32,
        learning_rate=0.001,
        lambda_reg=0.01,
        risk_aversion=1.0
    )
    
    # Train the model
    print("\nTraining the model...")
    history = model.train(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        verbose=1
    )
    
    # Plot training history
    model.plot_training_history(history)
    
    # Backtest the model
    print("\nBacktesting the model...")
    results = backtest_deep_hedging(
        model=model,
        data=data,
        lookback=10,
        rebalance_freq='W',  # Weekly rebalancing
        initial_capital=1000000
    )
    
    # Plot backtest results
    plot_backtest_results(results)
    
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
    
    # Save the model
    model.save_model('deep_hedging_model.h5')
    print("\nModel saved to 'deep_hedging_model.h5'")

if __name__ == "__main__":
    main()
