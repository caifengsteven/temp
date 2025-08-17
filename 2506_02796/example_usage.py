#!/usr/bin/env python3
"""
LSTM-BEKK Trading System - Example Usage

This script demonstrates how to use the LSTM-BEKK trading system
with a simple example using synthetic or Yahoo Finance data.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
from datetime import datetime, timedelta

# Import LSTM-BEKK components
from lstm_bekk_trading import (
    DataManager, LSTMBEKKModel, TradingStrategies, 
    RiskManager, BacktestEngine, Visualizer
)


def create_synthetic_data(n_assets=5, n_periods=1000, start_date='2020-01-01'):
    """
    Create synthetic return data for demonstration.
    
    Args:
        n_assets: Number of assets
        n_periods: Number of time periods
        start_date: Start date for the data
        
    Returns:
        DataFrame with synthetic returns
    """
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Create date index
    dates = pd.date_range(start=start_date, periods=n_periods, freq='D')
    
    # Generate correlated returns
    # Create a correlation matrix
    correlation_matrix = np.random.uniform(0.3, 0.7, (n_assets, n_assets))
    correlation_matrix = (correlation_matrix + correlation_matrix.T) / 2
    np.fill_diagonal(correlation_matrix, 1.0)
    
    # Generate returns with time-varying volatility
    returns = []
    volatilities = np.random.uniform(0.01, 0.03, n_assets)  # Base volatilities
    
    for t in range(n_periods):
        # Add volatility clustering
        vol_multiplier = 1 + 0.5 * np.sin(t / 50) + 0.3 * np.random.normal(0, 0.1)
        current_vols = volatilities * vol_multiplier
        
        # Generate correlated returns
        independent_returns = np.random.normal(0, current_vols)
        correlated_returns = np.linalg.cholesky(correlation_matrix) @ independent_returns
        
        returns.append(correlated_returns)
    
    # Create DataFrame
    asset_names = [f'Asset_{i+1}' for i in range(n_assets)]
    returns_df = pd.DataFrame(returns, index=dates, columns=asset_names)
    
    # Scale to percentage returns (as in the paper)
    returns_df = returns_df * 100
    
    return returns_df


def demonstrate_basic_usage():
    """Demonstrate basic LSTM-BEKK usage with synthetic data."""
    print("=" * 60)
    print("LSTM-BEKK TRADING SYSTEM - BASIC DEMONSTRATION")
    print("=" * 60)
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # 1. Create synthetic data
    print("\n1. Creating synthetic return data...")
    returns_data = create_synthetic_data(n_assets=5, n_periods=500)
    print(f"   Generated {len(returns_data)} periods for {len(returns_data.columns)} assets")
    print(f"   Date range: {returns_data.index[0]} to {returns_data.index[-1]}")
    
    # 2. Initialize LSTM-BEKK model
    print("\n2. Initializing LSTM-BEKK model...")
    n_assets = len(returns_data.columns)
    
    # Simple configuration
    config = {
        'lstm_bekk': {
            'hidden_size': n_assets,
            'num_layers': 2,
            'dropout': 0.1,
            'epochs': 20,  # Reduced for demo
            'learning_rate': 0.001
        }
    }
    
    model = LSTMBEKKModel(n_assets, config)
    print(f"   Model initialized for {n_assets} assets")
    
    # 3. Prepare data splits
    print("\n3. Preparing data splits...")
    train_size = int(0.7 * len(returns_data))
    val_size = int(0.15 * len(returns_data))
    
    train_data = returns_data.iloc[:train_size]
    val_data = returns_data.iloc[train_size:train_size + val_size]
    test_data = returns_data.iloc[train_size + val_size:]
    
    print(f"   Train: {len(train_data)} periods")
    print(f"   Validation: {len(val_data)} periods")
    print(f"   Test: {len(test_data)} periods")
    
    # 4. Train the model
    print("\n4. Training LSTM-BEKK model...")
    training_history = model.fit(train_data, val_data, verbose=True)
    print(f"   Training completed in {len(training_history)} epochs")
    
    # 5. Generate covariance forecasts
    print("\n5. Generating covariance forecasts...")
    cov_forecast = model.predict_covariance(train_data, steps=1)[0]
    print(f"   Covariance matrix shape: {cov_forecast.shape}")
    print(f"   Portfolio volatility forecast: {np.sqrt(np.trace(cov_forecast) / n_assets):.4f}")
    
    # 6. Initialize trading strategies
    print("\n6. Initializing trading strategies...")
    trading_config = {
        'trading': {
            'portfolio': {
                'max_position_size': 0.3,
                'min_position_size': 0.05,
                'rebalance_frequency': 5
            },
            'strategies': {
                'gmv': {'enabled': True},
                'volatility_sizing': {'enabled': True}
            }
        }
    }
    
    strategies = TradingStrategies(trading_config)
    
    # 7. Generate trading signals
    print("\n7. Generating trading signals...")
    model_outputs = {
        'covariance_forecast': cov_forecast,
        'volatility_forecast': np.sqrt(np.diag(cov_forecast)),
        'correlation_forecast': cov_forecast / np.outer(np.sqrt(np.diag(cov_forecast)), 
                                                       np.sqrt(np.diag(cov_forecast)))
    }
    
    signals = strategies.generate_trading_signals(
        model_outputs, train_data, datetime.now()
    )
    
    print(f"   GMV weights: {signals['gmv_weights']}")
    print(f"   Final weights: {signals['final_weights']}")
    
    # 8. Risk assessment
    print("\n8. Performing risk assessment...")
    risk_manager = RiskManager(trading_config)
    
    if signals['final_weights'] is not None:
        risk_assessment = risk_manager.assess_portfolio_risk(
            signals['final_weights'], cov_forecast, train_data, datetime.now()
        )
        
        print(f"   Portfolio volatility: {risk_assessment['portfolio_volatility']:.4f}")
        print(f"   VaR estimates: {risk_assessment['var_estimates']}")
        print(f"   Risk alerts: {len(risk_assessment['risk_alerts'])}")
    
    # 9. Simple backtest
    print("\n9. Running simple backtest...")
    portfolio_returns = []
    
    for i in range(len(test_data) - 1):
        if signals['final_weights'] is not None:
            weights = signals['final_weights']
        else:
            weights = np.ones(n_assets) / n_assets  # Equal weights fallback
        
        # Calculate portfolio return
        next_return = np.sum(weights * test_data.iloc[i + 1])
        portfolio_returns.append(next_return)
    
    portfolio_returns = pd.Series(portfolio_returns, index=test_data.index[1:])
    
    # Calculate performance metrics
    total_return = (1 + portfolio_returns / 100).prod() - 1
    annualized_return = (1 + total_return) ** (252 / len(portfolio_returns)) - 1
    volatility = portfolio_returns.std() * np.sqrt(252) / 100
    sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
    
    print(f"   Total return: {total_return:.2%}")
    print(f"   Annualized return: {annualized_return:.2%}")
    print(f"   Volatility: {volatility:.2%}")
    print(f"   Sharpe ratio: {sharpe_ratio:.3f}")
    
    # 10. Visualization
    print("\n10. Creating visualizations...")
    try:
        # Plot cumulative returns
        cumulative_returns = (1 + portfolio_returns / 100).cumprod()
        
        plt.figure(figsize=(12, 6))
        plt.subplot(2, 1, 1)
        plt.plot(cumulative_returns.index, cumulative_returns.values)
        plt.title('LSTM-BEKK Portfolio Cumulative Returns')
        plt.ylabel('Cumulative Return')
        plt.grid(True)
        
        # Plot portfolio weights
        plt.subplot(2, 1, 2)
        if signals['final_weights'] is not None:
            plt.bar(returns_data.columns, signals['final_weights'])
            plt.title('Portfolio Weights')
            plt.ylabel('Weight')
            plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig('lstm_bekk_demo_results.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        print("   Visualization saved as 'lstm_bekk_demo_results.png'")
        
    except Exception as e:
        print(f"   Visualization error: {e}")
    
    print("\n" + "=" * 60)
    print("DEMONSTRATION COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    
    return {
        'model': model,
        'training_history': training_history,
        'portfolio_returns': portfolio_returns,
        'signals': signals,
        'risk_assessment': risk_assessment if 'risk_assessment' in locals() else None
    }


def demonstrate_with_yahoo_data():
    """Demonstrate with real Yahoo Finance data."""
    print("\n" + "=" * 60)
    print("LSTM-BEKK WITH REAL DATA DEMONSTRATION")
    print("=" * 60)
    
    try:
        # Simple configuration for Yahoo Finance
        config = {
            'data': {
                'source': 'yahoo',
                'bloomberg': {
                    'start_date': '2020-01-01',
                    'end_date': '2023-12-31'
                },
                'universes': {
                    'demo': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
                }
            },
            'lstm_bekk': {
                'hidden_size': 5,
                'num_layers': 2,
                'dropout': 0.1,
                'epochs': 10,
                'learning_rate': 0.001
            }
        }
        
        # Initialize data manager
        data_manager = DataManager(config=config)
        
        # Load real data
        print("Loading real market data...")
        data_manager.load_data('demo')
        
        # Get data statistics
        stats = data_manager.get_data_statistics()
        print(f"Loaded {stats['shape'][0]} periods for {stats['shape'][1]} assets")
        
        # Run simple backtest
        backtest_engine = BacktestEngine(config)
        results = backtest_engine.run_backtest(data_manager, universe='demo')
        
        # Display results
        if 'performance_comparison' in results:
            print("\nPerformance Comparison:")
            comparison = results['performance_comparison']
            for strategy in comparison.index:
                metrics = comparison.loc[strategy]
                print(f"{strategy:15s}: Sharpe={metrics['Sharpe Ratio']:6.3f}, "
                      f"Return={metrics['Annualized Return']:7.2%}, "
                      f"Vol={metrics['Annualized Volatility']:6.2%}")
        
        print("\nReal data demonstration completed!")
        return results
        
    except Exception as e:
        print(f"Real data demonstration failed: {e}")
        print("This might be due to missing yfinance package or network issues.")
        return None


if __name__ == "__main__":
    # Run basic demonstration with synthetic data
    demo_results = demonstrate_basic_usage()
    
    # Optionally try with real data
    response = input("\nTry demonstration with real Yahoo Finance data? (y/n): ")
    if response.lower() == 'y':
        real_data_results = demonstrate_with_yahoo_data()
    
    print("\nDemo completed! Check the generated plots and results.")
