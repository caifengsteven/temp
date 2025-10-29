"""
Run Backtest for Curved Radius Supertrend Strategy

This script runs a comprehensive backtest using data from the database.
"""

import pandas as pd
import numpy as np
from database_connector import StockDataConnector
from backtest_engine import BacktestEngine
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


def print_statistics(stats: dict):
    """Print backtest statistics in a formatted way"""
    print("\n" + "=" * 70)
    print("BACKTEST RESULTS")
    print("=" * 70)
    
    print(f"\nPeriod: {stats['start_date']} to {stats['end_date']}")
    
    print("\n--- Trade Statistics ---")
    print(f"Total Trades:          {stats['total_trades']}")
    print(f"Winning Trades:        {stats['winning_trades']} ({stats['win_rate']:.2f}%)")
    print(f"Losing Trades:         {stats['losing_trades']}")
    print(f"Average Bars Held:     {stats['avg_bars_held']:.1f}")
    
    print("\n--- Performance Metrics ---")
    print(f"Total Return:          {stats['total_return_pct']:.2f}%")
    print(f"Total P&L:             ${stats['total_pnl']:,.2f}")
    print(f"Average P&L per Trade: ${stats['avg_pnl_per_trade']:,.2f}")
    print(f"Average Win:           ${stats['avg_win']:,.2f}")
    print(f"Average Loss:          ${stats['avg_loss']:,.2f}")
    print(f"Profit Factor:         {stats['profit_factor']:.2f}")
    
    print("\n--- Risk Metrics ---")
    print(f"Sharpe Ratio:          {stats['sharpe_ratio']:.2f}")
    print(f"Max Drawdown:          {stats['max_drawdown_pct']:.2f}%")
    
    print("\n--- Capital ---")
    print(f"Final Equity:          ${stats['final_equity']:,.2f}")
    
    print("\n" + "=" * 70)


def run_single_stock_backtest(
    ticker: str,
    start_date: str,
    end_date: str,
    indicator_params: dict = None,
    initial_capital: float = 100000.0,
    position_size: float = 0.95,
    allow_short: bool = False
):
    """
    Run backtest for a single stock
    
    Parameters:
    -----------
    ticker : str
        Stock ticker symbol
    start_date : str
        Start date (YYYY-MM-DD)
    end_date : str
        End date (YYYY-MM-DD)
    indicator_params : dict
        Curved Radius Supertrend parameters
    initial_capital : float
        Starting capital
    position_size : float
        Fraction of capital per trade
    allow_short : bool
        Whether to allow short positions
    """
    print(f"\n{'='*70}")
    print(f"BACKTESTING: {ticker}")
    print(f"Period: {start_date} to {end_date}")
    print(f"{'='*70}")
    
    # Default indicator parameters
    if indicator_params is None:
        indicator_params = {
            'atr_period': 10,
            'atr_multiplier': 3.0,
            'radius_strength': 0.5,
            'smoothness': 3
        }
    
    print("\nIndicator Parameters:")
    for key, value in indicator_params.items():
        print(f"  {key}: {value}")
    
    # Fetch data
    print(f"\nFetching data from database...")
    connector = StockDataConnector()
    
    try:
        data = connector.fetch_stock_data(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            min_volume=100000
        )
        
        if data.empty:
            print(f"ERROR: No data found for {ticker}")
            return None
        
        print(f"Retrieved {len(data)} trading days")
        
        # Run backtest
        engine = BacktestEngine(
            initial_capital=initial_capital,
            commission=0.001,
            slippage=0.0005,
            position_size=position_size,
            allow_short=allow_short
        )
        
        results = engine.run_backtest(data, indicator_params)
        
        # Print statistics
        print_statistics(results['statistics'])
        
        # Print recent trades
        if results['trades']:
            print("\n--- Recent Trades (Last 10) ---")
            for trade in results['trades'][-10:]:
                print(f"{trade.entry_date.strftime('%Y-%m-%d')} {trade.direction:5s} "
                      f"Entry: ${trade.entry_price:7.2f} Exit: ${trade.exit_price:7.2f} "
                      f"P&L: ${trade.pnl:8.2f} ({trade.return_pct:+6.2f}%)")
        
        return results
        
    finally:
        connector.close()


def run_multiple_stocks_backtest(
    tickers: list,
    start_date: str,
    end_date: str,
    indicator_params: dict = None,
    initial_capital: float = 100000.0
):
    """
    Run backtest for multiple stocks and compare results
    """
    print(f"\n{'='*70}")
    print(f"MULTI-STOCK BACKTEST")
    print(f"Tickers: {', '.join(tickers)}")
    print(f"Period: {start_date} to {end_date}")
    print(f"{'='*70}")
    
    all_results = {}
    
    for ticker in tickers:
        print(f"\n\nProcessing {ticker}...")
        print("-" * 70)
        
        results = run_single_stock_backtest(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            indicator_params=indicator_params,
            initial_capital=initial_capital,
            position_size=0.95,
            allow_short=False
        )
        
        if results:
            all_results[ticker] = results
    
    # Summary comparison
    if all_results:
        print("\n\n" + "="*70)
        print("COMPARISON SUMMARY")
        print("="*70)
        print(f"\n{'Ticker':<10} {'Trades':<8} {'Win%':<8} {'Return%':<10} {'Sharpe':<8} {'MaxDD%':<10}")
        print("-" * 70)
        
        for ticker, results in all_results.items():
            stats = results['statistics']
            print(f"{ticker:<10} {stats['total_trades']:<8} "
                  f"{stats['win_rate']:<8.1f} {stats['total_return_pct']:<10.2f} "
                  f"{stats['sharpe_ratio']:<8.2f} {stats['max_drawdown_pct']:<10.2f}")
    
    return all_results


def parameter_optimization(
    ticker: str,
    start_date: str,
    end_date: str,
    param_grid: dict
):
    """
    Test different parameter combinations
    
    Parameters:
    -----------
    ticker : str
        Stock ticker
    start_date, end_date : str
        Date range
    param_grid : dict
        Dictionary of parameter lists to test
        Example: {
            'radius_strength': [0.2, 0.5, 1.0],
            'atr_period': [10, 14, 20]
        }
    """
    print(f"\n{'='*70}")
    print(f"PARAMETER OPTIMIZATION: {ticker}")
    print(f"{'='*70}")
    
    # Fetch data once
    connector = StockDataConnector()
    try:
        data = connector.fetch_stock_data(ticker, start_date, end_date, min_volume=100000)
        
        if data.empty:
            print(f"ERROR: No data found for {ticker}")
            return None
        
        print(f"Retrieved {len(data)} trading days")
        
    finally:
        connector.close()
    
    # Generate parameter combinations
    from itertools import product
    
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    combinations = list(product(*param_values))
    
    print(f"\nTesting {len(combinations)} parameter combinations...")
    
    results_list = []
    
    for combo in combinations:
        params = dict(zip(param_names, combo))
        
        # Set defaults for missing parameters
        full_params = {
            'atr_period': 10,
            'atr_multiplier': 3.0,
            'radius_strength': 0.5,
            'smoothness': 3
        }
        full_params.update(params)
        
        # Run backtest
        engine = BacktestEngine(initial_capital=100000, position_size=0.95)
        result = engine.run_backtest(data, full_params)
        
        stats = result['statistics']
        stats['params'] = params
        results_list.append(stats)
    
    # Sort by total return
    results_list.sort(key=lambda x: x['total_return_pct'], reverse=True)
    
    # Print top 10 results
    print("\n" + "="*70)
    print("TOP 10 PARAMETER COMBINATIONS (by Total Return)")
    print("="*70)
    
    for i, stats in enumerate(results_list[:10], 1):
        print(f"\n#{i} - Return: {stats['total_return_pct']:.2f}% | "
              f"Sharpe: {stats['sharpe_ratio']:.2f} | "
              f"Win Rate: {stats['win_rate']:.1f}%")
        print(f"    Parameters: {stats['params']}")
        print(f"    Trades: {stats['total_trades']} | "
              f"Max DD: {stats['max_drawdown_pct']:.2f}%")
    
    return results_list


def main():
    """Main function to run backtests"""
    
    print("\n" + "="*70)
    print("CURVED RADIUS SUPERTREND - BACKTESTING SYSTEM")
    print("="*70)
    
    # Example 1: Single stock backtest
    print("\n\n### Example 1: Single Stock Backtest ###")
    results_aapl = run_single_stock_backtest(
        ticker='AAPL',
        start_date='2022-01-01',
        end_date='2023-12-31',
        indicator_params={
            'atr_period': 10,
            'atr_multiplier': 3.0,
            'radius_strength': 0.5,
            'smoothness': 3
        },
        initial_capital=100000.0,
        position_size=0.95,
        allow_short=False
    )
    
    # Example 2: Multiple stocks comparison
    print("\n\n### Example 2: Multiple Stocks Comparison ###")
    results_multi = run_multiple_stocks_backtest(
        tickers=['AAPL', 'MSFT', 'GOOGL', 'TSLA'],
        start_date='2023-01-01',
        end_date='2023-12-31',
        indicator_params={
            'atr_period': 10,
            'atr_multiplier': 3.0,
            'radius_strength': 0.5,
            'smoothness': 3
        },
        initial_capital=100000.0
    )
    
    print("\n\nBacktesting complete!")


if __name__ == "__main__":
    main()

