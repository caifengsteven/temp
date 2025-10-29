"""
Comprehensive Backtesting Example

This script demonstrates how to use the backtesting system with
the Curved Radius Supertrend strategy.
"""

import matplotlib.pyplot as plt
from database_connector import StockDataConnector
from backtest_engine import BacktestEngine
from backtest_visualizer import plot_backtest_results, plot_multi_stock_comparison
import warnings
warnings.filterwarnings('ignore')


def example_1_simple_backtest():
    """
    Example 1: Simple backtest for a single stock
    """
    print("\n" + "="*70)
    print("EXAMPLE 1: Simple Backtest - AAPL (2023)")
    print("="*70)
    
    # Fetch data
    connector = StockDataConnector()
    try:
        data = connector.fetch_stock_data(
            ticker='AAPL',
            start_date='2023-01-01',
            end_date='2023-12-31',
            min_volume=1000000
        )
        
        print(f"\nFetched {len(data)} trading days for AAPL")
        
        # Run backtest
        engine = BacktestEngine(
            initial_capital=100000.0,
            commission=0.001,
            slippage=0.0005,
            position_size=0.95,
            allow_short=False
        )
        
        indicator_params = {
            'atr_period': 10,
            'atr_multiplier': 3.0,
            'radius_strength': 0.5,
            'smoothness': 3
        }
        
        results = engine.run_backtest(data, indicator_params)
        
        # Print statistics
        stats = results['statistics']
        print(f"\n--- Results ---")
        print(f"Total Trades:     {stats['total_trades']}")
        print(f"Win Rate:         {stats['win_rate']:.2f}%")
        print(f"Total Return:     {stats['total_return_pct']:.2f}%")
        print(f"Sharpe Ratio:     {stats['sharpe_ratio']:.2f}")
        print(f"Max Drawdown:     {stats['max_drawdown_pct']:.2f}%")
        print(f"Final Equity:     ${stats['final_equity']:,.2f}")
        
        # Visualize
        fig = plot_backtest_results(results, ticker='AAPL', 
                                    save_path='backtest_aapl_2023.png')
        plt.close(fig)
        
        return results
        
    finally:
        connector.close()


def example_2_compare_stocks():
    """
    Example 2: Compare multiple stocks
    """
    print("\n" + "="*70)
    print("EXAMPLE 2: Multi-Stock Comparison (2023)")
    print("="*70)
    
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA']
    
    connector = StockDataConnector()
    all_results = {}
    
    try:
        for ticker in tickers:
            print(f"\nProcessing {ticker}...")
            
            data = connector.fetch_stock_data(
                ticker=ticker,
                start_date='2023-01-01',
                end_date='2023-12-31',
                min_volume=1000000
            )
            
            if data.empty or len(data) < 50:
                print(f"  Insufficient data for {ticker} ({len(data)} days)")
                continue

            print(f"  Fetched {len(data)} days")
            
            # Run backtest
            engine = BacktestEngine(
                initial_capital=100000.0,
                position_size=0.95,
                allow_short=False
            )
            
            results = engine.run_backtest(data, {
                'atr_period': 10,
                'atr_multiplier': 3.0,
                'radius_strength': 0.5,
                'smoothness': 3
            })
            
            all_results[ticker] = results
            
            stats = results['statistics']
            print(f"  Return: {stats['total_return_pct']:.2f}% | "
                  f"Sharpe: {stats['sharpe_ratio']:.2f} | "
                  f"Win Rate: {stats['win_rate']:.1f}%")
        
        # Comparison visualization
        if all_results:
            fig = plot_multi_stock_comparison(all_results, 
                                             save_path='backtest_comparison_2023.png')
            plt.close(fig)
        
        return all_results
        
    finally:
        connector.close()


def example_3_parameter_testing():
    """
    Example 3: Test different radius_strength values
    """
    print("\n" + "="*70)
    print("EXAMPLE 3: Parameter Testing - radius_strength")
    print("="*70)
    
    # Fetch data
    connector = StockDataConnector()
    try:
        data = connector.fetch_stock_data(
            ticker='AAPL',
            start_date='2023-01-01',
            end_date='2023-12-31',
            min_volume=1000000
        )
        
        print(f"\nFetched {len(data)} trading days for AAPL")
        
    finally:
        connector.close()
    
    # Test different radius_strength values
    radius_values = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]
    
    results_list = []
    
    print(f"\nTesting {len(radius_values)} different radius_strength values...")
    print(f"\n{'Radius':<10} {'Trades':<8} {'Win%':<8} {'Return%':<10} {'Sharpe':<8} {'MaxDD%':<10}")
    print("-" * 70)
    
    for radius in radius_values:
        engine = BacktestEngine(initial_capital=100000, position_size=0.95)
        
        results = engine.run_backtest(data, {
            'atr_period': 10,
            'atr_multiplier': 3.0,
            'radius_strength': radius,
            'smoothness': 3
        })
        
        stats = results['statistics']
        stats['params'] = {'radius_strength': radius}
        results_list.append(stats)
        
        print(f"{radius:<10.1f} {stats['total_trades']:<8} "
              f"{stats['win_rate']:<8.1f} {stats['total_return_pct']:<10.2f} "
              f"{stats['sharpe_ratio']:<8.2f} {stats['max_drawdown_pct']:<10.2f}")
    
    # Find best parameter
    best = max(results_list, key=lambda x: x['sharpe_ratio'])
    print(f"\nBest radius_strength (by Sharpe): {best['params']['radius_strength']}")
    print(f"  Return: {best['total_return_pct']:.2f}%")
    print(f"  Sharpe: {best['sharpe_ratio']:.2f}")
    
    return results_list


def example_4_long_term_backtest():
    """
    Example 4: Long-term backtest (2020-2023)
    """
    print("\n" + "="*70)
    print("EXAMPLE 4: Long-term Backtest - AAPL (2020-2023)")
    print("="*70)
    
    connector = StockDataConnector()
    try:
        data = connector.fetch_stock_data(
            ticker='AAPL',
            start_date='2020-01-01',
            end_date='2023-12-31',
            min_volume=1000000
        )
        
        print(f"\nFetched {len(data)} trading days")
        print(f"Period: {data['date'].min()} to {data['date'].max()}")
        
        # Run backtest
        engine = BacktestEngine(
            initial_capital=100000.0,
            position_size=0.95,
            allow_short=False
        )
        
        results = engine.run_backtest(data, {
            'atr_period': 10,
            'atr_multiplier': 3.0,
            'radius_strength': 0.5,
            'smoothness': 3
        })
        
        # Print detailed statistics
        stats = results['statistics']
        print(f"\n--- Performance Summary ---")
        print(f"Total Trades:          {stats['total_trades']}")
        print(f"Winning Trades:        {stats['winning_trades']}")
        print(f"Losing Trades:         {stats['losing_trades']}")
        print(f"Win Rate:              {stats['win_rate']:.2f}%")
        print(f"\nTotal Return:          {stats['total_return_pct']:.2f}%")
        print(f"Annualized Return:     {stats['total_return_pct']/4:.2f}%")
        print(f"Sharpe Ratio:          {stats['sharpe_ratio']:.2f}")
        print(f"Max Drawdown:          {stats['max_drawdown_pct']:.2f}%")
        print(f"Profit Factor:         {stats['profit_factor']:.2f}")
        print(f"\nAverage Win:           ${stats['avg_win']:,.2f}")
        print(f"Average Loss:          ${stats['avg_loss']:,.2f}")
        print(f"Average Bars Held:     {stats['avg_bars_held']:.1f} days")
        print(f"\nFinal Equity:          ${stats['final_equity']:,.2f}")
        
        # Visualize
        fig = plot_backtest_results(results, ticker='AAPL (2020-2023)', 
                                    save_path='backtest_aapl_longterm.png')
        plt.close(fig)
        
        # Print recent trades
        print(f"\n--- Last 10 Trades ---")
        for trade in results['trades'][-10:]:
            print(f"{trade.entry_date.strftime('%Y-%m-%d')} {trade.direction:5s} "
                  f"Entry: ${trade.entry_price:7.2f} Exit: ${trade.exit_price:7.2f} "
                  f"P&L: ${trade.pnl:8.2f} ({trade.return_pct:+6.2f}%)")
        
        return results
        
    finally:
        connector.close()


def main():
    """Run all examples"""
    
    print("\n" + "="*70)
    print("CURVED RADIUS SUPERTREND - BACKTESTING EXAMPLES")
    print("="*70)
    
    # Example 1: Simple backtest
    results_1 = example_1_simple_backtest()
    
    # Example 2: Compare multiple stocks
    results_2 = example_2_compare_stocks()
    
    # Example 3: Parameter testing
    results_3 = example_3_parameter_testing()
    
    # Example 4: Long-term backtest
    results_4 = example_4_long_term_backtest()
    
    print("\n" + "="*70)
    print("ALL EXAMPLES COMPLETED!")
    print("="*70)
    print("\nGenerated files:")
    print("  - backtest_aapl_2023.png")
    print("  - backtest_comparison_2023.png")
    print("  - backtest_aapl_longterm.png")
    print("\nCheck these files for detailed visualizations.")


if __name__ == "__main__":
    main()

