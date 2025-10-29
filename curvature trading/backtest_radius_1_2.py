"""
Backtest with radius_strength = 1.2 (Wide Arcs)
"""

import numpy as np
import pandas as pd
from database_connector import StockDataConnector
from backtest_engine import BacktestEngine
from backtest_visualizer import plot_backtest_results
import warnings
warnings.filterwarnings('ignore')


def run_backtest_with_radius_1_2(ticker='AAPL', start_date='2023-01-01', end_date='2023-12-31'):
    """
    Run backtest with radius_strength = 1.2
    """
    
    print("="*70)
    print(f"BACKTESTING: {ticker} with radius_strength = 1.2")
    print(f"Period: {start_date} to {end_date}")
    print("="*70)
    
    # Fetch data
    print(f"\nFetching {ticker} data...")
    connector = StockDataConnector()
    data = connector.fetch_stock_data(ticker, start_date, end_date)
    connector.close()
    
    print(f"Retrieved {len(data)} trading days")
    
    # Run backtest
    print("\nRunning backtest with radius_strength = 1.2...")
    engine = BacktestEngine(
        initial_capital=100000,
        commission=0.001,  # 0.1%
        slippage=0.0005,   # 0.05%
        position_size=0.95,
        allow_short=True
    )

    indicator_params = {
        'atr_period': 10,
        'atr_multiplier': 3.0,
        'radius_strength': 1.2,  # Wide arcs setting
        'smoothness': 3
    }

    results = engine.run_backtest(
        data=data,
        indicator_params=indicator_params
    )

    # Calculate statistics
    stats = engine.calculate_statistics(data)
    
    # Print results
    print("\n" + "="*70)
    print("BACKTEST RESULTS")
    print("="*70)

    print(f"\n📊 PERFORMANCE METRICS")
    print(f"   Total Return:        {stats['total_return_pct']:>10.2f}%")
    print(f"   Sharpe Ratio:        {stats['sharpe_ratio']:>10.2f}")
    print(f"   Max Drawdown:        {stats['max_drawdown_pct']:>10.2f}%")
    print(f"   Win Rate:            {stats['win_rate']:>10.2f}%")

    print(f"\n📈 TRADE STATISTICS")
    print(f"   Total Trades:        {stats['total_trades']:>10}")
    print(f"   Winning Trades:      {stats['winning_trades']:>10}")
    print(f"   Losing Trades:       {stats['losing_trades']:>10}")
    print(f"   Average Win:         ${stats['avg_win']:>10,.2f}")
    print(f"   Average Loss:        ${stats['avg_loss']:>10,.2f}")
    print(f"   Profit Factor:       {stats['profit_factor']:>10.2f}")
    print(f"   Avg Bars Held:       {stats['avg_bars_held']:>10.1f}")

    print(f"\n💰 CAPITAL")
    print(f"   Initial Capital:     ${engine.initial_capital:>10,.2f}")
    print(f"   Final Equity:        ${stats['final_equity']:>10,.2f}")
    print(f"   Total P&L:           ${stats['total_pnl']:>10,.2f}")

    # Performance rating
    score = 0
    if stats['total_return_pct'] > 0: score += 1
    if stats['sharpe_ratio'] > 1: score += 1
    if stats['win_rate'] > 50: score += 1
    if stats['max_drawdown_pct'] > -20: score += 1
    if stats['profit_factor'] > 1.5: score += 1
    
    rating = "⭐" * score
    performance = ["POOR", "FAIR", "GOOD", "VERY GOOD", "EXCELLENT"][score]
    
    print(f"\n🎯 PERFORMANCE RATING: {rating} ({score}/5) - {performance}")
    
    # Generate visualization
    print(f"\nGenerating visualization...")
    filename = f'backtest_{ticker.lower()}_radius_1_2.png'
    plot_backtest_results(
        results=results,
        ticker=ticker,
        save_path=filename
    )
    print(f"✅ Chart saved to: {filename}")
    
    print("\n" + "="*70)
    
    return results, stats


def compare_different_radius_settings(ticker='AAPL', start_date='2023-01-01', end_date='2023-12-31'):
    """
    Compare backtest results with different radius_strength values
    """
    
    print("\n" + "="*70)
    print(f"COMPARING DIFFERENT RADIUS SETTINGS FOR {ticker}")
    print(f"Period: {start_date} to {end_date}")
    print("="*70)
    
    # Fetch data once
    connector = StockDataConnector()
    data = connector.fetch_stock_data(ticker, start_date, end_date)
    connector.close()
    
    print(f"\nRetrieved {len(data)} trading days")
    
    # Test different radius values
    radius_values = [0.3, 0.5, 0.8, 1.0, 1.2, 1.5]
    
    results_summary = []
    
    for radius in radius_values:
        print(f"\n{'='*70}")
        print(f"Testing radius_strength = {radius}")
        print(f"{'='*70}")

        engine = BacktestEngine(
            initial_capital=100000,
            commission=0.001,
            slippage=0.0005,
            position_size=0.95,
            allow_short=True
        )

        indicator_params = {
            'atr_period': 10,
            'atr_multiplier': 3.0,
            'radius_strength': radius,
            'smoothness': 3
        }

        results = engine.run_backtest(
            data=data,
            indicator_params=indicator_params
        )

        stats = engine.calculate_statistics(data)
        
        results_summary.append({
            'radius': radius,
            'total_return': stats['total_return_pct'],
            'sharpe_ratio': stats['sharpe_ratio'],
            'max_drawdown': stats['max_drawdown_pct'],
            'win_rate': stats['win_rate'],
            'total_trades': stats['total_trades'],
            'profit_factor': stats['profit_factor']
        })

        print(f"   Return: {stats['total_return_pct']:>8.2f}% | "
              f"Sharpe: {stats['sharpe_ratio']:>6.2f} | "
              f"Drawdown: {stats['max_drawdown_pct']:>7.2f}% | "
              f"Win Rate: {stats['win_rate']:>6.2f}% | "
              f"Trades: {stats['total_trades']:>3}")
    
    # Print comparison table
    print("\n" + "="*70)
    print("COMPARISON SUMMARY")
    print("="*70)
    print(f"\n{'Radius':<10} {'Return':<12} {'Sharpe':<10} {'Drawdown':<12} {'Win Rate':<12} {'Trades':<10}")
    print("-" * 70)
    
    for result in results_summary:
        print(f"{result['radius']:<10.1f} "
              f"{result['total_return']:>10.2f}% "
              f"{result['sharpe_ratio']:>9.2f} "
              f"{result['max_drawdown']:>10.2f}% "
              f"{result['win_rate']:>10.2f}% "
              f"{result['total_trades']:>9}")
    
    # Find best performer
    best_return = max(results_summary, key=lambda x: x['total_return'])
    best_sharpe = max(results_summary, key=lambda x: x['sharpe_ratio'])
    best_drawdown = max(results_summary, key=lambda x: x['max_drawdown'])
    
    print("\n" + "="*70)
    print("BEST PERFORMERS")
    print("="*70)
    print(f"🏆 Best Return:     radius = {best_return['radius']:.1f} ({best_return['total_return']:.2f}%)")
    print(f"🏆 Best Sharpe:     radius = {best_sharpe['radius']:.1f} ({best_sharpe['sharpe_ratio']:.2f})")
    print(f"🏆 Best Drawdown:   radius = {best_drawdown['radius']:.1f} ({best_drawdown['max_drawdown']:.2f}%)")
    
    print("\n" + "="*70)
    
    return results_summary


if __name__ == "__main__":
    # Run backtest with radius = 1.2
    print("\n[1/2] Running backtest with radius_strength = 1.2...")
    results, stats = run_backtest_with_radius_1_2('AAPL', '2023-01-01', '2023-12-31')
    
    # Compare different settings
    print("\n[2/2] Comparing different radius settings...")
    comparison = compare_different_radius_settings('AAPL', '2023-01-01', '2023-12-31')
    
    print("\n✅ BACKTEST COMPLETE!")

