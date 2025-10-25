"""
Parameter Optimization for Curved Radius Supertrend
Test different parameter combinations to find optimal settings
"""

import pandas as pd
import numpy as np
from backtest_curved_supertrend import (
    CurvedRadiusSupertrendFixed, 
    BacktestEngine, 
    connect_to_nas, 
    get_stock_data
)

def test_parameters(df, radius_strength, atr_length=14, atr_mult=2.0, smoothness=5,
                   initial_capital=10000, commission=0.001, slippage=0.0005):
    """Test a specific parameter combination"""
    
    # Calculate indicator
    indicator = CurvedRadiusSupertrendFixed(
        atr_length=atr_length,
        atr_mult=atr_mult,
        radius_strength=radius_strength,
        smoothness=smoothness
    )
    
    signals = indicator.calculate(df['high'].values, df['low'].values, df['close'].values)
    
    # Run backtest
    engine = BacktestEngine(
        initial_capital=initial_capital,
        commission=commission,
        slippage=slippage
    )
    
    results = engine.run_backtest(df, signals)
    
    return {
        'radius_strength': radius_strength,
        'atr_length': atr_length,
        'atr_mult': atr_mult,
        'smoothness': smoothness,
        'num_signals': signals['buy_signals'].sum() + signals['sell_signals'].sum(),
        'num_trades': len(results['trades']),
        **results['metrics']
    }

def optimize_parameters(ticker='QQQ', table_name='200309', limit=2000):
    """
    Test multiple parameter combinations to find optimal settings
    """
    print("\n" + "=" * 80)
    print("PARAMETER OPTIMIZATION")
    print("=" * 80)
    
    # Connect and fetch data
    print(f"\nFetching data for {ticker} from table {table_name}...")
    connection = connect_to_nas()
    if not connection:
        return None
    
    df = get_stock_data(connection, table_name, ticker, limit)
    connection.close()
    
    if df is None or len(df) < 50:
        print("✗ Insufficient data")
        return None
    
    print(f"✓ Fetched {len(df)} bars")
    print(f"  Date range: {df['datetime'].min()} to {df['datetime'].max()}")
    
    # Test different radius_strength values
    print("\n" + "=" * 80)
    print("Testing different Radius Strength values...")
    print("=" * 80)
    
    # Test a wider range of radius strengths
    radius_strengths = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 0.80, 1.0]
    
    results_list = []
    
    for rs in radius_strengths:
        print(f"\nTesting radius_strength = {rs:.2f}...", end=" ")
        try:
            result = test_parameters(df, radius_strength=rs)
            results_list.append(result)
            print(f"✓ Signals: {result['num_signals']}, Trades: {result['num_trades']}, "
                  f"Return: {result['total_return_pct']:.2f}%, Win Rate: {result['win_rate']:.1f}%")
        except Exception as e:
            print(f"✗ Error: {e}")
    
    # Create results dataframe
    results_df = pd.DataFrame(results_list)
    
    # Sort by total return
    results_df = results_df.sort_values('total_return_pct', ascending=False)
    
    # Print summary
    print("\n" + "=" * 80)
    print("OPTIMIZATION RESULTS (Sorted by Total Return)")
    print("=" * 80)
    print(f"\n{'Radius':<8} {'Signals':<10} {'Trades':<10} {'Return %':<12} {'Win Rate':<12} "
          f"{'Profit Factor':<15} {'Max DD %':<12}")
    print("-" * 80)
    
    for _, row in results_df.iterrows():
        print(f"{row['radius_strength']:<8.2f} {row['num_signals']:<10} {row['num_trades']:<10} "
              f"{row['total_return_pct']:<12.2f} {row['win_rate']:<12.1f} "
              f"{row['profit_factor']:<15.2f} {row['max_drawdown_pct']:<12.2f}")
    
    # Find best parameter
    best = results_df.iloc[0]
    print("\n" + "=" * 80)
    print("BEST PARAMETERS")
    print("=" * 80)
    print(f"\nRadius Strength: {best['radius_strength']:.2f}")
    print(f"Total Return: {best['total_return_pct']:.2f}%")
    print(f"Win Rate: {best['win_rate']:.1f}%")
    print(f"Profit Factor: {best['profit_factor']:.2f}")
    print(f"Total Trades: {best['num_trades']}")
    print(f"Max Drawdown: {best['max_drawdown_pct']:.2f}%")
    
    return results_df

def test_multiple_tickers():
    """Test on multiple tickers to find robust parameters"""
    print("\n" + "=" * 80)
    print("MULTI-TICKER PARAMETER TEST")
    print("=" * 80)
    
    tickers = ['QQQ', 'SPY', 'MSFT', 'INTC', 'CSCO']
    radius_strengths = [0.20, 0.30, 0.40, 0.50, 0.60]
    
    connection = connect_to_nas()
    if not connection:
        return None
    
    all_results = []
    
    for ticker in tickers:
        print(f"\n{'='*80}")
        print(f"Testing {ticker}")
        print(f"{'='*80}")
        
        df = get_stock_data(connection, '200309', ticker, 2000)
        
        if df is None or len(df) < 100:
            print(f"✗ Insufficient data for {ticker}")
            continue
        
        print(f"✓ Fetched {len(df)} bars for {ticker}")
        
        for rs in radius_strengths:
            try:
                result = test_parameters(df, radius_strength=rs)
                result['ticker'] = ticker
                all_results.append(result)
                print(f"  RS={rs:.2f}: Return={result['total_return_pct']:>7.2f}%, "
                      f"Trades={result['num_trades']:>4}, WinRate={result['win_rate']:>5.1f}%")
            except Exception as e:
                print(f"  RS={rs:.2f}: Error - {e}")
    
    connection.close()
    
    # Analyze results
    results_df = pd.DataFrame(all_results)
    
    print("\n" + "=" * 80)
    print("AVERAGE PERFORMANCE BY RADIUS STRENGTH (Across All Tickers)")
    print("=" * 80)
    
    summary = results_df.groupby('radius_strength').agg({
        'total_return_pct': 'mean',
        'win_rate': 'mean',
        'profit_factor': 'mean',
        'num_trades': 'mean',
        'max_drawdown_pct': 'mean'
    }).round(2)
    
    print(summary)
    
    # Find most consistent parameter
    print("\n" + "=" * 80)
    print("BEST OVERALL RADIUS STRENGTH")
    print("=" * 80)
    
    best_rs = summary['total_return_pct'].idxmax()
    print(f"\nBest Radius Strength: {best_rs:.2f}")
    print(f"Average Return: {summary.loc[best_rs, 'total_return_pct']:.2f}%")
    print(f"Average Win Rate: {summary.loc[best_rs, 'win_rate']:.1f}%")
    print(f"Average Trades: {summary.loc[best_rs, 'num_trades']:.0f}")
    
    return results_df

if __name__ == "__main__":
    # First optimize on QQQ
    print("\n" + "=" * 80)
    print("STEP 1: OPTIMIZE ON QQQ")
    print("=" * 80)
    results_df = optimize_parameters(ticker='QQQ', table_name='200309', limit=2000)
    
    # Then test on multiple tickers
    print("\n\n" + "=" * 80)
    print("STEP 2: TEST ON MULTIPLE TICKERS")
    print("=" * 80)
    multi_results = test_multiple_tickers()
    
    print("\n" + "=" * 80)
    print("OPTIMIZATION COMPLETE")
    print("=" * 80)

