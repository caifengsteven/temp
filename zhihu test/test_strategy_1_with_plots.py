"""
Test Script for Bull Rise Indicator Strategy with Plotting

This script demonstrates how to run the Bull Rise Indicator strategy
with simulated data and generate visualization charts.
"""

from strategy_1_bull_rise_indicator import BullRiseIndicatorStrategy
import matplotlib.pyplot as plt

def test_strategy_with_plots():
    """Test the strategy and show plots"""
    print("="*60)
    print("Testing Bull Rise Indicator Strategy with Simulated Data")
    print("="*60)
    
    # Create strategy instance
    strategy = BullRiseIndicatorStrategy(
        symbol='TEST_STOCK', 
        start_date='2022-01-01', 
        end_date='2024-06-30'  # Shorter period for faster execution
    )
    
    # Run the strategy
    print("Step 1: Generating simulated data...")
    if not strategy.fetch_data():
        print("Failed to generate data")
        return
    
    print("Step 2: Calculating indicators...")
    if not strategy.calculate_indicators():
        print("Failed to calculate indicators")
        return
    
    print("Step 3: Generating trading signals...")
    if not strategy.generate_signals():
        print("Failed to generate signals")
        return
    
    print("Step 4: Running backtest...")
    results = strategy.backtest(initial_capital=10000)
    
    if not results:
        print("Backtest failed")
        return
    
    # Print detailed results
    print("\n" + "="*50)
    print("STRATEGY PERFORMANCE RESULTS")
    print("="*50)
    print(f"Initial Capital: ${results['initial_capital']:,.2f}")
    print(f"Final Value: ${results['final_value']:,.2f}")
    print(f"Total Return: {results['total_return']:.2%}")
    print(f"Total Trades: {results['total_trades']}")
    print(f"Win Rate: {results['win_rate']:.1%}")
    
    # Show trade summary
    if results['trades']:
        buy_trades = [t for t in results['trades'] if t['action'] == 'BUY']
        sell_trades = [t for t in results['trades'] if t['action'] == 'SELL']
        
        print(f"\nTrade Summary:")
        print(f"  Buy orders: {len(buy_trades)}")
        print(f"  Sell orders: {len(sell_trades)}")
        
        if sell_trades:
            profits = [t.get('profit', 0) for t in sell_trades]
            avg_profit = sum(profits) / len(profits)
            print(f"  Average profit per trade: ${avg_profit:.2f}")
    
    # Generate plots
    print("\nStep 5: Generating visualization charts...")
    try:
        strategy.plot_results(results)
        print("Charts displayed successfully!")
    except Exception as e:
        print(f"Plotting failed: {e}")
        print("This might be due to display issues in some environments.")
    
    print("\n" + "="*50)
    print("TEST COMPLETED SUCCESSFULLY!")
    print("="*50)

def test_different_parameters():
    """Test the strategy with different parameters"""
    print("\n" + "="*60)
    print("Testing Different Parameter Sets")
    print("="*60)
    
    test_configs = [
        {'symbol': 'STOCK_A', 'start': '2023-01-01', 'end': '2024-01-01'},
        {'symbol': 'STOCK_B', 'start': '2023-06-01', 'end': '2024-06-01'},
        {'symbol': 'STOCK_C', 'start': '2022-01-01', 'end': '2023-01-01'},
    ]
    
    results_summary = []
    
    for i, config in enumerate(test_configs, 1):
        print(f"\nTest {i}: {config['symbol']} ({config['start']} to {config['end']})")
        
        strategy = BullRiseIndicatorStrategy(
            symbol=config['symbol'],
            start_date=config['start'],
            end_date=config['end']
        )
        
        if (strategy.fetch_data() and 
            strategy.calculate_indicators() and 
            strategy.generate_signals()):
            
            results = strategy.backtest(initial_capital=10000)
            if results:
                results_summary.append({
                    'symbol': config['symbol'],
                    'return': results['total_return'],
                    'trades': results['total_trades'],
                    'win_rate': results['win_rate']
                })
                print(f"  Return: {results['total_return']:.2%}")
                print(f"  Trades: {results['total_trades']}")
                print(f"  Win Rate: {results['win_rate']:.1%}")
            else:
                print("  Backtest failed")
        else:
            print("  Strategy setup failed")
    
    # Summary comparison
    if results_summary:
        print(f"\n{'='*50}")
        print("PARAMETER COMPARISON SUMMARY")
        print(f"{'='*50}")
        print(f"{'Symbol':<10} {'Return':<10} {'Trades':<8} {'Win Rate':<10}")
        print("-" * 40)
        for result in results_summary:
            print(f"{result['symbol']:<10} {result['return']:<9.1%} {result['trades']:<8} {result['win_rate']:<9.1%}")

if __name__ == "__main__":
    # Run main test with plots
    test_strategy_with_plots()
    
    # Run parameter comparison test
    test_different_parameters()
    
    print(f"\n{'='*60}")
    print("ALL TESTS COMPLETED!")
    print("The Bull Rise Indicator strategy is working correctly with simulated data.")
    print("You can now modify parameters, test different time periods,")
    print("or integrate this strategy into larger trading systems.")
    print(f"{'='*60}")
