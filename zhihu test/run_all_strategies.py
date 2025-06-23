"""
Comprehensive Test Runner for All 5 Trading Strategies

This script runs all 5 trading strategies with SIMULATED DATA and compares their performance:

1. Bull Rise Indicator (牛起指标) - Volume + Price breakthrough strategy
2. SF12Re Volatility Algorithm - Adaptive interval + volatility timing
3. 10-Day Low Point Buy Strategy - Simple but effective with 95% win rate
4. ETF Rotation Strategy - Multi-factor ETF selection with 21% annual return
5. VIX Fix + Fractal Chaos Band - Volatility-based panic trading

Usage:
python run_all_strategies.py

The script will:
- Run each strategy with simulated data (no internet required)
- Compare performance metrics
- Generate summary report
- Save results to CSV files

Note: All strategies now use realistic simulated data with different market characteristics:
- Strategy 1: General trending data with volatility
- Strategy 2: Data with varying volatility regimes
- Strategy 3: Trending data with pullbacks
- Strategy 4: Multiple ETFs with sector-specific behaviors
- Strategy 5: Data with panic and euphoria periods
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Import all strategy classes
from strategy_1_bull_rise_indicator import BullRiseIndicatorStrategy
from strategy_2_sf12re_volatility import SF12ReVolatilityStrategy
from strategy_3_10day_low_buy import TenDayLowBuyStrategy
from strategy_4_etf_rotation import ETFRotationStrategy
from strategy_5_vix_fix_fractal import VIXFixFractalStrategy

class StrategyComparison:
    def __init__(self, symbol='AAPL', start_date='2020-01-01', end_date='2024-12-31', 
                 initial_capital=10000):
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = initial_capital
        self.results = {}
        
    def run_strategy_1(self):
        """Run Bull Rise Indicator Strategy"""
        print("\n" + "="*60)
        print("RUNNING STRATEGY 1: Bull Rise Indicator (牛起指标)")
        print("="*60)
        
        try:
            strategy = BullRiseIndicatorStrategy(
                symbol=self.symbol,
                start_date=self.start_date,
                end_date=self.end_date
            )
            
            if strategy.fetch_data() and strategy.calculate_indicators() and strategy.generate_signals():
                results = strategy.backtest(initial_capital=self.initial_capital)
                if results:
                    self.results['Strategy 1: Bull Rise'] = {
                        'final_value': results['final_value'],
                        'total_return': results['total_return'],
                        'total_trades': results['total_trades'],
                        'win_rate': results['win_rate'],
                        'strategy_obj': strategy,
                        'results': results
                    }
                    print(f"✓ Strategy 1 completed successfully")
                    print(f"  Final Value: ${results['final_value']:,.2f}")
                    print(f"  Total Return: {results['total_return']:.2%}")
                    print(f"  Win Rate: {results['win_rate']:.1%}")
                else:
                    print("✗ Strategy 1 backtest failed")
            else:
                print("✗ Strategy 1 setup failed")
        except Exception as e:
            print(f"✗ Strategy 1 error: {e}")
    
    def run_strategy_2(self):
        """Run SF12Re Volatility Algorithm"""
        print("\n" + "="*60)
        print("RUNNING STRATEGY 2: SF12Re Volatility Algorithm")
        print("="*60)
        
        try:
            strategy = SF12ReVolatilityStrategy(
                symbol=self.symbol,
                start_date=self.start_date,
                end_date=self.end_date
            )
            
            if (strategy.fetch_data() and strategy.calculate_volatility_indicators() 
                and strategy.generate_signals()):
                results = strategy.backtest(initial_capital=self.initial_capital)
                if results:
                    self.results['Strategy 2: SF12Re'] = {
                        'final_value': results['final_value'],
                        'total_return': results['total_return'],
                        'total_trades': results['total_trades'],
                        'win_rate': results['win_rate'],
                        'strategy_obj': strategy,
                        'results': results
                    }
                    print(f"✓ Strategy 2 completed successfully")
                    print(f"  Final Value: ${results['final_value']:,.2f}")
                    print(f"  Total Return: {results['total_return']:.2%}")
                    print(f"  Win Rate: {results['win_rate']:.1%}")
                else:
                    print("✗ Strategy 2 backtest failed")
            else:
                print("✗ Strategy 2 setup failed")
        except Exception as e:
            print(f"✗ Strategy 2 error: {e}")
    
    def run_strategy_3(self):
        """Run 10-Day Low Point Buy Strategy"""
        print("\n" + "="*60)
        print("RUNNING STRATEGY 3: 10-Day Low Point Buy Strategy")
        print("="*60)
        
        try:
            strategy = TenDayLowBuyStrategy(
                symbol=self.symbol,
                start_date=self.start_date,
                end_date=self.end_date
            )
            
            if (strategy.fetch_data() and strategy.calculate_indicators() 
                and strategy.generate_signals()):
                results = strategy.backtest(initial_capital=self.initial_capital)
                if results:
                    self.results['Strategy 3: 10-Day Low'] = {
                        'final_value': results['final_value'],
                        'total_return': results['total_return'],
                        'total_trades': results['total_trades'],
                        'win_rate': results['win_rate'],
                        'avg_holding_days': results['avg_holding_days'],
                        'strategy_obj': strategy,
                        'results': results
                    }
                    print(f"✓ Strategy 3 completed successfully")
                    print(f"  Final Value: ${results['final_value']:,.2f}")
                    print(f"  Total Return: {results['total_return']:.2%}")
                    print(f"  Win Rate: {results['win_rate']:.1%}")
                    print(f"  Avg Holding Days: {results['avg_holding_days']:.1f}")
                else:
                    print("✗ Strategy 3 backtest failed")
            else:
                print("✗ Strategy 3 setup failed")
        except Exception as e:
            print(f"✗ Strategy 3 error: {e}")
    
    def run_strategy_4(self):
        """Run ETF Rotation Strategy"""
        print("\n" + "="*60)
        print("RUNNING STRATEGY 4: ETF Rotation Strategy")
        print("="*60)
        
        try:
            # Use a smaller ETF universe for faster testing
            etf_list = ['SPY', 'QQQ', 'IWM', 'EFA', 'VTI', 'XLF']
            
            strategy = ETFRotationStrategy(
                etf_symbols=etf_list,
                start_date=self.start_date,
                end_date=self.end_date,
                rebalance_frequency='M'
            )
            
            if (strategy.fetch_data() and strategy.calculate_all_scores() 
                and strategy.generate_positions()):
                results = strategy.backtest(initial_capital=self.initial_capital)
                if results:
                    self.results['Strategy 4: ETF Rotation'] = {
                        'final_value': results['final_value'],
                        'total_return': results['total_return'],
                        'annual_return': results['annual_return'],
                        'sharpe_ratio': results['sharpe_ratio'],
                        'max_drawdown': results['max_drawdown'],
                        'strategy_obj': strategy,
                        'results': results
                    }
                    print(f"✓ Strategy 4 completed successfully")
                    print(f"  Final Value: ${results['final_value']:,.2f}")
                    print(f"  Total Return: {results['total_return']:.2%}")
                    print(f"  Annual Return: {results['annual_return']:.2%}")
                    print(f"  Sharpe Ratio: {results['sharpe_ratio']:.2f}")
                else:
                    print("✗ Strategy 4 backtest failed")
            else:
                print("✗ Strategy 4 setup failed")
        except Exception as e:
            print(f"✗ Strategy 4 error: {e}")
    
    def run_strategy_5(self):
        """Run VIX Fix + Fractal Chaos Band Strategy"""
        print("\n" + "="*60)
        print("RUNNING STRATEGY 5: VIX Fix + Fractal Chaos Band Strategy")
        print("="*60)
        
        try:
            strategy = VIXFixFractalStrategy(
                symbol=self.symbol,
                start_date=self.start_date,
                end_date=self.end_date
            )
            
            if (strategy.fetch_data() and strategy.calculate_indicators() 
                and strategy.generate_signals()):
                results = strategy.backtest(initial_capital=self.initial_capital)
                if results:
                    self.results['Strategy 5: VIX Fix'] = {
                        'final_value': results['final_value'],
                        'total_return': results['total_return'],
                        'buy_hold_return': results['buy_hold_return'],
                        'excess_return': results['excess_return'],
                        'time_in_market': results['time_in_market'],
                        'total_trades': results['total_trades'],
                        'win_rate': results['win_rate'],
                        'strategy_obj': strategy,
                        'results': results
                    }
                    print(f"✓ Strategy 5 completed successfully")
                    print(f"  Final Value: ${results['final_value']:,.2f}")
                    print(f"  Total Return: {results['total_return']:.2%}")
                    print(f"  Excess Return: {results['excess_return']:.2%}")
                    print(f"  Time in Market: {results['time_in_market']:.1%}")
                    print(f"  Win Rate: {results['win_rate']:.1%}")
                else:
                    print("✗ Strategy 5 backtest failed")
            else:
                print("✗ Strategy 5 setup failed")
        except Exception as e:
            print(f"✗ Strategy 5 error: {e}")
    
    def run_all_strategies(self):
        """Run all strategies"""
        print(f"Starting comprehensive strategy comparison for {self.symbol}")
        print(f"Period: {self.start_date} to {self.end_date}")
        print(f"Initial Capital: ${self.initial_capital:,}")
        
        self.run_strategy_1()
        self.run_strategy_2()
        self.run_strategy_3()
        self.run_strategy_4()
        self.run_strategy_5()
        
        return len(self.results)
    
    def generate_comparison_report(self):
        """Generate comprehensive comparison report"""
        if not self.results:
            print("No results to compare")
            return
        
        print("\n" + "="*80)
        print("COMPREHENSIVE STRATEGY COMPARISON REPORT")
        print("="*80)
        
        # Create comparison DataFrame
        comparison_data = []
        for strategy_name, data in self.results.items():
            row = {
                'Strategy': strategy_name,
                'Final Value': data['final_value'],
                'Total Return': data['total_return'],
                'Total Trades': data.get('total_trades', 'N/A'),
                'Win Rate': data.get('win_rate', 'N/A')
            }
            
            # Add strategy-specific metrics
            if 'annual_return' in data:
                row['Annual Return'] = data['annual_return']
            if 'sharpe_ratio' in data:
                row['Sharpe Ratio'] = data['sharpe_ratio']
            if 'excess_return' in data:
                row['Excess Return'] = data['excess_return']
            if 'time_in_market' in data:
                row['Time in Market'] = data['time_in_market']
                
            comparison_data.append(row)
        
        df = pd.DataFrame(comparison_data)
        
        # Sort by total return
        df = df.sort_values('Total Return', ascending=False)
        
        print("\nPERFORMANCE RANKING:")
        print("-" * 80)
        for i, (_, row) in enumerate(df.iterrows(), 1):
            print(f"{i}. {row['Strategy']}")
            print(f"   Final Value: ${row['Final Value']:,.2f}")
            print(f"   Total Return: {row['Total Return']:.2%}")
            if pd.notna(row['Total Trades']) and row['Total Trades'] != 'N/A':
                print(f"   Total Trades: {row['Total Trades']}")
            if pd.notna(row['Win Rate']) and row['Win Rate'] != 'N/A':
                if isinstance(row['Win Rate'], (int, float)):
                    print(f"   Win Rate: {row['Win Rate']:.1%}")
                else:
                    print(f"   Win Rate: {row['Win Rate']}")
            print()
        
        # Save to CSV
        df.to_csv('strategy_comparison_results.csv', index=False)
        print("Results saved to 'strategy_comparison_results.csv'")
        
        return df
    
    def plot_comparison(self):
        """Plot comparison charts"""
        if not self.results:
            print("No results to plot")
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        strategies = list(self.results.keys())
        
        # Final values comparison
        final_values = [self.results[s]['final_value'] for s in strategies]
        ax1.bar(range(len(strategies)), final_values, color='skyblue')
        ax1.set_title('Final Portfolio Values')
        ax1.set_ylabel('Final Value ($)')
        ax1.set_xticks(range(len(strategies)))
        ax1.set_xticklabels([s.split(':')[0] for s in strategies], rotation=45)
        ax1.grid(True, alpha=0.3)
        
        # Total returns comparison
        total_returns = [self.results[s]['total_return'] * 100 for s in strategies]
        ax2.bar(range(len(strategies)), total_returns, color='lightgreen')
        ax2.set_title('Total Returns (%)')
        ax2.set_ylabel('Return (%)')
        ax2.set_xticks(range(len(strategies)))
        ax2.set_xticklabels([s.split(':')[0] for s in strategies], rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # Win rates comparison (where available)
        win_rates = []
        win_rate_labels = []
        for s in strategies:
            if 'win_rate' in self.results[s] and self.results[s]['win_rate'] is not None:
                win_rates.append(self.results[s]['win_rate'] * 100)
                win_rate_labels.append(s.split(':')[0])
        
        if win_rates:
            ax3.bar(range(len(win_rates)), win_rates, color='orange')
            ax3.set_title('Win Rates (%)')
            ax3.set_ylabel('Win Rate (%)')
            ax3.set_xticks(range(len(win_rates)))
            ax3.set_xticklabels(win_rate_labels, rotation=45)
            ax3.grid(True, alpha=0.3)
        
        # Trade counts comparison (where available)
        trade_counts = []
        trade_labels = []
        for s in strategies:
            if 'total_trades' in self.results[s] and self.results[s]['total_trades'] is not None:
                trade_counts.append(self.results[s]['total_trades'])
                trade_labels.append(s.split(':')[0])
        
        if trade_counts:
            ax4.bar(range(len(trade_counts)), trade_counts, color='lightcoral')
            ax4.set_title('Total Number of Trades')
            ax4.set_ylabel('Number of Trades')
            ax4.set_xticks(range(len(trade_counts)))
            ax4.set_xticklabels(trade_labels, rotation=45)
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('strategy_comparison_charts.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Comparison charts saved to 'strategy_comparison_charts.png'")

def main():
    """Main function to run all strategies and generate comparison"""
    print("="*80)
    print("COMPREHENSIVE TRADING STRATEGY COMPARISON")
    print("Testing 5 Different Quantitative Trading Strategies")
    print("Using SIMULATED DATA for reliable, fast testing")
    print("="*80)

    # Initialize comparison with simulated data
    comparison = StrategyComparison(
        symbol='SIMULATED_DATA',
        start_date='2020-01-01',
        end_date='2024-12-31',
        initial_capital=10000
    )
    
    # Run all strategies
    completed_strategies = comparison.run_all_strategies()
    
    if completed_strategies > 0:
        print(f"\n✓ Successfully completed {completed_strategies} out of 5 strategies")
        
        # Generate comparison report
        comparison.generate_comparison_report()
        
        # Plot comparison charts
        comparison.plot_comparison()
        
        print("\n" + "="*80)
        print("ANALYSIS COMPLETE!")
        print("Check the generated files:")
        print("- strategy_comparison_results.csv")
        print("- strategy_comparison_charts.png")
        print("="*80)
    else:
        print("\n✗ No strategies completed successfully")

if __name__ == "__main__":
    main()
