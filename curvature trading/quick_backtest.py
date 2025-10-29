"""
Quick Backtest Script

Simple script to quickly backtest a stock with the Curved Radius Supertrend strategy.
"""

import sys
from database_connector import StockDataConnector
from backtest_engine import BacktestEngine
from backtest_visualizer import plot_backtest_results
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


def quick_backtest(
    ticker: str,
    start_date: str = '2023-01-01',
    end_date: str = '2023-12-31',
    radius_strength: float = 0.5,
    show_plot: bool = True
):
    """
    Quick backtest for a single stock
    
    Parameters:
    -----------
    ticker : str
        Stock ticker symbol (e.g., 'AAPL', 'MSFT')
    start_date : str
        Start date (YYYY-MM-DD)
    end_date : str
        End date (YYYY-MM-DD)
    radius_strength : float
        Curvature parameter (0.2=tight, 0.5=medium, 1.0=wide)
    show_plot : bool
        Whether to display the plot
    """
    
    print(f"\n{'='*70}")
    print(f"QUICK BACKTEST: {ticker}")
    print(f"Period: {start_date} to {end_date}")
    print(f"Radius Strength: {radius_strength}")
    print(f"{'='*70}\n")
    
    # Fetch data
    print("Fetching data from database...")
    connector = StockDataConnector()
    
    try:
        data = connector.fetch_stock_data(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            min_volume=100000
        )
        
        if data.empty:
            print(f"❌ ERROR: No data found for {ticker}")
            return None
        
        if len(data) < 50:
            print(f"⚠️  WARNING: Only {len(data)} trading days found. Results may not be reliable.")
        else:
            print(f"✅ Retrieved {len(data)} trading days")
        
    finally:
        connector.close()
    
    # Run backtest
    print("\nRunning backtest...")
    
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
        'radius_strength': radius_strength,
        'smoothness': 3
    }
    
    results = engine.run_backtest(data, indicator_params)
    
    # Print results
    stats = results['statistics']
    
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    
    print(f"\n📊 Trade Statistics")
    print(f"   Total Trades:        {stats['total_trades']}")
    print(f"   Winning Trades:      {stats['winning_trades']} ({stats['win_rate']:.1f}%)")
    print(f"   Losing Trades:       {stats['losing_trades']}")
    print(f"   Avg Holding Period:  {stats['avg_bars_held']:.1f} days")
    
    print(f"\n💰 Performance")
    print(f"   Total Return:        {stats['total_return_pct']:+.2f}%")
    print(f"   Total P&L:           ${stats['total_pnl']:+,.2f}")
    print(f"   Avg P&L per Trade:   ${stats['avg_pnl_per_trade']:+,.2f}")
    print(f"   Average Win:         ${stats['avg_win']:,.2f}")
    print(f"   Average Loss:        ${stats['avg_loss']:,.2f}")
    print(f"   Profit Factor:       {stats['profit_factor']:.2f}")
    
    print(f"\n⚠️  Risk Metrics")
    print(f"   Sharpe Ratio:        {stats['sharpe_ratio']:.2f}")
    print(f"   Max Drawdown:        {stats['max_drawdown_pct']:.2f}%")
    
    print(f"\n💵 Capital")
    print(f"   Initial Capital:     $100,000.00")
    print(f"   Final Equity:        ${stats['final_equity']:,.2f}")
    
    # Performance rating
    print(f"\n📈 Performance Rating")
    
    score = 0
    if stats['total_return_pct'] > 0:
        score += 1
        print("   ✅ Positive return")
    else:
        print("   ❌ Negative return")
    
    if stats['win_rate'] >= 50:
        score += 1
        print("   ✅ Win rate >= 50%")
    else:
        print("   ❌ Win rate < 50%")
    
    if stats['sharpe_ratio'] > 1.0:
        score += 1
        print("   ✅ Sharpe ratio > 1.0")
    elif stats['sharpe_ratio'] > 0.5:
        print("   ⚠️  Sharpe ratio moderate")
    else:
        print("   ❌ Sharpe ratio low")
    
    if stats['max_drawdown_pct'] > -20:
        score += 1
        print("   ✅ Max drawdown < 20%")
    else:
        print("   ❌ Max drawdown > 20%")
    
    if stats['profit_factor'] > 1.5:
        score += 1
        print("   ✅ Profit factor > 1.5")
    elif stats['profit_factor'] > 1.0:
        print("   ⚠️  Profit factor moderate")
    else:
        print("   ❌ Profit factor < 1.0")
    
    print(f"\n   Overall Score: {score}/5")
    
    if score >= 4:
        print("   🌟 EXCELLENT - Strong performance!")
    elif score >= 3:
        print("   👍 GOOD - Decent performance")
    elif score >= 2:
        print("   😐 FAIR - Needs improvement")
    else:
        print("   ⚠️  POOR - Consider different parameters or stock")
    
    # Recent trades
    if results['trades']:
        print(f"\n📋 Recent Trades (Last 5)")
        print(f"   {'Date':<12} {'Type':<6} {'Entry':<10} {'Exit':<10} {'P&L':<12} {'Return':<10}")
        print("   " + "-"*66)
        
        for trade in results['trades'][-5:]:
            date_str = trade.entry_date.strftime('%Y-%m-%d')
            entry_str = f"${trade.entry_price:.2f}"
            exit_str = f"${trade.exit_price:.2f}"
            pnl_str = f"${trade.pnl:+.2f}"
            ret_str = f"{trade.return_pct:+.2f}%"
            
            print(f"   {date_str:<12} {trade.direction:<6} {entry_str:<10} {exit_str:<10} {pnl_str:<12} {ret_str:<10}")
    
    print("\n" + "="*70)
    
    # Create visualization
    if show_plot:
        print("\nGenerating visualization...")
        filename = f"backtest_{ticker.lower()}_{start_date[:4]}.png"
        plot_backtest_results(results, ticker=ticker, save_path=filename)
        print(f"✅ Chart saved to: {filename}")
        plt.show()
    
    return results


def main():
    """Main function with command line support"""
    
    # Default values
    ticker = 'AAPL'
    start_date = '2023-01-01'
    end_date = '2023-12-31'
    radius_strength = 0.5
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        ticker = sys.argv[1].upper()
    if len(sys.argv) > 2:
        start_date = sys.argv[2]
    if len(sys.argv) > 3:
        end_date = sys.argv[3]
    if len(sys.argv) > 4:
        radius_strength = float(sys.argv[4])
    
    # Run backtest
    results = quick_backtest(
        ticker=ticker,
        start_date=start_date,
        end_date=end_date,
        radius_strength=radius_strength,
        show_plot=True
    )
    
    if results is None:
        print("\n❌ Backtest failed. Please check the ticker symbol and date range.")
        return 1
    
    return 0


if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════════╗
║         CURVED RADIUS SUPERTREND - QUICK BACKTEST                ║
╚══════════════════════════════════════════════════════════════════╝

Usage:
    python quick_backtest.py [TICKER] [START_DATE] [END_DATE] [RADIUS]

Examples:
    python quick_backtest.py AAPL
    python quick_backtest.py AAPL 2023-01-01 2023-12-31
    python quick_backtest.py AAPL 2023-01-01 2023-12-31 0.5
    python quick_backtest.py GOOGL 2022-01-01 2023-12-31 1.0

Parameters:
    TICKER       - Stock ticker symbol (default: AAPL)
    START_DATE   - Start date YYYY-MM-DD (default: 2023-01-01)
    END_DATE     - End date YYYY-MM-DD (default: 2023-12-31)
    RADIUS       - Radius strength 0.1-2.0 (default: 0.5)
                   0.2 = tight/scalping
                   0.5 = medium/swing
                   1.0 = wide/position
    """)
    
    exit(main())

