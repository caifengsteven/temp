"""
Backtest Curved Radius Supertrend on Top 100 US Stocks (2020-2023)
"""

import numpy as np
import pandas as pd
from database_connector import StockDataConnector
from backtest_engine import BacktestEngine
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime
import time


# Top 100 US stocks by market cap (as of 2023)
TOP_100_STOCKS = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK.B', 'UNH', 'XOM',
    'JNJ', 'JPM', 'V', 'PG', 'MA', 'HD', 'CVX', 'MRK', 'ABBV', 'PEP',
    'COST', 'AVGO', 'KO', 'ADBE', 'WMT', 'MCD', 'CSCO', 'CRM', 'ACN', 'TMO',
    'LIN', 'ABT', 'NFLX', 'NKE', 'DHR', 'VZ', 'TXN', 'ORCL', 'PM', 'DIS',
    'CMCSA', 'INTC', 'WFC', 'AMD', 'UPS', 'NEE', 'COP', 'RTX', 'QCOM', 'HON',
    'INTU', 'UNP', 'IBM', 'LOW', 'AMGN', 'BA', 'SPGI', 'ELV', 'AMAT', 'GE',
    'CAT', 'SBUX', 'DE', 'PLD', 'BKNG', 'GILD', 'ADP', 'ADI', 'TJX', 'MDLZ',
    'CVS', 'LMT', 'SYK', 'VRTX', 'AXP', 'ISRG', 'MMC', 'CI', 'REGN', 'BLK',
    'ZTS', 'PGR', 'TMUS', 'MO', 'CB', 'SO', 'DUK', 'BSX', 'ETN', 'SCHW',
    'C', 'EOG', 'ITW', 'HCA', 'PNC', 'NOC', 'USB', 'SLB', 'MS', 'GD'
]


def backtest_single_stock(ticker, start_date, end_date, radius_strength=1.2):
    """
    Run backtest on a single stock
    """
    try:
        # Fetch data
        connector = StockDataConnector()
        data = connector.fetch_stock_data(ticker, start_date, end_date)
        connector.close()
        
        if len(data) < 50:  # Need minimum data
            return None
        
        # Run backtest
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
            'radius_strength': radius_strength,
            'smoothness': 3
        }
        
        results = engine.run_backtest(
            data=data,
            indicator_params=indicator_params
        )
        
        stats = engine.calculate_statistics(data)
        
        return {
            'ticker': ticker,
            'total_return': stats['total_return_pct'],
            'sharpe_ratio': stats['sharpe_ratio'],
            'max_drawdown': stats['max_drawdown_pct'],
            'win_rate': stats['win_rate'],
            'total_trades': stats['total_trades'],
            'profit_factor': stats['profit_factor'],
            'final_equity': stats['final_equity'],
            'avg_bars_held': stats['avg_bars_held'],
            'winning_trades': stats['winning_trades'],
            'losing_trades': stats['losing_trades'],
            'days': len(data)
        }
        
    except Exception as e:
        print(f"   ❌ Error with {ticker}: {str(e)[:50]}")
        return None


def run_top100_backtest(start_date='2020-01-01', end_date='2023-12-31', radius_strength=1.2):
    """
    Run backtest on top 100 stocks
    """
    print("="*80)
    print(f"BACKTESTING TOP 100 US STOCKS")
    print(f"Period: {start_date} to {end_date}")
    print(f"Radius Strength: {radius_strength}")
    print("="*80)
    
    results = []
    total_stocks = len(TOP_100_STOCKS)
    
    start_time = time.time()
    
    for idx, ticker in enumerate(TOP_100_STOCKS, 1):
        print(f"\n[{idx}/{total_stocks}] Testing {ticker}...", end=' ')
        
        result = backtest_single_stock(ticker, start_date, end_date, radius_strength)
        
        if result:
            results.append(result)
            print(f"✅ Return: {result['total_return']:>8.2f}% | "
                  f"Sharpe: {result['sharpe_ratio']:>5.2f} | "
                  f"Trades: {result['total_trades']:>3}")
        else:
            print(f"⚠️  Skipped (insufficient data)")
    
    elapsed_time = time.time() - start_time
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Save to CSV
    csv_filename = f'backtest_top100_{start_date[:4]}_{end_date[:4]}.csv'
    df.to_csv(csv_filename, index=False)
    
    print("\n" + "="*80)
    print(f"BACKTEST COMPLETE!")
    print(f"Tested: {len(results)} stocks")
    print(f"Time: {elapsed_time:.1f} seconds")
    print(f"Results saved to: {csv_filename}")
    print("="*80)
    
    return df


def analyze_results(df):
    """
    Analyze and display results
    """
    print("\n" + "="*80)
    print("ANALYSIS OF TOP 100 STOCKS BACKTEST")
    print("="*80)
    
    # Overall statistics
    print(f"\n📊 OVERALL STATISTICS")
    print(f"   Total Stocks Tested:     {len(df)}")
    print(f"   Profitable Stocks:       {len(df[df['total_return'] > 0])} ({len(df[df['total_return'] > 0])/len(df)*100:.1f}%)")
    print(f"   Losing Stocks:           {len(df[df['total_return'] <= 0])} ({len(df[df['total_return'] <= 0])/len(df)*100:.1f}%)")
    
    print(f"\n📈 RETURN STATISTICS")
    print(f"   Average Return:          {df['total_return'].mean():>10.2f}%")
    print(f"   Median Return:           {df['total_return'].median():>10.2f}%")
    print(f"   Best Return:             {df['total_return'].max():>10.2f}%")
    print(f"   Worst Return:            {df['total_return'].min():>10.2f}%")
    print(f"   Std Deviation:           {df['total_return'].std():>10.2f}%")
    
    print(f"\n🎯 RISK METRICS")
    print(f"   Average Sharpe Ratio:    {df['sharpe_ratio'].mean():>10.2f}")
    print(f"   Average Max Drawdown:    {df['max_drawdown'].mean():>10.2f}%")
    print(f"   Average Win Rate:        {df['win_rate'].mean():>10.2f}%")
    print(f"   Average Profit Factor:   {df['profit_factor'].mean():>10.2f}")
    
    print(f"\n📊 TRADE STATISTICS")
    print(f"   Total Trades (All):      {df['total_trades'].sum()}")
    print(f"   Average Trades/Stock:    {df['total_trades'].mean():>10.1f}")
    print(f"   Average Holding Period:  {df['avg_bars_held'].mean():>10.1f} days")
    
    # Top performers
    print("\n" + "="*80)
    print("🏆 TOP 10 PERFORMERS (by Total Return)")
    print("="*80)
    top10 = df.nlargest(10, 'total_return')
    print(f"\n{'Rank':<6} {'Ticker':<8} {'Return':<12} {'Sharpe':<10} {'Drawdown':<12} {'Trades':<10}")
    print("-" * 80)
    for idx, row in enumerate(top10.itertuples(), 1):
        print(f"{idx:<6} {row.ticker:<8} {row.total_return:>10.2f}% "
              f"{row.sharpe_ratio:>9.2f} {row.max_drawdown:>10.2f}% "
              f"{row.total_trades:>9}")
    
    # Worst performers
    print("\n" + "="*80)
    print("📉 BOTTOM 10 PERFORMERS (by Total Return)")
    print("="*80)
    bottom10 = df.nsmallest(10, 'total_return')
    print(f"\n{'Rank':<6} {'Ticker':<8} {'Return':<12} {'Sharpe':<10} {'Drawdown':<12} {'Trades':<10}")
    print("-" * 80)
    for idx, row in enumerate(bottom10.itertuples(), 1):
        print(f"{idx:<6} {row.ticker:<8} {row.total_return:>10.2f}% "
              f"{row.sharpe_ratio:>9.2f} {row.max_drawdown:>10.2f}% "
              f"{row.total_trades:>9}")
    
    # Best Sharpe ratios
    print("\n" + "="*80)
    print("🎯 TOP 10 BY SHARPE RATIO (Risk-Adjusted Returns)")
    print("="*80)
    top_sharpe = df.nlargest(10, 'sharpe_ratio')
    print(f"\n{'Rank':<6} {'Ticker':<8} {'Sharpe':<10} {'Return':<12} {'Drawdown':<12} {'Trades':<10}")
    print("-" * 80)
    for idx, row in enumerate(top_sharpe.itertuples(), 1):
        print(f"{idx:<6} {row.ticker:<8} {row.sharpe_ratio:>9.2f} "
              f"{row.total_return:>10.2f}% {row.max_drawdown:>10.2f}% "
              f"{row.total_trades:>9}")
    
    # Return distribution
    print("\n" + "="*80)
    print("📊 RETURN DISTRIBUTION")
    print("="*80)
    bins = [
        (float('-inf'), -50, 'Loss > 50%'),
        (-50, -20, 'Loss 20-50%'),
        (-20, 0, 'Loss 0-20%'),
        (0, 20, 'Gain 0-20%'),
        (20, 50, 'Gain 20-50%'),
        (50, 100, 'Gain 50-100%'),
        (100, 500, 'Gain 100-500%'),
        (500, float('inf'), 'Gain > 500%')
    ]
    
    for low, high, label in bins:
        count = len(df[(df['total_return'] > low) & (df['total_return'] <= high)])
        pct = count / len(df) * 100
        bar = '█' * int(pct / 2)
        print(f"{label:<20} {count:>3} ({pct:>5.1f}%) {bar}")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    # Run backtest
    print("\n🚀 Starting backtest on Top 100 US stocks...")
    print("This may take several minutes...\n")
    
    df_results = run_top100_backtest(
        start_date='2020-01-01',
        end_date='2023-12-31',
        radius_strength=1.2
    )
    
    # Analyze results
    analyze_results(df_results)
    
    print("\n✅ BACKTEST COMPLETE!")
    print(f"📁 Results saved to: backtest_top100_2020_2023.csv")

