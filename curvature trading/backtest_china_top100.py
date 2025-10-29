"""
Backtest Curved Radius Supertrend on Top 100 Chinese Stocks
Using Tushare API
"""

import pandas as pd
import numpy as np
from tushare_connector import TushareConnector
from backtest_engine import BacktestEngine
from datetime import datetime
import time

def backtest_single_stock(connector, ts_code, start_date, end_date, radius_strength=1.2, position_size=0.10):
    """
    Backtest a single Chinese stock
    
    Parameters:
    -----------
    connector : TushareConnector
        Tushare connector instance
    ts_code : str
        Tushare stock code (e.g., '600519.SH')
    start_date : str
        Start date (YYYY-MM-DD)
    end_date : str
        End date (YYYY-MM-DD)
    radius_strength : float
        Radius strength parameter
    position_size : float
        Position size as fraction of equity (default: 0.10 = 10%)
    
    Returns:
    --------
    dict : Backtest results or None if failed
    """
    try:
        # Fetch data
        data = connector.fetch_stock_data(ts_code, start_date, end_date)
        
        if data is None or len(data) < 100:  # Need minimum data
            return None
        
        # Run backtest with 10% position sizing
        engine = BacktestEngine(
            initial_capital=100000,
            commission=0.001,  # 0.1% commission
            slippage=0.0005,   # 0.05% slippage
            position_size=position_size,
            allow_short=False  # Chinese market typically doesn't allow shorting for retail
        )
        
        indicator_params = {
            'atr_period': 10,
            'atr_multiplier': 3.0,
            'radius_strength': radius_strength,
            'smoothness': 3
        }
        
        results = engine.run_backtest(data=data, indicator_params=indicator_params)
        stats = engine.calculate_statistics(data)
        
        # Get stock name
        stock_name = connector.get_stock_name(ts_code)
        
        return {
            'ts_code': ts_code,
            'name': stock_name,
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
            'days': len(data),
            'bankruptcy': stats.get('bankruptcy', False)
        }
        
    except Exception as e:
        print(f"   Error backtesting {ts_code}: {str(e)}")
        return None


def main():
    """
    Main function to backtest top 100 Chinese stocks
    """
    # Configuration
    TUSHARE_TOKEN = 'bfd5b1c0e45f3c7288e35d6ac2a0f0cc55279b233b37c6980cb61ab3'
    START_DATE = '2015-01-01'
    END_DATE = '2025-01-01'
    RADIUS_STRENGTH = 1.2
    POSITION_SIZE = 0.10  # 10% position sizing
    NUM_STOCKS = 100
    
    print("=" * 100)
    print("BACKTESTING TOP 100 CHINESE STOCKS (2015-2025)")
    print("=" * 100)
    print()
    print(f"Configuration:")
    print(f"  Period: {START_DATE} to {END_DATE}")
    print(f"  Position Size: {POSITION_SIZE * 100}%")
    print(f"  Radius Strength: {RADIUS_STRENGTH}")
    print(f"  Allow Short: No (Chinese market restriction)")
    print(f"  Initial Capital: ¥100,000")
    print()
    print("=" * 100)
    print()
    
    # Initialize connector
    connector = TushareConnector(TUSHARE_TOKEN)
    
    # Get top stocks
    print(f"Fetching top {NUM_STOCKS} Chinese stocks by market cap...")
    top_stocks = connector.get_top_stocks(n=NUM_STOCKS)
    print(f"✓ Found {len(top_stocks)} stocks")
    print()
    
    # Backtest each stock
    results = []
    start_time = time.time()
    
    for i, ts_code in enumerate(top_stocks, 1):
        print(f"[{i}/{len(top_stocks)}] Testing {ts_code}...", end=' ')
        
        result = backtest_single_stock(
            connector=connector,
            ts_code=ts_code,
            start_date=START_DATE,
            end_date=END_DATE,
            radius_strength=RADIUS_STRENGTH,
            position_size=POSITION_SIZE
        )
        
        if result is not None:
            results.append(result)
            print(f"✓ Return: {result['total_return']:.1f}%, Sharpe: {result['sharpe_ratio']:.2f}, Trades: {result['total_trades']}")
        else:
            print(f"✗ Failed (insufficient data)")
        
        # Rate limiting - Tushare has API limits
        if i % 10 == 0:
            time.sleep(1)  # Sleep 1 second every 10 requests
    
    elapsed_time = time.time() - start_time
    
    print()
    print("=" * 100)
    print(f"✅ Backtest complete! Tested {len(results)} stocks in {elapsed_time/60:.1f} minutes")
    print("=" * 100)
    print()
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Save results
    output_file = 'backtest_china_top100_2015_2025.csv'
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"📁 Results saved to: {output_file}")
    print()
    
    # Display summary statistics
    print("=" * 100)
    print("📊 SUMMARY STATISTICS")
    print("=" * 100)
    print()
    
    total = len(df)
    profitable = len(df[df['total_return'] > 0])
    bankrupt = len(df[df['total_return'] <= -100])
    
    print(f"Total Stocks Tested:     {total}")
    print(f"Profitable:              {profitable} ({profitable/total*100:.1f}%)")
    print(f"Losing:                  {total-profitable} ({(total-profitable)/total*100:.1f}%)")
    print(f"Bankruptcies:            {bankrupt} ({bankrupt/total*100:.1f}%)")
    print()
    
    print(f"Average Return:          {df['total_return'].mean():.2f}%")
    print(f"Median Return:           {df['total_return'].median():.2f}%")
    print(f"Best Return:             {df['total_return'].max():.2f}%")
    print(f"Worst Return:            {df['total_return'].min():.2f}%")
    print()
    
    print(f"Average Sharpe Ratio:    {df['sharpe_ratio'].mean():.2f}")
    print(f"Median Sharpe Ratio:     {df['sharpe_ratio'].median():.2f}")
    print(f"Average Max Drawdown:    {df['max_drawdown'].mean():.2f}%")
    print(f"Average Win Rate:        {df['win_rate'].mean():.2f}%")
    print()
    
    print(f"Total Trades (All):      {df['total_trades'].sum():.0f}")
    print(f"Average Trades/Stock:    {df['total_trades'].mean():.1f}")
    print()
    
    # Top 20 performers
    print("=" * 100)
    print("🏆 TOP 20 PERFORMERS (By Total Return)")
    print("=" * 100)
    print()
    
    top_20 = df.nlargest(20, 'total_return')
    
    print(f"{'Rank':<6} {'Code':<12} {'Name':<20} {'Return':<12} {'Sharpe':<10} {'Drawdown':<12} {'Trades'}")
    print("-" * 100)
    
    for idx, row in enumerate(top_20.itertuples(), 1):
        name = row.name[:18] if len(row.name) > 18 else row.name
        print(f"{idx:<6} {row.ts_code:<12} {name:<20} {row.total_return:>10.2f}% {row.sharpe_ratio:>9.2f} {row.max_drawdown:>10.2f}% {row.total_trades:>7.0f}")
    
    print()
    
    # Top 20 by Sharpe ratio
    print("=" * 100)
    print("🎯 TOP 20 BY SHARPE RATIO (Risk-Adjusted Returns)")
    print("=" * 100)
    print()
    
    top_20_sharpe = df.nlargest(20, 'sharpe_ratio')
    
    print(f"{'Rank':<6} {'Code':<12} {'Name':<20} {'Return':<12} {'Sharpe':<10} {'Drawdown':<12} {'Trades'}")
    print("-" * 100)
    
    for idx, row in enumerate(top_20_sharpe.itertuples(), 1):
        name = row.name[:18] if len(row.name) > 18 else row.name
        print(f"{idx:<6} {row.ts_code:<12} {name:<20} {row.total_return:>10.2f}% {row.sharpe_ratio:>9.2f} {row.max_drawdown:>10.2f}% {row.total_trades:>7.0f}")
    
    print()
    print("=" * 100)
    print("✅ ANALYSIS COMPLETE!")
    print("=" * 100)


if __name__ == "__main__":
    main()

