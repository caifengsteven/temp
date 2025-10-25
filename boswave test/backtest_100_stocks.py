"""
Backtest 100 US stocks on DAILY bars from 2023 to today
"""

import pandas as pd
import numpy as np
import pymysql
import sys
sys.path.append('.')
from exact_pine_replication import CurvedRadiusSupertrendExact, BacktestEngine

def connect_to_nas_daily():
    """Connect to NAS database - DAILY bars"""
    config = {
        'host': '192.168.50.230',
        'port': 3306,
        'user': 'root',
        'password': '352471Cf!1',
        'database': 'us_stock_sip_day_aggs'
    }
    return pymysql.connect(**config)

def get_all_tickers():
    """Get list of all available tickers"""
    connection = connect_to_nas_daily()
    cursor = connection.cursor()
    
    # Get tickers from recent table (202410)
    query = """
    SELECT DISTINCT ticker 
    FROM `202410`
    ORDER BY ticker
    LIMIT 100
    """
    
    cursor.execute(query)
    tickers = [row[0] for row in cursor.fetchall()]
    
    cursor.close()
    connection.close()
    
    return tickers

def get_daily_data_fast(ticker, start_ym, end_ym):
    """Get DAILY data for a ticker - fast version without printing"""
    
    connection = connect_to_nas_daily()
    
    # Get all tables in range
    start_num = int(start_ym)
    end_num = int(end_ym)
    
    # Generate table list
    tables = []
    year = int(start_ym[:4])
    month = int(start_ym[4:])
    
    while int(f"{year}{month:02d}") <= end_num:
        tables.append(f"{year}{month:02d}")
        month += 1
        if month > 12:
            month = 1
            year += 1
    
    # Fetch data from all tables
    all_data = []
    
    for table in tables:
        try:
            query = f"""
            SELECT window_start, open, high, low, close, volume
            FROM `{table}`
            WHERE ticker = %s
            ORDER BY window_start ASC
            """
            
            df = pd.read_sql(query, connection, params=(ticker,))
            
            if len(df) > 0:
                df['datetime'] = pd.to_datetime(df['window_start'], unit='ns')
                for col in ['open', 'high', 'low', 'close']:
                    df[col] = df[col].astype(float)
                all_data.append(df)
        except:
            pass
    
    connection.close()
    
    # Combine all data
    if len(all_data) > 0:
        combined_df = pd.concat(all_data, ignore_index=True)
        combined_df = combined_df.sort_values('datetime').reset_index(drop=True)
        return combined_df
    else:
        return None

def backtest_single_stock(ticker, start_ym='202301', end_ym='202410',
                         radius_strength=0.25, initial_capital=10000):
    """
    Backtest a single stock - no printing, just return results
    Uses COMPOUNDING - each trade uses current equity
    """

    # Get data
    df = get_daily_data_fast(ticker, start_ym, end_ym)

    if df is None or len(df) < 50:  # Need at least 50 bars
        return None

    try:
        # Calculate indicator
        indicator = CurvedRadiusSupertrendExact(
            atr_length=14,
            atr_mult=2.0,
            radius_strength=radius_strength,
            smoothness=5
        )

        signals = indicator.calculate(df['high'].values, df['low'].values, df['close'].values)

        # Run backtest with COMPOUNDING
        backtester = BacktestEngine(
            initial_capital=initial_capital,
            commission=0.001
        )
        results = backtester.run(df, signals)
        
        # Calculate buy & hold
        buy_hold_return = ((df.iloc[-1]['close'] - df.iloc[0]['close']) / df.iloc[0]['close']) * 100
        
        # Return summary
        return {
            'ticker': ticker,
            'bars': len(df),
            'start_date': df['datetime'].min().date(),
            'end_date': df['datetime'].max().date(),
            'start_price': df.iloc[0]['close'],
            'end_price': df.iloc[-1]['close'],
            'total_trades': results['metrics']['total_trades'],
            'win_rate': results['metrics']['win_rate'],
            'profit_factor': results['metrics']['profit_factor'],
            'total_return_pct': results['metrics']['total_return_pct'],
            'buy_hold_return': buy_hold_return,
            'vs_buy_hold': results['metrics']['total_return_pct'] - buy_hold_return,
            'final_capital': results['metrics']['final_capital'],
            'winning_trades': results['metrics']['winning_trades'],
            'losing_trades': results['metrics']['losing_trades'],
        }
    
    except Exception as e:
        return None

def run_100_stocks_backtest():
    """
    Run backtest on 100 stocks
    """
    
    print("\n" + "="*100)
    print("BACKTESTING 100 US STOCKS - DAILY BARS (2023-2024)")
    print("="*100)
    
    # Get tickers
    print("\n[1/3] Getting list of 100 stocks...")
    tickers = get_all_tickers()
    print(f"✓ Found {len(tickers)} tickers")
    print(f"Tickers: {', '.join(tickers[:20])}...")
    
    # Run backtests
    print(f"\n[2/3] Running backtests on {len(tickers)} stocks...")
    print("This may take a few minutes...\n")
    
    results = []
    successful = 0
    failed = 0
    
    for i, ticker in enumerate(tickers):
        result = backtest_single_stock(ticker, start_ym='202301', end_ym='202410',
                                      radius_strength=0.25, initial_capital=10000)
        
        if result:
            results.append(result)
            successful += 1
            print(f"  [{i+1}/{len(tickers)}] ✓ {ticker:<6} - Return: {result['total_return_pct']:>10.2f}% | "
                  f"Trades: {result['total_trades']:>3} | Win Rate: {result['win_rate']:>5.1f}%")
        else:
            failed += 1
            print(f"  [{i+1}/{len(tickers)}] ✗ {ticker:<6} - Insufficient data or error")
    
    print(f"\n✓ Completed: {successful} successful, {failed} failed")
    
    # Analyze results
    print(f"\n[3/3] Analyzing results...")
    
    if len(results) == 0:
        print("❌ No successful backtests!")
        return None
    
    # Convert to DataFrame
    df_results = pd.DataFrame(results)
    
    # Print summary statistics
    print("\n" + "="*100)
    print("SUMMARY STATISTICS")
    print("="*100)
    
    print(f"\nTotal stocks tested:     {len(tickers)}")
    print(f"Successful backtests:    {len(results)}")
    print(f"Failed backtests:        {failed}")
    
    print(f"\n{'RETURN STATISTICS':^100}")
    print("-" * 100)
    print(f"Average Return:          {df_results['total_return_pct'].mean():>15.2f}%")
    print(f"Median Return:           {df_results['total_return_pct'].median():>15.2f}%")
    print(f"Best Return:             {df_results['total_return_pct'].max():>15.2f}% ({df_results.loc[df_results['total_return_pct'].idxmax(), 'ticker']})")
    print(f"Worst Return:            {df_results['total_return_pct'].min():>15.2f}% ({df_results.loc[df_results['total_return_pct'].idxmin(), 'ticker']})")
    print(f"Std Deviation:           {df_results['total_return_pct'].std():>15.2f}%")
    
    print(f"\n{'BUY & HOLD COMPARISON':^100}")
    print("-" * 100)
    print(f"Avg Buy & Hold Return:   {df_results['buy_hold_return'].mean():>15.2f}%")
    print(f"Avg Strategy Return:     {df_results['total_return_pct'].mean():>15.2f}%")
    print(f"Avg Outperformance:      {df_results['vs_buy_hold'].mean():>15.2f}%")
    
    # Count winners vs losers
    positive_returns = (df_results['total_return_pct'] > 0).sum()
    negative_returns = (df_results['total_return_pct'] < 0).sum()
    beat_buy_hold = (df_results['vs_buy_hold'] > 0).sum()
    
    print(f"\nPositive Returns:        {positive_returns} ({positive_returns/len(results)*100:.1f}%)")
    print(f"Negative Returns:        {negative_returns} ({negative_returns/len(results)*100:.1f}%)")
    print(f"Beat Buy & Hold:         {beat_buy_hold} ({beat_buy_hold/len(results)*100:.1f}%)")
    
    print(f"\n{'TRADE STATISTICS':^100}")
    print("-" * 100)
    print(f"Avg Trades per Stock:    {df_results['total_trades'].mean():>15.1f}")
    print(f"Avg Win Rate:            {df_results['win_rate'].mean():>15.2f}%")
    print(f"Avg Profit Factor:       {df_results['profit_factor'].mean():>15.2f}")
    
    # Top 10 performers
    print(f"\n{'TOP 10 PERFORMERS':^100}")
    print("-" * 100)
    print(f"{'Rank':<6} {'Ticker':<8} {'Return %':<15} {'Trades':<10} {'Win Rate':<12} {'vs B&H':<15}")
    print("-" * 100)
    
    top_10 = df_results.nlargest(10, 'total_return_pct')
    for i, row in top_10.iterrows():
        print(f"{top_10.index.get_loc(i)+1:<6} {row['ticker']:<8} {row['total_return_pct']:<15.2f} "
              f"{row['total_trades']:<10} {row['win_rate']:<12.2f} {row['vs_buy_hold']:<15.2f}")
    
    # Bottom 10 performers
    print(f"\n{'BOTTOM 10 PERFORMERS':^100}")
    print("-" * 100)
    print(f"{'Rank':<6} {'Ticker':<8} {'Return %':<15} {'Trades':<10} {'Win Rate':<12} {'vs B&H':<15}")
    print("-" * 100)
    
    bottom_10 = df_results.nsmallest(10, 'total_return_pct')
    for i, row in bottom_10.iterrows():
        print(f"{bottom_10.index.get_loc(i)+1:<6} {row['ticker']:<8} {row['total_return_pct']:<15.2f} "
              f"{row['total_trades']:<10} {row['win_rate']:<12.2f} {row['vs_buy_hold']:<15.2f}")
    
    # Save results to CSV
    csv_filename = 'backtest_100_stocks_results.csv'
    df_results.to_csv(csv_filename, index=False)
    print(f"\n✓ Results saved to: {csv_filename}")
    
    print("\n" + "="*100 + "\n")
    
    return df_results

if __name__ == "__main__":
    results = run_100_stocks_backtest()
    
    if results is not None:
        print("✓ Backtest completed successfully!")
        print(f"\nKey Takeaway:")
        print(f"  • Average Return: {results['total_return_pct'].mean():.2f}%")
        print(f"  • {(results['total_return_pct'] > 0).sum()}/{len(results)} stocks profitable")
        print(f"  • {(results['vs_buy_hold'] > 0).sum()}/{len(results)} stocks beat buy & hold")
    else:
        print("❌ Backtest failed!")

