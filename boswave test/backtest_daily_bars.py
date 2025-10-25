"""
Backtest QQQ from 2023 to today using DAILY bars from us_stock_sip_day_aggs
"""

import pandas as pd
import numpy as np
import pymysql
import matplotlib.pyplot as plt
from datetime import datetime
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
        'database': 'us_stock_sip_day_aggs'  # DAILY database
    }
    return pymysql.connect(**config)

def get_available_tables_daily():
    """Get all available tables in the daily database"""
    connection = connect_to_nas_daily()
    cursor = connection.cursor()
    cursor.execute("SHOW TABLES")
    tables = [table[0] for table in cursor.fetchall()]
    cursor.close()
    connection.close()
    return tables

def get_daily_data(ticker, start_year_month, end_year_month):
    """
    Get DAILY data for a ticker across multiple months
    """
    
    print(f"\n{'='*80}")
    print(f"Fetching {ticker} DAILY data from {start_year_month} to {end_year_month}")
    print(f"{'='*80}\n")
    
    # Get available tables
    all_tables = get_available_tables_daily()
    print(f"Total tables in daily database: {len(all_tables)}")
    
    # Filter tables for the period
    start_ym = int(start_year_month)
    end_ym = int(end_year_month)
    
    relevant_tables = []
    for table in all_tables:
        try:
            table_num = int(table)
            if start_ym <= table_num <= end_ym:
                relevant_tables.append(table)
        except:
            continue
    
    relevant_tables.sort()
    print(f"Relevant tables for period: {len(relevant_tables)}")
    print(f"Tables: {relevant_tables}")
    
    # Fetch data from each table
    all_data = []
    connection = connect_to_nas_daily()
    
    for table in relevant_tables:
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
                print(f"  ✓ {table}: {len(df):,} bars ({df['datetime'].min().date()} to {df['datetime'].max().date()})")
            else:
                print(f"  ✗ {table}: No data for {ticker}")
        
        except Exception as e:
            print(f"  ✗ {table}: Error - {e}")
    
    connection.close()
    
    # Combine all data
    if len(all_data) > 0:
        combined_df = pd.concat(all_data, ignore_index=True)
        combined_df = combined_df.sort_values('datetime').reset_index(drop=True)
        
        print(f"\n{'='*80}")
        print(f"COMBINED DAILY DATA SUMMARY")
        print(f"{'='*80}")
        print(f"Total bars: {len(combined_df):,}")
        print(f"Date range: {combined_df['datetime'].min().date()} to {combined_df['datetime'].max().date()}")
        print(f"Price range: ${combined_df['low'].min():.2f} - ${combined_df['high'].max():.2f}")
        print(f"{'='*80}\n")
        
        return combined_df
    else:
        print("\n❌ No data found for the specified period!")
        return None

def run_backtest_daily(ticker='QQQ', start_ym='202301', end_ym='202410', 
                       radius_strength=0.25, initial_capital=10000):
    """
    Run backtest on DAILY bars with adjusted radius_strength
    """
    
    print("\n" + "="*80)
    print(f"CURVED RADIUS SUPERTREND BACKTEST - DAILY BARS")
    print(f"Symbol: {ticker}")
    print(f"Period: {start_ym} to {end_ym}")
    print(f"Radius Strength: {radius_strength} (recommended for daily: 0.20-0.25)")
    print("="*80)
    
    # Get data
    df = get_daily_data(ticker, start_ym, end_ym)
    
    if df is None or len(df) == 0:
        print("❌ No data available for backtesting!")
        return None
    
    # Calculate indicator
    print(f"\n[1/3] Calculating Curved Radius Supertrend...")
    indicator = CurvedRadiusSupertrendExact(
        atr_length=14,
        atr_mult=2.0,
        radius_strength=radius_strength,
        smoothness=5
    )
    
    signals = indicator.calculate(df['high'].values, df['low'].values, df['close'].values)
    
    print(f"✓ Indicator calculated")
    print(f"  Buy signals: {signals['buy_signals'].sum():,}")
    print(f"  Sell signals: {signals['sell_signals'].sum():,}")
    print(f"  Signal frequency: {(signals['buy_signals'].sum() + signals['sell_signals'].sum()) / len(df) * 100:.2f}% of bars")
    
    # Run backtest
    print(f"\n[2/3] Running backtest...")
    backtester = BacktestEngine(initial_capital=initial_capital, commission=0.001)
    results = backtester.run(df, signals)
    
    print(f"✓ Backtest complete")
    print(f"  Total trades: {len(results['trades']):,}")
    
    # Print results
    print(f"\n[3/3] Generating report...")
    print_backtest_report(results, ticker, start_ym, end_ym, radius_strength, df)
    
    # Plot results
    plot_backtest_results(df, signals, results, ticker, start_ym, end_ym, radius_strength)
    
    return results

def print_backtest_report(results, ticker, start_ym, end_ym, radius_strength, df):
    """Print detailed backtest report"""
    
    metrics = results['metrics']
    
    print("\n" + "="*80)
    print("BACKTEST RESULTS - DAILY BARS")
    print("="*80)
    
    print(f"\n{'PERIOD INFORMATION':^80}")
    print("-" * 80)
    print(f"Symbol:              {ticker}")
    print(f"Timeframe:           DAILY")
    print(f"Period:              {start_ym} to {end_ym}")
    print(f"Start Date:          {df['datetime'].min().date()}")
    print(f"End Date:            {df['datetime'].max().date()}")
    print(f"Total Bars:          {len(df):,}")
    print(f"Radius Strength:     {radius_strength}")
    
    print(f"\n{'PERFORMANCE METRICS':^80}")
    print("-" * 80)
    print(f"Initial Capital:     ${metrics['initial_capital']:>15,.2f}")
    print(f"Final Capital:       ${metrics['final_capital']:>15,.2f}")
    print(f"Total Return:        ${metrics['total_return']:>15,.2f}")
    print(f"Return %:            {metrics['total_return_pct']:>15.2f}%")
    
    # Calculate additional metrics
    days = (df['datetime'].max() - df['datetime'].min()).days
    if days > 0:
        annual_return = (metrics['total_return_pct'] / days) * 365
        print(f"Annualized Return:   {annual_return:>15.2f}%")
    
    # Buy & Hold comparison
    buy_hold_return = ((df.iloc[-1]['close'] - df.iloc[0]['close']) / df.iloc[0]['close']) * 100
    print(f"\nBuy & Hold Return:   {buy_hold_return:>15.2f}%")
    print(f"Strategy vs B&H:     {metrics['total_return_pct'] - buy_hold_return:>15.2f}%")
    
    print(f"\n{'TRADE STATISTICS':^80}")
    print("-" * 80)
    print(f"Total Trades:        {metrics['total_trades']:>15,}")
    print(f"Winning Trades:      {metrics['winning_trades']:>15,}")
    print(f"Losing Trades:       {metrics['losing_trades']:>15,}")
    print(f"Win Rate:            {metrics['win_rate']:>15.2f}%")
    
    if metrics['total_trades'] > 0:
        print(f"Average Win:         ${metrics['avg_win']:>15,.2f}")
        print(f"Average Loss:        ${metrics['avg_loss']:>15,.2f}")
        print(f"Profit Factor:       {metrics['profit_factor']:>15.2f}")
        print(f"Avg Trade:           ${(metrics['total_return'] / metrics['total_trades']):>15,.2f}")
    
    # Show all trades
    if len(results['trades']) > 0:
        print(f"\n{'ALL TRADES':^80}")
        print("-" * 80)
        print(f"{'#':<4} {'Type':<8} {'Entry $':<10} {'Exit $':<10} {'P&L $':<12} {'P&L %':<10}")
        print("-" * 80)

        for i, trade in results['trades'].iterrows():
            print(f"{i+1:<4} {trade['type']:<8} ${trade['entry_price']:<9.2f} "
                  f"${trade['exit_price']:<9.2f} ${trade['pnl']:<11,.2f} {trade['pnl_pct']:<9.2f}%")
    
    print("\n" + "="*80 + "\n")

def plot_backtest_results(df, signals, results, ticker, start_ym, end_ym, radius_strength):
    """Plot backtest results for daily data"""
    
    print("Creating charts...")
    
    fig, axes = plt.subplots(3, 1, figsize=(20, 14), sharex=True)
    
    # ========================================================================
    # Chart 1: Price with signals
    # ========================================================================
    ax1 = axes[0]
    ax1.plot(df.index, df['close'], color='black', linewidth=1.5, label='Close Price')
    
    # Mark buy/sell signals
    buy_idx = np.where(signals['buy_signals'])[0]
    sell_idx = np.where(signals['sell_signals'])[0]
    
    ax1.scatter(buy_idx, df.iloc[buy_idx]['close'], marker='^', 
               color='green', s=150, label='Buy', zorder=5, edgecolors='darkgreen', linewidths=2)
    ax1.scatter(sell_idx, df.iloc[sell_idx]['close'], marker='v', 
               color='red', s=150, label='Sell', zorder=5, edgecolors='darkred', linewidths=2)
    
    # Add trend background
    direction = signals['direction']
    for i in range(len(direction)):
        if direction[i] == 1:
            ax1.axvspan(i-0.5, i+0.5, alpha=0.05, color='green', zorder=0)
        else:
            ax1.axvspan(i-0.5, i+0.5, alpha=0.05, color='red', zorder=0)
    
    ax1.set_ylabel('Price ($)', fontsize=12, fontweight='bold')
    ax1.set_title(f'{ticker} DAILY - Curved Radius Supertrend Backtest ({start_ym} to {end_ym}, RS={radius_strength})', 
                 fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # ========================================================================
    # Chart 2: Equity Curve
    # ========================================================================
    ax2 = axes[1]
    equity = results['equity']
    ax2.plot(equity, color='blue', linewidth=2, label='Equity')
    ax2.axhline(y=results['metrics']['initial_capital'], color='gray', 
               linestyle='--', alpha=0.5, label='Initial Capital')
    ax2.fill_between(range(len(equity)), results['metrics']['initial_capital'], equity,
                     where=(equity >= results['metrics']['initial_capital']),
                     color='green', alpha=0.2)
    ax2.fill_between(range(len(equity)), results['metrics']['initial_capital'], equity,
                     where=(equity < results['metrics']['initial_capital']),
                     color='red', alpha=0.2)
    
    # Mark trades on equity curve
    if len(buy_idx) > 0:
        ax2.scatter(buy_idx, equity[buy_idx], marker='^', color='green', s=100, zorder=5, alpha=0.7)
    if len(sell_idx) > 0:
        ax2.scatter(sell_idx, equity[sell_idx], marker='v', color='red', s=100, zorder=5, alpha=0.7)
    
    ax2.set_ylabel('Equity ($)', fontsize=12, fontweight='bold')
    ax2.set_title(f'Equity Curve (Final: ${results["metrics"]["final_capital"]:,.2f}, Return: {results["metrics"]["total_return_pct"]:.2f}%)', 
                 fontsize=12, fontweight='bold')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    
    # ========================================================================
    # Chart 3: Drawdown
    # ========================================================================
    ax3 = axes[2]
    
    # Calculate drawdown
    equity_curve = results['equity']
    running_max = np.maximum.accumulate(equity_curve)
    drawdown = ((equity_curve - running_max) / running_max) * 100
    
    ax3.fill_between(range(len(drawdown)), 0, drawdown, color='red', alpha=0.3)
    ax3.plot(drawdown, color='red', linewidth=1.5)
    ax3.set_ylabel('Drawdown (%)', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Bar Index (Days)', fontsize=12, fontweight='bold')
    ax3.set_title(f'Drawdown (Max: {drawdown.min():.2f}%)', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    filename = f'backtest_DAILY_{ticker}_{start_ym}_to_{end_ym}_RS{radius_strength}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"✓ Chart saved: {filename}")
    plt.show()

if __name__ == "__main__":
    print("\n" + "="*80)
    print("BACKTEST QQQ - DAILY BARS FROM 2023 TO TODAY")
    print("="*80)
    
    # Test with different radius_strength values recommended for daily timeframe
    test_params = [0.20, 0.25, 0.30]
    
    all_results = {}
    
    for rs in test_params:
        print(f"\n\n{'#'*80}")
        print(f"# TESTING WITH RADIUS_STRENGTH = {rs}")
        print(f"{'#'*80}\n")
        
        results = run_backtest_daily(
            ticker='QQQ',
            start_ym='202301',  # January 2023
            end_ym='202410',    # October 2024
            radius_strength=rs,
            initial_capital=10000
        )
        
        if results:
            all_results[rs] = results
    
    # Summary comparison
    if len(all_results) > 0:
        print("\n" + "="*80)
        print("COMPARISON OF DIFFERENT RADIUS_STRENGTH VALUES")
        print("="*80)
        print(f"{'RS':<8} {'Return %':<15} {'Trades':<10} {'Win Rate':<12} {'Profit Factor':<15}")
        print("-" * 80)
        
        for rs, res in all_results.items():
            m = res['metrics']
            print(f"{rs:<8} {m['total_return_pct']:<15.2f} {m['total_trades']:<10} "
                  f"{m['win_rate']:<12.2f} {m['profit_factor']:<15.2f}")
        
        print("="*80)
        
        # Find best
        best_rs = max(all_results.keys(), key=lambda x: all_results[x]['metrics']['total_return_pct'])
        print(f"\n✓ Best Radius Strength: {best_rs} with {all_results[best_rs]['metrics']['total_return_pct']:.2f}% return")
    
    print("\n✓ All backtests completed!")

