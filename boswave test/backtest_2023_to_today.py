"""
Backtest QQQ from 2023 to today using the Curved Radius Supertrend
"""

import pandas as pd
import numpy as np
import pymysql
import matplotlib.pyplot as plt
from datetime import datetime
import sys
sys.path.append('.')
from exact_pine_replication import CurvedRadiusSupertrendExact, BacktestEngine

def connect_to_nas():
    """Connect to NAS database"""
    config = {
        'host': '192.168.50.230',
        'port': 3306,
        'user': 'root',
        'password': '352471Cf!1',
        'database': 'us_stock_sip_min_aggs'
    }
    return pymysql.connect(**config)

def get_available_tables():
    """Get all available tables in the database"""
    connection = connect_to_nas()
    cursor = connection.cursor()
    cursor.execute("SHOW TABLES")
    tables = [table[0] for table in cursor.fetchall()]
    cursor.close()
    connection.close()
    return tables

def get_data_for_period(ticker, start_year_month, end_year_month):
    """
    Get data for a ticker across multiple months
    start_year_month: e.g., '202301' for January 2023
    end_year_month: e.g., '202410' for October 2024
    """
    
    print(f"\n{'='*80}")
    print(f"Fetching {ticker} data from {start_year_month} to {end_year_month}")
    print(f"{'='*80}\n")
    
    # Get available tables
    all_tables = get_available_tables()
    print(f"Total tables in database: {len(all_tables)}")
    
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
    print(f"Tables: {relevant_tables[:5]}...{relevant_tables[-5:] if len(relevant_tables) > 10 else ''}")
    
    # Fetch data from each table
    all_data = []
    connection = connect_to_nas()
    
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
                print(f"  ✓ {table}: {len(df):,} bars ({df['datetime'].min()} to {df['datetime'].max()})")
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
        print(f"COMBINED DATA SUMMARY")
        print(f"{'='*80}")
        print(f"Total bars: {len(combined_df):,}")
        print(f"Date range: {combined_df['datetime'].min()} to {combined_df['datetime'].max()}")
        print(f"Price range: ${combined_df['low'].min():.2f} - ${combined_df['high'].max():.2f}")
        print(f"{'='*80}\n")
        
        return combined_df
    else:
        print("\n❌ No data found for the specified period!")
        return None

def run_backtest_period(ticker='QQQ', start_ym='202301', end_ym='202410', 
                       radius_strength=0.10, initial_capital=10000):
    """
    Run backtest for a specific period
    """
    
    print("\n" + "="*80)
    print(f"CURVED RADIUS SUPERTREND BACKTEST")
    print(f"Symbol: {ticker}")
    print(f"Period: {start_ym} to {end_ym}")
    print(f"Radius Strength: {radius_strength}")
    print("="*80)
    
    # Get data
    df = get_data_for_period(ticker, start_ym, end_ym)
    
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
    print("BACKTEST RESULTS")
    print("="*80)
    
    print(f"\n{'PERIOD INFORMATION':^80}")
    print("-" * 80)
    print(f"Symbol:              {ticker}")
    print(f"Period:              {start_ym} to {end_ym}")
    print(f"Start Date:          {df['datetime'].min()}")
    print(f"End Date:            {df['datetime'].max()}")
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
    
    # Show sample trades
    if len(results['trades']) > 0:
        print(f"\n{'SAMPLE TRADES':^80}")
        print("-" * 80)
        print(f"First 10 trades:")
        print(f"{'Type':<8} {'Entry':<10} {'Exit':<10} {'P&L':<12} {'P&L %':<10}")
        print("-" * 80)
        
        for i, trade in results['trades'].head(10).iterrows():
            print(f"{trade['type']:<8} ${trade['entry_price']:<9.2f} "
                  f"${trade['exit_price']:<9.2f} ${trade['pnl']:<11,.2f} "
                  f"{trade['pnl_pct']:<9.2f}%")
        
        if len(results['trades']) > 10:
            print(f"\n... and {len(results['trades']) - 10:,} more trades")
    
    print("\n" + "="*80 + "\n")

def plot_backtest_results(df, signals, results, ticker, start_ym, end_ym, radius_strength):
    """Plot backtest results"""
    
    print("Creating charts...")
    
    # Sample data for plotting (use every Nth bar to avoid overcrowding)
    total_bars = len(df)
    if total_bars > 5000:
        sample_rate = total_bars // 5000
        plot_df = df.iloc[::sample_rate].reset_index(drop=True)
        plot_equity = results['equity'][::sample_rate]
        print(f"  Sampling data: Using every {sample_rate}th bar ({len(plot_df)} bars)")
    else:
        plot_df = df
        plot_equity = results['equity']
    
    fig, axes = plt.subplots(3, 1, figsize=(20, 14), sharex=True)
    
    # ========================================================================
    # Chart 1: Price
    # ========================================================================
    ax1 = axes[0]
    ax1.plot(plot_df.index, plot_df['close'], color='black', linewidth=1, label='Close Price')
    
    # Mark buy/sell signals on sampled data
    if total_bars > 5000:
        buy_idx = np.where(signals['buy_signals'][::sample_rate])[0]
        sell_idx = np.where(signals['sell_signals'][::sample_rate])[0]
    else:
        buy_idx = np.where(signals['buy_signals'])[0]
        sell_idx = np.where(signals['sell_signals'])[0]
    
    ax1.scatter(buy_idx, plot_df.iloc[buy_idx]['close'], marker='^', 
               color='green', s=100, label='Buy', zorder=5, alpha=0.7)
    ax1.scatter(sell_idx, plot_df.iloc[sell_idx]['close'], marker='v', 
               color='red', s=100, label='Sell', zorder=5, alpha=0.7)
    
    ax1.set_ylabel('Price ($)', fontsize=12, fontweight='bold')
    ax1.set_title(f'{ticker} - Curved Radius Supertrend Backtest ({start_ym} to {end_ym})', 
                 fontsize=14, fontweight='bold')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    
    # ========================================================================
    # Chart 2: Equity Curve
    # ========================================================================
    ax2 = axes[1]
    ax2.plot(plot_equity, color='blue', linewidth=2, label='Equity')
    ax2.axhline(y=results['metrics']['initial_capital'], color='gray', 
               linestyle='--', alpha=0.5, label='Initial Capital')
    ax2.fill_between(range(len(plot_equity)), results['metrics']['initial_capital'], plot_equity,
                     where=(plot_equity >= results['metrics']['initial_capital']),
                     color='green', alpha=0.2)
    ax2.fill_between(range(len(plot_equity)), results['metrics']['initial_capital'], plot_equity,
                     where=(plot_equity < results['metrics']['initial_capital']),
                     color='red', alpha=0.2)
    
    ax2.set_ylabel('Equity ($)', fontsize=12, fontweight='bold')
    ax2.set_title('Equity Curve', fontsize=12, fontweight='bold')
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
    
    if total_bars > 5000:
        plot_drawdown = drawdown[::sample_rate]
    else:
        plot_drawdown = drawdown
    
    ax3.fill_between(range(len(plot_drawdown)), 0, plot_drawdown, 
                     color='red', alpha=0.3)
    ax3.plot(plot_drawdown, color='red', linewidth=1)
    ax3.set_ylabel('Drawdown (%)', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Bar Index', fontsize=12, fontweight='bold')
    ax3.set_title(f'Drawdown (Max: {drawdown.min():.2f}%)', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    filename = f'backtest_{ticker}_{start_ym}_to_{end_ym}_RS{radius_strength}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"✓ Chart saved: {filename}")
    plt.show()

if __name__ == "__main__":
    print("\n" + "="*80)
    print("BACKTEST QQQ FROM 2023 TO TODAY")
    print("="*80)
    
    # Run backtest from January 2023 to October 2024
    results = run_backtest_period(
        ticker='QQQ',
        start_ym='202301',  # January 2023
        end_ym='202410',    # October 2024
        radius_strength=0.10,
        initial_capital=10000
    )
    
    if results:
        print("\n✓ Backtest completed successfully!")
    else:
        print("\n❌ Backtest failed!")

