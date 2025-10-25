"""
Verify that backtest includes both long and short trades
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

def get_daily_data_fast(ticker, start_ym, end_ym):
    """Get DAILY data for a ticker"""
    
    connection = connect_to_nas_daily()
    
    # Generate table list
    tables = []
    year = int(start_ym[:4])
    month = int(start_ym[4:])
    end_num = int(end_ym)
    
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

def verify_long_short_trades(ticker='AAPL'):
    """
    Verify that backtest includes both long and short trades
    """
    
    print("\n" + "="*80)
    print(f"VERIFYING LONG AND SHORT TRADES FOR {ticker}")
    print("="*80 + "\n")
    
    # Get data
    df = get_daily_data_fast(ticker, '202301', '202410')
    
    if df is None or len(df) < 50:
        print(f"❌ Insufficient data for {ticker}")
        return
    
    print(f"✓ Fetched {len(df)} bars")
    print(f"  Date range: {df['datetime'].min().date()} to {df['datetime'].max().date()}")
    
    # Calculate indicator
    indicator = CurvedRadiusSupertrendExact(
        atr_length=14,
        atr_mult=2.0,
        radius_strength=0.25,
        smoothness=5
    )
    
    signals = indicator.calculate(df['high'].values, df['low'].values, df['close'].values)
    
    print(f"\n✓ Signals calculated")
    print(f"  Buy signals: {signals['buy_signals'].sum()}")
    print(f"  Sell signals: {signals['sell_signals'].sum()}")
    
    # Run backtest
    backtester = BacktestEngine(
        initial_capital=10000, 
        commission=0.001,
        position_size=10000
    )
    results = backtester.run(df, signals)
    
    # Analyze trades
    trades_df = results['trades']
    
    if len(trades_df) == 0:
        print("\n❌ No trades generated!")
        return
    
    print(f"\n✓ Backtest complete")
    print(f"  Total trades: {len(trades_df)}")
    
    # Count long vs short trades
    long_trades = trades_df[trades_df['type'] == 'long']
    short_trades = trades_df[trades_df['type'] == 'short']
    
    print(f"\n{'='*80}")
    print(f"TRADE TYPE BREAKDOWN")
    print(f"{'='*80}\n")
    
    print(f"Long Trades:   {len(long_trades)} ({len(long_trades)/len(trades_df)*100:.1f}%)")
    print(f"Short Trades:  {len(short_trades)} ({len(short_trades)/len(trades_df)*100:.1f}%)")
    
    # Show statistics for each type
    print(f"\n{'LONG TRADES STATISTICS':^80}")
    print("-" * 80)
    if len(long_trades) > 0:
        long_wins = long_trades[long_trades['pnl'] > 0]
        long_losses = long_trades[long_trades['pnl'] < 0]
        print(f"Total:           {len(long_trades)}")
        print(f"Winners:         {len(long_wins)} ({len(long_wins)/len(long_trades)*100:.1f}%)")
        print(f"Losers:          {len(long_losses)} ({len(long_losses)/len(long_trades)*100:.1f}%)")
        print(f"Total P&L:       ${long_trades['pnl'].sum():,.2f}")
        print(f"Avg P&L:         ${long_trades['pnl'].mean():,.2f}")
        print(f"Avg Win:         ${long_wins['pnl'].mean():,.2f}" if len(long_wins) > 0 else "Avg Win:         N/A")
        print(f"Avg Loss:        ${long_losses['pnl'].mean():,.2f}" if len(long_losses) > 0 else "Avg Loss:        N/A")
    else:
        print("No long trades!")
    
    print(f"\n{'SHORT TRADES STATISTICS':^80}")
    print("-" * 80)
    if len(short_trades) > 0:
        short_wins = short_trades[short_trades['pnl'] > 0]
        short_losses = short_trades[short_trades['pnl'] < 0]
        print(f"Total:           {len(short_trades)}")
        print(f"Winners:         {len(short_wins)} ({len(short_wins)/len(short_trades)*100:.1f}%)")
        print(f"Losers:          {len(short_losses)} ({len(short_losses)/len(short_trades)*100:.1f}%)")
        print(f"Total P&L:       ${short_trades['pnl'].sum():,.2f}")
        print(f"Avg P&L:         ${short_trades['pnl'].mean():,.2f}")
        print(f"Avg Win:         ${short_wins['pnl'].mean():,.2f}" if len(short_wins) > 0 else "Avg Win:         N/A")
        print(f"Avg Loss:        ${short_losses['pnl'].mean():,.2f}" if len(short_losses) > 0 else "Avg Loss:        N/A")
    else:
        print("No short trades!")
    
    # Show all trades
    print(f"\n{'ALL TRADES':^80}")
    print("-" * 80)
    print(f"{'#':<4} {'Type':<8} {'Entry $':<10} {'Exit $':<10} {'P&L $':<12} {'P&L %':<10}")
    print("-" * 80)
    
    for i, trade in trades_df.iterrows():
        print(f"{i+1:<4} {trade['type']:<8} ${trade['entry_price']:<9.2f} "
              f"${trade['exit_price']:<9.2f} ${trade['pnl']:<11,.2f} {trade['pnl_pct']:<9.2f}%")
    
    print("\n" + "="*80)
    print(f"CONCLUSION: The backtest {'DOES' if len(short_trades) > 0 else 'DOES NOT'} include short trades!")
    print("="*80 + "\n")

if __name__ == "__main__":
    # Test with AAPL
    verify_long_short_trades('AAPL')
    
    # Test with another stock
    print("\n\n")
    verify_long_short_trades('ACB')

