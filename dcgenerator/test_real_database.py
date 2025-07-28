#!/usr/bin/env python3
"""
Test DC Generator with your actual database - this WILL connect to your real data.
"""

import sqlite3
import pandas as pd
import numpy as np
import sys
import os
import time

# Import the DC generator
import dcgenerator as dg

def test_database_connection():
    """Test connection to your actual database."""
    db_path = "I:/zhubi/cpp_implementation/sqlite_databases/2018/2018_01.db"
    
    print("=== DC Generator Real Database Test ===")
    print(f"Attempting to connect to: {db_path}")
    
    if not os.path.exists(db_path):
        print(f"❌ Database file not found: {db_path}")
        return False
    
    try:
        conn = sqlite3.connect(db_path)
        print("✅ Successfully connected to database!")
        
        # Get database info
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM trade")
        total_records = cursor.fetchone()[0]
        print(f"📊 Total records in database: {total_records:,}")
        
        # Get available symbols
        cursor.execute("SELECT DISTINCT symbol FROM trade ORDER BY symbol LIMIT 10")
        symbols = [row[0] for row in cursor.fetchall()]
        print(f"📈 Available symbols (first 10): {symbols}")
        
        if symbols:
            # Test with first symbol
            test_symbol = symbols[0]
            print(f"\n=== Testing with symbol: {test_symbol} ===")
            
            # Get sample data
            cursor.execute("""
                SELECT date, time, price, volume, buysell 
                FROM trade 
                WHERE symbol = ? 
                ORDER BY date, time 
                LIMIT 50000
            """, (test_symbol,))
            
            rows = cursor.fetchall()
            print(f"✅ Loaded {len(rows)} trades for {test_symbol}")
            
            if rows:
                # Convert to DataFrame
                df = pd.DataFrame(rows, columns=['date', 'time', 'price', 'volume', 'buysell'])
                
                print(f"📅 Date range: {df['date'].min()} to {df['date'].max()}")
                print(f"💰 Price range: ${df['price'].min():.2f} to ${df['price'].max():.2f}")
                print(f"📊 Sample data:")
                print(df.head())
                
                # Test DC algorithm
                test_dc_algorithm(df, test_symbol)
                
        conn.close()
        return True
        
    except Exception as e:
        print(f"❌ Database error: {e}")
        return False

def test_dc_algorithm(df, symbol):
    """Test DC algorithm with real data."""
    print(f"\n=== DC Algorithm Test: {symbol} ===")
    
    prices = df['price'].values
    print(f"Processing {len(prices)} price points...")
    
    # Test different thresholds
    thresholds = [0.0005, 0.001, 0.002, 0.005]
    
    print(f"\n{'Threshold':<10} {'Events':<8} {'Upturn':<8} {'Downturn':<10} {'Time(ms)':<10}")
    print("-" * 50)
    
    for threshold in thresholds:
        start_time = time.time()
        
        # Apply DC generator
        events = dg.generate(prices, d=threshold)
        
        processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        
        # Count events
        event_counts = events.value_counts()
        total_events = len(events[events != ''])
        upturn_ends = event_counts.get('end upturn', 0)
        downturn_ends = event_counts.get('end downturn', 0)
        
        print(f"{threshold*100:>8.2f}%  {total_events:>6}  {upturn_ends:>6}  {downturn_ends:>8}  {processing_time:>8.2f}")
    
    # Test trading strategy with optimal threshold
    optimal_threshold = 0.001  # 0.1%
    test_trading_strategy(df, symbol, optimal_threshold)

def test_trading_strategy(df, symbol, threshold):
    """Test a simple trading strategy."""
    print(f"\n=== Trading Strategy Test: {symbol} (Threshold: {threshold*100:.1f}%) ===")
    
    prices = df['price'].values
    events = dg.generate(prices, d=threshold)
    
    # Trading simulation
    initial_capital = 100000.0
    cash = initial_capital
    position = 0.0
    trades = []
    
    print("Trading signals:")
    
    for i in range(len(prices)):
        current_price = prices[i]
        event = events.iloc[i] if i < len(events) else ''
        
        if event == 'end downturn' and position == 0:
            # Buy signal
            cash_to_use = cash * 0.95
            position = cash_to_use / current_price
            cash -= cash_to_use
            trades.append(('BUY', current_price, df.iloc[i]['date'], df.iloc[i]['time']))
            print(f"  BUY at ${current_price:.2f} on {df.iloc[i]['date']} {df.iloc[i]['time']}")
            
        elif event == 'end upturn' and position > 0:
            # Sell signal
            cash += position * current_price
            trades.append(('SELL', current_price, df.iloc[i]['date'], df.iloc[i]['time']))
            print(f"  SELL at ${current_price:.2f} on {df.iloc[i]['date']} {df.iloc[i]['time']}")
            position = 0.0
    
    # Calculate results
    final_value = cash + position * prices[-1]
    total_return = (final_value - initial_capital) / initial_capital * 100
    
    print(f"\n=== Strategy Results ===")
    print(f"Initial Capital: ${initial_capital:,.2f}")
    print(f"Final Value: ${final_value:,.2f}")
    print(f"Total Return: {total_return:.2f}%")
    print(f"Number of Trades: {len(trades)}")
    
    if len(trades) >= 2:
        print(f"First Trade: {trades[0][0]} at ${trades[0][1]:.2f} on {trades[0][2]}")
        print(f"Last Trade: {trades[-1][0]} at ${trades[-1][1]:.2f} on {trades[-1][2]}")

def main():
    """Main function to run the test."""
    print("Starting DC Generator test with your real trading database...")
    print("This will connect to your actual SQLite database and process real trading data.")
    print()
    
    success = test_database_connection()
    
    if success:
        print("\n✅ Test completed successfully!")
        print("The DC Generator algorithm has been tested with your real high-frequency trading data.")
    else:
        print("\n❌ Test failed. Please check database path and permissions.")

if __name__ == "__main__":
    main()
