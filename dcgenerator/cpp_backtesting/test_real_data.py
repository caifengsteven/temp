#!/usr/bin/env python3
"""
Test DC Generator with your real database data.
This Python version will connect to your actual database and test the DC algorithm.
"""

import sqlite3
import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime
import time

# Add the dcgenerator module to path
sys.path.append('..')
import dcgenerator as dg

def connect_to_database(db_path):
    """Connect to your SQLite database."""
    try:
        conn = sqlite3.connect(db_path)
        print(f"✅ Successfully connected to: {db_path}")
        return conn
    except Exception as e:
        print(f"❌ Failed to connect to database: {e}")
        return None

def get_available_symbols(conn, limit=20):
    """Get available symbols from your database."""
    try:
        query = """
        SELECT symbol, COUNT(*) as trade_count,
               MIN(price) as min_price, MAX(price) as max_price,
               MIN(date) as start_date, MAX(date) as end_date
        FROM trade 
        GROUP BY symbol 
        ORDER BY trade_count DESC
        LIMIT ?
        """
        df = pd.read_sql_query(query, conn, params=(limit,))
        return df
    except Exception as e:
        print(f"Error getting symbols: {e}")
        return None

def load_trading_data(conn, symbol, limit=50000):
    """Load actual trading data from your database."""
    try:
        query = """
        SELECT date, time, price, volume, buysell 
        FROM trade 
        WHERE symbol = ? 
        ORDER BY date, time 
        LIMIT ?
        """
        df = pd.read_sql_query(query, conn, params=(symbol, limit))
        
        if df.empty:
            print(f"No data found for symbol: {symbol}")
            return None
            
        print(f"✅ Loaded {len(df)} trades for {symbol}")
        print(f"   Date range: {df['date'].min()} to {df['date'].max()}")
        print(f"   Price range: ${df['price'].min():.2f} to ${df['price'].max():.2f}")
        
        return df
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def test_dc_algorithm_real_data(df, symbol, thresholds=[0.0001, 0.0005, 0.001, 0.002, 0.005]):
    """Test DC algorithm with your real trading data."""
    print(f"\n=== Testing DC Algorithm on Real Data: {symbol} ===")
    
    prices = df['price'].values
    print(f"Processing {len(prices)} real price ticks...")
    
    results = {}
    
    for threshold in thresholds:
        print(f"\n--- Threshold: {threshold*100:.3f}% ---")
        
        start_time = time.time()
        
        # Apply DC Generator
        events = dg.generate(prices, d=threshold)
        
        processing_time = time.time() - start_time
        
        # Count events
        event_counts = events.value_counts()
        total_events = len(events[events != ''])
        
        # Key DC events for trading
        end_upturn_count = event_counts.get('end upturn', 0)
        end_downturn_count = event_counts.get('end downturn', 0)
        
        print(f"   Processing time: {processing_time:.4f} seconds")
        print(f"   Total DC events: {total_events}")
        print(f"   End upturn events: {end_upturn_count}")
        print(f"   End downturn events: {end_downturn_count}")
        print(f"   Events per 1000 ticks: {total_events * 1000.0 / len(prices):.2f}")
        
        results[threshold] = {
            'total_events': total_events,
            'end_upturn': end_upturn_count,
            'end_downturn': end_downturn_count,
            'processing_time': processing_time
        }
    
    return results

def simulate_trading_strategy_real_data(df, symbol, threshold=0.001):
    """Simulate trading strategy with your real data."""
    print(f"\n=== Trading Strategy Simulation: {symbol} (Threshold: {threshold*100:.3f}%) ===")
    
    prices = df['price'].values
    events = dg.generate(prices, d=threshold)
    
    # Trading simulation
    initial_capital = 100000.0
    cash = initial_capital
    position = 0.0
    trades = []
    
    for i in range(len(prices)):
        current_price = prices[i]
        event = events.iloc[i] if i < len(events) else ''
        
        # Trading logic based on DC events
        if event == 'end downturn' and position == 0:
            # Buy signal - market has turned up
            cash_to_use = cash * 0.95  # Use 95% of available cash
            position = cash_to_use / current_price
            cash -= cash_to_use
            trades.append({
                'index': i,
                'action': 'BUY',
                'price': current_price,
                'quantity': position,
                'date': df.iloc[i]['date'],
                'time': df.iloc[i]['time']
            })
            
        elif event == 'end upturn' and position > 0:
            # Sell signal - market has turned down
            cash += position * current_price
            trades.append({
                'index': i,
                'action': 'SELL',
                'price': current_price,
                'quantity': position,
                'date': df.iloc[i]['date'],
                'time': df.iloc[i]['time']
            })
            position = 0.0
    
    # Calculate performance
    final_value = cash + position * prices[-1]
    total_return = (final_value - initial_capital) / initial_capital * 100
    
    print(f"Initial Capital: ${initial_capital:,.2f}")
    print(f"Final Value: ${final_value:,.2f}")
    print(f"Total Return: {total_return:.2f}%")
    print(f"Number of Trades: {len(trades)}")
    
    if trades:
        print(f"\nFirst 3 trades:")
        for i, trade in enumerate(trades[:3]):
            print(f"  {i+1}. {trade['action']} at ${trade['price']:.2f} on {trade['date']} {trade['time']}")
        
        if len(trades) > 3:
            print(f"\nLast 3 trades:")
            for i, trade in enumerate(trades[-3:]):
                print(f"  {len(trades)-2+i}. {trade['action']} at ${trade['price']:.2f} on {trade['date']} {trade['time']}")
    
    return {
        'initial_capital': initial_capital,
        'final_value': final_value,
        'total_return': total_return,
        'num_trades': len(trades),
        'trades': trades
    }

def main():
    print("=== DC Generator Real Database Test ===")
    
    # Your database path
    db_path = "I:/zhubi/cpp_implementation/sqlite_databases/2018/2018_01.db"
    
    # Connect to database
    conn = connect_to_database(db_path)
    if not conn:
        print("Cannot proceed without database connection.")
        return
    
    try:
        # Get available symbols
        print("\n=== Available Symbols ===")
        symbols_df = get_available_symbols(conn)
        if symbols_df is not None:
            print(symbols_df.to_string(index=False))
            
            # Use the symbol with most trades
            if len(symbols_df) > 0:
                top_symbol = symbols_df.iloc[0]['symbol']
                print(f"\n=== Testing with top symbol: {top_symbol} ===")
                
                # Load trading data
                trading_data = load_trading_data(conn, top_symbol, limit=100000)
                
                if trading_data is not None:
                    # Test DC algorithm
                    dc_results = test_dc_algorithm_real_data(trading_data, top_symbol)
                    
                    # Test trading strategy
                    strategy_results = simulate_trading_strategy_real_data(trading_data, top_symbol)
                    
                    # Summary
                    print(f"\n=== SUMMARY FOR {top_symbol} ===")
                    print(f"Data points processed: {len(trading_data):,}")
                    print(f"Price range: ${trading_data['price'].min():.2f} - ${trading_data['price'].max():.2f}")
                    
                    print(f"\nDC Algorithm Results:")
                    for threshold, result in dc_results.items():
                        print(f"  {threshold*100:.3f}%: {result['total_events']} events "
                              f"({result['end_upturn']} upturn ends, {result['end_downturn']} downturn ends)")
                    
                    print(f"\nTrading Strategy Results:")
                    print(f"  Return: {strategy_results['total_return']:.2f}%")
                    print(f"  Trades: {strategy_results['num_trades']}")
                    
                else:
                    print("No trading data could be loaded.")
            else:
                print("No symbols found in database.")
        else:
            print("Could not retrieve symbols from database.")
            
    finally:
        conn.close()
        print("\n✅ Database connection closed.")

if __name__ == "__main__":
    main()
