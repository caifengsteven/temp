#!/usr/bin/env python3
"""
Test DC Generator algorithm with real high-frequency trading data.
"""

import sqlite3
import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime
import time

# Add the dcgenerator module to path
sys.path.append('.')
import dcgenerator as dg

def load_trades_data(db_path, symbol, limit=10000):
    """Load trades data from SQLite database."""
    print(f"Loading trades data from: {db_path}")
    print(f"Symbol: {symbol}, Limit: {limit}")
    
    try:
        conn = sqlite3.connect(db_path)
        
        # Get available symbols first
        symbols_df = pd.read_sql_query("SELECT DISTINCT symbol FROM trade LIMIT 20", conn)
        print(f"Available symbols: {symbols_df['symbol'].tolist()}")
        
        # Load trades data
        query = """
        SELECT date, time, price, volume, buysell 
        FROM trade 
        WHERE symbol = ? 
        ORDER BY date, time 
        LIMIT ?
        """
        
        df = pd.read_sql_query(query, conn, params=(symbol, limit))
        conn.close()
        
        if df.empty:
            print(f"No data found for symbol: {symbol}")
            return None
        
        print(f"Loaded {len(df)} trades")
        print(f"Date range: {df['date'].min()} to {df['date'].max()}")
        print(f"Price range: {df['price'].min():.2f} to {df['price'].max():.2f}")
        
        return df
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def test_dc_algorithm(df, thresholds=[0.0001, 0.0005, 0.001, 0.002, 0.005]):
    """Test DC algorithm with different thresholds."""
    print(f"\n=== Testing DC Algorithm ===")
    
    # Extract price series
    prices = df['price'].values
    print(f"Price series length: {len(prices)}")
    print(f"First 10 prices: {prices[:10]}")
    
    results = {}
    
    for threshold in thresholds:
        print(f"\n--- Testing threshold: {threshold*100:.3f}% ---")
        
        start_time = time.time()
        
        # Use the DC generator
        events = dg.generate(prices, d=threshold)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Count events
        event_counts = events.value_counts()
        total_events = len(events[events != ''])
        
        print(f"Processing time: {processing_time:.4f} seconds")
        print(f"Total DC events: {total_events}")
        print("Event breakdown:")
        for event_type, count in event_counts.items():
            if event_type != '':
                print(f"  {event_type}: {count}")
        
        # Calculate event frequency
        events_per_trade = total_events / len(prices) if len(prices) > 0 else 0
        print(f"Events per trade: {events_per_trade:.4f}")
        
        results[threshold] = {
            'total_events': total_events,
            'events_per_trade': events_per_trade,
            'processing_time': processing_time,
            'event_counts': event_counts
        }
    
    return results

def simulate_trading_strategy(df, threshold=0.001):
    """Simulate a simple trading strategy based on DC events."""
    print(f"\n=== Simulating Trading Strategy (Threshold: {threshold*100:.3f}%) ===")
    
    prices = df['price'].values
    events = dg.generate(prices, d=threshold)
    
    # Trading simulation
    initial_capital = 100000.0
    cash = initial_capital
    position = 0.0
    position_size_pct = 0.95
    
    trades = []
    portfolio_values = []
    
    for i in range(len(prices)):
        current_price = prices[i]
        event = events.iloc[i] if i < len(events) else ''
        
        # Trading logic
        if event == 'end downturn' and position == 0:
            # Buy signal
            cash_to_use = cash * position_size_pct
            position = cash_to_use / current_price
            cash -= cash_to_use
            trades.append({
                'index': i,
                'action': 'BUY',
                'price': current_price,
                'quantity': position,
                'cash': cash
            })
            print(f"BUY at {current_price:.2f}, Position: {position:.4f}")
            
        elif event == 'end upturn' and position > 0:
            # Sell signal
            cash += position * current_price
            trades.append({
                'index': i,
                'action': 'SELL',
                'price': current_price,
                'quantity': position,
                'cash': cash
            })
            print(f"SELL at {current_price:.2f}, Cash: {cash:.2f}")
            position = 0.0
        
        # Track portfolio value
        portfolio_value = cash + position * current_price
        portfolio_values.append(portfolio_value)
    
    # Calculate performance
    final_value = portfolio_values[-1] if portfolio_values else initial_capital
    total_return = (final_value - initial_capital) / initial_capital * 100
    
    print(f"\nTrading Results:")
    print(f"Initial Capital: ${initial_capital:,.2f}")
    print(f"Final Value: ${final_value:,.2f}")
    print(f"Total Return: {total_return:.2f}%")
    print(f"Number of Trades: {len(trades)}")
    
    if len(trades) > 0:
        print(f"First trade: {trades[0]}")
        print(f"Last trade: {trades[-1]}")
    
    return {
        'initial_capital': initial_capital,
        'final_value': final_value,
        'total_return': total_return,
        'num_trades': len(trades),
        'trades': trades,
        'portfolio_values': portfolio_values
    }

def main():
    # Database configuration
    db_path = "I:/zhubi/cpp_implementation/sqlite_databases/2018/2018_01.db"
    
    # Test with different symbols if needed
    test_symbols = ["BTCUSD", "ETHUSD", "XRPUSD", "LTCUSD"]
    
    print("=== DC Generator High-Frequency Trading Test ===")
    print(f"Database: {db_path}")
    
    # Check if database exists
    if not os.path.exists(db_path):
        print(f"Database not found: {db_path}")
        return
    
    # Try to find a symbol with data
    symbol_found = None
    df = None
    
    for symbol in test_symbols:
        print(f"\nTrying symbol: {symbol}")
        df = load_trades_data(db_path, symbol, limit=50000)
        if df is not None and len(df) > 100:
            symbol_found = symbol
            break
    
    if df is None or len(df) < 100:
        print("No suitable data found. Let's check what symbols are available...")
        try:
            conn = sqlite3.connect(db_path)
            symbols_df = pd.read_sql_query(
                "SELECT symbol, COUNT(*) as count FROM trade GROUP BY symbol ORDER BY count DESC LIMIT 10", 
                conn
            )
            print("Top symbols by trade count:")
            print(symbols_df)
            conn.close()
            
            if len(symbols_df) > 0:
                # Use the symbol with most trades
                symbol_found = symbols_df.iloc[0]['symbol']
                print(f"\nUsing symbol with most data: {symbol_found}")
                df = load_trades_data(db_path, symbol_found, limit=50000)
        except Exception as e:
            print(f"Error checking symbols: {e}")
            return
    
    if df is None or len(df) < 100:
        print("Still no suitable data found!")
        return
    
    print(f"\nUsing symbol: {symbol_found}")
    print(f"Data sample:")
    print(df.head())
    
    # Test DC algorithm with different thresholds
    dc_results = test_dc_algorithm(df)
    
    # Test trading strategy
    strategy_results = simulate_trading_strategy(df, threshold=0.001)
    
    # Summary
    print(f"\n=== SUMMARY ===")
    print(f"Symbol: {symbol_found}")
    print(f"Data points: {len(df)}")
    print(f"Price range: {df['price'].min():.2f} - {df['price'].max():.2f}")
    
    print(f"\nDC Algorithm Results:")
    for threshold, result in dc_results.items():
        print(f"  {threshold*100:.3f}%: {result['total_events']} events, "
              f"{result['events_per_trade']:.4f} events/trade")
    
    print(f"\nTrading Strategy (0.1% threshold):")
    print(f"  Return: {strategy_results['total_return']:.2f}%")
    print(f"  Trades: {strategy_results['num_trades']}")

if __name__ == "__main__":
    main()
