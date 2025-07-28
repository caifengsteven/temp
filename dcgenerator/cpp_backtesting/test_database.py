#!/usr/bin/env python3
"""
Test script to examine the SQLite database structure and sample data
for the DC Generator backtesting system.
"""

import sqlite3
import os
import sys
from datetime import datetime

def examine_database(db_path, db_type):
    """Examine the structure and content of a SQLite database."""
    print(f"\n=== Examining {db_type} Database: {db_path} ===")
    
    if not os.path.exists(db_path):
        print(f"Database file not found: {db_path}")
        return False
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get all tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        
        print(f"Tables found: {[table[0] for table in tables]}")
        
        for table in tables:
            table_name = table[0]
            print(f"\n--- Table: {table_name} ---")
            
            # Get table schema
            cursor.execute(f"PRAGMA table_info({table_name});")
            columns = cursor.fetchall()
            print("Columns:")
            for col in columns:
                print(f"  {col[1]} ({col[2]})")
            
            # Get row count
            cursor.execute(f"SELECT COUNT(*) FROM {table_name};")
            row_count = cursor.fetchone()[0]
            print(f"Row count: {row_count}")
            
            # Get sample data
            if row_count > 0:
                cursor.execute(f"SELECT * FROM {table_name} LIMIT 5;")
                sample_rows = cursor.fetchall()
                print("Sample data (first 5 rows):")
                for i, row in enumerate(sample_rows):
                    print(f"  Row {i+1}: {row}")
                
                # If there's a timestamp column, show time range
                timestamp_cols = ['timestamp', 'time', 'ts']
                for col_info in columns:
                    col_name = col_info[1].lower()
                    if any(ts_col in col_name for ts_col in timestamp_cols):
                        cursor.execute(f"SELECT MIN({col_info[1]}), MAX({col_info[1]}) FROM {table_name};")
                        min_ts, max_ts = cursor.fetchone()
                        if min_ts and max_ts:
                            # Try to convert timestamps (assuming nanoseconds or milliseconds)
                            try:
                                # Try nanoseconds first
                                min_dt = datetime.fromtimestamp(min_ts / 1e9)
                                max_dt = datetime.fromtimestamp(max_ts / 1e9)
                                print(f"Time range (ns): {min_dt} to {max_dt}")
                            except (ValueError, OSError):
                                try:
                                    # Try milliseconds
                                    min_dt = datetime.fromtimestamp(min_ts / 1e3)
                                    max_dt = datetime.fromtimestamp(max_ts / 1e3)
                                    print(f"Time range (ms): {min_dt} to {max_dt}")
                                except (ValueError, OSError):
                                    try:
                                        # Try seconds
                                        min_dt = datetime.fromtimestamp(min_ts)
                                        max_dt = datetime.fromtimestamp(max_ts)
                                        print(f"Time range (s): {min_dt} to {max_dt}")
                                    except (ValueError, OSError):
                                        print(f"Timestamp range: {min_ts} to {max_ts}")
                        break
        
        conn.close()
        return True
        
    except sqlite3.Error as e:
        print(f"SQLite error: {e}")
        return False
    except Exception as e:
        print(f"Error: {e}")
        return False

def find_database_files(base_path):
    """Find SQLite database files in the given path."""
    db_files = []
    
    if os.path.isfile(base_path):
        if base_path.endswith('.db') or base_path.endswith('.sqlite'):
            db_files.append(base_path)
    elif os.path.isdir(base_path):
        for root, dirs, files in os.walk(base_path):
            for file in files:
                if file.endswith('.db') or file.endswith('.sqlite'):
                    db_files.append(os.path.join(root, file))
    
    return db_files

def main():
    # Default paths from the main application
    orderbook_path = "J:/fenbi/cpp_implementation/sqlite_databases"
    trades_path = "I:/zhubi/cpp_implementation/sqlite_databases"
    
    # Allow command line override
    if len(sys.argv) > 1:
        orderbook_path = sys.argv[1]
    if len(sys.argv) > 2:
        trades_path = sys.argv[2]
    
    print("DC Generator Database Structure Analyzer")
    print("=" * 50)
    
    # Find and examine orderbook databases
    print(f"\nSearching for orderbook databases in: {orderbook_path}")
    orderbook_dbs = find_database_files(orderbook_path)
    
    if orderbook_dbs:
        print(f"Found {len(orderbook_dbs)} orderbook database(s):")
        for db in orderbook_dbs:
            examine_database(db, "Orderbook")
    else:
        print("No orderbook databases found!")
    
    # Find and examine trades databases
    print(f"\nSearching for trades databases in: {trades_path}")
    trades_dbs = find_database_files(trades_path)
    
    if trades_dbs:
        print(f"Found {len(trades_dbs)} trades database(s):")
        for db in trades_dbs:
            examine_database(db, "Trades")
    else:
        print("No trades databases found!")
    
    # Generate sample command
    if orderbook_dbs and trades_dbs:
        print("\n" + "=" * 50)
        print("SAMPLE COMMAND TO RUN BACKTEST:")
        print("=" * 50)
        
        # Use the first database found for each type
        sample_orderbook = orderbook_dbs[0]
        sample_trades = trades_dbs[0]
        
        print(f'./bin/DCGeneratorBacktesting \\')
        print(f'  --orderbook-db "{sample_orderbook}" \\')
        print(f'  --trades-db "{sample_trades}" \\')
        print(f'  --symbol BTCUSDT \\')
        print(f'  --dc-threshold 0.001 \\')
        print(f'  --capital 100000 \\')
        print(f'  --verbose')
        
        print("\nNote: You may need to adjust the --symbol parameter based on")
        print("the actual symbols available in your databases.")

if __name__ == "__main__":
    main()
