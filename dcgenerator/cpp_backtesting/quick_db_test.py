#!/usr/bin/env python3
"""
Quick test to examine one database file and understand the data structure.
"""

import sqlite3
import os
from datetime import datetime

def test_single_database():
    # Test one database file
    db_path = "I:/zhubi/cpp_implementation/sqlite_databases/2018/2018_01.db"
    
    if not os.path.exists(db_path):
        print(f"Database not found: {db_path}")
        return
    
    print(f"Examining: {db_path}")
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get table info
        cursor.execute("PRAGMA table_info(trade);")
        columns = cursor.fetchall()
        print("Columns in 'trade' table:")
        for col in columns:
            print(f"  {col[1]} ({col[2]})")
        
        # Get row count
        cursor.execute("SELECT COUNT(*) FROM trade;")
        row_count = cursor.fetchone()[0]
        print(f"\nTotal rows: {row_count}")
        
        # Get sample data
        cursor.execute("SELECT * FROM trade LIMIT 3;")
        sample_rows = cursor.fetchall()
        print("\nSample data:")
        for i, row in enumerate(sample_rows):
            print(f"  Row {i+1}: {row}")
        
        # Get unique symbols
        cursor.execute("SELECT DISTINCT symbol FROM trade LIMIT 10;")
        symbols = cursor.fetchall()
        print(f"\nAvailable symbols (first 10): {[s[0] for s in symbols]}")
        
        # Get time range for a specific symbol
        if symbols:
            symbol = symbols[0][0]
            cursor.execute("SELECT MIN(date || ' ' || time), MAX(date || ' ' || time) FROM trade WHERE symbol = ?;", (symbol,))
            min_time, max_time = cursor.fetchone()
            print(f"\nTime range for {symbol}: {min_time} to {max_time}")
            
            # Get sample data for this symbol
            cursor.execute("SELECT date, time, price, buysell, volume FROM trade WHERE symbol = ? LIMIT 5;", (symbol,))
            symbol_data = cursor.fetchall()
            print(f"\nSample {symbol} data:")
            for row in symbol_data:
                print(f"  {row}")
        
        conn.close()
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_single_database()
