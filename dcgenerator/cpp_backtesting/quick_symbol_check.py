#!/usr/bin/env python3
"""
Quick check of symbols in US market databases
"""

import sqlite3
import os

def quick_check():
    print("=== Quick Symbol Check ===")
    
    # Check stock database
    stock_db = "F:\\BaiduNetdiskDownload\\US stock ane etf 1mins\\US_stock_1min.db"
    
    if not os.path.exists(stock_db):
        print(f"Database not found: {stock_db}")
        return
    
    try:
        print(f"Connecting to: {stock_db}")
        conn = sqlite3.connect(stock_db)
        cursor = conn.cursor()
        
        # Get first table
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' LIMIT 1")
        result = cursor.fetchone()
        
        if not result:
            print("No tables found")
            return
            
        table_name = result[0]
        print(f"Using table: {table_name}")
        
        # Get sample data (quote table name since it starts with number)
        cursor.execute(f'SELECT * FROM "{table_name}" LIMIT 3')
        rows = cursor.fetchall()

        print("Sample data:")
        for i, row in enumerate(rows):
            print(f"  Row {i+1}: {row}")

        # Check for AAPL
        cursor.execute(f'SELECT COUNT(*) FROM "{table_name}" WHERE symbol = ?', ('AAPL',))
        count = cursor.fetchone()[0]
        print(f"AAPL records: {count}")

        if count == 0:
            # Try lowercase
            cursor.execute(f'SELECT COUNT(*) FROM "{table_name}" WHERE symbol = ?', ('aapl',))
            count = cursor.fetchone()[0]
            print(f"aapl records: {count}")

        if count > 0:
            # Get sample AAPL data
            cursor.execute(f'SELECT * FROM "{table_name}" WHERE symbol = ? LIMIT 3', ('AAPL' if count > 0 else 'aapl',))
            aapl_data = cursor.fetchall()
            print("Sample AAPL data:")
            for row in aapl_data:
                print(f"  {row}")

        # Get some symbols
        cursor.execute(f'SELECT DISTINCT symbol FROM "{table_name}" LIMIT 10')
        symbols = cursor.fetchall()
        print("Available symbols:")
        for sym in symbols:
            print(f"  {sym[0]}")
        
        conn.close()
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    quick_check()
