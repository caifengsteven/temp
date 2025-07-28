#!/usr/bin/env python3
"""
Debug script to check actual symbols in the US market databases
"""

import sqlite3
import os

def check_database(db_path, db_name):
    print(f"\n=== Checking {db_name} ===")
    
    if not os.path.exists(db_path):
        print(f"Database not found: {db_path}")
        return
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get first table
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' LIMIT 1")
        result = cursor.fetchone()
        
        if not result:
            print("No tables found")
            return
            
        table_name = result[0]
        print(f"Checking table: {table_name}")
        
        # Get column names
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = cursor.fetchall()
        print("Columns:")
        for col in columns:
            print(f"  {col[1]} ({col[2]})")
        
        # Get sample data
        cursor.execute(f"SELECT * FROM {table_name} LIMIT 5")
        rows = cursor.fetchall()
        print("\nSample data:")
        for i, row in enumerate(rows):
            print(f"  Row {i+1}: {row}")
        
        # Check for AAPL specifically
        print(f"\nLooking for AAPL in {table_name}:")
        
        # Try different variations
        variations = ['AAPL', 'aapl', 'Aapl']
        for var in variations:
            cursor.execute(f"SELECT COUNT(*) FROM {table_name} WHERE symbol = ?", (var,))
            count = cursor.fetchone()[0]
            if count > 0:
                print(f"  Found {count} records for '{var}'")
                
                # Get sample AAPL data
                cursor.execute(f"SELECT * FROM {table_name} WHERE symbol = ? LIMIT 3", (var,))
                aapl_rows = cursor.fetchall()
                print(f"  Sample {var} data:")
                for row in aapl_rows:
                    print(f"    {row}")
                break
        else:
            print("  AAPL not found in this table")
            
            # Show what symbols are available
            cursor.execute(f"SELECT DISTINCT symbol FROM {table_name} LIMIT 10")
            symbols = cursor.fetchall()
            print("  Available symbols (first 10):")
            for sym in symbols:
                print(f"    {sym[0]}")
        
        conn.close()
        
    except Exception as e:
        print(f"Error: {e}")

def main():
    print("=== US Market Database Symbol Debug ===")
    
    base_path = "F:\\BaiduNetdiskDownload\\US stock ane etf 1mins\\"
    
    # Check both databases
    etf_db = base_path + "US_ETF_1min.db"
    stock_db = base_path + "US_stock_1min.db"
    
    check_database(etf_db, "US_ETF_1min.db")
    check_database(stock_db, "US_stock_1min.db")

if __name__ == "__main__":
    main()
