#!/usr/bin/env python3
"""
Explore the actual symbols and data in the database without assumptions.
"""

import sqlite3
import pandas as pd
import os

def explore_database(db_path):
    """Explore what's actually in the database."""
    print(f"Exploring database: {db_path}")
    
    if not os.path.exists(db_path):
        print(f"Database not found: {db_path}")
        return
    
    try:
        conn = sqlite3.connect(db_path)
        
        # Get table structure
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(trade);")
        columns = cursor.fetchall()
        print("\nTable structure:")
        for col in columns:
            print(f"  {col[1]} ({col[2]})")
        
        # Get total number of records
        cursor.execute("SELECT COUNT(*) FROM trade;")
        total_records = cursor.fetchone()[0]
        print(f"\nTotal records: {total_records:,}")
        
        # Get all unique symbols and their counts
        print("\nAll symbols in database:")
        symbols_df = pd.read_sql_query("""
            SELECT symbol, COUNT(*) as trade_count,
                   MIN(price) as min_price, MAX(price) as max_price,
                   MIN(date) as start_date, MAX(date) as end_date
            FROM trade 
            GROUP BY symbol 
            ORDER BY trade_count DESC
        """, conn)
        
        print(symbols_df.to_string(index=False))
        
        # Show sample data for the top symbol
        if len(symbols_df) > 0:
            top_symbol = symbols_df.iloc[0]['symbol']
            print(f"\nSample data for {top_symbol}:")
            
            sample_df = pd.read_sql_query("""
                SELECT date, time, price, volume, buysell 
                FROM trade 
                WHERE symbol = ? 
                ORDER BY date, time 
                LIMIT 10
            """, conn, params=(top_symbol,))
            
            print(sample_df.to_string(index=False))
        
        conn.close()
        
    except Exception as e:
        print(f"Error: {e}")

def main():
    # Check a few database files
    db_paths = [
        "I:/zhubi/cpp_implementation/sqlite_databases/2018/2018_01.db",
        "I:/zhubi/cpp_implementation/sqlite_databases/2018/2018_02.db"
    ]
    
    for db_path in db_paths:
        if os.path.exists(db_path):
            explore_database(db_path)
            break
    else:
        print("No database files found at the expected locations.")
        
        # Try to find any .db files
        base_path = "I:/zhubi/cpp_implementation/sqlite_databases/"
        if os.path.exists(base_path):
            print(f"\nLooking for .db files in {base_path}...")
            for root, dirs, files in os.walk(base_path):
                for file in files:
                    if file.endswith('.db'):
                        db_file = os.path.join(root, file)
                        print(f"Found: {db_file}")
                        explore_database(db_file)
                        return

if __name__ == "__main__":
    main()
