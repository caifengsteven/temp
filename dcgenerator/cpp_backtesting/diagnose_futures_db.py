#!/usr/bin/env python3
"""
Diagnose US futures database structure
"""

import sqlite3
import os

def diagnose_futures_database():
    print("=== US Futures Database Diagnostic ===")
    
    db_path = "F:\\database\\us futures 1mins\\us_fut_1min.db"
    print(f"Database: {db_path}")
    
    if not os.path.exists(db_path):
        print(f"Database not found: {db_path}")
        print("Please check the path and make sure the database exists.")
        return
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get all tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
        tables = cursor.fetchall()
        
        print(f"\nALL TABLES FOUND ({len(tables)}):")
        for i, table in enumerate(tables):
            print(f"  {i+1:2d}. {table[0]}")
        
        # Analyze first few tables in detail
        for i, table in enumerate(tables[:3]):
            table_name = table[0]
            print(f"\n{'='*60}")
            print(f"=== DETAILED ANALYSIS: {table_name} ===")
            print(f"{'='*60}")
            
            # Get column info
            cursor.execute(f"PRAGMA table_info(\"{table_name}\")")
            columns = cursor.fetchall()
            print(f"\nCOLUMNS ({len(columns)}):")
            column_names = []
            for col in columns:
                column_names.append(col[1])
                print(f"  {col[1]} ({col[2]})")
            
            # Get total row count
            cursor.execute(f"SELECT COUNT(*) FROM \"{table_name}\"")
            total_rows = cursor.fetchone()[0]
            print(f"\nTOTAL ROWS: {total_rows:,}")
            
            # Get sample data
            cursor.execute(f"SELECT * FROM \"{table_name}\" LIMIT 5")
            rows = cursor.fetchall()
            print(f"\nSAMPLE DATA:")
            for j, row in enumerate(rows):
                print(f"\nRow {j+1}:")
                for k, value in enumerate(row):
                    if k < len(column_names):
                        print(f"  {column_names[k]}: {value}")
            
            # Try to find symbol column
            symbol_columns = ['symbol', 'Symbol', 'SYMBOL', 'ticker', 'Ticker', 'contract', 'Contract']
            symbol_column_found = None
            
            for col_name in symbol_columns:
                if col_name in column_names:
                    symbol_column_found = col_name
                    break
            
            if symbol_column_found:
                print(f"\n*** SYMBOL COLUMN FOUND: {symbol_column_found} ***")
                cursor.execute(f"SELECT DISTINCT {symbol_column_found} FROM \"{table_name}\" ORDER BY {symbol_column_found} LIMIT 20")
                symbols = cursor.fetchall()
                print(f"UNIQUE SYMBOLS ({len(symbols)} shown, max 20):")
                for symbol in symbols:
                    print(f"  {symbol[0]}")
                
                # Test specific symbols
                test_symbols = ['ES', 'NQ', 'CL', 'GC']
                print(f"\nTESTING COMMON FUTURES SYMBOLS:")
                for test_symbol in test_symbols:
                    cursor.execute(f"SELECT COUNT(*) FROM \"{table_name}\" WHERE {symbol_column_found} = ?", (test_symbol,))
                    count = cursor.fetchone()[0]
                    print(f"  {test_symbol}: {count} records")
                    
                    if count > 0:
                        # Get sample data for this symbol
                        cursor.execute(f"SELECT * FROM \"{table_name}\" WHERE {symbol_column_found} = ? LIMIT 3", (test_symbol,))
                        symbol_rows = cursor.fetchall()
                        print(f"    Sample {test_symbol} data:")
                        for row in symbol_rows:
                            print(f"      {row}")
            else:
                print(f"\n*** NO SYMBOL COLUMN FOUND ***")
                print("Available columns:", column_names)
            
            # Try to find price columns
            price_columns = ['close', 'Close', 'CLOSE', 'price', 'Price', 'PRICE', 'last', 'Last']
            price_column_found = None
            
            for col_name in price_columns:
                if col_name in column_names:
                    price_column_found = col_name
                    break
            
            if price_column_found:
                print(f"\n*** PRICE COLUMN FOUND: {price_column_found} ***")
                cursor.execute(f"SELECT MIN({price_column_found}), MAX({price_column_found}), AVG({price_column_found}) FROM \"{table_name}\" WHERE {price_column_found} > 0")
                price_stats = cursor.fetchone()
                if price_stats[0] is not None:
                    print(f"Price range: ${price_stats[0]:.2f} to ${price_stats[1]:.2f}")
                    print(f"Average price: ${price_stats[2]:.2f}")
            else:
                print(f"\n*** NO PRICE COLUMN FOUND ***")
                print("Available columns:", column_names)
        
        conn.close()
        
        # Summary and recommendations
        print(f"\n{'='*60}")
        print("SUMMARY AND RECOMMENDATIONS")
        print(f"{'='*60}")
        
        if len(tables) == 0:
            print("❌ No tables found in database")
        else:
            print(f"✅ Found {len(tables)} tables")
            
        print("\nTo fix the C++ program:")
        print("1. Use the correct table names shown above")
        print("2. Use the correct symbol column name")
        print("3. Use the correct price column name")
        print("4. Check symbol format (might need exact match)")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    diagnose_futures_database()
