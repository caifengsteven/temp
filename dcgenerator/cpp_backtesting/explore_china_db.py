#!/usr/bin/env python3
"""
Explore China market database structure
"""

import sqlite3
import os

def explore_china_database():
    print("=== China Market Database Explorer ===")
    
    db_path = "I:\\zhubi\\cpp_implementation\\sqlite_databases\\2018\\2018_01.db"
    print(f"Database: {db_path}")
    
    if not os.path.exists(db_path):
        print(f"Database not found: {db_path}")
        return
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get all tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()
        
        print(f"\nTables found ({len(tables)}):")
        for table in tables:
            print(f"  - {table[0]}")
        
        # Explore first few tables
        for i, table in enumerate(tables[:3]):
            table_name = table[0]
            print(f"\n=== Table: {table_name} ===")
            
            # Get column info
            cursor.execute(f"PRAGMA table_info({table_name})")
            columns = cursor.fetchall()
            print("Columns:")
            for col in columns:
                print(f"  {col[1]} ({col[2]})")
            
            # Get sample data
            cursor.execute(f"SELECT * FROM {table_name} LIMIT 3")
            rows = cursor.fetchall()
            print("Sample data:")
            for j, row in enumerate(rows):
                print(f"  Row {j+1}: {row}")
            
            # Check for sh600000
            try:
                cursor.execute(f"SELECT COUNT(*) FROM {table_name} WHERE symbol = 'sh600000'")
                count = cursor.fetchone()[0]
                if count > 0:
                    print(f"Found {count} records for sh600000 in {table_name}")
                    
                    # Get sample sh600000 data
                    cursor.execute(f"SELECT * FROM {table_name} WHERE symbol = 'sh600000' LIMIT 5")
                    sh600000_rows = cursor.fetchall()
                    print("Sample sh600000 data:")
                    for row in sh600000_rows:
                        print(f"    {row}")
            except:
                # Try other possible column names
                for col_name in ['stock_code', 'code', 'ticker']:
                    try:
                        cursor.execute(f"SELECT COUNT(*) FROM {table_name} WHERE {col_name} = 'sh600000'")
                        count = cursor.fetchone()[0]
                        if count > 0:
                            print(f"Found {count} records for sh600000 in {table_name} (column: {col_name})")
                            break
                    except:
                        continue
        
        conn.close()
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    explore_china_database()
