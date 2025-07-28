#!/usr/bin/env python3
"""
Simple check if futures database exists and is accessible
"""

import os
import sqlite3

def check_database():
    db_path = "F:\\database\\us futures 1mins\\us_fut_1min.db"
    
    print("=== US Futures Database Check ===")
    print(f"Database path: {db_path}")
    
    # Check if file exists
    if os.path.exists(db_path):
        print("✅ Database file exists")
        
        # Check file size
        size = os.path.getsize(db_path)
        print(f"✅ Database size: {size:,} bytes ({size/1024/1024:.1f} MB)")
        
        # Try to connect
        try:
            conn = sqlite3.connect(db_path)
            print("✅ Database connection successful")
            
            cursor = conn.cursor()
            
            # Get table count
            cursor.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table'")
            table_count = cursor.fetchone()[0]
            print(f"✅ Found {table_count} tables")
            
            # Get table names
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = cursor.fetchall()
            print("Tables:")
            for table in tables:
                print(f"  - {table[0]}")
            
            conn.close()
            print("✅ Database check completed successfully")
            
        except Exception as e:
            print(f"❌ Database connection error: {e}")
    else:
        print("❌ Database file does not exist")
        print("Please check:")
        print("1. Path is correct")
        print("2. Drive F: is accessible")
        print("3. File permissions")

if __name__ == "__main__":
    check_database()
