#!/usr/bin/env python3
"""
Simple test to verify CL futures data exists and can be loaded
"""

import sqlite3
import os

def test_cl_data():
    print("=== Testing CL (Crude Oil) Futures Data ===")
    
    db_path = "F:\\database\\us futures 1mins\\us_fut_1min.db"
    print(f"Database: {db_path}")
    
    if not os.path.exists(db_path):
        print(f"❌ Database not found: {db_path}")
        return
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        print("✅ Database connection successful")
        
        # Check CL record count
        cursor.execute("SELECT COUNT(*) FROM futures_data WHERE symbol = 'CL'")
        cl_count = cursor.fetchone()[0]
        print(f"✅ Total CL records: {cl_count:,}")
        
        if cl_count == 0:
            print("❌ No CL data found!")
            return
        
        # Get sample CL data
        cursor.execute("SELECT datetime, close FROM futures_data WHERE symbol = 'CL' ORDER BY datetime LIMIT 10")
        sample_data = cursor.fetchall()
        
        print("\nSample CL data:")
        print("DateTime\t\t\tClose Price")
        print("-" * 40)
        for row in sample_data:
            print(f"{row[0]}\t${row[1]:.2f}")
        
        # Load price data for analysis
        cursor.execute("SELECT close FROM futures_data WHERE symbol = 'CL' ORDER BY datetime LIMIT 10000")
        price_data = cursor.fetchall()
        
        prices = [row[0] for row in price_data if row[0] > 0]
        
        print(f"\n✅ Loaded {len(prices)} CL price points")
        
        if len(prices) > 0:
            min_price = min(prices)
            max_price = max(prices)
            price_range_pct = ((max_price - min_price) / min_price) * 100.0
            
            print(f"Price range: ${min_price:.2f} to ${max_price:.2f}")
            print(f"Price range %: {price_range_pct:.1f}%")
            print(f"First price: ${prices[0]:.2f}")
            print(f"Last price: ${prices[-1]:.2f}")
            
            print("\n🎯 CONCLUSION: CL futures data is available and ready for DC testing!")
            
        conn.close()
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_cl_data()
