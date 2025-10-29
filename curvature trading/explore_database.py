"""
Explore the database structure
"""
import pymysql
import pandas as pd

# Database connection parameters
DB_CONFIG = {
    'host': '192.168.50.230',
    'port': 3306,
    'user': 'root',
    'password': '352471Cf!1',
    'database': 'us_stock_sip_day_aggs'
}

def explore_database():
    """Explore the database structure"""
    try:
        # Connect to database
        print("Connecting to database...")
        conn = pymysql.connect(**DB_CONFIG)
        cursor = conn.cursor()
        
        print(f"✓ Connected to database: {DB_CONFIG['database']}")
        print("=" * 60)
        
        # Show all tables
        print("\n1. Tables in database:")
        cursor.execute("SHOW TABLES")
        tables = cursor.fetchall()
        for table in tables:
            print(f"   - {table[0]}")
        
        # For each table, show structure (just check first table)
        print("\n2. Table structure (checking first table):")
        if tables:
            table_name = tables[0][0]
            print(f"\n   Table: {table_name}")
            print("   " + "-" * 50)

            cursor.execute(f"DESCRIBE `{table_name}`")
            columns = cursor.fetchall()
            for col in columns:
                print(f"   {col[0]:20s} {col[1]:20s} {col[2]:5s} {col[3]:5s}")

            # Show sample data
            cursor.execute(f"SELECT * FROM `{table_name}` LIMIT 3")
            sample = cursor.fetchall()
            print(f"\n   Sample data (first 3 rows):")
            for row in sample:
                print(f"   {row}")

            # Show row count
            cursor.execute(f"SELECT COUNT(*) FROM `{table_name}`")
            count = cursor.fetchone()[0]
            print(f"\n   Total rows: {count:,}")
        
        # Check date range (first and last table)
        print("\n3. Date range analysis:")
        for table in [tables[0], tables[-1]]:
            table_name = table[0]
            # Try common date column names
            for date_col in ['date', 'Date', 'timestamp', 'time', 't']:
                try:
                    cursor.execute(f"SELECT MIN(`{date_col}`), MAX(`{date_col}`) FROM `{table_name}`")
                    min_date, max_date = cursor.fetchone()
                    print(f"   {table_name}.{date_col}: {min_date} to {max_date}")
                    break
                except:
                    continue

        # Check available symbols (first table only)
        print("\n4. Available symbols (sample from first table):")
        table_name = tables[0][0]
        for symbol_col in ['symbol', 'Symbol', 'ticker', 'Ticker', 'T']:
            try:
                cursor.execute(f"SELECT DISTINCT `{symbol_col}` FROM `{table_name}` LIMIT 10")
                symbols = cursor.fetchall()
                print(f"   {table_name}.{symbol_col}:")
                for sym in symbols:
                    print(f"      - {sym[0]}")
                break
            except:
                continue
        
        cursor.close()
        conn.close()
        print("\n" + "=" * 60)
        print("Database exploration complete!")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    explore_database()

