import pymysql

# Database connection parameters
DB_CONFIG = {
    'host': '192.168.50.230',
    'port': 3306,
    'user': 'root',
    'password': '352471Cf!1',
    'database': 'us_stock_sip_min_aggs'
}

try:
    conn = pymysql.connect(**DB_CONFIG)
    cursor = conn.cursor()
    
    # Show all tables
    cursor.execute("SHOW TABLES")
    tables = cursor.fetchall()
    
    print("Available tables in database:")
    for table in tables:
        print(f"  - {table[0]}")
    
    # If there are tables, show structure of first few
    if tables:
        print("\nChecking first table structure...")
        first_table = tables[0][0]
        cursor.execute(f"DESCRIBE `{first_table}`")
        columns = cursor.fetchall()
        print(f"\nColumns in {first_table}:")
        for col in columns:
            print(f"  - {col[0]} ({col[1]})")
        
        # Show sample data
        cursor.execute(f"SELECT * FROM `{first_table}` LIMIT 5")
        sample = cursor.fetchall()
        print(f"\nSample data from {first_table}:")
        for row in sample:
            print(f"  {row}")
    
    cursor.close()
    conn.close()
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

