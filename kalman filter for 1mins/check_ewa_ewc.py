import pymysql
import pandas as pd

def connect_to_mysql():
    """Connect to MySQL database"""
    try:
        conn = pymysql.connect(
            host='192.168.50.230',
            port=3306,
            user='root',
            password='352471Cf!1',
            database='us_stock_sip_min_aggs'
        )
        print("Successfully connected to MySQL!")
        return conn
    except Exception as e:
        print(f"Error connecting to MySQL: {e}")
        return None

def check_ticker_data(conn, ticker):
    """Check if ticker exists and count records"""
    cursor = conn.cursor()
    
    # Check a few recent tables
    tables_to_check = ['202501', '202502', '202503', '202504', '202505']
    
    total_records = 0
    found_in_tables = []
    
    for table in tables_to_check:
        try:
            query = f"SELECT COUNT(*) FROM `{table}` WHERE ticker = %s"
            cursor.execute(query, (ticker,))
            count = cursor.fetchone()[0]
            if count > 0:
                total_records += count
                found_in_tables.append(f"{table}: {count} records")
        except Exception as e:
            pass
    
    cursor.close()
    
    return total_records, found_in_tables

if __name__ == "__main__":
    conn = connect_to_mysql()
    
    if conn:
        print("\nChecking EWA (iShares MSCI Australia ETF)...")
        ewa_count, ewa_tables = check_ticker_data(conn, 'EWA')
        print(f"Total EWA records in recent months: {ewa_count}")
        if ewa_tables:
            for table_info in ewa_tables:
                print(f"  {table_info}")
        else:
            print("  No EWA data found in recent tables")
        
        print("\nChecking EWC (iShares MSCI Canada ETF)...")
        ewc_count, ewc_tables = check_ticker_data(conn, 'EWC')
        print(f"Total EWC records in recent months: {ewc_count}")
        if ewc_tables:
            for table_info in ewc_tables:
                print(f"  {table_info}")
        else:
            print("  No EWC data found in recent tables")
        
        conn.close()
        print("\nDatabase connection closed.")

