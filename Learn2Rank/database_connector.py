import pandas as pd
import mysql.connector
import numpy as np
from datetime import datetime, timedelta
import os
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class DatabaseConnector:
    def __init__(self, host='localhost', user='root', password='352471Cf', database='yuqerdata'):
        """
        Initialize database connection
        """
        self.host = host
        self.user = user
        self.password = password
        self.database = database
        self.connection = None
        
    def connect(self):
        """
        Establish database connection
        """
        try:
            self.connection = mysql.connector.connect(
                host=self.host,
                user=self.user,
                password=self.password,
                database=self.database
            )
            print(f"Successfully connected to database: {self.database}")
            return True
        except mysql.connector.Error as err:
            print(f"Error connecting to database: {err}")
            return False
    
    def disconnect(self):
        """
        Close database connection
        """
        if self.connection:
            self.connection.close()
            print("Database connection closed")
    
    def get_table_info(self, table_name='yq_mktstockfactorsonedayget'):
        """
        Get table structure and sample data
        """
        if not self.connection:
            print("No database connection")
            return None
            
        cursor = self.connection.cursor()
        
        # Get table structure
        cursor.execute(f"DESCRIBE {table_name}")
        columns = cursor.fetchall()
        
        print(f"\nTable structure for {table_name}:")
        print("-" * 80)
        print(f"{'Column':<30} {'Type':<20} {'Null':<10} {'Key':<10} {'Default':<10}")
        print("-" * 80)
        for col in columns:
            print(f"{col[0]:<30} {col[1]:<20} {col[2]:<10} {col[3]:<10} {str(col[4]) if col[4] is not None else 'None':<10}")
        
        # Get sample data
        cursor.execute(f"SELECT * FROM {table_name} LIMIT 5")
        sample_data = cursor.fetchall()
        
        print(f"\nSample data from {table_name}:")
        print("-" * 80)
        column_names = [desc[0] for desc in cursor.description]
        print("Columns:", column_names)
        for row in sample_data:
            print(row)
        
        # Get data count
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        count = cursor.fetchone()[0]
        print(f"\nTotal records: {count}")
        
        # Get date range
        cursor.execute(f"SELECT MIN(tradeDate), MAX(tradeDate) FROM {table_name}")
        date_range = cursor.fetchone()
        print(f"Date range: {date_range[0]} to {date_range[1]}")
        
        cursor.close()
        return column_names
    
    def get_stock_data(self, start_date=None, end_date=None, limit=None):
        """
        Extract stock factor data from database
        """
        if not self.connection:
            print("No database connection")
            return None
            
        cursor = self.connection.cursor()
        
        # Build query
        query = "SELECT * FROM yq_mktstockfactorsonedayget"
        conditions = []
        
        if start_date:
            conditions.append(f"tradeDate >= '{start_date}'")
        if end_date:
            conditions.append(f"tradeDate <= '{end_date}'")
            
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
            
        query += " ORDER BY tradeDate, ticker"
        
        if limit:
            query += f" LIMIT {limit}"
        
        print(f"Executing query: {query}")
        
        # Execute query and get data
        cursor.execute(query)
        data = cursor.fetchall()
        
        # Get column names
        column_names = [desc[0] for desc in cursor.description]
        
        # Create DataFrame
        df = pd.DataFrame(data, columns=column_names)
        
        cursor.close()
        return df

def main():
    """
    Main function to test database connection and examine data
    """
    # Initialize database connector
    db = DatabaseConnector()
    
    # Connect to database
    if not db.connect():
        return
    
    # Get table information
    columns = db.get_table_info()
    
    # Get sample data
    print("\nFetching sample data...")
    sample_df = db.get_stock_data(limit=1000)
    
    if sample_df is not None:
        print(f"\nSample DataFrame shape: {sample_df.shape}")
        print("\nDataFrame info:")
        print(sample_df.info())
        print("\nFirst few rows:")
        print(sample_df.head())
        
        # Check for missing values
        print("\nMissing values:")
        print(sample_df.isnull().sum())
        
        # Save sample to CSV for inspection
        sample_df.to_csv('sample_database_data.csv', index=False)
        print("\nSample data saved to 'sample_database_data.csv'")
    
    # Disconnect
    db.disconnect()

if __name__ == "__main__":
    main()
