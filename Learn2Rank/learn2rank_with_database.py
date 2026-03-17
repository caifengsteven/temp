import pandas as pd
import numpy as np
import torch
import mysql.connector
from datetime import datetime, timedelta
import os
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Import the existing algorithms
from algs.ranknet import RankNet, pairwise_data
from algs.listnet import ListNet, ListMLE_loss, ndcg
import xgboost as xgb

class Learn2RankPipeline:
    def __init__(self, db_host='localhost', db_user='root', db_password='352471Cf', db_name='yuqerdata'):
        """
        Initialize the Learn2Rank pipeline with database connection
        """
        self.db_host = db_host
        self.db_user = db_user
        self.db_password = db_password
        self.db_name = db_name
        self.connection = None
        
        # Set random seed for reproducibility
        torch.random.manual_seed(2021)
        np.random.seed(2021)
        
    def connect_database(self):
        """
        Connect to MySQL database
        """
        try:
            self.connection = mysql.connector.connect(
                host=self.db_host,
                user=self.db_user,
                password=self.db_password,
                database=self.db_name
            )
            print(f"Successfully connected to database: {self.db_name}")
            return True
        except mysql.connector.Error as err:
            print(f"Error connecting to database: {err}")
            print("Will use existing CSV data for demonstration")
            return False
    
    def get_database_schema(self):
        """
        Examine the database table structure
        """
        if not self.connection:
            return None
            
        cursor = self.connection.cursor()
        
        try:
            # Get table structure
            cursor.execute("DESCRIBE yq_mktstockfactorsonedayget")
            columns = cursor.fetchall()
            
            print("\nDatabase Table Structure:")
            print("-" * 80)
            print(f"{'Column':<30} {'Type':<20} {'Null':<10} {'Key':<10} {'Default':<10}")
            print("-" * 80)
            for col in columns:
                print(f"{col[0]:<30} {col[1]:<20} {col[2]:<10} {col[3]:<10} {str(col[4]):<10}")
            
            # Get sample data
            cursor.execute("SELECT * FROM yq_mktstockfactorsonedayget LIMIT 5")
            sample_data = cursor.fetchall()
            column_names = [desc[0] for desc in cursor.description]
            
            print(f"\nSample Data:")
            print("-" * 80)
            print("Columns:", column_names)
            for row in sample_data:
                print(row)
            
            # Get data statistics
            cursor.execute("SELECT COUNT(*) FROM yq_mktstockfactorsonedayget")
            count = cursor.fetchone()[0]
            print(f"\nTotal records: {count}")
            
            cursor.execute("SELECT MIN(trade_date), MAX(trade_date) FROM yq_mktstockfactorsonedayget")
            date_range = cursor.fetchone()
            print(f"Date range: {date_range[0]} to {date_range[1]}")
            
            return column_names
            
        except mysql.connector.Error as err:
            print(f"Error examining database: {err}")
            return None
        finally:
            cursor.close()
    
    def extract_database_data(self, start_date=None, end_date=None, limit=None):
        """
        Extract data from database and prepare for Learn2Rank
        """
        if not self.connection:
            return None
            
        cursor = self.connection.cursor()
        
        try:
            # Build query to extract relevant data
            query = """
            SELECT ts_code, trade_date, close, volume, market_cap, 
                   pe_ratio, pb_ratio, ps_ratio, pcf_ratio,
                   roe, roa, gross_profit_margin, net_profit_margin,
                   current_ratio, quick_ratio, debt_to_equity,
                   LEAD(close, 1) OVER (PARTITION BY ts_code ORDER BY trade_date) as next_close
            FROM yq_mktstockfactorsonedayget
            """
            
            conditions = []
            if start_date:
                conditions.append(f"trade_date >= '{start_date}'")
            if end_date:
                conditions.append(f"trade_date <= '{end_date}'")
                
            if conditions:
                query += " WHERE " + " AND ".join(conditions)
                
            query += " ORDER BY trade_date, ts_code"
            
            if limit:
                query += f" LIMIT {limit}"
            
            print(f"Executing query: {query}")
            cursor.execute(query)
            data = cursor.fetchall()
            
            # Get column names
            column_names = [desc[0] for desc in cursor.description]
            
            # Create DataFrame
            df = pd.DataFrame(data, columns=column_names)
            
            # Calculate returns
            df['return'] = (df['next_close'] - df['close']) / df['close']
            df = df.dropna()
            
            print(f"Extracted {len(df)} records from database")
            return df
            
        except mysql.connector.Error as err:
            print(f"Error extracting data: {err}")
            return None
        finally:
            cursor.close()
    
    def prepare_data_for_ranking(self, df):
        """
        Prepare data for ranking algorithms
        """
        # Select features for ranking (excluding target and identifiers)
        feature_cols = [col for col in df.columns if col not in ['ts_code', 'trade_date', 'next_close', 'return']]
        
        # Group by date and prepare ranking data
        monthly_data = []
        for date, group in df.groupby('trade_date'):
            if len(group) >= 100:  # Ensure sufficient stocks for ranking
                # Sort by return (descending for ranking)
                group_sorted = group.sort_values('return', ascending=False).reset_index(drop=True)
                
                # Prepare features and targets
                X = group_sorted[feature_cols].values
                y = group_sorted['return'].values
                
                monthly_data.append({
                    'date': date,
                    'X': torch.tensor(X, dtype=torch.float32),
                    'y': torch.tensor(y, dtype=torch.float32),
                    'df': group_sorted
                })
        
        return monthly_data
    
    def run_existing_demo(self):
        """
        Run Learn2Rank algorithms using existing CSV data
        """
        print("\n" + "="*80)
        print("RUNNING LEARN2RANK DEMO WITH EXISTING DATA")
        print("="*80)
        
        # Use existing data structure
        from glob import glob
        month_list = glob('./month/*.csv')
        if not month_list:
            # Use lambda results as demo
            month_list = glob('./lambda/*.csv')
            
        if not month_list:
            print("No existing data found. Please ensure data files are available.")
            return
            
        month_list.sort()
        print(f"Found {len(month_list)} data files")
        
        # Demonstrate with a few recent files
        demo_files = month_list[-5:] if len(month_list) >= 5 else month_list
        
        for file_path in demo_files:
            print(f"\nProcessing: {file_path}")
            
            # Read data
            df = pd.read_csv(file_path, index_col=0)
            print(f"Data shape: {df.shape}")
            print(f"Columns: {list(df.columns)}")
            
            if 'real_return' in df.columns:
                # Calculate ranking performance
                returns = df['real_return'].values
                if 'pred' in df.columns:
                    predictions = df['pred'].values
                    
                    # Sort by predictions
                    sorted_indices = np.argsort(predictions)[::-1]
                    sorted_returns = returns[sorted_indices]
                    
                    # Calculate top vs bottom performance
                    top_100_return = sorted_returns[:100].mean()
                    bottom_100_return = sorted_returns[-100:].mean()
                    long_short_return = top_100_return - bottom_100_return
                    
                    print(f"Top 100 stocks return: {top_100_return:.4f}")
                    print(f"Bottom 100 stocks return: {bottom_100_return:.4f}")
                    print(f"Long-Short return: {long_short_return:.4f}")
                else:
                    print("No predictions found in data")
            else:
                print("No return data found")
        
        return True

def main():
    """
    Main function to run the Learn2Rank pipeline
    """
    print("Learn2Rank Pipeline for Stock Ranking")
    print("="*50)
    
    # Initialize pipeline
    pipeline = Learn2RankPipeline()
    
    # Try to connect to database
    db_connected = pipeline.connect_database()
    
    if db_connected:
        print("\nExamining database structure...")
        columns = pipeline.get_database_schema()
        
        if columns:
            print("\nExtracting sample data...")
            sample_data = pipeline.extract_database_data(limit=10000)
            
            if sample_data is not None:
                print(f"Sample data shape: {sample_data.shape}")
                print("\nPreparing data for ranking...")
                ranking_data = pipeline.prepare_data_for_ranking(sample_data)
                print(f"Prepared {len(ranking_data)} time periods for ranking")
                
                # Save sample data for inspection
                sample_data.to_csv('database_sample.csv', index=False)
                print("Sample data saved to 'database_sample.csv'")
    
    # Run demonstration with existing data
    pipeline.run_existing_demo()
    
    print("\n" + "="*80)
    print("PIPELINE COMPLETED")
    print("="*80)

if __name__ == "__main__":
    main()
