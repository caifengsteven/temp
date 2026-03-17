import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import mysql.connector
from datetime import datetime, timedelta
import os
from tqdm import tqdm
import warnings
import copy
warnings.filterwarnings('ignore')

# Import the existing algorithms
from algs.ranknet import RankNet, pairwise_data
from algs.listnet import ListNet, ListMLE_loss, ndcg
import xgboost as xgb

class QuickYuqerDemo:
    def __init__(self, db_host='localhost', db_user='root', db_password='352471Cf', db_name='yuqerdata'):
        """
        Quick demo with Yuqer database
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
            return False
    
    def get_sample_data(self, sample_size=5000):
        """
        Get sample data for quick demo using ALL available factors
        """
        if not self.connection:
            print("No database connection")
            return None

        cursor = self.connection.cursor()

        try:
            # Get ALL available columns except identifiers
            cursor.execute("DESCRIBE yq_mktstockfactorsonedayget")
            all_columns = cursor.fetchall()

            # Exclude identifier columns and get all factor columns
            exclude_cols = ['secID', 'ticker', 'tradeDate']
            key_factors = [col[0] for col in all_columns if col[0] not in exclude_cols]

            factor_columns = ', '.join(key_factors)
            
            # Simple random sample query
            query = f"""
            SELECT 
                ticker,
                tradeDate,
                {factor_columns}
            FROM yq_mktstockfactorsonedayget
            WHERE tradeDate >= '2023-01-01'
            AND tradeDate <= '2023-12-31'
            AND ticker IS NOT NULL
            ORDER BY RAND()
            LIMIT %s
            """
            
            print(f"Extracting {sample_size} sample records...")
            print(f"Using ALL {len(key_factors)} available factors from database")
            
            cursor.execute(query, (sample_size,))
            data = cursor.fetchall()
            
            # Get column names
            column_names = ['ticker', 'tradeDate'] + key_factors
            
            # Create DataFrame
            df = pd.DataFrame(data, columns=column_names)
            print(f"Extracted {len(df)} records")
            
            return df, key_factors
            
        except mysql.connector.Error as err:
            print(f"Error extracting data: {err}")
            return None, None
        finally:
            cursor.close()
    
    def prepare_ranking_data(self, df, feature_cols):
        """
        Prepare data for ranking
        """
        print("Preparing data for ranking...")
        
        # Fill missing values
        for col in feature_cols:
            if df[col].dtype in ['float64', 'int64']:
                df[col] = df[col].fillna(df[col].median())
        
        # Create synthetic future returns based on factors
        df['future_return'] = (
            0.1 * df['ROE'].fillna(0) + 
            -0.02 * df['PE'].fillna(df['PE'].median()) + 
            0.05 * df['NetProfitGrowRate'].fillna(0) +
            0.03 * (df['RSI'].fillna(50) - 50) / 50 +
            np.random.normal(0, 0.03, len(df))
        )
        
        # Convert tradeDate to datetime and group by month for ranking
        df['tradeDate'] = pd.to_datetime(df['tradeDate'])
        df['month'] = df['tradeDate'].dt.to_period('M')
        monthly_data = []
        
        for month, group in df.groupby('month'):
            if len(group) >= 50:  # Ensure sufficient stocks
                # Sort by future return
                group_sorted = group.sort_values('future_return', ascending=False).reset_index(drop=True)
                
                # Prepare features and targets
                X = group_sorted[feature_cols].values
                y = group_sorted['future_return'].values
                
                # Check for valid data and clean
                X_clean = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
                y_clean = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

                if X_clean.shape[0] > 0 and y_clean.shape[0] > 0:
                    monthly_data.append({
                        'month': month,
                        'X': torch.tensor(X_clean, dtype=torch.float32),
                        'y': torch.tensor(y_clean, dtype=torch.float32),
                        'df': group_sorted,
                        'tickers': group_sorted['ticker'].values
                    })
        
        print(f"Prepared {len(monthly_data)} monthly periods for ranking")
        return monthly_data
    
    def train_lambdamart_demo(self, monthly_data, feature_cols):
        """
        Quick LambdaMART demo
        """
        print("\nTraining LambdaMART models...")
        
        if len(monthly_data) < 4:
            print("Insufficient data for training")
            return []
        
        results = []
        max_periods = min(6, len(monthly_data) - 2)  # Limit for demo
        
        for i in tqdm(range(max_periods)):
            # Use 2 months for training, 1 for testing
            train_data = monthly_data[i:i+2]
            test_data = monthly_data[i+2]
            
            # Prepare training data
            train_X = np.vstack([data['X'].numpy() for data in train_data])
            train_y_raw = np.hstack([data['y'].numpy() for data in train_data])

            # Clean and transform target values
            train_y_clean = np.nan_to_num(train_y_raw, nan=0.0, posinf=0.0, neginf=0.0)
            train_y = np.exp(train_y_clean + 1)  # Ensure positive values
            
            # Train model
            model = xgb.XGBRanker(
                booster='gbtree',
                objective='rank:pairwise',
                eval_metric='ndcg@50',
                random_state=42,
                learning_rate=0.1,
                max_depth=4,
                n_estimators=10,  # Very fast for demo
                subsample=0.8
            )
            
            model.fit(train_X, train_y, group=np.array([len(train_y)]), verbose=False)
            
            # Predict on test data
            pred = model.predict(test_data['X'].numpy())
            
            # Calculate performance
            sorted_indices = np.argsort(pred)[::-1]
            sorted_returns = test_data['y'].numpy()[sorted_indices]
            
            # Calculate top vs bottom performance
            n_stocks = len(sorted_returns)
            top_n = min(20, n_stocks // 4)
            bottom_n = min(20, n_stocks // 4)
            
            top_return = sorted_returns[:top_n].mean()
            bottom_return = sorted_returns[-bottom_n:].mean()
            long_short_return = top_return - bottom_return
            
            result = {
                'month': str(test_data['month']),
                'algorithm': 'LambdaMART',
                'top_return': top_return,
                'bottom_return': bottom_return,
                'long_short_return': long_short_return,
                'num_stocks': n_stocks,
                'top_n': top_n
            }
            results.append(result)
            
            # Show top predictions
            test_df = test_data['df'].copy()
            test_df['pred'] = pred
            test_df_sorted = test_df.sort_values('pred', ascending=False)
            
            print(f"\nMonth {test_data['month']} - Top 10 Stock Predictions:")
            print(test_df_sorted[['ticker', 'future_return', 'pred']].head(10).to_string(index=False))
        
        return results

def main():
    """
    Quick demo with Yuqer data
    """
    print("Quick Yuqer Database Learn2Rank Demo")
    print("=" * 50)
    
    # Initialize demo
    demo = QuickYuqerDemo()
    
    # Connect to database
    if not demo.connect_database():
        print("Failed to connect to database")
        return
    
    print("Connected to database successfully!")
    
    # Get sample data
    df, feature_cols = demo.get_sample_data(sample_size=3000)
    
    if df is not None and len(df) > 0:
        print(f"Sample data shape: {df.shape}")
        
        # Save sample data
        df.to_csv('yuqer_quick_sample.csv', index=False)
        print("Sample data saved to 'yuqer_quick_sample.csv'")
        
        # Prepare ranking data
        monthly_data = demo.prepare_ranking_data(df, feature_cols)
        
        if len(monthly_data) > 0:
            print(f"Number of features: {len(feature_cols)}")
            print(f"Features: {feature_cols}")
            
            # Train and evaluate
            results = demo.train_lambdamart_demo(monthly_data, feature_cols)
            
            if results:
                # Save and display results
                results_df = pd.DataFrame(results)
                results_df.to_csv('yuqer_quick_results.csv', index=False)
                
                # Print summary
                print("\n" + "="*60)
                print("YUQER QUICK DEMO RESULTS")
                print("="*60)
                avg_long_short = results_df['long_short_return'].mean()
                std_long_short = results_df['long_short_return'].std()
                win_rate = (results_df['long_short_return'] > 0).mean()
                
                print(f"LambdaMART Performance:")
                print(f"  Average Long-Short Return: {avg_long_short:.4f}")
                print(f"  Standard Deviation: {std_long_short:.4f}")
                print(f"  Win Rate: {win_rate:.2%}")
                print(f"  Number of Test Periods: {len(results_df)}")
                print(f"  Average Stocks per Period: {results_df['num_stocks'].mean():.0f}")
                
                print(f"\nDetailed results saved to 'yuqer_quick_results.csv'")
                print("Sample data saved to 'yuqer_quick_sample.csv'")
                
                print("\n" + "="*60)
                print("SUCCESS: Learn2Rank working with your Yuqer database!")
                print("="*60)
            else:
                print("No results generated")
        else:
            print("No monthly data prepared")
    else:
        print("No data extracted from database")
    
    # Close connection
    if demo.connection:
        demo.connection.close()
        print("\nDatabase connection closed")

if __name__ == "__main__":
    main()
