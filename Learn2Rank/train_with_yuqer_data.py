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

class YuqerDataTrainer:
    def __init__(self, db_host='localhost', db_user='root', db_password='352471Cf', db_name='yuqerdata'):
        """
        Initialize trainer for Yuqer database
        """
        self.db_host = db_host
        self.db_user = db_user
        self.db_password = db_password
        self.db_name = db_name
        self.connection = None
        
        # Set random seed for reproducibility
        torch.random.manual_seed(2021)
        np.random.seed(2021)
        
        # Create output directories
        for dir_name in ['./yuqer_ranknet_results', './yuqer_listmle_results', './yuqer_lambda_results']:
            if not os.path.exists(dir_name):
                os.makedirs(dir_name)
    
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
    
    def get_key_factors(self):
        """
        Select key financial factors for ranking
        """
        # Key fundamental factors
        fundamental_factors = [
            'PE', 'PB', 'PS', 'PCF',  # Valuation ratios
            'ROE', 'ROA', 'ROE5', 'ROA5',  # Profitability
            'CurrentRatio', 'QuickRatio', 'DebtEquityRatio',  # Liquidity & Leverage
            'NetProfitGrowRate', 'OperatingRevenueGrowRate', 'TotalAssetGrowRate',  # Growth
            'GrossIncomeRatio', 'NetProfitRatio', 'OperatingProfitRatio',  # Margins
        ]
        
        # Key technical factors
        technical_factors = [
            'RSI', 'MACD', 'BIAS20', 'CCI20',  # Momentum
            'MA5', 'MA20', 'MA60', 'EMA20',  # Moving averages
            'HBETA', 'HSIGMA', 'Volatility',  # Risk measures
            'RSTR12', 'RSTR24',  # Momentum
        ]
        
        # Market factors
        market_factors = [
            'LCAP', 'LFLO',  # Size factors
            'CTOP', 'ETOP',  # Value factors
        ]
        
        return fundamental_factors + technical_factors + market_factors
    
    def extract_sample_data(self, start_date='2020-01-01', end_date='2023-12-31', sample_size=50000):
        """
        Extract sample data for training
        """
        if not self.connection:
            print("No database connection")
            return None
            
        cursor = self.connection.cursor()
        
        try:
            # Get key factors
            key_factors = self.get_key_factors()
            factor_columns = ', '.join(key_factors)
            
            # Query to extract sample data
            query = f"""
            SELECT 
                ticker,
                tradeDate,
                {factor_columns}
            FROM yq_mktstockfactorsonedayget
            WHERE tradeDate BETWEEN %s AND %s
            AND ticker IS NOT NULL
            ORDER BY RAND()
            LIMIT %s
            """
            
            print(f"Extracting sample data from {start_date} to {end_date}...")
            print(f"Using {len(key_factors)} key factors")
            
            cursor.execute(query, (start_date, end_date, sample_size))
            data = cursor.fetchall()
            
            # Get column names
            column_names = ['ticker', 'tradeDate'] + key_factors
            
            # Create DataFrame
            df = pd.DataFrame(data, columns=column_names)
            print(f"Extracted {len(df)} records")
            
            return df
            
        except mysql.connector.Error as err:
            print(f"Error extracting data: {err}")
            return None
        finally:
            cursor.close()
    
    def extract_monthly_data(self, start_date='2020-01-01', end_date='2023-12-31'):
        """
        Extract monthly data for systematic training
        """
        if not self.connection:
            print("No database connection")
            return None
            
        cursor = self.connection.cursor()
        
        try:
            # Get key factors
            key_factors = self.get_key_factors()
            factor_columns = ', '.join(key_factors)
            
            # Query to extract monthly data (last trading day of each month)
            query = f"""
            SELECT 
                ticker,
                tradeDate,
                {factor_columns}
            FROM yq_mktstockfactorsonedayget t1
            WHERE tradeDate BETWEEN %s AND %s
            AND tradeDate = (
                SELECT MAX(tradeDate) 
                FROM yq_mktstockfactorsonedayget t2 
                WHERE t2.ticker = t1.ticker 
                AND YEAR(t2.tradeDate) = YEAR(t1.tradeDate)
                AND MONTH(t2.tradeDate) = MONTH(t1.tradeDate)
            )
            AND ticker IS NOT NULL
            ORDER BY tradeDate, ticker
            """
            
            print(f"Extracting monthly data from {start_date} to {end_date}...")
            print(f"Using {len(key_factors)} key factors")
            
            cursor.execute(query, (start_date, end_date))
            data = cursor.fetchall()
            
            # Get column names
            column_names = ['ticker', 'tradeDate'] + key_factors
            
            # Create DataFrame
            df = pd.DataFrame(data, columns=column_names)
            print(f"Extracted {len(df)} monthly records")
            
            return df
            
        except mysql.connector.Error as err:
            print(f"Error extracting data: {err}")
            return None
        finally:
            cursor.close()
    
    def prepare_ranking_data(self, df):
        """
        Prepare data for ranking algorithms
        """
        print("Preparing data for ranking...")
        
        # Calculate future returns (using next month's data)
        df = df.sort_values(['ticker', 'tradeDate'])
        
        # For simplicity, we'll use a proxy return based on momentum factors
        # In practice, you would calculate actual returns using price data
        df['future_return'] = (
            0.1 * df['RSTR12'].fillna(0) + 
            0.05 * df['ROE'].fillna(0) + 
            -0.02 * df['PE'].fillna(df['PE'].median()) + 
            0.03 * df['NetProfitGrowRate'].fillna(0) +
            np.random.normal(0, 0.02, len(df))
        )
        
        # Select feature columns (exclude identifiers and target)
        feature_cols = [col for col in df.columns if col not in ['ticker', 'tradeDate', 'future_return']]
        
        # Fill missing values with median
        for col in feature_cols:
            if df[col].dtype in ['float64', 'int64']:
                df[col] = df[col].fillna(df[col].median())
        
        # Group by date and prepare monthly data
        monthly_data = []
        for date, group in tqdm(df.groupby('tradeDate')):
            if len(group) >= 100:  # Ensure sufficient stocks
                # Sort by future return (descending)
                group_sorted = group.sort_values('future_return', ascending=False).reset_index(drop=True)
                
                # Prepare features and targets
                X = group_sorted[feature_cols].values
                y = group_sorted['future_return'].values
                
                # Handle any remaining NaN values
                if not np.isnan(X).any() and not np.isnan(y).any():
                    monthly_data.append({
                        'date': date,
                        'X': torch.tensor(X, dtype=torch.float32),
                        'y': torch.tensor(y, dtype=torch.float32),
                        'df': group_sorted,
                        'tickers': group_sorted['ticker'].values
                    })
        
        print(f"Prepared {len(monthly_data)} time periods for ranking")
        return monthly_data, feature_cols
    
    def train_lambdamart_quick(self, monthly_data, feature_cols, train_window=3):
        """
        Quick LambdaMART training (fastest algorithm)
        """
        print("\nTraining LambdaMART models (Quick Demo)...")
        
        results = []
        max_periods = min(10, len(monthly_data) - train_window - 1)  # Limit for demo
        
        for i in tqdm(range(max_periods)):
            # Prepare training and test data
            train_data = monthly_data[i:i+train_window]
            test_data = monthly_data[i+train_window]
            
            # Prepare training data for XGBoost
            train_X = np.vstack([data['X'].numpy() for data in train_data])
            train_y = np.hstack([np.exp(data['y'].numpy() + 1) for data in train_data])  # Ensure positive scores
            
            # Initialize and train model
            model = xgb.XGBRanker(
                booster='gbtree',
                objective='rank:pairwise',
                eval_metric='ndcg@100',
                random_state=42,
                learning_rate=0.1,
                max_depth=4,
                n_estimators=20,  # Reduced for speed
                subsample=0.8
            )
            
            model.fit(train_X, train_y, group=np.array([len(train_y)]), verbose=False)
            
            # Evaluate on test data
            pred = model.predict(test_data['X'].numpy())
            
            # Calculate performance
            sorted_indices = np.argsort(pred)[::-1]
            sorted_returns = test_data['y'].numpy()[sorted_indices]
            
            top_50_return = sorted_returns[:50].mean()
            bottom_50_return = sorted_returns[-50:].mean()
            long_short_return = top_50_return - bottom_50_return
            
            result = {
                'date': test_data['date'],
                'algorithm': 'LambdaMART',
                'top_50_return': top_50_return,
                'bottom_50_return': bottom_50_return,
                'long_short_return': long_short_return,
                'num_stocks': len(sorted_returns)
            }
            results.append(result)
            
            # Save top predictions
            test_df = test_data['df'].copy()
            test_df['pred'] = pred
            test_df_sorted = test_df.sort_values('pred', ascending=False)
            test_df_sorted.head(100).to_csv(f'./yuqer_lambda_results/{test_data["date"]}_top100.csv', index=False)
        
        return results

def main():
    """
    Main training function for Yuqer data
    """
    print("Yuqer Database Learn2Rank Training")
    print("=" * 50)
    
    # Initialize trainer
    trainer = YuqerDataTrainer()
    
    # Connect to database
    if not trainer.connect_database():
        print("Failed to connect to database")
        return
    
    print("Connected to database successfully!")
    
    # Extract monthly data
    print("\nExtracting monthly data...")
    df = trainer.extract_monthly_data(start_date='2022-01-01', end_date='2023-12-31')
    
    if df is None or len(df) == 0:
        print("No data extracted. Trying sample extraction...")
        df = trainer.extract_sample_data(sample_size=10000)
    
    if df is not None and len(df) > 0:
        print(f"Extracted data shape: {df.shape}")
        
        # Save sample data
        df.to_csv('yuqer_sample_data.csv', index=False)
        print("Sample data saved to 'yuqer_sample_data.csv'")
        
        # Prepare ranking data
        monthly_data, feature_cols = trainer.prepare_ranking_data(df)
        
        if len(monthly_data) > 3:
            print(f"Number of features: {len(feature_cols)}")
            print(f"Feature columns: {feature_cols[:10]}...")  # Show first 10
            
            # Train LambdaMART (fastest for demo)
            results = trainer.train_lambdamart_quick(monthly_data, feature_cols)
            
            # Save and display results
            if results:
                results_df = pd.DataFrame(results)
                results_df.to_csv('yuqer_training_results.csv', index=False)
                
                # Print summary
                print("\n" + "="*60)
                print("YUQER DATA TRAINING RESULTS")
                print("="*60)
                avg_long_short = results_df['long_short_return'].mean()
                std_long_short = results_df['long_short_return'].std()
                win_rate = (results_df['long_short_return'] > 0).mean()
                
                print(f"LambdaMART Performance:")
                print(f"  Average Long-Short Return: {avg_long_short:.4f}")
                print(f"  Standard Deviation: {std_long_short:.4f}")
                print(f"  Win Rate: {win_rate:.2%}")
                print(f"  Number of Periods: {len(results_df)}")
                
                print(f"\nDetailed results saved to 'yuqer_training_results.csv'")
                print("Top stock predictions saved to './yuqer_lambda_results/' folder")
            else:
                print("No results generated")
        else:
            print("Insufficient data for training")
    else:
        print("No data extracted from database")
    
    # Close connection
    if trainer.connection:
        trainer.connection.close()
        print("\nDatabase connection closed")

if __name__ == "__main__":
    main()
