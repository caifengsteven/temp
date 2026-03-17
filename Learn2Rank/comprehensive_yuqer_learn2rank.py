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

class ComprehensiveYuqerLearn2Rank:
    def __init__(self, db_host='localhost', db_user='root', db_password='352471Cf', db_name='yuqerdata'):
        """
        Comprehensive Learn2Rank system using ALL factors from Yuqer database
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
        for dir_name in ['./comprehensive_results', './comprehensive_predictions']:
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
    
    def get_all_factors(self):
        """
        Get ALL available factor columns from database
        """
        if not self.connection:
            return []
            
        cursor = self.connection.cursor()
        
        try:
            # Get ALL available columns
            cursor.execute("DESCRIBE yq_mktstockfactorsonedayget")
            all_columns = cursor.fetchall()
            
            # Exclude identifier columns, get all factor columns
            exclude_cols = ['secID', 'ticker', 'tradeDate']
            factor_columns = [col[0] for col in all_columns if col[0] not in exclude_cols]
            
            print(f"Found {len(factor_columns)} factor columns in database")
            return factor_columns
            
        except mysql.connector.Error as err:
            print(f"Error getting columns: {err}")
            return []
        finally:
            cursor.close()
    
    def extract_comprehensive_data(self, sample_size=10000, start_date='2023-01-01', end_date='2023-12-31'):
        """
        Extract comprehensive data using ALL available factors
        """
        if not self.connection:
            print("No database connection")
            return None, None
            
        cursor = self.connection.cursor()
        
        try:
            # Get all factor columns
            factor_columns = self.get_all_factors()
            if not factor_columns:
                print("No factor columns found")
                return None, None
            
            # Build query with all factors
            factor_columns_str = ', '.join(factor_columns)
            
            query = f"""
            SELECT 
                ticker,
                tradeDate,
                {factor_columns_str}
            FROM yq_mktstockfactorsonedayget
            WHERE tradeDate >= %s
            AND tradeDate <= %s
            AND ticker IS NOT NULL
            ORDER BY RAND()
            LIMIT %s
            """
            
            print(f"Extracting {sample_size} records with ALL {len(factor_columns)} factors...")
            print(f"Date range: {start_date} to {end_date}")
            
            cursor.execute(query, (start_date, end_date, sample_size))
            data = cursor.fetchall()
            
            # Create column names
            column_names = ['ticker', 'tradeDate'] + factor_columns
            
            # Create DataFrame
            df = pd.DataFrame(data, columns=column_names)
            print(f"Successfully extracted {len(df)} records")
            
            return df, factor_columns
            
        except mysql.connector.Error as err:
            print(f"Error extracting data: {err}")
            return None, None
        finally:
            cursor.close()
    
    def prepare_comprehensive_data(self, df, factor_columns):
        """
        Prepare comprehensive data for ranking using ALL factors
        """
        print("Preparing comprehensive data for ranking...")
        
        # Convert tradeDate to datetime
        df['tradeDate'] = pd.to_datetime(df['tradeDate'])
        
        # Fill missing values with median for numeric columns
        print("Cleaning missing values...")
        for col in tqdm(factor_columns, desc="Processing factors"):
            if df[col].dtype in ['float64', 'int64']:
                median_val = df[col].median()
                if pd.isna(median_val):
                    median_val = 0.0
                df[col] = df[col].fillna(median_val)
        
        # Create comprehensive synthetic returns using multiple factors
        print("Creating target returns using factor combination...")
        
        # Use key factors for return calculation with proper null handling
        roe = df['ROE'].fillna(0) if 'ROE' in df.columns else 0
        pe = df['PE'].fillna(df['PE'].median() if 'PE' in df.columns and not df['PE'].isna().all() else 20) if 'PE' in df.columns else 20
        growth = df['NetProfitGrowRate'].fillna(0) if 'NetProfitGrowRate' in df.columns else 0
        rsi = df['RSI'].fillna(50) if 'RSI' in df.columns else 50
        pb = df['PB'].fillna(df['PB'].median() if 'PB' in df.columns and not df['PB'].isna().all() else 2) if 'PB' in df.columns else 2
        
        df['future_return'] = (
            0.15 * roe + 
            -0.03 * (pe - 15) / 15 + 
            0.08 * growth +
            0.02 * (rsi - 50) / 50 +
            -0.02 * (pb - 1) +
            np.random.normal(0, 0.04, len(df))
        )
        
        # Group by month for ranking
        df['month'] = df['tradeDate'].dt.to_period('M')
        monthly_data = []
        
        print("Grouping data by month...")
        for month, group in tqdm(df.groupby('month'), desc="Processing months"):
            if len(group) >= 100:  # Ensure sufficient stocks for ranking
                # Sort by future return
                group_sorted = group.sort_values('future_return', ascending=False).reset_index(drop=True)
                
                # Prepare features and targets
                X = group_sorted[factor_columns].values
                y = group_sorted['future_return'].values
                
                # Clean data
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
    
    def train_all_algorithms(self, monthly_data, factor_columns):
        """
        Train all three Learn2Rank algorithms with comprehensive factors
        """
        if len(monthly_data) < 4:
            print("Insufficient data for training")
            return []
        
        print(f"\nTraining ALL algorithms with {len(factor_columns)} factors...")
        
        results = []
        max_periods = min(8, len(monthly_data) - 2)  # Reasonable number for demo
        
        for i in tqdm(range(max_periods), desc="Training periods"):
            # Use 2 months for training, 1 for testing
            train_data = monthly_data[i:i+2]
            test_data = monthly_data[i+2]
            
            period_results = self.train_single_period(train_data, test_data, factor_columns, i)
            results.extend(period_results)
        
        return results
    
    def train_single_period(self, train_data, test_data, factor_columns, period_idx):
        """
        Train all algorithms for a single period
        """
        results = []
        
        # Prepare training data
        train_X = np.vstack([data['X'].numpy() for data in train_data])
        train_y_raw = np.hstack([data['y'].numpy() for data in train_data])
        train_y_clean = np.nan_to_num(train_y_raw, nan=0.0, posinf=0.0, neginf=0.0)
        
        test_X = test_data['X'].numpy()
        test_y = test_data['y'].numpy()
        
        # 1. LambdaMART (fastest)
        try:
            model_lambda = xgb.XGBRanker(
                booster='gbtree',
                objective='rank:pairwise',
                eval_metric='ndcg@50',
                random_state=42,
                learning_rate=0.1,
                max_depth=6,
                n_estimators=50,
                subsample=0.8,
                colsample_bytree=0.8
            )
            
            train_y_lambda = np.exp(train_y_clean + 1)
            model_lambda.fit(train_X, train_y_lambda, group=np.array([len(train_y_lambda)]), verbose=False)
            pred_lambda = model_lambda.predict(test_X)
            
            result_lambda = self.evaluate_predictions(pred_lambda, test_y, test_data, 'LambdaMART', period_idx)
            results.append(result_lambda)
            
        except Exception as e:
            print(f"LambdaMART failed: {e}")
        
        # 2. ListMLE (if we have reasonable data size)
        if len(train_X) < 5000:  # Avoid memory issues
            try:
                model_listmle = ListNet(len(factor_columns), drop_out=0.3)
                loss_func = ListMLE_loss()
                optimizer = torch.optim.Adam(model_listmle.parameters(), lr=1e-3)
                
                train_dataset = TensorDataset(
                    torch.tensor(train_X, dtype=torch.float32),
                    torch.tensor(train_y_clean, dtype=torch.float32)
                )
                
                # Quick training
                for epoch in range(10):
                    model_listmle.train()
                    data_loader = DataLoader(train_dataset, shuffle=True, batch_size=256)
                    
                    for x_batch, y_batch in data_loader:
                        pred = model_listmle(x_batch)
                        loss = loss_func(pred, y_batch)
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                
                model_listmle.eval()
                with torch.no_grad():
                    pred_listmle = model_listmle(torch.tensor(test_X, dtype=torch.float32)).numpy().flatten()
                
                result_listmle = self.evaluate_predictions(pred_listmle, test_y, test_data, 'ListMLE', period_idx)
                results.append(result_listmle)
                
            except Exception as e:
                print(f"ListMLE failed: {e}")
        
        return results
    
    def evaluate_predictions(self, predictions, true_returns, test_data, algorithm, period_idx):
        """
        Evaluate predictions and return results
        """
        # Calculate performance
        sorted_indices = np.argsort(predictions)[::-1]
        sorted_returns = true_returns[sorted_indices]
        
        # Calculate metrics
        n_stocks = len(sorted_returns)
        top_n = min(50, n_stocks // 4)
        bottom_n = min(50, n_stocks // 4)
        
        top_return = sorted_returns[:top_n].mean()
        bottom_return = sorted_returns[-bottom_n:].mean()
        long_short_return = top_return - bottom_return
        
        # Save top predictions
        test_df = test_data['df'].copy()
        test_df['pred'] = predictions
        test_df_sorted = test_df.sort_values('pred', ascending=False)
        
        filename = f"./comprehensive_predictions/{algorithm}_{test_data['month']}_top100.csv"
        test_df_sorted.head(100).to_csv(filename, index=False)
        
        return {
            'month': str(test_data['month']),
            'algorithm': algorithm,
            'top_return': top_return,
            'bottom_return': bottom_return,
            'long_short_return': long_short_return,
            'num_stocks': n_stocks,
            'top_n': top_n,
            'period_idx': period_idx
        }

def main():
    """
    Main comprehensive training function
    """
    print("COMPREHENSIVE YUQER LEARN2RANK SYSTEM")
    print("Using ALL Available Factors from Database")
    print("=" * 60)
    
    # Initialize system
    system = ComprehensiveYuqerLearn2Rank()
    
    # Connect to database
    if not system.connect_database():
        print("Failed to connect to database")
        return
    
    print("Connected to database successfully!")
    
    # Extract comprehensive data
    df, factor_columns = system.extract_comprehensive_data(sample_size=8000)
    
    if df is not None and factor_columns is not None:
        print(f"Data shape: {df.shape}")
        print(f"Using ALL {len(factor_columns)} factors!")
        
        # Save comprehensive sample
        df.to_csv('comprehensive_yuqer_sample.csv', index=False)
        print("Comprehensive sample saved to 'comprehensive_yuqer_sample.csv'")
        
        # Prepare data
        monthly_data = system.prepare_comprehensive_data(df, factor_columns)
        
        if len(monthly_data) > 0:
            # Train all algorithms
            results = system.train_all_algorithms(monthly_data, factor_columns)
            
            if results:
                # Save and analyze results
                results_df = pd.DataFrame(results)
                results_df.to_csv('./comprehensive_results/all_algorithms_results.csv', index=False)
                
                # Print comprehensive summary
                print("\n" + "="*80)
                print("COMPREHENSIVE RESULTS WITH ALL FACTORS")
                print("="*80)
                
                for algo in results_df['algorithm'].unique():
                    algo_results = results_df[results_df['algorithm'] == algo]
                    if len(algo_results) > 0:
                        avg_return = algo_results['long_short_return'].mean()
                        std_return = algo_results['long_short_return'].std()
                        win_rate = (algo_results['long_short_return'] > 0).mean()
                        
                        print(f"\n{algo} (with {len(factor_columns)} factors):")
                        print(f"  Average Long-Short Return: {avg_return:.4f}")
                        print(f"  Standard Deviation: {std_return:.4f}")
                        print(f"  Win Rate: {win_rate:.2%}")
                        print(f"  Number of Periods: {len(algo_results)}")
                
                print(f"\nDetailed results saved to './comprehensive_results/'")
                print(f"Top predictions saved to './comprehensive_predictions/'")
                print(f"Sample data saved to 'comprehensive_yuqer_sample.csv'")
                
                print("\n" + "="*80)
                print("SUCCESS: COMPREHENSIVE LEARN2RANK WITH ALL FACTORS!")
                print("="*80)
            else:
                print("No results generated")
        else:
            print("No monthly data prepared")
    else:
        print("Failed to extract data")
    
    # Close connection
    if system.connection:
        system.connection.close()
        print("\nDatabase connection closed")

if __name__ == "__main__":
    main()
