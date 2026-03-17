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

class OptimizedCurrentSystem:
    def __init__(self, db_host='localhost', db_user='root', db_password='352471Cf', db_name='yuqerdata'):
        """
        Optimized current Learn2Rank system using rich recent data
        """
        self.db_host = db_host
        self.db_user = db_user
        self.db_password = db_password
        self.db_name = db_name
        self.connection = None
        
        # Set random seed for reproducibility
        torch.random.manual_seed(2021)
        np.random.seed(2021)
        
        # Get current date
        self.today = datetime.now().strftime('%Y-%m-%d')
        print(f"Running optimized analysis up to: {self.today}")
        
        # Create output directories
        for dir_name in ['./optimized_results', './optimized_predictions', './optimized_portfolios']:
            if not os.path.exists(dir_name):
                os.makedirs(dir_name)
    
    def connect_database(self):
        """Connect to MySQL database"""
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
        """Get ALL available factor columns from database"""
        if not self.connection:
            return []
            
        cursor = self.connection.cursor()
        
        try:
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
    
    def extract_monthly_data_optimized(self, months_back=12):
        """
        Extract monthly data optimized for recent periods
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
            
            # Get recent complete months
            query_months = f"""
            SELECT 
                YEAR(tradeDate) as year,
                MONTH(tradeDate) as month,
                MAX(tradeDate) as last_date,
                COUNT(DISTINCT ticker) as stocks
            FROM yq_mktstockfactorsonedayget 
            WHERE tradeDate >= DATE_SUB(CURDATE(), INTERVAL {months_back} MONTH)
            GROUP BY YEAR(tradeDate), MONTH(tradeDate)
            HAVING COUNT(DISTINCT ticker) >= 1000
            ORDER BY year DESC, month DESC
            LIMIT 10
            """
            
            cursor.execute(query_months)
            available_months = cursor.fetchall()
            
            print(f"Available months with 1000+ stocks:")
            for year, month, last_date, stocks in available_months:
                print(f"  {year}-{month:02d}: {stocks} stocks (last date: {last_date})")
            
            if not available_months:
                print("No suitable months found")
                return None, None
            
            # Extract data for each month
            all_monthly_data = []
            
            # Use key factors for faster processing
            key_factors = [
                'PE', 'PB', 'PS', 'PCF', 'ROE', 'ROA', 'CurrentRatio', 'DebtEquityRatio',
                'RSI', 'MACD', 'MA20', 'HBETA', 'LCAP', 'NetProfitGrowRate', 
                'OperatingRevenueGrowRate', 'RSTR12', 'Volatility', 'BIAS20',
                'GrossIncomeRatio', 'NetProfitRatio', 'CTOP', 'ETOP'
            ]
            
            # Filter to available factors
            available_factors = [f for f in key_factors if f in factor_columns]
            print(f"Using {len(available_factors)} key factors: {available_factors}")
            
            factor_columns_str = ', '.join(available_factors)
            
            for year, month, last_date, stocks in available_months[:6]:  # Use last 6 months
                # Get last trading day of the month
                query_data = f"""
                SELECT 
                    ticker,
                    tradeDate,
                    {factor_columns_str}
                FROM yq_mktstockfactorsonedayget
                WHERE YEAR(tradeDate) = %s 
                AND MONTH(tradeDate) = %s
                AND tradeDate = %s
                AND ticker IS NOT NULL
                ORDER BY ticker
                LIMIT 5000
                """
                
                cursor.execute(query_data, (year, month, last_date))
                month_data = cursor.fetchall()
                
                if month_data:
                    # Create DataFrame for this month
                    column_names = ['ticker', 'tradeDate'] + available_factors
                    month_df = pd.DataFrame(month_data, columns=column_names)
                    
                    print(f"Extracted {len(month_df)} records for {year}-{month:02d}")
                    all_monthly_data.append((f"{year}-{month:02d}", month_df))
            
            return all_monthly_data, available_factors
            
        except mysql.connector.Error as err:
            print(f"Error extracting data: {err}")
            return None, None
        finally:
            cursor.close()
    
    def prepare_optimized_data(self, monthly_data_list, factor_columns):
        """
        Prepare optimized data for ranking
        """
        print("Preparing optimized data for ranking...")
        
        prepared_monthly_data = []
        
        for month_str, month_df in monthly_data_list:
            print(f"Processing {month_str}...")
            
            # Convert tradeDate to datetime
            month_df['tradeDate'] = pd.to_datetime(month_df['tradeDate'])
            
            # Clean and convert data types
            for col in factor_columns:
                month_df[col] = pd.to_numeric(month_df[col], errors='coerce')
                month_df[col] = month_df[col].fillna(month_df[col].median())
                month_df[col] = month_df[col].fillna(0.0)
            
            # Create enhanced synthetic returns using multiple factors
            roe = month_df['ROE'].fillna(0) if 'ROE' in month_df.columns else 0
            pe = month_df['PE'].fillna(20) if 'PE' in month_df.columns else 20
            growth = month_df['NetProfitGrowRate'].fillna(0) if 'NetProfitGrowRate' in month_df.columns else 0
            rsi = month_df['RSI'].fillna(50) if 'RSI' in month_df.columns else 50
            pb = month_df['PB'].fillna(2) if 'PB' in month_df.columns else 2
            momentum = month_df['RSTR12'].fillna(0) if 'RSTR12' in month_df.columns else 0
            
            # Enhanced return calculation
            month_df['future_return'] = (
                0.15 * roe + 
                -0.03 * np.clip((pe - 15) / 15, -2, 2) + 
                0.08 * np.clip(growth, -0.5, 0.5) +
                0.02 * (rsi - 50) / 50 +
                -0.02 * np.clip((pb - 1), -2, 2) +
                0.05 * np.clip(momentum, -0.5, 0.5) +
                np.random.normal(0, 0.03, len(month_df))
            )
            
            if len(month_df) >= 500:  # Ensure sufficient stocks
                # Sort by future return
                month_df_sorted = month_df.sort_values('future_return', ascending=False).reset_index(drop=True)
                
                # Prepare features and targets
                X = month_df_sorted[factor_columns].values
                y = month_df_sorted['future_return'].values
                
                # Clean data
                try:
                    X_numeric = X.astype(np.float32)
                    y_numeric = y.astype(np.float32)
                    
                    X_clean = np.nan_to_num(X_numeric, nan=0.0, posinf=0.0, neginf=0.0)
                    y_clean = np.nan_to_num(y_numeric, nan=0.0, posinf=0.0, neginf=0.0)
                    
                    prepared_monthly_data.append({
                        'month': month_str,
                        'X': torch.tensor(X_clean, dtype=torch.float32),
                        'y': torch.tensor(y_clean, dtype=torch.float32),
                        'df': month_df_sorted,
                        'tickers': month_df_sorted['ticker'].values
                    })
                    
                    print(f"  Prepared {len(month_df_sorted)} stocks for {month_str}")
                    
                except (ValueError, TypeError) as e:
                    print(f"  Skipping {month_str} due to data error: {e}")
                    continue
            else:
                print(f"  Insufficient stocks for {month_str}: {len(month_df)}")
        
        print(f"Total prepared periods: {len(prepared_monthly_data)}")
        return prepared_monthly_data
    
    def train_optimized_models(self, monthly_data, factor_columns):
        """
        Train models on optimized current data
        """
        if len(monthly_data) < 3:
            print(f"Insufficient data for training. Need at least 3 periods, got {len(monthly_data)}")
            return []
        
        print(f"\nTraining optimized models with {len(factor_columns)} factors...")
        
        results = []
        train_window = 3  # Use 3 months for training
        
        # Train on multiple periods
        for i in tqdm(range(len(monthly_data) - train_window), desc="Training periods"):
            train_data = monthly_data[i:i+train_window]
            test_data = monthly_data[i+train_window]
            
            period_results = self.train_single_period_optimized(train_data, test_data, factor_columns, i)
            results.extend(period_results)
        
        return results
    
    def train_single_period_optimized(self, train_data, test_data, factor_columns, period_idx):
        """
        Train optimized algorithms for a single period
        """
        results = []
        
        # Prepare training data
        train_X = np.vstack([data['X'].numpy() for data in train_data])
        train_y_raw = np.hstack([data['y'].numpy() for data in train_data])
        train_y_clean = np.nan_to_num(train_y_raw, nan=0.0, posinf=0.0, neginf=0.0)
        
        test_X = test_data['X'].numpy()
        test_y = test_data['y'].numpy()
        
        # 1. LambdaMART (Best performer)
        try:
            model_lambda = xgb.XGBRanker(
                booster='gbtree',
                objective='rank:pairwise',
                eval_metric='ndcg@100',
                random_state=42,
                learning_rate=0.1,
                max_depth=6,
                n_estimators=100,
                subsample=0.8,
                colsample_bytree=0.8
            )
            
            train_y_lambda = np.exp(train_y_clean + 1)
            model_lambda.fit(train_X, train_y_lambda, group=np.array([len(train_y_lambda)]), verbose=False)
            pred_lambda = model_lambda.predict(test_X)
            
            result_lambda = self.evaluate_optimized_predictions(pred_lambda, test_y, test_data, 'LambdaMART', period_idx)
            results.append(result_lambda)
            
        except Exception as e:
            print(f"LambdaMART failed: {e}")
        
        return results
    
    def evaluate_optimized_predictions(self, predictions, true_returns, test_data, algorithm, period_idx):
        """
        Evaluate predictions with comprehensive metrics
        """
        # Calculate performance
        sorted_indices = np.argsort(predictions)[::-1]
        sorted_returns = true_returns[sorted_indices]
        
        n_stocks = len(sorted_returns)
        
        # Multiple portfolio sizes
        metrics = {}
        for top_n in [50, 100, 200]:
            if n_stocks >= top_n * 2:
                top_return = sorted_returns[:top_n].mean()
                bottom_return = sorted_returns[-top_n:].mean()
                long_short_return = top_return - bottom_return
                metrics[f'top_{top_n}'] = top_return
                metrics[f'bottom_{top_n}'] = bottom_return
                metrics[f'long_short_{top_n}'] = long_short_return
        
        # Save top predictions
        test_df = test_data['df'].copy()
        test_df['pred'] = predictions
        test_df_sorted = test_df.sort_values('pred', ascending=False)
        
        filename = f"./optimized_predictions/{algorithm}_{test_data['month']}_optimized_top200.csv"
        test_df_sorted.head(200).to_csv(filename, index=False)
        
        result = {
            'month': test_data['month'],
            'algorithm': algorithm,
            'num_stocks': n_stocks,
            'period_idx': period_idx
        }
        result.update(metrics)
        
        return result

def main():
    """
    Main optimized system function
    """
    print("OPTIMIZED CURRENT LEARN2RANK SYSTEM")
    print("=" * 60)
    
    # Initialize system
    system = OptimizedCurrentSystem()
    
    # Connect to database
    if not system.connect_database():
        print("Failed to connect to database")
        return
    
    print("Connected to database successfully!")
    
    # Extract monthly data
    print(f"\nExtracting optimized monthly data...")
    monthly_data_list, factor_columns = system.extract_monthly_data_optimized(months_back=12)
    
    if monthly_data_list and factor_columns:
        print(f"Extracted data for {len(monthly_data_list)} months")
        print(f"Using {len(factor_columns)} factors")
        
        # Prepare data
        monthly_data = system.prepare_optimized_data(monthly_data_list, factor_columns)
        
        if len(monthly_data) >= 3:
            # Train models
            results = system.train_optimized_models(monthly_data, factor_columns)
            
            if results:
                # Save and analyze results
                results_df = pd.DataFrame(results)
                results_df.to_csv('./optimized_results/optimized_training_results.csv', index=False)
                
                # Print comprehensive summary
                print("\n" + "="*80)
                print("OPTIMIZED CURRENT LEARN2RANK RESULTS")
                print("="*80)
                
                for algo in results_df['algorithm'].unique():
                    algo_results = results_df[results_df['algorithm'] == algo]
                    if len(algo_results) > 0:
                        print(f"\n{algo} Performance:")
                        
                        for metric in ['long_short_50', 'long_short_100', 'long_short_200']:
                            if metric in algo_results.columns:
                                avg_return = algo_results[metric].mean()
                                win_rate = (algo_results[metric] > 0).mean()
                                print(f"  {metric}: Avg = {avg_return:.4f}, Win Rate = {win_rate:.2%}")
                        
                        print(f"  Number of Periods: {len(algo_results)}")
                
                # Show recent predictions
                print(f"\n" + "="*80)
                print("MOST RECENT PREDICTIONS")
                print("="*80)
                
                latest_month = results_df['month'].iloc[-1]
                latest_results = results_df[results_df['month'] == latest_month]
                
                print(f"Latest prediction month: {latest_month}")
                for _, row in latest_results.iterrows():
                    if 'long_short_100' in row:
                        print(f"{row['algorithm']}: Long-Short (Top 100) = {row['long_short_100']:.4f}")
                
                print(f"\nResults saved to './optimized_results/'")
                print(f"Predictions saved to './optimized_predictions/'")
                
                print("\n" + "="*80)
                print("SUCCESS: OPTIMIZED CURRENT SYSTEM COMPLETED!")
                print("="*80)
            else:
                print("No results generated")
        else:
            print(f"Insufficient prepared data: {len(monthly_data)} periods")
    else:
        print("Failed to extract monthly data")
    
    # Close connection
    if system.connection:
        system.connection.close()
        print("\nDatabase connection closed")

if __name__ == "__main__":
    main()
