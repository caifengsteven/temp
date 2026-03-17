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

class Learn2RankTrainer:
    def __init__(self, db_host='localhost', db_user='root', db_password='352471Cf', db_name='yuqerdata'):
        """
        Initialize the Learn2Rank trainer
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
        for dir_name in ['./ranknet_results', './listmle_results', './lambda_results']:
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
    
    def extract_factor_data(self, start_date='2020-01-01', end_date='2023-12-31'):
        """
        Extract factor data from the database
        """
        if not self.connection:
            print("No database connection")
            return None
            
        cursor = self.connection.cursor()
        
        try:
            # Query to extract comprehensive factor data
            query = """
            SELECT 
                ts_code,
                trade_date,
                close,
                volume,
                market_cap,
                pe_ratio,
                pb_ratio,
                ps_ratio,
                pcf_ratio,
                roe,
                roa,
                gross_profit_margin,
                net_profit_margin,
                current_ratio,
                quick_ratio,
                debt_to_equity_ratio,
                revenue_growth,
                eps_growth,
                book_value_per_share,
                cash_per_share
            FROM yq_mktstockfactorsonedayget
            WHERE trade_date BETWEEN %s AND %s
            AND ts_code IS NOT NULL
            AND close IS NOT NULL
            ORDER BY trade_date, ts_code
            """
            
            print(f"Extracting data from {start_date} to {end_date}...")
            cursor.execute(query, (start_date, end_date))
            data = cursor.fetchall()
            
            # Get column names
            column_names = [desc[0] for desc in cursor.description]
            
            # Create DataFrame
            df = pd.DataFrame(data, columns=column_names)
            print(f"Extracted {len(df)} records")
            
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
        
        # Calculate future returns
        df = df.sort_values(['ts_code', 'trade_date'])
        df['future_return'] = df.groupby('ts_code')['close'].pct_change(periods=1).shift(-1)
        
        # Remove rows with missing future returns
        df = df.dropna(subset=['future_return'])
        
        # Select feature columns (exclude identifiers and target)
        feature_cols = [col for col in df.columns if col not in 
                       ['ts_code', 'trade_date', 'close', 'future_return']]
        
        # Fill missing values with median
        for col in feature_cols:
            if df[col].dtype in ['float64', 'int64']:
                df[col] = df[col].fillna(df[col].median())
        
        # Group by date and prepare monthly data
        monthly_data = []
        for date, group in tqdm(df.groupby('trade_date')):
            if len(group) >= 200:  # Ensure sufficient stocks
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
                        'ts_codes': group_sorted['ts_code'].values
                    })
        
        print(f"Prepared {len(monthly_data)} time periods for ranking")
        return monthly_data, feature_cols
    
    def train_ranknet(self, monthly_data, feature_cols, train_window=6):
        """
        Train RankNet model
        """
        print("\nTraining RankNet models...")
        
        results = []
        for i in tqdm(range(len(monthly_data) - train_window - 1)):
            # Prepare training and test data
            train_data = monthly_data[i:i+train_window]
            test_data = monthly_data[i+train_window]
            
            # Initialize model
            model = RankNet(len(feature_cols), drop_out=0.5)
            loss_func = nn.BCELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            
            # Prepare pairwise training data
            train_X = [data['X'] for data in train_data]
            pair_train_data = pairwise_data(train_X, step=100)
            
            # Train model
            for epoch in range(5):
                model.train()
                data_loader = DataLoader(pair_train_data, shuffle=True, batch_size=512)
                epoch_loss = 0
                
                for x_i, x_j in data_loader:
                    sig = model(x_i, x_j)
                    loss = loss_func(sig, torch.ones_like(sig))
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
            
            # Evaluate on test data
            model.eval()
            with torch.no_grad():
                pred = model.model(test_data['X']).view(-1).numpy()
            
            # Calculate performance
            test_df = test_data['df'].copy()
            test_df['pred'] = pred
            
            # Sort by predictions and calculate returns
            sorted_indices = np.argsort(pred)[::-1]
            sorted_returns = test_data['y'].numpy()[sorted_indices]
            
            top_100_return = sorted_returns[:100].mean()
            bottom_100_return = sorted_returns[-100:].mean()
            long_short_return = top_100_return - bottom_100_return
            
            result = {
                'date': test_data['date'],
                'algorithm': 'RankNet',
                'top_100_return': top_100_return,
                'bottom_100_return': bottom_100_return,
                'long_short_return': long_short_return
            }
            results.append(result)
            
            # Save predictions
            test_df.to_csv(f'./ranknet_results/{test_data["date"]}.csv', index=False)
        
        return results
    
    def train_listmle(self, monthly_data, feature_cols, train_window=6):
        """
        Train ListMLE model
        """
        print("\nTraining ListMLE models...")
        
        results = []
        for i in tqdm(range(len(monthly_data) - train_window - 1)):
            # Prepare training and test data
            train_data = monthly_data[i:i+train_window]
            val_data = monthly_data[i+train_window-1]
            test_data = monthly_data[i+train_window]
            
            # Initialize model
            model = ListNet(len(feature_cols), drop_out=0.5)
            loss_func = ListMLE_loss()
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            
            # Prepare training dataset
            train_X = torch.cat([data['X'] for data in train_data], axis=0)
            train_y = torch.cat([data['y'] for data in train_data], axis=0)
            train_dataset = TensorDataset(train_X, train_y)
            
            # Train with early stopping
            best_model = None
            best_ndcg = -np.inf
            patience = 10
            patience_counter = 0
            
            for epoch in range(100):
                # Training
                model.train()
                data_loader = DataLoader(train_dataset, shuffle=True, batch_size=256)
                epoch_loss = 0
                
                for x_batch, y_batch in data_loader:
                    pred = model(x_batch)
                    loss = loss_func(pred, y_batch)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                
                # Validation
                model.eval()
                with torch.no_grad():
                    val_pred = model(val_data['X'])
                    val_ndcg = ndcg(val_pred, val_data['y'])
                
                if val_ndcg > best_ndcg:
                    best_ndcg = val_ndcg
                    best_model = copy.deepcopy(model)
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        break
            
            # Evaluate on test data
            best_model.eval()
            with torch.no_grad():
                pred = best_model(test_data['X']).view(-1).numpy()
            
            # Calculate performance
            test_df = test_data['df'].copy()
            test_df['pred'] = pred
            
            # Sort by predictions and calculate returns
            sorted_indices = np.argsort(pred)[::-1]
            sorted_returns = test_data['y'].numpy()[sorted_indices]
            
            top_100_return = sorted_returns[:100].mean()
            bottom_100_return = sorted_returns[-100:].mean()
            long_short_return = top_100_return - bottom_100_return
            
            result = {
                'date': test_data['date'],
                'algorithm': 'ListMLE',
                'top_100_return': top_100_return,
                'bottom_100_return': bottom_100_return,
                'long_short_return': long_short_return
            }
            results.append(result)
            
            # Save predictions
            test_df.to_csv(f'./listmle_results/{test_data["date"]}.csv', index=False)
        
        return results

def main():
    """
    Main training function
    """
    print("Learn2Rank Model Training Pipeline")
    print("=" * 50)
    
    # Initialize trainer
    trainer = Learn2RankTrainer()
    
    # Try to connect to database
    if trainer.connect_database():
        print("Connected to database successfully!")
        
        # Extract data
        df = trainer.extract_factor_data(start_date='2020-01-01', end_date='2023-12-31')
        
        if df is not None and len(df) > 0:
            print(f"Extracted data shape: {df.shape}")
            
            # Prepare ranking data
            monthly_data, feature_cols = trainer.prepare_ranking_data(df)
            
            if len(monthly_data) > 10:
                print(f"Number of features: {len(feature_cols)}")
                print(f"Feature columns: {feature_cols}")
                
                # Train models
                ranknet_results = trainer.train_ranknet(monthly_data, feature_cols)
                listmle_results = trainer.train_listmle(monthly_data, feature_cols)
                
                # Combine and save results
                all_results = ranknet_results + listmle_results
                results_df = pd.DataFrame(all_results)
                results_df.to_csv('training_results.csv', index=False)
                
                # Print summary
                print("\nTraining Results Summary:")
                print("=" * 40)
                for algo in ['RankNet', 'ListMLE']:
                    algo_results = results_df[results_df['algorithm'] == algo]
                    if len(algo_results) > 0:
                        avg_long_short = algo_results['long_short_return'].mean()
                        print(f"{algo} - Average Long-Short Return: {avg_long_short:.4f}")
                
                print(f"\nDetailed results saved to 'training_results.csv'")
                print("Individual predictions saved to respective result folders")
            else:
                print("Insufficient data for training")
        else:
            print("No data extracted from database")
    else:
        print("Could not connect to database")
        print("Please ensure MySQL is running and credentials are correct")

if __name__ == "__main__":
    main()
