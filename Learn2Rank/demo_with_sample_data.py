import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
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

def generate_sample_factor_data():
    """
    Generate sample factor data similar to what would come from the database
    """
    print("Generating sample factor data...")
    
    # Generate sample data for 500 stocks over 24 months
    np.random.seed(42)
    
    dates = pd.date_range('2022-01-01', '2023-12-31', freq='M')
    stock_codes = [f"60{i:04d}" for i in range(500)]
    
    all_data = []
    
    for date in dates:
        for stock in stock_codes:
            # Generate realistic financial factors
            data = {
                'ts_code': stock,
                'trade_date': date,
                'close': np.random.uniform(10, 100),
                'volume': np.random.uniform(1000000, 100000000),
                'market_cap': np.random.uniform(1e9, 1e12),
                'pe_ratio': np.random.uniform(5, 50),
                'pb_ratio': np.random.uniform(0.5, 10),
                'ps_ratio': np.random.uniform(0.5, 20),
                'pcf_ratio': np.random.uniform(2, 30),
                'roe': np.random.uniform(-0.2, 0.3),
                'roa': np.random.uniform(-0.1, 0.2),
                'gross_profit_margin': np.random.uniform(0.1, 0.8),
                'net_profit_margin': np.random.uniform(-0.1, 0.3),
                'current_ratio': np.random.uniform(0.5, 5),
                'quick_ratio': np.random.uniform(0.3, 3),
                'debt_to_equity_ratio': np.random.uniform(0, 3),
                'revenue_growth': np.random.uniform(-0.3, 0.5),
                'eps_growth': np.random.uniform(-0.5, 1.0),
                'book_value_per_share': np.random.uniform(1, 50),
                'cash_per_share': np.random.uniform(0.5, 20)
            }
            all_data.append(data)
    
    df = pd.DataFrame(all_data)
    
    # Calculate future returns (target variable)
    df = df.sort_values(['ts_code', 'trade_date'])
    df['future_return'] = df.groupby('ts_code')['close'].pct_change(periods=1).shift(-1)
    
    # Add some correlation between factors and returns for realism
    df['future_return'] = (
        0.1 * df['roe'] + 
        0.05 * df['roa'] + 
        -0.02 * df['pe_ratio'] + 
        0.03 * df['revenue_growth'] +
        np.random.normal(0, 0.05, len(df))
    )
    
    # Remove rows with missing future returns
    df = df.dropna(subset=['future_return'])
    
    print(f"Generated {len(df)} sample records")
    return df

def prepare_ranking_data(df):
    """
    Prepare data for ranking algorithms
    """
    print("Preparing data for ranking...")
    
    # Select feature columns (exclude identifiers and target)
    feature_cols = [col for col in df.columns if col not in 
                   ['ts_code', 'trade_date', 'close', 'future_return']]
    
    # Group by date and prepare monthly data
    monthly_data = []
    for date, group in tqdm(df.groupby('trade_date')):
        if len(group) >= 200:  # Ensure sufficient stocks
            # Sort by future return (descending)
            group_sorted = group.sort_values('future_return', ascending=False).reset_index(drop=True)
            
            # Prepare features and targets
            X = group_sorted[feature_cols].values
            y = group_sorted['future_return'].values
            
            monthly_data.append({
                'date': date,
                'X': torch.tensor(X, dtype=torch.float32),
                'y': torch.tensor(y, dtype=torch.float32),
                'df': group_sorted,
                'ts_codes': group_sorted['ts_code'].values
            })
    
    print(f"Prepared {len(monthly_data)} time periods for ranking")
    return monthly_data, feature_cols

def train_and_evaluate_ranknet(monthly_data, feature_cols, train_window=6):
    """
    Train and evaluate RankNet model
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
        pair_train_data = pairwise_data(train_X, step=50)
        
        # Train model
        for epoch in range(3):  # Reduced for demo
            model.train()
            data_loader = DataLoader(pair_train_data, shuffle=True, batch_size=256)
            
            for x_i, x_j in data_loader:
                sig = model(x_i, x_j)
                loss = loss_func(sig, torch.ones_like(sig))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
        # Evaluate on test data
        model.eval()
        with torch.no_grad():
            pred = model.model(test_data['X']).view(-1).numpy()
        
        # Calculate performance
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
    
    return results

def train_and_evaluate_listmle(monthly_data, feature_cols, train_window=6):
    """
    Train and evaluate ListMLE model
    """
    print("\nTraining ListMLE models...")
    
    results = []
    for i in tqdm(range(len(monthly_data) - train_window - 1)):
        # Prepare training and test data
        train_data = monthly_data[i:i+train_window]
        test_data = monthly_data[i+train_window]
        
        # Initialize model
        model = ListNet(len(feature_cols), drop_out=0.5)
        loss_func = ListMLE_loss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        # Prepare training dataset
        train_X = torch.cat([data['X'] for data in train_data], axis=0)
        train_y = torch.cat([data['y'] for data in train_data], axis=0)
        train_dataset = TensorDataset(train_X, train_y)
        
        # Train model
        for epoch in range(5):  # Reduced for demo
            model.train()
            data_loader = DataLoader(train_dataset, shuffle=True, batch_size=256)
            
            for x_batch, y_batch in data_loader:
                pred = model(x_batch)
                loss = loss_func(pred, y_batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
        # Evaluate on test data
        model.eval()
        with torch.no_grad():
            pred = model(test_data['X']).view(-1).numpy()
        
        # Calculate performance
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
    
    return results

def train_and_evaluate_lambdamart(monthly_data, feature_cols, train_window=6):
    """
    Train and evaluate LambdaMART model
    """
    print("\nTraining LambdaMART models...")
    
    results = []
    for i in tqdm(range(len(monthly_data) - train_window - 1)):
        # Prepare training and test data
        train_data = monthly_data[i:i+train_window]
        test_data = monthly_data[i+train_window]
        
        # Prepare training data for XGBoost
        train_X = np.vstack([data['X'].numpy() for data in train_data])
        train_y = np.hstack([np.exp(data['y'].numpy()) for data in train_data])  # Ensure positive scores
        
        # Initialize and train model
        model = xgb.XGBRanker(
            booster='gbtree',
            objective='rank:pairwise',
            eval_metric='ndcg@100',
            random_state=42,
            learning_rate=0.1,
            max_depth=6,
            n_estimators=50,  # Reduced for demo
            subsample=0.8
        )
        
        model.fit(train_X, train_y, group=np.array([len(train_y)]), verbose=False)
        
        # Evaluate on test data
        pred = model.predict(test_data['X'].numpy())
        
        # Calculate performance
        sorted_indices = np.argsort(pred)[::-1]
        sorted_returns = test_data['y'].numpy()[sorted_indices]
        
        top_100_return = sorted_returns[:100].mean()
        bottom_100_return = sorted_returns[-100:].mean()
        long_short_return = top_100_return - bottom_100_return
        
        result = {
            'date': test_data['date'],
            'algorithm': 'LambdaMART',
            'top_100_return': top_100_return,
            'bottom_100_return': bottom_100_return,
            'long_short_return': long_short_return
        }
        results.append(result)
    
    return results

def main():
    """
    Main demonstration function
    """
    print("Learn2Rank Demo with Sample Data")
    print("=" * 50)
    
    # Generate sample data
    df = generate_sample_factor_data()
    
    # Save sample data
    df.to_csv('sample_factor_data.csv', index=False)
    print("Sample data saved to 'sample_factor_data.csv'")
    
    # Prepare ranking data
    monthly_data, feature_cols = prepare_ranking_data(df)
    
    print(f"\nNumber of features: {len(feature_cols)}")
    print(f"Number of time periods: {len(monthly_data)}")
    
    # Train and evaluate all models
    ranknet_results = train_and_evaluate_ranknet(monthly_data, feature_cols)
    listmle_results = train_and_evaluate_listmle(monthly_data, feature_cols)
    lambdamart_results = train_and_evaluate_lambdamart(monthly_data, feature_cols)
    
    # Combine results
    all_results = ranknet_results + listmle_results + lambdamart_results
    results_df = pd.DataFrame(all_results)
    
    # Save results
    results_df.to_csv('demo_results.csv', index=False)
    
    # Print summary
    print("\n" + "="*60)
    print("LEARN2RANK DEMO RESULTS SUMMARY")
    print("="*60)
    
    for algo in ['RankNet', 'ListMLE', 'LambdaMART']:
        algo_results = results_df[results_df['algorithm'] == algo]
        if len(algo_results) > 0:
            avg_long_short = algo_results['long_short_return'].mean()
            std_long_short = algo_results['long_short_return'].std()
            win_rate = (algo_results['long_short_return'] > 0).mean()
            
            print(f"\n{algo}:")
            print(f"  Average Long-Short Return: {avg_long_short:.4f}")
            print(f"  Standard Deviation: {std_long_short:.4f}")
            print(f"  Win Rate: {win_rate:.2%}")
    
    print(f"\nDetailed results saved to 'demo_results.csv'")
    print("Sample factor data saved to 'sample_factor_data.csv'")
    
    print("\n" + "="*60)
    print("NEXT STEPS FOR DATABASE INTEGRATION")
    print("="*60)
    print("1. Start MySQL service as administrator:")
    print("   - Open Command Prompt as Administrator")
    print("   - Run: net start MySQL57")
    print("2. Verify database connection:")
    print("   - Run: python database_connector.py")
    print("3. Train with real data:")
    print("   - Run: python train_learn2rank_models.py")

if __name__ == "__main__":
    main()
