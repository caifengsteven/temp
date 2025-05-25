import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import datetime
from tqdm import tqdm
import random
from collections import defaultdict
import math

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
random.seed(42)

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ====================== DATA GENERATION ======================

def generate_synthetic_fundamental_data(n_companies=300, n_years=30, start_year=1990):
    """
    Simplified data generation
    """
    print("Generating synthetic fundamental data...")
    
    # Create dates (monthly)
    dates = pd.date_range(start=f'{start_year}-01-01', periods=n_years*12, freq='M')
    
    # Company IDs
    company_ids = [f'COMP_{i:04d}' for i in range(n_companies)]
    
    # Industry sectors
    industries = [f'IND_{i:02d}' for i in range(10)]
    
    # List to store all data
    data = []
    
    # Generate data for each company
    for company_id in tqdm(company_ids):
        # Assign random industry
        industry = np.random.choice(industries)
        
        # Initial values
        revenue = np.random.uniform(500, 10000)
        ebit_margin = np.random.uniform(0.05, 0.3)
        ebit = revenue * ebit_margin
        assets = revenue * np.random.uniform(0.8, 2.5)
        equity = assets * np.random.uniform(0.3, 0.7)
        market_cap = equity * np.random.uniform(1.0, 3.0)
        enterprise_value = market_cap + (assets - equity) * 0.8
        price = market_cap / (np.random.uniform(5, 20) * 1e6)  # Assume shares outstanding
        
        # Generate time series
        for t, date in enumerate(dates):
            # Add growth and randomness
            growth = np.random.normal(0.04, 0.02) / 12  # Monthly growth
            noise = np.random.normal(0, 0.05)
            
            # Update values
            revenue *= (1 + growth + noise)
            ebit_margin += np.random.normal(0, 0.01)
            ebit_margin = max(0.01, min(0.4, ebit_margin))  # Keep margin within bounds
            
            ebit = revenue * ebit_margin
            assets *= (1 + growth * 0.8 + np.random.normal(0, 0.02))
            equity = assets * (equity / assets + np.random.normal(0, 0.01))
            equity = max(assets * 0.1, min(assets * 0.9, equity))  # Keep equity within bounds
            
            market_cap *= (1 + growth * 1.2 + np.random.normal(0, 0.08))
            enterprise_value = market_cap + (assets - equity) * 0.8
            price = market_cap / (np.random.uniform(5, 20) * 1e6)
            
            # Generate momentum features
            if t >= 1:
                mom_1m = price / prev_price - 1
            else:
                mom_1m = 0
                
            if t >= 3:
                mom_3m = price / price_3m - 1
            else:
                mom_3m = 0
            
            # Store previous prices
            prev_price = price
            if t == 0:
                price_3m = price
            if t % 3 == 0:
                price_3m = price
            
            # Append data
            data.append({
                'date': date,
                'company_id': company_id,
                'industry': industry,
                'revenue': revenue,
                'ebit': ebit,
                'assets': assets,
                'equity': equity,
                'market_cap': market_cap,
                'enterprise_value': enterprise_value,
                'mom_1m': mom_1m,
                'mom_3m': mom_3m,
                'price': price,
            })
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    df = df.sort_values(['date', 'company_id']).reset_index(drop=True)
    print(f"Generated {len(df)} rows of synthetic data.")
    return df

# ====================== DATA PREPROCESSING ======================

def preprocess_data(df, lookback_years=5, forecast_months=12, is_train=True):
    """
    Simplified data preprocessing
    """
    print("Preprocessing data...")
    
    # Set the lookback and forecast periods
    lookback_periods = lookback_years * 12
    
    # Convert date to datetime if it's not already
    if not pd.api.types.is_datetime64_any_dtype(df['date']):
        df['date'] = pd.to_datetime(df['date'])
    
    # Sort data
    df = df.sort_values(['company_id', 'date']).reset_index(drop=True)
    
    # Get companies and dates
    companies = df['company_id'].unique()
    
    # Create data dictionary
    data_dict = {}
    for _, row in tqdm(df.iterrows(), total=len(df)):
        company = row['company_id']
        date = row['date']
        if company not in data_dict:
            data_dict[company] = {}
        data_dict[company][date] = row.to_dict()
    
    # Features to use as inputs
    features = ['revenue', 'ebit', 'assets', 'equity', 'market_cap', 'mom_1m', 'mom_3m']
    
    # Target feature
    target_feature = 'ebit'
    
    # Prepare data
    X_data = []
    y_data = []
    metadata = []
    
    # Process each company
    for company in tqdm(companies):
        company_data = data_dict[company]
        company_dates = sorted(company_data.keys())
        
        # Skip if not enough data
        if len(company_dates) < lookback_periods + forecast_months:
            continue
        
        # Create samples
        for i in range(len(company_dates) - lookback_periods - forecast_months + 1):
            # Input dates (monthly)
            input_dates = [company_dates[i + j] for j in range(lookback_periods)]
            
            # Target date
            target_date = company_dates[i + lookback_periods + forecast_months - 1]
            
            # Prepare inputs
            inputs = []
            for date in input_dates:
                # Extract features
                feature_values = [company_data[date][feature] for feature in features]
                inputs.append(feature_values)
            
            # Target value
            target_value = company_data[target_date][target_feature]
            
            # Store data
            X_data.append(inputs)
            y_data.append([target_value])
            
            # For test data, store metadata
            if not is_train:
                metadata.append({
                    'company_id': company,
                    'date': target_date,
                    'market_cap': company_data[target_date]['market_cap'],
                    'enterprise_value': company_data[target_date]['enterprise_value'],
                    'ebit': company_data[target_date]['ebit'],
                    'price': company_data[target_date]['price'],
                })
    
    # Convert to arrays
    X = np.array(X_data)
    y = np.array(y_data)
    
    print(f"X shape: {X.shape}, y shape: {y.shape}")
    
    if is_train:
        return X, y
    else:
        return X, y, metadata

# ====================== NEURAL NETWORK MODELS ======================

class SimpleRNN(nn.Module):
    """
    Simple RNN model for EBIT prediction
    """
    def __init__(self, input_size, hidden_size=64):
        super(SimpleRNN, self).__init__()
        
        # RNN layer
        self.rnn = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True
        )
        
        # Output layer
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        # RNN forward pass
        out, _ = self.rnn(x)
        
        # Use the last output
        out = self.fc(out[:, -1, :])
        
        return out

class SimpleDataset(Dataset):
    """
    Simple PyTorch dataset
    """
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
            
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def train_simple_model(model, train_loader, val_loader, n_epochs=20, lr=0.001):
    """
    Simple training function
    """
    print("Training model...")
    model = model.to(device)
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Loss function
    criterion = nn.MSELoss()
    
    # For early stopping
    best_val_loss = float('inf')
    best_model = None
    patience = 5
    patience_counter = 0
    
    for epoch in range(n_epochs):
        # Training
        model.train()
        train_loss = 0.0
        
        for X_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epochs}"):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * len(X_batch)
        
        train_loss /= len(train_loader.dataset)
        
        # Validation
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                
                # Forward pass
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                
                val_loss += loss.item() * len(X_batch)
        
        val_loss /= len(val_loader.dataset)
        
        # Print metrics
        print(f"Epoch {epoch+1}/{n_epochs}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    # Load best model
    if best_model is not None:
        model.load_state_dict(best_model)
    
    return model

def evaluate_model(model, test_loader):
    """
    Evaluate model
    """
    model.eval()
    predictions = []
    targets = []
    
    with torch.no_grad():
        for X_batch, y_batch in tqdm(test_loader, desc="Evaluating"):
            X_batch = X_batch.to(device)
            
            # Forward pass
            outputs = model(X_batch)
            
            # Store predictions and targets
            predictions.append(outputs.cpu().numpy())
            targets.append(y_batch.numpy())
    
    # Concatenate results
    predictions = np.concatenate(predictions, axis=0)
    targets = np.concatenate(targets, axis=0)
    
    # Calculate MSE
    mse = np.mean((predictions - targets) ** 2)
    print(f"Test MSE: {mse:.6f}")
    
    return predictions, targets

# ====================== PORTFOLIO CONSTRUCTION ======================

def construct_portfolios(test_predictions, metadata, top_n=50):
    """
    Construct standard and predictive portfolios
    """
    print("Constructing portfolios...")
    
    # Sort by date
    date_sorted_data = defaultdict(list)
    for i, meta in enumerate(metadata):
        date = meta['date']
        date_sorted_data[date].append((i, meta))
    
    # Standard portfolio (EBIT/EV)
    standard_portfolios = {}
    
    # Predictive portfolio (predicted EBIT/EV)
    predictive_portfolios = {}
    
    # Create portfolios for each date
    for date, data_list in date_sorted_data.items():
        # Extract indices and metadata
        indices = [item[0] for item in data_list]
        meta_list = [item[1] for item in data_list]
        
        # Standard factor values
        standard_factors = []
        predictive_factors = []
        
        for i, meta in zip(indices, meta_list):
            try:
                # Skip if enterprise value is too small
                if meta['enterprise_value'] <= 1e-6:
                    continue
                
                # Standard factor (current EBIT/EV)
                standard_factor = meta['ebit'] / meta['enterprise_value']
                
                # Predictive factor (predicted EBIT/EV)
                predicted_ebit = test_predictions[i][0]
                predictive_factor = predicted_ebit / meta['enterprise_value']
                
                # Add to lists if valid
                if np.isfinite(standard_factor):
                    standard_factors.append((meta['company_id'], standard_factor))
                    
                if np.isfinite(predictive_factor):
                    predictive_factors.append((meta['company_id'], predictive_factor))
                    
            except Exception as e:
                print(f"Error calculating factor for {meta['company_id']} on {date}: {e}")
                continue
        
        # Sort and select top stocks
        if standard_factors:
            standard_factors.sort(key=lambda x: x[1], reverse=True)
            standard_portfolios[date] = [stock for stock, _ in standard_factors[:top_n]]
        else:
            standard_portfolios[date] = []
            
        if predictive_factors:
            predictive_factors.sort(key=lambda x: x[1], reverse=True)
            predictive_portfolios[date] = [stock for stock, _ in predictive_factors[:top_n]]
        else:
            predictive_portfolios[date] = []
    
    return standard_portfolios, predictive_portfolios

def simulate_performance(portfolios, data_dict, start_date, end_date):
    """
    Simulate portfolio performance
    """
    print("Simulating portfolio performance...")
    
    # Get dates in the simulation period
    all_dates = sorted([d for d in portfolios.keys() if start_date <= d <= end_date])
    
    if not all_dates:
        print("No dates in simulation period")
        return None
    
    # Initialize portfolio
    portfolio = {}
    cash = 1000000  # $1M initial cash
    
    # Performance tracking
    portfolio_values = []
    returns = []
    
    # Process each month
    for i, date in enumerate(all_dates):
        try:
            # Get portfolio constituents
            current_portfolio = portfolios[date]
            
            # Calculate current value
            current_value = cash
            for stock_id, shares in list(portfolio.items()):
                if stock_id in data_dict and date in data_dict[stock_id]:
                    price = data_dict[stock_id][date]['price']
                    current_value += shares * price
                else:
                    # Liquidate at zero if no price
                    portfolio[stock_id] = 0
            
            # Ensure positive value
            current_value = max(current_value, 1.0)
            
            # Store value
            portfolio_values.append(current_value)
            
            # Calculate return
            if i > 0 and portfolio_values[-2] > 0:
                monthly_return = current_value / portfolio_values[-2] - 1
                returns.append(monthly_return)
            elif i > 0:
                returns.append(0.0)
            
            # Rebalance portfolio
            if current_portfolio:
                # Sell all current holdings
                for stock_id, shares in list(portfolio.items()):
                    if shares > 0:
                        if stock_id in data_dict and date in data_dict[stock_id]:
                            price = data_dict[stock_id][date]['price']
                            cash += shares * price * 0.99  # 1% transaction cost
                        portfolio[stock_id] = 0
                
                # Buy new stocks
                num_stocks = len(current_portfolio)
                if num_stocks > 0:
                    value_per_stock = current_value / num_stocks
                    
                    for stock_id in current_portfolio:
                        if stock_id in data_dict and date in data_dict[stock_id]:
                            price = data_dict[stock_id][date]['price']
                            
                            # Skip if price is invalid
                            if price <= 0:
                                continue
                                
                            # Buy shares
                            shares_to_buy = (value_per_stock / price) * 0.99  # 1% transaction cost
                            
                            # Skip if too few shares or not enough cash
                            if shares_to_buy < 0.01 or shares_to_buy * price > cash:
                                continue
                                
                            # Update portfolio
                            cash -= shares_to_buy * price
                            portfolio[stock_id] = shares_to_buy
        except Exception as e:
            print(f"Error in simulation at date {date}: {e}")
            # Continue with previous value
            if portfolio_values:
                portfolio_values.append(portfolio_values[-1])
            else:
                portfolio_values.append(cash)
            if i > 0:
                returns.append(0.0)
    
    # Calculate final value
    try:
        final_date = all_dates[-1]
        final_value = cash
        for stock_id, shares in portfolio.items():
            if stock_id in data_dict and final_date in data_dict[stock_id]:
                price = data_dict[stock_id][final_date]['price']
                final_value += shares * price
        
        # Ensure positive
        final_value = max(final_value, 1.0)
    except Exception as e:
        print(f"Error calculating final value: {e}")
        final_value = portfolio_values[-1] if portfolio_values else cash
    
    # Calculate metrics
    returns = np.array(returns)
    portfolio_values = np.array(portfolio_values)
    
    # Compound annual return
    try:
        total_return = final_value / portfolio_values[0] - 1
        num_years = max(0.1, len(all_dates) / 12)
        car = (1 + total_return) ** (1 / num_years) - 1
    except Exception:
        car = 0.0
    
    # Monthly statistics
    try:
        avg_monthly_return = np.mean(returns) if len(returns) > 0 else 0.0
        std_monthly_return = np.std(returns) if len(returns) > 1 else 0.01
    except Exception:
        avg_monthly_return = 0.0
        std_monthly_return = 0.01
    
    # Sharpe ratio
    try:
        risk_free_rate = 0.02
        sharpe_ratio = (car - risk_free_rate) / (std_monthly_return * np.sqrt(12))
    except Exception:
        sharpe_ratio = 0.0
    
    # Maximum drawdown
    try:
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = np.zeros_like(portfolio_values)
        for i in range(len(portfolio_values)):
            if peak[i] > 0:
                drawdown[i] = (peak[i] - portfolio_values[i]) / peak[i]
        max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0.0
    except Exception:
        max_drawdown = 0.0
    
    # Results
    results = {
        'dates': all_dates,
        'portfolio_values': portfolio_values,
        'returns': returns,
        'car': car,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown
    }
    
    print(f"CAR: {car:.4f}")
    print(f"Sharpe Ratio: {sharpe_ratio:.4f}")
    print(f"Max Drawdown: {max_drawdown:.4f}")
    
    return results

# ====================== VISUALIZATION ======================

def plot_results(standard_results, predictive_results):
    """
    Plot performance comparison
    """
    plt.figure(figsize=(15, 10))
    
    # Cumulative returns
    plt.subplot(2, 2, 1)
    plt.plot(standard_results['dates'], 
             standard_results['portfolio_values'] / standard_results['portfolio_values'][0], 
             label=f"Standard (CAR: {standard_results['car']:.2%})")
    plt.plot(predictive_results['dates'], 
             predictive_results['portfolio_values'] / predictive_results['portfolio_values'][0], 
             label=f"Predictive (CAR: {predictive_results['car']:.2%})")
    plt.title('Cumulative Returns')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Monthly returns
    plt.subplot(2, 2, 2)
    plt.plot(standard_results['dates'][1:], standard_results['returns'], label="Standard")
    plt.plot(predictive_results['dates'][1:], predictive_results['returns'], label="Predictive")
    plt.title('Monthly Returns')
    plt.xlabel('Date')
    plt.ylabel('Monthly Return')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Drawdowns
    plt.subplot(2, 2, 3)
    peak_std = np.maximum.accumulate(standard_results['portfolio_values'])
    drawdown_std = (peak_std - standard_results['portfolio_values']) / peak_std
    
    peak_pred = np.maximum.accumulate(predictive_results['portfolio_values'])
    drawdown_pred = (peak_pred - predictive_results['portfolio_values']) / peak_pred
    
    plt.plot(standard_results['dates'], drawdown_std, 
             label=f"Standard (Max DD: {standard_results['max_drawdown']:.2%})")
    plt.plot(predictive_results['dates'], drawdown_pred, 
             label=f"Predictive (Max DD: {predictive_results['max_drawdown']:.2%})")
    
    plt.title('Drawdowns')
    plt.xlabel('Date')
    plt.ylabel('Drawdown')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Metrics comparison
    plt.subplot(2, 2, 4)
    metrics = ['car', 'sharpe_ratio']
    standard_values = [standard_results[m] for m in metrics]
    predictive_values = [predictive_results[m] for m in metrics]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    plt.bar(x - width/2, standard_values, width, label='Standard')
    plt.bar(x + width/2, predictive_values, width, label='Predictive')
    
    plt.xticks(x, ['CAR', 'Sharpe Ratio'])
    plt.title('Performance Metrics')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def plot_predictions(predictions, targets, n_samples=10):
    """
    Plot prediction vs actual
    """
    # Randomly select samples
    indices = np.random.choice(len(predictions), min(n_samples, len(predictions)), replace=False)
    
    plt.figure(figsize=(10, 6))
    plt.scatter(targets[indices], predictions[indices], alpha=0.7)
    
    # Add perfect prediction line
    min_val = min(np.min(targets[indices]), np.min(predictions[indices]))
    max_val = max(np.max(targets[indices]), np.max(predictions[indices]))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    plt.title('Predicted vs Actual EBIT')
    plt.xlabel('Actual EBIT')
    plt.ylabel('Predicted EBIT')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# ====================== MAIN FUNCTION ======================

def main():
    """
    Main function with simplified approach
    """
    # Generate synthetic data
    df = generate_synthetic_fundamental_data(n_companies=300, n_years=30, start_year=1990)
    
    # Split data
    train_df = df[df['date'] < pd.Timestamp('2000-01-01')]
    test_df = df[df['date'] >= pd.Timestamp('2000-01-01')]
    
    # Preprocess data
    X_train, y_train = preprocess_data(train_df, lookback_years=5, forecast_months=12, is_train=True)
    X_test, y_test, test_metadata = preprocess_data(test_df, lookback_years=5, forecast_months=12, is_train=False)
    
    # Scale data
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    # Reshape for scaling
    X_train_flat = X_train.reshape(-1, X_train.shape[-1])
    X_train_flat = scaler_X.fit_transform(X_train_flat)
    X_train = X_train_flat.reshape(X_train.shape)
    
    y_train_scaled = scaler_y.fit_transform(y_train)
    
    # Scale test data
    X_test_flat = X_test.reshape(-1, X_test.shape[-1])
    X_test_flat = scaler_X.transform(X_test_flat)
    X_test = X_test_flat.reshape(X_test.shape)
    
    y_test_scaled = scaler_y.transform(y_test)
    
    # Split training data
    X_train, X_val, y_train_scaled, y_val_scaled = train_test_split(
        X_train, y_train_scaled, test_size=0.2, random_state=42
    )
    
    # Create datasets
    train_dataset = SimpleDataset(X_train, y_train_scaled)
    val_dataset = SimpleDataset(X_val, y_val_scaled)
    test_dataset = SimpleDataset(X_test, y_test_scaled)
    
    # Create data loaders
    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Create and train model
    input_size = X_train.shape[2]
    model = SimpleRNN(input_size=input_size, hidden_size=64)
    model = train_simple_model(model, train_loader, val_loader, n_epochs=20, lr=0.001)
    
    # Evaluate model
    predictions_scaled, targets_scaled = evaluate_model(model, test_loader)
    
    # Inverse transform predictions and targets
    predictions = scaler_y.inverse_transform(predictions_scaled)
    targets = scaler_y.inverse_transform(targets_scaled)
    
    # Plot predictions
    plot_predictions(predictions, targets)
    
    # Create portfolios
    standard_portfolios, predictive_portfolios = construct_portfolios(predictions, test_metadata)
    
    # Create company-date dictionary
    data_dict = {}
    for _, row in df.iterrows():
        company = row['company_id']
        date = row['date']
        if company not in data_dict:
            data_dict[company] = {}
        data_dict[company][date] = row.to_dict()
    
    # Simulate performance
    start_date = pd.Timestamp('2000-01-01')
    end_date = pd.Timestamp('2019-12-31')
    
    standard_results = simulate_performance(standard_portfolios, data_dict, start_date, end_date)
    predictive_results = simulate_performance(predictive_portfolios, data_dict, start_date, end_date)
    
    # Plot results
    plot_results(standard_results, predictive_results)

if __name__ == "__main__":
    main()