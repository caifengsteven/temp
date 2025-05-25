import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.nn.utils import weight_norm
import time

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

#######################################################
# 1. Data Generation - Simulate Financial Time Series
#######################################################

def generate_simulated_financial_data(n_companies=100, n_quarters=40, n_daily=60):
    """
    Generate simulated financial data:
    - Quarterly accounting data (e.g., EPS, total assets, revenue)
    - Daily stock returns
    
    Parameters:
    -----------
    n_companies: int
        Number of companies to simulate
    n_quarters: int
        Number of quarterly reports to generate
    n_daily: int
        Number of daily stock returns per quarter
        
    Returns:
    --------
    quarterly_data: DataFrame
        Simulated quarterly financial data
    daily_data: DataFrame
        Simulated daily stock return data
    """
    # Company characteristics - some companies are more volatile than others
    company_volatility = np.random.uniform(0.05, 0.3, n_companies)
    company_growth = np.random.uniform(-0.02, 0.05, n_companies)
    company_size = np.random.uniform(1e8, 1e10, n_companies)
    company_seasonality = np.random.uniform(0.01, 0.05, n_companies)
    
    # Financial industry indicator (about 20% financial firms)
    is_financial = np.random.choice([0, 1], n_companies, p=[0.8, 0.2])
    
    quarterly_data = []
    daily_data = []
    
    for company_id in range(n_companies):
        # Base for simulating quarterly data
        base_eps = np.random.uniform(0.5, 5.0)
        base_assets = company_size[company_id]
        base_revenue = base_assets * np.random.uniform(0.1, 0.3)
        
        # Trend, seasonality and noise components
        trend = np.linspace(0, n_quarters * company_growth[company_id], n_quarters)
        seasonality = company_seasonality[company_id] * np.sin(np.linspace(0, 4*np.pi, n_quarters))
        noise = np.random.normal(0, company_volatility[company_id], n_quarters)
        
        # Generate quarterly EPS with trend, seasonality, and noise
        eps = base_eps + trend + seasonality + noise
        
        # Additional noise for financial firms
        if is_financial[company_id]:
            eps += np.random.normal(0, 0.15, n_quarters)
        
        # Generate other financial metrics
        assets = base_assets * (1 + 0.01 * trend + 0.01 * noise)
        revenue = base_revenue * (1 + 0.02 * trend + 0.02 * seasonality + 0.01 * noise)
        costs = revenue * np.random.uniform(0.65, 0.85, n_quarters)
        net_income = revenue - costs
        
        # Create quarterly dataframe
        for q in range(n_quarters):
            # Add more variability to financial statement items
            quarterly_data.append({
                'company_id': company_id,
                'quarter': q,
                'eps': eps[q],
                'atq': assets[q],  # Total assets
                'revtq': revenue[q],  # Revenue
                'cogsq': costs[q],  # Cost of goods sold
                'niq': net_income[q],  # Net income
                'oiadpq': net_income[q] * 1.2,  # Operating income after depreciation
                'is_financial': is_financial[company_id],
                # Simulate some accounting metrics with relationships to earnings
                'xrdq': revenue[q] * np.random.uniform(0.05, 0.15),  # R&D expense
                'dpq': assets[q] * np.random.uniform(0.01, 0.02),    # Depreciation
                'ppentq': assets[q] * np.random.uniform(0.3, 0.5),   # Property, plant & equipment
                'txtq': net_income[q] * np.random.uniform(0.2, 0.35) # Tax expense
            })
            
            # Generate daily stock returns related to the financial health
            daily_returns_base = np.random.normal(
                0.0001 * (eps[q] / base_eps), 
                0.01 * company_volatility[company_id], 
                n_daily
            )
            
            # Add some autocorrelation
            for i in range(1, n_daily):
                daily_returns_base[i] += 0.1 * daily_returns_base[i-1]
            
            # Calculate price from returns
            daily_price = 100 * np.exp(np.cumsum(daily_returns_base))
            daily_volume = np.random.lognormal(10, 1, n_daily) * (assets[q] / 1e9)
            
            for d in range(n_daily):
                daily_data.append({
                    'company_id': company_id,
                    'quarter': q,
                    'day': d,
                    'ret': daily_returns_base[d],
                    'prc': daily_price[d],
                    'vol': daily_volume[d],
                    'vwretd': daily_returns_base[d] + np.random.normal(0.0001, 0.001)  # Market return
                })
    
    # Convert to dataframes
    quarterly_df = pd.DataFrame(quarterly_data)
    daily_df = pd.DataFrame(daily_data)
    
    # Add analyst forecasts with some bias and error
    quarterly_df['eps_forecast'] = quarterly_df['eps'].shift(1) * (1 + np.random.normal(0.01, 0.05, len(quarterly_df)))
    
    return quarterly_df, daily_df

#######################################################
# 2. Data Preprocessing
#######################################################

def preprocess_data(quarterly_df, daily_df, window_size=20, prediction_horizon=1):
    """
    Preprocess the data according to the paper:
    - Normalize using total assets
    - Studentize (standardize) the data
    - Create sliding windows
    - Merge quarterly and daily data
    
    Parameters:
    -----------
    quarterly_df: DataFrame
        Quarterly financial data
    daily_df: DataFrame
        Daily stock return data
    window_size: int
        Number of past quarters to use for prediction
    prediction_horizon: int
        Number of quarters ahead to predict
        
    Returns:
    --------
    X_quarterly: array
        Preprocessed quarterly data for input to the model
    X_daily: array
        Preprocessed daily data for input to the model
    y: array
        Target EPS values to predict
    companies: array
        Company IDs for each sample
    quarters: array
        Quarter indices for each sample
    is_financial: array
        Financial industry indicator for each sample
    studentized_cols: list
        List of studentized column names
    daily_studentized_cols: list
        List of studentized daily column names
    """
    # Normalize using total assets as per equation (2) in the paper
    normalize_cols = ['revtq', 'niq', 'oiadpq', 'cogsq', 'xrdq', 'dpq', 'ppentq', 'txtq']
    
    for col in normalize_cols:
        quarterly_df[f'{col}_norm'] = quarterly_df[col] / np.maximum(1, quarterly_df['atq'])
    
    # Studentize (standardize) the normalized variables as per equation (3)
    studentize_cols = normalize_cols + ['eps', 'atq']
    studentized_cols = []
    
    for col in studentize_cols:
        col_name = f'{col}_norm' if col in normalize_cols else col
        new_col = f'{col}_std'
        studentized_cols.append(new_col)
        
        quarterly_df[new_col] = (quarterly_df[col_name] - quarterly_df[col_name].mean()) / quarterly_df[col_name].std()
    
    # Add eps_forecast to studentized columns
    quarterly_df['eps_forecast_std'] = (quarterly_df['eps_forecast'] - quarterly_df['eps_forecast'].mean()) / quarterly_df['eps_forecast'].std()
    studentized_cols.append('eps_forecast_std')
    
    # Studentize daily data
    daily_studentized_cols = []
    for col in ['ret', 'prc', 'vol', 'vwretd']:
        new_col = f'{col}_std'
        daily_studentized_cols.append(new_col)
        daily_df[new_col] = (daily_df[col] - daily_df[col].mean()) / daily_df[col].std()
    
    # Create samples with sliding windows
    samples = []
    company_ids = np.unique(quarterly_df['company_id'])
    
    for company_id in company_ids:
        company_data = quarterly_df[quarterly_df['company_id'] == company_id].sort_values('quarter')
        
        # If we have enough data for a window and prediction
        if len(company_data) >= window_size + prediction_horizon:
            for i in range(len(company_data) - window_size - prediction_horizon + 1):
                X_quarterly_window = company_data.iloc[i:i+window_size][studentized_cols].values
                y_value = company_data.iloc[i+window_size+prediction_horizon-1]['eps']
                
                # Get corresponding daily data and reshape
                daily_windows = []
                for q in range(i, i+window_size):
                    quarter_daily = daily_df[
                        (daily_df['company_id'] == company_id) & 
                        (daily_df['quarter'] == q)
                    ][daily_studentized_cols].values
                    
                    # Ensure consistent shape by taking the last n_daily entries if needed
                    n_daily = 60  # Number of days to keep per quarter
                    if len(quarter_daily) > n_daily:
                        quarter_daily = quarter_daily[-n_daily:]
                    elif len(quarter_daily) < n_daily:
                        # Pad with zeros if not enough days
                        padding = np.zeros((n_daily - len(quarter_daily), len(daily_studentized_cols)))
                        quarter_daily = np.vstack([padding, quarter_daily])
                    
                    daily_windows.append(quarter_daily)
                
                X_daily_window = np.array(daily_windows)
                
                # Flatten the daily data as in the paper's architecture
                X_daily_flat = X_daily_window.reshape(-1)
                
                # Record the sample
                samples.append({
                    'X_quarterly': X_quarterly_window,
                    'X_daily': X_daily_flat,
                    'y': y_value,
                    'company_id': company_id,
                    'quarter': i + window_size,
                    'is_financial': company_data.iloc[i]['is_financial']
                })
    
    # Convert to arrays
    X_quarterly = np.array([s['X_quarterly'] for s in samples])
    X_daily = np.array([s['X_daily'] for s in samples])
    y = np.array([s['y'] for s in samples])
    companies = np.array([s['company_id'] for s in samples])
    quarters = np.array([s['quarter'] for s in samples])
    is_financial = np.array([s['is_financial'] for s in samples])
    
    return X_quarterly, X_daily, y, companies, quarters, is_financial, studentized_cols, daily_studentized_cols

#######################################################
# 3. Model Building - Implement LSTM and TCN with PyTorch
#######################################################

class LSTMModel(nn.Module):
    """
    LSTM model architecture as described in the paper
    """
    def __init__(self, input_dim_quarterly, input_dim_daily):
        super(LSTMModel, self).__init__()
        
        # Quarterly data branch
        self.lstm1 = nn.LSTM(input_dim_quarterly, 76, batch_first=True, dropout=0.3)
        self.lstm2 = nn.LSTM(76, 38, batch_first=True, dropout=0.3)
        
        # Daily data branch
        self.daily_fc1 = nn.Linear(input_dim_daily, 660)
        self.daily_dropout1 = nn.Dropout(0.3)
        self.daily_fc2 = nn.Linear(660, 440)
        self.daily_dropout2 = nn.Dropout(0.3)
        self.daily_fc3 = nn.Linear(440, 220)
        self.daily_dropout3 = nn.Dropout(0.3)
        
        # Output layers after merging
        self.fc1 = nn.Linear(38 + 220, 19)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(19, 8)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(8, 1)
        
        # Activation function
        self.tanh = nn.Tanh()
    
    def forward(self, quarterly_data, daily_data):
        # Process quarterly data
        lstm_out, _ = self.lstm1(quarterly_data)
        lstm_out, _ = self.lstm2(lstm_out)
        lstm_out = lstm_out[:, -1, :]  # Take the output from the last time step
        
        # Process daily data
        daily_out = self.tanh(self.daily_fc1(daily_data))
        daily_out = self.daily_dropout1(daily_out)
        daily_out = self.tanh(self.daily_fc2(daily_out))
        daily_out = self.daily_dropout2(daily_out)
        daily_out = self.tanh(self.daily_fc3(daily_out))
        daily_out = self.daily_dropout3(daily_out)
        
        # Concatenate the outputs
        combined = torch.cat((lstm_out, daily_out), dim=1)
        
        # Final layers
        out = self.tanh(self.fc1(combined))
        out = self.dropout1(out)
        out = self.tanh(self.fc2(out))
        out = self.dropout2(out)
        out = self.fc3(out)
        
        return out

class CausalConv1d(nn.Module):
    """
    1D causal convolution with proper padding
    """
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super(CausalConv1d, self).__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, 
                              padding=self.padding, dilation=dilation)
    
    def forward(self, x):
        # Apply convolution with padding
        out = self.conv(x)
        # Remove padding at the end
        return out[:, :, :-self.padding]

class TCNBlock(nn.Module):
    """
    TCN block with dilated causal convolutions
    """
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super(TCNBlock, self).__init__()
        self.conv = CausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x):
        out = self.conv(x)
        out = self.relu(out)
        out = self.dropout(out)
        return out

class TCNModel(nn.Module):
    """
    TCN model architecture as described in the paper
    """
    def __init__(self, input_dim_quarterly, input_dim_daily):
        super(TCNModel, self).__init__()
        
        # TCN layers for quarterly data
        self.tcn_blocks = nn.ModuleList([
            TCNBlock(input_dim_quarterly, 32, kernel_size=3, dilation=1),
            TCNBlock(32, 32, kernel_size=3, dilation=2),
            TCNBlock(32, 32, kernel_size=3, dilation=4)
        ])
        
        # Daily data branch
        self.daily_fc1 = nn.Linear(input_dim_daily, 660)
        self.daily_dropout1 = nn.Dropout(0.3)
        self.daily_fc2 = nn.Linear(660, 440)
        self.daily_dropout2 = nn.Dropout(0.3)
        self.daily_fc3 = nn.Linear(440, 220)
        self.daily_dropout3 = nn.Dropout(0.3)
        
        # Determine flattened size of TCN output
        self.tcn_output_dim = 38
        
        # Output layers after merging
        self.fc1 = nn.Linear(self.tcn_output_dim + 220, 19)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(19, 8)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(8, 1)
        
        # Activation function
        self.tanh = nn.Tanh()
    
    def forward(self, quarterly_data, daily_data):
        # Process quarterly data with TCN
        # Input shape [batch, seq_len, channels] -> [batch, channels, seq_len] for Conv1d
        x = quarterly_data.permute(0, 2, 1)
        
        for block in self.tcn_blocks:
            x = block(x)
        
        # Flatten the output from last block
        x = x.flatten(start_dim=1)
        
        # Project to the desired dimension
        quarterly_out = self.tanh(nn.Linear(x.shape[1], self.tcn_output_dim).to(x.device)(x))
        
        # Process daily data
        daily_out = self.tanh(self.daily_fc1(daily_data))
        daily_out = self.daily_dropout1(daily_out)
        daily_out = self.tanh(self.daily_fc2(daily_out))
        daily_out = self.daily_dropout2(daily_out)
        daily_out = self.tanh(self.daily_fc3(daily_out))
        daily_out = self.daily_dropout3(daily_out)
        
        # Concatenate the outputs
        combined = torch.cat((quarterly_out, daily_out), dim=1)
        
        # Final layers
        out = self.tanh(self.fc1(combined))
        out = self.dropout1(out)
        out = self.tanh(self.fc2(out))
        out = self.dropout2(out)
        out = self.fc3(out)
        
        return out

#######################################################
# 4. Training and Evaluation Functions
#######################################################

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=100, patience=5, device=device):
    """
    Train the model with early stopping
    """
    # Move model to device
    model.to(device)
    
    # For early stopping
    best_val_loss = float('inf')
    counter = 0
    best_model_state = None
    
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        
        for quarterly_data, daily_data, targets in train_loader:
            # Move data to device
            quarterly_data = quarterly_data.to(device).float()
            daily_data = daily_data.to(device).float()
            targets = targets.to(device).float()
            
            # Zero the gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(quarterly_data, daily_data)
            
            # Compute loss
            loss = criterion(outputs, targets)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * quarterly_data.size(0)
        
        epoch_train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_train_loss)
        
        # Validation phase
        model.eval()
        running_val_loss = 0.0
        
        with torch.no_grad():
            for quarterly_data, daily_data, targets in val_loader:
                # Move data to device
                quarterly_data = quarterly_data.to(device).float()
                daily_data = daily_data.to(device).float()
                targets = targets.to(device).float()
                
                # Forward pass
                outputs = model(quarterly_data, daily_data)
                
                # Compute loss
                val_loss = criterion(outputs, targets)
                
                running_val_loss += val_loss.item() * quarterly_data.size(0)
            
            epoch_val_loss = running_val_loss / len(val_loader.dataset)
            val_losses.append(epoch_val_loss)
        
        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}/{num_epochs} - Train Loss: {epoch_train_loss:.4f} - Val Loss: {epoch_val_loss:.4f}')
        
        # Check for early stopping
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            counter = 0
            best_model_state = model.state_dict().copy()
        else:
            counter += 1
            if counter >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, {'train_loss': train_losses, 'val_loss': val_losses}

def calculate_skill_score(mse_model, mse_baseline):
    """
    Calculate the skill score as defined in the paper (equation 1)
    """
    return 1 - (mse_model / mse_baseline)

def evaluate_model(model, test_loader, device=device):
    """
    Evaluate the model on test data
    """
    model.eval()
    predictions = []
    targets = []
    
    with torch.no_grad():
        for quarterly_data, daily_data, y in test_loader:
            # Move data to device
            quarterly_data = quarterly_data.to(device).float()
            daily_data = daily_data.to(device).float()
            
            # Get predictions
            outputs = model(quarterly_data, daily_data)
            
            # Move back to CPU for numpy conversion
            predictions.append(outputs.cpu().numpy())
            targets.append(y.numpy())
    
    # Concatenate batches
    predictions = np.concatenate(predictions).flatten()
    targets = np.concatenate(targets).flatten()
    
    return predictions, targets

def evaluate_models_performance(X_quarterly, X_daily, y, companies, quarters, is_financial, studentized_cols, split_date=30):
    """
    Evaluate the LSTM and TCN models against baselines
    - Persistent model
    - Analyst forecasts (simulated)
    
    Parameters:
    -----------
    X_quarterly: array
        Preprocessed quarterly data
    X_daily: array
        Preprocessed daily data
    y: array
        Target EPS values
    companies: array
        Company IDs
    quarters: array
        Quarter indices
    is_financial: array
        Whether each sample is for a financial company
    studentized_cols: list
        List of studentized column names
    split_date: int
        Quarter index to split train/test data
    
    Returns:
    --------
    results: dict
        Dictionary of evaluation results
    """
    # Split data into train, validation, and test sets based on quarter
    train_idx = np.where(quarters < split_date - 2)[0]
    val_idx = np.where((quarters >= split_date - 2) & (quarters < split_date))[0]
    test_idx = np.where(quarters >= split_date)[0]
    
    # Split into financial and non-financial companies
    train_nofin_idx = np.where((quarters < split_date - 2) & (is_financial == 0))[0]
    val_nofin_idx = np.where((quarters >= split_date - 2) & (quarters < split_date) & (is_financial == 0))[0]
    test_nofin_idx = np.where((quarters >= split_date) & (is_financial == 0))[0]
    
    train_fin_idx = np.where((quarters < split_date - 2) & (is_financial == 1))[0]
    val_fin_idx = np.where((quarters >= split_date - 2) & (quarters < split_date) & (is_financial == 1))[0]
    test_fin_idx = np.where((quarters >= split_date) & (is_financial == 1))[0]
    
    results = {}
    
    # Define datasets for different company groups
    datasets = {
        'all': (train_idx, val_idx, test_idx),
        'nofin': (train_nofin_idx, val_nofin_idx, test_nofin_idx),
        'onlyfin': (train_fin_idx, val_fin_idx, test_fin_idx)
    }
    
    for dataset_name, (train_indices, val_indices, test_indices) in datasets.items():
        print(f"\nEvaluating on {dataset_name} companies")
        
        # Skip if we don't have enough data
        if len(train_indices) < 100 or len(test_indices) < 20:
            print(f"Skipping {dataset_name} due to insufficient data")
            continue
        
        # Training and test data
        X_quarterly_train, X_quarterly_val, X_quarterly_test = X_quarterly[train_indices], X_quarterly[val_indices], X_quarterly[test_indices]
        X_daily_train, X_daily_val, X_daily_test = X_daily[train_indices], X_daily[val_indices], X_daily[test_indices]
        y_train, y_val, y_test = y[train_indices], y[val_indices], y[test_indices]
        
        # Generate persistent baseline (naive prediction = previous quarter's value)
        # Assuming the last column of studentized_cols is eps_forecast_std
        eps_idx = studentized_cols.index('eps_std')
        y_persistent = X_quarterly_test[:, -1, eps_idx]
        
        # Create simulated analyst forecasts (previous EPS with some bias and error)
        eps_forecast_idx = studentized_cols.index('eps_forecast_std')
        y_analyst = X_quarterly_test[:, -1, eps_forecast_idx]
        
        # Calculate baseline MSEs
        mse_persistent = mean_squared_error(y_test, y_persistent)
        mse_analyst = mean_squared_error(y_test, y_analyst)
        
        print(f"Persistent model MSE: {mse_persistent:.4f}")
        print(f"Analyst forecast MSE: {mse_analyst:.4f}")
        
        # Create PyTorch datasets and dataloaders
        train_dataset = TensorDataset(torch.from_numpy(X_quarterly_train), torch.from_numpy(X_daily_train), 
                                     torch.from_numpy(y_train.reshape(-1, 1)))
        val_dataset = TensorDataset(torch.from_numpy(X_quarterly_val), torch.from_numpy(X_daily_val), 
                                   torch.from_numpy(y_val.reshape(-1, 1)))
        test_dataset = TensorDataset(torch.from_numpy(X_quarterly_test), torch.from_numpy(X_daily_test), 
                                    torch.from_numpy(y_test.reshape(-1, 1)))
        
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
        
        # Train and evaluate LSTM model
        print("Training LSTM model...")
        lstm_model = LSTMModel(X_quarterly_train.shape[2], X_daily_train.shape[1])
        criterion = nn.MSELoss()
        optimizer = optim.Adam(lstm_model.parameters(), lr=0.001)
        
        lstm_model, history_lstm = train_model(lstm_model, train_loader, val_loader, criterion, optimizer)
        
        y_pred_lstm, _ = evaluate_model(lstm_model, test_loader)
        mse_lstm = mean_squared_error(y_test, y_pred_lstm)
        print(f"LSTM model MSE: {mse_lstm:.4f}")
        
        # Train and evaluate TCN model
        print("Training TCN model...")
        tcn_model = TCNModel(X_quarterly_train.shape[2], X_daily_train.shape[1])
        optimizer = optim.Adam(tcn_model.parameters(), lr=0.001)
        
        tcn_model, history_tcn = train_model(tcn_model, train_loader, val_loader, criterion, optimizer)
        
        y_pred_tcn, _ = evaluate_model(tcn_model, test_loader)
        mse_tcn = mean_squared_error(y_test, y_pred_tcn)
        print(f"TCN model MSE: {mse_tcn:.4f}")
        
        # Calculate skill scores
        # LSTM vs Persistent
        ss_lstm_persistent = calculate_skill_score(mse_lstm, mse_persistent)
        # LSTM vs Analyst
        ss_lstm_analyst = calculate_skill_score(mse_lstm, mse_analyst)
        # TCN vs Persistent
        ss_tcn_persistent = calculate_skill_score(mse_tcn, mse_persistent)
        # TCN vs Analyst
        ss_tcn_analyst = calculate_skill_score(mse_tcn, mse_analyst)
        
        print(f"LSTM skill score vs persistent: {ss_lstm_persistent:.4f}")
        print(f"LSTM skill score vs analyst: {ss_lstm_analyst:.4f}")
        print(f"TCN skill score vs persistent: {ss_tcn_persistent:.4f}")
        print(f"TCN skill score vs analyst: {ss_tcn_analyst:.4f}")
        
        # Store results
        results[dataset_name] = {
            'mse_persistent': mse_persistent,
            'mse_analyst': mse_analyst,
            'mse_lstm': mse_lstm,
            'mse_tcn': mse_tcn,
            'ss_lstm_persistent': ss_lstm_persistent,
            'ss_lstm_analyst': ss_lstm_analyst,
            'ss_tcn_persistent': ss_tcn_persistent,
            'ss_tcn_analyst': ss_tcn_analyst,
            'y_test': y_test,
            'y_persistent': y_persistent,
            'y_analyst': y_analyst,
            'y_pred_lstm': y_pred_lstm,
            'y_pred_tcn': y_pred_tcn,
            'history_lstm': history_lstm,
            'history_tcn': history_tcn
        }
    
    return results

#######################################################
# 5. Visualization
#######################################################

def plot_results(results):
    """
    Visualize the results of the experiment
    
    Parameters:
    -----------
    results: dict
        Dictionary of evaluation results
    """
    # Create a summary DataFrame for skill scores
    data = []
    
    for dataset_name, result in results.items():
        data.append({
            'Dataset': dataset_name,
            'Model': 'LSTM',
            'vs_Persistent': result['ss_lstm_persistent'],
            'vs_Analyst': result['ss_lstm_analyst']
        })
        
        data.append({
            'Dataset': dataset_name,
            'Model': 'TCN',
            'vs_Persistent': result['ss_tcn_persistent'],
            'vs_Analyst': result['ss_tcn_analyst']
        })
    
    summary_df = pd.DataFrame(data)
    
    # Plot skill scores
    plt.figure(figsize=(12, 8))
    
    # Plot vs Persistent
    plt.subplot(2, 1, 1)
    sns.barplot(x='Dataset', y='vs_Persistent', hue='Model', data=summary_df)
    plt.title('Skill Score vs Persistent Model')
    plt.ylabel('Skill Score')
    plt.ylim(-0.1, 0.6)
    
    # Plot vs Analyst
    plt.subplot(2, 1, 2)
    sns.barplot(x='Dataset', y='vs_Analyst', hue='Model', data=summary_df)
    plt.title('Skill Score vs Analyst Forecasts')
    plt.ylabel('Skill Score')
    plt.ylim(-0.2, 0.3)
    
    plt.tight_layout()
    plt.savefig('skill_scores.png')
    plt.close()
    
    # For each dataset, plot predictions vs actual
    for dataset_name, result in results.items():
        plt.figure(figsize=(12, 8))
        
        # Get a subset of points to plot (for clarity)
        sample_idx = np.random.choice(range(len(result['y_test'])), min(50, len(result['y_test'])), replace=False)
        
        plt.subplot(2, 1, 1)
        plt.scatter(result['y_test'][sample_idx], result['y_persistent'][sample_idx], alpha=0.7, label='Persistent')
        plt.scatter(result['y_test'][sample_idx], result['y_analyst'][sample_idx], alpha=0.7, label='Analyst')
        plt.scatter(result['y_test'][sample_idx], result['y_pred_lstm'][sample_idx], alpha=0.7, label='LSTM')
        plt.plot([min(result['y_test']), max(result['y_test'])], [min(result['y_test']), max(result['y_test'])], 'k--')
        plt.xlabel('Actual EPS')
        plt.ylabel('Predicted EPS')
        plt.title(f'Predictions vs Actual - {dataset_name} (LSTM)')
        plt.legend()
        
        plt.subplot(2, 1, 2)
        plt.scatter(result['y_test'][sample_idx], result['y_persistent'][sample_idx], alpha=0.7, label='Persistent')
        plt.scatter(result['y_test'][sample_idx], result['y_analyst'][sample_idx], alpha=0.7, label='Analyst')
        plt.scatter(result['y_test'][sample_idx], result['y_pred_tcn'][sample_idx], alpha=0.7, label='TCN')
        plt.plot([min(result['y_test']), max(result['y_test'])], [min(result['y_test']), max(result['y_test'])], 'k--')
        plt.xlabel('Actual EPS')
        plt.ylabel('Predicted EPS')
        plt.title(f'Predictions vs Actual - {dataset_name} (TCN)')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f'predictions_{dataset_name}.png')
        plt.close()
        
        # Plot training history
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(result['history_lstm']['train_loss'], label='Training Loss')
        plt.plot(result['history_lstm']['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss (MSE)')
        plt.title(f'LSTM Training History - {dataset_name}')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(result['history_tcn']['train_loss'], label='Training Loss')
        plt.plot(result['history_tcn']['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss (MSE)')
        plt.title(f'TCN Training History - {dataset_name}')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f'training_history_{dataset_name}.png')
        plt.close()

#######################################################
# 6. Main Execution
#######################################################

if __name__ == "__main__":
    start_time = time.time()
    
    print("Generating simulated financial data...")
    quarterly_df, daily_df = generate_simulated_financial_data(n_companies=100, n_quarters=40)
    
    print("Preprocessing data...")
    X_quarterly, X_daily, y, companies, quarters, is_financial, studentized_cols, daily_studentized_cols = preprocess_data(
        quarterly_df, daily_df, window_size=20, prediction_horizon=1
    )
    
    print("Data shapes:")
    print(f"X_quarterly shape: {X_quarterly.shape}")
    print(f"X_daily shape: {X_daily.shape}")
    print(f"y shape: {y.shape}")
    print(f"studentized_cols: {studentized_cols}")
    
    print("Evaluating models...")
    results = evaluate_models_performance(
        X_quarterly, X_daily, y, companies, quarters, is_financial, studentized_cols, split_date=30
    )
    
    print("Plotting results...")
    plot_results(results)
    
    end_time = time.time()
    print(f"Experiment completed in {end_time - start_time:.2f} seconds!")