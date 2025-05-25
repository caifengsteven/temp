import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pdblp
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from tqdm import tqdm
import datetime as dt
import warnings
import os
warnings.filterwarnings('ignore')

# Check and optimize CUDA settings for faster training
if torch.cuda.is_available():
    print(f"CUDA is available with {torch.cuda.device_count()} devices")
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    # Set optimized settings for faster training
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    # Enable TF32 (improves performance on Ampere GPUs)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
else:
    print("CUDA is not available, using CPU")

# Connect to Bloomberg
print("Connecting to Bloomberg...")
con = pdblp.BCon(timeout=10000)
con.start()

# Define LSTM Model for time-series data
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1, output_dim=1, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # LSTM layer
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        
        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim, device=x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim, device=x.device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_dim)
        
        # Get the last time step output
        out = self.fc(out[:, -1, :])
        
        return out

# Simple DNN model for baseline comparison
class DNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim=1, dropout=0.2):
        super(DNNModel, self).__init__()
        
        layers = []
        
        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dims[0]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        
        # Hidden layers
        for i in range(len(hidden_dims)-1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        
        # Output layer
        layers.append(nn.Linear(hidden_dims[-1], output_dim))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

# Define the 2-LSTM model as described in the paper
class TwoLSTMModel(nn.Module):
    def __init__(self, intraday_input_dim, interday_input_dim, hidden_dim, dnn_layers=2, dnn_hidden_dim=32, output_dim=1, dropout=0.2):
        super(TwoLSTMModel, self).__init__()
        
        # Intraday LSTM
        self.intraday_lstm = nn.LSTM(intraday_input_dim, hidden_dim, batch_first=True)
        
        # Interday LSTM
        self.interday_lstm = nn.LSTM(interday_input_dim, hidden_dim, batch_first=True)
        
        # DNN layers
        dnn_layers_list = []
        dnn_layers_list.append(nn.Linear(hidden_dim * 2, dnn_hidden_dim))
        dnn_layers_list.append(nn.ReLU())
        dnn_layers_list.append(nn.Dropout(dropout))
        
        for _ in range(dnn_layers - 1):
            dnn_layers_list.append(nn.Linear(dnn_hidden_dim, dnn_hidden_dim))
            dnn_layers_list.append(nn.ReLU())
            dnn_layers_list.append(nn.Dropout(dropout))
            
        dnn_layers_list.append(nn.Linear(dnn_hidden_dim, output_dim))
        
        self.dnn = nn.Sequential(*dnn_layers_list)
        
    def forward(self, intraday_x, interday_x):
        # Process intraday data
        intraday_out, _ = self.intraday_lstm(intraday_x)
        intraday_out = intraday_out[:, -1, :]  # Last hidden state
        
        # Process interday data
        interday_out, _ = self.interday_lstm(interday_x)
        interday_out = interday_out[:, -1, :]  # Last hidden state
        
        # Concatenate
        combined = torch.cat((intraday_out, interday_out), dim=1)
        
        # Feed through DNN
        out = self.dnn(combined)
        
        return out

# Define the p-Pairs-learning 2-LSTM model as described in the paper
class MultiPairLSTMModel(nn.Module):
    def __init__(self, num_pairs, intraday_lag, interday_lag, hidden_dim, dnn_layers=2, dnn_hidden_dim=32, output_dim=1, dropout=0.2):
        super(MultiPairLSTMModel, self).__init__()
        
        self.num_pairs = num_pairs
        
        # Intraday LSTM - input has shape [batch_size, intraday_lag, num_pairs]
        self.intraday_lstm = nn.LSTM(num_pairs, hidden_dim, batch_first=True)
        
        # Interday LSTM - input has shape [batch_size, interday_lag, num_pairs]
        self.interday_lstm = nn.LSTM(num_pairs, hidden_dim, batch_first=True)
        
        # DNN layers
        dnn_layers_list = []
        dnn_layers_list.append(nn.Linear(hidden_dim * 2, dnn_hidden_dim))
        dnn_layers_list.append(nn.ReLU())
        dnn_layers_list.append(nn.Dropout(dropout))
        
        for _ in range(dnn_layers - 1):
            dnn_layers_list.append(nn.Linear(dnn_hidden_dim, dnn_hidden_dim))
            dnn_layers_list.append(nn.ReLU())
            dnn_layers_list.append(nn.Dropout(dropout))
            
        dnn_layers_list.append(nn.Linear(dnn_hidden_dim, output_dim))
        
        self.dnn = nn.Sequential(*dnn_layers_list)
        
    def forward(self, intraday_x, interday_x):
        # Process intraday data
        intraday_out, _ = self.intraday_lstm(intraday_x)
        intraday_out = intraday_out[:, -1, :]  # Last hidden state
        
        # Process interday data
        interday_out, _ = self.interday_lstm(interday_x)
        interday_out = interday_out[:, -1, :]  # Last hidden state
        
        # Concatenate
        combined = torch.cat((intraday_out, interday_out), dim=1)
        
        # Feed through DNN
        out = self.dnn(combined)
        
        return out

# Function to fetch FX data from Bloomberg
def fetch_fx_data(tickers, start_date, end_date, con):
    print(f"Fetching data for {len(tickers)} currency pairs...")
    
    # Define fields to fetch - common for FX data
    fields = ['PX_LAST', 'PX_HIGH', 'PX_LOW', 'PX_OPEN', 'PX_VOLUME']
    
    all_data = {}
    
    for ticker in tqdm(tickers, desc="Fetching Bloomberg data"):
        try:
            # Use bdh without periodicity parameter - default is DAILY
            data = con.bdh(ticker, fields, start_date, end_date)
            
            if not data.empty:
                # Reshape the data for easier processing
                data_reshaped = {}
                for field in fields:
                    if (ticker, field) in data.columns:
                        data_reshaped[field.lower().replace('px_', '')] = data[(ticker, field)]
                
                # Convert to DataFrame
                df = pd.DataFrame(data_reshaped)
                
                # Ensure we have high and low data
                if 'high' in df.columns and 'low' in df.columns:
                    # Store data
                    all_data[ticker] = df
                    print(f"✓ {ticker}: {len(df)} days of data")
                else:
                    print(f"✗ {ticker}: Missing high/low price data")
            else:
                print(f"✗ {ticker}: No data retrieved")
                
        except Exception as e:
            print(f"✗ {ticker}: Error - {str(e)}")
    
    return all_data

# Use mock data if Bloomberg retrieval fails
def generate_mock_data(tickers, start_date, end_date):
    """Generate mock FX data for testing purposes when Bloomberg is unavailable"""
    print("Generating mock FX data for testing...")
    
    all_data = {}
    
    # Convert dates to datetime objects
    start_dt = pd.to_datetime(start_date, format="%Y%m%d")
    end_dt = pd.to_datetime(end_date, format="%Y%m%d")
    
    # Generate a date range
    dates = pd.date_range(start=start_dt, end=end_dt)
    
    for ticker in tickers:
        # Generate mock price data
        base_price = 1.0
        if 'EURUSD' in ticker:
            base_price = 1.1
        elif 'EURSEK' in ticker:
            base_price = 10.5
        elif 'USDJPY' in ticker:
            base_price = 110.0
        elif 'USDMXN' in ticker:
            base_price = 20.0
        
        # Generate random price movements
        np.random.seed(hash(ticker) % 10000)  # Ensure different but consistent patterns per ticker
        
        # Generate daily returns with some autocorrelation
        daily_returns = np.random.normal(0, 0.005, len(dates))
        # Add some autocorrelation
        for i in range(1, len(daily_returns)):
            daily_returns[i] = 0.8 * daily_returns[i] + 0.2 * daily_returns[i-1]
        
        # Generate price levels
        price_levels = base_price * np.exp(np.cumsum(daily_returns))
        
        # Generate high, low, open, close prices
        high_prices = price_levels * np.exp(np.random.normal(0.003, 0.002, len(dates)))
        low_prices = price_levels * np.exp(np.random.normal(-0.003, 0.002, len(dates)))
        open_prices = price_levels * np.exp(np.random.normal(0, 0.001, len(dates)))
        close_prices = price_levels
        
        # Ensure high > low
        for i in range(len(high_prices)):
            if high_prices[i] < low_prices[i]:
                high_prices[i], low_prices[i] = low_prices[i], high_prices[i]
        
        # Create volume data
        volumes = np.abs(np.random.normal(1000000, 200000, len(dates)))
        
        # Create DataFrame
        df = pd.DataFrame({
            'open': open_prices,
            'high': high_prices,
            'low': low_prices,
            'close': close_prices,
            'volume': volumes
        }, index=dates)
        
        all_data[ticker] = df
        print(f"✓ {ticker}: {len(df)} days of mock data")
    
    return all_data

# Generate synthetic intraday data from daily data
def generate_synthetic_intraday_data(daily_data, minutes_per_day=1440):
    """
    Generate synthetic intraday data from daily data to simulate minutely patterns.
    This is used because we can't directly fetch minutely data from Bloomberg in this example.
    """
    intraday_data = {}
    
    for ticker, data in daily_data.items():
        all_minutes = []
        
        for date, row in data.iterrows():
            # Generate random intraday pattern that follows a typical volatility pattern
            # Morning activity, midday lull, end of day spike
            base_volatility_pattern = np.concatenate([
                np.linspace(0.5, 1.0, minutes_per_day // 4),  # Morning ramp-up
                np.linspace(1.0, 0.7, minutes_per_day // 4),  # Late morning decline
                np.linspace(0.7, 0.6, minutes_per_day // 4),  # Midday lull
                np.linspace(0.6, 1.2, minutes_per_day // 4)   # Afternoon activity
            ])
            
            # Add some gaussian noise
            noise = np.random.normal(0, 0.2, minutes_per_day)
            volatility_pattern = base_volatility_pattern + noise
            volatility_pattern = np.maximum(0.1, volatility_pattern)  # Ensure positive
            
            # Scale the pattern based on the day's high-low range
            day_range = row['high'] - row['low']
            minute_ranges = day_range * volatility_pattern / np.sum(volatility_pattern) * minutes_per_day
            
            # Create minute timestamps
            date_str = date.strftime('%Y-%m-%d')
            minute_timestamps = [pd.Timestamp(f"{date_str} {h:02d}:{m:02d}:00") 
                                for h in range(24) 
                                for m in range(60)][:minutes_per_day]
            
            # Generate high and low prices for each minute
            for i, ts in enumerate(minute_timestamps):
                minute_high = row['open'] + np.random.uniform(0, minute_ranges[i])
                minute_low = minute_high - minute_ranges[i]
                
                all_minutes.append({
                    'timestamp': ts,
                    'ticker': ticker,
                    'open': row['open'] + np.random.uniform(-minute_ranges[i]/2, minute_ranges[i]/2),
                    'high': minute_high,
                    'low': minute_low,
                    'close': minute_low + np.random.uniform(0, minute_ranges[i]),
                    'volume': row['volume'] * volatility_pattern[i] / np.sum(volatility_pattern) * minutes_per_day if 'volume' in row else 0
                })
        
        # Convert to DataFrame and sort by timestamp
        intraday_df = pd.DataFrame(all_minutes)
        intraday_df.set_index('timestamp', inplace=True)
        intraday_df.sort_index(inplace=True)
        
        intraday_data[ticker] = intraday_df
    
    return intraday_data

# Calculate log range from high and low prices
def calculate_log_range(data):
    log_high = np.log(data['high'])
    log_low = np.log(data['low'])
    log_range = log_high - log_low
    return log_range

# Optimized data preparation function
def prepare_data(all_data, intraday_lag=20, interday_lag=20):
    """
    Prepare data for the model based on the paper's methodology
    """
    # Generate synthetic intraday data since we can't get minutely data directly
    print("Generating synthetic intraday data...")
    intraday_data = generate_synthetic_intraday_data(all_data)
    
    log_ranges = {}
    
    # Calculate log ranges for each currency pair
    for ticker, data in intraday_data.items():
        # Ensure data is sorted by time
        data = data.sort_index()
        
        # Calculate log range
        data['log_range'] = calculate_log_range(data)
        
        # Store log range series
        ticker_short = ticker.split(' ')[0]  # Extract EURUSD from "EURUSD Curncy"
        log_ranges[ticker_short] = data['log_range']
        print(f"Processed log ranges for {ticker_short}")
    
    # Align all series to have the same index
    log_range_df = pd.DataFrame(log_ranges)
    log_range_df = log_range_df.fillna(method='ffill')
    
    print(f"Log range dataframe columns: {log_range_df.columns}")
    
    # Add time-based features
    log_range_df['minute'] = log_range_df.index.minute
    log_range_df['hour'] = log_range_df.index.hour
    log_range_df['day_of_week'] = log_range_df.index.dayofweek
    log_range_df['month'] = log_range_df.index.month
    log_range_df['is_month_end'] = log_range_df.index.is_month_end.astype(int)
    
    # Create minutely time stamps (0 to 1439 for each day)
    log_range_df['minute_of_day'] = log_range_df['hour'] * 60 + log_range_df['minute']
    
    # Group by date to organize intraday and interday data
    log_range_df['date'] = log_range_df.index.date
    grouped = log_range_df.groupby('date')
    
    # Prepare datasets for the different models
    datasets = {}
    
    # Plain DNN Input: time-based features
    print("Preparing data for DNN model...")
    dnn_data = log_range_df[['minute_of_day', 'day_of_week', 'month', 'is_month_end']].copy()
    
    # Scale DNN input data
    dnn_scaler_X = MinMaxScaler()
    dnn_data_scaled = dnn_scaler_X.fit_transform(dnn_data)
    
    # Scale target data
    target_scaler = MinMaxScaler()
    fx_pair_columns = [col for col in log_range_df.columns if col in log_ranges.keys()]
    target_data = log_range_df[fx_pair_columns].copy()
    target_data_scaled = target_scaler.fit_transform(target_data)
    
    # Create DNN dataset
    X_dnn = dnn_data_scaled[:-1]  # All but last data point
    y_dnn = target_data_scaled[1:]  # All but first data point
    
    datasets['plain_dnn'] = (X_dnn, y_dnn)
    
    # Process each timestamp with enough history
    print("Finding valid indices with sufficient history...")
    valid_indices = []
    for i in range(intraday_lag, len(log_range_df) - 1):
        # Check if we have enough interday data
        curr_date = log_range_df.iloc[i]['date']
        curr_min = log_range_df.iloc[i]['minute_of_day']
        
        # Get all available dates
        all_dates = sorted(grouped.groups.keys())
        if curr_date in all_dates:
            curr_date_idx = all_dates.index(curr_date)
            if curr_date_idx >= interday_lag:
                valid_indices.append(i)
    
    print(f"Found {len(valid_indices)} valid data points with sufficient history")
    
    # For efficiency, limit the number of sequences (optional)
    max_sequences = min(50000, len(valid_indices))  # Adjust this number based on your memory
    if len(valid_indices) > max_sequences:
        print(f"Limiting to {max_sequences} sequences for efficiency")
        valid_indices = valid_indices[:max_sequences]
    
    # Pre-allocate arrays for better memory efficiency
    intraday_data = np.zeros((len(valid_indices), intraday_lag, len(fx_pair_columns)), dtype=np.float32)
    interday_data = np.zeros((len(valid_indices), interday_lag, len(fx_pair_columns)), dtype=np.float32)
    targets = np.zeros((len(valid_indices), len(fx_pair_columns)), dtype=np.float32)
    
    # Track how many valid sequences we've created
    valid_count = 0
    
    # Create sequences for valid indices
    print("Creating LSTM sequences...")
    for idx, i in enumerate(tqdm(valid_indices)):
        # Get intraday lag data
        intraday_seq = target_data_scaled[i-intraday_lag:i]
        
        # Get interday lag data
        curr_date = log_range_df.iloc[i]['date']
        curr_min = log_range_df.iloc[i]['minute_of_day']
        all_dates = sorted(grouped.groups.keys())
        curr_date_idx = all_dates.index(curr_date)
        
        interday_seq = []
        for j in range(1, interday_lag + 1):
            prev_date = all_dates[curr_date_idx - j]
            prev_date_indices = grouped.get_group(prev_date).index
            
            # Find the same minute in previous day
            min_diff = float('inf')
            best_idx = None
            
            for prev_idx in prev_date_indices:
                prev_min = log_range_df.loc[prev_idx, 'minute_of_day']
                if abs(prev_min - curr_min) < min_diff:
                    min_diff = abs(prev_min - curr_min)
                    best_idx = log_range_df.index.get_loc(prev_idx)
            
            if best_idx is not None:
                interday_seq.append(target_data_scaled[best_idx])
        
        # Only use if we have enough interday data
        if len(interday_seq) == interday_lag:
            interday_seq = np.array(interday_seq)
            
            intraday_data[valid_count] = intraday_seq
            interday_data[valid_count] = interday_seq
            targets[valid_count] = target_data_scaled[i+1]  # Next minute's value
            valid_count += 1
    
    # Trim arrays to actual count of valid sequences
    if valid_count < len(valid_indices):
        print(f"Created {valid_count} valid sequences out of {len(valid_indices)} potential indices")
        intraday_data = intraday_data[:valid_count]
        interday_data = interday_data[:valid_count]
        targets = targets[:valid_count]
    
    if valid_count > 0:
        print(f"LSTM dataset shapes: Intraday {intraday_data.shape}, Interday {interday_data.shape}, Targets {targets.shape}")
        datasets['lstm'] = (intraday_data, interday_data, targets)
    else:
        # If we have no sequences, create dummy data for testing
        print("Warning: Could not create proper LSTM sequences. Using dummy data for testing.")
        num_samples = 1000
        num_pairs = len(log_ranges)
        
        intraday_data = np.random.rand(num_samples, intraday_lag, num_pairs)
        interday_data = np.random.rand(num_samples, interday_lag, num_pairs)
        targets = np.random.rand(num_samples, num_pairs)
        
        datasets['lstm'] = (intraday_data, interday_data, targets)
    
    # Return datasets and scalers for prediction
    return datasets, target_scaler

# Optimized training function for GPU
def train_two_lstm_model(model, train_loader, val_loader, epochs=10, learning_rate=0.001, device='cuda', model_name="model"):
    criterion = nn.MSELoss()
    
    # Use AdamW for better performance
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    
    # Learning rate scheduler for better convergence
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2, verbose=True
    )
    
    train_losses = []
    val_losses = []
    
    # Use early stopping
    best_val_loss = float('inf')
    patience = 3
    counter = 0
    best_model_path = f"{model_name}_best.pt"
    
    # Check if we can use mixed precision training
    scaler = torch.cuda.amp.GradScaler() if device == 'cuda' else None
    
    # Time tracking
    start_time = dt.datetime.now()
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - Training"):
            intraday_data = batch['intraday'].to(device)
            interday_data = batch['interday'].to(device)
            targets = batch['target'].to(device)
            
            # Forward pass with mixed precision where available
            optimizer.zero_grad()
            
            if scaler is not None:
                # Using mixed precision
                with torch.cuda.amp.autocast():
                    outputs = model(intraday_data, interday_data)
                    loss = criterion(outputs, targets)
                
                # Backward and optimize with scaled gradients
                scaler.scale(loss).backward()
                # Gradient clipping
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                # Regular precision
                outputs = model(intraday_data, interday_data)
                loss = criterion(outputs, targets)
                
                # Backward and optimize
                loss.backward()
                # Gradient clipping
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            
            train_loss += loss.item()
        
        # Validate
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} - Validation"):
                intraday_data = batch['intraday'].to(device)
                interday_data = batch['interday'].to(device)
                targets = batch['target'].to(device)
                
                if scaler is not None:
                    with torch.cuda.amp.autocast():
                        outputs = model(intraday_data, interday_data)
                        loss = criterion(outputs, targets)
                else:
                    outputs = model(intraday_data, interday_data)
                    loss = criterion(outputs, targets)
                
                val_loss += loss.item()
        
        # Calculate average losses
        train_loss = train_loss / len(train_loader)
        val_loss = val_loss / len(val_loader)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        # Update learning rate based on validation loss
        scheduler.step(val_loss)
        
        # Calculate elapsed time
        elapsed = dt.datetime.now() - start_time
        
        print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, Time: {elapsed}')
        
        # Early stopping and model saving
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
            
            # Save the best model
            torch.save(model.state_dict(), best_model_path)
            print(f"Model saved to {best_model_path}")
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    # Load the best model before returning
    model.load_state_dict(torch.load(best_model_path))
    
    return train_losses, val_losses

# Custom Dataset for TwoLSTM model with pin_memory for faster GPU transfer
class ForexDataset(Dataset):
    def __init__(self, intraday_data, interday_data, targets):
        self.intraday_data = torch.FloatTensor(intraday_data)
        self.interday_data = torch.FloatTensor(interday_data)
        self.targets = torch.FloatTensor(targets)
        
    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, idx):
        return {
            'intraday': self.intraday_data[idx],
            'interday': self.interday_data[idx],
            'target': self.targets[idx]
        }

# Optimized function to evaluate model performance
def evaluate_model(model, test_loader, criterion, device='cuda'):
    model.eval()
    test_loss = 0
    all_preds = []
    all_targets = []
    
    # Use mixed precision for evaluation if available
    scaler = torch.cuda.amp.GradScaler() if device == 'cuda' else None
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            intraday_data = batch['intraday'].to(device)
            interday_data = batch['interday'].to(device)
            targets = batch['target'].to(device)
            
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs = model(intraday_data, interday_data)
                    loss = criterion(outputs, targets)
            else:
                outputs = model(intraday_data, interday_data)
                loss = criterion(outputs, targets)
            
            test_loss += loss.item()
            
            all_preds.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
    
    # Calculate average loss
    avg_loss = test_loss / len(test_loader)
    
    # Combine predictions and targets
    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)
    
    return avg_loss, all_preds, all_targets

# Function to implement Diebold-Mariano test as mentioned in the paper
def diebold_mariano_test(actual, forecast1, forecast2, h=1):
    """
    Implements the Diebold-Mariano test for forecast accuracy comparison.
    
    Parameters:
    -----------
    actual: np.array, actual values
    forecast1: np.array, first forecast
    forecast2: np.array, second forecast
    h: int, forecast horizon
    
    Returns:
    --------
    float: The DM test statistic
    """
    # Calculate squared errors
    e1 = (actual - forecast1)**2
    e2 = (actual - forecast2)**2
    
    # Calculate the difference in errors
    d = e1 - e2
    
    # Calculate mean of the difference
    d_mean = np.mean(d)
    
    # Calculate autocovariance of the difference
    n = len(d)
    gamma_0 = np.var(d)
    gamma = [np.mean((d[:n-k] - d_mean) * (d[k:] - d_mean)) for k in range(1, min(h, n-1))]
    
    # Calculate variance of the mean
    V_d = gamma_0 + 2 * sum(gamma)
    
    # Calculate test statistic
    DM_stat = d_mean / np.sqrt(V_d / n) if V_d > 0 else 0
    
    return DM_stat

# Main function to run the FX volatility prediction
def run_fx_volatility_prediction():
    # Create output directory for plots and models
    os.makedirs("results", exist_ok=True)
    
    # Define currency pairs to analyze as in the paper
    fx_pairs = [
        'EURUSD Curncy',
        'EURSEK Curncy',
        'USDJPY Curncy',
        'USDMXN Curncy'
    ]
    
    # Define date range
    end_date = dt.datetime.now().strftime("%Y%m%d")
    start_date = (dt.datetime.now() - dt.timedelta(days=180)).strftime("%Y%m%d")
    print(f"Date range: {start_date} to {end_date}")
    
    # Fetch data from Bloomberg
    all_data = fetch_fx_data(fx_pairs, start_date, end_date, con)
    
    # If Bloomberg data retrieval fails, use mock data
    if not all_data:
        print("Bloomberg data retrieval failed. Using mock data instead.")
        all_data = generate_mock_data(fx_pairs, start_date, end_date)
    
    if not all_data:
        print("Could not generate any data. Exiting.")
        return
    
    # Print the keys of the data
    print(f"Data available for: {list(all_data.keys())}")
    
    # Prepare data for models
    datasets, target_scaler = prepare_data(all_data, intraday_lag=20, interday_lag=20)
    
    # Get LSTM datasets
    intraday_data, interday_data, targets = datasets['lstm']
    
    print(f"Dataset shapes - Intraday: {intraday_data.shape}, Interday: {interday_data.shape}, Targets: {targets.shape}")
    
    # Split data for training, validation, and testing
    # Use 70% for training, 15% for validation, 15% for testing
    train_size = int(0.7 * len(targets))
    val_size = int(0.15 * len(targets))
    test_size = len(targets) - train_size - val_size
    
    # Create datasets
    train_dataset = ForexDataset(
        intraday_data[:train_size], 
        interday_data[:train_size], 
        targets[:train_size]
    )
    
    val_dataset = ForexDataset(
        intraday_data[train_size:train_size+val_size], 
        interday_data[train_size:train_size+val_size], 
        targets[train_size:train_size+val_size]
    )
    
    test_dataset = ForexDataset(
        intraday_data[train_size+val_size:], 
        interday_data[train_size+val_size:], 
        targets[train_size+val_size:]
    )
    
    # Set device to GPU if available, otherwise use CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create data loaders with optimized settings for GPU
    batch_size = 128 if device == 'cuda' else 32  # Larger batch size for GPU
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        pin_memory=True if device == 'cuda' else False,
        num_workers=4 if device == 'cuda' else 0,  # Multiple workers for faster data loading
        persistent_workers=True if device == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size,
        pin_memory=True if device == 'cuda' else False,
        num_workers=4 if device == 'cuda' else 0,
        persistent_workers=True if device == 'cuda' else False
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size,
        pin_memory=True if device == 'cuda' else False,
        num_workers=4 if device == 'cuda' else 0,
        persistent_workers=True if device == 'cuda' else False
    )
    
    # Create TwoLSTM model
    # Determine dimensions from data
    num_pairs = intraday_data.shape[2]  # Number of currency pairs
    
    model = TwoLSTMModel(
        intraday_input_dim=num_pairs, 
        interday_input_dim=num_pairs,
        hidden_dim=128,  # Increased hidden dim for better learning
        dnn_layers=2,
        dnn_hidden_dim=64,  # Increased hidden dim for DNN
        output_dim=num_pairs
    ).to(device)
    
    # Print model summary
    print(model)
    
    # Count model parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total model parameters: {total_params:,}")
    
    # Train model
    train_losses, val_losses = train_two_lstm_model(
        model,
        train_loader,
        val_loader,
        epochs=20,  # More epochs, early stopping will prevent overfitting
        learning_rate=0.001,
        device=device,
        model_name="results/twolstm"
    )
    
    # Plot training and validation losses
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig('results/training_validation_loss.png')
    plt.close()
    
    # Evaluate on test set
    criterion = nn.MSELoss()
    test_loss, all_preds, all_targets = evaluate_model(model, test_loader, criterion, device)
    
    print(f"Test Loss: {test_loss:.6f}")
    
    # Invert scaling to get actual values
    all_preds_rescaled = target_scaler.inverse_transform(all_preds)
    all_targets_rescaled = target_scaler.inverse_transform(all_targets)
    
    # Get pair names from the original data
    pair_names = [ticker.split(' ')[0] for ticker in fx_pairs]
    print(f"Currency pairs: {pair_names}")
    
    # Plot predictions vs actuals for each currency pair
    for i, pair in enumerate(pair_names):
        if i < all_preds_rescaled.shape[1]:  # Check if we have predictions for this pair
            plt.figure(figsize=(12, 6))
            plt.plot(all_targets_rescaled[:100, i], label='Actual')
            plt.plot(all_preds_rescaled[:100, i], label='Predicted')
            plt.xlabel('Time')
            plt.ylabel('Log Range')
            plt.title(f'Actual vs Predicted Log Range - {pair}')
            plt.legend()
            plt.savefig(f'results/pred_vs_actual_{pair}.png')
            plt.close()
        else:
            print(f"Warning: No predictions for {pair}")
    
    # Calculate mean squared error for each currency pair
    mse_per_pair = {}
    for i, pair in enumerate(pair_names):
        if i < all_preds_rescaled.shape[1]:
            mse = np.mean((all_preds_rescaled[:, i] - all_targets_rescaled[:, i])**2)
            mse_per_pair[pair] = mse
            print(f"MSE for {pair}: {mse:.8f}")
    
    # Implement the p-Pairs learning model (4-Pairs as in the paper)
    p_pairs_model = MultiPairLSTMModel(
        num_pairs=num_pairs,
        intraday_lag=intraday_data.shape[1],
        interday_lag=interday_data.shape[1],
        hidden_dim=128,  # Increased hidden dim
        dnn_layers=2,
        dnn_hidden_dim=64,  # Increased hidden dim
        output_dim=num_pairs
    ).to(device)
    
    print(p_pairs_model)
    
    # Train p-Pairs model
    p_pairs_train_losses, p_pairs_val_losses = train_two_lstm_model(
        p_pairs_model,
        train_loader,
        val_loader,
        epochs=20,  # More epochs with early stopping
        learning_rate=0.001,
        device=device,
        model_name="results/p_pairs"
    )
    
    # Plot p-Pairs training and validation losses
    plt.figure(figsize=(10, 5))
    plt.plot(p_pairs_train_losses, label='Training Loss')
    plt.plot(p_pairs_val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('p-Pairs Learning Model: Training and Validation Loss')
    plt.legend()
    plt.savefig('results/p_pairs_training_validation_loss.png')
    plt.close()
    
    # Evaluate p-Pairs model on test set
    p_pairs_test_loss, p_pairs_all_preds, p_pairs_all_targets = evaluate_model(
        p_pairs_model, test_loader, criterion, device
    )
    
    print(f"p-Pairs Test Loss: {p_pairs_test_loss:.6f}")
    
    # Invert scaling to get actual values
    p_pairs_all_preds_rescaled = target_scaler.inverse_transform(p_pairs_all_preds)
    p_pairs_all_targets_rescaled = target_scaler.inverse_transform(p_pairs_all_targets)
    
    # Plot p-Pairs predictions vs actuals for each currency pair
    for i, pair in enumerate(pair_names):
        if i < p_pairs_all_preds_rescaled.shape[1]:
            plt.figure(figsize=(12, 6))
            plt.plot(p_pairs_all_targets_rescaled[:100, i], label='Actual')
            plt.plot(p_pairs_all_preds_rescaled[:100, i], label='Predicted')
            plt.xlabel('Time')
            plt.ylabel('Log Range')
            plt.title(f'p-Pairs: Actual vs Predicted Log Range - {pair}')
            plt.legend()
            plt.savefig(f'results/p_pairs_pred_vs_actual_{pair}.png')
            plt.close()
    
    # Calculate mean squared error for each currency pair using p-Pairs model
    p_pairs_mse_per_pair = {}
    for i, pair in enumerate(pair_names):
        if i < p_pairs_all_preds_rescaled.shape[1]:
            mse = np.mean((p_pairs_all_preds_rescaled[:, i] - p_pairs_all_targets_rescaled[:, i])**2)
            p_pairs_mse_per_pair[pair] = mse
            print(f"p-Pairs MSE for {pair}: {mse:.8f}")
    
    # Compare 2-LSTM and p-Pairs Learning models
    plt.figure(figsize=(12, 6))
    plt.plot(val_losses, label='2-LSTM Validation Loss')
    plt.plot(p_pairs_val_losses, label='p-Pairs Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('2-LSTM vs p-Pairs Learning: Validation Loss Comparison')
    plt.legend()
    plt.savefig('results/model_comparison.png')
    plt.close()
    
    # Compare MSE across models
    plt.figure(figsize=(12, 6))
    x = np.arange(len(mse_per_pair))
    width = 0.35
    
    plt.bar(x - width/2, list(mse_per_pair.values()), width, label='2-LSTM')
    plt.bar(x + width/2, list(p_pairs_mse_per_pair.values()), width, label='p-Pairs')
    
    plt.xlabel('Currency Pair')
    plt.ylabel('MSE')
    plt.title('MSE Comparison: 2-LSTM vs p-Pairs Learning')
    plt.xticks(x, mse_per_pair.keys())
    plt.legend()
    plt.savefig('results/mse_comparison.png')
    plt.close()
    
    # Perform Diebold-Mariano test to compare forecasts
    dm_results = []
    for i, pair in enumerate(pair_names):
        if i < all_preds_rescaled.shape[1] and i < p_pairs_all_preds_rescaled.shape[1]:
            try:
                dm_stat = diebold_mariano_test(
                    all_targets_rescaled[:, i],
                    all_preds_rescaled[:, i],
                    p_pairs_all_preds_rescaled[:, i]
                )
                
                dm_results.append({
                    'pair': pair,
                    'DM_stat': dm_stat,
                    'better_model': '2-LSTM' if dm_stat < 0 else 'p-Pairs' if dm_stat > 0 else 'Equal'
                })
                
                print(f"DM test for {pair}: {dm_stat:.4f} ({'2-LSTM better' if dm_stat < 0 else 'p-Pairs better' if dm_stat > 0 else 'Models are equal'})")
            except Exception as e:
                print(f"Error calculating DM test for {pair}: {e}")
    
    # Close Bloomberg connection
    con.stop()
    print("Bloomberg connection closed.")
    
    # Save results to file
    results = {
        '2lstm_mse': mse_per_pair,
        'p_pairs_mse': p_pairs_mse_per_pair,
        'dm_results': dm_results
    }
    
    # Convert to pandas DataFrame and save to CSV
    results_df = pd.DataFrame({
        'Currency': list(mse_per_pair.keys()),
        '2LSTM_MSE': list(mse_per_pair.values()),
        'pPairs_MSE': [p_pairs_mse_per_pair.get(pair, float('nan')) for pair in mse_per_pair.keys()],
        'Improvement': [(1 - p_pairs_mse_per_pair.get(pair, float('nan')) / mse) * 100 
                         for pair, mse in mse_per_pair.items()]
    })
    
    results_df.to_csv('results/performance_comparison.csv', index=False)
    print(f"Results saved to results/performance_comparison.csv")
    
    return results

if __name__ == "__main__":
    run_fx_volatility_prediction()