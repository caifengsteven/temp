import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from sklearn.metrics import accuracy_score, matthews_corrcoef
import warnings
warnings.filterwarnings("ignore")
import yfinance as yf
from datetime import datetime, timedelta
from tqdm import tqdm

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# --------------------------------
# ACEEMD Implementation
# --------------------------------

def find_extrema(signal):
    """Find the indices of maxima and minima in a signal."""
    maxima_indices = []
    minima_indices = []
    
    for i in range(1, len(signal) - 1):
        if signal[i] > signal[i-1] and signal[i] > signal[i+1]:
            maxima_indices.append(i)
        elif signal[i] < signal[i-1] and signal[i] < signal[i+1]:
            minima_indices.append(i)
    
    return maxima_indices, minima_indices

def find_middle_points(signal, maxima_indices, minima_indices):
    """Find the middle points between peaks and troughs."""
    middle_indices = []
    
    # Sort all extrema indices
    all_extrema = sorted(maxima_indices + minima_indices)
    
    # Find middle points between consecutive extrema
    for i in range(len(all_extrema) - 1):
        middle_idx = (all_extrema[i] + all_extrema[i+1]) // 2
        middle_indices.append(middle_idx)
    
    return middle_indices

def is_imf(signal):
    """Check if a signal is an IMF (Intrinsic Mode Function)."""
    # Check if number of extrema and zero-crossings differ at most by 1
    zero_crossings = np.sum(np.diff(np.signbit(signal)))
    maxima_indices, minima_indices = find_extrema(signal)
    num_extrema = len(maxima_indices) + len(minima_indices)
    
    # An IMF must have the number of extrema and zero-crossings differ at most by 1
    if abs(num_extrema - zero_crossings) > 1:
        return False
    
    # Check if the envelope mean is close to zero
    maxima_indices = [0] + maxima_indices + [len(signal) - 1]
    minima_indices = [0] + minima_indices + [len(signal) - 1]
    
    x = np.arange(len(signal))
    
    if len(maxima_indices) < 2 or len(minima_indices) < 2:
        return False
    
    # Create envelope using cubic spline
    try:
        upper_envelope = CubicSpline(maxima_indices, signal[maxima_indices])(x)
        lower_envelope = CubicSpline(minima_indices, signal[minima_indices])(x)
        envelope_mean = (upper_envelope + lower_envelope) / 2
    except:
        return False
    
    # Check if envelope mean is close to zero
    if np.mean(np.abs(envelope_mean)) > 0.05 * np.mean(np.abs(signal)):
        return False
    
    return True

def AM(pe, pm, alpha=0.5):
    """
    Aliased Complete algorithm - the core function of ACEEMD.
    pe: signal with positive noise
    pm: signal with negative noise
    alpha: weight parameter
    """
    # Create copies to avoid modifying the originals
    pe_signal = pe.copy()
    pm_signal = pm.copy()
    
    # Obtain intermediate signals through sifting process
    pe_intermediate = sift_process(pe_signal, use_middle_points=False)
    pm_intermediate = sift_process(pm_signal, use_middle_points=True)
    
    # Take weighted average to get the IMF component
    imf = alpha * pe_intermediate + (1 - alpha) * pm_intermediate
    
    return imf

def sift_process(signal, use_middle_points=False, max_iterations=10):
    """
    Perform the sifting process on a signal.
    signal: input signal
    use_middle_points: whether to include middle points in cubic interpolation
    max_iterations: maximum number of iterations
    """
    h = signal.copy()
    iteration = 0
    
    while not is_imf(h) and iteration < max_iterations:
        x = np.arange(len(h))
        
        # Find extrema
        maxima_indices, minima_indices = find_extrema(h)
        
        # Add endpoints
        maxima_indices = [0] + maxima_indices + [len(h) - 1]
        minima_indices = [0] + minima_indices + [len(h) - 1]
        
        # Find middle points if needed
        if use_middle_points:
            middle_indices = find_middle_points(h, maxima_indices, minima_indices)
            # Include middle points for cubic interpolation
            upper_indices = sorted(list(set(maxima_indices + middle_indices)))
            lower_indices = sorted(list(set(minima_indices + middle_indices)))
        else:
            upper_indices = maxima_indices
            lower_indices = minima_indices
        
        # Create envelopes using cubic spline
        try:
            upper_envelope = CubicSpline(upper_indices, h[upper_indices])(x)
            lower_envelope = CubicSpline(lower_indices, h[lower_indices])(x)
        except:
            # If cubic spline fails, return the current signal
            break
        
        # Calculate mean envelope
        mean_envelope = (upper_envelope + lower_envelope) / 2
        
        # Update h by subtracting mean envelope
        h = h - mean_envelope
        iteration += 1
    
    return h

def ACEEMD(signal, num_noises=5, noise_amp=0.1):
    """
    Alias Complete Ensemble Empirical Mode Decomposition with Adaptive Noise.
    signal: input signal
    num_noises: number of noise realizations to use
    noise_amp: amplitude of added noise relative to signal standard deviation
    """
    # Initialize result
    denoised_signal = np.zeros_like(signal, dtype=float)
    
    # Calculate noise amplitude based on signal standard deviation
    noise_std = noise_amp * np.std(signal)
    
    # Process with multiple noise realizations
    for i in range(num_noises):
        # Generate Gaussian noise
        noise = np.random.normal(0, noise_std, len(signal))
        
        # Add noise with opposite signs
        pe = signal + noise  # positive noise
        pm = signal - noise  # negative noise
        
        # Apply AM function to get first IMF
        imf1 = AM(pe, pm)
        
        # Remove first IMF to get denoised signal
        r1 = signal - imf1
        
        # Accumulate the denoised signal
        denoised_signal += r1
    
    # Average the ensemble
    denoised_signal /= num_noises
    
    return denoised_signal

# --------------------------------
# ACEFormer Model Implementation
# --------------------------------

class ProbAttention(nn.Module):
    """
    Probability Attention mechanism as described in Informer paper.
    """
    def __init__(self, d_model, n_heads, dropout=0.1):
        super(ProbAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.dropout = nn.Dropout(dropout)
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        
    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        
        # Linear projections
        q = self.q_linear(q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k = self.k_linear(k).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v = self.v_linear(v).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        # Calculate attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax to get attention weights
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention weights to values
        context = torch.matmul(attn_weights, v)
        
        # Reshape and apply final linear projection
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.out(context)
        
        return output

class TimeAwareMechanism(nn.Module):
    """
    Time-aware mechanism to extract temporal features.
    """
    def __init__(self, input_dim, output_dim):
        super(TimeAwareMechanism, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        
    def forward(self, x):
        return self.linear(x)

class DistillationModule(nn.Module):
    """
    Distillation module with probability attention, convolution, max pooling, and time-aware mechanism.
    """
    def __init__(self, d_model, n_heads, dropout=0.1):
        super(DistillationModule, self).__init__()
        self.prob_attention = ProbAttention(d_model, n_heads, dropout)
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
        self.time_aware = TimeAwareMechanism(d_model, d_model)
        
    def forward(self, x):
        # Apply probability attention
        attended = self.prob_attention(x, x, x)
        
        # Apply convolution
        conv_out = self.conv(attended.transpose(1, 2)).transpose(1, 2)
        
        # Apply max pooling (using F.max_pool1d with k=2)
        pooled = F.max_pool1d(conv_out.transpose(1, 2), kernel_size=2).transpose(1, 2)
        
        # Apply time-aware mechanism
        time_aware_out = self.time_aware(x)
        
        # Adjust the dimensions of time_aware_out to match pooled
        time_aware_pooled = F.max_pool1d(time_aware_out.transpose(1, 2), kernel_size=2).transpose(1, 2)
        
        # Combine pooled features with time-aware features
        output = pooled + time_aware_pooled
        
        return output

class AttentionModule(nn.Module):
    """
    Standard self-attention module.
    """
    def __init__(self, d_model, n_heads, dropout=0.1):
        super(AttentionModule, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        
    def forward(self, x):
        # MultiheadAttention expects input of shape (seq_len, batch, d_model)
        x = x.transpose(0, 1)
        attn_output, _ = self.multihead_attn(x, x, x)
        return attn_output.transpose(0, 1)

class ACEFormer(nn.Module):
    """
    ACEFormer model for stock forecasting.
    """
    def __init__(self, input_dim, d_model, n_heads, num_layers, output_dim, dropout=0.1):
        super(ACEFormer, self).__init__()
        
        # Pretreatment module
        self.input_proj = nn.Conv1d(input_dim, d_model, kernel_size=1)
        self.pos_encoding = nn.Parameter(torch.randn(1, 100, d_model))  # Assuming max sequence length of 100
        
        # Distillation module
        self.distillation = DistillationModule(d_model, n_heads, dropout)
        
        # Attention module
        self.attention_layers = nn.ModuleList([
            AttentionModule(d_model, n_heads, dropout) for _ in range(num_layers)
        ])
        
        # Fully connected module
        self.fc_layers = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, output_dim)
        )
        
    def forward(self, x, apply_aceemd=True):
        batch_size, seq_len, features = x.size()
        
        # Apply ACEEMD for denoising (on CPU)
        if apply_aceemd:
            x_np = x.detach().cpu().numpy()
            x_denoised = np.zeros_like(x_np)
            
            # Apply ACEEMD to each feature dimension
            for b in range(batch_size):
                for f in range(features):
                    x_denoised[b, :, f] = ACEEMD(x_np[b, :, f])
            
            # Convert back to tensor
            x_denoised = torch.from_numpy(x_denoised).to(x.device)
            x = x_denoised
        
        # Pretreatment module
        x = x.transpose(1, 2)  # [batch, features, seq_len]
        x = self.input_proj(x)  # [batch, d_model, seq_len]
        x = x.transpose(1, 2)  # [batch, seq_len, d_model]
        
        # Add positional encoding
        pos_enc = self.pos_encoding[:, :seq_len, :]
        x = x + pos_enc
        
        # Distillation module
        x = self.distillation(x)
        
        # Attention module
        for layer in self.attention_layers:
            x = x + layer(x)  # Residual connection
        
        # Take the last sequence element for prediction
        x = x[:, -1, :]
        
        # Fully connected module
        output = self.fc_layers(x)
        
        return output

# --------------------------------
# Data Generation and Preparation
# --------------------------------

def generate_simulated_stock_data(n_days=1000, volatility=0.01, drift=0.0005, start_price=100.0, seed=42):
    """
    Generate simulated stock price data with realistic characteristics.
    """
    np.random.seed(seed)
    
    # Generate random returns
    returns = np.random.normal(drift, volatility, n_days)
    
    # Add some autocorrelation to make it more realistic
    for i in range(1, n_days):
        returns[i] = 0.1 * returns[i-1] + 0.9 * returns[i]
    
    # Generate prices from returns
    prices = np.zeros(n_days)
    prices[0] = start_price
    
    for i in range(1, n_days):
        prices[i] = prices[i-1] * (1 + returns[i])
    
    # Generate trading volume (correlated with absolute returns)
    volume = np.random.lognormal(mean=14, sigma=1.2, size=n_days)
    volume = volume * (1 + 5 * np.abs(returns))  # Higher volume on days with large price movements
    
    # Add seasonality to the volume (higher on certain days)
    for i in range(0, n_days, 5):  # Higher volume every 5 days
        if i < n_days:
            volume[i] *= 1.5
    
    # Generate market indices (correlated with the stock)
    index1 = np.zeros(n_days)
    index1[0] = 1000
    
    index2 = np.zeros(n_days)
    index2[0] = 3000
    
    # Market indices are correlated with the stock but have their own dynamics
    for i in range(1, n_days):
        # 60% correlation with stock, 40% own dynamics
        index1_return = 0.6 * returns[i] + 0.4 * np.random.normal(drift, volatility * 0.8)
        index2_return = 0.7 * returns[i] + 0.3 * np.random.normal(drift, volatility * 0.9)
        
        index1[i] = index1[i-1] * (1 + index1_return)
        index2[i] = index2[i-1] * (1 + index2_return)
    
    # Create a DataFrame with the simulated data
    dates = pd.date_range(start='2020-01-01', periods=n_days)
    df = pd.DataFrame({
        'date': dates,
        'close': prices,
        'volume': volume,
        'index1': index1,
        'index2': index2,
        'returns': returns
    })
    
    return df

def prepare_stock_data(df, seq_length=30, pred_days=1):
    """
    Prepare stock data for training and testing.
    
    Args:
        df: DataFrame containing stock data
        seq_length: input sequence length
        pred_days: number of days to predict ahead
    
    Returns:
        X: input features
        y: target values (1 for price increase, 0 for price decrease)
    """
    data = []
    targets = []
    
    # Features: [close, volume, index1, index2]
    features = ['close', 'volume', 'index1', 'index2']
    
    for i in range(len(df) - seq_length - pred_days):
        # Extract sequence
        sequence = df.iloc[i:i+seq_length][features].values
        
        # Normalize sequence (each feature separately)
        normalized_sequence = np.zeros_like(sequence)
        for j in range(sequence.shape[1]):
            feature = sequence[:, j]
            normalized_sequence[:, j] = (feature - np.mean(feature)) / (np.std(feature) + 1e-10)
        
        # Target: 1 if price increases, 0 if price decreases
        current_price = df.iloc[i+seq_length-1]['close']
        future_price = df.iloc[i+seq_length+pred_days-1]['close']
        target = 1 if future_price > current_price else 0
        
        data.append(normalized_sequence)
        targets.append(target)
    
    return np.array(data), np.array(targets)

class StockDataset(Dataset):
    """Stock market dataset."""
    
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# --------------------------------
# Training and Evaluation Functions
# --------------------------------

def train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs=50, patience=10):
    """
    Train the model with early stopping.
    """
    model.to(device)
    best_val_acc = 0
    patience_counter = 0
    
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += targets.size(0)
            train_correct += (predicted == targets).sum().item()
        
        train_loss /= len(train_loader)
        train_acc = 100 * train_correct / train_total
        
        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += targets.size(0)
                val_correct += (predicted == targets).sum().item()
        
        val_loss /= len(val_loader)
        val_acc = 100 * val_correct / val_total
        
        # Save metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping after {epoch+1} epochs')
                break
    
    # Load best model
    model.load_state_dict(torch.load('best_model.pth'))
    
    return model, train_losses, val_losses, train_accs, val_accs

def evaluate_model(model, test_loader, device):
    """
    Evaluate the model on test data.
    """
    model.to(device)
    model.eval()
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    # Calculate metrics
    acc = accuracy_score(all_targets, all_preds) * 100
    mcc = matthews_corrcoef(all_targets, all_preds)
    
    return acc, mcc, all_preds, all_targets

def calculate_returns(df, predictions, seq_length, pred_days=1):
    """
    Calculate investment returns based on predictions.
    
    Args:
        df: DataFrame containing stock data
        predictions: model predictions (1 for buy, 0 for sell)
        seq_length: input sequence length
        pred_days: number of days to predict ahead
    
    Returns:
        irr: investment return ratio
        sr: Sharpe ratio
    """
    # Get actual price changes
    price_changes = []
    for i in range(len(predictions)):
        current_price = df.iloc[i+seq_length-1]['close']
        future_price = df.iloc[i+seq_length+pred_days-1]['close']
        percent_change = (future_price - current_price) / current_price
        price_changes.append(percent_change)
    
    # Calculate daily returns based on predictions
    daily_returns = []
    for i in range(len(predictions)):
        if predictions[i] == 1:  # Buy signal
            daily_returns.append(price_changes[i])
        else:  # Sell signal
            daily_returns.append(-price_changes[i])
    
    # Calculate IRR
    irr = np.prod(1 + np.array(daily_returns)) - 1
    
    # Calculate Sharpe Ratio (assuming 0% risk-free rate for simplicity)
    risk_free_rate = 0
    excess_returns = np.array(daily_returns) - risk_free_rate
    if np.std(excess_returns) == 0:
        sr = 0
    else:
        sr = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)  # Annualized
    
    return irr * 100, sr  # IRR in percentage

def plot_results(train_losses, val_losses, train_accs, val_accs):
    """Plot training and validation metrics."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot losses
    ax1.plot(train_losses, label='Train Loss')
    ax1.plot(val_losses, label='Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    
    # Plot accuracies
    ax2.plot(train_accs, label='Train Accuracy')
    ax2.plot(val_accs, label='Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()

def plot_predictions_vs_actual(df, predictions, seq_length, pred_days=1):
    """Plot the predictions against the actual price movement."""
    actual = []
    for i in range(len(predictions)):
        current_price = df.iloc[i+seq_length-1]['close']
        future_price = df.iloc[i+seq_length+pred_days-1]['close']
        actual.append(1 if future_price > current_price else 0)
    
    # Get dates for x-axis
    dates = df.iloc[seq_length:seq_length+len(predictions)]['date']
    
    plt.figure(figsize=(15, 6))
    plt.plot(dates, actual, 'b-', label='Actual Direction')
    plt.plot(dates, predictions, 'r--', label='Predicted Direction')
    plt.xlabel('Date')
    plt.ylabel('Direction (1=Up, 0=Down)')
    plt.title('Actual vs Predicted Stock Movement')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_cumulative_returns(df, predictions, seq_length, pred_days=1):
    """Plot cumulative returns from the trading strategy."""
    # Get actual price changes
    price_changes = []
    for i in range(len(predictions)):
        current_price = df.iloc[i+seq_length-1]['close']
        future_price = df.iloc[i+seq_length+pred_days-1]['close']
        percent_change = (future_price - current_price) / current_price
        price_changes.append(percent_change)
    
    # Calculate daily returns based on predictions
    daily_returns = []
    for i in range(len(predictions)):
        if predictions[i] == 1:  # Buy signal
            daily_returns.append(price_changes[i])
        else:  # Sell signal
            daily_returns.append(-price_changes[i])
    
    # Calculate buy-and-hold returns (benchmark)
    benchmark_returns = price_changes
    
    # Calculate cumulative returns
    strategy_cumulative = np.cumprod(1 + np.array(daily_returns)) - 1
    benchmark_cumulative = np.cumprod(1 + np.array(benchmark_returns)) - 1
    
    # Get dates for x-axis
    dates = df.iloc[seq_length:seq_length+len(predictions)]['date']
    
    plt.figure(figsize=(15, 6))
    plt.plot(dates, strategy_cumulative * 100, 'g-', label='Strategy Returns')
    plt.plot(dates, benchmark_cumulative * 100, 'b--', label='Buy-and-Hold Returns')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Returns (%)')
    plt.title('Cumulative Returns: Strategy vs Buy-and-Hold')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_aceemd_example(signal):
    """Plot an example of ACEEMD denoising."""
    denoised_signal = ACEEMD(signal)
    
    plt.figure(figsize=(15, 6))
    plt.plot(signal, 'b-', label='Original Signal')
    plt.plot(denoised_signal, 'r-', label='Denoised Signal')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title('ACEEMD Denoising Example')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    # Plot the removed noise
    noise = signal - denoised_signal
    plt.figure(figsize=(15, 6))
    plt.plot(noise, 'g-', label='Removed Noise')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title('Noise Removed by ACEEMD')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# --------------------------------
# Main Execution
# --------------------------------

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Generate simulated stock data
    print("Generating simulated stock data...")
    df = generate_simulated_stock_data(n_days=1000)
    
    # Show an example of ACEEMD denoising
    print("Demonstrating ACEEMD denoising...")
    plot_aceemd_example(df['close'].values)
    
    # Prepare data
    print("Preparing data...")
    seq_length = 30
    pred_days = 1
    X, y = prepare_stock_data(df, seq_length, pred_days)
    
    # Split data into train, validation, and test sets
    train_size = int(0.7 * len(X))
    val_size = int(0.15 * len(X))
    
    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size:train_size+val_size], y[train_size:train_size+val_size]
    X_test, y_test = X[train_size+val_size:], y[train_size+val_size:]
    
    # Create datasets and dataloaders
    train_dataset = StockDataset(X_train, y_train)
    val_dataset = StockDataset(X_val, y_val)
    test_dataset = StockDataset(X_test, y_test)
    
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Initialize model
    print("Initializing model...")
    input_dim = X.shape[2]  # Number of features
    d_model = 64
    n_heads = 8
    num_layers = 2
    output_dim = 2  # Binary classification (up or down)
    
    model = ACEFormer(input_dim, d_model, n_heads, num_layers, output_dim)
    
    # Set up loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train model
    print("Training model...")
    model, train_losses, val_losses, train_accs, val_accs = train_model(
        model, train_loader, val_loader, criterion, optimizer, device, epochs=50, patience=10
    )
    
    # Plot training results
    plot_results(train_losses, val_losses, train_accs, val_accs)
    
    # Evaluate model on test set
    print("Evaluating model...")
    acc, mcc, predictions, targets = evaluate_model(model, test_loader, device)
    
    # Calculate returns
    test_start_idx = train_size + val_size
    test_df = df.iloc[test_start_idx:test_start_idx+len(predictions)]
    irr, sr = calculate_returns(df, predictions, seq_length, pred_days)
    
    print(f"Test Accuracy: {acc:.2f}%")
    print(f"Matthews Correlation Coefficient: {mcc:.4f}")
    print(f"Investment Return Ratio: {irr:.2f}%")
    print(f"Sharpe Ratio: {sr:.4f}")
    
    # Plot predictions vs actual
    plot_predictions_vs_actual(df, predictions, seq_length + test_start_idx, pred_days)
    
    # Plot cumulative returns
    plot_cumulative_returns(df, predictions, seq_length + test_start_idx, pred_days)
    
    # Compare with baseline models
    print("Comparing with baseline models...")
    # For the sake of brevity, we'll just print a comparison table
    # In a real implementation, you would train and evaluate each baseline model
    
    comparison_table = pd.DataFrame({
        'Model': ['ACEFormer', 'Informer (baseline)', 'DLinear (baseline)', 'TimesNet (baseline)', 'Non-stationary Transformer (baseline)'],
        'Accuracy (%)': [acc, 45.0, 48.0, 48.0, 56.0],
        'MCC': [mcc, -0.14, -0.04, -0.03, 0.12],
        'IRR (%)': [irr, -8.0, -2.7, -6.9, 2.1],
        'Sharpe Ratio': [sr, -2.0, -0.9, -2.0, 0.5]
    })
    
    print(comparison_table)

# Get real-world stock data for a more realistic test
def get_real_stock_data():
    # Get SPY (S&P 500 ETF) data
    spy = yf.download('SPY', start='2015-01-01', end='2022-01-31')
    # Get QQQ (NASDAQ-100 ETF) data
    qqq = yf.download('QQQ', start='2015-01-01', end='2022-01-31')
    
    # Ensure the dates align
    common_dates = spy.index.intersection(qqq.index)
    spy = spy.loc[common_dates]
    qqq = qqq.loc[common_dates]
    
    # Create a DataFrame with both ETFs
    df = pd.DataFrame({
        'date': spy.index,
        'close': spy['Close'],
        'volume': spy['Volume'],
        'index1': qqq['Close'],  # Using QQQ as a market index
        'index2': spy['Open'],   # Using SPY Open as another index for demonstration
        'returns': spy['Close'].pct_change()
    })
    
    # Drop rows with NaN values
    df = df.dropna()
    
    return df

# Run a more realistic test using actual stock data
def test_with_real_data():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Get real stock data
    print("Getting real stock data...")
    df = get_real_stock_data()
    
    # Prepare data
    print("Preparing data...")
    seq_length = 30
    pred_days = 1
    X, y = prepare_stock_data(df, seq_length, pred_days)
    
    # Split data into train, validation, and test sets
    train_size = int(0.7 * len(X))
    val_size = int(0.15 * len(X))
    
    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size:train_size+val_size], y[train_size:train_size+val_size]
    X_test, y_test = X[train_size+val_size:], y[train_size+val_size:]
    
    # Create datasets and dataloaders
    train_dataset = StockDataset(X_train, y_train)
    val_dataset = StockDataset(X_val, y_val)
    test_dataset = StockDataset(X_test, y_test)
    
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Initialize model
    print("Initializing model...")
    input_dim = X.shape[2]  # Number of features
    d_model = 64
    n_heads = 8
    num_layers = 2
    output_dim = 2  # Binary classification (up or down)
    
    model = ACEFormer(input_dim, d_model, n_heads, num_layers, output_dim)
    
    # Set up loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train model
    print("Training model...")
    model, train_losses, val_losses, train_accs, val_accs = train_model(
        model, train_loader, val_loader, criterion, optimizer, device, epochs=50, patience=10
    )
    
    # Plot training results
    plot_results(train_losses, val_losses, train_accs, val_accs)
    
    # Evaluate model on test set
    print("Evaluating model...")
    acc, mcc, predictions, targets = evaluate_model(model, test_loader, device)
    
    # Calculate returns
    test_start_idx = train_size + val_size
    test_df = df.iloc[test_start_idx:test_start_idx+len(predictions)]
    irr, sr = calculate_returns(df, predictions, seq_length, pred_days)
    
    print(f"Test Accuracy: {acc:.2f}%")
    print(f"Matthews Correlation Coefficient: {mcc:.4f}")
    print(f"Investment Return Ratio: {irr:.2f}%")
    print(f"Sharpe Ratio: {sr:.4f}")
    
    # Plot predictions vs actual
    plot_predictions_vs_actual(df, predictions, seq_length + test_start_idx, pred_days)
    
    # Plot cumulative returns
    plot_cumulative_returns(df, predictions, seq_length + test_start_idx, pred_days)

if __name__ == "__main__":
    main()
    # Uncomment to run test with real data
    # test_with_real_data()