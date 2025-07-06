import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import pywt
import scipy.stats as stats
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42) if torch.cuda.is_available() else None

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Constants
INPUT_WINDOW_SIZE = 20  # T1
PREDICT_WINDOW_SIZE = 2  # T2
NUM_STOCKS = 50  # N, number of stocks (reduced from 300 for computational efficiency)
NUM_FACTORS = 360  # Number of price-volume factors
INPUT_DIM = NUM_FACTORS + 2  # +2 for return and trend direction
HIDDEN_DIM = 128  # de in the paper
NUM_HEADS = 1  # e in the paper
NUM_LAYERS = 2  # L in the paper
BATCH_SIZE = 16
LEARNING_RATE = 0.001
NUM_EPOCHS = 50
DROPOUT_RATE = 0.2

class StockDataGenerator:
    """Generate synthetic stock market data with price-volume factors."""
    
    def __init__(self, num_stocks=NUM_STOCKS, num_factors=NUM_FACTORS, 
                 num_days=1000, market_factors=3):
        self.num_stocks = num_stocks
        self.num_factors = num_factors
        self.num_days = num_days
        self.market_factors = market_factors
        
    def generate_data(self):
        """Generate synthetic stock market data."""
        # Generate market factors (common factors affecting all stocks)
        market_factor = np.random.normal(0, 1, (self.num_days, self.market_factors))
        
        # Generate stock-specific factors
        stock_specific = np.random.normal(0, 1, (self.num_days, self.num_stocks))
        
        # Generate stock prices based on factors
        prices = np.zeros((self.num_days, self.num_stocks))
        prices[0] = 100 * np.ones(self.num_stocks)  # Initial prices
        
        # Market factor loadings for each stock (beta)
        factor_loadings = np.random.normal(1, 0.3, (self.num_stocks, self.market_factors))
        
        # Generate daily returns based on factor model
        daily_returns = np.zeros((self.num_days-1, self.num_stocks))
        
        for t in range(1, self.num_days):
            # Market component + stock-specific component
            daily_returns[t-1] = (market_factor[t] @ factor_loadings.T) * 0.01 + stock_specific[t] * 0.005
            prices[t] = prices[t-1] * (1 + daily_returns[t-1])
        
        # Generate trend indicators (1 for up, 0 for down)
        trend = (daily_returns > 0).astype(int)
        
        # Generate volume data
        volume = np.random.lognormal(0, 1, (self.num_days, self.num_stocks))
        # Volume tends to be higher on days with larger price movements
        volume[1:] *= (1 + 5 * np.abs(daily_returns))
        
        # Generate price-volume factors
        factors = self._generate_alpha_factors(prices, volume)
        
        # Create a DataFrame for easier manipulation
        data = {
            'prices': prices[1:],  # Remove the first day since we don't have returns for it
            'returns': daily_returns,
            'trend': trend,
            'volume': volume[1:],
            'factors': factors
        }
        
        return data
    
    def _generate_alpha_factors(self, prices, volume):
        """Generate Alpha360-like factors."""
        num_days = prices.shape[0]
        factors = np.zeros((num_days-1, self.num_stocks, self.num_factors))
        
        # We'll generate 6 factor categories, each with 60 factors
        factor_idx = 0
        
        # 1. Close/Close factors
        for lag in range(1, 61):
            for t in range(lag, num_days-1):
                if t >= lag:
                    factors[t-1, :, factor_idx] = prices[t-lag] / prices[t]
            factor_idx += 1
            
        # 2. Simulate other factors like Open/Close, High/Close, Low/Close, etc.
        # For simplicity, we'll add some noise to the Close/Close factors
        for category in range(5):  # 5 more categories
            for lag in range(1, 61):
                noise = np.random.normal(0, 0.01, (num_days-1, self.num_stocks))
                factors[:, :, factor_idx] = factors[:, :, lag-1] + noise
                factor_idx += 1
        
        return factors
    
    def split_data(self, data, train_ratio=0.75, val_ratio=0.125):
        """Split data into training, validation, and testing sets."""
        num_samples = data['returns'].shape[0]
        train_idx = int(num_samples * train_ratio)
        val_idx = int(num_samples * (train_ratio + val_ratio))
        
        train_data = {k: v[:train_idx] for k, v in data.items()}
        val_data = {k: v[train_idx:val_idx] for k, v in data.items()}
        test_data = {k: v[val_idx:] for k, v in data.items()}
        
        return train_data, val_data, test_data
    
    def prepare_sequences(self, data, window_size=INPUT_WINDOW_SIZE, pred_size=PREDICT_WINDOW_SIZE):
        """Prepare sequences for training and inference."""
        num_days = data['returns'].shape[0]
        
        X = []
        y_reg = []  # Regression targets (returns)
        y_cls = []  # Classification targets (trend)
        
        for i in range(num_days - window_size - pred_size + 1):
            # Input window
            features = np.zeros((window_size, self.num_stocks, INPUT_DIM))
            
            # Returns and trend for the input window
            features[:, :, 0] = data['returns'][i:i+window_size]
            features[:, :, 1] = data['trend'][i:i+window_size]
            
            # Factors for the input window
            features[:, :, 2:] = data['factors'][i:i+window_size]
            
            # Target windows
            target_returns = data['returns'][i+window_size:i+window_size+pred_size]
            target_trend = data['trend'][i+window_size:i+window_size+pred_size]
            
            X.append(features)
            y_reg.append(target_returns)
            y_cls.append(target_trend)
        
        return np.array(X), np.array(y_reg), np.array(y_cls)

class WaveletTransform:
    """Apply wavelet transform to decompose signals into high and low frequencies."""
    
    def __init__(self, wavelet='db1', level=1):
        self.wavelet = wavelet
        self.level = level
    
    def decompose(self, signal):
        """Decompose a signal into low and high frequency components."""
        # Apply discrete wavelet transform
        coeffs = pywt.wavedec(signal, self.wavelet, level=self.level)
        
        # Reconstruct low and high frequency components
        low_freq = pywt.waverec([coeffs[0]] + [None] * self.level, self.wavelet)
        high_freq = pywt.waverec([None] + coeffs[1:], self.wavelet)
        
        # Ensure same length as original signal
        low_freq = low_freq[:len(signal)]
        high_freq = high_freq[:len(signal)]
        
        return low_freq, high_freq

class StockDataset(Dataset):
    """Dataset for stock sequences."""
    
    def __init__(self, X, y_reg, y_cls, transform=None):
        self.X = torch.FloatTensor(X)
        self.y_reg = torch.FloatTensor(y_reg)
        self.y_cls = torch.FloatTensor(y_cls)
        self.transform = transform
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        x = self.X[idx]
        
        if self.transform:
            # Apply wavelet transform to return sequences
            x_transformed = x.clone()
            
            for stock_idx in range(x.shape[1]):
                low_freq, high_freq = self.transform.decompose(x[:, stock_idx, 0].numpy())
                x_transformed[:, stock_idx, 0] = torch.FloatTensor(low_freq)
            
            return x_transformed, x, self.y_reg[idx], self.y_cls[idx]
        
        return x, x, self.y_reg[idx], self.y_cls[idx]

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""
    
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class TemporalAttention(nn.Module):
    """Temporal attention module for low-frequency components."""
    
    def __init__(self, input_dim, hidden_dim, num_heads, dropout=0.1):
        super(TemporalAttention, self).__init__()
        self.self_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(input_dim, hidden_dim)
        
    def forward(self, x):
        # x shape: [seq_len, batch_size * num_stocks, input_dim]
        x = self.fc(x)
        attn_output, _ = self.self_attn(x, x, x)
        x = x + self.dropout(attn_output)
        x = self.norm(x)
        return x

class DilatedCausalConv(nn.Module):
    """Dilated causal convolution for high-frequency components."""
    
    def __init__(self, input_dim, hidden_dim, kernel_size=2, dilation=1):
        super(DilatedCausalConv, self).__init__()
        self.conv = nn.Conv1d(input_dim, hidden_dim, kernel_size,
                              padding=(kernel_size-1) * dilation,
                              dilation=dilation)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # x shape: [batch_size * num_stocks, input_dim, seq_len]
        x = self.conv(x)
        # Remove padding to maintain causality
        x = x[:, :, :-(self.conv.padding[0])]
        x = self.relu(x)
        return x

class GraphAttention(nn.Module):
    """Graph attention module for capturing stock relationships."""
    
    def __init__(self, input_dim, hidden_dim, dropout=0.1):
        super(GraphAttention, self).__init__()
        self.fc = nn.Linear(input_dim, hidden_dim)
        self.attn = nn.Linear(hidden_dim * 2, 1)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, adj_matrix=None):
        # x shape: [seq_len, batch_size, num_stocks, hidden_dim]
        seq_len, batch_size, num_stocks, hidden_dim = x.shape
        x_flat = x.view(seq_len, batch_size * num_stocks, hidden_dim)
        
        # Transform features
        h = self.fc(x_flat)
        
        # Reshape back
        h = h.view(seq_len, batch_size, num_stocks, hidden_dim)
        
        # If adjacency matrix is not provided, use fully connected graph
        if adj_matrix is None:
            adj_matrix = torch.ones(batch_size, num_stocks, num_stocks, device=x.device)
        
        # Compute attention scores
        attention_scores = torch.zeros(batch_size, num_stocks, num_stocks, device=x.device)
        
        for i in range(num_stocks):
            for j in range(num_stocks):
                if adj_matrix[:, i, j].any():
                    # Concatenate features for nodes i and j
                    concat_features = torch.cat([h[-1, :, i, :], h[-1, :, j, :]], dim=1)
                    attention_scores[:, i, j] = self.attn(concat_features).squeeze()
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(attention_scores, dim=2)
        
        # Apply attention weights
        output = torch.zeros_like(h)
        for t in range(seq_len):
            for i in range(num_stocks):
                weighted_sum = torch.zeros(batch_size, hidden_dim, device=x.device)
                for j in range(num_stocks):
                    weighted_sum += attention_weights[:, i, j].unsqueeze(1) * h[t, :, j, :]
                output[t, :, i, :] = weighted_sum
        
        output_flat = output.view(seq_len, batch_size * num_stocks, hidden_dim)
        output_flat = output_flat + self.dropout(h.view(seq_len, batch_size * num_stocks, hidden_dim))
        output_flat = self.norm(output_flat)
        
        return output_flat.view(seq_len, batch_size, num_stocks, hidden_dim)

class FusionAttention(nn.Module):
    """Fusion attention to combine low and high frequency representations."""
    
    def __init__(self, hidden_dim, num_heads, dropout=0.1):
        super(FusionAttention, self).__init__()
        self.self_attn_low = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout)
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, low_freq, high_freq):
        # low_freq, high_freq shape: [seq_len, batch_size * num_stocks, hidden_dim]
        
        # Self-attention on low frequency
        attn_output_low, _ = self.self_attn_low(low_freq, low_freq, low_freq)
        low_freq = low_freq + self.dropout(attn_output_low)
        low_freq = self.norm1(low_freq)
        
        # Cross-attention between low and high frequency
        attn_output_cross, _ = self.cross_attn(low_freq, high_freq, high_freq)
        fusion = low_freq + self.dropout(attn_output_cross)
        fusion = self.norm2(fusion)
        
        return fusion

class DualFrequencyPredictor(nn.Module):
    """Predictor for future time steps."""
    
    def __init__(self, hidden_dim, output_dim, pred_steps):
        super(DualFrequencyPredictor, self).__init__()
        self.fc = nn.Linear(hidden_dim, output_dim * pred_steps)
        self.pred_steps = pred_steps
        self.output_dim = output_dim
        
    def forward(self, x):
        # x shape: [batch_size * num_stocks, hidden_dim]
        output = self.fc(x)
        output = output.view(-1, self.pred_steps, self.output_dim)
        return output

class Stockformer(nn.Module):
    """Stockformer model for stock prediction."""
    
    def __init__(self, input_dim, hidden_dim, num_stocks, pred_steps=PREDICT_WINDOW_SIZE,
                 num_heads=NUM_HEADS, num_layers=NUM_LAYERS, dropout=DROPOUT_RATE):
        super(Stockformer, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_stocks = num_stocks
        self.pred_steps = pred_steps
        
        # Decoupling Flow Layer
        self.fc_low = nn.Linear(input_dim, hidden_dim)
        self.fc_high = nn.Linear(input_dim, hidden_dim)
        
        # Dual-Frequency Spatiotemporal Encoder
        self.temporal_attn_low = nn.ModuleList([
            TemporalAttention(hidden_dim, hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        self.dilated_conv_high = nn.ModuleList([
            DilatedCausalConv(hidden_dim, hidden_dim, kernel_size=2, dilation=2**i)
            for i in range(num_layers)
        ])
        
        self.graph_attn_low = nn.ModuleList([
            GraphAttention(hidden_dim, hidden_dim, dropout)
            for _ in range(num_layers)
        ])
        
        self.graph_attn_high = nn.ModuleList([
            GraphAttention(hidden_dim, hidden_dim, dropout)
            for _ in range(num_layers)
        ])
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(hidden_dim)
        
        # Dual-Frequency Fusion Decoder
        self.predictor_low = DualFrequencyPredictor(hidden_dim, hidden_dim, pred_steps)
        self.predictor_high = DualFrequencyPredictor(hidden_dim, hidden_dim, pred_steps)
        
        self.fusion_attn = FusionAttention(hidden_dim, num_heads, dropout)
        
        # Output layers
        self.fc_reg = nn.Linear(hidden_dim, 1)  # Regression (returns)
        self.fc_cls = nn.Linear(hidden_dim, 1)  # Classification (trend)
        
        # Low frequency specific output layers for multi-supervision
        self.fc_low_reg = nn.Linear(hidden_dim, 1)
        self.fc_low_cls = nn.Linear(hidden_dim, 1)
        
    def forward(self, x_low, x_high):
        # x_low, x_high shape: [batch_size, seq_len, num_stocks, input_dim]
        batch_size, seq_len, num_stocks, _ = x_low.shape
        
        # Reshape for processing
        x_low = x_low.permute(1, 0, 2, 3).reshape(seq_len, batch_size * num_stocks, self.input_dim)
        x_high = x_high.permute(1, 0, 2, 3).reshape(seq_len, batch_size * num_stocks, self.input_dim)
        
        # Decoupling Flow Layer
        x_low = self.fc_low(x_low)
        x_high = self.fc_high(x_high)
        
        # Add positional encoding
        x_low = self.pos_encoder(x_low)
        x_high = self.pos_encoder(x_high)
        
        # Dual-Frequency Spatiotemporal Encoder
        for i in range(len(self.temporal_attn_low)):
            # Temporal attention for low frequency
            x_low = self.temporal_attn_low[i](x_low)
            
            # Dilated causal convolution for high frequency
            x_high_conv = x_high.permute(1, 2, 0)  # [batch*stocks, hidden_dim, seq_len]
            x_high_conv = self.dilated_conv_high[i](x_high_conv)
            x_high_conv = x_high_conv.permute(2, 0, 1)  # [seq_len, batch*stocks, hidden_dim]
            x_high = x_high + x_high_conv
            
            # Reshape for graph attention
            x_low_graph = x_low.view(seq_len, batch_size, num_stocks, self.hidden_dim)
            x_high_graph = x_high.view(seq_len, batch_size, num_stocks, self.hidden_dim)
            
            # Graph attention
            x_low_graph = self.graph_attn_low[i](x_low_graph)
            x_high_graph = self.graph_attn_high[i](x_high_graph)
            
            # Reshape back
            x_low = x_low_graph.view(seq_len, batch_size * num_stocks, self.hidden_dim)
            x_high = x_high_graph.view(seq_len, batch_size * num_stocks, self.hidden_dim)
        
        # Get last sequence representation
        last_low = x_low[-1]  # [batch_size * num_stocks, hidden_dim]
        last_high = x_high[-1]  # [batch_size * num_stocks, hidden_dim]
        
        # Dual-Frequency Fusion Decoder
        pred_low = self.predictor_low(last_low)  # [batch_size * num_stocks, pred_steps, hidden_dim]
        pred_high = self.predictor_high(last_high)  # [batch_size * num_stocks, pred_steps, hidden_dim]
        
        # Reshape for fusion attention
        pred_low = pred_low.permute(1, 0, 2)  # [pred_steps, batch_size * num_stocks, hidden_dim]
        pred_high = pred_high.permute(1, 0, 2)  # [pred_steps, batch_size * num_stocks, hidden_dim]
        
        # Fusion attention
        fused = self.fusion_attn(pred_low, pred_high)  # [pred_steps, batch_size * num_stocks, hidden_dim]
        
        # Output layers
        fused = fused.permute(1, 0, 2)  # [batch_size * num_stocks, pred_steps, hidden_dim]
        pred_low = pred_low.permute(1, 0, 2)  # [batch_size * num_stocks, pred_steps, hidden_dim]
        
        reg_out = self.fc_reg(fused).squeeze(-1)  # [batch_size * num_stocks, pred_steps]
        cls_out = torch.sigmoid(self.fc_cls(fused)).squeeze(-1)  # [batch_size * num_stocks, pred_steps]
        
        # Low frequency specific outputs for multi-supervision
        low_reg_out = self.fc_low_reg(pred_low).squeeze(-1)  # [batch_size * num_stocks, pred_steps]
        low_cls_out = torch.sigmoid(self.fc_low_cls(pred_low)).squeeze(-1)  # [batch_size * num_stocks, pred_steps]
        
        # Reshape outputs
        reg_out = reg_out.view(batch_size, num_stocks, self.pred_steps)
        cls_out = cls_out.view(batch_size, num_stocks, self.pred_steps)
        low_reg_out = low_reg_out.view(batch_size, num_stocks, self.pred_steps)
        low_cls_out = low_cls_out.view(batch_size, num_stocks, self.pred_steps)
        
        # Permute to match target shape [batch_size, pred_steps, num_stocks]
        reg_out = reg_out.permute(0, 2, 1)
        cls_out = cls_out.permute(0, 2, 1)
        low_reg_out = low_reg_out.permute(0, 2, 1)
        low_cls_out = low_cls_out.permute(0, 2, 1)
        
        return reg_out, cls_out, low_reg_out, low_cls_out

def train_model(model, train_loader, val_loader, epochs, criterion_reg, criterion_cls, optimizer, lambda_cls=2.0):
    """Train the Stockformer model."""
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        reg_loss_sum = 0
        cls_loss_sum = 0
        
        for x_low, x_high, y_reg, y_cls in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            x_low, x_high = x_low.to(device), x_high.to(device)
            y_reg, y_cls = y_reg.to(device), y_cls.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            reg_out, cls_out, low_reg_out, low_cls_out = model(x_low, x_high)
            
            # Calculate losses
            reg_loss = (criterion_reg(reg_out, y_reg) + criterion_reg(low_reg_out, y_reg)) / 2
            cls_loss = (criterion_cls(cls_out, y_cls) + criterion_cls(low_cls_out, y_cls)) / 2
            
            # Combine losses
            loss = reg_loss + lambda_cls * cls_loss
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            reg_loss_sum += reg_loss.item()
            cls_loss_sum += cls_loss.item()
        
        # Calculate average training loss
        avg_train_loss = total_loss / len(train_loader)
        avg_reg_loss = reg_loss_sum / len(train_loader)
        avg_cls_loss = cls_loss_sum / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        val_loss = 0
        val_reg_loss = 0
        val_cls_loss = 0
        
        with torch.no_grad():
            for x_low, x_high, y_reg, y_cls in val_loader:
                x_low, x_high = x_low.to(device), x_high.to(device)
                y_reg, y_cls = y_reg.to(device), y_cls.to(device)
                
                # Forward pass
                reg_out, cls_out, low_reg_out, low_cls_out = model(x_low, x_high)
                
                # Calculate losses
                reg_loss = (criterion_reg(reg_out, y_reg) + criterion_reg(low_reg_out, y_reg)) / 2
                cls_loss = (criterion_cls(cls_out, y_cls) + criterion_cls(low_cls_out, y_cls)) / 2
                
                # Combine losses
                loss = reg_loss + lambda_cls * cls_loss
                
                val_loss += loss.item()
                val_reg_loss += reg_loss.item()
                val_cls_loss += cls_loss.item()
        
        # Calculate average validation loss
        avg_val_loss = val_loss / len(val_loader)
        avg_val_reg_loss = val_reg_loss / len(val_loader)
        avg_val_cls_loss = val_cls_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        print(f"Epoch {epoch+1}/{epochs}, "
              f"Train Loss: {avg_train_loss:.4f} (Reg: {avg_reg_loss:.4f}, Cls: {avg_cls_loss:.4f}), "
              f"Val Loss: {avg_val_loss:.4f} (Reg: {avg_val_reg_loss:.4f}, Cls: {avg_val_cls_loss:.4f})")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_stockformer.pth')
            print("Model saved!")
    
    return train_losses, val_losses

def evaluate_model(model, test_loader):
    """Evaluate the trained model on the test set."""
    model.eval()
    test_reg_loss = 0
    test_cls_loss = 0
    
    all_reg_preds = []
    all_cls_preds = []
    all_reg_targets = []
    all_cls_targets = []
    
    criterion_reg = nn.L1Loss()
    criterion_cls = nn.BCELoss()
    
    with torch.no_grad():
        for x_low, x_high, y_reg, y_cls in test_loader:
            x_low, x_high = x_low.to(device), x_high.to(device)
            y_reg, y_cls = y_reg.to(device), y_cls.to(device)
            
            # Forward pass
            reg_out, cls_out, _, _ = model(x_low, x_high)
            
            # Calculate losses
            reg_loss = criterion_reg(reg_out, y_reg)
            cls_loss = criterion_cls(cls_out, y_cls)
            
            test_reg_loss += reg_loss.item()
            test_cls_loss += cls_loss.item()
            
            # Store predictions and targets for metrics calculation
            all_reg_preds.append(reg_out.cpu().numpy())
            all_cls_preds.append(cls_out.cpu().numpy())
            all_reg_targets.append(y_reg.cpu().numpy())
            all_cls_targets.append(y_cls.cpu().numpy())
    
    # Calculate average test losses
    avg_test_reg_loss = test_reg_loss / len(test_loader)
    avg_test_cls_loss = test_cls_loss / len(test_loader)
    
    # Concatenate all predictions and targets
    all_reg_preds = np.concatenate(all_reg_preds, axis=0)
    all_cls_preds = np.concatenate(all_cls_preds, axis=0)
    all_reg_targets = np.concatenate(all_reg_targets, axis=0)
    all_cls_targets = np.concatenate(all_cls_targets, axis=0)
    
    # Calculate IC and RankIC
    ic_values = []
    rank_ic_values = []
    
    for t in range(all_reg_preds.shape[1]):
        for s in range(all_reg_preds.shape[2]):
            ic = np.corrcoef(all_reg_preds[:, t, s], all_reg_targets[:, t, s])[0, 1]
            rank_ic = stats.spearmanr(all_reg_preds[:, t, s], all_reg_targets[:, t, s])[0]
            
            if not np.isnan(ic):
                ic_values.append(ic)
            if not np.isnan(rank_ic):
                rank_ic_values.append(rank_ic)
    
    ic = np.mean(ic_values)
    rank_ic = np.mean(rank_ic_values)
    icir = ic / np.std(ic_values) if len(ic_values) > 1 else 0
    rank_icir = rank_ic / np.std(rank_ic_values) if len(rank_ic_values) > 1 else 0
    
    # Calculate directional accuracy
    cls_preds_binary = (all_cls_preds > 0.5).astype(int)
    cls_targets_binary = all_cls_targets.astype(int)
    
    directional_accuracy = np.mean(cls_preds_binary == cls_targets_binary)
    
    results = {
        'reg_loss': avg_test_reg_loss,
        'cls_loss': avg_test_cls_loss,
        'IC': ic,
        'RankIC': rank_ic,
        'ICIR': icir,
        'RankICIR': rank_icir,
        'DirectionalAccuracy': directional_accuracy
    }
    
    return results, (all_reg_preds, all_cls_preds, all_reg_targets, all_cls_targets)

class LSTMBaseline(nn.Module):
    """LSTM baseline model for comparison."""
    
    def __init__(self, input_dim, hidden_dim, num_layers, num_stocks, pred_steps=PREDICT_WINDOW_SIZE):
        super(LSTMBaseline, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_stocks = num_stocks
        self.pred_steps = pred_steps
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc_reg = nn.Linear(hidden_dim, pred_steps)
        self.fc_cls = nn.Linear(hidden_dim, pred_steps)
        
    def forward(self, x):
        # x shape: [batch_size, seq_len, num_stocks, input_dim]
        batch_size, seq_len, num_stocks, _ = x.shape
        
        # Process each stock separately
        reg_outputs = []
        cls_outputs = []
        
        for i in range(num_stocks):
            # Extract features for current stock
            stock_features = x[:, :, i, :]  # [batch_size, seq_len, input_dim]
            
            # LSTM forward pass
            lstm_out, _ = self.lstm(stock_features)
            
            # Get last hidden state
            last_hidden = lstm_out[:, -1, :]  # [batch_size, hidden_dim]
            
            # Predict returns and trend
            reg_out = self.fc_reg(last_hidden)  # [batch_size, pred_steps]
            cls_out = torch.sigmoid(self.fc_cls(last_hidden))  # [batch_size, pred_steps]
            
            reg_outputs.append(reg_out)
            cls_outputs.append(cls_out)
        
        # Stack outputs
        reg_outputs = torch.stack(reg_outputs, dim=2)  # [batch_size, pred_steps, num_stocks]
        cls_outputs = torch.stack(cls_outputs, dim=2)  # [batch_size, pred_steps, num_stocks]
        
        return reg_outputs, cls_outputs

class BacktestStrategy:
    """Backtest a trading strategy using model predictions."""
    
    def __init__(self, prices, returns, predictions, start_capital=1000000, topk=5, drop=3):
        self.prices = prices
        self.returns = returns
        self.predictions = predictions  # Can be return predictions or class predictions
        self.start_capital = start_capital
        self.topk = topk
        self.drop = drop
        
        self.num_days = len(prices)
        self.num_stocks = prices.shape[1]
        
        self.positions = np.zeros((self.num_days, self.num_stocks))
        self.portfolio_value = np.zeros(self.num_days)
        self.portfolio_value[0] = start_capital
        
    def run(self, use_cls=False, transaction_cost=0.001):
        """Run the backtesting simulation."""
        # Initial portfolio has no positions
        current_holdings = set()
        
        for t in range(1, self.num_days):
            if t >= len(self.predictions):
                # If we've run out of predictions, hold the last positions
                self.positions[t] = self.positions[t-1]
                continue
                
            # Get predictions for the next day
            if use_cls:
                scores = self.predictions[t-1, 0]  # Use classification probabilities
            else:
                scores = self.predictions[t-1, 0]  # Use return predictions
            
            # Rank stocks based on scores
            ranked_stocks = np.argsort(scores)[::-1]
            
            # Apply TopK-Dropout strategy
            stocks_to_keep = set()
            
            # Keep stocks that are in the current holdings and still in top rankings
            for stock in current_holdings:
                rank = np.where(ranked_stocks == stock)[0][0]
                if rank < self.topk:
                    stocks_to_keep.add(stock)
            
            # Calculate how many new stocks to buy
            num_to_buy = self.topk - len(stocks_to_keep)
            
            # Find new stocks to buy
            for stock in ranked_stocks:
                if len(stocks_to_keep) >= self.topk:
                    break
                if stock not in stocks_to_keep and stock not in current_holdings:
                    stocks_to_keep.add(stock)
            
            # Update holdings
            to_sell = current_holdings - stocks_to_keep
            to_buy = stocks_to_keep - current_holdings
            
            # Calculate available capital from selling and existing cash
            cash = self.portfolio_value[t-1]
            for stock in to_sell:
                cash += self.positions[t-1, stock] * (1 - transaction_cost)
            
            # Set positions for day t
            self.positions[t] = self.positions[t-1].copy()
            
            # Sell stocks
            for stock in to_sell:
                self.positions[t, stock] = 0
            
            # Buy stocks with equal weight
            if to_buy:
                position_value = cash / len(to_buy) * (1 - transaction_cost)
                for stock in to_buy:
                    self.positions[t, stock] = position_value
            
            # Update current holdings
            current_holdings = stocks_to_keep
            
            # Calculate portfolio value for day t
            self.portfolio_value[t] = np.sum(self.positions[t] * (1 + self.returns[t]))
        
        return self.calculate_metrics()
    
    def calculate_metrics(self):
        """Calculate performance metrics for the backtest."""
        daily_returns = np.diff(self.portfolio_value) / self.portfolio_value[:-1]
        
        # Annualized return
        total_return = (self.portfolio_value[-1] / self.portfolio_value[0]) - 1
        years = self.num_days / 252  # Assuming 252 trading days per year
        annualized_return = (1 + total_return) ** (1 / years) - 1
        
        # Annualized volatility
        annualized_volatility = np.std(daily_returns) * np.sqrt(252)
        
        # Maximum drawdown
        cumulative_returns = self.portfolio_value / self.portfolio_value[0]
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (running_max - cumulative_returns) / running_max
        max_drawdown = np.max(drawdown)
        
        # Sharpe ratio (assuming risk-free rate of 0)
        sharpe_ratio = annualized_return / annualized_volatility if annualized_volatility > 0 else 0
        
        metrics = {
            'AnnualizedReturn': annualized_return,
            'AnnualizedVolatility': annualized_volatility,
            'MaxDrawdown': max_drawdown,
            'SharpeRatio': sharpe_ratio,
            'FinalValue': self.portfolio_value[-1]
        }
        
        return metrics, daily_returns, cumulative_returns

def simulate_market_conditions(base_returns, num_days=100, condition='uptrend', volatility=0.01):
    """Simulate different market conditions."""
    num_stocks = base_returns.shape[1]
    
    if condition == 'uptrend':
        # Uptrend: positive drift
        drift = np.random.uniform(0.0005, 0.002, num_stocks)
    elif condition == 'downtrend':
        # Downtrend: negative drift
        drift = np.random.uniform(-0.002, -0.0005, num_stocks)
    else:  # fluctuation
        # Fluctuation: mixed drift around zero
        drift = np.random.uniform(-0.0005, 0.0005, num_stocks)
    
    # Generate returns
    returns = np.zeros((num_days, num_stocks))
    for i in range(num_stocks):
        returns[:, i] = drift[i] + np.random.normal(0, volatility, num_days)
    
    # Generate prices
    prices = np.zeros((num_days, num_stocks))
    prices[0] = 100  # Initial price
    for t in range(1, num_days):
        prices[t] = prices[t-1] * (1 + returns[t-1])
    
    return prices, returns

def main():
    # Generate synthetic data
    print("Generating synthetic stock market data...")
    data_generator = StockDataGenerator(num_stocks=NUM_STOCKS, num_factors=NUM_FACTORS, num_days=1000)
    data = data_generator.generate_data()
    
    # Split data
    train_data, val_data, test_data = data_generator.split_data(data)
    
    # Prepare sequences
    X_train, y_reg_train, y_cls_train = data_generator.prepare_sequences(train_data)
    X_val, y_reg_val, y_cls_val = data_generator.prepare_sequences(val_data)
    X_test, y_reg_test, y_cls_test = data_generator.prepare_sequences(test_data)
    
    print(f"Training set: {X_train.shape}, Validation set: {X_val.shape}, Test set: {X_test.shape}")
    
    # Create datasets with wavelet transform
    wavelet_transform = WaveletTransform(wavelet='db1', level=1)
    
    train_dataset = StockDataset(X_train, y_reg_train, y_cls_train, transform=wavelet_transform)
    val_dataset = StockDataset(X_val, y_reg_val, y_cls_val, transform=wavelet_transform)
    test_dataset = StockDataset(X_test, y_reg_test, y_cls_test, transform=wavelet_transform)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    
    # Create model
    model = Stockformer(
        input_dim=INPUT_DIM,
        hidden_dim=HIDDEN_DIM,
        num_stocks=NUM_STOCKS,
        pred_steps=PREDICT_WINDOW_SIZE,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT_RATE
    ).to(device)
    
    # Define loss functions
    criterion_reg = nn.L1Loss()
    criterion_cls = nn.BCELoss()
    
    # Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Train model
    print("Training Stockformer model...")
    train_losses, val_losses = train_model(
        model, train_loader, val_loader, NUM_EPOCHS, criterion_reg, criterion_cls, optimizer, lambda_cls=2.0
    )
    
    # Load best model
    model.load_state_dict(torch.load('best_stockformer.pth'))
    
    # Evaluate model
    print("Evaluating Stockformer model...")
    stockformer_results, stockformer_preds = evaluate_model(model, test_loader)
    
    print("Stockformer Results:")
    for metric, value in stockformer_results.items():
        print(f"{metric}: {value:.4f}")
    
    # Train and evaluate LSTM baseline
    print("\nTraining LSTM baseline model...")
    lstm_model = LSTMBaseline(
        input_dim=INPUT_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        num_stocks=NUM_STOCKS,
        pred_steps=PREDICT_WINDOW_SIZE
    ).to(device)
    
    # Define optimizer for LSTM
    lstm_optimizer = optim.Adam(lstm_model.parameters(), lr=LEARNING_RATE)
    
    # Training loop for LSTM
    best_val_loss = float('inf')
    for epoch in range(NUM_EPOCHS):
        lstm_model.train()
        total_loss = 0
        
        for x_low, _, y_reg, y_cls in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}"):
            x_low = x_low.to(device)
            y_reg, y_cls = y_reg.to(device), y_cls.to(device)
            
            lstm_optimizer.zero_grad()
            
            # Forward pass
            reg_out, cls_out = lstm_model(x_low)
            
            # Calculate losses
            reg_loss = criterion_reg(reg_out, y_reg)
            cls_loss = criterion_cls(cls_out, y_cls)
            
            # Combine losses
            loss = reg_loss + 2.0 * cls_loss
            
            # Backward pass and optimize
            loss.backward()
            lstm_optimizer.step()
            
            total_loss += loss.item()
        
        # Validation
        lstm_model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for x_low, _, y_reg, y_cls in val_loader:
                x_low = x_low.to(device)
                y_reg, y_cls = y_reg.to(device), y_cls.to(device)
                
                # Forward pass
                reg_out, cls_out = lstm_model(x_low)
                
                # Calculate losses
                reg_loss = criterion_reg(reg_out, y_reg)
                cls_loss = criterion_cls(cls_out, y_cls)
                
                # Combine losses
                loss = reg_loss + 2.0 * cls_loss
                
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}, "
              f"Train Loss: {total_loss/len(train_loader):.4f}, "
              f"Val Loss: {avg_val_loss:.4f}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(lstm_model.state_dict(), 'best_lstm.pth')
            print("LSTM model saved!")
    
    # Load best LSTM model
    lstm_model.load_state_dict(torch.load('best_lstm.pth'))
    
    # Evaluate LSTM model
    print("Evaluating LSTM baseline model...")
    lstm_model.eval()
    lstm_reg_preds = []
    lstm_cls_preds = []
    
    with torch.no_grad():
        for x_low, _, y_reg, y_cls in test_loader:
            x_low = x_low.to(device)
            
            # Forward pass
            reg_out, cls_out = lstm_model(x_low)
            
            lstm_reg_preds.append(reg_out.cpu().numpy())
            lstm_cls_preds.append(cls_out.cpu().numpy())
    
    lstm_reg_preds = np.concatenate(lstm_reg_preds, axis=0)
    lstm_cls_preds = np.concatenate(lstm_cls_preds, axis=0)
    
    # Calculate LSTM metrics
    lstm_results = {}
    
    # Calculate IC and RankIC for LSTM
    ic_values = []
    rank_ic_values = []
    
    for t in range(lstm_reg_preds.shape[1]):
        for s in range(lstm_reg_preds.shape[2]):
            ic = np.corrcoef(lstm_reg_preds[:, t, s], y_reg_test[len(lstm_reg_preds)-len(y_reg_test):, t, s])[0, 1]
            rank_ic = stats.spearmanr(lstm_reg_preds[:, t, s], y_reg_test[len(lstm_reg_preds)-len(y_reg_test):, t, s])[0]
            
            if not np.isnan(ic):
                ic_values.append(ic)
            if not np.isnan(rank_ic):
                rank_ic_values.append(rank_ic)
    
    lstm_results['IC'] = np.mean(ic_values)
    lstm_results['RankIC'] = np.mean(rank_ic_values)
    lstm_results['ICIR'] = lstm_results['IC'] / np.std(ic_values) if len(ic_values) > 1 else 0
    lstm_results['RankICIR'] = lstm_results['RankIC'] / np.std(rank_ic_values) if len(rank_ic_values) > 1 else 0
    
    # Calculate directional accuracy for LSTM
    lstm_cls_preds_binary = (lstm_cls_preds > 0.5).astype(int)
    lstm_cls_targets = y_cls_test[len(lstm_cls_preds)-len(y_cls_test):].astype(int)
    
    lstm_results['DirectionalAccuracy'] = np.mean(lstm_cls_preds_binary == lstm_cls_targets)
    
    print("LSTM Baseline Results:")
    for metric, value in lstm_results.items():
        print(f"{metric}: {value:.4f}")
    
    # Compare models
    print("\nModel Comparison:")
    comparison = pd.DataFrame({
        'Stockformer': [
            stockformer_results['IC'],
            stockformer_results['RankIC'],
            stockformer_results['ICIR'],
            stockformer_results['RankICIR'],
            stockformer_results['DirectionalAccuracy']
        ],
        'LSTM': [
            lstm_results['IC'],
            lstm_results['RankIC'],
            lstm_results['ICIR'],
            lstm_results['RankICIR'],
            lstm_results['DirectionalAccuracy']
        ]
    }, index=['IC', 'RankIC', 'ICIR', 'RankICIR', 'DirectionalAccuracy'])
    
    print(comparison)
    
    # Backtesting in different market conditions
    print("\nBacktesting in different market conditions...")
    
    # Generate data for different market conditions
    uptrend_prices, uptrend_returns = simulate_market_conditions(
        test_data['returns'], num_days=100, condition='uptrend'
    )
    downtrend_prices, downtrend_returns = simulate_market_conditions(
        test_data['returns'], num_days=100, condition='downtrend'
    )
    fluctuation_prices, fluctuation_returns = simulate_market_conditions(
        test_data['returns'], num_days=100, condition='fluctuation'
    )
    
    # Run backtests
    # Use a subset of predictions for backtesting
    stockformer_reg_preds = stockformer_preds[0][:100]
    stockformer_cls_preds = stockformer_preds[1][:100]
    lstm_reg_preds = lstm_reg_preds[:100]
    lstm_cls_preds = lstm_cls_preds[:100]
    
    # Uptrend market
    print("\nUptrend Market:")
    stockformer_reg_backtest = BacktestStrategy(uptrend_prices, uptrend_returns, stockformer_reg_preds)
    stockformer_reg_metrics, _, _ = stockformer_reg_backtest.run(use_cls=False)
    
    stockformer_cls_backtest = BacktestStrategy(uptrend_prices, uptrend_returns, stockformer_cls_preds)
    stockformer_cls_metrics, _, _ = stockformer_cls_backtest.run(use_cls=True)
    
    lstm_reg_backtest = BacktestStrategy(uptrend_prices, uptrend_returns, lstm_reg_preds)
    lstm_reg_metrics, _, _ = lstm_reg_backtest.run(use_cls=False)
    
    lstm_cls_backtest = BacktestStrategy(uptrend_prices, uptrend_returns, lstm_cls_preds)
    lstm_cls_metrics, _, _ = lstm_cls_backtest.run(use_cls=True)
    
    uptrend_results = pd.DataFrame({
        'Stockformer (Reg)': [
            stockformer_reg_metrics['AnnualizedReturn'] * 100,
            stockformer_reg_metrics['AnnualizedVolatility'] * 100,
            stockformer_reg_metrics['MaxDrawdown'] * 100,
            stockformer_reg_metrics['SharpeRatio']
        ],
        'Stockformer (Cls)': [
            stockformer_cls_metrics['AnnualizedReturn'] * 100,
            stockformer_cls_metrics['AnnualizedVolatility'] * 100,
            stockformer_cls_metrics['MaxDrawdown'] * 100,
            stockformer_cls_metrics['SharpeRatio']
        ],
        'LSTM (Reg)': [
            lstm_reg_metrics['AnnualizedReturn'] * 100,
            lstm_reg_metrics['AnnualizedVolatility'] * 100,
            lstm_reg_metrics['MaxDrawdown'] * 100,
            lstm_reg_metrics['SharpeRatio']
        ],
        'LSTM (Cls)': [
            lstm_cls_metrics['AnnualizedReturn'] * 100,
            lstm_cls_metrics['AnnualizedVolatility'] * 100,
            lstm_cls_metrics['MaxDrawdown'] * 100,
            lstm_cls_metrics['SharpeRatio']
        ]
    }, index=['Annualized Return (%)', 'Annualized Volatility (%)', 'Maximum Drawdown (%)', 'Sharpe Ratio'])
    
    print(uptrend_results)
    
    # Downtrend market
    print("\nDowntrend Market:")
    stockformer_reg_backtest = BacktestStrategy(downtrend_prices, downtrend_returns, stockformer_reg_preds)
    stockformer_reg_metrics, _, _ = stockformer_reg_backtest.run(use_cls=False)
    
    stockformer_cls_backtest = BacktestStrategy(downtrend_prices, downtrend_returns, stockformer_cls_preds)
    stockformer_cls_metrics, _, _ = stockformer_cls_backtest.run(use_cls=True)
    
    lstm_reg_backtest = BacktestStrategy(downtrend_prices, downtrend_returns, lstm_reg_preds)
    lstm_reg_metrics, _, _ = lstm_reg_backtest.run(use_cls=False)
    
    lstm_cls_backtest = BacktestStrategy(downtrend_prices, downtrend_returns, lstm_cls_preds)
    lstm_cls_metrics, _, _ = lstm_cls_backtest.run(use_cls=True)
    
    downtrend_results = pd.DataFrame({
        'Stockformer (Reg)': [
            stockformer_reg_metrics['AnnualizedReturn'] * 100,
            stockformer_reg_metrics['AnnualizedVolatility'] * 100,
            stockformer_reg_metrics['MaxDrawdown'] * 100,
            stockformer_reg_metrics['SharpeRatio']
        ],
        'Stockformer (Cls)': [
            stockformer_cls_metrics['AnnualizedReturn'] * 100,
            stockformer_cls_metrics['AnnualizedVolatility'] * 100,
            stockformer_cls_metrics['MaxDrawdown'] * 100,
            stockformer_cls_metrics['SharpeRatio']
        ],
        'LSTM (Reg)': [
            lstm_reg_metrics['AnnualizedReturn'] * 100,
            lstm_reg_metrics['AnnualizedVolatility'] * 100,
            lstm_reg_metrics['MaxDrawdown'] * 100,
            lstm_reg_metrics['SharpeRatio']
        ],
        'LSTM (Cls)': [
            lstm_cls_metrics['AnnualizedReturn'] * 100,
            lstm_cls_metrics['AnnualizedVolatility'] * 100,
            lstm_cls_metrics['MaxDrawdown'] * 100,
            lstm_cls_metrics['SharpeRatio']
        ]
    }, index=['Annualized Return (%)', 'Annualized Volatility (%)', 'Maximum Drawdown (%)', 'Sharpe Ratio'])
    
    print(downtrend_results)
    
    # Fluctuation market
    print("\nFluctuation Market:")
    stockformer_reg_backtest = BacktestStrategy(fluctuation_prices, fluctuation_returns, stockformer_reg_preds)
    stockformer_reg_metrics, _, _ = stockformer_reg_backtest.run(use_cls=False)
    
    stockformer_cls_backtest = BacktestStrategy(fluctuation_prices, fluctuation_returns, stockformer_cls_preds)
    stockformer_cls_metrics, _, _ = stockformer_cls_backtest.run(use_cls=True)
    
    lstm_reg_backtest = BacktestStrategy(fluctuation_prices, fluctuation_returns, lstm_reg_preds)
    lstm_reg_metrics, _, _ = lstm_reg_backtest.run(use_cls=False)
    
    lstm_cls_backtest = BacktestStrategy(fluctuation_prices, fluctuation_returns, lstm_cls_preds)
    lstm_cls_metrics, _, _ = lstm_cls_backtest.run(use_cls=True)
    
    fluctuation_results = pd.DataFrame({
        'Stockformer (Reg)': [
            stockformer_reg_metrics['AnnualizedReturn'] * 100,
            stockformer_reg_metrics['AnnualizedVolatility'] * 100,
            stockformer_reg_metrics['MaxDrawdown'] * 100,
            stockformer_reg_metrics['SharpeRatio']
        ],
        'Stockformer (Cls)': [
            stockformer_cls_metrics['AnnualizedReturn'] * 100,
            stockformer_cls_metrics['AnnualizedVolatility'] * 100,
            stockformer_cls_metrics['MaxDrawdown'] * 100,
            stockformer_cls_metrics['SharpeRatio']
        ],
        'LSTM (Reg)': [
            lstm_reg_metrics['AnnualizedReturn'] * 100,
            lstm_reg_metrics['AnnualizedVolatility'] * 100,
            lstm_reg_metrics['MaxDrawdown'] * 100,
            lstm_reg_metrics['SharpeRatio']
        ],
        'LSTM (Cls)': [
            lstm_cls_metrics['AnnualizedReturn'] * 100,
            lstm_cls_metrics['AnnualizedVolatility'] * 100,
            lstm_cls_metrics['MaxDrawdown'] * 100,
            lstm_cls_metrics['SharpeRatio']
        ]
    }, index=['Annualized Return (%)', 'Annualized Volatility (%)', 'Maximum Drawdown (%)', 'Sharpe Ratio'])
    
    print(fluctuation_results)
    
    # Plot backtest results
    stockformer_reg_backtest = BacktestStrategy(uptrend_prices, uptrend_returns, stockformer_reg_preds)
    _, _, stockformer_reg_cum_returns = stockformer_reg_backtest.run(use_cls=False)
    
    stockformer_cls_backtest = BacktestStrategy(uptrend_prices, uptrend_returns, stockformer_cls_preds)
    _, _, stockformer_cls_cum_returns = stockformer_cls_backtest.run(use_cls=True)
    
    lstm_reg_backtest = BacktestStrategy(uptrend_prices, uptrend_returns, lstm_reg_preds)
    _, _, lstm_reg_cum_returns = lstm_reg_backtest.run(use_cls=False)
    
    lstm_cls_backtest = BacktestStrategy(uptrend_prices, uptrend_returns, lstm_cls_preds)
    _, _, lstm_cls_cum_returns = lstm_cls_backtest.run(use_cls=True)
    
    # Benchmark: equal-weighted portfolio
    equal_weighted_positions = np.ones((len(uptrend_prices), NUM_STOCKS)) / NUM_STOCKS
    equal_weighted_backtest = BacktestStrategy(uptrend_prices, uptrend_returns, None)
    equal_weighted_backtest.positions = equal_weighted_positions
    
    # Calculate portfolio value
    equal_weighted_backtest.portfolio_value[0] = 1000000
    for t in range(1, len(uptrend_prices)):
        equal_weighted_backtest.portfolio_value[t] = np.sum(
            equal_weighted_backtest.positions[t-1] * (1 + uptrend_returns[t])
        )
    
    # Calculate cumulative returns
    benchmark_cum_returns = equal_weighted_backtest.portfolio_value / equal_weighted_backtest.portfolio_value[0]
    
    plt.figure(figsize=(12, 6))
    plt.plot(stockformer_reg_cum_returns, label='Stockformer (Reg)')
    plt.plot(stockformer_cls_cum_returns, label='Stockformer (Cls)')
    plt.plot(lstm_reg_cum_returns, label='LSTM (Reg)')
    plt.plot(lstm_cls_cum_returns, label='LSTM (Cls)')
    plt.plot(benchmark_cum_returns, label='Equal-Weighted Benchmark')
    plt.title('Cumulative Returns in Uptrend Market')
    plt.xlabel('Days')
    plt.ylabel('Cumulative Return')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('cumulative_returns.png')
    plt.show()

if __name__ == "__main__":
    main()