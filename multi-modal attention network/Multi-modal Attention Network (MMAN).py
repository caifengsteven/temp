import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pdblp
from scipy.stats import pearsonr
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
from tqdm import tqdm
warnings.filterwarnings('ignore')

print("Connecting to Bloomberg...")
con = pdblp.BCon(timeout=10000)
con.start()

# Define custom dataset
class StockPredictionDataset(Dataset):
    def __init__(self, text_data, social_data, historical_data, labels):
        self.text_data = torch.FloatTensor(text_data)
        self.social_data = torch.FloatTensor(social_data)
        self.historical_data = torch.FloatTensor(historical_data)
        self.labels = torch.FloatTensor(labels).unsqueeze(1)
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return {
            'text': self.text_data[idx],
            'social': self.social_data[idx],
            'historical': self.historical_data[idx],
            'label': self.labels[idx]
        }

# Multi-head attention module
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)
        
    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        
        # Linear projections and reshape for multi-head
        q = self.q_linear(q).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_linear(k).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_linear(v).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Calculate attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax and get weighted values
        attention_weights = F.softmax(scores, dim=-1)
        out = torch.matmul(attention_weights, v)
        
        # Reshape and apply final linear projection
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        out = self.out_linear(out)
        
        return out, attention_weights

# Feed forward network for transformer
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        
    def forward(self, x):
        return self.linear2(F.relu(self.linear1(x)))

# Encoder layer for transformer
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        attn_output, _ = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x

# Decoder layer for transformer with social impact features
class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        # Self attention
        self_attn_output, _ = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(self_attn_output))
        
        # Cross attention (using encoder output)
        cross_attn_output, _ = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(cross_attn_output))
        
        # Feed forward
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        
        return x

# Simplified CNN for historical price data - redesigned to handle the specific dimensions
class HistoricalCNN(nn.Module):
    def __init__(self, d_model=512):
        super(HistoricalCNN, self).__init__()
        # First process the time dimension with 2D convolutions
        self.conv1 = nn.Conv2d(1, 16, kernel_size=(3, 3), padding=(1, 1))
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 1))  # Pool only along time dimension
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(3, 3), padding=(1, 1))
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 1))
        
        # Global average pooling instead of flattening
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        
        # Final projection to d_model
        self.projection = nn.Linear(32, d_model)
        
    def forward(self, x):
        # Input shape: [batch, 1, time_steps, features, 1]
        batch_size = x.size(0)
        
        # Remove the last singleton dimension and process as [batch, 1, time_steps, features]
        x = x.squeeze(-1)
        
        # Apply 2D convolutions
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        
        # Global average pooling 
        x = self.gap(x)
        
        # Reshape to [batch, channels]
        x = x.view(batch_size, 32)
        
        # Project to d_model
        x = self.projection(x)
        
        # Reshape to [batch, 1, d_model] for compatibility with text features
        x = x.unsqueeze(1)
        
        return x

# Inter-Intra Attention Module
class InterIntraAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(InterIntraAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        
        # Projection layers
        self.text_query = nn.Linear(d_model, d_model)
        self.text_key = nn.Linear(d_model, d_model)
        self.text_value = nn.Linear(d_model, d_model)
        
        self.hist_query = nn.Linear(d_model, d_model)
        self.hist_key = nn.Linear(d_model, d_model)
        self.hist_value = nn.Linear(d_model, d_model)
        
        # Inter-attention
        self.text_hist_attn = MultiHeadAttention(d_model, num_heads)
        self.hist_text_attn = MultiHeadAttention(d_model, num_heads)
        
        # Gating layers
        self.text_gate = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Sigmoid()
        )
        self.hist_gate = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Sigmoid()
        )
        
        # Final projections
        self.text_proj = nn.Linear(d_model, d_model)
        self.hist_proj = nn.Linear(d_model, d_model)
        
        # Layer norms
        self.text_norm1 = nn.LayerNorm(d_model)
        self.text_norm2 = nn.LayerNorm(d_model)
        self.hist_norm1 = nn.LayerNorm(d_model)
        self.hist_norm2 = nn.LayerNorm(d_model)
        
    def forward(self, text, hist):
        batch_size = text.size(0)
        
        # Project inputs
        text_q = self.text_query(text)
        text_k = self.text_key(text)
        text_v = self.text_value(text)
        
        hist_q = self.hist_query(hist)
        hist_k = self.hist_key(hist)
        hist_v = self.hist_value(hist)
        
        # Inter-attention
        text_hist_out, _ = self.text_hist_attn(text_q, hist_k, hist_v)
        hist_text_out, _ = self.hist_text_attn(hist_q, text_k, text_v)
        
        # Apply residual connection and normalization
        text = self.text_norm1(text + text_hist_out)
        hist = self.hist_norm1(hist + hist_text_out)
        
        # Compute gates
        text_avg = torch.mean(text, dim=1)
        hist_avg = torch.mean(hist, dim=1)
        
        text_gate = self.text_gate(hist_avg).unsqueeze(1)
        hist_gate = self.hist_gate(text_avg).unsqueeze(1)
        
        # Apply gates
        gated_text = text * (1 + text_gate)
        gated_hist = hist * (1 + hist_gate)
        
        # Intra-attention (simplified by using the existing attention mechanism)
        text_intra, _ = self.text_hist_attn(gated_text, gated_text, gated_text)
        hist_intra, _ = self.hist_text_attn(gated_hist, gated_hist, gated_hist)
        
        # Final residual and norm
        text_out = self.text_norm2(gated_text + text_intra)
        hist_out = self.hist_norm2(gated_hist + hist_intra)
        
        # Element-wise product for fusion
        fusion = text_out * hist_out
        
        return fusion

# Full MMAN model
class MMAttentionNetwork(nn.Module):
    def __init__(self, d_model=512, num_heads=8, max_texts=96, max_text_length=64):
        super(MMAttentionNetwork, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        
        # Embedding layers
        self.text_embedding = nn.Linear(max_text_length, d_model)
        self.social_embedding = nn.Linear(10, d_model)  # 10 social features
        
        # Encoder for text
        self.encoder_layer = EncoderLayer(d_model, num_heads, d_model * 4)
        
        # Decoder with social impact features
        self.decoder_layer = DecoderLayer(d_model, num_heads, d_model * 4)
        
        # CNN for historical data
        self.historical_cnn = HistoricalCNN(d_model=d_model)
        
        # Inter-Intra attention
        self.inter_intra_attn = InterIntraAttention(d_model, num_heads)
        
        # Final prediction layers
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(d_model, 64)
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(64, 1)
        
        # Reconstruction for regularization
        self.reconstruction = nn.Linear(1, d_model)
        
    def forward(self, text, social, historical):
        batch_size = text.size(0)
        
        try:
            # Embedding
            text_emb = self.text_embedding(text)
            social_emb = self.social_embedding(social)
            
            # Process text with encoder
            encoded_text = self.encoder_layer(text_emb)
            
            # Process social features with decoder
            decoded_text = self.decoder_layer(social_emb, encoded_text)
            
            # Process historical data with CNN
            historical_feat = self.historical_cnn(historical)
            
            # Repeat historical features to match sequence length of text if needed
            if historical_feat.size(1) != decoded_text.size(1):
                historical_feat = historical_feat.repeat(1, decoded_text.size(1), 1)
            
            # Apply inter-intra attention
            fusion = self.inter_intra_attn(decoded_text, historical_feat)
            
            # Global average pooling
            fusion = fusion.transpose(1, 2)  # [batch, d_model, seq_len]
            pooled = self.pool(fusion).squeeze(-1)  # [batch, d_model]
            
            # Final prediction
            x = F.relu(self.fc1(pooled))
            x = self.dropout(x)
            logits = torch.sigmoid(self.fc2(x))
            
            return logits
            
        except Exception as e:
            print(f"Error in forward pass: {e}")
            print(f"Text shape: {text.shape}")
            print(f"Social shape: {social.shape}")
            print(f"Historical shape: {historical.shape}")
            raise e
    
    def compute_loss(self, outputs, labels, fusion_map):
        # Margin loss as in the paper
        m_plus = 0.9
        m_minus = 0.1
        lambda_val = 0.5
        
        y = labels
        v = outputs
        
        # L_k = Y_k * max(0, m+ - ||V_k||)² + λ(1 - Y_k) * max(0, ||V_k|| - m-)²
        positive_loss = y * torch.pow(torch.clamp(m_plus - v, min=0), 2)
        negative_loss = lambda_val * (1.0 - y) * torch.pow(torch.clamp(v - m_minus, min=0), 2)
        margin_loss = positive_loss + negative_loss
        
        # Reconstruction loss for regularization
        reconstructed = self.reconstruction(outputs)
        rec_loss = torch.mean(torch.pow(fusion_map - reconstructed.unsqueeze(1), 2))
        
        # Total loss
        total_loss = margin_loss.mean() + 0.0005 * rec_loss
        
        return total_loss

# Function to fetch stock data from Bloomberg
def fetch_stock_data(tickers, start_date, end_date, con):
    """
    Fetch stock price data from Bloomberg.
    
    Parameters:
    -----------
    tickers: list, stock tickers
    start_date: str, start date in 'YYYYMMDD' format
    end_date: str, end date in 'YYYYMMDD' format
    con: Bloomberg connection
    
    Returns:
    --------
    prices_df: DataFrame with stock prices
    volume_df: DataFrame with trading volumes
    """
    print(f"Fetching data for {len(tickers)} tickers...")
    fields = ['PX_LAST', 'PX_VOLUME', 'PX_HIGH', 'PX_LOW', 'PX_OPEN', 'VOLATILITY_10D']
    
    # Show progress
    prices = {}
    volumes = {}
    highs = {}
    lows = {}
    opens = {}
    volatilities = {}
    
    # Use progress bar for data fetching
    for ticker in tqdm(tickers, desc="Fetching Bloomberg data"):
        try:
            ticker_data = con.bdh(ticker, fields, start_date, end_date)
            
            # Extract features if data was retrieved
            if not ticker_data.empty:
                prices[ticker] = ticker_data.loc[:, (ticker, 'PX_LAST')]
                volumes[ticker] = ticker_data.loc[:, (ticker, 'PX_VOLUME')]
                highs[ticker] = ticker_data.loc[:, (ticker, 'PX_HIGH')]
                lows[ticker] = ticker_data.loc[:, (ticker, 'PX_LOW')]
                opens[ticker] = ticker_data.loc[:, (ticker, 'PX_OPEN')]
                
                if (ticker, 'VOLATILITY_10D') in ticker_data.columns:
                    volatilities[ticker] = ticker_data.loc[:, (ticker, 'VOLATILITY_10D')]
                else:
                    # Calculate simple volatility if not available
                    daily_returns = ticker_data.loc[:, (ticker, 'PX_LAST')].pct_change()
                    volatilities[ticker] = daily_returns.rolling(10).std()
                
                print(f"✓ {ticker}: {len(ticker_data)} days of data")
            else:
                print(f"✗ {ticker}: No data retrieved")
        except Exception as e:
            print(f"✗ {ticker}: Error - {str(e)}")
    
    # Create DataFrames
    if not prices:
        raise ValueError("No price data was retrieved. Check Bloomberg connection and ticker symbols.")
    
    prices_df = pd.DataFrame(prices)
    volumes_df = pd.DataFrame(volumes)
    highs_df = pd.DataFrame(highs)
    lows_df = pd.DataFrame(lows)
    opens_df = pd.DataFrame(opens)
    volatility_df = pd.DataFrame(volatilities)
    
    print(f"Successfully fetched data for {len(prices)} tickers from {min(prices_df.index)} to {max(prices_df.index)}.")
    return prices_df, volumes_df, highs_df, lows_df, opens_df, volatility_df

# Generate historical trending features
def generate_historical_features(prices_df, volumes_df, highs_df, lows_df, opens_df, volatility_df, lookback=64):
    """
    Generate historical trending features for each stock.
    
    Parameters:
    -----------
    price and volume dataframes
    lookback: int, number of days to look back
    
    Returns:
    --------
    historical_features: dictionary of arrays
    """
    print("Generating historical features...")
    tickers = prices_df.columns
    historical_features = {}
    
    for ticker in tqdm(tickers, desc="Processing historical features"):
        # Basic features
        close = prices_df[ticker].values
        volume = volumes_df[ticker].values
        high = highs_df[ticker].values
        low = lows_df[ticker].values
        open_price = opens_df[ticker].values
        
        # Calculate additional features
        returns = np.diff(close) / close[:-1]
        returns = np.insert(returns, 0, 0)
        
        # High-Low range
        hl_range = (high - low) / close
        
        # Open-Close range
        oc_range = (close - open_price) / open_price
        
        # Normalized volume
        norm_volume = volume / np.mean(volume[~np.isnan(volume)])
        
        # Volatility (from Bloomberg or calculated)
        if ticker in volatility_df.columns:
            volatility = volatility_df[ticker].values
        else:
            volatility = np.zeros_like(close)
        
        # Create feature array for each day
        features = []
        dates = []
        
        for i in range(lookback, len(close)):
            # Extract lookback window
            window_close = close[i-lookback:i]
            window_volume = norm_volume[i-lookback:i]
            window_hl_range = hl_range[i-lookback:i]
            window_oc_range = oc_range[i-lookback:i]
            window_returns = returns[i-lookback:i]
            window_volatility = volatility[i-lookback:i]
            
            # Skip if there are NaN values
            if (np.isnan(window_close).any() or np.isnan(window_volume).any() or 
                np.isnan(window_hl_range).any() or np.isnan(window_oc_range).any() or 
                np.isnan(window_returns).any() or np.isnan(window_volatility).any()):
                continue
            
            # Stack features
            feature_array = np.column_stack([
                window_close, 
                window_volume, 
                window_hl_range,
                window_oc_range,
                window_returns,
                window_volatility,
                np.ones_like(window_close)  # Bias term
            ])
            
            features.append(feature_array)
            dates.append(prices_df.index[i])
        
        if features:
            historical_features[ticker] = {
                'features': np.array(features),
                'dates': dates
            }
    
    return historical_features

# Generate mock social media and impact features (since we can't access Xueqiu)
def generate_mock_social_features(tickers, prices_df, num_texts=96, lookback=14):
    """
    Generate mock social media and impact features.
    In a real implementation, this would fetch from Xueqiu or other platforms.
    
    Parameters:
    -----------
    tickers: list of stock tickers
    prices_df: dataframe of stock prices
    num_texts: maximum number of texts per stock-date
    lookback: days to look back for texts
    
    Returns:
    --------
    social_features: dictionary of arrays
    """
    print("Generating mock social features...")
    social_features = {}
    
    for ticker in tqdm(tickers, desc="Processing social features"):
        dates = prices_df.index[lookback:]
        text_features = []
        social_impact_features = []
        date_list = []
        
        for date in dates:
            # Get past days for this window
            past_dates = prices_df.index[prices_df.index < date][-lookback:]
            
            # Skip if we don't have enough past dates
            if len(past_dates) < lookback:
                continue
            
            # Generate random number of texts (between 10 and num_texts)
            n_texts = np.random.randint(10, num_texts+1)
            
            # Generate mock text embeddings (in a real scenario, these would come from word2vec/BERT)
            # Here we're just using random values for demonstration
            mock_texts = np.random.rand(num_texts, 64)  # 64 is max_text_length
            
            # Generate mock social impact features
            # In a real scenario: [fans_count, followers_count, post_count, likes, retweets, replies, etc.]
            mock_impact = np.random.rand(num_texts, 10)  # 10 social impact features
            
            text_features.append(mock_texts)
            social_impact_features.append(mock_impact)
            date_list.append(date)
        
        if text_features:  # Only add if we have features
            social_features[ticker] = {
                'texts': np.array(text_features),
                'impact': np.array(social_impact_features),
                'dates': date_list
            }
    
    return social_features

# Prepare data for model training
def prepare_training_data(historical_features, social_features, prices_df, prediction_horizon=5, threshold=0.0075):
    """
    Prepare training data for the model.
    
    Parameters:
    -----------
    historical_features: dictionary of historical feature arrays
    social_features: dictionary of social feature arrays
    prices_df: dataframe of stock prices
    prediction_horizon: days to predict ahead
    threshold: movement threshold (±0.75% in the paper)
    
    Returns:
    --------
    train_dataset, test_dataset, test_dates, test_tickers
    """
    print("Preparing training data...")
    X_texts = []
    X_social = []
    X_historical = []
    y_labels = []
    dates = []
    tickers_list = []
    
    for ticker in tqdm(historical_features.keys(), desc="Preparing data"):
        if ticker not in social_features:
            continue
            
        hist_dates = historical_features[ticker]['dates']
        social_dates = social_features[ticker]['dates']
        
        # Find common dates
        common_dates = set([d.strftime('%Y-%m-%d') for d in hist_dates]).intersection(
            set([d.strftime('%Y-%m-%d') for d in social_dates])
        )
        
        for date_str in common_dates:
            # Convert string back to datetime for indexing
            date = pd.to_datetime(date_str)
            
            # Find indices in each dataset
            hist_idx = [i for i, d in enumerate(hist_dates) if d.strftime('%Y-%m-%d') == date_str][0]
            social_idx = [i for i, d in enumerate(social_dates) if d.strftime('%Y-%m-%d') == date_str][0]
            
            # Get feature arrays
            hist_feature = historical_features[ticker]['features'][hist_idx]
            text_feature = social_features[ticker]['texts'][social_idx]
            impact_feature = social_features[ticker]['impact'][social_idx]
            
            # Calculate future return for label
            try:
                date_idx = prices_df.index.get_loc(date)
                if date_idx + prediction_horizon >= len(prices_df):
                    continue  # Skip if prediction horizon exceeds available data
                    
                current_price = prices_df[ticker].iloc[date_idx]
                future_price = prices_df[ticker].iloc[date_idx + prediction_horizon]
                
                return_pct = (future_price - current_price) / current_price
                
                # Apply threshold as in the paper
                if abs(return_pct) < threshold:
                    continue  # Skip if movement is too small
                    
                label = 1 if return_pct > 0 else 0
                
                X_texts.append(text_feature)
                X_social.append(impact_feature)
                X_historical.append(hist_feature)
                y_labels.append(label)
                dates.append(date)
                tickers_list.append(ticker)
            except Exception as e:
                print(f"Error processing {ticker} on {date}: {e}")
    
    if not X_texts:
        raise ValueError("No valid training samples were generated. Check your data preprocessing.")
    
    # Convert to arrays
    X_texts = np.array(X_texts)
    X_social = np.array(X_social)
    X_historical = np.array(X_historical)
    y_labels = np.array(y_labels)
    
    # Split data
    indices = np.arange(len(y_labels))
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)
    
    X_train_texts = X_texts[train_idx]
    X_train_social = X_social[train_idx]
    X_train_historical = X_historical[train_idx]
    y_train = y_labels[train_idx]
    
    X_test_texts = X_texts[test_idx]
    X_test_social = X_social[test_idx]
    X_test_historical = X_historical[test_idx]
    y_test = y_labels[test_idx]
    
    test_dates = [dates[i] for i in test_idx]
    test_tickers = [tickers_list[i] for i in test_idx]
    
    print(f"Training data: {len(y_train)} samples, Test data: {len(y_test)} samples")
    print(f"Text shape: {X_train_texts.shape}, Social shape: {X_train_social.shape}, Historical shape: {X_train_historical.shape}")
    
    # Reshape historical data to [batch, 1, time, features, 1] for 3D CNN
    X_train_historical = X_train_historical.reshape(X_train_historical.shape[0], 1, 
                                                    X_train_historical.shape[1], 
                                                    X_train_historical.shape[2], 1)
    X_test_historical = X_test_historical.reshape(X_test_historical.shape[0], 1, 
                                                 X_test_historical.shape[1], 
                                                 X_test_historical.shape[2], 1)
    
    # Create PyTorch datasets
    train_dataset = StockPredictionDataset(X_train_texts, X_train_social, X_train_historical, y_train)
    test_dataset = StockPredictionDataset(X_test_texts, X_test_social, X_test_historical, y_test)
    
    return train_dataset, test_dataset, test_dates, test_tickers

# Train function
def train_model(model, train_loader, val_loader, epochs=10, lr=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.BCELoss()
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        correct = 0
        total = 0
        
        # Training loop
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for batch in progress_bar:
            # Get data
            text = batch['text'].to(device)
            social = batch['social'].to(device)
            historical = batch['historical'].to(device)
            labels = batch['label'].to(device)
            
            # Forward pass
            optimizer.zero_grad()
            
            try:
                outputs = model(text, social, historical)
                
                # Loss
                loss = criterion(outputs, labels)
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                # Track loss and accuracy
                epoch_loss += loss.item()
                predicted = (outputs > 0.5).float()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # Update progress bar
                progress_bar.set_postfix(loss=loss.item(), acc=correct/total)
            except Exception as e:
                print(f"Error in training batch: {e}")
                continue
        
        if total > 0:  # Only calculate metrics if we processed at least one batch
            train_loss = epoch_loss / len(train_loader)
            train_acc = correct / total
            train_losses.append(train_loss)
            train_accs.append(train_acc)
        else:
            print("No batches were successfully processed in training.")
            continue
        
        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            progress_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]")
            for batch in progress_bar:
                try:
                    # Get data
                    text = batch['text'].to(device)
                    social = batch['social'].to(device)
                    historical = batch['historical'].to(device)
                    labels = batch['label'].to(device)
                    
                    # Forward pass
                    outputs = model(text, social, historical)
                    
                    # Loss
                    loss = criterion(outputs, labels)
                    
                    # Track loss and accuracy
                    val_loss += loss.item()
                    predicted = (outputs > 0.5).float()
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    
                    # Update progress bar
                    progress_bar.set_postfix(loss=loss.item(), acc=correct/total)
                except Exception as e:
                    print(f"Error in validation batch: {e}")
                    continue
        
        if total > 0:  # Only calculate metrics if we processed at least one batch
            val_loss = val_loss / len(val_loader)
            val_acc = correct / total
            val_losses.append(val_loss)
            val_accs.append(val_acc)
            
            # Update learning rate
            scheduler.step(val_loss)
            
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        else:
            print("No batches were successfully processed in validation.")
    
    return model, train_losses, val_losses, train_accs, val_accs

# Virtual trading simulation
def virtual_trading(model, test_loader, test_dates, test_tickers, prices_df, opens_df, initial_capital=10000):
    """
    Simulate virtual trading as described in the paper.
    
    Parameters:
    -----------
    model: trained model
    test_loader: DataLoader with test data
    test_dates, test_tickers: dates and tickers for test samples
    prices_df, opens_df: price dataframes
    initial_capital: initial investment per trade
    
    Returns:
    --------
    portfolio_value: final portfolio value
    """
    print("Simulating virtual trading...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    # Get predictions
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Making predictions"):
            try:
                text = batch['text'].to(device)
                social = batch['social'].to(device)
                historical = batch['historical'].to(device)
                labels = batch['label']
                
                outputs = model(text, social, historical)
                predictions = (outputs > 0.5).float().cpu().numpy()
                
                all_predictions.extend(predictions)
                all_labels.extend(labels.numpy())
            except Exception as e:
                print(f"Error in prediction batch: {e}")
                continue
    
    if not all_predictions:
        print("No valid predictions were made. Cannot perform virtual trading.")
        return 0, pd.DataFrame(), {}
    
    y_pred = np.array(all_predictions).flatten()
    y_test = np.array(all_labels).flatten()
    
    # Initialize results tracking
    trades = []
    portfolio_value = 0
    correct_predictions = 0
    
    # Group predictions by stock for analysis
    stock_performance = {}
    
    for i, (pred, true, date, ticker) in enumerate(zip(y_pred, y_test, test_dates, test_tickers)):
        # Initialize stock performance dictionary if needed
        if ticker not in stock_performance:
            stock_performance[ticker] = {'correct': 0, 'total': 0, 'profit': 0}
        
        try:    
            # Get date index
            date_idx = prices_df.index.get_loc(date)
            
            # Skip if we don't have enough future dates
            if date_idx + 5 >= len(prices_df):
                continue
                
            # Get prices
            open_price = opens_df[ticker].iloc[date_idx + 1]  # Next day's open
            future_prices = prices_df[ticker].iloc[date_idx + 1:date_idx + 6]  # Next 5 days
            
            # Record if prediction was correct
            actual_movement = true == 1
            predicted_movement = pred == 1
            is_correct = actual_movement == predicted_movement
            
            if is_correct:
                correct_predictions += 1
                stock_performance[ticker]['correct'] += 1
            
            stock_performance[ticker]['total'] += 1
            
            # Simulate trading
            if predicted_movement:  # Predicted rise
                # Buy at next day's open
                shares_bought = initial_capital / open_price
                
                # Check if we hit 2% profit target during holding period
                hit_target = False
                sell_price = future_prices.iloc[-1]  # Default to selling at end of period
                
                for day, price in enumerate(future_prices):
                    if price >= open_price * 1.02:  # 2% profit
                        hit_target = True
                        sell_price = price
                        break
                
                # Calculate profit
                profit = shares_bought * (sell_price - open_price)
                portfolio_value += profit
                stock_performance[ticker]['profit'] += profit
                
                trades.append({
                    'ticker': ticker,
                    'date': date,
                    'position': 'long',
                    'entry': open_price,
                    'exit': sell_price,
                    'profit': profit,
                    'hit_target': hit_target,
                    'correct': is_correct
                })
                
            else:  # Predicted fall
                # Short at next day's open
                shares_shorted = initial_capital / open_price
                
                # Check if we hit 1% drop target during holding period
                hit_target = False
                cover_price = future_prices.iloc[-1]  # Default to covering at end of period
                
                for day, price in enumerate(future_prices):
                    if price <= open_price * 0.99:  # 1% drop
                        hit_target = True
                        cover_price = price
                        break
                
                # Calculate profit (gain when price falls)
                profit = shares_shorted * (open_price - cover_price)
                portfolio_value += profit
                stock_performance[ticker]['profit'] += profit
                
                trades.append({
                    'ticker': ticker,
                    'date': date,
                    'position': 'short',
                    'entry': open_price,
                    'exit': cover_price,
                    'profit': profit,
                    'hit_target': hit_target,
                    'correct': is_correct
                })
        except Exception as e:
            print(f"Error simulating trade for {ticker} on {date}: {e}")
            continue
    
    # Calculate accuracy
    accuracy = correct_predictions / len(y_pred) if len(y_pred) > 0 else 0
    
    # Calculate industry performance
    industry_map = {
        # Map tickers to industries for reporting
        # This would typically come from Bloomberg sector classification
    }
    
    industry_performance = {}
    for ticker, perf in stock_performance.items():
        industry = industry_map.get(ticker, 'Other')
        if industry not in industry_performance:
            industry_performance[industry] = {'profit': 0, 'trades': 0}
        
        industry_performance[industry]['profit'] += perf['profit']
        industry_performance[industry]['trades'] += perf['total']
    
    # Print results
    print(f"Trading Results:")
    print(f"Total Trades: {len(trades)}")
    print(f"Prediction Accuracy: {accuracy:.2%}")
    print(f"Total Profit: ${portfolio_value:.2f}")
    
    if len(trades) > 0:
        print(f"Return on Investment: {portfolio_value / (initial_capital * len(trades)):.2%}")
    
    print("\nIndustry Performance:")
    for industry, perf in industry_performance.items():
        if perf['trades'] > 0:
            avg_profit = perf['profit'] / perf['trades']
            print(f"{industry}: ${perf['profit']:.2f} from {perf['trades']} trades (${avg_profit:.2f}/trade)")
    
    # Create trades dataframe for analysis
    trades_df = pd.DataFrame(trades)
    
    return portfolio_value, trades_df, stock_performance

# Main execution
def run_mman_strategy():
    # Define stock universe (S&P 500 stocks)
    sp500_tickers = [
        'AAPL US Equity', 'MSFT US Equity', 'AMZN US Equity', 'GOOGL US Equity', 'META US Equity',
        'NVDA US Equity', 'BRK/B US Equity', 'JPM US Equity', 'JNJ US Equity', 'V US Equity',
        'PG US Equity', 'UNH US Equity', 'HD US Equity', 'BAC US Equity', 'MA US Equity',
        'XOM US Equity', 'AVGO US Equity', 'COST US Equity', 'WMT US Equity', 'DIS US Equity',
        'CVX US Equity', 'CSCO US Equity', 'ABBV US Equity', 'PEP US Equity', 'LLY US Equity',
        'KO US Equity', 'MRK US Equity', 'ADBE US Equity', 'ORCL US Equity', 'CMCSA US Equity'
    ]
    
    # Define date range (using a recent period)
    import datetime as dt
    end_date = dt.datetime.now().strftime("%Y%m%d")
    start_date = (dt.datetime.now() - dt.timedelta(days=365)).strftime("%Y%m%d")
    print(f"Date range: {start_date} to {end_date}")
    
    try:
        # Fetch price data
        prices_df, volumes_df, highs_df, lows_df, opens_df, volatility_df = fetch_stock_data(
            sp500_tickers, start_date, end_date, con
        )
        
        # Generate historical trending features
        historical_features = generate_historical_features(
            prices_df, volumes_df, highs_df, lows_df, opens_df, volatility_df
        )
        
        # Generate mock social features (in a real implementation, these would come from social media APIs)
        social_features = generate_mock_social_features(prices_df.columns, prices_df)
        
        # Prepare training data
        train_dataset, test_dataset, test_dates, test_tickers = prepare_training_data(
            historical_features, social_features, prices_df
        )
        
        # Create data loaders
        train_size = int(0.9 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        
        # Ensure we have enough data for both training and validation
        if val_size == 0:
            val_size = 1
            train_size = len(train_dataset) - val_size
            
        train_subset, val_subset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
        
        # Use a smaller batch size if dataset is small
        batch_size = min(64, len(train_subset) // 2) if len(train_subset) > 1 else 1
        print(f"Using batch size: {batch_size}")
        
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=batch_size)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        
        # Build model
        print("Building and training model...")
        model = MMAttentionNetwork(d_model=256, num_heads=4)  # Smaller model to prevent overfitting
        
        # Count model parameters
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total model parameters: {total_params:,}")
        
        # Train model
        model, train_losses, val_losses, train_accs, val_accs = train_model(
            model, train_loader, val_loader, epochs=10, lr=0.001
        )
        
        # Evaluate model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.eval()
        criterion = nn.BCELoss()
        test_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluating model"):
                try:
                    text = batch['text'].to(device)
                    social = batch['social'].to(device)
                    historical = batch['historical'].to(device)
                    labels = batch['label'].to(device)
                    
                    outputs = model(text, social, historical)
                    loss = criterion(outputs, labels)
                    
                    test_loss += loss.item()
                    predicted = (outputs > 0.5).float()
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                except Exception as e:
                    print(f"Error evaluating batch: {e}")
                    continue
        
        if total > 0:
            test_loss = test_loss / len(test_loader)
            test_acc = correct / total
            
            print(f"Test Loss: {test_loss:.4f}")
            print(f"Test Accuracy: {test_acc:.4f}")
        else:
            print("No batches were successfully evaluated.")
        
        # Simulate virtual trading
        portfolio_value, trades_df, stock_performance = virtual_trading(
            model, test_loader, test_dates, test_tickers, prices_df, opens_df
        )
        
        # Plot training history
        if train_losses and val_losses:
            plt.figure(figsize=(12, 5))
            plt.subplot(1, 2, 1)
            plt.plot(train_losses, label='Train Loss')
            plt.plot(val_losses, label='Val Loss')
            plt.title('Model Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            
            plt.subplot(1, 2, 2)
            plt.plot(train_accs, label='Train Accuracy')
            plt.plot(val_accs, label='Val Accuracy')
            plt.title('Model Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.tight_layout()
            plt.savefig('mman_training_history.png')
            print("Saved training history plot to mman_training_history.png")
        
        # Plot profit distribution
        if not trades_df.empty:
            plt.figure(figsize=(12, 6))
            trades_df['profit'].hist(bins=50)
            plt.title('Profit Distribution')
            plt.xlabel('Profit ($)')
            plt.ylabel('Frequency')
            plt.savefig('profit_distribution.png')
            print("Saved profit distribution plot to profit_distribution.png")
            
            # Plot cumulative returns
            plt.figure(figsize=(12, 6))
            trades_df = trades_df.sort_values('date')
            trades_df['cumulative_profit'] = trades_df['profit'].cumsum()
            trades_df['cumulative_profit'].plot()
            plt.title('Cumulative Trading Profit')
            plt.xlabel('Trade')
            plt.ylabel('Cumulative Profit ($)')
            plt.grid(True)
            plt.savefig('cumulative_returns.png')
            print("Saved cumulative returns plot to cumulative_returns.png")
        else:
            print("No trades were executed. Check your data and model.")
        
        # Close Bloomberg connection
        con.stop()
        print("Bloomberg connection closed.")
        
        return model, trades_df, stock_performance
        
    except Exception as e:
        print(f"Error in main execution: {e}")
        con.stop()
        print("Bloomberg connection closed due to error.")
        raise e

# Set PyTorch random seeds for reproducibility
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Run the strategy
model, trades_df, stock_performance = run_mman_strategy()