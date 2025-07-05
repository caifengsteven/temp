import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import yfinance as yf
import random
from datetime import datetime, timedelta

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
random.seed(42)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---------------------- Data Preparation Functions ----------------------

def fetch_stock_data(tickers, start_date, end_date):
    """Fetch stock data for a list of tickers within date range"""
    data = {}
    for ticker in tqdm(tickers, desc="Fetching stock data"):
        try:
            stock_data = yf.download(ticker, start=start_date, end=end_date)
            if not stock_data.empty:
                data[ticker] = stock_data
        except Exception as e:
            print(f"Error fetching {ticker}: {e}")
    return data

def calculate_returns(data, window=5):
    """Calculate return ratios for each stock"""
    for ticker in data:
        # Calculate future return ratio
        data[ticker]['return_ratio'] = data[ticker]['Close'].pct_change(window).shift(-window)
    return data

def create_features(data):
    """Create stock factors for each stock"""
    for ticker in data:
        df = data[ticker]
        
        # Basic price features
        df['returns_1d'] = df['Close'].pct_change(1)
        df['returns_5d'] = df['Close'].pct_change(5)
        df['returns_10d'] = df['Close'].pct_change(10)
        df['returns_20d'] = df['Close'].pct_change(20)
        
        # Volume features
        df['volume_1d_change'] = df['Volume'].pct_change(1)
        df['volume_5d_change'] = df['Volume'].pct_change(5)
        df['volume_10d_change'] = df['Volume'].pct_change(10)
        
        # Price-volume relationships
        df['price_volume_1d'] = df['returns_1d'] * df['volume_1d_change']
        df['price_volume_5d'] = df['returns_5d'] * df['volume_5d_change']
        
        # Moving averages
        df['ma_5'] = df['Close'].rolling(window=5).mean()
        df['ma_10'] = df['Close'].rolling(window=10).mean()
        df['ma_20'] = df['Close'].rolling(window=20).mean()
        df['ma_50'] = df['Close'].rolling(window=50).mean()
        
        # Distance from moving averages
        df['dist_ma_5'] = df['Close'] / df['ma_5'] - 1
        df['dist_ma_10'] = df['Close'] / df['ma_10'] - 1
        df['dist_ma_20'] = df['Close'] / df['ma_20'] - 1
        df['dist_ma_50'] = df['Close'] / df['ma_50'] - 1
        
        # Volatility measures
        df['volatility_5d'] = df['returns_1d'].rolling(window=5).std()
        df['volatility_10d'] = df['returns_1d'].rolling(window=10).std()
        df['volatility_20d'] = df['returns_1d'].rolling(window=20).std()
        
        # Momentum indicators
        df['momentum_5d'] = df['Close'].pct_change(5)
        df['momentum_10d'] = df['Close'].pct_change(10)
        df['momentum_20d'] = df['Close'].pct_change(20)
        
        # Relative Strength Index (RSI) - simplified
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).fillna(0)
        loss = (-delta.where(delta < 0, 0)).fillna(0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        df['rsi_14'] = 100 - (100 / (1 + rs))
        
        # Fill NaN values
        df.fillna(method='bfill', inplace=True)
        df.fillna(method='ffill', inplace=True)
        df.fillna(0, inplace=True)
        
        data[ticker] = df
    
    return data

def prepare_sequences(data, seq_length=8, feature_cols=None):
    """Prepare sequence data for time series prediction"""
    X = []
    y = []
    tickers = []
    dates = []
    
    if feature_cols is None:
        # Exclude non-feature columns
        exclude_cols = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'return_ratio']
        
    for ticker in data:
        df = data[ticker]
        
        if feature_cols is None:
            # Use all columns except excluded ones
            features = [col for col in df.columns if col not in exclude_cols]
        else:
            features = feature_cols
        
        for i in range(len(df) - seq_length):
            # Extract sequence
            seq = df.iloc[i:i+seq_length][features].values
            target = df.iloc[i+seq_length-1]['return_ratio']
            
            if not np.isnan(target) and not np.any(np.isnan(seq)):
                X.append(seq)
                y.append(target)
                tickers.append(ticker)
                dates.append(df.index[i+seq_length-1])
    
    return np.array(X), np.array(y), tickers, dates

def assign_sectors(tickers):
    """Assign sector information to tickers"""
    # This is a simplified version - in reality, you'd fetch this from a database
    sectors = {}
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            sector = stock.info.get('sector', 'Unknown')
            sectors[ticker] = sector
        except:
            sectors[ticker] = 'Unknown'
    return sectors

# ---------------------- Diffusion Model Implementation ----------------------

class SinusoidalPositionEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = np.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class AdaptiveLayerNorm(nn.Module):
    def __init__(self, dim, condition_dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(condition_dim, dim * 2),
            nn.SiLU(),
            nn.Linear(dim * 2, dim * 2)
        )
        
        # Zero initialization
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)
        
    def forward(self, x, condition):
        x = self.norm(x)
        scale, shift = self.mlp(condition).chunk(2, dim=-1)
        x = x * (1 + scale) + shift
        return x

class TransformerBlock(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0, condition_dim=None):
        super().__init__()
        self.attn_norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        
        self.ff_norm = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim)
        )
        
        # Adaptive Layer Norm for conditioning
        if condition_dim is not None:
            self.aln1 = AdaptiveLayerNorm(dim, condition_dim)
            self.aln2 = AdaptiveLayerNorm(dim, condition_dim)
        else:
            self.aln1 = None
            self.aln2 = None
            
        # Zero initialization for scaling
        self.scale1 = nn.Parameter(torch.zeros(1))
        self.scale2 = nn.Parameter(torch.zeros(1))
        
    def forward(self, x, condition=None):
        # Self-attention with residual connection
        x_norm = self.attn_norm(x)
        if self.aln1 is not None and condition is not None:
            x_norm = self.aln1(x_norm, condition)
            
        attn_output, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + self.scale1 * attn_output
        
        # Feed-forward with residual connection
        x_norm = self.ff_norm(x)
        if self.aln2 is not None and condition is not None:
            x_norm = self.aln2(x_norm, condition)
            
        x = x + self.scale2 * self.ff(x_norm)
        
        return x

class DiffusionTransformer(nn.Module):
    def __init__(
        self,
        seq_length,
        feature_dim,
        dim=256,
        depth=6,
        heads=8,
        dim_head=64,
        dropout=0.0,
        num_conditions=0,
        condition_dim=64
    ):
        super().__init__()
        
        self.seq_length = seq_length
        self.feature_dim = feature_dim
        self.dim = dim
        self.num_conditions = num_conditions
        
        # Input projection
        self.input_proj = nn.Linear(feature_dim, dim)
        
        # Time embedding
        self.time_embedding = nn.Sequential(
            SinusoidalPositionEmbedding(dim),
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim)
        )
        
        # Conditioning embeddings
        if num_conditions > 0:
            self.condition_embedding = nn.ModuleList([
                nn.Embedding(100, condition_dim) for _ in range(num_conditions)
            ])
            self.condition_proj = nn.Sequential(
                nn.Linear(condition_dim * num_conditions, dim),
                nn.SiLU(),
                nn.Linear(dim, dim)
            )
            combined_condition_dim = dim * 2
        else:
            self.condition_embedding = None
            self.condition_proj = None
            combined_condition_dim = dim
        
        # Null embedding for classifier-free guidance
        self.null_embedding = nn.Parameter(torch.zeros(1, dim))
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                dim=dim,
                heads=heads,
                dim_head=dim_head,
                dropout=dropout,
                condition_dim=combined_condition_dim
            ) for _ in range(depth)
        ])
        
        # Output projection
        self.output_proj = nn.Linear(dim, feature_dim)
        
    def forward(self, x, time, conditions=None, use_null_cond=False):
        batch_size = x.shape[0]
        
        # Project input
        x = self.input_proj(x)
        
        # Time embedding
        time_emb = self.time_embedding(time)
        
        # Process conditions
        if self.num_conditions > 0 and conditions is not None:
            if use_null_cond:
                # Use null embedding for classifier-free guidance
                cond_emb = self.null_embedding.expand(batch_size, -1)
            else:
                # Combine condition embeddings
                cond_embs = []
                for i, condition in enumerate(conditions):
                    if i < len(self.condition_embedding):
                        # Convert to long for embedding lookup
                        condition = condition.long()
                        cond_embs.append(self.condition_embedding[i](condition))
                
                cond_emb = torch.cat(cond_embs, dim=-1)
                cond_emb = self.condition_proj(cond_emb)
                
            # Combine time and condition embeddings
            combined_cond = torch.cat([time_emb, cond_emb], dim=-1)
        else:
            combined_cond = time_emb
        
        # Apply transformer blocks
        for block in self.transformer_blocks:
            x = block(x, combined_cond)
        
        # Project output
        x = self.output_proj(x)
        
        return x

class DiffsFormer:
    def __init__(
        self,
        seq_length,
        feature_dim,
        diffusion_steps=1000,
        editing_step=300,
        dim=256,
        depth=6,
        beta_schedule="linear",
        loss_guided=True,
        device="cpu"
    ):
        self.seq_length = seq_length
        self.feature_dim = feature_dim
        self.diffusion_steps = diffusion_steps
        self.editing_step = editing_step
        self.device = device
        self.loss_guided = loss_guided
        
        # Set up betas according to schedule
        if beta_schedule == "linear":
            self.betas = torch.linspace(1e-4, 0.02, diffusion_steps).to(device)
        elif beta_schedule == "cosine":
            steps = torch.arange(diffusion_steps + 1) / diffusion_steps
            alpha_cumprod = torch.cos((steps + 0.008) / 1.008 * torch.pi / 2) ** 2
            alpha_cumprod = alpha_cumprod / alpha_cumprod[0]
            betas = 1 - (alpha_cumprod[1:] / alpha_cumprod[:-1])
            self.betas = torch.clamp(betas, 0.0001, 0.9999).to(device)
        
        # Pre-compute diffusion parameters
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        
        # Initialize model
        self.model = DiffusionTransformer(
            seq_length=seq_length,
            feature_dim=feature_dim,
            dim=dim,
            depth=depth,
            device=device
        ).to(device)
        
        # Save training losses for loss-guided diffusion
        self.sample_losses = {}
        
    def diffusion_forward(self, x0, t, noise=None):
        """Forward diffusion process: q(x_t | x_0)"""
        if noise is None:
            noise = torch.randn_like(x0)
            
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)
        
        # x_t = √(α_cumprod) * x_0 + √(1 - α_cumprod) * ε
        x_t = sqrt_alphas_cumprod_t * x0 + sqrt_one_minus_alphas_cumprod_t * noise
        
        return x_t, noise
    
    def diffusion_reverse_step(self, x_t, t, predicted_noise, guidance_scale=1.0, uncond_predicted_noise=None):
        """Single step of reverse diffusion process: p(x_{t-1} | x_t)"""
        # Get alpha and beta values for current timestep
        alpha = self.alphas[t]
        alpha_cumprod = self.alphas_cumprod[t]
        alpha_cumprod_prev = self.alphas_cumprod_prev[t]
        beta = self.betas[t]
        
        # Apply classifier-free guidance if provided
        if uncond_predicted_noise is not None and guidance_scale > 1.0:
            predicted_noise = uncond_predicted_noise + guidance_scale * (predicted_noise - uncond_predicted_noise)
        
        # Calculate predicted x0
        pred_x0 = (x_t - torch.sqrt(1 - alpha_cumprod).view(-1, 1, 1) * predicted_noise) / \
                 torch.sqrt(alpha_cumprod).view(-1, 1, 1)
        
        # Get posterior mean and variance
        posterior_mean = (alpha_cumprod_prev / alpha_cumprod) * pred_x0 + \
                         ((1 - alpha_cumprod_prev) / torch.sqrt(1 - alpha_cumprod)) * \
                         torch.sqrt(alpha).view(-1, 1, 1) * predicted_noise
        
        posterior_variance = self.posterior_variance[t].view(-1, 1, 1)
        
        # Sample x_{t-1}
        noise = torch.randn_like(x_t)
        x_t_minus_1 = posterior_mean + torch.sqrt(posterior_variance) * noise
        
        return x_t_minus_1
    
    def ddim_sample(self, x_t, t, t_prev, predicted_noise, guidance_scale=1.0, uncond_predicted_noise=None):
        """DDIM sampling step: more efficient than regular reverse diffusion"""
        # Apply classifier-free guidance if provided
        if uncond_predicted_noise is not None and guidance_scale > 1.0:
            predicted_noise = uncond_predicted_noise + guidance_scale * (predicted_noise - uncond_predicted_noise)
        
        # Calculate predicted x0
        alpha_cumprod_t = self.alphas_cumprod[t]
        alpha_cumprod_prev = self.alphas_cumprod[t_prev] if t_prev >= 0 else torch.tensor(1.0).to(self.device)
        
        pred_x0 = (x_t - torch.sqrt(1 - alpha_cumprod_t).view(-1, 1, 1) * predicted_noise) / \
                 torch.sqrt(alpha_cumprod_t).view(-1, 1, 1)
        
        # Get direction pointing to x_t
        dir_xt = torch.sqrt(1 - alpha_cumprod_prev - 0.0).view(-1, 1, 1) * predicted_noise
        
        # DDIM formula for x_{t-1}
        x_t_minus_1 = torch.sqrt(alpha_cumprod_prev).view(-1, 1, 1) * pred_x0 + dir_xt
        
        return x_t_minus_1
    
    def train_step(self, x0, optimizer):
        """Single training step"""
        self.model.train()
        batch_size = x0.shape[0]
        
        # Sample t uniformly from 1 to editing_step
        t = torch.randint(1, self.editing_step + 1, (batch_size,), device=self.device)
        
        # Sample noise and add it to x0
        noise = torch.randn_like(x0)
        x_t, target_noise = self.diffusion_forward(x0, t, noise)
        
        # Predict noise
        predicted_noise = self.model(x_t, t)
        
        # Calculate loss
        loss = F.mse_loss(predicted_noise, target_noise)
        
        # Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        return loss.item()
    
    def train(self, dataloader, epochs, lr=1e-4):
        """Train the diffusion model"""
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        
        for epoch in range(epochs):
            epoch_loss = 0
            for batch_idx, (x, y, idx) in enumerate(dataloader):
                x = x.to(self.device)
                loss = self.train_step(x, optimizer)
                epoch_loss += loss
                
                # Store sample losses for loss-guided diffusion
                if self.loss_guided:
                    for i, sample_idx in enumerate(idx.cpu().numpy()):
                        self.sample_losses[sample_idx] = loss
                
            scheduler.step()
            avg_loss = epoch_loss / len(dataloader)
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")
    
    def generate_samples(self, x0, num_samples=1, guidance_scale=1.0, sampling_steps=None):
        """Generate samples by diffusing and denoising"""
        self.model.eval()
        batch_size = x0.shape[0]
        
        # Default to editing_step if sampling_steps not provided
        if sampling_steps is None:
            sampling_steps = self.editing_step
        
        # Repeat x0 for multiple samples
        if num_samples > 1:
            x0 = x0.repeat(num_samples, 1, 1)
            batch_size *= num_samples
        
        # Determine noise scale based on loss if using loss-guided diffusion
        if self.loss_guided:
            noise_scale = torch.ones(batch_size, device=self.device)
            for i, idx in enumerate(range(batch_size)):
                if idx in self.sample_losses:
                    # Scale inversely with loss - lower loss points get more noise
                    noise_scale[i] = 1.0 + (1.0 / (1.0 + self.sample_losses[idx]))
        else:
            noise_scale = torch.ones(batch_size, device=self.device)
        
        # Diffuse forward to editing step
        with torch.no_grad():
            t = torch.ones(batch_size, device=self.device).long() * self.editing_step
            x_t, _ = self.diffusion_forward(x0, t)
            
            # DDIM sampling (faster than regular diffusion)
            sampling_seq = list(range(0, self.editing_step, self.editing_step // sampling_steps))
            if sampling_seq[-1] != self.editing_step:
                sampling_seq.append(self.editing_step)
            
            # Reverse sampling
            for i in range(len(sampling_seq) - 1, 0, -1):
                t_curr = torch.ones(batch_size, device=self.device).long() * sampling_seq[i]
                t_next = torch.ones(batch_size, device=self.device).long() * sampling_seq[i-1]
                
                # Predict noise
                predicted_noise = self.model(x_t, t_curr)
                
                # For unconditional guidance, we would predict noise again with null condition
                # (simplified here)
                uncond_predicted_noise = None
                
                # DDIM step
                x_t = self.ddim_sample(
                    x_t, 
                    t_curr, 
                    t_next, 
                    predicted_noise, 
                    guidance_scale, 
                    uncond_predicted_noise
                )
        
        return x_t
    
    def save(self, path):
        """Save the model"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'diffusion_steps': self.diffusion_steps,
            'editing_step': self.editing_step,
            'betas': self.betas,
            'alphas': self.alphas,
            'alphas_cumprod': self.alphas_cumprod,
            'alphas_cumprod_prev': self.alphas_cumprod_prev,
            'sqrt_alphas_cumprod': self.sqrt_alphas_cumprod,
            'sqrt_one_minus_alphas_cumprod': self.sqrt_one_minus_alphas_cumprod,
            'sqrt_recip_alphas': self.sqrt_recip_alphas,
            'posterior_variance': self.posterior_variance
        }, path)
    
    def load(self, path):
        """Load the model"""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.diffusion_steps = checkpoint['diffusion_steps']
        self.editing_step = checkpoint['editing_step']
        self.betas = checkpoint['betas']
        self.alphas = checkpoint['alphas']
        self.alphas_cumprod = checkpoint['alphas_cumprod']
        self.alphas_cumprod_prev = checkpoint['alphas_cumprod_prev']
        self.sqrt_alphas_cumprod = checkpoint['sqrt_alphas_cumprod']
        self.sqrt_one_minus_alphas_cumprod = checkpoint['sqrt_one_minus_alphas_cumprod']
        self.sqrt_recip_alphas = checkpoint['sqrt_recip_alphas']
        self.posterior_variance = checkpoint['posterior_variance']

# ---------------------- Backbone Models ----------------------

class MLPModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=256):
        super().__init__()
        self.flatten = nn.Flatten()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, x):
        x = self.flatten(x)
        return self.net(x).squeeze(-1)

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        output, (h_n, c_n) = self.lstm(x)
        return self.fc(output[:, -1, :]).squeeze(-1)

class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, num_layers=2):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        output, h_n = self.gru(x)
        return self.fc(output[:, -1, :]).squeeze(-1)

class TransformerModel(nn.Module):
    def __init__(self, input_dim, d_model=256, nhead=8, num_layers=3, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=d_model*4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pos_encoder = SinusoidalPositionEmbedding(d_model)
        self.fc = nn.Linear(d_model, 1)
        
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        # Add positional encoding
        pos = torch.arange(0, seq_len, device=x.device).unsqueeze(0).repeat(batch_size, 1)
        pos_enc = self.pos_encoder(pos)
        
        # Project input and add positional encoding
        x = self.input_proj(x) + pos_enc.unsqueeze(1)
        
        # Pass through transformer
        x = self.transformer_encoder(x)
        
        # Use last token for prediction
        return self.fc(x[:, -1, :]).squeeze(-1)

# ---------------------- Evaluation Functions ----------------------

def calculate_annualized_return(predictions, true_returns, top_k=30):
    """Calculate annualized return ratio by selecting top k stocks each period"""
    all_returns = []
    
    for preds, returns in zip(predictions, true_returns):
        # Get indices of top k predicted returns
        top_indices = np.argsort(preds)[-top_k:]
        
        # Calculate average return for selected stocks
        period_return = np.mean(returns[top_indices])
        all_returns.append(period_return)
    
    # Convert to annualized return (assuming daily returns)
    daily_return = np.mean(all_returns)
    annualized_return = (1 + daily_return) ** 252 - 1
    
    return annualized_return

def calculate_ic(predictions, true_returns):
    """Calculate Information Coefficient (Pearson correlation)"""
    return np.corrcoef(predictions, true_returns)[0, 1]

def calculate_rank_ic(predictions, true_returns):
    """Calculate Rank Information Coefficient (Spearman correlation)"""
    return pd.Series(predictions).corr(pd.Series(true_returns), method='spearman')

def evaluate_model(model, dataloader, device):
    """Evaluate model on test data"""
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for x, y, _ in dataloader:
            x = x.to(device)
            y = y.to(device)
            
            preds = model(x)
            
            all_preds.append(preds.cpu().numpy())
            all_targets.append(y.cpu().numpy())
    
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    
    # Calculate metrics
    rr = calculate_annualized_return([all_preds], [all_targets])
    ic = calculate_ic(all_preds, all_targets)
    rank_ic = calculate_rank_ic(all_preds, all_targets)
    
    return {
        'return_ratio': rr,
        'ic': ic,
        'rank_ic': rank_ic,
        'predictions': all_preds,
        'targets': all_targets
    }

# ---------------------- Main Experiment Functions ----------------------

def run_experiment_with_diffusion(
    X_train, y_train, X_test, y_test, 
    model_class, 
    diffusion_params=None, 
    device='cpu',
    batch_size=64,
    epochs=50
):
    """Run experiment with diffusion augmentation"""
    seq_length, feature_dim = X_train.shape[1], X_train.shape[2]
    
    # Create datasets and dataloaders
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train), 
        torch.FloatTensor(y_train),
        torch.arange(len(X_train))
    )
    test_dataset = TensorDataset(
        torch.FloatTensor(X_test), 
        torch.FloatTensor(y_test),
        torch.arange(len(X_test))
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Default diffusion parameters
    if diffusion_params is None:
        diffusion_params = {
            'diffusion_steps': 1000,
            'editing_step': 300,
            'dim': 256,
            'depth': 6
        }
    
    # Initialize and train diffusion model
    diffusion_model = DiffsFormer(
        seq_length=seq_length,
        feature_dim=feature_dim,
        diffusion_steps=diffusion_params['diffusion_steps'],
        editing_step=diffusion_params['editing_step'],
        dim=diffusion_params['dim'],
        depth=diffusion_params['depth'],
        device=device
    )
    
    print("Training diffusion model...")
    diffusion_model.train(train_loader, epochs=30, lr=1e-4)
    
    # Generate augmented samples
    print("Generating augmented samples...")
    augmented_X = []
    augmented_y = []
    
    for x, y, idx in train_loader:
        x = x.to(device)
        augmented = diffusion_model.generate_samples(
            x, 
            num_samples=1, 
            guidance_scale=1.0,
            sampling_steps=20
        )
        augmented_X.append(augmented.cpu().numpy())
        augmented_y.append(y.numpy())
    
    augmented_X = np.concatenate(augmented_X)
    augmented_y = np.concatenate(augmented_y)
    
    # Combine original and augmented data
    combined_X = np.concatenate([X_train, augmented_X])
    combined_y = np.concatenate([y_train, augmented_y])
    
    # Create new combined dataset and dataloader
    combined_dataset = TensorDataset(
        torch.FloatTensor(combined_X), 
        torch.FloatTensor(combined_y),
        torch.arange(len(combined_X))
    )
    combined_loader = DataLoader(combined_dataset, batch_size=batch_size, shuffle=True)
    
    # Train backbone model on original data
    backbone_original = model_class().to(device)
    optimizer_original = torch.optim.Adam(backbone_original.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    print("Training backbone model on original data...")
    for epoch in range(epochs):
        backbone_original.train()
        for x, y, _ in train_loader:
            x, y = x.to(device), y.to(device)
            
            optimizer_original.zero_grad()
            output = backbone_original(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer_original.step()
    
    # Train backbone model on augmented data
    backbone_augmented = model_class().to(device)
    optimizer_augmented = torch.optim.Adam(backbone_augmented.parameters(), lr=1e-3)
    
    print("Training backbone model on augmented data...")
    for epoch in range(epochs):
        backbone_augmented.train()
        for x, y, _ in combined_loader:
            x, y = x.to(device), y.to(device)
            
            optimizer_augmented.zero_grad()
            output = backbone_augmented(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer_augmented.step()
    
    # Evaluate both models
    print("Evaluating models...")
    results_original = evaluate_model(backbone_original, test_loader, device)
    results_augmented = evaluate_model(backbone_augmented, test_loader, device)
    
    # Calculate relative improvement
    rel_improvement = {
        'return_ratio': (results_augmented['return_ratio'] / results_original['return_ratio'] - 1) * 100,
        'ic': (results_augmented['ic'] / results_original['ic'] - 1) * 100,
        'rank_ic': (results_augmented['rank_ic'] / results_original['rank_ic'] - 1) * 100
    }
    
    return {
        'original': results_original,
        'augmented': results_augmented,
        'rel_improvement': rel_improvement,
        'diffusion_model': diffusion_model,
        'backbone_original': backbone_original,
        'backbone_augmented': backbone_augmented
    }

def visualize_results(results):
    """Visualize experiment results"""
    # Plot return ratio comparison
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.bar(['Original', 'Augmented'], 
            [results['original']['return_ratio'], results['augmented']['return_ratio']])
    plt.title(f'Annualized Return Ratio\nImprovement: {results["rel_improvement"]["return_ratio"]:.2f}%')
    plt.grid(axis='y', alpha=0.3)
    
    plt.subplot(1, 3, 2)
    plt.bar(['Original', 'Augmented'], 
            [results['original']['ic'], results['augmented']['ic']])
    plt.title(f'Information Coefficient\nImprovement: {results["rel_improvement"]["ic"]:.2f}%')
    plt.grid(axis='y', alpha=0.3)
    
    plt.subplot(1, 3, 3)
    plt.bar(['Original', 'Augmented'], 
            [results['original']['rank_ic'], results['augmented']['rank_ic']])
    plt.title(f'Rank Information Coefficient\nImprovement: {results["rel_improvement"]["rank_ic"]:.2f}%')
    plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('metrics_comparison.png')
    plt.close()
    
    # Plot predictions vs actual
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.scatter(results['original']['targets'], results['original']['predictions'], alpha=0.5)
    plt.xlabel('Actual Returns')
    plt.ylabel('Predicted Returns')
    plt.title('Original Model')
    plt.grid(alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.scatter(results['augmented']['targets'], results['augmented']['predictions'], alpha=0.5)
    plt.xlabel('Actual Returns')
    plt.ylabel('Predicted Returns')
    plt.title('Augmented Model')
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('predictions_comparison.png')
    plt.close()

def visualize_feature_distributions(X_original, X_augmented):
    """Visualize original and augmented feature distributions"""
    # Flatten the sequence dimension
    X_original_flat = X_original.reshape(-1, X_original.shape[-1])
    X_augmented_flat = X_augmented.reshape(-1, X_augmented.shape[-1])
    
    # Select a subset of features to visualize
    features_to_plot = min(6, X_original.shape[-1])
    
    plt.figure(figsize=(15, 10))
    for i in range(features_to_plot):
        plt.subplot(2, 3, i+1)
        sns.kdeplot(X_original_flat[:, i], label='Original')
        sns.kdeplot(X_augmented_flat[:, i], label='Augmented')
        plt.title(f'Feature {i+1}')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig('feature_distributions.png')
    plt.close()
    
    # Visualize with t-SNE
    from sklearn.manifold import TSNE
    
    # Sample points for t-SNE (it can be slow for large datasets)
    sample_size = min(5000, len(X_original_flat))
    idx_original = np.random.choice(len(X_original_flat), sample_size, replace=False)
    idx_augmented = np.random.choice(len(X_augmented_flat), sample_size, replace=False)
    
    X_original_sample = X_original_flat[idx_original]
    X_augmented_sample = X_augmented_flat[idx_augmented]
    
    # Combine for t-SNE
    X_combined = np.vstack([X_original_sample, X_augmented_sample])
    labels = np.array(['Original'] * sample_size + ['Augmented'] * sample_size)
    
    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(X_combined)
    
    # Plot
    plt.figure(figsize=(10, 8))
    for label, color in zip(['Original', 'Augmented'], ['blue', 'orange']):
        mask = labels == label
        plt.scatter(X_tsne[mask, 0], X_tsne[mask, 1], c=color, label=label, alpha=0.5)
    
    plt.legend()
    plt.title('t-SNE Visualization of Original vs Augmented Features')
    plt.savefig('tsne_visualization.png')
    plt.close()

# ---------------------- Main Execution ----------------------

def main():
    # Configure experiment
    use_simulated_data = True  # Set to False to use real stock data
    
    if use_simulated_data:
        # Generate simulated stock data
        print("Generating simulated stock data...")
        num_stocks = 100
        seq_length = 8
        feature_dim = 20
        num_samples = 10000
        
        # Simulate stock features with some correlation patterns
        X = np.random.randn(num_samples, seq_length, feature_dim)
        
        # Add some temporal dependencies
        for i in range(1, seq_length):
            X[:, i, :] = 0.7 * X[:, i-1, :] + 0.3 * X[:, i, :]
        
        # Add some feature dependencies
        for i in range(1, feature_dim):
            X[:, :, i] = 0.3 * X[:, :, i-1] + 0.7 * X[:, :, i]
        
        # Generate return ratios with noise
        base_signal = 0.2 * X[:, -1, 0] + 0.15 * X[:, -1, 1] - 0.1 * X[:, -1, 2]
        y = base_signal + 0.8 * np.random.randn(num_samples)  # Low signal-to-noise ratio
        
        # Split into train and test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Create sector information for conditioning
        sectors = np.random.randint(0, 10, size=num_samples)
        sectors_train, sectors_test = sectors[:len(X_train)], sectors[len(X_train):]
        
    else:
        # Use real stock data from Yahoo Finance
        print("Fetching real stock data...")
        # Get S&P 500 tickers
        sp500_url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        sp500_table = pd.read_html(sp500_url)[0]
        tickers = sp500_table['Symbol'].tolist()[:100]  # Limit to 100 for demo
        
        # Fetch data
        start_date = '2018-01-01'
        end_date = '2022-12-31'
        stock_data = fetch_stock_data(tickers, start_date, end_date)
        
        # Calculate return ratios
        stock_data = calculate_returns(stock_data)
        
        # Create features
        stock_data = create_features(stock_data)
        
        # Prepare sequences
        X, y, ticker_list, dates = prepare_sequences(stock_data)
        
        # Get sectors
        ticker_sectors = assign_sectors(list(set(ticker_list)))
        sectors = np.array([ticker_sectors.get(ticker, 'Unknown') for ticker in ticker_list])
        
        # Convert sectors to integers
        unique_sectors = list(set(sectors))
        sector_to_int = {sector: i for i, sector in enumerate(unique_sectors)}
        sectors = np.array([sector_to_int[sector] for sector in sectors])
        
        # Split into train and test by date
        split_date = datetime.strptime('2021-01-01', '%Y-%m-%d')
        train_indices = [i for i, date in enumerate(dates) if date < split_date]
        test_indices = [i for i, date in enumerate(dates) if date >= split_date]
        
        X_train, y_train = X[train_indices], y[train_indices]
        X_test, y_test = X[test_indices], y[test_indices]
        sectors_train, sectors_test = sectors[train_indices], sectors[test_indices]
    
    print(f"Data shapes: X_train: {X_train.shape}, y_train: {y_train.shape}, X_test: {X_test.shape}, y_test: {y_test.shape}")
    
    # Normalize data
    scaler = StandardScaler()
    X_train_flat = X_train.reshape(-1, X_train.shape[-1])
    X_test_flat = X_test.reshape(-1, X_test.shape[-1])
    
    X_train_flat = scaler.fit_transform(X_train_flat)
    X_test_flat = scaler.transform(X_test_flat)
    
    X_train = X_train_flat.reshape(X_train.shape)
    X_test = X_test_flat.reshape(X_test.shape)
    
    # Run experiments with different backbone models
    print("\nRunning experiments with different backbone models...")
    
    # Define model classes with appropriate input dimensions
    class MLPForExperiment(nn.Module):
        def __init__(self):
            super().__init__()
            input_dim = X_train.shape[1] * X_train.shape[2]  # Flatten sequence and features
            self.model = MLPModel(input_dim)
        
        def forward(self, x):
            return self.model(x)
    
    class LSTMForExperiment(nn.Module):
        def __init__(self):
            super().__init__()
            input_dim = X_train.shape[2]  # Feature dimension
            self.model = LSTMModel(input_dim)
        
        def forward(self, x):
            return self.model(x)
    
    class GRUForExperiment(nn.Module):
        def __init__(self):
            super().__init__()
            input_dim = X_train.shape[2]  # Feature dimension
            self.model = GRUModel(input_dim)
        
        def forward(self, x):
            return self.model(x)
    
    class TransformerForExperiment(nn.Module):
        def __init__(self):
            super().__init__()
            input_dim = X_train.shape[2]  # Feature dimension
            self.model = TransformerModel(input_dim)
        
        def forward(self, x):
            return self.model(x)
    
    # Run experiments
    models = {
        'MLP': MLPForExperiment,
        'LSTM': LSTMForExperiment,
        'GRU': GRUForExperiment,
        'Transformer': TransformerForExperiment
    }
    
    results = {}
    
    for model_name, model_class in models.items():
        print(f"\nRunning experiment with {model_name}...")
        model_results = run_experiment_with_diffusion(
            X_train, y_train, X_test, y_test, 
            model_class=model_class,
            device=device,
            epochs=20  # Reduced for demo
        )
        results[model_name] = model_results
        
        # Print results
        print(f"\n{model_name} Results:")
        print(f"Original - Return Ratio: {model_results['original']['return_ratio']:.6f}, IC: {model_results['original']['ic']:.6f}, Rank IC: {model_results['original']['rank_ic']:.6f}")
        print(f"Augmented - Return Ratio: {model_results['augmented']['return_ratio']:.6f}, IC: {model_results['augmented']['ic']:.6f}, Rank IC: {model_results['augmented']['rank_ic']:.6f}")
        print(f"Relative Improvement - Return Ratio: {model_results['rel_improvement']['return_ratio']:.2f}%, IC: {model_results['rel_improvement']['ic']:.2f}%, Rank IC: {model_results['rel_improvement']['rank_ic']:.2f}%")
        
        # Visualize results
        visualize_results(model_results)
    
    # Generate and visualize augmented samples
    print("\nGenerating augmented samples for visualization...")
    diffusion_model = results['Transformer']['diffusion_model']
    
    # Generate augmented samples
    batch_size = 64
    num_batches = min(10, len(X_train) // batch_size)
    
    augmented_samples = []
    original_samples = []
    
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = start_idx + batch_size
        
        batch_X = X_train[start_idx:end_idx]
        batch_X_tensor = torch.FloatTensor(batch_X).to(device)
        
        augmented = diffusion_model.generate_samples(
            batch_X_tensor, 
            num_samples=1,
            guidance_scale=1.0,
            sampling_steps=20
        )
        
        augmented_samples.append(augmented.cpu().numpy())
        original_samples.append(batch_X)
    
    augmented_samples = np.concatenate(augmented_samples)
    original_samples = np.concatenate(original_samples)
    
    # Visualize feature distributions
    visualize_feature_distributions(original_samples, augmented_samples)
    
    print("\nExperiments completed successfully!")

if __name__ == "__main__":
    main()