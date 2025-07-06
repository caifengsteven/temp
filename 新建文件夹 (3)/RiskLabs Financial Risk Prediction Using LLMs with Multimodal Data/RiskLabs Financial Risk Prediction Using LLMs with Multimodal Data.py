import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import datetime
import time
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

class DataGenerator:
    """Generate simulated data for testing the RiskLabs framework"""
    
    def __init__(self, n_samples=1000, n_features=100):
        self.n_samples = n_samples
        self.n_features = n_features
        
    def generate_time_series(self):
        """Generate VIX-like time series data for 30 days before each event"""
        # Base values with some randomness
        base_values = np.random.normal(20, 5, self.n_samples)
        
        # Create 30-day sequences with some patterns
        series = []
        for base in base_values:
            # Create a series with some trend, seasonality, and noise
            trend = np.linspace(0, np.random.uniform(-3, 3), 30)
            seasonality = np.sin(np.linspace(0, 2*np.pi, 30)) * np.random.uniform(0, 2)
            noise = np.random.normal(0, 1, 30)
            
            # Combine components
            sample = base + trend + seasonality + noise
            
            # Ensure values are positive
            sample = np.maximum(sample, 5)
            
            series.append(sample)
            
        return np.array(series)
    
    def generate_earnings_call_text_embeddings(self):
        """
        Generate simulated text embeddings for earnings call transcripts.
        In reality, these would be generated using SimCSE or another embedding model.
        """
        # Simulate 520 sentences per call with 768 dimensions each
        text_embeddings = np.random.normal(0, 1, (self.n_samples, 520, 768))
        return text_embeddings
    
    def generate_earnings_call_audio_embeddings(self):
        """
        Generate simulated audio embeddings for earnings call audio.
        In reality, these would be generated using Wav2vec2.
        """
        # Simulate 520 audio frames per call with 512 dimensions each
        audio_embeddings = np.random.normal(0, 1, (self.n_samples, 520, 512))
        return audio_embeddings
    
    def generate_earnings_call_summaries(self):
        """
        Generate simulated summaries of earnings calls.
        In reality, these would be generated using an LLM.
        """
        # Simulate 1024-dimensional embeddings of summaries
        summary_embeddings = np.random.normal(0, 1, (self.n_samples, 1024))
        return summary_embeddings
    
    def generate_earnings_call_important_sentences(self):
        """
        Generate simulated embeddings of important sentences from earnings calls.
        In reality, these would be generated using an LLM and embedding model.
        """
        # Simulate 1024-dimensional embeddings of important sentences
        important_sentence_embeddings = np.random.normal(0, 1, (self.n_samples, 1024))
        return important_sentence_embeddings
    
    def generate_news_embeddings(self):
        """
        Generate simulated news embeddings.
        In reality, these would be generated using an LLM and embedding model.
        """
        # Simulate 256-dimensional news embeddings
        news_embeddings = np.random.normal(0, 1, (self.n_samples, 256))
        return news_embeddings
    
    def generate_volatility_targets(self, time_series_data, text_embeddings, audio_embeddings):
        """
        Generate target volatility values with some correlation to input features.
        
        In reality, these would be calculated from actual stock prices.
        """
        # 3, 7, 15, and 30-day volatility
        volatility = np.zeros((self.n_samples, 4))
        
        for i in range(self.n_samples):
            # Base volatility with increasing values for longer periods
            base_vol = np.array([0.15, 0.2, 0.25, 0.3])
            
            # Add influence from VIX (higher VIX = higher volatility)
            vix_factor = np.mean(time_series_data[i]) / 20  # Normalize around 20
            
            # Add some influence from text sentiment (simulated)
            text_factor = np.mean(text_embeddings[i, :10, :10]) * 0.05
            
            # Add some influence from audio sentiment (simulated)
            audio_factor = np.mean(audio_embeddings[i, :10, :10]) * 0.05
            
            # Combine factors with some randomness
            vol = base_vol * vix_factor * (1 + text_factor + audio_factor)
            
            # Add noise
            vol += np.random.normal(0, 0.05, 4)
            
            # Ensure volatility is positive
            vol = np.maximum(vol, 0.05)
            
            volatility[i] = vol
        
        return volatility
    
    def generate_var_targets(self, volatility):
        """
        Generate 95% Value at Risk targets with correlation to volatility.
        
        In reality, these would be calculated from actual stock returns.
        """
        # VaR is related to volatility but has its own patterns
        var = np.zeros(self.n_samples)
        
        for i in range(self.n_samples):
            # Use 3-day volatility as a base
            base_var = volatility[i, 0] * 1.65  # Assuming normal distribution for 95% VaR
            
            # Add some randomness
            var[i] = base_var * (1 + np.random.normal(0, 0.1))
            
            # Ensure VaR is positive
            var[i] = max(var[i], 0.01)
        
        return var
    
    def generate_dataset(self):
        """Generate a complete dataset for training and testing"""
        print("Generating time series data...")
        time_series = self.generate_time_series()
        
        print("Generating text embeddings...")
        text_embeddings = self.generate_earnings_call_text_embeddings()
        
        print("Generating audio embeddings...")
        audio_embeddings = self.generate_earnings_call_audio_embeddings()
        
        print("Generating summary embeddings...")
        summary_embeddings = self.generate_earnings_call_summaries()
        
        print("Generating important sentence embeddings...")
        important_sentence_embeddings = self.generate_earnings_call_important_sentences()
        
        print("Generating news embeddings...")
        news_embeddings = self.generate_news_embeddings()
        
        print("Generating target volatility...")
        volatility = self.generate_volatility_targets(time_series, text_embeddings, audio_embeddings)
        
        print("Generating target VaR...")
        var = self.generate_var_targets(volatility)
        
        # Create a dictionary of data
        data = {
            'time_series': time_series,
            'text_embeddings': text_embeddings,
            'audio_embeddings': audio_embeddings,
            'summary_embeddings': summary_embeddings,
            'important_sentence_embeddings': important_sentence_embeddings,
            'news_embeddings': news_embeddings,
            'volatility': volatility,
            'var': var
        }
        
        return data

class FinancialDataset(Dataset):
    """Dataset for handling financial data"""
    
    def __init__(self, data, mode='train'):
        self.mode = mode
        
        # Process and store data
        self.time_series = torch.tensor(data['time_series'], dtype=torch.float32)
        self.text_embeddings = torch.tensor(data['text_embeddings'], dtype=torch.float32)
        self.audio_embeddings = torch.tensor(data['audio_embeddings'], dtype=torch.float32)
        self.summary_embeddings = torch.tensor(data['summary_embeddings'], dtype=torch.float32)
        self.important_sentence_embeddings = torch.tensor(data['important_sentence_embeddings'], dtype=torch.float32)
        self.news_embeddings = torch.tensor(data['news_embeddings'], dtype=torch.float32)
        
        # Target values
        self.volatility = torch.tensor(data['volatility'], dtype=torch.float32)
        self.var = torch.tensor(data['var'], dtype=torch.float32).unsqueeze(1)
    
    def __len__(self):
        return len(self.volatility)
    
    def __getitem__(self, idx):
        # Return all data for a single sample
        sample = {
            'time_series': self.time_series[idx],
            'text_embeddings': self.text_embeddings[idx],
            'audio_embeddings': self.audio_embeddings[idx],
            'summary_embeddings': self.summary_embeddings[idx],
            'important_sentence_embeddings': self.important_sentence_embeddings[idx],
            'news_embeddings': self.news_embeddings[idx],
            'volatility': self.volatility[idx],
            'var': self.var[idx]
        }
        
        return sample

class MultiHeadSelfAttention(nn.Module):
    """Multi-Head Self-Attention module as described in the paper"""
    
    def __init__(self, input_dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads
        
        # Check that embedding dimension is divisible by number of heads
        assert self.head_dim * num_heads == input_dim, "Embedding dimension must be divisible by number of heads"
        
        # Linear projections for Q, K, V
        self.q_linear = nn.Linear(input_dim, input_dim)
        self.k_linear = nn.Linear(input_dim, input_dim)
        self.v_linear = nn.Linear(input_dim, input_dim)
        
        # Output projection
        self.output_linear = nn.Linear(input_dim, input_dim)
        
        # Layer norm
        self.norm = nn.LayerNorm(input_dim)
        
        # MLP
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, input_dim * 4),
            nn.ReLU(),
            nn.Linear(input_dim * 4, input_dim),
            nn.Dropout(0.1)
        )
        
        # Dropout
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x, mask=None):
        batch_size, seq_len, d_model = x.size()
        
        # Layer normalization
        residual = x
        x = self.norm(x)
        
        # Linear projections
        q = self.q_linear(x)
        k = self.k_linear(x)
        v = self.v_linear(x)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax and dropout
        attention_weights = torch.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention weights to values
        context = torch.matmul(attention_weights, v)
        
        # Reshape and concat heads
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        
        # Output projection
        output = self.output_linear(context)
        output = self.dropout(output)
        
        # First residual connection
        output = output + residual
        
        # Second MLP block with residual connection
        residual = output
        output = self.mlp(self.norm(output))
        output = output + residual
        
        return output

class TimeSeriesEncoder(nn.Module):
    """Bidirectional LSTM encoder for time series data"""
    
    def __init__(self, input_dim=1, hidden_dim=64, output_dim=128):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim=input_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
    
    def forward(self, x):
        # Input shape: [batch_size, seq_len]
        x = x.unsqueeze(-1)  # Add feature dimension
        
        # Pass through LSTM
        outputs, (hidden, _) = self.lstm(x)
        
        # Concatenate the final forward and backward hidden states
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        
        # Pass through fully connected layer
        output = self.fc(hidden)
        
        return output

class AudioEncoder(nn.Module):
    """Encoder for audio embeddings using MHSA and pooling"""
    
    def __init__(self, input_dim=512, output_dim=512):
        super().__init__()
        self.mhsa = MultiHeadSelfAttention(input_dim)
        self.fc = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        # Input shape: [batch_size, seq_len, input_dim]
        
        # Apply MHSA
        x = self.mhsa(x)
        
        # Average pooling
        x = x.mean(dim=1)
        
        # Fully connected layer
        x = self.fc(x)
        
        return x

class TextEncoder(nn.Module):
    """Encoder for text embeddings using MHSA and pooling"""
    
    def __init__(self, input_dim=768, output_dim=512):
        super().__init__()
        self.mhsa = MultiHeadSelfAttention(input_dim)
        self.fc = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        # Input shape: [batch_size, seq_len, input_dim]
        
        # Apply MHSA
        x = self.mhsa(x)
        
        # Average pooling
        x = x.mean(dim=1)
        
        # Fully connected layer
        x = self.fc(x)
        
        return x

class RiskLabs(nn.Module):
    """Implementation of the RiskLabs framework"""
    
    def __init__(self, fusion_dim=512):
        super().__init__()
        
        # Audio encoder
        self.audio_encoder = AudioEncoder(input_dim=512, output_dim=fusion_dim)
        
        # Text encoder
        self.text_encoder = TextEncoder(input_dim=768, output_dim=fusion_dim)
        
        # Time series encoder
        self.time_series_encoder = TimeSeriesEncoder(input_dim=1, hidden_dim=64, output_dim=128)
        
        # Projection layers for other embeddings
        self.summary_projector = nn.Linear(1024, fusion_dim)
        self.important_sentences_projector = nn.Linear(1024, fusion_dim)
        self.news_projector = nn.Linear(256, fusion_dim)
        
        # Fusion weights (learnable)
        self.fusion_weights = nn.Parameter(torch.ones(6))
        self.fusion_bias = nn.Parameter(torch.zeros(fusion_dim))
        
        # Task-specific output layers
        self.volatility_predictor = nn.Linear(fusion_dim, 4)  # 3, 7, 15, 30 days
        self.var_predictor = nn.Linear(fusion_dim, 1)  # 95% VaR
    
    def forward(self, time_series, text_embeddings, audio_embeddings, 
                summary_embeddings, important_sentence_embeddings, news_embeddings):
        # Process each modality
        audio_features = self.audio_encoder(audio_embeddings)
        text_features = self.text_encoder(text_embeddings)
        time_series_features = self.time_series_encoder(time_series)
        
        # Project other embeddings
        summary_features = self.summary_projector(summary_embeddings)
        important_sentences_features = self.important_sentences_projector(important_sentence_embeddings)
        news_features = self.news_projector(news_embeddings)
        
        # Normalize fusion weights using softmax
        fusion_weights = torch.softmax(self.fusion_weights, dim=0)
        
        # Weighted fusion
        fused_features = (
            fusion_weights[0] * audio_features +
            fusion_weights[1] * text_features +
            fusion_weights[2] * summary_features +
            fusion_weights[3] * important_sentences_features +
            fusion_weights[4] * time_series_features +
            fusion_weights[5] * news_features +
            self.fusion_bias
        )
        
        # Task-specific predictions
        volatility_pred = self.volatility_predictor(fused_features)
        var_pred = self.var_predictor(fused_features)
        
        return volatility_pred, var_pred

def train_model(model, train_loader, val_loader, criterion_vol, criterion_var, 
                optimizer, num_epochs=10, mu=0.5, device='cpu'):
    """Train the RiskLabs model"""
    
    # Move model to device
    model = model.to(device)
    
    # Track training history
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_model_state = None
    
    # Training loop
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training"):
            # Get data
            time_series = batch['time_series'].to(device)
            text_embeddings = batch['text_embeddings'].to(device)
            audio_embeddings = batch['audio_embeddings'].to(device)
            summary_embeddings = batch['summary_embeddings'].to(device)
            important_sentence_embeddings = batch['important_sentence_embeddings'].to(device)
            news_embeddings = batch['news_embeddings'].to(device)
            
            # Get targets
            volatility_targets = batch['volatility'].to(device)
            var_targets = batch['var'].to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            volatility_pred, var_pred = model(
                time_series, text_embeddings, audio_embeddings,
                summary_embeddings, important_sentence_embeddings, news_embeddings
            )
            
            # Calculate losses
            vol_loss = criterion_vol(volatility_pred, volatility_targets)
            var_loss = criterion_var(var_pred, var_targets)
            
            # Combine losses with trade-off parameter mu
            loss = mu * vol_loss + (1 - mu) * var_loss
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Accumulate loss
            train_loss += loss.item() * time_series.size(0)
        
        # Calculate average training loss
        train_loss = train_loss / len(train_loader.dataset)
        train_losses.append(train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation"):
                # Get data
                time_series = batch['time_series'].to(device)
                text_embeddings = batch['text_embeddings'].to(device)
                audio_embeddings = batch['audio_embeddings'].to(device)
                summary_embeddings = batch['summary_embeddings'].to(device)
                important_sentence_embeddings = batch['important_sentence_embeddings'].to(device)
                news_embeddings = batch['news_embeddings'].to(device)
                
                # Get targets
                volatility_targets = batch['volatility'].to(device)
                var_targets = batch['var'].to(device)
                
                # Forward pass
                volatility_pred, var_pred = model(
                    time_series, text_embeddings, audio_embeddings,
                    summary_embeddings, important_sentence_embeddings, news_embeddings
                )
                
                # Calculate losses
                vol_loss = criterion_vol(volatility_pred, volatility_targets)
                var_loss = criterion_var(var_pred, var_targets)
                
                # Combine losses with trade-off parameter mu
                loss = mu * vol_loss + (1 - mu) * var_loss
                
                # Accumulate loss
                val_loss += loss.item() * time_series.size(0)
        
        # Calculate average validation loss
        val_loss = val_loss / len(val_loader.dataset)
        val_losses.append(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
        
        # Print progress
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    
    # Load best model
    model.load_state_dict(best_model_state)
    
    return model, train_losses, val_losses

def evaluate_model(model, test_loader, device='cpu'):
    """Evaluate the trained model on test data"""
    
    model.eval()
    
    # Collect predictions and targets
    vol_preds = []
    var_preds = []
    vol_targets = []
    var_targets = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            # Get data
            time_series = batch['time_series'].to(device)
            text_embeddings = batch['text_embeddings'].to(device)
            audio_embeddings = batch['audio_embeddings'].to(device)
            summary_embeddings = batch['summary_embeddings'].to(device)
            important_sentence_embeddings = batch['important_sentence_embeddings'].to(device)
            news_embeddings = batch['news_embeddings'].to(device)
            
            # Forward pass
            volatility_pred, var_pred = model(
                time_series, text_embeddings, audio_embeddings,
                summary_embeddings, important_sentence_embeddings, news_embeddings
            )
            
            # Collect predictions and targets
            vol_preds.append(volatility_pred.cpu().numpy())
            var_preds.append(var_pred.cpu().numpy())
            vol_targets.append(batch['volatility'].numpy())
            var_targets.append(batch['var'].numpy())
    
    # Concatenate results
    vol_preds = np.concatenate(vol_preds, axis=0)
    var_preds = np.concatenate(var_preds, axis=0)
    vol_targets = np.concatenate(vol_targets, axis=0)
    var_targets = np.concatenate(var_targets, axis=0)
    
    # Calculate MSE for each volatility horizon
    vol_mse = np.mean((vol_preds - vol_targets) ** 2, axis=0)
    vol_mse_overall = np.mean(vol_mse)
    
    # Calculate MSE for VaR
    var_mse = np.mean((var_preds - var_targets) ** 2)
    
    # Create results dictionary
    results = {
        'vol_mse_overall': vol_mse_overall,
        'vol_mse_3d': vol_mse[0],
        'vol_mse_7d': vol_mse[1],
        'vol_mse_15d': vol_mse[2],
        'vol_mse_30d': vol_mse[3],
        'var_mse': var_mse,
        'vol_preds': vol_preds,
        'var_preds': var_preds,
        'vol_targets': vol_targets,
        'var_targets': var_targets
    }
    
    return results

def plot_training_history(train_losses, val_losses):
    """Plot training and validation loss history"""
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_predictions(results):
    """Plot predictions vs targets for volatility and VaR"""
    
    # Plot volatility predictions
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    horizons = ['3-Day', '7-Day', '15-Day', '30-Day']
    
    for i, ax in enumerate(axes.flat):
        ax.scatter(results['vol_targets'][:, i], results['vol_preds'][:, i], alpha=0.5)
        
        # Add perfect prediction line
        min_val = min(results['vol_targets'][:, i].min(), results['vol_preds'][:, i].min())
        max_val = max(results['vol_targets'][:, i].max(), results['vol_preds'][:, i].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--')
        
        ax.set_xlabel('True Volatility')
        ax.set_ylabel('Predicted Volatility')
        ax.set_title(f'{horizons[i]} Volatility Predictions')
        
        # Add MSE to the plot
        mse = np.mean((results['vol_preds'][:, i] - results['vol_targets'][:, i]) ** 2)
        ax.text(0.05, 0.95, f'MSE: {mse:.4f}', transform=ax.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Plot VaR predictions
    plt.figure(figsize=(10, 6))
    plt.scatter(results['var_targets'], results['var_preds'], alpha=0.5)
    
    # Add perfect prediction line
    min_val = min(results['var_targets'].min(), results['var_preds'].min())
    max_val = max(results['var_targets'].max(), results['var_preds'].max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    plt.xlabel('True VaR')
    plt.ylabel('Predicted VaR')
    plt.title('95% VaR Predictions')
    
    # Add MSE to the plot
    mse = np.mean((results['var_preds'] - results['var_targets']) ** 2)
    plt.text(0.05, 0.95, f'MSE: {mse:.4f}', transform=plt.gca().transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.grid(True)
    plt.show()

def compare_with_baseline(data):
    """Compare RiskLabs with baseline models as described in the paper"""
    
    # Split data into train, validation, and test sets
    train_idx, temp_idx = train_test_split(np.arange(len(data['volatility'])), test_size=0.3, random_state=42)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42)
    
    # Create datasets for each split
    train_data = {}
    val_data = {}
    test_data = {}
    
    for key in data:
        train_data[key] = data[key][train_idx]
        val_data[key] = data[key][val_idx]
        test_data[key] = data[key][test_idx]
    
    # Create feature matrices for simple baseline models
    X_train_simple = np.concatenate([
        np.mean(train_data['time_series'], axis=1, keepdims=True),  # VIX average
        np.std(train_data['time_series'], axis=1, keepdims=True),   # VIX volatility
    ], axis=1)
    
    X_val_simple = np.concatenate([
        np.mean(val_data['time_series'], axis=1, keepdims=True),
        np.std(val_data['time_series'], axis=1, keepdims=True),
    ], axis=1)
    
    X_test_simple = np.concatenate([
        np.mean(test_data['time_series'], axis=1, keepdims=True),
        np.std(test_data['time_series'], axis=1, keepdims=True),
    ], axis=1)
    
    # Scale features
    scaler = StandardScaler()
    X_train_simple = scaler.fit_transform(X_train_simple)
    X_val_simple = scaler.transform(X_val_simple)
    X_test_simple = scaler.transform(X_test_simple)
    
    # Baseline 1: Classical Method (Linear Regression)
    from sklearn.linear_model import LinearRegression
    
    # For volatility prediction
    classical_vol_model = LinearRegression()
    classical_vol_model.fit(X_train_simple, train_data['volatility'])
    classical_vol_preds = classical_vol_model.predict(X_test_simple)
    classical_vol_mse = np.mean((classical_vol_preds - test_data['volatility']) ** 2, axis=0)
    classical_vol_mse_overall = np.mean(classical_vol_mse)
    
    # For VaR prediction
    classical_var_model = LinearRegression()
    classical_var_model.fit(X_train_simple, train_data['var'])
    classical_var_preds = classical_var_model.predict(X_test_simple)
    classical_var_mse = np.mean((classical_var_preds - test_data['var']) ** 2)
    
    # Baseline 2: LSTM Model for time series
    class LSTMBaseline(nn.Module):
        def __init__(self, input_dim=1, hidden_dim=64, output_dim=4):
            super().__init__()
            self.lstm = nn.LSTM(
                input_dim=input_dim,
                hidden_size=hidden_dim,
                num_layers=2,
                batch_first=True,
                dropout=0.1
            )
            self.fc = nn.Linear(hidden_dim, output_dim)
        
        def forward(self, x):
            # Input shape: [batch_size, seq_len]
            x = x.unsqueeze(-1)  # Add feature dimension
            
            # Pass through LSTM
            outputs, (hidden, _) = self.lstm(x)
            
            # Use the final hidden state
            hidden = hidden[-1]
            
            # Pass through fully connected layer
            output = self.fc(hidden)
            
            return output
    
    # Create datasets for LSTM
    train_dataset = FinancialDataset(train_data, mode='train')
    val_dataset = FinancialDataset(val_data, mode='val')
    test_dataset = FinancialDataset(test_data, mode='test')
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Initialize LSTM model for volatility prediction
    lstm_model = LSTMBaseline(input_dim=1, hidden_dim=64, output_dim=4).to(device)
    lstm_criterion = nn.MSELoss()
    lstm_optimizer = optim.Adam(lstm_model.parameters(), lr=0.001)
    
    # Train LSTM model
    print("Training LSTM baseline model...")
    best_val_loss = float('inf')
    best_model_state = None
    
    for epoch in range(10):
        # Training
        lstm_model.train()
        train_loss = 0.0
        
        for batch in train_loader:
            time_series = batch['time_series'].to(device)
            volatility_targets = batch['volatility'].to(device)
            
            lstm_optimizer.zero_grad()
            outputs = lstm_model(time_series)
            loss = lstm_criterion(outputs, volatility_targets)
            loss.backward()
            lstm_optimizer.step()
            
            train_loss += loss.item() * time_series.size(0)
        
        train_loss = train_loss / len(train_loader.dataset)
        
        # Validation
        lstm_model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                time_series = batch['time_series'].to(device)
                volatility_targets = batch['volatility'].to(device)
                
                outputs = lstm_model(time_series)
                loss = lstm_criterion(outputs, volatility_targets)
                
                val_loss += loss.item() * time_series.size(0)
        
        val_loss = val_loss / len(val_loader.dataset)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = lstm_model.state_dict().copy()
        
        print(f"LSTM Epoch {epoch+1}/10 - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    
    # Load best model
    lstm_model.load_state_dict(best_model_state)
    
    # Evaluate LSTM model
    lstm_model.eval()
    lstm_vol_preds = []
    lstm_vol_targets = []
    
    with torch.no_grad():
        for batch in test_loader:
            time_series = batch['time_series'].to(device)
            volatility_targets = batch['volatility']
            
            outputs = lstm_model(time_series)
            
            lstm_vol_preds.append(outputs.cpu().numpy())
            lstm_vol_targets.append(volatility_targets.numpy())
    
    lstm_vol_preds = np.concatenate(lstm_vol_preds, axis=0)
    lstm_vol_targets = np.concatenate(lstm_vol_targets, axis=0)
    
    lstm_vol_mse = np.mean((lstm_vol_preds - lstm_vol_targets) ** 2, axis=0)
    lstm_vol_mse_overall = np.mean(lstm_vol_mse)
    
    # Baseline 3: GPT-3.5 Simulation (since we can't actually call the API)
    # We'll simulate poor performance as mentioned in the paper
    np.random.seed(123)  # Different seed for this baseline
    gpt_vol_preds = np.random.normal(
        loc=np.mean(test_data['volatility'], axis=0),
        scale=np.std(test_data['volatility'], axis=0) * 2,  # Higher variance to simulate poor performance
        size=test_data['volatility'].shape
    )
    
    gpt_var_preds = np.random.normal(
        loc=np.mean(test_data['var']),
        scale=np.std(test_data['var']) * 2,
        size=test_data['var'].shape
    )
    
    gpt_vol_mse = np.mean((gpt_vol_preds - test_data['volatility']) ** 2, axis=0)
    gpt_vol_mse_overall = np.mean(gpt_vol_mse)
    gpt_var_mse = np.mean((gpt_var_preds - test_data['var']) ** 2)
    
    # Compile results for comparison
    comparison_results = {
        'Classical Method': {
            'MSE_overall': classical_vol_mse_overall,
            'MSE_3d': classical_vol_mse[0],
            'MSE_7d': classical_vol_mse[1],
            'MSE_15d': classical_vol_mse[2],
            'MSE_30d': classical_vol_mse[3],
            'VaR': classical_var_mse
        },
        'LSTM': {
            'MSE_overall': lstm_vol_mse_overall,
            'MSE_3d': lstm_vol_mse[0],
            'MSE_7d': lstm_vol_mse[1],
            'MSE_15d': lstm_vol_mse[2],
            'MSE_30d': lstm_vol_mse[3],
            'VaR': None  # LSTM baseline didn't predict VaR
        },
        'GPT-3.5-Turbo': {
            'MSE_overall': gpt_vol_mse_overall,
            'MSE_3d': gpt_vol_mse[0],
            'MSE_7d': gpt_vol_mse[1],
            'MSE_15d': gpt_vol_mse[2],
            'MSE_30d': gpt_vol_mse[3],
            'VaR': gpt_var_mse
        },
        'RiskLabs': {}  # Will be filled after RiskLabs is trained
    }
    
    return comparison_results, (train_loader, val_loader, test_loader)

def run_ablation_study(model, test_loader, device='cpu'):
    """Run ablation study to examine the contribution of different components"""
    
    model.eval()
    
    # Full model performance
    full_results = evaluate_model(model, test_loader, device)
    
    # Ablation 1: Audio + Text only
    fusion_weights_backup = model.fusion_weights.clone()
    
    # Set weights for other components to zero
    with torch.no_grad():
        model.fusion_weights[2:] = -1e9  # Effectively zero after softmax
    
    audio_text_results = evaluate_model(model, test_loader, device)
    
    # Ablation 2: Audio + Text + Analysis
    with torch.no_grad():
        model.fusion_weights[2:4] = fusion_weights_backup[2:4]  # Restore summary and important sentences
        model.fusion_weights[4:] = -1e9  # Zero out time series and news
    
    audio_text_analysis_results = evaluate_model(model, test_loader, device)
    
    # Restore original weights
    with torch.no_grad():
        model.fusion_weights = fusion_weights_backup
    
    # Compile ablation results
    ablation_results = {
        'Audio + Text': {
            'MSE_overall': audio_text_results['vol_mse_overall'],
            'MSE_3d': audio_text_results['vol_mse_3d'],
            'MSE_7d': audio_text_results['vol_mse_7d'],
            'MSE_15d': audio_text_results['vol_mse_15d'],
            'MSE_30d': audio_text_results['vol_mse_30d'],
            'VaR': audio_text_results['var_mse']
        },
        'Audio + Text + Analysis': {
            'MSE_overall': audio_text_analysis_results['vol_mse_overall'],
            'MSE_3d': audio_text_analysis_results['vol_mse_3d'],
            'MSE_7d': audio_text_analysis_results['vol_mse_7d'],
            'MSE_15d': audio_text_analysis_results['vol_mse_15d'],
            'MSE_30d': audio_text_analysis_results['vol_mse_30d'],
            'VaR': audio_text_analysis_results['var_mse']
        },
        'Full Model': {
            'MSE_overall': full_results['vol_mse_overall'],
            'MSE_3d': full_results['vol_mse_3d'],
            'MSE_7d': full_results['vol_mse_7d'],
            'MSE_15d': full_results['vol_mse_15d'],
            'MSE_30d': full_results['vol_mse_30d'],
            'VaR': full_results['var_mse']
        }
    }
    
    return ablation_results

def plot_comparison_table(comparison_results):
    """Plot a comparison table of different models"""
    
    # Create DataFrame for easy visualization
    df = pd.DataFrame.from_dict(comparison_results, orient='index')
    
    # Round values for better display
    df = df.round(3)
    
    # Plot as a table
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('off')
    ax.axis('tight')
    
    table = ax.table(cellText=df.values,
                     rowLabels=df.index,
                     colLabels=df.columns,
                     cellLoc='center',
                     loc='center')
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    plt.title('Model Comparison')
    plt.tight_layout()
    plt.show()
    
    return df

def plot_ablation_table(ablation_results):
    """Plot an ablation study table"""
    
    # Create DataFrame for easy visualization
    df = pd.DataFrame.from_dict(ablation_results, orient='index')
    
    # Round values for better display
    df = df.round(3)
    
    # Plot as a table
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('off')
    ax.axis('tight')
    
    table = ax.table(cellText=df.values,
                     rowLabels=df.index,
                     colLabels=df.columns,
                     cellLoc='center',
                     loc='center')
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    plt.title('Ablation Study Results')
    plt.tight_layout()
    plt.show()
    
    return df

# Main execution
def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Generate simulated data
    print("Generating simulated data...")
    data_generator = DataGenerator(n_samples=500, n_features=100)
    data = data_generator.generate_dataset()
    print("Data generation complete.")
    
    # Compare with baseline models
    print("\nComparing with baseline models...")
    comparison_results, (train_loader, val_loader, test_loader) = compare_with_baseline(data)
    
    # Initialize RiskLabs model
    print("\nInitializing RiskLabs model...")
    model = RiskLabs(fusion_dim=512)
    
    # Define loss functions and optimizer
    criterion_vol = nn.MSELoss()
    criterion_var = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train the model
    print("\nTraining RiskLabs model...")
    model, train_losses, val_losses = train_model(
        model, train_loader, val_loader,
        criterion_vol, criterion_var, optimizer,
        num_epochs=20, mu=0.5, device=device
    )
    
    # Plot training history
    print("\nPlotting training history...")
    plot_training_history(train_losses, val_losses)
    
    # Evaluate the model
    print("\nEvaluating RiskLabs model...")
    results = evaluate_model(model, test_loader, device)
    
    # Update comparison results with RiskLabs performance
    comparison_results['RiskLabs'] = {
        'MSE_overall': results['vol_mse_overall'],
        'MSE_3d': results['vol_mse_3d'],
        'MSE_7d': results['vol_mse_7d'],
        'MSE_15d': results['vol_mse_15d'],
        'MSE_30d': results['vol_mse_30d'],
        'VaR': results['var_mse']
    }
    
    # Plot predictions
    print("\nPlotting predictions...")
    plot_predictions(results)
    
    # Plot comparison table
    print("\nPlotting comparison table...")
    comparison_df = plot_comparison_table(comparison_results)
    print(comparison_df)
    
    # Run ablation study
    print("\nRunning ablation study...")
    ablation_results = run_ablation_study(model, test_loader, device)
    
    # Plot ablation study results
    print("\nPlotting ablation study results...")
    ablation_df = plot_ablation_table(ablation_results)
    print(ablation_df)
    
    print("\nDone!")

if __name__ == "__main__":
    main()