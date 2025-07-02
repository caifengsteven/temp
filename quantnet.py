import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import datetime
from tqdm import tqdm
import random
import os
from scipy.cluster.hierarchy import dendrogram, linkage

# Set seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
random.seed(42)

# Helper functions for data generation
def generate_synthetic_market_data(num_assets=30, num_days=1000, market_correlation=0.5):
    """
    Generate synthetic market data with a specified correlation structure.
    
    Parameters:
    -----------
    num_assets : int
        Number of assets in the market
    num_days : int
        Number of trading days
    market_correlation : float
        Base correlation between assets
    
    Returns:
    --------
    DataFrame with asset returns
    """
    # Create a correlation matrix with market_correlation off-diagonal elements
    corr_matrix = np.ones((num_assets, num_assets)) * market_correlation
    np.fill_diagonal(corr_matrix, 1.0)
    
    # Generate correlated returns
    mean_returns = np.random.normal(0.0005, 0.0002, num_assets)  # Mean daily returns around 0.05%
    volatilities = np.random.uniform(0.01, 0.03, num_assets)  # Daily volatilities between 1-3%
    
    # Cholesky decomposition for correlated random variables
    L = np.linalg.cholesky(corr_matrix)
    
    # Generate uncorrelated returns
    uncorrelated_returns = np.random.normal(0, 1, size=(num_days, num_assets))
    
    # Apply correlation structure and scale by volatilities
    correlated_returns = uncorrelated_returns @ L.T
    scaled_returns = correlated_returns * volatilities + mean_returns
    
    # Convert to DataFrame
    dates = pd.date_range(start='2020-01-01', periods=num_days, freq='B')
    returns_df = pd.DataFrame(scaled_returns, index=dates)
    returns_df.columns = [f'Asset_{i+1}' for i in range(num_assets)]
    
    return returns_df

def generate_multiple_markets(num_markets=10, assets_range=(20, 50), days_range=(800, 1200)):
    """
    Generate multiple synthetic markets with different characteristics.
    
    Parameters:
    -----------
    num_markets : int
        Number of markets to generate
    assets_range : tuple
        Range of number of assets for each market
    days_range : tuple
        Range of number of days for each market
    
    Returns:
    --------
    List of DataFrames, each representing a market
    """
    markets = []
    
    for i in range(num_markets):
        # Generate random number of assets and days
        num_assets = np.random.randint(assets_range[0], assets_range[1])
        num_days = np.random.randint(days_range[0], days_range[1])
        
        # Generate random correlation parameter
        market_corr = np.random.uniform(0.2, 0.8)
        
        # Generate market data
        market_data = generate_synthetic_market_data(
            num_assets=num_assets,
            num_days=num_days,
            market_correlation=market_corr
        )
        
        # Add some market-specific characteristics
        if i % 3 == 0:  # Add trend to some markets
            trend = np.linspace(-0.002, 0.002, num_days)
            market_data = market_data.add(trend[:, np.newaxis], axis=0)
        elif i % 3 == 1:  # Add seasonality to some markets
            seasonality = 0.001 * np.sin(np.linspace(0, 10*np.pi, num_days))
            market_data = market_data.add(seasonality[:, np.newaxis], axis=0)
        
        markets.append(market_data)
    
    return markets

# Define the QuantNet architecture
class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0.1):
        super(Encoder, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
    def forward(self, x):
        output, (hidden, cell) = self.lstm(x)
        return output, hidden

class TransferLayer(nn.Module):
    def __init__(self, input_size, output_size, dropout=0.1):
        super(TransferLayer, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        return self.dropout(self.linear(x))

class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, dropout=0.1):
        super(Decoder, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.linear = nn.Linear(hidden_size, output_size)
        
    def forward(self, x, hidden=None):
        output, (hidden, cell) = self.lstm(x, hidden)
        return self.linear(output), hidden

class QuantNet(nn.Module):
    def __init__(self, market_configs, transfer_size=10, encoder_layers=1, decoder_layers=1, dropout=0.1):
        super(QuantNet, self).__init__()
        self.market_configs = market_configs
        self.transfer_size = transfer_size
        
        # Create encoder and decoder for each market
        self.encoders = nn.ModuleDict()
        self.decoders = nn.ModuleDict()
        
        # Find common hidden size for all markets to avoid dimension mismatch
        self.hidden_size = min([config['hidden_size'] for config in market_configs.values()])
        
        for market_id, config in market_configs.items():
            input_size = config['input_size']
            output_size = config['output_size']
            
            self.encoders[market_id] = Encoder(
                input_size=input_size,
                hidden_size=self.hidden_size,  # Use common hidden size
                num_layers=encoder_layers,
                dropout=dropout
            )
            
            self.decoders[market_id] = Decoder(
                input_size=transfer_size,
                hidden_size=self.hidden_size,  # Use common hidden size
                output_size=output_size,
                num_layers=decoder_layers,
                dropout=dropout
            )
        
        # Create shared transfer layer with consistent dimensions
        self.transfer_layer = TransferLayer(
            input_size=self.hidden_size,
            output_size=transfer_size,
            dropout=dropout
        )
        
        # Final activation for trading signals
        self.tanh = nn.Tanh()
        
    def forward(self, x, market_id):
        # Get encoder and decoder for the specific market
        encoder = self.encoders[market_id]
        decoder = self.decoders[market_id]
        
        # Encode market-specific input
        encoder_output, encoder_hidden = encoder(x)
        
        # Process through transfer layer (using the last hidden state)
        transfer_output = self.transfer_layer(encoder_hidden[-1])
        
        # Expand transfer output to match sequence length
        transfer_output = transfer_output.unsqueeze(1).expand(-1, x.shape[1], -1)
        
        # Decode into trading signals
        decoder_output, _ = decoder(transfer_output)
        
        # Apply tanh for final trading signals
        signals = self.tanh(decoder_output)
        
        return signals

# Sharpe ratio loss function
class SharpeLoss(nn.Module):
    def __init__(self, annualization_factor=252):
        super(SharpeLoss, self).__init__()
        self.annualization_factor = annualization_factor
        
    def forward(self, signals, returns):
        # Calculate strategy returns
        strategy_returns = torch.sum(signals[:, :-1] * returns[:, 1:], dim=2)
        
        # Calculate mean and std of returns
        mean_return = torch.mean(strategy_returns, dim=1)
        std_return = torch.std(strategy_returns, dim=1) + 1e-6  # Add small constant to avoid division by zero
        
        # Calculate Sharpe ratio
        sharpe = mean_return / std_return * torch.sqrt(torch.tensor(self.annualization_factor))
        
        # We want to maximize Sharpe, so we minimize negative Sharpe
        return -torch.mean(sharpe)

# Training function
def train_quantnet(model, markets_data, num_epochs=50, batch_size=16, seq_length=20, learning_rate=0.001):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = SharpeLoss()
    
    market_ids = list(markets_data.keys())
    
    # Prepare datasets
    datasets = {}
    for market_id, data in markets_data.items():
        X = torch.tensor(data.values, dtype=torch.float32)
        
        # Create sequences with shape (num_sequences, seq_length+1, num_assets)
        sequences = []
        for i in range(len(X) - seq_length - 1):
            seq = X[i:i+seq_length+1]
            sequences.append(seq)
        
        if not sequences:
            print(f"Warning: Market {market_id} has insufficient data for sequence length {seq_length}")
            continue
            
        X = torch.stack(sequences)
        
        # Split into input sequences and target returns
        inputs = X[:, :-1, :]  # All but the last time step
        targets = X[:, 1:, :]  # All but the first time step
        
        datasets[market_id] = TensorDataset(inputs, targets)
    
    # Training loop
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        
        # Sample mini-batch of markets
        sampled_markets = random.sample(list(datasets.keys()), min(len(datasets), 4))
        
        for market_id in sampled_markets:
            # Create DataLoader for this market
            dataloader = DataLoader(datasets[market_id], batch_size=batch_size, shuffle=True)
            
            for inputs, targets in dataloader:
                optimizer.zero_grad()
                
                # Forward pass
                signals = model(inputs, market_id)
                
                # Compute loss
                loss = criterion(signals, targets)
                
                # Backward pass and optimize
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
        
        if (epoch + 1) % 5 == 0:
            print(f'Epoch {epoch+1}/{num_epochs}, Loss: {total_loss:.4f}')
    
    return model

# Evaluation functions
def calculate_sharpe_ratio(returns, annualization_factor=252):
    """Calculate Sharpe ratio from a series of returns"""
    if len(returns) == 0 or np.std(returns) == 0:
        return 0
    return np.mean(returns) / np.std(returns) * np.sqrt(annualization_factor)

def calculate_calmar_ratio(returns, annualization_factor=252):
    """Calculate Calmar ratio from a series of returns"""
    if len(returns) == 0:
        return 0
    
    # Calculate cumulative returns
    cumulative_returns = (1 + returns).cumprod() - 1
    
    # Calculate maximum drawdown
    running_max = np.maximum.accumulate(cumulative_returns)
    drawdown = (cumulative_returns - running_max) / (1 + running_max)
    max_drawdown = abs(np.min(drawdown) if len(drawdown) > 0 else 0)
    
    if max_drawdown == 0:
        return 0
    
    # Calculate annualized return
    ann_return = (1 + np.mean(returns)) ** annualization_factor - 1
    
    return ann_return / max_drawdown

def evaluate_strategy(model, market_data, market_id, seq_length=20):
    """Evaluate the QuantNet model on a specific market"""
    model.eval()
    
    with torch.no_grad():
        # Prepare data
        data = torch.tensor(market_data.values, dtype=torch.float32)
        
        # Initialize arrays for signals and returns
        all_signals = []
        all_returns = []
        
        # Generate signals in a sliding window
        for i in range(len(data) - seq_length):
            sequence = data[i:i+seq_length].unsqueeze(0)  # Add batch dimension
            signals = model(sequence, market_id)
            
            # Store the signal for the last day in the sequence
            all_signals.append(signals[0, -1].numpy())
            
            # Store the corresponding future return
            if i + seq_length < len(data):
                all_returns.append(data[i+seq_length].numpy())
        
        if not all_signals or not all_returns:
            print(f"Warning: Insufficient data for evaluation in {market_id}")
            return {
                'signals': np.array([]),
                'returns': np.array([]),
                'strategy_returns': np.array([]),
                'sharpe_ratio': 0,
                'calmar_ratio': 0,
                'cumulative_return': 0
            }
        
        all_signals = np.array(all_signals)
        all_returns = np.array(all_returns)
        
        # Calculate strategy returns
        strategy_returns = np.sum(all_signals[:-1] * all_returns[1:], axis=1)
        
        # Calculate performance metrics
        sharpe = calculate_sharpe_ratio(strategy_returns)
        calmar = calculate_calmar_ratio(strategy_returns)
        
        return {
            'signals': all_signals,
            'returns': all_returns,
            'strategy_returns': strategy_returns,
            'sharpe_ratio': sharpe,
            'calmar_ratio': calmar,
            'cumulative_return': (1 + strategy_returns).cumprod()[-1] - 1 if len(strategy_returns) > 0 else 0
        }

# Benchmark strategy implementations
def time_series_momentum(market_data, lookback=252):
    """Time series momentum strategy"""
    returns = market_data.values
    
    # Calculate past returns
    past_returns = np.zeros_like(returns)
    for i in range(lookback, len(returns)):
        past_returns[i] = np.mean(returns[i-lookback:i], axis=0)
    
    # Generate signals: +1 for positive returns, -1 for negative returns
    signals = np.sign(past_returns)
    
    # Calculate strategy returns
    strategy_returns = np.sum(signals[:-1] * returns[1:], axis=1)
    
    # Calculate performance metrics
    sharpe = calculate_sharpe_ratio(strategy_returns[lookback:])
    calmar = calculate_calmar_ratio(strategy_returns[lookback:])
    
    return {
        'signals': signals,
        'strategy_returns': strategy_returns[lookback:],
        'sharpe_ratio': sharpe,
        'calmar_ratio': calmar,
        'cumulative_return': (1 + strategy_returns[lookback:]).cumprod()[-1] - 1 if len(strategy_returns) > lookback else 0
    }

def cross_sectional_momentum(market_data, lookback=252, quantile=0.33):
    """Cross-sectional momentum strategy"""
    returns = market_data.values
    
    # Calculate past returns
    past_returns = np.zeros_like(returns)
    for i in range(lookback, len(returns)):
        past_returns[i] = np.mean(returns[i-lookback:i], axis=0)
    
    # Generate signals based on quantiles
    signals = np.zeros_like(past_returns)
    for i in range(lookback, len(returns)):
        # Calculate quantiles for this time step
        top_quantile = np.quantile(past_returns[i], 1-quantile)
        bottom_quantile = np.quantile(past_returns[i], quantile)
        
        # Assign signals: +1 for top quantile, -1 for bottom quantile
        signals[i, past_returns[i] > top_quantile] = 1
        signals[i, past_returns[i] < bottom_quantile] = -1
    
    # Calculate strategy returns
    strategy_returns = np.sum(signals[:-1] * returns[1:], axis=1)
    
    # Calculate performance metrics
    sharpe = calculate_sharpe_ratio(strategy_returns[lookback:])
    calmar = calculate_calmar_ratio(strategy_returns[lookback:])
    
    return {
        'signals': signals,
        'strategy_returns': strategy_returns[lookback:],
        'sharpe_ratio': sharpe,
        'calmar_ratio': calmar,
        'cumulative_return': (1 + strategy_returns[lookback:]).cumprod()[-1] - 1 if len(strategy_returns) > lookback else 0
    }

def buy_and_hold(market_data):
    """Buy and hold strategy"""
    returns = market_data.values
    
    # Generate signals: always 1 (long)
    signals = np.ones_like(returns)
    
    # Calculate strategy returns
    strategy_returns = np.sum(signals[:-1] * returns[1:], axis=1)
    
    # Calculate performance metrics
    sharpe = calculate_sharpe_ratio(strategy_returns)
    calmar = calculate_calmar_ratio(strategy_returns)
    
    return {
        'signals': signals,
        'strategy_returns': strategy_returns,
        'sharpe_ratio': sharpe,
        'calmar_ratio': calmar,
        'cumulative_return': (1 + strategy_returns).cumprod()[-1] - 1 if len(strategy_returns) > 0 else 0
    }

def risk_parity(market_data, lookback=252):
    """Risk parity strategy"""
    returns = market_data.values
    
    # Calculate rolling volatility
    volatility = np.zeros_like(returns)
    for i in range(lookback, len(returns)):
        volatility[i] = np.std(returns[i-lookback:i], axis=0)
    
    # Generate signals inversely proportional to volatility
    signals = np.zeros_like(volatility)
    for i in range(lookback, len(returns)):
        if np.sum(volatility[i]) > 0:
            signals[i] = 1 / (volatility[i] + 1e-8)
            # Normalize signals to sum to 1
            signals[i] = signals[i] / np.sum(signals[i])
    
    # Calculate strategy returns
    strategy_returns = np.sum(signals[:-1] * returns[1:], axis=1)
    
    # Calculate performance metrics
    sharpe = calculate_sharpe_ratio(strategy_returns[lookback:])
    calmar = calculate_calmar_ratio(strategy_returns[lookback:])
    
    return {
        'signals': signals,
        'strategy_returns': strategy_returns[lookback:],
        'sharpe_ratio': sharpe,
        'calmar_ratio': calmar,
        'cumulative_return': (1 + strategy_returns[lookback:]).cumprod()[-1] - 1 if len(strategy_returns) > lookback else 0
    }

def no_transfer_lstm(market_data, seq_length=20, hidden_size=50, num_epochs=30, batch_size=16, learning_rate=0.001):
    """No transfer LSTM strategy (independent LSTM for each market)"""
    # Prepare data
    data = market_data.values
    
    # Create sequences
    sequences = []
    for i in range(len(data) - seq_length - 1):
        seq = data[i:i+seq_length+1]
        sequences.append(seq)
        
    if not sequences:
        print("Warning: Insufficient data for No Transfer LSTM")
        return {
            'signals': np.array([]),
            'strategy_returns': np.array([]),
            'sharpe_ratio': 0,
            'calmar_ratio': 0,
            'cumulative_return': 0
        }
        
    X = torch.tensor(sequences, dtype=torch.float32)
    
    # Split into input sequences and target returns
    inputs = X[:, :-1, :]
    targets = X[:, 1:, :]
    
    # Create dataset and dataloader
    dataset = TensorDataset(inputs, targets)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Define model
    input_size = data.shape[1]
    model = nn.Sequential(
        nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True),
        nn.Linear(hidden_size, input_size),
        nn.Tanh()
    )
    
    # Define loss function and optimizer
    criterion = SharpeLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        
        for inputs, targets in dataloader:
            optimizer.zero_grad()
            
            # Forward pass (handle LSTM output tuple)
            lstm_out, _ = model[0](inputs)
            outputs = model[2](model[1](lstm_out))
            
            # Compute loss
            loss = criterion(outputs, targets)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
    
    # Evaluation
    model.eval()
    with torch.no_grad():
        all_signals = []
        all_returns = []
        
        for i in range(len(data) - seq_length):
            sequence = torch.tensor(data[i:i+seq_length], dtype=torch.float32).unsqueeze(0)
            lstm_out, _ = model[0](sequence)
            signal = model[2](model[1](lstm_out))
            
            all_signals.append(signal[0, -1].numpy())
            
            if i + seq_length < len(data):
                all_returns.append(data[i+seq_length])
        
        all_signals = np.array(all_signals)
        all_returns = np.array(all_returns)
        
        # Calculate strategy returns
        strategy_returns = np.sum(all_signals[:-1] * all_returns[1:], axis=1)
        
        # Calculate performance metrics
        sharpe = calculate_sharpe_ratio(strategy_returns)
        calmar = calculate_calmar_ratio(strategy_returns)
        
        return {
            'signals': all_signals,
            'strategy_returns': strategy_returns,
            'sharpe_ratio': sharpe,
            'calmar_ratio': calmar,
            'cumulative_return': (1 + strategy_returns).cumprod()[-1] - 1 if len(strategy_returns) > 0 else 0
        }

# Main execution
def main():
    # Generate synthetic data
    print("Generating synthetic market data...")
    markets = generate_multiple_markets(num_markets=5, assets_range=(20, 30), days_range=(800, 1000))
    
    # Process markets data
    markets_data = {}
    market_configs = {}
    
    for i, market_data in enumerate(markets):
        market_id = f'Market_{i+1}'
        markets_data[market_id] = market_data
        
        num_assets = market_data.shape[1]
        # Make sure all markets use the same hidden size to avoid dimension mismatch
        hidden_size = 50  # Fixed hidden size for all markets
        
        market_configs[market_id] = {
            'input_size': num_assets,
            'output_size': num_assets,
            'hidden_size': hidden_size
        }
    
    # Initialize QuantNet model
    print("Initializing QuantNet model...")
    model = QuantNet(
        market_configs=market_configs,
        transfer_size=10,
        encoder_layers=1,
        decoder_layers=1,
        dropout=0.2
    )
    
    # Train the model
    print("Training QuantNet model...")
    trained_model = train_quantnet(
        model=model,
        markets_data=markets_data,
        num_epochs=30,
        batch_size=16,
        seq_length=20,
        learning_rate=0.001
    )
    
    # Evaluate strategies
    print("\nEvaluating strategies across markets...")
    results = {}
    
    for market_id, market_data in markets_data.items():
        print(f"\nResults for {market_id}:")
        
        # Evaluate strategies
        quantnet_results = evaluate_strategy(trained_model, market_data, market_id)
        tsmom_results = time_series_momentum(market_data)
        csmom_results = cross_sectional_momentum(market_data)
        bnh_results = buy_and_hold(market_data)
        rp_results = risk_parity(market_data)
        no_transfer_results = no_transfer_lstm(market_data)
        
        # Store results
        results[market_id] = {
            'QuantNet': quantnet_results,
            'Time Series Momentum': tsmom_results,
            'Cross-Sectional Momentum': csmom_results,
            'Buy and Hold': bnh_results,
            'Risk Parity': rp_results,
            'No Transfer LSTM': no_transfer_results
        }
        
        # Print performance metrics
        print(f"Strategy\t\tSharpe Ratio\tCalmar Ratio\tCumulative Return")
        print("-" * 75)
        print(f"QuantNet\t\t{quantnet_results['sharpe_ratio']:.4f}\t\t{quantnet_results['calmar_ratio']:.4f}\t\t{quantnet_results['cumulative_return']:.4f}")
        print(f"No Transfer LSTM\t{no_transfer_results['sharpe_ratio']:.4f}\t\t{no_transfer_results['calmar_ratio']:.4f}\t\t{no_transfer_results['cumulative_return']:.4f}")
        print(f"Time Series Mom\t\t{tsmom_results['sharpe_ratio']:.4f}\t\t{tsmom_results['calmar_ratio']:.4f}\t\t{tsmom_results['cumulative_return']:.4f}")
        print(f"Cross-Sectional Mom\t{csmom_results['sharpe_ratio']:.4f}\t\t{csmom_results['calmar_ratio']:.4f}\t\t{csmom_results['cumulative_return']:.4f}")
        print(f"Buy and Hold\t\t{bnh_results['sharpe_ratio']:.4f}\t\t{bnh_results['calmar_ratio']:.4f}\t\t{bnh_results['cumulative_return']:.4f}")
        print(f"Risk Parity\t\t{rp_results['sharpe_ratio']:.4f}\t\t{rp_results['calmar_ratio']:.4f}\t\t{rp_results['cumulative_return']:.4f}")
    
    # Plot cumulative returns for one market
    try:
        example_market_id = list(results.keys())[0]
        example_results = results[example_market_id]
        
        plt.figure(figsize=(12, 6))
        for strategy_name, strategy_results in example_results.items():
            cum_returns = (1 + strategy_results['strategy_returns']).cumprod() - 1
            if len(cum_returns) > 0:
                plt.plot(cum_returns, label=strategy_name)
        
        plt.title(f'Cumulative Returns for {example_market_id}')
        plt.xlabel('Trading Days')
        plt.ylabel('Cumulative Return')
        plt.legend()
        plt.grid(True)
        plt.savefig('cumulative_returns.png')
        plt.close()
        
        # Plot Sharpe ratios across markets
        plt.figure(figsize=(12, 6))
        
        strategies = list(results[list(results.keys())[0]].keys())
        sharpe_ratios = {strategy: [] for strategy in strategies}
        
        for market_id, market_results in results.items():
            for strategy, strategy_results in market_results.items():
                sharpe_ratios[strategy].append(strategy_results['sharpe_ratio'])
        
        # Calculate mean Sharpe ratios
        mean_sharpe = {strategy: np.mean(values) for strategy, values in sharpe_ratios.items()}
        
        # Plot
        plt.bar(mean_sharpe.keys(), mean_sharpe.values())
        plt.title('Average Sharpe Ratio Across Markets')
        plt.ylabel('Sharpe Ratio')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('sharpe_ratios.png')
        plt.close()
        
        # Visualize QuantNet encoder representations
        # Extract encoder outputs for each market
        encoder_outputs = {}
        
        for market_id, market_data in markets_data.items():
            # Prepare data
            data = torch.tensor(market_data.values, dtype=torch.float32)
            
            # Create a batch of sequences
            sequences = []
            for i in range(0, len(data)-20, 20):
                sequences.append(data[i:i+20])
            
            if len(sequences) == 0:
                continue
                
            sequences = torch.stack(sequences)
            
            # Get encoder output
            with torch.no_grad():
                encoder = trained_model.encoders[market_id]
                _, hidden = encoder(sequences)
                encoder_outputs[market_id] = hidden[-1].numpy()
        
        # Combine encoder outputs
        all_outputs = np.vstack([output for output in encoder_outputs.values()])
        market_labels = np.concatenate([[market_id] * len(output) for market_id, output in encoder_outputs.items()])
        
        # Apply t-SNE for dimensionality reduction
        tsne = TSNE(n_components=2, random_state=42)
        tsne_results = tsne.fit_transform(all_outputs)
        
        # Plot t-SNE results
        plt.figure(figsize=(10, 8))
        
        unique_markets = np.unique(market_labels)
        colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_markets)))
        
        for i, market_id in enumerate(unique_markets):
            mask = market_labels == market_id
            plt.scatter(tsne_results[mask, 0], tsne_results[mask, 1], color=colors[i], label=market_id, alpha=0.7)
        
        plt.title('t-SNE Visualization of QuantNet Encoder Representations')
        plt.legend()
        plt.savefig('tsne_visualization.png')
        plt.close()
        
        # If there are enough markets, perform hierarchical clustering
        if len(encoder_outputs) > 2:
            # Perform hierarchical clustering on encoder outputs
            mean_outputs = {market_id: np.mean(output, axis=0) for market_id, output in encoder_outputs.items()}
            
            # Combine mean outputs into a matrix
            market_ids = list(mean_outputs.keys())
            X = np.vstack([mean_outputs[market_id] for market_id in market_ids])
            
            # Compute linkage
            Z = linkage(X, 'ward')
            
            # Plot dendrogram
            plt.figure(figsize=(10, 6))
            dendrogram(Z, labels=market_ids)
            plt.title('Hierarchical Clustering of Markets Based on QuantNet Encoder')
            plt.ylabel('Distance')
            plt.xticks(rotation=90)
            plt.tight_layout()
            plt.savefig('hierarchical_clustering.png')
            plt.close()
    except Exception as e:
        print(f"Error during plotting: {e}")
    
    print("\nEvaluation complete. Results saved to image files.")

if __name__ == "__main__":
    main()