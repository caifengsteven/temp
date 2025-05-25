import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import r2_score, roc_auc_score
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class DeepSparseDenoiseAutoencoder(nn.Module):
    """
    Implementation of the Deep Sparse Denoising Autoencoder architecture
    as described in the paper.
    """
    def __init__(self, n_sectors, latent_dim, hidden_dim, dropout_rate=0.2):
        super(DeepSparseDenoiseAutoencoder, self).__init__()
        
        self.dropout = nn.Dropout(dropout_rate)
        
        # Encoder
        self.encoder_hidden = nn.Linear(n_sectors + 1, hidden_dim)
        self.encoder_output = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder
        self.decoder_hidden = nn.Linear(latent_dim, hidden_dim)
        self.decoder_output = nn.Linear(hidden_dim, n_sectors)
        
        # Activation function
        self.elu = nn.ELU()
        
    def encode(self, x):
        x = self.dropout(x)
        h = self.elu(self.encoder_hidden(x))
        z = self.elu(self.encoder_output(h))
        return z
    
    def decode(self, z):
        h = self.elu(self.decoder_hidden(z))
        return self.decoder_output(h)
    
    def forward(self, x):
        z = self.encode(x)
        return self.decode(z), z

class ARRModel:
    """
    Implementation of the Autoencoder Reconstruction Ratio (ARR) model
    as described in the paper.
    """
    def __init__(self, n_sectors=11, latent_dim=2, hidden_dim=6):
        """
        Initialize the ARR model.
        
        Args:
            n_sectors: Number of market sectors.
            latent_dim: Dimension of the latent space (1/5 of n_sectors as per paper).
            hidden_dim: Dimension of the hidden layer (average of n_sectors and latent_dim).
        """
        self.n_sectors = n_sectors
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.model = DeepSparseDenoiseAutoencoder(
            n_sectors, latent_dim, hidden_dim).to(device)
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=latent_dim)
        
    def train(self, returns, time_of_day, epochs=100, batch_size=512, 
             validation_split=0.2, early_stopping=True, l1_weight=0.01):
        """
        Train the autoencoder model.
        
        Args:
            returns: Array of sector returns [n_samples, n_sectors].
            time_of_day: Array of time of day values [n_samples].
            epochs: Maximum number of training epochs.
            batch_size: Training batch size.
            validation_split: Fraction of data to use for validation.
            early_stopping: Whether to use early stopping.
            l1_weight: L1 regularization weight for the latent space.
        """
        # Scale the returns data
        scaled_returns = self.scaler.fit_transform(returns)
        
        # Normalize time of day to [0, 1]
        normalized_time = time_of_day / 86400.0  # seconds in a day
        
        # Combine returns with time of day
        inputs = np.column_stack([scaled_returns, normalized_time])
        
        # Convert to PyTorch tensors
        inputs_tensor = torch.FloatTensor(inputs).to(device)
        targets_tensor = torch.FloatTensor(scaled_returns).to(device)
        
        # Create dataset and data loaders
        dataset = TensorDataset(inputs_tensor, targets_tensor)
        val_size = int(validation_split * len(dataset))
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size])
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # Define optimizer
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        patience = 10 if early_stopping else float('inf')
        
        train_losses = []
        val_losses = []
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0
            for batch_inputs, batch_targets in train_loader:
                optimizer.zero_grad()
                
                reconstructed, encoded = self.model(batch_inputs)
                
                # MSE loss
                mse_loss = nn.MSELoss()(reconstructed, batch_targets)
                
                # L1 regularization on the latent space
                l1_reg = l1_weight * torch.mean(torch.abs(encoded))
                
                # Total loss
                loss = mse_loss + l1_reg
                
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item() * batch_inputs.size(0)
            
            train_loss = train_loss / len(train_dataset)
            train_losses.append(train_loss)
            
            # Validation phase
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch_inputs, batch_targets in val_loader:
                    reconstructed, encoded = self.model(batch_inputs)
                    mse_loss = nn.MSELoss()(reconstructed, batch_targets)
                    l1_reg = l1_weight * torch.mean(torch.abs(encoded))
                    loss = mse_loss + l1_reg
                    val_loss += loss.item() * batch_inputs.size(0)
            
            val_loss = val_loss / len(val_dataset)
            val_losses.append(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model weights
                best_model_state = self.model.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f'Early stopping at epoch {epoch+1}')
                    # Restore best model weights
                    self.model.load_state_dict(best_model_state)
                    break
        
        # Ensure the best model is loaded
        if early_stopping and epoch >= patience:
            self.model.load_state_dict(best_model_state)
            
        return {'train_loss': train_losses, 'val_loss': val_losses}
    
    def compute_arr(self, returns, time_of_day, interval_timesteps=300):
        """
        Compute the Autoencoder Reconstruction Ratio (ARR).
        
        Args:
            returns: Array of sector returns [n_samples, n_sectors].
            time_of_day: Array of time of day values [n_samples].
            interval_timesteps: Number of timesteps for ARR calculation.
            
        Returns:
            Array of ARR values for each interval.
        """
        # Scale the returns data
        scaled_returns = self.scaler.transform(returns)
        
        # Normalize time of day to [0, 1]
        normalized_time = time_of_day / 86400.0
        
        # Combine returns with time of day
        inputs = np.column_stack([scaled_returns, normalized_time])
        
        # Convert to PyTorch tensors
        inputs_tensor = torch.FloatTensor(inputs).to(device)
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Compute reconstructions
        with torch.no_grad():
            reconstructed, _ = self.model(inputs_tensor)
            
        # Convert reconstructions back to numpy
        reconstructed = reconstructed.cpu().numpy()
        
        # Calculate ARR for each interval
        arr_values = []
        for i in range(0, len(returns), interval_timesteps):
            end_idx = min(i + interval_timesteps, len(returns))
            interval_returns = returns[i:end_idx]
            interval_reconstructed = reconstructed[i:end_idx]
            
            # Calculate reconstruction error (numerator of ARR)
            reconstruction_error = np.sum(np.square(interval_returns - interval_reconstructed))
            
            # Calculate total variance (denominator of ARR)
            total_variance = np.sum(np.square(interval_returns))
            
            # Calculate ARR
            arr = reconstruction_error / (total_variance + 1e-10)  # Add epsilon to avoid division by zero
            arr_values.append(arr)
            
        return np.array(arr_values)
    
    def compute_absorption_ratio_pca(self, returns):
        """
        Compute the traditional Absorption Ratio using PCA.
        
        Args:
            returns: Array of sector returns [n_samples, n_sectors].
            
        Returns:
            Absorption Ratio value.
        """
        # Fit PCA to returns
        self.pca.fit(returns)
        
        # Calculate Absorption Ratio
        total_variance = np.sum(self.pca.explained_variance_)
        absorbed_variance = np.sum(self.pca.explained_variance_[:self.latent_dim])
        
        return absorbed_variance / total_variance
    
    def compute_reconstruction_r2(self, returns, time_of_day):
        """
        Compute the R-squared of the autoencoder reconstruction.
        
        Args:
            returns: Array of sector returns [n_samples, n_sectors].
            time_of_day: Array of time of day values [n_samples].
            
        Returns:
            R-squared value.
        """
        # Scale the returns data
        scaled_returns = self.scaler.transform(returns)
        
        # Normalize time of day to [0, 1]
        normalized_time = time_of_day / 86400.0
        
        # Combine returns with time of day
        inputs = np.column_stack([scaled_returns, normalized_time])
        
        # Convert to PyTorch tensors
        inputs_tensor = torch.FloatTensor(inputs).to(device)
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Compute reconstructions
        with torch.no_grad():
            reconstructed, _ = self.model(inputs_tensor)
            
        # Convert reconstructions back to numpy
        reconstructed = reconstructed.cpu().numpy()
        
        # Inverse transform the scaled data
        reconstructed_orig = self.scaler.inverse_transform(reconstructed)
        
        # Calculate R-squared
        r2 = r2_score(returns, reconstructed_orig)
        
        return r2

# ---------------------------------------------------------------
# ARR-Based Trading Strategies
# ---------------------------------------------------------------

def arr_based_asset_allocation(arr_values, arr_lookback=10, high_threshold=0.6, low_threshold=0.4):
    """
    Dynamic asset allocation based on ARR signals.
    
    Args:
        arr_values: Series of ARR values
        arr_lookback: Lookback period for ARR trend
        high_threshold: Threshold for high ARR (market stress)
        low_threshold: Threshold for low ARR (market stability)
    
    Returns:
        Allocation percentages for stocks, bonds, and cash
    """
    if len(arr_values) < arr_lookback:
        # Not enough data for analysis
        return {
            'stocks': 0.60,
            'bonds': 0.30,
            'cash': 0.10
        }
    
    # Get the current ARR and its moving average
    current_arr = arr_values[-1]
    arr_ma = np.mean(arr_values[-arr_lookback:])
    arr_trend = current_arr - arr_ma  # Positive means increasing fragility
    
    # Base allocation
    allocation = {
        'stocks': 0.60,
        'bonds': 0.30,
        'cash': 0.10
    }
    
    # Adjust based on ARR level and trend
    if current_arr > high_threshold:
        # High market fragility - reduce risk
        if arr_trend > 0:
            # Fragility still increasing - defensive positioning
            allocation['stocks'] = 0.30
            allocation['bonds'] = 0.40
            allocation['cash'] = 0.30
        else:
            # High but stabilizing - moderately defensive
            allocation['stocks'] = 0.40
            allocation['bonds'] = 0.40
            allocation['cash'] = 0.20
    elif current_arr < low_threshold:
        # Low market fragility - increase risk
        if arr_trend < 0:
            # Fragility still decreasing - aggressive positioning
            allocation['stocks'] = 0.80
            allocation['bonds'] = 0.15
            allocation['cash'] = 0.05
        else:
            # Low but rising - moderately aggressive
            allocation['stocks'] = 0.70
            allocation['bonds'] = 0.20
            allocation['cash'] = 0.10
    
    return allocation

def arr_volatility_targeting(arr_values, target_volatility=0.10, max_leverage=2.0, min_allocation=0.2):
    """
    Adjust portfolio allocation to target a specific volatility level based on ARR.
    
    Args:
        arr_values: Series of ARR values
        target_volatility: Target annualized volatility
        max_leverage: Maximum allowable leverage
        min_allocation: Minimum allocation to risky assets
    
    Returns:
        Target allocation percentage for risky assets
    """
    if len(arr_values) < 1:
        # Not enough data, return default allocation
        return 1.0
    
    # Use ARR to forecast future volatility
    # Higher ARR suggests higher future volatility
    current_arr = arr_values[-1]
    
    # Simple model: expected volatility scales with ARR
    # This would be improved with a proper calibration
    expected_daily_vol = 0.01 * (1 + current_arr * 2)  # Example scaling
    expected_annual_vol = expected_daily_vol * np.sqrt(252)
    
    # Calculate allocation to achieve target volatility
    allocation = target_volatility / expected_annual_vol
    
    # Apply constraints
    allocation = max(min_allocation, min(allocation, max_leverage))
    
    return allocation

def arr_sector_rotation(arr_values, sector_returns, lookback=20):
    """
    Implement sector rotation based on ARR and recent sector performance.
    
    Args:
        arr_values: Series of ARR values
        sector_returns: Matrix of sector returns [samples, sectors]
        lookback: Period for evaluating recent sector performance
    
    Returns:
        Dictionary of sector weights
    """
    if len(arr_values) < 1 or sector_returns.shape[0] < lookback:
        # Not enough data, equal weight sectors
        n_sectors = sector_returns.shape[1]
        equal_weight = 1.0 / n_sectors
        return {f'sector_{i}': equal_weight for i in range(n_sectors)}
    
    current_arr = arr_values[-1]
    
    # Calculate sector momentum
    sector_momentum = np.mean(sector_returns[-lookback:], axis=0)
    
    # Calculate sector correlation matrix
    sector_corr = np.corrcoef(sector_returns[-lookback:].T)
    
    # Calculate sector average correlation (connectedness)
    sector_connectedness = np.mean(sector_corr, axis=1)
    
    # In high ARR regimes (high fragility), prefer less connected sectors
    # In low ARR regimes, momentum matters more
    
    if current_arr > 0.6:  # High fragility
        # Give more weight to less connected sectors
        weights = 1 - sector_connectedness
    else:  # Normal or low fragility
        # Balance momentum and connectedness
        normalized_momentum = (sector_momentum - np.mean(sector_momentum)) / np.std(sector_momentum)
        normalized_connectedness = (sector_connectedness - np.mean(sector_connectedness)) / np.std(sector_connectedness)
        weights = normalized_momentum - 0.5 * normalized_connectedness
    
    # Ensure positive weights and normalize to sum to 1
    weights = np.maximum(weights, 0)
    weights = weights / np.sum(weights)
    
    # Create sector weights dictionary
    sector_weights = {f'sector_{i}': weights[i] for i in range(len(weights))}
    
    return sector_weights

def arr_options_hedging(arr_values, arr_lookback=10, high_threshold=0.7):
    """
    Determine options hedging allocation based on ARR signals.
    
    Args:
        arr_values: Series of ARR values
        arr_lookback: Lookback period for ARR trend
        high_threshold: Threshold for high ARR (market stress)
    
    Returns:
        Dictionary with hedging recommendations
    """
    if len(arr_values) < arr_lookback:
        # Not enough data, minimal hedging
        return {
            'put_option_allocation': 0.01,
            'vix_products_allocation': 0.00,
            'put_option_delta': -0.30,
            'put_option_expiration': '1-month',
        }
        
    current_arr = arr_values[-1]
    arr_ma = np.mean(arr_values[-arr_lookback:])
    arr_trend = current_arr - arr_ma  # Positive means increasing fragility
    
    hedging_strategy = {
        'put_option_allocation': 0.00,  # % of portfolio value
        'vix_products_allocation': 0.00,
        'put_option_delta': -0.30,  # Target delta for options
        'put_option_expiration': '1-month',  # Expiration timeframe
    }
    
    # Increase hedging when ARR is high or rising rapidly
    if current_arr > high_threshold:
        # High market fragility - significant hedging
        hedging_strategy['put_option_allocation'] = 0.05  # 5% of portfolio
        hedging_strategy['vix_products_allocation'] = 0.02  # 2% of portfolio
        
        # Deep out-of-money puts if trend is strong
        if arr_trend > 0.05:
            hedging_strategy['put_option_delta'] = -0.15
            hedging_strategy['put_option_expiration'] = '2-month'  # Longer expiration
    elif arr_trend > 0.10:
        # Rapid increase in ARR - moderate hedging
        hedging_strategy['put_option_allocation'] = 0.03
        hedging_strategy['vix_products_allocation'] = 0.01
    
    return hedging_strategy

def arr_cross_asset_strategy(arr_values, lookback=15):
    """
    Cross-asset strategy based on ARR signals.
    
    Args:
        arr_values: Series of ARR values
        lookback: Period for evaluating ARR trend
    
    Returns:
        Dictionary with cross-asset trade recommendations
    """
    if len(arr_values) < lookback:
        # Not enough data, neutral positions
        return {
            'equities': 0,
            'treasuries': 0,
            'gold': 0,
            'usd': 0,
            'volatility': 0
        }
        
    # Calculate ARR trend
    current_arr = arr_values[-1]
    arr_ma = np.mean(arr_values[-lookback:])
    arr_trend = (current_arr - arr_ma) / arr_ma  # Percentage change
    
    strategy = {
        'equities': 0,       # -1 (short), 0 (neutral), 1 (long)
        'treasuries': 0,
        'gold': 0,
        'usd': 0,
        'volatility': 0
    }
    
    # Rapid increase in ARR often precedes volatility spike
    if arr_trend > 0.15:
        strategy['equities'] = -1       # Short equities
        strategy['treasuries'] = 1      # Long treasuries (flight to safety)
        strategy['gold'] = 1            # Long gold (safe haven)
        strategy['usd'] = 1             # Long USD (typically strengthens in stress)
        strategy['volatility'] = 1      # Long volatility
    
    # Decreasing ARR after high values often signals recovery
    elif current_arr > 0.6 and arr_trend < -0.15:
        strategy['equities'] = 1        # Long equities (recovery)
        strategy['treasuries'] = -1     # Short treasuries (yield increase)
        strategy['volatility'] = -1     # Short volatility
    
    # Consistently low ARR suggests stable markets
    elif current_arr < 0.3 and abs(arr_trend) < 0.05:
        strategy['equities'] = 1        # Long equities (stability)
        strategy['volatility'] = -1     # Short volatility (stable)
    
    return strategy

class ARRTradingSystem:
    """
    Complete trading system using ARR as the primary signal generator.
    """
    def __init__(self, lookback_period=20, arr_high_threshold=0.65, arr_low_threshold=0.35):
        self.lookback_period = lookback_period
        self.arr_high_threshold = arr_high_threshold
        self.arr_low_threshold = arr_low_threshold
        self.arr_values = []
        self.model = None
        self.scaler = None
        self.last_signals = None
        self.historical_signals = []
        self.historical_positions = []
        self.current_positions = {
            'stocks': 0.6,
            'bonds': 0.3,
            'cash': 0.1,
            'hedges': 0,
            'sectors': {}
        }
        
    def fit(self, sector_returns, time_of_day):
        """Train the ARR model on historical data."""
        # Initialize and train the ARR model
        n_sectors = sector_returns.shape[1]
        latent_dim = max(2, n_sectors // 5)
        hidden_dim = (n_sectors + latent_dim) // 2
        
        print("Training ARR model...")
        self.model = ARRModel(n_sectors=n_sectors, latent_dim=latent_dim, hidden_dim=hidden_dim)
        history = self.model.train(sector_returns, time_of_day, epochs=100, batch_size=512, 
                                  validation_split=0.2, early_stopping=True)
        
        # Initialize ARR values array
        arr = self.model.compute_arr(sector_returns, time_of_day, interval_timesteps=5)
        self.arr_values = arr.tolist()
        
        print(f"Model trained. Initial ARR value: {self.arr_values[-1]:.4f}")
        return history
    
    def update(self, new_sector_returns, new_time_of_day, interval_size=5):
        """Update ARR values with new market data."""
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
            
        # Compute ARR for the new data
        new_arr = self.model.compute_arr(new_sector_returns, new_time_of_day, interval_timesteps=interval_size)
        self.arr_values.extend(new_arr.tolist())
        
        # Keep only the most recent values
        if len(self.arr_values) > self.lookback_period * 10:
            self.arr_values = self.arr_values[-self.lookback_period * 10:]
            
        print(f"Updated ARR. Current value: {self.arr_values[-1]:.4f}")
        
    def generate_signals(self, sector_returns=None):
        """Generate trading signals based on current ARR values."""
        if len(self.arr_values) < self.lookback_period:
            return {"status": "insufficient_data"}
        
        arr_array = np.array(self.arr_values)
            
        # Asset allocation
        allocation = arr_based_asset_allocation(
            arr_array, 
            arr_lookback=self.lookback_period,
            high_threshold=self.arr_high_threshold,
            low_threshold=self.arr_low_threshold
        )
        
        # Volatility targeting
        vol_target = arr_volatility_targeting(arr_array)
        
        # Options hedging
        hedging = arr_options_hedging(arr_array, self.lookback_period)
        
        # Cross-asset signals
        cross_asset = arr_cross_asset_strategy(arr_array)
        
        # Sector rotation (if sector returns provided)
        sector_weights = {}
        if sector_returns is not None and sector_returns.shape[0] > self.lookback_period:
            sector_weights = arr_sector_rotation(arr_array, sector_returns, self.lookback_period)
        
        signals = {
            "status": "success",
            "asset_allocation": allocation,
            "volatility_target": vol_target,
            "hedging_strategy": hedging,
            "cross_asset_signals": cross_asset,
            "sector_weights": sector_weights,
            "current_arr": arr_array[-1],
            "arr_trend": arr_array[-1] - np.mean(arr_array[-self.lookback_period:])
        }
        
        self.last_signals = signals
        self.historical_signals.append(signals)
        return signals
    
    def execute_signals(self, trading_costs=0.001, slippage=0.001):
        """
        Execute the trading signals by updating positions.
        
        Args:
            trading_costs: Cost per trade as a fraction of trade value
            slippage: Expected slippage as a fraction of trade value
            
        Returns:
            Dictionary with details of executed trades
        """
        if self.last_signals is None or self.last_signals["status"] != "success":
            return {"status": "no_signals"}
        
        # Save current positions before update
        old_positions = self.current_positions.copy()
        
        # Update asset allocation
        allocation = self.last_signals["asset_allocation"]
        self.current_positions["stocks"] = allocation["stocks"]
        self.current_positions["bonds"] = allocation["bonds"]
        self.current_positions["cash"] = allocation["cash"]
        
        # Update hedging allocation
        hedging = self.last_signals["hedging_strategy"]
        self.current_positions["hedges"] = hedging["put_option_allocation"] + hedging["vix_products_allocation"]
        
        # Update sector weights if available
        if self.last_signals["sector_weights"]:
            self.current_positions["sectors"] = self.last_signals["sector_weights"]
        
        # Calculate trades and costs
        trades = {}
        total_cost = 0
        
        for asset, new_value in self.current_positions.items():
            if asset != "sectors":  # Handle sectors separately
                old_value = old_positions.get(asset, 0)
                trade_size = abs(new_value - old_value)
                if trade_size > 0:
                    cost = trade_size * (trading_costs + slippage)
                    trades[asset] = {
                        "old_position": old_value,
                        "new_position": new_value,
                        "trade_size": trade_size,
                        "trade_cost": cost
                    }
                    total_cost += cost
        
        # Handle sector trades if they exist
        if "sectors" in self.current_positions and "sectors" in old_positions:
            sector_trades = {}
            for sector, weight in self.current_positions["sectors"].items():
                old_weight = old_positions["sectors"].get(sector, 0)
                trade_size = abs(weight - old_weight)
                if trade_size > 0:
                    cost = trade_size * (trading_costs + slippage)
                    sector_trades[sector] = {
                        "old_weight": old_weight,
                        "new_weight": weight,
                        "trade_size": trade_size,
                        "trade_cost": cost
                    }
                    total_cost += cost
            trades["sectors"] = sector_trades
        
        # Save current positions for historical tracking
        self.historical_positions.append(self.current_positions.copy())
        
        return {
            "status": "executed",
            "trades": trades,
            "total_cost": total_cost,
            "current_positions": self.current_positions,
            "arr_value": self.last_signals["current_arr"],
            "timestamp": pd.Timestamp.now()
        }
    
    def backtest(self, sector_returns, time_of_day, market_returns=None, rebalance_interval=5, transaction_costs=0.001):
        """
        Backtest the ARR trading strategy on historical data.
        
        Args:
            sector_returns: Sector returns data [samples, n_sectors]
            time_of_day: Time of day for each sample
            market_returns: Market index returns (if available)
            rebalance_interval: Number of samples between rebalancing
            transaction_costs: Transaction costs as fraction of trade value
            
        Returns:
            Dictionary with backtest results
        """
        if self.model is None:
            print("Training model on backtest data...")
            self.fit(sector_returns, time_of_day)
        
        # If market returns not provided, use average of sector returns
        if market_returns is None:
            market_returns = np.mean(sector_returns, axis=1)
        
        print("Starting backtest...")
        
        # Initialize backtest results
        results = {
            "timestamps": [],
            "arr_values": [],
            "portfolio_values": [],
            "benchmark_values": [],
            "positions": [],
            "signals": [],
            "trades": []
        }
        
        # Initial portfolio value
        portfolio_value = 100.0
        benchmark_value = 100.0
        
        # Loop through the data
        interval_size = 5  # 5-sample intervals for ARR calculation
        n_samples = len(market_returns)
        
        # For each rebalance date
        for i in range(0, n_samples, rebalance_interval):
            end_idx = min(i + rebalance_interval, n_samples)
            interval_returns = sector_returns[i:end_idx]
            interval_time = time_of_day[i:end_idx]
            
            # Update ARR values
            self.update(interval_returns, interval_time, interval_size)
            
            # Generate signals
            signals = self.generate_signals(sector_returns[:end_idx])
            
            # Execute trades
            if signals["status"] == "success":
                trade_result = self.execute_signals(trading_costs=transaction_costs)
                results["trades"].append(trade_result)
                
                # Calculate portfolio return based on allocation
                period_market_return = np.prod(1 + market_returns[i:end_idx]) - 1
                
                # Adjust return based on allocation and hedging
                stock_allocation = signals["asset_allocation"]["stocks"]
                hedge_effect = 0
                
                # Hedges reduce losses during market downturns
                if period_market_return < 0:
                    hedge_allocation = signals["hedging_strategy"]["put_option_allocation"]
                    hedge_effect = -period_market_return * hedge_allocation * 5  # Approximate options leverage
                
                # Portfolio return based on allocation and hedging
                portfolio_return = (period_market_return * stock_allocation) + hedge_effect
                
                # Apply trading costs
                portfolio_return -= trade_result.get("total_cost", 0)
                
                # Update portfolio value
                portfolio_value *= (1 + portfolio_return)
                benchmark_value *= (1 + period_market_return)
            
            # Record results
            results["timestamps"].append(i)
            results["arr_values"].append(self.arr_values[-1] if self.arr_values else 0)
            results["portfolio_values"].append(portfolio_value)
            results["benchmark_values"].append(benchmark_value)
            results["positions"].append(self.current_positions.copy())
            results["signals"].append(signals)
        
        # Calculate performance metrics
        returns = np.diff(results["portfolio_values"]) / results["portfolio_values"][:-1]
        benchmark_returns = np.diff(results["benchmark_values"]) / results["benchmark_values"][:-1]
        
        annualized_return = np.mean(returns) * 252 / rebalance_interval  # Assuming daily data
        annualized_vol = np.std(returns) * np.sqrt(252 / rebalance_interval)
        sharpe_ratio = annualized_return / annualized_vol if annualized_vol > 0 else 0
        
        benchmark_ann_return = np.mean(benchmark_returns) * 252 / rebalance_interval
        benchmark_ann_vol = np.std(benchmark_returns) * np.sqrt(252 / rebalance_interval)
        benchmark_sharpe = benchmark_ann_return / benchmark_ann_vol if benchmark_ann_vol > 0 else 0
        
        max_drawdown = 0
        peak = results["portfolio_values"][0]
        for value in results["portfolio_values"]:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            max_drawdown = max(max_drawdown, drawdown)
        
        # Add performance metrics to results
        results["performance"] = {
            "final_value": portfolio_value,
            "benchmark_final_value": benchmark_value,
            "total_return": (portfolio_value / 100) - 1,
            "benchmark_return": (benchmark_value / 100) - 1,
            "annualized_return": annualized_return,
            "annualized_volatility": annualized_vol,
            "sharpe_ratio": sharpe_ratio,
            "benchmark_sharpe": benchmark_sharpe,
            "max_drawdown": max_drawdown
        }
        
        print("Backtest completed.")
        print(f"Final portfolio value: ${portfolio_value:.2f}")
        print(f"Benchmark value: ${benchmark_value:.2f}")
        print(f"Annualized return: {annualized_return*100:.2f}%")
        print(f"Sharpe ratio: {sharpe_ratio:.2f}")
        print(f"Maximum drawdown: {max_drawdown*100:.2f}%")
        
        self.plot_backtest_results(results)
        
        return results
    
    def plot_backtest_results(self, results):
        """
        Plot backtest results.
        
        Args:
            results: Dictionary of backtest results
        """
        plt.figure(figsize=(14, 16))
        
        # Plot 1: Portfolio vs Benchmark
        plt.subplot(3, 1, 1)
        plt.plot(results["timestamps"], results["portfolio_values"], label="ARR Strategy")
        plt.plot(results["timestamps"], results["benchmark_values"], label="Benchmark", alpha=0.7)
        plt.title("Portfolio Performance")
        plt.ylabel("Value ($)")
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Plot 2: ARR Values
        plt.subplot(3, 1, 2)
        plt.plot(results["timestamps"], results["arr_values"], color="purple")
        plt.title("Autoencoder Reconstruction Ratio (ARR)")
        plt.ylabel("ARR")
        plt.axhline(y=self.arr_high_threshold, color="red", linestyle="--", alpha=0.5)
        plt.axhline(y=self.arr_low_threshold, color="green", linestyle="--", alpha=0.5)
        plt.grid(True, alpha=0.3)
        
        # Plot 3: Asset Allocation
        plt.subplot(3, 1, 3)
        
        # Extract asset allocations
        timestamps = results["timestamps"]
        stocks = [pos.get("stocks", 0) for pos in results["positions"]]
        bonds = [pos.get("bonds", 0) for pos in results["positions"]]
        cash = [pos.get("cash", 0) for pos in results["positions"]]
        hedges = [pos.get("hedges", 0) for pos in results["positions"]]
        
        plt.stackplot(timestamps, stocks, bonds, cash, hedges, 
                      labels=["Stocks", "Bonds", "Cash", "Hedges"],
                      alpha=0.7, colors=["blue", "green", "gray", "red"])
        plt.title("Asset Allocation Over Time")
        plt.ylabel("Allocation")
        plt.ylim(0, 1.1)
        plt.grid(True, alpha=0.3)
        plt.legend(loc="upper left")
        
        plt.tight_layout()
        plt.show()

# ---------------------------------------------------------------
# Simulation and Backtesting
# ---------------------------------------------------------------

def generate_simulated_market_data(n_days=1000, n_sectors=11, sectors_per_factor=3, 
                                  samples_per_day=390, volatility_regimes=True):
    """
    Generate simulated market data with sector returns exhibiting co-movement.
    
    Args:
        n_days: Number of trading days to simulate.
        n_sectors: Number of market sectors.
        sectors_per_factor: Number of sectors influenced by each latent factor.
        samples_per_day: Number of intraday samples per day (e.g., 390 for 1-min data in 6.5 hour trading day).
        volatility_regimes: Whether to include volatility regime switches.
        
    Returns:
        Dictionary containing simulated data.
    """
    n_samples = n_days * samples_per_day
    
    # Time information
    days = np.repeat(np.arange(n_days), samples_per_day)
    intraday_time = np.tile(np.arange(samples_per_day), n_days)
    time_of_day = intraday_time * 60  # in seconds from market open
    
    # Create latent factors (2 market-wide factors + sector-specific factors)
    n_factors = 2 + n_sectors // sectors_per_factor
    
    # Base volatility for each factor
    factor_volatility = 0.0005  # Base volatility per tick
    
    # Define volatility regimes if enabled
    if volatility_regimes:
        # Create periods of high volatility
        regime_changes = np.random.randint(0, n_days, size=15)  # 15 volatility regime changes
        high_vol_periods = []
        
        for rc in regime_changes:
            duration = np.random.randint(3, 15)  # High volatility for 3-15 days
            high_vol_periods.append((rc, rc + duration))
        
        # Create volatility multiplier array
        vol_multiplier = np.ones(n_days)
        for start, end in high_vol_periods:
            if start < n_days:
                end = min(end, n_days)
                # Higher volatility during stress periods (2-5x normal)
                vol_multiplier[start:end] = np.random.uniform(2, 5)
        
        # Expand to match per-sample level
        vol_multiplier = np.repeat(vol_multiplier, samples_per_day)
    else:
        vol_multiplier = np.ones(n_samples)
    
    # Create factors with intraday patterns
    factors = np.zeros((n_samples, n_factors))
    
    # Two market-wide factors
    for i in range(2):
        # Random walks with intraday patterns
        daily_changes = np.random.normal(0, factor_volatility, size=n_days)
        daily_changes = np.repeat(daily_changes, samples_per_day)
        
        # Add intraday pattern: higher volatility at open and close
        intraday_pattern = np.ones(samples_per_day)
        # U-shaped volatility pattern
        intraday_pattern[:30] = np.linspace(2.0, 1.0, 30)  # Higher at open
        intraday_pattern[-30:] = np.linspace(1.0, 1.8, 30)  # Higher at close
        
        intraday_vol = np.tile(intraday_pattern, n_days)
        
        # Combined volatility
        combined_vol = factor_volatility * intraday_vol * vol_multiplier
        
        # Create factor as cumulative sum of random changes
        factor_changes = np.random.normal(0, combined_vol)
        factors[:, i] = np.cumsum(factor_changes)
    
    # Sector-specific factors
    for i in range(2, n_factors):
        # More idiosyncratic behavior
        sector_vol = factor_volatility * 0.8  # Slightly lower than market factors
        daily_changes = np.random.normal(0, sector_vol, size=n_days)
        daily_changes = np.repeat(daily_changes, samples_per_day)
        
        # Add some random jumps (company-specific news)
        jumps = np.zeros(n_samples)
        jump_idx = np.random.choice(n_samples, size=20, replace=False)
        jumps[jump_idx] = np.random.normal(0, sector_vol*10, size=20)
        
        # Combined volatility
        combined_vol = sector_vol * vol_multiplier
        
        # Create factor
        factor_changes = np.random.normal(0, combined_vol) + jumps
        factors[:, i] = np.cumsum(factor_changes)
    
    # Create sector returns from factors
    sector_returns = np.zeros((n_samples, n_sectors))
    
    # Betas for market factors (all sectors have exposure to market factors)
    market_betas = np.random.uniform(0.5, 1.5, size=(n_sectors, 2))
    
    # Betas for sector factors (each sector mainly affected by one sector factor)
    sector_factor_betas = np.zeros((n_sectors, n_factors - 2))
    
    # Assign sectors to factors with some overlap
    for i in range(n_sectors):
        # Primary factor influence
        primary_factor = i // sectors_per_factor
        if primary_factor < n_factors - 2:
            sector_factor_betas[i, primary_factor] = np.random.uniform(0.8, 1.2)
            
            # Some influence from other factors
            for j in range(n_factors - 2):
                if j != primary_factor:
                    # Smaller influence from other factors
                    sector_factor_betas[i, j] = np.random.uniform(0, 0.3)
    
    # Combine all betas
    all_betas = np.column_stack([market_betas, sector_factor_betas])
    
    # Generate sector returns
    for i in range(n_sectors):
        # Systematic component (factor-driven)
        systematic_returns = np.zeros(n_samples)
        for j in range(n_factors):
            systematic_returns += all_betas[i, j] * (factors[:, j] - np.roll(factors[:, j], 1))
        
        # Add idiosyncratic noise
        idiosyncratic_vol = factor_volatility * 0.5 * vol_multiplier  # Lower than factor vols
        idiosyncratic_returns = np.random.normal(0, idiosyncratic_vol)
        
        # Total returns
        sector_returns[:, i] = systematic_returns + idiosyncratic_returns
    
    # First differences won't have a valid value, set to 0
    sector_returns[0, :] = 0
    
    # Calculate the "market" return as average of all sectors
    market_returns = np.mean(sector_returns, axis=1)
    
    # Create a realized volatility series
    window_sizes = {
        '5min': 5,
        '1hour': 60,
        '1day': samples_per_day,
        '1week': samples_per_day * 5
    }
    
    realized_volatility = {}
    for window_name, window_size in window_sizes.items():
        rv = np.zeros(n_samples)
        for i in range(window_size, n_samples):
            rv[i] = np.sqrt(np.sum(market_returns[i-window_size:i]**2))
        realized_volatility[window_name] = rv
    
    # Create drawdowns
    market_price = 100 * np.exp(np.cumsum(market_returns))
    rolling_max = np.maximum.accumulate(market_price)
    drawdowns = (market_price - rolling_max) / rolling_max
    
    # Return the data
    return {
        'sector_returns': sector_returns,
        'market_returns': market_returns,
        'days': days,
        'time_of_day': time_of_day,
        'realized_volatility': realized_volatility,
        'drawdowns': drawdowns,
        'market_price': market_price,
        'vol_multiplier': vol_multiplier.reshape(-1, samples_per_day).mean(axis=1)
    }

# ---------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------

if __name__ == "__main__":
    print("Simulating market data...")
    # Generate simulated market data
    sim_data = generate_simulated_market_data(n_days=500, n_sectors=11, samples_per_day=390)
    
    # Extract the data we need
    sector_returns = sim_data['sector_returns']
    market_returns = sim_data['market_returns']
    time_of_day = sim_data['time_of_day']
    
    # Create and train the ARR trading system
    trading_system = ARRTradingSystem(lookback_period=20, arr_high_threshold=0.6, arr_low_threshold=0.4)
    
    # Perform backtest
    backtest_results = trading_system.backtest(
        sector_returns, 
        time_of_day, 
        market_returns=market_returns,
        rebalance_interval=390,  # Daily rebalancing
        transaction_costs=0.001  # 10 basis points per trade
    )
    
    print("\nBacktest completed! Trading strategies based on ARR have been evaluated.")