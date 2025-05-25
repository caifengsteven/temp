import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import pdblp  # Python wrapper for Bloomberg API
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas_datareader.data as web
from urllib.error import URLError

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Parameters
lookback_period = 12  # 12 months formation period
holding_period = 1  # 1 month holding period
training_window = 300  # 300 months for training
noise_level = 0.25  # Noise level for denoising autoencoder
learning_rate = 0.0005  # As specified in the paper
batch_size = 100  # As specified in the paper
n_epochs = 300  # As specified in the paper

# Connect to Bloomberg if available
try:
    con = pdblp.BCon(timeout=5000)
    con.start()
    bloomberg_available = True
    print("Connected to Bloomberg")
except:
    bloomberg_available = False
    print("Bloomberg not available")

# PyTorch Model Definitions
class DenoisingAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout_rate=0.25):
        super(DenoisingAutoencoder, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, input_dim)
        self.activation = nn.Sigmoid()
        
    def forward(self, x, apply_noise=True):
        # Apply dropout noise during training
        if apply_noise:
            x = self.dropout(x)
        
        # Encode
        encoded = self.activation(self.encoder(x))
        
        # Decode
        decoded = self.activation(self.decoder(encoded))
        
        return decoded
    
    def encode(self, x, apply_noise=False):
        if apply_noise:
            x = self.dropout(x)
        return self.activation(self.encoder(x))


class StackedDenoisingAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_layers=(6, 6, 6), dropout_rate=0.25):
        super(StackedDenoisingAutoencoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.dropout_rate = dropout_rate
        
        # Create a list of autoencoders
        self.autoencoders = nn.ModuleList()
        
        # Create each autoencoder in the stack
        layer_dims = [input_dim] + list(hidden_layers)
        for i in range(len(layer_dims) - 1):
            self.autoencoders.append(
                DenoisingAutoencoder(layer_dims[i], layer_dims[i+1], dropout_rate)
            )
        
        # Final regression layer
        self.regression = nn.Linear(hidden_layers[-1], 1)
        
    def pretrain(self, dataloader, epochs=300, lr=0.0005):
        """Pretrain each autoencoder layer by layer"""
        current_input = None
        
        for i, autoencoder in enumerate(self.autoencoders):
            print(f"Pretraining autoencoder {i+1}/{len(self.autoencoders)}")
            
            # Move the current autoencoder to device
            autoencoder = autoencoder.to(device)
            
            # Create optimizer
            optimizer = optim.Adam(autoencoder.parameters(), lr=lr)
            
            # Train the current autoencoder
            for epoch in range(epochs):
                total_loss = 0
                for data in dataloader:
                    # Get only the data (no labels for pretraining)
                    inputs = data[0].to(device)
                    
                    # If this is not the first layer, use encoded output from the previous layer
                    if current_input is not None:
                        with torch.no_grad():
                            for prev_ae in self.autoencoders[:i]:
                                inputs = prev_ae.encode(inputs, apply_noise=False)
                    
                    # Forward pass
                    outputs = autoencoder(inputs)
                    
                    # Compute loss (reconstruction error)
                    loss = nn.MSELoss()(outputs, inputs)
                    
                    # Backward and optimize
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                
                if (epoch + 1) % 50 == 0:
                    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.6f}")
            
            # Store the encoded data for the next layer
            if i < len(self.autoencoders) - 1:
                new_dataloader = []
                with torch.no_grad():
                    for data in dataloader:
                        inputs = data[0].to(device)
                        
                        # Get encoded outputs from previous layers
                        if current_input is not None:
                            for prev_ae in self.autoencoders[:i]:
                                inputs = prev_ae.encode(inputs, apply_noise=False)
                        
                        # Encode with current autoencoder
                        encoded = autoencoder.encode(inputs, apply_noise=False)
                        new_dataloader.append((encoded, data[1]))
                
                # Create new dataloader with encoded data
                current_input = new_dataloader
    
    def forward(self, x, apply_noise=False):
        # Pass input through each autoencoder
        for autoencoder in self.autoencoders:
            x = autoencoder.encode(x, apply_noise=apply_noise)
        
        # Pass through the final regression layer
        x = self.regression(x)
        
        return x


# Training and prediction class
class DNN_SdAE:
    def __init__(self, input_dim=12, hidden_layers=(6, 6, 6), noise_level=0.25):
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.noise_level = noise_level
        self.model = StackedDenoisingAutoencoder(input_dim, hidden_layers, noise_level)
        self.scaler = MinMaxScaler()
        
    def build_and_train(self, X_train, y_train, X_val, y_val, epochs=300, batch_size=100, lr=0.0005):
        """Build and train the complete model."""
        # Scale inputs
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # Convert to PyTorch tensors
        X_train_tensor = torch.FloatTensor(X_train_scaled)
        y_train_tensor = torch.FloatTensor(y_train.reshape(-1, 1))
        X_val_tensor = torch.FloatTensor(X_val_scaled)
        y_val_tensor = torch.FloatTensor(y_val.reshape(-1, 1))
        
        # Create data loaders
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # Move model to device
        self.model = self.model.to(device)
        
        # Pretrain autoencoders
        self.model.pretrain(train_loader, epochs=epochs, lr=lr)
        
        # Fine-tune the entire model
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        
        best_val_loss = float('inf')
        patience = 20  # Increased patience
        patience_counter = 0
        best_model_state = None
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0
            for data, targets in train_loader:
                data, targets = data.to(device), targets.to(device)
                
                # Forward pass
                outputs = self.model(data)
                loss = criterion(outputs, targets)
                
                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validation phase
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for data, targets in val_loader:
                    data, targets = data.to(device), targets.to(device)
                    outputs = self.model(data)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item()
            
            # Print statistics
            if (epoch + 1) % 10 == 0:
                print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss/len(train_loader):.6f}, '
                      f'Val Loss: {val_loss/len(val_loader):.6f}')
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = self.model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f'Early stopping at epoch {epoch+1}')
                    break
        
        # Load best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        
        return self.model
    
    def predict(self, X):
        """Predict using the trained model."""
        # Scale input
        X_scaled = self.scaler.transform(X)
        
        # Convert to tensor
        X_tensor = torch.FloatTensor(X_scaled).to(device)
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Make prediction
        with torch.no_grad():
            predictions = self.model(X_tensor)
        
        # Return numpy array
        return predictions.cpu().numpy().flatten()


# Selective Momentum Strategy Implementation
class SelectiveMomentumStrategy:
    def __init__(self, prediction_model=None, volatility_scaling=True, target_vol=0.12):
        self.prediction_model = prediction_model
        self.volatility_scaling = volatility_scaling
        self.target_vol = target_vol
        self.positions = None
        self.returns = None
    
    def calculate_volatility_scaling(self, daily_returns, window=126):
        """Calculate volatility scaling factor based on past 6 months of daily returns."""
        # Assuming 21 trading days per month
        rolling_variance = daily_returns.rolling(window=window).var()
        target_variance = (self.target_vol**2) / 252  # Daily target variance
        scaling_factor = np.sqrt(target_variance / rolling_variance)
        
        # Cap the scaling factor to avoid extreme leverage
        scaling_factor = scaling_factor.clip(0.5, 2.0)
        
        return scaling_factor
    
    def generate_signals(self, top_decile_returns, bottom_decile_returns, daily_returns=None, lookback_period=12):
        """Generate trading signals based on predicted returns."""
        signals = pd.DataFrame(index=top_decile_returns.index[lookback_period:], 
                              columns=['top_position', 'bottom_position', 'scaling_factor'])
        
        for t in range(lookback_period, len(top_decile_returns)):
            date = top_decile_returns.index[t]
            
            # Input data for prediction
            X_top = top_decile_returns.iloc[t-lookback_period:t].values.reshape(1, -1)
            X_bottom = bottom_decile_returns.iloc[t-lookback_period:t].values.reshape(1, -1)
            
            # Get predictions
            if self.prediction_model:
                top_prediction = self.prediction_model.predict(X_top)[0]
                bottom_prediction = self.prediction_model.predict(X_bottom)[0]
            else:
                # If no model provided, use lookback period returns as prediction
                top_prediction = np.prod(1 + top_decile_returns.iloc[t-lookback_period:t].values) - 1
                bottom_prediction = np.prod(1 + bottom_decile_returns.iloc[t-lookback_period:t].values) - 1
            
            # Generate signals
            signals.loc[date, 'top_position'] = 1 if top_prediction > 0 else 0
            signals.loc[date, 'bottom_position'] = -1 if bottom_prediction < 0 else 0
            
            # Calculate volatility scaling factor if enabled
            if self.volatility_scaling and daily_returns is not None:
                # Get the last available daily data point before this monthly date
                last_daily_date = daily_returns.index[daily_returns.index < date][-1]
                signals.loc[date, 'scaling_factor'] = self.calculate_volatility_scaling(
                    daily_returns.loc[:last_daily_date]).iloc[-1]
            else:
                signals.loc[date, 'scaling_factor'] = 1.0
        
        return signals
    
    def backtest(self, top_decile_returns, bottom_decile_returns, signals, transaction_cost=0.002):
        """Backtest the strategy based on signals."""
        # Initialize portfolio returns series
        portfolio_returns = pd.Series(index=signals.index, dtype=float)
        
        # For each month in the holding period
        for i, date in enumerate(signals.index):
            # Skip the first date (no previous position)
            if i == 0:
                portfolio_returns[date] = 0
                continue
            
            prev_date = signals.index[i-1]
            
            # Get current month's returns
            top_return = top_decile_returns.loc[date]
            bottom_return = bottom_decile_returns.loc[date]
            
            # Get positions and scaling factor
            top_position = signals.loc[prev_date, 'top_position']
            bottom_position = signals.loc[prev_date, 'bottom_position']
            scaling_factor = signals.loc[prev_date, 'scaling_factor']
            
            # Calculate strategy return
            strategy_return = 0
            
            # Long position in top decile
            if top_position > 0:
                strategy_return += top_return * scaling_factor
            
            # Short position in bottom decile
            if bottom_position < 0:
                strategy_return += (-bottom_return) * scaling_factor
            
            # Apply transaction costs
            # Assuming positions can change every month
            if i > 1:
                prev_top_position = signals.loc[signals.index[i-2], 'top_position']
                prev_bottom_position = signals.loc[signals.index[i-2], 'bottom_position']
                
                # Calculate turnover and transaction costs
                if top_position != prev_top_position:
                    strategy_return -= transaction_cost
                
                if bottom_position != prev_bottom_position:
                    strategy_return -= transaction_cost
            
            portfolio_returns[date] = strategy_return
        
        self.returns = portfolio_returns
        return portfolio_returns
    
    def calculate_performance_metrics(self, returns):
        """Calculate performance metrics for the strategy."""
        # Converting monthly returns to annual
        ann_return = (1 + returns.mean()) ** 12 - 1
        ann_vol = returns.std() * np.sqrt(12)
        sharpe_ratio = ann_return / ann_vol if ann_vol > 0 else 0
        
        # Calculate drawdowns
        cum_returns = (1 + returns).cumprod()
        max_dd = (cum_returns / cum_returns.cummax() - 1).min()
        
        # Higher order moments
        kurtosis = returns.kurtosis()
        skewness = returns.skew()
        
        metrics = {
            'Annualized Return': ann_return,
            'Annualized Volatility': ann_vol,
            'Sharpe Ratio': sharpe_ratio,
            'Maximum Drawdown': max_dd,
            'Kurtosis': kurtosis,
            'Skewness': skewness
        }
        
        return metrics


def get_french_momentum_data(start_date=None, end_date=None):
    """
    Get Kenneth French's momentum data directly.
    This function downloads momentum decile portfolio data from Kenneth French's website.
    """
    print("Fetching Kenneth French's momentum data...")
    try:
        # Get Fama-French momentum portfolios data
        # Try different momentum-related datasets
        try:
            ff_mom = web.DataReader('10_Portfolios_Prior_12_2', 'famafrench')[0]
            print("Using 10_Portfolios_Prior_12_2 dataset")
        except:
            try:
                ff_mom = web.DataReader('10_Portfolios_ME_Prior_12_2', 'famafrench')[0]
                print("Using 10_Portfolios_ME_Prior_12_2 dataset")
            except:
                try:
                    ff_mom = web.DataReader('10_Portfolios_Prior_12_2_Daily', 'famafrench')[0]
                    print("Using 10_Portfolios_Prior_12_2_Daily dataset")
                except:
                    raise Exception("Could not find momentum portfolios")
        
        # Format data - the deciles are named from "Lo PRIOR" to "Hi PRIOR"
        if 'Hi PRIOR' in ff_mom.columns:
            top_decile_returns = ff_mom['Hi PRIOR'] / 100  # Convert percentage to decimal
            bottom_decile_returns = ff_mom['Lo PRIOR'] / 100
        else:
            # If column names are different, try to find the right ones
            columns = ff_mom.columns
            top_decile_returns = ff_mom[columns[-1]] / 100  # Last column should be highest momentum
            bottom_decile_returns = ff_mom[columns[0]] / 100  # First column should be lowest momentum
        
        # Convert index to datetime if it's not already
        if not isinstance(top_decile_returns.index, pd.DatetimeIndex):
            top_decile_returns.index = pd.to_datetime(top_decile_returns.index)
            bottom_decile_returns.index = pd.to_datetime(bottom_decile_returns.index)
        
        # Filter by date range if provided
        if start_date:
            start_date = pd.to_datetime(start_date)
            top_decile_returns = top_decile_returns[top_decile_returns.index >= start_date]
            bottom_decile_returns = bottom_decile_returns[bottom_decile_returns.index >= start_date]
        
        if end_date:
            end_date = pd.to_datetime(end_date)
            top_decile_returns = top_decile_returns[top_decile_returns.index <= end_date]
            bottom_decile_returns = bottom_decile_returns[bottom_decile_returns.index <= end_date]
        
        return top_decile_returns, bottom_decile_returns
        
    except Exception as e:
        print(f"Error getting French momentum data: {e}")
        return None, None

def get_french_daily_momentum_data():
    """Get daily momentum data from Kenneth French's library."""
    print("Fetching daily momentum data...")
    try:
        # Try to get daily momentum data
        ff_daily = web.DataReader('10_Portfolios_Prior_12_2_Daily', 'famafrench')[0]
        
        # Format data
        if 'Hi PRIOR' in ff_daily.columns:
            daily_top = ff_daily['Hi PRIOR'] / 100
            daily_bottom = ff_daily['Lo PRIOR'] / 100
        else:
            columns = ff_daily.columns
            daily_top = ff_daily[columns[-1]] / 100
            daily_bottom = ff_daily[columns[0]] / 100
        
        # Create WML returns
        daily_wml = daily_top - daily_bottom
        
        return daily_wml
    except:
        print("Could not get daily momentum data")
        return None

# Main execution
def main():
    # Set date range - try to get more historical data for better training
    start_date = '19930101'  # January 1, 1993
    end_date = '20230101'    # January 1, 2023
    
    # Try to get Fama-French Momentum data
    top_decile_returns, bottom_decile_returns = get_french_momentum_data(start_date, end_date)
    
    if top_decile_returns is None or len(top_decile_returns) < 100:
        print("Could not get sufficient French momentum data. Using synthetic data...")
        
        # Create synthetic data for demonstration
        dates = pd.date_range(start=start_date, end=end_date, freq='M')
        
        # Create momentum effect in synthetic data
        # Base returns (market-like)
        base_returns = np.random.normal(0.008, 0.045, size=len(dates))
        
        # Add momentum effect
        momentum_factor = np.zeros_like(base_returns)
        trend_length = 6  # Average trend length in months
        
        # Create trending periods
        for i in range(0, len(dates), trend_length):
            direction = 1 if np.random.rand() > 0.5 else -1
            momentum_factor[i:i+trend_length] = direction * np.linspace(0.002, 0.02, min(trend_length, len(dates)-i))
        
        # Winners tend to have higher returns
        top_decile_returns = pd.Series(
            base_returns + momentum_factor + 0.005,  # Higher mean
            index=dates
        )
        
        # Losers tend to have lower returns
        bottom_decile_returns = pd.Series(
            base_returns - momentum_factor - 0.003,  # Lower mean
            index=dates
        )
        
        # Add momentum effect (autocorrelation)
        for i in range(1, len(dates)):
            top_decile_returns.iloc[i] += 0.2 * top_decile_returns.iloc[i-1]
            bottom_decile_returns.iloc[i] += 0.2 * bottom_decile_returns.iloc[i-1]
    
    # Get daily returns for volatility scaling
    daily_returns = get_french_daily_momentum_data()
    
    if daily_returns is None or len(daily_returns) < 100:
        print("No daily momentum data available. Using approximated daily returns...")
        
        # Create a daily date range
        daily_dates = pd.date_range(start=top_decile_returns.index[0], 
                                  end=top_decile_returns.index[-1], freq='D')
        
        # Create daily returns with momentum characteristics
        daily_returns = pd.Series(index=daily_dates)
        
        # Fill with approximate daily returns
        monthly_dates = top_decile_returns.index
        for i in range(len(monthly_dates)-1):
            start_date = monthly_dates[i]
            end_date = monthly_dates[i+1]
            
            # Get all daily dates between this month and next month
            mask = (daily_dates >= start_date) & (daily_dates < end_date)
            days_in_month = mask.sum()
            
            if days_in_month > 0:
                # Calculate WML monthly return
                monthly_wml = (top_decile_returns.iloc[i] - bottom_decile_returns.iloc[i])
                
                # Add some noise to daily returns while preserving the monthly return
                daily_noise = np.random.normal(0, 0.008, days_in_month)
                daily_noise = daily_noise - daily_noise.mean()  # Make noise zero-mean
                
                # Calculate daily returns to match monthly return
                base_daily_return = (1 + monthly_wml) ** (1/days_in_month) - 1
                
                # Distribute the daily returns
                daily_wml_returns = base_daily_return + daily_noise
                
                daily_returns.loc[mask] = daily_wml_returns
    
    print(f"Data summary:")
    print(f"Monthly data: {len(top_decile_returns)} months from {top_decile_returns.index[0]} to {top_decile_returns.index[-1]}")
    if daily_returns is not None:
        print(f"Daily data: {len(daily_returns)} days from {daily_returns.index[0]} to {daily_returns.index[-1]}")
    
    # Split data for training and testing (last 3 years for testing)
    print("Splitting data for training and testing...")
    split_date = pd.to_datetime(end_date) - pd.DateOffset(years=3)
    
    train_top = top_decile_returns[top_decile_returns.index < split_date]
    train_bottom = bottom_decile_returns[bottom_decile_returns.index < split_date]
    test_top = top_decile_returns[top_decile_returns.index >= split_date]
    test_bottom = bottom_decile_returns[bottom_decile_returns.index >= split_date]
    
    # Make sure we have enough data
    if len(train_top) < lookback_period + 50:
        print("Not enough training data. Adjusting split date...")
        # Use 80% for training, 20% for testing
        train_size = int(0.8 * len(top_decile_returns))
        train_top = top_decile_returns.iloc[:train_size]
        train_bottom = bottom_decile_returns.iloc[:train_size]
        test_top = top_decile_returns.iloc[train_size:]
        test_bottom = bottom_decile_returns.iloc[train_size:]
    
    print(f"Training data: {len(train_top)} months")
    print(f"Testing data: {len(test_top)} months")
    
    # Prepare training data for the top decile
    X_train_top = []
    y_train_top = []
    
    for i in range(lookback_period, len(train_top) - holding_period):
        X_train_top.append(train_top.iloc[i-lookback_period:i].values)
        y_train_top.append(train_top.iloc[i+holding_period-1])
    
    X_train_top = np.array(X_train_top)
    y_train_top = np.array(y_train_top)
    
    # Prepare training data for the bottom decile
    X_train_bottom = []
    y_train_bottom = []
    
    for i in range(lookback_period, len(train_bottom) - holding_period):
        X_train_bottom.append(train_bottom.iloc[i-lookback_period:i].values)
        y_train_bottom.append(train_bottom.iloc[i+holding_period-1])
    
    X_train_bottom = np.array(X_train_bottom)
    y_train_bottom = np.array(y_train_bottom)
    
    # Create validation sets (last 15% of training data)
    val_size = max(1, int(0.15 * len(X_train_top)))
    
    X_val_top = X_train_top[-val_size:]
    y_val_top = y_train_top[-val_size:]
    X_train_top = X_train_top[:-val_size]
    y_train_top = y_train_top[:-val_size]
    
    X_val_bottom = X_train_bottom[-val_size:]
    y_val_bottom = y_train_bottom[-val_size:]
    X_train_bottom = X_train_bottom[:-val_size]
    y_train_bottom = y_train_bottom[:-val_size]
    
    # Train models for top and bottom deciles with adjusted parameters
    print("Training model for top decile...")
    top_model = DNN_SdAE(input_dim=lookback_period, hidden_layers=(8, 8, 8), noise_level=noise_level)
    top_model.build_and_train(X_train_top, y_train_top, X_val_top, y_val_top, 
                              epochs=n_epochs, batch_size=min(batch_size, len(X_train_top)), lr=learning_rate)
    
    print("Training model for bottom decile...")
    bottom_model = DNN_SdAE(input_dim=lookback_period, hidden_layers=(8, 8, 8), noise_level=noise_level)
    bottom_model.build_and_train(X_train_bottom, y_train_bottom, X_val_bottom, y_val_bottom, 
                                epochs=n_epochs, batch_size=min(batch_size, len(X_train_bottom)), lr=learning_rate)
    
    # Create full test dataset (including the portion we skipped in training due to lookback)
    if len(test_top) < lookback_period:
        # If test set is too small, include the end of training data for lookback
        extended_test_top = pd.concat([train_top.iloc[-lookback_period:], test_top])
        extended_test_bottom = pd.concat([train_bottom.iloc[-lookback_period:], test_bottom])
    else:
        extended_test_top = test_top
        extended_test_bottom = test_bottom
    
    # Initialize prediction models
    class TopDecilePredictionModel:
        def __init__(self, model):
            self.model = model
        
        def predict(self, X):
            return self.model.predict(X)
    
    class BottomDecilePredictionModel:
        def __init__(self, model):
            self.model = model
        
        def predict(self, X):
            return self.model.predict(X)
    
    top_prediction_model = TopDecilePredictionModel(top_model)
    bottom_prediction_model = BottomDecilePredictionModel(bottom_model)
    
    # Test period volatility
    if daily_returns is not None:
        test_daily_returns = daily_returns[daily_returns.index >= test_top.index[0]]
    else:
        test_daily_returns = None
    
    # Implement strategies
    # 1. Traditional WML strategy
    print("Backtesting traditional WML strategy...")
    wml_strategy = SelectiveMomentumStrategy(prediction_model=None, volatility_scaling=False)
    wml_signals = pd.DataFrame(index=test_top.index, columns=['top_position', 'bottom_position', 'scaling_factor'])
    wml_signals['top_position'] = 1
    wml_signals['bottom_position'] = -1
    wml_signals['scaling_factor'] = 1.0
    wml_returns = wml_strategy.backtest(test_top, test_bottom, wml_signals)
    wml_metrics = wml_strategy.calculate_performance_metrics(wml_returns)
    
    # 2. Winner Only (WO) strategy
    print("Backtesting Winner Only (WO) strategy...")
    wo_strategy = SelectiveMomentumStrategy(prediction_model=None, volatility_scaling=False)
    wo_signals = pd.DataFrame(index=test_top.index, columns=['top_position', 'bottom_position', 'scaling_factor'])
    wo_signals['top_position'] = 1
    wo_signals['bottom_position'] = 0  # No short position
    wo_signals['scaling_factor'] = 1.0
    wo_returns = wo_strategy.backtest(test_top, test_bottom, wo_signals)
    wo_metrics = wo_strategy.calculate_performance_metrics(wo_returns)
    
    # 3. Volatility-scaled WML strategy
    print("Backtesting volatility-scaled WML strategy...")
    vs_wml_strategy = SelectiveMomentumStrategy(prediction_model=None, volatility_scaling=True)
    vs_wml_signals = pd.DataFrame(index=test_top.index, columns=['top_position', 'bottom_position', 'scaling_factor'])
    vs_wml_signals['top_position'] = 1
    vs_wml_signals['bottom_position'] = -1
    
    # Calculate scaling factors
    if test_daily_returns is not None:
        for i, date in enumerate(test_top.index):
            # Find closest date in daily returns
            closest_dates = test_daily_returns.index[test_daily_returns.index <= date]
            if len(closest_dates) > 0:
                closest_date = closest_dates[-1]
                # Use past 6 months (126 trading days) for volatility calculation
                past_returns = test_daily_returns.loc[:closest_date].tail(126)
                if len(past_returns) > 0:
                    vol = past_returns.std() * np.sqrt(252)  # Annualize
                    vs_wml_signals.loc[date, 'scaling_factor'] = min(2.0, max(0.5, 0.12 / vol)) if vol > 0 else 1.0
                else:
                    vs_wml_signals.loc[date, 'scaling_factor'] = 1.0
            else:
                vs_wml_signals.loc[date, 'scaling_factor'] = 1.0
    else:
        vs_wml_signals['scaling_factor'] = 1.0
    
    vs_wml_returns = vs_wml_strategy.backtest(test_top, test_bottom, vs_wml_signals)
    vs_wml_metrics = vs_wml_strategy.calculate_performance_metrics(vs_wml_returns)
    
    # 4. Selective WML with SdAE (SWMLd in the paper)
    print("Backtesting Selective WML (SWMLd) strategy...")
    swml_signals = pd.DataFrame(index=test_top.index, columns=['top_position', 'bottom_position', 'scaling_factor'])
    
    # Generate signals based on model predictions
    for i, date in enumerate(test_top.index):
        # Check if we have enough historical data
        start_idx = None
        for j in range(len(extended_test_top)):
            if extended_test_top.index[j] == date:
                start_idx = j
                break
        
        if start_idx is None or start_idx < lookback_period:
            swml_signals.loc[date] = [0, 0, 1.0]
            continue
            
        # Get lookback period data
        X_top = extended_test_top.iloc[start_idx-lookback_period:start_idx].values.reshape(1, -1)
        X_bottom = extended_test_bottom.iloc[start_idx-lookback_period:start_idx].values.reshape(1, -1)
        
        # Get predictions
        top_prediction = top_prediction_model.predict(X_top)[0]
        bottom_prediction = bottom_prediction_model.predict(X_bottom)[0]
        
        # Set positions based on predictions
        top_position = 1 if top_prediction > 0 else 0
        bottom_position = -1 if bottom_prediction < 0 else 0
        
        swml_signals.loc[date] = [top_position, bottom_position, 1.0]
    
    swml_strategy = SelectiveMomentumStrategy(prediction_model=None, volatility_scaling=False)
    swml_returns = swml_strategy.backtest(test_top, test_bottom, swml_signals)
    swml_metrics = swml_strategy.calculate_performance_metrics(swml_returns)
    
    # 5. Volatility-scaled Selective WML (VS-SWMLd)
    print("Backtesting volatility-scaled Selective WML (VS-SWMLd) strategy...")
    vs_swml_strategy = SelectiveMomentumStrategy(prediction_model=None, volatility_scaling=True)
    vs_swml_signals = swml_signals.copy()
    
    # Apply same volatility scaling as in the VS-WML strategy
    vs_swml_signals['scaling_factor'] = vs_wml_signals['scaling_factor'].values
    
    vs_swml_returns = vs_swml_strategy.backtest(test_top, test_bottom, vs_swml_signals)
    vs_swml_metrics = vs_swml_strategy.calculate_performance_metrics(vs_swml_returns)
    
    # Print results
    print("\nWML Strategy Metrics:")
    for key, value in wml_metrics.items():
        print(f"{key}: {value:.4f}")
    
    print("\nWO Strategy Metrics:")
    for key, value in wo_metrics.items():
        print(f"{key}: {value:.4f}")
    
    print("\nVolatility-Scaled WML Strategy Metrics:")
    for key, value in vs_wml_metrics.items():
        print(f"{key}: {value:.4f}")
    
    print("\nSelective WML (SWMLd) Strategy Metrics:")
    for key, value in swml_metrics.items():
        print(f"{key}: {value:.4f}")
    
    print("\nVolatility-Scaled Selective WML (VS-SWMLd) Strategy Metrics:")
    for key, value in vs_swml_metrics.items():
        print(f"{key}: {value:.4f}")
    
    # Print strategy composition statistics
    print("\nStrategy Composition:")
    long_short = (swml_signals['top_position'] > 0) & (swml_signals['bottom_position'] < 0)
    long_only = (swml_signals['top_position'] > 0) & (swml_signals['bottom_position'] == 0)
    short_only = (swml_signals['top_position'] == 0) & (swml_signals['bottom_position'] < 0)
    no_position = (swml_signals['top_position'] == 0) & (swml_signals['bottom_position'] == 0)
    
    print(f"Long and Short: {long_short.mean():.4f}")
    print(f"Long Only: {long_only.mean():.4f}")
    print(f"Short Only: {short_only.mean():.4f}")
    print(f"No Position: {no_position.mean():.4f}")
    
    # Plot cumulative returns
    plt.figure(figsize=(12, 8))
    (1 + wml_returns).cumprod().plot(label='WML')
    (1 + wo_returns).cumprod().plot(label='WO')
    (1 + vs_wml_returns).cumprod().plot(label='VS-WML')
    (1 + swml_returns).cumprod().plot(label='SWMLd')
    (1 + vs_swml_returns).cumprod().plot(label='VS-SWMLd')
    plt.title('Cumulative Returns')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    plt.legend()
    plt.grid(True)
    plt.savefig('momentum_strategy_returns.png')
    plt.show()
    
    # Plot drawdowns
    plt.figure(figsize=(12, 8))
    def calculate_drawdown(returns):
        cum_returns = (1 + returns).cumprod()
        return cum_returns / cum_returns.cummax() - 1
    
    calculate_drawdown(wml_returns).plot(label='WML')
    calculate_drawdown(wo_returns).plot(label='WO')
    calculate_drawdown(vs_wml_returns).plot(label='VS-WML')
    calculate_drawdown(swml_returns).plot(label='SWMLd')
    calculate_drawdown(vs_swml_returns).plot(label='VS-SWMLd')
    plt.title('Drawdowns')
    plt.xlabel('Date')
    plt.ylabel('Drawdown')
    plt.legend()
    plt.grid(True)
    plt.savefig('momentum_strategy_drawdowns.png')
    plt.show()


if __name__ == "__main__":
    main()