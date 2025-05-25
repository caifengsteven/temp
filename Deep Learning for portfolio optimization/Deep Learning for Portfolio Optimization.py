import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import datetime as dt
import os
import time
from tqdm import tqdm
import logging
import blpapi
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("deep_learning_portfolio.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Configuration parameters
class Config:
    # Assets to trade
    assets = ['VTI US Equity', 'AGG US Equity', 'DBC US Equity', 'VIX INDEX']
    asset_names = ['Stock', 'Bond', 'Commodity', 'Volatility']
    
    # Features
    lookback_window = 50  # Days of history to use as input
    features = ['PX_LAST', 'RETURN']  # Price and return features
    
    # Training parameters
    batch_size = 64
    learning_rate = 0.001
    epochs = 100
    validation_split = 0.1
    
    # Model parameters
    lstm_units = 64
    dropout_rate = 0.0  # No dropout for single layer
    
    # Portfolio parameters
    transaction_cost = 0.0001  # 1 bps
    target_volatility = 0.10  # 10% annual volatility
    volatility_lookback = 50  # For volatility estimation
    rebalance_frequency = 1  # Daily
    
    # Periods
    training_start = dt.datetime(2006, 1, 1)
    testing_start = dt.datetime(2011, 1, 1)
    testing_end = dt.datetime(2023, 4, 30)
    
    # File paths
    data_dir = 'data'
    model_dir = 'models'
    results_dir = 'results'
    
    def __init__(self):
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        
config = Config()

class BloombergDataFetcher:
    """Class to fetch data from Bloomberg terminal."""
    
    def __init__(self):
        """Initialize Bloomberg connection."""
        self.session = None
        self.refdata_service = None
        
    def start_session(self):
        """Start Bloomberg session."""
        logger.info("Starting Bloomberg session...")
        sessionOptions = blpapi.SessionOptions()
        sessionOptions.setServerHost('localhost')
        sessionOptions.setServerPort(8194)
        
        self.session = blpapi.Session(sessionOptions)
        if not self.session.start():
            raise RuntimeError("Failed to start session.")
        
        if not self.session.openService("//blp/refdata"):
            raise RuntimeError("Failed to open //blp/refdata")
        
        self.refdata_service = self.session.getService("//blp/refdata")
        logger.info("Bloomberg session started successfully.")
    
    def stop_session(self):
        """Stop Bloomberg session."""
        if self.session:
            self.session.stop()
            logger.info("Bloomberg session stopped.")
    
    def fetch_historical_data(self, tickers, fields, start_date, end_date, 
                             period='DAILY'):
        """
        Fetch historical data from Bloomberg.
        
        Parameters:
        -----------
        tickers : list
            List of Bloomberg tickers
        fields : list
            List of Bloomberg fields (e.g., 'PX_LAST')
        start_date : datetime
            Start date
        end_date : datetime
            End date
        period : str
            Periodicity ('DAILY', 'WEEKLY', 'MONTHLY')
            
        Returns:
        --------
        DataFrame
            DataFrame with historical data
        """
        # Convert dates to string format for Bloomberg
        start_date_str = start_date.strftime('%Y%m%d')
        end_date_str = end_date.strftime('%Y%m%d')
        
        request = self.refdata_service.createRequest("HistoricalDataRequest")
        
        # Set request parameters
        for ticker in tickers:
            request.append("securities", ticker)
        
        for field in fields:
            request.append("fields", field)
        
        request.set("startDate", start_date_str)
        request.set("endDate", end_date_str)
        request.set("periodicitySelection", period)
        
        logger.info(f"Sending request for {len(tickers)} securities...")
        
        # Send request
        self.session.sendRequest(request)
        
        # Process response
        data = []
        
        while True:
            event = self.session.nextEvent(500)
            for msg in event:
                if msg.messageType() == "HistoricalDataResponse":
                    security_data = msg.getElement("securityData")
                    security_name = security_data.getElementAsString("security")
                    field_data = security_data.getElement("fieldData")
                    
                    for i in range(field_data.numValues()):
                        field_value = field_data.getValue(i)
                        date = field_value.getElementAsDatetime("date").strftime('%Y-%m-%d')
                        
                        row_data = {'date': date, 'security': security_name}
                        
                        for field in fields:
                            if field_value.hasElement(field):
                                row_data[field] = field_value.getElementAsFloat(field)
                            else:
                                row_data[field] = np.nan
                        
                        data.append(row_data)
            
            if event.eventType() == blpapi.Event.RESPONSE:
                break
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        if df.empty:
            logger.warning("No data received from Bloomberg.")
            return None
        
        # Convert date to datetime
        df['date'] = pd.to_datetime(df['date'])
        
        # Pivot to have tickers as columns
        pivot_df = pd.pivot_table(
            df, values=fields[0], index='date', columns='security'
        )
        
        # Ensure index is datetime
        pivot_df.index = pd.to_datetime(pivot_df.index)
        
        # Sort by date
        pivot_df.sort_index(inplace=True)
        
        logger.info(f"Received data for {pivot_df.shape[1]} securities from {pivot_df.index[0]} to {pivot_df.index[-1]}.")
        
        # Calculate returns
        returns_df = pivot_df.pct_change().dropna()
        
        # Create multi-index DataFrame
        multi_index_data = pd.concat({
            'PX_LAST': pivot_df,
            'RETURN': returns_df
        }, axis=1)
        
        return multi_index_data

# Data handling
class DataHandler:
    def __init__(self, config):
        self.config = config
        self.data = None
        self.scalers = {}
        self.bloomberg = BloombergDataFetcher()
        
    def load_data(self):
        """Load data from file or fetch from Bloomberg"""
        logger.info("Loading data...")
        
        # Check for existing files
        bloomberg_path = os.path.join(self.config.data_dir, 'bloomberg_data.pkl')
        
        # Try to load existing Bloomberg data
        if os.path.exists(bloomberg_path):
            try:
                logger.info(f"Loading existing Bloomberg data from {bloomberg_path}")
                self.data = pd.read_pickle(bloomberg_path)
                
                # Check if data is valid
                if self.data is not None and not self.data.empty:
                    logger.info("Successfully loaded Bloomberg data")
                    self._clean_data()
                    return self.data
                else:
                    logger.warning("Loaded Bloomberg data is empty.")
            except Exception as e:
                logger.error(f"Error loading Bloomberg data: {e}")
        
        # If no existing data, try fetching from Bloomberg
        logger.info("No valid Bloomberg data found, trying to fetch...")
        
        # Fetch Bloomberg data
        try:
            self.bloomberg.start_session()
            self.data = self.bloomberg.fetch_historical_data(
                tickers=self.config.assets,
                fields=['PX_LAST'],
                start_date=self.config.training_start,
                end_date=self.config.testing_end,
                period='DAILY'
            )
            self.bloomberg.stop_session()
            
            # Save the data
            if self.data is not None and not self.data.empty:
                data_path = os.path.join(self.config.data_dir, 'bloomberg_data.pkl')
                self.data.to_pickle(data_path)
                logger.info(f"Bloomberg data saved to {data_path}")
                self._clean_data()
                return self.data
        except Exception as e:
            logger.error(f"Error fetching Bloomberg data: {e}")
            logger.debug(traceback.format_exc())
            self.bloomberg.stop_session()
        
        # If we get here, we need to generate synthetic data
        logger.info("Falling back to synthetic data...")
        self.data = self.generate_synthetic_data()
        self._clean_data()
        return self.data
    
    def generate_synthetic_data(self):
        """Generate realistic synthetic data"""
        logger.info("Generating synthetic data...")
        
        # Generate date range
        date_range = pd.date_range(
            self.config.training_start, 
            self.config.testing_end, 
            freq='B'  # Business days
        )
        
        # Initialize price data with random walk
        prices = {}
        returns = {}
        
        # Define realistic parameters for each asset class
        asset_params = {
            'VTI US Equity': {  # Stock ETF
                'initial_price': 100.0,
                'annual_return': 0.09,  # 9% annual return
                'annual_vol': 0.16,     # 16% annual volatility
            },
            'AGG US Equity': {  # Bond ETF
                'initial_price': 100.0,
                'annual_return': 0.035,  # 3.5% annual return
                'annual_vol': 0.05,      # 5% annual volatility
            },
            'DBC US Equity': {  # Commodity ETF
                'initial_price': 100.0,
                'annual_return': 0.04,   # 4% annual return
                'annual_vol': 0.20,      # 20% annual volatility
            },
            'VIX INDEX': {  # Volatility Index
                'initial_price': 20.0,   # VIX typically starts around 15-20
                'annual_return': 0.0,    # No expected return (mean-reverting)
                'annual_vol': 0.85,      # Very high volatility (85%)
            }
        }
        
        # Create correlation matrix
        correlation_matrix = np.array([
            [1.0, -0.2, 0.4, -0.7],  # Stock correlations
            [-0.2, 1.0, 0.1, 0.3],  # Bond correlations
            [0.4, 0.1, 1.0, 0.2],  # Commodity correlations
            [-0.7, 0.3, 0.2, 1.0]   # VIX correlations
        ])
        
        # Extract volatilities
        vols = np.array([asset_params[asset]['annual_vol'] / np.sqrt(252) for asset in self.config.assets])
        
        # Calculate covariance matrix
        cov_matrix = np.diag(vols) @ correlation_matrix @ np.diag(vols)
        
        # Generate correlated returns
        num_days = len(date_range)
        daily_means = np.array([asset_params[asset]['annual_return'] / 252 for asset in self.config.assets])
        correlated_returns = np.random.multivariate_normal(
            mean=daily_means, 
            cov=cov_matrix, 
            size=num_days
        )
        
        # Generate price series
        prices_array = np.zeros((num_days, len(self.config.assets)))
        
        # Set initial prices
        for i, asset in enumerate(self.config.assets):
            prices_array[0, i] = asset_params[asset]['initial_price']
        
        # Calculate cumulative price series
        for t in range(1, num_days):
            prices_array[t] = prices_array[t-1] * (1 + correlated_returns[t])
        
        # Convert to DataFrames
        prices_df = pd.DataFrame(
            prices_array, 
            index=date_range,
            columns=self.config.assets
        )
        
        returns_df = pd.DataFrame(
            correlated_returns,
            index=date_range,
            columns=self.config.assets
        )
        
        # Combine into a multi-index DataFrame
        multi_index_data = pd.concat({
            'PX_LAST': prices_df,
            'RETURN': returns_df
        }, axis=1)
        
        # Save synthetic data
        data_path = os.path.join(self.config.data_dir, 'synthetic_data.pkl')
        multi_index_data.to_pickle(data_path)
        logger.info(f"Synthetic data saved to {data_path}")
        
        return multi_index_data
    
    def _clean_data(self):
        """Clean and prepare data"""
        if self.data is None:
            raise ValueError("No data available. Load or generate data first.")
        
        # Check if data is empty
        if self.data.empty:
            raise ValueError("Data is empty after loading.")
        
        # Forward fill missing values
        self.data = self.data.fillna(method='ffill')
        
        # Remove any remaining NaN values
        self.data = self.data.dropna()
        
        logger.info(f"Data shape after cleaning: {self.data.shape}")
        
        # Print some sample data to verify
        logger.info("\nSample data (first 3 rows):")
        logger.info(self.data.head(3))
    
    def prepare_training_data(self, start_date=None, end_date=None):
        """Prepare data for training"""
        if self.data is None:
            raise ValueError("No data available. Load or generate data first.")
        
        # Filter data by date
        if start_date is not None and end_date is not None:
            filtered_data = self.data.loc[start_date:end_date]
        else:
            filtered_data = self.data
        
        # Initialize list to hold all sequences
        X_sequences = []
        dates = []
        
        # Loop through each time step
        for i in range(self.config.lookback_window, len(filtered_data)):
            # Extract the lookback window
            sequence = filtered_data.iloc[i-self.config.lookback_window:i]
            
            # Flatten the sequence for each asset
            flattened_features = []
            for asset in self.config.assets:
                asset_features = []
                for feature in self.config.features:
                    if (feature, asset) in sequence.columns:
                        feature_values = sequence[(feature, asset)].values
                        asset_features.append(feature_values)
                    else:
                        # If feature is missing, use zeros
                        logger.warning(f"Missing feature {feature} for asset {asset}")
                        feature_values = np.zeros(self.config.lookback_window)
                        asset_features.append(feature_values)
                flattened_features.extend(asset_features)
            
            # Convert to numpy array
            flattened_features = np.array(flattened_features).T
            
            X_sequences.append(flattened_features)
            dates.append(filtered_data.index[i])
        
        # Convert to numpy arrays
        X = np.array(X_sequences)
        
        logger.info(f"Prepared training data shape: {X.shape}")
        logger.info(f"Number of sequences: {len(dates)}")
        
        return X, dates
    
    def standardize_data(self, X, fit=True):
        """Standardize features"""
        if fit:
            self.scalers = {}
            for i in range(X.shape[2]):  # For each feature
                self.scalers[i] = StandardScaler()
                feature_data = X[:, :, i].reshape(-1, 1)
                self.scalers[i].fit(feature_data)
        
        # Apply scaling
        X_scaled = np.zeros_like(X)
        for i in range(X.shape[2]):
            feature_data = X[:, :, i].reshape(-1, 1)
            X_scaled[:, :, i] = self.scalers[i].transform(feature_data).reshape(X.shape[0], X.shape[1])
        
        return X_scaled

# LSTM model for portfolio optimization
class LSTMPortfolioOptimizer(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_assets, dropout_rate=0.0):
        super(LSTMPortfolioOptimizer, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_assets = num_assets
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            dropout=dropout_rate
        )
        
        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, num_assets)
        
        # Softmax activation
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        # LSTM layer
        lstm_out, _ = self.lstm(x)
        
        # Use the output from the last time step
        last_time_step = lstm_out[:, -1, :]
        
        # Fully connected layer
        fc_out = self.fc(last_time_step)
        
        # Apply softmax for portfolio weights
        weights = self.softmax(fc_out)
        
        return weights

# Custom dataset for portfolio optimization
class PortfolioDataset(Dataset):
    def __init__(self, X, dates, returns):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.dates = [d.strftime('%Y-%m-%d') if hasattr(d, 'strftime') else str(d) for d in dates]  # Convert timestamps to strings
        self.returns = torch.tensor(returns, dtype=torch.float32)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return {
            'X': self.X[idx],
            'date_idx': idx,  # Just use the index instead of the date
            'returns': self.returns[idx]
        }

# Sharpe ratio loss function
class SharpeLoss(nn.Module):
    def __init__(self, risk_free_rate=0.0):
        super(SharpeLoss, self).__init__()
        self.risk_free_rate = risk_free_rate
    
    def forward(self, weights, returns, transaction_cost=0.0001, prev_weights=None):
        """
        Calculate the negative Sharpe ratio as loss
        
        Args:
            weights: Portfolio weights for each asset (batch_size, num_assets)
            returns: Asset returns for the next period (batch_size, num_assets)
            transaction_cost: Transaction cost rate
            prev_weights: Previous portfolio weights for calculating transaction costs
            
        Returns:
            Negative Sharpe ratio (to be minimized)
        """
        # Calculate portfolio returns
        portfolio_returns = torch.sum(weights * returns, dim=1)
        
        # Subtract transaction costs if previous weights are available
        if prev_weights is not None:
            # Calculate position changes
            position_changes = torch.abs(weights - prev_weights)
            
            # Calculate transaction costs
            costs = transaction_cost * torch.sum(position_changes, dim=1)
            
            # Adjust portfolio returns
            portfolio_returns = portfolio_returns - costs
        
        # Calculate mean return
        mean_return = torch.mean(portfolio_returns)
        
        # Calculate standard deviation of returns
        std_return = torch.std(portfolio_returns, unbiased=False)
        
        # Avoid division by zero
        epsilon = 1e-8
        
        # Calculate Sharpe ratio (using excess return over risk-free rate)
        sharpe_ratio = (mean_return - self.risk_free_rate) / (std_return + epsilon)
        
        # Return negative Sharpe ratio (to minimize)
        return -sharpe_ratio

# Portfolio Manager for backtesting
class PortfolioManager:
    def __init__(self, config):
        self.config = config
        self.model = None
        self.data_handler = DataHandler(config)
        self.data = None
        self.weights_history = []
        self.portfolio_value_history = []
        self.returns_history = []
        self.volatility_history = []
        self.transaction_cost_history = []
    
    def train_model(self, start_date=None, end_date=None, retrain=False):
        """Train the LSTM model on historical data"""
        # Load data if not already loaded
        if self.data is None:
            self.data = self.data_handler.load_data()
        
        # Prepare training data
        X, dates = self.data_handler.prepare_training_data(start_date, end_date)
        
        # Standardize features
        X_scaled = self.data_handler.standardize_data(X)
        
        # Prepare asset returns for the next day
        returns = np.zeros((len(dates), len(self.config.assets)))
        for i, date in enumerate(dates):
            next_date_idx = self.data.index.get_loc(date) + 1
            if next_date_idx < len(self.data):
                for j, asset in enumerate(self.config.assets):
                    if ('RETURN', asset) in self.data.columns:
                        returns[i, j] = self.data.loc[self.data.index[next_date_idx], ('RETURN', asset)]
                    else:
                        logger.warning(f"Missing return data for {asset} at {date}")
                        returns[i, j] = 0.0
        
        # Split data into training and validation sets
        n_train = int(len(X_scaled) * (1 - self.config.validation_split))
        X_train, X_val = X_scaled[:n_train], X_scaled[n_train:]
        dates_train, dates_val = dates[:n_train], dates[n_train:]
        returns_train, returns_val = returns[:n_train], returns[n_train:]
        
        # Create datasets
        train_dataset = PortfolioDataset(X_train, dates_train, returns_train)
        val_dataset = PortfolioDataset(X_val, dates_val, returns_val)
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=min(self.config.batch_size, len(train_dataset)),
            shuffle=True
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=min(self.config.batch_size, len(val_dataset)),
            shuffle=False
        )
        
        # Initialize the model
        input_dim = X_scaled.shape[2]
        self.model = LSTMPortfolioOptimizer(
            input_dim=input_dim,
            hidden_dim=self.config.lstm_units,
            num_assets=len(self.config.assets),
            dropout_rate=self.config.dropout_rate
        ).to(device)
        
        # Initialize optimizer
        optimizer = optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        
        # Initialize loss function
        criterion = SharpeLoss()
        
        # Training loop
        best_val_sharpe = -float('inf')
        patience = 10  # For early stopping
        patience_counter = 0
        
        for epoch in range(self.config.epochs):
            # Training phase
            self.model.train()
            train_losses = []
            
            for batch in train_loader:
                X_batch = batch['X'].to(device)
                returns_batch = batch['returns'].to(device)
                
                # Forward pass
                weights = self.model(X_batch)
                
                # Calculate loss
                loss = criterion(weights, returns_batch)
                
                # Backward pass and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                train_losses.append(loss.item())
            
            # Validation phase
            self.model.eval()
            val_losses = []
            
            with torch.no_grad():
                for batch in val_loader:
                    X_batch = batch['X'].to(device)
                    returns_batch = batch['returns'].to(device)
                    
                    # Forward pass
                    weights = self.model(X_batch)
                    
                    # Calculate loss
                    loss = criterion(weights, returns_batch)
                    
                    val_losses.append(loss.item())
            
            # Calculate average losses
            avg_train_loss = np.mean(train_losses)
            avg_val_loss = np.mean(val_losses)
            
            # Print progress
            logger.info(f"Epoch {epoch+1}/{self.config.epochs}, "
                  f"Train Sharpe: {-avg_train_loss:.4f}, "
                  f"Val Sharpe: {-avg_val_loss:.4f}")
            
            # Check for improvement
            if -avg_val_loss > best_val_sharpe:
                best_val_sharpe = -avg_val_loss
                patience_counter = 0
                
                # Save the best model
                model_path = os.path.join(self.config.model_dir, 'best_model.pt')
                torch.save(self.model.state_dict(), model_path)
                logger.info(f"Model saved to {model_path}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info("Early stopping triggered")
                    break
        
        # Load the best model
        model_path = os.path.join(self.config.model_dir, 'best_model.pt')
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        
        return self.model
    
    def backtest(self, start_date=None, end_date=None):
        """Backtest the model"""
        # Load data if not already loaded
        if self.data is None:
            self.data = self.data_handler.load_data()
        
        # Check if model exists
        if self.model is None:
            model_path = os.path.join(self.config.model_dir, 'best_model.pt')
            if os.path.exists(model_path):
                # Initialize the model
                X, _ = self.data_handler.prepare_training_data()
                input_dim = X.shape[2]
                self.model = LSTMPortfolioOptimizer(
                    input_dim=input_dim,
                    hidden_dim=self.config.lstm_units,
                    num_assets=len(self.config.assets),
                    dropout_rate=self.config.dropout_rate
                ).to(device)
                self.model.load_state_dict(torch.load(model_path))
                self.model.eval()
            else:
                logger.error("No trained model found. Training a new one.")
                self.train_model()
        
        # Filter data by date
        if start_date is None:
            start_date = self.config.testing_start
        if end_date is None:
            end_date = self.config.testing_end
        
        # Get test date range
        test_dates = self.data.loc[start_date:end_date].index
        
        # Initialize portfolio with $1
        portfolio_value = 1.0
        current_weights = np.zeros(len(self.config.assets))
        
        # Initialize history
        self.weights_history = []
        self.portfolio_value_history = [portfolio_value]
        self.returns_history = []
        self.transaction_cost_history = []
        self.volatility_history = []
        
        # Loop through dates for backtesting
        for i, current_date in enumerate(tqdm(test_dates)):
            # Skip if not enough history
            date_idx = self.data.index.get_loc(current_date)
            if date_idx < self.config.lookback_window:
                continue
            
            # Check if we need to rebalance (based on frequency)
            if i % self.config.rebalance_frequency != 0 and i > 0:
                # Get next day's returns
                if date_idx + 1 < len(self.data.index):
                    next_day = self.data.index[date_idx + 1]
                    next_returns = np.zeros(len(self.config.assets))
                    for j, asset in enumerate(self.config.assets):
                        if ('RETURN', asset) in self.data.columns:
                            next_returns[j] = self.data.loc[next_day, ('RETURN', asset)]
                    
                    # Calculate portfolio return
                    portfolio_return = np.sum(current_weights * next_returns)
                    
                    # Update portfolio value
                    portfolio_value *= (1 + portfolio_return)
                    
                    # Store history
                    self.weights_history.append(current_weights.copy())
                    self.portfolio_value_history.append(portfolio_value)
                    self.returns_history.append(portfolio_return)
                    self.transaction_cost_history.append(0)  # No rebalancing
                
                continue
            
            # Prepare input data
            lookback_data = self.data.iloc[date_idx - self.config.lookback_window + 1:date_idx + 1]
            
            # Extract features
            X = []
            for asset in self.config.assets:
                asset_features = []
                for feature in self.config.features:
                    if (feature, asset) in lookback_data.columns:
                        feature_values = lookback_data[(feature, asset)].values
                        asset_features.append(feature_values)
                    else:
                        # If feature is missing, use zeros
                        logger.warning(f"Missing feature {feature} for asset {asset} during backtesting")
                        feature_values = np.zeros(self.config.lookback_window)
                        asset_features.append(feature_values)
                X.extend(asset_features)
            
            # Reshape and standardize
            X = np.array(X).T.reshape(1, self.config.lookback_window, -1)
            
            # Standardize using pre-fitted scalers
            X_scaled = np.zeros_like(X)
            for j in range(X.shape[2]):
                feature_data = X[:, :, j].reshape(-1, 1)
                if j in self.data_handler.scalers:
                    X_scaled[:, :, j] = self.data_handler.scalers[j].transform(feature_data).reshape(X.shape[0], X.shape[1])
                else:
                    X_scaled[:, :, j] = feature_data.reshape(X.shape[0], X.shape[1])
            
            # Convert to tensor
            X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)
            
            # Get portfolio weights
            with torch.no_grad():
                weights = self.model(X_tensor).cpu().numpy()[0]
            
            # Calculate volatility for scaling
            volatility_lookback = min(self.config.volatility_lookback, date_idx)
            vol_data = self.data.iloc[date_idx - volatility_lookback + 1:date_idx + 1]
            asset_vols = np.zeros(len(self.config.assets))
            
            for j, asset in enumerate(self.config.assets):
                if ('RETURN', asset) in vol_data.columns:
                    asset_returns = vol_data[('RETURN', asset)].values
                    asset_vols[j] = np.std(asset_returns) * np.sqrt(252)  # Annualize
                else:
                    # If asset is missing, use average volatility
                    logger.warning(f"Missing return data for {asset} during volatility calculation")
                    asset_vols[j] = 0.16  # Default volatility
            
            # Calculate volatility scaling factor
            portfolio_vol = np.sqrt(weights.T @ np.diag(asset_vols**2) @ weights)
            scaling_factor = self.config.target_volatility / portfolio_vol if portfolio_vol > 0 else 1.0
            
            # Scale weights
            scaled_weights = weights * scaling_factor
            
            # Calculate transaction costs
            transaction_cost = self.config.transaction_cost * np.sum(np.abs(scaled_weights - current_weights))
            
            # Update current weights
            current_weights = scaled_weights.copy()
            
            # Get next day's returns
            if date_idx + 1 < len(self.data.index):
                next_day = self.data.index[date_idx + 1]
                next_returns = np.zeros(len(self.config.assets))
                for j, asset in enumerate(self.config.assets):
                    if ('RETURN', asset) in self.data.columns:
                        next_returns[j] = self.data.loc[next_day, ('RETURN', asset)]
                
                # Calculate portfolio return
                portfolio_return = np.sum(current_weights * next_returns) - transaction_cost
                
                # Update portfolio value
                portfolio_value *= (1 + portfolio_return)
                
                # Store history
                self.weights_history.append(current_weights.copy())
                self.portfolio_value_history.append(portfolio_value)
                self.returns_history.append(portfolio_return)
                self.transaction_cost_history.append(transaction_cost)
                self.volatility_history.append(portfolio_vol)
        
        # Convert history to pandas DataFrames for analysis
        self.weights_df = pd.DataFrame(
            self.weights_history, 
            index=test_dates[len(test_dates)-len(self.weights_history):],
            columns=self.config.asset_names
        )
        
        self.portfolio_value_df = pd.Series(
            self.portfolio_value_history, 
            index=[test_dates[0]] + list(test_dates[len(test_dates)-len(self.weights_history):])
        )
        
        self.returns_df = pd.Series(
            self.returns_history, 
            index=test_dates[len(test_dates)-len(self.weights_history):]
        )
        
        self.transaction_cost_df = pd.Series(
            self.transaction_cost_history, 
            index=test_dates[len(test_dates)-len(self.weights_history):]
        )
        
        self.volatility_df = pd.Series(
            self.volatility_history, 
            index=test_dates[len(test_dates)-len(self.weights_history):]
        )
        
        return self.portfolio_value_df
    
    def calculate_performance_metrics(self):
        """Calculate performance metrics for the backtest"""
        if not hasattr(self, 'returns_df') or self.returns_df is None or len(self.returns_df) == 0:
            raise ValueError("No backtest results available. Run backtest first.")
        
        # Calculate annualized return
        total_return = self.portfolio_value_df.iloc[-1] / self.portfolio_value_df.iloc[0] - 1
        years = (self.portfolio_value_df.index[-1] - self.portfolio_value_df.index[0]).days / 365.25
        annual_return = (1 + total_return) ** (1 / years) - 1
        
        # Calculate annualized volatility
        daily_vol = np.std(self.returns_df)
        annual_vol = daily_vol * np.sqrt(252)
        
        # Calculate Sharpe ratio
        sharpe_ratio = annual_return / annual_vol
        
        # Calculate maximum drawdown
        cumulative_returns = (1 + self.returns_df).cumprod()
        running_max = cumulative_returns.cummax()
        drawdown = (cumulative_returns / running_max - 1)
        max_drawdown = drawdown.min()
        
        # Calculate downside deviation
        negative_returns = self.returns_df[self.returns_df < 0]
        downside_deviation = np.std(negative_returns) * np.sqrt(252)
        
        # Calculate Sortino ratio
        sortino_ratio = annual_return / downside_deviation if downside_deviation > 0 else float('inf')
        
        # Calculate percentage of positive returns
        positive_days = np.sum(self.returns_df > 0)
        total_days = len(self.returns_df)
        pct_positive = positive_days / total_days
        
        # Calculate average profit/loss ratio
        avg_gain = np.mean(self.returns_df[self.returns_df > 0]) if len(self.returns_df[self.returns_df > 0]) > 0 else 0
        avg_loss = np.abs(np.mean(self.returns_df[self.returns_df < 0])) if len(self.returns_df[self.returns_df < 0]) > 0 else float('inf')
        profit_loss_ratio = avg_gain / avg_loss if avg_loss > 0 else float('inf')
        
        # Calculate average transaction cost
        avg_transaction_cost = np.mean(self.transaction_cost_df)
        
        # Calculate turnover
        turnover = np.sum(self.transaction_cost_df) / self.config.transaction_cost
        
        # Results dictionary
        metrics = {
            'Annual Return': annual_return,
            'Annual Volatility': annual_vol,
            'Sharpe Ratio': sharpe_ratio,
            'Sortino Ratio': sortino_ratio,
            'Maximum Drawdown': max_drawdown,
            'Downside Deviation': downside_deviation,
            'Percentage of Positive Returns': pct_positive,
            'Profit/Loss Ratio': profit_loss_ratio,
            'Average Transaction Cost': avg_transaction_cost,
            'Turnover': turnover,
            'Total Return': total_return
        }
        
        return metrics
    
    def plot_results(self, save_path=None):
        """Plot backtest results"""
        if not hasattr(self, 'portfolio_value_df') or self.portfolio_value_df is None or len(self.portfolio_value_df) == 0:
            raise ValueError("No backtest results available. Run backtest first.")
        
        # Create figure with subplots
        fig, axs = plt.subplots(3, 1, figsize=(15, 18), gridspec_kw={'height_ratios': [2, 1, 1]})
        
        # Plot portfolio value
        axs[0].plot(self.portfolio_value_df.index, self.portfolio_value_df.values, label='Portfolio Value')
        axs[0].set_title('Portfolio Value Over Time')
        axs[0].set_ylabel('Value ($)')
        axs[0].set_xlabel('Date')
        axs[0].grid(True)
        axs[0].legend()
        
        # Plot asset weights
        self.weights_df.plot(ax=axs[1], kind='area', stacked=True)
        axs[1].set_title('Asset Allocation Over Time')
        axs[1].set_ylabel('Weight')
        axs[1].set_xlabel('Date')
        axs[1].grid(True)
        
        # Plot volatility and transaction costs
        ax2 = axs[2]
        ax2.plot(self.volatility_df.index, self.volatility_df.values, label='Portfolio Volatility', color='blue')
        ax2.set_ylabel('Annualized Volatility', color='blue')
        ax2.tick_params(axis='y', labelcolor='blue')
        ax2.set_xlabel('Date')
        ax2.grid(True)
        
        ax3 = ax2.twinx()
        ax3.plot(self.transaction_cost_df.index, self.transaction_cost_df.values * 10000, label='Transaction Costs (bps)', color='red')
        ax3.set_ylabel('Transaction Costs (bps)', color='red')
        ax3.tick_params(axis='y', labelcolor='red')
        
        # Add legends
        lines1, labels1 = ax2.get_legend_handles_labels()
        lines2, labels2 = ax3.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Results saved to {save_path}")
        
        plt.show()
    
    def run_reallocation_strategy(self, allocation_weights, rebalance_frequency='Y'):
        """Run a simple reallocation strategy for comparison"""
        # Load data if not already loaded
        if self.data is None:
            self.data = self.data_handler.load_data()
        
        # Get test date range
        test_dates = self.data.loc[self.config.testing_start:self.config.testing_end].index
        
        # Convert rebalance frequency to pandas offset
        if rebalance_frequency == 'Y':
            rebalance_offset = pd.DateOffset(years=1)
        elif rebalance_frequency == 'Q':
            rebalance_offset = pd.DateOffset(months=3)
        elif rebalance_frequency == 'M':
            rebalance_offset = pd.DateOffset(months=1)
        else:
            rebalance_offset = pd.DateOffset(days=1)
        
        # Initialize portfolio with $1
        portfolio_value = 1.0
        current_weights = np.array(list(allocation_weights.values()))
        
        # Initialize history
        weights_history = []
        portfolio_value_history = [portfolio_value]
        returns_history = []
        transaction_cost_history = []
        
        # Calculate the next rebalance date
        next_rebalance_date = test_dates[0] + rebalance_offset
        
        # Loop through dates for backtesting
        for i, current_date in enumerate(tqdm(test_dates)):
            # Skip if not enough history
            date_idx = self.data.index.get_loc(current_date)
            if date_idx < self.config.lookback_window:
                continue
            
            # Check if we need to rebalance
            if current_date >= next_rebalance_date or i == 0:
                # Calculate volatility for scaling
                volatility_lookback = min(self.config.volatility_lookback, date_idx)
                vol_data = self.data.iloc[date_idx - volatility_lookback + 1:date_idx + 1]
                asset_vols = np.zeros(len(self.config.assets))
                
                for j, asset in enumerate(self.config.assets):
                    if ('RETURN', asset) in vol_data.columns:
                        asset_returns = vol_data[('RETURN', asset)].values
                        asset_vols[j] = np.std(asset_returns) * np.sqrt(252)  # Annualize
                    else:
                        # If asset is missing, use average volatility
                        logger.warning(f"Missing return data for {asset} during volatility calculation")
                        asset_vols[j] = 0.16  # Default volatility
                
                # Rebalance to target weights
                target_weights = np.array(list(allocation_weights.values()))
                
                # Calculate volatility scaling factor
                portfolio_vol = np.sqrt(target_weights.T @ np.diag(asset_vols**2) @ target_weights)
                scaling_factor = self.config.target_volatility / portfolio_vol if portfolio_vol > 0 else 1.0
                
                # Scale weights
                scaled_weights = target_weights * scaling_factor
                
                # Calculate transaction costs
                transaction_cost = self.config.transaction_cost * np.sum(np.abs(scaled_weights - current_weights))
                
                # Update current weights
                current_weights = scaled_weights.copy()
                
                # Update next rebalance date
                next_rebalance_date = current_date + rebalance_offset
            else:
                transaction_cost = 0
            
            # Get next day's returns
            if date_idx + 1 < len(self.data.index):
                next_day = self.data.index[date_idx + 1]
                next_returns = np.zeros(len(self.config.assets))
                for j, asset in enumerate(self.config.assets):
                    if ('RETURN', asset) in self.data.columns:
                        next_returns[j] = self.data.loc[next_day, ('RETURN', asset)]
                
                # Calculate portfolio return
                portfolio_return = np.sum(current_weights * next_returns) - transaction_cost
                
                # Update portfolio value
                portfolio_value *= (1 + portfolio_return)
                
                # Store history
                weights_history.append(current_weights.copy())
                portfolio_value_history.append(portfolio_value)
                returns_history.append(portfolio_return)
                transaction_cost_history.append(transaction_cost)
        
        # Convert history to pandas Series for return
        portfolio_value_series = pd.Series(
            portfolio_value_history, 
            index=[test_dates[0]] + list(test_dates[len(test_dates)-len(weights_history):])
        )
        
        # Calculate performance metrics
        returns_series = pd.Series(
            returns_history, 
            index=test_dates[len(test_dates)-len(weights_history):]
        )
        
        # Calculate annualized return
        total_return = portfolio_value_series.iloc[-1] / portfolio_value_series.iloc[0] - 1
        years = (portfolio_value_series.index[-1] - portfolio_value_series.index[0]).days / 365.25
        annual_return = (1 + total_return) ** (1 / years) - 1
        
        # Calculate annualized volatility
        daily_vol = np.std(returns_series)
        annual_vol = daily_vol * np.sqrt(252)
        
        # Calculate Sharpe ratio
        sharpe_ratio = annual_return / annual_vol
        
        # Calculate maximum drawdown
        cumulative_returns = (1 + returns_series).cumprod()
        running_max = cumulative_returns.cummax()
        drawdown = (cumulative_returns / running_max - 1)
        max_drawdown = drawdown.min()
        
        # Results
        metrics = {
            'Annual Return': annual_return,
            'Annual Volatility': annual_vol,
            'Sharpe Ratio': sharpe_ratio,
            'Maximum Drawdown': max_drawdown,
            'Total Return': total_return
        }
        
        return portfolio_value_series, metrics
    
    def run_all_benchmark_strategies(self):
        """Run various benchmark strategies for comparison"""
        logger.info("Running benchmark strategies...")
        
        # Define allocation strategies
        allocations = {
            'Equal Weight': {asset: 1.0 / len(self.config.assets) for asset in self.config.asset_names},
            'Stock Heavy': {'Stock': 0.60, 'Bond': 0.20, 'Commodity': 0.10, 'Volatility': 0.10},
            'Bond Heavy': {'Stock': 0.20, 'Bond': 0.60, 'Commodity': 0.10, 'Volatility': 0.10},
            'Risk Balanced': {'Stock': 0.30, 'Bond': 0.40, 'Commodity': 0.20, 'Volatility': 0.10}
        }
        
        # Run strategies
        benchmark_values = {}
        benchmark_metrics = {}
        
        for name, allocation in allocations.items():
            logger.info(f"Running {name} strategy...")
            try:
                values, metrics = self.run_reallocation_strategy(allocation, 'M')  # Monthly rebalancing
                benchmark_values[name] = values
                benchmark_metrics[name] = metrics
            except Exception as e:
                logger.error(f"Error running {name} strategy: {e}")
                logger.debug(traceback.format_exc())
        
        return benchmark_values, benchmark_metrics

    def plot_performance_comparison(self, benchmark_returns=None, save_path=None):
        """Plot performance comparison with benchmarks"""
        if not hasattr(self, 'portfolio_value_df') or self.portfolio_value_df is None or len(self.portfolio_value_df) == 0:
            raise ValueError("No backtest results available. Run backtest first.")
        
        fig, ax = plt.subplots(figsize=(15, 8))
        
        # Plot our portfolio
        ax.plot(self.portfolio_value_df.index, self.portfolio_value_df.values, label='LSTM Portfolio')
        
        # Plot benchmarks if available
        if benchmark_returns is not None:
            for name, returns in benchmark_returns.items():
                if isinstance(returns, pd.Series):
                    # Align index with our portfolio
                    aligned_returns = returns.reindex(self.portfolio_value_df.index, method='ffill')
                    ax.plot(aligned_returns.index, aligned_returns.values, label=name)
        
        ax.set_title('Performance Comparison')
        ax.set_ylabel('Value ($)')
        ax.set_xlabel('Date')
        ax.grid(True)
        ax.legend()
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Comparison saved to {save_path}")
        
        plt.show()


# Main execution
if __name__ == "__main__":
    print("====== Deep Learning for Portfolio Optimization ======")
    
    # Initialize Portfolio Manager
    portfolio_manager = PortfolioManager(config)
    
    # Load data
    data = portfolio_manager.data_handler.load_data()
    
    # Train model
    print("\nTraining model...")
    model = portfolio_manager.train_model()
    
    # Backtest the model
    print("\nBacktesting model...")
    portfolio_value = portfolio_manager.backtest()
    
    # Calculate performance metrics
    metrics = portfolio_manager.calculate_performance_metrics()
    
    # Print performance metrics
    print("\nPerformance Metrics:")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")
    
    # Plot results
    portfolio_manager.plot_results(save_path=os.path.join(config.results_dir, 'lstm_results.png'))
    
    # Run benchmarks
    print("\nRunning benchmark strategies...")
    benchmark_values, benchmark_metrics = portfolio_manager.run_all_benchmark_strategies()
    
    # Print benchmark performance
    print("\nBenchmark Performance Metrics:")
    metrics_df = pd.DataFrame(
        {name: metrics for name, metrics in benchmark_metrics.items()}
    ).transpose()
    
    # Add LSTM model performance
    lstm_metrics = {key: value for key, value in metrics.items() 
                  if key in ['Annual Return', 'Annual Volatility', 'Sharpe Ratio', 'Maximum Drawdown', 'Total Return']}
    metrics_df.loc['LSTM Portfolio'] = lstm_metrics
    
    print(metrics_df)
    
    # Plot performance comparison
    portfolio_manager.plot_performance_comparison(
        benchmark_returns=benchmark_values,
        save_path=os.path.join(config.results_dir, 'performance_comparison.png')
    )
    
    print("\nAnalysis complete.")