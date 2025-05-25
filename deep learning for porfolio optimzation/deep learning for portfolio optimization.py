#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Deep Learning for Portfolio Optimization
This script implements the deep learning portfolio optimization strategy 
based on the paper by Zhang et al. (2020).
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import datetime as dt
import logging
from typing import List, Dict, Tuple, Any
from bloomberg_data_fetcher import BloombergDataFetcher

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")

# Define the ETFs we want to use based on the paper
ETF_SYMBOLS = ["VTI US Equity", "AGG US Equity", "DBC US Equity", "VIX Index"]
ETF_NAMES = ["US Stock", "US Bond", "Commodity", "Volatility"]

class PortfolioLSTM(nn.Module):
    """LSTM model for portfolio optimization as described in the paper"""
    
    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 1, dropout: float = 0.2, output_size: int = len(ETF_NAMES)):
        """Initialize the LSTM model

        Args:
            input_size: Number of input features
            hidden_size: Size of the hidden layer
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            output_size: Number of assets to allocate (defaults to all ETFs)
        """
        super(PortfolioLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Fully connected output layer
        self.fc = nn.Linear(hidden_size, output_size)
        
        # Softmax to ensure weights sum to 1
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network

        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)

        Returns:
            torch.Tensor: Portfolio weights
        """
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate LSTM
        # out shape: (batch_size, seq_length, hidden_size)
        out, _ = self.lstm(x, (h0, c0))
        
        # Get the output from the last time step
        out = self.fc(out[:, -1, :])
        
        # Apply softmax to get portfolio weights (summing to 1)
        weights = self.softmax(out)
        
        return weights


class DeepPortfolioOptimizer:
    """Class to implement Deep Learning Portfolio Optimization"""
    
    def __init__(self, 
                lookback_window: int = 50, 
                hidden_size: int = 64, 
                num_layers: int = 1, 
                dropout: float = 0.2, 
                learning_rate: float = 0.005,
                batch_size: int = 64,
                num_epochs: int = 100,
                cost_rate: float = 0.0001,  # 1 basis point
                vol_target: float = 0.10,   # 10% annualized volatility
                vol_lookback: int = 50):
        """Initialize the portfolio optimizer

        Args:
            lookback_window: Number of days to look back for features
            hidden_size: Size of LSTM hidden layer
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            learning_rate: Learning rate for optimization
            batch_size: Batch size for training
            num_epochs: Number of training epochs
            cost_rate: Transaction cost rate (in decimal)
            vol_target: Target volatility for scaling (annualized)
            vol_lookback: Days to look back for volatility estimation
        """
        self.lookback_window = lookback_window
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.cost_rate = cost_rate
        self.vol_target = vol_target
        self.vol_lookback = vol_lookback
        
        # Other attributes to be initialized later
        self.model = None
        self.data = {}
        self.price_data = None
        self.return_data = None
        self.train_data = None
        self.test_data = None
        self.portfolio_weights = None
        self.portfolio_returns = None
        self.available_etfs = []
        
    def generate_synthetic_data(self, n_days: int = 1000, seed: int = 42) -> bool:
        """Generate synthetic data for testing when Bloomberg data is not available
        
        Args:
            n_days: Number of days of data to generate
            seed: Random seed for reproducibility
            
        Returns:
            bool: True if data was successfully generated
        """
        logger.info("Generating synthetic data for testing...")
        
        try:
            np.random.seed(seed)
            
            # Generate dates
            end_date = dt.datetime.now()
            start_date = end_date - dt.timedelta(days=n_days)
            dates = pd.date_range(start=start_date, end=end_date, freq='D')
            
            # For each ETF, create synthetic price data
            for i, symbol in enumerate(ETF_SYMBOLS):
                # Generate a random walk with drift
                returns = np.random.normal(0.0003, 0.01, len(dates))  # Mean daily return ~7.5% annually
                
                # Add some autocorrelation to simulate momentum
                for j in range(1, len(returns)):
                    returns[j] += 0.05 * returns[j-1]
                
                # Convert returns to prices starting at 100
                prices = 100 * np.cumprod(1 + returns)
                
                # Create a DataFrame
                df = pd.DataFrame({
                    'open': prices * (1 - np.random.uniform(0, 0.005, len(dates))),
                    'high': prices * (1 + np.random.uniform(0, 0.01, len(dates))),
                    'low': prices * (1 - np.random.uniform(0, 0.01, len(dates))),
                    'close': prices,
                    'volume': np.random.randint(100000, 10000000, len(dates)),
                    'numEvents': np.random.randint(1000, 50000, len(dates))
                }, index=dates)
                
                # Add some realistic correlations between ETFs
                if i == 1:  # Bond index (negative correlation with stocks)
                    stock_returns = np.random.normal(0.0003, 0.01, len(dates))
                    bond_returns = -0.3 * stock_returns + np.random.normal(0.0001, 0.003, len(dates))
                    bond_prices = 100 * np.cumprod(1 + bond_returns)
                    df['close'] = bond_prices
                elif i == 2:  # Commodity index (partial correlation with stocks)
                    df['close'] = prices * 0.5 + 50 * np.random.normal(1, 0.2, len(dates))
                elif i == 3:  # Volatility index (strong negative correlation with stocks)
                    # VIX tends to spike when stocks fall
                    vix_base = 15 + 10 * np.random.normal(0, 1, len(dates))
                    for j in range(1, len(dates)):
                        if returns[j] < -0.015:  # Big down day
                            vix_base[j] += 5 * abs(returns[j]) * 100
                    df['close'] = vix_base
                
                self.data[symbol] = df
            
            logger.info(f"Generated synthetic data with {len(dates)} days for {len(ETF_SYMBOLS)} ETFs")
            return True
        
        except Exception as e:
            logger.error(f"Error generating synthetic data: {e}")
            return False
        
    def fetch_bloomberg_data(self, start_date: dt.datetime = None, end_date: dt.datetime = None) -> bool:
        """Fetch data from Bloomberg for each ETF

        Args:
            start_date: Start date for data retrieval
            end_date: End date for data retrieval
            
        Returns:
            bool: True if data was successfully fetched
        """
        logger.info("Fetching data from Bloomberg...")
        
        fetcher = BloombergDataFetcher()
        success = False
        
        try:
            # Start the Bloomberg session
            if not fetcher.start_session():
                logger.error("Failed to initialize Bloomberg session.")
                return False
            
            # Create output directory
            output_dir = "bloomberg_data"
            os.makedirs(output_dir, exist_ok=True)
            
            # Process each ETF
            for symbol in ETF_SYMBOLS:
                # Get intraday bar data
                data = fetcher.get_intraday_bars(
                    symbol, 
                    event_type="TRADE", 
                    interval=30, 
                    start_date=start_date,
                    end_date=end_date
                )
                
                # Save data to file
                if not data.empty:
                    file_path = fetcher.save_data_to_csv(data, symbol, output_dir)
                    
                    # Store the data
                    self.data[symbol] = data
                else:
                    logger.warning(f"No data retrieved for {symbol}")
            
            success = True
        
        except Exception as e:
            logger.error(f"An error occurred: {e}")
            success = False
        
        finally:
            # Stop the Bloomberg session
            fetcher.stop_session()
            
        return success
    
    def load_bloomberg_data_from_files(self, directory: str = "bloomberg_data") -> bool:
        """Load Bloomberg data from saved CSV files

        Args:
            directory: Directory containing the data files
            
        Returns:
            bool: True if data was successfully loaded
        """
        logger.info("Loading Bloomberg data from files...")
        
        if not os.path.exists(directory):
            logger.warning(f"Directory {directory} does not exist.")
            return False
        
        success = False
        
        for symbol in ETF_SYMBOLS:
            # Create a valid filename from the security identifier
            filename = symbol.replace(" ", "_").replace("/", "_").replace("\\", "_")
            file_path = os.path.join(directory, f"{filename}_30min_bars.csv")
            
            if os.path.exists(file_path):
                try:
                    # Read CSV file
                    data = pd.read_csv(file_path)
                    
                    # Ensure the time column is converted to datetime
                    if 'time' in data.columns:
                        data['time'] = pd.to_datetime(data['time'])
                        # Set the time column as the index
                        data.set_index('time', inplace=True)
                    else:
                        logger.warning(f"No 'time' column found in {file_path}. Data may not be properly formatted.")
                        continue
                    
                    self.data[symbol] = data
                    logger.info(f"Loaded {len(data)} data points for {symbol}")
                    success = True
                except Exception as e:
                    logger.error(f"Error loading data from {file_path}: {e}")
            else:
                logger.warning(f"Data file not found for {symbol}: {file_path}")
        
        return success
    
    def inspect_data(self):
        """Inspect the loaded data to help debug issues"""
        logger.info("Inspecting loaded data...")
        
        if not self.data:
            logger.error("No data has been loaded.")
            return
        
        for symbol, data in self.data.items():
            logger.info(f"Data for {symbol}:")
            logger.info(f"  Type: {type(data)}")
            logger.info(f"  Shape: {data.shape}")
            logger.info(f"  Index type: {type(data.index)}")
            logger.info(f"  First 3 rows:")
            logger.info(f"{data.head(3)}")
            logger.info(f"  Column names: {list(data.columns)}")
            logger.info("-" * 40)
    
    def preprocess_data(self) -> None:
        """Preprocess the data for model training"""
        logger.info("Preprocessing data...")
        
        if not self.data:
            logger.error("No data available. Please fetch or load data first.")
            return
        
        # Resample to daily data (use last price of the day)
        daily_data = {}
        
        for symbol, data in self.data.items():
            try:
                # Check if the data has a datetime index
                if not isinstance(data.index, pd.DatetimeIndex):
                    logger.error(f"Data for {symbol} does not have a DatetimeIndex. Skipping.")
                    continue
                
                # Resample to daily (using last value of the day)
                daily = data.resample('D').last()
                
                # Forward fill to handle missing days
                daily = daily.fillna(method='ffill')
                
                # Store in daily_data
                daily_data[symbol] = daily
                logger.info(f"Successfully processed data for {symbol} with {len(daily)} daily observations")
            except Exception as e:
                logger.error(f"Error processing data for {symbol}: {e}")
        
        if not daily_data:
            logger.error("No data could be processed. Cannot continue.")
            return
        
        # Create DataFrame with close prices for all ETFs
        close_prices = pd.DataFrame()
        
        for i, symbol in enumerate(ETF_SYMBOLS):
            if symbol in daily_data:
                # Use the associated name instead of the symbol for better readability
                if 'close' in daily_data[symbol].columns:
                    close_prices[ETF_NAMES[i]] = daily_data[symbol]['close']
                    logger.info(f"Using 'close' column for {ETF_NAMES[i]}")
                elif 'Close' in daily_data[symbol].columns:
                    close_prices[ETF_NAMES[i]] = daily_data[symbol]['Close']
                    logger.info(f"Using 'Close' column for {ETF_NAMES[i]}")
                else:
                    available_cols = list(daily_data[symbol].columns)
                    logger.warning(f"No 'close' or 'Close' column found for {symbol}. "
                                  f"Available columns: {available_cols}")
        
        # Check if we have price data for all symbols
        if len(close_prices.columns) != len(ETF_NAMES):
            missing = set(ETF_NAMES) - set(close_prices.columns)
            logger.warning(f"Missing price data for: {missing}")
        
        # Drop rows with any missing data
        orig_len = len(close_prices)
        close_prices = close_prices.dropna()
        if len(close_prices) < orig_len:
            logger.info(f"Dropped {orig_len - len(close_prices)} rows with missing data")
        
        # Calculate daily returns
        returns = close_prices.pct_change().dropna()
        
        # Store the processed data
        self.price_data = close_prices
        self.return_data = returns
        
        # Update available ETFs
        self.available_etfs = [etf for etf in ETF_NAMES if etf in close_prices.columns]
        
        logger.info(f"Preprocessed data has {len(close_prices)} days and {len(close_prices.columns)} ETFs")
        logger.info(f"Available ETFs: {self.available_etfs}")
    
    def prepare_training_data(self, train_test_split: float = 0.8) -> None:
        """Prepare data for model training

        Args:
            train_test_split: Ratio of training data to total data
        """
        logger.info("Preparing training data...")
        
        if self.price_data is None or self.return_data is None:
            logger.error("Price or return data is missing. Please run preprocess_data first.")
            return
        
        try:
            # Make sure we use the same indices for prices and returns
            valid_indices = self.price_data.index.intersection(self.return_data.index)
            price_data = self.price_data.loc[valid_indices]
            return_data = self.return_data.loc[valid_indices]
            
            # Extract only ETFs that are available in both datasets
            available_etfs = [etf for etf in ETF_NAMES if etf in price_data.columns and etf in return_data.columns]
            n_etfs = len(available_etfs)
            
            # Update class attribute
            self.available_etfs = available_etfs
            
            logger.info(f"Using {n_etfs} ETFs for training: {available_etfs}")
            
            if n_etfs == 0:
                logger.error("No common ETFs found in price and return data.")
                return
            
            # Calculate features (price returns and direct returns)
            price_returns = price_data[available_etfs].pct_change().dropna()
            
            # Align return data with price returns
            common_indices = price_returns.index.intersection(return_data.index)
            price_returns = price_returns.loc[common_indices]
            returns = return_data[available_etfs].loc[common_indices]
            
            # Create features array
            features = np.zeros((len(common_indices), n_etfs * 2))
            
            for i, etf in enumerate(available_etfs):
                features[:, i] = price_returns[etf].values
                features[:, i + n_etfs] = returns[etf].values
            
            # Create sequences
            X, y = [], []
            
            for i in range(len(features) - self.lookback_window):
                # Input sequence
                sequence = features[i:i+self.lookback_window]
                X.append(sequence)
                
                # Target is the next day's returns
                next_returns = features[i+self.lookback_window, :n_etfs]
                y.append(next_returns)
            
            # Convert to numpy arrays
            X = np.array(X)
            y = np.array(y)
            
            # Ensure we have enough data
            if len(X) < 2:
                logger.error(f"Not enough data for training. Only {len(X)} sequences created.")
                return
            
            # Split into train and test sets
            train_size = max(1, int(len(X) * train_test_split))
            
            X_train, X_test = X[:train_size], X[train_size:]
            y_train, y_test = y[:train_size], y[train_size:]
            
            # Convert to PyTorch tensors
            X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
            y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
            X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
            y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(device)
            
            # Create datasets and dataloaders
            train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
            
            self.train_data = DataLoader(train_dataset, batch_size=min(self.batch_size, len(train_dataset)), shuffle=True)
            self.test_data = (X_test_tensor, y_test_tensor)
            
            logger.info(f"Prepared {len(X_train)} training and {len(X_test)} testing samples")
        
        except Exception as e:
            logger.error(f"Error preparing training data: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    def sharpe_ratio_loss(self, weights: torch.Tensor, returns: torch.Tensor) -> torch.Tensor:
        """Calculate the negative Sharpe ratio (to be minimized)

        Args:
            weights: Portfolio weights
            returns: Asset returns

        Returns:
            torch.Tensor: Negative Sharpe ratio
        """
        # Calculate portfolio returns
        portfolio_returns = torch.sum(weights * returns, dim=1)
        
        # Calculate mean and standard deviation
        mean_return = torch.mean(portfolio_returns)
        std_return = torch.std(portfolio_returns)
        
        # Add a small constant to avoid division by zero
        epsilon = 1e-6
        
        # Calculate Sharpe ratio (negated for minimization)
        sharpe_ratio = -mean_return / (std_return + epsilon)
        
        return sharpe_ratio
    
    def train_model(self) -> None:
        """Train the LSTM model to optimize Sharpe ratio"""
        logger.info("Training the model...")
        
        if self.train_data is None or self.test_data is None:
            logger.error("Training data is missing. Please run prepare_training_data first.")
            return
        
        try:
            # Get the input size from the training data
            input_size = self.train_data.dataset.tensors[0].shape[2]
            output_size = len(self.available_etfs)
            
            logger.info(f"Initializing model with input size {input_size} and output size {output_size}")
            
            # Initialize model
            self.model = PortfolioLSTM(
                input_size=input_size,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                dropout=self.dropout,
                output_size=output_size
            ).to(device)
            
            # Initialize optimizer
            optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
            
            # Keep track of best model
            best_sharpe = float('-inf')
            best_weights = None
            
            # Training loop
            for epoch in range(self.num_epochs):
                self.model.train()
                epoch_loss = 0.0
                
                for batch_X, batch_returns in self.train_data:
                    # Forward pass
                    weights = self.model(batch_X)
                    
                    # Calculate Sharpe ratio loss
                    loss = self.sharpe_ratio_loss(weights, batch_returns)
                    
                    # Backward pass and optimize
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                
                # Evaluate on test data every 10 epochs
                if (epoch + 1) % 10 == 0 or epoch == self.num_epochs - 1:
                    self.model.eval()
                    with torch.no_grad():
                        X_test, y_test = self.test_data
                        weights = self.model(X_test)
                        test_loss = self.sharpe_ratio_loss(weights, y_test)
                        
                        # Calculate Sharpe ratio (positive now for reporting)
                        sharpe = -test_loss.item()
                        
                        logger.info(f"Epoch {epoch+1}/{self.num_epochs}, "
                                    f"Train Loss: {epoch_loss/len(self.train_data):.4f}, "
                                    f"Test Sharpe: {sharpe:.4f}")
                        
                        # Update best model if this one is better
                        if sharpe > best_sharpe:
                            best_sharpe = sharpe
                            best_weights = self.model.state_dict().copy()
            
            # Use the best model for predictions
            if best_weights is not None:
                self.model.load_state_dict(best_weights)
                logger.info(f"Using best model with Sharpe ratio: {best_sharpe:.4f}")
            
            # Save the model
            try:
                torch.save(self.model.state_dict(), 'portfolio_lstm_model.pth')
                logger.info("Model saved to portfolio_lstm_model.pth")
            except Exception as e:
                logger.warning(f"Could not save model: {e}")
        
        except Exception as e:
            logger.error(f"Error training model: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    def generate_portfolio_weights(self) -> pd.DataFrame:
        """Generate portfolio weights for the entire dataset

        Returns:
            pd.DataFrame: Portfolio weights for each day
        """
        logger.info("Generating portfolio weights...")
        
        if self.model is None:
            logger.error("Model has not been trained yet.")
            return None
        
        if self.price_data is None or self.return_data is None:
            logger.error("Price or return data is missing.")
            return None
        
        try:
            self.model.eval()
            
            # Make sure we use the same indices for prices and returns
            valid_indices = self.price_data.index.intersection(self.return_data.index)
            price_data = self.price_data.loc[valid_indices]
            return_data = self.return_data.loc[valid_indices]
            
            # Get the true lengths after alignment
            logger.info(f"Price data length: {len(price_data)}, Return data length: {len(return_data)}")
            
            # Extract only ETFs that are available in both datasets
            available_etfs = self.available_etfs
            n_etfs = len(available_etfs)
            
            logger.info(f"Using {n_etfs} ETFs for weight generation: {available_etfs}")
            
            if n_etfs == 0:
                logger.error("No ETFs available for weight generation.")
                return None
            
            # Calculate features (returns and lagged returns)
            price_returns = price_data[available_etfs].pct_change().dropna()
            
            # Align return data with price returns
            common_indices = price_returns.index.intersection(return_data.index)
            
            if len(common_indices) <= self.lookback_window:
                logger.error(f"Not enough data points after alignment. Only {len(common_indices)} available.")
                return None
            
            price_returns = price_returns.loc[common_indices]
            returns = return_data[available_etfs].loc[common_indices]
            
            # Now create the features array
            features = np.zeros((len(common_indices), n_etfs * 2))
            
            for i, etf in enumerate(available_etfs):
                features[:, i] = price_returns[etf].values
                features[:, i + n_etfs] = returns[etf].values
            
            # Create sequences for the LSTM
            X = []
            # Use the common indices for dates, starting after the lookback_window
            dates = common_indices[self.lookback_window:]
            
            for i in range(len(features) - self.lookback_window):
                sequence = features[i:i+self.lookback_window]
                X.append(sequence)
            
            if not X:
                logger.error("No valid sequences could be created.")
                return None
            
            # Convert to numpy array
            X = np.array(X)
            
            # Convert to PyTorch tensor
            X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
            
            # Get portfolio weights from the model
            with torch.no_grad():
                weights = self.model(X_tensor).cpu().numpy()
            
            # Create DataFrame with weights for available ETFs
            weights_df = pd.DataFrame(weights, index=dates, columns=available_etfs)
            
            # Expand to include all ETFs (with 0 weight for unavailable ones)
            full_weights_df = pd.DataFrame(0, index=dates, columns=ETF_NAMES)
            for etf in available_etfs:
                full_weights_df[etf] = weights_df[etf]
            
            self.portfolio_weights = full_weights_df
            logger.info(f"Generated weights for {len(full_weights_df)} days.")
            
            return full_weights_df
        
        except Exception as e:
            logger.error(f"Error generating portfolio weights: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def calculate_portfolio_returns(self) -> pd.DataFrame:
        """Calculate portfolio returns using generated weights

        Returns:
            pd.DataFrame: Portfolio returns and metrics
        """
        logger.info("Calculating portfolio returns...")
        
        if self.portfolio_weights is None:
            logger.error("Portfolio weights have not been generated yet.")
            return None
        
        try:
            # Get the ETFs with non-zero weights
            active_etfs = [etf for etf in ETF_NAMES if etf in self.portfolio_weights.columns 
                          and self.portfolio_weights[etf].sum() > 0]
            
            logger.info(f"Calculating returns using {len(active_etfs)} active ETFs: {active_etfs}")
            
            if not active_etfs:
                logger.error("No ETFs with non-zero weights found.")
                return None
            
            # Align return data with weights (shift returns forward by 1 day)
            active_returns = self.return_data[active_etfs]
            aligned_returns = active_returns.shift(-1)
            
            # Find common indices between weights and aligned returns
            common_indices = self.portfolio_weights.index.intersection(aligned_returns.index)
            
            if len(common_indices) == 0:
                logger.error("No common dates found between portfolio weights and return data.")
                return None
            
            logger.info(f"Found {len(common_indices)} common dates for return calculation.")
            
            # Get weights and returns for common dates
            weights = self.portfolio_weights.loc[common_indices, active_etfs]
            asset_returns = aligned_returns.loc[common_indices]
            
            # Drop any dates with NaN values
            valid_indices = asset_returns.dropna(how='any').index
            weights = weights.loc[valid_indices]
            asset_returns = asset_returns.loc[valid_indices]
            
            logger.info(f"After dropping NaNs, using {len(valid_indices)} dates for return calculation.")
            
            # Calculate portfolio returns without transaction costs
            portfolio_returns = pd.Series(
                np.sum(weights.values * asset_returns.values, axis=1),
                index=valid_indices,
                name='portfolio_returns'
            )
            
            # Calculate transaction costs
            weights_shifted = weights.shift(1).fillna(0)
            turnover = np.abs(weights - weights_shifted).sum(axis=1)
            transaction_costs = turnover * self.cost_rate
            
            # Create a DataFrame to hold all results
            results = pd.DataFrame({
                'portfolio_returns': portfolio_returns,
                'transaction_costs': transaction_costs
            })
            
            # Adjust returns for transaction costs
            results['portfolio_returns_adj'] = results['portfolio_returns'] - results['transaction_costs']
            
            # Calculate volatility for scaling using expanding window if not enough data for rolling
            if len(results) >= self.vol_lookback:
                rolling_vol = results['portfolio_returns_adj'].rolling(window=self.vol_lookback).std() * np.sqrt(252)
            else:
                rolling_vol = results['portfolio_returns_adj'].expanding().std() * np.sqrt(252)
            
            results['volatility'] = rolling_vol
            
            # Fill NaN values in volatility (can happen at the beginning of the series)
            results['volatility'] = results['volatility'].fillna(method='bfill').fillna(0.01)  # Default to 1% if can't be calculated
            
            # Calculate scaling factors
            results['scaling_factors'] = self.vol_target / results['volatility']
            
            # Generate scaled returns
            results['scaled_returns'] = results['portfolio_returns_adj'] * results['scaling_factors']
            
            # Calculate cumulative returns
            results['cumulative_returns'] = (1 + results['portfolio_returns_adj']).cumprod() - 1
            results['scaled_cumulative_returns'] = (1 + results['scaled_returns']).cumprod() - 1
            
            self.portfolio_returns = results
            
            return results
        
        except Exception as e:
            logger.error(f"Error calculating portfolio returns: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return pd.DataFrame()  # Return empty DataFrame on error
    
    def calculate_performance_metrics(self) -> Dict[str, float]:
        """Calculate performance metrics for the portfolio

        Returns:
            Dict[str, float]: Dictionary of performance metrics
        """
        logger.info("Calculating performance metrics...")
        
        if self.portfolio_returns is None or self.portfolio_returns.empty:
            logger.error("Portfolio returns have not been calculated yet.")
            return {}
        
        try:
            # Extract returns
            returns = self.portfolio_returns['portfolio_returns_adj']
            scaled_returns = self.portfolio_returns['scaled_returns']
            
            # Calculate metrics
            metrics = {}
            
            # Expected return (annualized)
            metrics['expected_return'] = returns.mean() * 252
            metrics['scaled_expected_return'] = scaled_returns.mean() * 252
            
            # Volatility (annualized)
            metrics['volatility'] = returns.std() * np.sqrt(252)
            metrics['scaled_volatility'] = scaled_returns.std() * np.sqrt(252)
            
            # Sharpe ratio (annualized)
            metrics['sharpe_ratio'] = metrics['expected_return'] / metrics['volatility'] if metrics['volatility'] > 0 else 0
            metrics['scaled_sharpe_ratio'] = metrics['scaled_expected_return'] / metrics['scaled_volatility'] if metrics['scaled_volatility'] > 0 else 0
            
            # Downside deviation (annualized)
            downside_returns = returns[returns < 0]
            scaled_downside_returns = scaled_returns[scaled_returns < 0]
            metrics['downside_deviation'] = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
            metrics['scaled_downside_deviation'] = scaled_downside_returns.std() * np.sqrt(252) if len(scaled_downside_returns) > 0 else 0
            
            # Sortino ratio (annualized)
            metrics['sortino_ratio'] = metrics['expected_return'] / metrics['downside_deviation'] if metrics['downside_deviation'] > 0 else 0
            metrics['scaled_sortino_ratio'] = metrics['scaled_expected_return'] / metrics['scaled_downside_deviation'] if metrics['scaled_downside_deviation'] > 0 else 0
            
            # Maximum drawdown
            cumulative_returns = self.portfolio_returns['cumulative_returns']
            scaled_cumulative_returns = self.portfolio_returns['scaled_cumulative_returns']
            
            running_max = cumulative_returns.cummax()
            drawdowns = (cumulative_returns - running_max) / (1 + running_max)
            metrics['max_drawdown'] = drawdowns.min()
            
            scaled_running_max = scaled_cumulative_returns.cummax()
            scaled_drawdowns = (scaled_cumulative_returns - scaled_running_max) / (1 + scaled_running_max)
            metrics['scaled_max_drawdown'] = scaled_drawdowns.min()
            
            # Percentage of positive returns
            metrics['percent_positive'] = (returns > 0).mean()
            metrics['scaled_percent_positive'] = (scaled_returns > 0).mean()
            
            # Average positive return / Average negative return
            avg_positive = returns[returns > 0].mean() if len(returns[returns > 0]) > 0 else 0
            avg_negative = returns[returns < 0].mean() if len(returns[returns < 0]) > 0 else 0
            metrics['avg_positive_to_negative'] = abs(avg_positive / avg_negative) if avg_negative != 0 else 0
            
            scaled_avg_positive = scaled_returns[scaled_returns > 0].mean() if len(scaled_returns[scaled_returns > 0]) > 0 else 0
            scaled_avg_negative = scaled_returns[scaled_returns < 0].mean() if len(scaled_returns[scaled_returns < 0]) > 0 else 0
            metrics['scaled_avg_positive_to_negative'] = abs(scaled_avg_positive / scaled_avg_negative) if scaled_avg_negative != 0 else 0
            
            return metrics
        
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {}
    
    def plot_results(self, benchmark_symbol: str = "VTI US Equity") -> None:
        """Plot portfolio results

        Args:
            benchmark_symbol: Symbol to use as benchmark (default is US stock index)
        """
        logger.info("Plotting results...")
        
        if self.portfolio_returns is None or self.portfolio_weights is None:
            logger.error("Portfolio returns or weights have not been calculated yet.")
            return
        
        try:
            # Create figure with multiple subplots
            fig, axs = plt.subplots(3, 1, figsize=(14, 15), sharex=True)
            
            # Get the ETFs with non-zero weights
            active_etfs = [etf for etf in ETF_NAMES if etf in self.portfolio_weights.columns 
                          and self.portfolio_weights[etf].sum() > 0]
            
            # Plot 1: Portfolio weights over time
            self.portfolio_weights[active_etfs].plot(ax=axs[0], colormap='viridis')
            axs[0].set_ylabel('Allocation')
            axs[0].set_title('Portfolio Weights')
            axs[0].legend(loc='upper left')
            axs[0].grid(True)
            
            # Plot 2: Portfolio cumulative returns vs benchmark
            axs[1].plot(self.portfolio_returns.index, self.portfolio_returns['cumulative_returns'], 
                       label='Portfolio')
            axs[1].plot(self.portfolio_returns.index, self.portfolio_returns['scaled_cumulative_returns'],
                       label=f'Scaled Portfolio (Vol Target: {self.vol_target*100:.0f}%)')
            
            # Add benchmark if available
            benchmark_name = None
            if benchmark_symbol in ETF_SYMBOLS:
                idx = ETF_SYMBOLS.index(benchmark_symbol)
                benchmark_name = ETF_NAMES[idx]
            
            if benchmark_name and benchmark_name in self.return_data:
                # Only plot benchmark for the same date range as portfolio returns
                if len(self.portfolio_returns) > 0:
                    benchmark_returns = self.return_data[benchmark_name].loc[
                        self.return_data.index.intersection(self.portfolio_returns.index)
                    ]
                    if len(benchmark_returns) > 0:
                        benchmark_cumulative = (1 + benchmark_returns).cumprod() - 1
                        axs[1].plot(benchmark_returns.index, benchmark_cumulative, 
                                   label=f'Benchmark ({benchmark_name})', linestyle='--')
            
            axs[1].set_ylabel('Cumulative Return')
            axs[1].set_title('Portfolio Performance')
            axs[1].legend(loc='upper left')
            axs[1].grid(True)
            
            # Plot 3: Drawdowns
            cumulative_returns = self.portfolio_returns['cumulative_returns']
            running_max = cumulative_returns.cummax()
            drawdowns = (cumulative_returns - running_max) / (1 + running_max)
            
            scaled_cumulative_returns = self.portfolio_returns['scaled_cumulative_returns']
            scaled_running_max = scaled_cumulative_returns.cummax()
            scaled_drawdowns = (scaled_cumulative_returns - scaled_running_max) / (1 + scaled_running_max)
            
            axs[2].fill_between(drawdowns.index, 0, drawdowns, color='red', alpha=0.3, label='Portfolio Drawdowns')
            axs[2].fill_between(scaled_drawdowns.index, 0, scaled_drawdowns, color='blue', alpha=0.3, label='Scaled Portfolio Drawdowns')
            axs[2].set_ylabel('Drawdown')
            axs[2].set_title('Portfolio Drawdowns')
            axs[2].legend(loc='lower left')
            axs[2].grid(True)
            
            plt.tight_layout()
            plt.savefig('deep_portfolio_results.png', dpi=300)
            plt.show()
        
        except Exception as e:
            logger.error(f"Error plotting results: {e}")
            import traceback
            logger.error(traceback.format_exc())

    def analyze_asset_contributions(self) -> None:
        """Analyze how each asset contributes to portfolio performance"""
        logger.info("Analyzing asset contributions to portfolio performance...")
        
        if self.portfolio_returns is None or self.portfolio_weights is None:
            logger.error("Portfolio returns or weights have not been calculated yet.")
            return
        
        try:
            # Get active ETFs (those with non-zero weights)
            active_etfs = [etf for etf in ETF_NAMES if etf in self.portfolio_weights.columns 
                          and self.portfolio_weights[etf].sum() > 0]
            
            if not active_etfs:
                logger.error("No ETFs with non-zero weights found.")
                return
            
            # Get aligned data for calculation
            aligned_returns = self.return_data[active_etfs].shift(-1)
            
            # Find common indices between returns, weights, and portfolio returns
            common_indices = (self.portfolio_returns.index
                             .intersection(self.portfolio_weights.index)
                             .intersection(aligned_returns.index))
            
            if len(common_indices) == 0:
                logger.error("No common dates found between returns and weights.")
                return
            
            logger.info(f"Found {len(common_indices)} common dates for contribution analysis.")
            
            # Calculate contribution of each asset to portfolio returns
            contributions = pd.DataFrame(index=common_indices)
            
            weights = self.portfolio_weights.loc[common_indices, active_etfs]
            asset_returns = aligned_returns.loc[common_indices]
            
            # Drop any dates with NaN values
            valid_mask = ~asset_returns.isna().any(axis=1)
            valid_indices = common_indices[valid_mask]
            
            if len(valid_indices) == 0:
                logger.error("No valid dates found after dropping NaNs.")
                return
            
            weights = weights.loc[valid_indices]
            asset_returns = asset_returns.loc[valid_indices]
            
            # Calculate weighted returns for each asset
            for etf in active_etfs:
                contributions[f"{etf}_contrib"] = weights[etf] * asset_returns[etf]
            
            # Calculate percentage contribution
            total_returns = contributions.sum(axis=1)
            for etf in active_etfs:
                # Avoid division by zero
                non_zero_returns = total_returns != 0
                if non_zero_returns.any():
                    contributions.loc[non_zero_returns, f"{etf}_pct"] = (
                        contributions.loc[non_zero_returns, f"{etf}_contrib"] / 
                        total_returns.loc[non_zero_returns]
                    )
                else:
                    contributions[f"{etf}_pct"] = 0
            
            # Replace NaNs and infinities
            contributions = contributions.replace([np.inf, -np.inf], np.nan).fillna(0)
            
            # Calculate average contribution percentages
            avg_contributions = {}
            for etf in active_etfs:
                if f"{etf}_pct" in contributions.columns:
                    avg_contributions[etf] = contributions[f"{etf}_pct"].mean()
            
            if not avg_contributions:
                logger.warning("No contribution data could be calculated.")
                return
            
            # Create a pie chart of average contributions
            plt.figure(figsize=(10, 8))
            plt.pie(
                [abs(val) for val in avg_contributions.values()],
                labels=avg_contributions.keys(),
                autopct='%1.1f%%',
                startangle=90,
                colors=plt.cm.viridis(np.linspace(0, 1, len(avg_contributions))),
                normalize=True  # Fix the normalization warning
            )
            plt.title('Average Asset Contribution to Portfolio Returns')
            plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
            plt.savefig('asset_contributions.png', dpi=300)
            plt.show()
            
            # For the stacked area chart, use two separate charts for positive and negative contributions
            # and use line chart instead for combined view
            plt.figure(figsize=(14, 8))
            
            # Extract contribution data
            contribution_data = pd.DataFrame()
            for etf in active_etfs:
                if f"{etf}_contrib" in contributions.columns:
                    contribution_data[etf] = contributions[f"{etf}_contrib"]
            
            # Plot as a line chart showing each asset's contribution over time
            contribution_data.cumsum().plot(colormap='viridis', linewidth=2)
            plt.title('Cumulative Contribution to Returns by Asset')
            plt.ylabel('Cumulative Return')
            plt.grid(True, alpha=0.3)
            plt.legend(loc='upper left')
            plt.savefig('cumulative_contributions_line.png', dpi=300)
            plt.show()
            
            # Try to create separate stacked area charts for positive and negative contributions
            try:
                # Create separate DataFrames for positive and negative contributions
                positive_contrib = contribution_data.copy()
                negative_contrib = contribution_data.copy()
                
                # Set negative values to 0 in positive_contrib
                for col in positive_contrib.columns:
                    positive_contrib.loc[positive_contrib[col] < 0, col] = 0
                    
                # Set positive values to 0 in negative_contrib and take absolute value
                for col in negative_contrib.columns:
                    negative_contrib.loc[negative_contrib[col] >= 0, col] = 0
                    negative_contrib[col] = negative_contrib[col].abs() * -1  # Keep negative but take absolute for visualization
                
                # Plot positive contributions as stacked area
                if positive_contrib.sum().sum() > 0:
                    plt.figure(figsize=(14, 8))
                    positive_contrib.cumsum().plot.area(stacked=True, colormap='viridis', alpha=0.7)
                    plt.title('Cumulative Positive Contributions by Asset')
                    plt.ylabel('Cumulative Positive Return')
                    plt.grid(True, alpha=0.3)
                    plt.legend(loc='upper left')
                    plt.savefig('positive_contributions.png', dpi=300)
                    plt.show()
                
                # Plot negative contributions as stacked area (if there are any meaningful negative values)
                if negative_contrib.sum().sum() < 0:
                    plt.figure(figsize=(14, 8))
                    negative_contrib.cumsum().plot.area(stacked=True, colormap='viridis', alpha=0.7)
                    plt.title('Cumulative Negative Contributions by Asset')
                    plt.ylabel('Cumulative Negative Return')
                    plt.grid(True, alpha=0.3)
                    plt.legend(loc='lower left')
                    plt.savefig('negative_contributions.png', dpi=300)
                    plt.show()
            
            except Exception as e:
                logger.warning(f"Could not create stacked area charts: {e}")
                logger.warning("Falling back to basic line chart only.")
            
            # Print numerical results
            print("\n=== Average Asset Contributions ===")
            for etf, contrib in avg_contributions.items():
                print(f"{etf}: {contrib*100:.2f}%")
            
            # Also print total contribution of each asset
            print("\n=== Total Contribution by Asset ===")
            total_contribution = contribution_data.sum()
            total_abs = total_contribution.abs().sum()
            if total_abs > 0:
                for etf in contribution_data.columns:
                    print(f"{etf}: {total_contribution[etf]/total_abs*100:.2f}% (${total_contribution[etf]:.4f})")
        
        except Exception as e:
            logger.error(f"Error analyzing asset contributions: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    def run_strategy(self, fetch_new_data: bool = False, use_synthetic_data: bool = True) -> Dict[str, Any]:
        """Run the complete portfolio optimization strategy

        Args:
            fetch_new_data: Whether to fetch new data from Bloomberg
            use_synthetic_data: Whether to use synthetic data if no Bloomberg data is available

        Returns:
            Dict[str, Any]: Results and metrics
        """
        # Step 1: Load or fetch data
        data_loaded = False
        
        print("\n--------- DATA LOADING PHASE ---------")
        
        if fetch_new_data:
            print("Attempting to fetch new data from Bloomberg...")
            end_date = dt.datetime.now()
            start_date = end_date - dt.timedelta(days=500)  # Get about 2 years of data
            data_loaded = self.fetch_bloomberg_data(start_date, end_date)
        else:
            # Try to load existing data
            print("Attempting to load data from saved files...")
            data_loaded = self.load_bloomberg_data_from_files()
            
            # If no data found, fetch new data
            if not data_loaded:
                print("No existing data found. Attempting to fetch new data...")
                end_date = dt.datetime.now()
                start_date = end_date - dt.timedelta(days=500)
                data_loaded = self.fetch_bloomberg_data(start_date, end_date)
        
        # If we still don't have data and synthetic data is allowed, generate it
        if not data_loaded and use_synthetic_data:
            print("No Bloomberg data available. Generating synthetic data for testing...")
            data_loaded = self.generate_synthetic_data()
            if data_loaded:
                print("Synthetic data successfully generated.")
        
        if not data_loaded:
            print("ERROR: Failed to load or fetch data from any source.")
            return {}
        
        print("\n--------- DATA PREPROCESSING PHASE ---------")
        
        # Step 2: Preprocess data
        print("Preprocessing data...")
        self.preprocess_data()
        
        if self.price_data is None or len(self.price_data) == 0:
            print("ERROR: Failed to preprocess data. Cannot continue.")
            return {}
        
        print(f"Successfully preprocessed data with {len(self.price_data)} days.")
        print(f"ETFs in dataset: {list(self.price_data.columns)}")
        
        print("\n--------- MODEL TRAINING PHASE ---------")
        
        # Step 3: Prepare training data
        print("Preparing training data...")
        self.prepare_training_data()
        
        if self.train_data is None or self.test_data is None:
            print("ERROR: Failed to prepare training data. Cannot continue.")
            return {}
        
        # Step 4: Train model
        print("Training model...")
        self.train_model()
        
        if self.model is None:
            print("ERROR: Model training failed. Cannot continue.")
            return {}
        
        print("\n--------- PORTFOLIO OPTIMIZATION PHASE ---------")
        
        # Step 5: Generate portfolio weights
        print("Generating portfolio weights...")
        weights = self.generate_portfolio_weights()
        
        if weights is None or weights.empty:
            print("ERROR: Failed to generate portfolio weights. Cannot continue.")
            return {}
        
        print(f"Successfully generated weights for {len(weights)} days.")
        
        # Step 6: Calculate portfolio returns
        print("Calculating portfolio returns...")
        returns = self.calculate_portfolio_returns()
        
        if returns is None or returns.empty:
            print("ERROR: Failed to calculate portfolio returns. Cannot continue.")
            return {}
        
        print(f"Successfully calculated returns for {len(returns)} days.")
        
        # Step 7: Calculate performance metrics
        print("Calculating performance metrics...")
        metrics = self.calculate_performance_metrics()
        
        if not metrics:
            print("WARNING: Could not calculate performance metrics.")
        
        print("\n--------- VISUALIZATION PHASE ---------")
        
        # Step 8: Plot results
        try:
            print("Creating performance visualizations...")
            self.plot_results()
            
            # Step 9: Analyze asset contributions
            print("Analyzing asset contributions...")
            self.analyze_asset_contributions()
            
            print("Visualizations completed successfully.")
        except Exception as e:
            print(f"WARNING: Failed to create visualizations: {e}")
            import traceback
            print(traceback.format_exc())
        
        # Return results
        return {
            'metrics': metrics,
            'weights': self.portfolio_weights,
            'returns': self.portfolio_returns
        }


def main():
    """Main function to run the strategy"""
    # Initialize the portfolio optimizer
    optimizer = DeepPortfolioOptimizer(
        lookback_window=50,
        hidden_size=64,
        num_layers=1,
        dropout=0.2,
        learning_rate=0.005,
        batch_size=64,
        num_epochs=100,
        cost_rate=0.0001,  # 1 basis point
        vol_target=0.10,   # 10% annualized volatility
        vol_lookback=50
    )
    
    # First try to load and inspect Bloomberg data
    success = optimizer.load_bloomberg_data_from_files()
    if success:
        optimizer.inspect_data()
    
    # Run the strategy with synthetic data as fallback
    results = optimizer.run_strategy(fetch_new_data=False, use_synthetic_data=True)
    
    # Check if results were obtained
    if not results:
        logger.error("Strategy run failed.")
        return
    
    # Print performance metrics
    print("\n=== Portfolio Performance Metrics ===")
    metrics = results['metrics']
    
    print(f"Expected Return (Annualized): {metrics['expected_return']*100:.2f}%")
    print(f"Volatility (Annualized): {metrics['volatility']*100:.2f}%")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"Sortino Ratio: {metrics['sortino_ratio']:.2f}")
    print(f"Maximum Drawdown: {metrics['max_drawdown']*100:.2f}%")
    print(f"Percentage of Positive Returns: {metrics['percent_positive']*100:.2f}%")
    print(f"Avg Positive Return / Avg Negative Return: {metrics['avg_positive_to_negative']:.2f}")
    
    print("\n=== Scaled Portfolio Performance Metrics (Vol Target: 10%) ===")
    print(f"Expected Return (Annualized): {metrics['scaled_expected_return']*100:.2f}%")
    print(f"Volatility (Annualized): {metrics['scaled_volatility']*100:.2f}%")
    print(f"Sharpe Ratio: {metrics['scaled_sharpe_ratio']:.2f}")
    print(f"Sortino Ratio: {metrics['scaled_sortino_ratio']:.2f}")
    print(f"Maximum Drawdown: {metrics['scaled_max_drawdown']*100:.2f}%")
    print(f"Percentage of Positive Returns: {metrics['scaled_percent_positive']*100:.2f}%")
    print(f"Avg Positive Return / Avg Negative Return: {metrics['scaled_avg_positive_to_negative']:.2f}")


if __name__ == "__main__":
    main()