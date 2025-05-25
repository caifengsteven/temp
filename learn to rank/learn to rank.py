#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ListFold: Long-Short Portfolio Construction using Learn-to-Rank
Implementation based on the paper "Constructing long-short stock portfolio with a new listwise learn-to-rank algorithm"
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import logging
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import spearmanr

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import the Bloomberg API
try:
    import blpapi
    BLOOMBERG_AVAILABLE = True
except ImportError:
    logger.warning("Bloomberg API (blpapi) not available. Will use fallback data.")
    BLOOMBERG_AVAILABLE = False


class StockDataset(Dataset):
    """Dataset for stock data with factors."""
    def __init__(self, factors, returns):
        """
        Initialize the dataset.
        
        Args:
            factors: Factors data (n_samples, n_stocks, n_factors)
            returns: Returns data (n_samples, n_stocks)
        """
        self.factors = torch.FloatTensor(factors)
        self.returns = torch.FloatTensor(returns)
    
    def __len__(self):
        return len(self.factors)
    
    def __getitem__(self, idx):
        return self.factors[idx], self.returns[idx]


class FeedForwardNetwork(nn.Module):
    """Feed-forward neural network for scoring."""
    def __init__(self, input_dim, hidden_dims):
        """
        Initialize the network.
        
        Args:
            input_dim: Number of input features
            hidden_dims: List of hidden dimensions
        """
        super(FeedForwardNetwork, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        # Output layer with 1 output
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor (batch_size, n_stocks, n_features)
            
        Returns:
            Scores (batch_size, n_stocks, 1)
        """
        batch_size, n_stocks, n_features = x.shape
        
        # Reshape to (batch_size * n_stocks, n_features)
        x_reshaped = x.reshape(-1, n_features)
        
        # Pass through the network
        scores = self.network(x_reshaped)
        
        # Reshape back to (batch_size, n_stocks, 1)
        scores = scores.reshape(batch_size, n_stocks, 1)
        
        return scores


class ListMLELoss(nn.Module):
    """ListMLE loss function."""
    def __init__(self, transform='exp'):
        """
        Initialize the loss function.
        
        Args:
            transform: Transformation function ('exp' or 'sigmoid')
        """
        super(ListMLELoss, self).__init__()
        self.transform = transform
    
    def forward(self, scores, labels):
        """
        Forward pass of the loss function.
        
        Args:
            scores: Predicted scores (batch_size, n_stocks, 1)
            labels: Ground truth returns (batch_size, n_stocks)
            
        Returns:
            Loss value
        """
        batch_size, n_stocks, _ = scores.shape
        scores = scores.squeeze(-1)  # (batch_size, n_stocks)
        
        # Sort labels in descending order to get ground truth permutation
        sorted_labels, ground_truth = torch.sort(labels, dim=1, descending=True)
        
        loss = 0.0
        
        for i in range(batch_size):
            # Reorder scores based on ground truth
            ordered_scores = torch.gather(scores[i], 0, ground_truth[i])
            
            if self.transform == 'exp':
                transformed_scores = torch.exp(ordered_scores)
            elif self.transform == 'sigmoid':
                transformed_scores = torch.sigmoid(ordered_scores)
            else:
                transformed_scores = ordered_scores
            
            # Calculate ListMLE loss
            cum_sum = torch.cumsum(transformed_scores, dim=0)
            cum_sum_shifted = torch.cat([torch.zeros(1, device=scores.device), cum_sum[:-1]])
            log_probs = torch.log(transformed_scores) - torch.log(cum_sum - cum_sum_shifted)
            loss -= log_probs.sum()
        
        return loss / batch_size


class ListFoldLoss(nn.Module):
    """ListFold loss function."""
    def __init__(self, transform='exp'):
        """
        Initialize the loss function.
        
        Args:
            transform: Transformation function ('exp' or 'sigmoid')
        """
        super(ListFoldLoss, self).__init__()
        self.transform = transform
    
    def forward(self, scores, labels):
        """
        Forward pass of the loss function.
        
        Args:
            scores: Predicted scores (batch_size, n_stocks, 1)
            labels: Ground truth returns (batch_size, n_stocks)
            
        Returns:
            Loss value
        """
        batch_size, n_stocks, _ = scores.shape
        scores = scores.squeeze(-1)  # (batch_size, n_stocks)
        
        # Ensure even number of stocks
        if n_stocks % 2 != 0:
            scores = scores[:, :-1]
            labels = labels[:, :-1]
            n_stocks -= 1
        
        # Sort labels in descending order to get ground truth permutation
        sorted_labels, ground_truth = torch.sort(labels, dim=1, descending=True)
        
        loss = 0.0
        
        for i in range(batch_size):
            # Reorder scores based on ground truth
            ordered_scores = torch.gather(scores[i], 0, ground_truth[i])
            
            # ListFold loss calculation
            half_n = n_stocks // 2
            
            for j in range(half_n):
                # Get scores of the jth top and bottom pair
                top_score = ordered_scores[j]
                bottom_score = ordered_scores[n_stocks - 1 - j]
                score_diff = top_score - bottom_score
                
                # Calculate denominator (all possible score differences)
                denom = 0.0
                
                for u in range(j, half_n + 1):
                    for v in range(n_stocks - half_n + j - 1, n_stocks):
                        if u != v:
                            diff = ordered_scores[u] - ordered_scores[v]
                            if self.transform == 'exp':
                                denom += torch.exp(diff)
                            elif self.transform == 'sigmoid':
                                denom += torch.sigmoid(diff)
                            else:
                                denom += diff
                
                # Calculate the numerator
                if self.transform == 'exp':
                    numer = torch.exp(score_diff)
                elif self.transform == 'sigmoid':
                    numer = torch.sigmoid(score_diff)
                else:
                    numer = score_diff
                
                loss -= torch.log(numer / denom)
        
        return loss / batch_size


class MSELoss(nn.Module):
    """Mean Squared Error loss for MLP baseline."""
    def __init__(self):
        super(MSELoss, self).__init__()
        self.mse = nn.MSELoss()
    
    def forward(self, scores, labels):
        """
        Forward pass of the loss function.
        
        Args:
            scores: Predicted scores (batch_size, n_stocks, 1)
            labels: Ground truth returns (batch_size, n_stocks)
            
        Returns:
            Loss value
        """
        return self.mse(scores.squeeze(-1), labels)


class LongShortPortfolioConstructor:
    """Implements the ListFold algorithm for long-short portfolio construction."""
    
    def __init__(self):
        """Initialize the portfolio constructor."""
        self.data = None
        self.tickers = None
        self.factors = None
        self.returns = None
        self.models = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
    
    def fetch_data_from_bloomberg(self, 
                                tickers: List[str], 
                                start_date: dt.datetime,
                                end_date: dt.datetime,
                                factor_list: List[str]) -> None:
        """
        Fetch factor and return data from Bloomberg with improved handling for missing data.
        
        Args:
            tickers: List of stock tickers
            start_date: Start date for data
            end_date: End date for data
            factor_list: List of factor names to fetch
            
        Returns:
            Tuple of factors and returns data if successful, None otherwise
        """
        if not BLOOMBERG_AVAILABLE:
            logger.error("Bloomberg API not available. Cannot fetch data.")
            return None
        
        logger.info(f"Connecting to Bloomberg to fetch data for {len(tickers)} stocks...")
        
        # Store the tickers
        self.tickers = tickers
        
        try:
            # Initialize Bloomberg session
            session_options = blpapi.SessionOptions()
            session_options.setServerHost("localhost")
            session_options.setServerPort(8194)
            session = blpapi.Session(session_options)
            
            # Start session
            if not session.start():
                logger.error("Failed to start Bloomberg session")
                return None
                
            # Open reference data service
            if not session.openService("//blp/refdata"):
                logger.error("Failed to open //blp/refdata service")
                session.stop()
                return None
                
            refdata_service = session.getService("//blp/refdata")
            
            # Create a dictionary to store data for each ticker
            ticker_data = {}
            
            # First, fetch weekly price data directly (no need to resample)
            for ticker in tickers:
                logger.info(f"Fetching price data for {ticker}")
                
                # Create request for historical data
                request = refdata_service.createRequest("HistoricalDataRequest")
                request.getElement("securities").appendValue(ticker)
                
                # Add field for price
                request.getElement("fields").appendValue("PX_LAST")
                
                # Set date range - get WEEKLY data directly
                request.set("periodicitySelection", "WEEKLY")
                request.set("startDate", start_date.strftime("%Y%m%d"))
                request.set("endDate", end_date.strftime("%Y%m%d"))
                
                # Send request
                session.sendRequest(request)
                
                # Process response
                dates = []
                prices = []
                
                while True:
                    event = session.nextEvent(500)  # Timeout in milliseconds
                    
                    for msg in event:
                        if msg.messageType() == blpapi.Name("HistoricalDataResponse"):
                            security_data = msg.getElement("securityData")
                            field_data = security_data.getElement("fieldData")
                            
                            for j in range(field_data.numValues()):
                                field_value = field_data.getValue(j)
                                
                                # Get date
                                date_element = field_value.getElementAsDatetime("date")
                                if isinstance(date_element, dt.datetime):
                                    date = date_element.date()
                                else:
                                    date = date_element
                                
                                # Get price
                                if field_value.hasElement("PX_LAST"):
                                    price = field_value.getElementAsFloat("PX_LAST")
                                    dates.append(date)
                                    prices.append(price)
                    
                    if event.eventType() == blpapi.Event.RESPONSE:
                        break
                
                # Create a dataframe for this ticker's prices
                if dates and prices:
                    # Initialize data for this ticker
                    ticker_data[ticker] = {}
                    ticker_data[ticker]['dates'] = dates
                    ticker_data[ticker]['PX_LAST'] = prices
                    # Calculate returns directly
                    returns = [0]  # First return is 0
                    for i in range(1, len(prices)):
                        if prices[i-1] > 0:
                            ret = (prices[i] / prices[i-1]) - 1
                        else:
                            ret = 0
                        returns.append(ret)
                    ticker_data[ticker]['return'] = returns
                else:
                    logger.warning(f"No price data for {ticker}")
            
            # Now fetch factor data for each ticker
            for ticker in list(ticker_data.keys()):  # Use list to avoid changing dict during iteration
                logger.info(f"Fetching factor data for {ticker}")
                
                for factor in factor_list:
                    logger.info(f"Fetching {factor} for {ticker}")
                    
                    # Create request for historical data
                    request = refdata_service.createRequest("HistoricalDataRequest")
                    request.getElement("securities").appendValue(ticker)
                    
                    # Add field for factor
                    request.getElement("fields").appendValue(factor)
                    
                    # Set date range - WEEKLY to match the price data
                    request.set("periodicitySelection", "WEEKLY")
                    request.set("startDate", start_date.strftime("%Y%m%d"))
                    request.set("endDate", end_date.strftime("%Y%m%d"))
                    
                    # Send request
                    session.sendRequest(request)
                    
                    # Process response
                    factor_dates = []
                    factor_values = []
                    
                    while True:
                        event = session.nextEvent(500)  # Timeout in milliseconds
                        
                        for msg in event:
                            if msg.messageType() == blpapi.Name("HistoricalDataResponse"):
                                security_data = msg.getElement("securityData")
                                
                                # Check for security errors
                                if security_data.hasElement("securityError"):
                                    error = security_data.getElement("securityError")
                                    logger.warning(f"Security error for {ticker}: {error.getElementAsString('message')}")
                                    continue
                                
                                field_data = security_data.getElement("fieldData")
                                
                                for j in range(field_data.numValues()):
                                    field_value = field_data.getValue(j)
                                    
                                    # Get date
                                    date_element = field_value.getElementAsDatetime("date")
                                    if isinstance(date_element, dt.datetime):
                                        date = date_element.date()
                                    else:
                                        date = date_element
                                    
                                    # Get factor value
                                    if field_value.hasElement(factor):
                                        try:
                                            value = field_value.getElementAsFloat(factor)
                                            factor_dates.append(date)
                                            factor_values.append(value)
                                        except:
                                            # Some factors might not be numeric, skip them
                                            logger.warning(f"Could not parse {factor} as float for {ticker} on {date}")
                        
                        if event.eventType() == blpapi.Event.RESPONSE:
                            break
                    
                    # Store factor values in ticker data
                    if factor_dates and factor_values:
                        # Create a mapping from dates to factor values
                        date_to_factor = dict(zip(factor_dates, factor_values))
                        
                        # Map factor values to the same dates as the price data
                        ticker_factor_values = []
                        for date in ticker_data[ticker]['dates']:
                            if date in date_to_factor:
                                ticker_factor_values.append(date_to_factor[date])
                            else:
                                ticker_factor_values.append(None)  # Missing value
                        
                        ticker_data[ticker][factor] = ticker_factor_values
                        logger.info(f"Successfully added {factor} for {ticker}")
                    else:
                        logger.warning(f"No valid data for {factor} for {ticker}")
            
            # Close the Bloomberg session
            session.stop()
            
            # Check which factors are available for all tickers
            available_tickers = list(ticker_data.keys())
            if not available_tickers:
                logger.error("No valid ticker data retrieved")
                return None
            
            # Find the factors that are available for all tickers
            common_factors = set(factor_list)
            for ticker in available_tickers:
                available_factors = set(key for key in ticker_data[ticker].keys() 
                                      if key not in ['dates', 'PX_LAST', 'return'])
                common_factors &= available_factors
            
            common_factors = list(common_factors)
            if not common_factors:
                logger.error("No common factors across all tickers")
                return None
            
            logger.info(f"Using {len(common_factors)} common factors: {common_factors}")
            
            # Create unified arrays from the common factors and dates
            # First, find the common date range
            common_dates = set(ticker_data[available_tickers[0]]['dates'])
            for ticker in available_tickers[1:]:
                common_dates &= set(ticker_data[ticker]['dates'])
            
            common_dates = sorted(common_dates)
            if len(common_dates) < 10:
                logger.error(f"Only {len(common_dates)} common dates found, need at least 10")
                return None
            
            # Create the data arrays
            n_dates = len(common_dates)
            n_tickers = len(available_tickers)
            n_factors = len(common_factors)
            
            logger.info(f"Creating dataset with {n_dates} dates, {n_tickers} tickers, and {n_factors} factors")
            
            # Initialize arrays
            factors_array = np.zeros((n_dates, n_tickers, n_factors))
            returns_array = np.zeros((n_dates, n_tickers))
            
            # Fill the arrays
            for date_idx, date in enumerate(common_dates):
                for ticker_idx, ticker in enumerate(available_tickers):
                    # Get the index of this date in the ticker's data
                    ticker_date_idx = ticker_data[ticker]['dates'].index(date)
                    
                    # Fill return value
                    returns_array[date_idx, ticker_idx] = ticker_data[ticker]['return'][ticker_date_idx]
                    
                    # Fill factor values
                    for factor_idx, factor in enumerate(common_factors):
                        if factor in ticker_data[ticker] and ticker_data[ticker][factor][ticker_date_idx] is not None:
                            factors_array[date_idx, ticker_idx, factor_idx] = ticker_data[ticker][factor][ticker_date_idx]
            
            # Handle missing values in factors
            # Forward fill (each stock's factors persist until changed)
            for i in range(n_tickers):
                for j in range(n_factors):
                    mask = np.isnan(factors_array[:, i, j])
                    if mask.any():
                        # Get first valid index
                        valid_idx = np.where(~mask)[0]
                        if len(valid_idx) > 0:
                            first_valid = valid_idx[0]
                            # Forward fill
                            last_valid_value = factors_array[first_valid, i, j]
                            for k in range(first_valid+1, n_dates):
                                if mask[k]:
                                    factors_array[k, i, j] = last_valid_value
                                else:
                                    last_valid_value = factors_array[k, i, j]
            
            # Fill any remaining NaNs with column median for factors
            for j in range(n_factors):
                for i in range(n_tickers):
                    col_data = factors_array[:, i, j]
                    mask = np.isnan(col_data)
                    if mask.any():
                        col_median = np.nanmedian(factors_array[:, :, j])
                        factors_array[mask, i, j] = col_median if not np.isnan(col_median) else 0
            
            # Check if we have enough data
            if n_dates < 10:
                logger.error(f"Not enough data after processing: {n_dates} samples")
                return None
            
            logger.info(f"Final dataset: {n_dates} samples, {n_tickers} tickers, {n_factors} factors")
            
            # Update tickers to only include the ones we have data for
            self.tickers = available_tickers
            
            # Store the data
            self.factors = factors_array
            self.returns = returns_array
            
            return factors_array, returns_array
            
        except Exception as e:
            logger.error(f"Error fetching Bloomberg data: {e}")
            import traceback
            logger.error(traceback.format_exc())
            # Try to close the session if it exists
            if 'session' in locals():
                try:
                    session.stop()
                except:
                    pass
            return None
    
    def augment_bloomberg_data(self):
        """
        Augment the Bloomberg data to increase sample size.
        
        This method creates synthetic samples by slightly perturbing 
        the existing data, which can help increase the sample size
        for training when real data is limited.
        """
        if self.factors is None or self.returns is None:
            logger.error("No data loaded. Cannot augment.")
            return False
        
        logger.info(f"Augmenting data. Original shape: {self.factors.shape}")
        
        n_samples, n_stocks, n_factors = self.factors.shape
        
        # If we already have enough samples, don't augment
        if n_samples >= 100:
            logger.info("Already have sufficient samples. No augmentation needed.")
            return True
        
        # Create copies with small perturbations to factors
        # Start with the original data
        augmented_factors = [self.factors]
        augmented_returns = [self.returns]
        
        # Add noise to create synthetic samples
        n_copies = max(1, int(100 / n_samples))
        for i in range(n_copies):
            # Create a copy with small random perturbations to factors
            noise_scale = 0.01 * (i + 1)  # Scale noise with each copy
            factor_noise = np.random.normal(0, noise_scale, (n_samples, n_stocks, n_factors))
            new_factors = self.factors + factor_noise
            
            # Returns should be related to factors, but with some noise
            return_noise = np.random.normal(0, 0.002 * (i + 1), (n_samples, n_stocks))
            new_returns = self.returns + return_noise
            
            augmented_factors.append(new_factors)
            augmented_returns.append(new_returns)
        
        # Combine all samples
        self.factors = np.vstack(augmented_factors)
        self.returns = np.vstack(augmented_returns)
        
        logger.info(f"Augmented data. New shape: {self.factors.shape}")
        return True
    
    def generate_synthetic_data(self, 
                              n_stocks: int = 20, 
                              n_factors: int = 5, 
                              n_weeks: int = 200, 
                              seed: int = 42) -> None:
        """
        Generate synthetic data for testing or when real data is insufficient.
        
        Args:
            n_stocks: Number of stocks
            n_factors: Number of factors
            n_weeks: Number of weeks
            seed: Random seed
            
        Returns:
            Tuple of factors and returns data
        """
        np.random.seed(seed)
        
        # Generate synthetic tickers
        self.tickers = [f"STOCK{i}" for i in range(1, n_stocks + 1)]
        
        # Create factors data - start with random values
        factors_data = np.random.randn(n_weeks, n_stocks, n_factors)
        
        # Make factors more realistic by adding:
        # 1. Time series correlation (autocorrelation)
        # 2. Cross-sectional correlation (some stocks have similar factors)
        # 3. Different scales for different factors
        
        # Add autocorrelation
        for week in range(1, n_weeks):
            factors_data[week] = 0.8 * factors_data[week-1] + 0.2 * np.random.randn(n_stocks, n_factors)
        
        # Add cross-sectional correlation
        for factor in range(n_factors):
            # Add a common factor
            common = np.random.randn(n_weeks)
            for stock in range(n_stocks):
                # Mix of common factor and idiosyncratic component
                beta = 0.5 + 0.5 * np.random.rand()  # Random beta between 0.5 and 1.0
                factors_data[:, stock, factor] = beta * common + (1 - beta) * factors_data[:, stock, factor]
        
        # Scale factors differently
        for factor in range(n_factors):
            scale = 0.5 + 1.5 * np.random.rand()  # Random scale between 0.5 and 2.0
            factors_data[:, :, factor] *= scale
        
        # Generate returns with:
        # 1. Factor exposures (stocks respond to factors)
        # 2. Market component (all stocks move together somewhat)
        # 3. Idiosyncratic component (stock-specific moves)
        
        # Create factor coefficients
        factor_coefs = np.random.uniform(-0.01, 0.01, n_factors)
        
        # Create stock betas to market
        stock_betas = 0.8 + 0.4 * np.random.rand(n_stocks)
        
        # Generate market returns
        market_returns = 0.001 + 0.02 * np.random.randn(n_weeks)
        
        # Calculate returns
        returns_data = np.zeros((n_weeks, n_stocks))
        
        for week in range(n_weeks):
            for stock in range(n_stocks):
                # Factor component
                factor_return = np.dot(factors_data[week, stock], factor_coefs)
                
                # Market component
                market_return = stock_betas[stock] * market_returns[week]
                
                # Idiosyncratic component
                idiosyncratic = 0.02 * np.random.randn()
                
                # Total return
                returns_data[week, stock] = factor_return + market_return + idiosyncratic
        
        # Store the data
        self.factors = factors_data
        self.returns = returns_data
        
        logger.info(f"Generated synthetic data for {n_stocks} stocks over {n_weeks} weeks.")
        
        return factors_data, returns_data
    
    def train_models(self, 
                    training_window: int = 100, 
                    batch_size: int = 16, 
                    num_epochs: int = 5, 
                    learning_rate: float = 0.001,
                    hidden_dims: List[int] = [32, 16]) -> None:
        """
        Train the models with optimized settings.
        
        Args:
            training_window: Number of weeks for training
            batch_size: Batch size for training
            num_epochs: Number of epochs for training
            learning_rate: Learning rate for optimizer
            hidden_dims: Hidden dimensions for the network
        """
        # Check if data is loaded
        if self.factors is None or self.returns is None:
            logger.error("No data loaded. Cannot train models.")
            return
        
        # Get dimensions
        n_samples, n_stocks, n_factors = self.factors.shape
        
        # If we don't have enough data for the requested training window,
        # reduce it to fit what we have
        if n_samples <= training_window:
            logger.warning(f"Training window ({training_window}) is larger than available data ({n_samples}).")
            training_window = max(10, n_samples // 2)  # Use at least 10 samples, or half of available data
            logger.info(f"Adjusted training window to {training_window}")
        
        # Initialize models with simpler architecture
        input_dim = n_factors
        
        self.models = {
            "ListFold-exp": FeedForwardNetwork(input_dim, hidden_dims).to(self.device),
            "ListMLE": FeedForwardNetwork(input_dim, hidden_dims).to(self.device),
            "MLP": FeedForwardNetwork(input_dim, hidden_dims).to(self.device),
        }
        
        # Initialize loss functions
        loss_functions = {
            "ListFold-exp": ListFoldLoss(transform='exp'),
            "ListMLE": ListMLELoss(transform='exp'),
            "MLP": MSELoss()
        }
        
        # Determine test periods
        # If we have limited data, use fewer test periods
        if n_samples - training_window <= 10:
            test_week_indices = list(range(training_window, n_samples))
        else:
            test_week_indices = list(range(training_window, n_samples))
            step = max(1, len(test_week_indices) // 10)  # Use at most 10 test points
            test_week_indices = test_week_indices[::step]
        
        self.trained_models = {model_name: [] for model_name in self.models.keys()}
        
        # Train models for each test period
        for test_idx in tqdm(test_week_indices, desc="Training models"):
            # Get training data (previous training_window samples)
            train_indices = range(max(0, test_idx - training_window), test_idx)
            train_factors = self.factors[train_indices]
            train_returns = self.returns[train_indices]
            
            # Normalize factors
            scaler = MinMaxScaler()
            n_train, n_stocks, n_factors = train_factors.shape
            reshaped_factors = train_factors.reshape(-1, n_factors)
            normalized_factors = scaler.fit_transform(reshaped_factors).reshape(n_train, n_stocks, n_factors)
            
            # Create dataset and dataloader
            train_dataset = StockDataset(normalized_factors, train_returns)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            
            # Train each model
            for model_name, model in self.models.items():
                # Initialize a new model
                new_model = FeedForwardNetwork(input_dim, hidden_dims).to(self.device)
                
                # Train the model
                new_model.train()
                optimizer = optim.Adam(new_model.parameters(), lr=learning_rate)
                loss_fn = loss_functions[model_name]
                
                for epoch in range(num_epochs):
                    epoch_loss = 0.0
                    for batch_factors, batch_returns in train_loader:
                        batch_factors = batch_factors.to(self.device)
                        batch_returns = batch_returns.to(self.device)
                        
                        optimizer.zero_grad()
                        scores = new_model(batch_factors)
                        loss = loss_fn(scores, batch_returns)
                        loss.backward()
                        optimizer.step()
                        
                        epoch_loss += loss.item()
                    
                    # Log progress for first and last epoch
                    if epoch == 0 or epoch == num_epochs-1:
                        logger.info(f"Model: {model_name}, Test Period: {test_idx}, Epoch: {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")
                
                # Store the trained model
                self.trained_models[model_name].append(new_model)
        
        logger.info(f"Models trained successfully for {len(test_week_indices)} test periods")
        self.test_indices = test_week_indices
    
    def construct_portfolios(self, 
                           cutoff_percentage: float = 0.1) -> Dict:
        """
        Construct long-short portfolios based on the trained models.
        
        Args:
            cutoff_percentage: Percentage of stocks to long/short
            
        Returns:
            Dictionary with portfolio returns
        """
        # Check if models are trained
        if not hasattr(self, 'trained_models') or not self.trained_models:
            logger.error("No trained models. Call train_models first.")
            return None
        
        if not hasattr(self, 'test_indices') or not self.test_indices:
            logger.error("No test indices. Call train_models first.")
            return None
        
        # Get dimensions
        n_samples, n_stocks, n_factors = self.factors.shape
        
        # Initialize portfolio returns
        portfolio_returns = {
            f"{model_name}": np.zeros(len(self.test_indices))
            for model_name in self.trained_models.keys()
        }
        
        # Add shorting the average versions
        for model_name in list(self.trained_models.keys()):
            portfolio_returns[f"{model_name}-sa"] = np.zeros(len(self.test_indices))
        
        # Construct portfolios for each test period
        for i, test_idx in enumerate(tqdm(self.test_indices, desc="Constructing portfolios")):
            # Get test sample data
            test_factors = self.factors[test_idx]
            test_returns = self.returns[test_idx]
            
            # Normalize factors using the same scaler from training
            scaler = MinMaxScaler()
            train_indices = range(max(0, test_idx - 100), test_idx)  # Use up to 100 previous samples
            train_factors = self.factors[train_indices]
            reshaped_train_factors = train_factors.reshape(-1, n_factors)
            scaler.fit(reshaped_train_factors)
            
            normalized_test_factors = scaler.transform(test_factors.reshape(n_stocks, n_factors)).reshape(1, n_stocks, n_factors)
            normalized_test_factors = torch.FloatTensor(normalized_test_factors).to(self.device)
            
            # Get predictions from each model
            for model_name, models in self.trained_models.items():
                if i < len(models):  # Check if we have a model for this test index
                    model = models[i]
                    model.eval()
                    
                    with torch.no_grad():
                        scores = model(normalized_test_factors)
                        scores = scores.squeeze().cpu().numpy()
                        
                        # Create ranking
                        ranks = np.argsort(-scores)  # Descending
                        
                        # Get top stocks to long and bottom to short
                        n_stocks_to_trade = max(1, int(n_stocks * cutoff_percentage))
                        long_indices = ranks[:n_stocks_to_trade]
                        short_indices = ranks[-n_stocks_to_trade:]
                        
                        # Calculate portfolio return (equal weight)
                        long_return = np.mean(test_returns[long_indices])
                        short_return = np.mean(test_returns[short_indices])
                        portfolio_return = long_return - short_return
                        
                        portfolio_returns[model_name][i] = portfolio_return
                        
                        # Calculate portfolio return shorting the average
                        avg_return = np.mean(test_returns)
                        portfolio_return_sa = long_return - avg_return
                        
                        portfolio_returns[f"{model_name}-sa"][i] = portfolio_return_sa
        
        logger.info("Portfolios constructed successfully.")
        
        return portfolio_returns
    
    def evaluate_performance(self, portfolio_returns: Dict, risk_free_rate: float = 0.03, transaction_cost: float = 0.003) -> Dict:
        """
        Evaluate the performance of the portfolios.
        
        Args:
            portfolio_returns: Dictionary with portfolio returns
            risk_free_rate: Annualized risk-free rate
            transaction_cost: Transaction cost per trade (one-way)
            
        Returns:
            Dictionary with performance metrics
        """
        # Convert weekly risk-free rate
        weekly_rf = (1 + risk_free_rate) ** (1/52) - 1
        
        # Initialize performance metrics
        performance = {}
        
        for model_name, returns in portfolio_returns.items():
            # Skip if no returns
            if len(returns) == 0:
                logger.warning(f"No returns data for {model_name}")
                performance[model_name] = {
                    "Mean": float('nan'),
                    "Volatility": float('nan'),
                    "Sharpe Ratio": float('nan'),
                    "Max Drawdown": float('nan'),
                    "Cumulative Return": float('nan')
                }
                continue
            
            # Calculate mean return
            mean_return = np.mean(returns) * 52  # Annualized
            
            # Calculate volatility
            volatility = np.std(returns) * np.sqrt(52)  # Annualized
            
            # Calculate Sharpe ratio
            sharpe_ratio = (mean_return - risk_free_rate) / volatility if volatility > 0 else 0
            
            # Calculate maximum drawdown
            cumulative_returns = np.cumprod(1 + returns)
            peak = np.maximum.accumulate(cumulative_returns)
            drawdown = (cumulative_returns - peak) / peak
            max_drawdown = np.min(drawdown) if len(drawdown) > 0 else float('nan')
            
            # Store metrics
            performance[model_name] = {
                "Mean": mean_return,
                "Volatility": volatility,
                "Sharpe Ratio": sharpe_ratio,
                "Max Drawdown": max_drawdown,
                "Cumulative Return": cumulative_returns[-1] - 1 if len(cumulative_returns) > 0 else float('nan')
            }
        
        logger.info("Performance evaluation completed.")
        
        return performance
    
    def calculate_rank_metrics(self) -> Dict:
        """
        Calculate rank metrics (Spearman's ρ, NDCG, NDCG@ ± k).
        
        Returns:
            Dictionary with rank metrics
        """
        # Check if models are trained
        if not hasattr(self, 'trained_models') or not self.trained_models:
            logger.error("No trained models. Call train_models first.")
            return None
        
        if not hasattr(self, 'test_indices') or not self.test_indices:
            logger.error("No test indices. Call train_models first.")
            return None
        
        # Get dimensions
        n_samples, n_stocks, n_factors = self.factors.shape
        
        # Initialize rank metrics
        rank_metrics = {
            model_name: {
                "IC": [],
                "NDCG": [],
                "NDCG@k": [],
                "NDCG@-k": [],
                "NDCG@±k": []
            }
            for model_name in self.trained_models.keys()
        }
        
        # Define discount function for NDCG
        def get_discount(j):
            return 1 / np.log2(j + 2)
        
        # Calculate rank metrics for each test period
        for i, test_idx in enumerate(tqdm(self.test_indices, desc="Calculating rank metrics")):
            # Get test sample data
            test_factors = self.factors[test_idx]
            test_returns = self.returns[test_idx]
            
            # Normalize factors
            scaler = MinMaxScaler()
            train_indices = range(max(0, test_idx - 100), test_idx)  # Use up to 100 previous samples
            train_factors = self.factors[train_indices]
            reshaped_train_factors = train_factors.reshape(-1, n_factors)
            scaler.fit(reshaped_train_factors)
            
            normalized_test_factors = scaler.transform(test_factors.reshape(n_stocks, n_factors)).reshape(1, n_stocks, n_factors)
            normalized_test_factors = torch.FloatTensor(normalized_test_factors).to(self.device)
            
            # Get true ranking
            true_ranks = np.argsort(-test_returns)  # Descending
            
            # Convert returns to levels for NDCG (bucketing stocks into deciles)
            levels = np.zeros(n_stocks)
            for j in range(min(10, n_stocks)):
                start_idx = j * (n_stocks // min(10, n_stocks))
                end_idx = (j + 1) * (n_stocks // min(10, n_stocks)) if j < min(10, n_stocks) - 1 else n_stocks
                levels[true_ranks[start_idx:end_idx]] = min(10, n_stocks) - j
            
            # Calculate k for top/bottom stocks
            k = min(8, n_stocks // 4)  # Use at most 8 stocks, or 1/4 of available stocks
            
            # Get predictions from each model
            for model_name, models in self.trained_models.items():
                if i < len(models):  # Check if we have a model for this test index
                    model = models[i]
                    model.eval()
                    
                    with torch.no_grad():
                        scores = model(normalized_test_factors)
                        scores = scores.squeeze().cpu().numpy()
                        
                        # Create ranking
                        pred_ranks = np.argsort(-scores)  # Descending
                    
                        # Calculate Spearman's correlation (IC)
                        ic = spearmanr(np.argsort(true_ranks), np.argsort(pred_ranks))[0]
                        
                        # Calculate NDCG
                        pred_levels = levels[pred_ranks]
                        gain = 2 ** pred_levels - 1
                        
                        # Calculate ideal DCG
                        ideal_gain = 2 ** np.sort(levels)[::-1] - 1
                        ideal_dcg = np.sum(ideal_gain * np.array([get_discount(j) for j in range(n_stocks)]))
                        
                        # Calculate DCG
                        dcg = np.sum(gain * np.array([get_discount(j) for j in range(n_stocks)]))
                        
                        ndcg = dcg / ideal_dcg if ideal_dcg > 0 else 0
                        
                        # Calculate NDCG@k
                        dcg_k = np.sum(gain[:k] * np.array([get_discount(j) for j in range(k)]))
                        ideal_dcg_k = np.sum(ideal_gain[:k] * np.array([get_discount(j) for j in range(k)]))
                        ndcg_k = dcg_k / ideal_dcg_k if ideal_dcg_k > 0 else 0
                        
                        # Calculate NDCG@-k
                        reversed_pred_ranks = pred_ranks[::-1]
                        reversed_levels = levels[reversed_pred_ranks]
                        reversed_gain = 2 ** reversed_levels - 1
                        
                        reversed_dcg_k = np.sum(reversed_gain[:k] * np.array([get_discount(j) for j in range(k)]))
                        reversed_ideal_dcg_k = np.sum(ideal_gain[:k] * np.array([get_discount(j) for j in range(k)]))
                        reversed_ndcg_k = reversed_dcg_k / reversed_ideal_dcg_k if reversed_ideal_dcg_k > 0 else 0
                        
                        # Calculate NDCG@±k
                        ndcg_pm_k = (ndcg_k + reversed_ndcg_k) / 2
                        
                        # Store metrics
                        rank_metrics[model_name]["IC"].append(ic)
                        rank_metrics[model_name]["NDCG"].append(ndcg)
                        rank_metrics[model_name]["NDCG@k"].append(ndcg_k)
                        rank_metrics[model_name]["NDCG@-k"].append(reversed_ndcg_k)
                        rank_metrics[model_name]["NDCG@±k"].append(ndcg_pm_k)
        
        # Calculate average metrics
        for model_name in rank_metrics.keys():
            for metric in rank_metrics[model_name].keys():
                if rank_metrics[model_name][metric]:  # Check if list is not empty
                    rank_metrics[model_name][metric] = np.mean(rank_metrics[model_name][metric])
                else:
                    rank_metrics[model_name][metric] = float('nan')
        
        logger.info("Rank metrics calculated successfully.")
        
        return rank_metrics
    
    def plot_results(self, portfolio_returns: Dict) -> None:
        """
        Plot the results of the portfolios.
        
        Args:
            portfolio_returns: Dictionary with portfolio returns
        """
        # Check if we have any data
        if not portfolio_returns or all(len(returns) == 0 for returns in portfolio_returns.values()):
            logger.error("No portfolio returns data to plot")
            return
        
        # Filter out empty return series
        portfolio_returns = {k: v for k, v in portfolio_returns.items() if len(v) > 0}
        
        # Calculate cumulative returns
        cumulative_returns = {
            model_name: np.cumprod(1 + returns)
            for model_name, returns in portfolio_returns.items()
        }
        
        # Plot cumulative returns
        plt.figure(figsize=(12, 8))
        
        # Separate main strategies and SA strategies
        main_strategies = {}
        sa_strategies = {}
        
        for model_name, cum_returns in cumulative_returns.items():
            if "-sa" in model_name:
                sa_strategies[model_name] = cum_returns
            else:
                main_strategies[model_name] = cum_returns
        
        # Plot main strategies
        plt.figure(figsize=(12, 8))
        for model_name, cum_returns in main_strategies.items():
            plt.plot(cum_returns, label=model_name)
        
        plt.grid(True)
        plt.xlabel("Test Period")
        plt.ylabel("Cumulative Returns")
        plt.title("Long-Short Portfolio Performance")
        plt.legend()
        plt.savefig("portfolio_performance.png")
        plt.close()
        
        # Plot SA strategies if any
        if sa_strategies:
            plt.figure(figsize=(12, 8))
            for model_name, cum_returns in sa_strategies.items():
                plt.plot(cum_returns, label=model_name)
            
            plt.grid(True)
            plt.xlabel("Test Period")
            plt.ylabel("Cumulative Returns")
            plt.title("Long-Short (Shorting Average) Portfolio Performance")
            plt.legend()
            plt.savefig("portfolio_performance_sa.png")
            plt.close()
        
        # Plot returns distribution
        plt.figure(figsize=(12, 8))
        
        for model_name, returns in portfolio_returns.items():
            if "-sa" not in model_name:  # Only plot main strategies
                sns.kdeplot(returns, label=model_name)
        
        plt.grid(True)
        plt.xlabel("Return")
        plt.ylabel("Density")
        plt.title("Return Distribution by Strategy")
        plt.legend()
        plt.savefig("return_distribution.png")
        plt.close()
        
        # Plot returns over time
        plt.figure(figsize=(15, 10))
        
        for model_name, returns in portfolio_returns.items():
            if "-sa" not in model_name:  # Only plot main strategies
                plt.plot(returns, label=model_name, alpha=0.7)
        
        plt.grid(True)
        plt.xlabel("Test Period")
        plt.ylabel("Return")
        plt.title("Portfolio Returns Over Time")
        plt.legend()
        plt.savefig("returns_over_time.png")
        plt.close()
        
        logger.info("Results plotted successfully.")


def main():
    """Main function to run the ListFold strategy."""
    # Initialize the portfolio constructor
    portfolio = LongShortPortfolioConstructor()
    
    # Use Bloomberg data first with improved handling
    data_loaded = False
    
    if BLOOMBERG_AVAILABLE:
        try:
            logger.info("Attempting to load Bloomberg data...")
            
            # Define tickers
            tickers = [
                "AAPL US Equity",
                "MSFT US Equity",
                "AMZN US Equity", 
                "GOOGL US Equity",
                "META US Equity",
                "NVDA US Equity",
                "BRK/B US Equity",
                "JPM US Equity",
                "JNJ US Equity",
                "V US Equity",
                "PG US Equity",
                "XOM US Equity",
                "BAC US Equity",
                "UNH US Equity",
                "TSLA US Equity",
                "HD US Equity",
                "MA US Equity",
                "CVX US Equity",
                "AVGO US Equity",
                "LLY US Equity",
            ]
            
            # Define factor list - simpler list focusing on commonly available factors
            factors = [
                "PX_TO_BOOK_RATIO",      # Price/Book ratio
                "PE_RATIO",              # Price/Earnings ratio
                "RETURN_COM_EQY",        # Return on Equity
                "RETURN_ON_ASSET",       # Return on Assets
                "CURR_RATIO",            # Current Ratio
                "QUICK_RATIO",           # Quick Ratio
                "TOT_DEBT_TO_TOT_ASSET", # Debt to Assets
                "SALES_GROWTH",          # Sales growth
                "EPS_GROWTH",            # Earnings per share growth
                "EBITDA_GROWTH",         # EBITDA growth
            ]
            
            # Fetch data (using a longer period for more data points)
            start_date = dt.datetime(2020, 1, 1)  # Start from 2020
            end_date = dt.datetime(2025, 3, 1)    # Up to near present
            
            result = portfolio.fetch_data_from_bloomberg(tickers, start_date, end_date, factors)
            
            if result is not None:
                data_loaded = True
                logger.info(f"Successfully loaded Bloomberg data with shape {portfolio.factors.shape}")
                
                # If we have limited data, augment it
                if portfolio.factors.shape[0] < 50:
                    logger.info("Limited data available. Augmenting dataset...")
                    portfolio.augment_bloomberg_data()
            else:
                logger.warning("Failed to load Bloomberg data.")
        except Exception as e:
            logger.error(f"Error loading Bloomberg data: {e}")
    
    # If Bloomberg data failed, use synthetic data
    if not data_loaded:
        logger.info("Using synthetic data instead...")
        portfolio.generate_synthetic_data(n_stocks=20, n_factors=10, n_weeks=200)
    
    # Get data dimensions
    n_samples, n_stocks, n_factors = portfolio.factors.shape
    logger.info(f"Using dataset with {n_samples} samples, {n_stocks} stocks, and {n_factors} factors")
    
    # Determine training window based on available data
    training_window = min(100, max(10, n_samples // 2))
    logger.info(f"Using training window of {training_window} samples")
    
    # Train models
    portfolio.train_models(
        training_window=training_window,
        batch_size=16,
        num_epochs=5,
        learning_rate=0.001,
        hidden_dims=[32, 16]
    )
    
    # Construct portfolios
    logger.info("Constructing portfolios...")
    portfolio_returns = portfolio.construct_portfolios(cutoff_percentage=0.1)
    
    # Evaluate performance
    logger.info("Evaluating performance...")
    performance = portfolio.evaluate_performance(
        portfolio_returns=portfolio_returns,
        risk_free_rate=0.03,
        transaction_cost=0.003
    )
    
    # Calculate rank metrics
    logger.info("Calculating rank metrics...")
    rank_metrics = portfolio.calculate_rank_metrics()
    
    # Plot results
    logger.info("Plotting results...")
    portfolio.plot_results(portfolio_returns)
    
    # Print performance metrics
    print("\nPerformance Metrics:")
    for model_name, metrics in performance.items():
        print(f"\n{model_name}:")
        for metric_name, value in metrics.items():
            print(f"  {metric_name}: {value:.4f}")
    
    # Print rank metrics
    if rank_metrics:
        print("\nRank Metrics:")
        for model_name, metrics in rank_metrics.items():
            print(f"\n{model_name}:")
            for metric_name, value in metrics.items():
                print(f"  {metric_name}: {value:.4f}")
    
    logger.info("Analysis completed successfully.")


if __name__ == "__main__":
    main()