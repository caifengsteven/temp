import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import statsmodels.api as sm
from statsmodels.tsa.api import VAR
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import warnings
import seaborn as sns
from sklearn.model_selection import train_test_split
import math
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Set matplotlib parameters for better visualizations
plt.style.use('seaborn-whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

#----------------------------------------
# Data Generation Functions
#----------------------------------------

def generate_oil_price_process(n_days=1000, start_price=100, volatility=0.02, mean_reversion=0.02, 
                              long_term_mean=100, seasonality=True, jumps=True):
    """
    Generate a simulated crude oil price process with mean reversion, seasonality, and jumps.
    
    Parameters:
    -----------
    n_days : int
        Number of days to simulate
    start_price : float
        Starting price
    volatility : float
        Volatility parameter
    mean_reversion : float
        Mean reversion speed parameter
    long_term_mean : float
        Long-term mean price
    seasonality : bool
        Whether to add seasonal component
    jumps : bool
        Whether to add price jumps
    
    Returns:
    --------
    prices : numpy array
        Simulated daily price series
    """
    prices = np.zeros(n_days)
    prices[0] = start_price
    
    # Generate the price process
    for t in range(1, n_days):
        # Mean reversion component
        drift = mean_reversion * (long_term_mean - prices[t-1])
        
        # Seasonal component (if enabled)
        seasonal = 0
        if seasonality:
            # Annual seasonality pattern
            seasonal = 5 * np.sin(2 * np.pi * t / 252)
        
        # Random shock
        shock = volatility * prices[t-1] * np.random.normal(0, 1)
        
        # Price jump component (if enabled)
        jump = 0
        if jumps and np.random.random() < 0.005:  # 0.5% chance of a jump on any day
            jump = np.random.normal(0, 0.1) * prices[t-1]
        
        # Combine all components
        prices[t] = prices[t-1] + drift + seasonal + shock + jump
        
        # Ensure prices don't go negative
        if prices[t] < 1:
            prices[t] = 1
    
    return prices

def generate_term_structure(spot_prices, days_to_maturity=[30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 360,
                                                           390, 420, 450, 480, 510, 540, 570, 600, 630, 660, 690, 720],
                           convenience_yield=0.01, storage_cost=0.005, risk_premium=0.02):
    """
    Generate futures term structure based on spot prices and days to maturity.
    
    Parameters:
    -----------
    spot_prices : numpy array
        Simulated spot price series
    days_to_maturity : list
        Days to maturity for different futures contracts
    convenience_yield : float
        Convenience yield parameter
    storage_cost : float
        Storage cost parameter
    risk_premium : float
        Risk premium parameter
    
    Returns:
    --------
    futures_data : pandas DataFrame
        DataFrame containing futures prices for each maturity
    """
    n_days = len(spot_prices)
    n_maturities = len(days_to_maturity)
    
    # Initialize futures prices matrix
    futures_prices = np.zeros((n_days, n_maturities))
    
    # Convert days to maturity to years
    years_to_maturity = np.array(days_to_maturity) / 252
    
    # Generate term structure for each day
    for t in range(n_days):
        spot = spot_prices[t]
        
        # Calculate backwardation/contango pattern based on historical patterns
        # When prices are high, market tends to be in backwardation
        is_high_price = spot > np.mean(spot_prices[:t+1]) if t > 30 else True
        
        # Adjust convenience yield based on current price level
        cy = convenience_yield * 1.5 if is_high_price else convenience_yield * 0.5
        
        # Generate futures prices for each maturity
        for m in range(n_maturities):
            # Calculate futures price based on spot price and time to maturity
            # Incorporating convenience yield, storage costs, and risk premium
            time = years_to_maturity[m]
            
            # Use a simplified version of the futures pricing formula
            # If in backwardation (high prices), futures lower than spot
            # If in contango (low prices), futures higher than spot
            if is_high_price:
                # Backwardation: Futures price < Spot price
                futures_prices[t, m] = spot * np.exp((-cy + risk_premium) * time)
            else:
                # Contango: Futures price > Spot price
                futures_prices[t, m] = spot * np.exp((storage_cost + risk_premium) * time)
                
            # Add some random noise to make it more realistic
            futures_prices[t, m] *= (1 + np.random.normal(0, 0.005))
    
    # Create DataFrame
    dates = [datetime.now() - timedelta(days=n_days-i-1) for i in range(n_days)]
    futures_data = pd.DataFrame(futures_prices, index=dates, columns=days_to_maturity)
    
    return futures_data

#----------------------------------------
# Dynamic Nelson-Siegel Model Implementation
#----------------------------------------

class DynamicNelsonSiegel:
    """
    Implementation of the Dynamic Nelson-Siegel model for modeling term structure.
    """
    def __init__(self, lambda_fixed=None):
        """
        Initialize the DNS model.
        
        Parameters:
        -----------
        lambda_fixed : float or None
            Fixed lambda parameter. If None, lambda will be optimized.
        """
        self.lambda_fixed = lambda_fixed
        self.lambda_opt = None
        self.beta0 = None
        self.beta1 = None
        self.beta2 = None
    
    def _nelson_siegel_curve(self, beta0, beta1, beta2, lambda_val, tau):
        """
        Calculate the Nelson-Siegel curve for given parameters.
        
        Parameters:
        -----------
        beta0, beta1, beta2 : float
            Factor loadings
        lambda_val : float
            Decay parameter
        tau : numpy array
            Time to maturity
        
        Returns:
        --------
        y : numpy array
            Fitted values
        """
        # Calculate the Nelson-Siegel curve components
        exp_term = np.exp(-lambda_val * tau)
        term1 = 1.0
        term2 = (1 - exp_term) / (lambda_val * tau)
        term3 = term2 - exp_term
        
        # Combine components with factor loadings
        y = beta0 * term1 + beta1 * term2 + beta2 * term3
        
        return y
    
    def _estimate_betas(self, y, tau, lambda_val):
        """
        Estimate beta parameters for a given lambda value using OLS.
        
        Parameters:
        -----------
        y : numpy array
            Observed futures prices
        tau : numpy array
            Time to maturity
        lambda_val : float
            Decay parameter
        
        Returns:
        --------
        betas : numpy array
            Estimated beta parameters
        sse : float
            Sum of squared errors
        """
        # Calculate the Nelson-Siegel factor loadings
        exp_term = np.exp(-lambda_val * tau)
        term1 = np.ones_like(tau)
        term2 = (1 - exp_term) / (lambda_val * tau)
        term3 = term2 - exp_term
        
        # Create design matrix
        X = np.column_stack((term1, term2, term3))
        
        # Estimate betas using OLS
        betas, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        
        # Calculate fitted values
        y_fit = self._nelson_siegel_curve(betas[0], betas[1], betas[2], lambda_val, tau)
        
        # Calculate sum of squared errors
        sse = np.sum((y - y_fit) ** 2)
        
        return betas, sse
    
    def _objective_function(self, lambda_val, y, tau):
        """
        Objective function for lambda optimization.
        
        Parameters:
        -----------
        lambda_val : float
            Decay parameter
        y : numpy array
            Observed futures prices
        tau : numpy array
            Time to maturity
        
        Returns:
        --------
        sse : float
            Sum of squared errors
        """
        lambda_val = max(1e-10, lambda_val[0])  # Ensure lambda is positive
        _, sse = self._estimate_betas(y, tau, lambda_val)
        return sse
    
    def fit(self, futures_data):
        """
        Fit the Dynamic Nelson-Siegel model to the futures data.
        
        Parameters:
        -----------
        futures_data : pandas DataFrame
            DataFrame containing futures prices for each maturity
        
        Returns:
        --------
        factors : pandas DataFrame
            DataFrame containing the estimated factors
        """
        n_obs = len(futures_data)
        maturities = np.array([float(col) for col in futures_data.columns])
        
        # Initialize factor arrays
        self.beta0 = np.zeros(n_obs)
        self.beta1 = np.zeros(n_obs)
        self.beta2 = np.zeros(n_obs)
        
        # Optimize lambda if not fixed
        if self.lambda_fixed is None:
            # Use first day's data to find optimal lambda
            y = futures_data.iloc[0].values
            result = minimize(self._objective_function, x0=[0.05], args=(y, maturities), 
                             bounds=[(1e-6, 1.0)], method='L-BFGS-B')
            self.lambda_opt = result.x[0]
        else:
            self.lambda_opt = self.lambda_fixed
        
        # Estimate time-varying factors for each observation
        for t in range(n_obs):
            y = futures_data.iloc[t].values
            betas, _ = self._estimate_betas(y, maturities, self.lambda_opt)
            self.beta0[t] = betas[0]
            self.beta1[t] = betas[1]
            self.beta2[t] = betas[2]
        
        # Create factors DataFrame
        factors = pd.DataFrame({
            'level': self.beta0,
            'slope': self.beta1,
            'curvature': self.beta2
        }, index=futures_data.index)
        
        return factors
    
    def predict(self, factors, maturities):
        """
        Predict futures prices using estimated factors.
        
        Parameters:
        -----------
        factors : pandas DataFrame
            DataFrame containing the factors
        maturities : numpy array
            Time to maturity
        
        Returns:
        --------
        predicted : pandas DataFrame
            Predicted futures prices
        """
        n_obs = len(factors)
        n_maturities = len(maturities)
        
        # Initialize predictions array
        predictions = np.zeros((n_obs, n_maturities))
        
        # Generate predictions for each observation
        for t in range(n_obs):
            beta0 = factors.iloc[t]['level']
            beta1 = factors.iloc[t]['slope']
            beta2 = factors.iloc[t]['curvature']
            
            predictions[t] = self._nelson_siegel_curve(beta0, beta1, beta2, self.lambda_opt, maturities)
        
        # Create predictions DataFrame
        predicted = pd.DataFrame(predictions, index=factors.index, columns=maturities)
        
        return predicted

#----------------------------------------
# PyTorch Model Implementation
#----------------------------------------

class TimeSeriesDataset(Dataset):
    """
    Dataset for time series with lags.
    """
    def __init__(self, X, y, num_lags=1):
        self.X = X
        self.y = y
        self.num_lags = num_lags
        
    def __len__(self):
        return len(self.X) - self.num_lags
    
    def __getitem__(self, idx):
        # Create time-lagged feature tensor
        X_lagged = torch.zeros(self.num_lags, self.X.shape[1])
        for lag in range(self.num_lags):
            X_lagged[lag] = self.X[idx + self.num_lags - lag - 1]
        
        return X_lagged, self.y[idx + self.num_lags]


class FocusedTimeDelayNN(nn.Module):
    """
    Focused Time-Delay Neural Network implemented in PyTorch.
    """
    def __init__(self, input_dim, num_delays=1, hidden_sizes=[32, 16], activation='relu'):
        super(FocusedTimeDelayNN, self).__init__()
        
        self.input_dim = input_dim
        self.num_delays = num_delays
        
        # Select activation function
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            self.activation = nn.ReLU()
            
        # Flatten layer for time delay inputs
        self.flatten = nn.Flatten()
        
        # Create hidden layers
        layers = []
        input_size = input_dim * num_delays
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(self.activation)
            layers.append(nn.Dropout(0.2))  # Add dropout for regularization
            input_size = hidden_size
            
        # Output layer
        layers.append(nn.Linear(input_size, 1))
        
        # Combine all layers
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        x = self.flatten(x)
        return self.network(x)


class PyTorchFTDNN:
    """
    Wrapper class for the PyTorch FTDNN model.
    """
    def __init__(self, input_dim, num_delays=1, hidden_sizes=[32, 16], activation='relu',
                learning_rate=0.001, batch_size=32, epochs=100):
        """
        Initialize the PyTorch FTDNN model.
        
        Parameters:
        -----------
        input_dim : int
            Input dimension (number of features)
        num_delays : int
            Number of time delays
        hidden_sizes : list
            List of hidden layer sizes
        activation : str
            Activation function
        learning_rate : float
            Learning rate for optimizer
        batch_size : int
            Batch size for training
        epochs : int
            Maximum number of training epochs
        """
        self.input_dim = input_dim
        self.num_delays = num_delays
        self.hidden_sizes = hidden_sizes
        self.activation = activation
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        
        # Create the model
        self.model = FocusedTimeDelayNN(input_dim, num_delays, hidden_sizes, activation).to(device)
        
        # Define loss function and optimizer
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Scaling
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        
    def fit(self, X, y, validation_split=0.2, patience=10, verbose=1):
        """
        Fit the model to the data.
        
        Parameters:
        -----------
        X : numpy array
            Feature matrix
        y : numpy array
            Target values
        validation_split : float
            Fraction of data to use for validation
        patience : int
            Early stopping patience
        verbose : int
            Verbosity level
        
        Returns:
        --------
        history : dict
            Training history
        """
        # Scale the data
        X_scaled = self.scaler_X.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
        
        # Convert to PyTorch tensors
        X_tensor = torch.FloatTensor(X_scaled)
        y_tensor = torch.FloatTensor(y_scaled)
        
        # Split into train and validation sets
        val_size = int(len(X) * validation_split)
        train_size = len(X) - val_size
        
        X_train, X_val = X_tensor[:train_size], X_tensor[train_size:]
        y_train, y_val = y_tensor[:train_size], y_tensor[train_size:]
        
        # Create datasets and data loaders
        train_dataset = TimeSeriesDataset(X_train, y_train, self.num_delays)
        val_dataset = TimeSeriesDataset(X_val, y_val, self.num_delays)
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size)
        
        # Training history
        history = {
            'train_loss': [],
            'val_loss': []
        }
        
        # Early stopping variables
        best_val_loss = float('inf')
        best_epoch = 0
        best_model = None
        
        # Training loop
        for epoch in range(self.epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                
                # Forward pass
                self.optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = self.criterion(outputs.squeeze(), y_batch)
                
                # Backward pass and optimize
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item() * X_batch.size(0)
            
            # Calculate average training loss
            train_loss = train_loss / len(train_loader.dataset)
            history['train_loss'].append(train_loss)
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    outputs = self.model(X_batch)
                    loss = self.criterion(outputs.squeeze(), y_batch)
                    val_loss += loss.item() * X_batch.size(0)
            
            # Calculate average validation loss
            val_loss = val_loss / len(val_loader.dataset)
            history['val_loss'].append(val_loss)
            
            # Print progress
            if verbose and (epoch % 10 == 0 or epoch == self.epochs - 1):
                print(f'Epoch [{epoch+1}/{self.epochs}] - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
            
            # Check for improvement
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                best_model = self.model.state_dict().copy()
            
            # Early stopping
            if epoch - best_epoch >= patience:
                if verbose:
                    print(f'Early stopping at epoch {epoch+1}')
                break
        
        # Load the best model
        if best_model is not None:
            self.model.load_state_dict(best_model)
        
        return history
    
    def predict(self, X, h=1):
        """
        Make h-step ahead predictions.
        
        Parameters:
        -----------
        X : numpy array
            Feature matrix
        h : int
            Forecast horizon
        
        Returns:
        --------
        y_pred : numpy array
            Predicted values
        """
        # Scale the input data
        X_scaled = self.scaler_X.transform(X)
        
        # Convert to PyTorch tensor
        X_tensor = torch.FloatTensor(X_scaled)
        
        # Create dataset for prediction
        if len(X) >= self.num_delays:
            # Set the model to evaluation mode
            self.model.eval()
            
            # Process the last `num_delays` observations
            inputs = X_tensor[-self.num_delays:].unsqueeze(0).to(device)
            
            # Make prediction
            with torch.no_grad():
                output = self.model(inputs)
                
            # Inverse transform the prediction
            y_pred_scaled = output.cpu().numpy()
            y_pred = self.scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
            
            return y_pred
        else:
            raise ValueError(f"Input data must have at least {self.num_delays} observations")


class AR1Model:
    """
    Implementation of AR(1) model for factor prediction.
    """
    def __init__(self):
        """
        Initialize the AR(1) model.
        """
        self.model = None
        
    def fit(self, y):
        """
        Fit the AR(1) model to the data.
        
        Parameters:
        -----------
        y : numpy array
            Time series data
        
        Returns:
        --------
        self : object
            Returns self
        """
        # Fit AR(1) model
        self.model = sm.tsa.AutoReg(y, lags=1).fit()
        
        return self
    
    def predict(self, y, h=1):
        """
        Make h-step ahead predictions.
        
        Parameters:
        -----------
        y : numpy array
            Time series data
        h : int
            Forecast horizon
        
        Returns:
        --------
        y_pred : numpy array
            Predicted values
        """
        # Make predictions
        y_pred = self.model.predict(start=len(y), end=len(y) + h - 1)
        
        return y_pred


class VARModel:
    """
    Implementation of VAR(1) model for factor prediction.
    """
    def __init__(self):
        """
        Initialize the VAR(1) model.
        """
        self.model = None
        
    def fit(self, y):
        """
        Fit the VAR(1) model to the data.
        
        Parameters:
        -----------
        y : numpy array
            Multivariate time series data
        
        Returns:
        --------
        self : object
            Returns self
        """
        # Fit VAR(1) model
        try:
            self.model = VAR(y).fit(maxlags=1, ic=None)
        except:
            # Fallback to each column as separate AR(1) if VAR fails
            print("Warning: VAR model fitting failed, using AR(1) for each factor")
            self.model = "AR_fallback"
            self.ar_models = []
            for i in range(y.shape[1]):
                model = sm.tsa.AutoReg(y[:, i], lags=1).fit()
                self.ar_models.append(model)
        
        return self
    
    def predict(self, y, h=1):
        """
        Make h-step ahead predictions.
        
        Parameters:
        -----------
        y : numpy array or pandas DataFrame
            Multivariate time series data
        h : int
            Forecast horizon
        
        Returns:
        --------
        y_pred : numpy array
            Predicted values
        """
        # Make predictions
        if self.model == "AR_fallback":
            y_pred = np.zeros((h, y.shape[1]))
            for i in range(y.shape[1]):
                y_pred[:, i] = self.ar_models[i].predict(start=len(y), end=len(y) + h - 1)
        else:
            # Check if input is numpy array or pandas DataFrame
            if isinstance(y, np.ndarray):
                lag_data = y[-self.model.k_ar:]
            else:
                lag_data = y.values[-self.model.k_ar:]
                
            y_pred = self.model.forecast(lag_data, h)
        
        return y_pred


class RandomWalkModel:
    """
    Implementation of Random Walk model for factor prediction.
    """
    def __init__(self):
        """
        Initialize the Random Walk model.
        """
        pass
        
    def fit(self, y):
        """
        Fit the Random Walk model to the data.
        
        Parameters:
        -----------
        y : numpy array
            Time series data
        
        Returns:
        --------
        self : object
            Returns self
        """
        # No fitting required for Random Walk
        return self
    
    def predict(self, y, h=1):
        """
        Make h-step ahead predictions.
        
        Parameters:
        -----------
        y : numpy array
            Time series data
        h : int
            Forecast horizon
        
        Returns:
        --------
        y_pred : numpy array
            Predicted values
        """
        # Make predictions
        if len(y.shape) == 1:
            # Single time series
            y_pred = np.repeat(y[-1], h)
        else:
            # Multiple time series
            y_pred = np.tile(y[-1], (h, 1))
        
        return y_pred

#----------------------------------------
# Forecast Evaluation Functions
#----------------------------------------

def calculate_forecast_errors(actual, predicted):
    """
    Calculate forecast error metrics.
    
    Parameters:
    -----------
    actual : numpy array
        Actual values
    predicted : numpy array
        Predicted values
    
    Returns:
    --------
    metrics : dict
        Dictionary of error metrics
    """
    # Calculate error metrics
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mae = mean_absolute_error(actual, predicted)
    
    # Calculate mean mixed errors
    errors = actual - predicted
    under_pred = errors > 0
    over_pred = errors <= 0
    
    # MME(U) - gives more weight to under-predictions
    mme_u = np.mean(np.sqrt(np.abs(errors[under_pred]))) + np.mean(np.abs(errors[over_pred])) if np.any(under_pred) else np.mean(np.abs(errors))
    
    # MME(O) - gives more weight to over-predictions
    mme_o = np.mean(np.abs(errors[under_pred])) + np.mean(np.sqrt(np.abs(errors[over_pred]))) if np.any(over_pred) else np.mean(np.abs(errors))
    
    # Calculate relative metrics
    mape = np.mean(np.abs(errors / actual)) * 100
    
    return {
        'RMSE': rmse,
        'MAE': mae,
        'MME(U)': mme_u,
        'MME(O)': mme_o,
        'MAPE': mape
    }

def forecast_factors(factors, h=1, test_size=0.2, model_types=['FTDNN', 'AR1', 'VAR1', 'RW']):
    """
    Forecast factors using different models.
    
    Parameters:
    -----------
    factors : pandas DataFrame
        DataFrame containing the factors
    h : int
        Forecast horizon
    test_size : float
        Fraction of data to use for testing
    model_types : list
        List of models to use for forecasting
    
    Returns:
    --------
    forecasts : dict
        Dictionary of forecasts and evaluation metrics
    """
    # Split the data into training and testing sets
    n_test = int(len(factors) * test_size)
    n_train = len(factors) - n_test
    
    # Initialize forecasts dictionary
    forecasts = {}
    
    # Train and forecast for each model type
    for model_type in model_types:
        print(f"Forecasting with {model_type}...")
        
        # Initialize forecasts
        factor_forecasts = np.zeros((n_test, 3))
        
        if model_type == 'FTDNN':
            # Forecast each factor separately using PyTorch FTDNN
            for i, factor in enumerate(['level', 'slope', 'curvature']):
                # Get the factor data
                factor_data = factors[factor].values
                
                # Prepare input features (use all factors as inputs)
                X = factors.values
                y = factor_data
                
                # Split into training and testing
                X_train = X[:n_train]
                y_train = y[:n_train]
                X_test = X[n_train-1:n_train+n_test-1]  # Need the last training point for the first test prediction
                
                # Train the model
                model = PyTorchFTDNN(input_dim=3, num_delays=1, hidden_sizes=[32, 16], 
                                   learning_rate=0.001, batch_size=32, epochs=100)
                model.fit(X_train, y_train, validation_split=0.2, patience=10, verbose=0)
                
                # Make predictions
                for j in range(n_test):
                    # Get the data up to the current point
                    X_current = X[:(n_train + j)]
                    
                    # Make a one-step ahead prediction
                    prediction = model.predict(X_current[-model.num_delays:], h=1)
                    factor_forecasts[j, i] = prediction
                    
        elif model_type == 'AR1':
            # Forecast each factor separately using AR(1)
            for i, factor in enumerate(['level', 'slope', 'curvature']):
                # Get the factor data
                factor_data = factors[factor].values
                
                # Split into training and testing
                y_train = factor_data[:n_train]
                
                # Train the model
                model = AR1Model()
                model.fit(y_train)
                
                # Make predictions one step at a time
                for j in range(n_test):
                    # Get the data up to the current point
                    y_current = factor_data[:(n_train + j)]
                    
                    # Re-train the model if needed (for a real application, you might want to re-train less frequently)
                    if j % 10 == 0:
                        model.fit(y_current)
                    
                    # Make a one-step ahead prediction
                    factor_forecasts[j, i] = model.predict(y_current, h=1)[0]
                    
        elif model_type == 'VAR1':
            # Forecast all factors jointly using VAR(1)
            # Get the factor data
            factor_data = factors.values
            
            # Make predictions one step at a time
            for j in range(n_test):
                # Get the data up to the current point
                y_current = factor_data[:(n_train + j)]
                
                # Train the model (for a real application, you might want to re-train less frequently)
                if j % 10 == 0:
                    model = VARModel()
                    model.fit(y_current)
                
                # Make a one-step ahead prediction
                factor_forecasts[j] = model.predict(y_current, h=1)[0]
                
        elif model_type == 'RW':
            # Forecast using Random Walk (last observed value)
            # Get the factor data
            factor_data = factors.values
            
            # Make predictions one step at a time
            for j in range(n_test):
                # Get the last observed value
                factor_forecasts[j] = factor_data[n_train + j - 1]
        
        # Store the forecasts
        forecasts[model_type] = {
            'forecasts': factor_forecasts,
            'actual': factors.iloc[n_train:n_train+n_test].values,
            'actual_dates': factors.index[n_train:n_train+n_test]
        }
        
        # Calculate evaluation metrics
        metrics = {}
        for i, factor in enumerate(['level', 'slope', 'curvature']):
            metrics[factor] = calculate_forecast_errors(
                forecasts[model_type]['actual'][:, i],
                forecasts[model_type]['forecasts'][:, i]
            )
        
        forecasts[model_type]['metrics'] = metrics
    
    return forecasts

def forecast_term_structure(factors_forecasts, dns_model, maturities):
    """
    Forecast the term structure using the factor forecasts.
    
    Parameters:
    -----------
    factors_forecasts : dict
        Dictionary of factor forecasts
    dns_model : DynamicNelsonSiegel
        Fitted DNS model
    maturities : numpy array
        Time to maturity
    
    Returns:
    --------
    ts_forecasts : dict
        Dictionary of term structure forecasts
    """
    # Initialize term structure forecasts dictionary
    ts_forecasts = {}
    
    # Generate term structure forecasts for each model
    for model_type, forecast_data in factors_forecasts.items():
        # Get the forecasted factors
        factor_forecasts = forecast_data['forecasts']
        
        # Create DataFrame of forecasted factors
        factors_df = pd.DataFrame(
            factor_forecasts,
            columns=['level', 'slope', 'curvature'],
            index=forecast_data['actual_dates']
        )
        
        # Generate term structure forecasts
        ts_forecasts[model_type] = dns_model.predict(factors_df, maturities)
    
    return ts_forecasts

def evaluate_term_structure_forecasts(ts_forecasts, actual_ts, model_types=['FTDNN', 'AR1', 'VAR1', 'RW']):
    """
    Evaluate the term structure forecasts.
    
    Parameters:
    -----------
    ts_forecasts : dict
        Dictionary of term structure forecasts
    actual_ts : pandas DataFrame
        Actual term structure
    model_types : list
        List of models used for forecasting
    
    Returns:
    --------
    ts_metrics : dict
        Dictionary of evaluation metrics
    """
    # Initialize metrics dictionary
    ts_metrics = {}
    
    # Calculate metrics for each model and maturity
    for model_type in model_types:
        ts_metrics[model_type] = {}
        
        # Calculate metrics for each maturity
        for maturity in actual_ts.columns:
            ts_metrics[model_type][maturity] = calculate_forecast_errors(
                actual_ts[maturity].values,
                ts_forecasts[model_type][maturity].values
            )
    
    return ts_metrics

def compare_models(ts_metrics, metric='RMSE', maturities=None):
    """
    Compare the performance of different models.
    
    Parameters:
    -----------
    ts_metrics : dict
        Dictionary of evaluation metrics
    metric : str
        Metric to use for comparison
    maturities : list or None
        List of maturities to include in comparison
    
    Returns:
    --------
    comparison : pandas DataFrame
        DataFrame of model comparison
    """
    # Initialize the comparison DataFrame
    model_types = list(ts_metrics.keys())
    
    if maturities is None:
        maturities = list(ts_metrics[model_types[0]].keys())
    
    comparison = pd.DataFrame(index=maturities, columns=model_types)
    
    # Fill in the metrics
    for model_type in model_types:
        for maturity in maturities:
            comparison.loc[maturity, model_type] = ts_metrics[model_type][maturity][metric]
    
    return comparison

#----------------------------------------
# Trading Strategy Implementation
#----------------------------------------

class TermStructureTrader:
    """
    Trading strategy implementation for crude oil futures based on term structure forecasts.
    """
    def __init__(self, initial_capital=100000, position_size=0.1, stop_loss=0.02, take_profit=0.04,
                max_positions=3, transaction_cost=0.0005):
        """
        Initialize the trading strategy.
        
        Parameters:
        -----------
        initial_capital : float
            Initial trading capital
        position_size : float
            Size of each position as fraction of capital
        stop_loss : float
            Stop loss level as fraction of entry price
        take_profit : float
            Take profit level as fraction of entry price
        max_positions : int
            Maximum number of concurrent positions
        transaction_cost : float
            Transaction cost as fraction of trade value
        """
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.position_size = position_size
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.max_positions = max_positions
        self.transaction_cost = transaction_cost
        
        # Trading state
        self.positions = {}  # {maturity: {entry_price, position_type, size, entry_date}}
        self.closed_trades = []
        
        # Performance tracking
        self.equity_curve = []
        self.daily_returns = []
        self.trade_history = []
    
    def reset(self):
        """Reset the trader to initial state."""
        self.capital = self.initial_capital
        self.positions = {}
        self.closed_trades = []
        self.equity_curve = []
        self.daily_returns = []
        self.trade_history = []
    
    def identify_trading_signals(self, current_term_structure, forecast_term_structure, date):
        """
        Identify trading signals based on current and forecasted term structure.
        
        Parameters:
        -----------
        current_term_structure : pandas Series
            Current term structure (prices for different maturities)
        forecast_term_structure : pandas Series
            Forecasted term structure for next period
        date : datetime
            Current date
        
        Returns:
        --------
        signals : list
            List of trading signals
        """
        signals = []
        
        for maturity in current_term_structure.index:
            current_price = current_term_structure[maturity]
            forecast_price = forecast_term_structure[maturity]
            
            # Calculate expected return
            expected_return = (forecast_price - current_price) / current_price
            
            # Generate signals based on expected return
            if expected_return > 0.01:  # 1% threshold for long positions
                signal = {
                    'maturity': maturity,
                    'type': 'LONG',
                    'price': current_price,
                    'expected_return': expected_return,
                    'date': date
                }
                signals.append(signal)
            elif expected_return < -0.01:  # -1% threshold for short positions
                signal = {
                    'maturity': maturity,
                    'type': 'SHORT',
                    'price': current_price,
                    'expected_return': abs(expected_return),
                    'date': date
                }
                signals.append(signal)
        
        # Sort signals by expected return (highest first)
        signals.sort(key=lambda x: x['expected_return'], reverse=True)
        
        return signals
    
    def execute_trades(self, signals, date):
        """
        Execute trades based on trading signals.
        
        Parameters:
        -----------
        signals : list
            List of trading signals
        date : datetime
            Current date
        """
        # First, try to enter new positions
        for signal in signals:
            # Check if we already have maximum positions
            if len(self.positions) >= self.max_positions:
                break
            
            maturity = signal['maturity']
            
            # Check if we already have a position for this maturity
            if maturity in self.positions:
                continue
            
            # Calculate position size
            position_value = self.capital * self.position_size
            entry_price = signal['price']
            size = position_value / entry_price
            
            # Calculate transaction cost
            transaction_cost = position_value * self.transaction_cost
            
            # Enter position
            self.positions[maturity] = {
                'entry_price': entry_price,
                'position_type': signal['type'],
                'size': size,
                'entry_date': date,
                'value': position_value
            }
            
            # Deduct transaction cost from capital
            self.capital -= transaction_cost
            
            # Log trade
            self.trade_history.append({
                'date': date,
                'action': f'OPEN {signal["type"]}',
                'maturity': maturity,
                'price': entry_price,
                'size': size,
                'value': position_value,
                'cost': transaction_cost,
                'capital': self.capital
            })
    
    def manage_positions(self, current_term_structure, date):
        """
        Manage existing positions - check for stop loss, take profit, etc.
        
        Parameters:
        -----------
        current_term_structure : pandas Series
            Current term structure (prices for different maturities)
        date : datetime
            Current date
        """
        # List of positions to close
        to_close = []
        
        # Check each position
        for maturity, position in self.positions.items():
            # Skip if maturity is not in current term structure
            if maturity not in current_term_structure:
                continue
            
            current_price = current_term_structure[maturity]
            entry_price = position['entry_price']
            position_type = position['position_type']
            size = position['size']
            
            # Calculate current position value
            current_value = size * current_price
            
            # Calculate return
            if position_type == 'LONG':
                pnl_pct = (current_price - entry_price) / entry_price
            else:  # SHORT
                pnl_pct = (entry_price - current_price) / entry_price
            
            # Check stop loss
            if pnl_pct < -self.stop_loss:
                to_close.append((maturity, 'STOP_LOSS', current_price, pnl_pct))
            
            # Check take profit
            elif pnl_pct > self.take_profit:
                to_close.append((maturity, 'TAKE_PROFIT', current_price, pnl_pct))
        
        # Close positions
        for maturity, reason, price, pnl_pct in to_close:
            self.close_position(maturity, price, date, reason)
    
    def close_position(self, maturity, price, date, reason='MANUAL'):
        """
        Close a specific position.
        
        Parameters:
        -----------
        maturity : int
            Maturity of the position to close
        price : float
            Closing price
        date : datetime
            Current date
        reason : str
            Reason for closing the position
        """
        # Get position details
        position = self.positions[maturity]
        entry_price = position['entry_price']
        position_type = position['position_type']
        size = position['size']
        entry_date = position['entry_date']
        
        # Calculate position value and P&L
        entry_value = size * entry_price
        exit_value = size * price
        
        if position_type == 'LONG':
            pnl = exit_value - entry_value
        else:  # SHORT
            pnl = entry_value - exit_value
        
        # Calculate transaction cost
        transaction_cost = exit_value * self.transaction_cost
        
        # Update capital
        self.capital += exit_value + pnl - transaction_cost
        
        # Record closed trade
        self.closed_trades.append({
            'entry_date': entry_date,
            'exit_date': date,
            'maturity': maturity,
            'position_type': position_type,
            'entry_price': entry_price,
            'exit_price': price,
            'size': size,
            'pnl': pnl,
            'reason': reason
        })
        
        # Log trade
        self.trade_history.append({
            'date': date,
            'action': f'CLOSE {position_type}',
            'maturity': maturity,
            'price': price,
            'size': size,
            'value': exit_value,
            'pnl': pnl,
            'cost': transaction_cost,
            'reason': reason,
            'capital': self.capital
        })
        
        # Remove position
        del self.positions[maturity]
    
    def close_all_positions(self, current_term_structure, date):
        """
        Close all open positions.
        
        Parameters:
        -----------
        current_term_structure : pandas Series
            Current term structure (prices for different maturities)
        date : datetime
            Current date
        """
        # List of positions to close
        maturities = list(self.positions.keys())
        
        # Close each position
        for maturity in maturities:
            if maturity in current_term_structure:
                price = current_term_structure[maturity]
                self.close_position(maturity, price, date, 'END_OF_BACKTEST')
    
    def update_equity_curve(self, date):
        """
        Update equity curve with current portfolio value.
        
        Parameters:
        -----------
        date : datetime
            Current date
        """
        # Calculate portfolio value (capital + open positions)
        portfolio_value = self.capital
        
        # Add value of open positions
        for maturity, position in self.positions.items():
            portfolio_value += position['value']
        
        # Add to equity curve
        self.equity_curve.append({
            'date': date,
            'equity': portfolio_value
        })
        
        # Calculate daily return if we have at least 2 data points
        if len(self.equity_curve) >= 2:
            prev_equity = self.equity_curve[-2]['equity']
            daily_return = (portfolio_value - prev_equity) / prev_equity
            self.daily_returns.append(daily_return)
    
    def calculate_performance_metrics(self):
        """
        Calculate performance metrics for the trading strategy.
        
        Returns:
        --------
        metrics : dict
            Dictionary of performance metrics
        """
        # Convert equity curve to DataFrame
        if not self.equity_curve:
            return {
                'total_return': 0,
                'annualized_return': 0,
                'annualized_volatility': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'num_trades': 0
            }
        
        equity_df = pd.DataFrame(self.equity_curve)
        equity_df.set_index('date', inplace=True)
        
        # Calculate total return
        initial_equity = self.initial_capital
        final_equity = equity_df['equity'].iloc[-1]
        total_return = (final_equity - initial_equity) / initial_equity
        
        # Calculate annualized return and volatility
        if len(self.daily_returns) > 0:
            daily_return_mean = np.mean(self.daily_returns)
            daily_return_std = np.std(self.daily_returns)
            
            annualized_return = (1 + daily_return_mean) ** 252 - 1
            annualized_volatility = daily_return_std * np.sqrt(252)
            
            # Calculate Sharpe ratio
            risk_free_rate = 0.02  # Assuming 2% risk-free rate
            sharpe_ratio = (annualized_return - risk_free_rate) / annualized_volatility if annualized_volatility > 0 else 0
        else:
            annualized_return = 0
            annualized_volatility = 0
            sharpe_ratio = 0
        
        # Calculate maximum drawdown
        equity_series = equity_df['equity']
        equity_series_cummax = equity_series.cummax()
        drawdowns = (equity_series - equity_series_cummax) / equity_series_cummax
        max_drawdown = drawdowns.min() if len(drawdowns) > 0 else 0
        
        # Calculate win rate and profit factor
        if self.closed_trades:
            winning_trades = [trade for trade in self.closed_trades if trade['pnl'] > 0]
            losing_trades = [trade for trade in self.closed_trades if trade['pnl'] <= 0]
            
            num_winning = len(winning_trades)
            num_losing = len(losing_trades)
            num_trades = len(self.closed_trades)
            
            win_rate = num_winning / num_trades if num_trades > 0 else 0
            
            total_profit = sum(trade['pnl'] for trade in winning_trades)
            total_loss = sum(abs(trade['pnl']) for trade in losing_trades)
            
            profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
        else:
            win_rate = 0
            profit_factor = 0
            num_trades = 0
        
        # Return metrics
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'annualized_volatility': annualized_volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'num_trades': num_trades
        }


def backtest_term_structure_strategy(futures_data, forecast_model='FTDNN', test_size=0.2, 
                                    initial_capital=100000, position_size=0.1, 
                                    stop_loss=0.02, take_profit=0.04, max_positions=3,
                                    transaction_cost=0.0005):
    """
    Backtest the term structure trading strategy.
    
    Parameters:
    -----------
    futures_data : pandas DataFrame
        Historical futures data with term structure
    forecast_model : str
        Model to use for forecasting ('FTDNN', 'AR1', 'VAR1', 'RW')
    test_size : float
        Fraction of data to use for testing
    initial_capital : float
        Initial trading capital
    position_size : float
        Size of each position as fraction of capital
    stop_loss : float
        Stop loss level as fraction of entry price
    take_profit : float
        Take profit level as fraction of entry price
    max_positions : int
        Maximum number of concurrent positions
    transaction_cost : float
        Transaction cost as fraction of trade value
    
    Returns:
    --------
    results : dict
        Backtest results including performance metrics
    """
    print(f"Backtesting term structure trading strategy with {forecast_model} model...")
    
    # 1. Split data into training and testing sets
    n_test = int(len(futures_data) * test_size)
    n_train = len(futures_data) - n_test
    
    train_data = futures_data.iloc[:n_train]
    test_data = futures_data.iloc[n_train:]
    
    # 2. Fit Dynamic Nelson-Siegel model
    dns_model = DynamicNelsonSiegel(lambda_fixed=0.0058)
    factors = dns_model.fit(train_data)
    
    # 3. Forecast factors for test period
    factors_all = dns_model.fit(futures_data)
    
    # For simplicity, we'll use a rolling forecast approach
    maturities = futures_data.columns.values
    
    # Initialize trading strategy
    trader = TermStructureTrader(
        initial_capital=initial_capital,
        position_size=position_size,
        stop_loss=stop_loss,
        take_profit=take_profit,
        max_positions=max_positions,
        transaction_cost=transaction_cost
    )
    
    # Backtest period
    for i in range(1, len(test_data)):
        current_date = test_data.index[i]
        
        # Get current term structure
        current_ts = test_data.iloc[i]
        
        # Get previous data up to current point
        historical_data = futures_data.iloc[:n_train+i]
        
        # Extract factors
        historical_factors = dns_model.fit(historical_data)
        
        # Forecast factors for next period
        if forecast_model == 'FTDNN':
            # Forecast using FTDNN
            ftdnn_model = PyTorchFTDNN(input_dim=3, num_delays=1, hidden_sizes=[32, 16])
            
            # Get factor data
            X = historical_factors.values
            
            # Train model on each factor
            factor_forecasts = np.zeros(3)
            
            for j, factor in enumerate(['level', 'slope', 'curvature']):
                y = historical_factors[factor].values
                ftdnn_model.fit(X, y, validation_split=0.2, patience=10, verbose=0)
                prediction = ftdnn_model.predict(X[-ftdnn_model.num_delays:])
                factor_forecasts[j] = prediction[0]
            
        elif forecast_model == 'AR1':
            # Forecast using AR(1)
            factor_forecasts = np.zeros(3)
            
            for j, factor in enumerate(['level', 'slope', 'curvature']):
                y = historical_factors[factor].values
                model = AR1Model()
                model.fit(y)
                factor_forecasts[j] = model.predict(y, h=1)[0]
                
        elif forecast_model == 'VAR1':
            # Forecast using VAR(1)
            factor_data = historical_factors.values
            model = VARModel()
            model.fit(factor_data)
            factor_forecasts = model.predict(factor_data, h=1)[0]
            
        elif forecast_model == 'RW':
            # Forecast using Random Walk
            factor_forecasts = historical_factors.iloc[-1].values
        
        # Create forecast factors DataFrame
        forecast_factors_df = pd.DataFrame(
            [factor_forecasts],
            columns=['level', 'slope', 'curvature'],
            index=[current_date]
        )
        
        # Generate forecast term structure
        forecast_ts = dns_model.predict(forecast_factors_df, maturities).iloc[0]
        
        # Generate trading signals
        signals = trader.identify_trading_signals(current_ts, forecast_ts, current_date)
        
        # Execute trades
        trader.execute_trades(signals, current_date)
        
        # Manage existing positions
        trader.manage_positions(current_ts, current_date)
        
        # Update equity curve
        trader.update_equity_curve(current_date)
    
    # Close all positions at the end of backtest
    trader.close_all_positions(test_data.iloc[-1], test_data.index[-1])
    
    # Calculate performance metrics
    performance_metrics = trader.calculate_performance_metrics()
    
    # Format equity curve as DataFrame
    equity_curve = pd.DataFrame(trader.equity_curve)
    if not equity_curve.empty:
        equity_curve.set_index('date', inplace=True)
    
    # Format trade history as DataFrame
    trade_history = pd.DataFrame(trader.trade_history)
    
    # Return results
    return {
        'performance': performance_metrics,
        'equity_curve': equity_curve,
        'trade_history': trade_history,
        'closed_trades': trader.closed_trades
    }

def plot_backtest_results(backtest_results, title="Term Structure Trading Strategy Backtest Results"):
    """
    Plot the backtest results.
    
    Parameters:
    -----------
    backtest_results : dict
        Backtest results from backtest_term_structure_strategy
    title : str
        Plot title
    """
    # Extract data
    equity_curve = backtest_results['equity_curve']
    performance = backtest_results['performance']
    
    if equity_curve.empty:
        print("No equity curve data to plot.")
        return
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Plot equity curve
    plt.subplot(2, 1, 1)
    plt.plot(equity_curve.index, equity_curve['equity'])
    plt.title('Equity Curve')
    plt.ylabel('Equity')
    plt.grid(True)
    
    # Plot daily returns
    if 'daily_returns' in equity_curve.columns:
        plt.subplot(2, 1, 2)
        plt.plot(equity_curve.index, equity_curve['daily_returns'])
        plt.axhline(y=0, color='r', linestyle='-')
        plt.title('Daily Returns')
        plt.ylabel('Return')
        plt.grid(True)
    
    # Add performance metrics as text
    metrics_text = (
        f"Total Return: {performance['total_return']*100:.2f}%\n"
        f"Annualized Return: {performance['annualized_return']*100:.2f}%\n"
        f"Annualized Volatility: {performance['annualized_volatility']*100:.2f}%\n"
        f"Sharpe Ratio: {performance['sharpe_ratio']:.2f}\n"
        f"Max Drawdown: {abs(performance['max_drawdown'])*100:.2f}%\n"
        f"Win Rate: {performance['win_rate']*100:.2f}%\n"
        f"Profit Factor: {performance['profit_factor']:.2f}\n"
        f"Number of Trades: {performance['num_trades']}"
    )
    
    plt.figtext(0.15, 0.01, metrics_text, ha='left', fontsize=10, bbox={'facecolor': 'white', 'alpha': 0.9, 'pad': 5})
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)
    plt.show()

def compare_model_strategies(futures_data, models=['FTDNN', 'AR1', 'VAR1', 'RW'], 
                            test_size=0.2, initial_capital=100000):
    """
    Compare trading strategies using different forecasting models.
    
    Parameters:
    -----------
    futures_data : pandas DataFrame
        Historical futures data with term structure
    models : list
        List of models to compare
    test_size : float
        Fraction of data to use for testing
    initial_capital : float
        Initial trading capital
    
    Returns:
    --------
    results : dict
        Dictionary of backtest results for each model
    """
    results = {}
    
    for model in models:
        print(f"Testing strategy with {model} model...")
        backtest_result = backtest_term_structure_strategy(
            futures_data, 
            forecast_model=model,
            test_size=test_size,
            initial_capital=initial_capital
        )
        results[model] = backtest_result
    
    # Plot comparison of equity curves
    plt.figure(figsize=(12, 6))
    
    for model, result in results.items():
        equity_curve = result['equity_curve']
        if not equity_curve.empty:
            # Normalize to percentage return
            initial_equity = initial_capital
            normalized_equity = (equity_curve['equity'] - initial_equity) / initial_equity * 100
            plt.plot(equity_curve.index, normalized_equity, label=f'{model} (Return: {result["performance"]["total_return"]*100:.2f}%)')
    
    plt.title('Comparison of Trading Strategies with Different Forecasting Models')
    plt.ylabel('Return (%)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    # Print performance summary
    print("\nPerformance Summary:")
    metrics = ['total_return', 'sharpe_ratio', 'max_drawdown', 'win_rate', 'num_trades']
    summary = pd.DataFrame(index=models, columns=metrics)
    
    for model, result in results.items():
        for metric in metrics:
            if metric == 'total_return' or metric == 'max_drawdown' or metric == 'win_rate':
                summary.loc[model, metric] = f"{result['performance'][metric]*100:.2f}%"
            else:
                summary.loc[model, metric] = f"{result['performance'][metric]:.2f}"
    
    print(summary)
    
    return results

#----------------------------------------
# Visualization Functions
#----------------------------------------

def plot_simulated_spot_prices(spot_prices, title="Simulated Crude Oil Spot Prices"):
    """
    Plot the simulated spot prices.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(spot_prices)
    plt.title(title)
    plt.xlabel('Days')
    plt.ylabel('Price')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_term_structure(futures_data, sample_dates=None, title="Term Structure of Crude Oil Futures"):
    """
    Plot the term structure for selected dates.
    """
    if sample_dates is None:
        # Sample a few dates from the dataset
        sample_dates = [futures_data.index[0], futures_data.index[len(futures_data)//3], 
                        futures_data.index[2*len(futures_data)//3], futures_data.index[-1]]
    
    plt.figure(figsize=(12, 6))
    
    for date in sample_dates:
        plt.plot(futures_data.columns, futures_data.loc[date], label=date.strftime('%Y-%m-%d'))
    
    plt.title(title)
    plt.xlabel('Days to Maturity')
    plt.ylabel('Futures Price')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_fitted_curves(futures_data, dns_model, sample_dates=None, title="Fitted Nelson-Siegel Curves"):
    """
    Plot the fitted curves against actual data for selected dates.
    """
    if sample_dates is None:
        # Sample a few dates from the dataset
        sample_dates = [futures_data.index[0], futures_data.index[len(futures_data)//3], 
                        futures_data.index[2*len(futures_data)//3], futures_data.index[-1]]
    
    plt.figure(figsize=(15, 10))
    
    for i, date in enumerate(sample_dates):
        plt.subplot(2, 2, i+1)
        
        # Get the actual data
        actual = futures_data.loc[date]
        
        # Get the factors for this date
        beta0 = dns_model.beta0[futures_data.index.get_loc(date)]
        beta1 = dns_model.beta1[futures_data.index.get_loc(date)]
        beta2 = dns_model.beta2[futures_data.index.get_loc(date)]
        
        # Generate the fitted curve
        maturities = np.array([float(col) for col in futures_data.columns])
        fitted = dns_model._nelson_siegel_curve(beta0, beta1, beta2, dns_model.lambda_opt, maturities)
        
        # Plot
        plt.scatter(maturities, actual, label='Actual', alpha=0.7)
        plt.plot(maturities, fitted, 'r-', label='Fitted')
        plt.title(f'Date: {date.strftime("%Y-%m-%d")}')
        plt.xlabel('Days to Maturity')
        plt.ylabel('Futures Price')
        plt.legend()
        plt.grid(True)
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.show()

def plot_factors(factors, title="Estimated Nelson-Siegel Factors"):
    """
    Plot the estimated factors.
    """
    plt.figure(figsize=(12, 8))
    
    plt.subplot(3, 1, 1)
    plt.plot(factors.index, factors['level'])
    plt.title('Level Factor (β₀)')
    plt.ylabel('Value')
    plt.grid(True)
    
    plt.subplot(3, 1, 2)
    plt.plot(factors.index, factors['slope'])
    plt.title('Slope Factor (β₁)')
    plt.ylabel('Value')
    plt.grid(True)
    
    plt.subplot(3, 1, 3)
    plt.plot(factors.index, factors['curvature'])
    plt.title('Curvature Factor (β₂)')
    plt.ylabel('Value')
    plt.xlabel('Date')
    plt.grid(True)
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.show()

def plot_factor_forecasts(factors, forecasts, n_samples=100, title="Factor Forecasts Comparison"):
    """
    Plot the factor forecasts against actual values.
    """
    # Get the test data indices
    n_test = len(next(iter(forecasts.values()))['forecasts'])
    n_train = len(factors) - n_test
    
    # Take the last n_samples of the test set
    start_idx = max(0, n_test - n_samples)
    
    # Get the actual test values
    test_dates = factors.index[n_train:n_train+n_test]
    actual_factors = factors.iloc[n_train:n_train+n_test]
    
    # Create a figure for each factor
    factor_names = ['level', 'slope', 'curvature']
    
    plt.figure(figsize=(15, 10))
    
    for i, factor in enumerate(factor_names):
        plt.subplot(3, 1, i+1)
        
        # Plot actual values
        plt.plot(test_dates, actual_factors[factor], 'k-', label='Actual')
        
        # Plot forecasts for each model
        for model_type, forecast_data in forecasts.items():
            plt.plot(test_dates, forecast_data['forecasts'][:, i], label=model_type)
        
        plt.title(f'{factor.capitalize()} Factor (β{i})')
        plt.ylabel('Value')
        if i == 2:
            plt.xlabel('Date')
        plt.legend()
        plt.grid(True)
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.show()

def plot_term_structure_forecasts(ts_forecasts, actual_ts, model_types, sample_dates=None, sample_maturities=None, title="Term Structure Forecast Comparison"):
    """
    Plot the term structure forecasts for selected dates.
    """
    if sample_dates is None:
        # Sample a few dates from the dataset
        sample_dates = [actual_ts.index[0], actual_ts.index[len(actual_ts)//3], 
                        actual_ts.index[2*len(actual_ts)//3], actual_ts.index[-1]]
    
    if sample_maturities is None:
        # Use all maturities
        sample_maturities = actual_ts.columns.tolist()
    
    plt.figure(figsize=(15, 10))
    
    for i, date in enumerate(sample_dates):
        plt.subplot(2, 2, i+1)
        
        # Plot actual term structure
        plt.plot(sample_maturities, actual_ts.loc[date, sample_maturities], 'k-', label='Actual')
        
        # Plot forecasts for each model
        for model_type in model_types:
            plt.plot(sample_maturities, ts_forecasts[model_type].loc[date, sample_maturities], label=model_type)
        
        plt.title(f'Date: {date.strftime("%Y-%m-%d")}')
        plt.xlabel('Days to Maturity')
        plt.ylabel('Futures Price')
        plt.legend()
        plt.grid(True)
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.show()

def plot_forecast_errors(ts_metrics, metric='RMSE', title="Forecast Errors by Maturity"):
    """
    Plot the forecast errors for each model and maturity.
    """
    comparison = compare_models(ts_metrics, metric=metric)
    
    plt.figure(figsize=(12, 6))
    
    for model in comparison.columns:
        plt.plot(comparison.index, comparison[model], label=model)
    
    plt.title(title)
    plt.xlabel('Days to Maturity')
    plt.ylabel(metric)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    return comparison

def plot_error_heatmap(ts_metrics, metric='RMSE', title="Forecast Error Heatmap"):
    """
    Plot a heatmap of forecast errors.
    """
    comparison = compare_models(ts_metrics, metric=metric)
    
    # Convert to float for seaborn heatmap
    comparison = comparison.astype(float)
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(comparison, annot=True, cmap='YlGnBu', fmt='.2f')
    plt.title(title)
    plt.tight_layout()
    plt.show()
    
    return comparison

#----------------------------------------
# Main Execution - Run the complete analysis
#----------------------------------------

def main():
    """
    Run the complete analysis including the trading strategy backtest.
    """
    print("Simulating crude oil futures data...")
    
    # 1. Generate simulated data
    # Simulate spot prices
    n_days = 1000
    spot_prices = generate_oil_price_process(n_days=n_days, start_price=100, volatility=0.02, 
                                           mean_reversion=0.02, long_term_mean=100)
    
    # Generate term structure
    days_to_maturity = [30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 360,
                        390, 420, 450, 480, 510, 540, 570, 600, 630, 660, 690, 720]
    futures_data = generate_term_structure(spot_prices, days_to_maturity)
    
    # Plot simulated data
    plot_simulated_spot_prices(spot_prices)
    plot_term_structure(futures_data)
    
    # 2. Apply Dynamic Nelson-Siegel model
    print("Fitting Dynamic Nelson-Siegel model...")
    dns_model = DynamicNelsonSiegel(lambda_fixed=0.0058)  # Using fixed lambda as in the paper
    factors = dns_model.fit(futures_data)
    
    # Plot estimated factors
    plot_factors(factors)
    plot_fitted_curves(futures_data, dns_model)
    
    # 3. Forecast factors using different models
    print("Forecasting factors...")
    h = 1  # 1-month ahead forecasts
    forecasts = forecast_factors(factors, h=h, test_size=0.2, 
                                model_types=['FTDNN', 'AR1', 'VAR1', 'RW'])
    
    # Plot factor forecasts
    plot_factor_forecasts(factors, forecasts)
    
    # 4. Generate term structure forecasts
    print("Generating term structure forecasts...")
    maturities = np.array(days_to_maturity)
    ts_forecasts = forecast_term_structure(forecasts, dns_model, maturities)
    
    # Get actual test term structure
    n_test = int(len(futures_data) * 0.2)
    n_train = len(futures_data) - n_test
    actual_ts = futures_data.iloc[n_train:n_train+n_test]
    
    # 5. Evaluate term structure forecasts
    print("Evaluating term structure forecasts...")
    ts_metrics = evaluate_term_structure_forecasts(ts_forecasts, actual_ts, 
                                                 model_types=['FTDNN', 'AR1', 'VAR1', 'RW'])
    
    # 6. Compare models
    print("Comparing models...")
    rmse_comparison = plot_forecast_errors(ts_metrics, metric='RMSE')
    mae_comparison = plot_forecast_errors(ts_metrics, metric='MAE')
    plot_error_heatmap(ts_metrics, metric='RMSE')
    
    # 7. Print forecast performance summary
    print("\nForecast Performance Summary (Average RMSE across all maturities):")
    for model in rmse_comparison.columns:
        print(f"{model}: {rmse_comparison[model].mean():.4f}")
    
    # 8. Backtest trading strategy with the FTDNN model
    print("\nBacktesting trading strategy with FTDNN model...")
    backtest_result = backtest_term_structure_strategy(
        futures_data, 
        forecast_model='FTDNN',
        test_size=0.2,
        initial_capital=100000,
        position_size=0.1,
        stop_loss=0.02,
        take_profit=0.04,
        max_positions=3
    )
    
    # Plot backtest results
    plot_backtest_results(backtest_result, title="Trading Strategy with FTDNN Model")
    
    # 9. Compare strategies using different models
    print("\nComparing trading strategies with different forecasting models...")
    strategy_comparison = compare_model_strategies(
        futures_data,
        models=['FTDNN', 'AR1', 'VAR1', 'RW'],
        test_size=0.2,
        initial_capital=100000
    )
    
    print("\nDone!")

if __name__ == "__main__":
    main()