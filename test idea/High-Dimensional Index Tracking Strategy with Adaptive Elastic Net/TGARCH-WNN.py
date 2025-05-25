import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from statsmodels.tsa.stattools import adfuller, coint
from statsmodels.api import OLS
from arch import arch_model
import pywt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from datetime import datetime, timedelta
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Check if GPU is available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class WaveletNeuralNetwork(nn.Module):
    """
    PyTorch implementation of a wavelet neural network
    """
    def __init__(self, input_dim=6, hidden_dim=8):
        super(WaveletNeuralNetwork, self).__init__()
        self.hidden1 = nn.Linear(input_dim, hidden_dim)
        self.hidden2 = nn.Linear(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, 1)
        self.tanh = nn.Tanh()
        
    def forward(self, x):
        x = self.tanh(self.hidden1(x))
        x = self.tanh(self.hidden2(x))
        x = self.output(x)
        return x


class BackpropagationNN(nn.Module):
    """
    PyTorch implementation of a standard backpropagation neural network
    """
    def __init__(self, input_dim=6, hidden1_dim=13, hidden2_dim=11):
        super(BackpropagationNN, self).__init__()
        self.hidden1 = nn.Linear(input_dim, hidden1_dim)
        self.hidden2 = nn.Linear(hidden1_dim, hidden2_dim)
        self.output = nn.Linear(hidden2_dim, 1)
        self.tanh = nn.Tanh()
        
    def forward(self, x):
        x = self.tanh(self.hidden1(x))
        x = self.tanh(self.hidden2(x))
        x = self.output(x)
        return x


class TGARCH_WNN_Arbitrage:
    """
    Implementation of the TGARCH-Wavelet Neural Network statistical arbitrage model
    for metal futures market as described in the paper.
    """
    
    def __init__(self, commission_per_lot=10, margin_rate=0.1, 
                 max_position=1, stop_loss_multiplier=3):
        """
        Initialize the TGARCH-WNN arbitrage model
        
        Parameters:
        -----------
        commission_per_lot : float
            Commission charged per lot
        margin_rate : float
            Margin rate required for futures positions
        max_position : int
            Maximum position in each contract
        stop_loss_multiplier : float
            Multiplier for standard deviation to set stop loss level
        """
        self.commission_per_lot = commission_per_lot
        self.margin_rate = margin_rate
        self.max_position = max_position
        self.stop_loss_multiplier = stop_loss_multiplier
        
        # Model parameters
        self.beta0 = None  # Intercept in mean function
        self.beta1 = None  # Coefficient in mean function
        self.mu = None     # Mean of spread
        
        # TGARCH parameters
        self.omega = None
        self.alpha = None
        self.gamma = None
        self.beta = None
        
        # Trading parameters
        self.upper_threshold = None  # k1
        self.lower_threshold = None  # k2
        self.period_history = []  # History of thresholds for each period
        self.position = 0
        self.entry_spread = None
        self.entry_price_x = None
        self.entry_price_y = None
        
        # Wavelet neural network model
        self.wnn_model_k1 = None
        self.wnn_model_k2 = None
        
        # Performance tracking
        self.trades = []
        self.equity_curve = []
        self.current_equity = 0
        
    def check_cointegration(self, prices_x, prices_y):
        """
        Check cointegration relationship between two price series
        
        Parameters:
        -----------
        prices_x : array-like
            Price series of contract X
        prices_y : array-like
            Price series of contract Y
            
        Returns:
        --------
        bool : True if cointegrated, False otherwise
        """
        # Perform Phillips-Perron test on the residuals
        result = coint(prices_y, prices_x)
        pvalue = result[1]
        
        # If p-value is less than 0.05, reject the null hypothesis that
        # the series are not cointegrated
        return pvalue < 0.05
    
    def fit_tgarch(self, prices_x, prices_y, market_trend='stable'):
        """
        Fit TGARCH model to the price series
        
        Parameters:
        -----------
        prices_x : array-like
            Price series of contract X
        prices_y : array-like
            Price series of contract Y
        market_trend : str
            Market trend condition ('up', 'down', or 'stable')
            
        Returns:
        --------
        bool : True if fitted successfully, False otherwise
        """
        # Check cointegration first
        if not self.check_cointegration(prices_x, prices_y):
            print("Series are not cointegrated.")
            return False
        
        # Fit mean equation (OLS regression)
        X = np.column_stack((np.ones(len(prices_x)), prices_x))
        model = OLS(prices_y, X)
        results = model.fit()
        
        # Get parameters
        self.beta0 = results.params[0]
        self.beta1 = results.params[1]
        
        # Calculate residuals (spread series)
        residuals = results.resid
        self.mu = np.mean(residuals)
        
        # Decentralize spread
        mspread = residuals - self.mu
        
        # Fit TGARCH model to the residuals
        if market_trend == 'stable':
            # Standard GARCH for stable market (no asymmetric effect)
            garch_model = arch_model(mspread, vol='GARCH', p=1, q=1)
        else:
            # TGARCH for trending market
            garch_model = arch_model(mspread, vol='GARCH', p=1, o=1, q=1)
        
        try:
            garch_results = garch_model.fit(disp='off', show_warning=False)
            
            # Extract TGARCH parameters
            self.omega = garch_results.params['omega']
            self.alpha = garch_results.params['alpha[1]']
            self.beta = garch_results.params['beta[1]']
            
            if market_trend != 'stable':
                self.gamma = garch_results.params['gamma[1]']
            else:
                self.gamma = 0
                
            return True
        except:
            print("TGARCH model fitting failed.")
            return False
    
    def predict_volatility(self, mspread):
        """
        Predict volatility using the fitted TGARCH model
        
        Parameters:
        -----------
        mspread : array-like
            Mean-adjusted spread series
            
        Returns:
        --------
        array-like : Predicted conditional standard deviation
        """
        n = len(mspread)
        sigma_sq = np.zeros(n)
        sigma_sq[0] = self.omega / (1 - self.alpha - self.beta)  # Unconditional variance
        
        for t in range(1, n):
            resid_sq = mspread[t-1]**2
            # Asymmetric effect when residual is negative
            gamma_effect = self.gamma * resid_sq * (mspread[t-1] < 0)
            sigma_sq[t] = self.omega + (self.alpha + gamma_effect) * resid_sq + self.beta * sigma_sq[t-1]
        
        return np.sqrt(sigma_sq)
    
    def compute_optimal_thresholds(self, prices_x, prices_y, sigma, 
                                    initial_capital=100000, step=0.01):
        """
        Compute historically optimal trading thresholds using a grid search
        
        Parameters:
        -----------
        prices_x : array-like
            Price series of contract X
        prices_y : array-like
            Price series of contract Y
        sigma : array-like
            Predicted conditional standard deviation
        initial_capital : float
            Initial capital for backtesting
        step : float
            Step size for threshold grid search
            
        Returns:
        --------
        tuple : (optimal upper threshold, optimal lower threshold)
        """
        # Calculate spread
        spread = prices_y - (self.beta0 + self.beta1 * prices_x)
        mspread = spread - self.mu
        
        # Minimum threshold calculation based on commission
        min_threshold = (1 + self.beta1) * self.commission_per_lot / (prices_x[0] * sigma[0])
        
        # Range of thresholds to test
        k1_range = np.arange(min_threshold, self.stop_loss_multiplier, step)
        k2_range = np.arange(min_threshold, self.stop_loss_multiplier, step)
        
        max_profit = -np.inf
        optimal_k1 = min_threshold
        optimal_k2 = min_threshold
        
        # Grid search for optimal thresholds
        for k1 in k1_range:
            for k2 in k2_range:
                profit = self._backtest_thresholds(prices_x, prices_y, mspread, 
                                                  sigma, k1, k2, initial_capital)
                if profit > max_profit:
                    max_profit = profit
                    optimal_k1 = k1
                    optimal_k2 = k2
        
        return optimal_k1, optimal_k2
    
    def _backtest_thresholds(self, prices_x, prices_y, mspread, sigma, k1, k2, initial_capital):
        """
        Backtest the performance of given thresholds
        
        Parameters:
        -----------
        prices_x : array-like
            Price series of contract X
        prices_y : array-like
            Price series of contract Y
        mspread : array-like
            Mean-adjusted spread series
        sigma : array-like
            Conditional standard deviation series
        k1 : float
            Upper threshold multiplier
        k2 : float
            Lower threshold multiplier
        initial_capital : float
            Initial capital for backtesting
            
        Returns:
        --------
        float : Final capital after backtesting
        """
        position = 0
        capital = initial_capital
        stop_loss_upper = self.stop_loss_multiplier * sigma
        stop_loss_lower = -self.stop_loss_multiplier * sigma
        
        for i in range(1, len(mspread)):
            # If no position
            if position == 0:
                # Check for entry signals
                if mspread[i] > k1 * sigma[i]:
                    # Go short spread (buy X, sell Y)
                    position = -1
                    entry_price_x = prices_x[i]
                    entry_price_y = prices_y[i]
                    # Calculate required margin
                    margin = (self.beta1 * prices_x[i] + prices_y[i]) * self.margin_rate
                    # Deduct commission
                    capital -= (1 + self.beta1) * self.commission_per_lot
                elif mspread[i] < -k2 * sigma[i]:
                    # Go long spread (sell X, buy Y)
                    position = 1
                    entry_price_x = prices_x[i]
                    entry_price_y = prices_y[i]
                    # Calculate required margin
                    margin = (self.beta1 * prices_x[i] + prices_y[i]) * self.margin_rate
                    # Deduct commission
                    capital -= (1 + self.beta1) * self.commission_per_lot
            
            # If short spread position
            elif position == -1:
                # Check for exit signal
                if mspread[i] <= 0 or mspread[i] >= stop_loss_upper[i]:
                    # Close position
                    pnl = (entry_price_y - prices_y[i]) - self.beta1 * (prices_x[i] - entry_price_x)
                    capital += pnl - (1 + self.beta1) * self.commission_per_lot
                    position = 0
            
            # If long spread position
            elif position == 1:
                # Check for exit signal
                if mspread[i] >= 0 or mspread[i] <= stop_loss_lower[i]:
                    # Close position
                    pnl = (prices_y[i] - entry_price_y) - self.beta1 * (entry_price_x - prices_x[i])
                    capital += pnl - (1 + self.beta1) * self.commission_per_lot
                    position = 0
        
        # Close any open position at the end
        if position == -1:
            pnl = (entry_price_y - prices_y[-1]) - self.beta1 * (prices_x[-1] - entry_price_x)
            capital += pnl - (1 + self.beta1) * self.commission_per_lot
        elif position == 1:
            pnl = (prices_y[-1] - entry_price_y) - self.beta1 * (entry_price_x - prices_x[-1])
            capital += pnl - (1 + self.beta1) * self.commission_per_lot
            
        return capital
    
    def build_wavelet_features(self, k_series, n_inputs=6):
        """
        Build wavelet transformed features for neural network input
        
        Parameters:
        -----------
        k_series : array-like
            Historical threshold series
        n_inputs : int
            Number of inputs for prediction
            
        Returns:
        --------
        tuple : (X, y) input and target for neural network
        """
        # Apply wavelet transform
        coeffs = pywt.wavedec(k_series, 'haar', level=2)
        
        # Reconstruct signal with denoising (truncate detail coefficients)
        # This is a simple denoising approach - setting detail coefficients to zero
        coeffs[1:] = [np.zeros_like(detail) for detail in coeffs[1:]]
        denoised_series = pywt.waverec(coeffs, 'haar')
        
        # Match the length of the original signal
        denoised_series = denoised_series[:len(k_series)]
        
        # Create input and target arrays
        X = []
        y = []
        
        for i in range(n_inputs, len(denoised_series)):
            X.append(denoised_series[i-n_inputs:i])
            y.append(k_series[i])
        
        return np.array(X), np.array(y)
    
    def train_wnn(self, k1_history, k2_history, n_inputs=6, epochs=1000, batch_size=8, learning_rate=0.01):
        """
        Train the wavelet neural network for threshold prediction using PyTorch
        
        Parameters:
        -----------
        k1_history : array-like
            Historical upper threshold values
        k2_history : array-like
            Historical lower threshold values
        n_inputs : int
            Number of inputs for prediction
        epochs : int
            Number of training epochs
        batch_size : int
            Batch size for training
        learning_rate : float
            Learning rate for optimizer
            
        Returns:
        --------
        tuple : (k1_model, k2_model) trained models
        """
        # Build features
        X1, y1 = self.build_wavelet_features(k1_history, n_inputs)
        X2, y2 = self.build_wavelet_features(k2_history, n_inputs)
        
        # Convert numpy arrays to PyTorch tensors
        X1_tensor = torch.FloatTensor(X1).to(device)
        y1_tensor = torch.FloatTensor(y1).reshape(-1, 1).to(device)
        X2_tensor = torch.FloatTensor(X2).to(device)
        y2_tensor = torch.FloatTensor(y2).reshape(-1, 1).to(device)
        
        # Create DataLoader for batching
        train_dataset1 = TensorDataset(X1_tensor, y1_tensor)
        train_loader1 = DataLoader(train_dataset1, batch_size=batch_size, shuffle=True)
        
        train_dataset2 = TensorDataset(X2_tensor, y2_tensor)
        train_loader2 = DataLoader(train_dataset2, batch_size=batch_size, shuffle=True)
        
        # Initialize models
        k1_model = WaveletNeuralNetwork(input_dim=n_inputs).to(device)
        k2_model = WaveletNeuralNetwork(input_dim=n_inputs).to(device)
        
        # Loss function and optimizer
        criterion = nn.MSELoss()
        optimizer1 = optim.Adam(k1_model.parameters(), lr=learning_rate)
        optimizer2 = optim.Adam(k2_model.parameters(), lr=learning_rate)
        
        # Training loop for k1 model
        best_loss1 = float('inf')
        patience = 50
        patience_counter1 = 0
        
        for epoch in range(epochs):
            k1_model.train()
            epoch_loss = 0
            
            for X_batch, y_batch in train_loader1:
                optimizer1.zero_grad()
                outputs = k1_model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer1.step()
                epoch_loss += loss.item()
            
            # Early stopping
            if epoch_loss < best_loss1:
                best_loss1 = epoch_loss
                patience_counter1 = 0
                # Save the best model
                best_k1_model = WaveletNeuralNetwork(input_dim=n_inputs)
                best_k1_model.load_state_dict(k1_model.state_dict())
            else:
                patience_counter1 += 1
                if patience_counter1 >= patience:
                    break
        
        # Training loop for k2 model
        best_loss2 = float('inf')
        patience_counter2 = 0
        
        for epoch in range(epochs):
            k2_model.train()
            epoch_loss = 0
            
            for X_batch, y_batch in train_loader2:
                optimizer2.zero_grad()
                outputs = k2_model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer2.step()
                epoch_loss += loss.item()
            
            # Early stopping
            if epoch_loss < best_loss2:
                best_loss2 = epoch_loss
                patience_counter2 = 0
                # Save the best model
                best_k2_model = WaveletNeuralNetwork(input_dim=n_inputs)
                best_k2_model.load_state_dict(k2_model.state_dict())
            else:
                patience_counter2 += 1
                if patience_counter2 >= patience:
                    break
        
        # Set the models to evaluation mode
        best_k1_model.eval()
        best_k2_model.eval()
        
        self.wnn_model_k1 = best_k1_model.to(device)
        self.wnn_model_k2 = best_k2_model.to(device)
        
        return best_k1_model, best_k2_model
    
    def train_bp_nn(self, k1_history, k2_history, n_inputs=6, epochs=1000, batch_size=8, learning_rate=0.01):
        """
        Train the backpropagation neural network for threshold prediction using PyTorch
        
        Parameters:
        -----------
        k1_history : array-like
            Historical upper threshold values
        k2_history : array-like
            Historical lower threshold values
        n_inputs : int
            Number of inputs for prediction
        epochs : int
            Number of training epochs
        batch_size : int
            Batch size for training
        learning_rate : float
            Learning rate for optimizer
            
        Returns:
        --------
        tuple : (k1_model, k2_model) trained models
        """
        # Prepare input data (without wavelet transform)
        X1 = []
        y1 = []
        X2 = []
        y2 = []
        
        for i in range(n_inputs, len(k1_history)):
            X1.append(k1_history[i-n_inputs:i])
            y1.append(k1_history[i])
            
        for i in range(n_inputs, len(k2_history)):
            X2.append(k2_history[i-n_inputs:i])
            y2.append(k2_history[i])
        
        # Convert numpy arrays to PyTorch tensors
        X1_tensor = torch.FloatTensor(X1).to(device)
        y1_tensor = torch.FloatTensor(y1).reshape(-1, 1).to(device)
        X2_tensor = torch.FloatTensor(X2).to(device)
        y2_tensor = torch.FloatTensor(y2).reshape(-1, 1).to(device)
        
        # Create DataLoader for batching
        train_dataset1 = TensorDataset(X1_tensor, y1_tensor)
        train_loader1 = DataLoader(train_dataset1, batch_size=batch_size, shuffle=True)
        
        train_dataset2 = TensorDataset(X2_tensor, y2_tensor)
        train_loader2 = DataLoader(train_dataset2, batch_size=batch_size, shuffle=True)
        
        # Initialize models
        k1_model = BackpropagationNN(input_dim=n_inputs).to(device)
        k2_model = BackpropagationNN(input_dim=n_inputs).to(device)
        
        # Loss function and optimizer
        criterion = nn.MSELoss()
        optimizer1 = optim.Adam(k1_model.parameters(), lr=learning_rate)
        optimizer2 = optim.Adam(k2_model.parameters(), lr=learning_rate)
        
        # Training loop for k1 model
        best_loss1 = float('inf')
        patience = 50
        patience_counter1 = 0
        
        for epoch in range(epochs):
            k1_model.train()
            epoch_loss = 0
            
            for X_batch, y_batch in train_loader1:
                optimizer1.zero_grad()
                outputs = k1_model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer1.step()
                epoch_loss += loss.item()
            
            # Early stopping
            if epoch_loss < best_loss1:
                best_loss1 = epoch_loss
                patience_counter1 = 0
                # Save the best model
                best_k1_model = BackpropagationNN(input_dim=n_inputs)
                best_k1_model.load_state_dict(k1_model.state_dict())
            else:
                patience_counter1 += 1
                if patience_counter1 >= patience:
                    break
        
        # Training loop for k2 model
        best_loss2 = float('inf')
        patience_counter2 = 0
        
        for epoch in range(epochs):
            k2_model.train()
            epoch_loss = 0
            
            for X_batch, y_batch in train_loader2:
                optimizer2.zero_grad()
                outputs = k2_model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer2.step()
                epoch_loss += loss.item()
            
            # Early stopping
            if epoch_loss < best_loss2:
                best_loss2 = epoch_loss
                patience_counter2 = 0
                # Save the best model
                best_k2_model = BackpropagationNN(input_dim=n_inputs)
                best_k2_model.load_state_dict(k2_model.state_dict())
            else:
                patience_counter2 += 1
                if patience_counter2 >= patience:
                    break
        
        # Set the models to evaluation mode
        best_k1_model.eval()
        best_k2_model.eval()
        
        return best_k1_model.to(device), best_k2_model.to(device)
    
    def predict_thresholds_wnn(self, k1_history, k2_history, n_inputs=6):
        """
        Predict the next threshold values using trained WNN models
        
        Parameters:
        -----------
        k1_history : array-like
            Historical upper threshold values
        k2_history : array-like
            Historical lower threshold values
        n_inputs : int
            Number of inputs for prediction
            
        Returns:
        --------
        tuple : (predicted_k1, predicted_k2)
        """
        if self.wnn_model_k1 is None or self.wnn_model_k2 is None:
            print("WNN models not trained yet")
            return 1.0, 1.0  # Default values
        
        # Apply wavelet transform to history
        k1_coeffs = pywt.wavedec(k1_history, 'haar', level=2)
        k2_coeffs = pywt.wavedec(k2_history, 'haar', level=2)
        
        # Reconstruct with denoising
        k1_coeffs[1:] = [np.zeros_like(detail) for detail in k1_coeffs[1:]]
        k2_coeffs[1:] = [np.zeros_like(detail) for detail in k2_coeffs[1:]]
        
        k1_denoised = pywt.waverec(k1_coeffs, 'haar')[:len(k1_history)]
        k2_denoised = pywt.waverec(k2_coeffs, 'haar')[:len(k2_history)]
        
        # Get the latest data points for prediction
        k1_input = torch.FloatTensor(k1_denoised[-n_inputs:]).reshape(1, -1).to(device)
        k2_input = torch.FloatTensor(k2_denoised[-n_inputs:]).reshape(1, -1).to(device)
        
        # Set models to evaluation mode
        self.wnn_model_k1.eval()
        self.wnn_model_k2.eval()
        
        # Make predictions
        with torch.no_grad():
            predicted_k1 = self.wnn_model_k1(k1_input).item()
            predicted_k2 = self.wnn_model_k2(k2_input).item()
        
        # Ensure predictions are positive and below stop loss
        predicted_k1 = max(0.1, min(predicted_k1, self.stop_loss_multiplier - 0.1))
        predicted_k2 = max(0.1, min(predicted_k2, self.stop_loss_multiplier - 0.1))
        
        return predicted_k1, predicted_k2
    
    def predict_thresholds_bp(self, k1_history, k2_history, bp_model_k1, bp_model_k2, n_inputs=6):
        """
        Predict the next threshold values using trained BP models
        
        Parameters:
        -----------
        k1_history : array-like
            Historical upper threshold values
        k2_history : array-like
            Historical lower threshold values
        bp_model_k1 : torch.nn.Module
            Trained BP model for k1
        bp_model_k2 : torch.nn.Module
            Trained BP model for k2
        n_inputs : int
            Number of inputs for prediction
            
        Returns:
        --------
        tuple : (predicted_k1, predicted_k2)
        """
        # Get the latest data points for prediction
        k1_input = torch.FloatTensor(k1_history[-n_inputs:]).reshape(1, -1).to(device)
        k2_input = torch.FloatTensor(k2_history[-n_inputs:]).reshape(1, -1).to(device)
        
        # Set models to evaluation mode
        bp_model_k1.eval()
        bp_model_k2.eval()
        
        # Make predictions
        with torch.no_grad():
            predicted_k1 = bp_model_k1(k1_input).item()
            predicted_k2 = bp_model_k2(k2_input).item()
        
        # Ensure predictions are positive and below stop loss
        predicted_k1 = max(0.1, min(predicted_k1, self.stop_loss_multiplier - 0.1))
        predicted_k2 = max(0.1, min(predicted_k2, self.stop_loss_multiplier - 0.1))
        
        return predicted_k1, predicted_k2
    
    def calculate_spread(self, prices_x, prices_y):
        """
        Calculate the spread and mean-adjusted spread
        
        Parameters:
        -----------
        prices_x : array-like
            Price series of contract X
        prices_y : array-like
            Price series of contract Y
            
        Returns:
        --------
        tuple : (spread, mean-adjusted spread)
        """
        spread = prices_y - (self.beta0 + self.beta1 * prices_x)
        mspread = spread - self.mu
        return spread, mspread
    
    def run_arbitrage(self, prices_x, prices_y, upper_threshold, lower_threshold):
        """
        Run statistical arbitrage with given thresholds
        
        Parameters:
        -----------
        prices_x : array-like
            Price series of contract X
        prices_y : array-like
            Price series of contract Y
        upper_threshold : float
            Upper threshold multiplier (k1)
        lower_threshold : float
            Lower threshold multiplier (k2)
            
        Returns:
        --------
        dict : Trading results
        """
        # Calculate spread
        spread, mspread = self.calculate_spread(prices_x, prices_y)
        
        # Calculate volatility
        sigma = self.predict_volatility(mspread)
        
        # Set stop loss levels
        stop_loss_upper = self.stop_loss_multiplier * sigma
        stop_loss_lower = -self.stop_loss_multiplier * sigma
        
        # Trading variables
        position = 0
        trades = []
        equity = [self.current_equity]
        capital = self.current_equity
        trade_count = 0
        win_count = 0
        commission_paid = 0
        
        for i in range(1, len(mspread)):
            # If no position
            if position == 0:
                # Check for entry signals
                if mspread[i] > upper_threshold * sigma[i]:
                    # Go short spread (buy X, sell Y)
                    position = -1
                    entry_spread = mspread[i]
                    entry_price_x = prices_x[i]
                    entry_price_y = prices_y[i]
                    entry_time = i
                    # Calculate required margin
                    margin = (self.beta1 * prices_x[i] + prices_y[i]) * self.margin_rate
                    # Deduct commission
                    commission = (1 + self.beta1) * self.commission_per_lot
                    commission_paid += commission
                    
                elif mspread[i] < -lower_threshold * sigma[i]:
                    # Go long spread (sell X, buy Y)
                    position = 1
                    entry_spread = mspread[i]
                    entry_price_x = prices_x[i]
                    entry_price_y = prices_y[i]
                    entry_time = i
                    # Calculate required margin
                    margin = (self.beta1 * prices_x[i] + prices_y[i]) * self.margin_rate
                    # Deduct commission
                    commission = (1 + self.beta1) * self.commission_per_lot
                    commission_paid += commission
            
            # If short spread position
            elif position == -1:
                # Check for exit signal
                if mspread[i] <= 0 or mspread[i] >= stop_loss_upper[i]:
                    # Close position
                    exit_time = i
                    trade_count += 1
                    pnl = (entry_price_y - prices_y[i]) - self.beta1 * (prices_x[i] - entry_price_x)
                    # Deduct commission
                    commission = (1 + self.beta1) * self.commission_per_lot
                    commission_paid += commission
                    
                    net_pnl = pnl - commission
                    if net_pnl > 0:
                        win_count += 1
                    
                    capital += net_pnl
                    equity.append(capital)
                    
                    trades.append({
                        'entry_time': entry_time,
                        'exit_time': exit_time,
                        'position': position,
                        'entry_spread': entry_spread,
                        'exit_spread': mspread[i],
                        'pnl': pnl,
                        'net_pnl': net_pnl,
                        'stop_loss_hit': mspread[i] >= stop_loss_upper[i]
                    })
                    
                    position = 0
            
            # If long spread position
            elif position == 1:
                # Check for exit signal
                if mspread[i] >= 0 or mspread[i] <= stop_loss_lower[i]:
                    # Close position
                    exit_time = i
                    trade_count += 1
                    pnl = (prices_y[i] - entry_price_y) - self.beta1 * (entry_price_x - prices_x[i])
                    # Deduct commission
                    commission = (1 + self.beta1) * self.commission_per_lot
                    commission_paid += commission
                    
                    net_pnl = pnl - commission
                    if net_pnl > 0:
                        win_count += 1
                    
                    capital += net_pnl
                    equity.append(capital)
                    
                    trades.append({
                        'entry_time': entry_time,
                        'exit_time': exit_time,
                        'position': position,
                        'entry_spread': entry_spread,
                        'exit_spread': mspread[i],
                        'pnl': pnl,
                        'net_pnl': net_pnl,
                        'stop_loss_hit': mspread[i] <= stop_loss_lower[i]
                    })
                    
                    position = 0
        
        # Close any open position at the end
        if position != 0:
            exit_time = len(mspread) - 1
            trade_count += 1
            
            if position == -1:
                pnl = (entry_price_y - prices_y[-1]) - self.beta1 * (prices_x[-1] - entry_price_x)
            else:  # position == 1
                pnl = (prices_y[-1] - entry_price_y) - self.beta1 * (entry_price_x - prices_x[-1])
            
            # Deduct commission
            commission = (1 + self.beta1) * self.commission_per_lot
            commission_paid += commission
            
            net_pnl = pnl - commission
            if net_pnl > 0:
                win_count += 1
            
            capital += net_pnl
            equity.append(capital)
            
            trades.append({
                'entry_time': entry_time,
                'exit_time': exit_time,
                'position': position,
                'entry_spread': entry_spread,
                'exit_spread': mspread[-1],
                'pnl': pnl,
                'net_pnl': net_pnl,
                'stop_loss_hit': False
            })
        
        # Calculate win rate
        win_rate = win_count / trade_count if trade_count > 0 else 0
        
        # Calculate trading frequency (trades per minute)
        trading_frequency = trade_count / len(mspread) if len(mspread) > 0 else 0
        
        # Calculate cumulative yield (%)
        initial_capital = equity[0]
        cumulative_yield = ((capital - initial_capital) / initial_capital) * 100 if initial_capital > 0 else 0
        
        # Update object's state
        self.trades.extend(trades)
        self.equity_curve.extend(equity[1:])  # Skip the first element to avoid duplication
        self.current_equity = capital
        
        return {
            'final_capital': capital,
            'cumulative_yield': cumulative_yield,
            'win_rate': win_rate,
            'trade_count': trade_count,
            'trading_frequency': trading_frequency,
            'commission_paid': commission_paid,
            'trades': trades,
            'equity_curve': equity
        }


def simulate_futures_data(n_days=30, n_minutes_per_day=240, trend_type='stable',
                          price_x_start=2000, price_y_start=2100,
                          volatility=0.001, cointegration_strength=0.9):
    """
    Simulate futures price data with cointegration
    
    Parameters:
    -----------
    n_days : int
        Number of days to simulate
    n_minutes_per_day : int
        Number of minutes per trading day
    trend_type : str
        Type of market trend ('up', 'down', 'stable')
    price_x_start : float
        Starting price of contract X
    price_y_start : float
        Starting price of contract Y
    volatility : float
        Volatility of price changes
    cointegration_strength : float
        Strength of cointegration (0-1)
        
    Returns:
    --------
    tuple : (prices_x, prices_y, dates)
    """
    # Force a minimum number of periods to ensure enough data
    min_periods = 500
    
    # Calculate how many minutes we need per day to meet min_periods
    required_minutes_per_day = min_periods // n_days + 1
    
    # Use whichever is larger
    actual_minutes_per_day = max(n_minutes_per_day, required_minutes_per_day)
    
    print(f"Using {actual_minutes_per_day} minutes per day to generate sufficient data")
    
    # Generate a fixed number of periods instead of filtering by trading hours
    n_periods = n_days * actual_minutes_per_day
    
    # Generate dates (without filtering)
    dates = []
    start_date = datetime.now() - timedelta(days=n_days)
    for day in range(n_days):
        current_date = start_date + timedelta(days=day)
        for minute in range(actual_minutes_per_day):
            dates.append(current_date + timedelta(minutes=minute))
    
    # Initialize arrays with the correct size
    prices_x = np.zeros(n_periods)
    prices_y = np.zeros(n_periods)
    
    # Set initial price
    prices_x[0] = price_x_start
    
    # Generate innovations for X
    innovations_x = np.random.normal(0, volatility, n_periods)
    
    # Add trend component based on trend_type
    if trend_type == 'up':
        trend = np.linspace(0, 0.2, n_periods)  # Upward trend
    elif trend_type == 'down':
        trend = np.linspace(0, -0.2, n_periods)  # Downward trend
    else:  # 'stable'
        trend = np.zeros(n_periods)  # No trend
    
    # Generate price series for X
    for i in range(1, n_periods):
        prices_x[i] = prices_x[i-1] * (1 + innovations_x[i] + trend[i] / n_periods)
    
    # Generate Y based on cointegration with X
    # Y = beta0 + beta1 * X + error
    beta0 = price_y_start - price_x_start
    beta1 = 1.0
    
    # Generate stationary error process for cointegration
    error_volatility = volatility * (1 - cointegration_strength) * price_x_start
    errors = np.zeros(n_periods)
    
    # AR(1) process for errors
    ar_param = 0.8  # Autoregressive parameter
    errors[0] = np.random.normal(0, error_volatility)
    for i in range(1, n_periods):
        errors[i] = ar_param * errors[i-1] + np.random.normal(0, error_volatility)
    
    # Generate price series for Y
    prices_y = beta0 + beta1 * prices_x + errors
    
    print(f"Generated {len(prices_x)} periods of data")
    return prices_x, prices_y, dates


def test_tgarch_wnn_model(commission_levels=[8, 10, 12], 
                          trend_types=['up', 'down', 'stable']):
    """
    Test the TGARCH-WNN model with different commission levels
    and market trends
    
    Parameters:
    -----------
    commission_levels : list of float
        Commission levels to test
    trend_types : list of str
        Market trend types to test
        
    Returns:
    --------
    dict : Test results
    """
    results = {}
    
    for trend in trend_types:
        print(f"\nTesting {trend} trend...")
        results[trend] = {}
        
        # Generate data for this trend - ensure enough data
        prices_x, prices_y, dates = simulate_futures_data(
            n_days=10,  # 10 days
            n_minutes_per_day=100,  # 100 minutes per day (minimum)
            trend_type=trend,
            volatility=0.002  # Increase volatility for better cointegration testing
        )
        
        # Print data shape to verify
        print(f"Generated data shape: {prices_x.shape}")
        
        # Split data into training and testing
        split_idx = len(prices_x) // 2
        train_x, test_x = prices_x[:split_idx], prices_x[split_idx:]
        train_y, test_y = prices_y[:split_idx], prices_y[split_idx:]
        
        print(f"Training set: {len(train_x)} samples")
        print(f"Test set: {len(test_x)} samples")
        
        # Test for cointegration before proceeding
        print("Testing for cointegration...")
        test_model = TGARCH_WNN_Arbitrage()
        if not test_model.check_cointegration(train_x, train_y):
            print(f"WARNING: Data for {trend} trend is not cointegrated. Skipping.")
            continue
        
        # Prepare daily splits for testing (3 periods per day)
        test_size = len(test_x)
        # Each day will be approximately test_size / 5 (assuming 5 test days)
        day_size = test_size // 5
        # Each period will be day_size / 3 (3 periods per day)
        period_size = day_size // 3
        
        # Ensure period_size is at least 30 (for statistical significance)
        period_size = max(30, period_size)
        day_size = period_size * 3
        n_test_days = test_size // day_size
        
        print(f"Day size: {day_size}, Period size: {period_size}, Test days: {n_test_days}")
        
        # Prepare arrays to store historical optimal thresholds
        k1_history = []
        k2_history = []
        
        for commission in commission_levels:
            print(f"\nTesting commission level: {commission}")
            # Initialize models for this commission level
            # 1. Historical Optimal (HO) model
            ho_model = TGARCH_WNN_Arbitrage(commission_per_lot=commission)
            
            # 2. Backpropagation Neural Network (BP) model
            bp_model = TGARCH_WNN_Arbitrage(commission_per_lot=commission)
            
            # 3. Wavelet Neural Network (WNN) model
            wnn_model = TGARCH_WNN_Arbitrage(commission_per_lot=commission)
            
            # Fit TGARCH model on training data
            print("Fitting TGARCH models...")
            try:
                fit_success_ho = ho_model.fit_tgarch(train_x, train_y, market_trend=trend)
                fit_success_bp = bp_model.fit_tgarch(train_x, train_y, market_trend=trend)
                fit_success_wnn = wnn_model.fit_tgarch(train_x, train_y, market_trend=trend)
                
                if not (fit_success_ho and fit_success_bp and fit_success_wnn):
                    print("TGARCH fitting failed. Skipping this trend.")
                    continue
            except Exception as e:
                print(f"Error fitting TGARCH models: {str(e)}")
                continue
            
            # Collect results for each model
            ho_results = []
            bp_results = []
            wnn_results = []
            
            # If this is the first commission level tested, initialize k history
            if len(k1_history) == 0:
                print("Initializing threshold history...")
                try:
                    # Initialize with some training data thresholds
                    # We'll calculate optimal thresholds for each training period
                    num_init_periods = min(6, len(train_x) // period_size)
                    for i in range(num_init_periods):
                        start_idx = i * period_size
                        end_idx = start_idx + period_size
                        
                        period_train_x = train_x[start_idx:end_idx]
                        period_train_y = train_y[start_idx:end_idx]
                        
                        # Calculate spread
                        spread = period_train_y - (ho_model.beta0 + ho_model.beta1 * period_train_x)
                        mspread = spread - ho_model.mu
                        
                        # Predict volatility
                        sigma = ho_model.predict_volatility(mspread)
                        
                        # Find optimal thresholds
                        k1, k2 = ho_model.compute_optimal_thresholds(
                            period_train_x, period_train_y, sigma
                        )
                        
                        k1_history.append(k1)
                        k2_history.append(k2)
                    
                    print(f"Initial thresholds: {k1_history}, {k2_history}")
                except Exception as e:
                    print(f"Error initializing threshold history: {str(e)}")
                    # If we can't initialize, use default values
                    k1_history = [1.0] * 6
                    k2_history = [1.0] * 6
            
            # Train BP and WNN models on threshold history
            print("Training neural network models...")
            try:
                if len(k1_history) >= 6:  # Need at least 6 for training
                    bp_k1_model, bp_k2_model = bp_model.train_bp_nn(
                        np.array(k1_history), np.array(k2_history)
                    )
                    
                    # Train WNN models
                    wnn_k1_model, wnn_k2_model = wnn_model.train_wnn(
                        np.array(k1_history), np.array(k2_history)
                    )
                else:
                    print("Not enough threshold history. Using default models.")
                    bp_k1_model, bp_k2_model = None, None
                    wnn_k1_model, wnn_k2_model = None, None
            except Exception as e:
                print(f"Error training neural networks: {str(e)}")
                bp_k1_model, bp_k2_model = None, None
                wnn_k1_model, wnn_k2_model = None, None
            
            # Test on each period
            print("Running backtests...")
            try:
                for day in range(n_test_days):
                    for period in range(3):  # 3 periods per day
                        start_idx = day * day_size + period * period_size
                        end_idx = min(start_idx + period_size, len(test_x))
                        
                        if start_idx >= end_idx or end_idx - start_idx < 30:
                            continue
                        
                        period_x = test_x[start_idx:end_idx]
                        period_y = test_y[start_idx:end_idx]
                        
                        # Calculate spread and volatility
                        spread, mspread = ho_model.calculate_spread(period_x, period_y)
                        sigma = ho_model.predict_volatility(mspread)
                        
                        # 1. Historical Optimal approach - use previous period's optimal
                        if len(k1_history) >= 1:  # If we have at least one previous threshold
                            ho_k1 = k1_history[-1]
                            ho_k2 = k2_history[-1]
                        else:
                            # Default if not enough history
                            ho_k1 = ho_k2 = 1.0
                        
                        ho_result = ho_model.run_arbitrage(
                            period_x, period_y, ho_k1, ho_k2
                        )
                        ho_results.append(ho_result)
                        
                        # 2. BP Neural Network approach
                        if bp_k1_model is not None and len(k1_history) >= 6:
                            # Predict
                            bp_k1, bp_k2 = bp_model.predict_thresholds_bp(
                                np.array(k1_history), 
                                np.array(k2_history),
                                bp_k1_model, 
                                bp_k2_model
                            )
                        else:
                            bp_k1 = bp_k2 = 1.0
                        
                        bp_result = bp_model.run_arbitrage(
                            period_x, period_y, bp_k1, bp_k2
                        )
                        bp_results.append(bp_result)
                        
                        # 3. Wavelet Neural Network approach
                        if wnn_k1_model is not None and len(k1_history) >= 6:
                            # Set the WNN models
                            wnn_model.wnn_model_k1 = wnn_k1_model
                            wnn_model.wnn_model_k2 = wnn_k2_model
                            
                            # Predict
                            wnn_k1, wnn_k2 = wnn_model.predict_thresholds_wnn(
                                np.array(k1_history), 
                                np.array(k2_history)
                            )
                        else:
                            wnn_k1 = wnn_k2 = 1.0
                        
                        wnn_result = wnn_model.run_arbitrage(
                            period_x, period_y, wnn_k1, wnn_k2
                        )
                        wnn_results.append(wnn_result)
                        
                        # Find optimal thresholds for this period (for next period prediction)
                        optimal_k1, optimal_k2 = ho_model.compute_optimal_thresholds(
                            period_x, period_y, sigma
                        )
                        
                        # Add to history
                        k1_history.append(optimal_k1)
                        k2_history.append(optimal_k2)
                        
                        # If we have accumulated enough history, retrain the models
                        if len(k1_history) % 6 == 0 and len(k1_history) >= 12:
                            bp_k1_model, bp_k2_model = bp_model.train_bp_nn(
                                np.array(k1_history), np.array(k2_history)
                            )
                            
                            wnn_k1_model, wnn_k2_model = wnn_model.train_wnn(
                                np.array(k1_history), np.array(k2_history)
                            )
            except Exception as e:
                print(f"Error during backtesting: {str(e)}")
            
            # Aggregate results for this commission level
            if not ho_results or not bp_results or not wnn_results:
                print("No results for some models. Skipping this commission level.")
                continue
            
            results[trend][commission] = {
                'HO': {
                    'cumulative_yield': ho_model.current_equity,
                    'win_rate': np.mean([r['win_rate'] for r in ho_results if r['trade_count'] > 0] or [0]),
                    'trade_count': sum([r['trade_count'] for r in ho_results]),
                    'trading_frequency': np.mean([r['trading_frequency'] for r in ho_results] or [0]),
                    'commission_paid': sum([r['commission_paid'] for r in ho_results])
                },
                'BP': {
                    'cumulative_yield': bp_model.current_equity,
                    'win_rate': np.mean([r['win_rate'] for r in bp_results if r['trade_count'] > 0] or [0]),
                    'trade_count': sum([r['trade_count'] for r in bp_results]),
                    'trading_frequency': np.mean([r['trading_frequency'] for r in bp_results] or [0]),
                    'commission_paid': sum([r['commission_paid'] for r in bp_results])
                },
                'WNN': {
                    'cumulative_yield': wnn_model.current_equity,
                    'win_rate': np.mean([r['win_rate'] for r in wnn_results if r['trade_count'] > 0] or [0]),
                    'trade_count': sum([r['trade_count'] for r in wnn_results]),
                    'trading_frequency': np.mean([r['trading_frequency'] for r in wnn_results] or [0]),
                    'commission_paid': sum([r['commission_paid'] for r in wnn_results])
                }
            }
            
            print(f"Results for commission level {commission}:")
            print(f"  HO: {results[trend][commission]['HO']['cumulative_yield']:.2f}")
            print(f"  BP: {results[trend][commission]['BP']['cumulative_yield']:.2f}")
            print(f"  WNN: {results[trend][commission]['WNN']['cumulative_yield']:.2f}")
    
    return results


def plot_results(results):
    """
    Plot the test results
    
    Parameters:
    -----------
    results : dict
        Test results from test_tgarch_wnn_model
    """
    trend_types = list(results.keys())
    
    if not trend_types:
        print("No results to plot!")
        return
        
    commission_levels = []
    for trend in trend_types:
        commission_levels.extend(list(results[trend].keys()))
    commission_levels = sorted(list(set(commission_levels)))
    
    if not commission_levels:
        print("No commission levels to plot!")
        return
        
    models = ['HO', 'BP', 'WNN']
    
    # Set up figure
    plt.figure(figsize=(18, 15))
    
    # Plot cumulative yield for each trend type and commission level
    for i, trend in enumerate(trend_types):
        plt.subplot(len(trend_types), 1, i+1)
        
        for model in models:
            yields = []
            commissions = []
            
            for comm in commission_levels:
                if comm in results[trend] and model in results[trend][comm]:
                    yields.append(results[trend][comm][model]['cumulative_yield'])
                    commissions.append(comm)
            
            if yields:
                plt.plot(commissions, yields, marker='o', label=model)
        
        plt.title(f'Cumulative Yield vs Commission Level - {trend.capitalize()} Trend')
        plt.xlabel('Commission Level (yuan per lot)')
        plt.ylabel('Cumulative Yield')
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Plot trading frequency for each trend type and commission level
    plt.figure(figsize=(18, 15))
    
    for i, trend in enumerate(trend_types):
        plt.subplot(len(trend_types), 1, i+1)
        
        for model in models:
            freq = []
            commissions = []
            
            for comm in commission_levels:
                if comm in results[trend] and model in results[trend][comm]:
                    freq.append(results[trend][comm][model]['trading_frequency'])
                    commissions.append(comm)
            
            if freq:
                plt.plot(commissions, freq, marker='o', label=model)
        
        plt.title(f'Trading Frequency vs Commission Level - {trend.capitalize()} Trend')
        plt.xlabel('Commission Level (yuan per lot)')
        plt.ylabel('Trading Frequency (per minute)')
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Plot win rate for each trend type and commission level
    plt.figure(figsize=(18, 15))
    
    for i, trend in enumerate(trend_types):
        plt.subplot(len(trend_types), 1, i+1)
        
        for model in models:
            win_rates = []
            commissions = []
            
            for comm in commission_levels:
                if comm in results[trend] and model in results[trend][comm]:
                    win_rates.append(results[trend][comm][model]['win_rate'] * 100)
                    commissions.append(comm)
            
            if win_rates:
                plt.plot(commissions, win_rates, marker='o', label=model)
        
        plt.title(f'Win Rate vs Commission Level - {trend.capitalize()} Trend')
        plt.xlabel('Commission Level (yuan per lot)')
        plt.ylabel('Win Rate (%)')
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    plt.show()


# Run a simplified test first to ensure everything works
test_results = test_tgarch_wnn_model(
    commission_levels=[10],
    trend_types=['stable']
)

# If that works, then run the full test
if test_results:
    print("\nSimplified test successful! Running full test...")
    results = test_tgarch_wnn_model(
        commission_levels=[8, 10, 12],
        trend_types=['up', 'down', 'stable']
    )
    
    # Plot the results
    plot_results(results)
    
    # Print summary
    print("\nSummary of Results:")
    print("===================")
    
    for trend in results:
        print(f"\n{trend.upper()} TREND")
        print("-" * len(f"{trend.upper()} TREND"))
        
        for commission in results[trend]:
            print(f"\nCommission Level: {commission} yuan per lot")
            
            for model in ['HO', 'BP', 'WNN']:
                if model in results[trend][commission]:
                    model_results = results[trend][commission][model]
                    print(f"  {model} Model:")
                    print(f"    Cumulative Yield: {model_results['cumulative_yield']:.2f}")
                    print(f"    Win Rate: {model_results['win_rate']*100:.2f}%")
                    print(f"    Trade Count: {model_results['trade_count']}")
                    print(f"    Trading Frequency: {model_results['trading_frequency']:.4f} per minute")
                    print(f"    Commission Paid: {model_results['commission_paid']:.2f} yuan")
else:
    print("Simplified test failed. Please check the errors above.")