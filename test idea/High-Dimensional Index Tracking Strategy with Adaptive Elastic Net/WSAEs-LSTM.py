import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pywt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_percentage_error
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from datetime import datetime
import random
import math
import warnings
warnings.filterwarnings('ignore')

# Set seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
random.seed(42)

class WaveletDenoiser:
    """
    Wavelet transform for denoising financial time series
    """
    def __init__(self, wavelet='haar', level=2):
        self.wavelet = wavelet
        self.level = level
    
    def denoise(self, data):
        """
        Denoise the input time series using wavelet transform
        
        Parameters:
        -----------
        data : numpy.ndarray
            Input time series data
            
        Returns:
        --------
        numpy.ndarray
            Denoised time series data
        """
        # Apply wavelet decomposition
        coeffs = pywt.wavedec(data, self.wavelet, level=self.level)
        
        # Apply thresholding to detail coefficients
        for i in range(1, len(coeffs)):
            coeffs[i] = pywt.threshold(coeffs[i], np.std(coeffs[i])/2, mode='soft')
        
        # Reconstruct the signal
        denoised_data = pywt.waverec(coeffs, self.wavelet)
        
        # Adjust the length to match the original data
        if len(denoised_data) > len(data):
            denoised_data = denoised_data[:len(data)]
        
        return denoised_data

class Autoencoder(nn.Module):
    """
    Single layer autoencoder
    """
    def __init__(self, input_dim, hidden_dim, activation='sigmoid'):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, input_dim)
        
        if activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            raise ValueError(f"Activation {activation} not supported")
    
    def forward(self, x):
        encoded = self.activation(self.encoder(x))
        decoded = self.activation(self.decoder(encoded))
        return encoded, decoded

class StackedAutoencoder:
    """
    Stacked Autoencoder for feature extraction
    """
    def __init__(self, input_dim, hidden_dims=[10], activation='sigmoid', 
                 learning_rate=0.001, epochs=100, batch_size=32, device='cuda'):
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.activation = activation
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.autoencoders = []
    
    def train(self, data):
        """
        Train the stacked autoencoder
        
        Parameters:
        -----------
        data : numpy.ndarray
            Input data for training
            
        Returns:
        --------
        numpy.ndarray
            Encoded features
        """
        current_input = torch.FloatTensor(data).to(self.device)
        
        # Build encoder layers one by one
        for i, hidden_dim in enumerate(self.hidden_dims):
            print(f"Training autoencoder layer {i+1}/{len(self.hidden_dims)}")
            
            # Create autoencoder
            autoencoder = Autoencoder(
                input_dim=current_input.shape[1],
                hidden_dim=hidden_dim,
                activation=self.activation
            ).to(self.device)
            
            # Create optimizer
            optimizer = optim.Adam(autoencoder.parameters(), lr=self.learning_rate)
            
            # Create DataLoader
            dataset = torch.utils.data.TensorDataset(current_input, current_input)
            dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
            
            # Train autoencoder
            for epoch in range(self.epochs):
                epoch_loss = 0
                for batch_x, batch_y in dataloader:
                    # Forward pass
                    encoded, decoded = autoencoder(batch_x)
                    
                    # Calculate loss
                    loss = F.mse_loss(decoded, batch_y)
                    
                    # Backward pass and optimize
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                
                if (epoch + 1) % 20 == 0:
                    print(f'Layer {i+1}, Epoch [{epoch+1}/{self.epochs}], Loss: {epoch_loss/len(dataloader):.6f}')
            
            # Save trained autoencoder
            self.autoencoders.append(autoencoder)
            
            # Get encoded features for next layer
            with torch.no_grad():
                encoded_features, _ = autoencoder(current_input)
                current_input = encoded_features
        
        # Return the final encoded features
        return current_input.cpu().numpy()
    
    def encode(self, data):
        """
        Encode input data using trained autoencoders
        
        Parameters:
        -----------
        data : numpy.ndarray
            Input data for encoding
            
        Returns:
        --------
        numpy.ndarray
            Encoded features
        """
        if not self.autoencoders:
            raise ValueError("Autoencoders not trained yet. Call train() first.")
        
        # Convert to tensor
        current_input = torch.FloatTensor(data).to(self.device)
        
        # Encode through each autoencoder
        with torch.no_grad():
            for autoencoder in self.autoencoders:
                encoded, _ = autoencoder(current_input)
                current_input = encoded
        
        return current_input.cpu().numpy()

class LSTMModel(nn.Module):
    """
    LSTM model for time series prediction
    """
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim=1):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        # Initialize hidden state and cell state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        
        return out

class SimpleRNNModel(nn.Module):
    """
    Simple RNN model for time series prediction (for comparison)
    """
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim=1):
        super(SimpleRNNModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.rnn = nn.RNN(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        
        # Forward propagate RNN
        out, _ = self.rnn(x, h0)
        
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        
        return out

class TimeSeriesDataset(Dataset):
    """
    Dataset for time series with lookback window
    """
    def __init__(self, X, y, lookback):
        self.X = X
        self.y = y
        self.lookback = lookback
        
    def __len__(self):
        return len(self.X) - self.lookback
    
    def __getitem__(self, idx):
        return (
            self.X[idx:idx+self.lookback], 
            self.y[idx+self.lookback]
        )

class WSAEsLSTM:
    """
    Wavelet Stacked Autoencoders Long Short-Term Memory model
    """
    def __init__(self, input_dim, hidden_dims=[10], lstm_hidden_dim=50, lstm_layers=1, 
                 wavelet='haar', wavelet_level=2, lookback=10, activation='sigmoid', 
                 learning_rate=0.001, epochs=100, batch_size=32, device='cuda'):
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.lstm_hidden_dim = lstm_hidden_dim
        self.lstm_layers = lstm_layers
        self.lookback = lookback
        self.wavelet = wavelet
        self.wavelet_level = wavelet_level
        self.activation = activation
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device if torch.cuda.is_available() else 'cpu'
        
        # Initialize components
        self.denoiser = WaveletDenoiser(wavelet=wavelet, level=wavelet_level)
        self.autoencoder = StackedAutoencoder(
            input_dim=input_dim, 
            hidden_dims=hidden_dims,
            activation=activation,
            learning_rate=learning_rate,
            epochs=epochs,
            batch_size=batch_size,
            device=self.device
        )
        self.lstm_model = None
        self.scaler_X = MinMaxScaler()
        self.scaler_y = MinMaxScaler()
        
        print(f"Using device: {self.device}")
    
    def fit(self, X, y):
        """
        Fit the WSAEs-LSTM model
        
        Parameters:
        -----------
        X : numpy.ndarray
            Input features
        y : numpy.ndarray
            Target values
        """
        # Step 1: Normalize data
        X_scaled = self.scaler_X.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y.reshape(-1, 1))
        
        # Step 2: Denoise each feature column
        X_denoised = np.zeros_like(X_scaled)
        for i in range(X_scaled.shape[1]):
            X_denoised[:, i] = self.denoiser.denoise(X_scaled[:, i])
        
        # Step 3: Extract features using stacked autoencoders
        print("Training stacked autoencoders...")
        X_encoded = self.autoencoder.train(X_denoised)
        
        # Step 4: Prepare data for LSTM
        dataset = TimeSeriesDataset(X_encoded, y_scaled, self.lookback)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # Step 5: Build and train LSTM model
        print("Training LSTM model...")
        encoded_dim = self.hidden_dims[-1]
        self.lstm_model = LSTMModel(
            input_dim=encoded_dim,
            hidden_dim=self.lstm_hidden_dim,
            num_layers=self.lstm_layers,
            output_dim=1
        ).to(self.device)
        
        # Create optimizer and loss function
        optimizer = optim.Adam(self.lstm_model.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()
        
        # Train LSTM model
        for epoch in range(self.epochs):
            epoch_loss = 0
            for batch_X, batch_y in dataloader:
                # Move data to device
                batch_X = batch_X.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                
                # Forward pass
                outputs = self.lstm_model(batch_X)
                loss = criterion(outputs, batch_y)
                
                # Backward pass and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            if (epoch + 1) % 20 == 0:
                print(f'LSTM Epoch [{epoch+1}/{self.epochs}], Loss: {epoch_loss/len(dataloader):.6f}')
    
    def predict(self, X):
        """
        Predict using the trained WSAEs-LSTM model
        
        Parameters:
        -----------
        X : numpy.ndarray
            Input features
            
        Returns:
        --------
        numpy.ndarray
            Predicted values
        """
        # Step 1: Normalize data
        X_scaled = self.scaler_X.transform(X)
        
        # Step 2: Denoise each feature column
        X_denoised = np.zeros_like(X_scaled)
        for i in range(X_scaled.shape[1]):
            X_denoised[:, i] = self.denoiser.denoise(X_scaled[:, i])
        
        # Step 3: Extract features using stacked autoencoders
        X_encoded = self.autoencoder.encode(X_denoised)
        
        # Step 4: Prepare data for LSTM prediction
        predictions = []
        
        for i in range(len(X_encoded) - self.lookback):
            # Get lookback window
            sequence = X_encoded[i:i+self.lookback]
            
            # Convert to tensor
            sequence = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)
            
            # Predict
            with torch.no_grad():
                prediction = self.lstm_model(sequence)
                prediction = prediction.cpu().numpy()
                
            # Inverse transform
            prediction = self.scaler_y.inverse_transform(prediction)
            predictions.append(prediction[0, 0])
        
        return np.array(predictions)

class WLSTM:
    """
    Wavelet Long Short-Term Memory model (comparison model)
    """
    def __init__(self, input_dim, lstm_hidden_dim=50, lstm_layers=1, 
                 wavelet='haar', wavelet_level=2, lookback=10, 
                 learning_rate=0.001, epochs=100, batch_size=32, device='cuda'):
        self.input_dim = input_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        self.lstm_layers = lstm_layers
        self.lookback = lookback
        self.wavelet = wavelet
        self.wavelet_level = wavelet_level
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device if torch.cuda.is_available() else 'cpu'
        
        # Initialize components
        self.denoiser = WaveletDenoiser(wavelet=wavelet, level=wavelet_level)
        self.lstm_model = None
        self.scaler_X = MinMaxScaler()
        self.scaler_y = MinMaxScaler()
    
    def fit(self, X, y):
        """
        Fit the WLSTM model
        
        Parameters:
        -----------
        X : numpy.ndarray
            Input features
        y : numpy.ndarray
            Target values
        """
        # Step 1: Normalize data
        X_scaled = self.scaler_X.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y.reshape(-1, 1))
        
        # Step 2: Denoise each feature column
        X_denoised = np.zeros_like(X_scaled)
        for i in range(X_scaled.shape[1]):
            X_denoised[:, i] = self.denoiser.denoise(X_scaled[:, i])
        
        # Step 3: Prepare data for LSTM
        dataset = TimeSeriesDataset(X_denoised, y_scaled, self.lookback)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # Step 4: Build and train LSTM model
        print("Training LSTM model...")
        self.lstm_model = LSTMModel(
            input_dim=self.input_dim,
            hidden_dim=self.lstm_hidden_dim,
            num_layers=self.lstm_layers,
            output_dim=1
        ).to(self.device)
        
        # Create optimizer and loss function
        optimizer = optim.Adam(self.lstm_model.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()
        
        # Train LSTM model
        for epoch in range(self.epochs):
            epoch_loss = 0
            for batch_X, batch_y in dataloader:
                # Move data to device
                batch_X = batch_X.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                
                # Forward pass
                outputs = self.lstm_model(batch_X)
                loss = criterion(outputs, batch_y)
                
                # Backward pass and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            if (epoch + 1) % 20 == 0:
                print(f'LSTM Epoch [{epoch+1}/{self.epochs}], Loss: {epoch_loss/len(dataloader):.6f}')
    
    def predict(self, X):
        """
        Predict using the trained WLSTM model
        
        Parameters:
        -----------
        X : numpy.ndarray
            Input features
            
        Returns:
        --------
        numpy.ndarray
            Predicted values
        """
        # Step 1: Normalize data
        X_scaled = self.scaler_X.transform(X)
        
        # Step 2: Denoise each feature column
        X_denoised = np.zeros_like(X_scaled)
        for i in range(X_scaled.shape[1]):
            X_denoised[:, i] = self.denoiser.denoise(X_scaled[:, i])
        
        # Step 3: Prepare data for LSTM prediction
        predictions = []
        
        for i in range(len(X_denoised) - self.lookback):
            # Get lookback window
            sequence = X_denoised[i:i+self.lookback]
            
            # Convert to tensor
            sequence = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)
            
            # Predict
            with torch.no_grad():
                prediction = self.lstm_model(sequence)
                prediction = prediction.cpu().numpy()
                
            # Inverse transform
            prediction = self.scaler_y.inverse_transform(prediction)
            predictions.append(prediction[0, 0])
        
        return np.array(predictions)

class SimpleLSTM:
    """
    Simple LSTM model (comparison model)
    """
    def __init__(self, input_dim, lstm_hidden_dim=50, lstm_layers=1, lookback=10, 
                 learning_rate=0.001, epochs=100, batch_size=32, device='cuda'):
        self.input_dim = input_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        self.lstm_layers = lstm_layers
        self.lookback = lookback
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device if torch.cuda.is_available() else 'cpu'
        
        # Initialize components
        self.lstm_model = None
        self.scaler_X = MinMaxScaler()
        self.scaler_y = MinMaxScaler()
    
    def fit(self, X, y):
        """
        Fit the LSTM model
        
        Parameters:
        -----------
        X : numpy.ndarray
            Input features
        y : numpy.ndarray
            Target values
        """
        # Step 1: Normalize data
        X_scaled = self.scaler_X.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y.reshape(-1, 1))
        
        # Step 2: Prepare data for LSTM
        dataset = TimeSeriesDataset(X_scaled, y_scaled, self.lookback)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # Step 3: Build and train LSTM model
        print("Training LSTM model...")
        self.lstm_model = LSTMModel(
            input_dim=self.input_dim,
            hidden_dim=self.lstm_hidden_dim,
            num_layers=self.lstm_layers,
            output_dim=1
        ).to(self.device)
        
        # Create optimizer and loss function
        optimizer = optim.Adam(self.lstm_model.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()
        
        # Train LSTM model
        for epoch in range(self.epochs):
            epoch_loss = 0
            for batch_X, batch_y in dataloader:
                # Move data to device
                batch_X = batch_X.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                
                # Forward pass
                outputs = self.lstm_model(batch_X)
                loss = criterion(outputs, batch_y)
                
                # Backward pass and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            if (epoch + 1) % 20 == 0:
                print(f'LSTM Epoch [{epoch+1}/{self.epochs}], Loss: {epoch_loss/len(dataloader):.6f}')
    
    def predict(self, X):
        """
        Predict using the trained LSTM model
        
        Parameters:
        -----------
        X : numpy.ndarray
            Input features
            
        Returns:
        --------
        numpy.ndarray
            Predicted values
        """
        # Step 1: Normalize data
        X_scaled = self.scaler_X.transform(X)
        
        # Step 2: Prepare data for LSTM prediction
        predictions = []
        
        for i in range(len(X_scaled) - self.lookback):
            # Get lookback window
            sequence = X_scaled[i:i+self.lookback]
            
            # Convert to tensor
            sequence = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)
            
            # Predict
            with torch.no_grad():
                prediction = self.lstm_model(sequence)
                prediction = prediction.cpu().numpy()
                
            # Inverse transform
            prediction = self.scaler_y.inverse_transform(prediction)
            predictions.append(prediction[0, 0])
        
        return np.array(predictions)

class SimpleRNN:
    """
    Simple RNN model (comparison model)
    """
    def __init__(self, input_dim, rnn_hidden_dim=50, rnn_layers=1, lookback=10, 
                 learning_rate=0.001, epochs=100, batch_size=32, device='cuda'):
        self.input_dim = input_dim
        self.rnn_hidden_dim = rnn_hidden_dim
        self.rnn_layers = rnn_layers
        self.lookback = lookback
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device if torch.cuda.is_available() else 'cpu'
        
        # Initialize components
        self.rnn_model = None
        self.scaler_X = MinMaxScaler()
        self.scaler_y = MinMaxScaler()
    
    def fit(self, X, y):
        """
        Fit the RNN model
        
        Parameters:
        -----------
        X : numpy.ndarray
            Input features
        y : numpy.ndarray
            Target values
        """
        # Step 1: Normalize data
        X_scaled = self.scaler_X.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y.reshape(-1, 1))
        
        # Step 2: Prepare data for RNN
        dataset = TimeSeriesDataset(X_scaled, y_scaled, self.lookback)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # Step 3: Build and train RNN model
        print("Training RNN model...")
        self.rnn_model = SimpleRNNModel(
            input_dim=self.input_dim,
            hidden_dim=self.rnn_hidden_dim,
            num_layers=self.rnn_layers,
            output_dim=1
        ).to(self.device)
        
        # Create optimizer and loss function
        optimizer = optim.Adam(self.rnn_model.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()
        
        # Train RNN model
        for epoch in range(self.epochs):
            epoch_loss = 0
            for batch_X, batch_y in dataloader:
                # Move data to device
                batch_X = batch_X.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                
                # Forward pass
                outputs = self.rnn_model(batch_X)
                loss = criterion(outputs, batch_y)
                
                # Backward pass and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            if (epoch + 1) % 20 == 0:
                print(f'RNN Epoch [{epoch+1}/{self.epochs}], Loss: {epoch_loss/len(dataloader):.6f}')
    
    def predict(self, X):
        """
        Predict using the trained RNN model
        
        Parameters:
        -----------
        X : numpy.ndarray
            Input features
            
        Returns:
        --------
        numpy.ndarray
            Predicted values
        """
        # Step 1: Normalize data
        X_scaled = self.scaler_X.transform(X)
        
        # Step 2: Prepare data for RNN prediction
        predictions = []
        
        for i in range(len(X_scaled) - self.lookback):
            # Get lookback window
            sequence = X_scaled[i:i+self.lookback]
            
            # Convert to tensor
            sequence = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)
            
            # Predict
            with torch.no_grad():
                prediction = self.rnn_model(sequence)
                prediction = prediction.cpu().numpy()
                
            # Inverse transform
            prediction = self.scaler_y.inverse_transform(prediction)
            predictions.append(prediction[0, 0])
        
        return np.array(predictions)

def calculate_metrics(y_true, y_pred):
    """
    Calculate evaluation metrics
    
    Parameters:
    -----------
    y_true : numpy.ndarray
        True values
    y_pred : numpy.ndarray
        Predicted values
        
    Returns:
    --------
    dict
        Dictionary containing evaluation metrics
    """
    # Mean Absolute Percentage Error (MAPE)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    # Correlation coefficient (R)
    correlation = np.corrcoef(y_true, y_pred)[0, 1]
    
    # Theil's inequality coefficient (Theil U)
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    theil_u = rmse / (np.sqrt(np.mean(y_true ** 2)) + np.sqrt(np.mean(y_pred ** 2)))
    
    return {
        'MAPE': mape,
        'R': correlation,
        'Theil U': theil_u
    }

def calculate_trading_returns(y_true, y_pred, transaction_cost=0.0001):
    """
    Calculate trading returns using a buy-and-sell strategy
    
    Parameters:
    -----------
    y_true : numpy.ndarray
        True values
    y_pred : numpy.ndarray
        Predicted values
    transaction_cost : float
        Transaction cost as a percentage
        
    Returns:
    --------
    float
        Trading returns as a percentage
    """
    # Initialize variables
    buy_signals = 0
    sell_signals = 0
    position = 0  # 0: no position, 1: long, -1: short
    capital = 1.0  # Initial capital
    
    for i in range(len(y_pred) - 1):
        # Generate trading signals
        if y_pred[i+1] > y_true[i]:  # Predicted next price > current actual price
            signal = 1  # Buy signal
        else:
            signal = -1  # Sell signal
        
        # Execute trades
        if signal == 1 and position != 1:  # Buy signal and not already in long position
            if position == -1:  # Close short position
                capital = capital * (2 - y_true[i] / y_true[i-1] - transaction_cost)
                position = 0
            
            # Open long position
            position = 1
            capital = capital * (1 - transaction_cost)
            buy_signals += 1
        
        elif signal == -1 and position != -1:  # Sell signal and not already in short position
            if position == 1:  # Close long position
                capital = capital * (y_true[i] / y_true[i-1] - transaction_cost)
                position = 0
            
            # Open short position
            position = -1
            capital = capital * (1 - transaction_cost)
            sell_signals += 1
    
    # Close final position at the end
    if position == 1:
        capital = capital * (y_true[-1] / y_true[-2] - transaction_cost)
    elif position == -1:
        capital = capital * (2 - y_true[-1] / y_true[-2] - transaction_cost)
    
    # Calculate return as percentage
    returns = (capital - 1.0) * 100
    
    print(f"Buy signals: {buy_signals}, Sell signals: {sell_signals}")
    
    return returns

def simulate_stock_data(n_samples=1000, n_features=10):
    """
    Simulate stock price data with technical indicators and macroeconomic variables
    
    Parameters:
    -----------
    n_samples : int
        Number of samples to generate
    n_features : int
        Base number of features to generate
        
    Returns:
    --------
    tuple
        (X, y) where X is the feature matrix and y is the target vector
    """
    # Generate price data with enough points for calculating indicators
    total_points = n_samples + 30  # Add buffer
    
    # Generate base price series
    price = 100.0
    prices = []
    for _ in range(total_points):
        # Random price change
        change_percent = np.random.normal(0, 0.01)  # mean=0, std=1%
        price *= (1 + change_percent)
        prices.append(price)
    
    prices = np.array(prices)
    
    # Create OHLC data
    open_prices = prices[:-1]
    close_prices = prices[1:]
    high_prices = close_prices * (1 + np.abs(np.random.normal(0, 0.005, size=len(close_prices))))
    low_prices = close_prices * (1 - np.abs(np.random.normal(0, 0.005, size=len(close_prices))))
    volume = np.random.randint(1000, 10000, size=len(close_prices))
    
    # Calculate technical indicators (adjusting lengths as needed)
    
    # Simple Moving Averages
    ma5 = np.zeros_like(close_prices)
    ma10 = np.zeros_like(close_prices)
    
    for i in range(4, len(close_prices)):
        ma5[i] = np.mean(close_prices[i-4:i+1])
    
    for i in range(9, len(close_prices)):
        ma10[i] = np.mean(close_prices[i-9:i+1])
    
    # Price Rate of Change
    roc = np.zeros_like(close_prices)
    for i in range(10, len(close_prices)):
        roc[i] = (close_prices[i] - close_prices[i-10]) / close_prices[i-10] * 100
    
    # Momentum
    momentum = np.zeros_like(close_prices)
    for i in range(6, len(close_prices)):
        momentum[i] = close_prices[i] - close_prices[i-6]
    
    # MACD
    ema12 = np.zeros_like(close_prices)
    ema26 = np.zeros_like(close_prices)
    
    # Initialize EMA with SMA
    ema12[11] = np.mean(close_prices[:12])
    ema26[25] = np.mean(close_prices[:26])
    
    # Calculate EMA
    k12 = 2 / (12 + 1)
    k26 = 2 / (26 + 1)
    
    for i in range(12, len(close_prices)):
        ema12[i] = close_prices[i] * k12 + ema12[i-1] * (1 - k12)
    
    for i in range(26, len(close_prices)):
        ema26[i] = close_prices[i] * k26 + ema26[i-1] * (1 - k26)
    
    macd = np.zeros_like(close_prices)
    for i in range(26, len(close_prices)):
        macd[i] = ema12[i] - ema26[i]
    
    # Generate macroeconomic variables
    exchange_rate = np.random.normal(1.0, 0.05, size=len(close_prices))
    interest_rate = np.random.normal(0.05, 0.01, size=len(close_prices))
    
    # Ensure all features have the same length by trimming to the valid data region
    start_idx = 26  # Based on the indicators with longest lookback
    
    # Create feature matrix
    features = []
    features.append(open_prices[start_idx:])     # Open price
    features.append(high_prices[start_idx:])     # High price
    features.append(low_prices[start_idx:])      # Low price
    features.append(close_prices[start_idx:])    # Close price
    features.append(volume[start_idx:])          # Volume
    features.append(ma5[start_idx:])             # 5-day MA
    features.append(ma10[start_idx:])            # 10-day MA
    features.append(roc[start_idx:])             # Price Rate of Change
    features.append(momentum[start_idx:])        # Momentum
    features.append(macd[start_idx:])            # MACD
    features.append(exchange_rate[start_idx:])   # Exchange rate
    features.append(interest_rate[start_idx:])   # Interest rate
    
    # Stack features horizontally and trim to requested number of samples
    X = np.column_stack(features)[:n_samples]
    
    # Target is the next day's closing price
    y = close_prices[start_idx+1:start_idx+n_samples+1]
    
    print(f"Generated data shapes: X:{X.shape}, y:{y.shape}")
    
    return X, y

def plot_predictions(y_true, y_pred_dict, title="Stock Price Predictions"):
    """
    Plot true values and predictions from different models
    
    Parameters:
    -----------
    y_true : numpy.ndarray
        True values
    y_pred_dict : dict
        Dictionary containing model names and their predictions
    title : str
        Plot title
    """
    plt.figure(figsize=(15, 7))
    
    # Plot true values
    plt.plot(y_true, label='True', linewidth=2)
    
    # Plot predictions for each model
    colors = ['r', 'g', 'b', 'c', 'm', 'y']
    for i, (model_name, y_pred) in enumerate(y_pred_dict.items()):
        plt.plot(y_pred, label=model_name, linestyle='--', linewidth=1, color=colors[i % len(colors)])
    
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def main():
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Generate simulated data
    print("Generating simulated stock data...")
    X, y = simulate_stock_data(n_samples=1000, n_features=10)
    
    # Split data into train and test sets
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    print(f"Training data shape: {X_train.shape}, {y_train.shape}")
    print(f"Testing data shape: {X_test.shape}, {y_test.shape}")
    
    # For faster demo, reduce epochs
    params = {
        'lookback': 10,
        'epochs': 50,  # Reduced for quick demo
        'batch_size': 32,
        'learning_rate': 0.001,
        'device': device
    }
    
    # Train and evaluate WSAEs-LSTM
    print("\n===== Training WSAEs-LSTM model =====")
    wsaes_lstm = WSAEsLSTM(
        input_dim=X_train.shape[1],
        hidden_dims=[8, 4],
        lstm_hidden_dim=50,
        lstm_layers=1,
        wavelet='haar',
        wavelet_level=2,
        **params
    )
    wsaes_lstm.fit(X_train, y_train)
    y_pred_wsaes_lstm = wsaes_lstm.predict(X_test)
    
    # Train and evaluate WLSTM
    print("\n===== Training WLSTM model =====")
    wlstm = WLSTM(
        input_dim=X_train.shape[1],
        lstm_hidden_dim=50,
        lstm_layers=1,
        wavelet='haar',
        wavelet_level=2,
        **params
    )
    wlstm.fit(X_train, y_train)
    y_pred_wlstm = wlstm.predict(X_test)
    
    # Train and evaluate LSTM
    print("\n===== Training LSTM model =====")
    lstm = SimpleLSTM(
        input_dim=X_train.shape[1],
        lstm_hidden_dim=50,
        lstm_layers=1,
        **params
    )
    lstm.fit(X_train, y_train)
    y_pred_lstm = lstm.predict(X_test)
    
    # Train and evaluate RNN
    print("\n===== Training RNN model =====")
    rnn = SimpleRNN(
        input_dim=X_train.shape[1],
        rnn_hidden_dim=50,
        rnn_layers=1,
        **params
    )
    rnn.fit(X_train, y_train)
    y_pred_rnn = rnn.predict(X_test)
    
    # Calculate metrics
    y_test_actual = y_test[params['lookback']:]  # Adjust test set to match predictions
    
    metrics_wsaes_lstm = calculate_metrics(y_test_actual, y_pred_wsaes_lstm)
    metrics_wlstm = calculate_metrics(y_test_actual, y_pred_wlstm)
    metrics_lstm = calculate_metrics(y_test_actual, y_pred_lstm)
    metrics_rnn = calculate_metrics(y_test_actual, y_pred_rnn)
    
    # Calculate trading returns
    returns_wsaes_lstm = calculate_trading_returns(y_test_actual, y_pred_wsaes_lstm)
    returns_wlstm = calculate_trading_returns(y_test_actual, y_pred_wlstm)
    returns_lstm = calculate_trading_returns(y_test_actual, y_pred_lstm)
    returns_rnn = calculate_trading_returns(y_test_actual, y_pred_rnn)
    
    # Buy and hold returns
    buy_and_hold_returns = (y_test_actual[-1] / y_test_actual[0] - 1) * 100
    
    # Print results
    print("\n===== Prediction Accuracy Results =====")
    print(f"{'Model':<15} {'MAPE':<10} {'R':<10} {'Theil U':<10}")
    print("-" * 45)
    print(f"WSAEs-LSTM      {metrics_wsaes_lstm['MAPE']:.4f}%    {metrics_wsaes_lstm['R']:.4f}     {metrics_wsaes_lstm['Theil U']:.4f}")
    print(f"WLSTM           {metrics_wlstm['MAPE']:.4f}%    {metrics_wlstm['R']:.4f}     {metrics_wlstm['Theil U']:.4f}")
    print(f"LSTM            {metrics_lstm['MAPE']:.4f}%    {metrics_lstm['R']:.4f}     {metrics_lstm['Theil U']:.4f}")
    print(f"RNN             {metrics_rnn['MAPE']:.4f}%    {metrics_rnn['R']:.4f}     {metrics_rnn['Theil U']:.4f}")
    
    print("\n===== Trading Returns Results =====")
    print(f"{'Model':<15} {'Returns':<10}")
    print("-" * 25)
    print(f"WSAEs-LSTM      {returns_wsaes_lstm:.2f}%")
    print(f"WLSTM           {returns_wlstm:.2f}%")
    print(f"LSTM            {returns_lstm:.2f}%")
    print(f"RNN             {returns_rnn:.2f}%")
    print(f"Buy and Hold    {buy_and_hold_returns:.2f}%")
    
    # Plot predictions
    plot_predictions(
        y_test_actual,
        {
            'WSAEs-LSTM': y_pred_wsaes_lstm,
            'WLSTM': y_pred_wlstm,
            'LSTM': y_pred_lstm,
            'RNN': y_pred_rnn
        },
        title="Stock Price Predictions"
    )

if __name__ == "__main__":
    main()