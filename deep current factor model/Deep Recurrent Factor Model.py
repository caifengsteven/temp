import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
from scipy import stats
import math
import random
from tqdm import tqdm

# For deep learning models with PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

# For model evaluation and comparison
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
random.seed(42)

class LSTMModel(nn.Module):
    """
    Bidirectional LSTM model for stock return prediction
    """
    def __init__(self, input_dim, hidden_dim, num_layers=1, dropout=0.0):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Bidirectional LSTM layer
        self.lstm = nn.LSTM(
            input_dim, 
            hidden_dim, 
            num_layers=num_layers, 
            batch_first=True, 
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Output layer
        self.fc = nn.Linear(hidden_dim * 2, 1)  # * 2 for bidirectional
    
    def forward(self, x):
        # Forward pass through LSTM layer
        # x shape: (batch_size, seq_len, input_dim)
        lstm_out, _ = self.lstm(x)
        
        # Use only the output from the last time step
        # lstm_out shape: (batch_size, seq_len, hidden_dim*2)
        last_output = lstm_out[:, -1, :]
        
        # Forward pass through linear layer
        out = self.fc(last_output)
        
        return out

class DNNModel(nn.Module):
    """
    Deep Neural Network model for comparison
    """
    def __init__(self, input_dim, hidden_dims=[80, 40]):
        super(DNNModel, self).__init__()
        
        # Create a list of layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, 1))
        
        # Combine all layers into a sequential model
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

class FeaturePermutationImportance:
    """
    Feature importance using permutation method.
    This is a model-agnostic method that doesn't require
    accessing model gradients.
    """
    def __init__(self, model, loss_fn=None):
        self.model = model
        self.model.eval()  # Set model to evaluation mode
        # Default to MSE loss if none provided
        self.loss_fn = loss_fn if loss_fn is not None else nn.MSELoss()
    
    def compute_importance(self, X, y, n_repeats=5):
        """
        Compute feature importance by permuting each feature
        and measuring the increase in prediction error
        
        Parameters:
        -----------
        X : torch.Tensor
            Input features (batch_size, seq_len, features)
        y : torch.Tensor 
            Target values (batch_size,)
        n_repeats : int
            Number of times to repeat permutation for each feature
            
        Returns:
        --------
        numpy.ndarray : Feature importance scores
                        Shape: (batch_size, seq_len, features)
        """
        with torch.no_grad():
            # Get baseline predictions and error
            baseline_preds = self.model(X)
            baseline_error = self.loss_fn(baseline_preds.squeeze(), y)
            
            # Initialize importance scores
            importance = torch.zeros_like(X)
            
            # For each feature in each time step
            for t in range(X.shape[1]):  # For each time step in sequence
                for f in range(X.shape[2]):  # For each feature
                    # Repeat the permutation multiple times to reduce variance
                    errors = []
                    for _ in range(n_repeats):
                        # Create a permuted version of X
                        X_permuted = X.clone()
                        
                        # Permute the feature values across samples
                        perm_idx = torch.randperm(X.shape[0])
                        X_permuted[:, t, f] = X_permuted[perm_idx, t, f]
                        
                        # Predict with permuted data
                        perm_preds = self.model(X_permuted)
                        
                        # Compute error
                        perm_error = self.loss_fn(perm_preds.squeeze(), y)
                        
                        # Feature importance is the increase in error
                        errors.append((perm_error - baseline_error).item())
                    
                    # Average the errors across repeats
                    avg_error_increase = torch.tensor(errors).mean()
                    
                    # Store importance score for all samples
                    importance[:, t, f] = avg_error_increase
                    
        return importance.cpu().numpy()

class DeepRecurrentFactorModel:
    """
    Implementation of Deep Recurrent Factor Model as described in the paper
    "Deep Recurrent Factor Model: Interpretable Non-Linear and Time-Varying Multi-Factor Model"
    """
    
    def __init__(self, sequence_length=5, hidden_dim=16, use_feature_importance=True, device='cpu'):
        """
        Initialize the Deep Recurrent Factor Model
        
        Parameters:
        -----------
        sequence_length : int
            Length of the sequence for LSTM (number of months to look back)
        hidden_dim : int
            Number of hidden units in LSTM
        use_feature_importance : bool
            Whether to use feature importance for interpretation
        device : str
            Device to use for PyTorch ('cpu' or 'cuda')
        """
        self.sequence_length = sequence_length
        self.hidden_dim = hidden_dim
        self.use_feature_importance = use_feature_importance
        self.device = torch.device(device if torch.cuda.is_available() and device == 'cuda' else 'cpu')
        
        # Model components
        self.lstm_model = None
        self.dnn_model = None
        self.feature_importance_analyzer = None
        self.X_scaler = StandardScaler()
        self.y_scaler = StandardScaler()
        
        # Data components
        self.factor_names = []
        self.descriptor_names = []
        
        # Benchmark models
        self.linear_model = LinearRegression()
        self.svr_model = SVR()
        self.rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        
        # Results
        self.predictions = {}
        self.factor_contributions = {}
    
    def _generate_sample_data(self, tickers, start_date, end_date):
        """Generate sample data for demonstration purposes"""
        # Convert dates to datetime
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        
        # Generate date range
        date_range = pd.date_range(start=start, end=end, freq='M')
        
        # Factors and descriptors from the paper
        self.factor_names = ['Risk', 'Quality', 'Momentum', 'Value', 'Size']
        self.descriptor_names = [
            # Risk
            '60VOL', 'BETA', 'SKEW',
            # Quality
            'ROE', 'ROA', 'ACCRUALS', 'LEVERAGE',
            # Momentum
            '12-1MOM', '1MOM', '60MOM',
            # Value
            'PSR', 'PER', 'PBR', 'PCFR',
            # Size
            'CAP', 'ILLIQ'
        ]
        
        # Create a dataframe to store all data
        all_data = {}
        
        for ticker in tickers:
            # Use ticker name to seed random number generator for consistency
            seed_value = hash(ticker) % 10000
            np.random.seed(seed_value)
            
            # Generate descriptor data with some time series properties
            ticker_data = pd.DataFrame(index=date_range)
            
            # Generate sample data for each descriptor
            for descriptor in self.descriptor_names:
                # Base level for the descriptor (different for each ticker and descriptor)
                base_level = np.random.normal(0, 1)
                
                # Time series with some autocorrelation
                ar_param = 0.8  # Autocorrelation parameter
                n = len(date_range)
                
                # Generate random shocks
                shocks = np.random.normal(0, 0.1, n)
                
                # Generate AR(1) process
                series = np.zeros(n)
                series[0] = base_level + shocks[0]
                
                for t in range(1, n):
                    series[t] = base_level + ar_param * (series[t-1] - base_level) + shocks[t]
                
                ticker_data[descriptor] = series
            
            # Generate returns with some relationship to the descriptors
            # (simple model for demonstration)
            returns = np.zeros(len(date_range))
            
            # Different weights for different factors to simulate real market
            weights = {
                'Risk': -0.1,  # Higher risk, lower returns
                'Quality': 0.2,  # Higher quality, higher returns
                'Momentum': 0.15,  # Momentum effect
                'Value': 0.25,  # Value effect
                'Size': -0.05  # Size effect (smaller companies outperform)
            }
            
            # Add time-varying and non-linear effects
            for t in range(len(date_range)):
                # Base signal from descriptors
                factor_signals = {factor: 0 for factor in self.factor_names}
                
                # Risk factor
                factor_signals['Risk'] = (
                    ticker_data['60VOL'].iloc[t] * 0.4 + 
                    ticker_data['BETA'].iloc[t] * 0.4 + 
                    ticker_data['SKEW'].iloc[t] * 0.2
                )
                
                # Quality factor
                factor_signals['Quality'] = (
                    ticker_data['ROE'].iloc[t] * 0.3 + 
                    ticker_data['ROA'].iloc[t] * 0.3 + 
                    ticker_data['ACCRUALS'].iloc[t] * 0.2 -
                    ticker_data['LEVERAGE'].iloc[t] * 0.2
                )
                
                # Momentum factor
                factor_signals['Momentum'] = (
                    ticker_data['12-1MOM'].iloc[t] * 0.5 + 
                    ticker_data['1MOM'].iloc[t] * 0.3 + 
                    ticker_data['60MOM'].iloc[t] * 0.2
                )
                
                # Value factor
                factor_signals['Value'] = (
                    1/ticker_data['PSR'].iloc[t] * 0.25 + 
                    1/ticker_data['PER'].iloc[t] * 0.25 + 
                    1/ticker_data['PBR'].iloc[t] * 0.25 + 
                    1/ticker_data['PCFR'].iloc[t] * 0.25
                )
                
                # Size factor
                factor_signals['Size'] = (
                    -ticker_data['CAP'].iloc[t] * 0.7 + 
                    ticker_data['ILLIQ'].iloc[t] * 0.3
                )
                
                # Time varying component: some factors matter more at different times
                time_weight = np.sin(t/12 * np.pi) * 0.2  # Cycles with ~12 month period
                quality_adjusted = weights['Quality'] * (1 + time_weight)
                value_adjusted = weights['Value'] * (1 - time_weight)
                
                # Combine signals (with some non-linearity)
                raw_signal = (
                    weights['Risk'] * factor_signals['Risk'] +
                    quality_adjusted * np.tanh(factor_signals['Quality']) +  # Non-linear effect
                    weights['Momentum'] * factor_signals['Momentum'] * (1 + 0.5 * factor_signals['Momentum']) +  # Non-linear
                    value_adjusted * factor_signals['Value'] +
                    weights['Size'] * factor_signals['Size']
                )
                
                # Add some noise
                returns[t] = raw_signal + np.random.normal(0, 0.02)
            
            # Convert to monthly percentage returns
            ticker_data['Returns'] = returns * 100
            
            all_data[ticker] = ticker_data
        
        return all_data
    
    def prepare_data(self, data_dict, test_ratio=0.2):
        """
        Prepare data for the model
        
        Parameters:
        -----------
        data_dict : dict
            Dictionary of dataframes with factor and return data
        test_ratio : float
            Ratio of data to use for testing
            
        Returns:
        --------
        X_train, X_test, y_train, y_test
        """
        # Combine all tickers into a single dataframe
        combined_data = []
        
        for ticker, df in data_dict.items():
            ticker_df = df.copy()
            ticker_df['Ticker'] = ticker
            combined_data.append(ticker_df)
        
        all_data = pd.concat(combined_data)
        
        # Create sequences for LSTM
        sequences = []
        returns = []
        tickers = []
        dates = []
        
        for ticker, group in all_data.groupby('Ticker'):
            sorted_group = group.sort_index()
            
            for i in range(len(sorted_group) - self.sequence_length):
                # Get sequence of descriptor values
                seq = sorted_group.iloc[i:i+self.sequence_length][self.descriptor_names].values
                
                # Get the return for the next month
                target_return = sorted_group.iloc[i+self.sequence_length]['Returns']
                
                sequences.append(seq)
                returns.append(target_return)
                tickers.append(ticker)
                dates.append(sorted_group.index[i+self.sequence_length])
        
        X = np.array(sequences)
        y = np.array(returns)
        
        # Train-test split
        test_size = int(len(X) * test_ratio)
        
        if test_size > 0:
            # Use the most recent data as test set
            X_train, X_test = X[:-test_size], X[-test_size:]
            y_train, y_test = y[:-test_size], y[-test_size:]
            tickers_train, tickers_test = tickers[:-test_size], tickers[-test_size:]
            dates_train, dates_test = dates[:-test_size], dates[-test_size:]
        else:
            X_train, X_test = X, X[-100:]  # Just use a small sample for testing
            y_train, y_test = y, y[-100:]
            tickers_train, tickers_test = tickers, tickers[-100:]
            dates_train, dates_test = dates, dates[-100:]
        
        # Normalize the data
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        X_test_flat = X_test.reshape(X_test.shape[0], -1)
        
        self.X_scaler.fit(X_train_flat)
        X_train_scaled_flat = self.X_scaler.transform(X_train_flat)
        X_test_scaled_flat = self.X_scaler.transform(X_test_flat)
        
        # Reshape back to sequences
        X_train_scaled = X_train_scaled_flat.reshape(X_train.shape)
        X_test_scaled = X_test_scaled_flat.reshape(X_test.shape)
        
        # Scale y
        y_train_reshaped = y_train.reshape(-1, 1)
        self.y_scaler.fit(y_train_reshaped)
        y_train_scaled = self.y_scaler.transform(y_train_reshaped).flatten()
        y_test_scaled = self.y_scaler.transform(y_test.reshape(-1, 1)).flatten()
        
        # Convert to PyTorch tensors
        X_train_tensor = torch.FloatTensor(X_train_scaled).to(self.device)
        X_test_tensor = torch.FloatTensor(X_test_scaled).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train_scaled).to(self.device)
        y_test_tensor = torch.FloatTensor(y_test_scaled).to(self.device)
        
        # Also prepare flattened tensors for DNN
        X_train_flat_tensor = torch.FloatTensor(X_train_scaled_flat).to(self.device)
        X_test_flat_tensor = torch.FloatTensor(X_test_scaled_flat).to(self.device)
        
        return (X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor,
                X_train_flat_tensor, X_test_flat_tensor,
                tickers_train, tickers_test, dates_train, dates_test)
    
    def train_models(self, X_train, X_test, y_train, y_test, 
                    X_train_flat, X_test_flat, epochs=10, batch_size=32):
        """
        Train the LSTM and benchmark models
        
        Parameters:
        -----------
        X_train : torch.Tensor
            Training features for LSTM
        X_test : torch.Tensor
            Test features for LSTM
        y_train : torch.Tensor
            Training targets
        y_test : torch.Tensor
            Test targets
        X_train_flat : torch.Tensor
            Flattened training features for DNN
        X_test_flat : torch.Tensor
            Flattened test features for DNN
        epochs : int
            Number of training epochs
        batch_size : int
            Batch size for training
            
        Returns:
        --------
        Dictionary of model history
        """
        # Input dimensions
        input_dim = X_train.shape[2]  # Number of features
        seq_len = X_train.shape[1]    # Sequence length
        input_flat_dim = X_train_flat.shape[1]  # Flattened input dimension
        
        # Create LSTM model
        self.lstm_model = LSTMModel(input_dim, self.hidden_dim).to(self.device)
        
        # Create DNN model
        self.dnn_model = DNNModel(input_flat_dim, [80, 40]).to(self.device)
        
        # Create data loaders
        train_dataset_lstm = TensorDataset(X_train, y_train)
        test_dataset_lstm = TensorDataset(X_test, y_test)
        
        train_loader_lstm = DataLoader(train_dataset_lstm, batch_size=batch_size, shuffle=True)
        test_loader_lstm = DataLoader(test_dataset_lstm, batch_size=batch_size)
        
        train_dataset_dnn = TensorDataset(X_train_flat, y_train)
        test_dataset_dnn = TensorDataset(X_test_flat, y_test)
        
        train_loader_dnn = DataLoader(train_dataset_dnn, batch_size=batch_size, shuffle=True)
        test_loader_dnn = DataLoader(test_dataset_dnn, batch_size=batch_size)
        
        # Define loss function and optimizer for LSTM
        lstm_criterion = nn.MSELoss()
        lstm_optimizer = optim.Adam(self.lstm_model.parameters(), lr=0.001)
        
        # Define loss function and optimizer for DNN
        dnn_criterion = nn.MSELoss()
        dnn_optimizer = optim.Adam(self.dnn_model.parameters(), lr=0.001)
        
        # Training loop for LSTM
        lstm_history = {'train_loss': [], 'val_loss': []}
        best_val_loss = float('inf')
        best_lstm_state = None
        
        print("Training LSTM model...")
        for epoch in range(epochs):
            # Training
            self.lstm_model.train()
            train_loss = 0
            
            for batch_X, batch_y in train_loader_lstm:
                # Forward pass
                outputs = self.lstm_model(batch_X)
                loss = lstm_criterion(outputs.squeeze(), batch_y)
                
                # Backward and optimize
                lstm_optimizer.zero_grad()
                loss.backward()
                lstm_optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader_lstm)
            lstm_history['train_loss'].append(train_loss)
            
            # Validation
            self.lstm_model.eval()
            val_loss = 0
            
            with torch.no_grad():
                for batch_X, batch_y in test_loader_lstm:
                    outputs = self.lstm_model(batch_X)
                    loss = lstm_criterion(outputs.squeeze(), batch_y)
                    val_loss += loss.item()
            
            val_loss /= len(test_loader_lstm)
            lstm_history['val_loss'].append(val_loss)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_lstm_state = self.lstm_model.state_dict().copy()
            
            print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        # Load best model
        if best_lstm_state is not None:
            self.lstm_model.load_state_dict(best_lstm_state)
        
        # Training loop for DNN
        dnn_history = {'train_loss': [], 'val_loss': []}
        best_val_loss = float('inf')
        best_dnn_state = None
        
        print("\nTraining DNN model...")
        for epoch in range(epochs):
            # Training
            self.dnn_model.train()
            train_loss = 0
            
            for batch_X, batch_y in train_loader_dnn:
                # Forward pass
                outputs = self.dnn_model(batch_X)
                loss = dnn_criterion(outputs.squeeze(), batch_y)
                
                # Backward and optimize
                dnn_optimizer.zero_grad()
                loss.backward()
                dnn_optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader_dnn)
            dnn_history['train_loss'].append(train_loss)
            
            # Validation
            self.dnn_model.eval()
            val_loss = 0
            
            with torch.no_grad():
                for batch_X, batch_y in test_loader_dnn:
                    outputs = self.dnn_model(batch_X)
                    loss = dnn_criterion(outputs.squeeze(), batch_y)
                    val_loss += loss.item()
            
            val_loss /= len(test_loader_dnn)
            dnn_history['val_loss'].append(val_loss)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_dnn_state = self.dnn_model.state_dict().copy()
            
            print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        # Load best model
        if best_dnn_state is not None:
            self.dnn_model.load_state_dict(best_dnn_state)
        
        # Setup feature importance analyzer for LSTM
        if self.use_feature_importance:
            self.feature_importance_analyzer = FeaturePermutationImportance(self.lstm_model)
        
        # Train traditional ML models on numpy data
        print("\nTraining benchmark models...")
        X_train_np = X_train_flat.cpu().numpy()
        y_train_np = y_train.cpu().numpy()
        
        # Linear model
        self.linear_model.fit(X_train_np, y_train_np)
        
        # SVR (using a subset due to computational constraints)
        sample_size = min(5000, len(X_train_np))
        indices = np.random.choice(len(X_train_np), sample_size, replace=False)
        self.svr_model.fit(X_train_np[indices], y_train_np[indices])
        
        # Random Forest
        self.rf_model.fit(X_train_np, y_train_np)
        
        return {
            'lstm': lstm_history,
            'dnn': dnn_history
        }
    
    def evaluate_models(self, X_test, X_test_flat, y_test, tickers_test, dates_test):
        """
        Evaluate all models on test data
        
        Parameters:
        -----------
        X_test : torch.Tensor
            Test features for LSTM
        X_test_flat : torch.Tensor
            Flattened test features for DNN and traditional models
        y_test : torch.Tensor
            Test targets
        tickers_test : list
            Test tickers
        dates_test : list
            Test dates
            
        Returns:
        --------
        Dictionary of evaluation metrics
        """
        # Convert test data to numpy for traditional models
        X_test_np = X_test_flat.cpu().numpy()
        y_test_np = y_test.cpu().numpy()
        
        # Evaluate LSTM model
        self.lstm_model.eval()
        with torch.no_grad():
            lstm_preds = self.lstm_model(X_test).cpu().numpy()
        
        # Evaluate DNN model
        self.dnn_model.eval()
        with torch.no_grad():
            dnn_preds = self.dnn_model(X_test_flat).cpu().numpy()
        
        # Evaluate traditional models
        linear_preds = self.linear_model.predict(X_test_np)
        svr_preds = self.svr_model.predict(X_test_np)
        rf_preds = self.rf_model.predict(X_test_np)
        
        # Inverse transform predictions to original scale
        lstm_preds_orig = self.y_scaler.inverse_transform(lstm_preds).flatten()
        dnn_preds_orig = self.y_scaler.inverse_transform(dnn_preds).flatten()
        linear_preds_orig = self.y_scaler.inverse_transform(linear_preds.reshape(-1, 1)).flatten()
        svr_preds_orig = self.y_scaler.inverse_transform(svr_preds.reshape(-1, 1)).flatten()
        rf_preds_orig = self.y_scaler.inverse_transform(rf_preds.reshape(-1, 1)).flatten()
        y_test_orig = self.y_scaler.inverse_transform(y_test.cpu().numpy().reshape(-1, 1)).flatten()
        
        # Store predictions
        self.predictions = {
            'lstm': lstm_preds_orig,
            'dnn': dnn_preds_orig,
            'linear': linear_preds_orig,
            'svr': svr_preds_orig,
            'rf': rf_preds_orig,
            'true': y_test_orig,
            'tickers': tickers_test,
            'dates': dates_test
        }
        
        # Calculate metrics
        metrics = {}
        for model_name in ['lstm', 'dnn', 'linear', 'svr', 'rf']:
            preds = self.predictions[model_name]
            mae = mean_absolute_error(y_test_orig, preds)
            rmse = np.sqrt(mean_squared_error(y_test_orig, preds))
            metrics[model_name] = {'mae': mae, 'rmse': rmse}
        
        return metrics
    
    def calculate_feature_importance(self, X_test, y_test):
        """
        Calculate feature importance using permutation importance
        
        Parameters:
        -----------
        X_test : torch.Tensor
            Test features
        y_test : torch.Tensor
            Test targets
            
        Returns:
        --------
        DataFrame with factor contributions
        """
        if not self.use_feature_importance or self.feature_importance_analyzer is None:
            print("Feature importance analyzer not set up")
            return None
        
        # Calculate feature importance using permutation method
        print("Calculating feature importance...")
        
        # Process in small batches to avoid memory issues
        batch_size = 100
        all_importances = []
        
        # Use smaller subset for faster computation
        subset_size = min(len(X_test), 500)
        indices = torch.randperm(len(X_test))[:subset_size]
        X_subset = X_test[indices]
        y_subset = y_test[indices]
        
        # Calculate feature importance on the subset
        feature_importances = self.feature_importance_analyzer.compute_importance(X_subset, y_subset, n_repeats=3)
        
        # Map importances to factors
        factor_contributions = pd.DataFrame(index=range(len(X_subset)))
        
        # Create mapping from descriptor indices to factors
        descriptor_to_factor = {}
        factor_indices = {factor: [] for factor in self.factor_names}
        
        # For each descriptor, determine which factor it belongs to
        for seq_idx in range(self.sequence_length):
            for desc_idx, descriptor in enumerate(self.descriptor_names):
                # Calculate the flattened index
                flat_idx = seq_idx * len(self.descriptor_names) + desc_idx
                
                # Assign to the appropriate factor
                if descriptor in ['60VOL', 'BETA', 'SKEW']:
                    descriptor_to_factor[flat_idx] = 'Risk'
                    factor_indices['Risk'].append((seq_idx, desc_idx))
                elif descriptor in ['ROE', 'ROA', 'ACCRUALS', 'LEVERAGE']:
                    descriptor_to_factor[flat_idx] = 'Quality'
                    factor_indices['Quality'].append((seq_idx, desc_idx))
                elif descriptor in ['12-1MOM', '1MOM', '60MOM']:
                    descriptor_to_factor[flat_idx] = 'Momentum'
                    factor_indices['Momentum'].append((seq_idx, desc_idx))
                elif descriptor in ['PSR', 'PER', 'PBR', 'PCFR']:
                    descriptor_to_factor[flat_idx] = 'Value'
                    factor_indices['Value'].append((seq_idx, desc_idx))
                elif descriptor in ['CAP', 'ILLIQ']:
                    descriptor_to_factor[flat_idx] = 'Size'
                    factor_indices['Size'].append((seq_idx, desc_idx))
        
        # Calculate factor contributions by summing importances for each factor
        for factor in self.factor_names:
            factor_sum = np.zeros(len(X_subset))
            for seq_idx, feat_idx in factor_indices[factor]:
                factor_sum += feature_importances[:, seq_idx, feat_idx]
            factor_contributions[factor] = factor_sum
        
        # Normalize to percentages
        row_sums = factor_contributions.sum(axis=1)
        factor_contributions_pct = factor_contributions.div(row_sums, axis=0) * 100
        
        self.factor_contributions = factor_contributions_pct
        
        return factor_contributions_pct
    
    def calculate_simple_factor_importance(self, X_test):
        """
        Calculate a simple version of factor importance based on feature correlations
        and model weights. This is a fallback if the permutation method is too slow.
        
        Parameters:
        -----------
        X_test : torch.Tensor
            Test features
            
        Returns:
        --------
        DataFrame with factor contributions
        """
        # Create mapping of descriptors to factors
        descriptor_to_factor = {}
        for descriptor in self.descriptor_names:
            if descriptor in ['60VOL', 'BETA', 'SKEW']:
                descriptor_to_factor[descriptor] = 'Risk'
            elif descriptor in ['ROE', 'ROA', 'ACCRUALS', 'LEVERAGE']:
                descriptor_to_factor[descriptor] = 'Quality'
            elif descriptor in ['12-1MOM', '1MOM', '60MOM']:
                descriptor_to_factor[descriptor] = 'Momentum'
            elif descriptor in ['PSR', 'PER', 'PBR', 'PCFR']:
                descriptor_to_factor[descriptor] = 'Value'
            elif descriptor in ['CAP', 'ILLIQ']:
                descriptor_to_factor[descriptor] = 'Size'
        
        # Get linear coefficients from Linear model as a baseline
        X_flat = X_test.reshape(X_test.shape[0], -1).cpu().numpy()
        coefs = np.abs(self.linear_model.coef_)
        
        # Normalize coefficients
        coefs = coefs / np.sum(coefs)
        
        # Map coefficients to factors
        factor_importances = {factor: 0.0 for factor in self.factor_names}
        
        for i, descriptor in enumerate(self.descriptor_names):
            # For each time step in the sequence
            for t in range(self.sequence_length):
                # Get the corresponding coefficient index
                idx = t * len(self.descriptor_names) + i
                if idx < len(coefs):  # Ensure index is within bounds
                    factor = descriptor_to_factor[descriptor]
                    factor_importances[factor] += coefs[idx]
        
        # Normalize the importance values
        total = sum(factor_importances.values())
        for factor in factor_importances:
            factor_importances[factor] /= total
            factor_importances[factor] *= 100  # Convert to percentage
        
        # Create a DataFrame with the same factor importance for all samples
        n_samples = len(X_test)
        df = pd.DataFrame(index=range(n_samples))
        
        for factor in self.factor_names:
            df[factor] = factor_importances[factor]
        
        self.factor_contributions = df
        
        return df
    
    def build_portfolios(self, n_quantiles=5):
        """
        Build quantile portfolios based on model predictions
        
        Parameters:
        -----------
        n_quantiles : int
            Number of quantiles to divide predictions into
            
        Returns:
        --------
        Dictionary with portfolio returns
        """
        portfolio_returns = {}
        
        # For each prediction set
        for model_name in ['lstm', 'dnn', 'linear', 'svr', 'rf']:
            # Get predictions and true returns
            preds = self.predictions[model_name]
            true_returns = self.predictions['true']
            tickers = self.predictions['tickers']
            dates = self.predictions['dates']
            
            # Create a dataframe
            portfolio_df = pd.DataFrame({
                'ticker': tickers,
                'date': dates,
                'pred_return': preds,
                'true_return': true_returns
            })
            
            # Convert to monthly portfolio returns
            monthly_returns = {}
            
            for date, group in portfolio_df.groupby('date'):
                try:
                    # Divide into quantiles
                    group['quantile'] = pd.qcut(group['pred_return'], n_quantiles, labels=False)
                    
                    # Calculate returns for each quantile
                    quantile_returns = group.groupby('quantile')['true_return'].mean()
                    
                    # Calculate long-short portfolio return (top minus bottom quantile)
                    long_short_return = quantile_returns.iloc[-1] - quantile_returns.iloc[0]
                    
                    # Store returns
                    monthly_returns[date] = {
                        'quantiles': quantile_returns.to_dict(),
                        'long_short': long_short_return
                    }
                except Exception as e:
                    # In case of error (e.g., all predictions are the same), skip this date
                    print(f"Error processing date {date} for model {model_name}: {e}")
                    continue
            
            # Convert to dataframe for analysis
            monthly_df = pd.DataFrame.from_dict(monthly_returns, orient='index')
            portfolio_returns[model_name] = monthly_df
        
        return portfolio_returns
    
    def analyze_portfolio_performance(self, portfolio_returns):
        """
        Analyze the performance of quantile portfolios
        
        Parameters:
        -----------
        portfolio_returns : dict
            Dictionary with portfolio returns
            
        Returns:
        --------
        Dictionary with performance metrics
        """
        performance = {}
        
        for model_name, monthly_df in portfolio_returns.items():
            if len(monthly_df) == 0:
                # Skip if no monthly returns data
                print(f"No portfolio returns data for model {model_name}")
                performance[model_name] = {
                    'annualized_return': 0,
                    'annualized_vol': 0,
                    'sharpe_ratio': 0,
                    'max_drawdown': 0,
                    'monthly_returns': pd.Series(dtype=float)
                }
                continue
            
            try:
                # Extract long-short returns
                # This needs to be fixed - the data structure is different than expected
                if 'long_short' in monthly_df.columns:
                    # If long_short is a column with direct values
                    long_short_returns = monthly_df['long_short']
                else:
                    # Try to handle different data structures
                    print(f"Warning: Unexpected data structure for {model_name}")
                    # Just create a dummy series for now
                    long_short_returns = pd.Series(0.0, index=monthly_df.index)
                
                # Calculate metrics
                annualized_return = long_short_returns.mean() * 12
                annualized_vol = long_short_returns.std() * np.sqrt(12)
                sharpe_ratio = annualized_return / annualized_vol if annualized_vol != 0 else 0
                
                # Calculate drawdown
                cum_returns = (1 + long_short_returns / 100).cumprod()
                peak = cum_returns.expanding().max()
                drawdown = (cum_returns / peak - 1) * 100
                max_drawdown = drawdown.min()
                
                performance[model_name] = {
                    'annualized_return': annualized_return,
                    'annualized_vol': annualized_vol,
                    'sharpe_ratio': sharpe_ratio,
                    'max_drawdown': max_drawdown,
                    'monthly_returns': long_short_returns
                }
            except Exception as e:
                print(f"Error calculating performance for {model_name}: {e}")
                # Provide default values
                performance[model_name] = {
                    'annualized_return': 0,
                    'annualized_vol': 0,
                    'sharpe_ratio': 0,
                    'max_drawdown': 0,
                    'monthly_returns': pd.Series(dtype=float)
                }
        
        return performance
    
    def plot_cumulative_returns(self, performance):
        """
        Plot cumulative returns for each model
        
        Parameters:
        -----------
        performance : dict
            Dictionary with performance metrics
        """
        plt.figure(figsize=(12, 8))
        
        for model_name, metrics in performance.items():
            if len(metrics['monthly_returns']) > 0:
                monthly_returns = metrics['monthly_returns']
                cum_returns = (1 + monthly_returns / 100).cumprod() - 1
                cum_returns.plot(label=f"{model_name.upper()}")
        
        plt.title('Cumulative Returns of Long-Short Portfolios')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Return')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    
    def plot_factor_contributions(self):
        """Plot factor contributions for interpretation"""
        if self.factor_contributions is None or len(self.factor_contributions) == 0:
            print("Factor contributions not calculated")
            return
        
        plt.figure(figsize=(12, 8))
        
        # Average contributions across all predictions
        avg_contributions = self.factor_contributions.mean()
        
        # Plot
        ax = avg_contributions.plot(kind='bar', color='skyblue')
        
        plt.title('Average Factor Contributions (%)')
        plt.xlabel('Factor')
        plt.ylabel('Contribution (%)')
        plt.grid(True, axis='y')
        
        # Add value labels
        for i, v in enumerate(avg_contributions):
            ax.text(i, v + 0.5, f"{v:.1f}%", ha='center')
        
        plt.tight_layout()
        plt.show()
        
        # Additionally, plot a time series of contributions
        plt.figure(figsize=(14, 8))
        
        # Get dates from predictions
        dates = pd.to_datetime(self.predictions['dates'])
        
        # Create a dataframe with dates
        contribution_ts = self.factor_contributions.copy()
        contribution_ts['date'] = dates
        
        # Resample to monthly averages
        monthly_contributions = contribution_ts.set_index('date').resample('M').mean()
        
        # Plot stacked area chart
        monthly_contributions.plot.area(stacked=True, cmap='viridis')
        
        plt.title('Factor Contributions Over Time')
        plt.xlabel('Date')
        plt.ylabel('Contribution (%)')
        plt.legend(title='Factors')
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    
    def run_analysis(self, tickers, start_date, end_date, epochs=10, batch_size=32):
        """
        Run the full analysis pipeline
        
        Parameters:
        -----------
        tickers : list
            List of ticker symbols
        start_date : str
            Start date in YYYYMMDD format
        end_date : str
            End date in YYYYMMDD format
        epochs : int
            Number of training epochs
        batch_size : int
            Batch size for training
            
        Returns:
        --------
        Dictionary with results
        """
        print(f"Running Deep Recurrent Factor Model analysis for {len(tickers)} tickers...")
        print(f"Period: {start_date} to {end_date}")
        
        # Generate sample data
        print("\nGenerating simulated data...")
        data = self._generate_sample_data(
            tickers=tickers,
            start_date=start_date,
            end_date=end_date
        )
        
        # Prepare data
        print("\nPreparing data for model training...")
        (X_train, X_test, y_train, y_test, X_train_flat, X_test_flat,
         tickers_train, tickers_test, dates_train, dates_test) = self.prepare_data(
            data_dict=data,
            test_ratio=0.2
        )
        
        # Train models
        print("\nTraining models...")
        history = self.train_models(
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            X_train_flat=X_train_flat,
            X_test_flat=X_test_flat,
            epochs=epochs,
            batch_size=batch_size
        )
        
        # Evaluate models
        print("\nEvaluating models...")
        metrics = self.evaluate_models(
            X_test=X_test,
            X_test_flat=X_test_flat,
            y_test=y_test,
            tickers_test=tickers_test,
            dates_test=dates_test
        )
        
        # Print evaluation metrics
        print("\nModel Evaluation Metrics:")
        for model_name, model_metrics in metrics.items():
            print(f"{model_name.upper()}: MAE = {model_metrics['mae']:.4f}, RMSE = {model_metrics['rmse']:.4f}")
        
        # Calculate feature importance - use simple method (faster) or permutation method (more accurate but slower)
        if self.use_feature_importance:
            print("\nCalculating factor contributions...")
            try:
                # Try using permutation importance (computationally expensive but more accurate)
                factor_contributions = self.calculate_feature_importance(X_test, y_test)
            except Exception as e:
                print(f"Error calculating feature importance: {e}")
                print("Falling back to simple factor importance calculation...")
                factor_contributions = self.calculate_simple_factor_importance(X_test)
        
        # Build portfolios
        print("\nBuilding quantile portfolios...")
        portfolio_returns = self.build_portfolios(n_quantiles=5)
        
        # Analyze portfolio performance
        print("\nAnalyzing portfolio performance...")
        performance = self.analyze_portfolio_performance(portfolio_returns)
        
        # Print performance metrics
        print("\nPortfolio Performance:")
        for model_name, model_perf in performance.items():
            print(f"{model_name.upper()}: Return = {model_perf['annualized_return']:.2f}%, "
                  f"Vol = {model_perf['annualized_vol']:.2f}%, "
                  f"Sharpe = {model_perf['sharpe_ratio']:.2f}, "
                  f"Max DD = {model_perf['max_drawdown']:.2f}%")
        
        # Plot results
        print("\nPlotting results...")
        self.plot_cumulative_returns(performance)
        
        if self.use_feature_importance:
            self.plot_factor_contributions()
        
        return {
            'metrics': metrics,
            'portfolio_returns': portfolio_returns,
            'performance': performance,
            'factor_contributions': self.factor_contributions if self.use_feature_importance else None
        }

# Main execution
if __name__ == "__main__":
    # Check if CUDA is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Set up the model
    model = DeepRecurrentFactorModel(
        sequence_length=5,
        hidden_dim=16,
        use_feature_importance=True,
        device=device
    )
    
    # Define tickers (sample Japanese stock tickers)
    tickers = [
        '7203.T',  # Toyota
        '9984.T',  # SoftBank Group
        '6758.T',  # Sony
        '9432.T',  # NTT
        '9433.T',  # KDDI
        '4063.T',  # Shin-Etsu Chemical
        '4661.T',  # Oriental Land
        '6367.T',  # Daikin Industries
        '6501.T',  # Hitachi
        '6861.T',  # Keyence
        '7267.T',  # Honda
        '7741.T',  # HOYA
        '8031.T',  # Mitsui & Co
        '8035.T',  # Tokyo Electron
        '8058.T',  # Mitsubishi Corp
        '9613.T',  # NTT Data
        '7974.T',  # Nintendo
        '8267.T',  # Aeon
        '6857.T',  # Advantest
        '6902.T'   # Denso
    ]
    
    # Run the analysis
    results = model.run_analysis(
        tickers=tickers,
        start_date='20000101',
        end_date='20201231',
        epochs=10,
        batch_size=32
    )