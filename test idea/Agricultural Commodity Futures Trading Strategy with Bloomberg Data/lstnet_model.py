import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class LSTNet(nn.Module):
    def __init__(self, num_variables, window, horizon, CNN_kernel=6, RNN_hidden_dim=100, 
                 CNN_hidden_dim=100, skip=24, skip_RNN_hidden_dim=10, ar_window=24, dropout=0.2, output_fun='sigmoid'):
        """
        Initialize the LSTNet model
        
        Args:
            num_variables: Number of variables in the multivariate time series
            window: Input window size
            horizon: Forecasting horizon
            CNN_kernel: Size of the kernel in CNN
            RNN_hidden_dim: Hidden dimension in RNN
            CNN_hidden_dim: Hidden dimension in CNN
            skip: Number of skipped steps in Recurrent-skip layer
            skip_RNN_hidden_dim: Hidden dimension in Recurrent-skip layer
            ar_window: Window size for AutoRegressive component
            dropout: Dropout rate
            output_fun: Output activation function ('sigmoid' or 'linear')
        """
        super(LSTNet, self).__init__()
        self.num_variables = num_variables
        self.window = window
        self.horizon = horizon
        self.output_fun = output_fun
        self.CNN_kernel = CNN_kernel
        self.RNN_hidden_dim = RNN_hidden_dim
        self.CNN_hidden_dim = CNN_hidden_dim
        self.skip = skip
        self.skip_RNN_hidden_dim = skip_RNN_hidden_dim
        self.dropout = dropout
        self.ar_window = ar_window
        
        # CNN Layer
        self.conv1 = nn.Conv2d(1, self.CNN_hidden_dim, kernel_size=(self.CNN_kernel, self.num_variables))
        self.conv_dropout = nn.Dropout(self.dropout)
        
        # GRU Layer
        self.gru1 = nn.GRU(self.CNN_hidden_dim, self.RNN_hidden_dim)
        
        if self.skip > 0:
            # Recurrent-skip component
            self.gru_skip = nn.GRU(self.CNN_hidden_dim, self.skip_RNN_hidden_dim)
            self.linear_skip = nn.Linear(self.skip_RNN_hidden_dim * (self.window // self.skip), self.num_variables)
            
        # Autoregressive component
        if self.ar_window > 0:
            self.ar_linear = nn.Linear(self.ar_window, self.num_variables)
            
        # Output layer
        self.linear_out = nn.Linear(self.RNN_hidden_dim, self.num_variables)
        
        # Attention layer
        self.attention_linear = nn.Linear(self.RNN_hidden_dim + self.RNN_hidden_dim, 1)
        
    def forward(self, x, y_prev=None):
        """
        Forward pass of LSTNet
        
        Args:
            x: Input tensor of shape [batch_size, window, num_variables]
            y_prev: Previous output for AR model
            
        Returns:
            Output predictions
        """
        batch_size = x.size(0)
        
        # CNN Layer
        c = x.unsqueeze(1)  # Add channel dimension
        c = self.conv1(c)
        c = c.squeeze(3)  # Remove the last dimension
        c = c.permute(2, 0, 1)  # [window, batch, CNN_hidden_dim]
        c = self.conv_dropout(c)
        
        # RNN Layer
        r, _ = self.gru1(c)  # r: [window, batch, RNN_hidden_dim]
        r = r.transpose(0, 1)  # [batch, window, RNN_hidden_dim]
        
        # Get the last output from RNN
        r_last = r[:, -1, :]  # [batch, RNN_hidden_dim]
        
        # Apply temporal attention if skip is 0
        if self.skip == 0:
            # Compute attention weights
            attn_weights = []
            for i in range(r.size(1)):
                h_concat = torch.cat([r_last, r[:, i, :]], dim=1)
                attn_weights.append(self.attention_linear(h_concat))
            attn_weights = torch.softmax(torch.stack(attn_weights, dim=1).squeeze(-1), dim=1)
            
            # Apply attention weights
            context = torch.bmm(attn_weights.unsqueeze(1), r).squeeze(1)
            res = self.linear_out(context)
        else:
            # Recurrent-skip component
            s = c.clone()
            s = s.view(self.window // self.skip, self.skip, batch_size, self.CNN_hidden_dim)
            s = s.permute(1, 2, 0, 3).contiguous()  # [skip, batch, window//skip, CNN_hidden_dim]
            s = s.view(self.skip * batch_size, self.window // self.skip, self.CNN_hidden_dim)
            
            _, s = self.gru_skip(s.transpose(0, 1))  # s: [1, skip*batch, skip_RNN_hidden_dim]
            s = s.transpose(0, 1).view(batch_size, self.skip * self.skip_RNN_hidden_dim)
            s = self.linear_skip(s)
            
            r_out = self.linear_out(r_last)
            res = r_out + s
        
        # Autoregressive component
        if self.ar_window > 0 and y_prev is not None:
            ar_out = self.ar_linear(y_prev)
            res = res + ar_out
            
        # Apply output activation function
        if self.output_fun == 'sigmoid':
            res = torch.sigmoid(res)
            
        return res

class TimeSeriesDataset:
    def __init__(self, window_size=30, horizon=12, normalize=True):
        """
        Initialize TimeSeriesDataset
        
        Args:
            window_size: Input window size
            horizon: Forecasting horizon
            normalize: Whether to normalize the data
        """
        self.window_size = window_size
        self.horizon = horizon
        self.normalize = normalize
        self.scalers = None
        
    def create_dataset(self, data, train_size=0.6, val_size=0.2):
        """
        Create dataset from time series data
        
        Args:
            data: DataFrame with time series data
            train_size: Proportion of data for training
            val_size: Proportion of data for validation
            
        Returns:
            Train, validation, and test DataLoaders
        """
        # Convert to numpy array
        data_values = data.values
        n_samples, n_features = data_values.shape
        
        # Split into train, validation, and test sets
        train_end = int(n_samples * train_size)
        val_end = train_end + int(n_samples * val_size)
        
        train_data = data_values[:train_end]
        val_data = data_values[train_end:val_end]
        test_data = data_values[val_end:]
        
        print(f"Train data shape: {train_data.shape}")
        print(f"Validation data shape: {val_data.shape}")
        print(f"Test data shape: {test_data.shape}")
        
        # Normalize data
        if self.normalize:
            self.scalers = []
            train_normalized = np.zeros_like(train_data, dtype=np.float32)
            val_normalized = np.zeros_like(val_data, dtype=np.float32)
            test_normalized = np.zeros_like(test_data, dtype=np.float32)
            
            for i in range(n_features):
                scaler = StandardScaler()
                train_normalized[:, i] = scaler.fit_transform(train_data[:, i].reshape(-1, 1)).flatten()
                val_normalized[:, i] = scaler.transform(val_data[:, i].reshape(-1, 1)).flatten()
                test_normalized[:, i] = scaler.transform(test_data[:, i].reshape(-1, 1)).flatten()
                self.scalers.append(scaler)
        else:
            train_normalized = train_data.astype(np.float32)
            val_normalized = val_data.astype(np.float32)
            test_normalized = test_data.astype(np.float32)
        
        # Create windowed datasets
        X_train, y_train = self._create_windows(train_normalized)
        X_val, y_val = self._create_windows(val_normalized)
        X_test, y_test = self._create_windows(test_normalized)
        
        print(f"Training samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")
        print(f"Test samples: {len(X_test)}")
        
        return X_train, y_train, X_val, y_val, X_test, y_test
    
    def _create_windows(self, data):
        """Create windowed dataset from time series data"""
        X, y = [], []
        
        for i in range(len(data) - self.window_size - self.horizon + 1):
            X.append(data[i:i+self.window_size])
            y.append(data[i+self.window_size:i+self.window_size+self.horizon])
        
        return np.array(X), np.array(y)
    
    def inverse_transform(self, data, feature_idx=None):
        """
        Inverse transform normalized data
        
        Args:
            data: Normalized data
            feature_idx: Index of feature to inverse transform
            
        Returns:
            Inverse transformed data
        """
        if not self.normalize or self.scalers is None:
            return data
        
        if feature_idx is not None:
            return self.scalers[feature_idx].inverse_transform(data.reshape(-1, 1)).flatten()
        
        # Assume data has shape [samples, features]
        inverse_data = np.zeros_like(data)
        for i in range(data.shape[1]):
            inverse_data[:, i] = self.scalers[i].inverse_transform(data[:, i].reshape(-1, 1)).flatten()
        
        return inverse_data

class LSTNetTrainer:
    def __init__(self, num_variables, window_size=30, horizon=12, cnn_kernel=6, 
                rnn_hidden_dim=100, cnn_hidden_dim=100, skip=24, skip_rnn_hidden_dim=10, 
                ar_window=24, dropout=0.2, output_fun='linear', learning_rate=0.001):
        """
        Initialize LSTNet Trainer
        """
        self.model = LSTNet(
            num_variables=num_variables,
            window=window_size,
            horizon=horizon,
            CNN_kernel=cnn_kernel,
            RNN_hidden_dim=rnn_hidden_dim,
            CNN_hidden_dim=cnn_hidden_dim,
            skip=skip,
            skip_RNN_hidden_dim=skip_rnn_hidden_dim,
            ar_window=ar_window,
            dropout=dropout,
            output_fun=output_fun
        ).to(device)
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        self.window_size = window_size
        self.horizon = horizon
        self.history = {'train_loss': [], 'val_loss': []}
        
    def train(self, train_loader, val_loader, epochs=100, patience=10):
        """
        Train LSTNet model
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of training epochs
            patience: Early stopping patience
            
        Returns:
            Trained model
        """
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0
            
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                
                # Forward pass
                y_pred = self.model(batch_x)
                loss = self.criterion(y_pred, batch_y[:, -1, :])  # Predict only the last step
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            self.history['train_loss'].append(train_loss)
            
            # Validation
            self.model.eval()
            val_loss = 0
            
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x = batch_x.to(device)
                    batch_y = batch_y.to(device)
                    
                    # Forward pass
                    y_pred = self.model(batch_x)
                    loss = self.criterion(y_pred, batch_y[:, -1, :])  # Predict only the last step
                    
                    val_loss += loss.item()
            
            val_loss /= len(val_loader)
            self.history['val_loss'].append(val_loss)
            
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), 'best_lstnet_model.pth')
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
        # Load best model
        self.model.load_state_dict(torch.load('best_lstnet_model.pth'))
        return self.model
    
    def evaluate(self, test_loader, dataset, original_data, target_idx=None):
        """
        Evaluate model on test data
        
        Args:
            test_loader: Test data loader
            dataset: TimeSeriesDataset object
            original_data: Original data for reference
            target_idx: Index of target feature for visualization
            
        Returns:
            Evaluation metrics
        """
        self.model.eval()
        test_loss = 0
        predictions = []
        actuals = []
        
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                
                # Forward pass
                y_pred = self.model(batch_x)
                loss = self.criterion(y_pred, batch_y[:, -1, :])  # Predict only the last step
                
                test_loss += loss.item()
                
                # Store predictions and actuals
                predictions.append(y_pred.cpu().numpy())
                actuals.append(batch_y[:, -1, :].cpu().numpy())
        
        test_loss /= len(test_loader)
        predictions = np.vstack(predictions)
        actuals = np.vstack(actuals)
        
        # Inverse transform predictions and actuals if normalized
        if dataset.normalize and dataset.scalers is not None:
            predictions_orig = np.zeros_like(predictions)
            actuals_orig = np.zeros_like(actuals)
            
            for i in range(predictions.shape[1]):
                predictions_orig[:, i] = dataset.scalers[i].inverse_transform(predictions[:, i].reshape(-1, 1)).flatten()
                actuals_orig[:, i] = dataset.scalers[i].inverse_transform(actuals[:, i].reshape(-1, 1)).flatten()
            
            predictions = predictions_orig
            actuals = actuals_orig
        
        # Calculate metrics
        mse = np.mean((predictions - actuals) ** 2, axis=0)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(predictions - actuals), axis=0)
        
        # Calculate RSE, RAE, CORR
        # These are the metrics used in the paper
        denominator = np.mean((actuals - np.mean(actuals, axis=0)) ** 2, axis=0)
        rse = np.sqrt(np.sum(mse) / np.sum(denominator))
        
        denominator = np.sum(np.abs(actuals - np.mean(actuals, axis=0)), axis=0)
        rae = np.sum(np.sum(np.abs(predictions - actuals), axis=0)) / np.sum(denominator)
        
        corr = []
        for i in range(predictions.shape[1]):
            if np.std(predictions[:, i]) > 0 and np.std(actuals[:, i]) > 0:
                corr.append(np.corrcoef(predictions[:, i], actuals[:, i])[0, 1])
            else:
                corr.append(0)
        corr = np.mean(corr)
        
        import matplotlib.pyplot as plt
        
        # Visualization if target_idx is provided
        if target_idx is not None:
            plt.figure(figsize=(12, 6))
            
            # Plot predictions vs actuals
            plt.subplot(1, 1, 1)
            plt.plot(actuals[:100, target_idx], label='Actual')
            plt.plot(predictions[:100, target_idx], label='Predicted')
            plt.title(f'Predictions vs Actuals - {original_data.columns[target_idx]}')
            plt.xlabel('Time')
            plt.ylabel('Price')
            plt.legend()
            
            plt.tight_layout()
            plt.savefig(f'predictions_{original_data.columns[target_idx]}.png')
            plt.show()
        
        return {
            'test_loss': test_loss,
            'predictions': predictions,
            'actuals': actuals,
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'rse': rse,
            'rae': rae,
            'corr': corr
        }
