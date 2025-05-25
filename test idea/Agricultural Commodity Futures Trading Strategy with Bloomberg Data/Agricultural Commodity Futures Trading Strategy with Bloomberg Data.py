import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
import pdblp
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import warnings
warnings.filterwarnings("ignore")

# Set seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

#############################################################################
# PART 1: BLOOMBERG DATA CONNECTION
#############################################################################

class BloombergDataManager:
    def __init__(self):
        """Initialize Bloomberg connection"""
        self.con = None
        self.connected = False
        self.tickers_info = {}

    def connect(self):
        """Establish connection to Bloomberg API"""
        try:
            self.con = pdblp.BCon(debug=False, port=8194)
            self.con.start()
            self.connected = True
            print("✅ Successfully connected to Bloomberg")
            return True
        except Exception as e:
            print(f"❌ Failed to connect to Bloomberg: {e}")
            print("Please ensure Bloomberg Terminal is running and you have proper permissions.")
            return False

    def check_connection(self):
        """Check if Bloomberg connection is active"""
        if not self.connected:
            return self.connect()
        return True

    def get_historical_data(self, tickers, start_date, end_date, fields=['PX_LAST'], options=None):
        """
        Retrieve historical data from Bloomberg

        Args:
            tickers: List of Bloomberg tickers
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            fields: Bloomberg fields to retrieve
            options: Additional Bloomberg request options

        Returns:
            DataFrame with historical data
        """
        if not self.check_connection():
            print("❌ Bloomberg connection not available. Cannot retrieve data.")
            return None

        try:
            print(f"📊 Retrieving historical data for {len(tickers)} tickers ({start_date} to {end_date})...")
            # Remove options parameter if it's causing issues
            data = self.con.bdh(tickers, fields, start_date, end_date)

            # Check if data is empty
            if data.empty:
                print("⚠️ No data retrieved from Bloomberg")
                return None

            # Reshape if multiple fields
            if len(fields) == 1:
                data.columns = data.columns.get_level_values(0)

            print(f"✅ Successfully retrieved data with shape: {data.shape}")

            # Cache ticker info
            for ticker in tickers:
                if ticker not in self.tickers_info:
                    try:
                        info = self.con.ref(ticker, ['SECURITY_NAME', 'SECURITY_TYP', 'FUTURES_CATEGORY'])
                        self.tickers_info[ticker] = {
                            'name': info.iloc[0, 0] if not pd.isna(info.iloc[0, 0]) else ticker,
                            'type': info.iloc[0, 1] if not pd.isna(info.iloc[0, 1]) else 'Unknown',
                            'category': info.iloc[0, 2] if not pd.isna(info.iloc[0, 2]) else 'Unknown'
                        }
                    except:
                        self.tickers_info[ticker] = {'name': ticker, 'type': 'Unknown', 'category': 'Unknown'}

            return data
        except Exception as e:
            print(f"❌ Error retrieving data from Bloomberg: {e}")
            return None

    def get_intraday_data(self, tickers, start_datetime, end_datetime, interval=60, fields=['TRADE']):
        """
        Retrieve intraday data from Bloomberg

        Args:
            tickers: List of Bloomberg tickers
            start_datetime: Start datetime (YYYY-MM-DD HH:MM:SS)
            end_datetime: End datetime (YYYY-MM-DD HH:MM:SS)
            interval: Interval in minutes
            fields: Bloomberg fields to retrieve

        Returns:
            DataFrame with intraday data
        """
        if not self.check_connection():
            return None

        try:
            print(f"📊 Retrieving intraday data for {len(tickers)} tickers...")
            data = self.con.bdib(tickers, start_datetime, end_datetime, interval, fields)

            if data is None or data.empty:
                print("⚠️ No intraday data retrieved")
                return None

            print(f"✅ Successfully retrieved intraday data with shape: {data.shape}")
            return data
        except Exception as e:
            print(f"❌ Error retrieving intraday data: {e}")
            return None

    def get_futures_chain(self, ticker_root, date=None):
        """
        Get futures chain for a given root ticker

        Args:
            ticker_root: Root ticker symbol (e.g., 'C ')
            date: Reference date (defaults to today)

        Returns:
            List of futures chain tickers
        """
        if not self.check_connection():
            return None

        if date is None:
            date = datetime.now().strftime("%Y%m%d")
        else:
            date = pd.to_datetime(date).strftime("%Y%m%d")

        try:
            print(f"📊 Retrieving futures chain for {ticker_root}...")
            # Try different approaches to get the futures chain
            try:
                # First try without the reference date parameter
                chain = self.con.ref(f"{ticker_root} COMDTY", "FUT_CHAIN")
            except:
                try:
                    # Then try with the reference date as a parameter
                    chain = self.con.ref(f"{ticker_root} COMDTY", "FUT_CHAIN", [("REFERENCE_DATE", date)])
                except:
                    # If both fail, try a simpler approach
                    chain = self.con.ref(f"{ticker_root} COMDTY", "FUT_CHAIN")

            if chain is None or chain.empty:
                print(f"⚠️ No futures chain retrieved for {ticker_root}")
                return []

            # Extract the tickers from the chain
            futures_tickers = chain.iloc[0, 0].split()
            print(f"✅ Found {len(futures_tickers)} contracts in futures chain")
            return futures_tickers
        except Exception as e:
            print(f"❌ Error retrieving futures chain: {e}")
            return []

    def get_ticker_info(self, ticker):
        """Get detailed information about a ticker"""
        if not self.check_connection():
            return None

        try:
            fields = [
                'SECURITY_NAME', 'SECURITY_TYP', 'MARKET_SECTOR_DES',
                'FUTURES_CATEGORY', 'FUT_CONT_SIZE', 'FUT_VAL_PT',
                'FUT_FIRST_TRADE_DT', 'LAST_TRADEABLE_DT'
            ]

            info = self.con.ref(ticker, fields)

            if info is None or info.empty:
                print(f"⚠️ No information retrieved for {ticker}")
                return {}

            # Create a dictionary of information
            info_dict = {fields[i]: info.iloc[0, i] for i in range(len(fields))}
            return info_dict
        except Exception as e:
            print(f"❌ Error retrieving ticker information: {e}")
            return {}

    def close(self):
        """Close Bloomberg connection"""
        if self.connected and self.con is not None:
            try:
                self.con.stop()
                self.connected = False
                print("✅ Bloomberg connection closed")
            except Exception as e:
                print(f"❌ Error closing Bloomberg connection: {e}")

#############################################################################
# PART 2: LSTNET MODEL IMPLEMENTATION
#############################################################################

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
        r, h = self.gru1(c)  # r: [window, batch, RNN_hidden_dim]
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

#############################################################################
# PART 3: DATA PREPROCESSING AND MODEL TRAINING
#############################################################################

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

#############################################################################
# PART 4: TRADING STRATEGY IMPLEMENTATION
#############################################################################

class AgricultureFuturesStrategy:
    def __init__(self, model, dataset, tickers, threshold=0.005, position_size=0.1,
                trade_cost=0.001, window_size=30, horizon=12):
        """
        Initialize AgricultureFuturesStrategy

        Args:
            model: Trained LSTNet model
            dataset: TimeSeriesDataset object
            tickers: List of Bloomberg tickers
            threshold: Threshold for signal generation
            position_size: Position size as percentage of capital
            trade_cost: Trading cost as percentage
            window_size: Input window size
            horizon: Forecasting horizon
        """
        self.model = model
        self.dataset = dataset
        self.tickers = tickers
        self.threshold = threshold
        self.position_size = position_size
        self.trade_cost = trade_cost
        self.window_size = window_size
        self.horizon = horizon
        self.positions = {ticker: 0 for ticker in tickers}
        self.capital = 1000000  # Initial capital
        self.portfolio_value = self.capital
        self.performance_history = []

        # Set model to evaluation mode
        self.model.eval()

    def generate_signals(self, data, last_n_periods=1):
        """
        Generate trading signals based on model predictions

        Args:
            data: Input data as DataFrame
            last_n_periods: Number of most recent periods to consider for signals

        Returns:
            Dictionary of signals for each ticker
        """
        # Extract the last window_size + last_n_periods data points
        recent_data = data.iloc[-(self.window_size + last_n_periods):].values

        # Normalize data if dataset is normalized
        if self.dataset.normalize and self.dataset.scalers is not None:
            normalized_data = np.zeros_like(recent_data, dtype=np.float32)
            for i in range(recent_data.shape[1]):
                normalized_data[:, i] = self.dataset.scalers[i].transform(recent_data[:, i].reshape(-1, 1)).flatten()
        else:
            normalized_data = recent_data.astype(np.float32)

        signals = {}

        # Generate signals for the last n periods
        for i in range(last_n_periods):
            # Create input window
            window_start = i
            window_end = window_start + self.window_size
            window_data = normalized_data[window_start:window_end]

            # Convert to tensor
            X = torch.FloatTensor(window_data).unsqueeze(0).to(device)

            # Generate prediction
            with torch.no_grad():
                y_pred = self.model(X).cpu().numpy()[0]

            # Inverse transform prediction if normalized
            if self.dataset.normalize and self.dataset.scalers is not None:
                y_pred_orig = np.zeros_like(y_pred)
                for j in range(y_pred.shape[0]):
                    y_pred_orig[j] = self.dataset.scalers[j].inverse_transform(y_pred[j].reshape(-1, 1)).flatten()[0]
                y_pred = y_pred_orig

            # Current prices
            current_prices = recent_data[window_end-1]

            # Calculate expected returns
            expected_returns = (y_pred - current_prices) / current_prices

            # Generate signals
            period_signals = {}
            for j, ticker in enumerate(self.tickers):
                if expected_returns[j] > self.threshold:
                    period_signals[ticker] = 'BUY'
                elif expected_returns[j] < -self.threshold:
                    period_signals[ticker] = 'SELL'
                else:
                    period_signals[ticker] = 'HOLD'

            # Store signals for this period
            signals[data.index[-(last_n_periods-i)]] = period_signals

        return signals

    def execute_trades(self, signals, current_prices):
        """
        Execute trades based on signals

        Args:
            signals: Dictionary of signals for each ticker
            current_prices: Current prices as Series

        Returns:
            List of executed trades
        """
        executed_trades = []

        for ticker, signal in signals.items():
            current_price = current_prices[ticker]
            current_position = self.positions.get(ticker, 0)

            if signal == 'BUY' and current_position <= 0:
                # Calculate position size in units
                position_value = self.portfolio_value * self.position_size
                units_to_buy = position_value / current_price

                # Calculate transaction cost
                transaction_cost = position_value * self.trade_cost

                # Check if enough capital
                if position_value + transaction_cost <= self.capital:
                    # Execute buy order
                    self.positions[ticker] = units_to_buy
                    self.capital -= (position_value + transaction_cost)

                    # Record trade
                    executed_trades.append({
                        'ticker': ticker,
                        'action': 'BUY',
                        'price': current_price,
                        'units': units_to_buy,
                        'value': position_value,
                        'cost': transaction_cost,
                        'timestamp': datetime.now()
                    })

            elif signal == 'SELL' and current_position >= 0:
                if current_position > 0:
                    # Calculate position value
                    position_value = current_position * current_price

                    # Calculate transaction cost
                    transaction_cost = position_value * self.trade_cost

                    # Execute sell order
                    self.capital += (position_value - transaction_cost)
                    self.positions[ticker] = 0

                    # Record trade
                    executed_trades.append({
                        'ticker': ticker,
                        'action': 'SELL',
                        'price': current_price,
                        'units': current_position,
                        'value': position_value,
                        'cost': transaction_cost,
                        'timestamp': datetime.now()
                    })

                # Consider short selling if allowed
                position_value = self.portfolio_value * self.position_size
                units_to_short = position_value / current_price

                # Calculate transaction cost
                transaction_cost = position_value * self.trade_cost

                # Check if enough capital for margin
                if position_value * 0.5 <= self.capital:  # Assuming 50% margin requirement
                    # Execute short order
                    self.positions[ticker] = -units_to_short
                    self.capital -= transaction_cost

                    # Record trade
                    executed_trades.append({
                        'ticker': ticker,
                        'action': 'SHORT',
                        'price': current_price,
                        'units': units_to_short,
                        'value': position_value,
                        'cost': transaction_cost,
                        'timestamp': datetime.now()
                    })

        return executed_trades

    def update_portfolio_value(self, current_prices):
        """
        Update portfolio value based on current prices

        Args:
            current_prices: Current prices as Series

        Returns:
            Updated portfolio value
        """
        position_value = 0

        for ticker, units in self.positions.items():
            if ticker in current_prices:
                position_value += units * current_prices[ticker]

        self.portfolio_value = self.capital + position_value
        self.performance_history.append({
            'timestamp': datetime.now(),
            'portfolio_value': self.portfolio_value,
            'cash': self.capital,
            'position_value': position_value
        })

        return self.portfolio_value

    def get_performance_metrics(self):
        """
        Calculate performance metrics

        Returns:
            Dictionary of performance metrics
        """
        if len(self.performance_history) < 2:
            return {
                'total_return': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'volatility': 0,
                'current_value': self.portfolio_value
            }

        # Extract portfolio values
        values = [record['portfolio_value'] for record in self.performance_history]

        # Calculate returns
        returns = [(values[i] - values[i-1]) / values[i-1] for i in range(1, len(values))]

        # Calculate metrics
        total_return = (values[-1] / values[0]) - 1
        volatility = np.std(returns) * np.sqrt(252)  # Annualized
        sharpe_ratio = (np.mean(returns) * 252) / volatility if volatility > 0 else 0

        # Calculate drawdown
        drawdowns = [1 - values[i] / max(values[:i+1]) for i in range(len(values))]
        max_drawdown = max(drawdowns) if drawdowns else 0

        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'volatility': volatility,
            'current_value': self.portfolio_value
        }

    def backtest(self, data, start_idx=None, end_idx=None):
        """
        Backtest strategy on historical data

        Args:
            data: Historical data as DataFrame
            start_idx: Start index for backtest
            end_idx: End index for backtest

        Returns:
            Backtest results
        """
        if start_idx is None:
            start_idx = self.window_size

        if end_idx is None:
            end_idx = len(data)

        # Reset strategy state
        self.positions = {ticker: 0 for ticker in self.tickers}
        self.capital = 1000000
        self.portfolio_value = self.capital
        self.performance_history = []

        # Track trades and daily performance
        trades = []
        daily_performance = []

        # Iterate through each day
        for i in range(start_idx, end_idx):
            current_date = data.index[i]

            # Generate signals using data up to current date
            historical_data = data.iloc[:i]
            signals = self.generate_signals(historical_data, last_n_periods=1)

            # Get current day's signals
            if current_date in signals:
                current_signals = signals[current_date]

                # Get current prices
                current_prices = data.iloc[i]

                # Execute trades
                executed_trades = self.execute_trades(current_signals, current_prices)
                if executed_trades:
                    trades.extend(executed_trades)

                # Update portfolio value
                self.update_portfolio_value(current_prices)

            # Record daily performance
            daily_performance.append({
                'date': current_date,
                'portfolio_value': self.portfolio_value,
                'cash': self.capital,
                'positions': self.positions.copy()
            })

        # Calculate performance metrics
        metrics = self.get_performance_metrics()

        # Prepare DataFrames for analysis
        trades_df = pd.DataFrame(trades)
        daily_df = pd.DataFrame(daily_performance)
        daily_df.set_index('date', inplace=True)

        if not trades_df.empty:
            trades_df['profit'] = None

            # Calculate profit for each closed position
            for ticker in self.tickers:
                ticker_trades = trades_df[trades_df['ticker'] == ticker].copy()

                for i in range(len(ticker_trades)):
                    if ticker_trades.iloc[i]['action'] == 'SELL' and i > 0:
                        buy_price = ticker_trades.iloc[i-1]['price'] if ticker_trades.iloc[i-1]['action'] == 'BUY' else None
                        if buy_price is not None:
                            sell_price = ticker_trades.iloc[i]['price']
                            units = min(ticker_trades.iloc[i]['units'], ticker_trades.iloc[i-1]['units'])
                            profit = units * (sell_price - buy_price)
                            trades_df.loc[ticker_trades.index[i], 'profit'] = profit

        # Visualization
        self._visualize_backtest_results(daily_df, trades_df)

        return {
            'metrics': metrics,
            'trades': trades_df,
            'daily_performance': daily_df
        }

    def _visualize_backtest_results(self, daily_df, trades_df):
        """
        Visualize backtest results

        Args:
            daily_df: Daily performance DataFrame
            trades_df: Trades DataFrame
        """
        plt.figure(figsize=(14, 8))

        # Plot portfolio value
        plt.subplot(2, 1, 1)
        plt.plot(daily_df['portfolio_value'])
        plt.title('Portfolio Value Over Time')
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.grid(True)

        # Plot trades
        if not trades_df.empty and 'profit' in trades_df.columns:
            plt.subplot(2, 1, 2)
            profitable_trades = trades_df[trades_df['profit'] > 0]
            losing_trades = trades_df[trades_df['profit'] < 0]

            if not profitable_trades.empty:
                plt.scatter(profitable_trades['timestamp'], profitable_trades['profit'],
                         color='green', label='Profitable Trades')

            if not losing_trades.empty:
                plt.scatter(losing_trades['timestamp'], losing_trades['profit'],
                         color='red', label='Losing Trades')

            plt.title('Trade Profits')
            plt.xlabel('Date')
            plt.ylabel('Profit')
            plt.legend()
            plt.grid(True)

        plt.tight_layout()
        plt.savefig('backtest_results.png')
        plt.show()

        # Print performance metrics
        metrics = self.get_performance_metrics()
        print("\n===== Backtest Performance =====")
        print(f"Total Return: {metrics['total_return']:.2%}")
        print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        print(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
        print(f"Volatility: {metrics['volatility']:.2%}")
        print(f"Final Portfolio Value: ${metrics['current_value']:.2f}")

        if not trades_df.empty:
            total_trades = len(trades_df)
            profitable_trades = sum(1 for profit in trades_df['profit'] if profit is not None and profit > 0)
            win_rate = profitable_trades / sum(1 for profit in trades_df['profit'] if profit is not None) if sum(1 for profit in trades_df['profit'] if profit is not None) > 0 else 0

            print(f"Total Trades: {total_trades}")
            print(f"Win Rate: {win_rate:.2%}")

#############################################################################
# PART 5: MODEL COMPARISON AND STATIONARITY TESTS
#############################################################################

def test_stationarity(data):
    """
    Test for stationarity using ADF test

    Args:
        data: DataFrame with time series data

    Returns:
        DataFrame with test results
    """
    from statsmodels.tsa.stattools import adfuller

    results = []

    for column in data.columns:
        series = data[column].dropna()
        adf_result = adfuller(series, autolag='AIC')

        results.append({
            'ticker': column,
            'adf_statistic': adf_result[0],
            'p_value': adf_result[1],
            'critical_values': adf_result[4],
            'is_stationary': adf_result[1] < 0.05
        })

    return pd.DataFrame(results)

def compare_models(data, window_size=30, horizon=12, batch_size=128):
    """
    Compare LSTNet with baseline models

    Args:
        data: DataFrame with time series data
        window_size: Input window size
        horizon: Forecasting horizon
        batch_size: Batch size for training

    Returns:
        Dictionary with comparison results
    """
    from sklearn.linear_model import LinearRegression
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.vector_ar.var_model import VAR

    # Prepare data
    dataset = TimeSeriesDataset(window_size=window_size, horizon=horizon)
    X_train, y_train, X_val, y_val, X_test, y_test = dataset.create_dataset(data)

    # Create data loaders for LSTNet
    train_tensor_x = torch.FloatTensor(X_train)
    train_tensor_y = torch.FloatTensor(y_train)
    train_dataset = TensorDataset(train_tensor_x, train_tensor_y)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_tensor_x = torch.FloatTensor(X_val)
    val_tensor_y = torch.FloatTensor(y_val)
    val_dataset = TensorDataset(val_tensor_x, val_tensor_y)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    test_tensor_x = torch.FloatTensor(X_test)
    test_tensor_y = torch.FloatTensor(y_test)
    test_dataset = TensorDataset(test_tensor_x, test_tensor_y)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Initialize and train LSTNet
    print("\n===== Training LSTNet =====")
    lstnet_trainer = LSTNetTrainer(
        num_variables=data.shape[1],
        window_size=window_size,
        horizon=horizon
    )

    lstnet_trainer.train(train_loader, val_loader, epochs=50, patience=10)
    lstnet_results = lstnet_trainer.evaluate(test_loader, dataset, data)

    # Train baseline models
    baseline_results = {
        'LSTNet': lstnet_results,
        'Linear Regression': {},
        'ARIMA': {},
        'VAR': {}
    }

    # Linear Regression
    print("\n===== Training Linear Regression =====")
    lr_predictions = np.zeros((len(y_test), data.shape[1]))

    for i in range(data.shape[1]):
        model = LinearRegression()
        model.fit(X_train[:, :, i], y_train[:, -1, i])
        lr_predictions[:, i] = model.predict(X_test[:, :, i])

    if dataset.normalize and dataset.scalers is not None:
        lr_predictions_orig = np.zeros_like(lr_predictions)
        actuals_orig = np.zeros_like(y_test[:, -1, :])

        for i in range(lr_predictions.shape[1]):
            lr_predictions_orig[:, i] = dataset.scalers[i].inverse_transform(lr_predictions[:, i].reshape(-1, 1)).flatten()
            actuals_orig[:, i] = dataset.scalers[i].inverse_transform(y_test[:, -1, i].reshape(-1, 1)).flatten()

        lr_predictions = lr_predictions_orig
        actuals = actuals_orig
    else:
        actuals = y_test[:, -1, :]

    # Calculate metrics for Linear Regression
    mse = np.mean((lr_predictions - actuals) ** 2, axis=0)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(lr_predictions - actuals), axis=0)

    denominator = np.mean((actuals - np.mean(actuals, axis=0)) ** 2, axis=0)
    rse = np.sqrt(np.sum(mse) / np.sum(denominator))

    denominator = np.sum(np.abs(actuals - np.mean(actuals, axis=0)), axis=0)
    rae = np.sum(np.sum(np.abs(lr_predictions - actuals), axis=0)) / np.sum(denominator)

    corr = []
    for i in range(lr_predictions.shape[1]):
        if np.std(lr_predictions[:, i]) > 0 and np.std(actuals[:, i]) > 0:
            corr.append(np.corrcoef(lr_predictions[:, i], actuals[:, i])[0, 1])
        else:
            corr.append(0)
    corr = np.mean(corr)

    baseline_results['Linear Regression'] = {
        'predictions': lr_predictions,
        'actuals': actuals,
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'rse': rse,
        'rae': rae,
        'corr': corr
    }

    # ARIMA and VAR baseline models would be similarly implemented
    # But they're computationally expensive for many series, so we'll skip them here

    # Compare results
    print("\n===== Model Comparison =====")
    print(f"{'Model':<20} {'RSE':<10} {'RAE':<10} {'CORR':<10}")
    print("-" * 50)

    for model_name, results in baseline_results.items():
        if 'rse' in results:
            print(f"{model_name:<20} {results['rse']:<10.4f} {results['rae']:<10.4f} {results['corr']:<10.4f}")

    return baseline_results

#############################################################################
# PART 6: MAIN FUNCTION TO RUN THE STRATEGY
#############################################################################

def run_lstnet_agricultural_futures_strategy():
    """
    Run LSTNet strategy for agricultural commodity futures
    """
    # Initialize Bloomberg connection
    bloomberg = BloombergDataManager()
    bloomberg.connect()

    # Define agricultural commodity futures tickers
    tickers = [
        'CF.CZC',   # CZCE cotton
        'SR.CZC',   # CZCE sugar
        'SB.NYB',   # ICE eleventh sugar
        'A.DCE',    # DCE bean
        'B.DCE',    # DCE bean II
        'Y.DCE',    # DCE soybean oil
        'M.DCE',    # DCE cardamom
        'WH.CZC',   # CZCE strong wheat
        'C.DCE',    # DCE corn
        'KC.NYB',   # ICE coffee
        'CC.NYB',   # ICE cocoa
        'OJ.NYB'    # ICE frozen concentrated orange juice
    ]

    # Define date range
    start_date = '2017-01-01'
    end_date = datetime.now().strftime("%Y-%m-%d")

    # Retrieve data from Bloomberg
    data = bloomberg.get_historical_data(tickers, start_date, end_date)

    # Close Bloomberg connection
    bloomberg.close()

    if data is None or data.empty:
        print("❌ Failed to retrieve data. Exiting.")
        return

    # Save raw data
    data.to_csv('agricultural_futures_data.csv')
    print(f"✅ Data saved to agricultural_futures_data.csv")

    # Handle missing values
    data = data.fillna(method='ffill').dropna()
    print(f"✅ Data shape after handling missing values: {data.shape}")

    # Test for stationarity
    stationarity_results = test_stationarity(data)
    print("\n===== Stationarity Test Results =====")
    print(stationarity_results[['ticker', 'adf_statistic', 'p_value', 'is_stationary']])

    # Apply differencing if needed
    non_stationary = stationarity_results[~stationarity_results['is_stationary']]

    if len(non_stationary) > 0:
        print(f"\n⚠️ {len(non_stationary)} series are non-stationary. Applying first-order differencing.")
        data_diff = data.diff().dropna()
    else:
        data_diff = data

    # Initialize dataset
    window_size = 30
    horizon = 12
    dataset = TimeSeriesDataset(window_size=window_size, horizon=horizon)

    # Create datasets
    X_train, y_train, X_val, y_val, X_test, y_test = dataset.create_dataset(data_diff)

    # Create data loaders
    batch_size = 128
    train_tensor_x = torch.FloatTensor(X_train)
    train_tensor_y = torch.FloatTensor(y_train)
    train_dataset = TensorDataset(train_tensor_x, train_tensor_y)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_tensor_x = torch.FloatTensor(X_val)
    val_tensor_y = torch.FloatTensor(y_val)
    val_dataset = TensorDataset(val_tensor_x, val_tensor_y)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    test_tensor_x = torch.FloatTensor(X_test)
    test_tensor_y = torch.FloatTensor(y_test)
    test_dataset = TensorDataset(test_tensor_x, test_tensor_y)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Initialize trainer
    trainer = LSTNetTrainer(
        num_variables=data_diff.shape[1],
        window_size=window_size,
        horizon=horizon,
        skip=24,  # Skip length as suggested in the paper
        learning_rate=0.001
    )

    # Train model
    print("\n===== Training LSTNet Model =====")
    model = trainer.train(train_loader, val_loader, epochs=50, patience=10)

    # Evaluate model
    print("\n===== Evaluating LSTNet Model =====")
    evaluation = trainer.evaluate(test_loader, dataset, data_diff, target_idx=0)

    print(f"RSE: {evaluation['rse']:.4f}")
    print(f"RAE: {evaluation['rae']:.4f}")
    print(f"CORR: {evaluation['corr']:.4f}")

    # Compare with baseline models
    baseline_results = compare_models(data_diff, window_size, horizon, batch_size)

    # Initialize trading strategy
    strategy = AgricultureFuturesStrategy(
        model=model,
        dataset=dataset,
        tickers=tickers,
        threshold=0.005,
        position_size=0.1,
        trade_cost=0.001,
        window_size=window_size,
        horizon=horizon
    )

    # Backtest strategy
    print("\n===== Backtesting Trading Strategy =====")
    backtest_results = strategy.backtest(data_diff)

    # Save results
    backtest_results['daily_performance'].to_csv('backtest_daily_performance.csv')
    backtest_results['trades'].to_csv('backtest_trades.csv')

    print("\n===== Strategy Testing Completed =====")
    return {
        'model': model,
        'evaluation': evaluation,
        'baseline_results': baseline_results,
        'backtest_results': backtest_results,
        'data': data,
        'data_diff': data_diff,
        'dataset': dataset
    }

if __name__ == "__main__":
    print("Starting Agricultural Commodity Futures Trading Strategy...")
    try:
        results = run_lstnet_agricultural_futures_strategy()
        print("Strategy execution completed successfully.")
    except Exception as e:
        import traceback
        print(f"Error executing strategy: {e}")
        traceback.print_exc()