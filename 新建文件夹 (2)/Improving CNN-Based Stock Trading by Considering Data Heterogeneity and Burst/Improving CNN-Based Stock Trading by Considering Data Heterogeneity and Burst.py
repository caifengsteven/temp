import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import talib
from scipy.stats import entropy, chi2_contingency
import warnings
warnings.filterwarnings('ignore')

# Set seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class StockDataProcessor:
    """Class to process stock data and create technical indicators"""
    
    def __init__(self, ticker, start_date, end_date):
        """
        Initialize the stock data processor
        
        Parameters:
        - ticker: Stock ticker symbol
        - start_date: Start date for data collection
        - end_date: End date for data collection
        """
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.data = None
        self.technical_indicators = [
            'SMA', 'EMA', 'WMA', 'DEMA', 'TEMA', 
            'TRIMA', 'KAMA', 'MAMA', 'T3', 'RSI', 
            'WILLR', 'ADX', 'CCI', 'MACD', 'MOM'
        ]
        self.window_lengths = list(range(6, 21))  # 6 to 20 as per the paper
        
    def download_data(self):
        """Download stock data using yfinance"""
        try:
            self.data = yf.download(self.ticker, start=self.start_date, end=self.end_date)
            return True
        except Exception as e:
            print(f"Error downloading data for {self.ticker}: {e}")
            return False
    
    def calculate_technical_indicators(self):
        """Calculate all technical indicators for different window lengths"""
        if self.data is None or len(self.data) == 0:
            print("No data available. Please download data first.")
            return None
        
        # Dictionary to store all calculated indicators
        all_indicators = {}
        
        # Calculate indicators for each window length
        for length in self.window_lengths:
            # Create a dictionary to store indicators for this window length
            indicators = {}
            
            # Simple Moving Average (SMA)
            indicators['SMA'] = talib.SMA(self.data['Close'].values, timeperiod=length)
            
            # Exponential Moving Average (EMA)
            indicators['EMA'] = talib.EMA(self.data['Close'].values, timeperiod=length)
            
            # Weighted Moving Average (WMA)
            indicators['WMA'] = talib.WMA(self.data['Close'].values, timeperiod=length)
            
            # Double Exponential Moving Average (DEMA)
            indicators['DEMA'] = talib.DEMA(self.data['Close'].values, timeperiod=length)
            
            # Triple Exponential Moving Average (TEMA)
            indicators['TEMA'] = talib.TEMA(self.data['Close'].values, timeperiod=length)
            
            # Triangular Moving Average (TRIMA)
            indicators['TRIMA'] = talib.TRIMA(self.data['Close'].values, timeperiod=length)
            
            # Kaufman Adaptive Moving Average (KAMA)
            indicators['KAMA'] = talib.KAMA(self.data['Close'].values, timeperiod=length)
            
            # MESA Adaptive Moving Average (MAMA)
            mama, fama = talib.MAMA(self.data['Close'].values)
            indicators['MAMA'] = mama
            
            # Triple Exponential Moving Average (T3)
            indicators['T3'] = talib.T3(self.data['Close'].values, timeperiod=length)
            
            # Relative Strength Index (RSI)
            indicators['RSI'] = talib.RSI(self.data['Close'].values, timeperiod=length)
            
            # Williams' %R
            indicators['WILLR'] = talib.WILLR(self.data['High'].values, 
                                              self.data['Low'].values, 
                                              self.data['Close'].values, 
                                              timeperiod=length)
            
            # Average Directional Movement Index (ADX)
            indicators['ADX'] = talib.ADX(self.data['High'].values, 
                                          self.data['Low'].values, 
                                          self.data['Close'].values, 
                                          timeperiod=length)
            
            # Commodity Channel Index (CCI)
            indicators['CCI'] = talib.CCI(self.data['High'].values, 
                                          self.data['Low'].values, 
                                          self.data['Close'].values, 
                                          timeperiod=length)
            
            # Moving Average Convergence/Divergence (MACD)
            macd, macdsignal, macdhist = talib.MACD(self.data['Close'].values)
            indicators['MACD'] = macd
            
            # Momentum (MOM)
            indicators['MOM'] = talib.MOM(self.data['Close'].values, timeperiod=length)
            
            all_indicators[length] = indicators
        
        return all_indicators
    
    def create_image_data(self, all_indicators):
        """
        Create 2D image data from technical indicators
        
        Parameters:
        - all_indicators: Dictionary of all calculated technical indicators
        
        Returns:
        - images: List of 2D images (15x15 matrices)
        - labels: List of labels (0=buy, 1=hold, 2=sell)
        - dates: List of corresponding dates
        """
        if all_indicators is None:
            return None, None, None
        
        # Determine valid indices (where all indicators are available)
        max_nan_index = 0
        for length in self.window_lengths:
            for indicator in self.technical_indicators:
                nan_indices = np.isnan(all_indicators[length][indicator])
                if np.any(nan_indices):
                    max_nan_index = max(max_nan_index, np.where(~nan_indices)[0][0])
        
        # Start from the index where all indicators are available
        valid_indices = range(max_nan_index, len(self.data))
        
        # Create images and labels
        images = []
        labels = []
        dates = []
        
        for idx in valid_indices:
            # Create a 15x15 matrix (image)
            image = np.zeros((15, 15))
            
            # Fill the image with technical indicators
            for i, indicator in enumerate(self.technical_indicators):
                for j, length in enumerate(self.window_lengths):
                    image[i, j] = all_indicators[length][indicator][idx]
            
            # Check for NaN values
            if np.isnan(image).any():
                continue
            
            # Generate label based on the closing price compared to the 20-day window
            # As per the paper, we use quartile-based labeling
            window_close = self.data['Close'].iloc[max(0, idx-19):idx+1].values
            current_close = self.data['Close'].iloc[idx]
            
            if len(window_close) < 20:
                continue
                
            q1 = np.percentile(window_close, 25)
            q3 = np.percentile(window_close, 75)
            
            if current_close <= q1:
                label = 0  # Buy
            elif current_close >= q3:
                label = 2  # Sell
            else:
                label = 1  # Hold
            
            images.append(image)
            labels.append(label)
            dates.append(self.data.index[idx])
        
        return np.array(images), np.array(labels), np.array(dates)

class DataNormalizer:
    """Class to implement different normalization techniques"""
    
    @staticmethod
    def min_max_normalization(images):
        """
        Apply standard min-max normalization to the whole images
        
        Parameters:
        - images: Array of images to normalize
        
        Returns:
        - normalized_images: Array of normalized images
        """
        normalized_images = np.zeros_like(images, dtype=np.float32)
        
        for i in range(len(images)):
            min_val = np.min(images[i])
            max_val = np.max(images[i])
            
            if max_val == min_val:
                normalized_images[i] = 0.5
            else:
                normalized_images[i] = (images[i] - min_val) / (max_val - min_val)
        
        return normalized_images
    
    @staticmethod
    def proposed_normalization(images):
        """
        Apply the proposed row-by-row normalization with logarithmic scaling
        
        Parameters:
        - images: Array of images to normalize
        
        Returns:
        - normalized_images: Array of normalized images
        """
        normalized_images = np.zeros_like(images, dtype=np.float32)
        
        for i in range(len(images)):
            for j in range(images.shape[1]):  # For each row (technical indicator)
                # Get the row data
                row = images[i, j, :]
                
                # Apply log transformation to suppress bursty events
                log_transformed = np.log1p(np.abs(row))
                
                # Normalize the log-transformed data
                min_val = np.min(log_transformed)
                max_val = np.max(log_transformed)
                
                if max_val == min_val:
                    normalized_images[i, j, :] = 0.5
                else:
                    normalized_images[i, j, :] = (log_transformed - min_val) / (max_val - min_val)
        
        return normalized_images
    
    @staticmethod
    def compute_entropy(image):
        """
        Compute the entropy of an image
        
        Parameters:
        - image: 2D array representing an image
        
        Returns:
        - entropy_value: Entropy of the image
        """
        # Flatten the image and compute histogram
        hist, _ = np.histogram(image.flatten(), bins=256, density=True)
        
        # Remove zero probabilities
        hist = hist[hist > 0]
        
        # Compute entropy
        return -np.sum(hist * np.log2(hist))
    
    @staticmethod
    def chi_square_test(image):
        """
        Perform chi-square test for heterogeneity
        
        Parameters:
        - image: 2D array representing an image
        
        Returns:
        - chi2_values: Chi-square statistics for each row compared to the first row
        """
        chi2_values = []
        
        # Use the first row as the expected distribution
        expected = image[0, :]
        
        for i in range(1, image.shape[0]):
            observed = image[i, :]
            
            # Replace zeros in expected to avoid division by zero
            expected_safe = np.where(expected == 0, 1e-10, expected)
            
            # Compute chi-square statistic
            chi2 = np.sum((observed - expected)**2 / expected_safe)
            chi2_values.append(chi2)
        
        return np.array(chi2_values)

class CNNModel:
    """Class to implement the CNN model for stock trading"""
    
    def __init__(self, input_shape=(15, 15, 1)):
        """
        Initialize the CNN model
        
        Parameters:
        - input_shape: Shape of the input images
        """
        self.model = self._build_model(input_shape)
        
    def _build_model(self, input_shape):
        """
        Build the CNN model architecture as described in the paper
        
        Parameters:
        - input_shape: Shape of the input images
        
        Returns:
        - model: Compiled CNN model
        """
        model = Sequential([
            # Input layer
            Conv2D(32, (4, 4), activation='relu', padding='same', input_shape=input_shape),
            
            # Second convolutional layer
            Conv2D(64, (4, 4), activation='relu', padding='same'),
            
            # Max pooling layer
            MaxPooling2D(pool_size=(2, 2)),
            
            # Flatten layer
            Flatten(),
            
            # Fully connected layer
            Dense(128, activation='relu'),
            
            # Dropout layer
            Dropout(0.3),
            
            # Output layer
            Dense(3, activation='softmax')  # 3 classes: buy, hold, sell
        ])
        
        # Compile the model
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train(self, X_train, y_train, epochs=50, batch_size=32, validation_split=0.2):
        """
        Train the CNN model
        
        Parameters:
        - X_train: Training images
        - y_train: Training labels
        - epochs: Number of training epochs
        - batch_size: Batch size for training
        - validation_split: Fraction of data to use for validation
        
        Returns:
        - history: Training history
        """
        # Add early stopping
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        # Train the model
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=[early_stopping],
            verbose=1
        )
        
        return history
    
    def predict(self, X):
        """
        Make predictions with the trained model
        
        Parameters:
        - X: Input images
        
        Returns:
        - predictions: Predicted class indices
        """
        # Get predicted probabilities
        probs = self.model.predict(X)
        
        # Convert to class indices
        return np.argmax(probs, axis=1)
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate the model on test data
        
        Parameters:
        - X_test: Test images
        - y_test: Test labels
        
        Returns:
        - metrics: Dictionary of evaluation metrics
        """
        # Get predictions
        y_pred = self.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        
        # Calculate precision, recall, and F1 score for each class
        precision = {}
        recall = {}
        f1 = {}
        
        for cls in range(3):  # 0=buy, 1=hold, 2=sell
            precision[cls] = precision_score(y_test, y_pred, average=None, labels=[cls])[0] if cls in y_test else 0
            recall[cls] = recall_score(y_test, y_pred, average=None, labels=[cls])[0] if cls in y_test else 0
            f1[cls] = f1_score(y_test, y_pred, average=None, labels=[cls])[0] if cls in y_test else 0
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

class LSTMModel:
    """Class to implement the LSTM model for comparison"""
    
    def __init__(self, input_shape):
        """
        Initialize the LSTM model
        
        Parameters:
        - input_shape: Shape of the input data
        """
        self.model = self._build_model(input_shape)
        
    def _build_model(self, input_shape):
        """
        Build the LSTM model architecture
        
        Parameters:
        - input_shape: Shape of the input data
        
        Returns:
        - model: Compiled LSTM model
        """
        model = Sequential([
            # Reshape layer to convert 2D images to sequences
            tf.keras.layers.Reshape((input_shape[0], input_shape[1])),
            
            # LSTM layers
            tf.keras.layers.LSTM(50, return_sequences=True),
            tf.keras.layers.LSTM(50),
            
            # Dense layers
            Dense(50, activation='relu'),
            Dropout(0.3),
            Dense(3, activation='softmax')  # 3 classes: buy, hold, sell
        ])
        
        # Compile the model
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train(self, X_train, y_train, epochs=50, batch_size=32, validation_split=0.2):
        """
        Train the LSTM model
        
        Parameters:
        - X_train: Training data
        - y_train: Training labels
        - epochs: Number of training epochs
        - batch_size: Batch size for training
        - validation_split: Fraction of data to use for validation
        
        Returns:
        - history: Training history
        """
        # Add early stopping
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        # Train the model
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=[early_stopping],
            verbose=1
        )
        
        return history
    
    def predict(self, X):
        """
        Make predictions with the trained model
        
        Parameters:
        - X: Input data
        
        Returns:
        - predictions: Predicted class indices
        """
        # Get predicted probabilities
        probs = self.model.predict(X)
        
        # Convert to class indices
        return np.argmax(probs, axis=1)

class RegressionModel:
    """Class to implement the Regression model for comparison"""
    
    def __init__(self, alpha=0.01):
        """
        Initialize the Regression model
        
        Parameters:
        - alpha: L1 regularization parameter for Lasso
        """
        self.model = Lasso(alpha=alpha)
        
    def train(self, X_train, y_train):
        """
        Train the Regression model
        
        Parameters:
        - X_train: Training data (flattened images)
        - y_train: Training labels
        
        Returns:
        - self: Trained model
        """
        # Flatten the images
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        
        # Train the model
        self.model.fit(X_train_flat, y_train)
        
        return self
    
    def predict(self, X):
        """
        Make predictions with the trained model
        
        Parameters:
        - X: Input data
        
        Returns:
        - predictions: Predicted class indices
        """
        # Flatten the images
        X_flat = X.reshape(X.shape[0], -1)
        
        # Get predicted values
        y_pred_raw = self.model.predict(X_flat)
        
        # Convert to class indices (0=buy, 1=hold, 2=sell)
        # We use rounding and clipping to convert continuous predictions to discrete classes
        y_pred = np.round(np.clip(y_pred_raw, 0, 2)).astype(int)
        
        return y_pred

class TradingSimulator:
    """Class to simulate trading based on model predictions"""
    
    def __init__(self, initial_cash=10000, initial_shares=0):
        """
        Initialize the trading simulator
        
        Parameters:
        - initial_cash: Initial cash amount
        - initial_shares: Initial number of shares
        """
        self.initial_cash = initial_cash
        self.initial_shares = initial_shares
        
    def simulate(self, dates, prices, predictions):
        """
        Simulate trading based on predictions
        
        Parameters:
        - dates: Array of trading dates
        - prices: Array of stock prices
        - predictions: Array of predicted actions (0=buy, 1=hold, 2=sell)
        
        Returns:
        - portfolio_values: Dictionary with dates and portfolio values
        - transactions: List of transaction details
        """
        # Initialize portfolio
        cash = self.initial_cash
        shares = self.initial_shares
        
        # Track portfolio value over time
        portfolio_values = {}
        transactions = []
        
        for i, (date, price, prediction) in enumerate(zip(dates, prices, predictions)):
            # Calculate portfolio value before transaction
            portfolio_value = cash + shares * price
            
            # Record portfolio value
            portfolio_values[date] = portfolio_value
            
            # Execute transaction based on prediction
            if prediction == 0:  # Buy
                # Use 50% of available cash
                cash_to_spend = cash * 0.5
                shares_to_buy = cash_to_spend / price
                
                cash -= cash_to_spend
                shares += shares_to_buy
                
                transactions.append({
                    'date': date,
                    'action': 'buy',
                    'price': price,
                    'shares': shares_to_buy,
                    'cash_used': cash_to_spend,
                    'portfolio_value': portfolio_value
                })
                
            elif prediction == 2:  # Sell
                # Sell 50% of shares
                shares_to_sell = shares * 0.5
                cash_gained = shares_to_sell * price
                
                cash += cash_gained
                shares -= shares_to_sell
                
                transactions.append({
                    'date': date,
                    'action': 'sell',
                    'price': price,
                    'shares': shares_to_sell,
                    'cash_gained': cash_gained,
                    'portfolio_value': portfolio_value
                })
            
            # For hold (prediction == 1), do nothing
        
        # Calculate final portfolio value
        final_price = prices[-1]
        final_value = cash + shares * final_price
        
        # Calculate returns
        initial_value = self.initial_cash + self.initial_shares * prices[0]
        total_return = (final_value - initial_value) / initial_value
        
        # Calculate annualized return
        years = (dates[-1] - dates[0]).days / 365.25
        if years > 0:
            annualized_return = (1 + total_return) ** (1 / years) - 1
        else:
            annualized_return = 0
        
        return {
            'portfolio_values': portfolio_values,
            'transactions': transactions,
            'initial_value': initial_value,
            'final_value': final_value,
            'total_return': total_return,
            'annualized_return': annualized_return,
            'final_cash': cash,
            'final_shares': shares
        }

def run_experiment(ticker='AAPL', start_train='2005-01-01', end_train='2015-12-31', 
                  start_test='2016-01-01', end_test='2021-12-31'):
    """
    Run the full experiment for a given stock
    
    Parameters:
    - ticker: Stock ticker symbol
    - start_train: Start date for training data
    - end_train: End date for training data
    - start_test: Start date for testing data
    - end_test: End date for testing data
    
    Returns:
    - results: Dictionary of experiment results
    """
    print(f"\nRunning experiment for {ticker}...")
    
    # Process training data
    processor_train = StockDataProcessor(ticker, start_train, end_train)
    if not processor_train.download_data():
        return None
    
    all_indicators_train = processor_train.calculate_technical_indicators()
    X_train, y_train, dates_train = processor_train.create_image_data(all_indicators_train)
    
    # Process test data
    processor_test = StockDataProcessor(ticker, start_test, end_test)
    if not processor_test.download_data():
        return None
    
    all_indicators_test = processor_test.calculate_technical_indicators()
    X_test, y_test, dates_test = processor_test.create_image_data(all_indicators_test)
    
    if X_train is None or X_test is None:
        print(f"Not enough data for {ticker}")
        return None
    
    # Normalize data using different methods
    X_train_minmax = DataNormalizer.min_max_normalization(X_train)
    X_test_minmax = DataNormalizer.min_max_normalization(X_test)
    
    X_train_proposed = DataNormalizer.proposed_normalization(X_train)
    X_test_proposed = DataNormalizer.proposed_normalization(X_test)
    
    # Reshape for CNN input
    X_train_minmax = X_train_minmax.reshape(-1, 15, 15, 1)
    X_test_minmax = X_test_minmax.reshape(-1, 15, 15, 1)
    
    X_train_proposed = X_train_proposed.reshape(-1, 15, 15, 1)
    X_test_proposed = X_test_proposed.reshape(-1, 15, 15, 1)
    
    # Train and evaluate CNN with min-max normalization
    print("Training CNN with min-max normalization...")
    cnn_minmax = CNNModel(input_shape=(15, 15, 1))
    cnn_minmax.train(X_train_minmax, y_train, epochs=30, batch_size=32)
    
    # Train and evaluate CNN with proposed normalization
    print("Training CNN with proposed normalization...")
    cnn_proposed = CNNModel(input_shape=(15, 15, 1))
    cnn_proposed.train(X_train_proposed, y_train, epochs=30, batch_size=32)
    
    # Train and evaluate LSTM model
    print("Training LSTM model...")
    lstm = LSTMModel(input_shape=(15, 15))
    lstm.train(X_train_minmax, y_train, epochs=30, batch_size=32)
    
    # Train and evaluate Regression model
    print("Training Regression model...")
    regression = RegressionModel(alpha=0.01)
    regression.train(X_train_minmax, y_train)
    
    # Make predictions
    y_pred_cnn_minmax = cnn_minmax.predict(X_test_minmax)
    y_pred_cnn_proposed = cnn_proposed.predict(X_test_proposed)
    y_pred_lstm = lstm.predict(X_test_minmax)
    y_pred_regression = regression.predict(X_test_minmax)
    
    # Evaluate models
    metrics_cnn_minmax = cnn_minmax.evaluate(X_test_minmax, y_test)
    metrics_cnn_proposed = cnn_proposed.evaluate(X_test_proposed, y_test)
    
    # Calculate metrics for LSTM and Regression models
    metrics_lstm = {
        'accuracy': accuracy_score(y_test, y_pred_lstm),
        'precision': {cls: precision_score(y_test, y_pred_lstm, average=None, labels=[cls])[0] 
                     if cls in y_test else 0 for cls in range(3)},
        'recall': {cls: recall_score(y_test, y_pred_lstm, average=None, labels=[cls])[0] 
                  if cls in y_test else 0 for cls in range(3)},
        'f1': {cls: f1_score(y_test, y_pred_lstm, average=None, labels=[cls])[0] 
              if cls in y_test else 0 for cls in range(3)}
    }
    
    metrics_regression = {
        'accuracy': accuracy_score(y_test, y_pred_regression),
        'precision': {cls: precision_score(y_test, y_pred_regression, average=None, labels=[cls])[0] 
                     if cls in y_test else 0 for cls in range(3)},
        'recall': {cls: recall_score(y_test, y_pred_regression, average=None, labels=[cls])[0] 
                  if cls in y_test else 0 for cls in range(3)},
        'f1': {cls: f1_score(y_test, y_pred_regression, average=None, labels=[cls])[0] 
              if cls in y_test else 0 for cls in range(3)}
    }
    
    # Simulate trading
    test_prices = processor_test.data['Close'].values
    valid_indices = range(len(dates_test))
    
    simulator = TradingSimulator(initial_cash=10000, initial_shares=10000/test_prices[0])
    
    results_cnn_minmax = simulator.simulate(
        dates_test, 
        test_prices[valid_indices], 
        y_pred_cnn_minmax
    )
    
    results_cnn_proposed = simulator.simulate(
        dates_test, 
        test_prices[valid_indices], 
        y_pred_cnn_proposed
    )
    
    results_lstm = simulator.simulate(
        dates_test, 
        test_prices[valid_indices], 
        y_pred_lstm
    )
    
    results_regression = simulator.simulate(
        dates_test, 
        test_prices[valid_indices], 
        y_pred_regression
    )
    
    # Print results
    print(f"\nResults for {ticker}:")
    print(f"CNN (Min-Max) Accuracy: {metrics_cnn_minmax['accuracy']:.4f}, Return: {results_cnn_minmax['annualized_return']:.4f}")
    print(f"CNN (Proposed) Accuracy: {metrics_cnn_proposed['accuracy']:.4f}, Return: {results_cnn_proposed['annualized_return']:.4f}")
    print(f"LSTM Accuracy: {metrics_lstm['accuracy']:.4f}, Return: {results_lstm['annualized_return']:.4f}")
    print(f"Regression Accuracy: {metrics_regression['accuracy']:.4f}, Return: {results_regression['annualized_return']:.4f}")
    
    # Plot results
    plt.figure(figsize=(12, 6))
    
    # Convert dates to list for plotting
    dates_list = list(dates_test)
    
    # Get portfolio values
    portfolio_values_cnn_minmax = list(results_cnn_minmax['portfolio_values'].values())
    portfolio_values_cnn_proposed = list(results_cnn_proposed['portfolio_values'].values())
    portfolio_values_lstm = list(results_lstm['portfolio_values'].values())
    portfolio_values_regression = list(results_regression['portfolio_values'].values())
    
    # Plot portfolio values over time
    plt.plot(dates_list, portfolio_values_cnn_minmax, 'b-', label='CNN (Min-Max)')
    plt.plot(dates_list, portfolio_values_cnn_proposed, 'r-', label='CNN (Proposed)')
    plt.plot(dates_list, portfolio_values_lstm, 'g-', label='LSTM')
    plt.plot(dates_list, portfolio_values_regression, 'k-', label='Regression')
    
    plt.title(f'{ticker} Portfolio Value Over Time')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value ($)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{ticker}_portfolio_value.png')
    plt.close()
    
    # Return results for further analysis
    return {
        'ticker': ticker,
        'metrics_cnn_minmax': metrics_cnn_minmax,
        'metrics_cnn_proposed': metrics_cnn_proposed,
        'metrics_lstm': metrics_lstm,
        'metrics_regression': metrics_regression,
        'results_cnn_minmax': results_cnn_minmax,
        'results_cnn_proposed': results_cnn_proposed,
        'results_lstm': results_lstm,
        'results_regression': results_regression
    }

def analyze_heterogeneity():
    """Analyze and visualize data heterogeneity as described in the paper"""
    # Download sample stock data
    ticker = 'AAPL'
    processor = StockDataProcessor(ticker, '2015-01-01', '2015-12-31')
    processor.download_data()
    all_indicators = processor.calculate_technical_indicators()
    X, _, _ = processor.create_image_data(all_indicators)
    
    if X is None or len(X) == 0:
        print("Not enough data for analysis")
        return
    
    # Select a sample image
    sample_image = X[0]
    
    # Calculate entropy of the original image
    original_entropy = DataNormalizer.compute_entropy(sample_image)
    
    # Normalize using min-max normalization
    minmax_normalized = DataNormalizer.min_max_normalization(np.array([sample_image]))[0]
    minmax_entropy = DataNormalizer.compute_entropy(minmax_normalized)
    
    # Normalize using proposed method
    proposed_normalized = DataNormalizer.proposed_normalization(np.array([sample_image]))[0]
    proposed_entropy = DataNormalizer.compute_entropy(proposed_normalized)
    
    # Perform chi-square test
    chi2_values = DataNormalizer.chi_square_test(sample_image)
    
    # Create plots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    im0 = axes[0].imshow(sample_image, cmap='viridis')
    axes[0].set_title(f'Original Image\nEntropy: {original_entropy:.2f}')
    plt.colorbar(im0, ax=axes[0])
    
    # Min-max normalized
    im1 = axes[1].imshow(minmax_normalized, cmap='viridis')
    axes[1].set_title(f'Min-Max Normalized\nEntropy: {minmax_entropy:.2f}')
    plt.colorbar(im1, ax=axes[1])
    
    # Proposed normalized
    im2 = axes[2].imshow(proposed_normalized, cmap='viridis')
    axes[2].set_title(f'Proposed Normalized\nEntropy: {proposed_entropy:.2f}')
    plt.colorbar(im2, ax=axes[2])
    
    plt.tight_layout()
    plt.savefig('heterogeneity_analysis.png')
    plt.close()
    
    # Plot chi-square test results
    plt.figure(figsize=(10, 5))
    plt.bar(range(1, len(chi2_values) + 1), chi2_values)
    plt.axhline(y=29.141, color='r', linestyle='-', label='Significance Level (1%)')
    plt.xlabel('Row Index')
    plt.ylabel('Chi-Square Statistic')
    plt.title('Chi-Square Test for Heterogeneity')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('chi_square_test.png')
    plt.close()
    
    # Plot time series of selected indicators
    plt.figure(figsize=(15, 10))
    
    # Extract RSI, Williams%R, and EMA
    rsi_values = all_indicators[10]['RSI'][:447]  # Limit to 447 days as in the paper
    willr_values = all_indicators[10]['WILLR'][:447]
    ema_values = all_indicators[10]['EMA'][:447]
    
    # Plot time series
    plt.subplot(3, 2, 1)
    plt.plot(rsi_values)
    plt.title('Time Series of RSI')
    plt.grid(True)
    
    plt.subplot(3, 2, 2)
    plt.hist(rsi_values, bins=20)
    plt.title('Histogram of RSI')
    plt.grid(True)
    
    plt.subplot(3, 2, 3)
    plt.plot(willr_values)
    plt.title('Time Series of Williams%R')
    plt.grid(True)
    
    plt.subplot(3, 2, 4)
    plt.hist(willr_values, bins=20)
    plt.title('Histogram of Williams%R')
    plt.grid(True)
    
    plt.subplot(3, 2, 5)
    plt.plot(ema_values)
    plt.title('Time Series of EMA')
    plt.grid(True)
    
    plt.subplot(3, 2, 6)
    plt.hist(ema_values, bins=20)
    plt.title('Histogram of EMA')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('indicator_analysis.png')
    plt.close()
    
    print("Heterogeneity analysis completed. See output images.")

def main():
    """Main function to run experiments"""
    # Analyze data heterogeneity
    print("Analyzing data heterogeneity...")
    analyze_heterogeneity()
    
    # Define list of stocks to test
    stocks = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'FB']
    
    # Run experiments for each stock
    results = {}
    for ticker in stocks:
        result = run_experiment(ticker)
        if result:
            results[ticker] = result
    
    # Summarize results
    print("\n----- Summary of Results -----")
    print("Ticker | CNN (Min-Max) | CNN (Proposed) | LSTM | Regression")
    print("-----------------------------------------------------------")
    
    for ticker, result in results.items():
        cnn_minmax_return = result['results_cnn_minmax']['annualized_return']
        cnn_proposed_return = result['results_cnn_proposed']['annualized_return']
        lstm_return = result['results_lstm']['annualized_return']
        regression_return = result['results_regression']['annualized_return']
        
        print(f"{ticker} | {cnn_minmax_return:.4f} | {cnn_proposed_return:.4f} | {lstm_return:.4f} | {regression_return:.4f}")
    
    # Create comparison chart
    annualized_returns = {
        'CNN (Min-Max)': [results[ticker]['results_cnn_minmax']['annualized_return'] for ticker in results],
        'CNN (Proposed)': [results[ticker]['results_cnn_proposed']['annualized_return'] for ticker in results],
        'LSTM': [results[ticker]['results_lstm']['annualized_return'] for ticker in results],
        'Regression': [results[ticker]['results_regression']['annualized_return'] for ticker in results]
    }
    
    plt.figure(figsize=(12, 6))
    x = np.arange(len(results))
    width = 0.2
    
    plt.bar(x - 1.5*width, annualized_returns['CNN (Min-Max)'], width, label='CNN (Min-Max)')
    plt.bar(x - 0.5*width, annualized_returns['CNN (Proposed)'], width, label='CNN (Proposed)')
    plt.bar(x + 0.5*width, annualized_returns['LSTM'], width, label='LSTM')
    plt.bar(x + 1.5*width, annualized_returns['Regression'], width, label='Regression')
    
    plt.xlabel('Stock')
    plt.ylabel('Annualized Return')
    plt.title('Comparison of Trading Strategies')
    plt.xticks(x, results.keys())
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('trading_strategies_comparison.png')
    plt.close()

if __name__ == "__main__":
    main()


def simulate_stock_data(ticker='SIM', periods=1000, volatility=0.02):
    """
    Simulate stock data for testing
    
    Parameters:
    - ticker: Simulated ticker symbol
    - periods: Number of trading days to simulate
    - volatility: Daily volatility
    
    Returns:
    - df: DataFrame with simulated stock data
    """
    # Generate random returns
    returns = np.random.normal(0.0005, volatility, periods)
    
    # Add occasional jumps (burst events)
    jump_indices = np.random.choice(periods, size=int(periods * 0.03), replace=False)
    jump_sizes = np.random.normal(0, volatility * 5, size=len(jump_indices))
    returns[jump_indices] += jump_sizes
    
    # Create price series
    prices = 100 * np.cumprod(1 + returns)
    
    # Create date range
    start_date = pd.Timestamp('2010-01-01')
    dates = [start_date + pd.Timedelta(days=i) for i in range(periods)]
    
    # Create DataFrame
    df = pd.DataFrame({
        'Open': prices * (1 - 0.002),
        'High': prices * (1 + 0.005),
        'Low': prices * (1 - 0.005),
        'Close': prices,
        'Volume': np.random.randint(1000000, 10000000, periods)
    }, index=dates)
    
    return df

def test_with_simulated_data():
    """Test the trading strategy with simulated data"""
    print("Testing with simulated data...")
    
    # Simulate stock data
    train_data = simulate_stock_data(periods=1500, volatility=0.02)
    test_data = simulate_stock_data(periods=500, volatility=0.025)
    
    # Create a StockDataProcessor-like object for simulated data
    class SimulatedDataProcessor:
        def __init__(self, train_data, test_data):
            self.train_data = train_data
            self.test_data = test_data
            self.technical_indicators = [
                'SMA', 'EMA', 'WMA', 'DEMA', 'TEMA', 
                'TRIMA', 'KAMA', 'MAMA', 'T3', 'RSI', 
                'WILLR', 'ADX', 'CCI', 'MACD', 'MOM'
            ]
            self.window_lengths = list(range(6, 21))  # 6 to 20 as per the paper
            
        def calculate_indicators(self, data):
            """Calculate technical indicators for the simulated data"""
            # Dictionary to store all calculated indicators
            all_indicators = {}
            
            # Calculate indicators for each window length
            for length in self.window_lengths:
                # Create a dictionary to store indicators for this window length
                indicators = {}
                
                # Simple Moving Average (SMA)
                indicators['SMA'] = talib.SMA(data['Close'].values, timeperiod=length)
                
                # Exponential Moving Average (EMA)
                indicators['EMA'] = talib.EMA(data['Close'].values, timeperiod=length)
                
                # Weighted Moving Average (WMA)
                indicators['WMA'] = talib.WMA(data['Close'].values, timeperiod=length)
                
                # Double Exponential Moving Average (DEMA)
                indicators['DEMA'] = talib.DEMA(data['Close'].values, timeperiod=length)
                
                # Triple Exponential Moving Average (TEMA)
                indicators['TEMA'] = talib.TEMA(data['Close'].values, timeperiod=length)
                
                # Triangular Moving Average (TRIMA)
                indicators['TRIMA'] = talib.TRIMA(data['Close'].values, timeperiod=length)
                
                # Kaufman Adaptive Moving Average (KAMA)
                indicators['KAMA'] = talib.KAMA(data['Close'].values, timeperiod=length)
                
                # MESA Adaptive Moving Average (MAMA)
                mama, fama = talib.MAMA(data['Close'].values)
                indicators['MAMA'] = mama
                
                # Triple Exponential Moving Average (T3)
                indicators['T3'] = talib.T3(data['Close'].values, timeperiod=length)
                
                # Relative Strength Index (RSI)
                indicators['RSI'] = talib.RSI(data['Close'].values, timeperiod=length)
                
                # Williams' %R
                indicators['WILLR'] = talib.WILLR(data['High'].values, 
                                                data['Low'].values, 
                                                data['Close'].values, 
                                                timeperiod=length)
                
                # Average Directional Movement Index (ADX)
                indicators['ADX'] = talib.ADX(data['High'].values, 
                                            data['Low'].values, 
                                            data['Close'].values, 
                                            timeperiod=length)
                
                # Commodity Channel Index (CCI)
                indicators['CCI'] = talib.CCI(data['High'].values, 
                                            data['Low'].values, 
                                            data['Close'].values, 
                                            timeperiod=length)
                
                # Moving Average Convergence/Divergence (MACD)
                macd, macdsignal, macdhist = talib.MACD(data['Close'].values)
                indicators['MACD'] = macd
                
                # Momentum (MOM)
                indicators['MOM'] = talib.MOM(data['Close'].values, timeperiod=length)
                
                all_indicators[length] = indicators
            
            return all_indicators
        
        def create_image_data(self, data, all_indicators):
            """Create 2D image data from technical indicators"""
            # Determine valid indices (where all indicators are available)
            max_nan_index = 0
            for length in self.window_lengths:
                for indicator in self.technical_indicators:
                    nan_indices = np.isnan(all_indicators[length][indicator])
                    if np.any(nan_indices):
                        max_nan_index = max(max_nan_index, np.where(~nan_indices)[0][0])
            
            # Start from the index where all indicators are available
            valid_indices = range(max_nan_index, len(data))
            
            # Create images and labels
            images = []
            labels = []
            dates = []
            
            for idx in valid_indices:
                # Create a 15x15 matrix (image)
                image = np.zeros((15, 15))
                
                # Fill the image with technical indicators
                for i, indicator in enumerate(self.technical_indicators):
                    for j, length in enumerate(self.window_lengths):
                        image[i, j] = all_indicators[length][indicator][idx]
                
                # Check for NaN values
                if np.isnan(image).any():
                    continue
                
                # Generate label based on the closing price compared to the 20-day window
                window_close = data['Close'].iloc[max(0, idx-19):idx+1].values
                current_close = data['Close'].iloc[idx]
                
                if len(window_close) < 20:
                    continue
                    
                q1 = np.percentile(window_close, 25)
                q3 = np.percentile(window_close, 75)
                
                if current_close <= q1:
                    label = 0  # Buy
                elif current_close >= q3:
                    label = 2  # Sell
                else:
                    label = 1  # Hold
                
                images.append(image)
                labels.append(label)
                dates.append(data.index[idx])
            
            return np.array(images), np.array(labels), np.array(dates)
        
        def get_train_data(self):
            """Get training data"""
            all_indicators = self.calculate_indicators(self.train_data)
            return self.create_image_data(self.train_data, all_indicators)
        
        def get_test_data(self):
            """Get test data"""
            all_indicators = self.calculate_indicators(self.test_data)
            return self.create_image_data(self.test_data, all_indicators)
    
    # Process simulated data
    processor = SimulatedDataProcessor(train_data, test_data)
    X_train, y_train, dates_train = processor.get_train_data()
    X_test, y_test, dates_test = processor.get_test_data()
    
    # Normalize data using different methods
    X_train_minmax = DataNormalizer.min_max_normalization(X_train)
    X_test_minmax = DataNormalizer.min_max_normalization(X_test)
    
    X_train_proposed = DataNormalizer.proposed_normalization(X_train)
    X_test_proposed = DataNormalizer.proposed_normalization(X_test)
    
    # Analyze heterogeneity
    sample_image = X_train[0]
    original_entropy = DataNormalizer.compute_entropy(sample_image)
    minmax_entropy = DataNormalizer.compute_entropy(X_train_minmax[0])
    proposed_entropy = DataNormalizer.compute_entropy(X_train_proposed[0])
    
    print(f"Entropy Analysis:")
    print(f"Original: {original_entropy:.4f}")
    print(f"Min-Max: {minmax_entropy:.4f}")
    print(f"Proposed: {proposed_entropy:.4f}")
    
    # Reshape for CNN input
    X_train_minmax = X_train_minmax.reshape(-1, 15, 15, 1)
    X_test_minmax = X_test_minmax.reshape(-1, 15, 15, 1)
    
    X_train_proposed = X_train_proposed.reshape(-1, 15, 15, 1)
    X_test_proposed = X_test_proposed.reshape(-1, 15, 15, 1)
    
    # Train and evaluate CNN with min-max normalization
    print("Training CNN with min-max normalization...")
    cnn_minmax = CNNModel(input_shape=(15, 15, 1))
    cnn_minmax.train(X_train_minmax, y_train, epochs=20, batch_size=32)
    
    # Train and evaluate CNN with proposed normalization
    print("Training CNN with proposed normalization...")
    cnn_proposed = CNNModel(input_shape=(15, 15, 1))
    cnn_proposed.train(X_train_proposed, y_train, epochs=20, batch_size=32)
    
    # Make predictions
    y_pred_cnn_minmax = cnn_minmax.predict(X_test_minmax)
    y_pred_cnn_proposed = cnn_proposed.predict(X_test_proposed)
    
    # Evaluate models
    metrics_cnn_minmax = cnn_minmax.evaluate(X_test_minmax, y_test)
    metrics_cnn_proposed = cnn_proposed.evaluate(X_test_proposed, y_test)
    
    print("\nModel Evaluation:")
    print(f"CNN (Min-Max) Accuracy: {metrics_cnn_minmax['accuracy']:.4f}")
    print(f"CNN (Proposed) Accuracy: {metrics_cnn_proposed['accuracy']:.4f}")
    
    # Simulate trading
    simulator = TradingSimulator(initial_cash=10000, initial_shares=10000/test_data['Close'].iloc[0])
    
    results_cnn_minmax = simulator.simulate(
        dates_test, 
        test_data['Close'].loc[dates_test].values, 
        y_pred_cnn_minmax
    )
    
    results_cnn_proposed = simulator.simulate(
        dates_test, 
        test_data['Close'].loc[dates_test].values, 
        y_pred_cnn_proposed
    )
    
    print("\nTrading Simulation Results:")
    print(f"CNN (Min-Max) - Final Value: ${results_cnn_minmax['final_value']:.2f}, Return: {results_cnn_minmax['total_return']:.4f}")
    print(f"CNN (Proposed) - Final Value: ${results_cnn_proposed['final_value']:.2f}, Return: {results_cnn_proposed['total_return']:.4f}")
    
    # Plot portfolio values
    plt.figure(figsize=(12, 6))
    
    # Get portfolio values
    portfolio_values_cnn_minmax = list(results_cnn_minmax['portfolio_values'].values())
    portfolio_values_cnn_proposed = list(results_cnn_proposed['portfolio_values'].values())
    
    # Plot portfolio values over time
    plt.plot(dates_test, portfolio_values_cnn_minmax, 'b-', label='CNN (Min-Max)')
    plt.plot(dates_test, portfolio_values_cnn_proposed, 'r-', label='CNN (Proposed)')
    
    # Plot buy/hold/sell markers for proposed method
    buys = [date for date, pred in zip(dates_test, y_pred_cnn_proposed) if pred == 0]
    sells = [date for date, pred in zip(dates_test, y_pred_cnn_proposed) if pred == 2]
    
    # Plot markers at portfolio values
    for buy_date in buys[:10]:  # Limit to first 10 for clarity
        idx = np.where(dates_test == buy_date)[0][0]
        plt.scatter(buy_date, portfolio_values_cnn_proposed[idx], color='green', marker='^', s=100)
    
    for sell_date in sells[:10]:  # Limit to first 10 for clarity
        idx = np.where(dates_test == sell_date)[0][0]
        plt.scatter(sell_date, portfolio_values_cnn_proposed[idx], color='red', marker='v', s=100)
    
    plt.title('Simulated Portfolio Value Over Time')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value ($)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('simulated_portfolio_value.png')
    plt.show()
    
    return {
        'metrics_cnn_minmax': metrics_cnn_minmax,
        'metrics_cnn_proposed': metrics_cnn_proposed,
        'results_cnn_minmax': results_cnn_minmax,
        'results_cnn_proposed': results_cnn_proposed
    }

# Run the test with simulated data
test_results = test_with_simulated_data()

