import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, TensorDataset
import blpapi
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler  # Using MinMaxScaler instead of StandardScaler
import warnings
warnings.filterwarnings("ignore")

# Import our LSTNet model and related classes
from lstnet_model import LSTNet, LSTNetTrainer

# Set seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class BloombergDataManager:
    def __init__(self):
        """Initialize Bloomberg connection"""
        self.session = None
        self.connected = False
        
    def connect(self):
        """Establish connection to Bloomberg API"""
        try:
            self.session = blpapi.Session()
            if not self.session.start():
                print("❌ Failed to start Bloomberg session.")
                return False
            
            if not self.session.openService("//blp/refdata"):
                print("❌ Failed to open //blp/refdata service.")
                return False
            
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
    
    def get_historical_data(self, tickers, start_date, end_date, fields=['PX_LAST']):
        """
        Retrieve historical data from Bloomberg
        
        Args:
            tickers: List of Bloomberg tickers
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            fields: Bloomberg fields to retrieve
            
        Returns:
            DataFrame with historical data
        """
        if not self.check_connection():
            print("❌ Bloomberg connection not available. Cannot retrieve data.")
            return None
        
        try:
            print(f"📊 Retrieving historical data for {len(tickers)} tickers ({start_date} to {end_date})...")
            
            # Get the reference data service
            refDataService = self.session.getService("//blp/refdata")
            
            # Create a request for historical data
            request = refDataService.createRequest("HistoricalDataRequest")
            
            # Set the securities
            for ticker in tickers:
                request.append("securities", ticker)
            
            # Set the fields
            for field in fields:
                request.append("fields", field)
            
            # Set the date range
            request.set("startDate", start_date.replace("-", ""))
            request.set("endDate", end_date.replace("-", ""))
            
            # Send the request
            self.session.sendRequest(request)
            
            # Process the response
            data_dict = {}
            dates = set()
            
            while True:
                event = self.session.nextEvent(500)
                for msg in event:
                    if msg.messageType() == blpapi.Name("HistoricalDataResponse"):
                        securityData = msg.getElement("securityData")
                        security = securityData.getElementAsString("security")
                        fieldData = securityData.getElement("fieldData")
                        
                        for i in range(fieldData.numValues()):
                            row = fieldData.getValue(i)
                            date = row.getElementAsDatetime("date").strftime("%Y-%m-%d")
                            dates.add(date)
                            
                            for field in fields:
                                if row.hasElement(field):
                                    value = row.getElementAsFloat(field)
                                    if security not in data_dict:
                                        data_dict[security] = {}
                                    if date not in data_dict[security]:
                                        data_dict[security][date] = {}
                                    data_dict[security][date][field] = value
                
                if event.eventType() == blpapi.Event.RESPONSE:
                    break
            
            # Convert to DataFrame
            if not data_dict:
                print("⚠️ No data retrieved from Bloomberg")
                return None
            
            # Create a multi-index DataFrame
            dates_list = sorted(list(dates))
            index = pd.DatetimeIndex(dates_list)
            
            if len(fields) == 1:
                # Single field - create a simple DataFrame
                data = pd.DataFrame(index=index)
                for security in data_dict:
                    values = []
                    for date in dates_list:
                        if date in data_dict[security] and fields[0] in data_dict[security][date]:
                            values.append(data_dict[security][date][fields[0]])
                        else:
                            values.append(np.nan)
                    data[security] = values
            else:
                # Multiple fields - create a multi-level column DataFrame
                columns = pd.MultiIndex.from_product([tickers, fields])
                data = pd.DataFrame(index=index, columns=columns)
                
                for security in data_dict:
                    for date in dates_list:
                        if date in data_dict[security]:
                            for field in fields:
                                if field in data_dict[security][date]:
                                    data.loc[date, (security, field)] = data_dict[security][date][field]
            
            print(f"✅ Successfully retrieved data with shape: {data.shape}")
            return data
            
        except Exception as e:
            print(f"❌ Error retrieving data from Bloomberg: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def close(self):
        """Close Bloomberg connection"""
        if self.connected and self.session is not None:
            try:
                self.session.stop()
                self.connected = False
                print("✅ Bloomberg connection closed")
            except Exception as e:
                print(f"❌ Error closing Bloomberg connection: {e}")

class ImprovedTimeSeriesDataset:
    def __init__(self, window_size=30, horizon=5, normalize=True):
        """
        Initialize TimeSeriesDataset with improved preprocessing
        
        Args:
            window_size: Input window size
            horizon: Forecasting horizon (reduced from 12 to 5)
            normalize: Whether to normalize the data
        """
        self.window_size = window_size
        self.horizon = horizon
        self.normalize = normalize
        self.scalers = None
        
    def create_dataset(self, data, train_size=0.7, val_size=0.15):
        """
        Create dataset from time series data with improved preprocessing
        
        Args:
            data: DataFrame with time series data
            train_size: Proportion of data for training (increased from 0.6 to 0.7)
            val_size: Proportion of data for validation (reduced from 0.2 to 0.15)
            
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
        
        # Normalize data using MinMaxScaler instead of StandardScaler
        if self.normalize:
            self.scalers = []
            train_normalized = np.zeros_like(train_data, dtype=np.float32)
            val_normalized = np.zeros_like(val_data, dtype=np.float32)
            test_normalized = np.zeros_like(test_data, dtype=np.float32)
            
            for i in range(n_features):
                scaler = MinMaxScaler(feature_range=(-1, 1))  # Scale to [-1, 1] range
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
            # Only predict the last step of the horizon instead of all steps
            y.append(data[i+self.window_size+self.horizon-1:i+self.window_size+self.horizon])
        
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

def calculate_returns(data):
    """
    Calculate returns from price data
    
    Args:
        data: DataFrame with price data
        
    Returns:
        DataFrame with returns
    """
    returns = data.pct_change().dropna()
    print(f"✅ Calculated returns with shape: {returns.shape}")
    return returns

def train_lstnet_model():
    """
    Train LSTNet model on Bloomberg data with improved preprocessing
    """
    # Initialize Bloomberg connection
    bloomberg = BloombergDataManager()
    bloomberg.connect()
    
    # Define agricultural commodity futures tickers with proper Bloomberg syntax
    tickers = [
        'SB1 Comdty',       # ICE eleventh sugar
        'KC1 Comdty',       # ICE coffee
        'CC1 Comdty'        # ICE cocoa
    ]
    
    # Define date range
    start_date = '2018-01-01'  # Use more historical data for training
    end_date = datetime.now().strftime("%Y-%m-%d")
    
    # Retrieve data from Bloomberg
    data = bloomberg.get_historical_data(tickers, start_date, end_date)
    
    # Close Bloomberg connection
    bloomberg.close()
    
    if data is not None and not data.empty:
        # Save raw data
        data.to_csv('agricultural_futures_data.csv')
        print(f"✅ Data saved to agricultural_futures_data.csv")
        
        # Display sample data
        print("\nSample data:")
        print(data.head())
        
        # Handle missing values
        data = data.fillna(method='ffill').dropna()
        print(f"✅ Data shape after handling missing values: {data.shape}")
        
        # Calculate returns instead of using raw prices
        # returns_data = calculate_returns(data)
        # Using log returns for better numerical stability
        # log_returns = np.log(data / data.shift(1)).dropna()
        # print(f"✅ Calculated log returns with shape: {log_returns.shape}")
        
        # We'll stick with price data but apply better preprocessing
        processed_data = data
        
        # Initialize dataset with improved parameters
        window_size = 20  # Reduced from 30
        horizon = 5      # Reduced from 12
        dataset = ImprovedTimeSeriesDataset(window_size=window_size, horizon=horizon)
        
        # Create datasets
        X_train, y_train, X_val, y_val, X_test, y_test = dataset.create_dataset(processed_data)
        
        # Create data loaders
        batch_size = 16  # Reduced from 32
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
        
        # Initialize trainer with improved parameters
        trainer = LSTNetTrainer(
            num_variables=processed_data.shape[1],
            window_size=window_size,
            horizon=horizon,
            skip=0,  # Disable skip connection to avoid dimension issues
            rnn_hidden_dim=50,  # Reduced from 100
            cnn_hidden_dim=50,  # Reduced from 100
            dropout=0.3,        # Increased from 0.2
            learning_rate=0.0005  # Reduced from 0.001
        )
        
        # Train model
        print("\n===== Training LSTNet Model with Improved Preprocessing =====")
        model = trainer.train(train_loader, val_loader, epochs=100, patience=15)  # Increased patience
        
        # Evaluate model
        print("\n===== Evaluating LSTNet Model =====")
        for i, ticker in enumerate(processed_data.columns):
            evaluation = trainer.evaluate(test_loader, dataset, processed_data, target_idx=i)
            
            print(f"\nResults for {ticker}:")
            print(f"RSE: {evaluation['rse']:.4f}")
            print(f"RAE: {evaluation['rae']:.4f}")
            print(f"CORR: {evaluation['corr']:.4f}")
        
        # Plot training history
        plt.figure(figsize=(10, 6))
        plt.plot(trainer.history['train_loss'], label='Train Loss')
        plt.plot(trainer.history['val_loss'], label='Validation Loss')
        plt.title('LSTNet Training History (Improved)')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig('lstnet_training_history_improved.png')
        
        print(f"✅ Training history plot saved to lstnet_training_history_improved.png")
        
        return {
            'model': model,
            'dataset': dataset,
            'data': processed_data,
            'evaluation': evaluation
        }
    else:
        print("❌ Failed to retrieve data. Exiting.")
        return None

if __name__ == "__main__":
    print("Starting LSTNet model training with improved preprocessing...")
    try:
        results = train_lstnet_model()
        print("Training completed successfully.")
    except Exception as e:
        import traceback
        print(f"Error during training: {e}")
        traceback.print_exc()
