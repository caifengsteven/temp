import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pdblp
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.vector_ar.vecm import coint_johansen
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['savefig.dpi'] = 300

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

class StockDataHandler:
    """
    Class to handle the stock data acquisition and preparation
    """
    
    def __init__(self, use_bloomberg=True):
        """
        Initialize the data handler
        
        Parameters:
        -----------
        use_bloomberg : bool
            Whether to use Bloomberg data or Yahoo Finance data
        """
        self.use_bloomberg = use_bloomberg
        
        # Define the stock indices to use
        self.index_list = [
            {'name': 'S&P500', 'ticker': '^GSPC', 'bbg_ticker': 'SPX Index'},
            {'name': 'Russell2000', 'ticker': '^RUT', 'bbg_ticker': 'RTY Index'},
            {'name': 'Dow Jones', 'ticker': '^DJI', 'bbg_ticker': 'INDU Index'},
            {'name': 'NASDAQ', 'ticker': '^IXIC', 'bbg_ticker': 'CCMP Index'},
            {'name': 'USD/JPY', 'ticker': 'JPY=X', 'bbg_ticker': 'USDJPY Curncy'},
            {'name': 'Bitcoin USD', 'ticker': 'BTC-USD', 'bbg_ticker': 'XBT Curncy'},
            {'name': 'FTSE 100', 'ticker': '^FTSE', 'bbg_ticker': 'UKX Index'},
            {'name': 'Nikkei 225', 'ticker': '^N225', 'bbg_ticker': 'NKY Index'},
            {'name': 'Treasury Yield 10 Years', 'ticker': '^TNX', 'bbg_ticker': 'USGG10YR Index'},
            {'name': 'EUR/USD', 'ticker': 'EURUSD=X', 'bbg_ticker': 'EURUSD Curncy'},
            {'name': 'Gold', 'ticker': 'GC=F', 'bbg_ticker': 'XAU Curncy'},
            {'name': 'Silver', 'ticker': 'SI=F', 'bbg_ticker': 'XAG Curncy'},
            {'name': 'GBP/USD', 'ticker': 'GBPUSD=X', 'bbg_ticker': 'GBPUSD Curncy'},
            {'name': 'Fed Funds Rate', 'ticker': None, 'bbg_ticker': 'FEDL01 Index'},
            {'name': 'Crude Oil', 'ticker': 'CL=F', 'bbg_ticker': 'CL1 Comdty'}
        ]
        
        self.target_index = 'S&P500'
        
    def fetch_bloomberg_data(self, start_date, end_date):
        """
        Fetch data from Bloomberg
        
        Parameters:
        -----------
        start_date : str
            Start date in format 'YYYY-MM-DD'
        end_date : str
            End date in format 'YYYY-MM-DD'
            
        Returns:
        --------
        df : pandas.DataFrame
            DataFrame containing the stock data
        """
        print("Fetching data from Bloomberg...")
        
        try:
            # Initialize Bloomberg connection
            con = pdblp.BCon(debug=False, timeout=5000)
            con.start()
            
            tickers = [idx['bbg_ticker'] for idx in self.index_list]
            
            # Fetch the data
            data = con.bdh(tickers, 'PX_LAST', start_date, end_date)
            
            # Process the data
            df = pd.DataFrame()
            for idx in self.index_list:
                ticker = idx['bbg_ticker']
                if ticker in data.index.levels[0]:
                    series = data.loc[ticker]['PX_LAST']
                    df[idx['name']] = series
                else:
                    print(f"Warning: {ticker} not found in Bloomberg data")
            
            # Handle missing values
            df = df.interpolate(method='linear')
            
            con.stop()
            return df
            
        except Exception as e:
            print(f"Error fetching Bloomberg data: {e}")
            print("Falling back to synthetic data...")
            return self.generate_synthetic_data(start_date, end_date)
    
    def generate_synthetic_data(self, start_date, end_date):
        """
        Generate synthetic data for testing
        
        Parameters:
        -----------
        start_date : str
            Start date in format 'YYYY-MM-DD'
        end_date : str
            End date in format 'YYYY-MM-DD'
            
        Returns:
        --------
        df : pandas.DataFrame
            DataFrame containing the synthetic stock data
        """
        print("Generating synthetic data...")
        
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        
        # Create date range
        date_range = pd.date_range(start=start, end=end, freq='B')
        n_days = len(date_range)
        n_indices = len(self.index_list)
        
        # Set the seeds for correlated price movements
        np.random.seed(42)
        
        # Generate correlated returns
        # Create correlation matrix with realistic values
        corr_matrix = np.eye(n_indices)
        for i in range(n_indices):
            for j in range(i+1, n_indices):
                # Higher correlation for indices (like S&P and Dow)
                if i < 4 and j < 4:
                    corr_matrix[i, j] = corr_matrix[j, i] = np.random.uniform(0.7, 0.95)
                else:
                    corr_matrix[i, j] = corr_matrix[j, i] = np.random.uniform(-0.2, 0.7)
        
        # Ensure correlation matrix is positive definite
        min_eig = np.min(np.linalg.eigvals(corr_matrix))
        if min_eig < 0:
            corr_matrix += (-min_eig + 0.01) * np.eye(n_indices)
            
        # Define volatilities and returns for different asset classes
        vols = np.zeros(n_indices)
        means = np.zeros(n_indices)
        
        # Equity indices
        vols[0:4] = np.random.uniform(0.15, 0.25, 4) / np.sqrt(252)  # Annualized vol 15-25%
        means[0:4] = np.random.uniform(0.05, 0.15, 4) / 252  # Annualized return 5-15%
        
        # Forex
        vols[4:5] = np.random.uniform(0.08, 0.12, 1) / np.sqrt(252)
        means[4:5] = np.random.uniform(-0.02, 0.02, 1) / 252
        
        # Crypto
        vols[5:6] = np.random.uniform(0.5, 0.8, 1) / np.sqrt(252)
        means[5:6] = np.random.uniform(0.2, 0.5, 1) / 252
        
        # Other indices
        vols[6:8] = np.random.uniform(0.15, 0.25, 2) / np.sqrt(252)
        means[6:8] = np.random.uniform(0.05, 0.15, 2) / 252
        
        # Rates
        vols[8:9] = np.random.uniform(0.03, 0.1, 1) / np.sqrt(252)
        means[8:9] = np.random.uniform(-0.01, 0.02, 1) / 252
        
        # Other forex
        vols[9:10] = np.random.uniform(0.08, 0.12, 1) / np.sqrt(252)
        means[9:10] = np.random.uniform(-0.02, 0.02, 1) / 252
        
        # Commodities
        vols[10:12] = np.random.uniform(0.2, 0.3, 2) / np.sqrt(252)
        means[10:12] = np.random.uniform(0.03, 0.1, 2) / 252
        
        # More forex
        vols[12:13] = np.random.uniform(0.08, 0.12, 1) / np.sqrt(252)
        means[12:13] = np.random.uniform(-0.02, 0.02, 1) / 252
        
        # Rates
        vols[13:14] = np.random.uniform(0.01, 0.03, 1) / np.sqrt(252)
        means[13:14] = np.random.uniform(0, 0.01, 1) / 252
        
        # Oil
        vols[14:15] = np.random.uniform(0.25, 0.4, 1) / np.sqrt(252)
        means[14:15] = np.random.uniform(0.02, 0.1, 1) / 252
        
        # Create the covariance matrix
        cov_matrix = np.zeros((n_indices, n_indices))
        for i in range(n_indices):
            for j in range(n_indices):
                cov_matrix[i, j] = corr_matrix[i, j] * vols[i] * vols[j]
        
        # Generate returns using multivariate normal distribution
        returns = np.random.multivariate_normal(means, cov_matrix, n_days)
        
        # Convert to price levels
        prices = np.zeros((n_days, n_indices))
        
        # Set initial prices with realistic values
        initial_prices = np.array([
            3000,    # S&P500
            1800,    # Russell2000
            25000,   # Dow Jones
            10000,   # NASDAQ
            110,     # USD/JPY
            40000,   # Bitcoin
            7000,    # FTSE 100
            23000,   # Nikkei 225
            2.0,     # 10Y Treasury Yield
            1.2,     # EUR/USD
            1800,    # Gold
            25,      # Silver
            1.4,     # GBP/USD
            0.25,    # Fed Funds Rate
            60       # Crude Oil
        ])
        
        prices[0] = initial_prices
        
        # Calculate price levels
        for i in range(1, n_days):
            prices[i] = prices[i-1] * (1 + returns[i-1])
        
        # Create DataFrame
        df = pd.DataFrame(prices, index=date_range, columns=[idx['name'] for idx in self.index_list])
        
        # Add some cointegration relationship for testing purposes (especially between S&P and Dow)
        # Make S&P500 and Dow have a stronger relationship
        df['Dow Jones'] = df['S&P500'] * 8.5 + np.random.normal(0, 50, n_days)
        
        return df
    
    def fetch_data(self, start_date, end_date):
        """
        Fetch data from Bloomberg or generate synthetic data
        
        Parameters:
        -----------
        start_date : str
            Start date in format 'YYYY-MM-DD'
        end_date : str
            End date in format 'YYYY-MM-DD'
            
        Returns:
        --------
        df : pandas.DataFrame
            DataFrame containing the stock data
        """
        if self.use_bloomberg:
            try:
                df = self.fetch_bloomberg_data(start_date, end_date)
            except Exception as e:
                print(f"Error: {e}")
                print("Falling back to synthetic data...")
                df = self.generate_synthetic_data(start_date, end_date)
        else:
            df = self.generate_synthetic_data(start_date, end_date)
            
        return df
    
    def prepare_data(self, df, target_col, test_size=0.2, sequence_length=10):
        """
        Prepare the data for modeling
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame containing the stock data
        target_col : str
            Name of the target column
        test_size : float
            Proportion of data to use for testing
        sequence_length : int
            Length of the sequence for RNN input
            
        Returns:
        --------
        X_train, y_train, X_test, y_test, scaler_X, scaler_y
        """
        # Make a copy of the data
        data = df.copy()
        
        # Interpolate missing values if any
        data = data.interpolate(method='linear')
        
        # Split into train and test sets
        train_size = int(len(data) * (1 - test_size))
        train_data = data.iloc[:train_size]
        test_data = data.iloc[train_size:]
        
        # Scale the data
        scaler_X = MinMaxScaler(feature_range=(0, 1))
        scaler_y = MinMaxScaler(feature_range=(0, 1))
        
        # Scale features
        train_scaled = scaler_X.fit_transform(train_data)
        test_scaled = scaler_X.transform(test_data)
        
        # Scale target separately
        scaler_y.fit(train_data[[target_col]])
        
        # Prepare sequences
        X_train, y_train = self._create_sequences(train_scaled, train_data[target_col].values, 
                                                sequence_length, scaler_y)
        X_test, y_test = self._create_sequences(test_scaled, test_data[target_col].values, 
                                              sequence_length, scaler_y)
        
        return X_train, y_train, X_test, y_test, scaler_X, scaler_y, test_data
    
    def _create_sequences(self, data, target, sequence_length, scaler_y):
        """
        Create sequences for RNN input
        
        Parameters:
        -----------
        data : numpy.ndarray
            Scaled features data
        target : numpy.ndarray
            Target data
        sequence_length : int
            Length of the sequence
        scaler_y : sklearn.preprocessing.MinMaxScaler
            Scaler for the target data
            
        Returns:
        --------
        X, y : numpy.ndarray
            Sequences and targets
        """
        X, y = [], []
        
        for i in range(len(data) - sequence_length):
            X.append(data[i:i+sequence_length])
            # Scale the target value
            scaled_target = scaler_y.transform(target[i+sequence_length].reshape(-1, 1))[0, 0]
            y.append(scaled_target)
            
        return np.array(X), np.array(y)
    
    def perform_adf_test(self, df):
        """
        Perform Augmented Dickey-Fuller test to check for stationarity
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame containing the stock data
            
        Returns:
        --------
        adf_results : pandas.DataFrame
            DataFrame containing the ADF test results
        """
        print("Performing ADF test...")
        adf_results = pd.DataFrame(columns=['Index', 'p-value', 'Is Stationary (5%)'])
        
        for col in df.columns:
            result = adfuller(df[col].dropna())
            adf_results = adf_results.append({
                'Index': col,
                'p-value': result[1],
                'Is Stationary (5%)': result[1] < 0.05
            }, ignore_index=True)
            
        return adf_results
    
    def perform_cointegration_test(self, df, target_col):
        """
        Perform Johansen cointegration test
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame containing the stock data
        target_col : str
            Name of the target column
            
        Returns:
        --------
        coint_results : pandas.DataFrame
            DataFrame containing the cointegration test results
        """
        print("Performing cointegration test...")
        coint_results = pd.DataFrame(columns=['Index', 'p-value'])
        
        for col in df.columns:
            if col == target_col:
                continue
                
            # Prepare data for the test
            test_df = df[[target_col, col]].dropna()
            
            try:
                # Perform Johansen cointegration test
                result = coint_johansen(test_df, 0, 1)
                
                # Extract the p-value (trace statistic)
                # We're looking at the null hypothesis of no cointegration
                p_value = result.cvt[0, 1]  # Using the 5% critical value as an approximation
                trace_stat = result.lr1[0]
                
                # Calculate approximate p-value
                # If trace statistic > critical value, we reject the null hypothesis of no cointegration
                is_cointegrated = trace_stat > p_value
                
                # For simplicity, using a rough p-value based on is_cointegrated
                approx_p_value = 0.01 if is_cointegrated else 0.5
                
                coint_results = coint_results.append({
                    'Index': col,
                    'p-value': approx_p_value
                }, ignore_index=True)
            except Exception as e:
                print(f"Error in cointegration test for {col}: {e}")
                coint_results = coint_results.append({
                    'Index': col,
                    'p-value': 1.0
                }, ignore_index=True)
                
        return coint_results.sort_values('p-value')
    
    def calculate_correlations(self, df, target_col):
        """
        Calculate correlations between the target and other indices
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame containing the stock data
        target_col : str
            Name of the target column
            
        Returns:
        --------
        corr_results : pandas.DataFrame
            DataFrame containing the correlation results
        """
        print("Calculating correlations...")
        correlations = df.corr()[target_col].drop(target_col)
        corr_results = pd.DataFrame({
            'Index': correlations.index,
            'Correlation': correlations.values
        }).sort_values('Correlation', ascending=False)
        
        return corr_results
    
    def select_factors(self, df, target_col, method='cointegration', n_factors=5):
        """
        Select factors for modeling based on specified method
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame containing the stock data
        target_col : str
            Name of the target column
        method : str
            Method for factor selection ('all', 'cointegration', 'correlation')
        n_factors : int
            Number of factors to select
            
        Returns:
        --------
        selected_df : pandas.DataFrame
            DataFrame containing the selected factors
        """
        if method == 'all':
            print("Using all factors")
            return df
        
        elif method == 'cointegration':
            print(f"Selecting top {n_factors} factors based on cointegration test...")
            coint_results = self.perform_cointegration_test(df, target_col)
            
            # Select indices with lowest p-values (strongest cointegration relationship)
            selected_indices = list(coint_results.head(n_factors)['Index'])
            selected_indices.append(target_col)
            
            print("Selected indices based on cointegration:", selected_indices)
            return df[selected_indices]
        
        elif method == 'correlation':
            print(f"Selecting top {n_factors} factors based on correlation...")
            corr_results = self.calculate_correlations(df, target_col)
            
            # Select indices with highest correlation
            selected_indices = list(corr_results.head(n_factors)['Index'])
            selected_indices.append(target_col)
            
            print("Selected indices based on correlation:", selected_indices)
            return df[selected_indices]
        
        else:
            raise ValueError(f"Unknown method: {method}")


class LSTMModel(nn.Module):
    """
    LSTM model for stock prediction
    """
    
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout=0.2):
        """
        Initialize the model
        
        Parameters:
        -----------
        input_dim : int
            Number of input features
        hidden_dim : int
            Number of hidden units
        num_layers : int
            Number of LSTM layers
        output_dim : int
            Number of output features
        dropout : float
            Dropout probability
        """
        super(LSTMModel, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers, 
            batch_first=True, dropout=dropout if num_layers > 1 else 0
        )
        
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        """
        Forward pass
        
        Parameters:
        -----------
        x : torch.Tensor
            Input tensor of shape (batch_size, seq_len, input_dim)
            
        Returns:
        --------
        out : torch.Tensor
            Output tensor of shape (batch_size, output_dim)
        """
        # Initialize hidden state and cell state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        
        # LSTM forward pass
        out, _ = self.lstm(x, (h0, c0))
        
        # Take the output from the last time step
        out = self.fc(out[:, -1, :])
        
        return out


class GRUModel(nn.Module):
    """
    GRU model for stock prediction
    """
    
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout=0.2):
        """
        Initialize the model
        
        Parameters:
        -----------
        input_dim : int
            Number of input features
        hidden_dim : int
            Number of hidden units
        num_layers : int
            Number of GRU layers
        output_dim : int
            Number of output features
        dropout : float
            Dropout probability
        """
        super(GRUModel, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.gru = nn.GRU(
            input_dim, hidden_dim, num_layers, 
            batch_first=True, dropout=dropout if num_layers > 1 else 0
        )
        
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        """
        Forward pass
        
        Parameters:
        -----------
        x : torch.Tensor
            Input tensor of shape (batch_size, seq_len, input_dim)
            
        Returns:
        --------
        out : torch.Tensor
            Output tensor of shape (batch_size, output_dim)
        """
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        
        # GRU forward pass
        out, _ = self.gru(x, h0)
        
        # Take the output from the last time step
        out = self.fc(out[:, -1, :])
        
        return out


class QuantileLoss:
    """
    Quantile loss for predicting intervals
    """
    
    def __init__(self, quantiles=[0.1, 0.5, 0.9]):
        """
        Initialize the loss function
        
        Parameters:
        -----------
        quantiles : list
            List of quantiles to predict
        """
        self.quantiles = quantiles
        
    def __call__(self, preds, target):
        """
        Calculate the loss
        
        Parameters:
        -----------
        preds : torch.Tensor
            Predicted values
        target : torch.Tensor
            Actual values
            
        Returns:
        --------
        loss : torch.Tensor
            Quantile loss
        """
        assert preds.size(0) == target.size(0)
        
        # Initialize losses for each quantile
        losses = []
        
        for q in self.quantiles:
            errors = target - preds
            loss = torch.max((q - 1) * errors, q * errors)
            losses.append(loss.mean())
        
        # Average the losses across all quantiles
        return torch.stack(losses).mean()


class StockPredictor:
    """
    Main class for stock prediction using cointegration test and quantile loss
    """
    
    def __init__(self, model_type='LSTM', factor_selection='cointegration', loss_type='quantile',
                 n_factors=5, use_bloomberg=True):
        """
        Initialize the predictor
        
        Parameters:
        -----------
        model_type : str
            Type of model to use ('LSTM' or 'GRU')
        factor_selection : str
            Method for factor selection ('all', 'cointegration', 'correlation')
        loss_type : str
            Type of loss function to use ('quantile' or 'rmse')
        n_factors : int
            Number of factors to select
        use_bloomberg : bool
            Whether to use Bloomberg data or synthetic data
        """
        self.model_type = model_type
        self.factor_selection = factor_selection
        self.loss_type = loss_type
        self.n_factors = n_factors
        
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Initialize data handler
        self.data_handler = StockDataHandler(use_bloomberg=use_bloomberg)
        
        # Model parameters
        self.hidden_dim = 50
        self.num_layers = 2
        self.batch_size = 64
        self.learning_rate = 0.001
        self.num_epochs = 100
        self.sequence_length = 10
        
        # Initialize model, loss function, and optimizer
        self.model = None
        self.loss_fn = None
        self.optimizer = None
        
    def prepare_data(self, start_date, end_date, target_col='S&P500'):
        """
        Prepare the data for training and testing
        
        Parameters:
        -----------
        start_date : str
            Start date in format 'YYYY-MM-DD'
        end_date : str
            End date in format 'YYYY-MM-DD'
        target_col : str
            Name of the target column
            
        Returns:
        --------
        train_loader, test_loader, scaler_X, scaler_y, test_data
        """
        # Fetch data
        df = self.data_handler.fetch_data(start_date, end_date)
        
        # Select factors
        df = self.data_handler.select_factors(df, target_col, self.factor_selection, self.n_factors)
        
        # Prepare data for modeling
        X_train, y_train, X_test, y_test, scaler_X, scaler_y, test_data = self.data_handler.prepare_data(
            df, target_col, test_size=0.2, sequence_length=self.sequence_length
        )
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.FloatTensor(y_train)
        X_test_tensor = torch.FloatTensor(X_test)
        y_test_tensor = torch.FloatTensor(y_test)
        
        # Create datasets
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        
        # Save the dimensions for model initialization
        self.input_dim = X_train.shape[2]
        self.output_dim = 1
        
        return train_loader, test_loader, scaler_X, scaler_y, test_data, X_test, y_test
    
    def initialize_model(self):
        """
        Initialize the model, loss function, and optimizer
        """
        # Initialize model
        if self.model_type == 'LSTM':
            self.model = LSTMModel(
                self.input_dim, self.hidden_dim, self.num_layers, self.output_dim
            ).to(self.device)
        elif self.model_type == 'GRU':
            self.model = GRUModel(
                self.input_dim, self.hidden_dim, self.num_layers, self.output_dim
            ).to(self.device)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        # Initialize loss function
        if self.loss_type == 'quantile':
            self.loss_fn = QuantileLoss()
        elif self.loss_type == 'rmse':
            self.loss_fn = nn.MSELoss()
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
    def train(self, train_loader, val_loader=None):
        """
        Train the model
        
        Parameters:
        -----------
        train_loader : torch.utils.data.DataLoader
            DataLoader for training data
        val_loader : torch.utils.data.DataLoader
            DataLoader for validation data
            
        Returns:
        --------
        history : dict
            Training history
        """
        print(f"Training {self.model_type} model with {self.loss_type} loss...")
        
        history = {'train_loss': [], 'val_loss': []}
        
        for epoch in range(self.num_epochs):
            # Training
            self.model.train()
            train_loss = 0
            for X_batch, y_batch in train_loader:
                # Move tensors to the same device as model
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                
                # Forward pass
                y_pred = self.model(X_batch)
                
                # Reshape predictions to match target
                y_pred = y_pred.squeeze()
                
                # Calculate loss
                loss = self.loss_fn(y_pred, y_batch)
                
                # Backward pass and optimize
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item()
            
            # Average training loss
            train_loss /= len(train_loader)
            history['train_loss'].append(train_loss)
            
            # Validation
            if val_loader is not None:
                self.model.eval()
                val_loss = 0
                with torch.no_grad():
                    for X_batch, y_batch in val_loader:
                        X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                        y_pred = self.model(X_batch).squeeze()
                        loss = self.loss_fn(y_pred, y_batch)
                        val_loss += loss.item()
                
                val_loss /= len(val_loader)
                history['val_loss'].append(val_loss)
                
                print(f"Epoch {epoch+1}/{self.num_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            else:
                print(f"Epoch {epoch+1}/{self.num_epochs} | Train Loss: {train_loss:.4f}")
                
        return history
    
    def predict(self, X_test):
        """
        Make predictions with the trained model
        
        Parameters:
        -----------
        X_test : numpy.ndarray
            Test data
            
        Returns:
        --------
        predictions : numpy.ndarray
            Model predictions
        """
        self.model.eval()
        
        # Convert to tensor
        X_test_tensor = torch.FloatTensor(X_test).to(self.device)
        
        # Make predictions
        with torch.no_grad():
            predictions = self.model(X_test_tensor).cpu().numpy().flatten()
            
        return predictions
    
    def backtest(self, test_data, predictions, scaler_y, target_col='S&P500'):
        """
        Perform backtesting on the test data
        
        Parameters:
        -----------
        test_data : pandas.DataFrame
            Test data
        predictions : numpy.ndarray
            Model predictions
        scaler_y : sklearn.preprocessing.MinMaxScaler
            Scaler for the target data
        target_col : str
            Name of the target column
            
        Returns:
        --------
        results : dict
            Backtesting results
        """
        print("Performing backtesting...")
        
        # Inverse transform predictions
        predictions = scaler_y.inverse_transform(predictions.reshape(-1, 1)).flatten()
        
        # Get actual values
        actual = test_data[target_col].values[self.sequence_length:]
        
        # Ensure predictions and actual have the same length
        min_len = min(len(predictions), len(actual))
        predictions = predictions[:min_len]
        actual = actual[:min_len]
        
        # Create a DataFrame for backtesting
        backtest_df = pd.DataFrame({
            'Actual': actual,
            'Predicted': predictions
        }, index=test_data.index[self.sequence_length:self.sequence_length+min_len])
        
        # Calculate daily returns
        backtest_df['Actual_Return'] = backtest_df['Actual'].pct_change()
        
        # Calculate VD (Value Difference) as mentioned in the paper
        backtest_df['VD'] = (backtest_df['Actual'] / backtest_df['Predicted']) - 1
        
        # Determine position based on VD
        backtest_df['Position'] = backtest_df['VD'].apply(
            lambda x: 1 if x > 0.03 else (-1 if x < -0.03 else 0)
        )
        
        # Calculate strategy returns
        backtest_df['Strategy_Return'] = backtest_df['Position'].shift(1) * backtest_df['Actual_Return']
        
        # Fill NaN values with 0
        backtest_df['Strategy_Return'] = backtest_df['Strategy_Return'].fillna(0)
        
        # Calculate cumulative returns
        backtest_df['Cumulative_Market_Return'] = (1 + backtest_df['Actual_Return']).cumprod() - 1
        backtest_df['Cumulative_Strategy_Return'] = (1 + backtest_df['Strategy_Return']).cumprod() - 1
        
        # Calculate Sharpe ratio (assuming 252 trading days per year)
        risk_free_rate = 0  # Simplified assumption
        strategy_sharpe = np.sqrt(252) * (backtest_df['Strategy_Return'].mean() / backtest_df['Strategy_Return'].std())
        market_sharpe = np.sqrt(252) * (backtest_df['Actual_Return'].mean() / backtest_df['Actual_Return'].std())
        
        # Calculate maximum drawdown for strategy
        cumulative_returns = (1 + backtest_df['Strategy_Return']).cumprod()
        peak = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns / peak) - 1
        max_drawdown = drawdown.min()
        
        # Calculate other metrics
        total_trades = (backtest_df['Position'] != backtest_df['Position'].shift(1)).sum()
        profitable_trades = (backtest_df['Strategy_Return'] > 0).sum()
        win_rate = profitable_trades / total_trades if total_trades > 0 else 0
        
        # Compile results
        results = {
            'backtest_df': backtest_df,
            'cumulative_return': backtest_df['Cumulative_Strategy_Return'].iloc[-1],
            'market_return': backtest_df['Cumulative_Market_Return'].iloc[-1],
            'sharpe_ratio': strategy_sharpe,
            'market_sharpe': market_sharpe,
            'max_drawdown': max_drawdown,
            'total_trades': total_trades,
            'win_rate': win_rate
        }
        
        # Print summary
        print(f"Cumulative Return: {results['cumulative_return']:.4f}")
        print(f"Market Return: {results['market_return']:.4f}")
        print(f"Sharpe Ratio: {results['sharpe_ratio']:.4f}")
        print(f"Market Sharpe Ratio: {results['market_sharpe']:.4f}")
        print(f"Maximum Drawdown: {results['max_drawdown']:.4f}")
        print(f"Total Trades: {results['total_trades']}")
        print(f"Win Rate: {results['win_rate']:.4f}")
        
        return results
    
    def plot_results(self, results):
        """
        Plot backtesting results
        
        Parameters:
        -----------
        results : dict
            Backtesting results
        """
        backtest_df = results['backtest_df']
        
        # Create figure
        fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(12, 15), sharex=True)
        
        # Plot actual vs predicted prices
        axes[0].plot(backtest_df.index, backtest_df['Actual'], label='Actual')
        axes[0].plot(backtest_df.index, backtest_df['Predicted'], label='Predicted')
        axes[0].set_title('Actual vs Predicted Prices')
        axes[0].set_ylabel('Price')
        axes[0].legend()
        axes[0].grid(True)
        
        # Plot trading positions
        axes[1].plot(backtest_df.index, backtest_df['Position'], label='Position')
        axes[1].set_title('Trading Positions')
        axes[1].set_ylabel('Position')
        axes[1].legend()
        axes[1].grid(True)
        
        # Plot cumulative returns
        axes[2].plot(backtest_df.index, backtest_df['Cumulative_Strategy_Return'], 
                    label=f'Strategy Return ({results["cumulative_return"]:.2f})')
        axes[2].plot(backtest_df.index, backtest_df['Cumulative_Market_Return'], 
                    label=f'Market Return ({results["market_return"]:.2f})')
        axes[2].set_title('Cumulative Returns')
        axes[2].set_ylabel('Return')
        axes[2].set_xlabel('Date')
        axes[2].legend()
        axes[2].grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def run(self, start_date, end_date, target_col='S&P500'):
        """
        Run the complete stock prediction pipeline
        
        Parameters:
        -----------
        start_date : str
            Start date in format 'YYYY-MM-DD'
        end_date : str
            End date in format 'YYYY-MM-DD'
        target_col : str
            Name of the target column
            
        Returns:
        --------
        results : dict
            Results of the prediction and backtesting
        """
        # Prepare data
        train_loader, test_loader, scaler_X, scaler_y, test_data, X_test, y_test = self.prepare_data(
            start_date, end_date, target_col
        )
        
        # Initialize model
        self.initialize_model()
        
        # Train model
        history = self.train(train_loader, test_loader)
        
        # Make predictions
        predictions = self.predict(X_test)
        
        # Backtest
        results = self.backtest(test_data, predictions, scaler_y, target_col)
        results['history'] = history
        
        # Plot results
        self.plot_results(results)
        
        return results

def run_experiment(start_date='2018-03-05', end_date='2021-03-05', use_bloomberg=True):
    """
    Run the complete experiment comparing different approaches as described in the paper
    
    Parameters:
    -----------
    start_date : str
        Start date in format 'YYYY-MM-DD'
    end_date : str
        End date in format 'YYYY-MM-DD'
    use_bloomberg : bool
        Whether to use Bloomberg data or synthetic data
        
    Returns:
    --------
    results : dict
        Results of all the experiments
    """
    print("Running experiment comparing different approaches...")
    
    # Define the combinations to test
    experiments = [
        {'model_type': 'LSTM', 'factor_selection': 'all', 'loss_type': 'quantile'},
        {'model_type': 'GRU', 'factor_selection': 'all', 'loss_type': 'quantile'},
        {'model_type': 'LSTM', 'factor_selection': 'correlation', 'loss_type': 'quantile'},
        {'model_type': 'GRU', 'factor_selection': 'correlation', 'loss_type': 'quantile'},
        {'model_type': 'LSTM', 'factor_selection': 'cointegration', 'loss_type': 'quantile'},
        {'model_type': 'GRU', 'factor_selection': 'cointegration', 'loss_type': 'quantile'},
        {'model_type': 'LSTM', 'factor_selection': 'all', 'loss_type': 'rmse'},
        {'model_type': 'GRU', 'factor_selection': 'all', 'loss_type': 'rmse'},
        {'model_type': 'LSTM', 'factor_selection': 'correlation', 'loss_type': 'rmse'},
        {'model_type': 'GRU', 'factor_selection': 'correlation', 'loss_type': 'rmse'},
        {'model_type': 'LSTM', 'factor_selection': 'cointegration', 'loss_type': 'rmse'},
        {'model_type': 'GRU', 'factor_selection': 'cointegration', 'loss_type': 'rmse'}
    ]
    
    results = {}
    
    for i, exp in enumerate(experiments):
        print(f"\n=== Experiment {i+1}/{len(experiments)} ===")
        print(f"Model: {exp['model_type']}, Factor Selection: {exp['factor_selection']}, Loss: {exp['loss_type']}")
        
        # Initialize predictor with the current configuration
        predictor = StockPredictor(
            model_type=exp['model_type'],
            factor_selection=exp['factor_selection'],
            loss_type=exp['loss_type'],
            use_bloomberg=use_bloomberg
        )
        
        # Run the prediction pipeline
        exp_results = predictor.run(start_date, end_date)
        
        # Save results
        key = f"{exp['factor_selection']}+{exp['loss_type']}+{exp['model_type']}"
        results[key] = exp_results
    
    # Compile and compare results
    summary = pd.DataFrame({
        'Configuration': [],
        'Cumulative Return': [],
        'Sharpe Ratio': []
    })
    
    for key, res in results.items():
        summary = summary.append({
            'Configuration': key,
            'Cumulative Return': res['cumulative_return'],
            'Sharpe Ratio': res['sharpe_ratio']
        }, ignore_index=True)
    
    # Sort by Cumulative Return
    summary = summary.sort_values('Cumulative Return', ascending=False)
    
    print("\n=== Summary of Results ===")
    print(summary)
    
    # Plot comparison of cumulative returns
    plt.figure(figsize=(12, 8))
    for key, res in results.items():
        plt.plot(res['backtest_df'].index, res['backtest_df']['Cumulative_Strategy_Return'], label=key)
    
    plt.title('Comparison of Cumulative Returns')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    return results, summary

# Main execution
if __name__ == '__main__':
    # Set parameters
    start_date = '2018-03-05'
    end_date = '2021-03-05'
    
    # Run the full experiment
    results, summary = run_experiment(start_date, end_date, use_bloomberg=True)
    
    # Or run a specific configuration
    predictor = StockPredictor(
        model_type='GRU',
        factor_selection='cointegration',
        loss_type='quantile',
        use_bloomberg=True
    )
    
    specific_results = predictor.run(start_date, end_date)