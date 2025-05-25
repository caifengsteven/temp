import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from hmmlearn import hmm
from tqdm import tqdm
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class DataSimulator:
    """Class to generate simulated market and stock data"""
    
    def __init__(self, n_stocks=50, n_industries=10, start_date='2018-01-01', end_date='2022-12-31'):
        """
        Initialize the data simulator
        
        Parameters:
        -----------
        n_stocks : int
            Number of stocks to simulate
        n_industries : int
            Number of industries to simulate
        start_date : str
            Start date for the simulation
        end_date : str
            End date for the simulation
        """
        self.n_stocks = n_stocks
        self.n_industries = n_industries
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.date_range = pd.date_range(start=self.start_date, end=self.end_date, freq='B')
        self.n_days = len(self.date_range)
        
        # Generate stock and industry names
        self.industry_names = [f'Industry_{i}' for i in range(n_industries)]
        self.stock_symbols = [f'STOCK_{i}' for i in range(n_stocks)]
        
        # Assign stocks to industries
        self.industry_assignments = np.random.randint(0, n_industries, n_stocks)
        self.industry_mapping = {
            self.stock_symbols[i]: self.industry_names[self.industry_assignments[i]] 
            for i in range(n_stocks)
        }
        
        # Company information
        self.company_info = pd.DataFrame({
            'Symbol': self.stock_symbols,
            'Company': [f'Company {i}' for i in range(n_stocks)],
            'Industry': [self.industry_names[ind] for ind in self.industry_assignments],
            'Sector': [f'Sector_{ind % 5}' for ind in self.industry_assignments]
        })
    
    def simulate_market_data(self, volatility=0.01, drift=0.0005):
        """
        Simulate market index data
        
        Parameters:
        -----------
        volatility : float
            Daily volatility of market returns
        drift : float
            Daily drift (expected return) of market
            
        Returns:
        --------
        DataFrame with simulated market data
        """
        # Simulate daily returns
        daily_returns = np.random.normal(drift, volatility, self.n_days)
        
        # Start at price 100
        start_price = 100
        prices = start_price * np.cumprod(1 + daily_returns)
        
        # Create DataFrame
        market_data = pd.DataFrame({
            'Open': prices * np.random.uniform(0.998, 0.999, len(prices)),
            'High': prices * np.random.uniform(1.001, 1.003, len(prices)),
            'Low': prices * np.random.uniform(0.997, 0.999, len(prices)),
            'Close': prices,
            'Adj Close': prices,
            'Volume': np.random.randint(10000000, 100000000, len(prices))
        }, index=self.date_range)
        
        self.market_returns = daily_returns
        self.market_data = market_data
        
        return market_data
    
    def simulate_stock_data(self, market_beta_range=(0.5, 1.5), stock_volatility_range=(0.01, 0.03),
                           industry_volatility=0.015):
        """
        Simulate stock price data with market, industry, and idiosyncratic components
        
        Parameters:
        -----------
        market_beta_range : tuple
            Range of market betas for stocks
        stock_volatility_range : tuple
            Range of idiosyncratic volatilities
        industry_volatility : float
            Volatility of industry factors
            
        Returns:
        --------
        Dictionary of DataFrames with stock price data
        """
        # Generate market betas for stocks
        market_betas = np.random.uniform(market_beta_range[0], market_beta_range[1], self.n_stocks)
        
        # Generate industry factors
        industry_factors = {}
        for industry in self.industry_names:
            # Industry returns are correlated with market but have own shocks
            industry_beta = np.random.uniform(0.8, 1.2)
            industry_shocks = np.random.normal(0, industry_volatility, self.n_days)
            industry_factors[industry] = industry_beta * self.market_returns + industry_shocks
        
        # Generate idiosyncratic volatilities for each stock
        stock_volatilities = np.random.uniform(stock_volatility_range[0], stock_volatility_range[1], self.n_stocks)
        
        # Simulate stock returns and prices
        stock_data = {}
        stock_returns = {}
        
        for i, symbol in enumerate(self.stock_symbols):
            # Get stock's industry
            industry = self.industry_mapping[symbol]
            
            # Components of return
            market_component = market_betas[i] * self.market_returns
            industry_component = industry_factors[industry]
            idiosyncratic_component = np.random.normal(0, stock_volatilities[i], self.n_days)
            
            # Combined return
            returns = market_component + industry_component + idiosyncratic_component
            stock_returns[symbol] = returns
            
            # Convert to price series
            start_price = np.random.uniform(20, 200)  # Random starting price
            prices = start_price * np.cumprod(1 + returns)
            
            # Create DataFrame
            data = pd.DataFrame({
                'Open': prices * np.random.uniform(0.995, 0.998, len(prices)),
                'High': prices * np.random.uniform(1.002, 1.008, len(prices)),
                'Low': prices * np.random.uniform(0.992, 0.998, len(prices)),
                'Close': prices,
                'Adj Close': prices,
                'Volume': np.random.randint(100000, 10000000, len(prices))
            }, index=self.date_range)
            
            stock_data[symbol] = data
        
        self.stock_returns = stock_returns
        self.stock_data = stock_data
        
        return stock_data
    
    def simulate_earnings_announcements(self, quarters_per_year=4, max_surprise_pct=0.2):
        """
        Simulate earnings announcement dates and surprises
        
        Parameters:
        -----------
        quarters_per_year : int
            Number of earnings announcements per year
        max_surprise_pct : float
            Maximum percentage earnings surprise
            
        Returns:
        --------
        DataFrame with earnings announcement data
        """
        announcements = []
        
        # For each stock
        for symbol in self.stock_symbols:
            # Get stock's industry
            industry = self.industry_mapping[symbol]
            
            # Calculate number of quarters in the date range
            n_years = (self.end_date - self.start_date).days / 365.25
            n_quarters = int(n_years * quarters_per_year)
            
            # Generate announcement dates for each quarter
            for q in range(n_quarters):
                # Calculate quarter start and end
                quarter_start = self.start_date + pd.DateOffset(months=3*q)
                quarter_end = quarter_start + pd.DateOffset(months=3)
                
                # Find business days in this quarter
                quarter_dates = pd.date_range(quarter_start, quarter_end, freq='B')
                
                # Select a random date in the quarter for the announcement
                if len(quarter_dates) > 0:
                    announcement_date = np.random.choice(quarter_dates)
                    
                    # Generate earnings surprise
                    eps_estimate = np.random.uniform(0.5, 2.0)
                    surprise_pct = np.random.uniform(-max_surprise_pct, max_surprise_pct)
                    eps_actual = eps_estimate * (1 + surprise_pct)
                    
                    announcements.append({
                        'symbol': symbol,
                        'date': announcement_date,
                        'quarter': q,
                        'eps_estimate': eps_estimate,
                        'eps_actual': eps_actual,
                        'surprise_percent': surprise_pct,
                        'industry': industry
                    })
        
        # Convert to DataFrame
        announcements_df = pd.DataFrame(announcements)
        
        # Ensure announcements are within the date range
        announcements_df = announcements_df[
            (announcements_df['date'] >= self.start_date) & 
            (announcements_df['date'] <= self.end_date)
        ]
        
        # Sort by date
        announcements_df = announcements_df.sort_values('date').reset_index(drop=True)
        
        self.earnings_announcements = announcements_df
        
        return announcements_df
    
    def embed_anomalies(self, pead_magnitude=0.02, momentum_factor=0.01, pairs_factor=0.02):
        """
        Embed market anomalies in the simulated stock data
        
        Parameters:
        -----------
        pead_magnitude : float
            Magnitude of post-earnings announcement drift
        momentum_factor : float
            Magnitude of momentum effect
        pairs_factor : float
            Magnitude of pairs trading opportunity
            
        Returns:
        --------
        Dictionary of DataFrames with updated stock price data
        """
        # 1. Embed post-earnings announcement drift
        for _, event in self.earnings_announcements.iterrows():
            symbol = event['symbol']
            date = event['date']
            surprise_pct = event['surprise_percent']
            
            if abs(surprise_pct) >= 0.05:  # Only significant surprises cause drift
                # Find the index of the announcement date
                if date in self.date_range:
                    date_idx = self.date_range.get_loc(date)
                    
                    # Immediate reaction
                    self.stock_returns[symbol][date_idx] += surprise_pct * pead_magnitude * 5
                    
                    # Drift over next 60 days
                    drift_length = min(60, self.n_days - date_idx - 1)
                    if drift_length > 0:
                        # Gradually decaying drift
                        for d in range(1, drift_length + 1):
                            if date_idx + d < len(self.stock_returns[symbol]):
                                drift_effect = surprise_pct * pead_magnitude * (1 - d/drift_length)
                                self.stock_returns[symbol][date_idx + d] += drift_effect
        
        # 2. Embed momentum effect
        # Calculate rolling 6-month returns for each stock
        formation_period = 126  # ~6 months
        holding_period = 21  # ~1 month
        
        # Identify winner and loser stocks in each period
        for t in range(formation_period, self.n_days, holding_period):
            # Calculate past returns
            past_returns = {}
            for symbol in self.stock_symbols:
                past_return = np.sum(self.stock_returns[symbol][t-formation_period:t])
                past_returns[symbol] = past_return
            
            # Sort stocks by past return
            sorted_stocks = sorted(past_returns.items(), key=lambda x: x[1])
            num_extreme = max(1, self.n_stocks // 10)
            loser_stocks = [s[0] for s in sorted_stocks[:num_extreme]]
            winner_stocks = [s[0] for s in sorted_stocks[-num_extreme:]]
            
            # Add momentum effect for the holding period
            for h in range(holding_period):
                if t + h < self.n_days:
                    # Winners outperform
                    for symbol in winner_stocks:
                        self.stock_returns[symbol][t + h] += momentum_factor
                    
                    # Losers underperform
                    for symbol in loser_stocks:
                        self.stock_returns[symbol][t + h] -= momentum_factor
        
        # 3. Embed pairs trading opportunities
        # Create some cointegrated pairs within industries
        formation_period = 252  # ~1 year
        
        # Group stocks by industry
        industry_stocks = {}
        for symbol, industry in self.industry_mapping.items():
            if industry not in industry_stocks:
                industry_stocks[industry] = []
            industry_stocks[industry].append(symbol)
        
        # For each industry with at least 2 stocks
        for industry, symbols in industry_stocks.items():
            if len(symbols) >= 2:
                # Create pairs
                for i in range(0, len(symbols) - 1, 2):
                    stock1 = symbols[i]
                    stock2 = symbols[i + 1]
                    
                    # Make the pair have similar returns (cointegrated)
                    common_factor = np.random.normal(0, 0.01, self.n_days)
                    
                    # Add common factor to both stocks
                    self.stock_returns[stock1] += common_factor
                    self.stock_returns[stock2] += common_factor
                    
                    # Add occasional divergence that later converges (pairs trading opportunity)
                    for t in range(formation_period, self.n_days, 60):
                        if t + 20 < self.n_days:
                            # Temporary divergence
                            divergence = np.zeros(self.n_days)
                            
                            # Stock1 outperforms temporarily
                            divergence[t:t+10] = np.linspace(0, pairs_factor, 10)
                            divergence[t+10:t+20] = np.linspace(pairs_factor, 0, 10)
                            
                            # Add divergence
                            self.stock_returns[stock1] += divergence
                            self.stock_returns[stock2] -= divergence
        
        # Recalculate price series with the embedded anomalies
        for symbol in self.stock_symbols:
            returns = self.stock_returns[symbol]
            
            # Start price from the original data
            start_price = self.stock_data[symbol]['Adj Close'][0]
            
            # Calculate new price series
            prices = start_price * np.cumprod(1 + returns)
            
            # Update DataFrame
            self.stock_data[symbol]['Adj Close'] = prices
            self.stock_data[symbol]['Close'] = prices
            self.stock_data[symbol]['Open'] = prices * np.random.uniform(0.995, 0.998, len(prices))
            self.stock_data[symbol]['High'] = prices * np.random.uniform(1.002, 1.008, len(prices))
            self.stock_data[symbol]['Low'] = prices * np.random.uniform(0.992, 0.998, len(prices))
        
        return self.stock_data

class StockDataset(Dataset):
    """Dataset for stock price, market, and volume data"""
    
    def __init__(self, price_data, market_data, window_size):
        """
        Initialize the stock dataset
        
        Parameters:
        -----------
        price_data : DataFrame
            Stock price data
        market_data : DataFrame
            Market index data
        window_size : int
            Window size for sequence prediction
        """
        self.window_size = window_size
        
        # Extract price, market and volume data
        self.price = price_data['Adj Close'].values.reshape(-1, 1)
        self.market = market_data['Adj Close'].values.reshape(-1, 1)
        self.volume = price_data['Volume'].values.reshape(-1, 1)
        
        # Scale price and market data
        self.price_scaler = MinMaxScaler()
        self.market_scaler = MinMaxScaler()
        
        self.scaled_price = self.price_scaler.fit_transform(self.price)
        self.scaled_market = self.market_scaler.fit_transform(self.market)
        
        # Log transform volume data to reduce skewness (don't scale)
        self.log_volume = np.log1p(self.volume)
        
        # Create sequences
        self.X, self.y = self._create_sequences()
    
    def _create_sequences(self):
        """Create sequences for LSTM training"""
        X, y = [], []
        
        # Combine data
        full_data = np.hstack((self.scaled_price, self.scaled_market, self.log_volume))
        
        for i in range(len(full_data) - self.window_size):
            X.append(full_data[i:i+self.window_size-1])
            y.append(full_data[i+self.window_size-1])
        
        return X, y
    
    def __len__(self):
        """Return the number of sequences"""
        return len(self.X)
    
    def __getitem__(self, idx):
        """Return a specific sequence and target"""
        return torch.FloatTensor(self.X[idx]), torch.FloatTensor(self.y[idx])

class LSTMModel(nn.Module):
    """LSTM model for stock price, market, and volume prediction"""
    
    def __init__(self, input_dim=3, hidden_dim=64, num_layers=3, output_dim=3, dropout=0.2):
        """
        Initialize the LSTM model
        
        Parameters:
        -----------
        input_dim : int
            Number of input features (price, market, volume)
        hidden_dim : int
            Number of hidden units
        num_layers : int
            Number of LSTM layers
        output_dim : int
            Number of output features (price, market, volume)
        dropout : float
            Dropout rate
        """
        super(LSTMModel, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm1 = nn.LSTM(
            input_dim, hidden_dim, num_layers=1, 
            batch_first=True, dropout=0
        )
        self.dropout1 = nn.Dropout(dropout)
        
        self.lstm2 = nn.LSTM(
            hidden_dim, hidden_dim, num_layers=1, 
            batch_first=True, dropout=0
        )
        
        self.lstm3 = nn.LSTM(
            hidden_dim, hidden_dim//2, num_layers=1, 
            batch_first=True, dropout=0
        )
        self.dropout2 = nn.Dropout(dropout)
        
        # Output layer
        self.fc = nn.Linear(hidden_dim//2, output_dim)
    
    def forward(self, x):
        """Forward pass"""
        # LSTM layers
        out, _ = self.lstm1(x)
        out = self.dropout1(out)
        
        out, _ = self.lstm2(out)
        
        out, (h_n, _) = self.lstm3(out)
        out = self.dropout2(out[:, -1, :])  # Take only the last output
        
        # Output layer
        out = self.fc(out)
        
        return out

class SimpleLSTM(nn.Module):
    """Simple LSTM model for single feature prediction"""
    
    def __init__(self, input_size=1, hidden_size=32, output_size=1):
        super(SimpleLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out[:, -1, :]  # Take the last time step
        output = self.linear(lstm_out)
        return output

class MidLSTM:
    """
    Mid-LSTM implementation for midterm stock prediction
    """
    def __init__(self, window_size=60):
        """
        Initialize the Mid-LSTM model
        
        Parameters:
        -----------
        window_size : int
            Window size for sequence prediction
        """
        self.window_size = window_size
        self.lstm_model = None
        self.hmm_model = None
        self.regression_model = None
        self.price_scaler = MinMaxScaler()
        self.market_scaler = MinMaxScaler()
        self.n_components = 4  # Store the number of HMM components
        
    def build_lstm_model(self):
        """Build the LSTM model using PyTorch"""
        model = LSTMModel(input_dim=3, hidden_dim=64, num_layers=3, output_dim=3, dropout=0.2)
        return model
    
    def build_hmm_model(self, n_states=4):
        """
        Build the HMM model for hidden state extraction
        
        Parameters:
        -----------
        n_states : int
            Number of hidden states in the HMM
        """
        self.n_components = n_states
        model = hmm.GaussianHMM(n_components=n_states, covariance_type="full", n_iter=10)
        return model
    
    def prepare_data(self, price_data, market_data):
        """
        Prepare data for Mid-LSTM training
        
        Parameters:
        -----------
        price_data : DataFrame
            Stock price data
        market_data : DataFrame
            Market index data
        
        Returns:
        --------
        DataLoader for training
        """
        # Create dataset
        dataset = StockDataset(price_data, market_data, self.window_size)
        
        # Save scalers
        self.price_scaler = dataset.price_scaler
        self.market_scaler = dataset.market_scaler
        
        # Create data loader
        data_loader = DataLoader(
            dataset, 
            batch_size=32, 
            shuffle=True, 
            drop_last=False
        )
        
        return data_loader, dataset
    
    def prepare_hmm_data(self, price_data):
        """
        Prepare data for HMM training
        
        Parameters:
        -----------
        price_data : DataFrame
            Stock price data
        
        Returns:
        --------
        Features for HMM model
        """
        # Extract price and volume data
        price = price_data['Adj Close'].values
        volume = price_data['Volume'].values
        
        # Scale price data
        scaled_price = self.price_scaler.transform(price.reshape(-1, 1)).flatten()
        
        # Log transform volume and standardize
        log_volume = np.log1p(volume)
        log_volume = (log_volume - np.mean(log_volume)) / np.std(log_volume)
        
        # Combine features
        features = np.column_stack([scaled_price, log_volume])
        
        return features
    
    def train(self, price_data, market_data, epochs=50, learning_rate=0.001, verbose=1):
        """
        Train the Mid-LSTM model
        
        Parameters:
        -----------
        price_data : DataFrame
            Stock price data
        market_data : DataFrame
            Market index data
        epochs : int
            Number of training epochs
        learning_rate : float
            Learning rate for optimizer
        verbose : int
            Verbosity level
        """
        # 1. Train LSTM model
        print("Training LSTM model...")
        data_loader, dataset = self.prepare_data(price_data, market_data)
        
        self.lstm_model = self.build_lstm_model().to(device)
        optimizer = optim.Adam(self.lstm_model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        
        # Training loop
        self.lstm_model.train()
        for epoch in range(epochs):
            total_loss = 0
            for X_batch, y_batch in data_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                
                # Forward pass
                outputs = self.lstm_model(X_batch)
                loss = criterion(outputs, y_batch)
                
                # Backward pass and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(data_loader)
            if verbose and (epoch + 1) % 5 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.6f}')
        
        # 2. Train HMM model
        print("Training HMM model...")
        hmm_features = self.prepare_hmm_data(price_data)
        
        self.hmm_model = self.build_hmm_model()
        self.hmm_model.fit(hmm_features)
        
        # 3. Train Linear Regression model
        print("Training Linear Regression model...")
        # Generate LSTM predictions on training data
        self.lstm_model.eval()
        
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for X_batch, y_batch in data_loader:
                X_batch = X_batch.to(device)
                outputs = self.lstm_model(X_batch)
                all_predictions.append(outputs.cpu().numpy())
                all_targets.append(y_batch.cpu().numpy())
        
        lstm_predictions = np.vstack(all_predictions)
        targets = np.vstack(all_targets)
        
        # Extract predicted price, market index and volume
        pred_price = lstm_predictions[:, 0].reshape(-1, 1)
        pred_market = lstm_predictions[:, 1].reshape(-1, 1)
        pred_volume = lstm_predictions[:, 2].reshape(-1, 1)
        
        # Extract hidden states using HMM
        hmm_input = np.column_stack([pred_price, pred_volume])
        hidden_states = self.hmm_model.predict(hmm_input)
        hidden_states_onehot = pd.get_dummies(hidden_states).values
        
        # If there are missing states in the one-hot encoding, ensure all states are represented
        if hidden_states_onehot.shape[1] < self.n_components:
            temp = np.zeros((hidden_states_onehot.shape[0], self.n_components))
            for i, col in enumerate(pd.get_dummies(hidden_states).columns):
                temp[:, col] = hidden_states_onehot[:, i]
            hidden_states_onehot = temp
        
        # Calculate correlation coefficient
        correlation = np.zeros(len(pred_price))
        window = 20  # Correlation window
        
        for i in range(window, len(pred_price)):
            price_window = pred_price[i-window:i]
            market_window = pred_market[i-window:i]
            correlation[i] = np.corrcoef(price_window.flatten(), market_window.flatten())[0, 1]
        
        # Fill NaN values
        correlation[np.isnan(correlation)] = 0
        
        # Create regression features
        X_reg = np.hstack((
            pred_price,  # X^A_t
            pred_market,  # M^A_t
            correlation.reshape(-1, 1),  # ρ
            hidden_states_onehot  # S^A_t (one-hot encoded)
        ))
        
        # Target is the actual price
        true_price = dataset.price[self.window_size-1:self.window_size-1+len(lstm_predictions)]
        
        # Train linear regression
        self.regression_model = LinearRegression()
        self.regression_model.fit(X_reg, true_price)
        
        print("Training complete!")
        
        # Print regression coefficients for interpretation
        print("\nMid-ARMA Model Coefficients:")
        coeffs = self.regression_model.coef_[0]
        print(f"Alpha (weight for LSTM price): {coeffs[0]:.4f}")
        print(f"Lambda (weight for market): {coeffs[1]:.4f}")
        print(f"Rho weight: {coeffs[2]:.4f}")
        print(f"Gamma (weight for hidden states): {', '.join([f'{coeff:.4f}' for coeff in coeffs[3:3+self.n_components]])}")
        print(f"Constant: {self.regression_model.intercept_[0]:.4f}")
    
    def predict_full_sequence(self, initial_sequence_price, initial_sequence_market, prediction_length):
        """
        Predict the full sequence using the trained model
        
        Parameters:
        -----------
        initial_sequence_price : array-like
            Initial price sequence for prediction
        initial_sequence_market : array-like
            Initial market sequence for prediction
        prediction_length : int
            Length of the prediction sequence
            
        Returns:
        --------
        Predicted prices
        """
        price = initial_sequence_price[-self.window_size+1:].reshape(-1, 1)
        market = initial_sequence_market[-self.window_size+1:].reshape(-1, 1)
        
        # Scale price and market data
        scaled_price = self.price_scaler.transform(price)
        scaled_market = self.market_scaler.transform(market)
        
        # Get volume data for initial sequence (use simulated values)
        volume = np.random.lognormal(10, 1, len(price)).reshape(-1, 1)
        log_volume = np.log1p(volume)
        
        # Prepare initial sequence
        current_sequence = np.hstack((scaled_price, scaled_market, log_volume))
        
        # Storage for predictions
        predicted_prices = []
        
        # Set model to evaluation mode
        self.lstm_model.eval()
        
        # Make predictions step by step
        for _ in range(prediction_length):
            # Get current window for LSTM
            window_data = current_sequence[-self.window_size+1:]
            
            # Convert to tensor
            X = torch.FloatTensor(window_data).unsqueeze(0).to(device)
            
            # Get LSTM prediction
            with torch.no_grad():
                lstm_pred = self.lstm_model(X).cpu().numpy()[0]
            
            # Extract predicted price, market, and volume
            pred_price = lstm_pred[0]
            pred_market = lstm_pred[1]
            pred_volume = lstm_pred[2]
            
            # Create HMM feature for predicting hidden state
            hmm_feature = np.array([[pred_price, pred_volume]])
            
            # Get hidden state
            try:
                hidden_state = self.hmm_model.predict(hmm_feature)[0]
            except:
                # If there's an error, default to state 0
                hidden_state = 0
            
            # Create one-hot encoding for hidden state
            hidden_state_onehot = np.zeros(self.n_components)
            hidden_state_onehot[hidden_state] = 1
            
            # Calculate correlation (using a rolling window of the last 20 predictions)
            if current_sequence.shape[0] >= 20:
                price_window = current_sequence[-20:, 0]
                market_window = current_sequence[-20:, 1]
                correlation = np.corrcoef(price_window, market_window)[0, 1]
                if np.isnan(correlation):
                    correlation = 0
            else:
                correlation = 0
            
            # Create regression feature
            reg_feature = np.concatenate([
                [[pred_price]],
                [[pred_market]],
                [[correlation]],
                [hidden_state_onehot]
            ], axis=1)
            
            # Get final prediction from regression model
            final_pred = self.regression_model.predict(reg_feature)[0]
            predicted_prices.append(final_pred)
            
            # Update current sequence for next prediction
            # Convert final prediction back to scaled value for next LSTM input
            scaled_final = self.price_scaler.transform(final_pred.reshape(-1, 1))[0][0]
            
            # Create new row for the sequence
            new_row = np.array([scaled_final, pred_market, pred_volume])
            
            # Add to current sequence
            current_sequence = np.vstack((current_sequence, new_row))
        
        return np.array(predicted_prices)

def rolling_window_prediction(model, stock_data, market_data, window_size=60, prediction_length=60):
    """
    Perform rolling window prediction
    
    Parameters:
    -----------
    model : MidLSTM
        Trained Mid-LSTM model
    stock_data : DataFrame
        Stock price data
    market_data : DataFrame
        Market index data
    window_size : int
        Size of the rolling window
    prediction_length : int
        Length of each prediction
        
    Returns:
    --------
    Dictionary with true and predicted prices
    """
    # Get full price and market data
    full_price = stock_data['Adj Close'].values
    full_market = market_data['Adj Close'].values
    
    # Calculate number of windows
    n_windows = (len(full_price) - window_size) // prediction_length
    
    all_true_prices = []
    all_predicted_prices = []
    
    for i in range(n_windows):
        # Get window start and end
        start_idx = i * prediction_length
        end_idx = start_idx + window_size
        pred_end_idx = end_idx + prediction_length
        
        if pred_end_idx > len(full_price):
            break
        
        # Get initial sequence
        initial_price = full_price[start_idx:end_idx]
        initial_market = full_market[start_idx:end_idx]
        
        # Get true future prices
        true_future = full_price[end_idx:pred_end_idx]
        
        # Predict future prices
        predicted_future = model.predict_full_sequence(
            initial_price,
            initial_market,
            prediction_length
        )
        
        all_true_prices.append(true_future)
        all_predicted_prices.append(predicted_future)
    
    return {
        'true_prices': all_true_prices,
        'predicted_prices': all_predicted_prices
    }

def plot_predictions(true_prices, predicted_prices, window_idx=0, title='Mid-LSTM Stock Price Prediction'):
    """
    Plot the prediction results
    
    Parameters:
    -----------
    true_prices : list of arrays
        True stock prices for each window
    predicted_prices : list of arrays
        Predicted stock prices for each window
    window_idx : int
        Index of the window to plot
    title : str
        Plot title
    """
    plt.figure(figsize=(12, 6))
    
    # Plot specific window
    if window_idx < len(true_prices):
        plt.plot(true_prices[window_idx], label='True Price', color='blue')
        plt.plot(predicted_prices[window_idx], label='Predicted Price', color='red')
        plt.axvline(x=30, color='green', linestyle='--', label='Midterm Start')
        
        # Label midterm section
        plt.axvspan(30, 60, alpha=0.2, color='green')
        
        plt.title(title)
        plt.xlabel('Days')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)
    else:
        print(f"Window index {window_idx} is out of range.")
    
    plt.tight_layout()
    plt.show()

def calculate_metrics(true_prices, predicted_prices):
    """
    Calculate evaluation metrics for all windows
    
    Parameters:
    -----------
    true_prices : list of arrays
        True stock prices for each window
    predicted_prices : list of arrays
        Predicted stock prices for each window
        
    Returns:
    --------
    DataFrame with evaluation metrics
    """
    n_windows = len(true_prices)
    metrics = []
    
    for i in range(n_windows):
        # Get prices for current window
        true = true_prices[i]
        pred = predicted_prices[i]
        
        # Calculate metrics for full window
        full_mpa = 1 - np.mean(np.abs(true - pred) / true)
        full_rmse = np.sqrt(np.mean((true - pred)**2))
        
        # Calculate metrics for midterm section (days 30-60)
        if len(true) >= 60:
            midterm_true = true[30:60]
            midterm_pred = pred[30:60]
            midterm_mpa = 1 - np.mean(np.abs(midterm_true - midterm_pred) / midterm_true)
            midterm_rmse = np.sqrt(np.mean((midterm_true - midterm_pred)**2))
            
            # Calculate trend accuracy
            true_trend = midterm_true[-1] > midterm_true[0]
            pred_trend = midterm_pred[-1] > midterm_pred[0]
            midterm_ta = 1 if true_trend == pred_trend else 0
        else:
            midterm_mpa = None
            midterm_rmse = None
            midterm_ta = None
        
        metrics.append({
            'Window': i+1,
            'Full_MPA': full_mpa,
            'Full_RMSE': full_rmse,
            'Midterm_MPA': midterm_mpa,
            'Midterm_RMSE': midterm_rmse,
            'Midterm_TA': midterm_ta
        })
    
    metrics_df = pd.DataFrame(metrics)
    return metrics_df

def run_portfolio_allocation(true_prices, predicted_prices, initial_capital=1000000, risk_free_rate=0.015):
    """
    Run portfolio allocation based on predictions
    
    Parameters:
    -----------
    true_prices : list of arrays
        True stock prices for each window
    predicted_prices : list of arrays
        Predicted stock prices for each window
    initial_capital : float
        Initial investment capital
    risk_free_rate : float
        Annual risk-free rate
        
    Returns:
    --------
    Dictionary with portfolio performance metrics
    """
    n_windows = len(true_prices)
    
    # Calculate returns for each window
    portfolio_values = [initial_capital]
    portfolio_returns = []
    
    for i in range(n_windows):
        # Get midterm true prices (days 30-60)
        if len(true_prices[i]) >= 60:
            midterm_true = true_prices[i][30:60]
            midterm_pred = predicted_prices[i][30:60]
            
            # Calculate predicted return
            pred_return = (midterm_pred[-1] - midterm_pred[0]) / midterm_pred[0]
            
            # Only invest if predicted return is positive
            if pred_return > 0:
                # Calculate actual return
                actual_return = (midterm_true[-1] - midterm_true[0]) / midterm_true[0]
                
                # Update portfolio value
                current_value = portfolio_values[-1]
                new_value = current_value * (1 + actual_return)
                portfolio_values.append(new_value)
                portfolio_returns.append(actual_return)
            else:
                # No investment, value remains the same
                portfolio_values.append(portfolio_values[-1])
                portfolio_returns.append(0)
    
    # Calculate portfolio metrics
    if portfolio_returns:
        mean_return = np.mean(portfolio_returns)
        std_return = np.std(portfolio_returns)
        
        # Annualize return and volatility (assuming 252 trading days per year)
        annualized_return = (1 + mean_return) ** 252 - 1
        annualized_volatility = std_return * np.sqrt(252)
        
        # Calculate Sharpe ratio
        sharpe_ratio = (annualized_return - risk_free_rate) / annualized_volatility if annualized_volatility > 0 else 0
        
        # Calculate total return
        total_return = (portfolio_values[-1] / initial_capital) - 1
        
        return {
            'portfolio_values': portfolio_values,
            'portfolio_returns': portfolio_returns,
            'total_return': total_return,
            'annualized_return': annualized_return,
            'annualized_volatility': annualized_volatility,
            'sharpe_ratio': sharpe_ratio
        }
    else:
        return {
            'portfolio_values': [initial_capital],
            'portfolio_returns': [],
            'total_return': 0,
            'annualized_return': 0,
            'annualized_volatility': 0,
            'sharpe_ratio': 0
        }

def compare_with_traditional_methods(stock_data, market_data, window_size=60, prediction_length=60):
    """
    Compare Mid-LSTM with traditional methods
    
    Parameters:
    -----------
    stock_data : DataFrame
        Stock price data
    market_data : DataFrame
        Market index data
    window_size : int
        Size of the rolling window
    prediction_length : int
        Length of each prediction
        
    Returns:
    --------
    DataFrame with comparison results
    """
    from sklearn.linear_model import LinearRegression, Ridge
    from sklearn.ensemble import RandomForestRegressor
    
    # Get full price data
    full_price = stock_data['Adj Close'].values
    
    # Calculate number of windows
    n_windows = (len(full_price) - window_size) // prediction_length
    
    all_true_prices = []
    all_linear_predictions = []
    all_ridge_predictions = []
    all_rf_predictions = []
    all_lstm_predictions = []
    
    for i in range(n_windows):
        # Get window start and end
        start_idx = i * prediction_length
        end_idx = start_idx + window_size
        pred_end_idx = end_idx + prediction_length
        
        if pred_end_idx > len(full_price):
            break
        
        # Get initial sequence
        initial_price = full_price[start_idx:end_idx]
        
        # Get true future prices
        true_future = full_price[end_idx:pred_end_idx]
        all_true_prices.append(true_future)
        
        # Prepare features for traditional methods
        X = np.array([range(window_size)]).T
        y = initial_price
        
        # Linear Regression
        linear_model = LinearRegression()
        linear_model.fit(X, y)
        X_future = np.array([range(window_size, window_size + prediction_length)]).T
        linear_pred = linear_model.predict(X_future)
        all_linear_predictions.append(linear_pred)
        
        # Ridge Regression
        ridge_model = Ridge(alpha=1.0)
        ridge_model.fit(X, y)
        ridge_pred = ridge_model.predict(X_future)
        all_ridge_predictions.append(ridge_pred)
        
        # Random Forest
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X, y)
        rf_pred = rf_model.predict(X_future)
        all_rf_predictions.append(rf_pred)
        
        # Simple LSTM (without HMM and regression)
        model = SimpleLSTM().to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        # Create sequences for LSTM
        X_lstm, y_lstm = [], []
        for j in range(len(initial_price) - 5):
            X_lstm.append(initial_price[j:j+5])
            y_lstm.append(initial_price[j+5])
        
        X_lstm = torch.FloatTensor(X_lstm).reshape(-1, 5, 1).to(device)
        y_lstm = torch.FloatTensor(y_lstm).unsqueeze(1).to(device)
        
        # Train LSTM model
        model.train()
        for epoch in range(50):
            optimizer.zero_grad()
            outputs = model(X_lstm)
            loss = criterion(outputs, y_lstm)
            loss.backward()
            optimizer.step()
        
        # Predict using LSTM
        model.eval()
        lstm_pred = []
        current_sequence = torch.FloatTensor(initial_price[-5:]).reshape(1, 5, 1).to(device)
        
        with torch.no_grad():
            for _ in range(prediction_length):
                output = model(current_sequence)
                next_pred = output.item()
                lstm_pred.append(next_pred)
                # Update the sequence for next prediction
                current_sequence = torch.cat((
                    current_sequence[:, 1:, :], 
                    torch.FloatTensor([[[next_pred]]]).to(device)
                ), dim=1)
        
        all_lstm_predictions.append(np.array(lstm_pred))
    
    # Calculate metrics for each method
    results = []
    
    for i in range(n_windows):
        # Get true prices for current window
        true = all_true_prices[i]
        
        # Get midterm section (days 30-60)
        if len(true) >= 60:
            midterm_true = true[30:60]
            
            # Linear Regression
            midterm_linear = all_linear_predictions[i][30:60]
            linear_mpa = 1 - np.mean(np.abs(midterm_true - midterm_linear) / midterm_true)
            
            # Ridge Regression
            midterm_ridge = all_ridge_predictions[i][30:60]
            ridge_mpa = 1 - np.mean(np.abs(midterm_true - midterm_ridge) / midterm_true)
            
            # Random Forest
            midterm_rf = all_rf_predictions[i][30:60]
            rf_mpa = 1 - np.mean(np.abs(midterm_true - midterm_rf) / midterm_true)
            
            # LSTM
            midterm_lstm = all_lstm_predictions[i][30:60]
            lstm_mpa = 1 - np.mean(np.abs(midterm_true - midterm_lstm) / midterm_true)
            
            results.append({
                'Window': i+1,
                'Linear_MPA': linear_mpa,
                'Ridge_MPA': ridge_mpa,
                'RF_MPA': rf_mpa,
                'LSTM_MPA': lstm_mpa
            })
    
    results_df = pd.DataFrame(results)
    return results_df, all_true_prices, all_linear_predictions, all_ridge_predictions, all_rf_predictions, all_lstm_predictions

def main():
    print("Initializing data simulation...")
    
    # Initialize data simulator
    data_sim = DataSimulator(n_stocks=10, n_industries=5, start_date='2018-01-01', end_date='2022-12-31')
    
    # Simulate market data
    market_data = data_sim.simulate_market_data()
    
    # Simulate stock data
    stock_data = data_sim.simulate_stock_data()
    
    # Simulate earnings announcements
    data_sim.simulate_earnings_announcements()
    
    # Embed anomalies
    data_sim.embed_anomalies()
    
    print("Data simulation complete!")
    
    # Select one stock for demonstration
    selected_stock = data_sim.stock_symbols[0]
    stock_price_data = data_sim.stock_data[selected_stock]
    
    # Plot the simulated stock data
    plt.figure(figsize=(14, 7))
    plt.subplot(2, 1, 1)
    plt.plot(stock_price_data.index, stock_price_data['Adj Close'], label='Stock Price')
    plt.plot(market_data.index, market_data['Adj Close'], label='Market Index')
    plt.title(f'Simulated Stock Price and Market Index: {selected_stock}')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.plot(stock_price_data.index, stock_price_data['Volume'], label='Volume')
    plt.title(f'Simulated Trading Volume: {selected_stock}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    # Split data into training and testing
    train_size = int(len(stock_price_data) * 0.8)
    train_data = stock_price_data.iloc[:train_size]
    test_data = stock_price_data.iloc[train_size:]
    train_market = market_data.iloc[:train_size]
    test_market = market_data.iloc[train_size:]
    
    print(f"Training data size: {len(train_data)}")
    print(f"Testing data size: {len(test_data)}")
    
    # Initialize and train Mid-LSTM model
    mid_lstm = MidLSTM(window_size=60)
    mid_lstm.train(train_data, train_market, epochs=20, learning_rate=0.001)
    
    # Perform rolling window prediction
    prediction_results = rolling_window_prediction(
        mid_lstm, 
        test_data, 
        test_market, 
        window_size=60, 
        prediction_length=60
    )
    
    true_prices = prediction_results['true_prices']
    predicted_prices = prediction_results['predicted_prices']
    
    # Plot a sample prediction
    if true_prices and predicted_prices:
        plot_predictions(true_prices, predicted_prices, window_idx=0, title=f'Mid-LSTM Stock Price Prediction: {selected_stock}')
        
        # Calculate and display metrics
        metrics_df = calculate_metrics(true_prices, predicted_prices)
        print("\nPrediction Metrics:")
        print(metrics_df.describe())
        
        # Compare with traditional methods
        print("\nComparing with traditional methods...")
        comparison_results, all_true, all_linear, all_ridge, all_rf, all_lstm = compare_with_traditional_methods(
            test_data, 
            test_market
        )
        print("\nComparison Results:")
        print(comparison_results.describe())
        
        # Plot comparison for a sample window
        window_idx = 0
        if window_idx < len(all_true):
            plt.figure(figsize=(14, 7))
            plt.plot(all_true[window_idx], label='True Price', color='black')
            plt.plot(all_linear[window_idx], label='Linear Regression', color='blue')
            plt.plot(all_ridge[window_idx], label='Ridge Regression', color='green')
            plt.plot(all_rf[window_idx], label='Random Forest', color='orange')
            plt.plot(all_lstm[window_idx], label='LSTM', color='red')
            plt.plot(predicted_prices[window_idx], label='Mid-LSTM', color='purple')
            
            plt.axvline(x=30, color='green', linestyle='--', label='Midterm Start')
            plt.axvspan(30, 60, alpha=0.2, color='green')
            
            plt.title(f'Comparison of Different Methods: {selected_stock}')
            plt.xlabel('Days')
            plt.ylabel('Price')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()
        
        # Run portfolio allocation
        print("\nRunning portfolio allocation...")
        portfolio_results = run_portfolio_allocation(true_prices, predicted_prices)
        
        # Print portfolio metrics
        print("\nPortfolio Performance Metrics:")
        print(f"Total Return: {portfolio_results['total_return']:.2%}")
        print(f"Annualized Return: {portfolio_results['annualized_return']:.2%}")
        print(f"Annualized Volatility: {portfolio_results['annualized_volatility']:.2%}")
        print(f"Sharpe Ratio: {portfolio_results['sharpe_ratio']:.2f}")
        
        # Plot portfolio value over time
        plt.figure(figsize=(12, 6))
        plt.plot(portfolio_results['portfolio_values'])
        plt.title('Portfolio Value Over Time')
        plt.xlabel('Trading Windows')
        plt.ylabel('Portfolio Value ($)')
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    else:
        print("No prediction results available.")

if __name__ == "__main__":
    main()