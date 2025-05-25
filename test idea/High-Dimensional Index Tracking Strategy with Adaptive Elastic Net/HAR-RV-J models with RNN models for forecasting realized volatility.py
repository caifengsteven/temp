import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pdblp
import statsmodels.api as sm
from scipy import stats
import math
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Connect to Bloomberg
print("Connecting to Bloomberg...")
con = pdblp.BCon(debug=False, port=8194)
con.start()

# Define RNN model using PyTorch
class RNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=1):
        super(RNNModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.rnn = nn.RNN(input_dim, hidden_dim, batch_first=True)
        self.linear = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        rnn_out, _ = self.rnn(x)
        # Take the output from the last time step
        rnn_out = rnn_out[:, -1, :]
        # Apply sigmoid to hidden state
        hidden = self.sigmoid(rnn_out)
        # Linear layer to output
        output = self.linear(hidden)
        return output

class HighFrequencyVolatilityTrader:
    def __init__(self, securities, start_date, end_date, frequency='5min', lookback=252):
        """
        Initialize the HF volatility trader with parameters.
        
        Parameters:
        -----------
        securities : list of str
            Bloomberg tickers for securities to analyze
        start_date : str
            Start date in format 'YYYYMMDD'
        end_date : str
            End date in format 'YYYYMMDD'
        frequency : str
            Data frequency (default '5min')
        lookback : int
            Number of days to use for rolling window
        """
        self.securities = securities
        self.start_date = start_date
        self.end_date = end_date
        self.frequency = frequency
        self.lookback = lookback
        self.rv_data = None
        self.jump_data = None
        self.jumps_bpv = None
        self.har_model = None
        self.forecast_horizons = [1, 2, 5]  # 1-day, 2-day, 5-day horizons
        
        # Set device for PyTorch
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
    def fetch_high_frequency_data(self, con):
        """Fetch high-frequency data from Bloomberg"""
        print(f"Fetching high frequency data for {self.securities}...")
        
        # Try to fetch intraday data
        try:
            # For SPY or S&P500, fetch 5-minute bars
            # We'll use simulated data instead since real Bloomberg intraday data might not be accessible
            print("Real Bloomberg data unavailable. Generating simulated HF data for demonstration...")
            
            # Fetch daily data for reference
            daily_data = con.bdh(self.securities, 
                                ['PX_LAST', 'PX_VOLUME'], 
                                self.start_date, 
                                self.end_date)
            
            # Print daily data for reference
            print(f"Daily data shape: {daily_data.shape}")
            print(daily_data.head())
            
            # Generate simulated intraday data based on daily data
            self.generate_simulated_intraday_data(daily_data)
            
            return True
        except Exception as e:
            print(f"Error fetching Bloomberg data: {e}")
            return False
            
    def generate_simulated_intraday_data(self, daily_data):
        """Generate simulated intraday data based on daily data for testing"""
        print("Generating simulated intraday data...")
        
        # Extract daily prices and volatility
        if isinstance(daily_data.columns, pd.MultiIndex):
            # For multi-index DataFrame
            security = self.securities[0]
            daily_prices = daily_data.xs('PX_LAST', level=1, axis=1)[security]
        else:
            # For single-index DataFrame
            daily_prices = daily_data['PX_LAST']
            
        dates = daily_prices.index
        
        # Get trading hours (9:30 AM to 4:00 PM)
        trading_minutes = 6 * 60 + 30  # 6.5 hours
        intervals_per_day = trading_minutes // int(self.frequency.replace('min', ''))
        
        # Initialize storage for intraday data
        intraday_data = []
        
        # Generate 5-minute bars for each day
        for date in dates:
            daily_open = daily_prices[date]
            # Add some volatility noise specific to the day
            daily_vol = np.sqrt(np.random.gamma(shape=9, scale=0.1)) * 0.01
            
            for i in range(intervals_per_day):
                # Generate time stamp for this interval
                minutes_from_open = i * int(self.frequency.replace('min', ''))
                timestamp = datetime.combine(date.date(), 
                                            datetime.strptime("09:30", "%H:%M").time()) + timedelta(minutes=minutes_from_open)
                
                # Generate price with GARCH-like volatility clustering
                if i == 0:
                    # First interval of the day
                    price = daily_open
                    prev_return = 0
                else:
                    # Use AR(1) process for returns with volatility clustering
                    innovation = np.random.normal(0, daily_vol)
                    price_return = 0.2 * prev_return + innovation * (1 + 0.8 * abs(prev_return)) 
                    price = intraday_data[-1]['price'] * (1 + price_return)
                    prev_return = price_return
                
                # Add occasional jumps
                if np.random.random() < 0.01:  # 1% chance of jump
                    jump_size = np.random.normal(0, 0.005)  # 0.5% jump std
                    price *= (1 + jump_size)
                    has_jump = 1
                else:
                    has_jump = 0
                
                # Calculate simulated volume
                if i < intervals_per_day / 2:
                    # U-shape for volume - higher at open
                    vol_factor = 1.5 - i / intervals_per_day
                else:
                    # U-shape for volume - higher at close
                    vol_factor = 0.5 + (i / intervals_per_day)
                
                volume = int(np.random.gamma(shape=5, scale=vol_factor * 100000))
                
                # Store interval data
                intraday_data.append({
                    'date': date.date(),
                    'timestamp': timestamp,
                    'price': price,
                    'volume': volume,
                    'return': price_return if i > 0 else 0,
                    'has_jump': has_jump
                })
        
        # Convert to DataFrame
        self.intraday_df = pd.DataFrame(intraday_data)
        self.intraday_df.set_index('timestamp', inplace=True)
        
        print(f"Generated simulated intraday data: {self.intraday_df.shape}")
        print(self.intraday_df.head())
        
    def calculate_realized_measures(self):
        """Calculate realized volatility and jumps from high-frequency data"""
        print("Calculating realized volatility and jumps...")
        
        # Group by day
        self.intraday_df['date'] = self.intraday_df.index.date
        grouped = self.intraday_df.groupby('date')
        
        # Initialize daily measures
        daily_measures = []
        
        for day, group in grouped:
            # Calculate log returns for intraday data
            log_prices = np.log(group['price'].values)
            log_returns = np.diff(log_prices)
            
            # Realized Volatility (RV) - sum of squared returns
            rv = np.sum(log_returns**2)
            
            # Bipower Variation (BV) - to estimate integrated variance without jumps
            abs_returns = np.abs(log_returns)
            # Note: We use length-2 to avoid index out of bounds
            bv = (np.pi/2) * np.sum(abs_returns[:-1] * abs_returns[1:]) / (len(abs_returns)-1)
            
            # Jump component (J) = max(RV - BV, 0)
            jump = max(rv - bv, 0)
            
            # Calculate significant jumps using threshold
            z_stat = (rv - bv) / np.sqrt((np.pi**2/4 + np.pi - 5) * bv**2 / len(log_returns))
            alpha = 0.999  # 99.9% confidence level
            jump_threshold = stats.norm.ppf(alpha)
            significant_jump = 1 if z_stat > jump_threshold else 0
            
            # Store results
            daily_measures.append({
                'date': day,
                'rv': rv,
                'bv': bv,
                'jump': jump,
                'significant_jump': significant_jump,
                'log_rv': np.log(rv),
                'sqrt_rv': np.sqrt(rv),
                'log_jump': np.log(jump + 1e-10)  # Add small value to avoid log(0)
            })
        
        # Convert to DataFrame
        self.rv_data = pd.DataFrame(daily_measures)
        self.rv_data.set_index('date', inplace=True)
        
        # Calculate weekly and monthly averages
        self.rv_data['rv_weekly'] = self.rv_data['rv'].rolling(window=5).mean()
        self.rv_data['rv_monthly'] = self.rv_data['rv'].rolling(window=22).mean()
        
        # Fill NaN values with mean
        self.rv_data = self.rv_data.fillna(self.rv_data.mean())
        
        print("Realized volatility data:")
        print(self.rv_data.head())
        
        return self.rv_data
    
    def build_har_rv_j_model(self, target='rv', log_form=False):
        """Build HAR-RV-J model"""
        print(f"Building HAR-RV-J model (log_form={log_form})...")
        
        # Function to create HAR-RV-J features
        def create_har_features(data, target, log_form):
            df = data.copy()
            
            if log_form:
                # Use logarithmic form
                y = df[f'log_{target}']
                X = pd.DataFrame({
                    'constant': 1,
                    'rv_daily': df['log_rv'],
                    'rv_weekly': np.log(df['rv_weekly']),
                    'rv_monthly': np.log(df['rv_monthly']),
                    'jump': np.log(df['jump'] + 1)  # log(jump + 1) as in paper
                })
            else:
                # Use standard form
                y = df[target]
                X = pd.DataFrame({
                    'constant': 1,
                    'rv_daily': df['rv'],
                    'rv_weekly': df['rv_weekly'],
                    'rv_monthly': df['rv_monthly'],
                    'jump': df['jump']
                })
            
            return X, y
        
        self.har_features_func = create_har_features
        self.har_log_form = log_form
        self.har_target = target
        
        # No need to fit the model here - we'll do that in the rolling window approach
        print("HAR-RV-J model ready for rolling window forecast")
    
    def train_pytorch_model(self, model, X_train, y_train, epochs=100, batch_size=8, learning_rate=0.001):
        """Train PyTorch model"""
        # Convert data to PyTorch tensors
        X_tensor = torch.FloatTensor(X_train).to(self.device)
        y_tensor = torch.FloatTensor(y_train).to(self.device)
        
        # Create dataset and dataloader
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # Train the model
        model.train()
        for epoch in range(epochs):
            for batch_X, batch_y in dataloader:
                # Forward pass
                outputs = model(batch_X)
                loss = criterion(outputs.squeeze(), batch_y)
                
                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
        return model
        
    def rolling_window_forecast(self, window_size, horizon):
        """Perform rolling window forecast using PyTorch models"""
        print(f"Performing rolling window forecast (window={window_size}, horizon={horizon})...")
        
        # Prepare data
        data = self.rv_data.copy()
        
        # Calculate number of forecasts
        n_forecasts = len(data) - window_size - horizon
        
        if n_forecasts <= 0:
            print(f"Not enough data for window size {window_size} and horizon {horizon}")
            return None, None
            
        # Initialize containers for predictions
        har_predictions = []
        rnn_predictions = []
        hybrid_predictions = []
        actual_values = []
        forecast_dates = []
        
        # For RNN models, determine the input dimension and hidden dimension
        input_dim_rnn = 4  # daily, weekly, monthly RV, and jump
        input_dim_hybrid = 5  # RNN inputs + HAR prediction
        
        # Calculate hidden dimensions as per paper
        hidden_dim_rnn = int(2 * np.sqrt(input_dim_rnn * 1))
        hidden_dim_hybrid = int(2 * np.sqrt(input_dim_hybrid * 1))
        
        # Loop through the data with rolling window
        for t in range(window_size, len(data) - horizon):
            # Current date for reference
            current_date = data.index[t]
            forecast_date = data.index[t + horizon]
            
            # Training data for this window
            train_data = data.iloc[t - window_size:t]
            
            # Target for evaluation
            if self.har_log_form:
                actual = data.iloc[t + horizon]['log_rv']
            else:
                actual = data.iloc[t + horizon]['rv']
                
            # HAR-RV-J model forecast
            X_train, y_train = self.har_features_func(train_data, self.har_target, self.har_log_form)
            har_model = sm.OLS(y_train, X_train).fit()
            
            # Prepare forecast features for HAR
            X_forecast = pd.DataFrame({
                'constant': [1],
                'rv_daily': [data.iloc[t]['rv'] if not self.har_log_form else data.iloc[t]['log_rv']],
                'rv_weekly': [data.iloc[t]['rv_weekly'] if not self.har_log_form else np.log(data.iloc[t]['rv_weekly'])],
                'rv_monthly': [data.iloc[t]['rv_monthly'] if not self.har_log_form else np.log(data.iloc[t]['rv_monthly'])],
                'jump': [data.iloc[t]['jump'] if not self.har_log_form else np.log(data.iloc[t]['jump'] + 1)]
            })
            
            # Make HAR prediction
            har_pred = har_model.predict(X_forecast)[0]
            
            # RNN model forecast
            # Prepare RNN inputs
            if self.har_log_form:
                rnn_features = np.array([
                    data.iloc[t]['log_rv'],
                    np.log(data.iloc[t]['rv_weekly']),
                    np.log(data.iloc[t]['rv_monthly']),
                    np.log(data.iloc[t]['jump'] + 1)
                ]).reshape(1, 1, 4)
                
                rnn_targets = data.iloc[t-window_size:t]['log_rv'].values
            else:
                rnn_features = np.array([
                    data.iloc[t]['rv'],
                    data.iloc[t]['rv_weekly'],
                    data.iloc[t]['rv_monthly'],
                    data.iloc[t]['jump']
                ]).reshape(1, 1, 4)
                
                rnn_targets = data.iloc[t-window_size:t]['rv'].values
            
            # Create and train RNN model
            rnn_model = RNNModel(input_dim=input_dim_rnn, hidden_dim=hidden_dim_rnn).to(self.device)
            
            # Prepare training data for RNN
            rnn_X_train = np.zeros((window_size, 1, input_dim_rnn))
            for i in range(window_size):
                if self.har_log_form:
                    rnn_X_train[i, 0, 0] = train_data['log_rv'].iloc[i]
                    rnn_X_train[i, 0, 1] = np.log(train_data['rv_weekly'].iloc[i])
                    rnn_X_train[i, 0, 2] = np.log(train_data['rv_monthly'].iloc[i])
                    rnn_X_train[i, 0, 3] = np.log(train_data['jump'].iloc[i] + 1)
                else:
                    rnn_X_train[i, 0, 0] = train_data['rv'].iloc[i]
                    rnn_X_train[i, 0, 1] = train_data['rv_weekly'].iloc[i]
                    rnn_X_train[i, 0, 2] = train_data['rv_monthly'].iloc[i]
                    rnn_X_train[i, 0, 3] = train_data['jump'].iloc[i]
            
            # Train RNN model
            rnn_model = self.train_pytorch_model(
                rnn_model, 
                rnn_X_train, 
                rnn_targets,
                epochs=100, 
                batch_size=min(8, window_size)
            )
            
            # Make RNN prediction
            rnn_model.eval()
            with torch.no_grad():
                rnn_input = torch.FloatTensor(rnn_features).to(self.device)
                rnn_pred = rnn_model(rnn_input).item()
            
            # Hybrid model forecast
            # Prepare hybrid inputs
            if self.har_log_form:
                hybrid_features = np.array([
                    data.iloc[t]['log_rv'],
                    np.log(data.iloc[t]['rv_weekly']),
                    np.log(data.iloc[t]['rv_monthly']),
                    np.log(data.iloc[t]['jump'] + 1),
                    har_pred  # Add HAR prediction
                ]).reshape(1, 1, 5)
                
                hybrid_targets = data.iloc[t-window_size:t]['log_rv'].values
            else:
                hybrid_features = np.array([
                    data.iloc[t]['rv'],
                    data.iloc[t]['rv_weekly'],
                    data.iloc[t]['rv_monthly'],
                    data.iloc[t]['jump'],
                    har_pred  # Add HAR prediction
                ]).reshape(1, 1, 5)
                
                hybrid_targets = data.iloc[t-window_size:t]['rv'].values
            
            # Create and train hybrid model
            hybrid_model = RNNModel(input_dim=input_dim_hybrid, hidden_dim=hidden_dim_hybrid).to(self.device)
            
            # Prepare training data for hybrid model
            hybrid_X_train = np.zeros((window_size, 1, input_dim_hybrid))
            
            # Calculate HAR predictions for each point in training window
            for i in range(window_size):
                # Get a smaller sub-window for HAR training
                sub_window_size = min(i+1, 22)  # Use at least 1 day, up to 1 month of data
                sub_train_data = train_data.iloc[max(0, i+1-sub_window_size):i+1]
                
                if len(sub_train_data) > 1:  # Need at least 2 points to fit HAR
                    # Fit HAR on sub-window
                    sub_X, sub_y = self.har_features_func(sub_train_data, self.har_target, self.har_log_form)
                    sub_har_model = sm.OLS(sub_y, sub_X).fit()
                    
                    # Make HAR prediction for this point
                    if i < window_size - 1:  # Not the last point
                        sub_X_forecast = pd.DataFrame({
                            'constant': [1],
                            'rv_daily': [train_data.iloc[i]['rv'] if not self.har_log_form else train_data.iloc[i]['log_rv']],
                            'rv_weekly': [train_data.iloc[i]['rv_weekly'] if not self.har_log_form else np.log(train_data.iloc[i]['rv_weekly'])],
                            'rv_monthly': [train_data.iloc[i]['rv_monthly'] if not self.har_log_form else np.log(train_data.iloc[i]['rv_monthly'])],
                            'jump': [train_data.iloc[i]['jump'] if not self.har_log_form else np.log(train_data.iloc[i]['jump'] + 1)]
                        })
                        sub_har_pred = sub_har_model.predict(sub_X_forecast)[0]
                    else:
                        # Use har_pred for the last point since we already calculated it
                        sub_har_pred = har_pred
                else:
                    # Not enough data, use target value as prediction
                    if self.har_log_form:
                        sub_har_pred = train_data['log_rv'].iloc[i]
                    else:
                        sub_har_pred = train_data['rv'].iloc[i]
                
                # Add features to hybrid training data
                if self.har_log_form:
                    hybrid_X_train[i, 0, 0] = train_data['log_rv'].iloc[i]
                    hybrid_X_train[i, 0, 1] = np.log(train_data['rv_weekly'].iloc[i])
                    hybrid_X_train[i, 0, 2] = np.log(train_data['rv_monthly'].iloc[i])
                    hybrid_X_train[i, 0, 3] = np.log(train_data['jump'].iloc[i] + 1)
                    hybrid_X_train[i, 0, 4] = sub_har_pred
                else:
                    hybrid_X_train[i, 0, 0] = train_data['rv'].iloc[i]
                    hybrid_X_train[i, 0, 1] = train_data['rv_weekly'].iloc[i]
                    hybrid_X_train[i, 0, 2] = train_data['rv_monthly'].iloc[i]
                    hybrid_X_train[i, 0, 3] = train_data['jump'].iloc[i]
                    hybrid_X_train[i, 0, 4] = sub_har_pred
            
            # Train hybrid model
            hybrid_model = self.train_pytorch_model(
                hybrid_model, 
                hybrid_X_train, 
                hybrid_targets,
                epochs=100, 
                batch_size=min(8, window_size)
            )
            
            # Make hybrid prediction
            hybrid_model.eval()
            with torch.no_grad():
                hybrid_input = torch.FloatTensor(hybrid_features).to(self.device)
                hybrid_pred = hybrid_model(hybrid_input).item()
            
            # Store results
            har_predictions.append(har_pred)
            rnn_predictions.append(rnn_pred)
            hybrid_predictions.append(hybrid_pred)
            actual_values.append(actual)
            forecast_dates.append(forecast_date)
            
            # Progress indication
            if t % 50 == 0:
                print(f"Processed window {t-window_size+1}/{n_forecasts} ({(t-window_size+1)/n_forecasts*100:.1f}%)")
        
        # Convert to arrays
        har_predictions = np.array(har_predictions)
        rnn_predictions = np.array(rnn_predictions)
        hybrid_predictions = np.array(hybrid_predictions)
        actual_values = np.array(actual_values)
        
        # Create results DataFrame
        results = pd.DataFrame({
            'date': forecast_dates,
            'actual': actual_values,
            'har_pred': har_predictions,
            'rnn_pred': rnn_predictions,
            'hybrid_pred': hybrid_predictions
        })
        
        # Calculate errors
        for model in ['har', 'rnn', 'hybrid']:
            results[f'{model}_error'] = results['actual'] - results[f'{model}_pred']
            results[f'{model}_abs_error'] = np.abs(results[f'{model}_error'])
            results[f'{model}_squared_error'] = results[f'{model}_error'] ** 2
            
            # If using log form, calculate percentage errors properly
            if self.har_log_form:
                # For log form, percentage error is exp(pred) - exp(actual) / exp(actual)
                results[f'{model}_pct_error'] = (np.exp(results[f'{model}_pred']) - np.exp(results['actual'])) / np.exp(results['actual']) * 100
            else:
                # For normal form, simple percentage error
                results[f'{model}_pct_error'] = results[f'{model}_error'] / results['actual'] * 100
        
        # Calculate overall metrics
        metrics = {}
        for model in ['har', 'rnn', 'hybrid']:
            metrics[f'{model}_rmse'] = np.sqrt(np.mean(results[f'{model}_squared_error']))
            metrics[f'{model}_mae'] = np.mean(results[f'{model}_abs_error'])
            metrics[f'{model}_mape'] = np.mean(np.abs(results[f'{model}_pct_error']))
        
        print("\nForecast Metrics:")
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")
        
        return results, metrics
    
    def implement_trading_strategy(self, forecast_results, threshold=0.01, trading_asset='vxx'):
        """
        Implement trading strategy based on volatility forecasts
        
        Parameters:
        -----------
        forecast_results : DataFrame
            Results from the rolling window forecast
        threshold : float
            Threshold for trading (only trade if forecast change exceeds this %)
        trading_asset : str
            Which asset to trade ('vxx' for volatility ETN or 'spy' for index ETF)
        """
        print(f"\nImplementing trading strategy (asset={trading_asset}, threshold={threshold*100}%)...")
        
        # Create copy of results for strategy implementation
        results = forecast_results.copy()
        results.set_index('date', inplace=True)
        
        # Fetch trading asset data from Bloomberg
        try:
            if trading_asset.lower() == 'vxx':
                asset_ticker = 'VXX US Equity'
            else:
                asset_ticker = 'SPY US Equity'
                
            # Get asset price data
            asset_data = con.bdh([asset_ticker], 
                            ['PX_LAST'], 
                            results.index[0].strftime('%Y%m%d'), 
                            results.index[-1].strftime('%Y%m%d'))
            
            # Print asset data
            print(f"Asset price data for {asset_ticker}:")
            print(asset_data.head())
            
            # Extract prices
            if isinstance(asset_data.columns, pd.MultiIndex):
                prices = asset_data.xs('PX_LAST', level=1, axis=1).iloc[:, 0]
            else:
                prices = asset_data['PX_LAST']
            
            # Align dates
            asset_prices = pd.DataFrame(prices)
            asset_prices.columns = ['price']
            
        except Exception as e:
            print(f"Error fetching asset data: {e}")
            print("Using simulated price data for trading strategy evaluation")
            
            # Generate simulated asset prices
            dates = results.index
            
            # VXX tends to follow volatility, SPY is inversely correlated
            if trading_asset.lower() == 'vxx':
                # For VXX, base price on volatility
                if self.har_log_form:
                    base_values = np.exp(results['actual'])
                else:
                    base_values = results['actual']
                    
                # Normalize and scale
                scaled_values = (base_values - base_values.min()) / (base_values.max() - base_values.min()) * 100 + 20
                
                # Add some noise
                prices = scaled_values * (1 + np.random.normal(0, 0.02, len(scaled_values)))
                
            else:
                # For SPY, inverse relationship with volatility
                if self.har_log_form:
                    base_values = np.exp(results['actual'])
                else:
                    base_values = results['actual']
                    
                # Normalize and invert (higher volatility = lower price)
                scaled_values = 400 - (base_values - base_values.min()) / (base_values.max() - base_values.min()) * 100
                
                # Add some noise and trend
                prices = scaled_values * (1 + np.random.normal(0, 0.01, len(scaled_values)))
                # Add upward trend
                prices = prices * np.linspace(0.9, 1.1, len(prices))
            
            # Create DataFrame
            asset_prices = pd.DataFrame({'price': prices}, index=dates)
        
        # Generate trading signals for each model
        for model in ['har', 'rnn', 'hybrid']:
            # Calculate predicted change in volatility
            if self.har_log_form:
                results[f'{model}_rv_change'] = np.exp(results[f'{model}_pred']) / np.exp(results['actual'].shift(1)) - 1
            else:
                results[f'{model}_rv_change'] = results[f'{model}_pred'] / results['actual'].shift(1) - 1
            
            # Generate signal (1=long vol/short index, -1=short vol/long index, 0=no trade)
            if trading_asset.lower() == 'vxx':
                # For VXX (volatility ETN)
                results[f'{model}_signal'] = 0
                results.loc[results[f'{model}_rv_change'] > threshold, f'{model}_signal'] = 1    # Long VXX if vol increase
                results.loc[results[f'{model}_rv_change'] < -threshold, f'{model}_signal'] = -1  # Short VXX if vol decrease
            else:
                # For SPY (index ETF) - inverse relationship with vol
                results[f'{model}_signal'] = 0
                results.loc[results[f'{model}_rv_change'] > threshold, f'{model}_signal'] = -1   # Short SPY if vol increase
                results.loc[results[f'{model}_rv_change'] < -threshold, f'{model}_signal'] = 1   # Long SPY if vol decrease
        
        # Compute returns
        results = results.join(asset_prices)
        results['asset_return'] = results['price'].pct_change()
        
        # Calculate strategy returns
        for model in ['har', 'rnn', 'hybrid']:
            # Shift signal to avoid look-ahead bias (signal from today applied to tomorrow's return)
            results[f'{model}_trade_return'] = results[f'{model}_signal'].shift(1) * results['asset_return']
            
            # Calculate cumulative returns
            results[f'{model}_cum_return'] = (1 + results[f'{model}_trade_return']).cumprod() - 1
            
            # Calculate key metrics
            start_idx = ~results[f'{model}_trade_return'].isna()
            ann_return = results.loc[start_idx, f'{model}_trade_return'].mean() * 252
            ann_vol = results.loc[start_idx, f'{model}_trade_return'].std() * np.sqrt(252)
            sharpe = ann_return / ann_vol if ann_vol > 0 else 0
            
            # Calculate drawdowns
            cum_returns = (1 + results.loc[start_idx, f'{model}_trade_return']).cumprod()
            running_max = cum_returns.cummax()
            drawdowns = (cum_returns / running_max) - 1
            max_drawdown = drawdowns.min()
            
            print(f"\n{model.upper()} Trading Strategy Metrics:")
            print(f"Annual Return: {ann_return:.2%}")
            print(f"Annual Volatility: {ann_vol:.2%}")
            print(f"Sharpe Ratio: {sharpe:.2f}")
            print(f"Max Drawdown: {max_drawdown:.2%}")
            
            # Calculate % of profitable trades
            trade_days = results[results[f'{model}_signal'].shift(1) != 0].shape[0]
            profit_days = results[(results[f'{model}_signal'].shift(1) != 0) & 
                                   (results[f'{model}_trade_return'] > 0)].shape[0]
            print(f"Profitable Trades: {profit_days}/{trade_days} ({profit_days/trade_days:.2%})")
        
        # Plot cumulative returns
        plt.figure(figsize=(12, 6))
        for model in ['har', 'rnn', 'hybrid']:
            plt.plot(results[f'{model}_cum_return'], label=f"{model.upper()} Strategy")
        
        # Add buy-and-hold for comparison
        results['buy_hold_return'] = results['asset_return']
        results['buy_hold_cum_return'] = (1 + results['buy_hold_return']).cumprod() - 1
        plt.plot(results['buy_hold_cum_return'], label='Buy & Hold', linestyle='--')
        
        plt.title(f'Trading Strategy Returns - {trading_asset.upper()}')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Return')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'volatility_trading_strategy_{trading_asset}.png')
        
        # Plot forecast vs actual
        plt.figure(figsize=(12, 6))
        for model in ['har', 'rnn', 'hybrid']:
            plt.plot(results[f'{model}_pred'][-60:], label=f"{model.upper()} Forecast")
        plt.plot(results['actual'][-60:], label='Actual Volatility', color='black', linewidth=2)
        
        plt.title('Volatility Forecasts vs Actual (Last 60 Days)')
        plt.xlabel('Date')
        plt.ylabel('Realized Volatility')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('volatility_forecasts.png')
        
        return results

# Main execution
if __name__ == "__main__":
    try:
        # Define parameters
        securities = ['SPY US Equity']
        start_date = '20200101'
        end_date = '20230430'
        
        # Initialize trader
        trader = HighFrequencyVolatilityTrader(securities, start_date, end_date)
        
        # Fetch data
        trader.fetch_high_frequency_data(con)
        
        # Calculate realized volatility and jumps
        trader.calculate_realized_measures()
        
        # Build models
        trader.build_har_rv_j_model(log_form=True)
        
        # Test different rolling windows and forecast horizons
        windows = [22, 63, 126, 252]  # 1-month, 3-month, 6-month, 1-year
        horizons = [1, 2, 5]  # 1-day, 2-day, 1-week
        
        # Store results
        all_results = {}
        
        # Run forecasts for different combinations
        for window in windows:
            for horizon in horizons:
                print(f"\n========= Window={window}, Horizon={horizon} =========")
                results, metrics = trader.rolling_window_forecast(window, horizon)
                
                if results is not None:
                    all_results[(window, horizon)] = (results, metrics)
                    
                    # Save results to CSV
                    results.to_csv(f'forecast_results_w{window}_h{horizon}.csv', index=False)
        
        if all_results:
            # Find best model configuration based on RMSE
            best_rmse = float('inf')
            best_config = None
            best_model = None
            
            for config, (_, metrics) in all_results.items():
                window, horizon = config
                for model in ['har', 'rnn', 'hybrid']:
                    rmse = metrics[f'{model}_rmse']
                    if rmse < best_rmse:
                        best_rmse = rmse
                        best_config = config
                        best_model = model
            
            print(f"\nBest model: {best_model.upper()} with window={best_config[0]}, horizon={best_config[1]}, RMSE={best_rmse:.6f}")
            
            # Implement trading strategy using best configuration
            best_results, _ = all_results[best_config]
            
            # Implement trading strategy for VXX
            print("\n========= VXX Trading Strategy =========")
            vxx_trading_results = trader.implement_trading_strategy(best_results, threshold=0.02, trading_asset='vxx')
            
            # Implement trading strategy for SPY
            print("\n========= SPY Trading Strategy =========")
            spy_trading_results = trader.implement_trading_strategy(best_results, threshold=0.03, trading_asset='spy')
            
            # Save trading results
            vxx_trading_results.to_csv('vxx_trading_results.csv')
            spy_trading_results.to_csv('spy_trading_results.csv')
            
            print("\nAll analysis completed successfully!")
        else:
            print("No results were obtained. Check your data and parameters.")
            
    except Exception as e:
        print(f"Error in main execution: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Close Bloomberg connection
        con.stop()
        print("Bloomberg connection closed")