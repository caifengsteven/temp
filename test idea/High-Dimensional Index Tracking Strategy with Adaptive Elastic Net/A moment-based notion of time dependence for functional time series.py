import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import linalg
from tqdm import tqdm
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class FunctionalTimeSeries:
    """
    Class for generating and analyzing functional time series with different 
    dependence structures.
    """
    
    def __init__(self, T=500, n_points=100, basis_dim=5, beta=1.5, eigenvalue_decay=2.0, seed=None):
        """
        Initialize a functional time series object.
        
        Parameters:
        -----------
        T : int
            Number of time points
        n_points : int
            Number of points in the function domain [0,1]
        basis_dim : int
            Number of basis functions to use
        beta : float
            Parameter controlling the decay of autocovariances (beta < 1 for long memory)
        eigenvalue_decay : float
            Parameter controlling the decay of eigenvalues (alpha in the paper)
        seed : int or None
            Random seed for reproducibility
        """
        self.T = T
        self.n_points = n_points
        self.basis_dim = basis_dim
        self.beta = beta
        self.eigenvalue_decay = eigenvalue_decay
        
        if seed is not None:
            np.random.seed(seed)
        
        # Generate domain points
        self.domain = np.linspace(0, 1, n_points)
        
        # Generate orthonormal basis functions
        self.basis_functions = self._generate_basis_functions()
        
        # Generate eigenvalues with decay pattern
        self.eigenvalues = self._generate_eigenvalues()
        
        # The FTS data and projections will be set when generating a specific process
        self.data = None
        self.projections = None
    
    def _generate_basis_functions(self):
        """Generate an orthonormal basis for the functional space."""
        basis = []
        for j in range(1, self.basis_dim + 1):
            if j % 2 == 1:  # Odd: sine functions
                func = lambda x, j=j: np.sqrt(2) * np.sin(j * np.pi * x)
            else:  # Even: cosine functions
                func = lambda x, j=j: np.sqrt(2) * np.cos((j-1) * np.pi * x)
            basis.append(func)
        return basis
    
    def _generate_eigenvalues(self):
        """Generate eigenvalues with polynomial decay."""
        return np.array([k**(-self.eigenvalue_decay) for k in range(1, self.basis_dim + 1)])
    
    def evaluate_basis(self):
        """Evaluate all basis functions on the domain."""
        basis_eval = np.zeros((self.n_points, self.basis_dim))
        for j in range(self.basis_dim):
            basis_eval[:, j] = self.basis_functions[j](self.domain)
        return basis_eval
    
    def estimate_mean_function(self):
        """Estimate the mean function of the FTS."""
        if self.data is None:
            raise ValueError("No data available. Generate FTS first.")
        
        # Simply take the mean over time
        return np.mean(self.data, axis=0)
    
    def estimate_covariance_operator(self, centered=True):
        """
        Estimate the covariance operator of the FTS.
        
        Parameters:
        -----------
        centered : bool
            Whether to center the data before computing covariance
            
        Returns:
        --------
        cov_operator : numpy.ndarray
            Estimated covariance operator
        """
        if self.data is None:
            raise ValueError("No data available. Generate FTS first.")
        
        if centered:
            data_centered = self.data - np.mean(self.data, axis=0)
        else:
            data_centered = self.data
        
        # Compute empirical covariance
        cov_operator = np.zeros((self.n_points, self.n_points))
        for t in range(self.T):
            x_t = data_centered[t, :].reshape(-1, 1)
            cov_operator += x_t @ x_t.T
        
        cov_operator /= self.T
        
        return cov_operator
    
    def compute_FPCA(self, cov_operator=None):
        """
        Compute Functional Principal Component Analysis.
        
        Parameters:
        -----------
        cov_operator : numpy.ndarray or None
            Covariance operator. If None, estimate it from data.
            
        Returns:
        --------
        eigenvalues : numpy.ndarray
            Estimated eigenvalues
        eigenfunctions : numpy.ndarray
            Estimated eigenfunctions
        """
        if cov_operator is None:
            cov_operator = self.estimate_covariance_operator()
        
        # Perform eigendecomposition
        eigenvalues, eigenfunctions = linalg.eigh(cov_operator)
        
        # Sort in descending order (linalg.eigh returns ascending)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenfunctions = eigenfunctions[:, idx]
        
        return eigenvalues, eigenfunctions

class MarketDataSimulator:
    """Class for simulating market data, including stocks and yield curves."""
    
    def __init__(self, n_stocks=5, n_days=500, seed=None):
        """
        Initialize the market data simulator.
        
        Parameters:
        -----------
        n_stocks : int
            Number of stocks to simulate
        n_days : int
            Number of days to simulate
        seed : int
            Random seed for reproducibility
        """
        self.n_stocks = n_stocks
        self.n_days = n_days
        
        if seed is not None:
            np.random.seed(seed)
        
        # Market parameters
        self.market_volatility = 0.01  # Daily market volatility
        self.stock_volatility = 0.02  # Additional stock-specific volatility
        
        # Data structure to store simulated data
        self.stock_data = {}
        self.yield_curve_data = {}
        self.intraday_data = {}
        self.options_data = {}
        
        # Date range for simulation
        self.start_date = datetime(2020, 1, 1)
        self.dates = [self.start_date + timedelta(days=i) for i in range(n_days)]
        
        # Ticker symbols
        self.tickers = [f"STOCK{i+1}" for i in range(n_stocks)]
    
    def simulate_stock_prices(self):
        """
        Simulate daily stock prices with realistic properties.
        
        Returns:
        --------
        dict
            Dictionary of stock price dataframes
        """
        # Initialize stock prices at 100
        prices = np.zeros((self.n_days, self.n_stocks))
        prices[0, :] = 100
        
        # Simulate market factor - AR(1) process with mean reversion
        market_returns = np.zeros(self.n_days)
        market_returns[0] = np.random.normal(0.0005, self.market_volatility)
        
        for t in range(1, self.n_days):
            # Market return has mean reversion
            market_returns[t] = 0.0005 - 0.05 * market_returns[t-1] + np.random.normal(0, self.market_volatility)
        
        # Simulate stock-specific factors
        for i in range(self.n_stocks):
            # Stock-specific beta to market
            beta = np.random.uniform(0.8, 1.2)
            
            # Stock-specific returns with momentum
            stock_returns = np.zeros(self.n_days)
            
            for t in range(1, self.n_days):
                # Stock return has market component, momentum, and idiosyncratic component
                market_component = beta * market_returns[t]
                momentum = 0.1 * (prices[t-1, i] / prices[max(0, t-20), i] - 1)
                idiosyncratic = np.random.normal(0, self.stock_volatility)
                
                # Combine components
                stock_returns[t] = market_component + 0.05 * momentum + idiosyncratic
                
                # Update price
                prices[t, i] = prices[t-1, i] * (1 + stock_returns[t])
        
        # Create pandas DataFrames
        for i, ticker in enumerate(self.tickers):
            # Create OHLC data
            open_prices = prices[:, i]
            high_prices = open_prices * (1 + np.random.uniform(0, 0.015, self.n_days))
            low_prices = open_prices * (1 - np.random.uniform(0, 0.015, self.n_days))
            close_prices = open_prices * (1 + np.random.normal(0, 0.005, self.n_days))
            
            # Ensure high > open > low
            high_prices = np.maximum(high_prices, open_prices)
            low_prices = np.minimum(low_prices, open_prices)
            
            # Create volume data - higher on big moves
            returns = np.diff(np.append([0], close_prices)) / close_prices
            volume = np.random.normal(1e6, 2e5, self.n_days) * (1 + 5 * np.abs(returns))
            
            # Create DataFrame
            df = pd.DataFrame({
                'Open': open_prices,
                'High': high_prices,
                'Low': low_prices,
                'Close': close_prices,
                'Volume': volume
            }, index=self.dates)
            
            self.stock_data[ticker] = df
        
        return self.stock_data
    
    def simulate_yield_curves(self):
        """
        Simulate daily yield curves with realistic properties.
        
        Returns:
        --------
        dict
            Dictionary with yield curve data
        """
        # Maturities in years
        maturities = np.array([0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30])
        
        # Normalize maturities to [0,1] domain
        normalized_maturities = maturities / np.max(maturities)
        
        # Base yield curve (upward sloping)
        base_curve = 0.5 + 2.5 * (1 - np.exp(-0.3 * maturities))
        
        # Simulate yield curve factors (level, slope, curvature) using AR(1) processes
        level = np.zeros(self.n_days)
        slope = np.zeros(self.n_days)
        curvature = np.zeros(self.n_days)
        
        # Initial values
        level[0] = np.random.normal(0, 0.5)
        slope[0] = np.random.normal(0, 0.2)
        curvature[0] = np.random.normal(0, 0.1)
        
        # Generate AR(1) processes
        for t in range(1, self.n_days):
            level[t] = 0.98 * level[t-1] + np.random.normal(0, 0.1)  # Highly persistent
            slope[t] = 0.95 * slope[t-1] + np.random.normal(0, 0.05)
            curvature[t] = 0.90 * curvature[t-1] + np.random.normal(0, 0.03)
        
        # Generate yield curves for each day
        yields = np.zeros((self.n_days, len(maturities)))
        
        for t in range(self.n_days):
            # Nelson-Siegel model inspiration
            yields[t] = base_curve + level[t] - slope[t] * np.exp(-0.5 * maturities) + \
                        curvature[t] * (np.exp(-0.5 * maturities) - np.exp(-2 * maturities))
            
            # Add small noise
            yields[t] += np.random.normal(0, 0.03, len(maturities))
            
            # Ensure yields are positive
            yields[t] = np.maximum(yields[t], 0.05)
        
        # Store the results
        self.yield_curve_data = {
            'maturities': maturities,
            'normalized_maturities': normalized_maturities,
            'yields': yields,
            'dates': self.dates
        }
        
        return self.yield_curve_data
    
    def simulate_intraday_data(self, n_points=20):
        """
        Simulate intraday price patterns.
        
        Parameters:
        -----------
        n_points : int
            Number of intraday price points to simulate
            
        Returns:
        --------
        dict
            Dictionary with intraday data
        """
        # First ensure we have daily data
        if not self.stock_data:
            self.simulate_stock_prices()
        
        # For each stock and each day, generate intraday prices
        intraday_data = {}
        
        for ticker in self.tickers:
            daily_data = self.stock_data[ticker]
            ticker_intraday = []
            
            for date, row in daily_data.iterrows():
                open_price = row['Open']
                close_price = row['Close']
                high_price = row['High']
                low_price = row['Low']
                
                # Time points (normalized to [0,1])
                time_points = np.linspace(0, 1, n_points)
                
                # Generate intraday pattern
                # Use a random walk with constraints on open/close/high/low
                intraday_prices = np.zeros(n_points)
                intraday_prices[0] = open_price
                intraday_prices[-1] = close_price
                
                # First pass: random walk
                for i in range(1, n_points-1):
                    # More volatility in the middle of the day
                    time_factor = 4 * time_points[i] * (1 - time_points[i])  # Peaks at 0.5
                    volatility = 0.002 * time_factor
                    
                    # Random step
                    intraday_prices[i] = intraday_prices[i-1] * (1 + np.random.normal(0, volatility))
                
                # Second pass: ensure high/low constraints
                current_high = max(intraday_prices)
                current_low = min(intraday_prices)
                
                if current_high < high_price:
                    # Pick a random point to be the high
                    high_idx = np.random.randint(1, n_points-1)
                    intraday_prices[high_idx] = high_price
                
                if current_low > low_price:
                    # Pick a random point to be the low
                    low_idx = np.random.randint(1, n_points-1)
                    intraday_prices[low_idx] = low_price
                
                # Store the result
                ticker_intraday.append((date, time_points, intraday_prices))
            
            intraday_data[ticker] = ticker_intraday
        
        self.intraday_data = intraday_data
        return intraday_data
    
    def simulate_volatility_smiles(self, n_strikes=15):
        """
        Simulate volatility smiles for options.
        
        Parameters:
        -----------
        n_strikes : int
            Number of strike prices to simulate
            
        Returns:
        --------
        dict
            Dictionary with volatility smile data
        """
        # First ensure we have daily data
        if not self.stock_data:
            self.simulate_stock_prices()
        
        # For each stock and each day, generate volatility smiles
        vol_smile_data = {}
        
        for ticker in self.tickers:
            daily_data = self.stock_data[ticker]
            ticker_vol_smiles = []
            
            for date, row in daily_data.iterrows():
                close_price = row['Close']
                
                # Generate strikes around the current price (moneyness)
                moneyness = np.linspace(0.8, 1.2, n_strikes)
                strikes = close_price * moneyness
                
                # Normalized strikes to [0,1] domain
                normalized_strikes = (strikes - strikes.min()) / (strikes.max() - strikes.min())
                
                # Base volatility (higher for longer dated options)
                base_vol = 0.2
                
                # Simulate volatility smile (U-shaped curve)
                volatility = base_vol + 0.1 * (moneyness - 1)**2
                
                # Add some noise and time variation
                day_of_year = date.timetuple().tm_yday
                seasonal_factor = 0.05 * np.sin(2 * np.pi * day_of_year / 365)
                
                volatility += seasonal_factor + np.random.normal(0, 0.01, n_strikes)
                
                # Ensure volatilities are positive
                volatility = np.maximum(volatility, 0.05)
                
                # Store the result
                ticker_vol_smiles.append((date, normalized_strikes, volatility))
            
            vol_smile_data[ticker] = ticker_vol_smiles
        
        self.options_data = vol_smile_data
        return vol_smile_data
    
    def get_functional_data(self, data_type='intraday'):
        """
        Convert simulated data to functional data format for FTS analysis.
        
        Parameters:
        -----------
        data_type : str
            Type of data to convert ('intraday', 'yield_curve', or 'vol_smile')
            
        Returns:
        --------
        dict
            Dictionary of functional data
        """
        functional_data = {}
        
        if data_type == 'intraday':
            # Ensure we have intraday data
            if not self.intraday_data:
                self.simulate_intraday_data()
            
            # Convert to functional format
            for ticker, ticker_data in self.intraday_data.items():
                functional_data[ticker] = {
                    'data': [(time_points, prices) for _, time_points, prices in ticker_data],
                    'dates': [date for date, _, _ in ticker_data]
                }
        
        elif data_type == 'yield_curve':
            # Ensure we have yield curve data
            if not self.yield_curve_data:
                self.simulate_yield_curves()
            
            # Convert to functional format
            yield_data = []
            for i in range(self.n_days):
                yield_data.append((
                    self.yield_curve_data['normalized_maturities'],
                    self.yield_curve_data['yields'][i]
                ))
            
            functional_data['yield_curve'] = {
                'data': yield_data,
                'dates': self.dates
            }
        
        elif data_type == 'vol_smile':
            # Ensure we have volatility smile data
            if not self.options_data:
                self.simulate_volatility_smiles()
            
            # Convert to functional format
            for ticker, ticker_data in self.options_data.items():
                functional_data[ticker + '_vol'] = {
                    'data': [(strikes, vols) for _, strikes, vols in ticker_data],
                    'dates': [date for date, _, _ in ticker_data]
                }
        
        else:
            raise ValueError(f"Unknown data type: {data_type}")
        
        return functional_data

class FTSTrader:
    """
    Trading strategy implementation using Functional Time Series analysis
    """
    
    def __init__(self, fts_analyzer, lookback_days=30, forecast_horizon=5, n_eigenfunctions=3):
        """
        Initialize the FTS-based trading strategy.
        
        Parameters:
        -----------
        fts_analyzer : FunctionalTimeSeries
            The functional time series analyzer
        lookback_days : int
            Number of days to look back for pattern recognition
        forecast_horizon : int
            Number of days to forecast ahead
        n_eigenfunctions : int
            Number of eigenfunctions to use for forecasting
        """
        self.fts_analyzer = fts_analyzer
        self.lookback_days = lookback_days
        self.forecast_horizon = forecast_horizon
        self.n_eigenfunctions = n_eigenfunctions
        
        # Will store data for trading
        self.functional_data = None
        self.portfolio = None
        self.cash = 10000  # Starting cash
        self.positions = {}
        self.daily_values = []
    
    def convert_to_fts(self, ticker):
        """
        Convert functional data for a ticker to FTS format for analysis.
        
        Parameters:
        -----------
        ticker : str
            Ticker symbol
            
        Returns:
        --------
        numpy.ndarray
            FTS data ready for analysis
        """
        if ticker not in self.functional_data:
            raise ValueError(f"No data for ticker {ticker}")
        
        ticker_data = self.functional_data[ticker]['data']
        
        # Create a uniform grid for all functions
        n_points = 100
        domain = np.linspace(0, 1, n_points)
        
        # Interpolate each day's data to the uniform grid
        fts_data = np.zeros((len(ticker_data), n_points))
        
        for i, (times, values) in enumerate(ticker_data):
            # Interpolate to uniform grid
            fts_data[i] = np.interp(domain, times, values)
        
        # Set in the analyzer
        self.fts_analyzer.domain = domain
        self.fts_analyzer.data = fts_data
        self.fts_analyzer.T = len(fts_data)
        
        return fts_data
    
    def forecast_functional(self, ticker, day_index):
        """
        Forecast the next functional shape for a given ticker and day.
        
        Parameters:
        -----------
        ticker : str
            Ticker symbol
        day_index : int
            Index of the current day
            
        Returns:
        --------
        numpy.ndarray
            Forecasted function
        """
        # Convert data to FTS format
        self.convert_to_fts(ticker)
        
        # Get current day data
        if day_index < self.lookback_days:
            return None  # Not enough historical data
        
        # Use data from lookback_days prior to current day
        start_idx = max(0, day_index - self.lookback_days)
        lookback_data = self.fts_analyzer.data[start_idx:day_index]
        
        # Set analyzer data to lookback period
        self.fts_analyzer.data = lookback_data
        self.fts_analyzer.T = len(lookback_data)
        
        # Perform FPCA
        eigenvalues, eigenfunctions = self.fts_analyzer.compute_FPCA()
        
        # Get scores for lookback period
        scores = np.zeros((self.fts_analyzer.T, self.n_eigenfunctions))
        mean_function = self.fts_analyzer.estimate_mean_function()
        
        for t in range(self.fts_analyzer.T):
            centered = self.fts_analyzer.data[t] - mean_function
            for j in range(self.n_eigenfunctions):
                scores[t, j] = np.sum(centered * eigenfunctions[:, j])
        
        # For each eigenfunction, forecast the score
        forecasted_scores = np.zeros(self.n_eigenfunctions)
        
        for j in range(self.n_eigenfunctions):
            # Simple AR(1) forecast
            if len(scores) > 1:
                # Fit AR(1) model: s_t = a + b * s_{t-1}
                X = scores[:-1, j].reshape(-1, 1)
                y = scores[1:, j]
                
                # Add constant term
                X = np.hstack([np.ones_like(X), X])
                
                # Fit model
                beta = np.linalg.lstsq(X, y, rcond=None)[0]
                
                # Forecast
                forecasted_scores[j] = beta[0] + beta[1] * scores[-1, j]
            else:
                forecasted_scores[j] = scores[-1, j]
        
        # Reconstruct forecasted function
        forecasted_function = mean_function.copy()
        for j in range(self.n_eigenfunctions):
            forecasted_function += forecasted_scores[j] * eigenfunctions[:, j]
        
        return forecasted_function
    
    def generate_signals(self, ticker, threshold=0.02):
        """
        Generate trading signals based on forecasted curves.
        
        Parameters:
        -----------
        ticker : str
            Ticker symbol
        threshold : float
            Threshold for generating signals (% change)
            
        Returns:
        --------
        list
            List of trading signals (1: buy, -1: sell, 0: hold)
        """
        ticker_data = self.functional_data[ticker]['data']
        ticker_dates = self.functional_data[ticker]['dates']
        
        signals = []
        
        for i in range(self.lookback_days, len(ticker_data)):
            # Forecast next curve
            forecasted_curve = self.forecast_functional(ticker, i)
            
            if forecasted_curve is None:
                signals.append(0)
                continue
            
            # Get current curve and compute expected change
            _, current_values = ticker_data[i]
            
            if 'vol' in ticker:  # For volatility smiles
                # For volatility, we're looking at the ATM volatility (middle point)
                mid_idx = len(current_values) // 2
                current_vol = current_values[mid_idx]
                forecasted_vol = forecasted_curve[mid_idx]
                
                # Trading logic: Buy when volatility is expected to increase
                expected_change = (forecasted_vol / current_vol) - 1
                
                if expected_change > threshold:
                    signals.append(1)  # Long volatility
                elif expected_change < -threshold:
                    signals.append(-1)  # Short volatility
                else:
                    signals.append(0)  # Hold
                
            elif ticker == 'yield_curve':  # For yield curves
                # For yield curve, we're looking at the slope (long-short)
                short_idx = 0  # Shortest maturity
                long_idx = -1  # Longest maturity
                
                current_slope = current_values[long_idx] - current_values[short_idx]
                forecasted_slope = forecasted_curve[long_idx] - forecasted_curve[short_idx]
                
                # Trading logic: Buy when curve steepens (positive carry)
                slope_change = forecasted_slope - current_slope
                
                if slope_change > threshold:
                    signals.append(1)  # Steepening expected
                elif slope_change < -threshold:
                    signals.append(-1)  # Flattening expected
                else:
                    signals.append(0)  # Hold
                
            else:  # For regular stock data (intraday)
                # For intraday data, compare close price to forecasted close
                current_close = current_values[-1]
                forecasted_close = forecasted_curve[-1]
                
                # Calculate expected return
                expected_return = (forecasted_close / current_close) - 1
                
                # Generate signal
                if expected_return > threshold:
                    signals.append(1)  # Buy signal
                elif expected_return < -threshold:
                    signals.append(-1)  # Sell signal
                else:
                    signals.append(0)  # Hold
        
        return signals
    
    def backtest_strategy(self, tickers, start_capital=10000, position_size=0.1):
        """
        Backtest the trading strategy on historical data.
        
        Parameters:
        -----------
        tickers : list
            List of ticker symbols
        start_capital : float
            Initial capital for backtesting
        position_size : float
            Size of each position as a fraction of portfolio value
            
        Returns:
        --------
        dict
            Backtest results
        """
        # Initialize portfolio
        self.cash = start_capital
        self.positions = {ticker: 0 for ticker in tickers}
        self.daily_values = [start_capital]
        
        # Track trades
        trades = []
        
        # Align dates across all tickers
        all_dates = set()
        for ticker in tickers:
            if ticker in self.functional_data:
                all_dates.update(self.functional_data[ticker]['dates'])
        
        all_dates = sorted(all_dates)
        
        # For each date
        for date_idx, date in enumerate(all_dates[self.lookback_days:], self.lookback_days):
            portfolio_value = self.cash
            for ticker in tickers:
                if ticker in self.positions:
                    # Get current price
                    if date in self.functional_data[ticker]['dates']:
                        ticker_idx = self.functional_data[ticker]['dates'].index(date)
                        current_price = self.functional_data[ticker]['data'][ticker_idx][1][-1]  # Use closing price
                        portfolio_value += self.positions[ticker] * current_price
            
            # Generate signals
            for ticker in tickers:
                if ticker in self.functional_data and date in self.functional_data[ticker]['dates']:
                    ticker_idx = self.functional_data[ticker]['dates'].index(date)
                    
                    # Only proceed if we have enough history
                    if ticker_idx >= self.lookback_days:
                        # Get signal
                        signals = self.generate_signals(ticker)
                        signal = signals[ticker_idx - self.lookback_days] if ticker_idx - self.lookback_days < len(signals) else 0
                        
                        # Get current price
                        current_price = self.functional_data[ticker]['data'][ticker_idx][1][-1]  # Use closing price
                        
                        # Execute trades based on signal
                        if signal == 1:  # Buy signal
                            # Calculate position size
                            amount_to_invest = portfolio_value * position_size
                            shares_to_buy = amount_to_invest / current_price
                            
                            # Check if we have enough cash
                            if self.cash >= amount_to_invest:
                                self.positions[ticker] += shares_to_buy
                                self.cash -= amount_to_invest
                                
                                trades.append({
                                    'date': date,
                                    'ticker': ticker,
                                    'action': 'BUY',
                                    'price': current_price,
                                    'shares': shares_to_buy,
                                    'value': amount_to_invest
                                })
                        
                        elif signal == -1:  # Sell signal
                            # If we have a position, sell it
                            if self.positions[ticker] > 0:
                                amount_to_sell = self.positions[ticker] * current_price
                                self.cash += amount_to_sell
                                
                                trades.append({
                                    'date': date,
                                    'ticker': ticker,
                                    'action': 'SELL',
                                    'price': current_price,
                                    'shares': self.positions[ticker],
                                    'value': amount_to_sell
                                })
                                
                                self.positions[ticker] = 0
            
            # Update portfolio value
            portfolio_value = self.cash
            for ticker in tickers:
                if ticker in self.positions:
                    # Get current price if available
                    if date in self.functional_data[ticker]['dates']:
                        ticker_idx = self.functional_data[ticker]['dates'].index(date)
                        current_price = self.functional_data[ticker]['data'][ticker_idx][1][-1]  # Use closing price
                        portfolio_value += self.positions[ticker] * current_price
            
            self.daily_values.append(portfolio_value)
        
        # Calculate performance metrics
        returns = np.diff(self.daily_values) / self.daily_values[:-1]
        
        results = {
            'final_value': self.daily_values[-1],
            'total_return': (self.daily_values[-1] / start_capital) - 1,
            'annualized_return': ((self.daily_values[-1] / start_capital) ** (252 / len(self.daily_values))) - 1,
            'sharpe_ratio': np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0,
            'max_drawdown': self._calculate_max_drawdown(self.daily_values),
            'trades': trades,
            'daily_values': self.daily_values
        }
        
        return results
    
    def _calculate_max_drawdown(self, values):
        """Calculate maximum drawdown from a series of values."""
        peak = values[0]
        max_drawdown = 0
        
        for value in values:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            max_drawdown = max(max_drawdown, drawdown)
        
        return max_drawdown
    
    def plot_results(self, results):
        """
        Plot the backtest results.
        
        Parameters:
        -----------
        results : dict
            Results from backtest_strategy
        """
        plt.figure(figsize=(15, 10))
        
        # Plot portfolio value
        plt.subplot(2, 1, 1)
        plt.plot(results['daily_values'])
        plt.title('Portfolio Value Over Time')
        plt.xlabel('Trading Days')
        plt.ylabel('Portfolio Value ($)')
        plt.grid(True)
        
        # Plot trades
        plt.subplot(2, 1, 2)
        buy_dates = [i for i, t in enumerate(results['trades']) if t['action'] == 'BUY']
        buy_values = [t['value'] for t in results['trades'] if t['action'] == 'BUY']
        
        sell_dates = [i for i, t in enumerate(results['trades']) if t['action'] == 'SELL']
        sell_values = [t['value'] for t in results['trades'] if t['action'] == 'SELL']
        
        plt.bar(buy_dates, buy_values, color='green', alpha=0.6, label='Buy')
        plt.bar(sell_dates, sell_values, color='red', alpha=0.6, label='Sell')
        
        plt.title('Trades')
        plt.xlabel('Trading Day')
        plt.ylabel('Trade Value ($)')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
        
        # Print performance metrics
        print("\nPerformance Metrics:")
        print(f"Final Portfolio Value: ${results['final_value']:.2f}")
        print(f"Total Return: {results['total_return']:.2%}")
        print(f"Annualized Return: {results['annualized_return']:.2%}")
        print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
        print(f"Maximum Drawdown: {results['max_drawdown']:.2%}")
        print(f"Number of Trades: {len(results['trades'])}")

def main():
    """Demo FTS-based trading strategy with simulated data."""
    # Initialize market data simulator
    print("Simulating market data...")
    simulator = MarketDataSimulator(n_stocks=5, n_days=500, seed=42)
    
    # Simulate different types of functional data
    simulator.simulate_stock_prices()
    simulator.simulate_intraday_data(n_points=20)
    simulator.simulate_yield_curves()
    simulator.simulate_volatility_smiles(n_strikes=15)
    
    # Convert to functional format
    intraday_data = simulator.get_functional_data('intraday')
    yield_curve_data = simulator.get_functional_data('yield_curve')
    vol_smile_data = simulator.get_functional_data('vol_smile')
    
    # Initialize FTS analyzer
    fts_analyzer = FunctionalTimeSeries(T=0, n_points=100, basis_dim=5)
    
    # Initialize trader
    trader = FTSTrader(fts_analyzer, lookback_days=20, forecast_horizon=1, n_eigenfunctions=3)
    
    print("\nTesting intraday trading strategy...")
    # Test intraday strategy
    trader.functional_data = intraday_data
    intraday_results = trader.backtest_strategy(['STOCK1', 'STOCK2', 'STOCK3'], 
                                         start_capital=10000, position_size=0.1)
    trader.plot_results(intraday_results)
    
    print("\nTesting yield curve trading strategy...")
    # Test yield curve strategy
    trader.functional_data = yield_curve_data
    yield_results = trader.backtest_strategy(['yield_curve'], 
                                     start_capital=10000, position_size=0.2)
    trader.plot_results(yield_results)
    
    print("\nTesting volatility smile trading strategy...")
    # Test volatility smile strategy
    trader.functional_data = vol_smile_data
    vol_results = trader.backtest_strategy(['STOCK1_vol', 'STOCK2_vol'], 
                                   start_capital=10000, position_size=0.15)
    trader.plot_results(vol_results)
    
    # Compare strategies
    plt.figure(figsize=(12, 6))
    plt.plot(intraday_results['daily_values'], label='Intraday Strategy')
    plt.plot(yield_results['daily_values'], label='Yield Curve Strategy')
    plt.plot(vol_results['daily_values'], label='Volatility Smile Strategy')
    plt.title('Strategy Comparison')
    plt.xlabel('Trading Days')
    plt.ylabel('Portfolio Value ($)')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()