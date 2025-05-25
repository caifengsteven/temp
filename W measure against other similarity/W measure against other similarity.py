import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import tqdm
import time
import warnings
warnings.filterwarnings('ignore')

# For Bloomberg data access
try:
    import pdblp
    from pdblp import BCon
    has_bloomberg = True
except ImportError:
    has_bloomberg = False
    print("Bloomberg API not available. Using alternative data sources.")
    import yfinance as yf

class SimilarityMeasures:
    """
    Implementation of various similarity measures for financial time series
    as described in the paper by Juszczuk, Kozak, and Kania.
    """
    
    @staticmethod
    def euclidean_distance(series_a, series_b):
        """
        Calculate Euclidean distance between two time series
        """
        return np.sqrt(np.sum((series_a - series_b) ** 2))
    
    @staticmethod
    def manhattan_distance(series_a, series_b):
        """
        Calculate Manhattan distance between two time series
        """
        return np.sum(np.abs(series_a - series_b))
    
    @staticmethod
    def tschebyschev_distance(series_a, series_b):
        """
        Calculate Tschebyschev distance between two time series
        """
        return np.max(np.abs(series_a - series_b))
    
    @staticmethod
    def correlation(series_a, series_b):
        """
        Calculate Pearson correlation between two time series
        """
        return np.corrcoef(series_a, series_b)[0, 1]
    
    @staticmethod
    def dtw_distance(series_a, series_b, window=None):
        """
        Fast implementation of Dynamic Time Warping with Sakoe-Chiba band
        
        Parameters:
        -----------
        series_a, series_b : array-like
            Time series to compare
        window : int, optional
            Width of the Sakoe-Chiba band (default: None uses a heuristic)
        """
        n, m = len(series_a), len(series_b)
        
        # Default window size based on series length
        if window is None:
            window = max(int(max(n, m) * 0.1), 1)  # default 10% of max length
        
        # Initialize cost matrix
        cost_matrix = np.full((n + 1, m + 1), np.inf)
        cost_matrix[0, 0] = 0
        
        # Fill cost matrix
        for i in range(1, n + 1):
            for j in range(max(1, i - window), min(m + 1, i + window + 1)):
                cost = (series_a[i - 1] - series_b[j - 1]) ** 2
                cost_matrix[i, j] = cost + min(
                    cost_matrix[i - 1, j],     # insertion
                    cost_matrix[i, j - 1],     # deletion
                    cost_matrix[i - 1, j - 1]  # match
                )
        
        return np.sqrt(cost_matrix[n, m])
    
    @staticmethod
    def w_measure(series_a, series_b, epsilon=None):
        """
        Implementation of the W measure as described in the paper
        
        Parameters:
        -----------
        series_a, series_b : array-like
            Time series to compare
        epsilon : float, optional
            Filter threshold for unimportant fluctuations
        """
        # Make sure we're working with numpy arrays
        series_a = np.array(series_a)
        series_b = np.array(series_b)
        
        n = len(series_a)
        
        # If epsilon is not provided, use a heuristic based on the data
        if epsilon is None:
            epsilon = 0.001  # Default 0.1% change
        
        # Compute relative changes for both series
        # Delta_a = a_i / a_(i-j) - 1
        delta_a = np.zeros(n - 1)
        delta_b = np.zeros(n - 1)
        
        for i in range(1, n):
            delta_a[i - 1] = series_a[i] / series_a[i - 1] - 1
            delta_b[i - 1] = series_b[i] / series_b[i - 1] - 1
        
        # Apply filter for unimportant fluctuations
        delta_a_filtered = np.where(np.abs(delta_a) < epsilon, 0, delta_a)
        delta_b_filtered = np.where(np.abs(delta_b) < epsilon, 0, delta_b)
        
        # Calculate weights for direction changes
        total_weight = 0
        total_diff = 0
        
        for i in range(len(delta_a_filtered)):
            # Calculate sign-based components
            sign_a = np.sign(delta_a_filtered[i])
            sign_b = np.sign(delta_b_filtered[i])
            
            # Same direction - sd component
            sd = 1 if sign_a == sign_b else 0
            
            # Different direction - dd component
            dd = 1 if sign_a == -sign_b and sign_a != 0 and sign_b != 0 else 0
            
            # Special case for zero (filtered values) - sd0 component
            sd0 = 1 if (sign_a == 0 and sign_b != 0) or (sign_a != 0 and sign_b == 0) else 0
            
            # Calculate weight for this comparison
            weight = (dd + 1) / ((sd + 1) * 2 * (sd0 + 1))
            
            # Calculate absolute difference in changes
            diff = abs(delta_a_filtered[i] - delta_b_filtered[i])
            
            total_weight += weight
            total_diff += weight * diff
        
        # Normalize by the number of comparisons
        denominator = (n * (n - 1)) / 2
        
        if total_weight == 0:
            return 0  # Perfect match
        
        # Final W measure value
        w_value = total_diff / denominator
        
        return w_value
    
    @staticmethod
    def relative_euclidean(series_a, series_b):
        """
        Euclidean distance on relative changes
        """
        # Calculate relative changes
        delta_a = np.diff(series_a) / series_a[:-1]
        delta_b = np.diff(series_b) / series_b[:-1]
        
        # Calculate Euclidean distance on changes
        return np.sqrt(np.sum((delta_a - delta_b) ** 2))
    
    @staticmethod
    def relative_manhattan(series_a, series_b):
        """
        Manhattan distance on relative changes
        """
        # Calculate relative changes
        delta_a = np.diff(series_a) / series_a[:-1]
        delta_b = np.diff(series_b) / series_b[:-1]
        
        # Calculate Manhattan distance on changes
        return np.sum(np.abs(delta_a - delta_b))
    
    @staticmethod
    def relative_tschebyschev(series_a, series_b):
        """
        Tschebyschev distance on relative changes
        """
        # Calculate relative changes
        delta_a = np.diff(series_a) / series_a[:-1]
        delta_b = np.diff(series_b) / series_b[:-1]
        
        # Calculate Tschebyschev distance on changes
        return np.max(np.abs(delta_a - delta_b))


class TradingStrategy:
    """
    Trading strategy implementation based on similarity measures
    """
    
    def __init__(self, symbols, start_date='2010-01-01', end_date=None, use_bloomberg=True):
        """
        Initialize the trading strategy
        
        Parameters:
        -----------
        symbols : list or dict
            List of ticker symbols or dict of symbol categories to analyze
        start_date : str, optional
            Start date for historical data (default: '2010-01-01')
        end_date : str, optional
            End date for historical data (default: current date)
        use_bloomberg : bool, optional
            Whether to use Bloomberg for data (default: True)
        """
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date) if end_date else pd.to_datetime('today')
        self.use_bloomberg = use_bloomberg and has_bloomberg
        
        # Check if symbols is a dict or list
        if isinstance(symbols, dict):
            self.symbol_categories = symbols
            self.symbols = []
            for category, symbol_list in symbols.items():
                self.symbols.extend(symbol_list)
        else:
            self.symbols = symbols
            self.symbol_categories = None
        
        # Initialize data storage
        self.data = {}
        
        # Load the data
        self.load_data()
        
        # Initialize similarity measures
        self.measures = {
            'euclidean': SimilarityMeasures.euclidean_distance,
            'manhattan': SimilarityMeasures.manhattan_distance,
            'tschebyschev': SimilarityMeasures.tschebyschev_distance,
            'correlation': SimilarityMeasures.correlation,
            'dtw': SimilarityMeasures.dtw_distance,
            'w_measure': SimilarityMeasures.w_measure,
            'rel_euclidean': SimilarityMeasures.relative_euclidean,
            'rel_manhattan': SimilarityMeasures.relative_manhattan,
            'rel_tschebyschev': SimilarityMeasures.relative_tschebyschev
        }
        
        # Results storage
        self.results = {}
    
    def load_data(self):
        """
        Load historical data for all symbols
        """
        if self.use_bloomberg:
            self._load_data_bloomberg()
        else:
            try:
                import yfinance as yf
                self._load_data_yfinance()
            except ImportError:
                print("Neither Bloomberg nor yfinance is available. Cannot load data.")
                return
    
    def _load_data_bloomberg(self):
        """
        Load data using Bloomberg API
        """
        try:
            print("Connecting to Bloomberg...")
            # Connect to Bloomberg with increased timeout
            con = BCon(debug=False, port=8194, timeout=60000)  # Increase timeout to 60 seconds
            con.start()
            print("Connected to Bloomberg successfully.")
            
            print(f"Fetching data for {len(self.symbols)} symbols...")
            
            # Process symbols in smaller batches to avoid overwhelming the API
            batch_size = 5
            num_batches = (len(self.symbols) + batch_size - 1) // batch_size
            
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, len(self.symbols))
                batch_symbols = self.symbols[start_idx:end_idx]
                
                print(f"Processing batch {batch_idx+1}/{num_batches} ({len(batch_symbols)} symbols)")
                
                for symbol in tqdm.tqdm(batch_symbols, desc=f"Loading Bloomberg data (batch {batch_idx+1})"):
                    # Prepare Bloomberg ticker
                    bb_ticker = symbol
                    
                    # Fetch daily historical prices with retry mechanism
                    max_retries = 3
                    for retry in range(max_retries):
                        try:
                            # Request data for shorter time periods to avoid timeouts
                            # Split the date range into chunks if it's longer than 5 years
                            date_chunks = self._split_date_range(self.start_date, self.end_date, max_years=5)
                            
                            all_data = []
                            for chunk_start, chunk_end in date_chunks:
                                print(f"  Fetching {symbol} data from {chunk_start.strftime('%Y-%m-%d')} to {chunk_end.strftime('%Y-%m-%d')}")
                                
                                chunk_data = con.bdh(
                                    tickers=bb_ticker,
                                    flds=['PX_LAST'],
                                    start_date=chunk_start.strftime('%Y%m%d'),
                                    end_date=chunk_end.strftime('%Y%m%d')
                                )
                                
                                if not chunk_data.empty:
                                    all_data.append(chunk_data)
                                
                                # Add a small delay between requests to avoid overwhelming the API
                                time.sleep(1)
                            
                            if all_data:
                                # Combine all chunks
                                df = pd.concat(all_data)
                                
                                # Remove any duplicate dates
                                df = df[~df.index.duplicated(keep='first')]
                                
                                # Extract price data
                                if isinstance(df.columns, pd.MultiIndex):
                                    # Find the PX_LAST column
                                    price_cols = [col for col in df.columns if 'PX_LAST' in col[1]]
                                    if price_cols:
                                        price_series = df[price_cols[0]].copy()
                                    else:
                                        price_series = df.iloc[:, 0].copy()
                                else:
                                    price_series = df.iloc[:, 0].copy()
                                
                                price_series.name = symbol
                                
                                # Store the data
                                self.data[symbol] = price_series
                                print(f"Loaded {len(price_series)} data points for {symbol}")
                                
                                # Success - break the retry loop
                                break
                            else:
                                print(f"No data returned for {symbol} on attempt {retry+1}/{max_retries}")
                                if retry < max_retries - 1:
                                    print(f"Retrying in 5 seconds...")
                                    time.sleep(5)
                        
                        except Exception as e:
                            if "Timeout" in str(e) and retry < max_retries - 1:
                                print(f"Timeout error for {symbol} on attempt {retry+1}/{max_retries}. Retrying in 5 seconds...")
                                time.sleep(5)
                            else:
                                print(f"Error fetching data for {symbol}: {e}")
                                break
                
                # Add a delay between batches
                if batch_idx < num_batches - 1:
                    print(f"Waiting 10 seconds before processing next batch...")
                    time.sleep(10)
            
            # Close the connection
            con.stop()
            print("Bloomberg connection closed.")
            
            # Check how many symbols were loaded
            loaded_symbols = len(self.data)
            print(f"Successfully loaded data for {loaded_symbols}/{len(self.symbols)} symbols")
            
            # If we couldn't load any data, try the fallback
            if loaded_symbols == 0:
                print("No data loaded from Bloomberg. Falling back to alternative data source.")
                try:
                    import yfinance as yf
                    self._load_data_yfinance()
                except ImportError:
                    print("Cannot use fallback: yfinance not available.")
            
        except Exception as e:
            print(f"Error with Bloomberg connection: {e}")
            try:
                import yfinance as yf
                print("Falling back to yfinance...")
                self._load_data_yfinance()
            except ImportError:
                print("Cannot load data: neither Bloomberg nor yfinance is available.")
    
    def _split_date_range(self, start_date, end_date, max_years=5):
        """
        Split a date range into chunks of max_years to avoid timeouts
        
        Parameters:
        -----------
        start_date : datetime
            Start date of the range
        end_date : datetime
            End date of the range
        max_years : int, optional
            Maximum number of years per chunk (default: 5)
            
        Returns:
        --------
        list
            List of (chunk_start, chunk_end) date pairs
        """
        chunks = []
        current_start = start_date
        
        while current_start < end_date:
            # Calculate chunk end date (max_years from current_start, or end_date if sooner)
            years_to_add = pd.DateOffset(years=max_years)
            chunk_end = min(current_start + years_to_add, end_date)
            
            chunks.append((current_start, chunk_end))
            
            # Move to next chunk start
            current_start = chunk_end + pd.DateOffset(days=1)
        
        return chunks
    
    def _load_data_yfinance(self):
        """
        Load data using yfinance (fallback)
        """
        try:
            import yfinance as yf
            
            for symbol in tqdm.tqdm(self.symbols, desc="Loading data from Yahoo Finance"):
                try:
                    # Convert Bloomberg ticker to Yahoo Finance format
                    yf_ticker = self._bloomberg_to_yahoo(symbol)
                    
                    # Download data
                    ticker = yf.Ticker(yf_ticker)
                    df = ticker.history(
                        start=self.start_date.strftime('%Y-%m-%d'),
                        end=self.end_date.strftime('%Y-%m-%d'),
                        interval="1d"
                    )
                    
                    if not df.empty:
                        # Use close prices
                        self.data[symbol] = df['Close']
                        print(f"Loaded {len(df)} data points for {symbol} (Yahoo: {yf_ticker})")
                    else:
                        print(f"No data returned from Yahoo Finance for {symbol} ({yf_ticker})")
                    
                except Exception as e:
                    print(f"Error fetching data for {symbol} from Yahoo Finance: {e}")
        except ImportError:
            print("yfinance not available for fallback data source.")
    
    def _bloomberg_to_yahoo(self, bb_ticker):
        """
        Convert Bloomberg ticker to Yahoo Finance format
        
        Parameters:
        -----------
        bb_ticker : str
            Bloomberg ticker
            
        Returns:
        --------
        str
            Yahoo Finance ticker
        """
        # Currency pairs: "EURUSD Curncy" -> "EURUSD=X"
        if " Curncy" in bb_ticker:
            return bb_ticker.replace(" Curncy", "=X")
        
        # Indices: "SPX Index" -> "^GSPC"
        index_map = {
            "SPX Index": "^GSPC",      # S&P 500
            "INDU Index": "^DJI",      # Dow Jones
            "CCMP Index": "^IXIC",     # NASDAQ
            "UKX Index": "^FTSE",      # FTSE 100
            "DAX Index": "^GDAXI",     # DAX
            "CAC Index": "^FCHI",      # CAC 40
            "NKY Index": "^N225",      # Nikkei 225
            "HSI Index": "^HSI",       # Hang Seng
            "RUT Index": "^RUT"        # Russell 2000
        }
        if bb_ticker in index_map:
            return index_map[bb_ticker]
        
        # US Stocks: "AAPL US Equity" -> "AAPL"
        if " US Equity" in bb_ticker:
            return bb_ticker.replace(" US Equity", "")
        
        # For other tickers, just return as is
        return bb_ticker
    
    def find_similar_patterns(self, symbol, pattern_length=10, measure_name='w_measure', 
                             epsilon=0.001, top_n=5, start_idx=None, end_idx=None):
        """
        Find historical patterns similar to the current pattern
        
        Parameters:
        -----------
        symbol : str
            Ticker symbol to analyze
        pattern_length : int, optional
            Length of pattern to look for (default: 10)
        measure_name : str, optional
            Name of similarity measure to use (default: 'w_measure')
        epsilon : float, optional
            Threshold for W measure (default: 0.001)
        top_n : int, optional
            Number of top similar patterns to return (default: 5)
        start_idx, end_idx : int, optional
            Range of indices to search in (default: entire series)
        
        Returns:
        --------
        dict
            Dictionary with similarity scores and indices
        """
        # Get the price series
        price_series = self.data[symbol]
        
        # Set default indices if not provided
        if start_idx is None:
            start_idx = 0
        if end_idx is None:
            end_idx = len(price_series) - pattern_length
        
        # Current pattern (the most recent pattern_length days)
        current_pattern = price_series.iloc[-pattern_length:].values
        
        # Get the similarity measure function
        measure_func = self.measures[measure_name]
        
        # Storage for similarity scores
        similarities = []
        
        # Iterate through historical data to find similar patterns
        for i in range(start_idx, end_idx):
            # Historical pattern
            hist_pattern = price_series.iloc[i:i+pattern_length].values
            
            # Skip if pattern contains NaN values
            if np.isnan(hist_pattern).any() or np.isnan(current_pattern).any():
                continue
                
            # Calculate similarity
            if measure_name == 'w_measure':
                sim_score = measure_func(hist_pattern, current_pattern, epsilon)
            elif measure_name in ['correlation', 'dtw']:
                sim_score = measure_func(hist_pattern, current_pattern)
            else:
                sim_score = measure_func(hist_pattern, current_pattern)
            
            # For correlation, convert to distance (1 - correlation)
            if measure_name == 'correlation':
                sim_score = 1 - abs(sim_score)
                
            # Store result
            similarities.append((i, sim_score))
        
        # Sort by similarity (ascending for distances, descending for correlation)
        similarities.sort(key=lambda x: x[1])
        
        # Return top N similar patterns
        return {
            'indices': [i for i, _ in similarities[:top_n]],
            'scores': [s for _, s in similarities[:top_n]]
        }
    
    def get_signal_from_pattern(self, symbol, pattern_idx, pattern_length=10, 
                               forecast_length=5, epsilon=None):
        """
        Get trading signal based on pattern continuation
        
        Parameters:
        -----------
        symbol : str
            Ticker symbol to analyze
        pattern_idx : int
            Starting index of the pattern
        pattern_length : int, optional
            Length of pattern (default: 10)
        forecast_length : int, optional
            Length of forecast (default: 5)
        epsilon : float, optional
            Threshold for determining HOLD signals (default: None)
            
        Returns:
        --------
        str
            Trading signal: 'BUY', 'SELL', or 'HOLD'
        float
            Predicted change percentage
        """
        # Get the price series
        price_series = self.data[symbol]
        
        # Set epsilon if not provided (average daily change over last 50 days)
        if epsilon is None:
            recent_changes = abs(price_series.pct_change().iloc[-50:])
            epsilon = recent_changes.mean()
        
        # Pattern end value
        pattern_end_value = price_series.iloc[pattern_idx + pattern_length - 1]
        
        # Future value (after forecast_length days)
        future_idx = pattern_idx + pattern_length + forecast_length - 1
        if future_idx >= len(price_series):
            return 'HOLD', 0.0  # Can't predict beyond available data
        
        future_value = price_series.iloc[future_idx]
        
        # Calculate percentage change
        pct_change = (future_value - pattern_end_value) / pattern_end_value
        
        # Determine signal
        if pct_change > epsilon:
            signal = 'BUY'
        elif pct_change < -epsilon:
            signal = 'SELL'
        else:
            signal = 'HOLD'
            
        return signal, pct_change
    
    def backtest_strategy(self, symbol, lookback_window=2000, pattern_length=10, 
                         forecast_length=5, measure_name='w_measure', epsilon=None,
                         initial_capital=10000, position_size=0.1, use_stop_loss=False,
                         stop_loss_pct=0.05):
        """
        Backtest the trading strategy
        
        Parameters:
        -----------
        symbol : str
            Ticker symbol to analyze
        lookback_window : int, optional
            Number of historical days to look back (default: 2000)
        pattern_length : int, optional
            Length of pattern to match (default: 10)
        forecast_length : int, optional
            Length of forecast (default: 5)
        measure_name : str, optional
            Similarity measure to use (default: 'w_measure')
        epsilon : float, optional
            Threshold for signals (default: None)
        initial_capital : float, optional
            Initial capital for backtesting portfolio (default: 10000)
        position_size : float, optional
            Fraction of capital to risk per trade (default: 0.1 = 10%)
        use_stop_loss : bool, optional
            Whether to use stop loss (default: False)
        stop_loss_pct : float, optional
            Stop loss percentage (default: 0.05 = 5%)
            
        Returns:
        --------
        DataFrame
            Backtest results
        """
        if symbol not in self.data:
            raise ValueError(f"No data available for {symbol}")
            
        # Get the price series
        price_series = self.data[symbol]
        
        # Make sure we have enough data
        if len(price_series) < lookback_window + pattern_length + forecast_length:
            adjusted_lookback = len(price_series) - pattern_length - forecast_length - 10
            if adjusted_lookback <= 0:
                raise ValueError(f"Not enough data for {symbol}. Have {len(price_series)} points.")
            
            print(f"Warning: Not enough data for full lookback window. Reducing lookback from {lookback_window} to {adjusted_lookback}")
            lookback_window = adjusted_lookback
        
        # Storage for results
        results = []
        
        # Set epsilon if not provided
        if epsilon is None:
            recent_changes = abs(price_series.pct_change().iloc[-50:])
            epsilon = recent_changes.mean()
        
        # Backtest period
        start_idx = lookback_window
        end_idx = len(price_series) - pattern_length - forecast_length
        
        # Initialize portfolio variables
        capital = initial_capital
        shares_held = 0
        entry_price = 0
        positions = []  # For tracking open positions
        
        # Portfolio tracking variables
        portfolio_values = []  # Value at each point
        portfolio_dates = []   # Dates corresponding to the values
        trades = []            # Record of trades
        
        for i in tqdm.tqdm(range(start_idx, end_idx, forecast_length), 
                          desc=f"Backtesting {symbol} with {measure_name}"):
            # Current pattern
            current_pattern = price_series.iloc[i:i+pattern_length]
            current_date = current_pattern.index[-1]
            current_price = price_series.iloc[i+pattern_length-1]
            
            # Find similar patterns in history
            similar_patterns = self.find_similar_patterns(
                symbol=symbol,
                pattern_length=pattern_length,
                measure_name=measure_name,
                epsilon=epsilon,
                top_n=1,
                start_idx=0,
                end_idx=i-pattern_length-forecast_length if i-pattern_length-forecast_length > 0 else 0
            )
            
            # If no similar patterns found, skip
            if not similar_patterns['indices']:
                continue
            
            # Get best match
            best_idx = similar_patterns['indices'][0]
            best_score = similar_patterns['scores'][0]
            
            # Get signal from best match
            signal, pred_change = self.get_signal_from_pattern(
                symbol=symbol,
                pattern_idx=best_idx,
                pattern_length=pattern_length,
                forecast_length=forecast_length,
                epsilon=epsilon
            )
            
            # Calculate actual future value
            future_idx = i + pattern_length + forecast_length - 1
            if future_idx >= len(price_series):
                continue
                
            future_value = price_series.iloc[future_idx]
            actual_value = price_series.iloc[i+pattern_length-1]
            actual_change = (future_value - actual_value) / actual_value
            
            # Determine actual signal
            if actual_change > epsilon:
                actual_signal = 'BUY'
            elif actual_change < -epsilon:
                actual_signal = 'SELL'
            else:
                actual_signal = 'HOLD'
            
            # Execute trade based on signal
            # Close previous position if we have one and are getting a new signal
            if shares_held != 0:
                # Calculate PnL from trade
                if shares_held > 0:  # Long position
                    pnl = shares_held * (current_price - entry_price)
                else:  # Short position
                    pnl = -shares_held * (current_price - entry_price)
                
                # Update capital
                capital += pnl
                
                # Record the trade
                trades.append({
                    'entry_date': positions[-1]['date'],
                    'exit_date': current_date,
                    'entry_price': entry_price,
                    'exit_price': current_price,
                    'shares': shares_held,
                    'pnl': pnl,
                    'return': pnl / (abs(shares_held) * entry_price)
                })
                
                # Close position
                shares_held = 0
                entry_price = 0
                positions = []
            
            # Open new position based on the signal
            if signal == 'BUY':
                # Calculate position size in dollars
                trade_value = capital * position_size
                
                # Calculate number of shares
                shares_to_buy = trade_value / current_price
                
                # Update holdings
                shares_held = shares_to_buy
                entry_price = current_price
                
                # Record position
                positions.append({
                    'date': current_date,
                    'price': current_price,
                    'shares': shares_held,
                    'type': 'long'
                })
                
            elif signal == 'SELL':
                # Calculate position size in dollars
                trade_value = capital * position_size
                
                # Calculate number of shares to short
                shares_to_short = trade_value / current_price
                
                # Update holdings
                shares_held = -shares_to_short  # Negative for short
                entry_price = current_price
                
                # Record position
                positions.append({
                    'date': current_date,
                    'price': current_price,
                    'shares': shares_held,
                    'type': 'short'
                })
            
            # Calculate current portfolio value
            if shares_held != 0:
                # Value of current position
                position_value = abs(shares_held) * current_price
                
                # For short positions, adjust for potential loss/gain
                if shares_held < 0:
                    position_value = position_value - shares_held * (current_price - entry_price)
                
                portfolio_value = capital + position_value
            else:
                portfolio_value = capital
            
            # Record portfolio value
            portfolio_values.append(portfolio_value)
            portfolio_dates.append(current_date)
            
            # Store backtest result
            results.append({
                'date': current_date,
                'predicted_signal': signal,
                'actual_signal': actual_signal,
                'predicted_change': pred_change,
                'actual_change': actual_change,
                'similarity_score': best_score,
                'match_idx': best_idx,
                'price': current_price,
                'portfolio_value': portfolio_value
            })
        
        # Convert to DataFrame
        results_df = pd.DataFrame(results)
        trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
        
        # Create portfolio DataFrame
        portfolio_df = pd.DataFrame({
            'date': portfolio_dates,
            'value': portfolio_values
        })
        portfolio_df.set_index('date', inplace=True)
        
        # Calculate equity curve metrics
        if len(portfolio_df) > 0:
            # Calculate returns
            portfolio_df['returns'] = portfolio_df['value'].pct_change()
            
            # Calculate cumulative returns
            portfolio_df['cumulative_returns'] = (1 + portfolio_df['returns']).cumprod() - 1
            
            # Calculate drawdowns
            portfolio_df['peak'] = portfolio_df['value'].cummax()
            portfolio_df['drawdown'] = (portfolio_df['value'] / portfolio_df['peak']) - 1
            
            # Calculate metrics
            total_return = (portfolio_df['value'].iloc[-1] / initial_capital) - 1
            ann_return = (1 + total_return) ** (252 / len(portfolio_df)) - 1
            max_drawdown = portfolio_df['drawdown'].min()
            
            # Annualized volatility
            ann_vol = portfolio_df['returns'].std() * np.sqrt(252)
            
            # Sharpe ratio (assuming 0% risk-free rate)
            sharpe_ratio = ann_return / ann_vol if ann_vol > 0 else 0
            
            # Calmar ratio
            calmar_ratio = ann_return / abs(max_drawdown) if max_drawdown < 0 else float('inf')
            
            # Win rate
            if len(trades_df) > 0:
                win_rate = (trades_df['pnl'] > 0).mean()
                avg_win = trades_df.loc[trades_df['pnl'] > 0, 'pnl'].mean() if any(trades_df['pnl'] > 0) else 0
                avg_loss = trades_df.loc[trades_df['pnl'] < 0, 'pnl'].mean() if any(trades_df['pnl'] < 0) else 0
                profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
            else:
                win_rate = 0
                avg_win = 0
                avg_loss = 0
                profit_factor = 0
            
            # Store the equity curve data and metrics
            equity_metrics = {
                'total_return': total_return,
                'annualized_return': ann_return,
                'annualized_volatility': ann_vol,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'calmar_ratio': calmar_ratio,
                'win_rate': win_rate,
                'profit_factor': profit_factor,
                'num_trades': len(trades_df),
                'final_capital': portfolio_df['value'].iloc[-1]
            }
        else:
            equity_metrics = {
                'total_return': 0,
                'annualized_return': 0,
                'annualized_volatility': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'calmar_ratio': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'num_trades': 0,
                'final_capital': initial_capital
            }
            portfolio_df = pd.DataFrame({'value': [initial_capital], 'cumulative_returns': [0]})
            
        # Calculate accuracy metrics
        if len(results_df) > 0:
            correct_signals = (results_df['predicted_signal'] == results_df['actual_signal']).sum()
            total_signals = len(results_df)
            accuracy = correct_signals / total_signals if total_signals > 0 else 0
            
            # Calculate precision and recall for each signal type
            metrics = {}
            for signal_type in ['BUY', 'SELL', 'HOLD']:
                # True positives for this signal
                tp = ((results_df['predicted_signal'] == signal_type) & 
                       (results_df['actual_signal'] == signal_type)).sum()
                
                # All predicted as this signal
                pred_as_signal = (results_df['predicted_signal'] == signal_type).sum()
                
                # All actual as this signal
                actual_as_signal = (results_df['actual_signal'] == signal_type).sum()
                
                # Calculate precision and recall
                precision = tp / pred_as_signal if pred_as_signal > 0 else 0
                recall = tp / actual_as_signal if actual_as_signal > 0 else 0
                
                metrics[f'{signal_type}_precision'] = precision
                metrics[f'{signal_type}_recall'] = recall
            
            # Store results
            self.results[f"{symbol}_{measure_name}"] = {
                'accuracy': accuracy,
                'metrics': metrics,
                'results_df': results_df,
                'equity_curve': portfolio_df,
                'equity_metrics': equity_metrics,
                'trades': trades_df
            }
            
            print(f"Backtest results for {symbol} using {measure_name}:")
            print(f"Accuracy: {accuracy:.4f}")
            print("Precision/Recall:")
            for signal_type in ['BUY', 'SELL', 'HOLD']:
                print(f"  {signal_type}: Precision={metrics[f'{signal_type}_precision']:.4f}, "
                      f"Recall={metrics[f'{signal_type}_recall']:.4f}")
            
            print("\nPortfolio Performance:")
            print(f"  Total Return: {equity_metrics['total_return']*100:.2f}%")
            print(f"  Annualized Return: {equity_metrics['annualized_return']*100:.2f}%")
            print(f"  Sharpe Ratio: {equity_metrics['sharpe_ratio']:.2f}")
            print(f"  Max Drawdown: {equity_metrics['max_drawdown']*100:.2f}%")
            print(f"  Win Rate: {equity_metrics['win_rate']*100:.2f}%")
            print(f"  Number of Trades: {equity_metrics['num_trades']}")
            print(f"  Final Capital: ${equity_metrics['final_capital']:.2f}")
        
        return results_df
    
    def backtest_all_measures(self, symbol, lookback_window=2000, pattern_length=10, 
                            forecast_length=5, epsilon=None, initial_capital=10000,
                            position_size=0.1):
        """
        Backtest the strategy using all available measures
        
        Parameters:
        -----------
        symbol : str
            Ticker symbol to analyze
        lookback_window : int, optional
            Number of historical days to look back (default: 2000)
        pattern_length : int, optional
            Length of pattern to match (default: 10)
        forecast_length : int, optional
            Length of forecast (default: 5)
        epsilon : float, optional
            Threshold for signals (default: None)
        initial_capital : float, optional
            Initial capital for backtesting portfolio (default: 10000)
        position_size : float, optional
            Fraction of capital to risk per trade (default: 0.1 = 10%)
            
        Returns:
        --------
        dict
            Dictionary of results for each measure
        """
        results = {}
        
        # Test each measure
        for measure_name in self.measures.keys():
            print(f"\nTesting {measure_name} for {symbol}...")
            try:
                result_df = self.backtest_strategy(
                    symbol=symbol,
                    lookback_window=lookback_window,
                    pattern_length=pattern_length,
                    forecast_length=forecast_length,
                    measure_name=measure_name,
                    epsilon=epsilon,
                    initial_capital=initial_capital,
                    position_size=position_size
                )
                results[measure_name] = result_df
            except Exception as e:
                print(f"Error testing {measure_name}: {e}")
        
        return results
    
    def compare_measures(self, symbols=None):
        """
        Compare the performance of different measures across symbols
        
        Parameters:
        -----------
        symbols : list, optional
            List of symbols to compare (default: all loaded symbols)
            
        Returns:
        --------
        DataFrame
            Comparison of measures across symbols
        """
        if symbols is None:
            symbols = list(self.data.keys())  # Only use symbols for which we have data
        
        # Create comparison table
        comparison = []
        
        for symbol in symbols:
            for measure in self.measures.keys():
                result_key = f"{symbol}_{measure}"
                if result_key in self.results:
                    result = self.results[result_key]
                    
                    # Create row with basic metrics
                    row = {
                        'symbol': symbol,
                        'measure': measure,
                        'accuracy': result['accuracy'],
                    }
                    
                    # Add precision and recall metrics
                    for signal_type in ['BUY', 'SELL', 'HOLD']:
                        row[f'{signal_type}_precision'] = result['metrics'].get(f'{signal_type}_precision', 0)
                        row[f'{signal_type}_recall'] = result['metrics'].get(f'{signal_type}_recall', 0)
                    
                    # Add portfolio performance metrics
                    if 'equity_metrics' in result:
                        equity_metrics = result['equity_metrics']
                        row.update({
                            'total_return': equity_metrics.get('total_return', 0),
                            'annualized_return': equity_metrics.get('annualized_return', 0),
                            'sharpe_ratio': equity_metrics.get('sharpe_ratio', 0),
                            'max_drawdown': equity_metrics.get('max_drawdown', 0),
                            'win_rate': equity_metrics.get('win_rate', 0),
                            'num_trades': equity_metrics.get('num_trades', 0)
                        })
                    
                    comparison.append(row)
        
        # Convert to DataFrame
        comparison_df = pd.DataFrame(comparison)
        
        # Calculate average performance by measure
        if len(comparison_df) > 0:
            # Select columns to show in the summary
            summary_cols = ['measure', 'accuracy', 'total_return', 'annualized_return', 
                           'sharpe_ratio', 'max_drawdown', 'win_rate']
            
            # Filter for columns that exist
            available_cols = [col for col in summary_cols if col in comparison_df.columns]
            
            avg_by_measure = comparison_df.groupby('measure')[available_cols[1:]].mean().reset_index()
            print("\nAverage performance by measure:")
            print(avg_by_measure[available_cols])
            
            # Calculate average by category if we have categories
            if self.symbol_categories:
                print("\nPerformance by category:")
                for category, symbols_in_category in self.symbol_categories.items():
                    # Filter for symbols we actually have data for
                    available_symbols = [s for s in symbols_in_category if s in self.data]
                    if not available_symbols:
                        print(f"\n{category.upper()}: No data available")
                        continue
                        
                    category_df = comparison_df[comparison_df['symbol'].isin(available_symbols)]
                    if len(category_df) > 0:
                        avg_by_measure = category_df.groupby('measure')[available_cols[1:]].mean().reset_index()
                        print(f"\n{category.upper()}:")
                        print(avg_by_measure[available_cols])
        
        return comparison_df
    
    def plot_signals(self, symbol, measure_name, plot_signals=True, show_pattern=False):
        """
        Plot the results of the backtest with signals
        
        Parameters:
        -----------
        symbol : str
            Ticker symbol
        measure_name : str
            Similarity measure used
        plot_signals : bool, optional
            Whether to plot buy/sell signals (default: True)
        show_pattern : bool, optional
            Whether to show matched patterns (default: False)
        """
        result_key = f"{symbol}_{measure_name}"
        if result_key not in self.results:
            print(f"No results for {symbol} using {measure_name}")
            return
        
        # Get results
        results_df = self.results[result_key]['results_df']
        
        # Get price data
        price_series = self.data[symbol]
        
        # Create figure
        plt.figure(figsize=(12, 6))
        
        # Plot price
        plt.plot(price_series.index, price_series.values, label=symbol)
        
        # Plot signals if requested
        if plot_signals:
            # Buy signals (correct)
            buy_correct = results_df[(results_df['predicted_signal'] == 'BUY') & 
                                    (results_df['actual_signal'] == 'BUY')]
            if not buy_correct.empty:
                plt.scatter(buy_correct['date'], price_series.loc[buy_correct['date']], 
                           color='green', marker='^', s=100, label='Correct Buy')
            
            # Sell signals (correct)
            sell_correct = results_df[(results_df['predicted_signal'] == 'SELL') & 
                                     (results_df['actual_signal'] == 'SELL')]
            if not sell_correct.empty:
                plt.scatter(sell_correct['date'], price_series.loc[sell_correct['date']], 
                           color='red', marker='v', s=100, label='Correct Sell')
            
            # Buy signals (incorrect)
            buy_incorrect = results_df[(results_df['predicted_signal'] == 'BUY') & 
                                      (results_df['actual_signal'] != 'BUY')]
            if not buy_incorrect.empty:
                plt.scatter(buy_incorrect['date'], price_series.loc[buy_incorrect['date']], 
                           color='lightgreen', marker='^', s=50, label='Incorrect Buy')
            
            # Sell signals (incorrect)
            sell_incorrect = results_df[(results_df['predicted_signal'] == 'SELL') & 
                                       (results_df['actual_signal'] != 'SELL')]
            if not sell_incorrect.empty:
                plt.scatter(sell_incorrect['date'], price_series.loc[sell_incorrect['date']], 
                           color='lightcoral', marker='v', s=50, label='Incorrect Sell')
        
        # Add title and labels
        plt.title(f"{symbol} - Strategy Performance using {measure_name}")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Show the plot
        plt.tight_layout()
        plt.show()
        
        # Show accuracy metrics
        accuracy = self.results[result_key]['accuracy']
        metrics = self.results[result_key]['metrics']
        
        print(f"Accuracy: {accuracy:.4f}")
        print("Precision/Recall:")
        for signal_type in ['BUY', 'SELL', 'HOLD']:
            print(f"  {signal_type}: Precision={metrics[f'{signal_type}_precision']:.4f}, "
                  f"Recall={metrics[f'{signal_type}_recall']:.4f}")
    
    def plot_equity_curve(self, symbol, measure_name=None, benchmark_symbol=None):
        """
        Plot the equity curve for the backtest
        
        Parameters:
        -----------
        symbol : str
            Ticker symbol
        measure_name : str or list, optional
            Similarity measure(s) used. If None, plot all available measures.
        benchmark_symbol : str, optional
            Symbol to use as benchmark (e.g., 'SPX Index' for S&P 500)
        """
        # If measure_name is not specified, plot all available measures
        if measure_name is None:
            measure_names = []
            for key in self.results.keys():
                if key.startswith(f"{symbol}_"):
                    meas = key.replace(f"{symbol}_", "")
                    measure_names.append(meas)
        elif isinstance(measure_name, str):
            measure_names = [measure_name]
        else:
            measure_names = measure_name
        
        # Create figure
        plt.figure(figsize=(12, 6))
        
        # Plot equity curves for each measure
        for measure in measure_names:
            result_key = f"{symbol}_{measure}"
            if result_key in self.results and 'equity_curve' in self.results[result_key]:
                equity_curve = self.results[result_key]['equity_curve']
                
                if 'cumulative_returns' in equity_curve.columns:
                    plt.plot(equity_curve.index, equity_curve['cumulative_returns'], 
                             label=f"{symbol} - {measure}")
        
        # Plot benchmark if provided
        if benchmark_symbol and benchmark_symbol in self.data:
            # Calculate benchmark returns over the same period
            benchmark_price = self.data[benchmark_symbol]
            
            # Use the same date range as the equity curve
            earliest_date = None
            latest_date = None
            
            for measure in measure_names:
                result_key = f"{symbol}_{measure}"
                if result_key in self.results and 'equity_curve' in self.results[result_key]:
                    equity_curve = self.results[result_key]['equity_curve']
                    if earliest_date is None or equity_curve.index[0] < earliest_date:
                        earliest_date = equity_curve.index[0]
                    if latest_date is None or equity_curve.index[-1] > latest_date:
                        latest_date = equity_curve.index[-1]
            
            if earliest_date and latest_date:
                # Filter benchmark data to match the equity curve date range
                benchmark_filtered = benchmark_price[(benchmark_price.index >= earliest_date) & 
                                                   (benchmark_price.index <= latest_date)]
                
                # Calculate benchmark returns
                benchmark_returns = benchmark_filtered.pct_change().fillna(0)
                benchmark_cum_returns = (1 + benchmark_returns).cumprod() - 1
                
                # Plot benchmark
                plt.plot(benchmark_filtered.index, benchmark_cum_returns, 
                         label=f"{benchmark_symbol} (Benchmark)", linestyle='--', color='black')
        
        # Add title and labels
        if len(measure_names) == 1:
            plt.title(f"{symbol} - Equity Curve using {measure_names[0]}")
        else:
            plt.title(f"{symbol} - Equity Curves Comparison")
        
        plt.xlabel("Date")
        plt.ylabel("Cumulative Return")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add y-axis as percentage
        plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
        
        # Show the plot
        plt.tight_layout()
        plt.show()
        
        # Print performance metrics for each measure
        print("Performance Metrics:")
        for measure in measure_names:
            result_key = f"{symbol}_{measure}"
            if result_key in self.results and 'equity_metrics' in self.results[result_key]:
                metrics = self.results[result_key]['equity_metrics']
                print(f"\n{measure}:")
                print(f"  Total Return: {metrics['total_return']*100:.2f}%")
                print(f"  Annualized Return: {metrics['annualized_return']*100:.2f}%")
                print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
                print(f"  Max Drawdown: {metrics['max_drawdown']*100:.2f}%")
                print(f"  Win Rate: {metrics['win_rate']*100:.2f}%")
                print(f"  Number of Trades: {metrics['num_trades']}")
    
    def plot_drawdown(self, symbol, measure_name):
        """
        Plot the drawdown for the backtest
        
        Parameters:
        -----------
        symbol : str
            Ticker symbol
        measure_name : str
            Similarity measure used
        """
        result_key = f"{symbol}_{measure_name}"
        if result_key not in self.results or 'equity_curve' not in self.results[result_key]:
            print(f"No equity curve data for {symbol} using {measure_name}")
            return
        
        # Get equity curve
        equity_curve = self.results[result_key]['equity_curve']
        
        # Check if drawdown is calculated
        if 'drawdown' not in equity_curve.columns:
            # Calculate drawdown
            equity_curve['peak'] = equity_curve['value'].cummax()
            equity_curve['drawdown'] = (equity_curve['value'] / equity_curve['peak']) - 1
        
        # Create figure
        plt.figure(figsize=(12, 6))
        
        # Plot drawdown
        plt.fill_between(equity_curve.index, 0, equity_curve['drawdown'], color='red', alpha=0.3)
        plt.plot(equity_curve.index, equity_curve['drawdown'], color='red', label='Drawdown')
        
        # Add title and labels
        plt.title(f"{symbol} - Drawdown using {measure_name}")
        plt.xlabel("Date")
        plt.ylabel("Drawdown")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Format y-axis as percentage
        plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
        
        # Show the plot
        plt.tight_layout()
        plt.show()
        
        # Print drawdown metrics
        print("Drawdown Metrics:")
        metrics = self.results[result_key]['equity_metrics']
        print(f"  Max Drawdown: {metrics['max_drawdown']*100:.2f}%")
        print(f"  Calmar Ratio: {metrics['calmar_ratio']:.2f}")
    
    def summary_report(self, symbol):
        """
        Generate a summary report for all measures on a given symbol
        
        Parameters:
        -----------
        symbol : str
            Ticker symbol
        """
        # Check if we have data for this symbol
        if symbol not in self.data:
            print(f"No data available for {symbol}")
            return
        
        # Find all measures that have been tested on this symbol
        measures = []
        for key in self.results.keys():
            if key.startswith(f"{symbol}_"):
                measure = key.replace(f"{symbol}_", "")
                measures.append(measure)
        
        if not measures:
            print(f"No backtest results available for {symbol}")
            return
        
        print(f"===== SUMMARY REPORT FOR {symbol} =====")
        print(f"Data period: {self.data[symbol].index[0].strftime('%Y-%m-%d')} to {self.data[symbol].index[-1].strftime('%Y-%m-%d')}")
        print(f"Number of data points: {len(self.data[symbol])}")
        print(f"Measures tested: {', '.join(measures)}")
        print("\n")
        
        # Create summary table of metrics
        summary_data = []
        for measure in measures:
            result_key = f"{symbol}_{measure}"
            
            # Get signal accuracy metrics
            accuracy = self.results[result_key]['accuracy']
            metrics = self.results[result_key]['metrics']
            
            # Get portfolio performance metrics
            if 'equity_metrics' in self.results[result_key]:
                equity_metrics = self.results[result_key]['equity_metrics']
                
                summary_data.append({
                    'Measure': measure,
                    'Accuracy': accuracy,
                    'Buy Precision': metrics.get('BUY_precision', 0),
                    'Buy Recall': metrics.get('BUY_recall', 0),
                    'Sell Precision': metrics.get('SELL_precision', 0),
                    'Sell Recall': metrics.get('SELL_recall', 0),
                    'Total Return': equity_metrics.get('total_return', 0),
                    'Ann. Return': equity_metrics.get('annualized_return', 0),
                    'Sharpe Ratio': equity_metrics.get('sharpe_ratio', 0),
                    'Max Drawdown': equity_metrics.get('max_drawdown', 0),
                    'Win Rate': equity_metrics.get('win_rate', 0),
                    'Num Trades': equity_metrics.get('num_trades', 0)
                })
            else:
                summary_data.append({
                    'Measure': measure,
                    'Accuracy': accuracy,
                    'Buy Precision': metrics.get('BUY_precision', 0),
                    'Buy Recall': metrics.get('BUY_recall', 0),
                    'Sell Precision': metrics.get('SELL_precision', 0),
                    'Sell Recall': metrics.get('SELL_recall', 0)
                })
        
        # Convert to DataFrame and display
        summary_df = pd.DataFrame(summary_data)
        
        # Format percentage columns
        pct_columns = ['Accuracy', 'Buy Precision', 'Buy Recall', 'Sell Precision', 
                      'Sell Recall', 'Total Return', 'Ann. Return', 'Max Drawdown', 'Win Rate']
        
        for col in pct_columns:
            if col in summary_df.columns:
                summary_df[col] = summary_df[col].apply(lambda x: f"{x*100:.2f}%")
        
        # Display summary table
        print(summary_df.to_string(index=False))
        
        # Plot equity curves for all measures
        self.plot_equity_curve(symbol)


# Example usage
if __name__ == "__main__":
    # Define symbols to test (using Bloomberg tickers)
    # Using fewer symbols for faster testing
    symbols = {
        'currency_pairs': [
            'EURUSD Curncy',
            'GBPUSD Curncy',
            'USDJPY Curncy'
        ],
        'stocks': [
            'AAPL US Equity',
            'MSFT US Equity',
            'AMZN US Equity'
        ],
        'indices': [
            'SPX Index',       # S&P 500
            'DAX Index',       # DAX
            'UKX Index'        # FTSE 100
        ]
    }
    
    # Initialize the strategy with Bloomberg data
    # Use a shorter date range to reduce the amount of data needed
    strategy = TradingStrategy(symbols=symbols, start_date='2015-01-01', end_date='2022-01-01', use_bloomberg=True)
    
    # Test one symbol from each category
    test_symbols = ['EURUSD Curncy', 'AAPL US Equity', 'SPX Index']
    
    # Run backtests with small lookback window for faster testing
    for symbol in [s for s in test_symbols if s in strategy.data]:
        # Test only essential measures: W measure, DTW, and Euclidean
        for measure in ['w_measure', 'dtw', 'euclidean']:
            try:
                strategy.backtest_strategy(
                    symbol=symbol,
                    lookback_window=500,  # Reduced for faster testing
                    pattern_length=10,
                    forecast_length=5,
                    measure_name=measure,
                    initial_capital=10000,
                    position_size=0.1
                )
            except Exception as e:
                print(f"Error backtesting {symbol} with {measure}: {e}")
    
    # Compare results across measures
    comparison = strategy.compare_measures()
    
    # Generate equity curves and performance reports
    for symbol in [s for s in test_symbols if s in strategy.data]:
        try:
            # Plot equity curves for all measures
            strategy.plot_equity_curve(symbol)
            
            # Plot drawdown for W measure
            strategy.plot_drawdown(symbol, 'w_measure')
            
            # Generate summary report
            strategy.summary_report(symbol)
        except Exception as e:
            print(f"Error generating reports for {symbol}: {e}")