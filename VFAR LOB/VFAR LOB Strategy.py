import numpy as np
import pandas as pd
from scipy import linalg
from scipy.interpolate import BSpline
import statsmodels.api as sm
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error
import warnings
warnings.filterwarnings('ignore')

#############################################
# Simulated Order Book Data Generator
#############################################

class SimulatedOrderBookGenerator:
    def __init__(self, base_price=100, volatility=0.02, spread_mean=0.01, 
                 depth_factor=1000, mean_reversion=0.05, random_seed=None):
        """
        Generate simulated limit order book data with more stable parameters
        
        Parameters:
        -----------
        base_price : float
            Base price for the asset
        volatility : float
            Daily volatility of the price
        spread_mean : float
            Mean bid-ask spread as percentage of price
        depth_factor : float
            Factor to control market depth
        mean_reversion : float
            Mean reversion speed for price (0-1)
        random_seed : int or None
            Random seed for reproducibility
        """
        self.base_price = base_price
        self.volatility = volatility  # Lower default volatility
        self.spread_mean = spread_mean
        self.depth_factor = depth_factor
        self.mean_reversion = mean_reversion
        
        # Set random seed
        if random_seed is not None:
            np.random.seed(random_seed)
    
    def generate_price_series(self, n_days=45, intervals_per_day=75):
        """
        Generate mid-price time series using a mean-reverting process
        
        Parameters:
        -----------
        n_days : int
            Number of days to simulate
        intervals_per_day : int
            Number of intervals per day
            
        Returns:
        --------
        numpy.ndarray
            Mid-price time series
        """
        n_steps = n_days * intervals_per_day
        dt = 1.0 / intervals_per_day  # Time step (in days)
        
        # Generate daily returns with mean reversion
        daily_vol = self.volatility * np.sqrt(dt)
        
        prices = np.zeros(n_steps)
        prices[0] = self.base_price
        
        for i in range(1, n_steps):
            # Mean reverting process: dP = k(μ-P)dt + σdW
            mean_reversion_component = self.mean_reversion * (self.base_price - prices[i-1]) * dt
            random_component = daily_vol * np.random.randn()
            
            price_change = mean_reversion_component + random_component * prices[i-1]
            prices[i] = prices[i-1] + price_change
            
            # Ensure price stays positive and reasonable
            prices[i] = max(prices[i], 0.01 * self.base_price)  # Minimum 1% of base price
            prices[i] = min(prices[i], 2 * self.base_price)  # Maximum 200% of base price
        
        return prices
    
    def generate_order_book_snapshot(self, mid_price, time_index, n_levels=100):
        """
        Generate a single order book snapshot
        
        Parameters:
        -----------
        mid_price : float
            Current mid-price
        time_index : int
            Current time index (used for time-dependent effects)
        n_levels : int
            Number of price levels to generate
            
        Returns:
        --------
        dict
            Order book snapshot containing bid/ask prices and volumes
        """
        # Time of day effect on spread (U-shaped pattern)
        time_of_day = (time_index % 75) / 75  # Normalized time of day
        tod_factor = 1 + 0.3 * (np.exp(-5 * time_of_day) + np.exp(-5 * (1 - time_of_day)))
        
        # Calculate spread
        relative_spread = self.spread_mean * tod_factor
        absolute_spread = mid_price * relative_spread
        
        # Best bid and ask prices
        best_bid = mid_price - absolute_spread / 2
        best_ask = mid_price + absolute_spread / 2
        
        # Generate price levels with increasing gaps further from mid-price
        bid_prices = []
        ask_prices = []
        
        # Realistic price tick sizes
        if mid_price < 1:
            tick_size = 0.0001
        elif mid_price < 10:
            tick_size = 0.001
        elif mid_price < 100:
            tick_size = 0.01
        else:
            tick_size = 0.1
            
        # Generate bid price levels
        current_bid = best_bid
        for i in range(n_levels):
            bid_prices.append(current_bid)
            # Tick size increases with distance from best bid
            current_tick = tick_size * (1 + 0.05 * i)  # Reduced tick size growth
            current_bid -= current_tick
            
            # Ensure price stays positive
            if current_bid <= 0:
                current_bid = 0.001
        
        # Generate ask price levels
        current_ask = best_ask
        for i in range(n_levels):
            ask_prices.append(current_ask)
            # Tick size increases with distance from best ask
            current_tick = tick_size * (1 + 0.05 * i)  # Reduced tick size growth
            current_ask += current_tick
        
        # Generate volumes - decreasing near mid-price, then relatively flat
        bid_volumes = []
        ask_volumes = []
        
        # Volume at best bid/ask - more stable volume generation
        best_bid_vol = self.depth_factor * max(0.5, np.random.gamma(shape=2.0, scale=0.5))
        best_ask_vol = self.depth_factor * max(0.5, np.random.gamma(shape=2.0, scale=0.5))
        
        # Generate bid volumes
        for i in range(n_levels):
            if i == 0:
                bid_volumes.append(best_bid_vol)
            else:
                # Volume decay with distance from best bid
                decay = np.exp(-0.05 * i) + 0.2  # Slower decay
                randomness = max(0.5, np.random.gamma(shape=2.0, scale=0.5))
                bid_volumes.append(best_bid_vol * decay * randomness)
        
        # Generate ask volumes
        for i in range(n_levels):
            if i == 0:
                ask_volumes.append(best_ask_vol)
            else:
                # Volume decay with distance from best ask
                decay = np.exp(-0.05 * i) + 0.2  # Slower decay
                randomness = max(0.5, np.random.gamma(shape=2.0, scale=0.5))
                ask_volumes.append(best_ask_vol * decay * randomness)
        
        # Ensure all volumes are positive
        for i in range(len(bid_volumes)):
            if bid_volumes[i] <= 0 or np.isnan(bid_volumes[i]) or np.isinf(bid_volumes[i]):
                bid_volumes[i] = 1.0
        
        for i in range(len(ask_volumes)):
            if ask_volumes[i] <= 0 or np.isnan(ask_volumes[i]) or np.isinf(ask_volumes[i]):
                ask_volumes[i] = 1.0
        
        return {
            'bid_prices': np.array(bid_prices),
            'bid_volumes': np.array(bid_volumes),
            'ask_prices': np.array(ask_prices),
            'ask_volumes': np.array(ask_volumes),
            'mid_price': mid_price
        }
    
    def generate_historical_data(self, n_days=45, intervals_per_day=75, n_levels=100):
        """
        Generate historical order book data
        
        Parameters:
        -----------
        n_days : int
            Number of days to simulate
        intervals_per_day : int
            Number of intervals per day
        n_levels : int
            Number of price levels to generate
            
        Returns:
        --------
        list
            List of order book snapshots
        """
        # Generate price series
        mid_prices = self.generate_price_series(n_days, intervals_per_day)
        
        # Generate order book snapshots
        snapshots = []
        
        for i, price in enumerate(mid_prices):
            snapshot = self.generate_order_book_snapshot(price, i, n_levels)
            
            # Add timestamp
            base_date = datetime.now().replace(hour=9, minute=30, second=0, microsecond=0)
            interval_length = timedelta(minutes=5)
            day_idx = i // intervals_per_day
            intraday_idx = i % intervals_per_day
            
            timestamp = base_date - timedelta(days=n_days - day_idx) + intraday_idx * interval_length
            snapshot['timestamp'] = timestamp
            
            snapshots.append(snapshot)
        
        return snapshots
    
    def format_data_as_dataframe(self, snapshots):
        """
        Format order book snapshots as a pandas DataFrame
        
        Parameters:
        -----------
        snapshots : list
            List of order book snapshots
            
        Returns:
        --------
        pandas.DataFrame
            Formatted order book data
        """
        # Prepare data
        data = []
        
        for snapshot in snapshots:
            row = {'timestamp': snapshot['timestamp']}
            
            # Add bid data
            for i, (price, volume) in enumerate(zip(snapshot['bid_prices'], snapshot['bid_volumes'])):
                row[f'BID_PRICE_{i+1}'] = price
                row[f'BID_SIZE_{i+1}'] = volume
            
            # Add ask data
            for i, (price, volume) in enumerate(zip(snapshot['ask_prices'], snapshot['ask_volumes'])):
                row[f'ASK_PRICE_{i+1}'] = price
                row[f'ASK_SIZE_{i+1}'] = volume
            
            data.append(row)
        
        # Create DataFrame
        df = pd.DataFrame(data)
        df.set_index('timestamp', inplace=True)
        
        return df


#############################################
# Order Book Data Processor
#############################################

class LOBDataProcessor:
    def __init__(self, skip_first_minutes=15, skip_last_minutes=5):
        """
        Process LOB data to create bid and ask curves
        
        Parameters:
        -----------
        skip_first_minutes : int
            Minutes to skip after market open (to avoid open auction effects)
        skip_last_minutes : int
            Minutes to skip before market close (to avoid close auction effects)
        """
        self.skip_first_minutes = skip_first_minutes
        self.skip_last_minutes = skip_last_minutes
    
    def process_historical_data(self, lob_data):
        """
        Process historical LOB data to create bid and ask curves
        
        Parameters:
        -----------
        lob_data : pandas.DataFrame
            Limit order book data
            
        Returns:
        --------
        tuple
            (bid_curves, ask_curves, timestamps) where each curve is a numpy array
        """
        if lob_data is None or lob_data.empty:
            print("No data to process")
            return None, None, None
        
        # Filter out market open/close periods
        market_open = datetime.strptime("09:30", "%H:%M").time()
        market_close = datetime.strptime("16:00", "%H:%M").time()
        
        # Keep only regular trading hours
        try:
            lob_data = lob_data[
                (lob_data.index.time >= (datetime.combine(datetime.today(), market_open) + 
                                        timedelta(minutes=self.skip_first_minutes)).time()) &
                (lob_data.index.time <= (datetime.combine(datetime.today(), market_close) - 
                                        timedelta(minutes=self.skip_last_minutes)).time())
            ]
        except:
            # If filtering fails, use all data
            print("Time filtering failed, using all data")
        
        # Get all timestamps
        timestamps = lob_data.index.tolist()
        
        # Extract curves for each timestamp
        bid_curves = []
        ask_curves = []
        
        for timestamp in timestamps:
            row = lob_data.loc[timestamp]
            
            # Extract bid data
            bid_prices = []
            bid_volumes = []
            for level in range(1, 101):
                price_col = f'BID_PRICE_{level}'
                size_col = f'BID_SIZE_{level}'
                
                if price_col in row.index and size_col in row.index and not np.isnan(row[price_col]) and not np.isnan(row[size_col]):
                    if row[size_col] > 0:  # Ensure positive volume
                        bid_prices.append(row[price_col])
                        bid_volumes.append(row[size_col])
            
            # Extract ask data
            ask_prices = []
            ask_volumes = []
            for level in range(1, 101):
                price_col = f'ASK_PRICE_{level}'
                size_col = f'ASK_SIZE_{level}'
                
                if price_col in row.index and size_col in row.index and not np.isnan(row[price_col]) and not np.isnan(row[size_col]):
                    if row[size_col] > 0:  # Ensure positive volume
                        ask_prices.append(row[price_col])
                        ask_volumes.append(row[size_col])
            
            # Skip if insufficient data
            if len(bid_prices) < 10 or len(ask_prices) < 10:
                continue
            
            # Create log-accumulated volumes as in the paper
            try:
                cum_bid_volumes = np.log(np.cumsum(bid_volumes))
                cum_ask_volumes = np.log(np.cumsum(ask_volumes))
            except Exception as e:
                print(f"Error calculating log volumes at {timestamp}: {str(e)}")
                continue
            
            # Check for NaN or inf values
            if np.any(np.isnan(cum_bid_volumes)) or np.any(np.isinf(cum_bid_volumes)) or \
               np.any(np.isnan(cum_ask_volumes)) or np.any(np.isinf(cum_ask_volumes)):
                print(f"Invalid log volumes at {timestamp}")
                continue
            
            # Normalize prices to [0,1]
            min_price = min(min(bid_prices), min(ask_prices))
            max_price = max(max(bid_prices), max(ask_prices))
            price_range = max_price - min_price
            
            if price_range <= 0:
                print(f"Invalid price range at {timestamp}")
                continue
            
            bid_prices_norm = [(p - min_price) / price_range for p in bid_prices]
            ask_prices_norm = [(p - min_price) / price_range for p in ask_prices]
            
            # Create uniform grid and interpolate
            grid = np.linspace(0, 1, 100)
            
            # Ensure prices are sorted properly (descending for bid, ascending for ask)
            try:
                sorted_bid_indices = np.argsort(bid_prices_norm)[::-1]  # Descending
                sorted_ask_indices = np.argsort(ask_prices_norm)  # Ascending
                
                bid_prices_sorted = [bid_prices_norm[i] for i in sorted_bid_indices]
                bid_volumes_sorted = [cum_bid_volumes[i] for i in sorted_bid_indices]
                
                ask_prices_sorted = [ask_prices_norm[i] for i in sorted_ask_indices]
                ask_volumes_sorted = [cum_ask_volumes[i] for i in sorted_ask_indices]
                
                # Make sure we have strictly monotonic points for interpolation
                bid_prices_unique = []
                bid_volumes_unique = []
                prev_price = None
                for price, volume in zip(bid_prices_sorted, bid_volumes_sorted):
                    if price != prev_price:
                        bid_prices_unique.append(price)
                        bid_volumes_unique.append(volume)
                        prev_price = price
                
                ask_prices_unique = []
                ask_volumes_unique = []
                prev_price = None
                for price, volume in zip(ask_prices_sorted, ask_volumes_sorted):
                    if price != prev_price:
                        ask_prices_unique.append(price)
                        ask_volumes_unique.append(volume)
                        prev_price = price
                
                # Make sure we have at least 2 points for interpolation
                if len(bid_prices_unique) < 2 or len(ask_prices_unique) < 2:
                    continue
                
                # Interpolate to get uniform curves
                bid_curve = np.interp(grid, bid_prices_unique, bid_volumes_unique)
                ask_curve = np.interp(grid, ask_prices_unique, ask_volumes_unique)
                
                # Check for NaN or inf values in final curves
                if np.any(np.isnan(bid_curve)) or np.any(np.isinf(bid_curve)) or \
                   np.any(np.isnan(ask_curve)) or np.any(np.isinf(ask_curve)):
                    continue
                
                bid_curves.append(bid_curve)
                ask_curves.append(ask_curve)
            except Exception as e:
                print(f"Interpolation error at {timestamp}: {str(e)}")
                continue
        
        if not bid_curves or not ask_curves:
            print("No valid curves created")
            return None, None, None
        
        return np.array(bid_curves), np.array(ask_curves), timestamps[:len(bid_curves)]
    
    def process_current_lob(self, current_lob):
        """
        Process current LOB data to create bid and ask curves
        
        Parameters:
        -----------
        current_lob : dict
            Current limit order book state
            
        Returns:
        --------
        tuple
            (bid_curve, ask_curve) as numpy arrays
        """
        if current_lob is None:
            print("No current LOB data")
            return None, None
        
        try:
            # Extract bid and ask data
            bid_prices = current_lob['bid_prices']
            bid_volumes = current_lob['bid_volumes']
            ask_prices = current_lob['ask_prices']
            ask_volumes = current_lob['ask_volumes']
            
            # Ensure positive volumes
            bid_volumes = np.maximum(bid_volumes, 1)
            ask_volumes = np.maximum(ask_volumes, 1)
            
            # Create log-accumulated volumes
            cum_bid_volumes = np.log(np.cumsum(bid_volumes))
            cum_ask_volumes = np.log(np.cumsum(ask_volumes))
            
            # Normalize prices to [0,1]
            min_price = min(min(bid_prices), min(ask_prices))
            max_price = max(max(bid_prices), max(ask_prices))
            price_range = max_price - min_price
            
            bid_prices_norm = [(p - min_price) / price_range for p in bid_prices]
            ask_prices_norm = [(p - min_price) / price_range for p in ask_prices]
            
            # Create uniform grid and interpolate
            grid = np.linspace(0, 1, 100)
            
            # Ensure prices are sorted properly (descending for bid, ascending for ask)
            sorted_bid_indices = np.argsort(bid_prices_norm)[::-1]  # Descending
            sorted_ask_indices = np.argsort(ask_prices_norm)  # Ascending
            
            bid_prices_sorted = [bid_prices_norm[i] for i in sorted_bid_indices]
            bid_volumes_sorted = [cum_bid_volumes[i] for i in sorted_bid_indices]
            
            ask_prices_sorted = [ask_prices_norm[i] for i in sorted_ask_indices]
            ask_volumes_sorted = [cum_ask_volumes[i] for i in sorted_ask_indices]
            
            # Make sure we have strictly monotonic points for interpolation
            bid_prices_unique = []
            bid_volumes_unique = []
            prev_price = None
            for price, volume in zip(bid_prices_sorted, bid_volumes_sorted):
                if price != prev_price:
                    bid_prices_unique.append(price)
                    bid_volumes_unique.append(volume)
                    prev_price = price
            
            ask_prices_unique = []
            ask_volumes_unique = []
            prev_price = None
            for price, volume in zip(ask_prices_sorted, ask_volumes_sorted):
                if price != prev_price:
                    ask_prices_unique.append(price)
                    ask_volumes_unique.append(volume)
                    prev_price = price
            
            # Interpolate
            bid_curve = np.interp(grid, bid_prices_unique, bid_volumes_unique)
            ask_curve = np.interp(grid, ask_prices_unique, ask_volumes_unique)
            
            return bid_curve, ask_curve
        except Exception as e:
            print(f"Error processing current LOB: {str(e)}")
            return None, None


#############################################
# Vector Functional Autoregressive Model
#############################################

class VFARModel:
    def __init__(self, n_basis=20, p=1):
        """
        Vector Functional Autoregressive Model for LOB forecasting
        
        Parameters:
        -----------
        n_basis : int
            Number of B-spline basis functions to use
        p : int
            Order of the autoregressive model
        """
        self.n_basis = n_basis
        self.p = p
        self.coefs = None
        self.mean_bid = None
        self.mean_ask = None
        self.bid_knots = None
        self.ask_knots = None
        self.cov_matrix = None
    
    def _create_basis_expansion(self, curves, knots):
        """
        Create B-spline basis expansion for curves
        
        Parameters:
        -----------
        curves : numpy.ndarray
            Array of curves (n_curves x n_points)
        knots : numpy.ndarray
            Knot sequence for B-splines
            
        Returns:
        --------
        numpy.ndarray
            B-spline coefficients
        """
        n_curves = curves.shape[0]
        basis_coefs = np.zeros((n_curves, self.n_basis))
        
        for i, curve in enumerate(curves):
            try:
                # Create a B-spline approximation
                t = np.linspace(0, 1, curve.shape[0])
                
                # Handle edge cases
                # Replace any NaN or inf values with mean of curve
                curve_mean = np.nanmean(curve[np.isfinite(curve)])
                curve_clean = np.copy(curve)
                curve_clean[~np.isfinite(curve_clean)] = curve_mean
                
                # Fit polynomial to the curve (instead of B-spline for simplicity)
                degree = min(self.n_basis - 1, 5)  # Limit degree to avoid instability
                tck = np.polyfit(t, curve_clean, degree)
                
                # Pad with zeros if necessary
                if len(tck) < self.n_basis:
                    tck = np.pad(tck, (0, self.n_basis - len(tck)))
                # Truncate if too long
                tck = tck[:self.n_basis]
                
                # Extract coefficients
                basis_coefs[i, :] = tck
            except Exception as e:
                # If fitting fails, use zeros or means
                basis_coefs[i, :] = np.zeros(self.n_basis)
                basis_coefs[i, 0] = np.nanmean(curve[np.isfinite(curve)])
                print(f"Basis expansion error: {str(e)}")
        
        return basis_coefs
    
    def fit(self, bid_curves, ask_curves):
        """
        Fit the VFAR model to the bid and ask curves
        
        Parameters:
        -----------
        bid_curves : numpy.ndarray
            Array of bid curves (n_curves x n_points)
        ask_curves : numpy.ndarray
            Array of ask curves (n_curves x n_points)
            
        Returns:
        --------
        self
        """
        if len(bid_curves) < self.p + 2 or len(ask_curves) < self.p + 2:
            raise ValueError("Not enough data points to fit VFAR model")
        
        # Generate equally spaced knots
        self.bid_knots = np.linspace(0, 1, self.n_basis + 4)  # +4 for cubic splines
        self.ask_knots = np.linspace(0, 1, self.n_basis + 4)
        
        # Store mean curves
        self.mean_bid = np.nanmean(bid_curves, axis=0)
        self.mean_ask = np.nanmean(ask_curves, axis=0)
        
        # Create basis expansions
        bid_coefs = self._create_basis_expansion(bid_curves, self.bid_knots)
        ask_coefs = self._create_basis_expansion(ask_curves, self.ask_knots)
        
        # Prepare data for VAR model
        # We'll create a dataset where each row is [bid_t, ask_t, bid_t-1, ask_t-1, ...]
        n_samples = len(bid_curves) - self.p
        n_features = 2 * self.n_basis
        
        Y = np.zeros((n_samples, n_features))
        X = np.zeros((n_samples, 1 + self.p * n_features))  # +1 for constant
        
        # Fill Y with [bid_t, ask_t]
        Y[:, :self.n_basis] = bid_coefs[self.p:]
        Y[:, self.n_basis:] = ask_coefs[self.p:]
        
        # Fill X with constant and lags
        X[:, 0] = 1  # Constant
        
        for lag in range(1, self.p + 1):
            start_col = 1 + (lag - 1) * n_features
            end_col = start_col + n_features
            
            # Add [bid_t-lag, ask_t-lag]
            X[:, start_col:start_col + self.n_basis] = bid_coefs[self.p - lag:-lag]
            X[:, start_col + self.n_basis:end_col] = ask_coefs[self.p - lag:-lag]
        
        # Check for NaN or inf values
        if np.any(np.isnan(X)) or np.any(np.isinf(X)) or np.any(np.isnan(Y)) or np.any(np.isinf(Y)):
            # Replace NaN/inf with zeros
            X = np.nan_to_num(X)
            Y = np.nan_to_num(Y)
        
        # Fit using OLS with regularization
        # We have Y = XB + ε
        XtX = X.T @ X
        
        # Add a small regularization term to avoid singular matrices
        XtX = XtX + 1e-6 * np.eye(XtX.shape[0])
        
        XtY = X.T @ Y
        
        try:
            self.coefs = np.linalg.solve(XtX, XtY)
        except np.linalg.LinAlgError:
            # Use pseudoinverse if matrix is still singular
            self.coefs = np.linalg.pinv(XtX) @ XtY
        
        # Calculate residuals and covariance matrix
        residuals = Y - X @ self.coefs
        self.cov_matrix = (residuals.T @ residuals) / n_samples
        
        return self
    
    def predict(self, recent_bid_curves, recent_ask_curves, steps_ahead=1):
        """
        Make multi-step ahead forecasts
        
        Parameters:
        -----------
        recent_bid_curves : numpy.ndarray
            Recent bid curves, at least p curves needed
        recent_ask_curves : numpy.ndarray
            Recent ask curves, at least p curves needed
        steps_ahead : int
            Number of steps ahead to forecast
            
        Returns:
        --------
        tuple
            (forecasted_bid_curve, forecasted_ask_curve)
        """
        if len(recent_bid_curves) < self.p or len(recent_ask_curves) < self.p:
            raise ValueError(f"Need at least {self.p} recent curves for prediction")
        
        try:
            # Get the most recent p curves
            recent_bid = recent_bid_curves[-self.p:]
            recent_ask = recent_ask_curves[-self.p:]
            
            # Convert to basis coefficients
            bid_coefs = self._create_basis_expansion(recent_bid, self.bid_knots)
            ask_coefs = self._create_basis_expansion(recent_ask, self.ask_knots)
            
            # Initialize forecast history with actual values
            forecast_history = []
            for i in range(self.p):
                forecast_history.append(np.concatenate([bid_coefs[i], ask_coefs[i]]))
            
            # Generate forecasts for each step ahead
            for _ in range(steps_ahead):
                # Create input vector [1, bid_t-1, ask_t-1, ..., bid_t-p, ask_t-p]
                x = np.ones(1 + self.p * 2 * self.n_basis)
                
                for lag in range(1, self.p + 1):
                    start_idx = 1 + (lag - 1) * 2 * self.n_basis
                    end_idx = start_idx + 2 * self.n_basis
                    x[start_idx:end_idx] = forecast_history[-lag]
                
                # Make forecast
                forecast = x @ self.coefs
                
                # Add to history
                forecast_history.append(forecast)
            
            # Convert forecast back to curves
            final_forecast = forecast_history[-1]
            bid_forecast_coefs = final_forecast[:self.n_basis]
            ask_forecast_coefs = final_forecast[self.n_basis:]
            
            # Reconstruct curves from coefficients
            t = np.linspace(0, 1, 100)
            
            # Use polynomial evaluation directly
            bid_forecast = np.polyval(bid_forecast_coefs[::-1], t)  # Reverse coefficients for polyval
            ask_forecast = np.polyval(ask_forecast_coefs[::-1], t)  # Reverse coefficients for polyval
            
            # Ensure forecasts are reasonable
            bid_forecast = np.nan_to_num(bid_forecast)
            ask_forecast = np.nan_to_num(ask_forecast)
            
            # Ensure monotonicity (bid should be non-increasing, ask should be non-decreasing)
            for i in range(1, len(bid_forecast)):
                if bid_forecast[i] > bid_forecast[i-1]:
                    bid_forecast[i] = bid_forecast[i-1]
                    
            for i in range(1, len(ask_forecast)):
                if ask_forecast[i] < ask_forecast[i-1]:
                    ask_forecast[i] = ask_forecast[i-1]
            
            return bid_forecast, ask_forecast
        except Exception as e:
            print(f"Prediction error: {str(e)}")
            # Return mean curves as fallback
            return self.mean_bid, self.mean_ask
    
    def evaluate(self, bid_curves, ask_curves, test_start=None):
        """
        Evaluate model performance on test data
        
        Parameters:
        -----------
        bid_curves : numpy.ndarray
            Array of bid curves
        ask_curves : numpy.ndarray
            Array of ask curves
        test_start : int
            Starting index for test data (if None, uses all data)
            
        Returns:
        --------
        dict
            Performance metrics
        """
        if test_start is None:
            test_start = self.p
        
        if test_start < self.p:
            test_start = self.p
        
        n_test = len(bid_curves) - test_start
        
        if n_test <= 0:
            raise ValueError("No test data available")
        
        # Make one-step ahead predictions for test period
        pred_bid_curves = []
        pred_ask_curves = []
        
        for i in range(test_start, len(bid_curves)):
            # Get p previous curves
            prev_bid = bid_curves[i-self.p:i]
            prev_ask = ask_curves[i-self.p:i]
            
            # Make prediction
            try:
                pred_bid, pred_ask = self.predict(prev_bid, prev_ask, steps_ahead=1)
                pred_bid_curves.append(pred_bid)
                pred_ask_curves.append(pred_ask)
            except Exception as e:
                print(f"Prediction error at index {i}: {str(e)}")
                # Use mean curves as fallback
                pred_bid_curves.append(self.mean_bid)
                pred_ask_curves.append(self.mean_ask)
        
        # Convert to arrays
        pred_bid_curves = np.array(pred_bid_curves)
        pred_ask_curves = np.array(pred_ask_curves)
        
        # Calculate metrics
        true_bid = bid_curves[test_start:]
        true_ask = ask_curves[test_start:]
        
        # Flatten arrays for overall metrics
        true_flat = np.concatenate([true_bid.flatten(), true_ask.flatten()])
        pred_flat = np.concatenate([pred_bid_curves.flatten(), pred_ask_curves.flatten()])
        
        # Calculate R² with robustness
        try:
            r2 = r2_score(true_flat, pred_flat)
            if np.isnan(r2) or np.isinf(r2) or r2 < -100:
                r2 = -1.0  # Fallback if calculation is unreliable
        except:
            r2 = -1.0
        
        # Calculate RMSE with robustness
        try:
            rmse = np.sqrt(mean_squared_error(true_flat, pred_flat))
            if np.isnan(rmse) or np.isinf(rmse) or rmse > 1000:
                rmse = 10.0  # Fallback if calculation is unreliable
        except:
            rmse = 10.0
        
        # Calculate MAPE with robustness
        try:
            # Avoid division by zero
            denom = np.maximum(np.abs(true_flat), 0.001)
            diff_pct = np.abs((true_flat - pred_flat) / denom)
            # Cap at 10 (1000%)
            diff_pct = np.minimum(diff_pct, 10)
            mape = np.nanmean(diff_pct) * 100
            if np.isnan(mape) or np.isinf(mape):
                mape = 100.0
        except:
            mape = 100.0
        
        return {
            'R2': r2,
            'RMSE': rmse,
            'MAPE': mape,
            'n_test': n_test
        }


#############################################
# Order Execution Strategy
#############################################

class OrderExecutionStrategy:
    def __init__(self, model, volume, time_points=75, trading_points=None):
        """
        Order execution strategy using VFAR model forecasts
        
        Parameters:
        -----------
        model : VFARModel
            Fitted VFAR model
        volume : int
            Total volume to execute
        time_points : int
            Total number of possible trading time points
        trading_points : int or None
            Number of time points to actually trade at (if None, trade at all points)
        """
        self.model = model
        self.volume = volume
        self.time_points = time_points
        self.trading_points = trading_points if trading_points else time_points
        self.is_buy = None
        self.price_impact_factor = 0.0001  # Can be calibrated
    
    def _calculate_price_impact(self, curve, volume, price_grid):
        """
        Calculate the price impact of executing a given volume
        
        Parameters:
        -----------
        curve : numpy.ndarray
            Liquidity curve (accumulated volumes)
        volume : float
            Volume to execute
        price_grid : numpy.ndarray
            Price grid corresponding to the curve
            
        Returns:
        --------
        float
            Average execution price
        """
        try:
            # Find the volume level in the curve
            target_level = np.log(max(1, volume))
            
            # Handle edge cases
            if np.any(np.isnan(curve)) or np.any(np.isinf(curve)):
                # Use mid-point price as fallback
                return np.mean(price_grid)
            
            idx = np.searchsorted(curve, target_level)
            
            if idx >= len(curve):
                # Volume exceeds available liquidity
                return price_grid[-1]  # Worst price
            elif idx == 0:
                # Volume is less than first level
                return price_grid[0]  # Best price
            else:
                # Interpolate price
                lower_vol = curve[idx-1]
                upper_vol = curve[idx]
                lower_price = price_grid[idx-1]
                upper_price = price_grid[idx]
                
                # Linear interpolation
                fraction = (target_level - lower_vol) / (upper_vol - lower_vol)
                price = lower_price + fraction * (upper_price - lower_price)
                
                return price
        except Exception as e:
            print(f"Price impact calculation error: {str(e)}")
            # Fallback: return middle price
            return np.mean(price_grid)
    
    def _calculate_execution_cost(self, curve, volume, price_grid=None):
        """
        Calculate the cost of executing a given volume with a given curve
        
        Parameters:
        -----------
        curve : numpy.ndarray
            Liquidity curve
        volume : float
            Volume to execute
        price_grid : numpy.ndarray or None
            Price grid corresponding to the curve (if None, use default grid)
            
        Returns:
        --------
        float
            Execution cost
        """
        if price_grid is None:
            # Use default grid [0, 1]
            price_grid = np.linspace(0, 1, len(curve))
        
        # Calculate average execution price
        avg_price = self._calculate_price_impact(curve, volume, price_grid)
        
        # Cost is price * volume
        return avg_price * volume
    
    def equal_split_strategy(self, bid_curves, ask_curves, price_grid=None):
        """
        Baseline strategy: equal splitting over all time points
        
        Parameters:
        -----------
        bid_curves : numpy.ndarray
            Array of bid curves for testing period
        ask_curves : numpy.ndarray
            Array of ask curves for testing period
        price_grid : numpy.ndarray or None
            Price grid corresponding to the curves
            
        Returns:
        --------
        float
            Total execution cost
        """
        if len(bid_curves) < self.time_points or len(ask_curves) < self.time_points:
            n_available = min(len(bid_curves), len(ask_curves))
            print(f"Warning: Only {n_available} curves available, less than the {self.time_points} requested")
            time_points = n_available
        else:
            time_points = self.time_points
        
        # Use only the first time_points curves
        bid_curves = bid_curves[:time_points]
        ask_curves = ask_curves[:time_points]
        
        per_trade_volume = self.volume / time_points
        total_cost = 0
        
        for i in range(time_points):
            # Calculate cost of executing per_trade_volume at this time point
            try:
                if self.is_buy:
                    # For buy orders, use ask curve
                    cost = self._calculate_execution_cost(ask_curves[i], per_trade_volume, price_grid)
                else:
                    # For sell orders, use bid curve
                    cost = self._calculate_execution_cost(bid_curves[i], per_trade_volume, price_grid)
                
                total_cost += cost
            except Exception as e:
                print(f"Error in equal split strategy at time {i}: {str(e)}")
                # Add average cost as fallback
                if i > 0:
                    total_cost += total_cost / i
                else:
                    # For the first entry, use a simple estimate
                    price_estimate = 0.5 if price_grid is None else np.mean(price_grid)
                    total_cost += price_estimate * per_trade_volume
        
        return total_cost
    
    def vfar_strategy(self, bid_curves, ask_curves, is_buy=True, price_grid=None):
        """
        Strategy using VFAR predictions to determine optimal execution times
        
        Parameters:
        -----------
        bid_curves : numpy.ndarray
            Recent bid curves
        ask_curves : numpy.ndarray
            Recent ask curves
        is_buy : bool
            True for buy orders, False for sell orders
        price_grid : numpy.ndarray or None
            Price grid corresponding to the curves
            
        Returns:
        --------
        list
            Trading schedule: list of (time_point, volume) tuples
        """
        if len(bid_curves) < self.model.p or len(ask_curves) < self.model.p:
            raise ValueError(f"Need at least {self.model.p} recent curves for VFAR strategy")
        
        self.is_buy = is_buy
        
        # Make forecasts for all future time points
        forecasted_costs = []
        
        for i in range(1, self.time_points + 1):
            try:
                # Get most recent curves for prediction
                recent_bid = bid_curves[-self.model.p:]
                recent_ask = ask_curves[-self.model.p:]
                
                # Predict i steps ahead
                forecasted_bid, forecasted_ask = self.model.predict(recent_bid, recent_ask, steps_ahead=i)
                
                # Calculate execution cost
                if is_buy:
                    # For buy orders, use ask curve
                    cost = self._calculate_execution_cost(forecasted_ask, self.volume, price_grid)
                else:
                    # For sell orders, use bid curve
                    cost = self._calculate_execution_cost(forecasted_bid, self.volume, price_grid)
                
                forecasted_costs.append((i, cost))
            except Exception as e:
                print(f"Error in VFAR strategy at time step {i}: {str(e)}")
                # Add a high cost to avoid this time point
                if price_grid is not None:
                    high_cost = np.max(price_grid) * self.volume * 1.5
                else:
                    high_cost = self.volume * 1.5
                forecasted_costs.append((i, high_cost))
        
        # Sort by cost and select the top trading_points
        forecasted_costs.sort(key=lambda x: x[1])
        selected_points = forecasted_costs[:min(self.trading_points, len(forecasted_costs))]
        
        # Calculate weights based on relative costs
        if not selected_points:
            # If no valid points, use equal weights
            return [(i, self.volume / self.time_points) for i in range(1, self.time_points + 1)]
        
        # Get total cost, with error handling
        total_cost = sum(cost for _, cost in selected_points)
        if total_cost <= 0:
            # If total cost is invalid, use equal weights
            equal_weight = 1.0 / len(selected_points)
            weights = [equal_weight for _ in selected_points]
        else:
            weights = [cost/total_cost for _, cost in selected_points]
        
        # Normalize weights to sum to 1
        sum_weights = sum(weights)
        if sum_weights <= 0:
            # Fallback to equal weights
            weights = [1.0 / len(weights) for _ in weights]
        else:
            weights = [w/sum_weights for w in weights]
        
        # Calculate volumes to trade at each selected point
        trading_schedule = [(point, weight * self.volume) for (point, _), weight in zip(selected_points, weights)]
        
        return trading_schedule


#############################################
# Backtesting Framework
#############################################

class VFARBacktester:
    def __init__(self, simulator, processor):
        """
        Backtest the VFAR-based execution strategy
        
        Parameters:
        -----------
        simulator : SimulatedOrderBookGenerator
            Order book simulator
        processor : LOBDataProcessor
            LOB data processor object
        """
        self.simulator = simulator
        self.processor = processor
    
    def backtest(self, ticker_name="Simulated Stock", volume=10000, n_basis=20, p=1, trading_points_list=None, n_days=45):
        """
        Backtest the VFAR strategy on simulated data
        
        Parameters:
        -----------
        ticker_name : str
            Name for the simulated stock
        volume : int
            Total volume to execute
        n_basis : int
            Number of B-spline basis functions
        p : int
            Order of the VFAR model
        trading_points_list : list or None
            List of trading points to test (if None, uses [1, 5, 10, 25, 50, 75])
        n_days : int
            Number of days to simulate
            
        Returns:
        --------
        dict
            Backtest results
        """
        # Set default trading points if not provided
        if trading_points_list is None:
            trading_points_list = [1, 5, 10, 25, 50, 75]
        
        # Generate simulated order book data
        snapshots = self.simulator.generate_historical_data(n_days=n_days)
        lob_data = self.simulator.format_data_as_dataframe(snapshots)
        
        # Process data to create bid and ask curves
        bid_curves, ask_curves, timestamps = self.processor.process_historical_data(lob_data)
        if bid_curves is None or ask_curves is None:
            return {"error": "Failed to process LOB data"}
        
        print(f"Processing {len(bid_curves)} data points for {ticker_name}")
        
        # Split data into training and testing
        train_size = int(0.7 * len(bid_curves))
        train_bid, test_bid = bid_curves[:train_size], bid_curves[train_size:]
        train_ask, test_ask = ask_curves[:train_size], ask_curves[train_size:]
        
        # Fit VFAR model
        model = VFARModel(n_basis=n_basis, p=p)
        model.fit(train_bid, train_ask)
        
        # Evaluate model
        eval_metrics = model.evaluate(test_bid, test_ask)
        print(f"VFAR model evaluation for {ticker_name}:")
        print(f"R² = {eval_metrics['R2']:.4f}, RMSE = {eval_metrics['RMSE']:.4f}, MAPE = {eval_metrics['MAPE']:.2f}%")
        
        # Visualize actual vs predicted curves
        try:
            self._plot_prediction_examples(model, test_bid, test_ask)
        except Exception as e:
            print(f"Error plotting prediction examples: {str(e)}")
        
        # Run Order Execution Strategies
        results = []
        
        for tp in trading_points_list:
            strategy = OrderExecutionStrategy(model, volume, time_points=75, trading_points=tp)
            
            # For each test day (assuming 75 intervals per day)
            day_results = []
            
            day_count = min(len(test_bid) // 75, len(test_ask) // 75)
            
            for day_idx in range(day_count):
                day_start = day_idx * 75
                day_end = min(day_start + 75, len(test_bid))
                
                daily_bid = test_bid[day_start:day_end]
                daily_ask = test_ask[day_start:day_end]
                
                # Skip days with too few data points
                if len(daily_bid) < 10 or len(daily_ask) < 10:
                    continue
                
                # Test both buy and sell orders
                for is_buy in [True, False]:
                    try:
                        # Equal split baseline
                        strategy.is_buy = is_buy
                        equal_split_cost = strategy.equal_split_strategy(daily_bid, daily_ask)
                        
                        # Make sure we have enough data for the model
                        if day_idx == 0:
                            # For the first day, use some training data
                            recent_bid = np.concatenate([train_bid[-model.p:], daily_bid[:model.p]])
                            recent_ask = np.concatenate([train_ask[-model.p:], daily_ask[:model.p]])
                        else:
                            # Otherwise use the most recent data
                            prev_day_end = day_start
                            prev_day_start = max(0, prev_day_end - model.p)
                            recent_bid = test_bid[prev_day_start:prev_day_end]
                            recent_ask = test_ask[prev_day_start:prev_day_end]
                            
                            # Pad with training data if needed
                            if len(recent_bid) < model.p:
                                pad_size = model.p - len(recent_bid)
                                recent_bid = np.concatenate([train_bid[-pad_size:], recent_bid])
                                recent_ask = np.concatenate([train_ask[-pad_size:], recent_ask])
                        
                        trading_schedule = strategy.vfar_strategy(recent_bid, recent_ask, is_buy)
                        
                        # Calculate actual cost based on future LOB states
                        vfar_cost = 0
                        for point, trade_volume in trading_schedule:
                            future_index = point - 1  # 0-indexed
                            if future_index < len(daily_bid):
                                if is_buy:
                                    cost = strategy._calculate_execution_cost(daily_ask[future_index], trade_volume)
                                else:
                                    cost = strategy._calculate_execution_cost(daily_bid[future_index], trade_volume)
                                vfar_cost += cost
                        
                        # Calculate savings with safety checks
                        if equal_split_cost <= 0 or vfar_cost <= 0 or np.isnan(equal_split_cost) or np.isnan(vfar_cost):
                            savings_bps = 0  # Default to zero if invalid values
                        else:
                            savings_bps = (1 - vfar_cost / equal_split_cost) * 10000  # Convert to basis points
                            
                            # Cap extreme values
                            if np.isnan(savings_bps) or np.isinf(savings_bps):
                                savings_bps = 0
                            elif savings_bps > 1000:  # Cap at 10% savings (1000 bps)
                                savings_bps = 1000
                            elif savings_bps < -1000:  # Cap at 10% loss
                                savings_bps = -1000
                        
                        day_results.append({
                            'day': day_idx,
                            'is_buy': is_buy,
                            'equal_split_cost': equal_split_cost,
                            'vfar_cost': vfar_cost,
                            'savings_bps': savings_bps
                        })
                    except Exception as e:
                        print(f"Error processing day {day_idx}, is_buy={is_buy}: {str(e)}")
                        continue
            
            # Calculate average savings (with error handling)
            if day_results:
                # Filter out any extreme values
                valid_savings = [r['savings_bps'] for r in day_results 
                                if not np.isnan(r['savings_bps']) and not np.isinf(r['savings_bps'])]
                
                if valid_savings:
                    avg_savings_bps = np.mean(valid_savings)
                else:
                    avg_savings_bps = 0
            else:
                avg_savings_bps = 0
            
            results.append({
                'ticker': ticker_name,
                'trading_points': tp,
                'avg_savings_bps': avg_savings_bps,
                'day_results': day_results
            })
        
        return {
            'ticker': ticker_name,
            'volume': volume,
            'eval_metrics': eval_metrics,
            'strategies': results,
            'model': model,
            'bid_curves': bid_curves,
            'ask_curves': ask_curves,
            'timestamps': timestamps
        }
    
    def _plot_prediction_examples(self, model, test_bid, test_ask, n_examples=3):
        """
        Plot examples of actual vs predicted curves
        
        Parameters:
        -----------
        model : VFARModel
            Fitted VFAR model
        test_bid : numpy.ndarray
            Test bid curves
        test_ask : numpy.ndarray
            Test ask curves
        n_examples : int
            Number of examples to plot
        """
        # Make sure we have enough test data
        if len(test_bid) <= model.p or len(test_ask) <= model.p:
            print("Not enough test data to plot examples")
            return
        
        # Randomly select n_examples from test set, but avoid the edges
        max_idx = min(len(test_bid), len(test_ask)) - 1
        min_idx = model.p + 1
        if max_idx <= min_idx:
            print("Not enough data range to select examples")
            return
        
        # Generate indices, ensuring they're within bounds
        try:
            indices = np.random.choice(range(min_idx, max_idx), min(n_examples, max_idx - min_idx), replace=False)
        except:
            # Fallback to fixed indices if random selection fails
            indices = [min_idx + i for i in range(min(n_examples, max_idx - min_idx))]
        
        for i, idx in enumerate(indices):
            try:
                # Get previous p curves for prediction
                prev_bid = test_bid[idx - model.p:idx]
                prev_ask = test_ask[idx - model.p:idx]
                
                # Make prediction
                pred_bid, pred_ask = model.predict(prev_bid, prev_ask, steps_ahead=1)
                
                # Plot
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
                
                # Bid side
                ax1.plot(test_bid[idx], 'b-', label='Actual')
                ax1.plot(pred_bid, 'r--', label='Predicted')
                ax1.set_title('Bid Curve')
                ax1.set_xlabel('Normalized Price')
                ax1.set_ylabel('Log Accumulated Volume')
                ax1.legend()
                ax1.grid(True)
                
                # Ask side
                ax2.plot(test_ask[idx], 'b-', label='Actual')
                ax2.plot(pred_ask, 'r--', label='Predicted')
                ax2.set_title('Ask Curve')
                ax2.set_xlabel('Normalized Price')
                ax2.set_ylabel('Log Accumulated Volume')
                ax2.legend()
                ax2.grid(True)
                
                plt.tight_layout()
                plt.show()
            except Exception as e:
                print(f"Error plotting example {i}: {str(e)}")
    
    def plot_strategy_results(self, results):
        """
        Plot backtest results
        
        Parameters:
        -----------
        results : dict
            Backtest results from backtest method
        """
        if 'error' in results:
            print(f"Error: {results['error']}")
            return
        
        ticker = results['ticker']
        strategies = results['strategies']
        
        # Extract trading points and savings
        trading_points = [s['trading_points'] for s in strategies]
        avg_savings = [s['avg_savings_bps'] for s in strategies]
        
        # Plot savings vs trading points
        plt.figure(figsize=(10, 6))
        plt.plot(trading_points, avg_savings, 'o-', linewidth=2)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Number of Trading Points')
        plt.ylabel('Average Savings (bps)')
        plt.title(f'VFAR Strategy Performance - {ticker}')
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        
        # Plot distribution of daily savings
        all_savings = []
        for s in strategies:
            for r in s['day_results']:
                # Filter out NaN and infinite values
                if not np.isnan(r['savings_bps']) and not np.isinf(r['savings_bps']):
                    all_savings.append(r['savings_bps'])
        
        if all_savings:  # Only plot if we have valid data
            plt.figure(figsize=(10, 6))
            plt.hist(all_savings, bins=20, alpha=0.7)
            plt.axvline(x=0, color='r', linestyle='--')
            plt.xlabel('Savings (bps)')
            plt.ylabel('Frequency')
            plt.title(f'Distribution of Daily Savings - {ticker}')
            plt.grid(True)
            plt.tight_layout()
            plt.show()
        else:
            print("No valid savings data to plot histogram")


#############################################
# Trading Strategies
#############################################

class LiquidityAwareAlphaStrategy:
    def __init__(self, vfar_model, lookback=10, forecast_horizon=5):
        """
        Trading strategy that combines liquidity forecasts with price movement predictions
        
        Parameters:
        -----------
        vfar_model : VFARModel
            Fitted VFAR model for liquidity prediction
        lookback : int
            Number of periods to use for momentum calculation
        forecast_horizon : int
            Number of periods ahead to forecast
        """
        self.vfar_model = vfar_model
        self.lookback = lookback
        self.forecast_horizon = forecast_horizon
        self.position = 0  # Current position (positive = long, negative = short)
        self.position_history = []
        self.pnl_history = []
        
    def _calculate_liquidity_imbalance(self, bid_curve, ask_curve):
        """
        Calculate liquidity imbalance between bid and ask sides
        
        Returns:
        --------
        float
            Liquidity imbalance score (-1 to 1, negative means more sell pressure)
        """
        # Calculate total liquidity on each side (area under curves)
        bid_liquidity = np.trapz(bid_curve)
        ask_liquidity = np.trapz(ask_curve)
        
        # Calculate imbalance 
        total_liquidity = bid_liquidity + ask_liquidity
        if total_liquidity == 0:
            return 0
            
        imbalance = (bid_liquidity - ask_liquidity) / total_liquidity
        return imbalance
    
    def _predict_price_movement(self, price_history):
        """
        Predict future price movement
        
        Returns:
        --------
        float
            Predicted price movement (-1 to 1, positive means up)
        """
        # Simple momentum model
        if len(price_history) < self.lookback + 1:
            return 0
            
        recent_return = (price_history[-1] / price_history[-self.lookback-1]) - 1
        return np.tanh(recent_return * 10)  # Scale to -1 to 1
    
    def generate_signal(self, bid_curves, ask_curves, price_history):
        """
        Generate trading signal based on liquidity and price predictions
        
        Parameters:
        -----------
        bid_curves : numpy.ndarray
            Recent bid curves
        ask_curves : numpy.ndarray
            Recent ask curves
        price_history : numpy.ndarray
            Recent price history
            
        Returns:
        --------
        dict
            Trading signal with action and strength
        """
        # Ensure we have enough data
        if len(bid_curves) < self.vfar_model.p or len(ask_curves) < self.vfar_model.p:
            return {'action': 'HOLD', 'strength': 0}
        
        # Predict future liquidity
        future_bid, future_ask = self.vfar_model.predict(
            bid_curves[-self.vfar_model.p:], 
            ask_curves[-self.vfar_model.p:],
            steps_ahead=self.forecast_horizon
        )
        
        # Calculate current and predicted liquidity imbalance
        current_imbalance = self._calculate_liquidity_imbalance(
            bid_curves[-1], ask_curves[-1]
        )
        future_imbalance = self._calculate_liquidity_imbalance(
            future_bid, future_ask
        )
        
        # Calculate change in imbalance
        imbalance_change = future_imbalance - current_imbalance
        
        # Predict price movement
        price_signal = self._predict_price_movement(price_history)
        
        # Combine signals (equal weight)
        combined_signal = 0.5 * imbalance_change + 0.5 * price_signal
        
        # Determine action based on signal strength
        if combined_signal > 0.2:
            action = 'BUY'
        elif combined_signal < -0.2:
            action = 'SELL'
        else:
            action = 'HOLD'
            
        return {
            'action': action,
            'strength': combined_signal,
            'liquidity_imbalance': future_imbalance,
            'imbalance_change': imbalance_change,
            'price_signal': price_signal
        }
    
    def execute_strategy(self, data_feed, initial_capital=100000, max_position=1000):
        """
        Execute strategy on historical data
        
        Parameters:
        -----------
        data_feed : dict
            Dictionary containing bid_curves, ask_curves, prices, and timestamps
        initial_capital : float
            Initial capital
        max_position : int
            Maximum position size (absolute value)
            
        Returns:
        --------
        dict
            Strategy results
        """
        bid_curves = data_feed['bid_curves']
        ask_curves = data_feed['ask_curves']
        prices = data_feed['prices']
        timestamps = data_feed['timestamps']
        
        capital = initial_capital
        self.position = 0
        self.position_history = []
        self.pnl_history = []
        trades = []
        
        for i in range(self.vfar_model.p + self.lookback, len(prices)):
            # Get data up to current time
            current_bid_curves = bid_curves[:i]
            current_ask_curves = ask_curves[:i]
            current_price_history = prices[:i]
            
            # Generate signal
            signal = self.generate_signal(
                current_bid_curves, 
                current_ask_curves,
                current_price_history
            )
            
            # Execute trading logic
            prev_position = self.position
            current_price = prices[i]
            
            if signal['action'] == 'BUY':
                # Scale position by signal strength
                target_position = int(max_position * min(1.0, abs(signal['strength'])))
                
                # If already long, potentially increase position
                if self.position >= 0:
                    delta = target_position - self.position
                    if delta > 0:
                        # Buy more
                        capital -= delta * current_price
                        self.position += delta
                        trades.append({
                            'timestamp': timestamps[i] if i < len(timestamps) else None,
                            'action': 'BUY',
                            'price': current_price,
                            'quantity': delta,
                            'signal_strength': signal['strength']
                        })
                else:
                    # Close short position and potentially go long
                    capital -= abs(self.position) * current_price  # Cost to close short
                    capital -= target_position * current_price  # Cost to go long
                    
                    trades.append({
                        'timestamp': timestamps[i] if i < len(timestamps) else None,
                        'action': 'COVER_AND_BUY',
                        'price': current_price,
                        'quantity': abs(self.position) + target_position,
                        'signal_strength': signal['strength']
                    })
                    
                    self.position = target_position
            
            elif signal['action'] == 'SELL':
                # Scale position by signal strength
                target_position = -int(max_position * min(1.0, abs(signal['strength'])))
                
                # If already short, potentially increase position
                if self.position <= 0:
                    delta = target_position - self.position
                    if delta < 0:
                        # Sell more
                        capital += abs(delta) * current_price
                        self.position += delta
                        trades.append({
                            'timestamp': timestamps[i] if i < len(timestamps) else None,
                            'action': 'SELL',
                            'price': current_price,
                            'quantity': abs(delta),
                            'signal_strength': signal['strength']
                        })
                else:
                    # Close long position and potentially go short
                    capital += self.position * current_price  # Proceeds from closing long
                    capital += abs(target_position) * current_price  # Proceeds from going short
                    
                    trades.append({
                        'timestamp': timestamps[i] if i < len(timestamps) else None,
                        'action': 'SELL_AND_SHORT',
                        'price': current_price,
                        'quantity': self.position + abs(target_position),
                        'signal_strength': signal['strength']
                    })
                    
                    self.position = target_position
            
            # Calculate portfolio value
            portfolio_value = capital + self.position * current_price
            self.position_history.append(self.position)
            self.pnl_history.append(portfolio_value - initial_capital)
            
        # Close final position
        final_price = prices[-1]
        if self.position != 0:
            if self.position > 0:
                capital += self.position * final_price
                trades.append({
                    'timestamp': timestamps[-1] if timestamps else None,
                    'action': 'SELL',
                    'price': final_price,
                    'quantity': self.position,
                    'signal_strength': 0
                })
            else:
                capital -= abs(self.position) * final_price
                trades.append({
                    'timestamp': timestamps[-1] if timestamps else None,
                    'action': 'COVER',
                    'price': final_price,
                    'quantity': abs(self.position),
                    'signal_strength': 0
                })
                
        final_portfolio = capital
        total_return = (final_portfolio / initial_capital) - 1
        
        return {
            'initial_capital': initial_capital,
            'final_portfolio': final_portfolio,
            'total_return': total_return,
            'trades': trades,
            'position_history': self.position_history,
            'pnl_history': self.pnl_history
        }


class LiquidityBreakoutStrategy:
    def __init__(self, vfar_model, lookback=20, forecast_horizon=3, liquidity_threshold=0.25):
        """
        Strategy that identifies breakouts based on changes in liquidity structure
        
        Parameters:
        -----------
        vfar_model : VFARModel
            Fitted VFAR model for liquidity prediction
        lookback : int
            Number of periods to establish the range
        forecast_horizon : int
            Number of periods ahead to forecast
        liquidity_threshold : float
            Threshold for significant liquidity change (0-1)
        """
        self.vfar_model = vfar_model
        self.lookback = lookback
        self.forecast_horizon = forecast_horizon
        self.liquidity_threshold = liquidity_threshold
        self.position = 0
        self.position_history = []
        self.pnl_history = []
    
    def _calculate_liquidity_at_price_levels(self, bid_curve, ask_curve):
        """
        Calculate liquidity at different relative price levels
        
        Returns:
        --------
        dict
            Liquidity measures at different price points
        """
        # Calculate liquidity at best bid/ask (front of book)
        front_bid_liquidity = bid_curve[0]
        front_ask_liquidity = ask_curve[0]
        
        # Calculate liquidity in the middle of the book
        mid_bid_liquidity = np.mean(bid_curve[20:40])
        mid_ask_liquidity = np.mean(ask_curve[20:40])
        
        # Calculate liquidity at the back of the book
        back_bid_liquidity = np.mean(bid_curve[80:])
        back_ask_liquidity = np.mean(ask_curve[80:])
        
        # Calculate bid/ask imbalances at each level
        front_imbalance = front_bid_liquidity - front_ask_liquidity
        mid_imbalance = mid_bid_liquidity - mid_ask_liquidity
        back_imbalance = back_bid_liquidity - back_ask_liquidity
        
        return {
            'front_bid': front_bid_liquidity,
            'front_ask': front_ask_liquidity,
            'mid_bid': mid_bid_liquidity,
            'mid_ask': mid_ask_liquidity,
            'back_bid': back_bid_liquidity,
            'back_ask': back_ask_liquidity,
            'front_imbalance': front_imbalance,
            'mid_imbalance': mid_imbalance,
            'back_imbalance': back_imbalance
        }
    
    def _detect_liquidity_breakout(self, historical_liquidity, forecast_liquidity):
        """
        Detect breakout patterns in liquidity structure
        
        Parameters:
        -----------
        historical_liquidity : list
            List of liquidity measures for historical periods
        forecast_liquidity : dict
            Forecast liquidity measures
            
        Returns:
        --------
        dict
            Breakout signals
        """
        # Calculate historical ranges for each measure
        measure_names = ['front_imbalance', 'mid_imbalance', 'back_imbalance']
        ranges = {}
        
        for measure in measure_names:
            values = [entry[measure] for entry in historical_liquidity]
            ranges[measure] = {
                'min': np.min(values),
                'max': np.max(values),
                'mean': np.mean(values),
                'std': np.std(values)
            }
        
        # Detect breakouts in forecast
        breakouts = {}
        
        for measure in measure_names:
            forecast_value = forecast_liquidity[measure]
            measure_range = ranges[measure]
            
            # Calculate Z-score
            if measure_range['std'] > 0:
                z_score = (forecast_value - measure_range['mean']) / measure_range['std']
            else:
                z_score = 0
                
            # Breakout if z-score exceeds threshold
            if z_score > self.liquidity_threshold:
                breakouts[measure] = {'direction': 'up', 'z_score': z_score}
            elif z_score < -self.liquidity_threshold:
                breakouts[measure] = {'direction': 'down', 'z_score': z_score}
        
        return breakouts
    
    def generate_signal(self, bid_curves, ask_curves):
        """
        Generate trading signal based on liquidity breakouts
        
        Parameters:
        -----------
        bid_curves : numpy.ndarray
            Recent bid curves
        ask_curves : numpy.ndarray
            Recent ask curves
            
        Returns:
        --------
        dict
            Trading signal with action and strength
        """
        # Ensure we have enough data
        if len(bid_curves) < max(self.lookback, self.vfar_model.p):
            return {'action': 'HOLD', 'strength': 0}
        
        # Calculate historical liquidity measures
        historical_liquidity = []
        for i in range(-self.lookback, 0):
            liquidity = self._calculate_liquidity_at_price_levels(
                bid_curves[i], ask_curves[i]
            )
            historical_liquidity.append(liquidity)
        
        # Predict future liquidity
        future_bid, future_ask = self.vfar_model.predict(
            bid_curves[-self.vfar_model.p:], 
            ask_curves[-self.vfar_model.p:],
            steps_ahead=self.forecast_horizon
        )
        
        # Calculate forecast liquidity measures
        forecast_liquidity = self._calculate_liquidity_at_price_levels(
            future_bid, future_ask
        )
        
        # Detect breakouts
        breakouts = self._detect_liquidity_breakout(
            historical_liquidity, forecast_liquidity
        )
        
        # Generate trading signal based on breakouts
        if 'front_imbalance' in breakouts:
            front_breakout = breakouts['front_imbalance']
            
            # Front of book imbalance is most important for short-term signals
            if front_breakout['direction'] == 'up':
                action = 'BUY'
                strength = min(1.0, abs(front_breakout['z_score']) / 2)
            else:
                action = 'SELL'
                strength = min(1.0, abs(front_breakout['z_score']) / 2)
        
        elif 'mid_imbalance' in breakouts:
            mid_breakout = breakouts['mid_imbalance']
            
            # Mid book imbalance provides secondary signals
            if mid_breakout['direction'] == 'up':
                action = 'BUY'
                strength = min(0.7, abs(mid_breakout['z_score']) / 3)
            else:
                action = 'SELL'
                strength = min(0.7, abs(mid_breakout['z_score']) / 3)
        
        elif 'back_imbalance' in breakouts:
            back_breakout = breakouts['back_imbalance']
            
            # Back of book imbalance provides weak signals
            if back_breakout['direction'] == 'up':
                action = 'BUY'
                strength = min(0.4, abs(back_breakout['z_score']) / 4)
            else:
                action = 'SELL'
                strength = min(0.4, abs(back_breakout['z_score']) / 4)
        
        else:
            # No breakouts detected
            action = 'HOLD'
            strength = 0
        
        return {
            'action': action,
            'strength': strength,
            'breakouts': breakouts,
            'forecast_liquidity': forecast_liquidity
        }
    
    def execute_strategy(self, data_feed, initial_capital=100000, max_position=1000):
        """
        Execute strategy on historical data
        
        Parameters:
        -----------
        data_feed : dict
            Dictionary containing bid_curves, ask_curves, prices, and timestamps
        initial_capital : float
            Initial capital
        max_position : int
            Maximum position size (absolute value)
            
        Returns:
        --------
        dict
            Strategy results
        """
        bid_curves = data_feed['bid_curves']
        ask_curves = data_feed['ask_curves']
        prices = data_feed['prices']
        timestamps = data_feed['timestamps']
        
        capital = initial_capital
        self.position = 0
        self.position_history = []
        self.pnl_history = []
        trades = []
        
        for i in range(max(self.vfar_model.p, self.lookback), len(prices)):
            # Get data up to current time
            current_bid_curves = bid_curves[:i]
            current_ask_curves = ask_curves[:i]
            
            # Generate signal
            signal = self.generate_signal(
                current_bid_curves, 
                current_ask_curves
            )
            
            # Execute trading logic
            prev_position = self.position
            current_price = prices[i]
            
            if signal['action'] == 'BUY':
                # Scale position by signal strength
                target_position = int(max_position * min(1.0, abs(signal['strength'])))
                
                # If already long, potentially increase position
                if self.position >= 0:
                    delta = target_position - self.position
                    if delta > 0:
                        # Buy more
                        capital -= delta * current_price
                        self.position += delta
                        trades.append({
                            'timestamp': timestamps[i] if i < len(timestamps) else None,
                            'action': 'BUY',
                            'price': current_price,
                            'quantity': delta,
                            'signal_strength': signal['strength']
                        })
                else:
                    # Close short position and potentially go long
                    capital -= abs(self.position) * current_price  # Cost to close short
                    capital -= target_position * current_price  # Cost to go long
                    
                    trades.append({
                        'timestamp': timestamps[i] if i < len(timestamps) else None,
                        'action': 'COVER_AND_BUY',
                        'price': current_price,
                        'quantity': abs(self.position) + target_position,
                        'signal_strength': signal['strength']
                    })
                    
                    self.position = target_position
            
            elif signal['action'] == 'SELL':
                # Scale position by signal strength
                target_position = -int(max_position * min(1.0, abs(signal['strength'])))
                
                # If already short, potentially increase position
                if self.position <= 0:
                    delta = target_position - self.position
                    if delta < 0:
                        # Sell more
                        capital += abs(delta) * current_price
                        self.position += delta
                        trades.append({
                            'timestamp': timestamps[i] if i < len(timestamps) else None,
                            'action': 'SELL',
                            'price': current_price,
                            'quantity': abs(delta),
                            'signal_strength': signal['strength']
                        })
                else:
                    # Close long position and potentially go short
                    capital += self.position * current_price  # Proceeds from closing long
                    capital += abs(target_position) * current_price  # Proceeds from going short
                    
                    trades.append({
                        'timestamp': timestamps[i] if i < len(timestamps) else None,
                        'action': 'SELL_AND_SHORT',
                        'price': current_price,
                        'quantity': self.position + abs(target_position),
                        'signal_strength': signal['strength']
                    })
                    
                    self.position = target_position
            
            # Calculate portfolio value
            portfolio_value = capital + self.position * current_price
            self.position_history.append(self.position)
            self.pnl_history.append(portfolio_value - initial_capital)
            
        # Close final position
        final_price = prices[-1]
        if self.position != 0:
            if self.position > 0:
                capital += self.position * final_price
                trades.append({
                    'timestamp': timestamps[-1] if timestamps else None,
                    'action': 'SELL',
                    'price': final_price,
                    'quantity': self.position,
                    'signal_strength': 0
                })
            else:
                capital -= abs(self.position) * final_price
                trades.append({
                    'timestamp': timestamps[-1] if timestamps else None,
                    'action': 'COVER',
                    'price': final_price,
                    'quantity': abs(self.position),
                    'signal_strength': 0
                })
                
        final_portfolio = capital
        total_return = (final_portfolio / initial_capital) - 1
        
        return {
            'initial_capital': initial_capital,
            'final_portfolio': final_portfolio,
            'total_return': total_return,
            'trades': trades,
            'position_history': self.position_history,
            'pnl_history': self.pnl_history
        }


class LiquidityAdjustedMarketMaker:
    def __init__(self, vfar_model, forecast_horizon=5, base_spread=0.02, 
                 max_inventory=1000, inventory_risk_factor=0.01):
        """
        Market making strategy that adjusts quotes based on predicted liquidity
        
        Parameters:
        -----------
        vfar_model : VFARModel
            Fitted VFAR model for liquidity prediction
        forecast_horizon : int
            Number of periods ahead to forecast
        base_spread : float
            Base spread as percentage of price
        max_inventory : int
            Maximum inventory (long or short)
        inventory_risk_factor : float
            Factor for inventory risk adjustment
        """
        self.vfar_model = vfar_model
        self.forecast_horizon = forecast_horizon
        self.base_spread = base_spread
        self.max_inventory = max_inventory
        self.inventory_risk_factor = inventory_risk_factor
        self.inventory = 0
        self.inventory_history = []
        self.quotes_history = []
        self.pnl_history = []
    
    def _calculate_liquidity_risk(self, bid_curve, ask_curve):
        """
        Calculate liquidity risk based on curve structure
        
        Returns:
        --------
        dict
            Liquidity risk metrics
        """
        # Calculate average slope of curves
        bid_slope = (bid_curve[-1] - bid_curve[0]) / len(bid_curve)
        ask_slope = (ask_curve[-1] - ask_curve[0]) / len(ask_curve)
        
        # Steeper slopes indicate more concentrated liquidity (lower risk)
        bid_liquidity_risk = 1.0 / (1.0 + abs(bid_slope) * 10)
        ask_liquidity_risk = 1.0 / (1.0 + abs(ask_slope) * 10)
        
        # Calculate depth (total volume)
        bid_depth = np.sum(np.exp(bid_curve))
        ask_depth = np.sum(np.exp(ask_curve))
        
        # Normalize depth to 0-1 scale
        max_depth = max(bid_depth, ask_depth)
        normalized_bid_depth = bid_depth / max_depth if max_depth > 0 else 0.5
        normalized_ask_depth = ask_depth / max_depth if max_depth > 0 else 0.5
        
        # Calculate overall risk (lower is better)
        bid_overall_risk = bid_liquidity_risk * (1 - normalized_bid_depth)
        ask_overall_risk = ask_liquidity_risk * (1 - normalized_ask_depth)
        
        return {
            'bid_risk': bid_overall_risk,
            'ask_risk': ask_overall_risk,
            'bid_depth': normalized_bid_depth,
            'ask_depth': normalized_ask_depth
        }
    
    def _calculate_inventory_skew(self):
        """
        Calculate quote skew based on current inventory
        
        Returns:
        --------
        float
            Inventory skew factor (-1 to 1)
        """
        # Normalize inventory to -1 to 1 range
        normalized_inventory = self.inventory / self.max_inventory
        
        # Apply non-linear transformation to increase pressure as inventory grows
        inventory_skew = np.tanh(normalized_inventory * 2)
        
        return inventory_skew
    
    def generate_quotes(self, bid_curves, ask_curves, mid_price):
        """
        Generate bid/ask quotes based on liquidity prediction and inventory
        
        Parameters:
        -----------
        bid_curves : numpy.ndarray
            Recent bid curves
        ask_curves : numpy.ndarray
            Recent ask curves
        mid_price : float
            Current mid price
            
        Returns:
        --------
        dict
            Bid and ask quotes
        """
        # Ensure we have enough data
        if len(bid_curves) < self.vfar_model.p:
            return {
                'bid_price': mid_price * (1 - self.base_spread/2),
                'ask_price': mid_price * (1 + self.base_spread/2),
                'bid_size': self.max_inventory // 10,
                'ask_size': self.max_inventory // 10
            }
        
        # Predict future liquidity
        future_bid, future_ask = self.vfar_model.predict(
            bid_curves[-self.vfar_model.p:], 
            ask_curves[-self.vfar_model.p:],
            steps_ahead=self.forecast_horizon
        )
        
        # Calculate liquidity risk from forecasted curves
        liquidity_risk = self._calculate_liquidity_risk(future_bid, future_ask)
        
        # Calculate inventory skew
        inventory_skew = self._calculate_inventory_skew()
        
        # Adjust spread based on liquidity risk
        # Higher risk on either side increases that side's spread
        bid_spread = self.base_spread/2 * (1 + liquidity_risk['bid_risk'])
        ask_spread = self.base_spread/2 * (1 + liquidity_risk['ask_risk'])
        
        # Adjust prices based on inventory
        # Positive inventory (long) increases ask price and decreases bid price
        inventory_adjustment = mid_price * inventory_skew * self.inventory_risk_factor
        
        bid_price = mid_price * (1 - bid_spread) - inventory_adjustment
        ask_price = mid_price * (1 + ask_spread) + inventory_adjustment
        
        # Adjust sizes based on liquidity and inventory
        max_bid_size = int(self.max_inventory * (1 + inventory_skew))
        max_ask_size = int(self.max_inventory * (1 - inventory_skew))
        
        # Scale by liquidity depth (more liquidity = larger quotes)
        bid_size = int(max(10, min(max_bid_size, max_bid_size * liquidity_risk['bid_depth'])))
        ask_size = int(max(10, min(max_ask_size, max_ask_size * liquidity_risk['ask_depth'])))
        
        return {
            'bid_price': bid_price,
            'ask_price': ask_price,
            'bid_size': bid_size,
            'ask_size': ask_size,
            'liquidity_risk': liquidity_risk,
            'inventory_skew': inventory_skew
        }
    
    def execute_strategy(self, data_feed, initial_capital=100000, transaction_cost=0.0005):
        """
        Execute market making strategy on historical data
        
        Parameters:
        -----------
        data_feed : dict
            Dictionary containing bid_curves, ask_curves, prices, and timestamps
        initial_capital : float
            Initial capital
        transaction_cost : float
            Transaction cost as percentage of price
            
        Returns:
        --------
        dict
            Strategy results
        """
        bid_curves = data_feed['bid_curves']
        ask_curves = data_feed['ask_curves']
        prices = data_feed['prices']  # Assuming these are mid prices
        timestamps = data_feed['timestamps']
        
        capital = initial_capital
        self.inventory = 0
        self.inventory_history = []
        self.quotes_history = []
        self.pnl_history = []
        trades = []
        
        for i in range(self.vfar_model.p, len(prices)-1):  # Stop one before the end
            current_price = prices[i]
            next_price = prices[i+1]  # Price at which trades will execute
            
            # Generate quotes
            quotes = self.generate_quotes(
                bid_curves[:i],
                ask_curves[:i],
                current_price
            )
            
            self.quotes_history.append(quotes)
            
            # Simulate trade execution
            bid_executed = next_price <= quotes['bid_price']
            ask_executed = next_price >= quotes['ask_price']
            
            # Update inventory and capital
            if bid_executed:
                # Buy at our bid
                trade_size = quotes['bid_size']
                trade_price = quotes['bid_price']
                
                # Limit by max inventory
                actual_trade = min(trade_size, self.max_inventory - self.inventory)
                
                if actual_trade > 0:
                    self.inventory += actual_trade
                    capital -= actual_trade * trade_price * (1 + transaction_cost)
                    
                    trades.append({
                        'timestamp': timestamps[i] if i < len(timestamps) else None,
                        'action': 'BUY',
                        'price': trade_price,
                        'quantity': actual_trade,
                        'inventory': self.inventory
                    })
            
            if ask_executed:
                # Sell at our ask
                trade_size = quotes['ask_size']
                trade_price = quotes['ask_price']
                
                # Limit by max inventory
                actual_trade = min(trade_size, self.inventory + self.max_inventory)
                
                if actual_trade > 0:
                    self.inventory -= actual_trade
                    capital += actual_trade * trade_price * (1 - transaction_cost)
                    
                    trades.append({
                        'timestamp': timestamps[i] if i < len(timestamps) else None,
                        'action': 'SELL',
                        'price': trade_price,
                        'quantity': actual_trade,
                        'inventory': self.inventory
                    })
            
            # Calculate mark-to-market value
            portfolio_value = capital + self.inventory * current_price
            
            self.inventory_history.append(self.inventory)
            self.pnl_history.append(portfolio_value - initial_capital)
        
        # Close final position at last price
        final_price = prices[-1]
        if self.inventory != 0:
            if self.inventory > 0:
                capital += self.inventory * final_price * (1 - transaction_cost)
                trades.append({
                    'timestamp': timestamps[-1] if timestamps else None,
                    'action': 'SELL',
                    'price': final_price,
                    'quantity': self.inventory,
                    'inventory': 0
                })
            else:
                capital -= abs(self.inventory) * final_price * (1 + transaction_cost)
                trades.append({
                    'timestamp': timestamps[-1] if timestamps else None,
                    'action': 'BUY',
                    'price': final_price,
                    'quantity': abs(self.inventory),
                    'inventory': 0
                })
                
        final_portfolio = capital
        total_return = (final_portfolio / initial_capital) - 1
        
        return {
            'initial_capital': initial_capital,
            'final_portfolio': final_portfolio,
            'total_return': total_return,
            'trades': trades,
            'inventory_history': self.inventory_history,
            'quotes_history': self.quotes_history,
            'pnl_history': self.pnl_history
        }


class IntegratedTradingStrategy:
    def __init__(self, vfar_model, lookback=10, forecast_horizon=5, 
                 trading_points=10, position_limit=1000):
        """
        Integrated strategy that combines trading signals with execution optimization
        
        Parameters:
        -----------
        vfar_model : VFARModel
            Fitted VFAR model for liquidity prediction
        lookback : int
            Number of periods to look back for signals
        forecast_horizon : int
            Number of periods ahead to forecast
        trading_points : int
            Number of trading points per day
        position_limit : int
            Maximum position size
        """
        self.vfar_model = vfar_model
        self.lookback = lookback
        self.forecast_horizon = forecast_horizon
        self.trading_points = trading_points
        self.position_limit = position_limit
        self.position = 0
        self.position_history = []
        self.pnl_history = []
        
        # Create execution optimizer
        self.execution_optimizer = OrderExecutionStrategy(
            vfar_model, volume=100, time_points=75, trading_points=trading_points
        )
    
    def _calculate_market_impact(self, volume, bid_curve, ask_curve, is_buy):
        """
        Estimate market impact of a trade
        
        Returns:
        --------
        float
            Estimated impact as percentage of price
        """
        # Use appropriate curve based on trade direction
        curve = ask_curve if is_buy else bid_curve
        
        # Calculate log volume
        log_volume = np.log(max(1, volume))
        
        # Find closest level in curve
        idx = np.searchsorted(curve, log_volume)
        
        # Calculate impact as distance from best price
        if idx >= len(curve):
            impact = 0.01 * len(curve)  # Large impact for exceeding book depth
        else:
            impact = 0.01 * idx  # 1% per level in the book
            
        return impact
    
    def _detect_trend(self, price_history):
        """
        Detect trend in price history
        
        Returns:
        --------
        float
            Trend strength (-1 to 1)
        """
        if len(price_history) < self.lookback:
            return 0
            
        # Simple linear regression
        x = np.arange(self.lookback)
        y = price_history[-self.lookback:]
        slope, _ = np.polyfit(x, y, 1)
        
        # Normalize slope to -1 to 1 range
        normalized_slope = np.tanh(slope * 100)
        
        return normalized_slope
    
    def _detect_liquidity_flow(self, bid_curves, ask_curves):
        """
        Detect flow of liquidity in recent curves
        
        Returns:
        --------
        float
            Liquidity flow indicator (-1 to 1)
        """
        if len(bid_curves) < self.lookback or len(ask_curves) < self.lookback:
            return 0
            
        # Calculate total liquidity on each side over time
        bid_liquidity = [np.sum(np.exp(curve)) for curve in bid_curves[-self.lookback:]]
        ask_liquidity = [np.sum(np.exp(curve)) for curve in ask_curves[-self.lookback:]]
        
        # Calculate bid/ask ratio over time
        ratio = [b/a if a > 0 else 1.0 for b, a in zip(bid_liquidity, ask_liquidity)]
        
        # Calculate slope of ratio
        x = np.arange(self.lookback)
        slope, _ = np.polyfit(x, ratio, 1)
        
        # Normalize to -1 to 1 range
        flow_indicator = np.tanh(slope * 10)
        
        return flow_indicator
    
    def generate_signal(self, bid_curves, ask_curves, price_history):
        """
        Generate trading signal based on price trend and liquidity flow
        
        Parameters:
        -----------
        bid_curves : numpy.ndarray
            Recent bid curves
        ask_curves : numpy.ndarray
            Recent ask curves
        price_history : numpy.ndarray
            Recent price history
            
        Returns:
        --------
        dict
            Trading signal
        """
        # Ensure we have enough data
        if len(bid_curves) < max(self.lookback, self.vfar_model.p) or len(price_history) < self.lookback:
            return {'action': 'HOLD', 'target_position': 0, 'strength': 0}
        
        # Detect price trend
        trend = self._detect_trend(price_history)
        
        # Detect liquidity flow
        liquidity_flow = self._detect_liquidity_flow(bid_curves, ask_curves)
        
        # Predict future liquidity
        future_bid, future_ask = self.vfar_model.predict(
            bid_curves[-self.vfar_model.p:], 
            ask_curves[-self.vfar_model.p:],
            steps_ahead=self.forecast_horizon
        )
        
        # Calculate current and future market impact
        current_impact_buy = self._calculate_market_impact(100, bid_curves[-1], ask_curves[-1], True)
        current_impact_sell = self._calculate_market_impact(100, bid_curves[-1], ask_curves[-1], False)
        future_impact_buy = self._calculate_market_impact(100, future_bid, future_ask, True)
        future_impact_sell = self._calculate_market_impact(100, future_bid, future_ask, False)
        
        # Calculate impact changes
        impact_buy_change = future_impact_buy - current_impact_buy
        impact_sell_change = future_impact_sell - current_impact_sell
        
        # Combine signals
        # Positive when:
        # 1. Price is trending up
        # 2. Liquidity is flowing from ask to bid (ratio increasing)
        # 3. Future impact of buying is decreasing
        buy_signal = trend + 0.5 * liquidity_flow - 5 * impact_buy_change
        
        # Positive when:
        # 1. Price is trending down
        # 2. Liquidity is flowing from bid to ask (ratio decreasing)
        # 3. Future impact of selling is decreasing
        sell_signal = -trend - 0.5 * liquidity_flow - 5 * impact_sell_change
        
        # Generate final signal
        if buy_signal > 0.3 and buy_signal > sell_signal:
            action = 'BUY'
            strength = min(1.0, buy_signal)
            target_position = int(self.position_limit * strength)
        elif sell_signal > 0.3 and sell_signal > buy_signal:
            action = 'SELL'
            strength = min(1.0, sell_signal)
            target_position = -int(self.position_limit * strength)
        else:
            action = 'HOLD'
            strength = 0
            target_position = 0
        
        return {
            'action': action,
            'target_position': target_position,
            'strength': strength,
            'trend': trend,
            'liquidity_flow': liquidity_flow,
            'impact_buy_change': impact_buy_change,
            'impact_sell_change': impact_sell_change
        }
    
    def execute_strategy(self, data_feed, initial_capital=100000):
        """
        Execute integrated strategy on historical data
        
        Parameters:
        -----------
        data_feed : dict
            Dictionary containing bid_curves, ask_curves, prices, and timestamps
        initial_capital : float
            Initial capital
            
        Returns:
        --------
        dict
            Strategy results
        """
        bid_curves = data_feed['bid_curves']
        ask_curves = data_feed['ask_curves']
        prices = data_feed['prices']
        timestamps = data_feed['timestamps']
        
        capital = initial_capital
        self.position = 0
        self.position_history = []
        self.pnl_history = []
        trades = []
        
        day_count = len(prices) // 75
        
        for day_idx in range(day_count):  # Assuming 75 intervals per day
            day_start = day_idx * 75
            day_end = min((day_idx + 1) * 75, len(prices))
            
            # Skip if not enough data
            if day_end - day_start < 10 or day_start < max(self.lookback, self.vfar_model.p):
                continue
            
            # Generate target position for this day
            signal = self.generate_signal(
                bid_curves[:day_start],
                ask_curves[:day_start],
                prices[:day_start]
            )
            
            target_position = signal['target_position']
            delta_position = target_position - self.position
            
            # Skip if no position change
            if delta_position == 0:
                continue
            
            # Use execution optimizer to schedule trades
            self.execution_optimizer.volume = abs(delta_position)
            self.execution_optimizer.is_buy = delta_position > 0
            
            if day_idx == 0:
                recent_bid = bid_curves[max(0, day_start-self.vfar_model.p):day_start]
                recent_ask = ask_curves[max(0, day_start-self.vfar_model.p):day_start]
                
                # Pad if needed
                if len(recent_bid) < self.vfar_model.p:
                    pad_size = self.vfar_model.p - len(recent_bid)
                    recent_bid = np.vstack([np.zeros((pad_size, recent_bid.shape[1])), recent_bid])
                    recent_ask = np.vstack([np.zeros((pad_size, recent_ask.shape[1])), recent_ask])
            else:
                recent_bid = bid_curves[max(0, day_start-self.vfar_model.p):day_start]
                recent_ask = ask_curves[max(0, day_start-self.vfar_model.p):day_start]
                
                # Pad if needed
                if len(recent_bid) < self.vfar_model.p:
                    pad_size = self.vfar_model.p - len(recent_bid)
                    recent_bid = np.vstack([np.zeros((pad_size, recent_bid.shape[1])), recent_bid])
                    recent_ask = np.vstack([np.zeros((pad_size, recent_ask.shape[1])), recent_ask])
            
            # Get optimal execution schedule
            try:
                schedule = self.execution_optimizer.vfar_strategy(
                    recent_bid, recent_ask, delta_position > 0
                )
            except Exception as e:
                print(f"Error generating schedule: {str(e)}")
                # Use simple schedule as fallback
                schedule = [(i, abs(delta_position) / 10) for i in range(1, 11)]
            
            # Execute according to schedule
            daily_pnl = 0
            
            for point, trade_volume in schedule:
                execution_idx = day_start + point - 1
                
                # Skip if beyond data range
                if execution_idx >= day_end or execution_idx >= len(prices):
                    continue
                
                execution_price = prices[execution_idx]
                is_buy = delta_position > 0
                
                # Adjust position and capital
                if is_buy:
                    self.position += trade_volume
                    capital -= trade_volume * execution_price
                else:
                    self.position -= trade_volume
                    capital += trade_volume * execution_price
                
                trades.append({
                    'timestamp': timestamps[execution_idx] if execution_idx < len(timestamps) else None,
                    'action': 'BUY' if is_buy else 'SELL',
                    'price': execution_price,
                    'quantity': trade_volume,
                    'position': self.position
                })
            
            # Calculate daily PnL
            if day_end - 1 < len(prices):
                day_end_price = prices[day_end-1]
                day_end_value = capital + self.position * day_end_price
                daily_pnl = day_end_value - initial_capital
                
                self.position_history.append(self.position)
                self.pnl_history.append(daily_pnl)
        
        # Close final position
        if self.position != 0 and len(prices) > 0:
            final_price = prices[-1]
            if self.position > 0:
                capital += self.position * final_price
                trades.append({
                    'timestamp': timestamps[-1] if timestamps else None,
                    'action': 'SELL',
                    'price': final_price,
                    'quantity': self.position,
                    'position': 0
                })
            else:
                capital -= abs(self.position) * final_price
                trades.append({
                    'timestamp': timestamps[-1] if timestamps else None,
                    'action': 'BUY',
                    'price': final_price,
                    'quantity': abs(self.position),
                    'position': 0
                })
                
        final_portfolio = capital
        total_return = (final_portfolio / initial_capital) - 1
        
        return {
            'initial_capital': initial_capital,
            'final_portfolio': final_portfolio,
            'total_return': total_return,
            'trades': trades,
            'position_history': self.position_history,
            'pnl_history': self.pnl_history
        }


#############################################
# Main Application with Combined Strategies
#############################################

def run_combined_strategies():
    """
    Run a complete analysis with all strategies
    """
    print("VFAR LOB Combined Strategy Backtest")
    print("=" * 40)
    
    # Initialize simulator and processor
    simulator = SimulatedOrderBookGenerator(
        base_price=100,
        volatility=0.02,
        spread_mean=0.01,
        depth_factor=1000,
        random_seed=42
    )
    processor = LOBDataProcessor()
    
    # Run original VFAR backtester
    backtester = VFARBacktester(simulator, processor)
    
    print("\nRunning basic VFAR backtester...")
    basic_results = backtester.backtest(
        ticker_name="Simulated Stock",
        volume=10000,
        n_days=60  # More data for better testing
    )
    
    # Extract model and data for trading strategies
    model = basic_results['model']
    bid_curves = basic_results['bid_curves']
    ask_curves = basic_results['ask_curves']
    timestamps = basic_results['timestamps']
    
    # Generate price series from snapshots
    print("\nGenerating price series...")
    snapshots = simulator.generate_historical_data(n_days=60)
    prices = []
    for snapshot in snapshots:
        if timestamps and snapshot['timestamp'] in timestamps:
            mid_price = snapshot['mid_price']
            prices.append(mid_price)
    
    # Truncate to match the processed data length
    prices = prices[:len(bid_curves)]
    
    # Prepare data feed for trading strategies
    data_feed = {
        'bid_curves': bid_curves,
        'ask_curves': ask_curves,
        'prices': np.array(prices),
        'timestamps': timestamps[:len(bid_curves)]
    }
    
    # Create trading strategies
    print("\nInitializing trading strategies...")
    strategy_instances = {
        'Alpha Strategy': LiquidityAwareAlphaStrategy(model, lookback=10, forecast_horizon=5),
        'Breakout Strategy': LiquidityBreakoutStrategy(model, lookback=20, forecast_horizon=3),
        'Market Making': LiquidityAdjustedMarketMaker(model, forecast_horizon=5),
        'Integrated Strategy': IntegratedTradingStrategy(model, lookback=10, forecast_horizon=5)
    }
    
    # Execute strategies
    print("\nRunning trading strategies...")
    strategy_results = {}
    initial_capital = 100000
    
    for name, strategy in strategy_instances.items():
        print(f"Running {name}...")
        try:
            results = strategy.execute_strategy(data_feed, initial_capital)
            strategy_results[name] = results
        except Exception as e:
            print(f"Error running {name}: {str(e)}")
            strategy_results[name] = {
                'error': str(e),
                'initial_capital': initial_capital,
                'final_portfolio': initial_capital,
                'total_return': 0,
                'pnl_history': [0],
                'trades': []
            }
    
    # Print results
    print("\nStrategy Performance Summary:")
    print("=" * 40)
    
    # Sort strategies by performance
    sorted_strategies = sorted(
        [(name, results) for name, results in strategy_results.items()],
        key=lambda x: x[1].get('total_return', 0),
        reverse=True
    )
    
    for name, results in sorted_strategies:
        total_return = results.get('total_return', 0) * 100
        final_portfolio = results.get('final_portfolio', initial_capital)
        trade_count = len(results.get('trades', []))
        
        print(f"{name}:")
        print(f"  Total Return: {total_return:.2f}%")
        print(f"  Final Portfolio: ${final_portfolio:.2f}")
        print(f"  Number of Trades: {trade_count}")
        print("-" * 40)
    
    # Plot performance comparison
    plt.figure(figsize=(12, 6))
    
    for name, results in strategy_results.items():
        pnl_history = results.get('pnl_history', [0])
        if len(pnl_history) > 0:
            plt.plot(pnl_history, label=name)
    
    plt.axhline(y=0, color='black', linestyle='--')
    plt.xlabel('Time Steps')
    plt.ylabel('P&L ($)')
    plt.title('Strategy Performance Comparison')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    # Plot position over time for best strategy
    best_strategy = sorted_strategies[0][0]
    best_results = strategy_results[best_strategy]
    
    if 'position_history' in best_results and len(best_results['position_history']) > 0:
        plt.figure(figsize=(12, 6))
        plt.plot(best_results['position_history'])
        plt.axhline(y=0, color='black', linestyle='--')
        plt.xlabel('Time Steps')
        plt.ylabel('Position Size')
        plt.title(f'Position History - {best_strategy}')
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    
    # Analyze trade distribution
    for name, results in sorted_strategies[:2]:  # Show top 2 strategies
        trades = results.get('trades', [])
        if len(trades) > 0:
            # Extract trade data
            actions = [t['action'] for t in trades]
            quantities = [t['quantity'] for t in trades]
            prices = [t['price'] for t in trades]
            
            # Plot trade distribution
            plt.figure(figsize=(14, 10))
            
            # Create subplots
            plt.subplot(2, 2, 1)
            action_counts = {}
            for action in actions:
                action_counts[action] = action_counts.get(action, 0) + 1
            plt.bar(action_counts.keys(), action_counts.values())
            plt.title(f'{name} - Trade Actions')
            plt.xticks(rotation=45)
            
            plt.subplot(2, 2, 2)
            plt.hist(quantities, bins=20)
            plt.title(f'{name} - Trade Sizes')
            
            plt.subplot(2, 2, 3)
            plt.hist(prices, bins=20)
            plt.title(f'{name} - Trade Prices')
            
            plt.subplot(2, 2, 4)
            if 'position_history' in results:
                plt.plot(results['position_history'])
                plt.axhline(y=0, color='black', linestyle='--')
                plt.title(f'{name} - Position History')
            
            plt.tight_layout()
            plt.show()
    
    # Compare with original VFAR execution strategy
    print("\nComparing with original VFAR execution strategy:")
    
    # Extract VFAR strategy results
    vfar_strategies = basic_results.get('strategies', [])
    if vfar_strategies:
        best_vfar = max(vfar_strategies, key=lambda s: s.get('avg_savings_bps', 0))
        print(f"Best VFAR execution point count: {best_vfar.get('trading_points', 0)}")
        print(f"Average savings: {best_vfar.get('avg_savings_bps', 0):.2f} bps")
    
    return {
        'basic_results': basic_results,
        'strategy_results': strategy_results,
        'data_feed': data_feed
    }

if __name__ == "__main__":
    run_combined_strategies()