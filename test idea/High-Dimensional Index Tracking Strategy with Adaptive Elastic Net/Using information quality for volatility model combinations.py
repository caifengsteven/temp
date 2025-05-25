import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
from datetime import datetime, timedelta
import warnings

# Set random seed for reproducibility
np.random.seed(42)

class VolatilityModelCombination:
    """
    Implementation of the paper 'Using information quality for volatility model combinations' by Golosnoy and Okhrin
    
    This class implements the adaptive model combination methodology that accounts for information quality
    """
    
    def __init__(self, memory_parameter=0.95, threshold=0.0, strategy="winner"):
        """
        Initialize the volatility model combination framework
        
        Parameters:
        -----------
        memory_parameter : float
            Parameter φ₁ in the paper, controls the memory of the weight update (0-1)
        threshold : float
            Parameter α in the paper, minimum outperformance threshold
        strategy : str
            Either "winner" for winner-takes-all or "proportional" for proportional weighting
        """
        self.memory = memory_parameter
        self.threshold = threshold
        self.strategy = strategy
        
        # Initialize weights and models
        self.weights = None
        self.models = {}
        self.model_names = []
        self.benchmark_model = None
        
        # State space model parameters
        self.information_flow = []
        
    def add_model(self, name, model_func, is_benchmark=False):
        """
        Add a volatility forecasting model
        
        Parameters:
        -----------
        name : str
            Name of the model
        model_func : callable
            Function that produces volatility forecasts
        is_benchmark : bool
            Whether this model is the benchmark model
        """
        self.models[name] = model_func
        self.model_names.append(name)
        
        if is_benchmark:
            self.benchmark_model = name
            
        # Initialize weights equally
        n_models = len(self.model_names)
        self.weights = np.ones(n_models) / n_models
    
    def _calculate_information_quality(self, log_rv, volume):
        """
        Calculate the information quality measure λt
        
        Parameters:
        -----------
        log_rv : array-like
            Log of realized volatility
        volume : array-like
            Detrended volume time series
            
        Returns:
        --------
        lambda_t : float
            Information quality measure λt in (0,1]
        """
        # Handle empty arrays or NaN values
        if len(log_rv) == 0 or np.isnan(log_rv).any() or np.isnan(volume).any():
            return 1.0
            
        # If first observation, initialize and return full confidence
        if len(self.information_flow) < 2:
            if len(log_rv) > 0:
                self.information_flow.append(log_rv[-1])
            return 1.0
        
        # Simple AR(1) model for information flow
        prev_info = self.information_flow[-1]
        expected_info = self.memory * prev_info
        
        # Standardize inputs safely
        log_rv_std = 0
        if len(log_rv) > 1:
            log_rv_std_val = np.std(log_rv)
            if log_rv_std_val > 0:
                log_rv_std = (log_rv[-1] - np.mean(log_rv)) / log_rv_std_val
        
        vol_std = 0
        if len(volume) > 1:
            vol_std_val = np.std(volume)
            if vol_std_val > 0:
                vol_std = (volume[-1] - np.mean(volume)) / vol_std_val
        
        # Current information estimate (weighted average)
        current_info = 0.7 * log_rv_std + 0.3 * vol_std
        self.information_flow.append(current_info)
        
        # Calculate innovation
        innovation = current_info - expected_info
        
        # Estimate the standard deviation of innovations
        innovations = np.diff(self.information_flow)
        sigma_innovation = np.std(innovations) if len(innovations) > 5 else 0.1
        
        # Avoid division by zero
        if sigma_innovation == 0:
            sigma_innovation = 0.1
        
        # Calculate λt using equation (8) from the paper
        lambda_t = 2 - 2 * norm.cdf(abs(innovation) / sigma_innovation)
        lambda_t = max(0.01, min(lambda_t, 1.0))  # Ensure value is in (0,1]
        
        return lambda_t
        
    def _calculate_model_performance(self, forecasts, realized, realized_var=None):
        """
        Calculate the performance z-scores for each model
        
        Parameters:
        -----------
        forecasts : dict
            Dictionary of model forecasts
        realized : float
            Realized volatility value
        realized_var : float, optional
            Variance of the realized volatility estimator
            
        Returns:
        --------
        z_vals : dict
            Performance measure for each model
        """
        # Use default variance if not provided
        if realized_var is None or realized_var <= 0:
            realized_var = max(0.01 * realized**2, 1e-8)
            
        # Calculate the standardized forecast errors
        z_vals = {}
        for model_name in self.model_names:
            forecast = forecasts[model_name]
            std_err = np.sqrt(realized_var)
            if std_err == 0:
                std_err = 0.0001
            z = norm.cdf(abs(realized - forecast) / std_err)
            z_vals[model_name] = z
            
        return z_vals
        
    def _calculate_momentum_weights(self, z_vals):
        """
        Calculate weights based on model performance
        
        Parameters:
        -----------
        z_vals : dict
            Performance measure for each model
            
        Returns:
        --------
        momentum_weights : ndarray
            Momentum weights for each model
        """
        # Set default benchmark if none specified
        if self.benchmark_model is None and len(self.model_names) > 0:
            self.benchmark_model = self.model_names[0]
            
        b_idx = self.model_names.index(self.benchmark_model)
        benchmark_z = z_vals[self.benchmark_model]
        
        # Initialize indicators and weights
        indicators = np.zeros(len(self.model_names))
        weights = np.zeros(len(self.model_names))
        
        # Mark models that outperform the benchmark
        for i, model_name in enumerate(self.model_names):
            model_z = z_vals[model_name]
            indicators[i] = 1 if benchmark_z - model_z > self.threshold else 0
        
        # If no model outperforms the benchmark, use benchmark only
        if np.sum(indicators) == 0:
            weights[b_idx] = 1.0
            return weights
            
        if self.strategy == "winner":
            # Winner-takes-all strategy
            if np.sum(indicators) > 0:
                outperforming = [i for i, indicator in enumerate(indicators) if indicator == 1]
                if len(outperforming) > 0:
                    z_values = [z_vals[self.model_names[i]] for i in outperforming]
                    best_idx = outperforming[np.argmin(z_values)]
                    weights[best_idx] = 1.0
                else:
                    weights[b_idx] = 1.0
            else:
                weights[b_idx] = 1.0
                
        elif self.strategy == "proportional":
            # Proportional strategy
            excess_performance = np.zeros(len(self.model_names))
            
            for i, model_name in enumerate(self.model_names):
                if model_name != self.benchmark_model and indicators[i] == 1:
                    excess_performance[i] = benchmark_z - z_vals[model_name]
            
            total_excess = np.sum(excess_performance)
            
            if total_excess == 0:
                weights[b_idx] = 1.0
            elif total_excess < 0.5:
                for i in range(len(self.model_names)):
                    if i != b_idx:
                        weights[i] = 2 * excess_performance[i]
                weights[b_idx] = max(0, 1 - np.sum(weights))
            else:
                for i in range(len(self.model_names)):
                    if excess_performance[i] > 0:
                        weights[i] = excess_performance[i] / total_excess
        
        # Ensure weights sum to 1
        if np.sum(weights) == 0:
            weights[b_idx] = 1.0
        else:
            weights = weights / np.sum(weights)
            
        return weights
    
    def update_weights(self, forecasts, realized_vol, realized_var, log_rv, volume):
        """
        Update model weights based on performance and information quality
        
        Parameters:
        -----------
        forecasts : dict
            Dictionary of model forecasts
        realized_vol : float
            Realized volatility value
        realized_var : float
            Variance of the realized volatility estimator
        log_rv : array-like
            Log of realized volatility time series
        volume : array-like
            Detrended volume time series
            
        Returns:
        --------
        weights : ndarray
            Updated model weights
        lambda_t : float
            Information quality measure
        """
        # Ensure positive values
        realized_vol = max(realized_vol, 1e-8)
        realized_var = max(realized_var, 1e-8)
        
        # Make log_rv valid
        valid_log_rv = np.array(log_rv).copy()
        valid_log_rv[valid_log_rv <= 0] = 1e-8
        valid_log_rv = np.log(valid_log_rv)
        
        # Calculate model performance
        z_vals = self._calculate_model_performance(forecasts, realized_vol, realized_var)
        
        # Calculate information quality
        lambda_t = self._calculate_information_quality(valid_log_rv, volume)
        
        # Calculate momentum weights
        f_t = self._calculate_momentum_weights(z_vals)
        
        # Initialize unconditional weights (prior beliefs)
        f_0 = np.zeros(len(self.model_names))
        b_idx = self.model_names.index(self.benchmark_model)
        f_0[b_idx] = 1.0  # Unconditional weight is 1 for benchmark
        
        # Calculate information-adjusted weights using equation (6)
        p_t = (1 - lambda_t) * f_0 + lambda_t * f_t
        
        # Update weights using exponential smoothing
        self.weights = self.memory * self.weights + (1 - self.memory) * p_t
        self.weights = self.weights / np.sum(self.weights)  # Normalize
        
        return self.weights, lambda_t
        
    def forecast_combination(self, forecasts):
        """
        Combine volatility forecasts using current weights
        
        Parameters:
        -----------
        forecasts : dict
            Dictionary of model forecasts
            
        Returns:
        --------
        combined_forecast : float
            Combined volatility forecast
        """
        combined_forecast = 0
        for i, model_name in enumerate(self.model_names):
            combined_forecast += self.weights[i] * forecasts[model_name]
        return combined_forecast


class VolatilityTradingStrategy:
    """
    Trading strategy that uses volatility forecasts to make trading decisions
    """
    
    def __init__(self, lookback_window=252, vol_combiner=None, target_volatility=0.15,
                 vol_threshold=0.2, use_info_quality=True, stop_loss_pct=0.05):
        """
        Initialize the volatility trading strategy
        
        Parameters:
        -----------
        lookback_window : int
            Length of historical window to consider
        vol_combiner : VolatilityModelCombination or None
            Volatility model combiner. If None, a new one is created.
        target_volatility : float
            Target annual volatility for position sizing
        vol_threshold : float
            Volatility threshold for abnormal volatility detection
        use_info_quality : bool
            Whether to use information quality for position sizing
        stop_loss_pct : float
            Stop loss percentage
        """
        self.lookback_window = lookback_window
        self.target_volatility = target_volatility
        self.vol_threshold = vol_threshold
        self.use_info_quality = use_info_quality
        self.stop_loss_pct = stop_loss_pct
        
        # Initialize the volatility combiner if needed
        if vol_combiner is None:
            self.vol_combiner = VolatilityModelCombination(
                memory_parameter=0.95, 
                threshold=0.0, 
                strategy="proportional"
            )
        else:
            self.vol_combiner = vol_combiner
            
        # Strategy state variables
        self.position = 0  # Current position size (0-1)
        self.entry_price = None  # Price at which position was entered
        self.last_forecast = None  # Last volatility forecast
        self.last_lambda = None  # Last information quality
        
    def add_volatility_models(self, returns, realized_vol, volume=None):
        """
        Add volatility forecasting models to the combiner
        
        Parameters:
        -----------
        returns : ndarray
            Historical returns
        realized_vol : ndarray
            Historical realized volatility
        volume : ndarray or None
            Historical trading volume (can be None)
        """
        # If volume not provided, create dummy
        if volume is None:
            volume = np.ones_like(returns)
            
        # Detrend volume
        detrended_volume = self._detrend_volume(volume)
        
        # Add simplified models
        self.vol_combiner.add_model("SMA", lambda: self._simple_ma(realized_vol, window=22))
        self.vol_combiner.add_model("EMA", lambda: self._exp_ma(realized_vol, decay=0.94))
        self.vol_combiner.add_model("GARCH_Proxy", lambda: self._garch_proxy(returns, realized_vol))
        self.vol_combiner.add_model("LongMA", lambda: self._simple_ma(realized_vol, window=63), is_benchmark=True)
        
        # Add ARIMA model with proper error handling
        try:
            # Test if ARIMA works on this data
            test_forecast = self._arima_model(realized_vol)
            if not np.isnan(test_forecast) and test_forecast > 0:
                self.vol_combiner.add_model("ARIMA", lambda: self._arima_model(realized_vol))
        except:
            pass  # Skip ARIMA model if it fails
            
    def _detrend_volume(self, volume):
        """Remove linear trend from volume"""
        if len(volume) <= 2:
            return volume
            
        # Clean data
        clean_volume = np.array(volume).copy()
        clean_volume = np.nan_to_num(clean_volume)
            
        try:
            x = np.arange(len(clean_volume))
            trend = sm.OLS(clean_volume, sm.add_constant(x)).fit()
            detrended = clean_volume - trend.predict(sm.add_constant(x))
            return detrended
        except:
            # If OLS fails, return original volume
            return clean_volume
    
    def _simple_ma(self, vol_data, window=22):
        """Simple moving average of volatility"""
        if len(vol_data) < 2:
            return 0.01
            
        # Clean data
        clean_vol = np.array(vol_data).copy()
        clean_vol = np.nan_to_num(clean_vol, nan=0.01)
        clean_vol[clean_vol <= 0] = 0.01
        
        window = min(window, len(clean_vol))
        return np.mean(clean_vol[-window:])
    
    def _exp_ma(self, vol_data, decay=0.94):
        """Exponential moving average of volatility"""
        if len(vol_data) < 2:
            return 0.01
        
        # Clean data
        clean_vol = np.array(vol_data).copy()
        clean_vol = np.nan_to_num(clean_vol, nan=0.01)
        clean_vol[clean_vol <= 0] = 0.01
        
        window = min(63, len(clean_vol))
        recent_vol = clean_vol[-window:]
        
        # Calculate weights
        weights = np.power(decay, np.arange(window-1, -1, -1))
        weights = weights / np.sum(weights)
        
        # Calculate weighted average
        return np.sum(weights * recent_vol)
    
    def _garch_proxy(self, returns, vol_data):
        """Simple GARCH(1,1) approximation without ML estimation"""
        if len(returns) < 5 or len(vol_data) < 1:
            return 0.01
        
        # Clean data
        clean_returns = np.array(returns).copy()
        clean_returns = np.nan_to_num(clean_returns, nan=0)
        
        clean_vol = np.array(vol_data).copy()
        clean_vol = np.nan_to_num(clean_vol, nan=0.01)
        clean_vol[clean_vol <= 0] = 0.01
        
        # Use most recent volatility as starting point
        last_vol = clean_vol[-1]
        
        # Update with latest return
        alpha, beta = 0.1, 0.85  # GARCH parameters
        omega = 0.01 * (1 - alpha - beta)  # Long-term variance component
        
        # Last squared return
        last_ret_sq = clean_returns[-1]**2
        
        # GARCH forecast
        forecast = omega + alpha * last_ret_sq + beta * last_vol**2
        
        return np.sqrt(forecast)
    
    def _arima_model(self, vol_data, window=63, order=(1,1,0)):
        """ARIMA model with error handling"""
        if len(vol_data) < 10:
            return 0.01
        
        # Clean data
        clean_vol = np.array(vol_data[-window:]).copy()
        clean_vol = np.nan_to_num(clean_vol, nan=0.01)
        clean_vol[clean_vol <= 0] = 0.01
        
        # Log transform
        log_vol = np.log(clean_vol)
        
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # Fit ARIMA with limited iterations
                model = ARIMA(log_vol, order=order)
                results = model.fit(disp=False, maxiter=50)
                forecast = results.forecast(steps=1)
                
                # Transform back and validate
                vol_forecast = np.exp(forecast[0])
                if np.isnan(vol_forecast) or vol_forecast <= 0:
                    return np.mean(clean_vol)  # Fallback if invalid
                return vol_forecast
        except:
            # Fallback to simple average
            return np.mean(clean_vol)
        
    def generate_signal(self, returns, realized_vol, volume=None, current_price=None, threshold_multiplier=2.0):
        """
        Generate trading signal based on volatility forecast
        
        Parameters:
        -----------
        returns : ndarray
            Historical returns
        realized_vol : ndarray
            Historical realized volatility
        volume : ndarray or None
            Historical trading volume (can be None)
        current_price : float or None
            Current price for stop loss check
        threshold_multiplier : float
            Multiplier for historical volatility to determine abnormal volatility
            
        Returns:
        --------
        signal : dict
            Dictionary with trading signal information
        """
        # Clean data
        clean_returns = np.array(returns).copy()
        clean_returns = np.nan_to_num(clean_returns, nan=0)
        
        clean_vol = np.array(realized_vol).copy()
        clean_vol = np.nan_to_num(clean_vol, nan=0.01)
        clean_vol[clean_vol <= 0] = 0.01
        
        # If there's no volume data, create a dummy one
        if volume is None:
            volume = np.ones_like(clean_returns)
            
        # Detrend volume
        detrended_volume = self._detrend_volume(volume)
        
        # Calculate individual model forecasts
        forecasts = {
            "SMA": self._simple_ma(clean_vol, window=22),
            "EMA": self._exp_ma(clean_vol, decay=0.94),
            "GARCH_Proxy": self._garch_proxy(clean_returns, clean_vol),
            "LongMA": self._simple_ma(clean_vol, window=63)
        }
        
        # Add ARIMA forecast if model was added
        if "ARIMA" in self.vol_combiner.model_names:
            try:
                forecasts["ARIMA"] = self._arima_model(clean_vol)
            except:
                # Use mean of other forecasts if ARIMA fails
                other_values = [v for k, v in forecasts.items()]
                forecasts["ARIMA"] = np.mean(other_values)
        
        # If we have previous forecasts, update the weights
        lambda_t = 1.0
        if self.last_forecast is not None and len(clean_vol) > 0:
            # Use the previous day's realized volatility for updating
            prev_rv = clean_vol[-1]
            prev_var = 0.01 * prev_rv**2  # Approximation of realized vol variance
            
            # Update weights
            try:
                _, lambda_t = self.vol_combiner.update_weights(
                    forecasts=forecasts,
                    realized_vol=prev_rv,
                    realized_var=prev_var,
                    log_rv=np.log(clean_vol),
                    volume=detrended_volume
                )
            except Exception as e:
                print(f"Error updating weights: {e}")
                lambda_t = 1.0
        
        # Generate combined forecast
        vol_forecast = self.vol_combiner.forecast_combination(forecasts)
        self.last_forecast = vol_forecast
        self.last_lambda = lambda_t
        
        # Calculate historical volatility for comparison
        hist_vol = np.std(clean_returns[-min(63, len(clean_returns)):]) if len(clean_returns) > 0 else 0.01
        if hist_vol == 0:
            hist_vol = 0.01
        
        # Determine if current volatility is abnormal
        is_abnormal = vol_forecast > (threshold_multiplier * hist_vol)
        
        # Base position size calculation
        position_size = self.target_volatility / (vol_forecast * np.sqrt(252))  # Annualized
        
        # Adjust position size based on information quality if enabled
        if self.use_info_quality:
            position_size *= lambda_t  # Reduce position if information quality is low
        
        # Cap position size at 1.0 (100% allocation)
        position_size = min(1.0, max(0.0, position_size))  # Ensure between 0 and 1
        
        # Default is to maintain current position
        action = "HOLD"
        new_position = self.position
        
        # Check for stop loss
        if self.position > 0 and current_price is not None and self.entry_price is not None:
            price_change = (current_price / self.entry_price) - 1
            if price_change < -self.stop_loss_pct:
                action = "SELL"
                new_position = 0
                
        # If no stop loss triggered, determine action based on volatility
        if action == "HOLD":
            # If already in a position
            if self.position > 0:
                # Adjust position size if significant change
                if abs(position_size - self.position) > 0.1:
                    if position_size > self.position:
                        action = "BUY"  # Increase position
                    else:
                        action = "SELL"  # Decrease position
                    new_position = position_size
            else:
                # If not in a position, enter if forecast suggests it
                if position_size > 0.1:  # Minimum position threshold
                    action = "BUY"
                    new_position = position_size
        
        # Prepare signal dictionary
        signal = {
            "action": action,
            "position_size": new_position,
            "vol_forecast": vol_forecast,
            "hist_vol": hist_vol,
            "is_abnormal": is_abnormal,
            "lambda": lambda_t,
            "model_weights": {name: weight for name, weight in zip(self.vol_combiner.model_names, self.vol_combiner.weights)}
        }
        
        return signal
    
    def update_position(self, signal, current_price):
        """
        Update the current position based on the signal
        
        Parameters:
        -----------
        signal : dict
            Dictionary with trading signal information
        current_price : float
            Current price
            
        Returns:
        --------
        trade_info : dict
            Information about the trade executed
        """
        prev_position = self.position
        self.position = signal["position_size"]
        
        # Record entry price if entering a position
        if prev_position == 0 and self.position > 0:
            self.entry_price = current_price
            
        # Reset entry price if exiting a position
        if prev_position > 0 and self.position == 0:
            self.entry_price = None
        
        # Calculate trade size
        trade_size = self.position - prev_position
        
        trade_info = {
            "action": signal["action"],
            "trade_size": trade_size,
            "new_position": self.position,
            "entry_price": self.entry_price
        }
        
        return trade_info


def calculate_realized_volatility(prices, window=21):
    """
    Calculate realized volatility from price data
    
    Parameters:
    -----------
    prices : ndarray
        Historical prices
    window : int
        Window for realized volatility calculation
        
    Returns:
    --------
    realized_vol : ndarray
        Realized volatility series
    """
    # Clean prices
    clean_prices = np.array(prices).copy()
    clean_prices = np.nan_to_num(clean_prices, nan=0)
    
    # Ensure positive prices
    clean_prices[clean_prices <= 0] = 1.0
    
    # Calculate returns
    returns = np.zeros(len(clean_prices)-1)
    for i in range(1, len(clean_prices)):
        returns[i-1] = np.log(clean_prices[i] / clean_prices[i-1])
    
    # Replace NaN and infinite values
    returns = np.nan_to_num(returns, nan=0, posinf=0, neginf=0)
    
    # Calculate realized volatility
    realized_vol = np.zeros(len(returns))
    
    for i in range(len(returns)):
        if i < window:
            realized_vol[i] = np.std(returns[:i+1]) if i > 0 else 0.01
        else:
            realized_vol[i] = np.std(returns[i-window+1:i+1])
    
    # Ensure positive volatility
    realized_vol[realized_vol <= 0] = 0.01
    
    return realized_vol


def simulate_financial_data(n_days=1000):
    """
    Simulate financial time series data with realistic features
    
    Parameters:
    -----------
    n_days : int
        Number of days to simulate
        
    Returns:
    --------
    data : DataFrame
        Simulated financial data
    """
    # Generate dates
    start_date = datetime(2010, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(n_days)]
    
    # Create dataframe
    data = pd.DataFrame(index=dates)
    
    # Simulate volatility process (GARCH-like)
    volatility = np.zeros(n_days)
    volatility[0] = 0.01
    
    for i in range(1, n_days):
        # Volatility clustering (persistent)
        volatility[i] = 0.85 * volatility[i-1] + 0.15 * 0.01
        
        # Random shocks to volatility
        volatility[i] += np.random.normal(0, 0.002)
        
        # Ensure non-negative
        volatility[i] = max(0.001, volatility[i])
    
    # Add some volatility regimes
    for i in range(5):
        regime_start = np.random.randint(0, n_days-50)
        regime_length = np.random.randint(20, 50)
        regime_mult = np.random.uniform(1.5, 3.0)
        volatility[regime_start:regime_start+regime_length] *= regime_mult
    
    # Simulate returns
    returns = np.zeros(n_days)
    
    for i in range(n_days):
        # Base return
        returns[i] = np.random.normal(0.0001, volatility[i])
        
        # Add jumps
        if np.random.random() < 0.05:
            jump_direction = np.random.choice([-1, 1])
            returns[i] += jump_direction * np.random.exponential(0.02)
        
        # Add trend component
        returns[i] += 0.0001 * i / n_days
    
    # Convert returns to prices
    prices = 100 * np.cumprod(1 + returns)
    
    # Simulate trading volume
    volume = np.zeros(n_days)
    
    for i in range(n_days):
        # Base volume with trend
        volume[i] = 100000 * (1 + 0.0005 * i)
        
        # Relate volume to volatility
        volume[i] *= (1 + 0.7 * (volatility[i] / 0.01 - 1))
        
        # Add noise
        volume[i] *= np.random.lognormal(0, 0.3)
    
    # Calculate realized volatility
    realized_vol = calculate_realized_volatility(prices)
    
    # Create DataFrame
    data['Close'] = prices
    data['Return'] = returns
    data['Volume'] = volume.astype(int)
    data['RealizedVol'] = np.append(realized_vol, realized_vol[-1] if len(realized_vol) > 0 else 0.01)
    
    return data


def backtest_strategy_with_simulated_data(n_days=1000, target_volatility=0.15, 
                                        vol_threshold=0.2, use_info_quality=True,
                                        stop_loss_pct=0.05, initial_capital=100000.0, 
                                        transaction_cost=0.001):
    """
    Backtest the volatility trading strategy with simulated data
    
    Parameters:
    -----------
    n_days : int
        Number of days to simulate
    target_volatility : float
        Target annual volatility for position sizing
    vol_threshold : float
        Volatility threshold for abnormal volatility detection
    use_info_quality : bool
        Whether to use information quality for position sizing
    stop_loss_pct : float
        Stop loss percentage
    initial_capital : float
        Initial capital
    transaction_cost : float
        Transaction cost as percentage
        
    Returns:
    --------
    results : dict
        Backtest results
    """
    # Simulate financial data
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        data = simulate_financial_data(n_days=n_days)
    
    # Initialize strategy
    strategy = VolatilityTradingStrategy(
        lookback_window=min(252, n_days//4),
        target_volatility=target_volatility,
        vol_threshold=vol_threshold,
        use_info_quality=use_info_quality,
        stop_loss_pct=stop_loss_pct
    )
    
    # Add volatility models
    strategy.add_volatility_models(
        returns=data['Return'].values,
        realized_vol=data['RealizedVol'].values,
        volume=data['Volume'].values
    )
    
    # Initialize results storage
    results = {
        'Date': [],
        'Price': [],
        'Position': [],
        'VolForecast': [],
        'HistVol': [],
        'Lambda': [],
        'PortfolioValue': [],
        'Cash': [],
        'Action': [],
        'Shares': [],
        'ModelWeights': []
    }
    
    # Initialize portfolio tracking
    portfolio_value = initial_capital
    cash = initial_capital
    shares = 0
    last_action_date = None
    min_holding_days = 5  # Minimum holding period in days
    
    # Minimum lookback
    min_lookback = min(63, n_days // 10)  # About 3 months or 10% of data
    
    # Backtest loop
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i in tqdm(range(min_lookback, len(data))):
            current_date = data.index[i]
            current_price = data['Close'].iloc[i]
            
            # Get data up to current point (excluding current)
            returns = data['Return'].values[:i]
            realized_vol = data['RealizedVol'].values[:i]
            volume = data['Volume'].values[:i]
            
            # Generate signal
            try:
                signal = strategy.generate_signal(
                    returns=returns,
                    realized_vol=realized_vol,
                    volume=volume,
                    current_price=current_price
                )
            except Exception as e:
                print(f"Error generating signal at index {i}: {e}")
                # Use default signal
                signal = {
                    "action": "HOLD",
                    "position_size": strategy.position,
                    "vol_forecast": 0.01,
                    "hist_vol": 0.01,
                    "is_abnormal": False,
                    "lambda": 1.0,
                    "model_weights": {name: 1.0/len(strategy.vol_combiner.model_names) for name in strategy.vol_combiner.model_names}
                }
            
            # Check if minimum holding period has passed
            can_trade = True
            if last_action_date is not None:
                days_since_last_action = (current_date - last_action_date).days
                if days_since_last_action < min_holding_days and signal['action'] != "HOLD":
                    can_trade = False
                    signal['action'] = "HOLD"  # Force hold
            
            # Update position if can trade
            if can_trade:
                trade_info = strategy.update_position(signal, current_price)
                
                # Execute trade if needed
                if trade_info['trade_size'] != 0:
                    # Calculate new shares
                    new_shares = int((trade_info['trade_size'] * portfolio_value) / current_price)
                    
                    # Adjust for transaction cost
                    transaction_amount = abs(new_shares * current_price)
                    cost = transaction_amount * transaction_cost
                    
                    # Update cash and shares
                    cash -= (new_shares * current_price + cost)
                    shares += new_shares
                    
                    # Update last action date
                    last_action_date = current_date
            else:
                trade_info = {
                    'action': 'HOLD',
                    'trade_size': 0,
                    'new_position': strategy.position
                }
            
            # Calculate portfolio value
            portfolio_value = cash + (shares * current_price)
            
            # Store results
            results['Date'].append(current_date)
            results['Price'].append(current_price)
            results['Position'].append(strategy.position)
            results['VolForecast'].append(signal['vol_forecast'])
            results['HistVol'].append(signal.get('hist_vol', 0.01))
            results['Lambda'].append(signal['lambda'])
            results['PortfolioValue'].append(portfolio_value)
            results['Cash'].append(cash)
            results['Action'].append(trade_info['action'])
            results['Shares'].append(shares)
            results['ModelWeights'].append(signal['model_weights'])
    
    # Convert results to DataFrame
    results_df = pd.DataFrame({
        'Date': results['Date'],
        'Price': results['Price'],
        'Position': results['Position'],
        'VolForecast': results['VolForecast'],
        'HistVol': results['HistVol'],
        'Lambda': results['Lambda'],
        'PortfolioValue': results['PortfolioValue'],
        'Cash': results['Cash'],
        'Action': results['Action'],
        'Shares': results['Shares']
    })
    
    # Calculate strategy returns
    results_df['Return'] = results_df['PortfolioValue'].pct_change().fillna(0)
    
    # Calculate benchmark returns (buy and hold)
    initial_price = results_df['Price'].iloc[0]
    benchmark_shares = initial_capital / initial_price
    results_df['BenchmarkValue'] = benchmark_shares * results_df['Price']
    results_df['BenchmarkReturn'] = results_df['BenchmarkValue'].pct_change().fillna(0)
    
    # Calculate cumulative returns
    results_df['CumulativeReturn'] = (1 + results_df['Return']).cumprod() - 1
    results_df['CumulativeBenchmarkReturn'] = (1 + results_df['BenchmarkReturn']).cumprod() - 1
    
    # Calculate performance metrics
    performance = calculate_performance_metrics(results_df)
    
    return results_df, performance, results['ModelWeights']


def calculate_performance_metrics(results_df):
    """
    Calculate performance metrics for a strategy
    
    Parameters:
    -----------
    results_df : DataFrame
        Backtest results
        
    Returns:
    --------
    metrics : dict
        Performance metrics
    """
    # Clean data
    daily_returns = results_df['Return'].replace([np.inf, -np.inf], np.nan).fillna(0)
    benchmark_returns = results_df['BenchmarkReturn'].replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Total return
    total_return = results_df['CumulativeReturn'].iloc[-1]
    benchmark_total_return = results_df['CumulativeBenchmarkReturn'].iloc[-1]
    
    # Annualized return
    years = len(results_df) / 252
    annualized_return = (1 + total_return) ** (1 / years) - 1
    benchmark_annualized_return = (1 + benchmark_total_return) ** (1 / years) - 1
    
    # Volatility
    daily_vol = daily_returns.std()
    annualized_vol = daily_vol * np.sqrt(252)
    benchmark_daily_vol = benchmark_returns.std()
    benchmark_annualized_vol = benchmark_daily_vol * np.sqrt(252)
    
    # Sharpe ratio (assuming risk-free rate of 0 for simplicity)
    sharpe_ratio = annualized_return / annualized_vol if annualized_vol > 0 else 0
    benchmark_sharpe = benchmark_annualized_return / benchmark_annualized_vol if benchmark_annualized_vol > 0 else 0
    
    # Maximum drawdown
    cum_returns = (1 + daily_returns).cumprod()
    running_max = cum_returns.cummax()
    drawdown = (cum_returns / running_max) - 1
    max_drawdown = drawdown.min()
    
    benchmark_cum_returns = (1 + benchmark_returns).cumprod()
    benchmark_running_max = benchmark_cum_returns.cummax()
    benchmark_drawdown = (benchmark_cum_returns / benchmark_running_max) - 1
    benchmark_max_drawdown = benchmark_drawdown.min()
    
    # Calmar ratio
    calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown < 0 else 0
    benchmark_calmar = benchmark_annualized_return / abs(benchmark_max_drawdown) if benchmark_max_drawdown < 0 else 0
    
    # Win rate
    win_rate = (daily_returns > 0).mean()
    benchmark_win_rate = (benchmark_returns > 0).mean()
    
    metrics = {
        'Total Return': total_return,
        'Benchmark Total Return': benchmark_total_return,
        'Annualized Return': annualized_return,
        'Benchmark Annualized Return': benchmark_annualized_return,
        'Annualized Volatility': annualized_vol,
        'Benchmark Annualized Volatility': benchmark_annualized_vol,
        'Sharpe Ratio': sharpe_ratio,
        'Benchmark Sharpe Ratio': benchmark_sharpe,
        'Maximum Drawdown': max_drawdown,
        'Benchmark Maximum Drawdown': benchmark_max_drawdown,
        'Calmar Ratio': calmar_ratio,
        'Benchmark Calmar Ratio': benchmark_calmar,
        'Win Rate': win_rate,
        'Benchmark Win Rate': benchmark_win_rate
    }
    
    return metrics


def plot_backtest_results(results_df, performance, model_weights):
    """
    Plot backtest results
    
    Parameters:
    -----------
    results_df : DataFrame
        Backtest results
    performance : dict
        Performance metrics
    model_weights : list
        Model weights history
    """
    # Set up figure with multiple subplots
    fig = plt.figure(figsize=(15, 20))
    
    # Plot portfolio value vs benchmark
    ax1 = fig.add_subplot(511)
    ax1.plot(results_df['Date'], results_df['PortfolioValue'], 'b-', label='Strategy')
    ax1.plot(results_df['Date'], results_df['BenchmarkValue'], 'r-', label='Buy & Hold')
    ax1.set_title('Portfolio Value')
    ax1.legend()
    ax1.grid(True)
    
    # Plot cumulative returns
    ax2 = fig.add_subplot(512)
    ax2.plot(results_df['Date'], results_df['CumulativeReturn'], 'b-', label='Strategy')
    ax2.plot(results_df['Date'], results_df['CumulativeBenchmarkReturn'], 'r-', label='Buy & Hold')
    ax2.set_title('Cumulative Return')
    ax2.legend()
    ax2.grid(True)
    
    # Plot position size
    ax3 = fig.add_subplot(513)
    ax3.plot(results_df['Date'], results_df['Position'], 'g-')
    ax3.set_title('Position Size')
    ax3.set_ylim(0, 1.1)
    ax3.grid(True)
    
    # Plot volatility forecast
    ax4 = fig.add_subplot(514)
    ax4.plot(results_df['Date'], results_df['VolForecast'], 'b-', label='Forecast')
    ax4.plot(results_df['Date'], results_df['HistVol'], 'r-', label='Historical')
    ax4.set_title('Volatility')
    ax4.legend()
    ax4.grid(True)
    
    # Plot information quality (lambda)
    ax5 = fig.add_subplot(515)
    ax5.plot(results_df['Date'], results_df['Lambda'], 'b-')
    ax5.set_title('Information Quality Measure (λt)')
    ax5.set_ylim(0, 1.1)
    ax5.grid(True)
    
    # Add performance table
    performance_text = '\n'.join([
        f"Performance Metrics:",
        f"Total Return: {performance['Total Return']:.2%} (Benchmark: {performance['Benchmark Total Return']:.2%})",
        f"Annualized Return: {performance['Annualized Return']:.2%} (Benchmark: {performance['Benchmark Annualized Return']:.2%})",
        f"Annualized Volatility: {performance['Annualized Volatility']:.2%} (Benchmark: {performance['Benchmark Annualized Volatility']:.2%})",
        f"Sharpe Ratio: {performance['Sharpe Ratio']:.2f} (Benchmark: {performance['Benchmark Sharpe Ratio']:.2f})",
        f"Maximum Drawdown: {performance['Maximum Drawdown']:.2%} (Benchmark: {performance['Benchmark Maximum Drawdown']:.2%})",
        f"Calmar Ratio: {performance['Calmar Ratio']:.2f} (Benchmark: {performance['Benchmark Calmar Ratio']:.2f})",
        f"Win Rate: {performance['Win Rate']:.2%} (Benchmark: {performance['Benchmark Win Rate']:.2%})"
    ])
    
    fig.text(0.5, 0.01, performance_text, ha='center', va='bottom', fontsize=12)
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.show()
    
    # Create DataFrame for model weights
    weights_data = {}
    model_names = list(model_weights[0].keys()) if model_weights and model_weights[0] else []
    
    for name in model_names:
        weights_data[name] = [weights.get(name, 0) for weights in model_weights]
    
    weights_df = pd.DataFrame(weights_data)
    
    # Plot model weights evolution
    plt.figure(figsize=(12, 6))
    for col in weights_df.columns:
        plt.plot(results_df['Date'], weights_df[col], label=col)
    plt.title('Model Weights Evolution')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    # Plot strategy drawdown
    cum_returns = (1 + results_df['Return'].fillna(0)).cumprod()
    running_max = cum_returns.cummax()
    drawdown = (cum_returns / running_max) - 1
    
    benchmark_cum_returns = (1 + results_df['BenchmarkReturn'].fillna(0)).cumprod()
    benchmark_running_max = benchmark_cum_returns.cummax()
    benchmark_drawdown = (benchmark_cum_returns / benchmark_running_max) - 1
    
    plt.figure(figsize=(12, 6))
    plt.plot(results_df['Date'], drawdown, 'b-', label='Strategy')
    plt.plot(results_df['Date'], benchmark_drawdown, 'r-', label='Benchmark')
    plt.title('Drawdown')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def compare_strategies(n_days=1000, runs=2):
    """
    Compare different strategy configurations
    
    Parameters:
    -----------
    n_days : int
        Number of days to simulate
    runs : int
        Number of runs for each configuration
        
    Returns:
    --------
    results : DataFrame
        Comparison results
    """
    configurations = [
        {'name': 'Base Strategy', 'target_vol': 0.15, 'use_info_quality': True, 'vol_threshold': 0.2},
        {'name': 'High Target Vol', 'target_vol': 0.20, 'use_info_quality': True, 'vol_threshold': 0.2},
        {'name': 'No Info Quality', 'target_vol': 0.15, 'use_info_quality': False, 'vol_threshold': 0.2},
        {'name': 'Low Vol Threshold', 'target_vol': 0.15, 'use_info_quality': True, 'vol_threshold': 0.1}
    ]
    
    results = []
    
    for config in configurations:
        for run in range(runs):
            print(f"Running {config['name']} - Run {run+1}/{runs}")
            
            # Run backtest
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                results_df, performance, _ = backtest_strategy_with_simulated_data(
                    n_days=n_days,
                    target_volatility=config['target_vol'],
                    use_info_quality=config['use_info_quality'],
                    vol_threshold=config['vol_threshold']
                )
            
            # Store results
            results.append({
                'Strategy': config['name'],
                'Run': run + 1,
                'Sharpe Ratio': performance['Sharpe Ratio'],
                'Total Return': performance['Total Return'],
                'Max Drawdown': performance['Maximum Drawdown'],
                'Annualized Return': performance['Annualized Return'],
                'Annualized Volatility': performance['Annualized Volatility'],
                'Calmar Ratio': performance['Calmar Ratio'],
                'Win Rate': performance['Win Rate']
            })
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Calculate average metrics across runs
    avg_results = results_df.groupby('Strategy').mean().drop('Run', axis=1)
    
    # Print summary
    print("\nStrategy Comparison:")
    for metric in ['Sharpe Ratio', 'Annualized Return', 'Max Drawdown', 'Calmar Ratio']:
        print(f"\n{metric}:")
        for strategy, value in avg_results[metric].sort_values(ascending=False).items():
            if 'Return' in metric or 'Drawdown' in metric:
                print(f"  {strategy}: {value:.2%}")
            else:
                print(f"  {strategy}: {value:.4f}")
    
    return results_df, avg_results


# Main execution
if __name__ == "__main__":
    # Suppress all warnings
    warnings.filterwarnings("ignore")
    
    # Run backtest with simulated data
    results_df, performance, model_weights = backtest_strategy_with_simulated_data(
        n_days=1000,
        target_volatility=0.15,
        vol_threshold=0.2,
        use_info_quality=True,
        stop_loss_pct=0.05
    )
    
    # Plot results
    plot_backtest_results(results_df, performance, model_weights)
    
    # Print performance summary
    print("\nPerformance Summary:")
    print("===================")
    
    for metric, value in performance.items():
        if 'Return' in metric or 'Drawdown' in metric:
            print(f"{metric}: {value:.2%}")
        else:
            print(f"{metric}: {value:.4f}")
    
    # Compare different strategy configurations
    comparison_results, avg_results = compare_strategies(n_days=1000, runs=2)
    
    # Print averages across runs
    print("\nAverage Strategy Performance:")
    print(avg_results)