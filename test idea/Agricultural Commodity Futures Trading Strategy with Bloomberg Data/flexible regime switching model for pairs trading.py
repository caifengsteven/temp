import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import pairwise_distances
import networkx as nx
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
import datetime as dt
import os
import warnings
warnings.filterwarnings('ignore')

# For Bloomberg data - when available
try:
    import pdblp
except ImportError:
    print("pdblp not installed. Using simulated data instead of Bloomberg.")

class LevyDrivenOUProcess:
    """
    Lévy-driven Ornstein-Uhlenbeck process with jumps.
    This class implements the model described in the paper for a single regime.
    """
    def __init__(self, theta=None, mu=None, sigma=None, lambda_jump=None, jump_size_variance=None):
        """
        Initialize model parameters.
        
        Parameters:
        -----------
        theta : float
            Mean-reversion speed
        mu : float or function
            Mean-reversion level (can be time-dependent function)
        sigma : float
            Diffusion parameter (volatility)
        lambda_jump : float
            Jump intensity (Poisson process parameter)
        jump_size_variance : float
            Variance of the jump size
        """
        self.theta = theta  # mean-reversion speed
        self.mu = mu  # mean-reversion level (can be time-dependent)
        self.sigma = sigma  # diffusion parameter
        self.lambda_jump = lambda_jump  # jump intensity
        self.jump_size_variance = jump_size_variance  # jump size variance
        
    def fit(self, X, dt=1/250/391):
        """
        Estimate model parameters from time series data.
        
        Parameters:
        -----------
        X : array-like
            Time series data
        dt : float
            Time step between observations
        
        Returns:
        --------
        self : LevyDrivenOUProcess
            Fitted model
        """
        X = np.array(X)
        n = len(X)
        
        # Time-dependent mean-reversion level (last two daily opening and closing values)
        # For simplicity, we use the mean of the data
        self.mu = np.mean(X)
        
        # Estimate mean-reversion speed using jump-filtered data
        beta = 0.499  # Upper limit of beta in (0, 1/2)
        threshold = beta * np.sqrt(dt)
        
        # Compute increments
        dX = np.diff(X)
        
        # Filter out jumps (increments larger than threshold)
        is_continuous = np.abs(dX) <= threshold
        
        # Compute mean-reversion speed using jump-filtered data
        if np.sum(is_continuous) > 0:
            numerator = np.sum((self.mu - X[:-1][is_continuous]) * dX[is_continuous])
            denominator = np.sum((self.mu - X[:-1][is_continuous])**2 * dt)
            if denominator != 0:
                self.theta = numerator / denominator
            else:
                self.theta = 0
        else:
            self.theta = 0
        
        # Estimate diffusion parameter using jump-filtered data
        if np.sum(is_continuous) > 0:
            squared_increments = dX[is_continuous]**2
            self.sigma = np.sqrt(np.mean(squared_increments) / dt)
        else:
            self.sigma = np.std(dX) / np.sqrt(dt)
        
        # Estimate jump parameters
        is_jump = ~is_continuous
        jump_count = np.sum(is_jump)
        self.lambda_jump = jump_count / ((n-1) * dt)  # Poisson process intensity
        
        # Estimate jump size variance (assuming zero mean)
        if jump_count > 0:
            jump_sizes = dX[is_jump]
            self.jump_size_variance = np.var(jump_sizes)
        else:
            self.jump_size_variance = 0
            
        return self
    
    def conditional_expectation(self, X_t, t_delta=1):
        """
        Compute conditional expectation E[X_{t+t_delta} | X_t]
        
        Parameters:
        -----------
        X_t : float
            Current value
        t_delta : float
            Time step
            
        Returns:
        --------
        float : Expected value at t+t_delta
        """
        return self.mu + (X_t - self.mu) * np.exp(-self.theta * t_delta)
    
    def conditional_variance(self, t_delta=1):
        """
        Compute conditional variance Var[X_{t+t_delta} | X_t]
        
        Parameters:
        -----------
        t_delta : float
            Time step
            
        Returns:
        --------
        float : Variance at t+t_delta
        """
        # Variance from diffusion part
        diff_var = (self.sigma**2 / (2 * self.theta)) * (1 - np.exp(-2 * self.theta * t_delta))
        
        # Variance from jump part
        jump_var = self.lambda_jump * self.jump_size_variance * t_delta
        
        return diff_var + jump_var
    
    def simulate(self, n_steps, dt=1/250/391, X0=0, random_state=None):
        """
        Simulate Lévy-driven OU process path
        
        Parameters:
        -----------
        n_steps : int
            Number of steps to simulate
        dt : float
            Time step
        X0 : float
            Initial value
        random_state : int or None
            Random seed
            
        Returns:
        --------
        array : Simulated path
        """
        if random_state is not None:
            np.random.seed(random_state)
            
        X = np.zeros(n_steps)
        X[0] = X0
        
        for t in range(1, n_steps):
            # Brownian motion part
            dW = np.random.normal(0, np.sqrt(dt))
            
            # Jump part (Poisson process)
            n_jumps = np.random.poisson(self.lambda_jump * dt)
            if n_jumps > 0:
                jump_sizes = np.random.normal(0, np.sqrt(self.jump_size_variance), size=n_jumps)
                jump_part = np.sum(jump_sizes)
            else:
                jump_part = 0
                
            # Combine in OU process
            X[t] = (X[t-1] + 
                    self.theta * (self.mu - X[t-1]) * dt + 
                    self.sigma * dW + 
                    jump_part)
            
        return X


class RegimeClassification:
    """
    Regime Classification Algorithm as described in the paper.
    """
    def __init__(self, model_class=LevyDrivenOUProcess, min_regime_size=0.15):
        """
        Initialize algorithm with parameters.
        
        Parameters:
        -----------
        model_class : class
            Class to fit in each regime
        min_regime_size : float
            Minimum proportion of data in each regime
        """
        self.model_class = model_class
        self.min_regime_size = min_regime_size
        self.models = []
        self.thresholds = []
        self.n_regimes = 1
        
    def fit(self, X, volatility_window=1955):
        """
        Run the regime classification algorithm.
        
        Parameters:
        -----------
        X : array-like
            Time series data
        volatility_window : int
            Window size for rolling volatility calculation
            
        Returns:
        --------
        self : RegimeClassification
            Fitted model
        """
        X = np.array(X)
        n = len(X)
        
        # Initialize with one regime
        single_model = self.model_class().fit(X)
        
        # Calculate conditional least squares error (CLS)
        cls1 = self._calculate_cls(X, [single_model])
        
        # Calculate BIC for one regime
        m1 = 5  # number of parameters in the model (theta, mu, sigma, lambda, jump_var)
        bic1 = n * np.log(cls1 / n) + m1 * np.log(n)
        
        best_bic = bic1
        self.n_regimes = 1
        self.models = [single_model]
        self.thresholds = []
        
        # Calculate rolling volatility
        if len(X) > volatility_window:
            rolling_vol = self._calculate_rolling_volatility(X, window=volatility_window)
        else:
            # If data is too short, use full data volatility
            rolling_vol = np.std(X) * np.ones(len(X))
        
        # Try increasing the number of regimes
        r = 2
        continue_search = True
        
        while continue_search and r <= 4:  # Limit to max 4 regimes
            print(f"Testing {r} regimes...")
            
            # Find best thresholds for r regimes
            best_thresholds, best_bicr = self._find_best_thresholds(X, rolling_vol, r)
            
            # If BIC improves, update best model
            if best_bicr < best_bic:
                best_bic = best_bicr
                self.n_regimes = r
                self.thresholds = best_thresholds
                
                # Classify data into regimes
                regimes = self._classify_data(rolling_vol, best_thresholds)
                
                # Fit models to each regime
                self.models = []
                for regime_idx in range(r):
                    regime_data = X[regimes == regime_idx]
                    if len(regime_data) > 0:
                        model = self.model_class().fit(regime_data)
                        self.models.append(model)
                    else:
                        # If a regime is empty, use the model from one regime
                        self.models.append(single_model)
                
                r += 1
            else:
                continue_search = False
                
        print(f"Best number of regimes: {self.n_regimes}")
        return self
    
    def _calculate_rolling_volatility(self, X, window=1955):
        """Calculate rolling volatility on the time series."""
        rolling_vol = np.zeros(len(X))
        for i in range(window, len(X)+1):
            rolling_vol[i-1] = np.std(X[i-window:i])
        
        # Fill initial values with the first calculated volatility
        rolling_vol[:window-1] = rolling_vol[window-1]
        
        return rolling_vol
    
    def _classify_data(self, volatility, thresholds):
        """Classify data into regimes based on volatility and thresholds."""
        n = len(volatility)
        regimes = np.zeros(n, dtype=int)
        
        # Sort thresholds to ensure correct classification
        sorted_thresholds = np.sort(thresholds)
        
        for i in range(len(volatility)):
            regime = 0
            for t in sorted_thresholds:
                if volatility[i] >= t:
                    regime += 1
            regimes[i] = regime
            
        return regimes
    
    def _find_best_thresholds(self, X, volatility, r):
        """
        Find best thresholds for r regimes using a smart grid search.
        """
        n = len(X)
        
        # Set up initial grid
        vol_range = np.linspace(np.min(volatility), np.max(volatility), 10)
        
        # Require at least min_regime_size in each regime
        min_obs = int(self.min_regime_size * n)
        
        best_bic = np.inf
        best_thresholds = None
        
        # Start with a wide search
        if r == 2:
            # For 2 regimes, we need 1 threshold
            for c1 in vol_range:
                regimes = self._classify_data(volatility, [c1])
                
                # Check if each regime has enough observations
                regime_counts = np.bincount(regimes)
                if len(regime_counts) < r or np.any(regime_counts < min_obs):
                    continue
                
                models = []
                for regime_idx in range(r):
                    regime_data = X[regimes == regime_idx]
                    models.append(self.model_class().fit(regime_data))
                
                cls = self._calculate_cls(X, models, regimes)
                m = 5 * r  # number of parameters
                bic = n * np.log(cls / n) + m * np.log(n)
                
                if bic < best_bic:
                    best_bic = bic
                    best_thresholds = [c1]
        
        elif r == 3:
            # For 3 regimes, we need 2 thresholds (c1 < c2)
            for c1 in vol_range[:-1]:
                for c2 in vol_range[vol_range > c1]:
                    regimes = self._classify_data(volatility, [c1, c2])
                    
                    # Check if each regime has enough observations
                    regime_counts = np.bincount(regimes)
                    if len(regime_counts) < r or np.any(regime_counts < min_obs):
                        continue
                    
                    models = []
                    for regime_idx in range(r):
                        regime_data = X[regimes == regime_idx]
                        models.append(self.model_class().fit(regime_data))
                    
                    cls = self._calculate_cls(X, models, regimes)
                    m = 5 * r  # number of parameters
                    bic = n * np.log(cls / n) + m * np.log(n)
                    
                    if bic < best_bic:
                        best_bic = bic
                        best_thresholds = [c1, c2]
        
        elif r == 4:
            # For 4 regimes, we need 3 thresholds (c1 < c2 < c3)
            # This is a simplified grid search - in practice, the smart grid approach
            # would be more efficient but more complex to implement
            for c1 in vol_range[:-2]:
                for c2 in vol_range[(vol_range > c1) & (vol_range < vol_range[-1])]:
                    for c3 in vol_range[vol_range > c2]:
                        regimes = self._classify_data(volatility, [c1, c2, c3])
                        
                        # Check if each regime has enough observations
                        regime_counts = np.bincount(regimes)
                        if len(regime_counts) < r or np.any(regime_counts < min_obs):
                            continue
                        
                        models = []
                        for regime_idx in range(r):
                            regime_data = X[regimes == regime_idx]
                            models.append(self.model_class().fit(regime_data))
                        
                        cls = self._calculate_cls(X, models, regimes)
                        m = 5 * r  # number of parameters
                        bic = n * np.log(cls / n) + m * np.log(n)
                        
                        if bic < best_bic:
                            best_bic = bic
                            best_thresholds = [c1, c2, c3]
        
        return best_thresholds, best_bic
    
    def _calculate_cls(self, X, models, regimes=None):
        """
        Calculate conditional least squares error.
        
        Parameters:
        -----------
        X : array-like
            Time series data
        models : list
            List of fitted models for each regime
        regimes : array-like or None
            Regime assignments (if None, assume one regime)
            
        Returns:
        --------
        float : Conditional least squares error
        """
        if regimes is None:
            # Single regime
            model = models[0]
            mse = 0
            for t in range(1, len(X)):
                predicted = model.conditional_expectation(X[t-1])
                mse += (X[t] - predicted) ** 2
        else:
            # Multiple regimes
            mse = 0
            for t in range(1, len(X)):
                regime = regimes[t-1]
                model = models[regime]
                predicted = model.conditional_expectation(X[t-1])
                mse += (X[t] - predicted) ** 2
                
        return mse / (len(X) - 1)
    
    def predict_regime(self, volatility):
        """
        Predict regime based on volatility.
        
        Parameters:
        -----------
        volatility : float
            Volatility value
            
        Returns:
        --------
        int : Predicted regime
        """
        if not self.thresholds:
            return 0
        
        regime = 0
        for t in sorted(self.thresholds):
            if volatility >= t:
                regime += 1
                
        return regime
    
    def conditional_expectation(self, X_t, volatility, t_delta=1):
        """
        Compute conditional expectation based on current value and volatility.
        
        Parameters:
        -----------
        X_t : float
            Current value
        volatility : float
            Current volatility
        t_delta : float
            Time step
            
        Returns:
        --------
        float : Expected value
        """
        regime = self.predict_regime(volatility)
        # Ensure regime is within bounds
        regime = min(regime, len(self.models) - 1)
        model = self.models[regime]
        return model.conditional_expectation(X_t, t_delta)


class FlexibleRegimeSwitchingPairsTrading:
    """
    Implementation of the flexible regime switching model for pairs trading
    as described in the paper by Endres and Stübinger (2019).
    """
    def __init__(self, formation_days=30, trading_days=5, top_pairs=10, 
                 bollinger_k=0.5, transaction_cost=0.0020, min_regime_size=0.15):
        """
        Initialize the pairs trading strategy.
        
        Parameters:
        -----------
        formation_days : int
            Number of days for formation period
        trading_days : int
            Number of days for trading period
        top_pairs : int
            Number of top pairs to select for trading
        bollinger_k : float
            Multiplier for Bollinger bands (k*sigma)
        transaction_cost : float
            Transaction cost per half-turn (e.g., 0.0005 for 5 bps)
        min_regime_size : float
            Minimum proportion of data in each regime
        """
        self.formation_days = formation_days
        self.trading_days = trading_days
        self.top_pairs = top_pairs
        self.bollinger_k = bollinger_k
        self.transaction_cost = transaction_cost
        self.min_regime_size = min_regime_size
        
        self.pairs = []
        self.models = {}
        self.trading_signals = {}
        self.positions = {}
        self.returns = []
        
    def calculate_spread(self, price_a, price_b):
        """
        Calculate spread according to equation (1) in the paper.
        
        Parameters:
        -----------
        price_a : array-like
            Price series of stock A
        price_b : array-like
            Price series of stock B
            
        Returns:
        --------
        array : Spread series
        """
        # Ensure we're working with numpy arrays
        price_a = np.array(price_a)
        price_b = np.array(price_b)
        
        # Calculate log returns from initial price
        log_a = np.log(price_a / price_a[0])
        log_b = np.log(price_b / price_b[0])
        
        # Calculate spread
        spread = log_a - log_b
        
        return spread
    
    def select_pairs(self, price_data, industries=None, minutes_per_day=391):
        """
        Select pairs based on formation period data.
        
        Parameters:
        -----------
        price_data : dict
            Dictionary of price series for each stock
        industries : dict or None
            Dictionary mapping stock symbols to industry
        minutes_per_day : int
            Number of minutes in a trading day
            
        Returns:
        --------
        list : Selected pairs for trading
        """
        print("Starting pair selection...")
        formation_length = self.formation_days * minutes_per_day
        
        # Create all possible pairs
        stocks = list(price_data.keys())
        n_stocks = len(stocks)
        
        # To store pair metrics
        pair_metrics = []
        
        for i in range(n_stocks):
            for j in range(i+1, n_stocks):
                stock_a = stocks[i]
                stock_b = stocks[j]
                
                # Check if both stocks have enough data
                if (len(price_data[stock_a]) < formation_length or 
                    len(price_data[stock_b]) < formation_length):
                    continue
                
                # Check if both stocks are from the same industry (if industries are provided)
                if industries is not None:
                    if stock_a not in industries or stock_b not in industries:
                        continue
                    if industries[stock_a] != industries[stock_b]:
                        continue
                
                # Calculate spread
                price_a = price_data[stock_a][-formation_length:]
                price_b = price_data[stock_b][-formation_length:]
                spread = self.calculate_spread(price_a, price_b)
                
                # Fit regime switching model
                model = RegimeClassification(
                    model_class=LevyDrivenOUProcess,
                    min_regime_size=self.min_regime_size
                )
                model.fit(spread)
                
                # Calculate metrics
                theta = np.mean([m.theta for m in model.models])
                sigma = np.mean([m.sigma for m in model.models])
                lambda_jump = np.mean([m.lambda_jump for m in model.models])
                jump_var = np.mean([m.jump_size_variance for m in model.models])
                
                # Store pair with metrics
                pair_metrics.append({
                    'pair': (stock_a, stock_b),
                    'theta': theta,
                    'sigma': sigma,
                    'lambda_jump': lambda_jump,
                    'jump_var': jump_var,
                    'model': model
                })
                
                print(f"Analyzed pair: {stock_a}-{stock_b}, theta: {theta:.4f}, sigma: {sigma:.4f}")
        
        # Rank pairs based on metrics
        # Higher theta (faster mean reversion), sigma, lambda_jump, and jump_var are preferred
        if pair_metrics:
            # Create rankings for each metric
            theta_rank = pd.Series([p['theta'] for p in pair_metrics]).rank(ascending=False)
            sigma_rank = pd.Series([p['sigma'] for p in pair_metrics]).rank(ascending=False)
            lambda_rank = pd.Series([p['lambda_jump'] for p in pair_metrics]).rank(ascending=False)
            jump_var_rank = pd.Series([p['jump_var'] for p in pair_metrics]).rank(ascending=False)
            
            # Sum ranks to get total ranking
            total_rank = theta_rank + sigma_rank + lambda_rank + jump_var_rank
            
            # Select top pairs
            top_indices = total_rank.argsort()[:self.top_pairs]
            selected_pairs = []
            selected_models = {}
            
            for idx in top_indices:
                pair = pair_metrics[idx]['pair']
                model = pair_metrics[idx]['model']
                selected_pairs.append(pair)
                selected_models[pair] = model
                
            self.pairs = selected_pairs
            self.models = selected_models
            
            print(f"Selected {len(selected_pairs)} pairs for trading")
            return selected_pairs
        else:
            print("No valid pairs found")
            self.pairs = []
            self.models = {}
            return []
    
    def generate_trading_signals(self, price_data, minutes_per_day=391):
        """
        Generate trading signals for the trading period.
        
        Parameters:
        -----------
        price_data : dict
            Dictionary of price series for each stock
        minutes_per_day : int
            Number of minutes in a trading day
            
        Returns:
        --------
        dict : Trading signals for each pair
        """
        formation_length = self.formation_days * minutes_per_day
        trading_length = self.trading_days * minutes_per_day
        
        for pair in self.pairs:
            stock_a, stock_b = pair
            
            # Check if we have enough data
            if (len(price_data[stock_a]) < formation_length + trading_length or
                len(price_data[stock_b]) < formation_length + trading_length):
                print(f"Not enough data for pair {pair}. Skipping.")
                continue
            
            # Get price data for trading period
            price_a = price_data[stock_a][-(formation_length + trading_length):]
            price_b = price_data[stock_b][-(formation_length + trading_length):]
            
            # Calculate spread
            spread = self.calculate_spread(price_a, price_b)
            
            # Split into formation and trading periods
            formation_spread = spread[:formation_length]
            trading_spread = spread[formation_length:]
            
            # Calculate trading signals
            model = self.models[pair]
            
            # Calculate rolling volatility for the trading period
            vol_window = min(1955, formation_length)  # 5 trading days, or formation length if shorter
            rolling_vol = np.zeros(len(trading_spread))
            
            # Initial volatility based on formation period
            initial_vol = np.std(formation_spread[-vol_window:])
            
            # For each trading minute, calculate volatility and trading bands
            signals = np.zeros(len(trading_spread))
            
            for t in range(len(trading_spread)):
                # Update volatility with recent data
                if t < vol_window:
                    # Use formation data + available trading data
                    vol_data = np.concatenate([
                        formation_spread[-(vol_window-t):],
                        trading_spread[:t+1]
                    ])
                    rolling_vol[t] = np.std(vol_data)
                else:
                    # Use only trading data
                    rolling_vol[t] = np.std(trading_spread[t-vol_window+1:t+1])
                
                # Determine regime
                regime = model.predict_regime(rolling_vol[t])
                # Make sure regime is within bounds
                regime = min(regime, len(model.models) - 1)
                current_model = model.models[regime]
                
                # Calculate mean-reversion level (equilibrium level)
                mu = current_model.mu
                
                # Calculate Bollinger bands
                upper_band = mu + self.bollinger_k * rolling_vol[t]
                lower_band = mu - self.bollinger_k * rolling_vol[t]
                
                # Generate trading signals
                if trading_spread[t] > upper_band:
                    signals[t] = -1  # Short stock A, long stock B
                elif trading_spread[t] < lower_band:
                    signals[t] = 1   # Long stock A, short stock B
                else:
                    # If already in a position, check if we should close it
                    if t > 0 and signals[t-1] != 0:
                        if signals[t-1] == 1 and trading_spread[t] >= mu:
                            signals[t] = 0  # Close position
                        elif signals[t-1] == -1 and trading_spread[t] <= mu:
                            signals[t] = 0  # Close position
                        else:
                            signals[t] = signals[t-1]  # Maintain position
            
            self.trading_signals[pair] = signals
            
        return self.trading_signals
    
    def backtest(self, price_data, minutes_per_day=391):
        """
        Run backtest for the trading period.
        
        Parameters:
        -----------
        price_data : dict
            Dictionary of price series for each stock
        minutes_per_day : int
            Number of minutes in a trading day
            
        Returns:
        --------
        pd.Series : Returns for the trading period
        """
        if not self.pairs or not self.trading_signals:
            print("No pairs or trading signals available. Run select_pairs and generate_trading_signals first.")
            return None
            
        formation_length = self.formation_days * minutes_per_day
        trading_length = self.trading_days * minutes_per_day
        
        # Initialize positions and portfolio value
        self.positions = {pair: np.zeros(trading_length) for pair in self.pairs}
        portfolio_value = np.ones(trading_length)  # Start with 1 unit of capital
        
        # Track returns for each pair
        pair_returns = {pair: np.zeros(trading_length) for pair in self.pairs}
        
        # Run backtest for each pair
        for pair in self.pairs:
            stock_a, stock_b = pair
            
            # Get price data for trading period
            price_a = price_data[stock_a][-(formation_length + trading_length):]
            price_b = price_data[stock_b][-(formation_length + trading_length):]
            
            # Extract formation and trading period data
            # Make sure we're accessing valid indices
            formation_price_a = price_a[:formation_length]
            formation_price_b = price_b[:formation_length]
            trading_price_a = price_a[formation_length:]
            trading_price_b = price_b[formation_length:]
            
            # Get trading signals
            signals = self.trading_signals[pair]
            
            # Initial position
            position = 0
            
            for t in range(1, trading_length):
                # Ensure we don't go out of bounds
                if t >= len(trading_price_a) or t-1 >= len(trading_price_a):
                    continue
                    
                # Calculate returns from t-1 to t
                ret_a = trading_price_a[t] / trading_price_a[t-1] - 1
                ret_b = trading_price_b[t] / trading_price_b[t-1] - 1
                
                # If we have a position, calculate returns
                if position != 0:
                    # If position is 1, we're long A and short B
                    if position == 1:
                        pair_return = ret_a - ret_b
                    # If position is -1, we're short A and long B
                    else:
                        pair_return = ret_b - ret_a
                        
                    # Add position returns (before transaction costs)
                    pair_returns[pair][t] = pair_return
                else:
                    pair_returns[pair][t] = 0
                
                # Check if signal changes
                new_signal = signals[t]
                
                # If signal changes, apply transaction costs
                if new_signal != position:
                    # Close existing position
                    if position != 0:
                        pair_returns[pair][t] -= 2 * self.transaction_cost  # Cost to close both legs
                    
                    # Open new position
                    if new_signal != 0:
                        pair_returns[pair][t] -= 2 * self.transaction_cost  # Cost to open both legs
                    
                    position = new_signal
                
                # Update position history
                self.positions[pair][t] = position
        
        # Calculate portfolio returns (equal weight to all pairs)
        portfolio_returns = np.zeros(trading_length)
        
        for t in range(1, trading_length):
            # Average returns across all pairs
            active_pairs = sum(1 for pair in self.pairs if self.positions[pair][t] != 0)
            if active_pairs > 0:
                day_return = sum(pair_returns[pair][t] for pair in self.pairs) / len(self.pairs)
                portfolio_returns[t] = day_return
            
            # Update portfolio value
            portfolio_value[t] = portfolio_value[t-1] * (1 + portfolio_returns[t])
        
        # Convert to daily returns
        daily_returns = []
        for day in range(self.trading_days):
            day_start = day * minutes_per_day
            day_end = min((day + 1) * minutes_per_day, trading_length)
            
            if day_start < len(portfolio_value) - 1 and day_end <= len(portfolio_value):
                daily_return = portfolio_value[day_end - 1] / portfolio_value[day_start] - 1
                daily_returns.append(daily_return)
        
        # Convert to pandas Series
        self.returns = pd.Series(daily_returns)
        
        return self.returns
    
    def plot_pair_trading(self, pair, price_data, minutes_per_day=391):
        """
        Plot spread and trading positions for a specific pair.
        
        Parameters:
        -----------
        pair : tuple
            Stock pair (stock_a, stock_b)
        price_data : dict
            Dictionary of price series for each stock
        minutes_per_day : int
            Number of minutes in a trading day
        """
        if pair not in self.pairs or pair not in self.positions:
            print(f"Pair {pair} not found in trading history.")
            return
            
        formation_length = self.formation_days * minutes_per_day
        trading_length = self.trading_days * minutes_per_day
        
        stock_a, stock_b = pair
        
        # Get price data
        price_a = price_data[stock_a][-(formation_length + trading_length):]
        price_b = price_data[stock_b][-(formation_length + trading_length):]
        
        # Calculate spread
        spread = self.calculate_spread(price_a, price_b)
        
        # Split into formation and trading periods
        formation_spread = spread[:formation_length]
        trading_spread = spread[formation_length:]
        
        # Get positions
        positions = self.positions[pair]
        
        # Create time indices
        formation_time = np.arange(0, formation_length)
        trading_time = np.arange(formation_length, formation_length + trading_length)
        
        # Plot
        plt.figure(figsize=(14, 8))
        
        # Plot spread
        plt.subplot(2, 1, 1)
        plt.plot(formation_time, formation_spread, 'b-', label='Formation Period')
        plt.plot(trading_time, trading_spread, 'g-', label='Trading Period')
        
        # Highlight positions
        buy_indices = trading_time[positions == 1]
        sell_indices = trading_time[positions == -1]
        
        if len(buy_indices) > 0:
            buy_values = trading_spread[positions == 1]
            plt.scatter(buy_indices, buy_values, color='g', marker='^', s=100, label='Long A, Short B')
            
        if len(sell_indices) > 0:
            sell_values = trading_spread[positions == -1]
            plt.scatter(sell_indices, sell_values, color='r', marker='v', s=100, label='Short A, Long B')
            
        plt.axvline(x=formation_length, color='k', linestyle='--', label='End of Formation Period')
        plt.title(f'Spread and Positions for Pair {stock_a}-{stock_b}')
        plt.legend()
        plt.ylabel('Spread')
        
        # Plot individual stock prices
        plt.subplot(2, 1, 2)
        plt.plot(np.arange(len(price_a)), price_a / price_a[0], 'b-', label=stock_a)
        plt.plot(np.arange(len(price_b)), price_b / price_b[0], 'r-', label=stock_b)
        plt.axvline(x=formation_length, color='k', linestyle='--', label='End of Formation Period')
        plt.title(f'Normalized Prices for {stock_a} and {stock_b}')
        plt.legend()
        plt.ylabel('Normalized Price')
        plt.xlabel('Time (minutes)')
        
        plt.tight_layout()
        plt.show()


class PairsBacktest:
    """
    Class to run a full backtest of the pairs trading strategy over a rolling window.
    """
    def __init__(self, start_date, end_date, formation_days=30, trading_days=5,
                 top_pairs=10, bollinger_k=0.5, transaction_cost=0.0020):
        """
        Initialize backtest parameters.
        
        Parameters:
        -----------
        start_date : str or datetime
            Start date for backtest
        end_date : str or datetime
            End date for backtest
        formation_days : int
            Number of days for formation period
        trading_days : int
            Number of days for trading period
        top_pairs : int
            Number of top pairs to select for trading
        bollinger_k : float
            Multiplier for Bollinger bands
        transaction_cost : float
            Transaction cost per half-turn
        """
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.formation_days = formation_days
        self.trading_days = trading_days
        self.top_pairs = top_pairs
        self.bollinger_k = bollinger_k
        self.transaction_cost = transaction_cost
        
        self.returns = []
        self.cumulative_returns = []
        self.trading_dates = []
        self.pairs_history = []
        
    def get_bloomberg_data(self, tickers, fields=['PX_LAST'], start_date=None, end_date=None, minutes_per_day=391):
        """
        Fetch data from Bloomberg for the specified tickers and date range.
        
        Parameters:
        -----------
        tickers : list
            List of Bloomberg tickers
        fields : list
            List of Bloomberg fields to retrieve
        start_date : str or datetime
            Start date for data
        end_date : str or datetime
            End date for data
        minutes_per_day : int
            Number of minutes in a trading day
            
        Returns:
        --------
        dict : Dictionary of price series for each ticker
        """
        if start_date is None:
            start_date = self.start_date
        if end_date is None:
            end_date = self.end_date
            
        # Try to connect to Bloomberg
        try:
            con = pdblp.BCon(debug=False, port=8194)
            con.start()
            
            # Get minute-by-minute data
            data = {}
            
            for ticker in tickers:
                print(f"Fetching data for {ticker}...")
                # Use intraday_bar to get minute-by-minute data
                df = con.bdib(
                    ticker, 
                    event_type='TRADE',
                    interval=1,  # 1 minute intervals
                    start_datetime=start_date,
                    end_datetime=end_date
                )
                
                # Store in dictionary
                if 'value' in df.columns:
                    data[ticker] = df['value'].values
                    
            con.stop()
            return data
            
        except Exception as e:
            print(f"Error fetching Bloomberg data: {e}")
            print("Using simulated data instead.")
            return self.generate_simulated_data(tickers, start_date, end_date, minutes_per_day)
    
    def generate_simulated_data(self, tickers, start_date=None, end_date=None, minutes_per_day=391):
        """
        Generate simulated price data for backtesting.
        
        Parameters:
        -----------
        tickers : list
            List of ticker symbols
        start_date : str or datetime
            Start date for data
        end_date : str or datetime
            End date for data
        minutes_per_day : int
            Number of minutes in a trading day
            
        Returns:
        --------
        dict : Dictionary of price series for each ticker
        """
        if start_date is None:
            start_date = self.start_date
        if end_date is None:
            end_date = self.end_date
            
        # Calculate number of trading days
        trading_days = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days
        trading_days = max(int(trading_days * 0.7), 1)  # Adjust for weekends and holidays
        
        # Calculate total number of minutes
        total_minutes = trading_days * minutes_per_day
        
        # Generate stock data with sector correlations
        n_stocks = len(tickers)
        n_sectors = min(5, n_stocks)  # Number of sectors
        
        # Assign stocks to sectors
        sectors = np.random.randint(0, n_sectors, n_stocks)
        
        # Create sector mapping for industry constraints
        industry_map = {tickers[i]: f"Industry_{sectors[i]}" for i in range(n_stocks)}
        
        # Generate market factor
        mu_market = 0.0001  # Small positive drift
        sigma_market = 0.001  # Market volatility
        market = np.zeros(total_minutes)
        market[0] = 100
        
        for i in range(1, total_minutes):
            market[i] = market[i-1] * (1 + np.random.normal(mu_market, sigma_market))
        
        # Generate sector factors
        sector_data = {}
        for sector in range(n_sectors):
            mu_sector = 0.0001
            sigma_sector = 0.0015
            sector_data[sector] = np.zeros(total_minutes)
            sector_data[sector][0] = 100
            
            for i in range(1, total_minutes):
                # Sector follows market with own random component
                market_return = market[i] / market[i-1] - 1
                sector_data[sector][i] = sector_data[sector][i-1] * (
                    1 + 0.7 * market_return + np.random.normal(mu_sector, sigma_sector)
                )
        
        # Generate stock prices
        price_data = {}
        
        for i, ticker in enumerate(tickers):
            sector = sectors[i]
            
            # Initial price
            price = np.random.uniform(50, 200)
            
            # Add time series properties
            prices = np.zeros(total_minutes)
            prices[0] = price
            
            # Parameters
            mu_stock = 0.0001
            sigma_stock = 0.002
            jump_prob = 0.004  # Probability of a jump
            jump_size_std = 0.02  # Standard deviation of jump size
            
            for j in range(1, total_minutes):
                # Get sector and market returns
                sector_return = sector_data[sector][j] / sector_data[sector][j-1] - 1
                market_return = market[j] / market[j-1] - 1
                
                # Stock return is a combination of sector, market, and idiosyncratic components
                stock_return = (
                    0.5 * sector_return +  # Sector component
                    0.2 * market_return +  # Market component
                    np.random.normal(mu_stock, sigma_stock)  # Idiosyncratic component
                )
                
                # Add jumps
                if np.random.random() < jump_prob:
                    jump = np.random.normal(0, jump_size_std)
                    stock_return += jump
                
                # Update price
                prices[j] = prices[j-1] * (1 + stock_return)
            
            price_data[ticker] = prices
        
        return price_data, industry_map
    
    def run_backtest(self, tickers, minutes_per_day=391, data_source='simulated'):
        """
        Run full backtest over the specified date range.
        
        Parameters:
        -----------
        tickers : list
            List of ticker symbols
        minutes_per_day : int
            Number of minutes in a trading day
        data_source : str
            Source of data ('bloomberg' or 'simulated')
            
        Returns:
        --------
        pd.Series : Daily returns from the backtest
        """
        # Get price data
        if data_source.lower() == 'bloomberg':
            price_data = self.get_bloomberg_data(tickers, minutes_per_day=minutes_per_day)
            industry_map = None  # In real implementation, this would be obtained from Bloomberg
        else:
            price_data, industry_map = self.generate_simulated_data(tickers, minutes_per_day=minutes_per_day)
        
        # Calculate number of trading days
        total_minutes = min([len(prices) for prices in price_data.values()])
        total_days = total_minutes // minutes_per_day
        
        # Ensure we have enough data for at least one formation + trading period
        min_days_needed = self.formation_days + self.trading_days
        if total_days < min_days_needed:
            raise ValueError(f"Not enough data. Need at least {min_days_needed} days, but got {total_days} days.")
        
        # Calculate number of backtest periods
        backtest_days = total_days - self.formation_days
        
        # Create trading dates
        start_date = self.start_date
        trading_dates = [start_date + pd.Timedelta(days=i) for i in range(backtest_days)]
        
        all_returns = []
        cumulative_return = 1.0
        pairs_history = []
        
        # Run backtest for each trading day
        for day in range(0, backtest_days, self.trading_days):
            print(f"Backtest day {day}/{backtest_days}")
            
            # Calculate data ranges
            day_start_idx = day * minutes_per_day
            formation_end_idx = (day + self.formation_days) * minutes_per_day
            trading_end_idx = min((day + self.formation_days + self.trading_days) * minutes_per_day, total_minutes)
            
            # Ensure we have enough data for trading period
            if trading_end_idx - formation_end_idx < minutes_per_day:
                print(f"Not enough data for trading period at day {day}. Skipping.")
                continue
            
            # Extract data for this period
            period_data = {}
            for ticker in price_data:
                if day_start_idx < len(price_data[ticker]) and trading_end_idx <= len(price_data[ticker]):
                    period_data[ticker] = price_data[ticker][day_start_idx:trading_end_idx]
                else:
                    print(f"Not enough data for ticker {ticker} at day {day}. Skipping.")
                    continue
            
            # Skip if we don't have data for all tickers
            if len(period_data) < len(tickers):
                continue
            
            # Initialize strategy
            strategy = FlexibleRegimeSwitchingPairsTrading(
                formation_days=self.formation_days,
                trading_days=self.trading_days,
                top_pairs=self.top_pairs,
                bollinger_k=self.bollinger_k,
                transaction_cost=self.transaction_cost
            )
            
            # Select pairs
            selected_pairs = strategy.select_pairs(period_data, industries=industry_map, minutes_per_day=minutes_per_day)
            
            # Generate trading signals
            if selected_pairs:
                strategy.generate_trading_signals(period_data, minutes_per_day=minutes_per_day)
                
                # Run backtest
                period_returns = strategy.backtest(period_data, minutes_per_day=minutes_per_day)
                
                if period_returns is not None and len(period_returns) > 0:
                    # Store returns
                    all_returns.extend(period_returns)
                    
                    # Update cumulative return
                    for ret in period_returns:
                        cumulative_return *= (1 + ret)
                    
                    # Store pairs
                    pairs_history.append(selected_pairs)
                    
                    # Plot first pair trading for visualization
                    if day == 0 and len(selected_pairs) > 0:
                        first_pair = selected_pairs[0]
                        strategy.plot_pair_trading(first_pair, period_data, minutes_per_day=minutes_per_day)
        
        # Store results
        self.returns = pd.Series(all_returns)
        self.cumulative_returns = pd.Series([1.0] + [np.prod(1 + np.array(all_returns[:i+1])) for i in range(len(all_returns))])
        self.trading_dates = trading_dates[:len(all_returns)]
        self.pairs_history = pairs_history
        
        return self.returns
    
    def calculate_performance_metrics(self):
        """
        Calculate performance metrics for the backtest.
        
        Returns:
        --------
        dict : Performance metrics
        """
        if len(self.returns) == 0:
            print("No returns data available. Run backtest first.")
            return None
        
        # Annualize assuming 252 trading days per year
        annual_factor = 252
        
        # Calculate metrics
        total_return = self.cumulative_returns.iloc[-1] - 1
        annual_return = (1 + total_return) ** (annual_factor / len(self.returns)) - 1
        
        daily_std = np.std(self.returns)
        annual_std = daily_std * np.sqrt(annual_factor)
        
        sharpe_ratio = annual_return / annual_std if annual_std > 0 else 0
        
        # Calculate drawdown
        drawdown = 1 - self.cumulative_returns / np.maximum.accumulate(self.cumulative_returns)
        max_drawdown = np.max(drawdown)
        
        # Calculate win ratio
        win_ratio = np.mean(self.returns > 0)
        
        metrics = {
            'total_return': total_return,
            'annual_return': annual_return,
            'annual_std': annual_std,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_ratio': win_ratio
        }
        
        return metrics
    
    def plot_performance(self, benchmark=None):
        """
        Plot backtest performance.
        
        Parameters:
        -----------
        benchmark : pd.Series or None
            Benchmark returns for comparison
        """
        if len(self.returns) == 0:
            print("No returns data available. Run backtest first.")
            return
        
        # Calculate metrics
        metrics = self.calculate_performance_metrics()
        
        # Plot cumulative returns
        plt.figure(figsize=(14, 10))
        
        # Cumulative returns plot
        plt.subplot(2, 1, 1)
        plt.plot(self.cumulative_returns, label='Strategy')
        
        if benchmark is not None:
            # Align benchmark with strategy returns
            aligned_benchmark = benchmark.reindex(self.trading_dates, method='ffill')
            plt.plot(aligned_benchmark, label='Benchmark')
            
        plt.title('Cumulative Returns')
        plt.xlabel('Time')
        plt.ylabel('Cumulative Return')
        plt.legend()
        plt.grid(True)
        
        # Add performance metrics as text
        plt.figtext(0.15, 0.45, 
                   f"Total Return: {metrics['total_return']:.2%}\n"
                   f"Annual Return: {metrics['annual_return']:.2%}\n"
                   f"Annual Std Dev: {metrics['annual_std']:.2%}\n"
                   f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}\n"
                   f"Max Drawdown: {metrics['max_drawdown']:.2%}\n"
                   f"Win Ratio: {metrics['win_ratio']:.2%}",
                   bbox=dict(facecolor='white', alpha=0.8), fontsize=12)
        
        # Daily returns plot
        plt.subplot(2, 1, 2)
        plt.bar(range(len(self.returns)), self.returns)
        plt.title('Daily Returns')
        plt.xlabel('Day')
        plt.ylabel('Return')
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()


# Main execution
if __name__ == "__main__":
    print("Flexible Regime Switching Model for Pairs Trading")
    print("Based on the paper by Endres and Stübinger (2019)")
    
    # Parameters from the paper
    formation_days = 30
    trading_days = 5
    top_pairs = 10
    bollinger_k = 0.5
    transaction_cost = 0.0020  # 20 bps per round trip
    
    # Create backtest
    backtest = PairsBacktest(
        start_date='2020-01-01',
        end_date='2020-03-31',
        formation_days=formation_days,
        trading_days=trading_days,
        top_pairs=top_pairs,
        bollinger_k=bollinger_k,
        transaction_cost=transaction_cost
    )
    
    # Define tickers (for SP500, you'd have 500 tickers)
    # Using a smaller set for demonstration
    tickers = ['STOCK_{}'.format(i) for i in range(1, 31)]
    
    # Run backtest with simulated data
    returns = backtest.run_backtest(tickers, data_source='simulated')
    
    # Calculate performance metrics
    metrics = backtest.calculate_performance_metrics()
    
    # Print results
    print("\nBacktest Results:")
    print(f"Total Return: {metrics['total_return']:.2%}")
    print(f"Annualized Return: {metrics['annual_return']:.2%}")
    print(f"Annualized Volatility: {metrics['annual_std']:.2%}")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"Maximum Drawdown: {metrics['max_drawdown']:.2%}")
    print(f"Win Ratio: {metrics['win_ratio']:.2%}")
    
    # Create benchmark (S&P 500 equivalent)
    benchmark_returns = pd.Series(np.random.normal(0.0002, 0.01, len(returns)))
    benchmark_cumulative = pd.Series([1.0] + [np.prod(1 + benchmark_returns[:i+1]) for i in range(len(benchmark_returns))])
    
    # Plot performance
    backtest.plot_performance(benchmark=benchmark_cumulative)