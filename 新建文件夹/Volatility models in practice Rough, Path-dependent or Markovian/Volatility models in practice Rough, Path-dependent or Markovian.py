import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import norm
from scipy.optimize import minimize
from scipy.integrate import quad
from scipy.special import gamma
import time
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

class VolterraProcess:
    """
    Base class for simulating Volterra processes.
    """
    def __init__(self, T=1.0, n_steps=252, n_paths=10000):
        """
        Initialize the Volterra process.
        
        Parameters:
        -----------
        T : float
            Time horizon
        n_steps : int
            Number of time steps
        n_paths : int
            Number of simulation paths
        """
        self.T = T
        self.n_steps = n_steps
        self.n_paths = n_paths
        self.dt = T / n_steps
        self.time_grid = np.linspace(0, T, n_steps + 1)
        
    def kernel(self, t):
        """
        Kernel function K(t). To be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement this method")
        
    def simulate(self):
        """
        Simulate paths of the Volterra process.
        """
        raise NotImplementedError("Subclasses must implement this method")


class RoughBergomiModel:
    """
    Implementation of the rough Bergomi model.
    """
    def __init__(self, xi0=0.04, eta=1.0, rho=-0.7, H=0.1, T=1.0, n_steps=252, n_paths=10000):
        """
        Initialize the rough Bergomi model.
        
        Parameters:
        -----------
        xi0 : float or function
            Initial forward variance curve
        eta : float
            Volatility of volatility
        rho : float
            Correlation between spot and volatility
        H : float
            Hurst parameter (0 < H <= 0.5)
        T : float
            Time horizon
        n_steps : int
            Number of time steps
        n_paths : int
            Number of simulation paths
        """
        self.xi0 = xi0 if callable(xi0) else lambda t: xi0
        self.eta = eta
        self.rho = rho
        self.H = H
        self.T = T
        self.n_steps = n_steps
        self.n_paths = n_paths
        self.dt = T / n_steps
        self.time_grid = np.linspace(0, T, n_steps + 1)
        
    def kernel(self, t):
        """
        Fractional kernel K(t) = t^(H-1/2).
        """
        return t**(self.H - 0.5) * (t > 0)
    
    def simulate_volatility(self):
        """
        Simulate the volatility process.
        
        Returns:
        --------
        ndarray : Volatility paths of shape (n_paths, n_steps+1)
        """
        # Create Brownian motion for volatility
        dW2 = np.random.normal(0, np.sqrt(self.dt), (self.n_paths, self.n_steps))
        
        # Initialize process X
        X = np.zeros((self.n_paths, self.n_steps + 1))
        
        # For rough Bergomi, we need to compute the stochastic integral with fractional kernel
        for i in range(1, self.n_steps + 1):
            t = self.time_grid[i]
            
            # Compute the discretized stochastic integral
            for j in range(i):
                s = self.time_grid[j]
                if i > j:  # Ensure t > s
                    X[:, i] += self.eta * self.kernel(t - s) * dW2[:, j]
        
        # Compute the variance process
        xi = np.zeros_like(X)
        for i in range(self.n_steps + 1):
            t = self.time_grid[i]
            xi[:, i] = self.xi0(t) * np.exp(X[:, i] - 0.5 * self.eta**2 * 
                                          quad(lambda s: self.kernel(s)**2, 0, t)[0])
        
        return np.sqrt(xi)
    
    def simulate(self):
        """
        Simulate stock price paths.
        
        Returns:
        --------
        tuple : (S, vol) where S is the stock price paths and vol is the volatility paths
        """
        # Simulate volatility
        vol = self.simulate_volatility()
        
        # Create correlated Brownian motion for stock price
        dW1 = np.random.normal(0, np.sqrt(self.dt), (self.n_paths, self.n_steps))
        
        # Adjust for correlation
        dW1 = self.rho * dW1 + np.sqrt(1 - self.rho**2) * np.random.normal(0, np.sqrt(self.dt), (self.n_paths, self.n_steps))
        
        # Initialize stock price
        S = np.zeros((self.n_paths, self.n_steps + 1))
        S[:, 0] = 1.0
        
        # Simulate stock price paths
        for i in range(self.n_steps):
            S[:, i+1] = S[:, i] * np.exp(-0.5 * vol[:, i]**2 * self.dt + vol[:, i] * dW1[:, i])
        
        return S, vol


class PathDependentBergomiModel:
    """
    Implementation of the path-dependent Bergomi model.
    """
    def __init__(self, xi0=0.04, eta=1.0, rho=-0.7, H=0.1, epsilon=1/52, T=1.0, n_steps=252, n_paths=10000):
        """
        Initialize the path-dependent Bergomi model.
        
        Parameters:
        -----------
        xi0 : float or function
            Initial forward variance curve
        eta : float
            Volatility of volatility
        rho : float
            Correlation between spot and volatility
        H : float
            Hurst parameter (can be negative)
        epsilon : float
            Time shift parameter (typically 1/52 for weekly scale)
        T : float
            Time horizon
        n_steps : int
            Number of time steps
        n_paths : int
            Number of simulation paths
        """
        self.xi0 = xi0 if callable(xi0) else lambda t: xi0
        self.eta = eta
        self.rho = rho
        self.H = H
        self.epsilon = epsilon
        self.T = T
        self.n_steps = n_steps
        self.n_paths = n_paths
        self.dt = T / n_steps
        self.time_grid = np.linspace(0, T, n_steps + 1)
        
    def kernel(self, t):
        """
        Shifted fractional kernel K(t) = (t+ε)^(H-1/2).
        """
        return (t + self.epsilon)**(self.H - 0.5) * (t >= 0)
    
    def simulate_volatility(self):
        """
        Simulate the volatility process.
        
        Returns:
        --------
        ndarray : Volatility paths of shape (n_paths, n_steps+1)
        """
        # Create Brownian motion for volatility
        dW2 = np.random.normal(0, np.sqrt(self.dt), (self.n_paths, self.n_steps))
        
        # Initialize process X
        X = np.zeros((self.n_paths, self.n_steps + 1))
        
        # For path-dependent Bergomi, we use the shifted kernel
        for i in range(1, self.n_steps + 1):
            t = self.time_grid[i]
            
            # Compute the discretized stochastic integral
            for j in range(i):
                s = self.time_grid[j]
                X[:, i] += self.eta * self.kernel(t - s) * dW2[:, j]
        
        # Compute the variance process
        xi = np.zeros_like(X)
        for i in range(self.n_steps + 1):
            t = self.time_grid[i]
            xi[:, i] = self.xi0(t) * np.exp(X[:, i] - 0.5 * self.eta**2 * 
                                          quad(lambda s: self.kernel(s)**2, 0, t)[0])
        
        return np.sqrt(xi)
    
    def simulate(self):
        """
        Simulate stock price paths.
        
        Returns:
        --------
        tuple : (S, vol) where S is the stock price paths and vol is the volatility paths
        """
        # Simulate volatility
        vol = self.simulate_volatility()
        
        # Create correlated Brownian motion for stock price
        dW1 = np.random.normal(0, np.sqrt(self.dt), (self.n_paths, self.n_steps))
        
        # Adjust for correlation
        dW1 = self.rho * dW1 + np.sqrt(1 - self.rho**2) * np.random.normal(0, np.sqrt(self.dt), (self.n_paths, self.n_steps))
        
        # Initialize stock price
        S = np.zeros((self.n_paths, self.n_steps + 1))
        S[:, 0] = 1.0
        
        # Simulate stock price paths
        for i in range(self.n_steps):
            S[:, i+1] = S[:, i] * np.exp(-0.5 * vol[:, i]**2 * self.dt + vol[:, i] * dW1[:, i])
        
        return S, vol


class OneFactorBergomiModel:
    """
    Implementation of the one-factor Bergomi model.
    """
    def __init__(self, xi0=0.04, eta=1.0, rho=-0.7, H=0.1, epsilon=1/52, T=1.0, n_steps=252, n_paths=10000):
        """
        Initialize the one-factor Bergomi model.
        
        Parameters:
        -----------
        xi0 : float or function
            Initial forward variance curve
        eta : float
            Volatility of volatility
        rho : float
            Correlation between spot and volatility
        H : float
            Hurst parameter (can be negative)
        epsilon : float
            Time scale parameter (typically 1/52 for weekly scale)
        T : float
            Time horizon
        n_steps : int
            Number of time steps
        n_paths : int
            Number of simulation paths
        """
        self.xi0 = xi0 if callable(xi0) else lambda t: xi0
        self.eta = eta
        self.rho = rho
        self.H = H
        self.epsilon = epsilon
        self.T = T
        self.n_steps = n_steps
        self.n_paths = n_paths
        self.dt = T / n_steps
        self.time_grid = np.linspace(0, T, n_steps + 1)
        self.mean_reversion = (0.5 - H) / epsilon
        
    def kernel(self, t):
        """
        Exponential kernel K(t) = η·ε^(H-1/2)·exp(-(1/2-H)·ε^(-1)·t).
        """
        return self.eta * self.epsilon**(self.H - 0.5) * np.exp(-self.mean_reversion * t) * (t >= 0)
    
    def simulate(self):
        """
        Simulate stock price paths using the Ornstein-Uhlenbeck process representation.
        
        Returns:
        --------
        tuple : (S, vol) where S is the stock price paths and vol is the volatility paths
        """
        # Create Brownian motions
        dW1 = np.random.normal(0, np.sqrt(self.dt), (self.n_paths, self.n_steps))
        dW2 = np.random.normal(0, np.sqrt(self.dt), (self.n_paths, self.n_steps))
        
        # Adjust for correlation
        dW1 = self.rho * dW2 + np.sqrt(1 - self.rho**2) * dW1
        
        # Initialize OU process and volatility
        X = np.zeros((self.n_paths, self.n_steps + 1))
        vol = np.zeros((self.n_paths, self.n_steps + 1))
        
        # Initial volatility
        vol[:, 0] = np.sqrt(self.xi0(0))
        
        # OU process volatility parameter
        vol_param = self.eta * self.epsilon**(self.H - 0.5)
        
        # Simulate OU process
        for i in range(self.n_steps):
            X[:, i+1] = X[:, i] * np.exp(-self.mean_reversion * self.dt) + vol_param * np.sqrt(1 - np.exp(-2 * self.mean_reversion * self.dt)) / np.sqrt(2 * self.mean_reversion) * dW2[:, i]
            vol[:, i+1] = np.sqrt(self.xi0(self.time_grid[i+1]) * np.exp(X[:, i+1]))
        
        # Initialize stock price
        S = np.zeros((self.n_paths, self.n_steps + 1))
        S[:, 0] = 1.0
        
        # Simulate stock price paths
        for i in range(self.n_steps):
            S[:, i+1] = S[:, i] * np.exp(-0.5 * vol[:, i]**2 * self.dt + vol[:, i] * dW1[:, i])
        
        return S, vol


class TwoFactorBergomiModel:
    """
    Implementation of the two-factor Bergomi model.
    """
    def __init__(self, xi0=0.04, eta1=1.0, eta2=0.5, rho=-0.7, H1=0.1, H2=0.45, 
                 epsilon=1/52, T=1.0, n_steps=252, n_paths=10000):
        """
        Initialize the two-factor Bergomi model.
        
        Parameters:
        -----------
        xi0 : float or function
            Initial forward variance curve
        eta1 : float
            Volatility of volatility for the first factor
        eta2 : float
            Volatility of volatility for the second factor
        rho : float
            Correlation between spot and volatility
        H1 : float
            Hurst parameter for the first factor (can be negative)
        H2 : float
            Hurst parameter for the second factor (typically 0.45)
        epsilon : float
            Time scale parameter (typically 1/52 for weekly scale)
        T : float
            Time horizon
        n_steps : int
            Number of time steps
        n_paths : int
            Number of simulation paths
        """
        self.xi0 = xi0 if callable(xi0) else lambda t: xi0
        self.eta1 = eta1
        self.eta2 = eta2
        self.rho = rho
        self.H1 = H1
        self.H2 = H2
        self.epsilon = epsilon
        self.T = T
        self.n_steps = n_steps
        self.n_paths = n_paths
        self.dt = T / n_steps
        self.time_grid = np.linspace(0, T, n_steps + 1)
        self.mean_reversion1 = (0.5 - H1) / epsilon
        self.mean_reversion2 = (0.5 - H2) / epsilon
        
    def simulate(self):
        """
        Simulate stock price paths using the two-factor Ornstein-Uhlenbeck process representation.
        
        Returns:
        --------
        tuple : (S, vol) where S is the stock price paths and vol is the volatility paths
        """
        # Create Brownian motions
        dW1 = np.random.normal(0, np.sqrt(self.dt), (self.n_paths, self.n_steps))
        dW2 = np.random.normal(0, np.sqrt(self.dt), (self.n_paths, self.n_steps))
        
        # Adjust for correlation
        dW1 = self.rho * dW2 + np.sqrt(1 - self.rho**2) * dW1
        
        # Initialize OU processes and volatility
        X1 = np.zeros((self.n_paths, self.n_steps + 1))
        X2 = np.zeros((self.n_paths, self.n_steps + 1))
        vol = np.zeros((self.n_paths, self.n_steps + 1))
        
        # Initial volatility
        vol[:, 0] = np.sqrt(self.xi0(0))
        
        # OU process volatility parameters
        vol_param1 = self.eta1 * self.epsilon**(self.H1 - 0.5)
        vol_param2 = self.eta2 * self.epsilon**(self.H2 - 0.5)
        
        # Simulate OU processes (for simplicity, using the same Brownian motion)
        for i in range(self.n_steps):
            X1[:, i+1] = X1[:, i] * np.exp(-self.mean_reversion1 * self.dt) + vol_param1 * np.sqrt(1 - np.exp(-2 * self.mean_reversion1 * self.dt)) / np.sqrt(2 * self.mean_reversion1) * dW2[:, i]
            X2[:, i+1] = X2[:, i] * np.exp(-self.mean_reversion2 * self.dt) + vol_param2 * np.sqrt(1 - np.exp(-2 * self.mean_reversion2 * self.dt)) / np.sqrt(2 * self.mean_reversion2) * dW2[:, i]
            vol[:, i+1] = np.sqrt(self.xi0(self.time_grid[i+1]) * np.exp(X1[:, i+1] + X2[:, i+1]))
        
        # Initialize stock price
        S = np.zeros((self.n_paths, self.n_steps + 1))
        S[:, 0] = 1.0
        
        # Simulate stock price paths
        for i in range(self.n_steps):
            S[:, i+1] = S[:, i] * np.exp(-0.5 * vol[:, i]**2 * self.dt + vol[:, i] * dW1[:, i])
        
        return S, vol


def black_scholes_price(S0, K, T, r, sigma, option_type='call'):
    """
    Compute the Black-Scholes option price.
    
    Parameters:
    -----------
    S0 : float
        Spot price
    K : float
        Strike price
    T : float
        Time to maturity
    r : float
        Risk-free rate
    sigma : float
        Volatility
    option_type : str
        'call' or 'put'
    
    Returns:
    --------
    float : Option price
    """
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == 'call':
        return S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S0 * norm.cdf(-d1)


def implied_volatility(price, S0, K, T, r, option_type='call', initial_vol=0.2, max_iterations=100, precision=1e-8):
    """
    Compute the implied volatility using the Newton-Raphson method.
    
    Parameters:
    -----------
    price : float
        Market price of the option
    S0 : float
        Spot price
    K : float
        Strike price
    T : float
        Time to maturity
    r : float
        Risk-free rate
    option_type : str
        'call' or 'put'
    initial_vol : float
        Initial guess for volatility
    max_iterations : int
        Maximum number of iterations
    precision : float
        Desired precision
    
    Returns:
    --------
    float : Implied volatility
    """
    vol = initial_vol
    
    for i in range(max_iterations):
        # Compute price and vega
        d1 = (np.log(S0 / K) + (r + 0.5 * vol**2) * T) / (vol * np.sqrt(T))
        model_price = black_scholes_price(S0, K, T, r, vol, option_type)
        vega = S0 * np.sqrt(T) * norm.pdf(d1)
        
        # Break if vega is too small or we've reached precision
        if abs(vega) < 1e-10 or abs(model_price - price) < precision:
            break
        
        # Update volatility estimate
        vol = vol - (model_price - price) / vega
        
        # Ensure volatility stays positive
        vol = max(0.001, vol)
    
    return vol


def compute_option_prices(model, strikes, maturities, r=0.0, option_type='call'):
    """
    Compute option prices for a given model, strikes, and maturities.
    
    Parameters:
    -----------
    model : object
        Model instance with simulate method
    strikes : array
        Array of strike prices
    maturities : array
        Array of maturities
    r : float
        Risk-free rate
    option_type : str
        'call' or 'put'
    
    Returns:
    --------
    ndarray : Option prices of shape (len(maturities), len(strikes))
    """
    # Simulate paths
    S, vol = model.simulate()
    
    # Initialize option prices array
    option_prices = np.zeros((len(maturities), len(strikes)))
    
    # Compute option prices for each maturity and strike
    for i, T in enumerate(maturities):
        idx = int(T / model.dt)
        
        # Get terminal stock prices at maturity T
        S_T = S[:, idx]
        
        for j, K in enumerate(strikes):
            # Compute payoff
            if option_type == 'call':
                payoff = np.maximum(S_T - K, 0)
            else:
                payoff = np.maximum(K - S_T, 0)
            
            # Discount payoff
            option_prices[i, j] = np.exp(-r * T) * np.mean(payoff)
    
    return option_prices


def compute_implied_volatility_surface(option_prices, S0, strikes, maturities, r=0.0, option_type='call'):
    """
    Compute implied volatility surface from option prices.
    
    Parameters:
    -----------
    option_prices : ndarray
        Option prices of shape (len(maturities), len(strikes))
    S0 : float
        Spot price
    strikes : array
        Array of strike prices
    maturities : array
        Array of maturities
    r : float
        Risk-free rate
    option_type : str
        'call' or 'put'
    
    Returns:
    --------
    ndarray : Implied volatility surface of shape (len(maturities), len(strikes))
    """
    # Initialize implied volatility surface
    iv_surface = np.zeros_like(option_prices)
    
    # Compute implied volatility for each maturity and strike
    for i, T in enumerate(maturities):
        for j, K in enumerate(strikes):
            iv_surface[i, j] = implied_volatility(option_prices[i, j], S0, K, T, r, option_type)
    
    return iv_surface


def compute_atm_skew(iv_surface, strikes, maturities, S0):
    """
    Compute the at-the-money skew for each maturity.
    
    Parameters:
    -----------
    iv_surface : ndarray
        Implied volatility surface of shape (len(maturities), len(strikes))
    strikes : array
        Array of strike prices
    maturities : array
        Array of maturities
    S0 : float
        Spot price
    
    Returns:
    --------
    ndarray : ATM skews for each maturity
    """
    # Initialize ATM skew array
    atm_skew = np.zeros(len(maturities))
    
    # Find the index of the strike closest to S0
    atm_idx = np.abs(strikes - S0).argmin()
    
    # Compute the central difference approximation of the skew
    for i in range(len(maturities)):
        if atm_idx > 0 and atm_idx < len(strikes) - 1:
            atm_skew[i] = (iv_surface[i, atm_idx + 1] - iv_surface[i, atm_idx - 1]) / (strikes[atm_idx + 1] - strikes[atm_idx - 1])
        elif atm_idx == 0:
            atm_skew[i] = (iv_surface[i, atm_idx + 1] - iv_surface[i, atm_idx]) / (strikes[atm_idx + 1] - strikes[atm_idx])
        else:
            atm_skew[i] = (iv_surface[i, atm_idx] - iv_surface[i, atm_idx - 1]) / (strikes[atm_idx] - strikes[atm_idx - 1])
    
    return atm_skew


def calibrate_model(model_class, market_iv, strikes, maturities, S0, param_bounds=None, initial_params=None, fixed_params=None):
    """
    Calibrate a model to market implied volatilities.
    
    Parameters:
    -----------
    model_class : class
        Model class to calibrate
    market_iv : ndarray
        Market implied volatility surface of shape (len(maturities), len(strikes))
    strikes : array
        Array of strike prices
    maturities : array
        Array of maturities
    S0 : float
        Spot price
    param_bounds : dict, optional
        Dictionary with parameter bounds
    initial_params : dict, optional
        Dictionary with initial parameter values
    fixed_params : dict, optional
        Dictionary with fixed parameter values
    
    Returns:
    --------
    dict : Calibrated parameters
    float : RMSE
    """
    # Set default parameter bounds
    if param_bounds is None:
        param_bounds = {
            'eta': (0.01, 5.0),
            'rho': (-0.99, 0.0),
            'H': (0.01, 0.49) if model_class == RoughBergomiModel else (-2.0, 0.49)
        }
        if model_class == TwoFactorBergomiModel:
            param_bounds['eta2'] = (0.01, 5.0)
    
    # Set default initial parameters
    if initial_params is None:
        initial_params = {
            'eta': 1.0,
            'rho': -0.7,
            'H': 0.1
        }
        if model_class == TwoFactorBergomiModel:
            initial_params['eta2'] = 0.5
    
    # Set default fixed parameters
    if fixed_params is None:
        fixed_params = {
            'xi0': lambda t: market_iv[0, len(strikes)//2]**2,  # ATM volatility for shortest maturity
            'epsilon': 1/52,
            'T': max(maturities),
            'n_steps': 100,
            'n_paths': 5000
        }
        if model_class == TwoFactorBergomiModel:
            fixed_params['H2'] = 0.45
    
    # Extract parameters to calibrate
    params_to_calibrate = list(param_bounds.keys())
    initial_values = [initial_params[param] for param in params_to_calibrate]
    bounds = [param_bounds[param] for param in params_to_calibrate]
    
    # Define objective function for calibration
    def objective(params):
        # Create parameter dictionary
        param_dict = {param: params[i] for i, param in enumerate(params_to_calibrate)}
        param_dict.update(fixed_params)
        
        # Create model instance
        model = model_class(**param_dict)
        
        # Compute option prices
        option_prices = compute_option_prices(model, strikes, maturities)
        
        # Compute implied volatility surface
        iv_surface = compute_implied_volatility_surface(option_prices, S0, strikes, maturities)
        
        # Compute RMSE
        rmse = np.sqrt(np.mean((iv_surface - market_iv)**2))
        
        return rmse
    
    # Perform calibration using L-BFGS-B algorithm
    result = minimize(objective, initial_values, bounds=bounds, method='L-BFGS-B')
    
    # Extract calibrated parameters
    calibrated_params = {param: result.x[i] for i, param in enumerate(params_to_calibrate)}
    calibrated_params.update(fixed_params)
    
    return calibrated_params, result.fun


def estimate_hurst_index(time_series, q_values=None, max_lag=50):
    """
    Estimate the Hurst index of a time series using the q-variation method.
    
    Parameters:
    -----------
    time_series : array
        Time series data
    q_values : array, optional
        Values of q for which to compute the q-variation
    max_lag : int, optional
        Maximum lag for the q-variation
    
    Returns:
    --------
    float : Estimated Hurst index
    """
    if q_values is None:
        q_values = np.array([0.5, 1.0, 1.5, 2.0, 3.0])
    
    # Initialize q-variation array
    m_q_delta = np.zeros((len(q_values), max_lag))
    
    # Compute q-variation for different lags
    for i, q in enumerate(q_values):
        for delta in range(1, max_lag + 1):
            # Compute q-variation for lag delta
            increments = np.abs(time_series[delta:] - time_series[:-delta])
            m_q_delta[i, delta-1] = np.mean(increments**q)
    
    # Estimate slopes for each q
    slopes = np.zeros(len(q_values))
    for i, q in enumerate(q_values):
        # Perform linear regression on log-log scale
        log_delta = np.log(np.arange(1, max_lag + 1))
        log_m_q_delta = np.log(m_q_delta[i])
        slopes[i] = np.polyfit(log_delta, log_m_q_delta, 1)[0]
    
    # Estimate Hurst index from the relationship slopes = q*H
    H = np.polyfit(q_values, slopes, 1)[0]
    
    return H


def compute_realized_volatility(S, window=21):
    """
    Compute realized volatility time series from price paths.
    
    Parameters:
    -----------
    S : ndarray
        Price paths of shape (n_paths, n_steps+1)
    window : int, optional
        Window size for realized volatility (in days)
    
    Returns:
    --------
    ndarray : Realized volatility time series of shape (n_paths, n_steps+1-window)
    """
    # Compute log returns
    log_returns = np.diff(np.log(S), axis=1)
    
    # Initialize realized volatility array
    n_paths, n_steps = log_returns.shape
    rv = np.zeros((n_paths, n_steps - window + 1))
    
    # Compute realized volatility using rolling window
    for i in range(n_paths):
        for j in range(n_steps - window + 1):
            rv[i, j] = np.sqrt(np.sum(log_returns[i, j:j+window]**2))
    
    return rv


def run_empirical_study():
    """
    Run the empirical study comparing different volatility models.
    """
    # Set up parameters
    S0 = 100.0
    strikes = np.linspace(80, 120, 9)
    short_maturities = np.array([1/52, 1/12, 2/12, 3/12])  # 1 week to 3 months
    long_maturities = np.array([1/52, 1/12, 2/12, 3/12, 6/12, 1.0, 2.0, 3.0])  # 1 week to 3 years
    
    # Generate synthetic market data (using two-factor Bergomi model with realistic parameters)
    print("Generating synthetic market data...")
    market_model = TwoFactorBergomiModel(
        xi0=0.04,  # Initial variance
        eta1=1.5,  # Vol-of-vol for fast factor
        eta2=0.7,  # Vol-of-vol for slow factor
        rho=-0.7,  # Correlation
        H1=-0.3,   # Hurst for fast factor (negative)
        H2=0.45,   # Hurst for slow factor
        epsilon=1/52,  # Weekly scale
        T=3.0,     # 3 years
        n_steps=3*252,  # Daily steps
        n_paths=10000   # Number of paths
    )
    
    # Simulate market data
    S_market, vol_market = market_model.simulate()
    
    # Compute option prices for short maturities
    print("Computing market option prices for short maturities...")
    market_option_prices_short = compute_option_prices(market_model, strikes, short_maturities)
    market_iv_short = compute_implied_volatility_surface(market_option_prices_short, S0, strikes, short_maturities)
    
    # Compute option prices for all maturities
    print("Computing market option prices for all maturities...")
    market_option_prices_all = compute_option_prices(market_model, strikes, long_maturities)
    market_iv_all = compute_implied_volatility_surface(market_option_prices_all, S0, strikes, long_maturities)
    
    # Compute ATM skews
    market_atm_skew_short = compute_atm_skew(market_iv_short, strikes, short_maturities, S0)
    market_atm_skew_all = compute_atm_skew(market_iv_all, strikes, long_maturities, S0)
    
    # Models to evaluate
    models = {
        'Rough Bergomi': RoughBergomiModel,
        'Path-dependent Bergomi': PathDependentBergomiModel,
        'One-factor Bergomi': OneFactorBergomiModel,
        'Two-factor Bergomi': TwoFactorBergomiModel
    }
    
    # Results containers
    results_short = {}
    results_all = {}
    
    # Run calibration for short maturities
    print("\nCalibrating models to short maturities...")
    for model_name, model_class in tqdm(models.items()):
        print(f"\nCalibrating {model_name} to short maturities...")
        
        # Initial parameters and bounds
        initial_params = {
            'eta': 1.0,
            'rho': -0.7,
            'H': 0.1
        }
        param_bounds = {
            'eta': (0.01, 5.0),
            'rho': (-0.99, 0.0),
            'H': (0.01, 0.49) if model_class == RoughBergomiModel else (-2.0, 0.49)
        }
        
        if model_class == TwoFactorBergomiModel:
            initial_params['eta2'] = 0.5
            param_bounds['eta2'] = (0.01, 5.0)
        
        # Calibrate model
        calibrated_params, rmse = calibrate_model(
            model_class, 
            market_iv_short, 
            strikes, 
            short_maturities, 
            S0, 
            param_bounds, 
            initial_params
        )
        
        # Create calibrated model
        calibrated_model = model_class(**calibrated_params)
        
        # Compute option prices and implied volatilities
        option_prices = compute_option_prices(calibrated_model, strikes, short_maturities)
        iv_surface = compute_implied_volatility_surface(option_prices, S0, strikes, short_maturities)
        
        # Compute ATM skew
        atm_skew = compute_atm_skew(iv_surface, strikes, short_maturities, S0)
        
        # Compute skew error
        skew_rmse = np.sqrt(np.mean((atm_skew - market_atm_skew_short)**2))
        
        # Store results
        results_short[model_name] = {
            'calibrated_params': calibrated_params,
            'rmse': rmse,
            'skew_rmse': skew_rmse,
            'iv_surface': iv_surface,
            'atm_skew': atm_skew
        }
        
        print(f"  RMSE: {rmse:.6f}")
        print(f"  Skew RMSE: {skew_rmse:.6f}")
        print(f"  Calibrated parameters: {calibrated_params}")
    
    # Run calibration for all maturities
    print("\nCalibrating models to all maturities...")
    for model_name, model_class in tqdm(models.items()):
        print(f"\nCalibrating {model_name} to all maturities...")
        
        # Initial parameters and bounds
        initial_params = {
            'eta': 1.0,
            'rho': -0.7,
            'H': 0.1
        }
        param_bounds = {
            'eta': (0.01, 5.0),
            'rho': (-0.99, 0.0),
            'H': (0.01, 0.49) if model_class == RoughBergomiModel else (-2.0, 0.49)
        }
        
        if model_class == TwoFactorBergomiModel:
            initial_params['eta2'] = 0.5
            param_bounds['eta2'] = (0.01, 5.0)
        
        # Calibrate model
        calibrated_params, rmse = calibrate_model(
            model_class, 
            market_iv_all, 
            strikes, 
            long_maturities, 
            S0, 
            param_bounds, 
            initial_params
        )
        
        # Create calibrated model
        calibrated_model = model_class(**calibrated_params)
        
        # Compute option prices and implied volatilities
        option_prices = compute_option_prices(calibrated_model, strikes, long_maturities)
        iv_surface = compute_implied_volatility_surface(option_prices, S0, strikes, long_maturities)
        
        # Compute ATM skew
        atm_skew = compute_atm_skew(iv_surface, strikes, long_maturities, S0)
        
        # Compute skew error
        skew_rmse = np.sqrt(np.mean((atm_skew - market_atm_skew_all)**2))
        
        # Store results
        results_all[model_name] = {
            'calibrated_params': calibrated_params,
            'rmse': rmse,
            'skew_rmse': skew_rmse,
            'iv_surface': iv_surface,
            'atm_skew': atm_skew
        }
        
        print(f"  RMSE: {rmse:.6f}")
        print(f"  Skew RMSE: {skew_rmse:.6f}")
        print(f"  Calibrated parameters: {calibrated_params}")
    
    # Plot results for short maturities
    plot_results(results_short, market_iv_short, market_atm_skew_short, strikes, short_maturities, S0, "short")
    
    # Plot results for all maturities
    plot_results(results_all, market_iv_all, market_atm_skew_all, strikes, long_maturities, S0, "all")
    
    # Analyze realized volatility and estimate Hurst index
    print("\nAnalyzing realized volatility and estimating Hurst index...")
    analyze_realized_volatility(models, results_all, S0)


def plot_results(results, market_iv, market_atm_skew, strikes, maturities, S0, maturity_type):
    """
    Plot calibration results.
    
    Parameters:
    -----------
    results : dict
        Calibration results
    market_iv : ndarray
        Market implied volatility surface
    market_atm_skew : ndarray
        Market ATM skew
    strikes : array
        Strike prices
    maturities : array
        Maturities
    S0 : float
        Spot price
    maturity_type : str
        'short' or 'all'
    """
    # Convert maturities to more readable format
    maturity_labels = []
    for T in maturities:
        if T < 1/12:
            maturity_labels.append(f"{int(T*52)}W")
        elif T < 1:
            maturity_labels.append(f"{int(T*12)}M")
        else:
            maturity_labels.append(f"{int(T)}Y")
    
    # Create moneyness grid
    moneyness = strikes / S0
    
    # Plot implied volatility surfaces
    plt.figure(figsize=(15, 10))
    
    # Plot market implied volatility surface
    plt.subplot(2, 3, 1)
    for i, T in enumerate(maturities):
        plt.plot(moneyness, market_iv[i], 'o-', label=maturity_labels[i])
    plt.xlabel('Moneyness (K/S0)')
    plt.ylabel('Implied Volatility')
    plt.title('Market Implied Volatility')
    plt.grid(True)
    plt.legend()
    
    # Plot model implied volatility surfaces
    for i, (model_name, model_result) in enumerate(results.items()):
        plt.subplot(2, 3, i + 2)
        for j, T in enumerate(maturities):
            plt.plot(moneyness, model_result['iv_surface'][j], 'o-', label=maturity_labels[j])
        plt.xlabel('Moneyness (K/S0)')
        plt.ylabel('Implied Volatility')
        plt.title(f'{model_name} (RMSE: {model_result["rmse"]:.6f})')
        plt.grid(True)
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'iv_surfaces_{maturity_type}.png')
    plt.close()
    
    # Plot ATM skews
    plt.figure(figsize=(12, 8))
    
    # Plot ATM skews on normal scale
    plt.subplot(2, 1, 1)
    plt.plot(range(len(maturities)), market_atm_skew, 'ko-', label='Market')
    
    for model_name, model_result in results.items():
        plt.plot(range(len(maturities)), model_result['atm_skew'], 'o-', label=f'{model_name} (RMSE: {model_result["skew_rmse"]:.6f})')
    
    plt.xticks(range(len(maturities)), maturity_labels)
    plt.xlabel('Maturity')
    plt.ylabel('ATM Skew')
    plt.title(f'ATM Skew Comparison ({maturity_type} maturities)')
    plt.grid(True)
    plt.legend()
    
    # Plot ATM skews on log-log scale
    plt.subplot(2, 1, 2)
    plt.loglog(maturities, np.abs(market_atm_skew), 'ko-', label='Market')
    
    for model_name, model_result in results.items():
        plt.loglog(maturities, np.abs(model_result['atm_skew']), 'o-', label=model_name)
    
    plt.xlabel('Maturity (log scale)')
    plt.ylabel('|ATM Skew| (log scale)')
    plt.title(f'ATM Skew Comparison on Log-Log Scale ({maturity_type} maturities)')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'atm_skew_{maturity_type}.png')
    plt.close()
    
    # Compare calibrated parameters
    param_names = ['eta', 'rho', 'H']
    param_values = {param: [] for param in param_names}
    model_names = []
    
    for model_name, model_result in results.items():
        model_names.append(model_name)
        for param in param_names:
            if param in model_result['calibrated_params']:
                param_values[param].append(model_result['calibrated_params'][param])
            else:
                param_values[param].append(np.nan)
    
    # Add eta2 for two-factor model if present
    if 'Two-factor Bergomi' in results and 'eta2' in results['Two-factor Bergomi']['calibrated_params']:
        param_names.append('eta2')
        param_values['eta2'] = [np.nan] * len(model_names)
        param_values['eta2'][model_names.index('Two-factor Bergomi')] = results['Two-factor Bergomi']['calibrated_params']['eta2']
    
    # Plot calibrated parameters
    plt.figure(figsize=(12, 8))
    
    for i, param in enumerate(param_names):
        plt.subplot(2, 2, i + 1)
        plt.bar(model_names, param_values[param])
        plt.xlabel('Model')
        plt.ylabel(param)
        plt.title(f'Calibrated {param}')
        plt.grid(True)
        plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(f'calibrated_params_{maturity_type}.png')
    plt.close()


def analyze_realized_volatility(models, results, S0):
    """
    Analyze realized volatility and estimate Hurst index for different models.
    
    Parameters:
    -----------
    models : dict
        Dictionary of model classes
    results : dict
        Calibration results
    S0 : float
        Spot price
    """
    # Simulate 10 years of daily data
    T = 10.0
    n_steps = int(252 * T)
    n_paths = 1
    
    # Initialize models with calibrated parameters
    calibrated_models = {}
    for model_name, model_class in models.items():
        # Get calibrated parameters
        params = results[model_name]['calibrated_params'].copy()
        
        # Update parameters for long simulation
        params['T'] = T
        params['n_steps'] = n_steps
        params['n_paths'] = n_paths
        
        # Create model
        calibrated_models[model_name] = model_class(**params)
    
    # Simulate paths and compute realized volatility
    rv_data = {}
    for model_name, model in calibrated_models.items():
        print(f"Simulating {model_name}...")
        S, vol = model.simulate()
        
        # Compute daily realized volatility with 21-day window
        rv = compute_realized_volatility(S, window=21)
        rv_data[model_name] = {
            'S': S,
            'vol': vol,
            'rv': rv
        }
    
    # Plot sample paths of spot volatility and realized volatility
    plt.figure(figsize=(15, 10))
    
    for i, (model_name, data) in enumerate(rv_data.items()):
        plt.subplot(2, 2, i + 1)
        
        # Plot spot volatility
        plt.plot(np.linspace(0, T, n_steps + 1), data['vol'][0], 'b-', alpha=0.7, label='Spot Volatility')
        
        # Plot realized volatility
        plt.plot(np.linspace(21/252, T, n_steps - 20), data['rv'][0], 'r-', label='Realized Volatility (21-day)')
        
        plt.xlabel('Time (years)')
        plt.ylabel('Volatility')
        plt.title(f'{model_name}')
        plt.grid(True)
        plt.legend()
    
    plt.tight_layout()
    plt.savefig('volatility_paths.png')
    plt.close()
    
    # Estimate Hurst index for each model
    hurst_estimates = {}
    for model_name, data in rv_data.items():
        print(f"Estimating Hurst index for {model_name}...")
        
        # Estimate Hurst index for spot volatility and realized volatility
        H_spot = estimate_hurst_index(data['vol'][0])
        H_rv = estimate_hurst_index(data['rv'][0])
        
        hurst_estimates[model_name] = {
            'H_spot': H_spot,
            'H_rv': H_rv,
            'calibrated_H': results[model_name]['calibrated_params']['H']
        }
        
        print(f"  Calibrated H: {hurst_estimates[model_name]['calibrated_H']:.4f}")
        print(f"  Estimated H (spot): {H_spot:.4f}")
        print(f"  Estimated H (realized): {H_rv:.4f}")
    
    # Plot Hurst index estimates
    plt.figure(figsize=(12, 6))
    
    model_names = list(hurst_estimates.keys())
    H_calibrated = [hurst_estimates[model]['calibrated_H'] for model in model_names]
    H_spot = [hurst_estimates[model]['H_spot'] for model in model_names]
    H_rv = [hurst_estimates[model]['H_rv'] for model in model_names]
    
    bar_width = 0.25
    r1 = np.arange(len(model_names))
    r2 = [x + bar_width for x in r1]
    r3 = [x + bar_width for x in r2]
    
    plt.bar(r1, H_calibrated, width=bar_width, label='Calibrated H', color='blue')
    plt.bar(r2, H_spot, width=bar_width, label='Estimated H (spot)', color='green')
    plt.bar(r3, H_rv, width=bar_width, label='Estimated H (realized)', color='red')
    
    plt.xlabel('Model')
    plt.ylabel('Hurst Index')
    plt.title('Comparison of Hurst Index Estimates')
    plt.xticks([r + bar_width for r in range(len(model_names))], model_names, rotation=45)
    plt.axhline(y=0.5, color='black', linestyle='--', alpha=0.7)
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('hurst_estimates.png')
    plt.close()
    
    # Test predictive power (parameter stability)
    test_predictive_power(models, results, S0)


def test_predictive_power(models, results, S0):
    """
    Test the predictive power of calibrated models by simulating future volatility surfaces.
    
    Parameters:
    -----------
    models : dict
        Dictionary of model classes
    results : dict
        Calibration results
    S0 : float
        Spot price
    """
    # Setup for prediction test
    prediction_days = 20
    strikes = np.linspace(80, 120, 9)
    maturities = np.array([1/52, 1/12, 2/12, 3/12, 6/12, 1.0])
    
    # Generate new "market" data for future days
    print("\nGenerating future market data for prediction test...")
    market_model = TwoFactorBergomiModel(
        xi0=lambda t: 0.04 * (1 + 0.1 * np.sin(2 * np.pi * t)),  # Time-varying initial variance
        eta1=1.5,
        eta2=0.7,
        rho=-0.7,
        H1=-0.3,
        H2=0.45,
        epsilon=1/52,
        T=1.0 + prediction_days/252,
        n_steps=252 + prediction_days,
        n_paths=10000
    )
    
    # Simulate future market data
    S_future, vol_future = market_model.simulate()
    
    # Store future market implied volatility surfaces
    future_market_iv = []
    for day in range(prediction_days):
        # Shift the forward starting point
        future_model = TwoFactorBergomiModel(
            xi0=lambda t: 0.04 * (1 + 0.1 * np.sin(2 * np.pi * (t + day/252))),
            eta1=1.5,
            eta2=0.7,
            rho=-0.7,
            H1=-0.3,
            H2=0.45,
            epsilon=1/52,
            T=1.0,
            n_steps=252,
            n_paths=10000
        )
        
        # Compute option prices for this future day
        option_prices = compute_option_prices(future_model, strikes, maturities)
        iv_surface = compute_implied_volatility_surface(option_prices, S0, strikes, maturities)
        future_market_iv.append(iv_surface)
    
    # Initialize prediction error storage
    prediction_errors = {model_name: [] for model_name in models}
    
    # Test each model's predictive power
    for model_name, model_class in models.items():
        print(f"\nTesting predictive power of {model_name}...")
        
        # Get calibrated parameters
        params = results[model_name]['calibrated_params'].copy()
        
        # For each future day, compute prediction error
        for day in range(prediction_days):
            # Update forward variance curve for this day
            params['xi0'] = lambda t: 0.04 * (1 + 0.1 * np.sin(2 * np.pi * (t + day/252)))
            params['T'] = 1.0
            params['n_steps'] = 252
            params['n_paths'] = 5000
            
            # Create model with fixed parameters
            fixed_model = model_class(**params)
            
            # Compute predicted implied volatility surface
            option_prices = compute_option_prices(fixed_model, strikes, maturities)
            predicted_iv = compute_implied_volatility_surface(option_prices, S0, strikes, maturities)
            
            # Compute RMSE between predicted and market IV
            rmse = np.sqrt(np.mean((predicted_iv - future_market_iv[day])**2))
            prediction_errors[model_name].append(rmse)
            
            print(f"  Day {day+1}: RMSE = {rmse:.6f}")
    
    # Plot prediction errors
    plt.figure(figsize=(12, 6))
    
    for model_name, errors in prediction_errors.items():
        plt.plot(range(1, prediction_days + 1), errors, 'o-', label=model_name)
    
    plt.xlabel('Days Forward')
    plt.ylabel('Prediction RMSE')
    plt.title('Prediction Error vs. Days Forward')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('prediction_errors.png')
    plt.close()
    
    # Plot average prediction errors
    avg_errors = [np.mean(errors) for model_name, errors in prediction_errors.items()]
    model_names = list(prediction_errors.keys())
    
    plt.figure(figsize=(10, 6))
    plt.bar(model_names, avg_errors)
    plt.xlabel('Model')
    plt.ylabel('Average Prediction RMSE')
    plt.title('Average Prediction Error')
    plt.grid(True)
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('avg_prediction_errors.png')
    plt.close()


if __name__ == "__main__":
    run_empirical_study()