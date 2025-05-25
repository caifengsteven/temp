import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from tqdm import tqdm
from scipy.stats import norm, expon
from scipy.integrate import quad
from sklearn.preprocessing import StandardScaler

# PyTorch imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.nn.functional as F

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Parameters
MINUTES_PER_DAY = 390  # Trading minutes in a day (6.5 hours)
NUM_DAYS = 22  # Trading days per month
NUM_SIMULATION_PATHS = 10000  # Number of paths to simulate for training
BATCH_SIZE = 128  # Batch size for neural network training
EPOCHS = 100  # Number of epochs for training
LEARNING_RATE = 0.001  # Learning rate for Adam optimizer

class BlackScholes:
    """Black-Scholes option pricing model"""
    
    @staticmethod
    def d1(S, K, T, r, sigma):
        return (np.log(S/K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    
    @staticmethod
    def d2(S, K, T, r, sigma):
        return (np.log(S/K) + (r - 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    
    @staticmethod
    def call_price(S, K, T, r, sigma):
        if T <= 0:
            return np.maximum(0, S - K)
        d1 = BlackScholes.d1(S, K, T, r, sigma)
        d2 = BlackScholes.d2(S, K, T, r, sigma)
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    
    @staticmethod
    def put_price(S, K, T, r, sigma):
        if T <= 0:
            return np.maximum(0, K - S)
        d1 = BlackScholes.d1(S, K, T, r, sigma)
        d2 = BlackScholes.d2(S, K, T, r, sigma)
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    
    @staticmethod
    def call_delta(S, K, T, r, sigma):
        if T <= 0:
            return 1.0 if S > K else 0.0
        d1 = BlackScholes.d1(S, K, T, r, sigma)
        return norm.cdf(d1)
    
    @staticmethod
    def put_delta(S, K, T, r, sigma):
        if T <= 0:
            return -1.0 if S < K else 0.0
        d1 = BlackScholes.d1(S, K, T, r, sigma)
        return norm.cdf(d1) - 1
    
    @staticmethod
    def call_gamma(S, K, T, r, sigma):
        if T <= 0:
            return 0.0
        d1 = BlackScholes.d1(S, K, T, r, sigma)
        return norm.pdf(d1) / (S * sigma * np.sqrt(T))
    
    @staticmethod
    def put_gamma(S, K, T, r, sigma):
        return BlackScholes.call_gamma(S, K, T, r, sigma)  # Same for calls and puts
    
    @staticmethod
    def call_vega(S, K, T, r, sigma):
        if T <= 0:
            return 0.0
        d1 = BlackScholes.d1(S, K, T, r, sigma)
        return S * np.sqrt(T) * norm.pdf(d1) / 100  # Divided by 100 for percentage point change
    
    @staticmethod
    def put_vega(S, K, T, r, sigma):
        return BlackScholes.call_vega(S, K, T, r, sigma)  # Same for calls and puts
    
    @staticmethod
    def call_theta(S, K, T, r, sigma):
        if T <= 0:
            return 0.0
        d1 = BlackScholes.d1(S, K, T, r, sigma)
        d2 = BlackScholes.d2(S, K, T, r, sigma)
        return (-S * sigma * norm.pdf(d1)) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)
    
    @staticmethod
    def put_theta(S, K, T, r, sigma):
        if T <= 0:
            return 0.0
        d1 = BlackScholes.d1(S, K, T, r, sigma)
        d2 = BlackScholes.d2(S, K, T, r, sigma)
        return (-S * sigma * norm.pdf(d1)) / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2)

class HawkesProcess:
    """Implementation of Hawkes process with power law kernel"""
    
    def __init__(self, mu_func, alpha, beta, delta, T_end):
        """
        Initialize Hawkes process with power law kernel
        
        Parameters:
        -----------
        mu_func : function
            Time-varying baseline intensity function
        alpha : float
            Scale parameter for the power law kernel
        beta : float
            Shape parameter for the power law kernel
        delta : float
            Shift parameter for the power law kernel
        T_end : float
            End time for simulation
        """
        self.mu_func = mu_func
        self.alpha = alpha
        self.beta = beta
        self.delta = delta
        self.T_end = T_end
        self.events = []
    
    def intensity(self, t):
        """
        Calculate the intensity at time t
        
        Parameters:
        -----------
        t : float
            Time at which to calculate the intensity
            
        Returns:
        --------
        float
            Intensity at time t
        """
        baseline = self.mu_func(t)
        
        # Sum over all past events
        excitation = 0
        for s in self.events:
            if s < t:
                excitation += self.alpha / ((t - s + self.delta) ** self.beta)
        
        return baseline + excitation
    
    def simulate(self):
        """
        Simulate the Hawkes process using thinning method
        
        Returns:
        --------
        list
            List of event times
        """
        # Clear events
        self.events = []
        
        # Initialize
        t = 0
        
        # Sample time points to evaluate the baseline intensity
        time_points = np.linspace(0, self.T_end, 1000)
        
        # Find maximum baseline intensity
        max_mu = max([self.mu_func(t) for t in time_points])
        
        # Upper bound for intensity
        lambda_max = max_mu  # Initial upper bound
        
        while t < self.T_end:
            # Draw next candidate event time
            u = np.random.exponential(scale=1/lambda_max)
            t = t + u
            
            if t > self.T_end:
                break
            
            # Calculate current intensity
            lambda_t = self.intensity(t)
            
            # Accept or reject
            if np.random.uniform(0, 1) < lambda_t / lambda_max:
                self.events.append(t)
                
                # Update upper bound for intensity
                lambda_max = max(lambda_max, lambda_t + self.alpha / (self.delta ** self.beta))
            
        return np.array(self.events)
    
    def expected_number(self, t_start, t_end, num_points=100):
        """
        Calculate the expected number of events in interval [t_start, t_end]
        using numerical solution of Volterra equation
        
        Parameters:
        -----------
        t_start : float
            Start time
        t_end : float
            End time
        num_points : int
            Number of points for numerical solution
            
        Returns:
        --------
        float
            Expected number of events in interval [t_start, t_end]
        """
        if not self.events or t_start >= t_end:
            return 0
        
        # Duration
        duration = t_end - t_start
        
        # Function for the extended intensity
        def nu(u):
            t = t_start + u
            return self.mu_func(t) + sum(self.alpha / ((t - s + self.delta) ** self.beta) 
                                       for s in self.events if s <= t_start)
        
        # Function for the kernel
        def h(x):
            return self.alpha / ((x + self.delta) ** self.beta)
        
        # Solve Volterra equation numerically
        dt = duration / num_points
        t_grid = np.linspace(0, duration, num_points)
        psi = np.ones(num_points)
        
        for i in range(1, num_points):
            integral = 0
            for j in range(i):
                integral += h(t_grid[i] - t_grid[j]) * psi[j] * dt
            psi[i] = 1 + integral
        
        # Calculate expected number using numerical integration
        expected_num = 0
        for i in range(num_points):
            expected_num += nu(t_grid[num_points-1-i]) * psi[i] * dt
        
        return expected_num

class PoissonProcess:
    """Implementation of homogeneous Poisson process"""
    
    def __init__(self, intensity, T_end):
        """
        Initialize Poisson process
        
        Parameters:
        -----------
        intensity : float
            Constant intensity
        T_end : float
            End time for simulation
        """
        self.intensity = intensity
        self.T_end = T_end
    
    def simulate(self):
        """
        Simulate the Poisson process
        
        Returns:
        --------
        numpy.ndarray
            Array of event times
        """
        # Number of events
        N = np.random.poisson(self.intensity * self.T_end)
        
        # Event times (uniformly distributed)
        events = np.random.uniform(0, self.T_end, N)
        
        # Sort events
        events.sort()
        
        return events

def create_piecewise_intensity_function(intensity_values, breakpoints):
    """
    Create a piecewise linear intensity function
    
    Parameters:
    -----------
    intensity_values : list
        List of intensity values at breakpoints
    breakpoints : list
        List of breakpoints
        
    Returns:
    --------
    function
        Piecewise intensity function
    """
    def intensity_function(t):
        # Find the correct interval
        for i in range(len(breakpoints) - 1):
            if breakpoints[i] <= t < breakpoints[i+1]:
                # Linear interpolation
                return intensity_values[i] + (intensity_values[i+1] - intensity_values[i]) * \
                       (t - breakpoints[i]) / (breakpoints[i+1] - breakpoints[i])
        
        # If t is outside the range, return the first or last value
        if t < breakpoints[0]:
            return intensity_values[0]
        else:
            return intensity_values[-1]
    
    return intensity_function

def simulate_stock_prices(S0, mu, sigma, T, dt):
    """
    Simulate stock prices using geometric Brownian motion
    
    Parameters:
    -----------
    S0 : float
        Initial stock price
    mu : float
        Drift
    sigma : float
        Volatility
    T : float
        Time horizon
    dt : float
        Time step
        
    Returns:
    --------
    numpy.ndarray
        Array of stock prices
    """
    # Number of steps
    n_steps = int(T / dt)
    
    # Initialize stock prices
    S = np.zeros(n_steps + 1)
    S[0] = S0
    
    # Simulate stock prices
    for i in range(1, n_steps + 1):
        S[i] = S[i-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * np.random.normal())
    
    return S

def simulate_market_data(option_type, S0, K, T, r, sigma, dt, hawkes_params, kappa=10):
    """
    Simulate market data for option market making
    
    Parameters:
    -----------
    option_type : str
        Option type ('call' or 'put')
    S0 : float
        Initial stock price
    K : float
        Strike price
    T : float
        Time to maturity (in years)
    r : float
        Risk-free rate
    sigma : float
        Volatility
    dt : float
        Time step (in years)
    hawkes_params : dict
        Parameters for Hawkes processes (buy and sell)
    kappa : float
        Price impact parameter
        
    Returns:
    --------
    dict
        Dictionary with simulated market data
    """
    # Simulate stock prices
    stock_prices = simulate_stock_prices(S0, r, sigma, T, dt)
    time_grid = np.arange(0, T + dt, dt)
    
    # Calculate option prices and greeks
    n_steps = len(stock_prices)
    option_prices = np.zeros(n_steps)
    option_deltas = np.zeros(n_steps)
    option_gammas = np.zeros(n_steps)
    option_vegas = np.zeros(n_steps)
    
    for i in range(n_steps):
        t = time_grid[i]
        S = stock_prices[i]
        tau = max(0, T - t)  # Time to maturity
        
        if option_type == 'call':
            option_prices[i] = BlackScholes.call_price(S, K, tau, r, sigma)
            option_deltas[i] = BlackScholes.call_delta(S, K, tau, r, sigma)
            option_gammas[i] = BlackScholes.call_gamma(S, K, tau, r, sigma)
            option_vegas[i] = BlackScholes.call_vega(S, K, tau, r, sigma)
        else:  # put
            option_prices[i] = BlackScholes.put_price(S, K, tau, r, sigma)
            option_deltas[i] = BlackScholes.put_delta(S, K, tau, r, sigma)
            option_gammas[i] = BlackScholes.put_gamma(S, K, tau, r, sigma)
            option_vegas[i] = BlackScholes.put_vega(S, K, tau, r, sigma)
    
    # Simulate market buy orders using Hawkes process
    buy_hawkes = HawkesProcess(
        hawkes_params['buy_mu_func'],
        hawkes_params['buy_alpha'],
        hawkes_params['buy_beta'],
        hawkes_params['buy_delta'],
        T
    )
    buy_times = buy_hawkes.simulate()
    
    # Simulate market sell orders using Hawkes process
    sell_hawkes = HawkesProcess(
        hawkes_params['sell_mu_func'],
        hawkes_params['sell_alpha'],
        hawkes_params['sell_beta'],
        hawkes_params['sell_delta'],
        T
    )
    sell_times = sell_hawkes.simulate()
    
    # Calculate market buy/sell counts in each time interval
    buy_counts = np.zeros(n_steps - 1)
    sell_counts = np.zeros(n_steps - 1)
    
    for i in range(n_steps - 1):
        t_start = time_grid[i]
        t_end = time_grid[i+1]
        
        buy_counts[i] = np.sum((buy_times >= t_start) & (buy_times < t_end))
        sell_counts[i] = np.sum((sell_times >= t_start) & (sell_times < t_end))
    
    # Calculate Hawkes intensities at each decision time
    buy_intensities = np.zeros(n_steps - 1)
    sell_intensities = np.zeros(n_steps - 1)
    
    # Calculate expected number of orders in each interval
    buy_expected_counts = np.zeros(n_steps - 1)
    sell_expected_counts = np.zeros(n_steps - 1)
    
    for i in range(n_steps - 1):
        # Record past events for intensity calculation
        buy_hawkes.events = buy_times[buy_times < time_grid[i]].tolist()
        sell_hawkes.events = sell_times[sell_times < time_grid[i]].tolist()
        
        # Calculate intensities
        buy_intensities[i] = buy_hawkes.intensity(time_grid[i])
        sell_intensities[i] = sell_hawkes.intensity(time_grid[i])
        
        # Calculate expected counts
        buy_expected_counts[i] = buy_hawkes.expected_number(time_grid[i], time_grid[i+1])
        sell_expected_counts[i] = sell_hawkes.expected_number(time_grid[i], time_grid[i+1])
    
    return {
        'time_grid': time_grid,
        'stock_prices': stock_prices,
        'option_prices': option_prices,
        'option_deltas': option_deltas,
        'option_gammas': option_gammas,
        'option_vegas': option_vegas,
        'buy_times': buy_times,
        'sell_times': sell_times,
        'buy_counts': buy_counts,
        'sell_counts': sell_counts,
        'buy_intensities': buy_intensities,
        'sell_intensities': sell_intensities,
        'buy_expected_counts': buy_expected_counts,
        'sell_expected_counts': sell_expected_counts
    }

class FeedForwardNN(nn.Module):
    """Feed-forward neural network for market making strategy"""
    
    def __init__(self, input_dim, hidden_dim=64):
        """
        Initialize feed-forward neural network
        
        Parameters:
        -----------
        input_dim : int
            Input dimension
        hidden_dim : int
            Hidden layer dimension
        """
        super(FeedForwardNN, self).__init__()
        
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        """
        Forward pass
        
        Parameters:
        -----------
        x : torch.Tensor
            Input tensor
            
        Returns:
        --------
        torch.Tensor
            Output tensor
        """
        return self.layers(x)

class MarketMakingStrategy:
    """Market making strategy for a single option"""
    
    def __init__(self, strategy_type, gamma=0, feature_type='none', kappa=10):
        """
        Initialize market making strategy
        
        Parameters:
        -----------
        strategy_type : str
            Strategy type ('neural_network', 'constant')
        gamma : float
            Risk aversion parameter
        feature_type : str
            Type of feature to use ('none', 'intensity', 'expected_count', 'exact_count')
        kappa : float
            Price impact parameter
        """
        self.strategy_type = strategy_type
        self.gamma = gamma
        self.feature_type = feature_type
        self.kappa = kappa
        
        # Initialize neural networks and other parameters
        self.ask_networks = []
        self.bid_networks = []
        self.constant_ask = None
        self.constant_bid = None
    
    def initialize_networks(self, n_periods, input_dim):
        """
        Initialize neural networks for each period
        
        Parameters:
        -----------
        n_periods : int
            Number of periods
        input_dim : int
            Input dimension
        """
        if self.strategy_type == 'neural_network':
            self.ask_networks = [FeedForwardNN(input_dim).to(device) for _ in range(n_periods)]
            self.bid_networks = [FeedForwardNN(input_dim).to(device) for _ in range(n_periods)]
    
    def set_constant_quotes(self, ask, bid):
        """
        Set constant quotes
        
        Parameters:
        -----------
        ask : float
            Ask quote (distance from mid-price)
        bid : float
            Bid quote (distance from mid-price)
        """
        if self.strategy_type == 'constant':
            self.constant_ask = ask
            self.constant_bid = bid
    
    def get_quotes(self, period, state, features=None):
        """
        Get quotes for a given period and state
        
        Parameters:
        -----------
        period : int
            Current period
        state : tuple
            Current state (S, C, x, q_o, q_s)
        features : dict or None
            Additional features
            
        Returns:
        --------
        tuple
            (ask_quote, bid_quote)
        """
        if self.strategy_type == 'constant':
            return self.constant_ask, self.constant_bid
        
        elif self.strategy_type == 'neural_network':
            S, C, x, q_o, q_s = state
            
            # Create input for neural networks
            if self.feature_type == 'none':
                ask_input = torch.tensor([S, C, x, q_o, q_s], dtype=torch.float32).to(device)
                bid_input = torch.tensor([S, C, x, q_o, q_s], dtype=torch.float32).to(device)
            
            elif self.feature_type == 'intensity':
                ask_input = torch.tensor([S, C, x, q_o, q_s, features['sell_intensity']], dtype=torch.float32).to(device)
                bid_input = torch.tensor([S, C, x, q_o, q_s, features['buy_intensity']], dtype=torch.float32).to(device)
            
            elif self.feature_type == 'expected_count':
                ask_input = torch.tensor([S, C, x, q_o, q_s, features['sell_expected_count']], dtype=torch.float32).to(device)
                bid_input = torch.tensor([S, C, x, q_o, q_s, features['buy_expected_count']], dtype=torch.float32).to(device)
            
            elif self.feature_type == 'exact_count':
                ask_input = torch.tensor([S, C, x, q_o, q_s, features['sell_count']], dtype=torch.float32).to(device)
                bid_input = torch.tensor([S, C, x, q_o, q_s, features['buy_count']], dtype=torch.float32).to(device)
            
            # Get quotes from neural networks
            self.ask_networks[period].eval()
            self.bid_networks[period].eval()
            
            with torch.no_grad():
                ask_quote = self.ask_networks[period](ask_input).item()
                bid_quote = self.bid_networks[period](bid_input).item()
            
            return ask_quote, bid_quote
    
    def train(self, train_data, validation_data=None, epochs=100, batch_size=128, lr=0.001):
        """
        Train neural networks
        
        Parameters:
        -----------
        train_data : list
            List of training data paths
        validation_data : list or None
            List of validation data paths
        epochs : int
            Number of epochs
        batch_size : int
            Batch size
        lr : float
            Learning rate
        """
        if self.strategy_type != 'neural_network':
            return
        
        n_periods = len(self.ask_networks)
        
        # Create optimizers
        ask_optimizers = [optim.Adam(net.parameters(), lr=lr) for net in self.ask_networks]
        bid_optimizers = [optim.Adam(net.parameters(), lr=lr) for net in self.bid_networks]
        
        # Training loop
        for epoch in range(epochs):
            # Shuffle training data
            indices = torch.randperm(len(train_data))
            
            # Mini-batch training
            for i in range(0, len(train_data), batch_size):
                batch_indices = indices[i:i+batch_size]
                batch_data = [train_data[idx] for idx in batch_indices]
                
                # Process each period separately
                for period in range(n_periods):
                    # Collect states and rewards
                    states = []
                    rewards = []
                    
                    for path in batch_data:
                        # Extract state and feature information
                        S = path['stock_prices'][period]
                        C = path['option_prices'][period]
                        x = path['cash'][period] if 'cash' in path else 5000  # Initial cash if not in path
                        q_o = path['option_inventory'][period] if 'option_inventory' in path else 10  # Initial inventory if not in path
                        q_s = path['stock_inventory'][period] if 'stock_inventory' in path else -q_o * path['option_deltas'][period]  # Delta hedge if not in path
                        
                        if self.feature_type == 'intensity':
                            sell_feature = path['sell_intensities'][period]
                            buy_feature = path['buy_intensities'][period]
                        elif self.feature_type == 'expected_count':
                            sell_feature = path['sell_expected_counts'][period]
                            buy_feature = path['buy_expected_counts'][period]
                        elif self.feature_type == 'exact_count':
                            sell_feature = path['sell_counts'][period]
                            buy_feature = path['buy_counts'][period]
                        else:  # 'none'
                            sell_feature = 0
                            buy_feature = 0
                        
                        # Store state and feature information
                        states.append({
                            'S': S, 'C': C, 'x': x, 'q_o': q_o, 'q_s': q_s,
                            'sell_feature': sell_feature, 'buy_feature': buy_feature
                        })
                        
                        # Store reward (terminal utility)
                        terminal_utility = path['terminal_utility'] if 'terminal_utility' in path else 0
                        rewards.append(terminal_utility)
                    
                    # Convert rewards to tensor
                    rewards_tensor = torch.tensor(rewards, dtype=torch.float32).to(device)
                    
                    # Train ask network
                    ask_optimizers[period].zero_grad()
                    
                    # Forward pass for ask network
                    ask_inputs = []
                    for state in states:
                        if self.feature_type == 'none':
                            ask_input = [state['S'], state['C'], state['x'], state['q_o'], state['q_s']]
                        else:
                            ask_input = [state['S'], state['C'], state['x'], state['q_o'], state['q_s'], state['sell_feature']]
                        ask_inputs.append(ask_input)
                    
                    ask_inputs_tensor = torch.tensor(ask_inputs, dtype=torch.float32).to(device)
                    ask_outputs = self.ask_networks[period](ask_inputs_tensor)
                    
                    # Backward pass for ask network
                    ask_loss = -torch.mean(rewards_tensor * ask_outputs.squeeze())
                    ask_loss.backward()
                    ask_optimizers[period].step()
                    
                    # Train bid network
                    bid_optimizers[period].zero_grad()
                    
                    # Forward pass for bid network
                    bid_inputs = []
                    for state in states:
                        if self.feature_type == 'none':
                            bid_input = [state['S'], state['C'], state['x'], state['q_o'], state['q_s']]
                        else:
                            bid_input = [state['S'], state['C'], state['x'], state['q_o'], state['q_s'], state['buy_feature']]
                        bid_inputs.append(bid_input)
                    
                    bid_inputs_tensor = torch.tensor(bid_inputs, dtype=torch.float32).to(device)
                    bid_outputs = self.bid_networks[period](bid_inputs_tensor)
                    
                    # Backward pass for bid network
                    bid_loss = -torch.mean(rewards_tensor * bid_outputs.squeeze())
                    bid_loss.backward()
                    bid_optimizers[period].step()
            
            # Print progress
            if epoch % 10 == 0:
                print(f"Epoch {epoch}/{epochs}, Ask Loss: {ask_loss.item():.4f}, Bid Loss: {bid_loss.item():.4f}")
        
        print("Training completed.")
    
    def find_optimal_constant_quotes(self, train_data):
        """
        Find optimal constant quotes based on training data
        
        Parameters:
        -----------
        train_data : list
            List of training data paths
        """
        if self.strategy_type != 'constant':
            return
        
        # Grid of quotes to search
        ask_grid = np.linspace(0.01, 1.0, 20)  # Reduced grid size for efficiency
        bid_grid = np.linspace(0.01, 1.0, 20)
        
        best_ask = 0.5
        best_bid = 0.5
        best_utility = float('-inf')
        
        print("Finding optimal constant quotes...")
        
        # Evaluate each combination of ask and bid quotes
        for ask in tqdm(ask_grid):
            for bid in bid_grid:
                total_utility = 0
                
                # Evaluate on a subset of training data for efficiency
                subset_size = min(500, len(train_data))
                subset_indices = np.random.choice(len(train_data), subset_size, replace=False)
                subset_data = [train_data[i] for i in subset_indices]
                
                for path in subset_data:
                    # Simulate market making with constant quotes
                    utility = self.simulate_constant_strategy(path, ask, bid)
                    total_utility += utility
                
                average_utility = total_utility / len(subset_data)
                
                # Update best quotes
                if average_utility > best_utility:
                    best_utility = average_utility
                    best_ask = ask
                    best_bid = bid
        
        self.constant_ask = best_ask
        self.constant_bid = best_bid
        print(f"Optimal constant quotes: Ask = {best_ask:.4f}, Bid = {best_bid:.4f}")
    
    def simulate_constant_strategy(self, path, ask, bid):
        """
        Simulate market making with constant quotes
        
        Parameters:
        -----------
        path : dict
            Market data for one path
        ask : float
            Ask quote (distance from mid-price)
        bid : float
            Bid quote (distance from mid-price)
            
        Returns:
        --------
        float
            Terminal utility
        """
        # Extract data
        n_periods = len(path['stock_prices']) - 1
        stock_prices = path['stock_prices']
        option_prices = path['option_prices']
        option_deltas = path['option_deltas']
        buy_counts = path['buy_counts']
        sell_counts = path['sell_counts']
        
        # Initialize inventories and cash
        q_o = 10  # Initial option inventory
        q_s = -q_o * option_deltas[0]  # Initial stock inventory (delta-hedged)
        x = 5000  # Initial cash
        
        # Simulate market making
        for period in range(n_periods):
            # Current mid-price
            C = option_prices[period]
            
            # Limit order prices
            p_ask = C + ask
            p_bid = C - bid
            
            # Execution probability for ask quote
            p_ask_exec = 1 - (1 - np.exp(-self.kappa * ask)) ** sell_counts[period]
            
            # Execution probability for bid quote
            p_bid_exec = 1 - (1 - np.exp(-self.kappa * bid)) ** buy_counts[period]
            
            # Limit order execution
            ask_executed = np.random.random() < p_ask_exec
            bid_executed = np.random.random() < p_bid_exec
            
            # Update option inventory
            if ask_executed:
                q_o -= 1
            if bid_executed:
                q_o += 1
            
            # Update stock inventory (delta-hedging)
            q_s_next = -q_o * option_deltas[period + 1]
            delta_q_s = q_s_next - q_s
            q_s = q_s_next
            
            # Update cash
            if ask_executed:
                x += p_ask
            if bid_executed:
                x -= p_bid
            
            x -= delta_q_s * stock_prices[period + 1]
        
        # Terminal wealth
        W_T = x + q_o * option_prices[-1] + q_s * stock_prices[-1]
        W_0 = 5000 + 10 * option_prices[0] - 10 * option_deltas[0] * stock_prices[0]
        P_L = W_T - W_0
        
        # Utility
        if self.gamma == 0:  # Risk-neutral
            return P_L
        else:  # Risk-averse
            return (1 - np.exp(-self.gamma * P_L)) / self.gamma

def simulate_market_making(strategy, market_data, initial_option_inventory=10, initial_cash=5000):
    """
    Simulate market making with a given strategy
    
    Parameters:
    -----------
    strategy : MarketMakingStrategy
        Market making strategy
    market_data : dict
        Market data
    initial_option_inventory : int
        Initial option inventory
    initial_cash : float
        Initial cash
        
    Returns:
    --------
    dict
        Dictionary with simulation results
    """
    # Extract data
    time_grid = market_data['time_grid']
    stock_prices = market_data['stock_prices']
    option_prices = market_data['option_prices']
    option_deltas = market_data['option_deltas']
    buy_counts = market_data['buy_counts']
    sell_counts = market_data['sell_counts']
    buy_intensities = market_data['buy_intensities']
    sell_intensities = market_data['sell_intensities']
    buy_expected_counts = market_data['buy_expected_counts']
    sell_expected_counts = market_data['sell_expected_counts']
    
    # Initialize
    n_periods = len(time_grid) - 1
    q_o = initial_option_inventory
    q_s = -q_o * option_deltas[0]
    x = initial_cash
    
    # Arrays to store results
    cash = np.zeros(n_periods + 1)
    option_inventory = np.zeros(n_periods + 1)
    stock_inventory = np.zeros(n_periods + 1)
    ask_quotes = np.zeros(n_periods)
    bid_quotes = np.zeros(n_periods)
    executions_ask = np.zeros(n_periods, dtype=bool)
    executions_bid = np.zeros(n_periods, dtype=bool)
    
    # Initial values
    cash[0] = x
    option_inventory[0] = q_o
    stock_inventory[0] = q_s
    
    # Simulate market making
    for period in range(n_periods):
        # Current state
        S = stock_prices[period]
        C = option_prices[period]
        
        # Additional features
        features = {
            'buy_intensity': buy_intensities[period],
            'sell_intensity': sell_intensities[period],
            'buy_expected_count': buy_expected_counts[period],
            'sell_expected_count': sell_expected_counts[period],
            'buy_count': buy_counts[period],
            'sell_count': sell_counts[period]
        }
        
        # Get quotes from strategy
        ask, bid = strategy.get_quotes(period, (S, C, x, q_o, q_s), features)
        ask_quotes[period] = ask
        bid_quotes[period] = bid
        
        # Limit order prices
        p_ask = C + ask
        p_bid = C - bid
        
        # Execution probability for ask quote
        p_ask_exec = 1 - (1 - np.exp(-strategy.kappa * ask)) ** sell_counts[period]
        
        # Execution probability for bid quote
        p_bid_exec = 1 - (1 - np.exp(-strategy.kappa * bid)) ** buy_counts[period]
        
        # Limit order execution
        ask_executed = np.random.random() < p_ask_exec
        bid_executed = np.random.random() < p_bid_exec
        
        executions_ask[period] = ask_executed
        executions_bid[period] = bid_executed
        
        # Update option inventory
        if ask_executed:
            q_o -= 1
        if bid_executed:
            q_o += 1
        
        # Update stock inventory (delta-hedging)
        q_s_next = -q_o * option_deltas[period + 1]
        delta_q_s = q_s_next - q_s
        q_s = q_s_next
        
        # Update cash
        if ask_executed:
            x += p_ask
        if bid_executed:
            x -= p_bid
        
        x -= delta_q_s * stock_prices[period + 1]
        
        # Store values
        cash[period + 1] = x
        option_inventory[period + 1] = q_o
        stock_inventory[period + 1] = q_s
    
    # Terminal wealth
    W_T = x + q_o * option_prices[-1] + q_s * stock_prices[-1]
    W_0 = initial_cash + initial_option_inventory * option_prices[0] - initial_option_inventory * option_deltas[0] * stock_prices[0]
    P_L = W_T - W_0
    
    # Utility
    if strategy.gamma == 0:  # Risk-neutral
        utility = P_L
    else:  # Risk-averse
        utility = (1 - np.exp(-strategy.gamma * P_L)) / strategy.gamma
    
    return {
        'time_grid': time_grid,
        'stock_prices': stock_prices,
        'option_prices': option_prices,
        'cash': cash,
        'option_inventory': option_inventory,
        'stock_inventory': stock_inventory,
        'ask_quotes': ask_quotes,
        'bid_quotes': bid_quotes,
        'executions_ask': executions_ask,
        'executions_bid': executions_bid,
        'terminal_wealth': W_T,
        'initial_wealth': W_0,
        'profit_loss': P_L,
        'terminal_utility': utility
    }

# Parameters for U-shaped baseline intensity (similar to AAPL in the paper)
def create_u_shaped_intensity(max_intensity=25, min_intensity=2.5, n_periods=MINUTES_PER_DAY):
    """
    Create U-shaped baseline intensity function
    
    Parameters:
    -----------
    max_intensity : float
        Maximum intensity at market open and close
    min_intensity : float
        Minimum intensity during mid-day
    n_periods : int
        Number of periods in a day
        
    Returns:
    --------
    function
        U-shaped intensity function
    """
    # Breakpoints
    # Define 7 breakpoints for 6 intervals (as in the paper)
    breakpoints = np.linspace(0, n_periods / MINUTES_PER_DAY, 7)
    
    # Intensity values at breakpoints (U-shaped)
    intensity_values = [
        max_intensity,                  # Market open
        0.2 * max_intensity + 0.8 * min_intensity,  # Decreasing
        0.1 * max_intensity + 0.9 * min_intensity,  # Approaching midday
        min_intensity,                   # Midday
        0.1 * max_intensity + 0.9 * min_intensity,  # Increasing
        0.2 * max_intensity + 0.8 * min_intensity,  # Towards close
        max_intensity                   # Market close
    ]
    
    return create_piecewise_intensity_function(intensity_values, breakpoints)

# Parameters for L-shaped baseline intensity (similar to BAC in the paper)
def create_l_shaped_intensity(max_intensity=10, mid_intensity=5, min_intensity=2, n_periods=MINUTES_PER_DAY):
    """
    Create L-shaped baseline intensity function
    
    Parameters:
    -----------
    max_intensity : float
        Maximum intensity at market open
    mid_intensity : float
        Middle intensity
    min_intensity : float
        Minimum intensity during late day
    n_periods : int
        Number of periods in a day
        
    Returns:
    --------
    function
        L-shaped intensity function
    """
    # Breakpoints for 3 intervals (as mentioned for BAC in the paper)
    breakpoints = [0, n_periods / (3 * MINUTES_PER_DAY), 2 * n_periods / (3 * MINUTES_PER_DAY), n_periods / MINUTES_PER_DAY]
    
    # Intensity values at breakpoints (L-shaped)
    intensity_values = [max_intensity, mid_intensity, min_intensity, min_intensity]
    
    return create_piecewise_intensity_function(intensity_values, breakpoints)

def run_experiment():
    """
    Run the market making experiment and compare strategies
    """
    # Option parameters
    option_type = 'call'
    S0 = 160  # Initial stock price (similar to AAPL in the paper)
    K = 162  # Strike price (slightly OTM)
    T = 10/252  # Time to maturity (10 trading days)
    r = 0.01  # Risk-free rate
    sigma = 0.3  # Volatility
    
    # Market making parameters
    dt = 1/MINUTES_PER_DAY  # Time step (1 minute in trading day)
    n_periods = int(T / dt)  # Number of periods
    kappa = 10  # Price impact parameter
    gamma = 0  # Risk aversion parameter (risk-neutral)
    
    # Hawkes process parameters (based on AAPL values from the paper)
    # Buy orders
    buy_mu_func = create_u_shaped_intensity(max_intensity=25, min_intensity=2.5, n_periods=n_periods)
    buy_alpha = 0.04
    buy_beta = 1.2
    buy_delta = 0.003
    
    # Sell orders
    sell_mu_func = create_u_shaped_intensity(max_intensity=25, min_intensity=2.5, n_periods=n_periods)
    sell_alpha = 0.03
    sell_beta = 1.2
    sell_delta = 0.05
    
    # Poisson process parameters (constant intensity)
    poisson_intensity = 10  # Average intensity throughout the day
    
    # Hawkes process parameters
    hawkes_params = {
        'buy_mu_func': buy_mu_func,
        'buy_alpha': buy_alpha,
        'buy_beta': buy_beta,
        'buy_delta': buy_delta,
        'sell_mu_func': sell_mu_func,
        'sell_alpha': sell_alpha,
        'sell_beta': sell_beta,
        'sell_delta': sell_delta
    }
    
    # Generate training data with Hawkes process
    print("Generating training data with Hawkes process...")
    hawkes_train_data = []
    
    for i in tqdm(range(NUM_SIMULATION_PATHS)):
        # Simulate market data
        market_data = simulate_market_data(
            option_type, S0, K, T, r, sigma, dt, hawkes_params, kappa
        )
        
        # Store data
        hawkes_train_data.append(market_data)
    
    # Generate training data with Poisson process
    print("Generating training data with Poisson process...")
    poisson_train_data = []
    
    for i in tqdm(range(NUM_SIMULATION_PATHS)):
        # Create Poisson process parameters
        poisson_params = {
            'buy_mu_func': lambda t: poisson_intensity,
            'buy_alpha': 0,
            'buy_beta': 1,
            'buy_delta': 0.001,
            'sell_mu_func': lambda t: poisson_intensity,
            'sell_alpha': 0,
            'sell_beta': 1,
            'sell_delta': 0.001
        }
        
        # Simulate market data
        market_data = simulate_market_data(
            option_type, S0, K, T, r, sigma, dt, poisson_params, kappa
        )
        
        # Store data
        poisson_train_data.append(market_data)
    
    # Simulate market making on each training path to get terminal utilities
    print("Simulating market making on training data...")
    
    # Function to simulate and add terminal utility to a dataset
    def add_terminal_utilities(dataset, strategy_type='neural_network', feature_type='none'):
        # Create a temporary strategy for simulation
        temp_strategy = MarketMakingStrategy(strategy_type, gamma, feature_type, kappa)
        
        # Set constant values for the constant strategy case
        if strategy_type == 'constant':
            temp_strategy.set_constant_quotes(0.1, 0.1)  # Initial placeholder values
        
        # Process each path
        for i, path in enumerate(tqdm(dataset)):
            # Simulate market making
            result = simulate_market_making(temp_strategy, path)
            
            # Add terminal utility to the path
            path['terminal_utility'] = result['terminal_utility']
            
            # Add cash, option inventory, and stock inventory values at each period
            path['cash'] = result['cash']
            path['option_inventory'] = result['option_inventory']
            path['stock_inventory'] = result['stock_inventory']
    
    # Add terminal utilities to Hawkes dataset
    add_terminal_utilities(hawkes_train_data)
    
    # Add terminal utilities to Poisson dataset
    add_terminal_utilities(poisson_train_data)
    
    # Create strategies to compare
    strategies = {
        # Strategy 1: LIF (Looking-Into-Future)
        'LIF': MarketMakingStrategy('neural_network', gamma, 'exact_count', kappa),
        
        # Strategy 2: ExpN (Expected Number)
        'ExpN': MarketMakingStrategy('neural_network', gamma, 'expected_count', kappa),
        
        # Strategy 3: Intensity
        'Intensity': MarketMakingStrategy('neural_network', gamma, 'intensity', kappa),
        
        # Strategy 4: NoF (No Feature)
        'NoF': MarketMakingStrategy('neural_network', gamma, 'none', kappa),
        
        # Strategy 5: Poisson
        'Poisson': MarketMakingStrategy('neural_network', gamma, 'none', kappa),
        
        # Strategy 6: Constant
        'Constant': MarketMakingStrategy('constant', gamma, 'none', kappa)
    }
    
    # Initialize neural networks
    for name, strategy in strategies.items():
        if name in ['LIF', 'ExpN', 'Intensity', 'NoF']:
            # Determine input dimension
            if name == 'NoF':
                input_dim = 5  # (S, C, x, q_o, q_s)
            else:
                input_dim = 6  # (S, C, x, q_o, q_s, feature)
            
            strategy.initialize_networks(n_periods, input_dim)
    
    # Train strategies
    print("Training strategies...")
    
    # Split Hawkes train data into training and validation sets
    train_size = int(0.8 * len(hawkes_train_data))
    hawkes_train = hawkes_train_data[:train_size]
    hawkes_valid = hawkes_train_data[train_size:]
    
    # Split Poisson train data into training and validation sets
    poisson_train = poisson_train_data[:train_size]
    poisson_valid = poisson_train_data[train_size:]
    
    # Train LIF, ExpN, Intensity, NoF with Hawkes data
    for name in ['LIF', 'ExpN', 'Intensity', 'NoF']:
        print(f"Training {name} strategy...")
        strategies[name].train(hawkes_train, hawkes_valid, EPOCHS, BATCH_SIZE, LEARNING_RATE)
    
    # Train Poisson with Poisson data
    print("Training Poisson strategy...")
    strategies['Poisson'].train(poisson_train, poisson_valid, EPOCHS, BATCH_SIZE, LEARNING_RATE)
    
    # Find optimal constant quotes
    print("Finding optimal constant quotes...")
    strategies['Constant'].find_optimal_constant_quotes(hawkes_train)
    
    # Generate test data with Hawkes process
    print("Generating test data with Hawkes process...")
    test_data = []
    
    num_test_paths = 1000
    for i in tqdm(range(num_test_paths)):
        # Simulate market data
        market_data = simulate_market_data(
            option_type, S0, K, T, r, sigma, dt, hawkes_params, kappa
        )
        
        # Store data
        test_data.append(market_data)
    
    # Test strategies
    print("Testing strategies...")
    results = {}
    
    for name, strategy in strategies.items():
        print(f"Testing {name} strategy...")
        
        strategy_results = []
        for i in tqdm(range(len(test_data))):
            # Simulate market making
            result = simulate_market_making(strategy, test_data[i])
            
            # Store results
            strategy_results.append(result)
        
        # Calculate statistics
        profits = [result['profit_loss'] for result in strategy_results]
        utilities = [result['terminal_utility'] for result in strategy_results]
        
        results[name] = {
            'profits': profits,
            'utilities': utilities,
            'mean_profit': np.mean(profits),
            'std_profit': np.std(profits),
            'mean_utility': np.mean(utilities),
            'std_utility': np.std(utilities),
            'min_profit': np.min(profits),
            'max_profit': np.max(profits),
            'quantile_01': np.quantile(profits, 0.01),
            'quantile_99': np.quantile(profits, 0.99)
        }
    
    # Print results
    print("\nResults:")
    print("-" * 80)
    print(f"{'Strategy':<10} {'Mean Profit':<15} {'Std Profit':<15} {'Mean Utility':<15} {'1% Quantile':<15} {'99% Quantile':<15}")
    print("-" * 80)
    
    for name, result in results.items():
        print(f"{name:<10} {result['mean_profit']:<15.4f} {result['std_profit']:<15.4f} {result['mean_utility']:<15.4f} {result['quantile_01']:<15.4f} {result['quantile_99']:<15.4f}")
    
    # Plot results
    plt.figure(figsize=(12, 8))
    
    for name, result in results.items():
        sns.kdeplot(result['profits'], label=name)
    
    plt.title('Profit Distribution by Strategy')
    plt.xlabel('Profit')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('profit_distribution.png')
    plt.close()
    
    # Plot mean profits with error bars
    plt.figure(figsize=(10, 6))
    
    names = list(results.keys())
    means = [results[name]['mean_profit'] for name in names]
    stds = [results[name]['std_profit'] for name in names]
    
    plt.bar(names, means, yerr=stds, alpha=0.7, capsize=10)
    plt.title('Mean Profit by Strategy')
    plt.xlabel('Strategy')
    plt.ylabel('Mean Profit')
    plt.grid(True, alpha=0.3, axis='y')
    plt.savefig('mean_profit.png')
    plt.close()
    
    # Plot strategy quotes
    # Pick a random test path
    test_path_idx = np.random.randint(0, len(test_data))
    test_path = test_data[test_path_idx]
    
    plt.figure(figsize=(12, 8))
    
    for name, strategy in strategies.items():
        # Simulate market making
        result = simulate_market_making(strategy, test_path)
        
        # Plot ask quotes
        plt.plot(result['time_grid'][:-1], result['ask_quotes'], label=f'{name} Ask')
    
    plt.title('Ask Quotes by Strategy')
    plt.xlabel('Time')
    plt.ylabel('Distance from Mid-Price')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('ask_quotes.png')
    plt.close()
    
    plt.figure(figsize=(12, 8))
    
    for name, strategy in strategies.items():
        # Simulate market making
        result = simulate_market_making(strategy, test_path)
        
        # Plot bid quotes
        plt.plot(result['time_grid'][:-1], result['bid_quotes'], label=f'{name} Bid')
    
    plt.title('Bid Quotes by Strategy')
    plt.xlabel('Time')
    plt.ylabel('Distance from Mid-Price')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('bid_quotes.png')
    plt.close()
    
    # Plot inventory for each strategy
    plt.figure(figsize=(12, 8))
    
    for name, strategy in strategies.items():
        # Simulate market making
        result = simulate_market_making(strategy, test_path)
        
        # Plot option inventory
        plt.plot(result['time_grid'], result['option_inventory'], label=name)
    
    plt.title('Option Inventory by Strategy')
    plt.xlabel('Time')
    plt.ylabel('Option Inventory')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('option_inventory.png')
    plt.close()
    
    # Return results for further analysis
    return results, strategies, test_data

# Run the experiment
results, strategies, test_data = run_experiment()

# Additional analysis
def compare_strategy_pair(results, strategy1, strategy2):
    """
    Compare two strategies statistically
    
    Parameters:
    -----------
    results : dict
        Results dictionary
    strategy1 : str
        First strategy name
    strategy2 : str
        Second strategy name
    """
    # Calculate differences in profits
    profits1 = results[strategy1]['profits']
    profits2 = results[strategy2]['profits']
    
    differences = np.array(profits1) - np.array(profits2)
    
    # Calculate confidence interval for the difference
    mean_diff = np.mean(differences)
    std_diff = np.std(differences)
    n = len(differences)
    
    # 99% confidence interval
    z = 2.576  # z-score for 99% CI
    margin = z * std_diff / np.sqrt(n)
    
    ci_lower = mean_diff - margin
    ci_upper = mean_diff + margin
    
    print(f"Comparison of {strategy1} vs {strategy2}:")
    print(f"Mean difference in profit: {mean_diff:.4f}")
    print(f"99% Confidence interval: ({ci_lower:.4f}, {ci_upper:.4f})")
    print(f"Statistically significant difference: {'Yes' if ci_lower > 0 or ci_upper < 0 else 'No'}")
    print()

# Compare all strategies against each other
print("\nStrategy Comparisons:")
print("-" * 80)

strategy_names = list(results.keys())
for i in range(len(strategy_names)):
    for j in range(i+1, len(strategy_names)):
        compare_strategy_pair(results, strategy_names[i], strategy_names[j])

# Save the strategy example from the paper (Figure 6)
def plot_strategy_example():
    """Plot strategy example similar to Figure 6 in the paper"""
    # Generate a new test path
    # Option parameters
    option_type = 'call'
    S0 = 160
    K = 162
    T = 10/252
    r = 0.01
    sigma = 0.3
    
    # Market making parameters
    dt = 1/MINUTES_PER_DAY
    kappa = 10
    
    # Hawkes process parameters
    buy_mu_func = create_u_shaped_intensity(max_intensity=25, min_intensity=2.5)
    buy_alpha = 0.04
    buy_beta = 1.2
    buy_delta = 0.003
    
    sell_mu_func = create_u_shaped_intensity(max_intensity=25, min_intensity=2.5)
    sell_alpha = 0.03
    sell_beta = 1.2
    sell_delta = 0.05
    
    hawkes_params = {
        'buy_mu_func': buy_mu_func,
        'buy_alpha': buy_alpha,
        'buy_beta': buy_beta,
        'buy_delta': buy_delta,
        'sell_mu_func': sell_mu_func,
        'sell_alpha': sell_alpha,
        'sell_beta': sell_beta,
        'sell_delta': sell_delta
    }
    
    # Simulate market data
    market_data = simulate_market_data(
        option_type, S0, K, T, r, sigma, dt, hawkes_params, kappa
    )
    
    # Plot strategy quotes for one day
    # Plot in 4 subplots as in the paper
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot market buy intensities
    axs[0, 0].plot(market_data['time_grid'][:-1], market_data['buy_intensities'], 'b-')
    axs[0, 0].set_title('Market Buy Order Intensity')
    axs[0, 0].set_xlabel('Time (days)')
    axs[0, 0].set_ylabel('Intensity')
    axs[0, 0].grid(True, alpha=0.3)
    
    # Plot market sell intensities
    axs[0, 1].plot(market_data['time_grid'][:-1], market_data['sell_intensities'], 'r-')
    axs[0, 1].set_title('Market Sell Order Intensity')
    axs[0, 1].set_xlabel('Time (days)')
    axs[0, 1].set_ylabel('Intensity')
    axs[0, 1].grid(True, alpha=0.3)
    
    # Plot ask quotes
    for name, strategy in strategies.items():
        if name in ['ExpN', 'Intensity', 'NoF', 'Poisson']:
            result = simulate_market_making(strategy, market_data)
            axs[1, 0].plot(market_data['time_grid'][:-1], result['ask_quotes'], label=name)
    
    axs[1, 0].set_title('Ask Quotes by Strategy')
    axs[1, 0].set_xlabel('Time (days)')
    axs[1, 0].set_ylabel('Distance from Mid-Price')
    axs[1, 0].legend()
    axs[1, 0].grid(True, alpha=0.3)
    
    # Plot bid quotes
    for name, strategy in strategies.items():
        if name in ['ExpN', 'Intensity', 'NoF', 'Poisson']:
            result = simulate_market_making(strategy, market_data)
            axs[1, 1].plot(market_data['time_grid'][:-1], result['bid_quotes'], label=name)
    
    axs[1, 1].set_title('Bid Quotes by Strategy')
    axs[1, 1].set_xlabel('Time (days)')
    axs[1, 1].set_ylabel('Distance from Mid-Price')
    axs[1, 1].legend()
    axs[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('strategy_example.png')
    plt.close()

# Plot strategy example
plot_strategy_example()

# Extension: Multiple options with constraints
class PortfolioMarketMakingStrategy:
    """Portfolio market making strategy for multiple options"""
    
    def __init__(self, gamma=0, feature_type='intensity', kappa=10, constraints=None):
        """
        Initialize portfolio market making strategy
        
        Parameters:
        -----------
        gamma : float
            Risk aversion parameter
        feature_type : str
            Type of feature to use ('none', 'intensity', 'expected_count')
        kappa : float
            Price impact parameter
        constraints : dict or None
            Constraints on Greeks and inventory
        """
        self.gamma = gamma
        self.feature_type = feature_type
        self.kappa = kappa
        self.constraints = constraints if constraints is not None else {}
        
        # Initialize neural networks
        self.option_networks = {}  # Dict of {option_id: {'ask': [networks], 'bid': [networks]}}
    
    def initialize_networks(self, option_ids, n_periods, input_dim):
        """
        Initialize neural networks for each option and period
        
        Parameters:
        -----------
        option_ids : list
            List of option IDs
        n_periods : int
            Number of periods
        input_dim : int
            Input dimension
        """
        for option_id in option_ids:
            self.option_networks[option_id] = {
                'ask': [FeedForwardNN(input_dim).to(device) for _ in range(n_periods)],
                'bid': [FeedForwardNN(input_dim).to(device) for _ in range(n_periods)]
            }
    
    def get_quotes(self, period, option_id, state, features=None):
        """
        Get quotes for a given period, option, and state
        
        Parameters:
        -----------
        period : int
            Current period
        option_id : int
            Option ID
        state : dict
            Current state including:
            - 'S': stock price
            - 'C': option prices
            - 'x': cash
            - 'q_o': option inventories
            - 'q_s': stock inventory
            - 'gamma': portfolio gamma
            - 'vega': portfolio vega
        features : dict or None
            Additional features
            
        Returns:
        --------
        tuple
            (ask_quote, bid_quote)
        """
        # Extract state variables
        S = state['S']
        C = state['C'][option_id]
        x = state['x']
        q_o = state['q_o'][option_id]
        q_s = state['q_s']
        gamma = state.get('gamma', 0)
        vega = state.get('vega', 0)
        
        # Create input for neural networks
        if self.feature_type == 'none':
            ask_input = torch.tensor([S, C, x, q_o, q_s, gamma, vega], dtype=torch.float32).to(device)
            bid_input = torch.tensor([S, C, x, q_o, q_s, gamma, vega], dtype=torch.float32).to(device)
        
        elif self.feature_type == 'intensity':
            ask_input = torch.tensor([S, C, x, q_o, q_s, gamma, vega, 
                                     features['sell_intensity'][option_id]], dtype=torch.float32).to(device)
            bid_input = torch.tensor([S, C, x, q_o, q_s, gamma, vega,
                                     features['buy_intensity'][option_id]], dtype=torch.float32).to(device)
        
        elif self.feature_type == 'expected_count':
            ask_input = torch.tensor([S, C, x, q_o, q_s, gamma, vega,
                                     features['sell_expected_count'][option_id]], dtype=torch.float32).to(device)
            bid_input = torch.tensor([S, C, x, q_o, q_s, gamma, vega,
                                     features['buy_expected_count'][option_id]], dtype=torch.float32).to(device)
        
        # Get quotes from neural networks
        self.option_networks[option_id]['ask'][period].eval()
        self.option_networks[option_id]['bid'][period].eval()
        
        with torch.no_grad():
            ask_quote = self.option_networks[option_id]['ask'][period](ask_input).item()
            bid_quote = self.option_networks[option_id]['bid'][period](bid_input).item()
        
        return ask_quote, bid_quote
    
    def train(self, train_data, validation_data=None, epochs=100, batch_size=128, lr=0.001):
        """
        Train neural networks for portfolio market making
        
        Parameters:
        -----------
        train_data : list
            List of training data paths
        validation_data : list or None
            List of validation data paths
        epochs : int
            Number of epochs
        batch_size : int
            Batch size
        lr : float
            Learning rate
        """
        option_ids = list(self.option_networks.keys())
        n_periods = len(self.option_networks[option_ids[0]]['ask'])
        
        # Create optimizers
        optimizers = {}
        for option_id in option_ids:
            optimizers[option_id] = {
                'ask': [optim.Adam(net.parameters(), lr=lr) for net in self.option_networks[option_id]['ask']],
                'bid': [optim.Adam(net.parameters(), lr=lr) for net in self.option_networks[option_id]['bid']]
            }
        
        # Define loss function with constraints
        def constrained_loss(terminal_utility, terminal_state):
            loss = -terminal_utility
            
            # Add penalties for violating constraints
            if 'max_gamma' in self.constraints:
                gamma_penalty = F.relu(torch.abs(terminal_state['gamma']) - self.constraints['max_gamma']) ** 2
                loss += 2000 * gamma_penalty
            
            if 'max_vega' in self.constraints:
                vega_penalty = F.relu(torch.abs(terminal_state['vega']) - self.constraints['max_vega']) ** 2
                loss += 2000 * vega_penalty
            
            for option_id in option_ids:
                if f'max_inventory_{option_id}' in self.constraints:
                    inv_penalty = F.relu(torch.abs(terminal_state['q_o'][option_id]) - 
                                      self.constraints[f'max_inventory_{option_id}']) ** 2
                    loss += 2000 * inv_penalty
            
            return loss
        
        # Training loop
        for epoch in range(epochs):
            # Shuffle training data
            indices = torch.randperm(len(train_data))
            
            # Mini-batch training
            for i in range(0, len(train_data), batch_size):
                batch_indices = indices[i:i+batch_size]
                batch_data = [train_data[idx] for idx in batch_indices]
                
                # Process each period separately
                for period in range(n_periods):
                    # Process each option separately
                    for option_id in option_ids:
                        # Collect states and rewards
                        states = []
                        terminal_states = []
                        terminal_utilities = []
                        
                        for path in batch_data:
                            # Extract state and feature information
                            state = {
                                'S': path['stock_prices'][period],
                                'C': {opt_id: path['option_prices'][opt_id][period] for opt_id in option_ids},
                                'x': path['cash'][period],
                                'q_o': {opt_id: path['option_inventory'][opt_id][period] for opt_id in option_ids},
                                'q_s': path['stock_inventory'][period],
                                'gamma': path['gamma'][period],
                                'vega': path['vega'][period]
                            }
                            
                            # Extract feature information
                            if self.feature_type == 'intensity':
                                sell_feature = {opt_id: path['sell_intensities'][opt_id][period] for opt_id in option_ids}
                                buy_feature = {opt_id: path['buy_intensities'][opt_id][period] for opt_id in option_ids}
                            elif self.feature_type == 'expected_count':
                                sell_feature = {opt_id: path['sell_expected_counts'][opt_id][period] for opt_id in option_ids}
                                buy_feature = {opt_id: path['buy_expected_counts'][opt_id][period] for opt_id in option_ids}
                            else:  # 'none'
                                sell_feature = {opt_id: 0 for opt_id in option_ids}
                                buy_feature = {opt_id: 0 for opt_id in option_ids}
                            
                            # Store state and feature information
                            states.append({
                                'state': state,
                                'sell_feature': sell_feature,
                                'buy_feature': buy_feature
                            })
                            
                            # Store terminal state and utility
                            terminal_state = {
                                'gamma': path['gamma'][-1],
                                'vega': path['vega'][-1],
                                'q_o': {opt_id: path['option_inventory'][opt_id][-1] for opt_id in option_ids}
                            }
                            terminal_states.append(terminal_state)
                            terminal_utilities.append(path['terminal_utility'])
                        
                        # Convert to tensors
                        terminal_utilities_tensor = torch.tensor(terminal_utilities, dtype=torch.float32).to(device)
                        terminal_gammas = torch.tensor([s['gamma'] for s in terminal_states], dtype=torch.float32).to(device)
                        terminal_vegas = torch.tensor([s['vega'] for s in terminal_states], dtype=torch.float32).to(device)
                        terminal_inventories = {
                            opt_id: torch.tensor([s['q_o'][opt_id] for s in terminal_states], dtype=torch.float32).to(device)
                            for opt_id in option_ids
                        }
                        
                        terminal_states_tensor = {
                            'gamma': terminal_gammas,
                            'vega': terminal_vegas,
                            'q_o': terminal_inventories
                        }
                        
                        # Train ask network
                        optimizers[option_id]['ask'][period].zero_grad()
                        
                        # Forward pass for ask network
                        ask_inputs = []
                        for state_data in states:
                            state = state_data['state']
                            if self.feature_type == 'none':
                                ask_input = [state['S'], state['C'][option_id], state['x'], 
                                          state['q_o'][option_id], state['q_s'], state['gamma'], state['vega']]
                            else:
                                ask_input = [state['S'], state['C'][option_id], state['x'], 
                                          state['q_o'][option_id], state['q_s'], state['gamma'], state['vega'],
                                          state_data['sell_feature'][option_id]]
                            ask_inputs.append(ask_input)
                        
                        ask_inputs_tensor = torch.tensor(ask_inputs, dtype=torch.float32).to(device)
                        ask_outputs = self.option_networks[option_id]['ask'][period](ask_inputs_tensor)
                        
                        # Backward pass for ask network with constraints
                        ask_loss = constrained_loss(terminal_utilities_tensor, terminal_states_tensor)
                        ask_loss.backward()
                        optimizers[option_id]['ask'][period].step()
                        
                        # Train bid network
                        optimizers[option_id]['bid'][period].zero_grad()
                        
                        # Forward pass for bid network
                        bid_inputs = []
                        for state_data in states:
                            state = state_data['state']
                            if self.feature_type == 'none':
                                bid_input = [state['S'], state['C'][option_id], state['x'], 
                                          state['q_o'][option_id], state['q_s'], state['gamma'], state['vega']]
                            else:
                                bid_input = [state['S'], state['C'][option_id], state['x'], 
                                          state['q_o'][option_id], state['q_s'], state['gamma'], state['vega'],
                                          state_data['buy_feature'][option_id]]
                            bid_inputs.append(bid_input)
                        
                        bid_inputs_tensor = torch.tensor(bid_inputs, dtype=torch.float32).to(device)
                        bid_outputs = self.option_networks[option_id]['bid'][period](bid_inputs_tensor)
                        
                        # Backward pass for bid network with constraints
                        bid_loss = constrained_loss(terminal_utilities_tensor, terminal_states_tensor)
                        bid_loss.backward()
                        optimizers[option_id]['bid'][period].step()
            
            # Print progress
            if epoch % 10 == 0:
                print(f"Epoch {epoch}/{epochs}, Ask Loss: {ask_loss.item():.4f}, Bid Loss: {bid_loss.item():.4f}")
        
        print("Training completed.")

def simulate_portfolio_market_data(option_types, S0, Ks, Ts, r, sigma, dt, hawkes_params, kappa=10):
    """
    Simulate market data for multiple options
    
    Parameters:
    -----------
    option_types : list
        List of option types ('call' or 'put')
    S0 : float
        Initial stock price
    Ks : list
        List of strike prices
    Ts : list
        List of times to maturity (in years)
    r : float
        Risk-free rate
    sigma : float
        Volatility
    dt : float
        Time step (in years)
    hawkes_params : dict
        Parameters for Hawkes processes
    kappa : float
        Price impact parameter
        
    Returns:
    --------
    dict
        Dictionary with simulated market data for multiple options
    """
    # Validate input
    n_options = len(option_types)
    assert len(Ks) == n_options and len(Ts) == n_options, "Inconsistent option parameters"
    
    # Simulate stock prices
    stock_prices = simulate_stock_prices(S0, r, sigma, max(Ts), dt)
    time_grid = np.arange(0, max(Ts) + dt, dt)
    
    # Initialize data structures
    option_prices = {}
    option_deltas = {}
    option_gammas = {}
    option_vegas = {}
    buy_times = {}
    sell_times = {}
    buy_counts = {}
    sell_counts = {}
    buy_intensities = {}
    sell_intensities = {}
    buy_expected_counts = {}
    sell_expected_counts = {}
    
    # Process each option
    for i in range(n_options):
        option_id = i
        option_type = option_types[i]
        K = Ks[i]
        T = Ts[i]
        
        # Calculate option prices and Greeks
        n_steps = min(len(stock_prices), int(T / dt) + 1)
        prices = np.zeros(n_steps)
        deltas = np.zeros(n_steps)
        gammas = np.zeros(n_steps)
        vegas = np.zeros(n_steps)
        
        for j in range(n_steps):
            t = time_grid[j]
            S = stock_prices[j]
            tau = max(0, T - t)  # Time to maturity
            
            if option_type == 'call':
                prices[j] = BlackScholes.call_price(S, K, tau, r, sigma)
                deltas[j] = BlackScholes.call_delta(S, K, tau, r, sigma)
                gammas[j] = BlackScholes.call_gamma(S, K, tau, r, sigma)
                vegas[j] = BlackScholes.call_vega(S, K, tau, r, sigma)
            else:  # put
                prices[j] = BlackScholes.put_price(S, K, tau, r, sigma)
                deltas[j] = BlackScholes.put_delta(S, K, tau, r, sigma)
                gammas[j] = BlackScholes.put_gamma(S, K, tau, r, sigma)
                vegas[j] = BlackScholes.put_vega(S, K, tau, r, sigma)
        
        option_prices[option_id] = prices
        option_deltas[option_id] = deltas
        option_gammas[option_id] = gammas
        option_vegas[option_id] = vegas
        
        # Simulate market order arrivals
        buy_hawkes = HawkesProcess(
            hawkes_params['buy_mu_func'],
            hawkes_params['buy_alpha'],
            hawkes_params['buy_beta'],
            hawkes_params['buy_delta'],
            T
        )
        buy_times[option_id] = buy_hawkes.simulate()
        
        sell_hawkes = HawkesProcess(
            hawkes_params['sell_mu_func'],
            hawkes_params['sell_alpha'],
            hawkes_params['sell_beta'],
            hawkes_params['sell_delta'],
            T
        )
        sell_times[option_id] = sell_hawkes.simulate()
        
        # Calculate market order counts, intensities, and expected counts
        buy_counts[option_id] = np.zeros(n_steps - 1)
        sell_counts[option_id] = np.zeros(n_steps - 1)
        buy_intensities[option_id] = np.zeros(n_steps - 1)
        sell_intensities[option_id] = np.zeros(n_steps - 1)
        buy_expected_counts[option_id] = np.zeros(n_steps - 1)
        sell_expected_counts[option_id] = np.zeros(n_steps - 1)
        
        for j in range(n_steps - 1):
            t_start = time_grid[j]
            t_end = time_grid[j+1]
            
            buy_counts[option_id][j] = np.sum((buy_times[option_id] >= t_start) & (buy_times[option_id] < t_end))
            sell_counts[option_id][j] = np.sum((sell_times[option_id] >= t_start) & (sell_times[option_id] < t_end))
            
            # Record past events for intensity calculation
            buy_hawkes.events = buy_times[option_id][buy_times[option_id] < t_start].tolist()
            sell_hawkes.events = sell_times[option_id][sell_times[option_id] < t_start].tolist()
            
            # Calculate intensities
            buy_intensities[option_id][j] = buy_hawkes.intensity(t_start)
            sell_intensities[option_id][j] = sell_hawkes.intensity(t_start)
            
            # Calculate expected counts
            buy_expected_counts[option_id][j] = buy_hawkes.expected_number(t_start, t_end)
            sell_expected_counts[option_id][j] = sell_hawkes.expected_number(t_start, t_end)
    
    return {
        'time_grid': time_grid,
        'stock_prices': stock_prices,
        'option_prices': option_prices,
        'option_deltas': option_deltas,
        'option_gammas': option_gammas,
        'option_vegas': option_vegas,
        'buy_times': buy_times,
        'sell_times': sell_times,
        'buy_counts': buy_counts,
        'sell_counts': sell_counts,
        'buy_intensities': buy_intensities,
        'sell_intensities': sell_intensities,
        'buy_expected_counts': buy_expected_counts,
        'sell_expected_counts': sell_expected_counts
    }

def simulate_portfolio_market_making(strategy, market_data, initial_option_inventory=10, initial_cash=5000):
    """
    Simulate portfolio market making with a given strategy
    
    Parameters:
    -----------
    strategy : PortfolioMarketMakingStrategy
        Portfolio market making strategy
    market_data : dict
        Market data for multiple options
    initial_option_inventory : int
        Initial inventory for each option
    initial_cash : float
        Initial cash
        
    Returns:
    --------
    dict
        Dictionary with simulation results
    """
    # Extract data
    time_grid = market_data['time_grid']
    stock_prices = market_data['stock_prices']
    option_ids = list(market_data['option_prices'].keys())
    n_options = len(option_ids)
    
    # Initialize
    n_periods = len(time_grid) - 1
    q_o = {option_id: initial_option_inventory for option_id in option_ids}
    q_s = -sum(q_o[option_id] * market_data['option_deltas'][option_id][0] for option_id in option_ids)
    x = initial_cash
    
    # Arrays to store results
    cash = np.zeros(n_periods + 1)
    option_inventory = {option_id: np.zeros(n_periods + 1) for option_id in option_ids}
    stock_inventory = np.zeros(n_periods + 1)
    gamma = np.zeros(n_periods + 1)
    vega = np.zeros(n_periods + 1)
    ask_quotes = {option_id: np.zeros(n_periods) for option_id in option_ids}
    bid_quotes = {option_id: np.zeros(n_periods) for option_id in option_ids}
    executions_ask = {option_id: np.zeros(n_periods, dtype=bool) for option_id in option_ids}
    executions_bid = {option_id: np.zeros(n_periods, dtype=bool) for option_id in option_ids}
    
    # Initial values
    cash[0] = x
    for option_id in option_ids:
        option_inventory[option_id][0] = q_o[option_id]
    stock_inventory[0] = q_s
    
    # Calculate initial gamma and vega
    gamma[0] = sum(q_o[option_id] * market_data['option_gammas'][option_id][0] for option_id in option_ids)
    vega[0] = sum(q_o[option_id] * market_data['option_vegas'][option_id][0] for option_id in option_ids)
    
    # Simulate market making
    for period in range(n_periods):
        # Current state
        S = stock_prices[period]
        C = {option_id: market_data['option_prices'][option_id][period] for option_id in option_ids}
        
        # Current gammas and vegas
        current_gamma = gamma[period]
        current_vega = vega[period]
        
        # State dictionary
        state = {
            'S': S,
            'C': C,
            'x': x,
            'q_o': q_o,
            'q_s': q_s,
            'gamma': current_gamma,
            'vega': current_vega
        }
        
        # Additional features
        features = {
            'buy_intensity': {option_id: market_data['buy_intensities'][option_id][period] for option_id in option_ids},
            'sell_intensity': {option_id: market_data['sell_intensities'][option_id][period] for option_id in option_ids},
            'buy_expected_count': {option_id: market_data['buy_expected_counts'][option_id][period] for option_id in option_ids},
            'sell_expected_count': {option_id: market_data['sell_expected_counts'][option_id][period] for option_id in option_ids},
            'buy_count': {option_id: market_data['buy_counts'][option_id][period] for option_id in option_ids},
            'sell_count': {option_id: market_data['sell_counts'][option_id][period] for option_id in option_ids}
        }
        
        # Process each option
        for option_id in option_ids:
            # Get quotes from strategy
            ask, bid = strategy.get_quotes(period, option_id, state, features)
            ask_quotes[option_id][period] = ask
            bid_quotes[option_id][period] = bid
            
            # Limit order prices
            p_ask = C[option_id] + ask
            p_bid = C[option_id] - bid
            
            # Execution probability for ask quote
            p_ask_exec = 1 - (1 - np.exp(-strategy.kappa * ask)) ** market_data['sell_counts'][option_id][period]
            
            # Execution probability for bid quote
            p_bid_exec = 1 - (1 - np.exp(-strategy.kappa * bid)) ** market_data['buy_counts'][option_id][period]
            
            # Limit order execution
            ask_executed = np.random.random() < p_ask_exec
            bid_executed = np.random.random() < p_bid_exec
            
            executions_ask[option_id][period] = ask_executed
            executions_bid[option_id][period] = bid_executed
            
            # Update option inventory
            if ask_executed:
                q_o[option_id] -= 1
            if bid_executed:
                q_o[option_id] += 1
            
            # Update cash
            if ask_executed:
                x += p_ask
            if bid_executed:
                x -= p_bid
        
        # Update stock inventory (delta-hedging)
        q_s_next = -sum(q_o[option_id] * market_data['option_deltas'][option_id][period + 1] for option_id in option_ids)
        delta_q_s = q_s_next - q_s
        q_s = q_s_next
        
        # Update cash for stock trading
        x -= delta_q_s * stock_prices[period + 1]
        
        # Calculate new gamma and vega
        current_gamma = sum(q_o[option_id] * market_data['option_gammas'][option_id][period + 1] for option_id in option_ids)
        current_vega = sum(q_o[option_id] * market_data['option_vegas'][option_id][period + 1] for option_id in option_ids)
        
        # Store values
        cash[period + 1] = x
        for option_id in option_ids:
            option_inventory[option_id][period + 1] = q_o[option_id]
        stock_inventory[period + 1] = q_s
        gamma[period + 1] = current_gamma
        vega[period + 1] = current_vega
    
    # Terminal wealth
    W_T = x
    for option_id in option_ids:
        W_T += q_o[option_id] * market_data['option_prices'][option_id][-1]
    W_T += q_s * stock_prices[-1]
    
    # Initial wealth
    W_0 = initial_cash
    for option_id in option_ids:
        W_0 += initial_option_inventory * market_data['option_prices'][option_id][0]
    W_0 -= sum(initial_option_inventory * market_data['option_deltas'][option_id][0] for option_id in option_ids) * stock_prices[0]
    
    P_L = W_T - W_0
    
    # Utility
    if strategy.gamma == 0:  # Risk-neutral
        utility = P_L
    else:  # Risk-averse
        utility = (1 - np.exp(-strategy.gamma * P_L)) / strategy.gamma
    
    # Check constraint violations
    gamma_violation = False
    vega_violation = False
    inventory_violation = False
    
    if 'max_gamma' in strategy.constraints:
        gamma_violation = abs(gamma[-1]) > strategy.constraints['max_gamma']
    
    if 'max_vega' in strategy.constraints:
        vega_violation = abs(vega[-1]) > strategy.constraints['max_vega']
    
    for option_id in option_ids:
        if f'max_inventory_{option_id}' in strategy.constraints:
            if abs(q_o[option_id]) > strategy.constraints[f'max_inventory_{option_id}']:
                inventory_violation = True
    
    return {
        'time_grid': time_grid,
        'stock_prices': stock_prices,
        'option_prices': {option_id: market_data['option_prices'][option_id] for option_id in option_ids},
        'cash': cash,
        'option_inventory': option_inventory,
        'stock_inventory': stock_inventory,
        'gamma': gamma,
        'vega': vega,
        'ask_quotes': ask_quotes,
        'bid_quotes': bid_quotes,
        'executions_ask': executions_ask,
        'executions_bid': executions_bid,
        'terminal_wealth': W_T,
        'initial_wealth': W_0,
        'profit_loss': P_L,
        'terminal_utility': utility,
        'gamma_violation': gamma_violation,
        'vega_violation': vega_violation,
        'inventory_violation': inventory_violation
    }

def run_portfolio_experiment():
    """
    Run the portfolio market making experiment with constraints
    """
    # Option parameters
    option_types = ['call', 'call']  # Two call options
    S0 = 160  # Initial stock price
    Ks = [155, 165]  # Strike prices
    Ts = [10/252, 10/252]  # Time to maturity (10 trading days)
    r = 0.01  # Risk-free rate
    sigma = 0.3  # Volatility
    
    # Market making parameters
    dt = 1/MINUTES_PER_DAY  # Time step (1 minute in trading day)
    n_periods = int(max(Ts) / dt)  # Number of periods
    kappa = 10  # Price impact parameter
    gamma = 0  # Risk aversion parameter (risk-neutral)
    
    # Hawkes process parameters
    buy_mu_func = create_u_shaped_intensity(max_intensity=25, min_intensity=2.5, n_periods=n_periods)
    buy_alpha = 0.04
    buy_beta = 1.2
    buy_delta = 0.003
    
    sell_mu_func = create_u_shaped_intensity(max_intensity=25, min_intensity=2.5, n_periods=n_periods)
    sell_alpha = 0.03
    sell_beta = 1.2
    sell_delta = 0.05
    
    # Hawkes process parameters
    hawkes_params = {
        'buy_mu_func': buy_mu_func,
        'buy_alpha': buy_alpha,
        'buy_beta': buy_beta,
        'buy_delta': buy_delta,
        'sell_mu_func': sell_mu_func,
        'sell_alpha': sell_alpha,
        'sell_beta': sell_beta,
        'sell_delta': sell_delta
    }
    
    # Generate training data
    print("Generating portfolio training data...")
    train_data = []
    
    for i in tqdm(range(500)):  # Fewer paths for computational efficiency
        # Simulate market data
        market_data = simulate_portfolio_market_data(
            option_types, S0, Ks, Ts, r, sigma, dt, hawkes_params, kappa
        )
        
        # Store data
        train_data.append(market_data)
    
    # Process each path to add cash, inventory, and utility
    for path in tqdm(train_data, desc="Processing training data"):
        # Initialize cash, inventory, gamma, and vega
        option_ids = list(path['option_prices'].keys())
        initial_option_inventory = 10
        initial_cash = 5000
        
        n_periods = len(path['time_grid'])
        cash = np.zeros(n_periods)
        option_inventory = {option_id: np.zeros(n_periods) for option_id in option_ids}
        stock_inventory = np.zeros(n_periods)
        gamma = np.zeros(n_periods)
        vega = np.zeros(n_periods)
        
        # Set initial values
        cash[0] = initial_cash
        for option_id in option_ids:
            option_inventory[option_id][0] = initial_option_inventory
        
        # Initial stock inventory (delta-hedged)
        stock_inventory[0] = -sum(initial_option_inventory * path['option_deltas'][option_id][0] for option_id in option_ids)
        
        # Calculate initial gamma and vega
        gamma[0] = sum(initial_option_inventory * path['option_gammas'][option_id][0] for option_id in option_ids)
        vega[0] = sum(initial_option_inventory * path['option_vegas'][option_id][0] for option_id in option_ids)
        
        # Add to path
        path['cash'] = cash
        path['option_inventory'] = option_inventory
        path['stock_inventory'] = stock_inventory
        path['gamma'] = gamma
        path['vega'] = vega
        
        # Add terminal utility (placeholder)
        path['terminal_utility'] = 0
    
    # Define constraints for three settings as in the paper
    constraint_settings = [
        {},  # Setting 1: No constraints
        {'max_gamma': 5, 'max_vega': 5, 'max_inventory_0': 20, 'max_inventory_1': 20},  # Setting 2
        {'max_gamma': 3, 'max_vega': 3, 'max_inventory_0': 10, 'max_inventory_1': 10}   # Setting 3
    ]
    
    strategies = []
    
    for i, constraints in enumerate(constraint_settings):
        print(f"Creating strategy for Setting {i+1}...")
        
        # Create portfolio strategy
        strategy = PortfolioMarketMakingStrategy(
            gamma=gamma,
            feature_type='intensity',
            kappa=kappa,
            constraints=constraints
        )
        
        # Initialize networks
        input_dim = 8  # S, C, x, q_o, q_s, gamma, vega, feature
        strategy.initialize_networks([0, 1], n_periods, input_dim)
        
        # Simulate market making to get terminal utilities for each setting
        for path in tqdm(train_data, desc=f"Simulating market making for Setting {i+1}"):
            result = simulate_portfolio_market_making(strategy, path)
            path['terminal_utility'] = result['terminal_utility']
        
        # Train strategy
        print(f"Training strategy for Setting {i+1}...")
        strategy.train(train_data, None, EPOCHS // 2, BATCH_SIZE, LEARNING_RATE)
        
        strategies.append(strategy)
    
    # Generate test data
    print("Generating portfolio test data...")
    test_data = []
    
    for i in tqdm(range(100)):  # Fewer paths for computational efficiency
        # Simulate market data
        market_data = simulate_portfolio_market_data(
            option_types, S0, Ks, Ts, r, sigma, dt, hawkes_params, kappa
        )
        
        # Store data
        test_data.append(market_data)
    
    # Test strategies
    print("Testing portfolio strategies...")
    results = []
    
    for i, strategy in enumerate(strategies):
        print(f"Testing strategy for Setting {i+1}...")
        
        strategy_results = []
        for j in tqdm(range(len(test_data))):
            # Simulate market making
            result = simulate_portfolio_market_making(strategy, test_data[j])
            
            # Store results
            strategy_results.append(result)
        
        # Calculate statistics
        profits = [result['profit_loss'] for result in strategy_results]
        utilities = [result['terminal_utility'] for result in strategy_results]
        gamma_violations = [result['gamma_violation'] for result in strategy_results]
        vega_violations = [result['vega_violation'] for result in strategy_results]
        inventory_violations = [result['inventory_violation'] for result in strategy_results]
        
        results.append({
            'setting': i + 1,
            'profits': profits,
            'utilities': utilities,
            'gamma_violations': gamma_violations,
            'vega_violations': vega_violations,
            'inventory_violations': inventory_violations,
            'mean_profit': np.mean(profits),
            'std_profit': np.std(profits),
            'mean_utility': np.mean(utilities),
            'gamma_violation_rate': np.mean(gamma_violations),
            'vega_violation_rate': np.mean(vega_violations),
            'inventory_violation_rate': np.mean(inventory_violations)
        })
    
    # Print results
    print("\nPortfolio Results:")
    print("-" * 100)
    print(f"{'Setting':<10} {'Mean Profit':<15} {'Std Profit':<15} {'Gamma Violation':<15} {'Vega Violation':<15} {'Inventory Violation':<15}")
    print("-" * 100)
    
    for result in results:
        print(f"{result['setting']:<10} {result['mean_profit']:<15.4f} {result['std_profit']:<15.4f} {result['gamma_violation_rate']:<15.4f} {result['vega_violation_rate']:<15.4f} {result['inventory_violation_rate']:<15.4f}")
    
    # Plot inventory for different settings
    plt.figure(figsize=(15, 10))
    
    # Use the first test path
    test_path = test_data[0]
    
    # Plot in subplots
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot option 0 inventory
    for i, strategy in enumerate(strategies):
        result = simulate_portfolio_market_making(strategy, test_path)
        axs[0, 0].plot(result['time_grid'], result['option_inventory'][0], label=f'Setting {i+1}')
    
    axs[0, 0].set_title('Option 0 Inventory')
    axs[0, 0].set_xlabel('Time (days)')
    axs[0, 0].set_ylabel('Inventory')
    axs[0, 0].legend()
    axs[0, 0].grid(True, alpha=0.3)
    
    # Plot option 1 inventory
    for i, strategy in enumerate(strategies):
        result = simulate_portfolio_market_making(strategy, test_path)
        axs[0, 1].plot(result['time_grid'], result['option_inventory'][1], label=f'Setting {i+1}')
    
    axs[0, 1].set_title('Option 1 Inventory')
    axs[0, 1].set_xlabel('Time (days)')
    axs[0, 1].set_ylabel('Inventory')
    axs[0, 1].legend()
    axs[0, 1].grid(True, alpha=0.3)
    
    # Plot gamma
    for i, strategy in enumerate(strategies):
        result = simulate_portfolio_market_making(strategy, test_path)
        axs[1, 0].plot(result['time_grid'], result['gamma'], label=f'Setting {i+1}')
    
    axs[1, 0].set_title('Portfolio Gamma')
    axs[1, 0].set_xlabel('Time (days)')
    axs[1, 0].set_ylabel('Gamma')
    axs[1, 0].legend()
    axs[1, 0].grid(True, alpha=0.3)
    
    # Plot vega
    for i, strategy in enumerate(strategies):
        result = simulate_portfolio_market_making(strategy, test_path)
        axs[1, 1].plot(result['time_grid'], result['vega'], label=f'Setting {i+1}')
    
    axs[1, 1].set_title('Portfolio Vega')
    axs[1, 1].set_xlabel('Time (days)')
    axs[1, 1].set_ylabel('Vega')
    axs[1, 1].legend()
    axs[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('portfolio_constraints.png')
    plt.close()
    
    # Plot bid/ask quotes for the most constrained setting (Setting 3)
    plt.figure(figsize=(15, 10))
    
    # Get results for Setting 3
    result = simulate_portfolio_market_making(strategies[2], test_path)
    
    # Plot in subplots
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot option 0 ask quotes
    axs[0, 0].plot(test_path['time_grid'][:-1], result['ask_quotes'][0])
    axs[0, 0].set_title('Option 0 Ask Quotes (Setting 3)')
    axs[0, 0].set_xlabel('Time (days)')
    axs[0, 0].set_ylabel('Distance from Mid-Price')
    axs[0, 0].grid(True, alpha=0.3)
    
    # Plot option 0 bid quotes
    axs[0, 1].plot(test_path['time_grid'][:-1], result['bid_quotes'][0])
    axs[0, 1].set_title('Option 0 Bid Quotes (Setting 3)')
    axs[0, 1].set_xlabel('Time (days)')
    axs[0, 1].set_ylabel('Distance from Mid-Price')
    axs[0, 1].grid(True, alpha=0.3)
    
    # Plot option 1 ask quotes
    axs[1, 0].plot(test_path['time_grid'][:-1], result['ask_quotes'][1])
    axs[1, 0].set_title('Option 1 Ask Quotes (Setting 3)')
    axs[1, 0].set_xlabel('Time (days)')
    axs[1, 0].set_ylabel('Distance from Mid-Price')
    axs[1, 0].grid(True, alpha=0.3)
    
    # Plot option 1 bid quotes
    axs[1, 1].plot(test_path['time_grid'][:-1], result['bid_quotes'][1])
    axs[1, 1].set_title('Option 1 Bid Quotes (Setting 3)')
    axs[1, 1].set_xlabel('Time (days)')
    axs[1, 1].set_ylabel('Distance from Mid-Price')
    axs[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('setting3_quotes.png')
    plt.close()
    
    return results, strategies, test_data

# Run portfolio experiment
portfolio_results, portfolio_strategies, portfolio_test_data = run_portfolio_experiment()

print("Experiments completed!")