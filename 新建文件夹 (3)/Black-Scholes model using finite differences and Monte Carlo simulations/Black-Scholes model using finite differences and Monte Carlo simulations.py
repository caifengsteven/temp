import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from tqdm import tqdm
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

class BlackScholesModel:
    """
    Class for Black-Scholes model option pricing and hedging.
    """
    
    def __init__(self, S0, K, r, sigma, T):
        """
        Initialize the Black-Scholes model parameters.
        
        Parameters:
        S0 (float): Initial stock price
        K (float): Strike price
        r (float): Risk-free interest rate
        sigma (float): Volatility
        T (float): Time to maturity
        """
        self.S0 = S0
        self.K = K
        self.r = r
        self.sigma = sigma
        self.T = T
    
    def d1(self, S, t):
        """Calculate d1 in the Black-Scholes formula."""
        return (np.log(S / self.K) + (self.r + 0.5 * self.sigma**2) * (self.T - t)) / (self.sigma * np.sqrt(self.T - t))
    
    def d2(self, S, t):
        """Calculate d2 in the Black-Scholes formula."""
        return self.d1(S, t) - self.sigma * np.sqrt(self.T - t)
    
    def call_price(self, S, t):
        """Calculate the price of a European call option."""
        if self.T == t:
            return max(0, S - self.K)
        d1 = self.d1(S, t)
        d2 = self.d2(S, t)
        return S * norm.cdf(d1) - self.K * np.exp(-self.r * (self.T - t)) * norm.cdf(d2)
    
    def put_price(self, S, t):
        """Calculate the price of a European put option."""
        if self.T == t:
            return max(0, self.K - S)
        d1 = self.d1(S, t)
        d2 = self.d2(S, t)
        return self.K * np.exp(-self.r * (self.T - t)) * norm.cdf(-d2) - S * norm.cdf(-d1)
    
    def delta(self, S, t, option_type='call'):
        """Calculate the delta of an option."""
        if self.T == t:
            if option_type == 'call':
                return 1.0 if S > self.K else 0.0
            else:
                return -1.0 if S < self.K else 0.0
        
        d1 = self.d1(S, t)
        if option_type == 'call':
            return norm.cdf(d1)
        else:
            return norm.cdf(d1) - 1
    
    def gamma(self, S, t):
        """Calculate the gamma of an option."""
        if self.T == t:
            return 0.0
        
        d1 = self.d1(S, t)
        return norm.pdf(d1) / (S * self.sigma * np.sqrt(self.T - t))
    
    def theta(self, S, t, option_type='call'):
        """Calculate the theta of an option."""
        if self.T == t:
            return 0.0
        
        d1 = self.d1(S, t)
        d2 = self.d2(S, t)
        
        if option_type == 'call':
            return -S * norm.pdf(d1) * self.sigma / (2 * np.sqrt(self.T - t)) - self.r * self.K * np.exp(-self.r * (self.T - t)) * norm.cdf(d2)
        else:
            return -S * norm.pdf(d1) * self.sigma / (2 * np.sqrt(self.T - t)) + self.r * self.K * np.exp(-self.r * (self.T - t)) * norm.cdf(-d2)
    
    def vega(self, S, t):
        """Calculate the vega of an option."""
        if self.T == t:
            return 0.0
        
        d1 = self.d1(S, t)
        return S * np.sqrt(self.T - t) * norm.pdf(d1)
    
    def rho(self, S, t, option_type='call'):
        """Calculate the rho of an option."""
        if self.T == t:
            return 0.0
        
        d2 = self.d2(S, t)
        
        if option_type == 'call':
            return self.K * (self.T - t) * np.exp(-self.r * (self.T - t)) * norm.cdf(d2)
        else:
            return -self.K * (self.T - t) * np.exp(-self.r * (self.T - t)) * norm.cdf(-d2)

class FiniteDifferencePricer:
    """
    Class for pricing options using finite difference methods.
    """
    
    def __init__(self, S0, K, r, sigma, T, Smax=None, M=100, N=100, method='explicit'):
        """
        Initialize the finite difference pricer.
        
        Parameters:
        S0 (float): Initial stock price
        K (float): Strike price
        r (float): Risk-free interest rate
        sigma (float): Volatility
        T (float): Time to maturity
        Smax (float): Maximum stock price for grid
        M (int): Number of stock price steps
        N (int): Number of time steps
        method (str): Finite difference method ('explicit', 'implicit', 'crank-nicolson')
        """
        self.S0 = S0
        self.K = K
        self.r = r
        self.sigma = sigma
        self.T = T
        self.Smax = Smax if Smax else 2 * K
        self.M = M
        self.N = N
        self.method = method
        
        # Grid setup
        self.dt = self.T / self.N
        self.ds = self.Smax / self.M
        self.grid = np.zeros((self.M+1, self.N+1))
        
        # Precompute coefficients for explicit method
        j_values = np.arange(self.M+1)
        self.S_values = j_values * self.ds
        
        self.alpha = 0.5 * self.dt * ((self.sigma * j_values)**2 - self.r * j_values)
        self.beta = 1 - self.dt * ((self.sigma * j_values)**2 + self.r)
        self.gamma = 0.5 * self.dt * ((self.sigma * j_values)**2 + self.r * j_values)
    
    def _setup_boundary_conditions(self, option_type='call'):
        """Set up boundary conditions for the grid."""
        # Terminal condition (option payoff at maturity)
        if option_type == 'call':
            self.grid[:, -1] = np.maximum(0, self.S_values - self.K)
        else:  # put
            self.grid[:, -1] = np.maximum(0, self.K - self.S_values)
        
        # Boundary conditions
        for n in range(self.N + 1):
            t = n * self.dt
            if option_type == 'call':
                self.grid[0, n] = 0  # S = 0
                self.grid[-1, n] = self.Smax - self.K * np.exp(-self.r * (self.T - t))  # S = Smax
            else:  # put
                self.grid[0, n] = self.K * np.exp(-self.r * (self.T - t))  # S = 0
                self.grid[-1, n] = 0  # S = Smax
    
    def price_option(self, option_type='call'):
        """
        Price an option using finite difference method.
        
        Parameters:
        option_type (str): Type of option ('call' or 'put')
        
        Returns:
        float: Option price at S0
        """
        self._setup_boundary_conditions(option_type)
        
        if self.method == 'explicit':
            # Solve using explicit finite difference
            for n in range(self.N-1, -1, -1):
                for j in range(1, self.M):
                    self.grid[j, n] = (
                        self.alpha[j] * self.grid[j-1, n+1] + 
                        self.beta[j] * self.grid[j, n+1] + 
                        self.gamma[j] * self.grid[j+1, n+1]
                    )
        
        elif self.method == 'implicit':
            # Solve using implicit finite difference (would require solving a system of equations)
            # This is more complex and would require a tridiagonal solver
            pass
        
        elif self.method == 'crank-nicolson':
            # Solve using Crank-Nicolson method (combination of explicit and implicit)
            # This is more complex and would require a tridiagonal solver
            pass
        
        # Interpolate to get price at S0
        j = int(self.S0 / self.ds)
        if j == self.S0 / self.ds:  # S0 is exactly on a grid point
            return self.grid[j, 0]
        else:  # Linear interpolation
            w = (self.S0 - j * self.ds) / self.ds
            return (1 - w) * self.grid[j, 0] + w * self.grid[j+1, 0]
    
    def calculate_delta(self, option_type='call'):
        """
        Calculate delta using finite difference.
        
        Returns:
        float: Delta at S0
        """
        self.price_option(option_type)
        
        j = int(self.S0 / self.ds)
        if j < 1 or j >= self.M:
            return 0  # Outside valid range
        
        # Central difference
        return (self.grid[j+1, 0] - self.grid[j-1, 0]) / (2 * self.ds)
    
    def calculate_gamma(self, option_type='call'):
        """
        Calculate gamma using finite difference.
        
        Returns:
        float: Gamma at S0
        """
        self.price_option(option_type)
        
        j = int(self.S0 / self.ds)
        if j < 1 or j >= self.M:
            return 0  # Outside valid range
        
        # Second derivative approximation
        return (self.grid[j+1, 0] - 2*self.grid[j, 0] + self.grid[j-1, 0]) / (self.ds**2)

class MonteCarloSimulation:
    """
    Class for Monte Carlo simulation of option pricing and hedging.
    """
    
    def __init__(self, S0, K, r, sigma, T):
        """
        Initialize Monte Carlo simulation parameters.
        
        Parameters:
        S0 (float): Initial stock price
        K (float): Strike price
        r (float): Risk-free interest rate
        sigma (float): Volatility
        T (float): Time to maturity
        """
        self.S0 = S0
        self.K = K
        self.r = r
        self.sigma = sigma
        self.T = T
        self.bs_model = BlackScholesModel(S0, K, r, sigma, T)
    
    def simulate_paths(self, n_paths, n_steps, seed=None):
        """
        Simulate stock price paths.
        
        Parameters:
        n_paths (int): Number of paths to simulate
        n_steps (int): Number of time steps
        seed (int): Random seed for reproducibility
        
        Returns:
        numpy.ndarray: Array of simulated stock prices [n_paths, n_steps+1]
        """
        if seed is not None:
            np.random.seed(seed)
        
        dt = self.T / n_steps
        paths = np.zeros((n_paths, n_steps + 1))
        paths[:, 0] = self.S0
        
        # Generate random shocks
        random_shocks = np.random.normal(0, 1, (n_paths, n_steps))
        
        # Simulate paths
        for t in range(1, n_steps + 1):
            paths[:, t] = paths[:, t-1] * np.exp((self.r - 0.5 * self.sigma**2) * dt + 
                                              self.sigma * np.sqrt(dt) * random_shocks[:, t-1])
        
        return paths
    
    def price_option_mc(self, n_paths, n_steps, option_type='call', antithetic=False, control_variate=False):
        """
        Price an option using Monte Carlo simulation.
        
        Parameters:
        n_paths (int): Number of paths to simulate
        n_steps (int): Number of time steps
        option_type (str): Type of option ('call' or 'put')
        antithetic (bool): Whether to use antithetic variates for variance reduction
        control_variate (bool): Whether to use control variates for variance reduction
        
        Returns:
        tuple: (Option price, Standard error)
        """
        if antithetic:
            # Use antithetic variates for variance reduction
            n_paths = n_paths // 2  # Ensure even number of paths
            
            # Simulate first set of paths
            paths1 = self.simulate_paths(n_paths, n_steps)
            
            # Set the seed to get different random numbers
            paths2 = np.zeros_like(paths1)
            paths2[:, 0] = self.S0
            
            # Generate antithetic paths using negative of the random shocks
            dt = self.T / n_steps
            for t in range(1, n_steps + 1):
                # Calculate the random shock from the first path
                Z = (np.log(paths1[:, t] / paths1[:, t-1]) - (self.r - 0.5 * self.sigma**2) * dt) / (self.sigma * np.sqrt(dt))
                # Use negative of the shock for the antithetic path
                paths2[:, t] = paths2[:, t-1] * np.exp((self.r - 0.5 * self.sigma**2) * dt - 
                                                     self.sigma * np.sqrt(dt) * Z)
            
            # Combine paths
            paths = np.vstack((paths1, paths2))
        else:
            # Standard Monte Carlo
            paths = self.simulate_paths(n_paths, n_steps)
        
        # Calculate option payoffs at maturity
        if option_type == 'call':
            payoffs = np.maximum(0, paths[:, -1] - self.K)
        else:  # put
            payoffs = np.maximum(0, self.K - paths[:, -1])
        
        # Discount payoffs to present value
        discounted_payoffs = np.exp(-self.r * self.T) * payoffs
        
        if control_variate:
            # Use stock price as control variate
            final_stock_prices = paths[:, -1]
            expected_stock_price = self.S0 * np.exp(self.r * self.T)
            
            # Calculate optimal control variate coefficient
            cov_matrix = np.cov(np.stack([discounted_payoffs, final_stock_prices]), rowvar=True)
            beta = cov_matrix[0, 1] / cov_matrix[1, 1]
            
            # Apply control variate adjustment
            cv_adjusted_payoffs = discounted_payoffs - beta * (final_stock_prices - expected_stock_price)
            
            # Calculate price and standard error
            price = np.mean(cv_adjusted_payoffs)
            se = np.std(cv_adjusted_payoffs) / np.sqrt(len(cv_adjusted_payoffs))
        else:
            # Calculate price and standard error
            price = np.mean(discounted_payoffs)
            se = np.std(discounted_payoffs) / np.sqrt(len(discounted_payoffs))
        
        return price, se
    
    def importance_sampling_price(self, n_paths, n_steps, option_type='call', drift_shift=None):
        """
        Price an option using Monte Carlo simulation with importance sampling.
        
        Parameters:
        n_paths (int): Number of paths to simulate
        n_steps (int): Number of time steps
        option_type (str): Type of option ('call' or 'put')
        drift_shift (float): Amount to shift the drift for importance sampling
        
        Returns:
        tuple: (Option price, Standard error)
        """
        # Set default drift shift if not provided
        if drift_shift is None:
            # For a call option, shift the drift up to increase sampling in the money region
            # For a put option, shift the drift down
            if option_type == 'call':
                drift_shift = 0.1  # Positive shift for call options
            else:
                drift_shift = -0.1  # Negative shift for put options
        
        dt = self.T / n_steps
        paths = np.zeros((n_paths, n_steps + 1))
        paths[:, 0] = self.S0
        
        # Generate random shocks
        random_shocks = np.random.normal(0, 1, (n_paths, n_steps))
        
        # Simulate paths with shifted drift
        shifted_drift = self.r + drift_shift
        for t in range(1, n_steps + 1):
            paths[:, t] = paths[:, t-1] * np.exp((shifted_drift - 0.5 * self.sigma**2) * dt + 
                                              self.sigma * np.sqrt(dt) * random_shocks[:, t-1])
        
        # Calculate option payoffs at maturity
        if option_type == 'call':
            payoffs = np.maximum(0, paths[:, -1] - self.K)
        else:  # put
            payoffs = np.maximum(0, self.K - paths[:, -1])
        
        # Calculate likelihood ratio for importance sampling
        likelihood_ratio = np.exp(
            -drift_shift * self.T * random_shocks.sum(axis=1) * np.sqrt(dt) - 
            0.5 * (drift_shift**2) * self.T
        )
        
        # Apply likelihood ratio to payoffs
        adjusted_payoffs = np.exp(-self.r * self.T) * payoffs * likelihood_ratio
        
        # Calculate price and standard error
        price = np.mean(adjusted_payoffs)
        se = np.std(adjusted_payoffs) / np.sqrt(n_paths)
        
        return price, se
    
    def calculate_optimal_drift(self, n_paths, n_steps, option_type='call', drift_range=None, n_drifts=20):
        """
        Find the optimal drift shift for importance sampling.
        
        Parameters:
        n_paths (int): Number of paths to simulate for each drift
        n_steps (int): Number of time steps
        option_type (str): Type of option ('call' or 'put')
        drift_range (tuple): Range of drift shifts to test (min, max)
        n_drifts (int): Number of different drift shifts to test
        
        Returns:
        tuple: (Optimal drift shift, Variance at optimal drift)
        """
        if drift_range is None:
            if option_type == 'call':
                drift_range = (0, 0.5)
            else:
                drift_range = (-0.5, 0)
        
        drift_shifts = np.linspace(drift_range[0], drift_range[1], n_drifts)
        variances = []
        
        for drift in drift_shifts:
            _, se = self.importance_sampling_price(n_paths, n_steps, option_type, drift)
            variances.append(se**2 * n_paths)
        
        optimal_idx = np.argmin(variances)
        optimal_drift = drift_shifts[optimal_idx]
        min_variance = variances[optimal_idx]
        
        return optimal_drift, min_variance, drift_shifts, variances
    
    def delta_hedging_simulation(self, n_paths, n_steps, hedging_steps, option_type='call'):
        """
        Simulate delta hedging and calculate hedging errors.
        
        Parameters:
        n_paths (int): Number of paths to simulate
        n_steps (int): Number of time steps for price simulation
        hedging_steps (int): Number of rebalancing steps for the hedge
        option_type (str): Type of option ('call' or 'put')
        
        Returns:
        tuple: (Hedging errors, Stock paths)
        """
        dt = self.T / hedging_steps
        paths = self.simulate_paths(n_paths, n_steps)
        
        # Interpolate paths to match hedging steps
        hedge_times = np.linspace(0, self.T, hedging_steps + 1)
        sim_times = np.linspace(0, self.T, n_steps + 1)
        
        hedge_paths = np.zeros((n_paths, hedging_steps + 1))
        hedge_paths[:, 0] = self.S0
        
        for i in range(n_paths):
            hedge_paths[i, :] = np.interp(hedge_times, sim_times, paths[i, :])
        
        # Initialize portfolio and hedging errors
        portfolio_values = np.zeros((n_paths, hedging_steps + 1))
        option_values = np.zeros((n_paths, hedging_steps + 1))
        deltas = np.zeros((n_paths, hedging_steps))
        hedge_errors = np.zeros(n_paths)
        
        # Calculate initial option values and deltas
        for i in range(n_paths):
            option_values[i, 0] = self.bs_model.call_price(hedge_paths[i, 0], 0) if option_type == 'call' else self.bs_model.put_price(hedge_paths[i, 0], 0)
            deltas[i, 0] = self.bs_model.delta(hedge_paths[i, 0], 0, option_type)
        
        # Initialize portfolio with option and delta hedge
        portfolio_values[:, 0] = option_values[:, 0]
        cash = portfolio_values[:, 0] - deltas[:, 0] * hedge_paths[:, 0]
        
        # Simulate hedging
        for t in range(1, hedging_steps + 1):
            current_time = t * dt
            
            # Update option values
            for i in range(n_paths):
                option_values[i, t] = self.bs_model.call_price(hedge_paths[i, t], current_time) if option_type == 'call' else self.bs_model.put_price(hedge_paths[i, t], current_time)
            
            # Update portfolio values (including interest on cash)
            portfolio_values[:, t] = cash * np.exp(self.r * dt) + deltas[:, t-1] * hedge_paths[:, t]
            
            if t < hedging_steps:
                # Calculate new deltas
                for i in range(n_paths):
                    deltas[i, t] = self.bs_model.delta(hedge_paths[i, t], current_time, option_type)
                
                # Rebalance portfolio
                cash = portfolio_values[:, t] - deltas[:, t] * hedge_paths[:, t]
        
        # Calculate final hedging errors
        hedge_errors = portfolio_values[:, -1] - option_values[:, -1]
        
        return hedge_errors, hedge_paths

def analyze_hedge_errors(bs_model, n_paths=1000, n_steps=100, hedging_steps_list=[5, 10, 20, 50, 100]):
    """
    Analyze hedging errors for different hedging frequencies.
    
    Parameters:
    bs_model (BlackScholesModel): Black-Scholes model object
    n_paths (int): Number of paths to simulate
    n_steps (int): Number of time steps for price simulation
    hedging_steps_list (list): List of different hedging steps to test
    
    Returns:
    tuple: (Mean squared errors, Standard deviations of errors)
    """
    mc_simulation = MonteCarloSimulation(bs_model.S0, bs_model.K, bs_model.r, bs_model.sigma, bs_model.T)
    
    mse_list = []
    std_list = []
    
    for hedging_steps in hedging_steps_list:
        hedge_errors, _ = mc_simulation.delta_hedging_simulation(n_paths, n_steps, hedging_steps)
        
        mse = np.mean(hedge_errors**2)
        std = np.std(hedge_errors)
        
        mse_list.append(mse)
        std_list.append(std)
    
    return mse_list, std_list

def compare_hedging_methods(bs_model, n_paths=1000, n_steps=100, hedging_steps=20):
    """
    Compare different hedging methods.
    
    Parameters:
    bs_model (BlackScholesModel): Black-Scholes model object
    n_paths (int): Number of paths to simulate
    n_steps (int): Number of time steps for price simulation
    hedging_steps (int): Number of rebalancing steps for the hedge
    
    Returns:
    dict: Dictionary with hedging errors for different methods
    """
    mc_simulation = MonteCarloSimulation(bs_model.S0, bs_model.K, bs_model.r, bs_model.sigma, bs_model.T)
    
    # Standard delta hedging
    delta_errors, paths = mc_simulation.delta_hedging_simulation(n_paths, n_steps, hedging_steps)
    
    # Delta-gamma hedging (simplified implementation)
    delta_gamma_errors = np.zeros(n_paths)
    
    dt = bs_model.T / hedging_steps
    hedge_times = np.linspace(0, bs_model.T, hedging_steps + 1)
    
    for i in range(n_paths):
        # Initialize portfolio
        stock_position = bs_model.delta(paths[i, 0], 0)
        gamma = bs_model.gamma(paths[i, 0], 0)
        option_value = bs_model.call_price(paths[i, 0], 0)
        
        # Use a second option for gamma hedging (simplified)
        second_bs = BlackScholesModel(bs_model.S0, bs_model.K * 1.1, bs_model.r, bs_model.sigma, bs_model.T)
        second_delta = second_bs.delta(paths[i, 0], 0)
        second_gamma = second_bs.gamma(paths[i, 0], 0)
        second_option = second_bs.call_price(paths[i, 0], 0)
        
        # Calculate weights for delta-gamma hedge
        if abs(second_gamma - gamma) > 1e-10:
            weight_second = gamma / (second_gamma - gamma)
            weight_stock = stock_position - weight_second * second_delta
        else:
            weight_second = 0
            weight_stock = stock_position
        
        cash = option_value - weight_stock * paths[i, 0] - weight_second * second_option
        
        # Track portfolio value
        for t in range(1, hedging_steps + 1):
            current_time = hedge_times[t]
            
            # Update portfolio value
            portfolio_value = cash * np.exp(bs_model.r * dt) + weight_stock * paths[i, t]
            
            if t < hedging_steps:
                # Recalculate delta and gamma
                stock_position = bs_model.delta(paths[i, t], current_time)
                gamma = bs_model.gamma(paths[i, t], current_time)
                
                # Update second option values
                second_delta = second_bs.delta(paths[i, t], current_time)
                second_gamma = second_bs.gamma(paths[i, t], current_time)
                second_option = second_bs.call_price(paths[i, t], current_time)
                
                # Recalculate weights
                if abs(second_gamma - gamma) > 1e-10:
                    weight_second = gamma / (second_gamma - gamma)
                    weight_stock = stock_position - weight_second * second_delta
                else:
                    weight_second = 0
                    weight_stock = stock_position
                
                cash = portfolio_value - weight_stock * paths[i, t] - weight_second * second_option
        
        # Calculate final hedging error
        final_option_value = bs_model.call_price(paths[i, -1], bs_model.T)
        delta_gamma_errors[i] = portfolio_value - final_option_value
    
    return {
        'delta': delta_errors,
        'delta_gamma': delta_gamma_errors
    }

def compare_pricing_methods(bs_model, fd_methods=None):
    """
    Compare different option pricing methods.
    
    Parameters:
    bs_model (BlackScholesModel): Black-Scholes model object
    fd_methods (list): List of finite difference methods to test
    
    Returns:
    pd.DataFrame: DataFrame with pricing results
    """
    if fd_methods is None:
        fd_methods = ['explicit']
    
    # Analytical price
    bs_price = bs_model.call_price(bs_model.S0, 0)
    
    results = {'Analytical': bs_price}
    
    # Finite difference prices
    for method in fd_methods:
        fd_pricer = FiniteDifferencePricer(bs_model.S0, bs_model.K, bs_model.r, bs_model.sigma, bs_model.T, method=method)
        fd_price = fd_pricer.price_option()
        results[f'FD ({method})'] = fd_price
    
    # Monte Carlo prices
    mc_simulation = MonteCarloSimulation(bs_model.S0, bs_model.K, bs_model.r, bs_model.sigma, bs_model.T)
    
    # Standard Monte Carlo
    mc_price, mc_se = mc_simulation.price_option_mc(10000, 100)
    results['Monte Carlo'] = mc_price
    results['MC Std Error'] = mc_se
    
    # Monte Carlo with variance reduction
    mc_antithetic, mc_anti_se = mc_simulation.price_option_mc(10000, 100, antithetic=True)
    results['MC (Antithetic)'] = mc_antithetic
    results['MC Anti Std Error'] = mc_anti_se
    
    mc_cv, mc_cv_se = mc_simulation.price_option_mc(10000, 100, control_variate=True)
    results['MC (Control Variate)'] = mc_cv
    results['MC CV Std Error'] = mc_cv_se
    
    # Importance sampling
    optimal_drift, _, _, _ = mc_simulation.calculate_optimal_drift(1000, 100)
    mc_is, mc_is_se = mc_simulation.importance_sampling_price(10000, 100, drift_shift=optimal_drift)
    results['MC (Importance Sampling)'] = mc_is
    results['MC IS Std Error'] = mc_is_se
    
    return pd.DataFrame(results, index=['Price']).T

def plot_hedge_errors_vs_frequency(bs_model):
    """
    Plot hedging errors versus hedging frequency.
    
    Parameters:
    bs_model (BlackScholesModel): Black-Scholes model object
    """
    hedging_steps_list = [2, 5, 10, 20, 50, 100]
    mse_list, std_list = analyze_hedge_errors(bs_model, hedging_steps_list=hedging_steps_list)
    
    # Calculate theoretical O(dt) relationship
    dt_values = bs_model.T / np.array(hedging_steps_list)
    theoretical = 0.01 * dt_values  # Scaling factor chosen for visualization
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(hedging_steps_list, mse_list, 'o-', label='MSE')
    plt.plot(hedging_steps_list, theoretical, '--', label='O(dt)')
    plt.xlabel('Number of Hedging Steps')
    plt.ylabel('Mean Squared Error')
    plt.title('Hedging Error vs. Frequency')
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True)
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(hedging_steps_list, std_list, 'o-', label='Std Dev')
    plt.plot(hedging_steps_list, np.sqrt(theoretical), '--', label='O(sqrt(dt))')
    plt.xlabel('Number of Hedging Steps')
    plt.ylabel('Standard Deviation of Error')
    plt.title('Std Dev of Hedging Error vs. Frequency')
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def plot_hedging_methods_comparison(bs_model):
    """
    Plot comparison of different hedging methods.
    
    Parameters:
    bs_model (BlackScholesModel): Black-Scholes model object
    """
    hedging_results = compare_hedging_methods(bs_model)
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    for method, errors in hedging_results.items():
        sns.histplot(errors, bins=30, kde=True, label=f'{method.capitalize()} Hedging')
    plt.xlabel('Hedging Error')
    plt.ylabel('Frequency')
    plt.title('Distribution of Hedging Errors')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    error_stats = {
        method: {'Mean': np.mean(errors), 'Std Dev': np.std(errors), 'MSE': np.mean(errors**2)}
        for method, errors in hedging_results.items()
    }
    
    error_df = pd.DataFrame(error_stats).T
    error_df.plot(kind='bar', ax=plt.gca())
    plt.ylabel('Value')
    plt.title('Hedging Error Statistics')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

def plot_monte_carlo_convergence(bs_model):
    """
    Plot convergence of Monte Carlo methods.
    
    Parameters:
    bs_model (BlackScholesModel): Black-Scholes model object
    """
    mc_simulation = MonteCarloSimulation(bs_model.S0, bs_model.K, bs_model.r, bs_model.sigma, bs_model.T)
    
    # Define number of paths to test
    n_paths_list = [100, 500, 1000, 5000, 10000, 50000]
    
    # Initialize arrays for results
    mc_prices = []
    mc_se = []
    mc_anti_prices = []
    mc_anti_se = []
    mc_cv_prices = []
    mc_cv_se = []
    mc_is_prices = []
    mc_is_se = []
    
    # Analytical price
    bs_price = bs_model.call_price(bs_model.S0, 0)
    
    # Calculate optimal drift for importance sampling
    optimal_drift, _, _, _ = mc_simulation.calculate_optimal_drift(1000, 100)
    
    # Calculate prices and standard errors for each number of paths
    for n_paths in n_paths_list:
        # Standard Monte Carlo
        price, se = mc_simulation.price_option_mc(n_paths, 100)
        mc_prices.append(price)
        mc_se.append(se)
        
        # Antithetic variates
        price, se = mc_simulation.price_option_mc(n_paths, 100, antithetic=True)
        mc_anti_prices.append(price)
        mc_anti_se.append(se)
        
        # Control variates
        price, se = mc_simulation.price_option_mc(n_paths, 100, control_variate=True)
        mc_cv_prices.append(price)
        mc_cv_se.append(se)
        
        # Importance sampling
        price, se = mc_simulation.importance_sampling_price(n_paths, 100, drift_shift=optimal_drift)
        mc_is_prices.append(price)
        mc_is_se.append(se)
    
    plt.figure(figsize=(12, 10))
    
    # Plot prices
    plt.subplot(2, 2, 1)
    plt.semilogx(n_paths_list, mc_prices, 'o-', label='Standard MC')
    plt.semilogx(n_paths_list, mc_anti_prices, 's-', label='Antithetic Variates')
    plt.semilogx(n_paths_list, mc_cv_prices, '^-', label='Control Variates')
    plt.semilogx(n_paths_list, mc_is_prices, 'd-', label='Importance Sampling')
    plt.axhline(bs_price, color='r', linestyle='--', label='Analytical Price')
    plt.xlabel('Number of Paths')
    plt.ylabel('Option Price')
    plt.title('Monte Carlo Price Convergence')
    plt.grid(True)
    plt.legend()
    
    # Plot standard errors
    plt.subplot(2, 2, 2)
    plt.loglog(n_paths_list, mc_se, 'o-', label='Standard MC')
    plt.loglog(n_paths_list, mc_anti_se, 's-', label='Antithetic Variates')
    plt.loglog(n_paths_list, mc_cv_se, '^-', label='Control Variates')
    plt.loglog(n_paths_list, mc_is_se, 'd-', label='Importance Sampling')
    # Add reference line for 1/sqrt(n) convergence
    reference = mc_se[0] * np.sqrt(n_paths_list[0] / np.array(n_paths_list))
    plt.loglog(n_paths_list, reference, 'k--', label='1/sqrt(n)')
    plt.xlabel('Number of Paths')
    plt.ylabel('Standard Error')
    plt.title('Monte Carlo Standard Error Convergence')
    plt.grid(True)
    plt.legend()
    
    # Plot variance reduction ratios
    plt.subplot(2, 2, 3)
    variance_ratio_anti = np.array(mc_se)**2 / np.array(mc_anti_se)**2
    variance_ratio_cv = np.array(mc_se)**2 / np.array(mc_cv_se)**2
    variance_ratio_is = np.array(mc_se)**2 / np.array(mc_is_se)**2
    
    plt.semilogx(n_paths_list, variance_ratio_anti, 's-', label='Antithetic Variates')
    plt.semilogx(n_paths_list, variance_ratio_cv, '^-', label='Control Variates')
    plt.semilogx(n_paths_list, variance_ratio_is, 'd-', label='Importance Sampling')
    plt.axhline(1, color='k', linestyle='--')
    plt.xlabel('Number of Paths')
    plt.ylabel('Variance Reduction Ratio')
    plt.title('Variance Reduction Effectiveness')
    plt.grid(True)
    plt.legend()
    
    # Plot confidence intervals
    plt.subplot(2, 2, 4)
    ci_sizes = [
        2 * 1.96 * np.array(mc_se),
        2 * 1.96 * np.array(mc_anti_se),
        2 * 1.96 * np.array(mc_cv_se),
        2 * 1.96 * np.array(mc_is_se)
    ]
    
    plt.loglog(n_paths_list, ci_sizes[0], 'o-', label='Standard MC')
    plt.loglog(n_paths_list, ci_sizes[1], 's-', label='Antithetic Variates')
    plt.loglog(n_paths_list, ci_sizes[2], '^-', label='Control Variates')
    plt.loglog(n_paths_list, ci_sizes[3], 'd-', label='Importance Sampling')
    plt.xlabel('Number of Paths')
    plt.ylabel('95% Confidence Interval Size')
    plt.title('Confidence Interval Convergence')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def plot_importance_sampling_optimization(bs_model, option_type='call'):
    """
    Plot optimization of drift parameter for importance sampling.
    
    Parameters:
    bs_model (BlackScholesModel): Black-Scholes model object
    option_type (str): Type of option ('call' or 'put')
    """
    mc_simulation = MonteCarloSimulation(bs_model.S0, bs_model.K, bs_model.r, bs_model.sigma, bs_model.T)
    
    # Define range of drift shifts to test
    if option_type == 'call':
        drift_range = (-0.2, 0.5)
    else:
        drift_range = (-0.5, 0.2)
    
    # Calculate optimal drift
    optimal_drift, min_variance, drift_shifts, variances = mc_simulation.calculate_optimal_drift(
        1000, 100, option_type, drift_range, n_drifts=30
    )
    
    plt.figure(figsize=(10, 6))
    plt.plot(drift_shifts, variances, 'o-')
    plt.axvline(optimal_drift, color='r', linestyle='--', label=f'Optimal Drift = {optimal_drift:.4f}')
    plt.xlabel('Drift Shift Parameter')
    plt.ylabel('Variance')
    plt.title(f'Optimization of Importance Sampling Drift for {option_type.capitalize()} Option')
    plt.grid(True)
    plt.legend()
    plt.show()
    
    return optimal_drift, min_variance

def simulate_and_visualize_stock_paths(bs_model, n_paths=10, n_steps=100):
    """
    Simulate and visualize stock price paths.
    
    Parameters:
    bs_model (BlackScholesModel): Black-Scholes model object
    n_paths (int): Number of paths to simulate
    n_steps (int): Number of time steps
    """
    mc_simulation = MonteCarloSimulation(bs_model.S0, bs_model.K, bs_model.r, bs_model.sigma, bs_model.T)
    paths = mc_simulation.simulate_paths(n_paths, n_steps)
    
    plt.figure(figsize=(10, 6))
    times = np.linspace(0, bs_model.T, n_steps + 1)
    
    for i in range(n_paths):
        plt.plot(times, paths[i, :])
    
    plt.axhline(bs_model.K, color='r', linestyle='--', label='Strike Price')
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.title('Simulated Stock Price Paths')
    plt.grid(True)
    plt.legend()
    plt.show()

def plot_finite_difference_vs_analytical_solution(bs_model):
    """
    Plot finite difference solution compared to analytical solution.
    
    Parameters:
    bs_model (BlackScholesModel): Black-Scholes model object
    """
    # Create a grid of stock prices
    S_values = np.linspace(0.5 * bs_model.K, 1.5 * bs_model.K, 100)
    
    # Calculate analytical solution
    analytical_prices = np.array([bs_model.call_price(S, 0) for S in S_values])
    
    # Calculate finite difference solution
    fd_pricer = FiniteDifferencePricer(bs_model.S0, bs_model.K, bs_model.r, bs_model.sigma, bs_model.T)
    fd_prices = np.array([fd_pricer._setup_boundary_conditions() or fd_pricer.price_option() for _ in S_values])
    
    # Create an array of interpolated prices at the desired stock prices
    interp_fd_prices = np.zeros_like(S_values)
    for i, S in enumerate(S_values):
        fd_pricer = FiniteDifferencePricer(S, bs_model.K, bs_model.r, bs_model.sigma, bs_model.T)
        interp_fd_prices[i] = fd_pricer.price_option()
    
    plt.figure(figsize=(12, 5))
    
    # Plot prices
    plt.subplot(1, 2, 1)
    plt.plot(S_values, analytical_prices, 'b-', label='Analytical')
    plt.plot(S_values, interp_fd_prices, 'r--', label='Finite Difference')
    plt.xlabel('Stock Price')
    plt.ylabel('Option Price')
    plt.title('Option Price: Analytical vs. Finite Difference')
    plt.grid(True)
    plt.legend()
    
    # Plot absolute difference
    plt.subplot(1, 2, 2)
    plt.plot(S_values, np.abs(analytical_prices - interp_fd_prices))
    plt.xlabel('Stock Price')
    plt.ylabel('Absolute Difference')
    plt.title('Absolute Difference Between Methods')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

def plot_delta_gamma_comparison(bs_model):
    """
    Plot delta and gamma from different methods.
    
    Parameters:
    bs_model (BlackScholesModel): Black-Scholes model object
    """
    # Create a grid of stock prices
    S_values = np.linspace(0.5 * bs_model.K, 1.5 * bs_model.K, 100)
    
    # Calculate analytical delta and gamma
    analytical_delta = np.array([bs_model.delta(S, 0) for S in S_values])
    analytical_gamma = np.array([bs_model.gamma(S, 0) for S in S_values])
    
    # Calculate finite difference delta and gamma
    fd_delta = np.zeros_like(S_values)
    fd_gamma = np.zeros_like(S_values)
    
    for i, S in enumerate(S_values):
        fd_pricer = FiniteDifferencePricer(S, bs_model.K, bs_model.r, bs_model.sigma, bs_model.T)
        fd_delta[i] = fd_pricer.calculate_delta()
        fd_gamma[i] = fd_pricer.calculate_gamma()
    
    plt.figure(figsize=(12, 10))
    
    # Plot delta
    plt.subplot(2, 2, 1)
    plt.plot(S_values, analytical_delta, 'b-', label='Analytical')
    plt.plot(S_values, fd_delta, 'r--', label='Finite Difference')
    plt.xlabel('Stock Price')
    plt.ylabel('Delta')
    plt.title('Option Delta: Analytical vs. Finite Difference')
    plt.grid(True)
    plt.legend()
    
    # Plot gamma
    plt.subplot(2, 2, 2)
    plt.plot(S_values, analytical_gamma, 'b-', label='Analytical')
    plt.plot(S_values, fd_gamma, 'r--', label='Finite Difference')
    plt.xlabel('Stock Price')
    plt.ylabel('Gamma')
    plt.title('Option Gamma: Analytical vs. Finite Difference')
    plt.grid(True)
    plt.legend()
    
    # Plot absolute delta difference
    plt.subplot(2, 2, 3)
    plt.plot(S_values, np.abs(analytical_delta - fd_delta))
    plt.xlabel('Stock Price')
    plt.ylabel('Absolute Difference')
    plt.title('Absolute Delta Difference')
    plt.grid(True)
    
    # Plot absolute gamma difference
    plt.subplot(2, 2, 4)
    plt.plot(S_values, np.abs(analytical_gamma - fd_gamma))
    plt.xlabel('Stock Price')
    plt.ylabel('Absolute Difference')
    plt.title('Absolute Gamma Difference')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

def main():
    """Main function to run experiments."""
    # Define Black-Scholes model parameters
    S0 = 100.0     # Initial stock price
    K = 100.0      # Strike price
    r = 0.05       # Risk-free rate
    sigma = 0.2    # Volatility
    T = 1.0        # Time to maturity (in years)
    
    bs_model = BlackScholesModel(S0, K, r, sigma, T)
    
    # Print model information
    print("Black-Scholes Model Parameters:")
    print(f"Initial Stock Price (S0): {S0}")
    print(f"Strike Price (K): {K}")
    print(f"Risk-free Rate (r): {r}")
    print(f"Volatility (sigma): {sigma}")
    print(f"Time to Maturity (T): {T}")
    print()
    
    # Compare pricing methods
    print("Comparing Option Pricing Methods:")
    pricing_comparison = compare_pricing_methods(bs_model)
    print(pricing_comparison)
    print()
    
    # Visualize stock price paths
    print("Simulating and visualizing stock price paths...")
    simulate_and_visualize_stock_paths(bs_model)
    
    # Compare finite difference with analytical solution
    print("Comparing finite difference solution with analytical solution...")
    plot_finite_difference_vs_analytical_solution(bs_model)
    
    # Compare delta and gamma calculations
    print("Comparing delta and gamma calculations...")
    plot_delta_gamma_comparison(bs_model)
    
    # Analyze hedge errors vs frequency
    print("Analyzing hedge errors vs hedging frequency...")
    plot_hedge_errors_vs_frequency(bs_model)
    
    # Compare hedging methods
    print("Comparing different hedging methods...")
    plot_hedging_methods_comparison(bs_model)
    
    # Monte Carlo convergence analysis
    print("Analyzing Monte Carlo convergence...")
    plot_monte_carlo_convergence(bs_model)
    
    # Importance sampling optimization
    print("Optimizing importance sampling drift parameter...")
    plot_importance_sampling_optimization(bs_model)
    
    print("All experiments completed successfully!")

if __name__ == "__main__":
    main()