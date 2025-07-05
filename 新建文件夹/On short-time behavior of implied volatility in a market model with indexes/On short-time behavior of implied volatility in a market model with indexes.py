import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import brentq
import time
from tqdm import tqdm

# Set random seed for reproducibility
np.random.seed(42)

class BlackScholes:
    @staticmethod
    def d1(S, K, r, sigma, T):
        return (np.log(S/K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    
    @staticmethod
    def d2(S, K, r, sigma, T):
        return BlackScholes.d1(S, K, r, sigma, T) - sigma * np.sqrt(T)
    
    @staticmethod
    def call_price(S, K, r, sigma, T):
        if T <= 0:
            return max(0, S - K)
        d1 = BlackScholes.d1(S, K, r, sigma, T)
        d2 = BlackScholes.d2(S, K, r, sigma, T)
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    
    @staticmethod
    def implied_volatility(price, S, K, r, T, initial_vol=0.2, precision=1e-8, max_iterations=1000):
        if T <= 0 or price <= 0:
            return np.nan
        
        # For ATM or near-ATM options, use a better initial guess
        if abs(S/K - 1.0) < 0.1:
            initial_vol = np.sqrt(2 * np.pi / T) * price / S
        
        try:
            def objective(sigma):
                return BlackScholes.call_price(S, K, r, sigma, T) - price
            
            # Use Brent's method for root finding
            implied_vol = brentq(objective, 1e-10, 10.0, rtol=precision, maxiter=max_iterations)
            return implied_vol
        except:
            # Fall back to binary search if brentq fails
            vol_low = 1e-10
            vol_high = 10.0
            
            for _ in range(max_iterations):
                vol_mid = (vol_low + vol_high) / 2
                price_mid = BlackScholes.call_price(S, K, r, vol_mid, T)
                
                if abs(price_mid - price) < precision:
                    return vol_mid
                
                if price_mid < price:
                    vol_low = vol_mid
                else:
                    vol_high = vol_mid
                    
            return vol_mid
    
    @staticmethod
    def compute_atm_skew(S, r, sigma, T, h=0.001):
        """
        Compute ATM skew using finite difference approximation
        """
        k_atm = S
        k_up = S * (1 + h)
        k_down = S * (1 - h)
        
        # Calculate option prices
        price_atm = BlackScholes.call_price(S, k_atm, r, sigma, T)
        price_up = BlackScholes.call_price(S, k_up, r, sigma, T)
        price_down = BlackScholes.call_price(S, k_down, r, sigma, T)
        
        # Calculate implied volatilities
        iv_up = BlackScholes.implied_volatility(price_up, S, k_up, r, T)
        iv_atm = BlackScholes.implied_volatility(price_atm, S, k_atm, r, T)
        iv_down = BlackScholes.implied_volatility(price_down, S, k_down, r, T)
        
        # Central difference approximation of the first derivative
        skew = (iv_up - iv_down) / (2 * h * S)
        
        return skew

class MarketSimulator:
    def __init__(self, n_stocks=2, n_paths=10000, dt=1/252, T_max=0.25, r=0.0):
        """
        Initialize the market simulator
        
        Parameters:
        -----------
        n_stocks : int
            Number of stocks
        n_paths : int
            Number of simulation paths
        dt : float
            Time step size
        T_max : float
            Maximum time horizon
        r : float
            Risk-free rate
        """
        self.n_stocks = n_stocks
        self.n_paths = n_paths
        self.dt = dt
        self.T_max = T_max
        self.r = r
        self.n_steps = int(T_max / dt) + 1
        
    def simulate_gbm(self, S0, sigma, rho=None):
        """
        Simulate stock prices using geometric Brownian motion
        
        Parameters:
        -----------
        S0 : array-like
            Initial stock prices
        sigma : array-like
            Volatilities of the stocks
        rho : array-like or None
            Correlation matrix between stocks
            
        Returns:
        --------
        S : ndarray of shape (n_paths, n_steps, n_stocks)
            Simulated stock price paths
        """
        if rho is None:
            rho = np.eye(self.n_stocks)  # Default to identity (uncorrelated)
        
        # Ensure S0 and sigma are numpy arrays
        S0 = np.array(S0)
        sigma = np.array(sigma)
        
        # Generate correlated Brownian motions
        chol = np.linalg.cholesky(rho)
        
        # Initialize stock prices array
        S = np.zeros((self.n_paths, self.n_steps, self.n_stocks))
        S[:, 0, :] = S0
        
        # Time points
        time_points = np.arange(1, self.n_steps) * self.dt
        
        # Simulate paths
        for i in range(1, self.n_steps):
            # Generate random normal samples
            Z = np.random.normal(0, 1, (self.n_paths, self.n_stocks))
            
            # Apply correlation structure
            correlated_Z = Z @ chol.T
            
            # Calculate returns
            returns = (self.r - 0.5 * sigma**2) * self.dt + sigma * np.sqrt(self.dt) * correlated_Z
            
            # Update stock prices
            S[:, i, :] = S[:, i-1, :] * np.exp(returns)
        
        return S, time_points
    
    def create_ranked_index(self, S, weights=None):
        """
        Create an index by ranking stocks and applying weights
        
        Parameters:
        -----------
        S : ndarray of shape (n_paths, n_steps, n_stocks)
            Simulated stock price paths
        weights : array-like or None
            Weights to apply to ranked stocks (default: equal weights)
            
        Returns:
        --------
        I : ndarray of shape (n_paths, n_steps)
            Index values
        """
        if weights is None:
            weights = np.ones(self.n_stocks) / self.n_stocks
        
        # Initialize index array
        I = np.zeros((self.n_paths, self.n_steps))
        
        # For each time step and path
        for i in range(self.n_paths):
            for j in range(self.n_steps):
                # Sort stocks in descending order
                sorted_indices = np.argsort(-S[i, j, :])
                sorted_prices = S[i, j, sorted_indices]
                
                # Apply weights to create the index
                I[i, j] = np.sum(weights * sorted_prices)
        
        return I
    
    def compute_index_future(self, I, t_idx):
        """
        Compute index future price at time t for maturity T
        
        Parameters:
        -----------
        I : ndarray of shape (n_paths, n_steps)
            Index values
        t_idx : int
            Time index for future calculation
            
        Returns:
        --------
        F : float
            Future price
        """
        # Future price is the expected value of the index at maturity
        return np.mean(I[:, -1])
    
    def compute_option_prices(self, I, F, T, strikes):
        """
        Compute option prices for different strikes
        
        Parameters:
        -----------
        I : ndarray of shape (n_paths, n_steps)
            Index values
        F : float
            Future price
        T : float
            Time to maturity
        strikes : array-like
            Option strikes
            
        Returns:
        --------
        prices : ndarray
            Option prices for each strike
        """
        prices = np.zeros_like(strikes, dtype=float)
        
        # For each strike, compute the option price
        for i, K in enumerate(strikes):
            # Calculate payoff for each path
            payoffs = np.maximum(I[:, -1] - K, 0)
            
            # Option price is the expected discounted payoff
            prices[i] = np.exp(-self.r * T) * np.mean(payoffs)
        
        return prices
    
    def compute_implied_volatilities(self, option_prices, F, T, strikes):
        """
        Compute implied volatilities from option prices
        
        Parameters:
        -----------
        option_prices : array-like
            Option prices
        F : float
            Future price
        T : float
            Time to maturity
        strikes : array-like
            Option strikes
            
        Returns:
        --------
        ivs : ndarray
            Implied volatilities for each strike
        """
        ivs = np.zeros_like(strikes, dtype=float)
        
        # For each strike, compute the implied volatility
        for i, (price, K) in enumerate(zip(option_prices, strikes)):
            ivs[i] = BlackScholes.implied_volatility(price, F, K, self.r, T)
        
        return ivs
    
    def compute_atm_skew(self, I, maturities):
        """
        Compute ATM skew for different maturities
        
        Parameters:
        -----------
        I : ndarray of shape (n_paths, n_steps)
            Index values
        maturities : array-like
            Time to maturities
            
        Returns:
        --------
        skews : ndarray
            ATM skews for each maturity
        """
        skews = np.zeros_like(maturities, dtype=float)
        
        for i, T in enumerate(maturities):
            # Index for this maturity
            t_idx = min(int(T / self.dt), self.n_steps - 1)
            
            # Current index value (average across paths)
            S0 = np.mean(I[:, 0])
            
            # Future price
            F = self.compute_index_future(I, t_idx)
            
            # Define strikes around ATM
            h = 0.01  # Small increment for finite difference
            strikes = np.array([F * (1 - h), F, F * (1 + h)])
            
            # Compute option prices
            t_final_idx = min(int(T / self.dt), self.n_steps - 1)
            payoffs = np.maximum(I[:, t_final_idx].reshape(-1, 1) - strikes, 0)
            option_prices = np.exp(-self.r * T) * np.mean(payoffs, axis=0)
            
            # Compute implied volatilities
            ivs = np.zeros_like(strikes)
            for j, (price, K) in enumerate(zip(option_prices, strikes)):
                ivs[j] = BlackScholes.implied_volatility(price, F, K, self.r, T)
            
            # Compute skew using finite difference
            skews[i] = (ivs[2] - ivs[0]) / (2 * h * F)
        
        return skews

def run_simulation_case1():
    """
    Run simulation for case 1: Different initial prices
    s_1_0 > s_2_0 (no blow-up)
    """
    print("\nRunning simulation for Case 1: s_1_0 > s_2_0")
    
    # Parameters
    S0 = [100, 96]  # Initial stock prices
    sigma = [0.2, 0.6]  # Volatilities
    rho = np.eye(2)  # Correlation matrix (uncorrelated)
    weights = [1, 0]  # Index weights (only first stock contributes)
    
    # Create simulator
    sim = MarketSimulator(n_stocks=2, n_paths=10000, dt=0.05/365, T_max=0.25)
    
    # Simulate stock prices
    stock_prices, time_points = sim.simulate_gbm(S0, sigma, rho)
    
    # Create index
    index = sim.create_ranked_index(stock_prices, weights)
    
    # Compute ATM skew for different maturities
    maturities = np.linspace(0.01, 0.25, 20)
    skews = sim.compute_atm_skew(index, maturities)
    
    # Absolute value of skews
    abs_skews = np.abs(skews)
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.scatter(maturities, abs_skews, marker='o', color='blue', label='Simulated data')
    
    # Fit a power law: |Skew| ~ c * T^(-α)
    from scipy.optimize import curve_fit
    def power_law(x, c, alpha):
        return c * x**(-alpha)
    
    # Filter out any NaN values
    mask = ~np.isnan(abs_skews)
    params, _ = curve_fit(power_law, maturities[mask], abs_skews[mask], p0=[0.1, 0.1], bounds=([0, -1], [10, 1]))
    
    # Plot the fitted curve
    T_fit = np.linspace(0.01, 0.25, 100)
    plt.plot(T_fit, power_law(T_fit, *params), 'r-', label=f'Fit: c={params[0]:.4f}, α={params[1]:.4f}')
    
    plt.xlabel('Maturity (T)')
    plt.ylabel('|ATM Skew|')
    plt.title(f'ATM Skew vs Maturity (s_1_0={S0[0]}, s_2_0={S0[1]})')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig('case1_atm_skew.png')
    
    return maturities, abs_skews, params

def run_simulation_case2():
    """
    Run simulation for case 2: Same initial prices
    s_1_0 = s_2_0 (blow-up)
    """
    print("\nRunning simulation for Case 2: s_1_0 = s_2_0")
    
    # Parameters
    S0 = [100, 100]  # Initial stock prices (same)
    sigma = [0.2, 0.6]  # Volatilities
    rho = np.eye(2)  # Correlation matrix (uncorrelated)
    weights = [1, 0]  # Index weights (only first stock contributes)
    
    # Create simulator
    sim = MarketSimulator(n_stocks=2, n_paths=10000, dt=0.05/365, T_max=0.25)
    
    # Simulate stock prices
    stock_prices, time_points = sim.simulate_gbm(S0, sigma, rho)
    
    # Create index
    index = sim.create_ranked_index(stock_prices, weights)
    
    # Compute ATM skew for different maturities
    maturities = np.linspace(0.01, 0.25, 20)
    skews = sim.compute_atm_skew(index, maturities)
    
    # Absolute value of skews
    abs_skews = np.abs(skews)
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.scatter(maturities, abs_skews, marker='o', color='blue', label='Simulated data')
    
    # Fit a power law: |Skew| ~ c * T^(-α)
    from scipy.optimize import curve_fit
    def power_law(x, c, alpha):
        return c * x**(-alpha)
    
    # Filter out any NaN values
    mask = ~np.isnan(abs_skews)
    params, _ = curve_fit(power_law, maturities[mask], abs_skews[mask], p0=[0.1, 0.5], bounds=([0, 0], [10, 1]))
    
    # Plot the fitted curve
    T_fit = np.linspace(0.01, 0.25, 100)
    plt.plot(T_fit, power_law(T_fit, *params), 'r-', label=f'Fit: c={params[0]:.4f}, α={params[1]:.4f}')
    
    plt.xlabel('Maturity (T)')
    plt.ylabel('|ATM Skew|')
    plt.title(f'ATM Skew vs Maturity (s_1_0={S0[0]}, s_2_0={S0[1]})')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig('case2_atm_skew.png')
    
    return maturities, abs_skews, params

def run_quasi_blowup_analysis():
    """
    Run a simulation to demonstrate the quasi-blow-up phenomenon
    by varying the initial price of the second stock
    """
    print("\nRunning analysis for quasi-blow-up phenomenon")
    
    # Base parameters
    sigma = [0.2, 0.6]  # Volatilities
    rho = np.eye(2)  # Correlation matrix (uncorrelated)
    weights = [1, 0]  # Index weights (only first stock contributes)
    
    # Different initial prices for stock 2
    s2_values = [100, 98, 96, 94]
    results = []
    
    for s2 in s2_values:
        print(f"Processing s_2_0 = {s2}")
        S0 = [100, s2]  # Initial stock prices
        
        # Create simulator
        sim = MarketSimulator(n_stocks=2, n_paths=10000, dt=0.05/365, T_max=0.25)
        
        # Simulate stock prices
        stock_prices, time_points = sim.simulate_gbm(S0, sigma, rho)
        
        # Create index
        index = sim.create_ranked_index(stock_prices, weights)
        
        # Compute ATM skew for different maturities
        maturities = np.linspace(0.01, 0.25, 15)
        skews = sim.compute_atm_skew(index, maturities)
        
        # Absolute value of skews
        abs_skews = np.abs(skews)
        
        # Fit a power law: |Skew| ~ c * T^(-α)
        from scipy.optimize import curve_fit
        def power_law(x, c, alpha):
            return c * x**(-alpha)
        
        # Filter out any NaN values
        mask = ~np.isnan(abs_skews)
        if np.sum(mask) > 3:  # Need at least 3 points for fitting
            params, _ = curve_fit(power_law, maturities[mask], abs_skews[mask], p0=[0.1, 0.2], bounds=([0, 0], [10, 1]))
        else:
            params = [np.nan, np.nan]
        
        results.append({
            's2': s2,
            'maturities': maturities,
            'abs_skews': abs_skews,
            'params': params
        })
    
    # Plot results
    plt.figure(figsize=(12, 7))
    colors = ['blue', 'green', 'orange', 'red']
    
    for i, res in enumerate(results):
        plt.scatter(res['maturities'], res['abs_skews'], marker='o', color=colors[i], label=f'$S_2^0 = {res["s2"]}$')
        
        # Plot fitted curve if parameters are valid
        if not np.isnan(res['params'][0]):
            T_fit = np.linspace(0.01, 0.25, 100)
            plt.plot(T_fit, power_law(T_fit, *res['params']), color=colors[i], linestyle='-',
                    label=f'Fit: c={res["params"][0]:.4f}, α={res["params"][1]:.4f}')
    
    plt.xlabel('Maturity (T)')
    plt.ylabel('|ATM Skew|')
    plt.title('Quasi-blow-up Phenomenon: ATM Skew vs Maturity for Different Initial Prices')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig('quasi_blowup_analysis.png')
    
    return results

def simulate_fractional_model():
    """
    Simulate a modified fractional Stein-Stein model to demonstrate 
    both long memory in volatility and power law behavior in ATM skew
    """
    print("\nSimulating modified fractional Stein-Stein model")
    
    # Parameters
    n_paths = 5000
    n_steps = 100
    dt = 0.005
    T_max = n_steps * dt
    
    # Stock parameters
    S0 = [100, 100]  # Initial stock prices (same)
    sigma0 = [0.2, 0.6]  # Initial volatilities
    rho = [-0.5, -0.5]  # Correlation between price and volatility (leverage effect)
    weights = [1, 0]  # Index weights
    
    # Hurst parameters (long memory)
    H = [0.6, 0.7]  # Hurst exponents > 0.5 for long memory
    
    # Time grid
    time_grid = np.arange(n_steps) * dt
    maturities = np.linspace(0.01, T_max, 15)
    
    # Function to simulate fractional Brownian motion using Cholesky decomposition
    def simulate_fbm(n_steps, H, n_paths):
        # Covariance matrix
        cov_matrix = np.zeros((n_steps, n_steps))
        for i in range(n_steps):
            for j in range(n_steps):
                cov_matrix[i, j] = 0.5 * (dt**(2*H) * ((i+1)**(2*H) + (j+1)**(2*H) - abs(i-j)**(2*H)))
        
        # Cholesky decomposition
        try:
            L = np.linalg.cholesky(cov_matrix)
            
            # Generate paths
            Z = np.random.normal(0, 1, (n_paths, n_steps))
            fbm = Z @ L.T
            return fbm
        except:
            print("Warning: Covariance matrix is not positive definite. Using approximation.")
            # Use a simpler approximation
            fbm = np.zeros((n_paths, n_steps))
            for i in range(n_paths):
                fbm[i, 0] = np.random.normal(0, dt**H)
                for j in range(1, n_steps):
                    fbm[i, j] = fbm[i, j-1] + np.random.normal(0, dt**H)
            return fbm
    
    # Simulate stocks with fractional volatility
    stock_prices = np.zeros((n_paths, n_steps, 2))
    stock_prices[:, 0, :] = S0
    
    # Simulate independent Brownian motions for price and volatility
    W1 = np.random.normal(0, np.sqrt(dt), (n_paths, n_steps))
    W2 = np.random.normal(0, np.sqrt(dt), (n_paths, n_steps))
    B1 = np.random.normal(0, np.sqrt(dt), (n_paths, n_steps))
    B2 = np.random.normal(0, np.sqrt(dt), (n_paths, n_steps))
    
    # Simulate fractional Brownian motions for volatility
    fBm1 = simulate_fbm(n_steps, H[0], n_paths)
    fBm2 = simulate_fbm(n_steps, H[1], n_paths)
    
    # Constants for volatility normalization
    c0 = [5.0, 5.0]  # Large enough to keep volatility positive
    
    # Simulate volatility and stock prices
    vol1 = np.zeros((n_paths, n_steps))
    vol2 = np.zeros((n_paths, n_steps))
    
    for i in range(n_paths):
        # Initial volatility
        vol1[i, 0] = sigma0[0]
        vol2[i, 0] = sigma0[1]
        
        for j in range(1, n_steps):
            # Fractional volatility dynamics
            # Use a simple model: σ_t = σ_0 * (c_0 + fBm_t * exp(-fBm_t^2/2)) / c(t)
            # where c(t) is a normalization factor
            
            # Calculate normalization factor (approximately)
            c_t1 = np.sqrt(c0[0]**2 + 1/np.sqrt(2) * (j*dt)**(2*H[0]) / (0.5 + (j*dt)**(2*H[0]))**(3/2))
            c_t2 = np.sqrt(c0[1]**2 + 1/np.sqrt(2) * (j*dt)**(2*H[1]) / (0.5 + (j*dt)**(2*H[1]))**(3/2))
            
            # Calculate volatility
            vol1[i, j] = sigma0[0] * (c0[0] + fBm1[i, j] * np.exp(-fBm1[i, j]**2/2)) / c_t1
            vol2[i, j] = sigma0[1] * (c0[1] + fBm2[i, j] * np.exp(-fBm2[i, j]**2/2)) / c_t2
            
            # Ensure volatility is positive
            vol1[i, j] = max(vol1[i, j], 1e-6)
            vol2[i, j] = max(vol2[i, j], 1e-6)
            
            # Stock price dynamics with correlation
            dW1 = rho[0] * B1[i, j] + np.sqrt(1 - rho[0]**2) * W1[i, j]
            dW2 = rho[1] * B2[i, j] + np.sqrt(1 - rho[1]**2) * W2[i, j]
            
            # Update stock prices
            stock_prices[i, j, 0] = stock_prices[i, j-1, 0] * np.exp(-0.5 * vol1[i, j]**2 * dt + vol1[i, j] * dW1)
            stock_prices[i, j, 1] = stock_prices[i, j-1, 1] * np.exp(-0.5 * vol2[i, j]**2 * dt + vol2[i, j] * dW2)
    
    # Create ranked index
    index = np.zeros((n_paths, n_steps))
    for i in range(n_paths):
        for j in range(n_steps):
            # Sort stocks in descending order
            sorted_indices = np.argsort(-stock_prices[i, j, :])
            sorted_prices = stock_prices[i, j, sorted_indices]
            
            # Apply weights to create the index
            index[i, j] = np.sum(weights * sorted_prices)
    
    # Compute ATM skew for different maturities
    skews = np.zeros_like(maturities)
    
    for i, T in enumerate(maturities):
        # Index for this maturity
        t_idx = min(int(T / dt), n_steps - 1)
        
        # Current index value (average across paths)
        S0 = np.mean(index[:, 0])
        
        # Future price (expected value at maturity)
        F = np.mean(index[:, t_idx])
        
        # Define strikes around ATM
        h = 0.01  # Small increment for finite difference
        strikes = np.array([F * (1 - h), F, F * (1 + h)])
        
        # Compute option prices
        payoffs = np.maximum(index[:, t_idx].reshape(-1, 1) - strikes, 0)
        option_prices = np.mean(payoffs, axis=0)  # No discounting for simplicity
        
        # Compute implied volatilities
        ivs = np.zeros_like(strikes)
        for j, (price, K) in enumerate(zip(option_prices, strikes)):
            ivs[j] = BlackScholes.implied_volatility(price, F, K, 0, T)
        
        # Compute skew using finite difference
        if not np.isnan(ivs).any():
            skews[i] = (ivs[2] - ivs[0]) / (2 * h * F)
    
    # Absolute value of skews
    abs_skews = np.abs(skews)
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.scatter(maturities, abs_skews, marker='o', color='blue', label='Simulated data')
    
    # Fit a power law: |Skew| ~ c * T^(-α)
    from scipy.optimize import curve_fit
    def power_law(x, c, alpha):
        return c * x**(-alpha)
    
    # Filter out any NaN values
    mask = ~np.isnan(abs_skews)
    params, _ = curve_fit(power_law, maturities[mask], abs_skews[mask], p0=[0.1, 0.2], bounds=([0, 0], [10, 1]))
    
    # Plot the fitted curve
    T_fit = np.linspace(0.01, T_max, 100)
    plt.plot(T_fit, power_law(T_fit, *params), 'r-', label=f'Fit: c={params[0]:.4f}, α={params[1]:.4f}')
    
    plt.xlabel('Maturity (T)')
    plt.ylabel('|ATM Skew|')
    plt.title(f'Fractional Model: ATM Skew vs Maturity (H1={H[0]}, H2={H[1]})')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig('fractional_model_skew.png')
    
    # Plot volatility paths to show long memory
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    for i in range(min(5, n_paths)):
        plt.plot(time_grid, vol1[i, :], alpha=0.7)
    plt.title(f'Volatility Paths (H={H[0]})')
    plt.xlabel('Time')
    plt.ylabel('Volatility')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    # Calculate autocorrelation of volatility
    lag_max = 30
    lags = np.arange(lag_max)
    acf = np.zeros(lag_max)
    
    # Average autocorrelation across paths
    for i in range(n_paths):
        vol_centered = vol1[i, :] - np.mean(vol1[i, :])
        acf_i = np.correlate(vol_centered, vol_centered, mode='full')
        acf_i = acf_i[len(acf_i)//2:]
        acf_i = acf_i[:lag_max] / acf_i[0]
        acf += acf_i
    
    acf /= n_paths
    
    # Plot autocorrelation
    plt.bar(lags, acf, alpha=0.7)
    plt.title('Volatility Autocorrelation')
    plt.xlabel('Lag')
    plt.ylabel('Autocorrelation')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('fractional_model_volatility.png')
    
    return maturities, abs_skews, params

def main():
    # Run simulations for the two main cases
    case1_results = run_simulation_case1()
    case2_results = run_simulation_case2()
    
    # Run analysis for quasi-blow-up phenomenon
    quasi_blowup_results = run_quasi_blowup_analysis()
    
    # Simulate fractional model
    fractional_results = simulate_fractional_model()
    
    # Compare results
    print("\n=== Summary of Results ===")
    print("Case 1 (s_1_0 > s_2_0): Alpha =", case1_results[2][1])
    print("Case 2 (s_1_0 = s_2_0): Alpha =", case2_results[2][1])
    print("Fractional Model: Alpha =", fractional_results[2][1])
    
    # Compare the alphas from all quasi-blow-up scenarios
    print("\nQuasi-blow-up results:")
    for res in quasi_blowup_results:
        if not np.isnan(res['params'][0]):
            print(f"s_2_0 = {res['s2']}: Alpha = {res['params'][1]}")

if __name__ == "__main__":
    main()