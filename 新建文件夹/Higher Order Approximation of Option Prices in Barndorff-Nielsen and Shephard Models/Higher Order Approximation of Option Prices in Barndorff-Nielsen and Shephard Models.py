import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.integrate import quad
import time
from numba import jit
from tqdm import tqdm

# Set random seed for reproducibility
np.random.seed(42)

class BNSModel:
    """
    Barndorff-Nielsen and Shephard model for option pricing.
    """
    
    def __init__(self, model_type='inverse_gaussian', params=None):
        """
        Initialize the BNS model.
        
        Parameters:
        -----------
        model_type : str
            Type of BNS model ('inverse_gaussian' or 'gamma')
        params : dict
            Model parameters
        """
        self.model_type = model_type
        
        # Default parameters
        if params is None:
            if model_type == 'inverse_gaussian':
                self.params = {
                    'a': 0.5,     # Scale parameter
                    'b': 80.0,    # Shape parameter
                    'lambda': 5.0, # Mean reversion rate
                    'rho': -0.5,  # Correlation
                    'r': 0.03,    # Risk-free rate
                    'sigma0': 0.2  # Initial volatility
                }
            elif model_type == 'gamma':
                self.params = {
                    'a': 0.5,     # Scale parameter
                    'b': 80.0,    # Shape parameter
                    'lambda': 5.0, # Mean reversion rate
                    'rho': -0.5,  # Correlation
                    'r': 0.03,    # Risk-free rate
                    'sigma0': 0.2  # Initial volatility
                }
            else:
                raise ValueError(f"Unknown model type: {model_type}")
        else:
            self.params = params
        
        # Check if the correlation parameter satisfies the condition
        if self.model_type == 'inverse_gaussian':
            kappa_hat = self.params['b']**2 / 2
            if self.params['rho'] >= kappa_hat:
                raise ValueError(f"Correlation parameter {self.params['rho']} exceeds kappa_hat {kappa_hat}")
        elif self.model_type == 'gamma':
            kappa_hat = self.params['b']
            if self.params['rho'] >= kappa_hat:
                raise ValueError(f"Correlation parameter {self.params['rho']} exceeds kappa_hat {kappa_hat}")
    
    def kappa(self, theta):
        """
        Cumulant generating function of the background driving Lévy process.
        
        Parameters:
        -----------
        theta : float
            Argument of the cumulant function
            
        Returns:
        --------
        float
            Value of the cumulant function
        """
        if self.model_type == 'inverse_gaussian':
            a, b = self.params['a'], self.params['b']
            return a * theta * np.sqrt(b**2 - 2*theta) / b
        elif self.model_type == 'gamma':
            a, b = self.params['a'], self.params['b']
            return a * theta / (b - theta)
    
    def kappa_derivative(self, theta, order=1):
        """
        Derivative of the cumulant generating function.
        
        Parameters:
        -----------
        theta : float
            Argument of the cumulant function
        order : int
            Order of the derivative
            
        Returns:
        --------
        float
            Value of the derivative
        """
        if self.model_type == 'inverse_gaussian':
            a, b = self.params['a'], self.params['b']
            if order == 1:
                return a * (b**2 - theta) / (b * np.sqrt(b**2 - 2*theta))
            else:
                # Use recursive formula from the paper (equation 29)
                if order == 1:
                    return a * (b**2 - theta) / (b * np.sqrt(b**2 - 2*theta))
                else:
                    # Calculate phi_n recursively (equation 30)
                    phi_n = 1
                    for n in range(2, order+1):
                        phi_n = phi_n * (2*n - 3) + (2*n - 3)
                    
                    # Calculate kappa^(n)(theta) using phi_n (equation 29)
                    term1 = phi_n * a * (b**2 - 2*theta)**(-(2*order-1)/2)
                    term2 = (2*order - 1) * a * theta * (b**2 - 2*theta)**(-(2*order+1)/2)
                    return term1 + term2
        
        elif self.model_type == 'gamma':
            a, b = self.params['a'], self.params['b']
            if order == 1:
                return a * b / (b - theta)**2
            else:
                # Use formula from the paper (equation 32)
                return (
                    np.math.factorial(order) * a * (b - theta)**(-order) + 
                    np.math.factorial(order) * a * theta * (b - theta)**(-order-1)
                )
    
    def alpha(self, s, t):
        """
        Alpha function defined in equation (5).
        
        Parameters:
        -----------
        s, t : float
            Time parameters
            
        Returns:
        --------
        float
            Value of alpha(s,t)
        """
        lambda_ = self.params['lambda']
        return (1 - np.exp(-lambda_ * (t - s))) / lambda_
    
    def expected_integrated_variance(self, T):
        """
        Expected value of the integrated variance process.
        
        Parameters:
        -----------
        T : float
            Time horizon
            
        Returns:
        --------
        float
            Expected integrated variance
        """
        lambda_ = self.params['lambda']
        sigma0_squared = self.params['sigma0']**2
        kappa_prime_0 = self.kappa_derivative(0, order=1)
        
        alpha_0T = self.alpha(0, T)
        return alpha_0T * (sigma0_squared - kappa_prime_0) + kappa_prime_0 * T
    
    def integrated_variance_central_moment(self, T, order):
        """
        Calculate the central moment of the integrated variance.
        
        Parameters:
        -----------
        T : float
            Time horizon
        order : int
            Order of the moment
            
        Returns:
        --------
        float
            Central moment of the integrated variance
        """
        if order == 0:
            return 1
        elif order == 1:
            return 0  # Central moment of order 1 is always 0
        else:
            # Use the recursive formula from Corollary 2 (equation 27)
            lambda_ = self.params['lambda']
            kappa_prime_0 = self.kappa_derivative(0, order=1)
            alpha_0T = self.alpha(0, T)
            
            # Calculate H_{0,order} using the recursive relationship
            H = [0] * (order + 1)
            H[0] = 1
            
            for h in range(1, order + 1):
                term1 = kappa_prime_0 * (alpha_0T - T) * H[h-1]
                
                term2 = 0
                for i in range(1, h+1):
                    binomial_coef = np.math.comb(h-1, i-1)
                    kappa_i_0 = self.kappa_derivative(0, order=i)
                    
                    # Calculate the integral term
                    integral_term = 0
                    for j in range(1, i+1):
                        integral_term += (1/j) * np.math.comb(i, j) * (-1)**j * alpha_0T**j
                    
                    integral_term = T + integral_term
                    term2 += (1/lambda_**(i-1)) * binomial_coef * H[h-i] * kappa_i_0 * integral_term
                
                H[h] = term1 + term2
            
            return H[order]
    
    def price_term_moment(self, T, order, ell=0):
        """
        Calculate the moment E[P_T^ell * (I_T - E[I_T])^order].
        
        Parameters:
        -----------
        T : float
            Time horizon
        order : int
            Order of the integrated variance moment
        ell : int
            Power of the price term
            
        Returns:
        --------
        float
            Mixed moment
        """
        lambda_ = self.params['lambda']
        rho = self.params['rho']
        
        # For simple cases, we can use closed formulas
        if order == 0:
            if ell == 0:
                return 1
            else:
                # E[P_T^ell] = e^(lambda*T*(kappa(ell*rho) - ell*kappa(rho)))
                return np.exp(lambda_ * T * (self.kappa(ell*rho) - ell * self.kappa(rho)))
        
        if ell == 0:
            return self.integrated_variance_central_moment(T, order)
        
        # For mixed moments, use the recursive formula from Corollary 2
        # This is E[P_T^ell * (I_T - E[I_T])^order] = e^(lambda*T*(kappa(ell*rho) - ell*kappa(rho))) * H_{ell,order}
        
        # First, check if ell*rho is in the domain of kappa
        if self.model_type == 'inverse_gaussian':
            kappa_hat = self.params['b']**2 / 2
            if ell*rho >= kappa_hat:
                raise ValueError(f"ell*rho = {ell*rho} exceeds kappa_hat {kappa_hat}")
        elif self.model_type == 'gamma':
            kappa_hat = self.params['b']
            if ell*rho >= kappa_hat:
                raise ValueError(f"ell*rho = {ell*rho} exceeds kappa_hat {kappa_hat}")
        
        # Calculate the exponential term
        exp_term = np.exp(lambda_ * T * (self.kappa(ell*rho) - ell * self.kappa(rho)))
        
        # Calculate H_{ell,order} using the recursive relationship
        H = [0] * (order + 1)
        H[0] = 1
        
        kappa_prime_0 = self.kappa_derivative(0, order=1)
        alpha_0T = self.alpha(0, T)
        
        for h in range(1, order + 1):
            term1 = kappa_prime_0 * (alpha_0T - T) * H[h-1]
            
            term2 = 0
            for i in range(1, h+1):
                binomial_coef = np.math.comb(h-1, i-1)
                kappa_i_ell_rho = self.kappa_derivative(ell*rho, order=i)
                
                # Calculate the integral term
                integral_term = 0
                for j in range(1, i+1):
                    integral_term += (1/j) * np.math.comb(i, j) * (-1)**j * alpha_0T**j
                
                integral_term = T + integral_term
                term2 += (1/lambda_**(i-1)) * binomial_coef * H[h-i] * kappa_i_ell_rho * integral_term
            
            H[h] = term1 + term2
        
        return exp_term * H[order]
    
    def mixed_central_moment(self, T, n, k):
        """
        Calculate the mixed central moment E[(P_T - 1)^(n-k) * (I_T - E[I_T])^k].
        
        Parameters:
        -----------
        T : float
            Time horizon
        n : int
            Total order of the moment
        k : int
            Order of the integrated variance part
            
        Returns:
        --------
        float
            Mixed central moment
        """
        if k == 1 or n - k == 1:
            return 0  # First order central moments are zero
        
        # Use the binomial theorem (equation 25)
        result = 0
        for ell in range(n-k+1):
            binomial_coef = np.math.comb(n-k, ell)
            sign = (-1)**(n-k-ell)
            
            # Calculate E[P_T^ell * (I_T - E[I_T])^k]
            moment = self.price_term_moment(T, k, ell)
            
            result += binomial_coef * sign * moment
        
        return result

def bs_put(S, K, T, r, sigma):
    """
    Black-Scholes formula for a European put option.
    
    Parameters:
    -----------
    S : float
        Spot price
    K : float
        Strike price
    T : float
        Time to maturity
    r : float
        Risk-free rate
    sigma : float
        Volatility
    
    Returns:
    --------
    float
        Option price
    """
    if sigma <= 0:
        return max(K * np.exp(-r * T) - S, 0)
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def bs_put_partial_derivative(S, y, order_x, order_y):
    """
    Partial derivative of the Black-Scholes put price with respect to x (spot) and y (variance).
    
    Parameters:
    -----------
    S : float
        Spot price
    y : float
        Integrated variance
    order_x : int
        Order of derivative with respect to x
    order_y : int
        Order of derivative with respect to y
    
    Returns:
    --------
    float
        Value of the partial derivative
    """
    # We'll implement this for the specific orders needed in the approximation
    K = 1  # Normalized strike
    r = 0.03  # Risk-free rate (fixed for simplicity)
    
    if y <= 0:
        # Handle the case where variance is zero
        return 0
    
    sqrt_y = np.sqrt(y)
    d_plus = (np.log(S) + r * 1 + 0.5 * y) / sqrt_y
    d_minus = d_plus - sqrt_y
    
    # Calculate the density function
    phi_d_plus = norm.pdf(d_plus)
    
    # Second order derivatives from Remark 2
    if order_x == 2 and order_y == 0:
        return phi_d_plus / (S * sqrt_y)
    elif order_x == 0 and order_y == 2:
        return S * phi_d_plus * (d_minus * d_plus - 1) / (4 * y**(3/2))
    elif order_x == 1 and order_y == 1:
        return -phi_d_plus * d_minus / (2 * y)
    
    # Third order derivatives from equations (33)-(36)
    if order_x == 3 and order_y == 0:
        return -phi_d_plus * (d_plus + sqrt_y) / (S**2 * y)
    elif order_x == 0 and order_y == 3:
        return S * phi_d_plus * ((d_minus * d_plus - 2)**2 - d_plus**2 - d_minus**2 - 1) / (8 * y**(5/2))
    elif order_x == 2 and order_y == 1:
        return phi_d_plus * (d_minus * d_plus - 1) / (2 * S * y**(3/2))
    elif order_x == 1 and order_y == 2:
        return -phi_d_plus * (d_minus * d_plus / 2 - d_plus / 2 - d_minus) / (2 * y**2)
    
    # For higher orders, we would need to implement the recursive pattern from Proposition 1
    # This is quite complex and would require a more sophisticated implementation
    
    raise NotImplementedError(f"Partial derivative of order ({order_x}, {order_y}) not implemented")

def bns_option_approximation(model, S0, K, T, N=2):
    """
    Approximate the price of a European put option in the BNS model using an Nth order Taylor expansion.
    
    Parameters:
    -----------
    model : BNSModel
        The BNS model
    S0 : float
        Initial stock price
    K : float
        Strike price
    T : float
        Time to maturity
    N : int
        Order of the Taylor expansion
    
    Returns:
    --------
    float
        Approximated option price
    """
    # Calculate the expected integrated variance
    E_IT = model.expected_integrated_variance(T)
    
    # Calculate the base Black-Scholes price
    r = model.params['r']
    base_price = bs_put(S0, K, T, r, np.sqrt(E_IT / T))
    
    # If N = 0, return the base price (0th order approximation)
    if N == 0:
        return base_price
    
    # For higher orders, calculate the correction terms
    correction = 0
    for n in range(2, N+1):
        for k in range(n+1):
            # Calculate the mixed central moment
            moment = model.mixed_central_moment(T, n, k)
            
            # Calculate the partial derivative
            derivative = bs_put_partial_derivative(S0/K, E_IT, n-k, k)
            
            # Add the correction term
            binomial_coef = np.math.comb(n, k)
            correction += (1 / np.math.factorial(n)) * binomial_coef * S0**(n-k) * moment * derivative
    
    # Return the approximated option price
    return base_price + correction

def bns_characteristic_function(model, u, T):
    """
    Calculate the characteristic function of the log price in the BNS model.
    
    Parameters:
    -----------
    model : BNSModel
        The BNS model
    u : complex
        Argument of the characteristic function
    T : float
        Time horizon
    
    Returns:
    --------
    complex
        Value of the characteristic function
    """
    # Extract model parameters
    r = model.params['r']
    lambda_ = model.params['lambda']
    rho = model.params['rho']
    sigma0_squared = model.params['sigma0']**2
    
    # Calculate the first term
    term1 = 1j * u * (r * T - lambda_ * model.kappa(rho) * T)
    
    # Calculate the second term
    term2 = -0.5 * (1j * u + u**2) * sigma0_squared / lambda_ * (1 - np.exp(-lambda_ * T))
    
    # Calculate the integral term
    def f1(u):
        return 1j * u * rho - 0.5 * (1j * u + u**2) / lambda_ * (1 - np.exp(-lambda_ * T))
    
    def f2(u):
        return 1j * u * rho - 0.5 * (1j * u + u**2) / lambda_
    
    if model.model_type == 'inverse_gaussian':
        a, b = model.params['a'], model.params['b']
        
        # Use the formula from Appendix B (equation 72)
        term3_1 = a * (np.sqrt(b**2 - 2 * f1(u)) - np.sqrt(b**2 - 2 * 1j * u * rho))
        
        numerator = np.sqrt(b**2 - 2 * 1j * u * rho)
        denominator = np.sqrt(2 * f2(u) - b**2)
        
        term3_2 = 2 * a * f2(u) / denominator * (
            np.arctan(numerator / denominator) - 
            np.arctan(np.sqrt(b**2 - 2 * f1(u)) / denominator)
        )
        
        term3 = term3_1 + term3_2
        
    elif model.model_type == 'gamma':
        a, b = model.params['a'], model.params['b']
        
        # Use the formula from Appendix B (equation 75)
        term3 = a / (b - f2(u)) * (
            b * np.log((b - f1(u)) / (b - 1j * u * rho)) + 
            f2(u) * lambda_ * T
        )
    
    # Return the characteristic function
    return np.exp(term1 + term2 + term3)

def carr_madan_option_price(model, S0, K, T, option_type='put'):
    """
    Calculate the price of a European option using the Carr-Madan method with the characteristic function.
    
    Parameters:
    -----------
    model : BNSModel
        The BNS model
    S0 : float
        Initial stock price
    K : float
        Strike price
    T : float
        Time to maturity
    option_type : str
        Type of option ('put' or 'call')
    
    Returns:
    --------
    float
        Option price
    """
    # Extract the risk-free rate
    r = model.params['r']
    
    # Define the damping factor
    alpha = 1.5
    
    # Define the integrand
    def integrand(v):
        if option_type == 'put':
            u = v - (alpha + 1) * 1j
            modified_cf = bns_characteristic_function(model, u, T) / (alpha**2 + alpha - v**2 + 1j * (2 * alpha + 1) * v)
        else:  # call option
            u = v + 1j * alpha
            modified_cf = bns_characteristic_function(model, u, T) / (alpha**2 + alpha - v**2 + 1j * (2 * alpha + 1) * v)
        
        return np.real(np.exp(-1j * v * np.log(K / S0)) * modified_cf)
    
    # Perform the numerical integration
    integral, _ = quad(integrand, 0, 100, limit=1000)
    
    # Calculate the option price
    if option_type == 'put':
        price = K * np.exp(-r * T) * (0.5 + 1/np.pi * integral)
    else:  # call option
        price = S0 - K * np.exp(-r * T) + K * np.exp(-r * T) * (0.5 + 1/np.pi * integral)
    
    return price

def monte_carlo_option_price(model, S0, K, T, n_paths=10000, n_steps=100):
    """
    Calculate the price of a European put option using Monte Carlo simulation.
    
    Parameters:
    -----------
    model : BNSModel
        The BNS model
    S0 : float
        Initial stock price
    K : float
        Strike price
    T : float
        Time to maturity
    n_paths : int
        Number of simulation paths
    n_steps : int
        Number of time steps
    
    Returns:
    --------
    float
        Option price
    """
    # Extract model parameters
    r = model.params['r']
    lambda_ = model.params['lambda']
    rho = model.params['rho']
    sigma0_squared = model.params['sigma0']**2
    
    # Time step
    dt = T / n_steps
    
    # Initialize the stock price and variance paths
    S = np.zeros((n_paths, n_steps + 1))
    sigma_squared = np.zeros((n_paths, n_steps + 1))
    
    # Set initial values
    S[:, 0] = S0
    sigma_squared[:, 0] = sigma0_squared
    
    # Generate Brownian motion increments
    dW = np.random.normal(0, np.sqrt(dt), (n_paths, n_steps))
    
    # Generate Lévy process increments based on the model type
    if model.model_type == 'inverse_gaussian':
        a, b = model.params['a'], model.params['b']
        # For inverse Gaussian, we need to simulate inverse Gaussian increments
        dZ = np.random.normal(0, 1, (n_paths, n_steps))**2 / b**2
        dZ = a * dt * (1 + np.sqrt(2 * dZ / (a * dt)) * dZ)
    elif model.model_type == 'gamma':
        a, b = model.params['a'], model.params['b']
        # For gamma, we use gamma increments
        dZ = np.random.gamma(a * dt, 1/b, (n_paths, n_steps))
    
    # Simulate the paths
    for i in range(n_steps):
        # Update the variance process
        sigma_squared[:, i+1] = np.exp(-lambda_ * dt) * sigma_squared[:, i] + dZ[:, i]
        
        # Update the stock price
        drift = (r - lambda_ * model.kappa(rho) - 0.5 * sigma_squared[:, i]) * dt
        diffusion = np.sqrt(sigma_squared[:, i]) * dW[:, i]
        jump = rho * dZ[:, i]
        
        S[:, i+1] = S[:, i] * np.exp(drift + diffusion + jump)
    
    # Calculate the put option payoff
    payoff = np.maximum(K - S[:, -1], 0)
    
    # Calculate the option price
    price = np.exp(-r * T) * np.mean(payoff)
    
    return price

def compare_methods(model_type='inverse_gaussian', S0=1.0, K=1.0, T_values=None, lambda_values=None):
    """
    Compare different option pricing methods for the BNS model.
    
    Parameters:
    -----------
    model_type : str
        Type of BNS model ('inverse_gaussian' or 'gamma')
    S0 : float
        Initial stock price
    K : float
        Strike price
    T_values : list
        List of maturity times to test
    lambda_values : list
        List of mean reversion rates to test
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with the comparison results
    """
    if T_values is None:
        T_values = [0.25, 0.5, 1.0, 2.0]
    
    if lambda_values is None:
        lambda_values = [1.0, 5.0, 10.0, 20.0, 50.0, 100.0]
    
    # Initialize the results
    results = []
    
    # Create a base model
    base_params = {
        'a': 0.5,
        'b': 80.0,
        'lambda': 5.0,
        'rho': -0.5,
        'r': 0.03,
        'sigma0': 0.2
    }
    
    # Test different lambda values
    for lambda_ in lambda_values:
        # Update the model parameters
        params = base_params.copy()
        params['lambda'] = lambda_
        
        # Create the model
        model = BNSModel(model_type=model_type, params=params)
        
        # Calculate the option prices
        for T in T_values:
            # Characteristic function method
            start_time = time.time()
            try:
                cf_price = carr_madan_option_price(model, S0, K, T)
                cf_time = time.time() - start_time
            except Exception as e:
                cf_price = np.nan
                cf_time = np.nan
                print(f"CF method failed for lambda={lambda_}, T={T}: {e}")
            
            # Monte Carlo method
            start_time = time.time()
            try:
                mc_price = monte_carlo_option_price(model, S0, K, T)
                mc_time = time.time() - start_time
            except Exception as e:
                mc_price = np.nan
                mc_time = np.nan
                print(f"MC method failed for lambda={lambda_}, T={T}: {e}")
            
            # Approximation methods
            approx_prices = []
            approx_times = []
            
            for N in range(0, 7):
                start_time = time.time()
                try:
                    approx_price = bns_option_approximation(model, S0, K, T, N)
                    approx_time = time.time() - start_time
                except Exception as e:
                    approx_price = np.nan
                    approx_time = np.nan
                    print(f"Approximation (order {N}) failed for lambda={lambda_}, T={T}: {e}")
                
                approx_prices.append(approx_price)
                approx_times.append(approx_time)
            
            # Store the results
            results.append({
                'lambda': lambda_,
                'T': T,
                'CF_Price': cf_price,
                'CF_Time': cf_time,
                'MC_Price': mc_price,
                'MC_Time': mc_time,
                'Approx0_Price': approx_prices[0],
                'Approx0_Time': approx_times[0],
                'Approx1_Price': approx_prices[1],
                'Approx1_Time': approx_times[1],
                'Approx2_Price': approx_prices[2],
                'Approx2_Time': approx_times[2],
                'Approx3_Price': approx_prices[3],
                'Approx3_Time': approx_times[3],
                'Approx4_Price': approx_prices[4],
                'Approx4_Time': approx_times[4],
                'Approx5_Price': approx_prices[5],
                'Approx5_Time': approx_times[5],
                'Approx6_Price': approx_prices[6],
                'Approx6_Time': approx_times[6],
            })
    
    # Convert to DataFrame
    import pandas as pd
    df = pd.DataFrame(results)
    
    return df

def plot_option_prices(model, S0=1.0, K_values=None, T=1.0, N_values=None):
    """
    Plot option prices for different strikes and approximation orders.
    
    Parameters:
    -----------
    model : BNSModel
        The BNS model
    S0 : float
        Initial stock price
    K_values : list
        List of strike prices
    T : float
        Time to maturity
    N_values : list
        List of approximation orders
    
    Returns:
    --------
    None
    """
    if K_values is None:
        K_values = np.linspace(0.7, 1.3, 25)
    
    if N_values is None:
        N_values = [0, 2, 4, 6]
    
    # Calculate the option prices
    cf_prices = []
    approx_prices = {N: [] for N in N_values}
    
    for K in tqdm(K_values, desc="Calculating option prices"):
        # Characteristic function method
        try:
            cf_price = carr_madan_option_price(model, S0, K, T)
        except:
            cf_price = np.nan
        cf_prices.append(cf_price)
        
        # Approximation methods
        for N in N_values:
            try:
                approx_price = bns_option_approximation(model, S0, K, T, N)
            except:
                approx_price = np.nan
            approx_prices[N].append(approx_price)
    
    # Plot the option prices
    plt.figure(figsize=(12, 8))
    
    # Characteristic function prices
    plt.plot(K_values, cf_prices, 'k-', linewidth=2, label='Characteristic Function')
    
    # Approximation prices
    for N in N_values:
        plt.plot(K_values, approx_prices[N], '--', linewidth=1.5, label=f'Approximation (N={N})')
    
    plt.xlabel('Strike Price (K)')
    plt.ylabel('Option Price')
    plt.title(f'Option Prices for Different Strikes (T={T}, Model={model.model_type})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(f'option_prices_{model.model_type}_T{T}.png')
    plt.close()
    
    # Plot the absolute errors
    plt.figure(figsize=(12, 8))
    
    for N in N_values:
        abs_errors = [abs(approx - cf) for approx, cf in zip(approx_prices[N], cf_prices)]
        plt.plot(K_values, abs_errors, '--', linewidth=1.5, label=f'Approximation (N={N})')
    
    plt.xlabel('Strike Price (K)')
    plt.ylabel('Absolute Error')
    plt.yscale('log')
    plt.title(f'Absolute Errors for Different Strikes (T={T}, Model={model.model_type})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(f'option_errors_{model.model_type}_T{T}.png')
    plt.close()

def plot_lambda_stability(df, T_value=1.0):
    """
    Plot the stability of different methods as lambda increases.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with the comparison results
    T_value : float
        Time to maturity to focus on
    
    Returns:
    --------
    None
    """
    # Filter the data for the specified T
    df_filtered = df[df['T'] == T_value]
    
    # Plot the option prices
    plt.figure(figsize=(12, 8))
    
    # Characteristic function prices
    plt.plot(df_filtered['lambda'], df_filtered['CF_Price'], 'k-', linewidth=2, label='Characteristic Function')
    
    # Monte Carlo prices
    plt.plot(df_filtered['lambda'], df_filtered['MC_Price'], 'r-', linewidth=2, label='Monte Carlo')
    
    # Approximation prices
    plt.plot(df_filtered['lambda'], df_filtered['Approx2_Price'], 'b--', linewidth=1.5, label='Approximation (N=2)')
    plt.plot(df_filtered['lambda'], df_filtered['Approx4_Price'], 'g--', linewidth=1.5, label='Approximation (N=4)')
    plt.plot(df_filtered['lambda'], df_filtered['Approx6_Price'], 'm--', linewidth=1.5, label='Approximation (N=6)')
    
    plt.xlabel('Mean Reversion Rate (lambda)')
    plt.ylabel('Option Price')
    plt.title(f'Option Prices for Different Lambda Values (T={T_value})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(f'lambda_stability_T{T_value}.png')
    plt.close()
    
    # Plot the computation times
    plt.figure(figsize=(12, 8))
    
    # Characteristic function times
    plt.plot(df_filtered['lambda'], df_filtered['CF_Time'], 'k-', linewidth=2, label='Characteristic Function')
    
    # Monte Carlo times
    plt.plot(df_filtered['lambda'], df_filtered['MC_Time'], 'r-', linewidth=2, label='Monte Carlo')
    
    # Approximation times
    plt.plot(df_filtered['lambda'], df_filtered['Approx2_Time'], 'b--', linewidth=1.5, label='Approximation (N=2)')
    plt.plot(df_filtered['lambda'], df_filtered['Approx4_Time'], 'g--', linewidth=1.5, label='Approximation (N=4)')
    plt.plot(df_filtered['lambda'], df_filtered['Approx6_Time'], 'm--', linewidth=1.5, label='Approximation (N=6)')
    
    plt.xlabel('Mean Reversion Rate (lambda)')
    plt.ylabel('Computation Time (s)')
    plt.yscale('log')
    plt.title(f'Computation Times for Different Lambda Values (T={T_value})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(f'lambda_timing_T{T_value}.png')
    plt.close()

def verify_error_bounds(model, S0=1.0, K_values=None, T_values=None, N_values=None):
    """
    Verify the error bounds for the approximation method.
    
    Parameters:
    -----------
    model : BNSModel
        The BNS model
    S0 : float
        Initial stock price
    K_values : list
        List of strike prices
    T_values : list
        List of maturity times
    N_values : list
        List of approximation orders
    
    Returns:
    --------
    None
    """
    if K_values is None:
        K_values = np.logspace(-1, 1, 20)
    
    if T_values is None:
        T_values = [0.25, 0.5, 1.0, 2.0]
    
    if N_values is None:
        N_values = [2, 4, 6]
    
    # Calculate the errors for different K values
    for T in T_values:
        errors = {N: [] for N in N_values}
        
        for K in tqdm(K_values, desc=f"Calculating errors for T={T}"):
            # Characteristic function price (reference)
            cf_price = carr_madan_option_price(model, S0, K, T)
            
            # Approximation prices
            for N in N_values:
                approx_price = bns_option_approximation(model, S0, K, T, N)
                error = abs(approx_price - cf_price)
                errors[N].append(error)
        
        # Plot the errors
        plt.figure(figsize=(12, 8))
        
        for N in N_values:
            plt.plot(K_values, errors[N], '--', linewidth=1.5, label=f'Approximation (N={N})')
        
        # Add reference lines for the error bounds
        if model.model_type == 'inverse_gaussian':
            b = model.params['b']
            for N in N_values:
                if model.params['rho'] == 0:
                    # O(1/b^(N+2))
                    ref_curve = [1.0 / (b**(N+2)) for _ in K_values]
                else:
                    # O(1/b^(N+1))
                    ref_curve = [1.0 / (b**(N+1)) for _ in K_values]
                plt.plot(K_values, ref_curve, '-', linewidth=1, alpha=0.5, label=f'Reference O(1/b^{N+1 if model.params["rho"] != 0 else N+2})')
        
        elif model.model_type == 'gamma':
            b = model.params['b']
            for N in N_values:
                # O(1/b^(N+1))
                ref_curve = [1.0 / (b**(N+1)) for _ in K_values]
                plt.plot(K_values, ref_curve, '-', linewidth=1, alpha=0.5, label=f'Reference O(1/b^{N+1})')
        
        plt.xlabel('Strike Price (K)')
        plt.ylabel('Absolute Error')
        plt.xscale('log')
        plt.yscale('log')
        plt.title(f'Approximation Errors for T={T} ({model.model_type} model)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save the figure
        plt.savefig(f'error_bounds_{model.model_type}_T{T}.png')
        plt.close()
    
    # Calculate the errors for different T values
    for K in [0.8, 1.0, 1.2]:
        T_values_fine = np.linspace(0.1, 2.0, 20)
        errors = {N: [] for N in N_values}
        
        for T in tqdm(T_values_fine, desc=f"Calculating errors for K={K}"):
            # Characteristic function price (reference)
            cf_price = carr_madan_option_price(model, S0, K, T)
            
            # Approximation prices
            for N in N_values:
                approx_price = bns_option_approximation(model, S0, K, T, N)
                error = abs(approx_price - cf_price)
                errors[N].append(error)
        
        # Plot the errors
        plt.figure(figsize=(12, 8))
        
        for N in N_values:
            plt.plot(T_values_fine, errors[N], '--', linewidth=1.5, label=f'Approximation (N={N})')
        
        plt.xlabel('Time to Maturity (T)')
        plt.ylabel('Absolute Error')
        plt.yscale('log')
        plt.title(f'Approximation Errors for K={K} ({model.model_type} model)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save the figure
        plt.savefig(f'error_time_{model.model_type}_K{K}.png')
        plt.close()

# Run the tests
if __name__ == "__main__":
    # Create the models
    inverse_gaussian_model = BNSModel(model_type='inverse_gaussian')
    gamma_model = BNSModel(model_type='gamma')
    
    # Compare the methods for different lambda values
    df_ig = compare_methods(model_type='inverse_gaussian')
    df_gamma = compare_methods(model_type='gamma')
    
    # Plot the option prices for different strikes
    plot_option_prices(inverse_gaussian_model, T=0.5)
    plot_option_prices(inverse_gaussian_model, T=1.0)
    plot_option_prices(gamma_model, T=0.5)
    plot_option_prices(gamma_model, T=1.0)
    
    # Plot the stability for different lambda values
    plot_lambda_stability(df_ig, T_value=1.0)
    plot_lambda_stability(df_gamma, T_value=1.0)
    
    # Verify the error bounds
    verify_error_bounds(inverse_gaussian_model)
    verify_error_bounds(gamma_model)