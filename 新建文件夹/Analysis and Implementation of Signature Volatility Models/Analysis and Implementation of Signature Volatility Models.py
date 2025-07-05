import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import special, integrate, stats
from scipy.optimize import minimize
from tqdm import tqdm
import time
import multiprocessing as mp
from numba import jit

# Set random seed for reproducibility
np.random.seed(42)

#######################################
# PART 1: SIGNATURE IMPLEMENTATION
#######################################

class Signature:
    """Class to handle signature of a path"""
    
    def __init__(self, max_level=3, dimension=2):
        """
        Initialize signature object
        
        Args:
            max_level: Maximum level of the signature
            dimension: Dimension of the underlying path
        """
        self.max_level = max_level
        self.dimension = dimension
        self.truncation_size = self._compute_truncation_size()
        
        # Create index mappings
        self.word_to_index, self.index_to_word = self._create_index_mappings()
        
        # Pre-compute projection and shuffle operators
        self.projections = self._precompute_projections()
        self.shuffle_table = self._precompute_shuffle_product()
    
    def _compute_truncation_size(self):
        """Compute the size of the truncated signature"""
        return sum(self.dimension**k for k in range(self.max_level + 1))
    
    def _create_index_mappings(self):
        """Create mappings between words and indices"""
        word_to_index = {"": 0}  # Empty word maps to index 0
        index_to_word = [""]  # Index 0 corresponds to empty word
        
        idx = 1
        for level in range(1, self.max_level + 1):
            for word in self._generate_words(level):
                word_to_index[word] = idx
                index_to_word.append(word)
                idx += 1
        
        return word_to_index, index_to_word
    
    def _generate_words(self, level):
        """Generate all words of given level"""
        if level == 0:
            yield ""
            return
        
        for word in self._generate_words(level - 1):
            for i in range(1, self.dimension + 1):
                yield word + str(i)
    
    def _precompute_projections(self):
        """Precompute projection operators"""
        projections = {}
        
        for i in range(1, self.dimension + 1):
            projections[str(i)] = {}
            
            for word, idx in self.word_to_index.items():
                if len(word) >= 1:
                    new_word = word[1:] if word.startswith(str(i)) else ""
                    projections[str(i)][idx] = self.word_to_index.get(new_word, -1)
        
        # Add special projections for |1, |2, and |22
        special_projections = {"1": [], "2": [], "22": []}
        
        for word, idx in self.word_to_index.items():
            for proj in special_projections:
                if word.startswith(proj):
                    special_projections[proj].append((idx, self.word_to_index.get(word[len(proj):], -1)))
        
        projections.update(special_projections)
        
        return projections
    
    def _precompute_shuffle_product(self):
        """Precompute shuffle product table"""
        shuffle_table = {}
        
        for word1, idx1 in self.word_to_index.items():
            for word2, idx2 in self.word_to_index.items():
                # Only compute if both words are short enough
                if len(word1) + len(word2) <= self.max_level:
                    result = self._shuffle_product(word1, word2)
                    shuffle_table[(idx1, idx2)] = result
        
        return shuffle_table
    
    def _shuffle_product(self, word1, word2):
        """Compute shuffle product of two words"""
        if word1 == "":
            return {self.word_to_index[word2]: 1}
        if word2 == "":
            return {self.word_to_index[word1]: 1}
        
        result = {}
        
        # word1 = first_letter1 + rest1
        first_letter1 = word1[0]
        rest1 = word1[1:]
        
        # Recursive call for (rest1 ⊔⊔ word2) ⊗ first_letter1
        for word, coef in self._shuffle_product(rest1, word2).items():
            new_word = self.index_to_word[word] + first_letter1
            if len(new_word) <= self.max_level:
                idx = self.word_to_index.get(new_word, -1)
                if idx != -1:
                    result[idx] = result.get(idx, 0) + coef
        
        # word2 = first_letter2 + rest2
        first_letter2 = word2[0]
        rest2 = word2[1:]
        
        # Recursive call for (word1 ⊔⊔ rest2) ⊗ first_letter2
        for word, coef in self._shuffle_product(word1, rest2).items():
            new_word = self.index_to_word[word] + first_letter2
            if len(new_word) <= self.max_level:
                idx = self.word_to_index.get(new_word, -1)
                if idx != -1:
                    result[idx] = result.get(idx, 0) + coef
        
        return result
    
    def compute_signature(self, path):
        """
        Compute truncated signature of a path
        
        Args:
            path: numpy array of shape (n_points, dimension)
            
        Returns:
            signature: numpy array of shape (truncation_size,)
        """
        n_points = len(path)
        if n_points <= 1:
            # Return trivial signature for empty or single-point path
            sig = np.zeros(self.truncation_size)
            sig[0] = 1.0
            return sig
        
        # Initialize signature with 1 for the empty word
        signature = np.zeros(self.truncation_size)
        signature[0] = 1.0
        
        # Compute level 1 terms (iterated integrals)
        for i in range(self.dimension):
            idx = self.word_to_index[str(i+1)]
            signature[idx] = path[-1, i] - path[0, i]
        
        # Compute higher-level terms using Chen's relation
        for level in range(2, self.max_level + 1):
            for word in self._generate_words(level):
                idx = self.word_to_index[word]
                
                # Split the word into first letter and rest
                first_letter = word[0]
                rest = word[1:]
                rest_idx = self.word_to_index.get(rest, -1)
                
                if rest_idx != -1:
                    # Use Chen's relation: I(ai) = ∫ I(a) dxi
                    for k in range(1, n_points):
                        increment = path[k, int(first_letter) - 1] - path[k-1, int(first_letter) - 1]
                        # Use trapezoidal rule for approximation
                        signature[idx] += signature[rest_idx] * increment
        
        return signature
    
    def time_extended_bm_signature(self, t, w):
        """
        Compute the time-extended signature of Brownian motion
        
        Args:
            t: time point
            w: value of the Brownian motion at time t
            
        Returns:
            signature: numpy array of shape (truncation_size,)
        """
        # Start with the constant 1 for empty word
        signature = np.zeros(self.truncation_size)
        signature[0] = 1.0
        
        # First level: time and BM value
        signature[self.word_to_index["1"]] = t
        signature[self.word_to_index["2"]] = w
        
        # Second level
        if self.max_level >= 2:
            signature[self.word_to_index["11"]] = t**2/2
            signature[self.word_to_index["12"]] = t*w/2
            signature[self.word_to_index["21"]] = t*w/2
            signature[self.word_to_index["22"]] = w**2/2
            
        # Third level (if needed)
        if self.max_level >= 3:
            signature[self.word_to_index["111"]] = t**3/6
            signature[self.word_to_index["112"]] = t**2*w/6
            signature[self.word_to_index["121"]] = t**2*w/6
            signature[self.word_to_index["211"]] = t**2*w/6
            signature[self.word_to_index["122"]] = t*w**2/6
            signature[self.word_to_index["212"]] = t*w**2/6
            signature[self.word_to_index["221"]] = t*w**2/6
            signature[self.word_to_index["222"]] = w**3/6
        
        # We would need to continue for higher levels
        # For a general implementation, we should use the shuffle product
        
        return signature
    
    def linear_combination(self, coefficients, signature):
        """
        Compute linear combination of signature elements
        
        Args:
            coefficients: numpy array of shape (truncation_size,)
            signature: numpy array of shape (truncation_size,)
            
        Returns:
            result: scalar
        """
        return np.sum(coefficients * signature)
    
    def shuffle_product_full(self, tensor1, tensor2):
        """
        Compute the shuffle product of two tensors
        
        Args:
            tensor1: numpy array of shape (truncation_size,)
            tensor2: numpy array of shape (truncation_size,)
            
        Returns:
            result: numpy array of shape (truncation_size,)
        """
        result = np.zeros(self.truncation_size)
        
        for (idx1, idx2), shuffle_result in self.shuffle_table.items():
            for idx_result, coef in shuffle_result.items():
                result[idx_result] += tensor1[idx1] * tensor2[idx2] * coef
        
        return result

#######################################
# PART 2: STOCHASTIC VOLATILITY MODELS
#######################################

class StochasticVolatilityModel:
    """Base class for stochastic volatility models"""
    
    def __init__(self, rho=-0.7):
        """
        Initialize stochastic volatility model
        
        Args:
            rho: correlation between stock and volatility
        """
        self.rho = rho
    
    def simulate_paths(self, S0=1.0, n_paths=1, n_steps=252, T=1.0):
        """
        Simulate paths of the stock price and volatility
        
        Args:
            S0: initial stock price
            n_paths: number of paths to simulate
            n_steps: number of time steps
            T: time horizon
            
        Returns:
            times: array of time points
            stock_paths: array of stock price paths
            vol_paths: array of volatility paths
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def characteristic_function(self, u, t, T):
        """
        Compute the characteristic function E[e^{iu log(S_T/S_t)}|F_t]
        
        Args:
            u: complex parameter
            t: current time
            T: maturity
            
        Returns:
            cf: characteristic function value
        """
        raise NotImplementedError("Subclasses must implement this method")

class HestonModel(StochasticVolatilityModel):
    """Heston stochastic volatility model"""
    
    def __init__(self, kappa=2.0, theta=0.0625, eta=0.7, v0=0.0625, rho=-0.7):
        """
        Initialize Heston model
        
        Args:
            kappa: mean reversion speed
            theta: long-term variance
            eta: volatility of variance
            v0: initial variance
            rho: correlation between stock and volatility
        """
        super().__init__(rho)
        self.kappa = kappa
        self.theta = theta
        self.eta = eta
        self.v0 = v0
    
    def simulate_paths(self, S0=1.0, n_paths=1, n_steps=252, T=1.0):
        """Simulate paths using Euler scheme with full truncation"""
        dt = T/n_steps
        times = np.linspace(0, T, n_steps+1)
        
        # Initialize arrays
        stock_paths = np.zeros((n_paths, n_steps+1))
        vol_paths = np.zeros((n_paths, n_steps+1))
        
        # Set initial values
        stock_paths[:, 0] = S0
        vol_paths[:, 0] = self.v0
        
        # Generate correlated Brownian motions
        dW1 = np.random.normal(0, np.sqrt(dt), (n_paths, n_steps))
        dW2 = self.rho * dW1 + np.sqrt(1 - self.rho**2) * np.random.normal(0, np.sqrt(dt), (n_paths, n_steps))
        
        # Simulation loop
        for i in range(n_steps):
            # Ensure variance is non-negative for drift calculation (full truncation)
            vol_plus = np.maximum(vol_paths[:, i], 0)
            
            # Update variance
            vol_paths[:, i+1] = vol_paths[:, i] + self.kappa * (self.theta - vol_plus) * dt + \
                               self.eta * np.sqrt(vol_plus) * dW1[:, i]
            vol_paths[:, i+1] = np.maximum(vol_paths[:, i+1], 0)  # Ensure non-negative
            
            # Update stock price
            stock_paths[:, i+1] = stock_paths[:, i] * np.exp(-0.5 * vol_plus * dt + 
                                                            np.sqrt(vol_plus) * dW2[:, i])
        
        return times, stock_paths, vol_paths
    
    def characteristic_function(self, u, t, T):
        """
        Compute the characteristic function for the Heston model
        Based on the "Little Heston Trap" paper formulation
        """
        tau = T - t
        
        # Helper values
        xi = self.kappa - self.rho * self.eta * 1j * u
        d = np.sqrt(xi**2 + self.eta**2 * (u**2 + 1j * u))
        g = (xi + d) / (xi - d)
        
        # Compute A and B
        exp_dt = np.exp(d * tau)
        B = (self.kappa - self.rho * self.eta * 1j * u - d) / (self.eta**2) * \
            (1 - np.exp(-d * tau)) / (1 - g * np.exp(-d * tau))
        
        A = self.kappa * self.theta / (self.eta**2) * \
            ((self.kappa - self.rho * self.eta * 1j * u - d) * tau - 
             2 * np.log((1 - g * np.exp(-d * tau)) / (1 - g)))
        
        # Return characteristic function
        return np.exp(A + B * self.v0)

class SteinSteinModel(StochasticVolatilityModel):
    """Stein-Stein stochastic volatility model"""
    
    def __init__(self, kappa=1.0, theta=0.25, eta=1.2, v0=0.25, rho=-0.5):
        """
        Initialize Stein-Stein model
        
        Args:
            kappa: mean reversion speed
            theta: long-term volatility
            eta: volatility of volatility
            v0: initial volatility
            rho: correlation between stock and volatility
        """
        super().__init__(rho)
        self.kappa = kappa
        self.theta = theta
        self.eta = eta
        self.v0 = v0
    
    def simulate_paths(self, S0=1.0, n_paths=1, n_steps=252, T=1.0):
        """Simulate paths using Euler scheme"""
        dt = T/n_steps
        times = np.linspace(0, T, n_steps+1)
        
        # Initialize arrays
        stock_paths = np.zeros((n_paths, n_steps+1))
        vol_paths = np.zeros((n_paths, n_steps+1))
        
        # Set initial values
        stock_paths[:, 0] = S0
        vol_paths[:, 0] = self.v0
        
        # Generate correlated Brownian motions
        dW1 = np.random.normal(0, np.sqrt(dt), (n_paths, n_steps))
        dW2 = self.rho * dW1 + np.sqrt(1 - self.rho**2) * np.random.normal(0, np.sqrt(dt), (n_paths, n_steps))
        
        # Simulation loop
        for i in range(n_steps):
            # Update volatility (can be negative in Stein-Stein)
            vol_paths[:, i+1] = vol_paths[:, i] + self.kappa * (self.theta - vol_paths[:, i]) * dt + \
                               self.eta * dW1[:, i]
            
            # Update stock price using absolute volatility
            stock_paths[:, i+1] = stock_paths[:, i] * np.exp(-0.5 * vol_paths[:, i]**2 * dt + 
                                                            np.abs(vol_paths[:, i]) * dW2[:, i])
        
        return times, stock_paths, vol_paths
    
    def characteristic_function(self, u, t, T):
        """Compute the characteristic function for the Stein-Stein model"""
        tau = T - t
        
        # Complex parameters
        a = self.kappa
        b = 1j * u * self.rho * self.eta
        c = -0.5 * (u**2 + 1j * u)
        
        # Compute A, B, and G
        exp_at = np.exp(-a * tau)
        G = self.theta + (self.v0 - self.theta) * exp_at
        
        B = c * (self.v0**2 * tau + 2 * self.theta * (self.v0 - self.theta) * (1 - exp_at) / a + 
                 self.theta**2 * tau + (self.v0 - self.theta)**2 * (1 - exp_at)**2 / (2 * a))
        
        A = b * (self.v0 - self.theta) * (1 - exp_at) / a + b * self.theta * tau
        
        # The self.eta^2 term captures the variance of the OU process
        C = self.eta**2 / (2 * a**2) * (tau + (2 * exp_at - exp_at**2 - 1) / a + 
                                        2 * (1 - exp_at) / a)
        D = c * self.eta**2 * C
        
        return np.exp(A + B + D)

class SignatureVolatilityModel(StochasticVolatilityModel):
    """Signature-based stochastic volatility model"""
    
    def __init__(self, sig_coeffs, max_level=3, rho=-0.7):
        """
        Initialize signature volatility model
        
        Args:
            sig_coeffs: coefficients for the signature expansion
            max_level: maximum level of the signature
            rho: correlation between stock and volatility
        """
        super().__init__(rho)
        self.max_level = max_level
        self.sig_coeffs = sig_coeffs
        self.signature = Signature(max_level=max_level, dimension=2)
        
        # For Riccati equation solution
        self.n_steps_riccati = 100
    
    def simulate_paths(self, S0=1.0, n_paths=1, n_steps=252, T=1.0):
        """Simulate paths using Euler scheme"""
        dt = T/n_steps
        times = np.linspace(0, T, n_steps+1)
        
        # Initialize arrays
        stock_paths = np.zeros((n_paths, n_steps+1))
        vol_paths = np.zeros((n_paths, n_steps+1))
        
        # Set initial values
        stock_paths[:, 0] = S0
        
        # Generate Brownian motions
        dW = np.random.normal(0, np.sqrt(dt), (n_paths, n_steps))
        dW_perp = np.random.normal(0, np.sqrt(dt), (n_paths, n_steps))
        
        for p in range(n_paths):
            # Cumulative Brownian paths
            W = np.zeros(n_steps+1)
            t = np.zeros(n_steps+1)
            
            # Create correlated noise for stock price
            dB = self.rho * dW[p] + np.sqrt(1 - self.rho**2) * dW_perp[p]
            
            for i in range(n_steps):
                t[i+1] = t[i] + dt
                W[i+1] = W[i] + dW[p, i]
                
                # Compute time-extended signature
                sig = self.signature.time_extended_bm_signature(t[i], W[i])
                
                # Compute volatility as linear combination of signature elements
                vol = self.signature.linear_combination(self.sig_coeffs, sig)
                vol_paths[p, i] = vol
                
                # Update stock price
                stock_paths[p, i+1] = stock_paths[p, i] * np.exp(-0.5 * vol**2 * dt + vol * dB[i])
                
            # Compute final volatility
            sig = self.signature.time_extended_bm_signature(t[-1], W[-1])
            vol_paths[p, -1] = self.signature.linear_combination(self.sig_coeffs, sig)
        
        return times, stock_paths, vol_paths
    
    def characteristic_function(self, u, t, T):
        """
        Compute the characteristic function using the Riccati equation
        This is a simplified implementation that follows the paper's approach
        """
        # Setup for solving Riccati equation
        tau = T - t
        dt = tau / self.n_steps_riccati
        times = np.linspace(0, tau, self.n_steps_riccati + 1)
        
        # Initialize ψ(t) - truncated at 2*max_level as per the paper's recommendation
        psi = np.zeros(self.signature.truncation_size)
        
        # Backward iteration to solve Riccati equation
        for i in range(self.n_steps_riccati, 0, -1):
            # Compute RHS of Riccati equation using Runge-Kutta 4th order
            k1 = self._riccati_rhs(psi, u)
            k2 = self._riccati_rhs(psi + dt/2 * k1, u)
            k3 = self._riccati_rhs(psi + dt/2 * k2, u)
            k4 = self._riccati_rhs(psi + dt * k3, u)
            
            # Update ψ
            psi = psi + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
        
        # Compute the characteristic function value
        # We need the time-extended signature at current time
        if t == 0:  # At t=0, the BM is at 0
            sig_t = self.signature.time_extended_bm_signature(0, 0)
        else:
            # In a real implementation, we would need the actual BM value
            # Here we just use a placeholder
            sig_t = self.signature.time_extended_bm_signature(t, 0)
        
        # Return the characteristic function
        return np.exp(self.signature.linear_combination(psi, sig_t))
    
    def _riccati_rhs(self, psi, u):
        """
        Compute the right-hand side of the Riccati equation
        This is a simplified version that captures the essence of equation (4.1)
        """
        # Project psi to different components
        psi_2 = np.zeros_like(psi)
        for idx, proj_idx in self.signature.projections["2"]:
            if proj_idx != -1:
                psi_2[proj_idx] = psi[idx]
        
        # Compute the shuffle product (psi_2 ⊔⊔ psi_2)
        psi_2_shuffle_2 = self.signature.shuffle_product_full(psi_2, psi_2)
        
        # Compute the shuffle product (sigma ⊔⊔ psi_2)
        sigma_shuffle_psi_2 = self.signature.shuffle_product_full(self.sig_coeffs, psi_2)
        
        # Compute the shuffle product (sigma ⊔⊔ sigma)
        sigma_shuffle_sigma = self.signature.shuffle_product_full(self.sig_coeffs, self.sig_coeffs)
        
        # Compute components of the Riccati equation
        term1 = 0.5 * psi_2_shuffle_2
        term2 = self.rho * u * sigma_shuffle_psi_2
        
        # Additional terms would be added here in a complete implementation
        
        # Return the negative of the RHS (since we're solving backwards)
        return -(term1 + term2)

#######################################
# PART 3: OPTION PRICING WITH FOURIER
#######################################

def fourier_option_price(model, S0, K, T, option_type='call', n_points=100):
    """
    Price options using Fourier transform
    
    Args:
        model: stochastic volatility model with characteristic_function method
        S0: initial stock price
        K: strike price
        T: time to maturity
        option_type: 'call' or 'put'
        n_points: number of integration points
        
    Returns:
        price: option price
    """
    # Damping factor for call options
    alpha = 1.5
    
    # Integration bounds
    a, b = 0.0001, 100
    
    # Integrand function for call option
    def integrand_call(u):
        v = u - 1j * alpha
        numerator = np.exp(-1j * v * np.log(K/S0)) * model.characteristic_function(v, 0, T)
        denominator = 1j * v * (1j * v + 1)
        return np.real(numerator / denominator)
    
    # Perform numerical integration
    price, _ = integrate.quad(integrand_call, a, b, limit=1000)
    price = S0 - np.exp(-alpha * np.log(K/S0)) * price / np.pi
    
    # Convert to put if needed
    if option_type.lower() == 'put':
        price = price - S0 + K  # Put-call parity
    
    return price

def black_scholes_price(S0, K, T, sigma, option_type='call'):
    """Black-Scholes option pricing formula"""
    d1 = (np.log(S0/K) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type.lower() == 'call':
        return S0 * stats.norm.cdf(d1) - K * stats.norm.cdf(d2)
    else:
        return K * stats.norm.cdf(-d2) - S0 * stats.norm.cdf(-d1)

def black_scholes_delta(S0, K, T, sigma, option_type='call'):
    """Black-Scholes delta"""
    d1 = (np.log(S0/K) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
    
    if option_type.lower() == 'call':
        return stats.norm.cdf(d1)
    else:
        return stats.norm.cdf(d1) - 1

def implied_volatility(price, S0, K, T, option_type='call', initial_guess=0.2, tol=1e-6, max_iter=100):
    """Calculate implied volatility using Newton-Raphson method"""
    sigma = initial_guess
    
    for i in range(max_iter):
        price_diff = black_scholes_price(S0, K, T, sigma, option_type) - price
        
        if abs(price_diff) < tol:
            return sigma
        
        # Compute vega
        d1 = (np.log(S0/K) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
        vega = S0 * np.sqrt(T) * stats.norm.pdf(d1)
        
        # Update sigma
        if vega < 1e-10:  # Avoid division by zero
            sigma = sigma * 0.9 if price_diff > 0 else sigma * 1.1
        else:
            sigma = sigma - price_diff / vega
            
        # Ensure sigma stays positive
        sigma = max(0.001, min(sigma, 2.0))
    
    # Return best estimate if not converged
    return sigma

def compute_implied_volatility_surface(model, S0=1.0, T_values=None, moneyness_values=None):
    """Compute implied volatility surface for a given model"""
    if T_values is None:
        T_values = [1/52, 1/12, 3/12, 6/12, 1.0]  # 1w, 1m, 3m, 6m, 1y
    
    if moneyness_values is None:
        moneyness_values = [0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2]
    
    # Initialize result grid
    iv_surface = np.zeros((len(T_values), len(moneyness_values)))
    
    # Compute implied volatilities
    for i, T in enumerate(T_values):
        for j, m in enumerate(moneyness_values):
            K = S0 / m  # Convert moneyness to strike
            
            # Price option using Fourier methods
            option_price = fourier_option_price(model, S0, K, T, 'call')
            
            # Calculate implied volatility
            iv = implied_volatility(option_price, S0, K, T, 'call')
            iv_surface[i, j] = iv
    
    return T_values, moneyness_values, iv_surface

#######################################
# PART 4: QUADRATIC HEDGING
#######################################

def quadratic_hedge_simulation(model, S0, K, T, option_type='put', n_paths=100, n_steps=252):
    """
    Simulate quadratic hedging performance
    
    Args:
        model: stochastic volatility model
        S0: initial stock price
        K: strike price
        T: time to maturity
        option_type: 'call' or 'put'
        n_paths: number of paths to simulate
        n_steps: number of time steps
        
    Returns:
        pnl: P&L of the hedging strategy
    """
    # Simulate paths
    times, stock_paths, vol_paths = model.simulate_paths(S0, n_paths, n_steps, T)
    dt = T/n_steps
    
    # Initialize arrays for hedging results
    pnl = np.zeros(n_paths)
    
    # Loop over each path
    for p in range(n_paths):
        # Initial option price
        if isinstance(model, SignatureVolatilityModel):
            price = fourier_option_price(model, S0, K, T, option_type)
        else:
            # For known models, we can use their characteristic function
            price = fourier_option_price(model, S0, K, T, option_type)
        
        # Initial portfolio value
        portfolio_value = price
        
        # Initialize position
        delta = 0
        stock_position = 0
        
        for i in range(n_steps):
            # Current stock price and time to maturity
            S_curr = stock_paths[p, i]
            T_curr = T - times[i]
            
            if T_curr <= 0.0001:  # Close to expiry
                break
                
            # Compute delta using model's characteristic function
            # In practice, we would need to implement a proper delta calculation
            # For simplicity, we'll use Black-Scholes delta with current volatility
            delta = black_scholes_delta(S_curr, K, T_curr, vol_paths[p, i], option_type)
            
            # Update stock position
            new_stock_position = delta * S_curr
            
            # P&L from change in stock position
            if i > 0:
                portfolio_value += stock_position * (S_curr / stock_paths[p, i-1] - 1)
            
            # Update stock position
            stock_position = new_stock_position
        
        # Final P&L
        final_S = stock_paths[p, -1]
        final_payoff = max(K - final_S, 0) if option_type == 'put' else max(final_S - K, 0)
        
        pnl[p] = portfolio_value + stock_position * (final_S / stock_paths[p, -2] - 1) - final_payoff
    
    return pnl

#######################################
# PART 5: TESTING AND DEMONSTRATION
#######################################

def test_signature_representations():
    """Test signature representations of common stochastic volatility models"""
    # Create signature object
    sig = Signature(max_level=4, dimension=2)
    
    # Test Ornstein-Uhlenbeck representation
    print("Testing Ornstein-Uhlenbeck representation...")
    
    # Parameters
    kappa = 1.0
    theta = 0.25
    eta = 1.2
    x0 = 0.25
    
    # Create coefficient vector for OU process as described in the paper
    ou_coeffs = np.zeros(sig.truncation_size)
    
    # Level 0: constant term
    ou_coeffs[sig.word_to_index[""]] = x0
    
    # Level 1: linear terms
    ou_coeffs[sig.word_to_index["1"]] = -kappa * (x0 - theta)
    ou_coeffs[sig.word_to_index["2"]] = eta
    
    # Level 2: quadratic terms
    ou_coeffs[sig.word_to_index["11"]] = kappa**2 * (x0 - theta) / 2
    ou_coeffs[sig.word_to_index["21"]] = -kappa * eta
    
    # Simulate actual OU process
    dt = 0.01
    T = 1.0
    n_steps = int(T/dt)
    times = np.linspace(0, T, n_steps+1)
    
    # Generate Brownian motion
    np.random.seed(42)
    dW = np.random.normal(0, np.sqrt(dt), n_steps)
    W = np.cumsum(np.insert(dW, 0, 0))
    
    # Simulate OU process exactly
    t_array = np.linspace(0, T, n_steps+1)
    exact_ou = theta + (x0 - theta) * np.exp(-kappa * t_array) + \
               eta * np.exp(-kappa * t_array) * np.convolve(np.exp(kappa * t_array), np.insert(dW, 0, 0), mode='same')
    
    # Simulate using signature representation
    sig_ou = np.zeros(n_steps+1)
    for i in range(n_steps+1):
        sig_t = sig.time_extended_bm_signature(times[i], W[i])
        sig_ou[i] = sig.linear_combination(ou_coeffs, sig_t)
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(times, exact_ou, 'b-', label='Exact OU')
    plt.plot(times, sig_ou, 'r--', label=f'Signature Representation (M={sig.max_level})')
    plt.title('Ornstein-Uhlenbeck Process: Exact vs Signature Representation')
    plt.xlabel('Time')
    plt.ylabel('Process Value')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('ou_representation.png')
    plt.close()
    
    print("Test completed. Results saved to 'ou_representation.png'")
    
    return ou_coeffs

def test_option_pricing():
    """Test option pricing with different models"""
    print("Testing option pricing with different models...")
    
    # Parameters
    S0 = 1.0
    K = 1.0
    T_values = [1/52, 1/12, 3/12, 6/12, 1.0]  # 1w, 1m, 3m, 6m, 1y
    moneyness_values = [0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2]
    
    # Create models
    heston = HestonModel(kappa=2.0, theta=0.0625, eta=0.7, v0=0.0625, rho=-0.7)
    stein_stein = SteinSteinModel(kappa=1.0, theta=0.25, eta=1.2, v0=0.25, rho=-0.5)
    
    # Create signature volatility model based on OU process
    sig = Signature(max_level=3, dimension=2)
    ou_coeffs = np.zeros(sig.truncation_size)
    
    # Level 0: constant term
    ou_coeffs[sig.word_to_index[""]] = 0.25
    
    # Level 1: linear terms
    ou_coeffs[sig.word_to_index["1"]] = -1.0 * (0.25 - 0.25)
    ou_coeffs[sig.word_to_index["2"]] = 1.2
    
    # Level 2: quadratic terms
    ou_coeffs[sig.word_to_index["11"]] = 1.0**2 * (0.25 - 0.25) / 2
    ou_coeffs[sig.word_to_index["21"]] = -1.0 * 1.2
    
    sig_vol = SignatureVolatilityModel(ou_coeffs, max_level=3, rho=-0.5)
    
    # Compute implied volatility surfaces
    _, _, heston_iv = compute_implied_volatility_surface(heston, S0, T_values, moneyness_values)
    _, _, stein_stein_iv = compute_implied_volatility_surface(stein_stein, S0, T_values, moneyness_values)
    
    # Plot results for comparison
    plt.figure(figsize=(15, 10))
    
    # Select maturities to plot
    maturities_to_plot = [0, 1, 2, 4]  # 1w, 1m, 6m, 1y
    
    for i, idx in enumerate(maturities_to_plot):
        plt.subplot(2, 2, i+1)
        
        plt.plot(moneyness_values, heston_iv[idx, :], 'b-o', label='Heston')
        plt.plot(moneyness_values, stein_stein_iv[idx, :], 'r-x', label='Stein-Stein')
        
        plt.title(f'Maturity: {T_values[idx]:.2f} years')
        plt.xlabel('Moneyness (S0/K)')
        plt.ylabel('Implied Volatility')
        plt.grid(True)
        plt.legend()
    
    plt.tight_layout()
    plt.savefig('implied_volatility_comparison.png')
    plt.close()
    
    print("Test completed. Results saved to 'implied_volatility_comparison.png'")
    
    # Now compare option prices directly
    strikes = [0.8, 0.9, 1.0, 1.1, 1.2]
    T = 0.5  # 6 months
    
    heston_prices = [fourier_option_price(heston, S0, K*S0, T, 'call') for K in strikes]
    stein_stein_prices = [fourier_option_price(stein_stein, S0, K*S0, T, 'call') for K in strikes]
    
    # Print price comparison
    print("\nOption Price Comparison (T=6m, Call):")
    print(f"{'Strike':<10}{'Heston':<15}{'Stein-Stein':<15}")
    print("-" * 40)
    for i, K in enumerate(strikes):
        print(f"{K:<10.2f}{heston_prices[i]:<15.6f}{stein_stein_prices[i]:<15.6f}")
    
    return heston, stein_stein, sig_vol

def test_hedging():
    """Test quadratic hedging strategies"""
    print("\nTesting quadratic hedging strategies...")
    
    # Parameters
    S0 = 1.0
    K = 1.0
    T = 0.5  # 6 months
    n_paths = 100
    n_steps = 126  # Approximately 126 trading days in 6 months
    
    # Create models
    heston = HestonModel(kappa=2.0, theta=0.0625, eta=0.7, v0=0.0625, rho=-0.7)
    stein_stein = SteinSteinModel(kappa=1.0, theta=0.25, eta=1.2, v0=0.25, rho=-0.5)
    
    # Run hedging simulations
    heston_pnl = quadratic_hedge_simulation(heston, S0, K, T, 'put', n_paths, n_steps)
    stein_stein_pnl = quadratic_hedge_simulation(stein_stein, S0, K, T, 'put', n_paths, n_steps)
    
    # Compute statistics
    heston_mean = np.mean(heston_pnl)
    heston_std = np.std(heston_pnl)
    stein_stein_mean = np.mean(stein_stein_pnl)
    stein_stein_std = np.std(stein_stein_pnl)
    
    # Plot histograms of P&L
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.hist(heston_pnl, bins=20, alpha=0.7)
    plt.axvline(heston_mean, color='r', linestyle='--', label=f'Mean: {heston_mean:.6f}')
    plt.title('Heston Model: Hedging P&L Distribution')
    plt.xlabel('P&L')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.hist(stein_stein_pnl, bins=20, alpha=0.7)
    plt.axvline(stein_stein_mean, color='r', linestyle='--', label=f'Mean: {stein_stein_mean:.6f}')
    plt.title('Stein-Stein Model: Hedging P&L Distribution')
    plt.xlabel('P&L')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('hedging_pnl_distribution.png')
    plt.close()
    
    # Print summary statistics
    print("\nHedging Performance Summary:")
    print(f"{'Model':<15}{'Mean P&L':<15}{'Std Dev':<15}")
    print("-" * 45)
    print(f"{'Heston':<15}{heston_mean:<15.6f}{heston_std:<15.6f}")
    print(f"{'Stein-Stein':<15}{stein_stein_mean:<15.6f}{stein_stein_std:<15.6f}")
    
    print("\nTest completed. Results saved to 'hedging_pnl_distribution.png'")

def main():
    """Main function to run all tests"""
    print("Running tests for Signature Volatility Models paper...")
    
    # Test signature representations
    ou_coeffs = test_signature_representations()
    
    # Test option pricing
    heston, stein_stein, sig_vol = test_option_pricing()
    
    # Test hedging strategies
    test_hedging()
    
    print("\nAll tests completed.")

if __name__ == "__main__":
    main()