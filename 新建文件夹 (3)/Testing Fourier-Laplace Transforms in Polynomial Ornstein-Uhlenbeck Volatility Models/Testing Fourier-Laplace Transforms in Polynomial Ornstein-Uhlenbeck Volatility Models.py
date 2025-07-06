import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.optimize import minimize
from numba import njit
import time
from tqdm import tqdm

# Set random seed for reproducibility
np.random.seed(42)

class PolynomialOUModel:
    """Base class for polynomial OU volatility models"""
    
    def __init__(self, a=0, b=0, c=1, rho=0, var_curve=None):
        """
        Initialize the model with parameters a, b, c, rho and the forward variance curve.
        
        Parameters:
        - a, b, c: Parameters for the OU process dX_t = (a + b*X_t)dt + c*dW_t
        - rho: Correlation between stock price and volatility
        - var_curve: Forward variance curve function mapping time to variance
        """
        self.a = a
        self.b = b
        self.c = c
        self.rho = rho
        
        # Default variance curve if none provided
        if var_curve is None:
            self.var_curve = lambda t: 0.025
        else:
            self.var_curve = var_curve
    
    def power_series_coeffs(self, x):
        """
        Return coefficients of the power series p(x) = sum_k p_k * x^k
        This should be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def compute_convolution(self, p, q, max_k):
        """
        Compute the convolution (p * q)_k = sum_{l=0}^k p_l * q_{k-l} for k = 0, 1, ..., max_k
        """
        result = np.zeros(max_k + 1)
        for k in range(max_k + 1):
            for l in range(k + 1):
                if l < len(p) and (k - l) < len(q):
                    result[k] += p[l] * q[k - l]
        return result

class QuinticOUModel(PolynomialOUModel):
    """Quintic OU model with p(x) = p0 + p1*x + p3*x^3 + p5*x^5"""
    
    def __init__(self, p0, p1, p3, p5, alpha, kappa, rho, var_curve=None):
        """
        Initialize the Quintic OU model with specific parameters.
        
        Parameters:
        - p0, p1, p3, p5: Coefficients of the power series
        - alpha, kappa: Parameters for mean-reversion and time scale
        - rho: Correlation between stock price and volatility
        - var_curve: Forward variance curve function
        """
        # For the Quintic OU model, we have:
        # a = 0, b = alpha/kappa, c = kappa/alpha
        super().__init__(a=0, b=alpha/kappa, c=kappa/alpha, rho=rho, var_curve=var_curve)
        
        self.p0 = p0
        self.p1 = p1
        self.p3 = p3
        self.p5 = p5
        self.alpha = alpha
        self.kappa = kappa
    
    def power_series_coeffs(self, max_k):
        """Return coefficients of the power series up to max_k"""
        p = np.zeros(max_k + 1)
        p[0] = self.p0
        
        if max_k >= 1:
            p[1] = self.p1
        
        if max_k >= 3:
            p[3] = self.p3
        
        if max_k >= 5:
            p[5] = self.p5
        
        return p

class BergomiModel(PolynomialOUModel):
    """One-factor Bergomi model with p(x) = exp(nu*x - nu^2/4 * E[x^2])"""
    
    def __init__(self, nu, alpha, kappa, rho, var_curve=None, n_terms=8):
        """
        Initialize the Bergomi model with specific parameters.
        
        Parameters:
        - nu: Volatility of volatility parameter
        - alpha, kappa: Parameters for mean-reversion and time scale
        - rho: Correlation between stock price and volatility
        - var_curve: Forward variance curve function
        - n_terms: Number of terms to use in the Taylor expansion of exp
        """
        # For the Bergomi model, we have:
        # a = 0, b = alpha/kappa, c = kappa/alpha
        super().__init__(a=0, b=alpha/kappa, c=kappa/alpha, rho=rho, var_curve=var_curve)
        
        self.nu = nu
        self.alpha = alpha
        self.kappa = kappa
        self.n_terms = n_terms
    
    def power_series_coeffs(self, max_k):
        """Return coefficients of the Taylor expansion of exp(nu*x - nu^2/4 * E[x^2]) up to max_k"""
        p = np.zeros(max_k + 1)
        
        for k in range(min(max_k + 1, self.n_terms + 1)):
            p[k] = (self.nu**k) / (2**k * np.math.factorial(k))
        
        return p

class RiccatiSolver:
    """Class for solving the infinite-dimensional Riccati ODE system"""
    
    def __init__(self, model, max_k=32, k_max_cap=15):
        """
        Initialize the Riccati solver.
        
        Parameters:
        - model: The stochastic volatility model
        - max_k: Maximum index for truncation of the infinite ODE system
        - k_max_cap: Cap for (k+1)(k+2) to ensure numerical stability
        """
        self.model = model
        self.max_k = max_k
        self.k_max_cap = k_max_cap
    
    def solve_riccati(self, g1, g2, T, n_steps=100):
        """
        Solve the Riccati ODE system numerically.
        
        Parameters:
        - g1, g2: Functions for the characteristic functional
        - T: Time horizon
        - n_steps: Number of time steps for discretization
        
        Returns:
        - phi: Array of coefficients phi_k(t) for t = 0, ..., T
        """
        # Initialize arrays
        dt = T / n_steps
        phi = np.zeros((n_steps + 1, self.max_k + 1), dtype=np.complex128)
        
        # Terminal condition: phi_k(T) = 0
        phi[-1, :] = 0
        
        # Get power series coefficients
        p = self.model.power_series_coeffs(self.max_k)
        
        # Compute convolution p * p
        p_conv_p = self.model.compute_convolution(p, p, self.max_k)
        
        # Create diagonal matrix A
        A = np.diag([self.model.b * k for k in range(self.max_k + 1)])
        
        # Create upper triangular matrix Q
        Q = np.zeros((self.max_k + 1, self.max_k + 1))
        for k in range(self.max_k):
            Q[k, k+1] = self.model.a * (k+1)
            if k+2 <= self.max_k:
                # Cap (k+1)(k+2) for numerical stability
                factor = min((k+1)*(k+2)/2, self.k_max_cap**2/2)
                Q[k, k+2] = self.model.c**2 * factor
        
        # Time-stepping (backward in time)
        for i in range(n_steps, 0, -1):
            t = i * dt
            
            # Matrix exponential term
            exp_A_dt = np.diag([np.exp(self.model.b * k * dt) for k in range(self.max_k + 1)])
            
            # Compute G_dt matrix
            G_dt = np.zeros((self.max_k + 1, self.max_k + 1))
            for k in range(self.max_k + 1):
                if self.model.b * k == 0:
                    G_dt[k, k] = dt
                else:
                    G_dt[k, k] = (np.exp(self.model.b * k * dt) - 1) / (self.model.b * k)
            
            # Source term
            P_t = np.zeros(self.max_k + 1, dtype=np.complex128)
            for k in range(len(p_conv_p)):
                if k < len(p_conv_p) and p_conv_p[k] != 0:
                    P_t[k] = (g2(t) + g1(t)**2 * (g1(t) - 1)) * self.model.var_curve(T - t) * p_conv_p[k]
            
            # Compute phi_e (derivative term) for the current step
            phi_e = np.zeros(self.max_k + 1, dtype=np.complex128)
            for k in range(self.max_k):
                phi_e[k] = (k + 1) * phi[i, k + 1]
            
            # Compute convolution phi_e * phi_e
            phi_e_conv_phi_e = np.zeros(self.max_k + 1, dtype=np.complex128)
            for k in range(self.max_k + 1):
                for l in range(k + 1):
                    if l < len(phi_e) and (k - l) < len(phi_e):
                        phi_e_conv_phi_e[k] += phi_e[l] * phi_e[k - l]
            
            # Compute R_i matrix for phi_e * phi_e term
            R_i = np.zeros((self.max_k + 1, self.max_k + 1), dtype=np.complex128)
            for j in range(self.max_k + 1):
                for k in range(self.max_k + 1):
                    if j + k <= self.max_k:
                        if j + 1 < len(phi) and k < len(phi[i]):
                            R_i[j, k] = (j + 1) * phi[i, j + 1] * k
            
            # Compute N matrix for the rho term
            N = np.zeros((self.max_k + 1, self.max_k + 1))
            for j in range(self.max_k + 1):
                for k in range(self.max_k + 1):
                    if j + k <= self.max_k and k > 0:
                        if j < len(p) and k - 1 < len(p):
                            N[j, k] = k * p[j] * p[k - 1]
            
            # Create J matrix
            J = np.eye(self.max_k + 1) - G_dt @ (self.model.c**2/2 * R_i + g1(t) * self.model.rho * self.model.c * self.model.var_curve(T - t) * N + Q)
            
            # Solve for phi at the previous time step
            phi[i-1] = np.linalg.solve(J, exp_A_dt @ phi[i] + G_dt @ P_t)
        
        return phi

class FourierPricer:
    """Class for pricing options using Fourier techniques"""
    
    def __init__(self, model, solver):
        """
        Initialize the Fourier pricer.
        
        Parameters:
        - model: The stochastic volatility model
        - solver: The Riccati solver for computing the characteristic function
        """
        self.model = model
        self.solver = solver
    
    def characteristic_function(self, u, t, T, x=0):
        """
        Compute the characteristic function for log(S_T/S_t).
        
        Parameters:
        - u: Fourier variable
        - t: Current time
        - T: Maturity
        - x: Initial value of the OU process
        
        Returns:
        - Value of the characteristic function
        """
        # Define g1 and g2 for the characteristic function
        g1 = lambda s: 1j * u
        g2 = lambda s: 0
        
        # Solve the Riccati ODE system
        phi = self.solver.solve_riccati(g1, g2, T - t)
        
        # Compute the characteristic function as exp(sum_k phi_k(0) * x^k)
        result = 0
        for k in range(len(phi[0])):
            result += phi[0, k] * x**k
        
        return np.exp(result)
    
    def price_european_option(self, S0, K, T, is_call=True, control_variate=None):
        """
        Price a European option using Fourier techniques.
        
        Parameters:
        - S0: Initial stock price
        - K: Strike price
        - T: Time to maturity
        - is_call: True for call option, False for put
        - control_variate: Control variate model for variance reduction
        
        Returns:
        - Option price
        """
        # Log-moneyness
        k = np.log(K / S0)
        
        # Lewis formula for European call option
        def integrand(u):
            return np.real(np.exp(-1j * u * k) * self.characteristic_function(u - 0.5j, 0, T) / (u**2 + 0.25))
        
        # Integration using Gauss-Laguerre quadrature
        price, _ = quad(integrand, 0, np.inf, limit=100)
        price = S0 - K * np.exp(-k/2) * price / np.pi
        
        if control_variate is not None:
            # Add control variate for variance reduction
            cv_price = control_variate(S0, K, T, is_call)
            cv_integrand = lambda u: np.real(np.exp(-1j * u * k) * control_variate.characteristic_function(u - 0.5j, 0, T) / (u**2 + 0.25))
            cv_approx, _ = quad(cv_integrand, 0, np.inf, limit=100)
            cv_approx = S0 - K * np.exp(-k/2) * cv_approx / np.pi
            price = cv_price + (price - cv_approx)
        
        # Put-call parity for put option
        if not is_call:
            price = price - S0 + K
        
        return price
    
    def price_volatility_swap(self, T, q=0.5):
        """
        Price a q-volatility swap using Laplace inversion.
        
        Parameters:
        - T: Maturity of the swap
        - q: Power of the volatility (q=0.5 for standard volatility swap)
        
        Returns:
        - Fair strike of the volatility swap
        """
        # For q = 0.5 (standard volatility swap)
        def integrand(u):
            # Define g1 and g2 for the Laplace transform of the integrated variance
            g1 = lambda s: 0
            g2 = lambda s: -u / T
            
            # Solve the Riccati ODE system
            phi = self.solver.solve_riccati(g1, g2, T)
            
            # Compute the Laplace transform as exp(sum_k phi_k(0) * x^k)
            lt = np.exp(phi[0, 0])
            
            return (1 - lt) / u**(1 + q)
        
        # Compute the fair strike using numerical integration
        strike = 1 / (np.pi * T**q) * quad(integrand, 0, np.inf, limit=100)[0]
        
        # Multiply by gamma function and constants
        strike *= np.math.gamma(q + 0.5) / np.math.gamma(0.5)
        
        return strike

def generate_simulated_data(model, S0, T, n_paths=10000, n_steps=1000):
    """
    Generate simulated paths for the stock price and volatility process.
    
    Parameters:
    - model: Stochastic volatility model
    - S0: Initial stock price
    - T: Time horizon
    - n_paths: Number of paths to simulate
    - n_steps: Number of time steps
    
    Returns:
    - Dictionary with simulated paths and various statistics
    """
    dt = T / n_steps
    sqrt_dt = np.sqrt(dt)
    
    # Initialize arrays
    X = np.zeros((n_paths, n_steps + 1))
    S = np.zeros((n_paths, n_steps + 1))
    sigma = np.zeros((n_paths, n_steps + 1))
    integrated_var = np.zeros((n_paths, n_steps + 1))
    
    # Initial values
    X[:, 0] = 0
    S[:, 0] = S0
    
    # Get power series coefficients for the volatility
    p_coeffs = model.power_series_coeffs(10)  # Use enough terms for accuracy
    
    # Compute initial volatility
    for i in range(len(p_coeffs)):
        sigma[:, 0] += p_coeffs[i] * X[:, 0]**i
    sigma[:, 0] = np.sqrt(model.var_curve(0)) * sigma[:, 0]
    
    # Simulate paths
    for i in range(n_steps):
        t = i * dt
        
        # Generate correlated Brownian increments
        dW1 = np.random.normal(0, sqrt_dt, n_paths)
        dW2 = model.rho * dW1 + np.sqrt(1 - model.rho**2) * np.random.normal(0, sqrt_dt, n_paths)
        
        # Update X process (Ornstein-Uhlenbeck)
        X[:, i+1] = X[:, i] + (model.a + model.b * X[:, i]) * dt + model.c * dW2
        
        # Compute volatility from the polynomial function
        sigma[:, i+1] = np.zeros(n_paths)
        for j in range(len(p_coeffs)):
            sigma[:, i+1] += p_coeffs[j] * X[:, i+1]**j
        sigma[:, i+1] = np.sqrt(model.var_curve(t + dt)) * sigma[:, i+1]
        
        # Update stock price (log-normal)
        S[:, i+1] = S[:, i] * np.exp(-0.5 * sigma[:, i]**2 * dt + sigma[:, i] * dW1)
        
        # Update integrated variance
        integrated_var[:, i+1] = integrated_var[:, i] + sigma[:, i]**2 * dt
    
    # Compute statistics
    mean_S = np.mean(S, axis=0)
    std_S = np.std(S, axis=0)
    mean_sigma = np.mean(sigma, axis=0)
    mean_integrated_var = np.mean(integrated_var, axis=0)
    
    # Compute option prices via Monte Carlo
    def mc_option_price(K, is_call=True):
        payoffs = np.maximum(S[:, -1] - K, 0) if is_call else np.maximum(K - S[:, -1], 0)
        return np.mean(payoffs)
    
    # Compute volatility swap values via Monte Carlo
    mc_vol_swap = np.mean(np.sqrt(integrated_var[:, -1] / T))
    
    return {
        'S': S,
        'X': X,
        'sigma': sigma,
        'integrated_var': integrated_var,
        'mean_S': mean_S,
        'std_S': std_S,
        'mean_sigma': mean_sigma,
        'mean_integrated_var': mean_integrated_var,
        'mc_option_price': mc_option_price,
        'mc_vol_swap': mc_vol_swap
    }

def plot_implied_volatility(model, pricer, S0, T, strike_range, n_strikes=11, mc_data=None):
    """
    Plot implied volatility smile.
    
    Parameters:
    - model: Stochastic volatility model
    - pricer: Fourier pricer
    - S0: Initial stock price
    - T: Time to maturity
    - strike_range: Range of strikes as percentage of S0 (e.g., [0.8, 1.2])
    - n_strikes: Number of strikes to evaluate
    - mc_data: Monte Carlo data for comparison (optional)
    """
    from scipy.stats import norm
    
    def bs_price(S, K, T, sigma, is_call=True):
        """Black-Scholes formula"""
        d1 = (np.log(S/K) + (sigma**2/2)*T) / (sigma*np.sqrt(T))
        d2 = d1 - sigma*np.sqrt(T)
        
        if is_call:
            return S * norm.cdf(d1) - K * norm.cdf(d2)
        else:
            return K * norm.cdf(-d2) - S * norm.cdf(-d1)
    
    def implied_vol(price, S, K, T, is_call=True):
        """Compute implied volatility using Newton-Raphson method"""
        # Initial guess
        sigma = 0.2
        
        for _ in range(100):
            price_diff = bs_price(S, K, T, sigma, is_call) - price
            
            if abs(price_diff) < 1e-8:
                return sigma
            
            # Vega approximation
            vega = S * np.sqrt(T) * norm.pdf((np.log(S/K) + (sigma**2/2)*T) / (sigma*np.sqrt(T)))
            
            # Newton-Raphson update
            sigma = sigma - price_diff / vega
            
            # Ensure sigma stays positive and in reasonable bounds
            sigma = max(0.001, min(2.0, sigma))
        
        return sigma
    
    # Generate strikes
    strikes = np.linspace(S0 * strike_range[0], S0 * strike_range[1], n_strikes)
    moneyness = strikes / S0
    
    # Compute option prices and implied volatilities
    call_prices = []
    put_prices = []
    implied_vols = []
    
    for K in strikes:
        call_price = pricer.price_european_option(S0, K, T, is_call=True)
        put_price = pricer.price_european_option(S0, K, T, is_call=False)
        
        call_prices.append(call_price)
        put_prices.append(put_price)
        
        # Use calls for K >= S0 and puts for K < S0 to avoid numerical issues
        if K >= S0:
            iv = implied_vol(call_price, S0, K, T, is_call=True)
        else:
            iv = implied_vol(put_price, S0, K, T, is_call=False)
        
        implied_vols.append(iv)
    
    # Plot
    plt.figure(figsize=(12, 6))
    
    # Plot implied volatility smile
    plt.subplot(1, 2, 1)
    plt.plot(moneyness, implied_vols, 'o-', label='Fourier')
    
    # Add Monte Carlo results if available
    if mc_data is not None:
        mc_implied_vols = []
        for K in strikes:
            # Use Monte Carlo option prices
            if K >= S0:
                mc_price = mc_data['mc_option_price'](K, is_call=True)
                mc_iv = implied_vol(mc_price, S0, K, T, is_call=True)
            else:
                mc_price = mc_data['mc_option_price'](K, is_call=False)
                mc_iv = implied_vol(mc_price, S0, K, T, is_call=False)
            
            mc_implied_vols.append(mc_iv)
        
        plt.plot(moneyness, mc_implied_vols, 'x--', label='Monte Carlo')
    
    plt.xlabel('Moneyness (K/S0)')
    plt.ylabel('Implied Volatility')
    plt.title(f'Implied Volatility Smile (T = {T})')
    plt.grid(True)
    plt.legend()
    
    # Plot option prices
    plt.subplot(1, 2, 2)
    plt.plot(moneyness, call_prices, 'o-', label='Call (Fourier)')
    plt.plot(moneyness, put_prices, 's-', label='Put (Fourier)')
    
    # Add Monte Carlo results if available
    if mc_data is not None:
        mc_call_prices = [mc_data['mc_option_price'](K, is_call=True) for K in strikes]
        mc_put_prices = [mc_data['mc_option_price'](K, is_call=False) for K in strikes]
        
        plt.plot(moneyness, mc_call_prices, 'x--', label='Call (MC)')
        plt.plot(moneyness, mc_put_prices, '+--', label='Put (MC)')
    
    plt.xlabel('Moneyness (K/S0)')
    plt.ylabel('Option Price')
    plt.title(f'Option Prices (T = {T})')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def plot_vol_swaps(model, pricer, max_T=2, n_points=20, mc_data=None):
    """
    Plot volatility swap rates for different maturities.
    
    Parameters:
    - model: Stochastic volatility model
    - pricer: Fourier pricer
    - max_T: Maximum maturity
    - n_points: Number of maturity points
    - mc_data: Monte Carlo data for comparison (optional)
    """
    # Generate maturities
    maturities = np.linspace(0.1, max_T, n_points)
    
    # Compute volatility swap rates
    vol_swap_rates = [pricer.price_volatility_swap(T) for T in maturities]
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(maturities, vol_swap_rates, 'o-', label='Laplace Inversion')
    
    # Add Monte Carlo results if available
    if mc_data is not None and hasattr(mc_data, 'mc_vol_swap'):
        plt.axhline(y=mc_data['mc_vol_swap'], linestyle='--', color='red', label='Monte Carlo')
    
    plt.xlabel('Maturity (T)')
    plt.ylabel('Volatility Swap Rate')
    plt.title('Volatility Swap Rates')
    plt.grid(True)
    plt.legend()
    plt.show()

def calibrate_model(model_class, market_data, initial_params, bounds, S0, maturities, strikes):
    """
    Calibrate a model to market data.
    
    Parameters:
    - model_class: Class of the model to calibrate
    - market_data: Dictionary with market implied volatilities
    - initial_params: Initial parameters for the optimization
    - bounds: Bounds for the parameters
    - S0: Initial stock price
    - maturities: List of option maturities
    - strikes: List of option strikes for each maturity
    
    Returns:
    - Calibrated model parameters
    - Optimal objective function value
    """
    def objective(params):
        """Objective function for calibration (sum of squared errors)"""
        model = model_class(*params)
        solver = RiccatiSolver(model)
        pricer = FourierPricer(model, solver)
        
        total_error = 0
        
        for t_idx, T in enumerate(maturities):
            for k_idx, K in enumerate(strikes[t_idx]):
                # Compute model implied volatility
                price = pricer.price_european_option(S0, K, T)
                model_iv = implied_volatility(price, S0, K, T)
                
                # Compute error
                market_iv = market_data[t_idx][k_idx]
                error = (model_iv - market_iv)**2
                
                total_error += error
        
        return total_error
    
    # Perform optimization
    result = minimize(objective, initial_params, bounds=bounds, method='L-BFGS-B')
    
    return result.x, result.fun

def main():
    # Define forward variance curve
    var_curve = lambda t: 0.025 * np.exp(-5*t) + 0.06 * (1 - np.exp(-5*t))
    
    # Test 1: Quintic OU Model
    print("Testing Quintic OU Model...")
    
    # Model parameters for Quintic OU (from the paper)
    p0 = 0.0202
    p1 = 1.3332
    p3 = 0.0578
    p5 = 0.0071
    alpha = -0.6821
    kappa = 1/52
    rho = -0.6763
    
    # Create model, solver, and pricer
    quintic_model = QuinticOUModel(p0, p1, p3, p5, alpha, kappa, rho, var_curve)
    quintic_solver = RiccatiSolver(quintic_model)
    quintic_pricer = FourierPricer(quintic_model, quintic_solver)
    
    # Test 2: One-factor Bergomi Model
    print("Testing One-factor Bergomi Model...")
    
    # Model parameters for Bergomi (from the paper)
    nu = 1.1416
    alpha = -0.7377
    kappa = 1/52
    rho = -0.6744
    
    # Create model, solver, and pricer
    bergomi_model = BergomiModel(nu, alpha, kappa, rho, var_curve)
    bergomi_solver = RiccatiSolver(bergomi_model)
    bergomi_pricer = FourierPricer(bergomi_model, bergomi_solver)
    
    # Generate simulated data for validation
    print("Generating simulated data...")
    S0 = 100
    T = 0.25  # 3 months
    
    quintic_mc_data = generate_simulated_data(quintic_model, S0, T, n_paths=10000, n_steps=1000)
    bergomi_mc_data = generate_simulated_data(bergomi_model, S0, T, n_paths=10000, n_steps=1000)
    
    # Test Fourier pricing for European options
    print("Testing Fourier pricing for European options...")
    
    # Quintic OU model
    print("Quintic OU Model - Implied Volatility Smile")
    plot_implied_volatility(quintic_model, quintic_pricer, S0, T, [0.8, 1.2], n_strikes=11, mc_data=quintic_mc_data)
    
    # Bergomi model
    print("Bergomi Model - Implied Volatility Smile")
    plot_implied_volatility(bergomi_model, bergomi_pricer, S0, T, [0.8, 1.2], n_strikes=11, mc_data=bergomi_mc_data)
    
    # Test Laplace inversion for volatility swaps
    print("Testing Laplace inversion for volatility swaps...")
    
    # Quintic OU model
    print("Quintic OU Model - Volatility Swap Rates")
    plot_vol_swaps(quintic_model, quintic_pricer, max_T=2, n_points=20, mc_data=quintic_mc_data)
    
    # Bergomi model
    print("Bergomi Model - Volatility Swap Rates")
    plot_vol_swaps(bergomi_model, bergomi_pricer, max_T=2, n_points=20, mc_data=bergomi_mc_data)
    
    print("All tests completed.")

if __name__ == "__main__":
    main()
    