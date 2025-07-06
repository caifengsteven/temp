import numpy as np
import matplotlib.pyplot as plt
import time
import scipy.stats as stats
from scipy.interpolate import CubicSpline
from scipy.special import gamma, gammainc, factorial, kv
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

#####################################
# Characteristic Functions of Processes
#####################################

def OU_TS_LCF(u, t, alpha, beta, c, b, gamma_c=0):
    """
    Log-characteristic function of OU-TS process
    
    Parameters:
    -----------
    u : complex
        Point where to evaluate the LCF
    t : float
        Time horizon
    alpha : float
        Stability parameter
    beta : float
        Rate parameter
    c : float
        Scale parameter
    b : float
        Mean-reversion parameter
    gamma_c : float, optional
        Drift parameter, by default 0
    
    Returns:
    --------
    complex
        Value of the LCF at point u
    """
    # Handle the case when alpha = 0 or 1 specially
    if alpha == 0:
        # Gamma process case
        result = 1j*u*(1 - np.exp(-b*t))/b * (gamma_c - c/beta)
        result -= c/b * np.log(beta**t / (beta - 1j*u*np.exp(-b*t)) * (beta - 1j*u)**(-t))
        return result
    elif alpha == 1:
        # Special case alpha = 1
        result = 1j*u*(1 - np.exp(-b*t))/b * (gamma_c + c)
        result += c*beta/b * (np.log(1 - 1j*u/beta) - np.log((beta - 1j*u*np.exp(-b*t))/beta))
        return result
    
    # General case
    result = 1j*u*(1 - np.exp(-b*t))/b * gamma_c
    result += c*gamma(-alpha)/b * (
        (beta**alpha - (beta - 1j*u)**alpha) + 
        (beta**alpha - (beta - 1j*u*np.exp(-b*t))**alpha)*np.exp(-alpha*b*t)
    )
    
    return result

def TS_OU_LCF(u, t, alpha, beta, c, b, gamma_c=0):
    """
    Log-characteristic function of TS-OU process
    
    Parameters:
    -----------
    (Same as OU_TS_LCF)
    
    Returns:
    --------
    complex
        Value of the LCF at point u
    """
    # Handle the case when alpha = 0 or 1 specially
    if alpha == 0:
        # Gamma-OU process
        result = 1j*u*(1 - np.exp(-b*t)) * (gamma_c - c/beta)
        result += c * np.log((beta - 1j*u*np.exp(-b*t))/(beta - 1j*u))
        return result
    elif alpha == 1:
        # Special case alpha = 1
        result = 1j*u*(1 - np.exp(-b*t)) * (gamma_c + c)
        result += c*beta * (
            (1 - 1j*u/beta) * np.log(1 - 1j*u/beta) - 
            (1 - 1j*u*np.exp(-b*t)/beta) * np.log(1 - 1j*u*np.exp(-b*t)/beta)
        )
        return result
    
    # General case
    result = 1j*u*(1 - np.exp(-b*t)) * (gamma_c - c*beta**(alpha-1)*gamma(1-alpha))
    result += c*gamma(-alpha) * (
        (beta - 1j*u)**alpha - (beta - 1j*u*np.exp(-b*t))**alpha
    )
    
    return result

def OU_NTS_LCF(u, t, alpha, kappa, sigma, b, theta=0):
    """
    Log-characteristic function of OU-NTS process
    
    Parameters:
    -----------
    u : complex
        Point where to evaluate the LCF
    t : float
        Time horizon
    alpha : float
        Stability parameter
    kappa : float
        Variance of the subordinator
    sigma : float
        Volatility
    b : float
        Mean-reversion parameter
    theta : float, optional
        Drift parameter, by default 0
    
    Returns:
    --------
    complex
        Value of the LCF at point u
    """
    # Handle the case when alpha = 0 specially (VG process)
    if alpha == 0:
        # Numerical integration for the general case
        z_grid = np.logspace(np.log10(np.exp(-b*t)), 0, 1000)
        dz = np.diff(np.concatenate([[0], z_grid]))
        
        integrand = np.log(0.5*sigma**2*u**2*z_grid**2*kappa - 1j*theta*kappa*u*z_grid + 1)
        result = -np.sum(integrand * dz / z_grid) / kappa / b
        
        return result
    
    # Numerical integration for the general case
    z_grid = np.logspace(np.log10(np.exp(-b*t)), 0, 1000)
    dz = np.diff(np.concatenate([[0], z_grid]))
    
    integrand = ((0.5*sigma**2*u**2*z_grid**2 - 1j*theta*u*z_grid + (1-alpha)/kappa)**alpha - 
                ((1-alpha)/kappa)**alpha)
    result = (1-alpha)/(kappa**alpha) * np.sum(integrand * dz / z_grid) / (alpha*b)
    
    return result

def NTS_OU_LCF(u, t, alpha, kappa, sigma, b, theta=0):
    """
    Log-characteristic function of NTS-OU process
    
    Parameters:
    -----------
    (Same as OU_NTS_LCF)
    
    Returns:
    --------
    complex
        Value of the LCF at point u
    """
    # Handle the case when alpha = 0 specially (VG process)
    if alpha == 0:
        # VG-OU process
        numerator = 0.5*sigma**2*u**2*kappa*np.exp(-2*b*t) - 1j*theta*kappa*u*np.exp(-b*t) + 1
        denominator = 0.5*sigma**2*u**2*kappa - 1j*theta*kappa*u + 1
        result = np.log(numerator/denominator) / kappa
        
        return result
    
    # General case
    term1 = (1-alpha)/(kappa**alpha) * (1 - ((1-alpha)/kappa)*(
        theta*u*np.exp(-b*t)*1j + 0.5*sigma**2*u**2*np.exp(-2*b*t)
    ))**alpha
    
    term2 = (1-alpha)/(kappa**alpha) * (1 - ((1-alpha)/kappa)*(
        theta*u*1j + 0.5*sigma**2*u**2
    ))**alpha
    
    result = term1 - term2
    
    return result

def analyticity_strip_TS(alpha, beta):
    """
    Determine analyticity strip for TS process
    
    Parameters:
    -----------
    alpha : float
        Stability parameter
    beta : float
        Rate parameter
    
    Returns:
    --------
    tuple
        (p_minus, p_plus) boundaries of the analyticity strip
    """
    return (-beta, float('inf'))

def analyticity_strip_NTS(alpha, kappa, sigma, theta):
    """
    Determine analyticity strip for NTS process
    
    Parameters:
    -----------
    alpha : float
        Stability parameter
    kappa : float
        Variance of the subordinator
    sigma : float
        Volatility
    theta : float
        Drift parameter
    
    Returns:
    --------
    tuple
        (p_minus, p_plus) boundaries of the analyticity strip
    """
    A = np.sqrt(theta**2 + 2*sigma**2*(1-alpha)/kappa)
    p_minus = (theta - A)/sigma**2
    p_plus = (theta + A)/sigma**2
    
    return (p_minus, p_plus)

#####################################
# Simulation Algorithms
#####################################

def FGMC(CF, t, strip, M=16, is_FA=False, lambda_t=None, CF_jumps=None):
    """
    Fast and General Monte Carlo algorithm for OU processes
    
    Parameters:
    -----------
    CF : function
        Characteristic function of the process
    t : float
        Time horizon
    strip : tuple
        (p_minus, p_plus) boundaries of the analyticity strip
    M : int, optional
        Log2 of the number of FFT points, by default 16
    is_FA : bool, optional
        Whether the process has Finite Activity, by default False
    lambda_t : float, optional
        Intensity of the Compound Poisson process, required if is_FA=True
    CF_jumps : function, optional
        Characteristic function of the jumps, required if is_FA=True
    
    Returns:
    --------
    function
        A function that generates random samples from the process
    """
    p_minus, p_plus = strip
    N = 2**M
    
    # Choose the optimal shift parameter a
    if p_minus < 0 and p_plus > 0:
        a = 0.5 * min(-p_minus, p_plus)
    elif p_minus >= 0:
        a = 0.5 * p_plus
    else:  # p_plus <= 0
        a = 0.5 * p_minus
    
    if a == 0:
        a = 0.1  # Avoid zero shift
    
    # For Finite Activity processes, use Algorithm 2
    if is_FA:
        if lambda_t is None or CF_jumps is None:
            raise ValueError("For Finite Activity processes, lambda_t and CF_jumps must be provided")
        
        def generator(n_samples=1):
            # Generate Bernoulli variables
            B = np.random.random(n_samples) < 1 - np.exp(-lambda_t)
            
            # For samples with no jumps, return 0
            samples = np.zeros(n_samples)
            
            # For samples with jumps, use Algorithm 1
            if np.any(B):
                # Generate uniform random variables
                U = np.random.random(np.sum(B))
                
                # Invert the CDF for these samples
                jumps = invert_cdf(CF_jumps, t, strip, U, M)
                
                # Assign the results
                samples[B] = jumps
            
            return samples
        
        return generator
    
    # For Infinite Activity processes, use Algorithm 1
    def generator(n_samples=1):
        # Generate uniform random variables
        U = np.random.random(n_samples)
        
        # Invert the CDF
        return invert_cdf(CF, t, strip, U, M, a)
    
    return generator

def invert_cdf(CF, t, strip, U, M=16, a=None):
    """
    Invert the CDF using FFT
    
    Parameters:
    -----------
    CF : function
        Characteristic function of the process
    t : float
        Time horizon
    strip : tuple
        (p_minus, p_plus) boundaries of the analyticity strip
    U : ndarray
        Uniform random variables
    M : int, optional
        Log2 of the number of FFT points, by default 16
    a : float, optional
        Shift parameter, by default None
    
    Returns:
    --------
    ndarray
        Random samples from the process
    """
    p_minus, p_plus = strip
    N = 2**M
    
    # Choose the optimal shift parameter if not provided
    if a is None:
        if p_minus < 0 and p_plus > 0:
            a = 0.5 * min(-p_minus, p_plus)
        elif p_minus >= 0:
            a = 0.5 * p_plus
        else:  # p_plus <= 0
            a = 0.5 * p_minus
        
        if a == 0:
            a = 0.1  # Avoid zero shift
    
    # Determine integration step size h
    h = 0.1  # This could be optimized based on the decay rate
    
    # Set up the grid for x values
    x_min = -5.0
    x_max = 5.0
    x_grid = np.linspace(x_min, x_max, N)
    
    # Compute the CDF using FFT
    Ra = 0 if a > 0 else 1
    
    # Compute the integrand at grid points
    u_grid = np.arange(0, N) * h
    u_grid_shifted = u_grid + 1j*a
    
    # Apply the Gil-Pelaez formula
    integrand = np.zeros(N, dtype=complex)
    for i in range(N):
        u = u_grid[i] + 1j*a
        if u != 0:
            phi = np.exp(CF(u, t))
            integrand[i] = np.exp(-1j * u_grid[i] * x_grid[0]) * phi / (1j * u)
    
    # Handle the point u = 0 specially
    if a != 0:
        integrand[0] = 0
    
    # Apply FFT to get the CDF values
    cdf_values = Ra - np.real(np.fft.fft(integrand)) * h / np.pi
    
    # Apply exp(a*x) correction
    cdf_values *= np.exp(a * x_grid)
    
    # Ensure the CDF is monotonically increasing and in [0, 1]
    cdf_values = np.maximum(0, np.minimum(1, cdf_values))
    cdf_values.sort()  # Ensure monotonicity
    
    # Create spline interpolation
    cdf_spline = CubicSpline(cdf_values, x_grid)
    
    # Invert the CDF to get samples
    samples = cdf_spline(U)
    
    return samples

def simulate_OU_process(X0, t_grid, generator, b):
    """
    Simulate a Lévy-driven OU process
    
    Parameters:
    -----------
    X0 : float
        Initial value
    t_grid : ndarray
        Time grid
    generator : function
        Function that generates random samples from the process
    b : float
        Mean-reversion parameter
    
    Returns:
    --------
    ndarray
        Simulated paths
    """
    n_steps = len(t_grid) - 1
    X = np.zeros(len(t_grid))
    X[0] = X0
    
    for i in range(n_steps):
        dt = t_grid[i+1] - t_grid[i]
        Z = generator(1)[0]
        X[i+1] = X[i] * np.exp(-b * dt) + Z
    
    return X

#####################################
# Cumulant Calculation Functions
#####################################

def cumulants_OU_TS(t, alpha, beta, c, b, gamma_c=0):
    """
    Calculate the first four cumulants of the OU-TS process
    
    Parameters:
    -----------
    t : float
        Time horizon
    alpha : float
        Stability parameter
    beta : float
        Rate parameter
    c : float
        Scale parameter
    b : float
        Mean-reversion parameter
    gamma_c : float, optional
        Drift parameter, by default 0
    
    Returns:
    --------
    list
        [c1, c2, c3, c4] first four cumulants
    """
    # First cumulant (mean)
    c1 = (1 - np.exp(-b*t)) / b * gamma_c
    
    # Higher cumulants for k >= 2
    c2 = c * beta**(alpha-2) * gamma(2-alpha) * (1 - np.exp(-2*b*t)) / (2*b)
    c3 = c * beta**(alpha-3) * gamma(3-alpha) * (1 - np.exp(-3*b*t)) / (3*b)
    c4 = c * beta**(alpha-4) * gamma(4-alpha) * (1 - np.exp(-4*b*t)) / (4*b)
    
    return [c1, c2, c3, c4]

def cumulants_TS_OU(t, alpha, beta, c, b, gamma_c=0):
    """
    Calculate the first four cumulants of the TS-OU process
    
    Parameters:
    -----------
    (Same as cumulants_OU_TS)
    
    Returns:
    --------
    list
        [c1, c2, c3, c4] first four cumulants
    """
    # Cumulants of the stationary distribution
    c1_stat = gamma_c - c * beta**(alpha-1) * gamma(1-alpha)
    c2_stat = c * beta**(alpha-2) * gamma(2-alpha)
    c3_stat = c * beta**(alpha-3) * gamma(3-alpha)
    c4_stat = c * beta**(alpha-4) * gamma(4-alpha)
    
    # Cumulants of the process at time t
    c1 = (1 - np.exp(-b*t)) * c1_stat
    c2 = (1 - np.exp(-2*b*t)) * c2_stat
    c3 = (1 - np.exp(-3*b*t)) * c3_stat
    c4 = (1 - np.exp(-4*b*t)) * c4_stat
    
    return [c1, c2, c3, c4]

def cumulants_OU_NTS(t, alpha, kappa, sigma, b, theta=0):
    """
    Calculate the first four cumulants of the OU-NTS process
    
    Parameters:
    -----------
    t : float
        Time horizon
    alpha : float
        Stability parameter
    kappa : float
        Variance of the subordinator
    sigma : float
        Volatility
    b : float
        Mean-reversion parameter
    theta : float, optional
        Drift parameter, by default 0
    
    Returns:
    --------
    list
        [c1, c2, c3, c4] first four cumulants
    """
    # Calculate cumulants of NTS process at t=1
    c1_nts = theta
    
    c2_nts = 0
    for n in range(0, 2):
        if 2-2*n >= 0:
            term = factorial(2) / (factorial(n) * factorial(2-2*n))
            term *= gamma(2-alpha-n) / gamma(1-alpha)
            term *= (kappa/(1-alpha))**(2-1-n)
            term *= theta**(2-2*n)
            term *= (sigma**2/2)**n
            c2_nts += term
    
    c3_nts = 0
    for n in range(0, 2):
        if 3-2*n >= 0:
            term = factorial(3) / (factorial(n) * factorial(3-2*n))
            term *= gamma(3-alpha-n) / gamma(1-alpha)
            term *= (kappa/(1-alpha))**(3-1-n)
            term *= theta**(3-2*n)
            term *= (sigma**2/2)**n
            c3_nts += term
    
    c4_nts = 0
    for n in range(0, 3):
        if 4-2*n >= 0:
            term = factorial(4) / (factorial(n) * factorial(4-2*n))
            term *= gamma(4-alpha-n) / gamma(1-alpha)
            term *= (kappa/(1-alpha))**(4-1-n)
            term *= theta**(4-2*n)
            term *= (sigma**2/2)**n
            c4_nts += term
    
    # Convert to OU-NTS cumulants
    c1 = (1 - np.exp(-b*t)) / b * c1_nts
    c2 = (1 - np.exp(-2*b*t)) / (2*b) * c2_nts
    c3 = (1 - np.exp(-3*b*t)) / (3*b) * c3_nts
    c4 = (1 - np.exp(-4*b*t)) / (4*b) * c4_nts
    
    return [c1, c2, c3, c4]

def cumulants_NTS_OU(t, alpha, kappa, sigma, b, theta=0):
    """
    Calculate the first four cumulants of the NTS-OU process
    
    Parameters:
    -----------
    (Same as cumulants_OU_NTS)
    
    Returns:
    --------
    list
        [c1, c2, c3, c4] first four cumulants
    """
    # Calculate cumulants of NTS distribution
    c1_stat = theta
    
    c2_stat = 0
    for n in range(0, 2):
        if 2-2*n >= 0:
            term = factorial(2) / (factorial(n) * factorial(2-2*n))
            term *= gamma(2-alpha-n) / gamma(1-alpha)
            term *= (kappa/(1-alpha))**(2-1-n)
            term *= theta**(2-2*n)
            term *= (sigma**2/2)**n
            c2_stat += term
    
    c3_stat = 0
    for n in range(0, 2):
        if 3-2*n >= 0:
            term = factorial(3) / (factorial(n) * factorial(3-2*n))
            term *= gamma(3-alpha-n) / gamma(1-alpha)
            term *= (kappa/(1-alpha))**(3-1-n)
            term *= theta**(3-2*n)
            term *= (sigma**2/2)**n
            c3_stat += term
    
    c4_stat = 0
    for n in range(0, 3):
        if 4-2*n >= 0:
            term = factorial(4) / (factorial(n) * factorial(4-2*n))
            term *= gamma(4-alpha-n) / gamma(1-alpha)
            term *= (kappa/(1-alpha))**(4-1-n)
            term *= theta**(4-2*n)
            term *= (sigma**2/2)**n
            c4_stat += term
    
    # Cumulants of the process at time t
    c1 = (1 - np.exp(-b*t)) * c1_stat
    c2 = (1 - np.exp(-2*b*t)) * c2_stat
    c3 = (1 - np.exp(-3*b*t)) * c3_stat
    c4 = (1 - np.exp(-4*b*t)) * c4_stat
    
    return [c1, c2, c3, c4]

#####################################
# Option Pricing Functions
#####################################

def price_european_call(S0, K, r, T, paths):
    """
    Price a European call option
    
    Parameters:
    -----------
    S0 : float
        Initial stock price
    K : float
        Strike price
    r : float
        Risk-free rate
    T : float
        Time to maturity
    paths : ndarray
        Simulated paths at maturity
    
    Returns:
    --------
    float
        Option price
    """
    payoffs = np.maximum(paths - K, 0)
    price = np.exp(-r * T) * np.mean(payoffs)
    return price

def price_asian_call(S0, K, r, T, paths):
    """
    Price an Asian call option
    
    Parameters:
    -----------
    S0 : float
        Initial stock price
    K : float
        Strike price
    r : float
        Risk-free rate
    T : float
        Time to maturity
    paths : ndarray
        Simulated paths (each row is a path)
    
    Returns:
    --------
    float
        Option price
    """
    avg_prices = np.mean(paths, axis=1)
    payoffs = np.maximum(avg_prices - K, 0)
    price = np.exp(-r * T) * np.mean(payoffs)
    return price

#####################################
# Main Experiments
#####################################

def test_accuracy_and_speed():
    """
    Test the accuracy and speed of the FGMC algorithm
    """
    print("Testing accuracy and speed of FGMC algorithm...")
    
    # Parameters for TS processes
    b = 0.1
    beta = 2.5
    c = 0.5
    t = 1.0
    n_samples = 1_000_000  # 10^6 samples
    
    # Test for various alpha values
    alpha_values = [1.6, 1.2, 0.8, 0.4, -1.0, -2.0]
    
    # Results storage
    results = []
    
    for alpha in alpha_values:
        print(f"\nTesting OU-TS with alpha = {alpha}")
        
        # Calculate true cumulants
        true_cumulants = cumulants_OU_TS(t, alpha, beta, c, b)
        
        # Define characteristic function
        def cf_ou_ts(u, t):
            return OU_TS_LCF(u, t, alpha, beta, c, b)
        
        # Get analyticity strip
        strip = analyticity_strip_TS(alpha, beta)
        
        # Determine if process has Finite Activity
        is_FA = alpha < 0
        
        # Set up generator
        if is_FA:
            # For Finite Activity processes, we need lambda_t and CF_jumps
            lambda_t = c * beta**alpha * gamma(-alpha)  # Intensity
            
            # CF of jumps conditional on at least one jump
            def cf_jumps(u, t):
                f = lambda_t * ((np.exp(cf_ou_ts(u, t)) - 1) / (np.exp(lambda_t) - 1))
                return np.log(f)
            
            start_time = time.time()
            generator = FGMC(cf_ou_ts, t, strip, M=16, is_FA=True, 
                             lambda_t=lambda_t, CF_jumps=cf_jumps)
        else:
            start_time = time.time()
            generator = FGMC(cf_ou_ts, t, strip, M=16)
        
        # Generate samples
        samples = generator(n_samples)
        end_time = time.time()
        
        # Compute empirical cumulants
        emp_cumulants = [
            np.mean(samples),  # First cumulant (mean)
            np.var(samples),   # Second cumulant (variance)
            stats.skew(samples) * np.var(samples)**(3/2),  # Third cumulant
            stats.kurtosis(samples, fisher=False) * np.var(samples)**2 - 3 * np.var(samples)**2  # Fourth cumulant
        ]
        
        # Compute accuracy metrics
        abs_errors = [abs(true - emp) for true, emp in zip(true_cumulants, emp_cumulants)]
        rel_errors = [abs_err / (abs(true) + 1e-10) for abs_err, true in zip(abs_errors, true_cumulants)]
        
        print(f"True cumulants:   {[f'{c:.6f}' for c in true_cumulants]}")
        print(f"FGMC cumulants:   {[f'{c:.6f}' for c in emp_cumulants]}")
        print(f"Absolute errors:  {[f'{e:.6f}' for e in abs_errors]}")
        print(f"Relative errors:  {[f'{e:.6f}' for e in rel_errors]}")
        print(f"Computation time: {end_time - start_time:.4f} seconds")
        
        results.append({
            'Process': 'OU-TS',
            'Alpha': alpha,
            'True Cumulants': true_cumulants,
            'FGMC Cumulants': emp_cumulants,
            'Absolute Errors': abs_errors,
            'Relative Errors': rel_errors,
            'Time': end_time - start_time
        })
    
    # Now test for TS-OU process
    for alpha in [a for a in alpha_values if a >= 0]:  # TS-OU only defined for alpha >= 0
        print(f"\nTesting TS-OU with alpha = {alpha}")
        
        # Calculate true cumulants
        true_cumulants = cumulants_TS_OU(t, alpha, beta, c, b)
        
        # Define characteristic function
        def cf_ts_ou(u, t):
            return TS_OU_LCF(u, t, alpha, beta, c, b)
        
        # Get analyticity strip
        strip = analyticity_strip_TS(alpha, beta)
        
        # Determine if process has Finite Activity
        is_FA = alpha == 0
        
        # Set up generator
        if is_FA:
            # For Finite Activity processes, we need lambda_t and CF_jumps
            lambda_t = c * b  # Intensity for alpha=0
            
            # CF of jumps conditional on at least one jump
            def cf_jumps(u, t):
                f = lambda_t * ((np.exp(cf_ts_ou(u, t)) - 1) / (np.exp(lambda_t) - 1))
                return np.log(f)
            
            start_time = time.time()
            generator = FGMC(cf_ts_ou, t, strip, M=16, is_FA=True, 
                             lambda_t=lambda_t, CF_jumps=cf_jumps)
        else:
            start_time = time.time()
            generator = FGMC(cf_ts_ou, t, strip, M=16)
        
        # Generate samples
        samples = generator(n_samples)
        end_time = time.time()
        
        # Compute empirical cumulants
        emp_cumulants = [
            np.mean(samples),  # First cumulant (mean)
            np.var(samples),   # Second cumulant (variance)
            stats.skew(samples) * np.var(samples)**(3/2),  # Third cumulant
            stats.kurtosis(samples, fisher=False) * np.var(samples)**2 - 3 * np.var(samples)**2  # Fourth cumulant
        ]
        
        # Compute accuracy metrics
        abs_errors = [abs(true - emp) for true, emp in zip(true_cumulants, emp_cumulants)]
        rel_errors = [abs_err / (abs(true) + 1e-10) for abs_err, true in zip(abs_errors, true_cumulants)]
        
        print(f"True cumulants:   {[f'{c:.6f}' for c in true_cumulants]}")
        print(f"FGMC cumulants:   {[f'{c:.6f}' for c in emp_cumulants]}")
        print(f"Absolute errors:  {[f'{e:.6f}' for e in abs_errors]}")
        print(f"Relative errors:  {[f'{e:.6f}' for e in rel_errors]}")
        print(f"Computation time: {end_time - start_time:.4f} seconds")
        
        results.append({
            'Process': 'TS-OU',
            'Alpha': alpha,
            'True Cumulants': true_cumulants,
            'FGMC Cumulants': emp_cumulants,
            'Absolute Errors': abs_errors,
            'Relative Errors': rel_errors,
            'Time': end_time - start_time
        })
    
    # Parameters for NTS processes
    b = 0.2162
    kappa = 0.256
    sigma = 0.201
    theta = 0
    
    # Test for various alpha values
    alpha_values = [0.8, 0.6, 0.4, 0.2, -1.0, -2.0]
    
    for alpha in alpha_values:
        print(f"\nTesting OU-NTS with alpha = {alpha}")
        
        # Calculate true cumulants
        true_cumulants = cumulants_OU_NTS(t, alpha, kappa, sigma, b, theta)
        
        # Define characteristic function
        def cf_ou_nts(u, t):
            return OU_NTS_LCF(u, t, alpha, kappa, sigma, b, theta)
        
        # Get analyticity strip
        strip = analyticity_strip_NTS(alpha, kappa, sigma, theta)
        
        # Determine if process has Finite Activity
        is_FA = alpha < 0
        
        # Set up generator
        if is_FA:
            # For Finite Activity processes, we need lambda_t and CF_jumps
            lambda_t = (1 - alpha) / (kappa**abs(alpha))  # Intensity
            
            # CF of jumps conditional on at least one jump
            def cf_jumps(u, t):
                f = lambda_t * ((np.exp(cf_ou_nts(u, t)) - 1) / (np.exp(lambda_t) - 1))
                return np.log(f)
            
            start_time = time.time()
            generator = FGMC(cf_ou_nts, t, strip, M=16, is_FA=True, 
                             lambda_t=lambda_t, CF_jumps=cf_jumps)
        else:
            start_time = time.time()
            generator = FGMC(cf_ou_nts, t, strip, M=16)
        
        # Generate samples
        samples = generator(n_samples)
        end_time = time.time()
        
        # Compute empirical cumulants
        emp_cumulants = [
            np.mean(samples),  # First cumulant (mean)
            np.var(samples),   # Second cumulant (variance)
            stats.skew(samples) * np.var(samples)**(3/2),  # Third cumulant
            stats.kurtosis(samples, fisher=False) * np.var(samples)**2 - 3 * np.var(samples)**2  # Fourth cumulant
        ]
        
        # Compute accuracy metrics
        abs_errors = [abs(true - emp) for true, emp in zip(true_cumulants, emp_cumulants)]
        rel_errors = [abs_err / (abs(true) + 1e-10) for abs_err, true in zip(abs_errors, true_cumulants)]
        
        print(f"True cumulants:   {[f'{c:.6f}' for c in true_cumulants]}")
        print(f"FGMC cumulants:   {[f'{c:.6f}' for c in emp_cumulants]}")
        print(f"Absolute errors:  {[f'{e:.6f}' for e in abs_errors]}")
        print(f"Relative errors:  {[f'{e:.6f}' for e in rel_errors]}")
        print(f"Computation time: {end_time - start_time:.4f} seconds")
        
        results.append({
            'Process': 'OU-NTS',
            'Alpha': alpha,
            'True Cumulants': true_cumulants,
            'FGMC Cumulants': emp_cumulants,
            'Absolute Errors': abs_errors,
            'Relative Errors': rel_errors,
            'Time': end_time - start_time
        })
    
    # Now test for NTS-OU process
    for alpha in [a for a in alpha_values if a >= 0]:  # NTS-OU only defined for alpha >= 0
        print(f"\nTesting NTS-OU with alpha = {alpha}")
        
        # Calculate true cumulants
        true_cumulants = cumulants_NTS_OU(t, alpha, kappa, sigma, b, theta)
        
        # Define characteristic function
        def cf_nts_ou(u, t):
            return NTS_OU_LCF(u, t, alpha, kappa, sigma, b, theta)
        
        # Get analyticity strip
        strip = analyticity_strip_NTS(alpha, kappa, sigma, theta)
        
        # Determine if process has Finite Activity
        is_FA = alpha == 0
        
        # Set up generator
        if is_FA:
            # For Finite Activity processes, we need lambda_t and CF_jumps
            lambda_t = 2 * b / kappa  # Intensity for alpha=0
            
            # CF of jumps conditional on at least one jump
            def cf_jumps(u, t):
                f = lambda_t * ((np.exp(cf_nts_ou(u, t)) - 1) / (np.exp(lambda_t) - 1))
                return np.log(f)
            
            start_time = time.time()
            generator = FGMC(cf_nts_ou, t, strip, M=16, is_FA=True, 
                             lambda_t=lambda_t, CF_jumps=cf_jumps)
        else:
            start_time = time.time()
            generator = FGMC(cf_nts_ou, t, strip, M=16)
        
        # Generate samples
        samples = generator(n_samples)
        end_time = time.time()
        
        # Compute empirical cumulants
        emp_cumulants = [
            np.mean(samples),  # First cumulant (mean)
            np.var(samples),   # Second cumulant (variance)
            stats.skew(samples) * np.var(samples)**(3/2),  # Third cumulant
            stats.kurtosis(samples, fisher=False) * np.var(samples)**2 - 3 * np.var(samples)**2  # Fourth cumulant
        ]
        
        # Compute accuracy metrics
        abs_errors = [abs(true - emp) for true, emp in zip(true_cumulants, emp_cumulants)]
        rel_errors = [abs_err / (abs(true) + 1e-10) for abs_err, true in zip(abs_errors, true_cumulants)]
        
        print(f"True cumulants:   {[f'{c:.6f}' for c in true_cumulants]}")
        print(f"FGMC cumulants:   {[f'{c:.6f}' for c in emp_cumulants]}")
        print(f"Absolute errors:  {[f'{e:.6f}' for e in abs_errors]}")
        print(f"Relative errors:  {[f'{e:.6f}' for e in rel_errors]}")
        print(f"Computation time: {end_time - start_time:.4f} seconds")
        
        results.append({
            'Process': 'NTS-OU',
            'Alpha': alpha,
            'True Cumulants': true_cumulants,
            'FGMC Cumulants': emp_cumulants,
            'Absolute Errors': abs_errors,
            'Relative Errors': rel_errors,
            'Time': end_time - start_time
        })
    
    return results

def european_option_pricing():
    """
    Test the FGMC algorithm for pricing European options
    """
    print("\nPricing European options...")
    
    # Parameters
    S0 = 1.0
    r = 0.0
    T = 1.0
    n_samples = 100_000
    
    # Parameters for TS processes
    b_ts = 0.1
    beta_ts = 2.5
    c_ts = 0.5
    alpha_ts = 0.8
    
    # Parameters for NTS processes
    b_nts = 0.2162
    kappa_nts = 0.256
    sigma_nts = 0.201
    theta_nts = 0
    alpha_nts = 0.4
    
    # Moneyness levels
    moneyness_levels = np.linspace(-0.2, 0.2, 30) * np.sqrt(T)
    strikes = S0 * np.exp(moneyness_levels)
    
    # Define characteristic functions
    def cf_ou_ts(u, t):
        return OU_TS_LCF(u, t, alpha_ts, beta_ts, c_ts, b_ts)
    
    def cf_ts_ou(u, t):
        return TS_OU_LCF(u, t, alpha_ts, beta_ts, c_ts, b_ts)
    
    def cf_ou_nts(u, t):
        return OU_NTS_LCF(u, t, alpha_nts, kappa_nts, sigma_nts, b_nts, theta_nts)
    
    def cf_nts_ou(u, t):
        return NTS_OU_LCF(u, t, alpha_nts, kappa_nts, sigma_nts, b_nts, theta_nts)
    
    # Get analyticity strips
    strip_ts = analyticity_strip_TS(alpha_ts, beta_ts)
    strip_nts = analyticity_strip_NTS(alpha_nts, kappa_nts, sigma_nts, theta_nts)
    
    # Create generators
    generator_ou_ts = FGMC(cf_ou_ts, T, strip_ts, M=16)
    generator_ts_ou = FGMC(cf_ts_ou, T, strip_ts, M=16)
    generator_ou_nts = FGMC(cf_ou_nts, T, strip_nts, M=16)
    generator_nts_ou = FGMC(cf_nts_ou, T, strip_nts, M=16)
    
    # Generate samples
    samples_ou_ts = generator_ou_ts(n_samples)
    samples_ts_ou = generator_ts_ou(n_samples)
    samples_ou_nts = generator_ou_nts(n_samples)
    samples_nts_ou = generator_nts_ou(n_samples)
    
    # Calculate spot prices at maturity
    spots_ou_ts = S0 * np.exp(samples_ou_ts)
    spots_ts_ou = S0 * np.exp(samples_ts_ou)
    spots_ou_nts = S0 * np.exp(samples_ou_nts)
    spots_nts_ou = S0 * np.exp(samples_nts_ou)
    
    # Price options
    prices_ou_ts = []
    prices_ts_ou = []
    prices_ou_nts = []
    prices_nts_ou = []
    
    for K in strikes:
        prices_ou_ts.append(price_european_call(S0, K, r, T, spots_ou_ts))
        prices_ts_ou.append(price_european_call(S0, K, r, T, spots_ts_ou))
        prices_ou_nts.append(price_european_call(S0, K, r, T, spots_ou_nts))
        prices_nts_ou.append(price_european_call(S0, K, r, T, spots_nts_ou))
    
    # Plot results
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.plot(moneyness_levels, prices_ou_ts, 'b-', label='OU-TS')
    plt.title('European Call Prices (OU-TS)')
    plt.xlabel('Moneyness')
    plt.ylabel('Price')
    plt.grid(True)
    
    plt.subplot(2, 2, 2)
    plt.plot(moneyness_levels, prices_ts_ou, 'r-', label='TS-OU')
    plt.title('European Call Prices (TS-OU)')
    plt.xlabel('Moneyness')
    plt.ylabel('Price')
    plt.grid(True)
    
    plt.subplot(2, 2, 3)
    plt.plot(moneyness_levels, prices_ou_nts, 'g-', label='OU-NTS')
    plt.title('European Call Prices (OU-NTS)')
    plt.xlabel('Moneyness')
    plt.ylabel('Price')
    plt.grid(True)
    
    plt.subplot(2, 2, 4)
    plt.plot(moneyness_levels, prices_nts_ou, 'm-', label='NTS-OU')
    plt.title('European Call Prices (NTS-OU)')
    plt.xlabel('Moneyness')
    plt.ylabel('Price')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('european_option_prices.png')
    
    return {
        'Moneyness': moneyness_levels,
        'Prices OU-TS': prices_ou_ts,
        'Prices TS-OU': prices_ts_ou,
        'Prices OU-NTS': prices_ou_nts,
        'Prices NTS-OU': prices_nts_ou
    }

def asian_option_pricing():
    """
    Test the FGMC algorithm for pricing Asian options
    """
    print("\nPricing Asian options...")
    
    # Parameters
    S0 = 1.0
    r = 0.0
    T = 2.0
    n_paths = 10_000
    monitoring_dates = 24  # Monthly monitoring
    
    # Parameters for TS processes
    b_ts = 0.1
    beta_ts = 2.5
    c_ts = 0.5
    alpha_ts = 0.8
    
    # Parameters for NTS processes
    b_nts = 0.2162
    kappa_nts = 0.256
    sigma_nts = 0.201
    theta_nts = 0
    alpha_nts = 0.4
    
    # Moneyness levels for OTM, ATM, ITM
    moneyness_levels = np.array([-0.2, 0.0, 0.2]) * np.sqrt(T)
    strikes = S0 * np.exp(moneyness_levels)
    
    # Time grid
    t_grid = np.linspace(0, T, monitoring_dates + 1)
    
    # Define characteristic functions
    def cf_ou_ts(u, t):
        return OU_TS_LCF(u, t, alpha_ts, beta_ts, c_ts, b_ts)
    
    def cf_ts_ou(u, t):
        return TS_OU_LCF(u, t, alpha_ts, beta_ts, c_ts, b_ts)
    
    def cf_ou_nts(u, t):
        return OU_NTS_LCF(u, t, alpha_nts, kappa_nts, sigma_nts, b_nts, theta_nts)
    
    def cf_nts_ou(u, t):
        return NTS_OU_LCF(u, t, alpha_nts, kappa_nts, sigma_nts, b_nts, theta_nts)
    
    # Get analyticity strips
    strip_ts = analyticity_strip_TS(alpha_ts, beta_ts)
    strip_nts = analyticity_strip_NTS(alpha_nts, kappa_nts, sigma_nts, theta_nts)
    
    # Create generators for each time step
    generators_ou_ts = []
    generators_ts_ou = []
    generators_ou_nts = []
    generators_nts_ou = []
    
    for i in range(len(t_grid) - 1):
        dt = t_grid[i+1] - t_grid[i]
        generators_ou_ts.append(FGMC(cf_ou_ts, dt, strip_ts, M=16))
        generators_ts_ou.append(FGMC(cf_ts_ou, dt, strip_ts, M=16))
        generators_ou_nts.append(FGMC(cf_ou_nts, dt, strip_nts, M=16))
        generators_nts_ou.append(FGMC(cf_nts_ou, dt, strip_nts, M=16))
    
    # Simulate paths
    paths_ou_ts = np.zeros((n_paths, len(t_grid)))
    paths_ts_ou = np.zeros((n_paths, len(t_grid)))
    paths_ou_nts = np.zeros((n_paths, len(t_grid)))
    paths_nts_ou = np.zeros((n_paths, len(t_grid)))
    
    # Initial values
    paths_ou_ts[:, 0] = 0
    paths_ts_ou[:, 0] = 0
    paths_ou_nts[:, 0] = 0
    paths_nts_ou[:, 0] = 0
    
    # Simulate paths
    for i in range(len(t_grid) - 1):
        dt = t_grid[i+1] - t_grid[i]
        
        # OU-TS
        Z = generators_ou_ts[i](n_paths)
        paths_ou_ts[:, i+1] = paths_ou_ts[:, i] * np.exp(-b_ts * dt) + Z
        
        # TS-OU
        Z = generators_ts_ou[i](n_paths)
        paths_ts_ou[:, i+1] = paths_ts_ou[:, i] * np.exp(-b_ts * dt) + Z
        
        # OU-NTS
        Z = generators_ou_nts[i](n_paths)
        paths_ou_nts[:, i+1] = paths_ou_nts[:, i] * np.exp(-b_nts * dt) + Z
        
        # NTS-OU
        Z = generators_nts_ou[i](n_paths)
        paths_nts_ou[:, i+1] = paths_nts_ou[:, i] * np.exp(-b_nts * dt) + Z
    
    # Convert to spot price paths
    spot_paths_ou_ts = S0 * np.exp(paths_ou_ts)
    spot_paths_ts_ou = S0 * np.exp(paths_ts_ou)
    spot_paths_ou_nts = S0 * np.exp(paths_ou_nts)
    spot_paths_nts_ou = S0 * np.exp(paths_nts_ou)
    
    # Price Asian options
    prices_ou_ts = []
    prices_ts_ou = []
    prices_ou_nts = []
    prices_nts_ou = []
    
    for K in strikes:
        prices_ou_ts.append(price_asian_call(S0, K, r, T, spot_paths_ou_ts[:, 1:]))
        prices_ts_ou.append(price_asian_call(S0, K, r, T, spot_paths_ts_ou[:, 1:]))
        prices_ou_nts.append(price_asian_call(S0, K, r, T, spot_paths_ou_nts[:, 1:]))
        prices_nts_ou.append(price_asian_call(S0, K, r, T, spot_paths_nts_ou[:, 1:]))
    
    # Print results
    print("\nAsian Option Prices:")
    print("Moneyness: OTM (0.2), ATM (0.0), ITM (-0.2)")
    print(f"OU-TS:  {prices_ou_ts}")
    print(f"TS-OU:  {prices_ts_ou}")
    print(f"OU-NTS: {prices_ou_nts}")
    print(f"NTS-OU: {prices_nts_ou}")
    
    # Plot some sample paths
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    for i in range(5):
        plt.plot(t_grid, spot_paths_ou_ts[i])
    plt.title('Sample Paths (OU-TS)')
    plt.xlabel('Time')
    plt.ylabel('Spot Price')
    plt.grid(True)
    
    plt.subplot(2, 2, 2)
    for i in range(5):
        plt.plot(t_grid, spot_paths_ts_ou[i])
    plt.title('Sample Paths (TS-OU)')
    plt.xlabel('Time')
    plt.ylabel('Spot Price')
    plt.grid(True)
    
    plt.subplot(2, 2, 3)
    for i in range(5):
        plt.plot(t_grid, spot_paths_ou_nts[i])
    plt.title('Sample Paths (OU-NTS)')
    plt.xlabel('Time')
    plt.ylabel('Spot Price')
    plt.grid(True)
    
    plt.subplot(2, 2, 4)
    for i in range(5):
        plt.plot(t_grid, spot_paths_nts_ou[i])
    plt.title('Sample Paths (NTS-OU)')
    plt.xlabel('Time')
    plt.ylabel('Spot Price')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('asian_option_sample_paths.png')
    
    return {
        'Moneyness': ['OTM', 'ATM', 'ITM'],
        'Prices OU-TS': prices_ou_ts,
        'Prices TS-OU': prices_ts_ou,
        'Prices OU-NTS': prices_ou_nts,
        'Prices NTS-OU': prices_nts_ou
    }

# Run the experiments
if __name__ == "__main__":
    accuracy_results = test_accuracy_and_speed()
    european_results = european_option_pricing()
    asian_results = asian_option_pricing()
    
    plt.show()