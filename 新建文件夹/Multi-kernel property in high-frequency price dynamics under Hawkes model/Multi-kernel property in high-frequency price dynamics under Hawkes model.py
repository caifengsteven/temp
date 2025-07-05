import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# Function to simulate a multivariate Hawkes process with multiple kernels
def simulate_multivariate_hawkes(mu, alpha_matrices, beta_matrices, T, seed=None):
    """
    Simulate a multivariate Hawkes process with multiple kernels.
    
    Parameters:
    -----------
    mu : array
        Base intensities for each dimension
    alpha_matrices : list of arrays
        List of alpha matrices for each kernel
    beta_matrices : list of arrays
        List of beta matrices for each kernel
    T : float
        End time for simulation
    seed : int, optional
        Random seed
    
    Returns:
    --------
    event_times : list of arrays
        Event times for each dimension
    intensities : list of arrays
        Intensity processes at event times
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Dimensions
    d = len(mu)
    K = len(alpha_matrices)  # Number of kernels
    
    # Ensure matrices are numpy arrays
    mu = np.array(mu)
    alpha_matrices = [np.array(alpha) for alpha in alpha_matrices]
    beta_matrices = [np.array(beta) for beta in beta_matrices]
    
    # Initialize
    event_times = [[] for _ in range(d)]
    intensities = [[] for _ in range(d)]
    
    # Current intensity components for each kernel
    lambda_components = [[np.zeros(d) for _ in range(K)] for _ in range(d)]
    
    # Current time
    t = 0
    
    # Simulate
    while t < T:
        # Calculate current total intensity for each dimension
        lambda_total = np.zeros(d)
        for i in range(d):
            lambda_i = mu[i]
            for k in range(K):
                lambda_i += np.sum(lambda_components[i][k])
            lambda_total[i] = max(0, lambda_i)  # Ensure non-negative
        
        # Sample next event time
        lambda_sum = np.sum(lambda_total)
        if lambda_sum <= 0:
            # No events possible, move to end time
            t = T
            continue
        
        dt = np.random.exponential(scale=1.0/lambda_sum)
        t_next = t + dt
        
        # Update intensity components due to exponential decay
        for i in range(d):
            for j in range(d):
                for k in range(K):
                    # Decay exponentially
                    lambda_components[i][k][j] *= np.exp(-beta_matrices[k][i, j] * dt)
        
        # Check if event occurs within simulation time
        if t_next < T:
            # Determine dimension of event
            event_dim = np.random.choice(d, p=lambda_total/lambda_sum)
            
            # Record event
            event_times[event_dim].append(t_next)
            intensities[event_dim].append(lambda_total[event_dim])
            
            # Update intensity components due to excitation
            for i in range(d):
                for k in range(K):
                    lambda_components[i][k][event_dim] += alpha_matrices[k][i, event_dim]
        
        # Move to next time
        t = t_next
    
    # Convert to numpy arrays
    event_times = [np.array(times) for times in event_times]
    intensities = [np.array(ints) for ints in intensities]
    
    return event_times, intensities

# Function to compute residuals for model validation (Q-Q plot)
def compute_residuals(event_times, lambda_func, T):
    """
    Compute residuals for model validation.
    
    Parameters:
    -----------
    event_times : list of arrays
        Event times for each dimension
    lambda_func : function
        Function to compute intensity at a given time
    T : float
        End time
    
    Returns:
    --------
    residuals : list of arrays
        Residuals for each dimension
    """
    d = len(event_times)
    residuals = []
    
    for i in range(d):
        times = event_times[i]
        res_i = []
        
        if len(times) > 0:
            # First event
            if times[0] > 0:
                # Integral of intensity from 0 to first event
                t_prev = 0
                t_curr = times[0]
                integral = integrate_intensity(lambda_func, i, t_prev, t_curr)
                res_i.append(integral)
            
            # Subsequent events
            for j in range(1, len(times)):
                t_prev = times[j-1]
                t_curr = times[j]
                integral = integrate_intensity(lambda_func, i, t_prev, t_curr)
                res_i.append(integral)
            
            # Last event to T
            if times[-1] < T:
                t_prev = times[-1]
                t_curr = T
                integral = integrate_intensity(lambda_func, i, t_prev, t_curr)
                res_i.append(integral)
        
        residuals.append(np.array(res_i))
    
    return residuals

# Function to integrate intensity (approximate with numerical integration)
def integrate_intensity(lambda_func, dim, t_start, t_end, n_steps=100):
    """
    Numerically integrate intensity function.
    
    Parameters:
    -----------
    lambda_func : function
        Function to compute intensity at a given time
    dim : int
        Dimension of intensity
    t_start : float
        Start time
    t_end : float
        End time
    n_steps : int, optional
        Number of steps for numerical integration
    
    Returns:
    --------
    integral : float
        Integral of intensity function
    """
    if t_start >= t_end:
        return 0
    
    dt = (t_end - t_start) / n_steps
    integral = 0
    
    for i in range(n_steps):
        t = t_start + i * dt
        integral += lambda_func(dim, t) * dt
    
    return integral

# Function to compute log-likelihood for Hawkes process
def compute_log_likelihood(event_times, lambda_func, T):
    """
    Compute log-likelihood for Hawkes process.
    
    Parameters:
    -----------
    event_times : list of arrays
        Event times for each dimension
    lambda_func : function
        Function to compute intensity at a given time
    T : float
        End time
    
    Returns:
    --------
    log_likelihood : float
        Log-likelihood value
    """
    d = len(event_times)
    log_likelihood = 0
    
    # Sum log of intensities at event times
    for i in range(d):
        for t in event_times[i]:
            log_likelihood += np.log(lambda_func(i, t))
    
    # Subtract integral of intensities
    for i in range(d):
        log_likelihood -= integrate_intensity(lambda_func, i, 0, T)
    
    return log_likelihood

# Function to compute conditional expectation of arrival time
def compute_conditional_expectation(alpha, beta, num_points=1000):
    """
    Compute conditional expectation of arrival time.
    
    Parameters:
    -----------
    alpha : float
        Excitation parameter
    beta : float
        Decay parameter
    num_points : int, optional
        Number of points for numerical integration
    
    Returns:
    --------
    expected_time : float
        Conditional expectation of arrival time
    """
    # Define the function to integrate
    def integrand(u):
        return alpha * u * np.exp(-alpha * (1 - np.exp(-beta * u)) / beta - beta * u)
    
    # Numerical integration
    u_max = 10 / beta  # Practical upper limit for integration
    u_vals = np.linspace(0, u_max, num_points)
    du = u_vals[1] - u_vals[0]
    
    # Compute integral
    integral = 0
    for u in u_vals:
        integral += integrand(u) * du
    
    return integral

# Function to fit multi-kernel Hawkes model
def fit_multi_kernel_hawkes(event_times, T, num_kernels, initial_params=None, bounds=None, method='L-BFGS-B'):
    """
    Fit multi-kernel Hawkes model to event data.
    
    Parameters:
    -----------
    event_times : list of arrays
        Event times for each dimension
    T : float
        End time
    num_kernels : int
        Number of kernels
    initial_params : array, optional
        Initial parameter values
    bounds : list of tuples, optional
        Bounds for parameters
    method : str, optional
        Optimization method
    
    Returns:
    --------
    params : array
        Fitted parameters
    log_likelihood : float
        Log-likelihood at optimum
    """
    d = len(event_times)
    
    # Parameter structure: [mu, alpha_1, beta_1, alpha_2, beta_2, ...]
    # For bivariate case: mu has 2 params, each alpha and beta matrix has 4 params
    num_params = d + 2 * d * d * num_kernels
    
    if initial_params is None:
        # Default initialization
        initial_params = np.ones(num_params)
        # Set mus
        initial_params[:d] = np.array([len(times) / T for times in event_times])
        # Set alphas (decreasing for higher kernels)
        for k in range(num_kernels):
            alpha_idx = d + 2 * k * d * d
            initial_params[alpha_idx:alpha_idx + d*d] = 1.0 / (k + 1)
        # Set betas (increasing for higher kernels)
        for k in range(num_kernels):
            beta_idx = d + (2 * k + 1) * d * d
            initial_params[beta_idx:beta_idx + d*d] = 10.0 * (k + 1)
    
    if bounds is None:
        # Default bounds
        bounds = [(1e-6, None) for _ in range(num_params)]
    
    # Function to extract parameters
    def extract_params(params):
        mu = params[:d]
        alpha_matrices = []
        beta_matrices = []
        
        for k in range(num_kernels):
            alpha_idx = d + 2 * k * d * d
            beta_idx = d + (2 * k + 1) * d * d
            
            alpha_k = params[alpha_idx:alpha_idx + d*d].reshape((d, d))
            beta_k = params[beta_idx:beta_idx + d*d].reshape((d, d))
            
            alpha_matrices.append(alpha_k)
            beta_matrices.append(beta_k)
        
        return mu, alpha_matrices, beta_matrices
    
    # Function to compute intensity at a given time
    def lambda_func(params, dim, t):
        mu, alpha_matrices, beta_matrices = extract_params(params)
        
        # Base intensity
        intensity = mu[dim]
        
        # Add excitation from each kernel
        for k in range(num_kernels):
            for j in range(d):
                for event_time in event_times[j]:
                    if event_time < t:
                        intensity += alpha_matrices[k][dim, j] * np.exp(-beta_matrices[k][dim, j] * (t - event_time))
        
        return max(0, intensity)
    
    # Negative log-likelihood function to minimize
    def neg_log_likelihood(params):
        mu, alpha_matrices, beta_matrices = extract_params(params)
        
        # Check for invalid parameters
        if np.any(mu <= 0):
            return np.inf
        
        for k in range(num_kernels):
            if np.any(alpha_matrices[k] < 0) or np.any(beta_matrices[k] <= 0):
                return np.inf
        
        # Compute log-likelihood
        ll = 0
        
        # Sum log of intensities at event times
        for i in range(d):
            for t in event_times[i]:
                intensity = lambda_func(params, i, t)
                if intensity <= 0:
                    return np.inf
                ll += np.log(intensity)
        
        # Subtract integral of intensities
        for i in range(d):
            # Approximate integral using event times
            all_times = np.sort(np.concatenate([np.concatenate(event_times), [0, T]]))
            for j in range(len(all_times) - 1):
                t1, t2 = all_times[j], all_times[j+1]
                if t1 == t2:
                    continue
                
                # Midpoint approximation
                mid_t = (t1 + t2) / 2
                intensity_mid = lambda_func(params, i, mid_t)
                ll -= intensity_mid * (t2 - t1)
        
        return -ll
    
    # Optimize
    result = minimize(neg_log_likelihood, initial_params, method=method, bounds=bounds)
    
    return result.x, -result.fun

# Function to create mid-price from bidirectional Hawkes process
def create_mid_price(up_events, down_events, tick_size=0.01, initial_price=100.0):
    """
    Create mid-price process from up and down events.
    
    Parameters:
    -----------
    up_events : array
        Times of upward price movements
    down_events : array
        Times of downward price movements
    tick_size : float, optional
        Size of each price tick
    initial_price : float, optional
        Initial price
    
    Returns:
    --------
    times : array
        Times of all price changes
    prices : array
        Mid-prices at each time
    """
    # Combine events
    all_events = np.concatenate([
        np.column_stack([up_events, np.ones(len(up_events))]), 
        np.column_stack([down_events, -np.ones(len(down_events))])
    ])
    
    # Sort by time
    all_events = all_events[all_events[:, 0].argsort()]
    
    # Extract sorted times and directions
    times = all_events[:, 0]
    directions = all_events[:, 1]
    
    # Calculate price process
    prices = initial_price + np.cumsum(directions) * tick_size
    
    # Include initial price at time 0
    if times[0] > 0:
        times = np.insert(times, 0, 0)
        prices = np.insert(prices, 0, initial_price)
    
    return times, prices

# Function to calculate branching ratio
def calculate_branching_ratio(alpha_matrices, beta_matrices):
    """
    Calculate branching ratio for Hawkes process.
    
    Parameters:
    -----------
    alpha_matrices : list of arrays
        List of alpha matrices for each kernel
    beta_matrices : list of arrays
        List of beta matrices for each kernel
    
    Returns:
    --------
    branching_ratio : float
        Spectral radius of the branching matrix
    """
    d = alpha_matrices[0].shape[0]
    K = len(alpha_matrices)
    
    # Compute branching matrix
    G = np.zeros((d, d))
    for k in range(K):
        G += alpha_matrices[k] / beta_matrices[k]
    
    # Compute spectral radius
    eigenvalues = np.linalg.eigvals(G)
    branching_ratio = max(abs(eigenvalues))
    
    return branching_ratio

# Function to compute expected arrival time matrix
def compute_expected_arrival_times(alpha_matrices, beta_matrices):
    """
    Compute expected arrival times for each kernel and each dimension.
    
    Parameters:
    -----------
    alpha_matrices : list of arrays
        List of alpha matrices for each kernel
    beta_matrices : list of arrays
        List of beta matrices for each kernel
    
    Returns:
    --------
    expected_times : list of arrays
        Expected arrival times for each kernel
    """
    K = len(alpha_matrices)
    d = alpha_matrices[0].shape[0]
    
    expected_times = []
    
    for k in range(K):
        times_k = np.zeros((d, d))
        for i in range(d):
            for j in range(d):
                if alpha_matrices[k][i, j] > 0:
                    times_k[i, j] = compute_conditional_expectation(
                        alpha_matrices[k][i, j], beta_matrices[k][i, j])
                else:
                    times_k[i, j] = np.inf
        expected_times.append(times_k)
    
    return expected_times

# Function to analyze kernel contributions
def analyze_kernel_contributions(event_times, mu, alpha_matrices, beta_matrices, T):
    """
    Analyze contributions of each kernel to the intensity.
    
    Parameters:
    -----------
    event_times : list of arrays
        Event times for each dimension
    mu : array
        Base intensities
    alpha_matrices : list of arrays
        Alpha matrices for each kernel
    beta_matrices : list of arrays
        Beta matrices for each kernel
    T : float
        End time
    
    Returns:
    --------
    kernel_contributions : list of arrays
        Contribution of each kernel to intensity at each event time
    """
    d = len(event_times)
    K = len(alpha_matrices)
    
    # Combine all event times
    all_events = []
    for i in range(d):
        for t in event_times[i]:
            all_events.append((t, i))
    
    # Sort by time
    all_events.sort()
    
    kernel_contributions = [[] for _ in range(K)]
    base_contributions = []
    
    # Initialize intensity components
    lambda_components = np.zeros((d, K, d))
    
    # Analyze each event
    prev_time = 0
    for event_time, event_dim in all_events:
        # Decay intensity components
        dt = event_time - prev_time
        for i in range(d):
            for k in range(K):
                for j in range(d):
                    lambda_components[i, k, j] *= np.exp(-beta_matrices[k][i, j] * dt)
        
        # Compute total intensity for the event dimension
        total_intensity = mu[event_dim]
        kernel_intensities = np.zeros(K)
        
        for k in range(K):
            kernel_intensities[k] = np.sum(lambda_components[event_dim, k, :])
            total_intensity += kernel_intensities[k]
        
        # Compute contribution percentages
        base_contribution = mu[event_dim] / total_intensity
        base_contributions.append(base_contribution)
        
        for k in range(K):
            kernel_contribution = kernel_intensities[k] / total_intensity
            kernel_contributions[k].append(kernel_contribution)
        
        # Update intensity components due to the event
        for i in range(d):
            for k in range(K):
                lambda_components[i, k, event_dim] += alpha_matrices[k][i, event_dim]
        
        prev_time = event_time
    
    # Convert to numpy arrays
    kernel_contributions = [np.array(contribs) for contribs in kernel_contributions]
    base_contributions = np.array(base_contributions)
    
    return kernel_contributions, base_contributions

# Function to calculate AIC
def calculate_aic(log_likelihood, num_params):
    """
    Calculate Akaike Information Criterion.
    
    Parameters:
    -----------
    log_likelihood : float
        Log-likelihood value
    num_params : int
        Number of parameters
    
    Returns:
    --------
    aic : float
        AIC value
    """
    return 2 * num_params - 2 * log_likelihood

# Main function to simulate and analyze multi-kernel Hawkes model
def main():
    # Set parameters
    T = 100  # Simulation time
    d = 2    # Number of dimensions (up and down)
    
    print("Simulating multi-kernel Hawkes process for mid-price dynamics...")
    
    # Define parameters for multi-kernel model
    # Kernel 1: Ultra-High-Frequency (UHF)
    mu = [0.1, 0.1]  # Base intensities
    
    # UHF kernel (milliseconds scale)
    alpha1 = np.array([
        [0.6, 0.2],  # Self and cross-excitation for up
        [0.2, 0.6]   # Cross and self-excitation for down
    ])
    beta1 = np.array([
        [1500, 1500],  # Decay rates for up (self and cross)
        [1500, 1500]   # Decay rates for down (cross and self)
    ])
    
    # VHF kernel (seconds scale)
    alpha2 = np.array([
        [0.3, 0.1],
        [0.1, 0.3]
    ])
    beta2 = np.array([
        [50, 50],
        [50, 50]
    ])
    
    # HF kernel (tens of seconds scale)
    alpha3 = np.array([
        [0.05, 0.02],
        [0.02, 0.05]
    ])
    beta3 = np.array([
        [2, 2],
        [2, 2]
    ])
    
    # Simulate multi-kernel Hawkes process
    alpha_matrices = [alpha1, alpha2, alpha3]
    beta_matrices = [beta1, beta2, beta3]
    
    print("Simulating with 3 kernels (UHF, VHF, HF)...")
    event_times_3k, intensities_3k = simulate_multivariate_hawkes(mu, alpha_matrices, beta_matrices, T)
    
    # Simulate with 2 kernels (UHF, VHF)
    print("Simulating with 2 kernels (UHF, VHF)...")
    alpha_matrices_2k = [alpha1, alpha2]
    beta_matrices_2k = [beta1, beta2]
    event_times_2k, intensities_2k = simulate_multivariate_hawkes(mu, alpha_matrices_2k, beta_matrices_2k, T)
    
    # Simulate with 1 kernel (only UHF)
    print("Simulating with 1 kernel (UHF only)...")
    alpha_matrices_1k = [alpha1]
    beta_matrices_1k = [beta1]
    event_times_1k, intensities_1k = simulate_multivariate_hawkes(mu, alpha_matrices_1k, beta_matrices_1k, T)
    
    # Create mid-price process
    print("Creating mid-price processes...")
    times_3k, prices_3k = create_mid_price(event_times_3k[0], event_times_3k[1])
    times_2k, prices_2k = create_mid_price(event_times_2k[0], event_times_2k[1])
    times_1k, prices_1k = create_mid_price(event_times_1k[0], event_times_1k[1])
    
    # Plot price processes
    plt.figure(figsize=(12, 8))
    plt.subplot(3, 1, 1)
    plt.step(times_1k, prices_1k, where='post', label='1 Kernel')
    plt.title('Mid-Price Process with 1 Kernel')
    plt.ylabel('Price')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(3, 1, 2)
    plt.step(times_2k, prices_2k, where='post', label='2 Kernels')
    plt.title('Mid-Price Process with 2 Kernels')
    plt.ylabel('Price')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(3, 1, 3)
    plt.step(times_3k, prices_3k, where='post', label='3 Kernels')
    plt.title('Mid-Price Process with 3 Kernels')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('mid_price_processes.png')
    plt.close()
    
    # Calculate statistics
    print("\nStatistics for simulated processes:")
    print(f"1 Kernel - Number of events: Up={len(event_times_1k[0])}, Down={len(event_times_1k[1])}")
    print(f"2 Kernels - Number of events: Up={len(event_times_2k[0])}, Down={len(event_times_2k[1])}")
    print(f"3 Kernels - Number of events: Up={len(event_times_3k[0])}, Down={len(event_times_3k[1])}")
    
    # Calculate branching ratios
    br_1k = calculate_branching_ratio(alpha_matrices_1k, beta_matrices_1k)
    br_2k = calculate_branching_ratio(alpha_matrices_2k, beta_matrices_2k)
    br_3k = calculate_branching_ratio(alpha_matrices_3k, beta_matrices_3k)
    
    print(f"\nBranching ratios:")
    print(f"1 Kernel: {br_1k:.4f}")
    print(f"2 Kernels: {br_2k:.4f}")
    print(f"3 Kernels: {br_3k:.4f}")
    
    # Calculate expected arrival times
    print("\nComputing expected arrival times...")
    expected_times_1k = compute_expected_arrival_times(alpha_matrices_1k, beta_matrices_1k)
    expected_times_2k = compute_expected_arrival_times(alpha_matrices_2k, beta_matrices_2k)
    expected_times_3k = compute_expected_arrival_times(alpha_matrices_3k, beta_matrices_3k)
    
    print("\nExpected arrival times (in seconds):")
    print("1 Kernel model:")
    print("UHF kernel:")
    print("Self-excitation (Up): {:.2f} ms".format(expected_times_1k[0][0, 0] * 1000))
    print("Cross-excitation (Up->Down): {:.2f} ms".format(expected_times_1k[0][1, 0] * 1000))
    print("Cross-excitation (Down->Up): {:.2f} ms".format(expected_times_1k[0][0, 1] * 1000))
    print("Self-excitation (Down): {:.2f} ms".format(expected_times_1k[0][1, 1] * 1000))
    
    print("\n2 Kernel model:")
    print("UHF kernel:")
    print("Self-excitation (Up): {:.2f} ms".format(expected_times_2k[0][0, 0] * 1000))
    print("Cross-excitation (Up->Down): {:.2f} ms".format(expected_times_2k[0][1, 0] * 1000))
    print("VHF kernel:")
    print("Self-excitation (Up): {:.2f} ms".format(expected_times_2k[1][0, 0] * 1000))
    print("Cross-excitation (Up->Down): {:.2f} ms".format(expected_times_2k[1][1, 0] * 1000))
    
    print("\n3 Kernel model:")
    print("UHF kernel:")
    print("Self-excitation (Up): {:.2f} ms".format(expected_times_3k[0][0, 0] * 1000))
    print("Cross-excitation (Up->Down): {:.2f} ms".format(expected_times_3k[0][1, 0] * 1000))
    print("VHF kernel:")
    print("Self-excitation (Up): {:.2f} ms".format(expected_times_3k[1][0, 0] * 1000))
    print("Cross-excitation (Up->Down): {:.2f} ms".format(expected_times_3k[1][1, 0] * 1000))
    print("HF kernel:")
    print("Self-excitation (Up): {:.2f} ms".format(expected_times_3k[2][0, 0] * 1000))
    print("Cross-excitation (Up->Down): {:.2f} ms".format(expected_times_3k[2][1, 0] * 1000))
    
    # Analyze kernel contributions
    print("\nAnalyzing kernel contributions...")
    kernel_contrib_3k, base_contrib_3k = analyze_kernel_contributions(
        event_times_3k, mu, alpha_matrices, beta_matrices, T)
    
    # Calculate average contributions
    avg_base_contrib = np.mean(base_contrib_3k) * 100
    avg_kernel_contrib = [np.mean(contrib) * 100 for contrib in kernel_contrib_3k]
    
    print("\nAverage contributions to intensity:")
    print(f"Base intensity: {avg_base_contrib:.2f}%")
    print(f"UHF kernel: {avg_kernel_contrib[0]:.2f}%")
    print(f"VHF kernel: {avg_kernel_contrib[1]:.2f}%")
    print(f"HF kernel: {avg_kernel_contrib[2]:.2f}%")
    
    # Plot kernel contributions
    plt.figure(figsize=(10, 6))
    labels = ['Base', 'UHF', 'VHF', 'HF']
    values = [avg_base_contrib] + avg_kernel_contrib
    colors = ['gray', 'red', 'blue', 'green']
    
    plt.bar(labels, values, color=colors, alpha=0.7)
    plt.title('Average Contributions to Intensity')
    plt.ylabel('Contribution (%)')
    plt.grid(True, alpha=0.3)
    plt.savefig('kernel_contributions.png')
    plt.close()
    
    # Fit models to simulated data
    print("\nFitting Hawkes models to the 3-kernel simulated data...")
    
    # Fit 1-kernel model
    print("Fitting 1-kernel model...")
    params_1k, ll_1k = fit_multi_kernel_hawkes(event_times_3k, T, 1)
    
    # Fit 2-kernel model
    print("Fitting 2-kernel model...")
    params_2k, ll_2k = fit_multi_kernel_hawkes(event_times_3k, T, 2)
    
    # Fit 3-kernel model
    print("Fitting 3-kernel model...")
    params_3k, ll_3k = fit_multi_kernel_hawkes(event_times_3k, T, 3)
    
    # Calculate AIC
    num_params_1k = 2 + 2 * 2 * 2 * 1  # mu (2) + alpha and beta matrices (4 each per kernel)
    num_params_2k = 2 + 2 * 2 * 2 * 2
    num_params_3k = 2 + 2 * 2 * 2 * 3
    
    aic_1k = calculate_aic(ll_1k, num_params_1k)
    aic_2k = calculate_aic(ll_2k, num_params_2k)
    aic_3k = calculate_aic(ll_3k, num_params_3k)
    
    print("\nModel comparison:")
    print(f"1-kernel - Log-likelihood: {ll_1k:.2f}, AIC: {aic_1k:.2f}")
    print(f"2-kernel - Log-likelihood: {ll_2k:.2f}, AIC: {aic_2k:.2f}")
    print(f"3-kernel - Log-likelihood: {ll_3k:.2f}, AIC: {aic_3k:.2f}")
    
    # Plot AIC values
    plt.figure(figsize=(8, 5))
    plt.bar(['1 Kernel', '2 Kernels', '3 Kernels'], [aic_1k, aic_2k, aic_3k], alpha=0.7)
    plt.title('AIC Comparison')
    plt.ylabel('AIC (lower is better)')
    plt.grid(True, alpha=0.3)
    plt.savefig('aic_comparison.png')
    plt.close()
    
    # Extract fitted parameters
    d = 2
    mu_1k = params_1k[:d]
    alpha_1k = params_1k[d:d+d*d].reshape((d, d))
    beta_1k = params_1k[d+d*d:d+2*d*d].reshape((d, d))
    
    mu_2k = params_2k[:d]
    alpha_2k_1 = params_2k[d:d+d*d].reshape((d, d))
    beta_2k_1 = params_2k[d+d*d:d+2*d*d].reshape((d, d))
    alpha_2k_2 = params_2k[d+2*d*d:d+3*d*d].reshape((d, d))
    beta_2k_2 = params_2k[d+3*d*d:d+4*d*d].reshape((d, d))
    
    mu_3k = params_3k[:d]
    alpha_3k_1 = params_3k[d:d+d*d].reshape((d, d))
    beta_3k_1 = params_3k[d+d*d:d+2*d*d].reshape((d, d))
    alpha_3k_2 = params_3k[d+2*d*d:d+3*d*d].reshape((d, d))
    beta_3k_2 = params_3k[d+3*d*d:d+4*d*d].reshape((d, d))
    alpha_3k_3 = params_3k[d+4*d*d:d+5*d*d].reshape((d, d))
    beta_3k_3 = params_3k[d+5*d*d:d+6*d*d].reshape((d, d))
    
    print("\nFitted parameters:")
    print("1-kernel model:")
    print(f"mu = {mu_1k}")
    print(f"alpha = \n{alpha_1k}")
    print(f"beta = \n{beta_1k}")
    
    print("\n2-kernel model:")
    print(f"mu = {mu_2k}")
    print(f"alpha_1 (UHF) = \n{alpha_2k_1}")
    print(f"beta_1 (UHF) = \n{beta_2k_1}")
    print(f"alpha_2 (VHF) = \n{alpha_2k_2}")
    print(f"beta_2 (VHF) = \n{beta_2k_2}")
    
    print("\n3-kernel model:")
    print(f"mu = {mu_3k}")
    print(f"alpha_1 (UHF) = \n{alpha_3k_1}")
    print(f"beta_1 (UHF) = \n{beta_3k_1}")
    print(f"alpha_2 (VHF) = \n{alpha_3k_2}")
    print(f"beta_2 (VHF) = \n{beta_3k_2}")
    print(f"alpha_3 (HF) = \n{alpha_3k_3}")
    print(f"beta_3 (HF) = \n{beta_3k_3}")
    
    # Compare original and fitted parameters
    print("\nComparison of original and fitted parameters (3-kernel model):")
    print("Original parameters:")
    print(f"mu = {mu}")
    print(f"alpha_1 (UHF) = \n{alpha1}")
    print(f"beta_1 (UHF) = \n{beta1}")
    print(f"alpha_2 (VHF) = \n{alpha2}")
    print(f"beta_2 (VHF) = \n{beta2}")
    print(f"alpha_3 (HF) = \n{alpha3}")
    print(f"beta_3 (HF) = \n{beta3}")
    
    # Create a trading strategy based on the multi-kernel model
    print("\nSimulating a trading strategy based on the multi-kernel model...")
    
    # Strategy: trade when the intensity of one direction significantly exceeds the other
    def simulate_trading_strategy(event_times, mu, alpha_matrices, beta_matrices, T, threshold=1.5):
        """
        Simulate a trading strategy based on intensity differences.
        
        Parameters:
        -----------
        event_times : list of arrays
            Event times for each dimension
        mu : array
            Base intensities
        alpha_matrices : list of arrays
            Alpha matrices for each kernel
        beta_matrices : list of arrays
            Beta matrices for each kernel
        T : float
            End time
        threshold : float, optional
            Threshold for intensity ratio to trigger a trade
        
        Returns:
        --------
        trade_times : array
            Times when trades are executed
        trade_positions : array
            Positions taken at each trade time (1 for long, -1 for short)
        pnl : array
            Cumulative profit and loss
        """
        d = len(event_times)
        K = len(alpha_matrices)
        
        # Combine all event times
        all_events = []
        for i in range(d):
            for t in event_times[i]:
                all_events.append((t, i))
        
        # Sort by time
        all_events.sort()
        
        # Initialize
        trade_times = []
        trade_positions = []
        current_position = 0
        
        # Initialize intensity components
        lambda_components = np.zeros((d, K, d))
        
        # Analyze each event
        prev_time = 0
        for event_time, event_dim in all_events:
            # Decay intensity components
            dt = event_time - prev_time
            for i in range(d):
                for k in range(K):
                    for j in range(d):
                        lambda_components[i, k, j] *= np.exp(-beta_matrices[k][i, j] * dt)
            
            # Compute total intensities
            lambda_up = mu[0] + np.sum(lambda_components[0, :, :])
            lambda_down = mu[1] + np.sum(lambda_components[1, :, :])
            
            # Trading logic
            if lambda_up / lambda_down > threshold and current_position <= 0:
                # Go long
                trade_times.append(event_time)
                trade_positions.append(1)
                current_position = 1
            elif lambda_down / lambda_up > threshold and current_position >= 0:
                # Go short
                trade_times.append(event_time)
                trade_positions.append(-1)
                current_position = -1
            
            # Update intensity components due to the event
            for i in range(d):
                for k in range(K):
                    lambda_components[i, k, event_dim] += alpha_matrices[k][i, event_dim]
            
            prev_time = event_time
        
        # Close position at the end if still open
        if current_position != 0:
            trade_times.append(T)
            trade_positions.append(0)
        
        # Convert to numpy arrays
        trade_times = np.array(trade_times)
        trade_positions = np.array(trade_positions)
        
        # Calculate PnL
        pnl = np.zeros(len(trade_times))
        
        # Get price at each trade time
        trade_prices = np.interp(trade_times, times_3k, prices_3k)
        
        # Calculate PnL for each trade
        position = 0
        entry_price = 0
        
        for i in range(len(trade_times)):
            new_position = trade_positions[i]
            
            # If position changes, calculate PnL
            if position != new_position:
                # Close previous position
                if position == 1:  # Long
                    pnl[i] = trade_prices[i] - entry_price
                elif position == -1:  # Short
                    pnl[i] = entry_price - trade_prices[i]
                
                # Open new position
                if new_position != 0:
                    entry_price = trade_prices[i]
                
                position = new_position
        
        # Cumulative PnL
        cum_pnl = np.cumsum(pnl)
        
        return trade_times, trade_positions, cum_pnl
    
    # Simulate trading strategy for all models
    trade_times_1k, trade_positions_1k, pnl_1k = simulate_trading_strategy(
        event_times_3k, mu, [alpha1], [beta1], T)
    
    trade_times_2k, trade_positions_2k, pnl_2k = simulate_trading_strategy(
        event_times_3k, mu, [alpha1, alpha2], [beta1, beta2], T)
    
    trade_times_3k, trade_positions_3k, pnl_3k = simulate_trading_strategy(
        event_times_3k, mu, [alpha1, alpha2, alpha3], [beta1, beta2, beta3], T)
    
    # Plot trading strategy results
    plt.figure(figsize=(12, 10))
    
    # Plot price process
    plt.subplot(2, 1, 1)
    plt.step(times_3k, prices_3k, where='post', color='black', alpha=0.5, label='Mid-Price')
    
    # Plot entry and exit points
    for i in range(len(trade_times_1k)):
        if i > 0 and trade_positions_1k[i-1] == 1:  # Long position
            plt.scatter(trade_times_1k[i], np.interp(trade_times_1k[i], times_3k, prices_3k), 
                        color='green', marker='^', s=100, label='_nolegend_')
        elif i > 0 and trade_positions_1k[i-1] == -1:  # Short position
            plt.scatter(trade_times_1k[i], np.interp(trade_times_1k[i], times_3k, prices_3k), 
                        color='red', marker='v', s=100, label='_nolegend_')
    
    plt.title('Trading Strategy Based on 1-Kernel Model')
    plt.ylabel('Price')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Plot PnL
    plt.subplot(2, 1, 2)
    plt.plot(trade_times_1k, pnl_1k, label='1-Kernel Model', color='blue')
    plt.plot(trade_times_2k, pnl_2k, label='2-Kernel Model', color='orange')
    plt.plot(trade_times_3k, pnl_3k, label='3-Kernel Model', color='green')
    plt.title('Cumulative PnL Comparison')
    plt.xlabel('Time')
    plt.ylabel('Cumulative PnL')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('trading_strategy.png')
    plt.close()
    
    # Print trading strategy statistics
    print("\nTrading Strategy Statistics:")
    print(f"1-Kernel Model - Number of trades: {len(trade_times_1k)-1}, Final PnL: {pnl_1k[-1]:.4f}")
    print(f"2-Kernel Model - Number of trades: {len(trade_times_2k)-1}, Final PnL: {pnl_2k[-1]:.4f}")
    print(f"3-Kernel Model - Number of trades: {len(trade_times_3k)-1}, Final PnL: {pnl_3k[-1]:.4f}")
    
    # Sharpe ratio
    if len(pnl_1k) > 1:
        returns_1k = np.diff(pnl_1k)
        sharpe_1k = np.mean(returns_1k) / np.std(returns_1k) * np.sqrt(252)  # Annualized
        print(f"1-Kernel Model - Sharpe Ratio: {sharpe_1k:.4f}")
    
    if len(pnl_2k) > 1:
        returns_2k = np.diff(pnl_2k)
        sharpe_2k = np.mean(returns_2k) / np.std(returns_2k) * np.sqrt(252)
        print(f"2-Kernel Model - Sharpe Ratio: {sharpe_2k:.4f}")
    
    if len(pnl_3k) > 1:
        returns_3k = np.diff(pnl_3k)
        sharpe_3k = np.mean(returns_3k) / np.std(returns_3k) * np.sqrt(252)
        print(f"3-Kernel Model - Sharpe Ratio: {sharpe_3k:.4f}")
    
    print("\nSimulation and analysis complete!")

if __name__ == "__main__":
    main()