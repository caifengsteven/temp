import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import minimize_scalar
from statsmodels.distributions.empirical_distribution import ECDF
import yfinance as yf
from datetime import datetime, timedelta
import seaborn as sns
from tqdm import tqdm

# Set random seed for reproducibility
np.random.seed(42)

# Define Gaussian kernel
def gaussian_kernel(x, h):
    """Gaussian kernel function with bandwidth h"""
    return 1/(h*np.sqrt(2*np.pi)) * np.exp(-0.5*(x/h)**2)

# Nadaraya-Watson kernel estimator
def kernel_regression(x, y, x_grid, h):
    """
    Compute kernel regression estimator using Gaussian kernel
    
    Parameters:
    x : array_like
        Independent variable values
    y : array_like
        Dependent variable values
    x_grid : array_like
        Points at which to evaluate the regression
    h : float
        Bandwidth parameter
        
    Returns:
    array_like
        Estimated function values at x_grid points
    """
    y_hat = np.zeros_like(x_grid, dtype=float)
    
    for i, x0 in enumerate(x_grid):
        weights = gaussian_kernel((x - x0) / h, 1.0)
        y_hat[i] = np.sum(weights * y) / np.sum(weights)
    
    return y_hat

# Cross-validation function for bandwidth selection
def cv_bandwidth(x, y, h_grid):
    """
    Select bandwidth using leave-one-out cross-validation
    
    Parameters:
    x : array_like
        Independent variable values
    y : array_like
        Dependent variable values
    h_grid : array_like
        Grid of bandwidth values to try
        
    Returns:
    float
        Optimal bandwidth value
    """
    n = len(x)
    cv_scores = np.zeros_like(h_grid)
    
    for i, h in enumerate(h_grid):
        cv_error = 0
        
        for j in range(n):
            # Leave out j-th observation
            x_j = np.delete(x, j)
            y_j = np.delete(y, j)
            
            # Compute prediction at x[j]
            weights = gaussian_kernel((x_j - x[j]) / h, 1.0)
            if np.sum(weights) == 0:
                # Avoid division by zero
                y_hat_j = y[j]
            else:
                y_hat_j = np.sum(weights * y_j) / np.sum(weights)
            
            # Accumulate squared error
            cv_error += (y[j] - y_hat_j)**2
        
        cv_scores[i] = cv_error / n
    
    # Return bandwidth with minimum cross-validation score
    return h_grid[np.argmin(cv_scores)]

# Find local extrema in the kernel regression
def find_extrema(x, y_smooth):
    """
    Find local extrema in a smoothed time series
    
    Parameters:
    x : array_like
        Time points
    y_smooth : array_like
        Smoothed values
        
    Returns:
    tuple
        Lists of extrema indices, types (1 for max, -1 for min), and values
    """
    n = len(y_smooth)
    extrema_idx = []
    extrema_type = []
    extrema_val = []
    
    # Check for local extrema
    for i in range(1, n-1):
        if y_smooth[i-1] < y_smooth[i] and y_smooth[i] > y_smooth[i+1]:
            # Local maximum
            extrema_idx.append(i)
            extrema_type.append(1)
            extrema_val.append(y_smooth[i])
        elif y_smooth[i-1] > y_smooth[i] and y_smooth[i] < y_smooth[i+1]:
            # Local minimum
            extrema_idx.append(i)
            extrema_type.append(-1)
            extrema_val.append(y_smooth[i])
    
    return extrema_idx, extrema_type, extrema_val

# Define technical patterns based on local extrema
def identify_patterns(extrema_idx, extrema_type, extrema_val, prices, window_size=38):
    """
    Identify technical patterns in price series
    
    Parameters:
    extrema_idx : list
        Indices of extrema
    extrema_type : list
        Types of extrema (1 for max, -1 for min)
    extrema_val : list
        Values of extrema
    prices : array_like
        Original price series
    window_size : int
        Size of window to consider for pattern detection
        
    Returns:
    dict
        Dictionary of detected patterns with their indices
    """
    patterns = {
        'HS': [],       # Head and Shoulders
        'IHS': [],      # Inverse Head and Shoulders
        'BTOP': [],     # Broadening Top
        'BBOT': [],     # Broadening Bottom
        'TTOP': [],     # Triangle Top
        'TBOT': [],     # Triangle Bottom
        'RTOP': [],     # Rectangle Top
        'RBOT': [],     # Rectangle Bottom
        'DTOP': [],     # Double Top
        'DBOT': []      # Double Bottom
    }
    
    n = len(extrema_idx)
    
    # We need at least 5 consecutive extrema for most patterns
    for i in range(n-4):
        # Extract 5 consecutive extrema
        e_idx = extrema_idx[i:i+5]
        e_type = extrema_type[i:i+5]
        e_val = extrema_val[i:i+5]
        
        # Skip if the extrema are not within a reasonable window
        if e_idx[-1] - e_idx[0] > window_size:
            continue
        
        # Check for head-and-shoulders pattern
        if (e_type[0] == 1 and e_type[1] == -1 and e_type[2] == 1 and 
            e_type[3] == -1 and e_type[4] == 1 and
            e_val[2] > e_val[0] and e_val[2] > e_val[4] and
            abs(e_val[0] - e_val[4]) < 0.015 * ((e_val[0] + e_val[4])/2) and
            abs(e_val[1] - e_val[3]) < 0.015 * ((e_val[1] + e_val[3])/2)):
            patterns['HS'].append(e_idx[-1])
        
        # Check for inverse head-and-shoulders pattern
        if (e_type[0] == -1 and e_type[1] == 1 and e_type[2] == -1 and 
            e_type[3] == 1 and e_type[4] == -1 and
            e_val[2] < e_val[0] and e_val[2] < e_val[4] and
            abs(e_val[0] - e_val[4]) < 0.015 * ((e_val[0] + e_val[4])/2) and
            abs(e_val[1] - e_val[3]) < 0.015 * ((e_val[1] + e_val[3])/2)):
            patterns['IHS'].append(e_idx[-1])
        
        # Check for broadening top
        if (e_type[0] == 1 and e_type[1] == -1 and e_type[2] == 1 and 
            e_type[3] == -1 and e_type[4] == 1 and
            e_val[0] < e_val[2] and e_val[2] < e_val[4] and
            e_val[1] > e_val[3]):
            patterns['BTOP'].append(e_idx[-1])
        
        # Check for broadening bottom
        if (e_type[0] == -1 and e_type[1] == 1 and e_type[2] == -1 and 
            e_type[3] == 1 and e_type[4] == -1 and
            e_val[0] > e_val[2] and e_val[2] > e_val[4] and
            e_val[1] < e_val[3]):
            patterns['BBOT'].append(e_idx[-1])
        
        # Check for triangle top
        if (e_type[0] == 1 and e_type[1] == -1 and e_type[2] == 1 and 
            e_type[3] == -1 and e_type[4] == 1 and
            e_val[0] > e_val[2] and e_val[2] > e_val[4] and
            e_val[1] < e_val[3]):
            patterns['TTOP'].append(e_idx[-1])
        
        # Check for triangle bottom
        if (e_type[0] == -1 and e_type[1] == 1 and e_type[2] == -1 and 
            e_type[3] == 1 and e_type[4] == -1 and
            e_val[0] < e_val[2] and e_val[2] < e_val[4] and
            e_val[1] > e_val[3]):
            patterns['TBOT'].append(e_idx[-1])
        
        # Check for rectangle top
        if (e_type[0] == 1 and e_type[1] == -1 and e_type[2] == 1 and 
            e_type[3] == -1 and e_type[4] == 1 and
            abs(max(e_val[0], e_val[2], e_val[4]) - min(e_val[0], e_val[2], e_val[4])) < 0.0075 * ((e_val[0] + e_val[2] + e_val[4])/3) and
            abs(max(e_val[1], e_val[3]) - min(e_val[1], e_val[3])) < 0.0075 * ((e_val[1] + e_val[3])/2) and
            min(e_val[0], e_val[2], e_val[4]) > max(e_val[1], e_val[3])):
            patterns['RTOP'].append(e_idx[-1])
        
        # Check for rectangle bottom
        if (e_type[0] == -1 and e_type[1] == 1 and e_type[2] == -1 and 
            e_type[3] == 1 and e_type[4] == -1 and
            abs(max(e_val[0], e_val[2], e_val[4]) - min(e_val[0], e_val[2], e_val[4])) < 0.0075 * ((e_val[0] + e_val[2] + e_val[4])/3) and
            abs(max(e_val[1], e_val[3]) - min(e_val[1], e_val[3])) < 0.0075 * ((e_val[1] + e_val[3])/2) and
            min(e_val[1], e_val[3]) > max(e_val[0], e_val[2], e_val[4])):
            patterns['RBOT'].append(e_idx[-1])
    
    # For double tops and bottoms, we need a different approach
    n_prices = len(prices)
    
    # Check each peak/trough as potential first part of double top/bottom
    for i in range(n):
        idx = extrema_idx[i]
        typ = extrema_type[i]
        val = extrema_val[i]
        
        if typ == 1:  # Maximum (potential first peak of double top)
            # Look for another peak within reasonable distance and height
            for j in range(i+1, n):
                if extrema_type[j] == 1:  # Another maximum
                    idx2 = extrema_idx[j]
                    val2 = extrema_val[j]
                    
                    # Check if peaks are similar in height and far enough apart
                    if (abs(val - val2) < 0.015 * ((val + val2)/2) and
                        idx2 - idx >= 22):  # At least 22 days apart
                        patterns['DTOP'].append(idx2)
                        break
        
        elif typ == -1:  # Minimum (potential first trough of double bottom)
            # Look for another trough within reasonable distance and height
            for j in range(i+1, n):
                if extrema_type[j] == -1:  # Another minimum
                    idx2 = extrema_idx[j]
                    val2 = extrema_val[j]
                    
                    # Check if troughs are similar in height and far enough apart
                    if (abs(val - val2) < 0.015 * ((val + val2)/2) and
                        idx2 - idx >= 22):  # At least 22 days apart
                        patterns['DBOT'].append(idx2)
                        break
    
    return patterns

# Analyze returns conditional on patterns
def analyze_pattern_returns(prices, patterns, days_after=3, normalize=True):
    """
    Analyze returns following the occurrence of each pattern
    
    Parameters:
    prices : array_like
        Price series
    patterns : dict
        Dictionary of detected patterns with their indices
    days_after : int
        Number of days after pattern completion to compute returns
    normalize : bool
        Whether to normalize returns by subtracting mean and dividing by std
        
    Returns:
    dict
        Dictionary of returns for each pattern
    """
    # Calculate daily returns
    returns = np.diff(np.log(prices))
    
    # If normalize, standardize returns
    if normalize:
        returns = (returns - np.mean(returns)) / np.std(returns)
    
    # Dictionary to store conditional returns
    conditional_returns = {}
    
    # For each pattern type
    for pattern_type, pattern_indices in patterns.items():
        pattern_returns = []
        
        # For each occurrence of the pattern
        for idx in pattern_indices:
            # Check if we have enough data to calculate forward returns
            if idx + days_after < len(returns):
                # Get return days_after days after pattern completion
                pattern_returns.append(returns[idx + days_after])
        
        conditional_returns[pattern_type] = np.array(pattern_returns)
    
    return conditional_returns

# Function to calculate performance metrics
def calculate_performance_metrics(returns):
    """
    Calculate performance metrics for a return series
    
    Parameters:
    returns : array_like
        Return series
        
    Returns:
    dict
        Dictionary of performance metrics
    """
    metrics = {
        'Mean': np.mean(returns),
        'Median': np.median(returns),
        'StdDev': np.std(returns),
        'Skewness': 0 if len(returns) == 0 else (np.mean((returns - np.mean(returns))**3) / 
                                                (np.std(returns)**3)),
        'Kurtosis': 0 if len(returns) == 0 else (np.mean((returns - np.mean(returns))**4) / 
                                               (np.std(returns)**4) - 3),
        'Count': len(returns)
    }
    
    return metrics

# Statistical tests for pattern informativeness
def kolmogorov_smirnov_test(conditional_returns, unconditional_returns):
    """
    Perform Kolmogorov-Smirnov test to compare conditional and unconditional return distributions
    
    Parameters:
    conditional_returns : dict
        Dictionary of returns conditional on each pattern
    unconditional_returns : array_like
        Unconditional returns
        
    Returns:
    dict
        Dictionary of test statistics and p-values for each pattern
    """
    from scipy import stats
    
    results = {}
    
    for pattern, returns in conditional_returns.items():
        if len(returns) > 0:
            # Perform KS test
            ks_stat, p_value = stats.ks_2samp(returns, unconditional_returns)
            results[pattern] = {'KS_Stat': ks_stat, 'p_value': p_value}
        else:
            results[pattern] = {'KS_Stat': None, 'p_value': None}
    
    return results

# Function to compute quantiles and chi-square test
def goodness_of_fit_test(conditional_returns, unconditional_returns):
    """
    Perform goodness-of-fit test to compare conditional and unconditional return distributions
    
    Parameters:
    conditional_returns : dict
        Dictionary of returns conditional on each pattern
    unconditional_returns : array_like
        Unconditional returns
        
    Returns:
    dict
        Dictionary of test results for each pattern
    """
    from scipy import stats
    
    results = {}
    
    # Calculate unconditional deciles
    deciles = np.percentile(unconditional_returns, np.arange(10, 100, 10))
    
    for pattern, returns in conditional_returns.items():
        if len(returns) < 10:  # Need enough data for meaningful test
            results[pattern] = {'Q_stat': None, 'p_value': None, 'decile_freqs': None}
            continue
        
        # Count returns in each decile
        observed = np.zeros(10)
        for i in range(9):
            if i == 0:
                observed[i] = np.sum(returns <= deciles[i])
            else:
                observed[i] = np.sum((returns > deciles[i-1]) & (returns <= deciles[i]))
        observed[9] = np.sum(returns > deciles[8])
        
        # Expected counts (10% in each decile)
        expected = np.ones(10) * len(returns) / 10
        
        # Chi-square test
        chi2_stat = np.sum((observed - expected)**2 / expected)
        p_value = 1 - stats.chi2.cdf(chi2_stat, df=9)
        
        # Relative frequencies
        rel_freq = observed / len(returns)
        
        results[pattern] = {
            'Q_stat': chi2_stat,
            'p_value': p_value,
            'decile_freqs': rel_freq
        }
    
    return results

# Function to plot pattern detection
def plot_pattern_example(prices, smooth_prices, extrema_idx, extrema_type, pattern_type, pattern_idx):
    """
    Plot an example of a detected pattern
    
    Parameters:
    prices : array_like
        Original price series
    smooth_prices : array_like
        Smoothed price series
    extrema_idx : list
        Indices of extrema
    extrema_type : list
        Types of extrema (1 for max, -1 for min)
    pattern_type : str
        Type of pattern to plot
    pattern_idx : int
        Index of pattern in the time series
    """
    # Find the pattern in the extrema list
    pattern_extrema = []
    extrema_values = []
    
    # For head-and-shoulders and other 5-point patterns
    if pattern_type in ['HS', 'IHS', 'BTOP', 'BBOT', 'TTOP', 'TBOT', 'RTOP', 'RBOT']:
        # Find the index in extrema_idx that corresponds to pattern_idx
        for i in range(len(extrema_idx)):
            if extrema_idx[i] == pattern_idx:
                # Get the 5 extrema that form this pattern
                start_i = max(0, i-4)
                pattern_extrema = extrema_idx[start_i:i+1]
                pattern_types = extrema_type[start_i:i+1]
                extrema_values = [smooth_prices[idx] for idx in pattern_extrema]
                break
    # For double tops/bottoms
    elif pattern_type in ['DTOP', 'DBOT']:
        # These are more complex - find two similar extrema
        for i in range(len(extrema_idx)):
            if extrema_idx[i] == pattern_idx:
                # This is the second extremum
                second_idx = i
                second_val = smooth_prices[extrema_idx[i]]
                second_type = extrema_type[i]
                
                # Find the first extremum of the same type
                for j in range(i):
                    if (extrema_type[j] == second_type and 
                        abs(smooth_prices[extrema_idx[j]] - second_val) < 0.015 * ((smooth_prices[extrema_idx[j]] + second_val)/2) and
                        extrema_idx[i] - extrema_idx[j] >= 22):
                        pattern_extrema = [extrema_idx[j], extrema_idx[i]]
                        extrema_values = [smooth_prices[extrema_idx[j]], smooth_prices[extrema_idx[i]]]
                        break
                break
    
    if not pattern_extrema:
        print(f"No {pattern_type} pattern found at index {pattern_idx}")
        return
    
    # Plot the pattern
    plt.figure(figsize=(12, 6))
    
    # Plot original prices
    plt.plot(prices, 'b-', alpha=0.3, label='Original Prices')
    
    # Plot smoothed prices
    plt.plot(smooth_prices, 'g-', label='Smoothed Prices')
    
    # Mark the extrema points
    for i, idx in enumerate(pattern_extrema):
        if pattern_type in ['HS', 'IHS', 'BTOP', 'BBOT', 'TTOP', 'TBOT', 'RTOP', 'RBOT']:
            if i < len(pattern_extrema) - 1:
                marker = 'ro' if pattern_types[i] == 1 else 'go'
            else:
                marker = 'ro' if pattern_types[i] == 1 else 'go'
        else:  # DTOP, DBOT
            marker = 'ro' if pattern_type == 'DTOP' else 'go'
        
        plt.plot(idx, extrema_values[i], marker, markersize=10)
    
    # Add vertical line at pattern completion
    plt.axvline(x=pattern_idx, color='k', linestyle='--', alpha=0.5)
    
    # Set title and labels
    pattern_names = {
        'HS': 'Head-and-Shoulders',
        'IHS': 'Inverse Head-and-Shoulders',
        'BTOP': 'Broadening Top',
        'BBOT': 'Broadening Bottom',
        'TTOP': 'Triangle Top',
        'TBOT': 'Triangle Bottom',
        'RTOP': 'Rectangle Top',
        'RBOT': 'Rectangle Bottom',
        'DTOP': 'Double Top',
        'DBOT': 'Double Bottom'
    }
    
    plt.title(f"{pattern_names[pattern_type]} Pattern")
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.show()

# Function to simulate price data with different processes
def simulate_price_data(n_days=1000, process='gbm', mean=0.0001, volatility=0.01, 
                       jump_rate=0.05, jump_mean=0, jump_std=0.02, 
                       regime_change_prob=0.005, high_vol_factor=3):
    """
    Simulate price data with different processes
    
    Parameters:
    n_days : int
        Number of days to simulate
    process : str
        Type of process ('gbm', 'jump_diffusion', 'regime_switching')
    mean : float
        Mean daily return
    volatility : float
        Daily volatility
    jump_rate : float
        Probability of a jump on any given day
    jump_mean : float
        Mean of jump size
    jump_std : float
        Standard deviation of jump size
    regime_change_prob : float
        Probability of switching volatility regime
    high_vol_factor : float
        Factor by which volatility increases in high vol regime
        
    Returns:
    array_like
        Simulated price series
    """
    prices = np.zeros(n_days)
    returns = np.zeros(n_days)
    prices[0] = 100.0  # Starting price
    
    if process == 'gbm':
        # Geometric Brownian Motion
        for t in range(1, n_days):
            returns[t] = mean + volatility * np.random.normal()
            prices[t] = prices[t-1] * np.exp(returns[t])
    
    elif process == 'jump_diffusion':
        # Jump Diffusion Process
        for t in range(1, n_days):
            # Normal diffusion component
            diffusion = mean + volatility * np.random.normal()
            
            # Jump component
            jump = 0
            if np.random.random() < jump_rate:
                jump = np.random.normal(jump_mean, jump_std)
            
            returns[t] = diffusion + jump
            prices[t] = prices[t-1] * np.exp(returns[t])
    
    elif process == 'regime_switching':
        # Regime Switching Model (low/high volatility)
        high_vol_regime = False
        current_vol = volatility
        
        for t in range(1, n_days):
            # Check for regime switch
            if np.random.random() < regime_change_prob:
                high_vol_regime = not high_vol_regime
                current_vol = volatility * high_vol_factor if high_vol_regime else volatility
            
            returns[t] = mean + current_vol * np.random.normal()
            prices[t] = prices[t-1] * np.exp(returns[t])
    
    elif process == 'trending':
        # Trending with mean reversion
        trend = 0
        trend_change_prob = 0.01
        trend_mean_reversion = 0.02
        trend_volatility = 0.001
        
        for t in range(1, n_days):
            # Update trend
            if np.random.random() < trend_change_prob:
                trend = np.random.normal(0, 0.002)
            else:
                trend = trend * (1 - trend_mean_reversion) + np.random.normal(0, trend_volatility)
            
            returns[t] = mean + trend + volatility * np.random.normal()
            prices[t] = prices[t-1] * np.exp(returns[t])
    
    else:
        raise ValueError(f"Unknown process type: {process}")
    
    return prices

# Function to run a complete pattern detection and analysis pipeline
def run_pattern_detection(prices, window_size=38, bandwidth_factor=0.3):
    """
    Run the complete pattern detection and analysis pipeline
    
    Parameters:
    prices : array_like
        Price series
    window_size : int
        Size of window for pattern detection
    bandwidth_factor : float
        Factor to multiply optimal bandwidth by
        
    Returns:
    tuple
        (detected patterns, conditional returns, performance metrics, statistical tests)
    """
    # Time points
    x = np.arange(len(prices))
    
    # Select bandwidth using cross-validation
    h_grid = np.linspace(1, 20, 20)
    h_opt = cv_bandwidth(x, prices, h_grid)
    h = h_opt * bandwidth_factor  # Adjust bandwidth based on factor
    
    # Perform kernel regression to smooth prices
    smooth_prices = kernel_regression(x, prices, x, h)
    
    # Find local extrema in the smoothed price series
    extrema_idx, extrema_type, extrema_val = find_extrema(x, smooth_prices)
    
    # Identify technical patterns
    patterns = identify_patterns(extrema_idx, extrema_type, smooth_prices, prices, window_size)
    
    # Analyze returns conditional on patterns
    conditional_returns = analyze_pattern_returns(prices, patterns, days_after=3)
    
    # Calculate normalized returns for the entire series
    unconditional_returns = np.diff(np.log(prices))
    normalized_returns = (unconditional_returns - np.mean(unconditional_returns)) / np.std(unconditional_returns)
    
    # Calculate performance metrics
    performance_metrics = {}
    performance_metrics['Unconditional'] = calculate_performance_metrics(normalized_returns)
    
    for pattern, returns in conditional_returns.items():
        performance_metrics[pattern] = calculate_performance_metrics(returns)
    
    # Perform statistical tests
    ks_tests = kolmogorov_smirnov_test(conditional_returns, normalized_returns)
    gof_tests = goodness_of_fit_test(conditional_returns, normalized_returns)
    
    return (patterns, conditional_returns, performance_metrics, 
            {'ks_tests': ks_tests, 'gof_tests': gof_tests}, 
            smooth_prices, extrema_idx, extrema_type, extrema_val)

# Function to visualize pattern returns compared to unconditional returns
def visualize_returns_distribution(conditional_returns, unconditional_returns, pattern_type):
    """
    Visualize the distribution of returns conditional on a pattern vs unconditional returns
    
    Parameters:
    conditional_returns : dict
        Dictionary of returns conditional on each pattern
    unconditional_returns : array_like
        Unconditional returns
    pattern_type : str
        Type of pattern to visualize
    """
    if pattern_type not in conditional_returns or len(conditional_returns[pattern_type]) == 0:
        print(f"No returns data available for pattern {pattern_type}")
        return
    
    plt.figure(figsize=(12, 6))
    
    # Plot kernel density estimate
    sns.kdeplot(unconditional_returns, label='Unconditional', color='gray')
    sns.kdeplot(conditional_returns[pattern_type], label=f'Conditional on {pattern_type}')
    
    # Plot empirical CDFs
    plt.figure(figsize=(12, 6))
    
    # Compute ECDFs
    ecdf_uncond = ECDF(unconditional_returns)
    x_uncond = np.linspace(min(unconditional_returns), max(unconditional_returns), 1000)
    y_uncond = ecdf_uncond(x_uncond)
    
    ecdf_cond = ECDF(conditional_returns[pattern_type])
    x_cond = np.linspace(min(conditional_returns[pattern_type]), max(conditional_returns[pattern_type]), 1000)
    y_cond = ecdf_cond(x_cond)
    
    # Plot ECDFs
    plt.plot(x_uncond, y_uncond, label='Unconditional', color='gray')
    plt.plot(x_cond, y_cond, label=f'Conditional on {pattern_type}')
    
    plt.title(f'Empirical CDF: {pattern_type} vs Unconditional Returns')
    plt.xlabel('Return')
    plt.ylabel('Cumulative Probability')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.show()

# Function to analyze pattern frequency across different price processes
def analyze_pattern_frequency(n_simulations=5, n_days=1000, processes=['gbm', 'jump_diffusion', 'regime_switching', 'trending']):
    """
    Analyze pattern frequency across different price processes
    
    Parameters:
    n_simulations : int
        Number of simulations per process
    n_days : int
        Number of days to simulate
    processes : list
        List of process types to simulate
        
    Returns:
    DataFrame
        Data frame of pattern frequencies by process
    """
    results = []
    
    for process in processes:
        for i in range(n_simulations):
            # Simulate price data
            prices = simulate_price_data(n_days=n_days, process=process)
            
            # Run pattern detection
            patterns, _, _, _, _, _, _, _ = run_pattern_detection(prices)
            
            # Count patterns
            pattern_counts = {k: len(v) for k, v in patterns.items()}
            pattern_counts['process'] = process
            pattern_counts['simulation'] = i+1
            
            results.append(pattern_counts)
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    return df

# Function to compare pattern returns across different processes
def compare_pattern_returns(n_simulations=5, n_days=1000, processes=['gbm', 'jump_diffusion', 'regime_switching', 'trending']):
    """
    Compare pattern returns across different price processes
    
    Parameters:
    n_simulations : int
        Number of simulations per process
    n_days : int
        Number of days to simulate
    processes : list
        List of process types to simulate
        
    Returns:
    DataFrame
        Data frame of pattern returns by process
    """
    results = []
    
    for process in processes:
        for i in range(n_simulations):
            # Simulate price data
            prices = simulate_price_data(n_days=n_days, process=process)
            
            # Run pattern detection
            patterns, conditional_returns, performance_metrics, statistical_tests, _, _, _, _ = run_pattern_detection(prices)
            
            # Store return metrics
            for pattern, metrics in performance_metrics.items():
                if pattern != 'Unconditional':
                    entry = {
                        'process': process,
                        'simulation': i+1,
                        'pattern': pattern,
                        'count': metrics['Count'],
                        'mean': metrics['Mean'],
                        'median': metrics['Median'],
                        'stddev': metrics['StdDev'],
                        'skewness': metrics['Skewness'],
                        'kurtosis': metrics['Kurtosis']
                    }
                    
                    # Add statistical test p-values if available
                    if pattern in statistical_tests['ks_tests'] and statistical_tests['ks_tests'][pattern]['p_value'] is not None:
                        entry['ks_pvalue'] = statistical_tests['ks_tests'][pattern]['p_value']
                    else:
                        entry['ks_pvalue'] = np.nan
                    
                    if pattern in statistical_tests['gof_tests'] and statistical_tests['gof_tests'][pattern]['p_value'] is not None:
                        entry['gof_pvalue'] = statistical_tests['gof_tests'][pattern]['p_value']
                    else:
                        entry['gof_pvalue'] = np.nan
                    
                    results.append(entry)
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    return df

# Function to test pattern trading strategy
def test_pattern_trading_strategy(prices, patterns, holding_period=5, transaction_cost=0.001):
    """
    Test a simple pattern-based trading strategy
    
    Parameters:
    prices : array_like
        Price series
    patterns : dict
        Dictionary of detected patterns with their indices
    holding_period : int
        Number of days to hold position after pattern
    transaction_cost : float
        Transaction cost per trade (one-way)
        
    Returns:
    dict
        Dictionary of strategy performance metrics
    """
    # Calculate log returns
    log_returns = np.diff(np.log(prices))
    
    # Initialize strategy returns
    strategy_returns = np.zeros_like(log_returns)
    
    # Track positions
    in_position = False
    entry_price = 0
    entry_idx = 0
    
    # Define trading rules for each pattern
    bullish_patterns = ['IHS', 'BBOT', 'TBOT', 'RBOT', 'DBOT']
    bearish_patterns = ['HS', 'BTOP', 'TTOP', 'RTOP', 'DTOP']
    
    # Track trades
    trades = []
    
    # Iterate through each day
    for t in range(len(log_returns)):
        # Check if we need to exit an existing position
        if in_position and t - entry_idx >= holding_period:
            # Exit position
            exit_return = np.sum(log_returns[entry_idx+1:t+1]) - transaction_cost
            trade = {
                'entry_idx': entry_idx,
                'exit_idx': t,
                'pattern': current_pattern,
                'direction': current_direction,
                'return': exit_return
            }
            trades.append(trade)
            
            in_position = False
        
        # Check for new pattern signals if not in a position
        if not in_position:
            # Check each pattern type
            for pattern in bullish_patterns:
                if pattern in patterns and t in patterns[pattern]:
                    # Bullish pattern - go long
                    in_position = True
                    entry_price = prices[t]
                    entry_idx = t
                    current_pattern = pattern
                    current_direction = 'long'
                    break
            
            if not in_position:  # Still not in position, check bearish patterns
                for pattern in bearish_patterns:
                    if pattern in patterns and t in patterns[pattern]:
                        # Bearish pattern - go short
                        in_position = True
                        entry_price = prices[t]
                        entry_idx = t
                        current_pattern = pattern
                        current_direction = 'short'
                        break
    
    # Close any remaining position at the end
    if in_position:
        t = len(log_returns) - 1
        exit_return = np.sum(log_returns[entry_idx+1:t+1]) - transaction_cost
        trade = {
            'entry_idx': entry_idx,
            'exit_idx': t,
            'pattern': current_pattern,
            'direction': current_direction,
            'return': exit_return
        }
        trades.append(trade)
    
    # Calculate strategy performance
    if not trades:
        return {
            'total_return': 0,
            'num_trades': 0,
            'win_rate': 0,
            'avg_return': 0,
            'sharpe': 0,
            'trades': trades
        }
    
    trade_returns = [trade['return'] for trade in trades]
    win_rate = np.mean([r > 0 for r in trade_returns])
    
    return {
        'total_return': np.sum(trade_returns),
        'num_trades': len(trades),
        'win_rate': win_rate,
        'avg_return': np.mean(trade_returns),
        'sharpe': np.mean(trade_returns) / np.std(trade_returns) if len(trade_returns) > 1 else 0,
        'trades': trades
    }

# Main function to run all the analysis
def main():
    """Main function to run all the analysis"""
    
    # 1. Simulated Data Analysis
    print("Analyzing pattern frequency across different price processes...")
    frequency_df = analyze_pattern_frequency(n_simulations=3, n_days=1000)
    
    # Calculate average pattern count per process
    avg_frequency = frequency_df.groupby('process').mean()
    print("\nAverage pattern frequency by process:")
    print(avg_frequency.drop(columns=['simulation']))
    
    # 2. Pattern Return Analysis
    print("\nComparing pattern returns across different processes...")
    returns_df = compare_pattern_returns(n_simulations=3, n_days=1000)
    
    # Calculate average pattern return metrics by process and pattern
    avg_returns = returns_df.groupby(['process', 'pattern']).mean()
    print("\nStatistically significant pattern returns (KS test p-value < 0.05):")
    significant_patterns = returns_df[returns_df['ks_pvalue'] < 0.05]
    if len(significant_patterns) > 0:
        print(significant_patterns[['process', 'pattern', 'mean', 'ks_pvalue']])
    else:
        print("No statistically significant patterns found.")
    
    # 3. Pattern Trading Strategy Test
    print("\nTesting pattern trading strategy on simulated data...")
    processes = ['gbm', 'jump_diffusion', 'regime_switching', 'trending']
    strategy_results = []
    
    for process in processes:
        # Simulate price data
        prices = simulate_price_data(n_days=1000, process=process)
        
        # Run pattern detection
        patterns, _, _, _, _, _, _, _ = run_pattern_detection(prices)
        
        # Test trading strategy
        strategy_performance = test_pattern_trading_strategy(prices, patterns)
        
        strategy_results.append({
            'process': process,
            'total_return': strategy_performance['total_return'],
            'num_trades': strategy_performance['num_trades'],
            'win_rate': strategy_performance['win_rate'],
            'avg_return': strategy_performance['avg_return'],
            'sharpe': strategy_performance['sharpe']
        })
    
    strategy_df = pd.DataFrame(strategy_results)
    print("\nPattern trading strategy performance by process:")
    print(strategy_df)
    
    # 4. Detailed Analysis of a Single Simulation
    print("\nDetailed analysis of a single simulation (regime switching process)...")
    prices = simulate_price_data(n_days=1000, process='regime_switching')
    
    # Run pattern detection
    patterns, conditional_returns, performance_metrics, statistical_tests, smooth_prices, extrema_idx, extrema_type, extrema_val = run_pattern_detection(prices)
    
    # Print pattern frequency
    print("\nPattern frequency:")
    for pattern, indices in patterns.items():
        print(f"{pattern}: {len(indices)} occurrences")
    
    # Print return metrics
    print("\nReturn metrics:")
    metrics_df = pd.DataFrame({k: v for k, v in performance_metrics.items() if k != 'Unconditional'})
    print(metrics_df.T[['Mean', 'StdDev', 'Count']])
    
    # Print statistical test results
    print("\nStatistical test results:")
    for pattern in patterns.keys():
        if pattern in statistical_tests['ks_tests'] and statistical_tests['ks_tests'][pattern]['p_value'] is not None:
            ks_pvalue = statistical_tests['ks_tests'][pattern]['p_value']
            print(f"{pattern}: KS test p-value = {ks_pvalue:.4f} {'*' if ks_pvalue < 0.05 else ''}")
    
    # Plot example of each pattern type
    print("\nPlotting examples of detected patterns...")
    for pattern_type, indices in patterns.items():
        if indices:
            # Plot the first occurrence of each pattern type
            plot_pattern_example(prices, smooth_prices, extrema_idx, extrema_type, pattern_type, indices[0])
    
    # Visualize returns distribution for a selected pattern
    for pattern_type in ['HS', 'IHS', 'DTOP', 'DBOT']:
        if pattern_type in conditional_returns and len(conditional_returns[pattern_type]) > 0:
            print(f"\nVisualizing returns distribution for {pattern_type}...")
            visualize_returns_distribution(
                conditional_returns, 
                (np.diff(np.log(prices)) - np.mean(np.diff(np.log(prices)))) / np.std(np.diff(np.log(prices))),
                pattern_type
            )
            break
    
    print("\nAnalysis complete!")

# Run the main function if executed directly
if __name__ == "__main__":
    main()