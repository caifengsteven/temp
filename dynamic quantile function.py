import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.optimize import minimize
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# Set random seed for reproducibility
np.random.seed(42)

#######################################################################
# Part 1: G-and-H Distribution Functions
#######################################################################

def g_and_h_quantile(u, a, b, g, h):
    """
    g-and-h quantile function
    
    Parameters:
    u: quantile level (between 0 and 1)
    a: location parameter
    b: scale parameter
    g: asymmetry parameter
    h: tail heaviness parameter
    
    Returns:
    quantile value
    """
    z = stats.norm.ppf(u)
    
    # Handle extreme values to prevent numerical issues
    if np.isnan(z):
        if u < 0.5:
            z = -8.0
        else:
            z = 8.0
    
    if abs(g) < 1e-10:  # g is approximately 0
        return a + b * z * np.exp(h * z**2 / 2)
    else:
        G = (np.exp(g * z) - 1) / g
        H = np.exp(h * z**2 / 2)
        return a + b * G * H

def g_and_h_quantile_vectorized(u, a, b, g, h):
    """Vectorized version of g_and_h_quantile"""
    z = stats.norm.ppf(u)
    
    # Handle extreme values to prevent numerical issues
    z = np.clip(z, -8.0, 8.0)
    
    if abs(g) < 1e-10:  # g is approximately 0
        return a + b * z * np.exp(h * z**2 / 2)
    else:
        G = (np.exp(g * z) - 1) / g
        H = np.exp(h * z**2 / 2)
        return a + b * G * H

def generate_g_and_h_sample(n, a, b, g, h):
    """Generate a random sample from g-and-h distribution"""
    u = np.random.uniform(0, 1, n)
    return g_and_h_quantile_vectorized(u, a, b, g, h)

#######################################################################
# Part 2: L-moment Estimation for g-and-h Parameters (Simplified)
#######################################################################

def estimate_g_and_h_params(data):
    """
    Simplified estimation of g-and-h parameters
    
    Parameters:
    data: array of observations
    
    Returns:
    a, b, g, h parameters
    """
    # Compute sample statistics
    mean = np.mean(data)
    std = np.std(data)
    skew = stats.skew(data)
    kurt = stats.kurtosis(data, fisher=True)  # Excess kurtosis
    
    # Simple heuristic mapping from moments to g-and-h parameters
    # These are rough approximations
    a = mean
    b = std / 1.5  # Adjust for the effect of g and h
    
    # Map skewness to g (simple heuristic)
    g = np.clip(skew / 5, -1, 1)
    
    # Map kurtosis to h (simple heuristic)
    h = np.clip(kurt / 30, 0, 0.5)
    
    return a, b, g, h

#######################################################################
# Part 3: Dynamic Quantile Function (DQF) Model (Simplified)
#######################################################################

class DQFModel:
    def __init__(self, n_particles=500, phi_step=0.1):
        """
        Initialize the DQF model (simplified version)
        
        Parameters:
        n_particles: Not used in this simplified version
        phi_step: Not used in this simplified version
        """
        # Model parameters (to be estimated)
        self.params = {}
        
    def fit(self, qf_data, n_iter=1000):
        """
        Fit the DQF model to QF-valued data (simplified version)
        
        Parameters:
        qf_data: Time series of g-and-h parameters (a, b*, g, h)
        n_iter: Not used in this simplified version
        """
        # Extract time series
        a_series = qf_data[:, 0]
        b_log_series = qf_data[:, 1]
        g_series = qf_data[:, 2]
        h_series = qf_data[:, 3]
        
        # Fit AR(1) models for a, b*, g, h (simplified)
        self.params['a'] = self._fit_ar1(a_series)
        self.params['b_log'] = self._fit_ar1(b_log_series)
        self.params['g'] = self._fit_ar1(g_series)
        self.params['h'] = self._fit_ar1(h_series)
        
        return self
    
    def _fit_ar1(self, series):
        """
        Fit AR(1) model (simplified and robust)
        
        Parameters:
        series: Time series data
        
        Returns:
        Dictionary of estimated parameters
        """
        n = len(series)
        
        # Compute lag-1 autocorrelation (more robust than least squares)
        if n > 1:
            psi = np.corrcoef(series[:-1], series[1:])[0, 1]
            psi = np.clip(psi, -0.99, 0.99)  # Ensure stationarity
        else:
            psi = 0
            
        # Compute mean
        mu = np.mean(series)
        
        # Compute delta
        delta = mu * (1 - psi)
        
        # Compute residual variance
        eps = series[1:] - (delta + psi * series[:-1])
        var_eps = np.var(eps) if len(eps) > 0 else 0.01
        
        return {
            'delta': delta,
            'psi': psi,
            'phi': 0.0,  # Not used in this simplified version
            'omega': var_eps * 0.05,
            'alpha': 0.1,
            'beta': 0.8,
            'var_eps': var_eps
        }
    
    def forecast_one_step(self, last_qf_params):
        """
        Generate one-step-ahead forecast
        
        Parameters:
        last_qf_params: Last observation of g-and-h parameters [a, b*, g, h]
        
        Returns:
        Forecasted g-and-h parameters
        """
        # Forecast a
        a_params = self.params['a']
        a_mean = a_params['delta'] + a_params['psi'] * last_qf_params[0]
        
        # Forecast b*
        b_params = self.params['b_log']
        b_mean = b_params['delta'] + b_params['psi'] * last_qf_params[1]
        
        # Forecast g
        g_params = self.params['g']
        g_mean = g_params['delta'] + g_params['psi'] * last_qf_params[2]
        
        # Forecast h
        h_params = self.params['h']
        h_mean = h_params['delta'] + h_params['psi'] * last_qf_params[3]
        h_mean = max(0, h_mean)  # Ensure h is non-negative
        
        return np.array([a_mean, b_mean, g_mean, h_mean])
    
    def forecast_var(self, last_qf_params, u):
        """
        Forecast VaR at probability level u
        
        Parameters:
        last_qf_params: Last observation of g-and-h parameters [a, b*, g, h]
        u: probability level (e.g., 0.01 for 1% VaR)
        
        Returns:
        VaR forecast
        """
        # Get one-step-ahead forecast of QF parameters
        forecast_params = self.forecast_one_step(last_qf_params)
        
        # Convert b* back to b
        forecast_params[1] = np.exp(forecast_params[1])
        
        # Compute quantile at level u
        return g_and_h_quantile(u, *forecast_params)

#######################################################################
# Part 4: Data Generation and Testing
#######################################################################

def generate_simulated_intraday_returns(n_days=500, n_intraday=390, shock_days=None):
    """
    Generate simulated intraday returns
    
    Parameters:
    n_days: Number of days
    n_intraday: Number of intraday returns per day
    shock_days: List of days with volatility shocks
    
    Returns:
    intraday_returns: List of arrays, each containing one day's intraday returns
    """
    intraday_returns = []
    
    # Parameters for the g-and-h distribution over time
    a_series = np.zeros(n_days)
    b_series = np.ones(n_days) * 0.01  # Start with small volatility
    g_series = np.zeros(n_days)
    h_series = np.ones(n_days) * 0.1
    
    # Add AR(1) structure
    for i in range(1, n_days):
        a_series[i] = 0.01 * a_series[i-1] + np.random.normal(0, 0.0005)
        b_series[i] = 0.01 + 0.8 * b_series[i-1] + np.random.gamma(1, 0.001)
        g_series[i] = 0.2 * g_series[i-1] + np.random.normal(0, 0.1)
        h_series[i] = 0.05 + 0.7 * h_series[i-1] + np.random.gamma(1, 0.005)
    
    # Add some volatility shocks
    if shock_days is not None:
        for day in shock_days:
            if day < n_days:
                b_series[day] *= 3
                h_series[day] *= 0.5  # Lower h during high volatility periods
    
    # Generate intraday returns for each day
    for i in range(n_days):
        day_returns = generate_g_and_h_sample(n_intraday, a_series[i], b_series[i], 
                                              g_series[i], h_series[i])
        intraday_returns.append(day_returns)
    
    # Create true parameter time series
    true_params = np.column_stack((a_series, np.log(b_series), g_series, h_series))
    
    return intraday_returns, true_params

def estimate_daily_qf_params(intraday_returns):
    """
    Estimate g-and-h parameters for each day's intraday returns
    
    Parameters:
    intraday_returns: List of arrays, each containing one day's intraday returns
    
    Returns:
    qf_params: Array of estimated g-and-h parameters [a, b*, g, h]
    """
    n_days = len(intraday_returns)
    qf_params = np.zeros((n_days, 4))
    
    for i in range(n_days):
        try:
            a, b, g, h = estimate_g_and_h_params(intraday_returns[i])
            qf_params[i] = [a, np.log(max(b, 1e-6)), g, h]
        except:
            # If estimation fails, use previous day's parameters or defaults
            if i > 0:
                qf_params[i] = qf_params[i-1]
            else:
                qf_params[i] = [0, -5, 0, 0.1]
    
    return qf_params

def evaluate_var_forecasts(test_intraday_returns, var_forecasts, level=0.01):
    """
    Evaluate VaR forecasts
    
    Parameters:
    test_intraday_returns: List of arrays containing intraday returns
    var_forecasts: Array of VaR forecasts
    level: VaR level (e.g., 0.01 for 1% VaR)
    
    Returns:
    violation_rate: Proportion of days where VaR was exceeded
    mean_abs_error: Mean absolute error of VaR forecasts
    """
    n_days = len(test_intraday_returns)
    actual_quantiles = np.zeros(n_days)
    
    # Calculate actual quantiles for each day
    for i in range(n_days):
        actual_quantiles[i] = np.percentile(test_intraday_returns[i], level * 100)
    
    # Calculate violation rate
    violations = (actual_quantiles < var_forecasts)
    violation_rate = np.mean(violations)
    
    # Calculate mean absolute error
    mean_abs_error = np.mean(np.abs(actual_quantiles - var_forecasts))
    
    return violation_rate, mean_abs_error

def plot_var_forecasts(test_intraday_returns, var_forecasts, level=0.01, n_days_to_plot=100):
    """
    Plot VaR forecasts against actual quantiles
    
    Parameters:
    test_intraday_returns: List of arrays containing intraday returns
    var_forecasts: Array of VaR forecasts
    level: VaR level (e.g., 0.01 for 1% VaR)
    n_days_to_plot: Number of days to include in the plot
    """
    n_days = min(len(test_intraday_returns), len(var_forecasts), n_days_to_plot)
    actual_quantiles = np.zeros(n_days)
    
    # Calculate actual quantiles for each day
    for i in range(n_days):
        actual_quantiles[i] = np.percentile(test_intraday_returns[i], level * 100)
    
    # Plot actual quantiles and forecasts
    plt.figure(figsize=(12, 6))
    plt.plot(actual_quantiles, label=f'Actual {level*100}% Quantile')
    plt.plot(var_forecasts[:n_days], label=f'DQF VaR Forecast')
    plt.legend()
    plt.title(f'{level*100}% VaR Forecast Evaluation')
    plt.xlabel('Day')
    plt.ylabel('Return')
    plt.grid(True, alpha=0.3)
    plt.show()

#######################################################################
# Part 5: Compare with Simple Benchmark Methods
#######################################################################

def calculate_historical_var(intraday_returns, window_size=20, level=0.01):
    """
    Calculate historical VaR for intraday returns
    
    Parameters:
    intraday_returns: List of arrays containing intraday returns
    window_size: Number of days to use for historical estimation
    level: VaR level (e.g., 0.01 for 1% VaR)
    
    Returns:
    var_forecasts: Array of VaR forecasts
    """
    n_days = len(intraday_returns)
    var_forecasts = np.zeros(n_days - window_size)
    
    for i in range(window_size, n_days):
        # Concatenate intraday returns from the window
        window_returns = []
        for j in range(i - window_size, i):
            window_returns.extend(intraday_returns[j])
        
        # Calculate historical VaR
        var_forecasts[i - window_size] = np.percentile(window_returns, level * 100)
    
    return var_forecasts

def exponential_smoothing_var(intraday_returns, alpha=0.94, level=0.01):
    """
    Calculate exponentially smoothed VaR for intraday returns
    
    Parameters:
    intraday_returns: List of arrays containing intraday returns
    alpha: Smoothing parameter
    level: VaR level (e.g., 0.01 for 1% VaR)
    
    Returns:
    var_forecasts: Array of VaR forecasts
    """
    n_days = len(intraday_returns)
    var_forecasts = np.zeros(n_days - 1)
    
    # Initialize with first day's VaR
    var_0 = np.percentile(intraday_returns[0], level * 100)
    
    for i in range(1, n_days):
        # Calculate today's VaR
        var_t = np.percentile(intraday_returns[i], level * 100)
        
        # Forecast tomorrow's VaR using exponential smoothing
        if i == 1:
            var_forecasts[i - 1] = alpha * var_0 + (1 - alpha) * var_t
        else:
            var_forecasts[i - 1] = alpha * var_forecasts[i - 2] + (1 - alpha) * var_t
    
    return var_forecasts

#######################################################################
# Part 6: Main Testing Script
#######################################################################

# Generate simulated intraday returns
n_days = 500
n_intraday = 390  # Typical number of 1-minute returns in a trading day
train_size = 300

print("Generating simulated intraday returns...")
shock_days = [100, 200, 300, 400]  # Days with volatility shocks
intraday_returns, true_params = generate_simulated_intraday_returns(n_days, n_intraday, shock_days)

# Estimate g-and-h parameters for each day
print("Estimating daily QF parameters...")
estimated_qf_params = estimate_daily_qf_params(intraday_returns)

# Split into training and testing sets
train_qf_params = estimated_qf_params[:train_size]
test_qf_params = estimated_qf_params[train_size:]
test_intraday_returns = intraday_returns[train_size:]

# Fit DQF model
print("Fitting DQF model...")
dqf_model = DQFModel()
dqf_model.fit(train_qf_params)

# Generate one-step-ahead VaR forecasts
print("Generating VaR forecasts...")
var_forecasts_1pct = np.zeros(n_days - train_size)
var_forecasts_5pct = np.zeros(n_days - train_size)

for i in range(n_days - train_size):
    if i == 0:
        last_params = train_qf_params[-1]
    else:
        last_params = test_qf_params[i-1]
        
    var_forecasts_1pct[i] = dqf_model.forecast_var(last_params, 0.01)
    var_forecasts_5pct[i] = dqf_model.forecast_var(last_params, 0.05)

# Calculate benchmark VaR forecasts
print("Calculating benchmark VaR forecasts...")
historical_var_1pct = calculate_historical_var(intraday_returns[:train_size + len(test_intraday_returns)], 
                                              window_size=20, level=0.01)[-len(test_intraday_returns):]
historical_var_5pct = calculate_historical_var(intraday_returns[:train_size + len(test_intraday_returns)], 
                                              window_size=20, level=0.05)[-len(test_intraday_returns):]

es_var_1pct = exponential_smoothing_var(intraday_returns[:train_size + len(test_intraday_returns)], 
                                       alpha=0.94, level=0.01)[-len(test_intraday_returns):]
es_var_5pct = exponential_smoothing_var(intraday_returns[:train_size + len(test_intraday_returns)], 
                                       alpha=0.94, level=0.05)[-len(test_intraday_returns):]

# Evaluate VaR forecasts
print("\nVaR Forecast Evaluation:")
print("1% VaR:")
dqf_vr_1pct, dqf_mae_1pct = evaluate_var_forecasts(test_intraday_returns, var_forecasts_1pct, 0.01)
hist_vr_1pct, hist_mae_1pct = evaluate_var_forecasts(test_intraday_returns, historical_var_1pct, 0.01)
es_vr_1pct, es_mae_1pct = evaluate_var_forecasts(test_intraday_returns, es_var_1pct, 0.01)

print(f"DQF Model: Violation Rate = {dqf_vr_1pct:.4f}, MAE = {dqf_mae_1pct:.6f}")
print(f"Historical: Violation Rate = {hist_vr_1pct:.4f}, MAE = {hist_mae_1pct:.6f}")
print(f"Exp Smooth: Violation Rate = {es_vr_1pct:.4f}, MAE = {es_mae_1pct:.6f}")

print("\n5% VaR:")
dqf_vr_5pct, dqf_mae_5pct = evaluate_var_forecasts(test_intraday_returns, var_forecasts_5pct, 0.05)
hist_vr_5pct, hist_mae_5pct = evaluate_var_forecasts(test_intraday_returns, historical_var_5pct, 0.05)
es_vr_5pct, es_mae_5pct = evaluate_var_forecasts(test_intraday_returns, es_var_5pct, 0.05)

print(f"DQF Model: Violation Rate = {dqf_vr_5pct:.4f}, MAE = {dqf_mae_5pct:.6f}")
print(f"Historical: Violation Rate = {hist_vr_5pct:.4f}, MAE = {hist_mae_5pct:.6f}")
print(f"Exp Smooth: Violation Rate = {es_vr_5pct:.4f}, MAE = {es_mae_5pct:.6f}")

# Plot VaR forecasts
print("\nPlotting VaR forecasts...")
plt.figure(figsize=(12, 10))

# 1% VaR
plt.subplot(2, 1, 1)
days = np.arange(len(test_intraday_returns))
actual_1pct = np.array([np.percentile(day_returns, 1) for day_returns in test_intraday_returns])

plt.plot(days, actual_1pct, 'k-', label='Actual 1% Quantile')
plt.plot(days, var_forecasts_1pct, 'r-', label='DQF Model')
plt.plot(days, historical_var_1pct, 'g--', label='Historical')
plt.plot(days, es_var_1pct, 'b:', label='Exp Smoothing')
plt.title('1% VaR Forecast Comparison')
plt.xlabel('Day')
plt.ylabel('Return')
plt.legend()
plt.grid(True, alpha=0.3)

# 5% VaR
plt.subplot(2, 1, 2)
actual_5pct = np.array([np.percentile(day_returns, 5) for day_returns in test_intraday_returns])

plt.plot(days, actual_5pct, 'k-', label='Actual 5% Quantile')
plt.plot(days, var_forecasts_5pct, 'r-', label='DQF Model')
plt.plot(days, historical_var_5pct, 'g--', label='Historical')
plt.plot(days, es_var_5pct, 'b:', label='Exp Smoothing')
plt.title('5% VaR Forecast Comparison')
plt.xlabel('Day')
plt.ylabel('Return')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Plot true vs estimated parameters
plt.figure(figsize=(15, 10))

param_names = ['a (location)', 'b* (log-scale)', 'g (asymmetry)', 'h (tail heaviness)']
for i in range(4):
    plt.subplot(2, 2, i+1)
    plt.plot(true_params[:, i], label='True')
    plt.plot(estimated_qf_params[:, i], label='Estimated')
    plt.title(param_names[i])
    plt.legend()
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Plot distributions for a few days
days_to_plot = [50, 150, 250, 350]  # Some days to analyze
plt.figure(figsize=(15, 10))

for i, day in enumerate(days_to_plot):
    plt.subplot(2, 2, i+1)
    
    # Plot histogram of actual returns
    plt.hist(intraday_returns[day], bins=30, density=True, alpha=0.5, label='Actual Returns')
    
    # Get estimated g-and-h parameters
    a, b_log, g, h = estimated_qf_params[day]
    b = np.exp(b_log)
    
    # Generate points from fitted g-and-h distribution
    x = np.linspace(min(intraday_returns[day]), max(intraday_returns[day]), 1000)
    u = np.linspace(0.01, 0.99, 1000)
    fitted_quantiles = [g_and_h_quantile(ui, a, b, g, h) for ui in u]
    
    # Create a KDE from the fitted quantiles for visualization
    kde = stats.gaussian_kde(fitted_quantiles)
    plt.plot(x, kde(x), 'r-', label='Fitted g-and-h')
    
    plt.title(f'Day {day}')
    plt.xlabel('Return')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()