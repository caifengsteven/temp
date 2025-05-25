"""
Rough Volatility Models Implementation

This script implements the specific rough volatility models mentioned in the paper:
"Empirical analysis of rough and classical stochastic volatility models to the SPX and VIX markets"
by Sigurd Emil Rømer.

It includes:
1. Rough Bergomi model
2. Rough Heston model
3. Comparison with classical models
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma
import pandas as pd
from scipy import stats
import math

# ===== Utility Functions =====

def fractional_brownian_motion(n, hurst, dt=1.0):
    """
    Generate a fractional Brownian motion time series with Hurst parameter H.
    
    Parameters:
    - n: Number of points to generate
    - hurst: Hurst parameter (H), where:
        * H < 0.5: anti-persistent process (rough)
        * H = 0.5: standard Brownian motion
        * H > 0.5: persistent process (smooth)
    - dt: Time step
    
    Returns:
    - Time series of fractional Brownian motion
    """
    # Generate a standard Gaussian process
    gaussian_increments = np.random.normal(0.0, 1.0, n)
    
    # The first value of the fBm is zero
    fbm = np.zeros(n)
    
    # Scale factor for the process
    scale_factor = dt**hurst
    
    # Generate the fBm
    for i in range(1, n):
        fbm[i] = fbm[i-1] + scale_factor * gaussian_increments[i-1]
    
    return fbm

def correlated_brownian_motions(n, rho, dt=1.0):
    """
    Generate two correlated Brownian motions.
    
    Parameters:
    - n: Number of points to generate
    - rho: Correlation coefficient
    - dt: Time step
    
    Returns:
    - Two correlated Brownian motion paths
    """
    # Generate two independent standard Gaussian processes
    dW1 = np.random.normal(0.0, np.sqrt(dt), n)
    dW2_independent = np.random.normal(0.0, np.sqrt(dt), n)
    
    # Create correlated process
    dW2 = rho * dW1 + np.sqrt(1 - rho**2) * dW2_independent
    
    # Integrate to get Brownian motions
    W1 = np.cumsum(dW1)
    W2 = np.cumsum(dW2)
    
    # Prepend zeros
    W1 = np.insert(W1, 0, 0)
    W2 = np.insert(W2, 0, 0)
    
    return W1, W2

# ===== Model Implementations =====

def rough_bergomi_model(n, alpha, eta, rho, v0, dt=1.0):
    """
    Simulate the rough Bergomi model.
    
    Parameters:
    - n: Number of points to generate
    - alpha: Roughness parameter (alpha = H - 0.5)
    - eta: Volatility of volatility
    - rho: Correlation between price and volatility
    - v0: Initial variance
    - dt: Time step
    
    Returns:
    - Price and variance paths
    """
    # Hurst parameter
    H = alpha + 0.5
    
    # Generate fractional Brownian motion for volatility
    W_vol = fractional_brownian_motion(n, H, dt)
    
    # Generate correlated Brownian motion for price
    W_price_independent = fractional_brownian_motion(n, 0.5, dt)  # Standard BM
    W_price = rho * W_vol + np.sqrt(1 - rho**2) * W_price_independent
    
    # Calculate variance process
    variance = np.zeros(n)
    variance[0] = v0
    
    for i in range(1, n):
        # Rough Bergomi variance process
        variance[i] = v0 * np.exp(eta * W_vol[i] - 0.5 * eta**2 * i * dt**(2*H))
    
    # Calculate price process
    price = np.zeros(n)
    price[0] = 100  # Initial price
    
    for i in range(1, n):
        # Price process with volatility feedback
        price[i] = price[i-1] * np.exp((- 0.5 * variance[i-1]) * dt + np.sqrt(variance[i-1]) * (W_price[i] - W_price[i-1]))
    
    return price, variance

def rough_heston_model(n, alpha, lambda_param, eta, rho, v0, dt=1.0):
    """
    Simulate the rough Heston model.
    
    Parameters:
    - n: Number of points to generate
    - alpha: Roughness parameter (alpha = H - 0.5)
    - lambda_param: Mean-reversion speed
    - eta: Volatility of volatility
    - rho: Correlation between price and volatility
    - v0: Initial variance
    - dt: Time step
    
    Returns:
    - Price and variance paths
    """
    # Generate correlated Brownian motions
    W1, W2 = correlated_brownian_motions(n, rho, dt)
    
    # Calculate variance process
    variance = np.zeros(n+1)
    variance[0] = v0
    
    # Fractional kernel parameters
    H = alpha + 0.5
    
    # Discretize the rough Heston variance process
    for i in range(1, n+1):
        # Rough Heston variance process with fractional kernel
        kernel_sum = 0
        for j in range(1, i):
            # Fractional kernel
            kernel = (i - j + 1)**(H - 0.5) - (i - j)**(H - 0.5)
            kernel_sum += kernel * (lambda_param * (v0 - variance[j]) * dt + eta * np.sqrt(max(0, variance[j])) * (W1[j] - W1[j-1]))
        
        # Scale by gamma function
        kernel_scale = 1 / (gamma(H + 0.5))
        variance[i] = v0 + kernel_scale * kernel_sum
        
        # Ensure variance is positive
        variance[i] = max(0, variance[i])
    
    # Calculate price process
    price = np.zeros(n+1)
    price[0] = 100  # Initial price
    
    for i in range(1, n+1):
        # Price process with volatility feedback
        price[i] = price[i-1] * np.exp((- 0.5 * variance[i-1]) * dt + np.sqrt(variance[i-1]) * (W2[i] - W2[i-1]))
    
    return price[:-1], variance[:-1]  # Return n points

def classical_heston_model(n, kappa, theta, eta, rho, v0, dt=1.0):
    """
    Simulate the classical Heston model.
    
    Parameters:
    - n: Number of points to generate
    - kappa: Mean-reversion speed
    - theta: Long-term variance
    - eta: Volatility of volatility
    - rho: Correlation between price and volatility
    - v0: Initial variance
    - dt: Time step
    
    Returns:
    - Price and variance paths
    """
    # Generate correlated Brownian motions
    W1, W2 = correlated_brownian_motions(n, rho, dt)
    
    # Calculate variance process
    variance = np.zeros(n+1)
    variance[0] = v0
    
    for i in range(1, n+1):
        # Heston variance process (square-root diffusion)
        dv = kappa * (theta - variance[i-1]) * dt + eta * np.sqrt(max(0, variance[i-1])) * (W1[i] - W1[i-1])
        variance[i] = max(0, variance[i-1] + dv)  # Ensure variance is positive
    
    # Calculate price process
    price = np.zeros(n+1)
    price[0] = 100  # Initial price
    
    for i in range(1, n+1):
        # Price process with volatility feedback
        price[i] = price[i-1] * np.exp((- 0.5 * variance[i-1]) * dt + np.sqrt(variance[i-1]) * (W2[i] - W2[i-1]))
    
    return price[:-1], variance[:-1]  # Return n points

# ===== Analysis Functions =====

def calculate_log_returns(prices):
    """Calculate log returns from price series."""
    return np.diff(np.log(prices))

def calculate_realized_volatility(returns, window=30):
    """
    Calculate realized volatility from returns.
    
    Parameters:
    - returns: Asset returns time series
    - window: Window size for rolling volatility calculation
    
    Returns:
    - Realized volatility series
    """
    # Calculate squared returns
    squared_returns = returns**2
    
    # Calculate rolling sum of squared returns
    rolling_sum = np.zeros(len(returns) - window + 1)
    for i in range(len(rolling_sum)):
        rolling_sum[i] = np.sum(squared_returns[i:i+window])
    
    # Convert to annualized volatility
    annualization_factor = 252  # Trading days per year
    realized_vol = np.sqrt(rolling_sum / window * annualization_factor)
    
    return realized_vol

def main():
    """Main function to test rough volatility models."""
    print("=== Rough Volatility Models Analysis ===")
    
    # Simulation parameters
    n_points = 1000
    dt = 1.0/252  # Daily data (252 trading days per year)
    time = np.arange(0, n_points) * dt
    
    # Model parameters
    alpha = -0.4  # Roughness parameter
    v0 = 0.04  # Initial variance (corresponds to 20% volatility)
    
    # Rough Bergomi parameters
    rb_eta = 2.0  # Volatility of volatility
    rb_rho = -0.7  # Correlation
    
    # Rough Heston parameters
    rh_lambda = 0.3  # Mean-reversion speed
    rh_eta = 0.3  # Volatility of volatility
    rh_rho = -0.7  # Correlation
    
    # Classical Heston parameters
    ch_kappa = 1.0  # Mean-reversion speed
    ch_theta = 0.04  # Long-term variance
    ch_eta = 0.3  # Volatility of volatility
    ch_rho = -0.7  # Correlation
    
    print("\n1. Simulating Rough Bergomi model...")
    rb_price, rb_variance = rough_bergomi_model(n_points, alpha, rb_eta, rb_rho, v0, dt)
    rb_vol = np.sqrt(rb_variance)
    rb_returns = calculate_log_returns(rb_price)
    
    print("\n2. Simulating Rough Heston model...")
    rh_price, rh_variance = rough_heston_model(n_points, alpha, rh_lambda, rh_eta, rh_rho, v0, dt)
    rh_vol = np.sqrt(rh_variance)
    rh_returns = calculate_log_returns(rh_price)
    
    print("\n3. Simulating Classical Heston model...")
    ch_price, ch_variance = classical_heston_model(n_points, ch_kappa, ch_theta, ch_eta, ch_rho, v0, dt)
    ch_vol = np.sqrt(ch_variance)
    ch_returns = calculate_log_returns(ch_price)
    
    # Plot price paths
    plt.figure(figsize=(12, 6))
    plt.plot(time, rb_price, label="Rough Bergomi")
    plt.plot(time, rh_price, label="Rough Heston")
    plt.plot(time, ch_price, label="Classical Heston")
    plt.title("Simulated Price Paths")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    plt.savefig("model_price_paths.png")
    print("Saved price paths plot to model_price_paths.png")
    
    # Plot volatility paths
    plt.figure(figsize=(12, 6))
    plt.plot(time, rb_vol, label="Rough Bergomi")
    plt.plot(time, rh_vol, label="Rough Heston")
    plt.plot(time, ch_vol, label="Classical Heston")
    plt.title("Simulated Volatility Paths")
    plt.xlabel("Time")
    plt.ylabel("Volatility")
    plt.legend()
    plt.grid(True)
    plt.savefig("model_volatility_paths.png")
    print("Saved volatility paths plot to model_volatility_paths.png")
    
    # Calculate realized volatility
    rb_realized_vol = calculate_realized_volatility(rb_returns)
    rh_realized_vol = calculate_realized_volatility(rh_returns)
    ch_realized_vol = calculate_realized_volatility(ch_returns)
    
    # Plot realized volatility
    plt.figure(figsize=(12, 6))
    plt.plot(time[30:], rb_realized_vol, label="Rough Bergomi")
    plt.plot(time[30:], rh_realized_vol, label="Rough Heston")
    plt.plot(time[30:], ch_realized_vol, label="Classical Heston")
    plt.title("Realized Volatility")
    plt.xlabel("Time")
    plt.ylabel("Realized Volatility")
    plt.legend()
    plt.grid(True)
    plt.savefig("model_realized_vol.png")
    print("Saved realized volatility plot to model_realized_vol.png")
    
    # Calculate statistics
    print("\n4. Model Statistics:")
    
    # Rough Bergomi statistics
    print("\nRough Bergomi Model:")
    print(f"Mean Returns: {np.mean(rb_returns):.6f}")
    print(f"Std Dev Returns: {np.std(rb_returns):.6f}")
    print(f"Skewness: {stats.skew(rb_returns):.6f}")
    print(f"Kurtosis: {stats.kurtosis(rb_returns):.6f}")
    
    # Rough Heston statistics
    print("\nRough Heston Model:")
    print(f"Mean Returns: {np.mean(rh_returns):.6f}")
    print(f"Std Dev Returns: {np.std(rh_returns):.6f}")
    print(f"Skewness: {stats.skew(rh_returns):.6f}")
    print(f"Kurtosis: {stats.kurtosis(rh_returns):.6f}")
    
    # Classical Heston statistics
    print("\nClassical Heston Model:")
    print(f"Mean Returns: {np.mean(ch_returns):.6f}")
    print(f"Std Dev Returns: {np.std(ch_returns):.6f}")
    print(f"Skewness: {stats.skew(ch_returns):.6f}")
    print(f"Kurtosis: {stats.kurtosis(ch_returns):.6f}")
    
    # Save model data to CSV
    model_data = pd.DataFrame({
        'time': time[:-1],
        'rb_price': rb_price[:-1],
        'rb_volatility': rb_vol[:-1],
        'rb_returns': rb_returns,
        'rh_price': rh_price[:-1],
        'rh_volatility': rh_vol[:-1],
        'rh_returns': rh_returns,
        'ch_price': ch_price[:-1],
        'ch_volatility': ch_vol[:-1],
        'ch_returns': ch_returns
    })
    model_data.to_csv("model_comparison_data.csv", index=False)
    print("\nSaved model comparison data to model_comparison_data.csv")
    
    print("\n=== Analysis Complete ===")

if __name__ == "__main__":
    main()
