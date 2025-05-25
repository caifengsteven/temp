"""
Core Rough Volatility Analysis Script

This script implements the core concepts from rough volatility models
based on the paper "Empirical analysis of rough and classical stochastic
volatility models to the SPX and VIX markets" by Sigurd Emil Rømer.

It focuses on simulating rough volatility paths and analyzing their properties.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma
import pandas as pd
from scipy import stats

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

def rough_volatility_simulation(n, alpha, sigma0, dt=1.0):
    """
    Simulate rough volatility process based on fractional Brownian motion.

    Parameters:
    - n: Number of points to generate
    - alpha: Roughness parameter (alpha = H - 0.5)
    - sigma0: Initial volatility
    - dt: Time step

    Returns:
    - Simulated volatility path
    """
    # Hurst parameter H = alpha + 0.5
    hurst = alpha + 0.5

    # Generate fractional Brownian motion
    fbm = fractional_brownian_motion(n, hurst, dt)

    # Convert to volatility (using exponential to ensure positivity)
    vol = sigma0 * np.exp(fbm)

    return vol

def estimate_hurst_parameter(returns, q_values=None):
    """
    Estimate the Hurst parameter using the structure function method.

    Parameters:
    - returns: Asset returns time series
    - q_values: List of q values for the structure function

    Returns:
    - Estimated Hurst parameter
    """
    if q_values is None:
        q_values = [0.5, 1.0, 1.5, 2.0]

    n = len(returns)
    lags = np.logspace(0, np.log10(n//4), 20).astype(int)
    lags = np.unique(lags)

    # Calculate structure functions for different q values
    log_sf = np.zeros((len(q_values), len(lags)))

    for i, q in enumerate(q_values):
        for j, lag in enumerate(lags):
            # Structure function
            sf = np.mean(np.abs(returns[lag:] - returns[:-lag])**q)
            log_sf[i, j] = np.log(sf) if sf > 0 else 0

    # Estimate zeta(q) (scaling exponent)
    zeta_q = np.zeros(len(q_values))
    log_lags = np.log(lags)

    for i in range(len(q_values)):
        # Linear regression to estimate zeta(q)
        valid_idx = log_sf[i, :] > 0
        if np.sum(valid_idx) > 1:
            zeta_q[i] = np.polyfit(log_lags[valid_idx], log_sf[i, valid_idx], 1)[0]

    # Estimate H from zeta(q) = q*H
    H_estimates = zeta_q / np.array(q_values)
    H = np.median(H_estimates)

    return H

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
    """Main function to test rough volatility concepts."""
    print("=== Core Rough Volatility Analysis ===")

    # Simulation parameters
    n_points = 1000
    dt = 1.0/252  # Daily data (252 trading days per year)
    time = np.arange(0, n_points) * dt

    # Test different roughness parameters
    alphas = [-0.4, -0.3, -0.2, -0.1]  # Rough volatility: alpha < 0
    sigma0 = 0.2  # Initial volatility

    print("\n1. Simulating rough volatility paths...")
    # Simulate volatility paths
    vol_paths = []
    labels = []

    for alpha in alphas:
        vol = rough_volatility_simulation(n_points, alpha, sigma0, dt)
        vol_paths.append(vol)
        labels.append(f"α = {alpha}")

    # Plot simulated volatility paths
    plt.figure(figsize=(12, 6))

    for i, path in enumerate(vol_paths):
        plt.plot(time, path, label=labels[i])

    plt.title("Simulated Rough Volatility Paths")
    plt.xlabel("Time")
    plt.ylabel("Volatility")
    plt.legend()
    plt.grid(True)
    plt.savefig("rough_volatility_paths.png")
    print("Saved plot to rough_volatility_paths.png")

    print("\n2. Simulating returns with rough volatility...")
    # Simulate returns data with rough volatility
    alpha_true = -0.3
    vol = rough_volatility_simulation(n_points, alpha_true, sigma0, dt)
    returns = vol * np.random.normal(0, np.sqrt(dt), n_points)

    # Estimate roughness parameter from simulated returns
    alpha_est = estimate_hurst_parameter(returns) - 0.5

    print(f"True roughness parameter (alpha): {alpha_true}")
    print(f"Estimated roughness parameter (alpha): {alpha_est:.4f}")

    # Calculate realized volatility
    realized_vol = calculate_realized_volatility(returns)

    # Plot realized volatility
    plt.figure(figsize=(12, 6))
    plt.plot(time[29:], realized_vol, label="Realized Volatility")
    plt.plot(time[:-1], vol[:-1], label="True Volatility", alpha=0.5)
    plt.title("Realized vs True Volatility")
    plt.xlabel("Time")
    plt.ylabel("Volatility")
    plt.legend()
    plt.grid(True)
    plt.savefig("realized_vs_true_vol.png")
    print("Saved plot to realized_vs_true_vol.png")

    # Calculate some statistics
    print("\n3. Statistics for simulated returns:")
    print(f"Mean: {np.mean(returns):.6f}")
    print(f"Std Dev: {np.std(returns):.6f}")
    print(f"Skewness: {stats.skew(returns):.6f}")
    print(f"Kurtosis: {stats.kurtosis(returns):.6f}")

    # Save returns to CSV for further analysis
    # Make sure all arrays have the same length
    pd.DataFrame({
        'time': time[:len(returns)],
        'volatility': vol[:len(returns)],
        'returns': returns
    }).to_csv("simulated_rough_vol_data.csv", index=False)
    print("Saved simulated data to simulated_rough_vol_data.csv")

    # Analyze autocorrelation of volatility
    print("\n4. Analyzing volatility autocorrelation...")
    # Calculate log volatility
    log_vol = np.log(vol)

    # Calculate autocorrelation
    max_lag = 100
    acf = np.zeros(max_lag)
    for lag in range(max_lag):
        acf[lag] = np.corrcoef(log_vol[lag:], log_vol[:-lag if lag > 0 else None])[0, 1]

    # Plot autocorrelation
    plt.figure(figsize=(12, 6))
    plt.plot(range(max_lag), acf)
    plt.title("Autocorrelation of Log Volatility")
    plt.xlabel("Lag")
    plt.ylabel("Autocorrelation")
    plt.grid(True)
    plt.savefig("vol_autocorrelation.png")
    print("Saved plot to vol_autocorrelation.png")

    # Analyze volatility of volatility
    print("\n5. Analyzing volatility of volatility...")
    # Calculate log volatility returns
    log_vol_returns = np.diff(log_vol)

    # Calculate volatility of volatility
    vol_of_vol = np.std(log_vol_returns)
    print(f"Volatility of volatility: {vol_of_vol:.6f}")

    # Plot log volatility returns
    plt.figure(figsize=(12, 6))
    plt.plot(time[1:], log_vol_returns)
    plt.title("Log Volatility Returns")
    plt.xlabel("Time")
    plt.ylabel("Log Volatility Returns")
    plt.grid(True)
    plt.savefig("log_vol_returns.png")
    print("Saved plot to log_vol_returns.png")

    print("\n=== Analysis Complete ===")

if __name__ == "__main__":
    main()
