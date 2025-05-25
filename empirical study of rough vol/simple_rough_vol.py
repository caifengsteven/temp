"""
Simple Rough Volatility Testing Script

This script implements basic concepts from rough volatility models
based on the paper "Empirical analysis of rough and classical stochastic
volatility models to the SPX and VIX markets" by Sigurd Emil Rømer.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma
import pandas as pd
import datetime as dt

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

def main():
    """Main function to test rough volatility concepts."""
    print("Testing rough volatility concepts...")
    
    # Simulation parameters
    n_points = 1000
    dt = 1.0/252  # Daily data (252 trading days per year)
    time = np.arange(0, n_points) * dt
    
    # Test different roughness parameters
    alphas = [-0.3, -0.2, -0.1]  # Rough volatility: alpha < 0
    sigma0 = 0.2  # Initial volatility
    
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
    
    # Save the plot instead of showing it (which might not work in this environment)
    plt.savefig("rough_volatility_paths.png")
    print("Saved plot to rough_volatility_paths.png")
    
    # Simulate returns data
    alpha_true = -0.3
    vol = rough_volatility_simulation(n_points, alpha_true, sigma0, dt)
    returns = vol * np.random.normal(0, np.sqrt(dt), n_points)
    
    # Calculate some statistics
    print(f"\nStatistics for simulated returns with alpha = {alpha_true}:")
    print(f"Mean: {np.mean(returns):.6f}")
    print(f"Std Dev: {np.std(returns):.6f}")
    print(f"Skewness: {np.mean((returns - np.mean(returns))**3) / np.std(returns)**3:.6f}")
    print(f"Kurtosis: {np.mean((returns - np.mean(returns))**4) / np.std(returns)**4:.6f}")
    
    # Save returns to CSV for further analysis
    pd.DataFrame({
        'time': time,
        'volatility': vol,
        'returns': returns
    }).to_csv("simulated_rough_vol_data.csv", index=False)
    print("Saved simulated data to simulated_rough_vol_data.csv")

if __name__ == "__main__":
    main()
