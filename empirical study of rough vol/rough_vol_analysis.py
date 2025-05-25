"""
Rough Volatility Analysis Script

This script implements concepts from the paper:
"Empirical analysis of rough and classical stochastic volatility models to the SPX and VIX markets"
by Sigurd Emil Rømer.

It tests rough volatility models using either Bloomberg data (if available) or simulated data.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma
import pandas as pd
import datetime
import os
from scipy import stats
from scipy.optimize import minimize

# Try to import Bloomberg API if available
try:
    import pdblp
    BLOOMBERG_AVAILABLE = True
    print("Bloomberg API (pdblp) is available")
except ImportError:
    print("Bloomberg API (pdblp) not available. Will use simulated data.")
    BLOOMBERG_AVAILABLE = False

# ===== Simulation Functions =====

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

# ===== Bloomberg Data Functions =====

def get_bloomberg_data(ticker, start_date, end_date):
    """
    Fetch data from Bloomberg if available.

    Parameters:
    - ticker: Bloomberg ticker
    - start_date: Start date for data
    - end_date: End date for data

    Returns:
    - DataFrame with Bloomberg data or None if not available
    """
    if not BLOOMBERG_AVAILABLE:
        return None

    try:
        # Initialize Bloomberg connection
        con = pdblp.BCon(debug=False, port=8194)
        con.start()

        # Fetch historical data
        data = con.bdh(
            tickers=ticker,
            flds=['PX_LAST', 'VOLATILITY_10D', 'VOLATILITY_30D'],
            start_date=start_date,
            end_date=end_date
        )

        con.stop()
        return data
    except Exception as e:
        print(f"Error fetching Bloomberg data: {e}")
        return None

# ===== Analysis Functions =====

def calculate_log_returns(prices):
    """Calculate log returns from price series."""
    return np.diff(np.log(prices))

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

def estimate_roughness_parameter(returns):
    """
    Estimate the roughness parameter alpha from returns.

    Parameters:
    - returns: Asset returns time series

    Returns:
    - Estimated alpha (roughness parameter)
    """
    # Estimate Hurst parameter
    H = estimate_hurst_parameter(returns)

    # Convert to roughness parameter (alpha = H - 0.5)
    alpha = H - 0.5

    return alpha

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

def plot_volatility_paths(time, vol_paths, labels=None, title="Volatility Paths", save_path=None):
    """
    Plot volatility paths.

    Parameters:
    - time: Time points
    - vol_paths: List of volatility paths to plot
    - labels: Labels for each path
    - title: Plot title
    - save_path: Path to save the plot (if None, display the plot)
    """
    plt.figure(figsize=(12, 6))

    if labels is None:
        labels = [f"Path {i+1}" for i in range(len(vol_paths))]

    for i, path in enumerate(vol_paths):
        plt.plot(time, path, label=labels[i])

    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Volatility")
    plt.legend()
    plt.grid(True)

    if save_path:
        plt.savefig(save_path)
        print(f"Saved plot to {save_path}")
    else:
        plt.show()

def main():
    """Main function to test rough volatility concepts."""
    print("=== Rough Volatility Analysis ===")

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
    plot_volatility_paths(time, vol_paths, labels,
                         "Simulated Rough Volatility Paths",
                         "rough_volatility_paths.png")

    # Try to get Bloomberg data if available
    if BLOOMBERG_AVAILABLE:
        print("\n2. Attempting to fetch Bloomberg data...")

        # Example: S&P 500 data for the last 2 years
        today = datetime.datetime.now()
        start_date = (today - datetime.timedelta(days=2*365)).strftime('%Y%m%d')
        end_date = today.strftime('%Y%m%d')

        # Try to get SPX data
        spx_data = get_bloomberg_data('SPX Index', start_date, end_date)

        if spx_data is not None:
            print("Successfully retrieved SPX data from Bloomberg")

            # Calculate returns
            spx_prices = spx_data['SPX Index']['PX_LAST']
            spx_returns = calculate_log_returns(spx_prices.values)

            # Estimate roughness parameter
            alpha_spx = estimate_roughness_parameter(spx_returns)
            print(f"Estimated roughness parameter (alpha) for SPX: {alpha_spx:.4f}")

            # Calculate realized volatility
            realized_vol = calculate_realized_volatility(spx_returns)

            # Plot realized volatility
            vol_dates = spx_prices.index[29:]  # Adjust for window size
            plot_volatility_paths(
                range(len(realized_vol)),
                [realized_vol],
                ["SPX Realized Volatility"],
                "SPX Realized Volatility",
                "spx_realized_vol.png"
            )

            # Save SPX data to CSV
            spx_data.to_csv("spx_bloomberg_data.csv")
            print("Saved SPX data to spx_bloomberg_data.csv")

            # Try to get VIX data
            vix_data = get_bloomberg_data('VIX Index', start_date, end_date)

            if vix_data is not None:
                print("Successfully retrieved VIX data from Bloomberg")

                # Save VIX data to CSV
                vix_data.to_csv("vix_bloomberg_data.csv")
                print("Saved VIX data to vix_bloomberg_data.csv")

                # Plot VIX vs SPX realized volatility
                vix_prices = vix_data['VIX Index']['PX_LAST']

                # Align dates
                common_dates = set(vol_dates).intersection(set(vix_prices.index))
                common_dates = sorted(list(common_dates))

                if common_dates:
                    aligned_vol = [realized_vol[vol_dates.get_loc(date)] for date in common_dates if date in vol_dates]
                    aligned_vix = [vix_prices[date]/100 for date in common_dates if date in vix_prices.index]  # VIX is in percentage points

                    plt.figure(figsize=(12, 6))
                    plt.plot(range(len(aligned_vol)), aligned_vol, label="SPX Realized Volatility")
                    plt.plot(range(len(aligned_vix)), aligned_vix, label="VIX Index/100")
                    plt.title("SPX Realized Volatility vs VIX Index")
                    plt.xlabel("Time")
                    plt.ylabel("Volatility")
                    plt.legend()
                    plt.grid(True)
                    plt.savefig("spx_vs_vix.png")
                    print("Saved SPX vs VIX plot to spx_vs_vix.png")
        else:
            print("Could not retrieve Bloomberg data, using simulated data instead")
    else:
        print("\n2. Bloomberg API not available, using simulated data only")

    print("\n3. Simulating returns with rough volatility...")
    # Simulate returns data with rough volatility
    alpha_true = -0.3
    vol = rough_volatility_simulation(n_points, alpha_true, sigma0, dt)
    returns = vol * np.random.normal(0, np.sqrt(dt), n_points)

    # Estimate roughness parameter from simulated returns
    alpha_est = estimate_roughness_parameter(returns)

    print(f"True roughness parameter (alpha): {alpha_true}")
    print(f"Estimated roughness parameter (alpha): {alpha_est:.4f}")

    # Calculate some statistics
    print("\n4. Statistics for simulated returns:")
    print(f"Mean: {np.mean(returns):.6f}")
    print(f"Std Dev: {np.std(returns):.6f}")
    print(f"Skewness: {stats.skew(returns):.6f}")
    print(f"Kurtosis: {stats.kurtosis(returns):.6f}")

    # Save returns to CSV for further analysis
    pd.DataFrame({
        'time': time[:-1],  # Adjust for returns calculation
        'volatility': vol[:-1],
        'returns': returns
    }).to_csv("simulated_rough_vol_data.csv", index=False)
    print("Saved simulated data to simulated_rough_vol_data.csv")

    print("\n=== Analysis Complete ===")

if __name__ == "__main__":
    main()
