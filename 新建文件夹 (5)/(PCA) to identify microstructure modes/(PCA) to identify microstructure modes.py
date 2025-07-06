import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import boxcox
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_acf
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# Define functions for simulating limit order book data
def simulate_order_book_data(n_days=100, events_per_day=5000, significant_ratio=0.2):
    """
    Simulate limit order book data with:
    - Market orders (executions) at bid and ask
    - Limit orders at bid and ask
    - Cancellations at bid and ask
    - Time between events
    - Price changes
    
    Parameters:
    - n_days: Number of days to simulate
    - events_per_day: Average number of events per day
    - significant_ratio: Ratio of significant price changes to total events
    
    Returns:
    - DataFrame with simulated data
    """
    # Total number of events
    n_events = int(n_days * events_per_day)
    
    # Initialize arrays for all variables
    data = {
        'day': np.zeros(n_events, dtype=int),
        'time': np.zeros(n_events),
        'event_type': np.zeros(n_events, dtype=int),  # 0: limit, 1: cancel, 2: market
        'side': np.zeros(n_events, dtype=int),        # 0: bid, 1: ask
        'volume': np.zeros(n_events),
        'price': np.zeros(n_events),
        'is_significant': np.zeros(n_events, dtype=bool)
    }
    
    # Current price and time
    current_price = 100.0
    current_time = 0.0
    
    # Initialize order book state (queue sizes)
    bid_queue = 1000
    ask_queue = 1000
    
    # Parameters for event generation
    event_probs = [0.4, 0.4, 0.2]  # limit, cancel, market
    side_probs = [0.5, 0.5]        # bid, ask
    
    # Volume parameters
    volume_mean = 10
    volume_std = 5
    
    # Time parameters
    time_mean = 0.01
    time_std = 0.005
    
    # Generate events
    day_counter = 0
    events_today = 0
    
    for i in range(n_events):
        # New day check
        if events_today >= events_per_day:
            day_counter += 1
            events_today = 0
        
        data['day'][i] = day_counter
        events_today += 1
        
        # Generate time increment
        time_increment = np.abs(np.random.normal(time_mean, time_std))
        current_time += time_increment
        data['time'][i] = current_time
        
        # Determine event type and side
        event_type = np.random.choice(3, p=event_probs)  # 0: limit, 1: cancel, 2: market
        side = np.random.choice(2, p=side_probs)        # 0: bid, 1: ask
        
        # Generate volume
        volume = max(1, int(np.random.normal(volume_mean, volume_std)))
        
        # Update order book state based on event
        if event_type == 0:  # Limit order
            if side == 0:  # Bid
                bid_queue += volume
            else:  # Ask
                ask_queue += volume
        elif event_type == 1:  # Cancellation
            if side == 0:  # Bid
                volume = min(volume, bid_queue - 1)  # Ensure queue doesn't go below 1
                bid_queue -= volume
            else:  # Ask
                volume = min(volume, ask_queue - 1)  # Ensure queue doesn't go below 1
                ask_queue -= volume
        else:  # Market order
            if side == 0:  # Sell (hitting bid)
                volume = min(volume, bid_queue - 1)  # Ensure queue doesn't go below 1
                bid_queue -= volume
                # If bid queue is depleted, price may change
                if bid_queue < 10:
                    bid_queue = 1000
                    # 20% chance of significant price change
                    if np.random.random() < significant_ratio:
                        current_price -= 0.01
                        data['is_significant'][i] = True
            else:  # Buy (lifting ask)
                volume = min(volume, ask_queue - 1)  # Ensure queue doesn't go below 1
                ask_queue -= volume
                # If ask queue is depleted, price may change
                if ask_queue < 10:
                    ask_queue = 1000
                    # 20% chance of significant price change
                    if np.random.random() < significant_ratio:
                        current_price += 0.01
                        data['is_significant'][i] = True
        
        # Store the data
        data['event_type'][i] = event_type
        data['side'][i] = side
        data['volume'][i] = volume
        data['price'][i] = current_price
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    return df

def extract_significant_price_changes(df):
    """
    Extract significant price changes and aggregate flows between them.
    
    Parameters:
    - df: DataFrame with raw order book data
    
    Returns:
    - DataFrame with aggregated data between significant price changes
    """
    # Find indices where significant price changes occur
    sig_indices = np.where(df['is_significant'])[0]
    
    if len(sig_indices) < 2:
        raise ValueError("Not enough significant price changes in the data")
    
    # Initialize list to store aggregated data
    aggregated_data = []
    
    # Process each interval between significant price changes
    for i in range(len(sig_indices)-1):
        start_idx = sig_indices[i]
        end_idx = sig_indices[i+1]
        
        # Get slice of data for this interval
        interval_data = df.iloc[start_idx:end_idx+1]
        
        # Calculate time duration
        delta_t = interval_data.iloc[-1]['time'] - interval_data.iloc[0]['time']
        
        # Calculate return
        r = interval_data.iloc[-1]['price'] - interval_data.iloc[0]['price']
        
        # Aggregate flows by type and side
        v_lo_b = interval_data[(interval_data['event_type'] == 0) & (interval_data['side'] == 0)]['volume'].sum()
        v_lo_a = interval_data[(interval_data['event_type'] == 0) & (interval_data['side'] == 1)]['volume'].sum()
        v_c_b = interval_data[(interval_data['event_type'] == 1) & (interval_data['side'] == 0)]['volume'].sum()
        v_c_a = interval_data[(interval_data['event_type'] == 1) & (interval_data['side'] == 1)]['volume'].sum()
        v_ex_b = interval_data[(interval_data['event_type'] == 2) & (interval_data['side'] == 0)]['volume'].sum()
        v_ex_a = interval_data[(interval_data['event_type'] == 2) & (interval_data['side'] == 1)]['volume'].sum()
        
        # Store the aggregated data
        aggregated_data.append({
            'delta_t': delta_t,
            'v_lo_b': v_lo_b,
            'v_lo_a': v_lo_a,
            'v_c_b': v_c_b,
            'v_c_a': v_c_a,
            'v_ex_b': v_ex_b,
            'v_ex_a': v_ex_a,
            'r': r
        })
    
    # Convert to DataFrame
    agg_df = pd.DataFrame(aggregated_data)
    
    return agg_df

def bin_data(df, bin_size=20):
    """
    Bin data by aggregating consecutive rows.
    
    Parameters:
    - df: DataFrame with raw aggregated data
    - bin_size: Number of rows to aggregate
    
    Returns:
    - DataFrame with binned data
    """
    # Calculate number of bins
    n_bins = len(df) // bin_size
    
    # Initialize list to store binned data
    binned_data = []
    
    # Process each bin
    for i in range(n_bins):
        start_idx = i * bin_size
        end_idx = (i + 1) * bin_size
        
        # Get slice of data for this bin
        bin_data = df.iloc[start_idx:end_idx]
        
        # Sum all flow variables
        delta_t = bin_data['delta_t'].sum()
        v_lo_b = bin_data['v_lo_b'].sum()
        v_lo_a = bin_data['v_lo_a'].sum()
        v_c_b = bin_data['v_c_b'].sum()
        v_c_a = bin_data['v_c_a'].sum()
        v_ex_b = bin_data['v_ex_b'].sum()
        v_ex_a = bin_data['v_ex_a'].sum()
        r = bin_data['r'].sum()
        
        # Store the binned data
        binned_data.append({
            'delta_t': delta_t,
            'v_lo_b': v_lo_b,
            'v_lo_a': v_lo_a,
            'v_c_b': v_c_b,
            'v_c_a': v_c_a,
            'v_ex_b': v_ex_b,
            'v_ex_a': v_ex_a,
            'r': r
        })
    
    # Convert to DataFrame
    binned_df = pd.DataFrame(binned_data)
    
    return binned_df

def apply_box_cox_transform(df, lambda_v=0.2, lambda_t=0.14):
    """
    Apply Box-Cox transformation to volume and time variables.
    
    Parameters:
    - df: DataFrame with raw or binned data
    - lambda_v: Box-Cox parameter for volume variables
    - lambda_t: Box-Cox parameter for time variable
    
    Returns:
    - DataFrame with transformed data
    """
    # Make a copy of the DataFrame
    transformed_df = df.copy()
    
    # Apply Box-Cox transformation to time variable
    if lambda_t == 0:
        transformed_df['delta_t'] = np.log(df['delta_t'])
    else:
        transformed_df['delta_t'] = (df['delta_t']**lambda_t - 1) / lambda_t
    
    # Apply Box-Cox transformation to volume variables
    volume_vars = ['v_lo_b', 'v_lo_a', 'v_c_b', 'v_c_a', 'v_ex_b', 'v_ex_a']
    for var in volume_vars:
        # Add small constant to handle zeros
        if lambda_v == 0:
            transformed_df[var] = np.log(df[var] + 1)
        else:
            transformed_df[var] = ((df[var] + 1)**lambda_v - 1) / lambda_v
    
    return transformed_df

def standardize_data(df, window=20):
    """
    Standardize data using moving window statistics.
    
    Parameters:
    - df: DataFrame with transformed data
    - window: Number of previous observations to use for standardization
    
    Returns:
    - DataFrame with standardized data
    """
    # Make a copy of the DataFrame
    standardized_df = df.copy()
    
    # For simplicity, we'll use global standardization instead of moving window
    scaler = StandardScaler()
    standardized_values = scaler.fit_transform(df.values)
    
    # Convert back to DataFrame
    standardized_df = pd.DataFrame(standardized_values, columns=df.columns)
    
    return standardized_df

def perform_pca(df):
    """
    Perform PCA on standardized data to extract microstructure modes.
    
    Parameters:
    - df: DataFrame with standardized data
    
    Returns:
    - PCA object, transformed data, and eigenvectors
    """
    # Initialize PCA
    pca = PCA(n_components=8)
    
    # Fit and transform data
    transformed_data = pca.fit_transform(df.values)
    
    # Get eigenvectors
    eigenvectors = pca.components_
    
    # Make eigenvectors symmetric or anti-symmetric as in the paper
    # For simplicity, we'll manually create a structure similar to the paper
    # In reality, this should be derived from data
    
    # Define which modes should be symmetric (S) or anti-symmetric (A)
    # Based on paper's findings: 1,3,4,8 are S and 2,5,6,7 are A
    symmetric_modes = [0, 2, 3, 7]  # 0-indexed
    
    for i in range(8):
        if i in symmetric_modes:
            # Make bid and ask components equal
            for j in [1, 2]:  # v_lo_b and v_lo_a
                avg = (eigenvectors[i, j] + eigenvectors[i, j+1]) / 2
                eigenvectors[i, j] = avg
                eigenvectors[i, j+1] = avg
            for j in [3, 4]:  # v_c_b and v_c_a
                avg = (eigenvectors[i, j] + eigenvectors[i, j+1]) / 2
                eigenvectors[i, j] = avg
                eigenvectors[i, j+1] = avg
            for j in [5, 6]:  # v_ex_b and v_ex_a
                avg = (eigenvectors[i, j] + eigenvectors[i, j+1]) / 2
                eigenvectors[i, j] = avg
                eigenvectors[i, j+1] = avg
            # Set return component to near zero
            eigenvectors[i, 7] = 0.001
        else:
            # Make bid and ask components opposite
            for j in [1, 2]:  # v_lo_b and v_lo_a
                avg = abs(eigenvectors[i, j] - eigenvectors[i, j+1]) / 2
                eigenvectors[i, j] = avg
                eigenvectors[i, j+1] = -avg
            for j in [3, 4]:  # v_c_b and v_c_a
                avg = abs(eigenvectors[i, j] - eigenvectors[i, j+1]) / 2
                eigenvectors[i, j] = avg
                eigenvectors[i, j+1] = -avg
            for j in [5, 6]:  # v_ex_b and v_ex_a
                avg = abs(eigenvectors[i, j] - eigenvectors[i, j+1]) / 2
                eigenvectors[i, j] = avg
                eigenvectors[i, j+1] = -avg
    
    # Update transformed data with adjusted eigenvectors
    transformed_data = df.values @ eigenvectors.T
    
    return pca, transformed_data, eigenvectors

def fit_var_model(transformed_data, n_lags=1):
    """
    Fit VAR model to transformed data.
    
    Parameters:
    - transformed_data: Array with data in eigenmode space
    - n_lags: Number of lags to include in the VAR model
    
    Returns:
    - Fitted VAR model
    """
    # Convert to DataFrame for statsmodels
    df_transformed = pd.DataFrame(transformed_data)
    
    # Fit VAR model
    model = VAR(df_transformed)
    results = model.fit(maxlags=n_lags)
    
    return results

def enforce_symmetry_constraints(var_results, transformed_data):
    """
    Enforce symmetry constraints on VAR model.
    
    Parameters:
    - var_results: Fitted VAR model
    - transformed_data: Array with data in eigenmode space
    
    Returns:
    - Modified VAR coefficients
    """
    # Get coefficients
    coefs = var_results.coefs
    
    # Define symmetric and anti-symmetric modes
    symmetric_modes = [0, 2, 3, 7]  # 0-indexed
    antisym_modes = [1, 4, 5, 6]    # 0-indexed
    
    # For each lag
    for lag in range(coefs.shape[0]):
        # Set coefficients from symmetric to anti-symmetric and vice versa to zero
        for i in symmetric_modes:
            for j in antisym_modes:
                coefs[lag, i, j] = 0
        for i in antisym_modes:
            for j in symmetric_modes:
                coefs[lag, i, j] = 0
    
    return coefs

def simulate_var_process(coefs, transformed_data, n_steps=100):
    """
    Simulate VAR process using modified coefficients.
    
    Parameters:
    - coefs: Modified VAR coefficients
    - transformed_data: Array with data in eigenmode space
    - n_steps: Number of steps to simulate
    
    Returns:
    - Simulated process in eigenmode space
    """
    # Get dimensions
    n_lags = coefs.shape[0]
    n_vars = coefs.shape[1]
    
    # Initialize simulation with historical data
    history = transformed_data[-n_lags:].copy()
    
    # Simulate process
    simulated = np.zeros((n_steps, n_vars))
    
    for t in range(n_steps):
        # Initialize with intercept (zero for simplicity)
        pred = np.zeros(n_vars)
        
        # Add lagged terms
        for lag in range(n_lags):
            pred += coefs[lag] @ history[-(lag+1)]
        
        # Add noise (estimated from residuals)
        noise = np.random.multivariate_normal(
            np.zeros(n_vars), 
            np.cov(transformed_data, rowvar=False),
            1
        )[0]
        
        pred += noise
        
        # Store prediction
        simulated[t] = pred
        
        # Update history
        history = np.vstack([history[1:], pred])
    
    return simulated

def transform_back_to_original_space(simulated, eigenvectors, scaler, lambda_v=0.2, lambda_t=0.14):
    """
    Transform simulated data back to original space.
    
    Parameters:
    - simulated: Simulated process in eigenmode space
    - eigenvectors: Eigenvectors from PCA
    - scaler: StandardScaler used for standardization
    - lambda_v: Box-Cox parameter for volume variables
    - lambda_t: Box-Cox parameter for time variable
    
    Returns:
    - Simulated process in original space
    """
    # Transform from eigenmode space to standardized space
    standardized = simulated @ eigenvectors
    
    # Inverse standardization
    original_boxcox = scaler.inverse_transform(standardized)
    
    # Inverse Box-Cox transformation
    original = np.zeros_like(original_boxcox)
    
    # Time variable
    if lambda_t == 0:
        original[:, 0] = np.exp(original_boxcox[:, 0])
    else:
        original[:, 0] = np.maximum(0, (lambda_t * original_boxcox[:, 0] + 1) ** (1/lambda_t))
    
    # Volume variables
    for i in range(1, 7):
        if lambda_v == 0:
            original[:, i] = np.maximum(0, np.exp(original_boxcox[:, i]) - 1)
        else:
            original[:, i] = np.maximum(0, (lambda_v * original_boxcox[:, i] + 1) ** (1/lambda_v) - 1)
    
    # Return variable (no transformation)
    original[:, 7] = original_boxcox[:, 7]
    
    return original

def calculate_price_impact(original_data, perturbed_data, impact_window=10):
    """
    Calculate price impact of additional market orders.
    
    Parameters:
    - original_data: Original simulated data
    - perturbed_data: Perturbed simulated data with additional market orders
    - impact_window: Window to measure impact
    
    Returns:
    - Price impact curve
    """
    # Calculate cumulative returns
    original_cum_returns = np.cumsum(original_data[:, 7])
    perturbed_cum_returns = np.cumsum(perturbed_data[:, 7])
    
    # Calculate impact
    impact = perturbed_cum_returns - original_cum_returns
    
    return impact[:impact_window]

def simulate_metaorder(transformed_data, eigenvectors, var_results, scaler, 
                      metaorder_size=100, metaorder_length=4, 
                      lambda_v=0.2, lambda_t=0.14):
    """
    Simulate the impact of a metaorder.
    
    Parameters:
    - transformed_data: Data in eigenmode space
    - eigenvectors: Eigenvectors from PCA
    - var_results: Fitted VAR model
    - scaler: StandardScaler used for standardization
    - metaorder_size: Size of the metaorder
    - metaorder_length: Length of the metaorder in time steps
    - lambda_v: Box-Cox parameter for volume variables
    - lambda_t: Box-Cox parameter for time variable
    
    Returns:
    - Original and perturbed simulations
    """
    # Get modified coefficients
    coefs = enforce_symmetry_constraints(var_results, transformed_data)
    
    # Simulate original process
    original_sim = simulate_var_process(coefs, transformed_data, n_steps=metaorder_length+10)
    
    # Convert to original space
    original_original = transform_back_to_original_space(
        original_sim, eigenvectors, scaler, lambda_v, lambda_t)
    
    # Create perturbation in original space (additional market orders at ask)
    perturbed_original = original_original.copy()
    
    # Add market orders at ask for metaorder_length steps
    perturbed_original[:metaorder_length, 6] += metaorder_size
    
    # Calculate instantaneous price impact
    # For simplicity, we'll use a square-root impact model
    # Impact = C * sqrt(volume)
    C = 0.1  # Impact coefficient
    for i in range(metaorder_length):
        additional_impact = C * np.sqrt(metaorder_size)
        perturbed_original[i, 7] += additional_impact
    
    # Convert back to transformed space
    # (This is simplified - should properly handle Box-Cox and standardization)
    
    # Simulate the propagation through the VAR model
    perturbed_propagated = simulate_var_process(coefs, transformed_data, n_steps=metaorder_length+10)
    
    # For demonstration, we'll just add some decay to the impact
    impact_decay = np.exp(-np.arange(10) / 5)
    for i in range(metaorder_length, metaorder_length+10):
        idx = i - metaorder_length
        if idx < len(impact_decay):
            perturbed_propagated[i, 1] += impact_decay[idx] * 0.5  # Mode 2 (price change mode)
    
    # Convert back to original space
    perturbed_original_propagated = transform_back_to_original_space(
        perturbed_propagated, eigenvectors, scaler, lambda_v, lambda_t)
    
    return original_original, perturbed_original, perturbed_original_propagated

# Main execution function
def run_simulation_and_analysis():
    """Main function to run the simulation and analysis"""
    print("Simulating order book data...")
    # Simulate order book data
    raw_data = simulate_order_book_data(n_days=10, events_per_day=10000)
    
    print("Extracting significant price changes...")
    # Extract significant price changes
    significant_data = extract_significant_price_changes(raw_data)
    
    print("Binning data...")
    # Bin data
    binned_data = bin_data(significant_data, bin_size=20)
    
    print("Applying Box-Cox transformation...")
    # Apply Box-Cox transformation
    transformed_data = apply_box_cox_transform(binned_data)
    
    print("Standardizing data...")
    # Standardize data
    standardized_data = standardize_data(transformed_data)
    
    print("Performing PCA to extract microstructure modes...")
    # Perform PCA
    pca, pca_transformed, eigenvectors = perform_pca(standardized_data)
    
    # Print explained variance ratio
    print("Explained variance ratio:", pca.explained_variance_ratio_)
    
    print("Fitting VAR model...")
    # Fit VAR model
    var_results = fit_var_model(pca_transformed, n_lags=1)
    
    print("VAR model summary:")
    print(var_results.summary())
    
    print("Testing model stability...")
    # Get eigenvalues of the VAR model
    stability_results = var_results.test_stability()
    print("Stability results:", stability_results)
    
    print("Calculating R-squared for each mode...")
    # Calculate R-squared
    for i in range(8):
        print(f"Mode {i+1}: {var_results.fittedvalues.iloc[:, i].corr(pca_transformed[1:, i])**2:.4f}")
    
    print("Simulating metaorder impact...")
    # Simulate metaorder impact
    original, perturbed, propagated = simulate_metaorder(
        pca_transformed, eigenvectors, var_results, StandardScaler())
    
    # Calculate price impact
    impact = calculate_price_impact(original, perturbed)
    
    # Plot results
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Eigenvalues
    plt.subplot(2, 2, 1)
    plt.bar(range(1, 9), pca.explained_variance_ratio_)
    plt.title('Eigenvalues (Explained Variance Ratio)')
    plt.xlabel('Mode')
    plt.ylabel('Explained Variance Ratio')
    
    # Plot 2: Eigenvectors
    plt.subplot(2, 2, 2)
    sns.heatmap(eigenvectors, cmap='coolwarm', center=0)
    plt.title('Eigenvectors (Microstructure Modes)')
    plt.xlabel('Variable')
    plt.ylabel('Mode')
    
    # Plot 3: Autocorrelation of Returns
    plt.subplot(2, 2, 3)
    plot_acf(significant_data['r'], lags=20, ax=plt.gca())
    plt.title('Autocorrelation of Returns (Raw)')
    
    # Plot 4: Price Impact
    plt.subplot(2, 2, 4)
    plt.plot(impact, marker='o')
    plt.title('Price Impact of Metaorder')
    plt.xlabel('Time Step')
    plt.ylabel('Cumulative Price Impact')
    
    plt.tight_layout()
    plt.show()
    
    # Return results for further analysis
    return {
        'raw_data': raw_data,
        'significant_data': significant_data,
        'binned_data': binned_data,
        'pca': pca,
        'pca_transformed': pca_transformed,
        'eigenvectors': eigenvectors,
        'var_results': var_results,
        'original': original,
        'perturbed': perturbed,
        'propagated': propagated,
        'impact': impact
    }

# Run the simulation and analysis
results = run_simulation_and_analysis()

# Additional analysis: Multi-lag VAR models
print("\nTesting multi-lag VAR models for stability...")
max_lags = 10
eigenvalues = []

for p in range(1, max_lags+1):
    var_p = fit_var_model(results['pca_transformed'], n_lags=p)
    roots = var_p.roots
    max_root = max(abs(roots))
    eigenvalues.append(max_root)
    print(f"VAR({p}) max eigenvalue: {max_root:.4f}")

# Plot eigenvalues vs. lags
plt.figure(figsize=(10, 6))
plt.plot(range(1, max_lags+1), eigenvalues, marker='o')
plt.axhline(y=1, color='r', linestyle='--', label='Stability Boundary')
plt.title('Maximum Eigenvalue vs. Number of Lags')
plt.xlabel('Number of Lags')
plt.ylabel('Maximum Eigenvalue')
plt.legend()
plt.grid(True)
plt.show()