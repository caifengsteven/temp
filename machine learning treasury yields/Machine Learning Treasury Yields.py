import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import NMF
from sklearn.cluster import KMeans
from scipy.linalg import svd
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set seed for reproducibility
np.random.seed(42)

# Function to simulate Treasury yield data
def simulate_treasury_yields(n_days=500, n_maturities=12, noise_level=0.01):
    """
    Simulate Treasury yield data with realistic properties.
    
    Parameters:
    - n_days: Number of days to simulate
    - n_maturities: Number of maturities (default 12, like real Treasury data)
    - noise_level: Level of noise to add to the simulated data
    
    Returns:
    - DataFrame with simulated Treasury yields
    """
    # Define maturities in months
    maturities = np.array([1, 2, 3, 6, 12, 24, 36, 60, 84, 120, 240, 360])
    
    # Start date
    start_date = datetime(2018, 10, 16)
    dates = [start_date + timedelta(days=i) for i in range(n_days)]
    dates = [d for d in dates if d.weekday() < 5]  # Only business days
    
    # Create level factor (overall interest rate level)
    level = np.zeros(len(dates))
    level[0] = 2.0  # Starting level around 2%
    for i in range(1, len(dates)):
        level[i] = level[i-1] + np.random.normal(0, 0.01)  # Small daily changes
    level = np.maximum(level, 0.1)  # Keep rates positive
    
    # Create slope factor (difference between long and short term rates)
    slope = np.zeros(len(dates))
    slope[0] = 1.0  # Starting with positive slope
    for i in range(1, len(dates)):
        slope[i] = slope[i-1] + np.random.normal(0, 0.008)
    
    # Create curvature factor
    curvature = np.zeros(len(dates))
    curvature[0] = -0.5  # Starting with negative curvature
    for i in range(1, len(dates)):
        curvature[i] = curvature[i-1] + np.random.normal(0, 0.005)
    
    # Generate yields based on Nelson-Siegel model
    lambda_factor = 0.0609  # Parameter from Diebold-Li paper
    yields = np.zeros((len(dates), len(maturities)))
    
    for d in range(len(dates)):
        for i, mat in enumerate(maturities):
            # Convert maturity from months to years for the model
            t = mat / 12.0
            
            # Level component
            l_component = level[d]
            
            # Slope component (short-term factor)
            s_component = slope[d] * (1 - np.exp(-lambda_factor * t)) / (lambda_factor * t)
            
            # Curvature component (medium-term factor)
            c_component = curvature[d] * ((1 - np.exp(-lambda_factor * t)) / (lambda_factor * t) - np.exp(-lambda_factor * t))
            
            # Add components together with some noise
            yields[d, i] = l_component + s_component + c_component + np.random.normal(0, noise_level)
    
    # Ensure yields are positive
    yields = np.maximum(yields, 0.001)
    
    # Create DataFrame
    df = pd.DataFrame(yields, index=dates, columns=[f"{m}M" if m < 12 else f"{m//12}Y" for m in maturities])
    df.index.name = 'Date'
    
    return df

# Function to calculate effective rank (eRank)
def calculate_erank(eigenvalues, exclude_first=False):
    """
    Calculate the effective rank (eRank) of a matrix using its eigenvalues.
    
    Parameters:
    - eigenvalues: Array of eigenvalues
    - exclude_first: Whether to exclude the first (largest) eigenvalue
    
    Returns:
    - eRank value
    """
    # Keep only positive eigenvalues
    eigenvalues = eigenvalues[eigenvalues > 0]
    
    # Exclude first eigenvalue if requested
    if exclude_first and len(eigenvalues) > 1:
        eigenvalues = eigenvalues[1:]
    
    # Normalize eigenvalues to get probabilities
    p = eigenvalues / np.sum(eigenvalues)
    
    # Calculate Shannon entropy
    h = -np.sum(p * np.log(p))
    
    # Exponential of entropy is the eRank
    er = np.exp(h)
    
    # Add 1 if first eigenvalue was excluded
    if exclude_first:
        er += 1
        
    return er

# Function to perform vanilla NMF
def vanilla_nmf(yields_data, k=2, n_runs=100):
    """
    Perform vanilla NMF (without denoising) on yield data.
    
    Parameters:
    - yields_data: DataFrame with yield data
    - k: Number of factors
    - n_runs: Number of NMF runs to average over
    
    Returns:
    - Dictionary with weights, factors, errors, and fits
    """
    print(f"Running vanilla NMF with k={k}, n_runs={n_runs}")
    
    # Convert to numpy array
    Y = yields_data.values.T  # Matrix of shape (n_maturities, n_days)
    n_maturities, n_days = Y.shape
    
    # Initialize matrices to store results
    weights_runs = np.zeros((n_runs, n_maturities, k))
    factors_runs = np.zeros((n_runs, k, n_days))
    
    # Run NMF multiple times
    for i in range(n_runs):
        # Initialize and fit NMF model
        model = NMF(n_components=k, init='random', random_state=i, max_iter=1000)
        W = model.fit_transform(Y)  # Weights: n_maturities x k
        H = model.components_  # Factors: k x n_days
        
        # Normalize weights to sum to 1
        w_norms = np.sum(W, axis=0)
        W_norm = W / w_norms
        H_norm = H * w_norms[:, np.newaxis]
        
        # Store this run's results
        weights_runs[i] = W_norm
        factors_runs[i] = H_norm
    
    # Align factors from different runs using k-means clustering
    # Bootstrap weights
    weights_bootstrap = weights_runs.reshape(n_runs * k, n_maturities).T  # Shape: (n_maturities, n_runs*k)
    
    # Cluster the bootstrapped weights
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(weights_bootstrap.T)
    
    # Regroup the weights and factors based on clustering
    aligned_weights = np.zeros((n_maturities, k))
    aligned_weights_std = np.zeros((n_maturities, k))
    aligned_factors = np.zeros((k, n_days))
    aligned_factors_std = np.zeros((k, n_days))
    
    for j in range(k):
        # Find which columns belong to cluster j
        indices = np.where(clusters == j)[0]
        
        # Ensure some weights were assigned to this cluster
        if len(indices) == 0:
            print(f"Warning: Cluster {j} is empty. Try reducing k.")
            continue
            
        # Extract original run and factor index
        run_indices = indices // k
        factor_indices = indices % k
        
        # Get the corresponding weights and factors
        cluster_weights = np.array([weights_runs[r, :, f] for r, f in zip(run_indices, factor_indices)])
        cluster_factors = np.array([factors_runs[r, f, :] for r, f in zip(run_indices, factor_indices)])
        
        # Compute mean and std
        aligned_weights[:, j] = np.mean(cluster_weights, axis=0)
        aligned_weights_std[:, j] = np.std(cluster_weights, axis=0)
        aligned_factors[j, :] = np.mean(cluster_factors, axis=0)
        aligned_factors_std[j, :] = np.std(cluster_factors, axis=0)
    
    # Reconstruct the yields matrix
    Y_hat = aligned_weights @ aligned_factors
    
    # Calculate fits
    correlations = np.zeros(n_maturities)
    errors = np.zeros(n_maturities)
    
    for i in range(n_maturities):
        correlations[i] = np.corrcoef(Y[i, :], Y_hat[i, :])[0, 1]
        errors[i] = np.sum((Y[i, :] - Y_hat[i, :])**2)
    
    return {
        'weights': aligned_weights,
        'weights_std': aligned_weights_std,
        'factors': aligned_factors,
        'factors_std': aligned_factors_std,
        'correlations': correlations,
        'errors': errors
    }

# Function to perform denoised NMF
def denoised_nmf(yields_data, k=2, n_runs=100, denoising_method=1):
    """
    Perform denoised NMF on yield data.
    
    Parameters:
    - yields_data: DataFrame with yield data
    - k: Number of factors
    - n_runs: Number of NMF runs to average over
    - denoising_method: 1 for min subtraction, 2 for max subtraction
    
    Returns:
    - Dictionary with weights, factors, errors, fits and the level
    """
    print(f"Running denoised NMF with method={denoising_method}, k={k}, n_runs={n_runs}")
    
    # Convert to numpy array
    Y = yields_data.values.T  # Matrix of shape (n_maturities, n_days)
    n_maturities, n_days = Y.shape
    
    # Calculate the level based on denoising method
    if denoising_method == 1:
        # Method 1: Use minimum yield on each date
        level = np.min(Y, axis=0)
        Y_denoised = Y - level
    elif denoising_method == 2:
        # Method 2: Use maximum yield on each date
        level = np.max(Y, axis=0)
        Y_denoised = level - Y
    else:
        raise ValueError("denoising_method must be 1 or 2")
    
    # Initialize matrices to store results
    weights_runs = np.zeros((n_runs, n_maturities, k))
    factors_runs = np.zeros((n_runs, k, n_days))
    
    # Run NMF multiple times
    for i in range(n_runs):
        # Initialize and fit NMF model
        model = NMF(n_components=k, init='random', random_state=i, max_iter=1000)
        W = model.fit_transform(Y_denoised)  # Weights: n_maturities x k
        H = model.components_  # Factors: k x n_days
        
        # Normalize weights to sum to 1
        w_norms = np.sum(W, axis=0)
        W_norm = W / w_norms
        H_norm = H * w_norms[:, np.newaxis]
        
        # Store this run's results
        weights_runs[i] = W_norm
        factors_runs[i] = H_norm
    
    # Align factors from different runs using k-means clustering
    # Bootstrap weights
    weights_bootstrap = weights_runs.reshape(n_runs * k, n_maturities).T  # Shape: (n_maturities, n_runs*k)
    
    # Cluster the bootstrapped weights
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(weights_bootstrap.T)
    
    # Regroup the weights and factors based on clustering
    aligned_weights = np.zeros((n_maturities, k))
    aligned_weights_std = np.zeros((n_maturities, k))
    aligned_factors = np.zeros((k, n_days))
    aligned_factors_std = np.zeros((k, n_days))
    
    for j in range(k):
        # Find which columns belong to cluster j
        indices = np.where(clusters == j)[0]
        
        # Ensure some weights were assigned to this cluster
        if len(indices) == 0:
            print(f"Warning: Cluster {j} is empty. Try reducing k.")
            continue
            
        # Extract original run and factor index
        run_indices = indices // k
        factor_indices = indices % k
        
        # Get the corresponding weights and factors
        cluster_weights = np.array([weights_runs[r, :, f] for r, f in zip(run_indices, factor_indices)])
        cluster_factors = np.array([factors_runs[r, f, :] for r, f in zip(run_indices, factor_indices)])
        
        # Compute mean and std
        aligned_weights[:, j] = np.mean(cluster_weights, axis=0)
        aligned_weights_std[:, j] = np.std(cluster_weights, axis=0)
        aligned_factors[j, :] = np.mean(cluster_factors, axis=0)
        aligned_factors_std[j, :] = np.std(cluster_factors, axis=0)
    
    # Reconstruct the denoised yields matrix
    Y_hat_denoised = aligned_weights @ aligned_factors
    
    # Reconstruct the original yields matrix
    if denoising_method == 1:
        Y_hat = Y_hat_denoised + level
    else:
        Y_hat = level - Y_hat_denoised
    
    # Calculate fits for denoised matrix
    correlations_denoised = np.zeros(n_maturities)
    errors_denoised = np.zeros(n_maturities)
    
    for i in range(n_maturities):
        correlations_denoised[i] = np.corrcoef(Y_denoised[i, :], Y_hat_denoised[i, :])[0, 1]
        errors_denoised[i] = np.sum((Y_denoised[i, :] - Y_hat_denoised[i, :])**2)
    
    # Calculate fits for original matrix
    correlations = np.zeros(n_maturities)
    errors = np.zeros(n_maturities)
    
    for i in range(n_maturities):
        correlations[i] = np.corrcoef(Y[i, :], Y_hat[i, :])[0, 1]
        errors[i] = np.sum((Y[i, :] - Y_hat[i, :])**2)
    
    return {
        'weights': aligned_weights,
        'weights_std': aligned_weights_std,
        'factors': aligned_factors,
        'factors_std': aligned_factors_std,
        'level': level,
        'correlations_denoised': correlations_denoised,
        'errors_denoised': errors_denoised,
        'correlations': correlations,
        'errors': errors
    }

# Function to perform clustering-based analysis
def perform_clustering(yields_data, k=2):
    """
    Perform clustering-based analysis on yield data.
    
    Parameters:
    - yields_data: DataFrame with yield data
    - k: Number of clusters
    
    Returns:
    - Dictionary with clusters, weights, factors, and fits
    """
    print(f"Running clustering with k={k}")
    
    # Convert to numpy array
    Y = yields_data.values.T  # Matrix of shape (n_maturities, n_days)
    n_maturities, n_days = Y.shape
    
    # Normalize by volatility
    Y_norm = Y / np.std(Y, axis=1, keepdims=True)
    
    # Perform k-means clustering
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(Y_norm)
    
    # Initialize matrices for weights and factors
    weights = np.zeros((n_maturities, k))
    factors = np.zeros((k, n_days))
    
    # For each cluster, perform one-factor NMF (equivalent to rank-1 SVD)
    for j in range(k):
        # Get indices of maturities in this cluster
        indices = np.where(clusters == j)[0]
        
        if len(indices) == 0:
            continue
            
        # Extract data for this cluster
        Y_cluster = Y[indices, :]
        
        # Perform SVD
        U, S, Vt = svd(Y_cluster, full_matrices=False)
        
        # The first components give us the rank-1 approximation
        weights[indices, j] = U[:, 0] * np.sqrt(S[0])
        factors[j, :] = Vt[0, :]
        
        # Ensure non-negativity
        weights[indices, j] = np.abs(weights[indices, j])
        factors[j, :] = np.abs(factors[j, :])
    
    # Normalize weights to sum to 1 for each cluster
    for j in range(k):
        indices = np.where(clusters == j)[0]
        if len(indices) > 0:
            weight_sum = np.sum(weights[indices, j])
            weights[indices, j] /= weight_sum
            factors[j, :] *= weight_sum
    
    # Reconstruct the yields matrix
    Y_hat = np.zeros_like(Y)
    for j in range(k):
        indices = np.where(clusters == j)[0]
        Y_hat[indices, :] = np.outer(weights[indices, j], factors[j, :])
    
    # Calculate fits
    correlations = np.zeros(n_maturities)
    errors = np.zeros(n_maturities)
    
    for i in range(n_maturities):
        correlations[i] = np.corrcoef(Y[i, :], Y_hat[i, :])[0, 1]
        errors[i] = np.sum((Y[i, :] - Y_hat[i, :])**2)
    
    return {
        'clusters': clusters,
        'weights': weights,
        'factors': factors,
        'correlations': correlations,
        'errors': errors
    }

# Function to plot weights for NMF
def plot_nmf_weights(weights, weights_std, maturities, title):
    """Plot the weights from NMF with error bars."""
    plt.figure(figsize=(10, 6))
    
    # Convert maturities to log scale for x-axis
    log_maturities = np.log(maturities)
    
    for j in range(weights.shape[1]):
        plt.errorbar(log_maturities, weights[:, j], yerr=weights_std[:, j], 
                    fmt='-o', capsize=5, label=f'Factor {j+1}')
    
    plt.xlabel('Log(Maturity)')
    plt.ylabel('Weight')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# Function to plot factors for NMF
def plot_nmf_factors(factors, factors_std, dates, title):
    """Plot the factors from NMF with error bands."""
    plt.figure(figsize=(12, 6))
    
    for j in range(factors.shape[0]):
        plt.plot(dates, factors[j, :], label=f'Factor {j+1}')
        plt.fill_between(dates, 
                         factors[j, :] - factors_std[j, :],
                         factors[j, :] + factors_std[j, :],
                         alpha=0.2)
    
    plt.xlabel('Date')
    plt.ylabel('Factor Value')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# Function to plot clustering results
def plot_clustering_results(clusters, weights, factors, maturities, dates, title):
    """Plot the clusters, weights, and factors from clustering."""
    # Plot clustering assignment
    plt.figure(figsize=(10, 4))
    plt.scatter(range(len(maturities)), clusters, c=clusters, cmap='viridis')
    plt.yticks(range(max(clusters)+1))
    plt.xticks(range(len(maturities)), [f"{m}" for m in maturities], rotation=45)
    plt.xlabel('Maturity')
    plt.ylabel('Cluster')
    plt.title(f'Clustering of Maturities: {title}')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Plot factors
    plt.figure(figsize=(12, 6))
    for j in range(factors.shape[0]):
        plt.plot(dates, factors[j, :], label=f'Factor {j+1}')
    
    plt.xlabel('Date')
    plt.ylabel('Factor Value')
    plt.title(f'Factors from Clustering: {title}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Plot weights
    plt.figure(figsize=(10, 6))
    for j in range(weights.shape[1]):
        plt.scatter(maturities, weights[:, j], label=f'Factor {j+1}')
        # Connect non-zero weights with lines
        nonzero = weights[:, j] > 0
        if sum(nonzero) > 1:
            plt.plot(maturities[nonzero], weights[nonzero, j], '-')
    
    plt.xlabel('Maturity')
    plt.ylabel('Weight')
    plt.title(f'Weights from Clustering: {title}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# Function to analyze and compare eRank
def analyze_erank(yields_data):
    """Analyze the effective rank of yield correlations."""
    # Calculate correlation matrix
    corr_matrix = np.corrcoef(yields_data.values.T)
    
    # Calculate eigenvalues
    eigenvalues = np.linalg.eigvalsh(corr_matrix)
    eigenvalues = eigenvalues[::-1]  # Sort in descending order
    
    # Calculate eRank
    erank = calculate_erank(eigenvalues, exclude_first=False)
    moderank = calculate_erank(eigenvalues, exclude_first=True)
    
    # Calculate average pairwise correlation
    np.fill_diagonal(corr_matrix, np.nan)
    avg_corr = np.nanmean(corr_matrix)
    
    print(f"Average pairwise correlation: {avg_corr:.4f}")
    print(f"eRank: {erank:.2f}")
    print(f"ModeRank (eRank without first component): {moderank:.2f}")
    
    # Visualize correlation matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Correlation Matrix of Treasury Yields')
    plt.tight_layout()
    plt.show()
    
    # Plot eigenvalue spectrum
    plt.figure(figsize=(10, 6))
    plt.bar(range(1, len(eigenvalues)+1), eigenvalues)
    plt.xlabel('Eigenvalue Number')
    plt.ylabel('Magnitude')
    plt.title('Eigenvalue Spectrum of Yield Correlation Matrix')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    return erank, moderank, avg_corr

# Function to calculate slope and curvature
def calculate_yield_metrics(yields_data):
    """Calculate level, slope, and curvature."""
    # Level: 10-year yield
    level = yields_data['10Y'].values
    
    # Slope: 10Y - 3M
    slope = yields_data['10Y'].values - yields_data['3M'].values
    
    # Curvature: 2 * 2Y - 10Y - 3M
    curvature = 2 * yields_data['2Y'].values - yields_data['10Y'].values - yields_data['3M'].values
    
    return level, slope, curvature

# Function to analyze factor correlations
def analyze_factor_correlations(factors, level, slope, curvature):
    """Analyze correlations between factors and common yield curve metrics."""
    # Stack all factors and metrics
    all_factors = np.vstack([factors, level.reshape(1, -1), slope.reshape(1, -1), curvature.reshape(1, -1)])
    
    # Calculate correlation matrix
    corr_matrix = np.corrcoef(all_factors)
    
    # Create labels
    labels = [f'Factor {i+1}' for i in range(factors.shape[0])] + ['Level', 'Slope', 'Curvature']
    
    # Print correlations
    print("Factor correlation matrix:")
    for i in range(factors.shape[0]):
        for j in range(factors.shape[0]):
            print(f"Corr(Factor {i+1}, Factor {j+1}) = {corr_matrix[i, j]:.4f}")
    
    print("\nCorrelations with yield curve metrics:")
    for i in range(factors.shape[0]):
        print(f"Corr(Factor {i+1}, Level) = {corr_matrix[i, factors.shape[0]]:.4f}")
        print(f"Corr(Factor {i+1}, Slope) = {corr_matrix[i, factors.shape[0]+1]:.4f}")
        print(f"Corr(Factor {i+1}, Curvature) = {corr_matrix[i, factors.shape[0]+2]:.4f}")
    
    # Visualize correlation matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, xticklabels=labels, yticklabels=labels)
    plt.title('Correlation Matrix of Factors and Yield Metrics')
    plt.tight_layout()
    plt.show()
    
    return corr_matrix, labels

# Main analysis function
def analyze_treasury_yields(yields_data, k_values=[2, 3]):
    """Perform full analysis on Treasury yield data."""
    # Set maturities
    maturities = np.array([1, 2, 3, 6, 12, 24, 36, 60, 84, 120, 240, 360])
    maturity_labels = [f"{m}M" if m < 12 else f"{m//12}Y" for m in maturities]
    
    # Calculate yield metrics
    level, slope, curvature = calculate_yield_metrics(yields_data)
    
    # Print correlation between slope and curvature
    print(f"Correlation between slope and curvature: {np.corrcoef(slope, curvature)[0, 1]:.4f}")
    
    # 1. Analyze effective rank
    print("\n--- Effective Rank Analysis ---")
    erank, moderank, avg_corr = analyze_erank(yields_data)
    
    results = {}
    
    # 2. Vanilla NMF
    print("\n--- Vanilla NMF Analysis ---")
    for k in k_values:
        vanilla_results = vanilla_nmf(yields_data, k=k, n_runs=50)
        results[f'vanilla_k{k}'] = vanilla_results
        
        # Plot weights and factors
        plot_nmf_weights(vanilla_results['weights'], vanilla_results['weights_std'], 
                       maturities, f'Vanilla NMF Weights (k={k})')
        plot_nmf_factors(vanilla_results['factors'], vanilla_results['factors_std'], 
                        yields_data.index, f'Vanilla NMF Factors (k={k})')
        
        # Analyze factor correlations
        print(f"\nVanilla NMF (k={k}) factor correlations:")
        analyze_factor_correlations(vanilla_results['factors'], level, slope, curvature)
    
    # 3. Denoised NMF (Method 1: min subtraction)
    print("\n--- Denoised NMF (Method 1) Analysis ---")
    for k in k_values:
        denoised_results = denoised_nmf(yields_data, k=k, n_runs=50, denoising_method=1)
        results[f'denoised1_k{k}'] = denoised_results
        
        # Plot weights and factors
        plot_nmf_weights(denoised_results['weights'], denoised_results['weights_std'], 
                       maturities, f'Denoised NMF (Method 1) Weights (k={k})')
        plot_nmf_factors(denoised_results['factors'], denoised_results['factors_std'], 
                        yields_data.index, f'Denoised NMF (Method 1) Factors (k={k})')
        
        # Plot level factor
        plt.figure(figsize=(12, 6))
        plt.plot(yields_data.index, denoised_results['level'], label='Level Factor')
        plt.xlabel('Date')
        plt.ylabel('Yield')
        plt.title(f'Level Factor (Min Yield) for Denoised NMF (Method 1, k={k})')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()
        
        # Combine level with factors for correlation analysis
        all_factors = np.vstack([denoised_results['factors'], denoised_results['level'].reshape(1, -1)])
        factor_labels = [f'Factor {i+1}' for i in range(k)] + ['Level']
        
        print(f"\nDenoised NMF (Method 1, k={k}) factor correlations:")
        analyze_factor_correlations(all_factors, level, slope, curvature)
    
    # 4. Denoised NMF (Method 2: max subtraction)
    print("\n--- Denoised NMF (Method 2) Analysis ---")
    for k in k_values:
        denoised_results = denoised_nmf(yields_data, k=k, n_runs=50, denoising_method=2)
        results[f'denoised2_k{k}'] = denoised_results
        
        # Plot weights and factors
        plot_nmf_weights(denoised_results['weights'], denoised_results['weights_std'], 
                       maturities, f'Denoised NMF (Method 2) Weights (k={k})')
        plot_nmf_factors(denoised_results['factors'], denoised_results['factors_std'], 
                        yields_data.index, f'Denoised NMF (Method 2) Factors (k={k})')
        
        # Plot level factor
        plt.figure(figsize=(12, 6))
        plt.plot(yields_data.index, denoised_results['level'], label='Level Factor')
        plt.xlabel('Date')
        plt.ylabel('Yield')
        plt.title(f'Level Factor (Max Yield) for Denoised NMF (Method 2, k={k})')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()
        
        # Combine level with factors for correlation analysis
        all_factors = np.vstack([denoised_results['factors'], denoised_results['level'].reshape(1, -1)])
        factor_labels = [f'Factor {i+1}' for i in range(k)] + ['Level']
        
        print(f"\nDenoised NMF (Method 2, k={k}) factor correlations:")
        analyze_factor_correlations(all_factors, level, slope, curvature)
    
    # 5. Clustering-based analysis
    print("\n--- Clustering Analysis ---")
    for k in k_values:
        clustering_results = perform_clustering(yields_data, k=k)
        results[f'clustering_k{k}'] = clustering_results
        
        # Plot clustering results
        plot_clustering_results(clustering_results['clusters'], clustering_results['weights'], 
                              clustering_results['factors'], maturities, yields_data.index, 
                              f'k={k}')
        
        print(f"\nClustering (k={k}) factor correlations:")
        analyze_factor_correlations(clustering_results['factors'], level, slope, curvature)
    
    # 6. Compare fits
    print("\n--- Fit Comparison ---")
    fit_comparison = pd.DataFrame()
    
    for name, result in results.items():
        if 'correlations' in result:
            fit_comparison[f'{name}_corr'] = result['correlations']
        if 'errors' in result:
            fit_comparison[f'{name}_error'] = result['errors']
    
    fit_comparison.index = maturity_labels
    print(fit_comparison)
    
    # Plot correlation comparison
    corr_cols = [col for col in fit_comparison.columns if 'corr' in col]
    plt.figure(figsize=(12, 6))
    fit_comparison[corr_cols].plot(kind='bar')
    plt.title('Correlation Comparison by Maturity')
    plt.ylabel('Correlation')
    plt.xlabel('Maturity')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='lower center')
    plt.tight_layout()
    plt.show()
    
    return results, fit_comparison

# Test with simulated data
if __name__ == "__main__":
    # Simulate Treasury yield data
    yields_data = simulate_treasury_yields(n_days=300, noise_level=0.01)
    
    # Display the first few rows
    print("Simulated Treasury Yield Data (first 5 days):")
    print(yields_data.head())
    
    # Plot the yield curves over time
    plt.figure(figsize=(12, 6))
    for date in yields_data.index[::50]:  # Plot every 50th day
        plt.plot(yields_data.columns, yields_data.loc[date], label=date.strftime('%Y-%m-%d'))
    plt.xlabel('Maturity')
    plt.ylabel('Yield (%)')
    plt.title('Simulated Treasury Yield Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Analyze the simulated yield data
    results, fit_comparison = analyze_treasury_yields(yields_data, k_values=[2, 3])