import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# Function to calculate the extended correlation coefficient
def extended_correlation(data, use_mean=True):
    """
    Calculate the extended correlation coefficient for multiple variables.
    
    Parameters:
    - data: DataFrame or ndarray with variables as columns
    - use_mean: If True, use mean of maximal eigenvalues; otherwise use single computation
    
    Returns:
    - Extended correlation coefficient
    """
    if isinstance(data, pd.DataFrame):
        data = data.values
    
    n = data.shape[1]  # Number of variables
    
    if use_mean:
        # Calculate maximal eigenvalues for growing samples
        max_eigenvalues = []
        for i in range(10, data.shape[0] + 1, 10):  # Increment by 10 for efficiency
            subset = data[:i, :]
            # Calculate correlation matrix
            corr_matrix = np.corrcoef(subset, rowvar=False)
            # Calculate eigenvalues
            eigenvalues = np.linalg.eigvalsh(corr_matrix)
            max_eigenvalues.append(np.max(eigenvalues))
        
        # Calculate mean of maximal eigenvalues
        mean_max_eigenvalue = np.mean(max_eigenvalues)
    else:
        # Calculate correlation matrix for the entire dataset
        corr_matrix = np.corrcoef(data, rowvar=False)
        # Calculate norm (equivalent to maximal eigenvalue for symmetric matrices)
        mean_max_eigenvalue = np.linalg.norm(corr_matrix, 2)
    
    # Calculate extended correlation coefficient
    rho = (mean_max_eigenvalue - 1) / (n - 1)
    return rho

# Function to calculate noise in a dataset
def calculate_noise(data, target_column=-1):
    """
    Calculate predictor noise and labeling noise in a dataset.
    
    Parameters:
    - data: DataFrame or ndarray with variables as columns
    - target_column: Index of the target variable (default: last column)
    
    Returns:
    - predictor_noise: Noise in the predictors
    - labeling_noise: Noise in the labeling
    """
    if isinstance(data, pd.DataFrame):
        data = data.values
    
    # Separate predictors and target
    predictors = np.delete(data, target_column, axis=1)
    full_data = data
    
    # Calculate extended correlation for predictors
    rho_predictors = extended_correlation(predictors, use_mean=False)
    
    # Calculate extended correlation for full data
    rho_full = extended_correlation(full_data, use_mean=False)
    
    # Calculate noise
    predictor_noise = 1 - rho_predictors
    labeling_noise = max(0, (1 - rho_full) - predictor_noise)
    
    return predictor_noise, labeling_noise

# Function to generate FC-datasets
def generate_fc_dataset(n_samples, n_vars, correlation_pattern=None):
    """
    Generate a fully-correlated dataset.
    
    Parameters:
    - n_samples: Number of samples
    - n_vars: Number of variables
    - correlation_pattern: List indicating signs (1 or -1) for correlations
    
    Returns:
    - DataFrame with the fully-correlated dataset
    """
    if correlation_pattern is None:
        # Default: all positive correlations
        correlation_pattern = [1] * n_vars
    
    # Generate base variable
    base = np.random.normal(0, 1, n_samples)
    
    # Generate other variables based on correlation pattern
    data = np.zeros((n_samples, n_vars))
    data[:, 0] = base  # First variable is the base
    
    for i in range(1, n_vars):
        data[:, i] = correlation_pattern[i] * base
    
    return pd.DataFrame(data, columns=[f'Var{i+1}' for i in range(n_vars)])

# Function to generate FU-dataset
def generate_fu_dataset(n_samples, n_vars):
    """
    Generate a fully-uncorrelated dataset.
    
    Parameters:
    - n_samples: Number of samples
    - n_vars: Number of variables
    
    Returns:
    - DataFrame with the fully-uncorrelated dataset
    """
    data = np.random.normal(0, 1, (n_samples, n_vars))
    return pd.DataFrame(data, columns=[f'Var{i+1}' for i in range(n_vars)])

# Function to add noise to a dataset
def add_noise(data, noise_level):
    """
    Add random noise to a dataset.
    
    Parameters:
    - data: DataFrame or ndarray
    - noise_level: Noise level (0 to 1)
    
    Returns:
    - DataFrame with noise added
    """
    if isinstance(data, pd.DataFrame):
        columns = data.columns
        data = data.values
    
    # Generate noise
    noise = np.random.normal(0, noise_level, data.shape)
    
    # Add noise
    noisy_data = data + noise
    
    if isinstance(columns, pd.Index):
        return pd.DataFrame(noisy_data, columns=columns)
    return noisy_data

# Function to visualize data in 2D with labels
def visualize_labeled_data(data, target_col, title):
    """
    Visualize data in 2D with labels.
    
    Parameters:
    - data: DataFrame
    - target_col: Column name or index of the target variable
    - title: Plot title
    """
    if isinstance(data, np.ndarray):
        data = pd.DataFrame(data, columns=[f'Var{i+1}' for i in range(data.shape[1])])
    
    if isinstance(target_col, int):
        target_col = data.columns[target_col]
    
    # Get feature columns (all columns except target)
    feature_cols = [col for col in data.columns if col != target_col]
    
    # Use first two features for visualization
    x_col, y_col = feature_cols[:2]
    
    # Create labels based on median of target variable
    median = data[target_col].median()
    labels = (data[target_col] > median).astype(int)
    
    # Create scatter plot
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(data[x_col], data[y_col], c=labels, cmap='coolwarm', alpha=0.8)
    plt.colorbar(scatter, label='Class')
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

# Function to evaluate classification performance
def evaluate_classification(data, target_col, noise_levels):
    """
    Evaluate classification performance at different noise levels.
    
    Parameters:
    - data: Original clean dataset
    - target_col: Column name or index of the target variable
    - noise_levels: List of noise levels to test
    
    Returns:
    - Dictionary with results
    """
    if isinstance(data, np.ndarray):
        data = pd.DataFrame(data, columns=[f'Var{i+1}' for i in range(data.shape[1])])
    
    if isinstance(target_col, int):
        target_col = data.columns[target_col]
    
    # Get feature columns (all columns except target)
    feature_cols = [col for col in data.columns if col != target_col]
    
    results = {
        'noise_level': [],
        'extended_correlation': [],
        'predictor_noise': [],
        'labeling_noise': [],
        'accuracy': []
    }
    
    for noise in noise_levels:
        # Add noise to the dataset
        noisy_data = add_noise(data, noise)
        
        # Calculate extended correlation
        rho = extended_correlation(noisy_data, use_mean=False)
        
        # Calculate noise components
        predictor_noise, labeling_noise = calculate_noise(
            noisy_data, 
            target_column=list(data.columns).index(target_col)
        )
        
        # Prepare data for classification
        X = noisy_data[feature_cols]
        
        # Create binary target based on median
        median = noisy_data[target_col].median()
        y = (noisy_data[target_col] > median).astype(int)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train logistic regression model
        model = LogisticRegression(random_state=42)
        model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Store results
        results['noise_level'].append(noise)
        results['extended_correlation'].append(rho)
        results['predictor_noise'].append(predictor_noise)
        results['labeling_noise'].append(labeling_noise)
        results['accuracy'].append(accuracy)
    
    return pd.DataFrame(results)

# Test 1: Compare FC and FU datasets
def test_fc_fu_comparison():
    """Test and compare FC and FU datasets"""
    n_samples = 1000
    n_vars = 3
    
    print("Testing FC and FU datasets comparison...")
    
    # Generate FC dataset
    fc_data = generate_fc_dataset(n_samples, n_vars)
    fc_corr = extended_correlation(fc_data)
    print(f"FC dataset extended correlation: {fc_corr:.4f}")
    
    # Generate FU dataset
    fu_data = generate_fu_dataset(n_samples, n_vars)
    fu_corr = extended_correlation(fu_data)
    print(f"FU dataset extended correlation: {fu_corr:.4f}")
    
    # Generate dataset with intermediate correlation
    mixed_data = add_noise(fc_data, 0.5)
    mixed_corr = extended_correlation(mixed_data)
    print(f"Mixed dataset extended correlation: {mixed_corr:.4f}")
    
    # Visualize correlation matrices
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    sns.heatmap(fc_data.corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1, ax=axes[0])
    axes[0].set_title(f'FC Dataset Correlation\nExtended ρ = {fc_corr:.4f}')
    
    sns.heatmap(mixed_data.corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1, ax=axes[1])
    axes[1].set_title(f'Mixed Dataset Correlation\nExtended ρ = {mixed_corr:.4f}')
    
    sns.heatmap(fu_data.corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1, ax=axes[2])
    axes[2].set_title(f'FU Dataset Correlation\nExtended ρ = {fu_corr:.4f}')
    
    plt.tight_layout()
    plt.show()

# Test 2: Effect of noise on maximal eigenvalue distribution
def test_maximal_eigenvalue_distribution():
    """Test how noise affects the distribution of maximal eigenvalues"""
    n_samples = 1000
    n_vars = 3
    noise_levels = [0, 0.1, 0.5, 1.0, 2.0, 5.0]
    
    print("\nTesting maximal eigenvalue distribution...")
    
    plt.figure(figsize=(12, 8))
    
    # Generate FC dataset
    fc_data = generate_fc_dataset(n_samples, n_vars)
    
    for i, noise in enumerate(noise_levels):
        # Add noise to the dataset
        noisy_data = add_noise(fc_data, noise)
        
        # Calculate maximal eigenvalues for growing samples
        max_eigenvalues = []
        for j in range(10, n_samples + 1, 10):  # Increment by 10 for efficiency
            subset = noisy_data.values[:j, :]
            # Calculate correlation matrix
            corr_matrix = np.corrcoef(subset, rowvar=False)
            # Calculate eigenvalues
            eigenvalues = np.linalg.eigvalsh(corr_matrix)
            max_eigenvalues.append(np.max(eigenvalues))
        
        # Plot histogram of maximal eigenvalues
        plt.subplot(2, 3, i+1)
        plt.hist(max_eigenvalues, bins=20, alpha=0.7)
        plt.axvline(np.mean(max_eigenvalues), color='red', linestyle='--', 
                    label=f'Mean: {np.mean(max_eigenvalues):.2f}')
        plt.axvline(1, color='green', linestyle='--', label='Min bound: 1')
        plt.axvline(n_vars, color='blue', linestyle='--', label=f'Max bound: {n_vars}')
        plt.title(f'Noise Level: {noise}')
        plt.xlabel('Maximal Eigenvalue')
        plt.ylabel('Frequency')
        plt.legend()
    
    plt.tight_layout()
    plt.show()

# Test 3: Noise measurement and feature selection
def test_noise_measurement():
    """Test noise measurement and its effect on classification"""
    n_samples = 1000
    n_vars = 3
    noise_levels = np.linspace(0, 2, 10)
    
    print("\nTesting noise measurement and feature selection...")
    
    # Generate base FC dataset
    fc_data = generate_fc_dataset(n_samples, n_vars)
    
    # Evaluate classification at different noise levels
    results = evaluate_classification(fc_data, 'Var1', noise_levels)
    
    # Plot results
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot extended correlation and accuracy
    axes[0].plot(results['noise_level'], results['extended_correlation'], 'b-', label='Extended Correlation')
    axes[0].plot(results['noise_level'], results['accuracy'], 'r-', label='Classification Accuracy')
    axes[0].set_xlabel('Noise Level')
    axes[0].set_ylabel('Value')
    axes[0].set_title('Extended Correlation vs. Classification Accuracy')
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot noise components
    axes[1].plot(results['noise_level'], results['predictor_noise'], 'g-', label='Predictor Noise')
    axes[1].plot(results['noise_level'], results['labeling_noise'], 'm-', label='Labeling Noise')
    axes[1].set_xlabel('Noise Level')
    axes[1].set_ylabel('Noise Value')
    axes[1].set_title('Components of Noise')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Print detailed results
    print("Detailed results:")
    print(results)
    
    # Visualize labeled data at different noise levels
    for noise in [0, 0.5, 1.0, 2.0]:
        noisy_data = add_noise(fc_data, noise)
        rho = extended_correlation(noisy_data, use_mean=False)
        visualize_labeled_data(
            noisy_data, 
            'Var1', 
            f'Labeled Data with Noise Level: {noise}\nExtended ρ = {rho:.4f}'
        )

# Test 4: Demonstrate labeling noise
def test_labeling_noise():
    """Test the effect of labeling noise"""
    n_samples = 1000
    n_vars = 3
    
    print("\nTesting labeling noise...")
    
    # Generate base FC dataset
    fc_data = generate_fc_dataset(n_samples, n_vars)
    
    # Create datasets with different labeling noise levels
    datasets = []
    
    # Original dataset (no noise)
    datasets.append(('Original', fc_data.copy()))
    
    # Add noise only to the target variable
    for noise in [0.5, 1.0, 2.0]:
        noisy_data = fc_data.copy()
        noisy_data['Var1'] = fc_data['Var1'] + np.random.normal(0, noise, n_samples)
        datasets.append((f'Target Noise: {noise}', noisy_data))
    
    # Add noise only to the predictor variables
    for noise in [0.5, 1.0, 2.0]:
        noisy_data = fc_data.copy()
        for col in ['Var2', 'Var3']:
            noisy_data[col] = fc_data[col] + np.random.normal(0, noise, n_samples)
        datasets.append((f'Predictor Noise: {noise}', noisy_data))
    
    # Calculate noise components for each dataset
    results = []
    
    for name, data in datasets:
        # Calculate extended correlation
        rho_full = extended_correlation(data, use_mean=False)
        
        # Calculate noise components
        predictor_noise, labeling_noise = calculate_noise(data, target_column=0)
        
        results.append({
            'Dataset': name,
            'Extended Correlation': rho_full,
            'Predictor Noise': predictor_noise,
            'Labeling Noise': labeling_noise
        })
        
        # Visualize data
        visualize_labeled_data(
            data, 
            'Var1', 
            f'{name}\nρ = {rho_full:.4f}, Predictor Noise = {predictor_noise:.4f}, Labeling Noise = {labeling_noise:.4f}'
        )
    
    # Print results
    print("\nNoise Component Results:")
    for result in results:
        print(f"Dataset: {result['Dataset']}")
        print(f"  Extended Correlation: {result['Extended Correlation']:.4f}")
        print(f"  Predictor Noise: {result['Predictor Noise']:.4f}")
        print(f"  Labeling Noise: {result['Labeling Noise']:.4f}")
        print()

# Run all tests
if __name__ == "__main__":
    test_fc_fu_comparison()
    test_maximal_eigenvalue_distribution()
    test_noise_measurement()
    test_labeling_noise()