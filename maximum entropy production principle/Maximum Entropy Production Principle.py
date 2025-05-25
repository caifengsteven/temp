import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# Function to simulate stock returns with various characteristics
def simulate_stock_returns(n_days=2000, n_stocks=100, correlation=0.5, 
                           volatility_clustering=True, mean_reversion=True):
    """
    Simulate stock returns with realistic properties:
    - n_days: number of days to simulate
    - n_stocks: number of stocks to simulate
    - correlation: correlation between stock returns
    - volatility_clustering: whether to include volatility clustering
    - mean_reversion: whether to include mean reversion
    
    Returns:
    - DataFrame of simulated returns
    """
    # Common market factor
    market_factor = np.random.normal(0, 1, n_days)
    
    # Individual stock returns
    stock_returns = np.zeros((n_days, n_stocks))
    
    for i in range(n_stocks):
        # Idiosyncratic return component
        idiosyncratic = np.random.normal(0, np.sqrt(1-correlation**2), n_days)
        
        # Basic return: market + idiosyncratic
        returns = correlation * market_factor + idiosyncratic
        
        # Add mean reversion if specified
        if mean_reversion:
            for t in range(1, n_days):
                returns[t] -= 0.05 * returns[t-1]  # Mean reversion coefficient
        
        # Add volatility clustering if specified
        if volatility_clustering:
            volatility = np.ones(n_days)
            for t in range(1, n_days):
                volatility[t] = 0.9 * volatility[t-1] + 0.1 * returns[t-1]**2
            volatility = np.sqrt(np.abs(volatility))
            returns = returns * volatility
        
        # Scale returns to realistic values (daily returns are typically < 5%)
        returns = returns * 0.01
        
        stock_returns[:, i] = returns
    
    # Create DataFrame with date index
    dates = pd.date_range(start='2020-01-01', periods=n_days, freq='B')
    df = pd.DataFrame(stock_returns, index=dates)
    df.columns = [f'Stock_{i+1}' for i in range(n_stocks)]
    
    return df

# Function to discretize returns into states (quartiles)
def discretize_returns(returns, n_states=4):
    """
    Discretize returns into n_states states (quartiles by default)
    
    Parameters:
    - returns: pandas Series of returns
    - n_states: number of states to discretize into
    
    Returns:
    - Series of discretized returns (1 to n_states)
    """
    # Create equal-sized bins
    quantiles = [i/n_states for i in range(1, n_states)]
    bins = returns.quantile(quantiles).tolist()
    bins = [-np.inf] + bins + [np.inf]
    
    # Discretize returns
    discretized = pd.cut(returns, bins=bins, labels=False) + 1
    
    return discretized

# Lempel-Ziv complexity estimator for Shannon's entropy rate
def lempel_ziv_entropy(sequence):
    """
    Estimate Shannon's entropy rate using Lempel-Ziv algorithm
    
    Parameters:
    - sequence: sequence of discrete symbols
    
    Returns:
    - Estimated entropy rate
    """
    sequence = sequence.astype(str)
    n = len(sequence)
    
    # Compute LZ complexity
    i = 0
    complexity = 0
    
    while i < n:
        # Find the longest substring starting from i
        # that has already been seen
        longest_match = 0
        for j in range(1, min(n-i, n//2)):  # Limit search to half the sequence
            substring = ''.join(sequence[i:i+j])
            prefix = ''.join(sequence[:i])
            
            if substring in prefix:
                longest_match = j
            else:
                break
        
        complexity += 1
        i += max(1, longest_match)
    
    # Estimate entropy rate
    if complexity == 0:
        return 0
    
    return (n * np.log2(n)) / complexity

# Function to test Maximum Entropy Production Principle
def test_mepp(returns, window_size=50, n_states=4):
    """
    Test the Maximum Entropy Production Principle on stock returns
    
    Parameters:
    - returns: pandas Series of returns
    - window_size: size of the window for entropy calculation
    - n_states: number of states to discretize into
    
    Returns:
    - Fraction of correct predictions
    - Entropy rate of the full series
    """
    # Discretize returns
    discretized = discretize_returns(returns, n_states)
    
    # Calculate entropy rate for the full series
    full_entropy = lempel_ziv_entropy(discretized.values)
    
    # Test MEPP
    n_correct = 0
    n_tests = 0
    
    for t in range(window_size, len(discretized)-1):
        # Get history up to time t
        history = discretized.iloc[t-window_size:t].values
        
        # Try each possible next state and calculate entropy
        entropies = []
        for state in range(1, n_states+1):
            test_sequence = np.append(history, state)
            entropy = lempel_ziv_entropy(test_sequence)
            entropies.append(entropy)
        
        # Predict state that maximizes entropy
        predicted_state = np.argmax(entropies) + 1
        
        # Check if prediction is correct
        actual_state = discretized.iloc[t]
        if predicted_state == actual_state:
            n_correct += 1
        
        n_tests += 1
    
    return n_correct / n_tests, full_entropy

# Function to run MEPP tests for multiple window sizes
def run_mepp_tests(returns, window_sizes=range(20, 101, 10), n_states=4):
    """
    Run MEPP tests for multiple window sizes
    
    Parameters:
    - returns: pandas Series of returns
    - window_sizes: list of window sizes to test
    - n_states: number of states to discretize into
    
    Returns:
    - DataFrame with results for each window size
    """
    results = []
    
    for window in window_sizes:
        accuracy, entropy = test_mepp(returns, window, n_states)
        results.append({
            'window_size': window,
            'accuracy': accuracy,
            'entropy_rate': entropy
        })
    
    return pd.DataFrame(results)

# Function to analyze multiple stocks
def analyze_stocks(returns_df, window_sizes=range(20, 101, 10), n_states=4):
    """
    Analyze multiple stocks using MEPP
    
    Parameters:
    - returns_df: DataFrame with stock returns (columns = stocks)
    - window_sizes: list of window sizes to test
    - n_states: number of states to discretize into
    
    Returns:
    - DataFrame with results for each stock and window size
    """
    all_results = []
    
    for stock in tqdm(returns_df.columns):
        stock_returns = returns_df[stock]
        
        # Skip if there are NaN values
        if stock_returns.isna().any():
            continue
        
        # Run MEPP tests
        results = run_mepp_tests(stock_returns, window_sizes, n_states)
        results['stock'] = stock
        
        all_results.append(results)
    
    return pd.concat(all_results, ignore_index=True)

# Function to plot results (compatible with older seaborn versions)
def plot_mepp_results(results):
    """
    Plot MEPP results
    
    Parameters:
    - results: DataFrame with MEPP test results
    """
    # 1. Plot accuracy vs window size
    plt.figure(figsize=(10, 6))
    # Group by window size and calculate mean and std
    window_stats = results.groupby('window_size')['accuracy'].agg(['mean', 'std']).reset_index()
    
    plt.plot(window_stats['window_size'], window_stats['mean'], 'b-', label='Mean Accuracy')
    plt.fill_between(
        window_stats['window_size'],
        window_stats['mean'] - window_stats['std'],
        window_stats['mean'] + window_stats['std'],
        alpha=0.2,
        color='b'
    )
    plt.axhline(y=0.25, color='r', linestyle='--', label='Random (0.25)')  # For 4 states
    plt.title('MEPP Accuracy vs Window Size')
    plt.xlabel('Window Size (μ)')
    plt.ylabel('Accuracy (Ψ)')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # 2. Histogram of accuracies
    plt.figure(figsize=(10, 6))
    avg_accuracies = results.groupby('stock')['accuracy'].mean()
    
    plt.hist(avg_accuracies, bins=20, alpha=0.7, density=True)
    # Add KDE plot
    x = np.linspace(avg_accuracies.min(), avg_accuracies.max(), 100)
    kde = stats.gaussian_kde(avg_accuracies)
    plt.plot(x, kde(x), 'r-')
    
    plt.axvline(x=0.25, color='r', linestyle='--', label='Random (0.25)')
    plt.title('Distribution of MEPP Accuracies')
    plt.xlabel('Accuracy (Ψ)')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # 3. Scatterplot of accuracy vs entropy rate
    plt.figure(figsize=(10, 6))
    avg_results = results.groupby('stock').mean().reset_index()
    
    plt.scatter(avg_results['entropy_rate'], avg_results['accuracy'])
    plt.title('MEPP Accuracy vs Entropy Rate')
    plt.xlabel('Lempel-Ziv Entropy Rate')
    plt.ylabel('Accuracy (Ψ)')
    plt.axhline(y=0.25, color='r', linestyle='--', label='Random (0.25)')
    plt.legend()
    plt.grid(alpha=0.3)
    
    # Add regression line
    x = avg_results['entropy_rate']
    y = avg_results['accuracy']
    m, b = np.polyfit(x, y, 1)
    plt.plot(x, m*x + b, color='k', linestyle='-')
    
    # Add correlation coefficient
    corr = np.corrcoef(x, y)[0, 1]
    plt.annotate(f'Correlation: {corr:.4f}', xy=(0.05, 0.95), xycoords='axes fraction')
    
    plt.tight_layout()
    plt.show()
    
    # 4. Plot distribution on log scale
    plt.figure(figsize=(10, 6))
    
    # Sort the accuracies and calculate log-scaled frequency
    sorted_acc = np.sort(avg_accuracies)
    log_freq = stats.norm.sf(np.linspace(0.01, 0.99, len(sorted_acc)))
    
    plt.semilogy(sorted_acc, log_freq)
    plt.axvline(x=0.25, color='r', linestyle='--', label='Random (0.25)')
    plt.title('Log Distribution of MEPP Accuracies')
    plt.xlabel('Accuracy (Ψ)')
    plt.ylabel('Frequency (log scale)')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

# Main function to demonstrate the MEPP on simulated data
def main():
    print("Simulating stock returns...")
    # Simulate several market scenarios
    scenarios = {
        'Base Case': {
            'correlation': 0.5,
            'volatility_clustering': True,
            'mean_reversion': True
        },
        'High Correlation': {
            'correlation': 0.8, 
            'volatility_clustering': True,
            'mean_reversion': True
        },
        'No Volatility Clustering': {
            'correlation': 0.5,
            'volatility_clustering': False,
            'mean_reversion': True
        },
        'No Mean Reversion': {
            'correlation': 0.5,
            'volatility_clustering': True,
            'mean_reversion': False
        }
    }
    
    all_scenario_results = {}
    
    for name, params in scenarios.items():
        print(f"\nAnalyzing scenario: {name}")
        
        # Simulate returns
        returns_df = simulate_stock_returns(
            n_days=1000,  # Reduced for speed
            n_stocks=30,  # Reduced for speed
            correlation=params['correlation'],
            volatility_clustering=params['volatility_clustering'],
            mean_reversion=params['mean_reversion']
        )
        
        # Analyze all stocks
        print(f"Testing MEPP on {len(returns_df.columns)} stocks...")
        results = analyze_stocks(
            returns_df,
            window_sizes=range(20, 101, 20),  # Fewer window sizes for speed
            n_states=4
        )
        
        all_scenario_results[name] = results
        
        # Plot individual scenario results
        print(f"Plotting results for {name}...")
        plot_mepp_results(results)
        
        # Print summary statistics
        avg_accuracies = results.groupby('stock')['accuracy'].mean()
        print(f"\nSummary for {name}:")
        print(f"Mean accuracy: {avg_accuracies.mean():.4f}")
        print(f"Percent above random: {(avg_accuracies > 0.25).mean() * 100:.1f}%")
        
        # Correlation with entropy rate
        avg_results = results.groupby('stock').mean().reset_index()
        corr = np.corrcoef(avg_results['entropy_rate'], avg_results['accuracy'])[0, 1]
        print(f"Correlation with entropy rate: {corr:.4f}")
    
    # Compare scenarios
    print("\nComparing all scenarios...")
    plt.figure(figsize=(12, 8))
    
    for name, results in all_scenario_results.items():
        # Get average accuracy by window size
        avg_by_window = results.groupby('window_size')['accuracy'].mean()
        plt.plot(avg_by_window.index, avg_by_window.values, label=name)
    
    plt.axhline(y=0.25, color='r', linestyle='--', label='Random (0.25)')
    plt.title('MEPP Accuracy Across Different Market Scenarios')
    plt.xlabel('Window Size (μ)')
    plt.ylabel('Accuracy (Ψ)')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Compare accuracy distributions
    plt.figure(figsize=(12, 8))
    
    # Create KDE for each scenario
    for name, results in all_scenario_results.items():
        avg_accuracies = results.groupby('stock')['accuracy'].mean()
        # Use matplotlib-compatible approach instead of seaborn's kdeplot
        kde = stats.gaussian_kde(avg_accuracies)
        x = np.linspace(0.2, 0.4, 100)
        plt.plot(x, kde(x), label=name)
    
    plt.axvline(x=0.25, color='r', linestyle='--', label='Random (0.25)')
    plt.title('Distribution of MEPP Accuracies Across Scenarios')
    plt.xlabel('Accuracy (Ψ)')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

# Run the main function
if __name__ == "__main__":
    main()