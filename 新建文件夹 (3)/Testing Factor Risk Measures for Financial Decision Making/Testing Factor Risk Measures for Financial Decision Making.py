import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.optimize import minimize
from tqdm import tqdm

# Set random seed for reproducibility
np.random.seed(42)

# Set plotting style
plt.style.use('ggplot')
sns.set_style("whitegrid")

###########################################
# 1. UTILITY FUNCTIONS
###########################################

def var(X, alpha=0.95):
    """
    Calculate Value-at-Risk at level alpha
    
    Parameters:
    -----------
    X : array-like
        Random variable (losses)
    alpha : float
        Confidence level (default: 0.95)
        
    Returns:
    --------
    float
        Value-at-Risk at level alpha
    """
    return np.quantile(X, alpha)

def es(X, alpha=0.95):
    """
    Calculate Expected Shortfall at level alpha
    
    Parameters:
    -----------
    X : array-like
        Random variable (losses)
    alpha : float
        Confidence level (default: 0.95)
        
    Returns:
    --------
    float
        Expected Shortfall at level alpha
    """
    threshold = var(X, alpha)
    return np.mean(X[X >= threshold])

def conditional_var(X, W, alpha=0.95):
    """
    Calculate conditional VaR (VaR of X given W)
    
    Parameters:
    -----------
    X : array-like
        Random variable (losses)
    W : array-like
        Factor variable
    alpha : float
        Confidence level (default: 0.95)
        
    Returns:
    --------
    array-like
        Conditional VaR for each unique value of W
    """
    unique_w = np.unique(W)
    cond_vars = []
    
    for w in unique_w:
        X_w = X[W == w]
        if len(X_w) > 0:
            cond_vars.append(var(X_w, alpha))
        else:
            cond_vars.append(np.nan)
            
    return np.array(cond_vars), unique_w

def conditional_es(X, W, alpha=0.95):
    """
    Calculate conditional ES (ES of X given W)
    
    Parameters:
    -----------
    X : array-like
        Random variable (losses)
    W : array-like
        Factor variable
    alpha : float
        Confidence level (default: 0.95)
        
    Returns:
    --------
    array-like
        Conditional ES for each unique value of W
    """
    unique_w = np.unique(W)
    cond_es = []
    
    for w in unique_w:
        X_w = X[W == w]
        if len(X_w) > 0:
            cond_es.append(es(X_w, alpha))
        else:
            cond_es.append(np.nan)
            
    return np.array(cond_es), unique_w

def covar(X, W, alpha=0.95, beta=0.95):
    """
    Calculate CoVaR: VaR of X conditional on W exceeding its VaR at level alpha
    
    Parameters:
    -----------
    X : array-like
        Random variable (losses)
    W : array-like
        Factor variable
    alpha : float
        Confidence level for W (default: 0.95)
    beta : float
        Confidence level for X (default: 0.95)
        
    Returns:
    --------
    float
        CoVaR at levels alpha and beta
    """
    w_var = var(W, alpha)
    X_cond = X[W >= w_var]
    
    if len(X_cond) > 0:
        return var(X_cond, beta)
    else:
        return np.nan

def coes(X, W, alpha=0.95, beta=0.95):
    """
    Calculate CoES: ES of X conditional on W exceeding its VaR at level alpha
    
    Parameters:
    -----------
    X : array-like
        Random variable (losses)
    W : array-like
        Factor variable
    alpha : float
        Confidence level for W (default: 0.95)
    beta : float
        Confidence level for X (default: 0.95)
        
    Returns:
    --------
    float
        CoES at levels alpha and beta
    """
    w_var = var(W, alpha)
    X_cond = X[W >= w_var]
    
    if len(X_cond) > 0:
        return es(X_cond, beta)
    else:
        return np.nan

def mes(X, W, alpha=0.95):
    """
    Calculate Marginal Expected Shortfall (MES): E[X|W >= VaR_alpha(W)]
    
    Parameters:
    -----------
    X : array-like
        Random variable (losses)
    W : array-like
        Factor variable
    alpha : float
        Confidence level (default: 0.95)
        
    Returns:
    --------
    float
        MES at level alpha
    """
    w_var = var(W, alpha)
    X_cond = X[W >= w_var]
    
    if len(X_cond) > 0:
        return np.mean(X_cond)
    else:
        return np.nan

def var_of_conditional_var(X, W, p=0.95, q=0.95):
    """
    Calculate VaR_q(VaR_p(X|W))
    
    Parameters:
    -----------
    X : array-like
        Random variable (losses)
    W : array-like
        Factor variable
    p : float
        Confidence level for conditional VaR (default: 0.95)
    q : float
        Confidence level for outer VaR (default: 0.95)
        
    Returns:
    --------
    float
        VaR_q(VaR_p(X|W))
    """
    cond_vars, _ = conditional_var(X, W, p)
    return var(cond_vars, q)

def es_of_conditional_es(X, W, p=0.95, q=0.95):
    """
    Calculate ES_q(ES_p(X|W))
    
    Parameters:
    -----------
    X : array-like
        Random variable (losses)
    W : array-like
        Factor variable
    p : float
        Confidence level for conditional ES (default: 0.95)
    q : float
        Confidence level for outer ES (default: 0.95)
        
    Returns:
    --------
    float
        ES_q(ES_p(X|W))
    """
    cond_es_values, _ = conditional_es(X, W, p)
    return es(cond_es_values, q)

def expected_conditional_var(X, W, alpha=0.95):
    """
    Calculate E[VaR_alpha(X|W)]
    
    Parameters:
    -----------
    X : array-like
        Random variable (losses)
    W : array-like
        Factor variable
    alpha : float
        Confidence level (default: 0.95)
        
    Returns:
    --------
    float
        Expected conditional VaR
    """
    cond_vars, unique_w = conditional_var(X, W, alpha)
    
    # Calculate probability of each unique value of W
    w_counts = np.array([np.sum(W == w) for w in unique_w])
    w_probs = w_counts / np.sum(w_counts)
    
    # Calculate weighted average
    return np.nansum(cond_vars * w_probs)

def expected_conditional_es(X, W, alpha=0.95):
    """
    Calculate E[ES_alpha(X|W)]
    
    Parameters:
    -----------
    X : array-like
        Random variable (losses)
    W : array-like
        Factor variable
    alpha : float
        Confidence level (default: 0.95)
        
    Returns:
    --------
    float
        Expected conditional ES
    """
    cond_es_values, unique_w = conditional_es(X, W, alpha)
    
    # Calculate probability of each unique value of W
    w_counts = np.array([np.sum(W == w) for w in unique_w])
    w_probs = w_counts / np.sum(w_counts)
    
    # Calculate weighted average
    return np.nansum(cond_es_values * w_probs)

def esssup_conditional_var(X, W, alpha=0.95):
    """
    Calculate ess sup VaR_alpha(X|W)
    
    Parameters:
    -----------
    X : array-like
        Random variable (losses)
    W : array-like
        Factor variable
    alpha : float
        Confidence level (default: 0.95)
        
    Returns:
    --------
    float
        Essential supremum of conditional VaR
    """
    cond_vars, _ = conditional_var(X, W, alpha)
    return np.nanmax(cond_vars)

def esssup_conditional_es(X, W, alpha=0.95):
    """
    Calculate ess sup ES_alpha(X|W)
    
    Parameters:
    -----------
    X : array-like
        Random variable (losses)
    W : array-like
        Factor variable
    alpha : float
        Confidence level (default: 0.95)
        
    Returns:
    --------
    float
        Essential supremum of conditional ES
    """
    cond_es_values, _ = conditional_es(X, W, alpha)
    return np.nanmax(cond_es_values)

###########################################
# 2. DATA SIMULATION
###########################################

def simulate_financial_data(n_samples=10000, n_factors=3, corr_strength=0.5, heavy_tail=False):
    """
    Simulate financial return data with factor structure
    
    Parameters:
    -----------
    n_samples : int
        Number of samples
    n_factors : int
        Number of factors
    corr_strength : float
        Correlation strength between factors and returns
    heavy_tail : bool
        Whether to use heavy-tailed distributions
        
    Returns:
    --------
    tuple
        (X, W) where X is the return/loss variable and W are the factors
    """
    # Generate factor data
    if heavy_tail:
        # Use t-distribution with 5 degrees of freedom for heavy tails
        factors = stats.t.rvs(df=5, size=(n_samples, n_factors))
    else:
        # Use normal distribution
        factors = np.random.normal(0, 1, (n_samples, n_factors))
    
    # Generate coefficients (beta) for the factors
    betas = np.random.uniform(-1, 1, n_factors)
    
    # Generate idiosyncratic risk
    if heavy_tail:
        epsilon = stats.t.rvs(df=5, size=n_samples)
    else:
        epsilon = np.random.normal(0, 1, n_samples)
    
    # Generate returns
    X = corr_strength * (factors @ betas) + np.sqrt(1 - corr_strength**2) * epsilon
    
    # Convert to losses (negative returns)
    X = -X
    
    # Discretize factors for simplicity in conditional calculations
    W = np.zeros(n_samples)
    for i in range(n_factors):
        # Use the first factor as the main factor
        if i == 0:
            W = factors[:, i]
        else:
            # Add some influence from other factors
            W += 0.3 * factors[:, i]
    
    # Discretize W into 10 buckets for easier calculation of conditional measures
    W_discrete = pd.qcut(W, 10, labels=False)
    
    return X, W_discrete

###########################################
# 3. RISK MEASURE COMPARISON
###########################################

def compare_risk_measures(X, W, alpha=0.95, beta=0.95):
    """
    Compare various risk measures
    
    Parameters:
    -----------
    X : array-like
        Random variable (losses)
    W : array-like
        Factor variable
    alpha : float
        First confidence level (default: 0.95)
    beta : float
        Second confidence level (default: 0.95)
        
    Returns:
    --------
    dict
        Dictionary of risk measure values
    """
    results = {
        "VaR": var(X, alpha),
        "ES": es(X, alpha),
        "CoVaR": covar(X, W, alpha, beta),
        "CoES": coes(X, W, alpha, beta),
        "MES": mes(X, W, alpha),
        "VaR of Conditional VaR": var_of_conditional_var(X, W, alpha, beta),
        "ES of Conditional ES": es_of_conditional_es(X, W, alpha, beta),
        "Expected Conditional VaR": expected_conditional_var(X, W, alpha),
        "Expected Conditional ES": expected_conditional_es(X, W, alpha),
        "Esssup Conditional VaR": esssup_conditional_var(X, W, alpha),
        "Esssup Conditional ES": esssup_conditional_es(X, W, alpha)
    }
    
    return results

def sensitivity_analysis(n_simulations=100):
    """
    Perform sensitivity analysis on risk measures
    
    Parameters:
    -----------
    n_simulations : int
        Number of simulations
        
    Returns:
    --------
    DataFrame
        Results of sensitivity analysis
    """
    results = []
    
    for corr in [0.2, 0.5, 0.8]:
        for heavy_tail in [False, True]:
            for alpha in [0.9, 0.95, 0.99]:
                for beta in [0.9, 0.95, 0.99]:
                    for _ in tqdm(range(n_simulations), desc=f"Corr={corr}, Heavy={heavy_tail}, α={alpha}, β={beta}"):
                        X, W = simulate_financial_data(n_samples=5000, corr_strength=corr, heavy_tail=heavy_tail)
                        risk_measures = compare_risk_measures(X, W, alpha, beta)
                        
                        results.append({
                            "Correlation": corr,
                            "Heavy_tail": heavy_tail,
                            "Alpha": alpha,
                            "Beta": beta,
                            **risk_measures
                        })
    
    return pd.DataFrame(results)

###########################################
# 4. RISK SHARING APPLICATION
###########################################

def optimal_risk_sharing(X, n_agents=2, risk_measure_type="VaRp", alpha=0.95):
    """
    Find the optimal risk sharing allocation
    
    Parameters:
    -----------
    X : array-like
        Risk to be shared
    n_agents : int
        Number of agents
    risk_measure_type : str
        Type of risk measure to use
    alpha : float
        Confidence level
        
    Returns:
    --------
    tuple
        (X1, X2, ..., Xn) optimal allocation
    """
    n = len(X)
    
    # Initialize allocation
    allocation = np.ones((n_agents, n)) * X / n_agents
    
    # Define objective function based on Corollary 4 in the paper
    def objective(params):
        # Convert params to allocation
        cuts = np.concatenate(([0], np.sort(params), [1]))
        new_allocation = np.zeros((n_agents, n))
        
        # Sort X
        sorted_indices = np.argsort(X)
        sorted_X = X[sorted_indices]
        
        for i in range(n_agents):
            start_pct = cuts[i]
            end_pct = cuts[i+1]
            start_idx = int(start_pct * n)
            end_idx = int(end_pct * n)
            
            if end_idx > start_idx:
                agent_allocation = np.zeros(n)
                agent_allocation[sorted_indices[start_idx:end_idx]] = sorted_X[start_idx:end_idx]
                new_allocation[i] = agent_allocation
        
        # Check if allocation is valid
        total_allocation = np.sum(new_allocation, axis=0)
        if not np.allclose(total_allocation, X, rtol=1e-5, atol=1e-5):
            return 1e10
        
        # Calculate risk measure for each agent
        if risk_measure_type == "VaRp":
            risk_measures = [var(new_allocation[i], alpha) for i in range(n_agents)]
        elif risk_measure_type == "ESp":
            risk_measures = [es(new_allocation[i], alpha) for i in range(n_agents)]
        else:
            raise ValueError(f"Unknown risk measure type: {risk_measure_type}")
        
        # Return sum of risk measures
        return np.sum(risk_measures)
    
    # Optimize
    if n_agents > 1:
        initial_params = np.linspace(0, 1, n_agents+1)[1:-1]
        result = minimize(objective, initial_params, method='Nelder-Mead', 
                          options={'maxiter': 1000, 'disp': False})
        
        # Extract optimal allocation
        cuts = np.concatenate(([0], np.sort(result.x), [1]))
        optimal_allocation = np.zeros((n_agents, n))
        
        # Sort X
        sorted_indices = np.argsort(X)
        sorted_X = X[sorted_indices]
        
        for i in range(n_agents):
            start_pct = cuts[i]
            end_pct = cuts[i+1]
            start_idx = int(start_pct * n)
            end_idx = int(end_pct * n)
            
            if end_idx > start_idx:
                agent_allocation = np.zeros(n)
                agent_allocation[sorted_indices[start_idx:end_idx]] = sorted_X[start_idx:end_idx]
                optimal_allocation[i] = agent_allocation
        
        return optimal_allocation
    else:
        return np.array([X])

###########################################
# 5. EXPERIMENT 1: BASIC COMPARISON
###########################################

def experiment_basic_comparison():
    """
    Compare different risk measures on a simple simulated dataset
    """
    print("Experiment 1: Basic Comparison of Risk Measures")
    print("=" * 80)
    
    # Simulate data
    X, W = simulate_financial_data(n_samples=10000, corr_strength=0.5, heavy_tail=False)
    
    # Calculate risk measures
    results = compare_risk_measures(X, W, alpha=0.95, beta=0.95)
    
    # Print results
    for measure, value in results.items():
        print(f"{measure}: {value:.4f}")
    
    # Plot distribution of losses and conditional losses
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot 1: Overall distribution
    sns.histplot(X, bins=50, kde=True, ax=axes[0])
    axes[0].axvline(results["VaR"], color='r', linestyle='--', label=f'VaR_0.95 = {results["VaR"]:.4f}')
    axes[0].axvline(results["ES"], color='g', linestyle='--', label=f'ES_0.95 = {results["ES"]:.4f}')
    axes[0].set_title("Distribution of Losses")
    axes[0].set_xlabel("Loss")
    axes[0].set_ylabel("Frequency")
    axes[0].legend()
    
    # Plot 2: Conditional VaR and VaR of Conditional VaR
    cond_vars, unique_w = conditional_var(X, W, 0.95)
    
    axes[1].plot(unique_w, cond_vars, 'o-', label='Conditional VaR')
    axes[1].axhline(results["VaR of Conditional VaR"], color='r', linestyle='--', 
                   label=f'VaR_0.95(VaR_0.95(X|W)) = {results["VaR of Conditional VaR"]:.4f}')
    axes[1].axhline(results["Expected Conditional VaR"], color='g', linestyle='--', 
                   label=f'E[VaR_0.95(X|W)] = {results["Expected Conditional VaR"]:.4f}')
    axes[1].axhline(results["Esssup Conditional VaR"], color='b', linestyle='--', 
                   label=f'ess sup VaR_0.95(X|W) = {results["Esssup Conditional VaR"]:.4f}')
    axes[1].set_title("Conditional VaR vs. Factor Value")
    axes[1].set_xlabel("Factor Value (W)")
    axes[1].set_ylabel("Conditional VaR")
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig("basic_comparison.png")
    plt.show()
    
    return results

###########################################
# 6. EXPERIMENT 2: SENSITIVITY ANALYSIS
###########################################

def experiment_sensitivity_analysis():
    """
    Analyze sensitivity of risk measures to different parameters
    """
    print("Experiment 2: Sensitivity Analysis")
    print("=" * 80)
    
    # Run sensitivity analysis
    print("Running sensitivity analysis (this may take a while)...")
    results = sensitivity_analysis(n_simulations=20)
    
    # Save results
    results.to_csv("sensitivity_analysis.csv", index=False)
    print(f"Saved results to sensitivity_analysis.csv")
    
    # Calculate average risk measure values for different parameter combinations
    summary = results.groupby(['Correlation', 'Heavy_tail', 'Alpha', 'Beta']).mean().reset_index()
    
    # Plot heatmaps of key comparisons
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Filter for specific settings
    heavy_tail_data = summary[summary['Heavy_tail'] == True]
    normal_data = summary[summary['Heavy_tail'] == False]
    high_corr_data = summary[summary['Correlation'] == 0.8]
    
    # Prepare data for heatmaps
    pivot1 = pd.pivot_table(high_corr_data, values='VaR of Conditional VaR', 
                            index='Alpha', columns='Beta')
    pivot2 = pd.pivot_table(high_corr_data, values='Expected Conditional VaR', 
                            index='Alpha', columns='Beta')
    pivot3 = pd.pivot_table(normal_data[normal_data['Beta'] == 0.95], 
                            values=['VaR', 'VaR of Conditional VaR'], 
                            index='Alpha', columns='Correlation')
    pivot4 = pd.pivot_table(heavy_tail_data[heavy_tail_data['Beta'] == 0.95], 
                            values=['VaR', 'VaR of Conditional VaR'], 
                            index='Alpha', columns='Correlation')
    
    # Plot heatmaps
    sns.heatmap(pivot1, annot=True, fmt=".3f", cmap="YlGnBu", ax=axes[0, 0])
    axes[0, 0].set_title("VaR of Conditional VaR (High Correlation)")
    
    sns.heatmap(pivot2, annot=True, fmt=".3f", cmap="YlGnBu", ax=axes[0, 1])
    axes[0, 1].set_title("Expected Conditional VaR (High Correlation)")
    
    # Reshape for comparison plots
    pivot3_reshaped = pivot3.stack().reset_index()
    pivot3_reshaped.columns = ['Alpha', 'Correlation', 'Measure', 'Value']
    
    pivot4_reshaped = pivot4.stack().reset_index()
    pivot4_reshaped.columns = ['Alpha', 'Correlation', 'Measure', 'Value']
    
    # Plot comparison plots
    sns.barplot(data=pivot3_reshaped, x='Alpha', y='Value', hue='Measure', ax=axes[1, 0])
    axes[1, 0].set_title("VaR vs VaR of Conditional VaR (Normal Distribution)")
    axes[1, 0].legend(title='Measure')
    
    sns.barplot(data=pivot4_reshaped, x='Alpha', y='Value', hue='Measure', ax=axes[1, 1])
    axes[1, 1].set_title("VaR vs VaR of Conditional VaR (Heavy Tailed Distribution)")
    axes[1, 1].legend(title='Measure')
    
    plt.tight_layout()
    plt.savefig("sensitivity_analysis.png")
    plt.show()
    
    return results

###########################################
# 7. EXPERIMENT 3: RISK SHARING
###########################################

def experiment_risk_sharing():
    """
    Demonstrate optimal risk sharing with factor risk measures
    """
    print("Experiment 3: Risk Sharing")
    print("=" * 80)
    
    # Simulate data
    X, W = simulate_financial_data(n_samples=1000, corr_strength=0.7, heavy_tail=True)
    
    # Calculate optimal risk sharing
    print("Calculating optimal risk sharing...")
    allocations_var = optimal_risk_sharing(X, n_agents=2, risk_measure_type="VaRp", alpha=0.95)
    allocations_es = optimal_risk_sharing(X, n_agents=2, risk_measure_type="ESp", alpha=0.95)
    
    # Calculate risk measures for original and shared risk
    var_original = var(X, 0.95)
    es_original = es(X, 0.95)
    
    var_shared = [var(allocations_var[i], 0.95) for i in range(2)]
    es_shared = [es(allocations_es[i], 0.95) for i in range(2)]
    
    print(f"Original VaR: {var_original:.4f}")
    print(f"Shared VaR (Agent 1): {var_shared[0]:.4f}")
    print(f"Shared VaR (Agent 2): {var_shared[1]:.4f}")
    print(f"Sum of Shared VaR: {sum(var_shared):.4f}")
    print()
    
    print(f"Original ES: {es_original:.4f}")
    print(f"Shared ES (Agent 1): {es_shared[0]:.4f}")
    print(f"Shared ES (Agent 2): {es_shared[1]:.4f}")
    print(f"Sum of Shared ES: {sum(es_shared):.4f}")
    
    # Plot risk sharing
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Sort X for better visualization
    sorted_indices = np.argsort(X)
    sorted_X = X[sorted_indices]
    
    # Plot VaR-based allocation
    axes[0, 0].plot(sorted_X, label='Original Risk')
    axes[0, 0].plot(allocations_var[0][sorted_indices], label='Agent 1')
    axes[0, 0].plot(allocations_var[1][sorted_indices], label='Agent 2')
    axes[0, 0].set_title("VaR-based Risk Sharing Allocation")
    axes[0, 0].set_xlabel("Ordered Sample")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].legend()
    
    # Plot ES-based allocation
    axes[0, 1].plot(sorted_X, label='Original Risk')
    axes[0, 1].plot(allocations_es[0][sorted_indices], label='Agent 1')
    axes[0, 1].plot(allocations_es[1][sorted_indices], label='Agent 2')
    axes[0, 1].set_title("ES-based Risk Sharing Allocation")
    axes[0, 1].set_xlabel("Ordered Sample")
    axes[0, 1].set_ylabel("Loss")
    axes[0, 1].legend()
    
    # Plot distribution of VaR-based allocations
    sns.histplot(allocations_var[0], bins=30, kde=True, ax=axes[1, 0], label='Agent 1')
    sns.histplot(allocations_var[1], bins=30, kde=True, ax=axes[1, 0], label='Agent 2')
    axes[1, 0].axvline(var_shared[0], color='blue', linestyle='--', label=f'VaR Agent 1')
    axes[1, 0].axvline(var_shared[1], color='orange', linestyle='--', label=f'VaR Agent 2')
    axes[1, 0].set_title("Distribution of VaR-based Allocations")
    axes[1, 0].set_xlabel("Loss")
    axes[1, 0].legend()
    
    # Plot distribution of ES-based allocations
    sns.histplot(allocations_es[0], bins=30, kde=True, ax=axes[1, 1], label='Agent 1')
    sns.histplot(allocations_es[1], bins=30, kde=True, ax=axes[1, 1], label='Agent 2')
    axes[1, 1].axvline(es_shared[0], color='blue', linestyle='--', label=f'ES Agent 1')
    axes[1, 1].axvline(es_shared[1], color='orange', linestyle='--', label=f'ES Agent 2')
    axes[1, 1].set_title("Distribution of ES-based Allocations")
    axes[1, 1].set_xlabel("Loss")
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig("risk_sharing.png")
    plt.show()
    
    return {
        'allocations_var': allocations_var,
        'allocations_es': allocations_es,
        'var_original': var_original,
        'var_shared': var_shared,
        'es_original': es_original,
        'es_shared': es_shared
    }

###########################################
# 8. EXPERIMENT 4: VaRq(VaRp(X|W)) vs VaRp(X)
###########################################

def experiment_varq_varp_comparison():
    """
    Replicate and extend the comparison in Section 7.2 of the paper
    """
    print("Experiment 4: VaRq(VaRp(X|W)) vs VaRp(X)")
    print("=" * 80)
    
    # Parameters
    p_values = [0.95, 0.975, 0.99]
    q_values = np.linspace(0.5, 0.99, 10)
    beta_values = [0.5, 1.0, 1.5]  # Factor sensitivity
    sigma_values = [0.5, 1.0, 1.5]  # Idiosyncratic volatility
    
    # Create grid for heatmap
    p_grid, q_grid = np.meshgrid(p_values, q_values)
    diff_grid = {}
    
    for beta in beta_values:
        for sigma in sigma_values:
            key = f"beta={beta:.1f}, sigma={sigma:.1f}"
            diff_grid[key] = np.zeros_like(p_grid, dtype=float)
    
    # Simulation
    print("Running simulations...")
    n_samples = 10000
    for i, p in enumerate(p_values):
        for j, q in enumerate(q_values):
            for beta_idx, beta in enumerate(beta_values):
                for sigma_idx, sigma in enumerate(sigma_values):
                    # Simulate data
                    np.random.seed(42)  # For reproducibility
                    
                    # Factor
                    W = np.random.normal(0, 1, n_samples)
                    
                    # Idiosyncratic risk
                    epsilon = np.random.normal(0, 1, n_samples)
                    
                    # Loss (negative return)
                    X = beta * W + sigma * epsilon
                    
                    # Calculate risk measures
                    var_p = var(X, p)
                    var_q_var_p = var_of_conditional_var(X, pd.qcut(W, 10, labels=False), p, q)
                    
                    # Calculate percentage difference
                    diff = var_q_var_p / var_p - 1
                    diff_grid[f"beta={beta:.1f}, sigma={sigma:.1f}"][j, i] = diff
    
    # Plot heatmaps
    fig, axes = plt.subplots(len(beta_values), len(sigma_values), figsize=(15, 12))
    
    for beta_idx, beta in enumerate(beta_values):
        for sigma_idx, sigma in enumerate(sigma_values):
            key = f"beta={beta:.1f}, sigma={sigma:.1f}"
            ax = axes[beta_idx, sigma_idx]
            
            # Create heatmap
            im = ax.pcolormesh(p_grid, q_grid, diff_grid[key], cmap='RdBu_r', 
                              vmin=-0.2, vmax=0.5)
            
            # Add contour for zero difference
            cs = ax.contour(p_grid, q_grid, diff_grid[key], levels=[0], colors='black')
            ax.clabel(cs, inline=True, fontsize=10)
            
            ax.set_title(f"β = {beta}, σ = {sigma}")
            ax.set_xlabel('p')
            ax.set_ylabel('q')
    
    # Add colorbar
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.88, 0.15, 0.03, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('Percentage Difference: VaRq(VaRp(X|W))/VaRp(X) - 1')
    
    plt.suptitle("Comparison of VaRq(VaRp(X|W)) and VaRp(X)", fontsize=16)
    plt.tight_layout(rect=[0, 0, 0.85, 0.95])
    plt.savefig("varq_varp_comparison.png")
    plt.show()
    
    # Find q0 for which VaRq0(VaRp(X|W)) = VaRp(X) for a specific case
    beta, sigma, p = 1.0, 1.0, 0.95
    q0_values = []
    
    for _ in range(10):  # Multiple simulations for robustness
        # Simulate data
        np.random.seed(42 + _)
        
        # Factor
        W = np.random.normal(0, 1, n_samples)
        
        # Idiosyncratic risk
        epsilon = np.random.normal(0, 1, n_samples)
        
        # Loss (negative return)
        X = beta * W + sigma * epsilon
        
        # Calculate VaR_p(X)
        var_p_value = var(X, p)
        
        # Find q0 where VaRq0(VaRp(X|W)) ≈ VaRp(X)
        q_values_fine = np.linspace(0.1, 0.9, 100)
        differences = []
        
        for q in q_values_fine:
            var_q_var_p_value = var_of_conditional_var(X, pd.qcut(W, 10, labels=False), p, q)
            diff = abs(var_q_var_p_value - var_p_value)
            differences.append((q, diff))
        
        # Find q with smallest difference
        q0 = min(differences, key=lambda x: x[1])[0]
        q0_values.append(q0)
    
    q0_mean = np.mean(q0_values)
    print(f"For β = {beta}, σ = {sigma}, p = {p}:")
    print(f"Estimated q0 ≈ {q0_mean:.4f} (average of 10 simulations)")
    print(f"This means VaR_{p}(X) can satisfy the capital requirement for approximately {q0_mean*100:.1f}% of different scenarios.")
    
    return diff_grid

###########################################
# MAIN
###########################################

if __name__ == "__main__":
    # Run experiments
    print("\n" + "=" * 80)
    print("TESTING FACTOR RISK MEASURES")
    print("=" * 80 + "\n")
    
    # Experiment 1: Basic comparison
    results_1 = experiment_basic_comparison()
    
    # Experiment 2: Sensitivity analysis
    # Uncomment the line below to run the sensitivity analysis (takes time)
    # results_2 = experiment_sensitivity_analysis()
    
    # Experiment 3: Risk sharing
    results_3 = experiment_risk_sharing()
    
    # Experiment 4: VaRq(VaRp(X|W)) vs VaRp(X) comparison
    results_4 = experiment_varq_varp_comparison()
    
    print("\n" + "=" * 80)
    print("EXPERIMENTS COMPLETED")
    print("=" * 80)