import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import minimize
import pandas as pd
from tqdm import tqdm

# Set random seed for reproducibility
np.random.seed(42)

class PartiallyLawInvariantRiskMeasures:
    """
    Implementation of partially law-invariant risk measures from the paper
    "Partial Law Invariance and Risk Measures" by Yi Shen, Zachary Van Oosten, and Ruodu Wang.
    """
    
    def __init__(self):
        pass
    
    def expected_shortfall(self, X, alpha=0.95):
        """
        Calculate the standard Expected Shortfall (ES) at level alpha
        
        Parameters:
        X (array): Sample of random variable values
        alpha (float): Confidence level (default: 0.95)
        
        Returns:
        float: Expected Shortfall
        """
        X_sorted = np.sort(X)
        tail_idx = int(np.ceil(len(X) * (1 - alpha)))
        return np.mean(X_sorted[-tail_idx:])
    
    def conditional_expected_shortfall(self, X, Y, beta, alpha=0.95):
        """
        Calculate the conditional Expected Shortfall at level beta
        
        Parameters:
        X (array): Values of the random variable we're calculating ES for
        Y (array): Conditioning random variable
        beta (float): Parameter for robust ES (0 ≤ beta < 1)
        alpha (float): Confidence level
        
        Returns:
        float: Conditional Expected Shortfall
        """
        n = len(X)
        
        # Calculate conditional ES based on the paper's formulation
        def g_beta(r, sigma, x):
            """
            Helper function as described in Equation (23) of the paper
            """
            phi_inv_beta = norm.ppf(beta)
            phi_at_phi_inv_beta = norm.pdf(phi_inv_beta)
            
            if x <= r + sigma * phi_inv_beta:
                return r - x + sigma * phi_at_phi_inv_beta / (1 - beta)
            else:
                return ((r - x) * (1 - norm.cdf((x - r) / sigma)) / (1 - beta) + 
                        sigma * norm.pdf((x - r) / sigma) / (1 - beta))
        
        # Integration formula for f_beta as in Section 6 of the paper
        def f_beta(pi1, pi2, x, m1, m2, sigma1, sigma2, c):
            """
            Numerical integration for the expected value of conditional ES
            """
            # Number of integration points
            n_points = 100
            z_values = np.linspace(m1 - 4*sigma1, m1 + 4*sigma1, n_points)
            
            # M and sigma functions from the paper
            def M(pi1, pi2, z):
                return pi1*z + pi2*m2 + c*pi2*(sigma2/sigma1)*(z - m1)
            
            def sigma_func(pi2):
                return pi2 * np.sqrt((1 - c**2) * sigma2**2)
            
            # Compute integrand at each point
            integrand = np.zeros(n_points)
            for i, z in enumerate(z_values):
                integrand[i] = g_beta(M(pi1, pi2, z), sigma_func(pi2), x) * \
                              norm.pdf((z - m1) / sigma1) / sigma1
            
            # Numerical integration (trapezoidal rule)
            return np.trapz(integrand, z_values)
        
        # We'll implement the actual calculation based on the paper's Gaussian example
        # We need parameters: m1, m2, sigma1, sigma2, c
        # For demonstration, we'll extract these from the data
        
        m1 = np.mean(Y)
        m2 = np.mean(X) - m1 * np.cov(X, Y)[0, 1] / np.var(Y)  # Conditional mean
        sigma1 = np.std(Y)
        
        # Estimate correlation
        correlation = np.corrcoef(X, Y)[0, 1]
        c = correlation
        
        # Calculate conditional variance
        cond_var = np.var(X) * (1 - correlation**2)
        sigma2 = np.sqrt(cond_var)
        
        # For simplicity in this implementation, we're using a numerical approach
        # In the paper they derive an analytical formula
        
        # Function to optimize to find the ES
        def es_function(x, pi1, pi2):
            return x + f_beta(pi1, pi2, x, m1, m2, sigma1, sigma2, c) / (1 - alpha)
        
        # Find minimum over x for ES calculation
        result = minimize(lambda x: es_function(x[0], 0.5, 0.5), [0])
        return result.fun

    def partially_law_invariant_es(self, X, G, beta=0.95, alpha=0.95):
        """
        Calculate the partially law-invariant Expected Shortfall (ES) at level beta
        
        Parameters:
        X (array): Sample of random variable values
        G (array): The G-measurable random variable 
        beta (float): Parameter for robust ES (0 ≤ beta < 1)
        alpha (float): Confidence level
        
        Returns:
        float: Partially law-invariant ES
        """
        return self.conditional_expected_shortfall(X, G, beta, alpha)
    
    def optimize_portfolio(self, X1, X2, beta=0.95, alpha=0.95):
        """
        Find the optimal portfolio weights to minimize the partially law-invariant ES
        
        Parameters:
        X1 (array): Returns of asset 1
        X2 (array): Returns of asset 2
        beta (float): Parameter for robust ES
        alpha (float): Confidence level
        
        Returns:
        tuple: (optimal weight for asset 1, minimum risk value)
        """
        def portfolio_risk(pi1):
            pi2 = 1 - pi1
            portfolio_return = pi1 * X1 + pi2 * X2
            return self.partially_law_invariant_es(portfolio_return, X1, beta, alpha)
        
        # Optimize over pi1 in [0, 1]
        result = minimize(portfolio_risk, 0.5, bounds=[(0, 1)])
        return result.x[0], result.fun

def generate_correlated_assets(n_samples=1000, m1=0, m2=0, sigma1=0.1, sigma2=0.1, c=0.5):
    """
    Generate correlated asset returns following the Gaussian model in the paper
    
    Parameters:
    n_samples (int): Number of samples
    m1 (float): Mean of asset 1
    m2 (float): Mean of asset 2
    sigma1 (float): Volatility of asset 1
    sigma2 (float): Volatility of asset 2
    c (float): Correlation between assets
    
    Returns:
    tuple: (returns of asset 1, returns of asset 2)
    """
    # Covariance matrix
    cov_matrix = np.array([[sigma1**2, c*sigma1*sigma2], 
                            [c*sigma1*sigma2, sigma2**2]])
    
    # Generate multivariate normal returns
    returns = np.random.multivariate_normal([m1, m2], cov_matrix, n_samples)
    
    return returns[:, 0], returns[:, 1]

def run_experiment_varying_beta():
    """
    Reproduce Figure 1 from the paper: varying beta with fixed parameters
    """
    # Parameters from the paper
    m1 = 0
    sigma1 = 0.1
    sigma2 = 0.1
    c = 0.5
    alpha = 0.95
    
    # Vary m2 as in the paper
    m2_values = [0, -0.05, -0.1]
    beta_values = np.linspace(0, 0.95, 10)
    
    # Initialize risk measure
    rm = PartiallyLawInvariantRiskMeasures()
    
    plt.figure(figsize=(15, 5))
    
    for i, m2 in enumerate(m2_values):
        plt.subplot(1, 3, i+1)
        
        # Store optimal weights for each beta
        optimal_weights = []
        
        for beta in tqdm(beta_values, desc=f"m2={m2}"):
            pi1_values = np.linspace(0, 1, 101)
            risk_values = []
            
            # Generate a large sample of asset returns
            X1, X2 = generate_correlated_assets(n_samples=10000, m1=m1, m2=m2, 
                                              sigma1=sigma1, sigma2=sigma2, c=c)
            
            # Calculate risk for each portfolio weight
            for pi1 in pi1_values:
                pi2 = 1 - pi1
                portfolio_return = pi1 * X1 + pi2 * X2
                
                if beta == 0:
                    # Standard ES
                    risk = rm.expected_shortfall(portfolio_return, alpha)
                else:
                    # Partially law-invariant ES
                    risk = rm.partially_law_invariant_es(portfolio_return, X1, beta, alpha)
                
                risk_values.append(risk)
            
            # Plot risk vs. pi1
            plt.plot(pi1_values, risk_values, label=f'β={beta:.2f}')
            
            # Find optimal weight
            optimal_weights.append(pi1_values[np.argmin(risk_values)])
        
        plt.plot(optimal_weights, [np.min(risk_values) for _ in optimal_weights], 'r:')
        
        plt.xlabel('Weight in X1')
        plt.ylabel('Risk')
        plt.title(f'm1={m1}, m2={m2}, σ1={sigma1}, σ2={sigma2}, c={c}, α={alpha}')
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('varying_beta.png')
    plt.show()

def run_experiment_varying_sigma2():
    """
    Reproduce Figure 2 from the paper: varying sigma2 with fixed beta
    """
    # Parameters from the paper
    m1 = 0
    m2 = -0.1  # Negative m2 as in paper's Figure 2
    sigma1 = 0.1
    c = 0.5
    alpha = 0.95
    beta = 0.95  # Fixed beta
    
    # Vary sigma2 as in the paper
    sigma2_values = [0.05, 0.075, 0.1, 0.125]
    
    # Initialize risk measure
    rm = PartiallyLawInvariantRiskMeasures()
    
    plt.figure(figsize=(15, 5))
    
    for i, sigma2 in enumerate(sigma2_values):
        plt.subplot(1, 4, i+1)
        
        pi1_values = np.linspace(0, 1, 101)
        risk_values = []
        
        # Generate a large sample of asset returns
        X1, X2 = generate_correlated_assets(n_samples=10000, m1=m1, m2=m2, 
                                          sigma1=sigma1, sigma2=sigma2, c=c)
        
        # Calculate risk for each portfolio weight
        for pi1 in tqdm(pi1_values, desc=f"sigma2={sigma2}"):
            pi2 = 1 - pi1
            portfolio_return = pi1 * X1 + pi2 * X2
            
            # Partially law-invariant ES
            risk = rm.partially_law_invariant_es(portfolio_return, X1, beta, alpha)
            risk_values.append(risk)
        
        # Plot risk vs. pi1
        plt.plot(pi1_values, risk_values)
        
        # Find and mark optimal weight
        optimal_pi1 = pi1_values[np.argmin(risk_values)]
        min_risk = np.min(risk_values)
        plt.plot(optimal_pi1, min_risk, 'ro')
        
        plt.xlabel('Weight in X1')
        plt.ylabel('Risk')
        plt.title(f'm1={m1}, m2={m2}, σ1={sigma1}, σ2={sigma2}, c={c}, β={beta}')
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('varying_sigma2.png')
    plt.show()

def optimize_portfolio_comparison():
    """
    Compare optimal portfolio allocation using different risk measures:
    1. Standard ES (β=0)
    2. Partially law-invariant ES (β=0.95)
    """
    # Parameters
    m1 = 0
    m2 = -0.05  # Negative expected return for asset 2
    sigma1 = 0.1
    sigma2 = 0.1
    c = 0.5
    alpha = 0.95
    
    # Generate a large sample of asset returns
    X1, X2 = generate_correlated_assets(n_samples=10000, m1=m1, m2=m2, 
                                      sigma1=sigma1, sigma2=sigma2, c=c)
    
    # Initialize risk measure
    rm = PartiallyLawInvariantRiskMeasures()
    
    # Calculate optimal weights for different beta values
    beta_values = [0, 0.5, 0.95]
    results = []
    
    for beta in beta_values:
        # Find optimal portfolio
        optimal_pi1, min_risk = rm.optimize_portfolio(X1, X2, beta, alpha)
        
        # Compute portfolio returns and standard ES for this allocation
        portfolio_return = optimal_pi1 * X1 + (1 - optimal_pi1) * X2
        standard_es = rm.expected_shortfall(portfolio_return, alpha)
        expected_return = optimal_pi1 * m1 + (1 - optimal_pi1) * m2
        
        results.append({
            'beta': beta,
            'optimal_weight_X1': optimal_pi1,
            'optimal_weight_X2': 1 - optimal_pi1,
            'min_risk': min_risk,
            'standard_ES': standard_es,
            'expected_return': expected_return
        })
    
    # Display results
    results_df = pd.DataFrame(results)
    print("Optimal portfolio allocation comparison:")
    print(results_df)
    
    # Plot weights vs beta
    plt.figure(figsize=(10, 6))
    plt.plot(beta_values, results_df['optimal_weight_X1'], 'o-', label='Weight in X1')
    plt.plot(beta_values, results_df['optimal_weight_X2'], 'o-', label='Weight in X2')
    plt.xlabel('Beta (model uncertainty parameter)')
    plt.ylabel('Optimal Portfolio Weight')
    plt.title('Optimal Portfolio Allocation vs. Model Uncertainty')
    plt.legend()
    plt.grid(True)
    plt.savefig('optimal_weights_vs_beta.png')
    plt.show()

# Run simulations
if __name__ == "__main__":
    print("Running simulations for partially law-invariant risk measures...")
    
    # Run experiments
    run_experiment_varying_beta()
    run_experiment_varying_sigma2()
    optimize_portfolio_comparison()

def demonstrate_partial_law_invariance():
    """
    Demonstrate the core concept of partial law invariance with a simple example
    """
    np.random.seed(42)
    
    # Create two random variables with the same distribution but different sources
    # X1 is G-measurable (depends only on the first source)
    # X2 depends on both sources but has the same distribution as X1
    
    n_samples = 10000
    
    # First source (G-measurable)
    G = np.random.normal(0, 1, n_samples)
    
    # X1 is a function of G only
    X1 = np.exp(G) - 1  # Log-normal minus 1
    
    # Create another source independent of G
    H = np.random.normal(0, 1, n_samples)
    
    # X2 depends on both sources but has same distribution as X1
    # We use rank-based transformation to ensure identical distributions
    ranks_X1 = np.argsort(X1)
    Z = 0.7 * G + 0.7 * H  # Z depends on both sources
    X2 = np.zeros_like(Z)
    X2[np.argsort(Z)] = X1[ranks_X1]  # X2 has same distribution as X1
    
    # Verify they have the same distribution
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.hist(X1, bins=50, alpha=0.5, label='X1 (G-measurable)')
    plt.hist(X2, bins=50, alpha=0.5, label='X2 (not G-measurable)')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Distributions of X1 and X2')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.scatter(G, X1, alpha=0.3, label='X1 vs G')
    plt.scatter(G, X2, alpha=0.3, label='X2 vs G')
    plt.xlabel('G')
    plt.ylabel('Value')
    plt.title('Relationship with G')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('partial_law_invariance_example.png')
    plt.show()
    
    # Initialize risk measures
    rm = PartiallyLawInvariantRiskMeasures()
    
    # Compare standard ES (law-invariant) with partially law-invariant ES
    standard_es_X1 = rm.expected_shortfall(X1)
    standard_es_X2 = rm.expected_shortfall(X2)
    
    # Partially law-invariant ES with different beta values
    pli_es_X1_beta0 = rm.partially_law_invariant_es(X1, G, beta=0)
    pli_es_X1_beta95 = rm.partially_law_invariant_es(X1, G, beta=0.95)
    pli_es_X2_beta0 = rm.partially_law_invariant_es(X2, G, beta=0)
    pli_es_X2_beta95 = rm.partially_law_invariant_es(X2, G, beta=0.95)
    
    # Print results
    print("\nRisk measure comparison:")
    print(f"Standard ES (law-invariant):")
    print(f"  ES(X1) = {standard_es_X1:.4f}")
    print(f"  ES(X2) = {standard_es_X2:.4f}")
    print(f"  Difference: {abs(standard_es_X1 - standard_es_X2):.4f}")
    
    print(f"\nPartially law-invariant ES (beta=0):")
    print(f"  ES(X1) = {pli_es_X1_beta0:.4f}")
    print(f"  ES(X2) = {pli_es_X2_beta0:.4f}")
    print(f"  Difference: {abs(pli_es_X1_beta0 - pli_es_X2_beta0):.4f}")
    
    print(f"\nPartially law-invariant ES (beta=0.95):")
    print(f"  ES(X1) = {pli_es_X1_beta95:.4f}")
    print(f"  ES(X2) = {pli_es_X2_beta95:.4f}")
    print(f"  Difference: {abs(pli_es_X1_beta95 - pli_es_X2_beta95):.4f}")

# Run the demonstration
demonstrate_partial_law_invariance()
