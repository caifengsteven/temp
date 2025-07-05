import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.optimize import minimize
import time
from tqdm import tqdm

# Set random seed for reproducibility
np.random.seed(42)

# Styling for plots
plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

class NTSMarketModel:
    """
    Implementation of the Normal Tempered Stable (NTS) market model for portfolio optimization
    with relative tail risk measures (CoVaR and CoCVaR).
    """
    
    def __init__(self, alpha=1.1835, theta=0.0820):
        """
        Initialize the NTS market model with parameters.
        
        Parameters:
        -----------
        alpha : float, optional (default=1.1835)
            Tail parameter of the tempered stable distribution (0 < alpha < 2)
        theta : float, optional (default=0.0820)
            Tempering parameter of the tempered stable distribution (theta > 0)
        """
        self.alpha = alpha
        self.theta = theta
        self.mu = None
        self.sigma = None
        self.beta = None
        self.rho_matrix = None
        self.gamma = None
        self.n_assets = 0
        self.index_pos = None  # Position of the market index in the parameters
        
    def fit(self, returns, index_pos=0):
        """
        Fit the NTS market model to historical returns.
        
        Parameters:
        -----------
        returns : pandas.DataFrame
            DataFrame of returns with assets in columns and time in rows
        index_pos : int, optional (default=0)
            Position of the market index in returns dataframe
            
        Returns:
        --------
        self : NTSMarketModel
            Fitted model
        """
        self.n_assets = returns.shape[1]
        self.index_pos = index_pos
        
        # Calculate mean and standard deviation of returns
        self.mu = returns.mean().values
        self.sigma = returns.std().values
        
        # Standardize returns
        standardized_returns = (returns - returns.mean()) / returns.std()
        
        # Estimate beta parameters using the method described in the paper
        # For simplicity, we'll use values similar to those in the paper
        self.beta = np.random.uniform(-0.5, 0.5, self.n_assets)
        
        # Ensure beta values satisfy the constraint |beta_n| < sqrt(2*theta/(2-alpha))
        max_beta = np.sqrt(2 * self.theta / (2 - self.alpha))
        self.beta = np.clip(self.beta, -max_beta * 0.95, max_beta * 0.95)
        
        # Calculate gamma values using the formula in the paper
        self.gamma = np.sqrt(1 - (self.beta**2) * ((2-self.alpha)/(2*self.theta)))
        
        # Calculate correlation matrix from standardized returns
        self.rho_matrix = standardized_returns.corr().values
        
        return self
    
    def calculate_portfolio_params(self, w, index_weight=None):
        """
        Calculate portfolio parameters based on weights.
        
        Parameters:
        -----------
        w : numpy.ndarray
            Portfolio weights (excluding index)
        index_weight : float or None, optional (default=None)
            Weight of the index in the portfolio
            
        Returns:
        --------
        params : dict
            Dictionary containing portfolio parameters
        """
        # If index is included in weights, separate it
        if index_weight is not None:
            w_full = np.zeros(self.n_assets)
            w_full[1:] = w * (1 - index_weight)
            w_full[0] = index_weight
            w_assets = w_full[1:]
        else:
            w_assets = w
        
        # Calculate parameters excluding the index
        non_index_pos = np.array([i for i in range(self.n_assets) if i != self.index_pos])
        
        mu_assets = self.mu[non_index_pos]
        sigma_assets = self.sigma[non_index_pos]
        beta_assets = self.beta[non_index_pos]
        gamma_assets = self.gamma[non_index_pos]
        
        # Extract correlation between assets and index
        rho_assets_index = self.rho_matrix[non_index_pos, self.index_pos]
        
        # Extract correlation matrix among assets
        rho_assets = self.rho_matrix[np.ix_(non_index_pos, non_index_pos)]
        
        # Calculate portfolio expected return
        mu_p = np.sum(w_assets * mu_assets)
        
        # Calculate portfolio standard deviation
        sigma_p_squared = 0
        for i in range(len(w_assets)):
            for j in range(len(w_assets)):
                cov_ij = sigma_assets[i] * sigma_assets[j] * (
                    gamma_assets[i] * gamma_assets[j] * rho_assets[i, j] + 
                    beta_assets[i] * beta_assets[j] * ((2 - self.alpha) / (2 * self.theta))
                )
                sigma_p_squared += w_assets[i] * w_assets[j] * cov_ij
        
        sigma_p = np.sqrt(sigma_p_squared)
        
        # Calculate portfolio beta parameter
        beta_p = np.sum(w_assets * sigma_assets * beta_assets) / sigma_p
        
        # Calculate portfolio gamma parameter
        gamma_p = np.sqrt(1 - (beta_p**2) * ((2-self.alpha)/(2*self.theta)))
        
        # Calculate portfolio-index correlation
        numerator = np.sum(w_assets * gamma_assets * sigma_assets * rho_assets_index)
        denominator = np.sqrt(np.sum(np.outer(w_assets * gamma_assets * sigma_assets, 
                                            w_assets * gamma_assets * sigma_assets) * rho_assets))
        rho_p = numerator / denominator
        
        return {
            'mu_p': mu_p,
            'sigma_p': sigma_p,
            'beta_p': beta_p,
            'gamma_p': gamma_p,
            'rho_p': rho_p
        }
    
    def generate_ts_subordinator(self, n_samples=100000):
        """
        Generate tempered stable subordinator samples.
        
        Parameters:
        -----------
        n_samples : int, optional (default=100000)
            Number of samples to generate
            
        Returns:
        --------
        T : numpy.ndarray
            Samples from the tempered stable subordinator
        """
        # For simplicity, we'll approximate the tempered stable subordinator
        # using a gamma distribution with appropriate parameters
        # This is not exact but provides a reasonable approximation
        shape = self.alpha
        scale = 1 / (2 * self.theta)
        
        T = np.random.gamma(shape, scale, n_samples)
        return T
    
    def calculate_var(self, ζ, index=True):
        """
        Calculate VaR for the index or a specific asset.
        
        Parameters:
        -----------
        ζ : float
            Significance level (e.g., 0.05 for 95% VaR)
        index : bool, optional (default=True)
            If True, calculate VaR for the index, otherwise for the portfolio
            
        Returns:
        --------
        VaR : float
            Value at Risk
        """
        if index:
            # For simplicity, we'll use a normal approximation for the index VaR
            # In a full implementation, you would use the NTS distribution
            mu_0 = self.mu[self.index_pos]
            sigma_0 = self.sigma[self.index_pos]
            var = sigma_0 * stats.norm.ppf(ζ) - mu_0
        else:
            # For the portfolio VaR, we need to simulate from the NTS distribution
            # This is a simplified approximation
            var = 0  # Placeholder
            
        return var
    
    def calculate_covar_cocovar_mc(self, w, η=0.05, ζ=0.05, n_samples=100000):
        """
        Calculate CoVaR and CoCVaR using Monte Carlo simulation.
        
        Parameters:
        -----------
        w : numpy.ndarray
            Portfolio weights (excluding index)
        η : float, optional (default=0.05)
            Significance level for CoVaR/CoCVaR
        ζ : float, optional (default=0.05)
            Significance level for conditioning event (index VaR)
        n_samples : int, optional (default=100000)
            Number of samples for Monte Carlo simulation
            
        Returns:
        --------
        results : dict
            Dictionary containing CoVaR and CoCVaR values
        """
        # Calculate portfolio parameters
        params = self.calculate_portfolio_params(w)
        mu_p = params['mu_p']
        sigma_p = params['sigma_p']
        beta_p = params['beta_p']
        gamma_p = params['gamma_p']
        rho_p = params['rho_p']
        
        # Generate samples for the tempered stable subordinator
        T = self.generate_ts_subordinator(n_samples)
        
        # Generate standard normal samples for the index and portfolio
        Z0 = np.random.normal(0, 1, n_samples)
        Z1 = np.random.normal(0, 1, n_samples)
        
        # Generate correlated standard normal for the portfolio
        Zp = rho_p * Z0 + np.sqrt(1 - rho_p**2) * Z1
        
        # Calculate index returns
        mu_0 = self.mu[self.index_pos]
        sigma_0 = self.sigma[self.index_pos]
        beta_0 = self.beta[self.index_pos]
        gamma_0 = self.gamma[self.index_pos]
        
        index_returns = mu_0 + sigma_0 * (beta_0 * (T - 1) + gamma_0 * np.sqrt(T) * Z0)
        
        # Calculate portfolio returns
        portfolio_returns = mu_p + sigma_p * (beta_p * (T - 1) + gamma_p * np.sqrt(T) * Zp)
        
        # Calculate VaR of the index
        var_index = -np.percentile(index_returns, ζ * 100)
        
        # Identify samples where index returns are below -VaR
        index_distress = index_returns <= -var_index
        
        # Calculate CoVaR as the conditional VaR of the portfolio
        covar = -np.percentile(portfolio_returns[index_distress], η * 100)
        
        # Calculate CoCVaR as the conditional expected shortfall
        portfolio_extreme = portfolio_returns[index_distress] <= -covar
        cocovar = -np.mean(portfolio_returns[index_distress][portfolio_extreme])
        
        return {
            'CoVaR': covar,
            'CoCVaR': cocovar,
            'VaR_index': var_index
        }
    
    def calculate_marginal_contributions_mc(self, w, η=0.05, ζ=0.05, n_samples=100000, delta=1e-6):
        """
        Calculate marginal contributions to CoVaR and CoCVaR using Monte Carlo simulation.
        
        Parameters:
        -----------
        w : numpy.ndarray
            Portfolio weights (excluding index)
        η : float, optional (default=0.05)
            Significance level for CoVaR/CoCVaR
        ζ : float, optional (default=0.05)
            Significance level for conditioning event (index VaR)
        n_samples : int, optional (default=100000)
            Number of samples for Monte Carlo simulation
        delta : float, optional (default=1e-6)
            Small increment for numerical differentiation
            
        Returns:
        --------
        results : dict
            Dictionary containing marginal contributions
        """
        # Calculate base CoVaR and CoCVaR
        base_results = self.calculate_covar_cocovar_mc(w, η, ζ, n_samples)
        base_covar = base_results['CoVaR']
        base_cocovar = base_results['CoCVaR']
        
        # Initialize arrays for marginal contributions
        n_assets = len(w)
        marginal_covar = np.zeros(n_assets)
        marginal_cocovar = np.zeros(n_assets)
        
        # Calculate marginal contributions numerically
        for i in range(n_assets):
            # Create perturbed weight vector
            w_perturbed = w.copy()
            w_perturbed[i] += delta
            
            # Normalize weights to sum to 1
            w_perturbed = w_perturbed / np.sum(w_perturbed)
            
            # Calculate perturbed CoVaR and CoCVaR
            perturbed_results = self.calculate_covar_cocovar_mc(w_perturbed, η, ζ, n_samples)
            perturbed_covar = perturbed_results['CoVaR']
            perturbed_cocovar = perturbed_results['CoCVaR']
            
            # Calculate marginal contributions
            marginal_covar[i] = (perturbed_covar - base_covar) / delta
            marginal_cocovar[i] = (perturbed_cocovar - base_cocovar) / delta
        
        return {
            'Marginal_CoVaR': marginal_covar,
            'Marginal_CoCVaR': marginal_cocovar
        }
    
    def optimize_portfolio_cocovar(self, target_return, η=0.05, ζ=0.05, n_samples=10000):
        """
        Optimize portfolio to minimize CoCVaR subject to a target return.
        
        Parameters:
        -----------
        target_return : float
            Target portfolio return
        η : float, optional (default=0.05)
            Significance level for CoVaR/CoCVaR
        ζ : float, optional (default=0.05)
            Significance level for conditioning event (index VaR)
        n_samples : int, optional (default=10000)
            Number of samples for Monte Carlo simulation
            
        Returns:
        --------
        results : dict
            Dictionary containing optimized weights and metrics
        """
        n_assets = self.n_assets - 1  # Excluding the index
        
        # Define objective function to minimize CoCVaR
        def objective(w):
            # Normalize weights to sum to 1
            w_normalized = w / np.sum(w)
            
            # Calculate CoCVaR
            results = self.calculate_covar_cocovar_mc(w_normalized, η, ζ, n_samples)
            return results['CoCVaR']
        
        # Define constraint for target return
        def return_constraint(w):
            # Normalize weights to sum to 1
            w_normalized = w / np.sum(w)
            
            # Calculate portfolio parameters
            params = self.calculate_portfolio_params(w_normalized)
            return params['mu_p'] - target_return
        
        # Define constraint that weights sum to 1
        def sum_constraint(w):
            return np.sum(w) - 1
        
        # Initial weights (equal allocation)
        initial_weights = np.ones(n_assets) / n_assets
        
        # Bounds for weights (long-only)
        bounds = [(0, 1) for _ in range(n_assets)]
        
        # Define constraints
        constraints = [
            {'type': 'eq', 'fun': sum_constraint},
            {'type': 'ineq', 'fun': return_constraint}
        ]
        
        # Optimize
        result = minimize(
            objective,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 200}
        )
        
        # Get optimal weights
        optimal_weights = result['x'] / np.sum(result['x'])
        
        # Calculate metrics for optimal portfolio
        optimal_params = self.calculate_portfolio_params(optimal_weights)
        optimal_risk = self.calculate_covar_cocovar_mc(optimal_weights, η, ζ, n_samples)
        
        return {
            'weights': optimal_weights,
            'expected_return': optimal_params['mu_p'],
            'CoVaR': optimal_risk['CoVaR'],
            'CoCVaR': optimal_risk['CoCVaR'],
            'convergence': result['success']
        }
    
    def calculate_efficient_frontier(self, return_points=20, η=0.05, ζ=0.05, n_samples=10000):
        """
        Calculate the efficient frontier for CoCVaR optimization.
        
        Parameters:
        -----------
        return_points : int, optional (default=20)
            Number of points on the efficient frontier
        η : float, optional (default=0.05)
            Significance level for CoVaR/CoCVaR
        ζ : float, optional (default=0.05)
            Significance level for conditioning event (index VaR)
        n_samples : int, optional (default=10000)
            Number of samples for Monte Carlo simulation
            
        Returns:
        --------
        frontier : pandas.DataFrame
            DataFrame containing efficient frontier points
        """
        # Exclude the index from calculations
        non_index_pos = np.array([i for i in range(self.n_assets) if i != self.index_pos])
        mu_assets = self.mu[non_index_pos]
        
        # Determine min and max possible returns
        min_return = np.min(mu_assets)
        max_return = np.max(mu_assets)
        
        # Generate target returns
        target_returns = np.linspace(min_return, max_return, return_points)
        
        # Calculate optimal portfolios for each target return
        results = []
        for target_return in tqdm(target_returns, desc="Calculating Efficient Frontier"):
            try:
                optimal = self.optimize_portfolio_cocovar(target_return, η, ζ, n_samples)
                if optimal['convergence']:
                    results.append({
                        'target_return': target_return,
                        'realized_return': optimal['expected_return'],
                        'CoVaR': optimal['CoVaR'],
                        'CoCVaR': optimal['CoCVaR'],
                        'weights': optimal['weights']
                    })
            except:
                continue
        
        # Convert to DataFrame
        frontier = pd.DataFrame(results)
        return frontier
    
    def risk_budgeting(self, initial_weights, iterations=50, η=0.05, ζ=0.05, n_samples=10000):
        """
        Perform risk budgeting to reduce CoVaR and CoCVaR.
        
        Parameters:
        -----------
        initial_weights : numpy.ndarray
            Initial portfolio weights
        iterations : int, optional (default=50)
            Number of iterations for risk budgeting
        η : float, optional (default=0.05)
            Significance level for CoVaR/CoCVaR
        ζ : float, optional (default=0.05)
            Significance level for conditioning event (index VaR)
        n_samples : int, optional (default=10000)
            Number of samples for Monte Carlo simulation
            
        Returns:
        --------
        results : dict
            Dictionary containing risk budgeting results
        """
        n_assets = len(initial_weights)
        weights = initial_weights.copy()
        
        # Track risk measures over iterations
        covar_history = np.zeros(iterations + 1)
        cocovar_history = np.zeros(iterations + 1)
        
        # Calculate initial risk measures
        initial_risk = self.calculate_covar_cocovar_mc(weights, η, ζ, n_samples)
        covar_history[0] = initial_risk['CoVaR']
        cocovar_history[0] = initial_risk['CoCVaR']
        
        # Calculate initial expected return
        initial_params = self.calculate_portfolio_params(weights)
        initial_return = initial_params['mu_p']
        
        # Perform risk budgeting iterations
        for i in range(iterations):
            # Calculate marginal contributions
            marginal_contributions = self.calculate_marginal_contributions_mc(
                weights, η, ζ, n_samples
            )
            
            # Define the linear programming problem for risk budgeting
            def objective(delta_w):
                return np.sum(marginal_contributions['Marginal_CoCVaR'] * delta_w)
            
            # Constraint: weights must sum to 0 (rebalancing)
            def sum_constraint(delta_w):
                return np.sum(delta_w)
            
            # Constraint: expected return must not decrease
            def return_constraint(delta_w):
                non_index_pos = np.array([i for i in range(self.n_assets) if i != self.index_pos])
                mu_assets = self.mu[non_index_pos]
                return np.sum(delta_w * mu_assets)
            
            # Initial delta weights
            delta_w_initial = np.zeros(n_assets)
            
            # Bounds for delta weights (small adjustments)
            delta = 0.01  # Maximum weight adjustment
            bounds = [(-delta, delta) for _ in range(n_assets)]
            
            # Define constraints
            constraints = [
                {'type': 'eq', 'fun': sum_constraint},
                {'type': 'ineq', 'fun': return_constraint}
            ]
            
            # Optimize
            result = minimize(
                objective,
                delta_w_initial,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 100}
            )
            
            # Update weights
            delta_w_optimal = result['x']
            weights = weights + delta_w_optimal
            
            # Ensure weights are non-negative and sum to 1
            weights = np.maximum(weights, 0)
            weights = weights / np.sum(weights)
            
            # Calculate new risk measures
            new_risk = self.calculate_covar_cocovar_mc(weights, η, ζ, n_samples)
            covar_history[i + 1] = new_risk['CoVaR']
            cocovar_history[i + 1] = new_risk['CoCVaR']
        
        # Calculate final expected return
        final_params = self.calculate_portfolio_params(weights)
        final_return = final_params['mu_p']
        
        return {
            'initial_weights': initial_weights,
            'final_weights': weights,
            'initial_return': initial_return,
            'final_return': final_return,
            'initial_CoVaR': initial_risk['CoVaR'],
            'final_CoVaR': new_risk['CoVaR'],
            'initial_CoCVaR': initial_risk['CoCVaR'],
            'final_CoCVaR': new_risk['CoCVaR'],
            'CoVaR_history': covar_history,
            'CoCVaR_history': cocovar_history
        }

# Function to generate simulated stock returns
def generate_simulated_returns(n_assets=30, n_days=1000, index_vol=0.015, seed=42):
    """
    Generate simulated returns for a market index and multiple stocks.
    
    Parameters:
    -----------
    n_assets : int, optional (default=30)
        Number of stocks to generate
    n_days : int, optional (default=1000)
        Number of days of returns to generate
    index_vol : float, optional (default=0.015)
        Volatility of the market index
    seed : int, optional (default=42)
        Random seed for reproducibility
        
    Returns:
    --------
    returns : pandas.DataFrame
        DataFrame of simulated returns
    """
    np.random.seed(seed)
    
    # Generate market index returns
    index_mean = 0.0005  # Approximate daily return of 0.05%
    index_returns = np.random.normal(index_mean, index_vol, n_days)
    
    # Generate stock returns with varying betas and idiosyncratic volatility
    stock_returns = np.zeros((n_days, n_assets))
    
    # Stock parameters
    betas = np.random.uniform(0.5, 1.5, n_assets)
    idio_vols = np.random.uniform(0.01, 0.03, n_assets)
    alphas = np.random.normal(0.0001, 0.0003, n_assets)  # Small alpha (outperformance)
    
    # Generate correlated returns
    for i in range(n_assets):
        # Systematic component (market-driven)
        systematic = betas[i] * index_returns
        
        # Idiosyncratic component
        idiosyncratic = np.random.normal(0, idio_vols[i], n_days)
        
        # Combined return with alpha
        stock_returns[:, i] = alphas[i] + systematic + idiosyncratic
    
    # Create DataFrame
    cols = ['Market'] + [f'Stock_{i+1}' for i in range(n_assets)]
    returns_data = np.column_stack([index_returns, stock_returns])
    returns = pd.DataFrame(returns_data, columns=cols)
    
    return returns

# Main function to run simulations and analysis
def main():
    print("Generating simulated returns data...")
    returns = generate_simulated_returns(n_assets=30, n_days=1000)
    
    print("\nFitting NTS Market Model...")
    model = NTSMarketModel()
    model.fit(returns)
    
    print("\nTesting CoVaR and CoCVaR calculation...")
    # Equal-weight portfolio (excluding the index)
    equal_weights = np.ones(30) / 30
    
    # Calculate CoVaR and CoCVaR using Monte Carlo
    start_time = time.time()
    risk_measures = model.calculate_covar_cocovar_mc(equal_weights)
    calc_time = time.time() - start_time
    
    print(f"CoVaR: {risk_measures['CoVaR']:.6f}")
    print(f"CoCVaR: {risk_measures['CoCVaR']:.6f}")
    print(f"Calculation time: {calc_time:.2f} seconds")
    
    print("\nCalculating marginal contributions...")
    marginal_contributions = model.calculate_marginal_contributions_mc(equal_weights)
    
    # Plot top 5 and bottom 5 contributors to risk
    mct_covar = marginal_contributions['Marginal_CoVaR']
    mct_cocovar = marginal_contributions['Marginal_CoCVaR']
    
    # Sort by marginal CoCVaR contribution
    sorted_indices = np.argsort(mct_cocovar)
    lowest_contributors = sorted_indices[:5]
    highest_contributors = sorted_indices[-5:]
    
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.bar(range(5), mct_cocovar[lowest_contributors], color='green', alpha=0.7)
    plt.title('Lowest Marginal CoCVaR Contributors')
    plt.xticks(range(5), [f'Stock_{i+1}' for i in lowest_contributors])
    plt.ylabel('Marginal Contribution')
    
    plt.subplot(2, 1, 2)
    plt.bar(range(5), mct_cocovar[highest_contributors], color='red', alpha=0.7)
    plt.title('Highest Marginal CoCVaR Contributors')
    plt.xticks(range(5), [f'Stock_{i+1}' for i in highest_contributors])
    plt.ylabel('Marginal Contribution')
    
    plt.tight_layout()
    plt.savefig('marginal_contributions.png')
    plt.close()
    
    print("\nPerforming portfolio optimization...")
    # Calculate efficient frontier (with limited points due to computational intensity)
    frontier = model.calculate_efficient_frontier(return_points=10, n_samples=5000)
    
    # Plot efficient frontier
    plt.figure(figsize=(10, 6))
    plt.scatter(frontier['CoCVaR'], frontier['realized_return'], c='blue', alpha=0.7)
    plt.xlabel('CoCVaR')
    plt.ylabel('Expected Return')
    plt.title('Efficient Frontier: CoCVaR vs Expected Return')
    plt.grid(True, alpha=0.3)
    plt.savefig('efficient_frontier.png')
    plt.close()
    
    print("\nPerforming risk budgeting...")
    # Perform risk budgeting on equal-weight portfolio
    risk_budget_results = model.risk_budgeting(equal_weights, iterations=20, n_samples=5000)
    
    # Plot risk measures over iterations
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(range(len(risk_budget_results['CoVaR_history'])), 
             risk_budget_results['CoVaR_history'], 'b-', marker='o')
    plt.title('CoVaR Over Risk Budgeting Iterations')
    plt.xlabel('Iteration')
    plt.ylabel('CoVaR')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(range(len(risk_budget_results['CoCVaR_history'])), 
             risk_budget_results['CoCVaR_history'], 'r-', marker='o')
    plt.title('CoCVaR Over Risk Budgeting Iterations')
    plt.xlabel('Iteration')
    plt.ylabel('CoCVaR')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('risk_budgeting.png')
    plt.close()
    
    # Print summary of risk budgeting results
    print(f"Initial CoVaR: {risk_budget_results['initial_CoVaR']:.6f}")
    print(f"Final CoVaR: {risk_budget_results['final_CoVaR']:.6f}")
    print(f"Improvement: {(risk_budget_results['initial_CoVaR'] - risk_budget_results['final_CoVaR']) / risk_budget_results['initial_CoVaR'] * 100:.2f}%")
    
    print(f"Initial CoCVaR: {risk_budget_results['initial_CoCVaR']:.6f}")
    print(f"Final CoCVaR: {risk_budget_results['final_CoCVaR']:.6f}")
    print(f"Improvement: {(risk_budget_results['initial_CoCVaR'] - risk_budget_results['final_CoCVaR']) / risk_budget_results['initial_CoCVaR'] * 100:.2f}%")
    
    print(f"Initial Return: {risk_budget_results['initial_return']:.6f}")
    print(f"Final Return: {risk_budget_results['final_return']:.6f}")
    
    # Compare weight changes for top contributors
    initial_weights = risk_budget_results['initial_weights']
    final_weights = risk_budget_results['final_weights']
    weight_changes = final_weights - initial_weights
    
    # Sort by absolute weight change
    sorted_weight_indices = np.argsort(np.abs(weight_changes))[::-1]
    top_changed = sorted_weight_indices[:10]
    
    plt.figure(figsize=(12, 6))
    x = np.arange(len(top_changed))
    width = 0.35
    
    plt.bar(x - width/2, [initial_weights[i] for i in top_changed], width, label='Initial Weight', alpha=0.7)
    plt.bar(x + width/2, [final_weights[i] for i in top_changed], width, label='Final Weight', alpha=0.7)
    
    plt.xlabel('Stock')
    plt.ylabel('Weight')
    plt.title('Weight Changes After Risk Budgeting')
    plt.xticks(x, [f'Stock_{i+1}' for i in top_changed])
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('weight_changes.png')
    plt.close()
    
    print("\nAnalysis complete. Check the generated charts for visualization.")

if __name__ == "__main__":
    main()