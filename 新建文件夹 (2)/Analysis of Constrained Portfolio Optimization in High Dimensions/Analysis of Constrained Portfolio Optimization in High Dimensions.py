import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, Lasso
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf

# Set random seed for reproducibility
np.random.seed(42)

class ResidualNodewiseRegressionPortfolio:
    """
    Implementation of the Residual Nodewise Regression for constrained portfolio optimization
    as proposed in the paper.
    """
    
    def __init__(self, K=3, lambda_=0.01):
        """
        Initialize the portfolio optimizer.
        
        Parameters:
        -----------
        K : int
            Number of factors in the factor model
        lambda_ : float
            Regularization parameter for Lasso regression
        """
        self.K = K
        self.lambda_ = lambda_
        
    def fit(self, returns, benchmark=None):
        """
        Fit the model to the returns data.
        
        Parameters:
        -----------
        returns : numpy.ndarray
            Matrix of asset returns (p x T)
        benchmark : numpy.ndarray, optional
            Benchmark portfolio weights (p,)
        """
        self.p, self.T = returns.shape
        self.returns = returns
        self.benchmark = benchmark if benchmark is not None else np.ones(self.p) / self.p
        
        # Step 1: Extract factors using PCA
        self._extract_factors()
        
        # Step 2: Estimate factor loadings
        self._estimate_factor_loadings()
        
        # Step 3: Compute residuals
        self._compute_residuals()
        
        # Step 4: Perform residual nodewise regression
        self._perform_residual_nodewise_regression()
        
        # Step 5: Construct precision matrix estimate
        self._construct_precision_matrix()
        
        # Step 6: Estimate covariance matrix of returns
        self._estimate_covariance_matrix()
        
        return self
    
    def _extract_factors(self):
        """Extract factors using PCA."""
        cov_matrix = np.cov(self.returns)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        # Sort eigenvalues and eigenvectors in decreasing order
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Select top K eigenvectors as factors
        self.factor_loadings = eigenvectors[:, :self.K]
        
        # Compute factors
        self.factors = self.factor_loadings.T @ self.returns
        
    def _estimate_factor_loadings(self):
        """Estimate factor loadings using OLS."""
        self.B_hat = np.zeros((self.p, self.K))
        
        for j in range(self.p):
            # OLS regression for each asset
            model = LinearRegression(fit_intercept=False)
            model.fit(self.factors.T, self.returns[j])
            self.B_hat[j] = model.coef_
    
    def _compute_residuals(self):
        """Compute residuals from factor model."""
        self.residuals = self.returns - self.B_hat @ self.factors
    
    def _perform_residual_nodewise_regression(self):
        """
        Perform residual nodewise regression to estimate precision matrix.
        This implements equation (1) from the paper.
        """
        self.beta_hat = np.zeros((self.p, self.p - 1))
        self.tau_squared = np.zeros(self.p)
        
        for j in range(self.p):
            # Extract residuals for asset j
            u_j = self.residuals[j]
            
            # Create matrix of other residuals
            indices = np.arange(self.p) != j
            U_minus_j = self.residuals[indices].T  # T x (p-1)
            
            # Perform Lasso regression
            lasso = Lasso(alpha=self.lambda_, fit_intercept=False)
            lasso.fit(U_minus_j, u_j)
            
            # Store coefficients
            self.beta_hat[j] = lasso.coef_
            
            # Compute tau_squared
            pred = U_minus_j @ lasso.coef_
            self.tau_squared[j] = np.sum((u_j - pred) ** 2) / self.T
    
    def _construct_precision_matrix(self):
        """
        Construct precision matrix estimate from nodewise regression results.
        """
        self.Omega_hat = np.zeros((self.p, self.p))
        
        for j in range(self.p):
            # Create C_j vector
            C_j = np.zeros(self.p)
            C_j[j] = 1
            
            indices = np.arange(self.p) != j
            C_j[indices] = -self.beta_hat[j]
            
            # Compute j-th row of precision matrix
            self.Omega_hat[j] = C_j / self.tau_squared[j]
        
        # Make the precision matrix symmetric
        self.Omega_hat_sym = (self.Omega_hat + self.Omega_hat.T) / 2
    
    def _estimate_covariance_matrix(self):
        """
        Estimate covariance matrix of returns using the Sherman-Morrison-Woodbury formula.
        """
        # Estimate covariance matrix of factors
        self.cov_factors = np.cov(self.factors)
        
        # Compute precision matrix of returns using Sherman-Morrison-Woodbury formula
        term1 = self.Omega_hat_sym
        term2 = self.Omega_hat_sym @ self.B_hat
        term3 = np.linalg.inv(np.linalg.inv(self.cov_factors) + self.B_hat.T @ self.Omega_hat_sym @ self.B_hat)
        term4 = self.B_hat.T @ self.Omega_hat_sym
        
        self.Theta_hat = term1 - term2 @ term3 @ term4
        
        # Covariance matrix is the inverse of precision matrix
        try:
            self.Sigma_hat = np.linalg.inv(self.Theta_hat)
        except np.linalg.LinAlgError:
            # If matrix inversion fails, use Ledoit-Wolf estimator as fallback
            lw = LedoitWolf().fit(self.returns.T)
            self.Sigma_hat = lw.covariance_
    
    def optimize_tracking_error_portfolio(self, theta=1.0, tracking_error_target=0.1):
        """
        Optimize portfolio with tracking error constraints.
        This implements equation (3) from the paper.
        
        Parameters:
        -----------
        theta : float
            Risk tolerance parameter
        tracking_error_target : float
            Target tracking error
            
        Returns:
        --------
        dict : Dictionary containing portfolio weights and metrics
        """
        # Estimate expected returns
        expected_returns = np.mean(self.returns, axis=1)
        
        # Compute optimal portfolio weights with tracking error constraint
        precision = self.Theta_hat
        
        # Portfolio that maximizes Sharpe ratio
        numerator = precision @ expected_returns
        denominator = np.ones(self.p).T @ precision @ expected_returns
        max_sharpe_portfolio = numerator / denominator
        
        # Global minimum variance portfolio
        numerator_gmv = precision @ np.ones(self.p)
        denominator_gmv = np.ones(self.p).T @ precision @ np.ones(self.p)
        gmv_portfolio = numerator_gmv / denominator_gmv
        
        # Compute the difference portfolio (wd = w - benchmark)
        w_d = theta * (max_sharpe_portfolio - gmv_portfolio)
        
        # Final portfolio
        w = w_d + self.benchmark
        
        # Calculate tracking error
        tracking_error = np.sqrt(w_d.T @ self.Sigma_hat @ w_d)
        
        # Calculate portfolio return and risk
        portfolio_return = w.T @ expected_returns
        portfolio_risk = np.sqrt(w.T @ self.Sigma_hat @ w)
        sharpe_ratio = portfolio_return / portfolio_risk
        
        return {
            'weights': w,
            'difference_weights': w_d,
            'tracking_error': tracking_error,
            'return': portfolio_return,
            'risk': portfolio_risk,
            'sharpe_ratio': sharpe_ratio
        }
    
    def optimize_weight_constrained_portfolio(self, restricted_indices, target_weight, theta=1.0):
        """
        Optimize portfolio with weight constraints.
        This implements equation (7) from the paper.
        
        Parameters:
        -----------
        restricted_indices : list
            List of indices of restricted assets
        target_weight : float
            Target weight for restricted assets
        theta : float
            Risk tolerance parameter
            
        Returns:
        --------
        dict : Dictionary containing portfolio weights and metrics
        """
        # Estimate expected returns
        expected_returns = np.mean(self.returns, axis=1)
        
        # Compute precision matrix
        precision = self.Theta_hat
        
        # Create restriction vector
        R = np.zeros(self.p)
        R[restricted_indices] = 1
        
        # Compute k and a vectors
        numerator_k = precision @ R
        denominator_k = np.ones(self.p).T @ precision @ R
        k = numerator_k / denominator_k
        
        numerator_a = precision @ np.ones(self.p)
        denominator_a = np.ones(self.p).T @ precision @ np.ones(self.p)
        a = numerator_a / denominator_a
        
        # Compute wu scalar
        wu = R.T @ (precision @ expected_returns) / (np.ones(self.p).T @ precision @ expected_returns)
        wu -= R.T @ (precision @ np.ones(self.p)) / (np.ones(self.p).T @ precision @ np.ones(self.p))
        
        # Compute wk and wa scalars
        wk = R.T @ precision @ R / (R.T @ precision @ np.ones(self.p))
        wa = R.T @ precision @ np.ones(self.p) / (np.ones(self.p).T @ precision @ np.ones(self.p))
        
        # Compute l vector
        l = (k - a) / (wk - wa)
        
        # Compute tracking error constrained portfolio
        numerator_te = precision @ expected_returns
        denominator_te = np.ones(self.p).T @ precision @ expected_returns
        max_sharpe_portfolio = numerator_te / denominator_te
        
        numerator_gmv = precision @ np.ones(self.p)
        denominator_gmv = np.ones(self.p).T @ precision @ np.ones(self.p)
        gmv_portfolio = numerator_gmv / denominator_gmv
        
        w_d = theta * (max_sharpe_portfolio - gmv_portfolio)
        
        # Compute constrained portfolio
        w_cp = (target_weight - theta * wu) * l + w_d
        
        # Calculate portfolio metrics
        portfolio_return = w_cp.T @ expected_returns
        portfolio_risk = np.sqrt(w_cp.T @ self.Sigma_hat @ w_cp)
        sharpe_ratio = portfolio_return / portfolio_risk
        
        # Calculate weight on restricted assets
        restricted_weight = np.sum(w_cp[restricted_indices])
        
        return {
            'weights': w_cp,
            'return': portfolio_return,
            'risk': portfolio_risk,
            'sharpe_ratio': sharpe_ratio,
            'restricted_weight': restricted_weight
        }
    
    def optimize_combined_constraints(self, restricted_indices, target_weight, theta=1.0):
        """
        Optimize portfolio with both tracking error and weight constraints.
        This implements the combined approach from Sections 4 and 5 of the paper.
        
        Parameters:
        -----------
        restricted_indices : list
            List of indices of restricted assets
        target_weight : float
            Target weight for restricted assets
        theta : float
            Risk tolerance parameter
            
        Returns:
        --------
        dict : Dictionary containing portfolio weights and metrics
        """
        # First compute tracking error constrained portfolio
        te_portfolio = self.optimize_tracking_error_portfolio(theta)
        
        # Check if weight constraint is binding
        wu = 0
        for idx in restricted_indices:
            wu += te_portfolio['weights'][idx]
        
        # If constraint is not binding, return tracking error portfolio
        if wu <= target_weight:
            return {**te_portfolio, 'binding': False}
        
        # If constraint is binding, compute constrained portfolio
        weight_portfolio = self.optimize_weight_constrained_portfolio(
            restricted_indices, target_weight, theta)
        
        return {**weight_portfolio, 'binding': True}

# Simulate data according to the paper's specifications
def simulate_data(p=80, T=100, K=3):
    """
    Simulate asset returns data based on the factor model described in the paper.
    
    Parameters:
    -----------
    p : int
        Number of assets
    T : int
        Number of time periods
    K : int
        Number of factors
    
    Returns:
    --------
    tuple : (returns, factors, loadings, errors)
    """
    # Generate factor returns
    factor_means = np.array([0.005, 0.005, 0.005])
    factor_ar = np.array([0.03, -0.05, -0.05])
    
    factors = np.zeros((K, T))
    factors[:, 0] = 0  # Initial values
    
    for t in range(1, T):
        for i in range(K):
            error = np.random.normal(0, np.sqrt(1 - factor_ar[i]**2))
            factors[i, t] = factor_means[i] + factor_ar[i] * factors[i, t-1] + error
    
    # Generate factor loadings
    loading_means = np.array([-0.1, 0.1, 0.1])
    loadings = np.zeros((p, K))
    
    for j in range(p):
        for k in range(K):
            loadings[j, k] = np.random.normal(loading_means[k], 1)
    
    # Generate error covariance matrix with Toeplitz structure
    error_cov = np.zeros((p, p))
    for i in range(p):
        for j in range(p):
            error_cov[i, j] = 0.25 ** abs(i-j)
    
    # Generate errors
    errors = np.random.multivariate_normal(np.zeros(p), error_cov, T).T
    
    # Generate returns
    returns = loadings @ factors + errors
    
    return returns, factors, loadings, errors

# Function to evaluate portfolio performance
def evaluate_portfolios(returns, portfolios, benchmark=None):
    """
    Evaluate the performance of different portfolios.
    
    Parameters:
    -----------
    returns : numpy.ndarray
        Matrix of asset returns (p x T)
    portfolios : dict
        Dictionary of portfolio weights
    benchmark : numpy.ndarray, optional
        Benchmark portfolio weights
    
    Returns:
    --------
    pandas.DataFrame : Performance metrics
    """
    p, T = returns.shape
    
    # Calculate benchmark performance if provided
    if benchmark is not None:
        benchmark_return = benchmark @ np.mean(returns, axis=1)
        benchmark_risk = np.sqrt(benchmark @ np.cov(returns) @ benchmark)
        benchmark_sharpe = benchmark_return / benchmark_risk
    else:
        benchmark_return = np.nan
        benchmark_risk = np.nan
        benchmark_sharpe = np.nan
    
    # Calculate performance for each portfolio
    results = {
        'Portfolio': ['Benchmark'] + list(portfolios.keys()),
        'Return': [benchmark_return] + [port['return'] for port in portfolios.values()],
        'Risk': [benchmark_risk] + [port['risk'] for port in portfolios.values()],
        'Sharpe Ratio': [benchmark_sharpe] + [port['sharpe_ratio'] for port in portfolios.values()]
    }
    
    # Add tracking error if available
    if 'tracking_error' in portfolios[list(portfolios.keys())[0]]:
        results['Tracking Error'] = [np.nan] + [port.get('tracking_error', np.nan) for port in portfolios.values()]
    
    # Add restricted weight if available
    if 'restricted_weight' in portfolios[list(portfolios.keys())[0]]:
        results['Restricted Weight'] = [np.nan] + [port.get('restricted_weight', np.nan) for port in portfolios.values()]
    
    return pd.DataFrame(results)

# Main simulation and testing
def main():
    # Parameters
    p_values = [80, 120]  # Number of assets
    T_values = [100, 150]  # Number of time periods
    K = 3  # Number of factors
    
    for p in p_values:
        for T in T_values:
            print(f"\n===== Simulation with p={p}, T={T} =====")
            
            # Simulate data
            returns, factors, loadings, errors = simulate_data(p, T, K)
            
            # Create equal-weighted benchmark
            benchmark = np.ones(p) / p
            
            # Fit model
            rnr = ResidualNodewiseRegressionPortfolio(K=K, lambda_=0.01)
            rnr.fit(returns, benchmark)
            
            # 1. Test tracking error constrained portfolio
            portfolios = {}
            for te in [0.1, 0.2, 0.3]:
                print(f"Optimizing portfolio with tracking error target = {te}")
                portfolios[f'TE={te}'] = rnr.optimize_tracking_error_portfolio(theta=1.0, tracking_error_target=te)
            
            # Evaluate tracking error portfolios
            print("\nTracking Error Portfolio Performance:")
            te_results = evaluate_portfolios(returns, portfolios, benchmark)
            print(te_results)
            
            # 2. Test weight constrained portfolio
            restricted_indices = list(range(10))  # First 10 assets are restricted
            portfolios = {}
            for target in [0.2, 0.3, 0.4]:
                print(f"\nOptimizing portfolio with weight constraint = {target}")
                portfolios[f'Weight={target}'] = rnr.optimize_weight_constrained_portfolio(
                    restricted_indices, target, theta=1.0)
            
            # Evaluate weight constrained portfolios
            print("\nWeight Constrained Portfolio Performance:")
            wc_results = evaluate_portfolios(returns, portfolios, benchmark)
            print(wc_results)
            
            # 3. Test combined constraints
            portfolios = {}
            for te in [0.1, 0.2, 0.3]:
                for target in [0.2, 0.3, 0.4]:
                    print(f"\nOptimizing portfolio with TE={te}, Weight={target}")
                    portfolios[f'TE={te},W={target}'] = rnr.optimize_combined_constraints(
                        restricted_indices, target, theta=1.0)
            
            # Evaluate combined constraint portfolios
            print("\nCombined Constraint Portfolio Performance:")
            cc_results = evaluate_portfolios(returns, portfolios, benchmark)
            print(cc_results)
            
            # Plot weight distribution for different portfolios
            plt.figure(figsize=(12, 8))
            
            # Plot tracking error portfolio weights
            te_portfolio = rnr.optimize_tracking_error_portfolio(theta=1.0, tracking_error_target=0.2)
            plt.subplot(2, 2, 1)
            plt.bar(range(p), te_portfolio['weights'])
            plt.title(f'Tracking Error Portfolio (TE=0.2)\nSharpe: {te_portfolio["sharpe_ratio"]:.4f}, TE: {te_portfolio["tracking_error"]:.4f}')
            plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
            
            # Plot weight constrained portfolio weights
            wc_portfolio = rnr.optimize_weight_constrained_portfolio(restricted_indices, 0.3, theta=1.0)
            plt.subplot(2, 2, 2)
            plt.bar(range(p), wc_portfolio['weights'])
            plt.title(f'Weight Constrained Portfolio (W=0.3)\nSharpe: {wc_portfolio["sharpe_ratio"]:.4f}, Restricted: {wc_portfolio["restricted_weight"]:.4f}')
            plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
            
            # Highlight restricted assets
            plt.axvspan(-0.5, 9.5, alpha=0.2, color='yellow')
            
            # Plot combined constraint portfolio weights
            cc_portfolio = rnr.optimize_combined_constraints(restricted_indices, 0.3, theta=1.0)
            plt.subplot(2, 2, 3)
            plt.bar(range(p), cc_portfolio['weights'])
            binding_status = "Binding" if cc_portfolio.get('binding', False) else "Non-binding"
            plt.title(f'Combined Constraints Portfolio (TE=0.2, W=0.3, {binding_status})\nSharpe: {cc_portfolio["sharpe_ratio"]:.4f}')
            plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
            
            # Highlight restricted assets
            plt.axvspan(-0.5, 9.5, alpha=0.2, color='yellow')
            
            # Plot benchmark weights
            plt.subplot(2, 2, 4)
            plt.bar(range(p), benchmark)
            plt.title('Benchmark Portfolio (Equal-Weighted)')
            
            plt.tight_layout()
            plt.savefig(f'portfolio_weights_p{p}_T{T}.png')
            
            # Plot weight distributions as histograms
            plt.figure(figsize=(12, 8))
            
            plt.subplot(2, 2, 1)
            sns.histplot(te_portfolio['weights'], kde=True)
            plt.title(f'Tracking Error Portfolio Weight Distribution\nSharpe: {te_portfolio["sharpe_ratio"]:.4f}')
            
            plt.subplot(2, 2, 2)
            sns.histplot(wc_portfolio['weights'], kde=True)
            plt.title(f'Weight Constrained Portfolio Weight Distribution\nSharpe: {wc_portfolio["sharpe_ratio"]:.4f}')
            
            plt.subplot(2, 2, 3)
            sns.histplot(cc_portfolio['weights'], kde=True)
            plt.title(f'Combined Constraints Portfolio Weight Distribution\nSharpe: {cc_portfolio["sharpe_ratio"]:.4f}')
            
            plt.subplot(2, 2, 4)
            sns.histplot(benchmark, kde=True)
            plt.title('Benchmark Portfolio Weight Distribution')
            
            plt.tight_layout()
            plt.savefig(f'weight_distributions_p{p}_T{T}.png')

if __name__ == "__main__":
    main()

