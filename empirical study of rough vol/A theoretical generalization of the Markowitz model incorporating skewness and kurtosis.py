import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import minimize
import seaborn as sns
from sklearn.covariance import LedoitWolf

# Set random seed for reproducibility
np.random.seed(42)

class FourMomentOptimizer:
    """
    Portfolio optimizer that incorporates expected return, variance, skewness, and kurtosis
    based on Uberti's generalization of the Markowitz model.
    """
    
    def __init__(self, returns):
        """
        Initialize the optimizer with historical returns data.
        
        Parameters:
        -----------
        returns : pandas DataFrame
            Historical asset returns with assets as columns and time as rows
        """
        self.returns = returns
        self.n_assets = returns.shape[1]
        
        # Estimate return distribution parameters
        self.estimate_parameters()
        
    def estimate_parameters(self):
        """Estimate the parameters needed for the model, including mu, D, b, and t vectors."""
        # First, we'll generate the y and z variables as described in the paper
        T = self.returns.shape[0]
        
        # Generate y (skew normal distribution) and z (t-distribution)
        # y is for skewness, z is for kurtosis
        a = -1  # skewness parameter
        y = stats.skewnorm.rvs(a, size=T)
        y = (y - np.mean(y)) / np.std(y)  # standardize
        
        df = 5  # degrees of freedom for t-distribution
        z = stats.t.rvs(df, size=T)
        z = (z - np.mean(z)) / np.std(z)  # standardize
        
        # Store properties of y and z for reference
        self.y_skew = stats.skew(y)
        self.z_kurt = stats.kurtosis(z, fisher=False)  # Use fisher=False to get the actual kurtosis, not excess kurtosis
        
        # Estimate μ
        self.mu = self.returns.mean().values
        
        # Estimate D (covariance matrix) using robust estimation
        lw = LedoitWolf()
        lw.fit(self.returns)
        self.D = lw.covariance_
        
        # Estimate b and t vectors using regression
        # First, we need to regress returns on y and z
        X = np.column_stack([y, z])
        
        # For each asset, estimate b and t
        self.b = np.zeros(self.n_assets)
        self.t = np.zeros(self.n_assets)
        
        for i in range(self.n_assets):
            # Regress returns of asset i on y and z
            beta, _, _, _ = np.linalg.lstsq(X, self.returns.iloc[:, i].values, rcond=None)
            self.b[i] = beta[0]  # coefficient for y (skewness)
            self.t[i] = beta[1]  # coefficient for z (kurtosis)
        
        # Calculate the residuals for estimation of the covariance matrix C
        residuals = np.zeros_like(self.returns)
        for i in range(self.n_assets):
            residuals[:, i] = self.returns.iloc[:, i].values - (self.mu[i] + self.b[i]*y + self.t[i]*z)
        
        # Estimate C (covariance matrix of residuals)
        self.C = np.cov(residuals, rowvar=False)
        
    def calculate_P_matrix(self):
        """Calculate the P matrix as defined in the paper."""
        # Create the M matrix
        ones = np.ones(self.n_assets)
        M = np.column_stack([self.mu, ones, self.b, self.t])
        
        # Calculate D^-1
        D_inv = np.linalg.inv(self.D)
        
        # Calculate P = M^T D^-1 M
        P = M.T @ D_inv @ M
        
        return P, D_inv, M
        
    def _objective_function(self, weights):
        """
        Objective function: minimize negative kurtosis
        """
        kurtosis = weights @ self.t * self.z_kurt
        return -kurtosis  # maximize kurtosis = minimize negative kurtosis
    
    def _expected_return_constraint(self, weights, target_return):
        """Constraint: expected return equals target return"""
        return weights @ self.mu - target_return
    
    def _variance_constraint(self, weights, target_variance):
        """Constraint: portfolio variance equals target variance"""
        return weights @ self.D @ weights - target_variance
    
    def _skewness_constraint(self, weights, target_skewness):
        """Constraint: portfolio skewness equals target skewness"""
        return weights @ self.b * self.y_skew - target_skewness
    
    def _budget_constraint(self, weights):
        """Constraint: weights sum to 1"""
        return np.sum(weights) - 1.0
    
    def optimize_portfolio_numerical(self, target_return, target_variance, target_skewness):
        """
        Optimize portfolio using numerical optimization.
        
        Parameters:
        -----------
        target_return : float
            Target portfolio expected return
        target_variance : float
            Target portfolio variance
        target_skewness : float
            Target portfolio skewness
            
        Returns:
        --------
        weights : numpy array
            Optimal portfolio weights
        """
        # Initial guess (equal weight portfolio)
        initial_weights = np.ones(self.n_assets) / self.n_assets
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda weights: self._expected_return_constraint(weights, target_return)},
            {'type': 'eq', 'fun': lambda weights: self._variance_constraint(weights, target_variance)},
            {'type': 'eq', 'fun': lambda weights: self._skewness_constraint(weights, target_skewness)},
            {'type': 'eq', 'fun': lambda weights: self._budget_constraint(weights)}
        ]
        
        # Bounds (allow short selling for demonstration purposes)
        bounds = [(-1, 1) for _ in range(self.n_assets)]
        
        # Optimize
        result = minimize(
            self._objective_function,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'disp': False, 'maxiter': 1000}
        )
        
        if not result.success:
            print(f"Warning: Optimization did not converge: {result.message}")
        
        return result.x
    
    def optimize_portfolio(self, target_return, target_variance=None, target_skewness=None):
        """
        Optimize portfolio using the closed-form solution.
        
        Parameters:
        -----------
        target_return : float
            Target portfolio expected return
        target_variance : float, optional
            Target portfolio variance. If None, will use the minimum variance plus some excess.
        target_skewness : float, optional
            Target portfolio skewness. If None, will use a positive value.
            
        Returns:
        --------
        weights : numpy array
            Optimal portfolio weights
        """
        # Calculate matrices needed for the solution
        P, D_inv, M = self.calculate_P_matrix()
        
        # Extract submatrices
        P2 = P[:3, :3]
        A = P[:2, :2]
        psi = P[3, :3]
        
        # Calculate determinants for later use
        det_P = np.linalg.det(P)
        det_P2 = np.linalg.det(P2)
        det_A = np.linalg.det(A)
        
        # Invert P2
        P2_inv = np.linalg.inv(P2)
        
        # Calculate H
        H = psi @ P2_inv @ psi
        
        # Calculate s - H
        s = P[3, 3]
        s_minus_H = s - H
        
        if s_minus_H <= 0:
            raise ValueError("s - H is not positive, which violates a required condition.")
        
        # Determine minimum variance (sigma_A^2)
        ones = np.ones(self.n_assets)
        mu_target = np.array([target_return, 1.0])
        
        A_inv = np.linalg.inv(A)
        sigma_A_squared = mu_target @ A_inv @ mu_target
        
        # Set default target variance and skewness if not provided
        if target_variance is None:
            # Use minimum variance plus some excess
            target_variance = sigma_A_squared * 1.2  # 20% more than minimum variance
        
        if target_skewness is None:
            target_skewness = 0.1  # Positive skewness
        
        # Calculate beta
        beta = np.array([target_return, 1.0, target_skewness / self.y_skew])
        
        # Calculate sigma_P2_squared
        sigma_P2_squared = beta @ P2_inv @ beta
        
        # Check conditions
        if target_variance < sigma_P2_squared:
            raise ValueError("Target variance is too low to achieve the desired return and skewness.")
        
        # Calculate optimal weights using the formula from the paper
        x_mv = D_inv @ np.column_stack([self.mu, ones]) @ A_inv @ np.array([target_return, 1.0])
        
        # Calculate the second term
        f = P[0, 2]
        g = P[1, 2]
        e = P[2, 2]
        h = np.array([f, g]) @ A_inv @ np.array([f, g])
        e_minus_h = det_P2 / det_A
        
        x_sk = (D_inv @ self.b - D_inv @ np.column_stack([self.mu, ones]) @ A_inv @ np.array([f, g])) * np.sqrt((sigma_P2_squared - sigma_A_squared) / e_minus_h)
        
        # Calculate the third term
        p = P[0, 3]
        q = P[1, 3]
        
        x_k = (D_inv @ self.t - D_inv @ np.column_stack([self.mu, ones]) @ A_inv @ np.array([p, q])) * np.sqrt((target_variance - sigma_P2_squared) / s_minus_H)
        
        # Combine the three portfolios
        x_optimal = x_mv + x_sk + x_k
        
        return x_optimal, x_mv, x_sk, x_k
    
    def calculate_portfolio_moments(self, weights):
        """
        Calculate the first four moments of a portfolio with given weights.
        
        Parameters:
        -----------
        weights : numpy array
            Portfolio weights
            
        Returns:
        --------
        moments : tuple
            (expected return, variance, skewness, kurtosis)
        """
        expected_return = weights @ self.mu
        variance = weights @ self.D @ weights
        skewness = weights @ self.b * self.y_skew
        kurtosis = weights @ self.t * self.z_kurt
        
        return expected_return, variance, skewness, kurtosis
    
    def plot_portfolio_decomposition(self, weights_optimal, weights_mv, weights_sk, weights_k):
        """
        Visualize the decomposition of the optimal portfolio into MV, skewness, and kurtosis components.
        
        Parameters:
        -----------
        weights_optimal : numpy array
            Optimal portfolio weights
        weights_mv : numpy array
            Mean-variance portfolio weights
        weights_sk : numpy array
            Skewness portfolio weights
        weights_k : numpy array
            Kurtosis portfolio weights
        """
        assets = self.returns.columns
        
        # Create a DataFrame for better visualization
        df = pd.DataFrame({
            'Optimal': weights_optimal,
            'MV': weights_mv,
            'Skewness': weights_sk,
            'Kurtosis': weights_k
        }, index=assets)
        
        # Plot
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot optimal portfolio
        df['Optimal'].plot(kind='bar', ax=axes[0, 0], color='blue', alpha=0.7)
        axes[0, 0].set_title('Optimal Portfolio Weights')
        axes[0, 0].set_ylabel('Weight')
        axes[0, 0].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # Plot MV portfolio
        df['MV'].plot(kind='bar', ax=axes[0, 1], color='green', alpha=0.7)
        axes[0, 1].set_title('Mean-Variance Portfolio Weights')
        axes[0, 1].set_ylabel('Weight')
        axes[0, 1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # Plot skewness component
        df['Skewness'].plot(kind='bar', ax=axes[1, 0], color='purple', alpha=0.7)
        axes[1, 0].set_title('Skewness Component Weights')
        axes[1, 0].set_ylabel('Weight')
        axes[1, 0].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # Plot kurtosis component
        df['Kurtosis'].plot(kind='bar', ax=axes[1, 1], color='red', alpha=0.7)
        axes[1, 1].set_title('Kurtosis Component Weights')
        axes[1, 1].set_ylabel('Weight')
        axes[1, 1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Calculate portfolio moments
        moments_optimal = self.calculate_portfolio_moments(weights_optimal)
        moments_mv = self.calculate_portfolio_moments(weights_mv)
        
        print("Portfolio Moments Comparison:")
        print(f"Optimal Portfolio: Return = {moments_optimal[0]:.4f}, Variance = {moments_optimal[1]:.4f}, Skewness = {moments_optimal[2]:.4f}, Kurtosis = {moments_optimal[3]:.4f}")
        print(f"MV Portfolio: Return = {moments_mv[0]:.4f}, Variance = {moments_mv[1]:.4f}, Skewness = {moments_mv[2]:.4f}, Kurtosis = {moments_mv[3]:.4f}")
        
        # Verify that skewness and kurtosis components are self-financing
        print("\nSelf-financing verification:")
        print(f"Sum of Skewness Component Weights: {np.sum(weights_sk):.8f}")
        print(f"Sum of Kurtosis Component Weights: {np.sum(weights_k):.8f}")
        
        # Verify that skewness and kurtosis components don't affect expected return
        print("\nReturn neutrality verification:")
        print(f"Expected Return of Skewness Component: {weights_sk @ self.mu:.8f}")
        print(f"Expected Return of Kurtosis Component: {weights_k @ self.mu:.8f}")

def simulate_non_normal_returns(n_assets=10, n_days=1000, seed=42):
    """
    Simulate returns for multiple assets with non-normal distributions.
    
    Parameters:
    -----------
    n_assets : int
        Number of assets
    n_days : int
        Number of days
    seed : int
        Random seed
        
    Returns:
    --------
    returns : pandas DataFrame
        Simulated returns
    """
    np.random.seed(seed)
    
    # Create asset names
    asset_names = [f'Asset_{i+1}' for i in range(n_assets)]
    
    # Create correlation matrix (positive correlation with some randomness)
    correlation = np.random.uniform(0.1, 0.4, size=(n_assets, n_assets))
    np.fill_diagonal(correlation, 1.0)
    correlation = (correlation + correlation.T) / 2  # Make it symmetric
    
    # Create standard deviations (volatility)
    std_devs = np.random.uniform(0.01, 0.04, size=n_assets)
    
    # Create covariance matrix
    covariance = np.outer(std_devs, std_devs) * correlation
    
    # Generate normal returns
    normal_returns = np.random.multivariate_normal(
        mean=np.zeros(n_assets),
        cov=covariance,
        size=n_days
    )
    
    # Add non-normal components
    # Generate y (for skewness) and z (for kurtosis)
    y = stats.skewnorm.rvs(-2, size=n_days)  # Negative skew
    y = (y - np.mean(y)) / np.std(y)  # Standardize
    
    z = stats.t.rvs(5, size=n_days)  # t-distribution (fat tails)
    z = (z - np.mean(z)) / np.std(z)  # Standardize
    
    # Create b and t vectors (coefficients for skewness and kurtosis)
    b = np.random.uniform(-0.02, 0.02, size=n_assets)
    t = np.random.uniform(0.01, 0.03, size=n_assets)  # Positive for fat tails
    
    # Add trend (expected returns)
    mu = np.random.uniform(0.0002, 0.0006, size=n_assets)
    
    # Combine all components to generate returns
    returns = np.zeros((n_days, n_assets))
    for i in range(n_assets):
        returns[:, i] = mu[i] + normal_returns[:, i] + b[i] * y + t[i] * z
    
    return pd.DataFrame(returns, columns=asset_names)

def backtest_strategies(returns, window_size=252, rebalance_freq=63):
    """
    Backtest different portfolio strategies.
    
    Parameters:
    -----------
    returns : pandas DataFrame
        Historical returns
    window_size : int
        Size of the rolling window for parameter estimation
    rebalance_freq : int
        How often to rebalance the portfolio (in days)
        
    Returns:
    --------
    portfolio_values : pandas DataFrame
        Portfolio values for each strategy
    """
    n_days = returns.shape[0]
    n_assets = returns.shape[1]
    
    # Initialize portfolio values
    portfolio_values = pd.DataFrame(index=returns.index, columns=[
        'Equal Weight', 'Mean-Variance', 'Four Moments'
    ])
    portfolio_values.iloc[0] = 1.0  # Start with $1
    
    # Initialize weights
    weights = {
        'Equal Weight': np.ones(n_assets) / n_assets,
        'Mean-Variance': np.ones(n_assets) / n_assets,
        'Four Moments': np.ones(n_assets) / n_assets
    }
    
    # For each rebalancing period
    rebalance_points = range(window_size, n_days, rebalance_freq)
    
    for t in rebalance_points:
        # Use data from the rolling window for parameter estimation
        window_returns = returns.iloc[t-window_size:t]
        
        # Update equal weight portfolio (no change needed, just for consistency)
        weights['Equal Weight'] = np.ones(n_assets) / n_assets
        
        # Update Mean-Variance portfolio
        optimizer = FourMomentOptimizer(window_returns)
        
        # Calculate target return (average of asset returns)
        target_return = window_returns.mean().mean()
        
        try:
            # Optimize Mean-Variance portfolio
            # We'll use our four moment optimizer but set skewness to 0 and use minimum variance
            A, D_inv, _ = optimizer.calculate_P_matrix()
            A_inv = np.linalg.inv(A[:2, :2])
            mu_target = np.array([target_return, 1.0])
            sigma_A_squared = mu_target @ A_inv @ mu_target
            
            weights_mv = D_inv @ np.column_stack([optimizer.mu, np.ones(n_assets)]) @ A_inv @ np.array([target_return, 1.0])
            weights['Mean-Variance'] = weights_mv
            
            # Optimize Four Moments portfolio
            weights_optimal, _, _, _ = optimizer.optimize_portfolio(
                target_return,
                target_variance=sigma_A_squared * 1.2,  # 20% more variance
                target_skewness=0.1  # Positive skewness
            )
            weights['Four Moments'] = weights_optimal
            
        except Exception as e:
            print(f"Error at time {t}: {e}")
            # Keep previous weights if optimization fails
            pass
        
        # Apply weights to the next rebalance_freq days
        for i in range(t, min(t + rebalance_freq, n_days)):
            daily_return = returns.iloc[i]
            
            for strategy, weight in weights.items():
                portfolio_return = np.sum(weight * daily_return)
                portfolio_values.loc[returns.index[i], strategy] = portfolio_values.loc[returns.index[i-1], strategy] * (1 + portfolio_return)
    
    return portfolio_values

def plot_backtest_results(portfolio_values):
    """
    Plot the results of the backtest.
    
    Parameters:
    -----------
    portfolio_values : pandas DataFrame
        Portfolio values for each strategy
    """
    plt.figure(figsize=(12, 6))
    
    for strategy in portfolio_values.columns:
        plt.plot(portfolio_values.index, portfolio_values[strategy], label=strategy)
    
    plt.title('Portfolio Performance Comparison')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Calculate performance metrics
    returns = portfolio_values.pct_change().dropna()
    
    # Annualized return
    annual_returns = returns.mean() * 252
    
    # Annualized volatility
    annual_vol = returns.std() * np.sqrt(252)
    
    # Sharpe ratio (assuming risk-free rate = 0 for simplicity)
    sharpe_ratios = annual_returns / annual_vol
    
    # Maximum drawdown
    cumulative_returns = (1 + returns).cumprod()
    previous_peaks = cumulative_returns.cummax()
    drawdowns = (cumulative_returns - previous_peaks) / previous_peaks
    max_drawdowns = drawdowns.min()
    
    # Skewness and kurtosis of returns
    skewness = returns.skew()
    kurtosis = returns.kurtosis()
    
    # Print metrics
    metrics = pd.DataFrame({
        'Annual Return': annual_returns,
        'Annual Volatility': annual_vol,
        'Sharpe Ratio': sharpe_ratios,
        'Maximum Drawdown': max_drawdowns,
        'Skewness': skewness,
        'Kurtosis': kurtosis
    })
    
    print("Performance Metrics:")
    print(metrics)
    
    plt.show()

def plot_efficient_frontier_comparison(returns, n_points=20):
    """
    Plot the efficient frontier for both Mean-Variance and Four Moments optimization.
    
    Parameters:
    -----------
    returns : pandas DataFrame
        Historical returns
    n_points : int
        Number of points on the efficient frontier
    """
    optimizer = FourMomentOptimizer(returns)
    
    # Calculate minimum and maximum returns
    min_return = min(optimizer.mu)
    max_return = max(optimizer.mu)
    
    # Generate return targets between min and max
    return_targets = np.linspace(min_return, max_return, n_points)
    
    # Initialize arrays to store results
    mv_variance = np.zeros(n_points)
    mv_skewness = np.zeros(n_points)
    mv_kurtosis = np.zeros(n_points)
    
    fm_variance = np.zeros(n_points)
    fm_skewness = np.zeros(n_points)
    fm_kurtosis = np.zeros(n_points)
    
    # Calculate efficient frontiers
    for i, target_return in enumerate(return_targets):
        try:
            # Calculate minimum variance for the target return
            A, D_inv, _ = optimizer.calculate_P_matrix()
            A_inv = np.linalg.inv(A[:2, :2])
            mu_target = np.array([target_return, 1.0])
            sigma_A_squared = mu_target @ A_inv @ mu_target
            
            # Mean-Variance optimization
            weights_mv = D_inv @ np.column_stack([optimizer.mu, np.ones(optimizer.n_assets)]) @ A_inv @ np.array([target_return, 1.0])
            mv_moments = optimizer.calculate_portfolio_moments(weights_mv)
            mv_variance[i] = mv_moments[1]
            mv_skewness[i] = mv_moments[2]
            mv_kurtosis[i] = mv_moments[3]
            
            # Four Moments optimization
            weights_fm, _, _, _ = optimizer.optimize_portfolio(
                target_return,
                target_variance=sigma_A_squared * 1.2,  # 20% more variance
                target_skewness=0.1  # Positive skewness
            )
            fm_moments = optimizer.calculate_portfolio_moments(weights_fm)
            fm_variance[i] = fm_moments[1]
            fm_skewness[i] = fm_moments[2]
            fm_kurtosis[i] = fm_moments[3]
            
        except Exception as e:
            print(f"Error at return target {target_return}: {e}")
            # If optimization fails, use the previous point
            if i > 0:
                mv_variance[i] = mv_variance[i-1]
                mv_skewness[i] = mv_skewness[i-1]
                mv_kurtosis[i] = mv_kurtosis[i-1]
                
                fm_variance[i] = fm_variance[i-1]
                fm_skewness[i] = fm_skewness[i-1]
                fm_kurtosis[i] = fm_kurtosis[i-1]
    
    # Plot the efficient frontiers
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Return vs Variance
    axes[0].plot(mv_variance, return_targets, 'g-o', label='Mean-Variance')
    axes[0].plot(fm_variance, return_targets, 'b-o', label='Four Moments')
    axes[0].set_xlabel('Variance')
    axes[0].set_ylabel('Expected Return')
    axes[0].set_title('Return vs Variance')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Return vs Skewness
    axes[1].plot(mv_skewness, return_targets, 'g-o', label='Mean-Variance')
    axes[1].plot(fm_skewness, return_targets, 'b-o', label='Four Moments')
    axes[1].set_xlabel('Skewness')
    axes[1].set_ylabel('Expected Return')
    axes[1].set_title('Return vs Skewness')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Return vs Kurtosis
    axes[2].plot(mv_kurtosis, return_targets, 'g-o', label='Mean-Variance')
    axes[2].plot(fm_kurtosis, return_targets, 'b-o', label='Four Moments')
    axes[2].set_xlabel('Kurtosis')
    axes[2].set_ylabel('Expected Return')
    axes[2].set_title('Return vs Kurtosis')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# Main execution
if __name__ == "__main__":
    # Simulate returns for 10 assets over 1000 days
    simulated_returns = simulate_non_normal_returns(n_assets=10, n_days=2000)
    
    # Print descriptive statistics
    print("Descriptive Statistics for Simulated Returns:")
    stats_df = pd.DataFrame({
        'Mean': simulated_returns.mean(),
        'Std Dev': simulated_returns.std(),
        'Skewness': simulated_returns.skew(),
        'Kurtosis': simulated_returns.kurtosis()
    })
    print(stats_df)
    
    # Initialize optimizer
    optimizer = FourMomentOptimizer(simulated_returns)
    
    # Set target return
    target_return = simulated_returns.mean().mean()
    
    # Get optimal portfolio
    weights_optimal, weights_mv, weights_sk, weights_k = optimizer.optimize_portfolio(target_return)
    
    # Plot decomposition
    optimizer.plot_portfolio_decomposition(weights_optimal, weights_mv, weights_sk, weights_k)
    
    # Plot efficient frontier comparison
    plot_efficient_frontier_comparison(simulated_returns)
    
    # Backtest strategies
    portfolio_values = backtest_strategies(simulated_returns)
    
    # Plot backtest results
    plot_backtest_results(portfolio_values)