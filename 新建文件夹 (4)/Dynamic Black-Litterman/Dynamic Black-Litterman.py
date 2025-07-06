import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import cholesky, solve_continuous_are
from scipy.integrate import solve_ivp

# Set random seed for reproducibility
np.random.seed(42)

class DynamicBlackLitterman:
    def __init__(self, mu, sigma, rf, gamma, T, P, omega):
        """
        Initialize the Dynamic Black-Litterman model.
        
        Parameters:
        - mu: Vector of expected returns (N x 1)
        - sigma: Covariance matrix of returns (N x N)
        - rf: Risk-free rate
        - gamma: Risk aversion parameter
        - T: Investment horizon
        - P: View matrix (K x N)
        - omega: View covariance matrix (K x K)
        """
        self.mu = mu
        self.sigma = sigma
        self.rf = rf
        self.gamma = gamma
        self.T = T
        self.P = P
        self.omega = omega
        
        # Compute mu_x (drift of log-returns process)
        self.mu_x = mu - 0.5 * np.diag(sigma)
        
        # Compute Cholesky decomposition of sigma
        self.L = cholesky(sigma, lower=True)
        
        # Dimensions
        self.N = len(mu)  # Number of assets
        self.K = P.shape[0]  # Number of views
        
    def beta_1(self, t=0):
        """Compute beta_1 coefficient at time t."""
        return (1/self.T) * self.sigma @ self.P.T @ np.linalg.inv(self.P @ self.sigma @ self.P.T + self.omega)
    
    def beta_2(self, t):
        """Compute beta_2 coefficient at time t."""
        return self.sigma @ self.P.T @ np.linalg.inv((self.T - t) * self.P @ self.sigma @ self.P.T + self.T * self.omega) @ self.P
    
    def eta_t(self, t):
        """Compute eta_t coefficient at time t."""
        return -self.P.T @ np.linalg.inv((self.T - t) * self.P @ self.sigma @ self.P.T + self.T * self.omega) @ self.P
    
    def alpha_t(self, t, y):
        """Compute alpha_t coefficient at time t given view y."""
        expected_x_t = t * (self.mu_x + self.beta_1() @ (y - self.T * self.P @ self.mu_x))
        return self.mu + self.beta_1() @ (y - self.T * self.P @ self.mu_x) - self.sigma @ self.eta_t(t) @ expected_x_t
    
    def M_t(self, t):
        """Compute M(t) matrix at time t."""
        term1 = (self.gamma - 1) * (1 - t/self.T) * self.P.T @ np.linalg.inv(self.omega) @ self.P
        term2 = self.gamma * np.linalg.inv(self.sigma) + (1 - t/self.T) * self.P.T @ np.linalg.inv(self.omega) @ self.P
        return term1 @ np.linalg.inv(term2)
    
    def A_t(self, t):
        """Compute A(t) matrix at time t."""
        return 0.5 * (self.M_t(t) @ self.eta_t(t) + self.eta_t(t).T @ self.M_t(t).T)
    
    def b_t(self, t, y):
        """Compute b(t) vector at time t given view y."""
        return self.M_t(t) @ np.linalg.inv(self.sigma) @ (self.alpha_t(t, y) - self.rf * np.ones(self.N))
    
    def mu_tilde(self, t, x, y):
        """Compute mu_tilde at time t given state x and view y."""
        expected_x_t = t * (self.mu_x + self.beta_1() @ (y - self.T * self.P @ self.mu_x))
        return self.mu + self.beta_1() @ (y - self.T * self.P @ self.mu_x) + self.beta_2(t) @ (expected_x_t - x)
    
    def optimal_portfolio(self, t, x, y):
        """
        Compute optimal portfolio at time t given state x and view y.
        
        Returns:
        - Mean-variance portfolio
        - Hedging demand
        - Total optimal portfolio
        """
        # Mean-variance portfolio
        mv_portfolio = (1/self.gamma) * np.linalg.inv(self.sigma) @ (self.mu_tilde(t, x, y) - self.rf * np.ones(self.N))
        
        # Hedging demand
        hedging_demand = (1/self.gamma) * (self.A_t(t) @ x + self.b_t(t, y))
        
        # Total optimal portfolio
        total_portfolio = mv_portfolio + hedging_demand
        
        return mv_portfolio, hedging_demand, total_portfolio
    
    def simulate_price_path(self, S0, y, n_steps=1000):
        """
        Simulate price path given views y.
        
        Parameters:
        - S0: Initial price vector (N x 1)
        - y: View vector (K x 1)
        - n_steps: Number of time steps
        
        Returns:
        - time_grid: Array of time points
        - price_paths: Matrix of price paths (n_steps x N)
        - log_returns: Matrix of log returns (n_steps x N)
        """
        dt = self.T / n_steps
        time_grid = np.linspace(0, self.T, n_steps+1)
        
        # Initialize arrays
        price_paths = np.zeros((n_steps+1, self.N))
        log_returns = np.zeros((n_steps+1, self.N))
        price_paths[0] = S0
        
        # Simulate paths
        for i in range(1, n_steps+1):
            t = time_grid[i-1]
            x = log_returns[i-1]
            
            # Expected log-return
            expected_x_t = t * (self.mu_x + self.beta_1() @ (y - self.T * self.P @ self.mu_x))
            
            # Drift term
            drift = (self.mu_x + self.beta_1() @ (y - self.T * self.P @ self.mu_x) + 
                    self.beta_2(t) @ (expected_x_t - x)) * dt
            
            # Diffusion term
            dW = np.random.multivariate_normal(np.zeros(self.N), dt * np.eye(self.N))
            diffusion = self.L @ dW
            
            # Update log-returns
            log_returns[i] = x + drift + diffusion
            
            # Update prices
            price_paths[i] = S0 * np.exp(log_returns[i])
            
        return time_grid, price_paths, log_returns
    
    def simulate_wealth(self, S0, y, Z0, n_steps=1000):
        """
        Simulate wealth path given views y.
        
        Parameters:
        - S0: Initial price vector (N x 1)
        - y: View vector (K x 1)
        - Z0: Initial wealth
        - n_steps: Number of time steps
        
        Returns:
        - time_grid: Array of time points
        - wealth_dbl: Wealth path using Dynamic Black-Litterman strategy
        - wealth_mv: Wealth path using Mean-Variance strategy
        - portfolio_weights_dbl: Portfolio weights using Dynamic Black-Litterman strategy
        - portfolio_weights_mv: Portfolio weights using Mean-Variance strategy
        """
        dt = self.T / n_steps
        time_grid = np.linspace(0, self.T, n_steps+1)
        
        # Initialize arrays
        wealth_dbl = np.zeros(n_steps+1)
        wealth_mv = np.zeros(n_steps+1)
        wealth_dbl[0] = Z0
        wealth_mv[0] = Z0
        
        portfolio_weights_dbl = np.zeros((n_steps+1, self.N))
        portfolio_weights_mv = np.zeros((n_steps+1, self.N))
        
        # Simulate price paths
        _, price_paths, log_returns = self.simulate_price_path(S0, y, n_steps)
        
        # Simulate wealth paths
        for i in range(n_steps):
            t = time_grid[i]
            x = log_returns[i]
            
            # Compute optimal portfolios
            mv_portfolio, hedging_demand, total_portfolio = self.optimal_portfolio(t, x, y)
            
            # Store portfolio weights
            portfolio_weights_dbl[i] = total_portfolio
            portfolio_weights_mv[i] = mv_portfolio
            
            # Compute price returns
            returns = (price_paths[i+1] - price_paths[i]) / price_paths[i]
            
            # Update wealth
            wealth_dbl[i+1] = wealth_dbl[i] * (1 + self.rf * dt + total_portfolio @ returns)
            wealth_mv[i+1] = wealth_mv[i] * (1 + self.rf * dt + mv_portfolio @ returns)
            
        return time_grid, wealth_dbl, wealth_mv, portfolio_weights_dbl, portfolio_weights_mv
    
    def compare_rebalancing_strategies(self, S0, y, Z0, rebalancing_frequencies=[1, 5, 20, 60], n_steps=1000, n_simulations=50):
        """
        Compare rebalancing strategies with different frequencies.
        
        Parameters:
        - S0: Initial price vector (N x 1)
        - y: View vector (K x 1)
        - Z0: Initial wealth
        - rebalancing_frequencies: List of rebalancing frequencies (in days)
        - n_steps: Number of time steps
        - n_simulations: Number of Monte Carlo simulations
        
        Returns:
        - Results DataFrame with statistics for each strategy and rebalancing frequency
        """
        dt = self.T / n_steps
        time_grid = np.linspace(0, self.T, n_steps+1)
        
        results = []
        
        for freq in rebalancing_frequencies:
            # Arrays to store results across simulations
            final_wealth_dbl = np.zeros(n_simulations)
            final_wealth_rcbl = np.zeros(n_simulations)
            turnover_dbl = np.zeros(n_simulations)
            turnover_rcbl = np.zeros(n_simulations)
            
            for sim in range(n_simulations):
                # Simulate price paths
                _, price_paths, log_returns = self.simulate_price_path(S0, y, n_steps)
                
                # Initialize wealth and portfolio weights
                wealth_dbl = np.zeros(n_steps+1)
                wealth_rcbl = np.zeros(n_steps+1)
                wealth_dbl[0] = Z0
                wealth_rcbl[0] = Z0
                
                portfolio_weights_dbl = np.zeros((n_steps+1, self.N))
                portfolio_weights_rcbl = np.zeros((n_steps+1, self.N))
                
                # Compute rebalancing times
                rebalancing_indices = np.arange(0, n_steps+1, freq)
                
                # Simulate wealth paths
                for i in range(n_steps):
                    t = time_grid[i]
                    x = log_returns[i]
                    
                    # Compute optimal portfolios
                    if i in rebalancing_indices:
                        # Dynamic Black-Litterman
                        _, _, total_portfolio = self.optimal_portfolio(t, x, y)
                        portfolio_weights_dbl[i] = total_portfolio
                        
                        # Rebalanced Classical Black-Litterman (single-period)
                        # Compute SIGMA_BL|t using formula from paper
                        sigma_bl_inv = np.linalg.inv(self.sigma) + (1 - t/self.T) * self.P.T @ np.linalg.inv(self.omega) @ self.P
                        sigma_bl = np.linalg.inv(sigma_bl_inv)
                        
                        # Compute mu_tilde_x
                        mu_tilde_x = self.mu_tilde(t, x, y) - 0.5 * np.diag(self.sigma)
                        
                        # Compute RCBL portfolio
                        rcbl_portfolio = (1/self.gamma) * sigma_bl @ (mu_tilde_x - self.rf * np.ones(self.N))
                        portfolio_weights_rcbl[i] = rcbl_portfolio
                    else:
                        # Keep previous weights
                        portfolio_weights_dbl[i] = portfolio_weights_dbl[i-1]
                        portfolio_weights_rcbl[i] = portfolio_weights_rcbl[i-1]
                    
                    # Compute price returns
                    returns = (price_paths[i+1] - price_paths[i]) / price_paths[i]
                    
                    # Update wealth
                    wealth_dbl[i+1] = wealth_dbl[i] * (1 + self.rf * dt + portfolio_weights_dbl[i] @ returns)
                    wealth_rcbl[i+1] = wealth_rcbl[i] * (1 + self.rf * dt + portfolio_weights_rcbl[i] @ returns)
                
                # Compute turnover
                turnover_dbl[sim] = np.sum(np.abs(np.diff(portfolio_weights_dbl, axis=0)))
                turnover_rcbl[sim] = np.sum(np.abs(np.diff(portfolio_weights_rcbl, axis=0)))
                
                # Store final wealth
                final_wealth_dbl[sim] = wealth_dbl[-1]
                final_wealth_rcbl[sim] = wealth_rcbl[-1]
            
            # Compute statistics
            results.append({
                'Rebalancing Frequency': freq,
                'Strategy': 'DBL',
                'Mean Final Wealth': np.mean(final_wealth_dbl),
                'Std Final Wealth': np.std(final_wealth_dbl),
                'Mean Turnover': np.mean(turnover_dbl),
                'Sharpe Ratio': (np.mean(final_wealth_dbl) - Z0) / np.std(final_wealth_dbl)
            })
            
            results.append({
                'Rebalancing Frequency': freq,
                'Strategy': 'RCBL',
                'Mean Final Wealth': np.mean(final_wealth_rcbl),
                'Std Final Wealth': np.std(final_wealth_rcbl),
                'Mean Turnover': np.mean(turnover_rcbl),
                'Sharpe Ratio': (np.mean(final_wealth_rcbl) - Z0) / np.std(final_wealth_rcbl)
            })
        
        return pd.DataFrame(results)

# Set parameters for the simulation
N = 5  # Number of assets
K = 3  # Number of views

# Generate random expected returns and covariance matrix
mu = np.array([0.0320, 0.0447, 0.0269, 0.0679, 0.0672])  # From the paper
sigma = np.array([
    [0.0641, 0.0175, 0.0086, 0.0266, 0.0363],
    [0.0175, 0.1191, 0.0234, 0.0303, 0.0353],
    [0.0086, 0.0234, 0.1154, 0.0322, 0.0278],
    [0.0266, 0.0303, 0.0322, 0.1230, 0.0431],
    [0.0363, 0.0353, 0.0278, 0.0431, 0.1679]
])  # From the paper

# Set risk-free rate and risk aversion
rf = 0.03
gamma = 5

# Set investment horizon
T = 1.0

# Define view matrix P (from the paper)
# Views: 
# 1. Difference between asset A and B
# 2. Difference between asset A and F
# 3. Return of asset C
P = np.array([
    [1, -1, 0, 0, 0],  # Asset A - Asset B
    [1, 0, 0, 0, -1],  # Asset A - Asset F
    [0, 0, 1, 0, 0]    # Asset C
])

# Define view covariance matrix with moderate noise (alpha = 0.4 from paper)
alpha = 0.4
omega = alpha * P @ sigma @ P.T

# Generate random views
np.random.seed(42)
y = P @ mu * T + np.random.multivariate_normal(np.zeros(K), omega)

# Initialize model
dbl = DynamicBlackLitterman(mu, sigma, rf, gamma, T, P, omega)

# Set initial prices and wealth
S0 = np.ones(N) * 100
Z0 = 1000

# Run simulation with different rebalancing frequencies
rebalancing_frequencies = [1, 5, 20, 60]  # Daily, weekly, monthly, quarterly
results = dbl.compare_rebalancing_strategies(S0, y, Z0, rebalancing_frequencies)

# Print results
print(results)

# Plot wealth paths for different strategies
time_grid, wealth_dbl, wealth_mv, weights_dbl, weights_mv = dbl.simulate_wealth(S0, y, Z0)

plt.figure(figsize=(12, 8))
plt.plot(time_grid, wealth_dbl, 'b-', label='Dynamic Black-Litterman')
plt.plot(time_grid, wealth_mv, 'r--', label='Mean-Variance')
plt.xlabel('Time')
plt.ylabel('Wealth')
plt.title('Wealth Paths for Different Strategies')
plt.legend()
plt.grid(True)
plt.savefig('wealth_paths.png')
plt.close()

# Plot portfolio weights for DBL
plt.figure(figsize=(12, 8))
for i in range(N):
    plt.plot(time_grid, weights_dbl[:, i], label=f'Asset {i+1}')
plt.xlabel('Time')
plt.ylabel('Portfolio Weight')
plt.title('Dynamic Black-Litterman Portfolio Weights')
plt.legend()
plt.grid(True)
plt.savefig('dbl_weights.png')
plt.close()

# Plot efficient frontier for different strategies
def plot_efficient_frontier():
    # Define range of risk aversion parameters
    gammas = np.logspace(-1, 1, 20)
    
    # Arrays to store results
    returns_dbl = []
    risks_dbl = []
    returns_rcbl = []
    risks_rcbl = []
    
    # Compute efficient frontier
    for g in gammas:
        # Create model with current risk aversion
        model = DynamicBlackLitterman(mu, sigma, rf, g, T, P, omega)
        
        # Run simulations
        n_simulations = 30
        returns_dbl_sim = []
        risks_dbl_sim = []
        returns_rcbl_sim = []
        risks_rcbl_sim = []
        
        for _ in range(n_simulations):
            # Simulate wealth paths
            _, wealth_dbl, wealth_rcbl, _, _ = model.simulate_wealth(S0, y, Z0)
            
            # Compute returns and risks
            returns_dbl_sim.append((wealth_dbl[-1] - Z0) / Z0)
            risks_dbl_sim.append(np.std(np.diff(wealth_dbl) / wealth_dbl[:-1]))
            
            returns_rcbl_sim.append((wealth_rcbl[-1] - Z0) / Z0)
            risks_rcbl_sim.append(np.std(np.diff(wealth_rcbl) / wealth_rcbl[:-1]))
        
        # Average results
        returns_dbl.append(np.mean(returns_dbl_sim))
        risks_dbl.append(np.mean(risks_dbl_sim))
        returns_rcbl.append(np.mean(returns_rcbl_sim))
        risks_rcbl.append(np.mean(risks_rcbl_sim))
    
    # Plot efficient frontier
    plt.figure(figsize=(12, 8))
    plt.plot(risks_dbl, returns_dbl, 'bo-', label='Dynamic Black-Litterman')
    plt.plot(risks_rcbl, returns_rcbl, 'ro-', label='Rebalanced Classical Black-Litterman')
    plt.xlabel('Standard Deviation (Risk)')
    plt.ylabel('Expected Return')
    plt.title('Efficient Frontier')
    plt.legend()
    plt.grid(True)
    plt.savefig('efficient_frontier.png')
    plt.close()

# Plot efficient frontier
plot_efficient_frontier()

# Compare DBL and RCBL for different view precision levels
def compare_view_precision():
    # Define range of alpha parameters (view noise levels)
    alphas = [0.1, 0.2, 0.4, 0.8, 1.6]
    
    # Arrays to store results
    results = []
    
    for alpha in alphas:
        # Define view covariance matrix
        omega = alpha * P @ sigma @ P.T
        
        # Generate random views
        np.random.seed(42)
        y = P @ mu * T + np.random.multivariate_normal(np.zeros(K), omega)
        
        # Initialize model
        model = DynamicBlackLitterman(mu, sigma, rf, gamma, T, P, omega)
        
        # Run simulations
        n_simulations = 30
        final_wealth_dbl = []
        final_wealth_rcbl = []
        turnover_dbl = []
        turnover_rcbl = []
        
        for _ in range(n_simulations):
            # Simulate wealth paths with weekly rebalancing
            rebalancing_freq = 5  # Weekly
            
            # Simulate price paths
            _, price_paths, log_returns = model.simulate_price_path(S0, y)
            
            # Initialize wealth and portfolio weights
            n_steps = len(log_returns) - 1
            wealth_dbl = np.zeros(n_steps+1)
            wealth_rcbl = np.zeros(n_steps+1)
            wealth_dbl[0] = Z0
            wealth_rcbl[0] = Z0
            
            portfolio_weights_dbl = np.zeros((n_steps+1, N))
            portfolio_weights_rcbl = np.zeros((n_steps+1, N))
            
            # Compute rebalancing times
            rebalancing_indices = np.arange(0, n_steps+1, rebalancing_freq)
            
            # Simulate wealth paths
            dt = T / n_steps
            for i in range(n_steps):
                t = i * dt
                x = log_returns[i]
                
                # Compute optimal portfolios
                if i in rebalancing_indices:
                    # Dynamic Black-Litterman
                    _, _, total_portfolio = model.optimal_portfolio(t, x, y)
                    portfolio_weights_dbl[i] = total_portfolio
                    
                    # Rebalanced Classical Black-Litterman (single-period)
                    # Compute SIGMA_BL|t using formula from paper
                    sigma_bl_inv = np.linalg.inv(sigma) + (1 - t/T) * P.T @ np.linalg.inv(omega) @ P
                    sigma_bl = np.linalg.inv(sigma_bl_inv)
                    
                    # Compute mu_tilde_x
                    mu_tilde_x = model.mu_tilde(t, x, y) - 0.5 * np.diag(sigma)
                    
                    # Compute RCBL portfolio
                    rcbl_portfolio = (1/gamma) * sigma_bl @ (mu_tilde_x - rf * np.ones(N))
                    portfolio_weights_rcbl[i] = rcbl_portfolio
                else:
                    # Keep previous weights
                    portfolio_weights_dbl[i] = portfolio_weights_dbl[i-1]
                    portfolio_weights_rcbl[i] = portfolio_weights_rcbl[i-1]
                
                # Compute price returns
                returns = (price_paths[i+1] - price_paths[i]) / price_paths[i]
                
                # Update wealth
                wealth_dbl[i+1] = wealth_dbl[i] * (1 + rf * dt + portfolio_weights_dbl[i] @ returns)
                wealth_rcbl[i+1] = wealth_rcbl[i] * (1 + rf * dt + portfolio_weights_rcbl[i] @ returns)
            
            # Compute turnover
            turnover_dbl.append(np.sum(np.abs(np.diff(portfolio_weights_dbl, axis=0))))
            turnover_rcbl.append(np.sum(np.abs(np.diff(portfolio_weights_rcbl, axis=0))))
            
            # Store final wealth
            final_wealth_dbl.append(wealth_dbl[-1])
            final_wealth_rcbl.append(wealth_rcbl[-1])
        
        # Store results
        results.append({
            'Alpha': alpha,
            'Strategy': 'DBL',
            'Mean Final Wealth': np.mean(final_wealth_dbl),
            'Std Final Wealth': np.std(final_wealth_dbl),
            'Mean Turnover': np.mean(turnover_dbl),
            'Sharpe Ratio': (np.mean(final_wealth_dbl) - Z0) / np.std(final_wealth_dbl)
        })
        
        results.append({
            'Alpha': alpha,
            'Strategy': 'RCBL',
            'Mean Final Wealth': np.mean(final_wealth_rcbl),
            'Std Final Wealth': np.std(final_wealth_rcbl),
            'Mean Turnover': np.mean(turnover_rcbl),
            'Sharpe Ratio': (np.mean(final_wealth_rcbl) - Z0) / np.std(final_wealth_rcbl)
        })
    
    return pd.DataFrame(results)

# Compare view precision
view_precision_results = compare_view_precision()
print("\nView Precision Results:")
print(view_precision_results)

# Plot view precision results
plt.figure(figsize=(12, 8))
dbl_results = view_precision_results[view_precision_results['Strategy'] == 'DBL']
rcbl_results = view_precision_results[view_precision_results['Strategy'] == 'RCBL']

plt.plot(dbl_results['Alpha'], dbl_results['Mean Final Wealth'], 'bo-', label='DBL')
plt.plot(rcbl_results['Alpha'], rcbl_results['Mean Final Wealth'], 'ro-', label='RCBL')
plt.xlabel('Alpha (View Noise Level)')
plt.ylabel('Mean Final Wealth')
plt.title('Effect of View Precision on Final Wealth')
plt.legend()
plt.grid(True)
plt.savefig('view_precision.png')
plt.close()

# Plot turnover for different view precision levels
plt.figure(figsize=(12, 8))
plt.plot(dbl_results['Alpha'], dbl_results['Mean Turnover'], 'bo-', label='DBL')
plt.plot(rcbl_results['Alpha'], rcbl_results['Mean Turnover'], 'ro-', label='RCBL')
plt.xlabel('Alpha (View Noise Level)')
plt.ylabel('Mean Turnover')
plt.title('Effect of View Precision on Portfolio Turnover')
plt.legend()
plt.grid(True)
plt.savefig('turnover_by_precision.png')
plt.close()