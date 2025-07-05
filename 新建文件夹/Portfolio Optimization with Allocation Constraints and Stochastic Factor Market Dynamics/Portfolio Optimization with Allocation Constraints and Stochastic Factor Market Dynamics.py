import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.optimize import minimize
from scipy.linalg import expm
from scipy.integrate import solve_ivp
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

class OUShortRateModel:
    """
    Multi-Factor OU Short Rate Model (MOU) as described in Section 4.3 of the paper.
    """
    
    def __init__(self, m=2, T=1.0, n_steps=252, w0=0.02, w1=None, kappa=None, 
                 theta=None, sigma=None, eta=None, bond_maturities=None, z0=None):
        """
        Initialize the model parameters.
        
        Parameters:
        -----------
        m : int
            Number of factors
        T : float
            Investment horizon
        n_steps : int
            Number of time steps for simulation
        w0 : float
            Constant term in short rate equation
        w1 : array
            Weight vector for z in short rate equation
        kappa : array
            Mean reversion speed for z
        theta : array
            Long-term mean for z
        sigma : array
            Volatility matrix for z
        eta : array
            Market price of risk
        bond_maturities : array
            Maturities of traded zero-coupon bonds
        z0 : array
            Initial value of stochastic factor z
        """
        self.m = m
        self.T = T
        self.n_steps = n_steps
        self.dt = T / n_steps
        self.time_grid = np.linspace(0, T, n_steps + 1)
        
        # Default parameter values if not provided
        if w1 is None:
            w1 = np.ones(m) * 0.01
        if kappa is None:
            kappa = np.ones(m) * 0.5
        if theta is None:
            theta = np.zeros(m)
        if sigma is None:
            sigma = np.eye(m) * 0.1
        if eta is None:
            eta = np.ones(m) * 0.2
        if bond_maturities is None:
            bond_maturities = np.linspace(T + 0.5, T + m * 0.5, m)
        if z0 is None:
            z0 = np.zeros(m)
            
        self.w0 = w0
        self.w1 = w1
        self.kappa = kappa
        self.theta = theta
        self.sigma = sigma
        self.eta = eta
        self.bond_maturities = bond_maturities
        self.z0 = z0
        
        # Validate dimensions
        assert len(w1) == m, "w1 must have length m"
        assert len(kappa) == m, "kappa must have length m"
        assert len(theta) == m, "theta must have length m"
        assert sigma.shape == (m, m), "sigma must be m x m"
        assert len(eta) == m, "eta must have length m"
        assert len(bond_maturities) == m, "bond_maturities must have length m"
        assert len(z0) == m, "z0 must have length m"
        
        # Calculate b(τ) for bond pricing (equation 24 in the paper)
        self.b_func = self._calculate_b_func()
        
    def _calculate_b_func(self):
        """
        Calculate the b(τ) function for bond pricing.
        
        In affine term structure models, the zero-coupon bond price is:
        P(t,T) = exp(a(T-t) + b(T-t)' * z(t))
        
        For OU processes, b(τ) has a closed-form solution.
        """
        def b_ode(tau, b, kappa, w1):
            return -kappa * b - w1
        
        b_functions = []
        
        for i in range(self.m):
            # Solve ODE for each component of b
            result = solve_ivp(
                lambda tau, b: b_ode(tau, b, self.kappa[i], self.w1[i]),
                [0, 10],  # Solve for τ from 0 to 10 (long enough for our maturities)
                [0],      # Initial condition b(0) = 0
                t_eval=np.linspace(0, 10, 1000)
            )
            
            # Create interpolation function
            from scipy.interpolate import interp1d
            b_func = interp1d(result.t, result.y[0], bounds_error=False, fill_value="extrapolate")
            b_functions.append(b_func)
            
        return b_functions
    
    def get_b_matrix(self, t):
        """
        Calculate the b(t;T̂) matrix for all bond maturities at time t.
        """
        b_matrix = np.zeros((self.m, self.m))
        
        for i in range(self.m):
            for j in range(self.m):
                tau = self.bond_maturities[j] - t
                if tau > 0:
                    b_matrix[i, j] = self.b_func[i](tau)
                else:
                    b_matrix[i, j] = 0
                    
        return b_matrix
    
    def simulate_paths(self):
        """
        Simulate paths for the z process, short rate r, and bond prices.
        """
        # Initialize arrays
        z = np.zeros((self.n_steps + 1, self.m))
        r = np.zeros(self.n_steps + 1)
        bond_prices = np.zeros((self.n_steps + 1, self.m))
        
        # Set initial values
        z[0] = self.z0
        r[0] = self.w0 + np.dot(self.w1, self.z0)
        
        # Calculate initial bond prices
        for j in range(self.m):
            tau = self.bond_maturities[j]
            b_values = np.array([self.b_func[i](tau) for i in range(self.m)])
            bond_prices[0, j] = np.exp(-self.w0 * tau - np.dot(b_values, self.z0))
        
        # Generate random increments for the Brownian motion
        dW = np.random.normal(0, np.sqrt(self.dt), (self.n_steps, self.m))
        
        # Simulate paths
        for i in range(self.n_steps):
            t = self.time_grid[i]
            
            # Update z using Euler-Maruyama
            drift = (self.kappa * (self.theta - z[i])) * self.dt
            diffusion = np.dot(self.sigma, dW[i])
            z[i+1] = z[i] + drift + diffusion
            
            # Update short rate
            r[i+1] = self.w0 + np.dot(self.w1, z[i+1])
            
            # Update bond prices
            for j in range(self.m):
                tau = self.bond_maturities[j] - t - self.dt
                if tau > 0:
                    b_values = np.array([self.b_func[k](tau) for k in range(self.m)])
                    bond_prices[i+1, j] = np.exp(-self.w0 * tau - np.dot(b_values, z[i+1]))
                else:
                    # Bond has matured
                    bond_prices[i+1, j] = 1.0
        
        return {
            'time': self.time_grid,
            'z': z,
            'r': r,
            'bond_prices': bond_prices
        }
    
    def calculate_bond_returns(self, bond_prices):
        """
        Calculate the returns of the bonds.
        """
        returns = np.zeros((self.n_steps, self.m))
        
        for i in range(self.n_steps):
            for j in range(self.m):
                returns[i, j] = bond_prices[i+1, j] / bond_prices[i, j] - 1
                
        return returns
    
    def calculate_b_T_minus_t(self, t):
        """
        Calculate B(T-t) for the optimal portfolio formula.
        """
        b = np.zeros(self.m)
        
        for i in range(self.m):
            tau = self.T - t
            if tau > 0:
                b[i] = self.b_func[i](tau)
                
        return b
    
    def support_function(self, lambda_vec, K):
        """
        Calculate the support function δ_K(λ).
        
        For common constraint sets:
        - No shorting: δ_K(λ) = 0 if λ <= 0, else infinity
        - Box constraints: δ_K(λ) = sum_i max(0, λ_i * upper_i) + sum_i min(0, λ_i * lower_i)
        """
        if K == "no_shorting":
            # No shorting constraints: K = {π : π >= 0}
            if np.all(lambda_vec <= 0):
                return 0.0
            else:
                return float('inf')
        elif isinstance(K, tuple) and K[0] == "box":
            # Box constraints: K = {π : lower <= π <= upper}
            lower, upper = K[1], K[2]
            result = 0.0
            for i, lambda_i in enumerate(lambda_vec):
                if lambda_i > 0:
                    result += lambda_i * upper[i]
                else:
                    result += lambda_i * lower[i]
            return result
        elif isinstance(K, tuple) and K[0] == "simplex":
            # Simplex constraints: K = {π : π >= 0, sum(π) <= 1}
            c = K[1]  # The upper bound on sum(π)
            if np.all(lambda_vec <= 0):
                return c * max(0, np.max(lambda_vec))
            else:
                return float('inf')
        else:
            raise ValueError("Unsupported constraint set")
    
    def minimize_lambda(self, t, z, B, K, b, risk_aversion=0.5):
        """
        Solve the minimization problem for λ* in equation (15).
        
        Parameters:
        -----------
        t : float
            Current time
        z : array
            Current state of stochastic factor
        B : array
            Current value of B(T-t)
        K : object
            Constraint set specification
        b : float
            Risk aversion parameter, b < 1 and b ≠ 0
        
        Returns:
        --------
        lambda_star : array
            Optimal λ*
        """
        # Get the b(t;T̂) matrix
        b_matrix = self.get_b_matrix(t)
        
        # Calculate the terms needed for the objective function
        eta_term = self.eta + np.dot(self.sigma.T, B)
        
        # Define the objective function
        def objective(lambda_vec):
            # Support function term
            support_term = 2 * (1 - b) * self.support_function(lambda_vec, K)
            
            # Quadratic term
            quad_term = np.linalg.norm(eta_term + np.dot(np.linalg.inv(b_matrix.T @ self.sigma), lambda_vec))**2
            
            return support_term + quad_term
        
        # For "no_shorting" and "simplex" constraints, we can use a projected gradient method
        if K == "no_shorting" or (isinstance(K, tuple) and K[0] == "simplex"):
            # Initialize with a feasible point (all zeros)
            lambda_init = np.zeros(self.m)
            
            # Define bounds for the optimization
            bounds = [(None, 0) for _ in range(self.m)]  # λ <= 0 for no shorting
            
            # Solve the minimization problem
            result = minimize(objective, lambda_init, bounds=bounds, method='L-BFGS-B')
            
            return result.x
        
        # For box constraints, we can use the L-BFGS-B algorithm with appropriate bounds
        elif isinstance(K, tuple) and K[0] == "box":
            # Initialize with zeros
            lambda_init = np.zeros(self.m)
            
            # Solve the minimization problem
            result = minimize(objective, lambda_init, method='L-BFGS-B')
            
            return result.x
        
        else:
            raise ValueError("Unsupported constraint set")
    
    def optimal_portfolio(self, t, z, K, b=0.5):
        """
        Calculate the optimal portfolio allocation π* at time t.
        
        Parameters:
        -----------
        t : float
            Current time
        z : array
            Current state of stochastic factor
        K : object
            Constraint set specification
        b : float
            Risk aversion parameter, b < 1 and b ≠ 0
        
        Returns:
        --------
        pi_star : array
            Optimal portfolio allocation
        """
        # Calculate B(T-t)
        B = self.calculate_b_T_minus_t(t)
        
        # Calculate λ*
        lambda_star = self.minimize_lambda(t, z, B, K, b)
        
        # Get the b(t;T̂) matrix
        b_matrix = self.get_b_matrix(t)
        
        # Calculate the terms for π*
        sigma_term = self.sigma @ b_matrix
        eta_term = self.eta + lambda_star + np.dot(self.sigma.T, B)
        
        # Calculate π* using equation (18)
        pi_star = (1 / (1 - b)) * np.linalg.inv(sigma_term @ sigma_term.T) @ sigma_term @ eta_term
        
        # Project π* onto the constraint set if needed
        if K == "no_shorting":
            pi_star = np.maximum(pi_star, 0)
        elif isinstance(K, tuple) and K[0] == "box":
            lower, upper = K[1], K[2]
            pi_star = np.minimum(np.maximum(pi_star, lower), upper)
        elif isinstance(K, tuple) and K[0] == "simplex":
            c = K[1]
            pi_star = np.maximum(pi_star, 0)
            if np.sum(pi_star) > c:
                # Project onto the simplex
                pi_star = self._project_to_simplex(pi_star, c)
        
        return pi_star
    
    def _project_to_simplex(self, x, c=1.0):
        """
        Project a vector x onto the simplex: {π : π >= 0, sum(π) <= c}
        """
        # Sort x in descending order
        sorted_x = np.sort(x)[::-1]
        cumsum = np.cumsum(sorted_x)
        
        # Find the index where the projection happens
        indices = np.arange(1, len(x) + 1)
        rho = np.max(np.where((sorted_x - (cumsum - c) / indices) > 0)[0])
        
        # Calculate the threshold
        theta = (cumsum[rho] - c) / (rho + 1)
        
        # Project x
        return np.maximum(x - theta, 0)
    
    def simulate_wealth(self, K, b=0.5, initial_wealth=1.0):
        """
        Simulate wealth process using the optimal portfolio strategy.
        
        Parameters:
        -----------
        K : object
            Constraint set specification
        b : float
            Risk aversion parameter, b < 1 and b ≠ 0
        initial_wealth : float
            Initial wealth
        
        Returns:
        --------
        dict : Dictionary containing simulation results
        """
        # Simulate market paths
        paths = self.simulate_paths()
        time_grid = paths['time']
        z = paths['z']
        r = paths['r']
        bond_prices = paths['bond_prices']
        
        # Calculate bond returns
        bond_returns = self.calculate_bond_returns(bond_prices)
        
        # Initialize wealth process and portfolio allocations
        wealth = np.zeros(self.n_steps + 1)
        wealth[0] = initial_wealth
        portfolio = np.zeros((self.n_steps, self.m))
        
        # Simulate wealth process
        for i in range(self.n_steps):
            t = time_grid[i]
            
            # Calculate optimal portfolio
            portfolio[i] = self.optimal_portfolio(t, z[i], K, b)
            
            # Update wealth
            risk_free_return = np.exp(r[i] * self.dt) - 1
            risky_return = bond_returns[i]
            
            # Wealth dynamics equation (1)
            portfolio_return = risk_free_return + np.dot(portfolio[i], risky_return - risk_free_return)
            wealth[i+1] = wealth[i] * (1 + portfolio_return)
        
        return {
            'time': time_grid,
            'wealth': wealth,
            'portfolio': portfolio,
            'z': z,
            'r': r,
            'bond_prices': bond_prices,
            'bond_returns': bond_returns
        }
    
    def compare_strategies(self, strategies, initial_wealth=1.0):
        """
        Compare different portfolio strategies.
        
        Parameters:
        -----------
        strategies : list
            List of dictionaries with keys 'name', 'K', and 'b'
        initial_wealth : float
            Initial wealth
        
        Returns:
        --------
        dict : Dictionary containing simulation results for all strategies
        """
        results = {}
        
        # Simulate market paths once
        paths = self.simulate_paths()
        time_grid = paths['time']
        z = paths['z']
        r = paths['r']
        bond_prices = paths['bond_prices']
        bond_returns = self.calculate_bond_returns(bond_prices)
        
        for strategy in strategies:
            name = strategy['name']
            K = strategy['K']
            b = strategy.get('b', 0.5)
            
            # Initialize wealth process and portfolio allocations
            wealth = np.zeros(self.n_steps + 1)
            wealth[0] = initial_wealth
            portfolio = np.zeros((self.n_steps, self.m))
            
            # Simulate wealth process
            for i in range(self.n_steps):
                t = time_grid[i]
                
                # Calculate portfolio for this strategy
                if name == "Equal Weight":
                    portfolio[i] = np.ones(self.m) / self.m
                elif name == "Risk Free":
                    portfolio[i] = np.zeros(self.m)
                else:
                    # Calculate optimal portfolio
                    portfolio[i] = self.optimal_portfolio(t, z[i], K, b)
                
                # Update wealth
                risk_free_return = np.exp(r[i] * self.dt) - 1
                risky_return = bond_returns[i]
                
                # Wealth dynamics equation (1)
                portfolio_return = risk_free_return + np.dot(portfolio[i], risky_return - risk_free_return)
                wealth[i+1] = wealth[i] * (1 + portfolio_return)
            
            # Calculate performance metrics
            returns = np.diff(wealth) / wealth[:-1]
            annual_return = np.mean(returns) * 252
            annual_vol = np.std(returns) * np.sqrt(252)
            sharpe_ratio = annual_return / annual_vol if annual_vol > 0 else 0
            
            # Calculate drawdowns
            peak = np.maximum.accumulate(wealth)
            drawdowns = (peak - wealth) / peak
            max_drawdown = np.max(drawdowns)
            
            # Calculate terminal utility
            if b == 0:  # Log utility
                terminal_utility = np.log(wealth[-1])
            else:
                terminal_utility = wealth[-1]**b / b
            
            results[name] = {
                'wealth': wealth,
                'portfolio': portfolio,
                'annual_return': annual_return,
                'annual_vol': annual_vol,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'terminal_utility': terminal_utility
            }
        
        results['market_data'] = {
            'time': time_grid,
            'z': z,
            'r': r,
            'bond_prices': bond_prices,
            'bond_returns': bond_returns
        }
        
        return results

# Visualization functions
def plot_market_simulation(results):
    """
    Plot the simulated market data.
    """
    market_data = results['market_data']
    time = market_data['time']
    z = market_data['z']
    r = market_data['r']
    bond_prices = market_data['bond_prices']
    
    m = z.shape[1]
    
    plt.figure(figsize=(15, 10))
    
    # Plot stochastic factors
    plt.subplot(3, 1, 1)
    for i in range(m):
        plt.plot(time, z[:, i], label=f'z{i+1}')
    plt.title('Stochastic Factors')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    
    # Plot short rate
    plt.subplot(3, 1, 2)
    plt.plot(time, r, 'k')
    plt.title('Short Rate')
    plt.xlabel('Time')
    plt.ylabel('Rate')
    plt.grid(True)
    
    # Plot bond prices
    plt.subplot(3, 1, 3)
    for i in range(m):
        plt.plot(time, bond_prices[:, i], label=f'Bond {i+1}')
    plt.title('Bond Prices')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('market_simulation.png')
    plt.close()

def plot_wealth_processes(results):
    """
    Plot the wealth processes for different strategies.
    """
    plt.figure(figsize=(12, 6))
    
    time = results['market_data']['time']
    
    for name, data in results.items():
        if name != 'market_data':
            plt.plot(time, data['wealth'], label=name)
    
    plt.title('Wealth Processes')
    plt.xlabel('Time')
    plt.ylabel('Wealth')
    plt.legend()
    plt.grid(True)
    plt.savefig('wealth_processes.png')
    plt.close()

def plot_portfolio_allocations(results, strategy_name):
    """
    Plot the portfolio allocations for a specific strategy.
    """
    strategy_data = results[strategy_name]
    portfolio = strategy_data['portfolio']
    time = results['market_data']['time'][:-1]  # Exclude last time point
    
    m = portfolio.shape[1]
    
    plt.figure(figsize=(12, 6))
    
    for i in range(m):
        plt.plot(time, portfolio[:, i], label=f'Asset {i+1}')
    
    plt.title(f'Portfolio Allocations - {strategy_name}')
    plt.xlabel('Time')
    plt.ylabel('Allocation')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'portfolio_allocations_{strategy_name.replace(" ", "_").lower()}.png')
    plt.close()

def plot_performance_metrics(results):
    """
    Plot performance metrics for different strategies.
    """
    # Extract strategies and metrics
    strategies = [name for name in results.keys() if name != 'market_data']
    annual_returns = [results[name]['annual_return'] for name in strategies]
    annual_vols = [results[name]['annual_vol'] for name in strategies]
    sharpe_ratios = [results[name]['sharpe_ratio'] for name in strategies]
    max_drawdowns = [results[name]['max_drawdown'] for name in strategies]
    terminal_utilities = [results[name]['terminal_utility'] for name in strategies]
    
    # Create a figure with multiple subplots
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    
    # Plot annual returns
    axes[0, 0].bar(strategies, annual_returns)
    axes[0, 0].set_title('Annual Return')
    axes[0, 0].set_ylabel('Return')
    axes[0, 0].grid(True)
    
    # Plot annual volatility
    axes[0, 1].bar(strategies, annual_vols)
    axes[0, 1].set_title('Annual Volatility')
    axes[0, 1].set_ylabel('Volatility')
    axes[0, 1].grid(True)
    
    # Plot Sharpe ratio
    axes[1, 0].bar(strategies, sharpe_ratios)
    axes[1, 0].set_title('Sharpe Ratio')
    axes[1, 0].set_ylabel('Ratio')
    axes[1, 0].grid(True)
    
    # Plot maximum drawdown
    axes[1, 1].bar(strategies, max_drawdowns)
    axes[1, 1].set_title('Maximum Drawdown')
    axes[1, 1].set_ylabel('Drawdown')
    axes[1, 1].grid(True)
    
    # Plot terminal utility
    axes[2, 0].bar(strategies, terminal_utilities)
    axes[2, 0].set_title('Terminal Utility')
    axes[2, 0].set_ylabel('Utility')
    axes[2, 0].grid(True)
    
    # Leave the last subplot empty
    axes[2, 1].axis('off')
    
    # Adjust layout
    plt.tight_layout()
    plt.savefig('performance_metrics.png')
    plt.close()

def print_performance_summary(results):
    """
    Print a summary of performance metrics for different strategies.
    """
    # Create a DataFrame to display the results
    strategies = [name for name in results.keys() if name != 'market_data']
    metrics = ['Annual Return', 'Annual Volatility', 'Sharpe Ratio', 'Max Drawdown', 'Terminal Utility']
    
    data = []
    for name in strategies:
        row = [
            results[name]['annual_return'] * 100,  # Convert to percentage
            results[name]['annual_vol'] * 100,     # Convert to percentage
            results[name]['sharpe_ratio'],
            results[name]['max_drawdown'] * 100,   # Convert to percentage
            results[name]['terminal_utility']
        ]
        data.append(row)
    
    df = pd.DataFrame(data, index=strategies, columns=metrics)
    
    # Format the DataFrame for better readability
    formatted_df = df.copy()
    formatted_df['Annual Return'] = formatted_df['Annual Return'].map('{:.2f}%'.format)
    formatted_df['Annual Volatility'] = formatted_df['Annual Volatility'].map('{:.2f}%'.format)
    formatted_df['Sharpe Ratio'] = formatted_df['Sharpe Ratio'].map('{:.2f}'.format)
    formatted_df['Max Drawdown'] = formatted_df['Max Drawdown'].map('{:.2f}%'.format)
    formatted_df['Terminal Utility'] = formatted_df['Terminal Utility'].map('{:.4f}'.format)
    
    print("\nPerformance Summary:")
    print("===================")
    print(formatted_df)
    print("\n")

def main():
    # Set up the model
    m = 2  # Number of factors
    T = 1.0  # Investment horizon (1 year)
    n_steps = 252  # Number of time steps (daily)
    
    # Model parameters
    w0 = 0.02  # Constant term in short rate
    w1 = np.array([0.01, 0.005])  # Weights for z in short rate
    kappa = np.array([0.5, 0.3])  # Mean reversion speeds
    theta = np.array([0.0, 0.0])  # Long-term means
    sigma = np.array([[0.02, 0.005], [0.005, 0.015]])  # Volatility matrix
    eta = np.array([0.3, 0.2])  # Market price of risk
    bond_maturities = np.array([2.0, 3.0])  # Bond maturities
    z0 = np.array([0.0, 0.0])  # Initial state
    
    # Initialize the model
    model = OUShortRateModel(
        m=m, T=T, n_steps=n_steps, w0=w0, w1=w1, kappa=kappa, 
        theta=theta, sigma=sigma, eta=eta, bond_maturities=bond_maturities, z0=z0
    )
    
    # Define strategies to compare
    strategies = [
        {
            'name': 'Unconstrained (b=0.5)',
            'K': None,
            'b': 0.5
        },
        {
            'name': 'No Shorting (b=0.5)',
            'K': 'no_shorting',
            'b': 0.5
        },
        {
            'name': 'Box Constraints (b=0.5)',
            'K': ('box', np.array([0.0, 0.0]), np.array([0.8, 0.8])),
            'b': 0.5
        },
        {
            'name': 'Simplex (b=0.5)',
            'K': ('simplex', 1.0),
            'b': 0.5
        },
        {
            'name': 'Equal Weight',
            'K': None
        },
        {
            'name': 'Risk Free',
            'K': None
        }
    ]
    
    # Simulate and compare strategies
    results = model.compare_strategies(strategies)
    
    # Plot results
    plot_market_simulation(results)
    plot_wealth_processes(results)
    
    for strategy in strategies:
        plot_portfolio_allocations(results, strategy['name'])
    
    plot_performance_metrics(results)
    print_performance_summary(results)

if __name__ == "__main__":
    main()