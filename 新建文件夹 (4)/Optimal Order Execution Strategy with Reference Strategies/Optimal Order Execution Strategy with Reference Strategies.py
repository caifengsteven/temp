import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.linalg import expm
import random
from tqdm import tqdm

# Set random seed for reproducibility
np.random.seed(42)

class OptimalOrderExecution:
    def __init__(self, x0=100, A=0, T=1.0, eta=10, gamma=0.01, 
                 sigma=0.5, beta=1000, mu=0, theta=0.002, m0=0):
        """
        Initialize the optimal order execution model.
        
        Parameters:
        -----------
        x0 : float
            Initial inventory
        A : float
            Target inventory
        T : float
            Time horizon
        eta : float
            Temporary price impact parameter
        gamma : float
            Permanent price impact parameter
        sigma : float
            Volatility of price process
        beta : float
            Penalty for terminal deviation from target
        mu : float
            Drift of price process
        theta : float
            Risk aversion parameter
        m0 : float
            Execution risk parameter
        """
        self.x0 = x0
        self.A = A
        self.T = T
        self.eta = eta
        self.gamma = gamma
        self.sigma = sigma
        self.beta = beta
        self.mu = mu
        self.theta = theta
        self.m0 = m0
        
        # Compute kappa parameter for IS and TC strategies
        self.kappa = np.sqrt(theta * sigma**2 / (2 * eta))
        
    def is_strategy(self, t):
        """
        Implementation Shortfall (IS) order strategy.
        
        Parameters:
        -----------
        t : float or ndarray
            Time(s) at which to evaluate the strategy
            
        Returns:
        --------
        x_t : float or ndarray
            Inventory at time(s) t
        """
        return self.A + (self.x0 - self.A) * self.is_unit(t)
    
    def is_unit(self, t):
        """
        Unit Implementation Shortfall (IS) order.
        
        Parameters:
        -----------
        t : float or ndarray
            Time(s) at which to evaluate the unit IS
            
        Returns:
        --------
        is_t : float or ndarray
            Unit IS value at time(s) t
        """
        return np.sinh(self.kappa * (self.T - t)) / np.sinh(self.kappa * self.T)
    
    def tc_strategy(self, t):
        """
        Target Close (TC) order strategy.
        
        Parameters:
        -----------
        t : float or ndarray
            Time(s) at which to evaluate the strategy
            
        Returns:
        --------
        x_t : float or ndarray
            Inventory at time(s) t
        """
        return self.A + (self.x0 - self.A) * self.tc_unit(t)
    
    def tc_unit(self, t):
        """
        Unit Target Close (TC) order.
        
        Parameters:
        -----------
        t : float or ndarray
            Time(s) at which to evaluate the unit TC
            
        Returns:
        --------
        tc_t : float or ndarray
            Unit TC value at time(s) t
        """
        return (np.sinh(self.kappa * self.T) - np.sinh(self.kappa * t)) / np.sinh(self.kappa * self.T)
    
    def twap_strategy(self, t):
        """
        Time-Weighted Average Price (TWAP) strategy.
        
        Parameters:
        -----------
        t : float or ndarray
            Time(s) at which to evaluate the strategy
            
        Returns:
        --------
        x_t : float or ndarray
            Inventory at time(s) t
        """
        return self.A + (self.x0 - self.A) * (1 - t / self.T)
    
    def endpoints_only_strategy(self, t, R):
        """
        Endpoints-only reference strategy.
        
        Parameters:
        -----------
        t : float or ndarray
            Time(s) at which to evaluate the strategy
        R : float
            Reference level during (0,T)
            
        Returns:
        --------
        x_t : float or ndarray
            Inventory at time(s) t under the optimal strategy
        """
        merton_ratio = self.mu / (self.theta * self.sigma**2)
        
        return ((self.x0 - merton_ratio - R) * self.is_unit(t) + 
                (-self.A + merton_ratio + R) * self.tc_unit(t) + self.A)
    
    def piecewise_constant_strategy(self, t, Rs):
        """
        Optimal strategy for piece-wise constant reference strategy.
        
        Parameters:
        -----------
        t : float or ndarray
            Time(s) at which to evaluate the strategy
        Rs : list
            Reference levels for each sub-interval
            
        Returns:
        --------
        x_t : float or ndarray
            Inventory at time(s) t under the optimal strategy
        """
        n = len(Rs)
        dt = self.T / n
        
        # Handle scalar and array inputs differently
        if np.isscalar(t):
            k = min(int(t / dt), n-1)  # Determine which interval t falls into
            t_rel = t - k * dt  # Time relative to start of interval
            
            # Calculate the optimal xt value for the current interval
            if k < n-1:  # Not the last interval
                a_k = self.compute_a_k(k, Rs, n)
                a_k_minus_1 = self.compute_a_k(k-1, Rs, n) if k > 0 else self.x0
                
                return (a_k + 
                       (Rs[k] - a_k) * self.tc_unit_scaled(t_rel, dt) + 
                       (a_k_minus_1 - Rs[k]) * self.is_unit_scaled(t_rel, dt))
            else:  # Last interval
                a_n_minus_1 = self.compute_a_k(n-1, Rs, n)
                
                return (self.A + 
                       (Rs[n-1] - self.A) * self.tc_unit_scaled(t_rel, dt) + 
                       (a_n_minus_1 - Rs[n-1]) * self.is_unit_scaled(t_rel, dt))
        else:
            # For array inputs, process each element
            result = np.zeros_like(t, dtype=float)
            for i, t_i in enumerate(t):
                result[i] = self.piecewise_constant_strategy(t_i, Rs)
            return result
    
    def compute_a_k(self, k, Rs, n):
        """
        Compute a_k for piecewise constant strategy.
        
        Parameters:
        -----------
        k : int
            Index of interval
        Rs : list
            Reference levels
        n : int
            Total number of intervals
            
        Returns:
        --------
        a_k : float
            The a_k value
        """
        if k == n-1:
            return self.A
        
        # Compute b_i values
        b_values = [1]  # b_0 = 1
        for i in range(1, n):
            b_i = np.cosh(self.kappa * i * self.T / n) - np.cosh(self.kappa * (i-1) * self.T / n)
            b_values.append(b_i)
        
        # Initialize terms
        first_term = np.sinh(self.kappa * (self.T - (k+1) * self.T / n)) / np.sinh(self.kappa * self.T)
        second_term = np.sinh(self.kappa * (k+1) * self.T / n) / np.sinh(self.kappa * self.T)
        
        # Compute a_k
        a_k = 0
        
        # First sum
        for i in range(k+1):
            if i == 0:
                a_k += first_term * b_values[i] * self.x0
            else:
                a_k += first_term * b_values[i] * Rs[i-1]
        
        # Second sum
        for i in range(k+1, n+1):
            if i == n:
                a_k += second_term * b_values[n-i] * self.A
            else:
                a_k += second_term * b_values[n-i] * Rs[i-1]
        
        return a_k
    
    def is_unit_scaled(self, t, dt):
        """
        Scaled unit IS for piecewise constant strategy.
        
        Parameters:
        -----------
        t : float
            Time relative to interval start
        dt : float
            Interval length
            
        Returns:
        --------
        is_t : float
            Scaled unit IS
        """
        return np.sinh(self.kappa * (dt - t)) / np.sinh(self.kappa * dt)
    
    def tc_unit_scaled(self, t, dt):
        """
        Scaled unit TC for piecewise constant strategy.
        
        Parameters:
        -----------
        t : float
            Time relative to interval start
        dt : float
            Interval length
            
        Returns:
        --------
        tc_t : float
            Scaled unit TC
        """
        return (np.sinh(self.kappa * dt) - np.sinh(self.kappa * t)) / np.sinh(self.kappa * dt)
    
    def general_reference_strategy(self, t, Rt_func):
        """
        Optimal strategy for a general reference strategy.
        
        Parameters:
        -----------
        t : float or ndarray
            Time(s) at which to evaluate the strategy
        Rt_func : function
            Function that takes time t and returns the reference strategy value
            
        Returns:
        --------
        x_t : float or ndarray
            Inventory at time(s) t under the optimal strategy
        """
        # For general reference strategy, we use numerical integration
        if np.isscalar(t):
            # Define the integration grid
            N = 100  # Number of grid points
            grid = np.linspace(0, self.T, N+1)
            
            # Compute the integrand at each grid point
            integrand = np.zeros(N+1)
            for i, s in enumerate(grid):
                Rs = Rt_func(s)
                integrand[i] = Rs * np.sinh(self.kappa * min(s, t)) * np.sinh(self.kappa * (self.T - max(s, t)))
            
            # Numerical integration (trapezoidal rule)
            integral = np.trapz(integrand, grid)
            
            # Compute the optimal strategy
            result = (self.kappa / np.sinh(self.kappa * self.T)) * integral
            result += ((self.x0 - self.mu / (self.theta * self.sigma**2)) * self.is_unit(t) + 
                      (self.A - self.mu / (self.theta * self.sigma**2)) * (1 - self.is_unit(t) - self.tc_unit(t)))
            
            return result
        else:
            # For array inputs, process each element
            result = np.zeros_like(t, dtype=float)
            for i, t_i in enumerate(t):
                result[i] = self.general_reference_strategy(t_i, Rt_func)
            return result
    
    def compute_trading_rate(self, t, x_t, strategy_func):
        """
        Compute the trading rate at time t given the inventory.
        
        Parameters:
        -----------
        t : float
            Time at which to evaluate the trading rate
        x_t : float
            Current inventory
        strategy_func : function
            Function that computes the optimal inventory
            
        Returns:
        --------
        v_t : float
            Trading rate at time t
        """
        dt = 1e-6  # Small time increment
        
        # For the final time point, use the left derivative
        if abs(t - self.T) < dt:
            x_minus_dt = strategy_func(t - dt)
            return (x_t - x_minus_dt) / dt
        
        # Otherwise use central difference
        x_plus_dt = strategy_func(t + dt)
        return (x_t - x_plus_dt) / dt
    
    def simulate_price_process(self, strategy_func, N_steps=1000, N_paths=1):
        """
        Simulate the price process and execution strategy.
        
        Parameters:
        -----------
        strategy_func : function
            Function that computes the optimal inventory at time t
        N_steps : int
            Number of time steps in the simulation
        N_paths : int
            Number of price paths to simulate
            
        Returns:
        --------
        results : dict
            Dictionary containing simulation results
        """
        dt = self.T / N_steps
        time_grid = np.linspace(0, self.T, N_steps+1)
        
        # Initialize arrays
        S = np.zeros((N_paths, N_steps+1))
        S_tilde = np.zeros((N_paths, N_steps+1))
        x = np.zeros((N_paths, N_steps+1))
        v = np.zeros((N_paths, N_steps+1))
        
        # Set initial values
        S[:, 0] = 100.0  # Initial price
        x[:, 0] = self.x0  # Initial inventory
        
        # For each path
        for path in range(N_paths):
            # Generate price innovations
            dW = np.random.normal(0, np.sqrt(dt), N_steps)
            
            # For each time step
            for i in range(N_steps):
                t = time_grid[i]
                
                # Compute optimal inventory and trading rate
                x[path, i] = strategy_func(t)
                v[path, i] = self.compute_trading_rate(t, x[path, i], strategy_func)
                
                # Update price
                S[path, i+1] = S[path, i] + self.mu * dt + self.gamma * (x[path, i] - x[path, i+1]) + self.sigma * dW[i]
                
                # Compute execution price
                S_tilde[path, i] = S[path, i] - self.eta * v[path, i]
            
            # Compute final execution price
            v[path, -1] = 0  # No trading at final time
            S_tilde[path, -1] = S[path, -1]
            
            # Ensure final inventory matches target (if not, apply penalty)
            if abs(x[path, -1] - self.A) > 1e-6:
                print(f"Warning: Final inventory {x[path, -1]} does not match target {self.A}")
        
        # Compute P&L
        pnl = np.zeros(N_paths)
        for path in range(N_paths):
            # P&L from trading
            for i in range(N_steps):
                pnl[path] += S_tilde[path, i] * (x[path, i+1] - x[path, i])
            
            # Add penalty for final deviation
            pnl[path] -= self.beta * (x[path, -1] - self.A)**2
        
        # Compute utility
        utility = (1 - np.exp(-self.theta * pnl)) / self.theta if self.theta > 0 else pnl
        
        return {
            'time': time_grid,
            'price': S,
            'execution_price': S_tilde,
            'inventory': x,
            'trading_rate': v,
            'pnl': pnl,
            'utility': utility
        }

def compare_strategies(model, reference_strategies, N_steps=1000, N_paths=100):
    """
    Compare different reference strategies and their corresponding optimal strategies.
    
    Parameters:
    -----------
    model : OptimalOrderExecution
        The model object
    reference_strategies : dict
        Dictionary mapping strategy names to reference strategy functions
    N_steps : int
        Number of time steps in the simulation
    N_paths : int
        Number of price paths to simulate
        
    Returns:
    --------
    results : dict
        Dictionary containing comparison results
    """
    time_grid = np.linspace(0, model.T, N_steps+1)
    
    # Initialize results dictionary
    results = {
        'time': time_grid,
        'reference_strategies': {},
        'optimal_strategies': {},
        'simulation_results': {}
    }
    
    # For each reference strategy
    for name, ref_strategy in reference_strategies.items():
        # Store reference strategy
        results['reference_strategies'][name] = ref_strategy(time_grid) if callable(ref_strategy) else ref_strategy
        
        # Compute optimal strategy
        if name == 'IS':
            opt_strategy = model.is_strategy
        elif name == 'TC':
            opt_strategy = model.tc_strategy
        elif name == 'TWAP':
            opt_strategy = model.twap_strategy
        elif name == 'Endpoints-Only':
            R = reference_strategies[name]
            opt_strategy = lambda t: model.endpoints_only_strategy(t, R)
        elif name == 'Piecewise-Constant':
            Rs = reference_strategies[name]
            opt_strategy = lambda t: model.piecewise_constant_strategy(t, Rs)
        elif callable(ref_strategy):
            opt_strategy = lambda t: model.general_reference_strategy(t, ref_strategy)
        else:
            raise ValueError(f"Unknown reference strategy: {name}")
        
        # Store optimal strategy
        results['optimal_strategies'][name] = opt_strategy(time_grid)
        
        # Simulate price process and execution
        sim_results = model.simulate_price_process(opt_strategy, N_steps, N_paths)
        results['simulation_results'][name] = sim_results
    
    return results

def plot_strategy_comparison(results, path_idx=0, figsize=(15, 10)):
    """
    Plot the comparison of reference strategies and their optimal strategies.
    
    Parameters:
    -----------
    results : dict
        Dictionary containing comparison results
    path_idx : int
        Index of the price path to plot
    figsize : tuple
        Figure size
    """
    time_grid = results['time']
    
    # Create figure
    fig = plt.figure(figsize=figsize)
    
    # Inventory plot
    ax1 = fig.add_subplot(2, 2, 1)
    for name, opt_strategy in results['optimal_strategies'].items():
        ax1.plot(time_grid, opt_strategy, label=f"{name}")
    ax1.set_title('Optimal Inventory Trajectories')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Inventory')
    ax1.legend()
    ax1.grid(True)
    
    # Reference strategies plot
    ax2 = fig.add_subplot(2, 2, 2)
    for name, ref_strategy in results['reference_strategies'].items():
        if isinstance(ref_strategy, np.ndarray):
            ax2.plot(time_grid, ref_strategy, label=f"{name}")
    ax2.set_title('Reference Strategies')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Inventory')
    ax2.legend()
    ax2.grid(True)
    
    # Trading rate plot
    ax3 = fig.add_subplot(2, 2, 3)
    for name, sim_results in results['simulation_results'].items():
        ax3.plot(time_grid, sim_results['trading_rate'][path_idx], label=f"{name}")
    ax3.set_title('Trading Rates')
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Trading Rate')
    ax3.legend()
    ax3.grid(True)
    
    # Price plot
    ax4 = fig.add_subplot(2, 2, 4)
    for name, sim_results in results['simulation_results'].items():
        ax4.plot(time_grid, sim_results['price'][path_idx], label=f"{name}")
        ax4.plot(time_grid, sim_results['execution_price'][path_idx], '--', alpha=0.5)
    ax4.set_title('Price Paths')
    ax4.set_xlabel('Time')
    ax4.set_ylabel('Price')
    ax4.legend()
    ax4.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Plot P&L distribution
    plt.figure(figsize=(12, 6))
    for name, sim_results in results['simulation_results'].items():
        sns.histplot(sim_results['pnl'], kde=True, label=name)
    plt.title('P&L Distribution')
    plt.xlabel('P&L')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Plot Utility distribution
    plt.figure(figsize=(12, 6))
    for name, sim_results in results['simulation_results'].items():
        sns.histplot(sim_results['utility'], kde=True, label=name)
    plt.title('Utility Distribution')
    plt.xlabel('Utility')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print(f"{'Strategy':<20} {'Mean P&L':<15} {'Std P&L':<15} {'Mean Utility':<15} {'Std Utility':<15}")
    print("-" * 80)
    for name, sim_results in results['simulation_results'].items():
        print(f"{name:<20} {np.mean(sim_results['pnl']):<15.4f} {np.std(sim_results['pnl']):<15.4f} "
              f"{np.mean(sim_results['utility']):<15.4f} {np.std(sim_results['utility']):<15.4f}")

def endpoints_only_ref(t, T, x0, A, R):
    """Function to create endpoints-only reference strategy"""
    if np.isscalar(t):
        if t == 0:
            return x0
        elif t == T:
            return A
        else:
            return R
    else:
        result = np.ones_like(t) * R
        result[t == 0] = x0
        result[t == T] = A
        return result

def main():
    # Create model
    model = OptimalOrderExecution(
        x0=100,      # Initial inventory
        A=0,         # Target inventory
        T=1.0,       # Time horizon
        eta=10,      # Temporary price impact
        gamma=0.01,  # Permanent price impact
        sigma=0.5,   # Volatility
        beta=1000,   # Terminal penalty
        mu=0,        # Drift
        theta=0.002  # Risk aversion
    )
    
    # Define reference strategies
    reference_strategies = {
        'IS': model.is_strategy,
        'TC': model.tc_strategy,
        'TWAP': model.twap_strategy,
        'Endpoints-Only': 50,  # Reference level R
        'Piecewise-Constant': [80, 50, 20]  # Reference levels for 3 sub-intervals
    }
    
    # Compare strategies
    results = compare_strategies(model, reference_strategies, N_steps=100, N_paths=1000)
    
    # Plot comparison
    plot_strategy_comparison(results)
    
    # Now let's test with different values of kappa
    print("\nTesting with different values of kappa...")
    kappa_values = [0.1, 0.5, 2.0]
    kappa_results = []
    
    for kappa in kappa_values:
        # Adjust theta to get the desired kappa
        theta = 2 * model.eta * kappa**2 / model.sigma**2
        
        # Create model with new theta
        kappa_model = OptimalOrderExecution(
            x0=100, A=0, T=1.0, eta=10, gamma=0.01, sigma=0.5, 
            beta=1000, mu=0, theta=theta
        )
        
        # Verify kappa
        print(f"Theta: {theta:.6f}, Resulting kappa: {kappa_model.kappa:.6f}")
        
        # Define reference strategies
        reference_strategies = {
            'IS': kappa_model.is_strategy,
            'TC': kappa_model.tc_strategy,
            'TWAP': kappa_model.twap_strategy
        }
        
        # Compare strategies
        results = compare_strategies(kappa_model, reference_strategies, N_steps=100, N_paths=100)
        kappa_results.append((kappa, results))
    
    # Plot IS and TC trajectories for different kappa values
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    for kappa, results in kappa_results:
        plt.plot(results['time'], results['optimal_strategies']['IS'], label=f"IS, κ={kappa:.1f}")
    plt.title('Implementation Shortfall (IS) Orders')
    plt.xlabel('Time')
    plt.ylabel('Inventory')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    for kappa, results in kappa_results:
        plt.plot(results['time'], results['optimal_strategies']['TC'], label=f"TC, κ={kappa:.1f}")
    plt.title('Target Close (TC) Orders')
    plt.xlabel('Time')
    plt.ylabel('Inventory')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Test the affine structure with different reference points
    print("\nTesting the affine structure with different reference points...")
    R_values = [0, 50, 120]
    
    plt.figure(figsize=(12, 6))
    for R in R_values:
        t_values = np.linspace(0, model.T, 100)
        x_values = [model.endpoints_only_strategy(t, R) for t in t_values]
        plt.plot(t_values, x_values, label=f"R={R}")
    
    plt.title('Optimal Strategies with Different Reference Levels')
    plt.xlabel('Time')
    plt.ylabel('Inventory')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Test piecewise constant reference strategy
    print("\nTesting piecewise constant reference strategy...")
    
    # Different sets of reference levels
    piecewise_sets = [
        [80, 50, 20],  # Decreasing
        [20, 50, 80],  # Increasing
        [20, 80, 20]   # Non-monotonic
    ]
    
    plt.figure(figsize=(12, 6))
    for i, Rs in enumerate(piecewise_sets):
        t_values = np.linspace(0, model.T, 100)
        x_values = [model.piecewise_constant_strategy(t, Rs) for t in t_values]
        
        # Create reference strategy for plotting
        ref_values = np.zeros_like(t_values)
        dt = model.T / len(Rs)
        for j, R in enumerate(Rs):
            ref_values[(t_values >= j*dt) & (t_values < (j+1)*dt)] = R
        ref_values[t_values == 0] = model.x0
        ref_values[t_values == model.T] = model.A
        
        plt.subplot(1, 3, i+1)
        plt.plot(t_values, x_values, label='Optimal')
        plt.step(t_values, ref_values, 'r--', label='Reference', where='post')
        plt.title(f'Reference Set {i+1}')
        plt.xlabel('Time')
        plt.ylabel('Inventory')
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Performance comparison with stress testing
    print("\nPerforming stress testing...")
    
    # Base model
    base_model = OptimalOrderExecution(
        x0=100, A=0, T=1.0, eta=10, gamma=0.01, sigma=0.5, 
        beta=1000, mu=0, theta=0.002
    )
    
    # Define reference strategies
    reference_strategies = {
        'IS': base_model.is_strategy,
        'TC': base_model.tc_strategy,
        'TWAP': base_model.twap_strategy
    }
    
    # Baseline results
    base_results = compare_strategies(base_model, reference_strategies, N_steps=100, N_paths=500)
    
    # Print summary statistics for baseline
    print("\nBaseline Summary Statistics:")
    print(f"{'Strategy':<20} {'Mean P&L':<15} {'Std P&L':<15} {'Mean Utility':<15} {'Std Utility':<15}")
    print("-" * 80)
    for name, sim_results in base_results['simulation_results'].items():
        print(f"{name:<20} {np.mean(sim_results['pnl']):<15.4f} {np.std(sim_results['pnl']):<15.4f} "
              f"{np.mean(sim_results['utility']):<15.4f} {np.std(sim_results['utility']):<15.4f}")
    
    # Stress scenarios
    stress_scenarios = [
        ('High Volatility', {'sigma': 2.0}),
        ('High Permanent Impact', {'gamma': 0.1}),
        ('High Temporary Impact', {'eta': 50}),
        ('High Risk Aversion', {'theta': 0.01})
    ]
    
    # Run stress tests
    for scenario_name, params in stress_scenarios:
        print(f"\nStress Test: {scenario_name}")
        
        # Create stressed model
        stressed_params = {
            'x0': 100, 'A': 0, 'T': 1.0, 'eta': 10, 'gamma': 0.01, 
            'sigma': 0.5, 'beta': 1000, 'mu': 0, 'theta': 0.002
        }
        stressed_params.update(params)
        
        stressed_model = OptimalOrderExecution(**stressed_params)
        
        # Define reference strategies
        reference_strategies = {
            'IS': stressed_model.is_strategy,
            'TC': stressed_model.tc_strategy,
            'TWAP': stressed_model.twap_strategy
        }
        
        # Compare strategies under stress
        stressed_results = compare_strategies(stressed_model, reference_strategies, N_steps=100, N_paths=500)
        
        # Print summary statistics for stress test
        print(f"{'Strategy':<20} {'Mean P&L':<15} {'Std P&L':<15} {'Mean Utility':<15} {'Std Utility':<15}")
        print("-" * 80)
        for name, sim_results in stressed_results['simulation_results'].items():
            print(f"{name:<20} {np.mean(sim_results['pnl']):<15.4f} {np.std(sim_results['pnl']):<15.4f} "
                  f"{np.mean(sim_results['utility']):<15.4f} {np.std(sim_results['utility']):<15.4f}")
        
        # Plot comparison of P&L distributions under stress
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 2, 1)
        for name, sim_results in stressed_results['simulation_results'].items():
            sns.kdeplot(sim_results['pnl'], label=name)
        plt.title(f'P&L Distribution - {scenario_name}')
        plt.xlabel('P&L')
        plt.ylabel('Density')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        for name, sim_results in stressed_results['simulation_results'].items():
            sns.kdeplot(sim_results['utility'], label=name)
        plt.title(f'Utility Distribution - {scenario_name}')
        plt.xlabel('Utility')
        plt.ylabel('Density')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()