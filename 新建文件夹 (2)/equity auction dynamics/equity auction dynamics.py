import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import pandas as pd
from matplotlib.animation import FuncAnimation

class LatentLiquidityAuctionModel:
    def __init__(self, T=300, a=6.77, b=0.0058, xr=0.003, k=5.1, w=0.969, 
                 Cr=0.93, gamma_r=3.1, t0_r=210, 
                 nu_l=0.023, x0=0.0032, m=0.016,
                 Dr=2.2e-8, Dl=4.8e-9):
        """
        Initialize the latent liquidity auction model.
        
        Parameters:
        -----------
        T : int
            Auction duration in seconds
        a, b : float
            Parameters of the latent order book (linear slope and floor)
        xr : float
            Price scale for fast agents
        k : float
            Ratio of slow to fast price scales
        w : float
            Weight of fast agents
        Cr : float
            Submission rate scaling
        gamma_r : float
            Submission deadline offset
        t0_r : float
            Submission rate cutoff time
        nu_l : float
            Cancellation rate
        x0 : float
            Threshold for negative prices
        m : float
            Multiplicative factor for prices below -x0
        Dr, Dl : float
            Diffusion coefficients for revealed and latent order books
        """
        # Auction parameters
        self.T = T  # Auction duration in seconds
        
        # Latent order book parameters
        self.a = a
        self.b = b
        
        # Submission rate parameters
        self.xr = xr
        self.k = k
        self.w = w
        self.Cr = Cr
        self.gamma_r = gamma_r
        self.t0_r = t0_r
        
        # Cancellation rate parameter
        self.nu_l = nu_l
        
        # Threshold parameters for negative prices
        self.x0 = x0
        self.m = m
        
        # Diffusion coefficients
        self.Dr = Dr
        self.Dl = Dl
        
        # Calculate derived parameters for continuity at x = 0 and x = -x0
        self.A_star, self.xr_star = self._calculate_continuity_params()
        
    def _calculate_continuity_params(self):
        """Calculate parameters to ensure continuity at x = 0 and x = -x0"""
        # At x = 0, ensure continuity with the weighted sum of exponentials
        exp_value_at_zero = self.w + (1 - self.w)
        A_star = exp_value_at_zero
        
        # At x = -x0, ensure continuity with the constant value
        # Solve A_star * exp(x0/xr_star) = m * self.Cr / self.gamma_r
        target_value = self.m * self.Cr / self.gamma_r
        xr_star = self.x0 / np.log(target_value / A_star)
        
        return A_star, xr_star
    
    def latent_density(self, x):
        """Calculate the initial latent order density"""
        return np.maximum(self.a * x + self.b, self.b)
    
    def submission_rate(self, x, t):
        """Calculate the submission rate (nu_r * Gamma_r)"""
        if x >= 0:
            # For positive prices: weighted sum of two exponentials
            fast_term = self.w * self.Cr / (self.gamma_r + self.T - max(t, self.t0_r)) * np.exp(-x / self.xr)
            slow_term = (1 - self.w) * self.Cr / (self.gamma_r + self.T - self.t0_r) * np.exp(-x / (self.k * self.xr))
            return fast_term + slow_term
        elif x >= -self.x0:
            # For prices between -x0 and 0: single exponential
            return self.A_star * np.exp(x / self.xr_star)
        else:
            # For prices below -x0: constant
            return self.m * self.Cr / self.gamma_r
    
    def cancellation_rate(self, x, t):
        """Calculate the cancellation rate (nu_l * Gamma_l)"""
        return self.nu_l  # Constant cancellation rate
    
    def simulate(self, x_grid, t_grid):
        """
        Simulate the auction dynamics using numerical PDE solver.
        
        Parameters:
        -----------
        x_grid : array-like
            Grid points in price space
        t_grid : array-like
            Time points for the simulation
            
        Returns:
        --------
        rho_r : 2D array
            Revealed order density at each (x, t) point
        rho_l : 2D array
            Latent order density at each (x, t) point
        """
        dx = x_grid[1] - x_grid[0]
        nx = len(x_grid)
        nt = len(t_grid)
        
        # Initialize arrays
        rho_r = np.zeros((nt, nx))
        rho_l = np.zeros((nt, nx))
        
        # Initial conditions
        rho_r[0, :] = 0  # No revealed orders initially
        rho_l[0, :] = self.latent_density(x_grid)
        
        # Precompute submission and cancellation rates at each time
        sub_rates = np.zeros((nt, nx))
        for i, t in enumerate(t_grid):
            for j, x in enumerate(x_grid):
                sub_rates[i, j] = self.submission_rate(x, t)
        
        cancel_rate = self.nu_l  # Constant cancellation rate
        
        # Finite difference parameters
        r_Dr = self.Dr * dt / (dx**2)
        r_Dl = self.Dl * dt / (dx**2)
        
        # Time stepping
        for i in range(1, nt):
            t = t_grid[i]
            dt = t - t_grid[i-1]
            
            # Explicit finite difference for diffusion
            for j in range(1, nx-1):
                # Diffusion terms
                diff_r = r_Dr * (rho_r[i-1, j+1] - 2*rho_r[i-1, j] + rho_r[i-1, j-1])
                diff_l = r_Dl * (rho_l[i-1, j+1] - 2*rho_l[i-1, j] + rho_l[i-1, j-1])
                
                # Reaction terms
                sub = sub_rates[i-1, j] * rho_l[i-1, j] * dt
                cancel = cancel_rate * rho_r[i-1, j] * dt
                
                # Update
                rho_r[i, j] = rho_r[i-1, j] + diff_r + sub - cancel
                rho_l[i, j] = rho_l[i-1, j] + diff_l - sub + cancel
            
            # Boundary conditions (no flux)
            rho_r[i, 0] = rho_r[i, 1]
            rho_r[i, -1] = rho_r[i, -2]
            rho_l[i, 0] = rho_l[i, 1]
            rho_l[i, -1] = rho_l[i, -2]
        
        return rho_r, rho_l

    def pde_system(self, t, y, x_grid):
        """
        Define the PDE system for the solver.
        
        Parameters:
        -----------
        t : float
            Current time
        y : array
            Current state (first half is rho_r, second half is rho_l)
        x_grid : array
            Grid points in price space
            
        Returns:
        --------
        dydt : array
            Time derivatives of the state variables
        """
        nx = len(x_grid)
        dx = x_grid[1] - x_grid[0]
        
        # Extract rho_r and rho_l from the state vector
        rho_r = y[:nx]
        rho_l = y[nx:]
        
        # Initialize derivatives
        drho_r_dt = np.zeros_like(rho_r)
        drho_l_dt = np.zeros_like(rho_l)
        
        # Interior points: diffusion using central differences
        for i in range(1, nx-1):
            x = x_grid[i]
            
            # Diffusion terms
            d2rho_r_dx2 = (rho_r[i+1] - 2*rho_r[i] + rho_r[i-1]) / (dx**2)
            d2rho_l_dx2 = (rho_l[i+1] - 2*rho_l[i] + rho_l[i-1]) / (dx**2)
            
            # Reaction terms
            sub_rate = self.submission_rate(x, t)
            cancel_rate = self.cancellation_rate(x, t)
            
            # Update derivatives
            drho_r_dt[i] = self.Dr * d2rho_r_dx2 + sub_rate * rho_l[i] - cancel_rate * rho_r[i]
            drho_l_dt[i] = self.Dl * d2rho_l_dx2 - sub_rate * rho_l[i] + cancel_rate * rho_r[i]
        
        # Boundary conditions (no flux)
        drho_r_dt[0] = 0
        drho_r_dt[-1] = 0
        drho_l_dt[0] = 0
        drho_l_dt[-1] = 0
        
        # Combine derivatives into a single array
        dydt = np.concatenate([drho_r_dt, drho_l_dt])
        return dydt
    
    def simulate_with_solver(self, x_grid, t_grid):
        """
        Simulate the auction dynamics using SciPy's ODE solver.
        
        Parameters:
        -----------
        x_grid : array-like
            Grid points in price space
        t_grid : array-like
            Time points for the simulation
            
        Returns:
        --------
        rho_r : 2D array
            Revealed order density at each (x, t) point
        rho_l : 2D array
            Latent order density at each (x, t) point
        """
        nx = len(x_grid)
        
        # Initial conditions
        rho_r_0 = np.zeros(nx)
        rho_l_0 = self.latent_density(x_grid)
        y0 = np.concatenate([rho_r_0, rho_l_0])
        
        # Solve the PDE system
        solution = solve_ivp(
            lambda t, y: self.pde_system(t, y, x_grid),
            [t_grid[0], t_grid[-1]],
            y0,
            method='RK45',
            t_eval=t_grid,
            rtol=1e-4,
            atol=1e-6
        )
        
        # Extract solutions
        rho_r = solution.y[:nx, :].T
        rho_l = solution.y[nx:, :].T
        
        return rho_r, rho_l

# Create a simulation
def run_simulation():
    # Set up the grid
    x_min, x_max = -0.02, 0.02  # ±2% in log price
    nx = 100
    x_grid = np.linspace(x_min, x_max, nx)
    
    # Time grid
    t_max = 300  # 5 minutes in seconds
    nt = 61
    t_grid = np.linspace(0, t_max, nt)
    
    # Create model with parameters from the paper
    model = LatentLiquidityAuctionModel()
    
    # Simulate
    rho_r, rho_l = model.simulate(x_grid, t_grid)
    
    return x_grid, t_grid, rho_r, rho_l, model

def plot_results(x_grid, t_grid, rho_r, rho_l, model):
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Order book at different times
    times_to_plot = [0, 100, 200, 290]  # in seconds
    time_indices = [np.abs(t_grid - t).argmin() for t in times_to_plot]
    
    for i, idx in enumerate(time_indices):
        t = t_grid[idx]
        label = f't = {t:.0f}s'
        axes[0, 0].plot(x_grid*100, rho_r[idx], label=label)
    
    axes[0, 0].set_xlabel('Log Price Relative to Indicative Price (%)')
    axes[0, 0].set_ylabel('Revealed Order Density')
    axes[0, 0].set_title('Revealed Order Book at Different Times')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Plot 2: Time evolution at different prices
    prices_to_plot = [-0.01, -0.005, 0, 0.005, 0.01]  # in log price
    price_indices = [np.abs(x_grid - p).argmin() for p in prices_to_plot]
    
    for i, idx in enumerate(price_indices):
        x = x_grid[idx]
        label = f'x = {x*100:.1f}%'
        axes[0, 1].plot(t_grid, rho_r[:, idx], label=label)
    
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('Revealed Order Density')
    axes[0, 1].set_title('Time Evolution at Different Prices')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Plot 3: Submission rate
    t_vals = [0, 100, 200, 290]
    for t in t_vals:
        sub_rates = [model.submission_rate(x, t) for x in x_grid]
        axes[1, 0].plot(x_grid*100, sub_rates, label=f't = {t:.0f}s')
    
    axes[1, 0].set_xlabel('Log Price Relative to Indicative Price (%)')
    axes[1, 0].set_ylabel('Submission Rate')
    axes[1, 0].set_title('Submission Rate at Different Times')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Plot 4: Heatmap of revealed order book
    im = axes[1, 1].imshow(
        rho_r.T,
        aspect='auto',
        origin='lower',
        extent=[t_grid[0], t_grid[-1], x_grid[0]*100, x_grid[-1]*100],
        cmap='viridis'
    )
    
    axes[1, 1].set_xlabel('Time (s)')
    axes[1, 1].set_ylabel('Log Price Relative to Indicative Price (%)')
    axes[1, 1].set_title('Revealed Order Book Evolution')
    plt.colorbar(im, ax=axes[1, 1], label='Revealed Order Density')
    
    plt.tight_layout()
    plt.show()

def create_animation(x_grid, t_grid, rho_r):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    line, = ax.plot(x_grid*100, rho_r[0], 'b-', lw=2)
    
    ax.set_xlabel('Log Price Relative to Indicative Price (%)')
    ax.set_ylabel('Revealed Order Density')
    ax.set_title('Auction Order Book Evolution')
    ax.grid(True)
    
    # Set y-axis limits based on the maximum value in rho_r
    ax.set_ylim(0, np.max(rho_r) * 1.1)
    
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
    
    def update(frame):
        line.set_ydata(rho_r[frame])
        remaining = t_grid[-1] - t_grid[frame]
        time_text.set_text(f'Time: {t_grid[frame]:.0f}s (T-{remaining:.0f}s)')
        return line, time_text
    
    ani = FuncAnimation(fig, update, frames=len(t_grid), blit=True)
    plt.tight_layout()
    
    return ani

# Run the simulation and plot results
x_grid, t_grid, rho_r, rho_l, model = run_simulation()
plot_results(x_grid, t_grid, rho_r, rho_l, model)

# Create animation
ani = create_animation(x_grid, t_grid, rho_r)


def analyze_indicative_price_dynamics(model, n_simulations=100, duration=300):
    """
    Simulate the indicative price process and analyze its statistical properties.
    
    Parameters:
    -----------
    model : LatentLiquidityAuctionModel
        The auction model
    n_simulations : int
        Number of simulations to run
    duration : int
        Duration of each simulation in seconds
        
    Returns:
    --------
    results : dict
        Dictionary with analysis results
    """
    # Time grid
    t_grid = np.linspace(0, duration, duration+1)
    
    # Store the simulated price paths
    price_paths = np.zeros((n_simulations, len(t_grid)))
    
    # Simulate multiple price paths
    for i in range(n_simulations):
        # Initial price (normalized to 1.0)
        p0 = 1.0
        price = p0
        price_path = [price]
        
        # Simple model for sub-diffusive price process
        # Using the decreasing volatility observed in the paper
        for t in range(1, len(t_grid)):
            # Volatility decreases as auction approaches
            if t < 60:  # First minute
                vol = 1e-3
            elif t < 120:  # Second minute
                vol = 5e-4
            elif t < 240:  # Third and fourth minutes
                vol = 2e-4
            else:  # Fifth minute
                vol = 1e-4
            
            # Sub-diffusive price increments
            # Price changes are smaller than expected from normal diffusion
            price *= np.exp(np.random.normal(0, vol * np.sqrt(1/(t+1))))
            price_path.append(price)
        
        price_paths[i] = price_path
    
    # Calculate mean absolute returns at each time step
    mean_abs_returns = np.zeros(len(t_grid)-1)
    for t in range(len(t_grid)-1):
        returns = np.log(price_paths[:, t+1] / price_paths[:, t])
        mean_abs_returns[t] = np.mean(np.abs(returns))
    
    # Calculate variance of log returns for different time scales
    var_by_scale = []
    scales = [1, 2, 4, 8, 16, 32, 64]
    
    for scale in scales:
        vars_at_scale = []
        for t in range(0, len(t_grid)-scale, scale):
            log_returns = np.log(price_paths[:, t+scale] / price_paths[:, t])
            vars_at_scale.append(np.var(log_returns))
        
        var_by_scale.append(np.mean(vars_at_scale))
    
    # Estimate Hurst exponent using variance method
    log_scales = np.log(scales)
    log_vars = np.log(var_by_scale)
    
    # Linear regression to find Hurst exponent
    slope, _ = np.polyfit(log_scales, log_vars, 1)
    H = slope / 2  # Hurst exponent
    
    results = {
        'price_paths': price_paths,
        'mean_abs_returns': mean_abs_returns,
        'var_by_scale': var_by_scale,
        'scales': scales,
        'H': H
    }
    
    return results

def plot_price_analysis(results):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Sample price paths
    n_paths_to_plot = 5
    for i in range(n_paths_to_plot):
        axes[0, 0].plot(results['price_paths'][i])
    
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Indicative Price')
    axes[0, 0].set_title('Sample Indicative Price Paths')
    axes[0, 0].grid(True)
    
    # Plot 2: Mean absolute returns
    axes[0, 1].plot(results['mean_abs_returns'])
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('Mean Absolute Return')
    axes[0, 1].set_title('Mean Absolute Returns Over Time')
    axes[0, 1].set_yscale('log')
    axes[0, 1].grid(True)
    
    # Plot 3: Variance by time scale (log-log)
    axes[1, 0].loglog(results['scales'], results['var_by_scale'], 'o-')
    
    # Add line with slope 2H
    H = results['H']
    x_range = np.array([min(results['scales']), max(results['scales'])])
    y_range = results['var_by_scale'][0] * (x_range / results['scales'][0]) ** (2 * H)
    axes[1, 0].loglog(x_range, y_range, 'r--', label=f'Slope = {2*H:.2f}')
    
    axes[1, 0].set_xlabel('Time Scale')
    axes[1, 0].set_ylabel('Variance')
    axes[1, 0].set_title(f'Variance by Time Scale (H = {H:.3f})')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Plot 4: Histogram of returns at different time scales
    for scale in [1, 16, 64]:
        returns = []
        for t in range(0, len(results['price_paths'][0])-scale, scale):
            log_returns = np.log(results['price_paths'][:, t+scale] / results['price_paths'][:, t])
            returns.extend(log_returns)
        
        axes[1, 1].hist(returns, bins=50, alpha=0.3, label=f'Scale = {scale}')
    
    axes[1, 1].set_xlabel('Log Return')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Distribution of Returns at Different Time Scales')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.show()

# Run the price analysis
price_results = analyze_indicative_price_dynamics(model)
plot_price_analysis(price_results)


def test_trading_strategy(model, n_simulations=100, duration=300):
    """
    Test a trading strategy based on the paper's insights.
    
    Strategy:
    1. Submit limit orders away from the indicative price early in the auction
    2. As the auction approaches, move orders closer to the indicative price
    3. Compare with a baseline strategy of submitting orders at fixed prices
    
    Parameters:
    -----------
    model : LatentLiquidityAuctionModel
        The auction model
    n_simulations : int
        Number of simulations to run
    duration : int
        Duration of each simulation in seconds
        
    Returns:
    --------
    results : dict
        Dictionary with strategy performance metrics
    """
    # Time grid
    t_grid = np.linspace(0, duration, duration+1)
    
    # Store results
    strategy_returns = np.zeros(n_simulations)
    baseline_returns = np.zeros(n_simulations)
    
    # Simulate multiple auction scenarios
    for i in range(n_simulations):
        # Simulate price path
        p0 = 100.0  # Initial price
        price = p0
        price_path = [price]
        
        # Simple model for sub-diffusive price process
        for t in range(1, len(t_grid)):
            # Volatility decreases as auction approaches
            if t < 60:
                vol = 1e-3
            elif t < 120:
                vol = 5e-4
            elif t < 240:
                vol = 2e-4
            else:
                vol = 1e-4
            
            price *= np.exp(np.random.normal(0, vol * np.sqrt(1/(t+1))))
            price_path.append(price)
        
        # Final auction price
        auction_price = price_path[-1]
        
        # Strategy 1: Dynamic placement based on time to auction
        # Place buy orders at decreasing distance from indicative price
        strategy_distance = 0.005  # Start 0.5% away from indicative
        
        # Order placement times (every minute)
        placement_times = [0, 60, 120, 180, 240]
        
        # Order distances for each placement (getting closer to indicative price)
        distances = [0.005, 0.004, 0.003, 0.002, 0.001]
        
        # Calculate expected execution and P&L
        strategy_executed = False
        strategy_execution_price = 0
        
        for j, t in enumerate(placement_times):
            order_price = price_path[t] * (1 - distances[j])  # Buy order below indicative
            
            if order_price >= auction_price:
                # Order executed at auction price
                strategy_executed = True
                strategy_execution_price = auction_price
                break
        
        # Strategy 2: Baseline - fixed placement at beginning
        baseline_price = price_path[0] * 0.995  # Fixed 0.5% below initial indicative
        baseline_executed = baseline_price >= auction_price
        
        # Calculate returns
        if strategy_executed:
            # Profit is difference between true value (p0) and execution price
            strategy_returns[i] = (p0 / strategy_execution_price - 1) * 100
        else:
            strategy_returns[i] = 0
            
        if baseline_executed:
            baseline_returns[i] = (p0 / auction_price - 1) * 100
        else:
            baseline_returns[i] = 0
    
    # Calculate metrics
    results = {
        'strategy_returns': strategy_returns,
        'baseline_returns': baseline_returns,
        'strategy_mean': np.mean(strategy_returns),
        'baseline_mean': np.mean(baseline_returns),
        'strategy_std': np.std(strategy_returns),
        'baseline_std': np.std(baseline_returns),
        'strategy_win_rate': np.mean(strategy_returns > 0),
        'baseline_win_rate': np.mean(baseline_returns > 0),
        'strategy_execution_rate': np.mean(strategy_returns != 0),
        'baseline_execution_rate': np.mean(baseline_returns != 0)
    }
    
    return results

def plot_strategy_results(results):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Return distributions
    axes[0].hist(results['strategy_returns'], bins=30, alpha=0.5, label='Dynamic Strategy')
    axes[0].hist(results['baseline_returns'], bins=30, alpha=0.5, label='Fixed Strategy')
    axes[0].set_xlabel('Return (%)')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Distribution of Returns')
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot 2: Metrics comparison
    metrics = ['Mean Return', 'Win Rate', 'Execution Rate']
    strategy_values = [
        results['strategy_mean'],
        results['strategy_win_rate'] * 100,
        results['strategy_execution_rate'] * 100
    ]
    baseline_values = [
        results['baseline_mean'],
        results['baseline_win_rate'] * 100,
        results['baseline_execution_rate'] * 100
    ]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    axes[1].bar(x - width/2, strategy_values, width, label='Dynamic Strategy')
    axes[1].bar(x + width/2, baseline_values, width, label='Fixed Strategy')
    
    axes[1].set_ylabel('Value')
    axes[1].set_title('Strategy Performance Metrics')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(metrics)
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Print detailed results
    print(f"Dynamic Strategy:")
    print(f"  Mean Return: {results['strategy_mean']:.2f}%")
    print(f"  Return Std Dev: {results['strategy_std']:.2f}%")
    print(f"  Win Rate: {results['strategy_win_rate']*100:.1f}%")
    print(f"  Execution Rate: {results['strategy_execution_rate']*100:.1f}%")
    print()
    print(f"Fixed Strategy:")
    print(f"  Mean Return: {results['baseline_mean']:.2f}%")
    print(f"  Return Std Dev: {results['baseline_std']:.2f}%")
    print(f"  Win Rate: {results['baseline_win_rate']*100:.1f}%")
    print(f"  Execution Rate: {results['baseline_execution_rate']*100:.1f}%")

# Test the trading strategies
strategy_results = test_trading_strategy(model)
plot_strategy_results(strategy_results)


