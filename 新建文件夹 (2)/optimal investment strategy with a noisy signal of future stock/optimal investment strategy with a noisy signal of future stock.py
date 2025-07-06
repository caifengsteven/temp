import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from tqdm import tqdm

class NoisySignalInvestment:
    def __init__(self, S0=100, mu=0.1, sigma=0.2, gamma=0.7, Lambda=0.1, alpha=1, T=1, dt=0.01):
        """
        Initialize the model parameters
        
        Parameters:
        - S0: initial stock price
        - mu: drift of the stock price
        - sigma: volatility of the stock price
        - gamma: correlation parameter between the signal and the stock price
        - Lambda: temporary price impact parameter
        - alpha: risk aversion parameter
        - T: time horizon
        - dt: time step
        """
        self.S0 = S0
        self.mu = mu
        self.sigma = sigma
        self.gamma = gamma
        self.gamma_bar = np.sqrt(1 - gamma**2)
        self.Lambda = Lambda
        self.alpha = alpha
        self.T = T
        self.dt = dt
        self.time_grid = np.arange(0, T + dt, dt)
        self.n_steps = len(self.time_grid)
        self.rho = alpha * sigma**2 / Lambda
        
    def tau_function(self, t):
        """
        Time shift function that determines the peek-ahead horizon
        Default implementation: constant peek-ahead period of 0.2*T
        """
        peek_ahead = 0.2 * self.T
        return min(t + peek_ahead, self.T)
    
    def tau_inverse(self, t):
        """
        Right-continuous inverse of tau
        """
        return max(0, t - 0.2 * self.T)
    
    def upsilon_function(self, t, h):
        """
        Calculate the Upsilon function as defined in the paper
        """
        rho = self.rho
        gamma_bar = self.gamma_bar
        
        return (gamma_bar * np.cosh(gamma_bar * np.sqrt(rho) * h) 
                + np.tanh(np.sqrt(rho) * (self.T - self.tau_function(t))) 
                * np.sinh(gamma_bar * np.sqrt(rho) * h))
    
    def upsilon_derivative(self, t, h):
        """
        Calculate the derivative of the Upsilon function with respect to h
        """
        rho = self.rho
        gamma_bar = self.gamma_bar
        
        return (gamma_bar**2 * np.sqrt(rho) * np.sinh(gamma_bar * np.sqrt(rho) * h)
                + np.tanh(np.sqrt(rho) * (self.T - self.tau_function(t)))
                * gamma_bar * np.sqrt(rho) * np.cosh(gamma_bar * np.sqrt(rho) * h))
    
    def generate_paths(self, n_paths=1):
        """
        Generate stock price paths and signals
        
        Returns:
        - S: stock price paths
        - W: first Brownian motion paths
        - W_prime: second Brownian motion paths (the one that's partially known)
        """
        # Generate Brownian motions
        dW = np.random.normal(0, np.sqrt(self.dt), (n_paths, self.n_steps))
        dW_prime = np.random.normal(0, np.sqrt(self.dt), (n_paths, self.n_steps))
        
        # Integrate to get Brownian paths
        W = np.cumsum(dW, axis=1)
        W_prime = np.cumsum(dW_prime, axis=1)
        
        # Generate stock price paths
        S = np.zeros((n_paths, self.n_steps))
        S[:, 0] = self.S0
        
        for i in range(1, self.n_steps):
            S[:, i] = (S[:, 0] + self.mu * self.time_grid[i] 
                       + self.sigma * (self.gamma * W_prime[:, i] + self.gamma_bar * W[:, i]))
        
        return S, W, W_prime
    
    def project_prices(self, S, W_prime, t_idx):
        """
        Calculate the risk- and liquidity-weighted projection of prices S_bar
        """
        t = self.time_grid[t_idx]
        tau_t = self.tau_function(t)
        tau_t_idx = int(tau_t / self.dt)
        
        # Current stock price
        St = S[t_idx]
        
        # Time window for signal integration
        signal_window = int((tau_t - t) / self.dt)
        
        if signal_window <= 0:
            return St
        
        # Calculate S_hat for different horizons
        S_hat_values = []
        weights = []
        
        for h_idx in range(signal_window + 1):
            h = h_idx * self.dt
            
            # For the last value, use boundary condition
            if h_idx == signal_window:
                # S_hat for peek-ahead horizon
                future_idx = min(t_idx + h_idx, len(W_prime) - 1)
                S_hat = (St + self.mu * self.gamma**2 * h 
                         + self.sigma * self.gamma * (W_prime[future_idx] - W_prime[t_idx]))
                
                weight = self.upsilon_function(t, 0)
                S_hat_values.append(S_hat)
                weights.append(weight)
            else:
                # S_hat for intermediate horizons
                future_idx = t_idx + h_idx
                S_hat = (St + self.mu * self.gamma**2 * h 
                         + self.sigma * self.gamma * (W_prime[future_idx] - W_prime[t_idx]))
                
                weight = self.upsilon_derivative(t, h)
                S_hat_values.append(S_hat)
                weights.append(weight)
        
        # Calculate weighted average
        total_weight = sum(weights)
        S_bar = sum(s * w for s, w in zip(S_hat_values, weights)) / total_weight
        
        return S_bar
    
    def optimal_strategy(self, S, W_prime, initial_position=0):
        """
        Calculate the optimal trading strategy
        
        Parameters:
        - S: stock price path
        - W_prime: signal path
        - initial_position: initial stock position
        
        Returns:
        - phi: optimal trading rates
        - Phi: optimal positions
        - PnL: profit and loss
        """
        phi = np.zeros(self.n_steps)
        Phi = np.zeros(self.n_steps)
        PnL = np.zeros(self.n_steps)
        
        # Initial position
        Phi[0] = initial_position
        
        for t_idx in range(self.n_steps - 1):
            t = self.time_grid[t_idx]
            tau_t = self.tau_function(t)
            
            # Calculate S_bar (projected price)
            S_bar = self.project_prices(S, W_prime, t_idx)
            
            # Calculate optimal trading rate (equation 3.1)
            upsilon_ratio = self.upsilon_derivative(t, tau_t - t) / self.upsilon_function(t, tau_t - t)
            merton_ratio = self.mu / (self.alpha * self.sigma**2)
            
            phi[t_idx] = (1 / self.Lambda) * (S_bar - S[t_idx]) + upsilon_ratio * (merton_ratio - Phi[t_idx])
            
            # Update position
            Phi[t_idx + 1] = Phi[t_idx] + phi[t_idx] * self.dt
            
            # Calculate PnL increment
            PnL[t_idx + 1] = PnL[t_idx] + Phi[t_idx] * (S[t_idx + 1] - S[t_idx]) - (self.Lambda / 2) * phi[t_idx]**2 * self.dt
        
        return phi, Phi, PnL
    
    def compare_strategies(self, n_paths=100):
        """
        Compare the optimal strategy with benchmark strategies
        
        Returns:
        - results: dictionary of performance metrics
        """
        results = {
            'optimal': [],
            'no_signal': [],
            'buy_and_hold': []
        }
        
        for _ in tqdm(range(n_paths), desc="Simulating paths"):
            # Generate a single price path and signals
            S, W, W_prime = self.generate_paths(n_paths=1)
            S, W, W_prime = S[0], W[0], W_prime[0]
            
            # Optimal strategy with signal
            _, _, PnL_optimal = self.optimal_strategy(S, W_prime)
            results['optimal'].append(PnL_optimal[-1])
            
            # Strategy without signal (just Merton ratio)
            merton_ratio = self.mu / (self.alpha * self.sigma**2)
            phi_no_signal = np.zeros(self.n_steps)
            Phi_no_signal = np.zeros(self.n_steps)
            PnL_no_signal = np.zeros(self.n_steps)
            
            for t_idx in range(self.n_steps - 1):
                phi_no_signal[t_idx] = (merton_ratio - Phi_no_signal[t_idx]) / self.Lambda
                Phi_no_signal[t_idx + 1] = Phi_no_signal[t_idx] + phi_no_signal[t_idx] * self.dt
                PnL_no_signal[t_idx + 1] = (PnL_no_signal[t_idx] + Phi_no_signal[t_idx] * (S[t_idx + 1] - S[t_idx]) 
                                           - (self.Lambda / 2) * phi_no_signal[t_idx]**2 * self.dt)
            
            results['no_signal'].append(PnL_no_signal[-1])
            
            # Buy and hold strategy (static Merton ratio)
            Phi_bnh = np.ones(self.n_steps) * merton_ratio
            PnL_bnh = np.zeros(self.n_steps)
            
            for t_idx in range(self.n_steps - 1):
                PnL_bnh[t_idx + 1] = PnL_bnh[t_idx] + Phi_bnh[t_idx] * (S[t_idx + 1] - S[t_idx])
            
            results['buy_and_hold'].append(PnL_bnh[-1])
        
        # Calculate statistics
        for strategy in results:
            results[strategy] = {
                'mean': np.mean(results[strategy]),
                'std': np.std(results[strategy]),
                'sharpe': np.mean(results[strategy]) / np.std(results[strategy]),
                'min': np.min(results[strategy]),
                'max': np.max(results[strategy])
            }
        
        return results
    
    def plot_single_simulation(self):
        """
        Run a single simulation and plot the results
        """
        # Generate a single price path and signals
        S, W, W_prime = self.generate_paths(n_paths=1)
        S, W, W_prime = S[0], W[0], W_prime[0]
        
        # Calculate optimal strategy
        phi, Phi, PnL = self.optimal_strategy(S, W_prime)
        
        # Plot results
        fig, axs = plt.subplots(4, 1, figsize=(12, 16), sharex=True)
        
        # Stock price
        axs[0].plot(self.time_grid, S)
        axs[0].set_title('Stock Price')
        axs[0].set_ylabel('Price')
        axs[0].grid(True)
        
        # Trading rate (phi)
        axs[1].plot(self.time_grid, phi)
        axs[1].set_title('Trading Rate (phi)')
        axs[1].set_ylabel('Rate')
        axs[1].grid(True)
        
        # Position (Phi)
        axs[2].plot(self.time_grid, Phi)
        axs[2].set_title('Position (Phi)')
        axs[2].set_ylabel('Shares')
        axs[2].grid(True)
        
        # PnL
        axs[3].plot(self.time_grid, PnL)
        axs[3].set_title('Profit and Loss')
        axs[3].set_ylabel('PnL')
        axs[3].set_xlabel('Time')
        axs[3].grid(True)
        
        plt.tight_layout()
        plt.savefig('noisy_signal_simulation.png')
        plt.show()
        
        return S, phi, Phi, PnL
    
    def analyze_gamma_impact(self, gamma_values=None):
        """
        Analyze the impact of the signal quality parameter gamma
        """
        if gamma_values is None:
            gamma_values = [0.1, 0.3, 0.5, 0.7, 0.9]
        
        results = {}
        
        for gamma in tqdm(gamma_values, desc="Testing gamma values"):
            # Save original gamma
            original_gamma = self.gamma
            original_gamma_bar = self.gamma_bar
            
            # Set new gamma
            self.gamma = gamma
            self.gamma_bar = np.sqrt(1 - gamma**2)
            
            # Run comparison
            gamma_results = self.compare_strategies(n_paths=100)
            results[gamma] = gamma_results
            
            # Restore original gamma
            self.gamma = original_gamma
            self.gamma_bar = original_gamma_bar
        
        # Plot results
        plt.figure(figsize=(10, 6))
        
        # Extract sharpe ratios
        gammas = []
        optimal_sharpe = []
        no_signal_sharpe = []
        buy_and_hold_sharpe = []
        
        for gamma in gamma_values:
            gammas.append(gamma)
            optimal_sharpe.append(results[gamma]['optimal']['sharpe'])
            no_signal_sharpe.append(results[gamma]['no_signal']['sharpe'])
            buy_and_hold_sharpe.append(results[gamma]['buy_and_hold']['sharpe'])
        
        plt.plot(gammas, optimal_sharpe, 'o-', label='Optimal Strategy')
        plt.plot(gammas, no_signal_sharpe, 's-', label='No Signal Strategy')
        plt.plot(gammas, buy_and_hold_sharpe, '^-', label='Buy and Hold')
        
        plt.xlabel('Signal Quality (gamma)')
        plt.ylabel('Sharpe Ratio')
        plt.title('Impact of Signal Quality on Strategy Performance')
        plt.legend()
        plt.grid(True)
        plt.savefig('gamma_impact.png')
        plt.show()
        
        return results

# Run simulations
if __name__ == "__main__":
    # Set parameters
    params = {
        'S0': 100,        # Initial stock price
        'mu': 0.1,        # Drift
        'sigma': 0.2,     # Volatility
        'gamma': 0.7,     # Signal quality
        'Lambda': 0.1,    # Price impact
        'alpha': 1,       # Risk aversion
        'T': 1,           # Time horizon
        'dt': 0.01        # Time step
    }
    
    # Initialize model
    model = NoisySignalInvestment(**params)
    
    # Run a single simulation and plot
    print("Running single simulation...")
    S, phi, Phi, PnL = model.plot_single_simulation()
    
    # Compare strategies
    print("\nComparing strategies...")
    results = model.compare_strategies(n_paths=100)
    
    # Print results
    print("\nStrategy Comparison Results:")
    for strategy, metrics in results.items():
        print(f"\n{strategy.upper()} STRATEGY:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
    
    # Analyze impact of gamma
    print("\nAnalyzing impact of signal quality (gamma)...")
    gamma_results = model.analyze_gamma_impact(gamma_values=[0.1, 0.3, 0.5, 0.7, 0.9])


    