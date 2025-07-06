import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, poisson
from tqdm import tqdm

class RoughVolatilityFilter:
    def __init__(self, T=1, N=960, b=8000, H_true=0.3):
        """
        Initialize the rough volatility filter
        
        Parameters:
        - T: Time horizon
        - N: Number of time steps
        - b: Parameter controlling the intensity of the counting process
        - H_true: True Hurst parameter for simulation
        """
        self.T = T
        self.N = N
        self.dt = T / N
        self.b = b
        self.H_true = H_true
        self.time_grid = np.linspace(0, T, N+1)
        
    def compute_approximation_parameters(self, H, N=None, fixed_J=False):
        """
        Compute parameters for the OU approximation of fractional Brownian motion
        
        Parameters:
        - H: Hurst parameter
        - N: Number of time steps (optional)
        - fixed_J: Whether to use a fixed number of OU processes regardless of H
        
        Returns:
        - J: Number of OU processes
        - kappa: Mean reversion speeds
        - c: Coefficients
        """
        if N is None:
            N = self.N
            
        # Compute the number of OU processes based on equations (10) or (11)
        if fixed_J:
            # Equation (11): fixed dimension independent of H
            zeta = np.log(1 + 0.25)
            J = int(2 * N**zeta * np.log(N))
        else:
            # Equation (10): dimension depends on H
            zeta = np.log(1 + H)
            J = int(2 * N**zeta * np.log(N))
        
        # Parameters for the partition
        xi_0 = N**(-2*(H+0.5))
        xi_J = N**(4-2*(H+0.5))
        r = (xi_J / xi_0)**(1/J)  # Geometric ratio
        
        # Create the partition
        xi = np.zeros(J+1)
        xi[0] = xi_0
        for j in range(1, J+1):
            xi[j] = xi[j-1] * r
            
        # Compute kappa and c according to the paper
        kappa = np.zeros(J)
        c = np.zeros(J)
        
        for j in range(J):
            # Measure μ defined in equation (7)
            mu = lambda x: self.c_H(H) * x**(-H-0.5) / np.math.gamma(0.5-H)
            
            # Compute c_j and kappa_j
            c[j] = self.integrate_function(mu, xi[j], xi[j+1])
            kappa[j] = (1/c[j]) * self.integrate_function(lambda x: x*mu(x), xi[j], xi[j+1])
            
        return J, kappa, c
        
    def c_H(self, H):
        """
        Compute the constant c_H from equation (4)
        """
        numerator = np.pi * H * (2*H - 1)
        denominator = np.math.gamma(2-2*H) * np.math.gamma(H+0.5) * 2 * np.sin(np.pi*(H-0.5))
        return np.sqrt(numerator / denominator)
    
    def integrate_function(self, func, a, b, num_points=1000):
        """
        Numerically integrate a function over [a, b]
        """
        x = np.linspace(a, b, num_points)
        y = np.zeros_like(x)
        for i in range(len(x)):
            try:
                y[i] = func(x[i])
            except:
                y[i] = 0
        return np.trapz(y, x)
    
    def simulate_data(self, seed=None):
        """
        Simulate data according to the model
        
        Returns:
        - y: Observations (jumps in each interval)
        - Z: True OU processes
        - X: True state process
        """
        if seed is not None:
            np.random.seed(seed)
            
        # Compute parameters for the approximation
        J, kappa, c = self.compute_approximation_parameters(self.H_true)
        
        # Initialize arrays
        Z = np.zeros((self.N+1, J))
        X = np.zeros(self.N+1)
        y = np.zeros(self.N, dtype=int)
        
        # Generate Brownian increments
        dW = np.random.normal(0, np.sqrt(self.dt), self.N)
        
        # Simulate OU processes and state
        for n in range(1, self.N+1):
            # Update each OU process using exact updating formula (14)
            for j in range(J):
                Z[n, j] = Z[n-1, j] * np.exp(-kappa[j] * self.dt) + \
                          np.sqrt((1 - np.exp(-2 * kappa[j] * self.dt)) / (2 * kappa[j])) * dW[n-1]
            
            # Update state process X using equation (9)
            X[n] = np.sum(c * Z[n, :])
            
            # Generate observation using equation (13)
            lambda_n = self.b * np.exp(X[n-1])
            y[n-1] = np.random.poisson(lambda_n * self.dt)
            
        return y, Z, X
    
    def bootstrap_filter(self, y, H, M=600):
        """
        Bootstrap particle filter for known H
        
        Parameters:
        - y: Observations
        - H: Known Hurst parameter
        - M: Number of particles
        
        Returns:
        - X_filtered: Filtered state trajectory
        """
        # Compute parameters for the approximation
        J, kappa, c = self.compute_approximation_parameters(H)
        
        # Initialize particles and weights
        Z_particles = np.zeros((self.N+1, M, J))
        weights = np.ones(M) / M
        X_filtered = np.zeros(self.N+1)
        
        # Initialize particles at time 0
        Z_particles[0, :, :] = np.random.normal(0, 0.1, (M, J))
        X_filtered[0] = np.sum(weights * np.sum(c * Z_particles[0, :, :], axis=1))
        
        # Recursive filtering
        for n in range(1, self.N+1):
            # Propagate particles
            Z_bar = np.zeros((M, J))
            v = np.random.normal(0, 1, M)  # One random number per particle
            
            for j in range(J):
                # Update using exact updating formula (14)
                Z_bar[:, j] = Z_particles[n-1, :, j] * np.exp(-kappa[j] * self.dt) + \
                              np.sqrt((1 - np.exp(-2 * kappa[j] * self.dt)) / (2 * kappa[j])) * v
                
            # Calculate X for each particle
            X_bar = np.sum(c * Z_bar, axis=1)
            
            # Update weights based on likelihood
            lambda_bar = self.b * np.exp(X_bar)
            log_weights = y[n-1] * np.log(lambda_bar * self.dt) - lambda_bar * self.dt - np.log(np.math.factorial(y[n-1]))
            
            # Normalize weights (in log space to avoid numerical issues)
            log_weights = log_weights - np.max(log_weights)
            weights = np.exp(log_weights)
            weights = weights / np.sum(weights)
            
            # Resample
            indices = np.random.choice(M, size=M, p=weights)
            Z_particles[n, :, :] = Z_bar[indices, :]
            
            # Calculate filtered state
            X_filtered[n] = np.sum(weights * X_bar)
            
        return X_filtered
    
    def nested_particle_filter(self, y, K=30, M=100, H_prior=(0.05, 0.45)):
        """
        Nested particle filter for unknown H
        
        Parameters:
        - y: Observations
        - K: Number of parameter particles
        - M: Number of state particles per parameter
        - H_prior: Range for uniform prior on H
        
        Returns:
        - H_est: Estimated Hurst parameter over time
        - X_filtered: Filtered state trajectory
        """
        # Initialize parameter particles and weights
        H_particles = np.random.uniform(H_prior[0], H_prior[1], K)
        H_weights = np.ones(K) / K
        
        # Arrays to store results
        H_est = np.zeros(self.N+1)
        H_est[0] = np.sum(H_weights * H_particles)
        X_filtered = np.zeros(self.N+1)
        
        # Initialize state particles for each parameter particle
        Z_particles = np.zeros((K, M, self.N+1, 63))  # Fixed J=63 as per equation (11)
        
        # Initialize state particles at time 0
        for k in range(K):
            Z_particles[k, :, 0, :] = np.random.normal(0, 0.1, (M, 63))
        
        # Recursive filtering
        for n in range(1, self.N+1):
            # Parameter jittering (small random perturbation)
            H_bar = H_particles + np.random.normal(0, 0.01, K)
            H_bar = np.clip(H_bar, H_prior[0], H_prior[1])
            
            # For each jittered parameter
            log_likelihoods = np.zeros(K)
            
            for k in range(K):
                # Get parameters for this H
                _, kappa, c = self.compute_approximation_parameters(H_bar[k], fixed_J=True)
                
                # Propagate state particles for this parameter
                Z_bar = np.zeros((M, 63))
                X_bar = np.zeros(M)
                
                for m in range(M):
                    # Generate one random number for all OU processes
                    v = np.random.normal(0, 1)
                    
                    # Update each OU process
                    for j in range(63):
                        Z_bar[m, j] = Z_particles[k, m, n-1, j] * np.exp(-kappa[j] * self.dt) + \
                                     np.sqrt((1 - np.exp(-2 * kappa[j] * self.dt)) / (2 * kappa[j])) * v
                    
                    # Calculate X for this particle
                    X_bar[m] = np.sum(c * Z_bar[m, :])
                
                # Calculate likelihood for this parameter
                lambda_bar = self.b * np.exp(X_bar)
                log_particle_likelihoods = y[n-1] * np.log(lambda_bar * self.dt) - lambda_bar * self.dt - np.log(np.math.factorial(y[n-1]))
                log_likelihoods[k] = np.mean(log_particle_likelihoods)
                
                # Update state particles for this parameter
                particle_weights = np.exp(log_particle_likelihoods - np.max(log_particle_likelihoods))
                particle_weights = particle_weights / np.sum(particle_weights)
                
                # Resample state particles
                indices = np.random.choice(M, size=M, p=particle_weights)
                Z_particles[k, :, n, :] = Z_bar[indices, :]
            
            # Update parameter weights
            log_likelihoods = log_likelihoods - np.max(log_likelihoods)
            H_weights = np.exp(log_likelihoods)
            H_weights = H_weights / np.sum(H_weights)
            
            # Resample parameters
            indices = np.random.choice(K, size=K, p=H_weights)
            H_particles = H_bar[indices]
            Z_particles = Z_particles[indices, :, :, :]
            
            # Calculate estimates
            H_est[n] = np.sum(H_weights * H_particles)
            
            # Calculate filtered state (using best parameter)
            best_k = np.argmax(H_weights)
            _, kappa_best, c_best = self.compute_approximation_parameters(H_particles[best_k], fixed_J=True)
            X_filtered[n] = np.mean(np.sum(c_best * Z_particles[best_k, :, n, :], axis=1))
            
        return H_est, X_filtered
    
    def analyze_gamma_impact(self, num_simulations=10):
        """
        Analyze the impact of the signal quality parameter b
        
        Parameters:
        - num_simulations: Number of simulations to average over
        
        Returns:
        - results: Dictionary with results
        """
        b_values = [3000, 10000]
        results = {b: {'H_est': [], 'X_filtered': []} for b in b_values}
        
        for b in b_values:
            self.b = b
            
            for i in tqdm(range(num_simulations), desc=f"Simulating with b={b}"):
                # Simulate data
                y, Z, X = self.simulate_data(seed=i)
                
                # Estimate H and filter X
                H_est, X_filtered = self.nested_particle_filter(y, K=20, M=50)
                
                # Store results
                results[b]['H_est'].append(H_est)
                results[b]['X_filtered'].append(X_filtered)
                
        return results
    
    def plot_single_simulation(self, H=None):
        """
        Run a single simulation and plot the results
        
        Parameters:
        - H: Hurst parameter (if None, uses self.H_true)
        """
        if H is not None:
            self.H_true = H
            
        # Simulate data
        y, Z, X = self.simulate_data()
        
        # Filter with known H
        X_filtered_known = self.bootstrap_filter(y, self.H_true)
        
        # Filter with unknown H
        H_est, X_filtered_unknown = self.nested_particle_filter(y)
        
        # Plot results
        fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
        
        # Plot state process
        axes[0].plot(self.time_grid, X, 'k-', alpha=0.7, label='True State')
        axes[0].plot(self.time_grid, X_filtered_known, 'r-', label='Filtered (Known H)')
        axes[0].plot(self.time_grid, X_filtered_unknown, 'b--', label='Filtered (Unknown H)')
        axes[0].set_title(f'State Process and Filtered Estimates (H={self.H_true:.2f})')
        axes[0].legend()
        axes[0].grid(True)
        
        # Plot observations
        axes[1].step(self.time_grid[:-1], y, 'g-', where='post')
        axes[1].set_title('Observations (Jump Counts)')
        axes[1].grid(True)
        
        # Plot H estimates
        axes[2].plot(self.time_grid, H_est, 'b-')
        axes[2].axhline(self.H_true, color='r', linestyle='--', label='True H')
        axes[2].set_title('Estimated Hurst Parameter')
        axes[2].set_xlabel('Time')
        axes[2].legend()
        axes[2].grid(True)
        
        plt.tight_layout()
        plt.savefig(f"rough_volatility_H_{self.H_true}.png")
        plt.show()
        
        return y, Z, X, X_filtered_known, H_est, X_filtered_unknown
    
    def experiment_non_rough_volatility(self):
        """
        Run experiment with non-rough volatility models
        """
        # Case 1: Volatility as modulus of Brownian motion
        # Simulate Brownian motion
        np.random.seed(42)
        W = np.zeros(self.N+1)
        dW = np.random.normal(0, np.sqrt(self.dt), self.N)
        
        for n in range(1, self.N+1):
            W[n] = W[n-1] + dW[n-1]
        
        # Generate observations
        y_bm = np.zeros(self.N, dtype=int)
        for n in range(1, self.N+1):
            lambda_n = self.b * np.abs(W[n-1])**2
            y_bm[n-1] = np.random.poisson(lambda_n * self.dt)
        
        # Estimate H assuming rough volatility
        H_est_bm, X_filtered_bm = self.nested_particle_filter(y_bm, K=30, M=100)
        
        # Case 2: OU-OU model for volatility
        # Parameters from Rogers (2023)
        sigma_V = np.sqrt(20)
        sigma_R = np.sqrt(0.625)
        kappa = 210
        beta = 2.5
        
        # Simulate OU-OU process
        R = np.zeros(self.N+1)
        V = np.zeros(self.N+1)
        
        for n in range(1, self.N+1):
            dW0 = np.random.normal(0, np.sqrt(self.dt))
            dW1 = np.random.normal(0, np.sqrt(self.dt))
            
            R[n] = R[n-1] - beta * R[n-1] * self.dt + sigma_R * dW0
            V[n] = V[n-1] + kappa * (R[n-1] - V[n-1]) * self.dt + sigma_V * dW1
        
        # Generate observations
        y_ou = np.zeros(self.N, dtype=int)
        for n in range(1, self.N+1):
            lambda_n = self.b * V[n-1]**2
            y_ou[n-1] = np.random.poisson(lambda_n * self.dt)
        
        # Estimate H assuming rough volatility
        H_est_ou, X_filtered_ou = self.nested_particle_filter(y_ou, K=30, M=100)
        
        # Plot results
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot |W| and filtered trajectory
        axes[0, 0].plot(self.time_grid, np.abs(W), 'k-', alpha=0.7, label='|W|')
        axes[0, 0].plot(self.time_grid, X_filtered_bm, 'r-', label='Filtered')
        axes[0, 0].set_title('Volatility as |W| and Filtered Estimate')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Plot H estimates for |W|
        axes[0, 1].plot(self.time_grid, H_est_bm, 'b-')
        axes[0, 1].set_title('Estimated H (True model: |W|)')
        axes[0, 1].set_ylim(0, 0.5)
        axes[0, 1].grid(True)
        
        # Plot V and filtered trajectory
        axes[1, 0].plot(self.time_grid, V, 'k-', alpha=0.7, label='V')
        axes[1, 0].plot(self.time_grid, X_filtered_ou, 'r-', label='Filtered')
        axes[1, 0].set_title('OU-OU Volatility and Filtered Estimate')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Plot H estimates for OU-OU
        axes[1, 1].plot(self.time_grid, H_est_ou, 'b-')
        axes[1, 1].set_title('Estimated H (True model: OU-OU)')
        axes[1, 1].set_ylim(0, 0.5)
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig("non_rough_volatility_experiments.png")
        plt.show()
        
        return H_est_bm, X_filtered_bm, H_est_ou, X_filtered_ou

# Run experiments
if __name__ == "__main__":
    # Create model with default parameters
    model = RoughVolatilityFilter(T=1, N=960, b=8000, H_true=0.3)
    
    # Run a single simulation and plot results
    print("Running single simulation...")
    y, Z, X, X_filtered_known, H_est, X_filtered_unknown = model.plot_single_simulation()
    
    # Analyze impact of parameter b
    print("\nAnalyzing impact of parameter b...")
    b_results = model.analyze_gamma_impact(num_simulations=3)
    
    # Plot results for different b values
    plt.figure(figsize=(12, 6))
    
    # Plot H estimates
    for b, data in b_results.items():
        H_mean = np.mean(data['H_est'], axis=0)
        H_std = np.std(data['H_est'], axis=0)
        
        plt.plot(model.time_grid, H_mean, label=f'b={b}')
        plt.fill_between(model.time_grid, H_mean - H_std, H_mean + H_std, alpha=0.2)
    
    plt.axhline(model.H_true, color='k', linestyle='--', label='True H')
    plt.title('Estimated Hurst Parameter for Different b Values')
    plt.xlabel('Time')
    plt.ylabel('Estimated H')
    plt.legend()
    plt.grid(True)
    plt.savefig("b_impact.png")
    plt.show()
    
    # Run non-rough volatility experiments
    print("\nRunning non-rough volatility experiments...")
    model_non_rough = RoughVolatilityFilter(T=5, N=2400, b=8000)
    H_est_bm, X_filtered_bm, H_est_ou, X_filtered_ou = model_non_rough.experiment_non_rough_volatility()
    
    # Run experiment with different H values
    print("\nRunning experiments with different H values...")
    for H in [0.1, 0.2, 0.4]:
        model_H = RoughVolatilityFilter(T=1, N=960, b=8000, H_true=H)
        model_H.plot_single_simulation()

