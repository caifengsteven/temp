import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.special import gamma
import seaborn as sns

class HawkesProcess:
    def __init__(self, mu0, kernel_type='exponential', kernel_params=None):
        """
        Initialize a Hawkes process
        
        Parameters:
        - mu0: base intensity (constant)
        - kernel_type: type of kernel function ('exponential', 'power_law', 'mittag_leffler')
        - kernel_params: parameters for the kernel function
        """
        self.mu0 = mu0
        self.kernel_type = kernel_type
        
        # Default parameters
        if kernel_params is None:
            if kernel_type == 'exponential':
                # (alpha, beta) where phi(t) = alpha * exp(-beta * t)
                self.kernel_params = (0.9, 1.0)  # subcritical by default
            elif kernel_type == 'power_law':
                # (C, alpha) where phi(t) = C * t^(-alpha-1) for t > 0
                self.kernel_params = (0.9, 0.5)  # subcritical, alpha in (0,1)
            elif kernel_type == 'mittag_leffler':
                # (alpha, beta) for Mittag-Leffler type kernel
                self.kernel_params = (0.5, 1.0)  # critical when beta=1
        else:
            self.kernel_params = kernel_params
        
        # Calculate m = integral of kernel function
        self.m = self._calculate_m()
        
        # Calculate sigma = integral of t * kernel function dt
        self.sigma = self._calculate_sigma()
        
        # Determine the process type
        if self.m < 1:
            self.process_type = "subcritical"
        elif self.m == 1 and np.isfinite(self.sigma):
            self.process_type = "weakly_critical"
        elif self.m == 1 and not np.isfinite(self.sigma):
            self.process_type = "strongly_critical"
        else:
            self.process_type = "supercritical"
    
    def _calculate_m(self):
        """Calculate the average number of offspring m = integral of kernel function"""
        if self.kernel_type == 'exponential':
            alpha, beta = self.kernel_params
            return alpha / beta
        elif self.kernel_type == 'power_law':
            C, alpha = self.kernel_params
            if alpha <= 0:
                return float('inf')  # Infinite for alpha <= 0
            return C / alpha  # Integral of C*t^(-alpha-1) from 0 to infinity
        elif self.kernel_type == 'mittag_leffler':
            alpha, beta = self.kernel_params
            return 1.0  # Set to be exactly critical
        return None
    
    def _calculate_sigma(self):
        """Calculate sigma = integral of t * kernel function dt"""
        if self.kernel_type == 'exponential':
            alpha, beta = self.kernel_params
            return alpha / (beta**2)
        elif self.kernel_type == 'power_law':
            C, alpha = self.kernel_params
            if alpha <= 1:
                return float('inf')  # Infinite for alpha <= 1
            return C / (alpha * (alpha - 1))  # Integral of t * C*t^(-alpha-1)
        elif self.kernel_type == 'mittag_leffler':
            alpha, beta = self.kernel_params
            return float('inf')  # Usually infinite for Mittag-Leffler
        return None
    
    def kernel(self, t):
        """Evaluate the kernel function at time t"""
        if t <= 0:
            return 0
        
        if self.kernel_type == 'exponential':
            alpha, beta = self.kernel_params
            return alpha * np.exp(-beta * t)
        elif self.kernel_type == 'power_law':
            C, alpha = self.kernel_params
            return C * t**(-alpha-1)
        elif self.kernel_type == 'mittag_leffler':
            alpha, beta = self.kernel_params
            # Approximation of Mittag-Leffler density
            return beta * t**(alpha-1) * np.exp(-(t**alpha))
        return 0
    
    def simulate(self, T, dt=0.01):
        """
        Simulate the Hawkes process using Ogata's thinning method
        
        Parameters:
        - T: end time for simulation
        - dt: time discretization for intensity calculation
        
        Returns:
        - event_times: array of event times
        - intensity: array of intensity values at each time step
        - times: array of time points
        """
        times = np.arange(0, T, dt)
        intensity = np.zeros_like(times)
        event_times = []
        
        # Initialize with base intensity
        intensity[0] = self.mu0
        
        # Simulation using Ogata's thinning method
        t = 0
        while t < T:
            # Current maximum intensity
            max_lambda = intensity[int(t/dt)] if t/dt < len(intensity) else self.mu0
            
            # Add contribution from past events
            for event_time in event_times:
                if t > event_time:
                    max_lambda += self.kernel(t - event_time)
            
            # Generate next event candidate
            E = np.random.exponential(scale=1.0/max_lambda)
            t_candidate = t + E
            
            if t_candidate >= T:
                break
            
            # Calculate actual intensity at the candidate time
            actual_lambda = self.mu0
            for event_time in event_times:
                if t_candidate > event_time:
                    actual_lambda += self.kernel(t_candidate - event_time)
            
            # Accept/reject the candidate
            u = np.random.uniform(0, 1)
            if u <= actual_lambda / max_lambda:
                event_times.append(t_candidate)
            
            # Move to the next time
            t = t_candidate
        
        # Calculate intensity for plotting
        event_times = np.array(event_times)
        for i, t in enumerate(times):
            intensity[i] = self.mu0
            for event_time in event_times:
                if t > event_time:
                    intensity[i] += self.kernel(t - event_time)
        
        return np.array(event_times), intensity, times
    
    def calculate_N(self, event_times, times):
        """Calculate the counting process N(t)"""
        N = np.zeros_like(times)
        for i, t in enumerate(times):
            N[i] = np.sum(event_times <= t)
        return N

def test_subcritical_flt():
    """Test the functional limit theorem for subcritical Hawkes processes"""
    # Parameters
    mu0 = 1.0
    T = 100
    n_trials = 20
    n_values = [10, 50, 100, 500]
    
    # Create a subcritical Hawkes process
    hawkes = HawkesProcess(mu0, 'exponential', (0.5, 1.0))  # m = 0.5
    
    plt.figure(figsize=(12, 8))
    
    for n_idx, n in enumerate(n_values):
        # Normalize time
        plt.subplot(2, 2, n_idx+1)
        
        # Run trials
        for trial in range(n_trials):
            # Simulate
            event_times, _, times = hawkes.simulate(n*T, dt=0.1)
            
            # Calculate N(t)
            N = np.zeros(len(np.arange(0, T, 0.1)))
            t_values = np.arange(0, T, 0.1)
            
            for i, t in enumerate(t_values):
                N[i] = np.sum(event_times <= n*t) / n
            
            # Plot normalized process
            plt.plot(t_values, N, alpha=0.3, color='blue')
        
        # Plot theoretical limit
        theoretical_limit = mu0 * t_values / (1 - hawkes.m)
        plt.plot(t_values, theoretical_limit, 'r--', linewidth=2, label='Theoretical limit')
        
        plt.title(f'n = {n}')
        plt.xlabel('Time (t)')
        plt.ylabel('N(nt)/n')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig('subcritical_flt.png')
    plt.close()

def test_weakly_critical_flt():
    """Test the functional limit theorem for weakly critical Hawkes processes"""
    # Parameters
    mu0 = 1.0
    T = 100
    n_trials = 20
    n_values = [10, 50, 100, 500]
    
    # Create a weakly critical Hawkes process
    hawkes = HawkesProcess(mu0, 'exponential', (1.0, 1.0))  # m = 1, σ = 1
    
    plt.figure(figsize=(12, 8))
    
    for n_idx, n in enumerate(n_values):
        # Normalize time
        plt.subplot(2, 2, n_idx+1)
        
        # Run trials
        for trial in range(n_trials):
            # Simulate
            event_times, _, times = hawkes.simulate(n*T, dt=0.1)
            
            # Calculate N(t)
            N = np.zeros(len(np.arange(0, T, 0.1)))
            t_values = np.arange(0, T, 0.1)
            
            for i, t in enumerate(t_values):
                N[i] = np.sum(event_times <= n*t) / (n**2)
            
            # Plot normalized process
            plt.plot(t_values, N, alpha=0.3, color='blue')
        
        # Plot theoretical CIR limit approximation (would require solving SDE)
        theoretical_limit = mu0 * t_values**2 / (2 * hawkes.sigma)
        plt.plot(t_values, theoretical_limit, 'r--', linewidth=2, label='Theoretical approximation')
        
        plt.title(f'n = {n}')
        plt.xlabel('Time (t)')
        plt.ylabel('N(nt)/n²')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig('weakly_critical_flt.png')
    plt.close()

def test_strongly_critical_flt():
    """Test the functional limit theorem for strongly critical Hawkes processes"""
    # Parameters
    mu0 = 1.0
    T = 100
    n_trials = 20
    n_values = [10, 50, 100, 500]
    
    # Create a strongly critical Hawkes process
    hawkes = HawkesProcess(mu0, 'power_law', (1.0, 0.5))  # m = 1, σ = ∞
    
    plt.figure(figsize=(12, 8))
    
    for n_idx, n in enumerate(n_values):
        # Normalize time
        plt.subplot(2, 2, n_idx+1)
        
        # Run trials
        for trial in range(n_trials):
            # Simulate
            event_times, _, times = hawkes.simulate(n*T, dt=0.1)
            
            # Calculate N(t)
            N = np.zeros(len(np.arange(0, T, 0.1)))
            t_values = np.arange(0, T, 0.1)
            
            # Estimate I²ᵣ(n) (this is an approximation)
            I2R_n = n**(1.5) # This would require computing the resolvent
            
            for i, t in enumerate(t_values):
                N[i] = np.sum(event_times <= n*t) / I2R_n
            
            # Plot normalized process
            plt.plot(t_values, N, alpha=0.3, color='blue')
        
        # Plot theoretical limit
        alpha = 0.5  # from power law parameter
        theoretical_limit = mu0 * t_values**(alpha+1)
        plt.plot(t_values, theoretical_limit, 'r--', linewidth=2, label='Theoretical limit')
        
        plt.title(f'n = {n}')
        plt.xlabel('Time (t)')
        plt.ylabel('N(nt)/I²ᵣ(n)')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig('strongly_critical_flt.png')
    plt.close()

def test_convergence_rate():
    """Test the convergence rate for weakly critical Hawkes processes"""
    # Parameters
    mu0 = 1.0
    T = 10
    n_values = np.logspace(1, 3, 10).astype(int)
    n_trials = 20
    
    # Create a weakly critical Hawkes process
    hawkes = HawkesProcess(mu0, 'exponential', (1.0, 1.0))  # m = 1, σ = 1
    
    # Store Wasserstein distances
    wasserstein_distances = np.zeros((len(n_values), n_trials))
    
    for n_idx, n in enumerate(n_values):
        for trial in range(n_trials):
            # Simulate
            event_times, _, times = hawkes.simulate(n*T, dt=0.1)
            
            # Calculate N(t)
            N = np.zeros(len(np.arange(0, T, 0.1)))
            t_values = np.arange(0, T, 0.1)
            
            for i, t in enumerate(t_values):
                N[i] = np.sum(event_times <= n*t) / (n**2)
            
            # Calculate theoretical limit
            theoretical_limit = mu0 * t_values**2 / (2 * hawkes.sigma)
            
            # Compute approximate Wasserstein distance
            wasserstein_distances[n_idx, trial] = np.mean(np.abs(N - theoretical_limit))
    
    # Average over trials
    mean_distances = np.mean(wasserstein_distances, axis=1)
    std_distances = np.std(wasserstein_distances, axis=1)
    
    # Plot convergence rate
    plt.figure(figsize=(10, 6))
    plt.loglog(n_values, mean_distances, 'o-', label='Empirical convergence rate')
    plt.fill_between(n_values, 
                     mean_distances - std_distances, 
                     mean_distances + std_distances, 
                     alpha=0.2)
    
    # Plot theoretical rate (should be n^(-1/2) for exponential kernel)
    theoretical_rate = n_values**(-1/2) * mean_distances[0] * (n_values[0]**(1/2))
    plt.loglog(n_values, theoretical_rate, 'r--', label='n^(-1/2) rate')
    
    plt.xlabel('n')
    plt.ylabel('Wasserstein distance')
    plt.title('Convergence rate for weakly critical Hawkes process')
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.savefig('convergence_rate.png')
    plt.close()

def test_long_range_dependence():
    """Test for long-range dependence in critical Hawkes processes"""
    # Parameters
    mu0 = 1.0
    T = 1000
    
    # Create processes of different types
    hawkes_subcritical = HawkesProcess(mu0, 'exponential', (0.5, 1.0))  # m = 0.5
    hawkes_weakly_critical = HawkesProcess(mu0, 'exponential', (1.0, 1.0))  # m = 1, σ = 1
    hawkes_strongly_critical = HawkesProcess(mu0, 'power_law', (1.0, 0.5))  # m = 1, σ = ∞
    
    processes = [
        (hawkes_subcritical, "Subcritical"),
        (hawkes_weakly_critical, "Weakly Critical"),
        (hawkes_strongly_critical, "Strongly Critical")
    ]
    
    plt.figure(figsize=(15, 10))
    
    for p_idx, (hawkes, label) in enumerate(processes):
        # Simulate
        event_times, intensity, times = hawkes.simulate(T, dt=1.0)
        
        # Calculate increments of N(t)
        increments = np.zeros(len(times)-1)
        for i in range(len(times)-1):
            increments[i] = np.sum((event_times >= times[i]) & (event_times < times[i+1]))
        
        # Calculate autocorrelation
        max_lag = min(100, len(increments)//4)
        acf = np.zeros(max_lag)
        
        for lag in range(max_lag):
            if lag == 0:
                acf[lag] = 1.0
            else:
                acf[lag] = np.corrcoef(increments[:-lag], increments[lag:])[0, 1]
        
        # Plot autocorrelation
        plt.subplot(3, 1, p_idx+1)
        plt.plot(np.arange(max_lag), acf, 'o-')
        
        # Plot theoretical decay for comparison
        if label == "Subcritical":
            # Exponential decay
            plt.plot(np.arange(max_lag), np.exp(-0.1*np.arange(max_lag)), 'r--', 
                     label='Exponential decay')
        elif label == "Weakly Critical":
            # Power law decay with exponent -1
            plt.plot(np.arange(1, max_lag), 1/np.arange(1, max_lag), 'r--', 
                     label='1/lag decay')
        elif label == "Strongly Critical":
            # Power law decay with exponent -α
            alpha = 0.5  # from power law parameter
            plt.plot(np.arange(1, max_lag), np.arange(1, max_lag)**(-alpha), 'r--', 
                     label=f'lag^(-{alpha}) decay')
        
        plt.title(f'{label} Hawkes Process')
        plt.xlabel('Lag')
        plt.ylabel('Autocorrelation')
        plt.grid(True)
        plt.legend()
    
    plt.tight_layout()
    plt.savefig('long_range_dependence.png')
    plt.close()

# Run tests
print("Testing Functional Limit Theorems for Hawkes Processes")
print("\nTesting subcritical FLT...")
test_subcritical_flt()
print("Done! Saved as 'subcritical_flt.png'")

print("\nTesting weakly critical FLT...")
test_weakly_critical_flt()
print("Done! Saved as 'weakly_critical_flt.png'")

print("\nTesting strongly critical FLT...")
test_strongly_critical_flt()
print("Done! Saved as 'strongly_critical_flt.png'")

print("\nTesting convergence rate...")
test_convergence_rate()
print("Done! Saved as 'convergence_rate.png'")

print("\nTesting long-range dependence...")
test_long_range_dependence()
print("Done! Saved as 'long_range_dependence.png'")

print("\nAll tests completed!")