import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.stats import norm

# Market parameters
T = 3.0  # time horizon
mu = 0.06  # expected return
sigma = 0.3  # volatility
dt = 0.01  # time step
times = np.arange(0, T, dt)

# CRRA utility function (ρ=1 means logarithmic utility)
def utility(w, rho=1):
    if rho == 1:
        return np.log(w)
    else:
        return (w**(1-rho) - 1) / (1-rho)

def utility_prime(w, rho=1):
    if rho == 1:
        return 1/w
    else:
        return w**(-rho)

def utility_double_prime(w, rho=1):
    if rho == 1:
        return -1/(w**2)
    else:
        return -rho * w**(-rho-1)

# Function to compute the marginal rate of substitution for GDA preferences
def compute_mrs(x, delta, beta, rho=1):
    """
    Compute the MRS using Monte Carlo simulation
    x: cumulative risk (corresponds to sqrt(v))
    delta: GDA parameter
    beta: GDA parameter
    rho: CRRA utility parameter
    """
    # For EU preferences, the MRS is just 1/rho
    if beta == 0:
        return 1/rho
    
    # For δ=1 (DA preferences), return 0 (non-participation)
    if delta == 1:
        return 0
    
    # For GDA preferences with δ≠1, estimate the MRS
    num_samples = 10000
    z = np.exp(x * np.random.normal(size=num_samples) - x**2/2)
    
    # Iteratively find h(x) (the GDA value)
    def h_equation(h):
        if delta <= 1:
            benchmark = delta * h
        else:
            benchmark = delta * h
        
        expected_utility = np.mean(utility(z, rho))
        penalty = beta * np.mean(np.maximum(utility(benchmark, rho) - utility(z, rho), 0))
        
        return utility(h, rho) - expected_utility + penalty
    
    h = fsolve(h_equation, 1.0)[0]
    
    # Compute components for MRS
    benchmark = delta * h
    indicator = z < benchmark
    
    numerator = np.mean(utility_prime(z, rho) * z * (1 + beta * indicator))
    denominator = np.mean(utility_double_prime(z, rho) * z**2 * (1 + beta * indicator))
    
    # Add correction term for the benchmark effect
    if beta > 0:
        N_prime = norm.pdf(np.log(benchmark)/x - (1-rho)*x)
        correction = beta * utility_prime(benchmark, rho) * benchmark * N_prime / x
        denominator = denominator - correction
    
    # Return MRS
    return numerator / (-denominator)

# Function to generate equilibrium strategy
def compute_equilibrium_strategy(delta, beta, rho=1):
    strategy = np.zeros_like(times)
    
    # EU strategy (Merton solution) for comparison
    eu_strategy = np.ones_like(times) * mu / (rho * sigma**2)
    
    # For DA preferences (δ=1), the equilibrium is non-participation
    if delta == 1:
        return strategy, eu_strategy
    
    # For GDA preferences with δ≠1
    for i, t in enumerate(reversed(times[:-1])):
        # Compute remaining time
        tau = T - t
        
        # Calculate cumulative risk (approximate)
        remaining_steps = int(tau / dt)
        if remaining_steps == 0:
            x = 0.001  # Close to terminal time
        else:
            # Approximation of sqrt(v(t))
            x = np.sqrt(np.sum(strategy[-remaining_steps:]**2 * sigma**2 * dt))
        
        # Calculate MRS and equilibrium strategy
        mrs = compute_mrs(x, delta, beta, rho)
        strategy[-(i+1)] = mrs * mu / sigma**2
        
    return strategy, eu_strategy

# Plotting function
def plot_strategies(delta_values, beta, rho=1):
    plt.figure(figsize=(10, 6))
    
    for delta in delta_values:
        strategy, eu_strategy = compute_equilibrium_strategy(delta, beta, rho)
        plt.plot(times, strategy, label=f'δ={delta}')
    
    plt.plot(times, eu_strategy, 'k--', label='EU (Merton)')
    
    plt.title(f'Equilibrium Strategies (β={beta}, ρ={rho})')
    plt.xlabel('Time')
    plt.ylabel('Investment Proportion')
    plt.legend()
    plt.grid(True)
    plt.show()

# Test the effect of different δ values
plot_strategies([0.8, 0.9, 1.1, 1.2], beta=0.5)

# Test the effect of different β values
delta_values = [0.9, 1.1]
beta_values = [0.3, 0.5, 0.7]

for delta in delta_values:
    plt.figure(figsize=(10, 6))
    
    for beta in beta_values:
        strategy, eu_strategy = compute_equilibrium_strategy(delta, beta)
        plt.plot(times, strategy, label=f'β={beta}')
    
    plt.plot(times, eu_strategy, 'k--', label='EU (Merton)')
    
    plt.title(f'Equilibrium Strategies (δ={delta})')
    plt.xlabel('Time')
    plt.ylabel('Investment Proportion')
    plt.legend()
    plt.grid(True)
    plt.show()