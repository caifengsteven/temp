import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import pandas as pd
from scipy.special import gammainc, gamma

# Set random seed for reproducibility
np.random.seed(42)

def simulate_CIR_process(gamma_0, alpha, gamma_star, sigma, T, N):
    """
    Simulate the Cox-Ingersoll-Ross process for earning yield
    
    Parameters:
    gamma_0: initial earning yield
    alpha: mean-reversion speed
    gamma_star: long-term mean of the earning yield
    sigma: volatility parameter
    T: time horizon
    N: number of time steps
    
    Returns:
    time_points: array of time points
    gamma: array of simulated earning yields
    """
    dt = T/N
    time_points = np.linspace(0, T, N+1)
    gamma = np.zeros(N+1)
    gamma[0] = gamma_0
    
    for i in range(N):
        # Using the Euler-Maruyama discretization
        dW = np.random.normal(0, np.sqrt(dt))
        drift = alpha * (gamma_star - gamma[i]) * dt
        diffusion = sigma * np.sqrt(gamma[i]) * dW
        gamma[i+1] = max(0, gamma[i] + drift + diffusion)  # Ensure non-negativity
    
    return time_points, gamma

def calculate_price(E, gamma):
    """
    Calculate price from earning yield
    P = E/γ
    """
    return E / gamma

# Set parameters
E = 10  # Earnings (constant)
gamma_0 = 0.04  # Initial earning yield (4%)
gamma_star = 0.04  # Long-term mean of earning yield
alpha = 0.5  # Mean-reversion speed
sigma = 0.1  # Volatility of earning yield
T = 10  # Time horizon (years)
N = 2520  # Number of time steps (252 trading days per year for 10 years)

# Simulate earning yield
time_points, gamma = simulate_CIR_process(gamma_0, alpha, gamma_star, sigma, T, N)

# Calculate price
price = calculate_price(E, gamma)

# Calculate the theoretical long-term expected price multiplier (ξ)
H = 2 * alpha * E / (sigma**2)
P_star = E / gamma_star
xi = 1 / (1 - P_star/H) if P_star < H else float('inf')
expected_long_term_price = xi * P_star

# Analyze regime
if 2 * alpha * gamma_star > sigma**2:
    regime = "Non-explosive nonlinear regime"
    tail_exponent = 2 * alpha * gamma_star / (sigma**2)
else:
    regime = "Recurrent explosive bubble regime"
    tail_exponent = 2 * alpha * gamma_star / (sigma**2)

# Plot results
plt.figure(figsize=(15, 10))

# Plot price
plt.subplot(2, 2, 1)
plt.plot(time_points, price)
plt.axhline(y=P_star, color='r', linestyle='--', label=f'P* = {P_star:.2f}')
plt.axhline(y=expected_long_term_price, color='g', linestyle='--', 
           label=f'ξP* = {expected_long_term_price:.2f}')
plt.title('Simulated Price Process')
plt.xlabel('Time (years)')
plt.ylabel('Price')
plt.yscale('log')  # Log scale to observe super-exponential patterns
plt.legend()

# Plot earning yield
plt.subplot(2, 2, 2)
plt.plot(time_points, gamma)
plt.axhline(y=gamma_star, color='r', linestyle='--', label=f'γ* = {gamma_star:.4f}')
plt.title('Simulated Earning Yield Process (CIR)')
plt.xlabel('Time (years)')
plt.ylabel('Earning Yield')
plt.legend()

# Plot returns
returns = np.diff(price) / price[:-1]
plt.subplot(2, 2, 3)
plt.hist(returns, bins=50, density=True, alpha=0.7)
plt.title('Distribution of Returns')
plt.xlabel('Return')
plt.ylabel('Density')

# Plot tail distribution of prices
plt.subplot(2, 2, 4)
sorted_prices = np.sort(price)
ccdf = 1 - np.arange(1, len(sorted_prices) + 1) / len(sorted_prices)
plt.loglog(sorted_prices, ccdf)
plt.title(f'Tail Distribution of Prices (Regime: {regime})')
plt.xlabel('Price (log scale)')
plt.ylabel('CCDF (log scale)')
plt.text(0.1, 0.1, f'Theoretical tail exponent: {tail_exponent:.2f}', 
         transform=plt.gca().transAxes)

plt.tight_layout()
plt.show()

# Let's also simulate multiple paths to demonstrate the emergent risk premium
plt.figure(figsize=(12, 6))

num_paths = 5
for i in range(num_paths):
    np.random.seed(42 + i)  # Different seed for each path
    time_points, gamma_path = simulate_CIR_process(gamma_0, alpha, gamma_star, sigma, T, N)
    price_path = calculate_price(E, gamma_path)
    plt.plot(time_points, price_path, alpha=0.7, label=f'Path {i+1}')

plt.axhline(y=P_star, color='r', linestyle='--', label=f'P* = {P_star:.2f}')
plt.axhline(y=expected_long_term_price, color='g', linestyle='--', 
           label=f'ξP* = {expected_long_term_price:.2f}')
plt.title('Multiple Price Paths Showing Emergent Risk Premium')
plt.xlabel('Time (years)')
plt.ylabel('Price')
plt.yscale('log')
plt.legend()
plt.show()

# Analyze the super-exponential growth pattern
# We'll take the log of average price path over multiple simulations
num_paths = 100
all_paths = np.zeros((num_paths, N+1))

for i in range(num_paths):
    np.random.seed(42 + i)
    _, gamma_path = simulate_CIR_process(gamma_0, alpha, gamma_star, sigma, T, N)
    all_paths[i] = calculate_price(E, gamma_path)

avg_price = np.mean(all_paths, axis=0)
log_avg_price = np.log(avg_price)

# Calculate instantaneous growth rate (derivative of log price)
growth_rate = np.diff(log_avg_price) / (T/N)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(time_points, avg_price)
plt.axhline(y=P_star, color='r', linestyle='--', label=f'P* = {P_star:.2f}')
plt.axhline(y=expected_long_term_price, color='g', linestyle='--', 
           label=f'ξP* = {expected_long_term_price:.2f}')
plt.title('Average Price Path over 100 Simulations')
plt.xlabel('Time (years)')
plt.ylabel('Price')
plt.yscale('log')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(time_points[1:], growth_rate)
plt.title('Instantaneous Growth Rate')
plt.xlabel('Time (years)')
plt.ylabel('Growth Rate (d/dt log(P))')
plt.axhline(y=0, color='k', linestyle='--')
plt.tight_layout()
plt.show()

# Print summary of the model
print(f"Model Parameters:")
print(f"Earnings (E): {E}")
print(f"Initial earning yield (γ₀): {gamma_0:.4f}")
print(f"Long-term earning yield (γ*): {gamma_star:.4f}")
print(f"Mean-reversion speed (α): {alpha:.4f}")
print(f"Volatility (σ): {sigma:.4f}")
print(f"Regime: {regime}")
print(f"Tail exponent (τ*): {tail_exponent:.4f}")
print(f"Reference price (P*): {P_star:.4f}")
print(f"H = 2αE/σ²: {H:.4f}")
print(f"Emergent risk premium multiplier (ξ): {xi:.4f}")
print(f"Expected long-term price (ξP*): {expected_long_term_price:.4f}")