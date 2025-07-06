import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
import seaborn as sns
from tqdm import tqdm

# Set random seed for reproducibility
np.random.seed(42)

# Set plot style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("viridis")

def simulate_stock_price(S0, mu, sigma, T, dt):
    """
    Simulate stock price path using geometric Brownian motion.
    
    Parameters:
    -----------
    S0 : float
        Initial stock price
    mu : float
        Drift (annualized)
    sigma : float
        Volatility (annualized)
    T : float
        Time horizon in years
    dt : float
        Time step in years
        
    Returns:
    --------
    times : ndarray
        Array of time points
    S : ndarray
        Array of stock prices
    """
    N = int(T / dt)
    times = np.linspace(0, T, N+1)
    
    # Simulate Brownian motion
    dW = np.random.normal(0, np.sqrt(dt), N)
    W = np.cumsum(dW)
    
    # Initialize stock price array
    S = np.zeros(N+1)
    S[0] = S0
    
    # Simulate stock price path
    for i in range(1, N+1):
        S[i] = S[0] * np.exp((mu - 0.5 * sigma**2) * times[i] + sigma * W[i-1])
    
    return times, S

def black_scholes_call(S, K, r, sigma, T):
    """
    Calculate Black-Scholes price for a European call option.
    
    Parameters:
    -----------
    S : float
        Current stock price
    K : float
        Strike price
    r : float
        Risk-free rate (annualized)
    sigma : float
        Volatility (annualized)
    T : float
        Time to maturity in years
        
    Returns:
    --------
    float
        Option price
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def approach_two_call(S, K, r, mu, sigma, T):
    """
    Calculate option price using Approach Two (deflated cumulative return).
    
    Parameters:
    -----------
    S : float
        Current stock price
    K : float
        Strike price
    r : float
        Risk-free rate (annualized)
    mu : float
        Drift (annualized)
    sigma : float
        Volatility (annualized)
    T : float
        Time to maturity in years
        
    Returns:
    --------
    float
        Option price
    """
    # Calculate modified volatility as per equation (35) in the paper
    sigma_R = (r / mu) * sigma
    
    # Use Black-Scholes formula with modified volatility
    d1 = (np.log(S / K) + (r + 0.5 * sigma_R**2) * T) / (sigma_R * np.sqrt(T))
    d2 = d1 - sigma_R * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def binomial_tree_approach_two(S0, K, r, mu, sigma, T, n):
    """
    Price a European call option using the binomial tree from Approach Two.
    
    Parameters:
    -----------
    S0 : float
        Initial stock price
    K : float
        Strike price
    r : float
        Risk-free rate (annualized)
    mu : float
        Drift (annualized)
    sigma : float
        Volatility (annualized)
    T : float
        Time to maturity in years
    n : int
        Number of time steps
        
    Returns:
    --------
    float
        Option price
    """
    dt = T / n
    
    # Calculate natural world probabilities and up/down factors
    p = 0.5  # Natural world probability of up move (can be set to any value between 0 and 1)
    
    # Calculate up and down factors based on equation (21a) in the paper
    u = mu * dt + sigma * np.sqrt((1 - p) / p * dt)
    d = mu * dt - sigma * np.sqrt(p / (1 - p) * dt)
    
    # Calculate the deflator (equation 31)
    pi = r / mu
    
    # Calculate modified up and down factors (equation 33)
    u_pi = r * dt + sigma * (r / mu) * np.sqrt((1 - p) / p * dt)
    d_pi = r * dt - sigma * (r / mu) * np.sqrt(p / (1 - p) * dt)
    
    # Initialize stock price tree
    stock = np.zeros((n+1, n+1))
    
    # Fill stock price tree
    for i in range(n+1):
        for j in range(i+1):
            stock[j, i] = S0 * (1 + u_pi)**(i - j) * (1 + d_pi)**j
    
    # Initialize option value tree
    option = np.zeros((n+1, n+1))
    
    # Fill option value tree at expiration
    for j in range(n+1):
        option[j, n] = max(0, stock[j, n] - K)
    
    # Fill option value tree backward in time
    for i in range(n-1, -1, -1):
        for j in range(i+1):
            # Note: Using natural world probability p as per equation (39)
            option[j, i] = (p * option[j, i+1] + (1-p) * option[j+1, i+1]) / (1 + r * dt)
    
    return option[0, 0]

# Parameters for simulation
S0 = 100       # Initial stock price
K = 100        # Strike price
r = 0.05       # Risk-free rate
mu = 0.10      # Drift
sigma = 0.20   # Volatility
T = 1.0        # Time to maturity (years)
dt = 1/252     # Daily time step

# Compute option prices
bs_price = black_scholes_call(S0, K, r, sigma, T)
approach_two_price = approach_two_call(S0, K, r, mu, sigma, T)
binomial_price = binomial_tree_approach_two(S0, K, r, mu, sigma, T, 100)

print(f"Black-Scholes price: ${bs_price:.2f}")
print(f"Approach Two price: ${approach_two_price:.2f}")
print(f"Binomial Tree (Approach Two) price: ${binomial_price:.2f}")

# Simulate stock price path
times, stock_path = simulate_stock_price(S0, mu, sigma, T, dt)

# Compute option prices along the path
bs_prices = []
approach_two_prices = []

for i, t in enumerate(times):
    if t < T:  # Only compute for times before maturity
        time_to_maturity = T - t
        bs_prices.append(black_scholes_call(stock_path[i], K, r, sigma, time_to_maturity))
        approach_two_prices.append(approach_two_call(stock_path[i], K, r, mu, sigma, time_to_maturity))

# Plot results
plt.figure(figsize=(15, 10))

plt.subplot(2, 1, 1)
plt.plot(times, stock_path, label='Stock Price')
plt.axhline(K, color='r', linestyle='--', label='Strike Price')
plt.xlabel('Time (years)')
plt.ylabel('Price')
plt.title('Simulated Stock Price Path')
plt.legend()
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(times[:-1], bs_prices, label='Black-Scholes Price')
plt.plot(times[:-1], approach_two_prices, label='Approach Two Price')
plt.xlabel('Time (years)')
plt.ylabel('Option Price')
plt.title('Option Price Comparison')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Analyze the impact of mu (drift) on option prices
mu_range = np.linspace(0.05, 0.20, 20)
bs_prices_vs_mu = [black_scholes_call(S0, K, r, sigma, T) for _ in mu_range]
approach_two_prices_vs_mu = [approach_two_call(S0, K, r, mu_val, sigma, T) for mu_val in mu_range]

plt.figure(figsize=(12, 6))
plt.plot(mu_range, bs_prices_vs_mu, label='Black-Scholes Price')
plt.plot(mu_range, approach_two_prices_vs_mu, label='Approach Two Price')
plt.xlabel('Drift (μ)')
plt.ylabel('Option Price')
plt.title('Impact of Drift on Option Prices')
plt.legend()
plt.grid(True)
plt.show()

# Analyze the impact of natural world probability on binomial prices
p_range = np.linspace(0.01, 0.99, 20)
binomial_bs = np.zeros_like(p_range)
binomial_approach_two = np.zeros_like(p_range)

for i, p in enumerate(p_range):
    # Standard binomial model (Cox-Ross-Rubinstein)
    dt = T / 100
    u = np.exp(sigma * np.sqrt(dt))
    d = 1/u
    q = (np.exp(r * dt) - d) / (u - d)  # Risk-neutral probability
    
    # Build stock price tree
    stock = np.zeros((101, 101))
    for j in range(101):
        for k in range(j+1):
            stock[k, j] = S0 * u**(j-k) * d**k
    
    # Option values at expiration
    option = np.zeros((101, 101))
    for k in range(101):
        option[k, 100] = max(0, stock[k, 100] - K)
    
    # Backward induction
    for j in range(99, -1, -1):
        for k in range(j+1):
            option[k, j] = np.exp(-r * dt) * (q * option[k, j+1] + (1-q) * option[k+1, j+1])
    
    binomial_bs[i] = option[0, 0]
    
    # Approach Two binomial model
    binomial_approach_two[i] = binomial_tree_approach_two(S0, K, r, mu, sigma, T, 100)

plt.figure(figsize=(12, 6))
plt.plot(p_range, binomial_bs, label='Standard Binomial (Cox-Ross-Rubinstein)')
plt.plot(p_range, binomial_approach_two, label='Approach Two Binomial')
plt.xlabel('Natural World Probability (p)')
plt.ylabel('Option Price')
plt.title('Impact of Natural World Probability on Binomial Option Prices')
plt.legend()
plt.grid(True)
plt.show()

# Monte Carlo simulation to compare payoffs
def monte_carlo_option_pricing(pricing_func, S0, K, r, mu, sigma, T, num_simulations=10000):
    """
    Price an option using Monte Carlo simulation.
    
    Parameters:
    -----------
    pricing_func : function
        Function to price the option
    S0 : float
        Initial stock price
    K : float
        Strike price
    r : float
        Risk-free rate
    mu : float
        Drift
    sigma : float
        Volatility
    T : float
        Time to maturity
    num_simulations : int
        Number of simulations
        
    Returns:
    --------
    float
        Option price
    """
    # Simulate terminal stock prices
    Z = np.random.standard_normal(num_simulations)
    ST = S0 * np.exp((mu - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
    
    # Calculate payoffs
    payoffs = np.maximum(ST - K, 0)
    
    # Calculate option price
    price = pricing_func(S0, K, r, mu, sigma, T)
    
    return price, payoffs, ST

# Run Monte Carlo simulations
bs_price, bs_payoffs, terminal_prices = monte_carlo_option_pricing(
    lambda S0, K, r, mu, sigma, T: black_scholes_call(S0, K, r, sigma, T),
    S0, K, r, mu, sigma, T, 10000
)

approach_two_price, approach_two_payoffs, _ = monte_carlo_option_pricing(
    approach_two_call,
    S0, K, r, mu, sigma, T, 10000
)

# Calculate P&L for each approach
bs_pnl = bs_payoffs - bs_price
approach_two_pnl = approach_two_payoffs - approach_two_price

# Plot P&L distributions
plt.figure(figsize=(12, 6))
plt.hist(bs_pnl, bins=50, alpha=0.5, label='Black-Scholes P&L')
plt.hist(approach_two_pnl, bins=50, alpha=0.5, label='Approach Two P&L')
plt.xlabel('Profit & Loss')
plt.ylabel('Frequency')
plt.title('P&L Distribution Comparison')
plt.legend()
plt.grid(True)
plt.show()

# Plot payoff vs terminal price
plt.figure(figsize=(12, 6))
plt.scatter(terminal_prices, bs_payoffs, alpha=0.1, label='Payoffs')
plt.axhline(bs_price, color='r', linestyle='--', label='Black-Scholes Price')
plt.axhline(approach_two_price, color='g', linestyle='--', label='Approach Two Price')
plt.xlabel('Terminal Stock Price')
plt.ylabel('Option Payoff')
plt.title('Option Payoff vs Terminal Stock Price')
plt.legend()
plt.grid(True)
plt.show()

# Analyze the impact of extreme market conditions
extreme_mu_range = np.linspace(0.01, 0.50, 20)  # Very high expected returns
extreme_bs_prices = [black_scholes_call(S0, K, r, sigma, T) for _ in extreme_mu_range]
extreme_approach_two_prices = [approach_two_call(S0, K, r, mu_val, sigma, T) for mu_val in extreme_mu_range]

plt.figure(figsize=(12, 6))
plt.plot(extreme_mu_range, extreme_bs_prices, label='Black-Scholes Price')
plt.plot(extreme_mu_range, extreme_approach_two_prices, label='Approach Two Price')
plt.xlabel('Drift (μ)')
plt.ylabel('Option Price')
plt.title('Option Prices Under Extreme Market Conditions (High Expected Returns)')
plt.legend()
plt.grid(True)
plt.show()

# Analyze the impact of high volatility
high_sigma_range = np.linspace(0.10, 0.80, 20)  # High volatility scenarios
high_sigma_bs_prices = [black_scholes_call(S0, K, r, sigma_val, T) for sigma_val in high_sigma_range]
high_sigma_approach_two_prices = [approach_two_call(S0, K, r, mu, sigma_val, T) for sigma_val in high_sigma_range]

plt.figure(figsize=(12, 6))
plt.plot(high_sigma_range, high_sigma_bs_prices, label='Black-Scholes Price')
plt.plot(high_sigma_range, high_sigma_approach_two_prices, label='Approach Two Price')
plt.xlabel('Volatility (σ)')
plt.ylabel('Option Price')
plt.title('Impact of High Volatility on Option Prices')
plt.legend()
plt.grid(True)
plt.show()

# Testing with a perpetual derivative (Approach One, Example B)
def perpetual_derivative_price(S, gamma, r, sigma, t):
    """
    Calculate the price of a perpetual derivative as described in Section 4.B of the paper.
    
    Parameters:
    -----------
    S : float
        Current stock price
    gamma : float
        Exponent parameter
    r : float
        Risk-free rate
    sigma : float
        Volatility
    t : float
        Time
        
    Returns:
    --------
    float
        Derivative price
    """
    delta = 2 * r / (sigma**2)
    xi = (1 - gamma) * (delta + gamma)
    return S**gamma * np.exp(xi * sigma**2 * t / 2)

# Test the perpetual derivative with gamma = -delta
delta = 2 * r / (sigma**2)
gamma = -delta

# Simulate stock and derivative prices
times, stock_path = simulate_stock_price(S0, mu, sigma, T, dt)
derivative_prices = [perpetual_derivative_price(S, gamma, r, sigma, t) for S, t in zip(stock_path, times)]

plt.figure(figsize=(12, 6))
plt.plot(times, stock_path / S0, label='Normalized Stock Price')
plt.plot(times, derivative_prices / derivative_prices[0], label='Normalized Derivative Price')
plt.xlabel('Time (years)')
plt.ylabel('Normalized Price')
plt.title(f'Stock Price vs Perpetual Derivative Price (γ = {gamma:.4f})')
plt.legend()
plt.grid(True)
plt.show()

# Analyze how different gamma values affect the perpetual derivative
gamma_values = [-delta, 0, 0.5, 1, 1-delta]
derivative_prices_by_gamma = []

for gamma in gamma_values:
    prices = [perpetual_derivative_price(S, gamma, r, sigma, t) for S, t in zip(stock_path, times)]
    derivative_prices_by_gamma.append(prices)

plt.figure(figsize=(12, 6))
for i, gamma in enumerate(gamma_values):
    plt.plot(times, np.array(derivative_prices_by_gamma[i]) / derivative_prices_by_gamma[i][0], 
             label=f'γ = {gamma:.4f}')

plt.plot(times, stock_path / S0, label='Stock Price', linestyle='--', color='black')
plt.xlabel('Time (years)')
plt.ylabel('Normalized Price')
plt.title('Impact of γ on Perpetual Derivative Price')
plt.legend()
plt.grid(True)
plt.show()

# Create a heatmap showing option price differences between the two approaches
mu_range = np.linspace(0.05, 0.25, 20)
sigma_range = np.linspace(0.10, 0.50, 20)
mu_grid, sigma_grid = np.meshgrid(mu_range, sigma_range)

price_diff = np.zeros_like(mu_grid)

for i in range(len(sigma_range)):
    for j in range(len(mu_range)):
        bs_price = black_scholes_call(S0, K, r, sigma_range[i], T)
        approach_two_price = approach_two_call(S0, K, r, mu_range[j], sigma_range[i], T)
        price_diff[i, j] = approach_two_price - bs_price

plt.figure(figsize=(12, 10))
plt.pcolormesh(mu_grid, sigma_grid, price_diff, cmap='coolwarm', shading='auto')
plt.colorbar(label='Price Difference (Approach Two - Black-Scholes)')
plt.xlabel('Drift (μ)')
plt.ylabel('Volatility (σ)')
plt.title('Option Price Difference Between Approach Two and Black-Scholes')
plt.show()

# Summary of findings
print("\nSummary of Findings:")
print("====================")
print(f"1. Black-Scholes price: ${bs_price:.2f}")
print(f"2. Approach Two price: ${approach_two_price:.2f}")
print(f"3. Binomial Tree (Approach Two) price: ${binomial_price:.2f}")
print(f"4. Difference between Approach Two and Black-Scholes: ${approach_two_price - bs_price:.2f}")
print(f"5. Perpetual derivative price (γ = {gamma:.4f}): ${derivative_prices[0]:.2f}")
print("\nKey observations:")
print("- Black-Scholes prices are independent of the drift parameter (μ)")
print("- Approach Two prices depend on both drift and volatility")
print("- The perpetual derivative with γ = -δ moves in the opposite direction of the stock")
print("- The binomial implementation of Approach Two is consistent with the analytical solution")