import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cvxpy as cp
from scipy.optimize import minimize
from scipy.linalg import norm
from time import time
import seaborn as sns
np.random.seed(42)

# Function to simulate stock price paths
def simulate_stock_prices(n_stocks, n_days, mu, sigma, initial_price=100):
    """
    Simulate stock price paths using geometric Brownian motion.
    
    Parameters:
    -----------
    n_stocks : int
        Number of stocks
    n_days : int
        Number of days
    mu : array-like
        Expected annual returns (in decimal)
    sigma : array-like
        Annual volatilities (in decimal)
    initial_price : float
        Initial price for all stocks
        
    Returns:
    --------
    prices : ndarray
        Stock prices with shape (n_days+1, n_stocks)
    """
    dt = 1/252  # Daily time step (assuming 252 trading days per year)
    prices = np.zeros((n_days+1, n_stocks))
    prices[0] = initial_price
    
    for t in range(1, n_days+1):
        # Generate daily log-returns
        Z = np.random.normal(0, 1, n_stocks)
        log_returns = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
        
        # Update prices
        prices[t] = prices[t-1] * np.exp(log_returns)
    
    return prices

# Calculate simple returns and log returns from prices
def calculate_returns(prices):
    """
    Calculate simple returns and log returns from prices.
    
    Parameters:
    -----------
    prices : ndarray
        Stock prices with shape (n_days+1, n_stocks)
        
    Returns:
    --------
    simple_returns : ndarray
        Simple returns with shape (n_days, n_stocks)
    log_returns : ndarray
        Log returns with shape (n_days, n_stocks)
    """
    simple_returns = prices[1:] / prices[:-1] - 1
    log_returns = np.log(prices[1:] / prices[:-1])
    
    return simple_returns, log_returns

# Kelly Portfolio Optimization
def kelly_portfolio(log_returns):
    """
    Compute the Kelly portfolio weights based on historical log returns.
    
    Parameters:
    -----------
    log_returns : ndarray
        Log returns with shape (n_days, n_stocks)
        
    Returns:
    --------
    weights : ndarray
        Portfolio weights
    """
    n_stocks = log_returns.shape[1]
    
    # Define the objective function (negative of expected log return)
    def objective(w):
        # Convert log returns to simple returns for portfolio calculation
        simple_rets = np.exp(log_returns) - 1
        # Calculate portfolio returns
        port_rets = simple_rets @ w
        # Calculate log of wealth
        log_wealth = np.log(1 + port_rets)
        # Return negative mean of log wealth (for minimization)
        return -np.mean(log_wealth)
    
    # Define the constraint (weights sum to 1)
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    
    # Define the bounds (no shorting)
    bounds = tuple((0, 1) for _ in range(n_stocks))
    
    # Initial guess
    w0 = np.ones(n_stocks) / n_stocks
    
    # Solve the optimization problem
    result = minimize(objective, w0, bounds=bounds, constraints=constraints, method='SLSQP')
    
    return result.x

# Wasserstein-Kelly Portfolio Optimization (Type-2 Wasserstein)
def wasserstein_kelly_portfolio(log_returns, epsilon):
    """
    Compute the Wasserstein-Kelly portfolio weights based on historical log returns.
    
    Parameters:
    -----------
    log_returns : ndarray
        Log returns with shape (n_days, n_stocks)
    epsilon : float
        Radius of the Wasserstein ball
        
    Returns:
    --------
    weights : ndarray
        Portfolio weights
    """
    n_samples, n_stocks = log_returns.shape
    
    # Define optimization variables
    w = cp.Variable(n_stocks, nonneg=True)
    v = cp.Variable((n_samples, n_stocks), nonneg=True)
    lambda_var = cp.Variable(1, nonneg=True)
    
    # Define constraints
    constraints = [cp.sum(w) == 1]
    
    # Define objective function (maximizing expected log return)
    obj_terms = []
    p = 2  # Using Type-2 Wasserstein
    
    for j in range(n_samples):
        r_j = log_returns[j]
        # For each sample j, calculate r_j^T v_j + sum_i(v_ji log(w_i/v_ji))
        term = r_j @ v[j]
        for i in range(n_stocks):
            # Note: CVXPY handles the domain of log automatically
            term += v[j, i] * cp.log(w[i] / v[j, i])
        
        # Calculate the regularization term
        reg_term = (p - 1) * (p**(-p/(p-1))) * lambda_var * cp.norm(v[j] / lambda_var, 'fro')**(p/(p-1))
        obj_terms.append(term - reg_term)
    
    # Form the complete objective
    objective = cp.Maximize((1/n_samples) * cp.sum(cp.hstack(obj_terms)) - lambda_var * epsilon**p)
    
    # Solve the problem
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.SCS)
    
    return w.value

# Function to evaluate portfolio performance
def evaluate_portfolio(weights, prices, rebalance_period=None):
    """
    Evaluate portfolio performance.
    
    Parameters:
    -----------
    weights : ndarray or list of ndarrays
        Portfolio weights. If rebalance_period is None, this should be a single array.
        Otherwise, it should be a list of arrays for each rebalancing period.
    prices : ndarray
        Stock prices with shape (n_days+1, n_stocks)
    rebalance_period : int, optional
        Rebalancing period in days. If None, no rebalancing.
        
    Returns:
    --------
    portfolio_value : ndarray
        Portfolio value over time
    """
    n_days = prices.shape[0] - 1
    n_stocks = prices.shape[1]
    
    # Initialize portfolio
    portfolio_value = np.ones(n_days + 1)
    stock_holdings = np.zeros((n_days + 1, n_stocks))
    stock_holdings[0] = weights * portfolio_value[0] / prices[0]
    
    # Track portfolio over time
    for t in range(1, n_days + 1):
        # Update holdings value
        stock_holdings[t] = stock_holdings[t-1]
        
        # Calculate portfolio value
        portfolio_value[t] = np.sum(stock_holdings[t] * prices[t])
        
        # Rebalance if needed
        if rebalance_period is not None and t % rebalance_period == 0 and t < n_days:
            # If we have a list of weights for each rebalancing period, use the appropriate one
            if isinstance(weights, list):
                period_idx = t // rebalance_period - 1
                if period_idx < len(weights):
                    w = weights[period_idx]
                else:
                    w = weights[-1]  # Use the last set of weights if we run out
            else:
                w = weights
            
            stock_holdings[t] = w * portfolio_value[t] / prices[t]
    
    return portfolio_value

# Set up simulation parameters
n_stocks = 10
n_days_train = 252  # One year of training data
n_days_test = 756   # Three years of test data
n_days_total = n_days_train + n_days_test

# Generate random return and volatility parameters
mu = np.random.uniform(0.05, 0.15, n_stocks)      # Annual returns between 5% and 15%
sigma = np.random.uniform(0.15, 0.35, n_stocks)   # Annual volatilities between 15% and 35%

# Simulate stock prices
prices = simulate_stock_prices(n_stocks, n_days_total, mu, sigma)

# Calculate returns
simple_returns, log_returns = calculate_returns(prices)

# Split into training and testing periods
train_log_returns = log_returns[:n_days_train]
test_prices = prices[n_days_train:]

# Compute the Kelly portfolio
kelly_weights = kelly_portfolio(train_log_returns)

# Calculate different Wasserstein-Kelly portfolios with varying epsilon values
# epsilon is set as a proportion of the average log return
mean_log_return = np.mean(np.abs(train_log_returns))
epsilon_proportions = [0.1, 0.2, 0.3, 0.4]
wasserstein_kelly_weights = []

for delta in epsilon_proportions:
    epsilon = delta * mean_log_return
    weights = wasserstein_kelly_portfolio(train_log_returns, epsilon)
    wasserstein_kelly_weights.append(weights)

# Also include the equal-weight (1/N) portfolio for comparison
equal_weights = np.ones(n_stocks) / n_stocks

# Evaluate portfolios on the test set
kelly_performance = evaluate_portfolio(kelly_weights, test_prices)
wasserstein_performances = [
    evaluate_portfolio(weights, test_prices) for weights in wasserstein_kelly_weights
]
equal_performance = evaluate_portfolio(equal_weights, test_prices)

# Calculate performance metrics
def calculate_metrics(portfolio_value):
    returns = portfolio_value[1:] / portfolio_value[:-1] - 1
    annualized_return = np.mean(returns) * 252
    annualized_volatility = np.std(returns) * np.sqrt(252)
    sharpe_ratio = annualized_return / annualized_volatility if annualized_volatility > 0 else 0
    
    # Calculate maximum drawdown
    peak = portfolio_value[0]
    max_drawdown = 0
    for value in portfolio_value:
        if value > peak:
            peak = value
        drawdown = (peak - value) / peak
        max_drawdown = max(max_drawdown, drawdown)
    
    log_final_value = np.log(portfolio_value[-1])
    
    return {
        'Annualized Return': annualized_return,
        'Annualized Volatility': annualized_volatility,
        'Sharpe Ratio': sharpe_ratio,
        'Maximum Drawdown': max_drawdown,
        'Log Final Value': log_final_value
    }

kelly_metrics = calculate_metrics(kelly_performance)
wasserstein_metrics = [calculate_metrics(perf) for perf in wasserstein_performances]
equal_metrics = calculate_metrics(equal_performance)

# Plot portfolio weights
plt.figure(figsize=(14, 8))
bar_width = 0.15
x = np.arange(n_stocks)

plt.bar(x - 2*bar_width, kelly_weights, width=bar_width, label='Kelly')
plt.bar(x - bar_width, wasserstein_kelly_weights[0], width=bar_width, label=f'Wasserstein-Kelly (δ=0.1)')
plt.bar(x, wasserstein_kelly_weights[1], width=bar_width, label=f'Wasserstein-Kelly (δ=0.2)')
plt.bar(x + bar_width, wasserstein_kelly_weights[2], width=bar_width, label=f'Wasserstein-Kelly (δ=0.3)')
plt.bar(x + 2*bar_width, wasserstein_kelly_weights[3], width=bar_width, label=f'Wasserstein-Kelly (δ=0.4)')

plt.xlabel('Stock')
plt.ylabel('Weight')
plt.title('Portfolio Weights Comparison')
plt.xticks(x, [f'Stock {i+1}' for i in range(n_stocks)])
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

# Plot portfolio performance
plt.figure(figsize=(14, 8))
plt.plot(kelly_performance, label='Kelly', linewidth=2)
for i, perf in enumerate(wasserstein_performances):
    plt.plot(perf, label=f'Wasserstein-Kelly (δ={epsilon_proportions[i]})', linewidth=2)
plt.plot(equal_performance, label='Equal Weight (1/N)', linewidth=2)

plt.xlabel('Days')
plt.ylabel('Portfolio Value')
plt.title('Portfolio Performance Comparison')
plt.legend()
plt.grid(linestyle='--', alpha=0.7)
plt.tight_layout()

# Print performance metrics
print("Performance Metrics:\n")
print(f"Kelly Portfolio:")
for key, value in kelly_metrics.items():
    print(f"  {key}: {value:.4f}")

for i, metrics in enumerate(wasserstein_metrics):
    print(f"\nWasserstein-Kelly Portfolio (δ={epsilon_proportions[i]}):")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")

print("\nEqual Weight (1/N) Portfolio:")
for key, value in equal_metrics.items():
    print(f"  {key}: {value:.4f}")

# Create a bar chart for performance metrics comparison
metrics_names = list(kelly_metrics.keys())
portfolio_names = ['Kelly'] + [f'W-Kelly (δ={delta})' for delta in epsilon_proportions] + ['Equal Weight']

metrics_data = []
for metric_name in metrics_names:
    metric_values = [kelly_metrics[metric_name]] + [metrics[metric_name] for metrics in wasserstein_metrics] + [equal_metrics[metric_name]]
    metrics_data.append(metric_values)

plt.figure(figsize=(15, 12))
for i, (metric_name, metric_values) in enumerate(zip(metrics_names, metrics_data)):
    plt.subplot(3, 2, i+1)
    plt.bar(portfolio_names, metric_values)
    plt.title(metric_name)
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()

# Run a Monte Carlo simulation to test robustness
n_simulations = 100

# Containers for results
kelly_returns = []
wasserstein_returns = [[] for _ in epsilon_proportions]
equal_returns = []

kelly_sharpe = []
wasserstein_sharpe = [[] for _ in epsilon_proportions]
equal_sharpe = []

kelly_drawdown = []
wasserstein_drawdown = [[] for _ in epsilon_proportions]
equal_drawdown = []

for sim in range(n_simulations):
    # Generate new random return and volatility parameters
    mu = np.random.uniform(0.05, 0.15, n_stocks)
    sigma = np.random.uniform(0.15, 0.35, n_stocks)
    
    # Simulate new stock prices
    prices = simulate_stock_prices(n_stocks, n_days_total, mu, sigma)
    
    # Calculate returns
    simple_returns, log_returns = calculate_returns(prices)
    
    # Split into training and testing periods
    train_log_returns = log_returns[:n_days_train]
    test_prices = prices[n_days_train:]
    
    # Compute the Kelly portfolio
    kelly_weights = kelly_portfolio(train_log_returns)
    
    # Calculate Wasserstein-Kelly portfolios
    mean_log_return = np.mean(np.abs(train_log_returns))
    wasserstein_kelly_weights = []
    
    for delta in epsilon_proportions:
        epsilon = delta * mean_log_return
        weights = wasserstein_kelly_portfolio(train_log_returns, epsilon)
        wasserstein_kelly_weights.append(weights)
    
    # Evaluate portfolios on the test set
    kelly_performance = evaluate_portfolio(kelly_weights, test_prices)
    wasserstein_performances = [
        evaluate_portfolio(weights, test_prices) for weights in wasserstein_kelly_weights
    ]
    equal_performance = evaluate_portfolio(equal_weights, test_prices)
    
    # Calculate metrics
    kelly_metrics = calculate_metrics(kelly_performance)
    wasserstein_metrics = [calculate_metrics(perf) for perf in wasserstein_performances]
    equal_metrics = calculate_metrics(equal_performance)
    
    # Store results
    kelly_returns.append(kelly_metrics['Annualized Return'])
    kelly_sharpe.append(kelly_metrics['Sharpe Ratio'])
    kelly_drawdown.append(kelly_metrics['Maximum Drawdown'])
    
    for i, metrics in enumerate(wasserstein_metrics):
        wasserstein_returns[i].append(metrics['Annualized Return'])
        wasserstein_sharpe[i].append(metrics['Sharpe Ratio'])
        wasserstein_drawdown[i].append(metrics['Maximum Drawdown'])
    
    equal_returns.append(equal_metrics['Annualized Return'])
    equal_sharpe.append(equal_metrics['Sharpe Ratio'])
    equal_drawdown.append(equal_metrics['Maximum Drawdown'])

# Plot results of Monte Carlo simulation
plt.figure(figsize=(15, 12))

# Plot returns distribution
plt.subplot(3, 1, 1)
plt.boxplot([kelly_returns] + wasserstein_returns + [equal_returns], 
            labels=['Kelly'] + [f'W-Kelly (δ={delta})' for delta in epsilon_proportions] + ['Equal'])
plt.title('Distribution of Annualized Returns')
plt.grid(linestyle='--', alpha=0.7)

# Plot Sharpe ratio distribution
plt.subplot(3, 1, 2)
plt.boxplot([kelly_sharpe] + wasserstein_sharpe + [equal_sharpe], 
            labels=['Kelly'] + [f'W-Kelly (δ={delta})' for delta in epsilon_proportions] + ['Equal'])
plt.title('Distribution of Sharpe Ratios')
plt.grid(linestyle='--', alpha=0.7)

# Plot maximum drawdown distribution
plt.subplot(3, 1, 3)
plt.boxplot([kelly_drawdown] + wasserstein_drawdown + [equal_drawdown], 
            labels=['Kelly'] + [f'W-Kelly (δ={delta})' for delta in epsilon_proportions] + ['Equal'])
plt.title('Distribution of Maximum Drawdowns')
plt.grid(linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()