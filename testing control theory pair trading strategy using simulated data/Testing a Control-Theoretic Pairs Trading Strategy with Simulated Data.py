import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from scipy import stats
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

def generate_correlated_price_data(days=1000, correlation=0.7, mean_reversion_strength=0.02, 
                                  volatility=(0.01, 0.012), drift=(0.0002, 0.0001), 
                                  starting_price=(100, 100), shocks=True):
    """
    Generate simulated price data for two stocks with correlation and mean-reversion properties
    """
    # Create correlation matrix
    corr_matrix = np.array([[1, correlation], [correlation, 1]])
    
    # Cholesky decomposition
    cholesky = np.linalg.cholesky(corr_matrix)
    
    # Generate uncorrelated random returns
    uncorrelated_returns = np.random.normal(0, 1, (days, 2))
    
    # Generate correlated random returns
    correlated_returns = np.dot(uncorrelated_returns, cholesky.T)
    
    # Scale by volatility and add drift
    stock1_returns = correlated_returns[:, 0] * volatility[0] + drift[0]
    stock2_returns = correlated_returns[:, 1] * volatility[1] + drift[1]
    
    # Initialize price arrays
    stock1_prices = np.zeros(days)
    stock2_prices = np.zeros(days)
    stock1_prices[0] = starting_price[0]
    stock2_prices[0] = starting_price[1]
    
    # Generate price paths with mean reversion in the spread
    for i in range(1, days):
        # Calculate spread (simplified log ratio in this example)
        if i > 1:
            # Prevent log(negative) errors
            ratio = max(1e-10, stock2_prices[i-1] / stock1_prices[i-1])
            log_spread = np.log(ratio)
            # Apply mean reversion to returns
            mean_reversion_factor = -mean_reversion_strength * log_spread
            stock1_returns[i-1] += mean_reversion_factor
            stock2_returns[i-1] -= mean_reversion_factor
        
        # Update prices with a floor to prevent negative prices
        stock1_prices[i] = max(0.01, stock1_prices[i-1] * (1 + stock1_returns[i-1]))
        stock2_prices[i] = max(0.01, stock2_prices[i-1] * (1 + stock2_returns[i-1]))
    
    # Add occasional shocks to the relationship
    if shocks:
        # Add 3-4 random correlation regime shifts
        num_shocks = np.random.randint(3, 5)
        for _ in range(num_shocks):
            shock_start = np.random.randint(50, days - 50)
            shock_length = np.random.randint(10, 30)
            shock_magnitude = np.random.choice([-0.05, 0.05])  # 5% shock
            stock2_prices[shock_start:shock_start+shock_length] *= (1 + shock_magnitude)
    
    # Create DataFrame
    df = pd.DataFrame({
        'Stock1': stock1_prices,
        'Stock2': stock2_prices
    })
    
    return df

def calculate_spread(prices, model='log_ratio', params=None):
    """
    Calculate spread based on the chosen model
    """
    if model == 'log_ratio':
        # Simple log ratio spread S(p) = log(p2) - β*log(p1) - μ
        if params is None:
            # Add a small constant to prevent log(0)
            log_p1 = np.log(prices['Stock1'].clip(lower=1e-10))
            log_p2 = np.log(prices['Stock2'].clip(lower=1e-10))
            
            # Perform regression to find beta
            slope, intercept, _, _, _ = stats.linregress(log_p1, log_p2)
            beta = slope
            mu = intercept
        else:
            beta = params['beta']
            mu = params['mu']
            
        spread = np.log(prices['Stock2'].clip(lower=1e-10)) - beta * np.log(prices['Stock1'].clip(lower=1e-10)) - mu
        return spread, {'beta': beta, 'mu': mu}
    
    elif model == 'ratio':
        # Simple price ratio spread S(p) = p2 / p1 - μ
        if params is None:
            # Estimate parameter
            ratio = prices['Stock2'] / prices['Stock1'].clip(lower=1e-10)
            mu = ratio.mean()
        else:
            mu = params['mu']
            
        spread = prices['Stock2'] / prices['Stock1'].clip(lower=1e-10) - mu
        return spread, {'mu': mu}
    
    else:
        raise ValueError(f"Unknown spread model: {model}")

def estimate_eta(spread, window_size=40):
    """
    Estimate the mean reversion parameter η
    """
    # Handle empty or single-value spread
    if len(spread) <= 1:
        return 0
    
    delta_spread = spread.diff().dropna()
    sign_spread = np.sign(spread[:-1])
    
    # Calculate sign(S(k)) * ΔS(k)
    signed_delta = sign_spread * delta_spread
    
    # Calculate absolute spread values
    abs_spread = np.abs(spread[:-1])
    
    # Avoid division by zero
    if abs_spread.sum() == 0:
        return 0
    
    # Calculate η estimate
    eta = -signed_delta.sum() / abs_spread.sum()
    
    return eta

def calculate_threshold(eta, gamma, prices, spread_params, model='log_ratio'):
    """
    Calculate the trading threshold τ
    """
    if eta <= 0:
        return float('inf')
    
    p1 = max(1e-10, prices['Stock1'].iloc[-1])
    p2 = max(1e-10, prices['Stock2'].iloc[-1])
    
    if model == 'log_ratio':
        beta = spread_params['beta']
        # Calculate Hessian approximation
        hessian = np.array([
            [-beta/(p1**2), 0],
            [0, 1/(p2**2)]
        ])
        
        # Approximate max value for the Hessian term
        price_vector = np.array([p1, p2])
        hessian_term = gamma**2 * np.dot(np.dot(price_vector, hessian), price_vector)
        
        # Calculate threshold
        threshold = (hessian_term / (2 * eta))
        
    elif model == 'ratio':
        # For ratio model, the Hessian is simpler
        hessian_term = gamma**2 * (1/(p1**2))
        threshold = (hessian_term / (2 * eta))
    
    return threshold

def calculate_gradient(prices, spread_params, model='log_ratio'):
    """
    Calculate gradient of the spread function
    """
    p1 = max(1e-10, prices['Stock1'].iloc[-1])
    p2 = max(1e-10, prices['Stock2'].iloc[-1])
    
    if model == 'log_ratio':
        beta = spread_params['beta']
        # Gradient of S(p) = log(p2) - β*log(p1) - μ
        gradient = np.array([-beta/p1, 1/p2])
        
    elif model == 'ratio':
        # Gradient of S(p) = p2/p1 - μ
        gradient = np.array([-p2/(p1**2), 1/p1])
    
    return gradient

def pairs_trading_simulation(prices, model='log_ratio', window_size=40, 
                            trading_window=5, leverage=1.0, initial_capital=10000):
    """
    Simulate pairs trading using the control-theoretic approach
    """
    days = len(prices)
    account_value = np.zeros(days)
    account_value[0] = initial_capital
    
    # Initialize holdings
    stock1_shares = np.zeros(days)
    stock2_shares = np.zeros(days)
    
    # Initialize spread parameters
    spread_params = None
    
    # For storing data for analysis
    eta_values = np.zeros(days)
    threshold_values = np.zeros(days)
    spread_values = np.zeros(days)
    
    # Initial gamma estimate
    gamma = 0.02  # Starting with conservative estimate
    
    # Trading loop - use tqdm for progress tracking
    i = window_size
    pbar = tqdm(total=days-window_size, desc="Trading Simulation")
    
    while i < days:
        # Define training window
        train_start = max(0, i - window_size)
        train_end = i
        train_data = prices.iloc[train_start:train_end]
        
        # Calculate spread over training window
        try:
            spread, spread_params = calculate_spread(train_data, model)
            
            # Store spread value
            spread_values[i] = spread.iloc[-1] if not spread.empty else 0
            
            # Estimate gamma (max absolute return over training window)
            returns1 = train_data['Stock1'].pct_change().dropna().abs()
            returns2 = train_data['Stock2'].pct_change().dropna().abs()
            gamma = max(returns1.max(), returns2.max()) if not (returns1.empty or returns2.empty) else gamma
            
            # Estimate eta
            eta = estimate_eta(spread)
            eta_values[i] = eta
            
            # Calculate threshold
            threshold = calculate_threshold(eta, gamma, train_data.iloc[-1:], spread_params, model)
            threshold_values[i] = threshold
            
            # Calculate current spread
            current_spread, _ = calculate_spread(prices.iloc[i:i+1], model, spread_params)
            current_spread_value = current_spread.iloc[0] if not current_spread.empty else 0
            
            # Determine trading action
            if abs(current_spread_value) > threshold:
                # Calculate gradient
                gradient = calculate_gradient(prices.iloc[i:i+1], spread_params, model)
                
                # Calculate position sizes
                p1 = prices['Stock1'].iloc[i]
                p2 = prices['Stock2'].iloc[i]
                price_vector = np.array([p1, p2])
                
                # Calculate lambda (position sizing)
                lambda_value = leverage * account_value[i-1] / np.sum(np.abs(gradient) * price_vector)
                
                # Calculate number of shares
                stock1_shares[i] = -lambda_value * np.sign(current_spread_value) * gradient[0]
                stock2_shares[i] = -lambda_value * np.sign(current_spread_value) * gradient[1]
            else:
                # No trading
                stock1_shares[i] = 0
                stock2_shares[i] = 0
                
        except Exception as e:
            # Handle any calculation errors by not trading
            stock1_shares[i] = 0
            stock2_shares[i] = 0
            print(f"Error at day {i}: {e}")
        
        # Handle trading window
        trading_end = min(i + trading_window, days - 1)
        
        for j in range(i, trading_end):
            if j > i:
                # Keep same positions for the trading window
                stock1_shares[j] = stock1_shares[i]
                stock2_shares[j] = stock2_shares[i]
            
            # Update account value
            if j > 0:
                stock1_return = prices['Stock1'].iloc[j] / prices['Stock1'].iloc[j-1] - 1
                stock2_return = prices['Stock2'].iloc[j] / prices['Stock2'].iloc[j-1] - 1
                
                # Change in account value
                delta_v = stock1_shares[j-1] * prices['Stock1'].iloc[j-1] * stock1_return + \
                          stock2_shares[j-1] * prices['Stock2'].iloc[j-1] * stock2_return
                
                account_value[j] = account_value[j-1] + delta_v
        
        # Update progress bar
        pbar.update(trading_end - i)
        
        # Move to next window
        i = trading_end
    
    pbar.close()
    
    # Create results DataFrame
    results = pd.DataFrame({
        'Stock1': prices['Stock1'],
        'Stock2': prices['Stock2'],
        'AccountValue': account_value,
        'Stock1Shares': stock1_shares,
        'Stock2Shares': stock2_shares,
        'EtaValue': eta_values,
        'Threshold': threshold_values,
        'Spread': spread_values
    })
    
    return results

def buy_and_hold_simulation(prices, initial_capital=10000):
    """
    Simulate buy and hold strategies for comparison
    """
    days = len(prices)
    
    # Initialize account values
    stock1_value = np.zeros(days)
    stock2_value = np.zeros(days)
    
    # Calculate initial shares
    stock1_shares = initial_capital / prices['Stock1'].iloc[0]
    stock2_shares = initial_capital / prices['Stock2'].iloc[0]
    
    # Calculate account values
    stock1_value = prices['Stock1'] * stock1_shares
    stock2_value = prices['Stock2'] * stock2_shares
    
    # Create results DataFrame
    results = pd.DataFrame({
        'Stock1Value': stock1_value,
        'Stock2Value': stock2_value
    })
    
    return results

def plot_results(pairs_results, buy_hold_results, prices):
    """
    Plot the results of simulations
    """
    # Create figure with subplots
    fig, axs = plt.subplots(3, 1, figsize=(14, 18), gridspec_kw={'height_ratios': [2, 1, 2]})
    
    # Plot 1: Stock prices
    axs[0].plot(prices.index, prices['Stock1'], label='Stock 1')
    axs[0].plot(prices.index, prices['Stock2'], label='Stock 2')
    axs[0].set_title('Stock Prices')
    axs[0].set_xlabel('Trading Day')
    axs[0].set_ylabel('Price ($)')
    axs[0].legend()
    axs[0].grid(True)
    
    # Plot 2: Spread and threshold (filter out infinity values)
    threshold_values = pairs_results['Threshold'].replace([np.inf, -np.inf], np.nan)
    axs[1].plot(pairs_results.index, pairs_results['Spread'], label='Spread')
    axs[1].plot(pairs_results.index, threshold_values, label='Threshold', linestyle='--')
    axs[1].plot(pairs_results.index, -threshold_values, label='-Threshold', linestyle='--')
    axs[1].set_title('Spread vs Threshold')
    axs[1].set_xlabel('Trading Day')
    axs[1].set_ylabel('Value')
    axs[1].legend()
    axs[1].grid(True)
    
    # Plot 3: Account values comparison
    axs[2].plot(pairs_results.index, pairs_results['AccountValue'], label='Pairs Trading')
    axs[2].plot(buy_hold_results.index, buy_hold_results['Stock1Value'], label='Buy & Hold Stock 1')
    axs[2].plot(buy_hold_results.index, buy_hold_results['Stock2Value'], label='Buy & Hold Stock 2')
    axs[2].set_title('Account Value Comparison')
    axs[2].set_xlabel('Trading Day')
    axs[2].set_ylabel('Account Value ($)')
    axs[2].legend()
    axs[2].grid(True)
    
    plt.tight_layout()
    plt.savefig('pairs_trading_results.png', dpi=300)
    plt.show()
    
    # Plot additional details
    fig, axs = plt.subplots(2, 1, figsize=(14, 12))
    
    # Plot 1: Eta values
    axs[0].plot(pairs_results.index, pairs_results['EtaValue'])
    axs[0].set_title('Estimated η Values')
    axs[0].set_xlabel('Trading Day')
    axs[0].set_ylabel('η')
    axs[0].grid(True)
    
    # Plot 2: Stock positions
    axs[1].plot(pairs_results.index, pairs_results['Stock1Shares'], label='Stock 1 Shares')
    axs[1].plot(pairs_results.index, pairs_results['Stock2Shares'], label='Stock 2 Shares')
    axs[1].set_title('Stock Positions')
    axs[1].set_xlabel('Trading Day')
    axs[1].set_ylabel('Number of Shares')
    axs[1].legend()
    axs[1].grid(True)
    
    plt.tight_layout()
    plt.savefig('pairs_trading_details.png', dpi=300)
    plt.show()
    
    # Print performance metrics
    initial_value = pairs_results['AccountValue'].iloc[0]
    final_value = pairs_results['AccountValue'].iloc[-1]
    total_return = (final_value / initial_value - 1) * 100
    
    # Calculate max drawdown
    peak = pairs_results['AccountValue'].expanding().max()
    drawdown = (pairs_results['AccountValue'] / peak - 1) * 100
    max_drawdown = drawdown.min()
    
    # Calculate Sharpe ratio (annualized)
    daily_returns = pairs_results['AccountValue'].pct_change().dropna()
    sharpe_ratio = np.sqrt(252) * daily_returns.mean() / daily_returns.std() if daily_returns.std() != 0 else 0
    
    print(f"Performance Metrics for Pairs Trading Strategy:")
    print(f"Initial Capital: ${initial_value:.2f}")
    print(f"Final Capital: ${final_value:.2f}")
    print(f"Total Return: {total_return:.2f}%")
    print(f"Max Drawdown: {max_drawdown:.2f}%")
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    
    # Compare with buy and hold
    for stock_name, col in [('Stock 1', 'Stock1Value'), ('Stock 2', 'Stock2Value')]:
        initial_value = buy_hold_results[col].iloc[0]
        final_value = buy_hold_results[col].iloc[-1]
        total_return = (final_value / initial_value - 1) * 100
        
        peak = buy_hold_results[col].expanding().max()
        drawdown = (buy_hold_results[col] / peak - 1) * 100
        max_drawdown = drawdown.min()
        
        daily_returns = buy_hold_results[col].pct_change().dropna()
        sharpe_ratio = np.sqrt(252) * daily_returns.mean() / daily_returns.std() if daily_returns.std() != 0 else 0
        
        print(f"\nPerformance Metrics for Buy & Hold {stock_name}:")
        print(f"Initial Capital: ${initial_value:.2f}")
        print(f"Final Capital: ${final_value:.2f}")
        print(f"Total Return: {total_return:.2f}%")
        print(f"Max Drawdown: {max_drawdown:.2f}%")
        print(f"Sharpe Ratio: {sharpe_ratio:.2f}")

# Main execution
def main():
    # Generate simulated price data
    print("Generating simulated price data...")
    days = 1000
    prices = generate_correlated_price_data(
        days=days,
        correlation=0.7,
        mean_reversion_strength=0.03,
        volatility=(0.01, 0.012),
        drift=(0.0002, 0.0001),
        starting_price=(100, 100),
        shocks=True
    )
    prices.index = range(days)
    
    # Simulate pairs trading
    print("Running pairs trading simulation...")
    pairs_results = pairs_trading_simulation(
        prices,
        model='log_ratio',
        window_size=40,
        trading_window=5,
        leverage=1.0,
        initial_capital=10000
    )
    
    # Simulate buy and hold strategies
    print("Running buy and hold simulations...")
    buy_hold_results = buy_and_hold_simulation(prices, initial_capital=10000)
    
    # Plot results
    print("Plotting results...")
    plot_results(pairs_results, buy_hold_results, prices)

if __name__ == '__main__':
    main()