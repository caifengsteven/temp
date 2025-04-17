import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import dcgenerator as dg

# Set random seed for reproducibility
np.random.seed(42)

def generate_sample_data(n_points=1000, trend=0.001):
    """Generate sample price data with random walk and slight upward trend."""
    # Start with a base price of 100
    base_price = 100
    # Generate random price movements
    random_walk = np.random.normal(trend, 0.01, n_points).cumsum()
    # Create price series
    prices = base_price + random_walk * base_price
    # Convert to pandas Series
    return pd.Series(prices, name='Price')

def implement_dc_trading_strategy(prices, dc_threshold=0.01):
    """Implement a simple trading strategy based on DC events."""
    # Get DC events
    events = dg.generate(prices, d=dc_threshold)
    
    # Create a DataFrame with prices and events
    df = pd.DataFrame({'Price': prices, 'Event': events})
    
    # Initialize position and cash
    position = 0
    cash = 10000
    initial_cash = cash
    portfolio_value = []
    
    # Trading logic:
    # - Buy when we see 'end downturn' (market has turned up)
    # - Sell when we see 'end upturn' (market has turned down)
    for i in range(len(df)):
        current_price = df['Price'].iloc[i]
        
        # Skip if we can't access the event (e.g., last point)
        if i >= len(df) - 1:
            continue
            
        event = df['Event'].iloc[i]
        
        # Trading logic
        if event == 'end downturn' and position == 0:
            # Buy signal
            position = cash / current_price
            cash = 0
            print(f"Buy at {current_price:.2f}, Position: {position:.2f}")
        
        elif event == 'end upturn' and position > 0:
            # Sell signal
            cash = position * current_price
            position = 0
            print(f"Sell at {current_price:.2f}, Cash: {cash:.2f}")
        
        # Track portfolio value
        portfolio_value.append(cash + position * current_price)
    
    # Calculate returns
    total_return = (portfolio_value[-1] - initial_cash) / initial_cash * 100
    
    return {
        'portfolio_value': portfolio_value,
        'total_return': total_return,
        'events': df['Event']
    }

def plot_results(prices, strategy_results, dc_threshold):
    """Plot the price data, DC events, and portfolio value."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # Plot price data
    ax1.plot(prices.values, label='Price')
    
    # Mark DC events
    events = strategy_results['events']
    buy_signals = events == 'end downturn'
    sell_signals = events == 'end upturn'
    
    # Plot buy and sell signals
    ax1.scatter(buy_signals[buy_signals].index, 
                prices[buy_signals], 
                marker='^', color='green', s=100, label='Buy Signal')
    ax1.scatter(sell_signals[sell_signals].index, 
                prices[sell_signals], 
                marker='v', color='red', s=100, label='Sell Signal')
    
    ax1.set_title(f'Price and DC Events (Threshold: {dc_threshold*100:.1f}%)')
    ax1.set_ylabel('Price')
    ax1.legend()
    ax1.grid(True)
    
    # Plot portfolio value
    ax2.plot(strategy_results['portfolio_value'], label='Portfolio Value')
    ax2.set_title(f'Portfolio Value (Total Return: {strategy_results["total_return"]:.2f}%)')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Value')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('dc_trading_strategy_results.png')
    plt.show()

def test_multiple_thresholds(prices, thresholds):
    """Test the strategy with multiple DC thresholds."""
    results = {}
    
    for threshold in thresholds:
        print(f"\nTesting DC threshold: {threshold*100:.1f}%")
        strategy_result = implement_dc_trading_strategy(prices, dc_threshold=threshold)
        results[threshold] = strategy_result
        print(f"Total return: {strategy_result['total_return']:.2f}%")
    
    # Compare returns
    returns = [results[t]['total_return'] for t in thresholds]
    
    plt.figure(figsize=(10, 6))
    plt.bar([f"{t*100:.1f}%" for t in thresholds], returns)
    plt.title('Strategy Returns for Different DC Thresholds')
    plt.xlabel('DC Threshold (%)')
    plt.ylabel('Return (%)')
    plt.grid(True, axis='y')
    plt.savefig('dc_threshold_comparison.png')
    plt.show()
    
    # Return the best threshold
    best_threshold = thresholds[np.argmax(returns)]
    return best_threshold, results[best_threshold]

if __name__ == "__main__":
    # Generate sample data
    print("Generating sample price data...")
    prices = generate_sample_data(n_points=1000)
    
    # Test with a single threshold
    print("\nTesting with a single threshold (1%)...")
    strategy_results = implement_dc_trading_strategy(prices, dc_threshold=0.01)
    plot_results(prices, strategy_results, dc_threshold=0.01)
    
    # Test with multiple thresholds
    print("\nTesting with multiple thresholds...")
    thresholds = [0.005, 0.01, 0.02, 0.03, 0.05]
    best_threshold, best_results = test_multiple_thresholds(prices, thresholds)
    
    print(f"\nBest threshold: {best_threshold*100:.1f}%")
    print(f"Best return: {best_results['total_return']:.2f}%")
    
    # Plot results with the best threshold
    plot_results(prices, best_results, dc_threshold=best_threshold)
