import pandas as pd
import numpy as np
import dcgenerator as dg

# Set random seed for reproducibility
np.random.seed(42)

def generate_sample_data(n_points=500, trend=0.001):
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
    portfolio_values = []
    trades = []
    
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
            trades.append(f"Buy at {current_price:.2f}, Position: {position:.2f}")
        
        elif event == 'end upturn' and position > 0:
            # Sell signal
            cash = position * current_price
            position = 0
            trades.append(f"Sell at {current_price:.2f}, Cash: {cash:.2f}")
        
        # Track portfolio value
        portfolio_values.append(cash + position * current_price)
    
    # Calculate returns
    total_return = (portfolio_values[-1] - initial_cash) / initial_cash * 100
    
    return {
        'portfolio_values': portfolio_values,
        'total_return': total_return,
        'trades': trades
    }

def test_multiple_thresholds(prices, thresholds):
    """Test the strategy with multiple DC thresholds."""
    results = {}
    
    for threshold in thresholds:
        print(f"\nTesting DC threshold: {threshold*100:.1f}%")
        strategy_result = implement_dc_trading_strategy(prices, dc_threshold=threshold)
        results[threshold] = strategy_result
        
        # Print first 5 trades
        print("First 5 trades:")
        for i, trade in enumerate(strategy_result['trades'][:5]):
            print(f"  {i+1}. {trade}")
        
        # Print last 5 trades
        if len(strategy_result['trades']) > 5:
            print("Last 5 trades:")
            for i, trade in enumerate(strategy_result['trades'][-5:]):
                print(f"  {len(strategy_result['trades'])-4+i}. {trade}")
        
        print(f"Total trades: {len(strategy_result['trades'])}")
        print(f"Total return: {strategy_result['total_return']:.2f}%")
    
    # Find the best threshold
    returns = {t: results[t]['total_return'] for t in thresholds}
    best_threshold = max(returns, key=returns.get)
    
    print("\nSummary of returns:")
    for threshold in thresholds:
        print(f"  {threshold*100:.1f}%: {returns[threshold]:.2f}%")
    
    print(f"\nBest threshold: {best_threshold*100:.1f}%")
    print(f"Best return: {returns[best_threshold]:.2f}%")
    
    return best_threshold, results[best_threshold]

if __name__ == "__main__":
    # Generate sample data
    print("Generating sample price data...")
    prices = generate_sample_data(n_points=500)
    
    # Test with multiple thresholds
    print("\nTesting with multiple thresholds...")
    thresholds = [0.005, 0.01, 0.02, 0.03, 0.05]
    best_threshold, best_results = test_multiple_thresholds(prices, thresholds)
