import pandas as pd
import numpy as np
import dcgenerator as dg

# Set random seed for reproducibility
np.random.seed(42)

def generate_sample_data(n_points=100, trend=0.001):
    """Generate sample price data with random walk and slight upward trend."""
    # Start with a base price of 100
    base_price = 100
    # Generate random price movements
    random_walk = np.random.normal(trend, 0.01, n_points).cumsum()
    # Create price series
    prices = base_price + random_walk * base_price
    # Convert to pandas Series
    return pd.Series(prices, name='Price')

def test_dc_generator():
    """Test the DC generator with different thresholds."""
    # Generate sample data
    print("Generating sample price data...")
    prices = generate_sample_data(n_points=100)
    
    # Print first 10 prices
    print("\nSample prices (first 10):")
    print(prices.head(10))
    
    # Test with different thresholds
    thresholds = [0.005, 0.01, 0.02, 0.05]
    
    for threshold in thresholds:
        print(f"\nTesting DC threshold: {threshold*100:.1f}%")
        
        # Generate DC events
        events = dg.generate(prices, d=threshold)
        
        # Create a DataFrame with prices and events
        df = pd.DataFrame({'Price': prices, 'Event': events})
        
        # Count different types of events
        event_counts = df['Event'].value_counts()
        print("\nEvent counts:")
        print(event_counts)
        
        # Print sample of events (non-empty)
        print("\nSample of events (non-empty):")
        non_empty_events = df[df['Event'] != '']
        if len(non_empty_events) > 0:
            print(non_empty_events.head(10))
        else:
            print("No events detected with this threshold.")

if __name__ == "__main__":
    test_dc_generator()
