"""
Test and visualize the curved radius supertrend behavior
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from database_connector import StockDataConnector
from curved_radius_supertrend import CurvedRadiusSupertrend
import warnings
warnings.filterwarnings('ignore')


def test_curve_shape():
    """Test the curve shape on a specific period"""
    
    # Fetch data
    connector = StockDataConnector()
    data = connector.fetch_stock_data('AAPL', '2023-07-01', '2023-09-30')
    connector.close()
    
    print(f"Testing on {len(data)} days of AAPL data")
    
    # Calculate indicator
    indicator = CurvedRadiusSupertrend(
        atr_period=10,
        atr_multiplier=3.0,
        radius_strength=0.5,
        smoothness=3
    )
    
    result = indicator.calculate(
        data['high'].values,
        data['low'].values,
        data['close'].values
    )
    
    # Add to dataframe
    data['curved_upper'] = result['curved_upper']
    data['curved_lower'] = result['curved_lower']
    data['direction'] = result['direction']
    data['trend_line'] = result['trend_line']
    
    # Create detailed plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 12))
    fig.patch.set_facecolor('#000000')
    
    # Plot 1: Price and curves
    ax1.set_facecolor('#000000')
    x = np.arange(len(data))
    
    # Plot price
    ax1.plot(x, data['close'], color='white', linewidth=1, alpha=0.5, label='Close Price')
    
    # Plot trend line
    uptrend_mask = data['direction'] == 1
    downtrend_mask = data['direction'] == -1
    
    trend_up = data['trend_line'].copy()
    trend_up[downtrend_mask] = np.nan
    ax1.plot(x, trend_up, color='#00ff00', linewidth=3, label='Uptrend Curve', alpha=0.9)
    
    trend_down = data['trend_line'].copy()
    trend_down[uptrend_mask] = np.nan
    ax1.plot(x, trend_down, color='#ff0000', linewidth=3, label='Downtrend Curve', alpha=0.9)
    
    # Plot both bands for reference
    ax1.plot(x, data['curved_upper'], color='#888888', linewidth=1, linestyle=':', alpha=0.3, label='Upper Band')
    ax1.plot(x, data['curved_lower'], color='#888888', linewidth=1, linestyle=':', alpha=0.3, label='Lower Band')
    
    ax1.set_title('Price and Curved Trend Lines', color='white', fontsize=14)
    ax1.legend(loc='upper left')
    ax1.grid(True, color='#333333', alpha=0.3)
    
    # Plot 2: Distance from price
    ax2.set_facecolor('#000000')
    
    # Calculate distance
    distance_to_curve = np.where(
        data['direction'] == 1,
        data['close'] - data['trend_line'],  # Distance above support
        data['trend_line'] - data['close']   # Distance below resistance
    )
    
    ax2.plot(x, distance_to_curve, color='cyan', linewidth=2, label='Distance to Curve')
    ax2.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax2.fill_between(x, 0, distance_to_curve, where=(distance_to_curve > 0), 
                     color='green', alpha=0.3, label='Price above curve')
    ax2.fill_between(x, 0, distance_to_curve, where=(distance_to_curve < 0), 
                     color='red', alpha=0.3, label='Price below curve')
    
    ax2.set_title('Distance from Price to Curve', color='white', fontsize=14)
    ax2.legend(loc='upper left')
    ax2.grid(True, color='#333333', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('test_curves_debug.png', dpi=150, facecolor='#000000')
    print("Debug chart saved to: test_curves_debug.png")
    
    # Print statistics
    print("\n" + "="*60)
    print("CURVE STATISTICS")
    print("="*60)
    
    for direction_val, direction_name in [(1, 'UPTREND'), (-1, 'DOWNTREND')]:
        mask = data['direction'] == direction_val
        if mask.sum() > 0:
            distances = distance_to_curve[mask]
            print(f"\n{direction_name}:")
            print(f"  Average distance: {distances.mean():.2f}")
            print(f"  Min distance: {distances.min():.2f}")
            print(f"  Max distance: {distances.max():.2f}")
            print(f"  Std deviation: {distances.std():.2f}")
            print(f"  % of time price above curve: {(distances > 0).sum() / len(distances) * 100:.1f}%")
    
    plt.show()


if __name__ == "__main__":
    test_curve_shape()

