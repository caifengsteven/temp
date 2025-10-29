"""
Simple example of using the Curved Radius Supertrend indicator

This script demonstrates basic usage with sample data.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from curved_radius_supertrend import CurvedRadiusSupertrend


def main():
    """
    Simple example demonstrating the Curved Radius Supertrend
    """
    print("Curved Radius Supertrend - Simple Example")
    print("=" * 60)
    
    # Generate sample price data
    print("\n1. Generating sample price data...")
    np.random.seed(42)
    n = 200
    
    # Create realistic price movement
    trend = np.linspace(100, 130, n) + 10 * np.sin(np.linspace(0, 4*np.pi, n))
    noise = np.random.randn(n) * 2
    close = trend + noise
    high = close + np.abs(np.random.randn(n)) * 1.5
    low = close - np.abs(np.random.randn(n)) * 1.5
    
    print(f"   Generated {n} bars of price data")
    
    # Create indicator with default parameters
    print("\n2. Creating Curved Radius Supertrend indicator...")
    indicator = CurvedRadiusSupertrend(
        atr_period=10,          # ATR period
        atr_multiplier=3.0,     # ATR multiplier for bands
        radius_strength=0.5,    # Curvature strength
        smoothness=3            # Smoothing period
    )
    print("   Parameters:")
    print(f"   - ATR Period: {indicator.atr_period}")
    print(f"   - ATR Multiplier: {indicator.atr_multiplier}")
    print(f"   - Radius Strength: {indicator.radius_strength}")
    print(f"   - Smoothness: {indicator.smoothness}")
    
    # Calculate indicator
    print("\n3. Calculating indicator values...")
    result = indicator.calculate(high, low, close)
    
    # Display results
    print("\n4. Results:")
    print(f"   Current Trend: {'UPTREND' if result['direction'].iloc[-1] == 1 else 'DOWNTREND'}")
    print(f"   Current Price: {close[-1]:.2f}")
    print(f"   Trend Line: {result['trend_line'].iloc[-1]:.2f}")
    print(f"   Upper Band: {result['curved_upper'].iloc[-1]:.2f}")
    print(f"   Lower Band: {result['curved_lower'].iloc[-1]:.2f}")
    
    # Count trend changes
    trend_changes = (result['direction'].diff() != 0).sum()
    print(f"\n   Total Trend Changes: {trend_changes}")
    
    # Calculate trend statistics
    uptrend_bars = (result['direction'] == 1).sum()
    downtrend_bars = (result['direction'] == -1).sum()
    print(f"   Uptrend Bars: {uptrend_bars} ({uptrend_bars/n*100:.1f}%)")
    print(f"   Downtrend Bars: {downtrend_bars} ({downtrend_bars/n*100:.1f}%)")
    
    # Generate trading signals
    print("\n5. Trading Signals:")
    signals = []
    for i in range(1, len(result)):
        if result['direction'].iloc[i] == 1 and result['direction'].iloc[i-1] == -1:
            signals.append(('BUY', i, close[i]))
            print(f"   BUY signal at bar {i}, price: {close[i]:.2f}")
        elif result['direction'].iloc[i] == -1 and result['direction'].iloc[i-1] == 1:
            signals.append(('SELL', i, close[i]))
            print(f"   SELL signal at bar {i}, price: {close[i]:.2f}")
    
    # Create visualization
    print("\n6. Creating visualization...")
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Plot price
    ax.plot(close, label='Close Price', color='black', linewidth=1.5, alpha=0.7)
    
    # Plot curved bands
    ax.plot(result['curved_upper'], label='Upper Band', 
            color='red', linewidth=2, alpha=0.5, linestyle='--')
    ax.plot(result['curved_lower'], label='Lower Band', 
            color='green', linewidth=2, alpha=0.5, linestyle='--')
    
    # Plot trend line
    uptrend_mask = result['direction'] == 1
    downtrend_mask = result['direction'] == -1
    
    trend_up = result['trend_line'].copy()
    trend_up[downtrend_mask] = np.nan
    ax.plot(trend_up, color='green', linewidth=3, label='Uptrend', alpha=0.8)
    
    trend_down = result['trend_line'].copy()
    trend_down[uptrend_mask] = np.nan
    ax.plot(trend_down, color='red', linewidth=3, label='Downtrend', alpha=0.8)
    
    # Mark signals
    for signal_type, idx, price in signals:
        color = 'green' if signal_type == 'BUY' else 'red'
        marker = '^' if signal_type == 'BUY' else 'v'
        ax.scatter(idx, price, color=color, s=150, marker=marker, 
                  zorder=5, edgecolors='white', linewidths=2,
                  label=signal_type if idx == signals[0][1] else '')
    
    ax.set_title('Curved Radius Supertrend - Example', fontsize=14, fontweight='bold')
    ax.set_xlabel('Bar Index', fontsize=12)
    ax.set_ylabel('Price', fontsize=12)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Save plot
    output_file = 'example_output.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"   Plot saved to: {output_file}")
    
    # Show plot
    print("\n7. Displaying plot...")
    plt.show()
    
    print("\n" + "=" * 60)
    print("Example complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

