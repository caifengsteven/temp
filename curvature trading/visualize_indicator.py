"""
Visualization script for Curved Radius Supertrend

This script demonstrates the indicator with different parameter settings
and creates visual comparisons with standard Supertrend.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from curved_radius_supertrend import CurvedRadiusSupertrend


def generate_realistic_price_data(n_bars: int = 300, seed: int = 42) -> pd.DataFrame:
    """
    Generate realistic price data with trends and volatility
    
    Parameters:
    -----------
    n_bars : int
        Number of bars to generate
    seed : int
        Random seed for reproducibility
        
    Returns:
    --------
    pd.DataFrame with OHLC data
    """
    np.random.seed(seed)
    
    # Create base trend with multiple phases
    t = np.linspace(0, 10, n_bars)
    
    # Combine multiple trend components
    trend = (
        100 +  # Base price
        20 * np.sin(t * 0.5) +  # Long-term wave
        10 * np.sin(t * 2) +    # Medium-term wave
        t * 2                    # Upward drift
    )
    
    # Add volatility clustering
    volatility = 1.5 + 0.5 * np.sin(t * 1.5)
    noise = np.random.randn(n_bars) * volatility
    
    # Generate close prices
    close = trend + noise
    
    # Generate OHLC
    high = close + np.abs(np.random.randn(n_bars)) * volatility * 0.8
    low = close - np.abs(np.random.randn(n_bars)) * volatility * 0.8
    open_price = close + np.random.randn(n_bars) * volatility * 0.3
    
    df = pd.DataFrame({
        'open': open_price,
        'high': high,
        'low': low,
        'close': close
    })
    
    return df


def plot_curved_supertrend(
    df: pd.DataFrame,
    result: pd.DataFrame,
    title: str = "Curved Radius Supertrend",
    ax=None
):
    """
    Plot price data with Curved Radius Supertrend overlay
    
    Parameters:
    -----------
    df : pd.DataFrame
        OHLC data
    result : pd.DataFrame
        Indicator results
    title : str
        Plot title
    ax : matplotlib axis
        Axis to plot on (creates new if None)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 7))
    
    # Plot price
    ax.plot(df['close'], label='Close Price', color='black', linewidth=1.5, alpha=0.7)
    
    # Plot curved bands
    ax.plot(result['curved_upper'], label='Curved Upper Band', 
            color='red', linewidth=2, alpha=0.6, linestyle='--')
    ax.plot(result['curved_lower'], label='Curved Lower Band', 
            color='green', linewidth=2, alpha=0.6, linestyle='--')
    
    # Plot trend line with color coding
    uptrend_mask = result['direction'] == 1
    downtrend_mask = result['direction'] == -1
    
    # Plot uptrend segments
    trend_up = result['trend_line'].copy()
    trend_up[downtrend_mask] = np.nan
    ax.plot(trend_up, color='green', linewidth=3, label='Uptrend', alpha=0.8)
    
    # Plot downtrend segments
    trend_down = result['trend_line'].copy()
    trend_down[uptrend_mask] = np.nan
    ax.plot(trend_down, color='red', linewidth=3, label='Downtrend', alpha=0.8)
    
    # Mark trend changes
    trend_changes = np.where(result['direction'].diff() != 0)[0]
    if len(trend_changes) > 0:
        ax.scatter(trend_changes, df['close'].iloc[trend_changes], 
                  color='blue', s=100, zorder=5, marker='o', 
                  label='Trend Change', edgecolors='white', linewidths=2)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Bar Index', fontsize=12)
    ax.set_ylabel('Price', fontsize=12)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    return ax


def compare_parameters():
    """
    Compare different parameter settings for the Curved Radius Supertrend
    """
    # Generate data
    df = generate_realistic_price_data(n_bars=300)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.25)
    
    # Different parameter configurations
    configs = [
        {
            'name': 'Low Radius (Scalping)',
            'params': {'radius_strength': 0.2, 'atr_period': 10, 'atr_multiplier': 2.5, 'smoothness': 2}
        },
        {
            'name': 'Medium Radius (Day Trading)',
            'params': {'radius_strength': 0.5, 'atr_period': 10, 'atr_multiplier': 3.0, 'smoothness': 3}
        },
        {
            'name': 'High Radius (Swing Trading)',
            'params': {'radius_strength': 1.0, 'atr_period': 14, 'atr_multiplier': 3.5, 'smoothness': 5}
        },
        {
            'name': 'Very High Radius (Position Trading)',
            'params': {'radius_strength': 2.0, 'atr_period': 20, 'atr_multiplier': 4.0, 'smoothness': 7}
        },
        {
            'name': 'Tight Smoothing',
            'params': {'radius_strength': 0.5, 'atr_period': 10, 'atr_multiplier': 3.0, 'smoothness': 1}
        },
        {
            'name': 'Heavy Smoothing',
            'params': {'radius_strength': 0.5, 'atr_period': 10, 'atr_multiplier': 3.0, 'smoothness': 10}
        }
    ]
    
    # Plot each configuration
    for idx, config in enumerate(configs):
        row = idx // 2
        col = idx % 2
        ax = fig.add_subplot(gs[row, col])
        
        # Calculate indicator
        indicator = CurvedRadiusSupertrend(**config['params'])
        result = indicator.calculate(
            df['high'].values,
            df['low'].values,
            df['close'].values
        )
        
        # Plot
        plot_curved_supertrend(df, result, title=config['name'], ax=ax)
        
        # Add parameter info
        param_text = '\n'.join([f"{k}: {v}" for k, v in config['params'].items()])
        ax.text(0.02, 0.98, param_text, transform=ax.transAxes,
               fontsize=8, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('Curved Radius Supertrend - Parameter Comparison', 
                fontsize=16, fontweight='bold', y=0.995)
    
    return fig


def analyze_trend_statistics(df: pd.DataFrame, result: pd.DataFrame):
    """
    Analyze and print trend statistics
    
    Parameters:
    -----------
    df : pd.DataFrame
        OHLC data
    result : pd.DataFrame
        Indicator results
    """
    print("\n" + "="*60)
    print("CURVED RADIUS SUPERTREND - TREND ANALYSIS")
    print("="*60)
    
    # Count trend changes
    trend_changes = (result['direction'].diff() != 0).sum()
    print(f"\nTotal Trend Changes: {trend_changes}")
    
    # Calculate trend durations
    trend_durations = []
    current_duration = 1
    for i in range(1, len(result)):
        if result['direction'].iloc[i] == result['direction'].iloc[i-1]:
            current_duration += 1
        else:
            trend_durations.append(current_duration)
            current_duration = 1
    trend_durations.append(current_duration)
    
    print(f"Average Trend Duration: {np.mean(trend_durations):.2f} bars")
    print(f"Median Trend Duration: {np.median(trend_durations):.2f} bars")
    print(f"Max Trend Duration: {np.max(trend_durations)} bars")
    print(f"Min Trend Duration: {np.min(trend_durations)} bars")
    
    # Uptrend vs Downtrend
    uptrend_bars = (result['direction'] == 1).sum()
    downtrend_bars = (result['direction'] == -1).sum()
    total_bars = len(result)
    
    print(f"\nUptrend Bars: {uptrend_bars} ({uptrend_bars/total_bars*100:.1f}%)")
    print(f"Downtrend Bars: {downtrend_bars} ({downtrend_bars/total_bars*100:.1f}%)")
    
    # Price performance during trends
    uptrend_returns = df['close'][result['direction'] == 1].pct_change().mean()
    downtrend_returns = df['close'][result['direction'] == -1].pct_change().mean()
    
    print(f"\nAverage Return During Uptrend: {uptrend_returns*100:.4f}%")
    print(f"Average Return During Downtrend: {downtrend_returns*100:.4f}%")
    
    print("\n" + "="*60)


def main():
    """
    Main demonstration function
    """
    print("Curved Radius Supertrend - Visualization Demo")
    print("=" * 60)
    
    # Generate sample data
    print("\nGenerating sample price data...")
    df = generate_realistic_price_data(n_bars=300)
    
    # Calculate indicator with default parameters
    print("Calculating Curved Radius Supertrend...")
    indicator = CurvedRadiusSupertrend(
        atr_period=10,
        atr_multiplier=3.0,
        radius_strength=0.5,
        smoothness=3
    )
    
    result = indicator.calculate(
        df['high'].values,
        df['low'].values,
        df['close'].values
    )
    
    # Analyze statistics
    analyze_trend_statistics(df, result)
    
    # Create comparison plot
    print("\nGenerating parameter comparison plots...")
    fig = compare_parameters()
    
    # Save figure
    output_file = 'curved_supertrend_comparison.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {output_file}")
    
    # Show plot
    plt.show()
    
    print("\nVisualization complete!")


if __name__ == "__main__":
    main()

