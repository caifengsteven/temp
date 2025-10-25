"""
Clean visualization of QQQ with Curved Radius Supertrend signals
"""

import pandas as pd
import numpy as np
import pymysql
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

# Import the exact indicator
import sys
sys.path.append('.')
from exact_pine_replication import CurvedRadiusSupertrendExact, connect_to_nas, get_stock_data

def create_clean_visualization(ticker='QQQ', table_name='200309', limit=500, 
                               radius_strength=0.10):
    """
    Create a clean, professional chart showing:
    1. QQQ price (candlesticks)
    2. Curved Supertrend line
    3. Buy/Sell signals clearly marked
    """
    
    print(f"\n{'='*80}")
    print(f"Creating Clean Visualization for {ticker}")
    print(f"{'='*80}\n")
    
    # Fetch data
    print(f"[1/3] Fetching {limit} bars of {ticker} data...")
    connection = connect_to_nas()
    df = get_stock_data(connection, table_name, ticker, limit)
    connection.close()
    
    print(f"✓ Fetched {len(df)} bars")
    print(f"  Date range: {df['datetime'].min()} to {df['datetime'].max()}")
    print(f"  Price range: ${df['low'].min():.2f} - ${df['high'].max():.2f}")
    
    # Calculate indicator
    print(f"\n[2/3] Calculating Curved Radius Supertrend...")
    indicator = CurvedRadiusSupertrendExact(
        atr_length=14,
        atr_mult=2.0,
        radius_strength=radius_strength,
        smoothness=5
    )
    
    signals = indicator.calculate(df['high'].values, df['low'].values, df['close'].values)
    
    buy_signals = signals['buy_signals']
    sell_signals = signals['sell_signals']
    curved_band = signals['curved_band']
    direction = signals['direction']
    
    print(f"✓ Indicator calculated")
    print(f"  Buy signals: {buy_signals.sum()}")
    print(f"  Sell signals: {sell_signals.sum()}")
    
    # Create visualization
    print(f"\n[3/3] Creating chart...")
    
    fig = plt.figure(figsize=(20, 12))
    
    # Create grid for subplots
    gs = fig.add_gridspec(3, 1, height_ratios=[3, 1, 1], hspace=0.3)
    
    # ========================================================================
    # Main Chart: Price + Supertrend + Signals
    # ========================================================================
    ax1 = fig.add_subplot(gs[0])
    
    # Plot candlesticks manually
    for i in range(len(df)):
        color = 'green' if df.iloc[i]['close'] >= df.iloc[i]['open'] else 'red'
        alpha = 0.6
        
        # High-Low line
        ax1.plot([i, i], [df.iloc[i]['low'], df.iloc[i]['high']], 
                color=color, linewidth=0.8, alpha=alpha)
        
        # Open-Close body
        body_height = abs(df.iloc[i]['close'] - df.iloc[i]['open'])
        body_bottom = min(df.iloc[i]['open'], df.iloc[i]['close'])
        
        if body_height > 0:
            ax1.add_patch(plt.Rectangle((i-0.3, body_bottom), 0.6, body_height,
                                       facecolor=color, edgecolor=color, alpha=alpha))
    
    # Plot close price line
    ax1.plot(df.index, df['close'], color='black', linewidth=1.5, 
            label='Close Price', alpha=0.7, zorder=2)
    
    # Plot Curved Supertrend with color based on direction
    for i in range(1, len(curved_band)):
        if direction[i] == 1:  # Uptrend
            color = 'limegreen'
            linewidth = 3
        else:  # Downtrend
            color = 'red'
            linewidth = 3
        
        ax1.plot([i-1, i], [curved_band[i-1], curved_band[i]], 
                color=color, linewidth=linewidth, alpha=0.9, zorder=3)
    
    # Plot Buy signals
    buy_idx = np.where(buy_signals)[0]
    if len(buy_idx) > 0:
        ax1.scatter(buy_idx, df.iloc[buy_idx]['low'] * 0.999, 
                   marker='^', s=300, color='green', 
                   label='BUY Signal', zorder=5, edgecolors='darkgreen', linewidths=2)
        
        # Add text labels for buy signals
        for idx in buy_idx[:10]:  # Label first 10 to avoid clutter
            ax1.annotate('BUY', xy=(idx, df.iloc[idx]['low'] * 0.998),
                        xytext=(0, -20), textcoords='offset points',
                        ha='center', fontsize=9, fontweight='bold',
                        color='darkgreen',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.7))
    
    # Plot Sell signals
    sell_idx = np.where(sell_signals)[0]
    if len(sell_idx) > 0:
        ax1.scatter(sell_idx, df.iloc[sell_idx]['high'] * 1.001, 
                   marker='v', s=300, color='red', 
                   label='SELL Signal', zorder=5, edgecolors='darkred', linewidths=2)
        
        # Add text labels for sell signals
        for idx in sell_idx[:10]:  # Label first 10 to avoid clutter
            ax1.annotate('SELL', xy=(idx, df.iloc[idx]['high'] * 1.002),
                        xytext=(0, 20), textcoords='offset points',
                        ha='center', fontsize=9, fontweight='bold',
                        color='darkred',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightcoral', alpha=0.7))
    
    ax1.set_ylabel('Price ($)', fontsize=14, fontweight='bold')
    ax1.set_title(f'{ticker} - Curved Radius Supertrend Strategy (Radius Strength={radius_strength})', 
                 fontsize=16, fontweight='bold', pad=20)
    ax1.legend(loc='upper left', fontsize=12, framealpha=0.9)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_xlim(-10, len(df) + 10)
    
    # Add background shading for trend
    for i in range(1, len(direction)):
        if direction[i] == 1:  # Uptrend
            ax1.axvspan(i-1, i, alpha=0.05, color='green')
        else:  # Downtrend
            ax1.axvspan(i-1, i, alpha=0.05, color='red')
    
    # ========================================================================
    # Second Chart: Direction Indicator
    # ========================================================================
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    
    # Plot direction as colored bars
    colors = ['green' if d == 1 else 'red' for d in direction]
    ax2.bar(df.index, direction, color=colors, alpha=0.6, width=1.0)
    ax2.axhline(y=0, color='black', linewidth=1)
    ax2.set_ylabel('Trend\nDirection', fontsize=12, fontweight='bold')
    ax2.set_ylim(-1.5, 1.5)
    ax2.set_yticks([-1, 1])
    ax2.set_yticklabels(['DOWN', 'UP'])
    ax2.grid(True, alpha=0.3, axis='x')
    
    # Mark signals
    ax2.scatter(buy_idx, [1]*len(buy_idx), marker='^', s=150, 
               color='darkgreen', zorder=5)
    ax2.scatter(sell_idx, [-1]*len(sell_idx), marker='v', s=150, 
               color='darkred', zorder=5)
    
    # ========================================================================
    # Third Chart: Price Distance from Supertrend
    # ========================================================================
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    
    # Calculate distance
    distance = ((df['close'].values - curved_band) / curved_band) * 100
    
    # Plot distance
    colors_dist = ['green' if d > 0 else 'red' for d in distance]
    ax3.bar(df.index, distance, color=colors_dist, alpha=0.6, width=1.0)
    ax3.axhline(y=0, color='black', linewidth=1)
    ax3.set_ylabel('Distance from\nSupertrend (%)', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Bar Index', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # Mark signals
    ax3.scatter(buy_idx, distance[buy_idx], marker='^', s=150, 
               color='darkgreen', zorder=5)
    ax3.scatter(sell_idx, distance[sell_idx], marker='v', s=150, 
               color='darkred', zorder=5)
    
    # ========================================================================
    # Add summary text box
    # ========================================================================
    summary_text = f"""
    SUMMARY
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    Bars Analyzed: {len(df)}
    Buy Signals: {buy_signals.sum()}
    Sell Signals: {sell_signals.sum()}
    
    Parameters:
    • ATR Length: 14
    • ATR Multiplier: 2.0
    • Radius Strength: {radius_strength}
    • Smoothness: 5
    
    Signal Logic:
    • BUY when trend changes
      from DOWN to UP
    • SELL when trend changes
      from UP to DOWN
    """
    
    ax1.text(0.02, 0.98, summary_text, transform=ax1.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
            family='monospace')
    
    plt.tight_layout()
    
    # Save
    filename = f'clean_visualization_{ticker}_RS{radius_strength}.png'
    plt.savefig(filename, dpi=200, bbox_inches='tight')
    print(f"\n✓ Chart saved: {filename}")
    
    # Show
    plt.show()
    
    # Print signal details
    print(f"\n{'='*80}")
    print(f"SIGNAL DETAILS")
    print(f"{'='*80}\n")
    
    print("BUY SIGNALS (First 10):")
    print(f"{'Bar':<6} {'Time':<20} {'Price':<10} {'Supertrend':<12}")
    print("-" * 50)
    for idx in buy_idx[:10]:
        print(f"{idx:<6} {str(df.iloc[idx]['datetime']):<20} "
              f"${df.iloc[idx]['close']:<9.2f} ${curved_band[idx]:<11.2f}")
    
    if len(buy_idx) > 10:
        print(f"... and {len(buy_idx) - 10} more buy signals")
    
    print(f"\nSELL SIGNALS (First 10):")
    print(f"{'Bar':<6} {'Time':<20} {'Price':<10} {'Supertrend':<12}")
    print("-" * 50)
    for idx in sell_idx[:10]:
        print(f"{idx:<6} {str(df.iloc[idx]['datetime']):<20} "
              f"${df.iloc[idx]['close']:<9.2f} ${curved_band[idx]:<11.2f}")
    
    if len(sell_idx) > 10:
        print(f"... and {len(sell_idx) - 10} more sell signals")
    
    print(f"\n{'='*80}\n")
    
    return df, signals

if __name__ == "__main__":
    # Create clean visualization with 500 bars for better clarity
    print("\n" + "="*80)
    print("CURVED RADIUS SUPERTREND - CLEAN VISUALIZATION")
    print("="*80)
    
    df, signals = create_clean_visualization(
        ticker='QQQ',
        table_name='200309',
        limit=500,  # Use 500 bars for clearer visualization
        radius_strength=0.10
    )
    
    print("\n✓ Visualization complete!")
    print("\nThe chart shows:")
    print("  1. QQQ candlesticks with close price line (black)")
    print("  2. Curved Supertrend line (green=uptrend, red=downtrend)")
    print("  3. Buy signals (green triangles pointing up)")
    print("  4. Sell signals (red triangles pointing down)")
    print("  5. Trend direction indicator (middle panel)")
    print("  6. Distance from Supertrend (bottom panel)")

