"""
Better visualization with proper candlesticks using mplfinance
"""

import pandas as pd
import numpy as np
import pymysql
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

# Import the exact indicator
import sys
sys.path.append('.')
from exact_pine_replication import CurvedRadiusSupertrendExact, connect_to_nas, get_stock_data

def create_better_chart(ticker='QQQ', table_name='200309', limit=300, radius_strength=0.10):
    """
    Create a better chart with proper candlesticks
    """
    
    print(f"\n{'='*80}")
    print(f"Creating Better Visualization for {ticker}")
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
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 12), 
                                    gridspec_kw={'height_ratios': [3, 1]}, 
                                    sharex=True)
    
    # ========================================================================
    # Main Chart: Candlesticks + Supertrend + Signals
    # ========================================================================
    
    # Draw candlesticks properly
    width = 0.6
    width2 = 0.05
    
    up_color = 'green'
    down_color = 'red'
    
    for i in range(len(df)):
        open_price = df.iloc[i]['open']
        high_price = df.iloc[i]['high']
        low_price = df.iloc[i]['low']
        close_price = df.iloc[i]['close']
        
        # Determine color
        if close_price >= open_price:
            color = up_color
            body_low = open_price
            body_high = close_price
        else:
            color = down_color
            body_low = close_price
            body_high = open_price
        
        # Draw high-low line (wick)
        ax1.plot([i, i], [low_price, high_price], color=color, linewidth=1, solid_capstyle='round')
        
        # Draw open-close rectangle (body)
        if body_high > body_low:
            rect = Rectangle((i - width/2, body_low), width, body_high - body_low,
                           facecolor=color, edgecolor=color, alpha=0.8)
            ax1.add_patch(rect)
        else:
            # Doji - draw a line
            ax1.plot([i - width/2, i + width/2], [close_price, close_price], 
                    color=color, linewidth=2)
    
    # Plot Curved Supertrend with color based on direction
    for i in range(1, len(curved_band)):
        if direction[i] == 1:  # Uptrend
            color = 'blue'
            linewidth = 2.5
        else:  # Downtrend
            color = 'orange'
            linewidth = 2.5
        
        ax1.plot([i-1, i], [curved_band[i-1], curved_band[i]], 
                color=color, linewidth=linewidth, alpha=0.9, zorder=3)
    
    # Plot Buy signals
    buy_idx = np.where(buy_signals)[0]
    if len(buy_idx) > 0:
        for idx in buy_idx:
            # Draw arrow from below
            arrow_y = df.iloc[idx]['low'] - (df['high'].max() - df['low'].min()) * 0.02
            ax1.annotate('', xy=(idx, df.iloc[idx]['low']), 
                        xytext=(idx, arrow_y),
                        arrowprops=dict(arrowstyle='->', color='green', lw=3))
            ax1.text(idx, arrow_y, 'BUY', ha='center', va='top', 
                    fontsize=9, fontweight='bold', color='white',
                    bbox=dict(boxstyle='round,pad=0.4', facecolor='green', alpha=0.9))
    
    # Plot Sell signals
    sell_idx = np.where(sell_signals)[0]
    if len(sell_idx) > 0:
        for idx in sell_idx:
            # Draw arrow from above
            arrow_y = df.iloc[idx]['high'] + (df['high'].max() - df['low'].min()) * 0.02
            ax1.annotate('', xy=(idx, df.iloc[idx]['high']), 
                        xytext=(idx, arrow_y),
                        arrowprops=dict(arrowstyle='->', color='red', lw=3))
            ax1.text(idx, arrow_y, 'SELL', ha='center', va='bottom', 
                    fontsize=9, fontweight='bold', color='white',
                    bbox=dict(boxstyle='round,pad=0.4', facecolor='red', alpha=0.9))
    
    # Formatting
    ax1.set_ylabel('Price ($)', fontsize=14, fontweight='bold')
    ax1.set_title(f'{ticker} 1-Minute Chart with Curved Radius Supertrend (RS={radius_strength})', 
                 fontsize=16, fontweight='bold', pad=20)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_xlim(-5, len(df) + 5)
    
    # Add legend
    legend_elements = [
        Line2D([0], [0], color=up_color, lw=4, label='Bullish Candle'),
        Line2D([0], [0], color=down_color, lw=4, label='Bearish Candle'),
        Line2D([0], [0], color='blue', lw=3, label='Supertrend (Uptrend)'),
        Line2D([0], [0], color='orange', lw=3, label='Supertrend (Downtrend)'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='green', 
               markersize=12, label='Buy Signal', linestyle='None'),
        Line2D([0], [0], marker='v', color='w', markerfacecolor='red', 
               markersize=12, label='Sell Signal', linestyle='None'),
    ]
    ax1.legend(handles=legend_elements, loc='upper left', fontsize=11, framealpha=0.95)
    
    # Add price range info
    price_range = df['high'].max() - df['low'].min()
    info_text = f"""Data Info:
Bars: {len(df)}
Price Range: ${df['low'].min():.2f} - ${df['high'].max():.2f}
Range: ${price_range:.2f}
Buy Signals: {buy_signals.sum()}
Sell Signals: {sell_signals.sum()}"""
    
    ax1.text(0.99, 0.97, info_text, transform=ax1.transAxes,
            fontsize=10, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9),
            family='monospace')
    
    # ========================================================================
    # Volume Chart
    # ========================================================================
    
    # Color volume bars by price direction
    colors = [up_color if df.iloc[i]['close'] >= df.iloc[i]['open'] else down_color 
              for i in range(len(df))]
    
    ax2.bar(df.index, df['volume'], color=colors, alpha=0.6, width=0.8)
    ax2.set_ylabel('Volume', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Bar Index', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Mark signals on volume
    if len(buy_idx) > 0:
        ax2.scatter(buy_idx, df.iloc[buy_idx]['volume'], marker='^', 
                   s=200, color='green', zorder=5, edgecolors='darkgreen', linewidths=2)
    if len(sell_idx) > 0:
        ax2.scatter(sell_idx, df.iloc[sell_idx]['volume'], marker='v', 
                   s=200, color='red', zorder=5, edgecolors='darkred', linewidths=2)
    
    plt.tight_layout()
    
    # Save
    filename = f'better_chart_{ticker}_RS{radius_strength}_{limit}bars.png'
    plt.savefig(filename, dpi=200, bbox_inches='tight')
    print(f"\n✓ Chart saved: {filename}")
    
    # Show
    plt.show()
    
    # Print detailed signal information
    print(f"\n{'='*80}")
    print(f"DETAILED SIGNAL ANALYSIS")
    print(f"{'='*80}\n")
    
    if len(buy_idx) > 0:
        print(f"BUY SIGNALS ({len(buy_idx)} total):")
        print(f"{'Bar':<6} {'Time':<20} {'Open':<8} {'High':<8} {'Low':<8} {'Close':<8} {'ST':<10}")
        print("-" * 80)
        for idx in buy_idx:
            print(f"{idx:<6} {str(df.iloc[idx]['datetime']):<20} "
                  f"${df.iloc[idx]['open']:<7.2f} ${df.iloc[idx]['high']:<7.2f} "
                  f"${df.iloc[idx]['low']:<7.2f} ${df.iloc[idx]['close']:<7.2f} "
                  f"${curved_band[idx]:<9.2f}")
        print()
    
    if len(sell_idx) > 0:
        print(f"SELL SIGNALS ({len(sell_idx)} total):")
        print(f"{'Bar':<6} {'Time':<20} {'Open':<8} {'High':<8} {'Low':<8} {'Close':<8} {'ST':<10}")
        print("-" * 80)
        for idx in sell_idx:
            print(f"{idx:<6} {str(df.iloc[idx]['datetime']):<20} "
                  f"${df.iloc[idx]['open']:<7.2f} ${df.iloc[idx]['high']:<7.2f} "
                  f"${df.iloc[idx]['low']:<7.2f} ${df.iloc[idx]['close']:<7.2f} "
                  f"${curved_band[idx]:<9.2f}")
        print()
    
    print(f"{'='*80}\n")
    
    return df, signals

if __name__ == "__main__":
    print("\n" + "="*80)
    print("CURVED RADIUS SUPERTREND - BETTER VISUALIZATION")
    print("="*80)
    
    # Create chart with 300 bars for good detail
    df, signals = create_better_chart(
        ticker='QQQ',
        table_name='200309',
        limit=300,
        radius_strength=0.10
    )
    
    print("\n✓ Visualization complete!")
    print("\nThe chart clearly shows:")
    print("  • Green candlesticks = Price closed higher than open (bullish)")
    print("  • Red candlesticks = Price closed lower than open (bearish)")
    print("  • Blue line = Curved Supertrend during uptrend")
    print("  • Orange line = Curved Supertrend during downtrend")
    print("  • Green arrows + 'BUY' labels = Buy signals")
    print("  • Red arrows + 'SELL' labels = Sell signals")
    print("  • Bottom panel shows volume with signals marked")

