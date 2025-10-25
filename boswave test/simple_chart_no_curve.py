"""
Simple chart - just price and signals, no curved line
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

def create_simple_chart(ticker='QQQ', table_name='200309', limit=300, radius_strength=0.10):
    """
    Create a simple chart - just candlesticks and signals
    """
    
    print(f"\n{'='*80}")
    print(f"Creating Simple Chart for {ticker} (No Curve Line)")
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
    print(f"\n[2/3] Calculating signals...")
    indicator = CurvedRadiusSupertrendExact(
        atr_length=14,
        atr_mult=2.0,
        radius_strength=radius_strength,
        smoothness=5
    )
    
    signals = indicator.calculate(df['high'].values, df['low'].values, df['close'].values)
    
    buy_signals = signals['buy_signals']
    sell_signals = signals['sell_signals']
    direction = signals['direction']
    
    print(f"✓ Signals calculated")
    print(f"  Buy signals: {buy_signals.sum()}")
    print(f"  Sell signals: {sell_signals.sum()}")
    
    # Create visualization
    print(f"\n[3/3] Creating chart...")
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 12), 
                                    gridspec_kw={'height_ratios': [3, 1]}, 
                                    sharex=True)
    
    # ========================================================================
    # Main Chart: Candlesticks + Signals ONLY
    # ========================================================================
    
    # Draw candlesticks
    width = 0.6
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
    
    # Add background shading for trend direction
    for i in range(len(direction)):
        if direction[i] == 1:  # Uptrend
            ax1.axvspan(i-0.5, i+0.5, alpha=0.05, color='green', zorder=0)
        else:  # Downtrend
            ax1.axvspan(i-0.5, i+0.5, alpha=0.05, color='red', zorder=0)
    
    # Plot Buy signals
    buy_idx = np.where(buy_signals)[0]
    if len(buy_idx) > 0:
        for idx in buy_idx:
            # Draw arrow from below
            arrow_y = df.iloc[idx]['low'] - (df['high'].max() - df['low'].min()) * 0.03
            ax1.annotate('', xy=(idx, df.iloc[idx]['low']), 
                        xytext=(idx, arrow_y),
                        arrowprops=dict(arrowstyle='->', color='green', lw=4))
            ax1.text(idx, arrow_y - (df['high'].max() - df['low'].min()) * 0.01, 
                    'BUY', ha='center', va='top', 
                    fontsize=10, fontweight='bold', color='white',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='green', alpha=0.95))
    
    # Plot Sell signals
    sell_idx = np.where(sell_signals)[0]
    if len(sell_idx) > 0:
        for idx in sell_idx:
            # Draw arrow from above
            arrow_y = df.iloc[idx]['high'] + (df['high'].max() - df['low'].min()) * 0.03
            ax1.annotate('', xy=(idx, df.iloc[idx]['high']), 
                        xytext=(idx, arrow_y),
                        arrowprops=dict(arrowstyle='->', color='red', lw=4))
            ax1.text(idx, arrow_y + (df['high'].max() - df['low'].min()) * 0.01, 
                    'SELL', ha='center', va='bottom', 
                    fontsize=10, fontweight='bold', color='white',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='red', alpha=0.95))
    
    # Formatting
    ax1.set_ylabel('Price ($)', fontsize=14, fontweight='bold')
    ax1.set_title(f'{ticker} 1-Minute Chart with Buy/Sell Signals', 
                 fontsize=16, fontweight='bold', pad=20)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_xlim(-5, len(df) + 5)
    
    # Set y-axis to show only relevant price range
    price_range = df['high'].max() - df['low'].min()
    ax1.set_ylim(df['low'].min() - price_range * 0.1, 
                 df['high'].max() + price_range * 0.1)
    
    # Add legend
    legend_elements = [
        Line2D([0], [0], color=up_color, lw=4, label='Bullish Candle (Close > Open)'),
        Line2D([0], [0], color=down_color, lw=4, label='Bearish Candle (Close < Open)'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='green', 
               markersize=12, label='BUY Signal (Trend: Down→Up)', linestyle='None'),
        Line2D([0], [0], marker='v', color='w', markerfacecolor='red', 
               markersize=12, label='SELL Signal (Trend: Up→Down)', linestyle='None'),
    ]
    ax1.legend(handles=legend_elements, loc='upper left', fontsize=11, framealpha=0.95)
    
    # Add info box
    info_text = f"""Chart Info:
━━━━━━━━━━━━━━━━━━━━━━━━━━━
Symbol: {ticker}
Bars: {len(df)}
Timeframe: 1 minute
Date: {df['datetime'].min().strftime('%Y-%m-%d')}
Time: {df['datetime'].min().strftime('%H:%M')} - {df['datetime'].max().strftime('%H:%M')}

Price Range:
  High: ${df['high'].max():.2f}
  Low:  ${df['low'].min():.2f}
  Range: ${price_range:.2f}

Signals:
  BUY:  {buy_signals.sum()}
  SELL: {sell_signals.sum()}
  Total: {buy_signals.sum() + sell_signals.sum()}

Background Shading:
  Light Green = Uptrend
  Light Red = Downtrend"""
    
    ax1.text(0.99, 0.97, info_text, transform=ax1.transAxes,
            fontsize=9, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.95),
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
    filename = f'simple_chart_{ticker}_{limit}bars.png'
    plt.savefig(filename, dpi=200, bbox_inches='tight')
    print(f"\n✓ Chart saved: {filename}")
    
    # Show
    plt.show()
    
    # Print detailed signal information
    print(f"\n{'='*80}")
    print(f"SIGNAL DETAILS")
    print(f"{'='*80}\n")
    
    if len(buy_idx) > 0:
        print(f"BUY SIGNALS ({len(buy_idx)} total):")
        print(f"{'Bar':<6} {'Time':<20} {'Open':<8} {'High':<8} {'Low':<8} {'Close':<8} {'Volume':<10}")
        print("-" * 80)
        for idx in buy_idx:
            print(f"{idx:<6} {str(df.iloc[idx]['datetime']):<20} "
                  f"${df.iloc[idx]['open']:<7.2f} ${df.iloc[idx]['high']:<7.2f} "
                  f"${df.iloc[idx]['low']:<7.2f} ${df.iloc[idx]['close']:<7.2f} "
                  f"{df.iloc[idx]['volume']:<10}")
        print()
    
    if len(sell_idx) > 0:
        print(f"SELL SIGNALS ({len(sell_idx)} total):")
        print(f"{'Bar':<6} {'Time':<20} {'Open':<8} {'High':<8} {'Low':<8} {'Close':<8} {'Volume':<10}")
        print("-" * 80)
        for idx in sell_idx:
            print(f"{idx:<6} {str(df.iloc[idx]['datetime']):<20} "
                  f"${df.iloc[idx]['open']:<7.2f} ${df.iloc[idx]['high']:<7.2f} "
                  f"${df.iloc[idx]['low']:<7.2f} ${df.iloc[idx]['close']:<7.2f} "
                  f"{df.iloc[idx]['volume']:<10}")
        print()
    
    print(f"{'='*80}\n")
    
    return df, signals

if __name__ == "__main__":
    print("\n" + "="*80)
    print("QQQ CHART - CANDLESTICKS AND SIGNALS ONLY")
    print("="*80)
    
    # Create chart with 300 bars
    df, signals = create_simple_chart(
        ticker='QQQ',
        table_name='200309',
        limit=300,
        radius_strength=0.10
    )
    
    print("\n✓ Chart complete!")
    print("\nThe chart shows:")
    print("  ✓ Proper QQQ candlesticks (green=bullish, red=bearish)")
    print("  ✓ Buy signals marked with green arrows and 'BUY' labels")
    print("  ✓ Sell signals marked with red arrows and 'SELL' labels")
    print("  ✓ Background shading shows trend direction")
    print("  ✓ Volume panel at bottom")
    print("  ✓ NO curved line (removed due to scale issues)")

