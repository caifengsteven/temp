"""
Compare different radius_strength settings side-by-side
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from database_connector import StockDataConnector
from curved_radius_supertrend import CurvedRadiusSupertrend
import warnings
warnings.filterwarnings('ignore')


def plot_comparison(ticker='AAPL', start_date='2023-07-01', end_date='2023-09-30'):
    """
    Generate comparison chart with 4 different radius_strength settings
    """
    
    # Fetch data
    print(f"Fetching {ticker} data from {start_date} to {end_date}...")
    connector = StockDataConnector()
    data = connector.fetch_stock_data(ticker, start_date, end_date)
    connector.close()
    
    print(f"Retrieved {len(data)} trading days")
    
    # Different settings to compare
    settings = [
        {'radius': 0.3, 'name': 'Low (0.3) - Tight Curves'},
        {'radius': 0.5, 'name': 'Medium (0.5) - Balanced'},
        {'radius': 0.8, 'name': 'High (0.8) - Pronounced'},
        {'radius': 1.2, 'name': 'Very High (1.2) - Wide Arcs'}
    ]
    
    # Create figure with 4 subplots
    fig, axes = plt.subplots(2, 2, figsize=(24, 16))
    fig.patch.set_facecolor('#000000')
    axes = axes.flatten()
    
    for idx, setting in enumerate(settings):
        ax = axes[idx]
        ax.set_facecolor('#000000')
        
        radius = setting['radius']
        name = setting['name']
        
        print(f"\nCalculating indicator with radius_strength={radius}...")
        
        # Calculate indicator
        indicator = CurvedRadiusSupertrend(
            atr_period=10,
            atr_multiplier=3.0,
            radius_strength=radius,
            smoothness=3
        )
        
        result = indicator.calculate(
            data['high'].values,
            data['low'].values,
            data['close'].values
        )
        
        # Plot candlesticks
        x_values = np.arange(len(data))
        
        for i in range(len(data)):
            open_price = data['open'].iloc[i]
            high_price = data['high'].iloc[i]
            low_price = data['low'].iloc[i]
            close_price = data['close'].iloc[i]
            
            # Determine color
            if close_price >= open_price:
                color = '#00d9d9'  # Cyan for bullish
                body_color = '#00d9d9'
            else:
                color = '#d900d9'  # Magenta for bearish
                body_color = '#d900d9'
            
            # Draw high-low line (wick)
            ax.plot([i, i], [low_price, high_price], color=color, linewidth=0.8, alpha=0.8)
            
            # Draw body
            body_height = abs(close_price - open_price)
            body_bottom = min(open_price, close_price)
            
            if body_height > 0:
                rect = Rectangle((i - 0.3, body_bottom), 0.6, body_height,
                               facecolor=body_color, edgecolor=body_color, alpha=0.9)
                ax.add_patch(rect)
        
        # Plot trend lines
        uptrend_mask = result['direction'] == 1
        downtrend_mask = result['direction'] == -1
        
        # Uptrend (green)
        trend_up = result['trend_line'].copy()
        trend_up[downtrend_mask] = np.nan
        ax.plot(x_values, trend_up, color='#00ff00', linewidth=3, 
               label='Uptrend', alpha=0.9, zorder=10)
        
        # Downtrend (red)
        trend_down = result['trend_line'].copy()
        trend_down[uptrend_mask] = np.nan
        ax.plot(x_values, trend_down, color='#ff0000', linewidth=3,
               label='Downtrend', alpha=0.9, zorder=10)
        
        # Mark trend changes
        for i in range(1, len(data)):
            if result['direction'].iloc[i] != result['direction'].iloc[i-1]:
                price = data['close'].iloc[i]
                
                if result['direction'].iloc[i] == 1:  # Buy signal
                    ax.scatter(i, price, color='#00ff00', marker='^',
                             s=150, zorder=15, edgecolors='white', linewidths=1)
                else:  # Sell signal
                    ax.scatter(i, price, color='#ff0000', marker='v',
                             s=150, zorder=15, edgecolors='white', linewidths=1)
        
        # Styling
        ax.set_title(name, color='white', fontsize=14, fontweight='bold', pad=10)
        ax.grid(True, color='#333333', linestyle='-', linewidth=0.5, alpha=0.3)
        ax.set_axisbelow(True)
        
        # X-axis labels
        step = max(1, len(data) // 8)
        x_ticks = list(range(0, len(data), step))
        x_labels = [data['date'].iloc[i].strftime('%m-%d') if i < len(data) else '' 
                    for i in x_ticks]
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=9, color='white')
        
        # Y-axis
        ax.tick_params(colors='white', labelsize=9)
        
        # Legend
        legend = ax.legend(loc='upper left', fontsize=9, facecolor='#1a1a1a',
                          edgecolor='#666666')
        for text in legend.get_texts():
            text.set_color('white')
        
        # Add info text
        info_text = f"Radius: {radius}"
        ax.text(0.02, 0.02, info_text, transform=ax.transAxes,
               fontsize=10, color='yellow', verticalalignment='bottom',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='black', 
                        edgecolor='yellow', alpha=0.7))
    
    # Main title
    fig.suptitle(f'{ticker} - Curved Radius Supertrend Comparison\n{start_date} to {end_date}',
                color='white', fontsize=18, fontweight='bold', y=0.995)
    
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    # Save
    filename = f'comparison_{ticker.lower()}_{start_date[:4]}.png'
    plt.savefig(filename, dpi=200, facecolor='#000000', edgecolor='none')
    print(f"\n✅ Comparison chart saved to: {filename}")
    
    plt.show()


def plot_single_comparison(ticker='AAPL', start_date='2023-07-01', end_date='2023-09-30'):
    """
    Generate a single chart with multiple curves overlaid
    """
    
    # Fetch data
    print(f"\nFetching {ticker} data from {start_date} to {end_date}...")
    connector = StockDataConnector()
    data = connector.fetch_stock_data(ticker, start_date, end_date)
    connector.close()
    
    print(f"Retrieved {len(data)} trading days")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(20, 10))
    fig.patch.set_facecolor('#000000')
    ax.set_facecolor('#000000')
    
    # Plot candlesticks
    x_values = np.arange(len(data))
    
    for i in range(len(data)):
        open_price = data['open'].iloc[i]
        high_price = data['high'].iloc[i]
        low_price = data['low'].iloc[i]
        close_price = data['close'].iloc[i]
        
        # Determine color
        if close_price >= open_price:
            color = '#00d9d9'  # Cyan for bullish
            body_color = '#00d9d9'
        else:
            color = '#d900d9'  # Magenta for bearish
            body_color = '#d900d9'
        
        # Draw high-low line (wick)
        ax.plot([i, i], [low_price, high_price], color=color, linewidth=0.8, alpha=0.6)
        
        # Draw body
        body_height = abs(close_price - open_price)
        body_bottom = min(open_price, close_price)
        
        if body_height > 0:
            rect = Rectangle((i - 0.3, body_bottom), 0.6, body_height,
                           facecolor=body_color, edgecolor=body_color, alpha=0.6)
            ax.add_patch(rect)
    
    # Different settings with different colors
    settings = [
        {'radius': 0.3, 'color': '#00ffff', 'name': 'Tight (0.3)', 'style': '-'},
        {'radius': 0.5, 'color': '#00ff00', 'name': 'Medium (0.5)', 'style': '-'},
        {'radius': 0.8, 'color': '#ffff00', 'name': 'High (0.8)', 'style': '-'},
        {'radius': 1.2, 'color': '#ff8800', 'name': 'Very High (1.2)', 'style': '-'}
    ]
    
    for setting in settings:
        radius = setting['radius']
        color = setting['color']
        name = setting['name']
        style = setting['style']
        
        print(f"Calculating with radius_strength={radius}...")
        
        # Calculate indicator
        indicator = CurvedRadiusSupertrend(
            atr_period=10,
            atr_multiplier=3.0,
            radius_strength=radius,
            smoothness=3
        )
        
        result = indicator.calculate(
            data['high'].values,
            data['low'].values,
            data['close'].values
        )
        
        # Plot only uptrend curves for clarity
        uptrend_mask = result['direction'] == 1
        trend_up = result['trend_line'].copy()
        trend_up[~uptrend_mask] = np.nan
        
        ax.plot(x_values, trend_up, color=color, linewidth=2.5, 
               label=name, alpha=0.9, linestyle=style, zorder=10)
    
    # Styling
    ax.set_title(f'{ticker} - Curvature Comparison (Uptrend Only)\n{start_date} to {end_date}',
                color='white', fontsize=16, fontweight='bold', pad=15)
    ax.grid(True, color='#333333', linestyle='-', linewidth=0.5, alpha=0.3)
    ax.set_axisbelow(True)
    
    # X-axis labels
    step = max(1, len(data) // 10)
    x_ticks = list(range(0, len(data), step))
    x_labels = [data['date'].iloc[i].strftime('%Y-%m-%d') if i < len(data) else '' 
                for i in x_ticks]
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=10, color='white')
    
    # Y-axis
    ax.tick_params(colors='white', labelsize=10)
    ax.set_ylabel('Price', color='white', fontsize=12)
    
    # Legend
    legend = ax.legend(loc='upper left', fontsize=11, facecolor='#1a1a1a',
                      edgecolor='#666666', framealpha=0.9)
    for text in legend.get_texts():
        text.set_color('white')
    
    plt.tight_layout()
    
    # Save
    filename = f'overlay_comparison_{ticker.lower()}_{start_date[:4]}.png'
    plt.savefig(filename, dpi=200, facecolor='#000000', edgecolor='none')
    print(f"\n✅ Overlay comparison saved to: {filename}")
    
    plt.show()


if __name__ == "__main__":
    print("="*70)
    print("CURVED RADIUS SUPERTREND - SETTINGS COMPARISON")
    print("="*70)
    
    # Generate 4-panel comparison
    print("\n[1/2] Generating 4-panel comparison...")
    plot_comparison('AAPL', '2023-07-01', '2023-09-30')
    
    # Generate overlay comparison
    print("\n[2/2] Generating overlay comparison...")
    plot_single_comparison('AAPL', '2023-07-01', '2023-09-30')
    
    print("\n" + "="*70)
    print("✅ ALL COMPARISONS COMPLETE!")
    print("="*70)

