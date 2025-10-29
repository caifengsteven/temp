"""
Visualize Multiple Stocks with Curved Supertrend and Buy/Sell Signals
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from database_connector import StockDataConnector
from curved_radius_supertrend import CurvedRadiusSupertrend
from backtest_engine import BacktestEngine

def visualize_stock_with_signals(ticker, start_date, end_date, ax, radius_strength=1.2):
    """
    Visualize a single stock with curved supertrend and buy/sell signals
    """
    # Fetch data
    connector = StockDataConnector()
    data = connector.fetch_stock_data(ticker, start_date, end_date)
    connector.close()
    
    if len(data) < 100:
        print(f"Not enough data for {ticker}")
        return None
    
    # Calculate indicator
    indicator = CurvedRadiusSupertrend(
        atr_period=10,
        atr_multiplier=3.0,
        radius_strength=radius_strength,
        smoothness=3
    )

    result = indicator.calculate(
        high=data['high'].values,
        low=data['low'].values,
        close=data['close'].values
    )

    # Add OHLC and date columns to result
    result['date'] = data['date'].values
    result['open'] = data['open'].values
    result['high'] = data['high'].values
    result['low'] = data['low'].values
    result['close'] = data['close'].values
    
    # Run backtest to get trades
    engine = BacktestEngine(
        initial_capital=100000,
        commission=0.001,
        slippage=0.0005,
        position_size=0.10,
        allow_short=True
    )
    
    indicator_params = {
        'atr_period': 10,
        'atr_multiplier': 3.0,
        'radius_strength': radius_strength,
        'smoothness': 3
    }
    
    backtest_result = engine.run_backtest(data=data, indicator_params=indicator_params)
    stats = engine.calculate_statistics(data)
    
    # Plot candlesticks
    for i in range(len(result)):
        row = result.iloc[i]
        
        # Determine color
        if row['close'] >= row['open']:
            color = '#00CED1'  # Cyan for bullish
            edge_color = '#00CED1'
        else:
            color = '#FF1493'  # Magenta for bearish
            edge_color = '#FF1493'
        
        # Draw candlestick
        high_low = ax.plot([i, i], [row['low'], row['high']], 
                          color=edge_color, linewidth=0.8, alpha=0.6)
        
        body_height = abs(row['close'] - row['open'])
        body_bottom = min(row['open'], row['close'])
        
        if body_height > 0:
            rect = FancyBboxPatch(
                (i - 0.3, body_bottom), 0.6, body_height,
                boxstyle="round,pad=0.02",
                edgecolor=edge_color,
                facecolor=color,
                linewidth=0.8,
                alpha=0.8
            )
            ax.add_patch(rect)
    
    # Plot curved trend lines
    uptrend_indices = result[result['direction'] == 1].index
    downtrend_indices = result[result['direction'] == -1].index
    
    # Plot uptrend curve (green)
    if len(uptrend_indices) > 0:
        uptrend_curve = result.loc[uptrend_indices, 'curved_lower'].values
        ax.plot(uptrend_indices, uptrend_curve, 
               color='#00FF00', linewidth=2.5, label='Uptrend', alpha=0.9)
    
    # Plot downtrend curve (red)
    if len(downtrend_indices) > 0:
        downtrend_curve = result.loc[downtrend_indices, 'curved_upper'].values
        ax.plot(downtrend_indices, downtrend_curve, 
               color='#FF0000', linewidth=2.5, label='Downtrend', alpha=0.9)
    
    # Plot buy/sell signals
    for trade in engine.trades:
        # Find entry index
        entry_idx = result[result['date'] == trade.entry_date].index
        if len(entry_idx) > 0:
            entry_idx = entry_idx[0]
            entry_price = trade.entry_price
            
            if trade.direction == 'LONG':
                # Buy signal (green triangle up)
                ax.scatter(entry_idx, entry_price, marker='^', 
                          color='#00FF00', s=200, zorder=5, 
                          edgecolors='white', linewidths=1.5)
                ax.text(entry_idx, entry_price * 0.97, f'${entry_price:.2f}',
                       ha='center', va='top', fontsize=7, color='#00FF00',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='black', 
                                edgecolor='#00FF00', alpha=0.8))
            else:
                # Short signal (red triangle down)
                ax.scatter(entry_idx, entry_price, marker='v', 
                          color='#FF0000', s=200, zorder=5,
                          edgecolors='white', linewidths=1.5)
                ax.text(entry_idx, entry_price * 1.03, f'${entry_price:.2f}',
                       ha='center', va='bottom', fontsize=7, color='#FF0000',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='black', 
                                edgecolor='#FF0000', alpha=0.8))
        
        # Find exit index
        if trade.exit_date is not None:
            exit_idx = result[result['date'] == trade.exit_date].index
            if len(exit_idx) > 0:
                exit_idx = exit_idx[0]
                exit_price = trade.exit_price
                
                # Exit signal (yellow X)
                ax.scatter(exit_idx, exit_price, marker='x', 
                          color='#FFFF00', s=150, zorder=5, linewidths=2.5)
    
    # Styling
    ax.set_facecolor('#0a0a0a')
    ax.grid(True, alpha=0.15, color='#404040', linestyle='--', linewidth=0.5)
    ax.spines['bottom'].set_color('#404040')
    ax.spines['top'].set_color('#404040')
    ax.spines['left'].set_color('#404040')
    ax.spines['right'].set_color('#404040')
    ax.tick_params(colors='#808080', labelsize=8)
    
    # Title with stats
    title = f'{ticker} | Return: {stats["total_return_pct"]:.1f}% | Sharpe: {stats["sharpe_ratio"]:.2f} | Drawdown: {stats["max_drawdown_pct"]:.1f}% | Trades: {stats["total_trades"]}'
    ax.set_title(title, color='white', fontsize=10, fontweight='bold', pad=10)
    
    # Format x-axis to show dates
    num_ticks = 8
    tick_indices = np.linspace(0, len(result)-1, num_ticks, dtype=int)
    tick_labels = [result.iloc[i]['date'].strftime('%Y-%m') for i in tick_indices]
    ax.set_xticks(tick_indices)
    ax.set_xticklabels(tick_labels, rotation=45, ha='right')
    
    ax.set_ylabel('Price ($)', color='#808080', fontsize=9)
    
    return stats


# Select stocks to visualize
stocks_to_show = [
    ('AXON', 'Top Performer'),      # Best performer
    ('AAPL', 'Large Cap'),          # Large cap tech
    ('TSLA', 'High Volatility'),    # High volatility
    ('GE', 'Industrial'),           # Industrial
    ('NVDA', 'Semiconductor'),      # Semiconductor
    ('META', 'Social Media'),       # Social media
]

# Create figure
fig = plt.figure(figsize=(20, 24))
fig.patch.set_facecolor('#0a0a0a')

print("=" * 100)
print("VISUALIZING STOCKS WITH CURVED SUPERTREND AND BUY/SELL SIGNALS")
print("=" * 100)
print()

# Plot each stock
for idx, (ticker, category) in enumerate(stocks_to_show, 1):
    print(f"Processing {idx}/{len(stocks_to_show)}: {ticker} ({category})...")
    
    ax = plt.subplot(len(stocks_to_show), 1, idx)
    
    try:
        stats = visualize_stock_with_signals(
            ticker=ticker,
            start_date='2020-01-01',
            end_date='2025-01-01',
            ax=ax,
            radius_strength=1.2
        )
        
        if stats:
            print(f"   ✓ {ticker}: Return={stats['total_return_pct']:.1f}%, Sharpe={stats['sharpe_ratio']:.2f}, Trades={stats['total_trades']}")
        else:
            print(f"   ✗ {ticker}: Failed to process")
            
    except Exception as e:
        print(f"   ✗ {ticker}: Error - {str(e)}")
        ax.text(0.5, 0.5, f'Error loading {ticker}', 
               ha='center', va='center', transform=ax.transAxes,
               color='red', fontsize=12)
        ax.set_facecolor('#0a0a0a')

print()
print("=" * 100)

# Add legend to first subplot
axes = fig.get_axes()
if len(axes) > 0:
    legend = axes[0].legend(loc='upper left', fontsize=9, framealpha=0.8,
                           facecolor='#1a1a1a', edgecolor='#404040')
    # Set legend text color
    for text in legend.get_texts():
        text.set_color('white')

# Add overall title
fig.suptitle('Curved Radius Supertrend Strategy - Multiple Stocks (2020-2025)\n10% Position Sizing | Green=Buy | Red=Short | Yellow X=Exit',
            color='white', fontsize=14, fontweight='bold', y=0.995)

plt.tight_layout(rect=[0, 0, 1, 0.995])

# Save figure
output_file = 'multiple_stocks_with_signals.png'
plt.savefig(output_file, dpi=150, facecolor='#0a0a0a', edgecolor='none', bbox_inches='tight')
print(f"✅ Chart saved to: {output_file}")
print()

# Show the plot
plt.show()

print("=" * 100)
print("✅ VISUALIZATION COMPLETE!")
print("=" * 100)

