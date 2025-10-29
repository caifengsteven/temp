"""
Candlestick Chart Visualization with Curved Radius Supertrend

Creates a professional-looking chart similar to trading platforms with:
- Candlestick bars
- Curved trend lines
- Dark background
- Trade markers
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
import pandas as pd
import numpy as np
from database_connector import StockDataConnector
from curved_radius_supertrend import CurvedRadiusSupertrend
import warnings
warnings.filterwarnings('ignore')


def plot_candlestick_with_supertrend(
    data: pd.DataFrame,
    ticker: str = None,
    show_trades: bool = True,
    save_path: str = None
):
    """
    Create candlestick chart with Curved Radius Supertrend
    
    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame with columns: date, open, high, low, close, volume
    ticker : str
        Stock ticker for title
    show_trades : bool
        Whether to show trade entry/exit points
    save_path : str
        Path to save the figure
    """
    
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
    
    # Add indicator to data
    data = data.copy()
    data['direction'] = result['direction'].values
    data['trend_line'] = result['trend_line'].values
    data['curved_upper'] = result['curved_upper'].values
    data['curved_lower'] = result['curved_lower'].values
    
    # Create figure with dark background (similar to the reference image)
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(20, 10))
    fig.patch.set_facecolor('#000000')
    ax.set_facecolor('#000000')
    
    # Plot candlesticks
    for i in range(len(data)):
        row = data.iloc[i]
        date = i  # Use index for x-axis
        
        open_price = row['open']
        high_price = row['high']
        low_price = row['low']
        close_price = row['close']
        
        # Determine color (cyan/teal for bullish, magenta/pink for bearish)
        if close_price >= open_price:
            color = '#00d9d9'  # Teal/cyan for bullish
            body_color = '#00d9d9'
        else:
            color = '#d900d9'  # Magenta for bearish
            body_color = '#d900d9'
        
        # Draw high-low line (wick)
        ax.plot([date, date], [low_price, high_price], 
               color=color, linewidth=0.8, alpha=0.8)
        
        # Draw body
        body_height = abs(close_price - open_price)
        body_bottom = min(open_price, close_price)
        
        if body_height > 0:
            rect = Rectangle((date - 0.3, body_bottom), 0.6, body_height,
                           facecolor=body_color, edgecolor=body_color,
                           linewidth=0.5, alpha=0.9)
            ax.add_patch(rect)
    
    # Plot curved trend lines
    x_values = np.arange(len(data))
    
    # Separate uptrend and downtrend
    uptrend_mask = data['direction'] == 1
    downtrend_mask = data['direction'] == -1
    
    # Plot uptrend (green)
    trend_up = data['trend_line'].copy()
    trend_up[downtrend_mask] = np.nan
    ax.plot(x_values, trend_up, color='#00ff00', linewidth=2.5, 
           label='Uptrend', alpha=0.9, zorder=10)
    
    # Plot downtrend (red)
    trend_down = data['trend_line'].copy()
    trend_down[uptrend_mask] = np.nan
    ax.plot(x_values, trend_down, color='#ff0000', linewidth=2.5,
           label='Downtrend', alpha=0.9, zorder=10)
    
    # Plot curved bands (optional, dotted)
    ax.plot(x_values, data['curved_upper'], color='#888888', 
           linewidth=1, linestyle=':', alpha=0.3)
    ax.plot(x_values, data['curved_lower'], color='#888888',
           linewidth=1, linestyle=':', alpha=0.3)
    
    # Mark trend changes (trade signals)
    if show_trades:
        for i in range(1, len(data)):
            if data['direction'].iloc[i] != data['direction'].iloc[i-1]:
                price = data['close'].iloc[i]
                
                if data['direction'].iloc[i] == 1:  # Buy signal
                    ax.scatter(i, price, color='#00ff00', marker='^',
                             s=200, zorder=15, edgecolors='white', linewidths=1.5)
                    # Add label
                    ax.annotate(f'{price:.2f}', 
                              xy=(i, price), 
                              xytext=(0, -20),
                              textcoords='offset points',
                              ha='center',
                              fontsize=8,
                              color='#00ff00',
                              bbox=dict(boxstyle='round,pad=0.3', 
                                      facecolor='black', 
                                      edgecolor='#00ff00',
                                      alpha=0.7))
                else:  # Sell signal
                    ax.scatter(i, price, color='#ff0000', marker='v',
                             s=200, zorder=15, edgecolors='white', linewidths=1.5)
                    # Add label
                    ax.annotate(f'{price:.2f}', 
                              xy=(i, price), 
                              xytext=(0, 20),
                              textcoords='offset points',
                              ha='center',
                              fontsize=8,
                              color='#ff0000',
                              bbox=dict(boxstyle='round,pad=0.3', 
                                      facecolor='black', 
                                      edgecolor='#ff0000',
                                      alpha=0.7))
    
    # Customize grid
    ax.grid(True, color='#333333', linestyle='-', linewidth=0.5, alpha=0.3)
    ax.set_axisbelow(True)
    
    # Set x-axis labels (show dates at intervals)
    step = max(1, len(data) // 10)  # Show ~10 date labels
    x_ticks = list(range(0, len(data), step))
    x_labels = [data['date'].iloc[i].strftime('%Y-%m-%d') if i < len(data) else '' 
                for i in x_ticks]
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=9)
    
    # Labels and title
    title = 'Curved Radius Supertrend'
    if ticker:
        title = f'{ticker} - {title}'
    
    ax.set_title(title, fontsize=16, fontweight='bold', color='white', pad=20)
    ax.set_xlabel('Date', fontsize=12, color='white')
    ax.set_ylabel('Price', fontsize=12, color='white')
    
    # Add legend
    ax.legend(loc='upper left', fontsize=10, framealpha=0.7)
    
    # Set y-axis limits with some padding
    y_min = data[['low', 'curved_lower']].min().min()
    y_max = data[['high', 'curved_upper']].max().max()
    y_range = y_max - y_min
    ax.set_ylim(y_min - y_range * 0.05, y_max + y_range * 0.05)
    
    # Tight layout
    plt.tight_layout()
    
    # Save or show
    if save_path:
        plt.savefig(save_path, dpi=200, facecolor='#000000', edgecolor='none', bbox_inches='tight')
        print(f"Chart saved to: {save_path}")

    return fig, ax


def plot_professional_chart(
    ticker: str,
    start_date: str,
    end_date: str,
    radius_strength: float = 0.5,
    save_path: str = None
):
    """
    Create a professional trading chart similar to the reference image

    Parameters:
    -----------
    ticker : str
        Stock ticker symbol
    start_date : str
        Start date (YYYY-MM-DD)
    end_date : str
        End date (YYYY-MM-DD)
    radius_strength : float
        Curvature parameter
    save_path : str
        Path to save the figure
    """

    # Fetch data
    print(f"Fetching {ticker} data...")
    connector = StockDataConnector()

    try:
        data = connector.fetch_stock_data(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            min_volume=100000
        )

        if data.empty:
            print(f"ERROR: No data found for {ticker}")
            return None

        print(f"Retrieved {len(data)} trading days")

    finally:
        connector.close()

    # Create chart
    if save_path is None:
        save_path = f'professional_{ticker.lower()}_{start_date[:4]}.png'

    fig, ax = plot_candlestick_with_supertrend(
        data=data,
        ticker=ticker,
        show_trades=True,
        save_path=save_path
    )

    return fig, ax


def main():
    """Main function to demonstrate the visualization"""
    
    print("\n" + "="*70)
    print("CANDLESTICK CHART WITH CURVED RADIUS SUPERTREND")
    print("="*70)
    
    # Fetch data
    ticker = 'AAPL'
    start_date = '2023-01-01'
    end_date = '2023-12-31'
    
    print(f"\nFetching {ticker} data from {start_date} to {end_date}...")
    
    connector = StockDataConnector()
    try:
        data = connector.fetch_stock_data(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            min_volume=1000000
        )
        
        if data.empty:
            print(f"ERROR: No data found for {ticker}")
            return
        
        print(f"Retrieved {len(data)} trading days")
        
    finally:
        connector.close()
    
    # Create visualization
    print("\nGenerating candlestick chart...")
    
    fig, ax = plot_candlestick_with_supertrend(
        data=data,
        ticker=ticker,
        show_trades=True,
        save_path=f'candlestick_{ticker.lower()}_2023.png'
    )
    
    plt.show()
    
    print("\n" + "="*70)
    print("Visualization complete!")
    print("="*70)
    
    # Also create for GOOGL
    print(f"\nGenerating chart for GOOGL...")
    
    connector = StockDataConnector()
    try:
        data_googl = connector.fetch_stock_data(
            ticker='GOOGL',
            start_date='2023-01-01',
            end_date='2023-12-31',
            min_volume=1000000
        )
        
        if not data_googl.empty:
            fig2, ax2 = plot_candlestick_with_supertrend(
                data=data_googl,
                ticker='GOOGL',
                show_trades=True,
                save_path='candlestick_googl_2023.png'
            )
            plt.show()
        
    finally:
        connector.close()


if __name__ == "__main__":
    main()

