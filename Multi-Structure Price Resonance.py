import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta, time
import matplotlib.dates as mdates
import pytz
import random

# Function to generate simulated price data with intraday patterns
def generate_simulated_data(days=10, start_date=None, volatility=0.0015, trend_strength=0.0002, 
                           trend_changes=4, intraday_pattern=True, seed=42):
    """
    Generate simulated price data with intraday patterns and New York session features
    
    Parameters:
    - days: Number of days to simulate
    - start_date: Starting date (if None, uses current date - days)
    - volatility: Base volatility level
    - trend_strength: Strength of trend component
    - trend_changes: Number of trend changes in the period
    - intraday_pattern: Whether to add intraday patterns
    - seed: Random seed for reproducibility
    
    Returns:
    - DataFrame with datetime index and OHLCV data
    """
    np.random.seed(seed)
    
    # Create date range
    if start_date is None:
        end_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        start_date = end_date - timedelta(days=days)
    else:
        start_date = pd.to_datetime(start_date).replace(hour=0, minute=0, second=0, microsecond=0)
        end_date = start_date + timedelta(days=days)
    
    # Create 3-minute intervals from 00:00 to 23:57 for each day
    dates = []
    current_date = start_date
    while current_date < end_date:
        for hour in range(24):
            for minute in range(0, 60, 3):  # 3-minute intervals
                dates.append(current_date.replace(hour=hour, minute=minute))
        current_date += timedelta(days=1)
    
    # Initialize price data
    n = len(dates)
    close = np.zeros(n)
    open_price = np.zeros(n)
    high = np.zeros(n)
    low = np.zeros(n)
    volume = np.zeros(n)
    
    # Starting price
    close[0] = 100
    open_price[0] = 100
    high[0] = 100
    low[0] = 100
    volume[0] = 1000000
    
    # Generate trend changes
    trend_periods = np.linspace(0, n, trend_changes + 1).astype(int)
    trends = np.random.choice([-1, 1], size=trend_changes) * trend_strength
    
    # For intraday pattern effect
    ny_tz = pytz.timezone('America/New_York')
    
    # Generate prices with trends and random walks
    current_trend = 0
    for i in range(1, n):
        # Check if we need to change trend
        if i in trend_periods[1:]:
            current_trend += 1
        
        # Current date in NY timezone
        dt = dates[i]
        dt_ny = dt.astimezone(ny_tz) if dt.tzinfo else ny_tz.localize(dt)
        hour_ny = dt_ny.hour
        minute_ny = dt_ny.minute
        
        # Determine if current time is in NY session
        in_ny_morning = (9 <= hour_ny < 12) or (hour_ny == 9 and minute_ny >= 30)
        in_ny_afternoon = (13 <= hour_ny < 16) or (hour_ny == 13 and minute_ny >= 30)
        in_ny_session = in_ny_morning or in_ny_afternoon
        
        # Add trend component
        trend = trends[min(current_trend, trend_changes-1)]
        
        # Adjust volatility based on NY session - higher during NY hours
        current_volatility = volatility
        if intraday_pattern:
            if in_ny_session:
                current_volatility *= 1.5  # Higher volatility during NY sessions
            elif hour_ny >= 20 or hour_ny < 4:  # Asian session has moderate volatility
                current_volatility *= 0.8
            
            # Add special events with higher volatility (e.g., news releases)
            if random.random() < 0.005:  # 0.5% chance of news event
                current_volatility *= 3
        
        # Generate random component with current volatility
        random_change = np.random.normal(0, current_volatility)
        
        # Calculate close price
        close[i] = close[i-1] * (1 + trend + random_change)
        
        # Generate open with gap possibility
        gap = np.random.normal(0, current_volatility/2)
        open_price[i] = close[i-1] * (1 + gap)
        
        # Calculate price range
        price_range = close[i] * current_volatility * 2
        
        # Generate high and low
        if open_price[i] > close[i]:  # Bearish candle
            high[i] = open_price[i] + abs(np.random.normal(0, price_range/2))
            low[i] = close[i] - abs(np.random.normal(0, price_range/2))
        else:  # Bullish candle
            high[i] = close[i] + abs(np.random.normal(0, price_range/2))
            low[i] = open_price[i] - abs(np.random.normal(0, price_range/2))
        
        # Generate volume with correlation to volatility and NY session
        base_volume = 1000000
        volume_multiplier = 1.0
        
        if intraday_pattern:
            if in_ny_session:
                volume_multiplier = 1.5  # Higher volume during NY sessions
            elif hour_ny >= 20 or hour_ny < 4:  # Lower volume during Asian session
                volume_multiplier = 0.7
            
            # Add some randomness to volume
            volume_noise = np.random.normal(0, 0.3)
            volume_multiplier *= (1 + volume_noise)
            
            # Higher volume on price movements
            price_change_pct = abs((close[i] - close[i-1]) / close[i-1])
            volume_multiplier *= (1 + price_change_pct * 5)
        
        volume[i] = base_volume * max(0.2, volume_multiplier)
        
        # Create fair value gaps and order blocks occasionally
        if i > 2 and random.random() < 0.1:  # 10% chance of creating a pattern
            if random.random() < 0.5:  # Bullish pattern
                # Create bullish order block pattern
                open_price[i-2] = close[i-2] * 1.002
                close[i-2] = open_price[i-2] * 0.998  # Bearish candle
                
                open_price[i-1] = close[i-2]
                close[i-1] = open_price[i-1] * 1.001  # Slightly bullish
                
                open_price[i] = close[i-1]
                close[i] = open_price[i] * 1.003  # More bullish
                
                # Create bullish FVG (low[i] > high[i-2])
                high[i-2] = max(open_price[i-2], close[i-2]) * 1.001
                low[i-2] = min(open_price[i-2], close[i-2]) * 0.999
                
                high[i] = max(open_price[i], close[i]) * 1.001
                low[i] = high[i-2] * 1.002  # Ensure low is above high[i-2]
            else:  # Bearish pattern
                # Create bearish order block pattern
                open_price[i-2] = close[i-2] * 0.998
                close[i-2] = open_price[i-2] * 1.002  # Bullish candle
                
                open_price[i-1] = close[i-2]
                close[i-1] = open_price[i-1] * 0.999  # Slightly bearish
                
                open_price[i] = close[i-1]
                close[i] = open_price[i] * 0.997  # More bearish
                
                # Create bearish FVG (high[i] < low[i-2])
                high[i-2] = max(open_price[i-2], close[i-2]) * 1.001
                low[i-2] = min(open_price[i-2], close[i-2]) * 0.999
                
                high[i] = low[i-2] * 0.998  # Ensure high is below low[i-2]
                low[i] = min(open_price[i], close[i]) * 0.999
    
    # Create DataFrame
    df = pd.DataFrame({
        'datetime': dates,
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume.astype(int)
    })
    
    # Ensure logical OHLC values
    for i in range(len(df)):
        df.loc[i, 'high'] = max(df.loc[i, 'open'], df.loc[i, 'close'], df.loc[i, 'high'])
        df.loc[i, 'low'] = min(df.loc[i, 'open'], df.loc[i, 'close'], df.loc[i, 'low'])
    
    # Set datetime as index
    df.set_index('datetime', inplace=True)
    
    return df

# Check if a datetime is within New York trading session
def is_in_ny_session(dt):
    """
    Check if a datetime is within New York trading session hours
    
    Parameters:
    - dt: datetime object (must be timezone-aware or will be interpreted as UTC)
    
    Returns:
    - Boolean indicating if in NY session
    """
    ny_tz = pytz.timezone('America/New_York')
    if dt.tzinfo is None:
        dt = pytz.utc.localize(dt)
    dt_ny = dt.astimezone(ny_tz)
    
    # NY morning session: 9:30 AM - 12:00 PM
    morning_start = dt_ny.replace(hour=9, minute=30, second=0, microsecond=0)
    morning_end = dt_ny.replace(hour=12, minute=0, second=0, microsecond=0)
    
    # NY afternoon session: 1:30 PM - 4:00 PM
    afternoon_start = dt_ny.replace(hour=13, minute=30, second=0, microsecond=0)
    afternoon_end = dt_ny.replace(hour=16, minute=0, second=0, microsecond=0)
    
    return ((dt_ny >= morning_start and dt_ny < morning_end) or 
            (dt_ny >= afternoon_start and dt_ny < afternoon_end))

# Implement the Multi-Structure Price Resonance Strategy
def multi_structure_resonance_strategy(df):
    """
    Implement the Multi-Structure Price Resonance Strategy
    
    Parameters:
    - df: DataFrame with OHLC data
    
    Returns:
    - DataFrame with strategy results
    """
    # Make a copy of the dataframe to avoid modifying the original
    df = df.copy()
    
    # Check if index is timezone-aware, if not, localize to UTC
    if df.index.tzinfo is None:
        df.index = df.index.tz_localize('UTC')
    
    # Calculate New York session
    df['in_ny_session'] = df.index.map(is_in_ny_session)
    
    # Order Block logic
    df['bullish_ob'] = ((df['close'].shift(2) < df['open'].shift(2)) & 
                       (df['close'].shift(1) > df['close'].shift(2)) & 
                       (df['close'] > df['close'].shift(1)))
    
    df['bearish_ob'] = ((df['close'].shift(2) > df['open'].shift(2)) & 
                       (df['close'].shift(1) < df['close'].shift(2)) & 
                       (df['close'] < df['close'].shift(1)))
    
    # Fair Value Gap logic
    df['bullish_fvg'] = df['low'] > df['high'].shift(2)
    df['bearish_fvg'] = df['high'] < df['low'].shift(2)
    
    # Signal conditions
    df['bullish_signal'] = df['bullish_ob'] & df['bullish_fvg'] & df['in_ny_session']
    df['bearish_signal'] = df['bearish_ob'] & df['bearish_fvg'] & df['in_ny_session']
    
    # Trade management
    df['entry_price'] = df['close']
    df['sl_long'] = df['low'].shift(2)
    df['tp_long'] = df.apply(lambda x: x['entry_price'] + 2 * (x['entry_price'] - x['sl_long']) 
                           if x['bullish_signal'] else np.nan, axis=1)
    
    df['sl_short'] = df['high'].shift(2)
    df['tp_short'] = df.apply(lambda x: x['entry_price'] - 2 * (x['sl_short'] - x['entry_price']) 
                            if x['bearish_signal'] else np.nan, axis=1)
    
    # Initialize strategy columns
    df['position'] = 0  # 1 for long, -1 for short, 0 for flat
    df['exit_price'] = np.nan
    df['exit_type'] = ''  # 'tp', 'sl', or 'session_end'
    df['pnl'] = 0.0
    df['cumulative_pnl'] = 0.0
    
    # Apply strategy logic
    position = 0
    entry_price = 0
    stop_loss = 0
    take_profit = 0
    entry_index = None
    
    for i in range(len(df)):
        idx = df.index[i]
        
        # First, check if we need to close existing position
        if position == 1:  # Long position
            # Check if stop loss hit
            if df['low'].iloc[i] <= stop_loss:
                df.loc[idx, 'exit_price'] = stop_loss
                df.loc[idx, 'pnl'] = (stop_loss / entry_price - 1) * 100
                df.loc[idx, 'exit_type'] = 'sl'
                position = 0
            
            # Check if take profit hit
            elif df['high'].iloc[i] >= take_profit:
                df.loc[idx, 'exit_price'] = take_profit
                df.loc[idx, 'pnl'] = (take_profit / entry_price - 1) * 100
                df.loc[idx, 'exit_type'] = 'tp'
                position = 0
            
            # Update position
            df.loc[idx, 'position'] = position
        
        elif position == -1:  # Short position
            # Check if stop loss hit
            if df['high'].iloc[i] >= stop_loss:
                df.loc[idx, 'exit_price'] = stop_loss
                df.loc[idx, 'pnl'] = (entry_price / stop_loss - 1) * 100
                df.loc[idx, 'exit_type'] = 'sl'
                position = 0
            
            # Check if take profit hit
            elif df['low'].iloc[i] <= take_profit:
                df.loc[idx, 'exit_price'] = take_profit
                df.loc[idx, 'pnl'] = (entry_price / take_profit - 1) * 100
                df.loc[idx, 'exit_type'] = 'tp'
                position = 0
            
            # Update position
            df.loc[idx, 'position'] = position
        
        # Now check for new entry signals if we're flat
        if position == 0:
            # Check for bullish signal
            if df['bullish_signal'].iloc[i]:
                position = 1
                entry_price = df['entry_price'].iloc[i]
                stop_loss = df['sl_long'].iloc[i]
                take_profit = df['tp_long'].iloc[i]
                entry_index = idx
            
            # Check for bearish signal
            elif df['bearish_signal'].iloc[i]:
                position = -1
                entry_price = df['entry_price'].iloc[i]
                stop_loss = df['sl_short'].iloc[i]
                take_profit = df['tp_short'].iloc[i]
                entry_index = idx
            
            # Update position
            df.loc[idx, 'position'] = position
    
    # Close any open position at the end of the period
    if position != 0:
        last_idx = df.index[-1]
        last_price = df['close'].iloc[-1]
        
        if position == 1:  # Long position
            df.loc[last_idx, 'exit_price'] = last_price
            df.loc[last_idx, 'pnl'] = (last_price / entry_price - 1) * 100
            df.loc[last_idx, 'exit_type'] = 'session_end'
        else:  # Short position
            df.loc[last_idx, 'exit_price'] = last_price
            df.loc[last_idx, 'pnl'] = (entry_price / last_price - 1) * 100
            df.loc[last_idx, 'exit_type'] = 'session_end'
    
    # Calculate cumulative P&L
    df['cumulative_pnl'] = df['pnl'].cumsum()
    
    return df

# Analyze strategy performance
def analyze_performance(df):
    """
    Calculate performance metrics for the strategy
    
    Parameters:
    - df: DataFrame with strategy results
    
    Returns:
    - Dictionary with performance metrics
    """
    # Filter completed trades
    trades = df[df['pnl'] != 0].copy()
    
    # If no trades, return early
    if len(trades) == 0:
        return {
            'total_trades': 0,
            'win_rate': 0,
            'avg_return': 0,
            'total_return': 0,
            'max_drawdown': 0,
            'sharpe_ratio': 0,
            'profit_factor': 0,
            'long_trades': 0,
            'short_trades': 0,
            'tp_exits': 0,
            'sl_exits': 0,
            'session_end_exits': 0
        }
    
    # Calculate performance metrics
    total_trades = len(trades)
    winning_trades = len(trades[trades['pnl'] > 0])
    win_rate = winning_trades / total_trades if total_trades > 0 else 0
    avg_return = trades['pnl'].mean()
    total_return = trades['pnl'].sum()
    
    # Count trade types
    long_trades = len(trades[trades['position'].shift(1) == 1])
    short_trades = len(trades[trades['position'].shift(1) == -1])
    
    # Count exit types
    tp_exits = len(trades[trades['exit_type'] == 'tp'])
    sl_exits = len(trades[trades['exit_type'] == 'sl'])
    session_end_exits = len(trades[trades['exit_type'] == 'session_end'])
    
    # Calculate profit factor
    gross_profit = trades[trades['pnl'] > 0]['pnl'].sum()
    gross_loss = abs(trades[trades['pnl'] < 0]['pnl'].sum())
    profit_factor = gross_profit / gross_loss if gross_loss != 0 else float('inf')
    
    # Calculate drawdown
    df['peak'] = df['cumulative_pnl'].cummax()
    df['drawdown'] = df['cumulative_pnl'] - df['peak']
    max_drawdown = df['drawdown'].min()
    
    # Calculate Sharpe Ratio (simplified)
    if trades['pnl'].std() != 0:
        sharpe_ratio = (trades['pnl'].mean() / trades['pnl'].std()) * np.sqrt(252/total_trades)  # Annualized
    else:
        sharpe_ratio = 0
    
    return {
        'total_trades': total_trades,
        'win_rate': win_rate,
        'avg_return': avg_return,
        'total_return': total_return,
        'max_drawdown': max_drawdown,
        'sharpe_ratio': sharpe_ratio,
        'profit_factor': profit_factor,
        'long_trades': long_trades,
        'short_trades': short_trades,
        'tp_exits': tp_exits,
        'sl_exits': sl_exits,
        'session_end_exits': session_end_exits
    }

# Plot strategy results
def plot_strategy_results(df, start_date=None, end_date=None):
    """
    Visualize strategy results
    
    Parameters:
    - df: DataFrame with strategy results
    - start_date: Optional start date for visualization
    - end_date: Optional end date for visualization
    """
    # Filter data if date range provided
    if start_date is not None:
        df = df[df.index >= start_date]
    if end_date is not None:
        df = df[df.index <= end_date]
    
    # Create figure
    fig = plt.figure(figsize=(16, 16))
    
    # Price chart with signals
    ax1 = plt.subplot(4, 1, 1)
    
    # Plot OHLC as candlesticks
    width = 0.0002  # Width of candlestick bodies
    up = df[df['close'] >= df['open']]
    down = df[df['close'] < df['open']]
    
    # Plot up candles
    ax1.bar(up.index, up['close'] - up['open'], width, bottom=up['open'], color='green', alpha=0.6)
    ax1.bar(up.index, up['high'] - up['close'], width*0.2, bottom=up['close'], color='green', alpha=0.6)
    ax1.bar(up.index, up['open'] - up['low'], width*0.2, bottom=up['low'], color='green', alpha=0.6)
    
    # Plot down candles
    ax1.bar(down.index, down['open'] - down['close'], width, bottom=down['close'], color='red', alpha=0.6)
    ax1.bar(down.index, down['high'] - down['open'], width*0.2, bottom=down['open'], color='red', alpha=0.6)
    ax1.bar(down.index, down['close'] - down['low'], width*0.2, bottom=down['low'], color='red', alpha=0.6)
    
    # Highlight NY trading sessions
    ny_sessions = df[df['in_ny_session']]
    for idx in ny_sessions.index:
        ax1.axvspan(idx, idx + pd.Timedelta(minutes=3), alpha=0.1, color='blue')
    
    # Plot entry and exit points
    long_entries = df[df['bullish_signal']].index
    short_entries = df[df['bearish_signal']].index
    tp_exits = df[df['exit_type'] == 'tp'].index
    sl_exits = df[df['exit_type'] == 'sl'].index
    
    if len(long_entries) > 0:
        ax1.scatter(long_entries, df.loc[long_entries, 'entry_price'], marker='^', color='green', s=100, label='Long Entry')
    if len(short_entries) > 0:
        ax1.scatter(short_entries, df.loc[short_entries, 'entry_price'], marker='v', color='red', s=100, label='Short Entry')
    if len(tp_exits) > 0:
        ax1.scatter(tp_exits, df.loc[tp_exits, 'exit_price'], marker='o', color='blue', s=80, label='Take Profit Exit')
    if len(sl_exits) > 0:
        ax1.scatter(sl_exits, df.loc[sl_exits, 'exit_price'], marker='o', color='purple', s=80, label='Stop Loss Exit')
    
    # Plot stop loss and take profit levels for active positions
    long_positions = df[df['position'] == 1]
    short_positions = df[df['position'] == -1]
    
    if len(long_positions) > 0:
        ax1.plot(long_positions.index, long_positions['sl_long'], '--', color='purple', linewidth=1, label='Long Stop Loss')
        ax1.plot(long_positions.index, long_positions['tp_long'], '--', color='blue', linewidth=1, label='Long Take Profit')
    
    if len(short_positions) > 0:
        ax1.plot(short_positions.index, short_positions['sl_short'], '--', color='purple', linewidth=1, label='Short Stop Loss')
        ax1.plot(short_positions.index, short_positions['tp_short'], '--', color='blue', linewidth=1, label='Short Take Profit')
    
    ax1.set_title('Multi-Structure Price Resonance Strategy - Price Chart')
    ax1.set_ylabel('Price')
    ax1.grid(True)
    ax1.legend(loc='upper left')
    
    # Format x-axis to show dates properly
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    ax1.xaxis.set_major_locator(mdates.HourLocator(interval=4))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Plot signal components
    ax2 = plt.subplot(4, 1, 2)
    ax2.plot(df.index, df['bullish_ob'], label='Bullish Order Block', color='green', alpha=0.7)
    ax2.plot(df.index, df['bearish_ob'], label='Bearish Order Block', color='red', alpha=0.7)
    ax2.plot(df.index, df['bullish_fvg'], label='Bullish Fair Value Gap', color='blue', alpha=0.7)
    ax2.plot(df.index, df['bearish_fvg'], label='Bearish Fair Value Gap', color='purple', alpha=0.7)
    ax2.plot(df.index, df['in_ny_session'], label='NY Session', color='black', alpha=0.7)
    
    ax2.set_title('Signal Components')
    ax2.set_ylabel('Signal Active')
    ax2.set_ylim(-0.1, 1.1)
    ax2.grid(True)
    ax2.legend(loc='upper left')
    
    # Plot positions
    ax3 = plt.subplot(4, 1, 3)
    ax3.plot(df.index, df['position'], label='Position (1=Long, -1=Short, 0=Flat)', color='blue')
    ax3.fill_between(df.index, df['position'], 0, where=(df['position'] > 0), color='green', alpha=0.3)
    ax3.fill_between(df.index, df['position'], 0, where=(df['position'] < 0), color='red', alpha=0.3)
    
    ax3.set_title('Position')
    ax3.set_ylabel('Position')
    ax3.set_ylim(-1.1, 1.1)
    ax3.grid(True)
    
    # Plot cumulative P&L
    ax4 = plt.subplot(4, 1, 4)
    ax4.plot(df.index, df['cumulative_pnl'], label='Cumulative P&L (%)', color='green')
    
    ax4.set_title('Cumulative P&L')
    ax4.set_ylabel('P&L (%)')
    ax4.grid(True)
    
    plt.tight_layout()
    plt.show()

# Display trade details
def display_trade_details(df):
    """
    Display details of all completed trades
    
    Parameters:
    - df: DataFrame with strategy results
    """
    trades = df[df['pnl'] != 0].copy()
    if len(trades) == 0:
        print("No completed trades found.")
        return
    
    print("\nTrade Details:")
    print("-" * 120)
    print(f"{'No.':<4}{'Entry Time':<20}{'Exit Time':<20}{'Position':<10}{'Entry Price':<12}{'Exit Price':<12}{'Stop Loss':<12}{'Take Profit':<12}{'P&L %':<10}{'Exit Type':<12}")
    print("-" * 120)
    
    # To track which entry corresponds to which exit
    entry_indices = []
    entry_positions = []
    
    # Find all entries
    for i in range(len(df) - 1):
        if df['position'].iloc[i] == 0 and df['position'].iloc[i+1] != 0:
            entry_indices.append(df.index[i+1])
            entry_positions.append(df['position'].iloc[i+1])
    
    # Match exits with entries
    for i, (idx, trade) in enumerate(trades.iterrows()):
        # Find the most recent entry before this exit
        entry_idx = None
        position_type = None
        
        for j in range(len(entry_indices)):
            if entry_indices[j] < idx:
                entry_idx = entry_indices[j]
                position_type = entry_positions[j]
                
                # Remove this entry to avoid matching it multiple times
                entry_indices.pop(j)
                entry_positions.pop(j)
                break
        
        if entry_idx is not None:
            # Format timestamps for display
            entry_time_str = entry_idx.strftime('%Y-%m-%d %H:%M') if hasattr(entry_idx, 'strftime') else str(entry_idx)
            exit_time_str = idx.strftime('%Y-%m-%d %H:%M') if hasattr(idx, 'strftime') else str(idx)
            
            position_str = "Long" if position_type == 1 else "Short"
            
            # Get entry price, stop loss, and take profit values
            entry_price = df.loc[entry_idx, 'entry_price']
            
            if position_type == 1:  # Long
                stop_loss = df.loc[entry_idx, 'sl_long']
                take_profit = df.loc[entry_idx, 'tp_long']
            else:  # Short
                stop_loss = df.loc[entry_idx, 'sl_short']
                take_profit = df.loc[entry_idx, 'tp_short']
            
            print(f"{i+1:<4}{entry_time_str:<20}{exit_time_str:<20}{position_str:<10}"
                 f"{entry_price:<12.2f}{trade['exit_price']:<12.2f}{stop_loss:<12.2f}{take_profit:<12.2f}"
                 f"{trade['pnl']:<10.2f}{trade['exit_type']:<12}")
    
    print("-" * 120)

# Generate simulated data for 10 days with 3-minute intervals
df = generate_simulated_data(days=10, volatility=0.0015, trend_strength=0.0002, 
                            trend_changes=5, intraday_pattern=True)

# Run strategy
result_df = multi_structure_resonance_strategy(df)

# Analyze performance
performance = analyze_performance(result_df)
print("\nStrategy Performance:")
print(f"Total Trades: {performance['total_trades']}")
print(f"Win Rate: {performance['win_rate']:.2%}")
print(f"Average Return per Trade: {performance['avg_return']:.2f}%")
print(f"Total Return: {performance['total_return']:.2f}%")
print(f"Maximum Drawdown: {performance['max_drawdown']:.2f}%")
print(f"Sharpe Ratio: {performance['sharpe_ratio']:.2f}")
print(f"Profit Factor: {performance['profit_factor']:.2f}")
print(f"Long Trades: {performance['long_trades']}, Short Trades: {performance['short_trades']}")
print(f"Exit Types: Take Profit: {performance['tp_exits']}, Stop Loss: {performance['sl_exits']}, Session End: {performance['session_end_exits']}")

# Display trade details
display_trade_details(result_df)

# Plot strategy results
plot_strategy_results(result_df)