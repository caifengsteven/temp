import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import matplotlib.dates as mdates

# Function to generate simulated price data
def generate_simulated_data(days=300, volatility=0.015, trend_strength=0.0003, 
                           trend_changes=6, include_mean_reversion=True, seed=42):
    """
    Generate simulated price data with trends and mean reversion
    
    Parameters:
    - days: Number of days to simulate
    - volatility: Base volatility level
    - trend_strength: Strength of trend component
    - trend_changes: Number of trend changes in the period
    - include_mean_reversion: Whether to include mean reversion
    - seed: Random seed for reproducibility
    
    Returns:
    - DataFrame with datetime index and OHLCV data (1-minute intervals)
    """
    np.random.seed(seed)
    
    # Create date range (1-minute intervals)
    end_date = datetime.now().replace(microsecond=0)
    start_date = end_date - timedelta(days=days)
    
    # For simulation purposes, let's use a smaller subset of minutes per day
    # to keep the data size manageable (e.g., trading hours only)
    dates = []
    current_date = start_date
    while current_date < end_date:
        # Only include minutes from 9:30 AM to 4:00 PM (trading hours)
        if 9 <= current_date.hour < 16:
            for minute in range(0, 60, 1):  # 1-minute intervals
                dates.append(current_date.replace(minute=minute))
        current_date += timedelta(hours=1)
    
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
    
    # Generate prices with trends, mean reversion, and random walks
    current_trend = 0
    
    # Calculate long-term moving average (for mean reversion)
    ma_period = 374  # Same as the strategy
    ma = np.zeros(n)
    
    for i in range(1, n):
        # Check if we need to change trend
        if i in trend_periods[1:]:
            current_trend += 1
        
        # Add trend component
        trend = trends[min(current_trend, trend_changes-1)]
        
        # Calculate current moving average
        if i < ma_period:
            ma[i] = np.mean(close[:i+1])
        else:
            ma[i] = np.mean(close[i-ma_period+1:i+1])
        
        # Add mean reversion component
        mean_reversion = 0
        if include_mean_reversion and i > ma_period:
            # Calculate log difference
            log_diff = np.log(close[i-1]) - np.log(ma[i-1])
            # Add mean reversion effect (stronger when far from MA)
            mean_reversion = -log_diff * 0.05
        
        # Add random component
        random_change = np.random.normal(0, volatility)
        
        # Calculate close price
        close[i] = close[i-1] * (1 + trend + random_change + mean_reversion)
        
        # Generate open with gap possibility
        gap = np.random.normal(0, volatility/2)
        open_price[i] = close[i-1] * (1 + gap)
        
        # Calculate high and low with range based on volatility
        price_range = close[i] * volatility * 2
        high[i] = max(close[i], open_price[i]) + abs(np.random.normal(0, price_range/2))
        low[i] = min(close[i], open_price[i]) - abs(np.random.normal(0, price_range/2))
        
        # Generate volume with correlation to price movement
        base_volume = 1000000
        price_change_pct = abs((close[i] - close[i-1]) / close[i-1])
        volume_factor = 1 + np.random.normal(price_change_pct * 10, 0.5)
        
        # Higher volume on trend change days
        if i in trend_periods[1:]:
            volume_factor *= 2
        
        # Occasionally add volume spikes
        if np.random.random() < 0.01:  # 1% chance of volume spike
            volume_factor *= np.random.uniform(1.5, 3.0)
        
        volume[i] = base_volume * volume_factor
    
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

# Calculate SMA
def calculate_sma(series, period):
    """Calculate Simple Moving Average"""
    return series.rolling(window=period).mean()

# Implement the Normalized Risk Value Dynamic Threshold Trading Strategy
def normalized_risk_strategy(df, sma_period=374, 
                           buy_threshold=0.3, exit_long_threshold1=0.6, exit_long_threshold2=0.7,
                           sell_threshold=0.7, exit_short_threshold=0.4,
                           stop_loss_points=5):
    """
    Implement the Normalized Risk Value Dynamic Threshold Trading Strategy
    
    Parameters:
    - df: DataFrame with OHLC data
    - sma_period: Period for SMA calculation
    - buy_threshold: Risk value threshold for buy signals
    - exit_long_threshold1, exit_long_threshold2: Risk value thresholds for exiting long positions
    - sell_threshold: Risk value threshold for sell signals
    - exit_short_threshold: Risk value threshold for exiting short positions
    - stop_loss_points: Fixed point value for stop-loss
    
    Returns:
    - DataFrame with strategy results
    """
    # Make a copy of the dataframe
    df = df.copy()
    
    # Calculate 374-period SMA
    df['sma'] = calculate_sma(df['close'], sma_period)
    
    # Calculate bar_index (relative position in the dataset)
    df['bar_index'] = np.arange(1, len(df) + 1)
    
    # Calculate logarithmic difference
    df['log_diff'] = np.log(df['close']) - np.log(df['sma'])
    
    # Apply time factor
    df['time_factor'] = np.power(df['bar_index'], 0.395)
    df['risk_value'] = df['log_diff'] * df['time_factor']
    
    # Calculate highest and lowest risk values (running)
    df['risk_ath'] = df['risk_value'].cummax()
    df['risk_atl'] = df['risk_value'].cummin()
    
    # Normalize risk value
    df['risk_normalized'] = (df['risk_value'] - df['risk_atl']) / (df['risk_ath'] - df['risk_atl'])
    
    # Generate trading signals
    df['buy_signal'] = df['risk_normalized'] < buy_threshold
    df['exit_long_signal'] = (df['risk_normalized'] >= exit_long_threshold1) | (df['risk_normalized'] >= exit_long_threshold2)
    df['sell_signal'] = df['risk_normalized'] > sell_threshold
    df['exit_short_signal'] = df['risk_normalized'] <= exit_short_threshold
    
    # Initialize strategy columns
    df['position'] = 0  # 1 for long, -1 for short, 0 for flat
    df['entry_price'] = np.nan
    df['stop_loss'] = np.nan
    df['exit_price'] = np.nan
    df['exit_type'] = ''  # 'sl', 'exit', or 'end'
    df['pnl'] = 0.0
    
    # Calculate point value (1 point = 0.01% of price)
    point_value = df['close'].iloc[0] * 0.0001  # Adjust as needed
    
    # Apply strategy logic
    position = 0
    entry_price = 0
    stop_loss = 0
    
    for i in range(1, len(df)):
        # Check if we need to close existing position
        if position == 1:  # Long position
            # Check if stop loss hit
            if df['low'].iloc[i] <= stop_loss:
                df.loc[df.index[i], 'exit_price'] = stop_loss
                df.loc[df.index[i], 'exit_type'] = 'sl'
                df.loc[df.index[i], 'pnl'] = (stop_loss / entry_price - 1) * 100
                position = 0  # Reset position
            
            # Check if exit signal
            elif df['exit_long_signal'].iloc[i]:
                df.loc[df.index[i], 'exit_price'] = df['close'].iloc[i]
                df.loc[df.index[i], 'exit_type'] = 'exit'
                df.loc[df.index[i], 'pnl'] = (df['close'].iloc[i] / entry_price - 1) * 100
                position = 0  # Reset position
        
        elif position == -1:  # Short position
            # Check if stop loss hit
            if df['high'].iloc[i] >= stop_loss:
                df.loc[df.index[i], 'exit_price'] = stop_loss
                df.loc[df.index[i], 'exit_type'] = 'sl'
                df.loc[df.index[i], 'pnl'] = (entry_price / stop_loss - 1) * 100
                position = 0  # Reset position
            
            # Check if exit signal
            elif df['exit_short_signal'].iloc[i]:
                df.loc[df.index[i], 'exit_price'] = df['close'].iloc[i]
                df.loc[df.index[i], 'exit_type'] = 'exit'
                df.loc[df.index[i], 'pnl'] = (entry_price / df['close'].iloc[i] - 1) * 100
                position = 0  # Reset position
        
        # Check for new entry if not in a position
        if position == 0:
            # Buy signal
            if df['buy_signal'].iloc[i]:
                position = 1
                entry_price = df['close'].iloc[i]
                stop_loss = entry_price - (stop_loss_points * point_value)
                
                df.loc[df.index[i], 'position'] = position
                df.loc[df.index[i], 'entry_price'] = entry_price
                df.loc[df.index[i], 'stop_loss'] = stop_loss
            
            # Sell signal
            elif df['sell_signal'].iloc[i]:
                position = -1
                entry_price = df['close'].iloc[i]
                stop_loss = entry_price + (stop_loss_points * point_value)
                
                df.loc[df.index[i], 'position'] = position
                df.loc[df.index[i], 'entry_price'] = entry_price
                df.loc[df.index[i], 'stop_loss'] = stop_loss
        
        # Update position
        df.loc[df.index[i], 'position'] = position
    
    # Close any open position at the end of the period
    if position != 0:
        last_idx = df.index[-1]
        last_price = df['close'].iloc[-1]
        
        df.loc[last_idx, 'exit_price'] = last_price
        df.loc[last_idx, 'exit_type'] = 'end'
        
        if position == 1:  # Long position
            df.loc[last_idx, 'pnl'] = (last_price / entry_price - 1) * 100
        else:  # Short position
            df.loc[last_idx, 'pnl'] = (entry_price / last_price - 1) * 100
    
    # Calculate cumulative P&L and equity curve
    df['cumulative_pnl'] = df['pnl'].cumsum()
    initial_equity = 10000  # Starting equity
    df['equity_curve'] = initial_equity * (1 + df['pnl']/100).cumprod()
    
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
            'sl_exits': 0,
            'signal_exits': 0,
            'end_exits': 0
        }
    
    # Calculate performance metrics
    total_trades = len(trades)
    winning_trades = len(trades[trades['pnl'] > 0])
    win_rate = winning_trades / total_trades if total_trades > 0 else 0
    avg_return = trades['pnl'].mean()
    total_return = df['equity_curve'].iloc[-1] / df['equity_curve'].iloc[0] - 1
    
    # Count trade types
    long_trades = len(trades[trades['position'] == 1])
    short_trades = len(trades[trades['position'] == -1])
    
    # Count exit types
    sl_exits = len(trades[trades['exit_type'] == 'sl'])
    signal_exits = len(trades[trades['exit_type'] == 'exit'])
    end_exits = len(trades[trades['exit_type'] == 'end'])
    
    # Calculate profit factor
    gross_profit = trades[trades['pnl'] > 0]['pnl'].sum()
    gross_loss = abs(trades[trades['pnl'] < 0]['pnl'].sum())
    profit_factor = gross_profit / gross_loss if gross_loss != 0 else float('inf')
    
    # Calculate drawdown
    df['peak'] = df['equity_curve'].cummax()
    df['drawdown'] = (df['equity_curve'] / df['peak'] - 1) * 100
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
        'sl_exits': sl_exits,
        'signal_exits': signal_exits,
        'end_exits': end_exits
    }

# Plot strategy results
def plot_strategy_results(df):
    """
    Visualize strategy results
    
    Parameters:
    - df: DataFrame with strategy results
    """
    # Downsample data for plotting if necessary
    if len(df) > 5000:
        # Resample to daily data for better visualization
        plot_df = df.resample('D').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum',
            'sma': 'last',
            'risk_normalized': 'last',
            'position': 'last',
            'equity_curve': 'last',
            'drawdown': 'min'
        })
        plot_df = plot_df.dropna()
    else:
        plot_df = df
    
    fig = plt.figure(figsize=(16, 20))
    
    # Price chart
    ax1 = plt.subplot(4, 1, 1)
    ax1.plot(plot_df.index, plot_df['close'], label='Close Price', color='black', linewidth=1)
    ax1.plot(plot_df.index, plot_df['sma'], label=f'SMA ({374})', color='blue', linewidth=1)
    
    # Plot entry and exit points
    long_entries = df[(df['position'] == 1) & (df['position'].shift(1) != 1)].index
    short_entries = df[(df['position'] == -1) & (df['position'].shift(1) != -1)].index
    sl_exits = df[df['exit_type'] == 'sl'].index
    signal_exits = df[df['exit_type'] == 'exit'].index
    
    if len(long_entries) > 0:
        ax1.scatter(long_entries, df.loc[long_entries, 'entry_price'], marker='^', color='green', s=100, label='Long Entry')
    if len(short_entries) > 0:
        ax1.scatter(short_entries, df.loc[short_entries, 'entry_price'], marker='v', color='red', s=100, label='Short Entry')
    if len(sl_exits) > 0:
        ax1.scatter(sl_exits, df.loc[sl_exits, 'exit_price'], marker='o', color='purple', s=80, label='Stop Loss Exit')
    if len(signal_exits) > 0:
        ax1.scatter(signal_exits, df.loc[signal_exits, 'exit_price'], marker='o', color='blue', s=80, label='Signal Exit')
    
    # Plot stop loss levels
    for i in range(1, len(df)):
        if df['position'].iloc[i] != 0 and df['position'].iloc[i-1] == 0:  # New position
            # Get entry date and stop loss
            entry_date = df.index[i]
            stop_loss = df['stop_loss'].iloc[i]
            
            # Find exit date
            exit_date = None
            for j in range(i+1, len(df)):
                if df['position'].iloc[j] == 0:
                    exit_date = df.index[j]
                    break
            
            # If no exit found, use next date
            if exit_date is None and i < len(df) - 1:
                exit_date = df.index[i+1]
            
            # If we found a next point, draw the stop loss
            if exit_date is not None:
                if df['position'].iloc[i] == 1:  # Long position
                    ax1.plot([entry_date, exit_date], [stop_loss, stop_loss], 'r--', linewidth=1, alpha=0.7)
                else:  # Short position
                    ax1.plot([entry_date, exit_date], [stop_loss, stop_loss], 'r--', linewidth=1, alpha=0.7)
    
    ax1.set_title('Normalized Risk Value Dynamic Threshold Strategy - Price Chart')
    ax1.set_ylabel('Price')
    ax1.grid(True)
    ax1.legend(loc='upper left')
    
    # Format x-axis to show dates properly
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax1.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Normalized Risk Value
    ax2 = plt.subplot(4, 1, 2)
    ax2.plot(plot_df.index, plot_df['risk_normalized'], label='Normalized Risk Value', color='blue', linewidth=1)
    
    # Add threshold lines
    ax2.axhline(y=0.3, color='green', linestyle='--', linewidth=1, alpha=0.7, label='Buy Threshold (0.3)')
    ax2.axhline(y=0.4, color='orange', linestyle='--', linewidth=1, alpha=0.7, label='Exit Short Threshold (0.4)')
    ax2.axhline(y=0.6, color='orange', linestyle='--', linewidth=1, alpha=0.7, label='Exit Long Threshold 1 (0.6)')
    ax2.axhline(y=0.7, color='red', linestyle='--', linewidth=1, alpha=0.7, label='Sell/Exit Long Threshold 2 (0.7)')
    
    # Highlight zones
    ax2.fill_between(plot_df.index, 0, 0.3, alpha=0.2, color='green', label='Buy Zone')
    ax2.fill_between(plot_df.index, 0.7, 1, alpha=0.2, color='red', label='Sell Zone')
    
    ax2.set_title('Normalized Risk Value')
    ax2.set_ylabel('Risk Value')
    ax2.set_ylim(-0.1, 1.1)
    ax2.grid(True)
    ax2.legend(loc='upper left')
    
    # Position
    ax3 = plt.subplot(4, 1, 3)
    ax3.plot(plot_df.index, plot_df['position'], label='Position', color='blue', linewidth=1)
    ax3.fill_between(plot_df.index, plot_df['position'], 0, where=(plot_df['position'] > 0), color='green', alpha=0.3, label='Long')
    ax3.fill_between(plot_df.index, plot_df['position'], 0, where=(plot_df['position'] < 0), color='red', alpha=0.3, label='Short')
    
    ax3.set_title('Position (1=Long, -1=Short, 0=Flat)')
    ax3.set_ylabel('Position')
    ax3.set_ylim(-1.1, 1.1)
    ax3.grid(True)
    ax3.legend(loc='upper left')
    
    # Equity curve
    ax4 = plt.subplot(4, 1, 4)
    ax4.plot(plot_df.index, plot_df['equity_curve'], label='Equity Curve', color='green')
    
    # Add drawdown to secondary y-axis
    twin_ax = ax4.twinx()
    twin_ax.fill_between(plot_df.index, plot_df['drawdown'], 0, alpha=0.3, color='red', label='Drawdown')
    twin_ax.set_ylabel('Drawdown (%)')
    twin_ax.set_ylim(-30, 5)  # Adjust based on actual drawdowns
    
    ax4.set_title('Equity Curve & Drawdown')
    ax4.set_ylabel('Equity ($)')
    ax4.grid(True)
    ax4.legend(loc='upper left')
    twin_ax.legend(loc='lower left')
    
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
    print(f"{'No.':<4}{'Entry Date':<20}{'Exit Date':<20}{'Position':<10}{'Entry Price':<12}{'Exit Price':<12}{'Stop Loss':<12}{'P&L %':<10}{'Exit Type':<12}")
    print("-" * 120)
    
    # Find entry dates for each trade
    entry_dates = []
    for i, (idx, trade) in enumerate(trades.iterrows()):
        # Find the most recent entry before this exit
        entry_mask = ((df.index < idx) & 
                      (df['position'] != 0) & 
                      (df['position'].shift(1) == 0))
        
        if entry_mask.any():
            entry_date = df[entry_mask].index[-1]
            position_type = "Long" if df.loc[entry_date, 'position'] == 1 else "Short"
            
            # Format dates
            entry_date_str = entry_date.strftime('%Y-%m-%d %H:%M')
            exit_date_str = idx.strftime('%Y-%m-%d %H:%M')
            
            # Get trade details
            entry_price = df.loc[entry_date, 'entry_price']
            exit_price = trade['exit_price']
            stop_loss = df.loc[entry_date, 'stop_loss']
            pnl = trade['pnl']
            exit_type = trade['exit_type']
            
            print(f"{i+1:<4}{entry_date_str:<20}{exit_date_str:<20}{position_type:<10}"
                 f"{entry_price:<12.4f}{exit_price:<12.4f}{stop_loss:<12.4f}{pnl:<10.2f}{exit_type:<12}")
    
    print("-" * 120)

# Set strategy parameters
sma_period = 374
buy_threshold = 0.3
exit_long_threshold1 = 0.6
exit_long_threshold2 = 0.7
sell_threshold = 0.7
exit_short_threshold = 0.4
stop_loss_points = 5

# Generate simulated data
# For a 1-minute strategy, we need a good amount of data but don't want it to be too large
# Let's simulate about 30 days worth of 1-minute data
df = generate_simulated_data(days=30, volatility=0.01, trend_strength=0.0001, 
                            trend_changes=4, include_mean_reversion=True)

# Run strategy
result_df = normalized_risk_strategy(df, sma_period=sma_period, 
                                   buy_threshold=buy_threshold, 
                                   exit_long_threshold1=exit_long_threshold1, 
                                   exit_long_threshold2=exit_long_threshold2,
                                   sell_threshold=sell_threshold, 
                                   exit_short_threshold=exit_short_threshold,
                                   stop_loss_points=stop_loss_points)

# Analyze performance
performance = analyze_performance(result_df)

# Display results
print("\nNormalized Risk Value Dynamic Threshold Strategy Performance:")
print(f"Total Trades: {performance['total_trades']}")
print(f"Win Rate: {performance['win_rate']:.2%}")
print(f"Average Return per Trade: {performance['avg_return']:.2f}%")
print(f"Total Return: {performance['total_return']:.2%}")
print(f"Maximum Drawdown: {performance['max_drawdown']:.2f}%")
print(f"Sharpe Ratio: {performance['sharpe_ratio']:.2f}")
print(f"Profit Factor: {performance['profit_factor']:.2f}")
print(f"Long Trades: {performance['long_trades']}, Short Trades: {performance['short_trades']}")
print(f"Exit Types: Signal Exits: {performance['signal_exits']}, Stop Loss: {performance['sl_exits']}, End: {performance['end_exits']}")

# Display trade details
display_trade_details(result_df)

# Plot results
plot_strategy_results(result_df)