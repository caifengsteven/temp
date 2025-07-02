import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import matplotlib.dates as mdates

# Function to generate simulated price data
def generate_simulated_data(days=180, volatility=0.02, trend_strength=0.0004, 
                           trend_changes=6, seed=42, include_downtrends=True):
    np.random.seed(seed)
    
    # Create date range
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=days)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
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
    high[0] = 101
    low[0] = 99
    volume[0] = 1000000
    
    # Generate trend changes - ensure we include downtrends for a short-only strategy
    trend_periods = np.linspace(0, n, trend_changes + 1).astype(int)
    
    if include_downtrends:
        # Ensure at least half of the trends are downtrends (negative)
        trends = np.zeros(trend_changes)
        for i in range(trend_changes):
            if i % 2 == 0:  # Every other trend is downward
                trends[i] = -1 * np.random.uniform(0.5, 1.5) * trend_strength
            else:
                trends[i] = np.random.uniform(0.3, 1.0) * trend_strength
    else:
        # Random trends
        trends = np.random.choice([-1, 1], size=trend_changes) * trend_strength
    
    # Generate prices with trends and random walks
    current_trend = 0
    for i in range(1, n):
        # Check if we need to change trend
        if i in trend_periods[1:]:
            current_trend += 1
        
        # Add trend and random component
        trend = trends[min(current_trend, trend_changes-1)]
        random_change = np.random.normal(0, volatility)
        close[i] = close[i-1] * (1 + trend + random_change)
        
        # Generate open with gap possibility
        gap = np.random.normal(0, volatility/2)
        open_price[i] = close[i-1] * (1 + gap)
        
        # Generate high, low with reasonable ranges
        daily_range = close[i] * volatility * 2
        intraday_volatility = np.random.uniform(0.5, 1.5) * daily_range
        high[i] = max(close[i], open_price[i]) + abs(np.random.normal(0, intraday_volatility/2))
        low[i] = min(close[i], open_price[i]) - abs(np.random.normal(0, intraday_volatility/2))
        
        # Generate volume with correlation to price movement
        base_volume = 1000000
        price_change_pct = abs((close[i] - close[i-1]) / close[i-1])
        volume_factor = 1 + np.random.normal(price_change_pct * 10, 0.5)
        
        # Occasionally add volume spikes, especially during breakouts
        if np.random.random() < 0.2:  # 20% chance of volume spike
            volume_factor *= np.random.uniform(1.5, 3.0)
        
        volume[i] = base_volume * volume_factor
    
    # Create DataFrame
    df = pd.DataFrame({
        'date': dates,
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
    
    # Set date as index for easier plotting
    df.set_index('date', inplace=True)
    
    return df

# Calculate Average True Range (ATR)
def calculate_atr(high, low, close, period=14):
    tr1 = pd.DataFrame(high - low)
    tr2 = pd.DataFrame(abs(high - close.shift()))
    tr3 = pd.DataFrame(abs(low - close.shift()))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    return atr

# Calculate Simple Moving Average
def calculate_sma(series, period):
    return series.rolling(window=period).mean()

# Implement the ATR-Based Dynamic Stop-Loss Support Breakout Short Strategy
def atr_support_breakout_strategy(df, sr_range=20, vol_ma_length=20, atr_length=14,
                                 trail_multiplier=1.5, stop_multiplier=1.0,
                                 breakout_buffer=1.005, range_length=20, range_threshold=1.5):
    # Make a copy of the dataframe to avoid modifying the original
    df = df.copy()
    
    # Calculate indicators
    df['atr'] = calculate_atr(df['high'], df['low'], df['close'], atr_length)
    df['vol_ma'] = calculate_sma(df['volume'], vol_ma_length)
    
    # Calculate support level (lowest low of previous sr_range periods)
    df['support'] = df['low'].rolling(window=sr_range).min().shift(1)
    
    # Calculate sideways market detection
    df['high_range'] = df['high'].rolling(window=range_length).max()
    df['low_range'] = df['low'].rolling(window=range_length).min()
    df['price_range'] = df['high_range'] - df['low_range']
    df['is_sideways'] = df['price_range'] <= df['atr'] * range_threshold
    
    # Calculate trailing stop and initial stop values in price
    df['trail_stop'] = df['close'] + df['atr'] * trail_multiplier
    df['stop_loss'] = df['close'] + df['atr'] * stop_multiplier
    
    # Initialize strategy columns
    df['signal'] = 0  # -1 for short, 0 for no signal
    df['position'] = 0  # -1 for short, 0 for flat
    df['entry_price'] = np.nan
    df['exit_price'] = np.nan
    df['exit_reason'] = ''  # 'stop' or 'trail'
    df['pnl'] = 0.0
    
    # Apply strategy logic
    current_position = 0
    entry_price = 0
    stop_loss = 0
    trail_stop = 0
    
    for i in range(max(sr_range, vol_ma_length, atr_length, range_length) + 1, len(df)):
        # Copy previous position
        df.loc[df.index[i], 'position'] = current_position
        
        # Check for short entry conditions if not in a position
        if current_position == 0:
            # Support breakout condition
            breakout_condition = df['close'].iloc[i] <= df['support'].iloc[i] * breakout_buffer
            # Volume confirmation
            volume_condition = df['volume'].iloc[i] >= df['vol_ma'].iloc[i]
            # Not in sideways market
            not_sideways = not df['is_sideways'].iloc[i]
            
            # Short signal
            if breakout_condition and volume_condition and not_sideways:
                current_position = -1
                entry_price = df['close'].iloc[i]
                stop_loss = df['stop_loss'].iloc[i]
                trail_stop = df['trail_stop'].iloc[i]
                
                df.loc[df.index[i], 'position'] = current_position
                df.loc[df.index[i], 'entry_price'] = entry_price
                df.loc[df.index[i], 'signal'] = -1
        
        # Check for exit conditions if in a short position
        elif current_position == -1:
            # Update trailing stop if price moves in our favor
            if df['close'].iloc[i] < entry_price:
                new_trail_stop = df['close'].iloc[i] + df['atr'].iloc[i] * trail_multiplier
                if new_trail_stop < trail_stop:
                    trail_stop = new_trail_stop
            
            # Update stop_loss and trail_stop for display
            df.loc[df.index[i], 'stop_loss'] = stop_loss
            df.loc[df.index[i], 'trail_stop'] = trail_stop
            
            # Check if initial stop loss is hit
            if df['high'].iloc[i] >= stop_loss:
                df.loc[df.index[i], 'exit_price'] = stop_loss
                df.loc[df.index[i], 'pnl'] = (entry_price / stop_loss - 1) * 100
                df.loc[df.index[i], 'exit_reason'] = 'stop'
                
                # Reset position
                current_position = 0
                df.loc[df.index[i], 'position'] = 0
            
            # Check if trailing stop is hit
            elif df['high'].iloc[i] >= trail_stop:
                df.loc[df.index[i], 'exit_price'] = trail_stop
                df.loc[df.index[i], 'pnl'] = (entry_price / trail_stop - 1) * 100
                df.loc[df.index[i], 'exit_reason'] = 'trail'
                
                # Reset position
                current_position = 0
                df.loc[df.index[i], 'position'] = 0
    
    # Close any open position at the end of the period
    if current_position == -1:
        last_idx = df.index[-1]
        last_price = df['close'].iloc[-1]
        
        df.loc[last_idx, 'exit_price'] = last_price
        df.loc[last_idx, 'pnl'] = (entry_price / last_price - 1) * 100
        df.loc[last_idx, 'exit_reason'] = 'end'
    
    # Calculate cumulative P&L and equity curve
    df['cumulative_pnl'] = df['pnl'].cumsum()
    df['equity_curve'] = 100 * (1 + df['pnl']/100).cumprod()
    
    return df

# Analyze strategy performance
def analyze_performance(df):
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
            'profit_factor': 0
        }
    
    # Calculate performance metrics
    total_trades = len(trades)
    winning_trades = len(trades[trades['pnl'] > 0])
    win_rate = winning_trades / total_trades if total_trades > 0 else 0
    avg_return = trades['pnl'].mean()
    total_return = trades['pnl'].sum()
    
    # Count exits by reason
    stop_losses = len(trades[trades['exit_reason'] == 'stop'])
    trail_stops = len(trades[trades['exit_reason'] == 'trail'])
    end_of_period = len(trades[trades['exit_reason'] == 'end'])
    
    # Calculate profit factor
    gross_profit = trades[trades['pnl'] > 0]['pnl'].sum()
    gross_loss = abs(trades[trades['pnl'] < 0]['pnl'].sum())
    profit_factor = gross_profit / gross_loss if gross_loss != 0 else float('inf')
    
    # Calculate drawdown
    if 'equity_curve' not in df.columns:
        df['equity_curve'] = 100 * (1 + df['pnl']/100).cumprod()
    
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
        'stop_losses': stop_losses,
        'trail_stops': trail_stops,
        'end_of_period': end_of_period
    }

# Plot strategy results
def plot_strategy_results(df):
    fig = plt.figure(figsize=(16, 14))
    
    # Create a price chart with indicators
    ax1 = plt.subplot(4, 1, 1)
    ax1.plot(df.index, df['close'], label='Close Price', color='black', linewidth=1)
    ax1.plot(df.index, df['support'], label='Support Level', color='green', linewidth=1, alpha=0.7)
    
    # Plot entry signals
    short_entries = df[df['signal'] == -1].index
    if len(short_entries) > 0:
        ax1.scatter(short_entries, df.loc[short_entries, 'close'], marker='v', color='red', s=100, label='Short Entry')
    
    # Plot exit points
    stop_exits = df[df['exit_reason'] == 'stop'].index
    trail_exits = df[df['exit_reason'] == 'trail'].index
    end_exits = df[df['exit_reason'] == 'end'].index
    
    if len(stop_exits) > 0:
        ax1.scatter(stop_exits, df.loc[stop_exits, 'exit_price'], marker='o', color='red', s=80, label='Stop Loss')
    if len(trail_exits) > 0:
        ax1.scatter(trail_exits, df.loc[trail_exits, 'exit_price'], marker='o', color='blue', s=80, label='Trailing Stop')
    if len(end_exits) > 0:
        ax1.scatter(end_exits, df.loc[end_exits, 'exit_price'], marker='o', color='gray', s=80, label='End of Period')
    
    # Plot stop loss and trailing stop
    position_indices = df[df['position'] == -1].index
    if len(position_indices) > 0:
        ax1.plot(position_indices, df.loc[position_indices, 'stop_loss'], '--', color='red', linewidth=1, label='Stop Loss')
        ax1.plot(position_indices, df.loc[position_indices, 'trail_stop'], '--', color='blue', linewidth=1, label='Trailing Stop')
    
    # Highlight sideways markets
    sideways_indices = df[df['is_sideways']].index
    for idx in sideways_indices:
        ax1.axvspan(idx, idx + pd.Timedelta(days=1), alpha=0.2, color='orange')
    
    ax1.set_title('ATR-Based Dynamic Stop-Loss Support Breakout Short Strategy')
    ax1.set_ylabel('Price')
    ax1.grid(True)
    ax1.legend(loc='upper left')
    
    # Format x-axis to show dates properly
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax1.xaxis.set_major_locator(mdates.DayLocator(interval=14))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
    
    # Plot volume and volume MA
    ax2 = plt.subplot(4, 1, 2)
    ax2.bar(df.index, df['volume'], color='blue', alpha=0.3, label='Volume')
    ax2.plot(df.index, df['vol_ma'], color='red', linewidth=1, label=f'Volume SMA ({vol_ma_length})')
    ax2.set_title('Volume Analysis')
    ax2.set_ylabel('Volume')
    ax2.grid(True)
    ax2.legend()
    
    # Plot sideways detection
    ax3 = plt.subplot(4, 1, 3)
    ax3.plot(df.index, df['is_sideways'], label='Sideways Market', color='orange')
    ax3.plot(df.index, df['position'] * -1, label='Position (-1=Short, 0=Flat)', color='purple')
    ax3.set_title('Sideways Market Detection & Position')
    ax3.set_ylabel('Status')
    ax3.set_ylim(-1.1, 1.1)
    ax3.grid(True)
    ax3.legend()
    
    # Plot equity curve
    ax4 = plt.subplot(4, 1, 4)
    ax4.plot(df.index, df['equity_curve'], label='Equity Curve', color='green')
    ax4.set_title('Equity Curve (Starting at 100)')
    ax4.set_ylabel('Equity')
    ax4.grid(True)
    
    plt.tight_layout()
    plt.show()

# Display trade details
def display_trade_details(df):
    trades = df[df['pnl'] != 0].copy()
    if len(trades) == 0:
        print("No completed trades found.")
        return
    
    print("\nTrade Details:")
    print("-" * 110)
    print(f"{'No.':<4}{'Type':<8}{'Entry Date':<12}{'Exit Date':<12}{'Entry Price':<12}{'Exit Price':<12}{'PnL %':<10}{'Exit Reason':<12}")
    print("-" * 110)
    
    # Find entries for each exit
    trade_count = 1
    for i, (idx, trade) in enumerate(trades.iterrows()):
        # Find entry for this trade by looking at previous signals
        entry_idx = None
        
        # Look back for most recent signal change
        for j in range(i, 0, -1):
            if df['signal'].iloc[j] != 0:
                entry_idx = df.index[j]
                break
        
        if entry_idx is not None:
            entry_price = df.loc[entry_idx, 'entry_price']
            
            # Format dates for display
            entry_date_str = entry_idx.strftime('%Y-%m-%d') if hasattr(entry_idx, 'strftime') else str(entry_idx)
            exit_date_str = idx.strftime('%Y-%m-%d') if hasattr(idx, 'strftime') else str(idx)
            
            print(f"{trade_count:<4}{'SHORT':<8}{entry_date_str:<12}{exit_date_str:<12}"
                 f"{entry_price:<12.2f}{trade['exit_price']:<12.2f}{trade['pnl']:<10.2f}{trade['exit_reason']:<12}")
            
            trade_count += 1
    
    print("-" * 110)

# Set strategy parameters
sr_range = 20
vol_ma_length = 20
atr_length = 14
trail_multiplier = 1.5
stop_multiplier = 1.0
breakout_buffer = 1.005
range_length = 20
range_threshold = 1.5

# Generate simulated data with some strong downtrends
df = generate_simulated_data(days=180, volatility=0.02, trend_strength=0.0008, 
                            trend_changes=6, include_downtrends=True)

# Run strategy
result_df = atr_support_breakout_strategy(
    df, 
    sr_range=sr_range,
    vol_ma_length=vol_ma_length,
    atr_length=atr_length,
    trail_multiplier=trail_multiplier,
    stop_multiplier=stop_multiplier,
    breakout_buffer=breakout_buffer,
    range_length=range_length,
    range_threshold=range_threshold
)

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
print(f"Exit Types: Stop Loss: {performance['stop_losses']}, Trailing Stop: {performance['trail_stops']}, End of Period: {performance['end_of_period']}")

# Display trade details
display_trade_details(result_df)

# Plot results
plot_strategy_results(result_df)