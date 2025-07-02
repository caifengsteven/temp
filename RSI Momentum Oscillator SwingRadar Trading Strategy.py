import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import matplotlib.dates as mdates

# Function to generate simulated price data
def generate_simulated_data(days=120, volatility=0.02, trend_strength=0.0002, 
                           trend_changes=5, include_pullbacks=True, seed=42):
    """
    Generate simulated price data with trends and pullbacks
    
    Parameters:
    - days: Number of days to simulate
    - volatility: Base volatility level
    - trend_strength: Strength of trend component
    - trend_changes: Number of trend changes in the period
    - include_pullbacks: Whether to include pullback patterns
    - seed: Random seed for reproducibility
    
    Returns:
    - DataFrame with datetime index and OHLCV data
    """
    np.random.seed(seed)
    
    # Create date range (3-day intervals as per the strategy)
    end_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    start_date = end_date - timedelta(days=days)
    dates = pd.date_range(start=start_date, end=end_date, freq='3D')
    
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
    
    # Generate prices with trends and pullbacks
    current_trend = 0
    
    # Track if we're in a pullback
    in_pullback = False
    pullback_length = 0
    max_pullback_length = 7  # Maximum pullback duration
    
    for i in range(1, n):
        # Check if we need to change trend
        if i in trend_periods[1:]:
            current_trend += 1
            in_pullback = False  # Reset pullback state on trend change
        
        # Determine current trend direction
        trend = trends[min(current_trend, trend_changes-1)]
        
        # Add pullbacks occasionally (if enabled)
        if include_pullbacks and not in_pullback and np.random.random() < 0.15:  # 15% chance to start pullback
            in_pullback = True
            pullback_length = 0
            # Reverse the trend temporarily for pullback
            trend = -trend * np.random.uniform(1.5, 3.0)  # Stronger reverse movement
        
        # If in pullback, count its duration
        if in_pullback:
            pullback_length += 1
            # End pullback if it's gone on long enough
            if pullback_length >= max_pullback_length or np.random.random() < 0.3:  # 30% chance to end each day
                in_pullback = False
                trend = trends[min(current_trend, trend_changes-1)]  # Restore original trend
        
        # Add trend component
        trend_component = trend
        
        # Add random component
        random_change = np.random.normal(0, volatility)
        
        # Calculate close price
        close[i] = close[i-1] * (1 + trend_component + random_change)
        
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
        
        # Higher volume at pullback end
        if in_pullback and pullback_length >= max_pullback_length:
            volume_factor *= 1.5
        
        # Occasionally add volume spikes
        if np.random.random() < 0.1:  # 10% chance of volume spike
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
    
    # Set date as index
    df.set_index('date', inplace=True)
    
    return df

# Calculate RSI
def calculate_rsi(series, period=14):
    """Calculate Relative Strength Index"""
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    
    # Handle division by zero
    rs = avg_gain / avg_loss.replace(0, np.finfo(float).eps)
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

# Calculate Average True Range (ATR)
def calculate_atr(high, low, close, period=14):
    """Calculate Average True Range"""
    tr1 = pd.DataFrame(high - low)
    tr2 = pd.DataFrame(abs(high - close.shift()))
    tr3 = pd.DataFrame(abs(low - close.shift()))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    return atr

# Implement the RSI Momentum Oscillator SwingRadar Strategy
def rsi_momentum_strategy(df, rsi_period=14, rsi_ma_period=14, atr_period=14, 
                         rsi_oversold=35, atr_multiplier=0.5, risk_reward=4.0):
    """
    Implement the RSI Momentum Oscillator SwingRadar Strategy
    
    Parameters:
    - df: DataFrame with OHLC data
    - rsi_period: Period for RSI calculation
    - rsi_ma_period: Period for RSI moving average
    - atr_period: Period for ATR calculation
    - rsi_oversold: Threshold for oversold RSI
    - atr_multiplier: Multiplier for ATR to calculate stop loss
    - risk_reward: Risk-reward ratio for take profit
    
    Returns:
    - DataFrame with strategy results
    """
    # Make a copy of the dataframe
    df = df.copy()
    
    # Calculate indicators
    df['rsi'] = calculate_rsi(df['close'], rsi_period)
    df['rsi_ma'] = df['rsi'].rolling(window=rsi_ma_period).mean()
    df['atr'] = calculate_atr(df['high'], df['low'], df['close'], atr_period)
    
    # Generate signals
    df['rsi_cross_up'] = (df['rsi'] > df['rsi_ma']) & (df['rsi'].shift(1) <= df['rsi_ma'].shift(1))
    df['prev_rsi_oversold'] = df['rsi'].shift(1) < rsi_oversold
    
    df['buy_signal'] = df['rsi_cross_up'] & df['prev_rsi_oversold']
    
    # Initialize strategy columns
    df['position'] = 0  # 1 for long, 0 for flat
    df['entry_price'] = np.nan
    df['stop_loss'] = np.nan
    df['take_profit'] = np.nan
    df['exit_price'] = np.nan
    df['exit_type'] = ''  # 'tp', 'sl', or 'end'
    df['pnl'] = 0.0
    
    # Apply strategy logic
    position = 0
    entry_price = 0
    stop_loss = 0
    take_profit = 0
    
    for i in range(1, len(df)):
        # Check if we need to close existing position
        if position == 1:  # Long position
            # Check if stop loss hit
            if df['low'].iloc[i] <= stop_loss:
                df.loc[df.index[i], 'exit_price'] = stop_loss
                df.loc[df.index[i], 'exit_type'] = 'sl'
                df.loc[df.index[i], 'pnl'] = (stop_loss / entry_price - 1) * 100
                position = 0  # Reset position
            
            # Check if take profit hit
            elif df['high'].iloc[i] >= take_profit:
                df.loc[df.index[i], 'exit_price'] = take_profit
                df.loc[df.index[i], 'exit_type'] = 'tp'
                df.loc[df.index[i], 'pnl'] = (take_profit / entry_price - 1) * 100
                position = 0  # Reset position
        
        # Check for new entry if not in a position
        if position == 0:
            if df['buy_signal'].iloc[i]:
                position = 1
                entry_price = df['close'].iloc[i]
                
                # Calculate stop loss: current low - (ATR × multiplier)
                stop_loss = df['low'].iloc[i] - (df['atr'].iloc[i] * atr_multiplier)
                
                # Calculate stop distance
                stop_distance = entry_price - stop_loss
                
                # Calculate take profit: entry + (stop distance × risk_reward)
                take_profit = entry_price + (stop_distance * risk_reward)
                
                df.loc[df.index[i], 'position'] = position
                df.loc[df.index[i], 'entry_price'] = entry_price
                df.loc[df.index[i], 'stop_loss'] = stop_loss
                df.loc[df.index[i], 'take_profit'] = take_profit
        
        # Update position
        df.loc[df.index[i], 'position'] = position
    
    # Close any open position at the end of the period
    if position == 1:
        last_idx = df.index[-1]
        last_price = df['close'].iloc[-1]
        
        df.loc[last_idx, 'exit_price'] = last_price
        df.loc[last_idx, 'exit_type'] = 'end'
        df.loc[last_idx, 'pnl'] = (last_price / entry_price - 1) * 100
    
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
            'tp_exits': 0,
            'sl_exits': 0,
            'end_exits': 0
        }
    
    # Calculate performance metrics
    total_trades = len(trades)
    winning_trades = len(trades[trades['pnl'] > 0])
    win_rate = winning_trades / total_trades if total_trades > 0 else 0
    avg_return = trades['pnl'].mean()
    total_return = df['equity_curve'].iloc[-1] / df['equity_curve'].iloc[0] - 1
    
    # Count exit types
    tp_exits = len(trades[trades['exit_type'] == 'tp'])
    sl_exits = len(trades[trades['exit_type'] == 'sl'])
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
        'tp_exits': tp_exits,
        'sl_exits': sl_exits,
        'end_exits': end_exits
    }

# Plot strategy results
def plot_strategy_results(df):
    """
    Visualize strategy results
    
    Parameters:
    - df: DataFrame with strategy results
    """
    fig = plt.figure(figsize=(16, 20))
    
    # Price chart with signals
    ax1 = plt.subplot(4, 1, 1)
    ax1.plot(df.index, df['close'], label='Close Price', color='black', linewidth=1)
    
    # Plot entry and exit points
    entries = df[(df['position'] == 1) & (df['position'].shift(1) == 0)].index
    tp_exits = df[df['exit_type'] == 'tp'].index
    sl_exits = df[df['exit_type'] == 'sl'].index
    
    if len(entries) > 0:
        ax1.scatter(entries, df.loc[entries, 'entry_price'], marker='^', color='green', s=100, label='Entry')
    if len(tp_exits) > 0:
        ax1.scatter(tp_exits, df.loc[tp_exits, 'exit_price'], marker='o', color='blue', s=80, label='Take Profit Exit')
    if len(sl_exits) > 0:
        ax1.scatter(sl_exits, df.loc[sl_exits, 'exit_price'], marker='o', color='red', s=80, label='Stop Loss Exit')
    
    # Plot stop loss and take profit levels
    for i in range(len(df)):
        if df['position'].iloc[i] == 1:
            # Get the date and values
            date = df.index[i]
            entry = df['entry_price'].iloc[i]
            sl = df['stop_loss'].iloc[i]
            tp = df['take_profit'].iloc[i]
            
            # Find the next exit or the next date
            next_exit = None
            for j in range(i+1, len(df)):
                if df['exit_type'].iloc[j] != '':
                    next_exit = df.index[j]
                    break
            
            # If no exit found, use next date
            if next_exit is None and i < len(df) - 1:
                next_exit = df.index[i+1]
            
            # If we found a next point, draw the lines
            if next_exit is not None:
                ax1.plot([date, next_exit], [sl, sl], 'r--', linewidth=1, alpha=0.7)
                ax1.plot([date, next_exit], [tp, tp], 'b--', linewidth=1, alpha=0.7)
                
                # Draw fill between entry and stop (red) and entry and target (green)
                ax1.fill_between([date, next_exit], [entry, entry], [sl, sl], color='red', alpha=0.1)
                ax1.fill_between([date, next_exit], [entry, entry], [tp, tp], color='green', alpha=0.1)
    
    ax1.set_title('RSI Momentum Oscillator SwingRadar Strategy - Price Chart')
    ax1.set_ylabel('Price')
    ax1.grid(True)
    ax1.legend(loc='upper left')
    
    # Format x-axis to show dates properly
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax1.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # RSI and its MA
    ax2 = plt.subplot(4, 1, 2)
    ax2.plot(df.index, df['rsi'], label='RSI (14)', color='blue', linewidth=1)
    ax2.plot(df.index, df['rsi_ma'], label='RSI MA (14)', color='red', linewidth=1)
    ax2.axhline(y=70, color='red', linestyle='--', linewidth=0.5)
    ax2.axhline(y=50, color='black', linestyle='--', linewidth=0.5)
    ax2.axhline(y=35, color='green', linestyle='--', linewidth=0.5, label='Oversold (35)')
    ax2.axhline(y=30, color='green', linestyle='--', linewidth=0.5)
    
    # Highlight RSI crossovers
    crossovers = df[df['rsi_cross_up']].index
    for date in crossovers:
        ax2.plot(date, df.loc[date, 'rsi'], 'go', markersize=5)
    
    ax2.fill_between(df.index, df['rsi'], 35, where=(df['rsi'] <= 35), color='green', alpha=0.3)
    
    ax2.set_title('RSI and RSI MA')
    ax2.set_ylabel('RSI')
    ax2.set_ylim(0, 100)
    ax2.grid(True)
    ax2.legend(loc='upper left')
    
    # ATR
    ax3 = plt.subplot(4, 1, 3)
    ax3.plot(df.index, df['atr'], label='ATR (14)', color='purple', linewidth=1)
    ax3.fill_between(df.index, 0, df['atr'], alpha=0.3, color='purple')
    
    ax3.set_title('Average True Range (ATR)')
    ax3.set_ylabel('ATR')
    ax3.grid(True)
    
    # Equity curve
    ax4 = plt.subplot(4, 1, 4)
    ax4.plot(df.index, df['equity_curve'], label='Equity Curve', color='green')
    
    # Add drawdown to secondary y-axis
    twin_ax = ax4.twinx()
    twin_ax.fill_between(df.index, df['drawdown'], 0, alpha=0.3, color='red', label='Drawdown')
    twin_ax.set_ylabel('Drawdown (%)')
    twin_ax.set_ylim(-50, 5)  # Adjust based on actual drawdowns
    
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
    print(f"{'No.':<4}{'Entry Date':<12}{'Exit Date':<12}{'Entry Price':<12}{'Exit Price':<12}{'Stop Loss':<12}{'Take Profit':<12}{'P&L %':<10}{'Exit Type':<12}")
    print("-" * 120)
    
    # Find entry dates for each trade
    entry_dates = []
    for i, (idx, trade) in enumerate(trades.iterrows()):
        # Find the most recent entry before this exit
        entry_mask = ((df.index < idx) & 
                      (df['position'] == 1) & 
                      (df['position'].shift(1) == 0))
        
        if entry_mask.any():
            entry_date = df[entry_mask].index[-1]
            
            # Format dates
            entry_date_str = entry_date.strftime('%Y-%m-%d')
            exit_date_str = idx.strftime('%Y-%m-%d')
            
            # Get trade details
            entry_price = df.loc[entry_date, 'entry_price']
            exit_price = trade['exit_price']
            stop_loss = df.loc[entry_date, 'stop_loss']
            take_profit = df.loc[entry_date, 'take_profit']
            pnl = trade['pnl']
            exit_type = trade['exit_type']
            
            print(f"{i+1:<4}{entry_date_str:<12}{exit_date_str:<12}"
                 f"{entry_price:<12.2f}{exit_price:<12.2f}{stop_loss:<12.2f}{take_profit:<12.2f}"
                 f"{pnl:<10.2f}{exit_type:<12}")
    
    print("-" * 120)

# Set strategy parameters
rsi_period = 14
rsi_ma_period = 14
atr_period = 14
rsi_oversold = 35
atr_multiplier = 0.5
risk_reward = 4.0

# Generate simulated data
df = generate_simulated_data(days=365, volatility=0.02, trend_strength=0.0002, 
                            trend_changes=5, include_pullbacks=True)

# Run strategy
result_df = rsi_momentum_strategy(df, rsi_period=rsi_period, rsi_ma_period=rsi_ma_period,
                                atr_period=atr_period, rsi_oversold=rsi_oversold,
                                atr_multiplier=atr_multiplier, risk_reward=risk_reward)

# Analyze performance
performance = analyze_performance(result_df)

# Display results
print("\nRSI Momentum Oscillator SwingRadar Strategy Performance:")
print(f"Total Trades: {performance['total_trades']}")
print(f"Win Rate: {performance['win_rate']:.2%}")
print(f"Average Return per Trade: {performance['avg_return']:.2f}%")
print(f"Total Return: {performance['total_return']:.2%}")
print(f"Maximum Drawdown: {performance['max_drawdown']:.2f}%")
print(f"Sharpe Ratio: {performance['sharpe_ratio']:.2f}")
print(f"Profit Factor: {performance['profit_factor']:.2f}")
print(f"Exit Types: Take Profit: {performance['tp_exits']}, Stop Loss: {performance['sl_exits']}, End: {performance['end_exits']}")

# Display trade details
display_trade_details(result_df)

# Plot results
plot_strategy_results(result_df)