import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta, time
import pytz
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

class PriceResonanceStrategy:
    """
    Multi-Structure Price Resonance Strategy implementation
    
    This strategy identifies trading opportunities based on the convergence of 
    Order Blocks (OB) and Fair Value Gaps (FVG).
    """
    
    def __init__(self, risk_reward_ratio=2.0):
        """
        Initialize the strategy with parameters
        
        Parameters:
        -----------
        risk_reward_ratio : float
            Risk-reward ratio for setting profit targets
        """
        self.risk_reward_ratio = risk_reward_ratio
    
    def is_new_york_session(self, dt):
        """
        Check if datetime is within New York trading session
        
        Parameters:
        -----------
        dt : datetime
            Datetime to check
        
        Returns:
        --------
        bool
            True if within New York session, False otherwise
        """
        # Convert to New York timezone
        ny_tz = pytz.timezone('America/New_York')
        if dt.tzinfo is None:
            # If no timezone info, assume UTC
            dt = pytz.utc.localize(dt)
        dt_ny = dt.astimezone(ny_tz)
        
        # Check if weekend
        if dt_ny.weekday() >= 5:  # 5 is Saturday, 6 is Sunday
            return False
        
        # Morning session: 9:30 AM - 12:00 PM
        morning_start = time(9, 30)
        morning_end = time(12, 0)
        
        # Afternoon session: 1:30 PM - 4:00 PM
        afternoon_start = time(13, 30)
        afternoon_end = time(16, 0)
        
        current_time = dt_ny.time()
        
        # Check if within either morning or afternoon session
        return ((current_time >= morning_start and current_time < morning_end) or
                (current_time >= afternoon_start and current_time <= afternoon_end))
    
    def identify_order_blocks(self, data):
        """
        Identify bullish and bearish Order Blocks
        
        Parameters:
        -----------
        data : DataFrame
            Price data with OHLC columns
        
        Returns:
        --------
        DataFrame
            Data with Order Block signals
        """
        df = data.copy()
        
        # Initialize Order Block columns
        df['bullish_ob'] = False
        df['bearish_ob'] = False
        
        # Calculate Order Blocks
        for i in range(2, len(df)):
            # Bullish Order Block
            # Previous two candles are bearish, followed by stronger bullish momentum
            if (df['close'].iloc[i-2] < df['open'].iloc[i-2] and  # Bearish candle 2 periods ago
                df['close'].iloc[i-1] > df['close'].iloc[i-2] and  # Price moves up from low
                df['close'].iloc[i] > df['close'].iloc[i-1]):      # Continued upward momentum
                df['bullish_ob'].iloc[i] = True
            
            # Bearish Order Block
            # Previous two candles are bullish, followed by stronger bearish momentum
            if (df['close'].iloc[i-2] > df['open'].iloc[i-2] and  # Bullish candle 2 periods ago
                df['close'].iloc[i-1] < df['close'].iloc[i-2] and  # Price moves down from high
                df['close'].iloc[i] < df['close'].iloc[i-1]):      # Continued downward momentum
                df['bearish_ob'].iloc[i] = True
        
        return df
    
    def identify_fair_value_gaps(self, data):
        """
        Identify bullish and bearish Fair Value Gaps
        
        Parameters:
        -----------
        data : DataFrame
            Price data with OHLC columns
        
        Returns:
        --------
        DataFrame
            Data with Fair Value Gap signals
        """
        df = data.copy()
        
        # Initialize Fair Value Gap columns
        df['bullish_fvg'] = False
        df['bearish_fvg'] = False
        
        # Calculate Fair Value Gaps
        for i in range(2, len(df)):
            # Bullish Fair Value Gap
            # Current candle's low is higher than the high of candle 2 periods ago
            if df['low'].iloc[i] > df['high'].iloc[i-2]:
                df['bullish_fvg'].iloc[i] = True
            
            # Bearish Fair Value Gap
            # Current candle's high is lower than the low of candle 2 periods ago
            if df['high'].iloc[i] < df['low'].iloc[i-2]:
                df['bearish_fvg'].iloc[i] = True
        
        return df
    
    def apply_time_filter(self, data):
        """
        Apply New York session time filter
        
        Parameters:
        -----------
        data : DataFrame
            Price data with index as datetime
        
        Returns:
        --------
        DataFrame
            Data with NY session filter applied
        """
        df = data.copy()
        
        # Apply New York session filter
        df['in_ny_session'] = df.index.map(self.is_new_york_session)
        
        return df
    
    def generate_signals(self, data):
        """
        Generate trading signals based on Order Blocks, Fair Value Gaps, and time filter
        
        Parameters:
        -----------
        data : DataFrame
            Price data with OHLC columns
        
        Returns:
        --------
        DataFrame
            Data with trading signals
        """
        # Identify Order Blocks
        df = self.identify_order_blocks(data)
        
        # Identify Fair Value Gaps
        df = self.identify_fair_value_gaps(df)
        
        # Apply time filter
        df = self.apply_time_filter(df)
        
        # Generate trading signals
        df['buy_signal'] = df['bullish_ob'] & df['bullish_fvg'] & df['in_ny_session']
        df['sell_signal'] = df['bearish_ob'] & df['bearish_fvg'] & df['in_ny_session']
        
        # Calculate stop loss and take profit levels
        df['long_entry'] = np.nan
        df['long_sl'] = np.nan
        df['long_tp'] = np.nan
        
        df['short_entry'] = np.nan
        df['short_sl'] = np.nan
        df['short_tp'] = np.nan
        
        # Process each signal
        for i in range(2, len(df)):
            if df['buy_signal'].iloc[i]:
                # Long trade
                entry = df['close'].iloc[i]
                sl = df['low'].iloc[i-2]  # Stop loss at the low of the Order Block
                tp = entry + self.risk_reward_ratio * (entry - sl)  # Take profit based on risk-reward ratio
                
                df['long_entry'].iloc[i] = entry
                df['long_sl'].iloc[i] = sl
                df['long_tp'].iloc[i] = tp
            
            if df['sell_signal'].iloc[i]:
                # Short trade
                entry = df['close'].iloc[i]
                sl = df['high'].iloc[i-2]  # Stop loss at the high of the Order Block
                tp = entry - self.risk_reward_ratio * (sl - entry)  # Take profit based on risk-reward ratio
                
                df['short_entry'].iloc[i] = entry
                df['short_sl'].iloc[i] = sl
                df['short_tp'].iloc[i] = tp
        
        return df
    
    def backtest(self, data):
        """
        Run backtest on the provided data
        
        Parameters:
        -----------
        data : DataFrame
            Price data with OHLC columns
        
        Returns:
        --------
        DataFrame
            Data with trading signals and performance metrics
        """
        # Generate signals
        df = self.generate_signals(data)
        
        # Initialize trade tracking columns
        df['position'] = 0  # 1 for long, -1 for short, 0 for flat
        df['trade_active'] = False
        df['trade_entry_price'] = np.nan
        df['trade_exit_price'] = np.nan
        df['trade_exit_type'] = None
        df['trade_pnl'] = 0.0
        df['equity'] = 10000.0  # Starting equity
        
        # Track current trade
        in_long = False
        in_short = False
        entry_price = 0.0
        entry_index = 0
        stop_loss = 0.0
        take_profit = 0.0
        
        # Process each bar
        for i in range(2, len(df)):
            # Default to previous equity
            df['equity'].iloc[i] = df['equity'].iloc[i-1]
            
            # Check for exit conditions for existing trades
            if in_long:
                # Check if stop loss was hit
                if df['low'].iloc[i] <= stop_loss:
                    # Exit long at stop loss
                    df['position'].iloc[i] = 0
                    df['trade_active'].iloc[i] = False
                    df['trade_exit_price'].iloc[i] = stop_loss
                    df['trade_exit_type'].iloc[i] = 'Stop Loss'
                    
                    # Calculate PnL
                    pnl = (stop_loss - entry_price) / entry_price
                    df['trade_pnl'].iloc[i] = pnl
                    
                    # Update equity
                    df['equity'].iloc[i] = df['equity'].iloc[i-1] * (1 + pnl)
                    
                    # Reset trade state
                    in_long = False
                
                # Check if take profit was hit
                elif df['high'].iloc[i] >= take_profit:
                    # Exit long at take profit
                    df['position'].iloc[i] = 0
                    df['trade_active'].iloc[i] = False
                    df['trade_exit_price'].iloc[i] = take_profit
                    df['trade_exit_type'].iloc[i] = 'Take Profit'
                    
                    # Calculate PnL
                    pnl = (take_profit - entry_price) / entry_price
                    df['trade_pnl'].iloc[i] = pnl
                    
                    # Update equity
                    df['equity'].iloc[i] = df['equity'].iloc[i-1] * (1 + pnl)
                    
                    # Reset trade state
                    in_long = False
                
                # Otherwise, maintain position
                else:
                    df['position'].iloc[i] = 1
                    df['trade_active'].iloc[i] = True
                    df['trade_entry_price'].iloc[i] = entry_price
            
            elif in_short:
                # Check if stop loss was hit
                if df['high'].iloc[i] >= stop_loss:
                    # Exit short at stop loss
                    df['position'].iloc[i] = 0
                    df['trade_active'].iloc[i] = False
                    df['trade_exit_price'].iloc[i] = stop_loss
                    df['trade_exit_type'].iloc[i] = 'Stop Loss'
                    
                    # Calculate PnL
                    pnl = (entry_price - stop_loss) / entry_price
                    df['trade_pnl'].iloc[i] = pnl
                    
                    # Update equity
                    df['equity'].iloc[i] = df['equity'].iloc[i-1] * (1 + pnl)
                    
                    # Reset trade state
                    in_short = False
                
                # Check if take profit was hit
                elif df['low'].iloc[i] <= take_profit:
                    # Exit short at take profit
                    df['position'].iloc[i] = 0
                    df['trade_active'].iloc[i] = False
                    df['trade_exit_price'].iloc[i] = take_profit
                    df['trade_exit_type'].iloc[i] = 'Take Profit'
                    
                    # Calculate PnL
                    pnl = (entry_price - take_profit) / entry_price
                    df['trade_pnl'].iloc[i] = pnl
                    
                    # Update equity
                    df['equity'].iloc[i] = df['equity'].iloc[i-1] * (1 + pnl)
                    
                    # Reset trade state
                    in_short = False
                
                # Otherwise, maintain position
                else:
                    df['position'].iloc[i] = -1
                    df['trade_active'].iloc[i] = True
                    df['trade_entry_price'].iloc[i] = entry_price
            
            # Check for new signals if not in a trade
            if not in_long and not in_short:
                if df['buy_signal'].iloc[i]:
                    # Enter long
                    entry_price = df['long_entry'].iloc[i]
                    stop_loss = df['long_sl'].iloc[i]
                    take_profit = df['long_tp'].iloc[i]
                    entry_index = i
                    
                    df['position'].iloc[i] = 1
                    df['trade_active'].iloc[i] = True
                    df['trade_entry_price'].iloc[i] = entry_price
                    
                    in_long = True
                
                elif df['sell_signal'].iloc[i]:
                    # Enter short
                    entry_price = df['short_entry'].iloc[i]
                    stop_loss = df['short_sl'].iloc[i]
                    take_profit = df['short_tp'].iloc[i]
                    entry_index = i
                    
                    df['position'].iloc[i] = -1
                    df['trade_active'].iloc[i] = True
                    df['trade_entry_price'].iloc[i] = entry_price
                    
                    in_short = True
        
        # Calculate cumulative returns
        df['cumulative_returns'] = (1 + df['trade_pnl']).cumprod() - 1
        
        return df
    
    def calculate_performance_metrics(self, results):
        """
        Calculate performance metrics from backtest results
        
        Parameters:
        -----------
        results : DataFrame
            Backtest results
        
        Returns:
        --------
        dict
            Dictionary of performance metrics
        """
        # Extract trades
        trades = results[(results['trade_exit_type'].notna()) & (results['trade_pnl'] != 0)].copy()
        
        if len(trades) == 0:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'max_drawdown': 0,
                'sharpe_ratio': 0,
                'total_return': 0
            }
        
        # Calculate metrics
        total_trades = len(trades)
        winning_trades = trades[trades['trade_pnl'] > 0]
        losing_trades = trades[trades['trade_pnl'] <= 0]
        
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
        
        avg_win = winning_trades['trade_pnl'].mean() if len(winning_trades) > 0 else 0
        avg_loss = losing_trades['trade_pnl'].mean() if len(losing_trades) > 0 else 0
        
        total_profit = winning_trades['trade_pnl'].sum() if len(winning_trades) > 0 else 0
        total_loss = abs(losing_trades['trade_pnl'].sum()) if len(losing_trades) > 0 else 0
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
        
        # Calculate drawdown
        equity_curve = results['equity']
        peak = equity_curve.expanding().max()
        drawdown = (equity_curve / peak - 1) * 100  # in percentage
        max_drawdown = drawdown.min()
        
        # Calculate Sharpe ratio (assuming 0% risk-free rate)
        returns = results['trade_pnl'][results['trade_pnl'] != 0]
        sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if len(returns) > 0 and returns.std() > 0 else 0
        
        # Final return
        total_return = (results['equity'].iloc[-1] / results['equity'].iloc[0] - 1) * 100  # in percentage
        
        # Return metrics
        metrics = {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'total_return': total_return
        }
        
        return metrics
    
    def plot_results(self, results):
        """
        Plot backtest results
        
        Parameters:
        -----------
        results : DataFrame
            Backtest results
        """
        plt.figure(figsize=(16, 20))
        
        # Plot 1: Price chart with signals
        ax1 = plt.subplot(4, 1, 1)
        
        # Plot price
        ax1.plot(results.index, results['close'], color='black', linewidth=1, label='Close')
        
        # Highlight NY session periods
        for i in range(len(results)):
            if results['in_ny_session'].iloc[i]:
                ax1.axvspan(results.index[i], results.index[min(i+1, len(results)-1)], 
                           alpha=0.1, color='green')
        
        # Plot buy signals
        buy_signals = results[results['buy_signal']]
        sell_signals = results[results['sell_signal']]
        
        ax1.scatter(buy_signals.index, buy_signals['close'], 
                   marker='^', color='green', s=100, label='Buy Signal')
        ax1.scatter(sell_signals.index, sell_signals['close'], 
                   marker='v', color='red', s=100, label='Sell Signal')
        
        # Plot stop loss and take profit levels for active trades
        for i in range(len(results)):
            if results['position'].iloc[i] == 1:  # Long position
                ax1.axhline(y=results['long_sl'].iloc[i], color='red', linestyle='--', alpha=0.5)
                ax1.axhline(y=results['long_tp'].iloc[i], color='green', linestyle='--', alpha=0.5)
            elif results['position'].iloc[i] == -1:  # Short position
                ax1.axhline(y=results['short_sl'].iloc[i], color='red', linestyle='--', alpha=0.5)
                ax1.axhline(y=results['short_tp'].iloc[i], color='green', linestyle='--', alpha=0.5)
        
        # Plot trade exits
        trade_exits = results[results['trade_exit_type'].notna()]
        
        # Plot stop losses
        sl_exits = trade_exits[trade_exits['trade_exit_type'] == 'Stop Loss']
        ax1.scatter(sl_exits.index, sl_exits['trade_exit_price'], 
                   marker='x', color='red', s=100, label='Stop Loss')
        
        # Plot take profits
        tp_exits = trade_exits[trade_exits['trade_exit_type'] == 'Take Profit']
        ax1.scatter(tp_exits.index, tp_exits['trade_exit_price'], 
                   marker='x', color='green', s=100, label='Take Profit')
        
        ax1.set_title('Price Chart with Signals')
        ax1.set_ylabel('Price')
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Order Blocks and Fair Value Gaps
        ax2 = plt.subplot(4, 1, 2)
        
        ax2.plot(results.index, results['close'], color='black', linewidth=1, alpha=0.5)
        
        # Plot bullish Order Blocks
        bullish_ob = results[results['bullish_ob']]
        ax2.scatter(bullish_ob.index, bullish_ob['close'], 
                   marker='o', color='green', s=50, label='Bullish OB')
        
        # Plot bearish Order Blocks
        bearish_ob = results[results['bearish_ob']]
        ax2.scatter(bearish_ob.index, bearish_ob['close'], 
                   marker='o', color='red', s=50, label='Bearish OB')
        
        # Plot bullish Fair Value Gaps
        bullish_fvg = results[results['bullish_fvg']]
        ax2.scatter(bullish_fvg.index, bullish_fvg['close'], 
                   marker='s', color='blue', s=50, label='Bullish FVG')
        
        # Plot bearish Fair Value Gaps
        bearish_fvg = results[results['bearish_fvg']]
        ax2.scatter(bearish_fvg.index, bearish_fvg['close'], 
                   marker='s', color='purple', s=50, label='Bearish FVG')
        
        ax2.set_title('Order Blocks and Fair Value Gaps')
        ax2.set_ylabel('Price')
        ax2.legend(loc='upper left')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Position
        ax3 = plt.subplot(4, 1, 3)
        
        ax3.plot(results.index, results['position'], color='blue', linewidth=1)
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        ax3.set_title('Position (1 = Long, -1 = Short, 0 = Flat)')
        ax3.set_ylabel('Position')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Equity Curve
        ax4 = plt.subplot(4, 1, 4)
        
        ax4.plot(results.index, results['equity'], color='green', linewidth=1)
        
        ax4.set_title('Equity Curve')
        ax4.set_ylabel('Equity ($)')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def print_performance_summary(self, metrics):
        """
        Print performance summary
        
        Parameters:
        -----------
        metrics : dict
            Dictionary of performance metrics
        """
        print("=" * 50)
        print("PERFORMANCE SUMMARY")
        print("=" * 50)
        
        print(f"Total Trades: {metrics['total_trades']}")
        print(f"Win Rate: {metrics['win_rate']:.2%}")
        print(f"Profit Factor: {metrics['profit_factor']:.2f}")
        print(f"Average Win: {metrics['avg_win']:.2%}")
        print(f"Average Loss: {metrics['avg_loss']:.2%}")
        print(f"Maximum Drawdown: {metrics['max_drawdown']:.2f}%")
        print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        print(f"Total Return: {metrics['total_return']:.2f}%")
        
        print("=" * 50)

def generate_synthetic_data(days=60, minutes_per_day=390, start_date='2025-01-01', seed=42):
    """
    Generate synthetic price data with realistic patterns, including gaps between days
    
    Parameters:
    -----------
    days : int
        Number of trading days
    minutes_per_day : int
        Number of minutes per trading day
    start_date : str
        Start date in 'YYYY-MM-DD' format
    seed : int
        Random seed for reproducibility
    
    Returns:
    --------
    DataFrame
        Synthetic price data with OHLCV columns
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Parse start date
    start = datetime.strptime(start_date, '%Y-%m-%d')
    
    # Generate timestamps
    timestamps = []
    for day in range(days):
        current_date = start + timedelta(days=day)
        
        # Skip weekends
        if current_date.weekday() >= 5:  # 5 is Saturday, 6 is Sunday
            continue
        
        # Generate minutes for the trading day (9:30 AM to 4:00 PM)
        for minute in range(minutes_per_day):
            # Start at 9:30 AM
            current_time = current_date + timedelta(minutes=minute + 9*60 + 30)
            timestamps.append(current_time)
    
    # Initialize with a base price
    base_price = 100.0
    
    # Parameters for price generation
    volatility = 0.001  # Intraday volatility
    trend = 0.0001  # Small upward trend
    
    # Generate prices
    prices = [base_price]
    
    # Add a small price jump between days to simulate gaps
    day_boundaries = []
    
    for i in range(1, len(timestamps)):
        # Check if it's a new day
        if timestamps[i].date() != timestamps[i-1].date():
            # Add a day boundary
            day_boundaries.append(i)
            
            # Simulate a gap (up or down)
            gap = np.random.normal(0, 0.005)
            new_price = prices[-1] * (1 + gap)
        else:
            # Intraday price movement
            # Combine random walk with some mean reversion and momentum
            momentum = 0.1  # Momentum factor
            mean_reversion = 0.1  # Mean reversion factor
            
            # Calculate momentum component (continuation of recent direction)
            if len(prices) > 10:
                recent_direction = (prices[-1] / prices[-10]) - 1
                momentum_component = momentum * recent_direction
            else:
                momentum_component = 0
                
            # Calculate mean reversion component
            if len(prices) > 50:
                recent_mean = np.mean(prices[-50:])
                mean_reversion_component = mean_reversion * ((recent_mean / prices[-1]) - 1)
            else:
                mean_reversion_component = 0
            
            # Combine components
            random_component = np.random.normal(trend, volatility)
            total_return = random_component + momentum_component + mean_reversion_component
            
            new_price = prices[-1] * (1 + total_return)
        
        prices.append(new_price)
    
    # Generate OHLC data
    opens = np.zeros(len(timestamps))
    highs = np.zeros(len(timestamps))
    lows = np.zeros(len(timestamps))
    closes = np.array(prices)
    volumes = np.zeros(len(timestamps))
    
    # First bar
    opens[0] = prices[0]
    # Random high and low around close
    highs[0] = max(opens[0], closes[0]) * (1 + abs(np.random.normal(0, volatility)))
    lows[0] = min(opens[0], closes[0]) * (1 - abs(np.random.normal(0, volatility)))
    # Random volume
    volumes[0] = np.random.lognormal(10, 1)
    
    # Rest of the bars
    for i in range(1, len(timestamps)):
        if i in day_boundaries:
            # For bars at day boundaries, open is close of previous day with a gap
            opens[i] = closes[i-1] * (1 + np.random.normal(0, 0.005))
        else:
            # For intraday bars, open is close of previous bar
            opens[i] = closes[i-1]
        
        # Random high and low around open and close
        highs[i] = max(opens[i], closes[i]) * (1 + abs(np.random.normal(0, volatility)))
        lows[i] = min(opens[i], closes[i]) * (1 - abs(np.random.normal(0, volatility)))
        
        # Make sure high >= open, close and low <= open, close
        highs[i] = max(highs[i], opens[i], closes[i])
        lows[i] = min(lows[i], opens[i], closes[i])
        
        # Random volume with some clustering (higher volume on big moves)
        price_change = abs(closes[i] / closes[i-1] - 1)
        volume_factor = 1 + 5 * price_change  # Higher volume on larger price changes
        volumes[i] = np.random.lognormal(10, 1) * volume_factor
    
    # Create DataFrame
    df = pd.DataFrame({
        'open': opens,
        'high': highs,
        'low': lows,
        'close': closes,
        'volume': volumes
    }, index=timestamps)
    
    return df

def resample_to_timeframe(data, timeframe='3min'):
    """
    Resample data to the specified timeframe
    
    Parameters:
    -----------
    data : DataFrame
        Price data with OHLCV columns
    timeframe : str
        Timeframe to resample to (e.g., '3min', '5min', '15min', '1h')
    
    Returns:
    --------
    DataFrame
        Resampled price data
    """
    # Resample data
    resampled = pd.DataFrame()
    resampled['open'] = data['open'].resample(timeframe).first()
    resampled['high'] = data['high'].resample(timeframe).max()
    resampled['low'] = data['low'].resample(timeframe).min()
    resampled['close'] = data['close'].resample(timeframe).last()
    resampled['volume'] = data['volume'].resample(timeframe).sum()
    
    # Drop any rows with NaN values
    resampled = resampled.dropna()
    
    return resampled

def run_strategy_test():
    """
    Run a complete test of the strategy
    
    Returns:
    --------
    tuple
        (strategy, results, metrics)
    """
    # Generate synthetic data
    print("Generating synthetic data...")
    data = generate_synthetic_data(days=60, minutes_per_day=390, start_date='2025-01-01', seed=42)
    
    # Resample to 3-minute timeframe (as per the original strategy)
    print("Resampling to 3-minute timeframe...")
    data_3min = resample_to_timeframe(data, timeframe='3min')
    
    # Create strategy instance
    strategy = PriceResonanceStrategy(risk_reward_ratio=2.0)
    
    # Run backtest
    print("Running backtest...")
    results = strategy.backtest(data_3min)
    
    # Calculate performance metrics
    metrics = strategy.calculate_performance_metrics(results)
    
    # Print and plot results
    strategy.print_performance_summary(metrics)
    strategy.plot_results(results)
    
    return strategy, results, metrics

def analyze_parameter_sensitivity():
    """
    Analyze the sensitivity of the strategy to different risk-reward ratios
    
    Returns:
    --------
    DataFrame
        Results of parameter sensitivity analysis
    """
    # Generate synthetic data
    print("Generating synthetic data for parameter sensitivity analysis...")
    data = generate_synthetic_data(days=60, minutes_per_day=390, start_date='2025-01-01', seed=42)
    
    # Resample to 3-minute timeframe
    data_3min = resample_to_timeframe(data, timeframe='3min')
    
    # Test different risk-reward ratios
    rr_ratios = [1.0, 1.5, 2.0, 2.5, 3.0]
    results = []
    
    for rr in rr_ratios:
        print(f"Testing risk-reward ratio: {rr}")
        
        # Create strategy instance
        strategy = PriceResonanceStrategy(risk_reward_ratio=rr)
        
        # Run backtest
        backtest_results = strategy.backtest(data_3min)
        
        # Calculate performance metrics
        metrics = strategy.calculate_performance_metrics(backtest_results)
        
        # Store results
        result = {
            'risk_reward_ratio': rr,
            'total_trades': metrics['total_trades'],
            'win_rate': metrics['win_rate'],
            'profit_factor': metrics['profit_factor'],
            'total_return': metrics['total_return'],
            'max_drawdown': metrics['max_drawdown'],
            'sharpe_ratio': metrics['sharpe_ratio']
        }
        
        results.append(result)
    
    # Create DataFrame
    results_df = pd.DataFrame(results)
    
    # Plot results
    plt.figure(figsize=(15, 10))
    
    # Plot win rate vs. risk-reward ratio
    plt.subplot(2, 2, 1)
    plt.plot(results_df['risk_reward_ratio'], results_df['win_rate'], marker='o')
    plt.title('Win Rate vs. Risk-Reward Ratio')
    plt.xlabel('Risk-Reward Ratio')
    plt.ylabel('Win Rate')
    plt.grid(True, alpha=0.3)
    
    # Plot total return vs. risk-reward ratio
    plt.subplot(2, 2, 2)
    plt.plot(results_df['risk_reward_ratio'], results_df['total_return'], marker='o')
    plt.title('Total Return vs. Risk-Reward Ratio')
    plt.xlabel('Risk-Reward Ratio')
    plt.ylabel('Total Return (%)')
    plt.grid(True, alpha=0.3)
    
    # Plot profit factor vs. risk-reward ratio
    plt.subplot(2, 2, 3)
    plt.plot(results_df['risk_reward_ratio'], results_df['profit_factor'], marker='o')
    plt.title('Profit Factor vs. Risk-Reward Ratio')
    plt.xlabel('Risk-Reward Ratio')
    plt.ylabel('Profit Factor')
    plt.grid(True, alpha=0.3)
    
    # Plot Sharpe ratio vs. risk-reward ratio
    plt.subplot(2, 2, 4)
    plt.plot(results_df['risk_reward_ratio'], results_df['sharpe_ratio'], marker='o')
    plt.title('Sharpe Ratio vs. Risk-Reward Ratio')
    plt.xlabel('Risk-Reward Ratio')
    plt.ylabel('Sharpe Ratio')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return results_df

def analyze_time_filter_impact():
    """
    Analyze the impact of the NY session time filter
    
    Returns:
    --------
    dict
        Results with and without time filter
    """
    # Generate synthetic data
    print("Generating synthetic data for time filter analysis...")
    data = generate_synthetic_data(days=60, minutes_per_day=390, start_date='2025-01-01', seed=42)
    
    # Resample to 3-minute timeframe
    data_3min = resample_to_timeframe(data, timeframe='3min')
    
    # Strategy with time filter (original)
    print("Testing strategy with time filter...")
    strategy_with_filter = PriceResonanceStrategy(risk_reward_ratio=2.0)
    results_with_filter = strategy_with_filter.backtest(data_3min)
    metrics_with_filter = strategy_with_filter.calculate_performance_metrics(results_with_filter)
    
    # Create a custom version of the strategy without time filter
    class StrategyWithoutTimeFilter(PriceResonanceStrategy):
        def apply_time_filter(self, data):
            df = data.copy()
            df['in_ny_session'] = True  # Always in session
            return df
    
    # Strategy without time filter
    print("Testing strategy without time filter...")
    strategy_without_filter = StrategyWithoutTimeFilter(risk_reward_ratio=2.0)
    results_without_filter = strategy_without_filter.backtest(data_3min)
    metrics_without_filter = strategy_without_filter.calculate_performance_metrics(results_without_filter)
    
    # Print comparison
    print("\n" + "=" * 50)
    print("TIME FILTER IMPACT COMPARISON")
    print("=" * 50)
    
    print("\nWith Time Filter:")
    strategy_with_filter.print_performance_summary(metrics_with_filter)
    
    print("\nWithout Time Filter:")
    strategy_without_filter.print_performance_summary(metrics_without_filter)
    
    # Plot comparison
    plt.figure(figsize=(15, 7))
    
    plt.subplot(1, 2, 1)
    plt.plot(results_with_filter.index, results_with_filter['equity'], label='With Time Filter')
    plt.plot(results_without_filter.index, results_without_filter['equity'], label='Without Time Filter')
    plt.title('Equity Curve Comparison')
    plt.xlabel('Date')
    plt.ylabel('Equity ($)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    
    # Create comparison data
    comparison = pd.DataFrame({
        'With Filter': [
            metrics_with_filter['total_trades'],
            metrics_with_filter['win_rate'] * 100,
            metrics_with_filter['profit_factor'],
            metrics_with_filter['total_return']
        ],
        'Without Filter': [
            metrics_without_filter['total_trades'],
            metrics_without_filter['win_rate'] * 100,
            metrics_without_filter['profit_factor'],
            metrics_without_filter['total_return']
        ]
    }, index=['Total Trades', 'Win Rate (%)', 'Profit Factor', 'Total Return (%)'])
    
    comparison.plot(kind='bar', ax=plt.gca())
    plt.title('Performance Metrics Comparison')
    plt.ylabel('Value')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return {
        'with_filter': {
            'results': results_with_filter,
            'metrics': metrics_with_filter
        },
        'without_filter': {
            'results': results_without_filter,
            'metrics': metrics_without_filter
        }
    }

# Run the tests
if __name__ == "__main__":
    # Run main strategy test
    strategy, results, metrics = run_strategy_test()
    
    # Analyze parameter sensitivity
    param_results = analyze_parameter_sensitivity()
    
    # Analyze time filter impact
    time_filter_results = analyze_time_filter_impact()