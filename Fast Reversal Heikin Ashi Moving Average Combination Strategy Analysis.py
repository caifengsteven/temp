import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
import matplotlib.colors as mcolors
from datetime import datetime, timedelta
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

class HeikinAshiReversal:
    """
    Implementation of the Fast Reversal Heikin Ashi Moving Average Combination Strategy
    
    This strategy identifies reversal patterns using Heikin Ashi candles and two
    simple moving averages (SMA9 and SMA30).
    """
    
    def __init__(self, 
                 fast_sma_period=9, 
                 slow_sma_period=30, 
                 doji_threshold=0.3, 
                 wick_threshold=0.3):
        """
        Initialize the strategy with parameters
        
        Parameters:
        -----------
        fast_sma_period : int
            Period for the fast SMA (default: 9)
        slow_sma_period : int
            Period for the slow SMA (default: 30)
        doji_threshold : float
            Threshold for identifying a Doji candle (default: 0.3)
        wick_threshold : float
            Threshold for identifying a no-wick candle (default: 0.3)
        """
        self.fast_sma_period = fast_sma_period
        self.slow_sma_period = slow_sma_period
        self.doji_threshold = doji_threshold
        self.wick_threshold = wick_threshold
    
    def calculate_heikin_ashi(self, data):
        """
        Calculate Heikin Ashi candles from regular OHLC data
        
        Parameters:
        -----------
        data : DataFrame
            Price data with OHLC columns
            
        Returns:
        --------
        DataFrame
            Data with added Heikin Ashi columns
        """
        df = data.copy()
        
        # Initialize Heikin Ashi columns
        df['ha_open'] = np.nan
        df['ha_high'] = np.nan
        df['ha_low'] = np.nan
        df['ha_close'] = np.nan
        
        # Calculate first Heikin Ashi candle
        df.loc[df.index[0], 'ha_open'] = df.loc[df.index[0], 'open']
        df.loc[df.index[0], 'ha_close'] = (df.loc[df.index[0], 'open'] + 
                                          df.loc[df.index[0], 'high'] + 
                                          df.loc[df.index[0], 'low'] + 
                                          df.loc[df.index[0], 'close']) / 4
        df.loc[df.index[0], 'ha_high'] = df.loc[df.index[0], 'high']
        df.loc[df.index[0], 'ha_low'] = df.loc[df.index[0], 'low']
        
        # Calculate remaining Heikin Ashi candles
        for i in range(1, len(df)):
            df.loc[df.index[i], 'ha_close'] = (df.loc[df.index[i], 'open'] + 
                                              df.loc[df.index[i], 'high'] + 
                                              df.loc[df.index[i], 'low'] + 
                                              df.loc[df.index[i], 'close']) / 4
            df.loc[df.index[i], 'ha_open'] = (df.loc[df.index[i-1], 'ha_open'] + 
                                             df.loc[df.index[i-1], 'ha_close']) / 2
            df.loc[df.index[i], 'ha_high'] = max(df.loc[df.index[i], 'high'], 
                                                df.loc[df.index[i], 'ha_open'], 
                                                df.loc[df.index[i], 'ha_close'])
            df.loc[df.index[i], 'ha_low'] = min(df.loc[df.index[i], 'low'], 
                                               df.loc[df.index[i], 'ha_open'], 
                                               df.loc[df.index[i], 'ha_close'])
        
        return df
    
    def calculate_indicators(self, data):
        """
        Calculate all technical indicators required for the strategy
        
        Parameters:
        -----------
        data : DataFrame
            Price data with OHLC columns
            
        Returns:
        --------
        DataFrame
            Data with added indicator columns
        """
        # Calculate Heikin Ashi candles
        df = self.calculate_heikin_ashi(data)
        
        # Calculate SMAs on Heikin Ashi close
        df['sma_fast'] = df['ha_close'].rolling(window=self.fast_sma_period).mean()
        df['sma_slow'] = df['ha_close'].rolling(window=self.slow_sma_period).mean()
        
        # Calculate body and range for each candle
        df['ha_body'] = abs(df['ha_close'] - df['ha_open'])
        df['ha_range'] = df['ha_high'] - df['ha_low']
        
        # Identify Doji candles
        df['is_doji'] = df['ha_body'] <= df['ha_range'] * self.doji_threshold
        
        # Identify no-wick candles
        df['ha_upper_wick'] = df['ha_high'] - df[['ha_open', 'ha_close']].max(axis=1)
        df['ha_lower_wick'] = df[['ha_open', 'ha_close']].min(axis=1) - df['ha_low']
        df['is_no_wick'] = (df['ha_upper_wick'] <= df['ha_range'] * self.wick_threshold) & \
                           (df['ha_lower_wick'] <= df['ha_range'] * self.wick_threshold)
        
        # Identify bullish and bearish candles
        df['is_bull'] = df['ha_close'] > df['ha_open']
        df['is_bear'] = df['ha_close'] < df['ha_open']
        
        # Identify previous candle Doji condition
        df['prev_doji'] = df['is_doji'].shift(1)
        
        # Generate entry signals
        df['long_condition'] = (df['prev_doji']) & \
                              (df['is_no_wick']) & \
                              (df['is_bull']) & \
                              (df['ha_close'] > df['sma_fast'])
        
        df['short_condition'] = (df['prev_doji']) & \
                               (df['is_no_wick']) & \
                               (df['is_bear']) & \
                               (df['ha_close'] < df['sma_fast'])
        
        return df
    
    def backtest(self, data, initial_capital=10000, position_size_pct=100):
        """
        Run backtest on the provided data
        
        Parameters:
        -----------
        data : DataFrame
            Price data with OHLC columns
        initial_capital : float
            Initial capital for the backtest
        position_size_pct : float
            Position size as percentage of equity
            
        Returns:
        --------
        DataFrame
            Data with added signal and performance columns
        """
        # Calculate indicators
        df = self.calculate_indicators(data)
        
        # Initialize trading variables
        df['position'] = 0  # 1 for long, -1 for short, 0 for flat
        df['equity'] = initial_capital
        df['cash'] = initial_capital
        df['holdings'] = 0
        df['trade_pnl'] = 0
        
        # Process each bar
        position = 0
        entry_price = 0
        entry_index = 0
        for i in range(max(self.fast_sma_period, self.slow_sma_period) + 1, len(df)):
            # Default is to carry forward previous values
            if i > 0:
                df.loc[df.index[i], 'position'] = df.loc[df.index[i-1], 'position']
                df.loc[df.index[i], 'equity'] = df.loc[df.index[i-1], 'equity']
                df.loc[df.index[i], 'cash'] = df.loc[df.index[i-1], 'cash']
                df.loc[df.index[i], 'holdings'] = df.loc[df.index[i-1], 'holdings']
            
            current_close = df.loc[df.index[i], 'close']
            
            # Process long signal
            if df.loc[df.index[i], 'long_condition']:
                # If in short position, close it first
                if position == -1:
                    # Calculate profit/loss from short trade
                    trade_pnl = entry_price - current_close
                    df.loc[df.index[i], 'trade_pnl'] += trade_pnl
                    df.loc[df.index[i], 'equity'] += trade_pnl
                    df.loc[df.index[i], 'cash'] = df.loc[df.index[i], 'equity']
                
                # Enter long position
                position = 1
                entry_price = current_close
                entry_index = i
                
                # Calculate position size
                position_size = df.loc[df.index[i], 'equity'] * (position_size_pct / 100) / current_close
                
                # Update tracking variables
                df.loc[df.index[i], 'position'] = position
                df.loc[df.index[i], 'holdings'] = position_size * current_close
                df.loc[df.index[i], 'cash'] = df.loc[df.index[i], 'equity'] - df.loc[df.index[i], 'holdings']
            
            # Process short signal
            elif df.loc[df.index[i], 'short_condition']:
                # If in long position, close it first
                if position == 1:
                    # Calculate profit/loss from long trade
                    trade_pnl = current_close - entry_price
                    df.loc[df.index[i], 'trade_pnl'] += trade_pnl
                    df.loc[df.index[i], 'equity'] += trade_pnl
                    df.loc[df.index[i], 'cash'] = df.loc[df.index[i], 'equity']
                
                # Enter short position
                position = -1
                entry_price = current_close
                entry_index = i
                
                # Calculate position size
                position_size = df.loc[df.index[i], 'equity'] * (position_size_pct / 100) / current_close
                
                # Update tracking variables
                df.loc[df.index[i], 'position'] = position
                df.loc[df.index[i], 'holdings'] = -position_size * current_close
                df.loc[df.index[i], 'cash'] = df.loc[df.index[i], 'equity'] - df.loc[df.index[i], 'holdings']
            
            # Update value of holdings
            if position == 1:
                # Long position
                df.loc[df.index[i], 'holdings'] = position_size * current_close
                df.loc[df.index[i], 'equity'] = df.loc[df.index[i], 'cash'] + df.loc[df.index[i], 'holdings']
            elif position == -1:
                # Short position
                df.loc[df.index[i], 'holdings'] = -position_size * current_close
                df.loc[df.index[i], 'equity'] = df.loc[df.index[i], 'cash'] - df.loc[df.index[i], 'holdings']
        
        # Calculate daily returns
        df['daily_returns'] = df['equity'].pct_change()
        
        # Calculate cumulative returns
        df['cumulative_returns'] = (1 + df['daily_returns']).cumprod() - 1
        
        return df
    
    def calculate_performance_metrics(self, results):
        """
        Calculate performance metrics
        
        Parameters:
        -----------
        results : DataFrame
            Backtest results
            
        Returns:
        --------
        dict
            Dictionary of performance metrics
        """
        # Extract relevant data
        equity = results['equity']
        daily_returns = results['daily_returns'].dropna()
        trades = results[results['trade_pnl'] != 0]
        
        # Total return
        total_return = (equity[-1] / equity[0]) - 1
        
        # Annualized return (assuming 252 trading days per year)
        num_days = len(daily_returns)
        annualized_return = (1 + total_return) ** (252 / num_days) - 1
        
        # Volatility
        volatility = daily_returns.std() * np.sqrt(252)
        
        # Sharpe ratio (assuming risk-free rate = 0)
        sharpe_ratio = annualized_return / volatility if volatility != 0 else 0
        
        # Maximum drawdown
        cumulative_returns = (1 + daily_returns).cumprod()
        max_drawdown = (cumulative_returns / cumulative_returns.cummax() - 1).min()
        
        # Win rate
        total_trades = len(trades)
        winning_trades = len(trades[trades['trade_pnl'] > 0])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # Average profit/loss
        avg_profit = trades[trades['trade_pnl'] > 0]['trade_pnl'].mean() if winning_trades > 0 else 0
        avg_loss = trades[trades['trade_pnl'] < 0]['trade_pnl'].mean() if total_trades - winning_trades > 0 else 0
        
        # Profit factor
        gross_profit = trades[trades['trade_pnl'] > 0]['trade_pnl'].sum() if winning_trades > 0 else 0
        gross_loss = abs(trades[trades['trade_pnl'] < 0]['trade_pnl'].sum()) if total_trades - winning_trades > 0 else 0
        profit_factor = gross_profit / gross_loss if gross_loss != 0 else float('inf')
        
        # Return metrics
        metrics = {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'avg_profit': avg_profit,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'total_trades': total_trades
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
        plt.figure(figsize=(16, 16))
        
        # Plot 1: Price and Heikin Ashi candles
        ax1 = plt.subplot(4, 1, 1)
        
        # Plot Heikin Ashi candles
        for i in range(len(results)):
            # Determine candle color
            if results['ha_close'].iloc[i] >= results['ha_open'].iloc[i]:
                color = 'green'
                bodycolor = 'green'
            else:
                color = 'red'
                bodycolor = 'red'
            
            # Plot candle body
            body_height = abs(results['ha_close'].iloc[i] - results['ha_open'].iloc[i])
            if body_height == 0:
                body_height = 0.01  # Minimum height for visibility
            body_bottom = min(results['ha_close'].iloc[i], results['ha_open'].iloc[i])
            
            rect = Rectangle((i, body_bottom), 0.8, body_height, fill=True, color=bodycolor, alpha=0.6)
            ax1.add_patch(rect)
            
            # Plot candle wicks
            ax1.plot([i+0.4, i+0.4], [results['ha_low'].iloc[i], results['ha_high'].iloc[i]], color=color, linewidth=1)
        
        # Plot SMAs
        ax1.plot(results['sma_fast'], color='blue', linewidth=1.5, label=f'SMA {self.fast_sma_period}')
        ax1.plot(results['sma_slow'], color='red', linewidth=1.5, label=f'SMA {self.slow_sma_period}')
        
        # Plot signals
        long_signals = results[results['long_condition']].index
        short_signals = results[results['short_condition']].index
        
        for signal_idx in long_signals:
            signal_pos = results.index.get_loc(signal_idx)
            ax1.plot(signal_pos, results.loc[signal_idx, 'ha_low'] * 0.99, '^', color='green', markersize=10)
        
        for signal_idx in short_signals:
            signal_pos = results.index.get_loc(signal_idx)
            ax1.plot(signal_pos, results.loc[signal_idx, 'ha_high'] * 1.01, 'v', color='red', markersize=10)
        
        ax1.set_title('Heikin Ashi Candles with SMAs and Signals')
        ax1.set_ylabel('Price')
        ax1.set_xticks([])
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Pattern Indicators
        ax2 = plt.subplot(4, 1, 2)
        
        # Plot Doji indicator
        ax2.plot(results['is_doji'], label='Doji', color='blue', alpha=0.5)
        
        # Plot No-Wick indicator
        ax2.plot(results['is_no_wick'], label='No-Wick', color='green', alpha=0.5)
        
        # Plot combined patterns
        ax2.plot(results['prev_doji'] & results['is_no_wick'], label='Doji + No-Wick', color='red')
        
        ax2.set_title('Pattern Indicators')
        ax2.set_ylabel('Status')
        ax2.set_yticks([0, 1])
        ax2.set_yticklabels(['False', 'True'])
        ax2.set_xticks([])
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Position
        ax3 = plt.subplot(4, 1, 3)
        
        ax3.plot(results['position'], label='Position', color='blue')
        ax3.axhline(0, color='black', linestyle='--', alpha=0.3)
        
        ax3.set_title('Position (1 = Long, -1 = Short, 0 = Flat)')
        ax3.set_ylabel('Position')
        ax3.set_yticks([-1, 0, 1])
        ax3.set_xticks([])
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Equity Curve
        ax4 = plt.subplot(4, 1, 4)
        
        ax4.plot(results['equity'], label='Strategy Equity', color='green')
        
        # Calculate and plot buy & hold equity
        initial_capital = results['equity'].iloc[0]
        buy_hold_equity = initial_capital * (results['close'] / results['close'].iloc[0])
        ax4.plot(buy_hold_equity, label='Buy & Hold', color='blue', alpha=0.5)
        
        ax4.set_title('Equity Curve')
        ax4.set_ylabel('Equity')
        ax4.set_xlabel('Date')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Additional plot: Drawdown
        plt.figure(figsize=(16, 6))
        
        equity = results['equity']
        peak = equity.cummax()
        drawdown = (equity / peak - 1) * 100
        
        plt.plot(drawdown, color='red')
        plt.fill_between(range(len(drawdown)), drawdown, 0, color='red', alpha=0.3)
        
        plt.title('Drawdown (%)')
        plt.ylabel('Drawdown (%)')
        plt.xlabel('Date')
        plt.grid(True, alpha=0.3)
        
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
        
        print(f"Total Return: {metrics['total_return']:.2%}")
        print(f"Annualized Return: {metrics['annualized_return']:.2%}")
        print(f"Volatility: {metrics['volatility']:.2%}")
        print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        print(f"Maximum Drawdown: {metrics['max_drawdown']:.2%}")
        print(f"Win Rate: {metrics['win_rate']:.2%}")
        print(f"Average Profit: {metrics['avg_profit']:.2f}")
        print(f"Average Loss: {metrics['avg_loss']:.2f}")
        print(f"Profit Factor: {metrics['profit_factor']:.2f}")
        print(f"Total Trades: {metrics['total_trades']}")
        
        print("=" * 50)

def generate_market_data(days=500, intraday_points=24, trend_cycles=3, volatility_cycles=2, seed=42):
    """
    Generate synthetic market data with trends, reversals, and realistic intraday patterns
    
    Parameters:
    -----------
    days : int
        Number of trading days
    intraday_points : int
        Number of price points per day
    trend_cycles : int
        Number of major trend cycles
    volatility_cycles : int
        Number of volatility cycles
    seed : int
        Random seed for reproducibility
        
    Returns:
    --------
    DataFrame
        Synthetic market data with OHLC columns
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Generate timestamps
    timestamps = []
    for day in range(days):
        date = datetime(2023, 1, 1) + timedelta(days=day)
        if date.weekday() < 5:  # Only weekdays
            for point in range(intraday_points):
                hour = 9 + point // 2
                minute = (point % 2) * 30
                timestamps.append(datetime(date.year, date.month, date.day, hour, minute))
    
    # Total number of points
    total_points = len(timestamps)
    
    # Generate trends
    trend_period = total_points // trend_cycles
    trend_direction = np.sin(np.linspace(0, trend_cycles * 2 * np.pi, total_points))
    
    # Generate volatility cycles
    vol_period = total_points // volatility_cycles
    volatility = 0.5 + 0.5 * np.sin(np.linspace(0, volatility_cycles * 2 * np.pi, total_points))
    
    # Generate price series
    base_price = 100
    price_series = np.zeros(total_points)
    price_series[0] = base_price
    
    # Base price movement parameters
    base_drift = 0.0001
    base_volatility = 0.001
    
    # Intraday pattern parameters
    intraday_pattern = np.sin(np.linspace(0, 2 * np.pi, intraday_points))
    
    # Generate price movements
    for i in range(1, total_points):
        # Trend component
        trend_component = base_drift * trend_direction[i]
        
        # Volatility component
        vol_component = base_volatility * volatility[i] * np.random.normal()
        
        # Intraday pattern component
        intraday_index = i % intraday_points
        intraday_component = 0.0005 * intraday_pattern[intraday_index]
        
        # Calculate price change
        price_change = trend_component + vol_component + intraday_component
        
        # Apply price change
        price_series[i] = price_series[i-1] * (1 + price_change)
    
    # Generate OHLC data
    df = pd.DataFrame(index=timestamps)
    
    # Add gaps between days
    prev_day = timestamps[0].date()
    day_close = price_series[0]
    
    for i in range(total_points):
        current_day = timestamps[i].date()
        
        # If new day, add gap
        if current_day != prev_day:
            gap = np.random.normal(0, 0.005)  # Random gap between -0.5% and +0.5%
            price_series[i:] = price_series[i:] * (1 + gap)
            prev_day = current_day
    
    # Generate OHLC for each point
    opens = np.zeros(total_points)
    highs = np.zeros(total_points)
    lows = np.zeros(total_points)
    closes = price_series
    
    # First point
    opens[0] = price_series[0]
    intraday_vol = price_series[0] * 0.001 * volatility[0]
    highs[0] = price_series[0] + intraday_vol
    lows[0] = price_series[0] - intraday_vol
    
    # Rest of the points
    for i in range(1, total_points):
        # Open is close of previous point if same day, otherwise add gap
        if timestamps[i].date() == timestamps[i-1].date():
            opens[i] = closes[i-1]
        else:
            gap = np.random.normal(0, 0.005)
            opens[i] = closes[i-1] * (1 + gap)
        
        # Calculate intraday volatility
        intraday_vol = price_series[i] * 0.002 * volatility[i]
        
        # High and low based on open and close
        if opens[i] <= closes[i]:  # Up bar
            highs[i] = closes[i] + intraday_vol * np.random.random()
            lows[i] = opens[i] - intraday_vol * np.random.random()
        else:  # Down bar
            highs[i] = opens[i] + intraday_vol * np.random.random()
            lows[i] = closes[i] - intraday_vol * np.random.random()
        
        # Ensure high >= max(open, close) and low <= min(open, close)
        highs[i] = max(highs[i], opens[i], closes[i])
        lows[i] = min(lows[i], opens[i], closes[i])
    
    # Create DataFrame
    df['open'] = opens
    df['high'] = highs
    df['low'] = lows
    df['close'] = closes
    
    # Add volume (higher on trend changes and volatile periods)
    volumes = np.zeros(total_points)
    base_volume = 1000
    
    for i in range(total_points):
        # Base volume
        volumes[i] = base_volume * (1 + 0.5 * np.random.random())
        
        # Add volume spike on trend changes
        if i > 0 and np.sign(trend_direction[i]) != np.sign(trend_direction[i-1]):
            volumes[i] *= 2 + np.random.random()
        
        # Add volume based on volatility
        volumes[i] *= 1 + volatility[i]
        
        # Add intraday pattern (typically higher at open and close)
        intraday_index = i % intraday_points
        if intraday_index < 3 or intraday_index > intraday_points - 4:
            volumes[i] *= 1.5
    
    df['volume'] = volumes
    
    return df

def test_strategy(data=None):
    """
    Test the Heikin Ashi Reversal strategy
    
    Parameters:
    -----------
    data : DataFrame, optional
        Price data to use (if None, generates synthetic data)
        
    Returns:
    --------
    tuple
        (strategy, results, metrics)
    """
    # Generate synthetic data if not provided
    if data is None:
        print("Generating synthetic market data...")
        data = generate_market_data(days=60, intraday_points=24, trend_cycles=3, volatility_cycles=2, seed=42)
    
    # Create strategy instance
    strategy = HeikinAshiReversal(
        fast_sma_period=9,
        slow_sma_period=30,
        doji_threshold=0.3,
        wick_threshold=0.3
    )
    
    # Run backtest
    print("Running backtest...")
    results = strategy.backtest(data)
    
    # Calculate performance metrics
    metrics = strategy.calculate_performance_metrics(results)
    
    # Print performance summary
    strategy.print_performance_summary(metrics)
    
    # Plot results
    strategy.plot_results(results)
    
    return strategy, results, metrics

def parameter_sensitivity_analysis(data=None):
    """
    Analyze the sensitivity of the strategy to different parameters
    
    Parameters:
    -----------
    data : DataFrame, optional
        Price data to use (if None, generates synthetic data)
        
    Returns:
    --------
    DataFrame
        Results of parameter sensitivity analysis
    """
    # Generate synthetic data if not provided
    if data is None:
        print("Generating synthetic market data for parameter analysis...")
        data = generate_market_data(days=60, intraday_points=24, trend_cycles=3, volatility_cycles=2, seed=42)
    
    # Parameters to test
    fast_sma_periods = [5, 9, 14]
    slow_sma_periods = [20, 30, 50]
    doji_thresholds = [0.2, 0.3, 0.4]
    wick_thresholds = [0.2, 0.3, 0.4]
    
    # Store results
    results = []
    
    # Test different parameter combinations
    for fast_period in fast_sma_periods:
        for slow_period in slow_sma_periods:
            if fast_period >= slow_period:
                continue  # Skip invalid combinations
                
            for doji_thresh in doji_thresholds:
                for wick_thresh in wick_thresholds:
                    print(f"Testing parameters: Fast SMA={fast_period}, Slow SMA={slow_period}, Doji Threshold={doji_thresh}, Wick Threshold={wick_thresh}")
                    
                    # Create strategy with current parameters
                    strategy = HeikinAshiReversal(
                        fast_sma_period=fast_period,
                        slow_sma_period=slow_period,
                        doji_threshold=doji_thresh,
                        wick_threshold=wick_thresh
                    )
                    
                    # Run backtest
                    results_df = strategy.backtest(data)
                    
                    # Calculate performance metrics
                    metrics = strategy.calculate_performance_metrics(results_df)
                    
                    # Store results
                    result = {
                        'fast_sma_period': fast_period,
                        'slow_sma_period': slow_period,
                        'doji_threshold': doji_thresh,
                        'wick_threshold': wick_thresh,
                        'total_return': metrics['total_return'],
                        'sharpe_ratio': metrics['sharpe_ratio'],
                        'max_drawdown': metrics['max_drawdown'],
                        'win_rate': metrics['win_rate'],
                        'profit_factor': metrics['profit_factor'],
                        'total_trades': metrics['total_trades']
                    }
                    
                    results.append(result)
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Sort by Sharpe ratio
    results_df = results_df.sort_values('sharpe_ratio', ascending=False)
    
    # Print top 5 parameter combinations
    print("\nTop 5 Parameter Combinations:")
    print(results_df.head(5))
    
    # Plot parameter impact
    plt.figure(figsize=(16, 12))
    
    # Plot impact of Fast SMA Period
    plt.subplot(2, 2, 1)
    sns.boxplot(x='fast_sma_period', y='sharpe_ratio', data=results_df)
    plt.title('Impact of Fast SMA Period on Sharpe Ratio')
    plt.grid(True, alpha=0.3)
    
    # Plot impact of Slow SMA Period
    plt.subplot(2, 2, 2)
    sns.boxplot(x='slow_sma_period', y='sharpe_ratio', data=results_df)
    plt.title('Impact of Slow SMA Period on Sharpe Ratio')
    plt.grid(True, alpha=0.3)
    
    # Plot impact of Doji Threshold
    plt.subplot(2, 2, 3)
    sns.boxplot(x='doji_threshold', y='sharpe_ratio', data=results_df)
    plt.title('Impact of Doji Threshold on Sharpe Ratio')
    plt.grid(True, alpha=0.3)
    
    # Plot impact of Wick Threshold
    plt.subplot(2, 2, 4)
    sns.boxplot(x='wick_threshold', y='sharpe_ratio', data=results_df)
    plt.title('Impact of Wick Threshold on Sharpe Ratio')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Plot trade-offs
    plt.figure(figsize=(16, 6))
    
    # Plot Win Rate vs. Total Trades
    plt.subplot(1, 2, 1)
    plt.scatter(results_df['total_trades'], results_df['win_rate'], 
               c=results_df['sharpe_ratio'], cmap='viridis', s=50, alpha=0.7)
    plt.colorbar(label='Sharpe Ratio')
    plt.title('Win Rate vs. Total Trades')
    plt.xlabel('Total Trades')
    plt.ylabel('Win Rate')
    plt.grid(True, alpha=0.3)
    
    # Plot Total Return vs. Max Drawdown
    plt.subplot(1, 2, 2)
    plt.scatter(abs(results_df['max_drawdown']), results_df['total_return'], 
               c=results_df['sharpe_ratio'], cmap='viridis', s=50, alpha=0.7)
    plt.colorbar(label='Sharpe Ratio')
    plt.title('Return vs. Risk')
    plt.xlabel('Max Drawdown (abs)')
    plt.ylabel('Total Return')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return results_df

def test_strategy_enhancements(data=None):
    """
    Test enhanced versions of the strategy
    
    Parameters:
    -----------
    data : DataFrame, optional
        Price data to use (if None, generates synthetic data)
        
    Returns:
    --------
    dict
        Results of enhanced strategy tests
    """
    # Generate synthetic data if not provided
    if data is None:
        print("Generating synthetic market data for enhancement tests...")
        data = generate_market_data(days=60, intraday_points=24, trend_cycles=3, volatility_cycles=2, seed=42)
    
    # Test original strategy
    print("\nTesting original strategy...")
    original_strategy = HeikinAshiReversal()
    original_results = original_strategy.backtest(data)
    original_metrics = original_strategy.calculate_performance_metrics(original_results)
    original_strategy.print_performance_summary(original_metrics)
    
    # Enhanced strategy with stop loss and take profit
    class EnhancedStrategyWithSLTP(HeikinAshiReversal):
        def __init__(self, stop_loss_pct=1.0, take_profit_pct=2.0, **kwargs):
            super().__init__(**kwargs)
            self.stop_loss_pct = stop_loss_pct
            self.take_profit_pct = take_profit_pct
        
        def backtest(self, data, initial_capital=10000, position_size_pct=100):
            # Calculate indicators
            df = self.calculate_indicators(data)
            
            # Initialize trading variables
            df['position'] = 0  # 1 for long, -1 for short, 0 for flat
            df['equity'] = initial_capital
            df['cash'] = initial_capital
            df['holdings'] = 0
            df['trade_pnl'] = 0
            df['stop_loss'] = np.nan
            df['take_profit'] = np.nan
            df['exit_reason'] = None
            
            # Process each bar
            position = 0
            entry_price = 0
            entry_index = 0
            stop_loss = 0
            take_profit = 0
            
            for i in range(max(self.fast_sma_period, self.slow_sma_period) + 1, len(df)):
                # Default is to carry forward previous values
                if i > 0:
                    df.loc[df.index[i], 'position'] = df.loc[df.index[i-1], 'position']
                    df.loc[df.index[i], 'equity'] = df.loc[df.index[i-1], 'equity']
                    df.loc[df.index[i], 'cash'] = df.loc[df.index[i-1], 'cash']
                    df.loc[df.index[i], 'holdings'] = df.loc[df.index[i-1], 'holdings']
                    df.loc[df.index[i], 'stop_loss'] = df.loc[df.index[i-1], 'stop_loss']
                    df.loc[df.index[i], 'take_profit'] = df.loc[df.index[i-1], 'take_profit']
                
                current_close = df.loc[df.index[i], 'close']
                current_high = df.loc[df.index[i], 'high']
                current_low = df.loc[df.index[i], 'low']
                
                # Check for stop loss or take profit hits
                if position == 1:
                    # Check for stop loss
                    if current_low <= stop_loss:
                        # Calculate profit/loss
                        trade_pnl = (stop_loss - entry_price) * df.loc[df.index[i], 'holdings'] / entry_price
                        df.loc[df.index[i], 'trade_pnl'] = trade_pnl
                        df.loc[df.index[i], 'equity'] += trade_pnl
                        df.loc[df.index[i], 'cash'] = df.loc[df.index[i], 'equity']
                        df.loc[df.index[i], 'holdings'] = 0
                        df.loc[df.index[i], 'position'] = 0
                        df.loc[df.index[i], 'stop_loss'] = np.nan
                        df.loc[df.index[i], 'take_profit'] = np.nan
                        df.loc[df.index[i], 'exit_reason'] = 'Stop Loss'
                        position = 0
                    
                    # Check for take profit
                    elif current_high >= take_profit:
                        # Calculate profit/loss
                        trade_pnl = (take_profit - entry_price) * df.loc[df.index[i], 'holdings'] / entry_price
                        df.loc[df.index[i], 'trade_pnl'] = trade_pnl
                        df.loc[df.index[i], 'equity'] += trade_pnl
                        df.loc[df.index[i], 'cash'] = df.loc[df.index[i], 'equity']
                        df.loc[df.index[i], 'holdings'] = 0
                        df.loc[df.index[i], 'position'] = 0
                        df.loc[df.index[i], 'stop_loss'] = np.nan
                        df.loc[df.index[i], 'take_profit'] = np.nan
                        df.loc[df.index[i], 'exit_reason'] = 'Take Profit'
                        position = 0
                
                elif position == -1:
                    # Check for stop loss
                    if current_high >= stop_loss:
                        # Calculate profit/loss
                        trade_pnl = (entry_price - stop_loss) * -df.loc[df.index[i], 'holdings'] / entry_price
                        df.loc[df.index[i], 'trade_pnl'] = trade_pnl
                        df.loc[df.index[i], 'equity'] += trade_pnl
                        df.loc[df.index[i], 'cash'] = df.loc[df.index[i], 'equity']
                        df.loc[df.index[i], 'holdings'] = 0
                        df.loc[df.index[i], 'position'] = 0
                        df.loc[df.index[i], 'stop_loss'] = np.nan
                        df.loc[df.index[i], 'take_profit'] = np.nan
                        df.loc[df.index[i], 'exit_reason'] = 'Stop Loss'
                        position = 0
                    
                    # Check for take profit
                    elif current_low <= take_profit:
                        # Calculate profit/loss
                        trade_pnl = (entry_price - take_profit) * -df.loc[df.index[i], 'holdings'] / entry_price
                        df.loc[df.index[i], 'trade_pnl'] = trade_pnl
                        df.loc[df.index[i], 'equity'] += trade_pnl
                        df.loc[df.index[i], 'cash'] = df.loc[df.index[i], 'equity']
                        df.loc[df.index[i], 'holdings'] = 0
                        df.loc[df.index[i], 'position'] = 0
                        df.loc[df.index[i], 'stop_loss'] = np.nan
                        df.loc[df.index[i], 'take_profit'] = np.nan
                        df.loc[df.index[i], 'exit_reason'] = 'Take Profit'
                        position = 0
                
                # Process long signal
                if df.loc[df.index[i], 'long_condition'] and position != 1:
                    # If in short position, close it first
                    if position == -1:
                        # Calculate profit/loss from short trade
                        trade_pnl = (entry_price - current_close) * -df.loc[df.index[i], 'holdings'] / entry_price
                        df.loc[df.index[i], 'trade_pnl'] += trade_pnl
                        df.loc[df.index[i], 'equity'] += trade_pnl
                        df.loc[df.index[i], 'cash'] = df.loc[df.index[i], 'equity']
                        df.loc[df.index[i], 'exit_reason'] = 'Signal Flip'
                    
                    # Enter long position
                    position = 1
                    entry_price = current_close
                    entry_index = i
                    
                    # Set stop loss and take profit
                    stop_loss = entry_price * (1 - self.stop_loss_pct / 100)
                    take_profit = entry_price * (1 + self.take_profit_pct / 100)
                    
                    # Calculate position size
                    position_size = df.loc[df.index[i], 'equity'] * (position_size_pct / 100) / current_close
                    
                    # Update tracking variables
                    df.loc[df.index[i], 'position'] = position
                    df.loc[df.index[i], 'holdings'] = position_size * current_close
                    df.loc[df.index[i], 'cash'] = df.loc[df.index[i], 'equity'] - df.loc[df.index[i], 'holdings']
                    df.loc[df.index[i], 'stop_loss'] = stop_loss
                    df.loc[df.index[i], 'take_profit'] = take_profit
                
                # Process short signal
                elif df.loc[df.index[i], 'short_condition'] and position != -1:
                    # If in long position, close it first
                    if position == 1:
                        # Calculate profit/loss from long trade
                        trade_pnl = (current_close - entry_price) * df.loc[df.index[i], 'holdings'] / entry_price
                        df.loc[df.index[i], 'trade_pnl'] += trade_pnl
                        df.loc[df.index[i], 'equity'] += trade_pnl
                        df.loc[df.index[i], 'cash'] = df.loc[df.index[i], 'equity']
                        df.loc[df.index[i], 'exit_reason'] = 'Signal Flip'
                    
                    # Enter short position
                    position = -1
                    entry_price = current_close
                    entry_index = i
                    
                    # Set stop loss and take profit
                    stop_loss = entry_price * (1 + self.stop_loss_pct / 100)
                    take_profit = entry_price * (1 - self.take_profit_pct / 100)
                    
                    # Calculate position size
                    position_size = df.loc[df.index[i], 'equity'] * (position_size_pct / 100) / current_close
                    
                    # Update tracking variables
                    df.loc[df.index[i], 'position'] = position
                    df.loc[df.index[i], 'holdings'] = -position_size * current_close
                    df.loc[df.index[i], 'cash'] = df.loc[df.index[i], 'equity'] - df.loc[df.index[i], 'holdings']
                    df.loc[df.index[i], 'stop_loss'] = stop_loss
                    df.loc[df.index[i], 'take_profit'] = take_profit
                
                # Update value of holdings
                if position == 1:
                    # Long position
                    df.loc[df.index[i], 'holdings'] = position_size * current_close
                    df.loc[df.index[i], 'equity'] = df.loc[df.index[i], 'cash'] + df.loc[df.index[i], 'holdings']
                elif position == -1:
                    # Short position
                    df.loc[df.index[i], 'holdings'] = -position_size * current_close
                    df.loc[df.index[i], 'equity'] = df.loc[df.index[i], 'cash'] - df.loc[df.index[i], 'holdings']
            
            # Calculate daily returns
            df['daily_returns'] = df['equity'].pct_change()
            
            # Calculate cumulative returns
            df['cumulative_returns'] = (1 + df['daily_returns']).cumprod() - 1
            
            return df
    
    # Enhanced strategy with volume filter
    class EnhancedStrategyWithVolumeFilter(HeikinAshiReversal):
        def __init__(self, volume_ma_period=20, volume_threshold=1.5, **kwargs):
            super().__init__(**kwargs)
            self.volume_ma_period = volume_ma_period
            self.volume_threshold = volume_threshold
        
        def calculate_indicators(self, data):
            # Call parent method to calculate base indicators
            df = super().calculate_indicators(data)
            
            # Add volume filter
            df['volume_ma'] = df['volume'].rolling(window=self.volume_ma_period).mean()
            df['volume_ratio'] = df['volume'] / df['volume_ma']
            df['volume_filter'] = df['volume_ratio'] > self.volume_threshold
            
            # Add volume filter to entry conditions
            df['long_condition'] = df['long_condition'] & df['volume_filter']
            df['short_condition'] = df['short_condition'] & df['volume_filter']
            
            return df
    
    # Enhanced strategy with RSI filter
    class EnhancedStrategyWithRSIFilter(HeikinAshiReversal):
        def __init__(self, rsi_period=14, rsi_overbought=70, rsi_oversold=30, **kwargs):
            super().__init__(**kwargs)
            self.rsi_period = rsi_period
            self.rsi_overbought = rsi_overbought
            self.rsi_oversold = rsi_oversold
        
        def calculate_indicators(self, data):
            # Call parent method to calculate base indicators
            df = super().calculate_indicators(data)
            
            # Calculate RSI
            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            
            avg_gain = gain.rolling(window=self.rsi_period).mean()
            avg_loss = loss.rolling(window=self.rsi_period).mean()
            
            rs = avg_gain / avg_loss.replace(0, 0.001)  # Avoid division by zero
            df['rsi_filter'] = 100 - (100 / (1 + rs))
            
            # Add RSI filter to entry conditions
            df['long_condition'] = df['long_condition'] & (df['rsi_filter'] < self.rsi_overbought)
            df['short_condition'] = df['short_condition'] & (df['rsi_filter'] > self.rsi_oversold)
            
            return df
    
    # Test enhanced strategies
    print("\nTesting strategy with stop loss and take profit...")
    sltp_strategy = EnhancedStrategyWithSLTP(stop_loss_pct=1.0, take_profit_pct=2.0)
    sltp_results = sltp_strategy.backtest(data)
    sltp_metrics = sltp_strategy.calculate_performance_metrics(sltp_results)
    sltp_strategy.print_performance_summary(sltp_metrics)
    
    print("\nTesting strategy with volume filter...")
    volume_strategy = EnhancedStrategyWithVolumeFilter(volume_ma_period=20, volume_threshold=1.5)
    volume_results = volume_strategy.backtest(data)
    volume_metrics = volume_strategy.calculate_performance_metrics(volume_results)
    volume_strategy.print_performance_summary(volume_metrics)
    
    print("\nTesting strategy with RSI filter...")
    rsi_strategy = EnhancedStrategyWithRSIFilter(rsi_period=14, rsi_overbought=70, rsi_oversold=30)
    rsi_results = rsi_strategy.backtest(data)
    rsi_metrics = rsi_strategy.calculate_performance_metrics(rsi_results)
    rsi_strategy.print_performance_summary(rsi_metrics)
    
    # Compare equity curves
    plt.figure(figsize=(15, 8))
    
    plt.plot(original_results['equity'], label='Original Strategy')
    plt.plot(sltp_results['equity'], label='With Stop Loss & Take Profit')
    plt.plot(volume_results['equity'], label='With Volume Filter')
    plt.plot(rsi_results['equity'], label='With RSI Filter')
    
    plt.title('Comparison of Strategy Enhancements')
    plt.xlabel('Date')
    plt.ylabel('Equity')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Compare key metrics
    comparison = pd.DataFrame({
        'Original': [
            original_metrics['total_return'],
            original_metrics['sharpe_ratio'],
            original_metrics['max_drawdown'],
            original_metrics['win_rate'],
            original_metrics['profit_factor'],
            original_metrics['total_trades']
        ],
        'With SL/TP': [
            sltp_metrics['total_return'],
            sltp_metrics['sharpe_ratio'],
            sltp_metrics['max_drawdown'],
            sltp_metrics['win_rate'],
            sltp_metrics['profit_factor'],
            sltp_metrics['total_trades']
        ],
        'With Volume Filter': [
            volume_metrics['total_return'],
            volume_metrics['sharpe_ratio'],
            volume_metrics['max_drawdown'],
            volume_metrics['win_rate'],
            volume_metrics['profit_factor'],
            volume_metrics['total_trades']
        ],
        'With RSI Filter': [
            rsi_metrics['total_return'],
            rsi_metrics['sharpe_ratio'],
            rsi_metrics['max_drawdown'],
            rsi_metrics['win_rate'],
            rsi_metrics['profit_factor'],
            rsi_metrics['total_trades']
        ]
    }, index=['Total Return', 'Sharpe Ratio', 'Max Drawdown', 'Win Rate', 'Profit Factor', 'Total Trades'])
    
    print("\nStrategy Comparison:")
    print(comparison)
    
    # Plot comparison metrics
    plt.figure(figsize=(15, 12))
    
    metrics_to_plot = ['Total Return', 'Sharpe Ratio', 'Max Drawdown', 'Win Rate', 'Profit Factor']
    for i, metric in enumerate(metrics_to_plot):
        plt.subplot(3, 2, i+1)
        
        # For drawdown, make it positive for easier comparison
        values = comparison.loc[metric] * (-1 if metric == 'Max Drawdown' else 1)
        
        plt.bar(comparison.columns, values)
        plt.title(metric)
        plt.grid(True, alpha=0.3)
        
        # Format y-axis as percentage for return metrics
        if metric in ['Total Return', 'Max Drawdown', 'Win Rate']:
            plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{abs(x):.1%}'))
    
    plt.tight_layout()
    plt.show()
    
    return {
        'original': {'strategy': original_strategy, 'results': original_results, 'metrics': original_metrics},
        'sltp': {'strategy': sltp_strategy, 'results': sltp_results, 'metrics': sltp_metrics},
        'volume': {'strategy': volume_strategy, 'results': volume_results, 'metrics': volume_metrics},
        'rsi': {'strategy': rsi_strategy, 'results': rsi_results, 'metrics': rsi_metrics}
    }

# Run the tests
if __name__ == "__main__":
    # Test basic strategy
    strategy, results, metrics = test_strategy()
    
    # Analyze parameter sensitivity
    # param_results = parameter_sensitivity_analysis()
    
    # Test strategy enhancements
    # enhancement_results = test_strategy_enhancements()