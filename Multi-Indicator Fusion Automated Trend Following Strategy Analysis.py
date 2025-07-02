import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

class MultiIndicatorFusionStrategy:
    """
    Implementation of the Multi-Indicator Fusion Automated Trend Following and Trap Avoidance Strategy
    
    This strategy combines multiple technical indicators including EMAs, SMAs, MACD, and ATR
    to identify strong trends while filtering out false signals and unfavorable market environments.
    """
    
    def __init__(self, 
                 ema_fast_len=8, 
                 ema_slow_len=34, 
                 ma50_len=50, 
                 ma200_len=200,
                 macd_fast=12,
                 macd_slow=26,
                 macd_signal=9,
                 atr_period=14,
                 atr_mult=1.5, 
                 risk_reward=2.0, 
                 sideways_threshold=0.2,
                 show_zones=True):
        """
        Initialize the strategy with parameters
        
        Parameters:
        -----------
        ema_fast_len : int
            Period for fast EMA (default: 8)
        ema_slow_len : int
            Period for slow EMA (default: 34)
        ma50_len : int
            Period for medium-term SMA (default: 50)
        ma200_len : int
            Period for long-term SMA (default: 200)
        macd_fast : int
            MACD fast period (default: 12)
        macd_slow : int
            MACD slow period (default: 26)
        macd_signal : int
            MACD signal period (default: 9)
        atr_period : int
            Period for ATR calculation (default: 14)
        atr_mult : float
            Multiplier for ATR-based stop loss (default: 1.5)
        risk_reward : float
            Risk-to-reward ratio for take profit calculation (default: 2.0)
        sideways_threshold : float
            Threshold for sideways market detection (default: 0.2)
        show_zones : bool
            Whether to highlight trap and sideways zones (default: True)
        """
        self.ema_fast_len = ema_fast_len
        self.ema_slow_len = ema_slow_len
        self.ma50_len = ma50_len
        self.ma200_len = ma200_len
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.atr_period = atr_period
        self.atr_mult = atr_mult
        self.risk_reward = risk_reward
        self.sideways_threshold = sideways_threshold
        self.show_zones = show_zones
    
    def calculate_ema(self, prices, period):
        """
        Calculate Exponential Moving Average
        
        Parameters:
        -----------
        prices : Series
            Price series to calculate EMA for
        period : int
            Period for EMA calculation
            
        Returns:
        --------
        Series
            EMA values
        """
        return prices.ewm(span=period, adjust=False).mean()
    
    def calculate_sma(self, prices, period):
        """
        Calculate Simple Moving Average
        
        Parameters:
        -----------
        prices : Series
            Price series to calculate SMA for
        period : int
            Period for SMA calculation
            
        Returns:
        --------
        Series
            SMA values
        """
        return prices.rolling(window=period).mean()
    
    def calculate_macd(self, prices):
        """
        Calculate MACD indicator
        
        Parameters:
        -----------
        prices : Series
            Price series to calculate MACD for
            
        Returns:
        --------
        tuple
            (MACD line, Signal line, Histogram)
        """
        ema_fast = self.calculate_ema(prices, self.macd_fast)
        ema_slow = self.calculate_ema(prices, self.macd_slow)
        macd_line = ema_fast - ema_slow
        signal_line = self.calculate_ema(macd_line, self.macd_signal)
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    
    def calculate_atr(self, high, low, close):
        """
        Calculate Average True Range
        
        Parameters:
        -----------
        high : Series
            High prices
        low : Series
            Low prices
        close : Series
            Close prices
            
        Returns:
        --------
        Series
            ATR values
        """
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
        atr = tr.rolling(window=self.atr_period).mean()
        return atr
    
    def calculate_indicators(self, data):
        """
        Calculate all indicators needed for the strategy
        
        Parameters:
        -----------
        data : DataFrame
            Price data with OHLC columns
            
        Returns:
        --------
        DataFrame
            Data with added indicator columns
        """
        df = data.copy()
        
        # Calculate EMAs
        df['ema_fast'] = self.calculate_ema(df['close'], self.ema_fast_len)
        df['ema_slow'] = self.calculate_ema(df['close'], self.ema_slow_len)
        
        # Calculate SMAs
        df['ma50'] = self.calculate_sma(df['close'], self.ma50_len)
        df['ma200'] = self.calculate_sma(df['close'], self.ma200_len)
        
        # Calculate MACD
        df['macd_line'], df['signal_line'], _ = self.calculate_macd(df['close'])
        
        # Calculate ATR
        df['atr'] = self.calculate_atr(df['high'], df['low'], df['close'])
        
        # Calculate highest high and lowest low for trap detection
        df['highest_20'] = df['high'].rolling(window=20).max()
        df['lowest_20'] = df['low'].rolling(window=20).min()
        
        # Detect fake breakouts and trap zones
        df['trap_long'] = (df['high'] > df['highest_20'].shift(1)) & (df['close'] < df['open'])
        df['trap_short'] = (df['low'] < df['lowest_20'].shift(1)) & (df['close'] > df['open'])
        
        # Calculate EMA slope for sideways detection
        df['ema_diff'] = df['ema_fast'] - df['ema_slow']
        df['ema_slope'] = df['ema_diff'].rolling(window=5).mean().abs()
        df['is_sideways'] = (df['ema_slope'] < self.sideways_threshold) & (df['macd_line'].abs() < 0.1)
        
        # Calculate trading conditions
        df['long_cond'] = (df['ema_fast'] > df['ema_slow']) & \
                          (df['close'] > df['ma50']) & \
                          (df['close'] > df['ma200']) & \
                          (df['macd_line'] > df['signal_line']) & \
                          (df['macd_line'] > 0)
        
        df['short_cond'] = (df['ema_fast'] < df['ema_slow']) & \
                           (df['close'] < df['ma50']) & \
                           (df['close'] < df['ma200']) & \
                           (df['macd_line'] < df['signal_line']) & \
                           (df['macd_line'] < 0)
        
        # Final entry conditions
        df['can_long'] = df['long_cond'] & ~df['is_sideways'] & ~df['trap_long']
        df['can_short'] = df['short_cond'] & ~df['is_sideways'] & ~df['trap_short']
        
        # Calculate stop loss and take profit levels
        df['long_sl'] = df['close'] - df['atr'] * self.atr_mult
        df['long_tp'] = df['close'] + df['atr'] * self.atr_mult * self.risk_reward
        df['short_sl'] = df['close'] + df['atr'] * self.atr_mult
        df['short_tp'] = df['close'] - df['atr'] * self.atr_mult * self.risk_reward
        
        return df
    
    def backtest(self, data, initial_capital=10000):
        """
        Run backtest on the provided data
        
        Parameters:
        -----------
        data : DataFrame
            Price data with OHLC columns
        initial_capital : float
            Initial capital for backtest
            
        Returns:
        --------
        DataFrame
            Data with added backtest result columns
        """
        # Calculate indicators
        df = self.calculate_indicators(data)
        
        # Initialize trading variables
        df['position'] = 0  # 1 for long, -1 for short, 0 for flat
        df['entry_price'] = np.nan
        df['stop_loss'] = np.nan
        df['take_profit'] = np.nan
        df['equity'] = initial_capital
        df['trade_pnl'] = 0.0
        df['trade_result'] = None
        
        # Process each bar
        position = 0
        entry_price = 0
        stop_loss = 0
        take_profit = 0
        
        for i in range(max(self.ema_fast_len, self.ema_slow_len, self.ma50_len, self.ma200_len, self.macd_slow) + 1, len(df)):
            # Default is to carry forward previous values
            if i > 0:
                df.iloc[i, df.columns.get_indexer(['position', 'entry_price', 'stop_loss', 'take_profit', 'equity'])] = df.iloc[i-1, df.columns.get_indexer(['position', 'entry_price', 'stop_loss', 'take_profit', 'equity'])]
            
            current_high = df['high'].iloc[i]
            current_low = df['low'].iloc[i]
            current_close = df['close'].iloc[i]
            
            # Check for exit if in position
            if position == 1:  # Long position
                # Check if stop loss hit
                if current_low <= stop_loss:
                    # Calculate P&L
                    trade_pnl = (stop_loss - entry_price) * (initial_capital / entry_price)
                    df['trade_pnl'].iloc[i] = trade_pnl
                    df['equity'].iloc[i] += trade_pnl
                    df['trade_result'].iloc[i] = 'Stop Loss'
                    
                    # Reset position
                    position = 0
                    df['position'].iloc[i] = 0
                    df['entry_price'].iloc[i] = np.nan
                    df['stop_loss'].iloc[i] = np.nan
                    df['take_profit'].iloc[i] = np.nan
                
                # Check if take profit hit
                elif current_high >= take_profit:
                    # Calculate P&L
                    trade_pnl = (take_profit - entry_price) * (initial_capital / entry_price)
                    df['trade_pnl'].iloc[i] = trade_pnl
                    df['equity'].iloc[i] += trade_pnl
                    df['trade_result'].iloc[i] = 'Take Profit'
                    
                    # Reset position
                    position = 0
                    df['position'].iloc[i] = 0
                    df['entry_price'].iloc[i] = np.nan
                    df['stop_loss'].iloc[i] = np.nan
                    df['take_profit'].iloc[i] = np.nan
            
            elif position == -1:  # Short position
                # Check if stop loss hit
                if current_high >= stop_loss:
                    # Calculate P&L
                    trade_pnl = (entry_price - stop_loss) * (initial_capital / entry_price)
                    df['trade_pnl'].iloc[i] = trade_pnl
                    df['equity'].iloc[i] += trade_pnl
                    df['trade_result'].iloc[i] = 'Stop Loss'
                    
                    # Reset position
                    position = 0
                    df['position'].iloc[i] = 0
                    df['entry_price'].iloc[i] = np.nan
                    df['stop_loss'].iloc[i] = np.nan
                    df['take_profit'].iloc[i] = np.nan
                
                # Check if take profit hit
                elif current_low <= take_profit:
                    # Calculate P&L
                    trade_pnl = (entry_price - take_profit) * (initial_capital / entry_price)
                    df['trade_pnl'].iloc[i] = trade_pnl
                    df['equity'].iloc[i] += trade_pnl
                    df['trade_result'].iloc[i] = 'Take Profit'
                    
                    # Reset position
                    position = 0
                    df['position'].iloc[i] = 0
                    df['entry_price'].iloc[i] = np.nan
                    df['stop_loss'].iloc[i] = np.nan
                    df['take_profit'].iloc[i] = np.nan
            
            # Check for entry if not in position
            if position == 0:
                if df['can_long'].iloc[i]:
                    # Enter long position
                    position = 1
                    entry_price = current_close
                    stop_loss = df['long_sl'].iloc[i]
                    take_profit = df['long_tp'].iloc[i]
                    
                    df['position'].iloc[i] = position
                    df['entry_price'].iloc[i] = entry_price
                    df['stop_loss'].iloc[i] = stop_loss
                    df['take_profit'].iloc[i] = take_profit
                
                elif df['can_short'].iloc[i]:
                    # Enter short position
                    position = -1
                    entry_price = current_close
                    stop_loss = df['short_sl'].iloc[i]
                    take_profit = df['short_tp'].iloc[i]
                    
                    df['position'].iloc[i] = position
                    df['entry_price'].iloc[i] = entry_price
                    df['stop_loss'].iloc[i] = stop_loss
                    df['take_profit'].iloc[i] = take_profit
        
        # Calculate daily returns
        df['daily_returns'] = df['equity'].pct_change()
        
        # Calculate cumulative returns
        df['cumulative_returns'] = (1 + df['daily_returns']).cumprod() - 1
        
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
        trades = results[results['trade_result'].notna()]
        
        if len(trades) == 0:
            return {
                'total_return': 0,
                'annual_return': 0,
                'annual_volatility': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'total_trades': 0,
                'long_trades': 0,
                'short_trades': 0,
                'long_win_rate': 0,
                'short_win_rate': 0,
                'trade_results': {}
            }
        
        # Calculate metrics
        total_return = (results['equity'].iloc[-1] / results['equity'].iloc[0]) - 1
        
        # Calculate annual return (assuming 252 trading days per year)
        days = (results.index[-1] - results.index[0]).days if isinstance(results.index, pd.DatetimeIndex) else len(results)
        years = max(days / 252, 0.01)  # Avoid division by zero
        annual_return = (1 + total_return) ** (1 / years) - 1
        
        # Calculate volatility
        daily_returns = results['daily_returns'].dropna()
        annual_volatility = daily_returns.std() * np.sqrt(252)
        
        # Calculate Sharpe ratio (assuming risk-free rate = 0)
        sharpe_ratio = annual_return / annual_volatility if annual_volatility != 0 else 0
        
        # Calculate max drawdown
        equity_curve = results['equity']
        peak = equity_curve.cummax()
        drawdown = (equity_curve - peak) / peak
        max_drawdown = drawdown.min()
        
        # Calculate win rate
        winning_trades = trades[trades['trade_pnl'] > 0]
        win_rate = len(winning_trades) / len(trades)
        
        # Calculate profit factor
        gross_profit = winning_trades['trade_pnl'].sum()
        losing_trades = trades[trades['trade_pnl'] <= 0]
        gross_loss = abs(losing_trades['trade_pnl'].sum()) if len(losing_trades) > 0 else 1
        profit_factor = gross_profit / gross_loss
        
        # Calculate average win and loss
        avg_win = winning_trades['trade_pnl'].mean() if len(winning_trades) > 0 else 0
        avg_loss = losing_trades['trade_pnl'].mean() if len(losing_trades) > 0 else 0
        
        # Calculate metrics by position type
        long_trades = trades[trades['position'] > 0]
        short_trades = trades[trades['position'] < 0]
        
        long_wins = long_trades[long_trades['trade_pnl'] > 0]
        short_wins = short_trades[short_trades['trade_pnl'] > 0]
        
        long_win_rate = len(long_wins) / len(long_trades) if len(long_trades) > 0 else 0
        short_win_rate = len(short_wins) / len(short_trades) if len(short_trades) > 0 else 0
        
        # Calculate trade outcomes
        trade_results = trades['trade_result'].value_counts().to_dict()
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'annual_volatility': annual_volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'total_trades': len(trades),
            'long_trades': len(long_trades),
            'short_trades': len(short_trades),
            'long_win_rate': long_win_rate,
            'short_win_rate': short_win_rate,
            'trade_results': trade_results
        }
    
    def plot_results(self, results):
        """
        Plot backtest results
        
        Parameters:
        -----------
        results : DataFrame
            Backtest results
        """
        plt.figure(figsize=(16, 20))
        
        # Plot 1: Price chart with indicators
        ax1 = plt.subplot(4, 1, 1)
        
        # Plot price
        ax1.plot(results.index, results['close'], label='Close', color='black', linewidth=1)
        
        # Plot moving averages
        ax1.plot(results.index, results['ema_fast'], label=f'Fast EMA ({self.ema_fast_len})', color='orange', linewidth=1)
        ax1.plot(results.index, results['ema_slow'], label=f'Slow EMA ({self.ema_slow_len})', color='teal', linewidth=1)
        ax1.plot(results.index, results['ma50'], label=f'50 MA', color='blue', linewidth=1)
        ax1.plot(results.index, results['ma200'], label=f'200 MA', color='purple', linewidth=1)
        
        # Mark trap and sideways zones if enabled
        if self.show_zones:
            for i, row in results.iterrows():
                if row['is_sideways']:
                    ax1.axvspan(i, i, color='orange', alpha=0.2)
                if row['trap_long'] or row['trap_short']:
                    ax1.axvspan(i, i, color='red', alpha=0.2)
        
        # Mark entry and exit points
        long_entries = results[results['position'].diff() == 1]
        short_entries = results[results['position'].diff() == -1]
        exits = results[(results['position'].shift() != 0) & (results['position'] == 0)]
        
        ax1.scatter(long_entries.index, long_entries['close'], marker='^', color='green', s=100, label='Long Entry')
        ax1.scatter(short_entries.index, short_entries['close'], marker='v', color='red', s=100, label='Short Entry')
        ax1.scatter(exits.index, exits['close'], marker='x', color='black', s=100, label='Exit')
        
        # Plot stop loss and take profit levels
        for i, row in results.iterrows():
            if row['position'] == 1:  # Long position
                ax1.plot([i, i], [row['entry_price'], row['stop_loss']], 'r--', alpha=0.5)
                ax1.plot([i, i], [row['entry_price'], row['take_profit']], 'g--', alpha=0.5)
            elif row['position'] == -1:  # Short position
                ax1.plot([i, i], [row['entry_price'], row['stop_loss']], 'r--', alpha=0.5)
                ax1.plot([i, i], [row['entry_price'], row['take_profit']], 'g--', alpha=0.5)
        
        ax1.set_title('Price Chart with Indicators')
        ax1.set_ylabel('Price')
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: MACD
        ax2 = plt.subplot(4, 1, 2)
        
        ax2.plot(results.index, results['macd_line'], label='MACD Line', color='blue')
        ax2.plot(results.index, results['signal_line'], label='Signal Line', color='red')
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # Fill MACD histogram
        for i in range(1, len(results)):
            if results['macd_line'].iloc[i] >= results['signal_line'].iloc[i]:
                ax2.fill_between([results.index[i-1], results.index[i]], 
                                [results['macd_line'].iloc[i-1], results['macd_line'].iloc[i]], 
                                [results['signal_line'].iloc[i-1], results['signal_line'].iloc[i]], 
                                color='green', alpha=0.3)
            else:
                ax2.fill_between([results.index[i-1], results.index[i]], 
                                [results['macd_line'].iloc[i-1], results['macd_line'].iloc[i]], 
                                [results['signal_line'].iloc[i-1], results['signal_line'].iloc[i]], 
                                color='red', alpha=0.3)
        
        ax2.set_title('MACD')
        ax2.set_ylabel('Value')
        ax2.legend(loc='upper left')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Position
        ax3 = plt.subplot(4, 1, 3)
        
        ax3.plot(results.index, results['position'], label='Position', color='blue')
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # Fill areas for long and short positions
        ax3.fill_between(results.index, 0, results['position'], where=results['position'] > 0, 
                        color='green', alpha=0.3, label='Long')
        ax3.fill_between(results.index, 0, results['position'], where=results['position'] < 0, 
                        color='red', alpha=0.3, label='Short')
        
        ax3.set_title('Position (1: Long, -1: Short, 0: Flat)')
        ax3.set_ylabel('Position')
        ax3.set_yticks([-1, 0, 1])
        ax3.legend(loc='upper left')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Equity Curve
        ax4 = plt.subplot(4, 1, 4)
        
        ax4.plot(results.index, results['equity'], label='Strategy Equity', color='blue')
        
        # Mark trade results on equity curve
        winning_trades = results[(results['trade_pnl'] > 0) & (results['trade_pnl'].notna())]
        losing_trades = results[(results['trade_pnl'] <= 0) & (results['trade_pnl'].notna())]
        
        ax4.scatter(winning_trades.index, winning_trades['equity'], marker='o', color='green', s=50, label='Win')
        ax4.scatter(losing_trades.index, losing_trades['equity'], marker='o', color='red', s=50, label='Loss')
        
        # Plot buy & hold equity for comparison
        initial_capital = results['equity'].iloc[0]
        buy_hold_equity = initial_capital * (results['close'] / results['close'].iloc[0])
        ax4.plot(results.index, buy_hold_equity, label='Buy & Hold', color='gray', alpha=0.5)
        
        ax4.set_title('Equity Curve')
        ax4.set_ylabel('Equity')
        ax4.set_xlabel('Date')
        ax4.legend(loc='upper left')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Additional plot: Drawdown
        plt.figure(figsize=(16, 6))
        
        equity_curve = results['equity']
        peak = equity_curve.cummax()
        drawdown = ((equity_curve - peak) / peak) * 100
        
        plt.plot(results.index, drawdown, color='red')
        plt.fill_between(results.index, 0, drawdown, color='red', alpha=0.3)
        
        plt.title('Drawdown (%)')
        plt.ylabel('Drawdown (%)')
        plt.xlabel('Date')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Additional plot: Trade analysis
        if results['trade_result'].notna().sum() > 0:
            plt.figure(figsize=(16, 12))
            
            # Plot 1: Trade Results
            ax1 = plt.subplot(2, 2, 1)
            trade_results = results['trade_result'].value_counts()
            trade_results.plot(kind='bar', ax=ax1)
            ax1.set_title('Trade Outcomes')
            ax1.set_ylabel('Count')
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Trade P&L Distribution
            ax2 = plt.subplot(2, 2, 2)
            trades = results[results['trade_pnl'] != 0]
            sns.histplot(trades['trade_pnl'], bins=20, kde=True, ax=ax2)
            ax2.axvline(x=0, color='black', linestyle='--', alpha=0.7)
            ax2.set_title('Trade P&L Distribution')
            ax2.set_xlabel('P&L')
            ax2.grid(True, alpha=0.3)
            
            # Plot 3: Win/Loss by Position Type
            ax3 = plt.subplot(2, 2, 3)
            position_types = []
            for i, row in trades.iterrows():
                if row['position'] == 1:
                    position_types.append('Long')
                elif row['position'] == -1:
                    position_types.append('Short')
                else:
                    position_types.append('Unknown')
            
            trades['position_type'] = position_types
            trades['win'] = trades['trade_pnl'] > 0
            
            win_loss_by_type = pd.crosstab(trades['position_type'], trades['win'])
            win_loss_by_type.plot(kind='bar', stacked=True, ax=ax3, color=['red', 'green'])
            ax3.set_title('Win/Loss by Position Type')
            ax3.set_ylabel('Count')
            ax3.grid(True, alpha=0.3)
            
            # Plot 4: Cumulative P&L
            ax4 = plt.subplot(2, 2, 4)
            trades['cumulative_pnl'] = trades['trade_pnl'].cumsum()
            ax4.plot(trades.index, trades['cumulative_pnl'], color='blue')
            ax4.set_title('Cumulative P&L')
            ax4.set_ylabel('P&L')
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
        
        print(f"Total Return: {metrics['total_return']:.2%}")
        print(f"Annual Return: {metrics['annual_return']:.2%}")
        print(f"Annual Volatility: {metrics['annual_volatility']:.2%}")
        print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        print(f"Maximum Drawdown: {metrics['max_drawdown']:.2%}")
        print(f"Win Rate: {metrics['win_rate']:.2%}")
        print(f"Profit Factor: {metrics['profit_factor']:.2f}")
        print(f"Average Win: ${metrics['avg_win']:.2f}")
        print(f"Average Loss: ${metrics['avg_loss']:.2f}")
        print(f"Total Trades: {metrics['total_trades']}")
        print(f"  Long Trades: {metrics['long_trades']} (Win Rate: {metrics['long_win_rate']:.2%})")
        print(f"  Short Trades: {metrics['short_trades']} (Win Rate: {metrics['short_win_rate']:.2%})")
        
        print("\nTrade Outcomes:")
        for outcome, count in metrics['trade_results'].items():
            print(f"  {outcome}: {count}")
        
        print("=" * 50)

def generate_market_data(days=365, seed=42):
    """
    Generate synthetic market data with trends, volatility regimes, and market traps
    
    Parameters:
    -----------
    days : int
        Number of days to generate
    seed : int
        Random seed for reproducibility
        
    Returns:
    --------
    DataFrame
        Synthetic market data with OHLC columns
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Generate dates
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    dates = pd.date_range(start=start_date, end=end_date, periods=days)
    
    # Initialize with a base price
    base_price = 100.0
    
    # Generate price series with trends, volatility regimes, and traps
    prices = [base_price]
    
    # Parameters
    trend_cycles = 5  # Number of major trend cycles
    trend_period = days // trend_cycles
    vol_regime_cycles = 10  # Number of volatility regime cycles
    vol_regime_period = days // vol_regime_cycles
    sideways_probability = 0.2  # Probability of generating a sideways period
    trap_probability = 0.05  # Probability of generating a trap
    
    # Initialize state
    current_state = "uptrend"  # Start with an uptrend
    sideways_counter = 0  # Counter for sideways periods
    trap_counter = 0  # Counter for trap periods
    
    # Generate price movement
    for i in range(1, days):
        # Calculate trend component
        trend_phase = i % trend_period
        trend_progress = trend_phase / trend_period
        
        # Determine current state
        if sideways_counter > 0:
            current_state = "sideways"
            sideways_counter -= 1
        elif trap_counter > 0:
            current_state = "trap"
            trap_counter -= 1
        elif trend_progress < 0.45:
            current_state = "uptrend"
        elif trend_progress < 0.55:
            # Transition zone - potential for sideways or trap
            if np.random.random() < sideways_probability:
                current_state = "sideways"
                sideways_counter = int(trend_period * 0.1)  # Sideways for 10% of a trend cycle
            elif np.random.random() < trap_probability:
                current_state = "trap"
                trap_counter = 3  # Trap lasts for 3 days
            else:
                current_state = "uptrend"
        else:
            current_state = "downtrend"
        
        # Calculate volatility component
        vol_regime_phase = i % vol_regime_period
        vol_regime_progress = vol_regime_phase / vol_regime_period
        vol_regime = 0.5 + 0.5 * np.sin(2 * np.pi * vol_regime_progress)
        
        # Adjust volatility based on state
        if current_state == "sideways":
            volatility = 0.002 + 0.002 * vol_regime
            trend_component = 0.0
        elif current_state == "trap":
            volatility = 0.01 + 0.01 * vol_regime
            # For traps, we'll reverse the trend temporarily
            if trend_progress < 0.5:
                trend_component = -0.005  # False breakout down during uptrend
            else:
                trend_component = 0.005  # False breakout up during downtrend
        elif current_state == "uptrend":
            volatility = 0.005 + 0.005 * vol_regime
            trend_component = 0.001
        else:  # downtrend
            volatility = 0.005 + 0.005 * vol_regime
            trend_component = -0.001
        
        # Calculate price movement
        price_change = trend_component + np.random.normal(0, volatility)
        
        # Calculate new price
        new_price = prices[-1] * (1 + price_change)
        prices.append(new_price)
    
    # Generate OHLC data
    ohlc_data = pd.DataFrame(index=dates)
    ohlc_data['close'] = prices
    
    # Generate open, high, low
    ohlc_data['open'] = np.zeros(days)
    ohlc_data['high'] = np.zeros(days)
    ohlc_data['low'] = np.zeros(days)
    ohlc_data['volume'] = np.zeros(days)
    
    # First day
    ohlc_data['open'].iloc[0] = prices[0] * (1 - 0.005 * np.random.random())
    intraday_vol = prices[0] * 0.01
    ohlc_data['high'].iloc[0] = max(ohlc_data['open'].iloc[0], ohlc_data['close'].iloc[0]) + intraday_vol * np.random.random()
    ohlc_data['low'].iloc[0] = min(ohlc_data['open'].iloc[0], ohlc_data['close'].iloc[0]) - intraday_vol * np.random.random()
    ohlc_data['volume'].iloc[0] = 1000000 * (0.5 + np.random.random())
    
    # Remaining days
    for i in range(1, days):
        # Open is close of previous day with small gap
        gap = np.random.normal(0, 0.002)
        ohlc_data['open'].iloc[i] = ohlc_data['close'].iloc[i-1] * (1 + gap)
        
        # Calculate volatility for the day
        vol_regime_phase = i % vol_regime_period
        vol_regime_progress = vol_regime_phase / vol_regime_period
        vol_regime = 0.5 + 0.5 * np.sin(2 * np.pi * vol_regime_progress)
        intraday_vol = ohlc_data['close'].iloc[i] * (0.005 + 0.01 * vol_regime)
        
        # Generate high and low
        if ohlc_data['open'].iloc[i] <= ohlc_data['close'].iloc[i]:  # Up day
            ohlc_data['high'].iloc[i] = ohlc_data['close'].iloc[i] + intraday_vol * np.random.random()
            ohlc_data['low'].iloc[i] = ohlc_data['open'].iloc[i] - intraday_vol * np.random.random()
        else:  # Down day
            ohlc_data['high'].iloc[i] = ohlc_data['open'].iloc[i] + intraday_vol * np.random.random()
            ohlc_data['low'].iloc[i] = ohlc_data['close'].iloc[i] - intraday_vol * np.random.random()
        
        # Generate volume - higher on trend days, lower on sideways
        base_volume = 1000000 * (0.5 + np.random.random())
        if current_state == "uptrend" or current_state == "downtrend":
            volume_multiplier = 1.5
        elif current_state == "trap":
            volume_multiplier = 2.0  # Higher volume on trap days
        else:  # sideways
            volume_multiplier = 0.7  # Lower volume on sideways days
        
        ohlc_data['volume'].iloc[i] = base_volume * volume_multiplier
    
    # Ensure high/low are actually high/low
    ohlc_data['high'] = np.maximum.reduce([ohlc_data['high'], ohlc_data['open'], ohlc_data['close']])
    ohlc_data['low'] = np.minimum.reduce([ohlc_data['low'], ohlc_data['open'], ohlc_data['close']])
    
    return ohlc_data

def test_strategy():
    """
    Test the Multi-Indicator Fusion strategy with default parameters
    
    Returns:
    --------
    tuple
        (strategy, results, metrics)
    """
    # Generate synthetic data
    print("Generating synthetic market data...")
    data = generate_market_data(days=365, seed=42)
    
    # Create strategy instance
    strategy = MultiIndicatorFusionStrategy(
        ema_fast_len=8,
        ema_slow_len=34,
        ma50_len=50,
        ma200_len=200,
        macd_fast=12,
        macd_slow=26,
        macd_signal=9,
        atr_period=14,
        atr_mult=1.5,
        risk_reward=2.0,
        sideways_threshold=0.2,
        show_zones=True
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

def parameter_optimization():
    """
    Optimize strategy parameters
    
    Returns:
    --------
    DataFrame
        Results of parameter optimization
    """
    # Generate synthetic data
    print("Generating synthetic market data for optimization...")
    data = generate_market_data(days=365, seed=42)
    
    # Parameters to test
    ema_fast_lengths = [5, 8, 13]
    ema_slow_lengths = [21, 34, 55]
    atr_mults = [1.0, 1.5, 2.0]
    risk_rewards = [1.5, 2.0, 2.5]
    sideways_thresholds = [0.1, 0.2, 0.3]
    
    # Store results
    results = []
    
    # Test parameter combinations
    total_combinations = len(ema_fast_lengths) * len(ema_slow_lengths) * len(atr_mults) * len(risk_rewards) * len(sideways_thresholds)
    count = 0
    
    for ema_fast in ema_fast_lengths:
        for ema_slow in ema_slow_lengths:
            for atr_mult in atr_mults:
                for risk_reward in risk_rewards:
                    for sideways_threshold in sideways_thresholds:
                        count += 1
                        print(f"Testing parameter combination {count}/{total_combinations}...")
                        
                        # Create strategy instance
                        strategy = MultiIndicatorFusionStrategy(
                            ema_fast_len=ema_fast,
                            ema_slow_len=ema_slow,
                            atr_mult=atr_mult,
                            risk_reward=risk_reward,
                            sideways_threshold=sideways_threshold
                        )
                        
                        # Run backtest
                        backtest_results = strategy.backtest(data)
                        
                        # Calculate performance metrics
                        metrics = strategy.calculate_performance_metrics(backtest_results)
                        
                        # Store results
                        result = {
                            'ema_fast_len': ema_fast,
                            'ema_slow_len': ema_slow,
                            'atr_mult': atr_mult,
                            'risk_reward': risk_reward,
                            'sideways_threshold': sideways_threshold,
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
    plt.figure(figsize=(15, 15))
    
    # Plot impact of EMA fast length
    plt.subplot(3, 2, 1)
    sns.boxplot(x='ema_fast_len', y='sharpe_ratio', data=results_df)
    plt.title('Impact of Fast EMA Length on Sharpe Ratio')
    plt.grid(True, alpha=0.3)
    
    # Plot impact of EMA slow length
    plt.subplot(3, 2, 2)
    sns.boxplot(x='ema_slow_len', y='sharpe_ratio', data=results_df)
    plt.title('Impact of Slow EMA Length on Sharpe Ratio')
    plt.grid(True, alpha=0.3)
    
    # Plot impact of ATR multiplier
    plt.subplot(3, 2, 3)
    sns.boxplot(x='atr_mult', y='sharpe_ratio', data=results_df)
    plt.title('Impact of ATR Multiplier on Sharpe Ratio')
    plt.grid(True, alpha=0.3)
    
    # Plot impact of Risk/Reward ratio
    plt.subplot(3, 2, 4)
    sns.boxplot(x='risk_reward', y='sharpe_ratio', data=results_df)
    plt.title('Impact of Risk/Reward Ratio on Sharpe Ratio')
    plt.grid(True, alpha=0.3)
    
    # Plot impact of Sideways threshold
    plt.subplot(3, 2, 5)
    sns.boxplot(x='sideways_threshold', y='sharpe_ratio', data=results_df)
    plt.title('Impact of Sideways Threshold on Sharpe Ratio')
    plt.grid(True, alpha=0.3)
    
    # Plot return vs drawdown
    plt.subplot(3, 2, 6)
    plt.scatter(results_df['max_drawdown'], results_df['total_return'], 
               c=results_df['sharpe_ratio'], cmap='viridis', alpha=0.7, s=50)
    plt.colorbar(label='Sharpe Ratio')
    plt.title('Return vs Drawdown')
    plt.xlabel('Maximum Drawdown')
    plt.ylabel('Total Return')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return results_df

def test_enhanced_strategies():
    """
    Test enhanced versions of the strategy
    
    Returns:
    --------
    dict
        Results of enhanced strategy tests
    """
    # Generate synthetic data
    print("Generating synthetic market data for enhancement tests...")
    data = generate_market_data(days=365, seed=42)
    
    # Test original strategy
    print("\nTesting original strategy...")
    original_strategy = MultiIndicatorFusionStrategy()
    original_results = original_strategy.backtest(data)
    original_metrics = original_strategy.calculate_performance_metrics(original_results)
    original_strategy.print_performance_summary(original_metrics)
    
    # Enhanced strategy with volume confirmation
    class VolumeConfirmationStrategy(MultiIndicatorFusionStrategy):
        def __init__(self, volume_ma_period=20, volume_threshold=1.5, **kwargs):
            super().__init__(**kwargs)
            self.volume_ma_period = volume_ma_period
            self.volume_threshold = volume_threshold
        
        def calculate_indicators(self, data):
            # Call parent method to calculate base indicators
            df = super().calculate_indicators(data)
            
            # Calculate volume moving average
            df['volume_ma'] = df['volume'].rolling(window=self.volume_ma_period).mean()
            
            # Calculate volume ratio
            df['volume_ratio'] = df['volume'] / df['volume_ma']
            
            # Add volume confirmation to entry conditions
            df['volume_confirmed'] = df['volume_ratio'] > self.volume_threshold
            
            # Update entry conditions to require volume confirmation
            df['can_long'] = df['can_long'] & df['volume_confirmed']
            df['can_short'] = df['can_short'] & df['volume_confirmed']
            
            return df
    
    # Enhanced strategy with dynamic risk management
    class DynamicRiskStrategy(MultiIndicatorFusionStrategy):
        def __init__(self, atr_period_short=7, atr_period_long=21, **kwargs):
            super().__init__(**kwargs)
            self.atr_period_short = atr_period_short
            self.atr_period_long = atr_period_long
        
        def calculate_indicators(self, data):
            # Call parent method to calculate base indicators
            df = super().calculate_indicators(data)
            
            # Calculate short-term and long-term ATR
            high_low = data['high'] - data['low']
            high_close = abs(data['high'] - data['close'].shift(1))
            low_close = abs(data['low'] - data['close'].shift(1))
            tr = pd.DataFrame({'tr1': high_low, 'tr2': high_close, 'tr3': low_close}).max(axis=1)
            
            df['atr_short'] = tr.rolling(window=self.atr_period_short).mean()
            df['atr_long'] = tr.rolling(window=self.atr_period_long).mean()
            
            # Calculate ATR ratio
            df['atr_ratio'] = df['atr_short'] / df['atr_long']
            
            # Adjust ATR multiplier based on volatility
            df['dynamic_atr_mult'] = np.where(
                df['atr_ratio'] > 1.2,  # Higher short-term volatility
                self.atr_mult * 1.5,    # Use wider stops in volatile markets
                np.where(
                    df['atr_ratio'] < 0.8,  # Lower short-term volatility
                    self.atr_mult * 0.8,    # Use tighter stops in calm markets
                    self.atr_mult            # Use default multiplier otherwise
                )
            )
            
            # Recalculate stop loss and take profit levels with dynamic multiplier
            df['long_sl'] = df['close'] - df['atr'] * df['dynamic_atr_mult']
            df['long_tp'] = df['close'] + df['atr'] * df['dynamic_atr_mult'] * self.risk_reward
            df['short_sl'] = df['close'] + df['atr'] * df['dynamic_atr_mult']
            df['short_tp'] = df['close'] - df['atr'] * df['dynamic_atr_mult'] * self.risk_reward
            
            return df
    
    # Enhanced strategy with partial profit taking
    class PartialProfitStrategy(MultiIndicatorFusionStrategy):
        def __init__(self, profit_levels=[0.33, 0.67, 1.0], profit_portions=[0.3, 0.3, 0.4], **kwargs):
            super().__init__(**kwargs)
            self.profit_levels = profit_levels
            self.profit_portions = profit_portions
        
        def backtest(self, data, initial_capital=10000):
            # Calculate indicators
            df = self.calculate_indicators(data)
            
            # Initialize trading variables
            df['position'] = 0  # Can be decimal for partial positions
            df['entry_price'] = np.nan
            df['stop_loss'] = np.nan
            df['take_profit_1'] = np.nan
            df['take_profit_2'] = np.nan
            df['take_profit_3'] = np.nan
            df['equity'] = initial_capital
            df['trade_pnl'] = 0.0
            df['trade_result'] = None
            
            # Process each bar
            position = 0
            remaining_position = 0
            entry_price = 0
            stop_loss = 0
            take_profit_levels = [0, 0, 0]
            position_portions = [0, 0, 0]  # Track portions of position for each level
            
            for i in range(max(self.ema_fast_len, self.ema_slow_len, self.ma50_len, self.ma200_len, self.macd_slow) + 1, len(df)):
                # Default is to carry forward previous values
                if i > 0:
                    df.iloc[i, df.columns.get_indexer(['position', 'entry_price', 'stop_loss', 'take_profit_1', 'take_profit_2', 'take_profit_3', 'equity'])] = df.iloc[i-1, df.columns.get_indexer(['position', 'entry_price', 'stop_loss', 'take_profit_1', 'take_profit_2', 'take_profit_3', 'equity'])]
                
                current_high = df['high'].iloc[i]
                current_low = df['low'].iloc[i]
                current_close = df['close'].iloc[i]
                
                # Check for stop loss if in position
                if abs(position) > 0.01:  # Some position left
                    if position > 0 and current_low <= stop_loss:
                        # Calculate P&L for remaining position
                        trade_pnl = (stop_loss - entry_price) * position * (initial_capital / entry_price)
                        df['trade_pnl'].iloc[i] = trade_pnl
                        df['equity'].iloc[i] += trade_pnl
                        df['trade_result'].iloc[i] = 'Stop Loss'
                        
                        # Reset position
                        position = 0
                        df['position'].iloc[i] = 0
                        df['entry_price'].iloc[i] = np.nan
                        df['stop_loss'].iloc[i] = np.nan
                        df['take_profit_1'].iloc[i] = np.nan
                        df['take_profit_2'].iloc[i] = np.nan
                        df['take_profit_3'].iloc[i] = np.nan
                        position_portions = [0, 0, 0]
                    
                    elif position < 0 and current_high >= stop_loss:
                        # Calculate P&L for remaining position
                        trade_pnl = (entry_price - stop_loss) * abs(position) * (initial_capital / entry_price)
                        df['trade_pnl'].iloc[i] = trade_pnl
                        df['equity'].iloc[i] += trade_pnl
                        df['trade_result'].iloc[i] = 'Stop Loss'
                        
                        # Reset position
                        position = 0
                        df['position'].iloc[i] = 0
                        df['entry_price'].iloc[i] = np.nan
                        df['stop_loss'].iloc[i] = np.nan
                        df['take_profit_1'].iloc[i] = np.nan
                        df['take_profit_2'].iloc[i] = np.nan
                        df['take_profit_3'].iloc[i] = np.nan
                        position_portions = [0, 0, 0]
                
                # Check for partial take profits if in position
                if position > 0:
                    # Check take profit levels for long position
                    for j in range(3):
                        if position_portions[j] > 0:  # This portion not yet taken
                            tp_level = entry_price + (df['atr'].iloc[i] * self.atr_mult * self.risk_reward * self.profit_levels[j])
                            if current_high >= tp_level:
                                # Calculate P&L for this portion
                                trade_pnl = (tp_level - entry_price) * position_portions[j] * (initial_capital / entry_price)
                                df['trade_pnl'].iloc[i] += trade_pnl
                                df['equity'].iloc[i] += trade_pnl
                                df['trade_result'].iloc[i] = f'Take Profit {j+1}'
                                
                                # Reduce position
                                position -= position_portions[j]
                                df['position'].iloc[i] = position
                                position_portions[j] = 0
                
                elif position < 0:
                    # Check take profit levels for short position
                    for j in range(3):
                        if position_portions[j] < 0:  # This portion not yet taken
                            tp_level = entry_price - (df['atr'].iloc[i] * self.atr_mult * self.risk_reward * self.profit_levels[j])
                            if current_low <= tp_level:
                                # Calculate P&L for this portion
                                trade_pnl = (entry_price - tp_level) * abs(position_portions[j]) * (initial_capital / entry_price)
                                df['trade_pnl'].iloc[i] += trade_pnl
                                df['equity'].iloc[i] += trade_pnl
                                df['trade_result'].iloc[i] = f'Take Profit {j+1}'
                                
                                # Reduce position
                                position -= position_portions[j]
                                df['position'].iloc[i] = position
                                position_portions[j] = 0
                
                # Check for entry if not in position
                if abs(position) < 0.01:  # No position or very small position left
                    if df['can_long'].iloc[i]:
                        # Enter long position
                        position = 1
                        entry_price = current_close
                        stop_loss = df['long_sl'].iloc[i]
                        
                        # Calculate take profit levels
                        for j in range(3):
                            take_profit_levels[j] = entry_price + (df['atr'].iloc[i] * self.atr_mult * self.risk_reward * self.profit_levels[j])
                            position_portions[j] = self.profit_portions[j]  # Allocate position portions
                        
                        df['position'].iloc[i] = position
                        df['entry_price'].iloc[i] = entry_price
                        df['stop_loss'].iloc[i] = stop_loss
                        df['take_profit_1'].iloc[i] = take_profit_levels[0]
                        df['take_profit_2'].iloc[i] = take_profit_levels[1]
                        df['take_profit_3'].iloc[i] = take_profit_levels[2]
                    
                    elif df['can_short'].iloc[i]:
                        # Enter short position
                        position = -1
                        entry_price = current_close
                        stop_loss = df['short_sl'].iloc[i]
                        
                        # Calculate take profit levels
                        for j in range(3):
                            take_profit_levels[j] = entry_price - (df['atr'].iloc[i] * self.atr_mult * self.risk_reward * self.profit_levels[j])
                            position_portions[j] = -self.profit_portions[j]  # Negative for short positions
                        
                        df['position'].iloc[i] = position
                        df['entry_price'].iloc[i] = entry_price
                        df['stop_loss'].iloc[i] = stop_loss
                        df['take_profit_1'].iloc[i] = take_profit_levels[0]
                        df['take_profit_2'].iloc[i] = take_profit_levels[1]
                        df['take_profit_3'].iloc[i] = take_profit_levels[2]
            
            # Calculate daily returns
            df['daily_returns'] = df['equity'].pct_change()
            
            # Calculate cumulative returns
            df['cumulative_returns'] = (1 + df['daily_returns']).cumprod() - 1
            
            return df
    
    # Enhanced strategy with market regime classification
    class MarketRegimeStrategy(MultiIndicatorFusionStrategy):
        def __init__(self, adx_period=14, adx_threshold=25, volatility_period=20, **kwargs):
            super().__init__(**kwargs)
            self.adx_period = adx_period
            self.adx_threshold = adx_threshold
            self.volatility_period = volatility_period
        
        def calculate_indicators(self, data):
            # Call parent method to calculate base indicators
            df = super().calculate_indicators(data)
            
            # Calculate ADX for trend strength
            high = data['high']
            low = data['low']
            close = data['close']
            
            # Calculate True Range
            tr1 = high - low
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))
            tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
            
            # Calculate +DM and -DM
            up_move = high - high.shift(1)
            down_move = low.shift(1) - low
            
            plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
            minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
            
            # Calculate smoothed values
            smoothed_tr = tr.rolling(window=self.adx_period).mean()
            smoothed_plus_dm = pd.Series(plus_dm).rolling(window=self.adx_period).mean()
            smoothed_minus_dm = pd.Series(minus_dm).rolling(window=self.adx_period).mean()
            
            # Calculate +DI and -DI
            plus_di = 100 * smoothed_plus_dm / smoothed_tr
            minus_di = 100 * smoothed_minus_dm / smoothed_tr
            
            # Calculate DX and ADX
            dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
            adx = dx.rolling(window=self.adx_period).mean()
            
            df['adx'] = adx
            df['plus_di'] = plus_di
            df['minus_di'] = minus_di
            
            # Calculate historical volatility
            df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
            df['volatility'] = df['log_returns'].rolling(window=self.volatility_period).std() * np.sqrt(252)
            
            # Classify market regime
            df['strong_trend'] = df['adx'] > self.adx_threshold
            df['high_volatility'] = df['volatility'] > df['volatility'].rolling(window=self.volatility_period*2).mean()
            
            # Define market regimes:
            # 1: Strong trend, low volatility - ideal for trend following
            # 2: Strong trend, high volatility - good for trend following with wider stops
            # 3: Weak trend, low volatility - sideways or choppy market
            # 4: Weak trend, high volatility - potentially transitioning market
            
            df['market_regime'] = np.where(
                df['strong_trend'] & ~df['high_volatility'], 1,
                np.where(
                    df['strong_trend'] & df['high_volatility'], 2,
                    np.where(
                        ~df['strong_trend'] & ~df['high_volatility'], 3,
                        4
                    )
                )
            )
            
            # Adjust entry conditions based on market regime
            df['can_long'] = np.where(
                df['market_regime'] == 1, df['long_cond'] & ~df['trap_long'],  # Trade normally in strong trend, low volatility
                np.where(
                    df['market_regime'] == 2, df['long_cond'] & ~df['trap_long'] & (df['plus_di'] > df['minus_di']),  # More confirmation in high volatility
                    np.where(
                        df['market_regime'] == 3, False,  # Don't trade in sideways markets
                        df['long_cond'] & ~df['trap_long'] & ~df['is_sideways'] & (df['plus_di'] > 1.5 * df['minus_di'])  # Very strict in transitioning markets
                    )
                )
            )
            
            df['can_short'] = np.where(
                df['market_regime'] == 1, df['short_cond'] & ~df['trap_short'],  # Trade normally in strong trend, low volatility
                np.where(
                    df['market_regime'] == 2, df['short_cond'] & ~df['trap_short'] & (df['minus_di'] > df['plus_di']),  # More confirmation in high volatility
                    np.where(
                        df['market_regime'] == 3, False,  # Don't trade in sideways markets
                        df['short_cond'] & ~df['trap_short'] & ~df['is_sideways'] & (df['minus_di'] > 1.5 * df['plus_di'])  # Very strict in transitioning markets
                    )
                )
            )
            
            # Adjust stop loss and take profit based on market regime
            regime_atr_mults = {1: self.atr_mult, 2: self.atr_mult * 1.5, 3: self.atr_mult, 4: self.atr_mult * 2.0}
            regime_rr_ratios = {1: self.risk_reward, 2: self.risk_reward * 1.5, 3: self.risk_reward, 4: self.risk_reward * 0.8}
            
            df['regime_atr_mult'] = df['market_regime'].map(regime_atr_mults)
            df['regime_rr'] = df['market_regime'].map(regime_rr_ratios)
            
            # Recalculate stop loss and take profit levels
            df['long_sl'] = df['close'] - df['atr'] * df['regime_atr_mult']
            df['long_tp'] = df['close'] + df['atr'] * df['regime_atr_mult'] * df['regime_rr']
            df['short_sl'] = df['close'] + df['atr'] * df['regime_atr_mult']
            df['short_tp'] = df['close'] - df['atr'] * df['regime_atr_mult'] * df['regime_rr']
            
            return df
    
    # Test enhanced strategies
    print("\nTesting volume confirmation strategy...")
    volume_strategy = VolumeConfirmationStrategy(volume_ma_period=20, volume_threshold=1.5)
    volume_results = volume_strategy.backtest(data)
    volume_metrics = volume_strategy.calculate_performance_metrics(volume_results)
    volume_strategy.print_performance_summary(volume_metrics)
    
    print("\nTesting dynamic risk management strategy...")
    dynamic_risk_strategy = DynamicRiskStrategy(atr_period_short=7, atr_period_long=21)
    dynamic_risk_results = dynamic_risk_strategy.backtest(data)
    dynamic_risk_metrics = dynamic_risk_strategy.calculate_performance_metrics(dynamic_risk_results)
    dynamic_risk_strategy.print_performance_summary(dynamic_risk_metrics)
    
    print("\nTesting partial profit taking strategy...")
    partial_profit_strategy = PartialProfitStrategy(profit_levels=[0.33, 0.67, 1.0], profit_portions=[0.3, 0.3, 0.4])
    partial_profit_results = partial_profit_strategy.backtest(data)
    partial_profit_metrics = partial_profit_strategy.calculate_performance_metrics(partial_profit_results)
    partial_profit_strategy.print_performance_summary(partial_profit_metrics)
    
    print("\nTesting market regime classification strategy...")
    market_regime_strategy = MarketRegimeStrategy(adx_period=14, adx_threshold=25, volatility_period=20)
    market_regime_results = market_regime_strategy.backtest(data)
    market_regime_metrics = market_regime_strategy.calculate_performance_metrics(market_regime_results)
    market_regime_strategy.print_performance_summary(market_regime_metrics)
    
    # Compare equity curves
    plt.figure(figsize=(15, 8))
    
    plt.plot(original_results.index, original_results['equity'], label='Original Strategy')
    plt.plot(volume_results.index, volume_results['equity'], label='Volume Confirmation')
    plt.plot(dynamic_risk_results.index, dynamic_risk_results['equity'], label='Dynamic Risk Management')
    plt.plot(partial_profit_results.index, partial_profit_results['equity'], label='Partial Profit Taking')
    plt.plot(market_regime_results.index, market_regime_results['equity'], label='Market Regime Classification')
    
    plt.title('Equity Curve Comparison')
    plt.xlabel('Date')
    plt.ylabel('Equity')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Compare metrics
    comparison = pd.DataFrame({
        'Original': [
            original_metrics['total_return'],
            original_metrics['annual_return'],
            original_metrics['sharpe_ratio'],
            original_metrics['max_drawdown'],
            original_metrics['win_rate'],
            original_metrics['profit_factor'],
            original_metrics['total_trades']
        ],
        'Volume Confirmation': [
            volume_metrics['total_return'],
            volume_metrics['annual_return'],
            volume_metrics['sharpe_ratio'],
            volume_metrics['max_drawdown'],
            volume_metrics['win_rate'],
            volume_metrics['profit_factor'],
            volume_metrics['total_trades']
        ],
        'Dynamic Risk': [
            dynamic_risk_metrics['total_return'],
            dynamic_risk_metrics['annual_return'],
            dynamic_risk_metrics['sharpe_ratio'],
            dynamic_risk_metrics['max_drawdown'],
            dynamic_risk_metrics['win_rate'],
            dynamic_risk_metrics['profit_factor'],
            dynamic_risk_metrics['total_trades']
        ],
        'Partial Profit': [
            partial_profit_metrics['total_return'],
            partial_profit_metrics['annual_return'],
            partial_profit_metrics['sharpe_ratio'],
            partial_profit_metrics['max_drawdown'],
            partial_profit_metrics['win_rate'],
            partial_profit_metrics['profit_factor'],
            partial_profit_metrics['total_trades']
        ],
        'Market Regime': [
            market_regime_metrics['total_return'],
            market_regime_metrics['annual_return'],
            market_regime_metrics['sharpe_ratio'],
            market_regime_metrics['max_drawdown'],
            market_regime_metrics['win_rate'],
            market_regime_metrics['profit_factor'],
            market_regime_metrics['total_trades']
        ]
    }, index=['Total Return', 'Annual Return', 'Sharpe Ratio', 'Max Drawdown', 'Win Rate', 'Profit Factor', 'Total Trades'])
    
    print("\nStrategy Comparison:")
    print(comparison)
    
    # Plot comparison metrics
    plt.figure(figsize=(15, 10))
    
    metrics_to_plot = ['Total Return', 'Sharpe Ratio', 'Max Drawdown', 'Win Rate', 'Profit Factor']
    for i, metric in enumerate(metrics_to_plot):
        plt.subplot(3, 2, i+1)
        
        # For drawdown, make it positive for easier comparison
        values = comparison.loc[metric] * (-1 if metric == 'Max Drawdown' else 1)
        
        plt.bar(comparison.columns, values)
        plt.title(metric)
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return {
        'original': {'strategy': original_strategy, 'results': original_results, 'metrics': original_metrics},
        'volume': {'strategy': volume_strategy, 'results': volume_results, 'metrics': volume_metrics},
        'dynamic_risk': {'strategy': dynamic_risk_strategy, 'results': dynamic_risk_results, 'metrics': dynamic_risk_metrics},
        'partial_profit': {'strategy': partial_profit_strategy, 'results': partial_profit_results, 'metrics': partial_profit_metrics},
        'market_regime': {'strategy': market_regime_strategy, 'results': market_regime_results, 'metrics': market_regime_metrics}
    }

# Run the tests
if __name__ == "__main__":
    # Test the strategy with default parameters
    strategy, results, metrics = test_strategy()
    
    # Uncomment to run other tests
    # optimization_results = parameter_optimization()
    # enhancement_results = test_enhanced_strategies()