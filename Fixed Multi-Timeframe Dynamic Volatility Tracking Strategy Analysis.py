import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from datetime import datetime, timedelta
import talib as ta
import warnings

warnings.filterwarnings('ignore')

class DynamicVolatilityTrackingStrategy:
    """
    Implementation of the Multi-Timeframe Dynamic Volatility Tracking Strategy
    
    This strategy combines EMA crossovers with RSI filters, ATR-based risk management,
    and multi-level profit targets for short-term trading.
    """
    
    def __init__(self, 
                 ema_short_length=9, 
                 ema_long_length=21, 
                 rsi_length=14, 
                 atr_length=14, 
                 rsi_overbought=70, 
                 rsi_oversold=30,
                 higher_tf_rsi_overbought=70,
                 higher_tf_rsi_oversold=30,
                 pivot_lookback=5,
                 volume_lookback=20,
                 volume_multiplier=1.0,
                 risk_reward_ratio=1.2,
                 trailing_stop_multiplier=1.2,
                 confirm_bars=2,
                 tp1_profit_mult=1.0,
                 tp2_profit_mult=1.5,
                 tp3_profit_mult=2.0,
                 tp1_exit_percentage=33,
                 tp2_exit_percentage=33,
                 tp3_exit_percentage=34,
                 max_trades_per_trend=5,
                 trade_decrease_factor=0,
                 position_size=1.0):
        """
        Initialize the strategy with parameters
        """
        # EMA parameters
        self.ema_short_length = ema_short_length
        self.ema_long_length = ema_long_length
        
        # RSI parameters
        self.rsi_length = rsi_length
        self.rsi_overbought = rsi_overbought
        self.rsi_oversold = rsi_oversold
        
        # Higher timeframe RSI parameters
        self.higher_tf_rsi_overbought = higher_tf_rsi_overbought
        self.higher_tf_rsi_oversold = higher_tf_rsi_oversold
        
        # ATR parameters
        self.atr_length = atr_length
        
        # Pivot parameters
        self.pivot_lookback = pivot_lookback
        
        # Volume parameters
        self.volume_lookback = volume_lookback
        self.volume_multiplier = volume_multiplier
        
        # Risk parameters
        self.risk_reward_ratio = risk_reward_ratio
        self.trailing_stop_multiplier = trailing_stop_multiplier
        
        # Confirmation parameters
        self.confirm_bars = confirm_bars
        
        # Take profit parameters
        self.tp1_profit_mult = tp1_profit_mult
        self.tp2_profit_mult = tp2_profit_mult
        self.tp3_profit_mult = tp3_profit_mult
        self.tp1_exit_percentage = tp1_exit_percentage / 100
        self.tp2_exit_percentage = tp2_exit_percentage / 100
        self.tp3_exit_percentage = tp3_exit_percentage / 100
        
        # Trade count parameters
        self.max_trades_per_trend = max(1, max_trades_per_trend - trade_decrease_factor)
        
        # Position size
        self.position_size = position_size
        
    def calculate_indicators(self, data):
        """
        Calculate all technical indicators required for the strategy
        """
        # Make a copy of the data to avoid modifying the original
        df = data.copy()
        
        # Calculate EMAs
        df['ema_short'] = ta.EMA(df['close'], timeperiod=self.ema_short_length)
        df['ema_long'] = ta.EMA(df['close'], timeperiod=self.ema_long_length)
        
        # Calculate RSI
        df['rsi'] = ta.RSI(df['close'], timeperiod=self.rsi_length)
        
        # Calculate ATR
        df['atr'] = ta.ATR(df['high'], df['low'], df['close'], timeperiod=self.atr_length)
        
        # Calculate higher timeframe RSI
        # Since we're working with simulated data, we'll use a simple approach
        # to approximate higher timeframe by applying a longer period RSI
        higher_tf_multiplier = 3  # 3x the normal timeframe
        df['higher_tf_rsi'] = ta.RSI(df['close'], timeperiod=self.rsi_length * higher_tf_multiplier)
        
        # Calculate volume moving average
        df['volume_ma'] = df['volume'].rolling(window=self.volume_lookback).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma'].replace(0, 1)  # Avoid division by zero
        
        # Calculate raw signal conditions
        df['raw_long_signal'] = (df['ema_short'] > df['ema_long']) & (df['rsi'] < self.rsi_overbought)
        df['raw_short_signal'] = (df['ema_short'] < df['ema_long']) & (df['rsi'] > self.rsi_oversold)
        
        return df
    
    def confirm_condition(self, condition, bars):
        """
        Confirm a condition for N consecutive bars
        """
        result = condition.copy()
        
        for i in range(1, bars):
            result = result & condition.shift(i)
            
        return result
    
    def calculate_pivot_points(self, data):
        """
        Calculate pivot high and low points
        """
        df = data.copy()
        
        # Initialize pivot columns
        df['pivot_high'] = np.nan
        df['pivot_low'] = np.nan
        
        # Calculate pivot highs and lows
        for i in range(self.pivot_lookback, len(df) - self.pivot_lookback):
            # Check if current bar's high is highest in the lookback period
            if df.iloc[i]['high'] == df.iloc[i-self.pivot_lookback:i+self.pivot_lookback+1]['high'].max():
                df.loc[df.index[i], 'pivot_high'] = df.iloc[i]['high']
                
            # Check if current bar's low is lowest in the lookback period
            if df.iloc[i]['low'] == df.iloc[i-self.pivot_lookback:i+self.pivot_lookback+1]['low'].min():
                df.loc[df.index[i], 'pivot_low'] = df.iloc[i]['low']
        
        return df
    
    def backtest(self, data):
        """
        Run backtest on the provided data
        """
        # Calculate all indicators
        df = self.calculate_indicators(data)
        
        # Calculate pivot points
        df = self.calculate_pivot_points(df)
        
        # Confirm signals
        df['long_signal'] = self.confirm_condition(df['raw_long_signal'], self.confirm_bars)
        df['short_signal'] = self.confirm_condition(df['raw_short_signal'], self.confirm_bars)
        
        # Initialize trade tracking columns
        df['position'] = 0  # 1 for long, -1 for short, 0 for no position
        df['trailing_stop'] = np.nan
        df['tp1_level'] = np.nan
        df['tp2_level'] = np.nan
        df['tp3_level'] = np.nan
        df['tp1_hit'] = False
        df['tp2_hit'] = False
        df['tp3_hit'] = False
        df['exit_reason'] = None
        df['trade_count'] = 0
        df['position_size'] = 0.0
        df['entry_price'] = np.nan
        
        # Initialize performance tracking columns
        df['equity'] = 100000  # Starting equity
        df['trade_pnl'] = 0.0
        df['trade_return'] = 0.0
        
        # Iterate through the data to simulate trading
        in_long_trade = False
        in_short_trade = False
        long_trade_count = 0
        current_position_size = self.position_size
        entry_price = 0
        
        for i in range(self.confirm_bars + self.pivot_lookback, len(df)):
            # Get current and previous row
            curr = df.iloc[i]
            prev = df.iloc[i-1]
            
            # Default is to carry forward previous position and equity
            df.loc[df.index[i], 'position'] = prev['position']
            df.loc[df.index[i], 'position_size'] = prev['position_size']
            df.loc[df.index[i], 'equity'] = prev['equity']
            df.loc[df.index[i], 'entry_price'] = prev['entry_price']
            df.loc[df.index[i], 'tp1_hit'] = prev['tp1_hit']
            df.loc[df.index[i], 'tp2_hit'] = prev['tp2_hit']
            df.loc[df.index[i], 'tp3_hit'] = prev['tp3_hit']
            
            # Get adjusted stop loss based on volume
            vol_ratio = max(curr['volume_ratio'], 0.01)  # Ensure non-zero
            adj_sl = curr['atr'] / (vol_ratio * self.volume_multiplier)
            
            # Process long entry
            if curr['long_signal'] and not in_long_trade and not in_short_trade and long_trade_count < self.max_trades_per_trend:
                # Enter long position
                in_long_trade = True
                long_trade_count += 1
                entry_price = curr['close']
                trailing_stop = curr['low'] - adj_sl
                
                # Calculate take profit levels
                tp1 = entry_price + (curr['atr'] * self.risk_reward_ratio * self.tp1_profit_mult)
                tp2 = entry_price + (curr['atr'] * self.risk_reward_ratio * self.tp2_profit_mult)
                tp3 = entry_price + (curr['atr'] * self.risk_reward_ratio * self.tp3_profit_mult)
                
                # Update position tracking
                df.loc[df.index[i], 'position'] = 1
                df.loc[df.index[i], 'trailing_stop'] = trailing_stop
                df.loc[df.index[i], 'tp1_level'] = tp1
                df.loc[df.index[i], 'tp2_level'] = tp2
                df.loc[df.index[i], 'tp3_level'] = tp3
                df.loc[df.index[i], 'tp1_hit'] = False
                df.loc[df.index[i], 'tp2_hit'] = False
                df.loc[df.index[i], 'tp3_hit'] = False
                df.loc[df.index[i], 'trade_count'] = long_trade_count
                df.loc[df.index[i], 'position_size'] = self.position_size
                df.loc[df.index[i], 'entry_price'] = entry_price
            
            # Process short entry
            elif curr['short_signal'] and not in_short_trade and not in_long_trade:
                # Enter short position
                in_short_trade = True
                entry_price = curr['close']
                trailing_stop = curr['high'] + adj_sl
                
                # Calculate take profit levels
                tp1 = entry_price - (curr['atr'] * self.risk_reward_ratio * self.tp1_profit_mult)
                tp2 = entry_price - (curr['atr'] * self.risk_reward_ratio * self.tp2_profit_mult)
                tp3 = entry_price - (curr['atr'] * self.risk_reward_ratio * self.tp3_profit_mult)
                
                # Update position tracking
                df.loc[df.index[i], 'position'] = -1
                df.loc[df.index[i], 'trailing_stop'] = trailing_stop
                df.loc[df.index[i], 'tp1_level'] = tp1
                df.loc[df.index[i], 'tp2_level'] = tp2
                df.loc[df.index[i], 'tp3_level'] = tp3
                df.loc[df.index[i], 'tp1_hit'] = False
                df.loc[df.index[i], 'tp2_hit'] = False
                df.loc[df.index[i], 'tp3_hit'] = False
                df.loc[df.index[i], 'position_size'] = self.position_size
                df.loc[df.index[i], 'entry_price'] = entry_price
            
            # Update trailing stop if in a position
            elif in_long_trade:
                # Update trailing stop for long position
                base_stop = curr['close'] - adj_sl
                
                # Check pivot lows for trailing stop
                if not np.isnan(curr['pivot_low']) and curr['pivot_low'] > prev['trailing_stop']:
                    new_stop = curr['pivot_low']
                else:
                    new_stop = max(prev['trailing_stop'], base_stop)
                
                df.loc[df.index[i], 'trailing_stop'] = new_stop
                
                # Check for partial take profits (TP1)
                if not prev['tp1_hit'] and curr['close'] >= prev['tp1_level']:
                    # Take partial profit at TP1
                    df.loc[df.index[i], 'tp1_hit'] = True
                    partial_size = self.tp1_exit_percentage * self.position_size
                    remaining_size = df.loc[df.index[i], 'position_size'] - partial_size
                    df.loc[df.index[i], 'position_size'] = remaining_size
                    
                    # Calculate profit
                    trade_pnl = partial_size * (curr['close'] - entry_price)
                    df.loc[df.index[i], 'trade_pnl'] += trade_pnl
                    df.loc[df.index[i], 'equity'] += trade_pnl
                
                # Check for partial take profits (TP2)
                if prev['tp1_hit'] and not prev['tp2_hit'] and curr['close'] >= prev['tp2_level']:
                    # Take partial profit at TP2
                    df.loc[df.index[i], 'tp2_hit'] = True
                    partial_size = self.tp2_exit_percentage * self.position_size
                    remaining_size = df.loc[df.index[i], 'position_size'] - partial_size
                    df.loc[df.index[i], 'position_size'] = remaining_size
                    
                    # Calculate profit
                    trade_pnl = partial_size * (curr['close'] - entry_price)
                    df.loc[df.index[i], 'trade_pnl'] += trade_pnl
                    df.loc[df.index[i], 'equity'] += trade_pnl
                
                # Check for partial take profits (TP3)
                if prev['tp2_hit'] and not prev['tp3_hit'] and curr['close'] >= prev['tp3_level']:
                    # Take partial profit at TP3
                    df.loc[df.index[i], 'tp3_hit'] = True
                    partial_size = self.tp3_exit_percentage * self.position_size
                    remaining_size = df.loc[df.index[i], 'position_size'] - partial_size
                    df.loc[df.index[i], 'position_size'] = remaining_size
                    
                    # Calculate profit
                    trade_pnl = partial_size * (curr['close'] - entry_price)
                    df.loc[df.index[i], 'trade_pnl'] += trade_pnl
                    df.loc[df.index[i], 'equity'] += trade_pnl
                
                # Check for exit conditions
                exit_long = (
                    (curr['close'] < prev['trailing_stop']) or  # Trailing stop hit
                    (curr['rsi'] > self.rsi_overbought) or  # RSI overbought
                    (curr['higher_tf_rsi'] > self.higher_tf_rsi_overbought)  # Higher TF RSI overbought
                )
                
                if exit_long:
                    # Close position
                    exit_price = min(curr['close'], prev['trailing_stop'])  # Account for gaps
                    
                    # Calculate profit
                    remaining_size = df.loc[df.index[i], 'position_size']
                    trade_pnl = remaining_size * (exit_price - entry_price)
                    df.loc[df.index[i], 'trade_pnl'] += trade_pnl
                    df.loc[df.index[i], 'equity'] += trade_pnl
                    
                    # Update position tracking
                    df.loc[df.index[i], 'position'] = 0
                    df.loc[df.index[i], 'position_size'] = 0
                    df.loc[df.index[i], 'trailing_stop'] = np.nan
                    in_long_trade = False
                    
                    # Determine exit reason
                    if curr['close'] < prev['trailing_stop']:
                        df.loc[df.index[i], 'exit_reason'] = 'Trailing Stop'
                    elif curr['rsi'] > self.rsi_overbought:
                        df.loc[df.index[i], 'exit_reason'] = 'RSI Overbought'
                    else:
                        df.loc[df.index[i], 'exit_reason'] = 'Higher TF RSI Overbought'
            
            # Update trailing stop if in a short position
            elif in_short_trade:
                # Update trailing stop for short position
                base_stop = curr['close'] + adj_sl
                
                # Check pivot highs for trailing stop
                if not np.isnan(curr['pivot_high']) and curr['pivot_high'] < prev['trailing_stop']:
                    new_stop = curr['pivot_high']
                else:
                    new_stop = min(prev['trailing_stop'], base_stop)
                
                df.loc[df.index[i], 'trailing_stop'] = new_stop
                
                # Check for partial take profits (TP1)
                if not prev['tp1_hit'] and curr['close'] <= prev['tp1_level']:
                    # Take partial profit at TP1
                    df.loc[df.index[i], 'tp1_hit'] = True
                    partial_size = self.tp1_exit_percentage * self.position_size
                    remaining_size = df.loc[df.index[i], 'position_size'] - partial_size
                    df.loc[df.index[i], 'position_size'] = remaining_size
                    
                    # Calculate profit
                    trade_pnl = partial_size * (entry_price - curr['close'])
                    df.loc[df.index[i], 'trade_pnl'] += trade_pnl
                    df.loc[df.index[i], 'equity'] += trade_pnl
                
                # Check for partial take profits (TP2)
                if prev['tp1_hit'] and not prev['tp2_hit'] and curr['close'] <= prev['tp2_level']:
                    # Take partial profit at TP2
                    df.loc[df.index[i], 'tp2_hit'] = True
                    partial_size = self.tp2_exit_percentage * self.position_size
                    remaining_size = df.loc[df.index[i], 'position_size'] - partial_size
                    df.loc[df.index[i], 'position_size'] = remaining_size
                    
                    # Calculate profit
                    trade_pnl = partial_size * (entry_price - curr['close'])
                    df.loc[df.index[i], 'trade_pnl'] += trade_pnl
                    df.loc[df.index[i], 'equity'] += trade_pnl
                
                # Check for partial take profits (TP3)
                if prev['tp2_hit'] and not prev['tp3_hit'] and curr['close'] <= prev['tp3_level']:
                    # Take partial profit at TP3
                    df.loc[df.index[i], 'tp3_hit'] = True
                    partial_size = self.tp3_exit_percentage * self.position_size
                    remaining_size = df.loc[df.index[i], 'position_size'] - partial_size
                    df.loc[df.index[i], 'position_size'] = remaining_size
                    
                    # Calculate profit
                    trade_pnl = partial_size * (entry_price - curr['close'])
                    df.loc[df.index[i], 'trade_pnl'] += trade_pnl
                    df.loc[df.index[i], 'equity'] += trade_pnl
                
                # Check for exit conditions
                exit_short = (
                    (curr['close'] > prev['trailing_stop']) or  # Trailing stop hit
                    (curr['rsi'] < self.rsi_oversold) or  # RSI oversold
                    (curr['higher_tf_rsi'] < self.higher_tf_rsi_oversold)  # Higher TF RSI oversold
                )
                
                if exit_short:
                    # Close position
                    exit_price = max(curr['close'], prev['trailing_stop'])  # Account for gaps
                    
                    # Calculate profit
                    remaining_size = df.loc[df.index[i], 'position_size']
                    trade_pnl = remaining_size * (entry_price - exit_price)
                    df.loc[df.index[i], 'trade_pnl'] += trade_pnl
                    df.loc[df.index[i], 'equity'] += trade_pnl
                    
                    # Update position tracking
                    df.loc[df.index[i], 'position'] = 0
                    df.loc[df.index[i], 'position_size'] = 0
                    df.loc[df.index[i], 'trailing_stop'] = np.nan
                    in_short_trade = False
                    
                    # Determine exit reason
                    if curr['close'] > prev['trailing_stop']:
                        df.loc[df.index[i], 'exit_reason'] = 'Trailing Stop'
                    elif curr['rsi'] < self.rsi_oversold:
                        df.loc[df.index[i], 'exit_reason'] = 'RSI Oversold'
                    else:
                        df.loc[df.index[i], 'exit_reason'] = 'Higher TF RSI Oversold'
            
            # Reset trade counter when bullish trend ends
            if not curr['raw_long_signal'] and prev['raw_long_signal']:
                long_trade_count = 0
        
        # Calculate daily returns
        df['daily_returns'] = df['close'].pct_change()
        
        # Calculate equity curve and returns
        df['equity_returns'] = df['equity'].pct_change().fillna(0)
        df['cumulative_returns'] = (1 + df['equity_returns']).cumprod() - 1
        
        return df
    
    def calculate_performance_metrics(self, results):
        """
        Calculate performance metrics
        """
        # Extract trade information
        trades = results[(results['trade_pnl'] != 0) | (results['exit_reason'].notna())].copy()
        trades['trade_pnl'] = trades['trade_pnl'].fillna(0)  # Ensure no NaN values
        
        # Calculate metrics
        initial_equity = results['equity'].iloc[0]
        final_equity = results['equity'].iloc[-1]
        total_return = (final_equity / initial_equity) - 1
        
        # Annualized return (assuming 252 trading days per year)
        days = len(results)
        years = days / 252
        annual_return = (1 + total_return) ** (1 / years) - 1
        
        # Volatility
        daily_returns = results['equity_returns']
        annual_volatility = daily_returns.std() * np.sqrt(252)
        
        # Sharpe ratio (assuming risk-free rate = 0)
        sharpe_ratio = annual_return / annual_volatility if annual_volatility != 0 else 0
        
        # Drawdown
        equity_curve = results['equity']
        running_max = equity_curve.cummax()
        drawdown = (equity_curve / running_max) - 1
        max_drawdown = drawdown.min()
        
        # Win rate and related metrics
        winning_trades = trades[trades['trade_pnl'] > 0]
        losing_trades = trades[trades['trade_pnl'] <= 0]
        
        win_rate = len(winning_trades) / len(trades) if len(trades) > 0 else 0
        
        # Profit factor
        gross_profit = winning_trades['trade_pnl'].sum() if len(winning_trades) > 0 else 0
        gross_loss = abs(losing_trades['trade_pnl'].sum()) if len(losing_trades) > 0 else 1  # Avoid division by zero
        profit_factor = gross_profit / gross_loss if gross_loss != 0 else float('inf')
        
        # Average trade
        avg_trade = trades['trade_pnl'].mean() if len(trades) > 0 else 0
        avg_win = winning_trades['trade_pnl'].mean() if len(winning_trades) > 0 else 0
        avg_loss = losing_trades['trade_pnl'].mean() if len(losing_trades) > 0 else 0
        
        # Exit reasons
        exit_reasons = trades['exit_reason'].value_counts().to_dict() if len(trades) > 0 else {}
        
        # Create metrics dictionary
        metrics = {
            'total_return': total_return,
            'annual_return': annual_return,
            'annual_volatility': annual_volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_trade': avg_trade,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'num_trades': len(trades),
            'exit_reasons': exit_reasons
        }
        
        return metrics
    
    def plot_results(self, results):
        """
        Plot backtest results
        """
        # Set up the figure
        plt.figure(figsize=(16, 20))
        
        # Plot 1: Price chart with EMAs and signals
        ax1 = plt.subplot(4, 1, 1)
        ax1.plot(results.index, results['close'], color='black', linewidth=1, label='Close')
        ax1.plot(results.index, results['ema_short'], color='blue', linewidth=1, label=f'EMA ({self.ema_short_length})')
        ax1.plot(results.index, results['ema_long'], color='red', linewidth=1, label=f'EMA ({self.ema_long_length})')
        
        # Plot buy and sell signals
        long_entries = results[results['position'].diff() == 1]
        short_entries = results[results['position'].diff() == -1]
        long_exits = results[(results['position'].shift(1) == 1) & (results['position'] == 0)]
        short_exits = results[(results['position'].shift(1) == -1) & (results['position'] == 0)]
        
        ax1.scatter(long_entries.index, long_entries['close'], marker='^', color='green', s=100, label='Long Entry')
        ax1.scatter(short_entries.index, short_entries['close'], marker='v', color='red', s=100, label='Short Entry')
        ax1.scatter(long_exits.index, long_exits['close'], marker='x', color='black', s=100, label='Long Exit')
        ax1.scatter(short_exits.index, short_exits['close'], marker='x', color='black', s=100, label='Short Exit')
        
        # Plot trailing stops
        ax1.plot(results.index, results['trailing_stop'], 'k--', alpha=0.5, label='Trailing Stop')
        
        ax1.set_title('Price Chart with Signals')
        ax1.set_ylabel('Price')
        ax1.legend(loc='upper left')
        ax1.grid(True)
        
        # Plot 2: RSI
        ax2 = plt.subplot(4, 1, 2)
        ax2.plot(results.index, results['rsi'], color='purple', linewidth=1)
        ax2.axhline(y=self.rsi_overbought, color='red', linestyle='--')
        ax2.axhline(y=50, color='black', linestyle='-')
        ax2.axhline(y=self.rsi_oversold, color='green', linestyle='--')
        ax2.fill_between(results.index, results['rsi'], self.rsi_overbought, 
                        where=results['rsi'] > self.rsi_overbought, 
                        color='red', alpha=0.3)
        ax2.fill_between(results.index, results['rsi'], self.rsi_oversold, 
                        where=results['rsi'] < self.rsi_oversold, 
                        color='green', alpha=0.3)
        ax2.set_title('RSI Indicator')
        ax2.set_ylabel('RSI')
        ax2.set_ylim(0, 100)
        ax2.grid(True)
        
        # Plot 3: Higher Timeframe RSI
        ax3 = plt.subplot(4, 1, 3)
        ax3.plot(results.index, results['higher_tf_rsi'], color='blue', linewidth=1)
        ax3.axhline(y=self.higher_tf_rsi_overbought, color='red', linestyle='--')
        ax3.axhline(y=50, color='black', linestyle='-')
        ax3.axhline(y=self.higher_tf_rsi_oversold, color='green', linestyle='--')
        ax3.fill_between(results.index, results['higher_tf_rsi'], self.higher_tf_rsi_overbought, 
                        where=results['higher_tf_rsi'] > self.higher_tf_rsi_overbought, 
                        color='red', alpha=0.3)
        ax3.fill_between(results.index, results['higher_tf_rsi'], self.higher_tf_rsi_oversold, 
                        where=results['higher_tf_rsi'] < self.higher_tf_rsi_oversold, 
                        color='green', alpha=0.3)
        ax3.set_title('Higher Timeframe RSI')
        ax3.set_ylabel('RSI')
        ax3.set_ylim(0, 100)
        ax3.grid(True)
        
        # Plot 4: Equity Curve
        ax4 = plt.subplot(4, 1, 4)
        ax4.plot(results.index, results['equity'], color='blue', linewidth=1)
        ax4.set_title('Equity Curve')
        ax4.set_ylabel('Equity')
        ax4.grid(True)
        
        plt.tight_layout()
        plt.show()
        
    def print_performance_summary(self, metrics):
        """
        Print performance summary
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
        print(f"Average Trade: ${metrics['avg_trade']:.2f}")
        print(f"Average Win: ${metrics['avg_win']:.2f}")
        print(f"Average Loss: ${metrics['avg_loss']:.2f}")
        print(f"Number of Trades: {metrics['num_trades']}")
        
        print("\nExit Reasons:")
        for reason, count in metrics['exit_reasons'].items():
            if pd.notna(reason):
                print(f"  {reason}: {count}")
        
        print("=" * 50)

def generate_market_data(days=1000, trend_cycles=5, volatility_cycles=3, seed=None):
    """
    Generate synthetic market data with trends, volatility regimes, and realistic properties
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Generate dates
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    dates = pd.date_range(start=start_date, end=end_date, periods=days)
    
    # Generate trend component
    trend_period = days / trend_cycles
    t = np.arange(days)
    trend = 100 + 20 * np.sin(2 * np.pi * t / trend_period)
    
    # Generate volatility regimes
    vol_period = days / volatility_cycles
    volatility = 0.5 + 1.5 * (0.5 + 0.5 * np.sin(2 * np.pi * t / vol_period))
    
    # Generate daily returns with changing volatility
    daily_returns = np.random.normal(0, 0.01, days) * volatility
    
    # Add momentum effects
    momentum = np.zeros(days)
    momentum_strength = 0.3
    for i in range(1, days):
        momentum[i] = momentum[i-1] * 0.95 + daily_returns[i-1] * momentum_strength
        daily_returns[i] += momentum[i]
    
    # Generate price series
    prices = trend * np.cumprod(1 + daily_returns)
    
    # Generate realistic OHLC data
    daily_range_factor = 0.5 + 0.5 * volatility  # Higher volatility = wider daily range
    
    opens = np.zeros(days)
    highs = np.zeros(days)
    lows = np.zeros(days)
    closes = prices
    
    # First day
    opens[0] = prices[0] * (1 - 0.005 * daily_range_factor[0] * np.random.rand())
    intraday_vol = prices[0] * 0.015 * daily_range_factor[0]
    highs[0] = max(opens[0], closes[0]) + intraday_vol * np.random.rand()
    lows[0] = min(opens[0], closes[0]) - intraday_vol * np.random.rand()
    
    # Remaining days
    for i in range(1, days):
        # Open is based on previous close with a small gap
        gap = np.random.normal(0, 0.003) * daily_range_factor[i]
        opens[i] = closes[i-1] * (1 + gap)
        
        # Intraday volatility
        intraday_vol = closes[i] * 0.015 * daily_range_factor[i]
        
        # High and low based on open, close, and intraday volatility
        highs[i] = max(opens[i], closes[i]) + intraday_vol * np.random.rand()
        lows[i] = min(opens[i], closes[i]) - intraday_vol * np.random.rand()
        
        # Ensure high >= close >= low
        highs[i] = max(highs[i], closes[i])
        lows[i] = min(lows[i], closes[i])
    
    # Generate volume data
    base_volume = 1000000
    volume_volatility = 0.4
    volume = base_volume * (1 + np.random.lognormal(0, volume_volatility, days))
    
    # Volume tends to be higher on trend changes and high volatility days
    trend_change = np.abs(np.diff(np.sign(np.diff(trend)), prepend=[0, 0]))
    volatility_factor = (volatility - volatility.min()) / (volatility.max() - volatility.min())
    volume = volume * (1 + trend_change * 2) * (1 + volatility_factor)
    
    # Create DataFrame
    df = pd.DataFrame({
        'open': opens,
        'high': highs,
        'low': lows,
        'close': closes,
        'volume': volume
    }, index=dates)
    
    return df

def run_strategy_test(data=None, strategy_params=None):
    """
    Run a complete test of the strategy
    """
    # Generate data if not provided
    if data is None:
        print("Generating synthetic market data...")
        data = generate_market_data(days=1000, trend_cycles=5, volatility_cycles=3, seed=42)
    
    # Create strategy with default or custom parameters
    if strategy_params is None:
        strategy = DynamicVolatilityTrackingStrategy()
    else:
        strategy = DynamicVolatilityTrackingStrategy(**strategy_params)
    
    # Run backtest
    print("Running backtest...")
    results = strategy.backtest(data)
    
    # Calculate performance metrics
    metrics = strategy.calculate_performance_metrics(results)
    
    # Print and plot results
    strategy.print_performance_summary(metrics)
    strategy.plot_results(results)
    
    return strategy, results, metrics

def simplified_strategy_test():
    """
    Run a simplified version of the strategy test
    """
    # Generate synthetic data
    print("Generating synthetic market data...")
    data = generate_market_data(days=500, trend_cycles=3, volatility_cycles=2, seed=42)
    
    # Create strategy with default parameters
    strategy = DynamicVolatilityTrackingStrategy()
    
    # Run backtest
    print("Running backtest...")
    results = strategy.backtest(data)
    
    # Calculate performance metrics
    metrics = strategy.calculate_performance_metrics(results)
    
    # Print results
    strategy.print_performance_summary(metrics)
    
    # Plot results
    strategy.plot_results(results)
    
    # Return results for further analysis if needed
    return strategy, results, metrics

# Run the simplified test
if __name__ == "__main__":
    strategy, results, metrics = simplified_strategy_test()