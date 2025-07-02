import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

class VolatilityClusterReversionStrategy:
    """
    Implementation of the Volatility Cluster Reversion strategy
    
    This strategy identifies clusters of high-volatility days and takes contrarian positions
    after the cluster, with trend filter confirmation and ATR-based trailing stop loss.
    """
    
    def __init__(self, 
                 vol_window=7, 
                 vol_stats_window=30, 
                 vol_cluster_threshold_factor=1.0,
                 vol_cluster_days_trigger=3,
                 trend_filter_sma_window=30,
                 atr_window_sl=14,
                 atr_multiplier_sl=2.0,
                 trading_days_per_year=252):
        """
        Initialize the strategy with parameters
        
        Parameters:
        -----------
        vol_window : int
            Window for historical volatility calculation (default: 7)
        vol_stats_window : int
            Window for volatility statistics (mean, std) calculation (default: 30)
        vol_cluster_threshold_factor : float
            Factor to multiply std_hist_vol for high volatility threshold (default: 1.0)
        vol_cluster_days_trigger : int
            Number of consecutive high-volatility days to trigger a signal (default: 3)
        trend_filter_sma_window : int
            Window for trend filter SMA calculation (default: 30)
        atr_window_sl : int
            Window for ATR calculation for stop loss (default: 14)
        atr_multiplier_sl : float
            Multiplier for ATR-based stop loss (default: 2.0)
        trading_days_per_year : int
            Number of trading days per year for annualization (default: 252)
        """
        self.vol_window = vol_window
        self.vol_stats_window = vol_stats_window
        self.vol_cluster_threshold_factor = vol_cluster_threshold_factor
        self.vol_cluster_days_trigger = vol_cluster_days_trigger
        self.trend_filter_sma_window = trend_filter_sma_window
        self.atr_window_sl = atr_window_sl
        self.atr_multiplier_sl = atr_multiplier_sl
        self.trading_days_per_year = trading_days_per_year
    
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
        
        # Calculate daily returns
        df['Daily_Return'] = df['close'].pct_change()
        
        # 1. Historical Volatility (annualized)
        hist_vol_col = f"Hist_Vol_{self.vol_window}d"
        df[hist_vol_col] = df['Daily_Return'].rolling(window=self.vol_window).std() * np.sqrt(self.trading_days_per_year)
        
        # 2. Rolling Mean and Standard Deviation of Historical Volatility
        mean_hist_vol_col = f"Mean_Hist_Vol_{self.vol_stats_window}d"
        std_hist_vol_col = f"Std_Hist_Vol_{self.vol_stats_window}d"
        df[mean_hist_vol_col] = df[hist_vol_col].rolling(window=self.vol_stats_window).mean()
        df[std_hist_vol_col] = df[hist_vol_col].rolling(window=self.vol_stats_window).std()
        
        # 3. Identify High-Volatility Days
        is_high_vol_day_col = "Is_High_Vol_Day"
        df[is_high_vol_day_col] = df[hist_vol_col] > (df[mean_hist_vol_col] + self.vol_cluster_threshold_factor * df[std_hist_vol_col])
        
        # 4. Count Consecutive High-Volatility Days
        # Create groups based on changes in Is_High_Vol_Day status
        df['High_Vol_Group_ID'] = (df[is_high_vol_day_col] != df[is_high_vol_day_col].shift()).cumsum()
        
        # Calculate cumulative count within each group of consecutive high-volatility days
        consecutive_high_vol_col = "Consecutive_High_Vol_Days"
        df[consecutive_high_vol_col] = df.groupby('High_Vol_Group_ID').cumcount() + 1
        
        # If it's not a high-vol day, the consecutive count should be 0
        df.loc[~df[is_high_vol_day_col], consecutive_high_vol_col] = 0
        
        # 5. Trend Filter SMA
        trend_filter_sma_col = f"SMA_Trend_{self.trend_filter_sma_window}d"
        df[trend_filter_sma_col] = df['close'].rolling(window=self.trend_filter_sma_window).mean()
        
        # 6. ATR for Stop Loss
        atr_col_name_sl = f"ATR_{self.atr_window_sl}d_SL"
        df['H-L_sl'] = df['high'] - df['low']
        df['H-C1_sl'] = abs(df['high'] - df['close'].shift(1))
        df['L-C1_sl'] = abs(df['low'] - df['close'].shift(1))
        df['TR_sl'] = df[['H-L_sl', 'H-C1_sl', 'L-C1_sl']].max(axis=1)
        df[atr_col_name_sl] = df['TR_sl'].rolling(window=self.atr_window_sl).mean()
        
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
            Data with added backtest result columns
        """
        # Calculate indicators
        df_analysis = self.calculate_indicators(data)
        
        # Add columns for strategy tracking
        df_analysis['position'] = 0  # 0: flat, 1: long, -1: short
        df_analysis['entry_price'] = np.nan
        df_analysis['stop_loss'] = np.nan
        df_analysis['equity'] = 100.0  # Starting equity (100%)
        df_analysis['daily_return'] = 0.0
        df_analysis['signal'] = 0  # 0: no signal, 1: long signal, -1: short signal
        df_analysis['exit_reason'] = None
        
        # Variables for position tracking
        active_position = 0
        entry_price = 0.0
        stop_loss = 0.0
        equity = 100.0
        
        # Main backtesting loop
        for i in range(self.vol_stats_window + self.vol_window + self.trend_filter_sma_window + 1, len(df_analysis)):
            prev_idx = df_analysis.index[i-1]
            
            # Get today's values
            today_open = df_analysis['open'].iloc[i]
            today_high = df_analysis['high'].iloc[i]
            today_low = df_analysis['low'].iloc[i]
            today_close = df_analysis['close'].iloc[i]
            today_atr_sl = df_analysis[f"ATR_{self.atr_window_sl}d_SL"].iloc[i]
            
            # Get previous day's values
            prev_close = df_analysis['close'].iloc[i-1]
            prev_consecutive_high_vol = df_analysis['Consecutive_High_Vol_Days'].iloc[i-1]
            prev_trend_filter_sma = df_analysis[f"SMA_Trend_{self.trend_filter_sma_window}d"].iloc[i-1]
            prev_atr_sl = df_analysis[f"ATR_{self.atr_window_sl}d_SL"].iloc[i-1]
            
            # Default values for today
            df_analysis['position'].iloc[i] = active_position
            df_analysis['entry_price'].iloc[i] = entry_price if active_position != 0 else np.nan
            df_analysis['stop_loss'].iloc[i] = stop_loss if active_position != 0 else np.nan
            df_analysis['equity'].iloc[i] = equity
            
            # Check for stop loss hit
            if active_position == 1:  # Long position
                if today_low <= stop_loss:
                    # Stop loss hit for long position
                    trade_pnl = (stop_loss / entry_price) - 1.0
                    equity *= (1.0 + trade_pnl)
                    df_analysis['daily_return'].iloc[i] = trade_pnl
                    df_analysis['equity'].iloc[i] = equity
                    df_analysis['exit_reason'].iloc[i] = 'Stop Loss'
                    
                    # Reset position
                    active_position = 0
                    entry_price = 0.0
                    stop_loss = 0.0
                    df_analysis['position'].iloc[i] = active_position
                    df_analysis['entry_price'].iloc[i] = np.nan
                    df_analysis['stop_loss'].iloc[i] = np.nan
                else:
                    # Update trailing stop loss for long position
                    new_stop = today_close - (self.atr_multiplier_sl * today_atr_sl)
                    stop_loss = max(stop_loss, new_stop)
                    df_analysis['stop_loss'].iloc[i] = stop_loss
                    
                    # Calculate daily return for existing position
                    daily_return = (today_close / prev_close) - 1.0
                    df_analysis['daily_return'].iloc[i] = daily_return
                    equity *= (1.0 + daily_return)
                    df_analysis['equity'].iloc[i] = equity
            
            elif active_position == -1:  # Short position
                if today_high >= stop_loss:
                    # Stop loss hit for short position
                    trade_pnl = 1.0 - (stop_loss / entry_price)
                    equity *= (1.0 + trade_pnl)
                    df_analysis['daily_return'].iloc[i] = trade_pnl
                    df_analysis['equity'].iloc[i] = equity
                    df_analysis['exit_reason'].iloc[i] = 'Stop Loss'
                    
                    # Reset position
                    active_position = 0
                    entry_price = 0.0
                    stop_loss = 0.0
                    df_analysis['position'].iloc[i] = active_position
                    df_analysis['entry_price'].iloc[i] = np.nan
                    df_analysis['stop_loss'].iloc[i] = np.nan
                else:
                    # Update trailing stop loss for short position
                    new_stop = today_close + (self.atr_multiplier_sl * today_atr_sl)
                    stop_loss = min(stop_loss if stop_loss > 0 else float('inf'), new_stop)
                    df_analysis['stop_loss'].iloc[i] = stop_loss
                    
                    # Calculate daily return for existing position
                    daily_return = 1.0 - (today_close / prev_close)
                    df_analysis['daily_return'].iloc[i] = daily_return
                    equity *= (1.0 + daily_return)
                    df_analysis['equity'].iloc[i] = equity
            
            # Check for new entry signals if we don't have an active position
            if active_position == 0 and prev_consecutive_high_vol == self.vol_cluster_days_trigger:
                # A cluster of 'vol_cluster_days_trigger' days just ended.
                # Determine price direction during the cluster.
                idx_day_before_cluster_start_relative_to_i = i - 1 - self.vol_cluster_days_trigger
                potential_trade_direction = 0
                trade_allowed = False
                
                if idx_day_before_cluster_start_relative_to_i >= 0:  # Ensure valid index
                    day_before_cluster_starts_idx = df_analysis.index[idx_day_before_cluster_start_relative_to_i]
                    price_at_cluster_end = df_analysis.at[prev_idx, 'close']
                    price_before_cluster = df_analysis.at[day_before_cluster_starts_idx, 'close']
                    
                    if pd.notna(price_before_cluster) and pd.notna(price_at_cluster_end):
                        if price_at_cluster_end < price_before_cluster:
                            potential_trade_direction = 1  # Price fell during cluster, go long
                        elif price_at_cluster_end > price_before_cluster:
                            potential_trade_direction = -1  # Price rose during cluster, go short
                
                # Apply Trend Filter
                if potential_trade_direction == 1 and prev_close > prev_trend_filter_sma:
                    trade_allowed = True  # Long reversion aligned with uptrend
                elif potential_trade_direction == -1 and prev_close < prev_trend_filter_sma:
                    trade_allowed = True  # Short reversion aligned with downtrend
                
                if trade_allowed and potential_trade_direction != 0:
                    # Enter new position
                    active_position = potential_trade_direction
                    entry_price = today_open
                    df_analysis['position'].iloc[i] = active_position
                    df_analysis['entry_price'].iloc[i] = entry_price
                    df_analysis['signal'].iloc[i] = active_position
                    
                    # Set initial stop loss
                    if active_position == 1:  # Long
                        stop_loss = entry_price - (self.atr_multiplier_sl * prev_atr_sl)
                        # Calculate return for the entry day
                        daily_return = (today_close / entry_price) - 1.0
                        # Update stop loss based on end-of-day price
                        new_stop = today_close - (self.atr_multiplier_sl * today_atr_sl)
                        stop_loss = max(stop_loss, new_stop)
                    else:  # Short
                        stop_loss = entry_price + (self.atr_multiplier_sl * prev_atr_sl)
                        # Calculate return for the entry day
                        daily_return = 1.0 - (today_close / entry_price)
                        # Update stop loss based on end-of-day price
                        new_stop = today_close + (self.atr_multiplier_sl * today_atr_sl)
                        stop_loss = min(stop_loss, new_stop)
                    
                    df_analysis['stop_loss'].iloc[i] = stop_loss
                    df_analysis['daily_return'].iloc[i] = daily_return
                    equity *= (1.0 + daily_return)
                    df_analysis['equity'].iloc[i] = equity
                
            # End of main loop
        
        # Calculate cumulative returns
        df_analysis['cumulative_return'] = (1 + df_analysis['daily_return']).cumprod() - 1
        
        return df_analysis
    
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
        # Extract relevant data
        daily_returns = results['daily_return'].dropna()
        equity_curve = results['equity']
        
        # Calculate metrics
        total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1
        
        # Calculate annual return (assuming 252 trading days per year)
        days = (results.index[-1] - results.index[0]).days if isinstance(results.index, pd.DatetimeIndex) else len(results)
        years = max(days / self.trading_days_per_year, 0.01)  # Avoid division by zero
        annual_return = (1 + total_return) ** (1 / years) - 1
        
        # Calculate volatility
        annual_volatility = daily_returns.std() * np.sqrt(self.trading_days_per_year)
        
        # Calculate Sharpe ratio (assuming risk-free rate = 0)
        sharpe_ratio = annual_return / annual_volatility if annual_volatility != 0 else 0
        
        # Calculate max drawdown
        peak = equity_curve.cummax()
        drawdown = (equity_curve - peak) / peak
        max_drawdown = drawdown.min()
        
        # Calculate win rate and profit factor
        trades = results[results['exit_reason'].notna()]
        signals = results[results['signal'] != 0]
        
        total_trades = len(signals)
        
        # Calculate P&L for each trade
        trade_pnl = []
        entry_dates = []
        exit_dates = []
        trade_direction = []
        current_position = 0
        entry_date = None
        entry_price = 0
        
        for idx, row in results.iterrows():
            if row['signal'] != 0 and current_position == 0:
                current_position = row['signal']
                entry_date = idx
                entry_price = row['entry_price']
            elif row['exit_reason'] is not None and current_position != 0:
                exit_date = idx
                exit_price = row['close']
                
                if current_position == 1:  # Long
                    pnl = (exit_price / entry_price) - 1
                else:  # Short
                    pnl = 1 - (exit_price / entry_price)
                
                trade_pnl.append(pnl)
                entry_dates.append(entry_date)
                exit_dates.append(exit_date)
                trade_direction.append('Long' if current_position == 1 else 'Short')
                
                current_position = 0
        
        trade_df = pd.DataFrame({
            'entry_date': entry_dates,
            'exit_date': exit_dates,
            'direction': trade_direction,
            'pnl': trade_pnl
        })
        
        winning_trades = trade_df[trade_df['pnl'] > 0]
        losing_trades = trade_df[trade_df['pnl'] <= 0]
        
        win_rate = len(winning_trades) / len(trade_df) if len(trade_df) > 0 else 0
        
        # Calculate profit factor
        gross_profit = winning_trades['pnl'].sum() if len(winning_trades) > 0 else 0
        gross_loss = abs(losing_trades['pnl'].sum()) if len(losing_trades) > 0 else 1  # Avoid division by zero
        profit_factor = gross_profit / gross_loss
        
        # Calculate average win and loss
        avg_win = winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0
        avg_loss = losing_trades['pnl'].mean() if len(losing_trades) > 0 else 0
        
        # Calculate average trade
        avg_trade = trade_df['pnl'].mean() if len(trade_df) > 0 else 0
        
        # Calculate average holding period
        if isinstance(trade_df['entry_date'].iloc[0], pd.Timestamp) and isinstance(trade_df['exit_date'].iloc[0], pd.Timestamp):
            trade_df['holding_period'] = (pd.to_datetime(trade_df['exit_date']) - pd.to_datetime(trade_df['entry_date'])).dt.days
        else:
            # If dates are not timestamps, calculate index difference
            trade_df['holding_period'] = [results.index.get_loc(exit_date) - results.index.get_loc(entry_date) for entry_date, exit_date in zip(trade_df['entry_date'], trade_df['exit_date'])]
        
        avg_holding_period = trade_df['holding_period'].mean() if len(trade_df) > 0 else 0
        
        # Calculate percentage of long vs. short trades
        long_trades = trade_df[trade_df['direction'] == 'Long']
        short_trades = trade_df[trade_df['direction'] == 'Short']
        
        pct_long_trades = len(long_trades) / len(trade_df) if len(trade_df) > 0 else 0
        pct_short_trades = len(short_trades) / len(trade_df) if len(trade_df) > 0 else 0
        
        # Calculate win rate for long and short trades
        long_win_rate = len(long_trades[long_trades['pnl'] > 0]) / len(long_trades) if len(long_trades) > 0 else 0
        short_win_rate = len(short_trades[short_trades['pnl'] > 0]) / len(short_trades) if len(short_trades) > 0 else 0
        
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
            'avg_trade': avg_trade,
            'avg_holding_period': avg_holding_period,
            'total_trades': total_trades,
            'pct_long_trades': pct_long_trades,
            'pct_short_trades': pct_short_trades,
            'long_win_rate': long_win_rate,
            'short_win_rate': short_win_rate,
            'trades': trade_df
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
        
        # Plot 1: Price chart with signals and trailing stops
        ax1 = plt.subplot(4, 1, 1)
        
        # Plot price
        ax1.plot(results.index, results['close'], label='Close Price', color='black', linewidth=1)
        
        # Plot trend filter SMA
        ax1.plot(results.index, results[f"SMA_Trend_{self.trend_filter_sma_window}d"], 
                label=f"SMA({self.trend_filter_sma_window})", color='blue', linestyle='--', alpha=0.7)
        
        # Mark high volatility days
        high_vol_days = results[results['Is_High_Vol_Day']]
        for idx in high_vol_days.index:
            ax1.axvspan(idx, idx, color='gray', alpha=0.3)
        
        # Mark entry and exit points
        long_entries = results[results['signal'] == 1]
        short_entries = results[results['signal'] == -1]
        exits = results[results['exit_reason'].notna()]
        
        ax1.scatter(long_entries.index, long_entries['entry_price'], marker='^', color='green', s=100, label='Long Entry')
        ax1.scatter(short_entries.index, short_entries['entry_price'], marker='v', color='red', s=100, label='Short Entry')
        ax1.scatter(exits.index, exits['close'], marker='x', color='black', s=100, label='Exit')
        
        # Plot stop loss levels
        for i, row in results.iterrows():
            if row['position'] != 0 and not np.isnan(row['stop_loss']):
                ax1.plot([i, i], [row['entry_price'], row['stop_loss']], 'r--', alpha=0.5)
        
        ax1.set_title('Price Chart with Signals and Trailing Stops')
        ax1.set_ylabel('Price')
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Volatility and High Volatility Days
        ax2 = plt.subplot(4, 1, 2)
        
        hist_vol_col = f"Hist_Vol_{self.vol_window}d"
        mean_hist_vol_col = f"Mean_Hist_Vol_{self.vol_stats_window}d"
        std_hist_vol_col = f"Std_Hist_Vol_{self.vol_stats_window}d"
        
        ax2.plot(results.index, results[hist_vol_col], label=f"Historical Volatility ({self.vol_window}d)", color='blue')
        ax2.plot(results.index, results[mean_hist_vol_col], 
                label=f"Mean Historical Volatility ({self.vol_stats_window}d)", color='green')
        ax2.plot(results.index, results[mean_hist_vol_col] + self.vol_cluster_threshold_factor * results[std_hist_vol_col], 
                label=f"High Volatility Threshold", color='red', linestyle='--')
        
        # Mark high volatility days
        for idx in high_vol_days.index:
            ax2.axvspan(idx, idx, color='gray', alpha=0.3)
        
        ax2.set_title('Historical Volatility and High Volatility Days')
        ax2.set_ylabel('Volatility')
        ax2.legend(loc='upper left')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Consecutive High Volatility Days
        ax3 = plt.subplot(4, 1, 3)
        
        consecutive_high_vol_col = "Consecutive_High_Vol_Days"
        ax3.bar(results.index, results[consecutive_high_vol_col], label='Consecutive High Volatility Days', color='purple')
        ax3.axhline(y=self.vol_cluster_days_trigger, color='red', linestyle='--', 
                   label=f"Trigger Threshold ({self.vol_cluster_days_trigger} days)")
        
        # Mark trade signals
        for idx in long_entries.index:
            ax3.axvspan(idx, idx, color='green', alpha=0.5)
        for idx in short_entries.index:
            ax3.axvspan(idx, idx, color='red', alpha=0.5)
        
        ax3.set_title('Consecutive High Volatility Days')
        ax3.set_ylabel('Days')
        ax3.legend(loc='upper left')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Equity Curve
        ax4 = plt.subplot(4, 1, 4)
        
        ax4.plot(results.index, results['equity'], label='Strategy Equity', color='blue')
        
        # Mark trade entries and exits on equity curve
        for idx in long_entries.index:
            ax4.axvspan(idx, idx, color='green', alpha=0.5)
        for idx in short_entries.index:
            ax4.axvspan(idx, idx, color='red', alpha=0.5)
        for idx in exits.index:
            ax4.axvspan(idx, idx, color='black', alpha=0.5)
        
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
        metrics = self.calculate_performance_metrics(results)
        trade_df = metrics['trades']
        
        if len(trade_df) > 0:
            plt.figure(figsize=(16, 12))
            
            # Plot 1: Trade P&L Distribution
            ax1 = plt.subplot(2, 2, 1)
            sns.histplot(trade_df['pnl'], bins=20, kde=True, ax=ax1)
            ax1.axvline(x=0, color='black', linestyle='--', alpha=0.7)
            ax1.set_title('Trade P&L Distribution')
            ax1.set_xlabel('P&L')
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Cumulative P&L
            ax2 = plt.subplot(2, 2, 2)
            trade_df['cumulative_pnl'] = trade_df['pnl'].cumsum()
            ax2.plot(range(len(trade_df)), trade_df['cumulative_pnl'], color='blue')
            ax2.set_title('Cumulative P&L by Trade')
            ax2.set_ylabel('Cumulative P&L')
            ax2.set_xlabel('Trade #')
            ax2.grid(True, alpha=0.3)
            
            # Plot 3: P&L by Trade Direction
            ax3 = plt.subplot(2, 2, 3)
            sns.boxplot(x='direction', y='pnl', data=trade_df, ax=ax3)
            ax3.axhline(y=0, color='black', linestyle='--', alpha=0.7)
            ax3.set_title('P&L by Trade Direction')
            ax3.set_ylabel('P&L')
            ax3.grid(True, alpha=0.3)
            
            # Plot 4: Holding Period vs P&L
            ax4 = plt.subplot(2, 2, 4)
            sns.scatterplot(x='holding_period', y='pnl', hue='direction', data=trade_df, ax=ax4)
            ax4.axhline(y=0, color='black', linestyle='--', alpha=0.7)
            ax4.set_title('Holding Period vs P&L')
            ax4.set_xlabel('Holding Period (days)')
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
        print("VOLATILITY CLUSTER REVERSION STRATEGY PERFORMANCE")
        print("=" * 50)
        
        print(f"Total Return: {metrics['total_return']:.2%}")
        print(f"Annual Return: {metrics['annual_return']:.2%}")
        print(f"Annual Volatility: {metrics['annual_volatility']:.2%}")
        print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        print(f"Maximum Drawdown: {metrics['max_drawdown']:.2%}")
        print(f"Win Rate: {metrics['win_rate']:.2%}")
        print(f"Profit Factor: {metrics['profit_factor']:.2f}")
        print(f"Average Win: {metrics['avg_win']:.2%}")
        print(f"Average Loss: {metrics['avg_loss']:.2%}")
        print(f"Average Trade: {metrics['avg_trade']:.2%}")
        print(f"Average Holding Period: {metrics['avg_holding_period']:.2f} days")
        print(f"Total Trades: {metrics['total_trades']}")
        print(f"Long Trades: {metrics['pct_long_trades']:.2%} (Win Rate: {metrics['long_win_rate']:.2%})")
        print(f"Short Trades: {metrics['pct_short_trades']:.2%} (Win Rate: {metrics['short_win_rate']:.2%})")
        
        print("=" * 50)

def generate_forex_data(days=1000, seed=42):
    """
    Generate synthetic forex data with volatility clusters
    
    Parameters:
    -----------
    days : int
        Number of days to generate
    seed : int
        Random seed for reproducibility
        
    Returns:
    --------
    DataFrame
        Synthetic forex data with OHLC columns
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Generate dates
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    dates = pd.date_range(start=start_date, end=end_date, periods=days)
    
    # Initialize with a base price
    base_price = 1.15  # Example EUR/USD starting price
    
    # Generate price series with trends and volatility clusters
    prices = []
    volatilities = []
    
    # Parameters for simulation
    trend_periods = 20  # Length of trend periods
    trend_changes = days // trend_periods  # Number of trend changes
    vol_cluster_periods = 10  # Average length of volatility cluster
    vol_cluster_probability = 0.05  # Probability of starting a volatility cluster
    
    # Current state
    in_vol_cluster = False
    vol_cluster_days_left = 0
    current_trend = 0.0
    current_price = base_price
    
    # Generate price movement with volatility clusters
    for i in range(days):  # Changed to generate exactly 'days' number of prices
        # Check for trend change
        if i % trend_periods == 0:
            current_trend = np.random.normal(0, 0.0005)
        
        # Check for volatility cluster start or end
        if not in_vol_cluster and np.random.random() < vol_cluster_probability:
            in_vol_cluster = True
            vol_cluster_days_left = np.random.randint(3, vol_cluster_periods)
        
        if in_vol_cluster:
            vol_cluster_days_left -= 1
            if vol_cluster_days_left <= 0:
                in_vol_cluster = False
        
        # Determine volatility based on whether we're in a cluster
        if in_vol_cluster:
            volatility = np.random.uniform(0.005, 0.015)  # High volatility
        else:
            volatility = np.random.uniform(0.001, 0.005)  # Normal volatility
        
        volatilities.append(volatility)
        
        # Calculate price movement
        price_change = current_trend + np.random.normal(0, volatility)
        
        # Calculate new price
        current_price = current_price * (1 + price_change)
        prices.append(current_price)
    
    # Generate OHLC data
    forex_data = pd.DataFrame(index=dates)
    forex_data['close'] = prices  # Now prices array has exactly 'days' elements
    
    # Generate open, high, low
    forex_data['open'] = np.zeros(days)
    forex_data['high'] = np.zeros(days)
    forex_data['low'] = np.zeros(days)
    
    # First day
    forex_data['open'].iloc[0] = prices[0] * (1 - 0.001 * np.random.random())
    intraday_vol = prices[0] * volatilities[0]
    forex_data['high'].iloc[0] = max(forex_data['open'].iloc[0], forex_data['close'].iloc[0]) + intraday_vol * np.random.random()
    forex_data['low'].iloc[0] = min(forex_data['open'].iloc[0], forex_data['close'].iloc[0]) - intraday_vol * np.random.random()
    
    # Remaining days
    for i in range(1, days):
        # Open is close of previous day with small gap
        gap = np.random.normal(0, 0.0005)
        forex_data['open'].iloc[i] = forex_data['close'].iloc[i-1] * (1 + gap)
        
        intraday_vol = forex_data['close'].iloc[i] * volatilities[i]
        
        # Generate high and low
        if forex_data['open'].iloc[i] <= forex_data['close'].iloc[i]:  # Up day
            forex_data['high'].iloc[i] = forex_data['close'].iloc[i] + intraday_vol * np.random.random()
            forex_data['low'].iloc[i] = forex_data['open'].iloc[i] - intraday_vol * np.random.random()
        else:  # Down day
            forex_data['high'].iloc[i] = forex_data['open'].iloc[i] + intraday_vol * np.random.random()
            forex_data['low'].iloc[i] = forex_data['close'].iloc[i] - intraday_vol * np.random.random()
    
    # Ensure high/low are actually high/low
    forex_data['high'] = np.maximum.reduce([forex_data['high'], forex_data['open'], forex_data['close']])
    forex_data['low'] = np.minimum.reduce([forex_data['low'], forex_data['open'], forex_data['close']])
    
    return forex_data

def test_strategy():
    """
    Test the Volatility Cluster Reversion strategy with default parameters
    
    Returns:
    --------
    tuple
        (strategy, results, metrics)
    """
    # Generate synthetic data
    print("Generating synthetic forex data...")
    data = generate_forex_data(days=1000, seed=42)
    
    # Create strategy instance
    strategy = VolatilityClusterReversionStrategy(
        vol_window=7,
        vol_stats_window=30,
        vol_cluster_threshold_factor=1.0,
        vol_cluster_days_trigger=3,
        trend_filter_sma_window=30,
        atr_window_sl=14,
        atr_multiplier_sl=2.0,
        trading_days_per_year=252
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
    print("Generating synthetic forex data for optimization...")
    data = generate_forex_data(days=1000, seed=42)
    
    # Parameters to test
    vol_windows = [5, 7, 10]
    vol_cluster_threshold_factors = [0.8, 1.0, 1.2]
    vol_cluster_days_triggers = [2, 3, 4]
    trend_filter_sma_windows = [20, 30, 50]
    atr_multiplier_sls = [1.5, 2.0, 2.5]
    
    # Store results
    results = []
    
    # Calculate total combinations
    total_combinations = len(vol_windows) * len(vol_cluster_threshold_factors) * len(vol_cluster_days_triggers) * len(trend_filter_sma_windows) * len(atr_multiplier_sls)
    current_combination = 0
    
    # Test parameter combinations
    for vol_window in vol_windows:
        for vol_cluster_threshold_factor in vol_cluster_threshold_factors:
            for vol_cluster_days_trigger in vol_cluster_days_triggers:
                for trend_filter_sma_window in trend_filter_sma_windows:
                    for atr_multiplier_sl in atr_multiplier_sls:
                        current_combination += 1
                        print(f"Testing combination {current_combination}/{total_combinations}...")
                        
                        # Create strategy instance with current parameters
                        strategy = VolatilityClusterReversionStrategy(
                            vol_window=vol_window,
                            vol_stats_window=30,  # Fixed
                            vol_cluster_threshold_factor=vol_cluster_threshold_factor,
                            vol_cluster_days_trigger=vol_cluster_days_trigger,
                            trend_filter_sma_window=trend_filter_sma_window,
                            atr_window_sl=14,  # Fixed
                            atr_multiplier_sl=atr_multiplier_sl,
                            trading_days_per_year=252  # Fixed
                        )
                        
                        # Run backtest
                        backtest_results = strategy.backtest(data)
                        
                        # Calculate performance metrics
                        metrics = strategy.calculate_performance_metrics(backtest_results)
                        
                        # Store results
                        result = {
                            'vol_window': vol_window,
                            'vol_cluster_threshold_factor': vol_cluster_threshold_factor,
                            'vol_cluster_days_trigger': vol_cluster_days_trigger,
                            'trend_filter_sma_window': trend_filter_sma_window,
                            'atr_multiplier_sl': atr_multiplier_sl,
                            'total_return': metrics['total_return'],
                            'annual_return': metrics['annual_return'],
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
    print("\nTop 5 Parameter Combinations by Sharpe Ratio:")
    print(results_df.head(5))
    
    # Plot parameter impact
    plt.figure(figsize=(15, 15))
    
    # Plot impact of vol_window
    plt.subplot(3, 2, 1)
    sns.boxplot(x='vol_window', y='sharpe_ratio', data=results_df)
    plt.title('Impact of Historical Volatility Window on Sharpe Ratio')
    plt.grid(True, alpha=0.3)
    
    # Plot impact of vol_cluster_threshold_factor
    plt.subplot(3, 2, 2)
    sns.boxplot(x='vol_cluster_threshold_factor', y='sharpe_ratio', data=results_df)
    plt.title('Impact of Volatility Threshold Factor on Sharpe Ratio')
    plt.grid(True, alpha=0.3)
    
    # Plot impact of vol_cluster_days_trigger
    plt.subplot(3, 2, 3)
    sns.boxplot(x='vol_cluster_days_trigger', y='sharpe_ratio', data=results_df)
    plt.title('Impact of Cluster Days Trigger on Sharpe Ratio')
    plt.grid(True, alpha=0.3)
    
    # Plot impact of trend_filter_sma_window
    plt.subplot(3, 2, 4)
    sns.boxplot(x='trend_filter_sma_window', y='sharpe_ratio', data=results_df)
    plt.title('Impact of Trend Filter SMA Window on Sharpe Ratio')
    plt.grid(True, alpha=0.3)
    
    # Plot impact of atr_multiplier_sl
    plt.subplot(3, 2, 5)
    sns.boxplot(x='atr_multiplier_sl', y='sharpe_ratio', data=results_df)
    plt.title('Impact of ATR Multiplier on Sharpe Ratio')
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

def test_enhanced_strategy():
    """
    Test an enhanced version of the Volatility Cluster Reversion strategy
    
    Returns:
    --------
    tuple
        (enhanced_strategy, enhanced_results, enhanced_metrics)
    """
    # Generate synthetic data
    print("Generating synthetic forex data for enhanced strategy test...")
    data = generate_forex_data(days=1000, seed=42)
    
    # Create an enhanced strategy class
    class EnhancedVolatilityClusterReversionStrategy(VolatilityClusterReversionStrategy):
        def __init__(self, vol_threshold_multiplier=1.0, max_holding_days=10, rsi_window=14, rsi_exit_threshold=70, **kwargs):
            super().__init__(**kwargs)
            self.vol_threshold_multiplier = vol_threshold_multiplier
            self.max_holding_days = max_holding_days
            self.rsi_window = rsi_window
            self.rsi_exit_threshold = rsi_exit_threshold
        
        def calculate_indicators(self, data):
            # Call parent method to calculate base indicators
            df = super().calculate_indicators(data)
            
            # Add RSI for additional exit conditions
            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(window=self.rsi_window).mean()
            avg_loss = loss.rolling(window=self.rsi_window).mean()
            rs = avg_gain / avg_loss.replace(0, 0.001)  # Avoid division by zero
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # Add volume filter using simulated volume
            # For forex, we'll simulate volume based on volatility
            df['simulated_volume'] = df['high'] - df['low']
            df['volume_sma'] = df['simulated_volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['simulated_volume'] / df['volume_sma']
            df['volume_filter'] = df['volume_ratio'] > 1.2  # Volume must be 20% above average
            
            # Add dynamic volatility threshold based on market regime
            # Use a higher threshold in high-volatility regimes
            hist_vol_col = f"Hist_Vol_{self.vol_window}d"
            mean_hist_vol_col = f"Mean_Hist_Vol_{self.vol_stats_window}d"
            std_hist_vol_col = f"Std_Hist_Vol_{self.vol_stats_window}d"
            
            df['market_regime'] = np.where(
                df[hist_vol_col] > 2 * df[mean_hist_vol_col],
                'High Volatility',
                np.where(
                    df[hist_vol_col] < 0.5 * df[mean_hist_vol_col],
                    'Low Volatility',
                    'Normal Volatility'
                )
            )
            
            # Adjust volatility threshold based on market regime
            df['vol_threshold_multiplier'] = np.where(
                df['market_regime'] == 'High Volatility',
                self.vol_threshold_multiplier * 1.5,
                np.where(
                    df['market_regime'] == 'Low Volatility',
                    self.vol_threshold_multiplier * 0.8,
                    self.vol_threshold_multiplier
                )
            )
            
            # Update high volatility day definition with dynamic threshold
            df['dynamic_threshold'] = df[mean_hist_vol_col] + df['vol_threshold_multiplier'] * df[std_hist_vol_col]
            df['Is_High_Vol_Day'] = df[hist_vol_col] > df['dynamic_threshold']
            
            # Recalculate consecutive high volatility days with the new definition
            df['High_Vol_Group_ID'] = (df['Is_High_Vol_Day'] != df['Is_High_Vol_Day'].shift()).cumsum()
            df['Consecutive_High_Vol_Days'] = df.groupby('High_Vol_Group_ID').cumcount() + 1
            df.loc[~df['Is_High_Vol_Day'], 'Consecutive_High_Vol_Days'] = 0
            
            return df
        
        def backtest(self, data):
            # Calculate indicators
            df_analysis = self.calculate_indicators(data)
            
            # Add columns for strategy tracking
            df_analysis['position'] = 0  # 0: flat, 1: long, -1: short
            df_analysis['entry_price'] = np.nan
            df_analysis['stop_loss'] = np.nan
            df_analysis['equity'] = 100.0  # Starting equity (100%)
            df_analysis['daily_return'] = 0.0
            df_analysis['signal'] = 0  # 0: no signal, 1: long signal, -1: short signal
            df_analysis['exit_reason'] = None
            df_analysis['holding_days'] = 0
            
            # Variables for position tracking
            active_position = 0
            entry_price = 0.0
            stop_loss = 0.0
            equity = 100.0
            holding_days = 0
            entry_date = None
            
            # Main backtesting loop
            for i in range(self.vol_stats_window + self.vol_window + self.trend_filter_sma_window + 1, len(df_analysis)):
                prev_idx = df_analysis.index[i-1]
                
                # Get today's values
                today_open = df_analysis['open'].iloc[i]
                today_high = df_analysis['high'].iloc[i]
                today_low = df_analysis['low'].iloc[i]
                today_close = df_analysis['close'].iloc[i]
                today_atr_sl = df_analysis[f"ATR_{self.atr_window_sl}d_SL"].iloc[i]
                today_rsi = df_analysis['rsi'].iloc[i]
                
                # Get previous day's values
                prev_close = df_analysis['close'].iloc[i-1]
                prev_consecutive_high_vol = df_analysis['Consecutive_High_Vol_Days'].iloc[i-1]
                prev_trend_filter_sma = df_analysis[f"SMA_Trend_{self.trend_filter_sma_window}d"].iloc[i-1]
                prev_atr_sl = df_analysis[f"ATR_{self.atr_window_sl}d_SL"].iloc[i-1]
                prev_rsi = df_analysis['rsi'].iloc[i-1]
                
                # Default values for today
                df_analysis['position'].iloc[i] = active_position
                df_analysis['entry_price'].iloc[i] = entry_price if active_position != 0 else np.nan
                df_analysis['stop_loss'].iloc[i] = stop_loss if active_position != 0 else np.nan
                df_analysis['equity'].iloc[i] = equity
                
                if active_position != 0:
                    holding_days += 1
                    df_analysis['holding_days'].iloc[i] = holding_days
                
                # Check for exit conditions
                exit_triggered = False
                
                if active_position == 1:  # Long position
                    # Check for stop loss hit
                    if today_low <= stop_loss:
                        # Stop loss hit for long position
                        trade_pnl = (stop_loss / entry_price) - 1.0
                        equity *= (1.0 + trade_pnl)
                        df_analysis['daily_return'].iloc[i] = trade_pnl
                        df_analysis['equity'].iloc[i] = equity
                        df_analysis['exit_reason'].iloc[i] = 'Stop Loss'
                        exit_triggered = True
                    
                    # Check for RSI exit condition
                    elif today_rsi >= self.rsi_exit_threshold:
                        # RSI exit for long position
                        trade_pnl = (today_close / entry_price) - 1.0
                        equity *= (1.0 + trade_pnl)
                        df_analysis['daily_return'].iloc[i] = trade_pnl
                        df_analysis['equity'].iloc[i] = equity
                        df_analysis['exit_reason'].iloc[i] = 'RSI Exit'
                        exit_triggered = True
                    
                    # Check for maximum holding period
                    elif holding_days >= self.max_holding_days:
                        # Time-based exit for long position
                        trade_pnl = (today_close / entry_price) - 1.0
                        equity *= (1.0 + trade_pnl)
                        df_analysis['daily_return'].iloc[i] = trade_pnl
                        df_analysis['equity'].iloc[i] = equity
                        df_analysis['exit_reason'].iloc[i] = 'Time Exit'
                        exit_triggered = True
                    
                    elif not exit_triggered:
                        # Update trailing stop loss for long position
                        new_stop = today_close - (self.atr_multiplier_sl * today_atr_sl)
                        stop_loss = max(stop_loss, new_stop)
                        df_analysis['stop_loss'].iloc[i] = stop_loss
                        
                        # Calculate daily return for existing position
                        daily_return = (today_close / prev_close) - 1.0
                        df_analysis['daily_return'].iloc[i] = daily_return
                        equity *= (1.0 + daily_return)
                        df_analysis['equity'].iloc[i] = equity
                
                elif active_position == -1:  # Short position
                    # Check for stop loss hit
                    if today_high >= stop_loss:
                        # Stop loss hit for short position
                        trade_pnl = 1.0 - (stop_loss / entry_price)
                        equity *= (1.0 + trade_pnl)
                        df_analysis['daily_return'].iloc[i] = trade_pnl
                        df_analysis['equity'].iloc[i] = equity
                        df_analysis['exit_reason'].iloc[i] = 'Stop Loss'
                        exit_triggered = True
                    
                    # Check for RSI exit condition
                    elif today_rsi <= (100 - self.rsi_exit_threshold):
                        # RSI exit for short position
                        trade_pnl = 1.0 - (today_close / entry_price)
                        equity *= (1.0 + trade_pnl)
                        df_analysis['daily_return'].iloc[i] = trade_pnl
                        df_analysis['equity'].iloc[i] = equity
                        df_analysis['exit_reason'].iloc[i] = 'RSI Exit'
                        exit_triggered = True
                    
                    # Check for maximum holding period
                    elif holding_days >= self.max_holding_days:
                        # Time-based exit for short position
                        trade_pnl = 1.0 - (today_close / entry_price)
                        equity *= (1.0 + trade_pnl)
                        df_analysis['daily_return'].iloc[i] = trade_pnl
                        df_analysis['equity'].iloc[i] = equity
                        df_analysis['exit_reason'].iloc[i] = 'Time Exit'
                        exit_triggered = True
                    
                    elif not exit_triggered:
                        # Update trailing stop loss for short position
                        new_stop = today_close + (self.atr_multiplier_sl * today_atr_sl)
                        stop_loss = min(stop_loss if stop_loss > 0 else float('inf'), new_stop)
                        df_analysis['stop_loss'].iloc[i] = stop_loss
                        
                        # Calculate daily return for existing position
                        daily_return = 1.0 - (today_close / prev_close)
                        df_analysis['daily_return'].iloc[i] = daily_return
                        equity *= (1.0 + daily_return)
                        df_analysis['equity'].iloc[i] = equity
                
                # Reset position if exit was triggered
                if exit_triggered:
                    active_position = 0
                    entry_price = 0.0
                    stop_loss = 0.0
                    holding_days = 0
                    df_analysis['position'].iloc[i] = active_position
                    df_analysis['entry_price'].iloc[i] = np.nan
                    df_analysis['stop_loss'].iloc[i] = np.nan
                    df_analysis['holding_days'].iloc[i] = holding_days
                
                # Check for new entry signals if we don't have an active position
                if active_position == 0 and prev_consecutive_high_vol == self.vol_cluster_days_trigger:
                    # A cluster of 'vol_cluster_days_trigger' days just ended.
                    # Determine price direction during the cluster.
                    idx_day_before_cluster_start_relative_to_i = i - 1 - self.vol_cluster_days_trigger
                    potential_trade_direction = 0
                    trade_allowed = False
                    
                    if idx_day_before_cluster_start_relative_to_i >= 0:  # Ensure valid index
                        day_before_cluster_starts_idx = df_analysis.index[idx_day_before_cluster_start_relative_to_i]
                        price_at_cluster_end = df_analysis.at[prev_idx, 'close']
                        price_before_cluster = df_analysis.at[day_before_cluster_starts_idx, 'close']
                        
                        if pd.notna(price_before_cluster) and pd.notna(price_at_cluster_end):
                            if price_at_cluster_end < price_before_cluster:
                                potential_trade_direction = 1  # Price fell during cluster, go long
                            elif price_at_cluster_end > price_before_cluster:
                                potential_trade_direction = -1  # Price rose during cluster, go short
                    
                    # Apply Trend Filter
                    if potential_trade_direction == 1 and prev_close > prev_trend_filter_sma:
                        trade_allowed = True  # Long reversion aligned with uptrend
                    elif potential_trade_direction == -1 and prev_close < prev_trend_filter_sma:
                        trade_allowed = True  # Short reversion aligned with downtrend
                    
                    # Apply Volume Filter
                    volume_filter_passed = df_analysis['volume_filter'].iloc[i-1]
                    
                    if trade_allowed and potential_trade_direction != 0 and volume_filter_passed:
                        # Enter new position
                        active_position = potential_trade_direction
                        entry_price = today_open
                        df_analysis['position'].iloc[i] = active_position
                        df_analysis['entry_price'].iloc[i] = entry_price
                        df_analysis['signal'].iloc[i] = active_position
                        holding_days = 1
                        df_analysis['holding_days'].iloc[i] = holding_days
                        
                        # Set initial stop loss
                        if active_position == 1:  # Long
                            stop_loss = entry_price - (self.atr_multiplier_sl * prev_atr_sl)
                            # Calculate return for the entry day
                            daily_return = (today_close / entry_price) - 1.0
                            # Update stop loss based on end-of-day price
                            new_stop = today_close - (self.atr_multiplier_sl * today_atr_sl)
                            stop_loss = max(stop_loss, new_stop)
                        else:  # Short
                            stop_loss = entry_price + (self.atr_multiplier_sl * prev_atr_sl)
                            # Calculate return for the entry day
                            daily_return = 1.0 - (today_close / entry_price)
                            # Update stop loss based on end-of-day price
                            new_stop = today_close + (self.atr_multiplier_sl * today_atr_sl)
                            stop_loss = min(stop_loss, new_stop)
                        
                        df_analysis['stop_loss'].iloc[i] = stop_loss
                        df_analysis['daily_return'].iloc[i] = daily_return
                        equity *= (1.0 + daily_return)
                        df_analysis['equity'].iloc[i] = equity
                
                # End of main loop
            
            # Calculate cumulative returns
            df_analysis['cumulative_return'] = (1 + df_analysis['daily_return']).cumprod() - 1
            
            return df_analysis
    
    # Create enhanced strategy instance
    enhanced_strategy = EnhancedVolatilityClusterReversionStrategy(
        vol_window=7,
        vol_stats_window=30,
        vol_cluster_threshold_factor=1.0,
        vol_cluster_days_trigger=3,
        trend_filter_sma_window=30,
        atr_window_sl=14,
        atr_multiplier_sl=2.0,
        trading_days_per_year=252,
        vol_threshold_multiplier=1.0,
        max_holding_days=10,
        rsi_window=14,
        rsi_exit_threshold=70
    )
    
    # Run backtest
    print("Running enhanced strategy backtest...")
    enhanced_results = enhanced_strategy.backtest(data)
    
    # Calculate performance metrics
    enhanced_metrics = enhanced_strategy.calculate_performance_metrics(enhanced_results)
    
    # Print performance summary
    print("=" * 50)
    print("ENHANCED VOLATILITY CLUSTER REVERSION STRATEGY PERFORMANCE")
    print("=" * 50)
    print(f"Total Return: {enhanced_metrics['total_return']:.2%}")
    print(f"Annual Return: {enhanced_metrics['annual_return']:.2%}")
    print(f"Annual Volatility: {enhanced_metrics['annual_volatility']:.2%}")
    print(f"Sharpe Ratio: {enhanced_metrics['sharpe_ratio']:.2f}")
    print(f"Maximum Drawdown: {enhanced_metrics['max_drawdown']:.2%}")
    print(f"Win Rate: {enhanced_metrics['win_rate']:.2%}")
    print(f"Profit Factor: {enhanced_metrics['profit_factor']:.2f}")
    print(f"Average Win: {enhanced_metrics['avg_win']:.2%}")
    print(f"Average Loss: {enhanced_metrics['avg_loss']:.2%}")
    print(f"Average Trade: {enhanced_metrics['avg_trade']:.2%}")
    print(f"Average Holding Period: {enhanced_metrics['avg_holding_period']:.2f} days")
    print(f"Total Trades: {enhanced_metrics['total_trades']}")
    print(f"Long Trades: {enhanced_metrics['pct_long_trades']:.2%} (Win Rate: {enhanced_metrics['long_win_rate']:.2%})")
    print(f"Short Trades: {enhanced_metrics['pct_short_trades']:.2%} (Win Rate: {enhanced_metrics['short_win_rate']:.2%})")
    print("=" * 50)
    
    # Compare with original strategy
    original_strategy = VolatilityClusterReversionStrategy(
        vol_window=7,
        vol_stats_window=30,
        vol_cluster_threshold_factor=1.0,
        vol_cluster_days_trigger=3,
        trend_filter_sma_window=30,
        atr_window_sl=14,
        atr_multiplier_sl=2.0,
        trading_days_per_year=252
    )
    original_results = original_strategy.backtest(data)
    original_metrics = original_strategy.calculate_performance_metrics(original_results)
    
    # Plot comparison
    plt.figure(figsize=(15, 8))
    
    plt.plot(enhanced_results.index, enhanced_results['equity'], label='Enhanced Strategy', color='blue')
    plt.plot(original_results.index, original_results['equity'], label='Original Strategy', color='red')
    
    # Plot buy & hold equity for comparison
    initial_capital = 100.0
    buy_hold_equity = initial_capital * (data['close'] / data['close'].iloc[0])
    plt.plot(data.index, buy_hold_equity, label='Buy & Hold', color='gray', alpha=0.5)
    
    plt.title('Enhanced vs Original Strategy Comparison')
    plt.xlabel('Date')
    plt.ylabel('Equity')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Compare metrics
    comparison = pd.DataFrame({
        'Enhanced Strategy': [
            enhanced_metrics['total_return'],
            enhanced_metrics['annual_return'],
            enhanced_metrics['sharpe_ratio'],
            enhanced_metrics['max_drawdown'],
            enhanced_metrics['win_rate'],
            enhanced_metrics['profit_factor'],
            enhanced_metrics['total_trades']
        ],
        'Original Strategy': [
            original_metrics['total_return'],
            original_metrics['annual_return'],
            original_metrics['sharpe_ratio'],
            original_metrics['max_drawdown'],
            original_metrics['win_rate'],
            original_metrics['profit_factor'],
            original_metrics['total_trades']
        ]
    }, index=[
        'Total Return', 
        'Annual Return', 
        'Sharpe Ratio', 
        'Max Drawdown', 
        'Win Rate', 
        'Profit Factor', 
        'Total Trades'
    ])
    
    print("\nStrategy Comparison:")
    print(comparison)
    
    return enhanced_strategy, enhanced_results, enhanced_metrics

# Run the tests
if __name__ == "__main__":
    # Test the strategy with default parameters
    strategy, results, metrics = test_strategy()
    
    # Uncomment to run other tests
    # optimization_results = parameter_optimization()
    # enhanced_strategy, enhanced_results, enhanced_metrics = test_enhanced_strategy()