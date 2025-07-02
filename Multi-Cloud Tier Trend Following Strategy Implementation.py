import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta
import seaborn as sns
from tabulate import tabulate
import warnings
warnings.filterwarnings('ignore')

class MultiCloudTrendStrategy:
    """
    Multi-Cloud Tier Trend Following Strategy with EMA Crossover and Dynamic Stop-Loss
    
    This strategy uses multiple EMA layers to identify trends and determine entry/exit points.
    It employs a dynamic stop-loss mechanism that evolves as the trade progresses.
    """
    
    def __init__(self, 
                 ema50_len=50,
                 ema120_len=120,
                 ema180_len=180,
                 ema340_len=340,
                 ema500_len=500,
                 ema8_len=8,
                 ema9_len=9,
                 bars_for_trailing_sl=20,
                 bars_over_ema8_req=15,
                 sl_percent=1.0,
                 initial_capital=100000,
                 commission=0.001,
                 position_size_percent=100.0):
        """
        Initialize the strategy with configurable parameters
        
        Parameters:
        -----------
        ema50_len, ema120_len, etc.: int
            Periods for the various EMAs used in the strategy
        bars_for_trailing_sl: int
            Number of bars to hold before activating trailing stop loss
        bars_over_ema8_req: int
            Number of consecutive bars above/below EMA8 required to switch to EMA9 stop loss
        sl_percent: float
            Initial stop loss percentage
        initial_capital: float
            Initial capital for backtesting
        commission: float
            Commission rate per trade (e.g., 0.1% = 0.001)
        position_size_percent: float
            Percentage of equity to use per trade
        """
        # EMA parameters
        self.ema50_len = ema50_len
        self.ema120_len = ema120_len
        self.ema180_len = ema180_len
        self.ema340_len = ema340_len
        self.ema500_len = ema500_len
        self.ema8_len = ema8_len
        self.ema9_len = ema9_len
        
        # Stop loss parameters
        self.bars_for_trailing_sl = bars_for_trailing_sl
        self.bars_over_ema8_req = bars_over_ema8_req
        self.sl_percent = sl_percent
        
        # Backtest parameters
        self.initial_capital = initial_capital
        self.equity = initial_capital
        self.commission = commission
        self.position_size_percent = position_size_percent
        
        # Trading variables
        self.position = 0  # 0: no position, 1: long, -1: short
        self.entry_price = 0
        self.stop_loss_price = 0
        self.position_size = 0
        self.bars_since_entry = 0
        self.use_ema9_stop = False
        
        # Results tracking
        self.equity_curve = []
        self.trades = []
        self.current_trade = None
    
    def calculate_indicators(self, data):
        """
        Calculate all EMAs and strategy indicators
        
        Parameters:
        -----------
        data: DataFrame
            OHLCV data
            
        Returns:
        --------
        data: DataFrame
            Data with added indicators
        """
        # Calculate EMAs
        data['ema50'] = data['close'].ewm(span=self.ema50_len, adjust=False).mean()
        data['ema120'] = data['close'].ewm(span=self.ema120_len, adjust=False).mean()
        data['ema180'] = data['close'].ewm(span=self.ema180_len, adjust=False).mean()
        data['ema340'] = data['close'].ewm(span=self.ema340_len, adjust=False).mean()
        data['ema500'] = data['close'].ewm(span=self.ema500_len, adjust=False).mean()
        data['ema8'] = data['close'].ewm(span=self.ema8_len, adjust=False).mean()
        data['ema9'] = data['close'].ewm(span=self.ema9_len, adjust=False).mean()
        
        # Cloud 4 (Long-term trend)
        data['cloud4_up'] = data['ema340'] > data['ema500']
        data['cloud4_down'] = data['ema340'] < data['ema500']
        
        # Cloud 3 (Medium-term trend)
        # Calculate crossovers
        data['cloud3_cross_up'] = np.where(
            (data['ema50'] > data['ema120']) & (data['ema50'].shift(1) <= data['ema120'].shift(1)),
            1, 0
        )
        
        data['cloud3_cross_down'] = np.where(
            (data['ema50'] < data['ema120']) & (data['ema50'].shift(1) >= data['ema120'].shift(1)),
            1, 0
        )
        
        # Valid zone assessment
        data['valid_long_cross'] = (data['ema180'] < data['ema500']) | ((data['ema50'] >= data['ema500']) & (data['ema50'] <= data['ema340']))
        data['valid_short_cross'] = (data['ema50'] > data['ema500']) | ((data['ema50'] <= data['ema500']) & (data['ema50'] >= data['ema340']))
        
        # Entry conditions
        data['long_condition'] = data['cloud4_up'] & data['cloud3_cross_up'] & data['valid_long_cross']
        data['short_condition'] = data['cloud4_down'] & data['cloud3_cross_down'] & data['valid_short_cross']
        
        return data
    
    def backtest(self, data):
        """
        Run backtest on the provided data
        
        Parameters:
        -----------
        data: DataFrame
            OHLCV data
            
        Returns:
        --------
        equity_curve: Series
            Equity curve
        trades_df: DataFrame
            Trade details
        """
        # Calculate indicators
        data = self.calculate_indicators(data)
        
        # Reset trading variables
        self.equity = self.initial_capital
        self.position = 0
        self.equity_curve = [self.equity]
        self.trades = []
        
        # Skip initial period until we have enough data for indicators
        start_idx = max(self.ema50_len, self.ema120_len, self.ema180_len, 
                         self.ema340_len, self.ema500_len, self.ema8_len, self.ema9_len)
        
        # Loop through each bar
        for i in range(start_idx, len(data)):
            current_bar = data.iloc[i]
            
            # Update equity curve with current value of position
            if self.position != 0:
                # Calculate current value of position
                price_change = current_bar['close'] - self.entry_price
                unrealized_pnl = self.position * self.position_size * price_change
                current_equity = self.equity + unrealized_pnl
            else:
                current_equity = self.equity
            
            self.equity_curve.append(current_equity)
            
            # Update trade state if in a position
            if self.position != 0:
                self.bars_since_entry += 1
                
                # Dynamic stop loss logic
                if self.bars_since_entry >= self.bars_for_trailing_sl:
                    # Check for EMA8 conditions to switch to EMA9 stop loss
                    if not self.use_ema9_stop:
                        # For long positions
                        if self.position == 1:
                            # Check if price stayed above EMA8 for required number of bars
                            all_above = True
                            for j in range(self.bars_over_ema8_req):
                                if i-j >= 0 and data.iloc[i-j]['close'] < data.iloc[i-j]['ema8']:
                                    all_above = False
                                    break
                            
                            if all_above:
                                self.use_ema9_stop = True
                        
                        # For short positions
                        elif self.position == -1:
                            # Check if price stayed below EMA8 for required number of bars
                            all_below = True
                            for j in range(self.bars_over_ema8_req):
                                if i-j >= 0 and data.iloc[i-j]['close'] > data.iloc[i-j]['ema8']:
                                    all_below = False
                                    break
                            
                            if all_below:
                                self.use_ema9_stop = True
                    
                    # Set stop loss based on conditions
                    if self.use_ema9_stop:
                        self.stop_loss_price = current_bar['ema9']
                    else:
                        self.stop_loss_price = current_bar['ema500']
                
                # Check if stop loss is hit
                stop_loss_triggered = False
                if (self.position == 1 and current_bar['close'] < self.stop_loss_price) or \
                   (self.position == -1 and current_bar['close'] > self.stop_loss_price):
                    stop_loss_triggered = True
                
                # Execute exit if stop loss is triggered
                if stop_loss_triggered:
                    # Calculate profit/loss
                    exit_price = current_bar['close']
                    price_change = (exit_price - self.entry_price) * self.position
                    profit_loss = self.position_size * price_change
                    
                    # Deduct commission
                    commission_cost = self.position_size * exit_price * self.commission
                    profit_loss -= commission_cost
                    
                    # Update equity
                    self.equity += profit_loss
                    
                    # Record trade
                    self.current_trade.update({
                        'exit_date': current_bar.name,
                        'exit_price': exit_price,
                        'exit_reason': "Stop Loss",
                        'stop_loss_type': "EMA9" if self.use_ema9_stop else "EMA500" if self.bars_since_entry >= self.bars_for_trailing_sl else "Fixed",
                        'profit_loss': profit_loss,
                        'profit_loss_pct': (profit_loss / self.current_trade['initial_equity']) * 100,
                        'bars_held': self.bars_since_entry,
                        'equity': self.equity
                    })
                    self.trades.append(self.current_trade)
                    
                    # Reset position
                    self.position = 0
                    self.position_size = 0
                    self.stop_loss_price = 0
                    self.bars_since_entry = 0
                    self.use_ema9_stop = False
                    self.current_trade = None
            
            # Check for entry signals if not in a position
            if self.position == 0:
                # Long entry
                if current_bar['long_condition']:
                    self.position = 1
                    self.entry_price = current_bar['close']
                    
                    # Calculate initial stop loss (fixed percentage)
                    self.stop_loss_price = self.entry_price * (1 - self.sl_percent / 100)
                    
                    # Calculate position size
                    self.position_size = (self.equity * (self.position_size_percent / 100)) / self.entry_price
                    
                    # Account for commission
                    commission_cost = self.position_size * self.entry_price * self.commission
                    self.equity -= commission_cost
                    
                    # Reset trade tracking variables
                    self.bars_since_entry = 0
                    self.use_ema9_stop = False
                    
                    # Record trade
                    self.current_trade = {
                        'entry_date': current_bar.name,
                        'entry_price': self.entry_price,
                        'position': 'Long',
                        'position_size': self.position_size,
                        'initial_stop_loss': self.stop_loss_price,
                        'initial_equity': self.equity
                    }
                
                # Short entry
                elif current_bar['short_condition']:
                    self.position = -1
                    self.entry_price = current_bar['close']
                    
                    # Calculate initial stop loss (fixed percentage)
                    self.stop_loss_price = self.entry_price * (1 + self.sl_percent / 100)
                    
                    # Calculate position size
                    self.position_size = (self.equity * (self.position_size_percent / 100)) / self.entry_price
                    
                    # Account for commission
                    commission_cost = self.position_size * self.entry_price * self.commission
                    self.equity -= commission_cost
                    
                    # Reset trade tracking variables
                    self.bars_since_entry = 0
                    self.use_ema9_stop = False
                    
                    # Record trade
                    self.current_trade = {
                        'entry_date': current_bar.name,
                        'entry_price': self.entry_price,
                        'position': 'Short',
                        'position_size': self.position_size,
                        'initial_stop_loss': self.stop_loss_price,
                        'initial_equity': self.equity
                    }
        
        # Close any open position at the end of the backtest
        if self.position != 0:
            final_bar = data.iloc[-1]
            exit_price = final_bar['close']
            
            # Calculate profit/loss
            price_change = (exit_price - self.entry_price) * self.position
            profit_loss = self.position_size * price_change
            
            # Deduct commission
            commission_cost = self.position_size * exit_price * self.commission
            profit_loss -= commission_cost
            
            # Update equity
            self.equity += profit_loss
            
            # Record trade
            self.current_trade.update({
                'exit_date': final_bar.name,
                'exit_price': exit_price,
                'exit_reason': "End of Backtest",
                'stop_loss_type': "EMA9" if self.use_ema9_stop else "EMA500" if self.bars_since_entry >= self.bars_for_trailing_sl else "Fixed",
                'profit_loss': profit_loss,
                'profit_loss_pct': (profit_loss / self.current_trade['initial_equity']) * 100,
                'bars_held': self.bars_since_entry,
                'equity': self.equity
            })
            self.trades.append(self.current_trade)
            
            # Reset position
            self.position = 0
            self.position_size = 0
            self.current_trade = None
        
        # Convert equity curve to DataFrame
        self.equity_curve = pd.Series(self.equity_curve, index=data.index[-(len(self.equity_curve)):])
        
        # Convert trades to DataFrame if there are trades
        if self.trades:
            self.trades_df = pd.DataFrame(self.trades)
        else:
            self.trades_df = pd.DataFrame()
        
        # Calculate performance metrics
        self.calculate_performance_metrics()
        
        return self.equity_curve, self.trades_df, data
    
    def calculate_performance_metrics(self):
        """
        Calculate strategy performance metrics
        """
        self.metrics = {}
        
        # Skip if no trades
        if not self.trades:
            return
        
        # Total return
        self.metrics['total_return'] = (self.equity - self.initial_capital) / self.initial_capital * 100
        
        # Number of trades
        self.metrics['total_trades'] = len(self.trades)
        
        # Win rate
        winning_trades = [t for t in self.trades if t.get('profit_loss', 0) > 0]
        self.metrics['win_rate'] = len(winning_trades) / len(self.trades) * 100
        
        # Average win/loss
        if winning_trades:
            self.metrics['avg_win'] = sum(t.get('profit_loss', 0) for t in winning_trades) / len(winning_trades)
        else:
            self.metrics['avg_win'] = 0
        
        losing_trades = [t for t in self.trades if t.get('profit_loss', 0) <= 0]
        if losing_trades:
            self.metrics['avg_loss'] = sum(t.get('profit_loss', 0) for t in losing_trades) / len(losing_trades)
        else:
            self.metrics['avg_loss'] = 0
        
        # Profit factor
        total_wins = sum(t.get('profit_loss', 0) for t in winning_trades)
        total_losses = abs(sum(t.get('profit_loss', 0) for t in losing_trades)) if losing_trades else 1
        self.metrics['profit_factor'] = total_wins / total_losses if total_losses > 0 else float('inf')
        
        # Max drawdown
        equity_peaks = self.equity_curve.cummax()
        drawdowns = (self.equity_curve - equity_peaks) / equity_peaks * 100
        self.metrics['max_drawdown'] = drawdowns.min()
        
        # Sharpe ratio (assuming 252 trading days per year)
        daily_returns = self.equity_curve.pct_change().dropna()
        if len(daily_returns) > 0 and daily_returns.std() > 0:
            self.metrics['sharpe_ratio'] = np.sqrt(252) * daily_returns.mean() / daily_returns.std()
        else:
            self.metrics['sharpe_ratio'] = 0
        
        # Average trade duration
        self.metrics['avg_bars_held'] = np.mean([t.get('bars_held', 0) for t in self.trades])
        
        # Stop loss type distribution
        sl_types = [t.get('stop_loss_type', 'Unknown') for t in self.trades]
        self.metrics['fixed_sl_count'] = sl_types.count('Fixed')
        self.metrics['ema500_sl_count'] = sl_types.count('EMA500')
        self.metrics['ema9_sl_count'] = sl_types.count('EMA9')
        
        return self.metrics
    
    def plot_results(self, data=None, figsize=(15, 10)):
        """
        Plot backtest results
        
        Parameters:
        -----------
        data: DataFrame
            Data with calculated indicators for plotting
        figsize: tuple
            Figure size for plots
        """
        if not self.trades:
            print("No trades executed during the backtest period.")
            return
            
        plt.figure(figsize=figsize)
        
        # Plot equity curve
        plt.subplot(2, 1, 1)
        self.equity_curve.plot(label='Equity Curve')
        plt.title('Equity Curve')
        plt.xlabel('Date')
        plt.ylabel('Equity')
        plt.grid(True)
        plt.legend()
        
        # Plot price with trades and indicators
        if data is not None:
            plt.subplot(2, 1, 2)
            
            # Plot price and key EMAs
            data['close'].plot(label='Close Price', color='black', alpha=0.7)
            data['ema50'].plot(label=f'EMA50', color='yellow', alpha=0.7)
            data['ema120'].plot(label=f'EMA120', color='orange', alpha=0.7)
            data['ema340'].plot(label=f'EMA340', color='green', alpha=0.7)
            data['ema500'].plot(label=f'EMA500', color='red', alpha=0.7)
            
            # Plot entry and exit points for trades
            for trade in self.trades:
                if trade['position'] == 'Long':
                    plt.scatter(trade['entry_date'], trade['entry_price'], marker='^', color='green', s=100, label='Long Entry' if 'Long Entry' not in plt.gca().get_legend_handles_labels()[1] else "")
                    
                    if 'exit_date' in trade and 'exit_price' in trade:
                        plt.scatter(trade['exit_date'], trade['exit_price'], marker='v', color='red', s=100, label='Exit' if 'Exit' not in plt.gca().get_legend_handles_labels()[1] else "")
                        
                        # Plot stop loss at exit
                        plt.plot([trade['entry_date'], trade['exit_date']], 
                                [trade['initial_stop_loss'], trade['exit_price']], 
                                color='red', linestyle='--', alpha=0.5)
                else:
                    plt.scatter(trade['entry_date'], trade['entry_price'], marker='v', color='red', s=100, label='Short Entry' if 'Short Entry' not in plt.gca().get_legend_handles_labels()[1] else "")
                    
                    if 'exit_date' in trade and 'exit_price' in trade:
                        plt.scatter(trade['exit_date'], trade['exit_price'], marker='^', color='green', s=100, label='Exit' if 'Exit' not in plt.gca().get_legend_handles_labels()[1] else "")
                        
                        # Plot stop loss at exit
                        plt.plot([trade['entry_date'], trade['exit_date']], 
                                [trade['initial_stop_loss'], trade['exit_price']], 
                                color='red', linestyle='--', alpha=0.5)
            
            plt.title('Price Chart with Cloud EMAs and Trading Signals')
            plt.xlabel('Date')
            plt.ylabel('Price')
            plt.grid(True)
            plt.legend()
        
        plt.tight_layout()
        plt.savefig('multi_cloud_tier_strategy_results.png')
        plt.show()
        
        # Plot additional performance charts
        plt.figure(figsize=figsize)
        
        # Plot drawdown
        plt.subplot(2, 2, 1)
        equity_peaks = self.equity_curve.cummax()
        drawdowns = (self.equity_curve - equity_peaks) / equity_peaks * 100
        drawdowns.plot(label='Drawdown %')
        plt.title('Drawdown')
        plt.xlabel('Date')
        plt.ylabel('Drawdown %')
        plt.grid(True)
        
        # Plot trade P&L distribution
        plt.subplot(2, 2, 2)
        profit_losses = [t.get('profit_loss', 0) for t in self.trades]
        if profit_losses:
            sns.histplot(profit_losses, kde=True)
            plt.axvline(x=0, color='r', linestyle='--')
            plt.title('Trade P&L Distribution')
            plt.xlabel('P&L')
            plt.ylabel('Frequency')
            plt.grid(True)
        
        # Plot cumulative trades
        plt.subplot(2, 2, 3)
        if profit_losses:
            cumulative_pnl = pd.Series(profit_losses).cumsum()
            cumulative_pnl.plot()
            plt.title('Cumulative P&L')
            plt.xlabel('Trade #')
            plt.ylabel('Cumulative P&L')
            plt.grid(True)
        
        # Plot stop loss type distribution
        plt.subplot(2, 2, 4)
        sl_types = [t.get('stop_loss_type', 'Unknown') for t in self.trades]
        sl_counts = pd.Series(sl_types).value_counts()
        sl_counts.plot(kind='bar')
        plt.title('Stop Loss Type Distribution')
        plt.xlabel('Stop Loss Type')
        plt.ylabel('Count')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('multi_cloud_tier_strategy_performance.png')
        plt.show()
        
        # Plot EMA "clouds" visualization
        if data is not None:
            plt.figure(figsize=figsize)
            
            # Plot close price
            plt.plot(data.index, data['close'], color='black', alpha=0.7, label='Close Price')
            
            # Plot Cloud 4 (Long-term trend)
            plt.fill_between(data.index, data['ema340'], data['ema500'], 
                           where=data['ema340'] > data['ema500'], 
                           color='green', alpha=0.3, label='Cloud 4 Up')
            
            plt.fill_between(data.index, data['ema340'], data['ema500'], 
                           where=data['ema340'] < data['ema500'], 
                           color='red', alpha=0.3, label='Cloud 4 Down')
            
            # Plot Cloud 3 (Medium-term trend)
            plt.fill_between(data.index, data['ema50'], data['ema120'], 
                           where=data['ema50'] > data['ema120'], 
                           color='green', alpha=0.2, label='Cloud 3 Up')
            
            plt.fill_between(data.index, data['ema50'], data['ema120'], 
                           where=data['ema50'] < data['ema120'], 
                           color='red', alpha=0.2, label='Cloud 3 Down')
            
            # Plot EMA lines
            plt.plot(data.index, data['ema50'], color='yellow', label='EMA50')
            plt.plot(data.index, data['ema120'], color='orange', label='EMA120')
            plt.plot(data.index, data['ema340'], color='green', label='EMA340')
            plt.plot(data.index, data['ema500'], color='red', label='EMA500')
            
            # Add small EMAs
            plt.plot(data.index, data['ema8'], color='purple', linestyle='--', label='EMA8')
            plt.plot(data.index, data['ema9'], color='blue', linestyle='--', label='EMA9')
            
            plt.title('Multi-Cloud EMA Visualization')
            plt.xlabel('Date')
            plt.ylabel('Price')
            plt.grid(True)
            plt.legend()
            
            plt.tight_layout()
            plt.savefig('multi_cloud_tier_strategy_clouds.png')
            plt.show()
    
    def print_performance_summary(self):
        """
        Print performance summary
        """
        # Skip if no trades
        if not self.trades:
            print("No trades executed during the backtest period.")
            return
            
        print("\n===== STRATEGY PERFORMANCE SUMMARY =====")
        print(f"Initial Capital: ${self.initial_capital:,.2f}")
        print(f"Final Equity: ${self.equity:,.2f}")
        print(f"Total Return: {self.metrics['total_return']:.2f}%")
        print(f"Total Trades: {self.metrics['total_trades']}")
        print(f"Win Rate: {self.metrics['win_rate']:.2f}%")
        print(f"Average Win: ${self.metrics['avg_win']:,.2f}")
        print(f"Average Loss: ${self.metrics['avg_loss']:,.2f}")
        print(f"Profit Factor: {self.metrics['profit_factor']:.2f}")
        print(f"Max Drawdown: {self.metrics['max_drawdown']:.2f}%")
        print(f"Sharpe Ratio: {self.metrics['sharpe_ratio']:.2f}")
        print(f"Average Bars Held: {self.metrics['avg_bars_held']:.2f}")
        
        print("\n===== STOP LOSS DISTRIBUTION =====")
        print(f"Fixed Stop Loss: {self.metrics['fixed_sl_count']} trades ({self.metrics['fixed_sl_count']/self.metrics['total_trades']*100:.2f}%)")
        print(f"EMA500 Stop Loss: {self.metrics['ema500_sl_count']} trades ({self.metrics['ema500_sl_count']/self.metrics['total_trades']*100:.2f}%)")
        print(f"EMA9 Stop Loss: {self.metrics['ema9_sl_count']} trades ({self.metrics['ema9_sl_count']/self.metrics['total_trades']*100:.2f}%)")
        
        # Print trades table
        print("\n===== TRADES =====")
        trades_table = []
        
        # Limit to first 10 trades for display
        display_trades = self.trades[:10] if len(self.trades) > 10 else self.trades
        
        for i, trade in enumerate(display_trades):
            trade_data = [
                i+1,
                trade['entry_date'].strftime('%Y-%m-%d'),
                trade['position'],
                f"${trade['entry_price']:.2f}",
            ]
            
            if 'exit_date' in trade and 'exit_price' in trade:
                trade_data.extend([
                    trade['exit_date'].strftime('%Y-%m-%d'),
                    f"${trade['exit_price']:.2f}",
                    trade.get('exit_reason', 'N/A'),
                    trade.get('stop_loss_type', 'N/A'),
                    f"${trade.get('profit_loss', 0):.2f}",
                    f"{trade.get('profit_loss_pct', 0):.2f}%",
                    trade.get('bars_held', 0)
                ])
            else:
                trade_data.extend(['N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A'])
                
            trades_table.append(trade_data)
        
        headers = ["#", "Entry Date", "Position", "Entry Price", "Exit Date", "Exit Price", 
                 "Exit Reason", "Stop Loss Type", "P&L", "P&L %", "Bars Held"]
        print(tabulate(trades_table, headers=headers, tablefmt="grid"))
        
        if len(self.trades) > 10:
            print(f"... and {len(self.trades) - 10} more trades")

def generate_synthetic_data(periods=1000, trend_strength=0.002, volatility=0.015, cycle_periods=120, seed=42):
    """
    Generate synthetic price data with trends, cycles, and volatility
    
    Parameters:
    -----------
    periods: int
        Number of periods to generate
    trend_strength: float
        Strength of the long-term trend component
    volatility: float
        Base volatility of the price series
    cycle_periods: int
        Number of periods for one complete market cycle
    seed: int
        Random seed for reproducibility
    
    Returns:
    --------
    data: DataFrame
        OHLCV data
    """
    np.random.seed(seed)
    
    # Generate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=periods)
    date_range = pd.date_range(start=start_date, end=end_date, periods=periods)
    
    # Generate price series with trends and cycles
    price = 100  # Starting price
    prices = []
    
    # Time components
    time = np.arange(periods)
    
    # Trend component (random walk with drift)
    trend = trend_strength * time
    
    # Cycle component (sine wave)
    cycle = np.sin(2 * np.pi * time / cycle_periods)
    
    # Noise component
    noise = np.random.normal(0, volatility, periods)
    
    # Combine components
    for i in range(periods):
        if i == 0:
            prices.append(price)
        else:
            # Price changes with trend, cycle, and noise
            price_change = trend[i] - trend[i-1] + 0.05 * cycle[i] + noise[i]
            new_price = prices[-1] * (1 + price_change)
            prices.append(new_price)
    
    # Create OHLC data
    high_low_range = np.array([np.random.uniform(0.5, 1.5) * volatility * price for price in prices])
    open_close_range = high_low_range * np.random.uniform(0.2, 0.8, periods)
    
    df = pd.DataFrame(index=date_range)
    df['close'] = prices
    df['high'] = df['close'] + high_low_range / 2
    df['low'] = df['close'] - high_low_range / 2
    df['open'] = df['close'] - np.random.uniform(-1, 1, periods) * open_close_range
    df['volume'] = np.random.randint(100000, 10000000, periods)
    
    # Increase volume during strong price moves
    for i in range(1, periods):
        if abs(df['close'].iloc[i] - df['close'].iloc[i-1]) > 1.5 * volatility * prices[i-1]:
            df.loc[df.index[i], 'volume'] *= np.random.uniform(3, 5)
    
    return df

def fetch_data(symbol, start_date, end_date, interval='1d'):
    """
    Fetch historical data from Yahoo Finance
    
    Parameters:
    -----------
    symbol: str
        Ticker symbol
    start_date: str
        Start date in 'YYYY-MM-DD' format
    end_date: str
        End date in 'YYYY-MM-DD' format
    interval: str
        Data interval ('1d', '1wk', '1mo')
    
    Returns:
    --------
    data: DataFrame
        OHLCV data
    """
    data = yf.download(symbol, start=start_date, end=end_date, interval=interval)
    return data

def optimize_parameters(data, param_grid):
    """
    Simple grid search optimization
    
    Parameters:
    -----------
    data: DataFrame
        OHLCV data for backtesting
    param_grid: dict
        Dictionary of parameter ranges for grid search
    
    Returns:
    --------
    results_df: DataFrame
        Results of grid search
    best_params: dict
        Best parameters found
    """
    best_sharpe = -np.inf
    best_params = None
    results = []
    
    # Generate all parameter combinations
    param_combinations = []
    for ema50_len in param_grid['ema50_len']:
        for ema120_len in param_grid['ema120_len']:
            for bars_for_trailing_sl in param_grid['bars_for_trailing_sl']:
                for sl_percent in param_grid['sl_percent']:
                    param_combinations.append({
                        'ema50_len': ema50_len,
                        'ema120_len': ema120_len,
                        'bars_for_trailing_sl': bars_for_trailing_sl,
                        'sl_percent': sl_percent,
                    })
    
    total_combinations = len(param_combinations)
    print(f"Testing {total_combinations} parameter combinations...")
    
    for i, params in enumerate(param_combinations):
        # Create strategy with current parameters
        strategy = MultiCloudTrendStrategy(
            ema50_len=params['ema50_len'],
            ema120_len=params['ema120_len'],
            bars_for_trailing_sl=params['bars_for_trailing_sl'],
            sl_percent=params['sl_percent']
        )
        
        # Run backtest
        equity_curve, trades_df, _ = strategy.backtest(data)
        
        # Skip if no trades
        if not strategy.trades:
            continue
        
        # Record results
        results.append({
            **params,
            'sharpe_ratio': strategy.metrics.get('sharpe_ratio', 0),
            'total_return': strategy.metrics.get('total_return', 0),
            'max_drawdown': strategy.metrics.get('max_drawdown', 0),
            'win_rate': strategy.metrics.get('win_rate', 0),
            'profit_factor': strategy.metrics.get('profit_factor', 0),
            'total_trades': strategy.metrics.get('total_trades', 0)
        })
        
        # Update best parameters
        if strategy.metrics.get('sharpe_ratio', 0) > best_sharpe:
            best_sharpe = strategy.metrics.get('sharpe_ratio', 0)
            best_params = params
        
        # Print progress
        if (i + 1) % 10 == 0 or (i + 1) == total_combinations:
            print(f"Progress: {i + 1}/{total_combinations} combinations tested")
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    return results_df, best_params

def main():
    # Set random seed
    np.random.seed(42)
    
    # Use either real data or synthetic data
    use_real_data = False
    
    if use_real_data:
        # Fetch historical data
        symbol = "ETH-USD"  # Ethereum USD
        start_date = "2022-01-01"
        end_date = "2023-01-01"
        data = fetch_data(symbol, start_date, end_date, interval='1d')
        print(f"Fetched {len(data)} data points for {symbol}")
    else:
        # Generate synthetic data
        data = generate_synthetic_data(periods=1000, trend_strength=0.002, volatility=0.015, cycle_periods=120)
        print(f"Generated {len(data)} synthetic data points")
    
    # Option to optimize parameters
    optimize = True
    
    if optimize:
        # Define parameter grid for optimization
        param_grid = {
            'ema50_len': [40, 50, 60],
            'ema120_len': [100, 120, 140],
            'bars_for_trailing_sl': [15, 20, 25],
            'sl_percent': [0.8, 1.0, 1.2],
        }
        
        # Run optimization
        results_df, best_params = optimize_parameters(data, param_grid)
        
        # Print optimization results
        print("\n===== OPTIMIZATION RESULTS =====")
        print("Best parameters:")
        for param, value in best_params.items():
            print(f"{param}: {value}")
        
        # Show top 5 parameter combinations
        print("\nTop 5 parameter combinations by Sharpe ratio:")
        top_results = results_df.sort_values('sharpe_ratio', ascending=False).head(5)
        print(top_results[['ema50_len', 'ema120_len', 'bars_for_trailing_sl', 'sl_percent',
                         'sharpe_ratio', 'total_return', 'max_drawdown', 'win_rate']])
        
        # Use best parameters for final backtest
        strategy = MultiCloudTrendStrategy(
            ema50_len=best_params['ema50_len'],
            ema120_len=best_params['ema120_len'],
            bars_for_trailing_sl=best_params['bars_for_trailing_sl'],
            sl_percent=best_params['sl_percent']
        )
    else:
        # Use default parameters
        strategy = MultiCloudTrendStrategy()
    
    # Run backtest
    print("\nRunning backtest with final parameters...")
    equity_curve, trades_df, indicator_data = strategy.backtest(data)
    
    # Print performance summary
    strategy.print_performance_summary()
    
    # Plot results
    strategy.plot_results(indicator_data)
    
    return strategy, data

if __name__ == "__main__":
    strategy, data = main()