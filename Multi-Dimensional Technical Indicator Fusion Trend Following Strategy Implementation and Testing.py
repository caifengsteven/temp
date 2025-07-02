import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta
import talib as ta
import seaborn as sns
import mplfinance as mpf
from scipy.stats import norm
import warnings

warnings.filterwarnings('ignore')

class MultiIndicatorTrendFollowingStrategy:
    """
    Implementation of the Multi-Dimensional Technical Indicator Fusion Trend Following Strategy
    """
    
    def __init__(self, 
                 fast_ma_length=10, 
                 slow_ma_length=20, 
                 rsi_length=14, 
                 rsi_oversold=30, 
                 rsi_overbought=70, 
                 fast_macd_length=12, 
                 slow_macd_length=26, 
                 signal_macd_length=9, 
                 volume_ma_length=20, 
                 volume_multiplier=1.5, 
                 atr_period=14, 
                 stop_loss_atr_multi=2.0, 
                 take_profit_atr_multi=3.0,
                 percentage_of_equity=90,
                 initial_capital=100000):
        """
        Initialize the strategy with the given parameters
        """
        # Moving Averages
        self.fast_ma_length = fast_ma_length
        self.slow_ma_length = slow_ma_length
        
        # RSI
        self.rsi_length = rsi_length
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        
        # MACD
        self.fast_macd_length = fast_macd_length
        self.slow_macd_length = slow_macd_length
        self.signal_macd_length = signal_macd_length
        
        # Volume Filter
        self.volume_ma_length = volume_ma_length
        self.volume_multiplier = volume_multiplier
        
        # ATR for Stop Loss / Take Profit
        self.atr_period = atr_period
        self.stop_loss_atr_multi = stop_loss_atr_multi
        self.take_profit_atr_multi = take_profit_atr_multi
        
        # Position sizing
        self.percentage_of_equity = percentage_of_equity / 100
        self.initial_capital = initial_capital
        
    def calculate_indicators(self, data):
        """
        Calculate all technical indicators required for the strategy
        """
        # Make a copy of the data to avoid modifying the original
        df = data.copy()
        
        # Moving Averages
        df['fast_ma'] = ta.EMA(df['close'], timeperiod=self.fast_ma_length)
        df['slow_ma'] = ta.EMA(df['close'], timeperiod=self.slow_ma_length)
        
        # RSI
        df['rsi'] = ta.RSI(df['close'], timeperiod=self.rsi_length)
        
        # MACD
        macd, signal, hist = ta.MACD(df['close'], 
                                     fastperiod=self.fast_macd_length, 
                                     slowperiod=self.slow_macd_length, 
                                     signalperiod=self.signal_macd_length)
        df['macd'] = macd
        df['macd_signal'] = signal
        df['macd_hist'] = hist
        
        # Volume Moving Average
        df['volume_ma'] = ta.SMA(df['volume'], timeperiod=self.volume_ma_length)
        
        # ATR
        df['atr'] = ta.ATR(df['high'], df['low'], df['close'], timeperiod=self.atr_period)
        
        return df
    
    def generate_signals(self, data):
        """
        Generate buy/sell signals based on the strategy conditions
        """
        # Calculate indicators
        df = self.calculate_indicators(data)
        
        # Initialize signal columns
        df['signal'] = 0  # 1 for buy, -1 for sell, 0 for hold
        df['position'] = 0
        df['stop_loss'] = np.nan
        df['take_profit'] = np.nan
        df['equity'] = self.initial_capital
        df['in_position'] = False
        df['entry_price'] = np.nan
        df['exit_price'] = np.nan
        df['exit_reason'] = None
        df['trade_returns'] = 0
        
        # Entry Conditions
        df['long_condition'] = (
            (df['fast_ma'] > df['slow_ma']) & 
            (df['fast_ma'].shift(1) <= df['slow_ma'].shift(1)) &  # Crossover
            (df['rsi'] > 50) & 
            (df['volume'] > (df['volume_ma'] * self.volume_multiplier))
        )
        
        df['short_condition'] = (
            (df['fast_ma'] < df['slow_ma']) & 
            (df['fast_ma'].shift(1) >= df['slow_ma'].shift(1)) &  # Crossunder
            (df['rsi'] < 50) & 
            (df['volume'] > (df['volume_ma'] * self.volume_multiplier))
        )
        
        # Loop through data to generate signals and track positions
        position = 0
        entry_price = 0
        stop_loss = 0
        take_profit = 0
        
        for i in range(1, len(df)):
            # Default is to maintain the previous position
            df.loc[df.index[i], 'position'] = position
            
            # Check for exit if we're in a position
            if position != 0:
                current_price = df.loc[df.index[i], 'close']
                
                # For long positions
                if position > 0:
                    # Check if stop loss or take profit was hit
                    if df.loc[df.index[i], 'low'] <= stop_loss:
                        df.loc[df.index[i], 'signal'] = -1
                        df.loc[df.index[i], 'exit_price'] = stop_loss
                        df.loc[df.index[i], 'exit_reason'] = 'Stop Loss'
                        df.loc[df.index[i], 'in_position'] = False
                        df.loc[df.index[i], 'trade_returns'] = (stop_loss / entry_price) - 1
                        position = 0
                    elif df.loc[df.index[i], 'high'] >= take_profit:
                        df.loc[df.index[i], 'signal'] = -1
                        df.loc[df.index[i], 'exit_price'] = take_profit
                        df.loc[df.index[i], 'exit_reason'] = 'Take Profit'
                        df.loc[df.index[i], 'in_position'] = False
                        df.loc[df.index[i], 'trade_returns'] = (take_profit / entry_price) - 1
                        position = 0
                    # Check for new short signal
                    elif df.loc[df.index[i], 'short_condition']:
                        df.loc[df.index[i], 'signal'] = -1
                        df.loc[df.index[i], 'exit_price'] = current_price
                        df.loc[df.index[i], 'exit_reason'] = 'New Signal'
                        df.loc[df.index[i], 'in_position'] = False
                        df.loc[df.index[i], 'trade_returns'] = (current_price / entry_price) - 1
                        position = 0
                
                # For short positions
                elif position < 0:
                    # Check if stop loss or take profit was hit
                    if df.loc[df.index[i], 'high'] >= stop_loss:
                        df.loc[df.index[i], 'signal'] = 1
                        df.loc[df.index[i], 'exit_price'] = stop_loss
                        df.loc[df.index[i], 'exit_reason'] = 'Stop Loss'
                        df.loc[df.index[i], 'in_position'] = False
                        df.loc[df.index[i], 'trade_returns'] = 1 - (stop_loss / entry_price)
                        position = 0
                    elif df.loc[df.index[i], 'low'] <= take_profit:
                        df.loc[df.index[i], 'signal'] = 1
                        df.loc[df.index[i], 'exit_price'] = take_profit
                        df.loc[df.index[i], 'exit_reason'] = 'Take Profit'
                        df.loc[df.index[i], 'in_position'] = False
                        df.loc[df.index[i], 'trade_returns'] = 1 - (take_profit / entry_price)
                        position = 0
                    # Check for new long signal
                    elif df.loc[df.index[i], 'long_condition']:
                        df.loc[df.index[i], 'signal'] = 1
                        df.loc[df.index[i], 'exit_price'] = current_price
                        df.loc[df.index[i], 'exit_reason'] = 'New Signal'
                        df.loc[df.index[i], 'in_position'] = False
                        df.loc[df.index[i], 'trade_returns'] = 1 - (current_price / entry_price)
                        position = 0
            
            # Check for new entry signal
            if position == 0:
                if df.loc[df.index[i], 'long_condition']:
                    df.loc[df.index[i], 'signal'] = 1
                    df.loc[df.index[i], 'position'] = 1
                    df.loc[df.index[i], 'in_position'] = True
                    entry_price = df.loc[df.index[i], 'close']
                    df.loc[df.index[i], 'entry_price'] = entry_price
                    atr = df.loc[df.index[i], 'atr']
                    stop_loss = entry_price - (atr * self.stop_loss_atr_multi)
                    take_profit = entry_price + (atr * self.take_profit_atr_multi)
                    df.loc[df.index[i], 'stop_loss'] = stop_loss
                    df.loc[df.index[i], 'take_profit'] = take_profit
                    position = 1
                elif df.loc[df.index[i], 'short_condition']:
                    df.loc[df.index[i], 'signal'] = -1
                    df.loc[df.index[i], 'position'] = -1
                    df.loc[df.index[i], 'in_position'] = True
                    entry_price = df.loc[df.index[i], 'close']
                    df.loc[df.index[i], 'entry_price'] = entry_price
                    atr = df.loc[df.index[i], 'atr']
                    stop_loss = entry_price + (atr * self.stop_loss_atr_multi)
                    take_profit = entry_price - (atr * self.take_profit_atr_multi)
                    df.loc[df.index[i], 'stop_loss'] = stop_loss
                    df.loc[df.index[i], 'take_profit'] = take_profit
                    position = -1
        
        # Calculate equity curve
        equity = self.initial_capital
        for i in range(1, len(df)):
            if df.loc[df.index[i], 'signal'] != 0 and df.loc[df.index[i], 'trade_returns'] != 0:
                # Only apply trade returns when exiting a position
                equity *= (1 + (df.loc[df.index[i], 'trade_returns'] * self.percentage_of_equity))
            df.loc[df.index[i], 'equity'] = equity
            
        return df
    
    def backtest(self, data):
        """
        Run backtest and calculate performance metrics
        """
        # Generate signals
        results = self.generate_signals(data)
        
        # Extract trades
        trades = results[(results['signal'] != 0) & (results['exit_reason'].notnull())].copy()
        
        # Calculate performance metrics
        if len(trades) > 0:
            # Calculate returns
            final_equity = results['equity'].iloc[-1]
            total_return = (final_equity / self.initial_capital) - 1
            
            # Calculate annualized return
            days = (results.index[-1] - results.index[0]).days
            annual_return = ((1 + total_return) ** (365 / days)) - 1
            
            # Calculate drawdown
            rolling_max = results['equity'].cummax()
            drawdown = (results['equity'] - rolling_max) / rolling_max
            max_drawdown = drawdown.min()
            
            # Calculate Sharpe ratio (assuming risk-free rate = 0)
            daily_returns = results['equity'].pct_change().dropna()
            sharpe_ratio = np.sqrt(252) * daily_returns.mean() / daily_returns.std()
            
            # Calculate win rate
            winning_trades = trades[trades['trade_returns'] > 0]
            win_rate = len(winning_trades) / len(trades) if len(trades) > 0 else 0
            
            # Calculate average win/loss
            avg_win = winning_trades['trade_returns'].mean() if len(winning_trades) > 0 else 0
            losing_trades = trades[trades['trade_returns'] <= 0]
            avg_loss = losing_trades['trade_returns'].mean() if len(losing_trades) > 0 else 0
            
            # Calculate profit factor
            total_profit = winning_trades['trade_returns'].sum() if len(winning_trades) > 0 else 0
            total_loss = abs(losing_trades['trade_returns'].sum()) if len(losing_trades) > 0 else 1
            profit_factor = total_profit / total_loss if total_loss != 0 else float('inf')
            
            # Calculate recovery factor
            recovery_factor = total_return / abs(max_drawdown) if max_drawdown != 0 else float('inf')
            
            metrics = {
                'total_return': total_return,
                'annual_return': annual_return,
                'max_drawdown': max_drawdown,
                'sharpe_ratio': sharpe_ratio,
                'win_rate': win_rate,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'profit_factor': profit_factor,
                'recovery_factor': recovery_factor,
                'num_trades': len(trades)
            }
        else:
            metrics = {
                'total_return': 0,
                'annual_return': 0,
                'max_drawdown': 0,
                'sharpe_ratio': 0,
                'win_rate': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'profit_factor': 0,
                'recovery_factor': 0,
                'num_trades': 0
            }
        
        return results, metrics
    
    def plot_results(self, results, show_indicators=True):
        """
        Plot the backtest results
        """
        plt.figure(figsize=(15, 15))
        
        # Plot price with MA
        plt.subplot(4, 1, 1)
        plt.plot(results.index, results['close'], label='Close Price', alpha=0.5)
        plt.plot(results.index, results['fast_ma'], label=f'Fast EMA ({self.fast_ma_length})', color='blue')
        plt.plot(results.index, results['slow_ma'], label=f'Slow EMA ({self.slow_ma_length})', color='orange')
        
        # Plot buy/sell signals
        buy_signals = results[results['signal'] == 1]
        sell_signals = results[results['signal'] == -1]
        
        plt.scatter(buy_signals.index, buy_signals['close'], marker='^', color='green', s=100, label='Buy Signal')
        plt.scatter(sell_signals.index, sell_signals['close'], marker='v', color='red', s=100, label='Sell Signal')
        
        # Plot stop loss and take profit levels
        for i, row in results[results['in_position']].iterrows():
            if row['position'] == 1:  # Long position
                plt.axhline(y=row['stop_loss'], color='red', linestyle='--', alpha=0.3)
                plt.axhline(y=row['take_profit'], color='green', linestyle='--', alpha=0.3)
            elif row['position'] == -1:  # Short position
                plt.axhline(y=row['stop_loss'], color='red', linestyle='--', alpha=0.3)
                plt.axhline(y=row['take_profit'], color='green', linestyle='--', alpha=0.3)
        
        plt.title('Price Chart with Signals')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)
        
        # Plot RSI
        plt.subplot(4, 1, 2)
        plt.plot(results.index, results['rsi'], label='RSI', color='purple')
        plt.axhline(y=70, color='red', linestyle='--')
        plt.axhline(y=50, color='black', linestyle='-')
        plt.axhline(y=30, color='green', linestyle='--')
        plt.title('RSI Indicator')
        plt.ylabel('RSI')
        plt.legend()
        plt.grid(True)
        
        # Plot MACD
        plt.subplot(4, 1, 3)
        plt.plot(results.index, results['macd'], label='MACD', color='blue')
        plt.plot(results.index, results['macd_signal'], label='Signal', color='red')
        plt.bar(results.index, results['macd_hist'], label='Histogram', color='green', alpha=0.3)
        plt.axhline(y=0, color='black', linestyle='-')
        plt.title('MACD Indicator')
        plt.ylabel('MACD')
        plt.legend()
        plt.grid(True)
        
        # Plot equity curve
        plt.subplot(4, 1, 4)
        plt.plot(results.index, results['equity'], label='Equity Curve', color='blue')
        plt.title('Equity Curve')
        plt.ylabel('Equity')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
        
    def print_metrics(self, metrics):
        """
        Print performance metrics
        """
        print("=== Performance Metrics ===")
        print(f"Total Return: {metrics['total_return']:.2%}")
        print(f"Annual Return: {metrics['annual_return']:.2%}")
        print(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
        print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        print(f"Win Rate: {metrics['win_rate']:.2%}")
        print(f"Average Win: {metrics['avg_win']:.2%}")
        print(f"Average Loss: {metrics['avg_loss']:.2%}")
        print(f"Profit Factor: {metrics['profit_factor']:.2f}")
        print(f"Recovery Factor: {metrics['recovery_factor']:.2f}")
        print(f"Number of Trades: {metrics['num_trades']}")
        
    def optimize_parameters(self, data, fast_ma_range, slow_ma_range, rsi_length_range):
        """
        Simple parameter optimization using grid search
        """
        best_sharpe = -float('inf')
        best_params = {}
        
        results_list = []
        
        for fast_ma in fast_ma_range:
            for slow_ma in slow_ma_range:
                if fast_ma >= slow_ma:
                    continue  # Skip invalid combinations
                    
                for rsi_length in rsi_length_range:
                    # Update strategy parameters
                    self.fast_ma_length = fast_ma
                    self.slow_ma_length = slow_ma
                    self.rsi_length = rsi_length
                    
                    # Run backtest
                    _, metrics = self.backtest(data)
                    
                    # Store results
                    result = {
                        'fast_ma': fast_ma,
                        'slow_ma': slow_ma,
                        'rsi_length': rsi_length,
                        'sharpe_ratio': metrics['sharpe_ratio'],
                        'total_return': metrics['total_return'],
                        'max_drawdown': metrics['max_drawdown'],
                        'win_rate': metrics['win_rate'],
                        'num_trades': metrics['num_trades']
                    }
                    results_list.append(result)
                    
                    # Update best parameters
                    if metrics['sharpe_ratio'] > best_sharpe:
                        best_sharpe = metrics['sharpe_ratio']
                        best_params = {
                            'fast_ma': fast_ma,
                            'slow_ma': slow_ma,
                            'rsi_length': rsi_length
                        }
                        
        # Create DataFrame from results
        results_df = pd.DataFrame(results_list)
        
        return best_params, results_df


def generate_synthetic_data(n_days=1000, start_date='2023-01-01', seed=None):
    """
    Generate synthetic price data with trends, reversals, and volatility clusters
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Generate dates
    start = pd.to_datetime(start_date)
    dates = [start + pd.Timedelta(days=i) for i in range(n_days)]
    
    # Base parameters
    initial_price = 100
    daily_returns = np.zeros(n_days)
    
    # Generate a series with trends and reversals
    trend_length = np.random.randint(20, 100)
    trend_direction = 1  # Start with uptrend
    
    # Baseline volatility
    base_volatility = 0.01
    
    # Generate trends with varying length and strength
    for i in range(n_days):
        if i % trend_length == 0:
            trend_direction *= -1  # Reverse trend
            trend_length = np.random.randint(20, 100)  # New trend length
            trend_strength = np.random.uniform(0.0002, 0.0015)  # Trend strength
        
        # Add trend component
        daily_returns[i] += trend_direction * trend_strength
        
        # Add volatility clustering
        volatility = base_volatility * (1 + 0.5 * np.sin(i / 50))
        daily_returns[i] += np.random.normal(0, volatility)
    
    # Generate price series
    prices = initial_price * np.cumprod(1 + daily_returns)
    
    # Generate OHLC data
    daily_range = np.random.uniform(0.005, 0.025, n_days)
    highs = prices * (1 + daily_range/2)
    lows = prices * (1 - daily_range/2)
    opens = np.zeros(n_days)
    
    # First open is same as initial price
    opens[0] = initial_price
    
    # Generate other opens based on previous close and some randomness
    for i in range(1, n_days):
        prev_close = prices[i-1]
        gap = np.random.normal(0, 0.005)
        opens[i] = prev_close * (1 + gap)
    
    # Ensure OHLC integrity
    for i in range(n_days):
        highs[i] = max(highs[i], opens[i], prices[i])
        lows[i] = min(lows[i], opens[i], prices[i])
    
    # Generate volume data with spikes around trend reversals
    base_volume = 1000000
    volumes = np.random.uniform(0.5, 1.5, n_days) * base_volume
    
    # Add volume spikes at trend reversals
    for i in range(1, n_days):
        if abs(daily_returns[i] - daily_returns[i-1]) > 0.02:  # Significant change in return
            volumes[i] *= np.random.uniform(1.5, 3.0)  # Volume spike
    
    # Create DataFrame
    df = pd.DataFrame({
        'open': opens,
        'high': highs,
        'low': lows,
        'close': prices,
        'volume': volumes
    }, index=dates)
    
    return df

def run_simulation():
    """
    Run the simulation on synthetic data
    """
    # Generate synthetic data
    print("Generating synthetic data...")
    data = generate_synthetic_data(n_days=500, seed=42)
    
    # Initialize strategy with default parameters
    strategy = MultiIndicatorTrendFollowingStrategy()
    
    # Run backtest
    print("Running backtest with default parameters...")
    results, metrics = strategy.backtest(data)
    
    # Print metrics
    strategy.print_metrics(metrics)
    
    # Plot results
    strategy.plot_results(results)
    
    # Optimize parameters
    print("\nOptimizing parameters...")
    best_params, opt_results = strategy.optimize_parameters(
        data,
        fast_ma_range=range(5, 21, 5),
        slow_ma_range=range(15, 41, 5),
        rsi_length_range=range(7, 22, 7)
    )
    
    print("\nBest parameters found:")
    print(f"Fast MA Length: {best_params['fast_ma']}")
    print(f"Slow MA Length: {best_params['slow_ma']}")
    print(f"RSI Length: {best_params['rsi_length']}")
    
    # Run backtest with optimized parameters
    print("\nRunning backtest with optimized parameters...")
    strategy.fast_ma_length = best_params['fast_ma']
    strategy.slow_ma_length = best_params['slow_ma']
    strategy.rsi_length = best_params['rsi_length']
    
    optimized_results, optimized_metrics = strategy.backtest(data)
    
    # Print optimized metrics
    print("\nPerformance with optimized parameters:")
    strategy.print_metrics(optimized_metrics)
    
    # Plot optimized results
    strategy.plot_results(optimized_results)
    
    # Plot optimization results
    plt.figure(figsize=(12, 10))
    
    # Scatter plot of parameters vs Sharpe ratio
    plt.subplot(2, 2, 1)
    pivot = opt_results.pivot_table(index='fast_ma', columns='slow_ma', values='sharpe_ratio', aggfunc='mean')
    sns.heatmap(pivot, annot=True, cmap='viridis')
    plt.title('Sharpe Ratio by MA Parameters')
    plt.xlabel('Slow MA Length')
    plt.ylabel('Fast MA Length')
    
    # Bar chart of top 5 parameter combinations
    plt.subplot(2, 2, 2)
    top5 = opt_results.sort_values('sharpe_ratio', ascending=False).head(5)
    labels = [f"F{row['fast_ma']}/S{row['slow_ma']}/R{row['rsi_length']}" for _, row in top5.iterrows()]
    plt.bar(labels, top5['sharpe_ratio'])
    plt.title('Top 5 Parameter Combinations by Sharpe Ratio')
    plt.ylabel('Sharpe Ratio')
    plt.xticks(rotation=45)
    
    # Scatter plot of return vs drawdown
    plt.subplot(2, 2, 3)
    plt.scatter(opt_results['max_drawdown'], opt_results['total_return'], 
                c=opt_results['sharpe_ratio'], cmap='viridis')
    plt.colorbar(label='Sharpe Ratio')
    plt.title('Return vs Drawdown')
    plt.xlabel('Max Drawdown')
    plt.ylabel('Total Return')
    
    # Scatter plot of win rate vs number of trades
    plt.subplot(2, 2, 4)
    plt.scatter(opt_results['num_trades'], opt_results['win_rate'], 
                c=opt_results['sharpe_ratio'], cmap='viridis')
    plt.colorbar(label='Sharpe Ratio')
    plt.title('Win Rate vs Number of Trades')
    plt.xlabel('Number of Trades')
    plt.ylabel('Win Rate')
    
    plt.tight_layout()
    plt.show()
    
    return {
        'default': {
            'results': results,
            'metrics': metrics
        },
        'optimized': {
            'results': optimized_results,
            'metrics': optimized_metrics,
            'parameters': best_params
        },
        'optimization_results': opt_results
    }

# Run the simulation
if __name__ == "__main__":
    results_dict = run_simulation()