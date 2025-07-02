import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta
import seaborn as sns
from tabulate import tabulate
import warnings
warnings.filterwarnings('ignore')

class BollingerBreakoutStrategy:
    """
    Multi-Period Bollinger Breakout Momentum Capture Strategy
    
    This strategy captures strong upward momentum by entering long positions when price breaks
    above the upper Bollinger Band and exiting when price falls below the lower band.
    """
    
    def __init__(self, 
                 length=20,
                 ma_type="SMA",
                 std_dev_mult=2.0,
                 initial_capital=100000,
                 commission=0.001,
                 position_size_percent=100.0):
        """
        Initialize the strategy with configurable parameters
        
        Parameters:
        -----------
        length: int
            Period length for Bollinger Bands calculation
        ma_type: str
            Type of moving average ('SMA', 'EMA', 'SMMA', 'WMA', 'VWMA')
        std_dev_mult: float
            Standard deviation multiplier for band width
        initial_capital: float
            Initial capital for backtesting
        commission: float
            Commission rate per trade (e.g., 0.1% = 0.001)
        position_size_percent: float
            Percentage of equity to use per trade
        """
        # Strategy parameters
        self.length = length
        self.ma_type = ma_type
        self.std_dev_mult = std_dev_mult
        self.position_size_percent = position_size_percent
        
        # Backtest parameters
        self.initial_capital = initial_capital
        self.equity = initial_capital
        self.commission = commission
        
        # Trading variables
        self.position = 0  # 0: no position, 1: long
        self.entry_price = 0
        self.position_size = 0
        
        # Results tracking
        self.equity_curve = []
        self.trades = []
        self.current_trade = None
    
    def calculate_moving_average(self, data, source_column='close'):
        """
        Calculate moving average based on the specified type
        
        Parameters:
        -----------
        data: DataFrame
            Price data with OHLCV columns
        source_column: str
            Column to use for calculations
            
        Returns:
        --------
        ma_series: Series
            Moving average series
        """
        if self.ma_type == "SMA":
            return data[source_column].rolling(window=self.length).mean()
        elif self.ma_type == "EMA":
            return data[source_column].ewm(span=self.length, adjust=False).mean()
        elif self.ma_type == "SMMA" or self.ma_type == "RMA":
            return data[source_column].ewm(alpha=1/self.length, adjust=False).mean()
        elif self.ma_type == "WMA":
            weights = np.arange(1, self.length + 1)
            return data[source_column].rolling(window=self.length).apply(
                lambda x: np.sum(weights * x) / np.sum(weights), raw=True)
        elif self.ma_type == "VWMA":
            return (data[source_column] * data['volume']).rolling(window=self.length).sum() / \
                   data['volume'].rolling(window=self.length).sum()
        else:
            # Default to SMA
            return data[source_column].rolling(window=self.length).mean()
    
    def calculate_indicators(self, data):
        """
        Calculate Bollinger Bands and strategy indicators
        
        Parameters:
        -----------
        data: DataFrame
            OHLCV data
            
        Returns:
        --------
        data: DataFrame
            Data with added indicators
        """
        # Calculate middle band (moving average)
        data['basis'] = self.calculate_moving_average(data)
        
        # Calculate standard deviation
        data['std_dev'] = data['close'].rolling(window=self.length).std()
        
        # Calculate upper and lower bands
        data['upper_band'] = data['basis'] + (data['std_dev'] * self.std_dev_mult)
        data['lower_band'] = data['basis'] - (data['std_dev'] * self.std_dev_mult)
        
        # Calculate entry and exit signals
        data['enter_long'] = np.where(
            (data['close'] > data['upper_band']) & 
            (data['close'].shift(1) <= data['upper_band'].shift(1)),
            1, 0
        )
        
        data['exit_long'] = np.where(
            (data['close'] < data['lower_band']) & 
            (data['close'].shift(1) >= data['lower_band'].shift(1)),
            1, 0
        )
        
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
        start_idx = self.length
        
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
            
            # Check for exit signal if in a position
            if self.position == 1 and current_bar['exit_long'] == 1:
                # Exit long position
                exit_price = current_bar['close']
                
                # Calculate profit/loss
                price_change = exit_price - self.entry_price
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
                    'exit_reason': "Price below lower band",
                    'profit_loss': profit_loss,
                    'profit_loss_pct': (profit_loss / self.current_trade['initial_equity']) * 100,
                    'equity': self.equity
                })
                self.trades.append(self.current_trade)
                
                # Reset position
                self.position = 0
                self.position_size = 0
                self.current_trade = None
            
            # Check for entry signal if not in a position
            if self.position == 0 and current_bar['enter_long'] == 1:
                # Enter long position
                self.position = 1
                self.entry_price = current_bar['close']
                
                # Calculate position size
                self.position_size = (self.equity * (self.position_size_percent / 100)) / self.entry_price
                
                # Account for commission
                commission_cost = self.position_size * self.entry_price * self.commission
                self.equity -= commission_cost
                
                # Record trade
                self.current_trade = {
                    'entry_date': current_bar.name,
                    'entry_price': self.entry_price,
                    'position': 'Long',
                    'position_size': self.position_size,
                    'initial_equity': self.equity
                }
        
        # Close any open position at the end of the backtest
        if self.position != 0:
            final_bar = data.iloc[-1]
            exit_price = final_bar['close']
            
            # Calculate profit/loss
            price_change = exit_price - self.entry_price
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
                'exit_reason': "End of backtest",
                'profit_loss': profit_loss,
                'profit_loss_pct': (profit_loss / self.current_trade['initial_equity']) * 100,
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
        durations = []
        for t in self.trades:
            if 'exit_date' in t and 'entry_date' in t:
                duration = (t['exit_date'] - t['entry_date']).total_seconds() / (3600 * 24)  # in days
                durations.append(duration)
        
        self.metrics['avg_trade_duration'] = np.mean(durations) if durations else 0
        
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
        
        # Plot price with Bollinger Bands and trades
        if data is not None:
            plt.subplot(2, 1, 2)
            
            # Plot price and Bollinger Bands
            data['close'].plot(label='Close Price', color='black', alpha=0.7)
            data['basis'].plot(label=f'{self.ma_type} ({self.length})', color='blue', alpha=0.7)
            data['upper_band'].plot(label='Upper Band', color='red', alpha=0.7)
            data['lower_band'].plot(label='Lower Band', color='green', alpha=0.7)
            
            # Fill between bands
            plt.fill_between(data.index, data['upper_band'], data['lower_band'], color='blue', alpha=0.1)
            
            # Plot entry and exit points for trades
            for trade in self.trades:
                plt.scatter(trade['entry_date'], trade['entry_price'], marker='^', color='green', s=100, label='Entry' if 'Entry' not in plt.gca().get_legend_handles_labels()[1] else "")
                
                if 'exit_date' in trade and 'exit_price' in trade:
                    plt.scatter(trade['exit_date'], trade['exit_price'], marker='v', color='red', s=100, label='Exit' if 'Exit' not in plt.gca().get_legend_handles_labels()[1] else "")
            
            plt.title('Price Chart with Bollinger Bands and Trading Signals')
            plt.xlabel('Date')
            plt.ylabel('Price')
            plt.grid(True)
            plt.legend()
        
        plt.tight_layout()
        plt.savefig('bollinger_breakout_strategy_results.png')
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
        
        # Plot monthly returns
        plt.subplot(2, 2, 4)
        daily_returns = self.equity_curve.pct_change().dropna()
        if len(daily_returns) > 0:
            monthly_returns = daily_returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
            monthly_returns.plot(kind='bar')
            plt.title('Monthly Returns')
            plt.xlabel('Month')
            plt.ylabel('Return %')
            plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('bollinger_breakout_strategy_performance.png')
        plt.show()
        
        # Plot Bollinger Band analysis
        if data is not None:
            plt.figure(figsize=figsize)
            
            # Plot price distance from bands
            plt.subplot(2, 1, 1)
            
            # Calculate percent distance from middle band
            data['percent_b'] = (data['close'] - data['lower_band']) / (data['upper_band'] - data['lower_band'])
            data['percent_b'].plot(label='%B (Position within Bands)')
            plt.axhline(y=1.0, color='red', linestyle='--', label='Upper Band Level')
            plt.axhline(y=0.5, color='blue', linestyle='--', label='Middle Band Level')
            plt.axhline(y=0.0, color='green', linestyle='--', label='Lower Band Level')
            
            plt.title('Price Position within Bollinger Bands (%B)')
            plt.xlabel('Date')
            plt.ylabel('Position (0-1)')
            plt.grid(True)
            plt.legend()
            
            # Plot band width
            plt.subplot(2, 1, 2)
            
            # Calculate bandwidth
            data['band_width'] = (data['upper_band'] - data['lower_band']) / data['basis']
            data['band_width'].plot(label='Bandwidth')
            
            plt.title('Bollinger Band Width (Volatility)')
            plt.xlabel('Date')
            plt.ylabel('Band Width')
            plt.grid(True)
            plt.legend()
            
            plt.tight_layout()
            plt.savefig('bollinger_breakout_strategy_band_analysis.png')
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
        print(f"Average Trade Duration: {self.metrics['avg_trade_duration']:.2f} days")
        
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
                    f"${trade.get('profit_loss', 0):.2f}",
                    f"{trade.get('profit_loss_pct', 0):.2f}%"
                ])
            else:
                trade_data.extend(['N/A', 'N/A', 'N/A', 'N/A', 'N/A'])
                
            trades_table.append(trade_data)
        
        headers = ["#", "Entry Date", "Position", "Entry Price", "Exit Date", "Exit Price", 
                 "Exit Reason", "P&L", "P&L %"]
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
    for length in param_grid['length']:
        for ma_type in param_grid['ma_type']:
            for std_dev_mult in param_grid['std_dev_mult']:
                param_combinations.append({
                    'length': length,
                    'ma_type': ma_type,
                    'std_dev_mult': std_dev_mult,
                })
    
    total_combinations = len(param_combinations)
    print(f"Testing {total_combinations} parameter combinations...")
    
    for i, params in enumerate(param_combinations):
        # Create strategy with current parameters
        strategy = BollingerBreakoutStrategy(
            length=params['length'],
            ma_type=params['ma_type'],
            std_dev_mult=params['std_dev_mult']
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
        symbol = "BTC-USD"  # Bitcoin USD
        start_date = "2020-01-01"
        end_date = "2023-01-01"
        data = fetch_data(symbol, start_date, end_date)
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
            'length': [10, 20, 30],
            'ma_type': ['SMA', 'EMA'],
            'std_dev_mult': [1.5, 2.0, 2.5],
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
        print(top_results[['length', 'ma_type', 'std_dev_mult', 
                         'sharpe_ratio', 'total_return', 'max_drawdown', 'win_rate']])
        
        # Use best parameters for final backtest
        strategy = BollingerBreakoutStrategy(
            length=best_params['length'],
            ma_type=best_params['ma_type'],
            std_dev_mult=best_params['std_dev_mult']
        )
    else:
        # Use default parameters
        strategy = BollingerBreakoutStrategy()
    
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