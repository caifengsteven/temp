import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import matplotlib.patches as mpatches
from datetime import datetime, timedelta
import random
from sklearn.preprocessing import StandardScaler
try:
    import pywt
except ImportError:
    print("PyWavelets not installed. Wavelet analysis will not be available.")

# Set random seed for reproducibility
np.random.seed(42)

class ForexSimulator:
    """
    A simplified simulator for forex market with order flow
    """
    def __init__(self, initial_price=1.2000):
        self.initial_price = initial_price
        self.price = initial_price
        self.timestamps = []
        self.prices = []
        self.long_positions = []
        self.short_positions = []
        self.order_flows = []
        
    def generate_data(self, num_days=100):
        """
        Generate forex data with price and order flow
        
        Parameters:
        -----------
        num_days : int
            Number of days to simulate
        """
        current_time = datetime(2023, 1, 1)
        
        # Initial values
        price = self.initial_price
        long_pos = 30000
        short_pos = 30000
        
        # Parameters for random walk
        daily_volatility = 0.005  # 0.5% daily volatility
        
        # Generate price series (random walk with mean reversion)
        for day in range(num_days):
            # Create a trend that changes every 20 days
            trend = 0.0002 * np.sin(day / 20 * np.pi)
            
            # Daily price change (random + trend)
            daily_return = np.random.normal(trend, daily_volatility)
            
            # Update price (with mean reversion)
            price_change = daily_return * price
            price = price + price_change
            
            # Ensure price stays reasonable
            price = max(0.5 * self.initial_price, min(1.5 * self.initial_price, price))
            
            # Order flow dynamics - negatively correlated with price movement
            # When price goes up, long positions decrease and short positions increase
            long_change = -np.sign(price_change) * abs(price_change) * 500000 + np.random.normal(0, 2000)
            short_change = np.sign(price_change) * abs(price_change) * 500000 + np.random.normal(0, 2000)
            
            # Update positions
            long_pos = max(10000, long_pos + long_change)
            short_pos = max(10000, short_pos + short_change)
            
            # Generate hourly data for this day
            for hour in range(24):
                # Generate time
                timestamp = current_time + timedelta(days=day, hours=hour)
                
                # Intraday price variation (smaller than daily changes)
                intraday_noise = np.random.normal(0, daily_volatility / 5)
                hourly_price = price * (1 + intraday_noise)
                
                # Intraday order flow variation
                hourly_long = long_pos + np.random.normal(0, long_pos * 0.02)
                hourly_short = short_pos + np.random.normal(0, short_pos * 0.02)
                
                # Occasionally create signal conditions
                if random.random() < 0.05:  # 5% chance each hour
                    if random.random() < 0.5:  # 50% chance for buy signal
                        hourly_long *= 1.1  # Increase long positions
                        hourly_short *= 0.9  # Decrease short positions
                    else:  # 50% chance for sell signal
                        hourly_long *= 0.9  # Decrease long positions
                        hourly_short *= 1.1  # Increase short positions
                
                # Store data
                self.timestamps.append(timestamp)
                self.prices.append(hourly_price)
                self.long_positions.append(hourly_long)
                self.short_positions.append(hourly_short)
                self.order_flows.append(hourly_long - hourly_short)
        
        # Convert to DataFrame
        self.data = pd.DataFrame({
            'timestamp': self.timestamps,
            'price': self.prices,
            'long_positions': self.long_positions,
            'short_positions': self.short_positions,
            'order_flow': self.order_flows
        })
        
        return self.data

class OrderFlowTradingStrategy:
    """
    Trading strategy based on order flow data
    """
    def __init__(self, data, ma_short=8, ma_long=72):
        """
        Initialize the trading strategy
        
        Parameters:
        -----------
        data : pandas.DataFrame
            DataFrame containing price and order flow data
        ma_short : int
            Short period for moving average (in hours)
        ma_long : int
            Long period for moving average (in hours)
        """
        self.data = data.copy()
        self.ma_short = ma_short
        self.ma_long = ma_long
        
        # Calculate moving averages
        self.data['sma_short'] = self.data['price'].rolling(window=ma_short).mean()
        self.data['sma_long'] = self.data['price'].rolling(window=ma_long).mean()
        
        # Calculate exponential moving averages
        self.data['ema_short'] = self.data['price'].ewm(span=ma_short, adjust=False).mean()
        self.data['ema_long'] = self.data['price'].ewm(span=ma_long, adjust=False).mean()
        
        # Calculate order flow ratio
        total_positions = self.data['long_positions'] + self.data['short_positions']
        self.data['long_ratio'] = self.data['long_positions'] / total_positions
        self.data['short_ratio'] = self.data['short_positions'] / total_positions
        
        # Calculate position changes
        self.data['long_change'] = self.data['long_positions'].pct_change(5)
        self.data['short_change'] = self.data['short_positions'].pct_change(5)
        
        # Initialize signals column
        self.data['signal'] = 0
        
        # Add hour of day column for end-of-day logic
        self.data['hour'] = self.data['timestamp'].dt.hour
        
        # Drop NaN values from calculations
        self.data = self.data.dropna()
    
    def generate_signals(self):
        """
        Generate trading signals based on order flow and moving averages
        """
        # Buy signal conditions:
        # 1. Long positions ratio > 0.5 (more longs than shorts)
        # 2. Long positions are increasing over short time frame
        # 3. Short MA is above long MA (uptrend)
        
        # Sell signal conditions:
        # 1. Short positions ratio > 0.5 (more shorts than longs)
        # 2. Short positions are increasing over short time frame
        # 3. Short MA is below long MA (downtrend)
        
        buy_conditions = (
            (self.data['long_ratio'] > 0.52) &  # More longs than shorts
            (self.data['long_change'] > 0.02) &  # Long positions increased by at least 2%
            (self.data['ema_short'] > self.data['ema_long'])  # Uptrend
        )
        
        sell_conditions = (
            (self.data['short_ratio'] > 0.52) &  # More shorts than longs
            (self.data['short_change'] > 0.02) &  # Short positions increased by at least 2%
            (self.data['ema_short'] < self.data['ema_long'])  # Downtrend
        )
        
        # Apply signals
        self.data.loc[buy_conditions, 'signal'] = 1
        self.data.loc[sell_conditions, 'signal'] = -1
        
        # Count signals
        buy_signals = (self.data['signal'] == 1).sum()
        sell_signals = (self.data['signal'] == -1).sum()
        
        print(f"Generated {buy_signals} buy signals and {sell_signals} sell signals")
        
        return self.data
    
    def backtest(self, initial_capital=10000.0, position_size=0.2, stop_loss_pips=40, take_profit_pips=10):
        """
        Backtest the trading strategy
        
        Parameters:
        -----------
        initial_capital : float
            Initial capital
        position_size : float
            Size of each position as a percentage of capital
        stop_loss_pips : int
            Stop loss in pips
        take_profit_pips : int
            Take profit in pips
        """
        # Create a copy of the data for backtesting
        bt_data = self.data.copy()
        
        # Convert pips to actual price movements
        stop_loss = stop_loss_pips * 0.0001
        take_profit = take_profit_pips * 0.0001
        
        # Add columns for tracking trades
        bt_data['position'] = 0
        bt_data['entry_price'] = 0.0
        bt_data['exit_price'] = 0.0
        bt_data['pnl_pips'] = 0.0
        bt_data['capital'] = initial_capital
        bt_data['returns'] = 0.0  # Daily returns for Sharpe calculation
        
        # Track active position info
        active_position = 0
        entry_price = 0.0
        entry_index = 0
        
        # Process each bar (hour)
        for i in range(1, len(bt_data)):
            current_bar = bt_data.iloc[i]
            prev_bar = bt_data.iloc[i-1]
            
            # Copy previous capital value as default
            bt_data.loc[bt_data.index[i], 'capital'] = prev_bar['capital']
            
            # Check if we have an active position
            if active_position != 0:
                # Calculate current profit/loss
                if active_position == 1:  # Long position
                    current_pnl = current_bar['price'] - entry_price
                else:  # Short position
                    current_pnl = entry_price - current_bar['price']
                
                # Check if we should exit the trade
                hours_in_trade = i - entry_index
                hour_of_day = current_bar['hour']
                
                # Exit conditions: take profit, stop loss, or end of day
                if ((current_pnl >= take_profit and hours_in_trade >= 1) or 
                    current_pnl <= -stop_loss or 
                    hour_of_day == 21):
                    
                    # Record exit details
                    bt_data.loc[bt_data.index[i], 'exit_price'] = current_bar['price']
                    bt_data.loc[bt_data.index[i], 'pnl_pips'] = current_pnl * 10000
                    
                    # Update capital
                    trade_size = initial_capital * position_size
                    pnl_dollars = trade_size * current_pnl / entry_price
                    bt_data.loc[bt_data.index[i], 'capital'] = prev_bar['capital'] + pnl_dollars
                    
                    # Calculate returns for this bar
                    bt_data.loc[bt_data.index[i], 'returns'] = pnl_dollars / prev_bar['capital']
                    
                    # Clear position
                    active_position = 0
                    entry_price = 0.0
            
            # Track current position
            bt_data.loc[bt_data.index[i], 'position'] = active_position
            
            # Check for new entry signals if we don't have an active position
            if active_position == 0 and current_bar['signal'] != 0:
                # Open new position
                active_position = current_bar['signal']  # 1 or -1
                entry_price = current_bar['price']
                entry_index = i
                
                # Record entry price
                bt_data.loc[bt_data.index[i], 'entry_price'] = entry_price
        
        # Calculate performance metrics
        trades = bt_data[bt_data['exit_price'] > 0]
        total_trades = len(trades)
        win_trades = len(trades[trades['pnl_pips'] > 0])
        loss_trades = len(trades[trades['pnl_pips'] <= 0])
        
        # Calculate win rate and returns
        win_rate = win_trades / total_trades * 100 if total_trades > 0 else 0
        final_capital = bt_data['capital'].iloc[-1]
        total_return = (final_capital - initial_capital) / initial_capital * 100
        
        # Calculate drawdown
        bt_data['peak'] = bt_data['capital'].cummax()
        bt_data['drawdown'] = (bt_data['capital'] - bt_data['peak']) / bt_data['peak'] * 100
        max_drawdown = bt_data['drawdown'].min()
        
        # Calculate daily returns and Sharpe ratio
        # Group by date and get the last capital value of each day
        bt_data['date'] = bt_data['timestamp'].dt.date
        daily_equity = bt_data.groupby('date')['capital'].last().reset_index()
        daily_equity['daily_return'] = daily_equity['capital'].pct_change()
        
        # Calculate annualized Sharpe ratio (assuming 252 trading days)
        daily_returns = daily_equity['daily_return'].dropna()
        if len(daily_returns) > 0:
            sharpe_ratio = (np.mean(daily_returns) / np.std(daily_returns)) * np.sqrt(252)
        else:
            sharpe_ratio = 0
        
        # Print performance summary
        print(f"Total trades: {total_trades}")
        print(f"Win trades: {win_trades}")
        print(f"Loss trades: {loss_trades}")
        print(f"Win rate: {win_rate:.2f}%")
        print(f"Total return: {total_return:.2f}%")
        print(f"Maximum drawdown: {-max_drawdown:.2f}%")
        print(f"Sharpe ratio: {sharpe_ratio:.2f}")
        
        return bt_data, daily_equity, sharpe_ratio
    
    def plot_performance(self, bt_data, daily_equity, sharpe_ratio):
        """
        Plot the performance of the trading strategy
        
        Parameters:
        -----------
        bt_data : pandas.DataFrame
            DataFrame containing backtest results
        daily_equity : pandas.DataFrame
            DataFrame containing daily equity curve
        sharpe_ratio : float
            Calculated Sharpe ratio
        """
        # Create figure with 3 subplots
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12), sharex=True, 
                                          gridspec_kw={'height_ratios': [3, 1, 1]})
        
        # Plot price and moving averages
        ax1.plot(bt_data['timestamp'], bt_data['price'], label='Price', color='blue', alpha=0.7)
        ax1.plot(bt_data['timestamp'], bt_data['ema_short'], label=f'EMA {self.ma_short}', color='orange')
        ax1.plot(bt_data['timestamp'], bt_data['ema_long'], label=f'EMA {self.ma_long}', color='green')
        
        # Plot buy and sell signals
        buys = bt_data[bt_data['signal'] == 1]
        sells = bt_data[bt_data['signal'] == -1]
        
        ax1.scatter(buys['timestamp'], buys['price'], marker='^', color='green', s=100, label='Buy Signal')
        ax1.scatter(sells['timestamp'], sells['price'], marker='v', color='red', s=100, label='Sell Signal')
        
        # Plot trade entries and exits
        entries = bt_data[bt_data['entry_price'] > 0]
        exits = bt_data[bt_data['exit_price'] > 0]
        
        # Get profitable and losing trades
        profitable_exits = exits[exits['pnl_pips'] > 0]
        losing_exits = exits[exits['pnl_pips'] <= 0]
        
        ax1.scatter(entries['timestamp'], entries['entry_price'], marker='o', color='blue', s=80, label='Entry')
        ax1.scatter(profitable_exits['timestamp'], profitable_exits['exit_price'], 
                   marker='x', color='lime', s=100, label='Profit Exit')
        ax1.scatter(losing_exits['timestamp'], losing_exits['exit_price'], 
                   marker='x', color='red', s=100, label='Loss Exit')
        
        # Plot account equity
        ax2.plot(bt_data['timestamp'], bt_data['capital'], color='purple', label='Capital')
        
        # Add horizontal line at initial capital
        ax2.axhline(y=10000, color='black', linestyle='--', alpha=0.5)
        
        # Plot drawdown
        drawdown_color = np.where(bt_data['drawdown'] < 0, 'red', 'green')
        ax2.fill_between(bt_data['timestamp'], 0, bt_data['drawdown'], color='red', alpha=0.3, label='Drawdown')
        
        # Plot order flow
        ax3.plot(bt_data['timestamp'], bt_data['long_positions'], color='green', label='Long Positions')
        ax3.plot(bt_data['timestamp'], bt_data['short_positions'], color='red', label='Short Positions')
        ax3.plot(bt_data['timestamp'], bt_data['order_flow'], color='blue', label='Order Flow')
        
        # Add zero line
        ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        # Add titles and labels
        ax1.set_title(f'Order Flow Trading Strategy (Sharpe Ratio: {sharpe_ratio:.2f})')
        ax1.set_ylabel('Price')
        ax1.legend(loc='upper left')
        
        ax2.set_ylabel('Capital / Drawdown')
        ax2.legend(loc='upper left')
        
        ax3.set_xlabel('Time')
        ax3.set_ylabel('Order Flow')
        ax3.legend(loc='upper left')
        
        plt.tight_layout()
        plt.show()
        
        # Plot daily equity curve and drawdowns
        self.plot_equity_curve(daily_equity)
    
    def plot_equity_curve(self, daily_equity):
        """
        Plot detailed equity curve and drawdown
        
        Parameters:
        -----------
        daily_equity : pandas.DataFrame
            DataFrame containing daily equity data
        """
        # Calculate drawdown
        daily_equity['peak'] = daily_equity['capital'].cummax()
        daily_equity['drawdown'] = (daily_equity['capital'] - daily_equity['peak']) / daily_equity['peak'] * 100
        
        # Create figure with 2 subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True, 
                                     gridspec_kw={'height_ratios': [2, 1]})
        
        # Plot equity curve
        ax1.plot(daily_equity['date'], daily_equity['capital'], 'b-', linewidth=2, label='Equity Curve')
        ax1.axhline(y=10000, color='r', linestyle='--', alpha=0.5, label='Initial Capital')
        ax1.set_title('Equity Curve and Drawdown')
        ax1.set_ylabel('Capital ($)')
        ax1.legend()
        ax1.grid(True)
        
        # Format y-axis with dollar signs
        ax1.get_yaxis().set_major_formatter(
            plt.FuncFormatter(lambda x, p: f'${int(x):,}'))
        
        # Plot drawdown
        ax2.fill_between(daily_equity['date'], 0, daily_equity['drawdown'], color='r', alpha=0.3)
        ax2.plot(daily_equity['date'], daily_equity['drawdown'], 'r-', linewidth=1)
        ax2.set_ylabel('Drawdown (%)')
        ax2.set_xlabel('Date')
        ax2.grid(True)
        
        # Highlight maximum drawdown
        max_drawdown_idx = daily_equity['drawdown'].idxmin()
        max_drawdown_date = daily_equity.loc[max_drawdown_idx, 'date']
        max_drawdown_value = daily_equity.loc[max_drawdown_idx, 'drawdown']
        
        ax2.annotate(f'Max Drawdown: {max_drawdown_value:.2f}%', 
                   xy=(max_drawdown_date, max_drawdown_value),
                   xytext=(max_drawdown_date, max_drawdown_value - 5),
                   arrowprops=dict(facecolor='black', shrink=0.05, width=1.5))
        
        plt.tight_layout()
        plt.show()
        
        # Plot daily returns histogram
        self.plot_returns_distribution(daily_equity)
    
    def plot_returns_distribution(self, daily_equity):
        """
        Plot distribution of daily returns
        
        Parameters:
        -----------
        daily_equity : pandas.DataFrame
            DataFrame containing daily equity data
        """
        # Calculate statistics
        daily_returns = daily_equity['daily_return'].dropna()
        mean_return = daily_returns.mean()
        std_return = daily_returns.std()
        skew = stats.skew(daily_returns)
        kurtosis = stats.kurtosis(daily_returns)
        
        # Annualize statistics (assuming 252 trading days)
        annual_return = mean_return * 252 * 100
        annual_volatility = std_return * np.sqrt(252) * 100
        sharpe_ratio = annual_return / annual_volatility if annual_volatility > 0 else 0
        
        # Create figure
        plt.figure(figsize=(12, 8))
        
        # Plot histogram of returns
        n, bins, patches = plt.hist(daily_returns, bins=50, alpha=0.7, color='skyblue', 
                                   density=True, label='Daily Returns')
        
        # Plot normal distribution for comparison
        x = np.linspace(min(daily_returns), max(daily_returns), 1000)
        plt.plot(x, stats.norm.pdf(x, mean_return, std_return), 'r-', linewidth=2, 
                label='Normal Distribution')
        
        # Add vertical line at mean
        plt.axvline(x=mean_return, color='green', linestyle='--', alpha=0.7, 
                  label=f'Mean: {mean_return*100:.4f}%')
        
        # Add vertical line at zero
        plt.axvline(x=0, color='black', linestyle='--', alpha=0.7)
        
        # Add statistics as text
        stats_text = (
            f'Annual Return: {annual_return:.2f}%\n'
            f'Annual Volatility: {annual_volatility:.2f}%\n'
            f'Sharpe Ratio: {sharpe_ratio:.2f}\n'
            f'Skewness: {skew:.2f}\n'
            f'Kurtosis: {kurtosis:.2f}'
        )
        
        plt.annotate(stats_text, xy=(0.05, 0.95), xycoords='axes fraction', 
                    bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8),
                    va='top', fontsize=12)
        
        # Add labels and title
        plt.xlabel('Daily Return')
        plt.ylabel('Density')
        plt.title('Distribution of Daily Returns')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

def plot_data(df):
    """
    Plot forex data
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing forex data
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Plot price
    ax1.plot(df['timestamp'], df['price'], label='Price')
    ax1.set_ylabel('Price')
    ax1.set_title('Forex Data Overview')
    ax1.legend()
    
    # Plot order flow
    ax2.plot(df['timestamp'], df['long_positions'], label='Long Positions', color='green')
    ax2.plot(df['timestamp'], df['short_positions'], label='Short Positions', color='red')
    ax2.plot(df['timestamp'], df['order_flow'], label='Order Flow (Long - Short)', color='blue')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Positions')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()

def compute_correlation(df):
    """
    Compute correlation between price and order flow
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing forex data
    """
    # Calculate Pearson correlation
    corr_price_orderflow = stats.pearsonr(df['price'], df['order_flow'])
    corr_price_long = stats.pearsonr(df['price'], df['long_positions'])
    corr_price_short = stats.pearsonr(df['price'], df['short_positions'])
    
    # Print results
    print("Correlation Analysis:")
    print(f"Price vs Order Flow: {corr_price_orderflow[0]:.4f} (p-value: {corr_price_orderflow[1]:.4e})")
    print(f"Price vs Long Positions: {corr_price_long[0]:.4f} (p-value: {corr_price_long[1]:.4e})")
    print(f"Price vs Short Positions: {corr_price_short[0]:.4f} (p-value: {corr_price_short[1]:.4e})")

# Run simulation and backtesting for different currency pairs
pairs = [
    {"name": "EUR/USD", "initial_price": 1.2000},
    {"name": "GBP/USD", "initial_price": 1.3500},
    {"name": "EUR/GBP", "initial_price": 0.8900}
]

backtest_results = {}
sharpe_ratios = {}

for pair in pairs:
    print(f"\n\n=============== {pair['name']} ===============")
    
    # Generate simulated data
    print(f"Simulating {pair['name']} data...")
    simulator = ForexSimulator(initial_price=pair['initial_price'])
    df = simulator.generate_data(num_days=100)
    
    # Display data statistics
    print("\nData statistics:")
    print(df.describe())
    
    # Plot data
    plot_data(df)
    
    # Calculate correlation
    compute_correlation(df)
    
    # Create and apply trading strategy
    print("\nApplying order flow trading strategy...")
    strategy = OrderFlowTradingStrategy(df, ma_short=8, ma_long=72)
    df_signals = strategy.generate_signals()
    
    # Backtest strategy
    print("\nBacktesting strategy...")
    results, daily_equity, sharpe_ratio = strategy.backtest(initial_capital=10000.0, position_size=0.2, 
                                                         stop_loss_pips=40, take_profit_pips=10)
    backtest_results[pair['name']] = results
    sharpe_ratios[pair['name']] = sharpe_ratio
    
    # Plot performance
    strategy.plot_performance(results, daily_equity, sharpe_ratio)

# Compare performance across currency pairs
print("\n\nPerformance Comparison:")
print("=" * 75)
print(f"{'Currency Pair':<12} | {'Win Rate':<10} | {'Return':<10} | {'Max Drawdown':<15} | {'Sharpe Ratio':<12}")
print("-" * 75)

for pair_name, results in backtest_results.items():
    trades = results[results['exit_price'] > 0]
    total_trades = len(trades)
    
    if total_trades > 0:
        win_trades = len(trades[trades['pnl_pips'] > 0])
        win_rate = win_trades / total_trades * 100
        
        final_capital = results['capital'].iloc[-1]
        initial_capital = 10000.0
        total_return = (final_capital - initial_capital) / initial_capital * 100
        
        max_drawdown = results['drawdown'].min()
        sharpe = sharpe_ratios[pair_name]
        
        print(f"{pair_name:<12} | {win_rate:8.2f}% | {total_return:8.2f}% | {-max_drawdown:13.2f}% | {sharpe:10.2f}")
    else:
        print(f"{pair_name:<12} | No trades executed")