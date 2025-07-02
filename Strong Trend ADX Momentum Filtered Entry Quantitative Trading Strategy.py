import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

class ADXMomentumStrategy:
    """
    Implementation of the Strong Trend ADX Momentum Filtered Entry Quantitative Trading Strategy
    
    This strategy uses ADX and DMI indicators to identify strong trends and filter out weak market movements.
    """
    
    def __init__(self, 
                 adx_length=14, 
                 adx_threshold=25, 
                 position_pct=50.0, 
                 max_pyramiding=5,
                 commission_pct=0.05):
        """
        Initialize the strategy with parameters
        
        Parameters:
        -----------
        adx_length : int
            Period for ADX calculation (default: 14)
        adx_threshold : int
            Minimum ADX value to consider a trend strong (default: 25)
        position_pct : float
            Percentage of available funds to use per entry (default: 50.0)
        max_pyramiding : int
            Maximum number of additional positions in the same direction (default: 5)
        commission_pct : float
            Commission percentage per trade (default: 0.05%)
        """
        self.adx_length = adx_length
        self.adx_threshold = adx_threshold
        self.position_pct = position_pct / 100.0  # Convert to decimal
        self.max_pyramiding = max_pyramiding
        self.commission_pct = commission_pct / 100.0  # Convert to decimal
    
    def calculate_directional_movement(self, data):
        """
        Calculate Directional Movement components
        
        Parameters:
        -----------
        data : DataFrame
            Price data with OHLC columns
            
        Returns:
        --------
        DataFrame
            Data with Directional Movement components
        """
        df = data.copy()
        
        # Calculate True Range
        df['tr1'] = abs(df['high'] - df['low'])
        df['tr2'] = abs(df['high'] - df['close'].shift())
        df['tr3'] = abs(df['low'] - df['close'].shift())
        df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
        
        # Calculate Directional Movement
        df['up_move'] = df['high'] - df['high'].shift()
        df['down_move'] = df['low'].shift() - df['low']
        
        # Calculate Positive Directional Movement (+DM)
        df['+dm'] = np.where(
            (df['up_move'] > df['down_move']) & (df['up_move'] > 0),
            df['up_move'],
            0
        )
        
        # Calculate Negative Directional Movement (-DM)
        df['-dm'] = np.where(
            (df['down_move'] > df['up_move']) & (df['down_move'] > 0),
            df['down_move'],
            0
        )
        
        return df
    
    def calculate_directional_indicators(self, data):
        """
        Calculate ADX and Directional Movement Indicators
        
        Parameters:
        -----------
        data : DataFrame
            Price data with Directional Movement components
            
        Returns:
        --------
        DataFrame
            Data with ADX and Directional Movement Indicators
        """
        df = data.copy()
        
        # Calculate smoothed True Range and Directional Movement
        df['tr_' + str(self.adx_length)] = df['tr'].rolling(window=self.adx_length).sum()
        df['+dm_' + str(self.adx_length)] = df['+dm'].rolling(window=self.adx_length).sum()
        df['-dm_' + str(self.adx_length)] = df['-dm'].rolling(window=self.adx_length).sum()
        
        # Calculate +DI and -DI
        df['+di'] = 100 * df['+dm_' + str(self.adx_length)] / df['tr_' + str(self.adx_length)]
        df['-di'] = 100 * df['-dm_' + str(self.adx_length)] / df['tr_' + str(self.adx_length)]
        
        # Calculate Directional Index (DX)
        df['dx'] = 100 * abs(df['+di'] - df['-di']) / (df['+di'] + df['-di'])
        
        # Calculate Average Directional Index (ADX)
        df['adx'] = df['dx'].rolling(window=self.adx_length).mean()
        
        return df
    
    def calculate_indicators(self, data):
        """
        Calculate all indicators and conditions needed for the strategy
        
        Parameters:
        -----------
        data : DataFrame
            Price data with OHLC columns
            
        Returns:
        --------
        DataFrame
            Data with added indicator columns
        """
        # Calculate Directional Movement components
        df = self.calculate_directional_movement(data)
        
        # Calculate Directional Indicators
        df = self.calculate_directional_indicators(df)
        
        # Calculate trend conditions
        df['is_strong_trend'] = df['adx'] > self.adx_threshold
        df['plus_di_gt_minus_di'] = df['+di'] > df['-di']
        df['minus_di_gt_plus_di'] = df['-di'] > df['+di']
        
        # Calculate crossovers
        df['plus_di_cross_above_minus_di'] = (df['+di'] > df['-di']) & (df['+di'].shift() <= df['-di'].shift())
        df['minus_di_cross_above_plus_di'] = (df['-di'] > df['+di']) & (df['-di'].shift() <= df['+di'].shift())
        
        return df
    
    def backtest(self, data, initial_capital=10000):
        """
        Run backtest on the provided data
        
        Parameters:
        -----------
        data : DataFrame
            Price data with OHLC columns
        initial_capital : float
            Initial capital for the backtest
            
        Returns:
        --------
        DataFrame
            Data with added signal and performance columns
        """
        # Calculate indicators
        df = self.calculate_indicators(data)
        
        # Initialize trading variables
        df['position'] = 0  # 1 for long, -1 for short, 0 for flat
        df['position_count'] = 0  # Number of pyramid positions
        df['entry_price'] = np.nan
        df['equity'] = initial_capital
        df['cash'] = initial_capital
        df['holdings'] = 0
        df['trade_pnl'] = 0
        df['trade_result'] = None
        
        # Process each bar
        position = 0  # 0 for flat, 1 for long, -1 for short
        position_count = 0
        entry_price = 0
        position_size = 0
        
        for i in range(self.adx_length * 2, len(df)):
            # Default is to carry forward previous values
            if i > 0:
                df.loc[df.index[i], 'position'] = df.loc[df.index[i-1], 'position']
                df.loc[df.index[i], 'position_count'] = df.loc[df.index[i-1], 'position_count']
                df.loc[df.index[i], 'entry_price'] = df.loc[df.index[i-1], 'entry_price']
                df.loc[df.index[i], 'equity'] = df.loc[df.index[i-1], 'equity']
                df.loc[df.index[i], 'cash'] = df.loc[df.index[i-1], 'cash']
                df.loc[df.index[i], 'holdings'] = df.loc[df.index[i-1], 'holdings']
            
            # Extract current values
            current_close = df.loc[df.index[i], 'close']
            current_adx = df.loc[df.index[i], 'adx']
            current_plus_di = df.loc[df.index[i], '+di']
            current_minus_di = df.loc[df.index[i], '-di']
            
            # Check for exit conditions first
            if position == 1 and df.loc[df.index[i], 'minus_di_gt_plus_di']:
                # Exit long position on -DI crossing above +DI
                trade_pnl = (current_close - entry_price) * position_size
                df.loc[df.index[i], 'trade_pnl'] = trade_pnl
                
                # Apply commission
                commission = current_close * position_size * self.commission_pct
                trade_pnl -= commission
                
                df.loc[df.index[i], 'equity'] += trade_pnl
                df.loc[df.index[i], 'cash'] = df.loc[df.index[i], 'equity']
                df.loc[df.index[i], 'holdings'] = 0
                df.loc[df.index[i], 'position'] = 0
                df.loc[df.index[i], 'position_count'] = 0
                df.loc[df.index[i], 'trade_result'] = 'Exit Long'
                
                position = 0
                position_count = 0
            
            elif position == -1 and df.loc[df.index[i], 'plus_di_gt_minus_di']:
                # Exit short position on +DI crossing above -DI
                trade_pnl = (entry_price - current_close) * position_size
                df.loc[df.index[i], 'trade_pnl'] = trade_pnl
                
                # Apply commission
                commission = current_close * position_size * self.commission_pct
                trade_pnl -= commission
                
                df.loc[df.index[i], 'equity'] += trade_pnl
                df.loc[df.index[i], 'cash'] = df.loc[df.index[i], 'equity']
                df.loc[df.index[i], 'holdings'] = 0
                df.loc[df.index[i], 'position'] = 0
                df.loc[df.index[i], 'position_count'] = 0
                df.loc[df.index[i], 'trade_result'] = 'Exit Short'
                
                position = 0
                position_count = 0
            
            # Check for entry conditions
            elif position == 0 and df.loc[df.index[i], 'is_strong_trend']:
                if df.loc[df.index[i], 'plus_di_gt_minus_di']:
                    # Long entry
                    position = 1
                    position_count = 1
                    entry_price = current_close
                    
                    # Calculate position size
                    cash_to_use = df.loc[df.index[i], 'cash'] * self.position_pct
                    position_size = cash_to_use / current_close
                    
                    # Apply commission
                    commission = current_close * position_size * self.commission_pct
                    df.loc[df.index[i], 'cash'] -= commission
                    
                    df.loc[df.index[i], 'position'] = position
                    df.loc[df.index[i], 'position_count'] = position_count
                    df.loc[df.index[i], 'entry_price'] = entry_price
                    df.loc[df.index[i], 'holdings'] = position_size * current_close
                    df.loc[df.index[i], 'cash'] -= df.loc[df.index[i], 'holdings']
                    
                elif df.loc[df.index[i], 'minus_di_gt_plus_di']:
                    # Short entry
                    position = -1
                    position_count = 1
                    entry_price = current_close
                    
                    # Calculate position size
                    cash_to_use = df.loc[df.index[i], 'cash'] * self.position_pct
                    position_size = cash_to_use / current_close
                    
                    # Apply commission
                    commission = current_close * position_size * self.commission_pct
                    df.loc[df.index[i], 'cash'] -= commission
                    
                    df.loc[df.index[i], 'position'] = position
                    df.loc[df.index[i], 'position_count'] = position_count
                    df.loc[df.index[i], 'entry_price'] = entry_price
                    df.loc[df.index[i], 'holdings'] = -position_size * current_close
                    df.loc[df.index[i], 'cash'] -= df.loc[df.index[i], 'holdings']
            
            # Check for pyramiding opportunities
            elif position == 1 and df.loc[df.index[i], 'is_strong_trend'] and df.loc[df.index[i], 'plus_di_gt_minus_di'] and position_count < self.max_pyramiding:
                # Add to long position
                position_count += 1
                
                # Calculate position size for additional entry
                cash_to_use = df.loc[df.index[i], 'cash'] * self.position_pct
                additional_size = cash_to_use / current_close
                
                # Apply commission
                commission = current_close * additional_size * self.commission_pct
                df.loc[df.index[i], 'cash'] -= commission
                
                # Update entry price (weighted average)
                old_value = entry_price * position_size
                new_value = current_close * additional_size
                entry_price = (old_value + new_value) / (position_size + additional_size)
                position_size += additional_size
                
                df.loc[df.index[i], 'position_count'] = position_count
                df.loc[df.index[i], 'entry_price'] = entry_price
                df.loc[df.index[i], 'holdings'] = position_size * current_close
                df.loc[df.index[i], 'cash'] -= (additional_size * current_close)
            
            elif position == -1 and df.loc[df.index[i], 'is_strong_trend'] and df.loc[df.index[i], 'minus_di_gt_plus_di'] and position_count < self.max_pyramiding:
                # Add to short position
                position_count += 1
                
                # Calculate position size for additional entry
                cash_to_use = df.loc[df.index[i], 'cash'] * self.position_pct
                additional_size = cash_to_use / current_close
                
                # Apply commission
                commission = current_close * additional_size * self.commission_pct
                df.loc[df.index[i], 'cash'] -= commission
                
                # Update entry price (weighted average)
                old_value = entry_price * position_size
                new_value = current_close * additional_size
                entry_price = (old_value + new_value) / (position_size + additional_size)
                position_size += additional_size
                
                df.loc[df.index[i], 'position_count'] = position_count
                df.loc[df.index[i], 'entry_price'] = entry_price
                df.loc[df.index[i], 'holdings'] = -position_size * current_close
                df.loc[df.index[i], 'cash'] -= (-additional_size * current_close)
            
            # Update equity value
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
        trades = results[results['trade_result'].notnull()]
        
        # Calculate metrics
        total_return = (equity.iloc[-1] / equity.iloc[0]) - 1
        
        # Annualized return (assuming 252 trading days per year)
        num_days = len(daily_returns)
        annualized_return = (1 + total_return) ** (252 / num_days) - 1
        
        # Volatility
        annualized_volatility = daily_returns.std() * np.sqrt(252)
        
        # Sharpe ratio (assuming risk-free rate = 0)
        sharpe_ratio = annualized_return / annualized_volatility if annualized_volatility != 0 else 0
        
        # Maximum drawdown
        cumulative_returns = (1 + daily_returns).cumprod()
        peak = cumulative_returns.cummax()
        drawdown = (cumulative_returns / peak) - 1
        max_drawdown = drawdown.min()
        
        # Win rate
        winning_trades = trades[trades['trade_pnl'] > 0]
        win_rate = len(winning_trades) / len(trades) if len(trades) > 0 else 0
        
        # Average profit/loss
        avg_profit = winning_trades['trade_pnl'].mean() if len(winning_trades) > 0 else 0
        losing_trades = trades[trades['trade_pnl'] <= 0]
        avg_loss = losing_trades['trade_pnl'].mean() if len(losing_trades) > 0 else 0
        
        # Profit factor
        gross_profit = winning_trades['trade_pnl'].sum() if len(winning_trades) > 0 else 0
        gross_loss = abs(losing_trades['trade_pnl'].sum()) if len(losing_trades) > 0 else 0
        profit_factor = gross_profit / gross_loss if gross_loss != 0 else float('inf')
        
        # Average holding period
        # Would require additional tracking in the backtest function
        
        # Return metrics
        metrics = {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'annualized_volatility': annualized_volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'avg_profit': avg_profit,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'total_trades': len(trades),
            'num_winning_trades': len(winning_trades),
            'num_losing_trades': len(losing_trades)
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
        
        # Plot 1: Price and positions
        ax1 = plt.subplot(4, 1, 1)
        
        # Plot price
        ax1.plot(results.index, results['close'], label='Close Price', color='black', linewidth=1)
        
        # Highlight areas with positions
        for i in range(1, len(results)):
            if results['position'].iloc[i] == 1:  # Long position
                ax1.axvspan(results.index[i-1], results.index[i], alpha=0.2, color='green')
            elif results['position'].iloc[i] == -1:  # Short position
                ax1.axvspan(results.index[i-1], results.index[i], alpha=0.2, color='red')
        
        # Mark entry and exit points
        entries_long = results[(results['position'].diff() == 1)]
        entries_short = results[(results['position'].diff() == -1)]
        exits_long = results[(results['position'].shift(1) == 1) & (results['position'] == 0)]
        exits_short = results[(results['position'].shift(1) == -1) & (results['position'] == 0)]
        
        ax1.scatter(entries_long.index, entries_long['close'], marker='^', color='green', s=100, label='Long Entry')
        ax1.scatter(entries_short.index, entries_short['close'], marker='v', color='red', s=100, label='Short Entry')
        ax1.scatter(exits_long.index, exits_long['close'], marker='x', color='green', s=100, label='Long Exit')
        ax1.scatter(exits_short.index, exits_short['close'], marker='x', color='red', s=100, label='Short Exit')
        
        ax1.set_title('Price Chart with Positions')
        ax1.set_ylabel('Price')
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: ADX and DI lines
        ax2 = plt.subplot(4, 1, 2)
        
        ax2.plot(results.index, results['adx'], label='ADX', color='black')
        ax2.plot(results.index, results['+di'], label='+DI', color='green')
        ax2.plot(results.index, results['-di'], label='-DI', color='red')
        ax2.axhline(y=self.adx_threshold, color='blue', linestyle='--', alpha=0.5, label='ADX Threshold')
        
        ax2.set_title('ADX and DI Indicators')
        ax2.set_ylabel('Value')
        ax2.legend(loc='upper left')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Position count (pyramiding)
        ax3 = plt.subplot(4, 1, 3)
        
        ax3.plot(results.index, results['position_count'], label='Position Count', color='blue')
        
        ax3.set_title('Position Count (Pyramiding)')
        ax3.set_ylabel('Count')
        ax3.legend(loc='upper left')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Equity curve
        ax4 = plt.subplot(4, 1, 4)
        
        ax4.plot(results.index, results['equity'], label='Strategy Equity', color='blue')
        
        # Calculate and plot buy & hold equity
        initial_capital = results['equity'].iloc[0]
        buy_hold_equity = initial_capital * (results['close'] / results['close'].iloc[0])
        ax4.plot(results.index, buy_hold_equity, label='Buy & Hold Equity', color='gray', alpha=0.5)
        
        ax4.set_title('Equity Curve')
        ax4.set_ylabel('Equity')
        ax4.set_xlabel('Date')
        ax4.legend(loc='upper left')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Additional plot: Drawdown
        plt.figure(figsize=(16, 6))
        
        equity = results['equity']
        peak = equity.cummax()
        drawdown = (equity / peak - 1) * 100
        
        plt.plot(results.index, drawdown, color='red')
        plt.fill_between(results.index, drawdown, 0, color='red', alpha=0.3)
        
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
        print(f"Annualized Volatility: {metrics['annualized_volatility']:.2%}")
        print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        print(f"Maximum Drawdown: {metrics['max_drawdown']:.2%}")
        print(f"Win Rate: {metrics['win_rate']:.2%}")
        print(f"Profit Factor: {metrics['profit_factor']:.2f}")
        print(f"Average Profit: ${metrics['avg_profit']:.2f}")
        print(f"Average Loss: ${metrics['avg_loss']:.2f}")
        print(f"Total Trades: {metrics['total_trades']}")
        print(f"Winning Trades: {metrics['num_winning_trades']}")
        print(f"Losing Trades: {metrics['num_losing_trades']}")
        
        print("=" * 50)

def generate_market_data(days=500, trend_cycles=3, volatility_cycles=2, seed=42):
    """
    Generate synthetic market data with trends, reversals, and realistic properties
    
    Parameters:
    -----------
    days : int
        Number of trading days to generate
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
    
    # Generate dates
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    dates = pd.date_range(start=start_date, end=end_date, periods=days)
    
    # Initialize with a base price
    base_price = 100.0
    
    # Generate price series with trends and volatility cycles
    prices = [base_price]
    
    # Parameters
    trend_period = days // trend_cycles
    vol_period = days // volatility_cycles
    
    # Current trend (1 for up, -1 for down)
    trend = 1
    days_in_trend = 0
    
    for i in range(1, days):
        # Update trend
        days_in_trend += 1
        if days_in_trend >= trend_period:
            # Switch trend direction
            trend = -trend
            days_in_trend = 0
        
        # Calculate trend component
        trend_strength = 0.001 * trend
        
        # Calculate volatility component
        vol_cycle = 0.5 + 0.5 * np.sin(2 * np.pi * i / vol_period)
        volatility = 0.01 * (1 + vol_cycle)
        
        # Calculate daily return
        daily_return = trend_strength + np.random.normal(0, volatility)
        
        # Calculate new price
        new_price = prices[-1] * (1 + daily_return)
        prices.append(new_price)
    
    # Generate OHLC data
    opens = np.zeros(days)
    highs = np.zeros(days)
    lows = np.zeros(days)
    closes = np.array(prices)
    
    # First day
    opens[0] = prices[0] * (1 - 0.005 * np.random.random())
    intraday_vol = prices[0] * 0.02
    highs[0] = max(opens[0], closes[0]) + intraday_vol * np.random.random()
    lows[0] = min(opens[0], closes[0]) - intraday_vol * np.random.random()
    
    # Remaining days
    for i in range(1, days):
        # Open is close of previous day with small gap
        gap = np.random.normal(0, 0.003)
        opens[i] = closes[i-1] * (1 + gap)
        
        # Calculate intraday volatility
        vol_cycle = 0.5 + 0.5 * np.sin(2 * np.pi * i / vol_period)
        intraday_vol = closes[i] * 0.02 * (1 + vol_cycle)
        
        # Generate high and low
        if opens[i] <= closes[i]:  # Up day
            highs[i] = closes[i] + intraday_vol * np.random.random()
            lows[i] = opens[i] - intraday_vol * np.random.random()
        else:  # Down day
            highs[i] = opens[i] + intraday_vol * np.random.random()
            lows[i] = closes[i] - intraday_vol * np.random.random()
        
        # Ensure high >= max(open, close) and low <= min(open, close)
        highs[i] = max(highs[i], opens[i], closes[i])
        lows[i] = min(lows[i], opens[i], closes[i])
    
    # Create DataFrame
    df = pd.DataFrame({
        'open': opens,
        'high': highs,
        'low': lows,
        'close': closes
    }, index=dates)
    
    return df

def test_strategy(data=None):
    """
    Test the ADX Momentum strategy
    
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
        data = generate_market_data(days=500, trend_cycles=5, volatility_cycles=3, seed=42)
    
    # Create strategy instance
    strategy = ADXMomentumStrategy(
        adx_length=14,
        adx_threshold=25,
        position_pct=50.0,
        max_pyramiding=5,
        commission_pct=0.05
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

def analyze_parameter_sensitivity(data=None):
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
        data = generate_market_data(days=500, trend_cycles=5, volatility_cycles=3, seed=42)
    
    # Parameters to test
    adx_lengths = [10, 14, 20]
    adx_thresholds = [20, 25, 30]
    position_pcts = [25.0, 50.0, 75.0]
    max_pyramidings = [1, 3, 5]
    
    # Store results
    results = []
    
    # Test different parameter combinations
    for adx_length in adx_lengths:
        for adx_threshold in adx_thresholds:
            for position_pct in position_pcts:
                for max_pyramiding in max_pyramidings:
                    print(f"Testing parameters: ADX Length={adx_length}, ADX Threshold={adx_threshold}, Position %={position_pct}, Max Pyramiding={max_pyramiding}")
                    
                    # Create strategy with current parameters
                    strategy = ADXMomentumStrategy(
                        adx_length=adx_length,
                        adx_threshold=adx_threshold,
                        position_pct=position_pct,
                        max_pyramiding=max_pyramiding,
                        commission_pct=0.05
                    )
                    
                    # Run backtest
                    backtest_results = strategy.backtest(data)
                    
                    # Calculate performance metrics
                    metrics = strategy.calculate_performance_metrics(backtest_results)
                    
                    # Store results
                    result = {
                        'adx_length': adx_length,
                        'adx_threshold': adx_threshold,
                        'position_pct': position_pct,
                        'max_pyramiding': max_pyramiding,
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
    plt.figure(figsize=(15, 12))
    
    # Plot impact of ADX Length
    plt.subplot(2, 2, 1)
    sns.boxplot(x='adx_length', y='sharpe_ratio', data=results_df)
    plt.title('Impact of ADX Length on Sharpe Ratio')
    plt.grid(True, alpha=0.3)
    
    # Plot impact of ADX Threshold
    plt.subplot(2, 2, 2)
    sns.boxplot(x='adx_threshold', y='sharpe_ratio', data=results_df)
    plt.title('Impact of ADX Threshold on Sharpe Ratio')
    plt.grid(True, alpha=0.3)
    
    # Plot impact of Position %
    plt.subplot(2, 2, 3)
    sns.boxplot(x='position_pct', y='sharpe_ratio', data=results_df)
    plt.title('Impact of Position % on Sharpe Ratio')
    plt.grid(True, alpha=0.3)
    
    # Plot impact of Max Pyramiding
    plt.subplot(2, 2, 4)
    sns.boxplot(x='max_pyramiding', y='sharpe_ratio', data=results_df)
    plt.title('Impact of Max Pyramiding on Sharpe Ratio')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Plot trade-offs
    plt.figure(figsize=(15, 6))
    
    # Plot Return vs. Risk
    plt.subplot(1, 2, 1)
    plt.scatter(results_df['max_drawdown'], results_df['total_return'], 
               c=results_df['sharpe_ratio'], cmap='viridis', s=50, alpha=0.7)
    plt.colorbar(label='Sharpe Ratio')
    plt.title('Return vs. Risk')
    plt.xlabel('Maximum Drawdown')
    plt.ylabel('Total Return')
    plt.grid(True, alpha=0.3)
    
    # Plot Win Rate vs. Profit Factor
    plt.subplot(1, 2, 2)
    plt.scatter(results_df['win_rate'], results_df['profit_factor'], 
               c=results_df['sharpe_ratio'], cmap='viridis', s=50, alpha=0.7)
    plt.colorbar(label='Sharpe Ratio')
    plt.title('Win Rate vs. Profit Factor')
    plt.xlabel('Win Rate')
    plt.ylabel('Profit Factor')
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
        data = generate_market_data(days=500, trend_cycles=5, volatility_cycles=3, seed=42)
    
    # Test original strategy
    print("\nTesting original strategy...")
    original_strategy = ADXMomentumStrategy()
    original_results = original_strategy.backtest(data)
    original_metrics = original_strategy.calculate_performance_metrics(original_results)
    original_strategy.print_performance_summary(original_metrics)
    
    # Enhanced strategy with dynamic ADX threshold
    class EnhancedStrategyWithDynamicADX(ADXMomentumStrategy):
        def __init__(self, adx_lookback=50, **kwargs):
            super().__init__(**kwargs)
            self.adx_lookback = adx_lookback
        
        def calculate_indicators(self, data):
            # Call parent method to calculate base indicators
            df = super().calculate_indicators(data)
            
            # Calculate dynamic ADX threshold based on recent distribution
            df['dynamic_adx_threshold'] = df['adx'].rolling(window=self.adx_lookback).quantile(0.6)
            
            # Ensure threshold is not too low
            df['dynamic_adx_threshold'] = df['dynamic_adx_threshold'].clip(lower=15)
            
            # Replace static threshold with dynamic one
            df['is_strong_trend'] = df['adx'] > df['dynamic_adx_threshold']
            
            return df
    
    # Enhanced strategy with trailing stops
    class EnhancedStrategyWithTrailingStops(ADXMomentumStrategy):
        def __init__(self, trailing_stop_pct=3.0, **kwargs):
            super().__init__(**kwargs)
            self.trailing_stop_pct = trailing_stop_pct / 100.0  # Convert to decimal
        
        def backtest(self, data, initial_capital=10000):
            # Calculate indicators
            df = self.calculate_indicators(data)
            
            # Initialize trading variables
            df['position'] = 0  # 1 for long, -1 for short, 0 for flat
            df['position_count'] = 0  # Number of pyramid positions
            df['entry_price'] = np.nan
            df['equity'] = initial_capital
            df['cash'] = initial_capital
            df['holdings'] = 0
            df['trade_pnl'] = 0
            df['trade_result'] = None
            df['trailing_stop'] = np.nan
            
            # Process each bar
            position = 0  # 0 for flat, 1 for long, -1 for short
            position_count = 0
            entry_price = 0
            position_size = 0
            trailing_stop = 0
            highest_since_entry = 0
            lowest_since_entry = float('inf')
            
            for i in range(self.adx_length * 2, len(df)):
                # Default is to carry forward previous values
                if i > 0:
                    df.loc[df.index[i], 'position'] = df.loc[df.index[i-1], 'position']
                    df.loc[df.index[i], 'position_count'] = df.loc[df.index[i-1], 'position_count']
                    df.loc[df.index[i], 'entry_price'] = df.loc[df.index[i-1], 'entry_price']
                    df.loc[df.index[i], 'equity'] = df.loc[df.index[i-1], 'equity']
                    df.loc[df.index[i], 'cash'] = df.loc[df.index[i-1], 'cash']
                    df.loc[df.index[i], 'holdings'] = df.loc[df.index[i-1], 'holdings']
                    df.loc[df.index[i], 'trailing_stop'] = df.loc[df.index[i-1], 'trailing_stop']
                
                # Extract current values
                current_close = df.loc[df.index[i], 'close']
                current_high = df.loc[df.index[i], 'high']
                current_low = df.loc[df.index[i], 'low']
                
                # Update highest/lowest since entry
                if position == 1:  # Long position
                    highest_since_entry = max(highest_since_entry, current_high)
                    # Update trailing stop
                    new_stop = highest_since_entry * (1 - self.trailing_stop_pct)
                    trailing_stop = max(trailing_stop, new_stop)
                    df.loc[df.index[i], 'trailing_stop'] = trailing_stop
                    
                elif position == -1:  # Short position
                    lowest_since_entry = min(lowest_since_entry, current_low)
                    # Update trailing stop
                    new_stop = lowest_since_entry * (1 + self.trailing_stop_pct)
                    trailing_stop = min(trailing_stop, new_stop) if trailing_stop > 0 else new_stop
                    df.loc[df.index[i], 'trailing_stop'] = trailing_stop
                
                # Check for exit conditions first
                if position == 1:
                    # Exit long if -DI crosses above +DI or trailing stop hit
                    if df.loc[df.index[i], 'minus_di_gt_plus_di'] or current_low <= trailing_stop:
                        # Determine exit price
                        exit_price = current_close
                        if current_low <= trailing_stop:
                            exit_price = trailing_stop  # Use trailing stop as exit price
                            exit_reason = 'Trailing Stop'
                        else:
                            exit_reason = 'DI Crossover'
                        
                        # Calculate P&L
                        trade_pnl = (exit_price - entry_price) * position_size
                        df.loc[df.index[i], 'trade_pnl'] = trade_pnl
                        
                        # Apply commission
                        commission = exit_price * position_size * self.commission_pct
                        trade_pnl -= commission
                        
                        df.loc[df.index[i], 'equity'] += trade_pnl
                        df.loc[df.index[i], 'cash'] = df.loc[df.index[i], 'equity']
                        df.loc[df.index[i], 'holdings'] = 0
                        df.loc[df.index[i], 'position'] = 0
                        df.loc[df.index[i], 'position_count'] = 0
                        df.loc[df.index[i], 'trade_result'] = exit_reason
                        df.loc[df.index[i], 'trailing_stop'] = np.nan
                        
                        position = 0
                        position_count = 0
                        highest_since_entry = 0
                
                elif position == -1:
                    # Exit short if +DI crosses above -DI or trailing stop hit
                    if df.loc[df.index[i], 'plus_di_gt_minus_di'] or current_high >= trailing_stop:
                        # Determine exit price
                        exit_price = current_close
                        if current_high >= trailing_stop:
                            exit_price = trailing_stop  # Use trailing stop as exit price
                            exit_reason = 'Trailing Stop'
                        else:
                            exit_reason = 'DI Crossover'
                        
                        # Calculate P&L
                        trade_pnl = (entry_price - exit_price) * position_size
                        df.loc[df.index[i], 'trade_pnl'] = trade_pnl
                        
                        # Apply commission
                        commission = exit_price * position_size * self.commission_pct
                        trade_pnl -= commission
                        
                        df.loc[df.index[i], 'equity'] += trade_pnl
                        df.loc[df.index[i], 'cash'] = df.loc[df.index[i], 'equity']
                        df.loc[df.index[i], 'holdings'] = 0
                        df.loc[df.index[i], 'position'] = 0
                        df.loc[df.index[i], 'position_count'] = 0
                        df.loc[df.index[i], 'trade_result'] = exit_reason
                        df.loc[df.index[i], 'trailing_stop'] = np.nan
                        
                        position = 0
                        position_count = 0
                        lowest_since_entry = float('inf')
                
                # Check for entry conditions
                elif position == 0 and df.loc[df.index[i], 'is_strong_trend']:
                    if df.loc[df.index[i], 'plus_di_gt_minus_di']:
                        # Long entry
                        position = 1
                        position_count = 1
                        entry_price = current_close
                        
                        # Calculate position size
                        cash_to_use = df.loc[df.index[i], 'cash'] * self.position_pct
                        position_size = cash_to_use / current_close
                        
                        # Apply commission
                        commission = current_close * position_size * self.commission_pct
                        df.loc[df.index[i], 'cash'] -= commission
                        
                        # Set initial trailing stop
                        trailing_stop = entry_price * (1 - self.trailing_stop_pct)
                        highest_since_entry = current_high
                        
                        df.loc[df.index[i], 'position'] = position
                        df.loc[df.index[i], 'position_count'] = position_count
                        df.loc[df.index[i], 'entry_price'] = entry_price
                        df.loc[df.index[i], 'holdings'] = position_size * current_close
                        df.loc[df.index[i], 'cash'] -= df.loc[df.index[i], 'holdings']
                        df.loc[df.index[i], 'trailing_stop'] = trailing_stop
                        
                    elif df.loc[df.index[i], 'minus_di_gt_plus_di']:
                        # Short entry
                        position = -1
                        position_count = 1
                        entry_price = current_close
                        
                        # Calculate position size
                        cash_to_use = df.loc[df.index[i], 'cash'] * self.position_pct
                        position_size = cash_to_use / current_close
                        
                        # Apply commission
                        commission = current_close * position_size * self.commission_pct
                        df.loc[df.index[i], 'cash'] -= commission
                        
                        # Set initial trailing stop
                        trailing_stop = entry_price * (1 + self.trailing_stop_pct)
                        lowest_since_entry = current_low
                        
                        df.loc[df.index[i], 'position'] = position
                        df.loc[df.index[i], 'position_count'] = position_count
                        df.loc[df.index[i], 'entry_price'] = entry_price
                        df.loc[df.index[i], 'holdings'] = -position_size * current_close
                        df.loc[df.index[i], 'cash'] -= df.loc[df.index[i], 'holdings']
                        df.loc[df.index[i], 'trailing_stop'] = trailing_stop
                
                # Update equity value
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
    
    # Enhanced strategy with dynamic position sizing
    class EnhancedStrategyWithDynamicPositionSizing(ADXMomentumStrategy):
        def __init__(self, adx_scaling=True, **kwargs):
            super().__init__(**kwargs)
            self.adx_scaling = adx_scaling
        
        def backtest(self, data, initial_capital=10000):
            # Calculate indicators
            df = self.calculate_indicators(data)
            
            # Initialize trading variables
            df['position'] = 0  # 1 for long, -1 for short, 0 for flat
            df['position_count'] = 0  # Number of pyramid positions
            df['entry_price'] = np.nan
            df['equity'] = initial_capital
            df['cash'] = initial_capital
            df['holdings'] = 0
            df['trade_pnl'] = 0
            df['trade_result'] = None
            df['position_size_pct'] = np.nan  # Dynamic position size percentage
            
            # Process each bar
            position = 0  # 0 for flat, 1 for long, -1 for short
            position_count = 0
            entry_price = 0
            position_size = 0
            
            for i in range(self.adx_length * 2, len(df)):
                # Default is to carry forward previous values
                if i > 0:
                    df.loc[df.index[i], 'position'] = df.loc[df.index[i-1], 'position']
                    df.loc[df.index[i], 'position_count'] = df.loc[df.index[i-1], 'position_count']
                    df.loc[df.index[i], 'entry_price'] = df.loc[df.index[i-1], 'entry_price']
                    df.loc[df.index[i], 'equity'] = df.loc[df.index[i-1], 'equity']
                    df.loc[df.index[i], 'cash'] = df.loc[df.index[i-1], 'cash']
                    df.loc[df.index[i], 'holdings'] = df.loc[df.index[i-1], 'holdings']
                    df.loc[df.index[i], 'position_size_pct'] = df.loc[df.index[i-1], 'position_size_pct']
                
                # Extract current values
                current_close = df.loc[df.index[i], 'close']
                current_adx = df.loc[df.index[i], 'adx']
                current_plus_di = df.loc[df.index[i], '+di']
                current_minus_di = df.loc[df.index[i], '-di']
                
                # Check for exit conditions first
                if position == 1 and df.loc[df.index[i], 'minus_di_gt_plus_di']:
                    # Exit long position on -DI crossing above +DI
                    trade_pnl = (current_close - entry_price) * position_size
                    df.loc[df.index[i], 'trade_pnl'] = trade_pnl
                    
                    # Apply commission
                    commission = current_close * position_size * self.commission_pct
                    trade_pnl -= commission
                    
                    df.loc[df.index[i], 'equity'] += trade_pnl
                    df.loc[df.index[i], 'cash'] = df.loc[df.index[i], 'equity']
                    df.loc[df.index[i], 'holdings'] = 0
                    df.loc[df.index[i], 'position'] = 0
                    df.loc[df.index[i], 'position_count'] = 0
                    df.loc[df.index[i], 'trade_result'] = 'Exit Long'
                    df.loc[df.index[i], 'position_size_pct'] = np.nan
                    
                    position = 0
                    position_count = 0
                
                elif position == -1 and df.loc[df.index[i], 'plus_di_gt_minus_di']:
                    # Exit short position on +DI crossing above -DI
                    trade_pnl = (entry_price - current_close) * position_size
                    df.loc[df.index[i], 'trade_pnl'] = trade_pnl
                    
                    # Apply commission
                    commission = current_close * position_size * self.commission_pct
                    trade_pnl -= commission
                    
                    df.loc[df.index[i], 'equity'] += trade_pnl
                    df.loc[df.index[i], 'cash'] = df.loc[df.index[i], 'equity']
                    df.loc[df.index[i], 'holdings'] = 0
                    df.loc[df.index[i], 'position'] = 0
                    df.loc[df.index[i], 'position_count'] = 0
                    df.loc[df.index[i], 'trade_result'] = 'Exit Short'
                    df.loc[df.index[i], 'position_size_pct'] = np.nan
                    
                    position = 0
                    position_count = 0
                
                # Check for entry conditions
                elif position == 0 and df.loc[df.index[i], 'is_strong_trend']:
                    if df.loc[df.index[i], 'plus_di_gt_minus_di']:
                        # Long entry
                        position = 1
                        position_count = 1
                        entry_price = current_close
                        
                        # Calculate dynamic position size based on ADX strength
                        if self.adx_scaling:
                            # Scale position size from 25% to 75% based on ADX strength
                            adx_strength = min(1.0, (current_adx - self.adx_threshold) / 25)  # Normalize to 0-1
                            dynamic_position_pct = 0.25 + (0.5 * adx_strength)  # Scale from 25% to 75%
                        else:
                            dynamic_position_pct = self.position_pct
                        
                        df.loc[df.index[i], 'position_size_pct'] = dynamic_position_pct
                        
                        # Calculate position size
                        cash_to_use = df.loc[df.index[i], 'cash'] * dynamic_position_pct
                        position_size = cash_to_use / current_close
                        
                        # Apply commission
                        commission = current_close * position_size * self.commission_pct
                        df.loc[df.index[i], 'cash'] -= commission
                        
                        df.loc[df.index[i], 'position'] = position
                        df.loc[df.index[i], 'position_count'] = position_count
                        df.loc[df.index[i], 'entry_price'] = entry_price
                        df.loc[df.index[i], 'holdings'] = position_size * current_close
                        df.loc[df.index[i], 'cash'] -= df.loc[df.index[i], 'holdings']
                        
                    elif df.loc[df.index[i], 'minus_di_gt_plus_di']:
                        # Short entry
                        position = -1
                        position_count = 1
                        entry_price = current_close
                        
                        # Calculate dynamic position size based on ADX strength
                        if self.adx_scaling:
                            # Scale position size from 25% to 75% based on ADX strength
                            adx_strength = min(1.0, (current_adx - self.adx_threshold) / 25)  # Normalize to 0-1
                            dynamic_position_pct = 0.25 + (0.5 * adx_strength)  # Scale from 25% to 75%
                        else:
                            dynamic_position_pct = self.position_pct
                        
                        df.loc[df.index[i], 'position_size_pct'] = dynamic_position_pct
                        
                        # Calculate position size
                        cash_to_use = df.loc[df.index[i], 'cash'] * dynamic_position_pct
                        position_size = cash_to_use / current_close
                        
                        # Apply commission
                        commission = current_close * position_size * self.commission_pct
                        df.loc[df.index[i], 'cash'] -= commission
                        
                        df.loc[df.index[i], 'position'] = position
                        df.loc[df.index[i], 'position_count'] = position_count
                        df.loc[df.index[i], 'entry_price'] = entry_price
                        df.loc[df.index[i], 'holdings'] = -position_size * current_close
                        df.loc[df.index[i], 'cash'] -= df.loc[df.index[i], 'holdings']
                
                # Update equity value
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
    
    # Test enhanced strategies
    print("\nTesting enhanced strategy with dynamic ADX threshold...")
    dynamic_adx_strategy = EnhancedStrategyWithDynamicADX(adx_lookback=50)
    dynamic_adx_results = dynamic_adx_strategy.backtest(data)
    dynamic_adx_metrics = dynamic_adx_strategy.calculate_performance_metrics(dynamic_adx_results)
    dynamic_adx_strategy.print_performance_summary(dynamic_adx_metrics)
    
    print("\nTesting enhanced strategy with trailing stops...")
    trailing_stop_strategy = EnhancedStrategyWithTrailingStops(trailing_stop_pct=3.0)
    trailing_stop_results = trailing_stop_strategy.backtest(data)
    trailing_stop_metrics = trailing_stop_strategy.calculate_performance_metrics(trailing_stop_results)
    trailing_stop_strategy.print_performance_summary(trailing_stop_metrics)
    
    print("\nTesting enhanced strategy with dynamic position sizing...")
    dynamic_position_strategy = EnhancedStrategyWithDynamicPositionSizing(adx_scaling=True)
    dynamic_position_results = dynamic_position_strategy.backtest(data)
    dynamic_position_metrics = dynamic_position_strategy.calculate_performance_metrics(dynamic_position_results)
    dynamic_position_strategy.print_performance_summary(dynamic_position_metrics)
    
    # Compare equity curves
    plt.figure(figsize=(15, 8))
    
    plt.plot(original_results.index, original_results['equity'], label='Original Strategy')
    plt.plot(dynamic_adx_results.index, dynamic_adx_results['equity'], label='Dynamic ADX Threshold')
    plt.plot(trailing_stop_results.index, trailing_stop_results['equity'], label='Trailing Stops')
    plt.plot(dynamic_position_results.index, dynamic_position_results['equity'], label='Dynamic Position Sizing')
    
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
        'Dynamic ADX': [
            dynamic_adx_metrics['total_return'],
            dynamic_adx_metrics['sharpe_ratio'],
            dynamic_adx_metrics['max_drawdown'],
            dynamic_adx_metrics['win_rate'],
            dynamic_adx_metrics['profit_factor'],
            dynamic_adx_metrics['total_trades']
        ],
        'Trailing Stops': [
            trailing_stop_metrics['total_return'],
            trailing_stop_metrics['sharpe_ratio'],
            trailing_stop_metrics['max_drawdown'],
            trailing_stop_metrics['win_rate'],
            trailing_stop_metrics['profit_factor'],
            trailing_stop_metrics['total_trades']
        ],
        'Dynamic Position': [
            dynamic_position_metrics['total_return'],
            dynamic_position_metrics['sharpe_ratio'],
            dynamic_position_metrics['max_drawdown'],
            dynamic_position_metrics['win_rate'],
            dynamic_position_metrics['profit_factor'],
            dynamic_position_metrics['total_trades']
        ]
    }, index=['Total Return', 'Sharpe Ratio', 'Max Drawdown', 'Win Rate', 'Profit Factor', 'Total Trades'])
    
    print("\nStrategy Comparison:")
    print(comparison)
    
    # Plot comparison metrics
    plt.figure(figsize=(15, 12))
    
    metrics_to_plot = ['Total Return', 'Sharpe Ratio', 'Max Drawdown', 'Win Rate', 'Profit Factor', 'Total Trades']
    for i, metric in enumerate(metrics_to_plot):
        plt.subplot(3, 2, i+1)
        comparison.loc[metric].plot(kind='bar')
        plt.title(metric)
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return {
        'original': {'strategy': original_strategy, 'results': original_results, 'metrics': original_metrics},
        'dynamic_adx': {'strategy': dynamic_adx_strategy, 'results': dynamic_adx_results, 'metrics': dynamic_adx_metrics},
        'trailing_stop': {'strategy': trailing_stop_strategy, 'results': trailing_stop_results, 'metrics': trailing_stop_metrics},
        'dynamic_position': {'strategy': dynamic_position_strategy, 'results': dynamic_position_results, 'metrics': dynamic_position_metrics}
    }

# Run the tests
if __name__ == "__main__":
    # Test the strategy with default parameters
    strategy, results, metrics = test_strategy()
    
    # Analyze parameter sensitivity
    # param_results = analyze_parameter_sensitivity()
    
    # Test strategy enhancements
    # enhancement_results = test_strategy_enhancements()