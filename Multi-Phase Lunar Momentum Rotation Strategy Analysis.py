import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import seaborn as sns
from scipy.stats import ttest_ind, shapiro, levene, mannwhitneyu
import warnings

warnings.filterwarnings('ignore')

class LunarMomentumStrategy:
    """
    Multi-Phase Lunar Momentum Rotation Strategy implementation
    """
    
    def __init__(self, threshold_hours=12):
        """
        Initialize the strategy
        
        Parameters:
        -----------
        threshold_hours : int
            Time threshold in hours to consider "near" a lunar phase
        """
        self.threshold_hours = threshold_hours
        
        # Define full moon times for 2025 (format: year, month, day, hour, minute)
        self.full_moon_times = [
            datetime(2025, 1, 13, 10, 27),
            datetime(2025, 2, 12, 2, 53),
            datetime(2025, 3, 13, 17, 54),
            datetime(2025, 4, 12, 6, 21),
            datetime(2025, 5, 11, 17, 55),
            datetime(2025, 6, 10, 3, 20),
            datetime(2025, 7, 9, 11, 55),
            datetime(2025, 8, 7, 19, 39),
            datetime(2025, 9, 6, 3, 21),
            datetime(2025, 10, 5, 11, 47),
            datetime(2025, 11, 3, 21, 18),
            datetime(2025, 12, 3, 8, 15)
        ]
        
        # Define new moon times for 2025
        self.new_moon_times = [
            datetime(2025, 1, 29, 0, 37),
            datetime(2025, 2, 27, 12, 27),
            datetime(2025, 3, 29, 0, 58),
            datetime(2025, 4, 27, 13, 39),
            datetime(2025, 5, 27, 2, 22),
            datetime(2025, 6, 25, 17, 8),
            datetime(2025, 7, 25, 8, 12),
            datetime(2025, 8, 23, 23, 7),
            datetime(2025, 9, 22, 13, 50),
            datetime(2025, 10, 22, 3, 26),
            datetime(2025, 11, 20, 15, 48),
            datetime(2025, 12, 20, 3, 44)
        ]
        
    def add_historical_moon_phases(self, years_back=5):
        """
        Add historical moon phases for backtesting
        This is a simplified approach - in reality, you would use astronomical calculations
        or an ephemeris database for accurate historical lunar phases
        """
        historical_full_moons = []
        historical_new_moons = []
        
        # Generate approximate historical moon phases
        # In reality, lunar cycles are approximately 29.53 days
        lunar_cycle = 29.53  # days
        half_cycle = lunar_cycle / 2
        
        # Start with the first full moon of 2025 and work backwards
        first_full_moon_2025 = self.full_moon_times[0]
        current_full_moon = first_full_moon_2025
        current_new_moon = self.new_moon_times[0]
        
        for _ in range(years_back * 12):  # Approximately 12 lunar cycles per year
            # Previous full moon
            current_full_moon = current_full_moon - timedelta(days=lunar_cycle)
            historical_full_moons.append(current_full_moon)
            
            # Previous new moon
            current_new_moon = current_new_moon - timedelta(days=lunar_cycle)
            historical_new_moons.append(current_new_moon)
        
        # Reverse the lists so they're in chronological order
        historical_full_moons.reverse()
        historical_new_moons.reverse()
        
        # Add the 2025 moon phases
        historical_full_moons.extend(self.full_moon_times)
        historical_new_moons.extend(self.new_moon_times)
        
        self.full_moon_times = historical_full_moons
        self.new_moon_times = historical_new_moons
        
    def is_near_moon(self, date, moon_times):
        """
        Check if a given date is near a moon phase
        
        Parameters:
        -----------
        date : datetime
            The date to check
        moon_times : list
            List of moon phase times to check against
            
        Returns:
        --------
        bool
            True if the date is within threshold of any moon phase time
        """
        threshold = timedelta(hours=self.threshold_hours)
        
        for moon_time in moon_times:
            time_diff = abs(date - moon_time)
            if time_diff <= threshold:
                return True
        
        return False
    
    def generate_signals(self, data):
        """
        Generate trading signals based on lunar phases
        
        Parameters:
        -----------
        data : DataFrame
            Price data with datetime index
            
        Returns:
        --------
        DataFrame
            Data with added signal columns
        """
        # Make a copy of the data
        df = data.copy()
        
        # Initialize signal columns
        df['is_full_moon'] = False
        df['is_new_moon'] = False
        df['signal'] = 0  # 0: no action, 1: buy on full, 2: buy on new, -1: sell
        df['position'] = 0
        df['in_position'] = False
        df['entry_price'] = np.nan
        df['entry_type'] = None
        
        # Check each date against moon phases
        for i, date in enumerate(df.index):
            # Check if date is near a full or new moon
            is_full = self.is_near_moon(date, self.full_moon_times)
            is_new = self.is_near_moon(date, self.new_moon_times)
            
            df.loc[date, 'is_full_moon'] = is_full
            df.loc[date, 'is_new_moon'] = is_new
            
            # Generate signals
            current_position = df.loc[date, 'position']
            
            if is_full:
                if current_position == 0 or current_position == 2:  # No position or in "new moon" position
                    df.loc[date, 'signal'] = 1  # Buy on full moon
                    if i > 0:
                        prev_date = df.index[i-1]
                        if df.loc[prev_date, 'position'] == 2:  # Was in "new moon" position
                            df.loc[date, 'signal'] = -1  # Close "new moon" position
            
            elif is_new:
                if current_position == 0 or current_position == 1:  # No position or in "full moon" position
                    df.loc[date, 'signal'] = 2  # Buy on new moon
                    if i > 0:
                        prev_date = df.index[i-1]
                        if df.loc[prev_date, 'position'] == 1:  # Was in "full moon" position
                            df.loc[date, 'signal'] = -2  # Close "full moon" position
        
        # Process positions and entries
        position = 0
        for i, date in enumerate(df.index):
            signal = df.loc[date, 'signal']
            
            # Close positions first
            if signal == -1 or signal == -2:
                position = 0
                df.loc[date, 'in_position'] = False
                df.loc[date, 'entry_price'] = np.nan
                df.loc[date, 'entry_type'] = None
            
            # Then open new positions
            if signal == 1:  # Buy on full moon
                position = 1
                df.loc[date, 'in_position'] = True
                df.loc[date, 'entry_price'] = df.loc[date, 'close']
                df.loc[date, 'entry_type'] = 'Full Moon'
            elif signal == 2:  # Buy on new moon
                position = 2
                df.loc[date, 'in_position'] = True
                df.loc[date, 'entry_price'] = df.loc[date, 'close']
                df.loc[date, 'entry_type'] = 'New Moon'
            
            df.loc[date, 'position'] = position
        
        return df
    
    def calculate_returns(self, data, initial_capital=100000, position_size_pct=100):
        """
        Calculate strategy returns
        
        Parameters:
        -----------
        data : DataFrame
            Price data with signals
        initial_capital : float
            Initial capital
        position_size_pct : float
            Position size as percentage of equity
            
        Returns:
        --------
        DataFrame
            Data with added return columns
        """
        # Make a copy of the data
        df = data.copy()
        
        # Calculate position size
        position_size = position_size_pct / 100
        
        # Initialize columns
        df['capital'] = initial_capital
        df['holdings'] = 0
        df['cash'] = initial_capital
        df['equity'] = initial_capital
        df['returns'] = 0
        df['strategy_returns'] = 0
        df['buy_hold_returns'] = 0
        
        # Buy and hold returns
        df['buy_hold_returns'] = df['close'].pct_change()
        
        # Loop through data to calculate strategy returns
        for i in range(1, len(df)):
            prev_date = df.index[i-1]
            curr_date = df.index[i]
            
            # Default is to carry forward previous values
            df.loc[curr_date, 'capital'] = df.loc[prev_date, 'capital']
            df.loc[curr_date, 'holdings'] = df.loc[prev_date, 'holdings']
            df.loc[curr_date, 'cash'] = df.loc[prev_date, 'cash']
            
            # Calculate daily returns
            daily_return = df.loc[curr_date, 'close'] / df.loc[prev_date, 'close'] - 1
            df.loc[curr_date, 'returns'] = daily_return
            
            # Process signals
            signal = df.loc[curr_date, 'signal']
            
            # Close positions
            if signal == -1 or signal == -2:
                # Calculate return on closed position
                entry_price = df.loc[prev_date, 'entry_price']
                exit_price = df.loc[curr_date, 'close']
                position_return = (exit_price / entry_price - 1) * position_size
                
                # Update capital
                new_capital = df.loc[prev_date, 'capital'] * (1 + position_return)
                df.loc[curr_date, 'capital'] = new_capital
                df.loc[curr_date, 'cash'] = new_capital
                df.loc[curr_date, 'holdings'] = 0
            
            # Open new positions
            if signal == 1 or signal == 2:
                # Allocate capital to position
                allocated_capital = df.loc[curr_date, 'capital'] * position_size
                df.loc[curr_date, 'holdings'] = allocated_capital
                df.loc[curr_date, 'cash'] = df.loc[curr_date, 'capital'] - allocated_capital
            
            # Update holdings value if in position
            if df.loc[curr_date, 'in_position']:
                if df.loc[prev_date, 'in_position']:  # Was already in position
                    df.loc[curr_date, 'holdings'] = df.loc[prev_date, 'holdings'] * (1 + daily_return)
            
            # Calculate equity
            df.loc[curr_date, 'equity'] = df.loc[curr_date, 'cash'] + df.loc[curr_date, 'holdings']
            
            # Calculate strategy returns
            df.loc[curr_date, 'strategy_returns'] = df.loc[curr_date, 'equity'] / df.loc[prev_date, 'equity'] - 1
        
        # Calculate cumulative returns
        df['cum_strategy_returns'] = (1 + df['strategy_returns']).cumprod() - 1
        df['cum_buy_hold_returns'] = (1 + df['buy_hold_returns']).cumprod() - 1
        
        return df
    
    def calculate_performance_metrics(self, results):
        """
        Calculate performance metrics for the strategy
        
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
        strategy_returns = results['strategy_returns'].dropna()
        buy_hold_returns = results['buy_hold_returns'].dropna()
        
        # Calculate metrics
        total_days = len(strategy_returns)
        trading_days_per_year = 252
        
        # Total returns
        total_strategy_return = results['cum_strategy_returns'].iloc[-1]
        total_buy_hold_return = results['cum_buy_hold_returns'].iloc[-1]
        
        # Annualized returns
        years = total_days / trading_days_per_year
        annual_strategy_return = (1 + total_strategy_return) ** (1 / years) - 1
        annual_buy_hold_return = (1 + total_buy_hold_return) ** (1 / years) - 1
        
        # Volatility
        daily_strategy_vol = strategy_returns.std()
        annual_strategy_vol = daily_strategy_vol * np.sqrt(trading_days_per_year)
        
        daily_buy_hold_vol = buy_hold_returns.std()
        annual_buy_hold_vol = daily_buy_hold_vol * np.sqrt(trading_days_per_year)
        
        # Sharpe ratio (assuming risk-free rate = 0)
        strategy_sharpe = annual_strategy_return / annual_strategy_vol if annual_strategy_vol != 0 else 0
        buy_hold_sharpe = annual_buy_hold_return / annual_buy_hold_vol if annual_buy_hold_vol != 0 else 0
        
        # Maximum drawdown
        strategy_cum_returns = (1 + strategy_returns).cumprod()
        strategy_running_max = strategy_cum_returns.cummax()
        strategy_drawdown = (strategy_cum_returns / strategy_running_max - 1)
        strategy_max_drawdown = strategy_drawdown.min()
        
        buy_hold_cum_returns = (1 + buy_hold_returns).cumprod()
        buy_hold_running_max = buy_hold_cum_returns.cummax()
        buy_hold_drawdown = (buy_hold_cum_returns / buy_hold_running_max - 1)
        buy_hold_max_drawdown = buy_hold_drawdown.min()
        
        # Win rate
        strategy_win_days = sum(strategy_returns > 0)
        strategy_win_rate = strategy_win_days / total_days
        
        buy_hold_win_days = sum(buy_hold_returns > 0)
        buy_hold_win_rate = buy_hold_win_days / total_days
        
        # Analyze performance during full and new moons
        full_moon_returns = results.loc[results['is_full_moon'], 'returns']
        new_moon_returns = results.loc[results['is_new_moon'], 'returns']
        other_days_returns = results.loc[~(results['is_full_moon'] | results['is_new_moon']), 'returns']
        
        full_moon_avg_return = full_moon_returns.mean() if len(full_moon_returns) > 0 else 0
        new_moon_avg_return = new_moon_returns.mean() if len(new_moon_returns) > 0 else 0
        other_days_avg_return = other_days_returns.mean() if len(other_days_returns) > 0 else 0
        
        # Count trades
        full_moon_trades = len(results[(results['signal'] == 1) | (results['signal'] == -1)])
        new_moon_trades = len(results[(results['signal'] == 2) | (results['signal'] == -2)])
        total_trades = full_moon_trades + new_moon_trades
        
        # Metrics dictionary
        metrics = {
            'total_strategy_return': total_strategy_return,
            'total_buy_hold_return': total_buy_hold_return,
            'annual_strategy_return': annual_strategy_return,
            'annual_buy_hold_return': annual_buy_hold_return,
            'annual_strategy_vol': annual_strategy_vol,
            'annual_buy_hold_vol': annual_buy_hold_vol,
            'strategy_sharpe': strategy_sharpe,
            'buy_hold_sharpe': buy_hold_sharpe,
            'strategy_max_drawdown': strategy_max_drawdown,
            'buy_hold_max_drawdown': buy_hold_max_drawdown,
            'strategy_win_rate': strategy_win_rate,
            'buy_hold_win_rate': buy_hold_win_rate,
            'full_moon_avg_return': full_moon_avg_return,
            'new_moon_avg_return': new_moon_avg_return,
            'other_days_avg_return': other_days_avg_return,
            'total_trades': total_trades,
            'full_moon_trades': full_moon_trades,
            'new_moon_trades': new_moon_trades
        }
        
        return metrics
    
    def statistical_tests(self, results):
        """
        Perform statistical tests on returns during different lunar phases
        
        Parameters:
        -----------
        results : DataFrame
            Backtest results
            
        Returns:
        --------
        dict
            Dictionary of statistical test results
        """
        # Extract returns during different lunar phases
        returns = results['returns'].dropna()
        full_moon_returns = returns[results['is_full_moon']]
        new_moon_returns = returns[results['is_new_moon']]
        other_days_returns = returns[~(results['is_full_moon'] | results['is_new_moon'])]
        
        # Dictionary to store test results
        tests = {}
        
        # Check for normality
        try:
            tests['shapiro_full_moon'] = shapiro(full_moon_returns)
            tests['shapiro_new_moon'] = shapiro(new_moon_returns)
            tests['shapiro_other_days'] = shapiro(other_days_returns)
        except:
            tests['shapiro_full_moon'] = (None, None)
            tests['shapiro_new_moon'] = (None, None)
            tests['shapiro_other_days'] = (None, None)
        
        # Check for homogeneity of variance
        try:
            tests['levene_full_vs_other'] = levene(full_moon_returns, other_days_returns)
            tests['levene_new_vs_other'] = levene(new_moon_returns, other_days_returns)
            tests['levene_full_vs_new'] = levene(full_moon_returns, new_moon_returns)
        except:
            tests['levene_full_vs_other'] = (None, None)
            tests['levene_new_vs_other'] = (None, None)
            tests['levene_full_vs_new'] = (None, None)
        
        # T-test for means (parametric)
        try:
            tests['ttest_full_vs_other'] = ttest_ind(full_moon_returns, other_days_returns, equal_var=False)
            tests['ttest_new_vs_other'] = ttest_ind(new_moon_returns, other_days_returns, equal_var=False)
            tests['ttest_full_vs_new'] = ttest_ind(full_moon_returns, new_moon_returns, equal_var=False)
        except:
            tests['ttest_full_vs_other'] = (None, None)
            tests['ttest_new_vs_other'] = (None, None)
            tests['ttest_full_vs_new'] = (None, None)
        
        # Mann-Whitney U test (non-parametric)
        try:
            tests['mannwhitney_full_vs_other'] = mannwhitneyu(full_moon_returns, other_days_returns)
            tests['mannwhitney_new_vs_other'] = mannwhitneyu(new_moon_returns, other_days_returns)
            tests['mannwhitney_full_vs_new'] = mannwhitneyu(full_moon_returns, new_moon_returns)
        except:
            tests['mannwhitney_full_vs_other'] = (None, None)
            tests['mannwhitney_new_vs_other'] = (None, None)
            tests['mannwhitney_full_vs_new'] = (None, None)
        
        return tests
    
    def plot_results(self, results):
        """
        Plot backtest results
        
        Parameters:
        -----------
        results : DataFrame
            Backtest results
        """
        # Set up the figure
        plt.figure(figsize=(15, 12))
        
        # Plot 1: Price with lunar phase markers
        ax1 = plt.subplot(3, 1, 1)
        
        # Plot price
        ax1.plot(results.index, results['close'], label='Price', color='blue', alpha=0.7)
        
        # Mark full and new moons
        full_moon_dates = results[results['is_full_moon']].index
        new_moon_dates = results[results['is_new_moon']].index
        
        for date in full_moon_dates:
            ax1.axvline(x=date, color='green', linestyle='--', alpha=0.3)
        
        for date in new_moon_dates:
            ax1.axvline(x=date, color='orange', linestyle='--', alpha=0.3)
        
        # Mark buy and sell signals
        buy_full = results[results['signal'] == 1].index
        buy_new = results[results['signal'] == 2].index
        sell_full = results[results['signal'] == -1].index
        sell_new = results[results['signal'] == -2].index
        
        ax1.scatter(buy_full, results.loc[buy_full, 'close'], color='green', marker='^', s=100, label='Buy on Full Moon')
        ax1.scatter(buy_new, results.loc[buy_new, 'close'], color='orange', marker='^', s=100, label='Buy on New Moon')
        ax1.scatter(sell_full, results.loc[sell_full, 'close'], color='red', marker='v', s=100, label='Sell Full Moon Position')
        ax1.scatter(sell_new, results.loc[sell_new, 'close'], color='purple', marker='v', s=100, label='Sell New Moon Position')
        
        ax1.set_title('Price with Lunar Phases and Signals')
        ax1.set_ylabel('Price')
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Strategy vs Buy-and-Hold Returns
        ax2 = plt.subplot(3, 1, 2)
        
        ax2.plot(results.index, (1 + results['cum_strategy_returns']) * 100, label='Strategy', color='green')
        ax2.plot(results.index, (1 + results['cum_buy_hold_returns']) * 100, label='Buy & Hold', color='blue', alpha=0.6)
        
        ax2.set_title('Cumulative Returns: Strategy vs Buy-and-Hold')
        ax2.set_ylabel('Return (%)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Drawdowns
        ax3 = plt.subplot(3, 1, 3)
        
        strategy_returns = results['strategy_returns'].dropna()
        strategy_cum_returns = (1 + strategy_returns).cumprod()
        strategy_running_max = strategy_cum_returns.cummax()
        strategy_drawdown = (strategy_cum_returns / strategy_running_max - 1) * 100
        
        buy_hold_returns = results['buy_hold_returns'].dropna()
        buy_hold_cum_returns = (1 + buy_hold_returns).cumprod()
        buy_hold_running_max = buy_hold_cum_returns.cummax()
        buy_hold_drawdown = (buy_hold_cum_returns / buy_hold_running_max - 1) * 100
        
        ax3.plot(strategy_drawdown.index, strategy_drawdown, label='Strategy Drawdown', color='green')
        ax3.plot(buy_hold_drawdown.index, buy_hold_drawdown, label='Buy & Hold Drawdown', color='blue', alpha=0.6)
        
        ax3.set_title('Drawdowns: Strategy vs Buy-and-Hold')
        ax3.set_ylabel('Drawdown (%)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Additional plot: Returns during different lunar phases
        plt.figure(figsize=(12, 8))
        
        # Extract returns during different lunar phases
        returns = results['returns'].dropna()
        full_moon_returns = returns[results['is_full_moon']]
        new_moon_returns = returns[results['is_new_moon']]
        other_days_returns = returns[~(results['is_full_moon'] | results['is_new_moon'])]
        
        # Box plot of returns
        data_to_plot = [
            full_moon_returns * 100,
            new_moon_returns * 100,
            other_days_returns * 100
        ]
        
        plt.boxplot(data_to_plot, labels=['Full Moon', 'New Moon', 'Other Days'])
        plt.title('Return Distribution by Lunar Phase')
        plt.ylabel('Daily Return (%)')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def print_performance_summary(self, metrics, tests):
        """
        Print performance summary
        
        Parameters:
        -----------
        metrics : dict
            Dictionary of performance metrics
        tests : dict
            Dictionary of statistical test results
        """
        print("=" * 50)
        print("PERFORMANCE SUMMARY")
        print("=" * 50)
        
        print("\nReturns:")
        print(f"Total Strategy Return: {metrics['total_strategy_return']:.2%}")
        print(f"Total Buy & Hold Return: {metrics['total_buy_hold_return']:.2%}")
        print(f"Annual Strategy Return: {metrics['annual_strategy_return']:.2%}")
        print(f"Annual Buy & Hold Return: {metrics['annual_buy_hold_return']:.2%}")
        
        print("\nRisk Metrics:")
        print(f"Annual Strategy Volatility: {metrics['annual_strategy_vol']:.2%}")
        print(f"Annual Buy & Hold Volatility: {metrics['annual_buy_hold_vol']:.2%}")
        print(f"Strategy Sharpe Ratio: {metrics['strategy_sharpe']:.2f}")
        print(f"Buy & Hold Sharpe Ratio: {metrics['buy_hold_sharpe']:.2f}")
        print(f"Strategy Maximum Drawdown: {metrics['strategy_max_drawdown']:.2%}")
        print(f"Buy & Hold Maximum Drawdown: {metrics['buy_hold_max_drawdown']:.2%}")
        
        print("\nTrade Statistics:")
        print(f"Total Trades: {metrics['total_trades']}")
        print(f"Full Moon Trades: {metrics['full_moon_trades']}")
        print(f"New Moon Trades: {metrics['new_moon_trades']}")
        print(f"Strategy Win Rate: {metrics['strategy_win_rate']:.2%}")
        print(f"Buy & Hold Win Rate: {metrics['buy_hold_win_rate']:.2%}")
        
        print("\nLunar Phase Analysis:")
        print(f"Average Return on Full Moon: {metrics['full_moon_avg_return']:.2%}")
        print(f"Average Return on New Moon: {metrics['new_moon_avg_return']:.2%}")
        print(f"Average Return on Other Days: {metrics['other_days_avg_return']:.2%}")
        
        print("\nStatistical Tests:")
        print("\nT-Tests for Mean Differences:")
        if tests['ttest_full_vs_other'][0] is not None:
            print(f"Full Moon vs Other Days: t={tests['ttest_full_vs_other'][0]:.4f}, p-value={tests['ttest_full_vs_other'][1]:.4f}")
            print(f"New Moon vs Other Days: t={tests['ttest_new_vs_other'][0]:.4f}, p-value={tests['ttest_new_vs_other'][1]:.4f}")
            print(f"Full Moon vs New Moon: t={tests['ttest_full_vs_new'][0]:.4f}, p-value={tests['ttest_full_vs_new'][1]:.4f}")
        
        print("\nMann-Whitney U Tests (Non-parametric):")
        if tests['mannwhitney_full_vs_other'][0] is not None:
            print(f"Full Moon vs Other Days: U={tests['mannwhitney_full_vs_other'][0]:.4f}, p-value={tests['mannwhitney_full_vs_other'][1]:.4f}")
            print(f"New Moon vs Other Days: U={tests['mannwhitney_new_vs_other'][0]:.4f}, p-value={tests['mannwhitney_new_vs_other'][1]:.4f}")
            print(f"Full Moon vs New Moon: U={tests['mannwhitney_full_vs_new'][0]:.4f}, p-value={tests['mannwhitney_full_vs_new'][1]:.4f}")
        
        print("=" * 50)
        
    def backtest(self, data):
        """
        Run backtest on historical data
        
        Parameters:
        -----------
        data : DataFrame
            Historical price data with datetime index
            
        Returns:
        --------
        tuple
            (results, metrics, tests)
        """
        # Generate signals
        results = self.generate_signals(data)
        
        # Calculate returns
        results = self.calculate_returns(results)
        
        # Calculate performance metrics
        metrics = self.calculate_performance_metrics(results)
        
        # Perform statistical tests
        tests = self.statistical_tests(results)
        
        return results, metrics, tests

def generate_synthetic_data(start_date='2020-01-01', end_date='2025-06-30', periodicity=None):
    """
    Generate synthetic price data with random walk and optional lunar periodicity
    
    Parameters:
    -----------
    start_date : str
        Start date for synthetic data
    end_date : str
        End date for synthetic data
    periodicity : float or None
        Strength of lunar periodicity (0 to 1, None for no periodicity)
        
    Returns:
    --------
    DataFrame
        Synthetic price data
    """
    # Convert dates
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    
    # Generate date range
    dates = pd.date_range(start=start, end=end, freq='B')  # Business days
    
    # Set seed for reproducibility
    np.random.seed(42)
    
    # Generate random walk
    returns = np.random.normal(0.0005, 0.015, len(dates))  # Mean daily return ~0.05%, std ~1.5%
    
    # Add lunar periodicity if specified
    if periodicity is not None:
        # Create a lunar cycle (approximately 29.53 days)
        lunar_cycle = 29.53
        lunar_effect = np.sin(2 * np.pi * np.arange(len(dates)) / lunar_cycle)
        
        # Add lunar effect to returns
        returns = returns + periodicity * 0.01 * lunar_effect
    
    # Calculate prices from returns (starting at 100)
    prices = 100 * np.cumprod(1 + returns)
    
    # Generate OHLC data
    daily_range = np.random.uniform(0.005, 0.025, len(dates))
    opens = prices * (1 - daily_range/4)
    highs = prices * (1 + daily_range/2)
    lows = prices * (1 - daily_range/2)
    closes = prices
    
    # Generate volume data
    volumes = np.random.lognormal(mean=15, sigma=0.5, size=len(dates))
    
    # Create DataFrame
    df = pd.DataFrame({
        'open': opens,
        'high': highs,
        'low': lows,
        'close': closes,
        'volume': volumes
    }, index=dates)
    
    return df

def run_test_with_threshold(data, threshold_hours=12):
    """
    Run a single test with specific threshold hours
    """
    strategy = LunarMomentumStrategy(threshold_hours=threshold_hours)
    strategy.add_historical_moon_phases(years_back=5)
    
    results, metrics, tests = strategy.backtest(data)
    
    return results, metrics, tests

def run_sensitivity_analysis():
    """
    Run sensitivity analysis on the threshold parameter
    """
    # Generate synthetic data with medium lunar periodicity
    print("Generating synthetic data...")
    data = generate_synthetic_data(periodicity=0.2)
    
    # Test different threshold values
    thresholds = [6, 12, 24, 36, 48, 72]
    threshold_results = []
    
    print("\nRunning sensitivity analysis on threshold parameter...")
    for threshold in thresholds:
        print(f"Testing threshold = {threshold} hours...")
        results, metrics, tests = run_test_with_threshold(data, threshold_hours=threshold)
        
        threshold_results.append({
            'threshold_hours': threshold,
            'total_return': metrics['total_strategy_return'],
            'annual_return': metrics['annual_strategy_return'],
            'sharpe': metrics['strategy_sharpe'],
            'max_drawdown': metrics['strategy_max_drawdown'],
            'win_rate': metrics['strategy_win_rate'],
            'full_moon_avg_return': metrics['full_moon_avg_return'],
            'new_moon_avg_return': metrics['new_moon_avg_return'],
            'full_moon_p_value': tests['ttest_full_vs_other'][1] if tests['ttest_full_vs_other'][0] is not None else None,
            'new_moon_p_value': tests['ttest_new_vs_other'][1] if tests['ttest_new_vs_other'][0] is not None else None
        })
    
    # Create DataFrame with results
    threshold_df = pd.DataFrame(threshold_results)
    
    # Plot results
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    plt.plot(threshold_df['threshold_hours'], threshold_df['total_return'] * 100, marker='o')
    plt.title('Total Return by Threshold Hours')
    plt.xlabel('Threshold Hours')
    plt.ylabel('Total Return (%)')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 2)
    plt.plot(threshold_df['threshold_hours'], threshold_df['sharpe'], marker='o')
    plt.title('Sharpe Ratio by Threshold Hours')
    plt.xlabel('Threshold Hours')
    plt.ylabel('Sharpe Ratio')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 3)
    plt.plot(threshold_df['threshold_hours'], threshold_df['max_drawdown'] * 100, marker='o')
    plt.title('Max Drawdown by Threshold Hours')
    plt.xlabel('Threshold Hours')
    plt.ylabel('Max Drawdown (%)')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 4)
    plt.plot(threshold_df['threshold_hours'], threshold_df['win_rate'] * 100, marker='o')
    plt.title('Win Rate by Threshold Hours')
    plt.xlabel('Threshold Hours')
    plt.ylabel('Win Rate (%)')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Plot p-values
    plt.figure(figsize=(12, 6))
    
    plt.plot(threshold_df['threshold_hours'], threshold_df['full_moon_p_value'], marker='o', label='Full Moon vs Other')
    plt.plot(threshold_df['threshold_hours'], threshold_df['new_moon_p_value'], marker='s', label='New Moon vs Other')
    plt.axhline(y=0.05, color='r', linestyle='--', label='5% Significance Level')
    
    plt.title('Statistical Significance by Threshold Hours')
    plt.xlabel('Threshold Hours')
    plt.ylabel('p-value')
    plt.yscale('log')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return threshold_df

def run_periodicity_analysis():
    """
    Run analysis with different levels of lunar periodicity
    """
    print("Running analysis with different levels of lunar periodicity...")
    
    # Test different periodicity values
    periodicity_values = [0, 0.05, 0.1, 0.2, 0.3, 0.5]
    periodicity_results = []
    
    for periodicity in periodicity_values:
        print(f"\nTesting periodicity = {periodicity}...")
        
        # Generate data with specified periodicity
        data = generate_synthetic_data(periodicity=periodicity)
        
        # Run strategy with default threshold
        results, metrics, tests = run_test_with_threshold(data)
        
        periodicity_results.append({
            'periodicity': periodicity,
            'total_return': metrics['total_strategy_return'],
            'annual_return': metrics['annual_strategy_return'],
            'sharpe': metrics['strategy_sharpe'],
            'max_drawdown': metrics['strategy_max_drawdown'],
            'win_rate': metrics['strategy_win_rate'],
            'full_moon_avg_return': metrics['full_moon_avg_return'],
            'new_moon_avg_return': metrics['new_moon_avg_return'],
            'full_moon_p_value': tests['ttest_full_vs_other'][1] if tests['ttest_full_vs_other'][0] is not None else None,
            'new_moon_p_value': tests['ttest_new_vs_other'][1] if tests['ttest_new_vs_other'][0] is not None else None
        })
        
        # Plot results for extreme cases
        if periodicity == 0 or periodicity == 0.5:
            print(f"\nPlotting results for periodicity = {periodicity}...")
            strategy = LunarMomentumStrategy()
            strategy.add_historical_moon_phases(years_back=5)
            strategy.plot_results(results)
            strategy.print_performance_summary(metrics, tests)
    
    # Create DataFrame with results
    periodicity_df = pd.DataFrame(periodicity_results)
    
    # Plot results
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    plt.plot(periodicity_df['periodicity'], periodicity_df['total_return'] * 100, marker='o')
    plt.title('Total Return by Lunar Periodicity')
    plt.xlabel('Periodicity Strength')
    plt.ylabel('Total Return (%)')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 2)
    plt.plot(periodicity_df['periodicity'], periodicity_df['sharpe'], marker='o')
    plt.title('Sharpe Ratio by Lunar Periodicity')
    plt.xlabel('Periodicity Strength')
    plt.ylabel('Sharpe Ratio')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 3)
    plt.plot(periodicity_df['periodicity'], periodicity_df['max_drawdown'] * 100, marker='o')
    plt.title('Max Drawdown by Lunar Periodicity')
    plt.xlabel('Periodicity Strength')
    plt.ylabel('Max Drawdown (%)')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 4)
    plt.plot(periodicity_df['periodicity'], periodicity_df['win_rate'] * 100, marker='o')
    plt.title('Win Rate by Lunar Periodicity')
    plt.xlabel('Periodicity Strength')
    plt.ylabel('Win Rate (%)')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Plot p-values
    plt.figure(figsize=(12, 6))
    
    plt.plot(periodicity_df['periodicity'], periodicity_df['full_moon_p_value'], marker='o', label='Full Moon vs Other')
    plt.plot(periodicity_df['periodicity'], periodicity_df['new_moon_p_value'], marker='s', label='New Moon vs Other')
    plt.axhline(y=0.05, color='r', linestyle='--', label='5% Significance Level')
    
    plt.title('Statistical Significance by Lunar Periodicity')
    plt.xlabel('Periodicity Strength')
    plt.ylabel('p-value')
    plt.yscale('log')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return periodicity_df

def run_combined_strategy_analysis():
    """
    Test a combined strategy that incorporates lunar signals with technical indicators
    """
    print("\nTesting Combined Strategy (Lunar + Technical Indicators)...")
    
    # Generate data with medium periodicity
    data = generate_synthetic_data(periodicity=0.2)
    
    # Calculate technical indicators
    data['sma50'] = data['close'].rolling(window=50).mean()
    data['sma200'] = data['close'].rolling(window=200).mean()
    data['rsi'] = calculate_rsi(data['close'], window=14)
    
    # Run lunar strategy for comparison
    strategy = LunarMomentumStrategy()
    strategy.add_historical_moon_phases(years_back=5)
    lunar_results, lunar_metrics, _ = strategy.backtest(data)
    
    # Create a combined strategy
    # Only take lunar signals when technical indicators are favorable
    combined_results = lunar_results.copy()
    
    # Override signals based on technical conditions
    combined_results['technical_trend'] = np.where(
        combined_results['sma50'] > combined_results['sma200'],
        'uptrend',
        'downtrend'
    )
    
    # Only take lunar buy signals in uptrend
    # Only exit positions in downtrend
    for i in range(1, len(combined_results)):
        prev_date = combined_results.index[i-1]
        curr_date = combined_results.index[i]
        
        # Original lunar signal
        original_signal = combined_results.loc[curr_date, 'signal']
        
        # Technical condition
        is_uptrend = combined_results.loc[curr_date, 'technical_trend'] == 'uptrend'
        is_overbought = combined_results.loc[curr_date, 'rsi'] > 70
        is_oversold = combined_results.loc[curr_date, 'rsi'] < 30
        
        # Modify signal based on technical conditions
        if original_signal in [1, 2]:  # Buy signals
            if not is_uptrend or is_overbought:
                combined_results.loc[curr_date, 'signal'] = 0  # Cancel buy signal
        elif original_signal in [-1, -2]:  # Sell signals
            if is_uptrend and not is_overbought:
                combined_results.loc[curr_date, 'signal'] = 0  # Cancel sell signal
        
        # Add exit based on technical conditions
        if combined_results.loc[prev_date, 'position'] > 0:  # In position
            if not is_uptrend or is_overbought:
                combined_results.loc[curr_date, 'signal'] = -1  # Exit position
    
    # Recalculate positions
    position = 0
    for i, date in enumerate(combined_results.index):
        signal = combined_results.loc[date, 'signal']
        
        # Close positions first
        if signal == -1 or signal == -2:
            position = 0
            combined_results.loc[date, 'in_position'] = False
            combined_results.loc[date, 'entry_price'] = np.nan
            combined_results.loc[date, 'entry_type'] = None
        
        # Then open new positions
        if signal == 1:  # Buy on full moon
            position = 1
            combined_results.loc[date, 'in_position'] = True
            combined_results.loc[date, 'entry_price'] = combined_results.loc[date, 'close']
            combined_results.loc[date, 'entry_type'] = 'Full Moon'
        elif signal == 2:  # Buy on new moon
            position = 2
            combined_results.loc[date, 'in_position'] = True
            combined_results.loc[date, 'entry_price'] = combined_results.loc[date, 'close']
            combined_results.loc[date, 'entry_type'] = 'New Moon'
        
        combined_results.loc[date, 'position'] = position
    
    # Recalculate returns
    combined_results = strategy.calculate_returns(combined_results)
    
    # Calculate performance metrics
    combined_metrics = strategy.calculate_performance_metrics(combined_results)
    
    # Plot comparison
    plt.figure(figsize=(12, 8))
    
    plt.plot(lunar_results.index, (1 + lunar_results['cum_strategy_returns']) * 100, label='Lunar Only', color='blue')
    plt.plot(combined_results.index, (1 + combined_results['cum_strategy_returns']) * 100, label='Lunar + Technical', color='green')
    plt.plot(lunar_results.index, (1 + lunar_results['cum_buy_hold_returns']) * 100, label='Buy & Hold', color='gray', alpha=0.6)
    
    plt.title('Comparison: Lunar Only vs Combined Strategy')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print comparison
    print("\n" + "=" * 70)
    print("STRATEGY COMPARISON")
    print("=" * 70)
    
    comparison_df = pd.DataFrame({
        'Lunar Only': [
            lunar_metrics['total_strategy_return'] * 100,
            lunar_metrics['annual_strategy_return'] * 100,
            lunar_metrics['strategy_sharpe'],
            lunar_metrics['strategy_max_drawdown'] * 100,
            lunar_metrics['strategy_win_rate'] * 100,
            lunar_metrics['total_trades']
        ],
        'Lunar + Technical': [
            combined_metrics['total_strategy_return'] * 100,
            combined_metrics['annual_strategy_return'] * 100,
            combined_metrics['strategy_sharpe'],
            combined_metrics['strategy_max_drawdown'] * 100,
            combined_metrics['strategy_win_rate'] * 100,
            combined_metrics['total_trades']
        ],
        'Buy & Hold': [
            lunar_metrics['total_buy_hold_return'] * 100,
            lunar_metrics['annual_buy_hold_return'] * 100,
            lunar_metrics['buy_hold_sharpe'],
            lunar_metrics['buy_hold_max_drawdown'] * 100,
            lunar_metrics['buy_hold_win_rate'] * 100,
            0
        ]
    }, index=[
        'Total Return (%)',
        'Annual Return (%)',
        'Sharpe Ratio',
        'Max Drawdown (%)',
        'Win Rate (%)',
        'Total Trades'
    ])
    
    print(comparison_df)
    
    return {
        'lunar_results': lunar_results,
        'lunar_metrics': lunar_metrics,
        'combined_results': combined_results,
        'combined_metrics': combined_metrics
    }

def calculate_rsi(prices, window=14):
    """
    Calculate Relative Strength Index
    """
    deltas = np.diff(prices)
    seed = deltas[:window+1]
    up = seed[seed >= 0].sum()/window
    down = -seed[seed < 0].sum()/window
    rs = up/down if down != 0 else np.inf
    rsi = np.zeros_like(prices)
    rsi[:window] = 100. - 100./(1. + rs)
    
    for i in range(window, len(prices)):
        delta = deltas[i-1]
        if delta > 0:
            upval = delta
            downval = 0.
        else:
            upval = 0.
            downval = -delta
            
        up = (up * (window - 1) + upval) / window
        down = (down * (window - 1) + downval) / window
        
        rs = up/down if down != 0 else np.inf
        rsi[i] = 100. - 100./(1. + rs)
        
    return rsi

# Main execution
if __name__ == "__main__":
    # Run sensitivity analysis
    threshold_df = run_sensitivity_analysis()
    
    # Run periodicity analysis
    periodicity_df = run_periodicity_analysis()
    
    # Run combined strategy analysis
    combined_results = run_combined_strategy_analysis()