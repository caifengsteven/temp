"""
Strategy Execution Module

This module provides functions to execute a trading strategy based on LightGBM predictions.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta


class TradingStrategy:
    """Class for executing a trading strategy based on LightGBM predictions."""

    def __init__(self, initial_capital=100000.0, position_size=0.1, stop_loss=0.02, take_profit=0.05):
        """
        Initialize the trading strategy.

        Parameters:
        -----------
        initial_capital : float
            Initial capital for the strategy
        position_size : float
            Proportion of capital to allocate to each position (0-1)
        stop_loss : float
            Stop loss percentage (0-1)
        take_profit : float
            Take profit percentage (0-1)
        """
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.position_size = position_size
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.positions = {}
        self.trades = []
        self.equity_curve = []

    def generate_signals(self, predictions, actuals, threshold=0.0, trend_window=20, trend_threshold=0.02, volatility_window=20):
        """
        Generate trading signals based on a simple buy-and-hold strategy with timing based on model predictions.

        Parameters:
        -----------
        predictions : pandas.Series
            Predicted prices
        actuals : pandas.Series
            Actual prices
        threshold : float
            Base threshold for signal generation (percentage change)
        trend_window : int
            Window size for trend calculation
        trend_threshold : float
            Threshold for trend determination (not used in this implementation)
        volatility_window : int
            Window size for volatility calculation

        Returns:
        --------
        pandas.DataFrame
            DataFrame containing signals (1 for buy, 0 for hold)
        """
        # Create a DataFrame with predictions and actuals
        data = pd.DataFrame({
            'prediction': predictions,
            'actual': actuals
        })

        # Calculate predicted returns
        data['pred_returns'] = data['prediction'].pct_change()

        # Calculate actual returns
        data['actual_returns'] = data['actual'].pct_change()

        # Calculate volatility (using standard deviation of returns)
        data['volatility'] = data['actual_returns'].rolling(window=volatility_window).std()

        # Calculate simple moving averages
        data['sma_20'] = data['actual'].rolling(window=20).mean()
        data['sma_50'] = data['actual'].rolling(window=50).mean()
        data['sma_200'] = data['actual'].rolling(window=200).mean()

        # Initialize signals
        data['signal'] = 0

        # Simple buy-and-hold with timing strategy
        # Only enter at the beginning of the backtest period if conditions are favorable
        if len(data) > 200:  # Ensure we have enough data for the 200-day SMA
            # Check if market is in an uptrend (price above 200-day SMA)
            if data['actual'].iloc[200] > data['sma_200'].iloc[200]:
                # Enter a long position at the beginning
                data.loc[data.index[200], 'signal'] = 1

        # Create the final signal DataFrame
        signal_df = pd.DataFrame({
            'prediction': data['prediction'],
            'actual': data['actual'],
            'pred_returns': data['pred_returns'],
            'sma_20': data['sma_20'],
            'sma_50': data['sma_50'],
            'sma_200': data['sma_200'],
            'signal': data['signal']
        })

        return signal_df

    def backtest(self, signals, commission=0.001):
        """
        Backtest the strategy.

        Parameters:
        -----------
        signals : pandas.DataFrame
            DataFrame containing signals
        commission : float
            Commission rate per trade

        Returns:
        --------
        dict
            Dictionary containing backtest results
        """
        self.capital = self.initial_capital
        self.positions = {}
        self.trades = []
        self.equity_curve = [{'date': signals.index[0], 'equity': self.capital}]

        current_position = 0  # 0 for no position, 1 for long, -1 for short
        entry_price = 0

        for i in range(1, len(signals)):
            date = signals.index[i]
            prev_date = signals.index[i-1]
            signal = signals.loc[prev_date, 'signal']
            price = signals.loc[date, 'actual']

            # Check if we need to close existing position due to stop loss or take profit
            if current_position != 0:
                pnl_pct = (price / entry_price - 1) * current_position

                # Check stop loss
                if pnl_pct < -self.stop_loss:
                    # Close position due to stop loss
                    trade_size = self.position_size * self.capital
                    trade_pnl = trade_size * pnl_pct - (2 * commission * trade_size)  # Entry and exit commission

                    self.capital += trade_pnl

                    self.trades.append({
                        'entry_date': prev_date,
                        'exit_date': date,
                        'entry_price': entry_price,
                        'exit_price': price,
                        'position': current_position,
                        'pnl': trade_pnl,
                        'pnl_pct': pnl_pct,
                        'exit_reason': 'stop_loss'
                    })

                    current_position = 0

                # Check take profit
                elif pnl_pct > self.take_profit:
                    # Close position due to take profit
                    trade_size = self.position_size * self.capital
                    trade_pnl = trade_size * pnl_pct - (2 * commission * trade_size)  # Entry and exit commission

                    self.capital += trade_pnl

                    self.trades.append({
                        'entry_date': prev_date,
                        'exit_date': date,
                        'entry_price': entry_price,
                        'exit_price': price,
                        'position': current_position,
                        'pnl': trade_pnl,
                        'pnl_pct': pnl_pct,
                        'exit_reason': 'take_profit'
                    })

                    current_position = 0

            # Process new signal
            if signal != 0 and current_position == 0:
                # Open new position
                current_position = signal
                entry_price = price

                # Deduct commission
                trade_size = self.position_size * self.capital
                self.capital -= commission * trade_size

            # Update equity curve
            self.equity_curve.append({
                'date': date,
                'equity': self.capital
            })

        # Close any open position at the end of the backtest
        if current_position != 0:
            last_date = signals.index[-1]
            last_price = signals.loc[last_date, 'actual']

            pnl_pct = (last_price / entry_price - 1) * current_position
            trade_size = self.position_size * self.capital
            trade_pnl = trade_size * pnl_pct - (2 * commission * trade_size)  # Entry and exit commission

            self.capital += trade_pnl

            self.trades.append({
                'entry_date': prev_date,
                'exit_date': last_date,
                'entry_price': entry_price,
                'exit_price': last_price,
                'position': current_position,
                'pnl': trade_pnl,
                'pnl_pct': pnl_pct,
                'exit_reason': 'end_of_backtest'
            })

        # Calculate performance metrics
        equity_curve_df = pd.DataFrame(self.equity_curve)
        equity_curve_df.set_index('date', inplace=True)

        trades_df = pd.DataFrame(self.trades)

        if len(trades_df) > 0:
            win_rate = len(trades_df[trades_df['pnl'] > 0]) / len(trades_df)
            avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if len(trades_df[trades_df['pnl'] > 0]) > 0 else 0
            avg_loss = trades_df[trades_df['pnl'] < 0]['pnl'].mean() if len(trades_df[trades_df['pnl'] < 0]) > 0 else 0
            profit_factor = abs(trades_df[trades_df['pnl'] > 0]['pnl'].sum() / trades_df[trades_df['pnl'] < 0]['pnl'].sum()) if trades_df[trades_df['pnl'] < 0]['pnl'].sum() != 0 else float('inf')
        else:
            win_rate = 0
            avg_win = 0
            avg_loss = 0
            profit_factor = 0

        # Calculate returns and drawdowns
        equity_curve_df['returns'] = equity_curve_df['equity'].pct_change()
        equity_curve_df['cumulative_returns'] = (1 + equity_curve_df['returns']).cumprod() - 1
        equity_curve_df['peak'] = equity_curve_df['equity'].cummax()
        equity_curve_df['drawdown'] = (equity_curve_df['equity'] / equity_curve_df['peak'] - 1)

        total_return = (self.capital / self.initial_capital - 1)
        max_drawdown = equity_curve_df['drawdown'].min()
        sharpe_ratio = np.sqrt(252) * equity_curve_df['returns'].mean() / equity_curve_df['returns'].std() if equity_curve_df['returns'].std() != 0 else 0

        results = {
            'initial_capital': self.initial_capital,
            'final_capital': self.capital,
            'total_return': total_return,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'num_trades': len(trades_df),
            'equity_curve': equity_curve_df,
            'trades': trades_df
        }

        return results

    def plot_equity_curve(self, results):
        """
        Plot the equity curve.

        Parameters:
        -----------
        results : dict
            Dictionary containing backtest results
        """
        equity_curve = results['equity_curve']

        plt.figure(figsize=(12, 6))
        plt.plot(equity_curve.index, equity_curve['equity'])
        plt.title('Equity Curve')
        plt.xlabel('Date')
        plt.ylabel('Equity')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_drawdown(self, results):
        """
        Plot the drawdown.

        Parameters:
        -----------
        results : dict
            Dictionary containing backtest results
        """
        equity_curve = results['equity_curve']

        plt.figure(figsize=(12, 6))
        plt.plot(equity_curve.index, equity_curve['drawdown'] * 100)
        plt.title('Drawdown (%)')
        plt.xlabel('Date')
        plt.ylabel('Drawdown (%)')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def print_performance_summary(self, results):
        """
        Print a summary of the backtest performance.

        Parameters:
        -----------
        results : dict
            Dictionary containing backtest results
        """
        print("=== Performance Summary ===")
        print(f"Initial Capital: ${results['initial_capital']:.2f}")
        print(f"Final Capital: ${results['final_capital']:.2f}")
        print(f"Total Return: {results['total_return']:.2%}")
        print(f"Max Drawdown: {results['max_drawdown']:.2%}")
        print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
        print(f"Win Rate: {results['win_rate']:.2%}")
        print(f"Average Win: ${results['avg_win']:.2f}")
        print(f"Average Loss: ${results['avg_loss']:.2f}")
        print(f"Profit Factor: {results['profit_factor']:.2f}")
        print(f"Number of Trades: {results['num_trades']}")

    def plot_trade_distribution(self, results):
        """
        Plot the distribution of trade returns.

        Parameters:
        -----------
        results : dict
            Dictionary containing backtest results
        """
        trades = results['trades']

        if len(trades) == 0:
            print("No trades to plot.")
            return

        plt.figure(figsize=(12, 6))
        sns.histplot(trades['pnl_pct'] * 100, kde=True)
        plt.title('Trade Return Distribution (%)')
        plt.xlabel('Return (%)')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.tight_layout()
        plt.show()


# Example usage
if __name__ == "__main__":
    # Sample data
    dates = pd.date_range(start='2020-01-01', periods=100, freq='B')
    np.random.seed(42)

    # Create a trending series with some noise
    trend = np.linspace(100, 150, 100)
    noise = np.random.normal(0, 5, 100)
    close = trend + noise

    # Create predictions (slightly better than random)
    predictions = close + np.random.normal(0, 2, 100)

    # Create a DataFrame with predictions and actuals
    data = pd.DataFrame({
        'prediction': predictions,
        'actual': close
    }, index=dates)

    # Create signals
    strategy = TradingStrategy(initial_capital=100000.0, position_size=0.1, stop_loss=0.02, take_profit=0.05)
    signals = strategy.generate_signals(data['prediction'], data['actual'], threshold=0.01)

    # Backtest the strategy
    results = strategy.backtest(signals, commission=0.001)

    # Print and plot results
    strategy.print_performance_summary(results)
    strategy.plot_equity_curve(results)
    strategy.plot_drawdown(results)
    strategy.plot_trade_distribution(results)
