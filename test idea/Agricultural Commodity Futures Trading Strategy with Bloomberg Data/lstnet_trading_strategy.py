import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import blpapi
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

# Import our LSTNet model and related classes
from lstnet_model import LSTNet
from train_lstnet_improved import ImprovedTimeSeriesDataset, BloombergDataManager

# Set seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class LSTNetTradingStrategy:
    def __init__(self, model, dataset, data, window_size=20, horizon=5, threshold=0.01,
                 position_size=0.1, trade_cost=0.001):
        """
        Initialize LSTNet Trading Strategy

        Args:
            model: Trained LSTNet model
            dataset: Dataset object used for training
            data: Original price data
            window_size: Input window size
            horizon: Forecasting horizon
            threshold: Threshold for signal generation
            position_size: Position size as percentage of capital
            trade_cost: Trading cost as percentage
        """
        self.model = model
        self.dataset = dataset
        self.data = data
        self.window_size = window_size
        self.horizon = horizon
        self.threshold = threshold
        self.position_size = position_size
        self.trade_cost = trade_cost
        self.positions = {ticker: 0 for ticker in data.columns}
        self.capital = 1000000  # Initial capital
        self.portfolio_value = self.capital
        self.performance_history = []

        # Set model to evaluation mode
        self.model.eval()

    def generate_signals(self, current_window):
        """
        Generate trading signals based on model predictions

        Args:
            current_window: Current window of data

        Returns:
            Dictionary of signals for each ticker
        """
        # Normalize the window data
        if self.dataset.normalize and self.dataset.scalers is not None:
            normalized_window = np.zeros_like(current_window, dtype=np.float32)
            for i in range(current_window.shape[1]):
                normalized_window[:, i] = self.dataset.scalers[i].transform(
                    current_window[:, i].reshape(-1, 1)).flatten()
        else:
            normalized_window = current_window.astype(np.float32)

        # Convert to tensor
        X = torch.FloatTensor(normalized_window).unsqueeze(0).to(device)

        # Generate prediction
        with torch.no_grad():
            y_pred = self.model(X).cpu().numpy()[0]

        # Inverse transform prediction
        if self.dataset.normalize and self.dataset.scalers is not None:
            y_pred_orig = np.zeros_like(y_pred)
            for i in range(y_pred.shape[0]):
                y_pred_orig[i] = self.dataset.scalers[i].inverse_transform(
                    y_pred[i].reshape(-1, 1)).flatten()[0]
            y_pred = y_pred_orig

        # Current prices (last price in the window)
        current_prices = current_window[-1]

        # Calculate expected returns
        expected_returns = (y_pred - current_prices) / current_prices

        # Generate signals
        signals = {}
        for i, ticker in enumerate(self.data.columns):
            if expected_returns[i] > self.threshold:
                signals[ticker] = 'BUY'
            elif expected_returns[i] < -self.threshold:
                signals[ticker] = 'SELL'
            else:
                signals[ticker] = 'HOLD'

        return signals, y_pred, expected_returns

    def execute_trades(self, signals, current_prices):
        """
        Execute trades based on signals

        Args:
            signals: Dictionary of signals for each ticker
            current_prices: Current prices as Series

        Returns:
            List of executed trades
        """
        executed_trades = []

        for ticker, signal in signals.items():
            current_price = current_prices[ticker]
            current_position = self.positions.get(ticker, 0)

            if signal == 'BUY' and current_position <= 0:
                # Calculate position size in units
                position_value = self.portfolio_value * self.position_size
                units_to_buy = position_value / current_price

                # Calculate transaction cost
                transaction_cost = position_value * self.trade_cost

                # Check if enough capital
                if position_value + transaction_cost <= self.capital:
                    # Execute buy order
                    self.positions[ticker] = units_to_buy
                    self.capital -= (position_value + transaction_cost)

                    # Record trade
                    executed_trades.append({
                        'ticker': ticker,
                        'action': 'BUY',
                        'price': current_price,
                        'units': units_to_buy,
                        'value': position_value,
                        'cost': transaction_cost,
                        'timestamp': datetime.now()
                    })

            elif signal == 'SELL' and current_position >= 0:
                if current_position > 0:
                    # Calculate position value
                    position_value = current_position * current_price

                    # Calculate transaction cost
                    transaction_cost = position_value * self.trade_cost

                    # Execute sell order
                    self.capital += (position_value - transaction_cost)
                    self.positions[ticker] = 0

                    # Record trade
                    executed_trades.append({
                        'ticker': ticker,
                        'action': 'SELL',
                        'price': current_price,
                        'units': current_position,
                        'value': position_value,
                        'cost': transaction_cost,
                        'timestamp': datetime.now()
                    })

                # Consider short selling if allowed
                position_value = self.portfolio_value * self.position_size
                units_to_short = position_value / current_price

                # Calculate transaction cost
                transaction_cost = position_value * self.trade_cost

                # Check if enough capital for margin
                if position_value * 0.5 <= self.capital:  # Assuming 50% margin requirement
                    # Execute short order
                    self.positions[ticker] = -units_to_short
                    self.capital -= transaction_cost

                    # Record trade
                    executed_trades.append({
                        'ticker': ticker,
                        'action': 'SHORT',
                        'price': current_price,
                        'units': units_to_short,
                        'value': position_value,
                        'cost': transaction_cost,
                        'timestamp': datetime.now()
                    })

        return executed_trades

    def update_portfolio_value(self, current_prices):
        """
        Update portfolio value based on current prices

        Args:
            current_prices: Current prices as Series

        Returns:
            Updated portfolio value
        """
        position_value = 0

        for ticker, units in self.positions.items():
            if ticker in current_prices:
                position_value += units * current_prices[ticker]

        self.portfolio_value = self.capital + position_value
        self.performance_history.append({
            'timestamp': datetime.now(),
            'portfolio_value': self.portfolio_value,
            'cash': self.capital,
            'position_value': position_value
        })

        return self.portfolio_value

    def get_performance_metrics(self):
        """
        Calculate performance metrics

        Returns:
            Dictionary of performance metrics
        """
        if len(self.performance_history) < 2:
            return {
                'total_return': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'volatility': 0,
                'current_value': self.portfolio_value
            }

        # Extract portfolio values
        values = [record['portfolio_value'] for record in self.performance_history]

        # Calculate returns
        returns = [(values[i] - values[i-1]) / values[i-1] for i in range(1, len(values))]

        # Calculate metrics
        total_return = (values[-1] / values[0]) - 1
        volatility = np.std(returns) * np.sqrt(252)  # Annualized
        sharpe_ratio = (np.mean(returns) * 252) / volatility if volatility > 0 else 0

        # Calculate drawdown
        drawdowns = [1 - values[i] / max(values[:i+1]) for i in range(len(values))]
        max_drawdown = max(drawdowns) if drawdowns else 0

        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'volatility': volatility,
            'current_value': self.portfolio_value
        }

    def backtest(self, data=None):
        """
        Backtest strategy on historical data

        Args:
            data: Historical data as DataFrame (if None, use self.data)

        Returns:
            Backtest results
        """
        if data is None:
            data = self.data

        # Reset strategy state
        self.positions = {ticker: 0 for ticker in data.columns}
        self.capital = 1000000
        self.portfolio_value = self.capital
        self.performance_history = []

        # Track trades and daily performance
        trades = []
        daily_performance = []
        predictions = []

        # We need at least window_size data points to start
        start_idx = self.window_size

        # Iterate through each day
        for i in range(start_idx, len(data) - self.horizon + 1):
            current_date = data.index[i]

            # Get current window
            window_data = data.iloc[i-self.window_size:i].values

            # Generate signals
            signals, y_pred, expected_returns = self.generate_signals(window_data)

            # Store predictions
            predictions.append({
                'date': current_date,
                'actual': data.iloc[i+self.horizon-1].values,
                'predicted': y_pred,
                'expected_returns': expected_returns
            })

            # Get current prices
            current_prices = data.iloc[i]

            # Execute trades
            executed_trades = self.execute_trades(signals, current_prices)
            if executed_trades:
                trades.extend(executed_trades)

            # Update portfolio value
            self.update_portfolio_value(current_prices)

            # Record daily performance
            daily_performance.append({
                'date': current_date,
                'portfolio_value': self.portfolio_value,
                'cash': self.capital,
                'positions': self.positions.copy()
            })

        # Calculate performance metrics
        metrics = self.get_performance_metrics()

        # Prepare DataFrames for analysis
        trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
        daily_df = pd.DataFrame(daily_performance)
        if not daily_df.empty:
            daily_df.set_index('date', inplace=True)

        predictions_df = pd.DataFrame(predictions)
        if not predictions_df.empty:
            predictions_df.set_index('date', inplace=True)

        if not trades_df.empty:
            trades_df['profit'] = None

            # Calculate profit for each closed position
            for ticker in data.columns:
                ticker_trades = trades_df[trades_df['ticker'] == ticker].copy()

                for i in range(len(ticker_trades)):
                    if ticker_trades.iloc[i]['action'] == 'SELL' and i > 0:
                        buy_price = ticker_trades.iloc[i-1]['price'] if ticker_trades.iloc[i-1]['action'] == 'BUY' else None
                        if buy_price is not None:
                            sell_price = ticker_trades.iloc[i]['price']
                            units = min(ticker_trades.iloc[i]['units'], ticker_trades.iloc[i-1]['units'])
                            profit = units * (sell_price - buy_price)
                            trades_df.loc[ticker_trades.index[i], 'profit'] = profit

        # Visualization
        self._visualize_backtest_results(daily_df, trades_df, predictions_df, data)

        return {
            'metrics': metrics,
            'trades': trades_df,
            'daily_performance': daily_df,
            'predictions': predictions_df
        }

    def _visualize_backtest_results(self, daily_df, trades_df, predictions_df, original_data):
        """
        Visualize backtest results

        Args:
            daily_df: Daily performance DataFrame
            trades_df: Trades DataFrame
            predictions_df: Predictions DataFrame
            original_data: Original price data
        """
        # Plot portfolio value
        plt.figure(figsize=(14, 8))
        plt.subplot(2, 1, 1)
        plt.plot(daily_df['portfolio_value'])
        plt.title('Portfolio Value Over Time')
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.grid(True)

        # Plot trades
        if not trades_df.empty and 'profit' in trades_df.columns:
            plt.subplot(2, 1, 2)
            profitable_trades = trades_df[trades_df['profit'] > 0]
            losing_trades = trades_df[trades_df['profit'] < 0]

            if not profitable_trades.empty:
                plt.scatter(profitable_trades['timestamp'], profitable_trades['profit'],
                         color='green', label='Profitable Trades')

            if not losing_trades.empty:
                plt.scatter(losing_trades['timestamp'], losing_trades['profit'],
                         color='red', label='Losing Trades')

            plt.title('Trade Profits')
            plt.xlabel('Date')
            plt.ylabel('Profit')
            plt.legend()
            plt.grid(True)

        plt.tight_layout()
        plt.savefig('lstnet_backtest_results.png')

        # Plot predictions vs actual for each ticker
        for i, ticker in enumerate(original_data.columns):
            plt.figure(figsize=(14, 10))

            # Plot price and predictions
            plt.subplot(3, 1, 1)

            # Extract actual and predicted values for this ticker
            actual_values = np.array([pred['actual'][i] for pred in predictions_df.to_dict('records')])
            predicted_values = np.array([pred['predicted'][i] for pred in predictions_df.to_dict('records')])

            plt.plot(predictions_df.index, actual_values, label=f'Actual {ticker}')
            plt.plot(predictions_df.index, predicted_values, label=f'Predicted {ticker}')
            plt.title(f'{ticker} Price: Actual vs Predicted')
            plt.xlabel('Date')
            plt.ylabel('Price')
            plt.legend()
            plt.grid(True)

            # Plot expected returns
            plt.subplot(3, 1, 2)
            expected_returns = np.array([pred['expected_returns'][i] for pred in predictions_df.to_dict('records')])
            plt.plot(predictions_df.index, expected_returns)
            plt.axhline(y=self.threshold, color='g', linestyle='--', label='Buy Threshold')
            plt.axhline(y=-self.threshold, color='r', linestyle='--', label='Sell Threshold')
            plt.title(f'{ticker} Expected Returns')
            plt.xlabel('Date')
            plt.ylabel('Expected Return')
            plt.legend()
            plt.grid(True)

            # Plot prediction error
            plt.subplot(3, 1, 3)
            prediction_error = predicted_values - actual_values
            plt.plot(predictions_df.index, prediction_error)
            plt.title(f'{ticker} Prediction Error')
            plt.xlabel('Date')
            plt.ylabel('Error')
            plt.grid(True)

            plt.tight_layout()
            plt.savefig(f'lstnet_{ticker}_predictions.png')

        # Print performance metrics
        metrics = self.get_performance_metrics()
        print("\n===== Backtest Performance =====")
        print(f"Total Return: {metrics['total_return']:.2%}")
        print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        print(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
        print(f"Volatility: {metrics['volatility']:.2%}")
        print(f"Final Portfolio Value: ${metrics['current_value']:.2f}")

        if not trades_df.empty:
            total_trades = len(trades_df)
            profitable_trades = sum(1 for profit in trades_df['profit'] if profit is not None and profit > 0)
            win_rate = profitable_trades / sum(1 for profit in trades_df['profit'] if profit is not None) if sum(1 for profit in trades_df['profit'] if profit is not None) > 0 else 0

            print(f"Total Trades: {total_trades}")
            print(f"Win Rate: {win_rate:.2%}")

def load_model_and_run_strategy():
    """
    Load trained LSTNet model and run trading strategy
    """
    # Load the model
    try:
        # Model parameters
        window_size = 20
        horizon = 5
        num_variables = 3  # Number of tickers

        # Initialize model
        model = LSTNet(
            num_variables=num_variables,
            window=window_size,
            horizon=horizon,
            CNN_kernel=6,
            RNN_hidden_dim=50,
            CNN_hidden_dim=50,
            skip=0,
            skip_RNN_hidden_dim=10,
            ar_window=24,  # Match the saved model's ar_window
            dropout=0.3,
            output_fun='linear'
        ).to(device)

        # Load trained weights
        model.load_state_dict(torch.load('best_lstnet_model.pth'))
        model.eval()

        print("✅ Successfully loaded trained LSTNet model")

        # Initialize Bloomberg connection
        bloomberg = BloombergDataManager()
        bloomberg.connect()

        # Define agricultural commodity futures tickers with proper Bloomberg syntax
        tickers = [
            'SB1 Comdty',       # ICE eleventh sugar
            'KC1 Comdty',       # ICE coffee
            'CC1 Comdty'        # ICE cocoa
        ]

        # Define date range
        start_date = '2018-01-01'
        end_date = datetime.now().strftime("%Y-%m-%d")

        # Retrieve data from Bloomberg
        data = bloomberg.get_historical_data(tickers, start_date, end_date)

        # Close Bloomberg connection
        bloomberg.close()

        if data is not None and not data.empty:
            # Handle missing values
            data = data.fillna(method='ffill').dropna()

            # Initialize dataset
            dataset = ImprovedTimeSeriesDataset(window_size=window_size, horizon=horizon)

            # Create a dummy dataset to initialize the scalers
            X_train, y_train, X_val, y_val, X_test, y_test = dataset.create_dataset(data)

            # Initialize trading strategy
            strategy = LSTNetTradingStrategy(
                model=model,
                dataset=dataset,
                data=data,
                window_size=window_size,
                horizon=horizon,
                threshold=0.01,  # 1% expected return threshold
                position_size=0.1,
                trade_cost=0.001
            )

            # Run backtest
            print("\n===== Running LSTNet Trading Strategy Backtest =====")
            backtest_results = strategy.backtest()

            return {
                'model': model,
                'dataset': dataset,
                'data': data,
                'strategy': strategy,
                'backtest_results': backtest_results
            }
        else:
            print("❌ Failed to retrieve data. Exiting.")
            return None
    except Exception as e:
        import traceback
        print(f"❌ Error loading model or running strategy: {e}")
        traceback.print_exc()
        return None

if __name__ == "__main__":
    print("Starting LSTNet Trading Strategy...")
    try:
        results = load_model_and_run_strategy()
        print("Strategy execution completed successfully.")
    except Exception as e:
        import traceback
        print(f"Error executing strategy: {e}")
        traceback.print_exc()
