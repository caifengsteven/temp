"""
Main Script for Trading Strategy

This script ties together all components to implement a trading strategy
based on the research paper "Assets Forecasting with Feature Engineering and Transformation Methods for LightGBM".
Due to dependency issues with LightGBM, we're using scikit-learn's GradientBoostingRegressor as a fallback.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import datetime as dt

from bloomberg_data import BloombergDataFetcher
from feature_engineering import FeatureEngineer, DataTransformer
# Use scikit-learn model to avoid dependency issues with LightGBM
from sklearn_model import SklearnForecaster
from strategy import TradingStrategy


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='LightGBM Trading Strategy')

    parser.add_argument('--ticker', type=str, default='AAPL US Equity',
                        help='Bloomberg ticker symbol')
    parser.add_argument('--start-date', type=str, default='20180101',
                        help='Start date for data in format YYYYMMDD')
    parser.add_argument('--end-date', type=str, default=None,
                        help='End date for data in format YYYYMMDD')
    parser.add_argument('--target-transform', type=str, default='log_returns',
                        choices=['returns', 'log_returns', 'ema_ratio', 'standardized_returns',
                                'standardized_log_returns', 'standardized_ema_ratio', 'ema_difference_ratio'],
                        help='Transformation method for the target variable')
    parser.add_argument('--test-size', type=float, default=0.2,
                        help='Proportion of data to use for testing')
    parser.add_argument('--initial-capital', type=float, default=100000.0,
                        help='Initial capital for the strategy')
    parser.add_argument('--position-size', type=float, default=0.95,
                        help='Proportion of capital to allocate to each position')
    parser.add_argument('--stop-loss', type=float, default=0.1,
                        help='Stop loss percentage')
    parser.add_argument('--take-profit', type=float, default=0.2,
                        help='Take profit percentage')
    parser.add_argument('--signal-threshold', type=float, default=0.002,
                        help='Base threshold for signal generation')
    parser.add_argument('--trend-window', type=int, default=10,
                        help='Window size for trend calculation')
    parser.add_argument('--trend-threshold', type=float, default=0.0,
                        help='Threshold for trend determination (not used in current implementation)')
    parser.add_argument('--volatility-window', type=int, default=30,
                        help='Window size for volatility calculation')
    parser.add_argument('--commission', type=float, default=0.001,
                        help='Commission rate per trade')
    parser.add_argument('--random-state', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--plot', action='store_true',
                        help='Plot results')

    return parser.parse_args()


def run_backtest(ticker, start_date, end_date, args):
    """
    Run the backtest for a given ticker.

    Parameters:
    -----------
    ticker : str
        Bloomberg ticker symbol
    start_date : str
        Start date in format 'YYYYMMDD'
    end_date : str
        End date in format 'YYYYMMDD'
    args : argparse.Namespace
        Command line arguments

    Returns:
    --------
    dict
        Dictionary containing backtest results
    """
    print(f"\n{'='*50}")
    print(f"Running backtest for {ticker} from {start_date} to {end_date}...")
    print(f"{'='*50}")

    # Try to fetch data from Bloomberg
    try:
        fetcher = BloombergDataFetcher()
        data = fetcher.fetch_historical_data(ticker, start_date, end_date)

        if data is None or len(data) == 0:
            raise Exception("No data fetched from Bloomberg")

        print(f"Data fetched successfully. Shape: {data.shape}")

        # Process Bloomberg data
        print(f"Data columns: {data.columns.tolist()}")

        # Check if we need to process multi-level columns
        if isinstance(data.columns, pd.MultiIndex):
            print("Processing multi-level columns from Bloomberg data...")

            # Create a new DataFrame with standard column names
            processed_data = pd.DataFrame(index=data.index)

            # Map Bloomberg field names to our standard names
            field_map = {
                'PX_OPEN': 'open',
                'PX_HIGH': 'high',
                'PX_LOW': 'low',
                'PX_LAST': 'close',
                'PX_VOLUME': 'volume',
                'open': 'open',
                'high': 'high',
                'low': 'low',
                'close': 'close',
                'volume': 'volume'
            }

            # Try different approaches to extract the data
            try:
                # First approach: If columns are (ticker, field)
                if len(data.columns.levels) == 2:
                    ticker_name = data.columns.levels[0][0]
                    field_level_name = data.columns.names[1]

                    # Check if the second level has standard names
                    if field_level_name == 'field' and all(field in ['open', 'high', 'low', 'close', 'volume']
                                                          for field in data.columns.levels[1]):
                        for field in ['open', 'high', 'low', 'close', 'volume']:
                            if (ticker_name, field) in data.columns:
                                processed_data[field] = data[(ticker_name, field)]
                    else:
                        # Try with Bloomberg field names
                        for bbg_field, std_field in field_map.items():
                            if (ticker_name, bbg_field) in data.columns:
                                processed_data[std_field] = data[(ticker_name, bbg_field)]

                # Second approach: If columns are just the fields
                if processed_data.empty and len(data.columns) == 5:  # Assuming 5 columns for OHLCV
                    for i, (bbg_field, std_field) in enumerate(field_map.items()):
                        if bbg_field in data.columns:
                            processed_data[std_field] = data[bbg_field]

                # Third approach: Try to infer column mapping
                if processed_data.empty:
                    # Map columns by position
                    col_mapping = {
                        0: 'open',
                        1: 'high',
                        2: 'low',
                        3: 'close',
                        4: 'volume'
                    }
                    for i, col in enumerate(data.columns):
                        if i in col_mapping:
                            processed_data[col_mapping[i]] = data[col]

                data = processed_data

                print(f"Processed data columns: {processed_data.columns.tolist()}")
                print(f"Processed data shape: {processed_data.shape}")

            except Exception as e:
                print(f"Error processing Bloomberg data structure: {e}")
                # If all approaches fail, print the first few rows of the original data
                print("Original data sample:")
                print(data.head())

        # Check if the data already has the standard column names
        elif all(col in data.columns for col in ['open', 'high', 'low', 'close', 'volume']):
            print("Data already has standard column names.")

        # If not, try to map the columns
        else:
            print("Trying to map columns to standard names...")
            processed_data = pd.DataFrame(index=data.index)

            # Map by common names
            name_map = {
                'PX_OPEN': 'open', 'OPEN': 'open', 'Open': 'open',
                'PX_HIGH': 'high', 'HIGH': 'high', 'High': 'high',
                'PX_LOW': 'low', 'LOW': 'low', 'Low': 'low',
                'PX_LAST': 'close', 'CLOSE': 'close', 'Close': 'close', 'PX_CLOSE': 'close',
                'PX_VOLUME': 'volume', 'VOLUME': 'volume', 'Volume': 'volume'
            }

            for col in data.columns:
                if col in name_map:
                    processed_data[name_map[col]] = data[col]

            # If we couldn't map all columns, try by position
            if len(processed_data.columns) < 5 and len(data.columns) >= 5:
                col_mapping = {
                    0: 'open',
                    1: 'high',
                    2: 'low',
                    3: 'close',
                    4: 'volume'
                }
                for i, col in enumerate(data.columns):
                    if i in col_mapping and col_mapping[i] not in processed_data.columns:
                        processed_data[col_mapping[i]] = data[col]

            data = processed_data
            print(f"Mapped data columns: {data.columns.tolist()}")
            print(f"Mapped data shape: {data.shape}")

        # Check if we have all required columns
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in data.columns for col in required_columns):
            raise Exception(f"Missing required columns. Found: {data.columns.tolist()}")

    except Exception as e:
        print(f"Error with Bloomberg data: {e}")
        print("Using sample data instead...")

        # Generate sample data
        start = dt.datetime.strptime(start_date, '%Y%m%d')
        end = dt.datetime.strptime(end_date or dt.datetime.now().strftime('%Y%m%d'), '%Y%m%d')

        # Generate date range
        date_range = pd.date_range(start=start, end=end, freq='B')  # Business days

        # Create sample data
        np.random.seed(42)  # For reproducibility
        n = len(date_range)

        # Start with a base price and add random walks
        base_price = 100
        daily_returns = np.random.normal(0.0005, 0.015, n)
        prices = base_price * (1 + np.cumsum(daily_returns))

        # Generate OHLCV data
        data = pd.DataFrame({
            'open': prices * (1 + np.random.normal(0, 0.005, n)),
            'high': prices * (1 + np.abs(np.random.normal(0, 0.01, n))),
            'low': prices * (1 - np.abs(np.random.normal(0, 0.01, n))),
            'close': prices,
            'volume': np.random.lognormal(15, 1, n).astype(int)
        }, index=date_range)

        # Ensure high >= open, close, low and low <= open, close
        data['high'] = data[['high', 'open', 'close']].max(axis=1)
        data['low'] = data[['low', 'open', 'close']].min(axis=1)

    print(f"Final data shape: {data.shape}")
    print(f"Columns: {data.columns.tolist()}")

    # Create features
    print("Creating features...")
    engineer = FeatureEngineer(data)
    featured_data = engineer.create_all_features()

    print(f"Features created. Shape: {featured_data.shape}")

    # Train the model
    print(f"Training model with {args.target_transform} transformation...")
    forecaster = SklearnForecaster(target_transform=args.target_transform)
    results = forecaster.train(
        featured_data,
        test_size=args.test_size,
        random_state=args.random_state,
        verbose=True
    )

    print("Model training completed.")

    # Generate predictions
    print("Generating predictions...")
    predictions = forecaster.predict(featured_data)

    # Create signals and backtest the strategy
    print("Backtesting the strategy...")
    strategy = TradingStrategy(
        initial_capital=args.initial_capital,
        position_size=args.position_size,
        stop_loss=args.stop_loss,
        take_profit=args.take_profit
    )

    signals = strategy.generate_signals(
        predictions,
        featured_data['close'],
        threshold=args.signal_threshold,
        trend_window=args.trend_window,
        trend_threshold=args.trend_threshold,
        volatility_window=args.volatility_window
    )

    backtest_results = strategy.backtest(signals, commission=args.commission)

    # Print performance summary
    strategy.print_performance_summary(backtest_results)

    # Calculate buy-and-hold performance for comparison
    print("\n=== Buy and Hold Strategy Comparison ===")

    # Get the first and last close prices
    first_close = data['close'].iloc[0]
    last_close = data['close'].iloc[-1]

    # Calculate total return
    buy_hold_return = (last_close / first_close - 1) * 100

    # Calculate annualized return
    days = (data.index[-1] - data.index[0]).days
    years = days / 365.25
    annualized_return = ((1 + buy_hold_return/100) ** (1/years) - 1) * 100

    # Calculate max drawdown
    rolling_max = data['close'].cummax()
    drawdown = (data['close'] / rolling_max - 1) * 100
    max_drawdown = drawdown.min()

    # Print buy-and-hold results
    print(f"Buy & Hold Total Return: {buy_hold_return:.2f}%")
    print(f"Buy & Hold Annualized Return: {annualized_return:.2f}%")
    print(f"Buy & Hold Max Drawdown: {max_drawdown:.2f}%")

    # Compare with strategy
    strategy_return = backtest_results['total_return'] * 100
    strategy_max_drawdown = backtest_results['max_drawdown'] * 100

    print(f"\nStrategy vs Buy & Hold:")
    print(f"Total Return: {strategy_return:.2f}% vs {buy_hold_return:.2f}%")
    print(f"Max Drawdown: {strategy_max_drawdown:.2f}% vs {max_drawdown:.2f}%")
    print(f"Return Ratio (Strategy/Buy&Hold): {strategy_return/buy_hold_return if buy_hold_return != 0 else 'N/A':.2f}")

    # Return results for comparison
    return {
        'ticker': ticker,
        'strategy_return': strategy_return,
        'strategy_max_drawdown': strategy_max_drawdown,
        'buy_hold_return': buy_hold_return,
        'buy_hold_max_drawdown': max_drawdown,
        'return_ratio': strategy_return/buy_hold_return if buy_hold_return != 0 else float('nan'),
        'num_trades': backtest_results['num_trades'],
        'win_rate': backtest_results['win_rate'] * 100
    }


def main():
    """Main function to run the strategy."""
    # Parse arguments
    args = parse_arguments()

    # Define a list of tickers to test
    tickers = [
        'AAPL US Equity',  # Apple
        'MSFT US Equity',  # Microsoft
        'AMZN US Equity',  # Amazon
        'GOOGL US Equity', # Google
        'META US Equity',  # Meta (Facebook)
        'TSLA US Equity',  # Tesla
        'NVDA US Equity',  # NVIDIA
        'JPM US Equity',   # JPMorgan Chase
        'JNJ US Equity',   # Johnson & Johnson
        'V US Equity'      # Visa
    ]

    # Run backtest for each ticker
    results = []

    if args.ticker != 'AAPL US Equity':
        # If a specific ticker is provided, only test that one
        results.append(run_backtest(args.ticker, args.start_date, args.end_date, args))
    else:
        # Otherwise, test all tickers
        for ticker in tickers:
            try:
                result = run_backtest(ticker, args.start_date, args.end_date, args)
                results.append(result)
            except Exception as e:
                print(f"Error running backtest for {ticker}: {e}")

    # Print summary of results
    print("\n\n" + "="*80)
    print("SUMMARY OF RESULTS")
    print("="*80)
    print(f"{'Ticker':<15} {'Strategy Return':<15} {'Buy & Hold':<15} {'Return Ratio':<15} {'Strategy DD':<15} {'B&H DD':<15} {'Trades':<10} {'Win Rate':<10}")
    print("-"*80)

    for result in results:
        print(f"{result['ticker']:<15} {result['strategy_return']:>6.2f}% {result['buy_hold_return']:>13.2f}% {result['return_ratio']:>13.2f} {result['strategy_max_drawdown']:>13.2f}% {result['buy_hold_max_drawdown']:>10.2f}% {result['num_trades']:>8} {result['win_rate']:>8.2f}%")

    print("="*80)

    # Calculate average performance
    avg_strategy_return = sum(r['strategy_return'] for r in results) / len(results)
    avg_buy_hold_return = sum(r['buy_hold_return'] for r in results) / len(results)
    avg_return_ratio = sum(r['return_ratio'] for r in results if not np.isnan(r['return_ratio'])) / len([r for r in results if not np.isnan(r['return_ratio'])])
    avg_strategy_dd = sum(r['strategy_max_drawdown'] for r in results) / len(results)
    avg_buy_hold_dd = sum(r['buy_hold_max_drawdown'] for r in results) / len(results)
    avg_num_trades = sum(r['num_trades'] for r in results) / len(results)
    avg_win_rate = sum(r['win_rate'] for r in results) / len(results)

    print(f"{'AVERAGE':<15} {avg_strategy_return:>6.2f}% {avg_buy_hold_return:>13.2f}% {avg_return_ratio:>13.2f} {avg_strategy_dd:>13.2f}% {avg_buy_hold_dd:>10.2f}% {avg_num_trades:>8.1f} {avg_win_rate:>8.2f}%")
    print("="*80)

    print("\nDone!")


if __name__ == "__main__":
    main()
