# LightGBM Trading Strategy

This project implements a trading strategy based on the research paper "Assets Forecasting with Feature Engineering and Transformation Methods for LightGBM". It uses Bloomberg data and LightGBM to predict stock prices and generate trading signals.

## Features

- Fetches historical OHLCV data from Bloomberg
- Implements extensive feature engineering techniques from the paper
- Supports multiple target variable transformations:
  - Returns
  - Logarithmic returns
  - EMA ratios
  - Standardized versions
  - EMA difference ratios
- Trains a LightGBM model for price prediction
- Backtests the trading strategy with customizable parameters
- Visualizes results with various plots

## Requirements

- Python 3.7+
- Bloomberg Terminal with the Bloomberg API installed
- Required Python packages (see `requirements.txt`)

## Installation

1. Clone this repository
2. Install the required packages:

```bash
pip install -r requirements.txt
```

3. Ensure you have access to Bloomberg API and the Bloomberg Terminal is running

## Usage

Run the main script with your desired parameters:

```bash
python main.py --ticker "AAPL US Equity" --start-date 20200101 --end-date 20230101 --target-transform log_returns --plot
```

### Command Line Arguments

- `--ticker`: Bloomberg ticker symbol (default: "AAPL US Equity")
- `--start-date`: Start date for data in format YYYYMMDD (default: 20180101)
- `--end-date`: End date for data in format YYYYMMDD (default: today)
- `--target-transform`: Transformation method for the target variable (default: log_returns)
  - Options: returns, log_returns, ema_ratio, standardized_returns, standardized_log_returns, standardized_ema_ratio, ema_difference_ratio
- `--test-size`: Proportion of data to use for testing (default: 0.2)
- `--initial-capital`: Initial capital for the strategy (default: 100000.0)
- `--position-size`: Proportion of capital to allocate to each position (default: 0.1)
- `--stop-loss`: Stop loss percentage (default: 0.02)
- `--take-profit`: Take profit percentage (default: 0.05)
- `--signal-threshold`: Threshold for signal generation (default: 0.0)
- `--commission`: Commission rate per trade (default: 0.001)
- `--random-state`: Random seed for reproducibility (default: 42)
- `--plot`: Plot results (flag)

## Project Structure

- `bloomberg_data.py`: Module for fetching data from Bloomberg
- `feature_engineering.py`: Module for creating features and transforming data
- `lightgbm_model.py`: Module for training and using the LightGBM model
- `strategy.py`: Module for executing and backtesting the trading strategy
- `main.py`: Main script to tie everything together

## Example

```python
from bloomberg_data import BloombergDataFetcher
from feature_engineering import FeatureEngineer
from lightgbm_model import LightGBMForecaster
from strategy import TradingStrategy

# Fetch data
fetcher = BloombergDataFetcher()
data = fetcher.fetch_historical_data('AAPL US Equity', '20200101', '20230101')

# Create features
engineer = FeatureEngineer(data)
featured_data = engineer.create_all_features()

# Train model
forecaster = LightGBMForecaster(target_transform='log_returns')
results = forecaster.train(featured_data, test_size=0.2, verbose=True)

# Generate predictions
predictions = forecaster.predict(featured_data)

# Backtest strategy
strategy = TradingStrategy(initial_capital=100000.0)
signals = strategy.generate_signals(predictions, featured_data['close'])
backtest_results = strategy.backtest(signals)
strategy.print_performance_summary(backtest_results)
```

## Notes

- The Bloomberg connection requires a running Bloomberg Terminal
- If Bloomberg is not available, the system will use sample data for testing
- For optimal performance, ensure you have enough historical data (at least 2 years recommended)
- The feature engineering is based on the paper but can be customized for specific assets

## References

- "Assets Forecasting with Feature Engineering and Transformation Methods for LightGBM" by Konstantinos-Leonidas Bisdoulis
- LightGBM documentation: https://lightgbm.readthedocs.io/
- Bloomberg API documentation: https://www.bloomberg.com/professional/support/api-library/
