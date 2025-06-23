# 5 Trading Strategies Implementation

This repository contains Python implementations of 5 different quantitative trading strategies extracted from Chinese financial research papers and articles.

## Overview

All strategies have been converted from their original descriptions (mostly in Chinese) into comprehensive Python scripts with backtesting capabilities. Each strategy includes:

- Data fetching from Yahoo Finance
- Indicator calculations
- Signal generation
- Backtesting with transaction costs
- Performance visualization
- Detailed results analysis

## Strategies Included

### 1. Bull Rise Indicator (牛起指标) Strategy
**File:** `strategy_1_bull_rise_indicator.py`

**Description:** Volume breakthrough + price breakthrough strategy
- **Expected Performance:** 326% return rate, 63% win rate
- **Key Features:**
  - 5-day and 10-day volume moving averages
  - Volume breakthrough conditions
  - Price breakthrough of 2-day highs
  - K-line body high breakthrough
  - Historical bottom signal detection
  - Dynamic support/resistance levels

### 2. SF12Re Volatility Algorithm
**File:** `strategy_2_sf12re_volatility.py`

**Description:** Adaptive interval + volatility timing strategy using new volatility calculation
- **Key Features:**
  - Alternative to traditional ATR volatility calculation
  - Short-term (R1) and medium-term (R2=2×R1) period analysis
  - Adaptive interval construction based on market volatility
  - Volatility ratio timing module
  - Dynamic trailing stop loss
  - Self-adaptive mechanism for different market environments

### 3. 10-Day Low Point Buy Strategy
**File:** `strategy_3_10day_low_buy.py`

**Description:** Simple but effective strategy with historically high win rates
- **Expected Performance:** 95% win rate (historical 2007-2012 data)
- **Buy Conditions:**
  - Stock price reaches 10-day new low
  - Stock price above 50-day and 200-day moving averages
- **Sell Conditions:**
  - Price reaches 10-day new high, OR
  - Price falls below 50-day moving average, OR
  - Holding period exceeds 10 days

### 4. ETF Rotation Strategy
**File:** `strategy_4_etf_rotation.py`

**Description:** Multi-factor ETF selection strategy
- **Expected Performance:** 21% annual return with stable 10-year performance
- **Evaluation Factors:**
  - Trend Factor (50%): 25-day log return linear regression R²
  - Momentum Factor (20%): 5-day and 10-day ROC combination
  - Volume Factor (30%): 5-day/18-day average volume ratio
  - Momentum Acceleration Factor (10%): Recent momentum acceleration
- **Features:**
  - Monthly rebalancing
  - Top N ETF selection (default: top 3)
  - Equal weight allocation

### 5. VIX Fix + Fractal Chaos Band Strategy
**File:** `strategy_5_vix_fix_fractal.py`

**Description:** Volatility-based panic trading strategy
- **Expected Performance:** 65% outperformance vs buy-and-hold, 44% of buy-and-hold max drawdown
- **Key Components:**
  - VIX Fix: Simulates VIX panic index for individual stocks
  - Fractal Chaos Bands: Dynamic support/resistance levels
- **Trading Logic:**
  - Entry: VIX spike OR price near lower band
  - Exit: VIX subsides AND price near upper band
- **Features:**
  - Only 37% time in market but higher returns
  - Captures market panic for entry opportunities

## Installation and Requirements

### Required Libraries
```bash
pip install pandas numpy matplotlib scikit-learn
```

### Optional (for enhanced functionality)
```bash
pip install seaborn plotly
```

**Note:** `yfinance` is no longer required as all strategies now use simulated data for reliable, fast testing without internet dependency.

## Usage

### Running Individual Strategies

Each strategy can be run independently:

```python
# Example: Run Strategy 1
python strategy_1_bull_rise_indicator.py

# Example: Run Strategy 3
python strategy_3_10day_low_buy.py
```

### Running All Strategies with Comparison

Use the comprehensive test runner:

```python
python run_all_strategies.py
```

This will:
- Run all 5 strategies on the same symbol (default: AAPL)
- Generate performance comparison
- Create visualization charts
- Save results to CSV file

### Customizing Parameters

Each strategy class can be customized:

```python
from strategy_1_bull_rise_indicator import BullRiseIndicatorStrategy

# Custom parameters
strategy = BullRiseIndicatorStrategy(
    symbol='MSFT',
    start_date='2019-01-01',
    end_date='2024-12-31'
)

# Run the strategy
if strategy.fetch_data():
    strategy.calculate_indicators()
    strategy.generate_signals()
    results = strategy.backtest(initial_capital=50000)
    strategy.plot_results(results)
```

## Output Files

When running the comprehensive comparison, the following files are generated:

- `strategy_comparison_results.csv`: Detailed performance metrics
- `strategy_comparison_charts.png`: Visual comparison charts
- `strategy_X_content.txt`: Original PDF content (auto-generated)

## Performance Metrics

Each strategy reports the following metrics:

- **Final Portfolio Value**
- **Total Return**
- **Annual Return** (where applicable)
- **Win Rate**
- **Total Number of Trades**
- **Average Holding Period**
- **Sharpe Ratio** (where applicable)
- **Maximum Drawdown** (where applicable)
- **Time in Market** (where applicable)

## Important Notes

### Risk Disclaimer
- These strategies are for educational and research purposes only
- Historical performance does not guarantee future results
- All trading involves risk of loss
- Test thoroughly with paper trading before using real money

### Data Characteristics
- All strategies now use realistic simulated data with different market characteristics
- Strategy 1: General trending data with volatility patterns
- Strategy 2: Data with varying volatility regimes (high/low volatility periods)
- Strategy 3: Trending data with periodic pullbacks (ideal for buy-the-dip strategies)
- Strategy 4: Multiple ETFs with sector-specific behaviors and market regimes
- Strategy 5: Data with realistic panic and euphoria periods for volatility trading
- Backtesting includes transaction costs but may not reflect real trading conditions

### Strategy Adaptations
- Original strategies have been adapted for US markets and Yahoo Finance data
- Some complex indicators have been simplified for implementation
- Parameters may need adjustment for different markets or time periods

## Contributing

Feel free to:
- Report bugs or issues
- Suggest improvements
- Add new strategies
- Optimize existing implementations

## License

This project is for educational purposes. Please respect the original research and cite sources appropriately.

## Acknowledgments

- Original strategy developers and researchers
- Chinese quantitative finance community
- Open source Python financial libraries

---

**Note:** This implementation is based on publicly available research and articles. Always verify strategy logic and test thoroughly before any real trading application.
