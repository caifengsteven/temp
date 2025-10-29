# Curved Radius Supertrend - Backtesting Guide

## 📊 Overview

This guide explains how to use the comprehensive backtesting system for the Curved Radius Supertrend strategy. The system connects to a MySQL database containing historical stock data and provides detailed performance analysis.

## 🗄️ Database Configuration

The system connects to a NAS database with the following configuration:

- **Host**: 192.168.50.230:3306
- **Database**: us_stock_sip_day_aggs
- **Structure**: Monthly partitioned tables (YYYYMM format)
- **Data**: OHLC (Open, High, Low, Close) + Volume + Transactions

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

Required packages:
- numpy
- pandas
- matplotlib
- pymysql

### 2. Run Example Backtests

```bash
python backtest_example.py
```

This will run 4 comprehensive examples:
1. Simple backtest for AAPL (2023)
2. Multi-stock comparison
3. Parameter optimization
4. Long-term backtest (2020-2023)

### 3. Run Custom Backtest

```bash
python run_backtest.py
```

## 📁 File Structure

### Core Modules

1. **database_connector.py**
   - Connects to the MySQL database
   - Fetches OHLC data for specified tickers and date ranges
   - Handles monthly table partitioning

2. **backtest_engine.py**
   - Main backtesting engine
   - Simulates trades based on Curved Radius Supertrend signals
   - Calculates performance metrics
   - Tracks equity curve and drawdowns

3. **backtest_visualizer.py**
   - Creates comprehensive visualizations
   - Plots price charts with indicator
   - Shows equity curves and drawdowns
   - Generates comparison charts

4. **run_backtest.py**
   - Main script for running backtests
   - Supports single and multi-stock backtests
   - Parameter optimization functionality

5. **backtest_example.py**
   - Comprehensive examples
   - Demonstrates all features

## 💡 Usage Examples

### Example 1: Simple Single Stock Backtest

```python
from database_connector import StockDataConnector
from backtest_engine import BacktestEngine
from backtest_visualizer import plot_backtest_results

# Fetch data
connector = StockDataConnector()
data = connector.fetch_stock_data(
    ticker='AAPL',
    start_date='2023-01-01',
    end_date='2023-12-31',
    min_volume=1000000
)
connector.close()

# Run backtest
engine = BacktestEngine(
    initial_capital=100000.0,
    commission=0.001,      # 0.1%
    slippage=0.0005,       # 0.05%
    position_size=0.95,    # Use 95% of capital
    allow_short=False
)

results = engine.run_backtest(data, {
    'atr_period': 10,
    'atr_multiplier': 3.0,
    'radius_strength': 0.5,
    'smoothness': 3
})

# Print statistics
stats = results['statistics']
print(f"Total Return: {stats['total_return_pct']:.2f}%")
print(f"Sharpe Ratio: {stats['sharpe_ratio']:.2f}")
print(f"Win Rate: {stats['win_rate']:.2f}%")

# Visualize
plot_backtest_results(results, ticker='AAPL', save_path='backtest.png')
```

### Example 2: Compare Multiple Stocks

```python
from run_backtest import run_multiple_stocks_backtest

results = run_multiple_stocks_backtest(
    tickers=['AAPL', 'MSFT', 'GOOGL', 'TSLA'],
    start_date='2023-01-01',
    end_date='2023-12-31',
    indicator_params={
        'atr_period': 10,
        'atr_multiplier': 3.0,
        'radius_strength': 0.5,
        'smoothness': 3
    },
    initial_capital=100000.0
)
```

### Example 3: Parameter Optimization

```python
from run_backtest import parameter_optimization

results = parameter_optimization(
    ticker='AAPL',
    start_date='2023-01-01',
    end_date='2023-12-31',
    param_grid={
        'radius_strength': [0.2, 0.5, 1.0, 1.5, 2.0],
        'atr_period': [10, 14, 20],
        'atr_multiplier': [2.0, 3.0, 4.0]
    }
)
```

## 📈 Performance Metrics

The backtesting system calculates the following metrics:

### Trade Statistics
- **Total Trades**: Number of completed trades
- **Winning Trades**: Number of profitable trades
- **Losing Trades**: Number of losing trades
- **Win Rate**: Percentage of winning trades
- **Average Bars Held**: Average holding period in days

### Performance Metrics
- **Total Return**: Overall percentage return
- **Total P&L**: Total profit/loss in dollars
- **Average P&L per Trade**: Mean profit/loss per trade
- **Average Win**: Mean profit of winning trades
- **Average Loss**: Mean loss of losing trades
- **Profit Factor**: Ratio of gross profit to gross loss

### Risk Metrics
- **Sharpe Ratio**: Risk-adjusted return (annualized)
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Final Equity**: Ending portfolio value

## 🎯 Strategy Parameters

### Indicator Parameters

1. **atr_period** (default: 10)
   - Period for ATR calculation
   - Lower values = more responsive
   - Higher values = smoother

2. **atr_multiplier** (default: 3.0)
   - Multiplier for ATR bands
   - Lower values = tighter bands, more trades
   - Higher values = wider bands, fewer trades

3. **radius_strength** (default: 0.5)
   - Controls curvature acceleration
   - Lower values (0.2-0.5): Scalping/day trading
   - Medium values (0.5-1.0): Swing trading
   - Higher values (1.0-2.0): Position trading

4. **smoothness** (default: 3)
   - SMA smoothing period
   - Reduces noise in the curved bands

### Backtesting Parameters

1. **initial_capital** (default: 100,000)
   - Starting capital in dollars

2. **commission** (default: 0.001)
   - Commission rate per trade (0.1%)

3. **slippage** (default: 0.0005)
   - Slippage rate (0.05%)

4. **position_size** (default: 0.95)
   - Fraction of capital to use per trade (95%)

5. **allow_short** (default: False)
   - Whether to allow short positions

## 📊 Visualization Outputs

The system generates comprehensive visualizations:

### 1. Backtest Results Chart
- Price chart with curved bands
- Trade entry/exit markers
- Equity curve
- Drawdown chart
- Trade return distribution
- Statistics summary

### 2. Multi-Stock Comparison
- Total returns comparison
- Sharpe ratios comparison
- Maximum drawdowns comparison
- Win rates comparison

### 3. Parameter Optimization Charts
- Return vs parameter value
- Sharpe ratio vs parameter value
- Max drawdown vs parameter value
- Win rate vs parameter value

## 🔍 Understanding Results

### Good Performance Indicators
- ✅ Win rate > 50%
- ✅ Sharpe ratio > 1.0
- ✅ Profit factor > 1.5
- ✅ Max drawdown < 20%
- ✅ Positive total return

### Warning Signs
- ⚠️ Win rate < 40%
- ⚠️ Sharpe ratio < 0.5
- ⚠️ Profit factor < 1.0
- ⚠️ Max drawdown > 30%
- ⚠️ Very few trades (< 10)

## 🛠️ Customization

### Custom Trading Logic

You can modify the backtesting logic in `backtest_engine.py`:

```python
# Example: Add stop-loss
if self.current_position is not None:
    if self.current_position.direction == 'LONG':
        stop_loss = self.current_position.entry_price * 0.95
        if price < stop_loss:
            # Close position
            ...
```

### Custom Metrics

Add custom metrics in the `calculate_statistics` method:

```python
# Example: Calculate Sortino ratio
downside_returns = equity_df[equity_df['returns'] < 0]['returns']
sortino_ratio = np.sqrt(252) * equity_df['returns'].mean() / downside_returns.std()
```

## 📝 Best Practices

1. **Data Quality**
   - Ensure sufficient data (minimum 50 trading days)
   - Filter by minimum volume to avoid illiquid stocks
   - Check for data gaps

2. **Parameter Selection**
   - Start with default parameters
   - Use parameter optimization to find best values
   - Avoid overfitting (test on out-of-sample data)

3. **Risk Management**
   - Don't use 100% of capital per trade
   - Consider transaction costs
   - Monitor maximum drawdown

4. **Validation**
   - Test on multiple stocks
   - Test on different time periods
   - Compare with buy-and-hold strategy

## 🐛 Troubleshooting

### Issue: No data found
**Solution**: Check ticker symbol and date range. Ensure database connection is working.

### Issue: Insufficient data error
**Solution**: Increase date range or choose a more liquid stock.

### Issue: No trades executed
**Solution**: Adjust indicator parameters (try lower atr_multiplier or different radius_strength).

### Issue: Poor performance
**Solution**: 
- Try parameter optimization
- Test different time periods
- Consider market conditions (trending vs ranging)

## 📚 Additional Resources

- **Indicator Theory**: See `实现说明.md` for detailed explanation
- **Quick Start**: See `QUICKSTART.md` for basic usage
- **Code Examples**: See `backtest_example.py` for comprehensive examples

## 🎓 Next Steps

1. Run the example backtests to understand the system
2. Test on your favorite stocks
3. Optimize parameters for your trading style
4. Combine with other indicators or filters
5. Implement additional risk management rules

---

**Note**: Past performance does not guarantee future results. Always test thoroughly before live trading.

