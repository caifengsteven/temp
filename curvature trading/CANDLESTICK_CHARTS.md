# Candlestick Charts with Curved Radius Supertrend

## 📊 Overview

Professional candlestick charts with the Curved Radius Supertrend indicator, featuring:
- **Daily candlestick bars** (OHLC data)
- **Curved trend lines** (green for uptrend, red for downtrend)
- **Trade signals** (entry/exit markers with prices)
- **Dark background** (professional trading platform style)
- **High-resolution output** (200 DPI)

## 🎨 Visual Features

### Candlestick Colors
- **Cyan/Teal** (#00d9d9): Bullish candles (close > open)
- **Magenta** (#d900d9): Bearish candles (close < open)

### Trend Lines
- **Green** (#00ff00): Uptrend (curved support line)
- **Red** (#ff0000): Downtrend (curved resistance line)
- **Gray dotted**: Upper and lower bands (optional)

### Trade Signals
- **Green triangle ▲**: Buy signal (trend change to uptrend)
- **Red triangle ▼**: Sell signal (trend change to downtrend)
- **Price labels**: Show exact entry/exit prices

### Background
- **Pure black** (#000000): Professional dark theme
- **Dark gray grid** (#333333): Subtle grid lines
- **White text**: Clear labels and annotations

## 🚀 Quick Start

### Generate a Chart

```bash
# Simple usage (default: AAPL 2023)
python generate_chart.py

# Specify ticker
python generate_chart.py GOOGL

# Full customization
python generate_chart.py GOOGL 2023-01-01 2023-12-31 0.5
```

### Command Line Arguments

```
python generate_chart.py [TICKER] [START_DATE] [END_DATE] [RADIUS_STRENGTH]
```

- **TICKER**: Stock symbol (e.g., AAPL, GOOGL, MSFT)
- **START_DATE**: Start date in YYYY-MM-DD format
- **END_DATE**: End date in YYYY-MM-DD format
- **RADIUS_STRENGTH**: Curvature parameter (0.1 to 2.0)
  - 0.2 = Tight curves (scalping)
  - 0.5 = Medium curves (swing trading) - **default**
  - 1.0 = Wide curves (position trading)

## 💻 Python API

### Basic Usage

```python
from visualize_candlestick import plot_professional_chart
import matplotlib.pyplot as plt

# Generate chart
fig, ax = plot_professional_chart(
    ticker='AAPL',
    start_date='2023-01-01',
    end_date='2023-12-31',
    radius_strength=0.5
)

# Display
plt.show()
```

### Advanced Usage

```python
from database_connector import StockDataConnector
from visualize_candlestick import plot_candlestick_with_supertrend

# Fetch data
connector = StockDataConnector()
data = connector.fetch_stock_data('AAPL', '2023-01-01', '2023-12-31')
connector.close()

# Create custom chart
fig, ax = plot_candlestick_with_supertrend(
    data=data,
    ticker='AAPL',
    show_trades=True,
    save_path='my_chart.png'
)

plt.show()
```

## 📈 Chart Components

### 1. Candlestick Bars

Each candlestick represents one trading day:
- **Body**: Rectangle from open to close
- **Wick**: Line from low to high
- **Color**: Cyan (bullish) or Magenta (bearish)

### 2. Curved Trend Lines

The indicator calculates curved support/resistance lines:
- **Uptrend** (green): Price above the curved line
- **Downtrend** (red): Price below the curved line
- **Curvature**: Accelerates based on trend duration

### 3. Trade Signals

Markers appear when trend changes:
- **Buy signal**: Green triangle pointing up
- **Sell signal**: Red triangle pointing down
- **Price label**: Shows exact entry/exit price

### 4. Grid and Axes

- **X-axis**: Dates (automatically spaced)
- **Y-axis**: Price levels
- **Grid**: Subtle dark gray lines
- **Background**: Pure black

## 🎯 Examples

### Example 1: AAPL 2023

```bash
python generate_chart.py AAPL 2023-01-01 2023-12-31
```

**Output**: `professional_aapl_2023.png`

### Example 2: GOOGL with Tight Curves

```bash
python generate_chart.py GOOGL 2023-01-01 2023-12-31 0.2
```

**Output**: `professional_googl_2023.png`

### Example 3: Long-term Chart

```bash
python generate_chart.py AAPL 2020-01-01 2023-12-31 1.0
```

**Output**: `professional_aapl_2020.png`

## 🎨 Customization

### Change Colors

Edit `visualize_candlestick.py`:

```python
# Bullish candle color
color = '#00d9d9'  # Change to your preferred color

# Bearish candle color
color = '#d900d9'  # Change to your preferred color

# Uptrend line
ax.plot(..., color='#00ff00', ...)  # Change green

# Downtrend line
ax.plot(..., color='#ff0000', ...)  # Change red
```

### Adjust Figure Size

```python
fig, ax = plt.subplots(figsize=(20, 10))  # Width, Height in inches
```

### Change DPI (Resolution)

```python
plt.savefig(save_path, dpi=200, ...)  # Higher = better quality
```

### Hide Trade Signals

```python
plot_candlestick_with_supertrend(
    data=data,
    ticker='AAPL',
    show_trades=False,  # Hide markers
    save_path='chart.png'
)
```

## 📊 Chart Interpretation

### Reading the Chart

1. **Identify Trend**
   - Green line = Uptrend (consider buying)
   - Red line = Downtrend (consider selling or staying out)

2. **Look for Signals**
   - Green ▲ = Buy signal (trend just turned up)
   - Red ▼ = Sell signal (trend just turned down)

3. **Check Price Action**
   - Price above trend line = Bullish
   - Price below trend line = Bearish
   - Price touching trend line = Support/Resistance

4. **Observe Curvature**
   - Steeper curve = Accelerating trend
   - Flatter curve = Decelerating trend

### Trading Strategy

**Entry Rules:**
- Buy when green ▲ appears (uptrend starts)
- Sell/Short when red ▼ appears (downtrend starts)

**Exit Rules:**
- Exit long when red ▼ appears
- Exit short when green ▲ appears

**Risk Management:**
- Use the curved line as stop-loss level
- Position size based on distance to trend line

## 🔧 Technical Details

### Data Source
- Database: MySQL on NAS (192.168.50.230:3306)
- Tables: Monthly partitioned (YYYYMM format)
- Columns: date, open, high, low, close, volume

### Indicator Parameters
- **ATR Period**: 10 (default)
- **ATR Multiplier**: 3.0 (default)
- **Radius Strength**: 0.5 (default)
- **Smoothness**: 3 (default)

### Chart Specifications
- **Figure Size**: 20 x 10 inches
- **DPI**: 200 (high resolution)
- **Background**: #000000 (pure black)
- **Format**: PNG with transparency

## 📁 Generated Files

When you run the chart generator, it creates:

```
professional_[ticker]_[year].png
```

Examples:
- `professional_aapl_2023.png`
- `professional_googl_2023.png`
- `professional_msft_2023.png`

## 🎓 Tips and Best Practices

### 1. Choose the Right Timeframe
- **Short-term**: 1-3 months for day trading
- **Medium-term**: 6-12 months for swing trading
- **Long-term**: 2-5 years for position trading

### 2. Adjust Radius Strength
- **Volatile stocks**: Use lower values (0.2-0.3)
- **Stable stocks**: Use higher values (0.7-1.0)
- **Test different values**: Find what works best

### 3. Combine with Other Analysis
- Volume analysis
- Support/resistance levels
- Fundamental analysis
- Market conditions

### 4. Backtest Before Trading
- Use the backtesting system
- Test on historical data
- Verify performance metrics

## 🐛 Troubleshooting

### Issue: Chart looks too crowded
**Solution**: Reduce the date range or increase figure size

### Issue: Candles too thin
**Solution**: Reduce the number of days or adjust candle width in code

### Issue: Colors not visible
**Solution**: Check your display settings or adjust colors in code

### Issue: No data found
**Solution**: Verify ticker symbol and date range, check database connection

## 📚 Related Files

- **visualize_candlestick.py**: Main visualization module
- **generate_chart.py**: Quick chart generator script
- **database_connector.py**: Database interface
- **curved_radius_supertrend.py**: Indicator calculation

## 🎯 Next Steps

1. **Generate charts** for your favorite stocks
2. **Compare different** radius_strength values
3. **Analyze patterns** in the curved trend lines
4. **Backtest strategies** using the signals
5. **Customize colors** to your preference

## 📞 Usage Summary

### Quick Commands

```bash
# Default chart (AAPL 2023)
python generate_chart.py

# Custom ticker
python generate_chart.py GOOGL

# Full customization
python generate_chart.py TSLA 2023-01-01 2023-12-31 0.5

# Tight curves for scalping
python generate_chart.py AAPL 2023-01-01 2023-12-31 0.2

# Wide curves for position trading
python generate_chart.py AAPL 2023-01-01 2023-12-31 1.5
```

---

**Note**: These charts are for analysis purposes only. Always do your own research and risk management before trading.

