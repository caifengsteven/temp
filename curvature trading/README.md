# Curved Radius Supertrend (曲率半径超级趋势)

A sophisticated trend-following indicator that models market dynamics using curvature acceleration instead of simple linear bands. This implementation combines mathematical curvature dynamics with adaptive volatility processing.

## 📊 Theoretical Foundation (理论基础)

### The Problem with Standard Supertrend

Traditional Supertrend indicators use **linear ATR (Average True Range) envelopes** that move proportionally with price deviation. However, markets don't expand or contract linearly. Trend velocity typically accelerates and decelerates in **non-linear arcs**, forming natural parabolic patterns in price phases.

### The Curved Radius Solution

By embedding a **radius-based acceleration function**, this indicator models natural market behavior. The core variable `radiusStrength` controls how aggressively curvature accelerates over time. Instead of simply following price distance, the bands evolve based on **time-based acceleration** — each bar contributes incremental velocity, bending the trend line into radius-shaped curves.

This design enables the indicator to **anticipate** rather than just react to price movements, capturing momentum transitions as curved acceleration rather than binary flips.

### Key Benefits

✅ **Eliminates lag** typical of standard Supertrend  
✅ **Fluid directional movement** that reflects actual trend geometry  
✅ **Predictive capability** through acceleration modeling  
✅ **Adaptive to different timeframes** via radius strength parameter  

---

## 🔧 How It Works (实现方式)

The Curved Radius Supertrend is constructed through a multi-stage process:

### 1. Baseline Supertrend Core (基线超级趋势核心)

The framework starts with standard ATR-derived upper and lower band calculations. These define the volatility envelope that bounds potential price zones.

- **Upper Band** = HL_Average + (ATR × Multiplier)
- **Lower Band** = HL_Average - (ATR × Multiplier)

Direction bias is determined through crossover logic:
- Price above lower band → Uptrend confirmed
- Price below upper band → Downtrend confirmed

### 2. Curvature Acceleration Engine (曲率加速引擎)

Once trend direction is established, the curvature engine activates. This system uses `radiusStrength` as a coefficient to simulate acceleration per bar, progressively increasing velocity over time.

**Mathematical Model:**
```
displacement = 0.5 × radiusStrength × (bars_in_trend)² / 100
```

This creates **parabolic displacement** of the anchor price (price level at trend change), forming a curved motion path that dynamically widens or tightens as the trend matures.

The acceleration is **quadratic** — each new bar compounds previous velocity, creating an exponential displacement rate similar to curved inertia.

### 3. Adaptive Smoothing Layer (自适应平滑层)

After applying the radius curve, a smoothing phase (defined by the `smoothness` parameter) uses a Simple Moving Average to temper curve noise. This ensures visual coherence without sacrificing responsiveness, producing smooth arcs instead of jagged band steps.

---

## 📈 Usage Guide (使用方法)

### Installation

```bash
# Clone or download the repository
git clone <repository-url>
cd curvature-trading

# Install dependencies
pip install numpy pandas matplotlib
```

### Basic Usage

```python
from curved_radius_supertrend import CurvedRadiusSupertrend
import numpy as np

# Prepare your OHLC data
high = np.array([...])   # High prices
low = np.array([...])    # Low prices
close = np.array([...])  # Close prices

# Create indicator instance
indicator = CurvedRadiusSupertrend(
    atr_period=10,          # ATR calculation period
    atr_multiplier=3.0,     # ATR band multiplier
    radius_strength=0.5,    # Curvature acceleration strength
    smoothness=3            # Smoothing period
)

# Calculate indicator
result = indicator.calculate(high, low, close)

# Access results
print(result['direction'].iloc[-1])    # Current trend: 1=up, -1=down
print(result['trend_line'].iloc[-1])   # Current trend line value
```

### Visualization

```python
# Run the visualization demo
python visualize_indicator.py
```

This will generate a comprehensive comparison of different parameter settings and save the plot as `curved_supertrend_comparison.png`.

---

## 🎯 Interpreting the Indicator

### Uptrend Phase (上升趋势阶段)
- Band curves **upward** with increasing acceleration
- Reflects growing market directional velocity
- **Steeper curvature** = stronger conviction

### Downtrend Phase (下降趋势阶段)
- Band curves **downward** in mirror acceleration pattern
- Indicates sustained bearish momentum
- Acceleration compounds over time

### Trend Change Points (趋势变化点)
- Direction flips and forms new anchor point
- Curve **resets** — providing clean, early visual confirmation of structural reversal
- Marked with blue circles in visualization

### Parameter Interaction (平滑和半径相互作用)

**Lower `radius_strength`** (0.1 - 0.3):
- Tighter, more reactive curves
- Ideal for **scalping** or short timeframes
- More frequent trend changes

**Higher `radius_strength`** (1.0 - 2.0):
- Wide, sweeping arcs
- Optimized for **swing or position analysis**
- Fewer, more significant trend changes

**Smoothness** (1 - 10):
- Lower values: More responsive, noisier
- Higher values: Smoother curves, slight lag

---

## 📊 Parameter Reference

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `atr_period` | 10 | 5-50 | Period for ATR calculation |
| `atr_multiplier` | 3.0 | 1.0-5.0 | Multiplier for volatility bands |
| `radius_strength` | 0.5 | 0.1-3.0 | Curvature acceleration coefficient |
| `smoothness` | 3 | 1-20 | SMA smoothing period |

### Recommended Configurations

**Scalping (1-5 min charts):**
```python
CurvedRadiusSupertrend(
    atr_period=7,
    atr_multiplier=2.0,
    radius_strength=0.2,
    smoothness=2
)
```

**Day Trading (15-60 min charts):**
```python
CurvedRadiusSupertrend(
    atr_period=10,
    atr_multiplier=3.0,
    radius_strength=0.5,
    smoothness=3
)
```

**Swing Trading (4H-Daily charts):**
```python
CurvedRadiusSupertrend(
    atr_period=14,
    atr_multiplier=3.5,
    radius_strength=1.0,
    smoothness=5
)
```

**Position Trading (Daily-Weekly charts):**
```python
CurvedRadiusSupertrend(
    atr_period=20,
    atr_multiplier=4.0,
    radius_strength=2.0,
    smoothness=7
)
```

---

## 🔬 Technical Details

### Output DataFrame Columns

- `curved_upper`: Upper curved volatility band
- `curved_lower`: Lower curved volatility band
- `direction`: Trend direction (1 = uptrend, -1 = downtrend)
- `trend_line`: Active trend line (follows lower band in uptrend, upper band in downtrend)

### Algorithm Complexity

- **Time Complexity**: O(n) where n is the number of bars
- **Space Complexity**: O(n) for storing band arrays

---

## 📝 Example Output

```
CURVED RADIUS SUPERTREND - TREND ANALYSIS
============================================================

Total Trend Changes: 12
Average Trend Duration: 24.58 bars
Median Trend Duration: 18.00 bars
Max Trend Duration: 67 bars
Min Trend Duration: 5 bars

Uptrend Bars: 156 (52.0%)
Downtrend Bars: 144 (48.0%)

Average Return During Uptrend: 0.0234%
Average Return During Downtrend: -0.0198%
```

---

## 🎨 Visualization Features

The `visualize_indicator.py` script provides:

1. **Parameter Comparison**: Side-by-side comparison of 6 different configurations
2. **Trend Marking**: Visual markers for trend change points
3. **Color Coding**: Green for uptrends, red for downtrends
4. **Statistical Analysis**: Comprehensive trend statistics

---

## 🚀 Advanced Usage

### Integration with Trading Systems

```python
# Example: Generate trading signals
def generate_signals(result, close):
    signals = []
    
    for i in range(1, len(result)):
        # Trend change from down to up = BUY signal
        if result['direction'].iloc[i] == 1 and result['direction'].iloc[i-1] == -1:
            signals.append(('BUY', i, close[i]))
        
        # Trend change from up to down = SELL signal
        elif result['direction'].iloc[i] == -1 and result['direction'].iloc[i-1] == 1:
            signals.append(('SELL', i, close[i]))
    
    return signals
```

### Real-time Data Integration

```python
# Example: Update with new bar
def update_with_new_bar(indicator, historical_data, new_bar):
    # Append new bar to historical data
    updated_high = np.append(historical_data['high'], new_bar['high'])
    updated_low = np.append(historical_data['low'], new_bar['low'])
    updated_close = np.append(historical_data['close'], new_bar['close'])
    
    # Recalculate indicator
    result = indicator.calculate(updated_high, updated_low, updated_close)
    
    return result.iloc[-1]  # Return latest values
```

---

## 📚 References

This implementation is based on the theoretical framework of curvature dynamics applied to financial markets, combining:

- ATR-based volatility measurement
- Quadratic acceleration modeling
- Adaptive smoothing techniques

---

## 📄 License

This project is provided as-is for educational and research purposes.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

---

## ⚠️ Disclaimer

This indicator is for educational purposes only. Always perform your own analysis and risk management before making trading decisions. Past performance does not guarantee future results.

