# Enhanced DGNN Implementation with Bloomberg Data Integration

## Overview

This project enhances the original Dynamic Graph Neural Network (DGNN) implementation from paper 2410.23275v1 with real-time Bloomberg market data integration via the xbbg API. The enhanced version provides more realistic margin call forecasting by incorporating actual market conditions.

## Key Enhancements

### 1. Bloomberg Data Integration (`BloombergDataFetcher`)

**Features:**
- Real-time OIS (Overnight Index Swap) curve fetching
- Historical reference rate data retrieval
- Volatility surface integration
- Automatic fallback to simulated data when Bloomberg is unavailable

**API Usage (Following xbbg Documentation):**
```python
from xbbg import blp

# BDH - Historical Data (with timeout for reliability)
data = blp.bdh(tickers, 'PX_LAST', start_date=date, end_date=date, timeout=30)

# BDP - Point Data (current values)
data = blp.bdp(ticker, ['Security_Name', 'Crncy', 'Country'], timeout=30)

# BDS - Bulk Data (for complex datasets)
data = blp.bds(ticker, 'DVD_Hist_All', DVD_Start_Dt=start, DVD_End_Dt=end, timeout=30)

# Enhanced error handling and fallback mechanisms
if data.empty:
    print("No Bloomberg data available, using simulated data")
    return self._simulate_data()
```

**Supported Instruments:**
- USD OIS: USSO1Z, USSOA, USSOB, USSOC, USSOD, USSO1, USSO2, USSO5, USSO10
- EUR OIS: EUSWO1Z, EUSWOA, EUSWOB, EUSWOC, EUSWOD, EUSWO1, EUSWO2, EUSWO5, EUSWO10
- Reference Rates: FEDL01 Index (Fed Funds), EONIA Index

### 2. Enhanced CIR Process (`EnhancedCIRProcess`)

**Market Calibration:**
- Automatic parameter estimation from historical market data
- Mean reversion speed (k) calibrated from rate changes
- Long-term mean (theta) estimated from historical average
- Volatility (sigma) derived from empirical rate variance
- Feller condition enforcement for mathematical stability

**Improvements over Original:**
- Market-driven parameter initialization
- Robust fallback mechanisms
- Multi-currency support
- Real-time recalibration capability

### 3. Enhanced Financial Network (`EnhancedFinancialNetwork`)

**Market-Based Contract Pricing:**
- Yield curve interpolation for fair rate calculation
- Mark-to-market valuation using current market data
- Enhanced contract features (notional, currency, day count)
- Realistic spread adjustments based on volatility

**Contract Enhancements:**
```python
contract = {
    'id': f"OIS_{counter}",
    'start_time': t,
    'maturity': maturity,
    'nodes': (i, j),
    'rate': market_fair_rate,  # From yield curve
    'delta': (delta_i, delta_j),
    'principal': 1.0,
    'currency': 'USD',
    'contract_type': 'OIS',
    'notional': 1000000,
    'day_count': 'ACT/360'
}
```

### 4. Improved Margin Calculation

**Enhanced Features:**
- Present value calculations with proper discounting
- Minimum transfer amounts (thresholds)
- Overnight rate adjustments
- Multi-currency support

**Calculation Method:**
```python
# Enhanced mark-to-market
contract_value = self._calculate_contract_value(contract, current_rate, t)
net_value += delta * contract_value

# Variation margin with thresholds
margins = curr_values - prev_values * overnight_factor
margins = np.where(np.abs(margins) < threshold, 0, margins)
```

## File Structure

### Core Files
- `2410.23275v1_test_strategy.py` - Enhanced main implementation
- `enhanced_dgnn_demo.py` - Simplified demonstration script
- `requirements.txt` - Dependencies
- `ENHANCED_DGNN_SUMMARY.md` - This documentation

### Key Classes

1. **BloombergDataFetcher** - Market data integration
2. **EnhancedCIRProcess** - Market-calibrated interest rate simulation
3. **EnhancedFinancialNetwork** - Realistic contract simulation
4. **GCLSTM** - Graph Convolutional LSTM (unchanged)
5. **PricingModule** - Contract pricing neural network (enhanced)
6. **DGNN** - Main model architecture (enhanced)

## Usage Examples

### Basic Market Data Fetching
```python
from enhanced_dgnn_demo import EnhancedMarketDataDemo

demo = EnhancedMarketDataDemo('USD')
ois_curve = demo.fetch_ois_curve()
historical_rates = demo.fetch_reference_rate_history(30)
```

### Enhanced Model Training
```python
# Initialize with Bloomberg data
data_fetcher = BloombergDataFetcher()
cir = EnhancedCIRProcess(data_fetcher=data_fetcher, currency='USD')
network = EnhancedFinancialNetwork(n_nodes=5, data_fetcher=data_fetcher)

# Simulate with market data
rates = cir.simulate(total_days)
contracts = network.simulate_contracts(rates, total_days)
```

## Bloomberg API Requirements (Per xbbg Documentation)

### Installation
```bash
# Install xbbg
pip install xbbg

# Install Bloomberg official Python API
pip install blpapi --index-url=https://bcms.bloomberg.com/pip/simple/
```

### Prerequisites
- **Bloomberg C++ SDK version 3.12.1 or higher**
  - Visit [Bloomberg API Library](https://www.bloomberg.com/professional/support/api-library/)
  - Download C++ Supported Release
  - Copy `blpapi3_32.dll` and `blpapi3_64.dll` to Bloomberg `BLPAPI_ROOT` folder
- Bloomberg Terminal or BPIPE connection
- Valid Bloomberg API license
- Dependencies: numpy, pandas, ruamel.yaml, pyarrow

### xbbg Features Implemented
- **BDH (Historical Data)**: OIS curves, reference rate history
- **BDP (Point Data)**: Current rates, security information, volatility
- **BDS (Bulk Data)**: Available for dividend history, earnings data
- **Timeout Parameters**: All API calls include timeout for reliability
- **Excel Compatible Inputs**: Supports Bloomberg Excel-style parameters
- **Error Handling**: Comprehensive fallback to simulated data

### xbbg API Implementation Examples

**Historical Data (BDH):**
```python
# Fetch OIS curve with timeout
data = blp.bdh(tickers, 'PX_LAST', start_date=date, end_date=date, timeout=30)

# Fetch reference rate history
data = blp.bdh('FEDL01 Index', 'PX_LAST',
              start_date='2024-01-01', end_date='2024-12-31', timeout=60)
```

**Point Data (BDP):**
```python
# Fetch security information
info = blp.bdp('USSO1Z Curncy', ['Security_Name', 'Crncy', 'Country'])

# Fetch current volatility
vol = blp.bdp('USSV1Y Curncy', 'PX_LAST')
```

**Bulk Data (BDS):**
```python
# Fetch dividend history (available for equity analysis)
divs = blp.bds('AAPL US Equity', 'DVD_Hist_All',
              DVD_Start_Dt='20240101', DVD_End_Dt='20241231')
```

### Fallback Behavior
When Bloomberg is unavailable:
- Automatic fallback to realistic simulated data
- Maintains same interface and functionality
- Preserves model training capability
- Mock blp module provides consistent API

## Performance Improvements

### Optimization Features
- Reduced API calls (monthly updates vs. daily)
- Efficient data caching
- Vectorized calculations
- Batch processing for multiple contracts

### Scalability
- Configurable update frequencies
- Memory-efficient contract storage
- Parallel processing capability
- Multi-currency support

## Testing and Validation

### Demo Script Results
```
✓ Market data integration: Active/Simulated
✓ OIS curve points: 9
✓ Historical data points: 30
✓ Enhanced pricing: Enabled
✓ Visualization: enhanced_dgnn_demo.png
```

### Key Metrics
- Curve slope analysis (10Y-1M spread)
- Historical volatility calculation
- Contract valuation accuracy
- Model convergence validation

## Future Enhancements

### Planned Features
1. **Multi-Asset Support** - Corporate bonds, credit derivatives
2. **Real-Time Streaming** - Live market data feeds
3. **Risk Metrics** - VaR, Expected Shortfall calculations
4. **Regulatory Compliance** - BCBS, CFTC margin requirements
5. **Machine Learning** - Advanced calibration algorithms

### Technical Improvements
1. **Distributed Computing** - Spark/Dask integration
2. **Database Integration** - Time-series data storage
3. **API Optimization** - Async Bloomberg calls
4. **Model Validation** - Backtesting framework

## Conclusion

The enhanced DGNN implementation successfully integrates real Bloomberg market data while maintaining the original model's theoretical foundation. Key improvements include:

- **Realistic Market Data** - Live OIS curves and reference rates
- **Enhanced Pricing** - Market-based contract valuation
- **Robust Calibration** - Automatic parameter estimation
- **Production Ready** - Error handling and fallback mechanisms

This implementation bridges the gap between academic research and practical financial applications, providing a robust foundation for real-world margin call forecasting systems.
