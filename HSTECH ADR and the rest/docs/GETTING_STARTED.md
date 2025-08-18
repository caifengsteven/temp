# Getting Started with HSTECH Index Estimation System

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/your-org/hstech-estimation.git
cd hstech-estimation

# Install dependencies
pip install -r requirements.txt

# Or install in development mode
pip install -e .
```

### 2. Configuration

```bash
# Copy example configuration
cp config/config.example.yaml config/config.yaml

# Edit configuration with your API keys
nano config/config.yaml
```

**Required API Keys:**
- Bloomberg API key (for Bloomberg Cloud API) or Bloomberg Terminal access
- Alpha Vantage API key (fallback, free tier available)
- Yahoo Finance (fallback, no API key required)

### 3. Basic Usage

```python
# Simple estimation example
python example_usage.py --simple

# Full estimation with data fetching
python example_usage.py
```

### 4. Run Tests

```bash
# Run all tests with coverage
python run_tests.py

# Quick test run
python run_tests.py --quick

# Run specific test
python run_tests.py test_models.py
```

## Detailed Setup

### Environment Setup

1. **Python Version**: Requires Python 3.8 or higher
2. **Virtual Environment** (recommended):
   ```bash
   python -m venv hstech_env
   source hstech_env/bin/activate  # On Windows: hstech_env\Scripts\activate
   pip install -r requirements.txt
   ```

### API Key Configuration

1. **Bloomberg API** (primary data source):
   - **Bloomberg Terminal**: Requires Bloomberg Terminal subscription and BLPAPI
   - **Bloomberg Cloud API**: Requires Bloomberg Cloud subscription
   - Add to `config/config.yaml`:
     ```yaml
     api_keys:
       bloomberg_api_key: "YOUR_BLOOMBERG_API_KEY"
       bloomberg_cloud_url: "https://api.bloomberg.com"
     ```

2. **Alpha Vantage** (fallback for currency data):
   - Sign up at https://www.alphavantage.co/support/#api-key
   - Free tier: 5 API requests per minute, 500 per day
   - Add to `config/config.yaml`:
     ```yaml
     api_keys:
       alpha_vantage: "YOUR_API_KEY_HERE"
     ```

3. **Yahoo Finance**: No API key required, used as fallback data source

### Directory Structure

```
hstech-estimation/
├── src/                    # Source code
│   ├── models/            # Data models
│   ├── data/              # Data fetching and management
│   ├── estimation/        # Core estimation algorithms
│   ├── backtesting/       # Backtesting framework
│   └── utils/             # Utility functions
├── config/                # Configuration files
├── data/                  # Data storage
├── tests/                 # Unit tests
├── docs/                  # Documentation
├── logs/                  # Log files (created automatically)
└── example_usage.py       # Example script
```

## Basic Examples

### 1. Simple Estimation

```python
import asyncio
from src.models import Config
from src.estimation import HSTECHEstimator
from src.data import DataManager

async def simple_estimation():
    # Load configuration
    config = Config.from_yaml("config/config.yaml")
    
    # Initialize components
    estimator = HSTECHEstimator(config)
    data_manager = DataManager(config)
    
    # Fetch data
    success = await data_manager.fetch_all_current_data()
    if not success:
        print("Failed to fetch data")
        return
    
    # Get estimation data
    estimation_data = await data_manager.get_estimation_data()
    
    # Run estimation
    result = await estimator.estimate_current_price(*estimation_data)
    
    print(f"Estimated HSTECH: {result.estimated_value:.2f}")
    print(f"Confidence: {result.confidence:.1%}")

# Run the example
asyncio.run(simple_estimation())
```

### 2. Data Quality Check

```python
from src.data import DataManager
from src.models import Config

async def check_data_quality():
    config = Config()
    data_manager = DataManager(config)
    
    # Fetch current data
    await data_manager.fetch_all_current_data()
    
    # Get quality report
    report = data_manager.get_data_quality_report()
    
    print(f"Overall quality: {report['overall_quality']}")
    print(f"ADR coverage: {report['adr_coverage']['coverage_percent']:.1f}%")
    print(f"HK coverage: {report['hk_coverage']['coverage_percent']:.1f}%")

asyncio.run(check_data_quality())
```

### 3. Market Hours Check

```python
from src.utils import MarketHoursChecker
from src.models import Config, MarketHours

# Setup market hours
market_hours = {
    "hong_kong": MarketHours(open="01:30", close="08:00"),
    "us": MarketHours(open="14:30", close="21:00")
}

checker = MarketHoursChecker(market_hours)

# Check current status
status = checker.get_market_status_summary()
print(f"Should run estimation: {status['should_run_estimation']}")
print(f"Reason: {status['estimation_reason']}")
```

### 4. Backtesting Example

```python
from src.backtesting import HSTECHBacktester
from src.models import Config

async def run_simple_backtest():
    config = Config()
    backtester = HSTECHBacktester(config)
    
    # Run backtest for last 30 days
    result = await backtester.run_backtest(
        start_date="2024-07-01",
        end_date="2024-08-01"
    )
    
    print(f"Success rate: {result.successful_predictions/result.total_predictions:.1%}")
    print(f"MAE: {result.mae:.2f}")
    print(f"Correlation: {result.correlation:.3f}")

asyncio.run(run_simple_backtest())
```

## Configuration Guide

### Basic Configuration

```yaml
# config/config.yaml
api_keys:
  alpha_vantage: "YOUR_API_KEY"

estimation:
  lookback_days: 252
  weights:
    adr_based: 0.6
    covariance_based: 0.25
    market_indicators: 0.15

logging:
  level: "INFO"
  file: "logs/hstech_estimation.log"
```

### Advanced Configuration

```yaml
# Advanced settings
estimation:
  lookback_days: 504  # 2 years of data
  min_correlation_threshold: 0.4
  weights:
    adr_based: 0.7
    covariance_based: 0.2
    market_indicators: 0.1
  indicators:
    - "PDD"
    - "KWEB"
    - "ASHR"
    - "FXI"

data_sources:
  price_update_frequency: 30  # seconds
  currency_update_frequency: 300  # 5 minutes

market_hours:
  hong_kong:
    open: "01:30"  # 9:30 AM HKT in UTC
    close: "08:00"  # 4:00 PM HKT in UTC
  us:
    open: "14:30"  # 9:30 AM EST in UTC
    close: "21:00"  # 4:00 PM EST in UTC
```

## Troubleshooting

### Common Issues

1. **API Rate Limits**:
   - Alpha Vantage free tier: 5 requests/minute
   - Solution: Increase `price_update_frequency` in config
   - Or upgrade to paid Alpha Vantage plan

2. **Missing Data**:
   - Some stocks may not have ADR mappings
   - Some ETFs may be temporarily unavailable
   - Check data quality report for coverage

3. **Network Issues**:
   - Yahoo Finance may occasionally be slow
   - Implement retry logic or use Alpha Vantage as backup

4. **Time Zone Issues**:
   - Ensure system time is correct
   - Market hours are configured in UTC

### Debug Mode

```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Or in config.yaml
logging:
  level: "DEBUG"
```

### Data Validation

```python
# Check data quality before estimation
report = data_manager.get_data_quality_report()
if report['overall_quality'] == 'poor':
    print("Warning: Poor data quality detected")
    print(f"Missing data: {report['adr_coverage']['missing']}")
```

## Next Steps

1. **Customize Configuration**: Adjust weights and parameters for your use case
2. **Run Backtesting**: Validate performance on historical data
3. **Set Up Monitoring**: Implement logging and alerting
4. **Integrate**: Connect to your trading or monitoring systems
5. **Optimize**: Fine-tune parameters based on backtesting results

## Support

- **Documentation**: See `docs/` directory for detailed guides
- **API Reference**: `docs/API_REFERENCE.md`
- **Methodology**: `docs/METHODOLOGY.md`
- **Issues**: Report bugs and feature requests on GitHub
- **Tests**: Run `python run_tests.py` to verify installation
