# API Reference

## Core Classes

### HSTECHEstimator

Main class for HSTECH index estimation.

```python
from src.estimation import HSTECHEstimator
from src.models import Config

# Initialize with default config
estimator = HSTECHEstimator()

# Initialize with custom config
config = Config.from_yaml("config/config.yaml")
estimator = HSTECHEstimator(config)
```

#### Methods

##### `estimate_current_price()`

Estimate current HSTECH index price using all available methods.

```python
async def estimate_current_price(
    self,
    current_adr_prices: Dict[str, PriceData],
    last_hk_prices: Dict[str, PriceData],
    current_exchange_rate: CurrencyRate,
    last_hstech_close: IndexData,
    current_indicator_prices: Dict[str, PriceData],
    previous_indicator_prices: Dict[str, PriceData],
    historical_data: Optional[Dict[str, pd.DataFrame]] = None
) -> EstimationResult
```

**Parameters:**
- `current_adr_prices`: Current ADR prices {symbol: PriceData}
- `last_hk_prices`: Last known HK closing prices {symbol: PriceData}
- `current_exchange_rate`: Current USD/HKD exchange rate
- `last_hstech_close`: Last HSTECH index closing value
- `current_indicator_prices`: Current market indicator prices
- `previous_indicator_prices`: Previous market indicator prices
- `historical_data`: Historical price data for covariance modeling

**Returns:** `EstimationResult` with estimated value and confidence

##### `get_estimation_summary()`

Get summary of estimator configuration and capabilities.

```python
def get_estimation_summary(self) -> Dict[str, any]
```

**Returns:** Dict with configuration details and coverage statistics

### DataManager

Manages all data operations for the estimation system.

```python
from src.data import DataManager
from src.models import Config

config = Config()
data_manager = DataManager(config)
```

#### Methods

##### `fetch_all_current_data()`

Fetch all current market data needed for estimation.

```python
async def fetch_all_current_data(self) -> bool
```

**Returns:** True if all critical data was fetched successfully

##### `get_estimation_data()`

Get all data needed for HSTECH estimation.

```python
async def get_estimation_data(self) -> Tuple[Dict, Dict, CurrencyRate, IndexData, Dict, Dict]
```

**Returns:** Tuple of (current_adr_prices, last_hk_prices, exchange_rate, last_hstech_close, current_indicators, previous_indicators)

##### `fetch_historical_data()`

Fetch historical price data for covariance modeling.

```python
async def fetch_historical_data(
    self, 
    symbols: List[str], 
    days: int = 252
) -> Dict[str, pd.DataFrame]
```

**Parameters:**
- `symbols`: List of stock symbols
- `days`: Number of days of historical data

**Returns:** Dict of {symbol: DataFrame with date, close, returns columns}

##### `get_data_quality_report()`

Generate a report on current data quality.

```python
def get_data_quality_report(self) -> Dict[str, any]
```

**Returns:** Dict with data quality metrics

### ADRMapper

Handles mapping between Hong Kong stocks and their US ADR equivalents.

```python
from src.data import ADRMapper

mapper = ADRMapper()
```

#### Methods

##### `has_adr_mapping()`

Check if a Hong Kong stock has a US ADR mapping.

```python
def has_adr_mapping(self, hk_symbol: str) -> bool
```

##### `convert_adr_to_hk_price()`

Convert ADR price to equivalent Hong Kong stock price.

```python
def convert_adr_to_hk_price(
    self, 
    adr_price: Decimal, 
    hk_symbol: str, 
    exchange_rate: Decimal
) -> Optional[Decimal]
```

**Parameters:**
- `adr_price`: Price of the ADR in USD
- `hk_symbol`: Hong Kong stock symbol
- `exchange_rate`: USD/HKD exchange rate

**Returns:** Equivalent Hong Kong stock price in HKD

##### `get_all_adr_symbols()`

Get list of all US ADR symbols that map to HSTECH components.

```python
def get_all_adr_symbols(self) -> List[str]
```

### HSTECHBacktester

Comprehensive backtesting framework for HSTECH estimation.

```python
from src.backtesting import HSTECHBacktester
from src.models import Config

config = Config()
backtester = HSTECHBacktester(config)
```

#### Methods

##### `run_backtest()`

Run comprehensive backtest over specified period.

```python
async def run_backtest(
    self,
    start_date: str,
    end_date: str,
    estimation_frequency: str = "daily"
) -> BacktestResult
```

**Parameters:**
- `start_date`: Start date in YYYY-MM-DD format
- `end_date`: End date in YYYY-MM-DD format
- `estimation_frequency`: Frequency of estimations ("daily", "hourly")

**Returns:** `BacktestResult` with comprehensive metrics

## Data Models

### PriceData

Represents price data for a stock at a specific time.

```python
from src.models import PriceData
from decimal import Decimal
from datetime import datetime, timezone

price_data = PriceData(
    symbol="TCEHY",
    price=Decimal("45.50"),
    currency="USD",
    timestamp=datetime.now(timezone.utc),
    volume=1000000,
    source="yahoo_finance"
)
```

### EstimationResult

Represents the result of an HSTECH index estimation.

```python
from src.models import EstimationResult

# Access estimation results
print(f"Estimated value: {result.estimated_value}")
print(f"Confidence: {result.confidence}")
print(f"Method weights: {result.method_weights}")
print(f"Component contributions: {result.component_contributions}")
```

### Config

Main configuration class with validation.

```python
from src.models import Config

# Load from YAML
config = Config.from_yaml("config/config.yaml")

# Access configuration
print(f"Lookback days: {config.estimation.lookback_days}")
print(f"Method weights: {config.estimation.weights}")
print(f"API keys: {config.api_keys}")
```

## Utility Functions

### Backtesting Metrics

```python
from src.backtesting import (
    calculate_mse, calculate_mae, calculate_mape,
    create_performance_summary, print_performance_summary
)

# Calculate individual metrics
mse = calculate_mse(actual_values, predicted_values)
mae = calculate_mae(actual_values, predicted_values)
mape = calculate_mape(actual_values, predicted_values)

# Create comprehensive summary
summary = create_performance_summary(
    actual_values, 
    predicted_values,
    confidence_scores=confidence_scores
)

# Print formatted report
print_performance_summary(summary)
```

### Market Hours

```python
from src.utils import MarketHoursChecker

checker = MarketHoursChecker(config.market_hours)

# Check market status
hk_open = checker.is_hk_market_open()
us_open = checker.is_us_market_open()
should_estimate, reason = checker.should_run_estimation()

# Get comprehensive status
status = checker.get_market_status_summary()
```

### Logging

```python
from src.utils import setup_logging, get_logger

# Setup logging
logger = setup_logging(config.logging)

# Get logger in other modules
logger = get_logger("hstech_estimation")
logger.info("Starting estimation process")
```

## Error Handling

### Common Exceptions

- `ValueError`: Invalid input parameters or configuration
- `FileNotFoundError`: Missing configuration or data files
- `ConnectionError`: Network issues when fetching data
- `TimeoutError`: Data fetch timeouts

### Best Practices

```python
try:
    result = await estimator.estimate_current_price(**data)
    print(f"Estimation successful: {result.estimated_value}")
except ValueError as e:
    logger.error(f"Invalid input: {e}")
except ConnectionError as e:
    logger.error(f"Network error: {e}")
    # Implement fallback logic
except Exception as e:
    logger.error(f"Unexpected error: {e}", exc_info=True)
    # Handle gracefully
```

## Configuration Examples

### Basic Configuration

```yaml
# config/config.yaml
estimation:
  lookback_days: 252
  weights:
    adr_based: 0.6
    covariance_based: 0.25
    market_indicators: 0.15

api_keys:
  alpha_vantage: "YOUR_API_KEY"

logging:
  level: "INFO"
  file: "logs/hstech.log"
```

### Advanced Configuration

```yaml
estimation:
  lookback_days: 504  # 2 years
  min_correlation_threshold: 0.4
  weights:
    adr_based: 0.7
    covariance_based: 0.2
    market_indicators: 0.1
  indicators:
    - "PDD"
    - "KWEB"
    - "ASHR"

data_sources:
  price_update_frequency: 30  # seconds
  currency_update_frequency: 300  # seconds

backtesting:
  start_date: "2022-01-01"
  end_date: "2024-08-18"
  metrics:
    - "mse"
    - "mae"
    - "correlation"
    - "directional_accuracy"
```
