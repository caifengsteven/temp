# HSTECH Index Estimation System

A comprehensive system to estimate the HSTECH index price using US market data when Hong Kong markets are closed.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-passing-green.svg)](tests/)

## 🎯 Overview

When Hong Kong markets are closed but US markets are open, this system provides real-time HSTECH index price estimates using a sophisticated three-step approach:

1. **🔄 ADR-based Updates**: Direct calculation using dual-listed stocks (Tencent→TCEHY, Alibaba→BABA, etc.)
2. **📊 Statistical Modeling**: Covariance-based estimation for non-dual-listed components
3. **🚀 Market Enhancement**: Integration of broader market indicators (PDD, KWEB ETF, etc.)

## ✨ Key Features

- **Real-time Estimation**: Sub-minute updates when US markets are active
- **High Coverage**: ~40-50% direct ADR coverage + statistical modeling for remainder
- **Bloomberg Integration**: Primary data source with Terminal API and Cloud API support
- **Multi-source Fallbacks**: Yahoo Finance, Alpha Vantage for reliability
- **Robust Validation**: Comprehensive backtesting framework with multiple accuracy metrics
- **Production Ready**: Async architecture, error handling, logging, and monitoring

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/your-org/hstech-estimation.git
cd hstech-estimation
pip install -r requirements.txt

# For Bloomberg Terminal API support
pip install blpapi
```

### Basic Usage

```python
import asyncio
from src.estimation import HSTECHEstimator
from src.data import DataManager
from src.models import Config

async def estimate_hstech():
    # Initialize system
    config = Config.from_yaml("config/config.yaml")
    estimator = HSTECHEstimator(config)
    data_manager = DataManager(config)

    # Fetch current market data
    await data_manager.fetch_all_current_data()
    estimation_data = await data_manager.get_estimation_data()

    # Run estimation
    result = await estimator.estimate_current_price(*estimation_data)

    print(f"📈 Estimated HSTECH: {result.estimated_value:.2f}")
    print(f"🎯 Confidence: {result.confidence:.1%}")
    print(f"⚖️  Method weights: {result.method_weights}")

# Run estimation
asyncio.run(estimate_hstech())
```

### Example Output

```
📈 Estimated HSTECH: 10,247.85
🎯 Confidence: 87.3%
⚖️  Method weights: {'adr_based': 0.6, 'covariance_based': 0.25, 'market_indicators': 0.15}

Top Component Contributions:
  0700.HK (Tencent): +0.0023
  9988.HK (Alibaba): -0.0012
  3690.HK (Meituan): +0.0008
```

## 📋 System Architecture

### Three-Step Estimation Process

#### Step 1: ADR-based Estimation
- **Coverage**: Major HSTECH components with US listings
- **Method**: Direct price conversion with currency/ratio adjustments
- **Components**: TCEHY, BABA, JD, MPNGY, NTES, LI, XPEV, etc.

#### Step 2: Covariance Modeling
- **Purpose**: Estimate non-ADR components using statistical relationships
- **Methods**: Ridge regression, correlation weighting, factor models
- **Data**: 252-day rolling historical covariance matrix

#### Step 3: Market Enhancement
- **Indicators**: PDD (25%), KWEB (35%), ASHR (15%), FXI (15%), MCHI (10%)
- **Purpose**: Capture broader Chinese tech/internet sector sentiment
- **Integration**: Weighted enhancement factor applied to base estimation

## 📊 Performance Metrics

Based on backtesting (example results):

| Metric | Value |
|--------|-------|
| Mean Absolute Error (MAE) | 45.2 points |
| Mean Absolute Percentage Error (MAPE) | 0.43% |
| Correlation with Actual | 0.891 |
| Directional Accuracy | 78.5% |
| Coverage (by weight) | 47.3% direct + 52.7% estimated |

## 🛠️ Configuration

### Basic Setup

```yaml
# config/config.yaml
api_keys:
  # Bloomberg API (Primary)
  bloomberg_api_key: "YOUR_BLOOMBERG_API_KEY"
  bloomberg_cloud_url: "https://api.bloomberg.com"

  # Fallback APIs
  alpha_vantage: "YOUR_ALPHA_VANTAGE_API_KEY"

data_sources:
  primary_data_source: "bloomberg"
  fallback_data_source: "yahoo_finance"

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
estimation:
  lookback_days: 504  # 2 years
  min_correlation_threshold: 0.4
  weights:
    adr_based: 0.7
    covariance_based: 0.2
    market_indicators: 0.1

data_sources:
  price_update_frequency: 30  # seconds
  currency_update_frequency: 300  # 5 minutes
```

## 🧪 Testing & Validation

### Run Tests

```bash
# Full test suite with coverage
python run_tests.py

# Quick tests
python run_tests.py --quick

# Specific test file
python run_tests.py test_models.py
```

### Backtesting

```python
from src.backtesting import HSTECHBacktester

async def run_backtest():
    backtester = HSTECHBacktester(config)
    result = await backtester.run_backtest("2024-01-01", "2024-08-01")

    print(f"Success rate: {result.successful_predictions/result.total_predictions:.1%}")
    print(f"MAE: {result.mae:.2f}")
    print(f"Correlation: {result.correlation:.3f}")
```

## 📁 Project Structure

```
hstech-estimation/
├── src/
│   ├── models/           # Pydantic data models
│   ├── data/            # Data fetching & ADR mapping
│   ├── estimation/      # Core estimation algorithms
│   ├── backtesting/     # Validation framework
│   └── utils/           # Logging, market hours, etc.
├── config/              # YAML configuration
├── data/               # HSTECH components & mappings
├── tests/              # Unit & integration tests
├── docs/               # Comprehensive documentation
├── example_usage.py    # Demo script
└── run_tests.py       # Test runner
```

## 📚 Documentation

- **[Getting Started](docs/GETTING_STARTED.md)**: Installation and basic usage
- **[API Reference](docs/API_REFERENCE.md)**: Complete API documentation
- **[Methodology](docs/METHODOLOGY.md)**: Detailed technical approach
- **[Configuration Guide](config/config.example.yaml)**: All configuration options

## 🔧 Requirements

- **Python**: 3.8+
- **Key Dependencies**: pandas, numpy, scikit-learn, yfinance, aiohttp
- **Optional**: Alpha Vantage API key (free tier sufficient)
- **System**: Works on Windows, macOS, Linux

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Run tests (`python run_tests.py`)
4. Commit changes (`git commit -m 'Add amazing feature'`)
5. Push to branch (`git push origin feature/amazing-feature`)
6. Open Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Disclaimer

This system is for informational and research purposes only. It does not constitute financial advice. Always verify estimates against official market data and consult with qualified financial professionals before making investment decisions.

## 🙏 Acknowledgments

- Hong Kong Stock Exchange for HSTECH index methodology
- Yahoo Finance and Alpha Vantage for market data APIs
- Open source Python ecosystem for robust financial libraries
