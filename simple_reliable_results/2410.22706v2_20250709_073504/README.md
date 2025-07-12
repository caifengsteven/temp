# GSPHAR Model with Bloomberg Data Integration

Enhanced implementation of the Graph Signal Processing Heterogeneous Autoregressive (GSPHAR) model from paper **2410.22706v2** with real Bloomberg data integration via xbbg.

## Overview

This implementation extends the original GSPHAR model to work with real-world financial data from Bloomberg, providing:

- **Real Bloomberg Data**: Fetches actual market data for major stock indices
- **Realized Volatility Calculation**: Computes RV using multiple methods (daily OHLC, intraday)
- **Volatility Spillover Networks**: Constructs directed graphs using Diebold-Yilmaz methodology
- **Enhanced GSPHAR Model**: Full implementation with magnetic Laplacian operators
- **Comprehensive Visualization**: Multiple plots for analysis and interpretation

## Key Features

### 1. Bloomberg Data Integration
- Automatic data fetching via xbbg library
- Support for multiple stock indices worldwide
- Flexible date range and frequency selection
- Fallback to synthetic data if Bloomberg unavailable

### 2. Realized Volatility Calculation
- **Daily Method**: Garman-Klass estimator using OHLC data
- **Intraday Methods**: 5-minute and 1-minute interval support
- **Proper Scaling**: Annualized percentage volatility as in academic literature

### 3. Volatility Spillover Networks
- **Diebold-Yilmaz Framework**: VAR-based spillover analysis
- **Directed Graphs**: Captures asymmetric volatility transmission
- **Dynamic Networks**: Time-varying spillover relationships

### 4. Enhanced GSPHAR Model
- **Magnetic Laplacian**: Handles directed graph structures
- **Graph Fourier Transform**: Spectral domain processing
- **Learnable Filters**: Adaptive convolution weights
- **HAR Structure**: Daily, weekly, monthly volatility components

## Installation

### 1. Basic Requirements
```bash
pip install -r requirements.txt
```

### 2. Bloomberg Setup
See `bloomberg_setup_guide.md` for detailed Bloomberg API configuration.

### 3. Test Installation
```bash
python test_bloomberg_connection.py
```

## Usage

### Quick Start
```bash
python 2410.22706v2_test_strategy.py
```

### With Custom Configuration
```python
from bloomberg_data_manager import BloombergDataManager
from gsphar_model import GSPHAR

# Initialize data manager
bbg = BloombergDataManager()

# Fetch data
tickers = ['SPX Index', 'SX5E Index', 'NKY Index']
rv_data = bbg.get_realized_volatility_series(
    tickers=tickers,
    start_date='2020-01-01',
    end_date='2023-12-31'
)

# Train model
mae, model, adjacency = test_har_model(rv_data)
```

## Model Architecture

### GSPHAR Components

1. **Magnetic Laplacian Computation**
   - Handles directed volatility spillover networks
   - Preserves directional information via complex phases
   - Enables proper graph signal processing

2. **Graph Fourier Transform**
   - Transforms volatility signals to spectral domain
   - Enables frequency-based analysis
   - Facilitates convolution operations

3. **Learnable Convolution Filters**
   - Adaptive weights for mid-term (weekly) patterns
   - Adaptive weights for long-term (monthly) patterns
   - Convex combination constraints

4. **HAR Structure Integration**
   - Daily, weekly, monthly volatility components
   - Applied in both real and imaginary spectral domains
   - Merged via neural network

### Mathematical Framework

The model implements the following key equations from the paper:

- **Magnetic Laplacian**: `L_m = I - D^(-1/2) A D^(-1/2) exp(iΘ)`
- **Graph Fourier Transform**: `x̂ = U_m^H x`
- **Spectral Filtering**: `ŷ = g(L_m) x̂`
- **HAR Regression**: `RV_{t+1} = β₀ + β_d RV_t + β_w RV_w + β_m RV_m`

## Data Sources

### Supported Bloomberg Tickers
- **SPX Index**: S&P 500 (US)
- **SX5E Index**: Euro Stoxx 50 (Europe)
- **NKY Index**: Nikkei 225 (Japan)
- **UKX Index**: FTSE 100 (UK)
- **HSI Index**: Hang Seng (Hong Kong)
- **AS51 Index**: ASX 200 (Australia)
- **IBOV Index**: Bovespa (Brazil)
- **MEXBOL Index**: IPC Mexico

### Data Requirements
- Minimum 500 trading days for reliable spillover estimation
- Daily frequency recommended for volatility modeling
- Intraday data optional for enhanced RV calculation

## Output Files

The script generates several visualization and analysis files:

- `training_curve.png`: Model training progress
- `prediction_sample.png`: Sample volatility forecasts
- `weights_comparison.png`: Learned vs traditional HAR weights
- `spillover_network.png`: Volatility spillover network heatmap
- `rv_timeseries.png`: Realized volatility time series

## Performance Metrics

The model is evaluated using:
- **Mean Absolute Error (MAE)**: Primary loss function
- **Spillover Index**: Network connectivity measure
- **Filter Weight Analysis**: Comparison with traditional HAR

## Research Applications

This implementation supports research in:
- **Volatility Forecasting**: Multi-step ahead predictions
- **Risk Management**: Portfolio volatility modeling
- **Systemic Risk**: Financial contagion analysis
- **Market Microstructure**: High-frequency volatility patterns

## Customization

### Adding New Indices
Edit the ticker list in `run_bloomberg_experiment()`:
```python
bloomberg_tickers = [
    'YOUR_INDEX Index',
    # Add more tickers here
]
```

### Modifying Model Parameters
Adjust hyperparameters in the model initialization:
```python
model = GSPHAR(
    n_nodes=len(tickers),
    U_m=eigenvectors,
    # Add custom parameters
)
```

### Changing Spillover Analysis
Modify the spillover network construction:
```python
spillover_net = VolatilitySpilloverNetwork(
    window_size=252,  # 1 year rolling window
    forecast_horizon=10  # 10-step ahead FEVD
)
```

## Citation

If you use this implementation in your research, please cite the original paper:

```bibtex
@article{gsphar2024,
  title={Graph Signal Processing for Volatility Forecasting},
  journal={arXiv preprint arXiv:2410.22706v2},
  year={2024}
}
```

## License

This implementation is provided for academic and research purposes. Bloomberg data usage must comply with Bloomberg licensing agreements.

## Support

For technical issues:
1. Check `bloomberg_setup_guide.md` for setup problems
2. Run `test_bloomberg_connection.py` to diagnose Bloomberg issues
3. Review error logs for debugging information

For Bloomberg API support, consult Bloomberg documentation or help desk.
