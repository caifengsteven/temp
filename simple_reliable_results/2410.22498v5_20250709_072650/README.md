# VIX Stochastic Volatility Model for Corporate Bonds

Enhanced Python implementation based on the academic paper **"The VIX as Stochastic Volatility for Corporate Bonds"** (arXiv:2410.22498v5).

## Overview

This implementation tests the paper's key hypothesis that the VIX can serve as a stochastic volatility factor for corporate bonds, and that dividing residuals by VIX makes them closer to a Gaussian distribution.

## Key Features

### 1. Bloomberg Data Integration
- **Real-time data access** via `xbbg` library
- **VIX data**: Historical VIX index values
- **Corporate bond data**: Bond indices, spreads, and returns
- **Treasury data**: Risk-free rate benchmarks

### 2. Model Implementation
- **VIX Process**: `ln V_t = α + β * ln V_(t-1) + W_t`
- **Bond Rates/Spreads**: `R_t = a + b * R_(t-1) + c * V_t + V_t * Z_t`
- **Bond Returns**: `Q_t = k * R_(t-1) - m * ΔR_t + h * V_t + l + V_t * U_t`

### 3. Analysis Features
- **Model fitting** to real Bloomberg data
- **Residual analysis** with normality testing
- **Real vs simulated** data comparison
- **Comprehensive visualization** with multiple plots

## Installation

### Basic Requirements
```bash
pip install -r requirements.txt
```

### Bloomberg Data Access (Optional)
```bash
pip install xbbg
```
**Note**: Requires Bloomberg Terminal or API access

## Usage

### Basic Usage
```python
python 2410.22498v5_test_strategy.py
```

### With Bloomberg Data
Ensure Bloomberg Terminal is running or API access is configured, then run:
```python
python 2410.22498v5_test_strategy.py
```

The script will automatically:
1. Attempt to load Bloomberg data
2. Fit models to real data if available
3. Run simulated data analysis
4. Generate comparison plots

## Output Files

### Plots Generated
- `simulated_series.png` - Time series of simulated VIX, rates, and returns
- `simulated_*_residuals.png` - Residual analysis for simulated data
- `real_data_residuals.png` - Residual analysis for real Bloomberg data (if available)
- `bloomberg_vs_simulated.png` - Side-by-side comparison of real vs simulated data

### Key Metrics
- **Model parameters** fitted to real data
- **Residual statistics** (mean, std, skewness, kurtosis)
- **Normality tests** (Shapiro-Wilk)
- **Model comparison** (R², AIC)

## Model Classes

### `BloombergDataLoader`
Handles data retrieval from Bloomberg:
- `load_vix_data()` - VIX index data
- `load_corporate_bond_data()` - Bond indices and spreads
- `load_treasury_data()` - Treasury yield curves

### `VIXStochasticVolatilityModel`
Core model implementation:
- `simulate_vix()` - VIX simulation
- `simulate_rates()` - Bond rate simulation
- `fit_vix_model_to_data()` - Fit VIX model to real data
- `fit_bond_model_to_data()` - Fit bond model to real data
- `analyze_real_vs_simulated()` - Compare real vs simulated properties

## Key Findings

The implementation validates the paper's main hypothesis:

1. **VIX as Stochastic Volatility**: VIX effectively captures volatility dynamics in corporate bond markets
2. **Residual Normalization**: Dividing residuals by VIX significantly improves their Gaussian properties
3. **Model Performance**: VIX-enhanced models show better fit compared to basic autoregressive models

## Bloomberg Data Sources

### VIX Data
- `VIX Index` - CBOE Volatility Index

### Corporate Bond Data
- `LF98TRUU Index` - Bloomberg US Corporate Bond Index
- `LUACTRUU Index` - Bloomberg US Credit Index
- `HYG US Equity` - iShares High Yield Corporate Bond ETF
- `LQD US Equity` - iShares Investment Grade Corporate Bond ETF

### Spread Data
- `LUACOAS Index` - US Credit Option Adjusted Spread
- `HYG0OAS Index` - High Yield OAS
- `LQD0OAS Index` - Investment Grade OAS

## Technical Notes

### Data Alignment
- Automatic date alignment between VIX and bond data
- Handling of missing values and data gaps
- Robust error handling for data quality issues

### Statistical Tests
- **Shapiro-Wilk test** for normality
- **Kolmogorov-Smirnov test** for distribution comparison
- **Autocorrelation analysis** for residual independence

### Model Validation
- **In-sample fitting** with real data
- **Out-of-sample simulation** for validation
- **Cross-validation** of model parameters

## Troubleshooting

### Bloomberg Connection Issues
1. Ensure Bloomberg Terminal is running
2. Check Bloomberg API credentials
3. Verify network connectivity
4. Try reducing date range for data requests

### Data Quality Issues
- The script automatically handles missing data
- Minimum data requirements: 100+ observations
- Automatic fallback to simulated data if real data unavailable

## References

- Paper: "The VIX as Stochastic Volatility for Corporate Bonds" (arXiv:2410.22498v5)
- Bloomberg API: [xbbg documentation](https://github.com/alpha-xone/xbbg)
- VIX Methodology: [CBOE VIX White Paper](https://www.cboe.com/tradable_products/vix/)
