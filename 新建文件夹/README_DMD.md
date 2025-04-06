# Dynamic Mode Decomposition (DMD) for Stock Prediction

This repository contains Python scripts for implementing Dynamic Mode Decomposition (DMD) for stock price prediction, backtesting the approach on NASDAQ 100 stocks, and generating comprehensive reports.

## What is DMD?

Dynamic Mode Decomposition (DMD) is a data-driven method for analyzing complex dynamical systems. It was originally developed for fluid dynamics but has found applications in various fields, including finance.

DMD works by decomposing time-series data into spatial-temporal coherent structures (modes) with associated frequencies and growth/decay rates. These modes can capture underlying patterns in the data and be used for forecasting.

For stock prediction, DMD can identify cyclical patterns and trends in price movements, potentially providing insights that traditional time-series methods might miss.

## Scripts

1. **dmd_stock_prediction.py**: The main script that uses Bloomberg data for backtesting DMD on NASDAQ 100 stocks.

2. **dmd_stock_prediction_simple.py**: A simplified version that uses Yahoo Finance data instead of Bloomberg, for initial testing and understanding.

## Prerequisites

### For the Bloomberg version:
- Bloomberg Terminal installed with API access
- Bloomberg API Python library (`blpapi`)
- Python 3.6 or higher
- Required Python packages: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `scipy`, `blpapi`

### For the simplified version:
- Python 3.6 or higher
- Required Python packages: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `scipy`, `yfinance`

## Installation

1. Install the required packages:

```bash
# For the Bloomberg version
pip install pandas numpy matplotlib seaborn scikit-learn scipy
pip install --index-url=https://bcms.bloomberg.com/pip/simple/ blpapi

# For the simplified version
pip install pandas numpy matplotlib seaborn scikit-learn scipy yfinance
```

2. For the Bloomberg version, make sure your Bloomberg Terminal is running and the API service is enabled.

## Usage

### Bloomberg Version

```bash
python dmd_stock_prediction.py
```

This script will:
1. Connect to Bloomberg
2. Retrieve NASDAQ 100 constituents
3. Fetch historical price data
4. Apply DMD for prediction
5. Backtest the approach
6. Generate a comprehensive report in the `dmd_report` directory

### Simplified Version (Yahoo Finance)

```bash
python dmd_stock_prediction_simple.py
```

This script will:
1. Use a predefined list of NASDAQ 100 stocks
2. Fetch historical price data from Yahoo Finance
3. Apply DMD for prediction
4. Backtest the approach
5. Generate a comprehensive report in the `dmd_report_simple` directory

## How DMD Works for Stock Prediction

The DMD algorithm for stock prediction involves the following steps:

1. Arrange historical price data into a matrix
2. Split the data into two time-shifted matrices
3. Perform Singular Value Decomposition (SVD) on the first matrix
4. Compute the DMD operator that best maps between the matrices
5. Extract eigenvalues and eigenvectors (modes) from the DMD operator
6. Use these modes to reconstruct the dynamics and forecast future values

The rank parameter controls the number of modes used in the decomposition, acting as a form of regularization. A lower rank focuses on the most dominant patterns, potentially reducing noise.

## Customization

You can customize the scripts by modifying the following parameters:

- `train_window`: Number of days to use for training (default: 252, or 1 year of trading days)
- `predict_window`: Number of days to predict (default: 21, or 1 month of trading days)
- `rank`: Truncation rank for DMD (default: 10)

## Output

Both scripts generate a comprehensive report in HTML format, including:

- Overall performance metrics (MSE, MAE, R²)
- Distribution of metrics across stocks
- Top-performing stocks
- Example predictions
- Dominant frequencies identified by DMD
- Advantages and limitations of DMD for stock prediction
- Potential improvements

## Advantages of DMD for Stock Prediction

- Can identify underlying cyclical patterns in stock prices
- Provides interpretable modes with associated frequencies
- Works well for stocks with clear temporal patterns
- Requires relatively little historical data compared to some machine learning methods
- Can be used for both short-term and medium-term forecasting

## Limitations of DMD for Stock Prediction

- Assumes linear dynamics, which may not fully capture stock market behavior
- Sensitive to noise and outliers in the data
- Performance varies significantly across different stocks
- May struggle with abrupt regime changes or market shocks
- Requires careful selection of parameters (rank, training window)

## Potential Improvements

- Incorporate multiple features (volume, technical indicators) instead of just price
- Use adaptive rank selection based on the data characteristics
- Combine DMD with other methods (e.g., machine learning) for hybrid approaches
- Implement online DMD for continuous updating with new data
- Apply preprocessing techniques to handle non-stationarity in stock prices

## References

1. Schmid, P. J. (2010). Dynamic mode decomposition of numerical and experimental data. Journal of Fluid Mechanics, 656, 5-28.
2. Kutz, J. N., Brunton, S. L., Brunton, B. W., & Proctor, J. L. (2016). Dynamic Mode Decomposition: Data-Driven Modeling of Complex Systems. SIAM.
3. Mann, J., & Kutz, J. N. (2016). Dynamic mode decomposition for financial trading strategies. Quantitative Finance, 16(11), 1643-1655.
