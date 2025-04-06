# DMD Stock Analysis with Bloomberg Data

This repository contains Python scripts for analyzing stocks using Dynamic Mode Decomposition (DMD) with Bloomberg data. The scripts provide tools for analyzing individual stocks or batches of stocks, and generating comprehensive reports.

## What is DMD?

Dynamic Mode Decomposition (DMD) is a data-driven method for analyzing complex dynamical systems. It was originally developed for fluid dynamics but has found applications in various fields, including finance.

DMD works by decomposing time-series data into spatial-temporal coherent structures (modes) with associated frequencies and growth/decay rates. These modes can capture underlying patterns in the data and be used for forecasting.

For stock analysis, DMD can identify cyclical patterns and trends in price movements, potentially providing insights that traditional time-series methods might miss.

## Scripts

1. **dmd_bloomberg_analysis.py**: Analyzes a single stock using DMD with Bloomberg data.

2. **dmd_bloomberg_batch.py**: Analyzes multiple stocks from a file and generates a summary report.

## Prerequisites

- Bloomberg Terminal installed with API access
- Bloomberg API Python library (`blpapi`)
- Python 3.6 or higher
- Required Python packages: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scipy`, `blpapi`

## Installation

1. Install the required packages:

```bash
pip install pandas numpy matplotlib seaborn scipy
pip install --index-url=https://bcms.bloomberg.com/pip/simple/ blpapi
```

2. Make sure your Bloomberg Terminal is running and the API service is enabled.

## Usage

### Analyzing a Single Stock

```bash
python dmd_bloomberg_analysis.py --security "AAPL US Equity" --field "PX_LAST" --days 730 --rank 10
```

Parameters:
- `--security`: Bloomberg security identifier (default: "AAPL US Equity")
- `--field`: Bloomberg field to analyze (default: "PX_LAST")
- `--days`: Number of days of historical data to use (default: 730, or 2 years)
- `--rank`: Truncation rank for DMD (default: 10)
- `--train-ratio`: Ratio of data to use for training (default: 0.8)
- `--output-dir`: Directory to save results (default: "dmd_analysis")

### Analyzing Multiple Stocks

```bash
python dmd_bloomberg_batch.py --instruments "instruments.txt" --field "PX_LAST" --days 730 --rank 10
```

Parameters:
- `--instruments`: Path to instruments file (default: "instruments.txt")
- `--field`: Bloomberg field to analyze (default: "PX_LAST")
- `--days`: Number of days of historical data to use (default: 730, or 2 years)
- `--rank`: Truncation rank for DMD (default: 10)
- `--train-ratio`: Ratio of data to use for training (default: 0.8)
- `--output-dir`: Directory to save results (default: "dmd_analysis")

## Instruments File

The instruments file should contain one Bloomberg security identifier per line, for example:

```
AAPL US Equity
MSFT US Equity
AMZN US Equity
GOOGL US Equity
META US Equity
```

If the file doesn't exist, a sample file will be created.

## Output

The scripts generate the following outputs in the specified directory:

### For Each Stock

- Raw data CSV file
- Results JSON file with metrics and dominant frequencies/periods
- Prediction plot showing actual vs. predicted prices
- Frequency plot showing dominant frequencies
- Period plot showing dominant periods

### Summary Report (Batch Mode)

- Summary CSV file with metrics for all stocks
- Summary plots showing distributions of metrics
- HTML report with interactive links to individual stock analyses

## How DMD Works for Stock Analysis

The DMD algorithm for stock analysis involves the following steps:

1. Arrange historical price data into a matrix
2. Split the data into two time-shifted matrices
3. Perform Singular Value Decomposition (SVD) on the first matrix
4. Compute the DMD operator that best maps between the matrices
5. Extract eigenvalues and eigenvectors (modes) from the DMD operator
6. Use these modes to reconstruct the dynamics and forecast future values

The rank parameter controls the number of modes used in the decomposition, acting as a form of regularization. A lower rank focuses on the most dominant patterns, potentially reducing noise.

## Interpreting the Results

- **MSE/MAE**: Lower values indicate better prediction accuracy
- **Dominant Frequencies**: Represent the cycles detected in the price data (in cycles per day)
- **Dominant Periods**: The inverse of frequencies, representing the length of cycles (in days)
- **Prediction Plots**: Visual comparison of actual vs. predicted prices

## Advantages of DMD for Stock Analysis

- Can identify underlying cyclical patterns in stock prices
- Provides interpretable modes with associated frequencies
- Works well for stocks with clear temporal patterns
- Requires relatively little historical data compared to some machine learning methods
- Can be used for both short-term and medium-term forecasting

## Limitations of DMD for Stock Analysis

- Assumes linear dynamics, which may not fully capture stock market behavior
- Sensitive to noise and outliers in the data
- Performance varies significantly across different stocks
- May struggle with abrupt regime changes or market shocks
- Requires careful selection of parameters (rank, training window)

## References

1. Schmid, P. J. (2010). Dynamic mode decomposition of numerical and experimental data. Journal of Fluid Mechanics, 656, 5-28.
2. Kutz, J. N., Brunton, S. L., Brunton, B. W., & Proctor, J. L. (2016). Dynamic Mode Decomposition: Data-Driven Modeling of Complex Systems. SIAM.
3. Mann, J., & Kutz, J. N. (2016). Dynamic mode decomposition for financial trading strategies. Quantitative Finance, 16(11), 1643-1655.
