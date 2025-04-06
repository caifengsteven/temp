# Bloomberg VPIN Calculator

This script connects to Bloomberg, reads instruments from a file, calculates VPIN (Volume-Synchronized Probability of Informed Trading) for each instrument, and displays the results in the terminal.

## What is VPIN?

VPIN (Volume-Synchronized Probability of Informed Trading) is a metric developed by Easley, López de Prado, and O'Hara to measure the probability of informed trading in financial markets. It is designed to detect order flow toxicity and can be used as an early warning signal for market volatility.

## Prerequisites

- Bloomberg Terminal installed with API access
- Bloomberg API Python library (`blpapi`)
- Python 3.6 or higher
- Required Python packages: `pandas`, `numpy`, `blpapi`

## Installation

1. Install the required packages:

```bash
pip install pandas numpy
pip install --index-url=https://bcms.bloomberg.com/pip/simple/ blpapi
```

2. Make sure your Bloomberg Terminal is running and the API service is enabled.

## Usage

1. Create a file named `instruments.txt` with one Bloomberg security identifier per line, for example:

```
AAPL US Equity
MSFT US Equity
AMZN US Equity
```

2. Run the script:

```bash
python bloomberg_vpin_calculator.py
```

3. The script will:
   - Connect to Bloomberg
   - Read instruments from `instruments.txt`
   - Fetch intraday bar data for each instrument
   - Calculate VPIN for each instrument
   - Display the VPIN values in the terminal

## VPIN Calculation Methodology

The VPIN calculation in this script follows these steps:

1. **Data Collection**: Retrieve intraday bar data from Bloomberg.
2. **Bulk Volume Classification**: Classify each bar's volume as buy or sell using price changes.
3. **Bucket Creation**: Divide the total volume into equal-sized buckets.
4. **Volume Imbalance**: Calculate the absolute difference between buy and sell volume in each bucket.
5. **VPIN Calculation**: Calculate VPIN as the sum of volume imbalances divided by total volume over a rolling window.

## Parameters

The script uses the following default parameters for VPIN calculation:

- `num_buckets`: 50 (Number of buckets for VPIN calculation)
- `window_size`: 50 (Window size for VPIN calculation)
- `sigma_multiplier`: 1.0 (Multiplier for standard deviation in bulk classification)

These parameters can be adjusted in the `BloombergVPINCalculator` class initialization.

## Interpretation

VPIN values range from 0 to 1:
- Values closer to 0 indicate low probability of informed trading
- Values closer to 1 indicate high probability of informed trading

Higher VPIN values may signal increased market toxicity and potential for volatility.

## Troubleshooting

- Make sure your Bloomberg Terminal is running
- Verify that you have API access enabled
- Check that the instruments in your file have valid Bloomberg identifiers
- If you encounter connection issues, verify your network settings and Bloomberg configuration
