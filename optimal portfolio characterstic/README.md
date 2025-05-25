# Optimal Characteristic Portfolios

This repository implements the strategy described in the paper "Optimal Characteristic Portfolios" by Richard McGee and Jose Olmo. The implementation uses Python and simulated data to construct optimal characteristic-based portfolios.

## Overview

The paper proposes a new method for constructing characteristic-sorted portfolios that:

1. Makes no ex-ante assumptions about the relationship between characteristics and returns
2. Does not require manual selection of percentile breakpoints or portfolio weighting schemes
3. Derives portfolio weights directly from data through maximizing a Mean-Variance objective function
4. Uses non-parametric methods to estimate mean and variance from the cross-section of assets

## Requirements

- Python 3.7+
- Required Python packages:
  - pandas
  - numpy
  - matplotlib
  - scipy

## Installation

1. Clone this repository
2. Install the required packages:
   ```
   pip install pandas numpy matplotlib scipy
   ```

## File Structure

- `bloomberg_data.py`: Module for retrieving data from Bloomberg (not used in current implementation)
- `nonparametric.py`: Implementation of non-parametric estimation methods
- `portfolio.py`: Portfolio construction and backtesting
- `stock_universe.py`: Module for generating simulated stock universe and characteristics
- `main.py`: Main script to run the strategy
- `test_bloomberg.py`: Test script for Bloomberg connection (optional)

## Usage

Run the main script:
```
python main.py
```

This will:
1. Generate a simulated stock universe
2. Create simulated characteristics (size, value, momentum)
3. Generate simulated returns data
4. Construct optimal characteristic portfolios
5. Backtest the strategies from 2010 to 2020
6. Generate performance metrics and plots

## Customization

You can modify the parameters in the `main.py` file to customize the strategy:

- `start_date` and `end_date`: Time period for the backtest
- `num_stocks`: Number of stocks in the simulated universe
- `gamma`: Risk aversion parameter
- `rebalance_freq`: Rebalancing frequency ('monthly', 'quarterly', 'annual')
- `lookback_periods`: Number of periods to use for estimating the relationship

## Implementation Details

### Simulated Data Generation

The implementation uses simulated data:
- Stock universe: A set of stocks with realistic ticker names
- Characteristics: Size (market cap), value (book-to-market), and momentum values
- Returns: Random returns with some correlation to the characteristics

### Non-parametric Estimation

The strategy uses the Nadaraya-Watson estimator to non-parametrically estimate the conditional mean and variance of returns given a characteristic. The optimal bandwidth is selected using Silverman's rule of thumb.

### Portfolio Construction

Portfolio weights are determined by maximizing a mean-variance objective function:

```
w*(z) = μ(z) / (γ * σ²(z))
```

where:
- `w*(z)` is the optimal weight for a characteristic value `z`
- `μ(z)` is the conditional expected return
- `σ²(z)` is the conditional variance
- `γ` is the risk aversion parameter

The weights are then normalized to satisfy dollar-neutrality constraints.

## Results

The strategy generates three characteristic portfolios:

1. **Size Portfolio**: Based on market capitalization
2. **Value Portfolio**: Based on book-to-market ratio
3. **Momentum Portfolio**: Based on past 12-month returns (skipping the most recent month)

Performance metrics include:
- Annualized return
- Annualized volatility
- Sharpe ratio
- Maximum drawdown
- Win rate
- t-statistic

Sample results from the simulated data:

| Portfolio | Annualized Return | Volatility | Sharpe Ratio | Max Drawdown | Win Rate |
|-----------|-------------------|------------|--------------|--------------|----------|
| Size      | -2.06%            | 14.22%     | -0.30        | -45.55%      | 45.45%   |
| Value     | -0.92%            | 8.30%      | -0.41        | -24.81%      | 49.24%   |
| Momentum  | 5.60%             | 11.42%     | 0.29         | -16.81%      | 56.06%   |

The correlation matrix shows that the strategies are relatively uncorrelated, with momentum having a slight positive correlation with value.

## Using Real Data

If you have access to Bloomberg API, you can modify the code to use real data:

1. Update the `main.py` file to use the `bloomberg_data.py` module
2. Fix any API-specific issues in the Bloomberg data retriever
3. Run the `test_bloomberg.py` script to test your Bloomberg connection

## References

McGee, R. J., & Olmo, J. (2022). Optimal characteristic portfolios. Quantitative Finance, 22(10), 1853-1870.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
