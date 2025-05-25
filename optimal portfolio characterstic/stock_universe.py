"""
Stock Universe Generator for Optimal Characteristic Portfolios

This module generates a simulated stock universe for testing the optimal
characteristic portfolios strategy when index constituent data is not available.
"""

import pandas as pd
import numpy as np
import datetime as dt


def generate_stock_universe(start_date, end_date, num_stocks=100, seed=42):
    """
    Generate a simulated stock universe for testing

    Parameters:
    -----------
    start_date : str or datetime
        Start date for the simulation
    end_date : str or datetime
        End date for the simulation
    num_stocks : int
        Number of stocks in the universe
    seed : int
        Random seed for reproducibility

    Returns:
    --------
    dict
        Dictionary with dates as keys and lists of tickers as values
    """
    np.random.seed(seed)

    # Convert dates to datetime if they're strings
    if isinstance(start_date, str):
        start_date = dt.datetime.strptime(start_date, '%Y-%m-%d')
    if isinstance(end_date, str):
        end_date = dt.datetime.strptime(end_date, '%Y-%m-%d')

    # Generate monthly dates
    dates = pd.date_range(start=start_date, end=end_date, freq='M')

    # Generate stock tickers
    # Use real tickers for major stocks and generate additional ones if needed
    real_tickers = [
        "AAPL US Equity", "MSFT US Equity", "AMZN US Equity", "GOOGL US Equity",
        "META US Equity", "TSLA US Equity", "BRK/B US Equity", "JPM US Equity",
        "JNJ US Equity", "V US Equity", "PG US Equity", "UNH US Equity",
        "HD US Equity", "BAC US Equity", "MA US Equity", "DIS US Equity",
        "NVDA US Equity", "PYPL US Equity", "INTC US Equity", "VZ US Equity",
        "ADBE US Equity", "CMCSA US Equity", "NFLX US Equity", "CRM US Equity",
        "KO US Equity", "PEP US Equity", "ABT US Equity", "MRK US Equity",
        "WMT US Equity", "CSCO US Equity", "T US Equity", "PFE US Equity",
        "NKE US Equity", "TMO US Equity", "ABBV US Equity", "AVGO US Equity",
        "ACN US Equity", "TXN US Equity", "COST US Equity", "MCD US Equity"
    ]

    if num_stocks > len(real_tickers):
        # Generate additional tickers if needed
        additional_tickers = [f"STOCK{i} US Equity" for i in range(1, num_stocks - len(real_tickers) + 1)]
        all_tickers = real_tickers + additional_tickers
    else:
        all_tickers = real_tickers[:num_stocks]

    # Create universe with some randomness to simulate changes in index composition
    universe = {}

    # Start with all tickers
    current_universe = all_tickers.copy()

    for date in dates:
        # With small probability, replace some stocks
        if np.random.random() < 0.1:  # 10% chance of change each month
            # Number of stocks to replace (1-3)
            num_replace = np.random.randint(1, 4)

            # Select stocks to remove
            to_remove = np.random.choice(current_universe, num_replace, replace=False)

            # Select stocks to add from those not in current universe
            available_to_add = [t for t in all_tickers if t not in current_universe]
            if available_to_add:
                to_add = np.random.choice(available_to_add, min(num_replace, len(available_to_add)), replace=False)

                # Update universe
                for ticker in to_remove:
                    current_universe.remove(ticker)
                for ticker in to_add:
                    current_universe.append(ticker)

        universe[date] = current_universe.copy()

    return universe


def generate_characteristics(universe, characteristic_type='size', seed=42):
    """
    Generate simulated characteristics for a stock universe

    Parameters:
    -----------
    universe : dict
        Dictionary with dates as keys and lists of tickers as values
    characteristic_type : str
        Type of characteristic to generate ('size', 'value', 'momentum')
    seed : int
        Random seed for reproducibility

    Returns:
    --------
    dict
        Dictionary with dates as keys and DataFrames of characteristics as values
    """
    np.random.seed(seed)

    characteristics = {}

    # For each date in the universe
    for date, tickers in universe.items():
        if characteristic_type == 'size':
            # Generate market caps with log-normal distribution
            # Large caps have higher market caps
            market_caps = {}
            for ticker in tickers:
                if 'AAPL' in ticker or 'MSFT' in ticker or 'AMZN' in ticker:
                    # Large caps
                    market_caps[ticker] = np.random.lognormal(mean=25, sigma=0.5)
                elif 'GOOGL' in ticker or 'META' in ticker or 'TSLA' in ticker:
                    # Large caps
                    market_caps[ticker] = np.random.lognormal(mean=24.5, sigma=0.5)
                elif ticker in ['JPM US Equity', 'JNJ US Equity', 'V US Equity']:
                    # Mid-large caps
                    market_caps[ticker] = np.random.lognormal(mean=24, sigma=0.5)
                else:
                    # Other stocks
                    market_caps[ticker] = np.random.lognormal(mean=23, sigma=1.0)

            characteristics[date] = pd.DataFrame({
                'size': market_caps
            })

        elif characteristic_type == 'value':
            # Generate book-to-market ratios
            # Value stocks have higher B/M ratios
            bm_ratios = {}
            for ticker in tickers:
                if 'AAPL' in ticker or 'MSFT' in ticker or 'AMZN' in ticker:
                    # Growth stocks (low B/M)
                    bm_ratios[ticker] = np.random.uniform(0.1, 0.3)
                elif 'GOOGL' in ticker or 'META' in ticker or 'TSLA' in ticker:
                    # Growth stocks (low B/M)
                    bm_ratios[ticker] = np.random.uniform(0.2, 0.4)
                elif ticker in ['JPM US Equity', 'BAC US Equity', 'WMT US Equity']:
                    # Value stocks (high B/M)
                    bm_ratios[ticker] = np.random.uniform(0.7, 1.2)
                else:
                    # Other stocks
                    bm_ratios[ticker] = np.random.uniform(0.3, 0.8)

            characteristics[date] = pd.DataFrame({
                'value': bm_ratios
            })

        elif characteristic_type == 'momentum':
            # Generate momentum values
            # Momentum stocks have higher past returns
            momentum_values = {}
            for ticker in tickers:
                if 'AAPL' in ticker or 'MSFT' in ticker or 'AMZN' in ticker:
                    # High momentum
                    momentum_values[ticker] = np.random.uniform(0.1, 0.4)
                elif 'GOOGL' in ticker or 'META' in ticker or 'TSLA' in ticker:
                    # High momentum
                    momentum_values[ticker] = np.random.uniform(0.15, 0.5)
                elif ticker in ['T US Equity', 'KO US Equity', 'PEP US Equity']:
                    # Low momentum
                    momentum_values[ticker] = np.random.uniform(-0.1, 0.1)
                else:
                    # Other stocks
                    momentum_values[ticker] = np.random.uniform(-0.2, 0.3)

            characteristics[date] = pd.DataFrame({
                'momentum': momentum_values
            })

    return characteristics


# Example usage
if __name__ == "__main__":
    # Generate stock universe
    start_date = '2010-01-01'
    end_date = '2020-12-31'
    universe = generate_stock_universe(start_date, end_date, num_stocks=100)

    # Print sample
    sample_date = list(universe.keys())[0]
    print(f"Sample universe for {sample_date.strftime('%Y-%m-%d')}:")
    print(f"Number of stocks: {len(universe[sample_date])}")
    print(f"First 5 stocks: {universe[sample_date][:5]}")

    # Generate characteristics
    size_chars = generate_characteristics(universe, characteristic_type='size')
    value_chars = generate_characteristics(universe, characteristic_type='value')
    momentum_chars = generate_characteristics(universe, characteristic_type='momentum')

    # Print sample
    print("\nSample size characteristics:")
    print(size_chars[sample_date].head())

    print("\nSample value characteristics:")
    print(value_chars[sample_date].head())

    print("\nSample momentum characteristics:")
    print(momentum_chars[sample_date].head())
