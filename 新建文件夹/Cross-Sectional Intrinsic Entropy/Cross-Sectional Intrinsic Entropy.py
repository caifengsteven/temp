import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import datetime
import random
from tqdm import tqdm
import seaborn as sns

# Set plot style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("deep")
plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['font.size'] = 12

# Function to generate simulated market data
def generate_market_data(n_symbols=100, n_days=250, market_trend=0.0001, volatility=0.015, seed=42):
    """
    Generate simulated market data with OHLC prices and volumes for multiple symbols.
    
    Parameters:
    n_symbols (int): Number of symbols in the market
    n_days (int): Number of trading days
    market_trend (float): Overall market trend (drift)
    volatility (float): Base volatility for the market
    seed (int): Random seed for reproducibility
    
    Returns:
    DataFrame with simulated market data
    """
    np.random.seed(seed)
    
    # Create date range
    dates = pd.date_range(start='2021-01-01', periods=n_days, freq='B')
    
    # Generate common market factor
    market_returns = np.random.normal(market_trend, volatility, n_days)
    market_cumulative = np.exp(np.cumsum(market_returns))
    
    all_data = []
    
    for symbol_idx in range(n_symbols):
        # Generate symbol-specific parameters
        symbol_volatility = np.random.uniform(0.8, 1.5) * volatility
        symbol_beta = np.random.uniform(0.5, 1.5)
        symbol_alpha = np.random.normal(0.0001, 0.0002)
        
        # Generate returns with market correlation and specific noise
        specific_noise = np.random.normal(0, symbol_volatility, n_days)
        symbol_returns = symbol_alpha + symbol_beta * market_returns + specific_noise
        
        # Generate prices
        base_price = np.random.uniform(20, 200)
        close_prices = base_price * np.exp(np.cumsum(symbol_returns))
        
        # Generate OHLC data
        for day in range(n_days):
            day_volatility = symbol_volatility * np.random.uniform(0.8, 1.2)
            
            # Daily range around close price
            daily_range = close_prices[day] * day_volatility
            
            # Generate OHLC values
            open_price = close_prices[day] * np.exp(np.random.normal(0, day_volatility/2))
            high_price = max(open_price, close_prices[day]) + abs(np.random.normal(0, daily_range/2))
            low_price = min(open_price, close_prices[day]) - abs(np.random.normal(0, daily_range/2))
            
            # Ensure high >= open, close >= low
            high_price = max(high_price, open_price, close_prices[day])
            low_price = min(low_price, open_price, close_prices[day])
            
            # Generate volume (correlated with volatility)
            volume = int(np.random.gamma(2, 100000) * (1 + 2*day_volatility))
            
            # Add to dataset
            all_data.append({
                'Date': dates[day],
                'Symbol': f'SYM{symbol_idx+1:03d}',
                'Open': open_price,
                'High': high_price,
                'Low': low_price,
                'Close': close_prices[day],
                'Volume': volume,
                'Beta': symbol_beta,  # Store the true beta for validation
                'Alpha': symbol_alpha  # Store the true alpha for validation
            })
    
    # Convert to DataFrame
    df = pd.DataFrame(all_data)
    
    # Create a market index based on the weighted average of all symbols
    index_data = []
    symbols = df['Symbol'].unique()
    
    # Assign random weights to symbols for the index (simulating market cap weights)
    weights = np.random.exponential(1, size=len(symbols))
    weights = weights / weights.sum()
    weight_dict = {sym: w for sym, w in zip(symbols, weights)}
    
    for day in dates:
        day_data = df[df['Date'] == day]
        
        # Calculate weighted average for the index
        index_open = sum(day_data['Open'] * day_data['Symbol'].map(lambda x: weight_dict[x]))
        index_high = sum(day_data['High'] * day_data['Symbol'].map(lambda x: weight_dict[x]))
        index_low = sum(day_data['Low'] * day_data['Symbol'].map(lambda x: weight_dict[x]))
        index_close = sum(day_data['Close'] * day_data['Symbol'].map(lambda x: weight_dict[x]))
        index_volume = sum(day_data['Volume'])
        
        index_data.append({
            'Date': day,
            'Symbol': 'INDEX',
            'Open': index_open,
            'High': index_high,
            'Low': index_low,
            'Close': index_close,
            'Volume': index_volume,
            'Beta': 1.0,  # By definition, the index has a beta of 1
            'Alpha': 0.0  # By definition, the index has no alpha
        })
    
    # Add index data to the dataframe
    df = pd.concat([df, pd.DataFrame(index_data)], ignore_index=True)
    
    return df

# Implementation of Cross-Sectional Intrinsic Entropy (CSIE)
def calculate_csie(data, date):
    """
    Calculate Cross-Sectional Intrinsic Entropy for a specific date
    
    Parameters:
    data (DataFrame): DataFrame with market data
    date: The date for which to calculate CSIE
    
    Returns:
    float: CSIE value for the given date
    """
    day_data = data[data['Date'] == date].copy()
    
    # Skip if not enough symbols for that day
    if len(day_data) < 2:
        return np.nan
    
    # Calculate total traded value for the day
    day_data['TradedValue'] = day_data['Close'] * day_data['Volume']
    lambda_i = day_data['TradedValue'].sum()
    
    # Calculate ratio of individual symbols in the overall traded value
    day_data['Psi'] = day_data['TradedValue'] / lambda_i
    
    # Calculate the weight factor f_i based on the number of symbols
    m_i = len(day_data)
    alpha = 1.34  # As suggested in the paper
    f_i = (alpha - 1) / (alpha + ((m_i + 1) / (m_i - 1)))
    
    # Calculate H_i^OC component
    day_data['OC_term'] = ((day_data['Close'] / day_data['Open']) - 1) * day_data['Psi'] * np.log(day_data['Psi'])
    H_i_OC = -day_data['OC_term'].sum()
    
    # Calculate H_i^OLHC component
    day_data['OLHC_term1'] = ((day_data['High'] / day_data['Open']) - 1) * ((day_data['High'] / day_data['Close']) - 1)
    day_data['OLHC_term2'] = ((day_data['Low'] / day_data['Open']) - 1) * ((day_data['Low'] / day_data['Close']) - 1)
    day_data['OLHC_term'] = (day_data['OLHC_term1'] + day_data['OLHC_term2']) * day_data['Psi'] * np.log(day_data['Psi'])
    H_i_OLHC = -day_data['OLHC_term'].sum()
    
    # Calculate final CSIE
    H_i = (1 - f_i) * H_i_OC + f_i * H_i_OLHC
    
    return H_i

# Calculate Intrinsic Entropy (IE) for a single symbol's time series
def calculate_ie(symbol_data, window=10):
    """
    Calculate Intrinsic Entropy for a single symbol using time series data
    
    Parameters:
    symbol_data (DataFrame): DataFrame with time series data for a symbol
    window (int): Window size for the calculation
    
    Returns:
    Series: IE values for the symbol
    """
    # Sort by date to ensure time series integrity
    symbol_data = symbol_data.sort_values('Date')
    
    # Calculate IE for each day in the window
    ie_values = []
    
    for i in range(len(symbol_data) - window + 1):
        window_data = symbol_data.iloc[i:i+window]
        
        # Calculate traded value
        window_data['TradedValue'] = window_data['Close'] * window_data['Volume']
        total_value = window_data['TradedValue'].sum()
        
        # Calculate weight for each day in the window
        window_data['Psi'] = window_data['TradedValue'] / total_value
        
        # Calculate OC component for the window
        window_data['OC_term'] = ((window_data['Close'] / window_data['Open']) - 1) * window_data['Psi'] * np.log(window_data['Psi'])
        H_OC = -window_data['OC_term'].sum()
        
        # Calculate OLHC component for the window
        window_data['OLHC_term1'] = ((window_data['High'] / window_data['Open']) - 1) * ((window_data['High'] / window_data['Close']) - 1)
        window_data['OLHC_term2'] = ((window_data['Low'] / window_data['Open']) - 1) * ((window_data['Low'] / window_data['Close']) - 1)
        window_data['OLHC_term'] = (window_data['OLHC_term1'] + window_data['OLHC_term2']) * window_data['Psi'] * np.log(window_data['Psi'])
        H_OLHC = -window_data['OLHC_term'].sum()
        
        # Calculate the weight factor
        m = len(window_data)
        alpha = 1.34
        f = (alpha - 1) / (alpha + ((m + 1) / (m - 1)))
        
        # Calculate final IE
        ie = (1 - f) * H_OC + f * H_OLHC
        
        ie_values.append(ie)
    
    # Create a Series with IE values
    result_dates = symbol_data['Date'].iloc[window-1:].values
    ie_series = pd.Series(ie_values, index=result_dates)
    
    return ie_series

# Function to calculate the return rate for a given period
def calculate_return_rate(data, symbol, start_date, end_date):
    """
    Calculate the return rate for a given symbol and period
    
    Parameters:
    data (DataFrame): DataFrame with market data
    symbol (str): Symbol to calculate return for
    start_date: Start date of the period
    end_date: End date of the period
    
    Returns:
    float: Return rate in percentage
    """
    symbol_data = data[data['Symbol'] == symbol]
    
    # Get prices at start and end dates
    start_price = symbol_data[symbol_data['Date'] == start_date]['Close'].values[0]
    end_price = symbol_data[symbol_data['Date'] == end_date]['Close'].values[0]
    
    # Calculate return rate
    return_rate = (end_price - start_price) / start_price * 100
    
    return return_rate

# Function to calculate CSIE-based beta
def calculate_csie_beta(market_volatility, symbol_volatility):
    """
    Calculate CSIE-based beta between a symbol/portfolio volatility and market volatility
    
    Parameters:
    market_volatility (Series): Series with market volatility values
    symbol_volatility (Series): Series with symbol/portfolio volatility values
    
    Returns:
    float: CSIE-based beta
    """
    # Ensure both series have the same index
    common_index = market_volatility.index.intersection(symbol_volatility.index)
    market_vol = market_volatility.loc[common_index]
    symbol_vol = symbol_volatility.loc[common_index]
    
    # Calculate covariance and variance
    cov = np.cov(symbol_vol, market_vol)[0, 1]
    var = np.var(market_vol)
    
    # Calculate beta
    if var == 0:
        return np.nan
    else:
        return cov / var

# Function to implement the portfolio selection algorithm
def select_portfolio(data, market_csie, index_ie, start_date, end_date, window_size=10):
    """
    Implement the portfolio selection algorithm from the paper
    
    Parameters:
    data (DataFrame): DataFrame with market data
    market_csie (Series): Series with CSIE values for the market
    index_ie (Series): Series with IE values for the market index
    start_date: Start date of the period
    end_date: End date of the period
    window_size (int): Window size for volatility calculations
    
    Returns:
    DataFrame: Selected portfolio symbols with their metrics
    """
    # Calculate index return rate
    index_return = calculate_return_rate(data, 'INDEX', start_date, end_date)
    
    # Calculate index beta against market CSIE
    index_beta = calculate_csie_beta(market_csie, index_ie)
    
    # List to store selected symbols
    selected_symbols = []
    
    # Get all unique symbols except the index
    all_symbols = [s for s in data['Symbol'].unique() if s != 'INDEX']
    
    for symbol in tqdm(all_symbols, desc="Analyzing symbols"):
        # Check if symbol has data for the entire period
        symbol_data = data[data['Symbol'] == symbol]
        if not (start_date in symbol_data['Date'].values and end_date in symbol_data['Date'].values):
            continue
        
        # Calculate symbol return rate
        symbol_return = calculate_return_rate(data, symbol, start_date, end_date)
        
        # Calculate symbol IE
        symbol_ie = calculate_ie(symbol_data, window=window_size)
        
        # Calculate symbol beta against market CSIE
        symbol_beta = calculate_csie_beta(market_csie, symbol_ie)
        
        # Check constraints: beta <= index_beta and return >= index_return
        if not np.isnan(symbol_beta) and symbol_beta <= index_beta and symbol_beta > 0 and symbol_return >= index_return:
            selected_symbols.append({
                'Symbol': symbol,
                'Return': symbol_return,
                'Beta': symbol_beta,
                'True_Beta': symbol_data['Beta'].iloc[0]  # Store the true beta for validation
            })
    
    # Convert to DataFrame
    if selected_symbols:
        portfolio_df = pd.DataFrame(selected_symbols)
        # Sort by beta (lowest risk first)
        portfolio_df = portfolio_df.sort_values('Beta')
        return portfolio_df
    else:
        return pd.DataFrame(columns=['Symbol', 'Return', 'Beta', 'True_Beta'])

# Function to calculate portfolio CSIE
def calculate_portfolio_csie(data, symbols, dates):
    """
    Calculate CSIE for a portfolio of symbols
    
    Parameters:
    data (DataFrame): DataFrame with market data
    symbols (list): List of symbols in the portfolio
    dates (list): List of dates to calculate CSIE for
    
    Returns:
    Series: CSIE values for the portfolio
    """
    portfolio_csie = {}
    
    for date in dates:
        # Filter data for the portfolio symbols on the given date
        portfolio_data = data[(data['Symbol'].isin(symbols)) & (data['Date'] == date)]
        
        # Skip if not enough data
        if len(portfolio_data) < 2:
            continue
        
        # Calculate CSIE for the portfolio
        csie = calculate_csie(portfolio_data, date)
        portfolio_csie[date] = csie
    
    return pd.Series(portfolio_csie)

# Main function to test the strategy
def test_strategy():
    print("Generating simulated market data...")
    # Generate simulated market data
    market_data = generate_market_data(n_symbols=100, n_days=250, market_trend=0.0001, volatility=0.015)
    
    # Get list of dates
    dates = sorted(market_data['Date'].unique())
    start_date = dates[0]
    end_date = dates[-1]
    window_size = 10  # Window size for volatility calculations
    
    print("Calculating market CSIE...")
    # Calculate CSIE for each date
    csie_values = {}
    for date in tqdm(dates, desc="Calculating CSIE"):
        csie_values[date] = calculate_csie(market_data, date)
    
    market_csie = pd.Series(csie_values)
    
    print("Calculating index IE...")
    # Calculate IE for the market index
    index_data = market_data[market_data['Symbol'] == 'INDEX']
    index_ie = calculate_ie(index_data, window=window_size)
    
    print("Selecting portfolio...")
    # Select portfolio based on constraints
    portfolio = select_portfolio(market_data, market_csie, index_ie, start_date, end_date, window_size)
    
    # If portfolio is not empty, analyze it
    if not portfolio.empty:
        print(f"\nSelected {len(portfolio)} symbols that outperform the index with lower risk")
        
        # Calculate return rates for the selected portfolio
        index_return = calculate_return_rate(market_data, 'INDEX', start_date, end_date)
        avg_portfolio_return = portfolio['Return'].mean()
        
        print(f"Index return: {index_return:.2f}%")
        print(f"Average portfolio return: {avg_portfolio_return:.2f}%")
        print(f"Improvement: {avg_portfolio_return - index_return:.2f}%")
        
        # Calculate portfolio beta
        portfolio_symbols = portfolio['Symbol'].tolist()
        portfolio_csie = calculate_portfolio_csie(market_data, portfolio_symbols, dates)
        portfolio_beta = calculate_csie_beta(market_csie, portfolio_csie)
        
        index_beta = calculate_csie_beta(market_csie, index_ie)
        print(f"Index beta: {index_beta:.4f}")
        print(f"Portfolio beta: {portfolio_beta:.4f}")
        
        # Plot the top 15 symbols by lowest beta
        top_symbols = portfolio.head(min(15, len(portfolio)))
        
        plt.figure(figsize=(12, 8))
        plt.scatter(top_symbols['Return'], top_symbols['Beta'], s=100, alpha=0.7)
        
        for i, row in top_symbols.iterrows():
            plt.annotate(row['Symbol'], 
                         (row['Return'], row['Beta']),
                         xytext=(5, 5),
                         textcoords='offset points')
        
        # Add index reference lines
        plt.axvline(x=index_return, color='r', linestyle='--', label=f'Index Return: {index_return:.2f}%')
        plt.axhline(y=index_beta, color='r', linestyle='--', label=f'Index Beta: {index_beta:.4f}')
        
        plt.title('Selected Portfolio: Return vs Beta')
        plt.xlabel('Return Rate (%)')
        plt.ylabel('Beta (relative to market)')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig('portfolio_return_vs_beta.png')
        plt.show()
        
        # Validation plot: Compare estimated betas vs true betas
        plt.figure(figsize=(10, 6))
        plt.scatter(portfolio['Beta'], portfolio['True_Beta'], alpha=0.5)
        plt.plot([0, max(portfolio['Beta'].max(), portfolio['True_Beta'].max())], 
                 [0, max(portfolio['Beta'].max(), portfolio['True_Beta'].max())], 'r--')
        plt.title('Estimated Beta vs True Beta')
        plt.xlabel('Estimated Beta (CSIE-based)')
        plt.ylabel('True Beta (from simulation)')
        plt.grid(True)
        plt.savefig('beta_validation.png')
        plt.show()
        
        # Return distribution comparison
        plt.figure(figsize=(12, 6))
        plt.hist(portfolio['Return'], bins=20, alpha=0.5, label='Portfolio Returns')
        
        # Calculate returns for all symbols
        all_returns = []
        for symbol in market_data['Symbol'].unique():
            if symbol != 'INDEX':
                symbol_data = market_data[market_data['Symbol'] == symbol]
                if start_date in symbol_data['Date'].values and end_date in symbol_data['Date'].values:
                    ret = calculate_return_rate(market_data, symbol, start_date, end_date)
                    all_returns.append(ret)
        
        plt.hist(all_returns, bins=20, alpha=0.5, label='All Symbols Returns')
        plt.axvline(x=index_return, color='r', linestyle='--', label=f'Index Return: {index_return:.2f}%')
        plt.axvline(x=avg_portfolio_return, color='g', linestyle='--', label=f'Avg Portfolio Return: {avg_portfolio_return:.2f}%')
        
        plt.title('Return Distribution Comparison')
        plt.xlabel('Return Rate (%)')
        plt.ylabel('Count')
        plt.legend()
        plt.grid(True)
        plt.savefig('return_distribution.png')
        plt.show()
        
        return portfolio, index_return, index_beta
    else:
        print("No symbols found that meet the criteria.")
        return None, None, None

# Run the test
portfolio, index_return, index_beta = test_strategy()