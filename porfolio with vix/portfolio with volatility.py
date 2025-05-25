import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pdblp
from arch import arch_model
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set plot style
plt.style.use('seaborn-darkgrid')

# Connect to Bloomberg
try:
    con = pdblp.BCon(timeout=60000)
    con.start()
    print("Connected to Bloomberg")
except Exception as e:
    print(f"Error connecting to Bloomberg: {e}")
    print("Will use simulated data instead")
    con = None

# Function to fetch data from Bloomberg with error handling
def fetch_bloomberg_data(tickers, start_date, end_date, field='PX_LAST'):
    """Fetch price data from Bloomberg for the given tickers and date range"""
    if con is None:
        return simulate_data(tickers, start_date, end_date)
    
    try:
        print(f"Fetching {field} for {tickers} from Bloomberg...")
        
        # Format dates for Bloomberg query
        start_date_fmt = start_date.replace('-', '')
        end_date_fmt = end_date.replace('-', '')
        
        # Request data from Bloomberg
        data = con.bdh(tickers=tickers, 
                       flds=[field], 
                       start_date=start_date_fmt, 
                       end_date=end_date_fmt)
        
        # Create DataFrame for prices
        prices_df = pd.DataFrame(index=pd.date_range(start=start_date, end=end_date, freq='B'))
        
        # Extract data for each ticker
        for ticker in tickers:
            try:
                ticker_data = data.xs(ticker, axis=1, level=0)
                if field in ticker_data.columns:
                    prices_df[ticker] = ticker_data[field]
                    print(f"Downloaded {ticker} data: {len(ticker_data)} rows")
            except Exception as ex:
                print(f"Error extracting data for {ticker}: {ex}")
        
        # Drop rows with missing values
        prices_df = prices_df.dropna()
        
        print(f"Successfully downloaded data with shape: {prices_df.shape}")
        return prices_df
        
    except Exception as e:
        print(f"Error retrieving data from Bloomberg: {e}")
        print("Using simulated data instead.")
        return simulate_data(tickers, start_date, end_date)

# Function to simulate data if Bloomberg isn't available
def simulate_data(tickers, start_date, end_date):
    """Simulate market data for testing purposes"""
    print("Simulating market data...")
    
    # Convert dates to datetime
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    
    # Generate date range
    date_range = pd.date_range(start=start_date, end=end_date, freq='B')
    
    # Create DataFrame
    data = pd.DataFrame(index=date_range)
    
    # Seed for reproducibility
    np.random.seed(42)
    
    # Simulate S&P 500
    spx_returns = np.random.normal(0.0005, 0.01, len(date_range))
    spx_price = 100 * np.cumprod(1 + spx_returns)
    data['SPX Index'] = spx_price
    
    # Simulate VIX with mean reversion and negative correlation to S&P 500
    vix_base = 20 + np.zeros(len(date_range))
    for i in range(1, len(date_range)):
        # Mean reversion component
        mean_reversion = 0.1 * (20 - vix_base[i-1])
        # Negative correlation with S&P 500 component
        sp_correlation = -50 * spx_returns[i]
        # Random component
        random_component = np.random.normal(0, 1.5)
        # Combine components
        vix_base[i] = vix_base[i-1] + mean_reversion + sp_correlation + random_component
        # Ensure VIX stays positive
        vix_base[i] = max(vix_base[i], 9)
    
    data['VIX Index'] = vix_base
    
    # Simulate VIX futures with contango/backwardation
    for i, ticker in enumerate([t for t in tickers if 'UX' in t and 'Index' not in t]):
        # For futures, add a term premium that varies with VIX level
        months_forward = i + 1
        term_premium = np.zeros(len(date_range))
        
        for j in range(len(date_range)):
            # In normal markets (VIX < 20), futures trade at premium (contango)
            # In stressed markets (VIX > 30), futures trade at discount (backwardation)
            if vix_base[j] < 20:
                term_premium[j] = months_forward * 0.5  # contango
            elif vix_base[j] > 30:
                term_premium[j] = months_forward * -0.8  # backwardation
            else:
                # Linear interpolation between 20 and 30
                slope = (months_forward * -0.8 - months_forward * 0.5) / (30 - 20)
                term_premium[j] = months_forward * 0.5 + slope * (vix_base[j] - 20)
        
        data[ticker] = vix_base + term_premium
    
    print(f"Simulated data for {len(tickers)} tickers over {len(date_range)} days.")
    return data

# Function to properly calculate VIX30 based on the paper's methodology
def calculate_vix30(vix_futures_data, roll_calendar=None):
    """
    Calculate the 30-day constant maturity VIX futures following the paper's methodology
    """
    print("Calculating VIX30...")
    data = vix_futures_data.copy()
    
    # Rename columns to match expected names
    if 'UX1 Index' in data.columns:
        data['VIX1'] = data['UX1 Index']
    if 'UX2 Index' in data.columns:
        data['VIX2'] = data['UX2 Index']
    
    # Simple approximation of VIX30 using weighted average of front and second month
    # Assuming linear weighting with 21 trading days in a month
    days_per_month = 21
    
    data['day_of_month'] = data.index.day
    data['weight1'] = (days_per_month - (data['day_of_month'] % days_per_month)) / days_per_month
    data['weight1'] = data['weight1'].clip(0, 1)  # Ensure weights are between 0 and 1
    data['weight2'] = 1 - data['weight1']
    
    # Calculate VIX30
    data['VIX30'] = data['weight1'] * data['VIX1'] + data['weight2'] * data['VIX2']
    
    print("VIX30 calculation complete")
    return data['VIX30']

# More efficient GARCH volatility estimation
def estimate_volatility(returns, method='rolling', window=252):
    """
    Estimate volatility using different methods
    
    Parameters:
    -----------
    returns: Series
        Return series to estimate volatility for
    method: str
        'rolling' for rolling window, 'ewma' for exponentially weighted, 'garch' for GARCH(1,1)
    window: int
        Size of rolling/EWMA window
    
    Returns:
    --------
    Series with annualized volatility estimates
    """
    print(f"Estimating volatility using {method} method...")
    
    if method == 'rolling':
        # Simple rolling standard deviation
        vol = returns.rolling(window=window, min_periods=min(window//4, 20)).std() * np.sqrt(252)
    elif method == 'ewma':
        # Exponentially weighted moving average
        vol = returns.ewm(span=window).std() * np.sqrt(252)
    elif method == 'garch':
        # GARCH(1,1) - only estimate on a sample to speed things up
        vol = returns * np.nan
        
        # First fill with EWMA estimates
        vol_ewma = returns.ewm(span=window).std() * np.sqrt(252)
        vol = vol_ewma.copy()
        
        # Sample every 20 days for GARCH estimation to speed up computation
        sample_indices = np.arange(window, len(returns), 20)
        
        for i in tqdm(sample_indices, desc="GARCH estimation"):
            try:
                data = returns.iloc[max(0, i-window):i]
                model = arch_model(data, vol='Garch', p=1, q=1, rescale=False)
                result = model.fit(disp='off', show_warning=False)
                forecast = result.forecast(horizon=20)
                vol_forecast = np.sqrt(forecast.variance.values) * np.sqrt(252)
                
                # Fill the next 20 days (or remaining days) with GARCH forecasts
                end_idx = min(i+20, len(returns))
                vol.iloc[i:end_idx] = vol_forecast.flatten()[:end_idx-i]
            except Exception as e:
                # If GARCH fails, keep the EWMA estimate
                continue
    else:
        raise ValueError(f"Unknown volatility method: {method}")
    
    # Fill any remaining NaN values with the EWMA method
    if vol.isna().any():
        vol_ewma = returns.ewm(span=window).std() * np.sqrt(252)
        vol = vol.fillna(vol_ewma)
    
    # Ensure minimum volatility
    vol = vol.clip(lower=0.01)
    
    print("Volatility estimation complete")
    return vol

# Implement the Enhanced Portfolio strategy with improvements
def enhanced_portfolio_strategy(spx_prices, vix_index, vix1_prices, vix2_prices, vix_history=None):
    """
    Implement the Enhanced Portfolio strategy as described in the paper with improvements
    """
    print("Building Enhanced Portfolio strategy...")
    
    # Ensure all inputs are aligned
    data = pd.DataFrame({
        'SPX': spx_prices,
        'VIX': vix_index,
        'UX1': vix1_prices,
        'UX2': vix2_prices
    })
    
    # Drop rows with missing data
    data = data.dropna()
    
    # Calculate returns
    data['SPX_returns'] = data['SPX'].pct_change()
    
    # Calculate VIX30 using simplified method
    vix_futures = data[['UX1', 'UX2']].copy()
    vix_futures.columns = ['UX1 Index', 'UX2 Index']  # Rename for the function
    data['VIX30'] = calculate_vix30(vix_futures)
    data['VIX30_returns'] = data['VIX30'].pct_change()
    
    # Calculate VIX premium
    data['VIX_premium'] = data['VIX30'] - data['VIX']
    
    # If no VIX history is provided, use the data we have
    if vix_history is None:
        vix_history = data['VIX']
    
    # Calculate VIX decile (for the rule not to short VIX when in bottom decile)
    vix_decile_10 = vix_history.quantile(0.1)
    print(f"VIX bottom decile threshold: {vix_decile_10:.2f}")
    
    # Fast volatility estimation - use rolling window with EWMA to speed things up
    print("Estimating SPX volatility...")
    data['SPX_vol'] = estimate_volatility(data['SPX_returns'].dropna(), method='ewma')
    
    print("Estimating VIX30 volatility...")
    data['VIX30_vol'] = estimate_volatility(data['VIX30_returns'].dropna(), method='ewma')
    
    # Create portfolio allocations (with 1-day lag as in the paper)
    print("Determining allocations...")
    data['VIX_premium_sign'] = data['VIX_premium'].shift(1).apply(lambda x: 1 if x > 0 else -1)
    data['in_bottom_decile'] = data['VIX'].shift(1) < vix_decile_10
    
    # Calculate risk parity weights
    data['SPX_weight'] = data['VIX30_vol'].shift(1) / (data['SPX_vol'].shift(1) + data['VIX30_vol'].shift(1))
    data['VIX30_weight'] = data['SPX_vol'].shift(1) / (data['SPX_vol'].shift(1) + data['VIX30_vol'].shift(1))
    
    # Apply the rules from the paper
    # Initialize VIX30 allocation
    data['VIX30_allocation'] = 0.0
    
    # Short VIX30 when premium is positive and VIX not in bottom decile
    short_condition = (data['VIX_premium_sign'] > 0) & (~data['in_bottom_decile'])
    data.loc[short_condition, 'VIX30_allocation'] = -data.loc[short_condition, 'VIX30_weight']
    
    # Long VIX30 when premium is negative
    long_condition = data['VIX_premium_sign'] < 0
    data.loc[long_condition, 'VIX30_allocation'] = data.loc[long_condition, 'VIX30_weight']
    
    # Corresponding SPX allocation
    data['SPX_allocation'] = 1 - abs(data['VIX30_allocation'])
    
    # Calculate strategy returns
    print("Calculating strategy returns...")
    data['SPX_contribution'] = data['SPX_allocation'] * data['SPX_returns']
    data['VIX30_contribution'] = data['VIX30_allocation'] * data['VIX30_returns']
    data['portfolio_return'] = data['SPX_contribution'] + data['VIX30_contribution']
    
    # Track long and short VIX30 contributions separately
    data['long_VIX30_contrib'] = np.where(data['VIX30_allocation'] > 0, 
                                         data['VIX30_allocation'] * data['VIX30_returns'], 
                                         0)
    
    data['short_VIX30_contrib'] = np.where(data['VIX30_allocation'] < 0, 
                                          data['VIX30_allocation'] * data['VIX30_returns'],
                                          0)
    
    # Calculate cumulative returns
    data['portfolio_cumulative'] = (1 + data['portfolio_return']).cumprod()
    data['SPX_cumulative'] = (1 + data['SPX_returns']).cumprod()
    
    # Calculate percentage of time in long/short VIX
    data['in_long_VIX'] = (data['VIX30_allocation'] > 0).astype(int)
    data['in_short_VIX'] = (data['VIX30_allocation'] < 0).astype(int)
    data['in_flat_VIX'] = ((data['VIX30_allocation'] == 0) & (data['VIX_premium_sign'] > 0)).astype(int)
    
    print("Strategy construction complete")
    return data

# Function to calculate performance metrics
def calculate_performance_metrics(returns, rfr=0.0):
    """Calculate various performance metrics for a returns series"""
    # Convert to monthly if daily
    if len(returns) > 100:  # Assume it's daily if more than 100 observations
        monthly_returns = returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
    else:
        monthly_returns = returns
    
    ann_factor = 12  # Monthly to annual
    
    # Basic metrics
    avg_return = monthly_returns.mean()
    volatility = monthly_returns.std()
    ann_return = (1 + avg_return) ** ann_factor - 1
    ann_vol = volatility * np.sqrt(ann_factor)
    
    # Risk-adjusted metrics
    sharpe_ratio = (avg_return - rfr/12) / volatility if volatility > 0 else 0
    downside_deviation = monthly_returns[monthly_returns < 0].std()
    sortino_ratio = (avg_return - rfr/12) / downside_deviation if downside_deviation > 0 else 0
    
    # Drawdown analysis
    cumulative = (1 + monthly_returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative / running_max) - 1
    max_drawdown = drawdown.min()
    
    # Construct results
    metrics = {
        'Average Monthly Return': avg_return,
        'Monthly Volatility': volatility,
        'Annualized Return': ann_return,
        'Annualized Volatility': ann_vol,
        'Sharpe Ratio': sharpe_ratio,
        'Sortino Ratio': sortino_ratio,
        'Maximum Drawdown': max_drawdown,
        'Skewness': monthly_returns.skew(),
        'Kurtosis': monthly_returns.kurtosis()
    }
    
    return pd.Series(metrics)

# Calculate performance metrics by year
def calculate_annual_performance(strategy_data):
    """Calculate yearly performance for components of the strategy"""
    # Group by year
    yearly_data = strategy_data.groupby(strategy_data.index.year)
    
    # Calculate annual returns for each component
    annual_returns = pd.DataFrame({
        'S&P 500': yearly_data['SPX_returns'].apply(lambda x: (1 + x).prod() - 1),
        'Enhanced': yearly_data['portfolio_return'].apply(lambda x: (1 + x).prod() - 1),
        'S&P Contribution': yearly_data['SPX_contribution'].apply(lambda x: (1 + x).prod() - 1),
        'Long VIX30': yearly_data['long_VIX30_contrib'].apply(lambda x: (1 + x).sum()),
        'Short VIX30': yearly_data['short_VIX30_contrib'].apply(lambda x: (1 + x).sum()),
        'Long VIX30%': yearly_data['in_long_VIX'].mean() * 100
    })
    
    return annual_returns

# Modified main function to run the strategy with performance improvements
def main():
    print("Starting Enhanced Portfolio strategy backtest...")
    
    # Define date range (matching the paper's timeframe)
    start_date = '2007-01-01'  # Starting earlier to have data for volatility estimation
    end_date = '2018-12-31'
    
    # Define tickers to fetch (using UX1 and UX2 as requested)
    tickers = ['SPX Index', 'VIX Index', 'UX1 Index', 'UX2 Index']
    
    # Fetch data from Bloomberg (or simulated)
    data = fetch_bloomberg_data(tickers, start_date, end_date)
    
    # If we want to fetch longer VIX history for percentile calculation
    try:
        vix_history = fetch_bloomberg_data(['VIX Index'], '1990-01-01', end_date)['VIX Index']
        print(f"Using VIX history from {vix_history.index[0]} to {vix_history.index[-1]}")
    except:
        print("Using available VIX history")
        vix_history = data['VIX Index']
    
    # Run the Enhanced Portfolio strategy (starting from April 2007 as in the paper)
    start_date_strategy = '2007-04-01'
    strategy_data = enhanced_portfolio_strategy(
        data['SPX Index'],
        data['VIX Index'],
        data['UX1 Index'],
        data['UX2 Index'],
        vix_history
    )
    
    # Filter for strategy period
    strategy_data = strategy_data[strategy_data.index >= start_date_strategy]
    
    # Print allocation statistics
    print("\nAllocation Statistics:")
    print(f"Percent days long VIX: {strategy_data['in_long_VIX'].mean()*100:.2f}%")
    print(f"Percent days short VIX: {strategy_data['in_short_VIX'].mean()*100:.2f}%")
    print(f"Percent days flat VIX: {strategy_data['in_flat_VIX'].mean()*100:.2f}%")
    
    if strategy_data['in_long_VIX'].sum() > 0:
        print(f"Average long VIX allocation when active: {strategy_data.loc[strategy_data['in_long_VIX']==1, 'VIX30_allocation'].mean()*100:.2f}%")
    
    if strategy_data['in_short_VIX'].sum() > 0:
        print(f"Average short VIX allocation when active: {strategy_data.loc[strategy_data['in_short_VIX']==1, 'VIX30_allocation'].abs().mean()*100:.2f}%")
    
    # Calculate performance metrics
    print("\nCalculating performance metrics...")
    portfolio_metrics = calculate_performance_metrics(strategy_data['portfolio_return'])
    spx_metrics = calculate_performance_metrics(strategy_data['SPX_returns'])
    
    # Display results
    print("\nEnhanced Portfolio Performance Metrics:")
    print(portfolio_metrics)
    
    print("\nS&P 500 Performance Metrics:")
    print(spx_metrics)
    
    # Calculate annual performance
    annual_returns = calculate_annual_performance(strategy_data)
    print("\nAnnual Returns:")
    print(annual_returns)
    
    # Also analyze the two sub-periods as in the paper
    mid_date = '2013-01-01'
    
    # First sub-period: April 2007 - December 2012
    first_period = strategy_data[(strategy_data.index >= start_date_strategy) & 
                                (strategy_data.index < mid_date)]
    
    # Second sub-period: January 2013 - December 2018
    second_period = strategy_data[strategy_data.index >= mid_date]
    
    # Calculate metrics for sub-periods
    print("\nFirst Period (Apr 2007 - Dec 2012):")
    print(calculate_performance_metrics(first_period['portfolio_return']))
    print("\nS&P 500 First Period:")
    print(calculate_performance_metrics(first_period['SPX_returns']))
    
    print("\nSecond Period (Jan 2013 - Dec 2018):")
    print(calculate_performance_metrics(second_period['portfolio_return']))
    print("\nS&P 500 Second Period:")
    print(calculate_performance_metrics(second_period['SPX_returns']))
    
    # Create plots
    print("\nGenerating plots...")
    plot_results(strategy_data)
    
    print("Backtest completed successfully!")
    
    # Return the strategy data for further analysis if needed
    return strategy_data

# Function to create plots with improvements
def plot_results(data):
    # Plot 1: Cumulative returns
    plt.figure(figsize=(15, 8))
    plt.plot(data['portfolio_cumulative'], label='Enhanced Portfolio')
    plt.plot(data['SPX_cumulative'], label='S&P 500')
    plt.title('Cumulative Returns: Enhanced Portfolio vs S&P 500')
    plt.legend()
    plt.yscale('log')
    plt.grid(True)
    plt.savefig('cumulative_returns.png')
    plt.close()
    
    # Plot 2: VIX Index and VIX Premium
    plt.figure(figsize=(15, 8))
    fig, ax1 = plt.subplots(figsize=(15, 8))
    
    ax1.set_xlabel('Date')
    ax1.set_ylabel('VIX Index', color='blue')
    ax1.plot(data['VIX'], color='blue', label='VIX Index')
    ax1.tick_params(axis='y', labelcolor='blue')
    
    ax2 = ax1.twinx()
    ax2.set_ylabel('VIX Premium', color='red')
    ax2.plot(data['VIX_premium'], color='red', label='VIX Premium')
    ax2.tick_params(axis='y', labelcolor='red')
    
    plt.title('VIX Index and VIX Premium')
    fig.tight_layout()
    plt.savefig('vix_and_premium.png')
    plt.close()
    
    # Plot 3: VIX30 Allocation
    plt.figure(figsize=(15, 8))
    plt.plot(data['VIX30_allocation'], color='green')
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.title('VIX30 Allocation in Enhanced Portfolio')
    plt.ylabel('Allocation')
    plt.grid(True)
    plt.savefig('vix30_allocation.png')
    plt.close()
    
    # Plot 4: Monthly returns comparison
    monthly_portfolio = data['portfolio_return'].resample('M').apply(lambda x: (1 + x).prod() - 1)
    monthly_spx = data['SPX_returns'].resample('M').apply(lambda x: (1 + x).prod() - 1)
    
    plt.figure(figsize=(15, 8))
    width = 10
    plt.bar(monthly_portfolio.index, monthly_portfolio, width=width, alpha=0.5, label='Enhanced Portfolio')
    plt.bar(monthly_spx.index, monthly_spx, width=width, alpha=0.5, label='S&P 500')
    plt.title('Monthly Returns Comparison')
    plt.legend()
    plt.grid(True)
    plt.savefig('monthly_returns.png')
    plt.close()
    
    # Plot 5: VIX30 contribution by type (long vs short)
    plt.figure(figsize=(15, 8))
    
    # Resample to monthly for clarity
    long_contrib = data['long_VIX30_contrib'].resample('M').sum()
    short_contrib = data['short_VIX30_contrib'].resample('M').sum()
    
    plt.bar(long_contrib.index, long_contrib, width=width, color='green', alpha=0.7, label='Long VIX30 Contribution')
    plt.bar(short_contrib.index, short_contrib, width=width, color='red', alpha=0.7, label='Short VIX30 Contribution')
    plt.title('VIX30 Contribution by Position Type (Monthly)')
    plt.legend()
    plt.grid(True)
    plt.savefig('vix30_contribution.png')
    plt.close()

# Run the main function
if __name__ == "__main__":
    try:
        strategy_data = main()
        print("Strategy execution complete")
    except Exception as e:
        import traceback
        print(f"Error in strategy execution: {e}")
        traceback.print_exc()