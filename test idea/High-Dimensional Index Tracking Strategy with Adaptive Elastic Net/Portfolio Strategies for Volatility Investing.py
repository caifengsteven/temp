import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy import stats
from arch import arch_model
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# Define some helper functions for performance metrics
def annualized_return(returns):
    """Calculate annualized return from a series of returns"""
    return (1 + returns.mean()) ** (252) - 1

def annualized_volatility(returns):
    """Calculate annualized volatility from a series of returns"""
    return returns.std() * np.sqrt(252)

def sharpe_ratio(returns, risk_free_rate=0):
    """Calculate annualized Sharpe ratio"""
    excess_returns = returns - risk_free_rate / 252
    return annualized_return(excess_returns) / annualized_volatility(returns)

def sortino_ratio(returns, risk_free_rate=0):
    """Calculate Sortino ratio"""
    excess_returns = returns - risk_free_rate / 252
    downside_returns = returns[returns < 0]
    downside_deviation = downside_returns.std() * np.sqrt(252)
    return annualized_return(excess_returns) / downside_deviation if downside_deviation != 0 else np.nan

def max_drawdown(returns):
    """Calculate maximum drawdown"""
    cum_returns = (1 + returns).cumprod()
    peak = cum_returns.expanding(min_periods=1).max()
    drawdown = (cum_returns / peak) - 1
    return drawdown.min()

def mppm(returns, risk_aversion=3):
    """Calculate Manipulation Proof Performance Measure (MPPM)"""
    # Convert to monthly returns for MPPM
    if isinstance(returns, pd.Series):
        monthly_returns = returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
    else:
        # If returns is a DataFrame with a datetime index
        returns_series = returns.iloc[:, 0] if returns.shape[1] > 0 else pd.Series(0, index=returns.index)
        monthly_returns = returns_series.resample('M').apply(lambda x: (1 + x).prod() - 1)
    
    T = len(monthly_returns)
    if T == 0:
        return 0
    
    product_term = (1 + monthly_returns).pow(1 - risk_aversion).prod()
    mppm_value = (product_term ** (1 / ((1 - risk_aversion) * T)) - 1) / (1 - risk_aversion)
    return mppm_value

def calculate_performance_metrics(returns, risk_free_rate=0):
    """Calculate performance metrics for a return series"""
    if isinstance(returns, pd.Series):
        monthly_returns = returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
    else:
        # Handle the case where returns might be a DataFrame
        returns_series = returns if isinstance(returns, pd.Series) else returns.iloc[:, 0]
        monthly_returns = returns_series.resample('M').apply(lambda x: (1 + x).prod() - 1)
    
    metrics = {
        'Return (Monthly)': monthly_returns.mean(),
        'StDev (Monthly)': monthly_returns.std(),
        'Skewness': stats.skew(monthly_returns),
        'Kurtosis': stats.kurtosis(monthly_returns),
        'Sharpe Ratio': sharpe_ratio(returns, risk_free_rate),
        'Downside Deviation': monthly_returns[monthly_returns < 0].std(),
        'Sortino Ratio': sortino_ratio(returns, risk_free_rate),
        'Max Drawdown': max_drawdown(returns),
        'MPPM': mppm(returns, 3)
    }
    return metrics

# Improved simulation function
def simulate_market_data(start_date='2007-04-01', end_date='2018-12-31', freq='D'):
    """
    Simulate market data with more realistic VIX premium dynamics
    """
    # Create date range
    date_range = pd.date_range(start=start_date, end=end_date, freq=freq)
    n_days = len(date_range)
    
    # Parameters for S&P 500 simulation
    mu_spx = 0.00035  # Daily mean return
    sigma_spx = 0.012  # Daily volatility
    
    # Parameters for VIX simulation
    vix_mean = 20
    vix_vol = 8
    vix_mean_reversion = 0.05
    
    # Correlation between S&P 500 returns and VIX changes
    corr = -0.8
    
    # Initialize arrays
    spx_returns = np.zeros(n_days)
    vix_values = np.zeros(n_days)
    vix_futures1 = np.zeros(n_days)
    vix_futures2 = np.zeros(n_days)
    
    # Set initial values
    vix_values[0] = vix_mean
    vix_futures1[0] = vix_mean * 1.05  # Initial premium of 5%
    vix_futures2[0] = vix_mean * 1.08  # Initial premium of 8%
    
    # Create some market regimes
    # 0: normal, 1: stress (high volatility)
    market_regime = np.zeros(n_days)
    
    # Set stress periods (to mimic financial crisis and other stress periods)
    # 2008 financial crisis
    crisis_start = pd.Timestamp('2008-09-01')
    crisis_end = pd.Timestamp('2009-03-31')
    
    # 2011 debt ceiling crisis
    debt_crisis_start = pd.Timestamp('2011-07-01')
    debt_crisis_end = pd.Timestamp('2011-10-31')
    
    # 2015 flash crash
    flash_crash_start = pd.Timestamp('2015-08-01')
    flash_crash_end = pd.Timestamp('2015-09-30')
    
    # 2018 December selloff
    dec_2018_start = pd.Timestamp('2018-12-01')
    dec_2018_end = pd.Timestamp('2018-12-31')
    
    # Mark stress periods
    for i, date in enumerate(date_range):
        if ((crisis_start <= date <= crisis_end) or 
            (debt_crisis_start <= date <= debt_crisis_end) or
            (flash_crash_start <= date <= flash_crash_end) or
            (dec_2018_start <= date <= dec_2018_end)):
            market_regime[i] = 1
    
    # Simulate S&P 500 and VIX paths
    for t in range(1, n_days):
        # Generate correlated random variables
        z1 = np.random.normal(0, 1)
        z2 = corr * z1 + np.sqrt(1 - corr**2) * np.random.normal(0, 1)
        
        # Adjust return distribution based on regime
        if market_regime[t] == 1:  # Stress period
            # Higher volatility, negative drift in stress periods
            spx_returns[t] = -0.0005 + sigma_spx * 2 * z1
        else:  # Normal period
            spx_returns[t] = mu_spx + sigma_spx * z1
        
        # Add occasional jumps
        if np.random.random() < 0.01:  # 1% chance of jump
            if market_regime[t] == 1:
                # Larger negative jumps in stress periods
                spx_returns[t] += np.random.normal(-0.02, 0.01)
            else:
                # Smaller jumps in normal periods
                spx_returns[t] += np.random.normal(0, 0.01)
        
        # Simulate VIX with mean-reversion and regime-dependent behavior
        if market_regime[t] == 1:  # Stress period
            target_vix = vix_mean * 2.5  # Higher target in stress
            vix_change = vix_mean_reversion * 2 * (target_vix - vix_values[t-1]) + vix_vol * 0.2 * z2
        else:  # Normal period
            target_vix = vix_mean
            vix_change = vix_mean_reversion * (target_vix - vix_values[t-1]) + vix_vol * 0.1 * z2
        
        vix_values[t] = max(9, vix_values[t-1] + vix_change)  # Ensure VIX stays above 9
        
        # Add some jumps to VIX during market stress periods
        if spx_returns[t] < -2.5 * sigma_spx:
            vix_values[t] += np.random.exponential(5)
            
        # Simulate VIX futures with regime-dependent premium
        if market_regime[t] == 1:  # Stress period
            # During stress, futures can trade at a discount (negative premium)
            premium1 = np.random.uniform(-2.0, 0)
            premium2 = np.random.uniform(-1.0, 1.0)
        else:  # Normal period
            # During normal times, futures trade at a premium
            premium1 = np.random.uniform(0.2, 1.5)
            premium2 = np.random.uniform(0.5, 2.0)
        
        vix_futures1[t] = vix_values[t] + premium1
        vix_futures2[t] = vix_values[t] + premium2
    
    # Calculate S&P 500 index values
    spx_values = 1000 * np.cumprod(1 + spx_returns)
    
    # Create DataFrame
    data = pd.DataFrame({
        'SPX': spx_values,
        'SPX_Return': spx_returns,
        'VIX': vix_values,
        'VIX_Futures1': vix_futures1,
        'VIX_Futures2': vix_futures2,
        'Market_Regime': market_regime
    }, index=date_range)
    
    return data

# Create a simulated 30-day constant maturity VIX futures series (VIX30)
def calculate_vix30(data):
    """
    Calculate the 30-day constant maturity VIX futures (VIX30)
    
    Parameters:
    -----------
    data : DataFrame
        DataFrame containing VIX and VIX futures data
        
    Returns:
    --------
    DataFrame with VIX30 added
    """
    # For simplicity in the simulation, assume 30 days in a month
    # and a fixed rollover schedule
    days_in_month = 21  # Trading days in a month
    
    # Create a weight series for the front-month VIX future
    weights = np.zeros(len(data))
    for i in range(len(data)):
        # Determine days to expiration for the front month
        day_of_month = i % days_in_month
        weight = 1 - day_of_month / days_in_month
        weights[i] = weight
    
    # Calculate VIX30
    data['VIX30'] = weights * data['VIX_Futures1'] + (1 - weights) * data['VIX_Futures2']
    
    # Calculate VIX premium
    data['VIX_Premium'] = data['VIX30'] - data['VIX']
    
    # Calculate VIX30 returns
    data['VIX30_Return'] = data['VIX30'].pct_change()
    
    return data

# Strategy implementation
def enhanced_portfolio_strategy(data, risk_parity=True, vix_bottom_decile=0.1):
    """
    Implement the Enhanced Portfolio strategy
    
    Parameters:
    -----------
    data : DataFrame
        DataFrame containing S&P 500, VIX and VIX futures data
    risk_parity : bool
        Whether to use risk parity for allocations
    vix_bottom_decile : float
        Percentile threshold for not shorting volatility
        
    Returns:
    --------
    DataFrame with strategy returns and positions
    """
    # Calculate VIX bottom decile threshold
    vix_threshold = data['VIX'].quantile(vix_bottom_decile)
    
    # Initialize strategy DataFrame
    strategy = pd.DataFrame(index=data.index)
    strategy['VIX_Premium_Sign'] = np.sign(data['VIX_Premium'])
    
    # Determine position direction: long VIX when premium negative, short when positive
    strategy['VIX30_Position'] = strategy['VIX_Premium_Sign'].shift(1) * -1
    
    # Don't short volatility if VIX is too low
    low_vix_mask = (data['VIX'].shift(1) < vix_threshold)
    strategy.loc[low_vix_mask, 'VIX30_Position'] = 0
    
    # Initialize weights
    strategy['SPX_Weight'] = 0.0
    strategy['VIX30_Weight'] = 0.0
    
    # Calculate volatilities using a GARCH model (in practice)
    # For simulation, we'll use a rolling window standard deviation
    window = 21  # 21 trading days (~1 month)
    spx_vol = data['SPX_Return'].rolling(window=window).std().shift(1)
    vix_vol = data['VIX30_Return'].rolling(window=window).std().shift(1)
    
    # Handle NaN values
    spx_vol = spx_vol.fillna(data['SPX_Return'].std())
    vix_vol = vix_vol.fillna(data['VIX30_Return'].std())
    
    if risk_parity:
        # Calculate weights based on risk parity
        for i in range(len(strategy)):
            if i == 0:  # Skip first day (no historical data)
                strategy.iloc[i, strategy.columns.get_loc('SPX_Weight')] = 1.0
                strategy.iloc[i, strategy.columns.get_loc('VIX30_Weight')] = 0.0
                continue
                
            if strategy.iloc[i]['VIX30_Position'] == 0:
                # If no VIX position, allocate 100% to S&P 500
                strategy.iloc[i, strategy.columns.get_loc('SPX_Weight')] = 1.0
                strategy.iloc[i, strategy.columns.get_loc('VIX30_Weight')] = 0.0
            else:
                # Calculate risk contribution weights
                vix_vol_i = vix_vol.iloc[i-1]  # Use previous day's volatility
                spx_vol_i = spx_vol.iloc[i-1]
                
                # Avoid division by zero
                if vix_vol_i == 0 or spx_vol_i == 0:
                    strategy.iloc[i, strategy.columns.get_loc('SPX_Weight')] = 0.5
                    strategy.iloc[i, strategy.columns.get_loc('VIX30_Weight')] = strategy.iloc[i]['VIX30_Position'] * 0.5
                else:
                    strategy.iloc[i, strategy.columns.get_loc('SPX_Weight')] = vix_vol_i / (vix_vol_i + spx_vol_i)
                    strategy.iloc[i, strategy.columns.get_loc('VIX30_Weight')] = spx_vol_i / (vix_vol_i + spx_vol_i) * strategy.iloc[i]['VIX30_Position']
    else:
        # Equal weight allocation (for comparison)
        strategy['SPX_Weight'] = 0.5
        strategy['VIX30_Weight'] = 0.5 * strategy['VIX30_Position']
    
    # Calculate strategy returns
    strategy['SPX_Contribution'] = strategy['SPX_Weight'] * data['SPX_Return']
    strategy['VIX30_Contribution'] = strategy['VIX30_Weight'] * data['VIX30_Return']
    strategy['Return'] = strategy['SPX_Contribution'] + strategy['VIX30_Contribution']
    
    # Calculate strategy statistics
    strategy['Cumulative_Return'] = (1 + strategy['Return']).cumprod()
    strategy['Long_VIX30'] = (strategy['VIX30_Position'] > 0).astype(int)
    strategy['Short_VIX30'] = (strategy['VIX30_Position'] < 0).astype(int)
    
    return strategy

# Function to run Enhanced Portfolio and its variants
def run_enhanced_portfolio_variants(data):
    """
    Run the Enhanced Portfolio strategy and its variants
    
    Parameters:
    -----------
    data : DataFrame
        DataFrame containing market data
        
    Returns:
    --------
    Dictionary of DataFrames with strategy results
    """
    results = {}
    
    # Run the original Enhanced Portfolio
    results['Enhanced'] = enhanced_portfolio_strategy(data)
    
    # Run Enhanced Long (only long VIX positions)
    data_long = data.copy()
    strategy_long = enhanced_portfolio_strategy(data_long)
    strategy_long.loc[strategy_long['VIX30_Position'] < 0, 'VIX30_Position'] = 0
    strategy_long.loc[strategy_long['VIX30_Position'] == 0, 'SPX_Weight'] = 1
    strategy_long.loc[strategy_long['VIX30_Position'] == 0, 'VIX30_Weight'] = 0
    strategy_long['SPX_Contribution'] = strategy_long['SPX_Weight'] * data_long['SPX_Return']
    strategy_long['VIX30_Contribution'] = strategy_long['VIX30_Weight'] * data_long['VIX30_Return']
    strategy_long['Return'] = strategy_long['SPX_Contribution'] + strategy_long['VIX30_Contribution']
    strategy_long['Cumulative_Return'] = (1 + strategy_long['Return']).cumprod()
    strategy_long['Long_VIX30'] = (strategy_long['VIX30_Position'] > 0).astype(int)
    strategy_long['Short_VIX30'] = (strategy_long['VIX30_Position'] < 0).astype(int)
    results['EnhancedLong'] = strategy_long
    
    # Run Enhanced Short (only short VIX positions)
    data_short = data.copy()
    strategy_short = enhanced_portfolio_strategy(data_short)
    strategy_short.loc[strategy_short['VIX30_Position'] > 0, 'VIX30_Position'] = 0
    strategy_short.loc[strategy_short['VIX30_Position'] == 0, 'SPX_Weight'] = 1
    strategy_short.loc[strategy_short['VIX30_Position'] == 0, 'VIX30_Weight'] = 0
    strategy_short['SPX_Contribution'] = strategy_short['SPX_Weight'] * data_short['SPX_Return']
    strategy_short['VIX30_Contribution'] = strategy_short['VIX30_Weight'] * data_short['VIX30_Return']
    strategy_short['Return'] = strategy_short['SPX_Contribution'] + strategy_short['VIX30_Contribution']
    strategy_short['Cumulative_Return'] = (1 + strategy_short['Return']).cumprod()
    strategy_short['Long_VIX30'] = (strategy_short['VIX30_Position'] > 0).astype(int)
    strategy_short['Short_VIX30'] = (strategy_short['VIX30_Position'] < 0).astype(int)
    results['EnhancedShort'] = strategy_short
    
    # Run Enhanced90 (90% risk to S&P 500, 10% to VIX)
    data_90 = data.copy()
    strategy_90 = enhanced_portfolio_strategy(data_90, risk_parity=False)
    strategy_90['SPX_Weight'] = 0.9
    strategy_90['VIX30_Weight'] = 0.1 * strategy_90['VIX30_Position']
    strategy_90['SPX_Contribution'] = strategy_90['SPX_Weight'] * data_90['SPX_Return']
    strategy_90['VIX30_Contribution'] = strategy_90['VIX30_Weight'] * data_90['VIX30_Return']
    strategy_90['Return'] = strategy_90['SPX_Contribution'] + strategy_90['VIX30_Contribution']
    strategy_90['Cumulative_Return'] = (1 + strategy_90['Return']).cumprod()
    strategy_90['Long_VIX30'] = (strategy_90['VIX30_Position'] > 0).astype(int)
    strategy_90['Short_VIX30'] = (strategy_90['VIX30_Position'] < 0).astype(int)
    results['Enhanced90'] = strategy_90
    
    return results

# Benchmark strategies
def run_benchmark_strategies(data):
    """
    Run benchmark strategies for comparison
    
    Parameters:
    -----------
    data : DataFrame
        DataFrame containing market data
        
    Returns:
    --------
    Dictionary of DataFrames with benchmark results
    """
    benchmarks = {}
    
    # S&P 500 Buy and Hold
    benchmarks['S&P 500'] = pd.DataFrame(index=data.index)
    benchmarks['S&P 500']['Return'] = data['SPX_Return']
    benchmarks['S&P 500']['Cumulative_Return'] = (1 + benchmarks['S&P 500']['Return']).cumprod()
    
    # Simple implementation of covered call strategies
    
    # For the BXM (ATM covered call)
    benchmarks['BXM'] = pd.DataFrame(index=data.index)
    # Approximate BXM returns with a formula: SPX_Return - call premium + limited upside
    call_premium = 0.015 / 21  # Approximate monthly premium divided by days
    benchmarks['BXM']['Return'] = data['SPX_Return'].copy()
    # Use numpy.where instead of direct comparison
    benchmarks['BXM']['Return'] = np.where(
        benchmarks['BXM']['Return'] > 0.01,  # If market moves up strongly
        0.01,  # Cap returns at about 1%
        benchmarks['BXM']['Return'] + call_premium  # Otherwise get the premium
    )
    benchmarks['BXM']['Cumulative_Return'] = (1 + benchmarks['BXM']['Return']).cumprod()
    
    # For the BXY (2% OTM covered call)
    benchmarks['BXY'] = pd.DataFrame(index=data.index)
    call_premium_otm = 0.01 / 21  # Lower premium for OTM
    benchmarks['BXY']['Return'] = data['SPX_Return'].copy()
    benchmarks['BXY']['Return'] = np.where(
        benchmarks['BXY']['Return'] > 0.02,  # If market moves up strongly
        0.02,  # Cap returns at about 2%
        benchmarks['BXY']['Return'] + call_premium_otm  # Otherwise get the premium
    )
    benchmarks['BXY']['Cumulative_Return'] = (1 + benchmarks['BXY']['Return']).cumprod()
    
    # For PUT (ATM put writing)
    benchmarks['PUT'] = pd.DataFrame(index=data.index)
    put_premium = 0.015 / 21
    benchmarks['PUT']['Return'] = data['SPX_Return'].copy()
    # Use numpy's minimum function elementwise
    benchmarks['PUT']['Return'] = np.where(
        benchmarks['PUT']['Return'] < -0.01,
        -0.01,  # Floor on losses
        np.minimum(data['SPX_Return'].values, put_premium)  # Either market return or premium
    )
    benchmarks['PUT']['Cumulative_Return'] = (1 + benchmarks['PUT']['Return']).cumprod()
    
    # COND (conditional covered call)
    benchmarks['COND'] = pd.DataFrame(index=data.index)
    vix_median = data['VIX'].median()
    benchmarks['COND']['Return'] = data['SPX_Return'].copy()
    
    # Only sell calls when VIX is above median
    high_vix_mask = (data['VIX'].shift(1) > vix_median)
    high_vix_returns = benchmarks['COND'].loc[high_vix_mask, 'Return'].copy()
    
    # Apply the transformation only to the masked values
    benchmarks['COND'].loc[high_vix_mask, 'Return'] = np.where(
        high_vix_returns > 0.01,
        0.01,  # Cap returns at about 1% when we have a covered call
        high_vix_returns + call_premium  # Otherwise get the premium
    )
    benchmarks['COND']['Cumulative_Return'] = (1 + benchmarks['COND']['Return']).cumprod()
    
    return benchmarks

# Function to analyze drawdowns
def analyze_drawdowns(returns_dict, threshold=0.1):
    """
    Analyze drawdowns for different strategies
    
    Parameters:
    -----------
    returns_dict : dict
        Dictionary of DataFrames with strategy returns
    threshold : float
        Threshold for considering a drawdown significant
        
    Returns:
    --------
    DataFrame with drawdown analysis
    """
    drawdowns = {}
    
    for name, df in returns_dict.items():
        # Calculate drawdowns
        cum_returns = (1 + df['Return']).cumprod()
        running_max = cum_returns.cummax()
        drawdown = (cum_returns / running_max) - 1
        
        # Find drawdown periods
        is_drawdown = drawdown < -threshold
        # Find start and end of drawdown periods
        drawdown_starts = is_drawdown & ~is_drawdown.shift(1, fill_value=False)
        drawdown_ends = ~is_drawdown & is_drawdown.shift(1, fill_value=False)
        
        # Get the dates
        start_dates = drawdown_starts[drawdown_starts].index.tolist()
        end_dates = drawdown_ends[drawdown_ends].index.tolist()
        
        # If still in drawdown, add the last date
        if len(start_dates) > len(end_dates):
            end_dates.append(drawdown.index[-1])
        
        # Store the drawdown periods
        periods = []
        for i in range(min(len(start_dates), len(end_dates))):
            start_date = start_dates[i]
            end_date = end_dates[i]
            max_drawdown = drawdown.loc[start_date:end_date].min()
            drawdown_length = (end_date - start_date).days
            periods.append({
                'Start': start_date,
                'End': end_date,
                'Max Drawdown': max_drawdown,
                'Length (days)': drawdown_length
            })
        
        drawdowns[name] = periods
    
    return drawdowns

# Function to perform Monte Carlo analysis
def monte_carlo_analysis(enhanced_returns, spx_returns, num_samples=10000, min_length=252):
    """
    Perform Monte Carlo analysis by sampling random periods
    
    Parameters:
    -----------
    enhanced_returns : Series
        Returns for the Enhanced Portfolio
    spx_returns : Series
        Returns for the S&P 500
    num_samples : int
        Number of random samples to draw
    min_length : int
        Minimum length of each sample in trading days
        
    Returns:
    --------
    DataFrame with results of the Monte Carlo analysis
    """
    results = []
    
    for _ in range(num_samples):
        # Pick a random start date
        if len(enhanced_returns) <= min_length:
            start_idx = 0
            end_idx = len(enhanced_returns) - 1
        else:
            start_idx = np.random.randint(0, len(enhanced_returns) - min_length)
            end_idx = min(start_idx + min_length + np.random.randint(0, len(enhanced_returns) - start_idx - min_length), 
                         len(enhanced_returns) - 1)
        
        # Extract returns for this period
        enhanced_sample = enhanced_returns.iloc[start_idx:end_idx]
        spx_sample = spx_returns.iloc[start_idx:end_idx]
        
        # Calculate metrics
        enhanced_sharpe = sharpe_ratio(enhanced_sample)
        spx_sharpe = sharpe_ratio(spx_sample)
        
        enhanced_sortino = sortino_ratio(enhanced_sample)
        spx_sortino = sortino_ratio(spx_sample)
        
        enhanced_return = annualized_return(enhanced_sample)
        spx_return = annualized_return(spx_sample)
        
        enhanced_mppm_val = mppm(enhanced_sample.to_frame())
        spx_mppm_val = mppm(spx_sample.to_frame())
        
        # Store results
        results.append({
            'Enhanced_Sharpe': enhanced_sharpe,
            'SPX_Sharpe': spx_sharpe,
            'Sharpe_Diff': enhanced_sharpe - spx_sharpe,
            'Enhanced_Sortino': enhanced_sortino,
            'SPX_Sortino': spx_sortino,
            'Sortino_Diff': enhanced_sortino - spx_sortino,
            'Enhanced_Return': enhanced_return,
            'SPX_Return': spx_return,
            'Return_Diff': enhanced_return - spx_return,
            'Enhanced_MPPM': enhanced_mppm_val,
            'SPX_MPPM': spx_mppm_val,
            'MPPM_Diff': enhanced_mppm_val - spx_mppm_val
        })
    
    results_df = pd.DataFrame(results)
    
    # Calculate percentage of outperformance
    outperformance = {
        'Return': (results_df['Return_Diff'] > 0).mean() * 100,
        'Sharpe': (results_df['Sharpe_Diff'] > 0).mean() * 100,
        'Sortino': (results_df['Sortino_Diff'] > 0).mean() * 100,
        'MPPM': (results_df['MPPM_Diff'] > 0).mean() * 100
    }
    
    # Calculate average outperformance and underperformance
    avg_outperformance = {
        'Return': results_df.loc[results_df['Return_Diff'] > 0, 'Return_Diff'].mean(),
        'Sharpe': results_df.loc[results_df['Sharpe_Diff'] > 0, 'Sharpe_Diff'].mean(),
        'Sortino': results_df.loc[results_df['Sortino_Diff'] > 0, 'Sortino_Diff'].mean(),
        'MPPM': results_df.loc[results_df['MPPM_Diff'] > 0, 'MPPM_Diff'].mean()
    }
    
    avg_underperformance = {
        'Return': results_df.loc[results_df['Return_Diff'] < 0, 'Return_Diff'].mean(),
        'Sharpe': results_df.loc[results_df['Sharpe_Diff'] < 0, 'Sharpe_Diff'].mean(),
        'Sortino': results_df.loc[results_df['Sortino_Diff'] < 0, 'Sortino_Diff'].mean(),
        'MPPM': results_df.loc[results_df['MPPM_Diff'] < 0, 'MPPM_Diff'].mean()
    }
    
    return {
        'Results': results_df,
        'Outperformance': outperformance,
        'Avg_Outperformance': avg_outperformance,
        'Avg_Underperformance': avg_underperformance
    }

# Function to display performance metrics
def display_performance_table(strategies_dict, period_name="Full Period"):
    """
    Display performance metrics for different strategies
    
    Parameters:
    -----------
    strategies_dict : dict
        Dictionary of DataFrames with strategy returns
    period_name : str
        Name of the period being analyzed
        
    Returns:
    --------
    DataFrame with performance metrics
    """
    metrics = {}
    
    for name, df in strategies_dict.items():
        metrics[name] = calculate_performance_metrics(df['Return'])
    
    # Convert to DataFrame for display
    metrics_df = pd.DataFrame(metrics)
    
    print(f"\n--- Performance Metrics: {period_name} ---")
    return metrics_df.T

# Plotting functions
def plot_cumulative_returns(strategies_dict, title="Strategy Cumulative Returns"):
    """
    Plot cumulative returns for different strategies
    
    Parameters:
    -----------
    strategies_dict : dict
        Dictionary of DataFrames with strategy returns
    title : str
        Plot title
        
    Returns:
    --------
    Matplotlib figure
    """
    plt.figure(figsize=(12, 8))
    
    for name, df in strategies_dict.items():
        plt.plot(df['Cumulative_Return'], label=name)
    
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    plt.legend()
    plt.grid(True)
    return plt.gcf()

def plot_drawdowns(strategies_dict, title="Strategy Drawdowns"):
    """
    Plot drawdowns for different strategies
    
    Parameters:
    -----------
    strategies_dict : dict
        Dictionary of DataFrames with strategy returns
    title : str
        Plot title
        
    Returns:
    --------
    Matplotlib figure
    """
    plt.figure(figsize=(12, 8))
    
    for name, df in strategies_dict.items():
        # Calculate drawdowns
        cum_returns = (1 + df['Return']).cumprod()
        running_max = cum_returns.cummax()
        drawdown = (cum_returns / running_max) - 1
        plt.plot(drawdown, label=name)
    
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Drawdown')
    plt.legend()
    plt.grid(True)
    return plt.gcf()

def plot_vix_and_premium(data, title="VIX and VIX Premium"):
    """
    Plot VIX Index and VIX Premium
    
    Parameters:
    -----------
    data : DataFrame
        DataFrame containing VIX and VIX premium data
    title : str
        Plot title
        
    Returns:
    --------
    Matplotlib figure
    """
    fig, ax1 = plt.subplots(figsize=(12, 8))
    
    # Plot VIX
    ax1.plot(data['VIX'], 'b-', label='VIX Index')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('VIX Index', color='b')
    ax1.tick_params('y', colors='b')
    
    # Plot VIX Premium
    ax2 = ax1.twinx()
    ax2.plot(data['VIX_Premium'], 'r-', label='VIX Premium')
    ax2.set_ylabel('VIX Premium', color='r')
    ax2.tick_params('y', colors='r')
    
    # Add legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    plt.title(title)
    plt.grid(True)
    return fig

def plot_vix30_allocations(enhanced_strategy, data, title="VIX30 Allocations"):
    """
    Plot VIX30 allocations and VIX Index
    
    Parameters:
    -----------
    enhanced_strategy : DataFrame
        DataFrame containing Enhanced strategy data
    data : DataFrame
        DataFrame containing market data
    title : str
        Plot title
        
    Returns:
    --------
    Matplotlib figure
    """
    fig, ax1 = plt.subplots(figsize=(12, 8))
    
    # Plot VIX
    ax1.plot(data['VIX'], 'b-', label='VIX Index')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('VIX Index', color='b')
    ax1.tick_params('y', colors='b')
    
    # Plot VIX30 allocation
    ax2 = ax1.twinx()
    ax2.plot(enhanced_strategy['VIX30_Weight'].abs(), 'r-', label='|VIX30 Allocation|')
    ax2.set_ylabel('|VIX30 Allocation|', color='r')
    ax2.tick_params('y', colors='r')
    
    # Add legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    plt.title(title)
    plt.grid(True)
    return fig

def plot_vix30_position_and_spx(enhanced_strategy, data, title="VIX30 Position and S&P 500"):
    """
    Plot VIX30 position and S&P 500 equity curve
    
    Parameters:
    -----------
    enhanced_strategy : DataFrame
        DataFrame containing Enhanced strategy data
    data : DataFrame
        DataFrame containing market data
    title : str
        Plot title
        
    Returns:
    --------
    Matplotlib figure
    """
    fig, ax1 = plt.subplots(figsize=(12, 8))
    
    # Plot SPX
    spx_normalized = data['SPX'] / data['SPX'].iloc[0]
    ax1.plot(spx_normalized, 'b-', label='S&P 500 (Normalized)')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('S&P 500', color='b')
    ax1.tick_params('y', colors='b')
    
    # Plot VIX30 position
    ax2 = ax1.twinx()
    ax2.plot(enhanced_strategy['VIX30_Weight'], 'r-', label='VIX30 Position')
    ax2.set_ylabel('VIX30 Position', color='r')
    ax2.tick_params('y', colors='r')
    
    # Add a horizontal line at zero
    ax2.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    
    # Add legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    plt.title(title)
    plt.grid(True)
    return fig

def plot_allocations_over_time(enhanced_strategy, title="Enhanced Portfolio Allocations Over Time"):
    """
    Plot allocations to S&P 500 and VIX30 over time
    
    Parameters:
    -----------
    enhanced_strategy : DataFrame
        DataFrame containing Enhanced strategy data
    title : str
        Plot title
        
    Returns:
    --------
    Matplotlib figure
    """
    plt.figure(figsize=(12, 8))
    
    # Create a stacked area plot of allocations
    plt.stackplot(enhanced_strategy.index, 
                  enhanced_strategy['SPX_Weight'], 
                  enhanced_strategy['VIX30_Weight'], 
                  labels=['S&P 500', 'VIX30'],
                  colors=['blue', 'red'],
                  alpha=0.7)
    
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Allocation')
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # Add a horizontal line at 0 and 1
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.axhline(y=1, color='k', linestyle='-', alpha=0.3)
    
    return plt.gcf()

def plot_vix_premium_histogram(data, title="VIX Premium Distribution"):
    """
    Plot histogram of VIX premium values
    
    Parameters:
    -----------
    data : DataFrame
        DataFrame containing VIX premium data
    title : str
        Plot title
        
    Returns:
    --------
    Matplotlib figure
    """
    plt.figure(figsize=(10, 6))
    
    # Plot histogram
    plt.hist(data['VIX_Premium'], bins=50, alpha=0.7, color='blue')
    
    # Add vertical line at 0
    plt.axvline(x=0, color='r', linestyle='--', label='Zero Premium')
    
    plt.title(title)
    plt.xlabel('VIX Premium (VIX30 - VIX)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    return plt.gcf()

def plot_regime_vix_premium(data, title="VIX Premium by Market Regime"):
    """
    Plot VIX premium by market regime
    
    Parameters:
    -----------
    data : DataFrame
        DataFrame containing VIX premium and market regime data
    title : str
        Plot title
        
    Returns:
    --------
    Matplotlib figure
    """
    plt.figure(figsize=(10, 6))
    
    # Split data by regime
    normal_regime = data[data['Market_Regime'] == 0]['VIX_Premium']
    stress_regime = data[data['Market_Regime'] == 1]['VIX_Premium']
    
    # Plot histograms
    plt.hist(normal_regime, bins=50, alpha=0.5, color='green', label='Normal Regime')
    plt.hist(stress_regime, bins=50, alpha=0.5, color='red', label='Stress Regime')
    
    # Add vertical line at 0
    plt.axvline(x=0, color='k', linestyle='--', label='Zero Premium')
    
    plt.title(title)
    plt.xlabel('VIX Premium (VIX30 - VIX)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    return plt.gcf()

# Main function to run the entire analysis
def run_volatility_investing_analysis():
    """
    Run the entire volatility investing analysis
    """
    print("Starting volatility investing analysis...")
    
    # Simulate market data
    print("Simulating market data...")
    data = simulate_market_data()
    data = calculate_vix30(data)
    
    # Run Enhanced Portfolio strategy and variants
    print("Running Enhanced Portfolio strategies...")
    enhanced_strategies = run_enhanced_portfolio_variants(data)
    
    # Run benchmark strategies
    print("Running benchmark strategies...")
    benchmark_strategies = run_benchmark_strategies(data)
    
    # Combine all strategies
    all_strategies = {**enhanced_strategies, **benchmark_strategies}
    
    # Display VIX and VIX Premium
    plot_vix_and_premium(data)
    plt.savefig("vix_and_premium.png")
    plt.close()
    
    # VIX Premium distribution
    plot_vix_premium_histogram(data)
    plt.savefig("vix_premium_histogram.png")
    plt.close()
    
    # VIX Premium by regime
    plot_regime_vix_premium(data)
    plt.savefig("vix_premium_by_regime.png")
    plt.close()
    
    # Display VIX30 allocations
    plot_vix30_allocations(enhanced_strategies['Enhanced'], data)
    plt.savefig("vix30_allocations.png")
    plt.close()
    
    # Display VIX30 position and S&P 500
    plot_vix30_position_and_spx(enhanced_strategies['Enhanced'], data)
    plt.savefig("vix30_position_and_spx.png")
    plt.close()
    
    # Display allocations over time
    plot_allocations_over_time(enhanced_strategies['Enhanced'])
    plt.savefig("allocations_over_time.png")
    plt.close()
    
    # Plot cumulative returns
    plot_cumulative_returns(all_strategies)
    plt.savefig("cumulative_returns.png")
    plt.close()
    
    # Plot drawdowns
    plot_drawdowns(all_strategies)
    plt.savefig("drawdowns.png")
    plt.close()
    
    # Display performance metrics for the full period
    full_period_metrics = display_performance_table(all_strategies, "Full Period")
    
    # Split the data into two subperiods as in the paper
    midpoint = data.index[len(data) // 2]
    
    # First subperiod
    first_subperiod = {}
    for name, strategy in all_strategies.items():
        first_subperiod[name] = strategy.loc[:midpoint].copy()
    
    # Second subperiod
    second_subperiod = {}
    for name, strategy in all_strategies.items():
        second_subperiod[name] = strategy.loc[midpoint:].copy()
    
    # Display performance metrics for subperiods
    first_subperiod_metrics = display_performance_table(first_subperiod, "First Subperiod")
    second_subperiod_metrics = display_performance_table(second_subperiod, "Second Subperiod")
    
    # Plot cumulative returns for subperiods
    plot_cumulative_returns(first_subperiod, "First Subperiod Cumulative Returns")
    plt.savefig("first_subperiod_returns.png")
    plt.close()
    
    plot_cumulative_returns(second_subperiod, "Second Subperiod Cumulative Returns")
    plt.savefig("second_subperiod_returns.png")
    plt.close()
    
    # Analyze drawdowns
    print("\nAnalyzing drawdowns...")
    spx_drawdowns = analyze_drawdowns({'S&P 500': benchmark_strategies['S&P 500']}, 0.1)
    
    # Performance during S&P 500 drawdowns
    print("\n--- Performance during S&P 500 drawdowns greater than 10% ---")
    for period in spx_drawdowns['S&P 500']:
        start_date = period['Start']
        end_date = period['End']
        print(f"\nDrawdown from {start_date.date()} to {end_date.date()}, Max Drawdown: {period['Max Drawdown']:.2%}")
        
        for name, strategy in all_strategies.items():
            period_return = (strategy.loc[start_date:end_date, 'Cumulative_Return'].iloc[-1] / 
                            strategy.loc[start_date:end_date, 'Cumulative_Return'].iloc[0]) - 1
            print(f"{name}: {period_return:.2%}")
    
    # Enhanced Portfolio drawdowns
    enhanced_drawdowns = analyze_drawdowns({'Enhanced': enhanced_strategies['Enhanced']}, 0.1)
    
    print("\n--- Enhanced Portfolio drawdowns greater than 10% ---")
    for period in enhanced_drawdowns['Enhanced']:
        start_date = period['Start']
        end_date = period['End']
        print(f"\nDrawdown from {start_date.date()} to {end_date.date()}, Max Drawdown: {period['Max Drawdown']:.2%}")
        
        # Performance of VIX30 positions during this period
        long_vix30_masked = enhanced_strategies['Enhanced'].loc[start_date:end_date]
        long_vix30_masked = long_vix30_masked[long_vix30_masked['VIX30_Position'] > 0]
        
        short_vix30_masked = enhanced_strategies['Enhanced'].loc[start_date:end_date]
        short_vix30_masked = short_vix30_masked[short_vix30_masked['VIX30_Position'] < 0]
        
        long_vix30_return = long_vix30_masked['VIX30_Contribution'].sum() if not long_vix30_masked.empty else 0
        short_vix30_return = short_vix30_masked['VIX30_Contribution'].sum() if not short_vix30_masked.empty else 0
        
        print(f"Long VIX30 Contribution: {long_vix30_return:.2%}")
        print(f"Short VIX30 Contribution: {short_vix30_return:.2%}")
    
    # Monte Carlo analysis
    print("\nPerforming Monte Carlo analysis...")
    mc_results = monte_carlo_analysis(
        enhanced_strategies['Enhanced']['Return'],
        benchmark_strategies['S&P 500']['Return'],
        num_samples=1000,  # Reduced for speed
        min_length=252
    )
    
    print("\n--- Monte Carlo Analysis Results ---")
    print(f"Percentage of samples outperforming on return: {mc_results['Outperformance']['Return']:.2f}%")
    print(f"Percentage of samples outperforming on Sharpe: {mc_results['Outperformance']['Sharpe']:.2f}%")
    print(f"Percentage of samples outperforming on Sortino: {mc_results['Outperformance']['Sortino']:.2f}%")
    print(f"Percentage of samples outperforming on MPPM: {mc_results['Outperformance']['MPPM']:.2f}%")
    
    print("\nAverage outperformance when Enhanced beats S&P 500:")
    for metric, value in mc_results['Avg_Outperformance'].items():
        print(f"{metric}: {value:.4f}")
    
    print("\nAverage underperformance when Enhanced lags S&P 500:")
    for metric, value in mc_results['Avg_Underperformance'].items():
        print(f"{metric}: {value:.4f}")
    
    # Annual returns and contributions
    print("\n--- Annual Returns and Component Contributions ---")
    annual_returns = {}
    
    for year in sorted(set(data.index.year)):
        year_slice = slice(f"{year}-01-01", f"{year}-12-31")
        
        # Check if this slice exists in the data
        if not data.loc[year_slice].empty:
            spx_return = (1 + benchmark_strategies['S&P 500'].loc[year_slice, 'Return']).prod() - 1
            enhanced_return = (1 + enhanced_strategies['Enhanced'].loc[year_slice, 'Return']).prod() - 1
            
            # Component contributions for Enhanced
            spx_contrib = enhanced_strategies['Enhanced'].loc[year_slice, 'SPX_Contribution'].sum()
            
            # Long and short VIX30 contributions
            long_vix30_masked = enhanced_strategies['Enhanced'].loc[year_slice]
            long_vix30_masked = long_vix30_masked[long_vix30_masked['VIX30_Position'] > 0]
            
            short_vix30_masked = enhanced_strategies['Enhanced'].loc[year_slice]
            short_vix30_masked = short_vix30_masked[short_vix30_masked['VIX30_Position'] < 0]
            
            long_vix30_contrib = long_vix30_masked['VIX30_Contribution'].sum() if not long_vix30_masked.empty else 0
            short_vix30_contrib = short_vix30_masked['VIX30_Contribution'].sum() if not short_vix30_masked.empty else 0
            
            # Percentage of days long VIX30
            long_vix30_pct = (enhanced_strategies['Enhanced'].loc[year_slice, 'Long_VIX30'].sum() / 
                             len(enhanced_strategies['Enhanced'].loc[year_slice])) * 100
            
            annual_returns[year] = {
                'S&P 500': spx_return,
                'Enhanced': enhanced_return,
                'SPX Contribution': spx_contrib,
                'Long VIX30 Contribution': long_vix30_contrib,
                'Short VIX30 Contribution': short_vix30_contrib,
                'Long VIX30%': long_vix30_pct
            }
    
    annual_returns_df = pd.DataFrame(annual_returns).T
    print(annual_returns_df)
    
    # Calculate statistics about long VIX30 positions
    long_days = enhanced_strategies['Enhanced']['Long_VIX30'].sum()
    total_days = len(enhanced_strategies['Enhanced'])
    long_pct = long_days / total_days * 100
    
    print(f"\nPercentage of days long VIX30: {long_pct:.2f}%")
    
    # Print VIX premium stats
    print("\nVIX Premium Statistics:")
    print(f"Mean: {data['VIX_Premium'].mean():.2f}")
    print(f"Median: {data['VIX_Premium'].median():.2f}")
    print(f"Min: {data['VIX_Premium'].min():.2f}")
    print(f"Max: {data['VIX_Premium'].max():.2f}")
    print(f"Percentage Negative: {(data['VIX_Premium'] < 0).mean() * 100:.2f}%")
    
    print("\nVIX Premium by Market Regime:")
    print(f"Normal Regime Mean: {data[data['Market_Regime'] == 0]['VIX_Premium'].mean():.2f}")
    print(f"Stress Regime Mean: {data[data['Market_Regime'] == 1]['VIX_Premium'].mean():.2f}")
    print(f"Normal Regime % Negative: {(data[data['Market_Regime'] == 0]['VIX_Premium'] < 0).mean() * 100:.2f}%")
    print(f"Stress Regime % Negative: {(data[data['Market_Regime'] == 1]['VIX_Premium'] < 0).mean() * 100:.2f}%")
    
    print("\nAnalysis complete!")
    
    return {
        'data': data,
        'enhanced_strategies': enhanced_strategies,
        'benchmark_strategies': benchmark_strategies,
        'full_period_metrics': full_period_metrics,
        'first_subperiod_metrics': first_subperiod_metrics,
        'second_subperiod_metrics': second_subperiod_metrics,
        'spx_drawdowns': spx_drawdowns,
        'enhanced_drawdowns': enhanced_drawdowns,
        'monte_carlo_results': mc_results,
        'annual_returns': annual_returns_df
    }

# Run the analysis
if __name__ == "__main__":
    results = run_volatility_investing_analysis()