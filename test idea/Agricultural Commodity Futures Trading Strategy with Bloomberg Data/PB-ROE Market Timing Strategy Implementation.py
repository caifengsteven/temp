import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import matplotlib.gridspec as gridspec
from scipy.stats import pearsonr

# Set random seed for reproducibility
np.random.seed(42)

# Configure plot style
plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['font.size'] = 12

# Function to generate simulated data
def generate_data(start_date='2005-01-01', end_date='2020-12-31', freq='W-FRI'):
    """Generate a simulated dataset of market data"""
    # Create date range
    dates = pd.date_range(start=start_date, end=end_date, freq=freq)
    n = len(dates)
    
    # Generate base ROE with seasonality and long-term cycle
    t = np.arange(n)
    trend = 0.05 + 0.03 * np.sin(2 * np.pi * t / 260) + 0.01 * np.sin(2 * np.pi * t / 52)
    roe = trend + 0.015 * np.random.randn(n)
    
    # Generate 10-year interest rate (slow-moving)
    interest_base = 0.03 + 0.015 * np.sin(2 * np.pi * t / 520)
    interest = interest_base + 0.004 * np.random.randn(n)
    interest = np.maximum(interest, 0.005)  # Ensure interest rates are positive
    
    # Generate CPI with seasonality
    cpi_trend = 0.02 + 0.01 * np.sin(2 * np.pi * t / 52) 
    cpi = cpi_trend + 0.002 * np.random.randn(n)
    cpi = np.maximum(cpi, 0)  # Ensure CPI is non-negative
    
    # Calculate real interest rate
    real_interest = interest - cpi
    
    # Model parameters with artificial relationships
    a = 0.5
    b = 4.0  # ROE coefficient
    c = -2.0  # Real interest rate coefficient
    d = -1.0  # CPI coefficient
    
    # Generate PB with noise
    ln_pb_model = a + b * roe + c * real_interest + d * cpi
    noise = 0.05 * np.cumsum(np.random.randn(n))  # Add persistent noise to create meaningful residuals
    ln_pb = ln_pb_model + 0.15 * np.random.randn(n) + noise
    pb = np.exp(ln_pb)
    
    # Generate index price that correlates with PB and ROE
    index_price = np.zeros(n)
    index_price[0] = 1000
    
    # Make returns relate to changes in PB and ROE
    for i in range(1, n):
        pb_change = pb[i] / pb[i-1] - 1
        roe_change = roe[i] - roe[i-1]
        market_return = 0.002 + 0.3 * pb_change + 5 * roe_change + 0.02 * np.random.randn()
        index_price[i] = index_price[i-1] * (1 + market_return)
    
    # Create DataFrame
    data = pd.DataFrame({
        'Date': dates,
        'PB': pb,
        'ROE': roe,
        'Interest_Rate': interest,
        'CPI': cpi,
        'Real_Interest': real_interest,
        'Index': index_price,
        'ln_PB': ln_pb
    })
    
    data.set_index('Date', inplace=True)
    return data

# Function to build the PB-ROE regression model
def build_pb_roe_model(data, window_size=104):
    """Build a rolling PB-ROE regression model and calculate residuals"""
    result_data = data.copy()
    result_data['Fitted_ln_PB'] = np.nan
    result_data['Residual'] = np.nan
    result_data['T_Value'] = np.nan  # Investment period (T)
    
    X_cols = ['ROE', 'Real_Interest', 'CPI']
    
    for i in range(window_size, len(data)):
        # Get window data
        window_data = data.iloc[i-window_size:i]
        
        # Prepare variables
        X = window_data[X_cols]
        y = window_data['ln_PB']
        
        # Add constant for intercept
        X_with_const = sm.add_constant(X)
        
        # Fit model
        model = sm.OLS(y, X_with_const).fit()
        
        # Get coefficients
        a = model.params['const']
        b = model.params['ROE']  # This is the T value
        c = model.params['Real_Interest']
        d = model.params['CPI']
        
        # Use coefficients directly to calculate fitted value
        current_values = data.iloc[i][X_cols]
        fitted_ln_pb = (a + 
                       b * current_values['ROE'] + 
                       c * current_values['Real_Interest'] + 
                       d * current_values['CPI'])
        
        # Calculate residual
        residual = data.iloc[i]['ln_PB'] - fitted_ln_pb
        
        # Store results
        result_data.iloc[i, result_data.columns.get_loc('Fitted_ln_PB')] = fitted_ln_pb
        result_data.iloc[i, result_data.columns.get_loc('Residual')] = residual
        
        # Calculate T value (investment period)
        if b < 0:
            # Check if PB is rising and ROE is falling
            if (window_data['PB'].iloc[-1] > window_data['PB'].iloc[0] and 
                window_data['ROE'].iloc[-1] < window_data['ROE'].iloc[0]):
                # Use positive value
                result_data.iloc[i, result_data.columns.get_loc('T_Value')] = np.arctan(abs(b)) + np.pi/2
            else:
                # If PB is falling and ROE is rising
                result_data.iloc[i, result_data.columns.get_loc('T_Value')] = np.arctan(b)
        else:
            # Normal case
            result_data.iloc[i, result_data.columns.get_loc('T_Value')] = np.arctan(b)
    
    return result_data

# Function to calculate future returns
def calculate_future_returns(data, forward_periods=[1, 4, 13, 26, 52]):
    """Calculate future returns for various periods"""
    result_data = data.copy()
    
    for period in forward_periods:
        col_name = f'Future_Return_{period}w'
        result_data[col_name] = np.nan
        
        for i in range(len(data) - period):
            future_return = (data['Index'].iloc[i + period] / data['Index'].iloc[i]) - 1
            result_data.iloc[i, result_data.columns.get_loc(col_name)] = future_return
    
    return result_data

# Strategy 1: Buy when residuals make new highs, sell when they make new lows
def strategy_1(data, lookback=13):
    """Strategy 1: Buy when residuals make new highs, sell when they make new lows"""
    result_data = data.copy()
    result_data['Position_S1'] = 0
    
    # Find first valid data point
    valid_idx = np.where(~np.isnan(data['Residual'].values))[0]
    if len(valid_idx) == 0:
        return result_data
    
    start_idx = valid_idx[0] + lookback
    
    # Skip the initial lookback period
    for i in range(start_idx, len(data)):
        if np.isnan(data['Residual'].iloc[i]):
            continue
            
        window = data['Residual'].iloc[i-lookback:i]
        current_residual = data['Residual'].iloc[i]
        
        # Check for new high or new low
        if current_residual > window.max():
            # New high - Buy
            result_data.iloc[i, result_data.columns.get_loc('Position_S1')] = 1
        elif current_residual < window.min():
            # New low - Sell
            result_data.iloc[i, result_data.columns.get_loc('Position_S1')] = -1
        else:
            # No change - maintain previous position
            result_data.iloc[i, result_data.columns.get_loc('Position_S1')] = result_data.iloc[i-1, result_data.columns.get_loc('Position_S1')]
    
    return result_data

# Strategy 2: Buy when residuals move above a threshold, sell when they move below a threshold
def strategy_2(data, threshold_upper=0.1, threshold_lower=-0.1):
    """Strategy 2: Buy when residuals move above threshold, sell when they move below threshold"""
    result_data = data.copy()
    result_data['Position_S2'] = 0
    
    # Find first valid data point
    valid_idx = np.where(~np.isnan(data['Residual'].values))[0]
    if len(valid_idx) == 0:
        return result_data
    
    start_idx = valid_idx[0]
    
    for i in range(start_idx, len(data)):
        if np.isnan(data['Residual'].iloc[i]):
            continue
            
        current_residual = data['Residual'].iloc[i]
        
        if current_residual > threshold_upper:
            # Above upper threshold - Buy
            result_data.iloc[i, result_data.columns.get_loc('Position_S2')] = 1
        elif current_residual < threshold_lower:
            # Below lower threshold - Sell
            result_data.iloc[i, result_data.columns.get_loc('Position_S2')] = -1
        else:
            # Between thresholds - maintain previous position
            if i > start_idx:
                result_data.iloc[i, result_data.columns.get_loc('Position_S2')] = result_data.iloc[i-1, result_data.columns.get_loc('Position_S2')]
    
    return result_data

# Calculate strategy returns
def calculate_strategy_returns(data, tc=0.001):
    """Calculate returns for both strategies"""
    result_data = data.copy()
    
    # Calculate index returns
    result_data['Index_Return'] = result_data['Index'].pct_change()
    
    # Initialize return columns
    result_data['Return_S1'] = 0.0
    result_data['Return_S2'] = 0.0
    
    # Initialize cumulative return columns
    result_data['Cum_Return_S1'] = 1.0
    result_data['Cum_Return_S2'] = 1.0
    
    # Calculate transaction costs and returns for both strategies
    for i in range(1, len(result_data)):
        # Strategy 1
        pos_change_s1 = abs(result_data['Position_S1'].iloc[i] - result_data['Position_S1'].iloc[i-1])
        tc_s1 = pos_change_s1 * tc if pos_change_s1 > 0 else 0
        
        if result_data['Position_S1'].iloc[i-1] != 0:  # If we had a position
            result_data.iloc[i, result_data.columns.get_loc('Return_S1')] = (
                result_data['Position_S1'].iloc[i-1] * result_data['Index_Return'].iloc[i] - tc_s1
            )
        else:
            result_data.iloc[i, result_data.columns.get_loc('Return_S1')] = -tc_s1
        
        # Strategy 2
        pos_change_s2 = abs(result_data['Position_S2'].iloc[i] - result_data['Position_S2'].iloc[i-1])
        tc_s2 = pos_change_s2 * tc if pos_change_s2 > 0 else 0
        
        if result_data['Position_S2'].iloc[i-1] != 0:  # If we had a position
            result_data.iloc[i, result_data.columns.get_loc('Return_S2')] = (
                result_data['Position_S2'].iloc[i-1] * result_data['Index_Return'].iloc[i] - tc_s2
            )
        else:
            result_data.iloc[i, result_data.columns.get_loc('Return_S2')] = -tc_s2
    
    # Calculate cumulative returns
    for i in range(1, len(result_data)):
        result_data.iloc[i, result_data.columns.get_loc('Cum_Return_S1')] = (
            result_data['Cum_Return_S1'].iloc[i-1] * (1 + result_data['Return_S1'].iloc[i])
        )
        result_data.iloc[i, result_data.columns.get_loc('Cum_Return_S2')] = (
            result_data['Cum_Return_S2'].iloc[i-1] * (1 + result_data['Return_S2'].iloc[i])
        )
    
    return result_data

# Evaluate strategy performance
def evaluate_strategy(data, strategy_name):
    """Evaluate a trading strategy"""
    # Get valid data points
    valid_data = data.dropna(subset=[f'Return_{strategy_name}'])
    
    # Calculate metrics
    total_return = data[f'Cum_Return_{strategy_name}'].iloc[-1] - 1
    annual_return = (1 + total_return) ** (52 / len(valid_data)) - 1
    
    returns = valid_data[f'Return_{strategy_name}']
    volatility = returns.std() * np.sqrt(52)  # Annualized
    sharpe_ratio = annual_return / volatility if volatility > 0 else 0
    
    # Calculate max drawdown
    cum_returns = data[f'Cum_Return_{strategy_name}']
    running_max = cum_returns.cummax()
    drawdown = (cum_returns / running_max) - 1
    max_drawdown = drawdown.min()
    
    # Calculate trade statistics
    position_changes = valid_data[f'Position_{strategy_name}'].diff().abs()
    total_trades = position_changes[position_changes > 0].count()
    
    # Calculate positions and holding times
    positions = valid_data[f'Position_{strategy_name}']
    long_periods = positions[positions > 0].count()
    short_periods = positions[positions < 0].count()
    neutral_periods = positions[positions == 0].count()
    
    # Find longest consecutive positions
    consecutive_counts = {'long': 0, 'short': 0, 'neutral': 0}
    current_type = None
    current_count = 0
    
    for pos in positions:
        if pos > 0:
            pos_type = 'long'
        elif pos < 0:
            pos_type = 'short'
        else:
            pos_type = 'neutral'
            
        if pos_type == current_type:
            current_count += 1
        else:
            if current_type:
                consecutive_counts[current_type] = max(consecutive_counts[current_type], current_count)
            current_type = pos_type
            current_count = 1
    
    # Handle the last position
    if current_type:
        consecutive_counts[current_type] = max(consecutive_counts[current_type], current_count)
    
    return {
        'Total Return': total_return,
        'Annual Return': annual_return,
        'Volatility': volatility,
        'Sharpe Ratio': sharpe_ratio,
        'Max Drawdown': max_drawdown,
        'Total Trades': total_trades,
        'Long Periods': long_periods,
        'Short Periods': short_periods,
        'Neutral Periods': neutral_periods,
        'Longest Long': consecutive_counts['long'],
        'Longest Short': consecutive_counts['short'],
        'Longest Neutral': consecutive_counts['neutral']
    }

# Parameter sensitivity analysis
def parameter_sensitivity(data, lookback_range=[8, 13, 21, 26], threshold_pairs=[(0.05, -0.05), (0.1, -0.1), (0.15, -0.15)]):
    """Analyze strategy sensitivity to different parameters"""
    results = {
        'Strategy1': {},
        'Strategy2': {}
    }
    
    # Test Strategy 1 with different lookback periods
    for lookback in lookback_range:
        test_data = strategy_1(data, lookback=lookback)
        test_data = calculate_strategy_returns(test_data)
        results['Strategy1'][lookback] = evaluate_strategy(test_data, 'S1')
    
    # Test Strategy 2 with different thresholds
    for upper, lower in threshold_pairs:
        test_data = strategy_2(data, threshold_upper=upper, threshold_lower=lower)
        test_data = calculate_strategy_returns(test_data)
        results['Strategy2'][(upper, lower)] = evaluate_strategy(test_data, 'S2')
    
    return results

def main():
    # Generate simulated data
    print("Generating simulated market data...")
    data = generate_data(start_date='2009-01-01', end_date='2019-12-31')
    
    # Plot the data
    print("Plotting raw data...")
    fig, axs = plt.subplots(5, 1, figsize=(14, 18), sharex=True)

    axs[0].plot(data.index, data['Index'])
    axs[0].set_title('Market Index')
    axs[0].set_ylabel('Price')
    axs[0].grid(True)

    axs[1].plot(data.index, data['PB'])
    axs[1].set_title('Price-to-Book (PB) Ratio')
    axs[1].set_ylabel('PB')
    axs[1].grid(True)

    axs[2].plot(data.index, data['ROE'])
    axs[2].set_title('Return on Equity (ROE)')
    axs[2].set_ylabel('ROE')
    axs[2].grid(True)

    axs[3].plot(data.index, data['Interest_Rate'], label='Interest Rate')
    axs[3].plot(data.index, data['CPI'], label='CPI')
    axs[3].plot(data.index, data['Real_Interest'], label='Real Interest Rate')
    axs[3].set_title('Interest Rates and CPI')
    axs[3].set_ylabel('Rate')
    axs[3].legend()
    axs[3].grid(True)

    axs[4].scatter(data['ROE'], data['ln_PB'], alpha=0.5)
    axs[4].set_title('ROE vs ln(PB)')
    axs[4].set_xlabel('ROE')
    axs[4].set_ylabel('ln(PB)')
    axs[4].grid(True)

    plt.tight_layout()
    plt.show()

    # Build PB-ROE model with 2-year (104 weeks) rolling window
    print("Building PB-ROE model...")
    model_results = build_pb_roe_model(data, window_size=104)

    # Calculate future returns
    print("Calculating future returns...")
    data_with_returns = calculate_future_returns(model_results)

    # Filter to only valid data for correlation analysis
    valid_data = data_with_returns.dropna(subset=['Residual'])
    
    # Analyze correlation between residuals and future returns
    print("Analyzing correlation between residuals and future returns...")
    correlations = []
    forward_periods = [1, 4, 13, 26, 52]  # 1 week to 1 year

    for period in forward_periods:
        col_name = f'Future_Return_{period}w'
        # Make sure we have both residual and future return data
        temp_data = valid_data.dropna(subset=[col_name])
        
        if len(temp_data) > 0:
            corr, _ = pearsonr(temp_data['Residual'], temp_data[col_name])
            correlations.append((period, corr))

    # Plot correlations if we have any
    if correlations:
        periods, corrs = zip(*correlations)
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(periods)), corrs)
        plt.xticks(range(len(periods)), [f"{p}w" for p in periods])
        plt.xlabel('Forward Return Period')
        plt.ylabel('Correlation Coefficient')
        plt.title('Correlation between PB-ROE Residuals and Future Returns')
        plt.axhline(y=0, color='r', linestyle='-')
        plt.grid(True)
        plt.show()

    # Compare actual vs fitted PB
    print("Comparing actual vs fitted PB...")
    valid_data = model_results.dropna(subset=['ln_PB', 'Fitted_ln_PB'])
    
    if len(valid_data) > 0:
        plt.figure(figsize=(14, 7))
        plt.plot(valid_data.index, np.exp(valid_data['ln_PB']), label='Actual PB', color='blue')
        plt.plot(valid_data.index, np.exp(valid_data['Fitted_ln_PB']), label='Fitted PB', color='red')
        plt.fill_between(valid_data.index, 
                        np.exp(valid_data['ln_PB']), 
                        np.exp(valid_data['Fitted_ln_PB']), 
                        alpha=0.3, color='gray')
        plt.title('Actual vs. Fitted PB')
        plt.xlabel('Date')
        plt.ylabel('PB Ratio')
        plt.legend()
        plt.grid(True)
        plt.show()

    # Plot residuals
    print("Plotting residuals...")
    plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1])

    ax1 = plt.subplot(gs[0])
    ax1.plot(data.index, data['Index'], label='Market Index')
    ax1.set_title('Market Index')
    ax1.set_ylabel('Index Value')
    ax1.legend()
    ax1.grid(True)

    ax2 = plt.subplot(gs[1], sharex=ax1)
    valid_data = model_results.dropna(subset=['Residual'])
    if len(valid_data) > 0:
        ax2.plot(valid_data.index, valid_data['Residual'], label='PB-ROE Residual', color='green')
        ax2.fill_between(valid_data.index, 0, valid_data['Residual'], 
                        where=valid_data['Residual'] > 0, 
                        alpha=0.3, color='green', interpolate=True)
        ax2.fill_between(valid_data.index, valid_data['Residual'], 0, 
                        where=valid_data['Residual'] < 0, 
                        alpha=0.3, color='red', interpolate=True)
    ax2.set_title('PB-ROE Model Residuals')
    ax2.set_ylabel('Residual Value')
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

    # Apply trading strategies
    print("Applying trading strategies...")
    strategy_results = strategy_1(model_results, lookback=13)
    strategy_results = strategy_2(strategy_results, threshold_upper=0.1, threshold_lower=-0.1)

    # Calculate strategy returns
    print("Calculating strategy returns...")
    final_results = calculate_strategy_returns(strategy_results)

    # Find valid data starting point
    valid_idx = np.where(~np.isnan(model_results['Residual'].values))[0]
    if len(valid_idx) > 0:
        start_idx = valid_idx[0]
        
        # Add buy-and-hold strategy
        final_results['Cum_Return_BH'] = final_results['Index'] / final_results['Index'].iloc[start_idx]

        # Evaluate strategies
        print("Evaluating strategies...")
        results_s1 = evaluate_strategy(final_results, 'S1')
        results_s2 = evaluate_strategy(final_results, 'S2')

        # Display results
        print("\nStrategy 1 Performance:")
        for key, value in results_s1.items():
            if 'Return' in key or 'Drawdown' in key:
                print(f"{key}: {value:.2%}")
            else:
                print(f"{key}: {value:.2f}")
                
        print("\nStrategy 2 Performance:")
        for key, value in results_s2.items():
            if 'Return' in key or 'Drawdown' in key:
                print(f"{key}: {value:.2%}")
            else:
                print(f"{key}: {value:.2f}")

        # Plot strategy positions and cumulative returns
        print("Plotting strategy results...")
        plt.figure(figsize=(16, 12))

        # Get plot data from first valid point
        plot_data = final_results.iloc[start_idx:]

        # Plot market index and residuals
        ax1 = plt.subplot(3, 1, 1)
        ax1.plot(plot_data.index, plot_data['Index'], label='Market Index')
        ax1.set_title('Market Index')
        ax1.set_ylabel('Index Value')
        ax1.legend()
        ax1.grid(True)

        ax2 = plt.subplot(3, 1, 2, sharex=ax1)
        ax2.plot(plot_data.index, plot_data['Residual'], label='PB-ROE Residual', color='green')
        ax2.fill_between(plot_data.index, 0, plot_data['Residual'], where=plot_data['Residual'] > 0, 
                         alpha=0.3, color='green', interpolate=True)
        ax2.fill_between(plot_data.index, plot_data['Residual'], 0, where=plot_data['Residual'] < 0, 
                         alpha=0.3, color='red', interpolate=True)

        # Plot strategy positions
        ax2.plot(plot_data.index, plot_data['Position_S1'], 'o-', markersize=3, label='Strategy 1 Position', color='blue')
        ax2.plot(plot_data.index, plot_data['Position_S2'], 'o-', markersize=3, label='Strategy 2 Position', color='purple')
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax2.set_title('Model Residuals and Strategy Positions')
        ax2.set_ylabel('Value')
        ax2.legend()
        ax2.grid(True)

        # Plot cumulative returns
        ax3 = plt.subplot(3, 1, 3, sharex=ax1)
        ax3.plot(plot_data.index, plot_data['Cum_Return_S1'], label='Strategy 1', color='blue')
        ax3.plot(plot_data.index, plot_data['Cum_Return_S2'], label='Strategy 2', color='purple')
        ax3.plot(plot_data.index, plot_data['Cum_Return_BH'], label='Buy & Hold', color='gray')
        ax3.set_title('Cumulative Returns')
        ax3.set_xlabel('Date')
        ax3.set_ylabel('Cumulative Return')
        ax3.legend()
        ax3.grid(True)

        plt.tight_layout()
        plt.show()

        # Plot the T-value (investment period) over time
        print("Plotting T-value...")
        plt.figure(figsize=(14, 7))
        ax1 = plt.gca()
        ax2 = ax1.twinx()

        ax1.plot(plot_data.index, plot_data['Index'], label='Market Index', color='blue')
        ax1.set_ylabel('Index Value', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')
        ax1.grid(True)

        ax2.plot(plot_data.index, plot_data['T_Value'], label='Investment Period (T)', color='red')
        ax2.set_ylabel('T Value (radians)', color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        ax2.axhline(y=0, color='gray', linestyle='--')
        ax2.axhline(y=2, color='gray', linestyle='--')

        plt.title('Market Index vs Investment Period (T)')
        plt.show()

        # Run parameter sensitivity analysis
        print("Running parameter sensitivity analysis...")
        sensitivity_results = parameter_sensitivity(model_results)

        # Plot annual returns for different parameters
        print("Plotting parameter sensitivity...")
        plt.figure(figsize=(14, 6))

        # Strategy 1 - Lookback periods
        plt.subplot(1, 2, 1)
        lookbacks = list(sensitivity_results['Strategy1'].keys())
        annual_returns = [results['Annual Return'] for results in sensitivity_results['Strategy1'].values()]
        sharpe_ratios = [results['Sharpe Ratio'] for results in sensitivity_results['Strategy1'].values()]

        ax1 = plt.gca()
        ax2 = ax1.twinx()

        ax1.bar(range(len(lookbacks)), annual_returns, color='blue', alpha=0.6, label='Annual Return')
        ax1.set_xticks(range(len(lookbacks)))
        ax1.set_xticklabels([f"{lb}w" for lb in lookbacks])
        ax1.set_xlabel('Lookback Period')
        ax1.set_ylabel('Annual Return', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')
        ax1.grid(True)

        ax2.plot(range(len(lookbacks)), sharpe_ratios, 'o-', color='red', label='Sharpe Ratio')
        ax2.set_ylabel('Sharpe Ratio', color='red')
        ax2.tick_params(axis='y', labelcolor='red')

        plt.title('Strategy 1 Parameter Sensitivity')

        # Strategy 2 - Thresholds
        plt.subplot(1, 2, 2)
        thresholds = list(sensitivity_results['Strategy2'].keys())
        annual_returns = [results['Annual Return'] for results in sensitivity_results['Strategy2'].values()]
        sharpe_ratios = [results['Sharpe Ratio'] for results in sensitivity_results['Strategy2'].values()]

        ax1 = plt.gca()
        ax2 = ax1.twinx()

        ax1.bar(range(len(thresholds)), annual_returns, color='purple', alpha=0.6, label='Annual Return')
        ax1.set_xticks(range(len(thresholds)))
        ax1.set_xticklabels([f"({up}, {low})" for up, low in thresholds], rotation=45)
        ax1.set_xlabel('Threshold Levels (Upper, Lower)')
        ax1.set_ylabel('Annual Return', color='purple')
        ax1.tick_params(axis='y', labelcolor='purple')
        ax1.grid(True)

        ax2.plot(range(len(thresholds)), sharpe_ratios, 'o-', color='orange', label='Sharpe Ratio')
        ax2.set_ylabel('Sharpe Ratio', color='orange')
        ax2.tick_params(axis='y', labelcolor='orange')

        plt.title('Strategy 2 Parameter Sensitivity')
        plt.tight_layout()
        plt.show()
    else:
        print("No valid residual data found. Check the model parameters.")

    print("Analysis complete!")

if __name__ == "__main__":
    main()