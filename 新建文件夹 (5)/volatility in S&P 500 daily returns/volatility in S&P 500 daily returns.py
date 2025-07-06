import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from scipy.optimize import minimize
import seaborn as sns
from datetime import datetime, timedelta

# Set random seed for reproducibility
np.random.seed(42)

# Define functions from the paper
def calculate_u_v(returns):
    """
    Calculate u and v as defined in the paper.
    u = average log-return of the unleveraged index
    v = average squared daily percentage change
    """
    percentage_changes = returns.pct_change().dropna().values
    log_returns = np.log(1 + percentage_changes)
    
    u = np.mean(log_returns)
    v = np.mean(percentage_changes ** 2)
    
    return u, v

def calculate_higher_moments(returns):
    """
    Calculate higher moments (m3 and m4) as defined in the paper.
    m3 = average cubed daily percentage change
    m4 = average fourth power of daily percentage change
    """
    percentage_changes = returns.pct_change().dropna().values
    
    m3 = np.mean(percentage_changes ** 3)
    m4 = np.mean(percentage_changes ** 4)
    
    return m3, m4

def g_function(L, u, v):
    """
    g(L) function from the paper
    g(L) = (L - 1)(u - L*v/2)
    """
    return (L - 1) * (u - L * v / 2)

def ge_function(L, m3, m4):
    """
    ge(L) function from the paper for higher moments
    ge(L) = m3(L^3 - L)/3 - m4(L^4 - L)/4
    """
    return (L**3 - L) * m3 / 3 - (L**4 - L) * m4 / 4

def optimal_leverage(u, v):
    """
    Calculate L* (the optimal leverage) based on u and v
    L* = u/v + 1/2
    """
    return u/v + 0.5

def d_function(L, u, v, m3=0, m4=0, include_higher_moments=False):
    """
    Annualized return difference between leveraged and unleveraged ETF
    d(L) = 252 * g(L) + 252 * ge(L) if include_higher_moments=True
    """
    result = 252 * g_function(L, u, v)
    
    if include_higher_moments:
        result += 252 * ge_function(L, m3, m4)
        
    return result

def calculate_v_bounds(u, r0, r1):
    """
    Calculate v- and v+ bounds as defined in the paper
    v± = 2(√f(r1,r0) ± √(f(r1,r0) + u))²
    """
    f_r1_r0 = (r1 - r0) / 252  # Approximation of f(r1, r0)
    
    if u < -f_r1_r0:
        return None, None  # Bounds are undefined when u < f(r0, r1)
    
    v_minus = 2 * (np.sqrt(f_r1_r0) - np.sqrt(f_r1_r0 + u)) ** 2
    v_plus = 2 * (np.sqrt(f_r1_r0) + np.sqrt(f_r1_r0 + u)) ** 2
    
    return v_minus, v_plus

def simulate_index_returns(days=1000, initial_price=100, mean_return=0.00035, 
                          volatility=0.01, volatility_trend=0.00000005):
    """
    Simulate index returns with increasing volatility trend
    """
    # Generate random daily returns
    daily_returns = np.random.normal(mean_return, volatility, days)
    
    # Add increasing volatility trend
    volatility_multiplier = 1 + np.arange(days) * volatility_trend
    daily_returns = daily_returns * volatility_multiplier
    
    # Convert to prices
    prices = initial_price * np.cumprod(1 + daily_returns)
    
    # Create DataFrame
    dates = pd.date_range(start='2000-01-01', periods=days, freq='B')
    df = pd.DataFrame({'Close': prices}, index=dates)
    
    return df

def calculate_leveraged_returns(index_returns, leverage, fee_annual=0.0095, 
                               unleveraged_fee_annual=0.0009):
    """
    Calculate the returns of a leveraged ETF with given leverage
    """
    # Calculate daily percentage changes
    pct_change = index_returns.pct_change().dropna()
    
    # Apply leverage
    leveraged_pct_change = leverage * pct_change
    
    # Apply daily fees
    daily_fee = (1 - fee_annual/252)
    daily_unleveraged_fee = (1 - unleveraged_fee_annual/252)
    
    # Calculate cumulative returns
    leveraged_cum_return = (1 + leveraged_pct_change).cumprod() * daily_fee ** np.arange(len(leveraged_pct_change))
    unleveraged_cum_return = (1 + pct_change).cumprod() * daily_unleveraged_fee ** np.arange(len(pct_change))
    
    # Create DataFrame
    result = pd.DataFrame({
        f'{leverage}x ETF': leveraged_cum_return,
        '1x ETF': unleveraged_cum_return
    }, index=pct_change.index)
    
    return result

def backtest_leverage_strategies(index_returns, leverage_range=None, fee=0.0095, 
                               unleveraged_fee=0.0009, window_years=10):
    """
    Backtest different leverage strategies
    """
    if leverage_range is None:
        leverage_range = [-3, -2, -1, 0.5, 1, 2, 3]
    
    # Calculate window size in days
    window_size = window_years * 252
    
    # Initialize results
    results = []
    
    # Loop through each leverage value
    for leverage in leverage_range:
        # Skip leverage = 1 (same as unleveraged)
        if leverage == 1:
            continue
            
        # Calculate leveraged returns
        leveraged_returns = calculate_leveraged_returns(
            index_returns, leverage, fee, unleveraged_fee)
        
        # Calculate rolling performance
        if len(leveraged_returns) > window_size:
            # Calculate rolling return difference
            rolling_return_diff = (
                np.log(leveraged_returns[f'{leverage}x ETF'].rolling(window_size).apply(
                    lambda x: x.iloc[-1]/x.iloc[0])) - 
                np.log(leveraged_returns['1x ETF'].rolling(window_size).apply(
                    lambda x: x.iloc[-1]/x.iloc[0]))
            )
            
            # Annualize
            rolling_return_diff_annual = rolling_return_diff * (252 / window_size)
            
            # Calculate u and v for each window
            u_values = []
            v_values = []
            m3_values = []
            m4_values = []
            estimated_d_values = []
            
            for i in range(window_size, len(index_returns)):
                window_returns = index_returns.iloc[i-window_size:i]
                u, v = calculate_u_v(window_returns['Close'])
                m3, m4 = calculate_higher_moments(window_returns['Close'])
                
                u_values.append(u)
                v_values.append(v)
                m3_values.append(m3)
                m4_values.append(m4)
                
                # Calculate estimated d(L)
                estimated_d = d_function(leverage, u, v, m3, m4, include_higher_moments=True)
                estimated_d_values.append(estimated_d)
            
            # Store results
            temp_results = pd.DataFrame({
                'Date': leveraged_returns.index[window_size:],
                'Leverage': leverage,
                'Actual_Annual_Return_Diff': rolling_return_diff_annual.values,
                'Estimated_Annual_Return_Diff': estimated_d_values,
                'u': u_values,
                'v': v_values,
                'm3': m3_values,
                'm4': m4_values
            })
            
            results.append(temp_results)
    
    # Combine all results
    if results:
        final_results = pd.concat(results, ignore_index=True)
        return final_results
    else:
        return pd.DataFrame()

def plot_volatility_trend(index_returns, window_years=5):
    """
    Plot the volatility trend over time
    """
    # Calculate rolling volatility (sqrt of v)
    window_size = window_years * 252
    rolling_returns = index_returns['Close'].pct_change().dropna()
    rolling_volatility = rolling_returns.rolling(window_size).std() * np.sqrt(252)
    
    plt.figure(figsize=(12, 6))
    rolling_volatility.plot()
    plt.title(f'Rolling {window_years}-Year Annualized Volatility')
    plt.xlabel('Date')
    plt.ylabel('Annualized Volatility')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    # Calculate and plot v (average squared percentage change)
    rolling_v = rolling_returns.rolling(window_size).apply(lambda x: np.mean(x**2))
    
    plt.figure(figsize=(12, 6))
    rolling_v.plot()
    plt.title(f'Rolling {window_years}-Year Average Squared Daily Percentage Change (v)')
    plt.xlabel('Date')
    plt.ylabel('v')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    return rolling_volatility, rolling_v

def calculate_optimal_leverage_over_time(index_returns, window_years=5):
    """
    Calculate optimal leverage multiple over time
    """
    window_size = window_years * 252
    optimal_leverage_values = []
    dates = []
    
    for i in range(window_size, len(index_returns)):
        window_returns = index_returns.iloc[i-window_size:i]
        u, v = calculate_u_v(window_returns['Close'])
        
        # Calculate optimal leverage
        L_star = optimal_leverage(u, v)
        
        optimal_leverage_values.append(L_star)
        dates.append(index_returns.index[i])
    
    # Create DataFrame
    result = pd.DataFrame({
        'Optimal_Leverage': optimal_leverage_values
    }, index=dates)
    
    plt.figure(figsize=(12, 6))
    result['Optimal_Leverage'].plot()
    plt.axhline(y=1, color='r', linestyle='--', label='Unleveraged (1x)')
    plt.axhline(y=2, color='g', linestyle='--', label='2x Leverage')
    plt.axhline(y=3, color='b', linestyle='--', label='3x Leverage')
    plt.title(f'Optimal Leverage Multiple (L*) Over Time ({window_years}-Year Rolling Window)')
    plt.xlabel('Date')
    plt.ylabel('Optimal Leverage (L*)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    return result

def analyze_leverage_performance(backtest_results):
    """
    Analyze the performance of different leverage strategies
    """
    # Group by leverage
    grouped = backtest_results.groupby('Leverage')
    
    # Calculate mean actual and estimated return differences
    mean_differences = grouped[['Actual_Annual_Return_Diff', 'Estimated_Annual_Return_Diff']].mean()
    
    plt.figure(figsize=(10, 6))
    mean_differences.plot(kind='bar')
    plt.title('Average Annual Return Difference (Leveraged ETF - 1x ETF)')
    plt.xlabel('Leverage Multiple')
    plt.ylabel('Annualized Return Difference')
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.show()
    
    # Analyze correlation between actual and estimated return differences
    correlation = backtest_results.groupby('Leverage').apply(
        lambda x: np.corrcoef(x['Actual_Annual_Return_Diff'], x['Estimated_Annual_Return_Diff'])[0, 1])
    
    plt.figure(figsize=(10, 6))
    correlation.plot(kind='bar')
    plt.title('Correlation Between Actual and Estimated Return Differences')
    plt.xlabel('Leverage Multiple')
    plt.ylabel('Correlation Coefficient')
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.show()
    
    # Calculate average absolute error
    absolute_error = backtest_results.groupby('Leverage').apply(
        lambda x: np.mean(np.abs(x['Actual_Annual_Return_Diff'] - x['Estimated_Annual_Return_Diff'])))
    
    plt.figure(figsize=(10, 6))
    absolute_error.plot(kind='bar')
    plt.title('Average Absolute Error Between Actual and Estimated Return Differences')
    plt.xlabel('Leverage Multiple')
    plt.ylabel('Average Absolute Error')
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.show()
    
    return mean_differences, correlation, absolute_error

def volatility_threshold_strategy(index_returns, r0=0.0009, r1=0.0095, window_years=10):
    """
    Implement a strategy based on volatility thresholds
    """
    window_size = window_years * 252
    
    if len(index_returns) <= window_size:
        print("Not enough data for the specified window size")
        return None
    
    # Initialize results
    results = pd.DataFrame(index=index_returns.index[window_size:])
    results['Index_Value'] = index_returns['Close'].values[window_size:]
    
    # Calculate rolling u and v
    u_values = []
    v_values = []
    v_minus_values = []
    v_plus_values = []
    optimal_leverage_values = []
    selected_leverage_values = []
    
    for i in range(window_size, len(index_returns)):
        window_returns = index_returns.iloc[i-window_size:i]
        u, v = calculate_u_v(window_returns['Close'])
        
        u_values.append(u)
        v_values.append(v)
        
        # Calculate v bounds
        v_minus, v_plus = calculate_v_bounds(u, r0, r1)
        v_minus_values.append(v_minus)
        v_plus_values.append(v_plus)
        
        # Calculate optimal leverage
        L_star = optimal_leverage(u, v)
        optimal_leverage_values.append(L_star)
        
        # Strategy: Select leverage based on v relative to bounds
        if v_minus is None or v_plus is None:
            # If bounds are undefined, use unleveraged
            selected_leverage = 1
        elif v < v_minus:
            # If v is below v_minus, use optimal leverage (capped at 3)
            selected_leverage = min(L_star, 3)
        elif v > v_plus:
            # If v is above v_plus, use unleveraged
            selected_leverage = 1
        else:
            # If v is between v_minus and v_plus, use unleveraged
            selected_leverage = 1
        
        selected_leverage_values.append(selected_leverage)
    
    # Store calculated values
    results['u'] = u_values
    results['v'] = v_values
    results['v_minus'] = v_minus_values
    results['v_plus'] = v_plus_values
    results['Optimal_Leverage'] = optimal_leverage_values
    results['Selected_Leverage'] = selected_leverage_values
    
    # Calculate strategy returns
    strategy_returns = []
    unleveraged_returns = []
    fixed_3x_returns = []
    
    for i in range(len(results)):
        if i == 0:
            strategy_returns.append(1)
            unleveraged_returns.append(1)
            fixed_3x_returns.append(1)
        else:
            # Calculate daily return
            daily_return = index_returns['Close'].iloc[window_size+i] / index_returns['Close'].iloc[window_size+i-1] - 1
            
            # Apply leverage from previous day
            leveraged_return = results['Selected_Leverage'].iloc[i-1] * daily_return
            fixed_3x_return = 3 * daily_return
            
            # Apply fees
            leveraged_fee = 1 - (r1 if results['Selected_Leverage'].iloc[i-1] != 1 else r0) / 252
            unleveraged_fee = 1 - r0 / 252
            fixed_3x_fee = 1 - r1 / 252
            
            # Update cumulative returns
            strategy_returns.append(strategy_returns[-1] * (1 + leveraged_return) * leveraged_fee)
            unleveraged_returns.append(unleveraged_returns[-1] * (1 + daily_return) * unleveraged_fee)
            fixed_3x_returns.append(fixed_3x_returns[-1] * (1 + fixed_3x_return) * fixed_3x_fee)
    
    results['Strategy_Value'] = strategy_returns
    results['Unleveraged_Value'] = unleveraged_returns
    results['Fixed_3x_Value'] = fixed_3x_returns
    
    # Calculate summary statistics
    strategy_annual_return = (results['Strategy_Value'].iloc[-1] / results['Strategy_Value'].iloc[0]) ** (252 / len(results)) - 1
    unleveraged_annual_return = (results['Unleveraged_Value'].iloc[-1] / results['Unleveraged_Value'].iloc[0]) ** (252 / len(results)) - 1
    fixed_3x_annual_return = (results['Fixed_3x_Value'].iloc[-1] / results['Fixed_3x_Value'].iloc[0]) ** (252 / len(results)) - 1
    
    strategy_volatility = np.std(np.log(results['Strategy_Value'] / results['Strategy_Value'].shift(1)).dropna()) * np.sqrt(252)
    unleveraged_volatility = np.std(np.log(results['Unleveraged_Value'] / results['Unleveraged_Value'].shift(1)).dropna()) * np.sqrt(252)
    fixed_3x_volatility = np.std(np.log(results['Fixed_3x_Value'] / results['Fixed_3x_Value'].shift(1)).dropna()) * np.sqrt(252)
    
    strategy_sharpe = strategy_annual_return / strategy_volatility
    unleveraged_sharpe = unleveraged_annual_return / unleveraged_volatility
    fixed_3x_sharpe = fixed_3x_annual_return / fixed_3x_volatility
    
    # Calculate drawdowns
    strategy_drawdown = 1 - results['Strategy_Value'] / results['Strategy_Value'].cummax()
    unleveraged_drawdown = 1 - results['Unleveraged_Value'] / results['Unleveraged_Value'].cummax()
    fixed_3x_drawdown = 1 - results['Fixed_3x_Value'] / results['Fixed_3x_Value'].cummax()
    
    max_strategy_drawdown = strategy_drawdown.max()
    max_unleveraged_drawdown = unleveraged_drawdown.max()
    max_fixed_3x_drawdown = fixed_3x_drawdown.max()
    
    # Print summary
    print("\nStrategy Performance Summary:")
    print(f"Annual Return: Strategy={strategy_annual_return:.2%}, Unleveraged={unleveraged_annual_return:.2%}, Fixed 3x={fixed_3x_annual_return:.2%}")
    print(f"Annual Volatility: Strategy={strategy_volatility:.2%}, Unleveraged={unleveraged_volatility:.2%}, Fixed 3x={fixed_3x_volatility:.2%}")
    print(f"Sharpe Ratio: Strategy={strategy_sharpe:.2f}, Unleveraged={unleveraged_sharpe:.2f}, Fixed 3x={fixed_3x_sharpe:.2f}")
    print(f"Maximum Drawdown: Strategy={max_strategy_drawdown:.2%}, Unleveraged={max_unleveraged_drawdown:.2%}, Fixed 3x={max_fixed_3x_drawdown:.2%}")
    
    # Plot strategy performance
    plt.figure(figsize=(12, 6))
    results[['Strategy_Value', 'Unleveraged_Value', 'Fixed_3x_Value']].plot()
    plt.title('Strategy Performance Comparison')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # Plot leverage selection over time
    plt.figure(figsize=(12, 6))
    results['Selected_Leverage'].plot()
    plt.title('Selected Leverage Over Time')
    plt.xlabel('Date')
    plt.ylabel('Leverage')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    # Plot v with bounds
    plt.figure(figsize=(12, 6))
    results['v'].plot(label='v (volatility measure)')
    results['v_minus'].plot(label='v- (lower bound)')
    results['v_plus'].plot(label='v+ (upper bound)')
    plt.title('Volatility Measure (v) with Theoretical Bounds')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    return results

def main():
    # Simulate S&P 500 data with increasing volatility
    print("Simulating S&P 500 data with gradually increasing volatility...")
    simulated_data = simulate_index_returns(
        days=5000,          # About 20 years of trading days
        initial_price=100,
        mean_return=0.00035,  # Approximately 9% annual return
        volatility=0.008,     # Initial volatility
        volatility_trend=0.0000005  # Gradual increase in volatility
    )
    
    # Plot the simulated data
    plt.figure(figsize=(12, 6))
    simulated_data['Close'].plot()
    plt.title('Simulated S&P 500 Index')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    # Analyze volatility trend
    print("\nAnalyzing volatility trend...")
    rolling_vol, rolling_v = plot_volatility_trend(simulated_data)
    
    # Calculate optimal leverage over time
    print("\nCalculating optimal leverage over time...")
    optimal_leverage_df = calculate_optimal_leverage_over_time(simulated_data)
    
    # Backtest different leverage strategies
    print("\nBacktesting leverage strategies...")
    leverage_range = [-3, -2, -1, 0.5, 2, 3]
    backtest_results = backtest_leverage_strategies(
        simulated_data, 
        leverage_range=leverage_range,
        window_years=10
    )
    
    # Analyze leverage performance
    if not backtest_results.empty:
        print("\nAnalyzing leverage performance...")
        mean_diffs, corr, abs_error = analyze_leverage_performance(backtest_results)
        
        # Print results
        print("\nMean Differences (Actual vs. Estimated):")
        print(mean_diffs)
        
        print("\nCorrelations:")
        print(corr)
        
        print("\nAbsolute Errors:")
        print(abs_error)
    else:
        print("Not enough data for backtest analysis")
    
    # Implement volatility threshold strategy
    print("\nImplementing volatility threshold strategy...")
    strategy_results = volatility_threshold_strategy(simulated_data)
    
    # For reference, also fetch real S&P 500 data
    try:
        print("\nFetching real S&P 500 data for reference...")
        sp500_data = yf.download('^GSPC', start='1990-01-01')
        
        # Calculate and plot volatility trend for real data
        print("Analyzing real S&P 500 volatility trend...")
        real_rolling_vol, real_rolling_v = plot_volatility_trend(sp500_data)
        
        # Calculate optimal leverage for real data
        real_optimal_leverage = calculate_optimal_leverage_over_time(sp500_data)
        
    except Exception as e:
        print(f"Error fetching real data: {e}")

if __name__ == "__main__":
    main()