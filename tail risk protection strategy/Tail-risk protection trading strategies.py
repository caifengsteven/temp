import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy import optimize
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

def generate_synthetic_data(n_days=2000):
    """
    Generate synthetic price data mimicking financial market behavior
    """
    # Initialize arrays
    prices = np.zeros(n_days)
    prices[0] = 5000  # Initial price
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Parameters
    vol = 0.015  # Base volatility
    
    # Generate prices with occasional jumps
    for i in range(1, n_days):
        # Add jumps occasionally
        jump = 0
        if np.random.random() < 0.005:  # 0.5% chance of jump
            jump_direction = -1 if np.random.random() < 0.7 else 1  # More negative jumps
            jump = jump_direction * np.random.uniform(0.02, 0.05)
            
        # Daily return calculation
        daily_return = np.random.normal(0, vol) + jump
        prices[i] = prices[i-1] * (1 + daily_return)
    
    # Generate dates
    dates = pd.date_range(start='2010-01-01', periods=n_days, freq='B')
    
    # Create price DataFrame
    return pd.DataFrame({'Price': prices}, index=dates)

def calculate_returns(prices):
    """
    Calculate simple returns from price data
    """
    return prices.pct_change().fillna(0)

def fit_garch(returns, window_size=100):
    """
    Simplified GARCH(1,1) volatility estimation
    """
    window_returns = returns.iloc[-window_size:].values
    
    # Default parameters
    alpha0 = 1e-6
    alpha1 = 0.1
    beta = 0.85
    
    # Very basic GARCH implementation
    volatility = np.zeros(window_size)
    volatility[0] = np.std(window_returns)
    
    for t in range(1, window_size):
        volatility[t] = np.sqrt(alpha0 + alpha1 * window_returns[t-1]**2 + beta * volatility[t-1]**2)
    
    # Return last volatility as forecast
    return volatility[-1]

def extreme_value_var(returns, window_size=100, confidence_level=0.99, threshold_pct=0.9):
    """
    Calculate Value at Risk using extreme value theory (simplified)
    """
    # Get window of returns
    window_returns = returns.iloc[-window_size:].values
    
    # Sort returns in ascending order (negative = losses)
    sorted_returns = np.sort(window_returns)
    
    # Find threshold for extreme values (e.g., worst 10%)
    threshold_idx = int(window_size * threshold_pct)
    threshold = sorted_returns[threshold_idx]
    
    # Extract exceedances (values below threshold)
    exceedances = sorted_returns[:threshold_idx] - threshold
    
    if len(exceedances) < 5:
        # Not enough data for GPD fit, use normal distribution
        return stats.norm.ppf(1-confidence_level, loc=np.mean(window_returns), scale=np.std(window_returns))
    
    # Simplified GPD parameter estimation
    shape = 0.2  # Typical value for financial returns
    scale = np.mean(np.abs(exceedances))
    
    # Calculate VaR
    p = confidence_level
    n = len(window_returns)
    Nu = len(exceedances)
    
    # VaR calculation based on GPD
    var_gpd = threshold + (scale/shape) * (((n/Nu) * (1-p))**(-shape) - 1)
    
    return var_gpd

def normal_var(returns, window_size=100, confidence_level=0.99):
    """
    Calculate Value at Risk assuming normal distribution
    """
    # Get window of returns
    window_returns = returns.iloc[-window_size:].values
    
    # Calculate VaR using normal distribution
    mean = np.mean(window_returns)
    std = np.std(window_returns)
    var_normal = stats.norm.ppf(1-confidence_level, loc=mean, scale=std)
    
    return var_normal

def calculate_var_spread_series(returns, window_size=100, lookback=5):
    """
    Calculate VaR spread between extreme value and normal approaches
    """
    n_days = len(returns)
    var_normal = np.zeros(n_days)
    var_extreme = np.zeros(n_days)
    var_spread = np.zeros(n_days)
    
    print("Calculating VaR spreads...")
    
    # Skip initial window where we don't have enough data
    for i in range(window_size, n_days):
        if i % 500 == 0:
            print(f"Processing day {i}/{n_days}")
            
        # Get window of data
        window = returns.iloc[i-window_size:i]
        
        # Calculate volatility using GARCH
        vol = fit_garch(window)
        
        # Calculate VaR using normal and extreme value approaches
        var_normal[i] = normal_var(window)
        var_extreme[i] = extreme_value_var(window)
        
        # Calculate spread
        var_spread[i] = var_extreme[i] - var_normal[i]
    
    # Create DataFrame
    var_df = pd.DataFrame({
        'VaR_Normal': var_normal,
        'VaR_Extreme': var_extreme,
        'VaR_Spread': var_spread
    }, index=returns.index)
    
    return var_df

def calculate_slope(series, lookback=10):
    """
    Calculate slope of a time series over the last lookback periods
    """
    if len(series) < lookback:
        return 0
    
    y = series[-lookback:].values
    x = np.arange(1, lookback+1)
    
    # Add constant for regression
    X = sm.add_constant(x)
    
    try:
        # Simple linear regression
        model = sm.OLS(y, X).fit()
        return model.params[1]  # Return slope coefficient
    except:
        return 0

def generate_trading_signals(var_spread, lookback=10, threshold_factor=1.5):
    """
    Generate trading signals based on the steepness of VaR spread
    """
    n_days = len(var_spread)
    signals = np.ones(n_days)  # Start with fully invested
    
    # Calculate slopes for each day
    slopes = np.zeros(n_days)
    
    for i in range(lookback, n_days):
        # Calculate slope of VaR spread
        slopes[i] = calculate_slope(var_spread[:i], lookback)
        
        # Calculate average positive slope up to current day
        positive_slopes = slopes[lookback:i][slopes[lookback:i] > 0]
        avg_positive_slope = np.mean(positive_slopes) if len(positive_slopes) > 0 else 0
        
        # Generate signal (1=invest, 0=hedge)
        # Signal to hedge (0) if slope exceeds threshold * avg_positive_slope
        if np.isfinite(avg_positive_slope) and avg_positive_slope > 0:
            signals[i] = 0 if slopes[i] > threshold_factor * avg_positive_slope else 1
    
    return pd.Series(signals, index=var_spread.index)

def calculate_strategy_returns(returns, signals):
    """
    Calculate strategy returns based on trading signals
    """
    strategy_returns = returns * signals
    return strategy_returns

def calculate_protective_put(returns, strike_pct=0.9, option_maturity=63, implied_vol=0.2):
    """
    Simplified protective put strategy
    """
    from scipy.stats import norm
    
    # Calculate price series
    price = 100 * (1 + returns).cumprod()
    
    # Initialize return array
    pp_returns = np.zeros_like(returns.values)
    
    # Risk-free rate
    rf = 0.01
    
    # Loop through days
    for i in range(1, len(returns)):
        # Every option_maturity days, buy a new put
        if i % option_maturity == 0:
            # Current price and strike
            S = price.iloc[i-1]
            K = S * strike_pct
            
            # Black-Scholes for put option price
            d1 = (np.log(S/K) + (rf + 0.5*implied_vol**2)*(option_maturity/252)) / (implied_vol*np.sqrt(option_maturity/252))
            d2 = d1 - implied_vol*np.sqrt(option_maturity/252)
            put_price = K*np.exp(-rf*option_maturity/252)*norm.cdf(-d2) - S*norm.cdf(-d1)
            
            # Buy put (reduce return)
            pp_returns[i] = returns.iloc[i] - put_price/S
        else:
            # No put purchase today
            pp_returns[i] = returns.iloc[i]
    
    return pd.Series(pp_returns, index=returns.index)

def calculate_delta_replicated_put(returns, strike_pct=0.9, implied_vol=0.2):
    """
    Simplified delta-replicated put strategy
    """
    from scipy.stats import norm
    
    # Calculate price series
    price = 100 * (1 + returns).cumprod()
    
    # Initialize arrays
    drp_returns = np.zeros_like(returns.values)
    deltas = np.zeros_like(returns.values)
    
    # Risk-free rate and time to maturity (constant)
    rf = 0.01
    T = 63/252  # 3 months
    
    # Loop through days
    for i in range(1, len(returns)):
        # Current price and strike
        S = price.iloc[i-1]
        K = S * strike_pct
        
        # Calculate put delta
        d1 = (np.log(S/K) + (rf + 0.5*implied_vol**2)*T) / (implied_vol*np.sqrt(T))
        put_delta = norm.cdf(-d1) - 1  # Negative delta
        
        # Store delta
        deltas[i-1] = put_delta
        
        # Calculate strategy return
        drp_returns[i] = returns.iloc[i] * (1 + put_delta)
    
    return pd.Series(drp_returns, index=returns.index)

def calculate_performance_metrics(returns):
    """
    Calculate performance metrics
    """
    # Calculate annualized return
    annual_return = ((1 + returns).prod()) ** (252 / len(returns)) - 1
    
    # Calculate volatility
    annual_volatility = returns.std() * np.sqrt(252)
    
    # Calculate Sharpe ratio
    sharpe_ratio = annual_return / annual_volatility if annual_volatility > 0 else 0
    
    # Calculate drawdown
    cum_returns = (1 + returns).cumprod()
    peak = cum_returns.expanding(min_periods=1).max()
    drawdown = (cum_returns - peak) / peak
    max_drawdown = drawdown.min()
    
    # Calculate Calmar ratio
    calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
    
    # Calculate success rate
    success_rate = (returns > 0).mean()
    
    metrics = {
        'Annual Return': annual_return,
        'Annual Volatility': annual_volatility,
        'Sharpe Ratio': sharpe_ratio,
        'Max Drawdown': max_drawdown,
        'Calmar Ratio': calmar_ratio,
        'Success Rate': success_rate
    }
    
    return metrics

def plot_results(benchmark_returns, strategy_returns, pp_returns, drp_returns, trading_signals=None):
    """
    Plot strategy results
    
    Parameters:
    -----------
    benchmark_returns : pandas.Series
        Benchmark returns
    strategy_returns : pandas.Series
        Tail-risk protection strategy returns
    pp_returns : pandas.Series
        Protective put strategy returns
    drp_returns : pandas.Series
        Delta-replicated put strategy returns
    trading_signals : pandas.Series, optional
        Trading signals used for the strategy
    """
    # Calculate cumulative returns
    cum_benchmark = (1 + benchmark_returns).cumprod() - 1
    cum_strategy = (1 + strategy_returns).cumprod() - 1
    cum_pp = (1 + pp_returns).cumprod() - 1
    cum_drp = (1 + drp_returns).cumprod() - 1
    
    # Calculate drawdowns
    peak_benchmark = cum_benchmark.expanding(min_periods=1).max()
    dd_benchmark = (cum_benchmark - peak_benchmark) / (1 + peak_benchmark)
    
    peak_strategy = cum_strategy.expanding(min_periods=1).max()
    dd_strategy = (cum_strategy - peak_strategy) / (1 + peak_strategy)
    
    peak_pp = cum_pp.expanding(min_periods=1).max()
    dd_pp = (cum_pp - peak_pp) / (1 + peak_pp)
    
    peak_drp = cum_drp.expanding(min_periods=1).max()
    dd_drp = (cum_drp - peak_drp) / (1 + peak_drp)
    
    # Create figure
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot cumulative returns
    axes[0].plot(cum_benchmark, 'b-', label='Benchmark')
    axes[0].plot(cum_strategy, 'g-', label='Tail-Risk Protection')
    axes[0].plot(cum_pp, 'r-', label='Protective Put')
    axes[0].plot(cum_drp, 'y-', label='Delta-Replicated Put')
    axes[0].set_title('Cumulative Returns')
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot drawdowns
    axes[1].plot(dd_benchmark, 'b-', label='Benchmark')
    axes[1].plot(dd_strategy, 'g-', label='Tail-Risk Protection')
    axes[1].plot(dd_pp, 'r-', label='Protective Put')
    axes[1].plot(dd_drp, 'y-', label='Delta-Replicated Put')
    axes[1].set_title('Drawdowns')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Plot additional information if trading signals are provided
    if trading_signals is not None:
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(trading_signals, 'g-')
        ax.set_title('Trading Signals (1=Invested, 0=Hedged)')
        ax.set_ylim(-0.1, 1.1)
        ax.grid(True)
        plt.show()

def optimize_parameters(returns, var_df, in_sample_end):
    """
    Optimize strategy parameters using in-sample data
    """
    print("Optimizing strategy parameters...")
    
    # Define range of parameters to test
    lookback_values = range(5, 16)  # 5 to 15
    threshold_values = np.linspace(0.5, 3.0, 6)  # 0.5 to 3.0
    
    # Get in-sample data
    in_sample_returns = returns.iloc[:in_sample_end]
    in_sample_var_spread = var_df['VaR_Spread'].iloc[:in_sample_end]
    
    # Initialize variables for tracking best parameters
    best_sharpe = -np.inf
    best_lookback = 10
    best_threshold = 1.5
    
    # Grid search
    for lookback in lookback_values:
        for threshold in threshold_values:
            # Generate signals
            signals = generate_trading_signals(in_sample_var_spread, lookback, threshold)
            
            # Calculate strategy returns
            strategy_returns = calculate_strategy_returns(in_sample_returns, signals)
            
            # Calculate Sharpe ratio
            sharpe = calculate_performance_metrics(strategy_returns)['Sharpe Ratio']
            
            # Update if better
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_lookback = lookback
                best_threshold = threshold
    
    print(f"Optimal parameters: lookback={best_lookback}, threshold={best_threshold:.2f}")
    return best_lookback, best_threshold

def main():
    # Generate synthetic data
    print("Generating synthetic data...")
    price_data = generate_synthetic_data(n_days=2000)
    
    # Calculate returns
    returns = calculate_returns(price_data['Price'])
    print(f"Generated {len(returns)} days of return data")
    
    # Calculate VaR spreads
    window_size = 100  # Lookback window for VaR calculation
    var_df = calculate_var_spread_series(returns, window_size=window_size)
    
    # Split into training and testing periods
    test_start = len(returns) // 2
    
    # Optimize parameters on training data
    lookback, threshold_factor = optimize_parameters(returns, var_df, test_start)
    
    # Generate trading signals using optimized parameters
    signals = generate_trading_signals(var_df['VaR_Spread'], lookback, threshold_factor)
    
    # Get out-of-sample data
    training_returns = returns.iloc[:test_start]
    testing_returns = returns.iloc[test_start:]
    testing_signals = signals.iloc[test_start:]
    
    # Calculate strategy returns
    strategy_returns = calculate_strategy_returns(testing_returns, testing_signals)
    
    # Calculate traditional protection strategies
    pp_returns = calculate_protective_put(testing_returns)
    drp_returns = calculate_delta_replicated_put(testing_returns)
    
    # Calculate performance metrics
    print("\nCalculating performance metrics...")
    benchmark_metrics = calculate_performance_metrics(testing_returns)
    strategy_metrics = calculate_performance_metrics(strategy_returns)
    pp_metrics = calculate_performance_metrics(pp_returns)
    drp_metrics = calculate_performance_metrics(drp_returns)
    
    # Print results
    print("\nPerformance Metrics (Out-of-Sample):")
    print("\nBenchmark:")
    for metric, value in benchmark_metrics.items():
        if 'Ratio' in metric:
            print(f"  {metric}: {value:.2f}")
        else:
            print(f"  {metric}: {value:.2%}")
    
    print("\nTail-Risk Protection Strategy:")
    for metric, value in strategy_metrics.items():
        if 'Ratio' in metric:
            print(f"  {metric}: {value:.2f}")
        else:
            print(f"  {metric}: {value:.2%}")
    
    print("\nProtective Put Strategy:")
    for metric, value in pp_metrics.items():
        if 'Ratio' in metric:
            print(f"  {metric}: {value:.2f}")
        else:
            print(f"  {metric}: {value:.2%}")
    
    print("\nDelta-Replicated Put Strategy:")
    for metric, value in drp_metrics.items():
        if 'Ratio' in metric:
            print(f"  {metric}: {value:.2f}")
        else:
            print(f"  {metric}: {value:.2%}")
    
    # Calculate fraction of days invested
    investment_fraction = testing_signals.mean()
    print(f"\nFraction of days invested: {investment_fraction:.2%}")
    
    # Plot results
    print("\nPlotting results...")
    plot_results(testing_returns, strategy_returns, pp_returns, drp_returns, testing_signals)
    
    print("Analysis complete.")

if __name__ == "__main__":
    main()