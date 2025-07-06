import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from arch import arch_model
from sklearn.preprocessing import MinMaxScaler
from scipy import stats
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# Function to simulate ETF price data
def simulate_etf_data(days=1000, initial_price=3.0, volatility=0.015, trend=0.0001):
    """
    Simulate ETF price data with realistic volatility and trend
    
    Parameters:
    -----------
    days: int
        Number of trading days to simulate
    initial_price: float
        Starting price of the ETF
    volatility: float
        Daily volatility (standard deviation of returns)
    trend: float
        Daily drift term
        
    Returns:
    --------
    DataFrame with simulated price data
    """
    # Create date range (business days)
    dates = pd.date_range(start='2015-01-01', periods=days, freq='B')
    
    # Create simulated returns
    # Add some volatility clustering to simulate real market behavior
    vol = np.zeros(days)
    vol[0] = volatility
    returns = np.zeros(days)
    
    for i in range(1, days):
        # GARCH-like process for volatility
        if i > 20 and i % 100 < 20:  # Create periodic volatility spikes
            vol[i] = volatility * 2.5 + 0.85 * vol[i-1] + 0.1 * np.random.normal(0, 0.005)
        else:
            vol[i] = 0.1 * volatility + 0.85 * vol[i-1] + 0.1 * np.random.normal(0, 0.002)
        
        # Generate return with current volatility
        returns[i] = trend + np.random.normal(0, vol[i])
    
    # Convert returns to prices
    prices = initial_price * np.exp(np.cumsum(returns))
    
    # Create DataFrame
    df = pd.DataFrame({
        'Date': dates,
        'Close': prices,
        'True_Volatility': vol
    })
    df.set_index('Date', inplace=True)
    
    # Add additional columns
    df['Open'] = df['Close'].shift(1)
    df['Open'].iloc[0] = initial_price
    df['High'] = df['Close'] * (1 + np.abs(np.random.normal(0, 0.3)) * vol)
    df['Low'] = df['Close'] * (1 - np.abs(np.random.normal(0, 0.3)) * vol)
    df['Volume'] = np.random.lognormal(15, 1, size=len(df))
    
    # Calculate returns
    df['Returns'] = df['Close'].pct_change() * 100
    df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1)) * 100
    
    return df

# Function to simulate option prices based on Black-Scholes model
def simulate_option_prices(etf_data, strike_ratio=1.0, dte=30, risk_free_rate=0.03, 
                           iv_premium=0.05, iv_skew=0.02):
    """
    Simulate option prices using Black-Scholes model
    
    Parameters:
    -----------
    etf_data: DataFrame
        ETF price data
    strike_ratio: float
        Ratio of strike price to current price (1.0 = ATM)
    dte: int
        Days to expiration
    risk_free_rate: float
        Annual risk-free rate
    iv_premium: float
        Additional implied volatility premium
    iv_skew: float
        Amount of volatility skew (higher for puts)
        
    Returns:
    --------
    DataFrame with option prices
    """
    # Black-Scholes formula for option pricing
    def bs_call(S, K, T, r, sigma):
        d1 = (np.log(S/K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        call_price = S * stats.norm.cdf(d1) - K * np.exp(-r * T) * stats.norm.cdf(d2)
        return call_price
    
    def bs_put(S, K, T, r, sigma):
        d1 = (np.log(S/K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        put_price = K * np.exp(-r * T) * stats.norm.cdf(-d2) - S * stats.norm.cdf(-d1)
        return put_price
    
    df = etf_data.copy()
    
    # Calculate 20-day historical volatility
    df['Hist_Vol'] = df['Log_Returns'].rolling(window=20).std() * np.sqrt(252) / 100
    
    # Simulate implied volatility (usually higher than historical volatility)
    # Add a premium and some noise
    df['IV'] = df['Hist_Vol'] + iv_premium + np.random.normal(0, 0.02, size=len(df))
    
    # Time to expiration in years
    T = dte / 365
    
    # Strike price based on current price
    df['Strike'] = df['Close'] * strike_ratio
    
    # Calculate call and put prices
    df['Call_IV'] = df['IV'] * 0.95  # Slight IV skew (lower for calls)
    df['Put_IV'] = df['IV'] * (1 + iv_skew)  # IV skew (higher for puts)
    
    df['Call_Price'] = df.apply(
        lambda x: bs_call(x['Close'], x['Strike'], T, risk_free_rate, x['Call_IV']), axis=1)
    df['Put_Price'] = df.apply(
        lambda x: bs_put(x['Close'], x['Strike'], T, risk_free_rate, x['Put_IV']), axis=1)
    
    # Calculate option greeks (simplified)
    df['Call_Delta'] = stats.norm.cdf((np.log(df['Close']/df['Strike']) + 
                                      (risk_free_rate + 0.5 * df['Call_IV']**2) * T) / 
                                     (df['Call_IV'] * np.sqrt(T)))
    df['Put_Delta'] = df['Call_Delta'] - 1
    
    return df

# Function to calculate realized PnL for a short straddle position
def calculate_short_straddle_pnl(options_data, position_size=1.0):
    """
    Calculate daily PnL for a short straddle position
    
    Parameters:
    -----------
    options_data: DataFrame
        DataFrame with option prices
    position_size: float
        Number of contracts to trade
        
    Returns:
    --------
    DataFrame with PnL data
    """
    df = options_data.copy()
    
    # Calculate daily option price changes
    df['Call_PnL'] = -1 * (df['Call_Price'] - df['Call_Price'].shift(1)) * position_size
    df['Put_PnL'] = -1 * (df['Put_Price'] - df['Put_Price'].shift(1)) * position_size
    
    # Total PnL (short straddle)
    df['Total_PnL'] = df['Call_PnL'] + df['Put_PnL']
    
    # Calculate cumulative PnL
    df['Cum_PnL'] = df['Total_PnL'].cumsum()
    
    return df

# Function to run GARCH model and forecast volatility
def forecast_volatility_garch(returns, p=1, q=1):
    """
    Forecast volatility using GARCH model
    
    Parameters:
    -----------
    returns: Series
        Series of returns
    p: int
        GARCH lag order
    q: int
        ARCH lag order
        
    Returns:
    --------
    Series with forecasted volatility
    """
    # Remove NaN values
    returns = returns.dropna()
    
    # Fit GARCH model
    model = arch_model(returns, vol='Garch', p=p, q=q)
    results = model.fit(disp='off')
    
    # Extract conditional volatility
    forecasted_vol = results.conditional_volatility
    
    return forecasted_vol

# Function to implement the basic short volatility strategy
def basic_short_volatility_strategy(options_data, initial_capital=100000):
    """
    Implement a basic short volatility strategy
    
    Parameters:
    -----------
    options_data: DataFrame
        DataFrame with option prices
    initial_capital: float
        Initial capital
        
    Returns:
    --------
    DataFrame with strategy results
    """
    df = options_data.copy()
    
    # Calculate position size (90% of portfolio as described in the paper)
    df['Position_Size'] = (initial_capital * 0.9) / (df['Call_Price'] + df['Put_Price'])
    
    # Calculate PnL
    df['Call_PnL'] = -1 * (df['Call_Price'] - df['Call_Price'].shift(1)) * df['Position_Size']
    df['Put_PnL'] = -1 * (df['Put_Price'] - df['Put_Price'].shift(1)) * df['Position_Size']
    
    # Account for transaction costs (2 RMB per contract as in the paper)
    transaction_cost = 2  # RMB per contract
    df['Transaction_Cost'] = (df['Position_Size'] * 2) * transaction_cost  # 2 contracts (call + put)
    
    # Total daily PnL
    df['Daily_PnL'] = df['Call_PnL'] + df['Put_PnL'] - df['Transaction_Cost']
    
    # Calculate portfolio value
    df['Portfolio_Value'] = initial_capital + df['Daily_PnL'].cumsum()
    
    return df

# Function to implement the improved volatility-based strategy
def improved_volatility_strategy(options_data, garch_vol, initial_capital=100000):
    """
    Implement improved volatility-based strategy using GARCH forecasts
    
    Parameters:
    -----------
    options_data: DataFrame
        DataFrame with option prices
    garch_vol: Series
        GARCH forecasted volatility
    initial_capital: float
        Initial capital
        
    Returns:
    --------
    DataFrame with strategy results
    """
    df = options_data.copy()
    
    # Merge GARCH volatility with options data
    df['GARCH_Vol'] = garch_vol
    
    # Z-score normalization of forecasted volatility (as mentioned in the paper)
    df['Vol_Z_Score'] = (df['GARCH_Vol'] - df['GARCH_Vol'].mean()) / df['GARCH_Vol'].std()
    
    # Map to [-1, 1] range (position sizing factor)
    df['Position_Factor'] = np.clip(-df['Vol_Z_Score'], -1, 1)
    
    # Position adjustment logic (as described in the paper)
    # When predicted volatility is high (positive z-score), reduce position size
    # When predicted volatility is low (negative z-score), increase position size
    
    # Add smoothing logic (from the paper's optimization section)
    # If volatility has been negative for 3+ days, reduce position
    for i in range(3, len(df)):
        if all(df['Vol_Z_Score'].iloc[i-3:i] > 0):
            df.loc[df.index[i], 'Position_Factor'] = df['Position_Factor'].iloc[i] / np.sqrt(2)
        elif all(df['Vol_Z_Score'].iloc[i-3:i] < 0):
            df.loc[df.index[i], 'Position_Factor'] = df['Position_Factor'].iloc[i] / 2
    
    # Base position size (90% of portfolio)
    base_position = (initial_capital * 0.9) / (df['Call_Price'] + df['Put_Price'])
    
    # Adjust position size based on volatility forecast
    # Map from [-1,1] to [0,1] range for position sizing
    position_scale = (df['Position_Factor'] + 1) / 2
    df['Position_Size'] = base_position * position_scale
    
    # Calculate PnL
    df['Call_PnL'] = -1 * (df['Call_Price'] - df['Call_Price'].shift(1)) * df['Position_Size']
    df['Put_PnL'] = -1 * (df['Put_Price'] - df['Put_Price'].shift(1)) * df['Position_Size']
    
    # Account for transaction costs
    transaction_cost = 2  # RMB per contract
    df['Transaction_Cost'] = (df['Position_Size'] * 2) * transaction_cost
    
    # Total daily PnL
    df['Daily_PnL'] = df['Call_PnL'] + df['Put_PnL'] - df['Transaction_Cost']
    
    # Calculate portfolio value
    df['Portfolio_Value'] = initial_capital + df['Daily_PnL'].cumsum()
    
    return df

# Function to evaluate strategy performance
def evaluate_strategy(strategy_results, benchmark_data, risk_free_rate=0.03):
    """
    Evaluate strategy performance
    
    Parameters:
    -----------
    strategy_results: DataFrame
        DataFrame with strategy results
    benchmark_data: DataFrame
        DataFrame with benchmark data
    risk_free_rate: float
        Annual risk-free rate
        
    Returns:
    --------
    Dict with performance metrics
    """
    # Calculate daily returns
    strategy_returns = strategy_results['Portfolio_Value'].pct_change().dropna()
    benchmark_returns = benchmark_data['Close'].pct_change().dropna()
    
    # Align dates
    common_dates = strategy_returns.index.intersection(benchmark_returns.index)
    strategy_returns = strategy_returns.loc[common_dates]
    benchmark_returns = benchmark_returns.loc[common_dates]
    
    # Calculate performance metrics
    total_return = (strategy_results['Portfolio_Value'].iloc[-1] / 
                   strategy_results['Portfolio_Value'].iloc[0]) - 1
    annual_return = ((1 + total_return) ** (252 / len(strategy_returns))) - 1
    daily_rf = (1 + risk_free_rate) ** (1/252) - 1
    excess_returns = strategy_returns - daily_rf
    sharpe_ratio = np.sqrt(252) * (strategy_returns.mean() - daily_rf) / strategy_returns.std()
    max_drawdown = (strategy_results['Portfolio_Value'] / 
                   strategy_results['Portfolio_Value'].cummax() - 1).min()
    
    # Calculate beta and alpha
    cov_matrix = np.cov(strategy_returns, benchmark_returns)
    beta = cov_matrix[0, 1] / cov_matrix[1, 1]
    alpha = strategy_returns.mean() * 252 - beta * benchmark_returns.mean() * 252
    
    # Results dictionary
    results = {
        'Total Return': total_return,
        'Annual Return': annual_return,
        'Sharpe Ratio': sharpe_ratio,
        'Max Drawdown': max_drawdown,
        'Beta': beta,
        'Alpha': alpha
    }
    
    return results

# Main function to run the simulation and test strategies
def main():
    # 1. Simulate ETF data
    print("Simulating ETF data...")
    etf_data = simulate_etf_data(days=1000, initial_price=3.0, volatility=0.015)
    
    # Split data into pre-2018 and post-2018 periods to simulate the market shift described in the paper
    cutoff_date = '2018-01-01'
    pre_2018 = etf_data[etf_data.index < cutoff_date]
    post_2018 = etf_data[etf_data.index >= cutoff_date]
    
    # Increase volatility for post-2018 period to match the paper's findings
    post_2018['True_Volatility'] = post_2018['True_Volatility'] * 1.5
    post_2018['Returns'] = post_2018['Returns'] * 1.2
    post_2018['Log_Returns'] = post_2018['Log_Returns'] * 1.2
    
    # Recombine data
    etf_data = pd.concat([pre_2018, post_2018])
    
    # 2. Simulate option prices
    print("Simulating option prices...")
    options_data = simulate_option_prices(etf_data)
    
    # 3. Test basic short volatility strategy
    print("Testing basic short volatility strategy...")
    basic_strategy = basic_short_volatility_strategy(options_data)
    
    # 4. Forecast volatility using GARCH
    print("Forecasting volatility with GARCH...")
    garch_vol = forecast_volatility_garch(etf_data['Log_Returns'])
    
    # 5. Test improved volatility strategy
    print("Testing improved volatility strategy...")
    improved_strategy = improved_volatility_strategy(options_data, garch_vol)
    
    # 6. Evaluate and compare strategies
    print("Evaluating strategies...")
    basic_performance = evaluate_strategy(basic_strategy, etf_data)
    improved_performance = evaluate_strategy(improved_strategy, etf_data)
    
    # Print performance metrics
    print("\nBasic Short Volatility Strategy Performance:")
    for metric, value in basic_performance.items():
        print(f"{metric}: {value:.4f}")
    
    print("\nImproved Volatility Strategy Performance:")
    for metric, value in improved_performance.items():
        print(f"{metric}: {value:.4f}")
    
    # 7. Analyze pre-2018 vs post-2018 performance
    pre_2018_basic = basic_strategy[basic_strategy.index < cutoff_date]
    pre_2018_improved = improved_strategy[improved_strategy.index < cutoff_date]
    post_2018_basic = basic_strategy[basic_strategy.index >= cutoff_date]
    post_2018_improved = improved_strategy[improved_strategy.index >= cutoff_date]
    
    # Calculate returns
    pre_2018_basic_return = (pre_2018_basic['Portfolio_Value'].iloc[-1] / 
                            pre_2018_basic['Portfolio_Value'].iloc[0]) - 1
    pre_2018_improved_return = (pre_2018_improved['Portfolio_Value'].iloc[-1] / 
                               pre_2018_improved['Portfolio_Value'].iloc[0]) - 1
    post_2018_basic_return = (post_2018_basic['Portfolio_Value'].iloc[-1] / 
                             post_2018_basic['Portfolio_Value'].iloc[0]) - 1
    post_2018_improved_return = (post_2018_improved['Portfolio_Value'].iloc[-1] / 
                                post_2018_improved['Portfolio_Value'].iloc[0]) - 1
    
    print("\nPre-2018 Performance:")
    print(f"Basic Strategy: {pre_2018_basic_return:.4f}")
    print(f"Improved Strategy: {pre_2018_improved_return:.4f}")
    
    print("\nPost-2018 Performance:")
    print(f"Basic Strategy: {post_2018_basic_return:.4f}")
    print(f"Improved Strategy: {post_2018_improved_return:.4f}")
    
    # 8. Visualize results
    plt.figure(figsize=(12, 8))
    
    # Portfolio values
    plt.subplot(2, 1, 1)
    plt.plot(basic_strategy['Portfolio_Value'], label='Basic Short Volatility Strategy')
    plt.plot(improved_strategy['Portfolio_Value'], label='Improved Volatility Strategy')
    plt.axvline(x=pd.to_datetime(cutoff_date), color='r', linestyle='--', label='2018 Cutoff')
    plt.title('Strategy Performance Comparison')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value')
    plt.legend()
    plt.grid(True)
    
    # ETF price and volatility
    plt.subplot(2, 1, 2)
    plt.plot(etf_data['Close'], label='ETF Price')
    plt.plot(garch_vol * 100, label='GARCH Volatility (scaled)', alpha=0.7)
    plt.axvline(x=pd.to_datetime(cutoff_date), color='r', linestyle='--', label='2018 Cutoff')
    plt.title('ETF Price and Volatility')
    plt.xlabel('Date')
    plt.ylabel('Price / Volatility')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Additional analysis: Drawdown periods
    plt.figure(figsize=(12, 6))
    
    # Calculate drawdowns
    basic_drawdown = basic_strategy['Portfolio_Value'] / basic_strategy['Portfolio_Value'].cummax() - 1
    improved_drawdown = improved_strategy['Portfolio_Value'] / improved_strategy['Portfolio_Value'].cummax() - 1
    
    plt.plot(basic_drawdown, label='Basic Strategy Drawdown')
    plt.plot(improved_drawdown, label='Improved Strategy Drawdown')
    plt.axvline(x=pd.to_datetime(cutoff_date), color='r', linestyle='--', label='2018 Cutoff')
    plt.title('Strategy Drawdowns')
    plt.xlabel('Date')
    plt.ylabel('Drawdown')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Visualize position sizing comparison
    plt.figure(figsize=(12, 6))
    
    plt.plot(basic_strategy['Position_Size'], label='Basic Strategy Position Size')
    plt.plot(improved_strategy['Position_Size'], label='Improved Strategy Position Size')
    plt.axvline(x=pd.to_datetime(cutoff_date), color='r', linestyle='--', label='2018 Cutoff')
    plt.title('Position Sizing Comparison')
    plt.xlabel('Date')
    plt.ylabel('Position Size (Contracts)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

# Run the simulation
if __name__ == "__main__":
    main()