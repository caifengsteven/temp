import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

# Set seed for reproducibility
np.random.seed(42)

# Simulate financial time series data
def generate_simulated_data(n_assets=2, n_days=500, mu=0.0005, sigma=0.015, 
                            trend_strength=0.4, seasonality_strength=0.05):
    """
    Generate simulated stock data for multiple assets
    
    Parameters:
    -----------
    n_assets: int
        Number of assets to simulate
    n_days: int
        Number of trading days
    mu: float
        Mean daily return
    sigma: float
        Volatility of daily returns
    trend_strength: float
        Strength of trend component
    seasonality_strength: float
        Strength of seasonality component
    
    Returns:
    --------
    df: DataFrame
        DataFrame containing simulated price and return data
    """
    # Create date index
    dates = pd.date_range(start='2010-01-01', periods=n_days, freq='B')
    
    # Dictionary to store asset data
    data = {}
    
    # Generate data for each asset
    for asset_id in range(n_assets):
        # Generate daily returns with random walk
        daily_returns = np.random.normal(mu, sigma, n_days)
        
        # Add trend component
        trend_direction = np.random.choice([-1, 1])
        trend = np.linspace(0, trend_strength, n_days) * trend_direction * np.random.uniform(0.5, 1.5)
        
        # Add seasonality component
        seasonality = np.sin(np.linspace(0, 4 * np.pi, n_days)) * seasonality_strength * np.random.uniform(0.8, 1.2)
        
        # Combine components
        modified_returns = daily_returns + trend + seasonality
        
        # Generate price series starting at random price between 20 and 200
        initial_price = np.random.uniform(20, 200)
        prices = initial_price * np.cumprod(1 + modified_returns)
        
        # Store in dictionary
        data[f'price_{asset_id}'] = prices
        data[f'return_{asset_id}'] = modified_returns
        
        # Generate volatility using a simple GARCH-like process
        vol = np.zeros(n_days)
        vol[0] = sigma
        for i in range(1, n_days):
            vol[i] = 0.2 * vol[i-1] + 0.7 * abs(modified_returns[i-1]) + 0.1 * sigma
        
        data[f'vol_{asset_id}'] = vol
    
    # Create DataFrame
    df = pd.DataFrame(data, index=dates)
    
    return df

# Implement different rotation models

# 1. Naive model (equation 7 in the paper)
def naive_model(y, window_size=104):
    """
    Naive model using the mean of relative returns
    
    Parameters:
    -----------
    y: array-like
        Relative returns between two assets
    window_size: int
        Size of the rolling window
    
    Returns:
    --------
    pred: array-like
        Predictions for the sign of relative returns
    """
    pred = np.zeros(len(y))
    for t in range(window_size, len(y)):
        # Mean of relative returns in the window
        pred[t] = np.mean(y[t-window_size:t])
    
    return pred

# 2. ARIMA model (equation 8 in the paper)
def arima_model(y, window_size=104, order=(1,0,1)):
    """
    ARIMA model for relative returns
    
    Parameters:
    -----------
    y: array-like
        Relative returns between two assets
    window_size: int
        Size of the rolling window
    order: tuple
        Order of the ARIMA model (p,d,q)
    
    Returns:
    --------
    pred: array-like
        Predictions for the sign of relative returns
    """
    pred = np.zeros(len(y))
    for t in range(window_size, len(y)):
        # Fit ARIMA model on the window
        try:
            model = ARIMA(y[t-window_size:t], order=order)
            model_fit = model.fit()
            # Forecast one step ahead
            pred[t] = model_fit.forecast(steps=1)[0]
        except:
            # In case of convergence issues, use last value
            pred[t] = y[t-1]
    
    return pred

# 3. Difference of volatility model (first part of equation 10)
def volatility_diff_model(v1, v2, window_size=104, p=4):
    """
    Autoregressive model for the difference of log-volatilities
    
    Parameters:
    -----------
    v1, v2: array-like
        Volatility series for the two assets
    window_size: int
        Size of the rolling window
    p: int
        Order of the AR model
    
    Returns:
    --------
    pred: array-like
        Predictions for the sign of relative volatility
    """
    # Calculate log volatility difference
    log_vol_diff = np.log(v1) - np.log(v2)
    
    pred = np.zeros(len(log_vol_diff))
    for t in range(window_size, len(log_vol_diff)):
        # Fit AR model on the window
        try:
            model = ARIMA(log_vol_diff[t-window_size:t], order=(p,0,0))
            model_fit = model.fit()
            # Forecast one step ahead
            pred[t] = model_fit.forecast(steps=1)[0]
        except:
            # In case of convergence issues, use last value
            pred[t] = log_vol_diff[t-1]
    
    return pred

# 4. Piecewise linear model (equation 9 in the paper)
def piecewise_linear_model(y, v, window_size=104):
    """
    Piecewise linear model for relative returns
    
    Parameters:
    -----------
    y: array-like
        Relative returns between two assets
    v: array-like
        Relative volatility between two assets
    window_size: int
        Size of the rolling window
    
    Returns:
    --------
    pred: array-like
        Predictions for the sign of relative returns
    """
    pred = np.zeros(len(y))
    
    for t in range(window_size, len(y)):
        # Extract window
        y_window = y[t-window_size:t]
        v_window = v[t-window_size:t]
        
        # Create dummy variables for asymmetric response
        I_y = (y_window[:-1] < 0).astype(int)
        I_v = (v_window[:-1] < 0).astype(int)
        
        # Create design matrix
        X = np.column_stack((
            np.ones(len(y_window)-1),  # Intercept
            y_window[:-1],             # Lagged relative return
            v_window[:-1],             # Lagged relative volatility
            I_y,                       # Dummy for negative return
            I_v,                       # Dummy for negative volatility
            y_window[:-1] * I_y,       # Interaction term
            v_window[:-1] * I_v,       # Interaction term
            y_window[:-1] * I_v,       # Cross-interaction term
            v_window[:-1] * I_y        # Cross-interaction term
        ))
        
        # Dependent variable
        Y = y_window[1:]
        
        # Linear regression using numpy
        try:
            beta = np.linalg.lstsq(X, Y, rcond=None)[0]
            
            # Create prediction values
            x_pred = np.array([
                1,                          # Intercept
                y[t-1],                     # Last relative return
                v[t-1],                     # Last relative volatility
                int(y[t-1] < 0),            # Dummy for negative return
                int(v[t-1] < 0),            # Dummy for negative volatility
                y[t-1] * int(y[t-1] < 0),   # Interaction term
                v[t-1] * int(v[t-1] < 0),   # Interaction term
                y[t-1] * int(v[t-1] < 0),   # Cross-interaction term
                v[t-1] * int(y[t-1] < 0)    # Cross-interaction term
            ])
            
            # Forecast
            pred[t] = np.dot(x_pred, beta)
        except:
            # In case of issues, use simple prediction
            pred[t] = y[t-1]
    
    return pred

# Implement rotation trading strategy
def rotation_trading_strategy(returns_1, returns_2, model_predictions, initial_capital=1.0):
    """
    Implement rotation trading strategy based on model predictions
    
    Parameters:
    -----------
    returns_1, returns_2: array-like
        Returns for the two assets
    model_predictions: array-like
        Predictions for the sign of relative returns/volatility
    initial_capital: float
        Initial investment amount
    
    Returns:
    --------
    strategy_returns: array-like
        Returns from the rotation strategy
    positions: array-like
        Position indicators (1 for asset 1, 0 for asset 2)
    wealth: array-like
        Wealth evolution
    """
    n = len(returns_1)
    strategy_returns = np.zeros(n)
    positions = np.zeros(n)
    wealth = np.zeros(n+1)
    wealth[0] = initial_capital
    
    # For each time point
    for t in range(1, n):
        # If prediction is positive, invest in asset 1, else invest in asset 2
        if model_predictions[t] > 0:
            positions[t] = 1
            strategy_returns[t] = returns_1[t]
        else:
            positions[t] = 0
            strategy_returns[t] = returns_2[t]
        
        # Update wealth
        wealth[t+1] = wealth[t] * (1 + strategy_returns[t])
    
    return strategy_returns, positions, wealth

# Calculate performance metrics
def calculate_performance_metrics(returns):
    """
    Calculate performance metrics for a return series
    
    Parameters:
    -----------
    returns: array-like
        Return series
    
    Returns:
    --------
    metrics: dict
        Dictionary of performance metrics
    """
    metrics = {}
    
    # Calculate total return
    metrics['total_return'] = (1 + returns).prod() - 1
    
    # Calculate terminal wealth
    metrics['terminal_wealth'] = (1 + returns).prod()
    
    # Calculate average return
    metrics['avg_return'] = returns.mean()
    
    # Calculate standard deviation
    metrics['std_dev'] = returns.std()
    
    # Calculate Sharpe ratio (assuming risk-free rate of 0)
    metrics['sharpe_ratio'] = metrics['avg_return'] / metrics['std_dev']
    
    # Calculate maximum drawdown
    cumulative_returns = (1 + returns).cumprod()
    running_max = np.maximum.accumulate(cumulative_returns)
    drawdown = (cumulative_returns - running_max) / running_max
    metrics['max_drawdown'] = drawdown.min()
    
    # Calculate minimum realized return
    metrics['min_return'] = returns.min()
    
    return metrics

# Calculate sign success ratio
def calculate_sign_success_ratio(actual_signs, predicted_signs):
    """
    Calculate percentage of correct sign predictions
    
    Parameters:
    -----------
    actual_signs: array-like
        Actual signs of relative returns
    predicted_signs: array-like
        Predicted signs of relative returns
    
    Returns:
    --------
    success_ratio: float
        Percentage of correct sign predictions
    """
    success_ratio = accuracy_score(actual_signs, predicted_signs)
    return success_ratio

# Main function to run the experiment
def run_experiment(n_assets=2, n_days=500, window_size=104):
    """
    Run the experiment for different rotation models
    
    Parameters:
    -----------
    n_assets: int
        Number of assets to simulate
    n_days: int
        Number of trading days
    window_size: int
        Size of the rolling window
    
    Returns:
    --------
    results: dict
        Dictionary of performance results for different models
    """
    # Generate simulated data
    print("Generating simulated financial data...")
    df = generate_simulated_data(n_assets=n_assets, n_days=n_days)
    
    # Extract returns and volatility for the two assets
    returns_1 = df['return_0'].values
    returns_2 = df['return_1'].values
    vol_1 = df['vol_0'].values
    vol_2 = df['vol_1'].values
    
    # Calculate relative returns and volatility
    rel_returns = returns_1 - returns_2
    rel_vol = vol_1 - vol_2
    
    # Implement different models
    print("Implementing rotation models...")
    naive_predictions = naive_model(rel_returns, window_size)
    arima_predictions = arima_model(rel_returns, window_size)
    vol_diff_predictions = volatility_diff_model(vol_1, vol_2, window_size)
    piecewise_predictions = piecewise_linear_model(rel_returns, rel_vol, window_size)
    
    # Create equally weighted portfolio returns
    equal_weights_returns = (returns_1 + returns_2) / 2
    
    # Apply rotation trading strategy
    print("Applying trading strategies...")
    naive_strat_returns, naive_positions, naive_wealth = rotation_trading_strategy(returns_1, returns_2, naive_predictions)
    arima_strat_returns, arima_positions, arima_wealth = rotation_trading_strategy(returns_1, returns_2, arima_predictions)
    vol_diff_strat_returns, vol_diff_positions, vol_diff_wealth = rotation_trading_strategy(returns_1, returns_2, vol_diff_predictions)
    piecewise_strat_returns, piecewise_positions, piecewise_wealth = rotation_trading_strategy(returns_1, returns_2, piecewise_predictions)
    
    # Calculate performance metrics
    print("Calculating performance metrics...")
    naive_metrics = calculate_performance_metrics(naive_strat_returns[window_size:])
    arima_metrics = calculate_performance_metrics(arima_strat_returns[window_size:])
    vol_diff_metrics = calculate_performance_metrics(vol_diff_strat_returns[window_size:])
    piecewise_metrics = calculate_performance_metrics(piecewise_strat_returns[window_size:])
    asset1_metrics = calculate_performance_metrics(returns_1[window_size:])
    asset2_metrics = calculate_performance_metrics(returns_2[window_size:])
    equal_metrics = calculate_performance_metrics(equal_weights_returns[window_size:])
    
    # Calculate sign success ratios
    actual_signs = (rel_returns > 0).astype(int)
    naive_signs = (naive_predictions > 0).astype(int)
    arima_signs = (arima_predictions > 0).astype(int)
    vol_diff_signs = (vol_diff_predictions > 0).astype(int)
    piecewise_signs = (piecewise_predictions > 0).astype(int)
    
    naive_sign_success = calculate_sign_success_ratio(actual_signs[window_size:], naive_signs[window_size:])
    arima_sign_success = calculate_sign_success_ratio(actual_signs[window_size:], arima_signs[window_size:])
    vol_diff_sign_success = calculate_sign_success_ratio(actual_signs[window_size:], vol_diff_signs[window_size:])
    piecewise_sign_success = calculate_sign_success_ratio(actual_signs[window_size:], piecewise_signs[window_size:])
    
    # Collect results
    results = {
        'Naive': {
            'metrics': naive_metrics,
            'wealth': naive_wealth,
            'positions': naive_positions,
            'sign_success': naive_sign_success
        },
        'ARIMA': {
            'metrics': arima_metrics,
            'wealth': arima_wealth,
            'positions': arima_positions,
            'sign_success': arima_sign_success
        },
        'Vol_Diff': {
            'metrics': vol_diff_metrics,
            'wealth': vol_diff_wealth,
            'positions': vol_diff_positions,
            'sign_success': vol_diff_sign_success
        },
        'Piecewise_Linear': {
            'metrics': piecewise_metrics,
            'wealth': piecewise_wealth,
            'positions': piecewise_positions,
            'sign_success': piecewise_sign_success
        },
        'Asset_1_Buy_Hold': {
            'metrics': asset1_metrics
        },
        'Asset_2_Buy_Hold': {
            'metrics': asset2_metrics
        },
        'Equal_Weights': {
            'metrics': equal_metrics
        }
    }
    
    # Visualize results
    print("Visualizing results...")
    visualize_results(results, df, window_size)
    
    return results

# Visualize results
def visualize_results(results, df, window_size):
    """
    Visualize the results of the experiment
    
    Parameters:
    -----------
    results: dict
        Dictionary of performance results for different models
    df: DataFrame
        DataFrame containing the simulated data
    window_size: int
        Size of the rolling window
    """
    # Plot wealth evolution
    plt.figure(figsize=(12, 6))
    plt.plot(results['Naive']['wealth'][window_size:], label='Naive')
    plt.plot(results['ARIMA']['wealth'][window_size:], label='ARIMA')
    plt.plot(results['Vol_Diff']['wealth'][window_size:], label='Volatility Difference')
    plt.plot(results['Piecewise_Linear']['wealth'][window_size:], label='Piecewise Linear')
    
    # Add buy-and-hold for comparison
    buy_hold_1 = np.cumprod(1 + df['return_0'].values[window_size:])
    buy_hold_2 = np.cumprod(1 + df['return_1'].values[window_size:])
    equal_weights = np.cumprod(1 + (df['return_0'].values[window_size:] + df['return_1'].values[window_size:]) / 2)
    
    plt.plot(buy_hold_1, label='Buy & Hold Asset 1')
    plt.plot(buy_hold_2, label='Buy & Hold Asset 2')
    plt.plot(equal_weights, label='Equal Weights')
    
    plt.title('Wealth Evolution')
    plt.legend()
    plt.grid(True)
    plt.savefig('wealth_evolution.png')
    plt.close()
    
    # Plot asset prices
    plt.figure(figsize=(12, 6))
    plt.plot(df['price_0'], label='Asset 1 Price')
    plt.plot(df['price_1'], label='Asset 2 Price')
    plt.title('Asset Prices')
    plt.legend()
    plt.grid(True)
    plt.savefig('asset_prices.png')
    plt.close()
    
    # Plot position changes for the piecewise linear model
    plt.figure(figsize=(12, 6))
    plt.plot(results['Piecewise_Linear']['positions'][window_size:], label='Piecewise Linear Positions')
    plt.title('Position Changes (Piecewise Linear Model)')
    plt.ylabel('Position (1 = Asset 1, 0 = Asset 2)')
    plt.axhline(y=0.5, color='r', linestyle='-', alpha=0.3)
    plt.legend()
    plt.grid(True)
    plt.savefig('positions_piecewise.png')
    plt.close()
    
    # Plot relative returns and volatility
    plt.figure(figsize=(12, 10))
    
    plt.subplot(2, 1, 1)
    plt.plot(df['return_0'] - df['return_1'], label='Relative Returns')
    plt.title('Relative Returns (Asset 1 - Asset 2)')
    plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.plot(df['vol_0'] - df['vol_1'], label='Relative Volatility')
    plt.title('Relative Volatility (Asset 1 - Asset 2)')
    plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('relative_metrics.png')
    plt.close()
    
    # Create bar chart of terminal wealth
    models = list(results.keys())
    terminal_wealth = [results[model]['metrics']['terminal_wealth'] if 'metrics' in results[model] else 0 for model in models]
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(models, terminal_wealth, color='skyblue')
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom')
    
    plt.title('Terminal Wealth Comparison')
    plt.ylabel('Terminal Wealth')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig('terminal_wealth.png')
    plt.close()
    
    # Create bar chart of Sharpe ratios
    sharpe_ratios = [results[model]['metrics']['sharpe_ratio'] if 'metrics' in results[model] else 0 for model in models]
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(models, sharpe_ratios, color='lightgreen')
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom')
    
    plt.title('Sharpe Ratio Comparison')
    plt.ylabel('Sharpe Ratio')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig('sharpe_ratios.png')
    plt.close()
    
    # Create bar chart of sign success ratios
    sign_success_models = ['Naive', 'ARIMA', 'Vol_Diff', 'Piecewise_Linear']
    sign_success_ratios = [results[model]['sign_success'] for model in sign_success_models]
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(sign_success_models, sign_success_ratios, color='salmon')
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom')
    
    plt.title('Sign Success Ratio Comparison')
    plt.ylabel('Sign Success Ratio')
    plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.7, label='Random Guess')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig('sign_success.png')
    plt.close()
    
    # Print summary table
    print("\n=== Performance Summary ===")
    headers = ["Model", "Terminal Wealth", "Avg Return", "Std Dev", "Sharpe Ratio", "Max Drawdown", "Sign Success"]
    
    print(f"{headers[0]:<20} {headers[1]:<15} {headers[2]:<15} {headers[3]:<15} {headers[4]:<15} {headers[5]:<15} {headers[6]:<15}")
    print("-" * 110)
    
    for model in models:
        metrics = results[model]['metrics']
        
        sign_success = results[model].get('sign_success', 'N/A')
        sign_success_str = f"{sign_success:.4f}" if isinstance(sign_success, float) else sign_success
        
        print(f"{model:<20} {metrics['terminal_wealth']:<15.4f} {metrics['avg_return']:<15.4f} {metrics['std_dev']:<15.4f} "
              f"{metrics['sharpe_ratio']:<15.4f} {metrics['max_drawdown']:<15.4f} {sign_success_str:<15}")

# Run the experiment
if __name__ == "__main__":
    results = run_experiment(n_assets=2, n_days=500, window_size=104)