import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from scipy.optimize import minimize
from sklearn.metrics import r2_score
from scipy.stats import norm
import datetime as dt
from tqdm import tqdm

# Set random seed for reproducibility
np.random.seed(42)

# Function to fetch real SPX and VIX data (for reference)
def fetch_real_data():
    try:
        # Try to fetch real data
        start_date = "1990-01-01"
        end_date = "2022-12-31"
        
        # Download SPX and VIX data
        spx = yf.download("^GSPC", start=start_date, end=end_date)
        vix = yf.download("^VIX", start=start_date, end=end_date)
        
        # Align dates
        merged_data = pd.DataFrame({
            'SPX': spx['Adj Close'],
            'VIX': vix['Adj Close']
        })
        
        # Remove NaN values
        merged_data = merged_data.dropna()
        
        return merged_data
    except:
        # If fetching fails, return None
        print("Failed to fetch real data. Using simulated data only.")
        return None

# Function to generate simulated data for SPX and VIX
def simulate_data(n_days=8000, use_real_data_properties=False, real_data=None):
    """
    Simulate SPX and VIX data based on the scale-invariant LSV model.
    
    Parameters:
    n_days (int): Number of days to simulate
    use_real_data_properties (bool): Whether to use real data properties for simulation
    real_data (DataFrame): Real data to extract properties from
    
    Returns:
    DataFrame: Simulated SPX and VIX data
    """
    # Set model parameters
    if use_real_data_properties and real_data is not None:
        # Extract parameters from real data
        log_returns = np.log(real_data['SPX'] / real_data['SPX'].shift(1)).dropna()
        annual_vol = log_returns.std() * np.sqrt(252)
        mean_vix = real_data['VIX'].mean()
        initial_spx = real_data['SPX'].iloc[0]
    else:
        # Use default parameters
        annual_vol = 0.2  # 20% annual volatility
        mean_vix = 20.0   # Mean VIX value
        initial_spx = 100  # Initial SPX value
    
    # Model parameters
    dt = 1/252  # Daily time step (252 trading days per year)
    mu = 0.05  # Annual drift (risk-free rate minus dividend yield)
    kappa = 5.0  # Mean reversion speed for Y process
    nu = 0.3  # Volatility of volatility
    rho = -0.7  # Correlation between SPX and Y processes
    
    # Initialize arrays
    times = np.arange(n_days) * dt
    spx = np.zeros(n_days)
    y = np.zeros(n_days)
    vix = np.zeros(n_days)
    
    # Set initial values
    spx[0] = initial_spx
    y[0] = 0  # Starting at mean level
    
    # Create power law kernel with parameters similar to the optimized ones in the paper
    kernel_length = 250  # Using 250 days as in the paper
    # Power law weights approximating the shape in Figure 7
    weights = np.zeros(kernel_length)
    for i in range(kernel_length):
        t = (i+1) / kernel_length
        weights[i] = np.exp(-3.5 * t)  # Exponential decay approximating power law
    
    # Normalize weights to sum to 1
    weights = weights / np.sum(weights)
    
    # Pre-fill the SPX array with initial values for the first kernel_length days
    # This ensures we have enough history for the moving average
    for i in range(1, kernel_length):
        spx[i] = spx[0]
    
    # Function for scale-invariant local volatility
    def calc_sigma_loc(t_idx, s, weights, spx_history):
        """Calculate the local volatility based on the ratio of spot to weighted average"""
        if t_idx < kernel_length:
            # Not enough history, use constant volatility
            ratio = 1.0
        else:
            # Calculate weighted average
            start_idx = t_idx - kernel_length
            history = spx_history[start_idx:t_idx]
            weighted_avg = np.sum(weights * history)
            ratio = s / weighted_avg
        
        # Quadratic function of the ratio as in the paper
        # Using coefficients inspired by Table 3
        a, b, c = 70, 3, -50
        vol_level = a + b*ratio + c*ratio**2
        
        # Normalize to get realistic volatility levels
        normalized_vol = vol_level / 100
        return max(0.05, min(normalized_vol, 1.0))  # Cap between 5% and 100%
    
    # Simulate paths
    for t in range(kernel_length, n_days-1):
        # Calculate local volatility
        sigma_loc = calc_sigma_loc(t, spx[t], weights, spx)
        
        # Generate correlated random shocks
        z1 = np.random.normal(0, 1)
        z2 = rho * z1 + np.sqrt(1 - rho**2) * np.random.normal(0, 1)
        
        # Update Y process (stochastic volatility component)
        y[t+1] = y[t] - kappa * y[t] * dt + nu * np.sqrt(dt) * z1
        
        # Update SPX process with local and stochastic volatility
        vol = sigma_loc * np.exp(y[t])
        spx[t+1] = spx[t] * np.exp((mu - 0.5 * vol**2) * dt + vol * np.sqrt(dt) * z2)
        
        # Calculate VIX (30-day expected volatility in percentage points)
        # VIX is approximated as the local volatility adjusted by the stochastic component
        vix[t] = sigma_loc * np.exp(y[t]) * 100  # Convert to percentage points
    
    # Create DataFrame
    dates = pd.date_range(start='2000-01-01', periods=n_days, freq='B')
    data = pd.DataFrame({
        'Date': dates,
        'SPX': spx,
        'VIX': vix,
        'Y': y
    })
    
    return data

# Function to calculate moving averages with different kernels
def calculate_moving_averages(spx_series, lag_days=[50, 100, 200, 250]):
    """
    Calculate moving averages with different lag periods.
    
    Parameters:
    spx_series (Series): SPX price series
    lag_days (list): List of lag periods
    
    Returns:
    DataFrame: Moving averages
    """
    result = pd.DataFrame(index=spx_series.index)
    
    for lag in lag_days:
        # Equal weights moving average
        ma_name = f"MA_{lag}"
        result[ma_name] = spx_series.rolling(window=lag).mean()
        
        # Calculate dimensionless ratio (current price / moving average)
        ratio_name = f"Ratio_{lag}"
        result[ratio_name] = spx_series / result[ma_name]
    
    return result

# Function to optimize weights for better fit
def optimize_weights(spx_series, vix_series, kernel_length=250):
    """
    Optimize weights to maximize the fit of VIX as a function of SPX/weighted_avg.
    
    Parameters:
    spx_series (Series): SPX price series
    vix_series (Series): VIX price series
    kernel_length (int): Length of the kernel
    
    Returns:
    array: Optimized weights
    float: R² score
    dict: Regression parameters (a, b, c)
    """
    # Initialize with equal weights
    initial_weights = np.ones(kernel_length) / kernel_length
    
    # Reparameterize weights to ensure they sum to 1 and are positive
    def weights_from_params(params):
        exp_params = np.exp(params)
        return exp_params / np.sum(exp_params)
    
    # Initial parameters for optimization
    initial_params = np.log(initial_weights + 1e-10)
    
    # Function to calculate the index
    def calculate_index(weights):
        # Create a matrix of lagged values
        lagged_values = np.zeros((len(spx_series) - kernel_length, kernel_length))
        for i in range(kernel_length):
            lagged_values[:, i] = spx_series.iloc[i:i+len(spx_series)-kernel_length].values
        
        # Calculate weighted average
        weighted_avg = np.dot(lagged_values, weights)
        
        # Calculate the index (current price / weighted average)
        current_prices = spx_series.iloc[kernel_length:].values
        index = current_prices / weighted_avg
        
        return index
    
    # Function to fit quadratic model
    def fit_quadratic(x, y):
        # Fit model: y = a + b*x + c*x^2
        X = np.column_stack([np.ones_like(x), x, x**2])
        coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        a, b, c = coeffs
        
        # Calculate predictions
        y_pred = a + b*x + c*x**2
        
        # Calculate R² score
        r2 = r2_score(y, y_pred)
        
        return r2, (a, b, c), y_pred
    
    # Objective function to minimize (negative R² score)
    def objective(params):
        weights = weights_from_params(params)
        index = calculate_index(weights)
        vix_values = vix_series.iloc[kernel_length:].values
        r2, _, _ = fit_quadratic(index, vix_values)
        return -r2
    
    # Optimize weights
    result = minimize(objective, initial_params, method='BFGS')
    
    # Get optimized weights
    optimized_weights = weights_from_params(result.x)
    
    # Calculate final R² score and regression parameters
    index = calculate_index(optimized_weights)
    vix_values = vix_series.iloc[kernel_length:].values
    r2, params, predictions = fit_quadratic(index, vix_values)
    
    return optimized_weights, r2, params, index, predictions

# Function to analyze the relationship between SPX and VIX
def analyze_spx_vix_relationship(data):
    """
    Analyze the relationship between SPX and VIX.
    
    Parameters:
    data (DataFrame): Data containing SPX and VIX
    
    Returns:
    None
    """
    # Plot SPX and VIX
    plt.figure(figsize=(12, 8))
    
    # Create two y-axes
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    
    # Plot SPX on left axis
    ax1.plot(data.index, data['SPX'], 'b-', label='SPX')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('SPX', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    
    # Plot VIX on right axis
    ax2.plot(data.index, data['VIX'], 'r-', label='VIX')
    ax2.set_ylabel('VIX', color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    
    # Add legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    plt.title('SPX and VIX Time Series')
    plt.tight_layout()
    plt.savefig('spx_vix_time_series.png')
    plt.close()
    
    # Plot VIX vs SPX
    plt.figure(figsize=(10, 6))
    plt.scatter(data['SPX'], data['VIX'], alpha=0.5)
    plt.xlabel('SPX')
    plt.ylabel('VIX')
    plt.title('VIX vs SPX (Absolute Levels)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('vix_vs_spx_absolute.png')
    plt.close()
    
    # Calculate moving averages and ratios
    ma_data = calculate_moving_averages(data['SPX'])
    
    # Merge with original data
    full_data = pd.concat([data, ma_data], axis=1)
    full_data = full_data.dropna()
    
    # Plot VIX vs Dimensionless Ratios for different lag periods
    lags = [50, 100, 200, 250]
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, lag in enumerate(lags):
        ratio_col = f"Ratio_{lag}"
        axes[i].scatter(full_data[ratio_col], full_data['VIX'], alpha=0.5)
        
        # Fit quadratic regression
        x = full_data[ratio_col].values
        y = full_data['VIX'].values
        
        X = np.column_stack([np.ones_like(x), x, x**2])
        coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        a, b, c = coeffs
        
        # Generate points for regression line
        x_range = np.linspace(x.min(), x.max(), 100)
        y_pred = a + b*x_range + c*x_range**2
        
        # Plot regression line
        axes[i].plot(x_range, y_pred, 'r-', linewidth=2)
        
        # Calculate and display R² score
        y_fit = a + b*x + c*x**2
        r2 = r2_score(y, y_fit)
        
        axes[i].set_xlabel(f'SPX / {lag}-day MA')
        axes[i].set_ylabel('VIX')
        axes[i].set_title(f'{lag}-day MA Ratio (R² = {r2:.4f})')
        axes[i].grid(True, alpha=0.3)
        axes[i].text(0.05, 0.95, f'y = {a:.2f} + {b:.2f}x + {c:.2f}x²', 
                    transform=axes[i].transAxes, fontsize=10,
                    verticalalignment='top')
    
    plt.tight_layout()
    plt.savefig('vix_vs_dimensionless_ratios.png')
    plt.close()
    
    return full_data

# Function to optimize weights and analyze the results
def analyze_optimized_weights(data, kernel_length=250):
    """
    Optimize weights and analyze the results.
    
    Parameters:
    data (DataFrame): Data containing SPX and VIX
    kernel_length (int): Length of the kernel
    
    Returns:
    tuple: Optimized weights, R² score, regression parameters
    """
    # Optimize weights
    opt_weights, r2, params, index, predictions = optimize_weights(
        data['SPX'], data['VIX'], kernel_length)
    
    # Print results
    print(f"Optimized weights R² score: {r2:.4f}")
    print(f"Regression parameters (a, b, c): {params}")
    
    # Plot optimized weights
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, kernel_length+1), opt_weights, 'b-')
    plt.xlabel('Lag (days)')
    plt.ylabel('Weight')
    plt.title(f'Optimized Weights (Kernel Length = {kernel_length})')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('optimized_weights.png')
    plt.close()
    
    # Plot fit using optimized weights
    plt.figure(figsize=(10, 6))
    plt.scatter(index, data['VIX'].iloc[kernel_length:], alpha=0.5)
    
    # Sort for smooth curve
    sorted_indices = np.argsort(index)
    plt.plot(index[sorted_indices], predictions[sorted_indices], 'r-', linewidth=2)
    
    plt.xlabel('SPX / Weighted Average')
    plt.ylabel('VIX')
    plt.title(f'VIX vs Optimized Index (R² = {r2:.4f})')
    plt.grid(True, alpha=0.3)
    a, b, c = params
    plt.text(0.05, 0.95, f'y = {a:.2f} + {b:.2f}x + {c:.2f}x²', 
             transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top')
    plt.tight_layout()
    plt.savefig('vix_vs_optimized_index.png')
    plt.close()
    
    return opt_weights, r2, params

# Function to calculate innovations and fit AR(1) model
def analyze_innovations(data, optimized_weights, params, kernel_length=250):
    """
    Calculate innovations and fit AR(1) model.
    
    Parameters:
    data (DataFrame): Data containing SPX and VIX
    optimized_weights (array): Optimized weights
    params (tuple): Regression parameters (a, b, c)
    kernel_length (int): Length of the kernel
    
    Returns:
    array: Innovations
    float: AR(1) coefficient
    """
    # Create a matrix of lagged values
    lagged_values = np.zeros((len(data['SPX']) - kernel_length, kernel_length))
    for i in range(kernel_length):
        lagged_values[:, i] = data['SPX'].iloc[i:i+len(data['SPX'])-kernel_length].values
    
    # Calculate weighted average
    weighted_avg = np.dot(lagged_values, optimized_weights)
    
    # Calculate the index (current price / weighted average)
    current_prices = data['SPX'].iloc[kernel_length:].values
    index = current_prices / weighted_avg
    
    # Calculate predicted VIX using the quadratic model
    a, b, c = params
    vix_pred = a + b*index + c*index**2
    
    # Calculate innovations
    vix_actual = data['VIX'].iloc[kernel_length:].values
    innovations = np.log(vix_actual / vix_pred)
    
    # Fit AR(1) model
    X = innovations[:-1].reshape(-1, 1)
    y = innovations[1:]
    
    X_with_const = np.column_stack([np.ones_like(X), X])
    coeffs, _, _, _ = np.linalg.lstsq(X_with_const, y, rcond=None)
    intercept, ar_coef = coeffs
    
    # Calculate predictions
    y_pred = intercept + ar_coef * X.flatten()
    
    # Calculate R² score
    ar_r2 = r2_score(y, y_pred)
    
    # Print results
    print(f"AR(1) model: y_t+1 = {intercept:.4f} + {ar_coef:.4f} * y_t")
    print(f"AR(1) R² score: {ar_r2:.4f}")
    
    # Plot innovations
    plt.figure(figsize=(12, 6))
    plt.plot(data.index[kernel_length:], innovations, 'b-')
    plt.xlabel('Date')
    plt.ylabel('ln(VIX_actual / VIX_predicted)')
    plt.title('Innovations')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('innovations.png')
    plt.close()
    
    # Plot AR(1) fit
    plt.figure(figsize=(10, 6))
    plt.scatter(innovations[:-1], innovations[1:], alpha=0.5)
    
    # Generate points for regression line
    x_range = np.linspace(min(innovations[:-1]), max(innovations[:-1]), 100)
    y_range = intercept + ar_coef * x_range
    
    plt.plot(x_range, y_range, 'r-', linewidth=2)
    plt.xlabel('y_t')
    plt.ylabel('y_t+1')
    plt.title(f'AR(1) Fit (R² = {ar_r2:.4f})')
    plt.grid(True, alpha=0.3)
    plt.text(0.05, 0.95, f'y_t+1 = {intercept:.4f} + {ar_coef:.4f} * y_t', 
             transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top')
    plt.tight_layout()
    plt.savefig('ar1_fit.png')
    plt.close()
    
    return innovations, intercept, ar_coef, ar_r2

# Function to implement the LSV model for option pricing
def lsv_model_simulation(S0, T, r, q, strikes, params, n_paths=10000, n_steps=252):
    """
    Simulate paths using the LSV model and price vanilla options.
    
    Parameters:
    S0 (float): Initial spot price
    T (float): Time to maturity in years
    r (float): Risk-free rate
    q (float): Dividend yield
    strikes (array): Array of strike prices
    params (dict): Model parameters
    n_paths (int): Number of simulation paths
    n_steps (int): Number of time steps
    
    Returns:
    DataFrame: Option prices
    """
    # Extract parameters
    kappa = params['kappa']
    nu = params['nu']
    rho = params['rho']
    kernel_weights = params['kernel_weights']
    vix_params = params['vix_params']  # (a, b, c)
    ar_params = params['ar_params']    # (intercept, ar_coef)
    
    kernel_length = len(kernel_weights)
    dt = T / n_steps
    sqrt_dt = np.sqrt(dt)
    
    # Initialize arrays
    S = np.zeros((n_paths, n_steps + 1))
    Y = np.zeros((n_paths, n_steps + 1))
    
    # Set initial values
    S[:, 0] = S0
    Y[:, 0] = 0  # Start at mean level
    
    # Pre-fill with initial spot for calculating moving averages
    S_history = np.ones(kernel_length) * S0
    
    # Generate correlated random numbers
    Z1 = np.random.normal(0, 1, (n_paths, n_steps))
    Z2 = rho * Z1 + np.sqrt(1 - rho**2) * np.random.normal(0, 1, (n_paths, n_steps))
    
    # Simulate paths
    for t in range(n_steps):
        # Calculate weighted average for each path
        for i in range(n_paths):
            # Update history
            S_history = np.roll(S_history, -1)
            S_history[-1] = S[i, t]
            
            # Calculate local volatility
            weighted_avg = np.sum(kernel_weights * S_history)
            ratio = S[i, t] / weighted_avg
            
            # Apply quadratic function to get local volatility level
            a, b, c = vix_params
            vol_level = (a + b*ratio + c*ratio**2) / 100  # Convert to decimal
            
            # Apply stochastic factor
            vol = vol_level * np.exp(Y[i, t])
            
            # Ensure volatility is reasonable
            vol = max(0.05, min(vol, 1.0))
            
            # Update processes
            Y[i, t+1] = Y[i, t] - kappa * Y[i, t] * dt + nu * sqrt_dt * Z1[i, t]
            S[i, t+1] = S[i, t] * np.exp((r - q - 0.5 * vol**2) * dt + vol * sqrt_dt * Z2[i, t])
    
    # Price vanilla options
    call_prices = {}
    put_prices = {}
    discount = np.exp(-r * T)
    
    for K in strikes:
        # Call payoffs
        call_payoffs = np.maximum(0, S[:, -1] - K)
        call_price = discount * np.mean(call_payoffs)
        call_prices[K] = call_price
        
        # Put payoffs
        put_payoffs = np.maximum(0, K - S[:, -1])
        put_price = discount * np.mean(put_payoffs)
        put_prices[K] = put_price
    
    # Calculate implied volatilities
    call_ivs = {}
    put_ivs = {}
    
    for K in strikes:
        # For calls
        call_iv = implied_volatility(call_prices[K], S0, K, T, r, q, option_type='call')
        call_ivs[K] = call_iv
        
        # For puts
        put_iv = implied_volatility(put_prices[K], S0, K, T, r, q, option_type='put')
        put_ivs[K] = put_iv
    
    # Create DataFrames for prices and IVs
    strikes_array = np.array(strikes)
    df_prices = pd.DataFrame({
        'Strike': strikes,
        'Call_Price': [call_prices[K] for K in strikes],
        'Put_Price': [put_prices[K] for K in strikes]
    })
    
    df_ivs = pd.DataFrame({
        'Strike': strikes,
        'Call_IV': [call_ivs[K] for K in strikes],
        'Put_IV': [put_ivs[K] for K in strikes]
    })
    
    return df_prices, df_ivs, S

# Function to calculate Black-Scholes price
def black_scholes(S, K, T, r, q, sigma, option_type='call'):
    """
    Calculate Black-Scholes option price.
    
    Parameters:
    S (float): Spot price
    K (float): Strike price
    T (float): Time to maturity in years
    r (float): Risk-free rate
    q (float): Dividend yield
    sigma (float): Volatility
    option_type (str): Option type ('call' or 'put')
    
    Returns:
    float: Option price
    """
    d1 = (np.log(S/K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == 'call':
        price = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:  # put
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
    
    return price

# Function to calculate implied volatility
def implied_volatility(price, S, K, T, r, q, option_type='call', tol=1e-6, max_iter=100):
    """
    Calculate implied volatility using Newton-Raphson method.
    
    Parameters:
    price (float): Option price
    S (float): Spot price
    K (float): Strike price
    T (float): Time to maturity in years
    r (float): Risk-free rate
    q (float): Dividend yield
    option_type (str): Option type ('call' or 'put')
    tol (float): Tolerance for convergence
    max_iter (int): Maximum number of iterations
    
    Returns:
    float: Implied volatility
    """
    # Initial guess
    sigma = 0.2
    
    for i in range(max_iter):
        # Calculate price and vega
        bs_price = black_scholes(S, K, T, r, q, sigma, option_type)
        
        # Check if the price is close enough
        if abs(bs_price - price) < tol:
            return sigma
        
        # Calculate vega
        d1 = (np.log(S/K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        vega = S * np.exp(-q * T) * np.sqrt(T) * norm.pdf(d1)
        
        # Update sigma
        if vega < 1e-10:  # Avoid division by zero
            sigma = sigma * 1.5  # Just adjust and try again
        else:
            sigma = sigma - (bs_price - price) / vega
            
        # Bounds check
        sigma = max(0.001, min(sigma, 2.0))
    
    # If no convergence, return NaN
    return np.nan

# Function to price path-dependent options
def price_path_dependent_options(S_paths, T, r, barrier, strike, option_type='up-and-out-call'):
    """
    Price path-dependent options using Monte Carlo simulation.
    
    Parameters:
    S_paths (array): Simulated price paths
    T (float): Time to maturity in years
    r (float): Risk-free rate
    barrier (float): Barrier level
    strike (float): Strike price
    option_type (str): Option type
    
    Returns:
    float: Option price
    """
    n_paths, n_steps = S_paths.shape
    discount = np.exp(-r * T)
    
    if option_type == 'up-and-out-call':
        # Check if barrier is breached in each path
        barrier_breach = np.any(S_paths >= barrier, axis=1)
        
        # Calculate payoffs
        payoffs = np.zeros(n_paths)
        for i in range(n_paths):
            if not barrier_breach[i]:
                payoffs[i] = max(0, S_paths[i, -1] - strike)
        
        # Calculate price
        price = discount * np.mean(payoffs)
        
    elif option_type == 'variance-swap':
        # Calculate realized variance for each path
        log_returns = np.diff(np.log(S_paths), axis=1)
        realized_variance = np.sum(log_returns**2, axis=1) * (n_steps / T)
        
        # Calculate price (expected realized variance)
        price = np.mean(realized_variance) * 100  # Convert to percentage points
        
    elif option_type == 'volatility-swap':
        # Calculate realized volatility for each path
        log_returns = np.diff(np.log(S_paths), axis=1)
        realized_variance = np.sum(log_returns**2, axis=1) * (n_steps / T)
        realized_volatility = np.sqrt(realized_variance)
        
        # Calculate price (expected realized volatility)
        price = np.mean(realized_volatility) * 100  # Convert to percentage points
    
    return price

# Function to perform scenario analysis
def scenario_analysis(params, S0, T, r, q, n_scenarios=5, n_paths=1000, n_steps=252):
    """
    Perform scenario analysis by generating different market conditions.
    
    Parameters:
    params (dict): Model parameters
    S0 (float): Initial spot price
    T (float): Time to maturity in years
    r (float): Risk-free rate
    q (float): Dividend yield
    n_scenarios (int): Number of scenarios
    n_paths (int): Number of paths per scenario
    n_steps (int): Number of time steps
    
    Returns:
    DataFrame: Scenario results
    """
    # Define scenario parameters
    scenario_names = [
        "Base Case",
        "Bull Market",
        "Bear Market",
        "High Volatility",
        "Low Volatility"
    ]
    
    # Base parameters
    kappa = params['kappa']
    nu = params['nu']
    rho = params['rho']
    kernel_weights = params['kernel_weights']
    vix_params = params['vix_params']  # (a, b, c)
    ar_params = params['ar_params']    # (intercept, ar_coef)
    
    # Scenario parameters
    scenario_params = [
        # Base Case
        {
            'kappa': kappa,
            'nu': nu,
            'rho': rho,
            'kernel_weights': kernel_weights,
            'vix_params': vix_params,
            'ar_params': ar_params
        },
        # Bull Market
        {
            'kappa': kappa,
            'nu': nu * 0.8,
            'rho': rho,
            'kernel_weights': kernel_weights,
            'vix_params': (vix_params[0] * 0.8, vix_params[1], vix_params[2]),
            'ar_params': ar_params
        },
        # Bear Market
        {
            'kappa': kappa,
            'nu': nu * 1.2,
            'rho': rho * 1.2,
            'kernel_weights': kernel_weights,
            'vix_params': (vix_params[0] * 1.2, vix_params[1], vix_params[2]),
            'ar_params': ar_params
        },
        # High Volatility
        {
            'kappa': kappa * 0.8,
            'nu': nu * 1.5,
            'rho': rho,
            'kernel_weights': kernel_weights,
            'vix_params': (vix_params[0] * 1.5, vix_params[1], vix_params[2]),
            'ar_params': ar_params
        },
        # Low Volatility
        {
            'kappa': kappa * 1.2,
            'nu': nu * 0.6,
            'rho': rho,
            'kernel_weights': kernel_weights,
            'vix_params': (vix_params[0] * 0.6, vix_params[1], vix_params[2]),
            'ar_params': ar_params
        }
    ]
    
    # Define options to price
    atm_strike = S0
    barrier = 1.2 * S0
    
    # Initialize results
    results = []
    
    # Simulate each scenario
    for i, scenario in enumerate(scenario_params[:n_scenarios]):
        # Simulate paths
        print(f"Simulating scenario: {scenario_names[i]}")
        _, _, S_paths = lsv_model_simulation(
            S0, T, r, q, [atm_strike], scenario, n_paths=n_paths, n_steps=n_steps)
        
        # Price vanilla options
        atm_call_price = black_scholes(S0, atm_strike, T, r, q, scenario['vix_params'][0]/100)
        
        # Price path-dependent options
        up_out_call_price = price_path_dependent_options(
            S_paths, T, r, barrier, atm_strike, 'up-and-out-call')
        variance_swap_price = price_path_dependent_options(
            S_paths, T, r, barrier, atm_strike, 'variance-swap')
        volatility_swap_price = price_path_dependent_options(
            S_paths, T, r, barrier, atm_strike, 'volatility-swap')
        
        # Calculate final spot statistics
        final_spots = S_paths[:, -1]
        avg_spot = np.mean(final_spots)
        std_spot = np.std(final_spots)
        min_spot = np.min(final_spots)
        max_spot = np.max(final_spots)
        
        # Store results
        results.append({
            'Scenario': scenario_names[i],
            'Avg_Final_Spot': avg_spot,
            'Std_Final_Spot': std_spot,
            'Min_Final_Spot': min_spot,
            'Max_Final_Spot': max_spot,
            'ATM_Call_Price': atm_call_price,
            'Up_Out_Call_Price': up_out_call_price,
            'Variance_Swap_Price': variance_swap_price,
            'Volatility_Swap_Price': volatility_swap_price
        })
    
    # Create DataFrame
    df_results = pd.DataFrame(results)
    
    # Plot results
    plt.figure(figsize=(12, 8))
    
    # Plot option prices
    ax1 = plt.subplot(2, 2, 1)
    ax1.bar(df_results['Scenario'], df_results['ATM_Call_Price'], color='blue', alpha=0.7)
    ax1.set_title('ATM Call Price')
    ax1.set_xticklabels(df_results['Scenario'], rotation=45)
    
    ax2 = plt.subplot(2, 2, 2)
    ax2.bar(df_results['Scenario'], df_results['Up_Out_Call_Price'], color='green', alpha=0.7)
    ax2.set_title('Up-and-Out Call Price')
    ax2.set_xticklabels(df_results['Scenario'], rotation=45)
    
    ax3 = plt.subplot(2, 2, 3)
    ax3.bar(df_results['Scenario'], df_results['Variance_Swap_Price'], color='red', alpha=0.7)
    ax3.set_title('Variance Swap Price')
    ax3.set_xticklabels(df_results['Scenario'], rotation=45)
    
    ax4 = plt.subplot(2, 2, 4)
    ax4.bar(df_results['Scenario'], df_results['Volatility_Swap_Price'], color='purple', alpha=0.7)
    ax4.set_title('Volatility Swap Price')
    ax4.set_xticklabels(df_results['Scenario'], rotation=45)
    
    plt.tight_layout()
    plt.savefig('scenario_analysis.png')
    plt.close()
    
    return df_results

# Main function to run all analyses
def main():
    # Try to fetch real data first
    real_data = fetch_real_data()
    
    # Simulate data (use properties from real data if available)
    if real_data is not None:
        print("Using real data properties for simulation")
        data = simulate_data(n_days=3000, use_real_data_properties=True, real_data=real_data)
    else:
        print("Using default parameters for simulation")
        data = simulate_data(n_days=3000)
    
    # Set the index to Date for easier plotting
    data.set_index('Date', inplace=True)
    
    # Analyze SPX-VIX relationship
    print("Analyzing SPX-VIX relationship...")
    full_data = analyze_spx_vix_relationship(data)
    
    # Optimize weights
    print("Optimizing weights...")
    kernel_length = 250
    opt_weights, r2, vix_params = analyze_optimized_weights(data, kernel_length)
    
    # Analyze innovations
    print("Analyzing innovations...")
    innovations, intercept, ar_coef, ar_r2 = analyze_innovations(
        data, opt_weights, vix_params, kernel_length)
    
    # Set parameters for the LSV model
    lsv_params = {
        'kappa': 5.0,
        'nu': 0.3,
        'rho': -0.7,
        'kernel_weights': opt_weights,
        'vix_params': vix_params,
        'ar_params': (intercept, ar_coef)
    }
    
    # Price options using the LSV model
    print("Pricing options...")
    S0 = data['SPX'].iloc[-1]
    T = 1.0  # 1 year
    r = 0.05  # 5% risk-free rate
    q = 0.02  # 2% dividend yield
    
    # Generate strikes around current spot
    strikes = np.linspace(0.8 * S0, 1.2 * S0, 9)
    
    # Price options
    prices, ivs, S_paths = lsv_model_simulation(S0, T, r, q, strikes, lsv_params)
    
    # Plot implied volatility smile
    plt.figure(figsize=(10, 6))
    plt.plot(strikes / S0, ivs['Call_IV'], 'bo-', label='Call IV')
    plt.plot(strikes / S0, ivs['Put_IV'], 'ro-', label='Put IV')
    plt.xlabel('Moneyness (K/S)')
    plt.ylabel('Implied Volatility')
    plt.title('Implied Volatility Smile')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('iv_smile.png')
    plt.close()
    
    # Price path-dependent options
    print("Pricing path-dependent options...")
    barrier = 1.2 * S0
    strike = S0
    
    up_out_call_price = price_path_dependent_options(
        S_paths, T, r, barrier, strike, 'up-and-out-call')
    variance_swap_price = price_path_dependent_options(
        S_paths, T, r, barrier, strike, 'variance-swap')
    volatility_swap_price = price_path_dependent_options(
        S_paths, T, r, barrier, strike, 'volatility-swap')
    
    print(f"Up-and-Out Call Price: {up_out_call_price:.4f}")
    print(f"Variance Swap Price: {variance_swap_price:.4f}")
    print(f"Volatility Swap Price: {volatility_swap_price:.4f}")
    
    # Perform scenario analysis
    print("Performing scenario analysis...")
    scenario_results = scenario_analysis(lsv_params, S0, T, r, q)
    
    print("Analysis complete!")
    return {
        'data': data,
        'full_data': full_data,
        'opt_weights': opt_weights,
        'vix_params': vix_params,
        'ar_params': (intercept, ar_coef),
        'lsv_params': lsv_params,
        'option_prices': prices,
        'implied_volatilities': ivs,
        'path_dependent_prices': {
            'up_out_call': up_out_call_price,
            'variance_swap': variance_swap_price,
            'volatility_swap': volatility_swap_price
        },
        'scenario_results': scenario_results
    }

# Run the main function
if __name__ == "__main__":
    results = main()