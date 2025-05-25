import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm, t as t_dist  # Rename t to t_dist to avoid conflict
from scipy import optimize
from sklearn.linear_model import LinearRegression
import time

np.random.seed(42)

# Set plot style - using a style that's more widely compatible
try:
    plt.style.use('seaborn-darkgrid')  # For older versions of matplotlib
except:
    try:
        plt.style.use('seaborn')  # Fallback option
    except:
        pass  # If no seaborn style is available, use default style

#########################################################
# PART 1: Simulating State Space Models for Pairs Trading
#########################################################

def simulate_pair(model_type, params, n_days=1000):
    """
    Simulate a pair of asset prices based on the state space model.
    
    Parameters:
    -----------
    model_type : str
        'linear', 'nonlinear', 'heteroskedastic', or 'non_gaussian'
    params : dict
        Model parameters
    n_days : int
        Number of days to simulate
    
    Returns:
    --------
    pd.DataFrame with columns ['PA', 'PB', 'spread']
    """
    # Unpack parameters
    gamma = params['gamma']
    phi = params.get('phi', 0)
    sigma_e = params['sigma_e']
    
    # Initialize arrays
    PA = np.zeros(n_days)
    PB = np.zeros(n_days)
    spread = np.zeros(n_days)
    
    # Initialize with some starting values
    PB[0] = 100
    spread[0] = 0
    PA[0] = phi + gamma * PB[0] + spread[0] + np.random.normal(0, sigma_e)
    
    for t in range(1, n_days):
        # Simulate PB with random walk (with small drift)
        PB[t] = PB[t-1] * (1 + np.random.normal(0.0002, 0.01))
        
        # Simulate spread based on model type
        if model_type == 'linear':
            # Model 1: xt+1 = θ1*xt + θ0 + η_t (where η_t ~ N(0, θ2))
            theta_0 = params['theta_0']
            theta_1 = params['theta_1']
            theta_2 = params['theta_2']
            
            spread[t] = theta_0 + theta_1 * spread[t-1] + np.random.normal(0, theta_2)
            
        elif model_type == 'nonlinear':
            # Model 2: xt+1 = θ1*xt + θ3*x_t^2 + θ0 + η_t (where η_t ~ N(0, θ2))
            theta_0 = params['theta_0']
            theta_1 = params['theta_1']
            theta_2 = params['theta_2']
            theta_3 = params['theta_3']
            
            spread[t] = theta_0 + theta_1 * spread[t-1] + theta_3 * spread[t-1]**2 + np.random.normal(0, theta_2)
            
        elif model_type == 'heteroskedastic':
            # Model 3: xt+1 = θ1*xt + θ0 + √(θ2 + θ3*x_t^2) * η_t (where η_t ~ N(0, 1))
            theta_0 = params['theta_0']
            theta_1 = params['theta_1']
            theta_2 = params['theta_2']
            theta_3 = params['theta_3']
            
            vol = np.sqrt(theta_2 + theta_3 * spread[t-1]**2)
            spread[t] = theta_0 + theta_1 * spread[t-1] + vol * np.random.normal(0, 1)
            
        elif model_type == 'non_gaussian':
            # Model 4: xt+1 = θ1*xt + θ0 + θ2 * η_t (where η_t ~ t(df))
            theta_0 = params['theta_0']
            theta_1 = params['theta_1']
            theta_2 = params['theta_2']
            df = params['df']
            
            spread[t] = theta_0 + theta_1 * spread[t-1] + theta_2 * t_dist.rvs(df)  # Use t_dist instead of t
        
        # Generate PA based on spread and PB
        PA[t] = phi + gamma * PB[t] + spread[t] + np.random.normal(0, sigma_e)
    
    return pd.DataFrame({
        'PA': PA,
        'PB': PB,
        'spread': spread
    })

#########################################################
# PART 2: Implementing Kalman Filter for Spread Estimation
#########################################################

def estimate_parameters(PA, PB, model_type='linear'):
    """
    Estimate parameters of the state space model
    
    This is a simplified version using linear regression for the hedge ratio
    and simple moments for other parameters
    """
    # Step 1: Estimate gamma (hedge ratio) using linear regression
    model = LinearRegression()
    model.fit(PB.reshape(-1, 1), PA)
    gamma = model.coef_[0]
    phi = model.intercept_
    
    # Step 2: Estimate the spread
    estimated_spread = PA - (phi + gamma * PB)
    
    # Step 3: Estimate parameters depending on model
    params = {
        'gamma': gamma,
        'phi': phi,
        'sigma_e': np.std(estimated_spread - np.mean(estimated_spread))
    }
    
    if model_type == 'linear':
        # Fit AR(1) model to the spread
        X = estimated_spread[:-1]
        y = estimated_spread[1:]
        model = LinearRegression()
        model.fit(X.reshape(-1, 1), y)
        
        params['theta_0'] = model.intercept_
        params['theta_1'] = model.coef_[0]
        params['theta_2'] = np.std(y - model.predict(X.reshape(-1, 1)))
        
    elif model_type == 'nonlinear':
        # Fit nonlinear AR(1) model: xt+1 = θ0 + θ1*xt + θ3*x_t^2 + η_t
        X = np.column_stack((np.ones_like(estimated_spread[:-1]), 
                             estimated_spread[:-1], 
                             estimated_spread[:-1]**2))
        y = estimated_spread[1:]
        
        coeffs = np.linalg.lstsq(X, y, rcond=None)[0]
        
        params['theta_0'] = coeffs[0]
        params['theta_1'] = coeffs[1]
        params['theta_3'] = coeffs[2]
        params['theta_2'] = np.std(y - np.dot(X, coeffs))
        
    elif model_type == 'heteroskedastic':
        # Fit AR(1) model first
        X = estimated_spread[:-1]
        y = estimated_spread[1:]
        model = LinearRegression()
        model.fit(X.reshape(-1, 1), y)
        
        params['theta_0'] = model.intercept_
        params['theta_1'] = model.coef_[0]
        
        # Get residuals
        residuals = y - model.predict(X.reshape(-1, 1))
        
        # Fit ARCH(1) model to squared residuals
        X_arch = np.column_stack((np.ones_like(X), X**2))
        y_arch = residuals**2
        
        coeffs_arch = np.linalg.lstsq(X_arch, y_arch, rcond=None)[0]
        
        params['theta_2'] = max(0.0001, coeffs_arch[0])  # Ensure positivity
        params['theta_3'] = max(0.0001, coeffs_arch[1])  # Ensure positivity
        
    elif model_type == 'non_gaussian':
        # Fit AR(1) model to the spread
        X = estimated_spread[:-1]
        y = estimated_spread[1:]
        model = LinearRegression()
        model.fit(X.reshape(-1, 1), y)
        
        params['theta_0'] = model.intercept_
        params['theta_1'] = model.coef_[0]
        
        # Get residuals and estimate t-distribution parameters
        residuals = y - model.predict(X.reshape(-1, 1))
        
        # Estimate df using method of moments
        kurtosis = np.mean(residuals**4) / (np.std(residuals)**4)
        df = max(3, 4 * kurtosis / (kurtosis - 3))  # Ensure df > 2
        
        params['theta_2'] = np.std(residuals) * np.sqrt((df-2)/df)
        params['df'] = df
        
    return params, estimated_spread

def kalman_filter(PA, PB, params, model_type='linear'):
    """
    Simple Kalman filter for linear case, approximate for nonlinear cases
    """
    # Extract parameters
    gamma = params['gamma']
    phi = params.get('phi', 0)
    sigma_e = params['sigma_e']
    
    # Initialize
    n = len(PA)
    filtered_spread = np.zeros(n)
    prediction_spread = np.zeros(n)
    
    # Initial values
    filtered_spread[0] = 0
    P = 1.0  # Initial error covariance
    
    for t in range(1, n):
        # Prediction step
        if model_type == 'linear':
            theta_0 = params['theta_0']
            theta_1 = params['theta_1']
            theta_2 = params['theta_2']
            
            prediction_spread[t] = theta_0 + theta_1 * filtered_spread[t-1]
            Q = theta_2**2  # Process noise variance
            
        elif model_type == 'nonlinear':
            theta_0 = params['theta_0']
            theta_1 = params['theta_1']
            theta_2 = params['theta_2']
            theta_3 = params['theta_3']
            
            prediction_spread[t] = theta_0 + theta_1 * filtered_spread[t-1] + theta_3 * filtered_spread[t-1]**2
            Q = theta_2**2  # Process noise variance
            
        elif model_type == 'heteroskedastic':
            theta_0 = params['theta_0']
            theta_1 = params['theta_1']
            theta_2 = params['theta_2']
            theta_3 = params['theta_3']
            
            prediction_spread[t] = theta_0 + theta_1 * filtered_spread[t-1]
            Q = theta_2 + theta_3 * filtered_spread[t-1]**2  # Time-varying process noise variance
            
        elif model_type == 'non_gaussian':
            theta_0 = params['theta_0']
            theta_1 = params['theta_1']
            theta_2 = params['theta_2']
            
            prediction_spread[t] = theta_0 + theta_1 * filtered_spread[t-1]
            Q = theta_2**2 * (params['df'] / (params['df'] - 2))  # Adjusted for t-distribution
        
        # Prediction error covariance
        P_pred = theta_1**2 * P + Q
        
        # Update step
        residual = PA[t] - (phi + gamma * PB[t] + prediction_spread[t])
        S = P_pred + sigma_e**2
        K = P_pred / S  # Kalman gain
        
        # Update estimate
        filtered_spread[t] = prediction_spread[t] + K * residual
        
        # Update error covariance
        P = (1 - K) * P_pred
    
    return filtered_spread, prediction_spread

#########################################################
# PART 3: Implementing Trading Strategies
#########################################################

def implement_strategy(strategy, prices, filtered_spread, upper_boundary, lower_boundary, transaction_cost=0.002):
    """
    Implement trading strategies A, B, C as described in the paper
    
    Parameters:
    -----------
    strategy : str
        'A', 'B', or 'C'
    prices : pd.DataFrame
        DataFrame with columns ['PA', 'PB']
    filtered_spread : array-like
        Filtered spread estimates
    upper_boundary, lower_boundary : float
        Trading thresholds
    transaction_cost : float
        Transaction cost in percentage
    
    Returns:
    --------
    pd.DataFrame with columns ['position', 'pnl', 'cumulative_return']
    """
    n = len(filtered_spread)
    
    # Mean of spread
    mean_spread = np.mean(filtered_spread)
    
    # Initialize position and returns
    position = np.zeros(n)  # 1: long A, short B; -1: short A, long B; 0: no position
    pnl = np.zeros(n)
    
    # Implement strategy
    if strategy == 'A':
        for t in range(1, n):
            if position[t-1] == 0:  # No current position
                if filtered_spread[t] >= upper_boundary:
                    # Short A, long B (because A is overvalued)
                    position[t] = -1
                    pnl[t] = -transaction_cost  # Transaction cost
                elif filtered_spread[t] <= lower_boundary:
                    # Long A, short B (because A is undervalued)
                    position[t] = 1
                    pnl[t] = -transaction_cost  # Transaction cost
                else:
                    position[t] = 0
            
            elif position[t-1] == 1:  # Currently long A, short B
                if filtered_spread[t] >= mean_spread:
                    # Close position
                    pnl[t] = (prices['PA'].iloc[t] / prices['PA'].iloc[t-1] - 1) - \
                             (prices['PB'].iloc[t] / prices['PB'].iloc[t-1] - 1) - \
                             transaction_cost
                    position[t] = 0
                else:
                    # Keep position
                    position[t] = 1
                    pnl[t] = (prices['PA'].iloc[t] / prices['PA'].iloc[t-1] - 1) - \
                             (prices['PB'].iloc[t] / prices['PB'].iloc[t-1] - 1)
            
            elif position[t-1] == -1:  # Currently short A, long B
                if filtered_spread[t] <= mean_spread:
                    # Close position
                    pnl[t] = -(prices['PA'].iloc[t] / prices['PA'].iloc[t-1] - 1) + \
                             (prices['PB'].iloc[t] / prices['PB'].iloc[t-1] - 1) - \
                             transaction_cost
                    position[t] = 0
                else:
                    # Keep position
                    position[t] = -1
                    pnl[t] = -(prices['PA'].iloc[t] / prices['PA'].iloc[t-1] - 1) + \
                             (prices['PB'].iloc[t] / prices['PB'].iloc[t-1] - 1)
    
    elif strategy == 'B':
        for t in range(1, n):
            if filtered_spread[t] >= upper_boundary and filtered_spread[t-1] < upper_boundary:
                # Cross upper boundary from below
                if position[t-1] == 1:
                    # Close long position and open short position
                    pnl[t] = (prices['PA'].iloc[t] / prices['PA'].iloc[t-1] - 1) - \
                           (prices['PB'].iloc[t] / prices['PB'].iloc[t-1] - 1) - \
                           transaction_cost
                    position[t] = -1
                else:
                    # Open short position
                    pnl[t] = -transaction_cost
                    position[t] = -1
            
            elif filtered_spread[t] <= lower_boundary and filtered_spread[t-1] > lower_boundary:
                # Cross lower boundary from above
                if position[t-1] == -1:
                    # Close short position and open long position
                    pnl[t] = -(prices['PA'].iloc[t] / prices['PA'].iloc[t-1] - 1) + \
                           (prices['PB'].iloc[t] / prices['PB'].iloc[t-1] - 1) - \
                           transaction_cost
                    position[t] = 1
                else:
                    # Open long position
                    pnl[t] = -transaction_cost
                    position[t] = 1
            
            else:
                # Keep previous position
                position[t] = position[t-1]
                
                if position[t] == 1:
                    pnl[t] = (prices['PA'].iloc[t] / prices['PA'].iloc[t-1] - 1) - \
                           (prices['PB'].iloc[t] / prices['PB'].iloc[t-1] - 1)
                elif position[t] == -1:
                    pnl[t] = -(prices['PA'].iloc[t] / prices['PA'].iloc[t-1] - 1) + \
                           (prices['PB'].iloc[t] / prices['PB'].iloc[t-1] - 1)
    
    elif strategy == 'C':
        for t in range(1, n):
            # Previous position is 0 (no position)
            if position[t-1] == 0:
                if filtered_spread[t-1] > upper_boundary and filtered_spread[t] <= upper_boundary:
                    # Cross upper boundary from above
                    position[t] = -1
                    pnl[t] = -transaction_cost
                elif filtered_spread[t-1] < lower_boundary and filtered_spread[t] >= lower_boundary:
                    # Cross lower boundary from below
                    position[t] = 1
                    pnl[t] = -transaction_cost
                else:
                    position[t] = 0
            
            # Previous position is 1 (long A, short B)
            elif position[t-1] == 1:
                if filtered_spread[t] >= mean_spread:
                    # Cross mean from below - close position
                    position[t] = 0
                    pnl[t] = (prices['PA'].iloc[t] / prices['PA'].iloc[t-1] - 1) - \
                           (prices['PB'].iloc[t] / prices['PB'].iloc[t-1] - 1) - \
                           transaction_cost
                # In Strategy C, we close position when spread crosses upper boundary after we've opened a position
                elif filtered_spread[t] >= upper_boundary:
                    # Cross upper boundary - close position
                    position[t] = 0
                    pnl[t] = (prices['PA'].iloc[t] / prices['PA'].iloc[t-1] - 1) - \
                           (prices['PB'].iloc[t] / prices['PB'].iloc[t-1] - 1) - \
                           transaction_cost
                else:
                    # Keep position
                    position[t] = 1
                    pnl[t] = (prices['PA'].iloc[t] / prices['PA'].iloc[t-1] - 1) - \
                           (prices['PB'].iloc[t] / prices['PB'].iloc[t-1] - 1)
            
            # Previous position is -1 (short A, long B)
            elif position[t-1] == -1:
                if filtered_spread[t] <= mean_spread:
                    # Cross mean from above - close position
                    position[t] = 0
                    pnl[t] = -(prices['PA'].iloc[t] / prices['PA'].iloc[t-1] - 1) + \
                           (prices['PB'].iloc[t] / prices['PB'].iloc[t-1] - 1) - \
                           transaction_cost
                # In Strategy C, we close position when spread crosses lower boundary after we've opened a position
                elif filtered_spread[t] <= lower_boundary:
                    # Cross lower boundary - close position
                    position[t] = 0
                    pnl[t] = -(prices['PA'].iloc[t] / prices['PA'].iloc[t-1] - 1) + \
                           (prices['PB'].iloc[t] / prices['PB'].iloc[t-1] - 1) - \
                           transaction_cost
                else:
                    # Keep position
                    position[t] = -1
                    pnl[t] = -(prices['PA'].iloc[t] / prices['PA'].iloc[t-1] - 1) + \
                           (prices['PB'].iloc[t] / prices['PB'].iloc[t-1] - 1)
    
    # Calculate cumulative returns
    cumulative_return = np.cumprod(1 + pnl) - 1
    
    return pd.DataFrame({
        'position': position,
        'pnl': pnl,
        'cumulative_return': cumulative_return
    })

def calculate_performance_metrics(returns):
    """Calculate performance metrics for a strategy"""
    daily_returns = returns['pnl']
    
    # Annualized metrics (assuming 252 trading days)
    annual_return = np.prod(1 + daily_returns) ** (252 / len(daily_returns)) - 1
    annual_std = np.std(daily_returns) * np.sqrt(252)
    
    # Sharpe ratio (assuming 0% risk-free rate)
    sharpe_ratio = annual_return / annual_std if annual_std > 0 else 0
    
    # Maximum drawdown
    cumulative_returns = returns['cumulative_return']
    peak = np.maximum.accumulate(cumulative_returns)
    drawdown = (cumulative_returns - peak) / (1 + peak)
    max_drawdown = np.min(drawdown)
    
    # Calmar ratio
    calmar_ratio = -annual_return / max_drawdown if max_drawdown < 0 else np.inf
    
    # Pain index (average drawdown)
    pain_index = np.mean(np.abs(drawdown))
    
    return {
        'annual_return': annual_return,
        'annual_std': annual_std,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'calmar_ratio': calmar_ratio,
        'pain_index': pain_index,
        'total_trades': np.sum(np.abs(np.diff(returns['position']) != 0))
    }

def grid_search_optimal_boundaries(prices, filtered_spread, spread_std, strategy, transaction_cost=0.002):
    """Find optimal boundaries via grid search"""
    # Define range for upper and lower boundaries as multiples of standard deviation
    upper_multipliers = np.arange(0.1, 2.6, 0.1)
    lower_multipliers = -np.arange(0.1, 2.6, 0.1)
    
    best_sharpe = -np.inf
    best_return = -np.inf
    best_upper_sharpe = None
    best_lower_sharpe = None
    best_upper_return = None
    best_lower_return = None
    
    results = []
    
    for upper_mult in upper_multipliers:
        for lower_mult in lower_multipliers:
            upper_boundary = np.mean(filtered_spread) + upper_mult * spread_std
            lower_boundary = np.mean(filtered_spread) + lower_mult * spread_std
            
            # Implement strategy
            performance = implement_strategy(
                strategy, 
                prices, 
                filtered_spread, 
                upper_boundary, 
                lower_boundary,
                transaction_cost
            )
            
            # Calculate metrics
            metrics = calculate_performance_metrics(performance)
            
            results.append({
                'upper_mult': upper_mult,
                'lower_mult': lower_mult,
                'sharpe_ratio': metrics['sharpe_ratio'],
                'annual_return': metrics['annual_return']
            })
            
            # Check if we have a new best
            if metrics['sharpe_ratio'] > best_sharpe:
                best_sharpe = metrics['sharpe_ratio']
                best_upper_sharpe = upper_mult
                best_lower_sharpe = lower_mult
            
            if metrics['annual_return'] > best_return:
                best_return = metrics['annual_return']
                best_upper_return = upper_mult
                best_lower_return = lower_mult
    
    return {
        'best_upper_sharpe': best_upper_sharpe,
        'best_lower_sharpe': best_lower_sharpe,
        'best_sharpe': best_sharpe,
        'best_upper_return': best_upper_return,
        'best_lower_return': best_lower_return,
        'best_return': best_return,
        'results': pd.DataFrame(results)
    }

#########################################################
# PART 4: Running the Simulations and Evaluating Results
#########################################################

def run_simulation(model_type, params, duration=1000, transaction_cost=0.002):
    """Run full simulation for a model type and compare strategies"""
    print(f"\n==== Testing {model_type.upper()} Model ====")
    
    # 1. Simulate pair
    print("Simulating pairs trading data...")
    simulated_data = simulate_pair(model_type, params, n_days=duration)
    
    # 2. Estimate parameters
    print("Estimating model parameters...")
    estimated_params, raw_spread = estimate_parameters(
        simulated_data['PA'].values, 
        simulated_data['PB'].values,
        model_type
    )
    
    # 3. Apply Kalman filter
    print("Applying Kalman filter...")
    filtered_spread, _ = kalman_filter(
        simulated_data['PA'].values,
        simulated_data['PB'].values,
        estimated_params,
        model_type
    )
    
    # Compute spread statistics
    spread_mean = np.mean(filtered_spread)
    spread_std = np.std(filtered_spread)
    
    # 4. Find optimal parameters for each strategy via grid search
    print("Finding optimal trading boundaries for each strategy...")
    
    # Extract just the price data
    prices = simulated_data[['PA', 'PB']]
    
    # A) Find optimal parameters for Strategy A
    optimal_A = grid_search_optimal_boundaries(
        prices, filtered_spread, spread_std, 'A', transaction_cost
    )
    
    # B) Find optimal parameters for Strategy B
    optimal_B = grid_search_optimal_boundaries(
        prices, filtered_spread, spread_std, 'B', transaction_cost
    )
    
    # C) Find optimal parameters for Strategy C
    optimal_C = grid_search_optimal_boundaries(
        prices, filtered_spread, spread_std, 'C', transaction_cost
    )
    
    # 5. Implement each strategy with optimal parameters (for Sharpe)
    print("Implementing strategies with optimal parameters...")
    
    # Strategy A
    upper_A = spread_mean + optimal_A['best_upper_sharpe'] * spread_std
    lower_A = spread_mean + optimal_A['best_lower_sharpe'] * spread_std
    
    result_A = implement_strategy(
        'A', prices, filtered_spread, upper_A, lower_A, transaction_cost
    )
    metrics_A = calculate_performance_metrics(result_A)
    
    # Strategy B
    upper_B = spread_mean + optimal_B['best_upper_sharpe'] * spread_std
    lower_B = spread_mean + optimal_B['best_lower_sharpe'] * spread_std
    
    result_B = implement_strategy(
        'B', prices, filtered_spread, upper_B, lower_B, transaction_cost
    )
    metrics_B = calculate_performance_metrics(result_B)
    
    # Strategy C
    upper_C = spread_mean + optimal_C['best_upper_sharpe'] * spread_std
    lower_C = spread_mean + optimal_C['best_lower_sharpe'] * spread_std
    
    result_C = implement_strategy(
        'C', prices, filtered_spread, upper_C, lower_C, transaction_cost
    )
    metrics_C = calculate_performance_metrics(result_C)
    
    # 6. Compare results
    print("\n== Performance Summary (Based on Max Sharpe) ==")
    print(f"Strategy A: Sharpe={metrics_A['sharpe_ratio']:.4f}, Return={metrics_A['annual_return']*100:.2f}%, Trades={metrics_A['total_trades']}")
    print(f"Strategy B: Sharpe={metrics_B['sharpe_ratio']:.4f}, Return={metrics_B['annual_return']*100:.2f}%, Trades={metrics_B['total_trades']}")
    print(f"Strategy C: Sharpe={metrics_C['sharpe_ratio']:.4f}, Return={metrics_C['annual_return']*100:.2f}%, Trades={metrics_C['total_trades']}")
    
    print("\nOptimal Boundaries (as multiple of std dev):")
    print(f"Strategy A: Upper={optimal_A['best_upper_sharpe']:.1f}, Lower={optimal_A['best_lower_sharpe']:.1f}")
    print(f"Strategy B: Upper={optimal_B['best_upper_sharpe']:.1f}, Lower={optimal_B['best_lower_sharpe']:.1f}")
    print(f"Strategy C: Upper={optimal_C['best_upper_sharpe']:.1f}, Lower={optimal_C['best_lower_sharpe']:.1f}")
    
    # 7. Plot results
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Prices and estimated spread
    plt.subplot(3, 1, 1)
    plt.plot(simulated_data['PA'], label='Price A')
    plt.plot(simulated_data['PB'], label='Price B')
    plt.title(f'Simulated Prices - {model_type.capitalize()} Model')
    plt.legend()
    plt.grid(True)
    
    # Plot 2: True and filtered spread
    plt.subplot(3, 1, 2)
    plt.plot(simulated_data['spread'], label='True Spread', alpha=0.7)
    plt.plot(filtered_spread, label='Filtered Spread', alpha=0.7)
    plt.axhline(y=spread_mean, color='g', linestyle='-', label='Mean')
    plt.axhline(y=upper_A, color='r', linestyle='--', label=f'Upper Bound A')
    plt.axhline(y=lower_A, color='r', linestyle='--', label=f'Lower Bound A')
    plt.axhline(y=upper_C, color='m', linestyle=':', label=f'Upper Bound C')
    plt.axhline(y=lower_C, color='m', linestyle=':', label=f'Lower Bound C')
    plt.title('Spread Comparison')
    plt.legend()
    plt.grid(True)
    
    # Plot 3: Cumulative returns of strategies
    plt.subplot(3, 1, 3)
    plt.plot(result_A['cumulative_return'], label=f'Strategy A (Sharpe={metrics_A["sharpe_ratio"]:.2f})')
    plt.plot(result_B['cumulative_return'], label=f'Strategy B (Sharpe={metrics_B["sharpe_ratio"]:.2f})')
    plt.plot(result_C['cumulative_return'], label=f'Strategy C (Sharpe={metrics_C["sharpe_ratio"]:.2f})')
    plt.title('Cumulative Returns')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'pairs_trading_{model_type}_model.png', dpi=150)
    plt.close()
    
    # 8. Return results
    return {
        'model_type': model_type,
        'simulated_data': simulated_data,
        'estimated_params': estimated_params,
        'filtered_spread': filtered_spread,
        'optimal_A': optimal_A,
        'optimal_B': optimal_B,
        'optimal_C': optimal_C,
        'result_A': result_A,
        'result_B': result_B,
        'result_C': result_C,
        'metrics_A': metrics_A,
        'metrics_B': metrics_B,
        'metrics_C': metrics_C
    }

# Run simulations for all model types
def run_all_simulations():
    """Run simulations for all four model types"""
    results = {}
    
    # 1. Linear Model (Model I in the paper)
    linear_params = {
        'gamma': 1.0,
        'phi': 0.0,
        'sigma_e': 0.01,
        'theta_0': 0.0001,  # Small drift
        'theta_1': 0.959,   # Mean reversion coefficient
        'theta_2': 0.03     # Volatility
    }
    results['linear'] = run_simulation('linear', linear_params)
    
    # 2. Nonlinear Model
    nonlinear_params = {
        'gamma': 1.0,
        'phi': 0.0,
        'sigma_e': 0.01,
        'theta_0': 0.0001,
        'theta_1': 0.9,
        'theta_2': 0.03,
        'theta_3': 0.1      # Nonlinear coefficient
    }
    results['nonlinear'] = run_simulation('nonlinear', nonlinear_params)
    
    # 3. Heteroskedastic Model (Model II in the paper)
    heteroskedastic_params = {
        'gamma': 1.0,
        'phi': 0.0,
        'sigma_e': 0.01,
        'theta_0': 0.0001,
        'theta_1': 0.959,
        'theta_2': 0.001,   # Base volatility
        'theta_3': 0.1      # ARCH effect
    }
    results['heteroskedastic'] = run_simulation('heteroskedastic', heteroskedastic_params)
    
    # 4. Non-Gaussian Model
    non_gaussian_params = {
        'gamma': 1.0,
        'phi': 0.0,
        'sigma_e': 0.01,
        'theta_0': 0.0001,
        'theta_1': 0.959,
        'theta_2': 0.03,
        'df': 3              # Degrees of freedom for t-distribution
    }
    results['non_gaussian'] = run_simulation('non_gaussian', non_gaussian_params)
    
    # Summary of results
    summary = pd.DataFrame({
        'Model': ['Linear', 'Nonlinear', 'Heteroskedastic', 'Non-Gaussian'],
        'Strategy_A_Sharpe': [
            results['linear']['metrics_A']['sharpe_ratio'],
            results['nonlinear']['metrics_A']['sharpe_ratio'],
            results['heteroskedastic']['metrics_A']['sharpe_ratio'],
            results['non_gaussian']['metrics_A']['sharpe_ratio']
        ],
        'Strategy_B_Sharpe': [
            results['linear']['metrics_B']['sharpe_ratio'],
            results['nonlinear']['metrics_B']['sharpe_ratio'],
            results['heteroskedastic']['metrics_B']['sharpe_ratio'],
            results['non_gaussian']['metrics_B']['sharpe_ratio']
        ],
        'Strategy_C_Sharpe': [
            results['linear']['metrics_C']['sharpe_ratio'],
            results['nonlinear']['metrics_C']['sharpe_ratio'],
            results['heteroskedastic']['metrics_C']['sharpe_ratio'],
            results['non_gaussian']['metrics_C']['sharpe_ratio']
        ],
        'Strategy_A_Return': [
            results['linear']['metrics_A']['annual_return'],
            results['nonlinear']['metrics_A']['annual_return'],
            results['heteroskedastic']['metrics_A']['annual_return'],
            results['non_gaussian']['metrics_A']['annual_return']
        ],
        'Strategy_B_Return': [
            results['linear']['metrics_B']['annual_return'],
            results['nonlinear']['metrics_B']['annual_return'],
            results['heteroskedastic']['metrics_B']['annual_return'],
            results['non_gaussian']['metrics_B']['annual_return']
        ],
        'Strategy_C_Return': [
            results['linear']['metrics_C']['annual_return'],
            results['nonlinear']['metrics_C']['annual_return'],
            results['heteroskedastic']['metrics_C']['annual_return'],
            results['non_gaussian']['metrics_C']['annual_return']
        ]
    })
    
    # Plot summary of Sharpe ratios
    plt.figure(figsize=(12, 6))
    
    x = np.arange(len(summary['Model']))
    width = 0.25
    
    plt.bar(x - width, summary['Strategy_A_Sharpe'], width, label='Strategy A')
    plt.bar(x, summary['Strategy_B_Sharpe'], width, label='Strategy B')
    plt.bar(x + width, summary['Strategy_C_Sharpe'], width, label='Strategy C')
    
    plt.xlabel('Model Type')
    plt.ylabel('Sharpe Ratio')
    plt.title('Sharpe Ratio Comparison Across Models and Strategies')
    plt.xticks(x, summary['Model'])
    plt.legend()
    plt.grid(axis='y')
    
    plt.tight_layout()
    plt.savefig('pairs_trading_sharpe_comparison.png', dpi=150)
    plt.close()
    
    # Plot summary of returns
    plt.figure(figsize=(12, 6))
    
    plt.bar(x - width, summary['Strategy_A_Return'], width, label='Strategy A')
    plt.bar(x, summary['Strategy_B_Return'], width, label='Strategy B')
    plt.bar(x + width, summary['Strategy_C_Return'], width, label='Strategy C')
    
    plt.xlabel('Model Type')
    plt.ylabel('Annual Return')
    plt.title('Return Comparison Across Models and Strategies')
    plt.xticks(x, summary['Model'])
    plt.legend()
    plt.grid(axis='y')
    
    plt.tight_layout()
    plt.savefig('pairs_trading_return_comparison.png', dpi=150)
    plt.close()
    
    print("\n===== SUMMARY OF RESULTS =====")
    print(summary)
    
    return results, summary

# Execute all simulations
if __name__ == "__main__":
    start_time = time.time()
    results, summary = run_all_simulations()
    end_time = time.time()
    print(f"\nSimulations completed in {end_time - start_time:.2f} seconds")