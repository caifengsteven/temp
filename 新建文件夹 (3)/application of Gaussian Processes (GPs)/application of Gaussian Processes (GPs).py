import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, RationalQuadratic, Matern, ConstantKernel
import pandas as pd
from scipy.optimize import minimize

# Set random seed for reproducibility
np.random.seed(42)

# Function to simulate an Ornstein-Uhlenbeck process
def simulate_ou_process(alpha, mu, sigma, dt, T, x0=None):
    """
    Simulates an Ornstein-Uhlenbeck process.
    
    Parameters:
    - alpha: mean reversion rate
    - mu: long-term mean
    - sigma: volatility
    - dt: time step
    - T: total time
    - x0: initial value (optional)
    
    Returns:
    - Time series of the OU process
    """
    n_steps = int(T / dt)
    X = np.zeros(n_steps)
    
    if x0 is None:
        X[0] = mu
    else:
        X[0] = x0
    
    for t in range(1, n_steps):
        # Using exact method as mentioned in the paper (Gillespie, 1996)
        X[t] = X[t-1] * np.exp(-alpha * dt) + mu * (1 - np.exp(-alpha * dt)) + \
               sigma * np.sqrt((1 - np.exp(-2 * alpha * dt)) / (2 * alpha)) * np.random.normal(0, 1)
    
    return X

# Function to create a seasonal time series with an OU process
def create_structured_time_series(years, days_per_year, alpha, mu, sigma, season_amplitude):
    """
    Creates a structured time series with seasonal patterns and OU noise.
    
    Parameters:
    - years: number of years
    - days_per_year: days per year
    - alpha, mu, sigma: OU process parameters
    - season_amplitude: amplitude of the seasonal component
    
    Returns:
    - Time series with seasonal patterns and OU noise
    - Time points
    """
    total_days = years * days_per_year
    t = np.arange(total_days)
    
    # Seasonal component (sine wave with yearly pattern)
    seasonal = season_amplitude * np.sin(2 * np.pi * t / days_per_year)
    
    # OU process component
    ou = simulate_ou_process(alpha, mu, sigma, 1, total_days)
    
    # Combined time series
    series = seasonal + ou
    
    return series, t

# Function to prepare functional data representation
def prepare_functional_data(time_series, t, days_per_year):
    """
    Prepares functional data representation as described in the paper.
    
    Parameters:
    - time_series: the original time series
    - t: time points
    - days_per_year: days per year
    
    Returns:
    - X: features (year, day in year)
    - y: target values
    """
    years = t // days_per_year
    days = t % days_per_year
    
    X = np.column_stack((years, days))
    y = time_series
    
    return X, y

# Function to prepare augmented data representation
def prepare_augmented_data(time_series, t, days_per_year, max_delta=50):
    """
    Prepares augmented data representation as described in the paper.
    
    Parameters:
    - time_series: the original time series
    - t: time points
    - days_per_year: days per year
    - max_delta: maximum prediction horizon
    
    Returns:
    - X_aug: features (year, day, observe_value, delta)
    - y_aug: target values
    """
    X_aug = []
    y_aug = []
    
    for i in range(len(time_series) - max_delta):
        year = t[i] // days_per_year
        day = t[i] % days_per_year
        observe_value = time_series[i]
        
        for delta in range(1, max_delta + 1):
            if i + delta < len(time_series):
                X_aug.append([year, day, observe_value, delta])
                y_aug.append(time_series[i + delta])
    
    return np.array(X_aug), np.array(y_aug)

# Function to prepare functional-augmented data representation
def prepare_functional_augmented_data(time_series, t, days_per_year, max_delta=50):
    """
    Prepares functional-augmented data representation as described in the paper.
    
    Parameters:
    - time_series: the original time series
    - t: time points
    - days_per_year: days per year
    - max_delta: maximum prediction horizon
    
    Returns:
    - X_func_aug: features (year, day, observe_value, delta)
    - y_func_aug: target values
    """
    # Similar to augmented but we add the year and day as separate features
    X_func_aug = []
    y_func_aug = []
    
    for i in range(len(time_series) - max_delta):
        year = t[i] // days_per_year
        day = t[i] % days_per_year
        observe_value = time_series[i]
        
        for delta in range(1, max_delta + 1):
            if i + delta < len(time_series):
                X_func_aug.append([year, day, observe_value, delta])
                y_func_aug.append(time_series[i + delta])
    
    return np.array(X_func_aug), np.array(y_func_aug)

# Function to fit an AR(1) model
def fit_ar1(time_series):
    """
    Fits an AR(1) model to the time series.
    
    Parameters:
    - time_series: the time series data
    
    Returns:
    - phi: AR coefficient
    - c: constant term
    - sigma: standard deviation of residuals
    """
    y = time_series[1:]
    x = time_series[:-1]
    
    # Calculate phi and c using formulas
    n = len(x)
    phi = np.sum((x - np.mean(x)) * (y - np.mean(y))) / np.sum((x - np.mean(x))**2)
    c = np.mean(y) - phi * np.mean(x)
    
    # Calculate residuals and sigma
    residuals = y - (c + phi * x)
    sigma = np.std(residuals)
    
    return phi, c, sigma

# Function to predict using AR(1) model
def predict_ar1(last_value, phi, c, steps):
    """
    Predicts future values using an AR(1) model.
    
    Parameters:
    - last_value: the last observed value
    - phi: AR coefficient
    - c: constant term
    - steps: number of steps to predict
    
    Returns:
    - predictions: predicted values
    """
    predictions = np.zeros(steps)
    predictions[0] = c + phi * last_value
    
    for i in range(1, steps):
        predictions[i] = c + phi * predictions[i-1]
    
    return predictions

# Function to calculate mean squared error
def calculate_mse(predictions, actual):
    """
    Calculates mean squared error between predictions and actual values.
    
    Parameters:
    - predictions: predicted values
    - actual: actual values
    
    Returns:
    - mse: mean squared error
    """
    return np.mean((predictions - actual)**2)

# Main experiment function
def run_experiment(noise_levels, prediction_horizons=[10, 20, 30], n_test_simulations=1000):
    """
    Runs the experiment as described in the paper.
    
    Parameters:
    - noise_levels: list of noise levels (sigma) to test
    - prediction_horizons: list of prediction horizons to evaluate
    - n_test_simulations: number of test simulations to generate the test distribution
    
    Returns:
    - results: dictionary containing MSE results for each model and noise level
    """
    # Simulation parameters
    years = 10
    days_per_year = 252  # Trading days in a year
    alpha = 0.1  # Mean reversion rate
    mu = 0.0  # Long-term mean
    season_amplitude = 1.0  # Amplitude of seasonal component
    max_delta = 50  # Maximum prediction horizon for augmented data
    
    results = {horizon: {noise: {} for noise in noise_levels} for horizon in prediction_horizons}
    
    for sigma in noise_levels:
        print(f"Running experiment with noise level sigma = {sigma}")
        
        # Create training data
        train_series, train_t = create_structured_time_series(years, days_per_year, alpha, mu, sigma, season_amplitude)
        
        # Prepare data representations
        X_std, y_std = train_t.reshape(-1, 1), train_series
        X_func, y_func = prepare_functional_data(train_series, train_t, days_per_year)
        X_aug, y_aug = prepare_augmented_data(train_series, train_t, days_per_year, max_delta)
        X_func_aug, y_func_aug = prepare_functional_augmented_data(train_series, train_t, days_per_year, max_delta)
        
        # Subsample for GP efficiency as mentioned in the paper
        n_samples = 2000
        if len(X_aug) > n_samples:
            idx_aug = np.random.choice(len(X_aug), n_samples, replace=False)
            X_aug, y_aug = X_aug[idx_aug], y_aug[idx_aug]
        
        if len(X_func_aug) > n_samples:
            idx_func_aug = np.random.choice(len(X_func_aug), n_samples, replace=False)
            X_func_aug, y_func_aug = X_func_aug[idx_func_aug], y_func_aug[idx_func_aug]
        
        # Define kernels
        kernel_std = RationalQuadratic(length_scale=1.0, alpha=1.0)
        kernel_func = RationalQuadratic(length_scale=1.0, alpha=1.0)
        kernel_aug = RationalQuadratic(length_scale=1.0, alpha=1.0)
        kernel_func_aug = RationalQuadratic(length_scale=1.0, alpha=1.0)
        
        # Fit models
        gp_std = GaussianProcessRegressor(kernel=kernel_std, n_restarts_optimizer=5)
        gp_func = GaussianProcessRegressor(kernel=kernel_func, n_restarts_optimizer=5)
        gp_aug = GaussianProcessRegressor(kernel=kernel_aug, n_restarts_optimizer=5)
        gp_func_aug = GaussianProcessRegressor(kernel=kernel_func_aug, n_restarts_optimizer=5)
        
        gp_std.fit(X_std, y_std)
        gp_func.fit(X_func, y_func)
        gp_aug.fit(X_aug, y_aug)
        gp_func_aug.fit(X_func_aug, y_func_aug)
        
        # Fit AR(1) model
        phi, c, ar_sigma = fit_ar1(train_series)
        
        # Create test scenarios by simulating from last point
        last_point = train_series[-1]
        max_horizon = max(prediction_horizons)
        
        # Generate multiple test paths to get the test distribution
        test_paths = np.zeros((n_test_simulations, max_horizon))
        for i in range(n_test_simulations):
            test_path = simulate_ou_process(alpha, mu, sigma, 1, max_horizon, last_point)
            # Add seasonal component
            for j in range(max_horizon):
                day = (train_t[-1] + j + 1) % days_per_year
                test_path[j] += season_amplitude * np.sin(2 * np.pi * day / days_per_year)
            test_paths[i] = test_path
        
        # Calculate actual test distribution statistics
        test_means = np.mean(test_paths, axis=0)
        test_stds = np.std(test_paths, axis=0)
        
        # Make predictions with each model
        # Standard GP
        X_test_std = np.arange(train_t[-1] + 1, train_t[-1] + max_horizon + 1).reshape(-1, 1)
        y_pred_std, y_std_std = gp_std.predict(X_test_std, return_std=True)
        
        # Functional GP
        X_test_func = []
        for i in range(max_horizon):
            t_new = train_t[-1] + i + 1
            year = t_new // days_per_year
            day = t_new % days_per_year
            X_test_func.append([year, day])
        X_test_func = np.array(X_test_func)
        y_pred_func, y_std_func = gp_func.predict(X_test_func, return_std=True)
        
        # Augmented GP
        X_test_aug = []
        for delta in range(1, max_horizon + 1):
            X_test_aug.append([years-1, train_t[-1] % days_per_year, train_series[-1], delta])
        X_test_aug = np.array(X_test_aug)
        y_pred_aug, y_std_aug = gp_aug.predict(X_test_aug, return_std=True)
        
        # Functional-Augmented GP
        X_test_func_aug = []
        for delta in range(1, max_horizon + 1):
            X_test_func_aug.append([years-1, train_t[-1] % days_per_year, train_series[-1], delta])
        X_test_func_aug = np.array(X_test_func_aug)
        y_pred_func_aug, y_std_func_aug = gp_func_aug.predict(X_test_func_aug, return_std=True)
        
        # AR(1) predictions
        y_pred_ar = predict_ar1(train_series[-1], phi, c, max_horizon)
        
        # Evaluate for each prediction horizon
        for horizon in prediction_horizons:
            # Calculate MSE for mean predictions
            mse_std = calculate_mse(y_pred_std[:horizon], test_means[:horizon])
            mse_func = calculate_mse(y_pred_func[:horizon], test_means[:horizon])
            mse_aug = calculate_mse(y_pred_aug[:horizon], test_means[:horizon])
            mse_func_aug = calculate_mse(y_pred_func_aug[:horizon], test_means[:horizon])
            mse_ar = calculate_mse(y_pred_ar[:horizon], test_means[:horizon])
            
            # Store results
            results[horizon][sigma] = {
                'Standard GP': mse_std,
                'Functional GP': mse_func,
                'Augmented GP': mse_aug,
                'Functional-Augmented GP': mse_func_aug,
                'AR(1)': mse_ar
            }
            
            # Also store std deviation accuracy (optional)
            std_err_std = np.mean((y_std_std[:horizon] - test_stds[:horizon])**2)
            std_err_func = np.mean((y_std_func[:horizon] - test_stds[:horizon])**2)
            std_err_aug = np.mean((y_std_aug[:horizon] - test_stds[:horizon])**2)
            std_err_func_aug = np.mean((y_std_func_aug[:horizon] - test_stds[:horizon])**2)
            
            results[horizon][sigma].update({
                'Standard GP Std Err': std_err_std,
                'Functional GP Std Err': std_err_func,
                'Augmented GP Std Err': std_err_aug,
                'Functional-Augmented GP Std Err': std_err_func_aug
            })
    
    return results

# Function to visualize predictions
def visualize_predictions(noise_level=0.5, prediction_horizon=30):
    """
    Visualizes the predictions from different models.
    
    Parameters:
    - noise_level: noise level (sigma) to use
    - prediction_horizon: prediction horizon to visualize
    """
    # Simulation parameters
    years = 10
    days_per_year = 252
    alpha = 0.1
    mu = 0.0
    season_amplitude = 1.0
    max_delta = 50
    
    # Create training data
    train_series, train_t = create_structured_time_series(years, days_per_year, alpha, mu, noise_level, season_amplitude)
    
    # Prepare data representations
    X_std, y_std = train_t.reshape(-1, 1), train_series
    X_func, y_func = prepare_functional_data(train_series, train_t, days_per_year)
    X_aug, y_aug = prepare_augmented_data(train_series, train_t, days_per_year, max_delta)
    X_func_aug, y_func_aug = prepare_functional_augmented_data(train_series, train_t, days_per_year, max_delta)
    
    # Subsample
    n_samples = 2000
    if len(X_aug) > n_samples:
        idx_aug = np.random.choice(len(X_aug), n_samples, replace=False)
        X_aug, y_aug = X_aug[idx_aug], y_aug[idx_aug]
    
    if len(X_func_aug) > n_samples:
        idx_func_aug = np.random.choice(len(X_func_aug), n_samples, replace=False)
        X_func_aug, y_func_aug = X_func_aug[idx_func_aug], y_func_aug[idx_func_aug]
    
    # Define kernels
    kernel_std = RationalQuadratic(length_scale=1.0, alpha=1.0)
    kernel_func = RationalQuadratic(length_scale=1.0, alpha=1.0)
    kernel_aug = RationalQuadratic(length_scale=1.0, alpha=1.0)
    kernel_func_aug = RationalQuadratic(length_scale=1.0, alpha=1.0)
    
    # Fit models
    gp_std = GaussianProcessRegressor(kernel=kernel_std, n_restarts_optimizer=5)
    gp_func = GaussianProcessRegressor(kernel=kernel_func, n_restarts_optimizer=5)
    gp_aug = GaussianProcessRegressor(kernel=kernel_aug, n_restarts_optimizer=5)
    gp_func_aug = GaussianProcessRegressor(kernel=kernel_func_aug, n_restarts_optimizer=5)
    
    gp_std.fit(X_std, y_std)
    gp_func.fit(X_func, y_func)
    gp_aug.fit(X_aug, y_aug)
    gp_func_aug.fit(X_func_aug, y_func_aug)
    
    # Fit AR(1) model
    phi, c, ar_sigma = fit_ar1(train_series)
    
    # Generate test data
    n_test_simulations = 100
    test_paths = np.zeros((n_test_simulations, prediction_horizon))
    for i in range(n_test_simulations):
        test_path = simulate_ou_process(alpha, mu, noise_level, 1, prediction_horizon, train_series[-1])
        # Add seasonal component
        for j in range(prediction_horizon):
            day = (train_t[-1] + j + 1) % days_per_year
            test_path[j] += season_amplitude * np.sin(2 * np.pi * day / days_per_year)
        test_paths[i] = test_path
    
    # Calculate test distribution statistics
    test_means = np.mean(test_paths, axis=0)
    test_stds = np.std(test_paths, axis=0)
    
    # Make predictions
    # Standard GP
    X_test_std = np.arange(train_t[-1] + 1, train_t[-1] + prediction_horizon + 1).reshape(-1, 1)
    y_pred_std, y_std_std = gp_std.predict(X_test_std, return_std=True)
    
    # Functional GP
    X_test_func = []
    for i in range(prediction_horizon):
        t_new = train_t[-1] + i + 1
        year = t_new // days_per_year
        day = t_new % days_per_year
        X_test_func.append([year, day])
    X_test_func = np.array(X_test_func)
    y_pred_func, y_std_func = gp_func.predict(X_test_func, return_std=True)
    
    # Augmented GP
    X_test_aug = []
    for delta in range(1, prediction_horizon + 1):
        X_test_aug.append([years-1, train_t[-1] % days_per_year, train_series[-1], delta])
    X_test_aug = np.array(X_test_aug)
    y_pred_aug, y_std_aug = gp_aug.predict(X_test_aug, return_std=True)
    
    # Functional-Augmented GP
    X_test_func_aug = []
    for delta in range(1, prediction_horizon + 1):
        X_test_func_aug.append([years-1, train_t[-1] % days_per_year, train_series[-1], delta])
    X_test_func_aug = np.array(X_test_func_aug)
    y_pred_func_aug, y_std_func_aug = gp_func_aug.predict(X_test_func_aug, return_std=True)
    
    # AR(1) predictions
    y_pred_ar = predict_ar1(train_series[-1], phi, c, prediction_horizon)
    
    # Plot training data and predictions
    plt.figure(figsize=(15, 10))
    
    # Plot training data
    plt.subplot(2, 1, 1)
    plt.plot(train_t, train_series, 'k-', label='Training Data')
    
    # Plot test paths (just a few for visualization)
    for i in range(min(10, n_test_simulations)):
        plt.plot(np.arange(train_t[-1] + 1, train_t[-1] + prediction_horizon + 1), 
                 test_paths[i], 'g-', alpha=0.2)
    
    # Plot mean of test paths
    plt.plot(np.arange(train_t[-1] + 1, train_t[-1] + prediction_horizon + 1), 
             test_means, 'g-', label='Test Mean')
    
    # Plot predictions
    plt.subplot(2, 1, 2)
    plt.plot(np.arange(train_t[-1] - 50, train_t[-1] + 1), 
             train_series[-51:], 'k-', label='Training Data')
    
    # Plot test mean
    plt.plot(np.arange(train_t[-1] + 1, train_t[-1] + prediction_horizon + 1), 
             test_means, 'g-', label='Test Mean')
    
    # Plot model predictions
    plt.plot(np.arange(train_t[-1] + 1, train_t[-1] + prediction_horizon + 1), 
             y_pred_std, 'b-', label='Standard GP')
    plt.fill_between(np.arange(train_t[-1] + 1, train_t[-1] + prediction_horizon + 1),
                     y_pred_std - 2 * y_std_std, 
                     y_pred_std + 2 * y_std_std, 
                     alpha=0.1, color='b')
    
    plt.plot(np.arange(train_t[-1] + 1, train_t[-1] + prediction_horizon + 1), 
             y_pred_func, 'r-', label='Functional GP')
    plt.fill_between(np.arange(train_t[-1] + 1, train_t[-1] + prediction_horizon + 1),
                     y_pred_func - 2 * y_std_func, 
                     y_pred_func + 2 * y_std_func, 
                     alpha=0.1, color='r')
    
    plt.plot(np.arange(train_t[-1] + 1, train_t[-1] + prediction_horizon + 1), 
             y_pred_aug, 'c-', label='Augmented GP')
    plt.fill_between(np.arange(train_t[-1] + 1, train_t[-1] + prediction_horizon + 1),
                     y_pred_aug - 2 * y_std_aug, 
                     y_pred_aug + 2 * y_std_aug, 
                     alpha=0.1, color='c')
    
    plt.plot(np.arange(train_t[-1] + 1, train_t[-1] + prediction_horizon + 1), 
             y_pred_func_aug, 'm-', label='Functional-Augmented GP')
    plt.fill_between(np.arange(train_t[-1] + 1, train_t[-1] + prediction_horizon + 1),
                     y_pred_func_aug - 2 * y_std_func_aug, 
                     y_pred_func_aug + 2 * y_std_func_aug, 
                     alpha=0.1, color='m')
    
    plt.plot(np.arange(train_t[-1] + 1, train_t[-1] + prediction_horizon + 1), 
             y_pred_ar, 'y-', label='AR(1)')
    
    plt.legend()
    plt.title(f'Predictions with Noise Level σ = {noise_level}')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

# Run experiment with different noise levels
noise_levels = [0.1, 0.2, 0.4, 0.8, 1.0]
results = run_experiment(noise_levels)

# Display results
for horizon in results:
    print(f"\nResults for {horizon}-day prediction horizon:")
    for noise in results[horizon]:
        print(f"\nNoise level σ = {noise}:")
        for model, mse in results[horizon][noise].items():
            if 'Std Err' not in model:  # Only print mean function MSE
                print(f"{model}: MSE = {mse:.6f}")

# Visualize predictions
visualize_predictions(noise_level=0.4, prediction_horizon=30)

# Kernel experiment function
def run_kernel_experiment(noise_level=1.0, prediction_horizons=[10, 20, 30]):
    """
    Runs experiment to test different kernels as described in the paper.
    
    Parameters:
    - noise_level: noise level (sigma) to test
    - prediction_horizons: list of prediction horizons to evaluate
    
    Returns:
    - results: dictionary containing MSE results for each model and kernel
    """
    # Simulation parameters
    years = 10
    days_per_year = 252
    alpha = 0.1
    mu = 0.0
    season_amplitude = 1.0
    max_delta = 50
    n_test_simulations = 1000
    
    results = {horizon: {} for horizon in prediction_horizons}
    
    # Create training data
    train_series, train_t = create_structured_time_series(years, days_per_year, alpha, mu, noise_level, season_amplitude)
    
    # Prepare data representations
    X_std, y_std = train_t.reshape(-1, 1), train_series
    X_func, y_func = prepare_functional_data(train_series, train_t, days_per_year)
    X_aug, y_aug = prepare_augmented_data(train_series, train_t, days_per_year, max_delta)
    X_func_aug, y_func_aug = prepare_functional_augmented_data(train_series, train_t, days_per_year, max_delta)
    
    # Subsample
    n_samples = 2000
    if len(X_aug) > n_samples:
        idx_aug = np.random.choice(len(X_aug), n_samples, replace=False)
        X_aug, y_aug = X_aug[idx_aug], y_aug[idx_aug]
    
    if len(X_func_aug) > n_samples:
        idx_func_aug = np.random.choice(len(X_func_aug), n_samples, replace=False)
        X_func_aug, y_func_aug = X_func_aug[idx_func_aug], y_func_aug[idx_func_aug]
    
    # Define kernels to test
    kernels = {
        'RationalQuadratic': RationalQuadratic(length_scale=1.0, alpha=1.0),
        'RBF': RBF(length_scale=1.0),
        'Matern32': Matern(length_scale=1.0, nu=1.5),
        'OU': Matern(length_scale=1.0, nu=0.5)  # Matern with nu=0.5 is equivalent to OU
    }
    
    # Fit AR(1) model
    phi, c, ar_sigma = fit_ar1(train_series)
    
    # Create test scenarios
    max_horizon = max(prediction_horizons)
    
    # Generate multiple test paths
    test_paths = np.zeros((n_test_simulations, max_horizon))
    for i in range(n_test_simulations):
        test_path = simulate_ou_process(alpha, mu, noise_level, 1, max_horizon, train_series[-1])
        # Add seasonal component
        for j in range(max_horizon):
            day = (train_t[-1] + j + 1) % days_per_year
            test_path[j] += season_amplitude * np.sin(2 * np.pi * day / days_per_year)
        test_paths[i] = test_path
    
    # Calculate test distribution statistics
    test_means = np.mean(test_paths, axis=0)
    
    # AR(1) predictions
    y_pred_ar = predict_ar1(train_series[-1], phi, c, max_horizon)
    
    for kernel_name, kernel in kernels.items():
        print(f"Testing kernel: {kernel_name}")
        
        # Fit models with current kernel
        gp_std = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5)
        gp_func = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5)
        gp_aug = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5)
        gp_func_aug = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5)
        
        gp_std.fit(X_std, y_std)
        gp_func.fit(X_func, y_func)
        gp_aug.fit(X_aug, y_aug)
        gp_func_aug.fit(X_func_aug, y_func_aug)
        
        # Make predictions
        # Standard GP
        X_test_std = np.arange(train_t[-1] + 1, train_t[-1] + max_horizon + 1).reshape(-1, 1)
        y_pred_std, _ = gp_std.predict(X_test_std, return_std=True)
        
        # Functional GP
        X_test_func = []
        for i in range(max_horizon):
            t_new = train_t[-1] + i + 1
            year = t_new // days_per_year
            day = t_new % days_per_year
            X_test_func.append([year, day])
        X_test_func = np.array(X_test_func)
        y_pred_func, _ = gp_func.predict(X_test_func, return_std=True)
        
        # Augmented GP
        X_test_aug = []
        for delta in range(1, max_horizon + 1):
            X_test_aug.append([years-1, train_t[-1] % days_per_year, train_series[-1], delta])
        X_test_aug = np.array(X_test_aug)
        y_pred_aug, _ = gp_aug.predict(X_test_aug, return_std=True)
        
        # Functional-Augmented GP
        X_test_func_aug = []
        for delta in range(1, max_horizon + 1):
            X_test_func_aug.append([years-1, train_t[-1] % days_per_year, train_series[-1], delta])
        X_test_func_aug = np.array(X_test_func_aug)
        y_pred_func_aug, _ = gp_func_aug.predict(X_test_func_aug, return_std=True)
        
        # Evaluate for each prediction horizon
        for horizon in prediction_horizons:
            # Calculate MSE for mean predictions
            mse_std = calculate_mse(y_pred_std[:horizon], test_means[:horizon])
            mse_func = calculate_mse(y_pred_func[:horizon], test_means[:horizon])
            mse_aug = calculate_mse(y_pred_aug[:horizon], test_means[:horizon])
            mse_func_aug = calculate_mse(y_pred_func_aug[:horizon], test_means[:horizon])
            mse_ar = calculate_mse(y_pred_ar[:horizon], test_means[:horizon])
            
            # Store results
            results[horizon][kernel_name] = {
                'Standard GP': mse_std,
                'Functional GP': mse_func,
                'Augmented GP': mse_aug,
                'Functional-Augmented GP': mse_func_aug,
                'AR(1)': mse_ar
            }
    
    return results

# Run kernel experiment
kernel_results = run_kernel_experiment()

# Display kernel experiment results
for horizon in kernel_results:
    print(f"\nResults for {horizon}-day prediction horizon:")
    for kernel_name in kernel_results[horizon]:
        print(f"\nKernel: {kernel_name}")
        for model, mse in kernel_results[horizon][kernel_name].items():
            print(f"{model}: MSE = {mse:.6f}")

# Fat tails experiment function
def run_fat_tails_experiment(degrees_freedom=[1000, 100, 50, 20, 15, 5, 3, 2], prediction_horizons=[10, 20, 30]):
    """
    Runs experiment with fat-tailed noise as described in the paper.
    
    Parameters:
    - degrees_freedom: list of degrees of freedom for t-distribution
    - prediction_horizons: list of prediction horizons to evaluate
    
    Returns:
    - results: dictionary containing MSE results for each model and degrees of freedom
    """
    # Simulation parameters
    years = 10
    days_per_year = 252
    alpha = 0.1
    mu = 0.0
    season_amplitude = 1.0
    max_delta = 50
    sigma = 0.38  # Fixed noise level as in the paper
    n_test_simulations = 1000
    
    results = {horizon: {df: {} for df in degrees_freedom} for horizon in prediction_horizons}
    
    for df in degrees_freedom:
        print(f"Running experiment with degrees of freedom = {df}")
        
        # Create training data with t-distributed noise
        total_days = years * days_per_year
        t = np.arange(total_days)
        
        # Seasonal component
        seasonal = season_amplitude * np.sin(2 * np.pi * t / days_per_year)
        
        # OU process with t-distributed noise
        ou = np.zeros(total_days)
        ou[0] = mu
        
        for i in range(1, total_days):
            # Using t-distributed noise instead of Gaussian
            noise = np.random.standard_t(df) * sigma / np.sqrt(df / (df - 2)) if df > 2 else np.random.standard_t(df) * sigma
            ou[i] = ou[i-1] * np.exp(-alpha) + mu * (1 - np.exp(-alpha)) + noise
        
        # Combined time series
        train_series = seasonal + ou
        train_t = t
        
        # Prepare data representations
        X_std, y_std = train_t.reshape(-1, 1), train_series
        X_func, y_func = prepare_functional_data(train_series, train_t, days_per_year)
        X_aug, y_aug = prepare_augmented_data(train_series, train_t, days_per_year, max_delta)
        X_func_aug, y_func_aug = prepare_functional_augmented_data(train_series, train_t, days_per_year, max_delta)
        
        # Subsample
        n_samples = 2000
        if len(X_aug) > n_samples:
            idx_aug = np.random.choice(len(X_aug), n_samples, replace=False)
            X_aug, y_aug = X_aug[idx_aug], y_aug[idx_aug]
        
        if len(X_func_aug) > n_samples:
            idx_func_aug = np.random.choice(len(X_func_aug), n_samples, replace=False)
            X_func_aug, y_func_aug = X_func_aug[idx_func_aug], y_func_aug[idx_func_aug]
        
        # Define kernel (RationalQuadratic as in the paper)
        kernel = RationalQuadratic(length_scale=1.0, alpha=1.0)
        
        # Fit models
        gp_std = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5)
        gp_func = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5)
        gp_aug = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5)
        gp_func_aug = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5)
        
        gp_std.fit(X_std, y_std)
        gp_func.fit(X_func, y_func)
        gp_aug.fit(X_aug, y_aug)
        gp_func_aug.fit(X_func_aug, y_func_aug)
        
        # Fit AR(1) model
        phi, c, ar_sigma = fit_ar1(train_series)
        
        # Create test scenarios
        max_horizon = max(prediction_horizons)
        
        # Generate multiple test paths with t-distributed noise
        test_paths = np.zeros((n_test_simulations, max_horizon))
        for i in range(n_test_simulations):
            test_ou = np.zeros(max_horizon)
            test_ou[0] = ou[-1]
            
            for j in range(1, max_horizon):
                # Using t-distributed noise
                noise = np.random.standard_t(df) * sigma / np.sqrt(df / (df - 2)) if df > 2 else np.random.standard_t(df) * sigma
                test_ou[j] = test_ou[j-1] * np.exp(-alpha) + mu * (1 - np.exp(-alpha)) + noise
            
            # Add seasonal component
            for j in range(max_horizon):
                day = (train_t[-1] + j + 1) % days_per_year
                test_ou[j] += season_amplitude * np.sin(2 * np.pi * day / days_per_year)
            
            test_paths[i] = test_ou
        
        # Calculate test distribution statistics
        test_means = np.mean(test_paths, axis=0)
        
        # Make predictions
        # Standard GP
        X_test_std = np.arange(train_t[-1] + 1, train_t[-1] + max_horizon + 1).reshape(-1, 1)
        y_pred_std, _ = gp_std.predict(X_test_std, return_std=True)
        
        # Functional GP
        X_test_func = []
        for i in range(max_horizon):
            t_new = train_t[-1] + i + 1
            year = t_new // days_per_year
            day = t_new % days_per_year
            X_test_func.append([year, day])
        X_test_func = np.array(X_test_func)
        y_pred_func, _ = gp_func.predict(X_test_func, return_std=True)
        
        # Augmented GP
        X_test_aug = []
        for delta in range(1, max_horizon + 1):
            X_test_aug.append([years-1, train_t[-1] % days_per_year, train_series[-1], delta])
        X_test_aug = np.array(X_test_aug)
        y_pred_aug, _ = gp_aug.predict(X_test_aug, return_std=True)
        
        # Functional-Augmented GP
        X_test_func_aug = []
        for delta in range(1, max_horizon + 1):
            X_test_func_aug.append([years-1, train_t[-1] % days_per_year, train_series[-1], delta])
        X_test_func_aug = np.array(X_test_func_aug)
        y_pred_func_aug, _ = gp_func_aug.predict(X_test_func_aug, return_std=True)
        
        # AR(1) predictions
        y_pred_ar = predict_ar1(train_series[-1], phi, c, max_horizon)
        
        # Evaluate for each prediction horizon
        for horizon in prediction_horizons:
            # Calculate MSE for mean predictions
            mse_std = calculate_mse(y_pred_std[:horizon], test_means[:horizon])
            mse_func = calculate_mse(y_pred_func[:horizon], test_means[:horizon])
            mse_aug = calculate_mse(y_pred_aug[:horizon], test_means[:horizon])
            mse_func_aug = calculate_mse(y_pred_func_aug[:horizon], test_means[:horizon])
            mse_ar = calculate_mse(y_pred_ar[:horizon], test_means[:horizon])
            
            # Store results
            results[horizon][df] = {
                'Standard GP': mse_std,
                'Functional GP': mse_func,
                'Augmented GP': mse_aug,
                'Functional-Augmented GP': mse_func_aug,
                'AR(1)': mse_ar
            }
    
    return results

# Run fat tails experiment
fat_tails_results = run_fat_tails_experiment()

# Display fat tails experiment results
for horizon in fat_tails_results:
    print(f"\nResults for {horizon}-day prediction horizon:")
    for df in fat_tails_results[horizon]:
        print(f"\nDegrees of Freedom: {df}")
        for model, mse in fat_tails_results[horizon][df].items():
            print(f"{model}: MSE = {mse:.6f}")

# Function to demonstrate trading strategy using GPs
def simulate_trading_strategy(initial_capital=1000000, transaction_cost=0.0005, prediction_horizon=10):
    """
    Demonstrates a trading strategy using GPs as described in the paper.
    
    Parameters:
    - initial_capital: initial capital for trading
    - transaction_cost: transaction cost as a fraction of trade value
    - prediction_horizon: prediction horizon in days
    
    Returns:
    - equity_curve: equity curve of the trading strategy
    """
    # Simulation parameters
    years = 10
    days_per_year = 252
    alpha = 0.1
    mu = 0.0
    sigma = 0.4
    season_amplitude = 1.0
    max_delta = 50
    
    # Create time series
    series, t = create_structured_time_series(years, days_per_year, alpha, mu, sigma, season_amplitude)
    
    # Prepare for trading simulation
    capital = initial_capital
    position = 0  # 0 = flat, 1 = long, -1 = short
    equity_curve = [capital]
    positions = [position]
    
    # Use functional-augmented GP for trading
    kernel = RationalQuadratic(length_scale=1.0, alpha=1.0)
    
    # Trading loop
    for i in range(days_per_year * 2, len(series) - prediction_horizon):
        # Use a rolling window of 2 years of data for training
        train_start = i - days_per_year * 2
        train_series = series[train_start:i]
        train_t = t[train_start:i]
        
        # Prepare data representations for functional-augmented GP
        X_func_aug, y_func_aug = prepare_functional_augmented_data(
            train_series, train_t, days_per_year, max_delta)
        
        # Subsample if needed
        n_samples = 2000
        if len(X_func_aug) > n_samples:
            idx_func_aug = np.random.choice(len(X_func_aug), n_samples, replace=False)
            X_func_aug, y_func_aug = X_func_aug[idx_func_aug], y_func_aug[idx_func_aug]
        
        # Fit model
        gp_func_aug = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2)
        gp_func_aug.fit(X_func_aug, y_func_aug)
        
        # Make prediction
        X_test_func_aug = []
        for delta in range(1, prediction_horizon + 1):
            current_year = t[i] // days_per_year
            current_day = t[i] % days_per_year
            X_test_func_aug.append([current_year, current_day, series[i], delta])
        X_test_func_aug = np.array(X_test_func_aug)
        
        y_pred, y_std = gp_func_aug.predict(X_test_func_aug, return_std=True)
        
        # Calculate expected return and Sharpe ratio
        expected_return = y_pred[-1] - series[i]
        risk = y_std[-1]
        
        # Sharpe ratio adjusted for transaction costs
        tc_adjusted_return = expected_return
        if position != 1 and expected_return > 0:
            tc_adjusted_return -= series[i] * transaction_cost
        elif position != -1 and expected_return < 0:
            tc_adjusted_return -= series[i] * transaction_cost
        
        sharpe = tc_adjusted_return / risk if risk > 0 else 0
        
        # Trading decision based on Sharpe ratio
        new_position = 0
        if sharpe > 1.0:  # Go long if Sharpe > 1
            new_position = 1
        elif sharpe < -1.0:  # Go short if Sharpe < -1
            new_position = -1
        
        # Execute trade if position changes
        if new_position != position:
            # Close old position
            if position == 1:
                capital += series[i] - series[i] * transaction_cost
            elif position == -1:
                capital -= series[i] + series[i] * transaction_cost
            
            # Open new position
            if new_position == 1:
                capital -= series[i] + series[i] * transaction_cost
            elif new_position == -1:
                capital += series[i] - series[i] * transaction_cost
            
            position = new_position
        
        # Update equity based on position
        current_equity = capital
        if position == 1:
            current_equity += series[i]
        elif position == -1:
            current_equity -= series[i]
        
        equity_curve.append(current_equity)
        positions.append(position)
        
        # Print progress
        if i % 100 == 0:
            print(f"Day {i}/{len(series)}, Equity: {current_equity:.2f}")
    
    # Plot equity curve
    plt.figure(figsize=(12, 6))
    plt.plot(equity_curve)
    plt.title('Equity Curve')
    plt.xlabel('Days')
    plt.ylabel('Equity')
    plt.grid(True)
    plt.show()
    
    # Calculate trading statistics
    returns = np.diff(equity_curve) / np.array(equity_curve)[:-1]
    sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)  # Annualized Sharpe
    max_drawdown = np.max(np.maximum.accumulate(equity_curve) - equity_curve) / np.max(equity_curve)
    
    print(f"Final Equity: {equity_curve[-1]:.2f}")
    print(f"Return: {(equity_curve[-1] - initial_capital) / initial_capital * 100:.2f}%")
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    print(f"Max Drawdown: {max_drawdown * 100:.2f}%")
    
    return equity_curve, positions

# Simulate trading strategy
equity_curve, positions = simulate_trading_strategy()