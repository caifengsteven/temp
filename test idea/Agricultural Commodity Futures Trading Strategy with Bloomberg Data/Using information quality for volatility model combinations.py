import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
import statsmodels.api as sm
from pykalman import KalmanFilter
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

# Set random seed for reproducibility
np.random.seed(42)

class VolatilityModelCombination:
    """
    Implementation of the paper 'Using information quality for volatility model combinations' by Golosnoy and Okhrin
    
    This class implements the adaptive model combination methodology that accounts for information quality
    """
    
    def __init__(self, memory_parameter=0.95, threshold=0.0, strategy="winner"):
        """
        Initialize the volatility model combination framework
        
        Parameters:
        -----------
        memory_parameter : float
            Parameter φ₁ in the paper, controls the memory of the weight update (0-1)
        threshold : float
            Parameter α in the paper, minimum outperformance threshold
        strategy : str
            Either "winner" for winner-takes-all or "proportional" for proportional weighting
        """
        self.memory = memory_parameter
        self.threshold = threshold
        self.strategy = strategy
        
        # Initialize weights and models
        self.weights = None
        self.models = {}
        self.model_names = []
        self.benchmark_model = None
        
        # State space model parameters
        self.ss_params = None
        
    def add_model(self, name, model_func, is_benchmark=False):
        """
        Add a volatility forecasting model
        
        Parameters:
        -----------
        name : str
            Name of the model
        model_func : callable
            Function that produces volatility forecasts
        is_benchmark : bool
            Whether this model is the benchmark model
        """
        self.models[name] = model_func
        self.model_names.append(name)
        
        if is_benchmark:
            self.benchmark_model = name
            
        # Initialize weights equally
        n_models = len(self.model_names)
        self.weights = np.ones(n_models) / n_models
            
    def estimate_state_space_model(self, log_rv, volume):
        """
        Estimate the state space model for information flow
        
        Parameters:
        -----------
        log_rv : array-like
            Log of realized volatility
        volume : array-like
            Detrended trading volume
        """
        # Standardize the data
        log_rv_std = (log_rv - np.mean(log_rv)) / np.std(log_rv)
        volume_std = (volume - np.mean(volume)) / np.std(volume)
        
        # Setup Kalman filter for the state space model
        kf = KalmanFilter(
            transition_matrices=[[self.memory]],
            observation_matrices=[[1.0], [1.0]],
            initial_state_mean=[0],
            initial_state_covariance=[[1.0]],
            observation_covariance=[[0.1], [0.1]],
            transition_covariance=[[0.01]]
        )
        
        # Stack observations
        observations = np.column_stack([log_rv_std, volume_std])
        
        # Fit the model
        kf = kf.em(observations, n_iter=10)
        
        # Save parameters
        self.ss_params = {
            'kf': kf
        }
        
        return kf
        
    def _calculate_information_quality(self, log_rv, volume):
        """
        Calculate the information quality measure λt
        
        Parameters:
        -----------
        log_rv : array-like
            Log of realized volatility
        volume : array-like
            Detrended trading volume
            
        Returns:
        --------
        lambda_t : float
            Information quality measure λt in (0,1]
        """
        if self.ss_params is None:
            # Estimate the state space model if not already done
            self.estimate_state_space_model(log_rv, volume)
        
        # Standardize the data
        log_rv_std = (log_rv - np.mean(log_rv)) / np.std(log_rv)
        volume_std = (volume - np.mean(volume)) / np.std(volume)
        
        # Stack observations for the last point
        last_obs = np.array([[log_rv_std[-1], volume_std[-1]]])
        
        # Get the Kalman filter
        kf = self.ss_params['kf']
        
        # Get filtered states
        filtered_state, _ = kf.filter(last_obs)
        
        # Calculate innovation
        if len(filtered_state) > 1:
            innovation = filtered_state[-1, 0] - (self.memory * filtered_state[-2, 0])
            
            # Estimate the standard deviation of innovations
            sigma_innovation = np.std(np.diff(filtered_state[:, 0]))
            
            # Calculate λt using equation (8) from the paper
            lambda_t = 2 - 2 * norm.cdf(abs(innovation) / sigma_innovation)
            lambda_t = max(0.01, min(lambda_t, 1.0))  # Ensure value is in (0,1]
            return lambda_t
        else:
            return 1.0  # Default to full confidence for the first observation
        
    def _calculate_model_performance(self, forecasts, realized, realized_var=None):
        """
        Calculate the momentum performance of each model
        
        Parameters:
        -----------
        forecasts : dict
            Dictionary of model forecasts
        realized : float
            Realized volatility value
        realized_var : float, optional
            Variance of the realized volatility estimator
            
        Returns:
        --------
        z_vals : dict
            Performance measure for each model
        """
        # If variance not provided, use a default value
        if realized_var is None:
            realized_var = 0.01 * realized**2
            
        # Calculate the standardized forecast errors for each model
        z_vals = {}
        for model_name in self.model_names:
            forecast = forecasts[model_name]
            # Calculate z using equation (9) from the paper
            z = norm.cdf(abs(realized - forecast) / np.sqrt(realized_var))
            z_vals[model_name] = z
            
        return z_vals
        
    def _calculate_momentum_weights(self, z_vals):
        """
        Calculate the momentum weights based on model performance
        
        Parameters:
        -----------
        z_vals : dict
            Performance measure for each model
            
        Returns:
        --------
        momentum_weights : ndarray
            Momentum weights for each model
        """
        b_idx = self.model_names.index(self.benchmark_model)
        benchmark_z = z_vals[self.benchmark_model]
        
        # Initialize indicators and weights
        indicators = np.zeros(len(self.model_names))
        weights = np.zeros(len(self.model_names))
        
        # Calculate performance relative to benchmark
        for i, model_name in enumerate(self.model_names):
            model_z = z_vals[model_name]
            # Outperformance indicator
            indicators[i] = 1 if benchmark_z - model_z > self.threshold else 0
        
        # Handle the case where no model outperforms the benchmark
        if np.sum(indicators) == 0:
            weights[b_idx] = 1.0
            return weights
            
        if self.strategy == "winner":
            # Winner-takes-all strategy as in equation (10)
            if np.sum(indicators) > 0:
                # Find the best model
                non_bench_models = [i for i in range(len(self.model_names)) if i != b_idx]
                perf_diff = [z_vals[self.benchmark_model] - z_vals[model] for model in [self.model_names[i] for i in non_bench_models]]
                
                # Only consider models that outperform the benchmark by the threshold
                outperforming = [i for i, indicator in enumerate(indicators) if indicator == 1]
                if len(outperforming) > 0:
                    # Find the model with the best performance
                    z_values = [z_vals[self.model_names[i]] for i in outperforming]
                    best_idx = outperforming[np.argmin(z_values)]
                    weights[best_idx] = 1.0
                else:
                    weights[b_idx] = 1.0
            else:
                weights[b_idx] = 1.0
                
        elif self.strategy == "proportional":
            # Proportional strategy as in equation (11)
            excess_performance = np.zeros(len(self.model_names))
            
            for i, model_name in enumerate(self.model_names):
                if model_name != self.benchmark_model and indicators[i] == 1:
                    # Calculate outperformance
                    excess_performance[i] = benchmark_z - z_vals[model_name]
            
            # Total excess performance
            total_excess = np.sum(excess_performance)
            
            if total_excess == 0:
                # If no model outperforms benchmark
                weights[b_idx] = 1.0
            elif total_excess < 0.5:
                # If total outperformance is small
                for i in range(len(self.model_names)):
                    if i != b_idx:
                        weights[i] = 2 * excess_performance[i]
                weights[b_idx] = 1 - np.sum(weights)
            else:
                # If total outperformance is large
                for i in range(len(self.model_names)):
                    if excess_performance[i] > 0:
                        weights[i] = excess_performance[i] / total_excess
        
        return weights
    
    def update_weights(self, forecasts, realized_vol, realized_var, log_rv, volume):
        """
        Update the model weights based on performance
        
        Parameters:
        -----------
        forecasts : dict
            Dictionary of model forecasts
        realized_vol : float
            Realized volatility value
        realized_var : float
            Variance of the realized volatility estimator
        log_rv : array-like
            Log of realized volatility time series
        volume : array-like
            Detrended volume time series
            
        Returns:
        --------
        weights : ndarray
            Updated model weights
        lambda_t : float
            Information quality measure
        """
        # Calculate model performance
        z_vals = self._calculate_model_performance(forecasts, realized_vol, realized_var)
        
        # Calculate information quality
        lambda_t = self._calculate_information_quality(log_rv, volume)
        
        # Calculate momentum weights
        f_t = self._calculate_momentum_weights(z_vals)
        
        # Initialize unconditional weights (prior beliefs)
        f_0 = np.zeros(len(self.model_names))
        b_idx = self.model_names.index(self.benchmark_model)
        f_0[b_idx] = 1.0  # Unconditional weight is 1 for benchmark
        
        # Calculate information-adjusted weights using equation (6)
        p_t = (1 - lambda_t) * f_0 + lambda_t * f_t
        
        # Update weights using equation (5) - exponential smoothing
        self.weights = self.memory * self.weights + (1 - self.memory) * p_t
        
        return self.weights, lambda_t
        
    def forecast_combination(self, forecasts):
        """
        Combine the individual model forecasts
        
        Parameters:
        -----------
        forecasts : dict
            Dictionary of model forecasts
            
        Returns:
        --------
        combined_forecast : float
            Combined volatility forecast
        """
        combined_forecast = 0
        for i, model_name in enumerate(self.model_names):
            combined_forecast += self.weights[i] * forecasts[model_name]
            
        return combined_forecast


# Data Simulation functions
def simulate_returns(n_days=1000, mu=0, sigma_base=0.01, jump_prob=0.05, jump_size=0.03):
    """
    Simulate daily returns with occasional jumps
    """
    # Base returns (normal distribution)
    returns = np.random.normal(mu, sigma_base, n_days)
    
    # Add jumps
    jumps = np.random.binomial(1, jump_prob, n_days)
    jump_direction = np.random.choice([-1, 1], n_days)
    jump_magnitude = np.random.exponential(jump_size, n_days)
    
    returns += jumps * jump_direction * jump_magnitude
    
    return returns

def simulate_volume(n_days=1000, base=1000, trend=0.1, ar_coef=0.7):
    """
    Simulate trading volume with trend and autocorrelation
    """
    # Generate AR(1) process
    noise = np.random.normal(0, 0.3, n_days)
    volume = np.zeros(n_days)
    
    for i in range(n_days):
        if i == 0:
            volume[i] = noise[i]
        else:
            volume[i] = ar_coef * volume[i-1] + noise[i]
    
    # Add trend and scale
    volume = base + np.arange(n_days) * trend + volume * 100
    
    # Ensure non-negative
    volume = np.maximum(0, volume)
    
    return volume

def simulate_realized_volatility(returns, n_intraday=100, noise_level=0.1):
    """
    Simulate realized volatility based on returns
    """
    n_days = len(returns)
    
    # True daily volatility (absolute returns)
    true_vol = np.abs(returns)
    
    # Add noise to realized volatility (measurement error)
    realized_vol = true_vol * (1 + np.random.normal(0, noise_level, n_days))
    
    # Ensure non-negative
    realized_vol = np.maximum(0.0001, realized_vol)
    
    return realized_vol

def detrend_volume(volume):
    """
    Remove linear trend from volume
    """
    # Fit linear trend
    x = np.arange(len(volume))
    trend = sm.OLS(volume, sm.add_constant(x)).fit()
    
    # Remove trend
    detrended = volume - trend.predict(sm.add_constant(x))
    
    return detrended


# Volatility Model Implementations
def garch_model(returns, window=500):
    """
    GARCH(1,1) model for volatility forecasting
    """
    # Take a window of the most recent returns
    recent_returns = returns[-window:]
    
    # Fit GARCH model
    model = arch_model(recent_returns, vol='Garch', p=1, q=1)
    results = model.fit(disp='off')
    
    # Forecast volatility
    forecast = results.forecast(horizon=1)
    
    return np.sqrt(forecast.variance.iloc[-1, 0])

def riskmetrics_model(returns, window=500, decay=0.94):
    """
    RiskMetrics model (exponentially weighted moving average)
    """
    # Take a window of the most recent returns
    recent_returns = returns[-window:]
    squared_returns = recent_returns**2
    
    # Calculate exponentially weighted average
    weights = np.power(decay, np.arange(len(squared_returns)-1, -1, -1))
    weights = weights / np.sum(weights)
    
    # Calculate weighted average
    volatility = np.sqrt(np.sum(weights * squared_returns))
    
    return volatility

def sample_model(realized_vol, window=25):
    """
    Simple moving average of realized volatility
    """
    if len(realized_vol) < window:
        return np.mean(realized_vol)
    
    return np.mean(realized_vol[-window:])

def kernel_model(realized_vol, window=100):
    """
    Non-parametric kernel estimator
    """
    if len(realized_vol) < 5:
        return np.mean(realized_vol)
    
    # Log transform
    log_rv = np.log(realized_vol[-window:])
    
    # Simple nadaraya-watson kernel estimator with linear time trend
    x = np.arange(len(log_rv)).reshape(-1, 1)
    
    # Use sklearn's KernelReg if available, otherwise simple weighted average
    weights = np.exp(-0.5 * ((len(log_rv) - 1 - np.arange(len(log_rv))) / 10)**2)
    weights = weights / np.sum(weights)
    prediction = np.sum(weights * log_rv)
    
    # Transform back
    return np.exp(prediction)

def arfima_model(realized_vol, window=200):
    """
    ARFIMA model for long memory in log realized volatility
    
    Note: True ARFIMA requires specialized packages, here we use ARIMA as approximation
    """
    if len(realized_vol) < 10:
        return np.mean(realized_vol)
    
    # Log transform
    log_rv = np.log(realized_vol[-window:])
    
    # Fit ARIMA model (as proxy for ARFIMA)
    try:
        model = ARIMA(log_rv, order=(1, 1, 0))  # Using (1,d,0) with d=1 as in paper
        results = model.fit()
        forecast = results.forecast(steps=1)
        
        # Transform back
        return np.exp(forecast[0])
    except:
        # Fallback if ARIMA fails
        return np.exp(np.mean(log_rv))

def sv_model(realized_vol, volume, window=200):
    """
    Stochastic volatility model
    """
    if len(realized_vol) < 10:
        return np.mean(realized_vol)
    
    # Log transform
    log_rv = np.log(realized_vol[-window:])
    vol_std = (volume[-window:] - np.mean(volume[-window:])) / np.std(volume[-window:])
    
    # Simple state space model approximation
    try:
        kf = KalmanFilter(
            transition_matrices=[[0.97]],
            observation_matrices=[[1.0], [0.2]],
            initial_state_mean=[0],
            initial_state_covariance=[[1.0]],
            observation_covariance=[[0.1], [0.1]],
            transition_covariance=[[0.01]]
        )
        
        # Stack observations
        observations = np.column_stack([log_rv, vol_std])
        
        # Fit the model
        kf = kf.em(observations, n_iter=5)
        
        # Forecast
        state_means, _ = kf.filter(observations)
        next_state = kf.transition_matrices[0, 0] * state_means[-1, 0]
        prediction = next_state
        
        # Transform back
        return np.exp(prediction)
    except:
        # Fallback
        return np.exp(np.mean(log_rv))


# Main simulation and analysis
def run_simulation(n_days=1000, estimation_window=500, evaluation_window=500):
    """
    Simulate data and run the volatility model combination
    """
    # Simulate data
    returns = simulate_returns(n_days)
    volume = simulate_volume(n_days)
    realized_vol = simulate_realized_volatility(returns)
    
    # Detrend volume
    detrended_volume = detrend_volume(volume)
    
    # Log of realized volatility
    log_rv = np.log(realized_vol)
    
    # Set up model combiner
    combiner_winner = VolatilityModelCombination(memory_parameter=0.95, threshold=0.0, strategy="winner")
    combiner_prop = VolatilityModelCombination(memory_parameter=0.95, threshold=0.0, strategy="proportional")
    
    # Add models
    combiner_winner.add_model("GARCH", lambda: garch_model(returns, window=estimation_window))
    combiner_winner.add_model("RiskMetrics", lambda: riskmetrics_model(returns, window=estimation_window))
    combiner_winner.add_model("Sample", lambda: sample_model(realized_vol, window=25))
    combiner_winner.add_model("Kernel", lambda: kernel_model(realized_vol, window=estimation_window))
    combiner_winner.add_model("SV", lambda: sv_model(realized_vol, detrended_volume, window=estimation_window))
    combiner_winner.add_model("ARFIMA", lambda: arfima_model(realized_vol, window=estimation_window), is_benchmark=True)
    
    # Same models for proportional combiner
    combiner_prop.add_model("GARCH", lambda: garch_model(returns, window=estimation_window))
    combiner_prop.add_model("RiskMetrics", lambda: riskmetrics_model(returns, window=estimation_window))
    combiner_prop.add_model("Sample", lambda: sample_model(realized_vol, window=25))
    combiner_prop.add_model("Kernel", lambda: kernel_model(realized_vol, window=estimation_window))
    combiner_prop.add_model("SV", lambda: sv_model(realized_vol, detrended_volume, window=estimation_window))
    combiner_prop.add_model("ARFIMA", lambda: arfima_model(realized_vol, window=estimation_window), is_benchmark=True)
    
    # Initialize arrays to store forecasts and performance
    garch_forecasts = np.zeros(evaluation_window)
    riskmetrics_forecasts = np.zeros(evaluation_window)
    sample_forecasts = np.zeros(evaluation_window)
    kernel_forecasts = np.zeros(evaluation_window)
    sv_forecasts = np.zeros(evaluation_window)
    arfima_forecasts = np.zeros(evaluation_window)
    winner_combined_forecasts = np.zeros(evaluation_window)
    prop_combined_forecasts = np.zeros(evaluation_window)
    
    winner_weights_history = np.zeros((evaluation_window, len(combiner_winner.model_names)))
    prop_weights_history = np.zeros((evaluation_window, len(combiner_prop.model_names)))
    
    lambda_history = np.zeros(evaluation_window)
    
    # Run the out-of-sample evaluation
    start_idx = n_days - evaluation_window
    
    for i in tqdm(range(evaluation_window)):
        # Current index in the full dataset
        idx = start_idx + i
        
        # Calculate individual model forecasts
        garch_forecast = garch_model(returns[:idx], window=min(estimation_window, idx))
        riskmetrics_forecast = riskmetrics_model(returns[:idx], window=min(estimation_window, idx))
        sample_forecast = sample_model(realized_vol[:idx], window=min(25, idx))
        kernel_forecast = kernel_model(realized_vol[:idx], window=min(estimation_window, idx))
        sv_forecast = sv_model(realized_vol[:idx], detrended_volume[:idx], window=min(estimation_window, idx))
        arfima_forecast = arfima_model(realized_vol[:idx], window=min(estimation_window, idx))
        
        # Store forecasts
        garch_forecasts[i] = garch_forecast
        riskmetrics_forecasts[i] = riskmetrics_forecast
        sample_forecasts[i] = sample_forecast
        kernel_forecasts[i] = kernel_forecast
        sv_forecasts[i] = sv_forecast
        arfima_forecasts[i] = arfima_forecast
        
        # Combine forecasts
        forecasts = {
            "GARCH": garch_forecast,
            "RiskMetrics": riskmetrics_forecast,
            "Sample": sample_forecast,
            "Kernel": kernel_forecast,
            "SV": sv_forecast,
            "ARFIMA": arfima_forecast
        }
        
        # If this isn't the first forecast, update weights based on previous forecast performance
        if i > 0:
            # Get the realized volatility for the previous period
            previous_rv = realized_vol[idx-1]
            previous_var = 0.01 * previous_rv**2  # Approximation of realized vol variance
            
            # Update weights for the winner combiner
            winner_weights, lambda_t = combiner_winner.update_weights(
                forecasts=forecasts,
                realized_vol=previous_rv,
                realized_var=previous_var,
                log_rv=log_rv[:idx],
                volume=detrended_volume[:idx]
            )
            
            # Update weights for the proportional combiner
            prop_weights, _ = combiner_prop.update_weights(
                forecasts=forecasts,
                realized_vol=previous_rv,
                realized_var=previous_var,
                log_rv=log_rv[:idx],
                volume=detrended_volume[:idx]
            )
            
            # Store lambda
            lambda_history[i] = lambda_t
        
        # Generate combined forecasts
        winner_combined_forecasts[i] = combiner_winner.forecast_combination(forecasts)
        prop_combined_forecasts[i] = combiner_prop.forecast_combination(forecasts)
        
        # Store weights history
        winner_weights_history[i] = combiner_winner.weights
        prop_weights_history[i] = combiner_prop.weights
    
    # Evaluate forecasts
    actual_vol = realized_vol[start_idx:start_idx+evaluation_window]
    
    results = {
        'GARCH': compute_metrics(actual_vol, garch_forecasts),
        'RiskMetrics': compute_metrics(actual_vol, riskmetrics_forecasts),
        'Sample': compute_metrics(actual_vol, sample_forecasts),
        'Kernel': compute_metrics(actual_vol, kernel_forecasts),
        'SV': compute_metrics(actual_vol, sv_forecasts),
        'ARFIMA': compute_metrics(actual_vol, arfima_forecasts),
        'Winner Combined': compute_metrics(actual_vol, winner_combined_forecasts),
        'Proportional Combined': compute_metrics(actual_vol, prop_combined_forecasts)
    }
    
    # Visualization
    plot_results(
        actual_vol, 
        garch_forecasts, 
        riskmetrics_forecasts, 
        sample_forecasts, 
        kernel_forecasts, 
        sv_forecasts, 
        arfima_forecasts, 
        winner_combined_forecasts, 
        prop_combined_forecasts,
        lambda_history,
        winner_weights_history,
        prop_weights_history,
        combiner_winner.model_names,
        results
    )
    
    return results, {
        'actual_vol': actual_vol,
        'garch_forecasts': garch_forecasts,
        'riskmetrics_forecasts': riskmetrics_forecasts,
        'sample_forecasts': sample_forecasts,
        'kernel_forecasts': kernel_forecasts,
        'sv_forecasts': sv_forecasts,
        'arfima_forecasts': arfima_forecasts,
        'winner_combined_forecasts': winner_combined_forecasts,
        'prop_combined_forecasts': prop_combined_forecasts,
        'lambda_history': lambda_history,
        'winner_weights_history': winner_weights_history,
        'prop_weights_history': prop_weights_history
    }

def compute_metrics(actual, forecast):
    """Compute performance metrics for volatility forecasts"""
    # Squared error
    mse = mean_squared_error(actual, forecast)
    
    # Absolute error
    mae = np.mean(np.abs(actual - forecast))
    
    # Heteroskedasticity-adjusted squared error
    adj_mse = np.mean(np.square(1 - forecast / actual))
    
    # Heteroskedasticity-adjusted absolute error
    adj_mae = np.mean(np.abs(1 - forecast / actual))
    
    return {
        'MSE': mse,
        'MAE': mae,
        'Adj MSE': adj_mse,
        'Adj MAE': adj_mae
    }

def plot_results(actual_vol, garch, riskmetrics, sample, kernel, sv, arfima, 
                winner_combined, prop_combined, lambda_history, 
                winner_weights_history, prop_weights_history, model_names,
                results):
    """
    Plot the volatility forecasts and model weights
    """
    # Set up figure with multiple subplots
    fig = plt.figure(figsize=(15, 18))
    
    # Plot volatility forecasts
    ax1 = fig.add_subplot(411)
    ax1.plot(actual_vol, 'k-', label='Actual Volatility')
    ax1.plot(garch, 'b-', alpha=0.5, label='GARCH')
    ax1.plot(riskmetrics, 'g-', alpha=0.5, label='RiskMetrics')
    ax1.plot(sample, 'c-', alpha=0.5, label='Sample')
    ax1.plot(kernel, 'm-', alpha=0.5, label='Kernel')
    ax1.plot(sv, 'y-', alpha=0.5, label='SV')
    ax1.plot(arfima, 'r-', alpha=0.5, label='ARFIMA')
    ax1.set_title('Volatility Forecasts')
    ax1.legend()
    ax1.grid(True)
    
    # Plot combined forecasts
    ax2 = fig.add_subplot(412)
    ax2.plot(actual_vol, 'k-', label='Actual Volatility')
    ax2.plot(winner_combined, 'b-', label='Winner Combined')
    ax2.plot(prop_combined, 'r-', label='Proportional Combined')
    ax2.set_title('Combined Volatility Forecasts')
    ax2.legend()
    ax2.grid(True)
    
    # Plot information quality (lambda)
    ax3 = fig.add_subplot(413)
    ax3.plot(lambda_history, 'b-')
    ax3.set_title('Information Quality Measure (λt)')
    ax3.set_ylim(0, 1.1)
    ax3.grid(True)
    
    # Plot model weights
    ax4 = fig.add_subplot(424)
    for i, model in enumerate(model_names):
        ax4.plot(winner_weights_history[:, i], label=model)
    ax4.set_title('Winner-Takes-All Weights')
    ax4.legend()
    ax4.grid(True)
    
    ax5 = fig.add_subplot(428)
    for i, model in enumerate(model_names):
        ax5.plot(prop_weights_history[:, i], label=model)
    ax5.set_title('Proportional Weights')
    ax5.legend()
    ax5.grid(True)
    
    # Add performance table
    performance = '\n'.join([
        f"Model Performance Metrics:",
        f"MSE (Lower is better):",
        f"GARCH: {results['GARCH']['MSE']:.6f}",
        f"RiskMetrics: {results['RiskMetrics']['MSE']:.6f}",
        f"Sample: {results['Sample']['MSE']:.6f}",
        f"Kernel: {results['Kernel']['MSE']:.6f}",
        f"SV: {results['SV']['MSE']:.6f}",
        f"ARFIMA: {results['ARFIMA']['MSE']:.6f}",
        f"Winner Combined: {results['Winner Combined']['MSE']:.6f}",
        f"Proportional Combined: {results['Proportional Combined']['MSE']:.6f}",
        f"\nMAE (Lower is better):",
        f"GARCH: {results['GARCH']['MAE']:.6f}",
        f"RiskMetrics: {results['RiskMetrics']['MAE']:.6f}",
        f"Sample: {results['Sample']['MAE']:.6f}",
        f"Kernel: {results['Kernel']['MAE']:.6f}",
        f"SV: {results['SV']['MAE']:.6f}",
        f"ARFIMA: {results['ARFIMA']['MAE']:.6f}",
        f"Winner Combined: {results['Winner Combined']['MAE']:.6f}",
        f"Proportional Combined: {results['Proportional Combined']['MAE']:.6f}"
    ])
    
    fig.text(0.5, 0.05, performance, ha='center', va='center', fontsize=10)
    
    plt.tight_layout(rect=[0, 0.1, 1, 0.95])
    plt.show()

# Run the simulation
if __name__ == "__main__":
    results, forecasts = run_simulation(n_days=1500, estimation_window=500, evaluation_window=500)
    
    # Print a summary of results
    print("\nPerformance Summary:")
    print("===================")
    
    # MSE comparison
    mse_values = {model: metrics['MSE'] for model, metrics in results.items()}
    best_mse_model = min(mse_values, key=mse_values.get)
    
    print(f"\nBest model by MSE: {best_mse_model}")
    for model, mse in sorted(mse_values.items(), key=lambda x: x[1]):
        print(f"{model}: {mse:.6f} ({(mse / mse_values[best_mse_model] - 1) * 100:.2f}% worse than best)")
    
    # MAE comparison
    mae_values = {model: metrics['MAE'] for model, metrics in results.items()}
    best_mae_model = min(mae_values, key=mae_values.get)
    
    print(f"\nBest model by MAE: {best_mae_model}")
    for model, mae in sorted(mae_values.items(), key=lambda x: x[1]):
        print(f"{model}: {mae:.6f} ({(mae / mae_values[best_mae_model] - 1) * 100:.2f}% worse than best)")