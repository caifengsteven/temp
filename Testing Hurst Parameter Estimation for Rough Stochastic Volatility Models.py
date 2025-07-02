import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tqdm import tqdm
from scipy.optimize import minimize_scalar
from scipy.stats import linregress

class FBM:
    """Fractional Brownian Motion simulator"""
    
    def __init__(self, H, n=1000, T=1.0):
        """
        Initialize the FBM simulator
        
        Parameters:
        -----------
        H : float
            Hurst parameter, must be in (0, 1)
        n : int
            Number of steps
        T : float
            Time horizon
        """
        self.H = H
        self.n = n
        self.T = T
        self.dt = T / n
        self.times = np.linspace(0, T, n+1)
    
    def covariance_matrix(self):
        """Compute the covariance matrix of fBm"""
        n = self.n + 1
        H = self.H
        
        # Compute pairwise time differences
        t = self.times
        t_mesh = np.meshgrid(t, t)
        
        # Compute covariance matrix using the formula
        # E[B_t B_s] = 0.5 * (|t|^(2H) + |s|^(2H) - |t-s|^(2H))
        cov = 0.5 * (
            np.abs(t_mesh[0])**(2*H) + 
            np.abs(t_mesh[1])**(2*H) - 
            np.abs(t_mesh[0] - t_mesh[1])**(2*H)
        )
        
        return cov
    
    def simulate(self, seed=None):
        """
        Simulate a path of fractional Brownian motion
        
        Parameters:
        -----------
        seed : int, optional
            Random seed for reproducibility
        
        Returns:
        --------
        np.ndarray
            Simulated fBm path
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Compute covariance matrix
        cov = self.covariance_matrix()
        
        # Simulate multivariate normal with the covariance matrix
        return np.random.multivariate_normal(np.zeros(self.n + 1), cov)
    
    def simulate_drifted(self, drift_func=None, seed=None):
        """
        Simulate a drifted fractional Brownian motion: X_t = W^H_t + ∫_0^t ξ_s ds
        
        Parameters:
        -----------
        drift_func : callable, optional
            Function that takes (t, X_t) and returns the drift at time t
        seed : int, optional
            Random seed for reproducibility
        
        Returns:
        --------
        np.ndarray
            Simulated drifted fBm path
        """
        # Simulate standard fBm
        fBm_path = self.simulate(seed)
        
        if drift_func is None:
            return fBm_path
        
        # Add drift component
        t = self.times
        X = np.copy(fBm_path)
        
        for i in range(1, len(t)):
            # Euler-Maruyama integration of the drift
            drift = 0
            for j in range(i):
                drift += drift_func(t[j], X[j]) * self.dt
            X[i] = fBm_path[i] + drift
            
        return X

class FractionalOU:
    """Fractional Ornstein-Uhlenbeck process simulator"""
    
    def __init__(self, H, rho, mu, x0=0, n=1000, T=1.0):
        """
        Initialize the fractional OU process simulator
        
        Parameters:
        -----------
        H : float
            Hurst parameter, must be in (0, 1)
        rho : float
            Mean reversion speed
        mu : float
            Long-term mean
        x0 : float
            Initial value
        n : int
            Number of steps
        T : float
            Time horizon
        """
        self.H = H
        self.rho = rho
        self.mu = mu
        self.x0 = x0
        self.n = n
        self.T = T
        self.dt = T / n
        self.times = np.linspace(0, T, n+1)
        self.fbm = FBM(H, n, T)
    
    def simulate(self, seed=None):
        """
        Simulate a path of fractional Ornstein-Uhlenbeck process
        
        Parameters:
        -----------
        seed : int, optional
            Random seed for reproducibility
        
        Returns:
        --------
        np.ndarray
            Simulated fOU path
        """
        # Define the drift function for the fOU process
        def drift_func(t, x):
            return self.rho * (self.mu - x)
        
        # Simulate drifted fBm with the fOU drift
        X = self.fbm.simulate_drifted(drift_func, seed)
        
        # Add initial value
        X = X + self.x0
        
        return X

class HurstEstimator:
    """Hurst parameter estimator based on the paper by Han and Schied"""
    
    def __init__(self, m=3, alpha=None):
        """
        Initialize the Hurst estimator
        
        Parameters:
        -----------
        m : int
            Parameter for sequential scaling factor
        alpha : array-like, optional
            Weights for sequential scaling factor
        """
        self.m = m
        
        # Default weights if not provided
        if alpha is None:
            self.alpha = np.ones(m+1)
            self.alpha[0] = 1.0  # Ensure alpha_0 > 0
        else:
            self.alpha = np.asarray(alpha)
            assert len(self.alpha) == m+1, "alpha must have length m+1"
            assert self.alpha[0] > 0, "alpha_0 must be positive"
    
    def compute_vartheta(self, y, n, k):
        """
        Compute the coefficient vartheta_{n,k} for function y
        
        Parameters:
        -----------
        y : array-like
            Function values on dyadic grid
        n : int
            Scale parameter
        k : int
            Index parameter
        
        Returns:
        --------
        float
            Computed coefficient
        """
        # Using a safer approach to compute indices
        n_points = len(y)
        
        # Calculate indices for the vartheta coefficient
        idx_0 = (k * 4) * 2**(-(n+2)) * (n_points-1)
        idx_1 = (k * 4 + 1) * 2**(-(n+2)) * (n_points-1)
        idx_3 = (k * 4 + 3) * 2**(-(n+2)) * (n_points-1)
        idx_4 = (k * 4 + 4) * 2**(-(n+2)) * (n_points-1)
        
        # Convert to integers and ensure they are within bounds
        idx_0 = min(int(round(idx_0)), n_points-1)
        idx_1 = min(int(round(idx_1)), n_points-1)
        idx_3 = min(int(round(idx_3)), n_points-1)
        idx_4 = min(int(round(idx_4)), n_points-1)
        
        # Compute the vartheta coefficient
        vartheta = 2**(3*n/2 + 3) * (
            y[idx_0] - 
            2 * y[idx_1] + 
            2 * y[idx_3] - 
            y[idx_4]
        )
        
        return vartheta
    
    def Rb_n(self, y, n):
        """
        Compute the Rb_n estimator for function y
        
        Parameters:
        -----------
        y : array-like
            Function values on dyadic grid
        n : int
            Scale parameter
        
        Returns:
        --------
        float
            Estimated Hurst parameter
        """
        # The number of points needed for level n is 2^(n+2) + 1
        # We need to ensure we have enough points
        n_points = len(y)
        max_n = int(np.log2(n_points - 1)) - 2
        
        if n > max_n:
            raise ValueError(f"n={n} is too large for the given data with {n_points} points. Max n is {max_n}")
        
        # Compute the sum of squared vartheta coefficients
        vartheta_sum = 0
        for k in range(2**n):
            vartheta_nk = self.compute_vartheta(y, n, k)
            vartheta_sum += vartheta_nk**2
        
        # If the sum is zero, return a default value
        if vartheta_sum == 0:
            return 0.5  # Default to Brownian motion
        
        return 1 - (1/n) * np.log2(np.sqrt(vartheta_sum))
    
    def Rs_n(self, y, n):
        """
        Compute the Rs_n estimator for function y
        
        Parameters:
        -----------
        y : array-like
            Function values on dyadic grid
        n : int
            Scale parameter
        
        Returns:
        --------
        float
            Estimated Hurst parameter with scale invariance
        """
        if n <= self.m:
            return self.Rb_n(y, n)
        
        # Define objective function for scaling optimization
        def objective(log_lambda):
            lambda_val = np.exp(log_lambda)
            y_scaled = lambda_val * y
            
            # Compute Rb_k for k from n-m to n
            Rb_values = []
            for k in range(n-self.m, n+1):
                try:
                    Rb_values.append(self.Rb_n(y_scaled, k))
                except ValueError:
                    # If we can't compute for this k, use the previous value
                    if Rb_values:
                        Rb_values.append(Rb_values[-1])
                    else:
                        Rb_values.append(0.5)  # Default value
            
            # Compute differences
            diffs = [Rb_values[i+1] - Rb_values[i] for i in range(len(Rb_values)-1)]
            
            # Compute weighted sum of squared differences
            weighted_sum = 0
            for i in range(len(diffs)):
                weighted_sum += self.alpha[i] * diffs[i]**2
            
            return weighted_sum
        
        # Find optimal scaling factor
        result = minimize_scalar(objective, method='brent')
        lambda_star = np.exp(result.x)
        
        # Return Rb_n with optimal scaling
        try:
            return self.Rb_n(lambda_star * y, n)
        except ValueError:
            # If we can't compute for this n, return the last value we could compute
            for k in range(n-1, self.m, -1):
                try:
                    return self.Rb_n(lambda_star * y, k)
                except ValueError:
                    continue
            
            # If all else fails, return a default value
            return 0.5
    
    def compute_antiderivative(self, x, g=None):
        """
        Compute the antiderivative of g(x)
        
        Parameters:
        -----------
        x : array-like
            Function values on uniform grid
        g : callable, optional
            Nonlinear function to apply to x
        
        Returns:
        --------
        np.ndarray
            Antiderivative values on the same grid
        """
        if g is None:
            g = lambda x: x  # Identity function
        
        # Apply function g to x
        g_x = g(x)
        
        # Compute antiderivative using trapezoidal rule
        y = np.zeros_like(x)
        dt = 1.0 / (len(x) - 1)
        
        for i in range(1, len(x)):
            y[i] = y[i-1] + 0.5 * dt * (g_x[i-1] + g_x[i])
        
        return y
    
    def create_dyadic_grid(self, n_max):
        """
        Create a dyadic grid of level n_max
        
        Parameters:
        -----------
        n_max : int
            Maximum dyadic level
        
        Returns:
        --------
        np.ndarray
            Dyadic grid points
        """
        # The number of points is 2^(n_max+2) + 1
        n_points = 2**(n_max+2) + 1
        return np.linspace(0, 1, n_points)
    
    def interpolate_to_dyadic(self, x, y, n_max):
        """
        Interpolate function y to dyadic grid of level n_max
        
        Parameters:
        -----------
        x : array-like
            Original grid points
        y : array-like
            Function values on original grid
        n_max : int
            Maximum dyadic level
        
        Returns:
        --------
        np.ndarray
            Function values on dyadic grid
        """
        # Create dyadic grid
        x_dyadic = self.create_dyadic_grid(n_max)
        
        # Linear interpolation
        y_dyadic = np.interp(x_dyadic, x, y)
        
        return y_dyadic
    
    def estimate_hurst(self, x, n_values, g=None):
        """
        Estimate Hurst parameter for a range of n values
        
        Parameters:
        -----------
        x : array-like
            Sample path
        n_values : list
            List of n values to use for estimation
        g : callable, optional
            Nonlinear function to apply to x
        
        Returns:
        --------
        tuple
            (Rb_estimates, Rs_estimates)
        """
        # Find the maximum n value
        n_max = max(n_values)
        
        # Create uniform grid for input x
        x_grid = np.linspace(0, 1, len(x))
        
        # Compute antiderivative of g(x)
        y = self.compute_antiderivative(x, g)
        
        # Interpolate to dyadic grid
        y_dyadic = self.interpolate_to_dyadic(x_grid, y, n_max)
        
        # Compute estimates for each n
        Rb_estimates = []
        Rs_estimates = []
        
        for n in n_values:
            try:
                Rb_estimates.append(self.Rb_n(y_dyadic, n))
            except ValueError as e:
                print(f"Error computing Rb_{n}: {e}")
                Rb_estimates.append(np.nan)
            
            try:
                Rs_estimates.append(self.Rs_n(y_dyadic, n))
            except ValueError as e:
                print(f"Error computing Rs_{n}: {e}")
                Rs_estimates.append(np.nan)
        
        return Rb_estimates, Rs_estimates

def simulate_fractional_sv_model(H, rho, mu, vol_of_vol, x0, T=1.0, n=1000, seed=None):
    """
    Simulate a fractional stochastic volatility model
    
    Parameters:
    -----------
    H : float
        Hurst parameter
    rho : float
        Mean reversion speed
    mu : float
        Long-term mean
    vol_of_vol : float
        Volatility of volatility
    x0 : float
        Initial log-volatility
    T : float
        Time horizon
    n : int
        Number of steps
    seed : int, optional
        Random seed
    
    Returns:
    --------
    tuple
        (times, log_vol, vol, integrated_var)
    """
    # Initialize random seed
    if seed is not None:
        np.random.seed(seed)
    
    # Simulate fractional OU process for log-volatility
    fOU = FractionalOU(H, rho, mu, x0, n, T)
    log_vol = fOU.simulate(seed)
    
    # Convert to volatility
    vol = np.exp(log_vol)
    
    # Compute integrated variance
    times = np.linspace(0, T, n+1)
    integrated_var = np.zeros_like(times)
    
    for i in range(1, len(times)):
        dt = times[i] - times[i-1]
        integrated_var[i] = integrated_var[i-1] + 0.5 * dt * (vol[i-1]**2 + vol[i]**2)
    
    return times, log_vol, vol, integrated_var

def test_convergence_rate(H_values, n_values, repetitions=5, nonlinear_func=None):
    """
    Test the convergence rate of the estimators
    
    Parameters:
    -----------
    H_values : list
        List of Hurst parameters to test
    n_values : list
        List of n values to use for estimation
    repetitions : int
        Number of repetitions for each parameter set
    nonlinear_func : callable, optional
        Nonlinear function to apply to fBm
    
    Returns:
    --------
    pandas.DataFrame
        Results of the convergence tests
    """
    results = []
    
    # Default nonlinear function if not provided
    if nonlinear_func is None:
        nonlinear_func = lambda x: np.exp(2*x)  # e^(2x) as in the paper
    
    # Initialize estimator
    estimator = HurstEstimator()
    
    for H in H_values:
        for rep in tqdm(range(repetitions), desc=f"H={H}"):
            # Simulate fBm with sufficient points for dyadic grid
            n_points = 2**(max(n_values)+3)  # Ensure enough points for all n values
            fbm = FBM(H, n=n_points)
            x = fbm.simulate(seed=rep)
            
            # Estimate Hurst parameter
            Rb_estimates, Rs_estimates = estimator.estimate_hurst(x, n_values, nonlinear_func)
            
            # Record results
            for i, n in enumerate(n_values):
                results.append({
                    'H': H,
                    'n': n,
                    'repetition': rep,
                    'Rb': Rb_estimates[i],
                    'Rs': Rs_estimates[i],
                    'Rb_error': abs(Rb_estimates[i] - H) if not np.isnan(Rb_estimates[i]) else np.nan,
                    'Rs_error': abs(Rs_estimates[i] - H) if not np.isnan(Rs_estimates[i]) else np.nan
                })
    
    # Convert to DataFrame
    return pd.DataFrame(results)

def test_fsv_model():
    """
    Test the estimators on simulated fractional stochastic volatility model
    
    Returns:
    --------
    pandas.DataFrame
        Results of the tests
    """
    results = []
    
    # Parameters for fractional SV model
    H_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    rho = 0.5
    mu = 0.0
    vol_of_vol = 0.1
    x0 = 0.0
    n_values = list(range(3, 7))  # Lower n values to ensure stability
    repetitions = 5
    
    # Initialize estimator
    estimator = HurstEstimator()
    
    for H in H_values:
        for rep in tqdm(range(repetitions), desc=f"H={H}"):
            # Simulate fractional SV model with enough points
            n_points = 2**(max(n_values)+5)  # Extra points for stability
            _, _, _, integrated_var = simulate_fractional_sv_model(
                H, rho, mu, vol_of_vol, x0, n=n_points, seed=rep
            )
            
            # Estimate Hurst parameter directly from integrated variance
            Rb_estimates, Rs_estimates = estimator.estimate_hurst(integrated_var, n_values)
            
            # Record results
            for i, n in enumerate(n_values):
                results.append({
                    'H': H,
                    'n': n,
                    'repetition': rep,
                    'Rb': Rb_estimates[i],
                    'Rs': Rs_estimates[i],
                    'Rb_error': abs(Rb_estimates[i] - H) if not np.isnan(Rb_estimates[i]) else np.nan,
                    'Rs_error': abs(Rs_estimates[i] - H) if not np.isnan(Rs_estimates[i]) else np.nan
                })
    
    # Convert to DataFrame
    return pd.DataFrame(results)

def plot_results(results, filename=None):
    """
    Plot the results of the convergence tests
    
    Parameters:
    -----------
    results : pandas.DataFrame
        Results of the convergence tests
    filename : str, optional
        If provided, save the plot to this file
    """
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Rb error vs n for different H
    plt.subplot(2, 2, 1)
    for H in results['H'].unique():
        data = results[results['H'] == H]
        mean_errors = data.groupby('n')['Rb_error'].mean()
        plt.plot(mean_errors.index, mean_errors.values, 'o-', label=f'H={H}')
    
    plt.xlabel('n')
    plt.ylabel('|Rb - H|')
    plt.title('Convergence of Rb estimator')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')
    
    # Plot 2: Rs error vs n for different H
    plt.subplot(2, 2, 2)
    for H in results['H'].unique():
        data = results[results['H'] == H]
        mean_errors = data.groupby('n')['Rs_error'].mean()
        plt.plot(mean_errors.index, mean_errors.values, 'o-', label=f'H={H}')
    
    plt.xlabel('n')
    plt.ylabel('|Rs - H|')
    plt.title('Convergence of Rs estimator')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')
    
    # Plot 3: Rb vs true H for different n
    plt.subplot(2, 2, 3)
    for n in results['n'].unique():
        data = results[results['n'] == n]
        mean_estimates = data.groupby('H')['Rb'].mean()
        plt.plot(mean_estimates.index, mean_estimates.values, 'o-', label=f'n={n}')
    
    plt.plot([0, 1], [0, 1], 'k--', label='Perfect estimation')
    plt.xlabel('True H')
    plt.ylabel('Estimated Rb')
    plt.title('Accuracy of Rb estimator')
    plt.legend()
    plt.grid(True)
    
    # Plot 4: Rs vs true H for different n
    plt.subplot(2, 2, 4)
    for n in results['n'].unique():
        data = results[results['n'] == n]
        mean_estimates = data.groupby('H')['Rs'].mean()
        plt.plot(mean_estimates.index, mean_estimates.values, 'o-', label=f'n={n}')
    
    plt.plot([0, 1], [0, 1], 'k--', label='Perfect estimation')
    plt.xlabel('True H')
    plt.ylabel('Estimated Rs')
    plt.title('Accuracy of Rs estimator')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    
    if filename:
        plt.savefig(filename, dpi=300)
    
    plt.show()

def verify_convergence_rate(results):
    """
    Verify the theoretical convergence rate from the paper
    
    Parameters:
    -----------
    results : pandas.DataFrame
        Results of the convergence tests
    
    Returns:
    --------
    pandas.DataFrame
        Empirical convergence rates
    """
    rates = []
    
    for H in results['H'].unique():
        data = results[results['H'] == H]
        
        # Compute mean errors for each n
        mean_errors = data.groupby('n')['Rs_error'].mean()
        n_values = mean_errors.index.values
        
        # Skip if we have NaN values
        if mean_errors.isnull().any() or len(mean_errors) < 3:
            continue
            
        # Theoretical rate: O(sqrt(n) * 2^(-(H/2 ∧ 1/4)n))
        theoretical_rate = np.sqrt(n_values) * 2**(-min(H/2, 0.25) * n_values)
        
        # Normalize to compare shapes
        normalized_errors = mean_errors.values / mean_errors.values[0]
        normalized_theory = theoretical_rate / theoretical_rate[0]
        
        # Compute correlation between log errors and log theoretical rate
        log_errors = np.log(mean_errors.values)
        log_theory = np.log(theoretical_rate)
        correlation = np.corrcoef(log_errors, log_theory)[0, 1]
        
        # Linear regression to find empirical rate
        slope, intercept, r_value, p_value, std_err = linregress(n_values, log_errors)
        
        rates.append({
            'H': H,
            'correlation': correlation,
            'empirical_slope': slope,
            'theoretical_rate': f"O(sqrt(n) * 2^(-{min(H/2, 0.25)}n))"
        })
    
    return pd.DataFrame(rates)

def simulate_price_series(n_days=1000, H=0.1, rho=1.0, mu=-3.0, vol_of_vol=0.5, 
                          mean_return=0.0, return_vol=0.2, price_0=100, seed=None):
    """
    Simulate a price series with rough stochastic volatility
    
    Parameters:
    -----------
    n_days : int
        Number of days to simulate
    H : float
        Hurst parameter for volatility process
    rho : float
        Mean reversion speed for volatility
    mu : float
        Long-term mean log-volatility
    vol_of_vol : float
        Volatility of volatility
    mean_return : float
        Mean daily return
    return_vol : float
        Base volatility level
    price_0 : float
        Initial price
    seed : int, optional
        Random seed
    
    Returns:
    --------
    tuple
        (times, prices, returns, log_vol, vol, integrated_var)
    """
    # Set random seed
    if seed is not None:
        np.random.seed(seed)
    
    # Simulate fractional stochastic volatility
    times = np.linspace(0, 1, n_days+1)
    _, log_vol, vol, integrated_var = simulate_fractional_sv_model(
        H, rho, mu, vol_of_vol, mu, T=1.0, n=n_days, seed=seed
    )
    
    # Generate returns with stochastic volatility
    returns = np.zeros(n_days)
    for i in range(n_days):
        current_vol = vol[i]
        returns[i] = mean_return/252 + current_vol * return_vol/np.sqrt(252) * np.random.normal()
    
    # Generate prices
    prices = price_0 * np.cumprod(np.exp(returns))
    prices = np.insert(prices, 0, price_0)
    
    return times, prices, returns, log_vol, vol, integrated_var

class RoughVolatilityTradingStrategy:
    """Trading strategy based on Hurst parameter estimation in rough volatility models"""
    
    def __init__(self, window_size=252, estimation_freq=21, lookback_periods=4):
        """
        Initialize the trading strategy
        
        Parameters:
        -----------
        window_size : int
            Size of the rolling window for Hurst estimation (trading days)
        estimation_freq : int
            Frequency of re-estimating the Hurst parameter (trading days)
        lookback_periods : int
            Number of past periods to use for estimating volatility regime
        """
        self.window_size = window_size
        self.estimation_freq = estimation_freq
        self.lookback_periods = lookback_periods
        self.estimator = HurstEstimator()
        self.current_H = None
        self.last_estimate_day = 0
        self.vol_regime = 'normal'  # 'low', 'normal', 'high'
    
    def estimate_hurst_from_returns(self, returns, n_values=[3, 4, 5]):
        """
        Estimate Hurst parameter from returns data
        
        Parameters:
        -----------
        returns : array-like
            Daily returns
        n_values : list
            List of n values to use for estimation
        
        Returns:
        --------
        float
            Estimated Hurst parameter
        """
        # Calculate realized variance (squared returns)
        realized_var = returns**2
        
        # Calculate integrated variance (cumulative sum of realized variance)
        integrated_var = np.cumsum(realized_var)
        
        # Estimate Hurst parameter
        _, Rs_estimates = self.estimator.estimate_hurst(integrated_var, n_values)
        
        # Filter out NaN values
        valid_estimates = [est for est in Rs_estimates if not np.isnan(est)]
        
        # Return the average estimate across different n values
        if valid_estimates:
            return np.mean(valid_estimates)
        else:
            return 0.5  # Default to Brownian motion if no valid estimates
    
    def detect_volatility_regime(self, H_estimates, current_vol):
        """
        Detect volatility regime based on Hurst parameter history
        
        Parameters:
        -----------
        H_estimates : list
            Historical Hurst parameter estimates
        current_vol : float or array-like
            Current realized volatility or recent volatility history
        
        Returns:
        --------
        str
            Volatility regime: 'low', 'normal', or 'high'
        """
        if len(H_estimates) < 2:
            return 'normal'
        
        # Detect regime based on Hurst parameter and current volatility
        H_current = H_estimates[-1]
        H_trend = H_estimates[-1] - H_estimates[-2]
        
        # Make sure we're dealing with the most recent volatility value
        if isinstance(current_vol, (list, np.ndarray)):
            current_vol_value = current_vol[-1]
            # Calculate median of recent volatility
            vol_history = current_vol[-min(20, len(current_vol)):]
            vol_median = np.median(vol_history)
        else:
            current_vol_value = current_vol
            vol_median = current_vol  # No history available
        
        # Very rough volatility (H < 0.2) often indicates high volatility regime
        if H_current < 0.2:
            return 'high'
        
        # Decreasing Hurst parameter indicates increasing roughness, potentially higher volatility
        if H_trend < -0.05 and current_vol_value > vol_median:
            return 'high'
        
        # Increasing Hurst parameter indicates smoother process, potentially lower volatility
        if H_trend > 0.05 and current_vol_value < vol_median:
            return 'low'
        
        return 'normal'
    
    def compute_position_size(self, price, realized_vol, H):
        """
        Compute position size based on volatility and Hurst parameter
        
        Parameters:
        -----------
        price : float
            Current price
        realized_vol : float
            Realized volatility
        H : float
            Hurst parameter
        
        Returns:
        --------
        float
            Position size (-1 to 1)
        """
        # Base volatility adjustment
        vol_adjustment = 0.2 / realized_vol if realized_vol > 0 else 1.0
        
        # Hurst parameter adjustment
        # Lower H = rougher volatility = smaller position
        H_adjustment = 2 * (H - 0.2)  # Normalize around H=0.2
        H_adjustment = max(0.5, min(1.5, H_adjustment))  # Limit effect
        
        # Combine adjustments
        position_size = vol_adjustment * H_adjustment
        
        # Limit position size
        return max(-1.0, min(1.0, position_size))
    
    def backtest(self, prices, returns=None):
        """
        Backtest the strategy on historical price data
        
        Parameters:
        -----------
        prices : array-like
            Historical prices
        returns : array-like, optional
            Pre-computed returns. If None, returns are computed from prices.
        
        Returns:
        --------
        dict
            Backtest results
        """
        # Compute returns if not provided
        if returns is None:
            returns = np.diff(np.log(prices))
        
        # Initialize results
        n_days = len(returns)
        positions = np.zeros(n_days)
        H_estimates = []
        vol_regimes = []
        realized_vol = []
        
        # Initial volatility estimate
        vol_window = min(20, n_days // 4)
        current_vol = np.std(returns[:vol_window]) * np.sqrt(252)
        realized_vol.append(current_vol)
        
        for t in range(vol_window, n_days):
            # Update realized volatility (exponentially weighted)
            new_vol = 0.94 * realized_vol[-1] + 0.06 * abs(returns[t-1]) * np.sqrt(252)
            realized_vol.append(new_vol)
            
            # Re-estimate Hurst parameter periodically
            if (t - self.last_estimate_day) >= self.estimation_freq and t >= self.window_size:
                window_returns = returns[t-self.window_size:t]
                H_est = self.estimate_hurst_from_returns(window_returns)
                H_estimates.append(H_est)
                self.current_H = H_est
                self.last_estimate_day = t
                
                # Update volatility regime
                # Pass the most recent volatility value
                self.vol_regime = self.detect_volatility_regime(H_estimates, realized_vol[-1])
            
            vol_regimes.append(self.vol_regime)
            
            # Compute position
            if self.current_H is not None:
                # Adjust position based on volatility regime
                if self.vol_regime == 'high':
                    regime_factor = 0.5  # Reduce position in high volatility
                elif self.vol_regime == 'low':
                    regime_factor = 1.5  # Increase position in low volatility
                else:
                    regime_factor = 1.0
                
                # Base position from volatility and Hurst parameter
                base_position = self.compute_position_size(
                    prices[t], realized_vol[-1], self.current_H
                )
                
                # Apply regime adjustment
                positions[t] = base_position * regime_factor
            else:
                positions[t] = 0  # No position until we have a Hurst estimate
        
        # Compute strategy returns
        strategy_returns = np.zeros_like(returns)
        strategy_returns[1:] = positions[:-1] * returns[1:]
        
        # Compute cumulative returns
        cum_returns = np.cumprod(1 + returns) - 1
        cum_strategy = np.cumprod(1 + strategy_returns) - 1
        
        # Compute performance metrics
        if np.std(strategy_returns) > 0:
            sharpe = np.mean(strategy_returns) / np.std(strategy_returns) * np.sqrt(252)
        else:
            sharpe = 0
        
        max_drawdown = np.max(np.maximum.accumulate(cum_strategy) - cum_strategy)
        
        return {
            'positions': positions,
            'strategy_returns': strategy_returns,
            'cum_returns': cum_returns,
            'cum_strategy': cum_strategy,
            'H_estimates': H_estimates,
            'realized_vol': realized_vol,
            'vol_regimes': vol_regimes,
            'sharpe': sharpe,
            'max_drawdown': max_drawdown
        }

def plot_backtest_results(results, H, filename=None):
    """
    Plot backtest results
    
    Parameters:
    -----------
    results : dict
        Backtest results from RoughVolatilityTradingStrategy
    H : float
        True Hurst parameter
    filename : str, optional
        If provided, save the plot to this file
    """
    plt.figure(figsize=(15, 12))
    
    # Plot 1: Cumulative returns
    plt.subplot(3, 2, 1)
    plt.plot(results['cum_returns'], 'b-', label='Buy & Hold')
    plt.plot(results['cum_strategy'], 'g-', label='Strategy')
    plt.xlabel('Days')
    plt.ylabel('Cumulative Return')
    plt.title(f'Returns (H={H})')
    plt.legend()
    plt.grid(True)
    
    # Plot 2: Positions
    plt.subplot(3, 2, 2)
    plt.plot(results['positions'], 'r-')
    plt.xlabel('Days')
    plt.ylabel('Position Size')
    plt.title('Trading Positions')
    plt.grid(True)
    
    # Plot 3: Hurst estimates
    plt.subplot(3, 2, 3)
    est_indices = range(0, len(results['H_estimates'])*21, 21)
    if est_indices and results['H_estimates']:
        plt.plot(est_indices, 
                results['H_estimates'], 'o-')
        plt.axhline(y=H, color='r', linestyle='--', label='True H')
        plt.xlabel('Days')
        plt.ylabel('Hurst Parameter')
        plt.title('Hurst Parameter Estimates')
        plt.legend()
        plt.grid(True)
    
    # Plot 4: Realized volatility
    plt.subplot(3, 2, 4)
    plt.plot(results['realized_vol'], 'b-')
    plt.xlabel('Days')
    plt.ylabel('Annualized Volatility')
    plt.title('Realized Volatility')
    plt.grid(True)
    
    # Plot 5: Volatility regimes
    plt.subplot(3, 2, 5)
    if results['vol_regimes']:
        regime_values = np.array([0 if r == 'low' else (1 if r == 'normal' else 2) 
                                for r in results['vol_regimes']])
        plt.plot(regime_values, 'g-')
        plt.xlabel('Days')
        plt.ylabel('Regime (0=Low, 1=Normal, 2=High)')
        plt.title('Volatility Regimes')
        plt.yticks([0, 1, 2], ['Low', 'Normal', 'High'])
        plt.grid(True)
    
    # Plot 6: Performance metrics
    plt.subplot(3, 2, 6)
    plt.axis('off')
    plt.text(0.1, 0.9, f"Sharpe Ratio: {results['sharpe']:.2f}", fontsize=12)
    plt.text(0.1, 0.8, f"Max Drawdown: {results['max_drawdown']:.2%}", fontsize=12)
    plt.text(0.1, 0.7, f"Final Return: {results['cum_strategy'][-1]:.2%}", fontsize=12)
    plt.text(0.1, 0.6, f"Buy & Hold Return: {results['cum_returns'][-1]:.2%}", fontsize=12)
    plt.text(0.1, 0.5, f"Avg Position: {np.mean(np.abs(results['positions'])):.2f}", fontsize=12)
    plt.text(0.1, 0.4, f"True H: {H}", fontsize=12)
    if results['H_estimates']:
        plt.text(0.1, 0.3, f"Avg Estimated H: {np.mean(results['H_estimates']):.2f}", fontsize=12)
    plt.title('Performance Metrics')
    
    plt.tight_layout()
    
    if filename:
        plt.savefig(filename, dpi=300)
    
    plt.show()

def test_trading_strategy():
    """Test the rough volatility trading strategy on simulated data"""
    # Simulate multiple price series with different Hurst parameters
    H_values = [0.1, 0.3, 0.5, 0.7]
    n_days = 1000
    results = []
    
    for H in H_values:
        print(f"\nTesting strategy with H={H}")
        
        try:
            # Simulate price series
            _, prices, returns, _, _, _ = simulate_price_series(
                n_days=n_days, H=H, seed=42
            )
            
            # Initialize and backtest strategy
            strategy = RoughVolatilityTradingStrategy(
                window_size=252, estimation_freq=21, lookback_periods=4
            )
            backtest_results = strategy.backtest(prices, returns)
            
            # Store results
            results.append({
                'H': H,
                'backtest_results': backtest_results
            })
            
            # Plot results
            plot_backtest_results(backtest_results, H, f"strategy_H_{H}.png")
            
        except Exception as e:
            print(f"Error in strategy test for H={H}: {e}")
    
    return results

def main():
    # Set parameters
    H_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    n_values = list(range(3, 7))  # Lower n values to ensure stability
    repetitions = 5  # Fewer repetitions for faster execution
    
    print("Testing convergence with nonlinear function g(x) = exp(2x)...")
    results_nonlinear = test_convergence_rate(H_values, n_values, repetitions, lambda x: np.exp(2*x))
    
    print("\nVerifying theoretical convergence rates...")
    rates = verify_convergence_rate(results_nonlinear)
    print(rates)
    
    print("\nPlotting results...")
    plot_results(results_nonlinear, "nonlinear_convergence.png")
    
    print("\nTesting on fractional stochastic volatility model...")
    results_fsv = test_fsv_model()
    plot_results(results_fsv, "fsv_model_convergence.png")
    
    print("\nTesting trading strategy based on Hurst estimation...")
    strategy_results = test_trading_strategy()
    
    # Save results
    results_nonlinear.to_csv("nonlinear_results.csv", index=False)
    results_fsv.to_csv("fsv_results.csv", index=False)
    
    print("\nAll tests completed!")

if __name__ == "__main__":
    main()