import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
from scipy.stats import poisson
from tqdm import tqdm
import time
import warnings
warnings.filterwarnings('ignore')

class SDSHModel:
    """
    State-Dependent Spread Hawkes (SDSH) model for bid-ask spread dynamics
    """
    
    def __init__(self, K=1, L=2, max_spread=5, beta_values=None):
        """
        Initialize the SDSH model
        
        Parameters:
        - K: Maximum jump size
        - L: Number of exponential kernels
        - max_spread: Maximum spread value for f functions
        - beta_values: Decay rates for exponential kernels
        """
        self.K = K
        self.L = L
        self.max_spread = max_spread
        
        # Define the set of events E
        self.E = [f"+{i}" for i in range(1, K+1)] + [f"-{i}" for i in range(1, K+1)]
        
        # Initialize model parameters
        if beta_values is None:
            # Logarithmically spaced beta values
            self.beta_values = np.logspace(-1, 4, L)
        else:
            self.beta_values = beta_values
            
        # Exogenous intensities
        self.mu = {e: 0.1 for e in self.E}
        
        # Kernel parameters
        self.alpha = {(e, e_prime, l): 0.1 
                     for e in self.E 
                     for e_prime in self.E 
                     for l in range(L)}
        
        # State-dependent functions
        self.f = {}
        for e in self.E:
            self.f[e] = np.ones(max_spread + 1)
            
            # Set f to 0 for negative spread values
            if e.startswith('-'):
                jump_size = int(e[1:])
                for s in range(1, jump_size + 1):
                    self.f[e][s] = 0
                    
            # For positive jumps, make f decrease with spread
            if e.startswith('+'):
                for s in range(2, max_spread + 1):
                    self.f[e][s] = max(0.1, 1.0 / s)
                    
            # For negative jumps, make f increase with spread
            if e.startswith('-'):
                jump_size = int(e[1:])
                for s in range(jump_size + 1, max_spread + 1):
                    self.f[e][s] = min(5.0, s / 2)
        
    def kernel(self, t, e, e_prime):
        """
        Compute the kernel function value at time t
        
        Parameters:
        - t: Time
        - e: Target event type
        - e_prime: Source event type
        
        Returns:
        - Kernel value
        """
        result = 0
        for l in range(self.L):
            result += self.alpha[(e, e_prime, l)] * self.beta_values[l] * np.exp(-self.beta_values[l] * t)
        return result
    
    def compute_intensity(self, t, event_history, current_spread):
        """
        Compute the intensity for each event type at time t
        
        Parameters:
        - t: Current time
        - event_history: List of (time, event_type) tuples
        - current_spread: Current spread value
        
        Returns:
        - Dictionary of intensities for each event type
        """
        intensities = {}
        
        for e in self.E:
            # Base intensity
            intensity = self.mu[e]
            
            # Add contribution from past events
            for t_past, e_prime in event_history:
                if t_past < t:
                    dt = t - t_past
                    intensity += self.kernel(dt, e, e_prime)
            
            # Multiply by state-dependent function
            intensity *= self.f[e][current_spread]
            
            intensities[e] = max(0, intensity)  # Ensure non-negative intensity
            
        return intensities
    
    def simulate(self, T, seed=None):
        """
        Simulate the SDSH model
        
        Parameters:
        - T: End time for simulation
        - seed: Random seed
        
        Returns:
        - times: Event times
        - events: Event types
        - spread_path: Spread values
        """
        if seed is not None:
            np.random.seed(seed)
            
        # Initialize
        t = 0
        spread = 1  # Initial spread
        times = []
        events = []
        spread_path = [(0, spread)]
        event_history = []
        
        # Simulate using thinning algorithm
        while t < T:
            # Compute upper bound for total intensity
            intensities = self.compute_intensity(t, event_history, spread)
            total_intensity = sum(intensities.values())
            
            if total_intensity <= 0:
                # No more events possible, advance time
                t = T
                continue
                
            # Generate next event time
            dt = np.random.exponential(1 / total_intensity)
            t_new = t + dt
            
            if t_new > T:
                # Event occurs after end time
                break
                
            # Accept/reject
            intensities_new = self.compute_intensity(t_new, event_history, spread)
            total_intensity_new = sum(intensities_new.values())
            
            if np.random.random() < total_intensity_new / total_intensity:
                # Event accepted, determine type
                probs = [intensities_new[e] / total_intensity_new for e in self.E]
                event_type = np.random.choice(self.E, p=probs)
                
                # Update state
                times.append(t_new)
                events.append(event_type)
                event_history.append((t_new, event_type))
                
                # Update spread
                if event_type.startswith('+'):
                    jump_size = int(event_type[1:])
                    spread += jump_size
                elif event_type.startswith('-'):
                    jump_size = int(event_type[1:])
                    spread -= jump_size
                
                spread_path.append((t_new, spread))
                
                # Advance time
                t = t_new
            else:
                # Event rejected, try again
                t = t_new
                
        return times, events, spread_path
    
    def log_likelihood(self, data, times, spreads):
        """
        Compute the log-likelihood of the model given data
        
        Parameters:
        - data: List of (time, event_type) tuples
        - times: Array of all times for intensity computation
        - spreads: Array of spread values at each time
        
        Returns:
        - Log-likelihood value
        """
        log_lik = 0
        
        for i, (t, e) in enumerate(data):
            # Find the spread at time t
            spread_idx = np.searchsorted(times, t) - 1
            spread = spreads[spread_idx]
            
            # Compute intensity at time t for event e
            event_history = data[:i]
            intensities = self.compute_intensity(t, event_history, spread)
            
            # Add log intensity term
            if intensities[e] > 0:
                log_lik += np.log(intensities[e])
            else:
                log_lik -= 1000  # Penalize zero intensity heavily
            
            # Subtract integral term (approximated by rectangular quadrature)
            if i > 0:
                prev_t = data[i-1][0]
                for tj in np.linspace(prev_t, t, 10)[:-1]:
                    # Find spread at time tj
                    spread_idx_j = np.searchsorted(times, tj) - 1
                    spread_j = spreads[spread_idx_j]
                    
                    # Compute all intensities
                    intensities_j = self.compute_intensity(tj, event_history, spread_j)
                    total_intensity_j = sum(intensities_j.values())
                    
                    # Add to integral term
                    log_lik -= total_intensity_j * (t - prev_t) / 10
        
        return log_lik
    
    def fit(self, data, max_iter=100):
        """
        Fit the model to data using Maximum Likelihood Estimation
        
        Parameters:
        - data: List of (time, event_type, spread) tuples
        - max_iter: Maximum number of iterations
        
        Returns:
        - Self
        """
        # Extract times and spread values
        times = [t for t, _, _ in data]
        spreads = [s for _, _, s in data]
        
        # Extract event data
        event_data = [(t, e) for t, e, _ in data]
        
        # Initialize parameters
        # For simplicity, we'll use a coordinate descent approach
        for _ in range(max_iter):
            # Optimize mu
            for e in self.E:
                count = sum(1 for _, e_i, _ in data if e_i == e)
                self.mu[e] = count / times[-1]
            
            # Optimize alpha (simplified)
            for e in self.E:
                for e_prime in self.E:
                    for l in range(self.L):
                        # Count e_prime -> e influences
                        influence = 0
                        for i, (t, e_i, _) in enumerate(data):
                            if e_i == e:
                                # Sum kernel contribution from previous e_prime events
                                for j in range(i):
                                    if data[j][1] == e_prime:
                                        dt = t - data[j][0]
                                        influence += np.exp(-self.beta_values[l] * dt)
                        
                        # Update alpha
                        if influence > 0:
                            self.alpha[(e, e_prime, l)] = count / influence
                        else:
                            self.alpha[(e, e_prime, l)] = 0.1  # Default
            
            # Optimize f (simplified)
            for e in self.E:
                for s in range(1, self.max_spread + 1):
                    # Count events of type e when spread is s
                    count_e_given_s = sum(1 for t, e_i, s_i in data if e_i == e and s_i == s)
                    count_s = sum(1 for _, _, s_i in data if s_i == s)
                    
                    if count_s > 0:
                        self.f[e][s] = count_e_given_s / count_s
                    else:
                        # No observations of this spread
                        if e.startswith('-'):
                            jump_size = int(e[1:])
                            if s <= jump_size:
                                self.f[e][s] = 0  # Cannot go below 1
                            else:
                                self.f[e][s] = min(5.0, s / 2)  # Default
                        else:  # positive jumps
                            self.f[e][s] = max(0.1, 1.0 / s)  # Default
        
        return self
    
    def forecast(self, history, t0, delta, n_simulations=100):
        """
        Forecast the spread at time t0 + delta
        
        Parameters:
        - history: List of (time, event_type, spread) tuples up to time t0
        - t0: Current time
        - delta: Forecast horizon
        - n_simulations: Number of Monte Carlo simulations
        
        Returns:
        - forecasted_spread: Expected spread at time t0 + delta
        """
        # Extract the current spread
        current_spread = history[-1][2]
        
        # Filter history to include only the last minute
        filtered_history = [(t, e) for t, e, _ in history if t >= t0 - 60]
        
        # Perform Monte Carlo simulations
        spreads = []
        for _ in range(n_simulations):
            # Initialize simulation at current state
            t = t0
            spread = current_spread
            event_history = filtered_history.copy()
            
            # Simulate forward
            while t < t0 + delta:
                # Compute intensities
                intensities = self.compute_intensity(t, event_history, spread)
                total_intensity = sum(intensities.values())
                
                if total_intensity <= 0:
                    # No more events possible
                    break
                
                # Generate next event time
                dt = np.random.exponential(1 / total_intensity)
                t_new = t + dt
                
                if t_new > t0 + delta:
                    # No more events before forecast horizon
                    break
                
                # Determine event type
                probs = [intensities[e] / total_intensity for e in self.E]
                event_type = np.random.choice(self.E, p=probs)
                
                # Update spread
                if event_type.startswith('+'):
                    jump_size = int(event_type[1:])
                    spread += jump_size
                elif event_type.startswith('-'):
                    jump_size = int(event_type[1:])
                    spread -= jump_size
                
                # Update history and time
                event_history.append((t_new, event_type))
                t = t_new
            
            # Record final spread
            spreads.append(spread)
        
        # Return expected spread
        return np.mean(spreads)

class ACDPModel:
    """
    Autoregressive Conditional Double Poisson (ACDP) model for spread forecasting
    """
    
    def __init__(self, N=60):
        """
        Initialize the ACDP model
        
        Parameters:
        - N: Number of lags to consider
        """
        self.N = N
        self.c = 0.1
        self.alpha = 0.3
        self.beta = 0.6
        self.gamma = 1.0
    
    def fit(self, data, delta):
        """
        Fit the ACDP model to data
        
        Parameters:
        - data: List of (time, spread) tuples
        - delta: Time interval for subsampling
        
        Returns:
        - Self
        """
        # Subsample data at regular intervals
        times = np.array([t for t, _ in data])
        spreads = np.array([s for _, s in data])
        
        # Create regular time grid
        t_min, t_max = times[0], times[-1]
        grid = np.arange(t_min, t_max, delta)
        
        # Find spreads at grid points (using last observed value)
        subsampled_spreads = []
        for t in grid:
            idx = np.searchsorted(times, t) - 1
            if idx >= 0:
                subsampled_spreads.append(spreads[idx])
            else:
                subsampled_spreads.append(1)  # Default
        
        # Convert to S' = S - 1 (as in the paper)
        S_prime = np.array(subsampled_spreads) - 1
        
        # Simple parameter estimation (using method of moments)
        mean_S = np.mean(S_prime)
        var_S = np.var(S_prime)
        
        if len(S_prime) > 1:
            # Estimate parameters
            self.c = mean_S * (1 - self.beta) * 0.5
            self.alpha = 0.5 * (1 - self.beta)
            # beta is kept at default
            self.gamma = mean_S / var_S if var_S > 0 else 1.0
        
        return self
    
    def forecast(self, data, delta):
        """
        Forecast the next spread value
        
        Parameters:
        - data: List of (time, spread) tuples
        - delta: Time interval
        
        Returns:
        - forecasted_spread: Expected spread at next time step
        """
        # Extract recent history
        history = [(t, s) for t, s in data]
        
        # Subsample data at regular intervals
        times = np.array([t for t, _ in history])
        spreads = np.array([s for _, s in history])
        
        # Create regular time grid
        t_min, t_max = times[0], times[-1]
        grid = np.arange(t_min, t_max, delta)
        
        # Find spreads at grid points
        subsampled_spreads = []
        for t in grid:
            idx = np.searchsorted(times, t) - 1
            if idx >= 0:
                subsampled_spreads.append(spreads[idx])
            else:
                subsampled_spreads.append(1)  # Default
        
        # Convert to S' = S - 1
        S_prime = np.array(subsampled_spreads) - 1
        
        # Compute lambda using AR structure
        if len(S_prime) > 0:
            # Initialize lambda
            lambda_t = self.c
            
            # Add AR terms (up to N lags)
            for i in range(min(self.N, len(S_prime))):
                lambda_t += self.beta**i * (self.c + self.alpha * S_prime[-(i+1)])
            
            # Return E[S] = lambda + 1
            return lambda_t + 1
        else:
            return 1  # Default
        

def analyze_model(model, T=5000, seed=42):
    """
    Analyze the statistical properties of the model
    
    Parameters:
    - model: SDSH model
    - T: Simulation time
    - seed: Random seed
    
    Returns:
    - Dictionary of analysis results
    """
    # Simulate the model
    times, events, spread_path = model.simulate(T, seed)
    
    # Extract data
    event_times = np.array(times)
    event_types = np.array(events)
    spread_times = np.array([t for t, _ in spread_path])
    spread_values = np.array([s for _, s in spread_path])
    
    # 1. Compute spread distribution (calendar time)
    time_grid = np.linspace(0, T, 1000)
    calendar_spread = []
    
    for t in time_grid:
        idx = np.searchsorted(spread_times, t) - 1
        if idx >= 0:
            calendar_spread.append(spread_values[idx])
        else:
            calendar_spread.append(spread_values[0])
    
    calendar_dist = np.bincount(calendar_spread) / len(calendar_spread)
    
    # 2. Compute spread distribution (event time)
    event_dist = np.bincount(spread_values[:-1]) / (len(spread_values) - 1)
    
    # 3. Compute inter-event time distribution
    inter_event_times = np.diff(event_times)
    
    # 4. Compute spread autocorrelation
    # Sample spread at regular intervals
    sample_times = np.linspace(0, T, 1000)
    sampled_spread = []
    
    for t in sample_times:
        idx = np.searchsorted(spread_times, t) - 1
        if idx >= 0:
            sampled_spread.append(spread_values[idx])
        else:
            sampled_spread.append(spread_values[0])
    
    sampled_spread = np.array(sampled_spread)
    acf = []
    
    for lag in range(1, 100):
        if lag < len(sampled_spread):
            # Compute autocorrelation at lag
            corr = np.corrcoef(sampled_spread[:-lag], sampled_spread[lag:])[0, 1]
            acf.append(corr)
        else:
            acf.append(0)
    
    # 5. Compute autocovariance of spread increments
    spread_increments = np.diff(sampled_spread)
    acv = []
    
    for lag in range(1, 100):
        if lag < len(spread_increments):
            # Compute autocovariance at lag
            cov = np.cov(spread_increments[:-lag], spread_increments[lag:])[0, 1]
            acv.append(cov)
        else:
            acv.append(0)
    
    return {
        'times': times,
        'events': events,
        'spread_path': spread_path,
        'calendar_dist': calendar_dist,
        'event_dist': event_dist,
        'inter_event_times': inter_event_times,
        'acf': acf,
        'acv': acv
    }

def test_forecasting(model, T=5000, delta=30, seed=42):
    """
    Test the forecasting performance of the model
    
    Parameters:
    - model: SDSH model
    - T: Simulation time
    - delta: Forecast horizon (seconds)
    - seed: Random seed
    
    Returns:
    - MSE for different predictors
    """
    # Simulate the model
    times, events, spread_path = model.simulate(T, seed)
    
    # Convert to format for forecasting
    history = [(t, e, s) for (t, e), (_, s) in zip(zip(times, events), spread_path[1:])]
    
    # Create test points
    test_points = np.linspace(T/2, T-delta, 50)  # Test in second half of simulation
    
    # Initialize predictors
    sdsh_predictor = model  # Same model for simplicity
    acdp_model = ACDPModel()
    
    # Test predictions
    last_errors = []
    acdp_errors = []
    sdsh_errors = []
    
    for t0 in test_points:
        # Get history up to t0
        t0_history = [(t, e, s) for t, e, s in history if t < t0]
        
        if not t0_history:
            continue
            
        # Get actual spread at t0 + delta
        actual_idx = np.searchsorted(spread_times, t0 + delta) - 1
        if actual_idx >= 0:
            actual_spread = spread_values[actual_idx]
        else:
            continue
        
        # Last predictor: S_t0
        last_spread = t0_history[-1][2]
        last_error = (last_spread - actual_spread) ** 2
        last_errors.append(last_error)
        
        # ACDP predictor
        acdp_model.fit([(t, s) for t, _, s in t0_history], delta)
        acdp_spread = acdp_model.forecast([(t, s) for t, _, s in t0_history], delta)
        acdp_error = (acdp_spread - actual_spread) ** 2
        acdp_errors.append(acdp_error)
        
        # SDSH predictor
        sdsh_spread = sdsh_predictor.forecast(t0_history, t0, delta)
        sdsh_error = (sdsh_spread - actual_spread) ** 2
        sdsh_errors.append(sdsh_error)
    
    # Compute MSE
    last_mse = np.mean(last_errors) if last_errors else np.nan
    acdp_mse = np.mean(acdp_errors) if acdp_errors else np.nan
    sdsh_mse = np.mean(sdsh_errors) if sdsh_errors else np.nan
    
    return {
        'Last': last_mse,
        'ACDP': acdp_mse,
        'SDSH': sdsh_mse
    }

def run_simulation_tests():
    """
    Run tests on simulated data
    """
    # Initialize model
    model = SDSHModel(K=2, L=3, max_spread=5)
    
    # Set specific parameters for illustration
    model.mu = {"+1": 0.3, "+2": 0.1, "-1": 0.2, "-2": 0.1}
    
    # Set kernel parameters
    # For simplicity, we'll use a few key kernel shapes
    for e in model.E:
        for e_prime in model.E:
            # Excitation between opposite jump directions
            if (e.startswith('+') and e_prime.startswith('-')) or (e.startswith('-') and e_prime.startswith('+')):
                model.alpha[(e, e_prime, 0)] = 0.2
                model.alpha[(e, e_prime, 1)] = 0.1
                model.alpha[(e, e_prime, 2)] = 0.05
            # Self-excitation
            elif e == e_prime:
                model.alpha[(e, e_prime, 0)] = 0.1
                model.alpha[(e, e_prime, 1)] = 0.05
                model.alpha[(e, e_prime, 2)] = 0.02
            # Weaker cross-excitation
            else:
                model.alpha[(e, e_prime, 0)] = 0.05
                model.alpha[(e, e_prime, 1)] = 0.02
                model.alpha[(e, e_prime, 2)] = 0.01
    
    # Set state-dependent functions
    for s in range(1, model.max_spread + 1):
        # Cannot have negative spread
        model.f["-1"][1] = 0
        model.f["-2"][1] = 0
        model.f["-2"][2] = 0
        
        # Downward pressure when spread is large
        if s >= 2:
            model.f["+1"][s] = 1.0 / s
            model.f["+2"][s] = 0.5 / s
        
        if s >= 3:
            model.f["-1"][s] = min(5.0, s / 2)
            
        if s >= 4:
            model.f["-2"][s] = min(5.0, s / 3)
    
    # Analyze model
    print("Analyzing model...")
    results = analyze_model(model, T=5000)
    
    # Plot results
    plot_results(model, results)
    
    # Test forecasting
    print("\nTesting forecasting...")
    forecast_results = test_forecasting(model)
    
    print("\nForecasting MSE:")
    for predictor, mse in forecast_results.items():
        print(f"{predictor}: {mse:.6f}")
    
    # Compare with estimation
    print("\nTesting parameter estimation...")
    # Generate synthetic data
    times, events, spread_path = model.simulate(T=1000)
    
    # Convert to format for estimation
    data = []
    for i, (t, e) in enumerate(zip(times, events)):
        if i < len(spread_path) - 1:
            s = spread_path[i+1][1]  # Spread after event
            data.append((t, e, s))
    
    # Create new model for estimation
    est_model = SDSHModel(K=2, L=3, max_spread=5)
    
    # Fit model
    est_model.fit(data)
    
    # Compare parameters
    print("\nComparing original vs. estimated parameters:")
    print("Original mu:", {e: model.mu[e] for e in model.E})
    print("Estimated mu:", {e: est_model.mu[e] for e in est_model.E})
    
    print("\nOriginal f(\"+1\"):", model.f["+1"][:6])
    print("Estimated f(\"+1\"):", est_model.f["+1"][:6])
    
    print("\nOriginal f(\"-1\"):", model.f["-1"][:6])
    print("Estimated f(\"-1\"):", est_model.f["-1"][:6])

def plot_results(model, results):
    """
    Plot the analysis results
    
    Parameters:
    - model: SDSH model
    - results: Dictionary of analysis results
    """
    # Create figure
    plt.figure(figsize=(15, 12))
    
    # 1. Plot spread path
    plt.subplot(3, 2, 1)
    spread_times = [t for t, _ in results['spread_path']]
    spread_values = [s for _, s in results['spread_path']]
    plt.step(spread_times, spread_values, where='post')
    plt.title('Spread Path')
    plt.xlabel('Time')
    plt.ylabel('Spread')
    plt.grid(True)
    
    # 2. Plot spread distributions
    plt.subplot(3, 2, 2)
    max_spread = max(len(results['calendar_dist']), len(results['event_dist']))
    x = np.arange(1, max_spread + 1)
    
    cal_dist = np.zeros(max_spread)
    cal_dist[:len(results['calendar_dist'])-1] = results['calendar_dist'][1:]
    
    evt_dist = np.zeros(max_spread)
    evt_dist[:len(results['event_dist'])-1] = results['event_dist'][1:]
    
    plt.bar(x - 0.2, cal_dist, width=0.4, label='Calendar Time')
    plt.bar(x + 0.2, evt_dist, width=0.4, label='Event Time')
    plt.title('Spread Distributions')
    plt.xlabel('Spread')
    plt.ylabel('Probability')
    plt.legend()
    plt.grid(True)
    
    # 3. Plot inter-event time distribution
    plt.subplot(3, 2, 3)
    plt.hist(results['inter_event_times'], bins=50, density=True, alpha=0.7)
    plt.title('Inter-event Time Distribution')
    plt.xlabel('Time')
    plt.ylabel('Density')
    plt.grid(True)
    
    # 4. Plot spread autocorrelation
    plt.subplot(3, 2, 4)
    plt.plot(results['acf'])
    plt.title('Spread Autocorrelation')
    plt.xlabel('Lag')
    plt.ylabel('ACF')
    plt.grid(True)
    
    # 5. Plot f functions
    plt.subplot(3, 2, 5)
    max_spread = model.max_spread
    x = np.arange(1, max_spread + 1)
    
    for e in model.E:
        plt.plot(x, model.f[e][1:max_spread+1], label=f'f_{e}')
    
    plt.title('State-dependent Functions f')
    plt.xlabel('Spread')
    plt.ylabel('f value')
    plt.legend()
    plt.grid(True)
    
    # 6. Plot autocovariance of spread increments
    plt.subplot(3, 2, 6)
    plt.plot(results['acv'])
    plt.title('Autocovariance of Spread Increments')
    plt.xlabel('Lag')
    plt.ylabel('ACV')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('sdsh_model_analysis.png')
    plt.show()


