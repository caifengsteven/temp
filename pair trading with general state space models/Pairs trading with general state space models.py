import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm, t
from scipy.optimize import minimize
import datetime as dt
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# For Bloomberg data access
try:
    import pdblp
    from pdblp import BCon
    HAS_BLOOMBERG = True
    print("Bloomberg API available.")
except ImportError:
    HAS_BLOOMBERG = False
    print("Bloomberg API not available.")

class GeneralStateSpaceModel:
    """
    Implementation of general state space model for pairs trading
    as described in "Pairs trading with general state space models" by Guang Zhang
    """
    
    def __init__(self, model_type='linear_gaussian', params=None):
        """
        Initialize the state space model
        
        Parameters:
        -----------
        model_type : str
            Type of model ('linear_gaussian', 'nonlinear_mean', 'heteroscedastic', 'non_gaussian')
        params : dict
            Parameters for the model
        """
        self.model_type = model_type
        
        # Default parameters for different model types
        if params is None:
            if model_type == 'linear_gaussian':
                # Model I: xt+1 = θ0 + θ1*xt + θ2*ηt, ηt ~ N(0, 1)
                self.params = {
                    'theta0': 0,  # Constant term
                    'theta1': 0.95,  # AR(1) coefficient
                    'theta2': 0.01,  # Volatility
                    'gamma': 1.0,  # Hedge ratio
                    'sigma_epsilon': 0.01  # Observation noise
                }
            elif model_type == 'heteroscedastic':
                # Model II: xt+1 = θ0 + θ1*xt + (θ2 + θ3*x²_t)*ηt, ηt ~ N(0, 1)
                self.params = {
                    'theta0': 0,  # Constant term
                    'theta1': 0.95,  # AR(1) coefficient
                    'theta2': 0.01,  # Base volatility
                    'theta3': 0.1,  # ARCH term
                    'gamma': 1.0,  # Hedge ratio
                    'sigma_epsilon': 0.01  # Observation noise
                }
            elif model_type == 'nonlinear_mean':
                # xt+1 = θ0 + θ1*xt + θ2*x²_t + θ3*ηt, ηt ~ N(0, 1)
                self.params = {
                    'theta0': 0,  # Constant term
                    'theta1': 0.95,  # AR(1) coefficient
                    'theta2': 0.1,  # Nonlinear term
                    'theta3': 0.01,  # Volatility
                    'gamma': 1.0,  # Hedge ratio
                    'sigma_epsilon': 0.01  # Observation noise
                }
            elif model_type == 'non_gaussian':
                # xt+1 = θ0 + θ1*xt + θ2*ηt, ηt ~ t-distribution
                self.params = {
                    'theta0': 0,  # Constant term
                    'theta1': 0.95,  # AR(1) coefficient
                    'theta2': 0.01,  # Volatility
                    'gamma': 1.0,  # Hedge ratio
                    'sigma_epsilon': 0.01,  # Observation noise
                    'df': 3  # Degrees of freedom for t-distribution
                }
        else:
            self.params = params
    
    def f(self, x):
        """
        Conditional mean function for the state equation
        
        Parameters:
        -----------
        x : array-like
            Current state (spread)
            
        Returns:
        --------
        array-like
            Expected next state
        """
        if self.model_type == 'linear_gaussian' or self.model_type == 'heteroscedastic' or self.model_type == 'non_gaussian':
            return self.params['theta0'] + self.params['theta1'] * x
        elif self.model_type == 'nonlinear_mean':
            return self.params['theta0'] + self.params['theta1'] * x + self.params['theta2'] * x**2
    
    def g(self, x):
        """
        Volatility function for the state equation
        
        Parameters:
        -----------
        x : array-like
            Current state (spread)
            
        Returns:
        --------
        array-like
            Volatility for next state
        """
        if self.model_type == 'linear_gaussian' or self.model_type == 'non_gaussian':
            return self.params['theta2'] * np.ones_like(x)
        elif self.model_type == 'nonlinear_mean':
            return self.params['theta3'] * np.ones_like(x)
        elif self.model_type == 'heteroscedastic':
            return np.sqrt(self.params['theta2']**2 + self.params['theta3']**2 * x**2)
    
    def simulate_spread(self, n_steps, x0=0, seed=None):
        """
        Simulate the spread process
        
        Parameters:
        -----------
        n_steps : int
            Number of steps to simulate
        x0 : float
            Initial value of the spread
        seed : int
            Random seed
            
        Returns:
        --------
        array-like
            Simulated spread process
        """
        if seed is not None:
            np.random.seed(seed)
        
        x = np.zeros(n_steps + 1)
        x[0] = x0
        
        for t in range(n_steps):
            if self.model_type == 'non_gaussian':
                # t-distribution noise
                eta = t.rvs(df=self.params['df']) * np.sqrt((self.params['df'] - 2) / self.params['df'])
            else:
                # Gaussian noise
                eta = np.random.normal(0, 1)
            
            x[t+1] = self.f(x[t]) + self.g(x[t]) * eta
        
        return x


class QuasiMonteCarloKalmanFilter:
    """
    Implementation of Quasi Monte Carlo Kalman Filter for general state space models
    as described in "Pairs trading with general state space models" by Guang Zhang
    """
    
    def __init__(self, model, G=100, m=2):
        """
        Initialize the filter
        
        Parameters:
        -----------
        model : GeneralStateSpaceModel
            The state space model
        G : int
            Number of particles for Monte Carlo approximation
        m : int
            Number of Gaussian mixture components for approximating non-Gaussian distributions
        """
        self.model = model
        self.G = G
        self.m = m
        
    def generate_halton_sequence(self, n, base=2):
        """
        Generate Halton sequence for quasi-Monte Carlo sampling
        
        Parameters:
        -----------
        n : int
            Length of sequence
        base : int
            Base for Halton sequence
            
        Returns:
        --------
        array
            Halton sequence
        """
        result = np.zeros(n)
        for i in range(n):
            f = 1
            r = 0
            j = i + 1
            while j > 0:
                f = f / base
                r = r + f * (j % base)
                j = j // base
            result[i] = r
        return result
    
    def box_muller_transform(self, u1, u2):
        """
        Box-Muller transform to convert uniform to standard normal
        
        Parameters:
        -----------
        u1, u2 : array-like
            Uniform random variables
            
        Returns:
        --------
        tuple
            Standard normal random variables
        """
        z1 = np.sqrt(-2 * np.log(u1)) * np.cos(2 * np.pi * u2)
        z2 = np.sqrt(-2 * np.log(u1)) * np.sin(2 * np.pi * u2)
        return z1, z2
    
    def filter(self, PA, PB):
        """
        Run the filter to estimate the unobservable spread
        
        Parameters:
        -----------
        PA : array-like
            Price of asset A
        PB : array-like
            Price of asset B
            
        Returns:
        --------
        dict
            Dictionary with filtered spread and related information
        """
        T = len(PA)
        gamma = self.model.params['gamma']
        sigma_epsilon = self.model.params['sigma_epsilon']
        
        # Initialize
        filtered_spread = np.zeros(T)
        filtered_variance = np.zeros(T)
        
        # Initial distribution for x0
        mu0 = 0
        sigma0 = 0.1
        
        # Generate quasi-Monte Carlo samples for x0
        u1 = self.generate_halton_sequence(self.G, base=2)
        u2 = self.generate_halton_sequence(self.G, base=3)
        z1, z2 = self.box_muller_transform(u1, u2)
        x0_samples = mu0 + sigma0 * z1
        
        # Storage for current samples, mean and variance
        x_samples = x0_samples
        x_mean = mu0
        x_var = sigma0**2
        
        # Main filtering loop
        for t in range(T):
            # 1. Prediction step
            x_pred_samples = np.zeros((self.m, self.G))
            c = np.zeros(self.m)
            Q = np.zeros(self.m)
            
            for i in range(self.m):
                # Propagate particles through the state equation
                for g in range(self.G):
                    if self.model.model_type == 'non_gaussian':
                        # For non-Gaussian noise, use a mixture of Gaussians
                        eta = norm.ppf((g + 0.5) / self.G) * (i + 1) / self.m
                    else:
                        # For Gaussian noise, use standard normal
                        eta = norm.ppf((g + 0.5) / self.G)
                    
                    x_pred_samples[i, g] = self.model.f(x_samples[g]) + self.model.g(x_samples[g]) * eta
                
                # Compute moments
                c[i] = np.mean(x_pred_samples[i])
                Q[i] = np.var(x_pred_samples[i])
            
            # 2. Update step
            P_pred = np.zeros(self.m)
            K = np.zeros(self.m)
            V = np.zeros(self.m)
            S = np.zeros(self.m)
            b = np.zeros(self.m)
            beta = np.zeros(self.m)
            
            for i in range(self.m):
                # Generate observation samples
                PA_samples = np.zeros(self.G)
                for g in range(self.G):
                    PA_samples[g] = x_pred_samples[i, g] + gamma * PB[t] + np.random.normal(0, sigma_epsilon)
                
                # Compute observation moments
                PA_mean = np.mean(PA_samples)
                V[i] = np.var(PA_samples) + sigma_epsilon**2
                S[i] = np.mean((x_pred_samples[i] - c[i]) * (PA_samples - PA_mean))
                
                # Kalman gain and filtered moments
                K[i] = S[i] / V[i]
                P_pred[i] = Q[i] - K[i]**2 * V[i]
                b[i] = c[i] + K[i] * (PA[t] - PA_mean)
                
                # Component weight
                beta[i] = norm.pdf(PA[t], PA_mean, np.sqrt(V[i]))
            
            # Normalize weights
            beta = beta / np.sum(beta)
            
            # Compute overall filtered mean and variance
            x_mean = np.sum(beta * b)
            x_var = np.sum(beta * (P_pred + b**2)) - x_mean**2
            
            # Store filtered values
            filtered_spread[t] = x_mean
            filtered_variance[t] = x_var
            
            # Generate new samples for the next iteration
            u1 = self.generate_halton_sequence(self.G, base=2)
            u2 = self.generate_halton_sequence(self.G, base=3)
            z1, z2 = self.box_muller_transform(u1, u2)
            x_samples = x_mean + np.sqrt(x_var) * z1
        
        return {
            'filtered_spread': filtered_spread,
            'filtered_variance': filtered_variance
        }
    
    def estimate_parameters(self, PA, PB, init_params=None, method='grid_search'):
        """
        Estimate model parameters using maximum likelihood
        
        Parameters:
        -----------
        PA : array-like
            Price of asset A
        PB : array-like
            Price of asset B
        init_params : dict
            Initial parameters for optimization
        method : str
            Method for parameter estimation ('grid_search' or 'optimization')
            
        Returns:
        --------
        dict
            Estimated parameters
        """
        if init_params is None:
            init_params = self.model.params
        
        if method == 'grid_search':
            # Simple grid search for parameters
            best_params = init_params.copy()
            best_likelihood = -np.inf
            
            # Grid for each parameter
            gamma_grid = np.linspace(0.5, 2.0, 10)
            theta0_grid = np.linspace(-0.01, 0.01, 5)
            theta1_grid = np.linspace(0.8, 0.99, 10)
            
            if self.model.model_type == 'linear_gaussian' or self.model.model_type == 'non_gaussian':
                theta2_grid = np.linspace(0.001, 0.02, 5)
                
                for gamma in tqdm(gamma_grid, desc="Estimating parameters"):
                    for theta0 in theta0_grid:
                        for theta1 in theta1_grid:
                            for theta2 in theta2_grid:
                                params = {
                                    'gamma': gamma,
                                    'theta0': theta0,
                                    'theta1': theta1,
                                    'theta2': theta2,
                                    'sigma_epsilon': init_params['sigma_epsilon']
                                }
                                
                                if self.model.model_type == 'non_gaussian':
                                    params['df'] = init_params['df']
                                
                                # Set parameters and compute likelihood
                                self.model.params = params
                                likelihood = self.compute_likelihood(PA, PB)
                                
                                if likelihood > best_likelihood:
                                    best_likelihood = likelihood
                                    best_params = params.copy()
            
            elif self.model.model_type == 'heteroscedastic':
                theta2_grid = np.linspace(0.001, 0.02, 5)
                theta3_grid = np.linspace(0.01, 0.3, 5)
                
                for gamma in tqdm(gamma_grid, desc="Estimating parameters"):
                    for theta0 in theta0_grid:
                        for theta1 in theta1_grid:
                            for theta2 in theta2_grid:
                                for theta3 in theta3_grid:
                                    params = {
                                        'gamma': gamma,
                                        'theta0': theta0,
                                        'theta1': theta1,
                                        'theta2': theta2,
                                        'theta3': theta3,
                                        'sigma_epsilon': init_params['sigma_epsilon']
                                    }
                                    
                                    # Set parameters and compute likelihood
                                    self.model.params = params
                                    likelihood = self.compute_likelihood(PA, PB)
                                    
                                    if likelihood > best_likelihood:
                                        best_likelihood = likelihood
                                        best_params = params.copy()
            
            elif self.model.model_type == 'nonlinear_mean':
                theta2_grid = np.linspace(-0.2, 0.3, 5)
                theta3_grid = np.linspace(0.001, 0.02, 5)
                
                for gamma in tqdm(gamma_grid, desc="Estimating parameters"):
                    for theta0 in theta0_grid:
                        for theta1 in theta1_grid:
                            for theta2 in theta2_grid:
                                for theta3 in theta3_grid:
                                    params = {
                                        'gamma': gamma,
                                        'theta0': theta0,
                                        'theta1': theta1,
                                        'theta2': theta2,
                                        'theta3': theta3,
                                        'sigma_epsilon': init_params['sigma_epsilon']
                                    }
                                    
                                    # Set parameters and compute likelihood
                                    self.model.params = params
                                    likelihood = self.compute_likelihood(PA, PB)
                                    
                                    if likelihood > best_likelihood:
                                        best_likelihood = likelihood
                                        best_params = params.copy()
            
            # Set the best parameters
            self.model.params = best_params
            return best_params
        
        else:
            # Optimization method (not implemented in detail)
            # In practice, this would use scipy.optimize to minimize negative log likelihood
            print("Optimization method not fully implemented. Using grid search instead.")
            return self.estimate_parameters(PA, PB, init_params, method='grid_search')
    
    def compute_likelihood(self, PA, PB):
        """
        Compute log likelihood
        
        Parameters:
        -----------
        PA : array-like
            Price of asset A
        PB : array-like
            Price of asset B
            
        Returns:
        --------
        float
            Log likelihood
        """
        # Run the filter
        filter_results = self.filter(PA, PB)
        
        # Extract filtered values
        filtered_spread = filter_results['filtered_spread']
        filtered_variance = filter_results['filtered_variance']
        
        # Compute log likelihood
        gamma = self.model.params['gamma']
        sigma_epsilon = self.model.params['sigma_epsilon']
        
        log_likelihood = 0
        for t in range(1, len(PA)):
            # One-step ahead prediction
            pred_mean = self.model.f(filtered_spread[t-1])
            pred_var = self.model.g(filtered_spread[t-1])**2 + filtered_variance[t-1]
            
            # Observation prediction
            obs_mean = pred_mean + gamma * PB[t]
            obs_var = pred_var + sigma_epsilon**2
            
            # Log likelihood contribution
            log_likelihood += norm.logpdf(PA[t], obs_mean, np.sqrt(obs_var))
        
        return log_likelihood


class PairsTrading:
    """
    Implementation of pairs trading strategies as described in 
    "Pairs trading with general state space models" by Guang Zhang
    """
    
    def __init__(self, model_type='linear_gaussian', strategy='C', transaction_cost=0.002, risk_free_rate=0.02):
        """
        Initialize the pairs trading strategy
        
        Parameters:
        -----------
        model_type : str
            Type of model ('linear_gaussian', 'heteroscedastic')
        strategy : str
            Trading strategy ('A', 'B', or 'C')
        transaction_cost : float
            Transaction cost in decimal (e.g., 0.002 for 20 bps)
        risk_free_rate : float
            Annualized risk-free rate in decimal
        """
        self.model_type = model_type
        self.strategy = strategy
        self.transaction_cost = transaction_cost
        self.risk_free_rate = risk_free_rate
        
        # Initialize model
        if model_type == 'linear_gaussian':
            self.model = GeneralStateSpaceModel(model_type='linear_gaussian')
        elif model_type == 'heteroscedastic':
            self.model = GeneralStateSpaceModel(model_type='heteroscedastic')
        else:
            raise ValueError("Model type not recognized")
        
        # Initialize filter
        self.filter = QuasiMonteCarloKalmanFilter(self.model)
        
        # Trading parameters
        self.upper_threshold = None
        self.lower_threshold = None
        self.position = 0  # 0: no position, 1: long A/short B, -1: short A/long B
        
    def load_data_from_bloomberg(self, tickers, start_date, end_date, field='PX_LAST'):
        """
        Load price data from Bloomberg
        
        Parameters:
        -----------
        tickers : list
            List of two Bloomberg tickers [ticker_A, ticker_B]
        start_date : str
            Start date in format 'YYYY-MM-DD'
        end_date : str
            End date in format 'YYYY-MM-DD'
        field : str, optional
            Bloomberg field to retrieve
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with price data for both assets
        """
        print(f"Connecting to Bloomberg to retrieve {field} for {tickers}")
        
        try:
            # Initialize connection
            con = BCon(timeout=60000)
            con.start()
            
            # Format dates for Bloomberg query
            start_date_fmt = start_date.replace('-', '')
            end_date_fmt = end_date.replace('-', '')
            
            # Request data from Bloomberg
            data = con.bdh(tickers=tickers, 
                           flds=[field], 
                           start_date=start_date_fmt, 
                           end_date=end_date_fmt)
            
            # Close connection
            con.stop()
            
            # Create DataFrame for prices
            prices_df = pd.DataFrame(index=pd.date_range(start=start_date, end=end_date, freq='B'))
            
            # Extract data for each ticker
            for ticker in tickers:
                try:
                    ticker_data = data.xs(ticker, axis=1, level=0)
                    if field in ticker_data.columns:
                        prices_df[ticker] = ticker_data[field]
                except:
                    print(f"Error extracting data for {ticker}")
            
            # Drop rows with missing values
            prices_df = prices_df.dropna()
            
            print(f"Successfully downloaded data with shape: {prices_df.shape}")
            return prices_df
            
        except Exception as e:
            print(f"Error retrieving data from Bloomberg: {e}")
            
            # Try alternative method
            try:
                con = BCon(timeout=60000)
                con.start()
                
                prices_df = pd.DataFrame(index=pd.date_range(start=start_date, end=end_date, freq='B'))
                
                for ticker in tickers:
                    print(f"Trying to retrieve data for {ticker}...")
                    ticker_data = con.bdh(ticker, [field], start_date_fmt, end_date_fmt)
                    
                    if isinstance(ticker_data, pd.DataFrame) and not ticker_data.empty:
                        if isinstance(ticker_data.columns, pd.MultiIndex):
                            try:
                                prices_df[ticker] = ticker_data.xs(ticker, axis=1, level=0)[field]
                            except:
                                print(f"Could not extract data for {ticker}")
                        else:
                            try:
                                prices_df[ticker] = ticker_data[field]
                            except:
                                print(f"Could not extract data for {ticker}")
                
                con.stop()
                prices_df = prices_df.dropna()
                
                print(f"Successfully downloaded data with shape: {prices_df.shape}")
                return prices_df
                
            except Exception as e2:
                print(f"Second attempt also failed: {e2}")
                raise e2
    
    def fit(self, prices_df, train_ratio=0.7):
        """
        Fit the model and find optimal trading thresholds
        
        Parameters:
        -----------
        prices_df : pd.DataFrame
            DataFrame with price data for both assets
        train_ratio : float
            Ratio of data to use for training
            
        Returns:
        --------
        self
            Fitted model
        """
        # Split data into training and testing
        n = len(prices_df)
        train_size = int(n * train_ratio)
        
        train_data = prices_df.iloc[:train_size]
        test_data = prices_df.iloc[train_size:]
        
        # Extract prices
        tickers = prices_df.columns
        ticker_A, ticker_B = tickers[0], tickers[1]
        
        PA_train = train_data[ticker_A].values
        PB_train = train_data[ticker_B].values
        
        # Estimate model parameters
        self.filter.estimate_parameters(PA_train, PB_train)
        
        # Filter the spread
        filter_results = self.filter.filter(PA_train, PB_train)
        filtered_spread = filter_results['filtered_spread']
        filtered_std = np.sqrt(filter_results['filtered_variance'])
        
        # Find optimal trading thresholds using simulation-based method
        self.find_optimal_thresholds(filtered_spread, filtered_std)
        
        return self
    
    def find_optimal_thresholds(self, filtered_spread, filtered_std, n_sims=1000):
        """
        Find optimal trading thresholds using simulation-based method
        
        Parameters:
        -----------
        filtered_spread : array-like
            Filtered spread values
        filtered_std : array-like
            Filtered standard deviation of spread
        n_sims : int
            Number of simulations
            
        Returns:
        --------
        tuple
            Optimal upper and lower thresholds
        """
        # Compute mean of the filtered spread
        mean_spread = np.mean(filtered_spread)
        
        # Grid of threshold values (as multiples of standard deviation)
        upper_grid = np.arange(0.1, 2.6, 0.1)
        lower_grid = np.arange(-2.5, -0.0, 0.1)
        
        best_sharpe = -np.inf
        best_upper = None
        best_lower = None
        
        # Compute average standard deviation for homoscedastic case
        avg_std = np.mean(filtered_std)
        
        print("Finding optimal thresholds...")
        
        for upper_mult in tqdm(upper_grid):
            for lower_mult in lower_grid:
                # Set thresholds
                upper = mean_spread + upper_mult * avg_std
                lower = mean_spread + lower_mult * avg_std
                
                # Simulate trading with these thresholds
                cumulative_returns = []
                
                for _ in range(n_sims):
                    # Simulate spread
                    sim_spread = self.model.simulate_spread(len(filtered_spread), x0=filtered_spread[0])
                    
                    # Backtest with the simulated spread
                    if self.model_type == 'linear_gaussian':
                        # Fixed thresholds for homoscedastic model
                        returns = self.backtest_fixed_thresholds(
                            sim_spread, 
                            upper=upper, 
                            lower=lower, 
                            mean=mean_spread
                        )
                    else:
                        # Time-varying thresholds for heteroscedastic model
                        sim_std = np.zeros_like(sim_spread)
                        for t in range(1, len(sim_spread)):
                            sim_std[t] = self.model.g(sim_spread[t-1])
                        
                        returns = self.backtest_varying_thresholds(
                            sim_spread, 
                            upper_mult=upper_mult, 
                            lower_mult=lower_mult, 
                            std=sim_std,
                            mean=mean_spread
                        )
                    
                    # Compute cumulative return
                    cumulative_return = np.sum(returns)
                    cumulative_returns.append(cumulative_return)
                
                # Compute Sharpe ratio
                returns_array = np.array(cumulative_returns)
                mean_return = np.mean(returns_array)
                std_return = np.std(returns_array)
                
                if std_return > 0:
                    sharpe = mean_return / std_return
                else:
                    sharpe = 0
                
                # Update best thresholds
                if sharpe > best_sharpe:
                    best_sharpe = sharpe
                    best_upper = upper_mult
                    best_lower = lower_mult
        
        print(f"Optimal thresholds: Upper = {best_upper}σ, Lower = {best_lower}σ")
        
        # Store optimal thresholds
        self.upper_threshold = best_upper
        self.lower_threshold = best_lower
        
        return best_upper, best_lower
    
    def backtest_fixed_thresholds(self, spread, upper, lower, mean):
        """
        Backtest the strategy with fixed thresholds
        
        Parameters:
        -----------
        spread : array-like
            Spread values
        upper : float
            Upper threshold
        lower : float
            Lower threshold
        mean : float
            Mean spread (center threshold)
            
        Returns:
        --------
        array-like
            Returns from trading
        """
        n = len(spread)
        returns = np.zeros(n)
        position = 0  # 0: no position, 1: long A/short B, -1: short A/long B
        
        if self.strategy == 'A':
            # Strategy A: Open when crossing threshold, close when crossing mean
            for t in range(1, n):
                if position == 0:
                    if spread[t] >= upper:
                        # Open position: sell A, buy B
                        position = -1
                        # Account for transaction cost
                        returns[t] = -self.transaction_cost
                    elif spread[t] <= lower:
                        # Open position: buy A, sell B
                        position = 1
                        # Account for transaction cost
                        returns[t] = -self.transaction_cost
                elif position == -1:
                    if spread[t] <= mean:
                        # Close position
                        returns[t] = upper - spread[t] - self.transaction_cost
                        position = 0
                    else:
                        # Hold position
                        returns[t] = spread[t-1] - spread[t]
                elif position == 1:
                    if spread[t] >= mean:
                        # Close position
                        returns[t] = spread[t] - lower - self.transaction_cost
                        position = 0
                    else:
                        # Hold position
                        returns[t] = spread[t] - spread[t-1]
        
        elif self.strategy == 'B':
            # Strategy B: Open when crossing threshold, switch position when crossing other threshold
            for t in range(1, n):
                if position == 0:
                    if spread[t-1] < upper and spread[t] >= upper:
                        # Open position: sell A, buy B
                        position = -1
                        # Account for transaction cost
                        returns[t] = -self.transaction_cost
                    elif spread[t-1] > lower and spread[t] <= lower:
                        # Open position: buy A, sell B
                        position = 1
                        # Account for transaction cost
                        returns[t] = -self.transaction_cost
                elif position == -1:
                    if spread[t-1] > lower and spread[t] <= lower:
                        # Switch position
                        returns[t] = upper - spread[t] - 2 * self.transaction_cost
                        position = 1
                    else:
                        # Hold position
                        returns[t] = spread[t-1] - spread[t]
                elif position == 1:
                    if spread[t-1] < upper and spread[t] >= upper:
                        # Switch position
                        returns[t] = spread[t] - lower - 2 * self.transaction_cost
                        position = -1
                    else:
                        # Hold position
                        returns[t] = spread[t] - spread[t-1]
        
        elif self.strategy == 'C':
            # Strategy C: Open when crossing threshold from outside, close when crossing mean or other threshold
            for t in range(1, n):
                if position == 0:
                    if spread[t-1] > upper and spread[t] <= upper:
                        # Open position: buy A, sell B
                        position = 1
                        # Account for transaction cost
                        returns[t] = -self.transaction_cost
                    elif spread[t-1] < lower and spread[t] >= lower:
                        # Open position: sell A, buy B
                        position = -1
                        # Account for transaction cost
                        returns[t] = -self.transaction_cost
                elif position == -1:
                    if spread[t] >= mean or spread[t] > upper:
                        # Close position
                        returns[t] = spread[t-1] - spread[t] - self.transaction_cost
                        position = 0
                    else:
                        # Hold position
                        returns[t] = spread[t-1] - spread[t]
                elif position == 1:
                    if spread[t] <= mean or spread[t] < lower:
                        # Close position
                        returns[t] = spread[t] - spread[t-1] - self.transaction_cost
                        position = 0
                    else:
                        # Hold position
                        returns[t] = spread[t] - spread[t-1]
        
        return returns
    
    def backtest_varying_thresholds(self, spread, upper_mult, lower_mult, std, mean):
        """
        Backtest the strategy with time-varying thresholds for heteroscedastic model
        
        Parameters:
        -----------
        spread : array-like
            Spread values
        upper_mult : float
            Upper threshold multiplier (of standard deviation)
        lower_mult : float
            Lower threshold multiplier (of standard deviation)
        std : array-like
            Time-varying standard deviation
        mean : float
            Mean spread
            
        Returns:
        --------
        array-like
            Returns from trading
        """
        n = len(spread)
        returns = np.zeros(n)
        position = 0  # 0: no position, 1: long A/short B, -1: short A/long B
        
        # Time-varying thresholds
        upper = mean + upper_mult * std
        lower = mean + lower_mult * std
        
        if self.strategy == 'C':
            # Strategy C: Open when crossing threshold from outside, close when crossing mean or other threshold
            for t in range(1, n):
                if position == 0:
                    if spread[t-1] > upper[t-1] and spread[t] <= upper[t]:
                        # Open position: buy A, sell B
                        position = 1
                        # Account for transaction cost
                        returns[t] = -self.transaction_cost
                    elif spread[t-1] < lower[t-1] and spread[t] >= lower[t]:
                        # Open position: sell A, buy B
                        position = -1
                        # Account for transaction cost
                        returns[t] = -self.transaction_cost
                elif position == -1:
                    if spread[t] >= mean or spread[t] > upper[t]:
                        # Close position
                        returns[t] = spread[t-1] - spread[t] - self.transaction_cost
                        position = 0
                    else:
                        # Hold position
                        returns[t] = spread[t-1] - spread[t]
                elif position == 1:
                    if spread[t] <= mean or spread[t] < lower[t]:
                        # Close position
                        returns[t] = spread[t] - spread[t-1] - self.transaction_cost
                        position = 0
                    else:
                        # Hold position
                        returns[t] = spread[t] - spread[t-1]
        else:
            # Default to fixed threshold backtest
            returns = self.backtest_fixed_thresholds(spread, upper=mean + upper_mult * np.mean(std), 
                                               lower=mean + lower_mult * np.mean(std), mean=mean)
        
        return returns
    
    def backtest(self, prices_df, in_sample=False):
        """
        Backtest the strategy on price data
        
        Parameters:
        -----------
        prices_df : pd.DataFrame
            DataFrame with price data for both assets
        in_sample : bool
            Whether to backtest on in-sample data
            
        Returns:
        --------
        dict
            Backtest results
        """
        if self.upper_threshold is None or self.lower_threshold is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Extract prices
        tickers = prices_df.columns
        ticker_A, ticker_B = tickers[0], tickers[1]
        
        PA = prices_df[ticker_A].values
        PB = prices_df[ticker_B].values
        
        # Filter the spread
        filter_results = self.filter.filter(PA, PB)
        filtered_spread = filter_results['filtered_spread']
        filtered_std = np.sqrt(filter_results['filtered_variance'])
        
        # Compute mean of the filtered spread
        mean_spread = np.mean(filtered_spread)
        
        # Backtest with optimal thresholds
        if self.model_type == 'linear_gaussian':
            # Fixed thresholds for homoscedastic model
            avg_std = np.mean(filtered_std)
            upper = mean_spread + self.upper_threshold * avg_std
            lower = mean_spread + self.lower_threshold * avg_std
            
            returns = self.backtest_fixed_thresholds(
                filtered_spread, 
                upper=upper, 
                lower=lower, 
                mean=mean_spread
            )
        else:
            # Time-varying thresholds for heteroscedastic model
            returns = self.backtest_varying_thresholds(
                filtered_spread, 
                upper_mult=self.upper_threshold, 
                lower_mult=self.lower_threshold, 
                std=filtered_std,
                mean=mean_spread
            )
        
        # Compute performance metrics
        cumulative_return = np.sum(returns)
        annual_return = cumulative_return * 252 / len(returns)
        
        daily_return_series = pd.Series(returns)
        annual_std = daily_return_series.std() * np.sqrt(252)
        
        if annual_std > 0:
            sharpe_ratio = (annual_return - self.risk_free_rate) / annual_std
        else:
            sharpe_ratio = 0
        
        # Compute drawdown
        cum_returns = (1 + daily_return_series).cumprod()
        running_max = cum_returns.cummax()
        drawdown = (cum_returns / running_max - 1)
        max_drawdown = drawdown.min()
        
        if max_drawdown != 0:
            calmar_ratio = annual_return / abs(max_drawdown)
        else:
            calmar_ratio = 0
        
        # Compute pain index
        pain_index = abs(drawdown).mean()
        
        results = {
            'filtered_spread': filtered_spread,
            'filtered_std': filtered_std,
            'returns': returns,
            'cumulative_return': cumulative_return,
            'annual_return': annual_return,
            'annual_std': annual_std,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar_ratio,
            'pain_index': pain_index
        }
        
        return results
    
    def plot_backtest_results(self, prices_df, results):
        """
        Plot backtest results
        
        Parameters:
        -----------
        prices_df : pd.DataFrame
            DataFrame with price data for both assets
        results : dict
            Backtest results from backtest() method
        """
        # Extract data
        filtered_spread = results['filtered_spread']
        filtered_std = results['filtered_std']
        returns = results['returns']
        
        # Create figure with subplots
        fig, axs = plt.subplots(3, 1, figsize=(14, 14), sharex=True)
        
        # Plot prices
        axs[0].plot(prices_df.values, label=prices_df.columns)
        axs[0].set_title('Asset Prices')
        axs[0].legend()
        axs[0].grid(True)
        
        # Plot spread and thresholds
        mean_spread = np.mean(filtered_spread)
        dates = prices_df.index
        
        axs[1].plot(dates, filtered_spread, label='Filtered Spread', color='blue')
        
        if self.model_type == 'linear_gaussian':
            # Fixed thresholds for homoscedastic model
            avg_std = np.mean(filtered_std)
            upper = mean_spread + self.upper_threshold * avg_std
            lower = mean_spread + self.lower_threshold * avg_std
            
            axs[1].axhline(y=upper, color='r', linestyle='--', label=f'Upper Threshold ({self.upper_threshold}σ)')
            axs[1].axhline(y=lower, color='g', linestyle='--', label=f'Lower Threshold ({self.lower_threshold}σ)')
            axs[1].axhline(y=mean_spread, color='k', linestyle='-', label='Mean')
        else:
            # Time-varying thresholds for heteroscedastic model
            upper = mean_spread + self.upper_threshold * filtered_std
            lower = mean_spread + self.lower_threshold * filtered_std
            
            axs[1].plot(dates, upper, 'r--', label=f'Upper Threshold ({self.upper_threshold}σ)')
            axs[1].plot(dates, lower, 'g--', label=f'Lower Threshold ({self.lower_threshold}σ)')
            axs[1].axhline(y=mean_spread, color='k', linestyle='-', label='Mean')
        
        axs[1].set_title('Filtered Spread and Trading Thresholds')
        axs[1].legend()
        axs[1].grid(True)
        
        # Plot cumulative returns
        cumulative_returns = (1 + pd.Series(returns)).cumprod() - 1
        axs[2].plot(dates, cumulative_returns, label='Cumulative Returns', color='green')
        axs[2].set_title(f'Cumulative Returns (Annual: {results["annual_return"]:.2%}, Sharpe: {results["sharpe_ratio"]:.2f})')
        axs[2].legend()
        axs[2].grid(True)
        
        # Add performance metrics as text
        axs[2].text(0.01, 0.95, 
                   f'Annual Return: {results["annual_return"]:.2%}\n'
                   f'Annual Std: {results["annual_std"]:.2%}\n'
                   f'Sharpe Ratio: {results["sharpe_ratio"]:.2f}\n'
                   f'Max Drawdown: {results["max_drawdown"]:.2%}\n'
                   f'Calmar Ratio: {results["calmar_ratio"]:.2f}\n'
                   f'Pain Index: {results["pain_index"]:.4f}',
                   transform=axs[2].transAxes,
                   bbox=dict(facecolor='white', alpha=0.7))
        
        plt.tight_layout()
        plt.show()


def main():
    """
    Main function to run the pairs trading strategy
    """
    # Check if Bloomberg is available
    if not HAS_BLOOMBERG:
        print("Bloomberg API is not available.")
        return
    
    # Define pairs to test
    pairs_to_test = [
        # Example pairs from the paper
        ['PEP US Equity', 'KO US Equity'],  # PepsiCo vs Coca-Cola
        ['EWT US Equity', 'EWH US Equity'],  # Taiwan ETF vs Hong Kong ETF
        
        # Additional pairs
        ['JPM US Equity', 'BAC US Equity'],  # JP Morgan vs Bank of America
        ['CVX US Equity', 'XOM US Equity'],  # Chevron vs Exxon
        ['HD US Equity', 'LOW US Equity']    # Home Depot vs Lowe's
    ]
    
    # Define date range
    start_date = '2015-01-01'
    end_date = '2020-01-01'
    
    # Choose a pair from the list
    pair_index = 0  # Change this to test different pairs
    pair = pairs_to_test[pair_index]
    
    print(f"Testing pairs trading on {pair[0]} vs {pair[1]}")
    
    # Load data from Bloomberg
    try:
        pairs_trader = PairsTrading(model_type='heteroscedastic', strategy='C')
        prices_df = pairs_trader.load_data_from_bloomberg(pair, start_date, end_date)
        
        # Fit model and find optimal thresholds
        pairs_trader.fit(prices_df)
        
        # Backtest the strategy
        backtest_results = pairs_trader.backtest(prices_df)
        
        # Print results
        print("\nBacktest Results:")
        print(f"Annual Return: {backtest_results['annual_return']:.2%}")
        print(f"Annual Std Dev: {backtest_results['annual_std']:.2%}")
        print(f"Sharpe Ratio: {backtest_results['sharpe_ratio']:.2f}")
        print(f"Max Drawdown: {backtest_results['max_drawdown']:.2%}")
        print(f"Calmar Ratio: {backtest_results['calmar_ratio']:.2f}")
        print(f"Pain Index: {backtest_results['pain_index']:.4f}")
        
        # Plot results
        pairs_trader.plot_backtest_results(prices_df, backtest_results)
        
        # Compare with Model I (linear_gaussian) and Strategy A
        print("\nComparing with Model I (linear_gaussian) and Strategy A:")
        pairs_trader_baseline = PairsTrading(model_type='linear_gaussian', strategy='A')
        pairs_trader_baseline.fit(prices_df)
        baseline_results = pairs_trader_baseline.backtest(prices_df)
        
        print("\nBaseline Results (Model I, Strategy A):")
        print(f"Annual Return: {baseline_results['annual_return']:.2%}")
        print(f"Annual Std Dev: {baseline_results['annual_std']:.2%}")
        print(f"Sharpe Ratio: {baseline_results['sharpe_ratio']:.2f}")
        print(f"Max Drawdown: {baseline_results['max_drawdown']:.2%}")
        print(f"Calmar Ratio: {baseline_results['calmar_ratio']:.2f}")
        print(f"Pain Index: {baseline_results['pain_index']:.4f}")
        
        # Compare results
        print("\nImprovement (Model II, Strategy C vs Model I, Strategy A):")
        return_improvement = (backtest_results['annual_return'] - baseline_results['annual_return']) / abs(baseline_results['annual_return'])
        sharpe_improvement = (backtest_results['sharpe_ratio'] - baseline_results['sharpe_ratio']) / abs(baseline_results['sharpe_ratio'])
        
        print(f"Return Improvement: {return_improvement:.2%}")
        print(f"Sharpe Ratio Improvement: {sharpe_improvement:.2%}")
        
    except Exception as e:
        print(f"Error running pairs trading strategy: {e}")


if __name__ == "__main__":
    main()