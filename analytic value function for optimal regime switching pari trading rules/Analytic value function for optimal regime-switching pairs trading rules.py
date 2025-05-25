import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.linalg as la
import scipy.optimize as opt
from scipy import stats
from scipy.special import erf, erfi
import datetime as dt
import pdblp  # Bloomberg API
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from tqdm import tqdm
import warnings
import os
warnings.filterwarnings('ignore')

class MarkovModulatedOUModel:
    """
    Implementation of the Markov-Modulated Ornstein-Uhlenbeck (MMOU) process 
    for pairs trading with regime switching.
    
    Based on: "Analytic value function for optimal regime-switching pairs trading rules"
    by Yang Bai and Lan Wu (2018)
    """
    
    def __init__(self, n_states=2):
        """
        Initialize the MMOU model with n states
        
        Parameters:
        -----------
        n_states : int
            Number of states in the Markov chain
        """
        self.n_states = n_states
        
        # Parameters of the O-U process for each state
        self.lambda_vals = np.zeros(n_states)  # Mean reversion rates
        self.mu_vals = np.zeros(n_states)      # Long-term means
        self.sigma_vals = np.zeros(n_states)   # Volatilities
        
        # Transition probability matrix of the Markov chain
        self.transition_matrix = np.zeros((n_states, n_states))
        
        # Initial state probabilities
        self.initial_probs = np.ones(n_states) / n_states
        
        # Model fitted flag
        self.is_fitted = False
        
    def fit(self, spread_data, method='hmm'):
        """
        Fit the model parameters to historical spread data
        
        Parameters:
        -----------
        spread_data : array-like
            Historical time series of the spread
        method : str
            Method to use for parameter estimation ('hmm' or 'threshold')
        
        Returns:
        --------
        self : object
            Returns self
        """
        if method == 'hmm':
            # Using Hidden Markov Model approach
            self._fit_hmm(spread_data)
        elif method == 'threshold':
            # Using threshold method as described in the paper
            self._fit_threshold(spread_data)
        else:
            raise ValueError(f"Unknown method: {method}. Choose 'hmm' or 'threshold'")
        
        self.is_fitted = True
        return self
    
    def _fit_hmm(self, spread_data):
        """
        Fit model parameters using HMM approach (simplification for implementation)
        """
        # This is a simplified implementation - in practice, use a proper HMM library
        from sklearn.mixture import GaussianMixture
        
        # Ensure spread_data is valid
        spread_data = np.array(spread_data)
        valid_indices = ~np.isnan(spread_data) & ~np.isinf(spread_data)
        spread_data = spread_data[valid_indices]
        
        if len(spread_data) < 30:
            raise ValueError("Not enough valid data points for model fitting")
        
        # Prepare data: calculate returns to approximate discrete O-U process
        spread_returns = np.diff(spread_data)
        spread_levels = spread_data[:-1]
        
        # Fit Gaussian Mixture Model to identify regimes
        gmm = GaussianMixture(n_components=self.n_states, random_state=42)
        states = gmm.fit_predict(spread_returns.reshape(-1, 1))
        
        # Calculate transition probabilities
        transitions = np.zeros((self.n_states, self.n_states))
        for i in range(len(states) - 1):
            transitions[states[i], states[i+1]] += 1
            
        # Normalize to get probabilities
        for i in range(self.n_states):
            row_sum = np.sum(transitions[i, :])
            if row_sum > 0:
                transitions[i, :] /= row_sum
        
        self.transition_matrix = transitions
        
        # Estimate O-U parameters for each state
        for state in range(self.n_states):
            state_indices = (states == state)
            if np.sum(state_indices) > 10:  # Need enough data points
                X = spread_levels[state_indices].reshape(-1, 1)
                y = spread_returns[state_indices]
                
                # Linear regression to estimate parameters
                model = LinearRegression().fit(X, y)
                
                # Extract O-U parameters
                lambda_val = -model.coef_[0]
                mu_val = model.intercept_ / lambda_val if lambda_val > 0 else np.mean(spread_data)
                sigma_val = np.std(model.predict(X) - y)
                
                self.lambda_vals[state] = max(0.001, lambda_val)  # Ensure positive
                self.mu_vals[state] = mu_val
                self.sigma_vals[state] = max(0.001, sigma_val)   # Ensure positive
            else:
                # Not enough data for this state, use defaults
                self.lambda_vals[state] = 0.01
                self.mu_vals[state] = np.mean(spread_data)
                self.sigma_vals[state] = np.std(spread_returns)
    
    def _fit_threshold(self, spread_data):
        """
        Fit model parameters using threshold method as described in the paper
        """
        # Ensure spread_data is valid
        spread_data = np.array(spread_data)
        valid_indices = ~np.isnan(spread_data) & ~np.isinf(spread_data)
        spread_data = spread_data[valid_indices]
        
        if len(spread_data) < 30:
            raise ValueError("Not enough valid data points for model fitting")
            
        # Calculate returns to approximate discrete O-U process
        spread_returns = np.diff(spread_data)
        spread_levels = spread_data[:-1]
        
        # Grid search for optimal threshold
        best_threshold = None
        min_cls = float('inf')
        
        # Define the range of potential thresholds around the mean
        mean = np.mean(spread_data)
        std = np.std(spread_data)
        thresholds = np.linspace(mean - std, mean + std, 20)
        
        for threshold in thresholds:
            # Split data into two regimes
            regime_1 = spread_levels < threshold
            regime_2 = spread_levels >= threshold
            
            if np.sum(regime_1) < 10 or np.sum(regime_2) < 10:
                continue  # Skip if not enough data in either regime
            
            # Fit linear regression in each regime
            X1 = spread_levels[regime_1].reshape(-1, 1)
            y1 = spread_returns[regime_1]
            X2 = spread_levels[regime_2].reshape(-1, 1)
            y2 = spread_returns[regime_2]
            
            model1 = LinearRegression().fit(X1, y1)
            model2 = LinearRegression().fit(X2, y2)
            
            # Calculate conditional least squares
            pred1 = model1.predict(X1)
            pred2 = model2.predict(X2)
            cls = np.sum((y1 - pred1)**2) + np.sum((y2 - pred2)**2)
            
            if cls < min_cls:
                min_cls = cls
                best_threshold = threshold
        
        if best_threshold is None:
            # Fall back to simple method if threshold search fails
            self._fit_hmm(spread_data)
            return
            
        # Split data based on best threshold
        regime_1 = spread_levels < best_threshold
        regime_2 = spread_levels >= best_threshold
        
        # Fit O-U parameters for each regime
        X1 = spread_levels[regime_1].reshape(-1, 1)
        y1 = spread_returns[regime_1]
        X2 = spread_levels[regime_2].reshape(-1, 1)
        y2 = spread_returns[regime_2]
        
        model1 = LinearRegression().fit(X1, y1)
        model2 = LinearRegression().fit(X2, y2)
        
        # Extract O-U parameters for regime 1
        lambda_1 = -model1.coef_[0]
        mu_1 = model1.intercept_ / lambda_1 if lambda_1 > 0 else np.mean(spread_data[regime_1])
        sigma_1 = np.std(model1.predict(X1) - y1)
        
        # Extract O-U parameters for regime 2
        lambda_2 = -model2.coef_[0]
        mu_2 = model2.intercept_ / lambda_2 if lambda_2 > 0 else np.mean(spread_data[regime_2])
        sigma_2 = np.std(model2.predict(X2) - y2)
        
        # Ensure parameters are valid
        self.lambda_vals = np.array([max(0.001, lambda_1), max(0.001, lambda_2)])
        self.mu_vals = np.array([mu_1, mu_2])
        self.sigma_vals = np.array([max(0.001, sigma_1), max(0.001, sigma_2)])
        
        # Calculate transition probabilities
        states = np.zeros(len(spread_levels), dtype=int)
        states[regime_2] = 1
        
        transitions = np.zeros((2, 2))
        for i in range(len(states) - 1):
            transitions[states[i], states[i+1]] += 1
            
        # Normalize to get probabilities
        for i in range(2):
            row_sum = np.sum(transitions[i, :])
            if row_sum > 0:
                transitions[i, :] /= row_sum
        
        self.transition_matrix = transitions
    
    def calculate_regime_matrices(self):
        """
        Calculate matrices needed for the ODE solution as defined in the paper
        
        Returns:
        --------
        matrices : dict
            Dictionary containing the C, F and other matrices
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before calculating matrices")
        
        n = self.n_states
        
        # Extract transition rates
        pi = np.zeros(n)
        for i in range(n):
            pi[i] = -self.transition_matrix[i, i]
        
        # Create the generator matrix (Q-matrix)
        Q = self.transition_matrix.copy()
        for i in range(n):
            Q[i, i] = -np.sum(Q[i, :]) + Q[i, i]
        
        # Construct matrices as defined in the paper
        Pi_1 = np.diag(pi)
        Pi_2 = Q + Pi_1
        
        # Define the 2n x 2n block matrix C
        C = np.zeros((2*n, 2*n))
        
        # Fill in the blocks
        C[:n, n:] = np.eye(n)
        
        for i in range(n):
            C[n+i, i] = 2 * self.lambda_vals[i] * self.mu_vals[i] / self.sigma_vals[i]**2
            C[n+i, n+i] = -2 * self.lambda_vals[i] / self.sigma_vals[i]**2
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    C[n+i, n+j] = 2 * self.transition_matrix[i, j] / self.sigma_vals[i]**2
        
        # Define the 2n x 2n block matrix F
        F = np.zeros((2*n, 2*n))
        
        for i in range(n):
            F[n+i, i] = -2 * self.lambda_vals[i] / self.sigma_vals[i]**2
        
        return {'C': C, 'F': F, 'Q': Q, 'Pi_1': Pi_1, 'Pi_2': Pi_2}
    
    def probability_of_hitting_before(self, A, B, x, state_i=0, state_j=0):
        """
        Calculate the probability P^{ij}_1(A, B, x) in the paper
        
        Parameters:
        -----------
        A : float
            Lower boundary
        B : float
            Upper boundary
        x : float
            Initial spread value, where A < x < B
        state_i : int
            Initial state
        state_j : int
            Final state
            
        Returns:
        --------
        prob : float
            Probability of hitting boundary A before hitting boundary B, 
            starting from state i and ending in state j
        """
        if not (A < x < B):
            raise ValueError(f"Initial value x={x} must be between A={A} and B={B}")
        
        if self.n_states == 1:
            # Special case: one-state model
            # This is the formula in Proposition 3.4 in the paper
            lambda_val = self.lambda_vals[0]
            mu_val = self.mu_vals[0]
            sigma_val = self.sigma_vals[0]
            
            numerator = erfi(np.sqrt(lambda_val)*(x-mu_val)/sigma_val) - erfi(np.sqrt(lambda_val)*(B-mu_val)/sigma_val)
            denominator = erfi(np.sqrt(lambda_val)*(A-mu_val)/sigma_val) - erfi(np.sqrt(lambda_val)*(B-mu_val)/sigma_val)
            
            if state_i == state_j == 0:
                return numerator / denominator
            else:
                return 0.0
            
        # For multi-state model, we need to solve the ODE system
        # This is a simplified implementation that doesn't handle all cases properly
        matrices = self.calculate_regime_matrices()
        C = matrices['C']
        F = matrices['F']
        
        # Generate eigenvalues and eigenvectors of C
        eigenvalues, eigenvectors = la.eig(C)
        inv_eigenvectors = la.inv(eigenvectors)
        
        # Build solution for initial value problem using the method in Lemma 1
        n = self.n_states
        a_funcs = []
        
        for i in range(2*n):
            def a_func(x, alpha=eigenvalues[i], A=A):
                return np.exp(alpha * (x - A))
            a_funcs.append(a_func)
            
        # Boundary conditions: P^{ij}_1(A, B, A) = 1 if i=j, 0 otherwise
        # and P^{ij}_1(A, B, B) = 0
        
        # Set up linear system for initial derivative values
        # This is a simplified approach that doesn't fully implement the method in Theorem 3
        
        # For now, return an approximation based on one-state model adjusted by transition probs
        if state_i == state_j:
            # Equal states
            lambda_val = self.lambda_vals[state_i]
            mu_val = self.mu_vals[state_i]
            sigma_val = self.sigma_vals[state_i]
            
            numerator = erfi(np.sqrt(lambda_val)*(x-mu_val)/sigma_val) - erfi(np.sqrt(lambda_val)*(B-mu_val)/sigma_val)
            denominator = erfi(np.sqrt(lambda_val)*(A-mu_val)/sigma_val) - erfi(np.sqrt(lambda_val)*(B-mu_val)/sigma_val)
            
            # Adjust for probability of staying in the same state
            return (numerator / denominator) * (1 - np.sum([self.transition_matrix[state_i, k] for k in range(self.n_states) if k != state_i]))
        else:
            # Different states, probability lower due to need for state transition
            lambda_val = self.lambda_vals[state_i]
            mu_val = self.mu_vals[state_i]
            sigma_val = self.sigma_vals[state_i]
            
            # Base probability
            numerator = erfi(np.sqrt(lambda_val)*(x-mu_val)/sigma_val) - erfi(np.sqrt(lambda_val)*(B-mu_val)/sigma_val)
            denominator = erfi(np.sqrt(lambda_val)*(A-mu_val)/sigma_val) - erfi(np.sqrt(lambda_val)*(B-mu_val)/sigma_val)
            
            # Adjust for transition probability
            return (numerator / denominator) * self.transition_matrix[state_i, state_j] * 0.5
    
    def expected_stopping_time(self, A, B, x, state_i=0):
        """
        Calculate the expected stopping time T_i(x) in the paper
        
        Parameters:
        -----------
        A : float
            Lower boundary
        B : float
            Upper boundary
        x : float
            Initial spread value, where A < x < B
        state_i : int
            Initial state
            
        Returns:
        --------
        time : float
            Expected time until hitting either boundary A or B, starting from state i
        """
        if not (A < x < B):
            raise ValueError(f"Initial value x={x} must be between A={A} and B={B}")
        
        if self.n_states == 1:
            # Special case: one-state model
            # This is the formula in Proposition 3.5 in the paper
            lambda_val = self.lambda_vals[0]
            mu_val = self.mu_vals[0]
            sigma_val = self.sigma_vals[0]
            
            # Define g(x) function - this is an approximation for demonstration
            def g(x):
                u = np.sqrt(lambda_val) * (x - mu_val) / sigma_val
                return (sigma_val * np.sqrt(np.pi) / (2 * np.sqrt(lambda_val))) * (
                    erf(u) * erfi(u) + u**2 * np.exp(-u**2) * 2 / np.sqrt(np.pi)
                )
            
            # Calculate using formula from Proposition 3.5
            P_2 = self.probability_of_hitting_before(A, B, x, 0, 0)
            
            return (1 / (np.sqrt(lambda_val) * sigma_val)) * (
                P_2 * (g(A) - g(B)) + g(x) - g(A)
            )
        
        # For multi-state model, we need to solve the ODE system (non-homogeneous)
        # This is a simplified implementation
        
        # For now, return a weighted average of one-state results
        times = []
        for state in range(self.n_states):
            lambda_val = self.lambda_vals[state]
            mu_val = self.mu_vals[state]
            sigma_val = self.sigma_vals[state]
            
            # Approximate g(x) function
            def g(x):
                u = np.sqrt(lambda_val) * (x - mu_val) / sigma_val
                return (sigma_val * np.sqrt(np.pi) / (2 * np.sqrt(lambda_val))) * (
                    erf(u) * erfi(u) + u**2 * np.exp(-u**2) * 2 / np.sqrt(np.pi)
                )
            
            # Calculate transition probability
            P_2 = self.probability_of_hitting_before(A, B, x, state_i, state)
            
            # Calculate expected time
            time = (1 / (np.sqrt(lambda_val) * sigma_val)) * (
                P_2 * (g(A) - g(B)) + g(x) - g(A)
            )
            
            times.append(time)
        
        # Use average weighted by state transition probabilities
        weights = [self.transition_matrix[state_i, j] for j in range(self.n_states)]
        weights_sum = sum(weights)
        if weights_sum > 0:
            weights = [w / weights_sum for w in weights]
        else:
            weights = [1.0 / self.n_states] * self.n_states
            
        return sum(t * w for t, w in zip(times, weights))
    
    def optimize_thresholds(self, x, min_A=None, max_A=None, min_B=None, max_B=None, 
                           transaction_cost=0.01, stop_loss=0.5, trials=20):
        """
        Find optimal thresholds A* and B* for pairs trading strategy
        
        Parameters:
        -----------
        x : float
            Initial spread value
        min_A, max_A : float
            Range for threshold A
        min_B, max_B : float
            Range for threshold B
        transaction_cost : float
            Transaction cost parameter c
        stop_loss : float
            Stop-loss threshold L
        trials : int
            Number of optimization trials
            
        Returns:
        --------
        result : dict
            Dictionary containing optimal thresholds and expected return
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before optimizing thresholds")
            
        # Set default ranges if not provided
        if min_A is None:
            min_A = np.min(self.mu_vals) - 2 * np.max(self.sigma_vals)
        if max_A is None:
            max_A = x - 0.1
        if min_B is None:
            min_B = x + 0.1
        if max_B is None:
            max_B = np.max(self.mu_vals) + 2 * np.max(self.sigma_vals)
            
        # Ensure x is between A and B
        if not (min_A < x < min_B):
            raise ValueError(f"Initial value x={x} must be between min_A={min_A} and min_B={min_B}")
            
        # Define value function (to be maximized)
        def value_function(params):
            A, B = params
            
            if not (A < x < B):
                return -np.inf  # Invalid configuration
                
            # Calculate components for numerator and denominator
            numerator = 0
            denominator = 0
            
            for i in range(self.n_states):
                # Probability of reaching A before B
                P_i1 = 0
                for j in range(self.n_states):
                    P_i1 += self.probability_of_hitting_before(A, B, x, i, j)
                
                # Probability of reaching B before A
                P_i2 = 0
                for j in range(self.n_states):
                    P_i2 += self.probability_of_hitting_before(B, A, x, i, j)
                    
                # Expected profit when hitting A
                profit_A = 0
                for j in range(self.n_states):
                    # Probability of hitting A in state j
                    P_ij1 = self.probability_of_hitting_before(A, B, x, i, j)
                    if P_ij1 > 0:
                        # Probability of reaching B after hitting A
                        P_j2_after_A = 0
                        for k in range(self.n_states):
                            P_j2_after_A += self.probability_of_hitting_before(A-stop_loss, B, A, j, k)
                            
                        # Probability of reaching A-L after hitting A (stop-loss)
                        P_j1_after_A = 0
                        for k in range(self.n_states):
                            P_j1_after_A += self.probability_of_hitting_before(A, A-stop_loss, A, j, k)
                            
                        # Expected profit
                        profit_A += P_ij1 * ((B - A - transaction_cost) * P_j2_after_A - 
                                            (stop_loss + transaction_cost) * P_j1_after_A)
                
                # Expected profit when hitting B
                profit_B = 0
                for j in range(self.n_states):
                    # Probability of hitting B in state j
                    P_ij2 = self.probability_of_hitting_before(B, A, x, i, j)
                    if P_ij2 > 0:
                        # Probability of reaching A after hitting B
                        P_j1_after_B = 0
                        for k in range(self.n_states):
                            P_j1_after_B += self.probability_of_hitting_before(A, B+stop_loss, B, j, k)
                            
                        # Probability of reaching B+L after hitting B (stop-loss)
                        P_j2_after_B = 0
                        for k in range(self.n_states):
                            P_j2_after_B += self.probability_of_hitting_before(B+stop_loss, B, B, j, k)
                            
                        # Expected profit
                        profit_B += P_ij2 * ((B - A - transaction_cost) * P_j1_after_B - 
                                           (stop_loss + transaction_cost) * P_j2_after_B)
                
                # Add to numerator weighted by initial state probability
                numerator += self.initial_probs[i] * (profit_A + profit_B)
                
                # Calculate expected time components
                expected_time = 0
                for j in range(self.n_states):
                    # Time until hitting A in state j
                    if P_ij1 > 0:
                        expected_time += P_ij1 * self.expected_stopping_time(A-stop_loss, B, A, j)
                        
                    # Time until hitting B in state j
                    if P_ij2 > 0:
                        expected_time += P_ij2 * self.expected_stopping_time(A, B+stop_loss, B, j)
                
                # Add to denominator weighted by initial state probability
                denominator += self.initial_probs[i] * (expected_time - self.expected_stopping_time(A, B, x, i))
            
            # Handle division by zero
            if denominator <= 0:
                return -np.inf
                
            # Return negative for minimization
            return -(numerator / denominator)
            
        # Run multiple trials with different starting points
        best_result = None
        best_value = -np.inf
        
        for _ in range(trials):
            # Random starting point
            initial_A = np.random.uniform(min_A, max_A)
            initial_B = np.random.uniform(min_B, max_B)
            
            # Bounds for optimization
            bounds = [(min_A, max_A), (min_B, max_B)]
            
            # Run optimization
            result = opt.minimize(
                value_function, 
                [initial_A, initial_B],
                bounds=bounds,
                method='L-BFGS-B'
            )
            
            if result.success and -result.fun > best_value:
                best_result = result
                best_value = -result.fun
        
        if best_result is None:
            return {"A": None, "B": None, "expected_return": None, "success": False}
            
        # Extract optimal thresholds
        optimal_A, optimal_B = best_result.x
        
        return {
            "A": optimal_A,
            "B": optimal_B,
            "expected_return": best_value,
            "success": True
        }


class RegimeSwitchingPairsTrading:
    """
    Implementation of the regime-switching pairs trading strategy
    
    Based on: "Analytic value function for optimal regime-switching pairs trading rules"
    by Yang Bai and Lan Wu (2018)
    """
    
    def __init__(self, stock1, stock2, formation_window=60, n_states=2, 
                 transaction_cost=0.01, stop_loss=0.5):
        """
        Initialize the pairs trading strategy
        
        Parameters:
        -----------
        stock1 : str
            Ticker of the first stock
        stock2 : str
            Ticker of the second stock
        formation_window : int
            Number of days to use for model formation
        n_states : int
            Number of states in the regime-switching model
        transaction_cost : float
            Transaction cost per trade (as a percentage)
        stop_loss : float
            Stop-loss threshold for exiting the trade
        """
        self.stock1 = stock1
        self.stock2 = stock2
        self.formation_window = formation_window
        self.n_states = n_states
        self.transaction_cost = transaction_cost
        self.stop_loss = stop_loss
        
        # Bloomberg API connection
        self.bloomberg = None
        
        # Model for the trading pair
        self.model = MarkovModulatedOUModel(n_states=n_states)
        
        # Trading thresholds
        self.threshold_A = None
        self.threshold_B = None
        
        # Cointegration coefficient
        self.beta = None
        
        # Trading state
        self.in_position = False
        self.position_type = None  # 'long_spread' or 'short_spread'
        self.entry_level = None
        self.entry_date = None
        
        # Trading results
        self.trades = []
        self.daily_returns = []
    
    def connect_to_bloomberg(self):
        """
        Connect to Bloomberg
        
        Returns:
        --------
        success : bool
            True if connection successful, False otherwise
        """
        try:
            self.bloomberg = pdblp.BCon(timeout=50000)
            self.bloomberg.start()
            print("Connected to Bloomberg")
            return True
        except Exception as e:
            print(f"Failed to connect to Bloomberg: {e}")
            return False
    
    def get_historical_data(self, start_date, end_date):
        """
        Get historical price data for the pair
        
        Parameters:
        -----------
        start_date : str
            Start date in format 'YYYYMMDD'
        end_date : str
            End date in format 'YYYYMMDD'
            
        Returns:
        --------
        data : DataFrame
            Historical price data for the pair
        """
        # First try: Load from local CSV if available
        try:
            csv_path = f"{self.stock1}_{self.stock2}_prices.csv"
            print(f"Attempting to load data from {csv_path}")
            data = pd.read_csv(csv_path)
            
            # Validate CSV data structure
            required_columns = ['date', self.stock1, self.stock2]
            for col in required_columns:
                if col not in data.columns:
                    print(f"CSV file is missing required column: {col}")
                    raise ValueError(f"Invalid CSV format: missing {col}")
            
            # Convert date column
            data['date'] = pd.to_datetime(data['date'])
            
            # Check for and remove rows with zeros, NaNs, or negative values
            bad_rows = (data[self.stock1] <= 0) | np.isnan(data[self.stock1]) | np.isinf(data[self.stock1]) | \
                       (data[self.stock2] <= 0) | np.isnan(data[self.stock2]) | np.isinf(data[self.stock2])
            
            if bad_rows.any():
                print(f"Found {bad_rows.sum()} rows with invalid price data in CSV. Removing...")
                data = data[~bad_rows]
            
            # Filter by date range
            start_dt = dt.datetime.strptime(start_date, '%Y%m%d')
            end_dt = dt.datetime.strptime(end_date, '%Y%m%d')
            data = data[(data['date'] >= start_dt) & (data['date'] <= end_dt)]
            
            if len(data) > 0:
                print(f"Successfully loaded {len(data)} days of data from CSV")
                return data
            else:
                print("No data found in date range in CSV file")
        except Exception as e:
            print(f"Could not load from CSV: {e}")
        
        # Second try: Bloomberg
        if self.bloomberg is None:
            if not self.connect_to_bloomberg():
                print("Bloomberg connection failed")
                # Proceed to simulated data
            else:
                try:
                    # For Chinese A-shares, use proper Bloomberg suffixes
                    ticker1 = f"{self.stock1} CH Equity" if "CH" not in self.stock1 else f"{self.stock1}"
                    ticker2 = f"{self.stock2} CH Equity" if "CH" not in self.stock2 else f"{self.stock2}"
                    
                    print(f"Retrieving data from Bloomberg for tickers: {ticker1} and {ticker2}")
                    
                    # Get price data
                    data = self.bloomberg.bdh(
                        [ticker1, ticker2],
                        ['PX_LAST'],
                        start_date,
                        end_date
                    )
                    
                    # Reshape data
                    data = data.reset_index()
                    data.columns = ['date', self.stock1, self.stock2]
                    
                    # Save to CSV for future use
                    try:
                        data.to_csv(f"{self.stock1}_{self.stock2}_prices.csv", index=False)
                        print(f"Saved data to {self.stock1}_{self.stock2}_prices.csv")
                    except Exception as e:
                        print(f"Failed to save data to CSV: {e}")
                    
                    return data
                except Exception as e:
                    print(f"Failed to get Bloomberg data: {e}")
        
        # Third try: Simulated data
        print("Using simulated data for testing")
        return self.create_test_csv(start_date, end_date)
    
    def create_test_csv(self, start_date, end_date):
        """
        Create a test CSV file with simulated data
        
        Parameters:
        -----------
        start_date : str
            Start date in format 'YYYYMMDD'
        end_date : str
            End date in format 'YYYYMMDD'
            
        Returns:
        --------
        data : DataFrame
            Simulated price data
        """
        # Generate simulated data
        test_data = self._get_simulated_data(start_date, end_date)
        
        # Save to CSV
        csv_path = f"{self.stock1}_{self.stock2}_prices.csv"
        print(f"Creating test CSV file: {csv_path}")
        test_data.to_csv(csv_path, index=False)
        
        print(f"Created test CSV with {len(test_data)} days of data")
        return test_data
    
    def _get_simulated_data(self, start_date, end_date):
        """
        Generate simulated data for testing when Bloomberg is not available
        
        Parameters:
        -----------
        start_date : str
            Start date in format 'YYYYMMDD'
        end_date : str
            End date in format 'YYYYMMDD'
            
        Returns:
        --------
        data : DataFrame
            Simulated price data for the pair
        """
        # Convert dates to datetime
        start = dt.datetime.strptime(start_date, '%Y%m%d')
        end = dt.datetime.strptime(end_date, '%Y%m%d')
        
        # Create date range
        date_range = pd.date_range(start, end, freq='B')
        
        # Number of trading days
        n_days = len(date_range)
        
        # Generate random walk for first stock
        np.random.seed(42)
        stock1_prices = 100 * np.exp(np.cumsum(np.random.normal(0.0002, 0.01, n_days)))
        
        # Initialize stock2_prices array
        stock2_prices = np.zeros(n_days)
        stock2_prices[0] = 100  # Start at 100 like stock1
        
        # Generate correlated random walk for second stock with regime-switching
        regime_changes = np.random.rand(n_days) < 0.02  # 2% chance of regime switch each day
        regimes = np.cumsum(regime_changes) % 2
        
        # Different correlation in each regime
        correlation_regime0 = 0.8
        correlation_regime1 = 0.4
        
        # Generate correlated returns
        for i in range(1, n_days):
            # Get correlation for current regime
            correlation = correlation_regime0 if regimes[i] == 0 else correlation_regime1
            
            # Calculate return for stock 1
            stock1_return = np.log(stock1_prices[i] / stock1_prices[i-1])
            
            # Calculate mean-reverting term using previous prices
            mean_reverting_term = 0.1 * (np.log(stock1_prices[i-1]) - np.log(stock2_prices[i-1]))
            
            # Calculate return for stock 2
            stock2_return = correlation * stock1_return + (1-correlation) * np.random.normal(0.0002, 0.01) + mean_reverting_term
            
            # Update stock2 price
            stock2_prices[i] = stock2_prices[i-1] * np.exp(stock2_return)
        
        # Create DataFrame
        data = pd.DataFrame({
            'date': date_range,
            self.stock1: stock1_prices,
            self.stock2: stock2_prices
        })
        
        return data
    
    def calculate_spread(self, prices):
        """
        Calculate the spread between the two stocks
        
        Parameters:
        -----------
        prices : DataFrame
            Price data for the pair
            
        Returns:
        --------
        spread : Series
            Spread time series
        """
        # Make a copy to avoid modifying original data
        prices_clean = prices.copy()
        
        # Check for zeros, NaNs, or negative values in price data
        mask1 = (prices_clean[self.stock1] <= 0) | np.isnan(prices_clean[self.stock1]) | np.isinf(prices_clean[self.stock1])
        mask2 = (prices_clean[self.stock2] <= 0) | np.isnan(prices_clean[self.stock2]) | np.isinf(prices_clean[self.stock2])
        
        # Identify problematic rows
        problem_rows = mask1 | mask2
        
        if problem_rows.any():
            print(f"Found {problem_rows.sum()} problematic rows in price data. Cleaning...")
            
            # Method 1: Drop problematic rows
            prices_clean = prices_clean[~problem_rows]
            
            # If too many rows are dropped, try forward fill instead
            if len(prices_clean) < 0.8 * len(prices):
                print("Too many rows would be dropped, using forward fill instead")
                prices_clean = prices.copy()
                # Replace zeros, NaNs, and negative values with NaN
                prices_clean.loc[mask1, self.stock1] = np.nan
                prices_clean.loc[mask2, self.stock2] = np.nan
                # Forward fill NaNs
                prices_clean = prices_clean.fillna(method='ffill')
                # If still have NaNs at the beginning, backward fill
                prices_clean = prices_clean.fillna(method='bfill')
        
        # Take logs of prices
        log_prices1 = np.log(prices_clean[self.stock1])
        log_prices2 = np.log(prices_clean[self.stock2])
        
        # Check for infinities or NaNs in log prices
        if np.isnan(log_prices1).any() or np.isnan(log_prices2).any() or np.isinf(log_prices1).any() or np.isinf(log_prices2).any():
            print("Warning: Log prices contain NaNs or infinities after cleaning. Using simplified correlation method.")
            # Use simplified correlation method
            self.beta = np.corrcoef(prices_clean[self.stock1], prices_clean[self.stock2])[0, 1]
            spread = np.log(prices_clean[self.stock1]) - self.beta * np.log(prices_clean[self.stock2])
            # Fill any remaining NaNs or infinities with median
            median_spread = np.nanmedian(spread[~np.isinf(spread)])
            spread = spread.replace([np.inf, -np.inf, np.nan], median_spread)
            return spread
        
        # Perform linear regression
        X = sm.add_constant(log_prices2)
        model = sm.OLS(log_prices1, X)
        results = model.fit()
        
        # Store coefficient for trading
        self.beta = results.params[1]
        
        # Calculate spread
        spread = log_prices1 - self.beta * log_prices2
        
        return spread
    
    def check_cointegration(self, spread, significance=0.05):
        """
        Check if the spread is cointegrated
        
        Parameters:
        -----------
        spread : Series
            Spread time series
        significance : float
            Significance level for cointegration test
            
        Returns:
        --------
        is_cointegrated : bool
            True if the spread is cointegrated, False otherwise
        """
        # Perform Augmented Dickey-Fuller test
        try:
            result = adfuller(spread.dropna())
            
            # Check if p-value is below threshold
            return result[1] < significance
        except Exception as e:
            print(f"Cointegration test failed: {e}")
            return False
    
    def train_model(self, spread):
        """
        Train the regime-switching model on historical spread data
        
        Parameters:
        -----------
        spread : Series
            Spread time series
            
        Returns:
        --------
        success : bool
            True if model training was successful, False otherwise
        """
        try:
            # Ensure spread has no NaNs or infinities
            clean_spread = spread.replace([np.inf, -np.inf], np.nan).dropna()
            
            if len(clean_spread) < 30:
                print("Not enough valid data points for model training")
                return False
                
            # Fit the model
            self.model.fit(clean_spread.values, method='threshold')
            return True
        except Exception as e:
            print(f"Failed to train model: {e}")
            return False
    
    def find_optimal_thresholds(self, spread):
        """
        Find optimal trading thresholds A and B
        
        Parameters:
        -----------
        spread : Series
            Spread time series
            
        Returns:
        --------
        success : bool
            True if optimization was successful, False otherwise
        """
        try:
            # Ensure spread has no NaNs or infinities
            clean_spread = spread.replace([np.inf, -np.inf], np.nan).dropna()
            
            # Current spread value
            current_spread = clean_spread.iloc[-1]
            
            # Compute mean and std of the spread
            mean_spread = np.mean(clean_spread)
            std_spread = np.std(clean_spread)
            
            # Define search ranges for optimization
            min_A = mean_spread - 2 * std_spread
            max_A = mean_spread - 0.2 * std_spread
            min_B = mean_spread + 0.2 * std_spread
            max_B = mean_spread + 2 * std_spread
            
            # Ensure current_spread is between min_A and min_B
            if current_spread <= min_A:
                current_spread = min_A + 0.1 * (min_B - min_A)
            elif current_spread >= min_B:
                current_spread = min_B - 0.1 * (min_B - min_A)
            
            # Find optimal thresholds
            result = self.model.optimize_thresholds(
                current_spread,
                min_A=min_A,
                max_A=max_A,
                min_B=min_B,
                max_B=max_B,
                transaction_cost=self.transaction_cost,
                stop_loss=self.stop_loss
            )
            
            if result["success"]:
                self.threshold_A = result["A"]
                self.threshold_B = result["B"]
                print(f"Optimal thresholds: A={self.threshold_A:.4f}, B={self.threshold_B:.4f}")
                print(f"Expected return: {result['expected_return']:.6f}")
                return True
            else:
                # Use simple thresholds based on mean and std
                self.threshold_A = mean_spread - std_spread
                self.threshold_B = mean_spread + std_spread
                print(f"Using simple thresholds: A={self.threshold_A:.4f}, B={self.threshold_B:.4f}")
                return True
        except Exception as e:
            print(f"Failed to find optimal thresholds: {e}")
            # Use simple thresholds based on mean and std
            try:
                clean_spread = spread.replace([np.inf, -np.inf], np.nan).dropna()
                mean_spread = np.mean(clean_spread)
                std_spread = np.std(clean_spread)
                self.threshold_A = mean_spread - std_spread
                self.threshold_B = mean_spread + std_spread
                print(f"Using simple thresholds: A={self.threshold_A:.4f}, B={self.threshold_B:.4f}")
                return True
            except:
                print("Failed to set any thresholds. Using default values.")
                self.threshold_A = -1.0
                self.threshold_B = 1.0
                return False
    
    def execute_trading_rule(self, current_date, current_spread, current_prices):
        """
        Execute the trading rule based on current spread value
        
        Parameters:
        -----------
        current_date : datetime
            Current date
        current_spread : float
            Current spread value
        current_prices : Series
            Current prices of the pair
            
        Returns:
        --------
        action : str
            Trading action taken
        """
        action = "no_action"
        
        # If current_spread is invalid, skip
        if np.isnan(current_spread) or np.isinf(current_spread):
            return action
        
        # If not in a position, check for entry signals
        if not self.in_position:
            if current_spread <= self.threshold_A:
                # Spread is below lower threshold, go long the spread
                # Buy stock1, sell stock2
                print(f"{current_date}: Entry signal - Long spread at {current_spread:.4f}")
                self.in_position = True
                self.position_type = 'long_spread'
                self.entry_level = current_spread
                self.entry_date = current_date
                action = "enter_long_spread"
                
                # Record trade entry
                self.trades.append({
                    'entry_date': current_date,
                    'entry_spread': current_spread,
                    'entry_prices': current_prices.copy(),
                    'position_type': 'long_spread',
                    'threshold_A': self.threshold_A,
                    'threshold_B': self.threshold_B
                })
                
            elif current_spread >= self.threshold_B:
                # Spread is above upper threshold, go short the spread
                # Sell stock1, buy stock2
                print(f"{current_date}: Entry signal - Short spread at {current_spread:.4f}")
                self.in_position = True
                self.position_type = 'short_spread'
                self.entry_level = current_spread
                self.entry_date = current_date
                action = "enter_short_spread"
                
                # Record trade entry
                self.trades.append({
                    'entry_date': current_date,
                    'entry_spread': current_spread,
                    'entry_prices': current_prices.copy(),
                    'position_type': 'short_spread',
                    'threshold_A': self.threshold_A,
                    'threshold_B': self.threshold_B
                })
        
        # If in a position, check for exit signals
        else:
            if self.position_type == 'long_spread':
                # Exit if spread crosses upper threshold or stop-loss
                if (current_spread >= self.threshold_B or 
                    current_spread <= self.entry_level - self.stop_loss):
                    
                    # Calculate profit
                    spread_change = current_spread - self.entry_level
                    profit = spread_change - self.transaction_cost
                    
                    # Exit long spread position
                    print(f"{current_date}: Exit signal - Long spread at {current_spread:.4f}, profit: {profit:.4f}")
                    self.in_position = False
                    action = "exit_long_spread"
                    
                    # Update trade record
                    latest_trade = self.trades[-1]
                    latest_trade.update({
                        'exit_date': current_date,
                        'exit_spread': current_spread,
                        'exit_prices': current_prices.copy(),
                        'profit': profit,
                        'duration': (current_date - self.entry_date).days
                    })
                    
                    # Reset position data
                    self.position_type = None
                    self.entry_level = None
                    self.entry_date = None
                    
            elif self.position_type == 'short_spread':
                # Exit if spread crosses lower threshold or stop-loss
                if (current_spread <= self.threshold_A or 
                    current_spread >= self.entry_level + self.stop_loss):
                    
                    # Calculate profit
                    spread_change = self.entry_level - current_spread
                    profit = spread_change - self.transaction_cost
                    
                    # Exit short spread position
                    print(f"{current_date}: Exit signal - Short spread at {current_spread:.4f}, profit: {profit:.4f}")
                    self.in_position = False
                    action = "exit_short_spread"
                    
                    # Update trade record
                    latest_trade = self.trades[-1]
                    latest_trade.update({
                        'exit_date': current_date,
                        'exit_spread': current_spread,
                        'exit_prices': current_prices.copy(),
                        'profit': profit,
                        'duration': (current_date - self.entry_date).days
                    })
                    
                    # Reset position data
                    self.position_type = None
                    self.entry_level = None
                    self.entry_date = None
        
        return action
    
    def calculate_daily_returns(self, prices, spreads):
        """
        Calculate daily returns of the trading strategy
        
        Parameters:
        -----------
        prices : DataFrame
            Daily price data
        spreads : Series
            Daily spread values
            
        Returns:
        --------
        returns : DataFrame
            Daily returns of the strategy
        """
        # Create a DataFrame for daily returns
        returns = pd.DataFrame(index=prices.index)
        returns['strategy_return'] = 0.0
        returns['position'] = 'none'
        
        # Loop through trades to calculate daily returns
        for trade in self.trades:
            if 'exit_date' not in trade:
                continue
                
            entry_date = trade['entry_date']
            exit_date = trade['exit_date']
            position_type = trade['position_type']
            
            # Get dates between entry and exit
            trade_dates = prices[(prices.index >= entry_date) & (prices.index <= exit_date)].index
            
            # Skip if no dates found
            if len(trade_dates) < 2:
                continue
                
            # Calculate daily position returns
            entry_prices = trade['entry_prices']
            entry_spread = trade['entry_spread']
            
            for i in range(1, len(trade_dates)):
                current_date = trade_dates[i]
                prev_date = trade_dates[i-1]
                
                # Get daily price changes
                price1_change = (prices.loc[current_date, self.stock1] / prices.loc[prev_date, self.stock1]) - 1
                price2_change = (prices.loc[current_date, self.stock2] / prices.loc[prev_date, self.stock2]) - 1
                
                # Calculate position return
                if position_type == 'long_spread':
                    # Long stock1, short stock2
                    # Return is long stock1 return minus beta times short stock2 return
                    daily_return = price1_change - self.beta * price2_change
                else:  # short_spread
                    # Short stock1, long stock2
                    # Return is short stock1 return minus beta times long stock2 return
                    daily_return = -price1_change + self.beta * price2_change
                
                # Add to daily returns
                returns.loc[current_date, 'strategy_return'] = daily_return
                returns.loc[current_date, 'position'] = position_type
        
        # Fill NaN returns with 0 (days with no position)
        returns['strategy_return'] = returns['strategy_return'].fillna(0)
        
        # Calculate cumulative returns
        returns['cumulative_return'] = (1 + returns['strategy_return']).cumprod() - 1
        
        return returns
    
    def backtest(self, start_date, end_date, rebalance_freq='M'):
        """
        Backtest the pairs trading strategy
        
        Parameters:
        -----------
        start_date : str
            Start date in format 'YYYYMMDD'
        end_date : str
            End date in format 'YYYYMMDD'
        rebalance_freq : str
            Frequency to rebalance the model, 'D' for daily, 'W' for weekly, 'M' for monthly
            
        Returns:
        --------
        results : dict
            Dictionary containing backtest results
        """
        # Get historical data
        data = self.get_historical_data(start_date, end_date)
        
        # Convert date to datetime
        data['date'] = pd.to_datetime(data['date'])
        data = data.set_index('date')
        
        # Calculate spread
        print("Calculating spread...")
        spread = self.calculate_spread(data)
        
        # Check cointegration
        is_cointegrated = self.check_cointegration(spread)
        if not is_cointegrated:
            print("Warning: Spread is not cointegrated")
        
        # Initialize backtest
        test_data = data.copy()
        
        # Define rebalance dates
        if rebalance_freq == 'D':
            rebalance_dates = test_data.index
        elif rebalance_freq == 'W':
            rebalance_dates = pd.date_range(start=test_data.index[0], end=test_data.index[-1], freq='W')
        elif rebalance_freq == 'M':
            rebalance_dates = pd.date_range(start=test_data.index[0], end=test_data.index[-1], freq='M')
        else:
            rebalance_dates = [test_data.index[0], test_data.index[-1]]
        
        # Initialize model with formation period
        formation_window_size = min(self.formation_window, len(test_data)-1)
        if formation_window_size < 10:
            formation_window_size = min(10, len(test_data)-1)
            
        formation_end = test_data.index[formation_window_size]
        formation_data = test_data.loc[:formation_end]
        formation_spread = spread.loc[:formation_end]
        
        # Train initial model
        print(f"Training initial model with {len(formation_data)} days of data")
        self.train_model(formation_spread)
        
        # Find initial thresholds
        self.find_optimal_thresholds(formation_spread)
        
        # Run backtest
        print("Starting backtest...")
        for date in tqdm(test_data.index[formation_window_size:]):
            current_prices = test_data.loc[date]
            current_spread = spread.loc[date]
            
            # Check if rebalance is needed
            rebalance = False
            for rebalance_date in rebalance_dates:
                if date.date() == rebalance_date.date():
                    rebalance = True
                    break
            
            if rebalance:
                # Rebalance model using past formation_window days
                start_idx = test_data.index.get_loc(date) - formation_window_size
                if start_idx < 0:
                    start_idx = 0
                formation_start = test_data.index[start_idx]
                
                # Get formation data
                formation_data = test_data.loc[formation_start:date]
                formation_spread = spread.loc[formation_start:date]
                
                # Train model
                print(f"\nRebalancing model at {date} with {len(formation_data)} days of data")
                self.train_model(formation_spread)
                
                # Find optimal thresholds
                self.find_optimal_thresholds(formation_spread)
            
            # Execute trading rule
            action = self.execute_trading_rule(date, current_spread, current_prices)
        
        # Calculate performance metrics
        self.daily_returns = self.calculate_daily_returns(test_data, spread)
        
        # Close any open positions at the end
        if self.in_position:
            print(f"Closing open {self.position_type} position at the end of backtest")
            
            last_date = test_data.index[-1]
            last_spread = spread.iloc[-1]
            last_prices = test_data.iloc[-1]
            
            # Calculate profit
            if self.position_type == 'long_spread':
                spread_change = last_spread - self.entry_level
                profit = spread_change - self.transaction_cost
            else:  # short_spread
                spread_change = self.entry_level - last_spread
                profit = spread_change - self.transaction_cost
            
            # Update trade record
            self.trades[-1].update({
                'exit_date': last_date,
                'exit_spread': last_spread,
                'exit_prices': last_prices,
                'profit': profit,
                'duration': (last_date - self.entry_date).days
            })
            
            # Reset position data
            self.in_position = False
            self.position_type = None
            self.entry_level = None
            self.entry_date = None
        
        # Prepare results
        results = {
            'trades': pd.DataFrame(self.trades) if self.trades else pd.DataFrame(),
            'daily_returns': self.daily_returns,
            'spread': spread,
            'prices': test_data,
            'beta': self.beta,
            'parameters': {
                'lambda_vals': self.model.lambda_vals,
                'mu_vals': self.model.mu_vals,
                'sigma_vals': self.model.sigma_vals,
                'transition_matrix': self.model.transition_matrix
            },
            'is_cointegrated': is_cointegrated
        }
        
        # Calculate aggregated performance metrics
        if 'trades' in results and len(results['trades']) > 0:
            results['metrics'] = self.calculate_performance_metrics(results)
        
        return results
    
    def calculate_performance_metrics(self, results):
        """
        Calculate performance metrics for the strategy
        
        Parameters:
        -----------
        results : dict
            Dictionary containing backtest results
            
        Returns:
        --------
        metrics : dict
            Dictionary with performance metrics
        """
        trades_df = results['trades']
        returns_df = results['daily_returns']
        
        # Trade metrics
        num_trades = len(trades_df)
        profitable_trades = sum(trades_df['profit'] > 0) if 'profit' in trades_df.columns else 0
        win_rate = profitable_trades / num_trades if num_trades > 0 else 0
        
        avg_profit = trades_df['profit'].mean() if 'profit' in trades_df.columns else 0
        avg_duration = trades_df['duration'].mean() if 'duration' in trades_df.columns else 0
        
        # Returns metrics
        total_return = returns_df['cumulative_return'].iloc[-1]
        annual_return = ((1 + total_return) ** (252 / len(returns_df))) - 1
        daily_returns = returns_df['strategy_return']
        volatility = daily_returns.std() * np.sqrt(252)
        sharpe_ratio = annual_return / volatility if volatility > 0 else 0
        
        # Drawdown
        cumulative_returns = returns_df['cumulative_return']
        rolling_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - rolling_max) / (1 + rolling_max)
        max_drawdown = drawdown.min()
        
        metrics = {
            'num_trades': num_trades,
            'win_rate': win_rate,
            'avg_profit': avg_profit,
            'avg_duration': avg_duration,
            'total_return': total_return,
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown
        }
        
        return metrics
    
    def plot_results(self, results):
        """
        Plot strategy results
        
        Parameters:
        -----------
        results : dict
            Dictionary containing backtest results
        """
        # Create figure with subplots
        fig, axes = plt.subplots(3, 1, figsize=(14, 18), gridspec_kw={'height_ratios': [2, 1, 1]})
        
        # Plot 1: Stock prices
        ax1 = axes[0]
        prices = results['prices']
        normalized_prices = prices / prices.iloc[0]
        normalized_prices.plot(ax=ax1)
        ax1.set_title('Normalized Stock Prices')
        ax1.set_ylabel('Price (normalized)')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Spread with trading signals
        ax2 = axes[1]
        spread = results['spread']
        spread.plot(ax=ax2, label='Spread')
        
        # Add horizontal lines for thresholds
        if 'trades' in results and len(results['trades']) > 0:
            for trade in results['trades'].itertuples():
                try:
                    ax2.axhline(y=trade.threshold_A, xmin=trade.entry_date, xmax=trade.exit_date, 
                              color='g', linestyle='--', alpha=0.5)
                    ax2.axhline(y=trade.threshold_B, xmin=trade.entry_date, xmax=trade.exit_date, 
                              color='r', linestyle='--', alpha=0.5)
                    
                    # Mark entry and exit points
                    if trade.position_type == 'long_spread':
                        ax2.scatter(trade.entry_date, trade.entry_spread, marker='^', color='g', s=100)
                        ax2.scatter(trade.exit_date, trade.exit_spread, marker='v', color='g', s=100)
                    else:  # short_spread
                        ax2.scatter(trade.entry_date, trade.entry_spread, marker='v', color='r', s=100)
                        ax2.scatter(trade.exit_date, trade.exit_spread, marker='^', color='r', s=100)
                except Exception as e:
                    print(f"Error plotting trade signal: {e}")
        
        ax2.set_title('Spread with Trading Signals')
        ax2.set_ylabel('Spread')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Cumulative returns
        ax3 = axes[2]
        returns = results['daily_returns']
        returns['cumulative_return'].plot(ax=ax3)
        ax3.set_title('Cumulative Returns')
        ax3.set_ylabel('Return')
        ax3.grid(True, alpha=0.3)
        
        # Add metrics as text
        if 'metrics' in results:
            metrics = results['metrics']
            metrics_text = (
                f"Number of Trades: {metrics['num_trades']}\n"
                f"Win Rate: {metrics['win_rate']:.2%}\n"
                f"Average Profit: {metrics['avg_profit']:.4f}\n"
                f"Total Return: {metrics['total_return']:.2%}\n"
                f"Annual Return: {metrics['annual_return']:.2%}\n"
                f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}\n"
                f"Max Drawdown: {metrics['max_drawdown']:.2%}"
            )
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            ax3.text(0.05, 0.05, metrics_text, transform=ax3.transAxes, fontsize=10,
                    verticalalignment='bottom', bbox=props)
        
        plt.tight_layout()
        plt.show()
        
        # Plot regime-switching parameters
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        
        # Plot transition probability matrix as heatmap
        ax1 = axes[0]
        transition_matrix = results['parameters']['transition_matrix']
        im = ax1.imshow(transition_matrix, cmap='viridis')
        
        # Add annotations
        for i in range(transition_matrix.shape[0]):
            for j in range(transition_matrix.shape[1]):
                ax1.text(j, i, f"{transition_matrix[i, j]:.2f}", 
                        ha="center", va="center", color="w" if transition_matrix[i, j] < 0.5 else "black")
        
        ax1.set_title('Transition Probability Matrix')
        ax1.set_xticks(np.arange(transition_matrix.shape[1]))
        ax1.set_yticks(np.arange(transition_matrix.shape[0]))
        ax1.set_xticklabels([f"State {i+1}" for i in range(transition_matrix.shape[1])])
        ax1.set_yticklabels([f"State {i+1}" for i in range(transition_matrix.shape[0])])
        ax1.set_xlabel('To State')
        ax1.set_ylabel('From State')
        plt.colorbar(im, ax=ax1)
        
        # Plot O-U parameters for each state
        ax2 = axes[1]
        lambda_vals = results['parameters']['lambda_vals']
        mu_vals = results['parameters']['mu_vals']
        sigma_vals = results['parameters']['sigma_vals']
        
        x = np.arange(len(lambda_vals))
        width = 0.2
        
        ax2.bar(x - width, lambda_vals, width, label='lambda (mean reversion)')
        ax2.bar(x, mu_vals, width, label='mu (long-term mean)')
        ax2.bar(x + width, sigma_vals, width, label='sigma (volatility)')
        
        ax2.set_title('O-U Parameters by State')
        ax2.set_xticks(x)
        ax2.set_xticklabels([f"State {i+1}" for i in range(len(lambda_vals))])
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()


# Example usage
if __name__ == "__main__":
    # Create strategy for a pair of stocks
    # Use actual Bloomberg tickers for the Chinese stocks mentioned in the paper
    strategy = RegimeSwitchingPairsTrading(
        stock1="000001",  # PAB (Ping An Bank)
        stock2="600000",  # SPDB (Shanghai Pudong Development Bank)
        formation_window=60,
        n_states=2,
        transaction_cost=0.01,
        stop_loss=0.5
    )
    
    print("Starting backtest for PAB-SPDB pair from Jan 2015 to Sep 2016")
    
    # Check if we need to create a test CSV
    try:
        csv_path = f"{strategy.stock1}_{strategy.stock2}_prices.csv"
        data = pd.read_csv(csv_path)
        
        # Validate data
        if (data[strategy.stock1] <= 0).any() or (data[strategy.stock2] <= 0).any() or \
           np.isnan(data[strategy.stock1]).any() or np.isnan(data[strategy.stock2]).any() or \
           np.isinf(data[strategy.stock1]).any() or np.isinf(data[strategy.stock2]).any():
            print("CSV file contains invalid data. Creating new test data...")
            # Rename problematic file
            os.rename(csv_path, f"{csv_path}.bak")
            # Create new test data
            strategy.create_test_csv("20150101", "20160918")
    except Exception as e:
        print(f"CSV check failed: {e}. Creating test data...")
        strategy.create_test_csv("20150101", "20160918")
    
    # Backtest the strategy
    results = strategy.backtest(
        start_date="20150101",
        end_date="20160918",
        rebalance_freq="M"
    )
    
    # Plot results
    strategy.plot_results(results)
    
    # Print performance metrics
    if 'metrics' in results:
        print("\nPerformance Metrics:")
        for key, value in results['metrics'].items():
            if isinstance(value, float):
                print(f"{key}: {value:.4f}")
            else:
                print(f"{key}: {value}")
    
    # Print model parameters
    print("\nModel Parameters:")
    print(f"Lambda values (mean reversion rates): {results['parameters']['lambda_vals']}")
    print(f"Mu values (long-term means): {results['parameters']['mu_vals']}")
    print(f"Sigma values (volatilities): {results['parameters']['sigma_vals']}")
    print("\nTransition probability matrix:")
    print(results['parameters']['transition_matrix'])