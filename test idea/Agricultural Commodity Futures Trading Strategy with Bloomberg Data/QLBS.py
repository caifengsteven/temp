import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from scipy.stats import norm
from scipy.interpolate import BSpline, splrep
import pdblp  # Bloomberg Python API
from datetime import datetime, timedelta
from typing import List, Tuple, Dict, Optional, Union, Callable
from mpl_toolkits.mplot3d import Axes3D  # Import for 3D plotting

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Try to connect to Bloomberg
BLOOMBERG_AVAILABLE = False
try:
    con = pdblp.BCon(debug=False, port=8194)
    con.start()
    BLOOMBERG_AVAILABLE = True
    print("Successfully connected to Bloomberg")
except:
    print("Bloomberg not available. Will use simulated data.")


# Helper functions for Black-Scholes calculations
def bs_d1(S, K, T, r, sigma):
    """Calculate d1 in Black-Scholes formula"""
    if T <= 0 or sigma <= 0:
        return 0.0
    return (np.log(S/K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))

def bs_d2(S, K, T, r, sigma):
    """Calculate d2 in Black-Scholes formula"""
    if T <= 0 or sigma <= 0:
        return 0.0
    return bs_d1(S, K, T, r, sigma) - sigma * np.sqrt(T)

def bs_call_price(S, K, T, r, sigma):
    """Calculate Black-Scholes call option price"""
    if T <= 0:
        return max(0, S - K)
    d1 = bs_d1(S, K, T, r, sigma)
    d2 = bs_d2(S, K, T, r, sigma)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def bs_put_price(S, K, T, r, sigma):
    """Calculate Black-Scholes put option price"""
    if T <= 0:
        return max(0, K - S)
    d1 = bs_d1(S, K, T, r, sigma)
    d2 = bs_d2(S, K, T, r, sigma)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def bs_call_delta(S, K, T, r, sigma):
    """Calculate Black-Scholes call option delta"""
    if T <= 0:
        return 1.0 if S > K else 0.0
    d1 = bs_d1(S, K, T, r, sigma)
    return norm.cdf(d1)

def bs_put_delta(S, K, T, r, sigma):
    """Calculate Black-Scholes put option delta"""
    if T <= 0:
        return -1.0 if S < K else 0.0
    d1 = bs_d1(S, K, T, r, sigma)
    return norm.cdf(d1) - 1

def gbm_simulate(S0, mu, sigma, T, n_steps, n_paths):
    """
    Simulate geometric Brownian motion paths
    
    Parameters:
    -----------
    S0 : float
        Initial stock price
    mu : float
        Drift (annualized)
    sigma : float
        Volatility (annualized)
    T : float
        Time horizon in years
    n_steps : int
        Number of time steps
    n_paths : int
        Number of paths to simulate
        
    Returns:
    --------
    ndarray of shape (n_paths, n_steps+1)
        Simulated price paths
    """
    dt = T / n_steps
    
    # Generate random normal increments
    Z = np.random.normal(0, 1, size=(n_paths, n_steps))
    
    # Initialize the array for stock prices
    S = np.zeros((n_paths, n_steps + 1))
    S[:, 0] = S0
    
    # Simulate paths
    for t in range(1, n_steps + 1):
        S[:, t] = S[:, t-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z[:, t-1])
    
    return S


class BSplineBasis:
    """
    Class for creating and evaluating B-spline basis functions
    """
    def __init__(self, degree=3, n_basis=12, domain=None):
        self.degree = degree
        self.n_basis = n_basis
        self.domain = domain
        self.knots = None
        self.splines = []
        
    def fit(self, x_values):
        """Fit the B-spline basis to the provided x values"""
        if self.domain is None:
            self.domain = [np.min(x_values), np.max(x_values)]
        
        # Create knots including boundary knots with multiplicity degree+1
        interior_knots = np.linspace(self.domain[0], self.domain[1], self.n_basis - self.degree + 1)
        self.knots = np.concatenate([
            np.full(self.degree, self.domain[0]),
            interior_knots,
            np.full(self.degree, self.domain[1])
        ])
        
        self.splines = []
        # Create a BSpline object for each basis function
        for i in range(self.n_basis):
            c = np.zeros(self.n_basis)
            c[i] = 1.0
            self.splines.append(BSpline(self.knots, c, self.degree))
            
        return self
        
    def transform(self, x_values):
        """Transform x values into B-spline basis features"""
        if len(self.splines) == 0:
            raise ValueError("Basis functions not fitted. Call fit() first.")
        
        # Handle scalar input
        x_values = np.atleast_1d(x_values)
        
        # Clip x values to domain to avoid extrapolation issues
        x_clipped = np.clip(x_values, self.domain[0], self.domain[1])
        
        # Evaluate each basis function at each x value
        features = np.zeros((len(x_clipped), self.n_basis))
        for i, spline in enumerate(self.splines):
            features[:, i] = spline(x_clipped)
            
        return features


class QLBSModel:
    """
    Q-Learning Black-Scholes (QLBS) model for option pricing and hedging
    """
    def __init__(self, 
                 risk_aversion: float = 0.001, 
                 discount_factor: float = 0.97,
                 n_basis: int = 12, 
                 degree: int = 3,
                 regularization: float = 1e-3,
                 option_type: str = 'put'):
        """
        Initialize the QLBS model
        
        Parameters:
        -----------
        risk_aversion : float
            Markowitz risk aversion parameter λ
        discount_factor : float
            Discount factor γ for future rewards
        n_basis : int
            Number of basis functions for state representation
        degree : int
            Degree of B-spline basis functions
        regularization : float
            Regularization parameter for matrix inversion
        option_type : str
            Type of option ('call' or 'put')
        """
        self.risk_aversion = risk_aversion
        self.gamma = discount_factor
        self.n_basis = n_basis
        self.degree = degree
        self.regularization = regularization
        self.option_type = option_type.lower()
        
        # Validate option type
        if self.option_type not in ['call', 'put']:
            raise ValueError("Option type must be 'call' or 'put'")
        
        # Initialize basis functions
        self.basis = BSplineBasis(degree=degree, n_basis=n_basis)
        
        # Model parameters
        self.W_matrix = []  # Store Q-function parameters
        self.phi_matrix = []  # Store hedge parameters
        
        # Initial values
        self.initial_price = None
        self.initial_hedge = None
        self.initial_cash = None
    
    def _compute_rewards(self, X_t, a_t, X_t_plus_1, S_t, S_t_plus_1, cash_t, cash_t_plus_1):
        """
        Compute one-step rewards based on equation (7) in the paper
        
        Parameters:
        -----------
        X_t : ndarray
            Current state
        a_t : ndarray
            Current action (hedge)
        X_t_plus_1 : ndarray
            Next state
        S_t : ndarray
            Current stock price
        S_t_plus_1 : ndarray
            Next stock price
        cash_t : ndarray
            Current cash position
        cash_t_plus_1 : ndarray
            Next cash position
            
        Returns:
        --------
        ndarray
            Rewards
        """
        # Calculate portfolio values
        pi_t = a_t * S_t + cash_t
        pi_t_plus_1 = a_t * S_t_plus_1 + cash_t_plus_1
        
        # Calculate portfolio changes
        delta_pi = pi_t_plus_1 - pi_t
        
        # Calculate expected portfolio changes (to get deviations)
        mean_delta_pi = np.mean(delta_pi)
        delta_pi_hat = delta_pi - mean_delta_pi
        
        # Calculate stock price changes (to get deviations)
        delta_S = S_t_plus_1 - S_t
        mean_delta_S = np.mean(delta_S)
        delta_S_hat = delta_S - mean_delta_S
        
        # Compute rewards according to equation (7)
        rewards = self.gamma * a_t * delta_S - self.risk_aversion * (delta_pi_hat**2)
        
        return rewards
    
    def fit_dp(self, S_paths, r, T, K, n_steps):
        """
        Fit the model using Dynamic Programming (DP) solution
        
        Parameters:
        -----------
        S_paths : ndarray
            Stock price paths of shape (n_paths, n_steps+1)
        r : float
            Risk-free rate
        T : float
            Time to maturity in years
        K : float
            Strike price
        n_steps : int
            Number of time steps
        """
        n_paths, n_time_steps = S_paths.shape
        n_time_steps -= 1  # Adjust because paths include t=0
        
        dt = T / n_steps
        mu = r  # Risk-neutral drift
        
        # Convert stock prices to state variables X as per equation (3)
        X_paths = np.log(S_paths) - (mu - 0.5 * (np.std(np.log(S_paths[:, 1:] / S_paths[:, :-1])) / np.sqrt(dt))**2) * np.linspace(0, T, n_steps+1).reshape(1, -1)
        
        # Fit the basis functions to the state space
        self.basis.fit(X_paths.flatten())
        
        # Initialize matrices to store parameters
        self.W_matrix = [None] * (n_steps + 1)
        self.phi_matrix = [None] * (n_steps + 1)
        
        # Initialize arrays to store portfolio values and hedges
        a_optimal = np.zeros((n_paths, n_steps + 1))
        cash = np.zeros((n_paths, n_steps + 1))
        portfolio = np.zeros((n_paths, n_steps + 1))
        
        # Terminal conditions
        if self.option_type == 'call':
            payoff = np.maximum(0, S_paths[:, -1] - K)
        else:  # put
            payoff = np.maximum(0, K - S_paths[:, -1])
        
        # At maturity, the hedge position is 0 and cash equals payoff
        a_optimal[:, -1] = 0
        cash[:, -1] = payoff
        portfolio[:, -1] = payoff
        
        # Store terminal Q-function parameters
        X_terminal = X_paths[:, -1]
        phi_features = self.basis.transform(X_terminal)
        self.phi_matrix[-1] = np.zeros(self.n_basis)  # No hedge at maturity
        
        # Terminal Q-function is just the negative of the terminal portfolio value
        terminal_portfolio = payoff
        W_features = np.column_stack([
            phi_features,
            np.zeros((n_paths, self.n_basis)),
            np.zeros((n_paths, self.n_basis))
        ])
        self.W_matrix[-1] = np.linalg.lstsq(W_features, -terminal_portfolio, rcond=None)[0]
        
        # Backward recursion
        for t in range(n_steps - 1, -1, -1):
            X_t = X_paths[:, t]
            S_t = S_paths[:, t]
            X_t_plus_1 = X_paths[:, t + 1]
            S_t_plus_1 = S_paths[:, t + 1]
            
            # Transform states to features
            phi_features_t = self.basis.transform(X_t)
            
            # Calculate optimal hedge using equation (18) from the paper
            # For the second-to-last step, we can use delta from the terminal condition
            if t == n_steps - 1:
                # For the last step, we can approximate delta from the payoff function
                if self.option_type == 'call':
                    # For a call option, delta is approximately 1 for in-the-money options and 0 for out-of-the-money
                    a_t = np.where(S_t_plus_1 > K, 1.0, 0.0)
                else:  # put
                    # For a put option, delta is approximately -1 for in-the-money options and 0 for out-of-the-money
                    a_t = np.where(S_t_plus_1 < K, -1.0, 0.0)
            else:
                # For earlier time steps, use the covariance method from equation (18)
                # Calculate portfolio values at next step
                portfolio_t_plus_1 = portfolio[:, t+1]
                
                # Calculate means for deviations
                mean_portfolio = np.mean(portfolio_t_plus_1)
                mean_stock = np.mean(S_t_plus_1)
                
                # Calculate deviations from means
                portfolio_dev = portfolio_t_plus_1 - mean_portfolio
                stock_dev = S_t_plus_1 - mean_stock
                
                # Optimal hedge from equation (18) - using covariance of stock and portfolio
                cov_term = np.mean(stock_dev * portfolio_dev)
                var_term = np.mean(stock_dev ** 2)
                
                # Avoid division by zero
                if var_term > 1e-10:
                    hedge_ratio = cov_term / var_term
                else:
                    hedge_ratio = 0.0
                    
                # Apply the same hedge to all paths
                a_t = np.full(n_paths, hedge_ratio)
            
            # Store optimal action for this time step
            a_optimal[:, t] = a_t
            
            # Fit optimal hedge as function of state
            A_matrix = phi_features_t.T @ phi_features_t + self.regularization * np.eye(self.n_basis)
            b_vector = phi_features_t.T @ a_t
            self.phi_matrix[t] = np.linalg.solve(A_matrix, b_vector)
            
            # Update portfolio and cash positions
            portfolio[:, t] = a_t * S_t + (portfolio[:, t+1] - a_t * S_t_plus_1) * np.exp(-r * dt)
            cash[:, t] = portfolio[:, t] - a_t * S_t
            
            # Calculate rewards for Q-function fitting
            delta_portfolio = portfolio[:, t+1] - portfolio[:, t]
            mean_delta_portfolio = np.mean(delta_portfolio)
            delta_portfolio_dev = delta_portfolio - mean_delta_portfolio
            
            delta_stock = S_t_plus_1 - S_t
            mean_delta_stock = np.mean(delta_stock)
            delta_stock_dev = delta_stock - mean_delta_stock
            
            # Calculate Q-function target
            rewards = (self.gamma * a_t * delta_stock - 
                      self.risk_aversion * delta_portfolio_dev**2)
                    
            # Create features for Q-function
            Q_features = np.zeros((n_paths, 3 * self.n_basis))
            Q_features[:, :self.n_basis] = phi_features_t  # constant term
            Q_features[:, self.n_basis:2*self.n_basis] = phi_features_t * a_t.reshape(-1, 1)  # linear term
            Q_features[:, 2*self.n_basis:] = phi_features_t * (a_t**2).reshape(-1, 1)  # quadratic term
            
            # Calculate next state optimal Q-values
            phi_features_t_plus_1 = self.basis.transform(X_t_plus_1)
            next_optimal_actions = phi_features_t_plus_1 @ self.phi_matrix[t + 1]
            
            # Create features for next state Q-values
            Q_features_next = np.zeros((n_paths, 3 * self.n_basis))
            Q_features_next[:, :self.n_basis] = phi_features_t_plus_1  # constant term
            Q_features_next[:, self.n_basis:2*self.n_basis] = phi_features_t_plus_1 * next_optimal_actions.reshape(-1, 1)  # linear term
            Q_features_next[:, 2*self.n_basis:] = phi_features_t_plus_1 * (next_optimal_actions**2).reshape(-1, 1)  # quadratic term
            
            # Calculate target Q-values
            next_Q_values = Q_features_next @ self.W_matrix[t + 1]
            target_Q = rewards + self.gamma * next_Q_values
            
            # Fit Q-function parameters
            C_matrix = Q_features.T @ Q_features + self.regularization * np.eye(3 * self.n_basis)
            d_vector = Q_features.T @ target_Q
            self.W_matrix[t] = np.linalg.solve(C_matrix, d_vector)
        
        # Calculate initial option price (negative of optimal value function)
        X0 = X_paths[:, 0]
        S0 = S_paths[:, 0]
        phi_features_0 = self.basis.transform(X0)
        
        # Get optimal initial hedge
        a0 = phi_features_0 @ self.phi_matrix[0]
        
        # Create features for initial Q-value
        Q_features_0 = np.zeros((n_paths, 3 * self.n_basis))
        Q_features_0[:, :self.n_basis] = phi_features_0  # constant term
        Q_features_0[:, self.n_basis:2*self.n_basis] = phi_features_0 * a0.reshape(-1, 1)  # linear term
        Q_features_0[:, 2*self.n_basis:] = phi_features_0 * (a0**2).reshape(-1, 1)  # quadratic term
        
        # Initial Q-value (negative of option price)
        Q0 = Q_features_0 @ self.W_matrix[0]
        
        # Option price is negative of the optimal value function
        option_price = -np.mean(Q0)
        
        # Store initial values for later use
        self.initial_price = option_price
        self.initial_hedge = np.mean(a0)
        self.initial_cash = np.mean(cash[:, 0])
        
        return self
    
    def fit_rl(self, X_t, a_t, X_t_plus_1, S_t, S_t_plus_1, cash_t, cash_t_plus_1, r, dt):
        """
        Fit the model using Reinforcement Learning (RL) with Fitted Q Iteration
        
        Parameters:
        -----------
        X_t : ndarray
            Current states
        a_t : ndarray
            Current actions (hedges)
        X_t_plus_1 : ndarray
            Next states
        S_t : ndarray
            Current stock prices
        S_t_plus_1 : ndarray
            Next stock prices
        cash_t : ndarray
            Current cash positions
        cash_t_plus_1 : ndarray
            Next cash positions
        r : float
            Risk-free rate
        dt : float
            Time step size
        """
        # Fit the basis functions to the state space
        all_states = np.concatenate([X_t, X_t_plus_1])
        self.basis.fit(all_states)
        
        # Calculate rewards
        rewards = self._compute_rewards(X_t, a_t, X_t_plus_1, S_t, S_t_plus_1, cash_t, cash_t_plus_1)
        
        # Transform states to features
        phi_features_t = self.basis.transform(X_t)
        phi_features_t_plus_1 = self.basis.transform(X_t_plus_1)
        
        # Create features for Q-function
        Q_features = np.zeros((len(X_t), 3 * self.n_basis))
        Q_features[:, :self.n_basis] = phi_features_t  # constant term
        Q_features[:, self.n_basis:2*self.n_basis] = phi_features_t * a_t.reshape(-1, 1)  # linear term
        Q_features[:, 2*self.n_basis:] = phi_features_t * (a_t**2).reshape(-1, 1)  # quadratic term
        
        # Fit Q-function using Fitted Q Iteration (FQI)
        # First, we need to estimate the optimal hedge (action) for next states
        A_matrix = phi_features_t.T @ phi_features_t + self.regularization * np.eye(self.n_basis)
        b_vector = phi_features_t.T @ a_t
        phi_params = np.linalg.solve(A_matrix, b_vector)
        
        # Get optimal next actions
        a_t_plus_1 = phi_features_t_plus_1 @ phi_params
        
        # Create features for next state Q-values
        Q_features_next = np.zeros((len(X_t_plus_1), 3 * self.n_basis))
        Q_features_next[:, :self.n_basis] = phi_features_t_plus_1  # constant term
        Q_features_next[:, self.n_basis:2*self.n_basis] = phi_features_t_plus_1 * a_t_plus_1.reshape(-1, 1)  # linear term
        Q_features_next[:, 2*self.n_basis:] = phi_features_t_plus_1 * (a_t_plus_1**2).reshape(-1, 1)  # quadratic term
        
        # Initial guess for W parameters
        W_params = np.zeros(3 * self.n_basis)
        
        # Iterate to converge to optimal Q-function
        for _ in range(10):  # Usually converges quickly
            # Calculate next state Q-values
            Q_next = Q_features_next @ W_params
            
            # Calculate target Q-values
            target_Q = rewards + self.gamma * Q_next
            
            # Update W parameters
            C_matrix = Q_features.T @ Q_features + self.regularization * np.eye(3 * self.n_basis)
            d_vector = Q_features.T @ target_Q
            W_params = np.linalg.solve(C_matrix, d_vector)
        
        # Store the parameters
        self.W_matrix = [W_params]
        self.phi_matrix = [phi_params]
        
        # Calculate option price (negative of optimal value function)
        # Use the first state (assuming it's the initial state)
        X0 = X_t[0]
        S0 = S_t[0]
        phi_features_0 = self.basis.transform([X0])
        
        # Get optimal initial hedge
        a0 = phi_features_0 @ phi_params
        
        # Create features for initial Q-value
        Q_features_0 = np.zeros((1, 3 * self.n_basis))
        Q_features_0[0, :self.n_basis] = phi_features_0  # constant term
        Q_features_0[0, self.n_basis:2*self.n_basis] = phi_features_0 * a0  # linear term
        Q_features_0[0, 2*self.n_basis:] = phi_features_0 * (a0**2)  # quadratic term
        
        # Initial Q-value (negative of option price)
        Q0 = Q_features_0 @ W_params
        
        # Option price is negative of the optimal value function
        option_price = -Q0[0]
        
        # Store initial values for later use
        self.initial_price = option_price
        self.initial_hedge = a0[0]
        self.initial_cash = cash_t[0]
        
        return self
    
    def fit_irl(self, X_t, a_t, X_t_plus_1, S_t, S_t_plus_1, r, dt, lambda_grid=None):
        """
        Fit the model using Inverse Reinforcement Learning (IRL)
        
        Parameters:
        -----------
        X_t : ndarray
            Current states
        a_t : ndarray
            Current actions (hedges)
        X_t_plus_1 : ndarray
            Next states
        S_t : ndarray
            Current stock prices
        S_t_plus_1 : ndarray
            Next stock prices
        r : float
            Risk-free rate
        dt : float
            Time step size
        lambda_grid : ndarray
            Grid of risk aversion values to search over
        """
        if lambda_grid is None:
            lambda_grid = np.logspace(-4, -1, 20)
        
        # Fit the basis functions to the state space
        all_states = np.concatenate([X_t, X_t_plus_1])
        self.basis.fit(all_states)
        
        # Transform states to features
        phi_features_t = self.basis.transform(X_t)
        
        # Calculate expected rewards for different lambda values
        log_likelihoods = []
        
        for lambda_val in lambda_grid:
            # Calculate coefficients c0, c1, c2 from equation (33)
            # For portfolio changes
            delta_S = S_t_plus_1 - S_t
            mean_delta_S = np.mean(delta_S)
            delta_S_hat = delta_S - mean_delta_S
            
            # Estimate portfolio values (without knowing cash)
            delta_pi_hat_est = a_t * delta_S_hat
            
            c0 = -lambda_val * self.gamma**2 * np.mean(delta_pi_hat_est**2)
            c1 = self.gamma * mean_delta_S + 2 * lambda_val * self.gamma * np.mean(delta_S_hat * delta_pi_hat_est)
            c2 = 2 * lambda_val * self.gamma**2 * np.mean(delta_S_hat**2)
            
            # Calculate log-likelihood as per equation (35)
            log_likelihood = 0
            for i in range(len(X_t)):
                # Probability of observed action under the model
                expected_action = c1 / c2
                action_variance = 1.0 / c2
                log_prob = -0.5 * np.log(2 * np.pi * action_variance) - 0.5 * (a_t[i] - expected_action)**2 / action_variance
                log_likelihood += log_prob
            
            log_likelihoods.append(log_likelihood)
        
        # Find lambda with maximum likelihood
        best_idx = np.argmax(log_likelihoods)
        best_lambda = lambda_grid[best_idx]
        
        # Update risk aversion parameter
        self.risk_aversion = best_lambda
        
        # Now that we have estimated lambda, use RL to complete the model
        # We need to estimate cash positions (since we don't observe them in IRL)
        cash_t = np.zeros_like(S_t)
        cash_t_plus_1 = np.zeros_like(S_t_plus_1)
        
        # Discount factor for the time step
        discount = np.exp(-r * dt)
        
        # Estimate cash positions based on the action and risk-neutral pricing
        for i in range(len(S_t)):
            # Assuming risk-neutral drift
            expected_S_t_plus_1 = S_t[i] * discount
            
            # Estimate cash needed to maintain portfolio value
            cash_t_plus_1[i] = 0  # Simplification: assume zero cash at t+1
            cash_t[i] = discount * cash_t_plus_1[i] - a_t[i] * (S_t_plus_1[i] - expected_S_t_plus_1)
        
        # Now use RL with estimated rewards
        return self.fit_rl(X_t, a_t, X_t_plus_1, S_t, S_t_plus_1, cash_t, cash_t_plus_1, r, dt)
    
    def predict_price(self, X, S):
        """
        Predict option price for given state
        
        Parameters:
        -----------
        X : float or ndarray
            State variable(s)
        S : float or ndarray
            Stock price(s)
            
        Returns:
        --------
        float or ndarray
            Option price(s)
        """
        if len(self.W_matrix) == 0:
            raise ValueError("Model not fitted yet. Call fit_dp(), fit_rl(), or fit_irl() first.")
        
        # Ensure inputs are arrays
        X = np.atleast_1d(X)
        S = np.atleast_1d(S)
        
        # Transform states to features
        phi_features = self.basis.transform(X)
        
        # Get optimal hedge
        a = phi_features @ self.phi_matrix[0]
        
        # Create features for Q-function
        Q_features = np.zeros((len(X), 3 * self.n_basis))
        Q_features[:, :self.n_basis] = phi_features  # constant term
        Q_features[:, self.n_basis:2*self.n_basis] = phi_features * a.reshape(-1, 1)  # linear term
        Q_features[:, 2*self.n_basis:] = phi_features * (a**2).reshape(-1, 1)  # quadratic term
        
        # Calculate Q-value (negative of option price)
        Q = Q_features @ self.W_matrix[0]
        
        # Option price is negative of the optimal value function
        option_price = -Q
        
        return option_price[0] if len(option_price) == 1 else option_price
    
    def predict_hedge(self, X, S):
        """
        Predict optimal hedge for given state
        
        Parameters:
        -----------
        X : float or ndarray
            State variable(s)
        S : float or ndarray
            Stock price(s)
            
        Returns:
        --------
        float or ndarray
            Optimal hedge position(s)
        """
        if len(self.phi_matrix) == 0:
            raise ValueError("Model not fitted yet. Call fit_dp(), fit_rl(), or fit_irl() first.")
        
        # Ensure inputs are arrays
        X = np.atleast_1d(X)
        
        # Transform states to features
        phi_features = self.basis.transform(X)
        
        # Get optimal hedge
        a = phi_features @ self.phi_matrix[0]
        
        return a[0] if len(a) == 1 else a


def fetch_bloomberg_data(ticker, start_date, end_date, fields=None):
    """
    Fetch historical data from Bloomberg
    
    Parameters:
    -----------
    ticker : str
        Bloomberg ticker
    start_date : str
        Start date in format 'YYYYMMDD'
    end_date : str
        End date in format 'YYYYMMDD'
    fields : list
        Bloomberg fields to fetch
        
    Returns:
    --------
    pandas.DataFrame
        Historical data
    """
    if not BLOOMBERG_AVAILABLE:
        raise ValueError("Bloomberg API not available")
    
    if fields is None:
        fields = ['PX_LAST', 'OPEN', 'HIGH', 'LOW', 'VOLUME']
    
    data = con.bdh(tickers=ticker, flds=fields, start_date=start_date, end_date=end_date)
    
    return data


def fetch_option_chain(ticker, date=None):
    """
    Fetch option chain from Bloomberg
    
    Parameters:
    -----------
    ticker : str
        Bloomberg ticker for the underlying
    date : str
        Date for which to get the option chain in format 'YYYYMMDD'
        
    Returns:
    --------
    pandas.DataFrame
        Option chain data
    """
    if not BLOOMBERG_AVAILABLE:
        raise ValueError("Bloomberg API not available")
    
    # If date is not specified, use current date
    if date is None:
        date = datetime.now().strftime('%Y%m%d')
    
    # Get all available options for the ticker
    options_data = con.ref(ticker, "OPT_CHAIN", date)
    
    # Get details for each option
    option_details = []
    for opt_ticker in options_data.values.flatten():
        # Get option details
        details = con.ref(opt_ticker, ["OPT_STRIKE", "OPT_EXPIRE_DT", "OPT_PUT_CALL", "PX_LAST"])
        details['Ticker'] = opt_ticker
        option_details.append(details)
    
    # Combine into a single DataFrame
    option_chain = pd.concat(option_details)
    
    return option_chain


def run_qlbs_simulation():
    """Run a simulation with the QLBS model and compare with Black-Scholes"""
    print("\n=== Running QLBS Model Simulation ===\n")
    
    # Simulation parameters
    S0 = 100.0  # Initial stock price
    K = 100.0   # Strike price
    T = 1.0     # Time to maturity (years)
    r = 0.03    # Risk-free rate
    sigma = 0.15  # Volatility
    n_steps = 24  # Number of time steps (bi-weekly rehedging)
    n_paths = 50000  # Number of simulation paths
    
    # Lambda values to test
    lambdas = [0.0001, 0.001, 0.01]
    
    # Simulate stock price paths
    print(f"Simulating {n_paths} stock price paths...")
    S_paths = gbm_simulate(S0, r, sigma, T, n_steps, n_paths)
    
    # Black-Scholes price for comparison
    bs_put_value = bs_put_price(S0, K, T, r, sigma)
    bs_call_value = bs_call_price(S0, K, T, r, sigma)
    
    print(f"Black-Scholes Put Price: ${bs_put_value:.4f}")
    print(f"Black-Scholes Call Price: ${bs_call_value:.4f}")
    
    # Display QLBS prices for different risk aversion parameters
    for option_type in ['put', 'call']:
        print(f"\n--- {option_type.upper()} Option Results ---")
        
        for lambda_val in lambdas:
            # Create and fit QLBS model using DP
            qlbs = QLBSModel(risk_aversion=lambda_val, option_type=option_type)
            qlbs.fit_dp(S_paths, r, T, K, n_steps)
            
            # Get QLBS price and hedge
            qlbs_price = qlbs.initial_price
            qlbs_hedge = qlbs.initial_hedge
            
            # Compare with Black-Scholes
            bs_price = bs_put_value if option_type == 'put' else bs_call_value
            bs_delta = bs_put_delta(S0, K, T, r, sigma) if option_type == 'put' else bs_call_delta(S0, K, T, r, sigma)
            
            print(f"Risk aversion λ = {lambda_val:.4f}:")
            print(f"  QLBS Price: ${qlbs_price:.4f} (vs BS: ${bs_price:.4f}, diff: ${qlbs_price - bs_price:.4f})")
            print(f"  QLBS Hedge: {qlbs_hedge:.4f} (vs BS Delta: {bs_delta:.4f}, diff: {qlbs_hedge - bs_delta:.4f})")
    
    # Simulate a path and show hedging performance
    print("\n=== Hedging Simulation ===")
    
    # Create a single test path
    np.random.seed(123)  # For reproducibility
    test_path = gbm_simulate(S0, r, sigma, T, n_steps, 1)[0]
    
    # Create and fit QLBS model for put option
    qlbs_put = QLBSModel(risk_aversion=0.001, option_type='put')
    qlbs_put.fit_dp(S_paths, r, T, K, n_steps)
    
    # Convert stock prices to state variables for prediction
    dt = T / n_steps
    times = np.linspace(0, T, n_steps + 1)
    X_test = np.log(test_path) - (r - 0.5 * sigma**2) * times
    
    # Initialize portfolio for BS and QLBS
    bs_hedge = np.zeros(n_steps + 1)
    bs_cash = np.zeros(n_steps + 1)
    bs_portfolio = np.zeros(n_steps + 1)
    
    qlbs_hedge = np.zeros(n_steps + 1)
    qlbs_cash = np.zeros(n_steps + 1)
    qlbs_portfolio = np.zeros(n_steps + 1)
    
    # Initial hedges and portfolios
    bs_hedge[0] = bs_put_delta(test_path[0], K, T, r, sigma)
    bs_cash[0] = bs_put_value - bs_hedge[0] * test_path[0]
    bs_portfolio[0] = bs_hedge[0] * test_path[0] + bs_cash[0]
    
    qlbs_hedge[0] = qlbs_put.predict_hedge(X_test[0], test_path[0])
    qlbs_cash[0] = qlbs_put.initial_price - qlbs_hedge[0] * test_path[0]
    qlbs_portfolio[0] = qlbs_hedge[0] * test_path[0] + qlbs_cash[0]
    
    # Simulate hedging over time
    for t in range(1, n_steps + 1):
        # Time remaining
        tau = T - t * dt
        
        # Black-Scholes hedge
        if tau > 0:
            bs_hedge[t] = bs_put_delta(test_path[t], K, tau, r, sigma)
        else:
            bs_hedge[t] = 0
        
        # QLBS hedge
        if t < n_steps:
            qlbs_hedge[t] = qlbs_put.predict_hedge(X_test[t], test_path[t])
        else:
            qlbs_hedge[t] = 0
        
        # Update cash positions (self-financing)
        bs_cash[t] = bs_cash[t-1] * np.exp(r * dt) - (bs_hedge[t] - bs_hedge[t-1]) * test_path[t]
        qlbs_cash[t] = qlbs_cash[t-1] * np.exp(r * dt) - (qlbs_hedge[t] - qlbs_hedge[t-1]) * test_path[t]
        
        # Update portfolio values
        bs_portfolio[t] = bs_hedge[t] * test_path[t] + bs_cash[t]
        qlbs_portfolio[t] = qlbs_hedge[t] * test_path[t] + qlbs_cash[t]
    
    # Calculate final payoff
    put_payoff = max(0, K - test_path[-1])
    
    # Calculate P&L
    bs_pnl = bs_portfolio[-1] - put_payoff
    qlbs_pnl = qlbs_portfolio[-1] - put_payoff
    
    print(f"Final Stock Price: ${test_path[-1]:.2f}")
    print(f"Put Option Payoff: ${put_payoff:.2f}")
    print(f"BS Final Portfolio Value: ${bs_portfolio[-1]:.2f}, P&L: ${bs_pnl:.2f}")
    print(f"QLBS Final Portfolio Value: ${qlbs_portfolio[-1]:.2f}, P&L: ${qlbs_pnl:.2f}")
    
    # Plot results
    plt.figure(figsize=(15, 10))
    
    # Stock price path
    plt.subplot(2, 2, 1)
    plt.plot(times, test_path)
    plt.title('Stock Price Path')
    plt.xlabel('Time (years)')
    plt.ylabel('Price')
    plt.grid(True)
    
    # Hedges
    plt.subplot(2, 2, 2)
    plt.plot(times, bs_hedge, label='BS Hedge')
    plt.plot(times, qlbs_hedge, label='QLBS Hedge')
    plt.title('Hedging Positions')
    plt.xlabel('Time (years)')
    plt.ylabel('Position')
    plt.legend()
    plt.grid(True)
    
    # Portfolio values
    plt.subplot(2, 2, 3)
    plt.plot(times, bs_portfolio, label='BS Portfolio')
    plt.plot(times, qlbs_portfolio, label='QLBS Portfolio')
    plt.axhline(y=put_payoff, color='r', linestyle='--', label='Option Payoff')
    plt.title('Portfolio Values')
    plt.xlabel('Time (years)')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    
    # P&L
    plt.subplot(2, 2, 4)
    plt.plot(times, bs_portfolio - bs_portfolio[0], label='BS P&L')
    plt.plot(times, qlbs_portfolio - qlbs_portfolio[0], label='QLBS P&L')
    plt.title('Cumulative P&L')
    plt.xlabel('Time (years)')
    plt.ylabel('P&L')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('qlbs_simulation.png')
    plt.close()
    
    print("\nSimulation results plotted to 'qlbs_simulation.png'")


def run_rl_vs_dp_comparison():
    """Compare RL and DP solutions for the QLBS model"""
    print("\n=== Comparing RL and DP Solutions ===\n")
    
    # Simulation parameters
    S0 = 100.0  # Initial stock price
    K = 100.0   # Strike price
    T = 1.0     # Time to maturity (years)
    r = 0.03    # Risk-free rate
    sigma = 0.15  # Volatility
    n_steps = 24  # Number of time steps (bi-weekly rehedging)
    n_paths = 20000  # Number of simulation paths
    
    # Lambda values to test
    lambda_val = 0.001
    
    # Simulate stock price paths
    print(f"Simulating {n_paths} stock price paths...")
    S_paths = gbm_simulate(S0, r, sigma, T, n_steps, n_paths)
    
    # Create and fit QLBS model using DP
    print("Fitting DP model...")
    qlbs_dp = QLBSModel(risk_aversion=lambda_val, option_type='put')
    qlbs_dp.fit_dp(S_paths, r, T, K, n_steps)
    
    dp_price = qlbs_dp.initial_price
    dp_hedge = qlbs_dp.initial_hedge
    
    print(f"DP Solution - Price: ${dp_price:.4f}, Hedge: {dp_hedge:.4f}")
    
    # Prepare data for RL and IRL
    dt = T / n_steps
    times = np.linspace(0, T, n_steps)
    
    # Convert stock prices to state variables
    X_paths = np.log(S_paths) - (r - 0.5 * sigma**2) * np.linspace(0, T, n_steps+1).reshape(1, -1)
    
    # Extract state-action-reward-next-state tuples
    X_t = X_paths[:, :-1].flatten()
    X_t_plus_1 = X_paths[:, 1:].flatten()
    S_t = S_paths[:, :-1].flatten()
    S_t_plus_1 = S_paths[:, 1:].flatten()
    
    # Generate suboptimal actions with noise (for off-policy learning)
    noise_levels = [0.0, 0.15, 0.30, 0.50]
    
    for noise in noise_levels:
        print(f"\nRunning with noise level η = {noise:.2f}")
        
        # Create noisy actions
        # First, compute optimal hedges using DP solution
        optimal_hedges = np.zeros((n_paths, n_steps))
        for t in range(n_steps):
            for p in range(n_paths):
                optimal_hedges[p, t] = qlbs_dp.predict_hedge(X_paths[p, t], S_paths[p, t])
        
        # Add noise to hedges
        if noise > 0:
            noisy_hedges = optimal_hedges * np.random.uniform(1 - noise, 1 + noise, size=optimal_hedges.shape)
        else:
            noisy_hedges = optimal_hedges.copy()
        
        a_t = noisy_hedges.flatten()
        
        # Estimate cash positions (simplified)
        cash_t = np.zeros_like(S_t)
        cash_t_plus_1 = np.zeros_like(S_t_plus_1)
        
        for i in range(len(S_t)):
            # For simplicity, assume final cash position is option payoff
            if (i + 1) % n_steps == 0:  # Last step in a path
                cash_t_plus_1[i] = max(0, K - S_t_plus_1[i])
            
            # Backward calculate cash_t based on self-financing condition
            cash_t[i] = cash_t_plus_1[i] * np.exp(-r * dt) - a_t[i] * (S_t_plus_1[i] - S_t[i] * np.exp(r * dt))
        
        # Fit RL model
        print("Fitting RL model...")
        qlbs_rl = QLBSModel(risk_aversion=lambda_val, option_type='put')
        qlbs_rl.fit_rl(X_t, a_t, X_t_plus_1, S_t, S_t_plus_1, cash_t, cash_t_plus_1, r, dt)
        
        rl_price = qlbs_rl.initial_price
        rl_hedge = qlbs_rl.initial_hedge
        
        print(f"RL Solution - Price: ${rl_price:.4f}, Hedge: {rl_hedge:.4f}")
        print(f"Difference - Price: ${rl_price - dp_price:.4f}, Hedge: {rl_hedge - dp_hedge:.4f}")
        
        # Fit IRL model (we don't provide rewards)
        print("Fitting IRL model...")
        qlbs_irl = QLBSModel(option_type='put')  # Don't specify lambda, it will be estimated
        qlbs_irl.fit_irl(X_t, a_t, X_t_plus_1, S_t, S_t_plus_1, r, dt)
        
        irl_price = qlbs_irl.initial_price
        irl_hedge = qlbs_irl.initial_hedge
        irl_lambda = qlbs_irl.risk_aversion
        
        print(f"IRL Solution - Price: ${irl_price:.4f}, Hedge: {irl_hedge:.4f}")
        print(f"IRL Estimated λ: {irl_lambda:.6f} (True λ: {lambda_val:.6f})")
        print(f"Difference - Price: ${irl_price - dp_price:.4f}, Hedge: {irl_hedge - dp_hedge:.4f}")
    
    return qlbs_dp, qlbs_rl, qlbs_irl


def run_portfolio_analysis():
    """Analyze a portfolio of options using the QLBS model"""
    print("\n=== Portfolio Analysis with QLBS ===\n")
    
    # Parameters
    S0 = 100.0  # Initial stock price
    r = 0.03    # Risk-free rate
    sigma = 0.15  # Volatility
    T = 1.0     # Time to maturity (years)
    n_steps = 24  # Number of time steps (bi-weekly rehedging)
    n_paths = 20000  # Number of simulation paths
    lambda_val = 0.001  # Risk aversion
    
    # Define options in the portfolio
    options = [
        {'type': 'call', 'strike': 90.0, 'premium': bs_call_price(S0, 90.0, T, r, sigma)},
        {'type': 'call', 'strike': 100.0, 'premium': bs_call_price(S0, 100.0, T, r, sigma)},
        {'type': 'call', 'strike': 110.0, 'premium': bs_call_price(S0, 110.0, T, r, sigma)},
        {'type': 'put', 'strike': 90.0, 'premium': bs_put_price(S0, 90.0, T, r, sigma)},
        {'type': 'put', 'strike': 100.0, 'premium': bs_put_price(S0, 100.0, T, r, sigma)},
        {'type': 'put', 'strike': 110.0, 'premium': bs_put_price(S0, 110.0, T, r, sigma)}
    ]
    
    # Simulate stock price paths
    print(f"Simulating {n_paths} stock price paths...")
    S_paths = gbm_simulate(S0, r, sigma, T, n_steps, n_paths)
    
    # Convert stock prices to state variables
    X_paths = np.log(S_paths) - (r - 0.5 * sigma**2) * np.linspace(0, T, n_steps+1).reshape(1, -1)
    
    # Initialize results storage
    bs_prices = []
    qlbs_prices = []
    bs_deltas = []
    qlbs_hedges = []
    
    print("\nOption Prices and Hedges:")
    print("-" * 80)
    print(f"{'Type':<6} {'Strike':>8} {'BS Price':>10} {'QLBS Price':>12} {'Diff':>8} {'BS Delta':>10} {'QLBS Hedge':>12} {'Diff':>8}")
    print("-" * 80)
    
    # Analyze each option
    for option in options:
        opt_type = option['type']
        strike = option['strike']
        
        # Create and fit QLBS model
        qlbs = QLBSModel(risk_aversion=lambda_val, option_type=opt_type)
        qlbs.fit_dp(S_paths, r, T, strike, n_steps)
        
        # Get BS price and delta
        if opt_type == 'call':
            bs_price = bs_call_price(S0, strike, T, r, sigma)
            bs_delta = bs_call_delta(S0, strike, T, r, sigma)
        else:  # put
            bs_price = bs_put_price(S0, strike, T, r, sigma)
            bs_delta = bs_put_delta(S0, strike, T, r, sigma)
        
        # Get QLBS price and hedge
        qlbs_price = qlbs.initial_price
        qlbs_hedge = qlbs.initial_hedge
        
        # Store results
        bs_prices.append(bs_price)
        qlbs_prices.append(qlbs_price)
        bs_deltas.append(bs_delta)
        qlbs_hedges.append(qlbs_hedge)
        
        # Print results
        print(f"{opt_type:<6} {strike:>8.2f} {bs_price:>10.4f} {qlbs_price:>12.4f} {qlbs_price-bs_price:>8.4f} "
              f"{bs_delta:>10.4f} {qlbs_hedge:>12.4f} {qlbs_hedge-bs_delta:>8.4f}")
    
    # Create a portfolio with a specific strategy
    # For example, a Long Straddle: Long ATM Call + Long ATM Put
    call_idx = 1  # ATM Call (strike = 100)
    put_idx = 4   # ATM Put (strike = 100)
    
    # Calculate portfolio values
    portfolio_bs_price = bs_prices[call_idx] + bs_prices[put_idx]
    portfolio_qlbs_price = qlbs_prices[call_idx] + qlbs_prices[put_idx]
    portfolio_bs_delta = bs_deltas[call_idx] + bs_deltas[put_idx]
    portfolio_qlbs_hedge = qlbs_hedges[call_idx] + qlbs_hedges[put_idx]
    
    print("\nPortfolio Analysis (Long Straddle: ATM Call + ATM Put):")
    print(f"BS Portfolio Price: ${portfolio_bs_price:.4f}")
    print(f"QLBS Portfolio Price: ${portfolio_qlbs_price:.4f}")
    print(f"Difference: ${portfolio_qlbs_price - portfolio_bs_price:.4f}")
    print(f"BS Portfolio Delta: {portfolio_bs_delta:.4f}")
    print(f"QLBS Portfolio Hedge: {portfolio_qlbs_hedge:.4f}")
    print(f"Difference: {portfolio_qlbs_hedge - portfolio_bs_delta:.4f}")
    
    # Calculate implied volatility surface
    strikes = np.linspace(80, 120, 9)
    maturities = np.array([0.25, 0.5, 1.0])
    
    # Initialize arrays for implied volatilities
    call_iv_bs = np.zeros((len(maturities), len(strikes)))
    call_iv_qlbs = np.zeros((len(maturities), len(strikes)))
    put_iv_bs = np.zeros((len(maturities), len(strikes)))
    put_iv_qlbs = np.zeros((len(maturities), len(strikes)))
    
    print("\nCalculating Implied Volatility Surface...")
    
    for i, maturity in enumerate(maturities):
        # Simulate paths for this maturity
        S_paths_T = gbm_simulate(S0, r, sigma, maturity, int(n_steps * maturity), n_paths)
        
        for j, strike in enumerate(strikes):
            # Black-Scholes prices
            bs_call = bs_call_price(S0, strike, maturity, r, sigma)
            bs_put = bs_put_price(S0, strike, maturity, r, sigma)
            
            # QLBS models
            qlbs_call = QLBSModel(risk_aversion=lambda_val, option_type='call')
            qlbs_call.fit_dp(S_paths_T, r, maturity, strike, int(n_steps * maturity))
            
            qlbs_put = QLBSModel(risk_aversion=lambda_val, option_type='put')
            qlbs_put.fit_dp(S_paths_T, r, maturity, strike, int(n_steps * maturity))
            
            # QLBS prices
            qlbs_call_price_val = qlbs_call.initial_price
            qlbs_put_price_val = qlbs_put.initial_price
            
            # Calculate implied volatilities using bisection method
            # Function to find volatility that gives the target price
            def find_implied_vol(price_func, target_price, low=0.001, high=1.0, tol=1e-6, max_iter=100):
                for _ in range(max_iter):
                    mid = (low + high) / 2
                    price = price_func(S0, strike, maturity, r, mid)
                    
                    if abs(price - target_price) < tol:
                        return mid
                    
                    if price < target_price:
                        low = mid
                    else:
                        high = mid
                
                return mid
            
            # Calculate implied volatilities
            call_iv_bs[i, j] = sigma  # By construction
            call_iv_qlbs[i, j] = find_implied_vol(bs_call_price, qlbs_call_price_val)
            
            put_iv_bs[i, j] = sigma  # By construction
            put_iv_qlbs[i, j] = find_implied_vol(bs_put_price, qlbs_put_price_val)
    
    # Plot implied volatility smile
    plt.figure(figsize=(15, 10))
    
    for i, maturity in enumerate(maturities):
        plt.subplot(2, 2, i+1)
        plt.plot(strikes, call_iv_qlbs[i], 'o-', label=f'QLBS Call IV (T={maturity})')
        plt.plot(strikes, put_iv_qlbs[i], 'x-', label=f'QLBS Put IV (T={maturity})')
        plt.axhline(y=sigma, color='r', linestyle='--', label='BS Flat IV')
        plt.title(f'Implied Volatility Smile (T={maturity})')
        plt.xlabel('Strike')
        plt.ylabel('Implied Volatility')
        plt.legend()
        plt.grid(True)
    
    # Plot 3D volatility surface for calls
    ax = plt.subplot(2, 2, 4, projection='3d')
    X, Y = np.meshgrid(strikes, maturities)
    ax.plot_surface(X, Y, call_iv_qlbs, cmap='viridis', alpha=0.8)
    ax.set_title('QLBS Call Option Implied Volatility Surface')
    ax.set_xlabel('Strike')
    ax.set_ylabel('Maturity')
    ax.set_zlabel('Implied Volatility')
    
    plt.tight_layout()
    plt.savefig('qlbs_volatility_surface.png')
    plt.close()
    
    print("\nImplied volatility surface plotted to 'qlbs_volatility_surface.png'")


def run_bloomberg_analysis():
    """Run QLBS analysis with Bloomberg data if available"""
    if not BLOOMBERG_AVAILABLE:
        print("\nBloomberg API not available. Skipping real data analysis.")
        return
    
    print("\n=== Bloomberg Data Analysis with QLBS ===\n")
    
    # Get current date and 1 year ago
    today = datetime.now()
    start_date = (today - timedelta(days=365)).strftime('%Y%m%d')
    end_date = today.strftime('%Y%m%d')
    
    # Choose a ticker
    ticker = 'SPY US Equity'
    
    try:
        # Fetch historical stock data
        print(f"Fetching historical data for {ticker}...")
        stock_data = fetch_bloomberg_data(ticker, start_date, end_date)
        
        # Fetch current option chain
        print("Fetching option chain...")
        option_chain = fetch_option_chain(ticker)
        
        # Extract relevant data
        prices = stock_data['PX_LAST']
        S0 = prices.iloc[-1]  # Current price
        
        # Calculate historical volatility
        returns = np.log(prices / prices.shift(1)).dropna()
        annual_vol = returns.std() * np.sqrt(252)  # Annualized volatility
        
        # Get current risk-free rate (1-year Treasury yield as proxy)
        r = con.bdh('USGG1YR Index', 'PX_LAST', end_date, end_date).iloc[0, 0] / 100
        
        print(f"\nCurrent price: ${S0:.2f}")
        print(f"Historical volatility: {annual_vol:.4f}")
        print(f"Risk-free rate: {r:.4f}")
        
        # Filter options to analyze
        # For simplicity, focus on near-term ATM options
        near_term_options = option_chain[
            (option_chain['OPT_EXPIRE_DT'] > today) & 
            (option_chain['OPT_EXPIRE_DT'] < today + timedelta(days=90))
        ]
        
        if len(near_term_options) == 0:
            print("No suitable options found.")
            return
        
        # Find ATM options
        atm_strike = near_term_options['OPT_STRIKE'].values[
            np.abs(near_term_options['OPT_STRIKE'].values - S0).argmin()
        ]
        
        atm_options = near_term_options[near_term_options['OPT_STRIKE'] == atm_strike]
        
        if len(atm_options) < 2:  # Need at least one call and one put
            print("Not enough ATM options found.")
            return
        
        # Separate call and put
        atm_call = atm_options[atm_options['OPT_PUT_CALL'] == 'C'].iloc[0]
        atm_put = atm_options[atm_options['OPT_PUT_CALL'] == 'P'].iloc[0]
        
        # Extract option details
        call_ticker = atm_call['Ticker']
        call_strike = atm_call['OPT_STRIKE']
        call_expiry = atm_call['OPT_EXPIRE_DT']
        call_price = atm_call['PX_LAST']
        
        put_ticker = atm_put['Ticker']
        put_strike = atm_put['OPT_STRIKE']
        put_expiry = atm_put['OPT_EXPIRE_DT']
        put_price = atm_put['PX_LAST']
        
        # Calculate time to maturity in years
        T = (call_expiry - today).days / 365.0
        
        print(f"\nAnalyzing ATM options with strike ${atm_strike:.2f} and expiry {call_expiry.strftime('%Y-%m-%d')}")
        print(f"Time to maturity: {T:.4f} years")
        
        # Calculate implied volatility from market prices
        def find_implied_vol(price_func, market_price, S, K, T, r, low=0.001, high=1.0, tol=1e-6, max_iter=100):
            for _ in range(max_iter):
                mid = (low + high) / 2
                price = price_func(S, K, T, r, mid)
                
                if abs(price - market_price) < tol:
                    return mid
                
                if price < market_price:
                    low = mid
                else:
                    high = mid
            
            return mid
        
        call_iv = find_implied_vol(bs_call_price, call_price, S0, call_strike, T, r)
        put_iv = find_implied_vol(bs_put_price, put_price, S0, put_strike, T, r)
        
        print(f"Call option implied volatility: {call_iv:.4f}")
        print(f"Put option implied volatility: {put_iv:.4f}")
        
        # Use average implied volatility for simulations
        sigma = (call_iv + put_iv) / 2
        
        # Simulate paths for QLBS analysis
        n_steps = int(T * 50)  # Approx weekly rehedging
        n_paths = 20000
        
        print(f"\nSimulating {n_paths} stock price paths for QLBS analysis...")
        S_paths = gbm_simulate(S0, r, sigma, T, n_steps, n_paths)
        
        # Analyze with QLBS model with different risk aversions
        lambdas = [0.0001, 0.001, 0.01]
        
        print("\nQLBS Analysis Results:")
        print("-" * 70)
        print(f"{'Option':<10} {'λ':<8} {'Market':<10} {'BS':<10} {'QLBS':<10} {'Diff (BS)':<12} {'Diff (Market)':<12}")
        print("-" * 70)
        
        for lambda_val in lambdas:
            # Call option
            qlbs_call = QLBSModel(risk_aversion=lambda_val, option_type='call')
            qlbs_call.fit_dp(S_paths, r, T, call_strike, n_steps)
            qlbs_call_price = qlbs_call.initial_price
            
            bs_call_value = bs_call_price(S0, call_strike, T, r, sigma)
            
            print(f"{'Call':<10} {lambda_val:<8.4f} {call_price:<10.4f} {bs_call_value:<10.4f} {qlbs_call_price:<10.4f} "
                  f"{qlbs_call_price-bs_call_value:<12.4f} {qlbs_call_price-call_price:<12.4f}")
            
            # Put option
            qlbs_put = QLBSModel(risk_aversion=lambda_val, option_type='put')
            qlbs_put.fit_dp(S_paths, r, T, put_strike, n_steps)
            qlbs_put_price = qlbs_put.initial_price
            
            bs_put_value = bs_put_price(S0, put_strike, T, r, sigma)
            
            print(f"{'Put':<10} {lambda_val:<8.4f} {put_price:<10.4f} {bs_put_value:<10.4f} {qlbs_put_price:<10.4f} "
                  f"{qlbs_put_price-bs_put_value:<12.4f} {qlbs_put_price-put_price:<12.4f}")
        
        # Try to estimate risk aversion parameter using IRL
        print("\nEstimating risk aversion parameter using IRL...")
        
        # Fetch historical option data for delta hedges if available
        try:
            call_historical = fetch_bloomberg_data(call_ticker, start_date, end_date, fields=['PX_LAST', 'DELTA'])
            put_historical = fetch_bloomberg_data(put_ticker, start_date, end_date, fields=['PX_LAST', 'DELTA'])
            
            # Extract deltas
            call_deltas = call_historical['DELTA']
            put_deltas = put_historical['DELTA']
            
            print(f"Historical call option deltas: min={call_deltas.min():.4f}, max={call_deltas.max():.4f}, mean={call_deltas.mean():.4f}")
            print(f"Historical put option deltas: min={put_deltas.min():.4f}, max={put_deltas.max():.4f}, mean={put_deltas.mean():.4f}")
            
            # Run IRL to estimate risk aversion
            # This is a simplified example - in practice, you'd need time series of states, actions, etc.
            # Prepare data for IRL (simplified for illustration)
            n_samples = min(100, len(call_deltas))
            
            # Use most recent data
            X_t = np.random.normal(np.log(S0), 0.1, n_samples)
            a_t = call_deltas.iloc[-n_samples:].values
            X_t_plus_1 = X_t + np.random.normal(0, 0.02, n_samples)
            S_t = np.exp(X_t)
            S_t_plus_1 = np.exp(X_t_plus_1)
            
            # Run IRL
            qlbs_irl = QLBSModel(option_type='call')
            qlbs_irl.fit_irl(X_t, a_t, X_t_plus_1, S_t, S_t_plus_1, r, 1/252)
            
            print(f"IRL estimated risk aversion λ: {qlbs_irl.risk_aversion:.6f}")
            print(f"IRL option price: ${qlbs_irl.initial_price:.4f}")
            
        except Exception as e:
            print(f"Error fetching historical option data: {e}")
            print("Skipping IRL analysis.")
        
    except Exception as e:
        print(f"Error in Bloomberg analysis: {e}")


def main():
    """Main function to run QLBS experiments"""
    print("QLBS (Q-Learning Black-Scholes) Model Implementation")
    print("=" * 60)
    
    # Run simulation analysis
    run_qlbs_simulation()
    
    # Run RL vs DP comparison
    run_rl_vs_dp_comparison()
    
    # Run portfolio analysis with volatility smile
    run_portfolio_analysis()
    
    # Run Bloomberg analysis if available
    run_bloomberg_analysis()
    
    print("\nAll analyses complete!")


if __name__ == "__main__":
    main()