import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import norm
from scipy.optimize import minimize
from scipy.integrate import solve_ivp
import statsmodels.api as sm
from statsmodels.tsa.api import VAR
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

class CommonCausalPDEStrategy:
    """
    Implementation of the Portfolio's Common Causal Conditional Risk-Neutral PDE strategy.
    """
    
    def __init__(self, n_assets=5, n_drivers=3, window_size=60):
        """
        Initialize the strategy.
        
        Parameters:
        - n_assets: Number of assets in the portfolio
        - n_drivers: Number of common causal drivers to identify
        - window_size: Look-back window for calculations
        """
        self.n_assets = n_assets
        self.n_drivers = n_drivers
        self.window_size = window_size
        self.assets = None
        self.common_drivers = None
        self.correlations = None
        
    def identify_common_drivers(self, data, n_lags=1):
        """
        Identify common causal drivers of the assets.
        
        Parameters:
        - data: DataFrame with asset returns
        - n_lags: Number of lags to consider for causality
        
        Returns:
        - common_drivers: DataFrame with the common drivers
        """
        # Using lagged PCA as a simple proxy for common causal drivers
        # In a real implementation, this should use proper causal discovery methods
        
        # Create lagged data
        lagged_data = data.shift(n_lags).dropna()
        aligned_data = data.iloc[n_lags:].reset_index(drop=True)
        
        # Apply PCA to identify common components
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(lagged_data)
        
        pca = PCA(n_components=self.n_drivers)
        principal_components = pca.fit_transform(scaled_data)
        
        # Create a DataFrame with the drivers
        driver_names = [f'D{i+1}' for i in range(self.n_drivers)]
        common_drivers = pd.DataFrame(principal_components, columns=driver_names, index=aligned_data.index)
        
        # Normalize to [0,1] for copula calculation
        for col in common_drivers.columns:
            common_drivers[col] = stats.norm.cdf(common_drivers[col])
        
        self.common_drivers = common_drivers
        self.assets = aligned_data
        
        return common_drivers
    
    def calculate_correlations(self):
        """
        Calculate correlations between assets and common drivers.
        
        Returns:
        - correlations: Dictionary of correlation matrices
        """
        if self.assets is None or self.common_drivers is None:
            raise ValueError("Assets and common drivers must be identified first")
        
        correlations = {}
        
        for i, asset_col in enumerate(self.assets.columns):
            correlations[asset_col] = {}
            asset_data = self.assets[asset_col]
            
            for j, driver_col in enumerate(self.common_drivers.columns):
                driver_data = self.common_drivers[driver_col]
                # Calculate correlation using Gaussian copula
                u1 = stats.norm.cdf(asset_data)
                u2 = driver_data  # Already normalized to [0,1]
                
                # Convert to normal
                z1 = stats.norm.ppf(u1)
                z2 = stats.norm.ppf(u2)
                
                # Calculate correlation
                rho = np.corrcoef(z1, z2)[0, 1]
                correlations[asset_col][driver_col] = rho
        
        self.correlations = correlations
        return correlations
    
    def calculate_gaussian_copula_derivative(self, asset_val, driver_val, rho):
        """
        Calculate the partial derivative of the Gaussian copula with respect to the driver.
        
        Parameters:
        - asset_val: Asset value (normalized)
        - driver_val: Driver value (normalized)
        - rho: Correlation parameter
        
        Returns:
        - derivative: Partial derivative value
        """
        # Transform to standard normal
        x1 = stats.norm.ppf(asset_val)
        x2 = stats.norm.ppf(driver_val)
        
        # Calculate partial derivative of Gaussian copula w.r.t. driver
        numerator = np.exp(-(rho**2 * (x1**2 + x2**2) - 2*rho*x1*x2) / (2*(1-rho**2)))
        denominator = 2*np.pi*np.sqrt(1-rho**2)
        
        derivative = -numerator / denominator * (rho*x2 - x1) / np.sqrt(1-rho**2)
        
        return derivative
    
    def calculate_pi_matrix(self, asset_values, driver_values):
        """
        Calculate the Π matrix from equation (6) in the paper.
        
        Parameters:
        - asset_values: Asset values (normalized)
        - driver_values: Driver values (normalized)
        
        Returns:
        - pi_matrix: Matrix of copula partial derivatives
        """
        pi_matrix = np.zeros((self.n_assets, self.n_drivers))
        
        for i, asset_col in enumerate(self.assets.columns):
            for j, driver_col in enumerate(self.common_drivers.columns):
                rho = self.correlations[asset_col][driver_col]
                derivative = self.calculate_gaussian_copula_derivative(
                    asset_values[i], driver_values[j], rho
                )
                pi_matrix[i, j] = derivative
        
        return pi_matrix
    
    def solve_pde_system(self, asset_values, driver_values, weights, sigma_p, sigma_d, mu_d, dt=0.01, n_steps=100):
        """
        Solve the PDE system in equation (20) of the paper.
        
        Parameters:
        - asset_values: Asset values (normalized)
        - driver_values: Driver values (normalized)
        - weights: Portfolio weights
        - sigma_p: Asset volatilities
        - sigma_d: Driver volatilities
        - mu_d: Driver drifts
        - dt: Time step for numerical integration
        - n_steps: Number of steps to integrate
        
        Returns:
        - solution: Dictionary with the PDE solution
        """
        # Initial Pi matrix
        pi_matrix = self.calculate_pi_matrix(asset_values, driver_values)
        
        # Set up the system of PDEs
        def pde_system(t, pi_vector):
            pi_matrix = pi_vector.reshape(self.n_assets, self.n_drivers)
            
            # Calculate derivatives for the PDE
            d_pi_da = np.zeros((self.n_assets, self.n_drivers))
            d_pi_dd = np.zeros((self.n_assets, self.n_drivers))
            d2_pi_da2 = np.zeros((self.n_assets, self.n_drivers))
            d2_pi_dd2 = np.zeros((self.n_assets, self.n_drivers))
            d2_pi_dadd = np.zeros((self.n_assets, self.n_drivers))
            
            # Numerical approximation of derivatives (simple finite differences)
            epsilon = 1e-6
            
            for i in range(self.n_assets):
                for j in range(self.n_drivers):
                    # First derivatives
                    asset_plus = asset_values.copy()
                    asset_plus[i] += epsilon
                    driver_plus = driver_values.copy()
                    driver_plus[j] += epsilon
                    
                    pi_a_plus = self.calculate_gaussian_copula_derivative(
                        asset_plus[i], driver_values[j], self.correlations[self.assets.columns[i]][self.common_drivers.columns[j]]
                    )
                    pi_d_plus = self.calculate_gaussian_copula_derivative(
                        asset_values[i], driver_plus[j], self.correlations[self.assets.columns[i]][self.common_drivers.columns[j]]
                    )
                    
                    d_pi_da[i, j] = (pi_a_plus - pi_matrix[i, j]) / epsilon
                    d_pi_dd[i, j] = (pi_d_plus - pi_matrix[i, j]) / epsilon
                    
                    # Second derivatives
                    asset_plus_plus = asset_values.copy()
                    asset_plus_plus[i] += 2*epsilon
                    driver_plus_plus = driver_values.copy()
                    driver_plus_plus[j] += 2*epsilon
                    
                    pi_a_plus_plus = self.calculate_gaussian_copula_derivative(
                        asset_plus_plus[i], driver_values[j], self.correlations[self.assets.columns[i]][self.common_drivers.columns[j]]
                    )
                    pi_d_plus_plus = self.calculate_gaussian_copula_derivative(
                        asset_values[i], driver_plus_plus[j], self.correlations[self.assets.columns[i]][self.common_drivers.columns[j]]
                    )
                    
                    d2_pi_da2[i, j] = (pi_a_plus_plus - 2*pi_a_plus + pi_matrix[i, j]) / (epsilon**2)
                    d2_pi_dd2[i, j] = (pi_d_plus_plus - 2*pi_d_plus + pi_matrix[i, j]) / (epsilon**2)
                    
                    # Mixed derivative
                    asset_driver_plus = asset_values.copy()
                    asset_driver_plus[i] += epsilon
                    driver_asset_plus = driver_values.copy()
                    driver_asset_plus[j] += epsilon
                    
                    pi_ad_plus = self.calculate_gaussian_copula_derivative(
                        asset_driver_plus[i], driver_asset_plus[j], self.correlations[self.assets.columns[i]][self.common_drivers.columns[j]]
                    )
                    
                    d2_pi_dadd[i, j] = (pi_ad_plus - pi_a_plus - pi_d_plus + pi_matrix[i, j]) / (epsilon**2)
            
            # Implement equation (20) from the paper
            sigma_p_matrix = np.diag(sigma_p)
            sigma_d_matrix = np.diag(sigma_d)
            
            # Prepare driver matrix
            D = np.tile(driver_values, (self.n_assets, 1))
            
            # Calculate the terms in equation (20)
            term1 = 0.5 * np.sum(d2_pi_da2 * D * sigma_p_matrix.reshape(-1, 1))
            term2 = np.sum((d2_pi_dd2 * D + 2 * d_pi_dd) * sigma_d_matrix.reshape(1, -1))
            term3 = np.sum((d_pi_da + d2_pi_dadd * D) * sigma_p_matrix.reshape(-1, 1) * sigma_d_matrix.reshape(1, -1))
            term4 = np.sum(mu_d * D * pi_matrix)
            
            # Compute the derivative for each element of pi_matrix
            d_pi_dt = -(term1 + term2 + term3 + term4)
            
            return d_pi_dt.flatten()
        
        # Solve the PDE system
        pi_vector_initial = pi_matrix.flatten()
        t_span = (0, dt * n_steps)
        t_eval = np.linspace(0, dt * n_steps, n_steps + 1)
        
        solution = solve_ivp(
            pde_system, t_span, pi_vector_initial, 
            method='RK45', t_eval=t_eval
        )
        
        # Extract the solution
        pi_solution = solution.y.T.reshape(-1, self.n_assets, self.n_drivers)
        
        return {
            't': solution.t,
            'pi_solution': pi_solution
        }
    
    def calculate_conditional_probability(self, pi_matrix, driver_values, weights):
        """
        Calculate the conditional probability of the portfolio given the common drivers.
        
        Parameters:
        - pi_matrix: Matrix of copula partial derivatives
        - driver_values: Driver values (normalized)
        - weights: Portfolio weights
        
        Returns:
        - probability: Conditional probability value
        """
        # Implement equation (6) from the paper
        D = np.tile(driver_values, (self.n_assets, 1))
        probability = -np.sum(weights * np.sum(pi_matrix * D, axis=1))
        
        return probability
    
    def calculate_implied_volatility(self, pi_solution, driver_values, weights, sigma_p, dt=0.01):
        """
        Calculate the implied conditional portfolio volatility from the PDE solution.
        
        Parameters:
        - pi_solution: Solution of the PDE system
        - driver_values: Driver values (normalized)
        - weights: Portfolio weights
        - sigma_p: Asset volatilities
        - dt: Time step
        
        Returns:
        - implied_vols: Array of implied volatilities over time
        """
        n_steps = len(pi_solution)
        implied_vols = np.zeros(n_steps)
        
        for t in range(n_steps):
            pi_matrix = pi_solution[t]
            
            # Calculate the first term in equation (16)
            d2_pi_da2 = np.zeros((self.n_assets, self.n_drivers))
            
            # Numerical approximation of second derivative
            epsilon = 1e-6
            
            for i in range(self.n_assets):
                for j in range(self.n_drivers):
                    # Simple finite difference for second derivative
                    asset_values = np.array([stats.norm.ppf(0.5) for _ in range(self.n_assets)])
                    
                    asset_plus = asset_values.copy()
                    asset_plus[i] += epsilon
                    asset_minus = asset_values.copy()
                    asset_minus[i] -= epsilon
                    
                    pi_a_plus = self.calculate_gaussian_copula_derivative(
                        asset_plus[i], driver_values[j], self.correlations[self.assets.columns[i]][self.common_drivers.columns[j]]
                    )
                    pi_a = self.calculate_gaussian_copula_derivative(
                        asset_values[i], driver_values[j], self.correlations[self.assets.columns[i]][self.common_drivers.columns[j]]
                    )
                    pi_a_minus = self.calculate_gaussian_copula_derivative(
                        asset_minus[i], driver_values[j], self.correlations[self.assets.columns[i]][self.common_drivers.columns[j]]
                    )
                    
                    d2_pi_da2[i, j] = (pi_a_plus - 2*pi_a + pi_a_minus) / (epsilon**2)
            
            # Prepare driver matrix
            D = np.tile(driver_values, (self.n_assets, 1))
            
            # Calculate the term in equation (16)
            sigma_p_matrix = np.diag(sigma_p)
            term = np.sum(weights * np.sum(d2_pi_da2 * D, axis=1) * sigma_p**2)
            
            # Extract implied volatility
            implied_vols[t] = np.sqrt(abs(term) * 2)  # abs to ensure positive value
        
        return implied_vols
    
    def optimize_weights(self, pi_matrix, driver_values, sigma_p):
        """
        Optimize portfolio weights to minimize conditional risk.
        
        Parameters:
        - pi_matrix: Matrix of copula partial derivatives
        - driver_values: Driver values (normalized)
        - sigma_p: Asset volatilities
        
        Returns:
        - weights: Optimized portfolio weights
        """
        def objective(weights):
            # Ensure weights sum to 1
            weights = weights / np.sum(weights)
            
            # Calculate conditional probability
            probability = self.calculate_conditional_probability(pi_matrix, driver_values, weights)
            
            # Calculate portfolio variance
            portfolio_variance = np.sum((weights * sigma_p)**2)  # Assuming independence
            
            # Objective: minimize conditional risk (a combination of probability and variance)
            return portfolio_variance - probability  # Minimize variance, maximize probability
        
        # Initial weights (equal)
        initial_weights = np.ones(self.n_assets) / self.n_assets
        
        # Constraints
        constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})  # Sum to 1
        bounds = [(0, 1) for _ in range(self.n_assets)]  # Weights between 0 and 1
        
        # Optimize
        result = minimize(objective, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
        
        # Return optimized weights
        return result.x / np.sum(result.x)  # Ensure sum to 1
    
    def compute_portfolio_metrics(self, returns, weights):
        """
        Compute standard portfolio metrics.
        
        Parameters:
        - returns: DataFrame of asset returns
        - weights: Portfolio weights
        
        Returns:
        - metrics: Dictionary of portfolio metrics
        """
        # Calculate portfolio returns
        portfolio_returns = np.sum(returns * weights, axis=1)
        
        # Calculate metrics
        sharpe_ratio = np.mean(portfolio_returns) / np.std(portfolio_returns) * np.sqrt(252)  # Annualized
        volatility = np.std(portfolio_returns) * np.sqrt(252)  # Annualized
        cumulative_return = np.prod(1 + portfolio_returns) - 1
        
        return {
            'sharpe_ratio': sharpe_ratio,
            'volatility': volatility,
            'cumulative_return': cumulative_return,
            'portfolio_returns': portfolio_returns
        }
    
    def run_strategy(self, data, rebalance_freq=20):
        """
        Run the full strategy on the given data.
        
        Parameters:
        - data: DataFrame with asset returns
        - rebalance_freq: How often to rebalance the portfolio (in days)
        
        Returns:
        - results: Dictionary with strategy results
        """
        n_periods = len(data) - self.window_size
        
        # Initialize results
        weights_history = np.zeros((n_periods, self.n_assets))
        portfolio_returns = np.zeros(n_periods)
        implied_vols = np.zeros(n_periods)
        conditional_probs = np.zeros(n_periods)
        
        # Initial equal weights
        current_weights = np.ones(self.n_assets) / self.n_assets
        
        for t in range(n_periods):
            # Current window of data
            window = data.iloc[t:t+self.window_size]
            
            # Identify common drivers
            self.identify_common_drivers(window)
            
            # Calculate correlations
            self.calculate_correlations()
            
            # Current asset and driver values (last values in the window)
            asset_values = np.array([stats.norm.cdf(window.iloc[-1][col]) for col in window.columns])
            driver_values = np.array([self.common_drivers.iloc[-1][col] for col in self.common_drivers.columns])
            
            # Calculate Pi matrix
            pi_matrix = self.calculate_pi_matrix(asset_values, driver_values)
            
            # Calculate asset volatilities
            sigma_p = np.std(window, axis=0).values
            
            # Calculate driver volatilities and drifts
            sigma_d = np.std(self.common_drivers, axis=0).values
            mu_d = np.mean(self.common_drivers.diff().dropna(), axis=0).values / self.common_drivers.std().values
            
            # Rebalance if needed
            if t % rebalance_freq == 0:
                current_weights = self.optimize_weights(pi_matrix, driver_values, sigma_p)
            
            # Store weights
            weights_history[t] = current_weights
            
            # Calculate portfolio return for the period
            if t < n_periods - 1:
                next_returns = data.iloc[t+self.window_size+1].values
                portfolio_returns[t] = np.sum(current_weights * next_returns)
            
            # Calculate conditional probability
            conditional_probs[t] = self.calculate_conditional_probability(pi_matrix, driver_values, current_weights)
            
            # Solve PDE system for implied volatility (for a single step)
            solution = self.solve_pde_system(
                asset_values, driver_values, current_weights,
                sigma_p, sigma_d, mu_d, dt=0.01, n_steps=1
            )
            
            # Extract implied volatility
            implied_vols[t] = self.calculate_implied_volatility(
                solution['pi_solution'], driver_values, current_weights, sigma_p
            )[0]
            
            # Print progress
            if t % 10 == 0:
                print(f"Processing period {t}/{n_periods}")
        
        # Calculate overall metrics
        metrics = self.compute_portfolio_metrics(data.iloc[self.window_size:self.window_size+n_periods], weights_history)
        
        return {
            'weights': weights_history,
            'returns': portfolio_returns,
            'implied_vols': implied_vols,
            'conditional_probs': conditional_probs,
            'metrics': metrics
        }


class MarketSimulator:
    """
    Class to simulate market data with known causal relationships.
    """
    
    def __init__(self, n_assets=5, n_drivers=3, n_periods=1000, seed=42):
        """
        Initialize the market simulator.
        
        Parameters:
        - n_assets: Number of assets to simulate
        - n_drivers: Number of common causal drivers
        - n_periods: Number of time periods to simulate
        - seed: Random seed for reproducibility
        """
        self.n_assets = n_assets
        self.n_drivers = n_drivers
        self.n_periods = n_periods
        self.seed = seed
        np.random.seed(seed)
        
    def simulate_data(self, volatility=0.01, driver_impact=0.7, mean_return=0.0005):
        """
        Simulate market data with common causal drivers.
        
        Parameters:
        - volatility: Base volatility of assets
        - driver_impact: How much impact drivers have on assets (0-1)
        - mean_return: Mean daily return of assets
        
        Returns:
        - data: Dictionary with simulated data
        """
        # Simulate common drivers (AR(1) processes)
        drivers = np.zeros((self.n_periods, self.n_drivers))
        
        for j in range(self.n_drivers):
            ar_coef = 0.7 + 0.2 * np.random.rand()  # Random AR coefficient between 0.7 and 0.9
            drivers[0, j] = np.random.randn() * volatility
            
            for t in range(1, self.n_periods):
                drivers[t, j] = ar_coef * drivers[t-1, j] + np.random.randn() * volatility
        
        # Create impact matrix (how each driver affects each asset)
        impact_matrix = np.random.rand(self.n_assets, self.n_drivers) * driver_impact
        
        # Ensure row sums are normalized
        impact_matrix = impact_matrix / np.sum(impact_matrix, axis=1, keepdims=True)
        
        # Simulate asset returns
        asset_returns = np.zeros((self.n_periods, self.n_assets))
        
        for t in range(self.n_periods):
            # Common driver component
            common_component = np.dot(impact_matrix, drivers[t])
            
            # Idiosyncratic component
            idiosyncratic = np.random.randn(self.n_assets) * volatility * (1 - driver_impact)
            
            # Combine with mean return
            asset_returns[t] = mean_return + common_component + idiosyncratic
        
        # Convert to DataFrames
        driver_df = pd.DataFrame(drivers, columns=[f'Driver_{i+1}' for i in range(self.n_drivers)])
        asset_df = pd.DataFrame(asset_returns, columns=[f'Asset_{i+1}' for i in range(self.n_assets)])
        
        # Create prices from returns
        asset_prices = 100 * np.cumprod(1 + asset_returns, axis=0)
        price_df = pd.DataFrame(asset_prices, columns=[f'Asset_{i+1}' for i in range(self.n_assets)])
        
        return {
            'drivers': driver_df,
            'returns': asset_df,
            'prices': price_df,
            'impact_matrix': impact_matrix
        }

class BenchmarkStrategy:
    """
    Benchmark portfolio strategies for comparison.
    """
    
    @staticmethod
    def equal_weight(returns, window_size=60):
        """
        Equal weight strategy.
        
        Parameters:
        - returns: DataFrame of asset returns
        - window_size: Look-back window (not used but included for consistency)
        
        Returns:
        - results: Dictionary with strategy results
        """
        n_assets = returns.shape[1]
        n_periods = len(returns) - window_size
        
        # Equal weights
        weights = np.ones(n_assets) / n_assets
        
        # Calculate portfolio returns
        portfolio_returns = np.zeros(n_periods)
        
        for t in range(n_periods):
            if t < n_periods - 1:
                next_returns = returns.iloc[t+window_size+1].values
                portfolio_returns[t] = np.sum(weights * next_returns)
        
        # Calculate metrics
        sharpe_ratio = np.mean(portfolio_returns) / np.std(portfolio_returns) * np.sqrt(252)  # Annualized
        volatility = np.std(portfolio_returns) * np.sqrt(252)  # Annualized
        cumulative_return = np.prod(1 + portfolio_returns) - 1
        
        return {
            'weights': np.tile(weights, (n_periods, 1)),
            'returns': portfolio_returns,
            'metrics': {
                'sharpe_ratio': sharpe_ratio,
                'volatility': volatility,
                'cumulative_return': cumulative_return,
                'portfolio_returns': portfolio_returns
            }
        }
    
    @staticmethod
    def min_variance(returns, window_size=60, rebalance_freq=20):
        """
        Minimum variance strategy.
        
        Parameters:
        - returns: DataFrame of asset returns
        - window_size: Look-back window
        - rebalance_freq: How often to rebalance the portfolio (in days)
        
        Returns:
        - results: Dictionary with strategy results
        """
        n_assets = returns.shape[1]
        n_periods = len(returns) - window_size
        
        # Initialize weights
        weights_history = np.zeros((n_periods, n_assets))
        portfolio_returns = np.zeros(n_periods)
        
        # Initial equal weights
        current_weights = np.ones(n_assets) / n_assets
        
        for t in range(n_periods):
            # Rebalance if needed
            if t % rebalance_freq == 0:
                # Current window of data
                window = returns.iloc[t:t+window_size]
                
                # Calculate covariance matrix
                cov_matrix = window.cov().values
                
                # Ensure positive definite
                cov_matrix = cov_matrix + 1e-8 * np.eye(n_assets)
                
                # Minimize variance
                def objective(w):
                    return np.dot(w, np.dot(cov_matrix, w))
                
                constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
                bounds = [(0, 1) for _ in range(n_assets)]
                
                result = minimize(objective, current_weights, method='SLSQP', bounds=bounds, constraints=constraints)
                current_weights = result.x
            
            # Store weights
            weights_history[t] = current_weights
            
            # Calculate portfolio return for the period
            if t < n_periods - 1:
                next_returns = returns.iloc[t+window_size+1].values
                portfolio_returns[t] = np.sum(current_weights * next_returns)
        
        # Calculate metrics
        sharpe_ratio = np.mean(portfolio_returns) / np.std(portfolio_returns) * np.sqrt(252)  # Annualized
        volatility = np.std(portfolio_returns) * np.sqrt(252)  # Annualized
        cumulative_return = np.prod(1 + portfolio_returns) - 1
        
        return {
            'weights': weights_history,
            'returns': portfolio_returns,
            'metrics': {
                'sharpe_ratio': sharpe_ratio,
                'volatility': volatility,
                'cumulative_return': cumulative_return,
                'portfolio_returns': portfolio_returns
            }
        }
    
    @staticmethod
    def momentum(returns, window_size=60, rebalance_freq=20, lookback=20):
        """
        Momentum strategy.
        
        Parameters:
        - returns: DataFrame of asset returns
        - window_size: Look-back window
        - rebalance_freq: How often to rebalance the portfolio (in days)
        - lookback: Lookback period for momentum calculation
        
        Returns:
        - results: Dictionary with strategy results
        """
        n_assets = returns.shape[1]
        n_periods = len(returns) - window_size
        
        # Initialize weights
        weights_history = np.zeros((n_periods, n_assets))
        portfolio_returns = np.zeros(n_periods)
        
        # Initial equal weights
        current_weights = np.ones(n_assets) / n_assets
        
        for t in range(n_periods):
            # Rebalance if needed
            if t % rebalance_freq == 0:
                # Current window of data
                window = returns.iloc[t:t+window_size]
                
                # Calculate momentum (cumulative return over lookback period)
                momentum = np.prod(1 + window.iloc[-lookback:].values, axis=0) - 1
                
                # Assign weights proportional to momentum (only positive momentum)
                positive_momentum = np.maximum(momentum, 0)
                
                if np.sum(positive_momentum) > 0:
                    current_weights = positive_momentum / np.sum(positive_momentum)
                else:
                    # If no positive momentum, equal weight
                    current_weights = np.ones(n_assets) / n_assets
            
            # Store weights
            weights_history[t] = current_weights
            
            # Calculate portfolio return for the period
            if t < n_periods - 1:
                next_returns = returns.iloc[t+window_size+1].values
                portfolio_returns[t] = np.sum(current_weights * next_returns)
        
        # Calculate metrics
        sharpe_ratio = np.mean(portfolio_returns) / np.std(portfolio_returns) * np.sqrt(252)  # Annualized
        volatility = np.std(portfolio_returns) * np.sqrt(252)  # Annualized
        cumulative_return = np.prod(1 + portfolio_returns) - 1
        
        return {
            'weights': weights_history,
            'returns': portfolio_returns,
            'metrics': {
                'sharpe_ratio': sharpe_ratio,
                'volatility': volatility,
                'cumulative_return': cumulative_return,
                'portfolio_returns': portfolio_returns
            }
        }

def run_experiment():
    """
    Run a full experiment comparing the Common Causal PDE strategy with benchmarks.
    """
    # Simulate market data
    simulator = MarketSimulator(n_assets=5, n_drivers=3, n_periods=1000, seed=42)
    market_data = simulator.simulate_data(volatility=0.01, driver_impact=0.7, mean_return=0.0005)
    
    returns = market_data['returns']
    
    # Set up strategies
    window_size = 60
    rebalance_freq = 20
    
    # Initialize the Common Causal PDE strategy
    causal_pde = CommonCausalPDEStrategy(n_assets=5, n_drivers=3, window_size=window_size)
    
    # Run strategies
    print("Running Common Causal PDE strategy...")
    causal_results = causal_pde.run_strategy(returns, rebalance_freq=rebalance_freq)
    
    print("Running Equal Weight strategy...")
    equal_results = BenchmarkStrategy.equal_weight(returns, window_size=window_size)
    
    print("Running Min Variance strategy...")
    min_var_results = BenchmarkStrategy.min_variance(returns, window_size=window_size, rebalance_freq=rebalance_freq)
    
    print("Running Momentum strategy...")
    momentum_results = BenchmarkStrategy.momentum(returns, window_size=window_size, rebalance_freq=rebalance_freq)
    
    # Compare results
    strategies = {
        'Common Causal PDE': causal_results,
        'Equal Weight': equal_results,
        'Min Variance': min_var_results,
        'Momentum': momentum_results
    }
    
    # Print metrics
    print("\nStrategy Performance Metrics:")
    print("-" * 80)
    print(f"{'Strategy':<20} {'Sharpe Ratio':<15} {'Volatility':<15} {'Cumulative Return':<20}")
    print("-" * 80)
    
    for name, results in strategies.items():
        metrics = results['metrics']
        print(f"{name:<20} {metrics['sharpe_ratio']:.4f}        {metrics['volatility']:.4f}        {metrics['cumulative_return']:.4f}")
    
    # Plot cumulative returns
    plot_cumulative_returns(strategies)
    
    # Plot implied volatilities and conditional probabilities
    plot_risk_metrics(causal_results)
    
    # Plot weight allocations
    plot_weight_allocations(strategies)
    
    # Analyze PDE deviations
    plot_pde_deviations(causal_results, market_data)
    
    return {
        'market_data': market_data,
        'strategies': strategies
    }

def plot_cumulative_returns(strategies):
    """
    Plot cumulative returns for all strategies.
    """
    plt.figure(figsize=(12, 6))
    
    for name, results in strategies.items():
        returns = results['returns']
        cum_returns = np.cumprod(1 + returns) - 1
        plt.plot(cum_returns, label=name)
    
    plt.title('Cumulative Returns')
    plt.xlabel('Time')
    plt.ylabel('Cumulative Return')
    plt.legend()
    plt.grid(True)
    plt.savefig('cumulative_returns.png')
    plt.close()

def plot_risk_metrics(causal_results):
    """
    Plot implied volatilities and conditional probabilities.
    """
    plt.figure(figsize=(12, 10))
    
    # Plot implied volatilities
    plt.subplot(2, 1, 1)
    plt.plot(causal_results['implied_vols'], label='Implied Volatility')
    plt.title('Implied Portfolio Volatility')
    plt.xlabel('Time')
    plt.ylabel('Volatility')
    plt.grid(True)
    
    # Plot conditional probabilities
    plt.subplot(2, 1, 2)
    plt.plot(causal_results['conditional_probs'], label='Conditional Probability')
    plt.title('Conditional Probability of Portfolio Given Common Drivers')
    plt.xlabel('Time')
    plt.ylabel('Probability')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('risk_metrics.png')
    plt.close()

def plot_weight_allocations(strategies):
    """
    Plot weight allocations for all strategies.
    """
    n_strategies = len(strategies)
    n_assets = strategies[list(strategies.keys())[0]]['weights'].shape[1]
    
    plt.figure(figsize=(15, 10))
    
    for i, (name, results) in enumerate(strategies.items()):
        plt.subplot(n_strategies, 1, i+1)
        
        weights = results['weights']
        for j in range(n_assets):
            plt.plot(weights[:, j], label=f'Asset {j+1}')
        
        plt.title(f'{name} - Weight Allocation')
        plt.xlabel('Time')
        plt.ylabel('Weight')
        plt.legend(loc='upper right')
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('weight_allocations.png')
    plt.close()

def plot_pde_deviations(causal_results, market_data):
    """
    Plot PDE deviations for analysis.
    """
    # Calculate PDE deviations as the difference between actual and predicted returns
    actual_returns = causal_results['returns']
    
    # Create a simple model to predict returns based on conditional probabilities
    cond_probs = causal_results['conditional_probs']
    implied_vols = causal_results['implied_vols']
    
    # Use a linear model to predict returns
    X = np.column_stack([cond_probs, implied_vols])
    X = sm.add_constant(X)
    y = actual_returns
    
    model = sm.OLS(y, X)
    results = model.fit()
    
    predicted_returns = results.predict()
    deviations = actual_returns - predicted_returns
    
    # Plot deviations
    plt.figure(figsize=(12, 6))
    
    plt.subplot(2, 1, 1)
    plt.plot(deviations, label='PDE Deviations')
    plt.title('PDE Deviations')
    plt.xlabel('Time')
    plt.ylabel('Deviation')
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.hist(deviations, bins=50, alpha=0.7)
    plt.title('Distribution of PDE Deviations')
    plt.xlabel('Deviation')
    plt.ylabel('Frequency')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('pde_deviations.png')
    plt.close()

if __name__ == "__main__":
    results = run_experiment()


