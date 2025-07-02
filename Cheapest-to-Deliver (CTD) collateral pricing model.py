import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import minimize
import pandas as pd
from datetime import datetime, timedelta

# Set random seed for reproducibility
np.random.seed(42)

class CollateralChoiceOption:
    """Class for valuing and analyzing collateral choice options"""
    
    def __init__(self, currencies, initial_spreads, mean_reversion_speeds, volatilities, 
                 correlation_matrix, maturity, time_steps=200):
        """
        Initialize the collateral choice option model
        
        Parameters:
        -----------
        currencies : list of str
            Currency names
        initial_spreads : array-like
            Initial values of spreads vs base currency
        mean_reversion_speeds : array-like
            Speed of mean reversion parameters (kappa)
        volatilities : array-like
            Volatility parameters (xi)
        correlation_matrix : array-like
            Correlation matrix between spreads
        maturity : float
            Option maturity in years
        time_steps : int
            Number of time steps for discretization
        """
        self.currencies = currencies
        self.n_currencies = len(currencies) - 1  # Base currency + spread currencies
        self.q0 = np.array(initial_spreads)
        self.kappa = np.array(mean_reversion_speeds)
        self.xi = np.array(volatilities)
        self.correlation_matrix = np.array(correlation_matrix)
        self.T = maturity
        self.time_steps = time_steps
        self.times = np.linspace(0, maturity, time_steps)
        
        # Set long term mean to initial values for simplicity
        self.theta = self.q0.copy()
        
        # Compute model components
        self.means, self.variances = self._compute_means_variances()
        self.correlations = self._compute_correlations()
        self.gamma, self.C_var, self.A_means, self.A_vars = self._compute_common_factor_parameters()
        self.E_M, self.Var_M = self._compute_maximum_moments()
        self.probs = self._compute_maximum_probabilities()
        
    def _compute_means_variances(self):
        """Compute means and variances for Hull-White processes"""
        means = np.zeros((self.time_steps, self.n_currencies))
        variances = np.zeros((self.time_steps, self.n_currencies))
        
        for i, t in enumerate(self.times):
            for j in range(self.n_currencies):
                # Mean from equation (11)
                means[i, j] = self.q0[j] * np.exp(-self.kappa[j] * t) + self.theta[j] * (1 - np.exp(-self.kappa[j] * t))
                
                # Variance from equation (12)
                variances[i, j] = (self.xi[j]**2 / (2 * self.kappa[j])) * (1 - np.exp(-2 * self.kappa[j] * t))
        
        return means, variances
    
    def _compute_correlations(self):
        """Compute correlations between Hull-White processes"""
        correlations = np.zeros((self.time_steps, self.n_currencies, self.n_currencies))
        
        for i, t in enumerate(self.times):
            for j in range(self.n_currencies):
                for k in range(self.n_currencies):
                    if j == k:
                        correlations[i, j, k] = 1.0
                    else:
                        # Correlation formula from equation (13)
                        rho = self.correlation_matrix[j, k]
                        kj, kk = self.kappa[j], self.kappa[k]
                        
                        # Compute correlation at time t
                        num = 2 * rho * np.sqrt(kj * kk) * (1 - np.exp(-(kj + kk) * t))
                        denom = np.sqrt((1 - np.exp(-2 * kj * t)) * (1 - np.exp(-2 * kk * t)))
                        
                        # Handle potential numerical issues
                        if denom > 0:
                            correlations[i, j, k] = num / denom
                        else:
                            correlations[i, j, k] = rho
        
        return correlations
    
    def _compute_common_factor_parameters(self):
        """Compute common factor parameters"""
        gamma = np.zeros(self.time_steps)
        C_var = np.zeros(self.time_steps)
        A_means = np.zeros_like(self.means)
        A_vars = np.zeros_like(self.variances)
        
        for i in range(self.time_steps):
            # Skip if all variances are effectively zero
            if np.all(self.variances[i] < 1e-10):
                gamma[i] = 0.0
                C_var[i] = 0.0
                A_means[i] = self.means[i]
                A_vars[i] = self.variances[i]
                continue
                
            # For the case of exactly two currencies (N=1), gamma can be set analytically
            if self.n_currencies == 1:
                # Only one spread, so no correlation to consider
                gamma[i] = 0.0
            elif self.n_currencies == 2:
                # Two spreads, use equation (35)
                rho = self.correlations[i, 0, 1]
                
                # Handle numerical issues when computing the ratio
                min_var = np.min(self.variances[i])
                max_var = np.max(self.variances[i])
                
                if min_var < 1e-10:
                    sigma_ratio = 0
                else:
                    sigma_ratio = max_var / min_var
                
                gamma[i] = rho * sigma_ratio
                
                # Ensure gamma is within bounds
                gamma[i] = min(gamma[i], 0.9999)
            else:
                # For more than two spreads, optimize gamma
                gamma[i] = self._optimize_correlation_parameter(
                    self.variances[i], self.correlations[i], self.n_currencies)
            
            # Minimum variance
            sigma_min_squared = np.min(self.variances[i])
            
            # Variance of the common factor (equation 23)
            C_var[i] = sigma_min_squared * gamma[i]
            
            # Means and variances of individual factors (equation 24)
            A_means[i] = self.means[i]
            A_vars[i] = self.variances[i] - C_var[i]
            
            # Ensure A_vars is non-negative (numerical stability)
            A_vars[i] = np.maximum(A_vars[i], 1e-10)
        
        return gamma, C_var, A_means, A_vars
    
    def _optimize_correlation_parameter(self, variances, correlation_matrix, n_spreads):
        """Optimize the correlation parameter gamma"""
        # Minimum variance
        sigma_min_squared = np.min(variances)
        
        # Check if all variances are effectively zero
        if sigma_min_squared < 1e-10:
            return 0.0
        
        # Function to minimize
        def objective(gamma):
            gamma = abs(gamma)  # Ensure gamma is positive
            if gamma >= 1.0:
                return 1e10  # Large penalty for invalid gamma
            
            total_error = 0
            for i in range(n_spreads):
                for j in range(i+1, n_spreads):
                    # Compute approximated correlation from equation (26)
                    approx_corr = (sigma_min_squared * gamma) / (np.sqrt(variances[i] * variances[j]))
                    
                    # Compute error
                    error = (approx_corr - correlation_matrix[i, j])**2
                    total_error += error
            
            return np.sqrt(total_error)
        
        # Initial guess for gamma
        initial_gamma = 0.5
        
        # Minimize the objective function
        result = minimize(objective, initial_gamma, bounds=[(0, 0.9999)])
        
        return abs(result.x[0])
    
    def _compute_maximum_moments(self, delta=5e-5):
        """Compute moments of the maximum distribution"""
        E_M = np.zeros(self.time_steps)
        Var_M = np.zeros(self.time_steps)
        
        for i in range(self.time_steps):
            # Skip time 0
            if i == 0 and np.all(self.A_means[i] == 0) and np.all(self.A_vars[i] == 0):
                continue
                
            # Compute the upper limit L for integration
            max_mean = np.max(self.A_means[i])
            max_std = np.sqrt(np.max(self.A_vars[i] + self.C_var[i]))
            
            # Ensure max_std is not too small to avoid numerical issues
            max_std = max(max_std, 1e-6)
            
            # Calculate L with a buffer
            L = max(max_mean + 6 * max_std, 0.1)
            
            # Create grid for numerical integration with safety check
            if delta <= 0 or L <= 0 or L/delta > 1e6:
                delta = max(L / 1000, 1e-5)
            
            x_grid = np.arange(0, L, delta)
            
            # Make sure grid is not empty
            if len(x_grid) == 0:
                x_grid = np.array([0.0, delta])
            
            # Density of the common factor (equation 57)
            C_std = np.sqrt(self.C_var[i]) if self.C_var[i] > 0 else 1e-6
            f_C = stats.norm.pdf(x_grid, 0, C_std)
            
            # CDF of the maximum of individual factors (equation 58)
            F_max_A = np.ones(len(x_grid))
            for j in range(self.n_currencies):
                A_std = np.sqrt(self.A_vars[i, j]) if self.A_vars[i, j] > 0 else 1e-6
                F_max_A *= stats.norm.cdf(x_grid, self.A_means[i, j], A_std)
            
            # Add the zero component
            F_max_A *= stats.norm.cdf(x_grid, 0, 1e-10)
            
            # Compute convolution
            conv = np.zeros_like(x_grid)
            for j in range(len(x_grid)):
                x = x_grid[j]
                shift_grid = x - x_grid[x_grid <= x]
                if len(shift_grid) > 0:
                    conv[j] = np.sum(f_C[:len(shift_grid)] * F_max_A[:len(shift_grid)][::-1]) * delta
            
            # Compute expected value of the maximum (equation 42)
            E_M[i] = np.sum((1 - conv) * delta)
            
            # Compute second moment and variance (equation 59)
            second_moment = np.sum(2 * x_grid * (1 - conv) * delta)
            Var_M[i] = second_moment - E_M[i]**2
            
            # Ensure non-negative variance
            Var_M[i] = max(Var_M[i], 0)
        
        return E_M, Var_M
    
    def _compute_maximum_probabilities(self, delta=5e-5):
        """Compute probabilities of each spread being the maximum"""
        probs = np.zeros((self.time_steps, self.n_currencies))
        
        for i in range(self.time_steps):
            # Skip time 0
            if i == 0 and np.all(self.A_means[i] == 0) and np.all(self.A_vars[i] == 0):
                continue
                
            # Compute the range for integration
            max_abs_mean = max(np.max(np.abs(self.A_means[i])), 0.1)
            max_std = np.sqrt(np.max(self.A_vars[i] + self.C_var[i]))
            max_std = max(max_std, 1e-6)
            
            # Define integration limits
            L = max_abs_mean + 6 * max_std
            
            # Adjust delta if needed
            if delta <= 0 or 2*L/delta > 1e6:
                delta = max(2*L / 1000, 1e-5)
            
            # Create grid for numerical integration
            x_grid = np.linspace(-L, L, int(2*L/delta) + 1)
            
            # Ensure the grid is not empty
            if len(x_grid) < 2:
                x_grid = np.array([-L, 0, L])
            
            for j in range(self.n_currencies):
                # Density of A_j (equation 73)
                A_mean = self.A_means[i, j]
                A_std = np.sqrt(self.A_vars[i, j]) if self.A_vars[i, j] > 0 else 1e-6
                f_A_j = stats.norm.pdf(x_grid, A_mean, A_std)
                
                # CDF of the common factor C
                C_std = np.sqrt(self.C_var[i]) if self.C_var[i] > 0 else 1e-6
                F_C = stats.norm.cdf(x_grid, 0, C_std)
                
                # Product of CDFs of other individual factors
                F_A_prod = np.ones(len(x_grid))
                for k in range(self.n_currencies):
                    if k != j:
                        A_k_std = np.sqrt(self.A_vars[i, k]) if self.A_vars[i, k] > 0 else 1e-6
                        F_A_prod *= stats.norm.cdf(x_grid, self.A_means[i, k], A_k_std)
                
                # Compute the integrand
                integrand = f_A_j * F_C * F_A_prod
                
                # Compute the probability (equation 73)
                probs[i, j] = np.sum(integrand) * (x_grid[1] - x_grid[0])
                
                # Ensure probability is between 0 and 1
                probs[i, j] = max(0, min(1, probs[i, j]))
            
            # Normalize probabilities to sum to 1
            if np.sum(probs[i]) > 0:
                probs[i] /= np.sum(probs[i])
        
        return probs
    
    def value_option(self, method='diffusion'):
        """
        Value the collateral choice option
        
        Parameters:
        -----------
        method : str
            Valuation method: 'first_order', 'diffusion', or 'mean_reverting'
        
        Returns:
        --------
        value : float
            Option value as a discount factor
        """
        # First-order estimator
        CF1 = self._first_order_estimator()
        
        if method == 'first_order':
            return CF1
        elif method == 'diffusion':
            return self._diffusion_based_estimator(CF1)
        elif method == 'mean_reverting':
            return self._mean_reverting_estimator(CF1)
        else:
            raise ValueError("Method must be 'first_order', 'diffusion', or 'mean_reverting'")
    
    def _first_order_estimator(self):
        """Compute the first-order common factor estimator"""
        # Compute time steps
        dt = np.diff(self.times)
        
        # Compute integral approximation using trapezoidal rule
        integral = np.sum(0.5 * (self.E_M[1:] + self.E_M[:-1]) * dt)
        
        # First-order estimator (equation 45)
        return np.exp(-integral)
    
    def _diffusion_based_estimator(self, CF1):
        """Compute the diffusion-based second-order estimator"""
        # Compute time steps
        dt = np.diff(self.times)
        
        # Compute the diffusion-based variance estimator
        T = self.times[-1]
        Phi = 0.0
        
        # Simplified approximation of the double integral in equation (54)
        for i in range(1, len(self.times)-1):
            for j in range(1, i+1):
                # Time and variance at points i and j
                t_i = self.times[i]
                t_j = self.times[j]
                v_j = self.Var_M[j]
                
                # Time step
                dt_i = self.times[i+1] - self.times[i]
                dt_j = self.times[j] - self.times[j-1]
                
                # Contribution to the double integral
                Phi += (T - t_i) * v_j * dt_i * dt_j
        
        # Ensure Phi is non-negative and not too large
        Phi = min(max(Phi, 0), 5.0)
        
        # Second-order diffusion-based estimator (equation 68)
        return CF1 * (1 + 0.5 * Phi)
    
    def _mean_reverting_estimator(self, CF1):
        """Compute the mean-reverting second-order estimator"""
        # Compute time steps
        dt = np.diff(self.times)
        
        # Compute weighted mean reversion parameter (equation 76)
        kappa_weighted = np.zeros(len(self.times))
        for i in range(len(self.times)):
            kappa_weighted[i] = np.sum(self.probs[i] * self.kappa)
        
        # Compute the mean-reverting variance estimator
        chi = 0.0
        
        # Simplified approximation of the formula in equation (77)
        for i in range(1, len(self.times)-1):
            # Compute exponential terms
            exp_term = np.exp(-kappa_weighted[i] * (self.times[i] - self.times[1]))
            
            inner_sum = 0.0
            for j in range(1, i+1):
                exp_inner = np.exp(kappa_weighted[j] * (self.times[j] - self.times[1]))
                inner_sum += exp_inner * self.Var_M[j] * (self.times[j] - self.times[j-1])
            
            chi += 2 * exp_term * inner_sum * (self.times[i+1] - self.times[i])
        
        # Ensure chi is non-negative and not too large
        chi = min(max(chi, 0), 5.0)
        
        # Second-order mean-reverting estimator (equation 81)
        return CF1 * (1 + 0.5 * chi)
    
    def monte_carlo_value(self, n_paths=10000):
        """
        Compute the option value using Monte Carlo simulation
        
        Parameters:
        -----------
        n_paths : int
            Number of Monte Carlo paths
        
        Returns:
        --------
        value : float
            Option value as a discount factor
        """
        # Simulate paths
        simulated_paths = self._simulate_paths(n_paths)
        
        # Compute time steps
        dt = np.diff(self.times)
        
        # Compute the maximum for each path and time
        max_values = np.maximum(0, np.max(simulated_paths, axis=2))
        
        # Compute the integral for each path using trapezoidal rule
        integral = np.zeros(n_paths)
        for i in range(n_paths):
            # Skip t=0
            integral[i] = np.sum(0.5 * (max_values[i, 1:] + max_values[i, :-1]) * dt)
        
        # Compute the CTD discount factor
        return np.mean(np.exp(-integral))
    
    def _simulate_paths(self, n_paths):
        """Simulate paths of the Hull-White processes"""
        n_times = len(self.times)
        
        # Initialize array to store simulated paths
        simulated_paths = np.zeros((n_paths, n_times, self.n_currencies))
        simulated_paths[:, 0, :] = self.q0
        
        # Ensure correlation matrix is positive definite
        min_eig = np.min(np.real(np.linalg.eigvals(self.correlation_matrix)))
        if min_eig < 0:
            correlation_matrix = self.correlation_matrix - 1.1 * min_eig * np.eye(self.n_currencies)
        else:
            correlation_matrix = self.correlation_matrix
        
        # Cholesky decomposition for correlated random numbers
        L = np.linalg.cholesky(correlation_matrix)
        
        # Simulate paths
        for i in range(1, n_times):
            dt = self.times[i] - self.times[i-1]
            
            # Generate correlated normal random variables
            dW = np.random.normal(0, np.sqrt(dt), (n_paths, self.n_currencies))
            dW = np.dot(dW, L.T)
            
            for j in range(self.n_currencies):
                # Hull-White dynamics
                drift = self.kappa[j] * (self.theta[j] - simulated_paths[:, i-1, j]) * dt
                diffusion = self.xi[j] * dW[:, j]
                
                simulated_paths[:, i, j] = simulated_paths[:, i-1, j] + drift + diffusion
        
        return simulated_paths
    
    def compute_implied_basis(self, discount_factor):
        """
        Compute the implied basis spread from a discount factor
        
        Parameters:
        -----------
        discount_factor : float
            Discount factor from the option valuation
        
        Returns:
        --------
        basis : float
            Implied basis spread in basis points
        """
        # Convert discount factor to continuously compounded rate
        rate = -np.log(discount_factor) / self.T
        
        # Convert to basis points
        return rate * 10000
    
    def plot_option_value_by_correlation(self, rho_values=None):
        """Plot option value as a function of correlation"""
        if rho_values is None:
            rho_values = np.linspace(0, 0.7, 8)
        
        mc_values = []
        fo_values = []
        diff_values = []
        mr_values = []
        
        original_corr = self.correlation_matrix.copy()
        
        for rho in rho_values:
            # Set correlation
            if self.n_currencies == 2:
                self.correlation_matrix = np.array([[1.0, rho], [rho, 1.0]])
            else:
                # For more currencies, set all off-diagonal elements to rho
                self.correlation_matrix = np.eye(self.n_currencies)
                for i in range(self.n_currencies):
                    for j in range(i+1, self.n_currencies):
                        self.correlation_matrix[i, j] = self.correlation_matrix[j, i] = rho
            
            # Recompute model with new correlation
            self.correlations = self._compute_correlations()
            self.gamma, self.C_var, self.A_means, self.A_vars = self._compute_common_factor_parameters()
            self.E_M, self.Var_M = self._compute_maximum_moments()
            self.probs = self._compute_maximum_probabilities()
            
            # Value option with different methods
            mc_values.append(self.monte_carlo_value(5000))
            fo_values.append(self.value_option('first_order'))
            diff_values.append(self.value_option('diffusion'))
            mr_values.append(self.value_option('mean_reverting'))
        
        # Restore original correlation
        self.correlation_matrix = original_corr
        self.correlations = self._compute_correlations()
        self.gamma, self.C_var, self.A_means, self.A_vars = self._compute_common_factor_parameters()
        self.E_M, self.Var_M = self._compute_maximum_moments()
        self.probs = self._compute_maximum_probabilities()
        
        # Plot results
        plt.figure(figsize=(10, 6))
        plt.plot(rho_values, mc_values, 'k-', label='Monte Carlo')
        plt.plot(rho_values, fo_values, 'b--', label='First-order')
        plt.plot(rho_values, diff_values, 'r-.', label='Diffusion')
        plt.plot(rho_values, mr_values, 'g:', label='Mean-Reverting')
        plt.title('Effect of Correlation on CTD Option Value')
        plt.xlabel('Correlation')
        plt.ylabel('Option Value (Discount Factor)')
        plt.legend()
        plt.grid(True)
        
        # Convert to implied basis points
        mc_basis = [-np.log(v) / self.T * 10000 for v in mc_values]
        fo_basis = [-np.log(v) / self.T * 10000 for v in fo_values]
        diff_basis = [-np.log(v) / self.T * 10000 for v in diff_values]
        mr_basis = [-np.log(v) / self.T * 10000 for v in mr_values]
        
        # Plot basis points
        plt.figure(figsize=(10, 6))
        plt.plot(rho_values, mc_basis, 'k-', label='Monte Carlo')
        plt.plot(rho_values, fo_basis, 'b--', label='First-order')
        plt.plot(rho_values, diff_basis, 'r-.', label='Diffusion')
        plt.plot(rho_values, mr_basis, 'g:', label='Mean-Reverting')
        plt.title('Effect of Correlation on Implied Basis Spread')
        plt.xlabel('Correlation')
        plt.ylabel('Implied Basis (bps)')
        plt.legend()
        plt.grid(True)
        
        return {
            'rho_values': rho_values,
            'mc_values': mc_values,
            'fo_values': fo_values,
            'diff_values': diff_values,
            'mr_values': mr_values
        }
    
    def plot_option_value_by_volatility(self, vol_scales=None):
        """Plot option value as a function of volatility"""
        if vol_scales is None:
            vol_scales = np.linspace(0.5, 3.0, 6)
        
        mc_values = []
        fo_values = []
        diff_values = []
        mr_values = []
        
        original_xi = self.xi.copy()
        
        for scale in vol_scales:
            # Scale volatility
            self.xi = original_xi * scale
            
            # Recompute model with new volatility
            self.means, self.variances = self._compute_means_variances()
            self.correlations = self._compute_correlations()
            self.gamma, self.C_var, self.A_means, self.A_vars = self._compute_common_factor_parameters()
            self.E_M, self.Var_M = self._compute_maximum_moments()
            self.probs = self._compute_maximum_probabilities()
            
            # Value option with different methods
            mc_values.append(self.monte_carlo_value(5000))
            fo_values.append(self.value_option('first_order'))
            diff_values.append(self.value_option('diffusion'))
            mr_values.append(self.value_option('mean_reverting'))
        
        # Restore original volatility
        self.xi = original_xi
        self.means, self.variances = self._compute_means_variances()
        self.correlations = self._compute_correlations()
        self.gamma, self.C_var, self.A_means, self.A_vars = self._compute_common_factor_parameters()
        self.E_M, self.Var_M = self._compute_maximum_moments()
        self.probs = self._compute_maximum_probabilities()
        
        # Plot results
        plt.figure(figsize=(10, 6))
        plt.plot(vol_scales, mc_values, 'k-', label='Monte Carlo')
        plt.plot(vol_scales, fo_values, 'b--', label='First-order')
        plt.plot(vol_scales, diff_values, 'r-.', label='Diffusion')
        plt.plot(vol_scales, mr_values, 'g:', label='Mean-Reverting')
        plt.title('Effect of Volatility on CTD Option Value')
        plt.xlabel('Volatility Scale Factor')
        plt.ylabel('Option Value (Discount Factor)')
        plt.legend()
        plt.grid(True)
        
        # Convert to implied basis points
        mc_basis = [-np.log(v) / self.T * 10000 for v in mc_values]
        fo_basis = [-np.log(v) / self.T * 10000 for v in fo_values]
        diff_basis = [-np.log(v) / self.T * 10000 for v in diff_values]
        mr_basis = [-np.log(v) / self.T * 10000 for v in mr_values]
        
        # Plot basis points
        plt.figure(figsize=(10, 6))
        plt.plot(vol_scales, mc_basis, 'k-', label='Monte Carlo')
        plt.plot(vol_scales, fo_basis, 'b--', label='First-order')
        plt.plot(vol_scales, diff_basis, 'r-.', label='Diffusion')
        plt.plot(vol_scales, mr_basis, 'g:', label='Mean-Reverting')
        plt.title('Effect of Volatility on Implied Basis Spread')
        plt.xlabel('Volatility Scale Factor')
        plt.ylabel('Implied Basis (bps)')
        plt.legend()
        plt.grid(True)
        
        return {
            'vol_scales': vol_scales,
            'mc_values': mc_values,
            'fo_values': fo_values,
            'diff_values': diff_values,
            'mr_values': mr_values
        }
    
    def analyze_term_structure(self, maturities=None):
        """Analyze the term structure of option values"""
        if maturities is None:
            maturities = np.linspace(1, 20, 8)
        
        mc_values = []
        fo_values = []
        diff_values = []
        mr_values = []
        
        original_T = self.T
        original_times = self.times
        
        for T in maturities:
            # Set maturity
            self.T = T
            self.times = np.linspace(0, T, self.time_steps)
            
            # Recompute model with new maturity
            self.means, self.variances = self._compute_means_variances()
            self.correlations = self._compute_correlations()
            self.gamma, self.C_var, self.A_means, self.A_vars = self._compute_common_factor_parameters()
            self.E_M, self.Var_M = self._compute_maximum_moments()
            self.probs = self._compute_maximum_probabilities()
            
            # Value option with different methods
            mc_values.append(self.monte_carlo_value(5000))
            fo_values.append(self.value_option('first_order'))
            diff_values.append(self.value_option('diffusion'))
            mr_values.append(self.value_option('mean_reverting'))
        
        # Restore original maturity
        self.T = original_T
        self.times = original_times
        self.means, self.variances = self._compute_means_variances()
        self.correlations = self._compute_correlations()
        self.gamma, self.C_var, self.A_means, self.A_vars = self._compute_common_factor_parameters()
        self.E_M, self.Var_M = self._compute_maximum_moments()
        self.probs = self._compute_maximum_probabilities()
        
        # Plot results
        plt.figure(figsize=(10, 6))
        plt.plot(maturities, mc_values, 'k-', label='Monte Carlo')
        plt.plot(maturities, fo_values, 'b--', label='First-order')
        plt.plot(maturities, diff_values, 'r-.', label='Diffusion')
        plt.plot(maturities, mr_values, 'g:', label='Mean-Reverting')
        plt.title('Term Structure of CTD Option Values')
        plt.xlabel('Maturity (years)')
        plt.ylabel('Option Value (Discount Factor)')
        plt.legend()
        plt.grid(True)
        
        # Convert to implied basis points
        mc_basis = [-np.log(v) / T * 10000 for v, T in zip(mc_values, maturities)]
        fo_basis = [-np.log(v) / T * 10000 for v, T in zip(fo_values, maturities)]
        diff_basis = [-np.log(v) / T * 10000 for v, T in zip(diff_values, maturities)]
        mr_basis = [-np.log(v) / T * 10000 for v, T in zip(mr_values, maturities)]
        
        # Plot basis points
        plt.figure(figsize=(10, 6))
        plt.plot(maturities, mc_basis, 'k-', label='Monte Carlo')
        plt.plot(maturities, fo_basis, 'b--', label='First-order')
        plt.plot(maturities, diff_basis, 'r-.', label='Diffusion')
        plt.plot(maturities, mr_basis, 'g:', label='Mean-Reverting')
        plt.title('Term Structure of Implied Basis Spreads')
        plt.xlabel('Maturity (years)')
        plt.ylabel('Implied Basis (bps)')
        plt.legend()
        plt.grid(True)
        
        return {
            'maturities': maturities,
            'mc_values': mc_values,
            'fo_values': fo_values,
            'diff_values': diff_values,
            'mr_values': mr_values
        }
    
    def optimal_currency_analysis(self, days=365):
        """Analyze which currency is optimal over time"""
        # Create a finer time grid for analysis
        fine_times = np.linspace(0, self.T, days)
        
        # Initialize optimal currency matrix
        optimal_currency = np.zeros(days, dtype=int)
        
        # Interpolate mean and variance values
        from scipy.interpolate import interp1d
        
        mean_interp = [interp1d(self.times, self.means[:, i]) for i in range(self.n_currencies)]
        var_interp = [interp1d(self.times, self.variances[:, i]) for i in range(self.n_currencies)]
        
        # Simulate paths
        n_paths = 1000
        paths = self._simulate_paths(n_paths)
        
        # Find optimal currency at each time point
        for i, t in enumerate(fine_times):
            # Find closest time index in original grid
            idx = np.abs(self.times - t).argmin()
            
            # Count which currency is optimal at this time
            max_vals = np.maximum(0, np.max(paths[:, idx, :], axis=1))
            zero_best = np.sum(max_vals == 0)
            
            currency_counts = [zero_best]
            for j in range(self.n_currencies):
                curr_best = np.sum(paths[:, idx, j] == max_vals)
                currency_counts.append(curr_best)
            
            # Store most common optimal currency
            optimal_currency[i] = np.argmax(currency_counts)
        
        # Plot results
        plt.figure(figsize=(12, 6))
        
        # Create a color map
        colors = plt.cm.tab10(np.linspace(0, 1, self.n_currencies + 1))
        
        # Plot bands showing optimal currency
        for i in range(self.n_currencies + 1):
            mask = optimal_currency == i
            if np.any(mask):
                plt.fill_between(fine_times, 0, 1, where=mask, 
                                 color=colors[i], alpha=0.7, 
                                 label=f"{'Base' if i==0 else self.currencies[i]}")
        
        plt.title('Optimal Collateral Currency Over Time')
        plt.xlabel('Time (years)')
        plt.yticks([])
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=self.n_currencies+1)
        plt.tight_layout()
        
        return optimal_currency


class TradingStrategy:
    """Class for implementing trading strategies based on the collateral choice option model"""
    
    def __init__(self, option_model):
        """
        Initialize trading strategy
        
        Parameters:
        -----------
        option_model : CollateralChoiceOption
            Pricing model for the collateral choice option
        """
        self.model = option_model
        self.trade_log = []
    
    def relative_value_strategy(self, market_price, threshold=0.05):
        """
        Implement relative value trading strategy
        
        Parameters:
        -----------
        market_price : float
            Market price of the option as a discount factor
        threshold : float
            Trading threshold (proportional difference)
        
        Returns:
        --------
        trade : dict
            Trade recommendation
        """
        # Compute model price using diffusion method
        model_price = self.model.value_option('diffusion')
        
        # Calculate percentage difference
        pct_diff = (model_price - market_price) / market_price
        
        # Determine trade direction
        if pct_diff > threshold:
            action = "BUY"
            reason = f"Option undervalued by {pct_diff*100:.2f}%"
        elif pct_diff < -threshold:
            action = "SELL"
            reason = f"Option overvalued by {abs(pct_diff)*100:.2f}%"
        else:
            action = "HOLD"
            reason = "Price within threshold"
        
        trade = {
            'date': datetime.now().strftime("%Y-%m-%d"),
            'action': action,
            'market_price': market_price,
            'model_price': model_price,
            'pct_diff': pct_diff,
            'reason': reason
        }
        
        self.trade_log.append(trade)
        return trade
    
    def volatility_trading_strategy(self, current_vols, historical_vols, threshold=0.20):
        """
        Implement volatility-based trading strategy
        
        Parameters:
        -----------
        current_vols : array-like
            Current volatility parameters
        historical_vols : array-like
            Historical average volatility parameters
        threshold : float
            Trading threshold (proportional difference)
        
        Returns:
        --------
        trade : dict
            Trade recommendation
        """
        # Calculate average volatility change
        vol_changes = [(c - h) / h for c, h in zip(current_vols, historical_vols)]
        avg_vol_change = np.mean(vol_changes)
        
        # Original model price
        original_xi = self.model.xi.copy()
        original_price = self.model.value_option('diffusion')
        
        # Set current volatilities
        self.model.xi = np.array(current_vols)
        
        # Recompute model
        self.model.means, self.model.variances = self.model._compute_means_variances()
        self.model.correlations = self.model._compute_correlations()
        self.model.gamma, self.model.C_var, self.model.A_means, self.model.A_vars = self.model._compute_common_factor_parameters()
        self.model.E_M, self.model.Var_M = self.model._compute_maximum_moments()
        self.model.probs = self.model._compute_maximum_probabilities()
        
        # New model price
        new_price = self.model.value_option('diffusion')
        
        # Restore original volatilities
        self.model.xi = original_xi
        self.model.means, self.model.variances = self.model._compute_means_variances()
        self.model.correlations = self.model._compute_correlations()
        self.model.gamma, self.model.C_var, self.model.A_means, self.model.A_vars = self.model._compute_common_factor_parameters()
        self.model.E_M, self.model.Var_M = self.model._compute_maximum_moments()
        self.model.probs = self.model._compute_maximum_probabilities()
        
        # Calculate price impact
        price_impact = (new_price - original_price) / original_price
        
        # Determine trade direction
        if avg_vol_change > threshold and price_impact < 0:
            action = "BUY"
            reason = f"Volatility increased by {avg_vol_change*100:.2f}%, price impact {price_impact*100:.2f}%"
        elif avg_vol_change < -threshold and price_impact > 0:
            action = "SELL"
            reason = f"Volatility decreased by {abs(avg_vol_change)*100:.2f}%, price impact {price_impact*100:.2f}%"
        else:
            action = "HOLD"
            reason = "Volatility changes within threshold or price impact contrary to expectation"
        
        trade = {
            'date': datetime.now().strftime("%Y-%m-%d"),
            'action': action,
            'avg_vol_change': avg_vol_change,
            'price_impact': price_impact,
            'reason': reason
        }
        
        self.trade_log.append(trade)
        return trade
    
    def term_structure_strategy(self, market_term_structure, maturities):
        """
        Implement term structure trading strategy
        
        Parameters:
        -----------
        market_term_structure : array-like
            Market prices for different maturities
        maturities : array-like
            Corresponding maturities in years
        
        Returns:
        --------
        trades : list of dict
            Trade recommendations
        """
        # Compute model term structure
        original_T = self.model.T
        original_times = self.model.times
        
        model_prices = []
        for T in maturities:
            # Set maturity
            self.model.T = T
            self.model.times = np.linspace(0, T, self.model.time_steps)
            
            # Recompute model
            self.model.means, self.model.variances = self.model._compute_means_variances()
            self.model.correlations = self.model._compute_correlations()
            self.model.gamma, self.model.C_var, self.model.A_means, self.model.A_vars = self.model._compute_common_factor_parameters()
            self.model.E_M, self.model.Var_M = self.model._compute_maximum_moments()
            self.model.probs = self.model._compute_maximum_probabilities()
            
            # Get model price
            model_prices.append(self.model.value_option('diffusion'))
        
        # Restore original maturity
        self.model.T = original_T
        self.model.times = original_times
        self.model.means, self.model.variances = self.model._compute_means_variances()
        self.model.correlations = self.model._compute_correlations()
        self.model.gamma, self.model.C_var, self.model.A_means, self.model.A_vars = self.model._compute_common_factor_parameters()
        self.model.E_M, self.model.Var_M = self.model._compute_maximum_moments()
        self.model.probs = self.model._compute_maximum_probabilities()
        
        # Calculate percentage differences
        pct_diffs = [(m - p) / p for m, p in zip(model_prices, market_term_structure)]
        
        # Find trading opportunities
        trades = []
        for i, (maturity, pct_diff) in enumerate(zip(maturities, pct_diffs)):
            if pct_diff > 0.05:  # Undervalued
                action = "BUY"
                reason = f"{maturity}y maturity undervalued by {pct_diff*100:.2f}%"
            elif pct_diff < -0.05:  # Overvalued
                action = "SELL"
                reason = f"{maturity}y maturity overvalued by {abs(pct_diff)*100:.2f}%"
            else:
                action = "HOLD"
                reason = f"{maturity}y maturity fairly valued"
            
            trade = {
                'date': datetime.now().strftime("%Y-%m-%d"),
                'maturity': maturity,
                'action': action,
                'market_price': market_term_structure[i],
                'model_price': model_prices[i],
                'pct_diff': pct_diff,
                'reason': reason
            }
            
            trades.append(trade)
            self.trade_log.append(trade)
        
        # Look for calendar spread opportunities
        for i in range(len(maturities) - 1):
            for j in range(i + 1, len(maturities)):
                market_spread = market_term_structure[i] - market_term_structure[j]
                model_spread = model_prices[i] - model_prices[j]
                spread_diff = model_spread - market_spread
                
                if abs(spread_diff) > 0.02:  # Significant spread difference
                    if spread_diff > 0:
                        action = f"BUY {maturities[i]}y - SELL {maturities[j]}y"
                        reason = f"Calendar spread undervalued by {spread_diff*100:.2f}%"
                    else:
                        action = f"SELL {maturities[i]}y - BUY {maturities[j]}y"
                        reason = f"Calendar spread overvalued by {abs(spread_diff)*100:.2f}%"
                    
                    trade = {
                        'date': datetime.now().strftime("%Y-%m-%d"),
                        'type': 'calendar_spread',
                        'maturities': (maturities[i], maturities[j]),
                        'action': action,
                        'market_spread': market_spread,
                        'model_spread': model_spread,
                        'spread_diff': spread_diff,
                        'reason': reason
                    }
                    
                    trades.append(trade)
                    self.trade_log.append(trade)
        
        return trades
    
    def currency_diversification_strategy(self, available_currencies, current_portfolio):
        """
        Recommend optimal currency additions based on diversification benefits
        
        Parameters:
        -----------
        available_currencies : list
            List of available currency names
        current_portfolio : list
            List of currency names currently in portfolio
        
        Returns:
        --------
        recommendations : dict
            Currency addition recommendations
        """
        base_value = self.model.value_option('diffusion')
        
        # Value with just the current portfolio
        current_indices = [self.model.currencies.index(c) for c in current_portfolio 
                          if c in self.model.currencies]
        
        # If no current portfolio or all currencies already included, return base value
        if not current_indices or len(current_indices) == len(self.model.currencies) - 1:
            return {
                'current_value': base_value,
                'recommendations': []
            }
        
        # Create mask for current portfolio
        mask = np.zeros(self.model.n_currencies, dtype=bool)
        for idx in current_indices:
            if idx > 0:  # Skip base currency
                mask[idx-1] = True
        
        # Initialize model with only current portfolio
        original_model = {
            'n_currencies': self.model.n_currencies,
            'q0': self.model.q0.copy(),
            'kappa': self.model.kappa.copy(),
            'xi': self.model.xi.copy(),
            'theta': self.model.theta.copy(),
            'correlation_matrix': self.model.correlation_matrix.copy()
        }
        
        # Find value with current portfolio
        active_indices = np.where(mask)[0]
        if len(active_indices) > 0:
            # Adjust model to include only current portfolio
            self.model.n_currencies = len(active_indices)
            self.model.q0 = self.model.q0[active_indices]
            self.model.kappa = self.model.kappa[active_indices]
            self.model.xi = self.model.xi[active_indices]
            self.model.theta = self.model.theta[active_indices]
            
            # Adjust correlation matrix
            self.model.correlation_matrix = self.model.correlation_matrix[np.ix_(active_indices, active_indices)]
            
            # Recompute model
            self.model.means, self.model.variances = self.model._compute_means_variances()
            self.model.correlations = self.model._compute_correlations()
            self.model.gamma, self.model.C_var, self.model.A_means, self.model.A_vars = self.model._compute_common_factor_parameters()
            self.model.E_M, self.model.Var_M = self.model._compute_maximum_moments()
            self.model.probs = self.model._compute_maximum_probabilities()
            
            current_value = self.model.value_option('diffusion')
        else:
            # No currencies in portfolio, so value is 1.0 (no option)
            current_value = 1.0
        
        # Restore original model
        self.model.n_currencies = original_model['n_currencies']
        self.model.q0 = original_model['q0']
        self.model.kappa = original_model['kappa']
        self.model.xi = original_model['xi']
        self.model.theta = original_model['theta']
        self.model.correlation_matrix = original_model['correlation_matrix']
        
        # Recompute model
        self.model.means, self.model.variances = self.model._compute_means_variances()
        self.model.correlations = self.model._compute_correlations()
        self.model.gamma, self.model.C_var, self.model.A_means, self.model.A_vars = self.model._compute_common_factor_parameters()
        self.model.E_M, self.model.Var_M = self.model._compute_maximum_moments()
        self.model.probs = self.model._compute_maximum_probabilities()
        
        # Calculate value of adding each available currency
        recommendations = []
        for currency in available_currencies:
            if currency in current_portfolio or currency not in self.model.currencies:
                continue
            
            # Create a new portfolio with this currency added
            new_portfolio = current_portfolio + [currency]
            new_indices = [self.model.currencies.index(c) for c in new_portfolio 
                          if c in self.model.currencies]
            
            # Create mask for new portfolio
            new_mask = np.zeros(self.model.n_currencies, dtype=bool)
            for idx in new_indices:
                if idx > 0:  # Skip base currency
                    new_mask[idx-1] = True
            
            # Adjust model to include new portfolio
            active_indices = np.where(new_mask)[0]
            if len(active_indices) > 0:
                self.model.n_currencies = len(active_indices)
                self.model.q0 = original_model['q0'][active_indices]
                self.model.kappa = original_model['kappa'][active_indices]
                self.model.xi = original_model['xi'][active_indices]
                self.model.theta = original_model['theta'][active_indices]
                
                # Adjust correlation matrix
                self.model.correlation_matrix = original_model['correlation_matrix'][np.ix_(active_indices, active_indices)]
                
                # Recompute model
                self.model.means, self.model.variances = self.model._compute_means_variances()
                self.model.correlations = self.model._compute_correlations()
                self.model.gamma, self.model.C_var, self.model.A_means, self.model.A_vars = self.model._compute_common_factor_parameters()
                self.model.E_M, self.model.Var_M = self.model._compute_maximum_moments()
                self.model.probs = self.model._compute_maximum_probabilities()
                
                new_value = self.model.value_option('diffusion')
                
                # Calculate improvement
                improvement = (current_value - new_value) / current_value
                
                recommendations.append({
                    'currency': currency,
                    'value_with_currency': new_value,
                    'improvement': improvement,
                    'implied_basis_improvement': -np.log(new_value / current_value) / self.model.T * 10000
                })
        
        # Restore original model
        self.model.n_currencies = original_model['n_currencies']
        self.model.q0 = original_model['q0']
        self.model.kappa = original_model['kappa']
        self.model.xi = original_model['xi']
        self.model.theta = original_model['theta']
        self.model.correlation_matrix = original_model['correlation_matrix']
        
        # Recompute model
        self.model.means, self.model.variances = self.model._compute_means_variances()
        self.model.correlations = self.model._compute_correlations()
        self.model.gamma, self.model.C_var, self.model.A_means, self.model.A_vars = self.model._compute_common_factor_parameters()
        self.model.E_M, self.model.Var_M = self.model._compute_maximum_moments()
        self.model.probs = self.model._compute_maximum_probabilities()
        
        # Sort recommendations by improvement
        recommendations.sort(key=lambda x: x['improvement'], reverse=True)
        
        return {
            'current_value': current_value,
            'full_value': base_value,
            'potential_improvement': (current_value - base_value) / current_value,
            'recommendations': recommendations
        }
    
    def get_trade_log(self):
        """Get the log of all trading decisions"""
        return pd.DataFrame(self.trade_log)


# Example usage and results
def main():
    print("=== Collateral Choice Option Trading Strategy Analysis ===\n")
    
    # Define parameters for USD (base), EUR, and JPY
    currencies = ["USD", "EUR", "JPY"]
    initial_spreads = [0.000845, 0.001514]  # EUR and JPY spreads vs USD
    mean_reversion_speeds = [0.0078, 0.0076]
    volatilities = [0.0018, 0.0023]
    correlation_matrix = np.array([
        [1.0, 0.3],
        [0.3, 1.0]
    ])
    maturity = 20.0  # 20 years
    
    # Create the model
    print("Creating model for USD, EUR, and JPY collateral currencies...")
    ctd_model = CollateralChoiceOption(
        currencies=currencies,
        initial_spreads=initial_spreads,
        mean_reversion_speeds=mean_reversion_speeds,
        volatilities=volatilities,
        correlation_matrix=correlation_matrix,
        maturity=maturity
    )
    
    # Value the option with different methods
    print("\nValuing the collateral choice option...")
    mc_value = ctd_model.monte_carlo_value(10000)
    fo_value = ctd_model.value_option('first_order')
    diff_value = ctd_model.value_option('diffusion')
    mr_value = ctd_model.value_option('mean_reverting')
    
    print(f"Monte Carlo value: {mc_value:.8f}")
    print(f"First-order estimator: {fo_value:.8f}")
    print(f"Diffusion-based estimator: {diff_value:.8f}")
    print(f"Mean-reverting estimator: {mr_value:.8f}")
    
    # Convert to implied basis points
    mc_basis = -np.log(mc_value) / maturity * 10000
    fo_basis = -np.log(fo_value) / maturity * 10000
    diff_basis = -np.log(diff_value) / maturity * 10000
    mr_basis = -np.log(mr_value) / maturity * 10000
    
    print(f"\nImplied basis spreads:")
    print(f"Monte Carlo: {mc_basis:.2f} bps")
    print(f"First-order estimator: {fo_basis:.2f} bps")
    print(f"Diffusion-based estimator: {diff_basis:.2f} bps")
    print(f"Mean-reverting estimator: {mr_basis:.2f} bps")
    
    # Create trading strategy
    print("\n=== Trading Strategy Analysis ===\n")
    strategy = TradingStrategy(ctd_model)
    
    # 1. Relative Value Strategy
    print("Analyzing relative value trading opportunities...")
    # Assume market price is 5% higher than model price
    market_price = diff_value * 1.05
    trade = strategy.relative_value_strategy(market_price)
    print(f"Action: {trade['action']}")
    print(f"Reason: {trade['reason']}")
    print(f"Model price: {trade['model_price']:.8f}, Market price: {trade['market_price']:.8f}")
    
    # 2. Volatility Trading Strategy
    print("\nAnalyzing volatility-based trading opportunities...")
    # Assume current volatilities are 20% higher than historical
    current_vols = [v * 1.2 for v in volatilities]
    historical_vols = volatilities
    trade = strategy.volatility_trading_strategy(current_vols, historical_vols)
    print(f"Action: {trade['action']}")
    print(f"Reason: {trade['reason']}")
    print(f"Volatility change: {trade['avg_vol_change']*100:.2f}%, Price impact: {trade['price_impact']*100:.2f}%")
    
    # 3. Term Structure Strategy
    print("\nAnalyzing term structure trading opportunities...")
    maturities = [1, 3, 5, 7, 10, 15, 20]
    
    # Define a hypothetical market term structure (with some mispricings)
    market_term_structure = [0.99, 0.97, 0.95, 0.92, 0.88, 0.84, 0.79]
    trades = strategy.term_structure_strategy(market_term_structure, maturities)
    
    # Display directional trades
    print("Directional recommendations:")
    for trade in trades:
        if 'type' not in trade:
            print(f"{trade['maturity']}y: {trade['action']} - {trade['reason']}")
    
    # Display calendar spread trades
    print("\nCalendar spread recommendations:")
    for trade in trades:
        if 'type' in trade and trade['type'] == 'calendar_spread':
            print(f"{trade['action']} - {trade['reason']}")
    
    # 4. Currency Diversification Strategy
    print("\nAnalyzing currency diversification benefits...")
    
    # Assume we're starting with just EUR as collateral currency
    current_portfolio = ["EUR"]
    available_currencies = ["JPY", "GBP", "CHF"]  # Some may not be in our model
    recommendations = strategy.currency_diversification_strategy(available_currencies, current_portfolio)
    
    print(f"Current portfolio value: {recommendations['current_value']:.8f}")
    print(f"Potential improvement: {recommendations['potential_improvement']*100:.2f}%")
    
    print("\nCurrency addition recommendations:")
    for rec in recommendations['recommendations']:
        print(f"Add {rec['currency']}: Improves by {rec['improvement']*100:.2f}% " + 
              f"({rec['implied_basis_improvement']:.2f} bps)")
    
    # 5. Optimal Currency Analysis
    print("\nAnalyzing optimal collateral currency over time...")
    optimal_currency = ctd_model.optimal_currency_analysis(days=365)
    
    # Count days each currency is optimal
    counts = np.bincount(optimal_currency, minlength=len(currencies))
    for i, count in enumerate(counts):
        curr_name = "Base" if i == 0 else currencies[i]
        print(f"{curr_name}: Optimal for {count} days ({count/365*100:.1f}% of time)")
    
    print("\nTrading strategy analysis complete!")
    
    return {
        'model': ctd_model,
        'strategy': strategy,
        'option_values': {
            'monte_carlo': mc_value,
            'first_order': fo_value,
            'diffusion': diff_value,
            'mean_reverting': mr_value
        },
        'basis_spreads': {
            'monte_carlo': mc_basis,
            'first_order': fo_basis,
            'diffusion': diff_basis,
            'mean_reverting': mr_basis
        }
    }

if __name__ == "__main__":
    results = main()