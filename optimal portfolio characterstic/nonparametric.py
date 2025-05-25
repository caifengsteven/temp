"""
Non-parametric Estimation Module for Optimal Characteristic Portfolios

This module implements the non-parametric estimation methods described in the paper
"Optimal Characteristic Portfolios" by Richard McGee and Jose Olmo.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm


class NonParametricEstimator:
    """
    Class for non-parametric estimation of conditional mean and variance
    using the Nadaraya-Watson estimator
    """
    
    def __init__(self, kernel_type='gaussian'):
        """
        Initialize the non-parametric estimator
        
        Parameters:
        -----------
        kernel_type : str
            Type of kernel function to use ('gaussian', 'uniform', 'epanechnikov')
        """
        self.kernel_type = kernel_type
        
        # Define kernel functions
        if kernel_type == 'gaussian':
            self.kernel_func = lambda u: norm.pdf(u)
        elif kernel_type == 'uniform':
            self.kernel_func = lambda u: 0.5 * (np.abs(u) <= 1)
        elif kernel_type == 'epanechnikov':
            self.kernel_func = lambda u: 0.75 * (1 - u**2) * (np.abs(u) <= 1)
        else:
            print(f"Unknown kernel type: {kernel_type}. Using Gaussian kernel.")
            self.kernel_func = lambda u: norm.pdf(u)
    
    def kernel(self, z, z_i, h):
        """
        Compute kernel function value
        
        Parameters:
        -----------
        z : float
            Point at which to evaluate
        z_i : float
            Data point
        h : float
            Bandwidth parameter
        
        Returns:
        --------
        float
            Kernel function value
        """
        return self.kernel_func((z - z_i) / h) / h
    
    def estimate_bandwidth(self, z, rule='silverman'):
        """
        Estimate optimal bandwidth using rule-of-thumb methods
        
        Parameters:
        -----------
        z : array-like
            Data for which to estimate bandwidth
        rule : str
            Rule to use for bandwidth estimation ('silverman', 'scott')
        
        Returns:
        --------
        float
            Estimated bandwidth
        """
        n = len(z)
        std_z = np.std(z)
        
        if rule == 'silverman':
            # Silverman's rule of thumb
            return 0.9 * std_z * n**(-1/5)
        elif rule == 'scott':
            # Scott's rule of thumb
            return 1.06 * std_z * n**(-1/5)
        else:
            # Default rule from the paper
            return std_z * n**(-1/5)
    
    def nadaraya_watson(self, x, y, x_eval, h=None):
        """
        Nadaraya-Watson estimator for conditional mean
        
        Parameters:
        -----------
        x : array-like
            Predictor variable (characteristic)
        y : array-like
            Response variable (returns)
        x_eval : array-like
            Points at which to evaluate the estimator
        h : float, optional
            Bandwidth parameter. If None, estimated using rule of thumb.
        
        Returns:
        --------
        array-like
            Estimated conditional mean at x_eval points
        """
        x = np.asarray(x)
        y = np.asarray(y)
        x_eval = np.asarray(x_eval)
        
        if h is None:
            h = self.estimate_bandwidth(x)
        
        n = len(x)
        m = len(x_eval)
        
        # Initialize result array
        result = np.zeros(m)
        
        # Compute Nadaraya-Watson estimator
        for i in range(m):
            weights = np.array([self.kernel(x_eval[i], x[j], h) for j in range(n)])
            sum_weights = np.sum(weights)
            
            if sum_weights > 0:
                result[i] = np.sum(weights * y) / sum_weights
            else:
                result[i] = np.mean(y)  # Fallback if all weights are zero
        
        return result
    
    def estimate_conditional_variance(self, x, y, x_eval, h=None):
        """
        Estimate conditional variance using residuals from Nadaraya-Watson estimator
        
        Parameters:
        -----------
        x : array-like
            Predictor variable (characteristic)
        y : array-like
            Response variable (returns)
        x_eval : array-like
            Points at which to evaluate the estimator
        h : float, optional
            Bandwidth parameter. If None, estimated using rule of thumb.
        
        Returns:
        --------
        array-like
            Estimated conditional variance at x_eval points
        """
        x = np.asarray(x)
        y = np.asarray(y)
        x_eval = np.asarray(x_eval)
        
        if h is None:
            h = self.estimate_bandwidth(x)
        
        # First, estimate conditional mean
        y_hat = self.nadaraya_watson(x, y, x, h)
        
        # Compute squared residuals
        residuals_squared = (y - y_hat)**2
        
        # Estimate conditional variance using Nadaraya-Watson on squared residuals
        return self.nadaraya_watson(x, residuals_squared, x_eval, h)
    
    def estimate_optimal_weights(self, x, y, x_eval, gamma=1.0, h=None):
        """
        Estimate optimal portfolio weights based on mean-variance optimization
        
        Parameters:
        -----------
        x : array-like
            Predictor variable (characteristic)
        y : array-like
            Response variable (returns)
        x_eval : array-like
            Points at which to evaluate the weights
        gamma : float
            Risk aversion parameter
        h : float, optional
            Bandwidth parameter. If None, estimated using rule of thumb.
        
        Returns:
        --------
        array-like
            Estimated optimal weights at x_eval points
        """
        x = np.asarray(x)
        y = np.asarray(y)
        x_eval = np.asarray(x_eval)
        
        if h is None:
            h = self.estimate_bandwidth(x)
        
        # Estimate conditional mean and variance
        cond_mean = self.nadaraya_watson(x, y, x_eval, h)
        cond_var = self.estimate_conditional_variance(x, y, x_eval, h)
        
        # Compute optimal weights
        # w*(z) = μ(z) / (γ * σ²(z))
        weights = cond_mean / (gamma * cond_var)
        
        return weights
    
    def normalize_weights_dollar_neutral(self, weights):
        """
        Normalize weights to satisfy dollar-neutrality constraint
        
        Parameters:
        -----------
        weights : array-like
            Unnormalized weights
        
        Returns:
        --------
        array-like
            Normalized weights that sum to zero and have unit leverage
        """
        weights = np.asarray(weights)
        
        # Ensure weights sum to zero (dollar-neutral)
        weights = weights - np.mean(weights)
        
        # Scale to unit leverage (sum of absolute values of negative weights = 1)
        neg_weights = weights[weights < 0]
        pos_weights = weights[weights > 0]
        
        if len(neg_weights) > 0 and len(pos_weights) > 0:
            neg_sum = np.sum(np.abs(neg_weights))
            pos_sum = np.sum(pos_weights)
            
            # Scale negative and positive weights separately
            weights[weights < 0] = weights[weights < 0] / neg_sum
            weights[weights > 0] = weights[weights > 0] / pos_sum
        
        return weights


# Example usage
if __name__ == "__main__":
    # Generate some sample data
    np.random.seed(42)
    x = np.random.uniform(0, 10, 100)  # Characteristic (e.g., size)
    y = 0.1 * x + 0.5 * np.sin(x) + np.random.normal(0, 1, 100)  # Returns
    
    # Create non-parametric estimator
    estimator = NonParametricEstimator(kernel_type='gaussian')
    
    # Estimate optimal bandwidth
    h = estimator.estimate_bandwidth(x)
    print(f"Estimated bandwidth: {h:.4f}")
    
    # Evaluate at grid points
    x_grid = np.linspace(0, 10, 50)
    
    # Estimate conditional mean
    cond_mean = estimator.nadaraya_watson(x, y, x_grid, h)
    
    # Estimate conditional variance
    cond_var = estimator.estimate_conditional_variance(x, y, x_grid, h)
    
    # Estimate optimal weights
    weights = estimator.estimate_optimal_weights(x, y, x_grid, gamma=2.0, h=h)
    
    # Normalize weights
    norm_weights = estimator.normalize_weights_dollar_neutral(weights)
    
    # Print results
    print("\nSample results:")
    for i in range(5):
        print(f"x={x_grid[i]:.2f}, E[y|x]={cond_mean[i]:.4f}, Var[y|x]={cond_var[i]:.4f}, w*={norm_weights[i]:.4f}")
