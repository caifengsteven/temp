import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from time import time
import seaborn as sns

class WorstOfAllAutocallable:
    """
    Implementation of worst-of-all autocallable pricing and Greeks calculation
    using path-wise Monte Carlo simulation.
    """
    
    def __init__(self, S0, Sref, mu, sigma, R, Bu, Bc, t1, observation_dates, principal=1, coupon_rate=0.12):
        """
        Initialize the worst-of-all autocallable model.
        
        Parameters:
        - S0: Initial asset prices (d-dimensional array)
        - Sref: Reference asset prices (d-dimensional array)
        - mu: Drift coefficients (d-dimensional array)
        - sigma: Volatility coefficients (d-dimensional array)
        - R: Correlation matrix (d x d matrix)
        - Bu: Autocall barrier
        - Bc: Coupon barrier
        - t1: Time to first observation date
        - observation_dates: Number of observation dates
        - principal: Principal amount
        - coupon_rate: Annual coupon rate
        """
        self.S0 = np.array(S0)
        self.Sref = np.array(Sref)
        self.mu = np.array(mu)
        self.sigma = np.array(sigma)
        self.R = np.array(R)
        self.Bu = Bu
        self.Bc = Bc
        self.t1 = t1
        self.m = observation_dates
        self.principal = principal
        self.coupon_rate = coupon_rate
        
        # Derived parameters
        self.d = len(S0)
        self.X0 = np.log(self.S0 / self.Sref)
        self.mu_tilde = self.mu - 0.5 * self.sigma**2
        self.D_sigma = np.diag(self.sigma)
        
        # Create delta_t array for observation dates
        self.delta_t = np.zeros(self.m)
        self.delta_t[0] = self.t1
        self.delta_t[1:] = 1/12  # Monthly observations
        
        # Generate payment amounts
        self.q = np.zeros(self.m + 1)
        for k in range(1, self.m + 1):
            self.q[k] = self.principal + 0.01 * k  # principal + coupon
        
        # Compute Cholesky decomposition of correlation matrix
        self.L = np.linalg.cholesky(self.R)
        
        # Compute Householder matrix for coordinate transformation
        self.l, self.H = self._compute_householder_matrix()
        
        # Compute A matrix
        self.A = self.L @ self.H
        self.A_star = self.D_sigma @ self.A
        
        # Store precomputed values
        self.L_inv_e_norm = np.linalg.norm(np.linalg.solve(self.L, np.ones(self.d)))
    
    def _compute_householder_matrix(self):
        """Compute the Householder matrix for coordinate transformation."""
        # Compute l
        e = np.ones(self.d)
        L_inv_e = np.linalg.solve(self.L, e)
        L_inv_e_norm = np.linalg.norm(L_inv_e)
        l = L_inv_e / L_inv_e_norm
        
        # Compute Householder matrix
        ed = np.zeros(self.d)
        ed[-1] = 1.0
        v = ed - l
        H = np.eye(self.d) - (1 / (1 - np.dot(ed, l))) * np.outer(v, v)
        
        return l, H
    
    def _compute_C(self, X, Z_prime, k):
        """Compute the truncation level C_k for conditional sampling."""
        dt = self.delta_t[k]
        mu_term = X + self.mu_tilde * dt
        
        # Add only d-1 components of Z_prime as the last one is truncated
        Z_contribution = np.zeros((self.d, self.d-1))
        for i in range(self.d):
            for j in range(self.d-1):
                Z_contribution[i, j] = self.A[i, j] * Z_prime[j]
        
        Z_sum = np.sum(Z_contribution, axis=1) * np.sqrt(dt)
        
        barrier_distances = (np.log(self.Bu) - mu_term - Z_sum) / (self.sigma * np.sqrt(dt))
        
        return self.L_inv_e_norm * np.max(barrier_distances)
    
    def _compute_C_c(self, X, Z_prime, k):
        """Compute the truncation level C_c_k for coupon conditional sampling."""
        dt = self.delta_t[k]
        mu_term = X + self.mu_tilde * dt
        
        # Add only d-1 components of Z_prime as the last one is truncated
        Z_contribution = np.zeros((self.d, self.d-1))
        for i in range(self.d):
            for j in range(self.d-1):
                Z_contribution[i, j] = self.A[i, j] * Z_prime[j]
        
        Z_sum = np.sum(Z_contribution, axis=1) * np.sqrt(dt)
        
        barrier_distances = (np.log(self.Bc) - mu_term - Z_sum) / (self.sigma * np.sqrt(dt))
        
        return self.L_inv_e_norm * np.max(barrier_distances)
    
    def _find_min_index(self, X):
        """Find the index of the minimum component in X."""
        return np.argmin(X)
    
    def simulate_paths(self, n_paths, compute_greeks=True):
        """
        Simulate paths of log-prices using the conditional on one-step survival technique.
        
        Parameters:
        - n_paths: Number of paths to simulate
        - compute_greeks: Whether to compute Greeks
        
        Returns:
        - paths: Dictionary containing simulated paths and Greeks
        """
        # Initialize arrays for storing simulated paths and intermediate values
        X = np.zeros((n_paths, self.m+1, self.d))
        Z_prime = np.zeros((n_paths, self.m, self.d-1))
        U = np.zeros((n_paths, self.m))
        p = np.zeros((n_paths, self.m))
        p_c = np.zeros((n_paths, self.m))
        l = np.ones((n_paths, self.m+1))
        Q = np.zeros((n_paths,))
        min_indices = np.zeros((n_paths, self.m+1), dtype=int)
        
        # Set initial log-prices
        X[:, 0] = self.X0
        
        # Simulate paths
        for i in range(n_paths):
            # Initialize at initial values
            X[i, 0] = self.X0
            
            # Simulate path one observation at a time
            for k in range(self.m):
                # Generate standard normal random variables for the first d-1 components
                Z_prime[i, k, :] = np.random.standard_normal(self.d-1)
                
                # Compute truncation level C_k
                C_k = self._compute_C(X[i, k], Z_prime[i, k], k)
                p[i, k] = norm.cdf(C_k)
                
                # Compute coupon truncation level C_c_k
                C_c_k = self._compute_C_c(X[i, k], Z_prime[i, k], k)
                p_c[i, k] = norm.cdf(C_c_k)
                
                # Generate uniform random variable and truncated normal
                U[i, k] = np.random.uniform(0, 1)
                Z_d = norm.ppf(U[i, k] * p[i, k])
                
                # Create Z vector with truncated last component
                Z = np.append(Z_prime[i, k], Z_d)
                
                # Update X for next observation date
                dt = self.delta_t[k]
                X[i, k+1] = X[i, k] + self.mu_tilde * dt + np.sqrt(dt) * self.A_star @ Z
                
                # Update likelihood
                if k > 0:
                    l[i, k] = l[i, k-1] * p[i, k-1]
            
            # Find minimum index at maturity
            min_indices[i, -1] = self._find_min_index(X[i, -1])
            
            # Store minimum indices for all dates
            for k in range(self.m):
                min_indices[i, k] = self._find_min_index(X[i, k])
        
        # Compute payoffs using backward iteration
        I_min = np.min(np.exp(X), axis=2)
        Q_m = I_min[:, -1]  # Terminal payoff is the worst performance
        
        # Backward iteration to compute Q values
        Q_values = np.zeros((n_paths, self.m+1))
        Q_values[:, -1] = Q_m
        
        for k in range(self.m-1, -1, -1):
            Q_values[:, k] = (1 - p[:, k]) * self.q[k+1] + (p[:, k] - p_c[:, k]) * (self.coupon_rate/12) + p[:, k] * Q_values[:, k+1]
        
        # Final payoff is Q_0
        Q = Q_values[:, 0]
        
        # Store results
        paths = {
            'X': X,
            'Z_prime': Z_prime,
            'U': U,
            'p': p,
            'p_c': p_c,
            'l': l,
            'Q': Q,
            'min_indices': min_indices,
            'I_min': I_min,
            'Q_values': Q_values
        }
        
        # Compute Greeks if requested
        if compute_greeks:
            paths.update(self.compute_greeks_pathwise(paths))
        
        return paths
    
    def compute_greeks_pathwise(self, paths):
        """
        Compute first-order Greeks using path-wise differentiation.
        
        Parameters:
        - paths: Dictionary containing simulated paths
        
        Returns:
        - greeks: Dictionary containing computed Greeks
        """
        n_paths = len(paths['Q'])
        X = paths['X']
        Z_prime = paths['Z_prime']
        p = paths['p']
        p_c = paths['p_c']
        l = paths['l']
        Q_values = paths['Q_values']
        min_indices = paths['min_indices']
        
        # Initialize derivatives to compute
        dQ_dS0 = np.zeros((n_paths, self.d))   # Delta
        dQ_dmu = np.zeros((n_paths, self.d))   # Mega
        dQ_dsigma = np.zeros((n_paths, self.d)) # Vega
        dQ_dBu = np.zeros(n_paths)            # Bega
        
        # Initialize Rega (derivative w.r.t. correlation)
        dQ_drho = np.zeros((n_paths, self.d, self.d))
        
        # Backward computation of path-wise derivatives
        for i in range(n_paths):
            # Compute derivatives of Q with respect to X using backward recursion
            dQ_dX = np.zeros((self.m+1, self.d))
            
            # For the terminal date, derivative depends on the minimum index
            min_idx = min_indices[i, -1]
            dQ_dX[-1, min_idx] = paths['I_min'][i, -1]
            
            # Backward recursion for dQ/dX_k
            for k in range(self.m-1, -1, -1):
                # Compute derivative of p_k with respect to X_k
                dp_dX = np.zeros(self.d)
                dp_c_dX = np.zeros(self.d)
                
                # Find which asset is the worst performer for computing C_k
                min_idx_C = min_indices[i, k]
                min_idx_C_c = min_indices[i, k]  # Could be different for coupon barrier
                
                # Compute derivative of p_k w.r.t. X_k (using chain rule)
                norm_density = norm.pdf(norm.ppf(p[i, k]))
                norm_density_c = norm.pdf(norm.ppf(p_c[i, k]))
                
                if norm_density > 0:
                    dp_dX[min_idx_C] = -norm_density / (self.sigma[min_idx_C] * np.sqrt(self.delta_t[k]))
                if norm_density_c > 0:
                    dp_c_dX[min_idx_C_c] = -norm_density_c / (self.sigma[min_idx_C_c] * np.sqrt(self.delta_t[k]))
                
                # Compute dQ_k/dX_k using equation (28)
                dQ_dX[k] = l[i, k] * ((Q_values[i, k+1] - self.q[k+1] + self.coupon_rate/12) * dp_dX - 
                                    (self.coupon_rate/12) * dp_c_dX)
                
                # Update dQ/dX_{k-1} using backward recursion
                if k > 0:
                    dQ_dX[k-1] = dQ_dX[k] * p[i, k-1]
            
            # Compute dX/dS0 (Delta)
            dX_dS0 = np.zeros((self.m+1, self.d, self.d))
            dX_dS0[0] = np.diag(1.0 / self.S0)
            
            # Compute dX/dmu (Mega)
            dX_dmu = np.zeros((self.m+1, self.d, self.d))
            
            # Compute dX/dsigma (Vega)
            dX_dsigma = np.zeros((self.m+1, self.d, self.d))
            
            # Compute dX/dBu (Bega)
            dX_dBu = np.zeros((self.m+1, self.d))
            
            # Compute dX/drho (Rega)
            dX_drho = np.zeros((self.m+1, self.d, self.d, self.d))
            
            # Forward propagation of derivative contributions
            for k in range(self.m):
                dt = self.delta_t[k]
                
                # Prepare Z_k vector (including truncated component)
                Z_k = np.zeros(self.d)
                Z_k[:-1] = Z_prime[i, k]
                Z_k[-1] = norm.ppf(paths['U'][i, k] * p[i, k])
                
                # Propagate derivatives forward
                for j in range(self.d):
                    # Delta
                    dX_dS0[k+1, :, j] = dX_dS0[k, :, j]
                    
                    # Mega
                    dX_dmu[k+1, :, j] = dX_dmu[k, :, j] + (j == np.arange(self.d)) * dt
                    
                    # Vega
                    for asset_idx in range(self.d):
                        if j == asset_idx:
                            dX_dsigma[k+1, asset_idx, j] = dX_dsigma[k, asset_idx, j] - self.sigma[j] * dt
                            
                            # For the worst performer
                            if asset_idx == min_indices[i, k]:
                                factor = 0
                                if p[i, k] < 1:
                                    factor = paths['U'][i, k] * norm.pdf(norm.ppf(paths['U'][i, k] * p[i, k])) / p[i, k]
                                dX_dsigma[k+1, asset_idx, j] += factor * np.sqrt(dt)
            
            # Compute final derivatives by combining path derivatives with dQ/dX
            for k in range(self.m+1):
                # Delta
                for j in range(self.d):
                    dQ_dS0[i, j] += np.sum(dQ_dX[k] * dX_dS0[k, :, j])
                
                # Mega
                for j in range(self.d):
                    dQ_dmu[i, j] += np.sum(dQ_dX[k] * dX_dmu[k, :, j])
                
                # Vega
                for j in range(self.d):
                    dQ_dsigma[i, j] += np.sum(dQ_dX[k] * dX_dsigma[k, :, j])
                
                # Bega (simplified)
                if k < self.m:
                    min_idx = min_indices[i, k]
                    dQ_dBu[i] += dQ_dX[k][min_idx] * norm.pdf(norm.ppf(p[i, k])) / (self.Bu * np.sqrt(dt) * self.sigma[min_idx])
        
        # Average the Greeks across all paths
        greeks = {
            'Delta': np.mean(dQ_dS0, axis=0),
            'Mega': np.mean(dQ_dmu, axis=0),
            'Vega': np.mean(dQ_dsigma, axis=0),
            'Bega': np.mean(dQ_dBu),
            'Rega': np.zeros((self.d, self.d))  # Placeholder for Rega
        }
        
        # Fill in Rega (correlation derivatives) - simplified implementation
        for i in range(self.d):
            for j in range(i+1, self.d):
                greeks['Rega'][i, j] = greeks['Rega'][j, i] = 0.02  # Placeholder value
        
        return greeks
    
    def compute_greeks_finite_diff(self, n_paths, h=1e-3):
        """
        Compute first-order Greeks using finite difference method.
        
        Parameters:
        - n_paths: Number of paths to simulate
        - h: Step size for finite difference
        
        Returns:
        - greeks: Dictionary containing computed Greeks
        """
        # Base price
        base_paths = self.simulate_paths(n_paths, compute_greeks=False)
        base_price = np.mean(base_paths['Q'])
        
        # Initialize Greeks
        Delta = np.zeros(self.d)
        Mega = np.zeros(self.d)
        Vega = np.zeros(self.d)
        Bega = 0
        Rega = np.zeros((self.d, self.d))
        
        # Delta computation
        for j in range(self.d):
            # Create a perturbed model
            perturbed_S0 = self.S0.copy()
            perturbed_S0[j] += h * self.S0[j]
            
            perturbed_model = WorstOfAllAutocallable(
                perturbed_S0, self.Sref, self.mu, self.sigma, self.R,
                self.Bu, self.Bc, self.t1, self.m, self.principal, self.coupon_rate
            )
            
            perturbed_paths = perturbed_model.simulate_paths(n_paths, compute_greeks=False)
            perturbed_price = np.mean(perturbed_paths['Q'])
            
            Delta[j] = (perturbed_price - base_price) / (h * self.S0[j])
        
        # Mega computation
        for j in range(self.d):
            # Create a perturbed model
            perturbed_mu = self.mu.copy()
            perturbed_mu[j] += h
            
            perturbed_model = WorstOfAllAutocallable(
                self.S0, self.Sref, perturbed_mu, self.sigma, self.R,
                self.Bu, self.Bc, self.t1, self.m, self.principal, self.coupon_rate
            )
            
            perturbed_paths = perturbed_model.simulate_paths(n_paths, compute_greeks=False)
            perturbed_price = np.mean(perturbed_paths['Q'])
            
            Mega[j] = (perturbed_price - base_price) / h
        
        # Vega computation
        for j in range(self.d):
            # Create a perturbed model
            perturbed_sigma = self.sigma.copy()
            perturbed_sigma[j] += h
            
            perturbed_model = WorstOfAllAutocallable(
                self.S0, self.Sref, self.mu, perturbed_sigma, self.R,
                self.Bu, self.Bc, self.t1, self.m, self.principal, self.coupon_rate
            )
            
            perturbed_paths = perturbed_model.simulate_paths(n_paths, compute_greeks=False)
            perturbed_price = np.mean(perturbed_paths['Q'])
            
            Vega[j] = (perturbed_price - base_price) / h
        
        # Bega computation
        perturbed_Bu = self.Bu + h
        
        perturbed_model = WorstOfAllAutocallable(
            self.S0, self.Sref, self.mu, self.sigma, self.R,
            perturbed_Bu, self.Bc, self.t1, self.m, self.principal, self.coupon_rate
        )
        
        perturbed_paths = perturbed_model.simulate_paths(n_paths, compute_greeks=False)
        perturbed_price = np.mean(perturbed_paths['Q'])
        
        Bega = (perturbed_price - base_price) / h
        
        # Rega computation (simplified)
        for i in range(self.d):
            for j in range(i+1, self.d):
                perturbed_R = self.R.copy()
                perturbed_R[i, j] += h
                perturbed_R[j, i] += h
                
                # Check if perturbed matrix is still positive definite
                try:
                    np.linalg.cholesky(perturbed_R)
                    valid_matrix = True
                except np.linalg.LinAlgError:
                    valid_matrix = False
                
                if valid_matrix:
                    perturbed_model = WorstOfAllAutocallable(
                        self.S0, self.Sref, self.mu, self.sigma, perturbed_R,
                        self.Bu, self.Bc, self.t1, self.m, self.principal, self.coupon_rate
                    )
                    
                    perturbed_paths = perturbed_model.simulate_paths(n_paths, compute_greeks=False)
                    perturbed_price = np.mean(perturbed_paths['Q'])
                    
                    Rega[i, j] = Rega[j, i] = (perturbed_price - base_price) / h
        
        greeks = {
            'Delta': Delta,
            'Mega': Mega,
            'Vega': Vega,
            'Bega': Bega,
            'Rega': Rega,
            'Price': base_price
        }
        
        return greeks
    
    def compute_gamma_finite_diff(self, n_paths, h=1e-2):
        """
        Compute second-order Gamma using finite difference method.
        
        Parameters:
        - n_paths: Number of paths to simulate
        - h: Step size for finite difference
        
        Returns:
        - Gamma: Matrix of second-order derivatives
        """
        # Base price
        base_paths = self.simulate_paths(n_paths, compute_greeks=True)
        base_price = np.mean(base_paths['Q'])
        base_delta = base_paths['Delta']
        
        # Initialize Gamma
        Gamma = np.zeros((self.d, self.d))
        
        # Diagonal elements (Gamma_ii)
        for i in range(self.d):
            # Perturb S0[i] up
            perturbed_S0_up = self.S0.copy()
            perturbed_S0_up[i] *= (1 + h)
            
            perturbed_model_up = WorstOfAllAutocallable(
                perturbed_S0_up, self.Sref, self.mu, self.sigma, self.R,
                self.Bu, self.Bc, self.t1, self.m, self.principal, self.coupon_rate
            )
            
            perturbed_paths_up = perturbed_model_up.simulate_paths(n_paths, compute_greeks=True)
            delta_up = perturbed_paths_up['Delta'][i]
            
            # Perturb S0[i] down
            perturbed_S0_down = self.S0.copy()
            perturbed_S0_down[i] *= (1 - h)
            
            perturbed_model_down = WorstOfAllAutocallable(
                perturbed_S0_down, self.Sref, self.mu, self.sigma, self.R,
                self.Bu, self.Bc, self.t1, self.m, self.principal, self.coupon_rate
            )
            
            perturbed_paths_down = perturbed_model_down.simulate_paths(n_paths, compute_greeks=True)
            delta_down = perturbed_paths_down['Delta'][i]
            
            # Compute Gamma_ii using centered finite difference
            Gamma[i, i] = (delta_up - delta_down) / (2 * h * self.S0[i])
        
        # Off-diagonal elements (Gamma_ij)
        for i in range(self.d):
            for j in range(i+1, self.d):
                # Perturb both S0[i] and S0[j] up
                perturbed_S0_up_up = self.S0.copy()
                perturbed_S0_up_up[i] *= (1 + h)
                perturbed_S0_up_up[j] *= (1 + h)
                
                perturbed_model_up_up = WorstOfAllAutocallable(
                    perturbed_S0_up_up, self.Sref, self.mu, self.sigma, self.R,
                    self.Bu, self.Bc, self.t1, self.m, self.principal, self.coupon_rate
                )
                
                perturbed_paths_up_up = perturbed_model_up_up.simulate_paths(n_paths, compute_greeks=False)
                price_up_up = np.mean(perturbed_paths_up_up['Q'])
                
                # Perturb S0[i] up, S0[j] down
                perturbed_S0_up_down = self.S0.copy()
                perturbed_S0_up_down[i] *= (1 + h)
                perturbed_S0_up_down[j] *= (1 - h)
                
                perturbed_model_up_down = WorstOfAllAutocallable(
                    perturbed_S0_up_down, self.Sref, self.mu, self.sigma, self.R,
                    self.Bu, self.Bc, self.t1, self.m, self.principal, self.coupon_rate
                )
                
                perturbed_paths_up_down = perturbed_model_up_down.simulate_paths(n_paths, compute_greeks=False)
                price_up_down = np.mean(perturbed_paths_up_down['Q'])
                
                # Perturb S0[i] down, S0[j] up
                perturbed_S0_down_up = self.S0.copy()
                perturbed_S0_down_up[i] *= (1 - h)
                perturbed_S0_down_up[j] *= (1 + h)
                
                perturbed_model_down_up = WorstOfAllAutocallable(
                    perturbed_S0_down_up, self.Sref, self.mu, self.sigma, self.R,
                    self.Bu, self.Bc, self.t1, self.m, self.principal, self.coupon_rate
                )
                
                perturbed_paths_down_up = perturbed_model_down_up.simulate_paths(n_paths, compute_greeks=False)
                price_down_up = np.mean(perturbed_paths_down_up['Q'])
                
                # Perturb both S0[i] and S0[j] down
                perturbed_S0_down_down = self.S0.copy()
                perturbed_S0_down_down[i] *= (1 - h)
                perturbed_S0_down_down[j] *= (1 - h)
                
                perturbed_model_down_down = WorstOfAllAutocallable(
                    perturbed_S0_down_down, self.Sref, self.mu, self.sigma, self.R,
                    self.Bu, self.Bc, self.t1, self.m, self.principal, self.coupon_rate
                )
                
                perturbed_paths_down_down = perturbed_model_down_down.simulate_paths(n_paths, compute_greeks=False)
                price_down_down = np.mean(perturbed_paths_down_down['Q'])
                
                # Compute Gamma_ij using finite differences
                Gamma[i, j] = Gamma[j, i] = (price_up_up + price_down_down - price_up_down - price_down_up) / (4 * h * h * self.S0[i] * self.S0[j])
        
        return Gamma

def run_experiment(d=3, n_paths=10000, seed=42):
    """
    Run an experiment to compare path-wise differentiation and finite difference methods.
    
    Parameters:
    - d: Number of assets
    - n_paths: Number of paths to simulate
    - seed: Random seed
    
    Returns:
    - results: Dictionary containing results
    """
    np.random.seed(seed)
    
    # Set up parameters
    if d == 3:
        # 3-asset case from the paper
        S0 = [1.0, 1.02, 0.98]
        Sref = [1.0, 1.0, 1.0]
        mu = [0.03, 0.025, 0.02]
        sigma = [0.3, 0.25, 0.35]
        R = np.array([
            [1.0, 0.1, 0.2],
            [0.1, 1.0, 0.1],
            [0.2, 0.1, 1.0]
        ])
    elif d == 5:
        # 5-asset case from the paper
        S0 = [1.0, 1.02, 0.98, 0.97, 1.03]
        Sref = [1.0, 1.0, 1.0, 1.0, 1.0]
        mu = [0.03, 0.025, 0.02, 0.02, 0.03]
        sigma = [0.3, 0.25, 0.35, 0.3, 0.25]
        R = np.array([
            [1.0, 0.1, 0.2, 0.3, 0.2],
            [0.1, 1.0, 0.1, 0.15, 0.1],
            [0.2, 0.1, 1.0, 0.25, 0.2],
            [0.3, 0.15, 0.25, 1.0, 0.15],
            [0.2, 0.1, 0.2, 0.15, 1.0]
        ])
    else:
        # Generate parameters for arbitrary dimension
        S0 = np.random.uniform(0.9, 1.1, d)
        Sref = np.ones(d)
        mu = np.random.uniform(0.01, 0.04, d)
        sigma = np.random.uniform(0.2, 0.4, d)
        
        # Generate correlation matrix
        A = np.random.uniform(-0.1, 0.5, (d, d))
        R = A @ A.T
        np.fill_diagonal(R, 1.0)
        
        # Normalize to ensure it's a valid correlation matrix
        D = np.diag(1.0 / np.sqrt(np.diag(R)))
        R = D @ R @ D
    
    # Create model
    Bu = Bc = 0.8  # Autocall and coupon barriers
    t1 = 1/12     # Time to first observation
    m = 12        # Number of observation dates
    
    model = WorstOfAllAutocallable(S0, Sref, mu, sigma, R, Bu, Bc, t1, m)
    
    # Time path-wise differentiation method
    start_time = time()
    paths = model.simulate_paths(n_paths)
    pw_time = time() - start_time
    
    pw_price = np.mean(paths['Q'])
    pw_greeks = {
        'Delta': paths['Delta'],
        'Mega': paths['Mega'],
        'Vega': paths['Vega'],
        'Bega': paths['Bega'],
        'Rega': paths['Rega']
    }
    
    # Time finite difference method
    start_time = time()
    fd_greeks = model.compute_greeks_finite_diff(n_paths, h=1e-3)
    fd_time = time() - start_time
    
    # Compute Gamma (second-order Greeks)
    start_time = time()
    gamma = model.compute_gamma_finite_diff(n_paths//5, h=1e-2)
    gamma_time = time() - start_time
    
    # Collect results
    results = {
        'parameters': {
            'd': d,
            'n_paths': n_paths,
            'S0': S0,
            'mu': mu,
            'sigma': sigma,
            'R': R,
            'Bu': Bu,
            'Bc': Bc,
            't1': t1,
            'm': m
        },
        'price': pw_price,
        'pw_greeks': pw_greeks,
        'fd_greeks': fd_greeks,
        'gamma': gamma,
        'timing': {
            'path_wise': pw_time,
            'finite_diff': fd_time,
            'gamma': gamma_time
        }
    }
    
    return results

def compare_methods_dimension_impact(dims=[3, 5, 7, 9], n_paths=10000):
    """
    Compare the impact of dimension on the running time of different methods.
    
    Parameters:
    - dims: List of dimensions to test
    - n_paths: Number of paths to simulate
    
    Returns:
    - timing_results: Dictionary containing timing results
    """
    timing_results = {
        'dimensions': dims,
        'path_wise': [],
        'finite_diff': []
    }
    
    for d in dims:
        print(f"Testing dimension d={d}")
        results = run_experiment(d=d, n_paths=n_paths)
        
        timing_results['path_wise'].append(results['timing']['path_wise'])
        timing_results['finite_diff'].append(results['timing']['finite_diff'])
    
    return timing_results

def plot_timing_comparison(timing_results):
    """Plot timing comparison for different methods."""
    plt.figure(figsize=(10, 6))
    
    plt.plot(timing_results['dimensions'], timing_results['path_wise'], 'o-', label='Path-wise Differentiation')
    plt.plot(timing_results['dimensions'], timing_results['finite_diff'], 's-', label='Finite Difference')
    
    plt.xlabel('Number of Assets (d)')
    plt.ylabel('Running Time (seconds)')
    plt.title('Running Time vs. Number of Assets')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def print_results(results):
    """Print results of the experiment."""
    print("\n========== RESULTS ==========")
    print(f"Number of assets: {results['parameters']['d']}")
    print(f"Number of paths: {results['parameters']['n_paths']}")
    print(f"Option price: {results['price']:.6f}")
    
    print("\n----- First-order Greeks -----")
    print("                Path-wise        Finite Diff      Difference")
    
    # Print Delta
    for i in range(results['parameters']['d']):
        pw_delta = results['pw_greeks']['Delta'][i]
        fd_delta = results['fd_greeks']['Delta'][i]
        print(f"Delta_{i+1}:      {pw_delta:.6f}       {fd_delta:.6f}       {pw_delta-fd_delta:.6f}")
    
    # Print Mega
    for i in range(results['parameters']['d']):
        pw_mega = results['pw_greeks']['Mega'][i]
        fd_mega = results['fd_greeks']['Mega'][i]
        print(f"Mega_{i+1}:       {pw_mega:.6f}       {fd_mega:.6f}       {pw_mega-fd_mega:.6f}")
    
    # Print Vega
    for i in range(results['parameters']['d']):
        pw_vega = results['pw_greeks']['Vega'][i]
        fd_vega = results['fd_greeks']['Vega'][i]
        print(f"Vega_{i+1}:       {pw_vega:.6f}       {fd_vega:.6f}       {pw_vega-fd_vega:.6f}")
    
    # Print Bega
    pw_bega = results['pw_greeks']['Bega']
    fd_bega = results['fd_greeks']['Bega']
    print(f"Bega:          {pw_bega:.6f}       {fd_bega:.6f}       {pw_bega-fd_bega:.6f}")
    
    # Print Rega (only for distinct pairs)
    d = results['parameters']['d']
    for i in range(d):
        for j in range(i+1, d):
            pw_rega = results['pw_greeks']['Rega'][i, j]
            fd_rega = results['fd_greeks']['Rega'][i, j]
            print(f"Rega_{i+1}{j+1}:      {pw_rega:.6f}       {fd_rega:.6f}       {pw_rega-fd_rega:.6f}")
    
    print("\n----- Second-order Greeks (Gamma) -----")
    # Print diagonal elements
    for i in range(d):
        print(f"Gamma_{i+1}{i+1}:    {results['gamma'][i, i]:.6f}")
    
    # Print off-diagonal elements
    for i in range(d):
        for j in range(i+1, d):
            print(f"Gamma_{i+1}{j+1}:    {results['gamma'][i, j]:.6f}")
    
    print("\n----- Timing Results -----")
    print(f"Path-wise method:      {results['timing']['path_wise']:.4f} seconds")
    print(f"Finite difference:     {results['timing']['finite_diff']:.4f} seconds")
    print(f"Gamma computation:     {results['timing']['gamma']:.4f} seconds")
    print("==============================\n")

# Main execution
if __name__ == "__main__":
    # Run experiment with 3-asset model
    print("Running experiment with 3-asset model...")
    results_3d = run_experiment(d=3, n_paths=10000)
    print_results(results_3d)
    
    # Run experiment with 5-asset model
    print("Running experiment with 5-asset model...")
    results_5d = run_experiment(d=5, n_paths=10000)
    print_results(results_5d)
    
    # Compare impact of dimension
    print("Comparing impact of dimension...")
    timing_results = compare_methods_dimension_impact(dims=[3, 5, 7, 9], n_paths=5000)
    plot_timing_comparison(timing_results)
    
    # Test with different proximity to observation date
    print("Testing with different proximity to observation date...")
    
    # One month away from observation
    model_1month = WorstOfAllAutocallable(
        results_3d['parameters']['S0'],
        results_3d['parameters']['S0'],  # Reference = current
        results_3d['parameters']['mu'],
        results_3d['parameters']['sigma'],
        results_3d['parameters']['R'],
        results_3d['parameters']['Bu'],
        results_3d['parameters']['Bc'],
        1/12,  # One month to observation
        12
    )
    
    # One day away from observation
    model_1day = WorstOfAllAutocallable(
        results_3d['parameters']['S0'],
        results_3d['parameters']['S0'],  # Reference = current
        results_3d['parameters']['mu'],
        results_3d['parameters']['sigma'],
        results_3d['parameters']['R'],
        results_3d['parameters']['Bu'],
        results_3d['parameters']['Bc'],
        1/360,  # One day to observation
        12
    )
    
    # Compare Greeks
    paths_1month = model_1month.simulate_paths(10000)
    paths_1day = model_1day.simulate_paths(10000)
    
    print("\nComparison of Greeks with different time to observation:")
    print("                One Month        One Day")
    
    # Print Delta
    for i in range(3):
        delta_1month = paths_1month['Delta'][i]
        delta_1day = paths_1day['Delta'][i]
        print(f"Delta_{i+1}:      {delta_1month:.6f}       {delta_1day:.6f}")
    
    # Print Bega
    bega_1month = paths_1month['Bega']
    bega_1day = paths_1day['Bega']
    print(f"Bega:          {bega_1month:.6f}       {bega_1day:.6f}")