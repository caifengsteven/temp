import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.linalg import solve_banded
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class VIXRegimeSwitchingStrategy:
    """
    Implementation of the VIX futures trading strategy using mean reversion with regime switching
    """
    
    def __init__(self, 
                 S0=30,               # Initial VIX level
                 mu1=8.57, mu2=9.0,   # Mean reversion rates under historical measure
                 theta1=17.58, theta2=39.5,  # Long-run means under historical measure
                 mu_tilde1=4.55, mu_tilde2=4.59,  # Mean reversion rates under risk-neutral measure
                 theta_tilde1=18.16, theta_tilde2=40.36,  # Long-run means under risk-neutral measure
                 sigma1=5.33, sigma2=6.42,  # Volatilities
                 q12=0.1, q21=0.5,    # Transition rates between regimes
                 r=0.05,              # Risk-free rate
                 T=66/252,            # Futures expiration time (in years)
                 T_hat=22/252,        # Trading deadline (in years)
                 c=0.01, c_hat=0.01,  # Transaction costs
                 dt=1/252,            # Time step (daily)
                 ds=0.5,              # VIX step size
                 S_max=60):          # Maximum VIX level for grid
        
        # Model parameters
        self.S0 = S0
        self.mu = np.array([mu1, mu2])
        self.theta = np.array([theta1, theta2])
        self.mu_tilde = np.array([mu_tilde1, mu_tilde2])
        self.theta_tilde = np.array([theta_tilde1, theta_tilde2])
        self.sigma = np.array([sigma1, sigma2])
        
        # Ensure Feller condition is satisfied
        for i in range(2):
            if 2 * self.mu[i] * self.theta[i] < self.sigma[i]**2:
                print(f"Warning: Feller condition not satisfied for regime {i+1} under historical measure")
            if 2 * self.mu_tilde[i] * self.theta_tilde[i] < self.sigma[i]**2:
                print(f"Warning: Feller condition not satisfied for regime {i+1} under risk-neutral measure")
        
        # Regime switching parameters
        self.Q = np.array([[-q12, q12], [q21, -q21]])  # Generator matrix
        
        # Trading parameters
        self.r = r
        self.T = T
        self.T_hat = T_hat
        self.c = c
        self.c_hat = c_hat
        
        # Discretization parameters
        self.dt = dt
        self.ds = ds
        self.S_max = S_max
        
        # Grid dimensions
        self.M = int(S_max / ds) + 1  # Number of price steps
        self.N_T = int(T / dt) + 1     # Number of time steps until expiry
        self.N_T_hat = int(T_hat / dt) + 1  # Number of time steps until trading deadline
        
        # Grid arrays
        self.s_grid = np.linspace(0, S_max, self.M)
        self.t_grid = np.linspace(0, T, self.N_T)
        self.t_hat_grid = np.linspace(0, T_hat, self.N_T_hat)
        
        # Compute VIX futures prices
        self.futures_prices = self._compute_futures_prices()
        
        # Compute optimal trading boundaries
        self.optimal_boundaries = self._compute_optimal_boundaries()
    
    def _compute_futures_prices(self):
        """Compute VIX futures prices using finite difference method"""
        # Initialize futures price grid [regime, price, time]
        f = np.zeros((2, self.M, self.N_T))
        
        # Terminal condition: f(T, S, i) = S
        for i in range(2):
            f[i, :, -1] = self.s_grid
        
        # Solve PDE backwards in time
        for n in range(self.N_T-2, -1, -1):
            for i in range(2):
                # CIR model coefficients for the risk-neutral dynamics
                phi = self.mu_tilde[i] * (self.theta_tilde[i] - self.s_grid)
                sigma_sq = self.sigma[i]**2 * self.s_grid
                
                # Set up tridiagonal matrix for implicit scheme
                alpha = np.zeros(self.M)
                beta = np.zeros(self.M)
                gamma = np.zeros(self.M)
                
                for m in range(1, self.M-1):
                    alpha[m] = self.dt * (sigma_sq[m] / (2 * self.ds**2) - phi[m] / (2 * self.ds))
                    beta[m] = 1 + self.dt * (sigma_sq[m] / self.ds**2 + self.Q[i, i])
                    gamma[m] = self.dt * (sigma_sq[m] / (2 * self.ds**2) + phi[m] / (2 * self.ds))
                
                # Handle boundary conditions
                alpha[0] = 0
                beta[0] = 1
                gamma[0] = 0
                
                alpha[-1] = alpha[-2]
                beta[-1] = beta[-2]
                gamma[-1] = 0
                
                # Construct tridiagonal matrix
                A = np.zeros((self.M, self.M))
                
                # Main diagonal
                for m in range(self.M):
                    A[m, m] = beta[m]
                
                # Upper diagonal
                for m in range(self.M-1):
                    A[m, m+1] = -gamma[m]
                
                # Lower diagonal
                for m in range(1, self.M):
                    A[m, m-1] = -alpha[m]
                
                # Set up right-hand side
                rhs = f[i, :, n+1].copy()
                
                # Add regime switching terms
                j = 1 - i  # The other regime
                rhs += self.dt * self.Q[i, j] * f[j, :, n+1]
                
                # Apply boundary conditions
                rhs[0] = 0  # S = 0 boundary
                
                # Solve linear system
                f[i, :, n] = np.linalg.solve(A, rhs)
        
        return f
    
    def _compute_optimal_boundaries(self):
        """Compute optimal trading boundaries using PSOR method"""
        # Initialize value functions for both regimes
        V = np.zeros((2, self.M, self.N_T_hat))  # Value of long position (exit)
        J = np.zeros((2, self.M, self.N_T_hat))  # Value of entry to long-short strategy
        U = np.zeros((2, self.M, self.N_T_hat))  # Value of short position (exit)
        K = np.zeros((2, self.M, self.N_T_hat))  # Value of entry to short-long strategy
        P = np.zeros((2, self.M, self.N_T_hat))  # Value of market entry timing
        
        # Terminal conditions at trading deadline
        for i in range(2):
            V[i, :, -1] = np.maximum(self.futures_prices[i, :, self.N_T_hat-1] - self.c, 0)
            U[i, :, -1] = self.futures_prices[i, :, self.N_T_hat-1] + self.c_hat
            J[i, :, -1] = 0
            K[i, :, -1] = 0
            P[i, :, -1] = 0
        
        # Solve PDEs backwards in time using PSOR method
        for n in range(self.N_T_hat-2, -1, -1):
            # First solve for V and U (exit strategies)
            for i in range(2):
                # CIR model coefficients
                phi = self.mu[i] * (self.theta[i] - self.s_grid)
                sigma_sq = self.sigma[i]**2 * self.s_grid
                
                # Set up coefficients for PSOR
                alpha = np.zeros(self.M)
                beta = np.zeros(self.M)
                gamma = np.zeros(self.M)
                
                for m in range(1, self.M-1):
                    alpha[m] = self.dt * (sigma_sq[m] / (2 * self.ds**2) - phi[m] / (2 * self.ds))
                    beta[m] = 1 + self.dt * (self.r + sigma_sq[m] / self.ds**2 - self.Q[i, i])
                    gamma[m] = self.dt * (sigma_sq[m] / (2 * self.ds**2) + phi[m] / (2 * self.ds))
                
                # Handle boundary conditions
                alpha[0] = 0
                beta[0] = 1
                gamma[0] = 0
                
                alpha[-1] = alpha[-2]
                beta[-1] = beta[-2]
                gamma[-1] = 0
                
                # PSOR for V (exit from long position)
                reward_V = self.futures_prices[i, :, n] - self.c
                V[i, :, n] = self._psor(alpha, beta, gamma, V[i, :, n+1], V[1-i, :, n+1], 
                                       self.Q[i, 1-i], reward_V, 'max')
                
                # PSOR for U (exit from short position)
                reward_U = self.futures_prices[i, :, n] + self.c_hat
                U[i, :, n] = self._psor(alpha, beta, gamma, U[i, :, n+1], U[1-i, :, n+1], 
                                       self.Q[i, 1-i], reward_U, 'min')
            
            # Next solve for J, K, and P (entry strategies)
            for i in range(2):
                # CIR model coefficients
                phi = self.mu[i] * (self.theta[i] - self.s_grid)
                sigma_sq = self.sigma[i]**2 * self.s_grid
                
                # Set up coefficients for PSOR
                alpha = np.zeros(self.M)
                beta = np.zeros(self.M)
                gamma = np.zeros(self.M)
                
                for m in range(1, self.M-1):
                    alpha[m] = self.dt * (sigma_sq[m] / (2 * self.ds**2) - phi[m] / (2 * self.ds))
                    beta[m] = 1 + self.dt * (self.r + sigma_sq[m] / self.ds**2 - self.Q[i, i])
                    gamma[m] = self.dt * (sigma_sq[m] / (2 * self.ds**2) + phi[m] / (2 * self.ds))
                
                # Handle boundary conditions
                alpha[0] = 0
                beta[0] = 1
                gamma[0] = 0
                
                alpha[-1] = alpha[-2]
                beta[-1] = beta[-2]
                gamma[-1] = 0
                
                # PSOR for J (entry to long-short strategy)
                reward_J = np.maximum(V[i, :, n] - (self.futures_prices[i, :, n] + self.c_hat), 0)
                J[i, :, n] = self._psor(alpha, beta, gamma, J[i, :, n+1], J[1-i, :, n+1], 
                                       self.Q[i, 1-i], reward_J, 'max')
                
                # PSOR for K (entry to short-long strategy)
                reward_K = np.maximum((self.futures_prices[i, :, n] - self.c) - U[i, :, n], 0)
                K[i, :, n] = self._psor(alpha, beta, gamma, K[i, :, n+1], K[1-i, :, n+1], 
                                       self.Q[i, 1-i], reward_K, 'max')
                
                # PSOR for P (market entry timing)
                A = np.maximum(V[i, :, n] - (self.futures_prices[i, :, n] + self.c_hat), 0)
                B = np.maximum((self.futures_prices[i, :, n] - self.c) - U[i, :, n], 0)
                reward_P = np.maximum(A, B)
                P[i, :, n] = self._psor(alpha, beta, gamma, P[i, :, n+1], P[1-i, :, n+1], 
                                       self.Q[i, 1-i], reward_P, 'max')
        
        # Determine optimal boundaries
        boundaries = {}
        
        # Find boundaries for long-short strategy (J and V)
        for i in range(2):
            # Entry boundary (J)
            J_boundary = np.zeros(self.N_T_hat)
            for n in range(self.N_T_hat):
                # Find largest S where J > 0
                idx = np.where(J[i, :, n] > 0)[0]
                if len(idx) > 0:
                    J_boundary[n] = self.s_grid[idx[-1]]
                else:
                    J_boundary[n] = np.nan
            
            # Exit boundary (V)
            V_boundary = np.zeros(self.N_T_hat)
            for n in range(self.N_T_hat):
                # Find smallest S where V = futures_price - c
                idx = np.where(np.isclose(V[i, :, n], self.futures_prices[i, :, n] - self.c))[0]
                if len(idx) > 0:
                    V_boundary[n] = self.s_grid[idx[0]]
                else:
                    V_boundary[n] = np.nan
            
            boundaries[f'J{i+1}'] = J_boundary
            boundaries[f'V{i+1}'] = V_boundary
        
        # Find boundaries for short-long strategy (K and U)
        for i in range(2):
            # Entry boundary (K)
            K_boundary = np.zeros(self.N_T_hat)
            for n in range(self.N_T_hat):
                # Find smallest S where K > 0
                idx = np.where(K[i, :, n] > 0)[0]
                if len(idx) > 0:
                    K_boundary[n] = self.s_grid[idx[0]]
                else:
                    K_boundary[n] = np.nan
            
            # Exit boundary (U)
            U_boundary = np.zeros(self.N_T_hat)
            for n in range(self.N_T_hat):
                # Find largest S where U = futures_price + c_hat
                idx = np.where(np.isclose(U[i, :, n], self.futures_prices[i, :, n] + self.c_hat))[0]
                if len(idx) > 0:
                    U_boundary[n] = self.s_grid[idx[-1]]
                else:
                    U_boundary[n] = np.nan
            
            boundaries[f'K{i+1}'] = K_boundary
            boundaries[f'U{i+1}'] = U_boundary
        
        # Find boundaries for market entry (P)
        for i in range(2):
            # Entry boundary for long (P = A)
            P_A_boundary = np.zeros(self.N_T_hat)
            for n in range(self.N_T_hat):
                A = np.maximum(V[i, :, n] - (self.futures_prices[i, :, n] + self.c_hat), 0)
                B = np.maximum((self.futures_prices[i, :, n] - self.c) - U[i, :, n], 0)
                
                # Find largest S where P = A > 0 and A > B
                idx = np.where((P[i, :, n] > 0) & (A > B) & (A > 0))[0]
                if len(idx) > 0:
                    P_A_boundary[n] = self.s_grid[idx[-1]]
                else:
                    P_A_boundary[n] = np.nan
            
            # Entry boundary for short (P = B)
            P_B_boundary = np.zeros(self.N_T_hat)
            for n in range(self.N_T_hat):
                A = np.maximum(V[i, :, n] - (self.futures_prices[i, :, n] + self.c_hat), 0)
                B = np.maximum((self.futures_prices[i, :, n] - self.c) - U[i, :, n], 0)
                
                # Find smallest S where P = B > 0 and B > A
                idx = np.where((P[i, :, n] > 0) & (B > A) & (B > 0))[0]
                if len(idx) > 0:
                    P_B_boundary[n] = self.s_grid[idx[0]]
                else:
                    P_B_boundary[n] = np.nan
            
            boundaries[f'P{i+1}_A'] = P_A_boundary
            boundaries[f'P{i+1}_B'] = P_B_boundary
        
        # Store value functions
        self.value_functions = {
            'V': V,
            'J': J,
            'U': U,
            'K': K,
            'P': P
        }
        
        return boundaries
    
    def _psor(self, alpha, beta, gamma, f_next, f_other_next, q_ij, reward, extremum='max', 
             max_iter=100, tol=1e-6, omega=1.5):
        """
        Projected Successive Over-Relaxation (PSOR) method for solving the variational inequality
        
        Parameters:
        -----------
        alpha, beta, gamma: arrays
            Coefficients for the tridiagonal system
        f_next: array
            Value function at the next time step in the same regime
        f_other_next: array
            Value function at the next time step in the other regime
        q_ij: float
            Transition rate to the other regime
        reward: array
            Reward function (constraint)
        extremum: str
            'max' for max(Lf, reward-f) = 0, 'min' for min(Lf, reward-f) = 0
        max_iter: int
            Maximum number of iterations
        tol: float
            Tolerance for convergence
        omega: float
            Relaxation parameter
            
        Returns:
        --------
        f: array
            Solution of the variational inequality
        """
        M = len(alpha)
        f = f_next.copy()  # Initial guess
        
        for _ in range(max_iter):
            f_old = f.copy()
            
            # Iterate through grid points
            for j in range(1, M-1):
                # Compute right-hand side
                rhs = f_next[j] + self.dt * q_ij * f_other_next[j]
                
                # Compute new value
                f_new = (rhs + alpha[j] * f[j-1] + gamma[j] * f[j+1]) / beta[j]
                
                # Apply relaxation
                f_new = (1 - omega) * f[j] + omega * f_new
                
                # Project onto constraint
                if extremum == 'max':
                    f[j] = max(f_new, reward[j])
                else:  # extremum == 'min'
                    f[j] = min(f_new, reward[j])
            
            # Check convergence
            if np.max(np.abs(f - f_old)) < tol:
                break
        
        # Handle boundary conditions
        f[0] = f[1]  # Reflecting boundary at S = 0
        f[-1] = f[-2]  # Free boundary at S = S_max
        
        return f
    
    def plot_boundaries(self, days_to_plot=None):
        """Plot optimal trading boundaries"""
        if days_to_plot is None:
            days_to_plot = self.N_T_hat
        else:
            days_to_plot = min(days_to_plot, self.N_T_hat)
        
        # Convert from days to years for x-axis
        t_plot = self.t_hat_grid[:days_to_plot] * 252
        
        # Plot long-short strategy boundaries
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 2, 1)
        for i in range(2):
            plt.plot(t_plot, self.optimal_boundaries[f'J{i+1}'][:days_to_plot], label=f'J{i+1}')
            plt.plot(t_plot, self.optimal_boundaries[f'V{i+1}'][:days_to_plot], label=f'V{i+1}')
        plt.xlabel('Time (days)')
        plt.ylabel('VIX level')
        plt.title('Long-Short Strategy Boundaries')
        plt.legend()
        plt.grid(True)
        
        # Plot short-long strategy boundaries
        plt.subplot(2, 2, 2)
        for i in range(2):
            plt.plot(t_plot, self.optimal_boundaries[f'K{i+1}'][:days_to_plot], label=f'K{i+1}')
            plt.plot(t_plot, self.optimal_boundaries[f'U{i+1}'][:days_to_plot], label=f'U{i+1}')
        plt.xlabel('Time (days)')
        plt.ylabel('VIX level')
        plt.title('Short-Long Strategy Boundaries')
        plt.legend()
        plt.grid(True)
        
        # Plot market entry boundaries
        plt.subplot(2, 2, 3)
        for i in range(2):
            plt.plot(t_plot, self.optimal_boundaries[f'P{i+1}_A'][:days_to_plot], label=f'P{i+1}=A{i+1}')
            plt.plot(t_plot, self.optimal_boundaries[f'P{i+1}_B'][:days_to_plot], label=f'P{i+1}=B{i+1}')
        plt.xlabel('Time (days)')
        plt.ylabel('VIX level')
        plt.title('Market Entry Strategy Boundaries')
        plt.legend()
        plt.grid(True)
        
        # Combined plot
        plt.subplot(2, 2, 4)
        for i in range(2):
            plt.plot(t_plot, self.optimal_boundaries[f'J{i+1}'][:days_to_plot], label=f'J{i+1}')
            plt.plot(t_plot, self.optimal_boundaries[f'V{i+1}'][:days_to_plot], label=f'V{i+1}')
            plt.plot(t_plot, self.optimal_boundaries[f'P{i+1}_A'][:days_to_plot], '--', label=f'P{i+1}=A{i+1}')
            plt.plot(t_plot, self.optimal_boundaries[f'P{i+1}_B'][:days_to_plot], '--', label=f'P{i+1}=B{i+1}')
        plt.xlabel('Time (days)')
        plt.ylabel('VIX level')
        plt.title('Combined Strategy Boundaries')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
        
    def plot_value_functions(self, time_idx=0):
        """Plot value functions at a specific time"""
        plt.figure(figsize=(12, 8))
        
        # Plot J and V (long-short strategy)
        plt.subplot(2, 2, 1)
        for i in range(2):
            plt.plot(self.s_grid, self.value_functions['J'][i, :, time_idx], label=f'J{i+1}')
            plt.plot(self.s_grid, self.value_functions['V'][i, :, time_idx], label=f'V{i+1}')
            plt.plot(self.s_grid, self.futures_prices[i, :, time_idx] - self.c, '--', 
                    label=f'f{i+1}-c')
        plt.xlabel('VIX level')
        plt.ylabel('Value')
        plt.title('Long-Short Strategy Value Functions')
        plt.legend()
        plt.grid(True)
        
        # Plot K and U (short-long strategy)
        plt.subplot(2, 2, 2)
        for i in range(2):
            plt.plot(self.s_grid, self.value_functions['K'][i, :, time_idx], label=f'K{i+1}')
            plt.plot(self.s_grid, self.value_functions['U'][i, :, time_idx], label=f'U{i+1}')
            plt.plot(self.s_grid, self.futures_prices[i, :, time_idx] + self.c_hat, '--', 
                    label=f'f{i+1}+c_hat')
        plt.xlabel('VIX level')
        plt.ylabel('Value')
        plt.title('Short-Long Strategy Value Functions')
        plt.legend()
        plt.grid(True)
        
        # Plot P (market entry)
        plt.subplot(2, 2, 3)
        for i in range(2):
            A = np.maximum(self.value_functions['V'][i, :, time_idx] - 
                          (self.futures_prices[i, :, time_idx] + self.c_hat), 0)
            B = np.maximum((self.futures_prices[i, :, time_idx] - self.c) - 
                          self.value_functions['U'][i, :, time_idx], 0)
            
            plt.plot(self.s_grid, self.value_functions['P'][i, :, time_idx], label=f'P{i+1}')
            plt.plot(self.s_grid, A, '--', label=f'A{i+1}')
            plt.plot(self.s_grid, B, '-.', label=f'B{i+1}')
        plt.xlabel('VIX level')
        plt.ylabel('Value')
        plt.title('Market Entry Value Functions')
        plt.legend()
        plt.grid(True)
        
        # Plot optimal timing premium
        plt.subplot(2, 2, 4)
        for i in range(2):
            A = np.maximum(self.value_functions['V'][i, :, time_idx] - 
                          (self.futures_prices[i, :, time_idx] + self.c_hat), 0)
            B = np.maximum((self.futures_prices[i, :, time_idx] - self.c) - 
                          self.value_functions['U'][i, :, time_idx], 0)
            L = self.value_functions['P'][i, :, time_idx] - np.maximum(A, B)
            
            plt.plot(self.s_grid, L, label=f'L{i+1}')
        plt.xlabel('VIX level')
        plt.ylabel('Premium')
        plt.title('Optimal Timing Premium')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def simulate_vix(self, n_paths=1, n_steps=None, seed=None):
        """
        Simulate VIX paths with regime switching
        
        Parameters:
        -----------
        n_paths: int
            Number of paths to simulate
        n_steps: int
            Number of time steps
        seed: int
            Random seed
            
        Returns:
        --------
        paths: numpy.ndarray
            Simulated VIX paths of shape (n_paths, n_steps)
        regimes: numpy.ndarray
            Simulated regimes of shape (n_paths, n_steps)
        """
        if seed is not None:
            np.random.seed(seed)
        
        if n_steps is None:
            n_steps = self.N_T_hat
        
        # Initialize arrays
        paths = np.zeros((n_paths, n_steps))
        regimes = np.zeros((n_paths, n_steps), dtype=int)
        
        # Initial values
        paths[:, 0] = self.S0
        regimes[:, 0] = 0  # Start in regime 1
        
        # Simulate paths
        for i in range(n_paths):
            for j in range(1, n_steps):
                # Current values
                S = paths[i, j-1]
                regime = regimes[i, j-1]
                
                # Regime switching
                p_switch = 1 - np.exp(self.Q[regime, regime] * self.dt)
                if np.random.rand() < p_switch:
                    regime = 1 - regime  # Switch regime
                
                # CIR process simulation
                drift = self.mu[regime] * (self.theta[regime] - S) * self.dt
                volatility = self.sigma[regime] * np.sqrt(S * self.dt)
                dW = np.random.normal(0, 1)
                
                # Update VIX (with floor at 0)
                S_new = S + drift + volatility * dW
                paths[i, j] = max(S_new, 0)
                regimes[i, j] = regime
        
        return paths, regimes
    
    def execute_trading_strategy(self, vix_path, regime_path, strategy='optimal'):
        """
        Execute trading strategy on a simulated VIX path
        
        Parameters:
        -----------
        vix_path: numpy.ndarray
            Simulated VIX path
        regime_path: numpy.ndarray
            Simulated regime path
        strategy: str
            Trading strategy to use:
            - 'optimal': Use optimal boundaries
            - 'longshort': Long-short strategy only
            - 'shortlong': Short-long strategy only
            - 'buyhold': Buy and hold
            
        Returns:
        --------
        pnl: float
            Strategy profit and loss
        trades: list
            List of trades
        """
        n_steps = len(vix_path)
        
        # Initialize variables
        position = 0  # 0: no position, 1: long, -1: short
        entry_price = 0
        entry_time = 0
        pnl = 0
        trades = []
        
        # Loop through time steps
        for t in range(n_steps):
            vix = vix_path[t]
            regime = regime_path[t]
            
            # Find nearest grid point
            s_idx = min(int(vix / self.ds), self.M - 1)
            
            # Current futures price
            if t < self.N_T_hat:
                futures_price = self.futures_prices[regime, s_idx, t]
            else:
                # Use the last available futures price after trading deadline
                futures_price = self.futures_prices[regime, s_idx, -1]
            
            # Trading decision based on strategy
            if t < self.N_T_hat:  # Can only trade before deadline
                if strategy == 'optimal':
                    # No position
                    if position == 0:
                        # Check if VIX is below long entry boundary or above short entry boundary
                        if vix <= self.optimal_boundaries[f'P{regime+1}_A'][t]:
                            # Enter long position
                            position = 1
                            entry_price = futures_price + self.c_hat
                            entry_time = t
                            trades.append(('Enter Long', t, entry_price))
                        elif vix >= self.optimal_boundaries[f'P{regime+1}_B'][t]:
                            # Enter short position
                            position = -1
                            entry_price = futures_price - self.c
                            entry_time = t
                            trades.append(('Enter Short', t, entry_price))
                    
                    # Long position
                    elif position == 1:
                        # Check if VIX is above long exit boundary
                        if vix >= self.optimal_boundaries[f'V{regime+1}'][t]:
                            # Exit long position
                            exit_price = futures_price - self.c
                            pnl += exit_price - entry_price
                            position = 0
                            trades.append(('Exit Long', t, exit_price))
                    
                    # Short position
                    elif position == -1:
                        # Check if VIX is below short exit boundary
                        if vix <= self.optimal_boundaries[f'U{regime+1}'][t]:
                            # Exit short position
                            exit_price = futures_price + self.c_hat
                            pnl += entry_price - exit_price
                            position = 0
                            trades.append(('Exit Short', t, exit_price))
                
                elif strategy == 'longshort':
                    # No position
                    if position == 0:
                        # Check if VIX is below long entry boundary
                        if vix <= self.optimal_boundaries[f'J{regime+1}'][t]:
                            # Enter long position
                            position = 1
                            entry_price = futures_price + self.c_hat
                            entry_time = t
                            trades.append(('Enter Long', t, entry_price))
                    
                    # Long position
                    elif position == 1:
                        # Check if VIX is above long exit boundary
                        if vix >= self.optimal_boundaries[f'V{regime+1}'][t]:
                            # Exit long position
                            exit_price = futures_price - self.c
                            pnl += exit_price - entry_price
                            position = 0
                            trades.append(('Exit Long', t, exit_price))
                
                elif strategy == 'shortlong':
                    # No position
                    if position == 0:
                        # Check if VIX is above short entry boundary
                        if vix >= self.optimal_boundaries[f'K{regime+1}'][t]:
                            # Enter short position
                            position = -1
                            entry_price = futures_price - self.c
                            entry_time = t
                            trades.append(('Enter Short', t, entry_price))
                    
                    # Short position
                    elif position == -1:
                        # Check if VIX is below short exit boundary
                        if vix <= self.optimal_boundaries[f'U{regime+1}'][t]:
                            # Exit short position
                            exit_price = futures_price + self.c_hat
                            pnl += entry_price - exit_price
                            position = 0
                            trades.append(('Exit Short', t, exit_price))
                
                elif strategy == 'buyhold':
                    # Enter at t=0, exit at t=T_hat
                    if t == 0:
                        # Enter long position
                        position = 1
                        entry_price = futures_price + self.c_hat
                        entry_time = t
                        trades.append(('Enter Long', t, entry_price))
                    elif t == self.N_T_hat - 1 and position == 1:
                        # Exit long position
                        exit_price = futures_price - self.c
                        pnl += exit_price - entry_price
                        position = 0
                        trades.append(('Exit Long', t, exit_price))
        
        # Close any remaining position at the end
        if position != 0:
            final_price = vix_path[-1]  # At expiry, futures price equals spot VIX
            
            if position == 1:
                exit_price = final_price - self.c
                pnl += exit_price - entry_price
                trades.append(('Exit Long (Final)', n_steps-1, exit_price))
            else:  # position == -1
                exit_price = final_price + self.c_hat
                pnl += entry_price - exit_price
                trades.append(('Exit Short (Final)', n_steps-1, exit_price))
        
        return pnl, trades
    
    def backtest_strategy(self, n_paths=100, n_steps=None, strategies=None, seed=None):
        """
        Backtest trading strategies on simulated VIX paths
        
        Parameters:
        -----------
        n_paths: int
            Number of paths to simulate
        n_steps: int
            Number of time steps
        strategies: list
            List of strategies to test
        seed: int
            Random seed
            
        Returns:
        --------
        results: pandas.DataFrame
            Backtest results
        """
        if strategies is None:
            strategies = ['optimal', 'longshort', 'shortlong', 'buyhold']
        
        # Simulate VIX paths
        vix_paths, regime_paths = self.simulate_vix(n_paths=n_paths, n_steps=n_steps, seed=seed)
        
        # Initialize results
        results = {
            'strategy': [],
            'pnl': [],
            'num_trades': [],
            'path_id': []
        }
        
        # Execute strategies on each path
        for path_id in tqdm(range(n_paths), desc="Backtesting strategies"):
            vix_path = vix_paths[path_id]
            regime_path = regime_paths[path_id]
            
            for strategy in strategies:
                pnl, trades = self.execute_trading_strategy(vix_path, regime_path, strategy)
                
                results['strategy'].append(strategy)
                results['pnl'].append(pnl)
                results['num_trades'].append(len(trades) // 2)  # Count round trips
                results['path_id'].append(path_id)
        
        # Convert to DataFrame
        results_df = pd.DataFrame(results)
        
        # Calculate summary statistics
        summary = results_df.groupby('strategy').agg({
            'pnl': ['mean', 'std', 'min', 'max'],
            'num_trades': ['mean', 'min', 'max']
        })
        
        # Calculate Sharpe ratio
        summary[('pnl', 'sharpe')] = summary[('pnl', 'mean')] / summary[('pnl', 'std')]
        
        # Add win rate
        for strategy in strategies:
            win_rate = (results_df[results_df['strategy'] == strategy]['pnl'] > 0).mean()
            summary.loc[strategy, ('pnl', 'win_rate')] = win_rate
        
        print("Strategy Backtest Results:")
        print(summary)
        
        # Plot PnL distribution
        plt.figure(figsize=(12, 6))
        
        for i, strategy in enumerate(strategies):
            plt.subplot(1, len(strategies), i+1)
            plt.hist(results_df[results_df['strategy'] == strategy]['pnl'], bins=20, alpha=0.7)
            plt.axvline(x=0, color='r', linestyle='--')
            plt.title(f'{strategy} PnL Distribution')
            plt.xlabel('PnL')
            plt.ylabel('Frequency')
        
        plt.tight_layout()
        plt.show()
        
        return results_df, summary
    
    def plot_simulated_paths(self, n_paths=5, n_steps=None, seed=None):
        """Plot simulated VIX paths with trading signals"""
        if n_steps is None:
            n_steps = self.N_T_hat
        
        # Simulate paths
        vix_paths, regime_paths = self.simulate_vix(n_paths=n_paths, n_steps=n_steps, seed=seed)
        
        # Plot each path with trading signals
        for path_id in range(n_paths):
            vix_path = vix_paths[path_id]
            regime_path = regime_paths[path_id]
            
            # Execute trading strategy
            _, trades = self.execute_trading_strategy(vix_path, regime_path, 'optimal')
            
            # Plot path
            plt.figure(figsize=(12, 6))
            
            # Plot VIX path
            plt.plot(vix_path, label='VIX')
            
            # Plot regime changes
            regime_changes = np.where(np.diff(regime_path) != 0)[0]
            for t in regime_changes:
                plt.axvline(x=t, color='gray', linestyle='--', alpha=0.5)
            
            # Plot trades
            for trade in trades:
                trade_type, time, price = trade
                
                if 'Enter Long' in trade_type:
                    plt.scatter(time, vix_path[time], color='green', marker='^', s=100)
                elif 'Exit Long' in trade_type:
                    plt.scatter(time, vix_path[time], color='red', marker='v', s=100)
                elif 'Enter Short' in trade_type:
                    plt.scatter(time, vix_path[time], color='red', marker='v', s=100)
                elif 'Exit Short' in trade_type:
                    plt.scatter(time, vix_path[time], color='green', marker='^', s=100)
            
            # Add regime indicators
            plt.fill_between(range(n_steps), 0, vix_path, 
                           where=regime_path == 0, color='blue', alpha=0.1, label='Regime 1')
            plt.fill_between(range(n_steps), 0, vix_path, 
                           where=regime_path == 1, color='red', alpha=0.1, label='Regime 2')
            
            # Add optimal boundaries
            t = np.arange(min(n_steps, self.N_T_hat))
            for i in range(2):
                plt.plot(t, self.optimal_boundaries[f'J{i+1}'][:len(t)], '--', alpha=0.5, label=f'J{i+1}')
                plt.plot(t, self.optimal_boundaries[f'V{i+1}'][:len(t)], '--', alpha=0.5, label=f'V{i+1}')
                plt.plot(t, self.optimal_boundaries[f'K{i+1}'][:len(t)], '--', alpha=0.5, label=f'K{i+1}')
                plt.plot(t, self.optimal_boundaries[f'U{i+1}'][:len(t)], '--', alpha=0.5, label=f'U{i+1}')
            
            plt.xlabel('Time (days)')
            plt.ylabel('VIX')
            plt.title(f'Simulated VIX Path {path_id+1} with Trading Signals')
            plt.legend()
            plt.grid(True)
            plt.show()
            
            # Print trades
            print(f"Trades for Path {path_id+1}:")
            for i, trade in enumerate(trades):
                trade_type, time, price = trade
                print(f"{i+1}. {trade_type} at time {time}, price {price:.2f}")
            print()

# Example usage
if __name__ == "__main__":
    # Create strategy instance with smaller grid to speed up computation
    strategy = VIXRegimeSwitchingStrategy(S_max=60, ds=0.5)
    
    # Plot optimal boundaries
    strategy.plot_boundaries(days_to_plot=22)
    
    # Plot value functions
    strategy.plot_value_functions(time_idx=0)
    
    # Plot simulated paths with trading signals
    strategy.plot_simulated_paths(n_paths=2, seed=42)
    
    # Backtest strategy
    results_df, summary = strategy.backtest_strategy(n_paths=50, seed=42)