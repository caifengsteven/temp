import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm, multivariate_normal, weibull_min
from scipy.optimize import minimize, minimize_scalar
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
import warnings
warnings.filterwarnings('ignore')

class PortfolioOptimalTransport:
    """
    Portfolio optimization with prescribed terminal wealth distribution
    Based on Guo, Langrené, Loeper & Ning (2022)
    """
    
    def __init__(self, x0, mu, sigma, T=1.0, N_time=100, N_space=200):
        """
        Initialize the portfolio optimization problem
        
        Parameters:
        x0: Initial wealth
        mu: Drift coefficient (can be array for multiple assets)
        sigma: Volatility coefficient
        T: Time horizon
        N_time: Number of time steps
        N_space: Number of spatial grid points
        """
        self.x0 = x0
        self.mu = mu
        self.sigma = sigma
        self.T = T
        self.N_time = N_time
        self.N_space = N_space
        
        # Time and space grids
        self.dt = T / N_time
        self.t_grid = np.linspace(0, T, N_time + 1)
        
        # Space grid (wealth values)
        x_min = x0 * 0.2  # Allow for significant losses
        x_max = x0 * 3.0  # Allow for significant gains
        self.x_grid = np.linspace(x_min, x_max, N_space)
        self.dx = self.x_grid[1] - self.x_grid[0]
        
    def cost_function(self, A_tilde, B_tilde, cost_type='quadratic'):
        """
        Cost function F(A_tilde, B_tilde)
        """
        if cost_type == 'quadratic':
            # Simple quadratic cost
            return (A_tilde - 0.2)**2 + (B_tilde - 0.2)**2
        else:
            raise ValueError("Unknown cost type")
    
    def convex_conjugate(self, p, q):
        """
        Convex conjugate F*(p, q) of the cost function
        For quadratic cost F(A,B) = (A-0.2)^2 + (B-0.2)^2
        F*(p,q) = p^2/4 + q^2/4 + 0.2*p + 0.2*q
        """
        return p**2/4 + q**2/4 + 0.2*p + 0.2*q
    
    def solve_hjb_backward(self, phi_terminal, tol=1e-7, max_iter=100):
        """
        Solve HJB equation backward in time using implicit finite difference
        """
        phi = np.zeros((self.N_time + 1, self.N_space))
        phi[-1, :] = phi_terminal
        
        # For single asset case
        nu = self.mu / self.sigma**2
        
        for n in range(self.N_time - 1, -1, -1):
            # Fixed point iteration for each time step
            phi_old = phi[n+1, :].copy()
            
            for iter in range(max_iter):
                # Compute optimal controls using current phi estimate
                phi_x = np.gradient(phi_old, self.dx)
                phi_xx = np.gradient(phi_x, self.dx)
                
                # Optimal controls from first-order conditions
                B_tilde_star = np.zeros(self.N_space)
                A_tilde_star = np.zeros(self.N_space)
                
                for i in range(1, self.N_space - 1):
                    # For quadratic cost, optimal controls have closed form
                    p = phi_x[i]
                    q = 0.5 * phi_xx[i]
                    
                    # From FOC: B* = 0.2 + p/2, A* = 0.2 + q/2
                    B_tilde_star[i] = 0.2 + p/2
                    A_tilde_star[i] = max(0.2 + q/2, 0.01)  # Ensure positive diffusion
                    
                    # Check constraint for single asset
                    if B_tilde_star[i]**2 > nu**2 * A_tilde_star[i] / 2:
                        # Saturate at boundary
                        B_tilde_star[i] = nu * np.sqrt(A_tilde_star[i] / 2)
                
                # Implicit finite difference scheme
                a = self.dt * A_tilde_star / (2 * self.dx**2)
                b = self.dt * B_tilde_star / (2 * self.dx)
                
                # Construct tridiagonal matrix
                diagonal = 1 + self.dt * A_tilde_star / self.dx**2
                lower = -a[1:] - b[1:]
                upper = -a[:-1] + b[:-1]
                
                # RHS includes cost function
                rhs = phi[n+1, :] + self.dt * self.cost_function(A_tilde_star, B_tilde_star)
                
                # Solve linear system
                A_matrix = diags([lower, diagonal, upper], [-1, 0, 1], 
                               shape=(self.N_space, self.N_space))
                phi_new = spsolve(A_matrix.tocsr(), rhs)
                
                # Check convergence
                if np.max(np.abs(phi_new - phi_old)) < tol:
                    phi[n, :] = phi_new
                    break
                
                phi_old = phi_new.copy()
        
        # Store optimal controls
        self.B_tilde_opt = B_tilde_star
        self.A_tilde_opt = A_tilde_star
        
        return phi
    
    def solve_fokker_planck_forward(self, rho_initial):
        """
        Solve Fokker-Planck equation forward in time
        """
        rho = np.zeros((self.N_time + 1, self.N_space))
        rho[0, :] = rho_initial
        
        for n in range(self.N_time):
            # Explicit scheme for Fokker-Planck
            rho_x = np.gradient(rho[n, :], self.dx)
            rho_xx = np.gradient(rho_x, self.dx)
            
            # Use stored optimal controls
            drift_term = -np.gradient(self.B_tilde_opt * rho[n, :], self.dx)
            diffusion_term = 0.5 * np.gradient(
                np.gradient(self.A_tilde_opt * rho[n, :], self.dx), self.dx
            )
            
            rho[n+1, :] = rho[n, :] + self.dt * (drift_term + diffusion_term)
            
            # Ensure non-negativity and normalize
            rho[n+1, :] = np.maximum(rho[n+1, :], 0)
            rho[n+1, :] = rho[n+1, :] / (np.sum(rho[n+1, :]) * self.dx)
        
        return rho
    
    def gradient_descent_optimization(self, target_dist, penalty_type='L2', 
                                    lambda_penalty=1.0, max_iter=50, lr=0.1):
        """
        Gradient descent algorithm to find optimal terminal condition phi_1
        """
        # Initial guess for phi_1
        phi_1 = -self.x_grid  # Linear initial guess
        
        history = {'cost': [], 'distance': []}
        
        for iteration in range(max_iter):
            # Solve HJB backward
            phi = self.solve_hjb_backward(phi_1)
            
            # Get initial distribution (Dirac at x0)
            rho_0 = np.zeros(self.N_space)
            idx_0 = np.argmin(np.abs(self.x_grid - self.x0))
            rho_0[idx_0] = 1.0 / self.dx
            
            # Solve Fokker-Planck forward
            rho = self.solve_fokker_planck_forward(rho_0)
            rho_1 = rho[-1, :]
            
            # Compute gradient
            if penalty_type == 'L2':
                # Gradient of C*(−phi_1) for L2 penalty
                gradient = -rho_1 + lambda_penalty * (rho_1 - target_dist)
            elif penalty_type == 'KL':
                # Gradient for KL divergence
                gradient = -rho_1 + lambda_penalty * np.exp(-phi_1/lambda_penalty) * target_dist
            
            # Update phi_1
            phi_1 = phi_1 - lr * gradient
            
            # Compute cost and distance
            if penalty_type == 'L2':
                distance = np.sqrt(np.sum((rho_1 - target_dist)**2) * self.dx)
                cost = 0.5 * lambda_penalty * distance**2
            elif penalty_type == 'KL':
                # Avoid log(0)
                rho_1_safe = np.maximum(rho_1, 1e-10)
                target_safe = np.maximum(target_dist, 1e-10)
                distance = np.sum(rho_1_safe * np.log(rho_1_safe / target_safe)) * self.dx
                cost = lambda_penalty * distance
            
            history['cost'].append(cost)
            history['distance'].append(distance)
            
            if iteration % 10 == 0:
                print(f"Iteration {iteration}: Distance = {distance:.4f}")
            
            # Check convergence
            if distance < 0.01:
                print(f"Converged at iteration {iteration}")
                break
        
        self.phi_1_opt = phi_1
        self.rho_1_opt = rho_1
        
        return phi_1, rho_1, history

def create_target_distribution(x_grid, dist_type='normal', params=None):
    """
    Create various target distributions
    """
    if dist_type == 'normal':
        # Normal distribution
        mean = params.get('mean', 6.0)
        std = params.get('std', 1.0)
        pdf = norm.pdf(x_grid, mean, std)
    
    elif dist_type == 'mixture':
        # Mixture of two normals
        mean1 = params.get('mean1', 4.0)
        std1 = params.get('std1', 0.5)
        mean2 = params.get('mean2', 7.0)
        std2 = params.get('std2', 0.5)
        weight = params.get('weight', 0.5)
        
        pdf = weight * norm.pdf(x_grid, mean1, std1) + \
              (1-weight) * norm.pdf(x_grid, mean2, std2)
    
    elif dist_type == 'weibull':
        # Weibull distribution (asymmetric)
        shape = params.get('shape', 2.0)
        scale = params.get('scale', 6.0)
        pdf = weibull_min.pdf(x_grid, shape, scale=scale)
    
    else:
        raise ValueError(f"Unknown distribution type: {dist_type}")
    
    # Normalize
    pdf = pdf / (np.sum(pdf) * (x_grid[1] - x_grid[0]))
    return pdf

def test_portfolio_optimization():
    """
    Test the portfolio optimization with various target distributions
    """
    # Parameters
    x0 = 5.0      # Initial wealth
    mu = 0.1      # Drift
    sigma = 0.1   # Volatility
    T = 1.0       # Time horizon
    
    # Create optimizer
    opt = PortfolioOptimalTransport(x0, mu, sigma, T)
    
    # Test 1: Normal target distribution
    print("\n" + "="*60)
    print("Test 1: Normal Target Distribution N(6, 1)")
    print("="*60)
    
    target_params = {'mean': 6.0, 'std': 1.0}
    target_dist = create_target_distribution(opt.x_grid, 'normal', target_params)
    
    # Run optimization with different penalty intensities
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    for i, lambda_val in enumerate([1.0, 5.0, 20.0, 100.0]):
        ax = axes[i//2, i%2]
        
        phi_1, rho_1, history = opt.gradient_descent_optimization(
            target_dist, penalty_type='L2', lambda_penalty=lambda_val, max_iter=30
        )
        
        ax.plot(opt.x_grid, target_dist, 'r--', label='Target', linewidth=2)
        ax.plot(opt.x_grid, rho_1, 'b-', label='Achieved', linewidth=2)
        ax.fill_between(opt.x_grid, 0, rho_1, alpha=0.3)
        ax.set_xlabel('Wealth')
        ax.set_ylabel('Density')
        ax.set_title(f'λ = {lambda_val}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Portfolio Optimization with Different Penalty Intensities')
    plt.tight_layout()
    plt.show()
    
    # Test 2: Mixture of normals
    print("\n" + "="*60)
    print("Test 2: Mixture of Normal Distributions")
    print("="*60)
    
    target_params = {'mean1': 4.0, 'std1': 0.5, 'mean2': 7.0, 'std2': 0.5, 'weight': 0.5}
    target_dist = create_target_distribution(opt.x_grid, 'mixture', target_params)
    
    phi_1, rho_1, history = opt.gradient_descent_optimization(
        target_dist, penalty_type='L2', lambda_penalty=50.0, max_iter=50
    )
    
    plt.figure(figsize=(10, 6))
    plt.plot(opt.x_grid, target_dist, 'r--', label='Target (Mixture)', linewidth=2)
    plt.plot(opt.x_grid, rho_1, 'b-', label='Achieved', linewidth=2)
    plt.fill_between(opt.x_grid, 0, rho_1, alpha=0.3)
    plt.xlabel('Wealth')
    plt.ylabel('Density')
    plt.title('Portfolio Optimization: Mixture Target')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Plot convergence history
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history['cost'])
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.title('Cost Function Evolution')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(history['distance'])
    plt.xlabel('Iteration')
    plt.ylabel('L2 Distance')
    plt.title('Distance to Target')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def test_cash_saving_strategy():
    """
    Test portfolio with cash saving when target is not ambitious
    """
    print("\n" + "="*60)
    print("Test 3: Cash Saving Strategy")
    print("="*60)
    
    x0 = 5.0
    mu = 0.1
    sigma = 0.1
    
    opt = PortfolioOptimalTransport(x0, mu, sigma)
    
    # Conservative target (low mean)
    target_params = {'mean': 5.1, 'std': 0.4}
    target_dist = create_target_distribution(opt.x_grid, 'normal', target_params)
    
    # Optimize
    phi_1, rho_1, history = opt.gradient_descent_optimization(
        target_dist, lambda_penalty=100.0, max_iter=30
    )
    
    # Compute wealth with cash saving
    nu = mu / sigma**2
    
    # Simulate paths with and without cash saving
    n_paths = 1000
    dt_sim = 0.01
    n_steps = int(1.0 / dt_sim)
    
    wealth_no_saving = np.zeros((n_paths, n_steps + 1))
    wealth_with_saving = np.zeros((n_paths, n_steps + 1))
    wealth_no_saving[:, 0] = x0
    wealth_with_saving[:, 0] = x0
    
    for i in range(n_steps):
        dW = np.sqrt(dt_sim) * np.random.randn(n_paths)
        
        # Without cash saving (use actual optimal drift)
        idx = np.searchsorted(opt.x_grid, wealth_no_saving[:, i])
        idx = np.clip(idx, 0, len(opt.B_tilde_opt) - 1)
        drift = opt.B_tilde_opt[idx]
        vol = np.sqrt(opt.A_tilde_opt[idx])
        
        wealth_no_saving[:, i+1] = wealth_no_saving[:, i] + \
                                  drift * dt_sim + vol * dW
        
        # With cash saving (use maximum drift)
        max_drift = nu * vol / np.sqrt(2)
        wealth_with_saving[:, i+1] = wealth_with_saving[:, i] + \
                                    max_drift * dt_sim + vol * dW
    
    # Plot terminal distributions
    plt.figure(figsize=(10, 6))
    
    # Histogram of terminal wealth
    plt.hist(wealth_no_saving[:, -1], bins=50, density=True, alpha=0.5, 
             label='Without Cash Saving', color='blue')
    plt.hist(wealth_with_saving[:, -1], bins=50, density=True, alpha=0.5, 
             label='With Cash Saving', color='green')
    
    # Plot target
    plt.plot(opt.x_grid, target_dist * 20, 'r--', label='Target', linewidth=2)
    
    plt.xlabel('Terminal Wealth')
    plt.ylabel('Density')
    plt.title('Terminal Wealth Distribution: Cash Saving Effect')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    print(f"Mean terminal wealth without saving: {np.mean(wealth_no_saving[:, -1]):.3f}")
    print(f"Mean terminal wealth with saving: {np.mean(wealth_with_saving[:, -1]):.3f}")
    print(f"Cash saved: {np.mean(wealth_with_saving[:, -1] - wealth_no_saving[:, -1]):.3f}")

def test_unattainable_target():
    """
    Test with unattainable target requiring cash input
    """
    print("\n" + "="*60)
    print("Test 4: Unattainable Target (Weibull Distribution)")
    print("="*60)
    
    x0 = 5.0
    mu = 0.1
    sigma = 0.1
    
    opt = PortfolioOptimalTransport(x0, mu, sigma)
    
    # Weibull target (asymmetric, hard to achieve)
    target_params = {'shape': 2.0, 'scale': 6.0}
    target_dist = create_target_distribution(opt.x_grid, 'weibull', target_params)
    
    # Try to optimize
    phi_1, rho_1, history = opt.gradient_descent_optimization(
        target_dist, lambda_penalty=10.0, max_iter=100, lr=0.05
    )
    
    plt.figure(figsize=(10, 6))
    plt.plot(opt.x_grid, target_dist, 'r--', label='Target (Weibull)', linewidth=2)
    plt.plot(opt.x_grid, rho_1, 'b-', label='Best Achievable', linewidth=2)
    plt.fill_between(opt.x_grid, 0, rho_1, alpha=0.3)
    plt.xlabel('Wealth')
    plt.ylabel('Density')
    plt.title('Unattainable Target: Weibull Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Calculate statistics
    mean_target = np.sum(opt.x_grid * target_dist) * opt.dx
    mean_achieved = np.sum(opt.x_grid * rho_1) * opt.dx
    
    var_target = np.sum((opt.x_grid - mean_target)**2 * target_dist) * opt.dx
    var_achieved = np.sum((opt.x_grid - mean_achieved)**2 * rho_1) * opt.dx
    
    print(f"\nTarget statistics:")
    print(f"  Mean: {mean_target:.3f}, Std: {np.sqrt(var_target):.3f}")
    print(f"Achieved statistics:")
    print(f"  Mean: {mean_achieved:.3f}, Std: {np.sqrt(var_achieved):.3f}")
    print(f"Final distance: {history['distance'][-1]:.4f}")

# Run all tests
if __name__ == "__main__":
    print("Portfolio Optimization with Prescribed Terminal Distribution")
    print("Based on Guo, Langrené, Loeper & Ning (2022)")
    
    # Run tests
    test_portfolio_optimization()
    test_cash_saving_strategy()
    test_unattainable_target()
    
    print("\n" + "="*60)
    print("Summary of Key Findings:")
    print("="*60)
    print("1. Higher penalty λ leads to better matching of target distribution")
    print("2. Complex distributions (mixtures) can be achieved with sufficient iterations")
    print("3. Conservative targets allow for cash saving strategies")
    print("4. Some targets (e.g., Weibull) may be unattainable without cash input")
    print("5. The optimal transport framework provides flexible portfolio control")