import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp, quad
from scipy.optimize import root_scalar, minimize
import pandas as pd
from tqdm import tqdm

# Set random seed for reproducibility
np.random.seed(42)

class PortfolioLiquidationGame:
    def __init__(self, T=1.0, eta=5.0, kappa=10.0, lambda_=5.0, delta=0.0):
        """
        Initialize the portfolio liquidation game.
        
        Parameters:
        -----------
        T : float
            Trading horizon
        eta : float
            Instantaneous market impact parameter
        kappa : float
            Permanent market impact parameter
        lambda_ : float
            Risk aversion parameter
        delta : float
            Player's impact on aggregate trading (0 for mean-field game, 1/N for N-player game)
        """
        self.T = T
        self.eta = eta if callable(eta) else lambda t: eta
        self.kappa = kappa if callable(kappa) else lambda t: kappa
        self.lambda_ = lambda_ if callable(lambda_) else lambda t: lambda_
        self.delta = delta
        
    def solve_riccati_equation(self, t_eval=None):
        """
        Solve the Riccati equation for A and B.
        
        Parameters:
        -----------
        t_eval : array-like
            Time points at which to evaluate the solution
            
        Returns:
        --------
        A : array-like
            Solution to the first equation of the Riccati system
        B : array-like
            Solution to the second equation of the Riccati system
        """
        # For constant parameters, we can use the analytical solution
        if callable(self.eta) or callable(self.lambda_) or callable(self.kappa):
            # Implement numerical solver for the general case
            # Not implemented in this example
            pass
        else:
            # Analytical solution for constant parameters
            if t_eval is None:
                t_eval = np.linspace(0, self.T, 1000)
            
            # Calculate A(t)
            lambda_eta = self.lambda_(0) * self.eta(0)
            A = np.zeros_like(t_eval)
            for i, t in enumerate(t_eval):
                remaining_time = self.T - t
                A[i] = np.sqrt(lambda_eta) * (1 + np.exp(-2 * np.sqrt(lambda_eta) * remaining_time)) / (1 - np.exp(-2 * np.sqrt(lambda_eta) * remaining_time))
            
            # Calculate B(t) for a given A(t)
            # For simplicity, we'll set B(t) = 0 for this example
            B = np.zeros_like(t_eval)
            
            return A, B
    
    def compute_psi(self, mu, t_eval=None):
        """
        Compute the function ψ^{δ,T}_μ(t) as defined in (2.13).
        
        Parameters:
        -----------
        mu : callable
            Aggregate trading rate function
        t_eval : array-like
            Time points at which to evaluate the function
            
        Returns:
        --------
        psi : array-like
            Values of the function ψ^{δ,T}_μ at time points t_eval
        """
        if t_eval is None:
            t_eval = np.linspace(0, self.T, 1000)
        
        # Get A(t) solution
        A, _ = self.solve_riccati_equation(t_eval)
        
        # Compute α^δ_t as defined in Proposition 2.7
        alpha_delta = np.zeros_like(t_eval)
        for i, t in enumerate(t_eval):
            alpha_delta[i] = (A[i] - self.delta * self.kappa(0)) * np.exp(-self.integrate_A_over_eta(0, t))
        
        # Compute ψ^{δ,T}_μ(t)
        psi = np.zeros_like(t_eval)
        for i, t in enumerate(t_eval):
            def integrand(s):
                return np.exp(-self.integrate_A_over_eta(0, s)) * self.kappa(0) * mu(s)
            
            psi[i] = (1 / alpha_delta[i]) * quad(integrand, t, self.T)[0]
        
        return psi
    
    def compute_phi(self, mu, t_eval=None):
        """
        Compute the function φ_μ(t) as defined in (2.23).
        
        Parameters:
        -----------
        mu : callable
            Aggregate trading rate function
        t_eval : array-like
            Time points at which to evaluate the function
            
        Returns:
        --------
        phi : array-like
            Values of the function φ_μ at time points t_eval
        """
        if t_eval is None:
            t_eval = np.linspace(0, self.T, 1000)
        
        # Compute the h^δ function as needed for φ_μ
        h_delta = self.compute_h_delta(t_eval)
        
        # Compute φ_μ(t)
        phi = np.zeros_like(t_eval)
        for i, t in enumerate(t_eval):
            def integrand(u):
                return self.kappa(0) * mu(u) * h_delta[np.argmin(np.abs(t_eval - u))]
            
            phi[i] = quad(integrand, 0, t)[0]
        
        return phi
    
    def compute_h_delta(self, t_eval=None):
        """
        Compute the h^δ function used in the definition of φ_μ.
        
        Parameters:
        -----------
        t_eval : array-like
            Time points at which to evaluate the function
            
        Returns:
        --------
        h_delta : array-like
            Values of h^δ at time points t_eval
        """
        if t_eval is None:
            t_eval = np.linspace(0, self.T, 1000)
        
        # Get A(t) solution
        A, _ = self.solve_riccati_equation(t_eval)
        
        # Compute h^δ_t
        h_delta = np.zeros_like(t_eval)
        for i, t in enumerate(t_eval):
            if t == 0:
                h_delta[i] = 0
            else:
                def integrand(s):
                    idx = np.argmin(np.abs(t_eval - s))
                    return (1 / self.eta(s)) * np.exp(self.integrate_A_minus_delta_kappa_over_eta(0, s))
                
                h_delta[i] = np.exp(-self.integrate_A_over_eta(0, t)) * quad(integrand, 0, t)[0]
        
        return h_delta
    
    def integrate_A_over_eta(self, t_start, t_end):
        """
        Compute the integral of A(r)/η(r) from t_start to t_end.
        
        Parameters:
        -----------
        t_start : float
            Start of integration interval
        t_end : float
            End of integration interval
            
        Returns:
        --------
        integral : float
            Value of the integral
        """
        # For constant parameters, we can use a simpler formula
        if not callable(self.eta):
            lambda_eta = self.lambda_(0) * self.eta(0)
            return (1 / self.eta(0)) * np.sqrt(lambda_eta) * np.log(
                (np.exp(np.sqrt(lambda_eta) * (self.T - t_start)) + np.exp(-np.sqrt(lambda_eta) * (self.T - t_start))) /
                (np.exp(np.sqrt(lambda_eta) * (self.T - t_end)) + np.exp(-np.sqrt(lambda_eta) * (self.T - t_end)))
            )
        else:
            # Numerical integration for general case
            t_points = np.linspace(t_start, t_end, 100)
            A, _ = self.solve_riccati_equation(t_points)
            return np.trapz([A[i] / self.eta(t) for i, t in enumerate(t_points)], t_points)
    
    def integrate_A_minus_delta_kappa_over_eta(self, t_start, t_end):
        """
        Compute the integral of (A(r) - δκ)/η(r) from t_start to t_end.
        
        Parameters:
        -----------
        t_start : float
            Start of integration interval
        t_end : float
            End of integration interval
            
        Returns:
        --------
        integral : float
            Value of the integral
        """
        # For constant parameters, use simplified formula
        if not callable(self.eta) and not callable(self.kappa):
            lambda_eta = self.lambda_(0) * self.eta(0)
            base_integral = self.integrate_A_over_eta(t_start, t_end)
            delta_kappa_term = (self.delta * self.kappa(0) / self.eta(0)) * (t_end - t_start)
            return base_integral - delta_kappa_term
        else:
            # Numerical integration for general case
            t_points = np.linspace(t_start, t_end, 100)
            A, _ = self.solve_riccati_equation(t_points)
            return np.trapz([(A[i] - self.delta * self.kappa(t)) / self.eta(t) 
                           for i, t in enumerate(t_points)], t_points)
    
    def find_entry_time(self, x, mu):
        """
        Find the optimal entry time for a buyer with initial position x.
        
        Parameters:
        -----------
        x : float
            Initial position (negative for buyers)
        mu : callable
            Aggregate trading rate function
            
        Returns:
        --------
        entry_time : float
            Optimal entry time
        """
        if x >= 0:  # Sellers enter immediately
            return 0.0
        
        # Compute ψ_μ function
        t_eval = np.linspace(0, self.T, 1000)
        psi = self.compute_psi(mu, t_eval)
        
        # Find the time t such that ψ_μ(t) = -x
        psi_max = np.max(psi)
        if -x > psi_max:  # Entry time is 0 if |x| is large enough
            return 0.0
        
        # Find the first time where psi(t) = -x
        # Since psi is decreasing, we can use binary search
        if psi[0] <= -x:  # Edge case
            return 0.0
            
        # Binary search for the entry time
        left, right = 0, len(t_eval) - 1
        while left < right:
            mid = (left + right) // 2
            if psi[mid] > -x:
                left = mid + 1
            else:
                right = mid
                
        # Interpolate for more precision
        t_entry = t_eval[left]
        if left > 0:
            t_prev, psi_prev = t_eval[left-1], psi[left-1]
            t_curr, psi_curr = t_eval[left], psi[left]
            t_entry = t_prev + (t_curr - t_prev) * (-x - psi_prev) / (psi_curr - psi_prev)
            
        return t_entry
    
    def find_exit_time(self, x, mu):
        """
        Find the optimal exit time for a seller with initial position x.
        
        Parameters:
        -----------
        x : float
            Initial position (positive for sellers)
        mu : callable
            Aggregate trading rate function
            
        Returns:
        --------
        exit_time : float
            Optimal exit time
        """
        if x <= 0:  # Buyers exit at the terminal time
            return self.T
        
        # Compute φ_μ function
        t_eval = np.linspace(0, self.T, 1000)
        phi = self.compute_phi(mu, t_eval)
        
        # Find the time t such that φ_μ(t) = x
        phi_max = phi[-1]  # Maximum value at the end
        if x > phi_max:  # Exit time is T if x is large enough
            return self.T
        
        # Find the first time where phi(t) = x
        # Since phi is increasing, we can use binary search
        if phi[0] >= x:  # Edge case
            return 0.0
            
        # Binary search for the exit time
        left, right = 0, len(t_eval) - 1
        while left < right:
            mid = (left + right) // 2
            if phi[mid] < x:
                left = mid + 1
            else:
                right = mid
                
        # Interpolate for more precision
        t_exit = t_eval[left]
        if left > 0:
            t_prev, phi_prev = t_eval[left-1], phi[left-1]
            t_curr, phi_curr = t_eval[left], phi[left]
            t_exit = t_prev + (t_curr - t_prev) * (x - phi_prev) / (phi_curr - phi_prev)
            
        return t_exit
    
    def optimal_trading_strategy(self, x, mu, t_eval=None):
        """
        Compute the optimal trading strategy for a player with initial position x.
        
        Parameters:
        -----------
        x : float
            Initial position (negative for buyers, positive for sellers)
        mu : callable
            Aggregate trading rate function
        t_eval : array-like
            Time points at which to evaluate the strategy
            
        Returns:
        --------
        times : array-like
            Time points
        X : array-like
            Portfolio process
        xi : array-like
            Trading rate
        """
        if t_eval is None:
            t_eval = np.linspace(0, self.T, 1000)
        
        if x < 0:  # Buyer
            # Find entry time
            entry_time = self.find_entry_time(x, mu)
            exit_time = self.T  # Buyers exit at terminal time
            
            # Initialize arrays
            X = np.full_like(t_eval, x)
            xi = np.zeros_like(t_eval)
            
            # Compute portfolio process and trading rate
            active_indices = (t_eval >= entry_time) & (t_eval <= exit_time)
            active_times = t_eval[active_indices]
            
            if len(active_times) > 0:
                # Solve ODE for the portfolio process on [entry_time, exit_time]
                A, B = self.solve_riccati_equation(active_times)
                
                # Compute portfolio and trading rate
                for i, t in enumerate(active_times):
                    # Approximate the portfolio process
                    # For simplicity, we use a linear approximation here
                    # In a real implementation, you would solve the ODE more accurately
                    remaining_time = exit_time - t
                    X[active_indices][i] = x * (remaining_time / (exit_time - entry_time))
                    
                    # Trading rate
                    xi[active_indices][i] = -X[active_indices][i] / (remaining_time + 1e-10)
            
        else:  # Seller
            # Find exit time
            entry_time = 0.0  # Sellers enter immediately
            exit_time = self.find_exit_time(x, mu)
            
            # Initialize arrays
            X = np.full_like(t_eval, x)
            xi = np.zeros_like(t_eval)
            
            # Compute portfolio process and trading rate
            active_indices = (t_eval >= entry_time) & (t_eval <= exit_time)
            active_times = t_eval[active_indices]
            
            if len(active_times) > 0:
                # Solve ODE for the portfolio process on [entry_time, exit_time]
                A, B = self.solve_riccati_equation(active_times)
                
                # Compute portfolio and trading rate
                for i, t in enumerate(active_times):
                    # Approximate the portfolio process
                    # For simplicity, we use a linear approximation here
                    remaining_time = exit_time - t
                    X[active_indices][i] = x * (remaining_time / exit_time)
                    
                    # Trading rate
                    xi[active_indices][i] = X[active_indices][i] / (remaining_time + 1e-10)
        
        return t_eval, X, xi
    
    def compute_cost(self, x, xi, mu, t_eval):
        """
        Compute the cost for a player with initial position x and trading strategy xi.
        
        Parameters:
        -----------
        x : float
            Initial position
        xi : array-like
            Trading rate
        mu : callable
            Aggregate trading rate function
        t_eval : array-like
            Time points
            
        Returns:
        --------
        cost : float
            Total cost
        """
        # Initialize portfolio process
        X = np.zeros_like(t_eval)
        X[0] = x
        
        # Compute portfolio process from trading rate
        for i in range(1, len(t_eval)):
            dt = t_eval[i] - t_eval[i-1]
            X[i] = X[i-1] - xi[i-1] * dt
        
        # Compute cost components
        dt = t_eval[1] - t_eval[0]
        impact_cost = np.sum(0.5 * np.array([self.eta(t) for t in t_eval]) * xi**2) * dt
        permanent_cost = np.sum(np.array([self.kappa(t) for t in t_eval]) * mu(t_eval) * X) * dt
        risk_cost = np.sum(0.5 * np.array([self.lambda_(t) for t in t_eval]) * X**2) * dt
        
        return impact_cost + permanent_cost + risk_cost

class MeanFieldGameSimulation:
    def __init__(self, T=1.0, eta=5.0, kappa=10.0, lambda_=5.0, n_iterations=20):
        """
        Simulate a mean-field game of portfolio liquidation.
        
        Parameters:
        -----------
        T : float
            Trading horizon
        eta : float
            Instantaneous market impact parameter
        kappa : float
            Permanent market impact parameter
        lambda_ : float
            Risk aversion parameter
        n_iterations : int
            Number of iterations for fixed-point algorithm
        """
        self.T = T
        self.eta = eta
        self.kappa = kappa
        self.lambda_ = lambda_
        self.n_iterations = n_iterations
        self.game = PortfolioLiquidationGame(T, eta, kappa, lambda_, 0.0)  # MFG with delta=0
        
    def initialize_distribution(self, distribution_type='exponential', seller_weight=0.8, params=None):
        """
        Initialize the distribution of initial positions.
        
        Parameters:
        -----------
        distribution_type : str
            Type of distribution ('exponential', 'normal', etc.)
        seller_weight : float
            Proportion of sellers in the market
        params : dict
            Additional parameters for the distribution
            
        Returns:
        --------
        nu : callable
            Distribution function
        """
        if distribution_type == 'exponential':
            # Default parameters
            if params is None:
                params = {'buyer_rate': 1.0, 'seller_rate': 1.5}
            
            # Distribution function for exponential distribution
            self.seller_weight = seller_weight
            self.buyer_weight = 1.0 - seller_weight
            self.buyer_rate = params['buyer_rate']
            self.seller_rate = params['seller_rate']
            
            def nu_pdf(x):
                if x < 0:
                    return self.buyer_weight * self.buyer_rate * np.exp(self.buyer_rate * x)
                else:
                    return self.seller_weight * self.seller_rate * np.exp(-self.seller_rate * x)
            
            def nu_cdf(x):
                if x < 0:
                    return self.buyer_weight * (1 - np.exp(self.buyer_rate * x))
                else:
                    return self.buyer_weight + self.seller_weight * (1 - np.exp(-self.seller_rate * x))
            
            def expected_value():
                return -self.buyer_weight / self.buyer_rate + self.seller_weight / self.seller_rate
            
            self.nu_pdf = nu_pdf
            self.nu_cdf = nu_cdf
            self.E_nu = expected_value()
            
            # Tail probability functions p and q
            def p(x):
                if x <= 0:
                    return self.buyer_weight * (1 - np.exp(self.buyer_rate * x))
                else:
                    return 0.0
            
            def q(x):
                if x >= 0:
                    return self.seller_weight * np.exp(-self.seller_rate * x)
                else:
                    return self.seller_weight + self.buyer_weight * np.exp(self.buyer_rate * x)
            
            self.p = p
            self.q = q
            
            return nu_pdf
            
        else:
            raise ValueError(f"Distribution type {distribution_type} not implemented")
    
    def solve_mean_field_game(self):
        """
        Solve the mean-field game using a fixed-point iteration.
        
        Returns:
        --------
        mu : callable
            Equilibrium aggregate trading rate
        """
        # Initialize with a constant trading rate
        t_eval = np.linspace(0, self.T, 1000)
        mu_current = lambda t: self.E_nu / self.T * np.ones_like(t) if np.isscalar(t) else np.full_like(t, self.E_nu / self.T)
        
        # Fixed-point iteration
        for i in range(self.n_iterations):
            print(f"Iteration {i+1}/{self.n_iterations}")
            
            # Compute entry and exit times for different initial positions
            x_values = np.linspace(-5, 5, 100)  # Range of initial positions
            entry_times = np.zeros_like(x_values)
            exit_times = np.zeros_like(x_values)
            
            for j, x in enumerate(x_values):
                if x < 0:  # Buyer
                    entry_times[j] = self.game.find_entry_time(x, mu_current)
                    exit_times[j] = self.T
                else:  # Seller
                    entry_times[j] = 0.0
                    exit_times[j] = self.game.find_exit_time(x, mu_current)
            
            # Compute new aggregate trading rate
            def new_mu(t):
                if np.isscalar(t):
                    # Numerical integration for the aggregate trading rate
                    # This is a simplified version; in practice, you would need to solve the integral equation
                    result = 0.0
                    for j, x in enumerate(x_values):
                        if x < 0:  # Buyer
                            if entry_times[j] <= t <= exit_times[j]:
                                _, _, xi = self.game.optimal_trading_strategy(x, mu_current, np.array([t]))
                                result += xi[0] * self.nu_pdf(x) * (x_values[1] - x_values[0])
                        else:  # Seller
                            if entry_times[j] <= t <= exit_times[j]:
                                _, _, xi = self.game.optimal_trading_strategy(x, mu_current, np.array([t]))
                                result += xi[0] * self.nu_pdf(x) * (x_values[1] - x_values[0])
                    return result
                else:
                    return np.array([new_mu(ti) for ti in t])
            
            # Update mu with a damping factor to improve convergence
            damping = 0.7
            mu_next = lambda t: (1 - damping) * mu_current(t) + damping * new_mu(t)
            
            # Check convergence
            max_diff = np.max(np.abs(mu_next(t_eval) - mu_current(t_eval)))
            print(f"Maximum difference: {max_diff:.6f}")
            
            if max_diff < 1e-4:
                print(f"Converged after {i+1} iterations")
                break
                
            mu_current = mu_next
        
        self.mu_equilibrium = mu_current
        return mu_current
    
    def plot_equilibrium_results(self, n_players=10):
        """
        Plot the equilibrium results of the mean-field game.
        
        Parameters:
        -----------
        n_players : int
            Number of representative players to plot
        """
        t_eval = np.linspace(0, self.T, 1000)
        
        # Sample initial positions from the distribution
        np.random.seed(42)  # For reproducibility
        x_values = []
        
        # Generate initial positions with roughly the correct distribution
        for _ in range(n_players):
            if np.random.rand() < self.seller_weight:
                # Generate a positive position (seller)
                x = np.random.exponential(1/self.seller_rate)
            else:
                # Generate a negative position (buyer)
                x = -np.random.exponential(1/self.buyer_rate)
            x_values.append(x)
        
        # Sort for better visualization
        x_values.sort()
        
        # Plot portfolio processes
        plt.figure(figsize=(12, 8))
        
        # Plot equilibrium trading rate
        plt.subplot(2, 2, 1)
        plt.plot(t_eval, self.mu_equilibrium(t_eval), 'k-', linewidth=2, label='Aggregate Trading Rate')
        plt.xlabel('Time')
        plt.ylabel('Trading Rate')
        plt.title('Equilibrium Aggregate Trading Rate')
        plt.grid(True)
        plt.legend()
        
        # Plot portfolio processes
        plt.subplot(2, 2, 2)
        for x in x_values:
            _, X, _ = self.game.optimal_trading_strategy(x, self.mu_equilibrium, t_eval)
            color = 'r' if x < 0 else 'b'
            plt.plot(t_eval, X, color=color, alpha=0.7)
            
            # Mark entry and exit times
            if x < 0:  # Buyer
                entry_time = self.game.find_entry_time(x, self.mu_equilibrium)
                plt.scatter(entry_time, x, color='g', s=50, zorder=3)
            else:  # Seller
                exit_time = self.game.find_exit_time(x, self.mu_equilibrium)
                plt.scatter(exit_time, 0, color='g', s=50, zorder=3)
        
        plt.xlabel('Time')
        plt.ylabel('Portfolio Size')
        plt.title('Portfolio Processes')
        plt.grid(True)
        
        # Plot entry and exit times
        plt.subplot(2, 2, 3)
        x_range = np.linspace(-5, 5, 200)
        entry_times = np.zeros_like(x_range)
        exit_times = np.zeros_like(x_range)
        
        for i, x in enumerate(x_range):
            if x < 0:  # Buyer
                entry_times[i] = self.game.find_entry_time(x, self.mu_equilibrium)
                exit_times[i] = self.T
            else:  # Seller
                entry_times[i] = 0.0
                exit_times[i] = self.game.find_exit_time(x, self.mu_equilibrium)
        
        plt.plot(x_range, entry_times, 'g-', label='Entry Time')
        plt.plot(x_range, exit_times, 'r-', label='Exit Time')
        plt.xlabel('Initial Position')
        plt.ylabel('Time')
        plt.title('Entry and Exit Times')
        plt.grid(True)
        plt.legend()
        
        # Plot costs
        plt.subplot(2, 2, 4)
        costs = []
        
        for x in x_range:
            _, X, xi = self.game.optimal_trading_strategy(x, self.mu_equilibrium, t_eval)
            cost = self.game.compute_cost(x, xi, self.mu_equilibrium, t_eval)
            costs.append(cost)
        
        plt.plot(x_range, costs, 'b-')
        plt.xlabel('Initial Position')
        plt.ylabel('Cost')
        plt.title('Costs')
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
        
    def compare_with_unconstrained(self):
        """
        Compare the constrained solution with an unconstrained solution.
        """
        # For simplicity, we'll approximate the unconstrained solution
        # In practice, you would solve the unconstrained game properly
        
        t_eval = np.linspace(0, self.T, 1000)
        
        # Define a simple unconstrained strategy that allows changing trading direction
        def unconstrained_strategy(x, t):
            # Simple linear strategy that doesn't respect the trading constraint
            remaining_time = self.T - t
            if remaining_time < 1e-6:
                return 0
            return x / remaining_time
        
        # Compare for a few representative initial positions
        x_values = [-3.0, -1.0, -0.5, 0.5, 1.0, 3.0]
        
        plt.figure(figsize=(15, 10))
        
        # Plot trading rates
        plt.subplot(2, 2, 1)
        plt.plot(t_eval, self.mu_equilibrium(t_eval), 'k-', linewidth=2, label='Constrained')
        
        # Compute unconstrained aggregate trading rate (approximation)
        unconstrained_mu = np.zeros_like(t_eval)
        x_grid = np.linspace(-5, 5, 100)
        for x in x_grid:
            for i, t in enumerate(t_eval):
                unconstrained_mu[i] += unconstrained_strategy(x, t) * self.nu_pdf(x) * (x_grid[1] - x_grid[0])
        
        plt.plot(t_eval, unconstrained_mu, 'r--', linewidth=2, label='Unconstrained (Approx.)')
        plt.xlabel('Time')
        plt.ylabel('Trading Rate')
        plt.title('Aggregate Trading Rate Comparison')
        plt.grid(True)
        plt.legend()
        
        # Plot portfolio processes for a few players
        for i, x in enumerate(x_values):
            plt.subplot(2, 3, i+4)
            
            # Constrained strategy
            _, X_constrained, xi_constrained = self.game.optimal_trading_strategy(x, self.mu_equilibrium, t_eval)
            
            # Unconstrained strategy (approximation)
            X_unconstrained = np.zeros_like(t_eval)
            X_unconstrained[0] = x
            xi_unconstrained = np.zeros_like(t_eval)
            
            for j in range(len(t_eval)-1):
                xi_unconstrained[j] = unconstrained_strategy(X_unconstrained[j], t_eval[j])
                dt = t_eval[j+1] - t_eval[j]
                X_unconstrained[j+1] = X_unconstrained[j] - xi_unconstrained[j] * dt
            
            plt.plot(t_eval, X_constrained, 'b-', label='Constrained')
            plt.plot(t_eval, X_unconstrained, 'r--', label='Unconstrained')
            plt.xlabel('Time')
            plt.ylabel('Portfolio Size')
            plt.title(f'Initial Position: {x:.1f}')
            plt.grid(True)
            if i == 0:
                plt.legend()
        
        plt.tight_layout()
        plt.show()
        
        # Compare costs
        x_range = np.linspace(-5, 5, 100)
        constrained_costs = []
        unconstrained_costs = []
        
        for x in x_range:
            # Constrained costs
            _, _, xi_constrained = self.game.optimal_trading_strategy(x, self.mu_equilibrium, t_eval)
            constrained_cost = self.game.compute_cost(x, xi_constrained, self.mu_equilibrium, t_eval)
            constrained_costs.append(constrained_cost)
            
            # Unconstrained costs (approximation)
            X_unconstrained = np.zeros_like(t_eval)
            X_unconstrained[0] = x
            xi_unconstrained = np.zeros_like(t_eval)
            
            for j in range(len(t_eval)-1):
                xi_unconstrained[j] = unconstrained_strategy(X_unconstrained[j], t_eval[j])
                dt = t_eval[j+1] - t_eval[j]
                X_unconstrained[j+1] = X_unconstrained[j] - xi_unconstrained[j] * dt
            
            unconstrained_cost = self.game.compute_cost(x, xi_unconstrained, lambda t: unconstrained_mu, t_eval)
            unconstrained_costs.append(unconstrained_cost)
        
        plt.figure(figsize=(10, 6))
        plt.plot(x_range, constrained_costs, 'b-', linewidth=2, label='Constrained')
        plt.plot(x_range, unconstrained_costs, 'r--', linewidth=2, label='Unconstrained (Approx.)')
        plt.xlabel('Initial Position')
        plt.ylabel('Cost')
        plt.title('Cost Comparison')
        plt.grid(True)
        plt.legend()
        plt.show()
        
        # Print average costs
        avg_constrained_cost = np.mean(constrained_costs)
        avg_unconstrained_cost = np.mean(unconstrained_costs)
        
        print(f"Average Constrained Cost: {avg_constrained_cost:.4f}")
        print(f"Average Unconstrained Cost: {avg_unconstrained_cost:.4f}")
        print(f"Difference: {avg_unconstrained_cost - avg_constrained_cost:.4f}")
        
        return constrained_costs, unconstrained_costs

class NPlayerGameSimulation:
    def __init__(self, N, T=1.0, eta=5.0, kappa=10.0, lambda_=5.0, max_iterations=20):
        """
        Simulate an N-player game of portfolio liquidation.
        
        Parameters:
        -----------
        N : int
            Number of players
        T : float
            Trading horizon
        eta : float
            Instantaneous market impact parameter
        kappa : float
            Permanent market impact parameter
        lambda_ : float
            Risk aversion parameter
        max_iterations : int
            Maximum number of iterations for finding Nash equilibrium
        """
        self.N = N
        self.T = T
        self.eta = eta
        self.kappa = kappa
        self.lambda_ = lambda_
        self.max_iterations = max_iterations
        self.delta = 1.0 / N
        self.game = PortfolioLiquidationGame(T, eta, kappa, lambda_, self.delta)
    
    def initialize_players(self, x_values=None, distribution_type='exponential', seller_weight=0.8):
        """
        Initialize the players with initial positions.
        
        Parameters:
        -----------
        x_values : list
            Initial positions for each player (if None, they are sampled from the distribution)
        distribution_type : str
            Type of distribution to sample from
        seller_weight : float
            Proportion of sellers in the market
        """
        if x_values is not None:
            if len(x_values) != self.N:
                raise ValueError(f"Expected {self.N} initial positions, got {len(x_values)}")
            self.x_values = x_values
        else:
            # Initialize a distribution similar to the MFG
            mfg_sim = MeanFieldGameSimulation(self.T, self.eta, self.kappa, self.lambda_)
            mfg_sim.initialize_distribution(distribution_type, seller_weight)
            
            # Sample initial positions
            np.random.seed(42)  # For reproducibility
            self.x_values = []
            
            for _ in range(self.N):
                if np.random.rand() < seller_weight:
                    # Generate a positive position (seller)
                    x = np.random.exponential(1/mfg_sim.seller_rate)
                else:
                    # Generate a negative position (buyer)
                    x = -np.random.exponential(1/mfg_sim.buyer_rate)
                self.x_values.append(x)
    
    def solve_n_player_game(self):
        """
        Solve the N-player game using best-response dynamics.
        
        Returns:
        --------
        strategies : list
            Equilibrium strategies for each player
        """
        t_eval = np.linspace(0, self.T, 1000)
        
        # Initialize with uniform trading rates
        strategies = []
        for x in self.x_values:
            if x < 0:  # Buyer
                xi = np.zeros_like(t_eval)
                xi[t_eval >= 0.5 * self.T] = -x / (0.5 * self.T)  # Trade in second half
            else:  # Seller
                xi = np.zeros_like(t_eval)
                xi[t_eval <= 0.5 * self.T] = x / (0.5 * self.T)  # Trade in first half
            strategies.append(xi)
        
        # Best-response dynamics
        converged = False
        for iteration in range(self.max_iterations):
            print(f"Iteration {iteration+1}/{self.max_iterations}")
            
            max_diff = 0.0
            
            # Update each player's strategy
            for i in range(self.N):
                # Compute average trading rate of others
                mu = lambda t: np.sum([strategies[j](t) if callable(strategies[j]) else 
                                        np.interp(t, t_eval, strategies[j]) 
                                        for j in range(self.N) if j != i], axis=0) / (self.N - 1)
                
                # Compute best response
                _, _, new_xi = self.game.optimal_trading_strategy(self.x_values[i], mu, t_eval)
                
                # Check convergence
                if iteration > 0:
                    diff = np.max(np.abs(new_xi - strategies[i]))
                    max_diff = max(max_diff, diff)
                
                # Update strategy
                strategies[i] = lambda t, xi=new_xi: np.interp(t, t_eval, xi) if np.isscalar(t) else np.interp(t, t_eval, xi)
            
            print(f"Maximum strategy change: {max_diff:.6f}")
            
            if max_diff < 1e-4 and iteration > 0:
                print(f"Converged after {iteration+1} iterations")
                converged = True
                break
        
        if not converged:
            print("Warning: Did not converge within the maximum number of iterations")
        
        self.strategies = strategies
        
        # Compute aggregate trading rate
        self.mu_aggregate = lambda t: np.sum([s(t) for s in strategies]) / self.N
        
        return strategies
    
    def plot_n_player_results(self):
        """
        Plot the results of the N-player game.
        """
        t_eval = np.linspace(0, self.T, 1000)
        
        # Plot portfolio processes and trading rates
        plt.figure(figsize=(15, 10))
        
        # Plot aggregate trading rate
        plt.subplot(2, 2, 1)
        aggregate_rate = self.mu_aggregate(t_eval)
        plt.plot(t_eval, aggregate_rate, 'k-', linewidth=2)
        plt.xlabel('Time')
        plt.ylabel('Trading Rate')
        plt.title('Aggregate Trading Rate')
        plt.grid(True)
        
        # Plot individual trading rates
        plt.subplot(2, 2, 2)
        for i, strategy in enumerate(self.strategies):
            color = 'r' if self.x_values[i] < 0 else 'b'
            plt.plot(t_eval, strategy(t_eval), color=color, alpha=0.7)
        plt.xlabel('Time')
        plt.ylabel('Trading Rate')
        plt.title('Individual Trading Rates')
        plt.grid(True)
        
        # Plot portfolio processes
        plt.subplot(2, 2, 3)
        for i, (x, strategy) in enumerate(zip(self.x_values, self.strategies)):
            # Compute portfolio process
            X = np.zeros_like(t_eval)
            X[0] = x
            for j in range(1, len(t_eval)):
                dt = t_eval[j] - t_eval[j-1]
                X[j] = X[j-1] - strategy(t_eval[j-1]) * dt
            
            color = 'r' if x < 0 else 'b'
            plt.plot(t_eval, X, color=color, alpha=0.7)
        plt.xlabel('Time')
        plt.ylabel('Portfolio Size')
        plt.title('Portfolio Processes')
        plt.grid(True)
        
        # Plot entry and exit times
        plt.subplot(2, 2, 4)
        x_sorted = sorted(self.x_values)
        entry_times = []
        exit_times = []
        
        for x in x_sorted:
            if x < 0:  # Buyer
                entry_time = self.game.find_entry_time(x, self.mu_aggregate)
                exit_time = self.T
            else:  # Seller
                entry_time = 0.0
                exit_time = self.game.find_exit_time(x, self.mu_aggregate)
            
            entry_times.append(entry_time)
            exit_times.append(exit_time)
        
        plt.scatter(x_sorted, entry_times, c='g', label='Entry Time')
        plt.scatter(x_sorted, exit_times, c='r', label='Exit Time')
        plt.xlabel('Initial Position')
        plt.ylabel('Time')
        plt.title('Entry and Exit Times')
        plt.grid(True)
        plt.legend()
        
        plt.tight_layout()
        plt.show()
    
    def compare_with_mfg(self):
        """
        Compare the N-player game with the mean-field game.
        """
        # Solve the corresponding mean-field game
        mfg_sim = MeanFieldGameSimulation(self.T, self.eta, self.kappa, self.lambda_)
        
        # Use the same distribution of initial positions
        seller_weight = len([x for x in self.x_values if x > 0]) / self.N
        buyer_rate = 1.0 / np.mean([-x for x in self.x_values if x < 0]) if any(x < 0 for x in self.x_values) else 1.0
        seller_rate = 1.0 / np.mean([x for x in self.x_values if x > 0]) if any(x > 0 for x in self.x_values) else 1.0
        
        mfg_sim.initialize_distribution('exponential', seller_weight, {'buyer_rate': buyer_rate, 'seller_rate': seller_rate})
        mfg_mu = mfg_sim.solve_mean_field_game()
        
        t_eval = np.linspace(0, self.T, 1000)
        
        # Compare aggregate trading rates
        plt.figure(figsize=(10, 6))
        plt.plot(t_eval, self.mu_aggregate(t_eval), 'b-', linewidth=2, label=f'N-Player Game (N={self.N})')
        plt.plot(t_eval, mfg_mu(t_eval), 'r--', linewidth=2, label='Mean-Field Game')
        plt.xlabel('Time')
        plt.ylabel('Trading Rate')
        plt.title('Aggregate Trading Rate Comparison')
        plt.grid(True)
        plt.legend()
        plt.show()
        
        # Compare entry and exit times
        plt.figure(figsize=(10, 6))
        
        # N-player game
        x_sorted = sorted(self.x_values)
        n_entry_times = []
        n_exit_times = []
        
        for x in x_sorted:
            if x < 0:  # Buyer
                entry_time = self.game.find_entry_time(x, self.mu_aggregate)
                exit_time = self.T
            else:  # Seller
                entry_time = 0.0
                exit_time = self.game.find_exit_time(x, self.mu_aggregate)
            
            n_entry_times.append(entry_time)
            n_exit_times.append(exit_time)
        
        # MFG
        x_range = np.linspace(min(x_sorted), max(x_sorted), 100)
        mfg_entry_times = []
        mfg_exit_times = []
        
        for x in x_range:
            if x < 0:  # Buyer
                entry_time = mfg_sim.game.find_entry_time(x, mfg_mu)
                exit_time = self.T
            else:  # Seller
                entry_time = 0.0
                exit_time = mfg_sim.game.find_exit_time(x, mfg_mu)
            
            mfg_entry_times.append(entry_time)
            mfg_exit_times.append(exit_time)
        
        plt.scatter(x_sorted, n_entry_times, c='b', s=100, alpha=0.7, label=f'N-Player Entry (N={self.N})')
        plt.scatter(x_sorted, n_exit_times, c='r', s=100, alpha=0.7, label=f'N-Player Exit (N={self.N})')
        plt.plot(x_range, mfg_entry_times, 'b--', linewidth=2, label='MFG Entry')
        plt.plot(x_range, mfg_exit_times, 'r--', linewidth=2, label='MFG Exit')
        plt.xlabel('Initial Position')
        plt.ylabel('Time')
        plt.title('Entry and Exit Times Comparison')
        plt.grid(True)
        plt.legend()
        plt.show()
        
        # Calculate and print convergence metrics
        rate_diff = np.mean(np.abs(self.mu_aggregate(t_eval) - mfg_mu(t_eval)))
        print(f"Average difference in trading rates: {rate_diff:.6f}")
        
        return rate_diff

# Main experiment function
def run_experiments():
    # Experiment 1: MFG with seller-dominated market
    print("=== Experiment 1: Mean-Field Game with Seller-Dominated Market ===")
    mfg_sim = MeanFieldGameSimulation()
    mfg_sim.initialize_distribution('exponential', seller_weight=0.8)
    mfg_sim.solve_mean_field_game()
    mfg_sim.plot_equilibrium_results()
    constrained_costs, unconstrained_costs = mfg_sim.compare_with_unconstrained()
    
    # Experiment 2: N-player game with varying N
    print("\n=== Experiment 2: N-Player Game with Varying N ===")
    convergence_metrics = []
    N_values = [5, 10, 15, 30]
    
    for N in N_values:
        print(f"\nN-Player Game with N={N}")
        n_player_sim = NPlayerGameSimulation(N)
        n_player_sim.initialize_players()
        n_player_sim.solve_n_player_game()
        n_player_sim.plot_n_player_results()
        rate_diff = n_player_sim.compare_with_mfg()
        convergence_metrics.append(rate_diff)
    
    # Plot convergence to MFG
    plt.figure(figsize=(10, 6))
    plt.plot(N_values, convergence_metrics, 'bo-', linewidth=2)
    plt.xlabel('Number of Players (N)')
    plt.ylabel('Average Trading Rate Difference')
    plt.title('Convergence to Mean-Field Game')
    plt.grid(True)
    plt.show()
    
    # Experiment 3: Impact of market parameters
    print("\n=== Experiment 3: Impact of Market Parameters ===")
    
    # Vary kappa (permanent impact)
    kappa_values = [5, 10, 15, 20]
    avg_constrained_costs = []
    avg_unconstrained_costs = []
    
    for kappa in kappa_values:
        print(f"\nVarying Permanent Impact: kappa={kappa}")
        mfg_sim = MeanFieldGameSimulation(kappa=kappa)
        mfg_sim.initialize_distribution('exponential', seller_weight=0.8)
        mfg_sim.solve_mean_field_game()
        constrained_costs, unconstrained_costs = mfg_sim.compare_with_unconstrained()
        
        avg_constrained_costs.append(np.mean(constrained_costs))
        avg_unconstrained_costs.append(np.mean(unconstrained_costs))
    
    plt.figure(figsize=(10, 6))
    plt.plot(kappa_values, avg_constrained_costs, 'b-', linewidth=2, label='Constrained')
    plt.plot(kappa_values, avg_unconstrained_costs, 'r--', linewidth=2, label='Unconstrained')
    plt.xlabel('Permanent Impact (κ)')
    plt.ylabel('Average Cost')
    plt.title('Impact of Permanent Market Impact on Average Costs')
    plt.grid(True)
    plt.legend()
    plt.show()
    
    # Vary lambda (risk aversion)
    lambda_values = [0, 5, 10, 15]
    avg_constrained_costs = []
    avg_unconstrained_costs = []
    
    for lambda_ in lambda_values:
        print(f"\nVarying Risk Aversion: lambda={lambda_}")
        mfg_sim = MeanFieldGameSimulation(lambda_=lambda_)
        mfg_sim.initialize_distribution('exponential', seller_weight=0.8)
        mfg_sim.solve_mean_field_game()
        constrained_costs, unconstrained_costs = mfg_sim.compare_with_unconstrained()
        
        avg_constrained_costs.append(np.mean(constrained_costs))
        avg_unconstrained_costs.append(np.mean(unconstrained_costs))
    
    plt.figure(figsize=(10, 6))
    plt.plot(lambda_values, avg_constrained_costs, 'b-', linewidth=2, label='Constrained')
    plt.plot(lambda_values, avg_unconstrained_costs, 'r--', linewidth=2, label='Unconstrained')
    plt.xlabel('Risk Aversion (λ)')
    plt.ylabel('Average Cost')
    plt.title('Impact of Risk Aversion on Average Costs')
    plt.grid(True)
    plt.legend()
    plt.show()
    
    print("\nExperiments completed.")

# Run all experiments
if __name__ == "__main__":
    run_experiments()