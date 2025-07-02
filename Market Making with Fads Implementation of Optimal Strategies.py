import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from matplotlib.ticker import FormatStrFormatter

class MarketMakingWithFads:
    def __init__(self, params=None):
        """
        Initialize the market making model with default or provided parameters.
        """
        # Default parameters based on the paper
        default_params = {
            'T_horizon': 1.0,            # Time horizon
            'N_steps': 1000,             # Number of time steps for discretization
            'S0': 100.0,                 # Initial price
            'mu_price': 0.0,             # Drift of S_t
            'sigma_price': 1.0,          # Volatility of S_t
            'q_noise': 0.6,              # Coefficient for temporary component (fad)
            'p_noise': np.sqrt(1-0.6**2), # Coefficient for persistent component
            'eta_fad': 10.0,             # Mean-reversion speed of fad
            'U0': 0.0,                   # Initial fad value
            'Q0': 0.0,                   # Initial inventory
            'q_inv_min': -10.0,          # Minimum inventory
            'q_inv_max': 10.0,           # Maximum inventory
            'phi_uninformed': 15.0,      # Arrival intensity for uninformed traders
            'k_intensity': 1.0,          # Sensitivity of arrivals to displacements
            'gamma_intensity': 1.0,      # Sensitivity of informed traders to fad
            'alpha_penalty': 0.001,      # Terminal inventory penalty
            'phi_penalty': 0.1,          # Running inventory penalty
            'delta_a_lower_bound': 0.0,  # Lower bound for ask displacement
            'delta_b_lower_bound': 0.0,  # Lower bound for bid displacement
            'n_simulations': 100         # Number of simulation paths
        }
        
        # Update with provided parameters
        self.params = default_params.copy()
        if params:
            self.params.update(params)
        
        # Derived parameters
        self.params['dt'] = self.params['T_horizon'] / self.params['N_steps']
        
        # Calculate psi_informed to keep expected arrivals constant (Eq. 61)
        self.calibrate_psi_informed()
        
        # Store solutions to ODEs for the strategies
        self.solutions = {}
        
        # Initialize time grid
        self.t_grid = np.linspace(0, self.params['T_horizon'], self.params['N_steps'] + 1)
        
        print("Model initialized with parameters:")
        for key, value in self.params.items():
            print(f"  {key}: {value}")
    
    def calibrate_psi_informed(self):
        """
        Calibrate psi_informed to keep expected arrivals constant (Eq. 61).
        """
        # Extract parameters
        T = self.params['T_horizon']
        phi = self.params['phi_uninformed']
        gamma = self.params['gamma_intensity']
        sigma = self.params['sigma_price']
        q_noise = self.params['q_noise']
        eta = self.params['eta_fad']
        
        # For an OU process U_t, E[exp(c*U_t)] = exp(c^2 * var(U_t) / 2)
        # For stationary OU, var(U_t) = 1/(2*eta)
        # So E[exp(±gamma*sigma*q*U_t)] = exp((gamma*sigma*q)^2 / (4*eta))
        expected_exp_term = np.exp((gamma * sigma * q_noise)**2 / (4 * eta))
        
        # Calibrate psi_informed according to Eq. 61
        target_arrivals = 30  # As per the paper
        self.params['psi_informed'] = (target_arrivals - phi * T) / (expected_exp_term * T)
        
        print(f"Calibrated psi_informed = {self.params['psi_informed']:.4f}")
    
    def solve_full_information_hjb(self):
        """
        Solve the approximate HJB equation for full information strategy.
        This implements the ODEs for A(t), b0(t), b1(t), c0(t), c1(t), c2(t) 
        from Propositions 10 & 11.
        """
        print("Solving Full Information HJB approximation...")
        
        # Extract parameters
        T = self.params['T_horizon']
        phi_penalty = self.params['phi_penalty']
        alpha_penalty = self.params['alpha_penalty']
        k = self.params['k_intensity']
        phi_uninformed = self.params['phi_uninformed']
        psi_informed = self.params['psi_informed']
        mu = self.params['mu_price']
        sigma = self.params['sigma_price']
        q_noise = self.params['q_noise']
        eta = self.params['eta_fad']
        gamma = self.params['gamma_intensity']
        
        # Constants for the exponential expansion (before Eq 35)
        exp_minus_1 = np.exp(-1)
        total_intensity = phi_uninformed + psi_informed
        
        # Define ODE for A(t) (Riccati equation from Eq 36)
        def A_ode(t, A):
            return phi_penalty - (4/k) * total_intensity * exp_minus_1 * A**2
        
        # Solve A(t) backward from T to 0
        A_sol = solve_ivp(
            A_ode, 
            [T, 0], 
            [alpha_penalty], 
            t_eval=np.flip(self.t_grid),
            method='RK45',
            rtol=1e-8,
            atol=1e-10
        )
        
        # Extract A(t) solution (and flip to get increasing time)
        A_values = np.flip(A_sol.y[0])
        
        # Define system of ODEs for b0, b1, c0, c1, c2 (Eq 38)
        def bc_system(t, y):
            b0, b1, c0, c1, c2 = y
            
            # Current value of A(t) - interpolate from solution
            idx = np.searchsorted(self.t_grid, t)
            if idx == len(self.t_grid):
                idx = len(self.t_grid) - 1
            A_t = A_values[idx]
            
            # ODEs from Eq 38
            db0 = mu + (2/k) * total_intensity * exp_minus_1 * (2*A_t*b0)
            db1 = -sigma * q_noise * eta + (2/k) * total_intensity * exp_minus_1 * (2*A_t*b1)
            
            # Terms for c0, c1, c2 equations
            a_term = (2/k) * exp_minus_1 * (2*total_intensity)
            b_term = (2/k) * exp_minus_1 * (2*A_t*total_intensity)
            c_term = (2/k) * exp_minus_1 * (k**2 * total_intensity)
            
            # Compute terms with b0, b1 squared
            b0_sq_term = (2/k) * exp_minus_1 * (k**2 * total_intensity * b0**2 / 4)
            b1_sq_term = (2/k) * exp_minus_1 * (k**2 * total_intensity * b1**2 / 4)
            
            # Cross terms for b0*b1
            b0b1_term = (2/k) * exp_minus_1 * (k**2 * total_intensity * b0 * b1 / 2)
            
            # Informed trader terms
            informed_c1 = (2/k) * exp_minus_1 * psi_informed * gamma * sigma * q_noise
            informed_c2 = (2/k) * exp_minus_1 * psi_informed * (gamma * sigma * q_noise)**2 / 2
            
            # Compute the derivatives
            dc0 = a_term + b_term*c0 + b0_sq_term
            dc1 = b_term*c1 + b0b1_term + informed_c1
            dc2 = b_term*c2 + b1_sq_term + informed_c2
            
            return [db0, db1, dc0, dc1, dc2]
        
        # Solve system backward from T to 0 with zero terminal conditions
        bc_sol = solve_ivp(
            bc_system, 
            [T, 0], 
            [0.0, 0.0, 0.0, 0.0, 0.0], 
            t_eval=np.flip(self.t_grid),
            method='RK45',
            rtol=1e-8,
            atol=1e-10
        )
        
        # Extract solutions (and flip to get increasing time)
        b0_values = np.flip(bc_sol.y[0])
        b1_values = np.flip(bc_sol.y[1])
        c0_values = np.flip(bc_sol.y[2])
        c1_values = np.flip(bc_sol.y[3])
        c2_values = np.flip(bc_sol.y[4])
        
        # Store solutions
        self.solutions['FI'] = {
            'A': A_values,
            'b0': b0_values,
            'b1': b1_values,
            'c0': c0_values,
            'c1': c1_values,
            'c2': c2_values
        }
        
        print("Full Information HJB approximation solved.")
    
    def solve_partial_information_hjb(self):
        """
        Solve the approximate HJB equation for partial information strategy.
        This implements the ODEs for Riccati equation P_hat_t and 
        the ODEs for A_PI(t), b0_PI(t), etc. from Propositions 15 & 16.
        """
        print("Solving Partial Information HJB approximation...")
        
        # Extract parameters
        T = self.params['T_horizon']
        phi_penalty = self.params['phi_penalty']
        alpha_penalty = self.params['alpha_penalty']
        k = self.params['k_intensity']
        phi_uninformed = self.params['phi_uninformed']
        psi_informed = self.params['psi_informed']
        mu = self.params['mu_price']
        sigma = self.params['sigma_price']
        q_noise = self.params['q_noise']
        p_noise = self.params['p_noise']
        eta = self.params['eta_fad']
        gamma = self.params['gamma_intensity']
        
        # Constants for the exponential expansion
        exp_minus_1 = np.exp(-1)
        total_intensity = phi_uninformed + psi_informed
        
        # First, solve for P_hat_t (Conditional Variance, Riccati Eq. 47)
        def P_hat_ode(t, P_hat):
            return -eta**2 * q_noise**2 * P_hat**2 - P_hat * (2*eta - 2*eta*q_noise**2) + p_noise**2
        
        # Solve P_hat_t forward from 0 to T (initial condition: P_hat_0 = 0)
        P_hat_sol = solve_ivp(
            P_hat_ode, 
            [0, T], 
            [0.0], 
            t_eval=self.t_grid,
            method='RK45',
            rtol=1e-8,
            atol=1e-10
        )
        
        # Extract P_hat_t solution
        P_hat_values = P_hat_sol.y[0]
        
        # Define ODE for A_PI(t) (same as for FI)
        def A_PI_ode(t, A):
            return phi_penalty - (4/k) * total_intensity * exp_minus_1 * A**2
        
        # Solve A_PI(t) backward from T to 0
        A_PI_sol = solve_ivp(
            A_PI_ode, 
            [T, 0], 
            [alpha_penalty], 
            t_eval=np.flip(self.t_grid),
            method='RK45',
            rtol=1e-8,
            atol=1e-10
        )
        
        # Extract A_PI(t) solution (and flip to get increasing time)
        A_PI_values = np.flip(A_PI_sol.y[0])
        
        # Define system of ODEs for b0_PI, b1_PI, c0_PI, c1_PI, c2_PI (Eq 60)
        def bc_PI_system(t, y):
            b0_PI, b1_PI, c0_PI, c1_PI, c2_PI = y
            
            # Current value of A_PI(t) and P_hat_t - interpolate from solutions
            idx = np.searchsorted(self.t_grid, t)
            if idx == len(self.t_grid):
                idx = len(self.t_grid) - 1
            A_PI_t = A_PI_values[idx]
            P_hat_t = P_hat_values[idx]
            
            # x1 and x2 terms from Proposition 15
            x1 = mu - sigma * q_noise * eta  # Assuming u_hat=0 for simplicity in the drift term
            x2_sq = (-P_hat_t * q_noise * eta + q_noise)**2 * sigma**2
            
            # ODEs from Eq 60
            db0_PI = mu + (2/k) * total_intensity * exp_minus_1 * (2*A_PI_t*b0_PI)
            db1_PI = -sigma * q_noise * eta + (2/k) * total_intensity * exp_minus_1 * (2*A_PI_t*b1_PI)
            
            # Terms for c0_PI, c1_PI, c2_PI equations
            a_term = (2/k) * exp_minus_1 * (2*total_intensity)
            b_term = (2/k) * exp_minus_1 * (2*A_PI_t*total_intensity)
            
            # Compute terms with b0_PI, b1_PI squared
            b0_sq_term = (2/k) * exp_minus_1 * (k**2 * total_intensity * b0_PI**2 / 4)
            b1_sq_term = (2/k) * exp_minus_1 * (k**2 * total_intensity * b1_PI**2 / 4)
            
            # Cross terms for b0_PI*b1_PI
            b0b1_term = (2/k) * exp_minus_1 * (k**2 * total_intensity * b0_PI * b1_PI / 2)
            
            # Informed trader terms
            informed_c1 = (2/k) * exp_minus_1 * psi_informed * gamma * sigma * q_noise
            informed_c2 = (2/k) * exp_minus_1 * psi_informed * (gamma * sigma * q_noise)**2 / 2
            
            # Additional term for PI strategy (x2 term from Proposition 15)
            x2_term = 0.5 * x2_sq * c2_PI
            
            # Compute the derivatives
            dc0_PI = a_term + b_term*c0_PI + b0_sq_term + x2_term
            dc1_PI = b_term*c1_PI + b0b1_term + informed_c1
            dc2_PI = b_term*c2_PI + b1_sq_term + informed_c2
            
            return [db0_PI, db1_PI, dc0_PI, dc1_PI, dc2_PI]
        
        # Solve system backward from T to 0 with zero terminal conditions
        bc_PI_sol = solve_ivp(
            bc_PI_system, 
            [T, 0], 
            [0.0, 0.0, 0.0, 0.0, 0.0], 
            t_eval=np.flip(self.t_grid),
            method='RK45',
            rtol=1e-8,
            atol=1e-10
        )
        
        # Extract solutions (and flip to get increasing time)
        b0_PI_values = np.flip(bc_PI_sol.y[0])
        b1_PI_values = np.flip(bc_PI_sol.y[1])
        c0_PI_values = np.flip(bc_PI_sol.y[2])
        c1_PI_values = np.flip(bc_PI_sol.y[3])
        c2_PI_values = np.flip(bc_PI_sol.y[4])
        
        # Store solutions
        self.solutions['PI'] = {
            'A': A_PI_values,
            'b0': b0_PI_values,
            'b1': b1_PI_values,
            'c0': c0_PI_values,
            'c1': c1_PI_values,
            'c2': c2_PI_values,
            'P_hat': P_hat_values
        }
        
        print("Partial Information HJB approximation solved.")
    
    def solve_naive_hjb(self):
        """
        Solve the approximate HJB equation for naive strategy (CJP-like).
        This assumes q_noise = 0 (no fad in price) and psi_informed = 0 (no informed traders).
        """
        print("Solving Naive (CJP-like) HJB approximation...")
        
        # Extract parameters
        T = self.params['T_horizon']
        phi_penalty = self.params['phi_penalty']
        alpha_penalty = self.params['alpha_penalty']
        k = self.params['k_intensity']
        phi_uninformed = self.params['phi_uninformed']
        mu = self.params['mu_price']
        
        # Constants for the exponential expansion
        exp_minus_1 = np.exp(-1)
        
        # Define ODE for A_CJP(t)
        def A_CJP_ode(t, A):
            return phi_penalty - (4/k) * phi_uninformed * exp_minus_1 * A**2
        
        # Solve A_CJP(t) backward from T to 0
        A_CJP_sol = solve_ivp(
            A_CJP_ode, 
            [T, 0], 
            [alpha_penalty], 
            t_eval=np.flip(self.t_grid),
            method='RK45',
            rtol=1e-8,
            atol=1e-10
        )
        
        # Extract A_CJP(t) solution (and flip to get increasing time)
        A_CJP_values = np.flip(A_CJP_sol.y[0])
        
        # Define system of ODEs for b0_CJP and c0_CJP
        def bc_CJP_system(t, y):
            b0_CJP, c0_CJP = y
            
            # Current value of A_CJP(t) - interpolate from solution
            idx = np.searchsorted(self.t_grid, t)
            if idx == len(self.t_grid):
                idx = len(self.t_grid) - 1
            A_CJP_t = A_CJP_values[idx]
            
            # ODEs for b0_CJP and c0_CJP
            db0_CJP = mu + (4/k) * phi_uninformed * exp_minus_1 * A_CJP_t * b0_CJP
            
            # The ODE for c0_CJP
            dc0_CJP = (1/k) * exp_minus_1 * (
                2 * phi_uninformed + 
                2 * k * A_CJP_t * phi_uninformed + 
                k**2 * phi_uninformed * (A_CJP_t**2 + b0_CJP**2)
            )
            
            return [db0_CJP, dc0_CJP]
        
        # Solve system backward from T to 0 with zero terminal conditions
        bc_CJP_sol = solve_ivp(
            bc_CJP_system, 
            [T, 0], 
            [0.0, 0.0], 
            t_eval=np.flip(self.t_grid),
            method='RK45',
            rtol=1e-8,
            atol=1e-10
        )
        
        # Extract solutions (and flip to get increasing time)
        b0_CJP_values = np.flip(bc_CJP_sol.y[0])
        c0_CJP_values = np.flip(bc_CJP_sol.y[1])
        
        # Store solutions
        self.solutions['CJP'] = {
            'A': A_CJP_values,
            'b0': b0_CJP_values,
            'c0': c0_CJP_values
        }
        
        print("Naive (CJP-like) HJB approximation solved.")
    
    def calculate_displacements(self, strategy, t_idx, Q_t, U_t, U_hat_t=None):
        """
        Calculate optimal bid and ask displacements for a given strategy.
        
        Parameters:
        - strategy: 'FI', 'PI', or 'CJP'
        - t_idx: Index in time grid
        - Q_t: Current inventory
        - U_t: Current fad value (for FI)
        - U_hat_t: Estimated fad value (for PI)
        
        Returns:
        - delta_a: Ask displacement
        - delta_b: Bid displacement
        """
        # Extract parameters
        k = self.params['k_intensity']
        delta_a_lower_bound = self.params['delta_a_lower_bound']
        delta_b_lower_bound = self.params['delta_b_lower_bound']
        
        # Get solution values at current time
        sol = self.solutions[strategy]
        A_t = sol['A'][t_idx]
        
        if strategy == 'FI':
            # Full information strategy
            b0_t = sol['b0'][t_idx]
            b1_t = sol['b1'][t_idx]
            
            # Calculate value function differences (Eq 18, 19)
            delta_V_ask = -2 * Q_t * A_t + A_t - (b0_t + U_t * b1_t)
            delta_V_bid = 2 * Q_t * A_t + A_t + (b0_t + U_t * b1_t)
            
        elif strategy == 'PI':
            # Partial information strategy
            b0_t = sol['b0'][t_idx]
            b1_t = sol['b1'][t_idx]
            
            # Calculate value function differences using estimated fad U_hat_t
            delta_V_ask = -2 * Q_t * A_t + A_t - (b0_t + U_hat_t * b1_t)
            delta_V_bid = 2 * Q_t * A_t + A_t + (b0_t + U_hat_t * b1_t)
            
        else:  # 'CJP'
            # Naive strategy (no fad consideration)
            b0_t = sol['b0'][t_idx]
            
            # Calculate value function differences
            delta_V_ask = -2 * Q_t * A_t + A_t - b0_t
            delta_V_bid = 2 * Q_t * A_t + A_t + b0_t
        
        # Calculate optimal displacements (Eq 18, 19)
        delta_a = max(delta_a_lower_bound, (1/k) - delta_V_ask)
        delta_b = max(delta_b_lower_bound, (1/k) - delta_V_bid)
        
        return delta_a, delta_b
    
    def simulate_market(self):
        """
        Simulate the market with the market maker following different strategies.
        """
        print("Starting market simulation...")
        
        # Extract parameters
        T = self.params['T_horizon']
        dt = self.params['dt']
        N_steps = self.params['N_steps']
        S0 = self.params['S0']
        mu = self.params['mu_price']
        sigma = self.params['sigma_price']
        q_noise = self.params['q_noise']
        p_noise = self.params['p_noise']
        eta = self.params['eta_fad']
        U0 = self.params['U0']
        Q0 = self.params['Q0']
        q_inv_min = self.params['q_inv_min']
        q_inv_max = self.params['q_inv_max']
        phi_uninformed = self.params['phi_uninformed']
        psi_informed = self.params['psi_informed']
        k = self.params['k_intensity']
        gamma = self.params['gamma_intensity']
        alpha_penalty = self.params['alpha_penalty']
        phi_penalty = self.params['phi_penalty']
        n_simulations = self.params['n_simulations']
        
        # Initialize arrays for storing results
        strategies = ['FI', 'PI', 'CJP']
        
        # Performance metrics
        performance_metrics = {
            strategy: {'final_wealth': [], 'integral_penalty': []} 
            for strategy in strategies
        }
        
        # For storing one sample path
        sample_path = {
            'time': self.t_grid,
            'S': np.zeros(N_steps + 1),
            'U': np.zeros(N_steps + 1),
            'U_hat': np.zeros(N_steps + 1),
            'F': np.zeros(N_steps + 1),  # Fundamental price
            'Q': {strategy: np.zeros(N_steps + 1) for strategy in strategies},
            'X_cash': {strategy: np.zeros(N_steps + 1) for strategy in strategies},
            'delta_a': {strategy: np.zeros(N_steps + 1) for strategy in strategies},
            'delta_b': {strategy: np.zeros(N_steps + 1) for strategy in strategies},
            'S_ask': {strategy: np.zeros(N_steps + 1) for strategy in strategies},
            'S_bid': {strategy: np.zeros(N_steps + 1) for strategy in strategies}
        }
        
        # Create progress bar
        pbar = tqdm(total=n_simulations)
        
        # Simulate multiple paths
        for sim in range(n_simulations):
            # Initialize market state
            S_t = S0
            U_t = U0
            U_hat_t = U0  # Initial estimate (known for simplicity)
            F_t = S_t - sigma * q_noise * U_t  # Fundamental price
            
            # Initialize market maker state for each strategy
            Q_t = {strategy: Q0 for strategy in strategies}
            X_cash_t = {strategy: 0.0 for strategy in strategies}
            integral_penalty = {strategy: 0.0 for strategy in strategies}
            
            # Store initial state for sample path (only for first simulation)
            if sim == 0:
                sample_path['S'][0] = S_t
                sample_path['U'][0] = U_t
                sample_path['U_hat'][0] = U_hat_t
                sample_path['F'][0] = F_t
                for strategy in strategies:
                    sample_path['Q'][strategy][0] = Q_t[strategy]
                    sample_path['X_cash'][strategy][0] = X_cash_t[strategy]
            
            # Main simulation loop
            for t_idx in range(N_steps):
                t = self.t_grid[t_idx]
                
                # Generate Brownian increments
                dZ_t = np.sqrt(dt) * np.random.normal(0, 1)
                dB_t = np.sqrt(dt) * np.random.normal(0, 1)
                
                # Update fad process U_t (Eq 3)
                U_t_next = U_t - eta * U_t * dt + dB_t
                
                # Compute combined noise increment dW_bar_t (footnote 6)
                dW_bar_t = p_noise * dZ_t + q_noise * dB_t
                
                # Update mid-price S_t (Eq 1, modified by footnote 6)
                S_t_next = S_t + (mu - eta * sigma * q_noise * U_t) * dt + sigma * dW_bar_t
                
                # Calculate fundamental price
                F_t_next = S_t_next - sigma * q_noise * U_t_next
                
                # Market maker actions for each strategy
                for strategy in strategies:
                    # Calculate optimal displacements
                    if strategy == 'FI':
                        delta_a, delta_b = self.calculate_displacements(strategy, t_idx, Q_t[strategy], U_t)
                    elif strategy == 'PI':
                        delta_a, delta_b = self.calculate_displacements(strategy, t_idx, Q_t[strategy], None, U_hat_t)
                    else:  # 'CJP'
                        delta_a, delta_b = self.calculate_displacements(strategy, t_idx, Q_t[strategy], 0)
                    
                    # Determine posted quotes
                    S_ask_t = S_t + delta_a
                    S_bid_t = S_t - delta_b
                    
                    # Store displacements and quotes for sample path
                    if sim == 0:
                        sample_path['delta_a'][strategy][t_idx] = delta_a
                        sample_path['delta_b'][strategy][t_idx] = delta_b
                        sample_path['S_ask'][strategy][t_idx] = S_ask_t
                        sample_path['S_bid'][strategy][t_idx] = S_bid_t
                    
                    # Calculate true arrival intensities
                    if Q_t[strategy] < q_inv_max:  # Check if MM can sell
                        lambda_a_true_t = phi_uninformed * np.exp(-k * delta_a) + psi_informed * np.exp(-k * delta_a + gamma * sigma * q_noise * U_t)
                    else:
                        lambda_a_true_t = 0
                    
                    if Q_t[strategy] > q_inv_min:  # Check if MM can buy
                        lambda_b_true_t = phi_uninformed * np.exp(-k * delta_b) + psi_informed * np.exp(-k * delta_b - gamma * sigma * q_noise * U_t)
                    else:
                        lambda_b_true_t = 0
                    
                    # Simulate order arrivals
                    num_ask_arrivals = np.random.poisson(lambda_a_true_t * dt)
                    num_bid_arrivals = np.random.poisson(lambda_b_true_t * dt)
                    
                    # Update inventory and cash
                    if num_ask_arrivals > 0 and Q_t[strategy] - num_ask_arrivals >= q_inv_min:
                        X_cash_t[strategy] += num_ask_arrivals * S_ask_t
                        Q_t[strategy] -= num_ask_arrivals
                    
                    if num_bid_arrivals > 0 and Q_t[strategy] + num_bid_arrivals <= q_inv_max:
                        X_cash_t[strategy] -= num_bid_arrivals * S_bid_t
                        Q_t[strategy] += num_bid_arrivals
                    
                    # Update running inventory penalty
                    integral_penalty[strategy] += phi_penalty * Q_t[strategy]**2 * dt
                    
                    # Store state for sample path
                    if sim == 0:
                        sample_path['Q'][strategy][t_idx + 1] = Q_t[strategy]
                        sample_path['X_cash'][strategy][t_idx + 1] = X_cash_t[strategy]
                
                # Update filter for PI strategy (Kalman-Bucy filter)
                # Calculate innovation (Eq 43)
                dI_t = (S_t_next - S_t) - (mu - eta * sigma * q_noise * U_hat_t) * dt
                
                # Current value of P_hat_t
                P_hat_t = self.solutions['PI']['P_hat'][t_idx]
                
                # Update U_hat_t (Eq 46)
                dU_hat_t = -eta * U_hat_t * dt + (1/sigma) * (P_hat_t * (-eta * q_noise) + q_noise) * dI_t
                U_hat_t_next = U_hat_t + dU_hat_t
                
                # Update state variables
                S_t = S_t_next
                U_t = U_t_next
                U_hat_t = U_hat_t_next
                F_t = F_t_next
                
                # Store updated state for sample path
                if sim == 0:
                    sample_path['S'][t_idx + 1] = S_t
                    sample_path['U'][t_idx + 1] = U_t
                    sample_path['U_hat'][t_idx + 1] = U_hat_t
                    sample_path['F'][t_idx + 1] = F_t
            
            # Calculate final performance for each strategy
            for strategy in strategies:
                # Final performance (Eq 8)
                final_wealth = X_cash_t[strategy] + Q_t[strategy] * S_t - alpha_penalty * Q_t[strategy]**2 - integral_penalty[strategy]
                
                # Store performance metrics
                performance_metrics[strategy]['final_wealth'].append(final_wealth)
                performance_metrics[strategy]['integral_penalty'].append(integral_penalty[strategy])
            
            # Update progress bar
            pbar.update(1)
        
        # Close progress bar
        pbar.close()
        
        # Calculate average performance metrics
        avg_performance = {}
        for strategy in strategies:
            wealth_mean = np.mean(performance_metrics[strategy]['final_wealth'])
            wealth_std = np.std(performance_metrics[strategy]['final_wealth'])
            
            avg_performance[strategy] = {
                'wealth_mean': wealth_mean,
                'wealth_std': wealth_std,
                'integral_penalty_mean': np.mean(performance_metrics[strategy]['integral_penalty']),
                'performance_metrics': performance_metrics[strategy]
            }
        
        # Print performance summary
        print("\nPerformance Summary:")
        for strategy in strategies:
            print(f"  {strategy}:")
            print(f"    Mean Final Wealth: {avg_performance[strategy]['wealth_mean']:.4f}")
            print(f"    Std Dev Final Wealth: {avg_performance[strategy]['wealth_std']:.4f}")
            print(f"    Mean Inventory Penalty: {avg_performance[strategy]['integral_penalty_mean']:.4f}")
            
        # Calculate value of information
        print("\nValue of Information:")
        for base_strategy in ['CJP', 'PI']:
            for target_strategy in ['PI', 'FI']:
                if base_strategy != target_strategy:
                    voi = avg_performance[target_strategy]['wealth_mean'] - avg_performance[base_strategy]['wealth_mean']
                    voi_percent = voi / abs(avg_performance[base_strategy]['wealth_mean']) * 100
                    print(f"  {base_strategy} → {target_strategy}: {voi:.4f} ({voi_percent:.2f}%)")
        
        # Store results
        self.results = {
            'performance_metrics': performance_metrics,
            'avg_performance': avg_performance,
            'sample_path': sample_path
        }
        
        print("Market simulation completed.")
        
        return avg_performance, sample_path
    
    def plot_sample_path(self):
        """
        Plot sample path of market variables and market maker actions.
        """
        if not hasattr(self, 'results'):
            print("No simulation results available. Run simulate_market() first.")
            return
        
        sample_path = self.results['sample_path']
        
        # Set up figure with multiple subplots
        fig, axs = plt.subplots(3, 2, figsize=(16, 12))
        fig.subplots_adjust(hspace=0.4, wspace=0.3)
        
        # Plot 1: Price processes
        axs[0, 0].plot(sample_path['time'], sample_path['S'], 'b-', label='Mid-price (S)')
        axs[0, 0].plot(sample_path['time'], sample_path['F'], 'g--', label='Fundamental (F)')
        axs[0, 0].set_title('Price Processes')
        axs[0, 0].set_xlabel('Time')
        axs[0, 0].set_ylabel('Price')
        axs[0, 0].legend()
        axs[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Fad process and estimate
        axs[0, 1].plot(sample_path['time'], sample_path['U'], 'r-', label='True Fad (U)')
        axs[0, 1].plot(sample_path['time'], sample_path['U_hat'], 'b--', label='Estimated Fad (U_hat)')
        axs[0, 1].set_title('Fad Process')
        axs[0, 1].set_xlabel('Time')
        axs[0, 1].set_ylabel('Fad Value')
        axs[0, 1].legend()
        axs[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Inventory paths
        strategies = ['FI', 'PI', 'CJP']
        colors = ['b', 'g', 'r']
        for i, strategy in enumerate(strategies):
            axs[1, 0].plot(sample_path['time'], sample_path['Q'][strategy], color=colors[i], label=strategy)
        axs[1, 0].set_title('Inventory Paths (Q)')
        axs[1, 0].set_xlabel('Time')
        axs[1, 0].set_ylabel('Inventory')
        axs[1, 0].legend()
        axs[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Cash processes
        for i, strategy in enumerate(strategies):
            axs[1, 1].plot(sample_path['time'], sample_path['X_cash'][strategy], color=colors[i], label=strategy)
        axs[1, 1].set_title('Cash Processes (X)')
        axs[1, 1].set_xlabel('Time')
        axs[1, 1].set_ylabel('Cash')
        axs[1, 1].legend()
        axs[1, 1].grid(True, alpha=0.3)
        
        # Plot 5: Ask displacements
        for i, strategy in enumerate(strategies):
            axs[2, 0].plot(sample_path['time'], sample_path['delta_a'][strategy], color=colors[i], label=strategy)
        axs[2, 0].set_title('Ask Displacements (δa)')
        axs[2, 0].set_xlabel('Time')
        axs[2, 0].set_ylabel('Displacement')
        axs[2, 0].legend()
        axs[2, 0].grid(True, alpha=0.3)
        
        # Plot 6: Bid displacements
        for i, strategy in enumerate(strategies):
            axs[2, 1].plot(sample_path['time'], sample_path['delta_b'][strategy], color=colors[i], label=strategy)
        axs[2, 1].set_title('Bid Displacements (δb)')
        axs[2, 1].set_xlabel('Time')
        axs[2, 1].set_ylabel('Displacement')
        axs[2, 1].legend()
        axs[2, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('sample_path.png', dpi=300)
        plt.close()
        
        # Plot bid-ask spreads
        plt.figure(figsize=(10, 6))
        for i, strategy in enumerate(strategies):
            spread = sample_path['delta_a'][strategy] + sample_path['delta_b'][strategy]
            plt.plot(sample_path['time'], spread, color=colors[i], label=strategy)
        plt.title('Bid-Ask Spreads')
        plt.xlabel('Time')
        plt.ylabel('Spread')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('bid_ask_spreads.png', dpi=300)
        plt.close()
        
        print("Sample path plots saved.")
    
    def plot_displacement_strategies(self):
        """
        Plot optimal ask and bid displacements as functions of fad U_t and inventory Q_t.
        Similar to Figure 1 in the paper.
        """
        if not self.solutions:
            print("No solutions available. Solve HJB equations first.")
            return
        
        # Extract parameters
        q_inv_min = self.params['q_inv_min']
        q_inv_max = self.params['q_inv_max']
        
        # Create grid of U_t and Q_t values
        U_values = np.linspace(-2, 2, 100)
        Q_values = np.linspace(q_inv_min, q_inv_max, 11)
        
        # Set up figure with multiple subplots
        fig, axs = plt.subplots(2, 3, figsize=(18, 10))
        fig.subplots_adjust(hspace=0.4, wspace=0.3)
        
        # Titles and strategies
        titles = [
            'Full Information (FI) Ask Displacement',
            'Partial Information (PI) Ask Displacement',
            'Naive (CJP) Ask Displacement',
            'Full Information (FI) Bid Displacement',
            'Partial Information (PI) Bid Displacement',
            'Naive (CJP) Bid Displacement'
        ]
        
        strategies = ['FI', 'PI', 'CJP']
        
        # Plot optimal displacements
        for col, strategy in enumerate(strategies):
            # For each inventory level
            for Q in Q_values:
                # Ask displacements (top row)
                delta_a_values = []
                for U in U_values:
                    # For PI, we assume U_hat = U for this visualization
                    if strategy == 'FI':
                        delta_a, _ = self.calculate_displacements(strategy, 0, Q, U)
                    elif strategy == 'PI':
                        delta_a, _ = self.calculate_displacements(strategy, 0, Q, None, U)
                    else:  # 'CJP'
                        delta_a, _ = self.calculate_displacements(strategy, 0, Q, 0)
                    
                    delta_a_values.append(delta_a)
                
                axs[0, col].plot(U_values, delta_a_values, label=f'Q = {Q:.1f}')
                
                # Bid displacements (bottom row)
                delta_b_values = []
                for U in U_values:
                    # For PI, we assume U_hat = U for this visualization
                    if strategy == 'FI':
                        _, delta_b = self.calculate_displacements(strategy, 0, Q, U)
                    elif strategy == 'PI':
                        _, delta_b = self.calculate_displacements(strategy, 0, Q, None, U)
                    else:  # 'CJP'
                        _, delta_b = self.calculate_displacements(strategy, 0, Q, 0)
                    
                    delta_b_values.append(delta_b)
                
                axs[1, col].plot(U_values, delta_b_values, label=f'Q = {Q:.1f}')
        
        # Set titles and labels
        for i in range(2):
            for j in range(3):
                idx = i * 3 + j
                axs[i, j].set_title(titles[idx])
                axs[i, j].set_xlabel('Fad (U)' if strategies[j] != 'CJP' else 'U (irrelevant for CJP)')
                axs[i, j].set_ylabel('Displacement')
                axs[i, j].legend(loc='upper right', fontsize=8)
                axs[i, j].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('displacement_strategies.png', dpi=300)
        plt.close()
        
        print("Displacement strategies plot saved.")
    
    def plot_performance_distribution(self):
        """
        Plot distribution of final wealth for each strategy.
        """
        if not hasattr(self, 'results'):
            print("No simulation results available. Run simulate_market() first.")
            return
        
        # Set up figure
        plt.figure(figsize=(12, 8))
        
        # Get performance metrics
        metrics = self.results['performance_metrics']
        strategies = ['FI', 'PI', 'CJP']
        colors = ['b', 'g', 'r']
        
        # Plot histograms of final wealth
        for i, strategy in enumerate(strategies):
            final_wealth = metrics[strategy]['final_wealth']
            plt.hist(final_wealth, bins=20, alpha=0.5, color=colors[i], label=strategy)
        
        # Add vertical lines for means
        for i, strategy in enumerate(strategies):
            mean_wealth = np.mean(metrics[strategy]['final_wealth'])
            plt.axvline(mean_wealth, color=colors[i], linestyle='--', linewidth=2)
            
            # Add text label for mean
            plt.text(mean_wealth, plt.ylim()[1]*0.9-i*plt.ylim()[1]*0.1, 
                     f'{strategy} Mean: {mean_wealth:.2f}', 
                     color=colors[i], horizontalalignment='center')
        
        plt.title('Distribution of Final Wealth by Strategy')
        plt.xlabel('Final Wealth')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('performance_distribution.png', dpi=300)
        plt.close()
        
        print("Performance distribution plot saved.")
    
    def run_sensitivity_analysis(self, parameter, values):
        """
        Run sensitivity analysis for a given parameter.
        
        Parameters:
        - parameter: Parameter name to vary
        - values: List of values to test
        
        Returns:
        - results: Dictionary of performance metrics for each value
        """
        print(f"Running sensitivity analysis for parameter '{parameter}'...")
        
        # Store original parameter value
        original_value = self.params[parameter]
        
        # Initialize results
        sensitivity_results = {
            'parameter': parameter,
            'values': values,
            'performance': {value: {} for value in values}
        }
        
        # Run simulations for each parameter value
        for value in tqdm(values, desc=f"Testing {parameter} values"):
            # Update parameter
            self.params[parameter] = value
            
            # Recalibrate psi_informed if needed
            if parameter in ['phi_uninformed', 'q_noise', 'gamma_intensity', 'eta_fad']:
                self.calibrate_psi_informed()
            
            # Solve HJB equations
            self.solve_full_information_hjb()
            self.solve_partial_information_hjb()
            self.solve_naive_hjb()
            
            # Run simulation with fewer paths for efficiency
            original_n_simulations = self.params['n_simulations']
            self.params['n_simulations'] = max(20, original_n_simulations // 5)
            
            # Simulate market
            avg_performance, _ = self.simulate_market()
            
            # Store results
            sensitivity_results['performance'][value] = avg_performance
            
            # Restore original n_simulations
            self.params['n_simulations'] = original_n_simulations
        
        # Restore original parameter value
        self.params[parameter] = original_value
        self.calibrate_psi_informed()  # Recalibrate
        
        # Solve HJB equations with original parameters
        self.solve_full_information_hjb()
        self.solve_partial_information_hjb()
        self.solve_naive_hjb()
        
        print(f"Sensitivity analysis for '{parameter}' completed.")
        
        # Store results
        self.sensitivity_results = sensitivity_results
        
        return sensitivity_results
    
    def plot_sensitivity_analysis(self):
        """
        Plot results of sensitivity analysis.
        """
        if not hasattr(self, 'sensitivity_results'):
            print("No sensitivity analysis results available. Run run_sensitivity_analysis() first.")
            return
        
        # Extract results
        results = self.sensitivity_results
        parameter = results['parameter']
        values = results['values']
        
        # Set up figure
        plt.figure(figsize=(12, 8))
        
        # Extract wealth means for each strategy
        strategies = ['FI', 'PI', 'CJP']
        colors = ['b', 'g', 'r']
        markers = ['o', 's', '^']
        
        for i, strategy in enumerate(strategies):
            wealth_means = [results['performance'][value][strategy]['wealth_mean'] for value in values]
            
            # Convert to percentage improvement over CJP
            if strategy != 'CJP':
                baseline = [results['performance'][value]['CJP']['wealth_mean'] for value in values]
                relative_improvement = [(wm - b) / abs(b) * 100 for wm, b in zip(wealth_means, baseline)]
                plt.plot(values, relative_improvement, color=colors[i], marker=markers[i], label=f"{strategy} vs CJP")
        
        # Also plot FI vs PI
        fi_means = [results['performance'][value]['FI']['wealth_mean'] for value in values]
        pi_means = [results['performance'][value]['PI']['wealth_mean'] for value in values]
        relative_improvement_fi_pi = [(f - p) / abs(p) * 100 for f, p in zip(fi_means, pi_means)]
        plt.plot(values, relative_improvement_fi_pi, color='purple', marker='x', label="FI vs PI")
        
        # Format x-axis
        if max(values) / min(values) > 100:
            plt.xscale('log')
        
        # Add labels and title
        plt.xlabel(parameter)
        plt.ylabel('Percentage Improvement (%)')
        plt.title(f'Sensitivity Analysis: Impact of {parameter} on Strategy Performance')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Format y-axis as percentage
        plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.1f%%'))
        
        plt.tight_layout()
        plt.savefig(f'sensitivity_{parameter}.png', dpi=300)
        plt.close()
        
        # Also plot absolute performance
        plt.figure(figsize=(12, 8))
        
        for i, strategy in enumerate(strategies):
            wealth_means = [results['performance'][value][strategy]['wealth_mean'] for value in values]
            plt.plot(values, wealth_means, color=colors[i], marker=markers[i], label=strategy)
        
        # Format x-axis
        if max(values) / min(values) > 100:
            plt.xscale('log')
        
        # Add labels and title
        plt.xlabel(parameter)
        plt.ylabel('Mean Final Wealth')
        plt.title(f'Sensitivity Analysis: Absolute Performance vs {parameter}')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f'sensitivity_absolute_{parameter}.png', dpi=300)
        plt.close()
        
        print(f"Sensitivity analysis plots for '{parameter}' saved.")
    
    def run_full_analysis(self):
        """
        Run the full analysis pipeline: solve HJB equations, simulate market, and create plots.
        """
        # Solve HJB equations
        self.solve_full_information_hjb()
        self.solve_partial_information_hjb()
        self.solve_naive_hjb()
        
        # Simulate market
        self.simulate_market()
        
        # Create plots
        self.plot_sample_path()
        self.plot_displacement_strategies()
        self.plot_performance_distribution()
        
        # Run sensitivity analyses
        parameters_to_analyze = {
            'q_noise': np.linspace(0.1, 0.9, 5),
            'eta_fad': np.logspace(0, 2, 5),
            'gamma_intensity': np.logspace(-1, 1, 5),
            'phi_uninformed': np.linspace(5, 25, 5)
        }
        
        for param, values in parameters_to_analyze.items():
            self.run_sensitivity_analysis(param, values)
            self.plot_sensitivity_analysis()
        
        print("Full analysis completed.")

# Run the analysis
if __name__ == "__main__":
    # Initialize model with default parameters
    model = MarketMakingWithFads()
    
    # Run full analysis
    model.run_full_analysis()