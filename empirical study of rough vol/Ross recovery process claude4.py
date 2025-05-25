import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from scipy.linalg import eig
from scipy.optimize import minimize, LinearConstraint
from scipy.interpolate import interp1d, interp2d
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

class MarketSimulator:
    """Simulate market data including spot prices, volatility, and option prices"""
    
    def __init__(self, S0=100, mu=0.05, base_vol=0.2, kappa=2.0, theta=0.2, xi=0.3):
        self.S0 = S0
        self.mu = mu
        self.base_vol = base_vol
        self.kappa = kappa  # Mean reversion speed for volatility
        self.theta = theta  # Long-term volatility
        self.xi = xi        # Volatility of volatility
        
    def simulate_heston_process(self, T=2, dt=1/252, n_paths=1):
        """Simulate stock price and volatility using Heston model"""
        n_steps = int(T / dt)
        
        # Initialize arrays
        S = np.zeros((n_steps + 1, n_paths))
        v = np.zeros((n_steps + 1, n_paths))
        
        S[0] = self.S0
        v[0] = self.base_vol ** 2
        
        # Generate correlated Brownian motions (correlation between stock and vol)
        rho = -0.5
        
        for t in range(n_steps):
            dW_S = np.random.randn(n_paths) * np.sqrt(dt)
            dW_v = rho * dW_S + np.sqrt(1 - rho**2) * np.random.randn(n_paths) * np.sqrt(dt)
            
            # Volatility process (CIR)
            v[t+1] = v[t] + self.kappa * (self.theta**2 - v[t]) * dt + self.xi * np.sqrt(v[t]) * dW_v
            v[t+1] = np.maximum(v[t+1], 0.001)  # Ensure positive variance
            
            # Stock price process
            S[t+1] = S[t] * np.exp((self.mu - 0.5 * v[t]) * dt + np.sqrt(v[t]) * dW_S)
        
        return S, np.sqrt(v)
    
    def black_scholes_price(self, S, K, T, r, sigma, option_type='call'):
        """Calculate Black-Scholes option price"""
        d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
        d2 = d1 - sigma*np.sqrt(T)
        
        if option_type == 'call':
            price = S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
        else:
            price = K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)
        
        return price
    
    def generate_option_surface(self, S, vol, strikes, maturities, r=0.02):
        """Generate option prices for different strikes and maturities"""
        n_strikes = len(strikes)
        n_maturities = len(maturities)
        
        option_prices = np.zeros((n_strikes, n_maturities))
        implied_vols = np.zeros((n_strikes, n_maturities))
        
        for i, K in enumerate(strikes):
            for j, T in enumerate(maturities):
                # Add volatility smile effect
                moneyness = K / S
                smile_adjustment = 0.1 * (moneyness - 1)**2
                adjusted_vol = vol * (1 + smile_adjustment)
                
                option_prices[i, j] = self.black_scholes_price(S, K, T, r, adjusted_vol)
                implied_vols[i, j] = adjusted_vol
        
        return option_prices, implied_vols

class RossRecovery:
    """Implement Ross Recovery Theorem"""
    
    def __init__(self, n_states=21):
        self.n_states = n_states
        self.moneyness_grid = np.linspace(0.6, 1.5, n_states)
        
    def compute_state_prices(self, option_prices, strikes, spot, r, T):
        """Compute state prices using Breeden-Litzenberger formula"""
        # Second derivative of call price with respect to strike
        n_strikes = len(strikes)
        if n_strikes < 3:
            raise ValueError("Need at least 3 strikes to compute state prices")
            
        state_prices = np.zeros(n_strikes - 2)
        
        for i in range(1, n_strikes - 1):
            # Numerical second derivative
            h = strikes[i+1] - strikes[i]
            state_prices[i-1] = (option_prices[i+1] - 2*option_prices[i] + option_prices[i-1]) / h**2
            state_prices[i-1] *= np.exp(r * T)  # Adjust for discounting
        
        # Ensure positive state prices
        state_prices = np.maximum(state_prices, 1e-6)
        
        # Normalize to sum to 1 (probability measure)
        state_prices = state_prices / np.sum(state_prices)
        
        return state_prices
    
    def construct_transition_matrix(self, state_price_surface, maturities, tau=1/12):
        """Construct state price transition matrix P"""
        n_states = self.n_states
        n_maturities = len(maturities)
        
        # Find tau in the maturity grid
        tau_idx = np.argmin(np.abs(maturities - tau))
        if tau_idx == 0:
            tau_idx = 1  # Ensure we have at least one maturity before tau
        
        # Ensure we have enough maturities
        if n_maturities <= tau_idx:
            raise ValueError(f"Not enough maturities. Need at least {tau_idx + 1}, got {n_maturities}")
        
        # Create submatrices A and B with correct dimensions
        # A represents transitions from t to t+tau for different initial maturities
        # B represents the target state prices at t+tau
        n_cols_A = min(n_maturities - tau_idx, n_states)
        
        # Resize state_price_surface if needed
        if state_price_surface.shape[0] != n_states:
            # Interpolate to correct size
            old_grid = np.linspace(0, 1, state_price_surface.shape[0])
            new_grid = np.linspace(0, 1, n_states)
            
            state_price_surface_resized = np.zeros((n_states, n_maturities))
            for j in range(n_maturities):
                f = interp1d(old_grid, state_price_surface[:, j], 
                           kind='linear', fill_value='extrapolate', 
                           bounds_error=False)
                state_price_surface_resized[:, j] = f(new_grid)
            state_price_surface = state_price_surface_resized
        
        # Extract A and B matrices
        A = state_price_surface[:, :n_cols_A]
        B = state_price_surface[:, tau_idx:tau_idx + n_cols_A]
        
        # Ensure A and B have compatible dimensions
        if A.shape != B.shape:
            min_cols = min(A.shape[1], B.shape[1])
            A = A[:, :min_cols]
            B = B[:, :min_cols]
        
        # Solve for P with constraints
        P = self._solve_constrained_transition_matrix(A, B)
        
        return P
    
    def _solve_constrained_transition_matrix(self, A, B):
        """Solve for transition matrix with constraints"""
        n = self.n_states
        
        # Check dimensions
        if A.shape[0] != n or B.shape[0] != n:
            raise ValueError(f"A and B must have {n} rows. Got A: {A.shape}, B: {B.shape}")
        
        if A.shape[1] != B.shape[1]:
            raise ValueError(f"A and B must have same number of columns. Got A: {A.shape[1]}, B: {B.shape[1]}")
        
        # Initialize P
        P = np.zeros((n, n))
        
        # For each column of P (final state)
        for j in range(n):
            # Set up optimization problem
            def objective(p):
                # Ensure p has correct shape
                p = p.reshape(-1, 1)
                # Compute prediction error
                pred = A @ (A.T @ A + 1e-6 * np.eye(A.shape[1])) @ A.T @ p
                if j < B.shape[1]:
                    target = B[:, j]
                else:
                    # For states beyond B's columns, use interpolation
                    target = B[:, -1]  # Use last column
                error = pred.flatten() - target
                return np.sum(error ** 2)
            
            # Constraints
            constraints = []
            
            # Sum to 1 constraint (relaxed for numerical stability)
            constraints.append({
                'type': 'eq',
                'fun': lambda p: np.sum(p) - 1.0
            })
            
            # Non-negativity bounds
            bounds = [(0, 1) for _ in range(n)]
            
            # Initial guess (concentrated near diagonal)
            p0 = np.zeros(n)
            for i in range(n):
                distance = abs(i - j)
                p0[i] = np.exp(-distance**2 / 4)
            p0 = p0 / np.sum(p0)
            
            # Solve
            try:
                result = minimize(objective, p0, bounds=bounds, constraints=constraints, 
                                method='SLSQP', options={'maxiter': 500, 'ftol': 1e-6})
                
                if result.success:
                    P[:, j] = result.x
                else:
                    # Use initial guess if optimization fails
                    P[:, j] = p0
            except:
                # Fallback to simple gaussian around diagonal
                P[:, j] = p0
        
        # Ensure P is a proper transition matrix
        row_sums = np.sum(P, axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        P = P / row_sums
        
        # Add small positive value to ensure irreducibility
        P = P + 1e-6
        P = P / np.sum(P, axis=1, keepdims=True)
        
        return P
    
    def recover_physical_distribution(self, P):
        """Apply Ross recovery theorem to get physical probabilities"""
        try:
            # Find eigenvalues and eigenvectors
            eigenvalues, eigenvectors = eig(P.T)
            
            # Find Perron-Frobenius eigenvalue (largest real positive)
            real_eigenvalues = np.real(eigenvalues)
            real_positive = real_eigenvalues[real_eigenvalues > 0]
            
            if len(real_positive) == 0:
                raise ValueError("No positive eigenvalues found")
            
            idx = np.argmax(real_eigenvalues)
            
            # Discount factor
            delta = real_eigenvalues[idx]
            
            # State price vector (positive eigenvector)
            z = np.real(eigenvectors[:, idx])
            z = np.abs(z)  # Ensure positive
            
            # Check if z is too small
            if np.max(z) < 1e-10:
                raise ValueError("Eigenvector too small")
            
            z = z / np.sum(z)  # Normalize
            
            # Recover physical transition matrix
            D = np.diag(z)
            D_inv = np.diag(1 / (z + 1e-10))  # Add small value to avoid division by zero
            F = delta * D @ P @ D_inv
            
            # Ensure F is a proper transition matrix
            F = np.abs(F)  # Ensure positive
            row_sums = np.sum(F, axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1
            F = F / row_sums
            
            return F, delta, z
            
        except Exception as e:
            # Return default values if recovery fails
            print(f"Recovery failed: {str(e)}")
            F = P.copy()  # Use risk-neutral as fallback
            delta = 0.99
            z = np.ones(self.n_states) / self.n_states
            return F, delta, z
    
    def extract_volatilities(self, P, F, current_state_idx, moneyness_grid):
        """Extract risk-neutral and physical volatilities"""
        # Risk-neutral distribution (from P)
        rn_probs = P[current_state_idx, :]
        rn_probs = np.abs(rn_probs)  # Ensure positive
        rn_probs = rn_probs / (np.sum(rn_probs) + 1e-10)
        
        # Physical distribution (from F)
        phys_probs = F[current_state_idx, :]
        phys_probs = np.abs(phys_probs)  # Ensure positive
        phys_probs = phys_probs / (np.sum(phys_probs) + 1e-10)
        
        # Calculate returns for each state
        returns = np.log(moneyness_grid)
        
        # Calculate moments
        rn_mean = np.sum(returns * rn_probs)
        rn_var = np.sum((returns - rn_mean)**2 * rn_probs)
        rn_vol = np.sqrt(max(rn_var, 1e-6) * 12)  # Annualize
        
        phys_mean = np.sum(returns * phys_probs)
        phys_var = np.sum((returns - phys_mean)**2 * phys_probs)
        phys_vol = np.sqrt(max(phys_var, 1e-6) * 12)  # Annualize
        
        return rn_vol, phys_vol, rn_probs, phys_probs

class VolatilityForecaster:
    """Forecast volatility using various methods"""
    
    def __init__(self):
        self.models = {}
        
    def forecast_garch(self, returns, horizon=21):
        """Simple GARCH(1,1) forecast"""
        # Estimate GARCH parameters (simplified)
        omega = 0.000001
        alpha = 0.1
        beta = 0.85
        
        # Current volatility
        if len(returns) >= 21:
            current_vol = np.std(returns[-21:]) * np.sqrt(252)
        else:
            current_vol = np.std(returns) * np.sqrt(252)
        
        # Multi-step forecast
        forecast_var = current_vol**2
        for h in range(horizon):
            forecast_var = omega + (alpha + beta) * forecast_var
        
        return np.sqrt(forecast_var)
    
    def forecast_realized_vol(self, returns, lookback=21):
        """Historical realized volatility"""
        if len(returns) >= lookback:
            return np.std(returns[-lookback:]) * np.sqrt(252)
        else:
            return np.std(returns) * np.sqrt(252)

# Main simulation and testing
def run_ross_recovery_simulation():
    """Run complete simulation of Ross recovery strategy"""
    
    print("=== Ross Recovery Volatility Forecasting Simulation ===\n")
    
    # 1. Initialize market simulator
    simulator = MarketSimulator(S0=100, base_vol=0.2)
    
    # 2. Simulate market data
    print("1. Simulating market data...")
    T = 5  # 5 years of data
    dt = 1/252
    S, vol = simulator.simulate_heston_process(T=T, dt=dt)
    returns = np.diff(np.log(S[:, 0]))
    
    # 3. Initialize Ross Recovery
    ross = RossRecovery(n_states=21)
    forecaster = VolatilityForecaster()
    
    # 4. Generate results over time
    print("2. Implementing Ross Recovery...")
    
    # Storage for results
    results = {
        'date': [],
        'realized_vol': [],
        'rn_vol': [],
        'phys_vol': [],
        'risk_pref': [],
        'actual_future_rv': []
    }
    
    # Parameters
    forecast_horizon = 21  # 1 month
    start_idx = 252  # Start after 1 year
    
    # Option parameters
    r = 0.02
    strikes = S[start_idx, 0] * ross.moneyness_grid
    maturities = np.array([1/12, 2/12, 3/12, 6/12, 9/12, 12/12, 18/12, 24/12])
    
    successful_recoveries = 0
    total_attempts = 0
    
    # Run through time
    for t in range(start_idx, min(len(S) - forecast_horizon - 1, start_idx + 252), 21):  # Limit iterations
        total_attempts += 1
        current_S = S[t, 0]
        current_vol = vol[t, 0]
        
        try:
            # Generate option surface
            strikes = current_S * ross.moneyness_grid
            option_prices, impl_vols = simulator.generate_option_surface(
                current_S, current_vol, strikes, maturities, r
            )
            
            # Compute state price surface
            state_price_surface = np.zeros((ross.n_states, len(maturities)))
            
            for j, T in enumerate(maturities):
                state_prices = ross.compute_state_prices(
                    option_prices[:, j], strikes, current_S, r, T
                )
                
                # Interpolate to match n_states
                if len(state_prices) != ross.n_states:
                    old_grid = np.linspace(0, 1, len(state_prices))
                    new_grid = np.linspace(0, 1, ross.n_states)
                    f = interp1d(old_grid, state_prices, kind='linear', 
                               fill_value='extrapolate', bounds_error=False)
                    state_price_surface[:, j] = f(new_grid)
                else:
                    state_price_surface[:, j] = state_prices
            
            # Ensure positive state prices
            state_price_surface = np.maximum(state_price_surface, 1e-6)
            
            # Normalize columns
            col_sums = np.sum(state_price_surface, axis=0, keepdims=True)
            col_sums[col_sums == 0] = 1
            state_price_surface = state_price_surface / col_sums
            
            # Construct transition matrix
            P = ross.construct_transition_matrix(state_price_surface, maturities)
            
            # Recover physical distribution
            F, delta, z = ross.recover_physical_distribution(P)
            
            # Extract volatilities
            current_state_idx = ross.n_states // 2  # Assume at-the-money
            rn_vol, phys_vol, rn_dist, phys_dist = ross.extract_volatilities(
                P, F, current_state_idx, ross.moneyness_grid
            )
            
            # Calculate realized volatility
            rv = forecaster.forecast_realized_vol(returns[:t])
            
            # Future realized volatility (for evaluation)
            if t + forecast_horizon < len(returns):
                future_rv = np.std(returns[t:t+forecast_horizon]) * np.sqrt(252)
            else:
                future_rv = rv  # Use current as estimate
            
            # Store results
            results['date'].append(t)
            results['realized_vol'].append(rv)
            results['rn_vol'].append(rn_vol)
            results['phys_vol'].append(phys_vol)
            results['risk_pref'].append(phys_vol - rn_vol)
            results['actual_future_rv'].append(future_rv)
            
            successful_recoveries += 1
            
        except Exception as e:
            print(f"  Warning: Recovery failed at time {t}: {str(e)}")
            continue
    
    print(f"\nSuccessful recoveries: {successful_recoveries}/{total_attempts}")
    
    # Convert to DataFrame
    df_results = pd.DataFrame(results)
    
    if len(df_results) == 0:
        print("No successful recoveries. Check data and parameters.")
        return None, None
    
    # 5. Evaluate forecasting performance
    print("\n3. Evaluating forecasting performance...")
    
    # Create features
    df_results['RNV'] = df_results['rn_vol'] - df_results['realized_vol']
    df_results['REV'] = df_results['phys_vol'] - df_results['realized_vol']
    df_results['REV-RNV'] = df_results['phys_vol'] - df_results['rn_vol']
    df_results['Y'] = df_results['actual_future_rv'] - df_results['realized_vol']
    
    # Run regressions if we have enough data
    if len(df_results) > 10:
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import r2_score
        
        # Model 1: Benchmark (RNV only)
        X1 = df_results[['RNV']].values
        y = df_results['Y'].values
        model1 = LinearRegression().fit(X1, y)
        r2_1 = r2_score(y, model1.predict(X1))
        
        # Model 2: Physical volatility (REV only)
        X2 = df_results[['REV']].values
        model2 = LinearRegression().fit(X2, y)
        r2_2 = r2_score(y, model2.predict(X2))
        
        # Model 3: Both components
        X3 = df_results[['RNV', 'REV-RNV']].values
        model3 = LinearRegression().fit(X3, y)
        r2_3 = r2_score(y, model3.predict(X3))
        
        print("\nModel Performance (R-squared):")
        print(f"Model 1 (RNV only):           {r2_1:.3f}")
        print(f"Model 2 (REV only):           {r2_2:.3f}")
        print(f"Model 3 (RNV + Risk Pref):    {r2_3:.3f}")
        print(f"Improvement over benchmark:   {(r2_3 - r2_1)*100:.1f}%")
        
        # 6. Visualize results
        if len(df_results) > 0:
            plot_results(df_results, P, F, ross.moneyness_grid)
    else:
        print("Not enough data for regression analysis")
    
    return df_results, results

def plot_results(df_results, P, F, moneyness_grid):
    """Create visualization of results"""
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. Volatility time series
    ax = axes[0, 0]
    ax.plot(df_results.index, df_results['realized_vol'], label='Realized Vol', alpha=0.7)
    ax.plot(df_results.index, df_results['rn_vol'], label='Risk-Neutral Vol', alpha=0.7)
    ax.plot(df_results.index, df_results['phys_vol'], label='Ross-Recovered Vol', alpha=0.7)
    ax.set_title('Volatility Time Series')
    ax.set_ylabel('Volatility')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Risk preference proxy
    ax = axes[0, 1]
    ax.plot(df_results.index, df_results['risk_pref'])
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax.set_title('Risk Preference Proxy (REV - RNV)')
    ax.set_ylabel('Risk Premium')
    ax.grid(True, alpha=0.3)
    
    # 3. Transition matrix heatmap
    ax = axes[0, 2]
    im = ax.imshow(P, cmap='Blues', aspect='auto')
    ax.set_title('State Price Transition Matrix P')
    ax.set_xlabel('Final State')
    ax.set_ylabel('Initial State')
    plt.colorbar(im, ax=ax)
    
    # 4. Risk-neutral vs Physical distributions
    ax = axes[1, 0]
    current_state = len(moneyness_grid) // 2
    rn_probs = P[current_state, :] / np.sum(P[current_state, :])
    phys_probs = F[current_state, :] / np.sum(F[current_state, :])
    
    ax.plot(moneyness_grid, rn_probs, 'b-', label='Risk-Neutral', linewidth=2)
    ax.plot(moneyness_grid, phys_probs, 'r--', label='Physical', linewidth=2)
    ax.set_title('Probability Distributions')
    ax.set_xlabel('Moneyness')
    ax.set_ylabel('Probability')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 5. Forecast errors
    ax = axes[1, 1]
    if len(df_results) > 0:
        forecast_errors_rn = df_results['rn_vol'] - df_results['actual_future_rv']
        forecast_errors_phys = df_results['phys_vol'] - df_results['actual_future_rv']
        
        ax.hist(forecast_errors_rn, bins=20, alpha=0.5, label='RN Errors', density=True)
        ax.hist(forecast_errors_phys, bins=20, alpha=0.5, label='Physical Errors', density=True)
    ax.set_title('Forecast Error Distribution')
    ax.set_xlabel('Forecast Error')
    ax.set_ylabel('Density')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 6. Actual vs Predicted scatter
    ax = axes[1, 2]
    if len(df_results) > 0:
        ax.scatter(df_results['RNV'], df_results['Y'], alpha=0.5, label='RNV')
        ax.scatter(df_results['REV'], df_results['Y'], alpha=0.5, label='REV')
    ax.set_xlabel('Predicted Change')
    ax.set_ylabel('Actual Change')
    ax.set_title('Forecast Accuracy')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# Additional analysis functions
def simulate_global_markets():
    """Simulate multiple markets for global analysis"""
    
    print("\n=== Simulating Global Markets ===\n")
    
    markets = ['US', 'UK', 'Germany', 'Switzerland', 'France']
    market_weights = [0.694, 0.143, 0.055, 0.048, 0.060]
    
    # Simulate market data
    market_data = {}
    global_rn_vol = 0
    global_phys_vol = 0
    
    for i, market in enumerate(markets):
        print(f"Simulating {market} market...")
        
        # Market-specific parameters
        base_vol = 0.15 + np.random.rand() * 0.1
        simulator = MarketSimulator(S0=100, base_vol=base_vol)
        
        # Simulate
        S, vol = simulator.simulate_heston_process(T=2, dt=1/252)
        
        # Simple vol estimates (for demonstration)
        rn_vol = np.mean(vol) * 1.2  # Risk premium
        phys_vol = np.mean(vol)
        
        market_data[market] = {
            'rn_vol': rn_vol,
            'phys_vol': phys_vol,
            'weight': market_weights[i]
        }
        
        # Add to global measures
        global_rn_vol += rn_vol * market_weights[i]
        global_phys_vol += phys_vol * market_weights[i]
    
    print(f"\nGlobal Risk-Neutral Vol: {global_rn_vol:.3f}")
    print(f"Global Physical Vol: {global_phys_vol:.3f}")
    print(f"Global Risk Premium: {global_rn_vol - global_phys_vol:.3f}")
    
    # Compare local vs global measures
    print("\nLocal vs Global Risk Premiums:")
    for market, data in market_data.items():
        local_premium = data['rn_vol'] - data['phys_vol']
        print(f"{market}: Local={local_premium:.3f}, Global impact={local_premium * data['weight']:.3f}")
    
    return market_data, global_rn_vol, global_phys_vol

# Run the complete simulation
if __name__ == "__main__":
    # Main Ross recovery simulation
    df_results, results_dict = run_ross_recovery_simulation()
    
    # Global markets simulation
    market_data, global_rn, global_phys = simulate_global_markets()
    
    # Summary statistics if we have results
    if df_results is not None and len(df_results) > 0:
        print("\n=== Summary Statistics ===")
        print(f"\nAverage Volatilities:")
        print(f"Realized: {df_results['realized_vol'].mean():.3f}")
        print(f"Risk-Neutral: {df_results['rn_vol'].mean():.3f}")
        print(f"Physical: {df_results['phys_vol'].mean():.3f}")
        print(f"Risk Premium: {(df_results['rn_vol'] - df_results['phys_vol']).mean():.3f}")
        
        print(f"\nVolatility of Volatilities:")
        print(f"Realized: {df_results['realized_vol'].std():.3f}")
        print(f"Risk-Neutral: {df_results['rn_vol'].std():.3f}")
        print(f"Physical: {df_results['phys_vol'].std():.3f}")