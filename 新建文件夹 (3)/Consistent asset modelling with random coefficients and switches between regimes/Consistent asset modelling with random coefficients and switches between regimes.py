import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.integrate import quad
from tqdm import tqdm

# Set random seed for reproducibility
np.random.seed(42)

class RandomizedAssetModel:
    """
    Implementation of the asset model with random coefficients and regime switches
    as described in the paper.
    """
    def __init__(self, S0, r, 
                 regime_means, regime_std_devs, 
                 switching_times=None, 
                 stochastic_switching=False, 
                 switching_rates=None,
                 num_quadrature_points=7):
        """
        Initialize the model with parameters.
        
        Parameters:
        - S0: Initial asset price
        - r: Risk-free rate
        - regime_means: List of mean values for volatility in each regime
        - regime_std_devs: List of standard deviations for volatility in each regime
        - switching_times: List of times when regime switches occur (deterministic case)
        - stochastic_switching: Boolean indicating if switching times are stochastic
        - switching_rates: Rates for exponential distribution of sojourn times
        - num_quadrature_points: Number of points for Gauss quadrature
        """
        self.S0 = S0
        self.r = r
        self.regime_means = regime_means
        self.regime_std_devs = regime_std_devs
        self.num_regimes = len(regime_means)
        self.switching_times = switching_times if switching_times else [np.inf]
        self.stochastic_switching = stochastic_switching
        self.switching_rates = switching_rates if switching_rates else [2.0] * (self.num_regimes - 1)
        self.num_quadrature_points = num_quadrature_points
        
        # Compute quadrature weights and points for each regime
        self.quad_weights = []
        self.quad_points = []
        for i in range(self.num_regimes):
            weights, points = self._compute_gaussian_quadrature(
                self.regime_means[i], self.regime_std_devs[i], self.num_quadrature_points)
            self.quad_weights.append(weights)
            self.quad_points.append(points)
    
    def _compute_gaussian_quadrature(self, mean, std, n_points):
        """
        Compute Gaussian quadrature weights and points for normal distribution.
        
        This is a simplified implementation based on the Golub-Welsch algorithm.
        For a proper implementation, one would use orthogonal polynomials and recursion.
        """
        # For normal distribution, we can use Gauss-Hermite quadrature
        # But we need to adjust it for our specific normal distribution
        from scipy.special import roots_hermite
        
        # Get standard Gauss-Hermite quadrature points and weights
        x, w = roots_hermite(n_points)
        
        # Adjust for our normal distribution N(mean, std^2)
        points = np.sqrt(2) * std * x + mean
        weights = w / np.sqrt(np.pi)
        
        return weights, points
    
    def _drift(self, vol, t):
        """Calculate risk-neutral drift for Black-Scholes model"""
        return self.r - 0.5 * vol**2
    
    def simulate_randomized_paths(self, T, dt, num_paths, include_local_vol=True):
        """
        Simulate paths from the randomized model with regime switches.
        
        Parameters:
        - T: Time horizon
        - dt: Time step
        - num_paths: Number of paths to simulate
        - include_local_vol: Whether to include local volatility model paths
        
        Returns:
        - Dictionary containing time points and simulated paths
        """
        times = np.arange(0, T+dt, dt)
        num_steps = len(times)
        
        # Arrays to store paths
        randomized_paths = np.zeros((num_paths, num_steps))
        randomized_paths[:, 0] = self.S0
        
        local_vol_paths = np.zeros((num_paths, num_steps))
        local_vol_paths[:, 0] = self.S0
        
        # For each path
        for path in range(num_paths):
            # Generate Brownian motion for the entire path
            dW = np.random.normal(0, np.sqrt(dt), num_steps-1)
            
            # For randomized model, sample volatility parameters for each regime
            regime_vols = []
            for i in range(self.num_regimes):
                vol = np.random.normal(self.regime_means[i], self.regime_std_devs[i])
                regime_vols.append(max(0.001, vol))  # Ensure positive volatility
            
            # Determine switching times
            if self.stochastic_switching:
                switching_times = self._generate_stochastic_switching_times(T)
            else:
                switching_times = self.switching_times
            
            # Simulate the randomized path
            current_regime = 0
            for i in range(1, num_steps):
                t = times[i-1]
                
                # Check if we need to switch regime
                while current_regime < len(switching_times) and t >= switching_times[current_regime]:
                    current_regime += 1
                
                # Get volatility for current regime
                vol = regime_vols[min(current_regime, self.num_regimes-1)]
                
                # Calculate drift
                drift = self._drift(vol, t)
                
                # Update path
                randomized_paths[path, i] = randomized_paths[path, i-1] * np.exp(
                    drift * dt + vol * dW[i-1])
            
            # For local volatility model, use quadrature approximation
            if include_local_vol:
                for i in range(1, num_steps):
                    t = times[i-1]
                    
                    # Determine current regime
                    current_regime = 0
                    while current_regime < len(switching_times) and t >= switching_times[current_regime]:
                        current_regime += 1
                    regime_idx = min(current_regime, self.num_regimes-1)
                    
                    # Compute local volatility using quadrature
                    local_vol = self._compute_local_volatility(local_vol_paths[path, i-1], t, regime_idx)
                    
                    # Calculate drift (risk-neutral)
                    local_drift = self.r - 0.5 * local_vol**2
                    
                    # Update path
                    local_vol_paths[path, i] = local_vol_paths[path, i-1] * np.exp(
                        local_drift * dt + local_vol * dW[i-1])
        
        return {
            'times': times,
            'randomized_paths': randomized_paths,
            'local_vol_paths': local_vol_paths if include_local_vol else None
        }
    
    def _generate_stochastic_switching_times(self, T):
        """Generate stochastic switching times based on exponential distributions"""
        switching_times = []
        current_time = 0
        
        for rate in self.switching_rates:
            # Generate sojourn time from exponential distribution
            sojourn_time = np.random.exponential(1/rate)
            current_time += sojourn_time
            
            # If we've exceeded T, stop
            if current_time > T:
                break
                
            switching_times.append(current_time)
        
        return switching_times
    
    def _compute_local_volatility(self, S, t, regime_idx):
        """
        Compute local volatility using quadrature approximation.
        
        This is a simplified implementation of the local volatility formula in the paper.
        For a full implementation, one would need to solve the Fokker-Planck equation.
        """
        weights = self.quad_weights[regime_idx]
        points = self.quad_points[regime_idx]
        
        # Compute weighted average of squared volatilities
        weighted_vol_squared = np.sum(weights * (points**2))
        
        return np.sqrt(weighted_vol_squared)
    
    def price_european_call(self, K, T, method='quadrature'):
        """
        Price a European call option using the model.
        
        Parameters:
        - K: Strike price
        - T: Time to maturity
        - method: Pricing method ('quadrature' or 'monte_carlo')
        
        Returns:
        - Option price
        """
        if method == 'quadrature':
            # Price using characteristic function and quadrature
            return self._price_european_call_quadrature(K, T)
        elif method == 'monte_carlo':
            # Price using Monte Carlo
            return self._price_european_call_monte_carlo(K, T)
        else:
            raise ValueError(f"Unknown pricing method: {method}")
    
    def _price_european_call_quadrature(self, K, T, num_simulations=10000):
        """Price European call option using quadrature approximation"""
        # This is a simplified version using Monte Carlo for the quadrature approximation
        # In the paper, they use the characteristic function and the COS method
        
        # Initialize total price
        total_price = 0
        
        # Loop through regimes
        for regime_idx in range(self.num_regimes):
            weights = self.quad_weights[regime_idx]
            points = self.quad_points[regime_idx]
            
            # For each quadrature point
            for i in range(len(weights)):
                vol = points[i]
                weight = weights[i]
                
                # Black-Scholes formula for this volatility
                d1 = (np.log(self.S0/K) + (self.r + 0.5 * vol**2) * T) / (vol * np.sqrt(T))
                d2 = d1 - vol * np.sqrt(T)
                bs_price = self.S0 * norm.cdf(d1) - K * np.exp(-self.r * T) * norm.cdf(d2)
                
                # Add to total price
                total_price += weight * bs_price
        
        return total_price
    
    def _price_european_call_monte_carlo(self, K, T, num_paths=10000, dt=0.01):
        """Price European call option using Monte Carlo simulation"""
        sim_results = self.simulate_randomized_paths(T, dt, num_paths)
        
        # Get terminal stock prices
        terminal_prices = sim_results['randomized_paths'][:, -1]
        
        # Calculate payoffs
        payoffs = np.maximum(terminal_prices - K, 0)
        
        # Discount payoffs
        option_price = np.exp(-self.r * T) * np.mean(payoffs)
        
        return option_price
    
    def compute_implied_volatility_surface(self, strikes, maturities):
        """
        Compute implied volatility surface for a range of strikes and maturities.
        
        Parameters:
        - strikes: Array of strike prices
        - maturities: Array of maturities
        
        Returns:
        - DataFrame with implied volatilities
        """
        iv_data = []
        
        for T in tqdm(maturities, desc="Computing IV Surface"):
            for K in strikes:
                # Price option
                price = self.price_european_call(K, T)
                
                # Compute implied volatility
                iv = self._implied_volatility(price, K, T)
                
                iv_data.append({
                    'Maturity': T,
                    'Strike': K,
                    'IV': iv
                })
        
        return pd.DataFrame(iv_data)
    
    def _implied_volatility(self, price, K, T):
        """
        Compute implied volatility for a given option price.
        
        Parameters:
        - price: Option price
        - K: Strike price
        - T: Time to maturity
        
        Returns:
        - Implied volatility
        """
        def objective(vol):
            d1 = (np.log(self.S0/K) + (self.r + 0.5 * vol**2) * T) / (vol * np.sqrt(T))
            d2 = d1 - vol * np.sqrt(T)
            bs_price = self.S0 * norm.cdf(d1) - K * np.exp(-self.r * T) * norm.cdf(d2)
            return (bs_price - price)**2
        
        result = minimize_scalar(objective, bounds=(0.001, 1.0), method='bounded')
        
        return result.x

# Example usage of the model
def main():
    # Parameters
    S0 = 100.0  # Initial asset price
    r = 0.05    # Risk-free rate
    
    # Define regimes
    regime_means = [0.15, 0.3]  # Mean volatility in each regime
    regime_std_devs = [0.1, 0.2]  # Standard deviation of volatility in each regime
    
    # Deterministic switching times
    switching_times = [0.5]
    
    # Initialize model
    model = RandomizedAssetModel(
        S0=S0, 
        r=r,
        regime_means=regime_means,
        regime_std_devs=regime_std_devs,
        switching_times=switching_times,
        stochastic_switching=False
    )
    
    # Test 1: Simulate and visualize paths
    print("Simulating asset paths...")
    T = 1.0  # Time horizon
    dt = 0.01  # Time step
    num_paths = 5  # Number of paths to simulate
    
    sim_results = model.simulate_randomized_paths(T, dt, num_paths)
    
    # Plot simulated paths
    plt.figure(figsize=(12, 6))
    times = sim_results['times']
    
    plt.subplot(1, 2, 1)
    for i in range(num_paths):
        plt.plot(times, sim_results['randomized_paths'][i])
    plt.axvline(x=switching_times[0], color='r', linestyle='--', label='Regime Switch')
    plt.title('Randomized Model Paths')
    plt.xlabel('Time')
    plt.ylabel('Asset Price')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    for i in range(num_paths):
        plt.plot(times, sim_results['local_vol_paths'][i])
    plt.axvline(x=switching_times[0], color='r', linestyle='--', label='Regime Switch')
    plt.title('Local Volatility Model Paths')
    plt.xlabel('Time')
    plt.ylabel('Asset Price')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('deterministic_switching_paths.png')
    
    # Test 2: Stochastic switching
    model_stochastic = RandomizedAssetModel(
        S0=S0, 
        r=r,
        regime_means=regime_means,
        regime_std_devs=regime_std_devs,
        stochastic_switching=True,
        switching_rates=[2.0]  # Mean of 0.5 for the exponential distribution
    )
    
    print("Simulating asset paths with stochastic switching...")
    stoch_results = model_stochastic.simulate_randomized_paths(T, dt, num_paths)
    
    # Plot paths with stochastic switching
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    for i in range(num_paths):
        plt.plot(times, stoch_results['randomized_paths'][i])
    plt.title('Randomized Model Paths (Stochastic Switching)')
    plt.xlabel('Time')
    plt.ylabel('Asset Price')
    
    plt.subplot(1, 2, 2)
    for i in range(num_paths):
        plt.plot(times, stoch_results['local_vol_paths'][i])
    plt.title('Local Volatility Model Paths (Stochastic Switching)')
    plt.xlabel('Time')
    plt.ylabel('Asset Price')
    
    plt.tight_layout()
    plt.savefig('stochastic_switching_paths.png')
    
    # Test 3: Compare option prices and implied volatility smiles
    print("Computing option prices and implied volatilities...")
    
    # Define models for comparison
    model_no_switch = RandomizedAssetModel(
        S0=S0, 
        r=r,
        regime_means=[regime_means[1]],  # Only use the "excited" regime
        regime_std_devs=[regime_std_devs[1]],
        switching_times=[],
        stochastic_switching=False
    )
    
    # Define strikes and maturity
    strikes = np.linspace(80, 120, 9)
    T = 1.0
    
    # Compute prices and IVs
    prices_det = [model.price_european_call(K, T) for K in strikes]
    prices_stoch = [model_stochastic.price_european_call(K, T) for K in strikes]
    prices_no_switch = [model_no_switch.price_european_call(K, T) for K in strikes]
    
    ivs_det = [model._implied_volatility(p, K, T) for p, K in zip(prices_det, strikes)]
    ivs_stoch = [model_stochastic._implied_volatility(p, K, T) for p, K in zip(prices_stoch, strikes)]
    ivs_no_switch = [model_no_switch._implied_volatility(p, K, T) for p, K in zip(prices_no_switch, strikes)]
    
    # Plot implied volatility smile
    plt.figure(figsize=(10, 6))
    plt.plot(strikes, ivs_det, 'o-', label='Deterministic Switching')
    plt.plot(strikes, ivs_stoch, 's-', label='Stochastic Switching')
    plt.plot(strikes, ivs_no_switch, '^-', label='No Switching (Excited Regime)')
    plt.axvline(x=S0, color='gray', linestyle='--', label='ATM')
    plt.title('Implied Volatility Smile')
    plt.xlabel('Strike')
    plt.ylabel('Implied Volatility')
    plt.legend()
    plt.grid(True)
    plt.savefig('implied_volatility_smile.png')
    
    # Test 4: Compute and visualize IV surface for deterministic switching model
    print("Computing implied volatility surface...")
    maturities = np.array([0.5, 0.7, 1.0])
    iv_surface = model.compute_implied_volatility_surface(strikes, maturities)
    
    # Create pivot table for surface plot
    iv_pivot = iv_surface.pivot(index='Strike', columns='Maturity', values='IV')
    
    # Plot IV surface
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    X, Y = np.meshgrid(iv_pivot.columns, iv_pivot.index)
    Z = iv_pivot.values
    
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
    
    ax.set_xlabel('Maturity')
    ax.set_ylabel('Strike')
    ax.set_zlabel('Implied Volatility')
    ax.set_title('Implied Volatility Surface - Deterministic Switching')
    
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    plt.savefig('implied_volatility_surface.png')
    
    print("All tests completed. Check the output figures.")

if __name__ == "__main__":
    main()