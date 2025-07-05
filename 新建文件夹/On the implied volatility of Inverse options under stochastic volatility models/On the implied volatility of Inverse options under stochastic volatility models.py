import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import brentq
from scipy.special import erfc
import pandas as pd
from tqdm import tqdm

# Set random seed for reproducibility
np.random.seed(42)

class InverseOptionsModel:
    """Base class for models to price inverse options"""
    
    def __init__(self, S0=100, T=0.001, n_steps=50, n_paths=100000):
        """
        Initialize the model parameters.
        
        Parameters:
        -----------
        S0 : float
            Initial asset price
        T : float
            Time to maturity in years
        n_steps : int
            Number of time steps for simulation
        n_paths : int
            Number of simulation paths
        """
        self.S0 = S0
        self.T = T
        self.n_steps = n_steps
        self.n_paths = n_paths
        self.dt = T / n_steps
        
    def simulate_paths(self):
        """Abstract method to be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement this method")
    
    def price_inverse_call(self, K, paths=None):
        """
        Price an inverse European call option.
        
        Parameters:
        -----------
        K : float
            Strike price
        paths : ndarray, optional
            Pre-simulated price paths
            
        Returns:
        --------
        float : Option price
        """
        if paths is None:
            paths = self.simulate_paths()
        
        # Calculate payoff for inverse call: [(S_T - K)/S_T]^+
        payoff = np.maximum(0, (paths[:, -1] - K) / paths[:, -1])
        
        # Return the expected value under risk-neutral measure
        return np.mean(payoff)
    
    def BS_inverse_call(self, t, x, k, sigma):
        """
        Black-Scholes price of an inverse European call option.
        
        Parameters:
        -----------
        t : float
            Current time
        x : float
            Log-underlying price
        k : float
            Log-strike price
        sigma : float
            Volatility
            
        Returns:
        --------
        float : Option price
        """
        T_t = self.T - t
        if T_t <= 0:
            # At maturity, the price is the payoff
            return np.maximum(0, (np.exp(x) - np.exp(k)) / np.exp(x))
        
        d2 = (x - k) / (sigma * np.sqrt(T_t)) - sigma * np.sqrt(T_t) / 2
        d1 = d2 - sigma * np.sqrt(T_t)
        
        return norm.cdf(d2) - np.exp(sigma**2 * T_t / 2 + k - x) * norm.cdf(d1)
    
    def calculate_implied_volatility(self, price, k, min_vol=1e-4, max_vol=2.0):
        """
        Calculate implied volatility of an inverse European call option.
        
        Parameters:
        -----------
        price : float
            Option price
        k : float
            Log-strike price
            
        Returns:
        --------
        float : Implied volatility
        """
        x = np.log(self.S0)
        
        # Define the objective function (difference between market and model price)
        def objective(sigma):
            return self.BS_inverse_call(0, x, k, sigma) - price
        
        # Handle extreme cases
        if objective(min_vol) * objective(max_vol) > 0:
            if abs(objective(min_vol)) < abs(objective(max_vol)):
                return min_vol
            else:
                return max_vol
        
        # Find the root using Brent's method
        try:
            implied_vol = brentq(objective, min_vol, max_vol, xtol=1e-8)
            return implied_vol
        except ValueError:
            # If brentq fails, return NaN
            return np.nan
    
    def calculate_ATM_implied_volatility_skew(self, k, implied_vol, epsilon=1e-5):
        """
        Calculate the at-the-money implied volatility skew.
        
        Parameters:
        -----------
        k : float
            Log-strike price
        implied_vol : float
            Implied volatility at strike k
            
        Returns:
        --------
        float : Implied volatility skew
        """
        x = np.log(self.S0)
        
        # Formula from equation (18) in the paper
        numerator = -self.partial_k_BS(0, x, k, implied_vol) - self.expected_indicator_term(k)
        denominator = self.partial_sigma_BS(0, x, k, implied_vol)
        
        return numerator / denominator
    
    def partial_k_BS(self, t, x, k, sigma):
        """
        Partial derivative of BS price w.r.t. log-strike k.
        
        Parameters:
        -----------
        t : float
            Current time
        x : float
            Log-underlying price
        k : float
            Log-strike price
        sigma : float
            Volatility
            
        Returns:
        --------
        float : Partial derivative
        """
        T_t = self.T - t
        if T_t <= 0:
            return 0
        
        return self.BS_inverse_call(t, x, k, sigma) - 0.5 * erfc(sigma * np.sqrt(T_t) / (2 * np.sqrt(2)))
    
    def partial_sigma_BS(self, t, x, k, sigma):
        """
        Partial derivative of BS price w.r.t. volatility sigma.
        
        Parameters:
        -----------
        t : float
            Current time
        x : float
            Log-underlying price
        k : float
            Log-strike price
        sigma : float
            Volatility
            
        Returns:
        --------
        float : Partial derivative
        """
        T_t = self.T - t
        if T_t <= 0:
            return 0
        
        term1 = -sigma * T_t * np.exp(sigma**2 * T_t) * erfc(3 * sigma * np.sqrt(T_t) / (2 * np.sqrt(2)))
        term2 = np.exp(-sigma**2 * T_t / 8) * np.sqrt(T_t) / (np.sqrt(2 * np.pi))
        
        return term1 + term2
    
    def expected_indicator_term(self, k):
        """
        Expected value of e^(k-X_T) * 1_{X_T >= k}.
        
        Parameters:
        -----------
        k : float
            Log-strike price
            
        Returns:
        --------
        float : Expected value
        """
        paths = self.simulate_paths()
        X_T = np.log(paths[:, -1])
        indicator = (X_T >= k)
        
        return np.mean(np.exp(k - X_T) * indicator)


class SABRModel(InverseOptionsModel):
    """SABR stochastic volatility model for inverse options"""
    
    def __init__(self, S0=100, T=0.001, n_steps=50, n_paths=100000, alpha=0.3, sigma0=0.2, rho=-0.3):
        """
        Initialize the SABR model parameters.
        
        Parameters:
        -----------
        S0 : float
            Initial asset price
        T : float
            Time to maturity in years
        n_steps : int
            Number of time steps for simulation
        n_paths : int
            Number of simulation paths
        alpha : float
            Volatility of volatility
        sigma0 : float
            Initial volatility
        rho : float
            Correlation between asset and volatility
        """
        super().__init__(S0, T, n_steps, n_paths)
        self.alpha = alpha
        self.sigma0 = sigma0
        self.rho = rho
    
    def simulate_paths(self):
        """
        Simulate asset price paths under the SABR model.
        
        Returns:
        --------
        ndarray : Simulated price paths (n_paths x (n_steps+1))
        """
        # Initialize arrays
        S = np.zeros((self.n_paths, self.n_steps + 1))
        sigma = np.zeros((self.n_paths, self.n_steps + 1))
        
        # Set initial values
        S[:, 0] = self.S0
        sigma[:, 0] = self.sigma0
        
        # Generate random increments for correlated Brownian motions
        dt_sqrt = np.sqrt(self.dt)
        dW1 = np.random.normal(0, dt_sqrt, (self.n_paths, self.n_steps))
        dW2 = self.rho * dW1 + np.sqrt(1 - self.rho**2) * np.random.normal(0, dt_sqrt, (self.n_paths, self.n_steps))
        
        # Simulate paths
        for i in range(self.n_steps):
            # Update volatility
            sigma[:, i+1] = sigma[:, i] * np.exp(self.alpha * dW1[:, i] - 0.5 * self.alpha**2 * self.dt)
            
            # Update asset price (log-normal dynamics)
            S[:, i+1] = S[:, i] * np.exp(sigma[:, i] * dW2[:, i] - 0.5 * sigma[:, i]**2 * self.dt)
        
        return S
    
    def theoretical_ATM_IV_level(self):
        """
        Theoretical at-the-money implied volatility level from Theorem 1.
        
        Returns:
        --------
        float : Theoretical IV level
        """
        return self.sigma0
    
    def theoretical_ATM_IV_skew(self):
        """
        Theoretical at-the-money implied volatility skew from equation (16).
        
        Returns:
        --------
        float : Theoretical IV skew
        """
        return 0.5 * self.rho * self.alpha


class FractionalBergomiModel(InverseOptionsModel):
    """Fractional Bergomi stochastic volatility model for inverse options"""
    
    def __init__(self, S0=100, T=0.001, n_steps=50, n_paths=100000, H=0.5, v=0.5, sigma0=0.2, rho=-0.3):
        """
        Initialize the fractional Bergomi model parameters.
        
        Parameters:
        -----------
        S0 : float
            Initial asset price
        T : float
            Time to maturity in years
        n_steps : int
            Number of time steps for simulation
        n_paths : int
            Number of simulation paths
        H : float
            Hurst parameter
        v : float
            Volatility of volatility
        sigma0 : float
            Initial volatility
        rho : float
            Correlation between asset and volatility
        """
        super().__init__(S0, T, n_steps, n_paths)
        self.H = H
        self.v = v
        self.sigma0 = sigma0
        self.rho = rho
    
    def simulate_paths(self):
        """
        Simulate asset price paths under the fractional Bergomi model.
        
        Returns:
        --------
        ndarray : Simulated price paths (n_paths x (n_steps+1))
        """
        # Initialize arrays
        S = np.zeros((self.n_paths, self.n_steps + 1))
        sigma = np.zeros((self.n_paths, self.n_steps + 1))
        
        # Set initial values
        S[:, 0] = self.S0
        sigma[:, 0] = self.sigma0
        
        # Time grid
        t = np.linspace(0, self.T, self.n_steps + 1)
        
        # Generate correlated fractional Brownian motion
        # We use the Hosking method for simplicity in this demonstration
        # For production code, more efficient methods should be used
        Z = self.simulate_fBM(self.H, self.n_paths, self.n_steps + 1)
        
        # Generate random increments for correlated Brownian motions
        dt_sqrt = np.sqrt(self.dt)
        dW = np.random.normal(0, dt_sqrt, (self.n_paths, self.n_steps))
        
        # Simulate paths
        for i in range(self.n_steps):
            # Update volatility
            sigma[:, i+1] = self.sigma0 * np.exp(self.v * np.sqrt(2 * self.H) * Z[:, i+1] - 0.5 * self.v**2 * t[i+1]**(2*self.H))
            
            # Compute log-returns with correlation
            dX = -0.5 * sigma[:, i]**2 * self.dt + sigma[:, i] * (
                self.rho * (Z[:, i+1] - Z[:, i]) + 
                np.sqrt(1 - self.rho**2) * dW[:, i]
            )
            
            # Update asset price
            S[:, i+1] = S[:, i] * np.exp(dX)
        
        return S
    
    def simulate_fBM(self, H, n_paths, n_steps):
        """
        Simulate fractional Brownian motion using the Hosking method.
        
        Parameters:
        -----------
        H : float
            Hurst parameter
        n_paths : int
            Number of paths
        n_steps : int
            Number of time steps
            
        Returns:
        --------
        ndarray : Simulated fBM paths (n_paths x n_steps)
        """
        # Time grid
        t = np.linspace(0, self.T, n_steps)
        
        # Initialize fBM array
        fBM = np.zeros((n_paths, n_steps))
        
        # Generate standard Brownian motion
        dW = np.random.normal(0, np.sqrt(self.dt), (n_paths, n_steps-1))
        
        # Convert to fBM using the Mandelbrot-Van Ness representation (simplified)
        for i in range(1, n_steps):
            # Use the increments of fBM approximation
            kernel = (t[i] - t[:i])**(H-0.5) - (t[i-1] - t[:i])**(H-0.5)
            kernel[np.isnan(kernel)] = 0  # Handle t_i = t_j case
            
            if i > 1:
                fBM[:, i] = fBM[:, i-1] + np.sqrt(2*H) * np.sum(kernel[:-1][:, np.newaxis] * dW[:, :i-1].T, axis=0)
            else:
                fBM[:, i] = fBM[:, i-1] + np.sqrt(self.dt) * dW[:, 0]
        
        return fBM
    
    def theoretical_ATM_IV_level(self):
        """
        Theoretical at-the-money implied volatility level from Theorem 1.
        
        Returns:
        --------
        float : Theoretical IV level
        """
        return self.sigma0
    
    def theoretical_ATM_IV_skew(self):
        """
        Theoretical at-the-money implied volatility skew from equations (19) and (20).
        
        Returns:
        --------
        float : Theoretical IV skew
        """
        if self.H > 0.5:
            return 0.0
        elif self.H == 0.5:
            return self.rho * self.v / 4
        else:
            # For small T, the skew blows up like T^(H-1/2)
            # Since we're looking at the scaled version T^(1/2-H) * skew,
            # we return the limit in equation (20)
            return 2 * self.rho * self.v / np.sqrt(2 * self.H) * (3 + 4 * self.H * (2 + self.H))


def simulate_and_analyze(model_class, model_params, sigma0_range, experiment_name):
    """
    Run simulations and analyze the results for a given model.
    
    Parameters:
    -----------
    model_class : class
        Model class to instantiate
    model_params : dict
        Parameters for the model
    sigma0_range : list
        Range of initial volatilities to test
    experiment_name : str
        Name of the experiment for plot titles
    """
    # Initialize arrays to store results
    atmiv_levels = []
    atmiv_skews = []
    theoretical_levels = []
    theoretical_skews = []
    
    # Loop over different initial volatilities
    for sigma0 in tqdm(sigma0_range, desc=f"Processing {experiment_name}"):
        # Update model parameters
        params = model_params.copy()
        params['sigma0'] = sigma0
        
        # Instantiate model
        model = model_class(**params)
        
        # Simulate paths
        paths = model.simulate_paths()
        
        # Price ATM inverse call option
        k_star = np.log(model.S0)  # log(S0)
        price = model.price_inverse_call(model.S0, paths)
        
        # Calculate implied volatility
        implied_vol = model.calculate_implied_volatility(price, k_star)
        
        # Calculate IV skew
        skew = model.calculate_ATM_implied_volatility_skew(k_star, implied_vol)
        
        # Store results
        atmiv_levels.append(implied_vol)
        atmiv_skews.append(skew)
        
        # Calculate theoretical values
        theoretical_levels.append(model.theoretical_ATM_IV_level())
        theoretical_skews.append(model.theoretical_ATM_IV_skew())
    
    # Create DataFrame for results
    results = pd.DataFrame({
        'sigma0': sigma0_range,
        'implied_vol': atmiv_levels,
        'iv_skew': atmiv_skews,
        'theoretical_level': theoretical_levels,
        'theoretical_skew': theoretical_skews
    })
    
    # Plot results
    plt.figure(figsize=(14, 6))
    
    # Plot ATMIV level
    plt.subplot(1, 2, 1)
    plt.plot(sigma0_range, atmiv_levels, 'bo-', label='Simulated')
    plt.plot(sigma0_range, theoretical_levels, 'r--', label='Theoretical')
    plt.xlabel('Initial Volatility (σ₀)')
    plt.ylabel('ATM Implied Volatility Level')
    plt.title(f'{experiment_name}: ATM IV Level')
    plt.grid(True)
    plt.legend()
    
    # Plot ATMIV skew
    plt.subplot(1, 2, 2)
    plt.plot(sigma0_range, atmiv_skews, 'bo-', label='Simulated')
    plt.plot(sigma0_range, theoretical_skews, 'r--', label='Theoretical')
    plt.xlabel('Initial Volatility (σ₀)')
    plt.ylabel('ATM Implied Volatility Skew')
    plt.title(f'{experiment_name}: ATM IV Skew')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f"{experiment_name.replace(' ', '_').lower()}_results.png")
    plt.show()
    
    return results


def study_maturity_effect(model_class, model_params, T_range, experiment_name, scale_factor=None):
    """
    Study the effect of maturity on the ATM IV skew.
    
    Parameters:
    -----------
    model_class : class
        Model class to instantiate
    model_params : dict
        Parameters for the model
    T_range : list
        Range of maturities to test
    experiment_name : str
        Name of the experiment for plot titles
    scale_factor : callable, optional
        Function to scale the skew based on maturity
    """
    # Initialize arrays to store results
    atmiv_levels = []
    atmiv_skews = []
    scaled_skews = []
    
    # Loop over different maturities
    for T in tqdm(T_range, desc=f"Processing {experiment_name} maturities"):
        # Update model parameters
        params = model_params.copy()
        params['T'] = T
        
        # Instantiate model
        model = model_class(**params)
        
        # Simulate paths
        paths = model.simulate_paths()
        
        # Price ATM inverse call option
        k_star = np.log(model.S0)  # log(S0)
        price = model.price_inverse_call(model.S0, paths)
        
        # Calculate implied volatility
        implied_vol = model.calculate_implied_volatility(price, k_star)
        
        # Calculate IV skew
        skew = model.calculate_ATM_implied_volatility_skew(k_star, implied_vol)
        
        # Store results
        atmiv_levels.append(implied_vol)
        atmiv_skews.append(skew)
        
        # Scale skew if requested
        if scale_factor:
            scaled_skews.append(scale_factor(T) * skew)
        else:
            scaled_skews.append(skew)
    
    # Create DataFrame for results
    results = pd.DataFrame({
        'T': T_range,
        'implied_vol': atmiv_levels,
        'iv_skew': atmiv_skews,
        'scaled_skew': scaled_skews
    })
    
    # Plot results
    plt.figure(figsize=(14, 10))
    
    # Plot ATMIV level
    plt.subplot(2, 1, 1)
    plt.plot(T_range, atmiv_levels, 'bo-')
    plt.xlabel('Maturity (T)')
    plt.ylabel('ATM Implied Volatility Level')
    plt.title(f'{experiment_name}: ATM IV Level vs Maturity')
    plt.grid(True)
    
    # Plot ATMIV skew
    plt.subplot(2, 2, 3)
    plt.plot(T_range, atmiv_skews, 'bo-')
    plt.xlabel('Maturity (T)')
    plt.ylabel('ATM Implied Volatility Skew')
    plt.title(f'{experiment_name}: ATM IV Skew vs Maturity')
    plt.grid(True)
    
    # Plot scaled ATMIV skew if applicable
    if scale_factor:
        plt.subplot(2, 2, 4)
        plt.plot(T_range, scaled_skews, 'go-')
        plt.xlabel('Maturity (T)')
        plt.ylabel('Scaled ATM IV Skew')
        plt.title(f'{experiment_name}: Scaled ATM IV Skew vs Maturity')
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f"{experiment_name.replace(' ', '_').lower()}_maturity_effect.png")
    plt.show()
    
    # Fit power law to skew vs maturity
    if len(T_range) > 5:  # Only fit if we have enough data points
        try:
            import statsmodels.api as sm
            
            # Transform data to log-log scale for power law fitting
            log_T = np.log(T_range)
            log_skew = np.log(np.abs(atmiv_skews))
            
            # Fit linear model on log-log scale
            X = sm.add_constant(log_T)
            model = sm.OLS(log_skew, X).fit()
            
            # Extract power law exponent
            exponent = model.params[1]
            print(f"\nPower law fit for {experiment_name} ATM IV skew:")
            print(f"Skew ~ T^{exponent:.4f}")
            print(f"R-squared: {model.rsquared:.4f}")
            
            # Plot fit
            plt.figure(figsize=(8, 6))
            plt.loglog(T_range, np.abs(atmiv_skews), 'bo', label='Data')
            plt.loglog(T_range, np.exp(model.predict(X)), 'r-', label=f'Fit: T^{exponent:.4f}')
            plt.xlabel('Maturity (T)')
            plt.ylabel('|ATM IV Skew|')
            plt.title(f'{experiment_name}: Power Law Fit for ATM IV Skew')
            plt.legend()
            plt.grid(True)
            plt.savefig(f"{experiment_name.replace(' ', '_').lower()}_power_law_fit.png")
            plt.show()
        
        except Exception as e:
            print(f"Could not fit power law: {e}")
    
    return results


def main():
    # Parameters for SABR model
    sabr_params = {
        'S0': 100,
        'T': 0.001,  # Half a day
        'n_steps': 50,
        'n_paths': 100000,
        'alpha': 0.3,
        'sigma0': 0.2,
        'rho': -0.3
    }
    
    # Parameters for fractional Bergomi model
    fbergomi_params_rough = {
        'S0': 100,
        'T': 0.001,  # Half a day
        'n_steps': 50,
        'n_paths': 100000,
        'H': 0.4,  # Rough volatility
        'v': 0.5,
        'sigma0': 0.2,
        'rho': -0.3
    }
    
    fbergomi_params_smooth = {
        'S0': 100,
        'T': 0.001,  # Half a day
        'n_steps': 50,
        'n_paths': 100000,
        'H': 0.7,  # Smooth volatility
        'v': 0.5,
        'sigma0': 0.2,
        'rho': -0.3
    }
    
    # Range of initial volatilities to test
    sigma0_range = np.arange(0.1, 1.5, 0.1)
    
    # Run simulations for SABR model
    print("\nRunning SABR model simulations...")
    sabr_results = simulate_and_analyze(
        SABRModel,
        sabr_params,
        sigma0_range,
        "SABR Model (ρ=-0.3)"
    )
    
    # Run simulations for fractional Bergomi model with rough volatility
    print("\nRunning fractional Bergomi model simulations (rough volatility)...")
    fbergomi_rough_results = simulate_and_analyze(
        FractionalBergomiModel,
        fbergomi_params_rough,
        sigma0_range,
        "Fractional Bergomi Model (H=0.4)"
    )
    
    # Run simulations for fractional Bergomi model with smooth volatility
    print("\nRunning fractional Bergomi model simulations (smooth volatility)...")
    fbergomi_smooth_results = simulate_and_analyze(
        FractionalBergomiModel,
        fbergomi_params_smooth,
        sigma0_range,
        "Fractional Bergomi Model (H=0.7)"
    )
    
    # Study the effect of maturity on the ATM IV skew
    # Range of maturities to test
    T_range = np.logspace(-3, -1, 10)  # From 0.001 to 0.1
    
    # For rough volatility, scale by T^(1/2-H)
    scale_factor_rough = lambda T: T**(0.5 - fbergomi_params_rough['H'])
    
    print("\nStudying maturity effect for fractional Bergomi model (rough volatility)...")
    maturity_results_rough = study_maturity_effect(
        FractionalBergomiModel,
        fbergomi_params_rough,
        T_range,
        "Fractional Bergomi Model (H=0.4) Maturity Effect",
        scale_factor_rough
    )
    
    print("\nStudying maturity effect for fractional Bergomi model (smooth volatility)...")
    maturity_results_smooth = study_maturity_effect(
        FractionalBergomiModel,
        fbergomi_params_smooth,
        T_range,
        "Fractional Bergomi Model (H=0.7) Maturity Effect"
    )


if __name__ == "__main__":
    main()