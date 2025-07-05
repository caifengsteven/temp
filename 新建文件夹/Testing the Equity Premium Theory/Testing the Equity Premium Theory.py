import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import seaborn as sns
from typing import Tuple, Dict, List

# Set random seed for reproducibility
np.random.seed(42)

class EquityPremiumModel:
    """
    A class to simulate and test the equity premium model described in the paper.
    """
    
    def __init__(self, 
                 mean_consumption_growth: float, 
                 std_consumption_growth: float,
                 serial_correlation: float,
                 risk_free_rate: float,
                 stationary_prob: List[float] = [0.5, 0.5]):
        """
        Initialize the equity premium model.
        
        Parameters:
        -----------
        mean_consumption_growth : float
            Mean real consumption growth rate
        std_consumption_growth : float
            Standard deviation of real consumption growth rate
        serial_correlation : float
            Serial correlation of real consumption growth rate
        risk_free_rate : float
            Real return on riskless bonds
        stationary_prob : list, optional
            Stationary probability distribution of the Markov chain, default [0.5, 0.5]
        """
        self.mean_consumption_growth = mean_consumption_growth
        self.std_consumption_growth = std_consumption_growth
        self.serial_correlation = serial_correlation
        self.risk_free_rate = risk_free_rate
        self.stationary_prob = stationary_prob
        
        # Calculate time discount factor for bonds (from Theorem 1)
        self.beta = 1 / (1 + risk_free_rate)
        
        # Calculate consumption growth rates for each state
        self.lambda_values = self._calculate_lambda_values()
        
        # Calculate Markov transition matrix
        self.phi = self._calculate_markov_matrix()
    
    def _calculate_lambda_values(self) -> np.ndarray:
        """
        Calculate consumption growth rates for each state.
        
        Returns:
        --------
        lambda_values : ndarray
            Array of consumption growth rates for each state
        """
        # Using the mean and standard deviation of consumption growth,
        # calculate lambda_1 (growth state) and lambda_2 (contraction state)
        pi_1, pi_2 = self.stationary_prob
        
        # Solve for lambda_1 and lambda_2 using the mean and std
        # Mean = pi_1 * lambda_1 + pi_2 * lambda_2
        # Var = pi_1 * (lambda_1 - mean)^2 + pi_2 * (lambda_2 - mean)^2
        
        # From the mean equation:
        # lambda_2 = (mean - pi_1 * lambda_1) / pi_2
        
        # Substituting into the variance equation and solving for lambda_1
        var = self.std_consumption_growth ** 2
        a = pi_1 * pi_2
        b = -2 * pi_1 * pi_2 * self.mean_consumption_growth
        c = pi_1 * pi_2 * self.mean_consumption_growth ** 2 - var * pi_2
        
        # Quadratic formula
        discriminant = b ** 2 - 4 * a * c
        lambda_1 = (-b + np.sqrt(discriminant)) / (2 * a)
        
        # Calculate lambda_2 using the mean equation
        lambda_2 = (self.mean_consumption_growth - pi_1 * lambda_1) / pi_2
        
        return np.array([lambda_1, lambda_2])
    
    def _calculate_markov_matrix(self) -> np.ndarray:
        """
        Calculate the Markov transition matrix to match the serial correlation.
        
        Returns:
        --------
        phi : ndarray
            2x2 Markov transition matrix
        """
        pi_1, pi_2 = self.stationary_prob
        lambda_1, lambda_2 = self.lambda_values
        
        # The stationary distribution satisfies:
        # pi_1 = pi_1 * phi_11 + pi_2 * phi_21
        # pi_2 = pi_1 * phi_12 + pi_2 * phi_22
        
        # And the serial correlation is:
        # sigma = (pi_1 * phi_11 * (lambda_1 - mean) * (lambda_1 - mean) + 
        #          pi_1 * phi_12 * (lambda_1 - mean) * (lambda_2 - mean) +
        #          pi_2 * phi_21 * (lambda_2 - mean) * (lambda_1 - mean) +
        #          pi_2 * phi_22 * (lambda_2 - mean) * (lambda_2 - mean)) / var
        
        # Use the stationary distribution to derive one row in terms of the other
        # phi_21 = (pi_1 - pi_1 * phi_11) / pi_2
        # phi_22 = (pi_2 - pi_1 * phi_12) / pi_2
        
        # For a two-state Markov chain, each row sums to 1
        # phi_11 + phi_12 = 1
        # phi_21 + phi_22 = 1
        
        # Solve for phi_11 using the serial correlation
        mean = self.mean_consumption_growth
        var = self.std_consumption_growth ** 2
        
        dev_1 = lambda_1 - mean
        dev_2 = lambda_2 - mean
        
        # Set up the equation for serial correlation
        def correlation_equation(phi_11):
            phi_12 = 1 - phi_11
            phi_21 = (pi_1 - pi_1 * phi_11) / pi_2
            phi_22 = 1 - phi_21
            
            correlation = (pi_1 * phi_11 * dev_1 * dev_1 + 
                          pi_1 * phi_12 * dev_1 * dev_2 +
                          pi_2 * phi_21 * dev_2 * dev_1 + 
                          pi_2 * phi_22 * dev_2 * dev_2) / var
            
            return (correlation - self.serial_correlation) ** 2
        
        # Minimize to find phi_11 that matches the serial correlation
        result = minimize(correlation_equation, 0.5, bounds=[(0, 1)])
        phi_11 = result.x[0]
        
        # Calculate the rest of the transition matrix
        phi_12 = 1 - phi_11
        phi_21 = (pi_1 - pi_1 * phi_11) / pi_2
        phi_22 = 1 - phi_21
        
        return np.array([[phi_11, phi_12], [phi_21, phi_22]])
    
    def _calculate_equity_prices(self, alpha_e: float) -> np.ndarray:
        """
        Calculate equity prices for each state given a risk aversion coefficient.
        
        Parameters:
        -----------
        alpha_e : float
            Coefficient of Constant Relative Risk Aversion for equities
            
        Returns:
        --------
        w : ndarray
            Equity prices for each state
        """
        # Solve the system of equations for equity prices
        # w_i = beta * sum_j(phi_ij * lambda_j^(1-alpha_e) * (w_j + 1))
        
        # For a two-state model, we have:
        # w_1 = beta * (phi_11 * lambda_1^(1-alpha_e) * (w_1 + 1) + phi_12 * lambda_2^(1-alpha_e) * (w_2 + 1))
        # w_2 = beta * (phi_21 * lambda_1^(1-alpha_e) * (w_1 + 1) + phi_22 * lambda_2^(1-alpha_e) * (w_2 + 1))
        
        # Rewrite as a matrix equation:
        # [1 - beta * phi_11 * lambda_1^(1-alpha_e), -beta * phi_12 * lambda_2^(1-alpha_e)] [w_1]   [beta * phi_11 * lambda_1^(1-alpha_e) + beta * phi_12 * lambda_2^(1-alpha_e)]
        # [-beta * phi_21 * lambda_1^(1-alpha_e), 1 - beta * phi_22 * lambda_2^(1-alpha_e)] [w_2] = [beta * phi_21 * lambda_1^(1-alpha_e) + beta * phi_22 * lambda_2^(1-alpha_e)]
        
        lambda_1, lambda_2 = self.lambda_values
        phi_11, phi_12 = self.phi[0]
        phi_21, phi_22 = self.phi[1]
        
        # Calculate powers of lambda for efficiency
        lambda_1_power = lambda_1 ** (1 - alpha_e)
        lambda_2_power = lambda_2 ** (1 - alpha_e)
        
        # Coefficient matrix
        A = np.array([
            [1 - self.beta * phi_11 * lambda_1_power, -self.beta * phi_12 * lambda_2_power],
            [-self.beta * phi_21 * lambda_1_power, 1 - self.beta * phi_22 * lambda_2_power]
        ])
        
        # Right-hand side
        b = np.array([
            self.beta * phi_11 * lambda_1_power + self.beta * phi_12 * lambda_2_power,
            self.beta * phi_21 * lambda_1_power + self.beta * phi_22 * lambda_2_power
        ])
        
        # Solve for equity prices
        w = np.linalg.solve(A, b)
        
        return w
    
    def calculate_equity_return(self, alpha_e: float) -> Tuple[float, float]:
        """
        Calculate the expected return and standard deviation of equities.
        
        Parameters:
        -----------
        alpha_e : float
            Coefficient of Constant Relative Risk Aversion for equities
            
        Returns:
        --------
        return_e : float
            Expected return on equities
        std_e : float
            Standard deviation of return on equities
        """
        # Calculate equity prices for each state
        w = self._calculate_equity_prices(alpha_e)
        
        # Calculate equity returns for each state transition
        lambda_1, lambda_2 = self.lambda_values
        phi_11, phi_12 = self.phi[0]
        phi_21, phi_22 = self.phi[1]
        pi_1, pi_2 = self.stationary_prob
        
        # Calculate returns for each state transition
        r_11 = lambda_1 * (w[0] + 1) / w[0] - 1
        r_12 = lambda_2 * (w[1] + 1) / w[0] - 1
        r_21 = lambda_1 * (w[0] + 1) / w[1] - 1
        r_22 = lambda_2 * (w[1] + 1) / w[1] - 1
        
        # Calculate expected returns in each state
        R_1 = phi_11 * r_11 + phi_12 * r_12
        R_2 = phi_21 * r_21 + phi_22 * r_22
        
        # Calculate overall expected return
        return_e = pi_1 * R_1 + pi_2 * R_2
        
        # Calculate standard deviation of returns
        all_returns = np.array([r_11, r_12, r_21, r_22])
        probabilities = np.array([pi_1 * phi_11, pi_1 * phi_12, pi_2 * phi_21, pi_2 * phi_22])
        
        std_e = np.sqrt(np.sum(probabilities * (all_returns - return_e) ** 2))
        
        return return_e, std_e
    
    def find_optimal_alpha(self, alpha_range=np.linspace(0, 20, 201)) -> Dict:
        """
        Find the optimal alpha_e that maximizes return on volatility.
        
        Parameters:
        -----------
        alpha_range : ndarray, optional
            Range of alpha values to test
            
        Returns:
        --------
        results : dict
            Dictionary of results
        """
        results = {
            'alpha_e': [],
            'return_e': [],
            'std_e': [],
            'return_on_volatility': []
        }
        
        for alpha_e in alpha_range:
            return_e, std_e = self.calculate_equity_return(alpha_e)
            
            # Calculate return on volatility (Sharpe ratio with risk-free rate)
            if std_e > 0:
                return_on_volatility = (return_e - self.risk_free_rate) / std_e
            else:
                return_on_volatility = 0
            
            results['alpha_e'].append(alpha_e)
            results['return_e'].append(return_e)
            results['std_e'].append(std_e)
            results['return_on_volatility'].append(return_on_volatility)
        
        # Find the optimal alpha_e
        optimal_idx = np.argmax(results['return_on_volatility'])
        optimal_alpha_e = results['alpha_e'][optimal_idx]
        optimal_return_e = results['return_e'][optimal_idx]
        optimal_std_e = results['std_e'][optimal_idx]
        optimal_rov = results['return_on_volatility'][optimal_idx]
        
        return {
            'results': results,
            'optimal_alpha_e': optimal_alpha_e,
            'optimal_return_e': optimal_return_e,
            'optimal_std_e': optimal_std_e,
            'optimal_rov': optimal_rov
        }
    
    def plot_return_on_volatility(self, results, actual_equity_return=None):
        """
        Plot the return on volatility curve similar to the paper's figures.
        
        Parameters:
        -----------
        results : dict
            Results from find_optimal_alpha
        actual_equity_return : float, optional
            Actual historical equity return to compare with model
        """
        # Create a new figure
        plt.figure(figsize=(10, 6))
        
        # Plot equity return vs standard deviation
        plt.plot(results['results']['std_e'], results['results']['return_e'], 'b-', linewidth=2.5)
        
        # Mark the risk-free rate
        plt.scatter([0], [self.risk_free_rate], color='black', s=100)
        plt.annotate(f'Risk-free rate = {self.risk_free_rate:.4f}', 
                     xy=(0, self.risk_free_rate), 
                     xytext=(0.01, self.risk_free_rate - 0.01),
                     arrowprops=dict(arrowstyle="->"))
        
        # Mark the optimal point
        optimal_std = results['optimal_std_e']
        optimal_return = results['optimal_return_e']
        
        plt.scatter([optimal_std], [optimal_return], color='red', s=100)
        plt.annotate(f'Optimal: α_e = {results["optimal_alpha_e"]:.2f}, Return = {optimal_return:.4f}', 
                     xy=(optimal_std, optimal_return), 
                     xytext=(optimal_std + 0.01, optimal_return - 0.02),
                     arrowprops=dict(arrowstyle="->"))
        
        # Draw the tangent line
        x_vals = np.linspace(0, optimal_std * 1.5, 100)
        slope = (optimal_return - self.risk_free_rate) / optimal_std
        y_vals = self.risk_free_rate + slope * x_vals
        
        plt.plot(x_vals, y_vals, 'r--', linewidth=1.5)
        
        # If actual equity return is provided, mark it on the curve
        if actual_equity_return is not None:
            # Find the point on the curve with return closest to actual
            idx = np.argmin(np.abs(np.array(results['results']['return_e']) - actual_equity_return))
            actual_std = results['results']['std_e'][idx]
            actual_alpha = results['results']['alpha_e'][idx]
            
            plt.scatter([actual_std], [actual_equity_return], color='green', s=100)
            plt.annotate(f'Actual: α_e = {actual_alpha:.2f}, Return = {actual_equity_return:.4f}', 
                         xy=(actual_std, actual_equity_return), 
                         xytext=(actual_std + 0.01, actual_equity_return + 0.02),
                         arrowprops=dict(arrowstyle="->"))
        
        # Add labels and title
        plt.xlabel('Standard Deviation of Return')
        plt.ylabel('Expected Return')
        plt.title('Return on Volatility')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        return plt

def run_historical_periods():
    """
    Run the model on the two historical periods from the paper.
    """
    # Period 1: 1889-1979
    print("Period 1: 1889-1979")
    period1 = EquityPremiumModel(
        mean_consumption_growth=0.0183,
        std_consumption_growth=0.0357,
        serial_correlation=-0.14,
        risk_free_rate=0.008
    )
    
    results1 = period1.find_optimal_alpha()
    
    print(f"Optimal α_e: {results1['optimal_alpha_e']:.2f}")
    print(f"Optimal return on equities: {results1['optimal_return_e']:.4f}")
    print(f"Optimal standard deviation: {results1['optimal_std_e']:.4f}")
    print(f"Optimal return on volatility: {results1['optimal_rov']:.4f}")
    
    # Plot results for period 1
    plt1 = period1.plot_return_on_volatility(results1, actual_equity_return=0.0698)
    plt1.savefig('period1_1889_1979.png')
    
    # Period 2: 1960-2022
    print("\nPeriod 2: 1960-2022")
    period2 = EquityPremiumModel(
        mean_consumption_growth=0.0194,
        std_consumption_growth=0.0158,
        serial_correlation=0.15,
        risk_free_rate=0.0097
    )
    
    results2 = period2.find_optimal_alpha()
    
    print(f"Optimal α_e: {results2['optimal_alpha_e']:.2f}")
    print(f"Optimal return on equities: {results2['optimal_return_e']:.4f}")
    print(f"Optimal standard deviation: {results2['optimal_std_e']:.4f}")
    print(f"Optimal return on volatility: {results2['optimal_rov']:.4f}")
    
    # Plot results for period 2
    plt2 = period2.plot_return_on_volatility(results2, actual_equity_return=0.0733)
    plt2.savefig('period2_1960_2022.png')

def simulate_monte_carlo():
    """
    Run Monte Carlo simulations to validate the model's robustness.
    """
    # Base parameters from period 2: 1960-2022
    base_params = {
        'mean_consumption_growth': 0.0194,
        'std_consumption_growth': 0.0158,
        'serial_correlation': 0.15,
        'risk_free_rate': 0.0097
    }
    
    # Number of simulations
    n_sims = 100
    
    # Variables to store results
    optimal_alphas = []
    optimal_returns = []
    
    # Run simulations with randomly perturbed parameters
    for i in range(n_sims):
        # Perturb parameters slightly
        params = {
            'mean_consumption_growth': base_params['mean_consumption_growth'] * (1 + np.random.normal(0, 0.1)),
            'std_consumption_growth': base_params['std_consumption_growth'] * (1 + np.random.normal(0, 0.1)),
            'serial_correlation': base_params['serial_correlation'] + np.random.normal(0, 0.05),
            'risk_free_rate': base_params['risk_free_rate'] * (1 + np.random.normal(0, 0.1))
        }
        
        # Ensure serial correlation is between -1 and 1
        params['serial_correlation'] = np.clip(params['serial_correlation'], -0.99, 0.99)
        
        # Ensure risk-free rate is positive
        params['risk_free_rate'] = max(0.001, params['risk_free_rate'])
        
        try:
            # Create model and find optimal alpha
            model = EquityPremiumModel(**params)
            results = model.find_optimal_alpha()
            
            # Store results
            optimal_alphas.append(results['optimal_alpha_e'])
            optimal_returns.append(results['optimal_return_e'])
        except Exception as e:
            print(f"Simulation {i} failed: {e}")
    
    # Plot distribution of optimal alphas
    plt.figure(figsize=(10, 6))
    sns.histplot(optimal_alphas, kde=True, bins=20)
    plt.axvline(x=np.mean(optimal_alphas), color='r', linestyle='--', label=f'Mean: {np.mean(optimal_alphas):.2f}')
    plt.axvline(x=6.75, color='g', linestyle='--', label='Paper: 6.75')
    plt.xlabel('Optimal α_e')
    plt.ylabel('Frequency')
    plt.title('Distribution of Optimal α_e in Monte Carlo Simulations')
    plt.legend()
    plt.tight_layout()
    plt.savefig('monte_carlo_alpha_distribution.png')
    
    # Plot distribution of optimal returns
    plt.figure(figsize=(10, 6))
    sns.histplot(optimal_returns, kde=True, bins=20)
    plt.axvline(x=np.mean(optimal_returns), color='r', linestyle='--', label=f'Mean: {np.mean(optimal_returns):.4f}')
    plt.axvline(x=0.143, color='g', linestyle='--', label='Paper: 0.143')
    plt.xlabel('Optimal Return on Equities')
    plt.ylabel('Frequency')
    plt.title('Distribution of Optimal Returns in Monte Carlo Simulations')
    plt.legend()
    plt.tight_layout()
    plt.savefig('monte_carlo_return_distribution.png')
    
    # Return summary statistics
    return {
        'alpha_mean': np.mean(optimal_alphas),
        'alpha_std': np.std(optimal_alphas),
        'alpha_95ci': (np.percentile(optimal_alphas, 2.5), np.percentile(optimal_alphas, 97.5)),
        'return_mean': np.mean(optimal_returns),
        'return_std': np.std(optimal_returns),
        'return_95ci': (np.percentile(optimal_returns, 2.5), np.percentile(optimal_returns, 97.5))
    }

def simulate_investment_strategy():
    """
    Simulate an investment strategy based on the paper's theory.
    """
    # Set up simulation parameters
    n_years = 50
    n_periods_per_year = 252  # Trading days
    n_periods = n_years * n_periods_per_year
    
    # Model parameters for period 2: 1960-2022
    model_params = {
        'mean_consumption_growth': 0.0194,
        'std_consumption_growth': 0.0158,
        'serial_correlation': 0.15,
        'risk_free_rate': 0.0097
    }
    
    # Create model and find optimal alpha
    model = EquityPremiumModel(**model_params)
    results = model.find_optimal_alpha()
    
    # Extract parameters for simulation
    lambda_1, lambda_2 = model.lambda_values
    phi = model.phi
    
    # Initial state (randomly choose growth or contraction)
    current_state = np.random.choice([0, 1])
    
    # Initialize price paths
    risk_free_price = np.ones(n_periods)
    equity_price_optimal = np.ones(n_periods)
    equity_price_actual = np.ones(n_periods)
    
    # Calculate equity prices for optimal and actual alpha
    w_optimal = model._calculate_equity_prices(results['optimal_alpha_e'])
    
    # Find alpha that gives return closest to actual historical return (0.0733)
    alphas = np.linspace(0, 10, 101)
    returns = [model.calculate_equity_return(alpha)[0] for alpha in alphas]
    actual_alpha_idx = np.argmin(np.abs(np.array(returns) - 0.0733))
    actual_alpha = alphas[actual_alpha_idx]
    
    w_actual = model._calculate_equity_prices(actual_alpha)
    
    # Simulate price paths
    for t in range(1, n_periods):
        # Update risk-free price
        risk_free_price[t] = risk_free_price[t-1] * (1 + model.risk_free_rate / n_periods_per_year)
        
        # Transition to next state
        current_state = np.random.choice([0, 1], p=phi[current_state])
        
        # Get consumption growth for the new state
        lambda_t = model.lambda_values[current_state]
        
        # Update equity prices
        if current_state == 0:  # Growth state
            equity_price_optimal[t] = equity_price_optimal[t-1] * (1 + (lambda_1 * (w_optimal[0] + 1) / w_optimal[0] - 1) / n_periods_per_year)
            equity_price_actual[t] = equity_price_actual[t-1] * (1 + (lambda_1 * (w_actual[0] + 1) / w_actual[0] - 1) / n_periods_per_year)
        else:  # Contraction state
            equity_price_optimal[t] = equity_price_optimal[t-1] * (1 + (lambda_2 * (w_optimal[1] + 1) / w_optimal[1] - 1) / n_periods_per_year)
            equity_price_actual[t] = equity_price_actual[t-1] * (1 + (lambda_2 * (w_actual[1] + 1) / w_actual[1] - 1) / n_periods_per_year)
    
    # Create time axis
    time = np.arange(n_periods) / n_periods_per_year
    
    # Calculate strategy performance
    optimal_annual_return = (equity_price_optimal[-1] / equity_price_optimal[0]) ** (1 / n_years) - 1
    actual_annual_return = (equity_price_actual[-1] / equity_price_actual[0]) ** (1 / n_years) - 1
    risk_free_annual_return = (risk_free_price[-1] / risk_free_price[0]) ** (1 / n_years) - 1
    
    # Plot price paths
    plt.figure(figsize=(12, 6))
    plt.plot(time, equity_price_optimal, 'b-', label=f'Optimal α_e = {results["optimal_alpha_e"]:.2f} (Return: {optimal_annual_return:.2%})')
    plt.plot(time, equity_price_actual, 'g-', label=f'Historical α_e = {actual_alpha:.2f} (Return: {actual_annual_return:.2%})')
    plt.plot(time, risk_free_price, 'r-', label=f'Risk-free (Return: {risk_free_annual_return:.2%})')
    
    plt.xlabel('Time (years)')
    plt.ylabel('Price')
    plt.title('Simulated Price Paths')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig('simulated_price_paths.png')
    
    # Calculate drawdowns
    def calculate_drawdowns(prices):
        drawdowns = np.zeros_like(prices)
        peak = prices[0]
        for i in range(1, len(prices)):
            if prices[i] > peak:
                peak = prices[i]
            drawdowns[i] = (prices[i] / peak) - 1
        return drawdowns
    
    optimal_drawdowns = calculate_drawdowns(equity_price_optimal)
    actual_drawdowns = calculate_drawdowns(equity_price_actual)
    
    # Plot drawdowns
    plt.figure(figsize=(12, 6))
    plt.plot(time, optimal_drawdowns, 'b-', label=f'Optimal α_e = {results["optimal_alpha_e"]:.2f}')
    plt.plot(time, actual_drawdowns, 'g-', label=f'Historical α_e = {actual_alpha:.2f}')
    
    plt.xlabel('Time (years)')
    plt.ylabel('Drawdown')
    plt.title('Drawdowns')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('simulated_drawdowns.png')
    
    # Calculate rolling annual returns
    window = n_periods_per_year
    
    def calculate_rolling_returns(prices, window):
        returns = np.zeros(len(prices) - window)
        for i in range(len(returns)):
            returns[i] = (prices[i + window] / prices[i]) - 1
        return returns
    
    optimal_rolling_returns = calculate_rolling_returns(equity_price_optimal, window)
    actual_rolling_returns = calculate_rolling_returns(equity_price_actual, window)
    risk_free_rolling_returns = calculate_rolling_returns(risk_free_price, window)
    
    # Plot rolling annual returns
    plt.figure(figsize=(12, 6))
    plt.plot(time[window:], optimal_rolling_returns, 'b-', label=f'Optimal α_e = {results["optimal_alpha_e"]:.2f}')
    plt.plot(time[window:], actual_rolling_returns, 'g-', label=f'Historical α_e = {actual_alpha:.2f}')
    plt.plot(time[window:], risk_free_rolling_returns, 'r-', label='Risk-free')
    
    plt.xlabel('Time (years)')
    plt.ylabel('Annual Return')
    plt.title('Rolling Annual Returns')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('simulated_rolling_returns.png')
    
    # Calculate strategy statistics
    def calculate_stats(returns):
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        sharpe = mean_return / std_return if std_return > 0 else 0
        max_drawdown = np.min(calculate_drawdowns(returns + 1))
        return {
            'mean_return': mean_return,
            'std_return': std_return,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown
        }
    
    optimal_stats = calculate_stats(optimal_rolling_returns)
    actual_stats = calculate_stats(actual_rolling_returns)
    risk_free_stats = calculate_stats(risk_free_rolling_returns)
    
    return {
        'optimal_alpha': results['optimal_alpha_e'],
        'actual_alpha': actual_alpha,
        'optimal_stats': optimal_stats,
        'actual_stats': actual_stats,
        'risk_free_stats': risk_free_stats
    }

if __name__ == "__main__":
    # Run the model on historical periods
    run_historical_periods()
    
    # Run Monte Carlo simulations
    mc_results = simulate_monte_carlo()
    print("\nMonte Carlo Results:")
    print(f"Mean optimal α_e: {mc_results['alpha_mean']:.2f} ± {mc_results['alpha_std']:.2f}")
    print(f"95% CI for optimal α_e: ({mc_results['alpha_95ci'][0]:.2f}, {mc_results['alpha_95ci'][1]:.2f})")
    print(f"Mean optimal return: {mc_results['return_mean']:.4f} ± {mc_results['return_std']:.4f}")
    print(f"95% CI for optimal return: ({mc_results['return_95ci'][0]:.4f}, {mc_results['return_95ci'][1]:.4f})")
    
    # Simulate investment strategy
    strategy_results = simulate_investment_strategy()
    print("\nInvestment Strategy Results:")
    print(f"Optimal α_e: {strategy_results['optimal_alpha']:.2f}")
    print(f"Historical α_e: {strategy_results['actual_alpha']:.2f}")
    print("\nOptimal Strategy Stats:")
    print(f"Mean Annual Return: {strategy_results['optimal_stats']['mean_return']:.2%}")
    print(f"Standard Deviation: {strategy_results['optimal_stats']['std_return']:.2%}")
    print(f"Sharpe Ratio: {strategy_results['optimal_stats']['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {strategy_results['optimal_stats']['max_drawdown']:.2%}")
    print("\nHistorical Strategy Stats:")
    print(f"Mean Annual Return: {strategy_results['actual_stats']['mean_return']:.2%}")
    print(f"Standard Deviation: {strategy_results['actual_stats']['std_return']:.2%}")
    print(f"Sharpe Ratio: {strategy_results['actual_stats']['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {strategy_results['actual_stats']['max_drawdown']:.2%}")
    print("\nRisk-free Stats:")
    print(f"Mean Annual Return: {strategy_results['risk_free_stats']['mean_return']:.2%}")
    print(f"Standard Deviation: {strategy_results['risk_free_stats']['std_return']:.2%}")
    print(f"Sharpe Ratio: {strategy_results['risk_free_stats']['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {strategy_results['risk_free_stats']['max_drawdown']:.2%}")