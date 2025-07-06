import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, least_squares
from scipy.linalg import svd
import pandas as pd
from tqdm import tqdm

# Set random seed for reproducibility
np.random.seed(42)

class MultiCurveModel:
    """
    Implementation of a multi-curve interest rate model with Hull-White dynamics
    as described in the paper.
    """
    
    def __init__(self, num_tenors=2, dimension=1):
        """
        Initialize the model.
        
        Parameters:
        - num_tenors: Number of risk-sensitive rates (m in the paper)
        - dimension: Dimension of the driving Brownian motion (d in the paper)
        """
        self.m = num_tenors  # Number of tenors
        self.d = dimension   # Dimension of the Brownian motion
        
        # Parameter initialization
        self.a0 = 0.3  # Mean reversion parameter for risk-free rate
        self.sigma0 = 0.02  # Volatility parameter for risk-free rate
        
        # Parameters for risk-sensitive rates
        self.a = np.array([0.35, 0.37])[:self.m]  # Mean reversion parameters
        self.sigma = np.array([0.025, 0.03])[:self.m]  # Volatility parameters
        
        # Parameters for log-spread processes
        self.beta = np.array([0.4, 0.5])[:self.m]  # Volatility parameters
        
        # Combined parameter vector for calibration
        self.theta = np.concatenate([[self.a0, self.sigma0], 
                                    self.a, self.sigma, self.beta])
        
    def forward_rate_volatility(self, x, j):
        """
        Compute the volatility of forward rates.
        
        Parameters:
        - x: Time to maturity
        - j: Index (0 for risk-free, 1,...,m for risk-sensitive)
        
        Returns:
        - Volatility of the forward rate
        """
        if j == 0:
            return self.sigma0 * np.exp(-self.a0 * x)
        else:
            return self.sigma[j-1] * np.exp(-self.a[j-1] * x)
    
    def D_function(self, x, j):
        """
        Compute the D function as defined in equation (4.13) of the paper.
        
        Parameters:
        - x: Time to maturity
        - j: Index (0 for risk-free, 1,...,m for risk-sensitive)
        
        Returns:
        - D_j(x) value
        """
        lambda_j = self.forward_rate_volatility(x, j)
        
        if j == 0:
            a_j = self.a0
            return lambda_j * (1 - np.exp(-a_j * x)) / a_j
        else:
            a_j = self.a[j-1]
            return lambda_j * (1 - np.exp(-a_j * x)) / a_j
    
    def nelson_siegel(self, x, y, j):
        """
        Compute the Nelson-Siegel function for the forward rate curve.
        
        Parameters:
        - x: Time to maturity
        - y: Parameter vector (y0, y1, y2)
        - j: Index (0 for risk-free, 1,...,m for risk-sensitive)
        
        Returns:
        - Forward rate value
        """
        if j == 0:
            a_j = self.a0
        else:
            a_j = self.a[j-1]
            
        return y[0] + y[1] * np.exp(-a_j * x) + y[2] * x * np.exp(-a_j * x)
    
    def G_function(self, z, x, j, r_init, y_init):
        """
        Compute the G function for the forward rates as defined in equation (6.1).
        
        Parameters:
        - z: State vector (z0, z01, z11, z21, z31)
        - x: Time to maturity
        - j: Index (0 for risk-free, 1,...,m for risk-sensitive)
        - r_init: Initial forward rate curves
        - y_init: Initial log-spreads
        
        Returns:
        - Forward rate value
        """
        z0, z01, z11, z21, z31 = z
        
        if j <= self.m:  # Forward rate components
            if j == 0:
                sigma_j = self.sigma0
                a_j = self.a0
                beta_j = 0
            else:
                sigma_j = self.sigma[j-1]
                a_j = self.a[j-1]
                beta_j = self.beta[j-1]
            
            # Evaluate initial forward rate at shifted maturity
            r_init_shifted = self.nelson_siegel(x + z0, y_init, j)
            
            # Additional terms from equation (6.1)
            term1 = sigma_j * np.exp(-a_j * x) * (z01 - a_j * z11 + a_j**2 * z21 - a_j**3 * z31)
            term2 = 0.5 * sigma_j**2 / a_j * np.exp(-2 * a_j * x) * (np.exp(-2 * a_j * z0) - 1)
            term3 = -sigma_j / a_j * (sigma_j / a_j - beta_j * (j > 0)) * np.exp(-a_j * x) * (np.exp(-a_j * z0) - 1)
            
            return r_init_shifted + term1 + term2 + term3
        else:  # Log-spread components
            j_idx = j - self.m - 1  # Index of the log-spread
            sigma0 = self.sigma0
            a0 = self.a0
            sigma_j = self.sigma[j_idx]
            a_j = self.a[j_idx]
            beta_j = self.beta[j_idx]
            
            # Terms from equation (6.1)
            term1 = (sigma0 - sigma_j) * z11
            term2 = (-a0 * sigma0 + a_j * sigma_j) * z21
            term3 = ((a0**2 * sigma0) - (a_j**2 * sigma_j)) * z31
            term4 = beta_j * z01
            term5 = y_init[j_idx]
            
            # Integral term (simplified for simulation)
            term6 = z0 * (r_init[0][0] - r_init[j_idx+1][0])
            
            # Additional terms (simplified)
            term7 = 0.5 * (sigma0**2 / a0**2 - sigma_j**2 / a_j**2) * z0
            term8 = beta_j * sigma_j / a_j * z0
            term9 = -0.5 * beta_j**2 * z0
            
            return term1 + term2 + term3 + term4 + term5 + term6 + term7 + term8 + term9
    
    def zero_bond_price(self, forward_rates, maturity):
        """
        Compute the zero-bond price from forward rates.
        
        Parameters:
        - forward_rates: Array of forward rates
        - maturity: Maturity of the bond
        
        Returns:
        - Zero-bond price
        """
        # Numerical integration of forward rates
        x_grid = np.linspace(0, maturity, 100)
        dx = x_grid[1] - x_grid[0]
        
        # Interpolate forward rates at grid points
        f_values = np.zeros(len(x_grid))
        for i, x in enumerate(x_grid):
            f_values[i] = forward_rates(x)
        
        # Integrate to get the zero-bond price
        integral = np.sum(f_values) * dx
        return np.exp(-integral)
    
    def zero_bond_yield(self, forward_rates, maturity):
        """
        Compute the yield of a zero-bond from forward rates.
        
        Parameters:
        - forward_rates: Array of forward rates
        - maturity: Maturity of the bond
        
        Returns:
        - Zero-bond yield
        """
        price = self.zero_bond_price(forward_rates, maturity)
        return -np.log(price) / maturity
    
    def calibrate_to_market_data(self, market_data, maturities, initial_params=None):
        """
        Calibrate the model to market data using the algorithm described in Section 6.3.
        
        Parameters:
        - market_data: Dictionary with market data for each day
        - maturities: List of maturities
        - initial_params: Initial parameter guess
        
        Returns:
        - Calibrated parameters
        """
        if initial_params is not None:
            self.theta = initial_params
        
        # Step P.2: Minimize the sum of squared residuals
        def objective_function(theta):
            # Update model parameters
            self.a0, self.sigma0 = theta[0], theta[1]
            self.a = theta[2:2+self.m]
            self.sigma = theta[2+self.m:2+2*self.m]
            self.beta = theta[2+2*self.m:2+3*self.m]
            
            total_error = 0
            
            for date, data in market_data.items():
                # Step P.1: Minimize residuals for each date
                z1, y = self.estimate_state_variables(data, maturities)
                
                # Compute residuals
                residuals = self.compute_residuals(data, maturities, z1, y)
                total_error += np.sum(residuals**2)
            
            return total_error
        
        # Optimize using a trust-region reflective algorithm
        bounds = [(0.001, 1.0)] * len(self.theta)  # Ensure positive parameters
        result = minimize(objective_function, self.theta, method='trust-constr', bounds=bounds)
        
        # Update model parameters
        self.theta = result.x
        self.a0, self.sigma0 = self.theta[0], self.theta[1]
        self.a = self.theta[2:2+self.m]
        self.sigma = self.theta[2+self.m:2+2*self.m]
        self.beta = self.theta[2+2*self.m:2+3*self.m]
        
        return result
    
    def estimate_state_variables(self, data, maturities):
        """
        Estimate state variables using SVD algorithm as described in step P.1.
        
        Parameters:
        - data: Market data for a specific date
        - maturities: List of maturities
        
        Returns:
        - Estimated state variables (z1, y)
        """
        # Extract bond prices and log-spreads from data
        B0 = data['B0']  # Risk-free ZCB prices
        B = [data[f'B{j+1}'] for j in range(self.m)]  # Risk-sensitive ZCB prices
        Y = data['Y']  # Log-spreads
        
        # Set up the linear system for state estimation
        # This is a simplification of the actual procedure
        
        # For demonstration, we'll use a simpler approach
        z1 = np.zeros(4)  # z01, z11, z21, z31
        y = np.zeros(3)   # y0, y1, y2
        
        # Use least squares to estimate z1 and y
        def residual_func(params):
            z1_est = params[:4]
            y_est = params[4:7]
            
            residuals = []
            
            # Compute residuals for risk-free bond prices
            for i, maturity in enumerate(maturities):
                r_function = lambda x: self.G_function([0, *z1_est], x, 0, None, y_est)
                yield_model = -np.log(self.zero_bond_price(r_function, maturity)) / maturity
                yield_market = -np.log(B0[i]) / maturity
                residuals.append(yield_model - yield_market)
            
            # Compute residuals for risk-sensitive bond prices
            for j in range(self.m):
                for i, maturity in enumerate(maturities):
                    r_function = lambda x: self.G_function([0, *z1_est], x, j+1, None, y_est)
                    yield_model = -np.log(self.zero_bond_price(r_function, maturity)) / maturity
                    yield_market = -np.log(B[j][i]) / maturity
                    residuals.append(yield_model - yield_market)
            
            # Compute residuals for log-spreads
            for j in range(self.m):
                spread_model = self.G_function([0, *z1_est], 0, self.m+1+j, None, y_est)
                residuals.append(spread_model - Y[j])
            
            return np.array(residuals)
        
        initial_guess = np.zeros(7)  # [z1, y]
        result = least_squares(residual_func, initial_guess)
        
        z1 = result.x[:4]
        y = result.x[4:7]
        
        return z1, y
    
    def compute_residuals(self, data, maturities, z1, y):
        """
        Compute residuals between model and market data.
        
        Parameters:
        - data: Market data for a specific date
        - maturities: List of maturities
        - z1: State variables
        - y: Nelson-Siegel parameters
        
        Returns:
        - Residuals vector
        """
        # Extract bond prices and log-spreads from data
        B0 = data['B0']  # Risk-free ZCB prices
        B = [data[f'B{j+1}'] for j in range(self.m)]  # Risk-sensitive ZCB prices
        Y = data['Y']  # Log-spreads
        
        residuals = []
        
        # Compute residuals for risk-free bond prices
        for i, maturity in enumerate(maturities):
            r_function = lambda x: self.G_function([0, *z1], x, 0, None, y)
            yield_model = -np.log(self.zero_bond_price(r_function, maturity)) / maturity
            yield_market = -np.log(B0[i]) / maturity
            residuals.append(yield_model - yield_market)
        
        # Compute residuals for risk-sensitive bond prices
        for j in range(self.m):
            for i, maturity in enumerate(maturities):
                r_function = lambda x: self.G_function([0, *z1], x, j+1, None, y)
                yield_model = -np.log(self.zero_bond_price(r_function, maturity)) / maturity
                yield_market = -np.log(B[j][i]) / maturity
                residuals.append(yield_model - yield_market)
        
        # Compute residuals for log-spreads
        for j in range(self.m):
            spread_model = self.G_function([0, *z1], 0, self.m+1+j, None, y)
            residuals.append(spread_model - Y[j])
        
        return np.array(residuals)
    
    def simulate_market_data(self, num_days=100, maturities=[0.5, 1, 2, 3, 5, 7, 10]):
        """
        Simulate market data for testing the calibration algorithm.
        
        Parameters:
        - num_days: Number of days to simulate
        - maturities: List of maturities to simulate
        
        Returns:
        - Simulated market data
        """
        dt = 1/252  # Daily time step in years
        
        # Initialize state process
        z = np.zeros((num_days, 5))  # (z0, z01, z11, z21, z31)
        z0 = np.arange(0, num_days*dt, dt)  # Time component
        
        # Simulate Brownian motion
        dW = np.random.normal(0, np.sqrt(dt), num_days-1)
        
        # Coefficients for the state process (from equation 4.11)
        a = np.zeros((num_days-1, 4))  # z01, z11, z21, z31
        b = np.zeros((num_days-1, 4))
        
        # Simplified coefficients for demonstration
        a[:, 0] = 0  # a01 = 0
        a[:, 1] = z[:-1, 2]  # a11 = z21
        a[:, 2] = z[:-1, 3]  # a21 = z31
        a[:, 3] = -(self.a0 + self.a[0] + self.a[1]) * z[:-1, 3]  # a31 = -α3*z31
        
        b[:, 0] = 1  # b01 = 1
        
        # Simulate state process
        for i in range(num_days-1):
            z[i+1, 0] = z0[i+1]
            z[i+1, 1:] = z[i, 1:] + a[i, :] * dt + b[i, :] * dW[i]
        
        # Initialize Nelson-Siegel parameters
        y_init = np.array([0.02, -0.01, 0.005])  # Initial values for (y0, y1, y2)
        
        # Initial log-spreads
        y_spread_init = np.array([0.001, 0.002])[:self.m]
        
        # Generate market data
        market_data = {}
        
        for i in range(num_days):
            # Forward rate functions
            r_init = []
            for j in range(self.m + 1):
                r_init.append(lambda x, j=j: self.nelson_siegel(x, y_init, j))
            
            # Bond prices
            B0 = []
            B1 = []
            B2 = []
            
            for maturity in maturities:
                # Risk-free forward rate function
                r0_func = lambda x: self.G_function(z[i], x, 0, r_init, y_spread_init)
                B0.append(self.zero_bond_price(r0_func, maturity))
                
                # Risk-sensitive forward rate functions
                r1_func = lambda x: self.G_function(z[i], x, 1, r_init, y_spread_init)
                B1.append(self.zero_bond_price(r1_func, maturity))
                
                if self.m > 1:
                    r2_func = lambda x: self.G_function(z[i], x, 2, r_init, y_spread_init)
                    B2.append(self.zero_bond_price(r2_func, maturity))
            
            # Log-spreads
            Y = []
            for j in range(self.m):
                Y.append(self.G_function(z[i], 0, self.m+1+j, r_init, y_spread_init))
            
            # Store market data
            market_data[i] = {
                'B0': np.array(B0),
                'B1': np.array(B1),
                'Y': np.array(Y)
            }
            
            if self.m > 1:
                market_data[i]['B2'] = np.array(B2)
        
        return market_data, maturities, z

def test_model_calibration():
    """
    Test the calibration of the multi-curve model using simulated data.
    """
    # Create a model with 2 risk-sensitive rates
    model = MultiCurveModel(num_tenors=2)
    
    # Store the true parameters
    true_params = model.theta.copy()
    
    # Simulate market data
    print("Simulating market data...")
    market_data, maturities, true_state = model.simulate_market_data(num_days=120)
    
    # Perturb parameters for initial guess
    perturbed_params = true_params * (1 + 0.2 * np.random.randn(len(true_params)))
    perturbed_params = np.abs(perturbed_params)  # Ensure positive parameters
    
    # Split data into training and testing
    train_data = {k: v for k, v in market_data.items() if k < 100}
    test_data = {k: v for k, v in market_data.items() if k >= 100}
    
    # Calibrate the model
    print("Calibrating the model...")
    model.calibrate_to_market_data(train_data, maturities, initial_params=perturbed_params)
    
    # Compare estimated parameters to true parameters
    print("\nParameter Comparison:")
    print(f"{'Parameter':<10} {'True':<10} {'Estimated':<10} {'Relative Error':<15}")
    print("-" * 45)
    param_names = ['a0', 'sigma0'] + [f'a{i+1}' for i in range(model.m)] + \
                  [f'sigma{i+1}' for i in range(model.m)] + [f'beta{i+1}' for i in range(model.m)]
    
    for i, (name, true, est) in enumerate(zip(param_names, true_params, model.theta)):
        rel_error = np.abs(true - est) / np.abs(true) if true != 0 else np.abs(est)
        print(f"{name:<10} {true:<10.5f} {est:<10.5f} {rel_error:<15.5f}")
    
    # Evaluate model performance on test data
    print("\nEvaluating model performance on test data...")
    test_errors = []
    
    for date, data in test_data.items():
        # Estimate state variables
        z1, y = model.estimate_state_variables(data, maturities)
        
        # Compute residuals
        residuals = model.compute_residuals(data, maturities, z1, y)
        
        # Compute RMSE
        rmse = np.sqrt(np.mean(residuals**2))
        test_errors.append(rmse)
    
    avg_rmse = np.mean(test_errors)
    print(f"Average RMSE on test data: {avg_rmse:.6f}")
    
    # Visualize results
    # 1. Yield curves comparison for the last day in test data
    last_date = max(test_data.keys())
    last_data = test_data[last_date]
    
    z1, y = model.estimate_state_variables(last_data, maturities)
    
    plt.figure(figsize=(15, 10))
    
    # Risk-free yield curve
    plt.subplot(3, 1, 1)
    market_yields_0 = -np.log(last_data['B0']) / np.array(maturities)
    
    model_yields_0 = []
    for maturity in maturities:
        r0_func = lambda x: model.G_function([0, *z1], x, 0, None, y)
        model_yields_0.append(-np.log(model.zero_bond_price(r0_func, maturity)) / maturity)
    
    plt.plot(maturities, market_yields_0 * 100, 'o-', label='Market (Simulated)')
    plt.plot(maturities, np.array(model_yields_0) * 100, 's--', label='Model Calibrated')
    plt.title('Risk-free Yield Curve')
    plt.xlabel('Maturity (years)')
    plt.ylabel('Yield (%)')
    plt.legend()
    plt.grid(True)
    
    # 3M Yield curve
    plt.subplot(3, 1, 2)
    market_yields_1 = -np.log(last_data['B1']) / np.array(maturities)
    
    model_yields_1 = []
    for maturity in maturities:
        r1_func = lambda x: model.G_function([0, *z1], x, 1, None, y)
        model_yields_1.append(-np.log(model.zero_bond_price(r1_func, maturity)) / maturity)
    
    plt.plot(maturities, market_yields_1 * 100, 'o-', label='Market (Simulated)')
    plt.plot(maturities, np.array(model_yields_1) * 100, 's--', label='Model Calibrated')
    plt.title('3M Yield Curve')
    plt.xlabel('Maturity (years)')
    plt.ylabel('Yield (%)')
    plt.legend()
    plt.grid(True)
    
    # 6M Yield curve
    if model.m > 1:
        plt.subplot(3, 1, 3)
        market_yields_2 = -np.log(last_data['B2']) / np.array(maturities)
        
        model_yields_2 = []
        for maturity in maturities:
            r2_func = lambda x: model.G_function([0, *z1], x, 2, None, y)
            model_yields_2.append(-np.log(model.zero_bond_price(r2_func, maturity)) / maturity)
        
        plt.plot(maturities, market_yields_2 * 100, 'o-', label='Market (Simulated)')
        plt.plot(maturities, np.array(model_yields_2) * 100, 's--', label='Model Calibrated')
        plt.title('6M Yield Curve')
        plt.xlabel('Maturity (years)')
        plt.ylabel('Yield (%)')
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('yield_curves_comparison.png')
    
    # 2. Parameter stability analysis
    # Simulate stability analysis as described in Section 6.5.2
    print("\nPerforming parameter stability analysis...")
    
    window_size = 30  # 30-day window
    num_iterations = 20
    
    param_history = np.zeros((num_iterations, len(model.theta)))
    
    for i in range(num_iterations):
        # Create a rolling window of data
        window_data = {k: v for k, v in market_data.items() if i <= k < i + window_size}
        
        # Use the previously calibrated parameters as initial guess
        initial_guess = model.theta if i > 0 else perturbed_params
        
        # Calibrate the model
        model.calibrate_to_market_data(window_data, maturities, initial_params=initial_guess)
        
        # Store the calibrated parameters
        param_history[i] = model.theta
    
    # Compute parameter stability statistics
    param_avg = np.mean(param_history, axis=0)
    param_std = np.std(param_history, axis=0)
    
    print("\nParameter Stability Analysis:")
    print(f"{'Parameter':<10} {'Average':<10} {'Std Dev':<10} {'CV (%)':<10}")
    print("-" * 40)
    
    for i, name in enumerate(param_names):
        cv = param_std[i] / param_avg[i] * 100 if param_avg[i] != 0 else float('inf')
        print(f"{name:<10} {param_avg[i]:<10.5f} {param_std[i]:<10.5f} {cv:<10.2f}")
    
    # Visualize parameter stability
    plt.figure(figsize=(15, 10))
    
    for i, name in enumerate(param_names):
        plt.subplot(3, 3, i+1)
        plt.plot(range(num_iterations), param_history[:, i])
        plt.title(f'Parameter: {name}')
        plt.xlabel('Iteration')
        plt.ylabel('Value')
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('parameter_stability.png')
    
    # 3. Analysis of state variables
    # Compare estimated state variables with true state variables
    plt.figure(figsize=(15, 10))
    
    # Extract the true state variables for the test period
    true_state_test = true_state[100:, 1:]  # Exclude z0 (time)
    
    # Estimate state variables for the test period
    estimated_state = np.zeros((len(test_data), 4))
    
    for i, (date, data) in enumerate(test_data.items()):
        z1, _ = model.estimate_state_variables(data, maturities)
        estimated_state[i] = z1
    
    # Plot comparison
    state_labels = ['z01', 'z11', 'z21', 'z31']
    
    for i in range(4):
        plt.subplot(2, 2, i+1)
        plt.plot(range(len(test_data)), true_state_test[:len(test_data), i], 'b-', label='True')
        plt.plot(range(len(test_data)), estimated_state[:, i], 'r--', label='Estimated')
        plt.title(f'State Variable: {state_labels[i]}')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('state_variables_comparison.png')
    
    return model, market_data, maturities

def test_consistency_property():
    """
    Test the consistency property of the multi-curve model.
    
    This checks if the model always generates forward curves that belong to 
    the specified parameterized family when starting from a curve in that family.
    """
    print("\nTesting consistency property...")
    
    # Create a model
    model = MultiCurveModel(num_tenors=2)
    
    # Define a parameterized family of forward curves (Nelson-Siegel)
    y_init = np.array([0.02, -0.01, 0.005])  # (y0, y1, y2)
    
    # Initialize forward rate functions
    r_init = []
    for j in range(model.m + 1):
        r_init.append(lambda x, j=j: model.nelson_siegel(x, y_init, j))
    
    # Initial log-spreads
    y_spread_init = np.array([0.001, 0.002])[:model.m]
    
    # Generate FDR from the parameterized family
    print("Simulating forward curves evolution...")
    
    # Set up parameters for simulation
    dt = 1/252  # Daily time step
    num_days = 100
    maturities = np.linspace(0.1, 10, 20)  # Maturities for evaluation
    
    # Initialize state process
    z = np.zeros((num_days, 5))  # (z0, z01, z11, z21, z31)
    z[0, 0] = 0  # Initial time
    
    # Simulate Brownian motion
    dW = np.random.normal(0, np.sqrt(dt), num_days-1)
    
    # Coefficients for the state process (from equation 4.11)
    a = np.zeros((num_days-1, 4))  # z01, z11, z21, z31
    b = np.zeros((num_days-1, 4))
    
    # Simplified coefficients for demonstration
    a[:, 0] = 0  # a01 = 0
    a[:, 1] = z[:-1, 2]  # a11 = z21
    a[:, 2] = z[:-1, 3]  # a21 = z31
    a[:, 3] = -(model.a0 + model.a[0] + model.a[1]) * z[:-1, 3]  # a31 = -α3*z31
    
    b[:, 0] = 1  # b01 = 1
    
    # Simulate state process
    for i in range(num_days-1):
        z[i+1, 0] = z[i, 0] + dt
        z[i+1, 1:] = z[i, 1:] + a[i, :] * dt + b[i, :] * dW[i]
    
    # Store forward curves for each day
    forward_curves = np.zeros((num_days, model.m+1, len(maturities)))
    
    for i in range(num_days):
        for j in range(model.m + 1):
            for k, maturity in enumerate(maturities):
                forward_curves[i, j, k] = model.G_function(z[i], maturity, j, r_init, y_spread_init)
    
    # Check if the evolution is consistent with the parameterized family
    # For each day, try to fit a Nelson-Siegel curve to the simulated forward curves
    fit_errors = np.zeros((num_days, model.m+1))
    
    for i in range(num_days):
        for j in range(model.m + 1):
            # Data to fit
            curve_data = forward_curves[i, j]
            
            # Fit Nelson-Siegel parameters
            def objective(params):
                y0, y1, y2 = params
                ns_values = []
                for k, maturity in enumerate(maturities):
                    if j == 0:
                        a_j = model.a0
                    else:
                        a_j = model.a[j-1]
                    ns_values.append(y0 + y1 * np.exp(-a_j * maturity) + y2 * maturity * np.exp(-a_j * maturity))
                return np.sum((np.array(ns_values) - curve_data)**2)
            
            # Initial guess
            initial_guess = y_init
            
            # Optimize
            result = minimize(objective, initial_guess, method='BFGS')
            
            # Calculate RMSE of the fit
            y0, y1, y2 = result.x
            ns_values = []
            for k, maturity in enumerate(maturities):
                if j == 0:
                    a_j = model.a0
                else:
                    a_j = model.a[j-1]
                ns_values.append(y0 + y1 * np.exp(-a_j * maturity) + y2 * maturity * np.exp(-a_j * maturity))
            
            fit_errors[i, j] = np.sqrt(np.mean((np.array(ns_values) - curve_data)**2))
    
    # Visualize the fitting errors
    plt.figure(figsize=(12, 8))
    
    for j in range(model.m + 1):
        label = 'Risk-free' if j == 0 else f'Risk-sensitive {j}'
        plt.semilogy(range(num_days), fit_errors[:, j], label=label)
    
    plt.title('Consistency Check: Fitting Errors over Time')
    plt.xlabel('Time (days)')
    plt.ylabel('RMSE (log scale)')
    plt.legend()
    plt.grid(True)
    plt.savefig('consistency_check.png')
    
    # Check if the fitting errors are small (indicating consistency)
    max_error = np.max(fit_errors)
    avg_error = np.mean(fit_errors)
    
    print(f"Maximum fitting error: {max_error:.6f}")
    print(f"Average fitting error: {avg_error:.6f}")
    print(f"Consistency property: {'SATISFIED' if max_error < 1e-3 else 'NOT SATISFIED'}")
    
    # Visualize forward curves evolution
    plt.figure(figsize=(15, 10))
    
    # Plot forward curves at different times
    time_points = [0, 25, 50, 75, 99]  # Days to plot
    
    for j in range(model.m + 1):
        plt.subplot(1, model.m+1, j+1)
        
        for t in time_points:
            plt.plot(maturities, forward_curves[t, j], label=f'Day {t}')
        
        title = 'Risk-free' if j == 0 else f'Risk-sensitive {j}'
        plt.title(f'{title} Forward Curves')
        plt.xlabel('Maturity (years)')
        plt.ylabel('Forward Rate')
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('forward_curves_evolution.png')
    
    return forward_curves, fit_errors

if __name__ == "__main__":
    print("Testing Multi-Curve Interest Rate Models")
    print("=======================================")
    
    # Test model calibration
    model, market_data, maturities = test_model_calibration()
    
    # Test consistency property
    forward_curves, fit_errors = test_consistency_property()
    
    print("\nAll tests completed.")