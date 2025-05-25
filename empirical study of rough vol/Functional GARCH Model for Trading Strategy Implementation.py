import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import block_diag
from scipy.integrate import trapz
from scipy.optimize import minimize
from numpy.linalg import inv, cholesky
from scipy.special import comb
import pandas as pd
from tqdm import tqdm

# Set random seed for reproducibility
np.random.seed(42)

class FunctionalGARCH:
    """
    Implements a Functional GARCH(1,1) model for intraday return data
    """
    def __init__(self, n_basis=3, intraday_points=100):
        """
        Initialize the Functional GARCH model
        
        Parameters:
        -----------
        n_basis : int
            Number of basis functions to use
        intraday_points : int
            Number of intraday time points
        """
        self.n_basis = n_basis
        self.intraday_points = intraday_points
        self.u = np.linspace(0, 1, intraday_points)  # Normalized intraday time
        
        # Initialize model parameters
        # delta, alpha coefficients, beta coefficients
        self.params = None
        self.basis_functions = None
        
        # Create basis functions (Bernstein polynomials)
        self._create_basis_functions()
        
        # Initialize simulation parameters
        self.delta0 = None
        self.alpha0 = None
        self.beta0 = None
    
    def _create_basis_functions(self):
        """Create Bernstein polynomial basis functions"""
        basis_functions = np.zeros((self.n_basis, len(self.u)))
        
        for k in range(self.n_basis):
            # Bernstein polynomials of degree n-1
            coef = comb(self.n_basis-1, k, exact=True)
            basis_functions[k] = coef * (self.u**k) * ((1-self.u)**(self.n_basis-1-k))
        
        # Normalize basis functions to have max value of 1
        for k in range(self.n_basis):
            max_val = np.max(basis_functions[k])
            if max_val > 0:
                basis_functions[k] = basis_functions[k] / max_val
                
        self.basis_functions = basis_functions
    
    def _setup_simulation_parameters(self):
        """Setup parameters for simulation"""
        # delta: intercept function
        self.delta0 = np.zeros(self.n_basis)
        self.delta0[0] = 0.01  # Constant term
        
        # Set alpha kernel parameters (n_basis x n_basis matrix)
        self.alpha0 = np.zeros((self.n_basis, self.n_basis))
        self.alpha0[0, 0] = 0.05
        
        # Set beta kernel parameters (n_basis x n_basis matrix)
        self.beta0 = np.zeros((self.n_basis, self.n_basis))
        self.beta0[0, 0] = 0.9
        
        # Check stationarity condition
        max_eigenvalue = np.max(np.abs(np.linalg.eigvals(self.alpha0 + self.beta0)))
        if max_eigenvalue >= 1:
            print(f"Warning: Model may not be stationary. Max eigenvalue: {max_eigenvalue}")
    
    def simulate(self, n_days, burn_in=100):
        """
        Simulate return curves from a functional GARCH(1,1) model
        
        Parameters:
        -----------
        n_days : int
            Number of days to simulate
        burn_in : int
            Number of burn-in days
            
        Returns:
        --------
        y : ndarray (n_days, intraday_points)
            Simulated return curves
        sigma2 : ndarray (n_days, intraday_points)
            Volatility curves
        """
        if self.delta0 is None:
            self._setup_simulation_parameters()
        
        # Initialize arrays
        total_days = n_days + burn_in
        y = np.zeros((total_days, self.intraday_points))
        sigma2 = np.zeros((total_days, self.intraday_points))
        
        # Initialize coefficients for sigma^2
        h = np.zeros((total_days, self.n_basis))
        
        # Start with unconditional variance
        A = np.eye(self.n_basis) - self.alpha0 - self.beta0
        h[0] = np.linalg.solve(A, self.delta0)
        
        # Ensure h[0] is positive
        h[0] = np.maximum(h[0], 1e-6)
        
        # Initial volatility curve
        sigma2[0] = np.maximum(np.dot(self.basis_functions.T, h[0]), 1e-6)
        
        # Simulate the process
        for t in range(1, total_days):
            # Simulate innovation process (Ornstein-Uhlenbeck process)
            epsilon = self._simulate_ou_process()
            
            # Ensure sigma2 is positive
            sigma2[t] = np.maximum(np.dot(self.basis_functions.T, h[t-1]), 1e-6)
            
            # Generate return curve
            y[t] = np.sqrt(sigma2[t]) * epsilon
            
            # Compute projections of y_t^2 onto basis functions
            y2_proj = np.array([trapz(y[t]**2 * self.basis_functions[j], self.u) 
                               for j in range(self.n_basis)])
            
            # Update h coefficients and ensure they're positive
            h[t] = np.maximum(self.delta0 + np.dot(self.alpha0, y2_proj) + np.dot(self.beta0, h[t-1]), 1e-6)
        
        # Discard burn-in period
        return y[burn_in:], sigma2[burn_in:]
    
    def _simulate_ou_process(self):
        """
        Simulate an Ornstein-Uhlenbeck process on [0,1]
        
        Returns:
        --------
        epsilon : ndarray
            Simulated OU process
        """
        # Parameters
        theta = 0.5  # Mean reversion speed
        sigma = 0.2  # Volatility
        dt = 1.0 / (self.intraday_points - 1)
        
        # Initialize process
        epsilon = np.zeros(self.intraday_points)
        epsilon[0] = np.random.normal(0, 1)
        
        # Simulate OU process
        for i in range(1, self.intraday_points):
            epsilon[i] = epsilon[i-1] * np.exp(-theta * dt) + \
                         sigma * np.sqrt((1 - np.exp(-2 * theta * dt)) / (2 * theta)) * \
                         np.random.normal(0, 1)
        
        # Standardize to mean 0, variance 1
        std = np.std(epsilon)
        if std > 0:
            epsilon = (epsilon - np.mean(epsilon)) / std
        else:
            epsilon = np.random.normal(0, 1, self.intraday_points)
        
        return epsilon
    
    def _log_likelihood(self, params, y):
        """
        Compute negative log quasi-likelihood for Functional GARCH
        
        Parameters:
        -----------
        params : ndarray
            Model parameters (delta, vec(alpha), vec(beta))
        y : ndarray (n_days, intraday_points)
            Return curves
            
        Returns:
        --------
        neg_ll : float
            Negative log-likelihood
        """
        n_days = y.shape[0]
        
        # Reshape parameters
        delta = params[:self.n_basis]
        alpha_vec = params[self.n_basis:self.n_basis*(1+self.n_basis)]
        beta_vec = params[self.n_basis*(1+self.n_basis):]
        
        alpha = alpha_vec.reshape((self.n_basis, self.n_basis))
        beta = beta_vec.reshape((self.n_basis, self.n_basis))
        
        # Initialize h and y2_proj
        h = np.zeros((n_days, self.n_basis))
        h[0] = delta  # Initial value
        
        # Compute quasi-likelihood
        ll = 0
        
        for t in range(1, n_days):
            # Volatility curve (ensure positivity)
            sigma2_t = np.maximum(np.dot(self.basis_functions.T, h[t-1]), 1e-6)
            
            # Compute projections of y_t^2 onto basis functions
            y2_proj = np.array([trapz(y[t]**2 * self.basis_functions[j], self.u) 
                               for j in range(self.n_basis)])
            
            # Compute log-likelihood term
            for m in range(self.n_basis):
                sigma2_proj_m = trapz(sigma2_t * self.basis_functions[m], self.u)
                # Ensure sigma2_proj_m is positive
                sigma2_proj_m = max(sigma2_proj_m, 1e-6)
                y2_proj_m = y2_proj[m]
                
                ll += (y2_proj_m / sigma2_proj_m + np.log(sigma2_proj_m))
            
            # Update h coefficients for next step (ensure positivity)
            h[t] = np.maximum(delta + np.dot(alpha, y2_proj) + np.dot(beta, h[t-1]), 1e-6)
        
        return ll
    
    def fit(self, y, method='L-BFGS-B', verbose=True):
        """
        Fit Functional GARCH model to return curves using QMLE
        
        Parameters:
        -----------
        y : ndarray (n_days, intraday_points)
            Return curves
        method : str
            Optimization method for scipy.optimize.minimize
        verbose : bool
            Whether to print progress information
            
        Returns:
        --------
        self : FunctionalGARCH
            Fitted model
        """
        n_days = y.shape[0]
        
        # Initial parameter values
        delta_init = np.ones(self.n_basis) * 0.01
        alpha_init = np.eye(self.n_basis) * 0.05
        beta_init = np.eye(self.n_basis) * 0.8
        
        initial_params = np.concatenate([
            delta_init,
            alpha_init.flatten(),
            beta_init.flatten()
        ])
        
        # Parameter bounds to ensure positivity
        bounds = [(1e-6, None)] * self.n_basis  # delta > 0
        bounds += [(0, None)] * (self.n_basis**2)  # alpha >= 0
        bounds += [(0, 0.999)] * (self.n_basis**2)  # 0 <= beta < 1
        
        # Fit the model
        if verbose:
            print("Fitting Functional GARCH model...")
        
        result = minimize(
            lambda params: self._log_likelihood(params, y),
            initial_params,
            method=method,
            bounds=bounds,
            options={'maxiter': 1000}
        )
        
        if verbose:
            print(f"Optimization result: {result.message}")
            print(f"Function evaluations: {result.nfev}")
        
        # Extract fitted parameters
        self.params = result.x
        self.delta = self.params[:self.n_basis]
        self.alpha = self.params[self.n_basis:self.n_basis*(1+self.n_basis)].reshape((self.n_basis, self.n_basis))
        self.beta = self.params[self.n_basis*(1+self.n_basis):].reshape((self.n_basis, self.n_basis))
        
        return self
    
    def predict_volatility(self, y, h_ahead=1):
        """
        Predict future volatility curves
        
        Parameters:
        -----------
        y : ndarray (n_days, intraday_points)
            Historical return curves
        h_ahead : int
            Number of steps ahead to predict
            
        Returns:
        --------
        sigma2_pred : ndarray (h_ahead, intraday_points)
            Predicted volatility curves
        """
        if self.params is None:
            raise ValueError("Model must be fitted before prediction")
        
        n_days = y.shape[0]
        
        # Initialize h coefficients
        h = np.zeros((n_days + h_ahead, self.n_basis))
        
        # Compute historical h values
        for t in range(1, n_days):
            # Compute projections of y_t^2 onto basis functions
            y2_proj = np.array([trapz(y[t]**2 * self.basis_functions[j], self.u) 
                               for j in range(self.n_basis)])
            
            # Update h coefficients and ensure they're positive
            h[t] = np.maximum(self.delta + np.dot(self.alpha, y2_proj) + np.dot(self.beta, h[t-1]), 1e-6)
        
        # Forecast future h values
        for t in range(n_days, n_days + h_ahead):
            h[t] = np.maximum(self.delta + np.dot(self.beta, h[t-1]), 1e-6)
        
        # Convert h coefficients to volatility curves (ensure positivity)
        sigma2_pred = np.zeros((h_ahead, self.intraday_points))
        for i in range(h_ahead):
            sigma2_pred[i] = np.maximum(np.dot(self.basis_functions.T, h[n_days+i]), 1e-6)
        
        return sigma2_pred


class VolatilityTradingStrategy:
    """
    Implements a trading strategy based on volatility forecasts
    """
    def __init__(self, model, risk_aversion=0.5):
        """
        Initialize the trading strategy
        
        Parameters:
        -----------
        model : FunctionalGARCH
            Fitted functional GARCH model
        risk_aversion : float
            Risk aversion parameter
        """
        self.model = model
        self.risk_aversion = risk_aversion
    
    def backtest(self, y, sigma2, transaction_cost=0.001):
        """
        Backtest the trading strategy
        
        Parameters:
        -----------
        y : ndarray (n_days, intraday_points)
            Return curves
        sigma2 : ndarray (n_days, intraday_points)
            Volatility curves
        transaction_cost : float
            Transaction cost as a fraction of position size
            
        Returns:
        --------
        results : dict
            Dictionary containing backtest results
        """
        n_days = y.shape[0]
        
        # Initialize variables
        positions = np.zeros(n_days)
        returns = np.zeros(n_days)
        
        # Calculate integrated volatility (daily)
        integrated_vol = np.sqrt(trapz(sigma2, axis=1))
        
        # Calculate daily returns (sum of intraday returns)
        daily_returns = np.sum(y, axis=1)
        
        # Strategy: inverse volatility weighting
        # When predicted volatility is high, reduce position size
        for t in range(1, n_days):
            try:
                # Predict next day's volatility
                sigma2_pred = self.model.predict_volatility(y[:t], h_ahead=1)
                pred_integrated_vol = np.sqrt(trapz(sigma2_pred[0]))
                
                # Determine position size inversely proportional to volatility
                # With a safeguard against very small volatility
                pred_integrated_vol = max(pred_integrated_vol, 1e-4)
                target_position = 1.0 / (self.risk_aversion * pred_integrated_vol)
                
                # Limit position size
                target_position = np.clip(target_position, -3, 3)
                
                # Transaction costs
                tc = np.abs(target_position - positions[t-1]) * transaction_cost
                
                # Update position
                positions[t] = target_position
                
                # Calculate return (position * return - transaction costs)
                returns[t] = positions[t] * daily_returns[t] - tc
                
            except Exception as e:
                print(f"Error at day {t}: {e}")
                positions[t] = positions[t-1]
                returns[t] = 0
        
        # Calculate cumulative returns with safeguard against numerical issues
        returns = np.clip(returns, -0.5, 0.5)  # Prevent extreme values
        cum_returns = np.cumprod(1 + returns) - 1
        
        # Calculate performance metrics with safeguards
        if np.std(returns) > 0:
            sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)
        else:
            sharpe_ratio = 0
            
        max_drawdown = 0
        try:
            peak = np.maximum.accumulate(cum_returns)
            drawdown = peak - cum_returns
            max_drawdown = np.max(drawdown)
        except:
            pass
        
        results = {
            'positions': positions,
            'returns': returns,
            'cum_returns': cum_returns,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'total_return': cum_returns[-1]
        }
        
        return results
    
    def optimized_strategy(self, y, window_size=60, transaction_cost=0.001):
        """
        Implement an optimized trading strategy with rolling estimation
        
        Parameters:
        -----------
        y : ndarray (n_days, intraday_points)
            Return curves
        window_size : int
            Size of the rolling window for model re-estimation
        transaction_cost : float
            Transaction cost as a fraction of position size
            
        Returns:
        --------
        results : dict
            Dictionary containing backtest results
        """
        n_days = y.shape[0]
        
        # Initialize variables
        positions = np.zeros(n_days)
        returns = np.zeros(n_days)
        realized_vol = np.zeros(n_days)
        
        # Calculate daily returns (sum of intraday returns)
        daily_returns = np.sum(y, axis=1)
        
        # We need at least window_size days for initial estimation
        for t in tqdm(range(window_size, n_days), desc="Backtesting"):
            try:
                # Re-estimate the model every 20 days to save time
                if (t - window_size) % 20 == 0:
                    # Fit model on rolling window
                    train_y = y[t-window_size:t]
                    self.model.fit(train_y, verbose=False)
                
                # Predict next day's volatility
                sigma2_pred = self.model.predict_volatility(y[t-window_size:t], h_ahead=1)
                pred_integrated_vol = np.sqrt(trapz(sigma2_pred[0]))
                
                # Calculate realized volatility for the current day
                realized_vol[t] = np.sqrt(np.sum(y[t]**2))
                
                # Determine position size inversely proportional to volatility
                # With a safeguard against very small volatility
                pred_integrated_vol = max(pred_integrated_vol, 1e-4)
                target_position = 1.0 / (self.risk_aversion * pred_integrated_vol)
                
                # Limit position size
                target_position = np.clip(target_position, -3, 3)
                
                # Transaction costs
                tc = np.abs(target_position - positions[t-1]) * transaction_cost
                
                # Update position
                positions[t] = target_position
                
                # Calculate return (position * return - transaction costs)
                returns[t] = positions[t] * daily_returns[t] - tc
                
            except Exception as e:
                print(f"Error at day {t}: {e}")
                positions[t] = positions[t-1]
                returns[t] = 0
        
        # Calculate cumulative returns with safeguard against numerical issues
        returns = np.clip(returns[window_size:], -0.5, 0.5)  # Prevent extreme values
        cum_returns = np.cumprod(1 + returns) - 1
        
        # Calculate performance metrics with safeguards
        if np.std(returns) > 0:
            sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)
        else:
            sharpe_ratio = 0
            
        max_drawdown = 0
        try:
            peak = np.maximum.accumulate(cum_returns)
            drawdown = peak - cum_returns
            max_drawdown = np.max(drawdown)
        except:
            pass
        
        results = {
            'positions': positions,
            'returns': returns,
            'cum_returns': cum_returns,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'total_return': cum_returns[-1],
            'realized_vol': realized_vol
        }
        
        return results


def run_simulation_experiment():
    """
    Run a simulation experiment to test the Functional GARCH model and trading strategy
    """
    # Parameters
    n_days = 500  # Reduced for faster execution
    intraday_points = 100
    n_basis = 3
    
    # Create model
    model = FunctionalGARCH(n_basis=n_basis, intraday_points=intraday_points)
    
    # Simulate data
    print("Simulating data...")
    y, sigma2 = model.simulate(n_days, burn_in=200)
    
    # Plot some example return curves
    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    for i in range(5):
        plt.plot(model.u, y[i], alpha=0.7, label=f'Day {i+1}')
    plt.title('Simulated Return Curves')
    plt.xlabel('Intraday Time')
    plt.ylabel('Return')
    plt.legend()
    
    # Plot some example volatility curves
    plt.subplot(122)
    for i in range(5):
        plt.plot(model.u, sigma2[i], alpha=0.7, label=f'Day {i+1}')
    plt.title('Simulated Volatility Curves')
    plt.xlabel('Intraday Time')
    plt.ylabel('Volatility')
    plt.legend()
    plt.tight_layout()
    plt.savefig('simulated_curves.png')
    plt.close()
    
    # Split data into in-sample and out-of-sample
    train_size = int(0.7 * n_days)
    y_train = y[:train_size]
    y_test = y[train_size:]
    sigma2_train = sigma2[:train_size]
    sigma2_test = sigma2[train_size:]
    
    # Fit the model
    model_fit = FunctionalGARCH(n_basis=n_basis, intraday_points=intraday_points)
    model_fit.fit(y_train)
    
    # Compare true and estimated parameters
    print("\nParameter Estimation:")
    print(f"True delta: {model.delta0}")
    print(f"Est. delta: {model_fit.delta}")
    print("\nTrue alpha:")
    print(model.alpha0)
    print("Est. alpha:")
    print(model_fit.alpha)
    print("\nTrue beta:")
    print(model.beta0)
    print("Est. beta:")
    print(model_fit.beta)
    
    # Predict volatility
    pred_sigma2 = model_fit.predict_volatility(y_train, h_ahead=1)
    
    # Plot predicted vs true volatility for the next day
    plt.figure(figsize=(10, 6))
    plt.plot(model.u, sigma2_train[-1], 'b-', label='True Volatility')
    plt.plot(model.u, pred_sigma2[0], 'r--', label='Predicted Volatility')
    plt.title('True vs Predicted Volatility Curve')
    plt.xlabel('Intraday Time')
    plt.ylabel('Volatility')
    plt.legend()
    plt.savefig('volatility_prediction.png')
    plt.close()
    
    # Trading strategy backtest
    strategy = VolatilityTradingStrategy(model_fit)
    backtest_results = strategy.backtest(y_test, sigma2_test)
    
    # Plot backtest results
    plt.figure(figsize=(12, 8))
    
    plt.subplot(311)
    plt.plot(backtest_results['positions'])
    plt.title('Strategy Positions')
    plt.xlabel('Day')
    plt.ylabel('Position Size')
    plt.grid(True)
    
    plt.subplot(312)
    plt.plot(backtest_results['returns'], 'g-')
    plt.title('Daily Strategy Returns')
    plt.xlabel('Day')
    plt.ylabel('Return')
    plt.grid(True)
    
    plt.subplot(313)
    plt.plot(backtest_results['cum_returns'], 'b-')
    plt.title('Cumulative Strategy Returns')
    plt.xlabel('Day')
    plt.ylabel('Cumulative Return')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('strategy_backtest.png')
    plt.close()
    
    # Print performance metrics
    print("\nStrategy Performance:")
    print(f"Total Return: {backtest_results['total_return']:.2%}")
    print(f"Sharpe Ratio: {backtest_results['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {backtest_results['max_drawdown']:.2%}")
    
    # Compare to buy and hold
    buy_hold_return = np.clip(np.sum(y_test, axis=1), -0.5, 0.5)  # Prevent extreme values
    buy_hold_cum_return = np.cumprod(1 + buy_hold_return) - 1
    
    # Calculate buy and hold metrics with safeguards
    if np.std(buy_hold_return) > 0:
        buy_hold_sharpe = np.mean(buy_hold_return) / np.std(buy_hold_return) * np.sqrt(252)
    else:
        buy_hold_sharpe = 0
        
    buy_hold_max_dd = 0
    try:
        peak = np.maximum.accumulate(buy_hold_cum_return)
        drawdown = peak - buy_hold_cum_return
        buy_hold_max_dd = np.max(drawdown)
    except:
        pass
    
    print("\nBuy & Hold Performance:")
    print(f"Total Return: {buy_hold_cum_return[-1]:.2%}")
    print(f"Sharpe Ratio: {buy_hold_sharpe:.2f}")
    print(f"Max Drawdown: {buy_hold_max_dd:.2%}")
    
    # Optimized strategy with rolling estimation
    print("\nRunning optimized strategy with rolling estimation...")
    strategy_opt = VolatilityTradingStrategy(FunctionalGARCH(n_basis=n_basis, intraday_points=intraday_points))
    opt_results = strategy_opt.optimized_strategy(y, window_size=60)
    
    # Plot optimized strategy results
    plt.figure(figsize=(12, 8))
    
    plt.subplot(311)
    plt.plot(opt_results['positions'][60:])
    plt.title('Optimized Strategy Positions')
    plt.xlabel('Day')
    plt.ylabel('Position Size')
    plt.grid(True)
    
    plt.subplot(312)
    plt.plot(opt_results['returns'], 'g-')
    plt.title('Daily Optimized Strategy Returns')
    plt.xlabel('Day')
    plt.ylabel('Return')
    plt.grid(True)
    
    plt.subplot(313)
    plt.plot(opt_results['cum_returns'], 'b-')
    plt.title('Cumulative Optimized Strategy Returns')
    plt.xlabel('Day')
    plt.ylabel('Cumulative Return')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('optimized_strategy.png')
    plt.close()
    
    # Print optimized performance metrics
    print("\nOptimized Strategy Performance:")
    print(f"Total Return: {opt_results['total_return']:.2%}")
    print(f"Sharpe Ratio: {opt_results['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {opt_results['max_drawdown']:.2%}")
    
    # Plot realized vs predicted volatility
    plt.figure(figsize=(10, 6))
    plt.plot(opt_results['realized_vol'][60:], 'b-', alpha=0.5, label='Realized Volatility')
    plt.title('Realized Volatility')
    plt.xlabel('Day')
    plt.ylabel('Volatility')
    plt.grid(True)
    plt.savefig('realized_volatility.png')
    plt.close()
    
    return {
        'model': model_fit,
        'backtest_results': backtest_results,
        'opt_results': opt_results
    }

if __name__ == "__main__":
    results = run_simulation_experiment()