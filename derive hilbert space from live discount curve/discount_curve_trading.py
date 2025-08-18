"""
Reproducing Kernel Hilbert Space Methods for Discount Curve Trading
Implementation based on the paper: "REPRODUCING KERNEL HILBERT SPACE METHODS FOR MODELLING THE DISCOUNT CURVE"

This implementation demonstrates:
1. Fetching US Treasury bond data using xbbg
2. Implementing fully consistent kernels for discount curve modeling
3. Two-step calibration procedure
4. Trading signal generation based on discount curve analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.linalg import expm
import warnings
warnings.filterwarnings('ignore')

try:
    import xbbg
    BLOOMBERG_AVAILABLE = True
    print("Bloomberg xbbg library available")
except ImportError:
    BLOOMBERG_AVAILABLE = False
    print("Bloomberg xbbg not available - using simulated data")

class DiscountCurveModel:
    """
    Implementation of the RKHS-based discount curve model from the paper
    """
    
    def __init__(self, alpha=1.0, beta=1.0, lambda_reg=0.01):
        """
        Initialize the discount curve model
        
        Parameters:
        alpha, beta: Kernel parameters for fully consistent kernels
        lambda_reg: Regularization parameter for ridge regression
        """
        self.alpha = alpha
        self.beta = beta
        self.lambda_reg = lambda_reg
        self.fitted_curves = []
        self.bond_data = None
        
    def fully_consistent_kernel(self, x, y, polynomial_degree=2):
        """
        Implement fully consistent kernel as defined in Proposition 3.8
        
        k(x,y) = p((√β*x - α/√β)(√β*y - α/√β)) * exp(β*x*y - α*(x+y))
        
        where p is a polynomial with non-negative coefficients
        """
        sqrt_beta = np.sqrt(self.beta)
        
        # Simple polynomial p(t) = 1 + t + t^2/2 (ensuring non-negative coefficients)
        def polynomial(t):
            return 1 + t + t**2/2
        
        # Transform coordinates
        x_transformed = sqrt_beta * x - self.alpha / sqrt_beta
        y_transformed = sqrt_beta * y - self.alpha / sqrt_beta
        
        # Compute kernel
        poly_term = polynomial(x_transformed * y_transformed)
        exp_term = np.exp(self.beta * x * y - self.alpha * (x + y))
        
        return poly_term * exp_term
    
    def build_kernel_matrix(self, tenors):
        """Build the kernel matrix K for given tenors"""
        n = len(tenors)
        K = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                K[i, j] = self.fully_consistent_kernel(tenors[i], tenors[j])
        
        return K
    
    def fit_discount_curve(self, bond_prices, cashflow_matrix, tenors):
        """
        First step: Fit discount curve using kernel regression (Representer Theorem)
        
        Minimize: ||C*h - P||^2 + λ*||h||_H^2
        where h is the zero-coupon bond price curve, C is cashflow matrix, P is bond prices
        """
        # Convert to discount curve: H = 1 - h (where h is zero-coupon bond prices)
        n_tenors = len(tenors)
        n_bonds = len(bond_prices)
        
        # Build kernel matrix
        K = self.build_kernel_matrix(tenors)
        
        # Solve the optimization problem using Representer Theorem
        # Solution has form: h(x) = sum_i alpha_i * k(x, x_i)
        # This reduces to solving: (C*K*C^T + λ*I)*α = P
        
        CKC_T = cashflow_matrix @ K @ cashflow_matrix.T
        regularization = self.lambda_reg * np.eye(n_bonds)
        
        try:
            # Solve for coefficients
            alpha = np.linalg.solve(CKC_T + regularization, bond_prices)
            
            # Compute fitted zero-coupon bond prices
            h_fitted = cashflow_matrix.T @ alpha
            
            # Convert to discount curve
            discount_curve = 1 - h_fitted
            
            return discount_curve, alpha
            
        except np.linalg.LinAlgError:
            print("Warning: Singular matrix encountered, using pseudo-inverse")
            alpha = np.linalg.pinv(CKC_T + regularization) @ bond_prices
            h_fitted = cashflow_matrix.T @ alpha
            discount_curve = 1 - h_fitted
            return discount_curve, alpha
    
    def extract_factors(self, discount_curves_history, n_factors=3):
        """
        Second step: Extract low-dimensional factors from fitted discount curves
        using PCA to identify the main driving factors
        """
        # Stack all discount curves
        curve_matrix = np.array(discount_curves_history)
        
        # Perform PCA
        mean_curve = np.mean(curve_matrix, axis=0)
        centered_curves = curve_matrix - mean_curve
        
        # Compute covariance matrix
        cov_matrix = np.cov(centered_curves.T)
        
        # Eigenvalue decomposition
        eigenvals, eigenvecs = np.linalg.eigh(cov_matrix)
        
        # Sort by eigenvalues (descending)
        idx = np.argsort(eigenvals)[::-1]
        eigenvals = eigenvals[idx]
        eigenvecs = eigenvecs[:, idx]
        
        # Extract top factors
        factors = eigenvecs[:, :n_factors]
        factor_loadings = centered_curves @ factors
        
        # Explained variance
        explained_variance = eigenvals[:n_factors] / np.sum(eigenvals)
        
        return factors, factor_loadings, explained_variance, mean_curve

def fetch_treasury_data(start_date='2023-01-01', end_date='2023-12-31'):
    """
    Fetch US Treasury bond data using xbbg (if available)
    """
    if not BLOOMBERG_AVAILABLE:
        print("Generating simulated Treasury data...")
        return generate_simulated_data(start_date, end_date)
    
    try:
        # Treasury tickers for different maturities
        tickers = [
            'GT2 Govt',   # 2-Year
            'GT5 Govt',   # 5-Year  
            'GT10 Govt',  # 10-Year
            'GT30 Govt'   # 30-Year
        ]
        
        # Fetch yield data
        data = xbbg.bdh(tickers, 'PX_LAST', start_date, end_date)
        
        # Clean and process data
        data = data.dropna()
        
        # Convert yields to prices (simplified)
        # P = 100 / (1 + y/100)^T (approximation for zero-coupon)
        tenors = np.array([2, 5, 10, 30])
        
        bond_data = {}
        for i, ticker in enumerate(tickers):
            yields = data[ticker] / 100  # Convert to decimal
            prices = 100 / (1 + yields)**tenors[i]
            bond_data[f'tenor_{tenors[i]}'] = prices
        
        bond_df = pd.DataFrame(bond_data, index=data.index)
        
        return bond_df, tenors
        
    except Exception as e:
        print(f"Error fetching Bloomberg data: {e}")
        print("Falling back to simulated data...")
        return generate_simulated_data(start_date, end_date)

def generate_simulated_data(start_date, end_date):
    """Generate simulated Treasury bond data for testing"""
    dates = pd.date_range(start_date, end_date, freq='B')  # Business days
    tenors = np.array([2, 5, 10, 30])
    
    # Simulate yield curves with realistic dynamics
    np.random.seed(42)
    n_days = len(dates)
    
    # Base yield curve (upward sloping)
    base_yields = np.array([0.02, 0.025, 0.03, 0.035])
    
    # Add time-varying component
    yields_data = {}
    for i, tenor in enumerate(tenors):
        # Random walk with mean reversion
        yields = [base_yields[i]]
        for _ in range(n_days - 1):
            shock = np.random.normal(0, 0.001)  # 10bp daily volatility
            mean_reversion = 0.01 * (base_yields[i] - yields[-1])
            new_yield = yields[-1] + mean_reversion + shock
            yields.append(max(0.001, new_yield))  # Floor at 0.1%
        
        # Convert to bond prices (zero-coupon approximation)
        prices = 100 / (1 + np.array(yields))**tenor
        yields_data[f'tenor_{tenor}'] = prices
    
    bond_df = pd.DataFrame(yields_data, index=dates)
    return bond_df, tenors

class DiscountCurveTradingStrategy:
    """
    Trading strategy based on discount curve analysis
    """

    def __init__(self, model, lookback_window=20):
        self.model = model
        self.lookback_window = lookback_window
        self.signals = []
        self.positions = []

    def generate_trading_signals(self, bond_data, tenors):
        """
        Generate trading signals based on discount curve analysis
        """
        dates = bond_data.index
        signals_df = pd.DataFrame(index=dates)

        # Initialize storage for fitted curves
        fitted_curves_history = []
        factor_loadings_history = []

        print("Generating trading signals...")

        for i, date in enumerate(dates):
            if i < self.lookback_window:
                continue

            # Get current bond prices
            current_prices = bond_data.iloc[i].values

            # Create simple cashflow matrix (assuming zero-coupon bonds)
            cashflow_matrix = np.eye(len(tenors))

            # Fit discount curve
            try:
                discount_curve, alpha = self.model.fit_discount_curve(
                    current_prices, cashflow_matrix, tenors
                )
                fitted_curves_history.append(discount_curve)

                # Extract factors if we have enough history
                if len(fitted_curves_history) >= self.lookback_window:
                    factors, factor_loadings, explained_var, mean_curve = self.model.extract_factors(
                        fitted_curves_history[-self.lookback_window:], n_factors=3
                    )

                    # Current factor loadings
                    current_loading = factor_loadings[-1]
                    factor_loadings_history.append(current_loading)

                    # Generate signals based on factor analysis
                    signals = self.analyze_factors_for_signals(
                        factor_loadings_history, explained_var
                    )

                    # Store signals
                    for j, tenor in enumerate(tenors):
                        signals_df.loc[date, f'signal_tenor_{tenor}'] = signals[j] if j < len(signals) else 0

            except Exception as e:
                print(f"Error processing date {date}: {e}")
                continue

        return signals_df.dropna()

    def analyze_factors_for_signals(self, factor_loadings_history, explained_variance):
        """
        Analyze factor loadings to generate trading signals
        """
        if len(factor_loadings_history) < 5:
            return [0, 0, 0, 0]  # No signal

        # Convert to numpy array
        loadings_array = np.array(factor_loadings_history)

        # Calculate z-scores for each factor
        recent_loadings = loadings_array[-5:]  # Last 5 observations
        mean_loadings = np.mean(loadings_array[:-1], axis=0)
        std_loadings = np.std(loadings_array[:-1], axis=0)

        current_loadings = loadings_array[-1]
        z_scores = (current_loadings - mean_loadings) / (std_loadings + 1e-8)

        # Generate signals based on factor analysis
        signals = []

        # Factor 1 (Level): If unusually high/low, expect mean reversion
        level_signal = -np.sign(z_scores[0]) if abs(z_scores[0]) > 1.5 else 0

        # Factor 2 (Slope): If slope is steep, expect flattening
        slope_signal = -np.sign(z_scores[1]) if abs(z_scores[1]) > 1.5 else 0

        # Factor 3 (Curvature): If high curvature, expect normalization
        curvature_signal = -np.sign(z_scores[2]) if abs(z_scores[2]) > 1.5 else 0

        # Combine signals for different tenors
        # Short-term (2Y): Sensitive to level and slope
        signals.append(0.6 * level_signal + 0.4 * slope_signal)

        # Medium-term (5Y): Balanced exposure
        signals.append(0.4 * level_signal + 0.4 * slope_signal + 0.2 * curvature_signal)

        # Long-term (10Y): Sensitive to slope and curvature
        signals.append(0.2 * level_signal + 0.5 * slope_signal + 0.3 * curvature_signal)

        # Very long-term (30Y): Mainly curvature
        signals.append(0.1 * level_signal + 0.3 * slope_signal + 0.6 * curvature_signal)

        return signals

    def backtest_strategy(self, bond_data, signals_df, transaction_cost=0.001):
        """
        Backtest the trading strategy
        """
        # Calculate returns
        returns = bond_data.pct_change().dropna()

        # Align signals with returns
        common_dates = signals_df.index.intersection(returns.index)
        signals_aligned = signals_df.loc[common_dates]
        returns_aligned = returns.loc[common_dates]

        # Calculate strategy returns
        strategy_returns = pd.DataFrame(index=common_dates)

        for i, tenor in enumerate([2, 5, 10, 30]):
            signal_col = f'signal_tenor_{tenor}'
            return_col = f'tenor_{tenor}'

            if signal_col in signals_aligned.columns and return_col in returns_aligned.columns:
                # Lag signals by 1 day (can't trade on same day signal is generated)
                lagged_signals = signals_aligned[signal_col].shift(1)

                # Calculate gross returns
                gross_returns = lagged_signals * returns_aligned[return_col]

                # Apply transaction costs
                position_changes = lagged_signals.diff().abs()
                costs = position_changes * transaction_cost
                net_returns = gross_returns - costs

                strategy_returns[f'strategy_tenor_{tenor}'] = net_returns

        # Portfolio returns (equal weight)
        portfolio_returns = strategy_returns.mean(axis=1)

        # Calculate performance metrics
        total_return = (1 + portfolio_returns).prod() - 1
        annualized_return = (1 + total_return)**(252/len(portfolio_returns)) - 1
        volatility = portfolio_returns.std() * np.sqrt(252)
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
        max_drawdown = self.calculate_max_drawdown(portfolio_returns)

        performance_metrics = {
            'Total Return': f"{total_return:.2%}",
            'Annualized Return': f"{annualized_return:.2%}",
            'Volatility': f"{volatility:.2%}",
            'Sharpe Ratio': f"{sharpe_ratio:.2f}",
            'Max Drawdown': f"{max_drawdown:.2%}"
        }

        return portfolio_returns, performance_metrics

    def calculate_max_drawdown(self, returns):
        """Calculate maximum drawdown"""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()

def main():
    """
    Main execution function
    """
    print("=== RKHS Discount Curve Trading Strategy ===\n")

    # Initialize model
    model = DiscountCurveModel(alpha=0.5, beta=1.0, lambda_reg=0.01)

    # Fetch data
    print("Fetching Treasury bond data...")
    bond_data, tenors = fetch_treasury_data('2023-01-01', '2023-12-31')
    print(f"Loaded data for tenors: {tenors} years")
    print(f"Data shape: {bond_data.shape}")
    print(f"Date range: {bond_data.index[0]} to {bond_data.index[-1]}\n")

    # Initialize trading strategy
    strategy = DiscountCurveTradingStrategy(model, lookback_window=20)

    # Generate trading signals
    signals_df = strategy.generate_trading_signals(bond_data, tenors)
    print(f"Generated signals for {len(signals_df)} trading days\n")

    # Backtest strategy
    print("Backtesting strategy...")
    portfolio_returns, performance_metrics = strategy.backtest_strategy(bond_data, signals_df)

    # Display results
    print("\n=== PERFORMANCE METRICS ===")
    for metric, value in performance_metrics.items():
        print(f"{metric}: {value}")

    # Plot results
    plot_results(bond_data, signals_df, portfolio_returns, tenors)

    return model, strategy, bond_data, signals_df, portfolio_returns

def plot_results(bond_data, signals_df, portfolio_returns, tenors):
    """
    Plot trading results
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    # Plot 1: Bond prices over time
    axes[0].plot(bond_data.index, bond_data)
    axes[0].set_title('Treasury Bond Prices Over Time')
    axes[0].set_ylabel('Price')
    axes[0].legend([f'{t}Y' for t in tenors])
    axes[0].grid(True)

    # Plot 2: Trading signals
    if not signals_df.empty:
        for i, tenor in enumerate(tenors):
            signal_col = f'signal_tenor_{tenor}'
            if signal_col in signals_df.columns:
                axes[1].plot(signals_df.index, signals_df[signal_col],
                           label=f'{tenor}Y', alpha=0.7)
        axes[1].set_title('Trading Signals')
        axes[1].set_ylabel('Signal Strength')
        axes[1].legend()
        axes[1].grid(True)

    # Plot 3: Cumulative returns
    if not portfolio_returns.empty:
        cumulative_returns = (1 + portfolio_returns).cumprod()
        axes[2].plot(cumulative_returns.index, cumulative_returns)
        axes[2].set_title('Cumulative Strategy Returns')
        axes[2].set_ylabel('Cumulative Return')
        axes[2].grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    model, strategy, bond_data, signals_df, portfolio_returns = main()
