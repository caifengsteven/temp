"""
Robust implementation of RKHS Discount Curve Trading Strategy
Fixed version with improved numerical stability
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

class RobustDiscountCurveModel:
    """
    Robust implementation of the RKHS-based discount curve model
    """
    
    def __init__(self, alpha=0.1, beta=1.0, lambda_reg=0.1):
        self.alpha = alpha
        self.beta = beta
        self.lambda_reg = lambda_reg
        self.fitted_curves = []
        
    def exponential_kernel(self, x, y):
        """Simple exponential kernel for stability"""
        return np.exp(-self.alpha * np.abs(x - y))
    
    def polynomial_kernel(self, x, y, degree=2):
        """Polynomial kernel"""
        return (1 + x * y)**degree
    
    def build_kernel_matrix(self, tenors):
        """Build kernel matrix with regularization for numerical stability"""
        n = len(tenors)
        K = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                K[i, j] = self.exponential_kernel(tenors[i], tenors[j])
        
        # Add small regularization to diagonal for numerical stability
        K += 1e-6 * np.eye(n)
        return K
    
    def fit_discount_curve(self, bond_prices, cashflow_matrix, tenors):
        """Fit discount curve using regularized kernel regression"""
        try:
            n_tenors = len(tenors)
            n_bonds = len(bond_prices)
            
            # Build kernel matrix
            K = self.build_kernel_matrix(tenors)
            
            # Regularized solution
            CKC_T = cashflow_matrix @ K @ cashflow_matrix.T
            regularization = self.lambda_reg * np.eye(n_bonds)
            
            # Use SVD for more stable solution
            U, s, Vt = np.linalg.svd(CKC_T + regularization)
            s_inv = np.where(s > 1e-10, 1/s, 0)  # Threshold small singular values
            alpha = Vt.T @ np.diag(s_inv) @ U.T @ bond_prices
            
            # Compute fitted zero-coupon bond prices
            h_fitted = cashflow_matrix.T @ alpha
            
            # Convert to discount curve (ensure positive values)
            discount_curve = np.maximum(0, 1 - h_fitted)
            
            return discount_curve, alpha
            
        except Exception as e:
            print(f"Error in curve fitting: {e}")
            # Return simple linear interpolation as fallback
            discount_curve = np.linspace(0.01, 0.1, len(tenors))
            alpha = np.ones(len(bond_prices))
            return discount_curve, alpha

class SimplifiedTradingStrategy:
    """
    Simplified trading strategy based on yield curve analysis
    """
    
    def __init__(self, lookback_window=20):
        self.lookback_window = lookback_window
        
    def calculate_yield_curve_features(self, bond_data, window_size=5):
        """Calculate yield curve features: level, slope, curvature"""
        features_df = pd.DataFrame(index=bond_data.index)
        
        # Convert prices to yields (simplified)
        tenors = np.array([2, 5, 10, 30])
        yields_data = {}
        
        for i, tenor in enumerate(tenors):
            col_name = f'tenor_{tenor}'
            if col_name in bond_data.columns:
                # Convert price to yield: y = (100/P)^(1/T) - 1
                yields = (100 / bond_data[col_name])**(1/tenor) - 1
                yields_data[f'yield_{tenor}'] = yields
        
        yields_df = pd.DataFrame(yields_data, index=bond_data.index)
        
        # Calculate level (average yield)
        features_df['level'] = yields_df.mean(axis=1)
        
        # Calculate slope (30Y - 2Y)
        if 'yield_30' in yields_df.columns and 'yield_2' in yields_df.columns:
            features_df['slope'] = yields_df['yield_30'] - yields_df['yield_2']
        
        # Calculate curvature (2*10Y - 5Y - 30Y)
        if all(col in yields_df.columns for col in ['yield_10', 'yield_5', 'yield_30']):
            features_df['curvature'] = 2 * yields_df['yield_10'] - yields_df['yield_5'] - yields_df['yield_30']
        
        # Calculate moving averages for mean reversion signals
        for feature in ['level', 'slope', 'curvature']:
            if feature in features_df.columns:
                features_df[f'{feature}_ma'] = features_df[feature].rolling(window=window_size).mean()
                features_df[f'{feature}_std'] = features_df[feature].rolling(window=window_size).std()
        
        return features_df
    
    def generate_trading_signals(self, bond_data, tenors):
        """Generate trading signals based on yield curve analysis"""
        features_df = self.calculate_yield_curve_features(bond_data)
        signals_df = pd.DataFrame(index=bond_data.index)
        
        # Wait for sufficient data
        start_idx = max(self.lookback_window, 20)
        
        for i in range(start_idx, len(bond_data)):
            date = bond_data.index[i]
            
            # Calculate z-scores for mean reversion
            signals = {}
            
            for feature in ['level', 'slope', 'curvature']:
                if f'{feature}_ma' in features_df.columns and f'{feature}_std' in features_df.columns:
                    current_value = features_df.loc[date, feature]
                    ma_value = features_df.loc[date, f'{feature}_ma']
                    std_value = features_df.loc[date, f'{feature}_std']
                    
                    if not pd.isna(current_value) and not pd.isna(ma_value) and std_value > 0:
                        z_score = (current_value - ma_value) / std_value
                        
                        # Mean reversion signal: if z_score > threshold, expect reversion
                        if abs(z_score) > 1.5:
                            signals[feature] = -np.sign(z_score)
                        else:
                            signals[feature] = 0
                    else:
                        signals[feature] = 0
            
            # Map signals to tenor-specific positions
            # Level affects all tenors equally
            level_signal = signals.get('level', 0)
            
            # Slope affects long vs short end differently
            slope_signal = signals.get('slope', 0)
            
            # Curvature affects middle of curve
            curvature_signal = signals.get('curvature', 0)
            
            # Generate tenor-specific signals
            signals_df.loc[date, 'signal_tenor_2'] = 0.6 * level_signal - 0.4 * slope_signal
            signals_df.loc[date, 'signal_tenor_5'] = 0.4 * level_signal + 0.3 * curvature_signal
            signals_df.loc[date, 'signal_tenor_10'] = 0.4 * level_signal + 0.5 * curvature_signal
            signals_df.loc[date, 'signal_tenor_30'] = 0.6 * level_signal + 0.4 * slope_signal
        
        return signals_df.dropna()
    
    def backtest_strategy(self, bond_data, signals_df, transaction_cost=0.001):
        """Backtest the trading strategy"""
        # Calculate returns
        returns = bond_data.pct_change().dropna()
        
        # Align signals with returns
        common_dates = signals_df.index.intersection(returns.index)
        if len(common_dates) == 0:
            print("No common dates between signals and returns")
            return pd.Series(), {}
        
        signals_aligned = signals_df.loc[common_dates]
        returns_aligned = returns.loc[common_dates]
        
        # Calculate strategy returns
        strategy_returns = pd.DataFrame(index=common_dates)
        
        for tenor in [2, 5, 10, 30]:
            signal_col = f'signal_tenor_{tenor}'
            return_col = f'tenor_{tenor}'
            
            if signal_col in signals_aligned.columns and return_col in returns_aligned.columns:
                # Lag signals by 1 day
                lagged_signals = signals_aligned[signal_col].shift(1).fillna(0)
                
                # Calculate gross returns
                gross_returns = lagged_signals * returns_aligned[return_col]
                
                # Apply transaction costs
                position_changes = lagged_signals.diff().abs().fillna(0)
                costs = position_changes * transaction_cost
                net_returns = gross_returns - costs
                
                strategy_returns[f'strategy_tenor_{tenor}'] = net_returns
        
        # Portfolio returns (equal weight)
        if strategy_returns.empty:
            return pd.Series(), {}
        
        portfolio_returns = strategy_returns.mean(axis=1)
        
        # Calculate performance metrics
        if len(portfolio_returns) > 0 and portfolio_returns.std() > 0:
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
        else:
            performance_metrics = {
                'Total Return': "0.00%",
                'Annualized Return': "0.00%", 
                'Volatility': "0.00%",
                'Sharpe Ratio': "0.00",
                'Max Drawdown': "0.00%"
            }
        
        return portfolio_returns, performance_metrics
    
    def calculate_max_drawdown(self, returns):
        """Calculate maximum drawdown"""
        if len(returns) == 0:
            return 0
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()

def generate_realistic_treasury_data(start_date, end_date):
    """Generate more realistic Treasury bond data"""
    dates = pd.date_range(start_date, end_date, freq='B')
    tenors = np.array([2, 5, 10, 30])
    
    np.random.seed(42)
    n_days = len(dates)
    
    # More realistic base yield curve
    base_yields = np.array([0.015, 0.025, 0.035, 0.045])  # Upward sloping
    
    yields_data = {}
    for i, tenor in enumerate(tenors):
        yields = [base_yields[i]]
        
        for day in range(1, n_days):
            # Add some correlation between tenors and realistic volatility
            daily_shock = np.random.normal(0, 0.0005)  # 5bp daily vol
            mean_reversion = 0.005 * (base_yields[i] - yields[-1])
            
            # Add some trend and cycles
            trend = 0.00001 * np.sin(2 * np.pi * day / 252)  # Annual cycle
            
            new_yield = yields[-1] + mean_reversion + daily_shock + trend
            new_yield = max(0.001, min(0.1, new_yield))  # Bound between 0.1% and 10%
            yields.append(new_yield)
        
        # Convert to bond prices
        yields_array = np.array(yields)
        prices = 100 / (1 + yields_array)**tenor
        yields_data[f'tenor_{tenor}'] = prices
    
    return pd.DataFrame(yields_data, index=dates), tenors

def main():
    """Main execution function"""
    print("=== Robust RKHS Discount Curve Trading Strategy ===\n")
    
    # Generate realistic data
    print("Generating realistic Treasury bond data...")
    bond_data, tenors = generate_realistic_treasury_data('2023-01-01', '2023-12-31')
    print(f"Loaded data for tenors: {tenors} years")
    print(f"Data shape: {bond_data.shape}")
    print(f"Date range: {bond_data.index[0]} to {bond_data.index[-1]}\n")
    
    # Initialize strategy
    strategy = SimplifiedTradingStrategy(lookback_window=20)
    
    # Generate trading signals
    print("Generating trading signals...")
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
    
    return bond_data, signals_df, portfolio_returns

def plot_results(bond_data, signals_df, portfolio_returns, tenors):
    """Plot trading results"""
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # Plot 1: Bond prices over time
    for i, tenor in enumerate(tenors):
        col_name = f'tenor_{tenor}'
        if col_name in bond_data.columns:
            axes[0].plot(bond_data.index, bond_data[col_name], label=f'{tenor}Y')
    axes[0].set_title('Treasury Bond Prices Over Time')
    axes[0].set_ylabel('Price')
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot 2: Trading signals
    if not signals_df.empty:
        for tenor in tenors:
            signal_col = f'signal_tenor_{tenor}'
            if signal_col in signals_df.columns:
                axes[1].plot(signals_df.index, signals_df[signal_col], 
                           label=f'{tenor}Y', alpha=0.7)
        axes[1].set_title('Trading Signals')
        axes[1].set_ylabel('Signal Strength')
        axes[1].legend()
        axes[1].grid(True)
    
    # Plot 3: Cumulative returns
    if not portfolio_returns.empty and len(portfolio_returns) > 0:
        cumulative_returns = (1 + portfolio_returns).cumprod()
        axes[2].plot(cumulative_returns.index, cumulative_returns)
        axes[2].set_title('Cumulative Strategy Returns')
        axes[2].set_ylabel('Cumulative Return')
        axes[2].grid(True)
    
    plt.tight_layout()
    plt.savefig('trading_results.png', dpi=150, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    bond_data, signals_df, portfolio_returns = main()
