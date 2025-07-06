import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.optimize as optimize
from scipy.special import lambertw
from datetime import datetime, timedelta
import yfinance as yf
from tqdm import tqdm

class DynamicAMM:
    """
    Dynamic AMM simulation with different weight change strategies
    """
    def __init__(self, initial_reserves, initial_weights, market_prices):
        """
        Initialize the AMM pool
        
        Parameters:
        -----------
        initial_reserves: array-like
            Initial token reserves in the pool
        initial_weights: array-like
            Initial weights for each token (must sum to 1)
        market_prices: array-like
            Market prices for each token
        """
        self.reserves = np.array(initial_reserves, dtype=float)
        self.weights = np.array(initial_weights, dtype=float)
        self.prices = np.array(market_prices, dtype=float)
        self.N = len(initial_reserves)  # Number of tokens
        
        # Validate inputs
        assert len(self.weights) == self.N, "Weights must have same length as reserves"
        assert len(self.prices) == self.N, "Prices must have same length as reserves"
        assert np.isclose(np.sum(self.weights), 1.0), "Weights must sum to 1"
        assert np.all(self.weights > 0) and np.all(self.weights < 1), "Weights must be between 0 and 1"
        
        # Calculate the invariant k
        self.k = self._calculate_k(self.reserves, self.weights)
        
        # Store history
        self.reserve_history = [self.reserves.copy()]
        self.weight_history = [self.weights.copy()]
        self.price_history = [self.prices.copy()]
        self.value_history = [self._calculate_pool_value()]
        
    def _calculate_k(self, reserves, weights):
        """Calculate the invariant k for G3M"""
        return np.prod(reserves ** weights)
    
    def _calculate_pool_value(self):
        """Calculate the total value of the pool in terms of market prices"""
        return np.sum(self.reserves * self.prices)
    
    def one_step_rebalance(self, target_weights):
        """
        Rebalance the pool to target weights in one step
        
        Parameters:
        -----------
        target_weights: array-like
            Target weights for each token (must sum to 1)
        """
        target_weights = np.array(target_weights, dtype=float)
        
        # Validate target weights
        assert len(target_weights) == self.N, "Target weights must have same length as reserves"
        assert np.isclose(np.sum(target_weights), 1.0), "Target weights must sum to 1"
        assert np.all(target_weights > 0) and np.all(target_weights < 1), "Weights must be between 0 and 1"
        
        # Apply weight change and calculate new reserves
        new_reserves = self._rebalance(self.reserves, self.weights, target_weights, self.prices)
        
        # Update state
        self.reserves = new_reserves
        self.weights = target_weights
        self.k = self._calculate_k(self.reserves, self.weights)
        
        # Store history
        self.reserve_history.append(self.reserves.copy())
        self.weight_history.append(self.weights.copy())
        self.price_history.append(self.prices.copy())
        self.value_history.append(self._calculate_pool_value())
        
        return self.reserves
    
    def linear_interpolation_rebalance(self, target_weights, steps):
        """
        Rebalance the pool to target weights using linear interpolation
        
        Parameters:
        -----------
        target_weights: array-like
            Target weights for each token (must sum to 1)
        steps: int
            Number of steps to use for interpolation
        """
        target_weights = np.array(target_weights, dtype=float)
        
        # Validate target weights
        assert len(target_weights) == self.N, "Target weights must have same length as reserves"
        assert np.isclose(np.sum(target_weights), 1.0), "Target weights must sum to 1"
        assert np.all(target_weights > 0) and np.all(target_weights < 1), "Weights must be between 0 and 1"
        
        initial_weights = self.weights.copy()
        
        for step in range(1, steps + 1):
            # Linear interpolation: w(t) = (1 - t/steps) * w0 + (t/steps) * wf
            t = step / steps
            intermediate_weights = (1 - t) * initial_weights + t * target_weights
            
            # Normalize to ensure weights sum to 1 (to handle numerical issues)
            intermediate_weights = intermediate_weights / np.sum(intermediate_weights)
            
            # Apply weight change and calculate new reserves
            new_reserves = self._rebalance(self.reserves, self.weights, intermediate_weights, self.prices)
            
            # Update state
            self.reserves = new_reserves
            self.weights = intermediate_weights
            self.k = self._calculate_k(self.reserves, self.weights)
            
            # Store history
            self.reserve_history.append(self.reserves.copy())
            self.weight_history.append(self.weights.copy())
            self.price_history.append(self.prices.copy())
            self.value_history.append(self._calculate_pool_value())
        
        return self.reserves
    
    def approximate_optimal_rebalance(self, target_weights, steps):
        """
        Rebalance the pool to target weights using the approximately optimal method
        
        Parameters:
        -----------
        target_weights: array-like
            Target weights for each token (must sum to 1)
        steps: int
            Number of steps to use for interpolation
        """
        target_weights = np.array(target_weights, dtype=float)
        
        # Validate target weights
        assert len(target_weights) == self.N, "Target weights must have same length as reserves"
        assert np.isclose(np.sum(target_weights), 1.0), "Target weights must sum to 1"
        assert np.all(target_weights > 0) and np.all(target_weights < 1), "Weights must be between 0 and 1"
        
        initial_weights = self.weights.copy()
        
        for step in range(1, steps + 1):
            # Calculate intermediate weights using the approximately optimal method
            k = step / steps
            
            # Arithmetic mean interpolation (Eq. 8)
            w_AM = (1 - k) * initial_weights + k * target_weights
            
            # Geometric mean interpolation (Eq. 9)
            w_GM = initial_weights ** (1 - k) * target_weights ** k
            
            # Approximately optimal interpolation (Eq. 10)
            intermediate_weights = (w_AM + w_GM) / np.sum(w_AM + w_GM)
            
            # Apply weight change and calculate new reserves
            new_reserves = self._rebalance(self.reserves, self.weights, intermediate_weights, self.prices)
            
            # Update state
            self.reserves = new_reserves
            self.weights = intermediate_weights
            self.k = self._calculate_k(self.reserves, self.weights)
            
            # Store history
            self.reserve_history.append(self.reserves.copy())
            self.weight_history.append(self.weights.copy())
            self.price_history.append(self.prices.copy())
            self.value_history.append(self._calculate_pool_value())
        
        return self.reserves
    
    def optimal_rebalance(self, target_weights, steps):
        """
        Rebalance the pool to target weights using the optimal method with Lambert W function
        
        Parameters:
        -----------
        target_weights: array-like
            Target weights for each token (must sum to 1)
        steps: int
            Number of steps to use for interpolation
        """
        target_weights = np.array(target_weights, dtype=float)
        
        # Validate target weights
        assert len(target_weights) == self.N, "Target weights must have same length as reserves"
        assert np.isclose(np.sum(target_weights), 1.0), "Target weights must sum to 1"
        assert np.all(target_weights > 0) and np.all(target_weights < 1), "Weights must be between 0 and 1"
        
        initial_weights = self.weights.copy()
        
        # For small steps, use direct optimization (costly but accurate)
        if steps <= 3:
            for step in range(1, steps + 1):
                t = step / steps
                
                # Solve for optimal weights numerically
                intermediate_weights = self._find_optimal_weights(initial_weights, target_weights, t)
                
                # Apply weight change and calculate new reserves
                new_reserves = self._rebalance(self.reserves, self.weights, intermediate_weights, self.prices)
                
                # Update state
                self.reserves = new_reserves
                self.weights = intermediate_weights
                self.k = self._calculate_k(self.reserves, self.weights)
                
                # Store history
                self.reserve_history.append(self.reserves.copy())
                self.weight_history.append(self.weights.copy())
                self.price_history.append(self.prices.copy())
                self.value_history.append(self._calculate_pool_value())
        else:
            # For many steps, use Lambert W function for better efficiency
            for step in range(1, steps + 1):
                k = step / steps
                
                # Lambert W approach for each weight
                # Using Eq. 6: w_i = w_f / W0(e * w_f / w_0)
                
                # But we need to normalize afterwards to ensure weights sum to 1
                unnormalized_weights = np.zeros(self.N)
                
                for i in range(self.N):
                    # Interpolate using Lambert W function
                    if initial_weights[i] == target_weights[i]:
                        unnormalized_weights[i] = initial_weights[i]
                    else:
                        # Direct Lambert W approach might cause numerical issues for extreme weight changes
                        # Instead, we use a hybrid approach that's more stable
                        ratio = target_weights[i] / initial_weights[i]
                        if ratio > 0.01 and ratio < 100:  # Within reasonable range
                            try:
                                lambert_arg = np.exp(1) * target_weights[i] / initial_weights[i]
                                unnormalized_weights[i] = target_weights[i] / np.real(lambertw(lambert_arg))
                            except:
                                # Fallback to geometric/arithmetic hybrid if Lambert W fails
                                w_AM = (1 - k) * initial_weights[i] + k * target_weights[i]
                                w_GM = initial_weights[i] ** (1 - k) * target_weights[i] ** k
                                unnormalized_weights[i] = (w_AM + w_GM) / 2
                        else:
                            # For extreme ratios, use geometric/arithmetic hybrid
                            w_AM = (1 - k) * initial_weights[i] + k * target_weights[i]
                            w_GM = initial_weights[i] ** (1 - k) * target_weights[i] ** k
                            unnormalized_weights[i] = (w_AM + w_GM) / 2
                
                # Normalize weights to ensure they sum to 1
                intermediate_weights = unnormalized_weights / np.sum(unnormalized_weights)
                
                # Apply weight change and calculate new reserves
                new_reserves = self._rebalance(self.reserves, self.weights, intermediate_weights, self.prices)
                
                # Update state
                self.reserves = new_reserves
                self.weights = intermediate_weights
                self.k = self._calculate_k(self.reserves, self.weights)
                
                # Store history
                self.reserve_history.append(self.reserves.copy())
                self.weight_history.append(self.weights.copy())
                self.price_history.append(self.prices.copy())
                self.value_history.append(self._calculate_pool_value())
        
        return self.reserves
    
    def _find_optimal_weights(self, initial_weights, target_weights, t):
        """
        Find the optimal intermediate weights by direct numerical optimization
        This is a costly operation but gives the most accurate results
        """
        def objective(w):
            # Normalize weights to ensure they sum to 1
            w_normalized = w / np.sum(w)
            
            # Calculate reserves after rebalancing
            old_reserves = self.reserves.copy()
            old_weights = self.weights.copy()
            
            # Apply weight change and calculate new reserves
            new_reserves = self._rebalance(old_reserves, old_weights, w_normalized, self.prices)
            
            # We want to maximize the value of the pool
            return -np.sum(new_reserves * self.prices)
        
        # Initial guess: linear interpolation
        initial_guess = (1 - t) * initial_weights + t * target_weights
        
        # Constraints: weights must be positive and sum to 1
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},  # Sum to 1
        ]
        
        bounds = [(0.001, 0.999) for _ in range(self.N)]  # Weights between 0 and 1
        
        # Run optimization
        result = optimize.minimize(
            objective, 
            initial_guess, 
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        # Return normalized weights
        return result.x / np.sum(result.x)
    
    def _rebalance(self, reserves, old_weights, new_weights, prices):
        """
        Rebalance the pool to new weights given constant market prices
        
        This implements Eq. 2 from the paper:
        R(t') = R(t0) * [w(t')/w(t0)] * Prod_j [(w_j(t0)/w_j(t'))^(w_j(t'))]
        """
        # Calculate the product term
        product_term = 1
        for j in range(self.N):
            if old_weights[j] != new_weights[j]:  # Avoid division by zero or 0^0
                product_term *= (old_weights[j] / new_weights[j]) ** new_weights[j]
        
        # Calculate new reserves (Eq. 2)
        new_reserves = reserves * (new_weights / old_weights) * product_term
        
        return new_reserves
    
    def update_market_prices(self, new_prices):
        """
        Update market prices (typically this would happen between rebalancing steps)
        
        Parameters:
        -----------
        new_prices: array-like
            New market prices for each token
        """
        self.prices = np.array(new_prices, dtype=float)
        
        # Store history
        self.price_history.append(self.prices.copy())
        self.value_history.append(self._calculate_pool_value())
    
    def get_pool_value(self):
        """Get the current value of the pool"""
        return self._calculate_pool_value()
    
    def get_value_history(self):
        """Get the history of pool values"""
        return self.value_history
    
    def plot_weights(self):
        """Plot the weight history for each token"""
        weights_history = np.array(self.weight_history)
        
        plt.figure(figsize=(10, 6))
        for i in range(self.N):
            plt.plot(weights_history[:, i], label=f'Token {i+1}')
        
        plt.xlabel('Step')
        plt.ylabel('Weight')
        plt.title('Weight Evolution')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    def plot_value(self):
        """Plot the pool value history"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.value_history)
        plt.xlabel('Step')
        plt.ylabel('Pool Value')
        plt.title('Pool Value Evolution')
        plt.grid(True)
        plt.show()

def compare_rebalancing_strategies(initial_reserves, initial_weights, target_weights, market_prices, steps=10):
    """
    Compare different rebalancing strategies
    
    Parameters:
    -----------
    initial_reserves: array-like
        Initial token reserves in the pool
    initial_weights: array-like
        Initial weights for each token
    target_weights: array-like
        Target weights for each token
    market_prices: array-like
        Market prices for each token
    steps: int
        Number of steps to use for interpolation
    
    Returns:
    --------
    dict: Dictionary with results for each strategy
    """
    strategies = {
        'One Step': lambda: DynamicAMM(initial_reserves, initial_weights, market_prices).one_step_rebalance(target_weights),
        'Linear': lambda: DynamicAMM(initial_reserves, initial_weights, market_prices).linear_interpolation_rebalance(target_weights, steps),
        'Approximately Optimal': lambda: DynamicAMM(initial_reserves, initial_weights, market_prices).approximate_optimal_rebalance(target_weights, steps),
        'Optimal': lambda: DynamicAMM(initial_reserves, initial_weights, market_prices).optimal_rebalance(target_weights, steps)
    }
    
    results = {}
    
    for name, strategy_fn in strategies.items():
        print(f"Running {name} strategy...")
        amm = DynamicAMM(initial_reserves, initial_weights, market_prices)
        
        if name == 'One Step':
            amm.one_step_rebalance(target_weights)
        elif name == 'Linear':
            amm.linear_interpolation_rebalance(target_weights, steps)
        elif name == 'Approximately Optimal':
            amm.approximate_optimal_rebalance(target_weights, steps)
        elif name == 'Optimal':
            amm.optimal_rebalance(target_weights, steps)
        
        results[name] = {
            'final_value': amm.get_pool_value(),
            'value_history': amm.get_value_history(),
            'final_reserves': amm.reserves,
            'amm': amm
        }
    
    # Calculate relative performance
    base_value = results['One Step']['final_value']
    for name, data in results.items():
        data['relative_performance'] = (data['final_value'] - base_value) / base_value * 100
    
    # Print results
    print("\nResults:")
    print(f"{'Strategy':<25} {'Final Value':<15} {'Relative to One Step':<20}")
    print("-" * 60)
    for name, data in results.items():
        print(f"{name:<25} {data['final_value']:<15.4f} {data['relative_performance']:+<20.2f}%")
    
    # Plot comparison
    plt.figure(figsize=(12, 6))
    for name, data in results.items():
        plt.plot(data['value_history'], label=name)
    
    plt.xlabel('Step')
    plt.ylabel('Pool Value')
    plt.title('Pool Value Evolution for Different Rebalancing Strategies')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    return results

def run_historical_backtest(initial_reserves, initial_weights, strategy_fn, symbols, start_date, end_date, rebalance_freq='1d'):
    """
    Run a historical backtest of a rebalancing strategy using real market data
    
    Parameters:
    -----------
    initial_reserves: array-like
        Initial token reserves in the pool
    initial_weights: array-like
        Initial weights for each token
    strategy_fn: function
        Function that takes (amm, target_weights, steps) and applies a rebalancing strategy
    symbols: list
        List of token symbols to fetch data for
    start_date, end_date: str
        Start and end dates for the backtest in YYYY-MM-DD format
    rebalance_freq: str
        Frequency of rebalancing (e.g., '1d' for daily)
        
    Returns:
    --------
    DataFrame with backtest results
    """
    # Fetch historical price data
    data = yf.download(symbols, start=start_date, end=end_date)['Adj Close']
    
    # Resample to the desired frequency
    data = data.resample(rebalance_freq).last()
    
    # Fill missing values (if any)
    data = data.ffill()
    
    # Initialize AMM
    initial_prices = data.iloc[0].values
    amm = DynamicAMM(initial_reserves, initial_weights, initial_prices)
    
    # Initialize results
    results = pd.DataFrame(index=data.index)
    results['pool_value'] = 0.0
    
    # Store initial state
    results.loc[data.index[0], 'pool_value'] = amm.get_pool_value()
    
    # Run simulation
    print(f"Running backtest from {start_date} to {end_date}...")
    for i in tqdm(range(1, len(data))):
        date = data.index[i]
        
        # Update market prices
        new_prices = data.iloc[i].values
        amm.update_market_prices(new_prices)
        
        # Generate target weights (in a real strategy this would be based on some model)
        # For this demo, we'll use a simple momentum-based strategy
        price_changes = data.iloc[i] / data.iloc[i-1] - 1
        
        # Increase weights for tokens with positive momentum, decrease for negative
        target_weights = amm.weights.copy()
        
        # Simple momentum strategy: adjust weights proportionally to recent returns
        adjustment = 0.05 * price_changes.values  # 5% adjustment based on price change
        target_weights += adjustment
        
        # Ensure weights are positive and sum to 1
        target_weights = np.maximum(target_weights, 0.01)  # Minimum weight of 1%
        target_weights = target_weights / np.sum(target_weights)
        
        # Apply rebalancing strategy
        strategy_fn(amm, target_weights, steps=5)
        
        # Store results
        results.loc[date, 'pool_value'] = amm.get_pool_value()
    
    # Calculate returns
    results['return'] = results['pool_value'].pct_change()
    results['cumulative_return'] = (1 + results['return']).cumprod() - 1
    
    return results

def compare_strategies_backtest(initial_reserves, initial_weights, symbols, start_date, end_date, rebalance_freq='1d'):
    """
    Compare different rebalancing strategies on historical data
    
    Parameters:
    -----------
    initial_reserves: array-like
        Initial token reserves in the pool
    initial_weights: array-like
        Initial weights for each token
    symbols: list
        List of token symbols to fetch data for
    start_date, end_date: str
        Start and end dates for the backtest in YYYY-MM-DD format
    rebalance_freq: str
        Frequency of rebalancing (e.g., '1d' for daily)
        
    Returns:
    --------
    Dictionary with results for each strategy
    """
    strategies = {
        'One Step': lambda amm, target_weights, steps: amm.one_step_rebalance(target_weights),
        'Linear': lambda amm, target_weights, steps: amm.linear_interpolation_rebalance(target_weights, steps),
        'Approximately Optimal': lambda amm, target_weights, steps: amm.approximate_optimal_rebalance(target_weights, steps),
    }
    
    results = {}
    
    for name, strategy_fn in strategies.items():
        print(f"\nRunning {name} strategy backtest...")
        results[name] = run_historical_backtest(
            initial_reserves,
            initial_weights,
            strategy_fn,
            symbols,
            start_date,
            end_date,
            rebalance_freq
        )
    
    # Calculate performance metrics
    metrics = {}
    for name, result in results.items():
        # Annualization factor
        if rebalance_freq == '1d':
            annualization = 252
        elif rebalance_freq == '1w':
            annualization = 52
        elif rebalance_freq == '1m':
            annualization = 12
        else:
            annualization = 252  # Default to daily
            
        final_return = result['cumulative_return'].iloc[-1]
        annual_return = (1 + final_return) ** (annualization / len(result)) - 1
        volatility = result['return'].std() * np.sqrt(annualization)
        sharpe = annual_return / volatility if volatility > 0 else 0
        max_drawdown = (result['pool_value'] / result['pool_value'].cummax() - 1).min()
        
        metrics[name] = {
            'Final Return': final_return * 100,
            'Annual Return': annual_return * 100,
            'Volatility': volatility * 100,
            'Sharpe Ratio': sharpe,
            'Max Drawdown': max_drawdown * 100
        }
    
    # Print metrics
    print("\nPerformance Metrics:")
    metrics_df = pd.DataFrame(metrics).T
    print(metrics_df)
    
    # Plot cumulative returns
    plt.figure(figsize=(12, 6))
    for name, result in results.items():
        plt.plot(result['cumulative_return'] * 100, label=name)
    
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return (%)')
    plt.title('Strategy Performance Comparison')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Plot drawdowns
    plt.figure(figsize=(12, 6))
    for name, result in results.items():
        drawdown = result['pool_value'] / result['pool_value'].cummax() - 1
        plt.plot(drawdown * 100, label=name)
    
    plt.xlabel('Date')
    plt.ylabel('Drawdown (%)')
    plt.title('Strategy Drawdowns')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    return results, metrics_df

def test_trading_fees_impact(initial_reserves, initial_weights, symbols, start_date, end_date, fee_levels=[0.0, 0.001, 0.003, 0.01]):
    """
    Test the impact of trading fees on different rebalancing strategies
    
    Parameters:
    -----------
    initial_reserves: array-like
        Initial token reserves in the pool
    initial_weights: array-like
        Initial weights for each token
    symbols: list
        List of token symbols to fetch data for
    start_date, end_date: str
        Start and end dates for the backtest in YYYY-MM-DD format
    fee_levels: list
        List of fee levels to test
        
    Returns:
    --------
    DataFrame with results for each strategy and fee level
    """
    strategies = {
        'Linear': lambda amm, target_weights, steps, fee: linear_rebalance_with_fees(amm, target_weights, steps, fee),
        'Approximately Optimal': lambda amm, target_weights, steps, fee: approx_optimal_rebalance_with_fees(amm, target_weights, steps, fee),
    }
    
    # Helper functions to incorporate fees
    def linear_rebalance_with_fees(amm, target_weights, steps, fee):
        """Linear interpolation with fees"""
        old_reserves = amm.reserves.copy()
        amm.linear_interpolation_rebalance(target_weights, steps)
        # Apply fee reduction to the final value
        fee_amount = calculate_fee(old_reserves, amm.reserves, fee)
        amm.reserves = amm.reserves * (1 - fee_amount / amm.get_pool_value())
        return amm
    
    def approx_optimal_rebalance_with_fees(amm, target_weights, steps, fee):
        """Approximately optimal rebalance with fees"""
        old_reserves = amm.reserves.copy()
        amm.approximate_optimal_rebalance(target_weights, steps)
        # Apply fee reduction to the final value
        fee_amount = calculate_fee(old_reserves, amm.reserves, fee)
        amm.reserves = amm.reserves * (1 - fee_amount / amm.get_pool_value())
        return amm
    
    def calculate_fee(old_reserves, new_reserves, fee_rate):
        """Calculate trading fee based on reserve changes"""
        # Simple fee model: fee applies to the absolute change in reserves
        reserve_changes = np.abs(new_reserves - old_reserves)
        fee_amount = np.sum(reserve_changes) * fee_rate
        return fee_amount
    
    # Fetch historical price data
    data = yf.download(symbols, start=start_date, end=end_date)['Adj Close']
    data = data.resample('1d').last().ffill()
    
    results = {}
    
    for strategy_name, strategy_fn in strategies.items():
        for fee in fee_levels:
            key = f"{strategy_name} (Fee: {fee*100:.1f}%)"
            print(f"\nRunning {key}...")
            
            # Initialize AMM
            initial_prices = data.iloc[0].values
            amm = DynamicAMM(initial_reserves, initial_weights, initial_prices)
            
            # Initialize results
            strategy_results = pd.DataFrame(index=data.index)
            strategy_results['pool_value'] = 0.0
            
            # Store initial state
            strategy_results.loc[data.index[0], 'pool_value'] = amm.get_pool_value()
            
            # Run simulation
            for i in tqdm(range(1, len(data))):
                date = data.index[i]
                
                # Update market prices
                new_prices = data.iloc[i].values
                amm.update_market_prices(new_prices)
                
                # Generate target weights (momentum strategy)
                price_changes = data.iloc[i] / data.iloc[i-1] - 1
                target_weights = amm.weights.copy()
                adjustment = 0.05 * price_changes.values
                target_weights += adjustment
                target_weights = np.maximum(target_weights, 0.01)
                target_weights = target_weights / np.sum(target_weights)
                
                # Apply rebalancing strategy with fees
                strategy_fn(amm, target_weights, steps=5, fee=fee)
                
                # Store results
                strategy_results.loc[date, 'pool_value'] = amm.get_pool_value()
            
            # Calculate returns
            strategy_results['return'] = strategy_results['pool_value'].pct_change()
            strategy_results['cumulative_return'] = (1 + strategy_results['return']).cumprod() - 1
            
            results[key] = strategy_results
    
    # Calculate performance metrics
    metrics = {}
    for name, result in results.items():
        final_return = result['cumulative_return'].iloc[-1]
        annual_return = (1 + final_return) ** (252 / len(result)) - 1
        volatility = result['return'].std() * np.sqrt(252)
        sharpe = annual_return / volatility if volatility > 0 else 0
        max_drawdown = (result['pool_value'] / result['pool_value'].cummax() - 1).min()
        
        metrics[name] = {
            'Final Return': final_return * 100,
            'Annual Return': annual_return * 100,
            'Volatility': volatility * 100,
            'Sharpe Ratio': sharpe,
            'Max Drawdown': max_drawdown * 100
        }
    
    # Print metrics
    print("\nPerformance Metrics with Different Fee Levels:")
    metrics_df = pd.DataFrame(metrics).T
    print(metrics_df)
    
    # Plot cumulative returns
    plt.figure(figsize=(14, 7))
    for name, result in results.items():
        plt.plot(result['cumulative_return'] * 100, label=name)
    
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return (%)')
    plt.title('Strategy Performance with Different Fee Levels')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Plot bar chart of final returns
    strategies = list(set([name.split(' (Fee:')[0] for name in results.keys()]))
    fee_labels = [f"{fee*100:.1f}%" for fee in fee_levels]
    
    grouped_data = {}
    for strategy in strategies:
        grouped_data[strategy] = [metrics[f"{strategy} (Fee: {fee*100:.1f}%)"]["Final Return"] for fee in fee_levels]
    
    # Plot grouped bar chart
    fig, ax = plt.subplots(figsize=(12, 6))
    bar_width = 0.35
    index = np.arange(len(fee_labels))
    
    for i, (strategy, values) in enumerate(grouped_data.items()):
        ax.bar(index + i*bar_width, values, bar_width, label=strategy)
    
    ax.set_xlabel('Fee Level')
    ax.set_ylabel('Final Return (%)')
    ax.set_title('Final Returns by Strategy and Fee Level')
    ax.set_xticks(index + bar_width/2)
    ax.set_xticklabels(fee_labels)
    ax.legend()
    plt.grid(True, axis='y')
    plt.show()
    
    return results, metrics_df

# Basic example with simulated data
def run_basic_example():
    # Set up a 3-token pool
    initial_reserves = [100, 100, 100]  # Initial reserves
    initial_weights = [0.4, 0.3, 0.3]   # Initial weights
    market_prices = [1.0, 2.0, 1.5]     # Market prices
    
    # Target weights for rebalancing
    target_weights = [0.6, 0.2, 0.2]    # Target weights
    
    # Compare rebalancing strategies
    results = compare_rebalancing_strategies(
        initial_reserves, 
        initial_weights, 
        target_weights, 
        market_prices,
        steps=10
    )
    
    return results

# Example with Bitcoin, Ethereum, and a stablecoin
def run_crypto_example():
    # Set up a 3-token pool (BTC, ETH, USDT)
    initial_reserves = [1, 10, 1000]  # Initial reserves (1 BTC, 10 ETH, 1000 USDT)
    initial_weights = [0.33, 0.33, 0.34]  # Initial weights
    
    # Compare strategies on historical data
    symbols = ['BTC-USD', 'ETH-USD', 'USDT-USD']
    start_date = '2022-07-01'
    end_date = '2023-06-30'
    
    results, metrics = compare_strategies_backtest(
        initial_reserves,
        initial_weights,
        symbols,
        start_date,
        end_date,
        rebalance_freq='1d'
    )
    
    # Test impact of trading fees
    fee_results, fee_metrics = test_trading_fees_impact(
        initial_reserves,
        initial_weights,
        symbols,
        start_date,
        end_date
    )
    
    return results, metrics, fee_results, fee_metrics

if __name__ == "__main__":
    print("Running basic example with simulated data...")
    basic_results = run_basic_example()
    
    print("\nRunning example with historical crypto data...")
    crypto_results, crypto_metrics, fee_results, fee_metrics = run_crypto_example()