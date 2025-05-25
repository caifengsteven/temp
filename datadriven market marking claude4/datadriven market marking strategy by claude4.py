import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
from scipy.stats import kstest, expon
from scipy.integrate import quad
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# 1. Hawkes Process Implementation
class HawkesProcess:
    """
    Hawkes process with power-law kernel and time-varying baseline
    """
    def __init__(self, baseline_type='piecewise_linear', n_intervals=6):
        self.baseline_type = baseline_type
        self.n_intervals = n_intervals
        self.params = {}
        
    def power_law_kernel(self, t, alpha, beta, delta):
        """Power law kernel: g(t) = alpha / (t + delta)^beta"""
        return alpha / (t + delta)**beta
    
    def exponential_kernel(self, t, alpha, beta):
        """Exponential kernel: g(t) = alpha * exp(-beta * t)"""
        return alpha * np.exp(-beta * t)
    
    def piecewise_linear_baseline(self, t, params, T=390):
        """Piecewise linear baseline intensity (for AAPL)"""
        interval_length = T / self.n_intervals
        interval_idx = min(int(t / interval_length), self.n_intervals - 1)
        
        t_start = interval_idx * interval_length
        t_end = (interval_idx + 1) * interval_length
        
        mu_start = params[f'mu{interval_idx+1}']
        mu_end = params[f'mu{interval_idx+2}'] if interval_idx < self.n_intervals - 1 else params[f'mu{self.n_intervals+1}']
        
        # Linear interpolation
        return mu_start + (mu_end - mu_start) * (t - t_start) / interval_length
    
    def piecewise_constant_baseline(self, t, params, T=390):
        """Piecewise constant baseline intensity (for BAC)"""
        interval_length = T / 3
        interval_idx = min(int(t / interval_length), 2)
        return params[f'mu{interval_idx+1}']
    
    def simulate(self, T, params, kernel='power_law', max_events=10000):
        """Simulate Hawkes process using thinning algorithm"""
        self.params = params
        
        events = []
        t = 0
        
        # Upper bound for thinning
        lambda_max = max(params[f'mu{i}'] for i in range(1, 8)) * 2
        
        while t < T and len(events) < max_events:
            # Generate next candidate event time
            t = t - np.log(np.random.rand()) / lambda_max
            
            if t > T:
                break
                
            # Calculate intensity at time t
            if self.baseline_type == 'piecewise_linear':
                baseline = self.piecewise_linear_baseline(t, params, T)
            else:
                baseline = self.piecewise_constant_baseline(t, params, T)
            
            # Add contribution from past events
            intensity = baseline
            for event_time in events:
                if kernel == 'power_law':
                    intensity += self.power_law_kernel(
                        t - event_time, 
                        params['alpha'], 
                        params['beta'], 
                        params['delta']
                    )
                else:
                    intensity += self.exponential_kernel(
                        t - event_time,
                        params['alpha'],
                        params['beta']
                    )
            
            # Accept/reject
            if np.random.rand() < intensity / lambda_max:
                events.append(t)
                # Update lambda_max if needed
                lambda_max = max(lambda_max, intensity * 1.2)
                
        return np.array(events)
    
    def calculate_intensity_at_times(self, times, events, params, kernel='power_law'):
        """Calculate intensity at given times"""
        intensities = []
        
        for t in times:
            if self.baseline_type == 'piecewise_linear':
                baseline = self.piecewise_linear_baseline(t, params, times[-1])
            else:
                baseline = self.piecewise_constant_baseline(t, params, times[-1])
            
            intensity = baseline
            for event_time in events[events < t]:
                if kernel == 'power_law':
                    intensity += self.power_law_kernel(
                        t - event_time,
                        params['alpha'],
                        params['beta'],
                        params['delta']
                    )
                else:
                    intensity += self.exponential_kernel(
                        t - event_time,
                        params['alpha'],
                        params['beta']
                    )
            
            intensities.append(intensity)
            
        return np.array(intensities)

# 2. Market Microstructure Model
class OptionsMarket:
    """Options market with order execution dynamics"""
    
    def __init__(self, kappa=10):
        self.kappa = kappa
        
    def execution_probability(self, distance, n_market_orders):
        """Probability of limit order execution given market orders"""
        return 1 - (1 - np.exp(-self.kappa * distance))**n_market_orders
    
    def simulate_execution(self, distance, n_market_orders):
        """Simulate whether limit order is executed"""
        prob = self.execution_probability(distance, n_market_orders)
        return np.random.rand() < prob

# 3. Option Pricing (Black-Scholes)
class BlackScholes:
    """Black-Scholes model for option pricing"""
    
    def __init__(self, r=0.02):
        self.r = r
        
    def d1(self, S, K, T, sigma):
        return (np.log(S/K) + (self.r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    
    def d2(self, S, K, T, sigma):
        return self.d1(S, K, T, sigma) - sigma*np.sqrt(T)
    
    def call_price(self, S, K, T, sigma):
        """European call option price"""
        from scipy.stats import norm
        if T <= 0:
            return max(S - K, 0)
        d1 = self.d1(S, K, T, sigma)
        d2 = self.d2(S, K, T, sigma)
        return S * norm.cdf(d1) - K * np.exp(-self.r * T) * norm.cdf(d2)
    
    def delta(self, S, K, T, sigma):
        """Option delta"""
        from scipy.stats import norm
        if T <= 0:
            return 1 if S > K else 0
        return norm.cdf(self.d1(S, K, T, sigma))
    
    def simulate_stock_path(self, S0, mu, sigma, T, n_steps):
        """Simulate stock price path"""
        dt = T / n_steps
        dW = np.random.randn(n_steps) * np.sqrt(dt)
        
        S = np.zeros(n_steps + 1)
        S[0] = S0
        
        for i in range(n_steps):
            S[i+1] = S[i] * np.exp((mu - 0.5*sigma**2)*dt + sigma*dW[i])
            
        return S

# 4. Neural Network Market Maker
class NeuralMarketMaker:
    """Deep learning market maker using feedforward neural networks"""
    
    def __init__(self, n_features=6, n_hidden=64, learning_rate=0.001):
        self.n_features = n_features
        self.n_hidden = n_hidden
        self.learning_rate = learning_rate
        self.models = {}
        
    def create_network(self, name):
        """Create a feedforward neural network"""
        model = keras.Sequential([
            layers.Input(shape=(self.n_features,)),
            layers.Dense(self.n_hidden, activation='relu'),
            layers.Dense(self.n_hidden, activation='relu'),
            layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='mse'
        )
        
        self.models[name] = model
        return model
    
    def predict_quotes(self, features, time_idx):
        """Predict bid and ask distances"""
        bid_model = self.models.get(f'bid_{time_idx}')
        ask_model = self.models.get(f'ask_{time_idx}')
        
        if bid_model is None or ask_model is None:
            return 0.1, 0.1  # Default distances
            
        bid_distance = bid_model.predict(features.reshape(1, -1), verbose=0)[0, 0]
        ask_distance = ask_model.predict(features.reshape(1, -1), verbose=0)[0, 0]
        
        return bid_distance, ask_distance

# 5. Market Making Simulation
class MarketMakingSimulator:
    """Simulate market making with different strategies"""
    
    def __init__(self, initial_cash=5000, initial_inventory=10, gamma=0):
        self.initial_cash = initial_cash
        self.initial_inventory = initial_inventory
        self.gamma = gamma
        self.market = OptionsMarket()
        self.bs = BlackScholes()
        
    def utility(self, wealth_change):
        """Exponential utility function"""
        if self.gamma == 0:
            return wealth_change
        else:
            return (1 - np.exp(-self.gamma * wealth_change)) / self.gamma
    
    def simulate_single_day(self, stock_prices, option_params, market_orders, 
                           strategy='constant', neural_model=None):
        """Simulate one day of market making"""
        n_periods = len(stock_prices) - 1
        
        # Initialize state
        cash = self.initial_cash
        option_inventory = self.initial_inventory
        stock_inventory = 0
        
        # Track P&L
        trades = []
        inventories = []
        
        for t in range(n_periods):
            # Current stock and option prices
            S = stock_prices[t]
            time_to_maturity = (n_periods - t) / (252 * 390)  # Convert to years
            C = self.bs.call_price(S, option_params['K'], time_to_maturity, option_params['sigma'])
            delta = self.bs.delta(S, option_params['K'], time_to_maturity, option_params['sigma'])
            
            # Delta hedge
            target_stock = -option_inventory * delta
            stock_trade = target_stock - stock_inventory
            cash -= stock_trade * S
            stock_inventory = target_stock
            
            # Get market orders in this period
            period_orders_buy = np.sum((market_orders['buy'] >= t) & (market_orders['buy'] < t+1))
            period_orders_sell = np.sum((market_orders['sell'] >= t) & (market_orders['sell'] < t+1))
            
            # Determine quotes based on strategy
            if strategy == 'constant':
                bid_distance = 0.15
                ask_distance = 0.15
            elif strategy == 'neural' and neural_model is not None:
                features = np.array([S, C, cash, option_inventory, stock_inventory, 0])
                bid_distance, ask_distance = neural_model.predict_quotes(features, t)
            else:
                bid_distance = 0.1
                ask_distance = 0.1
            
            # Simulate executions
            if period_orders_buy > 0:
                if self.market.simulate_execution(bid_distance, period_orders_buy):
                    # Sell to market buy order
                    cash += C + ask_distance
                    option_inventory -= 1
                    trades.append({'time': t, 'type': 'sell', 'price': C + ask_distance})
            
            if period_orders_sell > 0:
                if self.market.simulate_execution(ask_distance, period_orders_sell):
                    # Buy from market sell order
                    cash -= C - bid_distance
                    option_inventory += 1
                    trades.append({'time': t, 'type': 'buy', 'price': C - bid_distance})
            
            inventories.append({
                'time': t,
                'option': option_inventory,
                'stock': stock_inventory,
                'cash': cash
            })
        
        # Final valuation
        S_final = stock_prices[-1]
        C_final = max(S_final - option_params['K'], 0)  # At maturity
        final_wealth = cash + option_inventory * C_final + stock_inventory * S_final
        pnl = final_wealth - self.initial_cash
        
        return {
            'pnl': pnl,
            'trades': trades,
            'inventories': inventories,
            'final_wealth': final_wealth
        }

# 6. Generate Simulated Data and Run Experiments
def generate_sample_hawkes_params():
    """Generate sample parameters mimicking AAPL options"""
    return {
        'mu1': 25.0, 'mu2': 5.5, 'mu3': 4.0, 'mu4': 2.5,
        'mu5': 2.8, 'mu6': 2.6, 'mu7': 6.6,
        'alpha': 0.04, 'beta': 1.2, 'delta': 0.003
    }

def run_market_making_experiment():
    """Run complete market making experiment"""
    print("=== Options Market Making Experiment ===\n")
    
    # 1. Generate Hawkes process for market orders
    print("1. Simulating market order arrivals using Hawkes process...")
    
    hawkes = HawkesProcess(baseline_type='piecewise_linear')
    params = generate_sample_hawkes_params()
    
    # Simulate buy and sell orders
    buy_orders = hawkes.simulate(390, params)  # One trading day = 390 minutes
    sell_orders = hawkes.simulate(390, params)
    
    print(f"   Generated {len(buy_orders)} buy orders and {len(sell_orders)} sell orders")
    
    # 2. Simulate stock price path
    print("\n2. Simulating stock price path...")
    
    bs = BlackScholes()
    stock_prices = bs.simulate_stock_path(
        S0=220,  # AAPL-like price
        mu=0.0001,  # Daily drift
        sigma=0.02,  # Daily volatility
        T=1,
        n_steps=390
    )
    
    # 3. Set up option parameters
    option_params = {
        'K': 220,  # At-the-money
        'sigma': 0.25,  # Implied volatility
        'T': 5/252  # 5 days to maturity
    }
    
    # 4. Run different strategies
    print("\n3. Testing different market making strategies...")
    
    simulator = MarketMakingSimulator()
    market_orders = {'buy': buy_orders, 'sell': sell_orders}
    
    strategies = ['constant']
    results = {}
    
    for strategy in strategies:
        result = simulator.simulate_single_day(
            stock_prices, option_params, market_orders, strategy
        )
        results[strategy] = result
        print(f"   {strategy}: PnL = ${result['pnl']:.2f}")
    
    # 5. Analyze order flow patterns
    print("\n4. Analyzing order flow patterns...")
    
    # Create time bins
    time_bins = np.arange(0, 391, 10)  # 10-minute bins
    buy_counts = np.histogram(buy_orders, bins=time_bins)[0]
    sell_counts = np.histogram(sell_orders, bins=time_bins)[0]
    
    # 6. Test goodness of fit
    print("\n5. Testing Hawkes model goodness of fit...")
    
    # Calculate compensator
    times = np.sort(np.concatenate([buy_orders, [0, 390]]))
    compensator_values = []
    
    for i in range(1, len(times)):
        intensity_func = lambda t: hawkes.calculate_intensity_at_times(
            [t], buy_orders[buy_orders < t], params
        )[0]
        
        integral, _ = quad(intensity_func, times[i-1], times[i])
        compensator_values.append(integral)
    
    # KS test
    ks_stat, p_value = kstest(compensator_values, 'expon', args=(1,))
    print(f"   KS test p-value: {p_value:.4f}")
    print(f"   Model {'passes' if p_value > 0.05 else 'fails'} goodness of fit test")
    
    # 7. Visualizations
    print("\n6. Creating visualizations...")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Order arrivals over time
    ax = axes[0, 0]
    ax.bar(time_bins[:-1], buy_counts, width=10, alpha=0.6, label='Buy orders')
    ax.bar(time_bins[:-1], sell_counts, width=10, alpha=0.6, label='Sell orders')
    ax.set_xlabel('Time (minutes)')
    ax.set_ylabel('Number of orders')
    ax.set_title('Market Order Arrivals (U-shaped pattern)')
    ax.legend()
    
    # Plot 2: Hawkes intensity
    ax = axes[0, 1]
    time_grid = np.linspace(0, 390, 1000)
    buy_intensity = hawkes.calculate_intensity_at_times(time_grid, buy_orders, params)
    ax.plot(time_grid, buy_intensity, 'b-', label='Buy intensity')
    ax.set_xlabel('Time (minutes)')
    ax.set_ylabel('Intensity')
    ax.set_title('Hawkes Process Intensity')
    ax.legend()
    
    # Plot 3: Stock price path
    ax = axes[1, 0]
    ax.plot(stock_prices, 'g-')
    ax.set_xlabel('Time (minutes)')
    ax.set_ylabel('Stock Price ($)')
    ax.set_title('Simulated Stock Price Path')
    
    # Plot 4: Inventory evolution
    ax = axes[1, 1]
    if 'constant' in results:
        inventories = results['constant']['inventories']
        times = [inv['time'] for inv in inventories]
        option_inv = [inv['option'] for inv in inventories]
        ax.plot(times, option_inv, 'r-', label='Option inventory')
        ax.axhline(y=10, color='k', linestyle='--', alpha=0.5, label='Initial')
    ax.set_xlabel('Time (minutes)')
    ax.set_ylabel('Option Inventory')
    ax.set_title('Inventory Evolution')
    ax.legend()
    
    plt.tight_layout()
    plt.show()
    
    # 8. Summary statistics
    print("\n7. Summary Statistics:")
    print("="*50)
    
    for strategy, result in results.items():
        trades = result['trades']
        n_buys = sum(1 for t in trades if t['type'] == 'buy')
        n_sells = sum(1 for t in trades if t['type'] == 'sell')
        
        print(f"\nStrategy: {strategy}")
        print(f"  Total trades: {len(trades)}")
        print(f"  Buy trades: {n_buys}")
        print(f"  Sell trades: {n_sells}")
        print(f"  Final PnL: ${result['pnl']:.2f}")
        print(f"  Final wealth: ${result['final_wealth']:.2f}")
    
    return results

# 7. Expected Number of Arrivals (Volterra Equation)
def calculate_expected_arrivals(hawkes_process, events, params, current_time, horizon):
    """Calculate E[N_{t,t+h} | F_t] using Volterra equation"""
    
    # Simplified approximation for demonstration
    baseline_contribution = params['mu1'] * horizon
    
    # Self-excitation contribution
    excitation = 0
    for event in events[events < current_time]:
        remaining_effect = params['alpha'] / (current_time - event + params['delta'])**params['beta']
        excitation += remaining_effect * horizon * 0.5  # Approximation
    
    return baseline_contribution + excitation

# 8. Feature comparison experiment
def compare_features_experiment():
    """Compare different features for neural network"""
    print("\n=== Feature Comparison Experiment ===\n")
    
    # Generate data
    hawkes = HawkesProcess()
    params = generate_sample_hawkes_params()
    
    results_by_feature = {}
    
    feature_types = ['no_feature', 'intensity', 'expected_arrivals']
    
    for feature_type in feature_types:
        print(f"\nTesting feature: {feature_type}")
        
        # Simulate multiple days
        daily_pnls = []
        
        for day in range(20):  # 20 days
            # Generate new market orders
            buy_orders = hawkes.simulate(390, params)
            sell_orders = hawkes.simulate(390, params)
            
            # Simulate stock prices
            bs = BlackScholes()
            stock_prices = bs.simulate_stock_path(
                S0=220 + np.random.randn() * 5,  # Some variation
                mu=0.0001,
                sigma=0.02,
                T=1,
                n_steps=390
            )
            
            # Calculate features at different times
            if feature_type == 'intensity':
                buy_intensities = hawkes.calculate_intensity_at_times(
                    np.arange(390), buy_orders, params
                )
                sell_intensities = hawkes.calculate_intensity_at_times(
                    np.arange(390), sell_orders, params
                )
            elif feature_type == 'expected_arrivals':
                # Simplified calculation
                buy_expected = []
                sell_expected = []
                for t in range(390):
                    buy_exp = calculate_expected_arrivals(hawkes, buy_orders, params, t, 1)
                    sell_exp = calculate_expected_arrivals(hawkes, sell_orders, params, t, 1)
                    buy_expected.append(buy_exp)
                    sell_expected.append(sell_exp)
            
            # Simple strategy based on features
            if feature_type == 'no_feature':
                avg_distance = 0.12
            elif feature_type == 'intensity':
                # Higher intensity -> larger spread
                avg_intensity = np.mean(buy_intensities + sell_intensities)
                avg_distance = 0.08 + 0.002 * avg_intensity
            else:  # expected_arrivals
                avg_expected = np.mean(buy_expected + sell_expected)
                avg_distance = 0.08 + 0.02 * avg_expected
            
            # Simulate with adjusted strategy
            simulator = MarketMakingSimulator()
            result = simulator.simulate_single_day(
                stock_prices, 
                {'K': 220, 'sigma': 0.25, 'T': 5/252},
                {'buy': buy_orders, 'sell': sell_orders},
                'constant'  # Using constant with adjusted distance
            )
            
            daily_pnls.append(result['pnl'])
        
        results_by_feature[feature_type] = {
            'mean_pnl': np.mean(daily_pnls),
            'std_pnl': np.std(daily_pnls),
            'sharpe': np.mean(daily_pnls) / (np.std(daily_pnls) + 1e-6) * np.sqrt(252)
        }
    
    # Display results
    print("\n" + "="*60)
    print("Feature Comparison Results:")
    print("="*60)
    
    for feature, metrics in results_by_feature.items():
        print(f"\n{feature}:")
        print(f"  Mean daily PnL: ${metrics['mean_pnl']:.2f}")
        print(f"  Std daily PnL: ${metrics['std_pnl']:.2f}")
        print(f"  Annualized Sharpe: {metrics['sharpe']:.3f}")
    
    return results_by_feature

# Run the experiments
if __name__ == "__main__":
    # Run main experiment
    results = run_market_making_experiment()
    
    # Run feature comparison
    feature_results = compare_features_experiment()
    
    # Additional analysis: Self-excitation visualization
    print("\n=== Self-Excitation Analysis ===")
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Generate sample data for kernel visualization
    t_values = np.linspace(0.001, 10, 1000)
    
    # Power law kernel
    ax = axes[0]
    alpha, beta, delta = 0.04, 1.2, 0.003
    power_law_values = alpha / (t_values + delta)**beta
    ax.plot(t_values, power_law_values, 'b-', linewidth=2)
    ax.set_xlabel('Time lag (minutes)')
    ax.set_ylabel('Kernel value')
    ax.set_title('Power Law Kernel (Non-Markovian)')
    ax.set_xlim(0, 10)
    
    # Exponential kernel for comparison
    ax = axes[1]
    exp_values = alpha * np.exp(-beta * t_values)
    ax.plot(t_values, exp_values, 'r-', linewidth=2)
    ax.set_xlabel('Time lag (minutes)')
    ax.set_ylabel('Kernel value')
    ax.set_title('Exponential Kernel (Markovian)')
    ax.set_xlim(0, 10)
    
    plt.tight_layout()
    plt.show()
    
    print("\nExperiment completed successfully!")