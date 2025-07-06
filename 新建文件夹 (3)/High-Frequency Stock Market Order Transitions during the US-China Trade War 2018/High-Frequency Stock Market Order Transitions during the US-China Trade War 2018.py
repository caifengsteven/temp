import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from scipy import stats
from scipy.stats import chi2
import datetime
import warnings
import random
from tqdm import tqdm

warnings.filterwarnings('ignore')
np.random.seed(42)
random.seed(42)

# Order types (as described in the paper)
ORDER_TYPES = ['AB', 'AA', 'DB', 'DA', 'FB', 'FA', 'EB', 'EA', 'CB', 'CA']

# Define order type descriptions
ORDER_DESCRIPTIONS = {
    'AB': 'Add Bid order',
    'AA': 'Add Ask order',
    'DB': 'Delete outstanding Bid order in full',
    'DA': 'Delete outstanding Ask order in full',
    'FB': 'Execute outstanding Bid order in full',
    'FA': 'Execute outstanding Ask order in full',
    'EB': 'Execute outstanding Bid order in part',
    'EA': 'Execute outstanding Ask order in part',
    'CB': 'Cancel outstanding Bid order in part',
    'CA': 'Cancel outstanding Ask order in part'
}

class MarkovChainOrderAnalysis:
    def __init__(self, order_types=ORDER_TYPES):
        self.order_types = order_types
        self.n_states = len(order_types)
        self.transition_matrix = None
        self.state_map = {state: i for i, state in enumerate(order_types)}
        self.reverse_state_map = {i: state for i, state in enumerate(order_types)}
        
    def estimate_transition_matrix(self, order_sequence):
        """
        Estimate transition probability matrix using Maximum Likelihood Estimation.
        """
        # Initialize transition count matrix
        transition_counts = np.zeros((self.n_states, self.n_states))
        
        # Count transitions
        for i in range(len(order_sequence) - 1):
            from_state = self.state_map[order_sequence[i]]
            to_state = self.state_map[order_sequence[i + 1]]
            transition_counts[from_state, to_state] += 1
        
        # Convert counts to probabilities
        transition_matrix = np.zeros((self.n_states, self.n_states))
        row_sums = transition_counts.sum(axis=1)
        
        for i in range(self.n_states):
            if row_sums[i] > 0:
                transition_matrix[i, :] = transition_counts[i, :] / row_sums[i]
        
        self.transition_matrix = transition_matrix
        return transition_matrix
    
    def chi_square_test(self, order_sequence):
        """
        Perform Chi-square test to check if the order sequence has Markov property.
        """
        # Count transitions
        transition_counts = np.zeros((self.n_states, self.n_states))
        for i in range(len(order_sequence) - 1):
            from_state = self.state_map[order_sequence[i]]
            to_state = self.state_map[order_sequence[i + 1]]
            transition_counts[from_state, to_state] += 1
        
        # Calculate row and column sums
        row_sums = transition_counts.sum(axis=1)
        col_sums = transition_counts.sum(axis=0)
        total = transition_counts.sum()
        
        # Calculate expected counts under independence
        expected_counts = np.outer(row_sums, col_sums) / total
        
        # Calculate chi-square statistic
        valid_indices = (expected_counts > 0)
        chi_square_stat = np.sum(((transition_counts[valid_indices] - expected_counts[valid_indices]) ** 2) / 
                                  expected_counts[valid_indices])
        
        # Degrees of freedom: (r-1)*(c-1) where r and c are the numbers of non-zero rows and columns
        r = np.sum(row_sums > 0)
        c = np.sum(col_sums > 0)
        df = (r - 1) * (c - 1)
        
        # Calculate p-value
        p_value = 1 - chi2.cdf(chi_square_stat, df)
        
        return chi_square_stat, p_value, df
    
    def calculate_stationary_distribution(self):
        """
        Calculate the stationary distribution of the Markov chain.
        """
        if self.transition_matrix is None:
            raise ValueError("Transition matrix must be estimated first")
        
        # Initialize with uniform distribution
        n = self.transition_matrix.shape[0]
        pi = np.ones(n) / n
        
        # Power method to find the stationary distribution
        max_iter = 1000
        epsilon = 1e-8
        
        for _ in range(max_iter):
            pi_new = pi @ self.transition_matrix
            if np.max(np.abs(pi_new - pi)) < epsilon:
                break
            pi = pi_new
        
        return pi
    
    def calculate_mean_recurrence_time(self):
        """
        Calculate the mean recurrence time for each state.
        """
        stationary_dist = self.calculate_stationary_distribution()
        mean_recurrence_time = 1 / stationary_dist
        return mean_recurrence_time
    
    def calculate_spectral_gap(self):
        """
        Calculate the spectral gap of the transition matrix.
        """
        if self.transition_matrix is None:
            raise ValueError("Transition matrix must be estimated first")
        
        # Calculate eigenvalues
        eigenvalues = np.linalg.eigvals(self.transition_matrix)
        
        # Sort eigenvalues by magnitude
        eigenvalues = np.sort(np.abs(eigenvalues))[::-1]
        
        # Spectral gap is the difference between the two largest eigenvalues
        spectral_gap = eigenvalues[0] - eigenvalues[1]
        
        # Relaxation rate is the inverse of the second largest eigenvalue
        relaxation_rate = 1 / (1 - eigenvalues[1]) if eigenvalues[1] < 1 else float('inf')
        
        return spectral_gap, relaxation_rate
    
    def calculate_entropy_rate(self):
        """
        Calculate the entropy rate of the Markov chain.
        """
        if self.transition_matrix is None:
            raise ValueError("Transition matrix must be estimated first")
        
        # Get stationary distribution
        pi = self.calculate_stationary_distribution()
        
        # Calculate entropy rate
        entropy_rate = 0
        for i in range(self.n_states):
            for j in range(self.n_states):
                if self.transition_matrix[i, j] > 0 and pi[i] > 0:
                    entropy_rate -= pi[i] * self.transition_matrix[i, j] * np.log2(self.transition_matrix[i, j])
        
        return entropy_rate
    
    def plot_transition_matrix(self, title="Transition Probability Matrix"):
        """
        Plot the transition probability matrix as a heatmap.
        """
        if self.transition_matrix is None:
            raise ValueError("Transition matrix must be estimated first")
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(self.transition_matrix, annot=True, fmt=".2f", cmap="YlGnBu", 
                    xticklabels=self.order_types, yticklabels=self.order_types)
        plt.title(title)
        plt.xlabel("Current State")
        plt.ylabel("Next State")
        plt.tight_layout()
        plt.show()
    
    def generate_sequence(self, length, initial_state=None):
        """
        Generate a sequence of orders based on the transition matrix.
        """
        if self.transition_matrix is None:
            raise ValueError("Transition matrix must be estimated first")
        
        # Initialize sequence
        sequence = []
        
        # Select initial state
        if initial_state is None:
            initial_idx = np.random.choice(self.n_states)
        else:
            initial_idx = self.state_map[initial_state]
        
        current_state = initial_idx
        sequence.append(self.reverse_state_map[current_state])
        
        # Generate sequence
        for _ in range(length - 1):
            probs = self.transition_matrix[current_state, :]
            next_state = np.random.choice(self.n_states, p=probs)
            sequence.append(self.reverse_state_map[next_state])
            current_state = next_state
        
        return sequence

class OrderSimulator:
    def __init__(self, volatility='low'):
        # Set transition probabilities based on the paper's findings
        self.volatility = volatility
        
        # Create transition matrices for high and low volatility scenarios
        # These are approximate values based on the paper's heatmaps
        # High volatility transition matrix
        self.high_vol_trans_matrix = {
            'AB': {'AB': 0.30, 'AA': 0.15, 'DB': 0.40, 'DA': 0.05, 'FB': 0.05, 'FA': 0.01, 'EB': 0.02, 'EA': 0.01, 'CB': 0.01, 'CA': 0.00},
            'AA': {'AB': 0.15, 'AA': 0.30, 'DB': 0.05, 'DA': 0.40, 'FB': 0.01, 'FA': 0.05, 'EB': 0.01, 'EA': 0.02, 'CB': 0.00, 'CA': 0.01},
            'DB': {'AB': 0.60, 'AA': 0.10, 'DB': 0.20, 'DA': 0.05, 'FB': 0.02, 'FA': 0.01, 'EB': 0.01, 'EA': 0.01, 'CB': 0.00, 'CA': 0.00},
            'DA': {'AB': 0.10, 'AA': 0.60, 'DB': 0.05, 'DA': 0.20, 'FB': 0.01, 'FA': 0.02, 'EB': 0.01, 'EA': 0.01, 'CB': 0.00, 'CA': 0.00},
            'FB': {'AB': 0.15, 'AA': 0.52, 'DB': 0.10, 'DA': 0.10, 'FB': 0.05, 'FA': 0.05, 'EB': 0.01, 'EA': 0.01, 'CB': 0.01, 'CA': 0.00},
            'FA': {'AB': 0.51, 'AA': 0.15, 'DB': 0.10, 'DA': 0.10, 'FB': 0.05, 'FA': 0.05, 'EB': 0.01, 'EA': 0.01, 'CB': 0.00, 'CA': 0.02},
            'EB': {'AB': 0.20, 'AA': 0.40, 'DB': 0.10, 'DA': 0.15, 'FB': 0.05, 'FA': 0.05, 'EB': 0.02, 'EA': 0.01, 'CB': 0.01, 'CA': 0.01},
            'EA': {'AB': 0.40, 'AA': 0.20, 'DB': 0.15, 'DA': 0.10, 'FB': 0.05, 'FA': 0.05, 'EB': 0.01, 'EA': 0.02, 'CB': 0.01, 'CA': 0.01},
            'CB': {'AB': 0.30, 'AA': 0.20, 'DB': 0.25, 'DA': 0.10, 'FB': 0.05, 'FA': 0.02, 'EB': 0.03, 'EA': 0.02, 'CB': 0.02, 'CA': 0.01},
            'CA': {'AB': 0.20, 'AA': 0.30, 'DB': 0.10, 'DA': 0.25, 'FB': 0.02, 'FA': 0.05, 'EB': 0.02, 'EA': 0.03, 'CB': 0.01, 'CA': 0.02}
        }
        
        # Low volatility transition matrix
        self.low_vol_trans_matrix = {
            'AB': {'AB': 0.35, 'AA': 0.15, 'DB': 0.35, 'DA': 0.05, 'FB': 0.04, 'FA': 0.02, 'EB': 0.02, 'EA': 0.01, 'CB': 0.01, 'CA': 0.00},
            'AA': {'AB': 0.15, 'AA': 0.35, 'DB': 0.05, 'DA': 0.35, 'FB': 0.02, 'FA': 0.04, 'EB': 0.01, 'EA': 0.02, 'CB': 0.00, 'CA': 0.01},
            'DB': {'AB': 0.55, 'AA': 0.10, 'DB': 0.25, 'DA': 0.05, 'FB': 0.02, 'FA': 0.01, 'EB': 0.01, 'EA': 0.01, 'CB': 0.00, 'CA': 0.00},
            'DA': {'AB': 0.10, 'AA': 0.55, 'DB': 0.05, 'DA': 0.25, 'FB': 0.01, 'FA': 0.02, 'EB': 0.01, 'EA': 0.01, 'CB': 0.00, 'CA': 0.00},
            'FB': {'AB': 0.15, 'AA': 0.47, 'DB': 0.10, 'DA': 0.10, 'FB': 0.10, 'FA': 0.05, 'EB': 0.01, 'EA': 0.01, 'CB': 0.01, 'CA': 0.00},
            'FA': {'AB': 0.45, 'AA': 0.15, 'DB': 0.10, 'DA': 0.10, 'FB': 0.05, 'FA': 0.10, 'EB': 0.01, 'EA': 0.01, 'CB': 0.00, 'CA': 0.03},
            'EB': {'AB': 0.20, 'AA': 0.35, 'DB': 0.10, 'DA': 0.15, 'FB': 0.07, 'FA': 0.05, 'EB': 0.05, 'EA': 0.01, 'CB': 0.01, 'CA': 0.01},
            'EA': {'AB': 0.35, 'AA': 0.20, 'DB': 0.15, 'DA': 0.10, 'FB': 0.05, 'FA': 0.07, 'EB': 0.01, 'EA': 0.05, 'CB': 0.01, 'CA': 0.01},
            'CB': {'AB': 0.30, 'AA': 0.20, 'DB': 0.20, 'DA': 0.10, 'FB': 0.05, 'FA': 0.02, 'EB': 0.05, 'EA': 0.02, 'CB': 0.05, 'CA': 0.01},
            'CA': {'AB': 0.20, 'AA': 0.30, 'DB': 0.10, 'DA': 0.20, 'FB': 0.02, 'FA': 0.05, 'EB': 0.02, 'EA': 0.05, 'CB': 0.01, 'CA': 0.05}
        }
        
    def simulate_orders(self, n_orders, initial_state='AB'):
        """
        Simulate a sequence of n_orders using the transition matrix.
        """
        # Choose the appropriate transition matrix
        trans_matrix = self.high_vol_trans_matrix if self.volatility == 'high' else self.low_vol_trans_matrix
        
        # Initialize order sequence
        order_sequence = [initial_state]
        
        # Generate sequence
        for _ in range(n_orders - 1):
            current_state = order_sequence[-1]
            next_state = np.random.choice(ORDER_TYPES, p=[trans_matrix[current_state][next_state] for next_state in ORDER_TYPES])
            order_sequence.append(next_state)
        
        return order_sequence

class StockPriceSimulator:
    def __init__(self, initial_price=100, volatility='low', sector='IT'):
        self.price = initial_price
        self.volatility = volatility
        self.sector = sector
        
        # Set volatility parameters
        if volatility == 'high':
            self.vol_factor = 0.002  # Higher price changes
            self.jump_prob = 0.05    # Higher probability of jumps
        else:
            self.vol_factor = 0.001  # Lower price changes
            self.jump_prob = 0.02    # Lower probability of jumps
        
        # Sector-specific bias (based on the paper's findings)
        self.sector_bias = {
            'Energy': 0.0,  # Neutral
            'Finance & Banking': 0.0005,  # Slightly positive (resilient)
            'FMCG': 0.0,  # Neutral
            'Healthcare': -0.0002,  # Slightly negative
            'IT': -0.0003,  # Slightly negative
            'Real Estate': -0.0001  # Slightly negative
        }.get(sector, 0.0)
        
        self.order_book = {
            'bids': [],  # (price, quantity)
            'asks': []   # (price, quantity)
        }
        
        # Initialize with some orders
        for i in range(10):
            self.order_book['bids'].append((initial_price * (0.99 - i*0.002), 100))
            self.order_book['asks'].append((initial_price * (1.01 + i*0.002), 100))
    
    def update_price(self, order_type, quantity=100):
        """
        Update the stock price based on the order type.
        """
        # Add some randomness to the price change
        random_factor = np.random.normal(0, self.vol_factor)
        
        # Apply sector bias
        bias = self.sector_bias
        
        # Process order and update price
        if order_type in ['AB', 'EB', 'FB']:  # Buying pressure
            price_change = self.vol_factor + random_factor + bias
            # Add to order book if it's an add order
            if order_type == 'AB':
                bid_price = self.price * (1 - np.random.uniform(0, 0.01))
                self.order_book['bids'].append((bid_price, quantity))
                self.order_book['bids'].sort(reverse=True)
            # Execute if it's an execution order
            elif order_type in ['EB', 'FB'] and self.order_book['asks']:
                self.order_book['asks'].pop()  # Remove the lowest ask
        
        elif order_type in ['AA', 'EA', 'FA']:  # Selling pressure
            price_change = -self.vol_factor + random_factor + bias
            # Add to order book if it's an add order
            if order_type == 'AA':
                ask_price = self.price * (1 + np.random.uniform(0, 0.01))
                self.order_book['asks'].append((ask_price, quantity))
                self.order_book['asks'].sort()
            # Execute if it's an execution order
            elif order_type in ['EA', 'FA'] and self.order_book['bids']:
                self.order_book['bids'].pop(0)  # Remove the highest bid
        
        elif order_type in ['DB', 'CB']:  # Delete or cancel bid
            price_change = -0.0001 + random_factor + bias
            # Remove from order book
            if self.order_book['bids']:
                del_idx = np.random.randint(0, len(self.order_book['bids']))
                self.order_book['bids'].pop(del_idx)
        
        elif order_type in ['DA', 'CA']:  # Delete or cancel ask
            price_change = 0.0001 + random_factor + bias
            # Remove from order book
            if self.order_book['asks']:
                del_idx = np.random.randint(0, len(self.order_book['asks']))
                self.order_book['asks'].pop(del_idx)
        
        else:
            price_change = random_factor + bias
        
        # Random jumps based on volatility
        if np.random.random() < self.jump_prob:
            jump_size = np.random.normal(0, self.vol_factor * 10)
            price_change += jump_size
        
        # Update price
        self.price *= (1 + price_change)
        
        # Ensure price doesn't go negative
        self.price = max(self.price, 0.01)
        
        return self.price
    
    def get_bid_ask_spread(self):
        """Get the current bid-ask spread"""
        if not self.order_book['bids'] or not self.order_book['asks']:
            return None
        
        highest_bid = self.order_book['bids'][0][0] if self.order_book['bids'] else 0
        lowest_ask = self.order_book['asks'][0][0] if self.order_book['asks'] else float('inf')
        
        return lowest_ask - highest_bid
    
    def get_order_book_imbalance(self):
        """Calculate order book imbalance (bid volume - ask volume)"""
        bid_volume = sum(qty for _, qty in self.order_book['bids'])
        ask_volume = sum(qty for _, qty in self.order_book['asks'])
        
        total_volume = bid_volume + ask_volume
        if total_volume == 0:
            return 0
        
        return (bid_volume - ask_volume) / total_volume

class MarkovTradingStrategy:
    def __init__(self, markov_model, initial_cash=100000):
        self.model = markov_model
        self.cash = initial_cash
        self.shares = 0
        self.trades = []
        self.portfolio_values = []
        self.recent_orders = []  # Store recent orders to look for patterns
        
    def update_orders(self, order):
        """Add a new order to the recent orders list"""
        self.recent_orders.append(order)
        if len(self.recent_orders) > 10:  # Keep only the last 10 orders
            self.recent_orders.pop(0)
    
    def should_trade(self, price, order_book_imbalance):
        """
        Decide whether to trade based on recent order patterns and model predictions.
        Returns 'buy', 'sell', or None.
        """
        if len(self.recent_orders) < 5:
            return None
        
        # Count consecutive add/delete patterns
        add_delete_count = 0
        for i in range(len(self.recent_orders) - 1):
            if (self.recent_orders[i] == 'AB' and self.recent_orders[i+1] == 'DB') or \
               (self.recent_orders[i] == 'AA' and self.recent_orders[i+1] == 'DA'):
                add_delete_count += 1
        
        # Count full executions
        fb_count = self.recent_orders.count('FB')
        fa_count = self.recent_orders.count('FA')
        
        # Look for patterns identified in the paper
        if 'FB' in self.recent_orders[-2:] and 'AA' in self.recent_orders[-1:]:
            # After a full execution of a buy order, many traders add ask orders
            # This might indicate a local top
            return 'sell'
        
        if 'FA' in self.recent_orders[-2:] and 'AB' in self.recent_orders[-1:]:
            # After a full execution of a sell order, many traders add bid orders
            # This might indicate a local bottom
            return 'buy'
        
        # If there are many add/delete patterns, be cautious (market manipulation)
        if add_delete_count >= 3:
            # Use order book imbalance to decide
            if order_book_imbalance > 0.3:  # More bids than asks
                return 'buy'
            elif order_book_imbalance < -0.3:  # More asks than bids
                return 'sell'
        
        # If there are multiple full executions in the same direction, follow the trend
        if fb_count >= 2 and fa_count == 0:
            return 'buy'
        elif fa_count >= 2 and fb_count == 0:
            return 'sell'
        
        return None
    
    def execute_trade(self, action, price, timestamp):
        """Execute a buy or sell trade"""
        if action == 'buy' and self.cash > 0:
            shares_to_buy = int(self.cash * 0.1 / price)  # Use 10% of available cash
            if shares_to_buy > 0:
                cost = shares_to_buy * price
                self.cash -= cost
                self.shares += shares_to_buy
                self.trades.append({
                    'timestamp': timestamp,
                    'action': 'buy',
                    'price': price,
                    'shares': shares_to_buy,
                    'value': cost
                })
        
        elif action == 'sell' and self.shares > 0:
            shares_to_sell = int(self.shares * 0.5)  # Sell 50% of shares
            if shares_to_sell > 0:
                revenue = shares_to_sell * price
                self.cash += revenue
                self.shares -= shares_to_sell
                self.trades.append({
                    'timestamp': timestamp,
                    'action': 'sell',
                    'price': price,
                    'shares': shares_to_sell,
                    'value': revenue
                })
    
    def update_portfolio_value(self, current_price, timestamp):
        """Update the portfolio value"""
        value = self.cash + (self.shares * current_price)
        self.portfolio_values.append({
            'timestamp': timestamp,
            'cash': self.cash,
            'shares': self.shares,
            'share_price': current_price,
            'total_value': value
        })
        
        return value

def simulate_trading_day(volatility='low', sector='IT', n_orders=5000):
    """
    Simulate a trading day with the given volatility and generate orders, prices, and execute a trading strategy.
    """
    # Initialize simulators
    order_sim = OrderSimulator(volatility=volatility)
    price_sim = StockPriceSimulator(volatility=volatility, sector=sector)
    
    # Generate orders
    orders = order_sim.simulate_orders(n_orders)
    
    # Analyze orders with Markov model
    markov_model = MarkovChainOrderAnalysis()
    markov_model.estimate_transition_matrix(orders)
    
    # Initialize trading strategy
    strategy = MarkovTradingStrategy(markov_model)
    
    # Simulate trading
    prices = []
    timestamps = []
    order_book_imbalances = []
    
    start_time = datetime.datetime(2023, 1, 1, 9, 30, 0)  # Market open
    
    for i, order in enumerate(orders):
        # Update timestamp (random time increment between orders)
        if i == 0:
            current_time = start_time
        else:
            seconds_increment = np.random.exponential(5)  # Average 5 seconds between orders
            current_time = current_time + datetime.timedelta(seconds=seconds_increment)
        
        timestamps.append(current_time)
        
        # Update price based on order
        price = price_sim.update_price(order)
        prices.append(price)
        
        # Get order book imbalance
        imbalance = price_sim.get_order_book_imbalance()
        order_book_imbalances.append(imbalance)
        
        # Update strategy with new order
        strategy.update_orders(order)
        
        # Decide whether to trade
        action = strategy.should_trade(price, imbalance)
        if action:
            strategy.execute_trade(action, price, current_time)
        
        # Update portfolio value
        strategy.update_portfolio_value(price, current_time)
    
    # Create DataFrame with results
    results_df = pd.DataFrame({
        'timestamp': timestamps,
        'order': orders,
        'price': prices,
        'order_book_imbalance': order_book_imbalances
    })
    
    # Convert portfolio values to DataFrame
    portfolio_df = pd.DataFrame(strategy.portfolio_values)
    
    # Trade history
    trades_df = pd.DataFrame(strategy.trades) if strategy.trades else pd.DataFrame()
    
    return results_df, portfolio_df, trades_df, markov_model

def analyze_markov_properties(markov_model, order_sequence):
    """
    Analyze and print various properties of the Markov chain.
    """
    # Chi-square test
    chi2_stat, p_value, df = markov_model.chi_square_test(order_sequence)
    print(f"Chi-square test: statistic={chi2_stat:.2f}, p-value={p_value:.8f}, df={df}")
    
    # Stationary distribution
    stationary_dist = markov_model.calculate_stationary_distribution()
    print("\nStationary Distribution:")
    for i, order in enumerate(ORDER_TYPES):
        print(f"{order}: {stationary_dist[i]:.4f}")
    
    # Mean recurrence time
    mrt = markov_model.calculate_mean_recurrence_time()
    print("\nMean Recurrence Time:")
    for i, order in enumerate(ORDER_TYPES):
        print(f"{order}: {mrt[i]:.2f}")
    
    # Spectral gap and relaxation rate
    spectral_gap, relaxation_rate = markov_model.calculate_spectral_gap()
    print(f"\nSpectral Gap: {spectral_gap:.4f}")
    print(f"Relaxation Rate: {relaxation_rate:.4f}")
    
    # Entropy rate
    entropy_rate = markov_model.calculate_entropy_rate()
    print(f"Entropy Rate: {entropy_rate:.4f}")
    
    # Plot transition matrix
    markov_model.plot_transition_matrix()

def compare_volatility_scenarios(sectors=['IT', 'Finance & Banking']):
    """
    Compare trading strategy performance under high and low volatility scenarios.
    """
    # Results storage
    all_results = []
    
    for sector in sectors:
        for volatility in ['high', 'low']:
            print(f"Simulating {volatility} volatility day for {sector} sector...")
            results_df, portfolio_df, trades_df, markov_model = simulate_trading_day(
                volatility=volatility, sector=sector, n_orders=5000)
            
            # Calculate performance metrics
            initial_value = portfolio_df['total_value'].iloc[0] if not portfolio_df.empty else 100000
            final_value = portfolio_df['total_value'].iloc[-1] if not portfolio_df.empty else 100000
            returns = (final_value / initial_value) - 1
            
            # Calculate volatility of portfolio
            portfolio_returns = portfolio_df['total_value'].pct_change().dropna()
            portfolio_volatility = portfolio_returns.std() * np.sqrt(252)  # Annualized
            
            # Calculate Sharpe ratio (assuming risk-free rate of 0)
            sharpe_ratio = (returns / portfolio_volatility) if portfolio_volatility > 0 else 0
            
            # Number of trades
            n_trades = len(trades_df) if not trades_df.empty else 0
            
            # Calculate properties of the Markov chain
            stationary_dist = markov_model.calculate_stationary_distribution()
            mrt = markov_model.calculate_mean_recurrence_time()
            spectral_gap, relaxation_rate = markov_model.calculate_spectral_gap()
            entropy_rate = markov_model.calculate_entropy_rate()
            
            # Store results
            all_results.append({
                'sector': sector,
                'volatility': volatility,
                'returns': returns,
                'portfolio_volatility': portfolio_volatility,
                'sharpe_ratio': sharpe_ratio,
                'n_trades': n_trades,
                'stationary_dist': dict(zip(ORDER_TYPES, stationary_dist)),
                'mean_recurrence_time': dict(zip(ORDER_TYPES, mrt)),
                'spectral_gap': spectral_gap,
                'relaxation_rate': relaxation_rate,
                'entropy_rate': entropy_rate,
                'results_df': results_df,
                'portfolio_df': portfolio_df,
                'trades_df': trades_df,
                'markov_model': markov_model
            })
    
    # Create comparison table
    comparison_table = []
    for result in all_results:
        row = {
            'Sector': result['sector'],
            'Volatility': result['volatility'],
            'Returns': f"{result['returns']*100:.2f}%",
            'Portfolio Volatility': f"{result['portfolio_volatility']*100:.2f}%",
            'Sharpe Ratio': f"{result['sharpe_ratio']:.2f}",
            'Number of Trades': result['n_trades'],
            'Spectral Gap': f"{result['spectral_gap']:.4f}",
            'Entropy Rate': f"{result['entropy_rate']:.4f}"
        }
        comparison_table.append(row)
    
    # Display comparison table
    comparison_df = pd.DataFrame(comparison_table)
    print("\nComparison of Volatility Scenarios:")
    print(comparison_df)
    
    # Plot price and portfolio value for each scenario
    fig, axes = plt.subplots(len(sectors), 2, figsize=(15, 5*len(sectors)))
    
    for i, sector in enumerate(sectors):
        for j, volatility in enumerate(['high', 'low']):
            result = next(r for r in all_results if r['sector'] == sector and r['volatility'] == volatility)
            
            # Get axis
            if len(sectors) > 1:
                ax = axes[i, j]
            else:
                ax = axes[j]
            
            # Plot price
            ax.plot(result['results_df']['price'], label='Stock Price', alpha=0.7)
            
            # Add a second y-axis for portfolio value
            ax2 = ax.twinx()
            if not result['portfolio_df'].empty:
                ax2.plot(result['portfolio_df']['total_value'], 'g-', label='Portfolio Value')
            
            # Add trade markers
            if not result['trades_df'].empty:
                for _, trade in result['trades_df'].iterrows():
                    idx = result['results_df']['timestamp'].searchsorted(trade['timestamp'])
                    if trade['action'] == 'buy':
                        ax.plot(idx, result['results_df']['price'].iloc[idx], '^', markersize=8, color='g')
                    else:  # sell
                        ax.plot(idx, result['results_df']['price'].iloc[idx], 'v', markersize=8, color='r')
            
            ax.set_title(f"{sector} - {volatility.capitalize()} Volatility")
            ax.set_xlabel('Time (Order Sequence)')
            ax.set_ylabel('Price')
            ax2.set_ylabel('Portfolio Value')
            
            # Add legend
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    plt.tight_layout()
    plt.show()
    
    # Plot transition matrices for comparison
    fig, axes = plt.subplots(len(sectors), 2, figsize=(15, 7*len(sectors)))
    
    for i, sector in enumerate(sectors):
        for j, volatility in enumerate(['high', 'low']):
            result = next(r for r in all_results if r['sector'] == sector and r['volatility'] == volatility)
            
            # Get axis
            if len(sectors) > 1:
                ax = axes[i, j]
            else:
                ax = axes[j]
            
            # Plot transition matrix
            sns.heatmap(result['markov_model'].transition_matrix, annot=True, fmt=".2f", 
                        cmap="YlGnBu", xticklabels=ORDER_TYPES, yticklabels=ORDER_TYPES, ax=ax)
            ax.set_title(f"{sector} - {volatility.capitalize()} Volatility")
            ax.set_xlabel("Current State")
            ax.set_ylabel("Next State")
    
    plt.tight_layout()
    plt.show()
    
    # Plot stationary distributions for comparison
    fig, ax = plt.subplots(figsize=(12, 6))
    
    bar_width = 0.2
    index = np.arange(len(ORDER_TYPES))
    
    for i, result in enumerate(all_results):
        label = f"{result['sector']} - {result['volatility'].capitalize()} Volatility"
        ax.bar(index + i*bar_width, [result['stationary_dist'][order] for order in ORDER_TYPES], 
               bar_width, label=label)
    
    ax.set_xlabel('Order Type')
    ax.set_ylabel('Stationary Probability')
    ax.set_title('Comparison of Stationary Distributions')
    ax.set_xticks(index + bar_width * (len(all_results) - 1) / 2)
    ax.set_xticklabels(ORDER_TYPES)
    ax.legend()
    
    plt.tight_layout()
    plt.show()
    
    return all_results, comparison_df

# Main execution
if __name__ == "__main__":
    # Simulate a trading day with high volatility
    print("Simulating high volatility trading day...")
    high_vol_results, high_vol_portfolio, high_vol_trades, high_vol_model = simulate_trading_day(
        volatility='high', n_orders=5000)
    
    # Analyze Markov properties
    print("\nMarkov Properties for High Volatility Day:")
    analyze_markov_properties(high_vol_model, high_vol_results['order'].tolist())
    
    # Compare different volatility scenarios
    print("\nComparing volatility scenarios across sectors...")
    all_results, comparison_df = compare_volatility_scenarios()
    
    print("\nSimulation complete!")