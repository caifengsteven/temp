import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import pandas as pd
from matplotlib.ticker import MaxNLocator
from typing import List, Tuple, Dict, Optional
import time
from scipy.stats import poisson

class LimitOrderBook:
    """
    A class implementing the mathematical framework for limit order book dynamics as described in
    "A mathematical framework for modelling order book dynamics" by Cont, Degond, and Xuan.
    """
    
    def __init__(self, d: int = 10, initial_mid_price: float = 100.0, tick_size: float = 0.01):
        """
        Initialize the limit order book.
        
        Args:
            d: Maximum price distance from mid price (defines the size of the order book)
            initial_mid_price: Initial mid price
            tick_size: Minimum price increment
        """
        self.d = d
        self.tick_size = tick_size
        self.mid_price = initial_mid_price
        
        # Initialize the order book
        # X+ represents buy orders, X- represents sell orders
        # Each index represents a price level (relative to mid price)
        self.X_plus = np.zeros(d+1, dtype=int)  # Buy side
        self.X_minus = np.zeros(d+1, dtype=int)  # Sell side
        
        # Set some initial orders in the book to avoid empty book
        self.X_plus[0:3] = [3, 2, 1]  # Initial buy orders
        self.X_minus[1:4] = [1, 2, 3]  # Initial sell orders
        
        # Track price history
        self.price_history = [initial_mid_price]
        self.time_history = [0]
        
        # Track order book state history
        self.book_history = []
        self.save_state()
        
        # Calculate initial bid and ask prices
        self._update_bid_ask()
        
    def _update_bid_ask(self):
        """
        Update the bid and ask prices based on the current state of the book.
        """
        # Find the highest price with buy orders (the bid)
        bid_index = np.max(np.where(self.X_plus > 0)[0]) if np.any(self.X_plus > 0) else 0
        self.bid_price = self.mid_price - (self.d - bid_index) * self.tick_size
        
        # Find the lowest price with sell orders (the ask)
        ask_index = np.min(np.where(self.X_minus > 0)[0]) if np.any(self.X_minus > 0) else self.d
        self.ask_price = self.mid_price + (ask_index) * self.tick_size
        
        # Check constraint that ask > bid
        if self.ask_price <= self.bid_price:
            # This shouldn't happen after proper market clearing
            print(f"Warning: Ask price {self.ask_price} <= Bid price {self.bid_price}")
    
    def b(self, X=None):
        """
        Get the bid price index (highest price with buy orders).
        Args:
            X: Optional order book state to check (default is current state)
        """
        if X is None:
            X_plus = self.X_plus
        else:
            X_plus = X[0]
            
        return np.max(np.where(X_plus > 0)[0]) if np.any(X_plus > 0) else 0
    
    def a(self, X=None):
        """
        Get the ask price index (lowest price with sell orders).
        Args:
            X: Optional order book state to check (default is current state)
        """
        if X is None:
            X_minus = self.X_minus
        else:
            X_minus = X[1]
            
        return np.min(np.where(X_minus > 0)[0]) if np.any(X_minus > 0) else self.d
    
    def save_state(self):
        """Save the current state of the order book for later analysis."""
        self.book_history.append({
            'time': self.time_history[-1],
            'mid_price': self.mid_price,
            'bid_price': self.bid_price,
            'ask_price': self.ask_price,
            'X_plus': self.X_plus.copy(),
            'X_minus': self.X_minus.copy()
        })
    
    def get_B_X(self, k):
        """
        Calculate B_X(k): total volume of buy orders at prices >= k
        """
        return np.sum(self.X_plus[k:])
    
    def get_S_X(self, k):
        """
        Calculate S_X(k): total volume of sell orders at prices <= k
        """
        return np.sum(self.X_minus[:k+1])
    
    def get_g_X(self, k):
        """
        Calculate g_X(k): S_X(k) - B_X(k)
        """
        return self.get_S_X(k) - self.get_B_X(k)
    
    def place_limit_buy_order(self, price_level, size):
        """
        Place a limit buy order at the specified price level.
        
        Args:
            price_level: Index representing the price level (0 to d)
            size: Number of shares to buy
        """
        # Create intermediate state after receiving order
        X_plus_new = self.X_plus.copy()
        X_plus_new[price_level] += size
        X_minus_new = self.X_minus.copy()
        
        # Apply market clearing operator
        self.X_plus, self.X_minus = self.clear_market(X_plus_new, X_minus_new)
        self._update_bid_ask()
    
    def place_limit_sell_order(self, price_level, size):
        """
        Place a limit sell order at the specified price level.
        
        Args:
            price_level: Index representing the price level (0 to d)
            size: Number of shares to sell
        """
        # Create intermediate state after receiving order
        X_plus_new = self.X_plus.copy()
        X_minus_new = self.X_minus.copy()
        X_minus_new[price_level] += size
        
        # Apply market clearing operator
        self.X_plus, self.X_minus = self.clear_market(X_plus_new, X_minus_new)
        self._update_bid_ask()
    
    def cancel_buy_order(self, price_level, size):
        """
        Cancel a buy order at the specified price level.
        
        Args:
            price_level: Index representing the price level (0 to d)
            size: Number of shares to cancel
        """
        if price_level < 0 or price_level > self.d:
            return
            
        # Create intermediate state after cancellation
        X_plus_new = self.X_plus.copy()
        X_minus_new = self.X_minus.copy()
        
        # Ensure we don't cancel more than what's available
        size = min(size, X_plus_new[price_level])
        X_plus_new[price_level] -= size
        
        # Apply market clearing operator
        self.X_plus, self.X_minus = self.clear_market(X_plus_new, X_minus_new)
        self._update_bid_ask()
    
    def cancel_sell_order(self, price_level, size):
        """
        Cancel a sell order at the specified price level.
        
        Args:
            price_level: Index representing the price level (0 to d)
            size: Number of shares to cancel
        """
        if price_level < 0 or price_level > self.d:
            return
            
        # Create intermediate state after cancellation
        X_plus_new = self.X_plus.copy()
        X_minus_new = self.X_minus.copy()
        
        # Ensure we don't cancel more than what's available
        size = min(size, X_minus_new[price_level])
        X_minus_new[price_level] -= size
        
        # Apply market clearing operator
        self.X_plus, self.X_minus = self.clear_market(X_plus_new, X_minus_new)
        self._update_bid_ask()
    
    def place_market_buy_order(self, size):
        """
        Place a market buy order (immediately executed at best available price).
        
        Args:
            size: Number of shares to buy
        """
        # Market buy order is equivalent to a limit buy order at a very high price
        self.place_limit_buy_order(self.d, size)
    
    def place_market_sell_order(self, size):
        """
        Place a market sell order (immediately executed at best available price).
        
        Args:
            size: Number of shares to sell
        """
        # Market sell order is equivalent to a limit sell order at a very low price
        self.place_limit_sell_order(0, size)
    
    def clear_market(self, X_plus, X_minus):
        """
        Market clearing operator as defined in the paper.
        
        Args:
            X_plus: Buy side of the order book
            X_minus: Sell side of the order book
        
        Returns:
            Tuple of (cleared buy side, cleared sell side)
        """
        # Create a copy of the input states
        X = (X_plus.copy(), X_minus.copy())
        
        # Calculate g_X(k) for all k
        g_X = np.zeros(self.d + 1)
        for k in range(self.d + 1):
            S_X_k = np.sum(X_minus[:k+1])
            B_X_k = np.sum(X_plus[k:])
            g_X[k] = S_X_k - B_X_k
        
        # Find p_B(X) and p_A(X) as defined in the paper
        p_B = np.argmax(g_X < 0) - 1 if np.any(g_X < 0) else self.d
        p_A = np.argmin(g_X > 0) if np.any(g_X > 0) else 0
        
        # Calculate B_X and S_X for each price level
        B_X = np.zeros(self.d + 1)
        S_X = np.zeros(self.d + 1)
        for k in range(self.d + 1):
            B_X[k] = np.sum(X_plus[k:])
            S_X[k] = np.sum(X_minus[:k+1])
        
        # Apply the clearing operator for buy side
        C_X_plus = np.zeros_like(X_plus)
        if p_B >= 0:
            if g_X[p_B] <= -X_plus[p_B]:
                # First case in Eq. (2.13)
                C_X_plus[:p_B+1] = X_plus[:p_B+1]
            else:
                # Second case in Eq. (2.13)
                C_X_plus[:p_B] = X_plus[:p_B]
                C_X_plus[p_B] = X_plus[p_B] + min(0, g_X[p_B])
        
        # Apply the clearing operator for sell side
        C_X_minus = np.zeros_like(X_minus)
        if p_A <= self.d:
            if g_X[p_A] >= X_minus[p_A]:
                # First case in Eq. (2.14)
                C_X_minus[p_A:] = X_minus[p_A:]
            else:
                # Second case in Eq. (2.14)
                C_X_minus[p_A+1:] = X_minus[p_A+1:]
                C_X_minus[p_A] = X_minus[p_A] + max(0, g_X[p_A])
        
        return C_X_plus, C_X_minus
    
    def simulate_order_flow(self, duration, lambda_plus, lambda_minus, 
                           cancel_plus, cancel_minus, seed=None):
        """
        Simulate order flow using Poisson processes as described in the paper.
        
        Args:
            duration: Time duration for simulation
            lambda_plus: Rate function for buy limit orders
            lambda_minus: Rate function for sell limit orders
            cancel_plus: Rate function for buy cancellations
            cancel_minus: Rate function for sell cancellations
            seed: Random seed for reproducibility
        """
        if seed is not None:
            np.random.seed(seed)
            
        # Current simulation time
        current_time = self.time_history[-1]
        end_time = current_time + duration
        
        # Simulation loop
        while current_time < end_time:
            # Calculate total event rate
            total_rate = 0
            
            # Buy limit order rates at each price level
            buy_rates = np.zeros(self.d + 1)
            for i in range(self.d + 1):
                buy_rates[i] = lambda_plus(i, self)
            total_rate += np.sum(buy_rates)
            
            # Sell limit order rates at each price level
            sell_rates = np.zeros(self.d + 1)
            for i in range(self.d + 1):
                sell_rates[i] = lambda_minus(i, self)
            total_rate += np.sum(sell_rates)
            
            # Buy cancellation rates at each price level
            cancel_buy_rates = np.zeros(self.d + 1)
            for i in range(self.d + 1):
                if self.X_plus[i] > 0:
                    cancel_buy_rates[i] = cancel_plus(i, self)
            total_rate += np.sum(cancel_buy_rates)
            
            # Sell cancellation rates at each price level
            cancel_sell_rates = np.zeros(self.d + 1)
            for i in range(self.d + 1):
                if self.X_minus[i] > 0:
                    cancel_sell_rates[i] = cancel_minus(i, self)
            total_rate += np.sum(cancel_sell_rates)
            
            # Sample time to next event (exponential distribution)
            if total_rate > 0:
                dt = np.random.exponential(1.0 / total_rate)
            else:
                dt = end_time - current_time
                
            # Update current time
            current_time += dt
            
            if current_time < end_time:
                # Sample event type
                r = np.random.uniform(0, total_rate)
                
                # Process event types
                if r < np.sum(buy_rates):
                    # Buy limit order
                    price_level = np.searchsorted(np.cumsum(buy_rates), r)
                    size = 1  # Fixed size for simplicity, could be sampled
                    self.place_limit_buy_order(price_level, size)
                    event_type = "buy_limit"
                    
                elif r < np.sum(buy_rates) + np.sum(sell_rates):
                    # Sell limit order
                    r -= np.sum(buy_rates)
                    price_level = np.searchsorted(np.cumsum(sell_rates), r)
                    size = 1  # Fixed size for simplicity
                    self.place_limit_sell_order(price_level, size)
                    event_type = "sell_limit"
                    
                elif r < np.sum(buy_rates) + np.sum(sell_rates) + np.sum(cancel_buy_rates):
                    # Cancel buy order
                    r -= (np.sum(buy_rates) + np.sum(sell_rates))
                    price_level = np.searchsorted(np.cumsum(cancel_buy_rates), r)
                    size = 1  # Fixed size for simplicity
                    self.cancel_buy_order(price_level, size)
                    event_type = "cancel_buy"
                    
                else:
                    # Cancel sell order
                    r -= (np.sum(buy_rates) + np.sum(sell_rates) + np.sum(cancel_buy_rates))
                    price_level = np.searchsorted(np.cumsum(cancel_sell_rates), r)
                    size = 1  # Fixed size for simplicity
                    self.cancel_sell_order(price_level, size)
                    event_type = "cancel_sell"
                
                # Update mid price based on bid and ask
                self.mid_price = (self.bid_price + self.ask_price) / 2
                
                # Update history
                self.time_history.append(current_time)
                self.price_history.append(self.mid_price)
                self.save_state()
    
    def plot_order_book(self, timestamp=None):
        """
        Plot the current state of the order book.
        
        Args:
            timestamp: If provided, plot the order book at this timestamp
        """
        if timestamp is not None:
            # Find the closest timestamp in history
            idx = np.argmin(np.abs(np.array(self.time_history) - timestamp))
            state = self.book_history[idx]
            X_plus = state['X_plus']
            X_minus = state['X_minus']
            mid_price = state['mid_price']
            bid_price = state['bid_price']
            ask_price = state['ask_price']
        else:
            X_plus = self.X_plus
            X_minus = self.X_minus
            mid_price = self.mid_price
            bid_price = self.bid_price
            ask_price = self.ask_price
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Calculate price levels
        price_levels = np.arange(mid_price - self.d * self.tick_size, 
                                 mid_price + (self.d+1) * self.tick_size, 
                                 self.tick_size)
        
        # Plot buy orders
        buy_volumes = np.zeros(len(price_levels))
        buy_volumes[:self.d+1] = X_plus[::-1]  # Reverse to match price levels
        ax.bar(price_levels[:self.d+1], buy_volumes[:self.d+1], width=self.tick_size*0.8, 
               color='blue', alpha=0.5, label='Buy Orders')
        
        # Plot sell orders
        sell_volumes = np.zeros(len(price_levels))
        sell_volumes[self.d:] = X_minus  # Offset to match price levels
        ax.bar(price_levels[self.d:], sell_volumes[self.d:], width=self.tick_size*0.8, 
               color='red', alpha=0.5, label='Sell Orders')
        
        # Plot mid, bid, and ask prices
        ax.axvline(x=mid_price, color='black', linestyle='-', label='Mid Price')
        ax.axvline(x=bid_price, color='blue', linestyle='--', label='Bid Price')
        ax.axvline(x=ask_price, color='red', linestyle='--', label='Ask Price')
        
        # Set labels and title
        ax.set_xlabel('Price')
        ax.set_ylabel('Volume')
        ax.set_title('Limit Order Book State')
        ax.legend()
        
        # Format x-axis to show prices
        ax.xaxis.set_major_locator(MaxNLocator(10))
        
        return fig, ax
    
    def plot_price_history(self):
        """Plot the price history of the simulation."""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot mid price history
        ax.plot(self.time_history, self.price_history, 'k-', label='Mid Price')
        
        # Extract bid and ask prices from history
        bid_prices = [state['bid_price'] for state in self.book_history]
        ask_prices = [state['ask_price'] for state in self.book_history]
        
        # Plot bid and ask price histories
        ax.plot(self.time_history, bid_prices, 'b--', label='Bid Price')
        ax.plot(self.time_history, ask_prices, 'r--', label='Ask Price')
        
        # Set labels and title
        ax.set_xlabel('Time')
        ax.set_ylabel('Price')
        ax.set_title('Price History')
        ax.legend()
        
        return fig, ax

# Define custom rate functions for different order types
def lambda_plus(i, book):
    """Rate function for buy limit orders at price level i."""
    # Higher rate closer to the bid price
    bid_idx = book.b()
    if i <= bid_idx:
        return 1.5  # Rate for orders at or below bid
    else:
        return 1.5 * np.exp(-0.5 * (i - bid_idx))  # Decreasing rate away from bid

def lambda_minus(i, book):
    """Rate function for sell limit orders at price level i."""
    # Higher rate closer to the ask price
    ask_idx = book.a()
    if i >= ask_idx:
        return 1.5  # Rate for orders at or above ask
    else:
        return 1.5 * np.exp(-0.5 * (ask_idx - i))  # Decreasing rate away from ask

def cancel_plus(i, book):
    """Rate function for canceling buy orders at price level i."""
    # Cancellation rate proportional to volume at that level
    return 0.5 * book.X_plus[i]

def cancel_minus(i, book):
    """Rate function for canceling sell orders at price level i."""
    # Cancellation rate proportional to volume at that level
    return 0.5 * book.X_minus[i]

# Function to create a model example for demonstration
def model_example_1():
    """Create and run a simulation example using Model 1 from the paper."""
    # Initialize a limit order book
    book = LimitOrderBook(d=10, initial_mid_price=100.0, tick_size=0.1)
    
    # Run simulation for a specified duration
    book.simulate_order_flow(
        duration=100,
        lambda_plus=lambda_plus, 
        lambda_minus=lambda_minus,
        cancel_plus=cancel_plus,
        cancel_minus=cancel_minus,
        seed=42
    )
    
    # Plot the final state of the order book
    fig1, ax1 = book.plot_order_book()
    plt.tight_layout()
    plt.savefig('order_book_final_state.png')
    
    # Plot the price history
    fig2, ax2 = book.plot_price_history()
    plt.tight_layout()
    plt.savefig('price_history.png')
    
    # Plot order book at different time points
    times = [0, 25, 50, 75, 100]
    for t in times:
        fig, ax = book.plot_order_book(timestamp=t)
        plt.tight_layout()
        plt.savefig(f'order_book_time_{t}.png')
        plt.close(fig)
    
    return book

# Function to analyze the model
def analyze_model(book):
    """Analyze various aspects of the model simulation."""
    # Calculate spread over time
    spreads = [state['ask_price'] - state['bid_price'] for state in book.book_history]
    
    # Plot spread over time
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(book.time_history, spreads)
    ax.set_xlabel('Time')
    ax.set_ylabel('Spread')
    ax.set_title('Bid-Ask Spread Over Time')
    plt.tight_layout()
    plt.savefig('spread_over_time.png')
    
    # Calculate order imbalance (buy volume - sell volume)
    buy_volumes = [np.sum(state['X_plus']) for state in book.book_history]
    sell_volumes = [np.sum(state['X_minus']) for state in book.book_history]
    imbalances = [b - s for b, s in zip(buy_volumes, sell_volumes)]
    
    # Plot order imbalance over time
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(book.time_history, imbalances)
    ax.set_xlabel('Time')
    ax.set_ylabel('Order Imbalance')
    ax.set_title('Order Imbalance (Buy - Sell) Over Time')
    plt.tight_layout()
    plt.savefig('order_imbalance.png')
    
    # Calculate price changes
    price_changes = np.diff(book.price_history)
    
    # Plot histogram of price changes
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.hist(price_changes, bins=30, alpha=0.7)
    ax.set_xlabel('Price Change')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Price Changes')
    plt.tight_layout()
    plt.savefig('price_change_distribution.png')
    
    return {
        'mean_spread': np.mean(spreads),
        'mean_imbalance': np.mean(imbalances),
        'price_volatility': np.std(price_changes)
    }

# Function to compare model variants
def compare_models():
    """Compare different model variants."""
    # Model 1: Original model
    book1 = LimitOrderBook(d=10, initial_mid_price=100.0, tick_size=0.1)
    book1.simulate_order_flow(
        duration=100,
        lambda_plus=lambda_plus, 
        lambda_minus=lambda_minus,
        cancel_plus=cancel_plus,
        cancel_minus=cancel_minus,
        seed=42
    )
    
    # Model 2: Higher order arrival rates
    def lambda_plus_high(i, book):
        bid_idx = book.b()
        if i <= bid_idx:
            return 3.0
        else:
            return 3.0 * np.exp(-0.5 * (i - bid_idx))
    
    def lambda_minus_high(i, book):
        ask_idx = book.a()
        if i >= ask_idx:
            return 3.0
        else:
            return 3.0 * np.exp(-0.5 * (ask_idx - i))
    
    book2 = LimitOrderBook(d=10, initial_mid_price=100.0, tick_size=0.1)
    book2.simulate_order_flow(
        duration=100,
        lambda_plus=lambda_plus_high, 
        lambda_minus=lambda_minus_high,
        cancel_plus=cancel_plus,
        cancel_minus=cancel_minus,
        seed=42
    )
    
    # Model 3: Higher cancellation rates
    def cancel_plus_high(i, book):
        return 1.0 * book.X_plus[i]
    
    def cancel_minus_high(i, book):
        return 1.0 * book.X_minus[i]
    
    book3 = LimitOrderBook(d=10, initial_mid_price=100.0, tick_size=0.1)
    book3.simulate_order_flow(
        duration=100,
        lambda_plus=lambda_plus, 
        lambda_minus=lambda_minus,
        cancel_plus=cancel_plus_high,
        cancel_minus=cancel_minus_high,
        seed=42
    )
    
    # Compare price histories
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(book1.time_history, book1.price_history, 'b-', label='Model 1: Base')
    ax.plot(book2.time_history, book2.price_history, 'r-', label='Model 2: High Order Rate')
    ax.plot(book3.time_history, book3.price_history, 'g-', label='Model 3: High Cancel Rate')
    ax.set_xlabel('Time')
    ax.set_ylabel('Mid Price')
    ax.set_title('Price Comparison Across Models')
    ax.legend()
    plt.tight_layout()
    plt.savefig('model_price_comparison.png')
    
    # Compare spreads
    spreads1 = [state['ask_price'] - state['bid_price'] for state in book1.book_history]
    spreads2 = [state['ask_price'] - state['bid_price'] for state in book2.book_history]
    spreads3 = [state['ask_price'] - state['bid_price'] for state in book3.book_history]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(book1.time_history, spreads1, 'b-', label='Model 1: Base')
    ax.plot(book2.time_history, spreads2, 'r-', label='Model 2: High Order Rate')
    ax.plot(book3.time_history, spreads3, 'g-', label='Model 3: High Cancel Rate')
    ax.set_xlabel('Time')
    ax.set_ylabel('Spread')
    ax.set_title('Spread Comparison Across Models')
    ax.legend()
    plt.tight_layout()
    plt.savefig('model_spread_comparison.png')
    
    # Compare volumes
    volumes1 = [np.sum(state['X_plus']) + np.sum(state['X_minus']) for state in book1.book_history]
    volumes2 = [np.sum(state['X_plus']) + np.sum(state['X_minus']) for state in book2.book_history]
    volumes3 = [np.sum(state['X_plus']) + np.sum(state['X_minus']) for state in book3.book_history]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(book1.time_history, volumes1, 'b-', label='Model 1: Base')
    ax.plot(book2.time_history, volumes2, 'r-', label='Model 2: High Order Rate')
    ax.plot(book3.time_history, volumes3, 'g-', label='Model 3: High Cancel Rate')
    ax.set_xlabel('Time')
    ax.set_ylabel('Total Volume')
    ax.set_title('Volume Comparison Across Models')
    ax.legend()
    plt.tight_layout()
    plt.savefig('model_volume_comparison.png')
    
    # Compute summary statistics
    metrics1 = analyze_model(book1)
    metrics2 = analyze_model(book2)
    metrics3 = analyze_model(book3)
    
    print("Model 1 (Base):")
    print(f"  Mean Spread: {metrics1['mean_spread']:.4f}")
    print(f"  Mean Imbalance: {metrics1['mean_imbalance']:.4f}")
    print(f"  Price Volatility: {metrics1['price_volatility']:.4f}")
    
    print("Model 2 (High Order Rate):")
    print(f"  Mean Spread: {metrics2['mean_spread']:.4f}")
    print(f"  Mean Imbalance: {metrics2['mean_imbalance']:.4f}")
    print(f"  Price Volatility: {metrics2['price_volatility']:.4f}")
    
    print("Model 3 (High Cancel Rate):")
    print(f"  Mean Spread: {metrics3['mean_spread']:.4f}")
    print(f"  Mean Imbalance: {metrics3['mean_imbalance']:.4f}")
    print(f"  Price Volatility: {metrics3['price_volatility']:.4f}")
    
    return book1, book2, book3

# Function to run a specific demonstration of the market clearing mechanism
def demonstrate_clearing():
    """Demonstrate the market clearing mechanism specifically."""
    # Create an order book with a specific initial state
    book = LimitOrderBook(d=10, initial_mid_price=100.0, tick_size=0.1)
    
    # Clear the initial state
    book.X_plus = np.zeros(11, dtype=int)
    book.X_minus = np.zeros(11, dtype=int)
    
    # Set up a specific configuration
    book.X_plus[5:8] = [3, 2, 1]  # Buy orders at price levels 5, 6, 7
    book.X_minus[4:7] = [1, 2, 3]  # Sell orders at price levels 4, 5, 6
    book._update_bid_ask()
    
    # Plot the initial state
    fig, ax = book.plot_order_book()
    plt.title('Initial State Before Clearing')
    plt.tight_layout()
    plt.savefig('clearing_initial_state.png')
    plt.close(fig)
    
    # Now place a large buy order that will cross the book
    print("Placing a buy limit order at price level 6 with size 4")
    book.place_limit_buy_order(6, 4)
    
    # Plot the final state after clearing
    fig, ax = book.plot_order_book()
    plt.title('Final State After Clearing')
    plt.tight_layout()
    plt.savefig('clearing_final_state.png')
    plt.close(fig)
    
    # Now demonstrate a market sell order
    print("\nPlacing a market sell order with size 2")
    book.place_market_sell_order(2)
    
    # Plot the state after market order
    fig, ax = book.plot_order_book()
    plt.title('State After Market Sell Order')
    plt.tight_layout()
    plt.savefig('clearing_after_market_order.png')
    plt.close(fig)
    
    # Print explanation of what happened
    print("\nExplanation of market clearing process:")
    print("1. Initial state had buy orders at price levels 5-7 and sell orders at 4-6")
    print("2. Placing a buy limit order at level 6 with size 4 created an overlap in the book")
    print("3. The clearing mechanism matched orders to maximize volume")
    print("4. After clearing, we see the buy and sell sides no longer overlap")
    print("5. The market sell order was then executed against the highest-priced buy orders")
    
    return book

# Main execution
if __name__ == "__main__":
    print("Running Order Book Simulation Framework...")
    
    # Run the basic model example
    print("\n===== Basic Model Example =====")
    book = model_example_1()
    
    # Analyze the model
    print("\n===== Model Analysis =====")
    metrics = analyze_model(book)
    print(f"Mean Spread: {metrics['mean_spread']:.4f}")
    print(f"Mean Order Imbalance: {metrics['mean_imbalance']:.4f}")
    print(f"Price Volatility: {metrics['price_volatility']:.4f}")
    
    # Compare different model variants
    print("\n===== Model Comparison =====")
    book1, book2, book3 = compare_models()
    
    # Demonstrate clearing mechanism
    print("\n===== Demonstration of Clearing Mechanism =====")
    clearing_book = demonstrate_clearing()
    
    print("\nSimulation complete. All figures saved.")