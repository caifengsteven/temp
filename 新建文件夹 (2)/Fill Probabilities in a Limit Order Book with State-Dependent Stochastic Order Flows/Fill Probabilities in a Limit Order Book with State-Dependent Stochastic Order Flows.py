import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import expon
from tqdm import tqdm

class LimitOrderBookSimulation:
    """
    A simplified simulation of a limit order book based on the model in the paper.
    """
    def __init__(self, 
                 initial_bid_quantity=5, 
                 initial_ask_quantity=5, 
                 initial_spread=2,
                 price_grid_size=20,
                 initial_mid_price=100):
        """
        Initialize the limit order book simulation.
        
        Parameters:
        -----------
        initial_bid_quantity : int
            Initial number of orders at the best bid
        initial_ask_quantity : int
            Initial number of orders at the best ask
        initial_spread : int
            Initial spread in ticks
        price_grid_size : int
            Size of the price grid
        initial_mid_price : float
            Initial mid price
        """
        self.price_grid_size = price_grid_size
        self.initial_mid_price = initial_mid_price
        
        # Initialize the order book
        self.order_book = np.zeros(price_grid_size)
        
        # Calculate initial best bid and ask positions
        self.best_bid_pos = price_grid_size // 2 - initial_spread // 2
        self.best_ask_pos = self.best_bid_pos + initial_spread
        
        # Set initial quantities
        self.order_book[self.best_bid_pos] = -initial_bid_quantity  # Negative for bid orders
        self.order_book[self.best_ask_pos] = initial_ask_quantity   # Positive for ask orders
        
        # Initialize prices
        self.tick_size = 0.01
        self.prices = np.array([initial_mid_price + (i - price_grid_size // 2) * self.tick_size 
                                for i in range(price_grid_size)])
        
        # Initialize state-dependent rates
        self.setup_rates()
        
        # Initialize history
        self.history = []
        self.record_state()
        
    def setup_rates(self):
        """
        Set up state-dependent rates for order arrivals and cancellations.
        These rates are simplified from Model III in the paper.
        """
        # Base rates
        self.lambda_base = 1.0  # Base rate for limit order arrivals
        self.mu_base = 0.5      # Base rate for market order arrivals
        self.theta_base = 0.2   # Base rate for cancellations
        
        # Spread dependency factors
        self.lambda_spread_factor = 0.8  # Decreasing rate with increasing spread
        self.mu_spread_factor = 0.7      # Decreasing rate with increasing spread
        self.theta_spread_factor = 0.9   # Decreasing rate with increasing spread
        
        # Distance dependency factors
        self.lambda_distance_factor = 0.5  # Decreasing rate with increasing distance
        
    def get_lambda(self, distance, spread):
        """
        Get the arrival rate of limit orders based on distance from opposite best quote and spread.
        
        Parameters:
        -----------
        distance : int
            Distance in ticks from the opposite best quote
        spread : int
            Current spread in ticks
            
        Returns:
        --------
        float
            The arrival rate of limit orders
        """
        return self.lambda_base * (self.lambda_spread_factor ** (spread - 1)) * (self.lambda_distance_factor ** (distance - 1))
    
    def get_mu(self, spread):
        """
        Get the arrival rate of market orders based on spread.
        
        Parameters:
        -----------
        spread : int
            Current spread in ticks
            
        Returns:
        --------
        float
            The arrival rate of market orders
        """
        return self.mu_base * (self.mu_spread_factor ** (spread - 1))
    
    def get_theta(self, distance, spread, quantity):
        """
        Get the cancellation rate based on distance from opposite best quote, spread, and quantity.
        
        Parameters:
        -----------
        distance : int
            Distance in ticks from the opposite best quote
        spread : int
            Current spread in ticks
        quantity : int
            Absolute number of orders at the price level
            
        Returns:
        --------
        float
            The cancellation rate
        """
        return self.theta_base * (self.theta_spread_factor ** (spread - 1)) * abs(quantity)
    
    def get_spread(self):
        """
        Get the current spread in ticks.
        
        Returns:
        --------
        int
            The current spread in ticks
        """
        return self.best_ask_pos - self.best_bid_pos
    
    def get_mid_price(self):
        """
        Get the current mid price.
        
        Returns:
        --------
        float
            The current mid price
        """
        return (self.prices[self.best_bid_pos] + self.prices[self.best_ask_pos]) / 2
    
    def record_state(self):
        """
        Record the current state of the order book.
        """
        state = {
            'time': len(self.history),
            'best_bid_pos': self.best_bid_pos,
            'best_ask_pos': self.best_ask_pos,
            'best_bid_quantity': abs(self.order_book[self.best_bid_pos]),
            'best_ask_quantity': self.order_book[self.best_ask_pos],
            'best_bid_price': self.prices[self.best_bid_pos],
            'best_ask_price': self.prices[self.best_ask_pos],
            'mid_price': self.get_mid_price(),
            'spread': self.get_spread(),
            'order_book': self.order_book.copy()
        }
        self.history.append(state)
    
    def update_best_quotes(self):
        """
        Update the best bid and ask positions.
        """
        # Find the best bid position (highest price with negative quantity)
        bid_positions = np.where(self.order_book < 0)[0]
        if len(bid_positions) > 0:
            self.best_bid_pos = np.max(bid_positions)
        else:
            # If no bid orders, set to a default position
            self.best_bid_pos = 0
        
        # Find the best ask position (lowest price with positive quantity)
        ask_positions = np.where(self.order_book > 0)[0]
        if len(ask_positions) > 0:
            self.best_ask_pos = np.min(ask_positions)
        else:
            # If no ask orders, set to a default position
            self.best_ask_pos = self.price_grid_size - 1
    
    def simulate_step(self):
        """
        Simulate one step of the order book dynamics.
        """
        current_spread = self.get_spread()
        
        # Compute rates for all possible events
        rates = []
        events = []
        
        # 1. Limit order arrivals
        for i in range(self.price_grid_size):
            # Bid limit orders (can be placed below the best ask)
            if i < self.best_ask_pos:
                distance = self.best_ask_pos - i
                rate = self.get_lambda(distance, current_spread)
                rates.append(rate)
                events.append(('limit_bid', i))
            
            # Ask limit orders (can be placed above the best bid)
            if i > self.best_bid_pos:
                distance = i - self.best_bid_pos
                rate = self.get_lambda(distance, current_spread)
                rates.append(rate)
                events.append(('limit_ask', i))
        
        # 2. Market order arrivals (only at the best quotes)
        market_rate = self.get_mu(current_spread)
        if self.order_book[self.best_bid_pos] < 0:  # If there are bid orders
            rates.append(market_rate)
            events.append(('market_sell', self.best_bid_pos))
        
        if self.order_book[self.best_ask_pos] > 0:  # If there are ask orders
            rates.append(market_rate)
            events.append(('market_buy', self.best_ask_pos))
        
        # 3. Cancellations
        for i in range(self.price_grid_size):
            if self.order_book[i] < 0:  # Bid orders
                distance = self.best_ask_pos - i
                rate = self.get_theta(distance, current_spread, self.order_book[i])
                rates.append(rate)
                events.append(('cancel_bid', i))
            elif self.order_book[i] > 0:  # Ask orders
                distance = i - self.best_bid_pos
                rate = self.get_theta(distance, current_spread, self.order_book[i])
                rates.append(rate)
                events.append(('cancel_ask', i))
        
        # Convert rates to probabilities
        total_rate = sum(rates)
        probs = [r / total_rate for r in rates]
        
        # Sample an event
        event_idx = np.random.choice(len(events), p=probs)
        event_type, position = events[event_idx]
        
        # Update the order book based on the event
        if event_type == 'limit_bid':
            self.order_book[position] -= 1  # Add a bid order
        elif event_type == 'limit_ask':
            self.order_book[position] += 1  # Add an ask order
        elif event_type == 'market_sell':
            self.order_book[position] += 1  # Remove a bid order (market sell)
        elif event_type == 'market_buy':
            self.order_book[position] -= 1  # Remove an ask order (market buy)
        elif event_type == 'cancel_bid':
            self.order_book[position] += 1  # Cancel a bid order
        elif event_type == 'cancel_ask':
            self.order_book[position] -= 1  # Cancel an ask order
        
        # Update best quotes
        self.update_best_quotes()
        
        # Record the state
        self.record_state()
        
        return event_type, position
    
    def simulate(self, n_steps):
        """
        Simulate the order book for n steps.
        
        Parameters:
        -----------
        n_steps : int
            Number of steps to simulate
        """
        for _ in tqdm(range(n_steps)):
            self.simulate_step()
    
    def plot_order_book(self, step=-1):
        """
        Plot the order book at a given step.
        
        Parameters:
        -----------
        step : int
            Step to plot. Default is the last step.
        """
        if step < 0 or step >= len(self.history):
            step = len(self.history) - 1
        
        state = self.history[step]
        order_book = state['order_book']
        
        plt.figure(figsize=(12, 6))
        
        # Plot the bid side (negative quantities)
        bid_positions = np.where(order_book < 0)[0]
        bid_quantities = -order_book[bid_positions]  # Convert to positive
        bid_prices = self.prices[bid_positions]
        
        # Plot the ask side (positive quantities)
        ask_positions = np.where(order_book > 0)[0]
        ask_quantities = order_book[ask_positions]
        ask_prices = self.prices[ask_positions]
        
        plt.bar(bid_prices, bid_quantities, width=0.005, color='g', alpha=0.7, label='Bid')
        plt.bar(ask_prices, ask_quantities, width=0.005, color='r', alpha=0.7, label='Ask')
        
        plt.axvline(x=state['best_bid_price'], color='g', linestyle='--', label='Best Bid')
        plt.axvline(x=state['best_ask_price'], color='r', linestyle='--', label='Best Ask')
        plt.axvline(x=state['mid_price'], color='b', linestyle='-', label='Mid Price')
        
        plt.xlabel('Price')
        plt.ylabel('Quantity')
        plt.title(f'Order Book (Step {step}, Mid Price: {state["mid_price"]:.4f}, Spread: {state["spread"]})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def plot_mid_price_history(self):
        """
        Plot the history of the mid price.
        """
        times = [state['time'] for state in self.history]
        mid_prices = [state['mid_price'] for state in self.history]
        
        plt.figure(figsize=(12, 6))
        plt.plot(times, mid_prices)
        plt.xlabel('Time Step')
        plt.ylabel('Mid Price')
        plt.title('Mid Price History')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def plot_spread_history(self):
        """
        Plot the history of the spread.
        """
        times = [state['time'] for state in self.history]
        spreads = [state['spread'] for state in self.history]
        
        plt.figure(figsize=(12, 6))
        plt.plot(times, spreads)
        plt.xlabel('Time Step')
        plt.ylabel('Spread (ticks)')
        plt.title('Spread History')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def compute_mid_price_movement_probability(self, num_simulations=1000):
        """
        Compute the probability of an increase in mid-price based on simulations.
        
        Parameters:
        -----------
        num_simulations : int
            Number of simulations to run
            
        Returns:
        --------
        float
            The probability of an increase in mid-price
        """
        increase_count = 0
        decrease_count = 0
        
        initial_state = self.order_book.copy()
        initial_best_bid_pos = self.best_bid_pos
        initial_best_ask_pos = self.best_ask_pos
        initial_mid_price = self.get_mid_price()
        
        for _ in range(num_simulations):
            # Reset to initial state
            self.order_book = initial_state.copy()
            self.best_bid_pos = initial_best_bid_pos
            self.best_ask_pos = initial_best_ask_pos
            
            # Simulate until mid-price changes
            while True:
                event_type, position = self.simulate_step()
                current_mid_price = self.get_mid_price()
                
                if current_mid_price > initial_mid_price:
                    increase_count += 1
                    break
                elif current_mid_price < initial_mid_price:
                    decrease_count += 1
                    break
        
        # Restore the initial state
        self.order_book = initial_state.copy()
        self.best_bid_pos = initial_best_bid_pos
        self.best_ask_pos = initial_best_ask_pos
        self.history = self.history[:1]  # Keep only the initial state
        
        # Compute probability
        if increase_count + decrease_count > 0:
            increase_probability = increase_count / (increase_count + decrease_count)
            return increase_probability
        else:
            return None
    
    def compute_fill_probability(self, side='bid', level=0, num_simulations=1000):
        """
        Compute the fill probability of an order.
        
        Parameters:
        -----------
        side : str
            'bid' or 'ask'
        level : int
            0 for best quote, 1 for one level deeper, etc.
        num_simulations : int
            Number of simulations to run
            
        Returns:
        --------
        float
            The fill probability
        """
        fill_count = 0
        
        initial_state = self.order_book.copy()
        initial_best_bid_pos = self.best_bid_pos
        initial_best_ask_pos = self.best_ask_pos
        initial_mid_price = self.get_mid_price()
        
        # Determine the position of the order
        if side == 'bid':
            if level == 0:
                order_position = self.best_bid_pos
            else:
                order_position = self.best_bid_pos - level
        else:  # 'ask'
            if level == 0:
                order_position = self.best_ask_pos
            else:
                order_position = self.best_ask_pos + level
        
        # Check if the position is valid
        if order_position < 0 or order_position >= self.price_grid_size:
            return None
        
        # Add a tag to the order to track if it gets filled
        tag_quantity = 1
        if side == 'bid':
            # For bids, we use a small negative quantity to distinguish from other orders
            self.order_book[order_position] -= tag_quantity
        else:
            # For asks, we use a small positive quantity
            self.order_book[order_position] += tag_quantity
        
        for _ in range(num_simulations):
            # Reset to initial state with the tagged order
            self.order_book = initial_state.copy()
            self.best_bid_pos = initial_best_bid_pos
            self.best_ask_pos = initial_best_ask_pos
            
            # Simulate until either the order is filled or the mid-price changes
            while True:
                event_type, position = self.simulate_step()
                current_mid_price = self.get_mid_price()
                
                # Check if the order is filled (position matches and right event type)
                if position == order_position:
                    if (side == 'bid' and event_type == 'market_sell') or \
                       (side == 'ask' and event_type == 'market_buy'):
                        fill_count += 1
                        break
                
                # Check if mid-price has changed
                if current_mid_price != initial_mid_price:
                    break
        
        # Restore the initial state without the tagged order
        self.order_book = initial_state.copy()
        self.best_bid_pos = initial_best_bid_pos
        self.best_ask_pos = initial_best_ask_pos
        self.history = self.history[:1]  # Keep only the initial state
        
        # Compute probability
        fill_probability = fill_count / num_simulations
        return fill_probability


# Test the simulation
np.random.seed(42)
lob = LimitOrderBookSimulation(
    initial_bid_quantity=5,
    initial_ask_quantity=5,
    initial_spread=2,
    price_grid_size=20,
    initial_mid_price=100
)

# Simulate the order book for 1000 steps
lob.simulate(1000)

# Plot the final state of the order book
lob.plot_order_book()

# Plot the history of the mid price
lob.plot_mid_price_history()

# Plot the history of the spread
lob.plot_spread_history()

# Compute mid-price movement probability
mid_price_up_prob = lob.compute_mid_price_movement_probability(num_simulations=500)
print(f"Probability of mid-price increase: {mid_price_up_prob:.4f}")

# Compute fill probabilities for different scenarios
print("\nFill Probabilities at Different Levels:")
print("---------------------------------------")

# 1. At the best bid (level 0)
fill_prob_best_bid = lob.compute_fill_probability(side='bid', level=0, num_simulations=500)
print(f"Fill probability at best bid: {fill_prob_best_bid:.4f}")

# 2. At one level below the best bid (level 1)
fill_prob_level1_bid = lob.compute_fill_probability(side='bid', level=1, num_simulations=500)
print(f"Fill probability at one level below best bid: {fill_prob_level1_bid:.4f}")

# 3. At the best ask (level 0)
fill_prob_best_ask = lob.compute_fill_probability(side='ask', level=0, num_simulations=500)
print(f"Fill probability at best ask: {fill_prob_best_ask:.4f}")

# 4. At one level above the best ask (level 1)
fill_prob_level1_ask = lob.compute_fill_probability(side='ask', level=1, num_simulations=500)
print(f"Fill probability at one level above best ask: {fill_prob_level1_ask:.4f}")

# Compare fill probabilities for different spread sizes
print("\nFill Probabilities for Different Spread Sizes:")
print("----------------------------------------------")

spreads = [1, 2, 3, 5]
for spread in spreads:
    # Create a new simulation with the specified spread
    lob_spread = LimitOrderBookSimulation(
        initial_bid_quantity=5,
        initial_ask_quantity=5,
        initial_spread=spread,
        price_grid_size=20,
        initial_mid_price=100
    )
    
    # Compute fill probability at best bid
    fill_prob = lob_spread.compute_fill_probability(side='bid', level=0, num_simulations=300)
    print(f"Spread = {spread}, Fill probability at best bid: {fill_prob:.4f}")

# Compare fill probabilities for different quantities at the best bid
print("\nFill Probabilities for Different Quantities at Best Bid:")
print("-------------------------------------------------------")

quantities = [1, 3, 5, 10]
for qty in quantities:
    # Create a new simulation with the specified quantity
    lob_qty = LimitOrderBookSimulation(
        initial_bid_quantity=qty,
        initial_ask_quantity=5,
        initial_spread=2,
        price_grid_size=20,
        initial_mid_price=100
    )
    
    # Compute fill probability at best bid
    fill_prob = lob_qty.compute_fill_probability(side='bid', level=0, num_simulations=300)
    print(f"Quantity = {qty}, Fill probability at best bid: {fill_prob:.4f}")

# Compare fill probabilities for different quantities at the best ask
print("\nFill Probabilities for Different Quantities at Best Ask:")
print("-------------------------------------------------------")

quantities = [1, 3, 5, 10]
for qty in quantities:
    # Create a new simulation with the specified quantity
    lob_qty = LimitOrderBookSimulation(
        initial_bid_quantity=5,
        initial_ask_quantity=qty,
        initial_spread=2,
        price_grid_size=20,
        initial_mid_price=100
    )
    
    # Compute fill probability at best ask
    fill_prob = lob_qty.compute_fill_probability(side='ask', level=0, num_simulations=300)
    print(f"Quantity = {qty}, Fill probability at best ask: {fill_prob:.4f}")