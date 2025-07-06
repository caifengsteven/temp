import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import deque, defaultdict
from tqdm import tqdm
from scipy.stats import expon, kstest
import time

# Set random seed for reproducibility
np.random.seed(42)

class LimitOrderBook:
    """
    A basic limit order book implementation that tracks bid and ask prices and volumes.
    """
    def __init__(self, tick_size=0.01, init_mid_price=100.0, max_levels=10):
        """
        Initialize the limit order book.
        
        Parameters:
        -----------
        tick_size : float
            The minimum price increment
        init_mid_price : float
            The initial mid price
        max_levels : int
            Maximum number of price levels to track on each side of the book
        """
        self.tick_size = tick_size
        self.max_levels = max_levels
        
        # Initialize with a reasonable spread
        self.best_bid_price = init_mid_price - tick_size
        self.best_ask_price = init_mid_price + tick_size
        
        # Initialize order book with empty dictionaries
        self.bids = {}  # price -> volume mapping for bids
        self.asks = {}  # price -> volume mapping for asks
        
        # Initialize with some liquidity at best bid and ask
        self.bids[self.best_bid_price] = 100
        self.asks[self.best_ask_price] = 100
        
        # Track historical data
        self.mid_prices = [self.get_mid_price()]
        self.spreads = [self.get_spread()]
        self.timestamps = [0]
        self.bid_volumes = [100]
        self.ask_volumes = [100]
        
        # Track order flow
        self.order_flow = []
        
    def get_mid_price(self):
        """Return the mid price."""
        return (self.best_bid_price + self.best_ask_price) / 2
    
    def get_spread(self):
        """Return the spread in ticks."""
        return (self.best_ask_price - self.best_bid_price) / self.tick_size
    
    def get_book_imbalance(self):
        """Return the order book imbalance."""
        total_bid_volume = sum(self.bids.values())
        total_ask_volume = sum(self.asks.values())
        
        if total_bid_volume + total_ask_volume > 0:
            return (total_bid_volume - total_ask_volume) / (total_bid_volume + total_ask_volume)
        else:
            return 0.0
    
    def limit_order(self, side, price, volume, timestamp):
        """
        Process a limit order.
        
        Parameters:
        -----------
        side : str
            'bid' or 'ask'
        price : float
            Price of the limit order
        volume : int
            Size of the limit order
        timestamp : float
            Current time
        """
        if side == 'bid':
            # If price is greater than or equal to best ask, it would cross the book
            if price >= self.best_ask_price:
                # This would be a marketable limit order, treat as market order
                self.market_order('bid', volume, timestamp)
            else:
                # Add to bid side
                if price in self.bids:
                    self.bids[price] += volume
                else:
                    self.bids[price] = volume
                
                # Update best bid if necessary
                if price > self.best_bid_price:
                    self.best_bid_price = price
        
        elif side == 'ask':
            # If price is less than or equal to best bid, it would cross the book
            if price <= self.best_bid_price:
                # This would be a marketable limit order, treat as market order
                self.market_order('ask', volume, timestamp)
            else:
                # Add to ask side
                if price in self.asks:
                    self.asks[price] += volume
                else:
                    self.asks[price] = volume
                
                # Update best ask if necessary
                if price < self.best_ask_price:
                    self.best_ask_price = price
        
        # Update historical data
        self.update_history(timestamp)
        # Record order flow
        self.order_flow.append(('limit', side, price, volume, timestamp))
    
    def market_order(self, side, volume, timestamp):
        """
        Process a market order.
        
        Parameters:
        -----------
        side : str
            'bid' or 'ask'
        volume : int
            Size of the market order
        timestamp : float
            Current time
        """
        remaining_volume = volume
        
        if side == 'bid':
            # Buy order matches with asks
            ask_prices = sorted(self.asks.keys())
            
            while remaining_volume > 0 and ask_prices:
                best_ask = ask_prices[0]
                matched_volume = min(remaining_volume, self.asks[best_ask])
                
                # Update the order book
                self.asks[best_ask] -= matched_volume
                remaining_volume -= matched_volume
                
                # If the level is depleted, remove it and update best ask
                if self.asks[best_ask] == 0:
                    del self.asks[best_ask]
                    ask_prices.pop(0)
                    if ask_prices:
                        self.best_ask_price = ask_prices[0]
                    else:
                        # No more asks, set to a high value
                        self.best_ask_price = self.best_bid_price + 10 * self.tick_size
                        # Add some liquidity back
                        new_ask_price = self.best_ask_price
                        self.asks[new_ask_price] = 100
        
        elif side == 'ask':
            # Sell order matches with bids
            bid_prices = sorted(self.bids.keys(), reverse=True)
            
            while remaining_volume > 0 and bid_prices:
                best_bid = bid_prices[0]
                matched_volume = min(remaining_volume, self.bids[best_bid])
                
                # Update the order book
                self.bids[best_bid] -= matched_volume
                remaining_volume -= matched_volume
                
                # If the level is depleted, remove it and update best bid
                if self.bids[best_bid] == 0:
                    del self.bids[best_bid]
                    bid_prices.pop(0)
                    if bid_prices:
                        self.best_bid_price = bid_prices[0]
                    else:
                        # No more bids, set to a low value
                        self.best_bid_price = self.best_ask_price - 10 * self.tick_size
                        # Add some liquidity back
                        new_bid_price = self.best_bid_price
                        self.bids[new_bid_price] = 100
        
        # Update historical data
        self.update_history(timestamp)
        # Record order flow
        self.order_flow.append(('market', side, None, volume, timestamp))
    
    def cancel_order(self, side, price, volume, timestamp):
        """
        Process a cancel order.
        
        Parameters:
        -----------
        side : str
            'bid' or 'ask'
        price : float
            Price level to cancel from
        volume : int
            Size of the cancel order
        timestamp : float
            Current time
        """
        if side == 'bid' and price in self.bids:
            # Cancel from bid side
            self.bids[price] = max(0, self.bids[price] - volume)
            
            # If the level is depleted, remove it and update best bid if necessary
            if self.bids[price] == 0:
                del self.bids[price]
                if price == self.best_bid_price and self.bids:
                    self.best_bid_price = max(self.bids.keys())
                elif not self.bids:
                    # No more bids, set to a low value
                    self.best_bid_price = self.best_ask_price - 10 * self.tick_size
                    # Add some liquidity back
                    new_bid_price = self.best_bid_price
                    self.bids[new_bid_price] = 100
        
        elif side == 'ask' and price in self.asks:
            # Cancel from ask side
            self.asks[price] = max(0, self.asks[price] - volume)
            
            # If the level is depleted, remove it and update best ask if necessary
            if self.asks[price] == 0:
                del self.asks[price]
                if price == self.best_ask_price and self.asks:
                    self.best_ask_price = min(self.asks.keys())
                elif not self.asks:
                    # No more asks, set to a high value
                    self.best_ask_price = self.best_bid_price + 10 * self.tick_size
                    # Add some liquidity back
                    new_ask_price = self.best_ask_price
                    self.asks[new_ask_price] = 100
        
        # Update historical data
        self.update_history(timestamp)
        # Record order flow
        self.order_flow.append(('cancel', side, price, volume, timestamp))
    
    def update_history(self, timestamp):
        """Update historical data."""
        self.mid_prices.append(self.get_mid_price())
        self.spreads.append(self.get_spread())
        self.timestamps.append(timestamp)
        self.bid_volumes.append(sum(self.bids.values()))
        self.ask_volumes.append(sum(self.asks.values()))
    
    def get_book_state(self):
        """Return the current state of the order book."""
        # Sort bids in descending order
        sorted_bids = sorted([(price, vol) for price, vol in self.bids.items()], 
                             key=lambda x: x[0], reverse=True)
        
        # Sort asks in ascending order
        sorted_asks = sorted([(price, vol) for price, vol in self.asks.items()], 
                             key=lambda x: x[0])
        
        return {
            'bids': sorted_bids[:self.max_levels],
            'asks': sorted_asks[:self.max_levels],
            'mid_price': self.get_mid_price(),
            'spread': self.get_spread()
        }
    
    def get_centered_book(self):
        """Get the book centered around the mid price."""
        mid_price = self.get_mid_price()
        centered_bids = {}
        centered_asks = {}
        
        for price, volume in self.bids.items():
            distance = mid_price - price
            centered_bids[distance] = volume
            
        for price, volume in self.asks.items():
            distance = price - mid_price
            centered_asks[distance] = volume
            
        return centered_bids, centered_asks
    
    def get_historical_data(self):
        """Return historical data as a DataFrame."""
        data = {
            'timestamp': self.timestamps,
            'mid_price': self.mid_prices,
            'spread': self.spreads,
            'bid_volume': self.bid_volumes,
            'ask_volume': self.ask_volumes
        }
        return pd.DataFrame(data)


class PoissonLOBSimulator:
    """
    A zero-intelligence model for LOB simulation using Poisson processes
    as described in Cont, Stoikov, and Talreja (2010).
    """
    def __init__(self, lob, simulation_time=1000, time_step=1.0,
                 lambda_limit_base=1.0, lambda_market=0.5, lambda_cancel=0.3,
                 size_limit_mean=50, size_market_mean=50, size_cancel_mean=50,
                 price_range_factor=5.0):
        """
        Initialize the Poisson simulator.
        
        Parameters:
        -----------
        lob : LimitOrderBook
            The limit order book instance
        simulation_time : float
            Total simulation time
        time_step : float
            Time step for simulation
        lambda_limit_base : float
            Base intensity for limit orders
        lambda_market : float
            Intensity for market orders
        lambda_cancel : float
            Intensity for cancel orders
        size_limit_mean, size_market_mean, size_cancel_mean : float
            Mean sizes for different order types
        price_range_factor : float
            Factor to determine range of prices around mid price
        """
        self.lob = lob
        self.simulation_time = simulation_time
        self.time_step = time_step
        
        # Intensities
        self.lambda_limit_base = lambda_limit_base
        self.lambda_market = lambda_market
        self.lambda_cancel = lambda_cancel
        
        # Size distributions (geometric with these means)
        self.size_limit_mean = size_limit_mean
        self.size_market_mean = size_market_mean
        self.size_cancel_mean = size_cancel_mean
        
        # Price range
        self.price_range_factor = price_range_factor
        
        # Track order arrival times for each type
        self.limit_order_times = []
        self.market_order_times = []
        self.cancel_order_times = []
    
    def simulate(self, verbose=True):
        """Run the simulation."""
        current_time = 0.0
        progress_bar = tqdm(total=self.simulation_time) if verbose else None
        
        while current_time < self.simulation_time:
            # Generate inter-arrival times for each order type
            mid_price = self.lob.get_mid_price()
            spread = self.lob.get_spread() * self.lob.tick_size
            
            # Intensities
            lambda_limit = self.lambda_limit_base * np.exp(-spread / (self.lob.tick_size * 5))
            lambda_market = self.lambda_market
            lambda_cancel = self.lambda_cancel * (sum(self.lob.bids.values()) + sum(self.lob.asks.values())) / 1000
            
            # Generate exponentially distributed inter-arrival times
            time_to_limit = np.random.exponential(1.0 / lambda_limit)
            time_to_market = np.random.exponential(1.0 / lambda_market)
            time_to_cancel = np.random.exponential(1.0 / lambda_cancel)
            
            # Determine the next event
            min_time = min(time_to_limit, time_to_market, time_to_cancel)
            current_time += min_time
            
            if current_time >= self.simulation_time:
                break
                
            # Update progress bar
            if verbose:
                progress_bar.update(min_time)
            
            # Process the event
            if min_time == time_to_limit:
                # Limit order
                self.limit_order_times.append(current_time)
                side = 'bid' if np.random.rand() < 0.5 else 'ask'
                
                # Price is distributed around mid price based on distance
                if side == 'bid':
                    # For bids, price is below mid_price
                    price_delta = np.random.exponential(self.lob.tick_size * 5)
                    price = max(mid_price - price_delta, mid_price - self.price_range_factor * spread)
                    price = round(price / self.lob.tick_size) * self.lob.tick_size  # Round to nearest tick
                else:
                    # For asks, price is above mid_price
                    price_delta = np.random.exponential(self.lob.tick_size * 5)
                    price = min(mid_price + price_delta, mid_price + self.price_range_factor * spread)
                    price = round(price / self.lob.tick_size) * self.lob.tick_size  # Round to nearest tick
                
                # Volume is geometrically distributed
                volume = np.random.geometric(1.0 / self.size_limit_mean)
                
                self.lob.limit_order(side, price, volume, current_time)
                
            elif min_time == time_to_market:
                # Market order
                self.market_order_times.append(current_time)
                side = 'bid' if np.random.rand() < 0.5 else 'ask'
                
                # Volume is geometrically distributed
                volume = np.random.geometric(1.0 / self.size_market_mean)
                
                self.lob.market_order(side, volume, current_time)
                
            else:
                # Cancel order
                self.cancel_order_times.append(current_time)
                side = 'bid' if np.random.rand() < 0.5 else 'ask'
                
                if side == 'bid' and self.lob.bids:
                    # Choose a price level weighted by volume
                    prices = list(self.lob.bids.keys())
                    volumes = list(self.lob.bids.values())
                    probabilities = np.array(volumes) / sum(volumes)
                    price = np.random.choice(prices, p=probabilities)
                    
                    # Volume is geometrically distributed, capped by available volume
                    max_volume = self.lob.bids[price]
                    volume = min(np.random.geometric(1.0 / self.size_cancel_mean), max_volume)
                    
                    self.lob.cancel_order(side, price, volume, current_time)
                    
                elif side == 'ask' and self.lob.asks:
                    # Choose a price level weighted by volume
                    prices = list(self.lob.asks.keys())
                    volumes = list(self.lob.asks.values())
                    probabilities = np.array(volumes) / sum(volumes)
                    price = np.random.choice(prices, p=probabilities)
                    
                    # Volume is geometrically distributed, capped by available volume
                    max_volume = self.lob.asks[price]
                    volume = min(np.random.geometric(1.0 / self.size_cancel_mean), max_volume)
                    
                    self.lob.cancel_order(side, price, volume, current_time)
        
        if verbose:
            progress_bar.close()


class HawkesLOBSimulator:
    """
    A Hawkes process-based model for LOB simulation.
    Implements a simplified version of the model described in
    Bacry, Jaisson, and Muzy (2016).
    """
    def __init__(self, lob, simulation_time=1000, time_step=1.0,
                 base_intensities=None, decay_constants=None, 
                 excitation_matrix=None, size_means=None,
                 price_range_factor=5.0):
        """
        Initialize the Hawkes simulator.
        
        Parameters:
        -----------
        lob : LimitOrderBook
            The limit order book instance
        simulation_time : float
            Total simulation time
        time_step : float
            Time step for simulation
        base_intensities : dict
            Base intensities for each order type
        decay_constants : dict
            Decay constants for exponential kernels
        excitation_matrix : dict
            Excitation coefficients between order types
        size_means : dict
            Mean sizes for different order types
        price_range_factor : float
            Factor to determine range of prices around mid price
        """
        self.lob = lob
        self.simulation_time = simulation_time
        self.time_step = time_step
        self.price_range_factor = price_range_factor
        
        # Define order types
        self.order_types = [
            'limit_bid', 'limit_ask', 
            'market_bid', 'market_ask',
            'cancel_bid', 'cancel_ask'
        ]
        
        # Set default parameters if not provided
        if base_intensities is None:
            self.base_intensities = {
                'limit_bid': 1.0, 'limit_ask': 1.0,
                'market_bid': 0.5, 'market_ask': 0.5,
                'cancel_bid': 0.3, 'cancel_ask': 0.3
            }
        else:
            self.base_intensities = base_intensities
            
        if decay_constants is None:
            self.decay_constants = {
                'limit_bid': 0.1, 'limit_ask': 0.1,
                'market_bid': 0.2, 'market_ask': 0.2,
                'cancel_bid': 0.15, 'cancel_ask': 0.15
            }
        else:
            self.decay_constants = decay_constants
            
        if excitation_matrix is None:
            # Create a simple excitation matrix where orders excite same-side orders
            # and market orders increase opposite-side limit orders
            self.excitation_matrix = {}
            for source in self.order_types:
                self.excitation_matrix[source] = {}
                for target in self.order_types:
                    # Default low excitation
                    self.excitation_matrix[source][target] = 0.05
                    
                    # Same side excitation
                    if source.endswith('_bid') and target.endswith('_bid'):
                        self.excitation_matrix[source][target] = 0.2
                    if source.endswith('_ask') and target.endswith('_ask'):
                        self.excitation_matrix[source][target] = 0.2
                    
                    # Market orders increase opposite-side limit orders
                    if source == 'market_bid' and target == 'limit_ask':
                        self.excitation_matrix[source][target] = 0.4
                    if source == 'market_ask' and target == 'limit_bid':
                        self.excitation_matrix[source][target] = 0.4
        else:
            self.excitation_matrix = excitation_matrix
            
        if size_means is None:
            self.size_means = {
                'limit_bid': 50, 'limit_ask': 50,
                'market_bid': 50, 'market_ask': 50,
                'cancel_bid': 50, 'cancel_ask': 50
            }
        else:
            self.size_means = size_means
            
        # Track event history for Hawkes process
        self.event_history = {order_type: [] for order_type in self.order_types}
        
        # Track order arrival times for each type
        self.limit_order_times = []
        self.market_order_times = []
        self.cancel_order_times = []
    
    def compute_hawkes_intensity(self, order_type, current_time):
        """
        Compute the current intensity for a given order type.
        
        Parameters:
        -----------
        order_type : str
            The order type
        current_time : float
            Current time
            
        Returns:
        --------
        float
            The current intensity
        """
        # Base intensity
        intensity = self.base_intensities[order_type]
        
        # Add excitation from past events
        for source_type in self.order_types:
            for event_time in self.event_history[source_type]:
                # Only consider events in the past
                if event_time < current_time:
                    # Compute exponential decay
                    time_diff = current_time - event_time
                    excitation = self.excitation_matrix[source_type][order_type] * \
                                np.exp(-self.decay_constants[order_type] * time_diff)
                    intensity += excitation
        
        # Add state-dependent component based on book imbalance
        imbalance = self.lob.get_book_imbalance()
        
        # More bids than asks increases ask-side activity and decreases bid-side
        if order_type.endswith('_bid'):
            intensity *= max(0.5, 1.0 - 0.5 * imbalance)
        else:
            intensity *= max(0.5, 1.0 + 0.5 * imbalance)
        
        return max(0, intensity)  # Ensure intensity is non-negative
    
    def sample_next_event(self, current_time):
        """
        Sample the next event time and type using thinning algorithm.
        
        Parameters:
        -----------
        current_time : float
            Current time
            
        Returns:
        --------
        tuple
            (next_event_time, next_event_type)
        """
        # Calculate upper bounds for each intensity
        upper_bounds = {}
        for order_type in self.order_types:
            intensity = self.compute_hawkes_intensity(order_type, current_time)
            # Add a small buffer to the upper bound
            upper_bounds[order_type] = intensity * 1.2
        
        # Use thinning algorithm to sample next event
        next_event_time = float('inf')
        next_event_type = None
        
        for order_type in self.order_types:
            if upper_bounds[order_type] > 0:
                # Sample from homogeneous Poisson process with rate upper_bounds[order_type]
                time_to_next = np.random.exponential(1.0 / upper_bounds[order_type])
                candidate_time = current_time + time_to_next
                
                # Accept with probability intensity(candidate_time) / upper_bound
                if candidate_time < next_event_time:
                    # Calculate actual intensity at candidate time
                    actual_intensity = self.compute_hawkes_intensity(order_type, candidate_time)
                    
                    # Accept with probability actual_intensity / upper_bounds[order_type]
                    if np.random.rand() < actual_intensity / upper_bounds[order_type]:
                        next_event_time = candidate_time
                        next_event_type = order_type
        
        return next_event_time, next_event_type
    
    def simulate(self, verbose=True):
        """Run the simulation."""
        current_time = 0.0
        progress_bar = tqdm(total=self.simulation_time) if verbose else None
        
        while current_time < self.simulation_time:
            # Sample the next event
            next_time, next_type = self.sample_next_event(current_time)
            
            # Break if we've reached the simulation time
            if next_time >= self.simulation_time:
                break
                
            # Update current time
            time_delta = next_time - current_time
            current_time = next_time
            
            # Update progress bar
            if verbose:
                progress_bar.update(time_delta)
            
            # Record the event in history
            self.event_history[next_type].append(current_time)
            
            # Process the event
            if next_type.startswith('limit_'):
                self.limit_order_times.append(current_time)
                side = 'bid' if next_type == 'limit_bid' else 'ask'
                
                # Price is distributed around mid price based on distance
                mid_price = self.lob.get_mid_price()
                spread = self.lob.get_spread() * self.lob.tick_size
                
                if side == 'bid':
                    # For bids, price is below mid_price
                    price_delta = np.random.exponential(self.lob.tick_size * 5)
                    price = max(mid_price - price_delta, mid_price - self.price_range_factor * spread)
                    price = round(price / self.lob.tick_size) * self.lob.tick_size  # Round to nearest tick
                else:
                    # For asks, price is above mid_price
                    price_delta = np.random.exponential(self.lob.tick_size * 5)
                    price = min(mid_price + price_delta, mid_price + self.price_range_factor * spread)
                    price = round(price / self.lob.tick_size) * self.lob.tick_size  # Round to nearest tick
                
                # Volume is geometrically distributed
                volume = np.random.geometric(1.0 / self.size_means[next_type])
                
                self.lob.limit_order(side, price, volume, current_time)
                
            elif next_type.startswith('market_'):
                self.market_order_times.append(current_time)
                side = 'bid' if next_type == 'market_bid' else 'ask'
                
                # Volume is geometrically distributed
                volume = np.random.geometric(1.0 / self.size_means[next_type])
                
                self.lob.market_order(side, volume, current_time)
                
            elif next_type.startswith('cancel_'):
                self.cancel_order_times.append(current_time)
                side = 'bid' if next_type == 'cancel_bid' else 'ask'
                
                book_side = self.lob.bids if side == 'bid' else self.lob.asks
                
                if book_side:
                    # Choose a price level weighted by volume
                    prices = list(book_side.keys())
                    volumes = list(book_side.values())
                    probabilities = np.array(volumes) / sum(volumes)
                    price = np.random.choice(prices, p=probabilities)
                    
                    # Volume is geometrically distributed, capped by available volume
                    max_volume = book_side[price]
                    volume = min(np.random.geometric(1.0 / self.size_means[next_type]), max_volume)
                    
                    self.lob.cancel_order(side, price, volume, current_time)
        
        if verbose:
            progress_bar.close()


def analyze_simulation_results(lob, simulator_type, inter_arrival_times=None):
    """
    Analyze simulation results and plot stylized facts.
    
    Parameters:
    -----------
    lob : LimitOrderBook
        The limit order book instance
    simulator_type : str
        Type of simulator used
    inter_arrival_times : dict
        Dictionary of inter-arrival times for different order types
    """
    # Get historical data
    data = lob.get_historical_data()
    
    # Calculate returns at different timescales
    data['returns_1'] = data['mid_price'].pct_change()
    data['returns_5'] = data['mid_price'].pct_change(5)
    data['returns_10'] = data['mid_price'].pct_change(10)
    data['returns_50'] = data['mid_price'].pct_change(50)
    
    # Calculate absolute returns
    data['abs_returns_1'] = data['returns_1'].abs()
    
    # Calculate log returns
    data['log_returns_1'] = np.log(data['mid_price']).diff()
    
    # Create figure
    fig = plt.figure(figsize=(18, 24))
    fig.suptitle(f'Stylized Facts for {simulator_type} Simulation', fontsize=16)
    
    # 1. Price path
    ax1 = fig.add_subplot(5, 2, 1)
    ax1.plot(data['timestamp'], data['mid_price'])
    ax1.set_title('Mid Price Path')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Price')
    ax1.grid(True)
    
    # 2. Returns distribution
    ax2 = fig.add_subplot(5, 2, 2)
    # Drop NaN values
    returns = data['returns_1'].dropna()
    sns.histplot(returns, kde=True, ax=ax2)
    ax2.set_title('Returns Distribution')
    ax2.set_xlabel('Return')
    ax2.set_ylabel('Frequency')
    ax2.grid(True)
    
    # 3. Autocorrelation of returns
    ax3 = fig.add_subplot(5, 2, 3)
    pd.plotting.autocorrelation_plot(data['returns_1'].dropna(), ax=ax3)
    ax3.set_xlim(0, 50)
    ax3.set_title('Autocorrelation of Returns')
    ax3.grid(True)
    
    # 4. Autocorrelation of absolute returns (volatility clustering)
    ax4 = fig.add_subplot(5, 2, 4)
    pd.plotting.autocorrelation_plot(data['abs_returns_1'].dropna(), ax=ax4)
    ax4.set_xlim(0, 50)
    ax4.set_title('Autocorrelation of Absolute Returns (Volatility Clustering)')
    ax4.grid(True)
    
    # 5. Spread distribution
    ax5 = fig.add_subplot(5, 2, 5)
    sns.histplot(data['spread'], kde=True, ax=ax5)
    ax5.set_title('Spread Distribution (in ticks)')
    ax5.set_xlabel('Spread')
    ax5.set_ylabel('Frequency')
    ax5.grid(True)
    
    # 6. Volume distribution
    ax6 = fig.add_subplot(5, 2, 6)
    sns.histplot(data['bid_volume'], kde=True, alpha=0.5, label='Bid Volume', ax=ax6)
    sns.histplot(data['ask_volume'], kde=True, alpha=0.5, label='Ask Volume', ax=ax6)
    ax6.set_title('Volume Distribution')
    ax6.set_xlabel('Volume')
    ax6.set_ylabel('Frequency')
    ax6.legend()
    ax6.grid(True)
    
    # 7. Average LOB shape (for last 100 states)
    ax7 = fig.add_subplot(5, 2, 7)
    # Collect data for average book shape
    centered_books = []
    for i in range(max(0, len(data)-100), len(data)):
        centered_bids, centered_asks = lob.get_centered_book()
        # Convert to regular dictionaries for easier manipulation
        centered_books.append((centered_bids, centered_asks))
    
    # Aggregate data for average shape
    avg_bid_volumes = defaultdict(list)
    avg_ask_volumes = defaultdict(list)
    
    for centered_bids, centered_asks in centered_books:
        for distance, volume in centered_bids.items():
            avg_bid_volumes[distance].append(volume)
        for distance, volume in centered_asks.items():
            avg_ask_volumes[distance].append(volume)
    
    # Calculate averages
    avg_bid_distances = []
    avg_bid_volumes_list = []
    avg_ask_distances = []
    avg_ask_volumes_list = []
    
    for distance, volumes in avg_bid_volumes.items():
        avg_bid_distances.append(distance)
        avg_bid_volumes_list.append(np.mean(volumes))
    
    for distance, volumes in avg_ask_volumes.items():
        avg_ask_distances.append(distance)
        avg_ask_volumes_list.append(np.mean(volumes))
    
    # Sort by distance
    bid_data = sorted(zip(avg_bid_distances, avg_bid_volumes_list))
    ask_data = sorted(zip(avg_ask_distances, avg_ask_volumes_list))
    
    bid_distances, bid_volumes = zip(*bid_data) if bid_data else ([], [])
    ask_distances, ask_volumes = zip(*ask_data) if ask_data else ([], [])
    
    ax7.bar([-d for d in bid_distances], bid_volumes, width=0.2, color='green', alpha=0.5, label='Bid')
    ax7.bar(ask_distances, ask_volumes, width=0.2, color='red', alpha=0.5, label='Ask')
    ax7.set_title('Average LOB Shape')
    ax7.set_xlabel('Distance from Mid Price')
    ax7.set_ylabel('Average Volume')
    ax7.legend()
    ax7.grid(True)
    
    # 8. Inter-arrival times distribution (if available)
    if hasattr(simulator_type, 'limit_order_times'):
        ax8 = fig.add_subplot(5, 2, 8)
        
        # Calculate inter-arrival times
        limit_inter_arrivals = np.diff(simulator_type.limit_order_times)
        market_inter_arrivals = np.diff(simulator_type.market_order_times)
        cancel_inter_arrivals = np.diff(simulator_type.cancel_order_times)
        
        # Plot distributions
        if len(limit_inter_arrivals) > 0:
            sns.histplot(limit_inter_arrivals, kde=True, alpha=0.5, label='Limit Orders', ax=ax8)
        if len(market_inter_arrivals) > 0:
            sns.histplot(market_inter_arrivals, kde=True, alpha=0.5, label='Market Orders', ax=ax8)
        if len(cancel_inter_arrivals) > 0:
            sns.histplot(cancel_inter_arrivals, kde=True, alpha=0.5, label='Cancel Orders', ax=ax8)
            
        ax8.set_title('Inter-arrival Times Distribution')
        ax8.set_xlabel('Time')
        ax8.set_ylabel('Frequency')
        ax8.legend()
        ax8.grid(True)
        
        # Test for exponential distribution (expected for Poisson)
        if len(limit_inter_arrivals) > 20:
            # Normalized for testing
            normalized_limit = limit_inter_arrivals / np.mean(limit_inter_arrivals)
            ks_stat, p_value = kstest(normalized_limit, 'expon')
            
            ax8.set_title(f'Inter-arrival Times (KS test p-value: {p_value:.4f})')
    
    # 9. Returns at different timescales
    ax9 = fig.add_subplot(5, 2, 9)
    sns.histplot(data['returns_1'].dropna(), kde=True, alpha=0.5, label='1-step', ax=ax9)
    sns.histplot(data['returns_5'].dropna(), kde=True, alpha=0.5, label='5-step', ax=ax9)
    sns.histplot(data['returns_10'].dropna(), kde=True, alpha=0.5, label='10-step', ax=ax9)
    sns.histplot(data['returns_50'].dropna(), kde=True, alpha=0.5, label='50-step', ax=ax9)
    ax9.set_title('Returns Distribution at Different Timescales')
    ax9.set_xlabel('Return')
    ax9.set_ylabel('Frequency')
    ax9.legend()
    ax9.grid(True)
    
    # 10. Volatility (standard deviation of returns) vs. volume
    ax10 = fig.add_subplot(5, 2, 10)
    window_size = 20
    data['rolling_vol'] = data['returns_1'].rolling(window=window_size).std()
    data['rolling_volume'] = (data['bid_volume'] + data['ask_volume']).rolling(window=window_size).mean()
    
    # Drop NaN values
    df_clean = data.dropna(subset=['rolling_vol', 'rolling_volume'])
    
    ax10.scatter(df_clean['rolling_volume'], df_clean['rolling_vol'], alpha=0.5)
    ax10.set_title('Volatility vs. Volume')
    ax10.set_xlabel('Volume (20-period average)')
    ax10.set_ylabel('Volatility (20-period std)')
    ax10.grid(True)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.savefig(f'{simulator_type}_stylized_facts.png')
    plt.close()
    
    return data


def test_market_impact(lob, simulator_type, impact_sizes=[100, 500, 1000], repetitions=10):
    """
    Test the market impact by introducing large market orders.
    
    Parameters:
    -----------
    lob : LimitOrderBook
        The limit order book instance
    simulator_type : str
        Type of simulator used
    impact_sizes : list
        List of sizes for the market orders
    repetitions : int
        Number of repetitions for each size
    """
    # Create figure
    fig, axs = plt.subplots(len(impact_sizes), 1, figsize=(15, 5*len(impact_sizes)))
    fig.suptitle(f'Market Impact Analysis for {simulator_type}', fontsize=16)
    
    if len(impact_sizes) == 1:
        axs = [axs]
    
    for i, size in enumerate(impact_sizes):
        # Initialize data collection
        price_paths = []
        
        for rep in range(repetitions):
            # Create a copy of the LOB
            lob_copy = LimitOrderBook(tick_size=lob.tick_size, 
                                    init_mid_price=lob.get_mid_price(),
                                    max_levels=lob.max_levels)
            
            # Copy the current state
            for price, volume in lob.bids.items():
                lob_copy.bids[price] = volume
            for price, volume in lob.asks.items():
                lob_copy.asks[price] = volume
            
            lob_copy.best_bid_price = lob.best_bid_price
            lob_copy.best_ask_price = lob.best_ask_price
            
            # Record initial mid price
            initial_mid = lob_copy.get_mid_price()
            
            # Execute a large market order
            lob_copy.market_order('bid', size, 0)
            
            # Simulate for a short period to observe recovery
            if simulator_type == 'Poisson':
                sim = PoissonLOBSimulator(lob_copy, simulation_time=100, time_step=1.0)
            else:  # Hawkes
                sim = HawkesLOBSimulator(lob_copy, simulation_time=100, time_step=1.0)
            
            sim.simulate(verbose=False)
            
            # Get the price path
            data = lob_copy.get_historical_data()
            # Normalize to the initial mid price
            data['normalized_mid'] = data['mid_price'] / initial_mid - 1
            
            price_paths.append(data['normalized_mid'].values)
        
        # Calculate average and std of price paths
        max_length = max(len(path) for path in price_paths)
        padded_paths = [np.pad(path, (0, max_length - len(path)), 'constant', constant_values=np.nan) 
                        for path in price_paths]
        paths_array = np.array(padded_paths)
        
        mean_path = np.nanmean(paths_array, axis=0)
        std_path = np.nanstd(paths_array, axis=0)
        
        # Plot
        x = np.arange(len(mean_path))
        axs[i].plot(x, mean_path, label=f'Mean Path (Size {size})')
        axs[i].fill_between(x, mean_path - std_path, mean_path + std_path, alpha=0.3, label='±1 Std')
        axs[i].axhline(y=0, color='k', linestyle='--', alpha=0.5)
        axs[i].set_title(f'Market Impact of Order Size {size}')
        axs[i].set_xlabel('Time Steps')
        axs[i].set_ylabel('Normalized Price Change')
        axs[i].legend()
        axs[i].grid(True)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.savefig(f'{simulator_type}_market_impact.png')
    plt.close()


def compare_models():
    """
    Compare Poisson and Hawkes models directly.
    """
    # Initialize a fresh LOB
    lob_poisson = LimitOrderBook(tick_size=0.01, init_mid_price=100.0)
    
    # Create Poisson simulator
    poisson_sim = PoissonLOBSimulator(lob_poisson, simulation_time=5000)
    
    # Run Poisson simulation
    print("Running Poisson simulation...")
    start_time = time.time()
    poisson_sim.simulate()
    poisson_time = time.time() - start_time
    print(f"Poisson simulation completed in {poisson_time:.2f} seconds")
    
    # Analyze Poisson results
    poisson_data = analyze_simulation_results(lob_poisson, "Poisson", poisson_sim)
    
    # Initialize a fresh LOB for Hawkes
    lob_hawkes = LimitOrderBook(tick_size=0.01, init_mid_price=100.0)
    
    # Create Hawkes simulator
    hawkes_sim = HawkesLOBSimulator(lob_hawkes, simulation_time=5000)
    
    # Run Hawkes simulation
    print("Running Hawkes simulation...")
    start_time = time.time()
    hawkes_sim.simulate()
    hawkes_time = time.time() - start_time
    print(f"Hawkes simulation completed in {hawkes_time:.2f} seconds")
    
    # Analyze Hawkes results
    hawkes_data = analyze_simulation_results(lob_hawkes, "Hawkes", hawkes_sim)
    
    # Test market impact
    print("Testing market impact for Poisson model...")
    test_market_impact(lob_poisson, "Poisson")
    
    print("Testing market impact for Hawkes model...")
    test_market_impact(lob_hawkes, "Hawkes")
    
    # Create comparison plots
    fig, axs = plt.subplots(3, 2, figsize=(15, 15))
    fig.suptitle('Comparison of Poisson and Hawkes Models', fontsize=16)
    
    # 1. Price paths
    axs[0, 0].plot(poisson_data['timestamp'], poisson_data['mid_price'], label='Poisson')
    axs[0, 0].plot(hawkes_data['timestamp'], hawkes_data['mid_price'], label='Hawkes')
    axs[0, 0].set_title('Mid Price Paths')
    axs[0, 0].set_xlabel('Time')
    axs[0, 0].set_ylabel('Price')
    axs[0, 0].legend()
    axs[0, 0].grid(True)
    
    # 2. Returns distribution
    poisson_returns = poisson_data['returns_1'].dropna()
    hawkes_returns = hawkes_data['returns_1'].dropna()
    
    sns.histplot(poisson_returns, kde=True, alpha=0.5, label='Poisson', ax=axs[0, 1])
    sns.histplot(hawkes_returns, kde=True, alpha=0.5, label='Hawkes', ax=axs[0, 1])
    axs[0, 1].set_title('Returns Distribution')
    axs[0, 1].set_xlabel('Return')
    axs[0, 1].set_ylabel('Frequency')
    axs[0, 1].legend()
    axs[0, 1].grid(True)
    
    # 3. Autocorrelation of returns
    from statsmodels.graphics.tsaplots import plot_acf
    
    plot_acf(poisson_returns, lags=50, alpha=0.05, ax=axs[1, 0], label='Poisson')
    plot_acf(hawkes_returns, lags=50, alpha=0.05, ax=axs[1, 0], label='Hawkes')
    axs[1, 0].set_title('Autocorrelation of Returns')
    axs[1, 0].legend()
    
    # 4. Autocorrelation of absolute returns
    poisson_abs_returns = poisson_data['abs_returns_1'].dropna()
    hawkes_abs_returns = hawkes_data['abs_returns_1'].dropna()
    
    plot_acf(poisson_abs_returns, lags=50, alpha=0.05, ax=axs[1, 1], label='Poisson')
    plot_acf(hawkes_abs_returns, lags=50, alpha=0.05, ax=axs[1, 1], label='Hawkes')
    axs[1, 1].set_title('Autocorrelation of Absolute Returns')
    axs[1, 1].legend()
    
    # 5. Inter-arrival times
    poisson_limit_inter_arrivals = np.diff(poisson_sim.limit_order_times)
    hawkes_limit_inter_arrivals = np.diff(hawkes_sim.limit_order_times)
    
    # Normalized for comparison
    poisson_normalized = poisson_limit_inter_arrivals / np.mean(poisson_limit_inter_arrivals)
    hawkes_normalized = hawkes_limit_inter_arrivals / np.mean(hawkes_limit_inter_arrivals)
    
    # KS test against exponential
    poisson_ks, poisson_p = kstest(poisson_normalized, 'expon')
    hawkes_ks, hawkes_p = kstest(hawkes_normalized, 'expon')
    
    sns.histplot(poisson_normalized, kde=True, alpha=0.5, label=f'Poisson (p={poisson_p:.4f})', ax=axs[2, 0])
    sns.histplot(hawkes_normalized, kde=True, alpha=0.5, label=f'Hawkes (p={hawkes_p:.4f})', ax=axs[2, 0])
    axs[2, 0].set_title('Normalized Inter-arrival Times')
    axs[2, 0].set_xlabel('Normalized Time')
    axs[2, 0].set_ylabel('Frequency')
    axs[2, 0].legend()
    axs[2, 0].grid(True)
    
    # 6. Q-Q plot against exponential
    from scipy.stats import expon
    
    # Calculate quantiles
    q_theoretical = np.linspace(0, 1, 100)
    q_poisson = np.quantile(poisson_normalized, q_theoretical)
    q_hawkes = np.quantile(hawkes_normalized, q_theoretical)
    q_exp = expon.ppf(q_theoretical)
    
    axs[2, 1].plot(q_exp, q_poisson, 'o', label='Poisson')
    axs[2, 1].plot(q_exp, q_hawkes, 'o', label='Hawkes')
    axs[2, 1].plot([0, 5], [0, 5], 'k--', label='Exponential')
    axs[2, 1].set_title('Q-Q Plot against Exponential Distribution')
    axs[2, 1].set_xlabel('Theoretical Quantiles')
    axs[2, 1].set_ylabel('Sample Quantiles')
    axs[2, 1].legend()
    axs[2, 1].grid(True)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.savefig('model_comparison.png')
    plt.close()
    
    # Summary statistics
    print("\nSummary Statistics:")
    print("-" * 50)
    print(f"{'Metric':<25} {'Poisson':<15} {'Hawkes':<15}")
    print("-" * 50)
    print(f"{'Mean Return':<25} {poisson_returns.mean():<15.6f} {hawkes_returns.mean():<15.6f}")
    print(f"{'Std Dev of Returns':<25} {poisson_returns.std():<15.6f} {hawkes_returns.std():<15.6f}")
    print(f"{'Kurtosis of Returns':<25} {poisson_returns.kurtosis():<15.6f} {hawkes_returns.kurtosis():<15.6f}")
    print(f"{'Mean Spread (ticks)':<25} {poisson_data['spread'].mean():<15.6f} {hawkes_data['spread'].mean():<15.6f}")
    print(f"{'KS Test p-value':<25} {poisson_p:<15.6f} {hawkes_p:<15.6f}")
    print(f"{'Simulation Time (s)':<25} {poisson_time:<15.2f} {hawkes_time:<15.2f}")
    
    return poisson_data, hawkes_data


if __name__ == "__main__":
    # Run comparison between Poisson and Hawkes models
    poisson_data, hawkes_data = compare_models()