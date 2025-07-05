import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import deque
import random
from tqdm import tqdm
from scipy import stats

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

class OrderBook:
    """A simplified limit order book implementation."""
    
    def __init__(self, initial_price=100.0, initial_spread=0.1, 
                 initial_depth=100, depth_decay=0.9):
        self.mid_price = initial_price
        self.spread = initial_spread
        self.depth_decay = depth_decay
        
        # Initialize bid and ask sides with initial liquidity
        self.bids = {}  # price -> volume
        self.asks = {}  # price -> volume
        
        # Create initial liquidity
        self.best_bid = self.mid_price - self.spread/2
        self.best_ask = self.mid_price + self.spread/2
        
        # Populate order book with initial liquidity
        for i in range(1, 11):
            bid_price = round(self.best_bid - (i-1)*0.01, 2)
            ask_price = round(self.best_ask + (i-1)*0.01, 2)
            self.bids[bid_price] = initial_depth * (self.depth_decay ** (i-1))
            self.asks[ask_price] = initial_depth * (self.depth_decay ** (i-1))
    
    def get_book_state(self):
        """Return the current state of the order book."""
        return {
            'mid_price': self.mid_price,
            'best_bid': self.best_bid,
            'best_ask': self.best_ask,
            'spread': self.spread,
            'bid_volume': sum(self.bids.values()),
            'ask_volume': sum(self.asks.values()),
            'best_bid_volume': self.bids.get(self.best_bid, 0),
            'best_ask_volume': self.asks.get(self.best_ask, 0)
        }
    
    def add_limit_order(self, side, price, volume):
        """Add a limit order to the book."""
        if side == 'buy':
            if price in self.bids:
                self.bids[price] += volume
            else:
                self.bids[price] = volume
            
            # Update best bid if necessary
            if price > self.best_bid:
                self.best_bid = price
                self.spread = self.best_ask - self.best_bid
                self.mid_price = (self.best_bid + self.best_ask) / 2
                
        elif side == 'sell':
            if price in self.asks:
                self.asks[price] += volume
            else:
                self.asks[price] = volume
            
            # Update best ask if necessary
            if price < self.best_ask:
                self.best_ask = price
                self.spread = self.best_ask - self.best_bid
                self.mid_price = (self.best_bid + self.best_ask) / 2
    
    def cancel_limit_order(self, side, price, volume):
        """Cancel a limit order from the book."""
        if side == 'buy' and price in self.bids:
            self.bids[price] = max(0, self.bids[price] - volume)
            if self.bids[price] == 0:
                del self.bids[price]
                if price == self.best_bid:
                    self.best_bid = max(self.bids.keys()) if self.bids else (self.mid_price - self.spread/2)
                    self.spread = self.best_ask - self.best_bid
                    self.mid_price = (self.best_bid + self.best_ask) / 2
        
        elif side == 'sell' and price in self.asks:
            self.asks[price] = max(0, self.asks[price] - volume)
            if self.asks[price] == 0:
                del self.asks[price]
                if price == self.best_ask:
                    self.best_ask = min(self.asks.keys()) if self.asks else (self.mid_price + self.spread/2)
                    self.spread = self.best_ask - self.best_bid
                    self.mid_price = (self.best_bid + self.best_ask) / 2
    
    def execute_market_order(self, side, volume):
        """Execute a market order against the book."""
        executed_volume = 0
        avg_price = 0
        
        if side == 'buy':
            # Execute against ask side
            available_prices = sorted(self.asks.keys())
            remaining_volume = volume
            total_cost = 0
            
            for price in available_prices:
                available_volume = self.asks[price]
                if remaining_volume <= available_volume:
                    # Complete execution
                    self.asks[price] -= remaining_volume
                    if self.asks[price] == 0:
                        del self.asks[price]
                    total_cost += remaining_volume * price
                    executed_volume += remaining_volume
                    remaining_volume = 0
                    break
                else:
                    # Partial execution
                    total_cost += available_volume * price
                    executed_volume += available_volume
                    remaining_volume -= available_volume
                    del self.asks[price]
            
            # Update best ask if necessary
            if available_prices and price == available_prices[0]:
                self.best_ask = min(self.asks.keys()) if self.asks else (self.mid_price + self.spread/2)
                self.spread = self.best_ask - self.best_bid
                self.mid_price = (self.best_bid + self.best_ask) / 2
            
            if executed_volume > 0:
                avg_price = total_cost / executed_volume
        
        elif side == 'sell':
            # Execute against bid side
            available_prices = sorted(self.bids.keys(), reverse=True)
            remaining_volume = volume
            total_revenue = 0
            
            for price in available_prices:
                available_volume = self.bids[price]
                if remaining_volume <= available_volume:
                    # Complete execution
                    self.bids[price] -= remaining_volume
                    if self.bids[price] == 0:
                        del self.bids[price]
                    total_revenue += remaining_volume * price
                    executed_volume += remaining_volume
                    remaining_volume = 0
                    break
                else:
                    # Partial execution
                    total_revenue += available_volume * price
                    executed_volume += available_volume
                    remaining_volume -= available_volume
                    del self.bids[price]
            
            # Update best bid if necessary
            if available_prices and price == available_prices[0]:
                self.best_bid = max(self.bids.keys()) if self.bids else (self.mid_price - self.spread/2)
                self.spread = self.best_ask - self.best_bid
                self.mid_price = (self.best_bid + self.best_ask) / 2
            
            if executed_volume > 0:
                avg_price = total_revenue / executed_volume
        
        return executed_volume, avg_price

class LiquidityProvider:
    """Electronic Liquidity Provider - provides liquidity around the spread."""
    
    def __init__(self, order_book, avg_orders_per_step=5, order_size_mean=50, 
                 order_size_std=20, price_delta_mean=0.02, price_delta_std=0.01,
                 cancellation_rate=0.3):
        self.order_book = order_book
        self.avg_orders_per_step = avg_orders_per_step
        self.order_size_mean = order_size_mean
        self.order_size_std = order_size_std
        self.price_delta_mean = price_delta_mean
        self.price_delta_std = price_delta_std
        self.cancellation_rate = cancellation_rate
        self.orders = []  # Track active orders: (side, price, volume)
    
    def act(self):
        """Add liquidity to the order book."""
        # Cancel some existing orders
        self._cancel_orders()
        
        # Add new orders
        num_orders = np.random.poisson(self.avg_orders_per_step)
        
        for _ in range(num_orders):
            side = 'buy' if np.random.random() < 0.5 else 'sell'
            size = max(1, int(np.random.normal(self.order_size_mean, self.order_size_std)))
            
            if side == 'buy':
                price_delta = abs(np.random.normal(self.price_delta_mean, self.price_delta_std))
                price = round(self.order_book.best_bid - price_delta, 2)
            else:
                price_delta = abs(np.random.normal(self.price_delta_mean, self.price_delta_std))
                price = round(self.order_book.best_ask + price_delta, 2)
            
            self.order_book.add_limit_order(side, price, size)
            self.orders.append((side, price, size))
    
    def _cancel_orders(self):
        """Cancel a fraction of existing orders."""
        to_cancel = []
        for i, (side, price, volume) in enumerate(self.orders):
            if np.random.random() < self.cancellation_rate:
                to_cancel.append(i)
                self.order_book.cancel_limit_order(side, price, volume)
        
        # Remove cancelled orders from tracking
        for i in sorted(to_cancel, reverse=True):
            self.orders.pop(i)

class FundamentalistTrader:
    """A trader who trades based on a perceived fundamental value."""
    
    def __init__(self, order_book, initial_fundamental=100.0, 
                 fundamental_drift=0, fundamental_volatility=0.02,
                 trade_probability=0.1, order_size_mean=100, order_size_std=50):
        self.order_book = order_book
        self.fundamental_value = initial_fundamental
        self.fundamental_drift = fundamental_drift
        self.fundamental_volatility = fundamental_volatility
        self.trade_probability = trade_probability
        self.order_size_mean = order_size_mean
        self.order_size_std = order_size_std
        self.cash = 1000000  # Initial cash
        self.inventory = 0  # Initial inventory
    
    def update_fundamental(self):
        """Update the perceived fundamental value."""
        self.fundamental_value += self.fundamental_drift
        self.fundamental_value *= (1 + np.random.normal(0, self.fundamental_volatility))
    
    def act(self):
        """Decide whether to trade and execute if so."""
        # Only trade with some probability
        if np.random.random() > self.trade_probability:
            return None, 0
        
        # Update fundamental value
        self.update_fundamental()
        
        # Decide on trade direction based on fundamental vs. market price
        if self.fundamental_value > self.order_book.mid_price:
            side = 'buy'
        else:
            side = 'sell'
        
        # Determine order size
        size = max(1, int(np.random.normal(self.order_size_mean, self.order_size_std)))
        
        # Don't sell more than we own
        if side == 'sell':
            size = min(size, self.inventory)
            if size == 0:
                return None, 0
        
        # Execute market order
        executed_volume, avg_price = self.order_book.execute_market_order(side, size)
        
        # Update cash and inventory
        if side == 'buy':
            self.cash -= executed_volume * avg_price
            self.inventory += executed_volume
        else:
            self.cash += executed_volume * avg_price
            self.inventory -= executed_volume
        
        return side, executed_volume

class ChartistTrader:
    """A trader who trades based on price trends."""
    
    def __init__(self, order_book, window_size=20, trade_probability=0.1,
                 order_size_mean=100, order_size_std=50, threshold=0.001):
        self.order_book = order_book
        self.window_size = window_size
        self.price_history = deque(maxlen=window_size)
        self.trade_probability = trade_probability
        self.order_size_mean = order_size_mean
        self.order_size_std = order_size_std
        self.threshold = threshold
        self.cash = 1000000  # Initial cash
        self.inventory = 0  # Initial inventory
    
    def detect_trend(self):
        """Detect price trend based on moving averages."""
        if len(self.price_history) < self.window_size:
            return 0
        
        # Simple trend detection based on moving averages
        short_window = self.window_size // 4
        short_ma = np.mean(list(self.price_history)[-short_window:])
        long_ma = np.mean(list(self.price_history))
        
        # Calculate trend strength
        trend = (short_ma - long_ma) / long_ma
        
        return trend
    
    def act(self):
        """Decide whether to trade and execute if so."""
        # Record current price
        self.price_history.append(self.order_book.mid_price)
        
        # Only trade with some probability and if we have enough history
        if np.random.random() > self.trade_probability or len(self.price_history) < self.window_size:
            return None, 0
        
        # Detect trend
        trend = self.detect_trend()
        
        # Only trade if trend is strong enough
        if abs(trend) < self.threshold:
            return None, 0
        
        # Decide on trade direction based on trend
        if trend > 0:
            side = 'buy'
        else:
            side = 'sell'
        
        # Determine order size
        size = max(1, int(np.random.normal(self.order_size_mean, self.order_size_std)))
        
        # Don't sell more than we own
        if side == 'sell':
            size = min(size, self.inventory)
            if size == 0:
                return None, 0
        
        # Execute market order
        executed_volume, avg_price = self.order_book.execute_market_order(side, size)
        
        # Update cash and inventory
        if side == 'buy':
            self.cash -= executed_volume * avg_price
            self.inventory += executed_volume
        else:
            self.cash += executed_volume * avg_price
            self.inventory -= executed_volume
        
        return side, executed_volume

class QLearningAgent:
    """A Q-learning agent for optimal execution."""
    
    def __init__(self, order_book, agent_type="I", side="buy", parent_order_size=1000,
                 twap_intervals=10, state_size=5, learning_rate=0.1, discount_factor=0.95,
                 exploration_rate=1.0, exploration_decay=0.995, exploration_min=0.01,
                 lambda_r=0.5, gamma_r=0.01):
        self.order_book = order_book
        self.agent_type = agent_type  # "I" (MO only) or "II" (MO + LO)
        self.side = side  # "buy" or "sell"
        self.parent_order_size = parent_order_size
        self.twap_intervals = twap_intervals
        
        # State space parameters (as described in Table 1 of the paper)
        self.state_size = state_size  # Number of discretization bins for each state variable
        
        # Learning parameters
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.exploration_min = exploration_min
        
        # Reward function parameters (from equation 1 in the paper)
        self.lambda_r = lambda_r
        self.gamma_r = gamma_r
        
        # Initialize state and action spaces
        self.init_state_action_spaces()
        
        # Initialize Q-table
        self.q_table = {}
        
        # Trading variables
        self.remaining_inventory = parent_order_size
        self.executed_orders = []  # List of (volume, price) tuples
        self.vwap = 0
        self.current_interval = 0
        self.twap_size = parent_order_size / twap_intervals
        
        # Performance tracking
        self.rewards = []
        self.cumulative_reward = 0
    
    def init_state_action_spaces(self):
        """Initialize state and action spaces based on agent type."""
        # For Type I agents (MO only)
        if self.agent_type == "I":
            self.actions = [0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2]  # Multiples of TWAP
        
        # For Type II agents (MO + LO)
        elif self.agent_type == "II":
            # Actions: (MO multiplier, LO depth, LO rate)
            # MO multiplier: 0, 0.5, 1, 1.5, 2
            # LO depth: shallow (0.01) or deep (1)
            # LO rate: fast (1/100), moderate (1/10), slow (1)
            self.actions = [
                # MO only actions
                (0, None, None),
                (0.5, None, None),
                (1, None, None),
                (1.5, None, None),
                (2, None, None),
                # LO only actions
                (0, 0.01, 100),  # Shallow, fast
                (0, 0.01, 10),   # Shallow, moderate
                (0, 0.01, 1),    # Shallow, slow
                (0, 1, 100),     # Deep, fast
                (0, 1, 10),      # Deep, moderate
                (0, 1, 1),       # Deep, slow
                # Mixed actions (MO + LO)
                (0.5, 0.01, 10),  # Moderate MO + shallow, moderate LO
                (1, 0.01, 10),    # TWAP MO + shallow, moderate LO
                (0.5, 1, 10),     # Moderate MO + deep, moderate LO
                (1, 1, 10),       # TWAP MO + deep, moderate LO
            ]
    
    def discretize_state(self, inventory, time, volume, spread):
        """Discretize the state variables into bins."""
        # Inventory state
        inventory_bin = min(int(inventory / (self.parent_order_size / self.state_size)), self.state_size - 1)
        
        # Time state
        time_bin = min(int(time / (self.twap_intervals / self.state_size)), self.state_size - 1)
        
        # Volume state - simplified binning based on quantiles
        if volume <= 50:
            volume_bin = 0
        elif volume <= 200:
            volume_bin = 1
        elif volume <= 500:
            volume_bin = 2
        elif volume <= 1000:
            volume_bin = 3
        else:
            volume_bin = 4
        
        # Spread state - simplified binning based on typical spreads
        if spread <= 0.01:
            spread_bin = 0
        elif spread <= 0.02:
            spread_bin = 1
        elif spread <= 0.03:
            spread_bin = 2
        elif spread <= 0.05:
            spread_bin = 3
        else:
            spread_bin = 4
        
        return (inventory_bin, time_bin, volume_bin, spread_bin)
    
    def get_state(self):
        """Get the current state for decision making."""
        book_state = self.order_book.get_book_state()
        
        # Get relevant state variables
        inventory = self.remaining_inventory
        time = self.current_interval
        
        # Get volume at best bid/ask depending on agent side
        if self.side == "buy":
            volume = book_state['best_ask_volume']
        else:
            volume = book_state['best_bid_volume']
        
        spread = book_state['spread']
        
        # Discretize state
        return self.discretize_state(inventory, time, volume, spread)
    
    def get_q_value(self, state, action):
        """Get Q-value for a state-action pair, initializing if not present."""
        if state not in self.q_table:
            self.q_table[state] = np.zeros(len(self.actions))
        
        action_idx = self.actions.index(action)
        return self.q_table[state][action_idx]
    
    def update_q_value(self, state, action, reward, next_state):
        """Update Q-value for a state-action pair."""
        if state not in self.q_table:
            self.q_table[state] = np.zeros(len(self.actions))
        
        action_idx = self.actions.index(action)
        
        # Calculate the maximum Q-value for the next state
        if next_state not in self.q_table:
            self.q_table[next_state] = np.zeros(len(self.actions))
        
        max_next_q = np.max(self.q_table[next_state])
        
        # Update Q-value using the Q-learning update rule
        self.q_table[state][action_idx] += self.learning_rate * (
            reward + self.discount_factor * max_next_q - self.q_table[state][action_idx]
        )
    
    def choose_action(self, state, training=True):
        """Choose an action using epsilon-greedy policy."""
        if training and np.random.random() < self.exploration_rate:
            # Exploration: choose a random action
            return random.choice(self.actions)
        else:
            # Exploitation: choose the best action according to Q-values
            if state not in self.q_table:
                self.q_table[state] = np.zeros(len(self.actions))
            
            return self.actions[np.argmax(self.q_table[state])]
    
    def execute_action(self, action):
        """Execute the chosen action in the market."""
        twap_size = self.twap_size
        executed_volume = 0
        avg_price = 0
        
        if self.agent_type == "I":
            # Type I: Market Order only
            mo_size = int(action * twap_size)
            mo_size = min(mo_size, self.remaining_inventory)
            
            if mo_size > 0:
                executed_volume, avg_price = self.order_book.execute_market_order(self.side, mo_size)
        
        elif self.agent_type == "II":
            # Type II: Market Order and/or Limit Order
            mo_mult, lo_depth, lo_rate = action
            
            # Execute market order if applicable
            if mo_mult is not None and mo_mult > 0:
                mo_size = int(mo_mult * twap_size)
                mo_size = min(mo_size, self.remaining_inventory)
                
                if mo_size > 0:
                    mo_executed, mo_avg_price = self.order_book.execute_market_order(self.side, mo_size)
                    executed_volume += mo_executed
                    avg_price = mo_avg_price  # Simplified for now
            
            # Place limit order if applicable
            if lo_depth is not None and lo_rate is not None:
                lo_size = int(twap_size / lo_rate)
                lo_size = min(lo_size, self.remaining_inventory - executed_volume)
                
                if lo_size > 0:
                    # Calculate limit order price
                    if self.side == "buy":
                        lo_price = round(self.order_book.best_bid + lo_depth, 2)
                    else:
                        lo_price = round(self.order_book.best_ask - lo_depth, 2)
                    
                    # Place the limit order
                    self.order_book.add_limit_order(self.side, lo_price, lo_size)
                    
                    # Simplification: assume some percentage of LO gets executed
                    # In a more realistic model, this would depend on market dynamics over time
                    lo_executed = int(lo_size * 0.3)  # Assume 30% execution rate
                    if lo_executed > 0:
                        executed_volume += lo_executed
                        avg_price = lo_price if avg_price == 0 else (avg_price + lo_price) / 2
        
        # Update remaining inventory
        self.remaining_inventory -= executed_volume
        
        # Record execution
        if executed_volume > 0:
            self.executed_orders.append((executed_volume, avg_price))
        
        return executed_volume, avg_price
    
    def calculate_reward(self, executed_volume, avg_price):
        """Calculate reward based on execution performance."""
        # Calculate VWAP for agent's trades
        total_volume = sum(vol for vol, _ in self.executed_orders)
        if total_volume > 0:
            agent_vwap = sum(vol * price for vol, price in self.executed_orders) / total_volume
        else:
            agent_vwap = 0
        
        # Calculate market VWAP (simplified for simulation)
        # In reality, this would be based on all trades excluding the agent's
        market_vwap = self.order_book.mid_price
        
        # Calculate slippage component (equation 1 in the paper)
        if agent_vwap > 0 and market_vwap > 0:
            slippage = np.log(agent_vwap / market_vwap)
            # Adjust sign based on agent side
            slippage = slippage if self.side == "sell" else -slippage
        else:
            slippage = 0
        
        # Calculate penalty component
        penalty = 0
        if executed_volume > 0:
            penalty = (self.remaining_inventory / executed_volume) * self.lambda_r * np.exp(self.gamma_r * self.current_interval)
        
        # Final reward
        reward = slippage - penalty
        
        return reward
    
    def step(self, training=True):
        """Take a step in the environment."""
        # Get current state
        state = self.get_state()
        
        # Choose and execute action
        action = self.choose_action(state, training)
        executed_volume, avg_price = self.execute_action(action)
        
        # Calculate reward
        reward = self.calculate_reward(executed_volume, avg_price)
        self.rewards.append(reward)
        self.cumulative_reward += reward
        
        # Update current interval
        self.current_interval += 1
        
        # Get next state
        next_state = self.get_state()
        
        # Update Q-value if training
        if training:
            self.update_q_value(state, action, reward, next_state)
            
            # Decay exploration rate
            self.exploration_rate = max(self.exploration_min, 
                                       self.exploration_rate * self.exploration_decay)
        
        return state, action, reward, next_state, executed_volume, avg_price
    
    def reset(self):
        """Reset agent state for a new episode."""
        self.remaining_inventory = self.parent_order_size
        self.executed_orders = []
        self.vwap = 0
        self.current_interval = 0
        self.rewards = []
        self.cumulative_reward = 0

class MarketSimulation:
    """A simulation of a financial market with multiple agent types."""
    
    def __init__(self, num_fundamentalists=5, num_chartists=5, num_liquidity_providers=1,
                 rl_agent_configs=None, initial_price=100, simulation_steps=1000):
        self.order_book = OrderBook(initial_price=initial_price)
        self.simulation_steps = simulation_steps
        
        # Initialize agent populations
        self.liquidity_providers = [
            LiquidityProvider(self.order_book) for _ in range(num_liquidity_providers)
        ]
        
        self.fundamentalists = [
            FundamentalistTrader(
                self.order_book, 
                initial_fundamental=initial_price * (1 + np.random.normal(0, 0.05))
            ) for _ in range(num_fundamentalists)
        ]
        
        self.chartists = [
            ChartistTrader(self.order_book) for _ in range(num_chartists)
        ]
        
        # Initialize RL agents
        self.rl_agents = []
        if rl_agent_configs:
            for config in rl_agent_configs:
                agent_type = config.get('agent_type', 'I')
                side = config.get('side', 'buy')
                parent_order_size = config.get('parent_order_size', 1000)
                
                self.rl_agents.append(
                    QLearningAgent(
                        self.order_book,
                        agent_type=agent_type,
                        side=side,
                        parent_order_size=parent_order_size,
                        twap_intervals=config.get('twap_intervals', 10),
                        state_size=config.get('state_size', 5),
                        learning_rate=config.get('learning_rate', 0.1),
                        discount_factor=config.get('discount_factor', 0.95),
                        exploration_rate=config.get('exploration_rate', 1.0),
                        exploration_decay=config.get('exploration_decay', 0.995),
                        exploration_min=config.get('exploration_min', 0.01),
                        lambda_r=config.get('lambda_r', 0.5),
                        gamma_r=config.get('gamma_r', 0.01)
                    )
                )
        
        # Data recording
        self.price_history = []
        self.volume_history = []
        self.trade_signs = []  # +1 for buy, -1 for sell
        self.spread_history = []
        self.best_bid_history = []
        self.best_ask_history = []
    
    def run_simulation(self, training=True):
        """Run the market simulation for the specified number of steps."""
        for step in range(self.simulation_steps):
            # Record market state
            book_state = self.order_book.get_book_state()
            self.price_history.append(book_state['mid_price'])
            self.spread_history.append(book_state['spread'])
            self.best_bid_history.append(book_state['best_bid'])
            self.best_ask_history.append(book_state['best_ask'])
            
            # Liquidity providers act
            for lp in self.liquidity_providers:
                lp.act()
            
            # Fundamentalists act
            for fund in self.fundamentalists:
                side, volume = fund.act()
                if side:
                    self.trade_signs.append(1 if side == 'buy' else -1)
                    self.volume_history.append(volume)
            
            # Chartists act
            for chart in self.chartists:
                side, volume = chart.act()
                if side:
                    self.trade_signs.append(1 if side == 'buy' else -1)
                    self.volume_history.append(volume)
            
            # RL agents act
            for agent in self.rl_agents:
                state, action, reward, next_state, executed_volume, avg_price = agent.step(training)
                if executed_volume > 0:
                    self.trade_signs.append(1 if agent.side == 'buy' else -1)
                    self.volume_history.append(executed_volume)
        
        return {
            'price_history': self.price_history,
            'volume_history': self.volume_history,
            'trade_signs': self.trade_signs,
            'spread_history': self.spread_history,
            'best_bid_history': self.best_bid_history,
            'best_ask_history': self.best_ask_history
        }
    
    def reset(self):
        """Reset the simulation for a new run."""
        # Reset order book
        initial_price = self.order_book.mid_price
        self.order_book = OrderBook(initial_price=initial_price)
        
        # Reset agents
        for agent in self.rl_agents:
            agent.order_book = self.order_book
            agent.reset()
        
        for lp in self.liquidity_providers:
            lp.order_book = self.order_book
            lp.orders = []
        
        for fund in self.fundamentalists:
            fund.order_book = self.order_book
        
        for chart in self.chartists:
            chart.order_book = self.order_book
            chart.price_history.clear()
        
        # Reset data recording
        self.price_history = []
        self.volume_history = []
        self.trade_signs = []
        self.spread_history = []
        self.best_bid_history = []
        self.best_ask_history = []

def calculate_acf(data, lags=100):
    """Calculate autocorrelation function up to specified lags."""
    acf = [1]  # Autocorrelation at lag 0 is always 1
    mean = np.mean(data)
    variance = np.var(data)
    
    for lag in range(1, lags + 1):
        numerator = 0
        for i in range(len(data) - lag):
            numerator += (data[i] - mean) * (data[i + lag] - mean)
        
        acf.append(numerator / (len(data) * variance))
    
    return acf

def calculate_trade_sign_acf(trade_signs, lags=100):
    """Calculate autocorrelation function of trade signs."""
    return calculate_acf(trade_signs, lags)

def calculate_absolute_returns_acf(prices, lags=100):
    """Calculate autocorrelation function of absolute returns."""
    returns = np.diff(np.log(prices))
    abs_returns = np.abs(returns)
    return calculate_acf(abs_returns, lags)

def calculate_price_impact(trade_signs, volumes, price_changes, num_bins=10):
    """Calculate price impact curve."""
    # Group trades by volume
    volumes = np.array(volumes)
    trade_signs = np.array(trade_signs)
    price_changes = np.array(price_changes)
    
    # Create volume bins (log scale)
    volume_bins = np.logspace(np.log10(min(volumes)), np.log10(max(volumes)), num_bins)
    
    # Initialize arrays to store results
    buy_impact = np.zeros(num_bins)
    sell_impact = np.zeros(num_bins)
    bin_centers = np.zeros(num_bins)
    
    # Calculate average price impact for each volume bin
    for i in range(num_bins-1):
        bin_mask = (volumes >= volume_bins[i]) & (volumes < volume_bins[i+1])
        bin_centers[i] = np.sqrt(volume_bins[i] * volume_bins[i+1])
        
        # Buy impact
        buy_mask = bin_mask & (trade_signs == 1)
        if np.sum(buy_mask) > 0:
            buy_impact[i] = np.mean(price_changes[buy_mask])
        
        # Sell impact
        sell_mask = bin_mask & (trade_signs == -1)
        if np.sum(sell_mask) > 0:
            sell_impact[i] = np.mean(price_changes[sell_mask])
    
    return bin_centers[:-1], buy_impact[:-1], sell_impact[:-1]

# Set up and run different configurations as in the paper

# Base case (Case 0): No RL agents
print("Running Base Case (Case 0): No RL agents")
sim_case0 = MarketSimulation(
    num_fundamentalists=5,
    num_chartists=5,
    num_liquidity_providers=2,
    rl_agent_configs=None,
    simulation_steps=5000
)
results_case0 = sim_case0.run_simulation(training=False)

# Case 3: Single Type I buying agent
print("Running Case 3: Single Type I buying agent")
sim_case3 = MarketSimulation(
    num_fundamentalists=5,
    num_chartists=5,
    num_liquidity_providers=2,
    rl_agent_configs=[{
        'agent_type': 'I',
        'side': 'buy',
        'parent_order_size': 2000,
        'twap_intervals': 20
    }],
    simulation_steps=5000
)

# Training the RL agent
num_episodes = 100
for episode in tqdm(range(num_episodes), desc="Training Type I buying agent"):
    sim_case3.run_simulation(training=True)
    sim_case3.reset()

# Evaluating the trained agent
sim_case3.reset()
results_case3 = sim_case3.run_simulation(training=False)

# Case 4: Single Type I selling agent
print("Running Case 4: Single Type I selling agent")
sim_case4 = MarketSimulation(
    num_fundamentalists=5,
    num_chartists=5,
    num_liquidity_providers=2,
    rl_agent_configs=[{
        'agent_type': 'I',
        'side': 'sell',
        'parent_order_size': 2000,
        'twap_intervals': 20
    }],
    simulation_steps=5000
)

# Training the RL agent
for episode in tqdm(range(num_episodes), desc="Training Type I selling agent"):
    sim_case4.run_simulation(training=True)
    sim_case4.reset()

# Evaluating the trained agent
sim_case4.reset()
results_case4 = sim_case4.run_simulation(training=False)

# Case 5: Type I buying and selling agents
print("Running Case 5: Type I buying and selling agents")
sim_case5 = MarketSimulation(
    num_fundamentalists=5,
    num_chartists=5,
    num_liquidity_providers=2,
    rl_agent_configs=[
        {
            'agent_type': 'I',
            'side': 'buy',
            'parent_order_size': 1000,
            'twap_intervals': 20
        },
        {
            'agent_type': 'I',
            'side': 'sell',
            'parent_order_size': 1000,
            'twap_intervals': 20
        }
    ],
    simulation_steps=5000
)

# Training the RL agents
for episode in tqdm(range(num_episodes), desc="Training Type I buying and selling agents"):
    sim_case5.run_simulation(training=True)
    sim_case5.reset()

# Evaluating the trained agents
sim_case5.reset()
results_case5 = sim_case5.run_simulation(training=False)

# Case 6: Single Type II buying agent
print("Running Case 6: Single Type II buying agent")
sim_case6 = MarketSimulation(
    num_fundamentalists=5,
    num_chartists=5,
    num_liquidity_providers=2,
    rl_agent_configs=[{
        'agent_type': 'II',
        'side': 'buy',
        'parent_order_size': 2000,
        'twap_intervals': 20
    }],
    simulation_steps=5000
)

# Training the RL agent
for episode in tqdm(range(num_episodes), desc="Training Type II buying agent"):
    sim_case6.run_simulation(training=True)
    sim_case6.reset()

# Evaluating the trained agent
sim_case6.reset()
results_case6 = sim_case6.run_simulation(training=False)

# Case 7: Type II buying and selling agents
print("Running Case 7: Type II buying and selling agents")
sim_case7 = MarketSimulation(
    num_fundamentalists=5,
    num_chartists=5,
    num_liquidity_providers=2,
    rl_agent_configs=[
        {
            'agent_type': 'II',
            'side': 'buy',
            'parent_order_size': 1000,
            'twap_intervals': 20
        },
        {
            'agent_type': 'II',
            'side': 'sell',
            'parent_order_size': 1000,
            'twap_intervals': 20
        }
    ],
    simulation_steps=5000
)

# Training the RL agents
for episode in tqdm(range(num_episodes), desc="Training Type II buying and selling agents"):
    sim_case7.run_simulation(training=True)
    sim_case7.reset()

# Evaluating the trained agents
sim_case7.reset()
results_case7 = sim_case7.run_simulation(training=False)

# Analyze results

# Calculate price changes for price impact
def calculate_price_changes(results):
    mid_prices = np.array(results['price_history'])
    price_changes = np.diff(mid_prices)
    # Pad with a zero at the beginning to match length
    return np.concatenate([[0], price_changes])

price_changes_case0 = calculate_price_changes(results_case0)
price_changes_case3 = calculate_price_changes(results_case3)
price_changes_case4 = calculate_price_changes(results_case4)
price_changes_case5 = calculate_price_changes(results_case5)
price_changes_case6 = calculate_price_changes(results_case6)
price_changes_case7 = calculate_price_changes(results_case7)

# Calculate ACFs
lags = 50

# Trade sign ACFs
acf_trade_signs_case0 = calculate_trade_sign_acf(results_case0['trade_signs'], lags)
acf_trade_signs_case3 = calculate_trade_sign_acf(results_case3['trade_signs'], lags)
acf_trade_signs_case4 = calculate_trade_sign_acf(results_case4['trade_signs'], lags)
acf_trade_signs_case5 = calculate_trade_sign_acf(results_case5['trade_signs'], lags)
acf_trade_signs_case6 = calculate_trade_sign_acf(results_case6['trade_signs'], lags)
acf_trade_signs_case7 = calculate_trade_sign_acf(results_case7['trade_signs'], lags)

# Absolute returns ACFs
acf_abs_returns_case0 = calculate_absolute_returns_acf(results_case0['price_history'], lags)
acf_abs_returns_case3 = calculate_absolute_returns_acf(results_case3['price_history'], lags)
acf_abs_returns_case4 = calculate_absolute_returns_acf(results_case4['price_history'], lags)
acf_abs_returns_case5 = calculate_absolute_returns_acf(results_case5['price_history'], lags)
acf_abs_returns_case6 = calculate_absolute_returns_acf(results_case6['price_history'], lags)
acf_abs_returns_case7 = calculate_absolute_returns_acf(results_case7['price_history'], lags)

# Calculate price impact
bin_centers_case0, buy_impact_case0, sell_impact_case0 = calculate_price_impact(
    results_case0['trade_signs'], results_case0['volume_history'], price_changes_case0)
bin_centers_case3, buy_impact_case3, sell_impact_case3 = calculate_price_impact(
    results_case3['trade_signs'], results_case3['volume_history'], price_changes_case3)
bin_centers_case4, buy_impact_case4, sell_impact_case4 = calculate_price_impact(
    results_case4['trade_signs'], results_case4['volume_history'], price_changes_case4)
bin_centers_case5, buy_impact_case5, sell_impact_case5 = calculate_price_impact(
    results_case5['trade_signs'], results_case5['volume_history'], price_changes_case5)
bin_centers_case6, buy_impact_case6, sell_impact_case6 = calculate_price_impact(
    results_case6['trade_signs'], results_case6['volume_history'], price_changes_case6)
bin_centers_case7, buy_impact_case7, sell_impact_case7 = calculate_price_impact(
    results_case7['trade_signs'], results_case7['volume_history'], price_changes_case7)

# Plot results
plt.figure(figsize=(20, 15))

# Plot 1: Price evolution
plt.subplot(3, 3, 1)
plt.plot(results_case0['price_history'], label='Case 0: No RL')
plt.plot(results_case3['price_history'], label='Case 3: Type I Buy')
plt.plot(results_case4['price_history'], label='Case 4: Type I Sell')
plt.plot(results_case5['price_history'], label='Case 5: Type I Buy+Sell')
plt.plot(results_case6['price_history'], label='Case 6: Type II Buy')
plt.plot(results_case7['price_history'], label='Case 7: Type II Buy+Sell')
plt.title('Price Evolution')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.grid(True)

# Plot 2: Trade Sign ACF (as in Figure 6)
plt.subplot(3, 3, 2)
plt.plot(acf_trade_signs_case0, label='Case 0: No RL')
plt.plot(acf_trade_signs_case3, label='Case 3: Type I Buy')
plt.plot(acf_trade_signs_case4, label='Case 4: Type I Sell')
plt.plot(acf_trade_signs_case5, label='Case 5: Type I Buy+Sell')
plt.plot(acf_trade_signs_case6, label='Case 6: Type II Buy')
plt.plot(acf_trade_signs_case7, label='Case 7: Type II Buy+Sell')
plt.title('Trade Sign ACF')
plt.xlabel('Lag')
plt.ylabel('ACF')
plt.legend()
plt.grid(True)

# Plot 3: Absolute Returns ACF (as in Figure 5)
plt.subplot(3, 3, 3)
plt.plot(acf_abs_returns_case0, label='Case 0: No RL')
plt.plot(acf_abs_returns_case3, label='Case 3: Type I Buy')
plt.plot(acf_abs_returns_case4, label='Case 4: Type I Sell')
plt.plot(acf_abs_returns_case5, label='Case 5: Type I Buy+Sell')
plt.plot(acf_abs_returns_case6, label='Case 6: Type II Buy')
plt.plot(acf_abs_returns_case7, label='Case 7: Type II Buy+Sell')
plt.title('Absolute Returns ACF')
plt.xlabel('Lag')
plt.ylabel('ACF')
plt.legend()
plt.grid(True)

# Plot 4: Buy-initiated Price Impact (as in Figure 4)
plt.subplot(3, 3, 4)
plt.loglog(bin_centers_case0, buy_impact_case0, 'o-', label='Case 0: No RL')
plt.loglog(bin_centers_case3, buy_impact_case3, 'o-', label='Case 3: Type I Buy')
plt.loglog(bin_centers_case5, buy_impact_case5, 'o-', label='Case 5: Type I Buy+Sell')
plt.loglog(bin_centers_case6, buy_impact_case6, 'o-', label='Case 6: Type II Buy')
plt.loglog(bin_centers_case7, buy_impact_case7, 'o-', label='Case 7: Type II Buy+Sell')
plt.title('Buy-Initiated Price Impact')
plt.xlabel('Volume (log)')
plt.ylabel('Price Impact (log)')
plt.legend()
plt.grid(True)

# Plot 5: Sell-initiated Price Impact (as in Figure 4)
plt.subplot(3, 3, 5)
plt.loglog(bin_centers_case0, -sell_impact_case0, 'o-', label='Case 0: No RL')
plt.loglog(bin_centers_case4, -sell_impact_case4, 'o-', label='Case 4: Type I Sell')
plt.loglog(bin_centers_case5, -sell_impact_case5, 'o-', label='Case 5: Type I Buy+Sell')
plt.loglog(bin_centers_case7, -sell_impact_case7, 'o-', label='Case 7: Type II Buy+Sell')
plt.title('Sell-Initiated Price Impact')
plt.xlabel('Volume (log)')
plt.ylabel('Price Impact (log)')
plt.legend()
plt.grid(True)

# Plot 6: Phase Space Plot (simplified version of Figure 7)
plt.subplot(3, 3, 6)
prices_case0 = np.array(results_case0['price_history'])
prices_lag_case0 = np.roll(prices_case0, 10)[:-10]
prices_case0 = prices_case0[:-10]
plt.scatter(prices_case0, prices_lag_case0, s=1, label='Case 0: No RL')

prices_case5 = np.array(results_case5['price_history'])
prices_lag_case5 = np.roll(prices_case5, 10)[:-10]
prices_case5 = prices_case5[:-10]
plt.scatter(prices_case5, prices_lag_case5, s=1, label='Case 5: Type I Buy+Sell')

prices_case6 = np.array(results_case6['price_history'])
prices_lag_case6 = np.roll(prices_case6, 10)[:-10]
prices_case6 = prices_case6[:-10]
plt.scatter(prices_case6, prices_lag_case6, s=1, label='Case 6: Type II Buy')
plt.title('Phase Space Plot')
plt.xlabel('Price(t)')
plt.ylabel('Price(t+10)')
plt.legend()
plt.grid(True)

# Plot 7: Learning Curves
plt.subplot(3, 3, 7)
rewards_case3 = sim_case3.rl_agents[0].rewards
rewards_case4 = sim_case4.rl_agents[0].rewards
rewards_case6 = sim_case6.rl_agents[0].rewards
plt.plot(rewards_case3, label='Type I Buy')
plt.plot(rewards_case4, label='Type I Sell')
plt.plot(rewards_case6, label='Type II Buy')
plt.title('Learning Curves (Rewards)')
plt.xlabel('Step')
plt.ylabel('Reward')
plt.legend()
plt.grid(True)

# Plot 8: Execution Progress
plt.subplot(3, 3, 8)
# Assume we recorded inventory levels during the simulation
inventory_case3 = np.linspace(sim_case3.rl_agents[0].parent_order_size, 0, len(results_case3['price_history']))
inventory_case4 = np.linspace(sim_case4.rl_agents[0].parent_order_size, 0, len(results_case4['price_history']))
inventory_case6 = np.linspace(sim_case6.rl_agents[0].parent_order_size, 0, len(results_case6['price_history']))
plt.plot(inventory_case3, label='Type I Buy')
plt.plot(inventory_case4, label='Type I Sell')
plt.plot(inventory_case6, label='Type II Buy')
plt.title('Execution Progress (Inventory)')
plt.xlabel('Step')
plt.ylabel('Remaining Inventory')
plt.legend()
plt.grid(True)

# Plot 9: Order Book Depth
plt.subplot(3, 3, 9)
bid_volume_case0 = np.array([sum(sim_case0.order_book.bids.values()) for _ in range(len(results_case0['price_history']))])
ask_volume_case0 = np.array([sum(sim_case0.order_book.asks.values()) for _ in range(len(results_case0['price_history']))])
bid_volume_case5 = np.array([sum(sim_case5.order_book.bids.values()) for _ in range(len(results_case5['price_history']))])
ask_volume_case5 = np.array([sum(sim_case5.order_book.asks.values()) for _ in range(len(results_case5['price_history']))])
bid_volume_case7 = np.array([sum(sim_case7.order_book.bids.values()) for _ in range(len(results_case7['price_history']))])
ask_volume_case7 = np.array([sum(sim_case7.order_book.asks.values()) for _ in range(len(results_case7['price_history']))])

plt.plot(bid_volume_case0 / ask_volume_case0, label='Case 0: No RL')
plt.plot(bid_volume_case5 / ask_volume_case5, label='Case 5: Type I Buy+Sell')
plt.plot(bid_volume_case7 / ask_volume_case7, label='Case 7: Type II Buy+Sell')
plt.title('Order Book Imbalance (Bid/Ask Volume Ratio)')
plt.xlabel('Step')
plt.ylabel('Bid/Ask Volume Ratio')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('market_simulation_results.png', dpi=300)
plt.show()

# Print summary statistics
def print_statistics(case_name, results):
    returns = np.diff(np.log(results['price_history']))
    
    print(f"\n{case_name} Statistics:")
    print(f"Mean Price: {np.mean(results['price_history']):.4f}")
    print(f"Price Volatility: {np.std(results['price_history']):.4f}")
    print(f"Mean Return: {np.mean(returns):.6f}")
    print(f"Return Volatility: {np.std(returns):.6f}")
    print(f"Return Kurtosis: {stats.kurtosis(returns):.4f}")
    print(f"Mean Spread: {np.mean(results['spread_history']):.4f}")
    print(f"Mean Trade Size: {np.mean(results['volume_history']):.2f}")
    print(f"Total Trading Volume: {np.sum(results['volume_history']):.0f}")
    
    # Trade sign persistence (ACF at lag 1)
    if len(results['trade_signs']) > 1:
        acf_lag1 = np.corrcoef(results['trade_signs'][:-1], results['trade_signs'][1:])[0, 1]
        print(f"Trade Sign Persistence (ACF lag 1): {acf_lag1:.4f}")
    
    # Absolute return persistence (ACF at lag 1)
    abs_returns = np.abs(returns)
    if len(abs_returns) > 1:
        abs_ret_acf_lag1 = np.corrcoef(abs_returns[:-1], abs_returns[1:])[0, 1]
        print(f"Absolute Return Persistence (ACF lag 1): {abs_ret_acf_lag1:.4f}")

print_statistics("Case 0: No RL agents", results_case0)
print_statistics("Case 3: Type I buying agent", results_case3)
print_statistics("Case 4: Type I selling agent", results_case4)
print_statistics("Case 5: Type I buying and selling agents", results_case5)
print_statistics("Case 6: Type II buying agent", results_case6)
print_statistics("Case 7: Type II buying and selling agents", results_case7)