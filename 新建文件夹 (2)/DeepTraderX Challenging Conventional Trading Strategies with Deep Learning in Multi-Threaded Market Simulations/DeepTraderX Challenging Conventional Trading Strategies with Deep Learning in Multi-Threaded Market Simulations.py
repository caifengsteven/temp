import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from collections import deque
import random
from tqdm import tqdm

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

class LimitOrderBook:
    """A simplified limit order book implementation."""
    
    def __init__(self):
        self.bids = []  # List of (price, quantity) tuples sorted in descending order
        self.asks = []  # List of (price, quantity) tuples sorted in ascending order
        self.trades = []  # List of executed trades
    
    def add_bid(self, price, quantity):
        """Add a bid order to the book."""
        # Check if the bid can be matched with existing asks
        remaining_quantity = quantity
        executed_trades = []
        
        while remaining_quantity > 0 and self.asks and price >= self.asks[0][0]:
            ask_price, ask_quantity = self.asks[0]
            executed_quantity = min(remaining_quantity, ask_quantity)
            remaining_quantity -= executed_quantity
            
            # Record the trade
            executed_trades.append((ask_price, executed_quantity))
            
            # Update or remove the matched ask
            if executed_quantity == ask_quantity:
                self.asks.pop(0)
            else:
                self.asks[0] = (ask_price, ask_quantity - executed_quantity)
        
        # If there's remaining quantity, add it to the bids
        if remaining_quantity > 0:
            self.bids.append((price, remaining_quantity))
            self.bids.sort(key=lambda x: x[0], reverse=True)
        
        return executed_trades
    
    def add_ask(self, price, quantity):
        """Add an ask order to the book."""
        # Check if the ask can be matched with existing bids
        remaining_quantity = quantity
        executed_trades = []
        
        while remaining_quantity > 0 and self.bids and price <= self.bids[0][0]:
            bid_price, bid_quantity = self.bids[0]
            executed_quantity = min(remaining_quantity, bid_quantity)
            remaining_quantity -= executed_quantity
            
            # Record the trade
            executed_trades.append((bid_price, executed_quantity))
            
            # Update or remove the matched bid
            if executed_quantity == bid_quantity:
                self.bids.pop(0)
            else:
                self.bids[0] = (bid_price, bid_quantity - executed_quantity)
        
        # If there's remaining quantity, add it to the asks
        if remaining_quantity > 0:
            self.asks.append((price, remaining_quantity))
            self.asks.sort(key=lambda x: x[0])
        
        return executed_trades
    
    def get_best_bid(self):
        """Get the best (highest) bid price and quantity."""
        return self.bids[0] if self.bids else (None, None)
    
    def get_best_ask(self):
        """Get the best (lowest) ask price and quantity."""
        return self.asks[0] if self.asks else (None, None)
    
    def get_midprice(self):
        """Get the midprice of the order book."""
        best_bid_price, _ = self.get_best_bid()
        best_ask_price, _ = self.get_best_ask()
        
        if best_bid_price is None or best_ask_price is None:
            return None
        
        return (best_bid_price + best_ask_price) / 2
    
    def get_microprice(self):
        """Get the microprice (volume-weighted midprice) of the order book."""
        best_bid_price, best_bid_qty = self.get_best_bid()
        best_ask_price, best_ask_qty = self.get_best_ask()
        
        if best_bid_price is None or best_ask_price is None:
            return None
        
        total_qty = best_bid_qty + best_ask_qty
        return (best_bid_price * best_ask_qty + best_ask_price * best_bid_qty) / total_qty
    
    def get_spread(self):
        """Get the spread of the order book."""
        best_bid_price, _ = self.get_best_bid()
        best_ask_price, _ = self.get_best_ask()
        
        if best_bid_price is None or best_ask_price is None:
            return None
        
        return best_ask_price - best_bid_price
    
    def get_imbalance(self):
        """Get the imbalance of the order book."""
        bid_volume = sum(qty for _, qty in self.bids)
        ask_volume = sum(qty for _, qty in self.asks)
        
        if bid_volume + ask_volume == 0:
            return 0
        
        return (bid_volume - ask_volume) / (bid_volume + ask_volume)
    
    def get_total_quotes(self):
        """Get the total number of quotes in the order book."""
        return len(self.bids) + len(self.asks)
    
    def __str__(self):
        """String representation of the order book."""
        bid_str = "\n".join([f"Bid: {price:.2f} x {qty}" for price, qty in self.bids])
        ask_str = "\n".join([f"Ask: {price:.2f} x {qty}" for price, qty in self.asks])
        return f"Bids:\n{bid_str}\n\nAsks:\n{ask_str}"


class MarketSimulator:
    """A simplified market simulator that simulates trading based on a continuous double auction."""
    
    def __init__(self, n_traders=10, time_steps=1000, initial_price=100.0, volatility=0.01):
        self.lob = LimitOrderBook()
        self.n_traders = n_traders
        self.time_steps = time_steps
        self.current_step = 0
        self.initial_price = initial_price
        self.volatility = volatility
        self.price_history = [initial_price]
        self.market_data = []
        
        # Initialize the LOB with some random orders
        for _ in range(10):
            price = initial_price * (1 - random.uniform(0, 0.05))
            self.lob.add_bid(price, random.randint(1, 10))
            
            price = initial_price * (1 + random.uniform(0, 0.05))
            self.lob.add_ask(price, random.randint(1, 10))
    
    def step(self, actions=None):
        """Execute one step in the market simulation."""
        self.current_step += 1
        
        # If no actions are provided, generate random orders
        if actions is None:
            actions = []
            for _ in range(random.randint(1, 5)):  # Random number of orders
                is_bid = random.choice([True, False])
                price = self.price_history[-1] * (1 + random.uniform(-0.02, 0.02))
                quantity = random.randint(1, 10)
                
                actions.append((is_bid, price, quantity))
        
        # Process each order
        for is_bid, price, quantity in actions:
            if is_bid:
                trades = self.lob.add_bid(price, quantity)
            else:
                trades = self.lob.add_ask(price, quantity)
            
            # Record the trades
            for trade_price, trade_quantity in trades:
                self.lob.trades.append((self.current_step, trade_price, trade_quantity))
        
        # Record market data
        midprice = self.lob.get_midprice() or self.price_history[-1]
        self.price_history.append(midprice)
        
        # Calculate features for the market state
        best_bid_price, best_bid_qty = self.lob.get_best_bid() or (midprice * 0.99, 0)
        best_ask_price, best_ask_qty = self.lob.get_best_ask() or (midprice * 1.01, 0)
        
        microprice = self.lob.get_microprice() or midprice
        imbalance = self.lob.get_imbalance()
        spread = self.lob.get_spread() or (midprice * 0.02)
        total_quotes = self.lob.get_total_quotes()
        
        # Estimate equilibrium price (P*) - simplified
        equilibrium_price = midprice
        
        # Smith's alpha - simplified
        alpha = abs((midprice - equilibrium_price) / equilibrium_price)
        
        # Record market data
        market_state = {
            'time': self.current_step,
            'last_trade_type': 'bid' if is_bid else 'ask',
            'last_trade_price': price,
            'midprice': midprice,
            'microprice': microprice,
            'imbalance': imbalance,
            'spread': spread,
            'best_bid': best_bid_price,
            'best_ask': best_ask_price,
            'time_since_last_trade': 1,  # Simplified
            'total_quotes': total_quotes,
            'equilibrium_price': equilibrium_price,
            'alpha': alpha
        }
        
        self.market_data.append(market_state)
        
        return market_state
    
    def run_simulation(self):
        """Run the full market simulation."""
        for _ in range(self.time_steps):
            self.step()
        
        return pd.DataFrame(self.market_data)


class DeepTraderX:
    """A simplified implementation of the DeepTraderX trading agent."""
    
    def __init__(self, input_features=14, lstm_units=10, dense_units=[5, 3]):
        self.input_features = input_features
        self.lstm_units = lstm_units
        self.dense_units = dense_units
        self.model = self._build_model()
        self.scaler = MinMaxScaler()
        self.memory = deque(maxlen=100)  # Store recent market states
        
    def _build_model(self):
        """Build the neural network model for DeepTraderX."""
        model = Sequential()
        model.add(LSTM(self.lstm_units, input_shape=(None, self.input_features), return_sequences=False))
        
        for units in self.dense_units:
            model.add(Dense(units, activation='relu'))
        
        model.add(Dense(1, activation='linear'))  # Output layer for price prediction
        
        model.compile(optimizer=Adam(learning_rate=1.5e-5), loss='mse')
        return model
    
    def preprocess_data(self, market_data):
        """Preprocess market data for training or prediction."""
        features = pd.DataFrame(market_data)
        
        # Convert categorical variable to numerical
        features['last_trade_type'] = features['last_trade_type'].map({'bid': 0, 'ask': 1})
        
        # Extract features as in the paper
        X = features[['time', 'last_trade_type', 'last_trade_price', 'midprice', 
                      'microprice', 'imbalance', 'spread', 'best_bid', 'best_ask', 
                      'time_since_last_trade', 'total_quotes', 'equilibrium_price', 'alpha']]
        
        # Target variable - trade price
        y = features['last_trade_price']
        
        # Normalize features
        X_scaled = self.scaler.fit_transform(X)
        
        # Reshape for LSTM input [samples, time steps, features]
        X_reshaped = X_scaled.reshape(X_scaled.shape[0], 1, X_scaled.shape[1])
        
        return X_reshaped, y
    
    def train(self, market_data, epochs=10, batch_size=32, validation_split=0.2):
        """Train the model on historical market data."""
        X, y = self.preprocess_data(market_data)
        
        history = self.model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=1
        )
        
        return history
    
    def predict_price(self, market_state):
        """Predict the price to quote based on the current market state."""
        # Add market state to memory
        self.memory.append(market_state)
        
        if len(self.memory) < 1:
            # Not enough data to make a prediction
            return market_state['midprice']
        
        # Preprocess the most recent market state
        X, _ = self.preprocess_data([market_state])
        
        # Make prediction
        predicted_price = self.model.predict(X)[0][0]
        
        # Ensure the predicted price is reasonable
        midprice = market_state['midprice']
        best_bid = market_state['best_bid']
        best_ask = market_state['best_ask']
        
        # Fallback mechanism as mentioned in the paper
        if predicted_price <= best_bid or predicted_price >= best_ask:
            # Quote just above the best bid or just below the best ask
            if random.choice([True, False]):
                predicted_price = best_bid + 0.01
            else:
                predicted_price = best_ask - 0.01
        
        return predicted_price
    
    def place_order(self, market_state, is_buyer, limit_price=None):
        """
        Place an order in the market based on the current market state.
        
        Parameters:
        - market_state: Dictionary containing the current market state
        - is_buyer: Boolean indicating if the agent is a buyer (True) or seller (False)
        - limit_price: The agent's limit price (maximum buy or minimum sell price)
        
        Returns:
        - Tuple (is_bid, price, quantity) representing the order
        """
        predicted_price = self.predict_price(market_state)
        
        # Check if the predicted price is profitable given the limit price
        if limit_price is not None:
            if is_buyer and predicted_price > limit_price:
                # Cannot buy at a price higher than the limit price
                predicted_price = limit_price
            elif not is_buyer and predicted_price < limit_price:
                # Cannot sell at a price lower than the limit price
                predicted_price = limit_price
        
        # Determine if it's a bid or ask order
        is_bid = is_buyer
        
        # Determine quantity (simplified)
        quantity = random.randint(1, 5)
        
        return (is_bid, predicted_price, quantity)


class ZeroIntelligenceConstrained:
    """Implementation of the Zero Intelligence Constrained (ZIC) trading strategy."""
    
    def __init__(self):
        pass
    
    def place_order(self, market_state, is_buyer, limit_price):
        """
        Place an order based on the ZIC strategy.
        
        ZIC places random orders constrained by the limit price.
        """
        if is_buyer:
            # For buyers, generate a random price between minimum price and limit price
            min_price = 0.01  # Minimum possible price
            price = random.uniform(min_price, limit_price)
        else:
            # For sellers, generate a random price between limit price and maximum price
            max_price = market_state['midprice'] * 2  # Arbitrary high price
            price = random.uniform(limit_price, max_price)
        
        quantity = random.randint(1, 5)
        is_bid = is_buyer
        
        return (is_bid, price, quantity)


class ZeroIntelligencePlus:
    """Implementation of the Zero Intelligence Plus (ZIP) trading strategy."""
    
    def __init__(self, learning_rate=0.1, momentum=0.2, initial_profit_margin=0.05):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.profit_margin = initial_profit_margin
        self.last_update = 0
    
    def place_order(self, market_state, is_buyer, limit_price):
        """
        Place an order based on the ZIP strategy.
        
        ZIP adjusts its profit margin based on market conditions.
        """
        # Extract market information
        best_bid = market_state['best_bid']
        best_ask = market_state['best_ask']
        midprice = market_state['midprice']
        
        # Update profit margin based on market conditions (simplified implementation)
        if is_buyer:
            # Buyer wants to increase profit margin when prices are falling
            if 'last_midprice' in market_state and midprice < market_state['last_midprice']:
                target_margin = max(0.01, self.profit_margin - self.learning_rate)
            else:
                # Otherwise, decrease margin to be more competitive
                target_margin = min(0.2, self.profit_margin + self.learning_rate)
        else:
            # Seller wants to increase profit margin when prices are rising
            if 'last_midprice' in market_state and midprice > market_state['last_midprice']:
                target_margin = max(0.01, self.profit_margin - self.learning_rate)
            else:
                # Otherwise, decrease margin to be more competitive
                target_margin = min(0.2, self.profit_margin + self.learning_rate)
        
        # Apply momentum
        self.profit_margin = self.profit_margin * self.momentum + target_margin * (1 - self.momentum)
        
        # Calculate quote price
        if is_buyer:
            price = limit_price * (1 - self.profit_margin)
            price = min(price, best_ask - 0.01) if best_ask is not None else price
        else:
            price = limit_price * (1 + self.profit_margin)
            price = max(price, best_bid + 0.01) if best_bid is not None else price
        
        quantity = random.randint(1, 5)
        is_bid = is_buyer
        
        return (is_bid, price, quantity)


def run_balanced_group_test(trader1, trader2, n_traders=20, time_steps=500, trials=10):
    """
    Run a balanced group test between two trading strategies.
    
    Parameters:
    - trader1: First trading strategy
    - trader2: Second trading strategy
    - n_traders: Total number of traders (should be even)
    - time_steps: Number of time steps in each trial
    - trials: Number of trials to run
    
    Returns:
    - DataFrame with profit per trader results for each trial
    """
    results = []
    
    for trial in tqdm(range(trials), desc="Running BGT"):
        # Initialize market simulator
        simulator = MarketSimulator(n_traders=n_traders, time_steps=time_steps)
        
        # Initialize profits for each trader type
        trader1_profits = 0
        trader2_profits = 0
        
        # Count of each trader type
        n_trader1 = n_traders // 2
        n_trader2 = n_traders - n_trader1
        
        # Generate limit prices for each trader
        limit_prices = []
        for i in range(n_traders):
            # Half buyers, half sellers
            is_buyer = i < n_traders // 2
            
            if is_buyer:
                # Buyer's limit price (maximum willing to pay)
                limit_price = simulator.initial_price * (1 + random.uniform(0, 0.1))
            else:
                # Seller's limit price (minimum willing to accept)
                limit_price = simulator.initial_price * (1 - random.uniform(0, 0.1))
            
            limit_prices.append((is_buyer, limit_price))
        
        # Run the simulation
        for _ in range(time_steps):
            # Get current market state
            if simulator.market_data:
                market_state = simulator.market_data[-1]
            else:
                # Initial market state
                market_state = {
                    'time': 0,
                    'last_trade_type': 'bid' if random.choice([True, False]) else 'ask',
                    'last_trade_price': simulator.initial_price,
                    'midprice': simulator.initial_price,
                    'microprice': simulator.initial_price,
                    'imbalance': 0,
                    'spread': simulator.initial_price * 0.02,
                    'best_bid': simulator.initial_price * 0.99,
                    'best_ask': simulator.initial_price * 1.01,
                    'time_since_last_trade': 1,
                    'total_quotes': 20,
                    'equilibrium_price': simulator.initial_price,
                    'alpha': 0
                }
            
            # Collect orders from each trader
            actions = []
            
            for i in range(n_traders):
                is_buyer, limit_price = limit_prices[i]
                
                # Determine which trader strategy to use
                if i < n_trader1:
                    trader = trader1
                else:
                    trader = trader2
                
                # Get order from trader
                order = trader.place_order(market_state, is_buyer, limit_price)
                actions.append(order)
            
            # Execute the orders
            simulator.step(actions)
            
            # Calculate profits (simplified)
            for i, (is_buyer, limit_price) in enumerate(limit_prices):
                # Get the trade price from the most recent market state
                if simulator.market_data:
                    trade_price = simulator.market_data[-1]['last_trade_price']
                else:
                    trade_price = simulator.initial_price
                
                # Calculate profit
                if is_buyer:
                    profit = limit_price - trade_price
                else:
                    profit = trade_price - limit_price
                
                # Add profit to the appropriate trader type
                if i < n_trader1:
                    trader1_profits += max(0, profit)
                else:
                    trader2_profits += max(0, profit)
        
        # Calculate profit per trader for each type
        trader1_ppt = trader1_profits / n_trader1
        trader2_ppt = trader2_profits / n_trader2
        
        results.append({
            'trial': trial,
            'trader1_ppt': trader1_ppt,
            'trader2_ppt': trader2_ppt
        })
    
    return pd.DataFrame(results)


def run_one_to_many_test(trader1, trader2, n_traders=20, time_steps=500, trials=10):
    """
    Run a one-to-many test where one trader (defector) competes against many traders of the other type.
    
    Parameters:
    - trader1: Defector trading strategy
    - trader2: Majority trading strategy
    - n_traders: Total number of traders (should be even)
    - time_steps: Number of time steps in each trial
    - trials: Number of trials to run
    
    Returns:
    - DataFrame with profit per trader results for each trial
    """
    results = []
    
    for trial in tqdm(range(trials), desc="Running OTM"):
        # Initialize market simulator
        simulator = MarketSimulator(n_traders=n_traders, time_steps=time_steps)
        
        # Initialize profits for each trader type
        trader1_profits = 0  # Defector
        trader2_profits = 0  # Majority
        
        # Count of each trader type
        n_trader1 = 2  # One buyer, one seller
        n_trader2 = n_traders - n_trader1
        
        # Generate limit prices for each trader
        limit_prices = []
        for i in range(n_traders):
            # Half buyers, half sellers
            is_buyer = i < n_traders // 2
            
            if is_buyer:
                # Buyer's limit price (maximum willing to pay)
                limit_price = simulator.initial_price * (1 + random.uniform(0, 0.1))
            else:
                # Seller's limit price (minimum willing to accept)
                limit_price = simulator.initial_price * (1 - random.uniform(0, 0.1))
            
            limit_prices.append((is_buyer, limit_price))
        
        # Run the simulation
        for _ in range(time_steps):
            # Get current market state
            if simulator.market_data:
                market_state = simulator.market_data[-1]
            else:
                # Initial market state
                market_state = {
                    'time': 0,
                    'last_trade_type': 'bid' if random.choice([True, False]) else 'ask',
                    'last_trade_price': simulator.initial_price,
                    'midprice': simulator.initial_price,
                    'microprice': simulator.initial_price,
                    'imbalance': 0,
                    'spread': simulator.initial_price * 0.02,
                    'best_bid': simulator.initial_price * 0.99,
                    'best_ask': simulator.initial_price * 1.01,
                    'time_since_last_trade': 1,
                    'total_quotes': 20,
                    'equilibrium_price': simulator.initial_price,
                    'alpha': 0
                }
            
            # Collect orders from each trader
            actions = []
            
            for i in range(n_traders):
                is_buyer, limit_price = limit_prices[i]
                
                # Determine which trader strategy to use
                # Trader1 (defector) is the first buyer and first seller
                if i == 0 or i == n_traders // 2:
                    trader = trader1
                else:
                    trader = trader2
                
                # Get order from trader
                order = trader.place_order(market_state, is_buyer, limit_price)
                actions.append(order)
            
            # Execute the orders
            simulator.step(actions)
            
            # Calculate profits (simplified)
            for i, (is_buyer, limit_price) in enumerate(limit_prices):
                # Get the trade price from the most recent market state
                if simulator.market_data:
                    trade_price = simulator.market_data[-1]['last_trade_price']
                else:
                    trade_price = simulator.initial_price
                
                # Calculate profit
                if is_buyer:
                    profit = limit_price - trade_price
                else:
                    profit = trade_price - limit_price
                
                # Add profit to the appropriate trader type
                if i == 0 or i == n_traders // 2:  # Trader1 (defector)
                    trader1_profits += max(0, profit)
                else:  # Trader2 (majority)
                    trader2_profits += max(0, profit)
        
        # Calculate profit per trader for each type
        trader1_ppt = trader1_profits / n_trader1
        trader2_ppt = trader2_profits / n_trader2
        
        results.append({
            'trial': trial,
            'trader1_ppt': trader1_ppt,
            'trader2_ppt': trader2_ppt
        })
    
    return pd.DataFrame(results)


def plot_results(results, trader1_name, trader2_name, test_type):
    """
    Plot the results of the experiments.
    
    Parameters:
    - results: DataFrame with experiment results
    - trader1_name: Name of the first trader strategy
    - trader2_name: Name of the second trader strategy
    - test_type: Type of test (BGT or OTM)
    """
    # Box plot
    plt.figure(figsize=(10, 6))
    data = [results['trader1_ppt'], results['trader2_ppt']]
    labels = [trader1_name, trader2_name]
    
    plt.boxplot(data, labels=labels)
    plt.title(f'{test_type} - {trader1_name} vs {trader2_name}')
    plt.ylabel('Profit per Trader')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(results['trader2_ppt'], results['trader1_ppt'])
    plt.plot([0, results['trader2_ppt'].max()], [0, results['trader2_ppt'].max()], 'r--')
    plt.xlabel(f'{trader2_name} Profit per Trader')
    plt.ylabel(f'{trader1_name} Profit per Trader')
    plt.title(f'{test_type} - {trader1_name} vs {trader2_name}')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Calculate statistics
    trader1_mean = results['trader1_ppt'].mean()
    trader2_mean = results['trader2_ppt'].mean()
    
    print(f"Average {trader1_name} PPT: {trader1_mean:.4f}")
    print(f"Average {trader2_name} PPT: {trader2_mean:.4f}")
    
    # Perform Wilcoxon signed-rank test
    from scipy.stats import wilcoxon
    
    w, p = wilcoxon(results['trader1_ppt'], results['trader2_ppt'])
    print(f"Wilcoxon signed-rank test: p-value = {p:.4f}")
    
    if p < 0.05:
        if trader1_mean > trader2_mean:
            print(f"{trader1_name} significantly outperforms {trader2_name}")
        else:
            print(f"{trader2_name} significantly outperforms {trader1_name}")
    else:
        print(f"No significant difference between {trader1_name} and {trader2_name}")


# Generate training data
print("Generating training data...")
simulator = MarketSimulator(time_steps=10000)
training_data = simulator.run_simulation()

# Initialize and train DeepTraderX
print("Training DeepTraderX...")
dtx = DeepTraderX()
history = dtx.train(simulator.market_data, epochs=20, batch_size=32)

# Plot training history
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.show()

# Initialize other trading strategies
zic = ZeroIntelligenceConstrained()
zip_trader = ZeroIntelligencePlus()

# Run experiments

# 1. DeepTraderX vs ZIC - Balanced Group Test
print("\nRunning BGT: DeepTraderX vs ZIC")
bgt_dtx_zic = run_balanced_group_test(dtx, zic, n_traders=20, time_steps=500, trials=20)
plot_results(bgt_dtx_zic, 'DeepTraderX', 'ZIC', 'BGT')

# 2. DeepTraderX vs ZIC - One to Many Test
print("\nRunning OTM: DeepTraderX vs ZIC")
otm_dtx_zic = run_one_to_many_test(dtx, zic, n_traders=20, time_steps=500, trials=20)
plot_results(otm_dtx_zic, 'DeepTraderX', 'ZIC', 'OTM')

# 3. DeepTraderX vs ZIP - Balanced Group Test
print("\nRunning BGT: DeepTraderX vs ZIP")
bgt_dtx_zip = run_balanced_group_test(dtx, zip_trader, n_traders=20, time_steps=500, trials=20)
plot_results(bgt_dtx_zip, 'DeepTraderX', 'ZIP', 'BGT')

# 4. DeepTraderX vs ZIP - One to Many Test
print("\nRunning OTM: DeepTraderX vs ZIP")
otm_dtx_zip = run_one_to_many_test(dtx, zip_trader, n_traders=20, time_steps=500, trials=20)
plot_results(otm_dtx_zip, 'DeepTraderX', 'ZIP', 'OTM')