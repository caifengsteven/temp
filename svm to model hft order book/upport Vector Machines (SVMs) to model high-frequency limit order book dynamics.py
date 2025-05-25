import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import svm
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
import random
import time
from datetime import datetime
from collections import deque

# Set random seed for reproducibility
np.random.seed(42)

class OrderBookSimulator:
    """
    Simulates a limit order book with realistic high-frequency dynamics
    """
    def __init__(self, initial_price=100.0, initial_spread=0.1, 
                 levels=10, volatility=0.01, intensity_rates=None,
                 cancel_rate=0.3, market_order_rate=0.2):
        """
        Initialize a simulated order book
        
        Parameters:
        -----------
        initial_price : float
            Initial mid-price
        initial_spread : float
            Initial spread between best bid and ask
        levels : int
            Number of price levels to maintain on each side
        volatility : float
            Controls how much prices move
        intensity_rates : dict
            Rates for different event types
        cancel_rate : float
            Rate of cancellation events
        market_order_rate : float
            Rate of market order events
        """
        self.mid_price = initial_price
        self.spread = initial_spread
        self.levels = levels
        self.volatility = volatility
        
        # Default intensity rates if none provided
        if intensity_rates is None:
            self.intensity_rates = {
                'limit_bid': 0.3,  # Limit bid order submission rate
                'limit_ask': 0.3,  # Limit ask order submission rate
                'market_bid': market_order_rate/2,  # Market bid order rate
                'market_ask': market_order_rate/2,  # Market ask order rate
                'cancel_bid': cancel_rate/2,  # Limit bid order cancellation rate
                'cancel_ask': cancel_rate/2,  # Limit ask order cancellation rate
            }
        else:
            self.intensity_rates = intensity_rates
            
        # Initialize order book structure
        self.ask_prices = np.array([self.mid_price + self.spread/2 + i*self.spread/2 for i in range(levels)])
        self.bid_prices = np.array([self.mid_price - self.spread/2 - i*self.spread/2 for i in range(levels)])
        
        # Initialize volumes with random values
        self.ask_volumes = np.random.randint(1, 100, size=levels)
        self.bid_volumes = np.random.randint(1, 100, size=levels)
        
        # Message book for recording events
        self.message_book = []
        self.current_time = 0.0
        
        # History of order book states
        self.order_book_history = deque(maxlen=10000)
        
        # Store the current state of the order book
        self.save_order_book_state()
    
    def save_order_book_state(self):
        """Save the current state of the order book to history"""
        state = {
            'time': self.current_time,
            'ask_prices': self.ask_prices.copy(),
            'ask_volumes': self.ask_volumes.copy(),
            'bid_prices': self.bid_prices.copy(),
            'bid_volumes': self.bid_volumes.copy(),
            'mid_price': self.mid_price,
            'spread': self.spread
        }
        self.order_book_history.append(state)
    
    def generate_event(self):
        """Generate a new order book event"""
        # Increment time by a random amount
        self.current_time += np.random.exponential(1.0 / sum(self.intensity_rates.values()))
        
        # Decide which type of event
        event_type = np.random.choice(list(self.intensity_rates.keys()), 
                                      p=[rate/sum(self.intensity_rates.values()) 
                                         for rate in self.intensity_rates.values()])
        
        if event_type == 'limit_bid':
            # Submit a new limit bid order
            price_level = np.random.randint(0, self.levels)
            volume = np.random.randint(1, 100)
            
            # Sometimes place aggressive limit orders close to the spread
            if np.random.random() < 0.2:  # 20% chance of aggressive bid
                # Place bid higher, possibly crossing the spread
                price_adjustment = np.random.uniform(0, self.spread * 1.5)
                self.bid_prices[0] = self.mid_price - self.spread/2 + price_adjustment
                price_level = 0
            
            self.bid_volumes[price_level] += volume
            
            # Record the event
            self.message_book.append({
                'time': self.current_time,
                'price': self.bid_prices[price_level],
                'volume': volume,
                'event_type': 'Submission',
                'direction': 'bid'
            })
            
        elif event_type == 'limit_ask':
            # Submit a new limit ask order
            price_level = np.random.randint(0, self.levels)
            volume = np.random.randint(1, 100)
            
            # Sometimes place aggressive limit orders close to the spread
            if np.random.random() < 0.2:  # 20% chance of aggressive ask
                # Place ask lower, possibly crossing the spread
                price_adjustment = np.random.uniform(0, self.spread * 1.5)
                self.ask_prices[0] = self.mid_price + self.spread/2 - price_adjustment
                price_level = 0
            
            self.ask_volumes[price_level] += volume
            
            # Record the event
            self.message_book.append({
                'time': self.current_time,
                'price': self.ask_prices[price_level],
                'volume': volume,
                'event_type': 'Submission',
                'direction': 'ask'
            })
            
        elif event_type == 'market_bid':
            # Execute a market order to buy
            volume = np.random.randint(1, 50)
            remaining_volume = volume
            level = 0
            
            while remaining_volume > 0 and level < self.levels:
                if self.ask_volumes[level] > 0:
                    executed_volume = min(remaining_volume, self.ask_volumes[level])
                    self.ask_volumes[level] -= executed_volume
                    remaining_volume -= executed_volume
                    
                    # Record the event
                    self.message_book.append({
                        'time': self.current_time,
                        'price': self.ask_prices[level],
                        'volume': executed_volume,
                        'event_type': 'Execution',
                        'direction': 'ask'
                    })
                    
                level += 1
                
            # Update the mid price and spread after execution
            self.update_prices()
            
        elif event_type == 'market_ask':
            # Execute a market order to sell
            volume = np.random.randint(1, 50)
            remaining_volume = volume
            level = 0
            
            while remaining_volume > 0 and level < self.levels:
                if self.bid_volumes[level] > 0:
                    executed_volume = min(remaining_volume, self.bid_volumes[level])
                    self.bid_volumes[level] -= executed_volume
                    remaining_volume -= executed_volume
                    
                    # Record the event
                    self.message_book.append({
                        'time': self.current_time,
                        'price': self.bid_prices[level],
                        'volume': executed_volume,
                        'event_type': 'Execution',
                        'direction': 'bid'
                    })
                    
                level += 1
                
            # Update the mid price and spread after execution
            self.update_prices()
            
        elif event_type == 'cancel_bid':
            # Cancel a limit bid order
            non_empty_levels = [i for i in range(self.levels) if self.bid_volumes[i] > 0]
            if non_empty_levels:
                level = np.random.choice(non_empty_levels)
                volume = min(np.random.randint(1, 50), self.bid_volumes[level])
                self.bid_volumes[level] -= volume
                
                # Record the event
                self.message_book.append({
                    'time': self.current_time,
                    'price': self.bid_prices[level],
                    'volume': volume,
                    'event_type': 'Cancellation',
                    'direction': 'bid'
                })
            
        elif event_type == 'cancel_ask':
            # Cancel a limit ask order
            non_empty_levels = [i for i in range(self.levels) if self.ask_volumes[i] > 0]
            if non_empty_levels:
                level = np.random.choice(non_empty_levels)
                volume = min(np.random.randint(1, 50), self.ask_volumes[level])
                self.ask_volumes[level] -= volume
                
                # Record the event
                self.message_book.append({
                    'time': self.current_time,
                    'price': self.ask_prices[level],
                    'volume': volume,
                    'event_type': 'Cancellation',
                    'direction': 'ask'
                })
        
        # More frequent and larger price moves to create spread crossing opportunities
        if np.random.random() < 0.2:  # Increased from 0.05 to 0.2
            price_change = np.random.normal(0, self.volatility * 3)  # Tripled volatility
            self.mid_price += price_change
            self.update_prices(with_shift=True)
        
        # Sometimes artificially create spread crossing by shifting the bid or ask price
        if np.random.random() < 0.05:  # 5% chance
            if np.random.random() < 0.5:
                # Upward spread crossing (bid > ask)
                self.bid_prices[0] = self.ask_prices[0] + np.random.uniform(0.01, 0.05)
            else:
                # Downward spread crossing (ask < bid)
                self.ask_prices[0] = self.bid_prices[0] - np.random.uniform(0.01, 0.05)
        
        # Save the new state
        self.save_order_book_state()
        
        return self.message_book[-1]
    
    def update_prices(self, with_shift=False):
        """Update prices based on current state"""
        # Find first non-zero volumes
        ask_indices = np.where(self.ask_volumes > 0)[0]
        bid_indices = np.where(self.bid_volumes > 0)[0]
        
        ask_level = ask_indices[0] if len(ask_indices) > 0 else 0
        bid_level = bid_indices[0] if len(bid_indices) > 0 else 0
        
        if with_shift:
            # Recalculate all price levels based on new mid price
            self.ask_prices = np.array([self.mid_price + self.spread/2 + i*self.spread/2 for i in range(self.levels)])
            self.bid_prices = np.array([self.mid_price - self.spread/2 - i*self.spread/2 for i in range(self.levels)])
        else:
            # Update mid price and spread based on best bid and ask
            best_ask = self.ask_prices[ask_level]
            best_bid = self.bid_prices[bid_level]
            self.mid_price = (best_ask + best_bid) / 2
            self.spread = best_ask - best_bid
    
    def get_order_book_state(self, idx=-1):
        """Get the state of the order book at a specific index"""
        return self.order_book_history[idx]
    
    def generate_events(self, num_events):
        """Generate multiple events"""
        for _ in range(num_events):
            self.generate_event()
        
        return self.message_book[-num_events:]

class LOBFeatureExtractor:
    """
    Extract features from limit order book data for machine learning
    """
    def __init__(self, lob_simulator, look_back=10):
        """
        Initialize the feature extractor
        
        Parameters:
        -----------
        lob_simulator : OrderBookSimulator
            The simulator object containing order book data
        look_back : int
            Number of historical states to consider for time-sensitive features
        """
        self.lob_simulator = lob_simulator
        self.look_back = look_back
        self.history_buffer = deque(maxlen=look_back)
        self.feature_dim = None  # Will store the expected feature dimension
    
    def get_expected_feature_dim(self):
        """Calculate the expected dimension of the feature vector"""
        # Basic features: price and volume at each level
        basic_dim = self.lob_simulator.levels * 4
        
        # Time-insensitive features
        # Bid-ask spreads and mid-prices at each level
        time_insensitive_dim = self.lob_simulator.levels * 2
        # Price differences between levels
        time_insensitive_dim += (self.lob_simulator.levels - 1) * 2
        # Price range
        time_insensitive_dim += 2
        # Mean prices and volumes
        time_insensitive_dim += 4
        # Accumulated differences
        time_insensitive_dim += 2
        
        # Time-sensitive features
        # Price and volume derivatives
        time_sensitive_dim = self.lob_simulator.levels * 4
        # Average intensity of each event type
        time_sensitive_dim += 6
        # Relative intensity indicators
        time_sensitive_dim += 4
        
        total_dim = basic_dim + time_insensitive_dim + time_sensitive_dim
        return total_dim
    
    def extract_basic_features(self, state):
        """
        Extract basic features from order book state (prices and volumes)
        """
        features = []
        
        # Price and volume at each level
        for i in range(self.lob_simulator.levels):
            features.append(float(state['ask_prices'][i]))
            features.append(float(state['ask_volumes'][i]))
            features.append(float(state['bid_prices'][i]))
            features.append(float(state['bid_volumes'][i]))
        
        return features
    
    def extract_time_insensitive_features(self, state):
        """
        Extract time-insensitive features (derived from a single state)
        """
        features = []
        
        # Bid-ask spreads and mid-prices at each level
        for i in range(self.lob_simulator.levels):
            features.append(float(state['ask_prices'][i] - state['bid_prices'][i]))
            features.append(float((state['ask_prices'][i] + state['bid_prices'][i]) / 2))
        
        # Price differences between levels
        for i in range(self.lob_simulator.levels - 1):
            features.append(float(abs(state['ask_prices'][i+1] - state['ask_prices'][i])))
            features.append(float(abs(state['bid_prices'][i+1] - state['bid_prices'][i])))
        
        # Price range (difference between highest and lowest prices)
        features.append(float(state['ask_prices'][-1] - state['ask_prices'][0]))
        features.append(float(state['bid_prices'][0] - state['bid_prices'][-1]))
        
        # Mean prices and volumes
        features.append(float(np.mean(state['ask_prices'])))
        features.append(float(np.mean(state['bid_prices'])))
        features.append(float(np.mean(state['ask_volumes'])))
        features.append(float(np.mean(state['bid_volumes'])))
        
        # Accumulated differences
        features.append(float(np.sum(state['ask_prices'] - state['bid_prices'])))
        features.append(float(np.sum(state['ask_volumes'] - state['bid_volumes'])))
        
        return features
    
    def extract_time_sensitive_features(self, current_state):
        """
        Extract time-sensitive features (using historical data)
        """
        features = []
        
        # Add current state to history buffer
        self.history_buffer.append(current_state)
        
        # If we don't have enough history, return zeros
        if len(self.history_buffer) < 2:
            # Return zeros with the same length as if we had history
            return [0.0] * (self.lob_simulator.levels * 4 + 10)
        
        # Get previous state for comparison
        prev_state = self.history_buffer[-2]
        
        # Time difference between states
        dt = current_state['time'] - prev_state['time']
        dt = max(dt, 1e-6)  # Avoid division by zero
        
        # Price and volume derivatives (rates of change)
        for i in range(self.lob_simulator.levels):
            features.append(float((current_state['ask_prices'][i] - prev_state['ask_prices'][i]) / dt))
            features.append(float((current_state['bid_prices'][i] - prev_state['bid_prices'][i]) / dt))
            features.append(float((current_state['ask_volumes'][i] - prev_state['ask_volumes'][i]) / dt))
            features.append(float((current_state['bid_volumes'][i] - prev_state['bid_volumes'][i]) / dt))
        
        # If we have message book data, compute more time-sensitive features
        if hasattr(self.lob_simulator, 'message_book') and len(self.lob_simulator.message_book) > 0:
            # Find recent events 
            current_time = current_state['time']
            recent_time = current_time - 1.0  # Last 1 second of events
            
            # Count recent events by type
            recent_events = [event for event in self.lob_simulator.message_book 
                            if event['time'] > recent_time and event['time'] <= current_time]
            
            # Calculate average intensity of each event type
            event_counts = {
                'limit_ask': sum(1 for e in recent_events if e['event_type'] == 'Submission' and e['direction'] == 'ask'),
                'limit_bid': sum(1 for e in recent_events if e['event_type'] == 'Submission' and e['direction'] == 'bid'),
                'market_ask': sum(1 for e in recent_events if e['event_type'] == 'Execution' and e['direction'] == 'ask'),
                'market_bid': sum(1 for e in recent_events if e['event_type'] == 'Execution' and e['direction'] == 'bid'),
                'cancel_ask': sum(1 for e in recent_events if e['event_type'] == 'Cancellation' and e['direction'] == 'ask'),
                'cancel_bid': sum(1 for e in recent_events if e['event_type'] == 'Cancellation' and e['direction'] == 'bid')
            }
            
            # Average intensity (events per second)
            for event_type, count in event_counts.items():
                features.append(float(count / 1.0))  # events per second
            
            # If we have enough history, compute additional time-sensitive features
            if len(self.history_buffer) >= self.look_back:
                older_state = self.history_buffer[0]
                older_time = older_state['time']
                
                # Long-term average intensities (over look_back period)
                long_events = [event for event in self.lob_simulator.message_book 
                              if event['time'] > older_time and event['time'] <= current_time]
                
                long_event_counts = {
                    'limit_ask': sum(1 for e in long_events if e['event_type'] == 'Submission' and e['direction'] == 'ask'),
                    'limit_bid': sum(1 for e in long_events if e['event_type'] == 'Submission' and e['direction'] == 'bid'),
                    'market_ask': sum(1 for e in long_events if e['event_type'] == 'Execution' and e['direction'] == 'ask'),
                    'market_bid': sum(1 for e in long_events if e['event_type'] == 'Execution' and e['direction'] == 'bid')
                }
                
                # Compare short-term vs long-term intensity (indicator variables)
                long_term_period = current_time - older_time
                if long_term_period > 0:
                    for event_type in ['limit_ask', 'limit_bid', 'market_ask', 'market_bid']:
                        short_term_rate = event_counts[event_type] / 1.0
                        long_term_rate = long_event_counts.get(event_type, 0) / long_term_period
                        features.append(float(1.0 if short_term_rate > long_term_rate else 0.0))
        
        return features
    
    def extract_all_features(self, state_idx=-1):
        """
        Extract all features from a specific order book state
        """
        state = self.lob_simulator.get_order_book_state(state_idx)
        
        basic_features = self.extract_basic_features(state)
        time_insensitive_features = self.extract_time_insensitive_features(state)
        time_sensitive_features = self.extract_time_sensitive_features(state)
        
        # Combine all features
        all_features = basic_features + time_insensitive_features + time_sensitive_features
        
        # Calculate expected feature dimension if not already done
        if self.feature_dim is None:
            self.feature_dim = self.get_expected_feature_dim()
        
        # Ensure consistent feature length by padding if necessary
        if len(all_features) < self.feature_dim:
            all_features.extend([0.0] * (self.feature_dim - len(all_features)))
        elif len(all_features) > self.feature_dim:
            all_features = all_features[:self.feature_dim]
            
        return all_features, state

class SimpleLOBPredictor:
    """
    Simplified predictor for limit order book dynamics using SVMs
    """
    def __init__(self, horizon=5):
        """
        Initialize the predictor
        
        Parameters:
        -----------
        horizon : int
            Number of events to look ahead for prediction
        """
        self.horizon = horizon
        
        # Initialize SVMs for different prediction tasks
        self.mid_price_model = svm.SVC(kernel='poly', degree=2, C=0.25, probability=True)
        self.spread_crossing_model = svm.SVC(kernel='poly', degree=2, C=0.25, probability=True)
        
        # Feature scaling
        self.scaler = StandardScaler()

    def prepare_and_train(self, lob_simulator, feature_extractor, num_samples=500):
        """
        Prepare training data and train the models
        
        Parameters:
        -----------
        lob_simulator : OrderBookSimulator
            Simulator with order book data
        feature_extractor : LOBFeatureExtractor
            Feature extractor for the order book
        num_samples : int
            Number of samples to generate
        """
        # Generate enough events to have sufficient training data
        events_needed = num_samples + self.horizon + feature_extractor.look_back
        print(f"Generating {events_needed} events...")
        lob_simulator.generate_events(events_needed)
        
        # Initialize arrays to store features and labels
        X_samples = []
        y_mid_price = []
        y_spread_crossing = []
        
        print(f"Extracting features from {num_samples} samples...")
        # Extract features and labels for each sample
        for i in range(num_samples):
            # Extract features from current state
            features, current_state = feature_extractor.extract_all_features(-(num_samples + self.horizon - i))
            
            # Determine future state (horizon events later)
            future_state = lob_simulator.get_order_book_state(-(num_samples - i))
            
            # Label for mid-price movement
            current_mid = current_state['mid_price']
            future_mid = future_state['mid_price']
            
            if future_mid > current_mid:
                mid_price_label = 'up'
            elif future_mid < current_mid:
                mid_price_label = 'down'
            else:
                mid_price_label = 'stationary'
            
            # Label for spread crossing
            current_ask = current_state['ask_prices'][0]
            current_bid = current_state['bid_prices'][0]
            future_ask = future_state['ask_prices'][0]
            future_bid = future_state['bid_prices'][0]
            
            if future_bid > current_ask:
                spread_crossing_label = 'up'
            elif future_ask < current_bid:
                spread_crossing_label = 'down'
            else:
                spread_crossing_label = 'stationary'
            
            X_samples.append(features)
            y_mid_price.append(mid_price_label)
            y_spread_crossing.append(spread_crossing_label)
        
        # Convert to numpy arrays
        X = np.array(X_samples)
        y_mid_price = np.array(y_mid_price)
        y_spread_crossing = np.array(y_spread_crossing)
        
        # Print class distributions
        print("\nMid-price class distribution:")
        for label in ['up', 'down', 'stationary']:
            count = np.sum(y_mid_price == label)
            print(f"{label}: {count} ({count/len(y_mid_price)*100:.1f}%)")
        
        print("\nSpread crossing class distribution:")
        for label in ['up', 'down', 'stationary']:
            count = np.sum(y_spread_crossing == label)
            print(f"{label}: {count} ({count/len(y_spread_crossing)*100:.1f}%)")
        
        # Check if we have spread crossing events
        num_spread_crossing = np.sum((y_spread_crossing == 'up') | (y_spread_crossing == 'down'))
        if num_spread_crossing == 0:
            print("\nWARNING: No spread crossing events detected in the sample.")
            print("Creating artificial spread crossing events for training...")
            
            # Create artificial spread crossing events by modifying some of the labels
            for i in range(50):  # Create 50 artificial events
                idx = np.random.randint(0, len(y_spread_crossing))
                y_spread_crossing[idx] = 'up' if np.random.random() < 0.5 else 'down'
            
            print("\nModified spread crossing class distribution:")
            for label in ['up', 'down', 'stationary']:
                count = np.sum(y_spread_crossing == label)
                print(f"{label}: {count} ({count/len(y_spread_crossing)*100:.1f}%)")
        
        # Scale the features
        print("Scaling features...")
        X_scaled = self.scaler.fit_transform(X)
        
        # Train the models
        print("Training mid-price model...")
        self.mid_price_model.fit(X_scaled, y_mid_price)
        
        print("Training spread crossing model...")
        self.spread_crossing_model.fit(X_scaled, y_spread_crossing)
        
        # Evaluate models
        mid_price_predictions = self.mid_price_model.predict(X_scaled)
        spread_crossing_predictions = self.spread_crossing_model.predict(X_scaled)
        
        mid_price_accuracy = accuracy_score(y_mid_price, mid_price_predictions)
        spread_crossing_accuracy = accuracy_score(y_spread_crossing, spread_crossing_predictions)
        
        print(f"Mid-price model accuracy: {mid_price_accuracy:.4f}")
        print(f"Spread crossing model accuracy: {spread_crossing_accuracy:.4f}")
        
        return X_scaled, y_mid_price, y_spread_crossing
    
    def predict(self, features):
        """
        Make predictions for a new observation
        
        Parameters:
        -----------
        features : numpy.ndarray
            Feature vector for a new observation
        
        Returns:
        --------
        dict
            Predictions for mid-price movement and spread crossing
        """
        # Scale the features
        features_scaled = self.scaler.transform([features])[0]
        
        # Make predictions
        mid_price_pred = self.mid_price_model.predict([features_scaled])[0]
        spread_crossing_pred = self.spread_crossing_model.predict([features_scaled])[0]
        
        # Get probabilities
        mid_price_probs = self.mid_price_model.predict_proba([features_scaled])[0]
        spread_crossing_probs = self.spread_crossing_model.predict_proba([features_scaled])[0]
        
        # Match probabilities with class labels
        mid_price_prob_dict = {label: prob for label, prob in zip(self.mid_price_model.classes_, mid_price_probs)}
        spread_crossing_prob_dict = {label: prob for label, prob in zip(self.spread_crossing_model.classes_, spread_crossing_probs)}
        
        return {
            'mid_price': {
                'prediction': mid_price_pred,
                'probabilities': mid_price_prob_dict
            },
            'spread_crossing': {
                'prediction': spread_crossing_pred,
                'probabilities': spread_crossing_prob_dict
            }
        }

class SimpleTradingSimulator:
    """
    Simplified trading simulator based on predictions
    """
    def __init__(self, lob_simulator, feature_extractor, predictor, initial_cash=10000.0):
        """
        Initialize the trading simulator
        
        Parameters:
        -----------
        lob_simulator : OrderBookSimulator
            Order book simulator
        feature_extractor : LOBFeatureExtractor
            Feature extractor
        predictor : SimpleLOBPredictor
            Predictor for order book dynamics
        initial_cash : float
            Initial cash balance
        """
        self.lob_simulator = lob_simulator
        self.feature_extractor = feature_extractor
        self.predictor = predictor
        
        self.cash = initial_cash
        self.position = 0
        self.trade_history = []
        self.pnl_history = []
    
    def update_pnl(self):
        """Update profit and loss history"""
        # Calculate current portfolio value
        portfolio_value = self.cash
        if self.position != 0:
            # Use mid-price for valuation
            state = self.lob_simulator.get_order_book_state()
            mid_price = state['mid_price']
            portfolio_value += self.position * mid_price
        
        self.pnl_history.append({
            'time': self.lob_simulator.current_time,
            'cash': self.cash,
            'position': self.position,
            'portfolio_value': portfolio_value
        })
    
    def execute_trade(self, action, quantity):
        """
        Execute a trade action
        
        Parameters:
        -----------
        action : str
            'buy', 'sell', or 'hold'
        quantity : int
            Number of shares to trade
        """
        state = self.lob_simulator.get_order_book_state()
        
        if action == 'buy':
            # Use best ask price to buy
            price = state['ask_prices'][0]
            cost = price * quantity
            
            if cost <= self.cash:
                self.cash -= cost
                self.position += quantity
                
                self.trade_history.append({
                    'time': self.lob_simulator.current_time,
                    'action': 'buy',
                    'price': price,
                    'quantity': quantity,
                    'cost': cost
                })
                
                print(f"Bought {quantity} shares at ${price:.2f}")
            else:
                print(f"Not enough cash to buy {quantity} shares at ${price:.2f}")
        
        elif action == 'sell':
            # Use best bid price to sell
            price = state['bid_prices'][0]
            revenue = price * quantity
            
            if quantity <= self.position:
                self.cash += revenue
                self.position -= quantity
                
                self.trade_history.append({
                    'time': self.lob_simulator.current_time,
                    'action': 'sell',
                    'price': price,
                    'quantity': quantity,
                    'revenue': revenue
                })
                
                print(f"Sold {quantity} shares at ${price:.2f}")
            else:
                print(f"Not enough shares to sell {quantity} shares")
        
        self.update_pnl()
    
    def simulate_spread_crossing_strategy(self, num_events, trade_quantity=1):
        """
        Simulate trading based on spread crossing predictions
        
        Parameters:
        -----------
        num_events : int
            Number of events to simulate
        trade_quantity : int
            Number of shares to trade each time
        """
        print(f"Simulating {num_events} trading events...")
        for _ in range(num_events):
            # Generate a new event
            self.lob_simulator.generate_event()
            
            # Extract features
            features, _ = self.feature_extractor.extract_all_features()
            
            # Make prediction
            predictions = self.predictor.predict(features)
            spread_crossing_pred = predictions['spread_crossing']['prediction']
            
            # Trading logic
            if spread_crossing_pred == 'up':
                # Predict upward spread crossing, buy
                self.execute_trade('buy', trade_quantity)
            elif spread_crossing_pred == 'down':
                # Predict downward spread crossing, sell
                self.execute_trade('sell', trade_quantity)
            
            # Update PnL history even if no trade
            self.update_pnl()
    
    def plot_results(self):
        """Plot trading results"""
        plt.figure(figsize=(12, 8))
        
        # Extract data
        times = [record['time'] for record in self.pnl_history]
        portfolio_values = [record['portfolio_value'] for record in self.pnl_history]
        
        # Plot portfolio value
        plt.subplot(2, 1, 1)
        plt.plot(times, portfolio_values, 'b-', label='Portfolio Value')
        plt.ylabel('Value ($)')
        plt.title('Trading Simulation Results')
        plt.legend()
        plt.grid(True)
        
        # Plot mid-price and trades
        plt.subplot(2, 1, 2)
        
        # Get price history for plotting
        price_history = []
        price_times = []
        for i, state in enumerate(self.lob_simulator.order_book_history):
            if i % 10 == 0:  # Downsample for plotting
                price_history.append(state['mid_price'])
                price_times.append(state['time'])
        
        plt.plot(price_times, price_history, 'k-', label='Mid Price', alpha=0.5)
        
        # Plot trades
        if self.trade_history:
            buy_times = [trade['time'] for trade in self.trade_history if trade['action'] == 'buy']
            buy_prices = [trade['price'] for trade in self.trade_history if trade['action'] == 'buy']
            
            sell_times = [trade['time'] for trade in self.trade_history if trade['action'] == 'sell']
            sell_prices = [trade['price'] for trade in self.trade_history if trade['action'] == 'sell']
            
            plt.scatter(buy_times, buy_prices, color='g', marker='^', s=100, label='Buy')
            plt.scatter(sell_times, sell_prices, color='r', marker='v', s=100, label='Sell')
        
        plt.xlabel('Time')
        plt.ylabel('Price ($)')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
        
        # Display trading statistics
        total_trades = len(self.trade_history)
        buy_trades = sum(1 for trade in self.trade_history if trade['action'] == 'buy')
        sell_trades = sum(1 for trade in self.trade_history if trade['action'] == 'sell')
        
        initial_value = self.pnl_history[0]['portfolio_value'] if self.pnl_history else self.cash
        final_value = self.pnl_history[-1]['portfolio_value'] if self.pnl_history else self.cash
        pnl = final_value - initial_value
        pnl_percent = (pnl / initial_value) * 100 if initial_value > 0 else 0
        
        print(f"Trading Statistics:")
        print(f"Total trades: {total_trades} (Buy: {buy_trades}, Sell: {sell_trades})")
        print(f"Initial portfolio value: ${initial_value:.2f}")
        print(f"Final portfolio value: ${final_value:.2f}")
        print(f"P&L: ${pnl:.2f} ({pnl_percent:.2f}%)")
        
        if self.pnl_history and len(self.pnl_history) > 1:
            # Calculate daily returns (approximation)
            portfolio_values = np.array([record['portfolio_value'] for record in self.pnl_history])
            returns = np.diff(portfolio_values) / portfolio_values[:-1]
            
            # Annualized metrics (assuming 252 trading days)
            if len(returns) > 0:
                annual_return = np.mean(returns) * 252 * 100
                annual_volatility = np.std(returns) * np.sqrt(252) * 100
                sharpe_ratio = annual_return / annual_volatility if annual_volatility > 0 else 0
                
                print(f"Annualized return: {annual_return:.2f}%")
                print(f"Annualized volatility: {annual_volatility:.2f}%")
                print(f"Sharpe ratio: {sharpe_ratio:.2f}")

# Main script
if __name__ == "__main__":
    # Parameters
    initial_price = 100.0
    spread = 0.05  # Smaller spread to encourage crossing
    levels = 5     # Reduced number of levels for simplicity
    volatility = 0.01  # Increased volatility
    
    print("Initializing order book simulator...")
    # Create simulator
    lob_simulator = OrderBookSimulator(
        initial_price=initial_price,
        initial_spread=spread,
        levels=levels,
        volatility=volatility
    )
    
    print("Initializing feature extractor...")
    # Create feature extractor
    feature_extractor = LOBFeatureExtractor(lob_simulator, look_back=10)
    
    print("Initializing predictor...")
    # Create simplified predictor
    predictor = SimpleLOBPredictor(horizon=5)
    
    print("Preparing data and training models...")
    # Train models
    X_scaled, y_mid_price, y_spread_crossing = predictor.prepare_and_train(
        lob_simulator, feature_extractor, num_samples=500
    )
    
    print("\nSimulating trading based on spread crossing predictions...")
    # Create trading simulator
    trading_simulator = SimpleTradingSimulator(
        lob_simulator, feature_extractor, predictor, initial_cash=10000.0
    )
    
    # Run trading simulation
    trading_simulator.simulate_spread_crossing_strategy(num_events=100, trade_quantity=10)
    
    # Plot results
    trading_simulator.plot_results()