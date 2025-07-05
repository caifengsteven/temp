import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import random
from tqdm import tqdm

# Generate simulated price data
def generate_price_data(n_points=5000, trend='mixed', volatility=0.02, seed=42):
    np.random.seed(seed)
    
    # Base price
    if trend == 'bullish':
        drift = 0.001  # Positive drift for bullish trend
    elif trend == 'bearish':
        drift = -0.001  # Negative drift for bearish trend
    else:  # mixed
        drift = 0.0005  # Slight positive drift
    
    # Generate log returns
    returns = np.random.normal(drift, volatility, n_points)
    
    # Generate price series
    prices = 100 * np.exp(np.cumsum(returns))
    
    # Add some cyclical patterns
    t = np.linspace(0, 4*np.pi, n_points)
    cycle1 = np.sin(t) * 5
    cycle2 = np.sin(t/3) * 10
    
    if trend == 'mixed':
        # Add a trend reversal in the middle for mixed scenario
        trend_factor = np.ones(n_points)
        trend_factor[n_points//2:] = -1
        cycle3 = np.cumsum(trend_factor) * 0.05
        prices = prices + cycle1 + cycle2 + cycle3
    else:
        prices = prices + cycle1 + cycle2
    
    # Generate corresponding volumes
    base_volume = 100000
    volumes = np.random.gamma(2, 20000, n_points)
    volumes = volumes * (1 + 0.5 * np.abs(returns))  # Higher volume on higher returns
    
    # Create a DataFrame
    df = pd.DataFrame({
        'price': prices,
        'volume': volumes
    })
    
    # Add timestamps (assuming 1-minute data)
    df['timestamp'] = pd.date_range(start='2023-01-01', periods=n_points, freq='1min')
    
    return df

# Fast Learning Network (FLN) for Q-function approximation
class FLN:
    def __init__(self, input_size, output_size, hidden_size=50):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        
        # Initialize weights
        # Input to hidden weights (random and fixed)
        self.whi = np.random.uniform(-1, 1, (hidden_size, input_size))
        
        # Normalize hidden weights
        for i in range(hidden_size):
            self.whi[i] = self.whi[i] / np.linalg.norm(self.whi[i])
        
        # Output weights (to be learned)
        self.woi = np.random.uniform(-1, 1, (output_size, input_size))
        self.woh = np.random.uniform(-1, 1, (output_size, hidden_size))
        
        self.max_norm = 1.0
        
    def activation(self, x):
        # Logistic function
        return 1.0 / (1.0 + np.exp(-x))
    
    def forward(self, x):
        # Direct path
        direct_output = np.dot(self.woi, x)
        
        # Hidden layer path
        hidden_input = np.dot(self.whi, x)
        hidden_output = self.activation(hidden_input)
        hidden_path = np.dot(self.woh, hidden_output)
        
        # Combine paths
        output = direct_output + hidden_path
        
        return output, hidden_output
    
    def update(self, x, hidden_output, action, target, alpha):
        # Combine input and hidden layer output
        gradient = np.concatenate([x, hidden_output])
        
        # Extract weights for the specific action
        woi_action = self.woi[action]
        woh_action = self.woh[action]
        
        # Update only the weights for the chosen action
        action_weights = np.concatenate([woi_action, woh_action])
        action_weights = action_weights + alpha * (target - np.dot(action_weights, gradient)) * gradient
        
        # Weight renormalization
        norm = np.linalg.norm(action_weights)
        self.max_norm = max(self.max_norm, norm)
        if norm > 1.0:
            action_weights = action_weights / self.max_norm
        
        # Split and assign back
        self.woi[action] = action_weights[:self.input_size]
        self.woh[action] = action_weights[self.input_size:]

# Deep Reinforcement Learning Trader
class DRLTrader:
    def __init__(self, feature_size=27, action_size=19, hidden_size=50, gamma=0.05, 
                 alpha_min=0.001, alpha_max=1.0, alpha_period=1000, mlimn=75):
        # Q-networks
        self.q1 = FLN(feature_size, action_size, hidden_size)
        self.q2 = FLN(feature_size, action_size, hidden_size)
        
        # Parameters
        self.gamma = gamma
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.alpha_period = alpha_period
        self.alpha_counter = 0
        self.epsilon_counter = 0
        self.epsilon_reset_prob = 0.0001
        self.mlimn = mlimn
        
        # Action space
        self.action_size = action_size
        
        # Money pools
        self.mon = 100.0  # Trading pool
        self.sav = 0.0    # Savings (profit taken)
        self.res = 0.0    # Reserve pool
        self.cns = 0.0    # Assets
        self.mlim = self.mon  # Money limit
        
        # For RSI calculation
        self.rsi_window = 15
        self.price_history = []
        
    def get_alpha(self):
        # Cyclical learning rate
        alpha = self.alpha_min + 0.5 * (self.alpha_max - self.alpha_min) * \
                (1 + np.cos(self.alpha_counter / self.alpha_period * np.pi))
        self.alpha_counter += 1
        return alpha
    
    def get_epsilon(self):
        # Update epsilon
        if random.random() <= self.epsilon_reset_prob and self.epsilon_counter >= int(np.ceil((np.exp(5/2) - 2)/5)):
            self.epsilon_counter = int(np.ceil((np.exp(5/2) - 2)/5))
        else:
            self.epsilon_counter += 1
        
        # Calculate epsilon
        epsilon = 1.0 / np.log(self.epsilon_counter * 5.0 + 2.0)
        return epsilon
    
    def calculate_rsi(self, prices):
        if len(prices) < 15:
            return 50.0  # Default value if not enough history
        
        # Get the last 15 prices
        price_window = prices[-15:]
        
        # Calculate price changes
        changes = np.diff(price_window)
        
        # Separate gains and losses
        gains = np.where(changes > 0, changes, 0)
        losses = np.where(changes < 0, -changes, 0)
        
        # Calculate average gain and loss
        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)
        
        # Calculate RSI
        if avg_loss == 0:
            return 100.0
        else:
            rs = avg_gain / avg_loss
            rsi = 100.0 - (100.0 / (1.0 + rs))
            return rsi
    
    def preprocess_state(self, prices, volumes):
        # Ensure we have enough data
        if len(prices) < 5 or len(volumes) < 5:
            return None
        
        # Extract the prices and volumes we need
        pr = prices[-5:]
        vols = volumes[-5:]
        
        # Initial price (from beginning of episode)
        ipr = prices[-5]
        
        # Calculate volume averages
        cav = np.mean(vols)
        if len(volumes) >= 100:
            av = np.mean(volumes[-100:])
        else:
            av = np.mean(volumes)
        
        # Calculate relative price movements
        nmd1 = [(pr[i+1] - pr[i])/pr[i] for i in range(4)]
        nmd2 = [(nmd1[i+1] - nmd1[i])/abs(nmd1[i]) if nmd1[i] != 0 else 0 for i in range(3)]
        nmd3 = [(nmd2[i+1] - nmd2[i])/abs(nmd2[i]) if nmd2[i] != 0 else 0 for i in range(2)]
        nmd4 = (nmd3[1] - nmd3[0])/abs(nmd3[0]) if nmd3[0] != 0 else 0
        
        # Calculate RSI
        rsi = self.calculate_rsi(prices)
        
        # Construct feature vector
        feat = [1.0]  # Bias term
        feat.extend(pr)  # Prices
        feat.append(ipr)  # Initial price
        feat.append((pr[-1] - ipr)/ipr)  # Relative price change
        feat.append(self.mon)  # Money
        feat.append(self.cns)  # Assets
        feat.append(cav)  # Current average volume
        feat.append(av)  # Moving average volume
        feat.append((cav - av)/av if av != 0 else 0)  # Relative volume change
        feat.append((vols[-1] - av)/av if av != 0 else 0)  # Last volume relative to average
        feat.append((vols[-1] - cav)/cav if cav != 0 else 0)  # Last volume relative to current average
        feat.append(rsi)  # RSI
        feat.extend(nmd1)  # First derivatives
        feat.extend(nmd2)  # Second derivatives
        feat.extend(nmd3)  # Third derivatives
        feat.append(nmd4)  # Fourth derivative
        feat.append(self.mlim)  # Money limit
        
        # Normalize feature vector
        feat = np.array(feat)
        norm = np.linalg.norm(feat)
        if norm > 0:
            feat = 6.0 * (feat / norm)
        
        feat[0] = 1.0  # Keep bias term as 1
        
        return feat
    
    def select_action(self, state, epsilon):
        # ε-greedy policy
        if random.random() < epsilon:
            return random.randint(0, self.action_size - 1)
        else:
            # Forward pass through both networks
            q1_values, _ = self.q1.forward(state)
            q2_values, _ = self.q2.forward(state)
            
            # Average Q-values
            q_values = (q1_values + q2_values) / 2.0
            
            # Select action with highest Q-value
            return np.argmax(q_values)
    
    def take_action(self, action, current_price):
        # Initialize success flag
        success = True
        
        # Buy actions (1-9)
        if 0 <= action <= 8:
            amount = (action + 1) * 10  # 10, 20, ..., 90
            if self.mon >= amount:
                self.cns += (0.999 * (amount / current_price))  # 0.1% fee
                self.mon -= amount
            else:
                success = False
        
        # Sell actions (10-18)
        elif 9 <= action <= 17:
            amount = (action - 8) * 10  # 10, 20, ..., 90
            if current_price * self.cns >= amount:
                self.mon += (0.999 * amount)  # 0.1% fee
                self.cns -= (amount / current_price)
            else:
                success = False
        
        # Hold (19)
        # No action needed for hold
        
        return success
    
    def calculate_reward(self, old_wealth, new_wealth, success):
        # Calculate the reward
        wealth_change = new_wealth - old_wealth
        
        if success:
            return wealth_change - (wealth_change ** 2) / 2
        else:
            return wealth_change - (wealth_change ** 2) / 2 - 0.1
    
    def update_networks(self, state, action, reward, next_state, done):
        # Get current Q-values
        q1_values, h1 = self.q1.forward(state)
        q2_values, h2 = self.q2.forward(state)
        
        # Get next Q-values
        q1_next, h1_next = self.q1.forward(next_state)
        q2_next, h2_next = self.q2.forward(next_state)
        
        # Double Q-learning targets
        if random.random() < 0.5:
            # Update Q1
            next_action = np.argmax(q1_next)
            target = reward + (0 if done else self.gamma * q2_next[next_action])
            self.q1.update(state, h1, action, target, self.get_alpha())
        else:
            # Update Q2
            next_action = np.argmax(q2_next)
            target = reward + (0 if done else self.gamma * q1_next[next_action])
            self.q2.update(state, h2, action, target, self.get_alpha())
    
    def check_terminal_state(self, q_value, rsi):
        # Terminal state 1: mon > mlim
        if self.mon > self.mlim:
            mdf = self.mon - self.mlim
            self.sav += mdf * 0.34
            self.res += mdf * 0.33
            self.mon = self.mlim + mdf * 0.33
            self.mlim = self.mon + mdf
            return True, 0.34 * mdf  # Extra reward
        
        # Terminal state 2: low mon, positive Q, high RSI
        if self.mon < self.mlim and self.calculate_wealth() < self.mlimn and q_value > 0 and rsi > 70:
            self.mon += self.res / 2
            self.res -= self.res / 2
            self.mlim = max(self.mlimn, self.mon + self.res / 2)
            return True, 0
        
        # Terminal state 3: low mon, negative Q, low RSI
        if self.mon < self.mlim and self.calculate_wealth() >= self.mlimn and q_value < 0 and rsi < 30:
            self.mlim = self.calculate_wealth()
            return True, 0
        
        return False, 0
    
    def calculate_wealth(self):
        return self.mon + self.cns * self.current_price
    
    def calculate_total_wealth(self):
        return self.calculate_wealth() + self.sav + self.res
    
    def reset(self):
        self.mon = 100.0
        self.sav = 0.0
        self.res = 0.0
        self.cns = 0.0
        self.mlim = self.mon
        self.price_history = []
        self.epsilon_counter = 0
        self.alpha_counter = 0
    
    def train(self, prices, volumes, max_steps=None):
        self.reset()
        
        if max_steps is None:
            max_steps = len(prices) - 5
        
        step = 0
        episode = 0
        total_reward = 0
        
        while step < max_steps:
            # Update price history
            self.price_history = list(prices[:step + 5])
            self.current_price = prices[step + 4]  # Current price is the 5th price
            
            # Calculate the current state
            state = self.preprocess_state(prices[:step + 5], volumes[:step + 5])
            if state is None:
                step += 1
                continue
            
            # Calculate current wealth
            current_wealth = self.calculate_wealth()
            
            # Select action
            epsilon = self.get_epsilon()
            action = self.select_action(state, epsilon)
            
            # Take action
            success = self.take_action(action, self.current_price)
            
            # Move to next step
            step += 1
            if step >= max_steps:
                break
            
            # Calculate next state
            self.current_price = prices[step + 4]  # Next price
            next_state = self.preprocess_state(prices[:step + 5], volumes[:step + 5])
            if next_state is None:
                continue
            
            # Calculate new wealth and reward
            new_wealth = self.calculate_wealth()
            reward = self.calculate_reward(current_wealth, new_wealth, success)
            total_reward += reward
            
            # Get Q-values for checking terminal state
            q1_values, _ = self.q1.forward(state)
            q2_values, _ = self.q2.forward(state)
            q_value = (q1_values[action] + q2_values[action]) / 2.0
            
            # Check for terminal state
            rsi = self.calculate_rsi(prices[:step + 5])
            is_terminal, extra_reward = self.check_terminal_state(q_value, rsi)
            
            # Add extra reward if terminal
            if is_terminal:
                reward += extra_reward
                total_reward += extra_reward
                episode += 1
            
            # Update networks
            self.update_networks(state, action, reward, next_state, is_terminal)
            
            # If terminal, continue from next step as a new episode
            if is_terminal:
                continue
        
        return {
            'final_wealth': self.calculate_total_wealth(),
            'saved': self.sav,
            'episodes': episode,
            'total_reward': total_reward
        }
    
    def random_actions(self, prices, volumes, max_steps=None):
        self.reset()
        
        if max_steps is None:
            max_steps = len(prices) - 5
        
        step = 0
        
        while step < max_steps:
            # Update current price
            self.current_price = prices[step + 4]
            
            # Take random action
            action = random.randint(0, self.action_size - 1)
            self.take_action(action, self.current_price)
            
            # Move to next step
            step += 1
            
        return {
            'final_wealth': self.mon + self.cns * prices[-1],
            'saved': 0  # No savings mechanism for random
        }

# Test the strategy
def test_strategy(market_type='mixed', num_runs=100):
    # Generate simulated data
    data = generate_price_data(n_points=5000, trend=market_type, seed=42)
    
    prices = data['price'].values
    volumes = data['volume'].values
    
    # Initialize results storage
    drl_results = []
    random_results = []
    
    # Run multiple tests
    for i in tqdm(range(num_runs), desc=f"Testing {market_type} market"):
        # DRL trader
        trader = DRLTrader()
        result = trader.train(prices, volumes)
        drl_results.append({
            'total_wealth': result['final_wealth'],
            'saved': result['saved']
        })
        
        # Random actions
        trader = DRLTrader()
        result = trader.random_actions(prices, volumes)
        random_results.append({
            'total_wealth': result['final_wealth'],
            'saved': 0
        })
    
    # Convert to DataFrames
    drl_df = pd.DataFrame(drl_results)
    random_df = pd.DataFrame(random_results)
    
    # Calculate statistics
    drl_stats = {
        'mean': drl_df['total_wealth'].mean(),
        'median': drl_df['total_wealth'].median(),
        'std': drl_df['total_wealth'].std(),
        'min': drl_df['total_wealth'].min(),
        'max': drl_df['total_wealth'].max(),
        'loss_prob': (drl_df['total_wealth'] <= 100).mean(),
        'sav_mean': drl_df['saved'].mean()
    }
    
    random_stats = {
        'mean': random_df['total_wealth'].mean(),
        'median': random_df['total_wealth'].median(),
        'std': random_df['total_wealth'].std(),
        'min': random_df['total_wealth'].min(),
        'max': random_df['total_wealth'].max(),
        'loss_prob': (random_df['total_wealth'] <= 100).mean()
    }
    
    # Create histograms
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.hist(drl_df['total_wealth'], bins=20, alpha=0.7, label='DRL Strategy')
    plt.axvline(x=100, color='red', linestyle='--', label='Initial Investment')
    plt.title(f'DRL Strategy - {market_type.capitalize()} Market')
    plt.xlabel('Total Wealth (USDT)')
    plt.ylabel('Count')
    plt.legend()
    
    plt.subplot(1, 3, 2)
    plt.hist(random_df['total_wealth'], bins=20, alpha=0.7, label='Random Strategy')
    plt.axvline(x=100, color='red', linestyle='--', label='Initial Investment')
    plt.title(f'Random Strategy - {market_type.capitalize()} Market')
    plt.xlabel('Total Wealth (USDT)')
    plt.ylabel('Count')
    plt.legend()
    
    plt.subplot(1, 3, 3)
    plt.hist(drl_df['saved'], bins=20, alpha=0.7, label='Savings')
    plt.title(f'Savings - {market_type.capitalize()} Market')
    plt.xlabel('Saved Amount (USDT)')
    plt.ylabel('Count')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'{market_type}_results.png')
    
    return drl_stats, random_stats, drl_df, random_df

# Test in different market conditions
market_types = ['bullish', 'bearish', 'mixed']
results = {}

for market_type in market_types:
    drl_stats, random_stats, drl_df, random_df = test_strategy(market_type=market_type, num_runs=100)
    results[market_type] = {
        'drl': drl_stats,
        'random': random_stats
    }

# Print comparison results
for market_type, stats in results.items():
    print(f"\n{market_type.upper()} MARKET RESULTS:")
    print("----------------------------------------")
    print(f"DRL Strategy:")
    print(f"  Mean Total Wealth: ${stats['drl']['mean']:.2f}")
    print(f"  Median Total Wealth: ${stats['drl']['median']:.2f}")
    print(f"  Std Dev: ${stats['drl']['std']:.2f}")
    print(f"  Min: ${stats['drl']['min']:.2f}, Max: ${stats['drl']['max']:.2f}")
    print(f"  Loss Probability: {stats['drl']['loss_prob']*100:.2f}%")
    print(f"  Mean Savings: ${stats['drl']['sav_mean']:.2f}")
    
    print(f"\nRandom Strategy:")
    print(f"  Mean Total Wealth: ${stats['random']['mean']:.2f}")
    print(f"  Median Total Wealth: ${stats['random']['median']:.2f}")
    print(f"  Std Dev: ${stats['random']['std']:.2f}")
    print(f"  Min: ${stats['random']['min']:.2f}, Max: ${stats['random']['max']:.2f}")
    print(f"  Loss Probability: {stats['random']['loss_prob']*100:.2f}%")
    
    print(f"\nPerformance Difference:")
    mean_diff = (stats['drl']['mean'] - stats['random']['mean']) / stats['random']['mean'] * 100
    median_diff = (stats['drl']['median'] - stats['random']['median']) / stats['random']['median'] * 100
    loss_prob_diff = (stats['random']['loss_prob'] - stats['drl']['loss_prob']) / stats['random']['loss_prob'] * 100
    
    print(f"  Mean Wealth Increase: {mean_diff:.2f}%")
    print(f"  Median Wealth Increase: {median_diff:.2f}%")
    print(f"  Loss Probability Reduction: {loss_prob_diff:.2f}%")
    print("----------------------------------------")

# Plot the simulated price data
plt.figure(figsize=(12, 6))
for market_type in market_types:
    data = generate_price_data(n_points=5000, trend=market_type, seed=42)
    plt.plot(data['price'], label=f'{market_type.capitalize()} Market')

plt.title('Simulated Price Data for Different Market Conditions')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.savefig('simulated_price_data.png')
plt.show()