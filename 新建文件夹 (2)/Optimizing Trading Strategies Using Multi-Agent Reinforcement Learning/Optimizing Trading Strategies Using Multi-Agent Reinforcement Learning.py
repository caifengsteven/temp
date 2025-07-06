import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Concatenate
from tensorflow.keras.optimizers import Adam
import os
import random
from collections import deque
import yfinance as yf
from datetime import datetime, timedelta
import copy

# Set seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

class ReplayBuffer:
    """Experience replay buffer for storing agent experiences"""
    
    def __init__(self, buffer_size, batch_size):
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size
    
    def add(self, state, action, reward, next_state, done):
        """Add experience to buffer"""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self):
        """Sample a batch of experiences from buffer"""
        batch = random.sample(self.buffer, min(len(self.buffer), self.batch_size))
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)

class Actor:
    """Actor network for MADDPG"""
    
    def __init__(self, state_dim, action_dim, action_high, name="actor"):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_high = action_high
        self.name = name
        self.model = self._build_model()
        self.target_model = self._build_model()
        # Make target weights equal to model weights
        self.target_model.set_weights(self.model.get_weights())
        
    def _build_model(self):
        """Build actor network"""
        inputs = Input(shape=(self.state_dim,))
        x = Dense(256, activation='relu')(inputs)
        x = Dense(128, activation='relu')(x)
        outputs = Dense(self.action_dim, activation='tanh')(x)
        # Scale outputs to action space
        outputs = tf.multiply(outputs, self.action_high)
        
        model = Model(inputs=inputs, outputs=outputs, name=self.name)
        return model
    
    def predict(self, state):
        """Predict action based on state"""
        return self.model.predict(state, verbose=0)
    
    def target_predict(self, state):
        """Predict action based on state using target model"""
        return self.target_model.predict(state, verbose=0)
    
    def update_target(self, tau):
        """Update target model weights"""
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(weights)):
            target_weights[i] = tau * weights[i] + (1 - tau) * target_weights[i]
        self.target_model.set_weights(target_weights)
    
    def train(self, states, critic_grads):
        """Train actor using policy gradient"""
        with tf.GradientTape() as tape:
            actions = self.model(states)
            actor_loss = -tf.reduce_mean(critic_grads * actions)
        
        actor_grads = tape.gradient(actor_loss, self.model.trainable_variables)
        optimizer = Adam(learning_rate=0.001)
        optimizer.apply_gradients(zip(actor_grads, self.model.trainable_variables))
        
        return actor_loss

class Critic:
    """Critic network for MADDPG"""
    
    def __init__(self, state_dim, action_dim, name="critic"):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.name = name
        self.model = self._build_model()
        self.target_model = self._build_model()
        # Make target weights equal to model weights
        self.target_model.set_weights(self.model.get_weights())
        
    def _build_model(self):
        """Build critic network"""
        state_input = Input(shape=(self.state_dim,))
        action_input = Input(shape=(self.action_dim,))
        
        state_x = Dense(128, activation='relu')(state_input)
        action_x = Dense(128, activation='relu')(action_input)
        
        x = Concatenate()([state_x, action_x])
        x = Dense(128, activation='relu')(x)
        x = Dense(64, activation='relu')(x)
        outputs = Dense(1, activation='linear')(x)
        
        model = Model(inputs=[state_input, action_input], outputs=outputs, name=self.name)
        model.compile(optimizer=Adam(learning_rate=0.002), loss='mse')
        return model
    
    def predict(self, state, action):
        """Predict Q-value based on state and action"""
        return self.model.predict([state, action], verbose=0)
    
    def target_predict(self, state, action):
        """Predict Q-value based on state and action using target model"""
        return self.target_model.predict([state, action], verbose=0)
    
    def update_target(self, tau):
        """Update target model weights"""
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(weights)):
            target_weights[i] = tau * weights[i] + (1 - tau) * target_weights[i]
        self.target_model.set_weights(target_weights)
    
    def train(self, states, actions, target):
        """Train critic using TD error"""
        return self.model.train_on_batch([states, actions], target)

class MADDPGAgent:
    """Multi-Agent Deep Deterministic Policy Gradient Agent"""
    
    def __init__(self, state_dim, action_dim, action_high, agent_id, num_agents, discount_factor=0.95, tau=0.01, 
                 buffer_size=100000, batch_size=64, noise_std=0.1, insurance_strategy=None):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_high = action_high
        self.agent_id = agent_id
        self.num_agents = num_agents
        self.discount_factor = discount_factor
        self.tau = tau
        self.batch_size = batch_size
        self.noise_std = noise_std
        self.insurance_strategy = insurance_strategy  # 'CPPI' or 'TIPP' or None
        
        # Create actor and critic
        self.actor = Actor(state_dim, action_dim, action_high, name=f"actor_{agent_id}")
        # Critic takes all states and actions as input
        self.critic = Critic(state_dim * num_agents, action_dim * num_agents, name=f"critic_{agent_id}")
        
        # Create shared replay buffer
        self.buffer = ReplayBuffer(buffer_size, batch_size)
        
        # Portfolio parameters for insurance strategies
        self.k = 3.0  # Risk factor for CPPI/TIPP
        self.phi = 0.8  # Floor percentage for TIPP
        self.floor = None  # Will be initialized in the first step
    
    def get_action(self, state, add_noise=True):
        """Get action based on state with optional exploration noise"""
        state = np.reshape(state, [1, self.state_dim])
        action = self.actor.predict(state)[0]
        
        # Add exploration noise
        if add_noise:
            noise = np.random.normal(0, self.noise_std, size=self.action_dim)
            action = action + noise
        
        # Apply insurance strategy if specified
        if self.insurance_strategy and state[0][-1] > 0:  # Check if there's balance available
            total_asset = np.sum(state[0][self.action_dim:2*self.action_dim] * state[0][:self.action_dim]) + state[0][-1]
            
            if self.floor is None:
                # Initialize floor on first step
                self.floor = 0.8 * total_asset  # Initial floor at 80% of total asset
            
            if self.insurance_strategy == 'CPPI':
                cushion = max(0, total_asset - self.floor)
                exposure = self.k * cushion
                max_allocation = min(1.0, exposure / total_asset)
                
                # Normalize actions to sum to max_allocation
                if np.sum(action) > 0:
                    action = action / np.sum(action) * max_allocation
                
            elif self.insurance_strategy == 'TIPP':
                # Update floor for TIPP
                self.floor = max(self.phi * total_asset, self.floor)
                cushion = max(0, total_asset - self.floor)
                exposure = self.k * cushion
                max_allocation = min(1.0, exposure / total_asset)
                
                # Normalize actions to sum to max_allocation
                if np.sum(action) > 0:
                    action = action / np.sum(action) * max_allocation
        
        # Clip action to valid range
        return np.clip(action, 0, self.action_high)
    
    def remember(self, state, action, reward, next_state, done):
        """Add experience to replay buffer"""
        self.buffer.add(state, action, reward, next_state, done)
    
    def learn(self, agents):
        """Update agent's actor and critic networks"""
        if len(self.buffer) < self.batch_size:
            return
        
        # Sample from replay buffer
        states, actions, rewards, next_states, dones = self.buffer.sample()
        
        # Extract individual states and actions
        all_states = np.zeros((len(states), self.state_dim * self.num_agents))
        all_actions = np.zeros((len(actions), self.action_dim * self.num_agents))
        all_next_states = np.zeros((len(next_states), self.state_dim * self.num_agents))
        all_next_actions = np.zeros((len(next_states), self.action_dim * self.num_agents))
        
        for i in range(self.num_agents):
            all_states[:, i*self.state_dim:(i+1)*self.state_dim] = states
            all_actions[:, i*self.action_dim:(i+1)*self.action_dim] = actions
            all_next_states[:, i*self.state_dim:(i+1)*self.state_dim] = next_states
            
            # Get next actions from target actor for each agent
            agent_next_actions = agents[i].actor.target_predict(
                next_states.reshape(-1, self.state_dim)
            )
            all_next_actions[:, i*self.action_dim:(i+1)*self.action_dim] = agent_next_actions
        
        # Update critic
        target_q = rewards[:, self.agent_id:self.agent_id+1] + self.discount_factor * (1 - dones[:, self.agent_id:self.agent_id+1]) * \
                  self.critic.target_predict(all_next_states, all_next_actions)
        
        critic_loss = self.critic.train(all_states, all_actions, target_q)
        
        # Update actor
        # Predict actions for current states (without noise)
        my_actions = self.actor.model(states)
        
        # Replace this agent's actions in all_actions
        actor_actions = all_actions.copy()
        actor_actions[:, self.agent_id*self.action_dim:(self.agent_id+1)*self.action_dim] = my_actions
        
        # Compute policy gradient
        with tf.GradientTape() as tape:
            q_values = self.critic.model([all_states, actor_actions])
            actor_loss = -tf.reduce_mean(q_values)
            
            # Add correlation penalty
            if self.num_agents > 1:
                correlation_penalty = 0
                for i in range(self.num_agents):
                    if i != self.agent_id:
                        other_actions = actor_actions[:, i*self.action_dim:(i+1)*self.action_dim]
                        # Calculate correlation
                        correlation = tf.reduce_mean(tf.reduce_sum(my_actions * other_actions, axis=1))
                        correlation_penalty += correlation ** 2
                
                actor_loss = 0.8 * actor_loss + 0.2 * correlation_penalty
        
        # Update actor weights
        actor_grads = tape.gradient(actor_loss, self.actor.model.trainable_variables)
        actor_optimizer = Adam(learning_rate=0.001)
        actor_optimizer.apply_gradients(zip(actor_grads, self.actor.model.trainable_variables))
        
        # Update target networks
        self.actor.update_target(self.tau)
        self.critic.update_target(self.tau)
        
        return critic_loss, actor_loss

class StockMarketEnv:
    """Simulated stock market environment"""
    
    def __init__(self, stock_data, num_stocks, initial_balance=10000, transaction_cost=0.001, window_size=10):
        self.stock_data = stock_data  # DataFrame with stock prices
        self.num_stocks = num_stocks  # Number of stocks
        self.initial_balance = initial_balance  # Initial cash balance
        self.transaction_cost = transaction_cost  # Transaction cost as a fraction of trade value
        self.window_size = window_size  # Number of days to include in state
        
        self.current_step = window_size
        self.shares = np.zeros(num_stocks)
        self.balance = initial_balance
        self.total_steps = len(stock_data) - window_size
        
    def reset(self):
        """Reset the environment to initial state"""
        self.current_step = self.window_size
        self.shares = np.zeros(self.num_stocks)
        self.balance = self.initial_balance
        return self._get_observation()
    
    def _get_observation(self):
        """Get current state observation"""
        # Get stock prices for the current day
        prices = self.stock_data.iloc[self.current_step].values[:self.num_stocks]
        
        # Calculate features from the past window
        window_data = self.stock_data.iloc[self.current_step-self.window_size:self.current_step]
        features = []
        
        # Simple features: prices, normalized returns, and volatilities
        for i in range(self.num_stocks):
            price_series = window_data.iloc[:, i].values
            returns = np.diff(price_series) / price_series[:-1]
            features.append(price_series[-1])  # Current price
            features.append(np.mean(returns))  # Mean return
            features.append(np.std(returns))   # Volatility
        
        # Combine state: features + current holdings + balance
        state = np.concatenate([
            prices,  # Current prices
            self.shares,  # Current shares
            [self.balance]  # Current balance
        ])
        
        return state
    
    def step(self, actions, agent_id=0):
        """Take a step in the environment based on actions"""
        # Current prices and portfolio value before action
        current_prices = self.stock_data.iloc[self.current_step].values[:self.num_stocks]
        portfolio_value_before = np.sum(self.shares * current_prices) + self.balance
        
        # Convert actions to target portfolio weights
        target_weights = actions / np.sum(actions) if np.sum(actions) > 0 else actions
        target_value_in_stocks = portfolio_value_before * np.sum(target_weights)
        target_shares = target_value_in_stocks * target_weights / current_prices
        
        # Calculate shares to buy/sell
        shares_diff = target_shares - self.shares
        
        # Execute trades
        cost = 0
        for i in range(self.num_stocks):
            if shares_diff[i] > 0:  # Buy
                cost_to_buy = shares_diff[i] * current_prices[i] * (1 + self.transaction_cost)
                if cost_to_buy <= self.balance:
                    self.shares[i] += shares_diff[i]
                    self.balance -= cost_to_buy
                    cost += shares_diff[i] * current_prices[i] * self.transaction_cost
                else:
                    # Buy as much as possible with available balance
                    max_shares = self.balance / (current_prices[i] * (1 + self.transaction_cost))
                    self.shares[i] += max_shares
                    self.balance = 0
                    cost += max_shares * current_prices[i] * self.transaction_cost
            
            elif shares_diff[i] < 0:  # Sell
                self.shares[i] += shares_diff[i]  # Reduce shares
                sell_value = -shares_diff[i] * current_prices[i]
                self.balance += sell_value * (1 - self.transaction_cost)
                cost += sell_value * self.transaction_cost
        
        # Move to next day
        self.current_step += 1
        done = self.current_step >= len(self.stock_data) - 1
        
        # Calculate portfolio value after action
        if not done:
            next_prices = self.stock_data.iloc[self.current_step].values[:self.num_stocks]
            portfolio_value_after = np.sum(self.shares * next_prices) + self.balance
        else:
            # If done, use current prices for final valuation
            portfolio_value_after = np.sum(self.shares * current_prices) + self.balance
        
        # Calculate reward: change in portfolio value minus costs
        reward = (portfolio_value_after - portfolio_value_before) / portfolio_value_before - cost / portfolio_value_before
        
        # Get new state
        next_state = self._get_observation()
        
        # Additional info
        info = {
            'portfolio_value': portfolio_value_after,
            'shares': self.shares,
            'balance': self.balance,
            'cost': cost
        }
        
        return next_state, reward, done, info
    
    def render(self):
        """Render the current state of the environment"""
        current_prices = self.stock_data.iloc[self.current_step].values[:self.num_stocks]
        portfolio_value = np.sum(self.shares * current_prices) + self.balance
        
        print(f"Step: {self.current_step}")
        print(f"Portfolio Value: ${portfolio_value:.2f}")
        print(f"Cash Balance: ${self.balance:.2f}")
        print("Holdings:")
        for i in range(self.num_stocks):
            print(f"  Stock {i}: {self.shares[i]:.2f} shares at ${current_prices[i]:.2f} = ${self.shares[i] * current_prices[i]:.2f}")

class MADDPGTrainer:
    """Trainer for MADDPG agents"""
    
    def __init__(self, env, num_agents, state_dim, action_dim, action_high, insurance_strategies=None):
        self.env = env
        self.num_agents = num_agents
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_high = action_high
        
        # Create agents
        self.agents = []
        for i in range(num_agents):
            strategy = None if insurance_strategies is None else insurance_strategies[i]
            agent = MADDPGAgent(state_dim, action_dim, action_high, i, num_agents, 
                               insurance_strategy=strategy)
            self.agents.append(agent)
        
        # Metrics for tracking performance
        self.episode_rewards = []
        self.portfolio_values = []
    
    def train(self, num_episodes, max_steps_per_episode=None):
        """Train agents for a specified number of episodes"""
        for episode in range(num_episodes):
            state = self.env.reset()
            episode_reward = 0
            step = 0
            done = False
            
            while not done:
                if max_steps_per_episode is not None and step >= max_steps_per_episode:
                    break
                
                # Get actions from all agents
                actions = []
                for agent in self.agents:
                    action = agent.get_action(state)
                    actions.append(action)
                
                # Take step in environment with the first agent's action
                # (we'll consider the first agent's perspective for now)
                next_state, reward, done, info = self.env.step(actions[0])
                
                # Store experience in all agents' replay buffers
                for agent in self.agents:
                    agent.remember(state, actions[0], np.array([reward]), next_state, np.array([done]))
                
                # Learn from experiences
                for agent in self.agents:
                    agent.learn(self.agents)
                
                state = next_state
                episode_reward += reward
                step += 1
            
            # Log metrics
            self.episode_rewards.append(episode_reward)
            self.portfolio_values.append(info['portfolio_value'])
            
            # Print progress
            if (episode + 1) % 10 == 0:
                print(f"Episode: {episode+1}/{num_episodes}, Reward: {episode_reward:.4f}, " +
                      f"Portfolio Value: ${info['portfolio_value']:.2f}")
    
    def test(self, num_episodes):
        """Test trained agents"""
        test_rewards = []
        test_portfolio_values = []
        
        for episode in range(num_episodes):
            state = self.env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                # Get actions from first agent (without exploration noise)
                action = self.agents[0].get_action(state, add_noise=False)
                
                # Take step in environment
                next_state, reward, done, info = self.env.step(action)
                
                state = next_state
                episode_reward += reward
            
            test_rewards.append(episode_reward)
            test_portfolio_values.append(info['portfolio_value'])
            
            print(f"Test Episode: {episode+1}/{num_episodes}, Reward: {episode_reward:.4f}, " +
                  f"Portfolio Value: ${info['portfolio_value']:.2f}")
        
        avg_reward = np.mean(test_rewards)
        avg_portfolio_value = np.mean(test_portfolio_values)
        
        print(f"Average Test Reward: {avg_reward:.4f}")
        print(f"Average Test Portfolio Value: ${avg_portfolio_value:.2f}")
        
        return test_rewards, test_portfolio_values
    
    def plot_results(self):
        """Plot training results"""
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.episode_rewards)
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title('Episode Rewards During Training')
        
        plt.subplot(1, 2, 2)
        plt.plot(self.portfolio_values)
        plt.xlabel('Episode')
        plt.ylabel('Portfolio Value ($)')
        plt.title('Portfolio Value During Training')
        
        plt.tight_layout()
        plt.show()

# Universal Portfolio (UP) implementation as a baseline
class UniversalPortfolio:
    """Universal Portfolio strategy as a baseline"""
    
    def __init__(self, num_stocks, initial_balance=10000):
        self.num_stocks = num_stocks
        self.initial_balance = initial_balance
        self.weights = np.ones(num_stocks) / num_stocks  # Equal weight initially
    
    def get_action(self, state):
        """Get portfolio allocation based on UP strategy"""
        # Extract current prices and returns
        prices = state[:self.num_stocks]
        
        # Update weights based on past performance
        if hasattr(self, 'last_prices'):
            returns = prices / self.last_prices
            self.weights = self.weights * returns
            self.weights = self.weights / np.sum(self.weights) if np.sum(self.weights) > 0 else self.weights
        
        self.last_prices = prices.copy()
        return self.weights

# MADQN implementation for comparison
class MADQNAgent:
    """Multi-Agent DQN Agent for comparison"""
    
    def __init__(self, state_dim, action_dim, agent_id, num_agents, discount_factor=0.95, 
                 buffer_size=100000, batch_size=64, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.agent_id = agent_id
        self.num_agents = num_agents
        self.discount_factor = discount_factor
        self.batch_size = batch_size
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Discretize action space
        self.num_actions = 5  # Number of discrete levels per stock
        self.action_levels = np.linspace(0, 1, self.num_actions)
        
        # Create Q-network
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.target_model.set_weights(self.model.get_weights())
        
        # Create replay buffer
        self.buffer = ReplayBuffer(buffer_size, batch_size)
    
    def _build_model(self):
        """Build DQN model"""
        model = tf.keras.Sequential([
            Dense(256, activation='relu', input_shape=(self.state_dim,)),
            Dense(128, activation='relu'),
            Dense(self.num_actions ** self.action_dim, activation='linear')
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        return model
    
    def get_action(self, state, add_noise=True):
        """Get action based on state with epsilon-greedy exploration"""
        state = np.reshape(state, [1, self.state_dim])
        
        if add_noise and np.random.random() < self.epsilon:
            # Random action
            action_idx = np.random.randint(0, self.num_actions ** self.action_dim)
        else:
            # Greedy action
            q_values = self.model.predict(state, verbose=0)[0]
            action_idx = np.argmax(q_values)
        
        # Convert action index to portfolio weights
        action = self._idx_to_action(action_idx)
        return action
    
    def _idx_to_action(self, idx):
        """Convert action index to portfolio weights"""
        action = np.zeros(self.action_dim)
        remaining_idx = idx
        
        for i in range(self.action_dim):
            action_level = remaining_idx % self.num_actions
            action[i] = self.action_levels[action_level]
            remaining_idx = remaining_idx // self.num_actions
        
        # Normalize to sum to 1
        if np.sum(action) > 0:
            action = action / np.sum(action)
        
        return action
    
    def remember(self, state, action, reward, next_state, done):
        """Add experience to replay buffer"""
        # Convert action to index
        action_idx = self._action_to_idx(action)
        self.buffer.add(state, action_idx, reward, next_state, done)
    
    def _action_to_idx(self, action):
        """Convert portfolio weights to action index (approximately)"""
        idx = 0
        multiplier = 1
        
        for i in range(self.action_dim):
            # Find closest action level
            level_idx = np.argmin(np.abs(self.action_levels - action[i]))
            idx += level_idx * multiplier
            multiplier *= self.num_actions
        
        return idx
    
    def learn(self):
        """Update agent's Q-network"""
        if len(self.buffer) < self.batch_size:
            return
        
        # Sample from replay buffer
        states, action_idxs, rewards, next_states, dones = self.buffer.sample()
        
        # Get current Q values
        current_q = self.model.predict(states, verbose=0)
        
        # Get next Q values from target model
        next_q = self.target_model.predict(next_states, verbose=0)
        max_next_q = np.max(next_q, axis=1)
        
        # Update Q values for taken actions
        for i in range(len(states)):
            current_q[i, action_idxs[i]] = rewards[i] + (1 - dones[i]) * self.discount_factor * max_next_q[i]
        
        # Train model
        loss = self.model.train_on_batch(states, current_q)
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        # Update target network
        self.target_model.set_weights(self.model.get_weights())
        
        return loss

def fetch_stock_data(symbols, start_date, end_date):
    """Fetch historical stock data using yfinance"""
    data = yf.download(symbols, start=start_date, end=end_date)['Adj Close']
    
    # Handle potential missing values
    data = data.fillna(method='ffill')
    
    return data

def simulate_stock_data(num_stocks, num_days, volatility=0.01, drift=0.0002):
    """Generate simulated stock price data"""
    prices = np.zeros((num_days, num_stocks))
    
    # Initial prices
    prices[0] = np.random.uniform(50, 200, size=num_stocks)
    
    # Generate price series with drift and volatility
    for i in range(1, num_days):
        for j in range(num_stocks):
            # Random shock
            shock = np.random.normal(drift, volatility)
            prices[i, j] = prices[i-1, j] * (1 + shock)
    
    # Convert to DataFrame
    dates = [datetime.now() + timedelta(days=i) for i in range(num_days)]
    df = pd.DataFrame(prices, index=dates, columns=[f'Stock_{i}' for i in range(num_stocks)])
    
    return df

def run_experiment(use_real_data=False, num_stocks=5, num_episodes=100):
    """Run the full experiment comparing different strategies"""
    # Set up data
    if use_real_data:
        # Use real stock data
        symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META'][:num_stocks]
        train_start = '2018-01-01'
        train_end = '2020-12-31'
        test_start = '2021-01-01'
        test_end = '2021-12-31'
        
        train_data = fetch_stock_data(symbols, train_start, train_end)
        test_data = fetch_stock_data(symbols, test_start, test_end)
    else:
        # Use simulated data
        num_train_days = 750  # ~3 years of trading days
        num_test_days = 252   # ~1 year of trading days
        
        train_data = simulate_stock_data(num_stocks, num_train_days)
        test_data = simulate_stock_data(num_stocks, num_test_days, volatility=0.015)  # Slightly higher volatility for test
    
    # Set up environment
    train_env = StockMarketEnv(train_data, num_stocks, initial_balance=10000, transaction_cost=0.001)
    test_env = StockMarketEnv(test_data, num_stocks, initial_balance=10000, transaction_cost=0.001)
    
    state_dim = num_stocks * 2 + 1  # prices + shares + balance
    action_dim = num_stocks
    action_high = 1.0
    
    # Train and test MADDPG
    print("\n--- Training MADDPG ---")
    maddpg_trainer = MADDPGTrainer(
        train_env, 
        num_agents=3, 
        state_dim=state_dim, 
        action_dim=action_dim, 
        action_high=action_high,
        insurance_strategies=[None, None, None]
    )
    maddpg_trainer.train(num_episodes)
    _, maddpg_values = maddpg_trainer.test(1)
    
    # Train and test CPPI-MADDPG
    print("\n--- Training CPPI-MADDPG ---")
    cppi_maddpg_trainer = MADDPGTrainer(
        train_env, 
        num_agents=3, 
        state_dim=state_dim, 
        action_dim=action_dim, 
        action_high=action_high,
        insurance_strategies=['CPPI', 'CPPI', 'CPPI']
    )
    cppi_maddpg_trainer.train(num_episodes)
    _, cppi_maddpg_values = cppi_maddpg_trainer.test(1)
    
    # Train and test TIPP-MADDPG
    print("\n--- Training TIPP-MADDPG ---")
    tipp_maddpg_trainer = MADDPGTrainer(
        train_env, 
        num_agents=3, 
        state_dim=state_dim, 
        action_dim=action_dim, 
        action_high=action_high,
        insurance_strategies=['TIPP', 'TIPP', 'TIPP']
    )
    tipp_maddpg_trainer.train(num_episodes)
    _, tipp_maddpg_values = tipp_maddpg_trainer.test(1)
    
    # Test Universal Portfolio
    print("\n--- Testing Universal Portfolio ---")
    up = UniversalPortfolio(num_stocks)
    state = test_env.reset()
    up_values = []
    
    done = False
    while not done:
        action = up.get_action(state)
        state, _, done, info = test_env.step(action)
        up_values.append(info['portfolio_value'])
    
    # Test MADQN
    print("\n--- Training MADQN ---")
    madqn_agent = MADQNAgent(state_dim, action_dim, 0, 1)
    madqn_values = []
    
    for episode in range(num_episodes // 2):  # Fewer episodes for DQN due to computational complexity
        state = train_env.reset()
        done = False
        
        while not done:
            action = madqn_agent.get_action(state)
            next_state, reward, done, info = train_env.step(action)
            madqn_agent.remember(state, action, reward, next_state, done)
            madqn_agent.learn()
            state = next_state
        
        if (episode + 1) % 10 == 0:
            print(f"MADQN Episode: {episode+1}/{num_episodes//2}")
    
    # Test MADQN
    state = test_env.reset()
    done = False
    
    while not done:
        action = madqn_agent.get_action(state, add_noise=False)
        state, _, done, info = test_env.step(action)
        madqn_values.append(info['portfolio_value'])
    
    # Compare results
    final_values = {
        'UP': up_values[-1],
        'MADQN': madqn_values[-1],
        'MADDPG': maddpg_values[-1],
        'CPPI-MADDPG': cppi_maddpg_values[-1],
        'TIPP-MADDPG': tipp_maddpg_values[-1]
    }
    
    # Calculate annualized returns
    initial_value = 10000
    days = len(test_data)
    
    annual_returns = {}
    for strategy, value in final_values.items():
        total_return = (value - initial_value) / initial_value
        annual_return = (1 + total_return) ** (252 / days) - 1  # Annualize assuming 252 trading days/year
        annual_returns[strategy] = annual_return * 100  # Convert to percentage
    
    # Calculate Sharpe ratios (simplified)
    sharpe_ratios = {}
    
    # UP
    up_returns = np.diff(up_values) / up_values[:-1]
    sharpe_ratios['UP'] = np.mean(up_returns) / np.std(up_returns) if np.std(up_returns) > 0 else 0
    
    # MADQN
    madqn_returns = np.diff(madqn_values) / madqn_values[:-1]
    sharpe_ratios['MADQN'] = np.mean(madqn_returns) / np.std(madqn_returns) if np.std(madqn_returns) > 0 else 0
    
    # MADDPG
    maddpg_returns = np.diff(maddpg_values) / maddpg_values[:-1]
    sharpe_ratios['MADDPG'] = np.mean(maddpg_returns) / np.std(maddpg_returns) if np.std(maddpg_returns) > 0 else 0
    
    # CPPI-MADDPG
    cppi_returns = np.diff(cppi_maddpg_values) / cppi_maddpg_values[:-1]
    sharpe_ratios['CPPI-MADDPG'] = np.mean(cppi_returns) / np.std(cppi_returns) if np.std(cppi_returns) > 0 else 0
    
    # TIPP-MADDPG
    tipp_returns = np.diff(tipp_maddpg_values) / tipp_maddpg_values[:-1]
    sharpe_ratios['TIPP-MADDPG'] = np.mean(tipp_returns) / np.std(tipp_returns) if np.std(tipp_returns) > 0 else 0
    
    # Calculate maximum drawdowns
    max_drawdowns = {}
    
    # UP
    max_drawdowns['UP'] = calculate_max_drawdown(up_values)
    
    # MADQN
    max_drawdowns['MADQN'] = calculate_max_drawdown(madqn_values)
    
    # MADDPG
    max_drawdowns['MADDPG'] = calculate_max_drawdown(maddpg_values)
    
    # CPPI-MADDPG
    max_drawdowns['CPPI-MADDPG'] = calculate_max_drawdown(cppi_maddpg_values)
    
    # TIPP-MADDPG
    max_drawdowns['TIPP-MADDPG'] = calculate_max_drawdown(tipp_maddpg_values)
    
    # Print results
    print("\n----- Results -----")
    print(f"{'Strategy':<15} {'Final Value':<15} {'Annual Return':<15} {'Sharpe Ratio':<15} {'Max Drawdown':<15}")
    print("-" * 75)
    for strategy in ['UP', 'MADQN', 'MADDPG', 'CPPI-MADDPG', 'TIPP-MADDPG']:
        print(f"{strategy:<15} ${final_values[strategy]:<14.2f} {annual_returns[strategy]:<14.2f}% {sharpe_ratios[strategy]:<14.2f} {max_drawdowns[strategy]:<14.2f}%")
    
    # Plot portfolio values over time
    plt.figure(figsize=(12, 6))
    
    # Create time axis
    time_axis = range(len(test_data))
    
    # Plot each strategy
    plt.plot(time_axis[:len(up_values)], up_values, label='UP')
    plt.plot(time_axis[:len(madqn_values)], madqn_values, label='MADQN')
    plt.plot(time_axis[:len(maddpg_values)], maddpg_values, label='MADDPG')
    plt.plot(time_axis[:len(cppi_maddpg_values)], cppi_maddpg_values, label='CPPI-MADDPG')
    plt.plot(time_axis[:len(tipp_maddpg_values)], tipp_maddpg_values, label='TIPP-MADDPG')
    
    plt.xlabel('Trading Day')
    plt.ylabel('Portfolio Value ($)')
    plt.title('Portfolio Value Over Time')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('portfolio_comparison.png')
    plt.show()
    
    # Return results for further analysis
    return {
        'final_values': final_values,
        'annual_returns': annual_returns,
        'sharpe_ratios': sharpe_ratios,
        'max_drawdowns': max_drawdowns,
        'portfolio_values': {
            'UP': up_values,
            'MADQN': madqn_values,
            'MADDPG': maddpg_values,
            'CPPI-MADDPG': cppi_maddpg_values,
            'TIPP-MADDPG': tipp_maddpg_values
        }
    }

def calculate_max_drawdown(portfolio_values):
    """Calculate maximum drawdown from portfolio values"""
    portfolio_values = np.array(portfolio_values)
    peak = np.maximum.accumulate(portfolio_values)
    drawdown = (portfolio_values - peak) / peak * 100
    return abs(np.min(drawdown))

def analyze_asset_allocation(trainer, test_env, num_steps=100):
    """Analyze how agents allocate assets across stocks"""
    agent = trainer.agents[0]  # Use the first agent
    state = test_env.reset()
    
    allocations = []
    
    for _ in range(num_steps):
        action = agent.get_action(state, add_noise=False)
        allocations.append(action)
        
        next_state, _, done, _ = test_env.step(action)
        state = next_state
        
        if done:
            break
    
    # Convert to numpy array
    allocations = np.array(allocations)
    
    # Plot allocation heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(allocations, cmap='viridis', xticklabels=[f'Stock {i}' for i in range(test_env.num_stocks)], 
                yticklabels=[f'Day {i}' for i in range(len(allocations))])
    plt.xlabel('Stocks')
    plt.ylabel('Trading Days')
    plt.title('Asset Allocation Over Time')
    plt.tight_layout()
    plt.savefig('asset_allocation.png')
    plt.show()
    
    return allocations

# Main execution
if __name__ == "__main__":
    # Set to True to use real data or False for simulated data
    use_real_data = False
    
    # Run experiment
    results = run_experiment(use_real_data=use_real_data, num_stocks=5, num_episodes=50)
    
    # Print final summary
    print("\nExperiment complete. Results saved to portfolio_comparison.png")

