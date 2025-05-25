import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import random
from collections import deque
import seaborn as sns
from tqdm import tqdm
import os
import time
from typing import List, Tuple, Dict

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
random.seed(42)

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ================ DATA GENERATION ================

def generate_stock_data(num_stocks=5, num_days=750, seed=42):
    """
    Generate synthetic stock data with various price patterns.
    
    Args:
        num_stocks: Number of stocks to simulate
        num_days: Number of trading days
        seed: Random seed for reproducibility
        
    Returns:
        DataFrame with OHLC prices for each stock
    """
    np.random.seed(seed)
    
    # Base parameters
    base_price = 100.0
    vol = 0.01  # Daily volatility
    
    # Generate different trend patterns for stocks
    trends = []
    for i in range(num_stocks):
        # Each stock has a slightly different trend
        if i % 3 == 0:
            # Upward trend
            trend = np.linspace(0, 0.0003, num_days)
        elif i % 3 == 1:
            # Downward trend
            trend = np.linspace(0, -0.0002, num_days)
        else:
            # Sideways with slight upward bias
            trend = np.linspace(0, 0.0001, num_days)
            
        # Add regime shifts
        for shift in range(3):
            # Random regime shift point
            shift_point = np.random.randint(num_days // 4, 3 * num_days // 4)
            
            # Change the trend after the shift point
            if np.random.random() > 0.5:
                trend[shift_point:] += 0.0002
            else:
                trend[shift_point:] -= 0.0002
                
        trends.append(trend)
    
    # Create DataFrames for each stock
    all_data = {}
    stocks = [f"Stock_{i+1}" for i in range(num_stocks)]
    
    for i, stock in enumerate(stocks):
        # Generate log returns with the trend and volatility
        log_returns = trends[i] + np.random.normal(0, vol, num_days)
        
        # Generate price series from log returns
        prices = base_price * np.exp(np.cumsum(log_returns))
        
        # Generate OHLC data
        dates = pd.date_range(start='2015-01-01', periods=num_days)
        
        stock_data = pd.DataFrame({
            'Date': dates,
            'Open': prices * np.random.uniform(0.995, 1.005, num_days),
            'High': prices * np.random.uniform(1.001, 1.015, num_days),
            'Low': prices * np.random.uniform(0.985, 0.999, num_days),
            'Close': prices,
            'Volume': np.random.randint(100000, 1000000, num_days)
        })
        
        all_data[stock] = stock_data
    
    return all_data

# ================ SUM TREE FOR PRIORITIZED EXPERIENCE REPLAY ================

class SumTree:
    """
    SumTree data structure for prioritized experience replay.
    """
    def __init__(self, capacity):
        self.capacity = capacity  # Number of leaf nodes (samples)
        self.tree = np.zeros(2 * capacity - 1)  # Binary tree structure
        self.data_pointer = 0  # Pointer to current data
        self.n_entries = 0  # Number of entries in the tree
        
    def add(self, priority, data):
        """Add new sample to the tree with given priority."""
        # Find the leaf index
        tree_idx = self.data_pointer + self.capacity - 1
        
        # Update the leaf
        self.update(tree_idx, priority)
        
        # Add the experience to data storage
        self.data_pointer += 1
        
        # If we reach the capacity, go back to the beginning
        if self.data_pointer >= self.capacity:
            self.data_pointer = 0
            
        if self.n_entries < self.capacity:
            self.n_entries += 1
            
    def update(self, tree_idx, priority):
        """Update the priority of a sample."""
        # Change = new priority - old priority
        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        
        # Propagate the change through the tree
        while tree_idx != 0:
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change
            
    def get_leaf(self, v):
        """Search for a sample based on a priority value."""
        parent_idx = 0
        
        while True:
            left_child_idx = 2 * parent_idx + 1
            right_child_idx = left_child_idx + 1
            
            # If we reach a leaf node, return
            if left_child_idx >= len(self.tree):
                leaf_idx = parent_idx
                break
                
            # Otherwise go to the appropriate child
            if v <= self.tree[left_child_idx]:
                parent_idx = left_child_idx
            else:
                v -= self.tree[left_child_idx]
                parent_idx = right_child_idx
                
        data_idx = leaf_idx - (self.capacity - 1)
        
        return leaf_idx, self.tree[leaf_idx], data_idx
    
    def total_priority(self):
        """Return the sum of all priorities."""
        return self.tree[0]


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay buffer for DQN.
    """
    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_increment=0.001):
        self.capacity = capacity
        self.alpha = alpha  # How much prioritization to use (0 = uniform)
        self.beta = beta    # Importance sampling correction (0 = no correction)
        self.beta_increment = beta_increment  # Increment beta towards 1
        self.max_priority = 1.0  # Max priority
        self.tree = SumTree(capacity)
        self.memory = np.empty(capacity, dtype=object)
        
    def add(self, error, sample):
        """Add a new experience to memory with given TD error."""
        # Convert error to priority with alpha
        priority = (np.abs(error) + 1e-5) ** self.alpha
        
        # Clip priority to prevent overflow
        priority = min(priority, self.max_priority)
        
        # Add to sum tree and memory
        self.tree.add(priority, sample)
        self.memory[self.tree.data_pointer - 1] = sample
        
    def sample(self, batch_size):
        """Sample a batch of experiences based on their priority."""
        batch = []
        idxs = []
        priorities = []
        
        # Calculate the priority segment for each sample
        segment = self.tree.total_priority() / batch_size
        
        # Increase beta over time for less bias
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        for i in range(batch_size):
            # Get a random value from the segment
            a = segment * i
            b = segment * (i + 1)
            v = np.random.uniform(a, b)
            
            # Get an experience using the priority value
            idx, priority, data_idx = self.tree.get_leaf(v)
            
            # Ensure the data_idx is valid
            if data_idx < len(self.memory) and self.memory[data_idx] is not None:
                batch.append(self.memory[data_idx])
                idxs.append(idx)
                priorities.append(priority)
            
        # Calculate importance sampling weights
        sampling_probabilities = priorities / self.tree.total_priority()
        is_weights = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        is_weights /= is_weights.max()  # Normalize weights
        
        return batch, idxs, is_weights
    
    def update_priorities(self, idxs, errors):
        """Update priorities based on new TD errors."""
        for i, idx in enumerate(idxs):
            # Convert error to priority with alpha
            priority = (np.abs(errors[i]) + 1e-5) ** self.alpha
            
            # Clip priority to prevent overflow
            priority = min(priority, self.max_priority)
            
            # Update the priority in the tree
            self.tree.update(idx, priority)
            
            # Update max priority for new experiences
            self.max_priority = max(self.max_priority, priority)


# ================ PORTFOLIO ENVIRONMENT ================

class PortfolioEnvironment:
    """
    Portfolio management environment with discrete actions for DQN.
    """
    def __init__(self, data, lookback_window=10, N=3, commission=0.0):
        """
        Initialize the environment.
        
        Args:
            data: Dictionary of DataFrames with OHLC data for each stock
            lookback_window: Number of days to look back for state representation
            N: Number of divisions for the portfolio (controls action granularity)
            commission: Transaction cost as a percentage
        """
        self.data = data
        self.lookback_window = lookback_window
        self.N = N  # Number of divisions for discrete actions
        self.commission = commission
        
        # Extract stock symbols and dates
        self.symbols = list(data.keys())
        self.num_stocks = len(self.symbols)
        self.dates = data[self.symbols[0]]['Date'].values
        self.num_days = len(self.dates)
        
        # Calculate number of trading days
        self.trading_days = self.num_days - self.lookback_window
        
        # Initialize current day
        self.current_day = self.lookback_window
        
        # Initialize portfolio
        self.cash_idx = 0  # Index for cash position
        self.num_assets = self.num_stocks + 1  # Including cash
        self.weights = np.zeros(self.num_assets)
        self.weights[self.cash_idx] = 1.0  # Start with all cash
        
        # Generate discrete action space
        self.actions = self._generate_actions()
        self.num_actions = len(self.actions)
        
        print(f"Environment initialized with {self.num_stocks} stocks, {self.trading_days} trading days, "
              f"and {self.num_actions} possible actions")
        
    def _generate_actions(self):
        """
        Generate discrete action space for portfolio weights.
        Each action represents a specific allocation of portfolio.
        """
        actions = []
        
        # Algorithm 1 from the paper - Action Discretization
        def generate_weights(remaining, idx, current_weights):
            if idx == self.num_assets - 1:
                # Assign remaining weight to the last asset
                new_weights = current_weights.copy()
                new_weights[idx] = remaining / self.N
                actions.append(new_weights)
                return
            
            # Try different weights for current asset
            for w in range(remaining + 1):
                new_weights = current_weights.copy()
                new_weights[idx] = w / self.N
                generate_weights(remaining - w, idx + 1, new_weights)
        
        # Start recursion with all weight available and index 0
        initial_weights = np.zeros(self.num_assets)
        generate_weights(self.N, 0, initial_weights)
        
        return actions
    
    def reset(self):
        """Reset the environment to the start of the trading period."""
        self.current_day = self.lookback_window
        self.weights = np.zeros(self.num_assets)
        self.weights[self.cash_idx] = 1.0  # Start with all cash
        self.portfolio_value = 1.0  # Start with 1.0 (100%)
        return self._get_state()
    
    def step(self, action_idx):
        """
        Take a step in the environment.
        
        Args:
            action_idx: Index of the action to take (portfolio allocation)
            
        Returns:
            next_state: The next state after taking the action
            reward: The reward for taking the action
            done: Whether the episode is done
            info: Additional information
        """
        # Get the action (weight vector)
        new_weights = self.actions[action_idx]
        
        # Calculate transaction costs
        if self.commission > 0:
            cost = np.sum(np.abs(new_weights - self.weights)) * self.commission
        else:
            cost = 0
        
        # Update weights
        old_weights = self.weights.copy()
        self.weights = new_weights
        
        # Move to next day
        self.current_day += 1
        
        # Calculate returns
        returns = self._calculate_returns()
        
        # Update portfolio value (including transaction costs)
        portfolio_return = np.sum(old_weights * returns) - cost
        self.portfolio_value *= (1 + portfolio_return)
        
        # Calculate reward (log return as defined in the paper)
        reward = np.log(1 + portfolio_return)
        
        # Check if done
        done = self.current_day >= self.num_days - 1
        
        # Get next state
        next_state = self._get_state()
        
        # Additional info
        info = {
            'portfolio_value': self.portfolio_value,
            'returns': returns,
            'weights': self.weights,
            'cost': cost
        }
        
        return next_state, reward, done, info
    
    def _calculate_returns(self):
        """Calculate asset returns from close to close (including cash)."""
        returns = np.zeros(self.num_assets)
        
        # Cash return is 0
        returns[self.cash_idx] = 0
        
        # Calculate returns for each stock
        for i, symbol in enumerate(self.symbols):
            prev_close = self.data[symbol].iloc[self.current_day-1]['Close']
            curr_close = self.data[symbol].iloc[self.current_day]['Close']
            returns[i+1] = curr_close / prev_close - 1
            
        return returns
    
    def _get_state(self):
        """
        Construct the state representation as defined in the paper.
        State includes price tensor and previous weights.
        """
        # Extract lookback window price data
        start_idx = self.current_day - self.lookback_window
        end_idx = self.current_day
        
        # Initialize price tensor with shape (num_stocks, lookback_window, 4)
        # 4 channels: Open, High, Low, Close (normalized)
        price_tensor = np.zeros((self.num_stocks, self.lookback_window, 4))
        
        # Fill price tensor
        for i, symbol in enumerate(self.symbols):
            stock_data = self.data[symbol].iloc[start_idx:end_idx]
            
            # Normalize prices by the current closing price
            current_close = stock_data.iloc[-1]['Close']
            
            for j in range(self.lookback_window):
                price_tensor[i, j, 0] = stock_data.iloc[j]['Open'] / current_close
                price_tensor[i, j, 1] = stock_data.iloc[j]['High'] / current_close
                price_tensor[i, j, 2] = stock_data.iloc[j]['Low'] / current_close
                price_tensor[i, j, 3] = stock_data.iloc[j]['Close'] / current_close
        
        # Apply expansion coefficient (alpha) as defined in the paper
        alpha = 10
        price_tensor = alpha * (price_tensor - 1)
        
        # Current portfolio weights (excluding cash for CNN input)
        weights = self.weights[1:]
        
        # Return state as dictionary with tensors
        state = {
            'price_tensor': price_tensor,
            'weights': weights
        }
        
        return state


# ================ DQN MODEL ================

class SimpleDuelingDQN(nn.Module):
    """
    Simplified Dueling DQN architecture for portfolio management.
    This simpler architecture avoids complex tensor shape manipulations.
    """
    def __init__(self, num_stocks, lookback_window, num_actions):
        super(SimpleDuelingDQN, self).__init__()
        
        # Flatten the input price tensor
        self.input_size = num_stocks * lookback_window * 4
        self.weights_size = num_stocks
        
        # Feature extraction network
        self.feature_network = nn.Sequential(
            nn.Linear(self.input_size + self.weights_size, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
        )
        
        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 1)
        )
        
        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.Linear(64, num_actions)
        )
        
    def forward(self, price_tensor, weights):
        """
        Forward pass of the network.
        
        Args:
            price_tensor: Tensor of shape (batch, num_stocks, lookback_window, 4)
            weights: Tensor of shape (batch, num_stocks) - previous portfolio weights
            
        Returns:
            q_values: Q-values for each action
        """
        batch_size = price_tensor.size(0)
        
        # Flatten the price tensor
        price_flat = price_tensor.reshape(batch_size, -1)
        
        # Concatenate with weights
        combined = torch.cat([price_flat, weights], dim=1)
        
        # Extract features
        features = self.feature_network(combined)
        
        # Get value and advantage
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        
        # Combine using dueling formula
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        
        return q_values


# ================ DQN AGENT ================

class DQNAgent:
    """
    DQN Agent for portfolio management.
    """
    def __init__(
        self,
        state_dim,
        num_actions,
        num_stocks,
        lookback_window,
        learning_rate=1e-4,
        gamma=0.99,
        buffer_size=10000,
        batch_size=64,
        target_update=10,
        eps_start=1.0,
        eps_end=0.01,
        eps_decay=0.995,
        double_dqn=True
    ):
        self.num_actions = num_actions
        self.num_stocks = num_stocks
        self.lookback_window = lookback_window
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update = target_update
        self.double_dqn = double_dqn
        
        # Epsilon greedy parameters
        self.eps = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        
        # Create evaluation and target networks
        self.eval_net = SimpleDuelingDQN(num_stocks, lookback_window, num_actions).to(device)
        self.target_net = SimpleDuelingDQN(num_stocks, lookback_window, num_actions).to(device)
        self.target_net.load_state_dict(self.eval_net.state_dict())
        self.target_net.eval()  # Target network is in evaluation mode
        
        # Optimizer
        self.optimizer = optim.Adam(self.eval_net.parameters(), lr=learning_rate)
        
        # Replay buffer
        self.buffer = PrioritizedReplayBuffer(buffer_size)
        
        # Tracking variables
        self.learn_step_counter = 0
        self.initial_error = 1.0  # Initial error for new experiences
        
    def select_action(self, state, training=True):
        """
        Select an action using epsilon-greedy policy.
        
        Args:
            state: The current state
            training: Whether the agent is in training mode
            
        Returns:
            action_idx: Index of the selected action
        """
        # Extract state components
        price_tensor = torch.FloatTensor(state['price_tensor']).unsqueeze(0).to(device)
        weights = torch.FloatTensor(state['weights']).unsqueeze(0).to(device)
        
        # Epsilon-greedy action selection
        if training and random.random() < self.eps:
            # Random action
            action_idx = random.randint(0, self.num_actions - 1)
        else:
            # Greedy action
            with torch.no_grad():
                q_values = self.eval_net(price_tensor, weights)
                action_idx = q_values.max(1)[1].item()
        
        return action_idx
    
    def store_transition(self, state, action, reward, next_state, done):
        """
        Store a transition in the replay buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether the episode is done
        """
        # Prepare the sample
        sample = (state, action, reward, next_state, done)
        
        # Add to buffer with initial TD error
        self.buffer.add(self.initial_error, sample)
    
    def learn(self):
        """Train the agent by sampling from the replay buffer."""
        # Check if we have enough samples
        if self.buffer.tree.n_entries < self.batch_size:
            return
        
        # Sample a batch
        batch, idxs, is_weights = self.buffer.sample(self.batch_size)
        
        # Unpack the batch
        states_batch, actions_batch, rewards_batch, next_states_batch, dones_batch = zip(*batch)
        
        # Prepare batch tensors
        batch_size = len(batch)
        
        # Prepare price tensors and weights
        price_tensors = torch.FloatTensor(np.array([s['price_tensor'] for s in states_batch])).to(device)
        weights = torch.FloatTensor(np.array([s['weights'] for s in states_batch])).to(device)
        
        next_price_tensors = torch.FloatTensor(np.array([s['price_tensor'] for s in next_states_batch])).to(device)
        next_weights = torch.FloatTensor(np.array([s['weights'] for s in next_states_batch])).to(device)
        
        actions = torch.LongTensor(actions_batch).to(device)
        rewards = torch.FloatTensor(rewards_batch).to(device)
        dones = torch.FloatTensor(dones_batch).to(device)
        is_weights = torch.FloatTensor(is_weights).to(device)
        
        # Get current Q values
        q_eval = self.eval_net(price_tensors, weights)
        q_eval = q_eval.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Get next Q values
        if self.double_dqn:
            # Double DQN
            with torch.no_grad():
                # Get actions from evaluation net
                q_next_eval = self.eval_net(next_price_tensors, next_weights)
                next_actions = q_next_eval.max(1)[1]
                
                # Get Q values from target net for those actions
                q_next_target = self.target_net(next_price_tensors, next_weights)
                q_next = q_next_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)
                
                # Calculate target Q values
                q_target = rewards + self.gamma * q_next * (1 - dones)
        else:
            # Standard DQN
            with torch.no_grad():
                q_next = self.target_net(next_price_tensors, next_weights).max(1)[0]
                q_target = rewards + self.gamma * q_next * (1 - dones)
        
        # Calculate TD errors for updating priorities
        td_errors = torch.abs(q_target - q_eval).detach().cpu().numpy()
        
        # Update priorities in buffer
        self.buffer.update_priorities(idxs, td_errors)
        
        # Calculate loss with importance sampling weights
        loss = (is_weights * F.mse_loss(q_eval, q_target, reduction='none')).mean()
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        # Clip gradients to prevent explosion
        torch.nn.utils.clip_grad_norm_(self.eval_net.parameters(), 1.0)
        self.optimizer.step()
        
        # Update target network periodically
        self.learn_step_counter += 1
        if self.learn_step_counter % self.target_update == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        
        # Decay epsilon
        self.eps = max(self.eps_end, self.eps * self.eps_decay)
        
        return loss.item()


# ================ BASELINE PORTFOLIO STRATEGIES ================

class UniformBuyAndHold:
    """
    Uniform Buy and Hold strategy.
    Equally distributes funds across all assets and holds.
    """
    def __init__(self, num_assets):
        self.num_assets = num_assets
        self.weights = np.ones(num_assets) / num_assets
        
    def get_action(self, state):
        return self.weights

class ConstantRebalanced:
    """
    Constant Rebalanced Portfolio (CRP) strategy.
    Rebalances to target weights every period.
    """
    def __init__(self, num_assets):
        self.num_assets = num_assets
        # Equal weight to all assets
        self.target_weights = np.ones(num_assets) / num_assets
        
    def get_action(self, state):
        return self.target_weights

class OLMAR:
    """
    Online Moving Average Reversion strategy.
    """
    def __init__(self, num_assets, window=5, eps=10):
        self.num_assets = num_assets
        self.window = window
        self.eps = eps
        self.weights = np.ones(num_assets) / num_assets
        
    def get_action(self, price_history):
        """
        Update portfolio based on OLMAR strategy.
        
        Args:
            price_history: Array of shape (window, num_assets) with asset prices
            
        Returns:
            weights: New portfolio weights
        """
        if len(price_history) < self.window:
            return self.weights
        
        # Calculate moving average price for each asset
        ma_prices = np.mean(price_history[-self.window:], axis=0)
        
        # Current prices are the most recent prices
        curr_prices = price_history[-1]
        
        # Expected price relatives (ma / current)
        b = ma_prices / curr_prices
        
        # Find optimal weights using OLMAR formula
        # Implement simplified version for demonstration
        mean_b = np.mean(b)
        
        # Check if mean_b is close to 1 (no significant mean reversion signal)
        if abs(mean_b - 1) < 1e-5:
            return self.weights
        
        # Calculate weights based on mean reversion signal
        weights = np.zeros(self.num_assets)
        
        # Assets with expected returns above average get more weight
        above_avg = b > mean_b
        if np.any(above_avg):
            weights[above_avg] = b[above_avg] / np.sum(b[above_avg])
        else:
            # If no assets above average, use equal weights
            weights = np.ones(self.num_assets) / self.num_assets
        
        self.weights = weights
        return self.weights


# ================ TRAINING AND EVALUATION FUNCTIONS ================

def train_dqn(env, agent, num_episodes, max_steps=None):
    """
    Train the DQN agent.
    
    Args:
        env: The portfolio environment
        agent: The DQN agent
        num_episodes: Number of episodes to train
        max_steps: Maximum steps per episode (None for full episode)
        
    Returns:
        rewards_history: List of episode rewards
        portfolio_values: List of final portfolio values
    """
    rewards_history = []
    portfolio_values = []
    
    for episode in tqdm(range(num_episodes), desc="Training"):
        state = env.reset()
        episode_reward = 0
        step = 0
        done = False
        
        while not done:
            # Select action
            action = agent.select_action(state)
            
            # Take action
            next_state, reward, done, info = env.step(action)
            
            # Store transition
            agent.store_transition(state, action, reward, next_state, done)
            
            # Learn
            loss = agent.learn()
            
            # Update state and counters
            state = next_state
            episode_reward += reward
            step += 1
            
            if max_steps is not None and step >= max_steps:
                break
        
        # Log results
        rewards_history.append(episode_reward)
        portfolio_values.append(info['portfolio_value'])
        
        if episode % 10 == 0:
            print(f"Episode {episode}, Reward: {episode_reward:.4f}, "
                  f"Portfolio Value: {info['portfolio_value']:.4f}, "
                  f"Epsilon: {agent.eps:.4f}")
    
    return rewards_history, portfolio_values

def evaluate_strategy(env, strategy, is_dqn=False):
    """
    Evaluate a portfolio strategy.
    
    Args:
        env: The portfolio environment
        strategy: The strategy to evaluate (DQN agent or baseline)
        is_dqn: Whether the strategy is a DQN agent
        
    Returns:
        portfolio_values: Daily portfolio values
        returns: Daily returns
        weights_history: History of portfolio weights
    """
    state = env.reset()
    done = False
    portfolio_values = [1.0]
    returns = []
    weights_history = [env.weights.copy()]
    
    while not done:
        if is_dqn:
            action = strategy.select_action(state, training=False)
            next_state, reward, done, info = env.step(action)
        else:
            # For baseline strategies
            if isinstance(strategy, OLMAR):
                # Create price history for OLMAR
                price_history = []
                for i in range(env.lookback_window):
                    day_idx = env.current_day - env.lookback_window + i
                    prices = [env.data[symbol].iloc[day_idx]['Close'] for symbol in env.symbols]
                    price_history.append(prices)
                
                weights = strategy.get_action(np.array(price_history))
                
                # Find closest discrete action
                distances = [np.sum(np.abs(weights - action[1:])) for action in env.actions]
                action = np.argmin(distances)
                
                next_state, reward, done, info = env.step(action)
            else:
                weights = strategy.get_action(state)
                
                # Find closest discrete action
                distances = [np.sum(np.abs(weights - action[1:])) for action in env.actions]
                action = np.argmin(distances)
                
                next_state, reward, done, info = env.step(action)
        
        state = next_state
        portfolio_values.append(info['portfolio_value'])
        returns.append(reward)
        weights_history.append(env.weights.copy())
    
    return portfolio_values, returns, weights_history

def calculate_metrics(portfolio_values, returns):
    """
    Calculate performance metrics for a strategy.
    
    Args:
        portfolio_values: List of portfolio values
        returns: List of returns
        
    Returns:
        metrics: Dictionary of performance metrics
    """
    # Convert to numpy arrays
    portfolio_values = np.array(portfolio_values)
    returns = np.array(returns)
    
    # Calculate metrics
    total_return = portfolio_values[-1] / portfolio_values[0] - 1
    
    # Annualized return (assuming 252 trading days per year)
    n_days = len(returns)
    annual_return = (1 + total_return) ** (252 / n_days) - 1
    
    # Volatility (annualized)
    daily_vol = np.std(returns)
    annual_vol = daily_vol * np.sqrt(252)
    
    # Sharpe ratio (assuming risk-free rate of 0)
    sharpe_ratio = annual_return / annual_vol if annual_vol > 0 else 0
    
    # Maximum drawdown
    peak = np.maximum.accumulate(portfolio_values)
    drawdown = (peak - portfolio_values) / peak
    max_drawdown = drawdown.max()
    
    # Calmar ratio
    calmar_ratio = annual_return / max_drawdown if max_drawdown > 0 else float('inf')
    
    metrics = {
        'Total Return (%)': total_return * 100,
        'Annual Return (%)': annual_return * 100,
        'Annual Volatility (%)': annual_vol * 100,
        'Sharpe Ratio': sharpe_ratio,
        'Max Drawdown (%)': max_drawdown * 100,
        'Calmar Ratio': calmar_ratio
    }
    
    return metrics

def plot_results(portfolio_values_dict, title="Portfolio Performance Comparison"):
    """
    Plot portfolio values for different strategies.
    
    Args:
        portfolio_values_dict: Dictionary mapping strategy names to portfolio values
        title: Plot title
    """
    plt.figure(figsize=(12, 6))
    
    for name, values in portfolio_values_dict.items():
        plt.plot(values, label=name)
    
    plt.title(title)
    plt.xlabel('Trading Day')
    plt.ylabel('Portfolio Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def plot_weights_history(weights_history, asset_names, title="Portfolio Weights Over Time"):
    """
    Plot the evolution of portfolio weights over time.
    
    Args:
        weights_history: List of weight arrays
        asset_names: Names of the assets (including cash)
        title: Plot title
    """
    weights_history = np.array(weights_history)
    
    plt.figure(figsize=(12, 6))
    
    for i in range(weights_history.shape[1]):
        plt.plot(weights_history[:, i], label=asset_names[i])
    
    plt.title(title)
    plt.xlabel('Trading Day')
    plt.ylabel('Weight')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


# ================ MAIN FUNCTION ================

def main():
    # Generate stock data
    print("Generating synthetic stock data...")
    stock_data = generate_stock_data(num_stocks=5, num_days=750)
    
    # Plot stock prices
    plt.figure(figsize=(12, 6))
    for symbol, data in stock_data.items():
        plt.plot(data['Date'], data['Close'], label=symbol)
    plt.title('Synthetic Stock Prices')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Create environment with a much smaller action space
    env = PortfolioEnvironment(
        data=stock_data,
        lookback_window=10,
        N=3,  # Use minimal divisions for much fewer actions
        commission=0.0025  # 0.25% transaction cost
    )
    
    # Initialize DQN agent
    state = env.reset()
    num_actions = env.num_actions
    num_stocks = env.num_stocks
    lookback_window = env.lookback_window
    
    agent = DQNAgent(
        state_dim=(num_stocks, lookback_window, 4),
        num_actions=num_actions,
        num_stocks=num_stocks,
        lookback_window=lookback_window,
        learning_rate=1e-4,
        gamma=0.99,
        buffer_size=10000,
        batch_size=64,
        target_update=10,
        eps_start=1.0,
        eps_end=0.01,
        eps_decay=0.995,
        double_dqn=True
    )
    
    # Train the agent
    print(f"Training DQN agent with {num_actions} possible actions...")
    rewards_history, portfolio_values = train_dqn(env, agent, num_episodes=100)
    
    # Plot training results
    plt.figure(figsize=(12, 6))
    plt.plot(rewards_history)
    plt.title('DQN Training Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    plt.figure(figsize=(12, 6))
    plt.plot(portfolio_values)
    plt.title('DQN Training Portfolio Values')
    plt.xlabel('Episode')
    plt.ylabel('Portfolio Value')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Initialize baseline strategies
    buy_and_hold = UniformBuyAndHold(num_assets=env.num_assets)
    crp = ConstantRebalanced(num_assets=env.num_assets)
    olmar = OLMAR(num_assets=env.num_stocks, window=5, eps=10)
    
    # Evaluate all strategies
    print("Evaluating strategies...")
    
    # DQN
    dqn_values, dqn_returns, dqn_weights = evaluate_strategy(env, agent, is_dqn=True)
    
    # Reset environment for each strategy
    env.reset()
    bah_values, bah_returns, bah_weights = evaluate_strategy(env, buy_and_hold)
    
    env.reset()
    crp_values, crp_returns, crp_weights = evaluate_strategy(env, crp)
    
    env.reset()
    olmar_values, olmar_returns, olmar_weights = evaluate_strategy(env, olmar)
    
    # Calculate performance metrics
    dqn_metrics = calculate_metrics(dqn_values, dqn_returns)
    bah_metrics = calculate_metrics(bah_values, bah_returns)
    crp_metrics = calculate_metrics(crp_values, crp_returns)
    olmar_metrics = calculate_metrics(olmar_values, olmar_returns)
    
    # Print metrics
    print("\nPerformance Metrics:")
    print("\nDQN Strategy:")
    for k, v in dqn_metrics.items():
        print(f"{k}: {v:.2f}")
    
    print("\nBuy and Hold Strategy:")
    for k, v in bah_metrics.items():
        print(f"{k}: {v:.2f}")
    
    print("\nConstant Rebalanced Portfolio Strategy:")
    for k, v in crp_metrics.items():
        print(f"{k}: {v:.2f}")
    
    print("\nOLMAR Strategy:")
    for k, v in olmar_metrics.items():
        print(f"{k}: {v:.2f}")
    
    # Plot portfolio values
    portfolio_values_dict = {
        'DQN': dqn_values,
        'Buy and Hold': bah_values,
        'CRP': crp_values,
        'OLMAR': olmar_values
    }
    
    plot_results(portfolio_values_dict, title="Portfolio Performance Comparison")
    
    # Plot DQN weights
    asset_names = ['Cash'] + [f"Stock_{i+1}" for i in range(num_stocks)]
    plot_weights_history(dqn_weights, asset_names, title="DQN Portfolio Weights Over Time")
    
    # Create a comparison table
    metrics_df = pd.DataFrame({
        'Metric': list(dqn_metrics.keys()),
        'DQN': [f"{v:.2f}" for v in dqn_metrics.values()],
        'Buy and Hold': [f"{v:.2f}" for v in bah_metrics.values()],
        'CRP': [f"{v:.2f}" for v in crp_metrics.values()],
        'OLMAR': [f"{v:.2f}" for v in olmar_metrics.values()]
    })
    
    print("\nComparison Table:")
    print(metrics_df)
    
    return {
        'agent': agent,
        'env': env,
        'metrics': {
            'DQN': dqn_metrics,
            'Buy and Hold': bah_metrics,
            'CRP': crp_metrics,
            'OLMAR': olmar_metrics
        },
        'portfolio_values': portfolio_values_dict,
        'weights': {
            'DQN': dqn_weights,
            'Buy and Hold': bah_weights,
            'CRP': crp_weights,
            'OLMAR': olmar_weights
        }
    }

if __name__ == "__main__":
    results = main()