import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import random
from collections import deque
from tqdm import tqdm

# Set seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#--------------------------------------------------
# Price Simulation
#--------------------------------------------------

def simulate_price_path(n_days, n_seconds_per_day, initial_price=100, 
                       mu=0.0001, sigma=0.001, mean_reversion=0.01, 
                       mean_level=100, jump_prob=0.005, jump_size_mean=0,
                       jump_size_std=0.01):
    """
    Simulate price paths following a mean-reverting process with jumps
    
    Increased sigma and jump_prob to create more volatile prices
    """
    total_seconds = n_days * n_seconds_per_day
    prices = np.zeros(total_seconds)
    prices[0] = initial_price
    
    # Trends for different periods to make the environment more complex
    trends = np.random.normal(0, 0.0005, n_days)
    daily_trends = np.repeat(trends, n_seconds_per_day)
    
    for t in range(1, total_seconds):
        # Current trend
        trend = daily_trends[t-1]
        
        # Mean reversion component
        drift = mean_reversion * (mean_level - prices[t-1])
        
        # Diffusion component with increased volatility for more price movement
        diffusion = sigma * np.random.normal()
        
        # Jump component (increased probability and size)
        jump = 0
        if np.random.random() < jump_prob:
            jump = np.random.normal(jump_size_mean, jump_size_std) * prices[t-1]
        
        # Price update with trend
        price_change = mu + drift + diffusion + trend
        # Limit extreme changes
        price_change = np.clip(price_change, -0.05, 0.05)
        
        prices[t] = max(1, prices[t-1] * (1 + price_change) + jump)
    
    return prices

#--------------------------------------------------
# Environment Setup
#--------------------------------------------------

class OptimalExecutionEnv:
    """
    Environment for the optimal execution problem
    """
    def __init__(self, prices, n_periods=5, initial_inventory=20, quadratic_penalty=0.1, 
                 features=['time', 'inventory', 'price', 'qv']):
        """
        Increased quadratic_penalty to 0.1 to make market impact more significant
        """
        self.prices = prices
        self.n_periods = n_periods
        self.initial_inventory = initial_inventory
        self.quadratic_penalty = quadratic_penalty
        self.features = features
        
        # Calculate derived values
        self.seconds_per_period = len(prices) // n_periods
        self.seconds_per_day = len(prices)
        
        # Create period boundaries
        self.period_starts = [i * self.seconds_per_period for i in range(n_periods)]
        self.period_ends = [(i+1) * self.seconds_per_period - 1 for i in range(n_periods)]
        
        # Precompute QV if needed
        if 'qv' in features:
            self.qv = self._compute_qv()
        
        # Initialize state
        self.reset()
    
    def _compute_qv(self):
        """Compute quadratic variation at each period"""
        qv = np.zeros(self.n_periods)
        for k in range(self.n_periods):
            start_idx = self.period_starts[k]
            end_idx = self.period_ends[k]
            
            price_changes = np.diff(self.prices[start_idx:end_idx+1])
            price_changes = np.clip(price_changes, -1, 1)  # Clip extreme values
            
            qv[k] = np.sum(price_changes**2)
            
        # Handle potential NaN or Inf values
        qv = np.nan_to_num(qv, nan=0.0, posinf=1.0, neginf=0.0)
        
        # Normalize QV
        qv_mean = np.mean(qv)
        qv_std = max(np.std(qv), 1e-8)
        normalized_qv = (qv - qv_mean) / (2 * qv_std)
        
        return normalized_qv
    
    def reset(self):
        """Reset the environment to initial state"""
        self.current_period = 0
        self.inventory = self.initial_inventory
        self.start_price = self.prices[0]
        self.reward_history = []
        self.executed_shares = []
        
        # Build initial state
        self.state = self._build_state()
        return self.state
    
    def _build_state(self):
        """Build state vector based on current conditions and selected features"""
        state_dict = {}
        
        # Time feature (normalized to [-1, 1])
        if 'time' in self.features:
            normalized_time = 2 * self.current_period / (self.n_periods - 1) - 1
            state_dict['time'] = normalized_time
        
        # Inventory feature (normalized to [-1, 1])
        if 'inventory' in self.features:
            normalized_inventory = 2 * self.inventory / self.initial_inventory - 1
            state_dict['inventory'] = normalized_inventory
        
        # Price feature (normalized around starting price)
        if 'price' in self.features:
            current_price = self.prices[self.period_starts[self.current_period]]
            # Normalize price to have relative price changes in [-1, 1] range
            price_change = (current_price - self.start_price) / (self.start_price * 0.1)
            normalized_price = np.clip(price_change, -1, 1)
            state_dict['price'] = normalized_price
        
        # Quadratic variation feature
        if 'qv' in self.features:
            if self.current_period > 0:
                state_dict['qv'] = self.qv[self.current_period-1]
            else:
                state_dict['qv'] = 0
        
        # Convert dict to vector ordered by feature names
        state_vector = np.array([state_dict[f] for f in self.features])
        return state_vector
    
    def step(self, action):
        """
        Take action in the current period and move to next period
        
        Args:
            action: Number of lots to sell (normalized in [-1,1])
        
        Returns:
            next_state: The new state after executing the action
            reward: The reward received from the action
            done: Whether the episode is complete
            info: Additional information
        """
        # Transform normalized action back to actual shares to sell
        actual_action = self._transform_action(action)
        
        # Ensure action is within valid range
        actual_action = np.clip(actual_action, 0, self.inventory)
        
        # Store executed shares for analysis
        self.executed_shares.append(actual_action)
        
        # Calculate reward
        reward = self._calculate_reward(actual_action)
        self.reward_history.append(reward)
        
        # Update inventory
        self.inventory -= actual_action
        
        # Move to next period
        self.current_period += 1
        
        # Check if done
        done = (self.current_period >= self.n_periods) or (self.inventory <= 0)
        
        # If last period and inventory > 0, apply penalty for not executing all shares
        if self.current_period >= self.n_periods and self.inventory > 0:
            # Liquidate remaining shares with a penalty (increased penalty for end of trading)
            final_reward = self._calculate_reward(self.inventory, final_liquidation=True)
            self.reward_history.append(final_reward)
            self.executed_shares.append(self.inventory)
            self.inventory = 0
        
        # Build next state
        next_state = self._build_state() if not done else None
        
        # Additional info
        info = {
            'action': actual_action,
            'inventory': self.inventory,
            'period': self.current_period,
        }
        
        return next_state, reward, done, info
    
    def _transform_action(self, norm_action):
        """Transform normalized action [-1,1] to actual shares [0,inventory]"""
        actual_action = int((norm_action + 1) / 2 * self.inventory)
        return actual_action
    
    def _calculate_reward(self, shares_sold, final_liquidation=False):
        """
        Calculate reward for selling a given number of shares
        
        Args:
            shares_sold: Number of shares to sell
            final_liquidation: Whether this is a forced final liquidation
        
        Returns:
            Total reward for the period
        """
        if shares_sold == 0:
            return 0
        
        # Get current period boundaries
        if final_liquidation:
            # Use last second price for final liquidation with increased penalty
            current_price = self.prices[-1]
            # Higher penalty for having to liquidate at the end
            return shares_sold * current_price - 2 * self.quadratic_penalty * shares_sold**2
        else:
            start_idx = self.period_starts[self.current_period]
            end_idx = self.period_ends[self.current_period]
            
            # Calculate number of seconds in period
            seconds_in_period = end_idx - start_idx + 1
            
            # Shares to sell each second
            shares_per_second = shares_sold / seconds_in_period
            
            # Calculate total reward for the period
            total_reward = 0
            
            for i in range(start_idx, end_idx + 1):
                # Get current price
                current_price = self.prices[i]
                
                # P&L minus quadratic penalty for market impact
                second_reward = shares_per_second * current_price - self.quadratic_penalty * shares_per_second**2
                
                # Add to total reward
                total_reward += second_reward
            
            return total_reward
    
    def get_twap_reward(self):
        """Calculate reward for Time-Weighted Average Price (TWAP) strategy"""
        # TWAP executes equal shares in each period
        shares_per_period = self.initial_inventory / self.n_periods
        
        # Calculate TWAP reward
        total_reward = 0
        remaining_inventory = self.initial_inventory
        
        for period in range(self.n_periods):
            start_idx = self.period_starts[period]
            end_idx = self.period_ends[period]
            
            # Calculate shares per second
            seconds_in_period = end_idx - start_idx + 1
            shares_per_second = shares_per_period / seconds_in_period
            
            for i in range(start_idx, end_idx + 1):
                # Get current price
                current_price = self.prices[i]
                
                # Calculate reward
                second_reward = shares_per_second * current_price - self.quadratic_penalty * shares_per_second**2
                
                # Add to total reward
                total_reward += second_reward
                
                # Update remaining inventory
                remaining_inventory -= shares_per_second
        
        return total_reward

#--------------------------------------------------
# PyTorch Neural Network Model
#--------------------------------------------------

class QNetwork(nn.Module):
    """Neural network model for Q-function approximation"""
    def __init__(self, state_size, hidden_size=64):
        super(QNetwork, self).__init__()
        
        # Increased network capacity with larger hidden layers
        self.network = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        
        # Initialize weights to small random values
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
        
    def forward(self, state_action):
        """Forward pass through the network"""
        return self.network(state_action)

#--------------------------------------------------
# Double Deep Q-Learning Implementation
#--------------------------------------------------

class DoubleDQNAgent:
    def __init__(self, state_size, action_size, learning_rate=0.0005, gamma=0.99, 
                 epsilon=1.0, epsilon_decay=0.99, epsilon_min=0.01,
                 memory_size=10000, batch_size=64, update_target_freq=10):
        """
        Reduced learning rate, increased batch size, faster target network updates,
        and slower epsilon decay for more stable learning.
        """
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.update_target_freq = update_target_freq
        
        # Initialize replay memory
        self.memory = deque(maxlen=memory_size)
        
        # Initialize networks with larger hidden layers
        self.main_model = QNetwork(state_size, hidden_size=64).to(device)
        self.target_model = QNetwork(state_size, hidden_size=64).to(device)
        
        # Initialize target network weights to match main network
        self.update_target_model()
        
        # Initialize optimizer with lower learning rate
        self.optimizer = optim.Adam(self.main_model.parameters(), lr=learning_rate)
        
        # Counter for target model updates
        self.target_update_counter = 0
        
        # For tracking learning progress
        self.loss_history = []
    
    def update_target_model(self):
        """Copy weights from main model to target model"""
        self.target_model.load_state_dict(self.main_model.state_dict())
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay memory"""
        self.memory.append((state, action, reward, next_state, done))
    
    def choose_action(self, state, inventory, period, n_periods):
        """Choose action using epsilon-greedy policy"""
        if np.random.random() < self.epsilon:
            # Exploration: choose random action
            # Make exploration smarter by biasing toward TWAP strategy
            remaining_periods = max(1, n_periods - period)
            if period == n_periods - 1:  # Last period
                # Liquidate all remaining inventory in last period
                return inventory
            
            # Sample from a distribution centered around TWAP execution rate
            twap_execution = inventory / remaining_periods
            # Add some randomness but bias toward TWAP
            action = int(np.clip(
                np.random.normal(twap_execution, twap_execution/2), 
                0, inventory
            ))
            return action
        else:
            # Exploitation: choose best action according to Q-values
            q_values = []
            
            # Discretize action space more finely for large inventories
            num_actions = min(inventory + 1, 20)
            possible_actions = []
            
            if num_actions <= 1:
                possible_actions = [0]
            else:
                # Create evenly spaced possible actions
                for i in range(num_actions):
                    possible_actions.append(int(i * inventory / (num_actions - 1)))
            
            for a in possible_actions:
                # Normalize action to [-1, 1]
                norm_a = 2 * a / inventory - 1 if inventory > 0 else 0
                
                # Get state-action pair Q-value
                state_action = np.append(state, norm_a)
                try:
                    state_action_tensor = torch.FloatTensor(state_action).to(device)
                    q_value = self.main_model(state_action_tensor).item()
                    if np.isnan(q_value) or np.isinf(q_value):
                        q_value = -1000
                except:
                    q_value = -1000
                    
                q_values.append((a, q_value))
            
            # Choose action with highest Q-value
            if any(not np.isnan(q) and not np.isinf(q) and q != -1000 for _, q in q_values):
                best_action = max(q_values, key=lambda x: x[1])[0]
            else:
                # Fallback to TWAP-like strategy
                remaining_periods = max(1, n_periods - period)
                best_action = int(inventory / remaining_periods)
            
            return best_action
    
    def replay(self, batch_size):
        """Train network on batch of experiences"""
        if len(self.memory) < batch_size:
            return
        
        # Sample batch from memory
        minibatch = random.sample(self.memory, batch_size)
        
        # Convert batch to numpy arrays for faster processing
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []
        
        for state, action, reward, next_state, done in minibatch:
            states.append(np.append(state, action))
            rewards.append(reward)
            dones.append(done)
            
            if not done and next_state is not None:
                next_states.append(next_state)
            else:
                # For terminal states, use a placeholder
                next_states.append(np.zeros_like(state))
        
        # Convert to torch tensors in one batch operation
        states_tensor = torch.FloatTensor(np.array(states)).to(device)
        rewards_tensor = torch.FloatTensor(rewards).to(device)
        dones_tensor = torch.FloatTensor(dones).to(device)
        next_states_tensor = torch.FloatTensor(np.array(next_states)).to(device)
        
        # Get current Q values
        current_q_values = self.main_model(states_tensor)
        
        # Initialize target Q values as rewards
        target_q_values = rewards_tensor.clone().detach()
        
        # For non-terminal states, update target Q values
        non_terminal_indices = torch.where(dones_tensor == 0)[0]
        
        if len(non_terminal_indices) > 0:
            # For each non-terminal state, find best action using main network
            for idx in non_terminal_indices:
                next_state = next_states_tensor[idx].cpu().numpy()
                inventory = int((next_state[1] + 1) / 2 * self.action_size)  # Denormalize inventory
                
                # Skip if inventory is 0
                if inventory <= 0:
                    continue
                
                # Get Q-values for each possible action
                next_q_values = []
                for a in range(min(inventory + 1, 20)):
                    norm_a = 2 * a / inventory - 1
                    next_state_action = np.append(next_state, norm_a)
                    next_state_action_tensor = torch.FloatTensor(next_state_action).to(device)
                    
                    try:
                        q_value = self.main_model(next_state_action_tensor).item()
                        if np.isnan(q_value) or np.isinf(q_value):
                            q_value = -1000
                    except:
                        q_value = -1000
                        
                    next_q_values.append((a, q_value))
                
                # Find best action from main model
                if any(q[1] != -1000 for q in next_q_values):
                    best_action = max(next_q_values, key=lambda x: x[1])[0]
                    norm_best_action = 2 * best_action / inventory - 1
                    
                    # Get Q-value from target network
                    next_state_best_action = np.append(next_state, norm_best_action)
                    next_state_best_action_tensor = torch.FloatTensor(next_state_best_action).to(device)
                    
                    try:
                        next_q = self.target_model(next_state_best_action_tensor).item()
                        if not np.isnan(next_q) and not np.isinf(next_q):
                            target_q_values[idx] += self.gamma * next_q
                    except:
                        pass
        
        # Reshape target for loss calculation
        target_q_values = target_q_values.unsqueeze(1)
        
        # Calculate loss
        loss = nn.MSELoss()(current_q_values, target_q_values)
        
        # Store loss for tracking
        self.loss_history.append(loss.item())
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        
        # Clip gradients to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.main_model.parameters(), 1.0)
        
        self.optimizer.step()
        
        # Update target model if needed
        self.target_update_counter += 1
        if self.target_update_counter >= self.update_target_freq:
            self.update_target_model()
            self.target_update_counter = 0
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def load(self, name):
        """Load model weights"""
        self.main_model.load_state_dict(torch.load(name))
        self.update_target_model()
    
    def save(self, name):
        """Save model weights"""
        torch.save(self.main_model.state_dict(), name)

#--------------------------------------------------
# Training Function
#--------------------------------------------------

def train_dqn(agent, env, n_episodes=300, batch_size=64):
    """Train agent on environment"""
    rewards = []
    twap_rewards = []
    execution_patterns = []
    
    for episode in tqdm(range(n_episodes), desc="Training Episodes"):
        # Reset environment
        state = env.reset()
        total_reward = 0
        done = False
        
        # Reset inventory and period
        inventory = env.initial_inventory
        period = 0
        
        # For tracking execution pattern
        execution_pattern = []
        
        while not done and period < env.n_periods:
            # Choose action
            action = agent.choose_action(state, inventory, period, env.n_periods)
            
            # Normalize action to [-1, 1]
            norm_action = 2 * action / inventory - 1 if inventory > 0 else 0
            
            # Execute action
            next_state, reward, done, info = env.step(norm_action)
            
            # Record execution
            execution_pattern.append(action)
            
            # Update inventory and period
            inventory = info['inventory']
            period = info['period']
            
            # Store experience in replay memory
            agent.remember(state, norm_action, reward, next_state, done)
            
            # Update state and total reward
            state = next_state
            total_reward += reward
            
            # Train neural network
            agent.replay(batch_size)
        
        # Calculate TWAP reward for comparison
        twap_reward = env.get_twap_reward()
        
        # Store rewards and execution pattern
        rewards.append(total_reward)
        twap_rewards.append(twap_reward)
        execution_patterns.append(execution_pattern)
        
        # Print progress every 20 episodes
        if episode % 20 == 0:
            # Calculate statistics safely
            recent_rewards = [r for r in rewards[-20:] if not np.isnan(r)]
            recent_twap = [r for r in twap_rewards[-20:] if not np.isnan(r)]
            
            if recent_rewards and recent_twap:
                avg_reward = np.mean(recent_rewards)
                avg_twap = np.mean(recent_twap)
                
                if avg_twap != 0:
                    rel_performance = (avg_reward - avg_twap) / abs(avg_twap) * 10000  # In basis points
                else:
                    rel_performance = 0
                    
                # Also show recent loss
                avg_loss = np.mean(agent.loss_history[-100:]) if agent.loss_history else 0
                    
                print(f"Episode: {episode}, Avg Reward: {avg_reward:.2f}, Avg TWAP: {avg_twap:.2f}, " +
                      f"Relative Performance: {rel_performance:.2f} bps, Epsilon: {agent.epsilon:.4f}, " +
                      f"Avg Loss: {avg_loss:.6f}")
            else:
                print(f"Episode: {episode}, Insufficient data for statistics, Epsilon: {agent.epsilon:.4f}")
    
    return rewards, twap_rewards, execution_patterns

#--------------------------------------------------
# Evaluation Function
#--------------------------------------------------

def evaluate_strategy(agent, env, n_episodes=100):
    """Evaluate trained agent on environment"""
    rewards = []
    twap_rewards = []
    relative_performances = []
    execution_patterns = []
    
    for _ in tqdm(range(n_episodes), desc="Evaluation Episodes"):
        # Reset environment
        state = env.reset()
        total_reward = 0
        done = False
        
        # Reset inventory and period
        inventory = env.initial_inventory
        period = 0
        
        # For tracking execution pattern
        execution_pattern = []
        
        # Execute strategy
        while not done and period < env.n_periods:
            # Choose best action (no exploration)
            action = agent.choose_action(state, inventory, period, env.n_periods)
            execution_pattern.append(action)
            
            # Set epsilon to 0 to ensure no random actions
            temp_epsilon = agent.epsilon
            agent.epsilon = 0
            
            # Normalize action to [-1, 1]
            norm_action = 2 * action / inventory - 1 if inventory > 0 else 0
            
            # Execute action
            next_state, reward, done, info = env.step(norm_action)
            
            # Update inventory and period
            inventory = info['inventory']
            period = info['period']
            
            # Update state and total reward
            state = next_state
            total_reward += reward
            
            # Restore epsilon
            agent.epsilon = temp_epsilon
        
        # Calculate TWAP reward for comparison
        twap_reward = env.get_twap_reward()
        
        # Calculate relative performance in basis points
        if twap_reward != 0:
            rel_performance = (total_reward - twap_reward) / abs(twap_reward) * 10000
        else:
            rel_performance = 0
            
        # Store results
        rewards.append(total_reward)
        twap_rewards.append(twap_reward)
        relative_performances.append(rel_performance)
        execution_patterns.append(execution_pattern)
    
    # Calculate performance metrics
    valid_performances = [p for p in relative_performances if not np.isnan(p) and not np.isinf(p)]
    
    if valid_performances:
        mean_rel_perf = np.mean(valid_performances)
        median_rel_perf = np.median(valid_performances)
        std_rel_perf = np.std(valid_performances)
        positive_prob = np.mean(np.array(valid_performances) > 0)
        
        # Calculate gain-loss ratio
        gains = [perf for perf in valid_performances if perf > 0]
        losses = [abs(perf) for perf in valid_performances if perf < 0]
        
        if losses and np.mean(losses) > 0:
            gain_loss_ratio = np.mean(gains) / np.mean(losses) if gains else 0
        else:
            gain_loss_ratio = float('inf') if gains else 0
    else:
        mean_rel_perf = median_rel_perf = std_rel_perf = positive_prob = gain_loss_ratio = 0
    
    # Print summary statistics
    print("\nEvaluation Results:")
    print(f"Mean Relative Performance: {mean_rel_perf:.2f} bps")
    print(f"Median Relative Performance: {median_rel_perf:.2f} bps")
    print(f"Std Dev of Relative Performance: {std_rel_perf:.2f} bps")
    print(f"Probability of Outperforming TWAP: {positive_prob:.2%}")
    print(f"Gain-Loss Ratio: {gain_loss_ratio:.2f}")
    
    return rewards, twap_rewards, relative_performances, execution_patterns

#--------------------------------------------------
# Main Execution
#--------------------------------------------------

# Simulate price paths for training (with more volatility)
n_days_train = 200
seconds_per_day = 3600  # 1 hour
prices_train = simulate_price_path(n_days_train, seconds_per_day)

# Create environment with different feature sets
features_TIP = ['time', 'inventory', 'price']
features_TIPQV = ['time', 'inventory', 'price', 'qv']

env_TIP = OptimalExecutionEnv(prices_train, features=features_TIP)
env_TIPQV = OptimalExecutionEnv(prices_train, features=features_TIPQV)

# Create agents for each feature set
state_size_TIP = len(features_TIP) + 1  # +1 for action in state
state_size_TIPQV = len(features_TIPQV) + 1
action_size = 21  # 21 possible actions (0 to 20 lots)

agent_TIP = DoubleDQNAgent(state_size=state_size_TIP, action_size=action_size)
agent_TIPQV = DoubleDQNAgent(state_size=state_size_TIPQV, action_size=action_size)

# Train agents
print("Training agent with TIP features...")
rewards_TIP, twap_rewards_TIP, exec_patterns_TIP = train_dqn(agent_TIP, env_TIP, n_episodes=300)

print("\nTraining agent with TIPQV features...")
rewards_TIPQV, twap_rewards_TIPQV, exec_patterns_TIPQV = train_dqn(agent_TIPQV, env_TIPQV, n_episodes=300)

# Simulate new price paths for testing (with more volatility)
n_days_test = 50
prices_test = simulate_price_path(n_days_test, seconds_per_day)

# Create test environments
test_env_TIP = OptimalExecutionEnv(prices_test, features=features_TIP)
test_env_TIPQV = OptimalExecutionEnv(prices_test, features=features_TIPQV)

# Evaluate agents
print("\nEvaluating TIP strategy...")
test_rewards_TIP, test_twap_rewards_TIP, rel_perf_TIP, test_exec_TIP = evaluate_strategy(agent_TIP, test_env_TIP)

print("\nEvaluating TIPQV strategy...")
test_rewards_TIPQV, test_twap_rewards_TIPQV, rel_perf_TIPQV, test_exec_TIPQV = evaluate_strategy(agent_TIPQV, test_env_TIPQV)

#--------------------------------------------------
# Visualization
#--------------------------------------------------

# Function to safely clean data for plotting
def clean_data_for_plotting(data):
    """Replace NaN and Inf values with reasonable numbers for plotting"""
    if isinstance(data, list):
        return [x if not np.isnan(x) and not np.isinf(x) else 0 for x in data]
    else:
        return data if not np.isnan(data) and not np.isinf(data) else 0

# Plot training rewards
plt.figure(figsize=(15, 10))

# Plot 1: Training Rewards
plt.subplot(2, 2, 1)
clean_rewards_TIP = clean_data_for_plotting(rewards_TIP)
clean_twap_rewards_TIP = clean_data_for_plotting(twap_rewards_TIP)
plt.plot(clean_rewards_TIP, label='TIP Strategy')
plt.plot(clean_twap_rewards_TIP, label='TWAP Strategy')
plt.title('Training Rewards - TIP Features')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.legend()
plt.grid(True)

# Plot 2: Relative Performance During Training
plt.subplot(2, 2, 2)
rel_perf_training = [(r-t)/abs(t)*10000 if t != 0 else 0 for r, t in zip(clean_rewards_TIP, clean_twap_rewards_TIP)]
rel_perf_training = [p for p in rel_perf_training if not np.isnan(p) and not np.isinf(p) and abs(p) < 1000]
plt.plot(rel_perf_training)
plt.title('Relative Performance (bps) - TIP Training')
plt.xlabel('Episode')
plt.ylabel('Basis Points')
plt.axhline(y=0, color='r', linestyle='--')
plt.grid(True)

# Plot 3: Training Loss
plt.subplot(2, 2, 3)
plt.plot(agent_TIP.loss_history)
plt.title('Training Loss - TIP')
plt.xlabel('Update Step')
plt.ylabel('Loss')
plt.yscale('log')
plt.grid(True)

# Plot 4: Average Execution Pattern
plt.subplot(2, 2, 4)

# Get execution patterns from last 50 episodes of training
recent_patterns = exec_patterns_TIP[-50:]
# Pad shorter patterns with zeros to match longest pattern
max_length = max(len(p) for p in recent_patterns)
padded_patterns = [p + [0] * (max_length - len(p)) for p in recent_patterns]
# Calculate average execution at each step
avg_pattern = np.mean(padded_patterns, axis=0)

# Get TWAP pattern
twap_pattern = [env_TIP.initial_inventory / env_TIP.n_periods] * env_TIP.n_periods

plt.bar(range(len(avg_pattern)), avg_pattern, alpha=0.7, label='DQN Strategy')
plt.plot(range(len(twap_pattern)), twap_pattern, 'r--', linewidth=2, label='TWAP Strategy')
plt.title('Average Execution Pattern - TIP')
plt.xlabel('Period')
plt.ylabel('Shares Executed')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Plot relative performance histograms and TIPQV results
plt.figure(figsize=(15, 10))

# Filter out extreme outliers for better visualization
def filter_outliers(data, percentile=95):
    data = np.array([x for x in data if not np.isnan(x) and not np.isinf(x)])
    if len(data) == 0:
        return []
    threshold = np.percentile(np.abs(data), percentile)
    return data[np.abs(data) < threshold]

# Plot 1: Relative Performance Histogram - TIP
plt.subplot(2, 2, 1)
filtered_perf_TIP = filter_outliers(rel_perf_TIP)
if len(filtered_perf_TIP) > 0:
    plt.hist(filtered_perf_TIP, bins=20, alpha=0.7)
    median_val = np.median(filtered_perf_TIP)
    mean_val = np.mean(filtered_perf_TIP)
    plt.axvline(median_val, color='r', linestyle='dashed', linewidth=2, label=f'Median: {median_val:.2f}')
    plt.axvline(mean_val, color='g', linestyle='dashed', linewidth=2, label=f'Mean: {mean_val:.2f}')
plt.title('Relative Performance (bps) - TIP Features')
plt.xlabel('Basis Points')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True)

# Plot 2: Relative Performance Histogram - TIPQV
plt.subplot(2, 2, 2)
filtered_perf_TIPQV = filter_outliers(rel_perf_TIPQV)
if len(filtered_perf_TIPQV) > 0:
    plt.hist(filtered_perf_TIPQV, bins=20, alpha=0.7)
    median_val = np.median(filtered_perf_TIPQV)
    mean_val = np.mean(filtered_perf_TIPQV)
    plt.axvline(median_val, color='r', linestyle='dashed', linewidth=2, label=f'Median: {median_val:.2f}')
    plt.axvline(mean_val, color='g', linestyle='dashed', linewidth=2, label=f'Mean: {mean_val:.2f}')
plt.title('Relative Performance (bps) - TIPQV Features')
plt.xlabel('Basis Points')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True)

# Plot 3: Example Price Path
plt.subplot(2, 2, 3)
plt.plot(prices_test[:500])  # Plot first 500 seconds
plt.title('Example Price Path (Test Set)')
plt.xlabel('Seconds')
plt.ylabel('Price')
plt.grid(True)

# Plot 4: Example Execution Strategies
plt.subplot(2, 2, 4)

# Get an example execution pattern
if test_exec_TIP and test_exec_TIPQV:
    # Find a good example with complete execution
    good_examples_TIP = [p for p in test_exec_TIP if len(p) == env_TIP.n_periods]
    good_examples_TIPQV = [p for p in test_exec_TIPQV if len(p) == env_TIPQV.n_periods]
    
    if good_examples_TIP and good_examples_TIPQV:
        example_TIP = good_examples_TIP[0]
        example_TIPQV = good_examples_TIPQV[0]
        
        # Calculate cumulative execution
        cum_TIP = np.cumsum(example_TIP)
        cum_TIPQV = np.cumsum(example_TIPQV)
        cum_TWAP = np.cumsum([env_TIP.initial_inventory / env_TIP.n_periods] * env_TIP.n_periods)
        
        plt.plot(range(len(cum_TIP)), cum_TIP, 'b-', marker='o', label='TIP Strategy')
        plt.plot(range(len(cum_TIPQV)), cum_TIPQV, 'g-', marker='s', label='TIPQV Strategy')
        plt.plot(range(len(cum_TWAP)), cum_TWAP, 'r--', linewidth=2, label='TWAP Strategy')
        plt.title('Example Execution Strategies')
        plt.xlabel('Period')
        plt.ylabel('Cumulative Shares Executed')
        plt.legend()
        plt.grid(True)

plt.tight_layout()
plt.show()

# Visualize optimal strategies for different price levels
def visualize_optimal_strategies(agent, env, feature_set):
    """
    Visualize optimal execution strategies for different price trends
    """
    # Create subplots for different price scenarios
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    price_scenarios = ['Falling', 'Flat', 'Rising']
    
    # Define initial inventory and periods
    initial_inventory = env.initial_inventory
    n_periods = env.n_periods
    
    # Loop through different price scenarios
    for i, scenario in enumerate(price_scenarios):
        # Define a price trend for this scenario
        if scenario == 'Falling':
            price_trend = -0.5  # Normalized price trend (falling)
        elif scenario == 'Rising':
            price_trend = 0.5   # Normalized price trend (rising)
        else:
            price_trend = 0.0   # Normalized price trend (flat)
        
        # Create matrices to store execution at each time/inventory state
        optimal_actions = np.zeros((initial_inventory + 1, n_periods))
        
        # For each time period and inventory level
        for inventory in range(1, initial_inventory + 1):
            for period in range(n_periods):
                # Normalize time and inventory
                normalized_time = 2 * period / (n_periods - 1) - 1
                normalized_inventory = 2 * inventory / initial_inventory - 1
                
                # Create state vector
                state = []
                for feature in feature_set:
                    if feature == 'time':
                        state.append(normalized_time)
                    elif feature == 'inventory':
                        state.append(normalized_inventory)
                    elif feature == 'price':
                        state.append(price_trend)
                    elif feature == 'qv':
                        state.append(0)  # Default QV
                
                # Get the best action
                action = agent.choose_action(np.array(state), inventory, period, n_periods)
                
                # Store the action
                optimal_actions[inventory, period] = action
        
        # Create heatmap
        im = axes[i].imshow(optimal_actions, cmap='hot', aspect='auto', origin='lower', 
                           interpolation='nearest')
                           
        # Add labels
        axes[i].set_title(f'{scenario} Price Trend')
        axes[i].set_xlabel('Time Period')
        axes[i].set_ylabel('Inventory')
        axes[i].set_xticks(range(n_periods))
        axes[i].set_yticks(range(0, initial_inventory + 1, 5))
        
    # Add colorbar
    cbar = fig.colorbar(im, ax=axes.ravel().tolist())
    cbar.set_label('Shares to Execute')
    
    plt.suptitle(f'Optimal Execution Strategies with Different Price Trends - {", ".join(feature_set)} Features', 
                fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

# Visualize optimal strategies
visualize_optimal_strategies(agent_TIP, env_TIP, features_TIP)
visualize_optimal_strategies(agent_TIPQV, env_TIPQV, features_TIPQV)