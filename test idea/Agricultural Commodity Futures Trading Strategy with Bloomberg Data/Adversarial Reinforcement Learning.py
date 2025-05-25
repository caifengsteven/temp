import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal, Beta
import random
from tqdm import tqdm
import time

# Set seeds for reproducibility
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
random.seed(seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

#############################################
# Market Environment based on Avellaneda-Stoikov
#############################################

class MarketEnvironment:
    def __init__(self, 
                 drift=0.0, 
                 volatility=2.0, 
                 liquidity_factor_a=140.0, 
                 liquidity_factor_k=1.5,
                 initial_price=100.0,
                 time_step=0.005,
                 terminal_time=1.0,
                 max_inventory=50,
                 adversarial_mode='FIXED'):
        """
        Initialize the market environment based on Avellaneda-Stoikov model.
        """
        self.volatility = volatility
        self.initial_price = initial_price
        self.time_step = time_step
        self.terminal_time = terminal_time
        self.max_inventory = max_inventory
        self.adversarial_mode = adversarial_mode
        
        # Parameters that might be controlled by the adversary
        self.drift = drift
        self.liquidity_factor_a = liquidity_factor_a
        self.liquidity_factor_k = liquidity_factor_k
        
        # Parameter ranges for RANDOM and STRATEGIC modes
        self.drift_range = [-5.0, 5.0]
        self.liquidity_a_range = [105.0, 175.0]
        self.liquidity_k_range = [1.125, 1.875]
        
        # Calculate total time steps
        self.total_steps = int(terminal_time / time_step)
        
        # Initialize state
        self.reset()
    
    def reset(self, initial_inventory=None):
        """Reset the environment to initial state"""
        # Initialize time
        self.current_step = 0
        self.current_time = 0.0
        
        # Initialize price process
        self.price = self.initial_price
        
        # Initialize inventory
        if initial_inventory is None:
            self.inventory = np.random.randint(-self.max_inventory, self.max_inventory + 1)
        else:
            self.inventory = initial_inventory
            
        # Initialize cash
        self.cash = 0.0
        
        # Initialize mark-to-market value
        self.mtm_value = self.cash + self.inventory * self.price
        
        # Initialize parameters if using RANDOM mode
        if self.adversarial_mode == 'RANDOM':
            self.drift = np.random.uniform(*self.drift_range)
            self.liquidity_factor_a = np.random.uniform(*self.liquidity_a_range) 
            self.liquidity_factor_k = np.random.uniform(*self.liquidity_k_range)
        
        # Return initial state
        return self._get_state()
    
    def step(self, action, adversary_action=None):
        """
        Take a step in the environment.
        """
        # Update market parameters if using strategic adversary
        if self.adversarial_mode == 'STRATEGIC' and adversary_action is not None:
            self.drift = adversary_action.get('drift', self.drift)
            self.liquidity_factor_a = adversary_action.get('liquidity_a', self.liquidity_factor_a)
            self.liquidity_factor_k = adversary_action.get('liquidity_k', self.liquidity_factor_k)
        
        # Extract bid and ask offsets from action
        bid_offset = max(0.01, action['bid_offset'])  # Ensure positive offset
        ask_offset = max(0.01, action['ask_offset'])  # Ensure positive offset
        
        # Calculate bid and ask prices
        bid_price = self.price - bid_offset
        ask_price = self.price + ask_offset
        
        # Store previous MTM value
        prev_mtm = self.mtm_value
        
        # Execute market orders
        self._execute_orders(bid_price, ask_price)
        
        # Update price process
        self._update_price()
        
        # Update time
        self.current_step += 1
        self.current_time = self.current_step * self.time_step
        
        # Calculate mark-to-market value
        self.mtm_value = self.cash + self.inventory * self.price
        
        # Calculate reward (change in MTM value)
        reward = self.mtm_value - prev_mtm
        
        # Add inventory penalty if using risk-averse reward
        # (Using running penalty ζ=0.01 as per the paper)
        inventory_penalty = 0.01 * (self.inventory ** 2)
        reward -= inventory_penalty
        
        # Check if episode is done
        done = self.current_time >= self.terminal_time
        
        # Add terminal inventory penalty if done
        if done:
            # Using terminal penalty η=1.0 as per the paper
            terminal_penalty = 1.0 * (self.inventory ** 2)
            reward -= terminal_penalty
        
        # Get next state
        next_state = self._get_state()
        
        # Additional info
        info = {
            'price': self.price,
            'cash': self.cash,
            'inventory': self.inventory,
            'mtm_value': self.mtm_value,
            'bid': bid_price,
            'ask': ask_price,
            'spread': ask_offset + bid_offset
        }
        
        return next_state, reward, done, info
    
    def _execute_orders(self, bid_price, ask_price):
        """Execute incoming market orders based on Poisson processes"""
        # Calculate arrival intensities
        bid_intensity = self.liquidity_factor_a * np.exp(-self.liquidity_factor_k * (self.price - bid_price))
        ask_intensity = self.liquidity_factor_a * np.exp(-self.liquidity_factor_k * (ask_price - self.price))
        
        # Limit trading if inventory bounds are reached
        if self.inventory >= self.max_inventory:
            bid_intensity = 0  # Can't buy more
        if self.inventory <= -self.max_inventory:
            ask_intensity = 0  # Can't sell more
        
        # Generate Poisson random variables for arrivals
        bid_executed = np.random.poisson(bid_intensity * self.time_step)
        ask_executed = np.random.poisson(ask_intensity * self.time_step)
        
        # For simplicity, limit to at most one execution per step
        bid_executed = min(bid_executed, 1)
        ask_executed = min(ask_executed, 1)
        
        # Update inventory and cash based on executions
        if bid_executed > 0:  # MM buys
            self.inventory += 1
            self.cash -= bid_price
        
        if ask_executed > 0:  # MM sells
            self.inventory -= 1
            self.cash += ask_price
    
    def _update_price(self):
        """Update the price process using GBM"""
        # Generate price innovation
        price_change = self.drift * self.time_step + self.volatility * np.sqrt(self.time_step) * np.random.normal(0, 1)
        
        # Update price
        self.price *= (1 + price_change)
    
    def _get_state(self):
        """Get the current state for the agent"""
        return {
            'time': self.current_time,
            'inventory': self.inventory,
            'price': self.price
        }

#############################################
# Neural Network Models - Simpler Implementation
#############################################

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(PolicyNetwork, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )
        
        # Mean and log_std are separate outputs
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim) - 0.5)
        
        # Initialize weights
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=1.0)
                layer.bias.data.zero_()
        
        nn.init.orthogonal_(self.mean.weight, gain=0.01)
        self.mean.bias.data.zero_()
    
    def forward(self, x):
        x = self.network(x)
        mean = self.mean(x)
        
        # Constrain log_std for stability
        log_std = self.log_std.expand_as(mean)
        log_std = torch.clamp(log_std, min=-20, max=2)
        
        return mean, torch.exp(log_std)

class ValueNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim=64):
        super(ValueNetwork, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Initialize weights
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=1.0)
                if layer.bias is not None:
                    layer.bias.data.zero_()
    
    def forward(self, x):
        return self.network(x)

class AdversaryNetwork(nn.Module):
    def __init__(self, state_dim, action_dim=3, hidden_dim=64):
        super(AdversaryNetwork, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )
        
        # Parameters for Beta distribution
        self.alpha = nn.Linear(hidden_dim, action_dim)
        self.beta = nn.Linear(hidden_dim, action_dim)
        
        # Initialize weights
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=1.0)
                layer.bias.data.zero_()
        
        nn.init.orthogonal_(self.alpha.weight, gain=0.01)
        self.alpha.bias.data.fill_(1.0)
        
        nn.init.orthogonal_(self.beta.weight, gain=0.01)
        self.beta.bias.data.fill_(1.0)
    
    def forward(self, x):
        x = self.network(x)
        
        # Ensure alpha, beta > 1 for unimodal distribution
        alpha = F.softplus(self.alpha(x)) + 1.0
        beta = F.softplus(self.beta(x)) + 1.0
        
        return alpha, beta

#############################################
# Simplified Agents
#############################################

class MarketMakerAgent:
    def __init__(self, state_dim, action_dim, hidden_dim=64, lr=3e-4):
        self.policy = PolicyNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.value = ValueNetwork(state_dim, hidden_dim).to(device)
        
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.value_optimizer = optim.Adam(self.value.parameters(), lr=lr)
    
    def get_action(self, state, deterministic=False):
        """Select an action from the policy"""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).to(device)
            mean, std = self.policy(state_tensor)
            
            if deterministic:
                action = mean
            else:
                # Sample from Normal distribution
                dist = Normal(mean, std)
                action = dist.sample()
            
            # Ensure positive offsets
            action = F.softplus(action)
            
            return action.cpu().numpy()
    
    def compute_loss(self, states, actions, returns):
        """Compute policy and value losses"""
        # Value loss
        values = self.value(states).squeeze()
        value_loss = F.mse_loss(values, returns)
        
        # Compute advantages
        advantages = returns - values.detach()
        
        # Policy loss
        mean, std = self.policy(states)
        dist = Normal(mean, std)
        log_probs = dist.log_prob(actions).sum(dim=1)
        entropy = dist.entropy().sum(dim=1).mean()
        
        policy_loss = -(log_probs * advantages).mean()
        
        return policy_loss, value_loss, entropy
    
    def update(self, states, actions, rewards):
        """Update policy and value networks"""
        # Convert to tensors
        states = torch.FloatTensor(states).to(device)
        actions = torch.FloatTensor(actions).to(device)
        
        # Compute discounted returns
        returns = []
        discounted_sum = 0
        gamma = 0.99
        
        for r in reversed(rewards):
            discounted_sum = r + gamma * discounted_sum
            returns.insert(0, discounted_sum)
        
        returns = torch.FloatTensor(returns).to(device)
        
        # Normalize returns for training stability
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # Compute losses
        policy_loss, value_loss, entropy = self.compute_loss(states, actions, returns)
        
        # Add entropy term for exploration
        loss = policy_loss + 0.5 * value_loss - 0.01 * entropy
        
        # Update value network
        self.value_optimizer.zero_grad()
        value_loss.backward(retain_graph=True)  # Retain graph for policy update
        torch.nn.utils.clip_grad_norm_(self.value.parameters(), 0.5)
        self.value_optimizer.step()
        
        # Update policy network
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
        self.policy_optimizer.step()

class AdversaryAgent:
    def __init__(self, state_dim, action_dim=3, hidden_dim=64, lr=3e-4):
        self.network = AdversaryNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        
        # Parameter ranges
        self.drift_range = [-5.0, 5.0]
        self.liquidity_a_range = [105.0, 175.0]
        self.liquidity_k_range = [1.125, 1.875]
    
    def get_action(self, state, deterministic=False):
        """Select an action from the policy"""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).to(device)
            alphas, betas = self.network(state_tensor)
            
            if deterministic:
                # Mode of Beta distribution: (alpha - 1) / (alpha + beta - 2) for alpha, beta > 1
                normalized_actions = (alphas - 1) / (alphas + betas - 2)
            else:
                # Sample from Beta distribution
                dists = [Beta(a, b) for a, b in zip(alphas, betas)]
                normalized_actions = torch.stack([dist.sample() for dist in dists])
            
            # Rescale actions to parameter ranges
            scaled_actions = {
                'drift': normalized_actions[0].item() * (self.drift_range[1] - self.drift_range[0]) + self.drift_range[0],
                'liquidity_a': normalized_actions[1].item() * (self.liquidity_a_range[1] - self.liquidity_a_range[0]) + self.liquidity_a_range[0],
                'liquidity_k': normalized_actions[2].item() * (self.liquidity_k_range[1] - self.liquidity_k_range[0]) + self.liquidity_k_range[0]
            }
            
            return scaled_actions
    
    def update(self, states, actions, rewards):
        """Update adversary network"""
        # Convert to tensors
        states = torch.FloatTensor(states).to(device)
        actions = torch.FloatTensor(actions).to(device)
        
        # Compute discounted returns (negative since adversary wants to minimize MM's reward)
        returns = []
        discounted_sum = 0
        gamma = 0.99
        
        for r in reversed(rewards):
            discounted_sum = r + gamma * discounted_sum
            returns.insert(0, discounted_sum)
        
        returns = torch.FloatTensor(returns).to(device)
        
        # Normalize returns for training stability
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # Get distribution parameters
        alphas, betas = self.network(states)
        
        # Compute log probabilities
        log_probs = 0
        entropy = 0
        
        for i in range(actions.shape[1]):
            dist = Beta(alphas[:, i], betas[:, i])
            log_probs += dist.log_prob(actions[:, i])
            entropy += dist.entropy()
        
        # Simple policy gradient loss
        policy_loss = -(log_probs * returns).mean()
        entropy_loss = -entropy.mean()  # Encourage exploration
        
        loss = policy_loss + 0.01 * entropy_loss
        
        # Gradient step
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
        self.optimizer.step()

#############################################
# Training Functions
#############################################

def convert_state_to_tensor(state):
    """Convert state dictionary to flat tensor for NN input"""
    return np.array([
        state['time'], 
        state['inventory'] / 50.0,  # Normalize inventory
        state['price'] / 100.0      # Normalize price
    ])

def collect_episode(env, mm_agent, adv_agent=None):
    """Collect a single episode of experience"""
    state = env.reset()
    state_tensor = convert_state_to_tensor(state)
    
    mm_states, mm_actions, mm_rewards = [], [], []
    adv_states, adv_actions, adv_rewards = [], [], []
    
    done = False
    
    while not done:
        # Market maker selects action
        mm_action_tensor = mm_agent.get_action(state_tensor)
        mm_action = {
            'bid_offset': float(mm_action_tensor[0]),
            'ask_offset': float(mm_action_tensor[1])
        }
        
        # Store market maker experience
        mm_states.append(state_tensor)
        mm_actions.append(mm_action_tensor)
        
        # Adversary selects action if available
        adv_action = None
        if adv_agent is not None and env.adversarial_mode == 'STRATEGIC':
            adv_action = adv_agent.get_action(state_tensor)
            
            # Store adversary experience
            adv_states.append(state_tensor)
            adv_actions.append([
                (adv_action['drift'] - env.drift_range[0]) / (env.drift_range[1] - env.drift_range[0]),
                (adv_action['liquidity_a'] - env.liquidity_a_range[0]) / (env.liquidity_a_range[1] - env.liquidity_a_range[0]),
                (adv_action['liquidity_k'] - env.liquidity_k_range[0]) / (env.liquidity_k_range[1] - env.liquidity_k_range[0])
            ])
        
        # Take environment step
        next_state, reward, done, info = env.step(mm_action, adv_action)
        next_state_tensor = convert_state_to_tensor(next_state)
        
        # Store rewards
        mm_rewards.append(reward)
        
        if adv_agent is not None and env.adversarial_mode == 'STRATEGIC':
            adv_rewards.append(-reward)  # Adversary gets negative of MM's reward
        
        # Update state
        state = next_state
        state_tensor = next_state_tensor
    
    # Return collected experiences
    episode_data = {
        'mm': {
            'states': np.array(mm_states),
            'actions': np.array(mm_actions),
            'rewards': np.array(mm_rewards),
            'return': sum(mm_rewards)
        }
    }
    
    if adv_agent is not None and env.adversarial_mode == 'STRATEGIC':
        episode_data['adv'] = {
            'states': np.array(adv_states),
            'actions': np.array(adv_actions),
            'rewards': np.array(adv_rewards),
            'return': sum(adv_rewards)
        }
    
    return episode_data, info

def train_agents(num_episodes=1000):
    """Train market maker and adversary agents"""
    # Create environment
    env = MarketEnvironment(
        drift=0.0, 
        volatility=2.0, 
        liquidity_factor_a=140.0, 
        liquidity_factor_k=1.5,
        initial_price=100.0,
        time_step=0.005,
        terminal_time=1.0,
        max_inventory=50,
        adversarial_mode='STRATEGIC'
    )
    
    # Create agents
    state_dim = 3  # time, inventory, price
    mm_action_dim = 2  # bid_offset, ask_offset
    adv_action_dim = 3  # drift, liquidity_a, liquidity_k
    
    mm_agent = MarketMakerAgent(state_dim, mm_action_dim)
    adv_agent = AdversaryAgent(state_dim, adv_action_dim)
    
    # Training metrics
    episode_returns = []
    
    # Training loop
    for episode in tqdm(range(num_episodes), desc="Training"):
        # Collect episode
        episode_data, _ = collect_episode(env, mm_agent, adv_agent)
        
        # Update agents
        mm_agent.update(
            episode_data['mm']['states'],
            episode_data['mm']['actions'],
            episode_data['mm']['rewards']
        )
        
        if 'adv' in episode_data:
            adv_agent.update(
                episode_data['adv']['states'],
                episode_data['adv']['actions'],
                episode_data['adv']['rewards']
            )
        
        # Record metrics
        episode_returns.append(episode_data['mm']['return'])
        
        # Print progress
        if (episode + 1) % 50 == 0:
            avg_return = np.mean(episode_returns[-50:])
            print(f"Episode {episode+1}, Average Return (last 50): {avg_return:.2f}")
    
    return mm_agent, adv_agent, episode_returns

def train_traditional_agent(num_episodes=1000):
    """Train a traditional market maker agent in FIXED environment"""
    # Create environment
    env = MarketEnvironment(
        drift=0.0, 
        volatility=2.0, 
        liquidity_factor_a=140.0, 
        liquidity_factor_k=1.5,
        initial_price=100.0,
        time_step=0.005,
        terminal_time=1.0,
        max_inventory=50,
        adversarial_mode='FIXED'
    )
    
    # Create agent
    state_dim = 3  # time, inventory, price
    mm_action_dim = 2  # bid_offset, ask_offset
    
    mm_agent = MarketMakerAgent(state_dim, mm_action_dim)
    
    # Training metrics
    episode_returns = []
    
    # Training loop
    for episode in tqdm(range(num_episodes), desc="Training Traditional"):
        # Collect episode
        episode_data, _ = collect_episode(env, mm_agent)
        
        # Update agent
        mm_agent.update(
            episode_data['mm']['states'],
            episode_data['mm']['actions'],
            episode_data['mm']['rewards']
        )
        
        # Record metrics
        episode_returns.append(episode_data['mm']['return'])
        
        # Print progress
        if (episode + 1) % 50 == 0:
            avg_return = np.mean(episode_returns[-50:])
            print(f"Episode {episode+1}, Average Return (last 50): {avg_return:.2f}")
    
    return mm_agent, episode_returns

#############################################
# Evaluation Functions
#############################################

def evaluate_strategy(agent, num_episodes=100, adversarial_mode='FIXED'):
    """Evaluate a market making strategy"""
    # Create environment
    env = MarketEnvironment(
        drift=0.0, 
        volatility=2.0, 
        liquidity_factor_a=140.0, 
        liquidity_factor_k=1.5,
        initial_price=100.0,
        time_step=0.005,
        terminal_time=1.0,
        max_inventory=50,
        adversarial_mode=adversarial_mode
    )
    
    # Metrics
    terminal_wealth = []
    terminal_inventory = []
    spreads = []
    returns = []
    
    for episode in tqdm(range(num_episodes), desc=f"Evaluating ({adversarial_mode})"):
        # Collect evaluation episode
        episode_data, info = collect_episode(env, agent)
        
        # Record metrics
        terminal_wealth.append(info['mtm_value'])
        terminal_inventory.append(info['inventory'])
        
        # Calculate average spread
        bid_offsets = episode_data['mm']['actions'][:, 0]
        ask_offsets = episode_data['mm']['actions'][:, 1]
        avg_spread = np.mean(bid_offsets + ask_offsets)
        spreads.append(avg_spread)
        
        # Record episode return
        returns.append(episode_data['mm']['return'])
    
    # Calculate aggregated metrics
    sharpe_ratio = np.mean(returns) / (np.std(returns) + 1e-8)
    
    results = {
        'terminal_wealth': {
            'mean': np.mean(terminal_wealth),
            'std': np.std(terminal_wealth)
        },
        'terminal_inventory': {
            'mean': np.mean(terminal_inventory),
            'std': np.std(terminal_inventory)
        },
        'spread': {
            'mean': np.mean(spreads),
            'std': np.std(spreads)
        },
        'sharpe_ratio': sharpe_ratio
    }
    
    return results

def compare_strategies(arl_agent, trad_agent, num_episodes=50):
    """Compare ARL strategy with traditional strategy across different scenarios"""
    scenarios = ['FIXED', 'RANDOM', 'STRATEGIC']
    
    results = {}
    
    for scenario in scenarios:
        print(f"\nEvaluating in {scenario} scenario:")
        
        arl_results = evaluate_strategy(arl_agent, num_episodes=num_episodes, adversarial_mode=scenario)
        trad_results = evaluate_strategy(trad_agent, num_episodes=num_episodes, adversarial_mode=scenario)
        
        results[scenario] = {
            'ARL': arl_results,
            'Traditional': trad_results
        }
        
        # Print comparison
        print(f"ARL Strategy - Terminal Wealth: {arl_results['terminal_wealth']['mean']:.2f} ± {arl_results['terminal_wealth']['std']:.2f}")
        print(f"Traditional Strategy - Terminal Wealth: {trad_results['terminal_wealth']['mean']:.2f} ± {trad_results['terminal_wealth']['std']:.2f}")
        
        print(f"ARL Strategy - Sharpe Ratio: {arl_results['sharpe_ratio']:.2f}")
        print(f"Traditional Strategy - Sharpe Ratio: {trad_results['sharpe_ratio']:.2f}")
        
        print(f"ARL Strategy - Term. Inventory: {arl_results['terminal_inventory']['mean']:.2f} ± {arl_results['terminal_inventory']['std']:.2f}")
        print(f"Traditional Strategy - Term. Inventory: {trad_results['terminal_inventory']['mean']:.2f} ± {trad_results['terminal_inventory']['std']:.2f}")
        
        print(f"ARL Strategy - Avg. Spread: {arl_results['spread']['mean']:.2f} ± {arl_results['spread']['std']:.2f}")
        print(f"Traditional Strategy - Avg. Spread: {trad_results['spread']['mean']:.2f} ± {trad_results['spread']['std']:.2f}")
    
    return results

#############################################
# Main Execution
#############################################

def main():
    # Reduce the number of episodes for quicker testing
    num_episodes = 200
    
    print("Training ARL Market Maker...")
    mm_agent, adv_agent, arl_returns = train_agents(num_episodes=num_episodes)
    
    print("Training Traditional Market Maker (FIXED environment)...")
    trad_agent, trad_returns = train_traditional_agent(num_episodes=num_episodes)
    
    # Compare strategies
    results = compare_strategies(mm_agent, trad_agent, num_episodes=30)
    
    # Plot learning curves
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    
    # Use moving average for smoother curves
    window_size = 20
    smoothed_arl = np.convolve(arl_returns, np.ones(window_size)/window_size, mode='valid')
    smoothed_trad = np.convolve(trad_returns, np.ones(window_size)/window_size, mode='valid')
    
    plt.plot(smoothed_arl, label='ARL Agent')
    plt.plot(smoothed_trad, label='Traditional Agent')
    plt.title('Learning Curves (Moving Average)')
    plt.xlabel('Episode')
    plt.ylabel('Episode Return')
    plt.legend()
    plt.grid(True)
    
    # Plot wealth comparison
    plt.subplot(1, 2, 2)
    scenarios = ['FIXED', 'RANDOM', 'STRATEGIC']
    x = np.arange(len(scenarios))
    width = 0.35
    
    arl_wealth = [results[s]['ARL']['terminal_wealth']['mean'] for s in scenarios]
    trad_wealth = [results[s]['Traditional']['terminal_wealth']['mean'] for s in scenarios]
    
    plt.bar(x - width/2, arl_wealth, width, label='ARL Strategy')
    plt.bar(x + width/2, trad_wealth, width, label='Traditional Strategy')
    
    plt.xlabel('Scenario')
    plt.ylabel('Terminal Wealth')
    plt.title('Wealth Comparison Across Scenarios')
    plt.xticks(x, scenarios)
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Plot Sharpe ratios
    plt.figure(figsize=(10, 5))
    arl_sharpe = [results[s]['ARL']['sharpe_ratio'] for s in scenarios]
    trad_sharpe = [results[s]['Traditional']['sharpe_ratio'] for s in scenarios]
    
    plt.bar(x - width/2, arl_sharpe, width, label='ARL Strategy')
    plt.bar(x + width/2, trad_sharpe, width, label='Traditional Strategy')
    
    plt.xlabel('Scenario')
    plt.ylabel('Sharpe Ratio')
    plt.title('Sharpe Ratio Comparison Across Scenarios')
    plt.xticks(x, scenarios)
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Plot inventory volatility
    plt.figure(figsize=(10, 5))
    arl_inv_std = [results[s]['ARL']['terminal_inventory']['std'] for s in scenarios]
    trad_inv_std = [results[s]['Traditional']['terminal_inventory']['std'] for s in scenarios]
    
    plt.bar(x - width/2, arl_inv_std, width, label='ARL Strategy')
    plt.bar(x + width/2, trad_inv_std, width, label='Traditional Strategy')
    
    plt.xlabel('Scenario')
    plt.ylabel('Terminal Inventory Std Dev')
    plt.title('Inventory Risk Comparison Across Scenarios')
    plt.xticks(x, scenarios)
    plt.legend()
    plt.grid(True)
    plt.show()
    
    return mm_agent, trad_agent, results

if __name__ == "__main__":
    mm_agent, trad_agent, results = main()