import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal, Bernoulli
from tqdm import tqdm
import seaborn as sns
import random
from collections import deque, namedtuple
import time
import datetime as dt
import warnings
warnings.filterwarnings('ignore')

# Set seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
random.seed(42)

class EnhancedLimitOrderBookEnvironment:
    """
    Enhanced implementation of a Limit Order Book environment with more realistic features.
    """
    def __init__(self, N=3, Imax=5, T=5, temperatures=None, reward_type='linear', alpha=1.0, beta=0.5, risk_aversion=0.1):
        """
        Initialize the enhanced LOB environment.
        
        Args:
            N (int): Number of trading agents (TA)
            Imax (int): Maximum inventory limit for the expert agent
            T (int): Maximum time steps per episode
            temperatures (list): Temperature parameters for the N trading agents
            reward_type (str): Type of reward function ('linear', 'exponential', or 'enhanced')
            alpha (float): Alpha parameter for exponential reward
            beta (float): Beta parameter for exponential reward
            risk_aversion (float): Risk aversion parameter
        """
        self.N = N  # Number of trading agents
        self.Imax = Imax  # Maximum inventory for expert agent
        self.T = T  # Maximum time steps per episode
        self.risk_aversion = risk_aversion
        
        # Set temperature parameters for trading agents
        if temperatures is None:
            self.temperatures = np.array([0.1, 0.5, 1.0])
        else:
            self.temperatures = np.array(temperatures)
            
        # Ensure we have N temperature parameters
        assert len(self.temperatures) == self.N, "Number of temperature parameters must match N"
        
        # Reward function parameters
        self.reward_type = reward_type
        self.alpha = alpha
        self.beta = beta
        
        # Define state space and action space
        self.state_shape = (3,)  # [bid_volume, ask_volume, EA_inventory]
        self.action_shape = (2,)  # [EA_bid_volume, EA_ask_volume]
        
        # Define the discrete state space for easier IRL
        self.discrete_states = self._create_discrete_state_space()
        self.state_to_idx = {tuple(state): idx for idx, state in enumerate(self.discrete_states)}
        self.num_states = len(self.discrete_states)
        
        # Define the discrete action space
        self.discrete_actions = self._create_discrete_action_space()
        self.action_to_idx = {tuple(action): idx for idx, action in enumerate(self.discrete_actions)}
        self.num_actions = len(self.discrete_actions)
        
        # Initialize transition probability matrix
        self.transition_matrix = self._create_transition_matrix()
        
        # Reset the environment
        self.reset()
    
    def _create_discrete_state_space(self):
        """
        Create a list of all possible discrete states.
        Each state is a tuple of (bid_volume, ask_volume, EA_inventory).
        """
        states = []
        for bid_volume in range(self.N + 1):
            for ask_volume in range(self.N + 1):
                for inventory in range(-self.Imax, self.Imax + 1):
                    # Ensure bid_volume + ask_volume <= N (due to our environment constraints)
                    if bid_volume + ask_volume <= self.N:
                        states.append([bid_volume, ask_volume, inventory])
        return states
    
    def _create_discrete_action_space(self):
        """
        Create a list of all possible discrete actions.
        Each action is a tuple of (EA_bid_volume, EA_ask_volume).
        """
        actions = []
        for bid_volume in range(self.N + 1):
            for ask_volume in range(self.N + 1):
                # Ensure bid_volume + ask_volume <= N (expert can't trade more than available)
                if bid_volume + ask_volume <= self.N:
                    actions.append([bid_volume, ask_volume])
        return actions
    
    def _poisson_binomial_pmf(self, k, p):
        """
        Calculate PMF of Poisson Binomial distribution.
        
        Args:
            k (int): Number of successes
            p (array): Probabilities of success for each Bernoulli trial
            
        Returns:
            float: Probability of k successes
        """
        # Use the recursive formula for Poisson Binomial PMF for small N
        n = len(p)
        
        # Base cases
        if k < 0 or k > n:
            return 0.0
        if n == 0:
            return 1.0 if k == 0 else 0.0
        
        # Recursive case using dynamic programming
        dp = np.zeros((n + 1, k + 1))
        dp[0, 0] = 1.0
        
        for i in range(1, n + 1):
            dp[i, 0] = dp[i-1, 0] * (1 - p[i-1])
            for j in range(1, k + 1):
                dp[i, j] = dp[i-1, j] * (1 - p[i-1]) + dp[i-1, j-1] * p[i-1]
        
        return dp[n, k]
    
    def _create_transition_matrix(self):
        """
        Create the transition probability matrix for the MDP.
        
        The transition matrix has shape (num_states, num_actions, num_states)
        where T[s, a, s'] = P(s'|s, a)
        """
        transition_matrix = np.zeros((self.num_states, self.num_actions, self.num_states))
        
        for s_idx, state in enumerate(self.discrete_states):
            bid_volume, ask_volume, inventory = state
            
            for a_idx, action in enumerate(self.discrete_actions):
                ea_bid_volume, ea_ask_volume = action
                
                # Calculate Bernoulli probabilities for each trading agent
                probs = np.zeros(self.N)
                for i in range(self.N):
                    # Add small epsilon to avoid division by zero
                    epsilon = 1e-10
                    p_bid = np.exp((bid_volume + epsilon) / self.temperatures[i]) / (
                        np.exp((bid_volume + epsilon) / self.temperatures[i]) + 
                        np.exp((ask_volume + epsilon) / self.temperatures[i])
                    )
                    probs[i] = p_bid
                
                # Calculate probabilities for the next belief state (intermediate LOB state)
                for n_bid in range(self.N + 1):
                    n_ask = self.N - n_bid  # Each TA places exactly one order
                    
                    # Probability of this belief state
                    prob_belief = self._poisson_binomial_pmf(n_bid, probs)
                    
                    # Calculate the next state based on EA's action and belief state
                    next_bid_volume = max(0, n_bid - ea_ask_volume)
                    next_ask_volume = max(0, n_ask - ea_bid_volume)
                    next_inventory = inventory + min(ea_bid_volume, n_ask) - min(ea_ask_volume, n_bid)
                    
                    # Check if this leads to a terminal state
                    if abs(next_inventory) > self.Imax:
                        # Terminal state - absorbing state with high negative index
                        next_s_idx = -1
                    else:
                        next_state = [next_bid_volume, next_ask_volume, next_inventory]
                        next_s_idx = self.state_to_idx.get(tuple(next_state))
                        if next_s_idx is None:
                            continue
                    
                    # Update the transition probability
                    if next_s_idx >= 0:
                        transition_matrix[s_idx, a_idx, next_s_idx] += prob_belief
        
        # Normalize each row to ensure they sum to 1
        for s in range(self.num_states):
            for a in range(self.num_actions):
                row_sum = np.sum(transition_matrix[s, a])
                if row_sum > 0:
                    transition_matrix[s, a] = transition_matrix[s, a] / row_sum
        
        return transition_matrix
    
    def reset(self):
        """
        Reset the environment to a random initial state.
        
        Returns:
            np.array: Initial state
        """
        # Randomly select a valid initial state
        valid_states = [s for s in self.discrete_states if abs(s[2]) <= self.Imax]
        self.state = random.choice(valid_states)
        self.t = 0
        self.done = False
        return np.array(self.state)
    
    def step(self, action):
        """
        Take a step in the environment given the expert agent's action.
        
        Args:
            action (np.array): EA's action [ea_bid_volume, ea_ask_volume]
            
        Returns:
            next_state (np.array): Next state
            reward (float): Reward for this step
            done (bool): Whether episode is done
            info (dict): Additional information
        """
        if self.done:
            return np.array(self.state), 0.0, True, {}
        
        # Validate action
        assert len(action) == 2, "Action must be a 2D array [ea_bid_volume, ea_ask_volume]"
        ea_bid_volume, ea_ask_volume = action
        assert ea_bid_volume + ea_ask_volume <= self.N, "EA can't trade more than N orders"
        
        # Get current state
        bid_volume, ask_volume, inventory = self.state
        
        # Calculate Bernoulli probabilities for each trading agent
        probs = np.zeros(self.N)
        for i in range(self.N):
            # Add small epsilon to avoid division by zero
            epsilon = 1e-10
            p_bid = np.exp((bid_volume + epsilon) / self.temperatures[i]) / (
                np.exp((bid_volume + epsilon) / self.temperatures[i]) + 
                np.exp((ask_volume + epsilon) / self.temperatures[i])
            )
            probs[i] = p_bid
        
        # Sample the intermediate belief state (number of bids from TAs)
        ta_bids = sum([np.random.rand() < p for p in probs])
        ta_asks = self.N - ta_bids
        
        # Update state based on EA's action and belief state
        next_bid_volume = max(0, ta_bids - ea_ask_volume)
        next_ask_volume = max(0, ta_asks - ea_bid_volume)
        next_inventory = inventory + min(ea_bid_volume, ta_asks) - min(ea_ask_volume, ta_bids)
        
        # Check if this leads to a terminal state
        if abs(next_inventory) > self.Imax or self.t >= self.T - 1:
            self.done = True
        
        # Calculate reward
        if self.reward_type == 'linear':
            reward = self.N - next_bid_volume - next_ask_volume
        elif self.reward_type == 'exponential':
            reward = 1 - np.exp(-self.alpha * (self.N - next_bid_volume - next_ask_volume - self.beta * abs(next_inventory)))
        elif self.reward_type == 'enhanced':
            # Enhanced reward function includes:
            # 1. Spread capture
            # 2. Inventory risk penalty
            # 3. Trading activity incentive
            trading_activity = min(ea_bid_volume, ta_asks) + min(ea_ask_volume, ta_bids)
            inventory_penalty = -self.risk_aversion * abs(next_inventory)**2
            reward = trading_activity + inventory_penalty
        else:
            raise ValueError(f"Unknown reward type: {self.reward_type}")
        
        # Update state and time step
        self.state = [next_bid_volume, next_ask_volume, next_inventory]
        self.t += 1
        
        return np.array(self.state), reward, self.done, {}
    
    def reward_function(self, state):
        """
        Calculate reward for a given state.
        
        Args:
            state (np.array): State [bid_volume, ask_volume, inventory]
            
        Returns:
            float: Reward
        """
        bid_volume, ask_volume, inventory = state
        
        if self.reward_type == 'linear':
            return self.N - bid_volume - ask_volume
        elif self.reward_type == 'exponential':
            return 1 - np.exp(-self.alpha * (self.N - bid_volume - ask_volume - self.beta * abs(inventory)))
        elif self.reward_type == 'enhanced':
            # Simplified version of the enhanced reward for state evaluation
            trading_potential = self.N - bid_volume - ask_volume
            inventory_penalty = -self.risk_aversion * abs(inventory)**2
            return trading_potential + inventory_penalty
        else:
            raise ValueError(f"Unknown reward type: {self.reward_type}")

class PolicyIteration:
    """
    Policy iteration algorithm to find the optimal policy for the LOB environment.
    """
    def __init__(self, env, gamma=0.99, theta=1e-8, max_iterations=1000):
        """
        Initialize the policy iteration algorithm.
        
        Args:
            env (EnhancedLimitOrderBookEnvironment): The LOB environment
            gamma (float): Discount factor
            theta (float): Threshold for convergence
            max_iterations (int): Maximum number of iterations
        """
        self.env = env
        self.gamma = gamma
        self.theta = theta
        self.max_iterations = max_iterations
        
        # Initialize random policy
        self.policy = np.zeros(env.num_states, dtype=int)
        for s in range(env.num_states):
            valid_actions = np.where(np.sum(env.transition_matrix[s], axis=1) > 0)[0]
            if len(valid_actions) > 0:
                self.policy[s] = np.random.choice(valid_actions)
        
        # Initialize value function
        self.V = np.zeros(env.num_states)
    
    def policy_evaluation(self):
        """
        Evaluate the current policy.
        """
        delta = float('inf')
        iterations = 0
        
        while delta > self.theta and iterations < self.max_iterations:
            delta = 0
            for s in range(self.env.num_states):
                old_v = self.V[s]
                
                # Calculate state value
                s_next_probs = self.env.transition_matrix[s, self.policy[s]]
                rewards = np.array([self.env.reward_function(self.env.discrete_states[s_next]) for s_next in range(self.env.num_states)])
                self.V[s] = np.sum(s_next_probs * (rewards + self.gamma * self.V))
                
                delta = max(delta, abs(old_v - self.V[s]))
            
            iterations += 1
    
    def policy_improvement(self):
        """
        Improve the current policy based on the value function.
        
        Returns:
            bool: True if the policy was improved, False otherwise
        """
        policy_stable = True
        
        for s in range(self.env.num_states):
            old_action = self.policy[s]
            
            # Calculate state-action values
            action_values = np.zeros(self.env.num_actions)
            for a in range(self.env.num_actions):
                s_next_probs = self.env.transition_matrix[s, a]
                rewards = np.array([self.env.reward_function(self.env.discrete_states[s_next]) for s_next in range(self.env.num_states)])
                action_values[a] = np.sum(s_next_probs * (rewards + self.gamma * self.V))
            
            # Choose the action with the highest value
            valid_actions = np.where(np.sum(self.env.transition_matrix[s], axis=1) > 0)[0]
            if len(valid_actions) > 0:
                # Only consider valid actions
                valid_action_values = action_values[valid_actions]
                self.policy[s] = valid_actions[np.argmax(valid_action_values)]
            
            if old_action != self.policy[s]:
                policy_stable = False
        
        return policy_stable
    
    def run(self):
        """
        Run the policy iteration algorithm.
        
        Returns:
            tuple: (policy, value_function)
        """
        iterations = 0
        policy_stable = False
        
        while not policy_stable and iterations < self.max_iterations:
            self.policy_evaluation()
            policy_stable = self.policy_improvement()
            iterations += 1
            
            if iterations % 10 == 0:
                print(f"Policy iteration: {iterations} iterations completed")
        
        return self.policy, self.V

class SimplerNeuralNetwork(nn.Module):
    """
    Simple neural network for IRL.
    """
    def __init__(self, input_dim, hidden_dims=[32, 16]):
        """
        Initialize the neural network.
        
        Args:
            input_dim (int): Input dimension
            hidden_dims (list): List of hidden layer dimensions
        """
        super(SimplerNeuralNetwork, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, 1))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Output tensor
        """
        return self.model(x)

class SimpleIRL:
    """
    Simple Inverse Reinforcement Learning for LOB.
    """
    def __init__(self, env, expert_policy, lr=1e-3, batch_size=16):
        """
        Initialize the Simple IRL algorithm.
        
        Args:
            env (EnhancedLimitOrderBookEnvironment): The LOB environment
            expert_policy (np.array): Expert policy
            lr (float): Learning rate
            batch_size (int): Batch size
        """
        self.env = env
        self.expert_policy = expert_policy
        self.lr = lr
        self.batch_size = batch_size
        
        # Input dimension is the state dimension
        self.input_dim = len(self.env.discrete_states[0])
        
        # Create the neural network
        self.nn = SimplerNeuralNetwork(self.input_dim)
        
        # Create the optimizer
        self.optimizer = optim.Adam(self.nn.parameters(), lr=self.lr)
    
    def generate_expert_demonstrations(self, num_demos=1000):
        """
        Generate expert demonstrations.
        
        Args:
            num_demos (int): Number of demonstrations to generate
            
        Returns:
            list: Expert demonstrations
        """
        demos = []
        
        for _ in range(num_demos):
            # Reset the environment
            state = self.env.reset()
            state_idx = self.env.state_to_idx[tuple(state)]
            
            # Run one episode
            done = False
            trajectory = []
            
            while not done:
                # Choose action according to expert policy
                action_idx = self.expert_policy[state_idx]
                action = self.env.discrete_actions[action_idx]
                
                # Take a step in the environment
                next_state, reward, done, _ = self.env.step(action)
                
                # Get next state index
                if done and tuple(next_state) not in self.env.state_to_idx:
                    next_state_idx = -1
                else:
                    next_state_idx = self.env.state_to_idx.get(tuple(next_state), -1)
                
                # Add transition to trajectory
                trajectory.append((state_idx, action_idx, next_state_idx, reward))
                
                # Update state
                state = next_state
                state_idx = next_state_idx
                
                if state_idx == -1:
                    break
            
            if len(trajectory) > 0:
                demos.append(trajectory)
        
        return demos
    
    def compute_reward_estimates(self, demos):
        """
        Compute reward estimates directly from demonstrations.
        
        Args:
            demos (list): Expert demonstrations
            
        Returns:
            tuple: (states, rewards)
        """
        states = []
        rewards = []
        
        # Collect state-reward pairs from demonstrations
        for demo in demos:
            for state_idx, _, _, reward in demo:
                if state_idx >= 0:
                    states.append(self.env.discrete_states[state_idx])
                    rewards.append(reward)
        
        return np.array(states), np.array(rewards)
    
    def train(self, num_demos=1000, epochs=100):
        """
        Train the IRL algorithm.
        
        Args:
            num_demos (int): Number of demonstrations to generate
            epochs (int): Number of training epochs
            
        Returns:
            SimplerNeuralNetwork: Trained neural network
        """
        print("Generating expert demonstrations...")
        demos = self.generate_expert_demonstrations(num_demos)
        print(f"Generated {len(demos)} demonstrations")
        
        print("Computing reward estimates...")
        states, rewards = self.compute_reward_estimates(demos)
        print(f"Computed rewards for {len(states)} state-reward pairs")
        
        # Convert to PyTorch tensors
        states_tensor = torch.FloatTensor(states)
        rewards_tensor = torch.FloatTensor(rewards).unsqueeze(1)
        
        print("Training neural network...")
        for epoch in range(epochs):
            # Shuffle data
            indices = torch.randperm(len(states_tensor))
            states_shuffled = states_tensor[indices]
            rewards_shuffled = rewards_tensor[indices]
            
            # Train in batches
            total_loss = 0
            num_batches = 0
            
            for i in range(0, len(states_tensor), self.batch_size):
                # Get batch
                batch_states = states_shuffled[i:i+self.batch_size]
                batch_rewards = rewards_shuffled[i:i+self.batch_size]
                
                # Forward pass
                pred_rewards = self.nn(batch_states)
                
                # Compute loss
                loss = F.mse_loss(pred_rewards, batch_rewards)
                
                # Backward pass and optimize
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            # Print progress
            if (epoch + 1) % 10 == 0:
                avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
                print(f"Epoch {epoch+1}/{epochs}, Avg Loss: {avg_loss:.6f}")
        
        print("Training complete")
        return self.nn
    
    def predict_rewards(self, states):
        """
        Predict rewards for states.
        
        Args:
            states (np.array): States
            
        Returns:
            np.array: Predicted rewards
        """
        # Convert to PyTorch tensor
        states_tensor = torch.FloatTensor(states)
        
        # Forward pass
        with torch.no_grad():
            pred_rewards = self.nn(states_tensor)
        
        return pred_rewards.numpy()
    
    def extract_policy(self, gamma=0.99):
        """
        Extract the policy from the learned reward function.
        
        Args:
            gamma (float): Discount factor
            
        Returns:
            np.array: Extracted policy
        """
        # Predict rewards for all states
        state_features = np.array(self.env.discrete_states)
        rewards = self.predict_rewards(state_features).flatten()
        
        # Create a new environment with learned rewards
        new_env = EnhancedLimitOrderBookEnvironment(
            N=self.env.N, 
            Imax=self.env.Imax, 
            T=self.env.T, 
            temperatures=self.env.temperatures,
            reward_type=self.env.reward_type
        )
        
        # Define a custom reward function using the learned rewards
        def custom_reward_function(state):
            state_tuple = tuple(state)
            if state_tuple in new_env.state_to_idx:
                idx = new_env.state_to_idx[state_tuple]
                return rewards[idx]
            return 0.0
        
        # Store the original reward function
        original_reward_function = new_env.reward_function
        
        # Set the custom reward function
        new_env.reward_function = custom_reward_function
        
        # Solve the MDP with the learned reward function
        pi = PolicyIteration(new_env, gamma=gamma)
        policy, _ = pi.run()
        
        # Restore the original reward function
        new_env.reward_function = original_reward_function
        
        return policy

def evaluate_policy(env, policy, num_episodes=100):
    """
    Evaluate a policy on the environment.
    
    Args:
        env (EnhancedLimitOrderBookEnvironment): The LOB environment
        policy (np.array): Policy to evaluate
        num_episodes (int): Number of episodes to run
        
    Returns:
        tuple: (mean_reward, std_reward)
    """
    rewards = []
    
    for _ in range(num_episodes):
        # Reset the environment
        state = env.reset()
        state_idx = env.state_to_idx[tuple(state)]
        
        # Run one episode
        done = False
        episode_reward = 0.0
        
        while not done:
            # Choose action according to policy
            action_idx = policy[state_idx]
            action = env.discrete_actions[action_idx]
            
            # Take a step in the environment
            next_state, reward, done, _ = env.step(action)
            
            # Update cumulative reward
            episode_reward += reward
            
            # Update state
            state = next_state
            if tuple(state) in env.state_to_idx:
                state_idx = env.state_to_idx[tuple(state)]
            else:
                break
        
        rewards.append(episode_reward)
    
    return np.mean(rewards), np.std(rewards)

def visualize_rewards(env, rewards, title="Reward Function"):
    """
    Visualize the reward function for a fixed inventory level.
    
    Args:
        env (EnhancedLimitOrderBookEnvironment): The LOB environment
        rewards (np.array): Rewards for each state
        title (str): Plot title
    """
    # Create a grid of bid and ask volumes for inventory = 0
    bid_volumes = np.arange(env.N + 1)
    ask_volumes = np.arange(env.N + 1)
    
    # Initialize the reward grid with zeros
    reward_grid = np.zeros((len(bid_volumes), len(ask_volumes)))
    
    # Create a mask for invalid states
    mask = np.ones_like(reward_grid, dtype=bool)
    
    # Fill in the reward grid for inventory = 0
    for bid in bid_volumes:
        for ask in ask_volumes:
            if bid + ask <= env.N:  # Valid state constraint
                state = [bid, ask, 0]  # Fixed inventory = 0
                if tuple(state) in env.state_to_idx:
                    state_idx = env.state_to_idx[tuple(state)]
                    reward_grid[bid, ask] = rewards[state_idx]
                    mask[bid, ask] = False
    
    # Create the plot
    plt.figure(figsize=(10, 8))
    
    # Plot the rewards
    sns.heatmap(reward_grid, annot=True, cmap="viridis", mask=mask, fmt=".2f", cbar=True)
    
    plt.title(title)
    plt.xlabel("Ask Volume")
    plt.ylabel("Bid Volume")
    plt.tight_layout()
    plt.show()

def visualize_policy(env, policy, title="Policy"):
    """
    Visualize the policy for a fixed inventory level.
    
    Args:
        env (EnhancedLimitOrderBookEnvironment): The LOB environment
        policy (np.array): Policy to visualize
        title (str): Plot title
    """
    # Create a grid of bid and ask volumes for inventory = 0
    bid_volumes = np.arange(env.N + 1)
    ask_volumes = np.arange(env.N + 1)
    
    # Initialize grids for bid and ask actions
    bid_actions = np.zeros((len(bid_volumes), len(ask_volumes)))
    ask_actions = np.zeros((len(bid_volumes), len(ask_volumes)))
    
    # Create a mask for invalid states
    mask = np.ones_like(bid_actions, dtype=bool)
    
    # Fill in the action grids for inventory = 0
    for bid in bid_volumes:
        for ask in ask_volumes:
            if bid + ask <= env.N:  # Valid state constraint
                state = [bid, ask, 0]  # Fixed inventory = 0
                if tuple(state) in env.state_to_idx:
                    state_idx = env.state_to_idx[tuple(state)]
                    action_idx = policy[state_idx]
                    action = env.discrete_actions[action_idx]
                    bid_actions[bid, ask] = action[0]  # EA bid volume
                    ask_actions[bid, ask] = action[1]  # EA ask volume
                    mask[bid, ask] = False
    
    # Create the plots
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Plot bid actions
    sns.heatmap(bid_actions, annot=True, cmap="Blues", mask=mask, fmt=".0f", cbar=True, ax=axes[0])
    axes[0].set_title("Bid Actions (EA_bid_volume)")
    axes[0].set_xlabel("Ask Volume")
    axes[0].set_ylabel("Bid Volume")
    
    # Plot ask actions
    sns.heatmap(ask_actions, annot=True, cmap="Reds", mask=mask, fmt=".0f", cbar=True, ax=axes[1])
    axes[1].set_title("Ask Actions (EA_ask_volume)")
    axes[1].set_xlabel("Ask Volume")
    axes[1].set_ylabel("Bid Volume")
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

def generate_enhanced_simulated_market_data(num_days=252, num_stocks=3, volatility=0.01, mean_reversion=0.1):
    """
    Generate enhanced simulated market data for LOB testing with more realistic price dynamics.
    
    Args:
        num_days (int): Number of trading days
        num_stocks (int): Number of stocks
        volatility (float): Price volatility
        mean_reversion (float): Mean reversion strength
        
    Returns:
        pd.DataFrame: Simulated market data
    """
    # Generate dates
    dates = pd.date_range(start='2020-01-01', periods=num_days, freq='B')
    
    # Initialize data dictionary
    data = {}
    
    # Generate data for each stock
    for stock_idx in range(num_stocks):
        stock_name = f"Stock_{stock_idx+1}"
        
        # Initialize with random starting prices
        price = 100 * (1 + 0.5 * np.random.randn())
        
        # Generate price series with random walk and mean reversion
        prices = [price]
        
        for _ in range(1, num_days):
            # Random price change with mean reversion
            price_change = volatility * np.random.randn() - mean_reversion * (price / 100 - 1)
            price *= (1 + price_change)
            prices.append(price)
        
        # Generate bid and ask prices with realistic spreads
        bid_prices = []
        ask_prices = []
        
        for p in prices:
            # Spread as a function of price and volatility
            spread = max(0.01, p * 0.001 * (1 + np.random.rand()))
            bid_prices.append(p - spread/2)
            ask_prices.append(p + spread/2)
        
        # Generate volumes with time-varying properties
        bid_volumes = []
        ask_volumes = []
        
        # Start with equal volumes
        bid_vol = 2
        ask_vol = 2
        
        for i in range(num_days):
            # Volumes are influenced by price trend
            if i > 0:
                price_trend = (prices[i] / prices[i-1]) - 1
                # If price is going up, more bids than asks tend to get filled (so fewer bids remain)
                if price_trend > 0:
                    bid_vol = max(1, min(3, bid_vol - 0.2 * price_trend * 10))
                    ask_vol = min(3, max(1, ask_vol + 0.1 * price_trend * 10))
                else:
                    bid_vol = min(3, max(1, bid_vol + 0.1 * abs(price_trend) * 10))
                    ask_vol = max(1, min(3, ask_vol - 0.2 * abs(price_trend) * 10))
            
            # Add some randomness
            bid_vol_actual = max(1, min(3, int(bid_vol + np.random.randint(-1, 2))))
            ask_vol_actual = max(1, min(3, int(ask_vol + np.random.randint(-1, 2))))
            
            bid_volumes.append(bid_vol_actual)
            ask_volumes.append(ask_vol_actual)
        
        # Create DataFrame for this stock
        stock_data = pd.DataFrame({
            'price': prices,
            'bid_price': bid_prices,
            'ask_price': ask_prices,
            'bid_volume': bid_volumes,
            'ask_volume': ask_volumes
        }, index=dates)
        
        data[stock_name] = stock_data
    
    # Combine all stocks into a multi-index DataFrame
    combined_data = pd.concat(data, axis=1)
    
    return combined_data

class ImprovedMarketMaker:
    """
    Improved market maker strategy that uses direct optimization based on market conditions.
    """
    def __init__(self, n=3, imax=5, spread_capture_weight=1.0, inventory_risk_weight=0.1):
        """
        Initialize the improved market maker.
        
        Args:
            n (int): Maximum orders to place
            imax (int): Maximum inventory
            spread_capture_weight (float): Weight for spread capture component
            inventory_risk_weight (float): Weight for inventory risk component
        """
        self.n = n
        self.imax = imax
        self.spread_capture_weight = spread_capture_weight
        self.inventory_risk_weight = inventory_risk_weight
    
    def get_action(self, state, bid_price, ask_price):
        """
        Get the optimal action based on current state and prices.
        
        Args:
            state (np.array): Current state [bid_volume, ask_volume, inventory]
            bid_price (float): Current bid price
            ask_price (float): Current ask price
            
        Returns:
            tuple: Action (ea_bid_volume, ea_ask_volume)
        """
        bid_volume, ask_volume, inventory = state
        spread = ask_price - bid_price
        
        # Calculate the optimal action
        best_action = None
        best_score = float('-inf')
        
        for ea_bid in range(self.n + 1):
            for ea_ask in range(self.n + 1):
                if ea_bid + ea_ask <= self.n:
                    # Simulate the action
                    new_inventory = inventory + ea_bid - ea_ask
                    
                    # Skip actions that would exceed inventory limits
                    if abs(new_inventory) > self.imax:
                        continue
                    
                    # Calculate expected fill rates based on available volume
                    expected_bid_fill = min(ea_bid, ask_volume)
                    expected_ask_fill = min(ea_ask, bid_volume)
                    
                    # Calculate expected profit components
                    spread_capture = expected_bid_fill * bid_price - expected_ask_fill * ask_price
                    inventory_risk = -self.inventory_risk_weight * abs(new_inventory)**2
                    
                    # Calculate total score
                    score = self.spread_capture_weight * spread_capture + inventory_risk
                    
                    if score > best_score:
                        best_score = score
                        best_action = (ea_bid, ea_ask)
        
        if best_action is None:
            # Default action if no valid action was found
            return (0, 0)
        
        return best_action

def backtest_strategy(market_data, strategy, n=3, imax=5):
    """
    Backtest a strategy on market data.
    
    Args:
        market_data (pd.DataFrame): Market data
        strategy (object): Strategy with get_action method
        n (int): Maximum orders
        imax (int): Maximum inventory
        
    Returns:
        dict: Backtest results
    """
    # Initialize results
    returns = []
    daily_pnls = []
    trades = []
    inventories = {stock: 0 for stock in market_data.columns.levels[0]}
    cash = 0
    
    # Run backtest day by day
    for day in range(len(market_data.index) - 1):
        current_date = market_data.index[day]
        next_date = market_data.index[day+1]
        daily_pnl = 0
        
        # For each stock
        for stock in market_data.columns.levels[0]:
            try:
                # Get current data for this stock
                bid_volume = min(int(market_data.loc[current_date, (stock, 'bid_volume')]), n)
                ask_volume = min(int(market_data.loc[current_date, (stock, 'ask_volume')]), n)
                inventory = inventories[stock]
                bid_price = market_data.loc[current_date, (stock, 'bid_price')]
                ask_price = market_data.loc[current_date, (stock, 'ask_price')]
                
                # Skip if inventory is outside the valid range
                if abs(inventory) > imax:
                    continue
                
                # Create state
                state = [bid_volume, ask_volume, inventory]
                
                # Get action from strategy
                ea_bid_volume, ea_ask_volume = strategy.get_action(state, bid_price, ask_price)
                
                # Execute action
                bid_fill = min(ea_bid_volume, ask_volume)
                ask_fill = min(ea_ask_volume, bid_volume)
                
                # Update inventory
                old_inventory = inventories[stock]
                inventories[stock] += bid_fill - ask_fill
                
                # Calculate cash flow
                bid_cost = bid_fill * ask_price  # Buy at ask price
                ask_revenue = ask_fill * bid_price  # Sell at bid price
                cash_flow = ask_revenue - bid_cost
                cash += cash_flow
                
                # Calculate P&L
                current_price = market_data.loc[current_date, (stock, 'price')]
                next_price = market_data.loc[next_date, (stock, 'price')]
                
                # MTM the inventory
                inventory_pnl = (next_price - current_price) * inventories[stock]
                
                # Total P&L for this stock
                stock_pnl = cash_flow + inventory_pnl
                daily_pnl += stock_pnl
                
                # Record trade details
                if bid_fill > 0 or ask_fill > 0:
                    trades.append({
                        'date': current_date,
                        'stock': stock,
                        'bid_fill': bid_fill,
                        'ask_fill': ask_fill,
                        'bid_price': ask_price,
                        'ask_price': bid_price,
                        'old_inventory': old_inventory,
                        'new_inventory': inventories[stock],
                        'cash_flow': cash_flow,
                        'inventory_pnl': inventory_pnl,
                        'total_pnl': stock_pnl
                    })
            except Exception as e:
                print(f"Error processing {stock} on {current_date}: {e}")
        
        daily_pnls.append(daily_pnl)
    
    # Calculate performance metrics
    daily_returns = np.array(daily_pnls)
    cumulative_return = np.sum(daily_returns)
    
    if len(daily_returns) > 0:
        sharpe_ratio = np.mean(daily_returns) / (np.std(daily_returns) + 1e-8) * np.sqrt(252)
        max_drawdown = calculate_max_drawdown(daily_returns)
    else:
        sharpe_ratio = 0
        max_drawdown = 0
    
    return {
        'daily_returns': daily_returns,
        'cumulative_return': cumulative_return,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'trades': trades,
        'final_inventories': inventories,
        'final_cash': cash
    }

def calculate_max_drawdown(returns):
    """
    Calculate maximum drawdown from a series of returns.
    
    Args:
        returns (np.array): Array of returns
        
    Returns:
        float: Maximum drawdown
    """
    # Calculate cumulative returns
    cum_returns = np.cumsum(returns)
    
    # Calculate running maximum
    running_max = np.maximum.accumulate(cum_returns)
    
    # Calculate drawdown
    drawdown = running_max - cum_returns
    
    # Calculate maximum drawdown
    max_drawdown = np.max(drawdown)
    
    return max_drawdown

def train_and_test_enhanced_strategy():
    """
    Train and test the enhanced IRL strategy on the LOB environment.
    """
    try:
        # Initialize enhanced environment with risk-sensitive reward
        print("Initializing environment with enhanced reward...")
        env_enhanced = EnhancedLimitOrderBookEnvironment(N=3, Imax=5, T=5, reward_type='enhanced', risk_aversion=0.1)
        
        # Solve the MDP to get the optimal policy
        print("Solving MDP for enhanced reward...")
        pi_enhanced = PolicyIteration(env_enhanced, max_iterations=100)
        expert_policy_enhanced, _ = pi_enhanced.run()
        
        # Visualize the expert policy
        visualize_policy(env_enhanced, expert_policy_enhanced, title="Expert Policy for Enhanced Reward")
        
        # Train IRL on the expert policy
        print("Training IRL for enhanced reward...")
        simple_irl_enhanced = SimpleIRL(env_enhanced, expert_policy_enhanced)
        nn_enhanced = simple_irl_enhanced.train(num_demos=1000, epochs=100)
        
        # Extract policy from learned reward function
        print("Extracting policy from learned reward function...")
        irl_policy_enhanced = simple_irl_enhanced.extract_policy()
        
        # Visualize the learned policy
        visualize_policy(env_enhanced, irl_policy_enhanced, title="IRL Policy for Enhanced Reward")
        
        # Evaluate expert and IRL policies
        print("Evaluating policies for enhanced reward...")
        expert_mean, expert_std = evaluate_policy(env_enhanced, expert_policy_enhanced, num_episodes=100)
        irl_mean, irl_std = evaluate_policy(env_enhanced, irl_policy_enhanced, num_episodes=100)
        
        print(f"Enhanced Reward - Expert Policy: Mean Reward = {expert_mean:.4f}, Std Reward = {expert_std:.4f}")
        print(f"Enhanced Reward - IRL Policy: Mean Reward = {irl_mean:.4f}, Std Reward = {irl_std:.4f}")
        
        # Generate enhanced simulated market data
        print("\nGenerating enhanced simulated market data...")
        market_data = generate_enhanced_simulated_market_data(num_days=252, num_stocks=3)
        
        print("Enhanced simulated market data sample:")
        print(market_data.head())
        
        # Create the improved market maker
        improved_mm = ImprovedMarketMaker(n=3, imax=5, spread_capture_weight=1.0, inventory_risk_weight=0.1)
        
        # Backtest the improved strategy
        print("\nBacktesting improved market maker strategy...")
        improved_results = backtest_strategy(market_data, improved_mm)
        
        # Print backtest results
        print(f"Improved Market Maker - Total Return: {improved_results['cumulative_return']:.2f}, Sharpe Ratio: {improved_results['sharpe_ratio']:.2f}, Max Drawdown: {improved_results['max_drawdown']:.2f}")
        
        # Create a policy-based strategy that uses the learned IRL policy
        class IRLPolicyStrategy:
            def __init__(self, env, policy):
                self.env = env
                self.policy = policy
            
            def get_action(self, state, bid_price, ask_price):
                if tuple(state) in self.env.state_to_idx:
                    state_idx = self.env.state_to_idx[tuple(state)]
                    action_idx = self.policy[state_idx]
                    return tuple(self.env.discrete_actions[action_idx])
                return (0, 0)  # Default action if state not found
        
        # Create IRL policy strategy
        irl_strategy = IRLPolicyStrategy(env_enhanced, irl_policy_enhanced)
        
        # Backtest the IRL strategy
        print("\nBacktesting IRL policy strategy...")
        irl_results = backtest_strategy(market_data, irl_strategy)
        
        # Print backtest results
        print(f"IRL Policy - Total Return: {irl_results['cumulative_return']:.2f}, Sharpe Ratio: {irl_results['sharpe_ratio']:.2f}, Max Drawdown: {irl_results['max_drawdown']:.2f}")
        
        # Plot cumulative returns
        plt.figure(figsize=(12, 6))
        plt.plot(np.cumsum(improved_results['daily_returns']), label=f'Improved MM (Sharpe: {improved_results["sharpe_ratio"]:.2f})')
        plt.plot(np.cumsum(irl_results['daily_returns']), label=f'IRL Policy (Sharpe: {irl_results["sharpe_ratio"]:.2f})')
        plt.title('Cumulative Returns on Enhanced Simulated Market Data')
        plt.xlabel('Trading Day')
        plt.ylabel('Cumulative Return')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        
        # Return results
        return {
            'enhanced_policy': expert_policy_enhanced,
            'irl_policy': irl_policy_enhanced,
            'improved_results': improved_results,
            'irl_results': irl_results
        }
        
    except Exception as e:
        import traceback
        print(f"An error occurred: {e}")
        traceback.print_exc()
        return None

if __name__ == "__main__":
    # Train and test the enhanced strategy
    results = train_and_test_enhanced_strategy()