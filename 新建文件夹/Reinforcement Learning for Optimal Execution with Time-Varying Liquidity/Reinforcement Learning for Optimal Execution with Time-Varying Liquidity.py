import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LeakyReLU
from tensorflow.keras.optimizers import Adam
from collections import deque
import random
from tqdm import tqdm

# Set random seeds for reproducibility
np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)

class MarketEnvironment:
    """
    Environment simulating the market for optimal execution.
    """
    def __init__(self, 
                 q0=20,            # Initial inventory
                 N=10,             # Number of time steps
                 S0=10,            # Initial price
                 sigma=0.00001,    # Stock volatility
                 impact_type='constant',  # Type of impact (constant, increasing, decreasing, stochastic)
                 kappa0=0.001,     # Initial permanent impact
                 alpha0=0.002,     # Initial temporary impact
                 beta_kappa=0.0,   # Slope for linear trend in permanent impact
                 beta_alpha=0.0,   # Slope for linear trend in temporary impact
                 # Parameters for stochastic impact
                 lambda_kappa=1.0, # Mean reversion speed for permanent impact
                 lambda_alpha=1.0, # Mean reversion speed for temporary impact
                 theta_kappa=0.001,# Long-run mean for permanent impact
                 theta_alpha=0.002,# Long-run mean for temporary impact
                 sigma_kappa=0.002,# Volatility for permanent impact
                 sigma_alpha=0.002,# Volatility for temporary impact
                 omega=0.9):       # Correlation between impact processes
        
        self.q0 = q0
        self.N = N
        self.S0 = S0
        self.sigma = sigma
        
        self.impact_type = impact_type
        self.kappa0 = kappa0
        self.alpha0 = alpha0
        self.beta_kappa = beta_kappa
        self.beta_alpha = beta_alpha
        
        # Stochastic impact parameters
        self.lambda_kappa = lambda_kappa
        self.lambda_alpha = lambda_alpha
        self.theta_kappa = theta_kappa
        self.theta_alpha = theta_alpha
        self.sigma_kappa = sigma_kappa
        self.sigma_alpha = sigma_alpha
        self.omega = omega
        
        # Current state variables
        self.reset()
    
    def reset(self):
        """Reset the environment to initial state."""
        self.t = 0
        self.q = self.q0
        self.S = self.S0
        
        # Initialize impact parameters based on impact_type
        if self.impact_type == 'constant':
            self.kappa = self.kappa0
            self.alpha = self.alpha0
            self.kappa_path = [self.kappa]
            self.alpha_path = [self.alpha]
        
        elif self.impact_type in ['increasing', 'decreasing']:
            self.kappa = self.kappa0
            self.alpha = self.alpha0
            self.kappa_path = [self.kappa]
            self.alpha_path = [self.alpha]
        
        elif self.impact_type == 'stochastic':
            self.kappa = self.theta_kappa
            self.alpha = self.theta_alpha
            self.kappa_path = [self.kappa]
            self.alpha_path = [self.alpha]
        
        self.price_path = [self.S]
        self.inventory_path = [self.q]
        self.trades = []
        
        return self._get_state()
    
    def _get_state(self):
        """Return the current state representation."""
        # Normalize the state variables
        q_norm = 2 * (self.q / self.q0) - 1
        t_norm = 2 * (self.t / self.N) - 1
        S_norm = 2 * (self.S / self.S0) - 1
        
        # Return state based on feature selection
        return np.array([q_norm, t_norm, S_norm])
    
    def _update_impacts(self):
        """Update impact parameters based on their dynamics."""
        if self.impact_type == 'constant':
            # No change in impacts
            pass
        
        elif self.impact_type == 'increasing':
            # Linear increasing trend
            self.kappa = self.kappa0 + self.beta_kappa * self.t
            self.alpha = self.alpha0 + self.beta_alpha * self.t
        
        elif self.impact_type == 'decreasing':
            # Linear decreasing trend
            self.kappa = self.kappa0 - self.beta_kappa * self.t
            self.alpha = self.alpha0 - self.beta_alpha * self.t
            
            # Ensure impacts remain positive
            self.kappa = max(0.00001, self.kappa)
            self.alpha = max(0.00001, self.alpha)
        
        elif self.impact_type == 'stochastic':
            # Generate correlated random shocks
            Z1 = np.random.normal(0, 1)
            Z2 = self.omega * Z1 + np.sqrt(1 - self.omega**2) * np.random.normal(0, 1)
            
            # Update kappa using square-root mean-reverting process
            drift_kappa = self.lambda_kappa * (self.theta_kappa - self.kappa)
            diffusion_kappa = self.sigma_kappa * np.sqrt(max(0.00001, self.kappa)) * Z1
            self.kappa += drift_kappa + diffusion_kappa
            self.kappa = max(0.00001, self.kappa)  # Ensure positive
            
            # Update alpha using square-root mean-reverting process
            drift_alpha = self.lambda_alpha * (self.theta_alpha - self.alpha)
            diffusion_alpha = self.sigma_alpha * np.sqrt(max(0.00001, self.alpha)) * Z2
            self.alpha += drift_alpha + diffusion_alpha
            self.alpha = max(0.00001, self.alpha)  # Ensure positive
        
        self.kappa_path.append(self.kappa)
        self.alpha_path.append(self.alpha)
    
    def step(self, v):
        """
        Execute a trade of size v and advance the environment.
        
        Args:
            v: Number of shares to sell (must be non-negative and not exceed current inventory)
            
        Returns:
            next_state: Next state representation
            reward: Reward for the action
            done: Whether the episode is finished
            info: Additional information
        """
        # Ensure v is valid
        v = max(0, min(v, self.q))
        
        # Calculate execution price
        S_tilde = self.S - self.alpha * v
        
        # Calculate reward (cash received from sale)
        reward = S_tilde * v
        
        # Update price with permanent impact and random walk
        self.S = self.S - self.kappa * v + self.sigma * np.random.normal(0, 1)
        
        # Update inventory
        self.q -= v
        
        # Advance time
        self.t += 1
        
        # Update impact parameters for next step
        self._update_impacts()
        
        # Record history
        self.price_path.append(self.S)
        self.inventory_path.append(self.q)
        self.trades.append(v)
        
        # Check if episode is done
        done = (self.t >= self.N) or (self.q <= 0)
        
        # Get next state
        next_state = self._get_state()
        
        # Additional info
        info = {
            'execution_price': S_tilde,
            'mid_price': self.S,
            'permanent_impact': self.kappa,
            'temporary_impact': self.alpha
        }
        
        return next_state, reward, done, info
    
    def theoretical_solution(self):
        """
        Calculate the theoretical optimal solution for comparison.
        
        Returns:
            Dictionary containing optimal schedule and expected implementation shortfall
        """
        if self.impact_type == 'constant':
            # For constant impact, TWAP is optimal for risk-neutral trader
            optimal_trades = [self.q0 / self.N] * self.N
            optimal_inventory = [self.q0 - sum(optimal_trades[:i]) for i in range(self.N + 1)]
            
        elif self.impact_type in ['increasing', 'decreasing']:
            # Solve quadratic optimization problem for time-varying impacts
            # This is a simplified approximation
            kappa_path = []
            alpha_path = []
            
            for t in range(self.N):
                if self.impact_type == 'increasing':
                    kappa = self.kappa0 + self.beta_kappa * t
                    alpha = self.alpha0 + self.beta_alpha * t
                else:  # decreasing
                    kappa = max(0.00001, self.kappa0 - self.beta_kappa * t)
                    alpha = max(0.00001, self.alpha0 - self.beta_alpha * t)
                kappa_path.append(kappa)
                alpha_path.append(alpha)
            
            # Simple front-loaded or back-loaded schedule based on impact trend
            if self.impact_type == 'increasing':
                # Front-load trading
                weights = np.linspace(2, 0, self.N)
                weights /= weights.sum()
                optimal_trades = weights * self.q0
            else:
                # Back-load trading
                weights = np.linspace(0, 2, self.N)
                weights /= weights.sum()
                optimal_trades = weights * self.q0
            
            optimal_inventory = [self.q0]
            for trade in optimal_trades:
                optimal_inventory.append(optimal_inventory[-1] - trade)
            
        elif self.impact_type == 'stochastic':
            # Use approximation from Barger and Lorig (2019)
            # This is highly simplified
            # In practice, this would depend on the actual stochastic paths
            optimal_trades = [self.q0 / self.N] * self.N  # Start with TWAP
            optimal_inventory = [self.q0 - sum(optimal_trades[:i]) for i in range(self.N + 1)]
        
        return {
            'optimal_trades': optimal_trades,
            'optimal_inventory': optimal_inventory
        }

    def twap_solution(self):
        """
        Calculate the TWAP solution (equal-sized trades).
        
        Returns:
            Dictionary containing TWAP schedule and expected implementation shortfall
        """
        trades = [self.q0 / self.N] * self.N
        inventory = [self.q0 - sum(trades[:i]) for i in range(self.N + 1)]
        
        return {
            'trades': trades,
            'inventory': inventory
        }
    
    def simulate_solution(self, trading_schedule):
        """
        Simulate a trading schedule and calculate the implementation shortfall.
        
        Args:
            trading_schedule: List of trades to execute
            
        Returns:
            Dictionary containing simulation results
        """
        # Reset environment
        self.reset()
        
        # Track variables
        cash_received = 0
        
        # Execute trading schedule
        for v in trading_schedule:
            v = min(v, self.q)  # Ensure we don't sell more than we have
            
            # Calculate execution price
            S_tilde = self.S - self.alpha * v
            
            # Update cash received
            cash_received += S_tilde * v
            
            # Update price with permanent impact and random walk
            self.S = self.S - self.kappa * v + self.sigma * np.random.normal(0, 1)
            
            # Update inventory
            self.q -= v
            
            # Update impact parameters
            self._update_impacts()
        
        # Calculate implementation shortfall
        implementation_shortfall = self.S0 * self.q0 - cash_received
        
        return {
            'cash_received': cash_received,
            'implementation_shortfall': implementation_shortfall,
            'final_price': self.S,
            'final_inventory': self.q
        }

class ReplayBuffer:
    """
    Experience replay buffer for storing and sampling transitions.
    """
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state, action, reward, next_state, done):
        """Add a transition to the buffer."""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """Sample a random batch of transitions."""
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        """Return the current size of the buffer."""
        return len(self.buffer)

class DDQNAgent:
    """
    Double Deep Q-Network Agent for optimal execution.
    """
    def __init__(self, 
                 state_size, 
                 max_action=20,
                 memory_capacity=15000,
                 batch_size=32,
                 gamma=1.0,
                 epsilon=1.0,
                 epsilon_decay=0.995,
                 epsilon_min=0.01,
                 learning_rate=0.0001):
        
        self.state_size = state_size
        self.max_action = max_action  # Maximum inventory to sell
        self.action_space = max_action + 1  # Including 0
        
        # Experience replay buffer
        self.memory = ReplayBuffer(memory_capacity)
        self.batch_size = batch_size
        
        # Discount factor and exploration parameters
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Learning rate
        self.learning_rate = learning_rate
        
        # Create models
        self.main_model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()
    
    def _build_model(self):
        """Build a neural network model for Q-function approximation."""
        model = Sequential()
        model.add(Dense(30, input_dim=self.state_size))
        model.add(LeakyReLU(alpha=0.01))
        model.add(Dense(30))
        model.add(LeakyReLU(alpha=0.01))
        model.add(Dense(30))
        model.add(LeakyReLU(alpha=0.01))
        model.add(Dense(30))
        model.add(LeakyReLU(alpha=0.01))
        model.add(Dense(30))
        model.add(LeakyReLU(alpha=0.01))
        model.add(Dense(self.action_space, activation='linear'))
        
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model
    
    def update_target_model(self):
        """Update the target model with weights from the main model."""
        self.target_model.set_weights(self.main_model.get_weights())
    
    def act(self, state, inventory, time_step, total_steps):
        """
        Select an action (number of shares to sell).
        
        Args:
            state: Current state
            inventory: Current inventory
            time_step: Current time step
            total_steps: Total number of time steps
            
        Returns:
            Selected action
        """
        if np.random.rand() <= self.epsilon:
            # Exploration: sample from binomial distribution
            p_success = 1 / (total_steps - time_step) if time_step < total_steps else 1
            action = np.random.binomial(inventory, p_success)
            return action
        
        # Exploitation: select best action from Q-values
        q_values = self.main_model.predict(state.reshape(1, -1), verbose=0)[0]
        
        # Mask invalid actions (selling more than inventory)
        for a in range(inventory + 1, self.action_space):
            q_values[a] = -np.inf
        
        return np.argmax(q_values)
    
    def remember(self, state, action, reward, next_state, done):
        """Store transition in replay memory."""
        self.memory.add(state, action, reward, next_state, done)
    
    def replay(self):
        """
        Train the model using experience replay.
        
        Returns:
            Average loss from training
        """
        if len(self.memory) < self.batch_size:
            return 0
        
        # Sample random batch from replay memory
        batch = self.memory.sample(self.batch_size)
        
        # Extract batch components
        states = np.array([transition[0] for transition in batch])
        actions = np.array([transition[1] for transition in batch])
        rewards = np.array([transition[2] for transition in batch])
        next_states = np.array([transition[3] for transition in batch])
        dones = np.array([transition[4] for transition in batch])
        
        # Calculate target Q-values
        targets = np.zeros((self.batch_size, self.action_space))
        for i in range(self.batch_size):
            targets[i] = self.main_model.predict(states[i].reshape(1, -1), verbose=0)
            
            if dones[i]:
                targets[i, actions[i]] = rewards[i]
            else:
                # Double DQN update
                # Select action using main network
                a = np.argmax(self.main_model.predict(next_states[i].reshape(1, -1), verbose=0)[0])
                # Evaluate action using target network
                targets[i, actions[i]] = rewards[i] + self.gamma * self.target_model.predict(
                    next_states[i].reshape(1, -1), verbose=0)[0][a]
        
        # Train main network
        history = self.main_model.fit(states, targets, epochs=1, verbose=0)
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return history.history['loss'][0]
    
    def save(self, filepath):
        """Save the main model weights."""
        self.main_model.save_weights(filepath)
    
    def load(self, filepath):
        """Load the main model weights."""
        self.main_model.load_weights(filepath)

def train_ddqn_agent(env, agent, 
                     num_episodes=10000, 
                     target_update_freq=100, 
                     eval_freq=500,
                     use_price_feature=True):
    """
    Train the DDQN agent on the given environment.
    
    Args:
        env: Market environment
        agent: DDQN agent
        num_episodes: Number of training episodes
        target_update_freq: Frequency of target network updates
        eval_freq: Frequency of evaluation
        use_price_feature: Whether to use price as a feature
        
    Returns:
        Dictionary containing training metrics and the trained agent
    """
    # Track metrics
    rewards_history = []
    is_costs_history = []
    epsilon_history = []
    loss_history = []
    
    # Track best performance for model saving
    best_is_cost = float('inf')
    
    # For evaluation
    eval_episodes = 10
    
    # Training loop
    for episode in tqdm(range(num_episodes)):
        # Reset environment
        state = env.reset()
        
        # If not using price feature, set it to 0
        if not use_price_feature:
            state[2] = 0
        
        done = False
        episode_reward = 0
        steps = 0
        
        # Episode loop
        while not done and steps < env.N:
            # Select action
            action = agent.act(state, env.q, steps, env.N)
            
            # Execute action
            next_state, reward, done, _ = env.step(action)
            
            # If not using price feature, set it to 0
            if not use_price_feature:
                next_state[2] = 0
            
            # Store transition
            agent.remember(state, action, reward, next_state, done)
            
            # Update state
            state = next_state
            
            # Accumulate reward
            episode_reward += reward
            
            # Train agent
            loss = agent.replay()
            if loss > 0:
                loss_history.append(loss)
            
            # Update step count
            steps += 1
        
        # Update target network
        if episode % target_update_freq == 0:
            agent.update_target_model()
        
        # Calculate implementation shortfall
        is_cost = env.S0 * env.q0 - episode_reward
        
        # Record metrics
        rewards_history.append(episode_reward)
        is_costs_history.append(is_cost)
        epsilon_history.append(agent.epsilon)
        
        # Evaluate agent
        if episode % eval_freq == 0:
            eval_is_costs = []
            
            for _ in range(eval_episodes):
                eval_env = MarketEnvironment(
                    impact_type=env.impact_type,
                    kappa0=env.kappa0,
                    alpha0=env.alpha0,
                    beta_kappa=env.beta_kappa,
                    beta_alpha=env.beta_alpha,
                    lambda_kappa=env.lambda_kappa,
                    lambda_alpha=env.lambda_alpha,
                    theta_kappa=env.theta_kappa,
                    theta_alpha=env.theta_alpha,
                    sigma_kappa=env.sigma_kappa,
                    sigma_alpha=env.sigma_alpha
                )
                
                eval_state = eval_env.reset()
                if not use_price_feature:
                    eval_state[2] = 0
                
                eval_done = False
                eval_reward = 0
                eval_steps = 0
                
                while not eval_done and eval_steps < eval_env.N:
                    # Always use exploitation during evaluation
                    q_values = agent.main_model.predict(eval_state.reshape(1, -1), verbose=0)[0]
                    
                    # Mask invalid actions
                    for a in range(eval_env.q + 1, agent.action_space):
                        q_values[a] = -np.inf
                    
                    eval_action = np.argmax(q_values)
                    
                    # Execute action
                    eval_next_state, eval_step_reward, eval_done, _ = eval_env.step(eval_action)
                    
                    if not use_price_feature:
                        eval_next_state[2] = 0
                    
                    # Update state and reward
                    eval_state = eval_next_state
                    eval_reward += eval_step_reward
                    eval_steps += 1
                
                # Calculate implementation shortfall
                eval_is_cost = eval_env.S0 * eval_env.q0 - eval_reward
                eval_is_costs.append(eval_is_cost)
            
            avg_eval_is_cost = np.mean(eval_is_costs)
            
            # Save model if it's the best
            if avg_eval_is_cost < best_is_cost:
                best_is_cost = avg_eval_is_cost
                agent.save(f"ddqn_best_{env.impact_type}.h5")
                
            print(f"Episode {episode}/{num_episodes}, Avg Eval IS Cost: {avg_eval_is_cost:.4f}, Epsilon: {agent.epsilon:.4f}")
    
    # Return metrics
    return {
        'rewards_history': rewards_history,
        'is_costs_history': is_costs_history,
        'epsilon_history': epsilon_history,
        'loss_history': loss_history,
        'agent': agent
    }

def test_agent(env, agent, num_episodes=1000, use_price_feature=True):
    """
    Test the agent's performance on the environment.
    
    Args:
        env: Market environment
        agent: Trained DDQN agent
        num_episodes: Number of test episodes
        use_price_feature: Whether to use price as a feature
        
    Returns:
        Dictionary containing test metrics
    """
    # Load best model weights
    try:
        agent.load(f"ddqn_best_{env.impact_type}.h5")
        print(f"Loaded best model for {env.impact_type} impact")
    except:
        print(f"No saved model found for {env.impact_type} impact, using current model")
    
    # Track metrics
    is_costs = []
    trading_schedules = []
    inventory_paths = []
    
    # Theoretical solutions for comparison
    theoretical_is_costs = []
    twap_is_costs = []
    
    # Test loop
    for episode in tqdm(range(num_episodes)):
        # Reset environment
        state = env.reset()
        
        # If not using price feature, set it to 0
        if not use_price_feature:
            state[2] = 0
        
        done = False
        episode_reward = 0
        episode_trades = []
        
        # Episode loop
        while not done:
            # Select action (always exploit)
            q_values = agent.main_model.predict(state.reshape(1, -1), verbose=0)[0]
            
            # Mask invalid actions
            for a in range(env.q + 1, agent.action_space):
                q_values[a] = -np.inf
            
            action = np.argmax(q_values)
            
            # Execute action
            next_state, reward, done, _ = env.step(action)
            
            # If not using price feature, set it to 0
            if not use_price_feature:
                next_state[2] = 0
            
            # Update state
            state = next_state
            
            # Accumulate reward and trades
            episode_reward += reward
            episode_trades.append(action)
        
        # Calculate implementation shortfall
        is_cost = env.S0 * env.q0 - episode_reward
        
        # Get theoretical solution
        theo_sol = env.theoretical_solution()
        theo_result = env.simulate_solution(theo_sol['optimal_trades'])
        
        # Get TWAP solution
        twap_sol = env.twap_solution()
        twap_result = env.simulate_solution(twap_sol['trades'])
        
        # Record metrics
        is_costs.append(is_cost)
        theoretical_is_costs.append(theo_result['implementation_shortfall'])
        twap_is_costs.append(twap_result['implementation_shortfall'])
        trading_schedules.append(episode_trades)
        inventory_paths.append(env.inventory_path)
    
    # Calculate metrics
    avg_is_cost = np.mean(is_costs)
    std_is_cost = np.std(is_costs)
    
    avg_theo_is_cost = np.mean(theoretical_is_costs)
    std_theo_is_cost = np.std(theoretical_is_costs)
    
    avg_twap_is_cost = np.mean(twap_is_costs)
    std_twap_is_cost = np.std(twap_is_costs)
    
    # Calculate P&L difference
    delta_pnl_theo = (avg_is_cost - avg_theo_is_cost) / avg_theo_is_cost * 10000  # in basis points
    delta_pnl_twap = (avg_is_cost - avg_twap_is_cost) / avg_twap_is_cost * 10000  # in basis points
    
    # Average trading schedule
    avg_trading_schedule = np.mean(np.array(trading_schedules), axis=0)
    
    # Average inventory path
    avg_inventory_path = np.mean(np.array(inventory_paths), axis=0)
    
    return {
        'avg_is_cost': avg_is_cost,
        'std_is_cost': std_is_cost,
        'avg_theo_is_cost': avg_theo_is_cost,
        'std_theo_is_cost': std_theo_is_cost,
        'avg_twap_is_cost': avg_twap_is_cost,
        'std_twap_is_cost': std_twap_is_cost,
        'delta_pnl_theo': delta_pnl_theo,
        'delta_pnl_twap': delta_pnl_twap,
        'avg_trading_schedule': avg_trading_schedule,
        'avg_inventory_path': avg_inventory_path
    }

def visualize_results(test_results, env_type, use_price_feature):
    """
    Visualize test results.
    
    Args:
        test_results: Dictionary of test results
        env_type: Environment type (impact model)
        use_price_feature: Whether price was used as a feature
    """
    # Create figure
    fig, axs = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot trading schedule
    axs[0].bar(range(1, len(test_results['avg_trading_schedule'])+1), 
               test_results['avg_trading_schedule'], 
               alpha=0.7, 
               label='DDQN')
    
    # Calculate TWAP schedule
    twap_schedule = [env.q0 / env.N] * env.N
    
    # Calculate theoretical schedule based on env_type
    if env_type == 'constant':
        theo_schedule = twap_schedule
    elif env_type == 'increasing':
        # Front-loaded
        weights = np.linspace(2, 0, env.N)
        weights /= weights.sum()
        theo_schedule = weights * env.q0
    elif env_type == 'decreasing':
        # Back-loaded
        weights = np.linspace(0, 2, env.N)
        weights /= weights.sum()
        theo_schedule = weights * env.q0
    else:  # stochastic
        theo_schedule = twap_schedule
    
    axs[0].plot(range(1, len(theo_schedule)+1), theo_schedule, 'g-', marker='o', label='Theoretical')
    axs[0].plot(range(1, len(twap_schedule)+1), twap_schedule, 'r--', marker='x', label='TWAP')
    
    axs[0].set_xlabel('Time Step')
    axs[0].set_ylabel('Shares Sold')
    axs[0].set_title(f'Average Trading Schedule - {env_type.capitalize()} Impact')
    axs[0].legend()
    axs[0].grid(alpha=0.3)
    
    # Plot inventory path
    axs[1].plot(range(len(test_results['avg_inventory_path'])), 
                test_results['avg_inventory_path'], 
                'b-', 
                marker='o',
                label='DDQN')
    
    # Calculate TWAP inventory path
    twap_inventory = [env.q0 - sum(twap_schedule[:i]) for i in range(env.N + 1)]
    
    # Calculate theoretical inventory path
    theo_inventory = [env.q0 - sum(theo_schedule[:i]) for i in range(env.N + 1)]
    
    axs[1].plot(range(len(theo_inventory)), theo_inventory, 'g-', marker='o', label='Theoretical')
    axs[1].plot(range(len(twap_inventory)), twap_inventory, 'r--', marker='x', label='TWAP')
    
    axs[1].set_xlabel('Time Step')
    axs[1].set_ylabel('Inventory')
    axs[1].set_title(f'Average Inventory Path - {env_type.capitalize()} Impact')
    axs[1].legend()
    axs[1].grid(alpha=0.3)
    
    # Add text with metrics
    plt.figtext(0.5, 0.01, 
                f"DDQN IS Cost: {test_results['avg_is_cost']:.4f} ± {test_results['std_is_cost']:.4f}\n"
                f"Theoretical IS Cost: {test_results['avg_theo_is_cost']:.4f} ± {test_results['std_theo_is_cost']:.4f}\n"
                f"TWAP IS Cost: {test_results['avg_twap_is_cost']:.4f} ± {test_results['std_twap_is_cost']:.4f}\n"
                f"ΔP&L vs Theoretical: {test_results['delta_pnl_theo']:.2f} bps\n"
                f"ΔP&L vs TWAP: {test_results['delta_pnl_twap']:.2f} bps\n"
                f"Using Price Feature: {use_price_feature}",
                ha='center', fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout(rect=[0, 0.08, 1, 0.95])
    plt.savefig(f"ddqn_{env_type}_impact_results.png", dpi=300, bbox_inches='tight')
    plt.close()

# Create and configure environment
env = MarketEnvironment(
    q0=20,  # Initial inventory
    N=10,   # Number of time steps
    S0=10,  # Initial price
    sigma=0.00001,  # Stock volatility
    impact_type='constant',  # Type of impact
    kappa0=0.001,  # Initial permanent impact
    alpha0=0.002   # Initial temporary impact
)

# Create agent
state_size = 3  # (q, t, S)
agent = DDQNAgent(
    state_size=state_size,
    max_action=env.q0,
    memory_capacity=15000,
    batch_size=32,
    gamma=1.0,
    epsilon=1.0,
    epsilon_decay=0.995,
    epsilon_min=0.01,
    learning_rate=0.0001
)

# Experiment configurations
experiments = [
    {
        'name': 'Constant Impact',
        'impact_type': 'constant',
        'kappa0': 0.001,
        'alpha0': 0.002,
        'beta_kappa': 0.0,
        'beta_alpha': 0.0,
        'use_price_feature': [False, True]
    },
    {
        'name': 'Increasing Impact',
        'impact_type': 'increasing',
        'kappa0': 0.0001,
        'alpha0': 0.0001,
        'beta_kappa': 0.0002,
        'beta_alpha': 0.0004,
        'use_price_feature': [False, True]
    },
    {
        'name': 'Decreasing Impact',
        'impact_type': 'decreasing',
        'kappa0': 0.002,
        'alpha0': 0.004,
        'beta_kappa': 0.0002,
        'beta_alpha': 0.0004,
        'use_price_feature': [False, True]
    },
    {
        'name': 'Stochastic Impact (Low Mean Reversion)',
        'impact_type': 'stochastic',
        'lambda_kappa': 1.0,
        'lambda_alpha': 1.0,
        'theta_kappa': 0.001,
        'theta_alpha': 0.002,
        'sigma_kappa': 0.002,
        'sigma_alpha': 0.002,
        'use_price_feature': [True]
    },
    {
        'name': 'Stochastic Impact (High Mean Reversion)',
        'impact_type': 'stochastic',
        'lambda_kappa': 5.0,
        'lambda_alpha': 5.0,
        'theta_kappa': 0.001,
        'theta_alpha': 0.002,
        'sigma_kappa': 0.002,
        'sigma_alpha': 0.002,
        'use_price_feature': [True]
    }
]

# Run experiments
results = {}

for exp in experiments:
    exp_name = exp['name']
    print(f"\n=== Running experiment: {exp_name} ===")
    
    for use_price in exp['use_price_feature']:
        feature_name = "with_price" if use_price else "without_price"
        print(f"\n--- {feature_name} ---")
        
        # Create environment with specified parameters
        if exp['impact_type'] == 'stochastic':
            env = MarketEnvironment(
                impact_type=exp['impact_type'],
                lambda_kappa=exp['lambda_kappa'],
                lambda_alpha=exp['lambda_alpha'],
                theta_kappa=exp['theta_kappa'],
                theta_alpha=exp['theta_alpha'],
                sigma_kappa=exp['sigma_kappa'],
                sigma_alpha=exp['sigma_alpha']
            )
        else:
            env = MarketEnvironment(
                impact_type=exp['impact_type'],
                kappa0=exp['kappa0'],
                alpha0=exp['alpha0'],
                beta_kappa=exp.get('beta_kappa', 0.0),
                beta_alpha=exp.get('beta_alpha', 0.0)
            )
        
        # Create new agent
        agent = DDQNAgent(
            state_size=state_size,
            max_action=env.q0
        )
        
        # Train agent
        print("Training agent...")
        train_results = train_ddqn_agent(
            env=env,
            agent=agent,
            num_episodes=10000,  # Reduced for demonstration
            use_price_feature=use_price
        )
        
        # Test agent
        print("Testing agent...")
        test_results = test_agent(
            env=env,
            agent=agent,
            num_episodes=1000,  # Reduced for demonstration
            use_price_feature=use_price
        )
        
        # Store results
        key = f"{exp_name}_{feature_name}"
        results[key] = {
            'train': train_results,
            'test': test_results
        }
        
        # Visualize results
        visualize_results(test_results, exp['impact_type'], use_price)

# Summarize results
print("\n=== Summary of Results ===")
print("{:<50} {:<15} {:<15} {:<15}".format(
    "Experiment", "DDQN IS Cost", "Theoretical IS Cost", "TWAP IS Cost"))
print("-" * 95)

for key, result in results.items():
    test_result = result['test']
    print("{:<50} {:<15.4f} {:<15.4f} {:<15.4f}".format(
        key, 
        test_result['avg_is_cost'],
        test_result['avg_theo_is_cost'],
        test_result['avg_twap_is_cost']))

print("\n{:<50} {:<15} {:<15}".format(
    "Experiment", "ΔP&L vs Theo (bps)", "ΔP&L vs TWAP (bps)"))
print("-" * 80)

for key, result in results.items():
    test_result = result['test']
    print("{:<50} {:<15.2f} {:<15.2f}".format(
        key, 
        test_result['delta_pnl_theo'],
        test_result['delta_pnl_twap']))

# Create comparison of trading schedules
plt.figure(figsize=(15, 10))

for i, exp in enumerate(experiments[:3]):  # Just constant, increasing, decreasing
    plt.subplot(3, 1, i+1)
    
    # Get results with price feature
    key = f"{exp['name']}_with_price"
    if key in results:
        test_result = results[key]['test']
        plt.plot(range(1, len(test_result['avg_trading_schedule'])+1), 
                test_result['avg_trading_schedule'], 
                'b-', marker='o', label='DDQN')
        
        # Add theoretical and TWAP for comparison
        if exp['impact_type'] == 'constant':
            theo_schedule = [env.q0 / env.N] * env.N
        elif exp['impact_type'] == 'increasing':
            weights = np.linspace(2, 0, env.N)
            weights /= weights.sum()
            theo_schedule = weights * env.q0
        elif exp['impact_type'] == 'decreasing':
            weights = np.linspace(0, 2, env.N)
            weights /= weights.sum()
            theo_schedule = weights * env.q0
        
        twap_schedule = [env.q0 / env.N] * env.N
        
        plt.plot(range(1, len(theo_schedule)+1), theo_schedule, 'g-', marker='o', label='Theoretical')
        plt.plot(range(1, len(twap_schedule)+1), twap_schedule, 'r--', marker='x', label='TWAP')
        
        plt.title(f"{exp['name']} - Trading Schedule")
        plt.xlabel('Time Step')
        plt.ylabel('Shares Sold')
        plt.grid(alpha=0.3)
        plt.legend()

plt.tight_layout()
plt.savefig("ddqn_trading_schedules_comparison.png", dpi=300, bbox_inches='tight')
plt.close()

# Create comparison of implementation shortfall costs
plt.figure(figsize=(12, 6))

# Collect data for bar chart
exp_names = []
ddqn_costs = []
theo_costs = []
twap_costs = []

for exp in experiments:
    for use_price in exp['use_price_feature']:
        feature_name = "with_price" if use_price else "without_price"
        key = f"{exp['name']}_{feature_name}"
        
        if key in results:
            exp_names.append(f"{exp['name']}\n({feature_name})")
            test_result = results[key]['test']
            ddqn_costs.append(test_result['avg_is_cost'])
            theo_costs.append(test_result['avg_theo_is_cost'])
            twap_costs.append(test_result['avg_twap_is_cost'])

# Create grouped bar chart
x = np.arange(len(exp_names))
width = 0.25

plt.bar(x - width, ddqn_costs, width, label='DDQN', color='blue', alpha=0.7)
plt.bar(x, theo_costs, width, label='Theoretical', color='green', alpha=0.7)
plt.bar(x + width, twap_costs, width, label='TWAP', color='red', alpha=0.7)

plt.xlabel('Experiment')
plt.ylabel('Implementation Shortfall Cost')
plt.title('Comparison of Implementation Shortfall Costs')
plt.xticks(x, exp_names, rotation=45, ha='right')
plt.legend()
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()

plt.savefig("ddqn_costs_comparison.png", dpi=300, bbox_inches='tight')
plt.close()

# Mixed Impact Experiment (Train on both increasing and decreasing)
print("\n=== Running Mixed Impact Experiment ===")

# Create environments
env_increasing = MarketEnvironment(
    impact_type='increasing',
    kappa0=0.0001,
    alpha0=0.0001,
    beta_kappa=0.0002,
    beta_alpha=0.0004
)

env_decreasing = MarketEnvironment(
    impact_type='decreasing',
    kappa0=0.002,
    alpha0=0.004,
    beta_kappa=0.0002,
    beta_alpha=0.0004
)

# Create agent
agent = DDQNAgent(
    state_size=state_size,
    max_action=env_increasing.q0
)

# Train agent on mixed environments
print("Training agent on mixed environments...")

# Custom training loop
num_episodes = 20000
target_update_freq = 100
rewards_history = []
is_costs_history = []
epsilon_history = []

for episode in tqdm(range(num_episodes)):
    # Alternate between increasing and decreasing impact
    if episode % 2 == 0:
        env = env_increasing
    else:
        env = env_decreasing
    
    # Reset environment
    state = env.reset()
    
    done = False
    episode_reward = 0
    steps = 0
    
    # Episode loop
    while not done and steps < env.N:
        # Select action
        action = agent.act(state, env.q, steps, env.N)
        
        # Execute action
        next_state, reward, done, _ = env.step(action)
        
        # Store transition
        agent.remember(state, action, reward, next_state, done)
        
        # Update state
        state = next_state
        
        # Accumulate reward
        episode_reward += reward
        
        # Train agent
        agent.replay()
        
        # Update step count
        steps += 1
    
    # Update target network
    if episode % target_update_freq == 0:
        agent.update_target_model()
    
    # Calculate implementation shortfall
    is_cost = env.S0 * env.q0 - episode_reward
    
    # Record metrics
    rewards_history.append(episode_reward)
    is_costs_history.append(is_cost)
    epsilon_history.append(agent.epsilon)

# Save model
agent.save("ddqn_mixed_impact.h5")

# Test agent on increasing impact
print("Testing agent on increasing impact...")
test_results_increasing = test_agent(
    env=env_increasing,
    agent=agent,
    num_episodes=1000
)

# Test agent on decreasing impact
print("Testing agent on decreasing impact...")
test_results_decreasing = test_agent(
    env=env_decreasing,
    agent=agent,
    num_episodes=1000
)

# Visualize results for increasing impact
visualize_results(test_results_increasing, 'increasing', True)

# Visualize results for decreasing impact
visualize_results(test_results_decreasing, 'decreasing', True)

# Print summary
print("\n=== Mixed Impact Experiment Results ===")
print("\nIncreasing Impact Test:")
print(f"DDQN IS Cost: {test_results_increasing['avg_is_cost']:.4f}")
print(f"Theoretical IS Cost: {test_results_increasing['avg_theo_is_cost']:.4f}")
print(f"TWAP IS Cost: {test_results_increasing['avg_twap_is_cost']:.4f}")
print(f"ΔP&L vs Theoretical: {test_results_increasing['delta_pnl_theo']:.2f} bps")
print(f"ΔP&L vs TWAP: {test_results_increasing['delta_pnl_twap']:.2f} bps")

print("\nDecreasing Impact Test:")
print(f"DDQN IS Cost: {test_results_decreasing['avg_is_cost']:.4f}")
print(f"Theoretical IS Cost: {test_results_decreasing['avg_theo_is_cost']:.4f}")
print(f"TWAP IS Cost: {test_results_decreasing['avg_twap_is_cost']:.4f}")
print(f"ΔP&L vs Theoretical: {test_results_decreasing['delta_pnl_theo']:.2f} bps")
print(f"ΔP&L vs TWAP: {test_results_decreasing['delta_pnl_twap']:.2f} bps")

# Create comparison of mixed impact results
plt.figure(figsize=(12, 6))

# Data for bar chart
scenarios = ['Increasing Impact', 'Decreasing Impact']
ddqn_costs = [test_results_increasing['avg_is_cost'], test_results_decreasing['avg_is_cost']]
theo_costs = [test_results_increasing['avg_theo_is_cost'], test_results_decreasing['avg_theo_is_cost']]
twap_costs = [test_results_increasing['avg_twap_is_cost'], test_results_decreasing['avg_twap_is_cost']]

# Create grouped bar chart
x = np.arange(len(scenarios))
width = 0.25

plt.bar(x - width, ddqn_costs, width, label='DDQN (Mixed Training)', color='blue', alpha=0.7)
plt.bar(x, theo_costs, width, label='Theoretical', color='green', alpha=0.7)
plt.bar(x + width, twap_costs, width, label='TWAP', color='red', alpha=0.7)

plt.xlabel('Test Scenario')
plt.ylabel('Implementation Shortfall Cost')
plt.title('Mixed Impact Training Results')
plt.xticks(x, scenarios)
plt.legend()
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()

plt.savefig("ddqn_mixed_impact_results.png", dpi=300, bbox_inches='tight')
plt.close()

print("\nExperiment completed. Results saved as PNG files.")