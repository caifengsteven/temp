import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam
import seaborn as sns
from collections import deque
import random
from tqdm import tqdm
import time

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

class BlackScholesEnvironment:
    """
    Environment for option hedging under the Black-Scholes model
    """
    
    def __init__(self, S0=100, K=100, T=30, r=0, sigma=0.2, mu=0.05, alpha=0.001, dt=1):
        """
        Initialize the Black-Scholes environment
        
        Parameters:
        -----------
        S0 : float
            Initial stock price
        K : float
            Strike price of the option
        T : int
            Time to maturity in days
        r : float
            Risk-free interest rate (annualized)
        sigma : float
            Volatility of the stock (annualized)
        mu : float
            Expected return of the stock (annualized)
        alpha : float
            Transaction cost parameter
        dt : float
            Time step in days
        """
        self.S0 = S0
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.mu = mu
        self.alpha = alpha
        self.dt = dt
        
        # Calculate the number of time steps
        self.n_steps = int(T / dt)
        
        # Convert annual parameters to daily
        self.daily_mu = mu / 252
        self.daily_r = r / 252
        self.daily_sigma = sigma / np.sqrt(252)
        
        # Initialize state variables
        self.reset()
    
    def reset(self):
        """
        Reset the environment to its initial state
        
        Returns:
        --------
        numpy.ndarray
            Initial state
        """
        self.current_step = 0
        self.S = self.S0
        self.stock_holdings = 0
        self.cash = 0
        self.done = False
        
        # Create array to store stock price path
        self.price_path = np.zeros(self.n_steps + 1)
        self.price_path[0] = self.S0
        
        # Create arrays to store portfolio values and stock holdings
        self.portfolio_values = np.zeros(self.n_steps + 1)
        self.holdings_path = np.zeros(self.n_steps + 1)
        
        return self._get_state()
    
    def step(self, action):
        """
        Take a step in the environment
        
        Parameters:
        -----------
        action : float
            New target stock holdings
        
        Returns:
        --------
        tuple
            (new_state, reward, done, info)
        """
        # Calculate the change in stock holdings
        delta_holdings = action - self.stock_holdings
        
        # Calculate transaction cost
        transaction_cost = self.alpha * abs(delta_holdings * self.S)
        
        # Update cash based on stock purchases/sales and transaction costs
        self.cash -= delta_holdings * self.S + transaction_cost
        
        # Update stock holdings
        self.stock_holdings = action
        self.holdings_path[self.current_step] = action
        
        # Calculate current portfolio value (before stock price changes)
        portfolio_value = self.cash + self.stock_holdings * self.S
        self.portfolio_values[self.current_step] = portfolio_value
        
        # Move to next time step
        self.current_step += 1
        
        # Generate new stock price
        if self.current_step <= self.n_steps:
            # Generate random normal for stock price movement
            z = np.random.normal(0, 1)
            
            # Update stock price using log-normal model
            self.S = self.S * np.exp((self.daily_mu - 0.5 * self.daily_sigma**2) * self.dt + 
                                    self.daily_sigma * np.sqrt(self.dt) * z)
            
            # Store stock price
            self.price_path[self.current_step] = self.S
        
        # Check if we've reached maturity
        if self.current_step == self.n_steps:
            self.done = True
            
            # Calculate option payoff at maturity
            option_payoff = max(self.S - self.K, 0)
            
            # Calculate final portfolio value
            final_portfolio_value = self.cash + self.stock_holdings * self.S
            
            # Calculate the final transaction cost (liquidation of position)
            final_transaction_cost = self.alpha * abs(self.stock_holdings * self.S)
            
            # Calculate P&L (negative of hedging error)
            # For short option position, we need to pay the option payoff
            self.pnl = final_portfolio_value - final_transaction_cost - option_payoff
            
            # For the final quadratic hedging objective, we use negative squared P&L as reward
            reward = -self.pnl**2
        else:
            # No intermediate reward in final quadratic hedging
            reward = 0
            
        return self._get_state(), reward, self.done, {"pnl": self.pnl if self.done else None}
    
    def _get_state(self):
        """
        Get the current state representation
        
        Returns:
        --------
        numpy.ndarray
            Current state
        """
        # State includes: time to maturity, stock price, stock holdings
        time_to_maturity = (self.n_steps - self.current_step) / self.n_steps
        
        return np.array([time_to_maturity, self.S / self.S0, self.stock_holdings])
    
    def black_scholes_delta(self):
        """
        Calculate Black-Scholes delta for hedging
        
        Returns:
        --------
        float
            Black-Scholes delta (hedge ratio)
        """
        # Time to maturity in years
        tau = (self.n_steps - self.current_step) * self.dt / 252
        
        if tau <= 0:
            # At maturity, delta is either 0 or 1
            return 1.0 if self.S > self.K else 0.0
        
        # Calculate d1 from Black-Scholes formula
        d1 = (np.log(self.S / self.K) + (self.r + 0.5 * self.sigma**2) * tau) / (self.sigma * np.sqrt(tau))
        
        # Calculate delta
        delta = self._norm_cdf(d1)
        
        return delta
    
    def _norm_cdf(self, x):
        """
        Standard normal cumulative distribution function
        
        Parameters:
        -----------
        x : float
            Input value
        
        Returns:
        --------
        float
            CDF value
        """
        return 0.5 * (1 + np.erf(x / np.sqrt(2)))

class SABREnvironment(BlackScholesEnvironment):
    """
    Environment for option hedging under the SABR model
    """
    
    def __init__(self, S0=100, K=100, T=30, r=0, sigma0=0.2, mu=0.05, alpha=0.001, 
                 eta=0.0095, rho=0.5, dt=1):
        """
        Initialize the SABR environment
        
        Additional Parameters (compared to BlackScholesEnvironment):
        -----------
        sigma0 : float
            Initial volatility
        eta : float
            Volatility of volatility (annualized)
        rho : float
            Correlation between stock price and volatility
        """
        super().__init__(S0, K, T, r, sigma0, mu, alpha, dt)
        
        self.sigma0 = sigma0
        self.eta = eta
        self.rho = rho
        
        # Convert annual parameters to daily
        self.daily_eta = eta / np.sqrt(252)
        
        # Initialize additional state variables
        self.sigma = sigma0
        
        # Create array to store volatility path
        self.vol_path = np.zeros(self.n_steps + 1)
        self.vol_path[0] = sigma0
    
    def reset(self):
        """
        Reset the environment to its initial state
        
        Returns:
        --------
        numpy.ndarray
            Initial state
        """
        self.current_step = 0
        self.S = self.S0
        self.sigma = self.sigma0
        self.stock_holdings = 0
        self.cash = 0
        self.done = False
        
        # Create arrays to store paths
        self.price_path = np.zeros(self.n_steps + 1)
        self.price_path[0] = self.S0
        
        self.vol_path = np.zeros(self.n_steps + 1)
        self.vol_path[0] = self.sigma0
        
        self.portfolio_values = np.zeros(self.n_steps + 1)
        self.holdings_path = np.zeros(self.n_steps + 1)
        
        return self._get_state()
    
    def step(self, action):
        """
        Take a step in the SABR environment
        
        Parameters:
        -----------
        action : float
            New target stock holdings
        
        Returns:
        --------
        tuple
            (new_state, reward, done, info)
        """
        # Calculate the change in stock holdings
        delta_holdings = action - self.stock_holdings
        
        # Calculate transaction cost
        transaction_cost = self.alpha * abs(delta_holdings * self.S)
        
        # Update cash based on stock purchases/sales and transaction costs
        self.cash -= delta_holdings * self.S + transaction_cost
        
        # Update stock holdings
        self.stock_holdings = action
        self.holdings_path[self.current_step] = action
        
        # Calculate current portfolio value (before stock price changes)
        portfolio_value = self.cash + self.stock_holdings * self.S
        self.portfolio_values[self.current_step] = portfolio_value
        
        # Move to next time step
        self.current_step += 1
        
        # Generate new stock price and volatility
        if self.current_step <= self.n_steps:
            # Generate correlated random normals
            z1 = np.random.normal(0, 1)
            z2 = self.rho * z1 + np.sqrt(1 - self.rho**2) * np.random.normal(0, 1)
            
            # Update volatility using SABR model
            self.sigma = self.sigma * np.exp(self.daily_eta/2 * np.sqrt(self.dt) * z2)
            
            # Update stock price using SABR model
            self.S = self.S * np.exp((self.daily_mu - 0.5 * self.sigma**2) * self.dt + 
                                     self.sigma * np.sqrt(self.dt) * z1)
            
            # Store stock price and volatility
            self.price_path[self.current_step] = self.S
            self.vol_path[self.current_step] = self.sigma
        
        # Check if we've reached maturity
        if self.current_step == self.n_steps:
            self.done = True
            
            # Calculate option payoff at maturity
            option_payoff = max(self.S - self.K, 0)
            
            # Calculate final portfolio value
            final_portfolio_value = self.cash + self.stock_holdings * self.S
            
            # Calculate the final transaction cost (liquidation of position)
            final_transaction_cost = self.alpha * abs(self.stock_holdings * self.S)
            
            # Calculate P&L (negative of hedging error)
            # For short option position, we need to pay the option payoff
            self.pnl = final_portfolio_value - final_transaction_cost - option_payoff
            
            # For the final quadratic hedging objective, we use negative squared P&L as reward
            reward = -self.pnl**2
        else:
            # No intermediate reward in final quadratic hedging
            reward = 0
            
        return self._get_state(), reward, self.done, {"pnl": self.pnl if self.done else None}
    
    def bartlett_delta(self):
        """
        Calculate Bartlett's delta for hedging under SABR model
        
        Returns:
        --------
        float
            Bartlett's delta (hedge ratio)
        """
        # Time to maturity in years
        tau = (self.n_steps - self.current_step) * self.dt / 252
        
        if tau <= 0:
            # At maturity, delta is either 0 or 1
            return 1.0 if self.S > self.K else 0.0
        
        # Calculate implied volatility using SABR approximation
        implied_vol = self._sabr_implied_vol(tau)
        
        # Calculate d1 from Black-Scholes formula with implied vol
        d1 = (np.log(self.S / self.K) + (self.r + 0.5 * implied_vol**2) * tau) / (implied_vol * np.sqrt(tau))
        
        # Calculate BS delta
        bs_delta = self._norm_cdf(d1)
        
        # Calculate SABR correction term
        M = (self.eta / self.sigma) * np.log(self.K / self.S)
        f = self._sabr_f_function(M)
        f_prime = self._sabr_f_prime(M)
        
        # Bartlett's delta includes vega term correction
        vega = self.S * self._norm_pdf(d1) * np.sqrt(tau)
        vega_correction = (self.eta / (2 * self.S)) * (self.rho * f - (self.rho * M + 2) * f_prime)
        
        # Bartlett's delta
        bartlett_delta = bs_delta + vega * vega_correction
        
        return bartlett_delta
    
    def _sabr_implied_vol(self, tau):
        """
        Calculate SABR implied volatility
        
        Parameters:
        -----------
        tau : float
            Time to maturity in years
        
        Returns:
        --------
        float
            SABR implied volatility
        """
        # SABR parameters
        alpha = self.sigma
        beta = 1.0  # log-normal SABR
        nu = self.eta
        rho = self.rho
        
        # ATM approximation for simplicity
        F = self.S
        K = self.K
        
        # Calculate z
        z = (nu / alpha) * (np.log(F / K))
        
        # Calculate implied volatility using SABR formula
        I = alpha * self._sabr_f_function(z)
        
        return I
    
    def _sabr_f_function(self, y):
        """
        SABR f function
        
        Parameters:
        -----------
        y : float
            Input value
        
        Returns:
        --------
        float
            f(y) value
        """
        rho = self.rho
        if abs(y) < 1e-6:
            # Special case for small y to avoid numerical issues
            return 1.0
        
        # Regular case
        term1 = y / 2
        term2 = np.log((1 - rho) / np.sqrt(1 + rho * y + y**2/4))
        term3 = - rho - y/2
        
        return term1 / (term2 + term3)
    
    def _sabr_f_prime(self, y):
        """
        Derivative of SABR f function
        
        Parameters:
        -----------
        y : float
            Input value
        
        Returns:
        --------
        float
            f'(y) value
        """
        # Numerical approximation of derivative
        h = 1e-5
        return (self._sabr_f_function(y + h) - self._sabr_f_function(y - h)) / (2 * h)
    
    def _norm_pdf(self, x):
        """
        Standard normal probability density function
        
        Parameters:
        -----------
        x : float
            Input value
        
        Returns:
        --------
        float
            PDF value
        """
        return np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)

class DeltaHedgeAgent:
    """
    Agent that implements Delta hedging
    """
    
    def __init__(self, env):
        """
        Initialize the Delta hedging agent
        
        Parameters:
        -----------
        env : Environment
            The option hedging environment
        """
        self.env = env
    
    def act(self, state):
        """
        Choose action based on Black-Scholes delta
        
        Parameters:
        -----------
        state : numpy.ndarray
            Current state
        
        Returns:
        --------
        float
            Delta hedge ratio
        """
        # Use Black-Scholes delta for hedging
        if isinstance(self.env, SABREnvironment):
            return self.env.bartlett_delta()
        else:
            return self.env.black_scholes_delta()

class ReplayBuffer:
    """
    Experience replay buffer for storing and sampling transitions
    """
    
    def __init__(self, capacity=100000):
        """
        Initialize the replay buffer
        
        Parameters:
        -----------
        capacity : int
            Maximum capacity of the buffer
        """
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state, action, reward, next_state, done):
        """
        Add a transition to the buffer
        
        Parameters:
        -----------
        state : numpy.ndarray
            Current state
        action : float
            Action taken
        reward : float
            Reward received
        next_state : numpy.ndarray
            Next state
        done : bool
            Whether the episode is done
        """
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """
        Sample a batch of transitions from the buffer
        
        Parameters:
        -----------
        batch_size : int
            Size of the batch to sample
        
        Returns:
        --------
        tuple
            (states, actions, rewards, next_states, dones)
        """
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, next_states, dones = zip(*[self.buffer[i] for i in indices])
        
        return (np.array(states), np.array(actions), 
                np.array(rewards), np.array(next_states), 
                np.array(dones))
    
    def __len__(self):
        """
        Get the current size of the buffer
        
        Returns:
        --------
        int
            Current size of the buffer
        """
        return len(self.buffer)

class RLQHAgent:
    """
    Reinforcement Learning agent for Quadratic Hedging (RL-QH)
    Based on the Deep Deterministic Policy Gradient algorithm variant for K-function
    """
    
    def __init__(self, state_dim, action_dim, action_bounds, gamma=1.0):
        """
        Initialize the RL-QH agent
        
        Parameters:
        -----------
        state_dim : int
            Dimension of the state space
        action_dim : int
            Dimension of the action space
        action_bounds : tuple
            (min_action, max_action)
        gamma : float
            Discount factor
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_bounds = action_bounds
        self.gamma = gamma
        
        # Actor (policy) network
        self.actor = self._build_actor()
        self.target_actor = self._build_actor()
        self.target_actor.set_weights(self.actor.get_weights())
        
        # Q network
        self.critic_q = self._build_critic()
        self.target_critic_q = self._build_critic()
        self.target_critic_q.set_weights(self.critic_q.get_weights())
        
        # K network (for second moment)
        self.critic_k = self._build_critic()
        self.target_critic_k = self._build_critic()
        self.target_critic_k.set_weights(self.critic_k.get_weights())
        
        # Learning rates
        self.actor_lr = 1e-4
        self.critic_lr = 1e-4
        
        # Optimizers
        self.actor_optimizer = Adam(self.actor_lr)
        self.critic_q_optimizer = Adam(self.critic_lr)
        self.critic_k_optimizer = Adam(self.critic_lr)
        
        # Noise for exploration
        self.noise_std = 0.1
        
        # Target network update rate
        self.tau = 1e-3
        
        # Create replay buffer
        self.buffer = ReplayBuffer(capacity=100000)
        self.batch_size = 64
        
        # Training info
        self.training_step = 0
    
    def _build_actor(self):
        """
        Build the actor (policy) network
        
        Returns:
        --------
        tensorflow.keras.Model
            Actor model
        """
        inputs = Input(shape=(self.state_dim,))
        
        # Batch normalization
        x = BatchNormalization()(inputs)
        
        # Hidden layers
        x = Dense(32, activation='relu')(x)
        x = BatchNormalization()(x)
        
        x = Dense(64, activation='relu')(x)
        x = BatchNormalization()(x)
        
        # Output layer with sigmoid activation to bound actions
        outputs = Dense(self.action_dim, activation='sigmoid')(x)
        
        # Scale outputs to action bounds
        min_action, max_action = self.action_bounds
        outputs = min_action + (max_action - min_action) * outputs
        
        return Model(inputs, outputs)
    
    def _build_critic(self):
        """
        Build the critic (Q/K) network
        
        Returns:
        --------
        tensorflow.keras.Model
            Critic model
        """
        # State input
        state_input = Input(shape=(self.state_dim,))
        state_x = BatchNormalization()(state_input)
        
        # Action input
        action_input = Input(shape=(self.action_dim,))
        
        # Merge state and action
        x = tf.concat([state_x, action_input], axis=1)
        
        # Hidden layers
        x = Dense(32, activation='relu')(x)
        x = BatchNormalization()(x)
        
        x = Dense(64, activation='relu')(x)
        x = BatchNormalization()(x)
        
        # Output layer (Q/K value)
        outputs = Dense(1, activation=None)(x)
        
        return Model([state_input, action_input], outputs)
    
    def act(self, state, add_noise=True):
        """
        Choose an action based on the current state
        
        Parameters:
        -----------
        state : numpy.ndarray
            Current state
        add_noise : bool
            Whether to add exploration noise
        
        Returns:
        --------
        numpy.ndarray
            Action
        """
        state = np.reshape(state, [1, self.state_dim])
        action = self.actor.predict(state)[0]
        
        if add_noise:
            noise = np.random.normal(0, self.noise_std, size=self.action_dim)
            action += noise
            
            # Clip action to bounds
            min_action, max_action = self.action_bounds
            action = np.clip(action, min_action, max_action)
        
        return action
    
    def remember(self, state, action, reward, next_state, done):
        """
        Store a transition in the replay buffer
        
        Parameters:
        -----------
        state : numpy.ndarray
            Current state
        action : numpy.ndarray
            Action taken
        reward : float
            Reward received
        next_state : numpy.ndarray
            Next state
        done : bool
            Whether the episode is done
        """
        self.buffer.add(state, action, reward, next_state, done)
    
    def learn(self):
        """
        Update networks using a batch of experiences from the replay buffer
        """
        # Check if enough samples in buffer
        if len(self.buffer) < self.batch_size:
            return
        
        # Sample a batch of transitions
        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
        
        # Reshape for network input
        states = np.reshape(states, [self.batch_size, self.state_dim])
        actions = np.reshape(actions, [self.batch_size, self.action_dim])
        rewards = np.reshape(rewards, [self.batch_size, 1])
        next_states = np.reshape(next_states, [self.batch_size, self.state_dim])
        dones = np.reshape(dones, [self.batch_size, 1])
        
        # Get target actions for next states
        target_actions = self.target_actor.predict(next_states)
        
        # Get target Q-values
        target_q_values = self.target_critic_q.predict([next_states, target_actions])
        
        # Compute target Q-values
        q_targets = rewards + self.gamma * target_q_values * (1 - dones)
        
        # Update Q-network
        with tf.GradientTape() as tape:
            q_values = self.critic_q([states, actions])
            q_loss = tf.reduce_mean(tf.square(q_targets - q_values))
        
        q_gradients = tape.gradient(q_loss, self.critic_q.trainable_variables)
        self.critic_q_optimizer.apply_gradients(zip(q_gradients, self.critic_q.trainable_variables))
        
        # Get target K-values
        target_k_values = self.target_critic_k.predict([next_states, target_actions])
        
        # Compute target K-values based on the K-function recursion
        k_targets = rewards**2 + 2 * self.gamma * rewards * target_q_values + self.gamma**2 * target_k_values * (1 - dones)
        
        # Update K-network
        with tf.GradientTape() as tape:
            k_values = self.critic_k([states, actions])
            k_loss = tf.reduce_mean(tf.square(k_targets - k_values))
        
        k_gradients = tape.gradient(k_loss, self.critic_k.trainable_variables)
        self.critic_k_optimizer.apply_gradients(zip(k_gradients, self.critic_k.trainable_variables))
        
        # Update policy (actor) network
        with tf.GradientTape() as tape:
            new_actions = self.actor(states)
            k_values = self.critic_k([states, new_actions])
            # Minimize K values (variance)
            actor_loss = tf.reduce_mean(k_values)
        
        actor_gradients = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_gradients, self.actor.trainable_variables))
        
        # Update target networks
        self._update_target_networks()
        
        # Increment training step
        self.training_step += 1
        
        return q_loss, k_loss, actor_loss
    
    def _update_target_networks(self):
        """
        Update target networks using soft update
        """
        # Update target actor
        actor_weights = self.actor.get_weights()
        target_actor_weights = self.target_actor.get_weights()
        for i in range(len(actor_weights)):
            target_actor_weights[i] = self.tau * actor_weights[i] + (1 - self.tau) * target_actor_weights[i]
        self.target_actor.set_weights(target_actor_weights)
        
        # Update target Q-critic
        critic_q_weights = self.critic_q.get_weights()
        target_critic_q_weights = self.target_critic_q.get_weights()
        for i in range(len(critic_q_weights)):
            target_critic_q_weights[i] = self.tau * critic_q_weights[i] + (1 - self.tau) * target_critic_q_weights[i]
        self.target_critic_q.set_weights(target_critic_q_weights)
        
        # Update target K-critic
        critic_k_weights = self.critic_k.get_weights()
        target_critic_k_weights = self.target_critic_k.get_weights()
        for i in range(len(critic_k_weights)):
            target_critic_k_weights[i] = self.tau * critic_k_weights[i] + (1 - self.tau) * target_critic_k_weights[i]
        self.target_critic_k.set_weights(target_critic_k_weights)
    
    def save_models(self, path_prefix):
        """
        Save all models to files
        
        Parameters:
        -----------
        path_prefix : str
            Prefix for model file paths
        """
        self.actor.save(f"{path_prefix}_actor.h5")
        self.critic_q.save(f"{path_prefix}_critic_q.h5")
        self.critic_k.save(f"{path_prefix}_critic_k.h5")
    
    def load_models(self, path_prefix):
        """
        Load all models from files
        
        Parameters:
        -----------
        path_prefix : str
            Prefix for model file paths
        """
        self.actor = tf.keras.models.load_model(f"{path_prefix}_actor.h5")
        self.target_actor = tf.keras.models.load_model(f"{path_prefix}_actor.h5")
        
        self.critic_q = tf.keras.models.load_model(f"{path_prefix}_critic_q.h5")
        self.target_critic_q = tf.keras.models.load_model(f"{path_prefix}_critic_q.h5")
        
        self.critic_k = tf.keras.models.load_model(f"{path_prefix}_critic_k.h5")
        self.target_critic_k = tf.keras.models.load_model(f"{path_prefix}_critic_k.h5")

class DeepQHAgent:
    """
    Deep Trajectory-based Stochastic Optimal Control agent for Quadratic Hedging (Deep-QH)
    """
    
    def __init__(self, state_dim, action_dim, action_bounds):
        """
        Initialize the Deep-QH agent
        
        Parameters:
        -----------
        state_dim : int
            Dimension of the state space
        action_dim : int
            Dimension of the action space
        action_bounds : tuple
            (min_action, max_action)
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_bounds = action_bounds
        
        # Build policy networks for each time step
        self.networks = []
        
        # Learning rate
        self.learning_rate = 1e-3
    
    def _build_network(self):
        """
        Build a policy network for a single time step
        
        Returns:
        --------
        tensorflow.keras.Model
            Policy model
        """
        model = Sequential([
            BatchNormalization(input_shape=(self.state_dim,)),
            Dense(10, activation='relu'),
            BatchNormalization(),
            Dropout(0.25),
            Dense(15, activation='relu'),
            BatchNormalization(),
            Dropout(0.25),
            Dense(10, activation='relu'),
            BatchNormalization(),
            Dropout(0.25),
            Dense(self.action_dim, activation='sigmoid')
        ])
        
        # Scale output to action bounds
        min_action, max_action = self.action_bounds
        
        # Custom scaling layer
        class ScaleLayer(tf.keras.layers.Layer):
            def __init__(self, min_val, max_val):
                super(ScaleLayer, self).__init__()
                self.min_val = min_val
                self.max_val = max_val
            
            def call(self, inputs):
                return self.min_val + (self.max_val - self.min_val) * inputs
        
        # Add scaling layer
        scaled_model = Sequential([
            model,
            ScaleLayer(min_action, max_action)
        ])
        
        scaled_model.compile(optimizer=Adam(self.learning_rate), loss='mse')
        
        return scaled_model
    
    def initialize(self, n_steps):
        """
        Initialize networks for all time steps
        
        Parameters:
        -----------
        n_steps : int
            Number of time steps
        """
        self.networks = [self._build_network() for _ in range(n_steps)]
    
    def act(self, state, step_idx):
        """
        Choose an action based on the current state and time step
        
        Parameters:
        -----------
        state : numpy.ndarray
            Current state
        step_idx : int
            Current time step index
        
        Returns:
        --------
        numpy.ndarray
            Action
        """
        state = np.reshape(state, [1, self.state_dim])
        action = self.networks[step_idx].predict(state)[0]
        
        return action
    
    def train(self, env, n_episodes=1000, batch_size=64):
        """
        Train the Deep-QH agent using the entire trajectory approach
        
        Parameters:
        -----------
        env : Environment
            The option hedging environment
        n_episodes : int
            Number of training episodes
        batch_size : int
            Batch size for training
        
        Returns:
        --------
        list
            Training losses
        """
        # Initialize networks if not already done
        if not self.networks:
            self.initialize(env.n_steps)
        
        # Training losses
        losses = []
        
        # Training loop
        for episode in tqdm(range(n_episodes)):
            # Generate batch of trajectories
            batch_pnls = []
            batch_states = [[] for _ in range(env.n_steps)]
            batch_actions = [[] for _ in range(env.n_steps)]
            
            for _ in range(batch_size):
                # Reset environment
                state = env.reset()
                done = False
                step_idx = 0
                trajectory_states = []
                trajectory_actions = []
                
                # Generate a trajectory
                while not done:
                    # Choose action
                    action = self.act(state, step_idx)
                    
                    # Store state and action
                    trajectory_states.append(state)
                    trajectory_actions.append(action)
                    
                    # Take action in environment
                    next_state, reward, done, info = env.step(action)
                    
                    # Move to next state
                    state = next_state
                    step_idx += 1
                
                # Store PnL and trajectory in batch
                batch_pnls.append(info["pnl"])
                
                # Store states and actions by time step
                for i, (s, a) in enumerate(zip(trajectory_states, trajectory_actions)):
                    batch_states[i].append(s)
                    batch_actions[i].append(a)
            
            # Train networks for each time step
            batch_loss = 0
            
            # Backward training (from last time step to first)
            for t in range(env.n_steps - 1, -1, -1):
                # Stack states and actions for this time step
                X = np.array(batch_states[t])
                y = np.array(batch_actions[t])
                
                # Train network for this time step
                loss = self.networks[t].train_on_batch(X, y)
                batch_loss += loss
            
            # Store average loss
            losses.append(batch_loss / env.n_steps)
            
            # Adaptive learning rate
            if episode > 0 and episode % 1000 == 0:
                self.learning_rate *= 0.9
                for network in self.networks:
                    network.optimizer.learning_rate.assign(self.learning_rate)
        
        return losses
    
    def save_models(self, path_prefix):
        """
        Save all models to files
        
        Parameters:
        -----------
        path_prefix : str
            Prefix for model file paths
        """
        for i, network in enumerate(self.networks):
            network.save(f"{path_prefix}_step_{i}.h5")
    
    def load_models(self, path_prefix, n_steps):
        """
        Load all models from files
        
        Parameters:
        -----------
        path_prefix : str
            Prefix for model file paths
        n_steps : int
            Number of time steps
        """
        self.networks = []
        for i in range(n_steps):
            self.networks.append(tf.keras.models.load_model(f"{path_prefix}_step_{i}.h5"))

def train_rl_qh_agent(env, n_episodes=10000):
    """
    Train an RL-QH agent
    
    Parameters:
    -----------
    env : Environment
        The option hedging environment
    n_episodes : int
        Number of training episodes
    
    Returns:
    --------
    RLQHAgent
        Trained agent
    """
    # Initialize agent
    state_dim = 3  # [time_to_maturity, normalized_stock_price, stock_holdings]
    action_dim = 1  # stock_holdings
    action_bounds = (0, 1)  # Range of possible hedge ratios
    
    agent = RLQHAgent(state_dim, action_dim, action_bounds)
    
    # Training loop
    for episode in tqdm(range(n_episodes)):
        # Reset environment
        state = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            # Choose action
            action = agent.act(state)
            
            # Take action in environment
            next_state, reward, done, _ = env.step(action)
            
            # Store transition in replay buffer
            agent.remember(state, action, reward, next_state, done)
            
            # Learn from experiences
            if len(agent.buffer) >= agent.batch_size:
                agent.learn()
            
            # Move to next state
            state = next_state
            episode_reward += reward
        
        # Decrease exploration noise over time
        if episode % 100 == 0 and agent.noise_std > 0.01:
            agent.noise_std *= 0.99
    
    return agent

def train_deep_qh_agent(env, n_episodes=10000, batch_size=64):
    """
    Train a Deep-QH agent
    
    Parameters:
    -----------
    env : Environment
        The option hedging environment
    n_episodes : int
        Number of training episodes
    batch_size : int
        Batch size for training
    
    Returns:
    --------
    DeepQHAgent
        Trained agent
    """
    # Initialize agent
    state_dim = 3  # [time_to_maturity, normalized_stock_price, stock_holdings]
    action_dim = 1  # stock_holdings
    action_bounds = (0, 1)  # Range of possible hedge ratios
    
    agent = DeepQHAgent(state_dim, action_dim, action_bounds)
    
    # Train agent
    losses = agent.train(env, n_episodes, batch_size)
    
    # Plot training losses
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.title('Training Loss')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.grid(True)
    plt.show()
    
    return agent

def evaluate_agent(env, agent, n_episodes=1000, agent_type='rl_qh'):
    """
    Evaluate an agent's performance
    
    Parameters:
    -----------
    env : Environment
        The option hedging environment
    agent : Agent
        The agent to evaluate
    n_episodes : int
        Number of evaluation episodes
    agent_type : str
        Type of agent ('rl_qh', 'deep_qh', or 'delta')
    
    Returns:
    --------
    tuple
        (pnls, portfolio_paths, stock_paths)
    """
    # Store PnLs and paths
    pnls = []
    portfolio_paths = []
    stock_paths = []
    holdings_paths = []
    
    # Evaluation loop
    for _ in tqdm(range(n_episodes)):
        # Reset environment
        state = env.reset()
        done = False
        step_idx = 0
        
        while not done:
            # Choose action based on agent type
            if agent_type == 'rl_qh':
                action = agent.act(state, add_noise=False)
            elif agent_type == 'deep_qh':
                action = agent.act(state, step_idx)
            elif agent_type == 'delta':
                action = agent.act(state)
            
            # Take action in environment
            next_state, _, done, info = env.step(action)
            
            # Move to next state
            state = next_state
            step_idx += 1
        
        # Store PnL and paths
        pnls.append(info["pnl"])
        portfolio_paths.append(env.portfolio_values.copy())
        stock_paths.append(env.price_path.copy())
        holdings_paths.append(env.holdings_path.copy())
    
    return pnls, portfolio_paths, stock_paths, holdings_paths

def compare_agents(pnls_dict, title="Hedging Error Comparison"):
    """
    Compare hedging errors of different agents
    
    Parameters:
    -----------
    pnls_dict : dict
        Dictionary mapping agent names to lists of PnLs
    title : str
        Plot title
    """
    plt.figure(figsize=(12, 8))
    
    for agent_name, pnls in pnls_dict.items():
        sns.histplot(pnls, label=agent_name, kde=True, alpha=0.5)
    
    plt.axvline(x=0, color='black', linestyle='--')
    plt.title(title)
    plt.xlabel('Hedging Error (Positive = Loss, Negative = Profit)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True)
    
    # Print statistics
    print("\nHedging Error Statistics:")
    for agent_name, pnls in pnls_dict.items():
        print(f"{agent_name}:")
        print(f"  Mean: {np.mean(pnls):.4f}")
        print(f"  Std Dev: {np.std(pnls):.4f}")
        print(f"  Min: {np.min(pnls):.4f}")
        print(f"  Max: {np.max(pnls):.4f}")
        print(f"  MSE: {np.mean(np.square(pnls)):.4f}")
    
    plt.show()

def plot_sample_paths(stock_paths, portfolio_paths, holdings_paths, agent_names, n_samples=3):
    """
    Plot sample paths of stock price, portfolio value, and holdings
    
    Parameters:
    -----------
    stock_paths : list
        List of stock price paths for each agent
    portfolio_paths : list
        List of portfolio value paths for each agent
    holdings_paths : list
        List of holdings paths for each agent
    agent_names : list
        List of agent names
    n_samples : int
        Number of sample paths to plot
    """
    # Choose random samples
    sample_indices = np.random.choice(len(stock_paths[0]), n_samples, replace=False)
    
    # Plot one sample at a time
    for idx in sample_indices:
        plt.figure(figsize=(15, 12))
        
        # Plot stock price
        plt.subplot(3, 1, 1)
        plt.plot(stock_paths[0][idx], label='Stock Price', color='black')
        plt.title(f'Sample Path {idx+1}')
        plt.ylabel('Stock Price')
        plt.grid(True)
        plt.legend()
        
        # Plot portfolio values
        plt.subplot(3, 1, 2)
        for i, agent_name in enumerate(agent_names):
            plt.plot(portfolio_paths[i][idx], label=f'{agent_name} Portfolio')
        plt.ylabel('Portfolio Value')
        plt.grid(True)
        plt.legend()
        
        # Plot holdings
        plt.subplot(3, 1, 3)
        for i, agent_name in enumerate(agent_names):
            plt.plot(holdings_paths[i][idx], label=f'{agent_name} Holdings')
        plt.xlabel('Time Step')
        plt.ylabel('Stock Holdings')
        plt.grid(True)
        plt.legend()
        
        plt.tight_layout()
        plt.show()

# Main function to run the experiments
def main():
    # Parameters
    n_train_episodes = 10000  # Number of training episodes
    n_eval_episodes = 1000   # Number of evaluation episodes
    
    # 1. Black-Scholes without transaction costs
    print("\n=== Experiment 1: Black-Scholes without transaction costs ===")
    
    # Create environment
    bs_env = BlackScholesEnvironment(S0=100, K=100, T=30, r=0, sigma=0.2, mu=0.05, alpha=0)
    
    # Train agents
    print("Training RL-QH agent...")
    rl_qh_agent = train_rl_qh_agent(bs_env, n_train_episodes)
    
    print("Training Deep-QH agent...")
    deep_qh_agent = train_deep_qh_agent(bs_env, n_train_episodes)
    
    # Create Delta hedging agent
    delta_agent = DeltaHedgeAgent(bs_env)
    
    # Evaluate agents
    print("Evaluating agents...")
    rl_qh_pnls, rl_qh_portfolios, rl_qh_stocks, rl_qh_holdings = evaluate_agent(
        bs_env, rl_qh_agent, n_eval_episodes, 'rl_qh')
    
    deep_qh_pnls, deep_qh_portfolios, deep_qh_stocks, deep_qh_holdings = evaluate_agent(
        bs_env, deep_qh_agent, n_eval_episodes, 'deep_qh')
    
    delta_pnls, delta_portfolios, delta_stocks, delta_holdings = evaluate_agent(
        bs_env, delta_agent, n_eval_episodes, 'delta')
    
    # Compare results
    pnls_dict = {
        'RL-QH': rl_qh_pnls,
        'Deep-QH': deep_qh_pnls,
        'Delta Hedge': delta_pnls
    }
    compare_agents(pnls_dict, "Hedging Error - Black-Scholes without Transaction Costs")
    
    # Plot sample paths
    plot_sample_paths(
        [rl_qh_stocks, deep_qh_stocks, delta_stocks],
        [rl_qh_portfolios, deep_qh_portfolios, delta_portfolios],
        [rl_qh_holdings, deep_qh_holdings, delta_holdings],
        ['RL-QH', 'Deep-QH', 'Delta Hedge']
    )
    
    # 2. Black-Scholes with transaction costs
    print("\n=== Experiment 2: Black-Scholes with transaction costs ===")
    
    # Create environment
    bs_tc_env = BlackScholesEnvironment(S0=100, K=100, T=30, r=0, sigma=0.2, mu=0.05, alpha=0.001)
    
    # Train agents
    print("Training RL-QH agent with transaction costs...")
    rl_qh_tc_agent = train_rl_qh_agent(bs_tc_env, n_train_episodes)
    
    print("Training Deep-QH agent with transaction costs...")
    deep_qh_tc_agent = train_deep_qh_agent(bs_tc_env, n_train_episodes)
    
    # Create Delta hedging agent
    delta_tc_agent = DeltaHedgeAgent(bs_tc_env)
    
    # Evaluate agents
    print("Evaluating agents...")
    rl_qh_tc_pnls, rl_qh_tc_portfolios, rl_qh_tc_stocks, rl_qh_tc_holdings = evaluate_agent(
        bs_tc_env, rl_qh_tc_agent, n_eval_episodes, 'rl_qh')
    
    deep_qh_tc_pnls, deep_qh_tc_portfolios, deep_qh_tc_stocks, deep_qh_tc_holdings = evaluate_agent(
        bs_tc_env, deep_qh_tc_agent, n_eval_episodes, 'deep_qh')
    
    delta_tc_pnls, delta_tc_portfolios, delta_tc_stocks, delta_tc_holdings = evaluate_agent(
        bs_tc_env, delta_tc_agent, n_eval_episodes, 'delta')
    
    # Compare results
    pnls_dict_tc = {
        'RL-QH': rl_qh_tc_pnls,
        'Deep-QH': deep_qh_tc_pnls,
        'Delta Hedge': delta_tc_pnls
    }
    compare_agents(pnls_dict_tc, "Hedging Error - Black-Scholes with Transaction Costs")
    
    # Plot sample paths
    plot_sample_paths(
        [rl_qh_tc_stocks, deep_qh_tc_stocks, delta_tc_stocks],
        [rl_qh_tc_portfolios, deep_qh_tc_portfolios, delta_tc_portfolios],
        [rl_qh_tc_holdings, deep_qh_tc_holdings, delta_tc_holdings],
        ['RL-QH', 'Deep-QH', 'Delta Hedge']
    )
    
    # 3. SABR model without transaction costs
    print("\n=== Experiment 3: SABR model without transaction costs ===")
    
    # Create environment
    sabr_env = SABREnvironment(S0=100, K=100, T=30, r=0, sigma0=0.2, mu=0.05, alpha=0, 
                               eta=0.0095, rho=0.5)
    
    # Train agents
    print("Training RL-QH agent for SABR...")
    rl_qh_sabr_agent = train_rl_qh_agent(sabr_env, n_train_episodes)
    
    print("Training Deep-QH agent for SABR...")
    deep_qh_sabr_agent = train_deep_qh_agent(sabr_env, n_train_episodes)
    
    # Create Bartlett's Delta hedging agent
    bartlett_agent = DeltaHedgeAgent(sabr_env)
    
    # Evaluate agents
    print("Evaluating agents...")
    rl_qh_sabr_pnls, rl_qh_sabr_portfolios, rl_qh_sabr_stocks, rl_qh_sabr_holdings = evaluate_agent(
        sabr_env, rl_qh_sabr_agent, n_eval_episodes, 'rl_qh')
    
    deep_qh_sabr_pnls, deep_qh_sabr_portfolios, deep_qh_sabr_stocks, deep_qh_sabr_holdings = evaluate_agent(
        sabr_env, deep_qh_sabr_agent, n_eval_episodes, 'deep_qh')
    
    bartlett_pnls, bartlett_portfolios, bartlett_stocks, bartlett_holdings = evaluate_agent(
        sabr_env, bartlett_agent, n_eval_episodes, 'delta')
    
    # Compare results
    pnls_dict_sabr = {
        'RL-QH': rl_qh_sabr_pnls,
        'Deep-QH': deep_qh_sabr_pnls,
        'Bartlett Delta': bartlett_pnls
    }
    compare_agents(pnls_dict_sabr, "Hedging Error - SABR without Transaction Costs")
    
    # Plot sample paths
    plot_sample_paths(
        [rl_qh_sabr_stocks, deep_qh_sabr_stocks, bartlett_stocks],
        [rl_qh_sabr_portfolios, deep_qh_sabr_portfolios, bartlett_portfolios],
        [rl_qh_sabr_holdings, deep_qh_sabr_holdings, bartlett_holdings],
        ['RL-QH', 'Deep-QH', 'Bartlett Delta']
    )
    
    # 4. Testing agents trained on Black-Scholes but evaluated on SABR
    print("\n=== Experiment 4: Testing Black-Scholes agents on SABR model ===")
    
    # Evaluate Black-Scholes trained agents on SABR environment
    print("Evaluating agents...")
    rl_qh_bs_on_sabr_pnls, rl_qh_bs_on_sabr_portfolios, rl_qh_bs_on_sabr_stocks, rl_qh_bs_on_sabr_holdings = evaluate_agent(
        sabr_env, rl_qh_agent, n_eval_episodes, 'rl_qh')
    
    deep_qh_bs_on_sabr_pnls, deep_qh_bs_on_sabr_portfolios, deep_qh_bs_on_sabr_stocks, deep_qh_bs_on_sabr_holdings = evaluate_agent(
        sabr_env, deep_qh_agent, n_eval_episodes, 'deep_qh')
    
    # Compare results
    pnls_dict_bs_on_sabr = {
        'RL-QH (BS trained)': rl_qh_bs_on_sabr_pnls,
        'Deep-QH (BS trained)': deep_qh_bs_on_sabr_pnls,
        'Bartlett Delta': bartlett_pnls
    }
    compare_agents(pnls_dict_bs_on_sabr, "Hedging Error - Black-Scholes Agents on SABR Model")
    
    # Plot sample paths
    plot_sample_paths(
        [rl_qh_bs_on_sabr_stocks, deep_qh_bs_on_sabr_stocks, bartlett_stocks],
        [rl_qh_bs_on_sabr_portfolios, deep_qh_bs_on_sabr_portfolios, bartlett_portfolios],
        [rl_qh_bs_on_sabr_holdings, deep_qh_bs_on_sabr_holdings, bartlett_holdings],
        ['RL-QH (BS trained)', 'Deep-QH (BS trained)', 'Bartlett Delta']
    )

if __name__ == "__main__":
    main()