import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.ensemble import ExtraTreesRegressor
from tqdm import tqdm

# Set random seed for reproducibility
np.random.seed(42)

class HestonModel:
    """
    Implementation of the Heston stochastic volatility model
    """
    def __init__(self, kappa=2, theta=0.04, sigma=0.3, rho=-0.7, r=0.0, initial_price=100, initial_vol=0.04):
        """
        Initialize Heston model parameters
        
        Parameters:
        -----------
        kappa : float
            Rate of mean reversion
        theta : float
            Long-run mean of variance
        sigma : float
            Volatility of variance
        rho : float
            Correlation between Brownian motions
        r : float
            Risk-free interest rate
        initial_price : float
            Initial stock price
        initial_vol : float
            Initial volatility
        """
        self.kappa = kappa        # Mean reversion rate
        self.theta = theta        # Long-run mean of variance
        self.sigma = sigma        # Volatility of variance
        self.rho = rho            # Correlation between price and volatility
        self.r = r                # Risk-free rate
        self.initial_price = initial_price
        self.initial_vol = initial_vol
        
    def simulate(self, T=1.0, N=252, paths=1, mean_reverting=False, reversion_level=None, reversion_speed=None):
        """
        Simulate stock price paths using the Heston model
        
        Parameters:
        -----------
        T : float
            Time horizon in years
        N : int
            Number of time steps
        paths : int
            Number of paths to simulate
        mean_reverting : bool
            Whether to add mean reversion to the stock price (for arbitrage opportunity)
        reversion_level : float
            Level to which stock price reverts (if mean_reverting is True)
        reversion_speed : float
            Speed at which stock price reverts to reversion_level (if mean_reverting is True)
            
        Returns:
        --------
        S : ndarray
            Simulated stock price paths of shape (paths, N+1)
        V : ndarray
            Simulated variance paths of shape (paths, N+1)
        """
        dt = T / N
        
        # Initialize arrays for stock prices and volatilities
        S = np.zeros((paths, N + 1))
        V = np.zeros((paths, N + 1))
        
        # Set initial values
        S[:, 0] = self.initial_price
        V[:, 0] = self.initial_vol
        
        # Generate correlated random numbers
        z1 = np.random.normal(0, 1, size=(paths, N))
        z2 = self.rho * z1 + np.sqrt(1 - self.rho**2) * np.random.normal(0, 1, size=(paths, N))
        
        # Simulate paths
        for t in range(N):
            # Ensure volatility is positive
            V[:, t] = np.maximum(V[:, t], 0)
            
            # Simulate variance (volatility squared)
            V[:, t+1] = V[:, t] + self.kappa * (self.theta - V[:, t]) * dt + self.sigma * np.sqrt(V[:, t] * dt) * z2[:, t]
            
            # Drift term
            drift = self.r
            
            # Add mean reversion if specified (to create arbitrage opportunity)
            if mean_reverting and reversion_level is not None and reversion_speed is not None:
                drift += reversion_speed * (reversion_level - S[:, t]) / S[:, t]
            
            # Simulate stock price
            S[:, t+1] = S[:, t] * np.exp((drift - 0.5 * V[:, t]) * dt + np.sqrt(V[:, t] * dt) * z1[:, t])
        
        return S, V

    def fit_to_data(self, prices, method='calibration'):
        """
        Fit Heston model parameters to price data using a simple calibration approach
        
        Parameters:
        -----------
        prices : ndarray
            Historical price data
        method : str
            Method to use for fitting ('calibration' or 'ekf')
            
        Returns:
        --------
        dict
            Fitted parameters
        """
        # Calculate log returns
        log_returns = np.diff(np.log(prices))
        
        # Estimate initial variance as squared volatility of returns
        initial_vol = np.var(log_returns)
        
        # Simple calibration for illustration purposes
        if method == 'calibration':
            # Estimate long-run variance
            theta = initial_vol
            
            # Estimate volatility of variance using standard deviation of squared returns
            squared_returns = log_returns**2
            sigma = np.std(squared_returns) * np.sqrt(252)
            
            # Estimate mean reversion rate
            # Use autocorrelation of squared returns as a proxy
            kappa = 2.0  # Default value
            if len(squared_returns) > 10:
                acf_1 = np.corrcoef(squared_returns[:-1], squared_returns[1:])[0, 1]
                if acf_1 < 1 and acf_1 > 0:
                    kappa = -np.log(acf_1) * 252
            
            # Estimate correlation between returns and changes in variance
            var_changes = np.diff(squared_returns)
            if len(var_changes) > 1:
                try:
                    rho = np.corrcoef(log_returns[1:], var_changes)[0, 1]
                except:
                    rho = -0.7  # Default value
            else:
                rho = -0.7
            
            # Update model parameters
            self.kappa = max(0.1, min(kappa, 10.0))  # Constrain to reasonable values
            self.theta = max(0.001, min(theta, 0.2))
            self.sigma = max(0.05, min(sigma, 2.0))
            self.rho = max(-1.0, min(rho, 0.0))
            self.initial_vol = initial_vol
        
        # In a real implementation, the EKF method would be here
        elif method == 'ekf':
            # The paper uses Extended Kalman Filter, which is more complex
            # and requires implementation of the algorithm described in the paper
            print("Extended Kalman Filter method not implemented. Using default parameters.")
        
        return {
            'kappa': self.kappa,
            'theta': self.theta,
            'sigma': self.sigma,
            'rho': self.rho,
            'initial_vol': self.initial_vol
        }


class TradingEnvironment:
    """
    Trading environment for reinforcement learning
    """
    def __init__(self, prices, volatilities=None, lot_size=100, tick_size=0.01, 
                 max_position=10, max_trade_size=5, risk_aversion=1e-4):
        """
        Initialize trading environment
        
        Parameters:
        -----------
        prices : ndarray
            Asset prices
        volatilities : ndarray, optional
            Asset volatilities (optional)
        lot_size : int
            Number of shares in one round lot
        tick_size : float
            Minimum price movement
        max_position : int
            Maximum position in round lots (positive or negative)
        max_trade_size : int
            Maximum trade size in round lots (positive or negative)
        risk_aversion : float
            Risk aversion parameter for reward function
        """
        self.prices = prices
        self.volatilities = volatilities
        self.lot_size = lot_size
        self.tick_size = tick_size
        self.max_position = max_position
        self.max_trade_size = max_trade_size
        self.risk_aversion = risk_aversion
        
        self.time = 0
        self.position = 0
        self.done = False
        self.states = []
        self.actions = []
        self.rewards = []
        self.pnl = []
        
    def reset(self):
        """
        Reset the environment to initial state
        
        Returns:
        --------
        state : tuple
            Initial state (position, price, volatility)
        """
        self.time = 0
        self.position = 0
        self.done = False
        self.states = []
        self.actions = []
        self.rewards = []
        self.pnl = []
        
        state = self._get_state()
        self.states.append(state)
        return state
    
    def _get_state(self):
        """
        Get current state of the environment
        
        Returns:
        --------
        state : tuple
            Current state (position, price, volatility)
        """
        if self.volatilities is not None:
            return (self.position, self.prices[self.time], self.volatilities[self.time])
        else:
            return (self.position, self.prices[self.time])
    
    def step(self, action):
        """
        Take a trading action and compute the next state and reward
        
        Parameters:
        -----------
        action : int
            Trading action in round lots
            
        Returns:
        --------
        next_state : tuple
            Next state
        reward : float
            Reward for the action
        done : bool
            Whether the episode is done
        info : dict
            Additional information
        """
        # Clip action to allowed range
        action = max(-self.max_trade_size, min(action, self.max_trade_size))
        
        # Ensure position limits are respected
        action = max(-self.max_position - self.position, min(action, self.max_position - self.position))
        
        # Record action
        self.actions.append(action)
        
        # Calculate transaction costs
        spread_cost = self.tick_size * abs(action) * self.lot_size
        impact_cost = (action**2) * (self.tick_size / self.lot_size) * self.lot_size
        transaction_cost = spread_cost + impact_cost
        
        # Move to next time step
        self.time += 1
        
        # Check if episode is done
        if self.time >= len(self.prices) - 1:
            self.done = True
            
        # Calculate price change and P&L
        price_change = self.prices[self.time] - self.prices[self.time - 1]
        position_pnl = self.position * price_change * self.lot_size
        pnl = position_pnl - transaction_cost
        
        # Update position
        self.position += action
        
        # Record P&L
        self.pnl.append(pnl)
        
        # Calculate reward using mean-variance utility
        reward = pnl - 0.5 * self.risk_aversion * (pnl**2)
        self.rewards.append(reward)
        
        # Get new state
        next_state = self._get_state()
        self.states.append(next_state)
        
        return next_state, reward, self.done, {"pnl": pnl}
    
    def render(self):
        """
        Render the environment (for visualization)
        """
        plt.figure(figsize=(15, 10))
        
        # Plot price
        plt.subplot(3, 1, 1)
        plt.plot(self.prices[:self.time+1])
        plt.title('Asset Price')
        plt.grid(True)
        
        # Plot position
        plt.subplot(3, 1, 2)
        positions = [state[0] for state in self.states]
        plt.plot(positions)
        plt.title('Position (in round lots)')
        plt.grid(True)
        
        # Plot cumulative P&L
        plt.subplot(3, 1, 3)
        cumulative_pnl = np.cumsum(self.pnl)
        plt.plot(cumulative_pnl)
        plt.title('Cumulative P&L')
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()


class FittedQAgent:
    """
    Reinforcement learning agent using Fitted Q Iteration
    """
    def __init__(self, state_dim, action_space, gamma=0.999, n_estimators=10,
                 min_samples_split=5, min_samples_leaf=5, verbose=False):
        """
        Initialize the Fitted Q agent
        
        Parameters:
        -----------
        state_dim : int
            Dimension of the state space
        action_space : list
            List of possible actions
        gamma : float
            Discount factor
        n_estimators : int
            Number of trees in the extra trees regressor
        min_samples_split : int
            Minimum number of samples required to split a node
        min_samples_leaf : int
            Minimum number of samples required at a leaf node
        verbose : bool
            Whether to print progress information
        """
        self.state_dim = state_dim
        self.action_space = action_space
        self.gamma = gamma
        self.n_estimators = n_estimators
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.verbose = verbose
        
        # Initialize Q-function approximator (Extra Trees Regressor)
        self.Q = None
    
    def _preprocess_state(self, state):
        """
        Preprocess state for input to Q-function approximator
        
        Parameters:
        -----------
        state : tuple
            Environment state
            
        Returns:
        --------
        processed_state : ndarray
            Processed state
        """
        # Convert to numpy array and ensure correct shape
        return np.array(state).reshape(1, -1)
    
    def select_action(self, state, epsilon=0.0):
        """
        Select action using epsilon-greedy policy
        
        Parameters:
        -----------
        state : tuple
            Environment state
        epsilon : float
            Exploration probability
            
        Returns:
        --------
        action : int
            Selected action
        """
        if self.Q is None or np.random.random() < epsilon:
            # Random action if Q-function is not trained or for exploration
            return np.random.choice(self.action_space)
        else:
            # Greedy action selection
            processed_state = self._preprocess_state(state)
            
            # Evaluate Q-values for all actions
            q_values = []
            for a in self.action_space:
                # Create input for Q-function
                X = np.hstack([processed_state, np.array([[a]])])
                # Predict Q-value
                q_value = self.Q.predict(X)[0]
                q_values.append(q_value)
            
            # Select action with highest Q-value
            return self.action_space[np.argmax(q_values)]
    
    def fitted_q_iteration(self, dataset, n_iterations=10):
        """
        Train Q-function approximator using Fitted Q Iteration
        
        Parameters:
        -----------
        dataset : list
            List of tuples (state, action, reward, next_state)
        n_iterations : int
            Number of iterations to perform
            
        Returns:
        --------
        None
        """
        if len(dataset) == 0:
            print("Empty dataset, cannot train Q-function")
            return
        
        # Extract states, actions, rewards, and next_states from dataset
        states, actions, rewards, next_states = zip(*dataset)
        
        # Convert to numpy arrays
        states = np.array(states)
        actions = np.array(actions).reshape(-1, 1)
        rewards = np.array(rewards)
        next_states = np.array(next_states)
        
        # Initialize Q-function as zero
        self.Q = ExtraTreesRegressor(
            n_estimators=self.n_estimators,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf
        )
        
        # Initial training target is just the rewards
        X = np.hstack([states, actions])
        y = rewards
        
        # Fit initial Q-function
        self.Q.fit(X, y)
        
        # Iteratively improve Q-function
        for iteration in range(n_iterations):
            if self.verbose and iteration % 10 == 0:
                print(f"Iteration {iteration}/{n_iterations}")
            
            # Compute new target values
            max_q_values = np.zeros(len(dataset))
            
            for i, next_state in enumerate(next_states):
                q_values = []
                for a in self.action_space:
                    # Create input for Q-function
                    X_next = np.hstack([next_state.reshape(1, -1), np.array([[a]])])
                    # Predict Q-value
                    q_value = self.Q.predict(X_next)[0]
                    q_values.append(q_value)
                max_q_values[i] = np.max(q_values)
            
            # Update targets using Bellman equation
            y = rewards + self.gamma * max_q_values
            
            # Fit Q-function to new targets
            self.Q.fit(X, y)
    
    def train(self, env, n_episodes=1000, max_steps=1000, epsilon=0.1, n_iterations=10):
        """
        Train the agent on the environment
        
        Parameters:
        -----------
        env : TradingEnvironment
            Trading environment
        n_episodes : int
            Number of episodes to train
        max_steps : int
            Maximum number of steps per episode
        epsilon : float
            Exploration probability
        n_iterations : int
            Number of fitted Q iterations
            
        Returns:
        --------
        dataset : list
            Dataset of (state, action, reward, next_state) tuples
        """
        dataset = []
        
        for episode in tqdm(range(n_episodes), desc="Training episodes"):
            state = env.reset()
            
            for step in range(max_steps):
                action = self.select_action(state, epsilon)
                next_state, reward, done, _ = env.step(action)
                
                # Add transition to dataset
                dataset.append((np.array(state), action, reward, np.array(next_state)))
                
                state = next_state
                
                if done:
                    break
        
        # Train Q-function using fitted Q iteration
        self.fitted_q_iteration(dataset, n_iterations)
        
        return dataset


def evaluate_agent(agent, env, n_episodes=10, render_final=False):
    """
    Evaluate agent performance on the environment
    
    Parameters:
    -----------
    agent : FittedQAgent
        Trained agent
    env : TradingEnvironment
        Trading environment
    n_episodes : int
        Number of episodes to evaluate
    render_final : bool
        Whether to render the final episode
        
    Returns:
    --------
    results : dict
        Dictionary containing evaluation results
    """
    total_returns = []
    sharpe_ratios = []
    
    for episode in range(n_episodes):
        state = env.reset()
        episode_rewards = []
        
        while not env.done:
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            episode_rewards.append(info['pnl'])
            state = next_state
        
        # Calculate returns and Sharpe ratio
        returns = np.array(episode_rewards)
        total_return = np.sum(returns)
        sharpe_ratio = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252)
        
        total_returns.append(total_return)
        sharpe_ratios.append(sharpe_ratio)
        
        if render_final and episode == n_episodes - 1:
            env.render()
    
    # Calculate summary statistics
    mean_return = np.mean(total_returns)
    std_return = np.std(total_returns)
    mean_sharpe = np.mean(sharpe_ratios)
    std_sharpe = np.std(sharpe_ratios)
    
    print(f"Mean return: {mean_return:.4f} ± {std_return:.4f}")
    print(f"Mean Sharpe ratio: {mean_sharpe:.4f} ± {std_sharpe:.4f}")
    
    results = {
        'returns': total_returns,
        'sharpe_ratios': sharpe_ratios,
        'mean_return': mean_return,
        'std_return': std_return,
        'mean_sharpe': mean_sharpe,
        'std_sharpe': std_sharpe
    }
    
    return results


def run_experiment_with_arbitrage():
    """
    Run experiment in an environment that allows for arbitrage opportunity
    (similar to the first experiment in the paper)
    """
    print("Running experiment with arbitrage opportunity...")
    
    # Initialize Heston model with parameters
    heston = HestonModel(kappa=2.0, theta=0.04, sigma=0.3, rho=-0.7, r=0.0)
    
    # Set parameters for mean reversion (to create arbitrage opportunity)
    reversion_level = 100.0  # Stock price will revert to this level
    reversion_speed = 0.5    # Speed of mean reversion
    
    # Simulate stock prices with mean reversion
    train_prices, train_vols = heston.simulate(T=1.0, N=10000, paths=1, 
                                              mean_reverting=True, 
                                              reversion_level=reversion_level,
                                              reversion_speed=reversion_speed)
    
    # Extract first path
    train_prices = train_prices[0]
    train_vols = train_vols[0]
    
    # Set up trading environment
    lot_size = 100
    tick_size = 0.01
    max_position = 10
    max_trade_size = 5
    risk_aversion = 1e-4
    
    env = TradingEnvironment(
        prices=train_prices,
        volatilities=train_vols,
        lot_size=lot_size,
        tick_size=tick_size,
        max_position=max_position,
        max_trade_size=max_trade_size,
        risk_aversion=risk_aversion
    )
    
    # Define action space
    action_space = list(range(-max_trade_size, max_trade_size + 1))
    
    # Initialize agent
    state_dim = 3  # position, price, volatility
    agent = FittedQAgent(
        state_dim=state_dim,
        action_space=action_space,
        gamma=0.999,
        n_estimators=10,
        min_samples_split=5,
        min_samples_leaf=5,
        verbose=True
    )
    
    # Train agent
    dataset = agent.train(
        env=env,
        n_episodes=1,
        max_steps=9000,
        epsilon=0.1,
        n_iterations=100
    )
    
    print(f"Training complete. Dataset size: {len(dataset)}")
    
    # Generate test data
    test_paths = 10
    test_prices, test_vols = heston.simulate(T=1.0, N=1000, paths=test_paths, 
                                           mean_reverting=True, 
                                           reversion_level=reversion_level,
                                           reversion_speed=reversion_speed)
    
    # Evaluate agent on each test path
    all_returns = []
    all_sharpe_ratios = []
    
    for i in range(test_paths):
        test_env = TradingEnvironment(
            prices=test_prices[i],
            volatilities=test_vols[i],
            lot_size=lot_size,
            tick_size=tick_size,
            max_position=max_position,
            max_trade_size=max_trade_size,
            risk_aversion=risk_aversion
        )
        
        results = evaluate_agent(agent, test_env, n_episodes=1, render_final=(i==0))
        all_returns.append(results['mean_return'])
        all_sharpe_ratios.append(results['mean_sharpe'])
    
    # Plot results
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.hist(all_returns, bins=15)
    plt.title('Distribution of Returns')
    plt.xlabel('Return')
    plt.ylabel('Frequency')
    
    plt.subplot(1, 2, 2)
    plt.hist(all_sharpe_ratios, bins=15)
    plt.title('Distribution of Sharpe Ratios')
    plt.xlabel('Sharpe Ratio')
    plt.ylabel('Frequency')
    
    plt.tight_layout()
    plt.show()
    
    # Print summary
    print(f"Average return across {test_paths} test paths: {np.mean(all_returns):.4f} ± {np.std(all_returns):.4f}")
    print(f"Average Sharpe ratio across {test_paths} test paths: {np.mean(all_sharpe_ratios):.4f} ± {np.std(all_sharpe_ratios):.4f}")


def run_experiment_with_real_data_simulation():
    """
    Run experiment with simulated data based on real-world parameters
    (similar to the second experiment in the paper)
    """
    print("Running experiment with simulated data based on real-world parameters...")
    
    # Generate synthetic "real" price data
    T = 2.0  # 2 years
    N = 504  # ~252 trading days per year
    
    # Initialize Heston model with realistic parameters
    heston = HestonModel(kappa=1.5, theta=0.04, sigma=0.4, rho=-0.6, r=0.001)
    
    # Simulate "historical" data
    historical_prices, _ = heston.simulate(T=T, N=N, paths=1)
    historical_prices = historical_prices[0]
    
    # Fit Heston model to "historical" data
    fitted_params = heston.fit_to_data(historical_prices)
    print("Fitted parameters:", fitted_params)
    
    # Simulate large training dataset using fitted parameters
    train_prices, train_vols = heston.simulate(T=5.0, N=10000, paths=1)
    
    # Extract first path
    train_prices = train_prices[0]
    train_vols = train_vols[0]
    
    # Set up trading environment
    lot_size = 100
    tick_size = 0.01
    max_position = 10
    max_trade_size = 5
    risk_aversion = 1e-4
    
    env = TradingEnvironment(
        prices=train_prices,
        volatilities=train_vols,
        lot_size=lot_size,
        tick_size=tick_size,
        max_position=max_position,
        max_trade_size=max_trade_size,
        risk_aversion=risk_aversion
    )
    
    # Define action space
    action_space = list(range(-max_trade_size, max_trade_size + 1))
    
    # Initialize agent
    state_dim = 3  # position, price, volatility
    agent = FittedQAgent(
        state_dim=state_dim,
        action_space=action_space,
        gamma=0.999,
        n_estimators=10,
        min_samples_split=5,
        min_samples_leaf=5,
        verbose=True
    )
    
    # Train agent
    dataset = agent.train(
        env=env,
        n_episodes=1,
        max_steps=9000,
        epsilon=0.1,
        n_iterations=100
    )
    
    print(f"Training complete. Dataset size: {len(dataset)}")
    
    # Generate "out-of-sample" test data (similar to using 5 years of future data as in the paper)
    test_paths = 10
    test_T = 5.0  # 5 years
    test_N = 1260  # ~252 trading days per year * 5
    
    test_prices, test_vols = heston.simulate(T=test_T, N=test_N, paths=test_paths)
    
    # Evaluate agent on each test path
    all_returns = []
    all_sharpe_ratios = []
    
    for i in range(test_paths):
        test_env = TradingEnvironment(
            prices=test_prices[i],
            volatilities=test_vols[i],
            lot_size=lot_size,
            tick_size=tick_size,
            max_position=max_position,
            max_trade_size=max_trade_size,
            risk_aversion=risk_aversion
        )
        
        results = evaluate_agent(agent, test_env, n_episodes=1, render_final=(i==0))
        all_returns.append(results['mean_return'])
        all_sharpe_ratios.append(results['mean_sharpe'])
    
    # Plot results
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.hist(all_returns, bins=10)
    plt.title('Distribution of Returns')
    plt.xlabel('Return')
    plt.ylabel('Frequency')
    
    plt.subplot(1, 2, 2)
    plt.hist(all_sharpe_ratios, bins=10)
    plt.title('Distribution of Sharpe Ratios')
    plt.xlabel('Sharpe Ratio')
    plt.ylabel('Frequency')
    
    plt.tight_layout()
    plt.show()
    
    # Print summary
    print(f"Average return across {test_paths} test paths: {np.mean(all_returns):.4f} ± {np.std(all_returns):.4f}")
    print(f"Average Sharpe ratio across {test_paths} test paths: {np.mean(all_sharpe_ratios):.4f} ± {np.std(all_sharpe_ratios):.4f}")


if __name__ == "__main__":
    # Run experiment with arbitrage opportunity
    run_experiment_with_arbitrage()
    
    # Run experiment with simulated data based on real-world parameters
    run_experiment_with_real_data_simulation()