import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random
import os
import copy
import pdblp  # Python wrapper for Bloomberg API
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Set up Bloomberg connection
def setup_bloomberg():
    print("Using synthetic data instead of Bloomberg")
    return None

# Data acquisition functions
def get_spx_data(con, start_date, end_date):
    """
    Get S&P 500 index hourly data from Bloomberg
    """
    # Always use synthetic data for this example
    print("Using synthetic SPX data")
    dates = pd.date_range(start=start_date, end=end_date, freq='H')
    dates = dates[dates.dayofweek < 5]  # Weekdays only
    dates = dates[(dates.hour >= 9) & (dates.hour <= 16)]  # Trading hours

    # Generate random price series with realistic volatility
    n = len(dates)
    returns = np.random.normal(0, 0.0005, n)
    prices = 4500 * np.exp(np.cumsum(returns))

    # Create dataframe
    df = pd.DataFrame({
        'PX_LAST': prices,
        'VOLATILITY_30D': np.random.uniform(0.15, 0.25, n)
    }, index=dates)
    return df

def get_option_chains(con, spx_data, start_date, end_date):
    """
    Get S&P 500 option chain data from Bloomberg
    """
    # Always use synthetic option data for this example
    print("Using synthetic option data")
    # This will create realistic option prices based on Black-Scholes
    df_list = []

    for date in spx_data.index:
        # Generate a few options for each date with different strikes and maturities
        spot = spx_data.loc[date, 'PX_LAST']
        vol = spx_data.loc[date, 'VOLATILITY_30D']

        # Generate options with different moneyness and maturities
        for moneyness in [0.9, 0.95, 1.0, 1.05, 1.1]:
            for days in [10, 30, 60]:
                strike = spot * moneyness

                # Calculate option price using Black-Scholes
                r = 0.03  # Assume 3% risk-free rate
                T = days / 365

                # Calculate d1 and d2
                d1 = (np.log(spot/strike) + (r + 0.5 * vol**2) * T) / (vol * np.sqrt(T))
                d2 = d1 - vol * np.sqrt(T)

                # Calculate call and put prices
                call_price = spot * norm.cdf(d1) - strike * np.exp(-r * T) * norm.cdf(d2)
                put_price = strike * np.exp(-r * T) * norm.cdf(-d2) - spot * norm.cdf(-d1)

                # Add some bid-ask spread
                call_bid = call_price * 0.99
                call_ask = call_price * 1.01
                put_bid = put_price * 0.99
                put_ask = put_price * 1.01

                # Add row for call option
                df_list.append({
                    'DATE': date,
                    'UNDERLYING_PRICE': spot,
                    'STRIKE': strike,
                    'DAYS_TO_EXPIRY': days,
                    'OPTION_TYPE': 'C',
                    'OPTION_PRICE': call_price,
                    'OPTION_BID': call_bid,
                    'OPTION_ASK': call_ask,
                    'IMPLIED_VOL': vol,
                    'DELTA': norm.cdf(d1),
                    'RISK_FREE_RATE': r
                })

                # Add row for put option
                df_list.append({
                    'DATE': date,
                    'UNDERLYING_PRICE': spot,
                    'STRIKE': strike,
                    'DAYS_TO_EXPIRY': days,
                    'OPTION_TYPE': 'P',
                    'OPTION_PRICE': put_price,
                    'OPTION_BID': put_bid,
                    'OPTION_ASK': put_ask,
                    'IMPLIED_VOL': vol,
                    'DELTA': norm.cdf(d1) - 1,
                    'RISK_FREE_RATE': r
                })

    df = pd.DataFrame(df_list)
    df.set_index('DATE', inplace=True)
    return df

def get_risk_free_rates(con, start_date, end_date):
    """
    Get risk-free rate data from Bloomberg
    """
    # Always use synthetic risk-free rate data
    print("Using synthetic risk-free rate data")
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    rates = np.random.uniform(0.02, 0.04, len(dates))  # 2-4% range
    return pd.DataFrame({'RATE': rates}, index=dates)

# Data preprocessing functions
def preprocess_data(df):
    """
    Prepare data for the DRL agent
    """
    # Make a copy to avoid modifying the original
    processed_df = df.copy()

    # Calculate moneyness
    processed_df['MONEYNESS'] = processed_df['UNDERLYING_PRICE'] / processed_df['STRIKE']

    # Normalize features using z-score
    features = ['MONEYNESS', 'DAYS_TO_EXPIRY', 'IMPLIED_VOL']
    scaler = StandardScaler()
    processed_df[features] = scaler.fit_transform(processed_df[features])

    return processed_df, scaler

def preprocess_data_with_scaler(df, scaler):
    """
    Preprocess new data using an existing scaler
    """
    processed_df = df.copy()
    processed_df['MONEYNESS'] = processed_df['UNDERLYING_PRICE'] / processed_df['STRIKE']

    features = ['MONEYNESS', 'DAYS_TO_EXPIRY', 'IMPLIED_VOL']
    processed_df[features] = scaler.transform(processed_df[features])

    return processed_df

def create_episode_paths(data, path_length=5):
    """
    Group data into episode paths of specified length
    """
    paths = []

    # Group by option identifier (combination of strike, expiry, and type)
    data['OPTION_ID'] = data['STRIKE'].astype(str) + '_' + data['DAYS_TO_EXPIRY'].astype(str) + '_' + data['OPTION_TYPE']

    for _, group in data.groupby('OPTION_ID'):
        # Sort by date
        group = group.sort_index()

        # Create paths of specified length
        for i in range(0, len(group) - path_length + 1, path_length):
            path = group.iloc[i:i+path_length]
            if len(path) == path_length:
                paths.append(path)

    # If no paths were created, create some synthetic paths
    if len(paths) == 0:
        print("No paths created from data, generating synthetic paths")
        # Take a sample option and duplicate it with slight variations
        if not data.empty:
            sample_option = data.iloc[0:1]

            # Create 100 synthetic paths
            for i in range(100):
                path_data = []
                base_price = sample_option['OPTION_PRICE'].values[0]
                base_underlying = sample_option['UNDERLYING_PRICE'].values[0]

                for j in range(path_length):
                    # Create a copy of the sample with slightly modified prices
                    new_row = sample_option.copy()
                    price_change = np.random.normal(0, 0.01)  # 1% standard deviation
                    underlying_change = np.random.normal(0, 0.005)  # 0.5% standard deviation

                    new_row['OPTION_PRICE'] = base_price * (1 + price_change)
                    new_row['UNDERLYING_PRICE'] = base_underlying * (1 + underlying_change)

                    path_data.append(new_row)

                # Combine into a path
                path = pd.concat(path_data)
                paths.append(path)

    print(f"Created {len(paths)} paths")
    return paths

# Actor and Critic Networks for TD3
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, 250)
        self.l2 = nn.Linear(250, 250)
        self.l3 = nn.Linear(250, 250)
        self.l4 = nn.Linear(250, action_dim)

        self.max_action = max_action

    def forward(self, state):
        a = F.leaky_relu(self.l1(state), 0.05)
        a = F.leaky_relu(self.l2(a), 0.05)
        a = F.leaky_relu(self.l3(a), 0.05)
        return torch.tanh(self.l4(a)) * self.max_action

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        # First Critic network
        self.l1 = nn.Linear(state_dim + action_dim, 250)
        self.l2 = nn.Linear(250, 250)
        self.l3 = nn.Linear(250, 250)
        self.l4 = nn.Linear(250, 1)

        # Second Critic network (for TD3)
        self.l5 = nn.Linear(state_dim + action_dim, 250)
        self.l6 = nn.Linear(250, 250)
        self.l7 = nn.Linear(250, 250)
        self.l8 = nn.Linear(250, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.leaky_relu(self.l1(sa), 0.05)
        q1 = F.leaky_relu(self.l2(q1), 0.05)
        q1 = F.leaky_relu(self.l3(q1), 0.05)
        q1 = self.l4(q1)

        q2 = F.leaky_relu(self.l5(sa), 0.05)
        q2 = F.leaky_relu(self.l6(q2), 0.05)
        q2 = F.leaky_relu(self.l7(q2), 0.05)
        q2 = self.l8(q2)

        return q1, q2

    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.leaky_relu(self.l1(sa), 0.05)
        q1 = F.leaky_relu(self.l2(q1), 0.05)
        q1 = F.leaky_relu(self.l3(q1), 0.05)
        q1 = self.l4(q1)

        return q1

# TD3 Agent Implementation
class TD3:
    def __init__(
        self,
        state_dim,
        action_dim,
        max_action,
        discount=0.99,
        tau=0.001,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_freq=2,
        device="cuda"
    ):
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-4)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-4)

        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.device = device

        self.total_it = 0

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        return self.actor(state).cpu().data.numpy().flatten()

    def train(self, replay_buffer, batch_size=100):
        self.total_it += 1

        # Sample replay buffer
        batch = random.sample(replay_buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)

        state = torch.FloatTensor(np.array(state)).to(self.device)
        action = torch.FloatTensor(np.array(action)).to(self.device)
        reward = torch.FloatTensor(np.array(reward).reshape(-1, 1)).to(self.device)
        next_state = torch.FloatTensor(np.array(next_state)).to(self.device)
        done = torch.FloatTensor(np.array(done).reshape(-1, 1)).to(self.device)

        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (
                torch.randn_like(action) * self.policy_noise
            ).clamp(-self.noise_clip, self.noise_clip)

            next_action = (
                self.actor_target(next_state) + noise
            ).clamp(-self.max_action, self.max_action)

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + (1 - done) * self.discount * target_Q

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(state, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:

            # Compute actor loss
            actor_loss = -self.critic.Q1(state, self.actor(state)).mean()

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")

        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")

    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
        self.actor_target = copy.deepcopy(self.actor)

# Hedging Environment
class HedgingEnvironment:
    def __init__(self, paths, transaction_cost=0.0001, xi=1):
        """
        Initialize hedging environment

        Args:
            paths: List of dataframes, each representing a 5-day episode
            transaction_cost: Transaction cost percentage (1 BPS = 0.0001)
            xi: Risk-return trade-off parameter
        """
        self.paths = paths
        self.transaction_cost = transaction_cost
        self.xi = xi
        self.current_step = 0
        self.current_path_idx = 0
        self.current_path = None
        self.option_position = -1  # Short 1 option
        self.stock_position = 0

    def reset(self, path_idx=None):
        """Reset environment to start new episode"""
        if path_idx is None:
            path_idx = np.random.randint(0, len(self.paths))

        self.current_path_idx = path_idx
        self.current_path = self.paths[path_idx]
        self.current_step = 0
        self.stock_position = 0
        self.option_position = -1  # Short 1 option

        return self._get_state()

    def _get_state(self):
        """Return current state observation"""
        current_data = self.current_path.iloc[self.current_step]

        # State features: moneyness, time to maturity, current holdings, implied volatility
        moneyness = current_data['MONEYNESS']
        time_to_maturity = current_data['DAYS_TO_EXPIRY'] / 365
        implied_vol = current_data['IMPLIED_VOL']

        return np.array([
            moneyness,
            time_to_maturity,
            self.stock_position,
            implied_vol
        ])

    def step(self, action):
        """Take action and return new state, reward, done flag"""
        current_data = self.current_path.iloc[self.current_step]

        # Action is the desired stock position between -1 and 1
        new_stock_position = float(action)

        # Calculate transaction costs
        stock_change = new_stock_position - self.stock_position
        transaction_cost = abs(stock_change) * current_data['UNDERLYING_PRICE'] * self.transaction_cost

        # Move to next step to get price changes
        self.current_step += 1
        next_data = self.current_path.iloc[self.current_step]

        # Calculate P&L components as in equation (9) from the paper
        option_pnl = self.option_position * (next_data['OPTION_PRICE'] - current_data['OPTION_PRICE'])
        stock_pnl = self.stock_position * (next_data['UNDERLYING_PRICE'] - current_data['UNDERLYING_PRICE'])
        pnl = option_pnl + stock_pnl - transaction_cost

        # Update positions
        self.stock_position = new_stock_position

        # Calculate reward as in equation (8) from the paper
        pnl_scaled = pnl / current_data['UNDERLYING_PRICE']  # Scale by underlying price
        reward = pnl_scaled - self.xi * abs(pnl_scaled)

        # Check if episode is done
        done = self.current_step >= len(self.current_path) - 1

        return self._get_state(), reward, done, {
            "pnl": pnl_scaled,
            "transaction_cost": transaction_cost / current_data['UNDERLYING_PRICE'],
            "option_pnl": option_pnl / current_data['UNDERLYING_PRICE'],
            "stock_pnl": stock_pnl / current_data['UNDERLYING_PRICE']
        }

# Black-Scholes functions for benchmarking
def black_scholes_delta(S, K, T, r, sigma, option_type="C"):
    """Calculate Black-Scholes delta"""
    d1 = (np.log(S/K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))

    if option_type == "C":
        return norm.cdf(d1)
    else:  # P for put
        return norm.cdf(d1) - 1

def run_bs_benchmark(paths, transaction_cost=0.0001, xi=1):
    """Run Black-Scholes delta hedging benchmark"""
    results = []

    for path in paths:
        stock_position = 0
        option_position = -1  # Short 1 option
        pnls = []
        transaction_costs = []

        try:
            for i in range(len(path) - 1):
                current = path.iloc[i]
                next_step = path.iloc[i+1]

                # Make sure we have positive values for time and volatility
                days_to_expiry = max(1, current['DAYS_TO_EXPIRY'])  # Ensure at least 1 day
                implied_vol = max(0.001, current['IMPLIED_VOL'])  # Ensure positive volatility

                # Calculate delta using Black-Scholes
                try:
                    delta = black_scholes_delta(
                        current['UNDERLYING_PRICE'],
                        current['STRIKE'],
                        days_to_expiry / 365,
                        current['RISK_FREE_RATE'],
                        implied_vol,
                        current['OPTION_TYPE']
                    )

                    # Handle potential NaN delta
                    if np.isnan(delta):
                        delta = 0.5  # Use a default delta

                    # Delta hedge requires this position in the underlying
                    new_stock_position = delta * option_position * -1

                    # Calculate transaction costs
                    stock_change = new_stock_position - stock_position
                    tx_cost = abs(stock_change) * current['UNDERLYING_PRICE'] * transaction_cost
                    transaction_costs.append(tx_cost / current['UNDERLYING_PRICE'])

                    # Calculate P&L
                    option_pnl = option_position * (next_step['OPTION_PRICE'] - current['OPTION_PRICE'])
                    stock_pnl = stock_position * (next_step['UNDERLYING_PRICE'] - current['UNDERLYING_PRICE'])
                    pnl = (option_pnl + stock_pnl - tx_cost) / current['UNDERLYING_PRICE']
                    pnls.append(pnl)

                    # Update position
                    stock_position = new_stock_position
                except Exception as e:
                    print(f"Error in BS calculation: {e}")
                    continue

            # Calculate episode statistics if we have any PnL data
            if pnls:
                episode_pnl = sum(pnls)
                episode_std = np.std(pnls) if len(pnls) > 1 else 0
                episode_tx_cost = sum(transaction_costs) if transaction_costs else 0
                episode_reward = episode_pnl - xi * episode_std

                results.append({
                    'pnl': episode_pnl,
                    'std': episode_std,
                    'tx_cost': episode_tx_cost,
                    'reward': episode_reward
                })
        except Exception as e:
            print(f"Error processing path: {e}")
            continue

    # If no valid results, create a dummy result
    if not results:
        print("No valid BS benchmark results, creating dummy data")
        results.append({
            'pnl': 0,
            'std': 0,
            'tx_cost': 0,
            'reward': 0
        })

    return pd.DataFrame(results)

# Training functions
def train_agent(env, agent, num_episodes=100000, batch_size=10000, exploration_noise=0.3):
    """Train the DRL agent"""
    rewards = []
    replay_buffer = deque(maxlen=1000000)
    best_reward = -float('inf')

    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        done = False

        while not done:
            # Select action with noise for exploration
            action = agent.select_action(state)
            noise = np.random.normal(0, exploration_noise, size=action.shape)
            noisy_action = np.clip(action + noise, -1, 1)

            # Take action in environment
            next_state, reward, done, info = env.step(noisy_action)
            episode_reward += reward

            # Store transition in replay buffer
            replay_buffer.append((state, noisy_action, reward, next_state, done))
            state = next_state

            # Train agent after collecting enough samples
            if len(replay_buffer) > batch_size:
                agent.train(replay_buffer, batch_size=batch_size)

        rewards.append(episode_reward)

        # Decay exploration noise
        exploration_noise = max(0.1, exploration_noise * 0.995)

        # Print progress
        if episode % 100 == 0:
            avg_reward = np.mean(rewards[-100:]) if len(rewards) >= 100 else np.mean(rewards)
            print(f"Episode {episode}, Avg Reward: {avg_reward:.4f}, Exploration Noise: {exploration_noise:.3f}")

            # Save best model
            if avg_reward > best_reward:
                best_reward = avg_reward
                agent.save(f"./models/td3_hedging_best")
                print(f"New best model saved with reward: {best_reward:.4f}")

    return agent, rewards

def evaluate_agent(agent, env, num_episodes=100):
    """Evaluate the trained agent performance"""
    results = []

    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        pnls = []
        transaction_costs = []

        while not done:
            # Select action without exploration noise
            action = agent.select_action(state)

            # Take action in environment
            next_state, reward, done, info = env.step(action)
            episode_reward += reward
            pnls.append(info['pnl'])
            transaction_costs.append(info['transaction_cost'])

            state = next_state

        results.append({
            'episode': episode,
            'reward': episode_reward,
            'pnl': sum(pnls),
            'std': np.std(pnls),
            'tx_cost': sum(transaction_costs)
        })

    return pd.DataFrame(results)

def compare_results(drl_results, bs_results):
    """Compare DRL vs Black-Scholes performance"""
    try:
        print("=== DRL Agent Performance ===")
        print(f"Mean Episode P&L: {drl_results['pnl'].mean()*100:.6f}%")
        print(f"Std Episode P&L: {drl_results['std'].mean()*100:.6f}%")
        print(f"Mean Transaction Costs: {drl_results['tx_cost'].mean()*100:.6f}%")
        print(f"Mean Reward: {drl_results['reward'].mean():.4f}")

        print("\n=== Black-Scholes Performance ===")
        print(f"Mean Episode P&L: {bs_results['pnl'].mean()*100:.6f}%")
        print(f"Std Episode P&L: {bs_results['std'].mean()*100:.6f}%")
        print(f"Mean Transaction Costs: {bs_results['tx_cost'].mean()*100:.6f}%")
        print(f"Mean Reward: {bs_results['reward'].mean():.4f}")

        # Create comparison plots
        plt.figure(figsize=(16, 12))

        # Helper function to safely plot histograms
        def safe_hist(data, **kwargs):
            try:
                # Filter out NaN values
                filtered_data = data.dropna()
                if len(filtered_data) > 0:
                    plt.hist(filtered_data, **kwargs)
            except Exception as e:
                print(f"Error plotting histogram: {e}")

        plt.subplot(2, 2, 1)
        plt.title('P&L Distribution')
        safe_hist(drl_results['pnl']*100, alpha=0.5, label='DRL', bins=20)
        safe_hist(bs_results['pnl']*100, alpha=0.5, label='BS', bins=20)
        plt.xlabel('P&L (%)')
        plt.ylabel('Frequency')
        plt.legend()

        plt.subplot(2, 2, 2)
        plt.title('Risk (Std) Distribution')
        safe_hist(drl_results['std']*100, alpha=0.5, label='DRL', bins=20)
        safe_hist(bs_results['std']*100, alpha=0.5, label='BS', bins=20)
        plt.xlabel('Standard Deviation (%)')
        plt.ylabel('Frequency')
        plt.legend()

        plt.subplot(2, 2, 3)
        plt.title('Transaction Costs')
        safe_hist(drl_results['tx_cost']*100, alpha=0.5, label='DRL', bins=20)
        safe_hist(bs_results['tx_cost']*100, alpha=0.5, label='BS', bins=20)
        plt.xlabel('Transaction Costs (%)')
        plt.ylabel('Frequency')
        plt.legend()

        plt.subplot(2, 2, 4)
        plt.title('Reward Distribution')
        safe_hist(drl_results['reward'], alpha=0.5, label='DRL', bins=20)
        safe_hist(bs_results['reward'], alpha=0.5, label='BS', bins=20)
        plt.xlabel('Reward')
        plt.ylabel('Frequency')
        plt.legend()

        plt.tight_layout()
        plt.savefig('hedging_comparison.png')
        plt.show()

        return {
            'drl_mean_pnl': drl_results['pnl'].mean(),
            'bs_mean_pnl': bs_results['pnl'].mean(),
            'drl_std': drl_results['std'].mean(),
            'bs_std': bs_results['std'].mean(),
            'drl_tx_cost': drl_results['tx_cost'].mean(),
            'bs_tx_cost': bs_results['tx_cost'].mean(),
            'drl_reward': drl_results['reward'].mean(),
            'bs_reward': bs_results['reward'].mean()
        }
    except Exception as e:
        print(f"Error in compare_results: {e}")
        return {
            'drl_mean_pnl': drl_results['pnl'].mean() if not drl_results.empty else 0,
            'bs_mean_pnl': bs_results['pnl'].mean() if not bs_results.empty else 0,
            'error': str(e)
        }

def plot_training_progress(rewards):
    """Plot the training progress over episodes"""
    plt.figure(figsize=(10, 6))
    plt.plot(rewards)
    plt.title('Training Progress')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.grid(True)
    plt.savefig('training_progress.png')
    plt.show()

def main():
    # Create directories for saving models and results
    os.makedirs("./models", exist_ok=True)
    os.makedirs("./results", exist_ok=True)

    # Parameters
    transaction_cost = 0.0001  # 1 BPS
    xi = 1  # Risk-return trade-off

    # 1. Get data from Bloomberg or use synthetic data
    print("Fetching data from Bloomberg...")
    con = setup_bloomberg()

    start_date = "2022-01-01"
    end_date = "2023-04-30"

    # Training data (1 year)
    train_start = "2022-01-01"
    train_end = "2022-12-31"

    # Test data (4 months)
    test_start = "2023-01-01"
    test_end = "2023-04-30"

    # Get market data
    spx_data = get_spx_data(con, start_date, end_date)
    option_data = get_option_chains(con, spx_data, start_date, end_date)
    rates_data = get_risk_free_rates(con, start_date, end_date)

    # Add risk-free rates to option data
    for date in option_data.index:
        rate_date = pd.to_datetime(date).date()
        if rate_date in rates_data.index:
            option_data.loc[date, 'RISK_FREE_RATE'] = rates_data.loc[rate_date, 'RATE']
        else:
            # Use closest available date
            closest_date = rates_data.index[rates_data.index.get_indexer([rate_date], method='nearest')[0]]
            option_data.loc[date, 'RISK_FREE_RATE'] = rates_data.loc[closest_date, 'RATE']

    # 2. Preprocess data
    print("Preprocessing data...")
    train_data = option_data.loc[train_start:train_end].copy()
    test_data = option_data.loc[test_start:test_end].copy()

    train_data, scaler = preprocess_data(train_data)
    test_data = preprocess_data_with_scaler(test_data, scaler)

    # 3. Create episode paths
    train_paths = create_episode_paths(train_data)
    test_paths = create_episode_paths(test_data)

    print(f"Created {len(train_paths)} training paths and {len(test_paths)} test paths")

    # 4. Create environments
    train_env = HedgingEnvironment(train_paths, transaction_cost=transaction_cost, xi=xi)
    test_env = HedgingEnvironment(test_paths, transaction_cost=transaction_cost, xi=xi)

    # 5. Initialize agent
    state_dim = 4  # moneyness, time to maturity, current holdings, implied volatility
    action_dim = 1  # position in the underlying asset
    max_action = 1.0  # maximum position size

    agent = TD3(
        state_dim=state_dim,
        action_dim=action_dim,
        max_action=max_action,
        device=device
    )

    # 6. Train agent
    print("Training DRL agent...")
    _, training_rewards = train_agent(
        train_env,
        agent,
        num_episodes=500,  # More episodes for better results
        batch_size=100,
        exploration_noise=0.3
    )

    # Plot training progress
    plot_training_progress(training_rewards)

    # 7. Load best model for evaluation
    agent.load("./models/td3_hedging_best")

    # 8. Evaluate agent performance
    print("Evaluating DRL agent on test data...")
    drl_results = evaluate_agent(agent, test_env, num_episodes=len(test_paths))

    # 9. Run Black-Scholes benchmark
    print("Running Black-Scholes benchmark on test data...")
    bs_results = run_bs_benchmark(test_paths, transaction_cost=transaction_cost, xi=xi)

    # 10. Compare results
    comparison = compare_results(drl_results, bs_results)

    # 11. Save results
    drl_results.to_csv("./results/drl_results.csv")
    bs_results.to_csv("./results/bs_results.csv")

    print("Done!")
    return comparison

if __name__ == "__main__":
    main()