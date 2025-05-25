import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from sklearn.preprocessing import StandardScaler
import random
from tqdm import tqdm

# Set seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
random.seed(42)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class LSTMHistoryAttention(nn.Module):
    """
    Long Short-Term Memory with History state Attention (LSTM-HA) network
    """
    def __init__(self, input_dim, hidden_dim, attention_dim):
        super(LSTMHistoryAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.attention_dim = attention_dim
        
        # LSTM layer
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1)
        )
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        
        # Process through LSTM
        lstm_out, (h_n, c_n) = self.lstm(x)
        # lstm_out shape: (batch_size, seq_len, hidden_dim)
        # h_n shape: (1, batch_size, hidden_dim)
        
        # Get attention weights
        batch_size, seq_len, _ = lstm_out.size()
        h_n_expanded = h_n.permute(1, 0, 2).expand(-1, seq_len, -1)
        # h_n_expanded shape: (batch_size, seq_len, hidden_dim)
        
        # Concatenate h_n with each hidden state
        attn_input = torch.cat((lstm_out, h_n_expanded), dim=2)
        # attn_input shape: (batch_size, seq_len, hidden_dim*2)
        
        # Calculate attention scores
        attn_scores = self.attention(attn_input).squeeze(-1)
        # attn_scores shape: (batch_size, seq_len)
        
        # Apply softmax to get attention weights
        attn_weights = torch.softmax(attn_scores, dim=1).unsqueeze(1)
        # attn_weights shape: (batch_size, 1, seq_len)
        
        # Apply attention weights to the LSTM output
        context = torch.bmm(attn_weights, lstm_out).squeeze(1)
        # context shape: (batch_size, hidden_dim)
        
        return context


class CrossAssetAttentionNetwork(nn.Module):
    """
    Cross-Asset Attention Network (CAAN) to model interrelationships among stocks
    """
    def __init__(self, input_dim, output_dim, use_price_prior=True):
        super(CrossAssetAttentionNetwork, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_price_prior = use_price_prior
        
        # Query, key, and value transformations
        self.query = nn.Linear(input_dim, output_dim)
        self.key = nn.Linear(input_dim, output_dim)
        self.value = nn.Linear(input_dim, output_dim)
        
        # Winner score layer
        self.score = nn.Sequential(
            nn.Linear(output_dim, 1),
            nn.Sigmoid()
        )
        
        # Prior embedding for price rising rank
        if use_price_prior:
            self.price_rank_embedding = nn.Embedding(20, 1)  # 20 discretized distance bins
            self.price_rank_weight = nn.Parameter(torch.ones(1))
        
        # Scaling factor
        self.scale = np.sqrt(output_dim)
        
    def forward(self, x, price_rank=None):
        # x shape: (batch_size, num_stocks, input_dim)
        batch_size, num_stocks, _ = x.size()
        
        # Calculate query, key, value
        q = self.query(x)  # (batch_size, num_stocks, output_dim)
        k = self.key(x)    # (batch_size, num_stocks, output_dim)
        v = self.value(x)  # (batch_size, num_stocks, output_dim)
        
        # Calculate attention scores
        attention_scores = torch.bmm(q, k.transpose(1, 2)) / self.scale
        # attention_scores shape: (batch_size, num_stocks, num_stocks)
        
        # If using price ranking prior
        if self.use_price_prior and price_rank is not None:
            # Calculate discretized distance
            expanded_rank = price_rank.unsqueeze(2).expand(batch_size, num_stocks, num_stocks)
            rank_distance = torch.abs(expanded_rank - expanded_rank.transpose(1, 2))
            
            # Discretize distance into bins (0-19)
            discretized_distance = torch.clamp(torch.div(rank_distance, 5, rounding_mode='floor'), 0, 19).long()
            
            # Get embeddings for distances
            prior_embedding = self.price_rank_embedding(discretized_distance)
            prior_weight = torch.sigmoid(self.price_rank_weight * prior_embedding).squeeze(-1)
            
            # Apply prior to attention scores
            attention_scores = attention_scores * prior_weight
        
        # Apply softmax to get attention weights
        attention_weights = torch.softmax(attention_scores, dim=2)
        # attention_weights shape: (batch_size, num_stocks, num_stocks)
        
        # Apply attention weights to values
        context = torch.bmm(attention_weights, v)
        # context shape: (batch_size, num_stocks, output_dim)
        
        # Calculate winner scores
        winner_scores = self.score(context).squeeze(-1)
        # winner_scores shape: (batch_size, num_stocks)
        
        return winner_scores


class AlphaStock(nn.Module):
    """
    AlphaStock model integrating LSTM-HA and CAAN for portfolio generation
    """
    def __init__(self, input_dim, hidden_dim=64, attention_dim=32, caan_dim=32, 
                 look_back=12, use_price_prior=True):
        super(AlphaStock, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.attention_dim = attention_dim
        self.caan_dim = caan_dim
        self.look_back = look_back
        self.use_price_prior = use_price_prior
        
        # LSTM-HA for stock representation
        self.lstm_ha = LSTMHistoryAttention(input_dim, hidden_dim, attention_dim)
        
        # CAAN for cross-asset attention
        self.caan = CrossAssetAttentionNetwork(hidden_dim, caan_dim, use_price_prior)
        
    def forward(self, x, price_rank=None):
        # x shape: (batch_size, num_stocks, seq_len, input_dim)
        batch_size, num_stocks, seq_len, _ = x.size()
        
        # Process each stock with LSTM-HA
        stock_representations = []
        for i in range(num_stocks):
            r_i = self.lstm_ha(x[:, i, :, :])
            stock_representations.append(r_i)
        
        # Concatenate all stock representations
        stock_repr_tensor = torch.stack(stock_representations, dim=1)
        # stock_repr_tensor shape: (batch_size, num_stocks, hidden_dim)
        
        # Process with CAAN to get winner scores
        winner_scores = self.caan(stock_repr_tensor, price_rank)
        # winner_scores shape: (batch_size, num_stocks)
        
        return winner_scores
    
    def generate_portfolio(self, winner_scores, portfolio_size):
        """
        Generate long and short portfolios based on winner scores
        """
        batch_size, num_stocks = winner_scores.size()
        
        # Sort stocks by winner scores
        sorted_indices = torch.argsort(winner_scores, dim=1, descending=True)
        
        # Create empty portfolios
        long_portfolio = torch.zeros_like(winner_scores)
        short_portfolio = torch.zeros_like(winner_scores)
        
        # Select top stocks for long portfolio
        for i in range(batch_size):
            # Top stocks for long portfolio
            long_indices = sorted_indices[i, :portfolio_size]
            long_scores = winner_scores[i, long_indices]
            long_weights = torch.softmax(long_scores, dim=0)
            
            # Bottom stocks for short portfolio
            short_indices = sorted_indices[i, -portfolio_size:]
            short_scores = 1 - winner_scores[i, short_indices]
            short_weights = torch.softmax(short_scores, dim=0)
            
            # Assign weights
            long_portfolio[i, long_indices] = long_weights
            short_portfolio[i, short_indices] = short_weights
        
        return long_portfolio, short_portfolio


def generate_simulated_stock_data(num_stocks=20, num_days=1000, with_factors=True):
    """
    Generate simulated stock price data with various factors
    
    Parameters:
    -----------
    num_stocks : int
        Number of stocks to simulate
    num_days : int
        Number of days to simulate
    with_factors : bool
        Whether to include factor impacts
        
    Returns:
    --------
    tuple
        (stock_prices, stock_features)
    """
    # Set initial prices
    prices = np.ones((num_days, num_stocks)) * 100
    
    # Generate factors that affect stock prices
    market_factor = np.random.normal(0.0005, 0.005, num_days)  # Market factor
    sector_factors = np.random.normal(0, 0.005, (5, num_days))  # 5 sector factors
    
    # Assign sector factor loadings to stocks
    sector_loadings = np.zeros((num_stocks, 5))
    sectors = np.random.choice(5, num_stocks)
    for i in range(num_stocks):
        sector_loadings[i, sectors[i]] = 1 + np.random.normal(0, 0.2)
    
    # Company-specific characteristics
    market_cap = np.exp(np.random.normal(10, 1, num_stocks))  # Market capitalization
    pe_ratio = np.exp(np.random.normal(3, 0.5, num_stocks))    # Price-earnings ratio
    bm_ratio = np.exp(np.random.normal(0, 0.5, num_stocks))    # Book-to-market ratio
    
    # Volatility characteristics
    stock_volatility = np.random.uniform(0.005, 0.015, num_stocks)
    
    # Features array
    features = np.zeros((num_days, num_stocks, 7))  # 7 features
    
    # Simulate price movements
    for t in range(1, num_days):
        # Market and sector effects
        market_effect = market_factor[t]
        sector_effect = np.dot(sector_loadings, np.array([factor[t] for factor in sector_factors]))
        
        # Stock-specific effect
        stock_specific = np.random.normal(0, stock_volatility)
        
        # Momentum effect (positive autocorrelation)
        momentum = 0.05 * ((prices[t-1] / prices[max(0, t-20)]) - 1)
        
        # Mean reversion effect (negative autocorrelation for short-term)
        reversion = -0.1 * ((prices[t-1] / prices[max(0, t-5)]) - 1)
        
        # Combine effects
        if with_factors:
            # Ensure combined factors don't cause extreme price changes
            price_change = np.clip(market_effect + sector_effect + stock_specific + momentum + reversion, -0.05, 0.05)
        else:
            price_change = np.clip(np.random.normal(0.0005, 0.01, num_stocks), -0.05, 0.05)
        
        # Update prices
        prices[t] = prices[t-1] * (1 + price_change)
        
        # Calculate features
        # Price Rising Rate (PR)
        features[t, :, 0] = prices[t] / prices[t-1] - 1
        
        # Fine-grained Volatility (VOL)
        if t >= 20:
            features[t, :, 1] = np.std(prices[t-20:t], axis=0) / prices[t-1]
        else:
            features[t, :, 1] = np.std(prices[:t], axis=0) / prices[t-1]
        
        # Trade Volume (TV) - simulated based on volatility and price change
        features[t, :, 2] = np.abs(price_change) * np.random.lognormal(0, 0.5, num_stocks)
        
        # Market Capitalization (MC)
        features[t, :, 3] = market_cap * prices[t] / prices[0]
        
        # Price-earnings Ratio (PE)
        features[t, :, 4] = pe_ratio * (1 + np.random.normal(0, 0.01, num_stocks))
        
        # Book-to-market Ratio (BM)
        features[t, :, 5] = bm_ratio / (prices[t] / prices[0])
        
        # Dividend (Div) - simulated quarterly dividend
        features[t, :, 6] = np.where(t % 60 == 0, 0.01 * prices[t] * np.random.uniform(0, 1, num_stocks), 0)
    
    return prices, features


class StockEnvironment:
    """
    Environment for stock trading simulation
    """
    def __init__(self, prices, features, look_back=12, episode_length=12, portfolio_size=5):
        """
        Initialize the environment
        
        Parameters:
        -----------
        prices : numpy.ndarray
            Stock prices array of shape (num_days, num_stocks)
        features : numpy.ndarray
            Stock features array of shape (num_days, num_stocks, num_features)
        look_back : int
            Number of past time steps to consider
        episode_length : int
            Length of an episode in months
        portfolio_size : int
            Number of stocks in long/short portfolios
        """
        self.prices = prices
        self.features = features
        self.look_back = look_back
        self.episode_length = episode_length
        self.portfolio_size = portfolio_size
        
        self.num_days, self.num_stocks, self.num_features = features.shape
        
        # Calculate monthly returns for training
        # Assuming 20 trading days per month
        self.trading_days_per_month = 20
        self.num_months = self.num_days // self.trading_days_per_month
        
        # Reshape data into monthly format
        self.monthly_prices = np.zeros((self.num_months, self.num_stocks))
        self.monthly_features = np.zeros((self.num_months, self.num_stocks, self.num_features))
        
        for m in range(self.num_months):
            month_start = m * self.trading_days_per_month
            month_end = min((m + 1) * self.trading_days_per_month, self.num_days)
            
            # Use end of month prices
            self.monthly_prices[m] = self.prices[month_end-1]
            
            # Average monthly features
            self.monthly_features[m] = np.mean(self.features[month_start:month_end], axis=0)
        
        # Normalize features
        self.scaler = StandardScaler()
        self.normalized_features = np.zeros_like(self.monthly_features)
        
        for f in range(self.num_features):
            feature_data = self.monthly_features[:, :, f].flatten()
            normalized_data = self.scaler.fit_transform(feature_data.reshape(-1, 1)).flatten()
            self.normalized_features[:, :, f] = normalized_data.reshape(self.num_months, self.num_stocks)
        
        # Calculate price rising ranks for each month
        self.price_ranks = np.zeros((self.num_months, self.num_stocks), dtype=np.int64)
        for m in range(1, self.num_months):
            monthly_returns = self.monthly_prices[m] / self.monthly_prices[m-1] - 1
            self.price_ranks[m] = np.argsort(np.argsort(-monthly_returns))  # Higher return gets lower rank
    
    def reset(self, start_month=None):
        """
        Reset the environment to start a new episode
        
        Parameters:
        -----------
        start_month : int or None
            Starting month index. If None, a random valid month is chosen.
            
        Returns:
        --------
        tuple
            (state, price_rank)
        """
        if start_month is None:
            # Random start, ensuring there's enough history and future data
            self.current_month = np.random.randint(self.look_back, self.num_months - self.episode_length)
        else:
            self.current_month = min(start_month, self.num_months - self.episode_length - 1)
            self.current_month = max(self.current_month, self.look_back)
        
        self.episode_step = 0
        return self._get_state()
    
    def _get_state(self):
        """
        Get current state representation
        
        Returns:
        --------
        tuple
            (state, price_rank)
        """
        # Get history features for each stock
        history_start = self.current_month - self.look_back
        history_end = self.current_month
        
        # State shape: (1, num_stocks, look_back, num_features)
        state = self.normalized_features[history_start:history_end].transpose(1, 0, 2)
        state = np.expand_dims(state, axis=0)
        
        # Current price rank
        price_rank = self.price_ranks[self.current_month]
        price_rank = np.expand_dims(price_rank, axis=0)  # Add batch dimension
        
        return state, price_rank
    
    def step(self, long_portfolio, short_portfolio):
        """
        Execute one step in the environment
        
        Parameters:
        -----------
        long_portfolio : numpy.ndarray
            Weights for long portfolio
        short_portfolio : numpy.ndarray
            Weights for short portfolio
            
        Returns:
        --------
        tuple
            (next_state, price_rank, reward, done)
        """
        # Calculate returns for the current month
        current_prices = self.monthly_prices[self.current_month]
        next_prices = self.monthly_prices[self.current_month + 1]
        
        # Calculate price rising rates
        price_rising_rates = next_prices / current_prices
        
        # Calculate portfolio returns (zero-investment strategy)
        long_return = np.sum(long_portfolio * price_rising_rates)
        short_return = np.sum(short_portfolio * price_rising_rates)
        
        # Combined return (long minus short)
        portfolio_return = long_return - short_return
        
        # Move to next month
        self.current_month += 1
        self.episode_step += 1
        
        # Check if episode is done
        done = self.episode_step >= self.episode_length
        
        # Get next state
        next_state, price_rank = self._get_state()
        
        return next_state, price_rank, portfolio_return, done
    
    def get_sharpe_ratio(self, returns):
        """
        Calculate the Sharpe ratio
        
        Parameters:
        -----------
        returns : list
            List of portfolio returns
            
        Returns:
        --------
        float
            Sharpe ratio
        """
        if len(returns) < 2:
            return 0
        
        # Annualized Sharpe ratio (assuming monthly returns)
        returns = np.array(returns)
        avg_return = np.mean(returns) * 12
        std_return = np.std(returns) * np.sqrt(12)
        
        # Avoid division by zero
        if std_return == 0:
            return 0
        
        # Assuming risk-free rate of 0 for simplicity
        return avg_return / std_return


def train_alphastock(env, model, episodes=100, lr=0.001, gamma=0.99):
    """
    Train the AlphaStock model using reinforcement learning
    
    Parameters:
    -----------
    env : StockEnvironment
        The stock trading environment
    model : AlphaStock
        The AlphaStock model
    episodes : int
        Number of episodes to train
    lr : float
        Learning rate
    gamma : float
        Discount factor
        
    Returns:
    --------
    list
        Training loss history
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_history = []
    
    for episode in tqdm(range(episodes), desc="Training episodes"):
        # Reset environment
        state, price_rank = env.reset()
        
        # Convert to torch tensors
        state_tensor = torch.FloatTensor(state).to(device)
        price_rank_tensor = torch.LongTensor(price_rank).to(device)
        
        episode_rewards = []
        log_probs = []
        
        # Run episode
        done = False
        while not done:
            # Forward pass to get winner scores
            winner_scores = model(state_tensor, price_rank_tensor)
            
            # Generate portfolios
            long_portfolio, short_portfolio = model.generate_portfolio(
                winner_scores, env.portfolio_size)
            
            # Take action and observe reward
            next_state, next_price_rank, reward, done = env.step(
                long_portfolio.detach().cpu().numpy()[0],
                short_portfolio.detach().cpu().numpy()[0]
            )
            
            # Store reward
            episode_rewards.append(reward)
            
            # Calculate log probability of selected portfolios
            # This is a simplified version for our use case
            portfolio_log_prob = torch.sum(torch.log(torch.clamp(long_portfolio, 1e-10, 1.0)) * long_portfolio) + \
                              torch.sum(torch.log(torch.clamp(short_portfolio, 1e-10, 1.0)) * short_portfolio)
            log_probs.append(portfolio_log_prob)
            
            # Update state
            state_tensor = torch.FloatTensor(next_state).to(device)
            price_rank_tensor = torch.LongTensor(next_price_rank).to(device)
        
        # Calculate Sharpe ratio as the reward signal
        sharpe_ratio = env.get_sharpe_ratio(episode_rewards)
        
        # Market benchmark Sharpe ratio
        market_sharpe = 0.2
        
        # Calculate policy loss
        policy_loss = torch.tensor(0.0, requires_grad=True).to(device)
        
        if sharpe_ratio > market_sharpe:
            # Encourage strategies that beat the market
            for log_prob in log_probs:
                policy_loss = policy_loss - log_prob * (sharpe_ratio - market_sharpe)
        else:
            # Discourage strategies that don't beat the market
            for log_prob in log_probs:
                policy_loss = policy_loss + log_prob * (market_sharpe - sharpe_ratio) * 0.1
        
        # Update model
        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()
        
        loss_history.append(policy_loss.item())
        
        if (episode + 1) % 10 == 0:
            print(f"Episode {episode+1}, Sharpe Ratio: {sharpe_ratio:.4f}, Loss: {policy_loss.item():.4f}")
    
    return loss_history


def evaluate_alphastock(env, model, num_episodes=10, start_month=None):
    """
    Evaluate the AlphaStock model
    
    Parameters:
    -----------
    env : StockEnvironment
        The stock trading environment
    model : AlphaStock
        The trained AlphaStock model
    num_episodes : int
        Number of episodes to evaluate
    start_month : int or None
        Starting month for evaluation
        
    Returns:
    --------
    dict
        Evaluation metrics
    """
    model.eval()
    
    all_returns = []
    portfolio_values = [1.0]  # Initial portfolio value
    
    for episode in range(num_episodes):
        if start_month is not None:
            # Sequential evaluation
            episode_start = start_month + episode * env.episode_length
            if episode_start + env.episode_length >= env.num_months:
                break
        else:
            # Random evaluation
            episode_start = None
        
        state, price_rank = env.reset(start_month=episode_start)
        
        episode_returns = []
        done = False
        
        while not done:
            # Convert to torch tensors
            state_tensor = torch.FloatTensor(state).to(device)
            price_rank_tensor = torch.LongTensor(price_rank).to(device)
            
            # Forward pass
            with torch.no_grad():
                winner_scores = model(state_tensor, price_rank_tensor)
                long_portfolio, short_portfolio = model.generate_portfolio(
                    winner_scores, env.portfolio_size)
            
            # Take action
            next_state, next_price_rank, reward, done = env.step(
                long_portfolio.cpu().numpy()[0],
                short_portfolio.cpu().numpy()[0]
            )
            
            # Store return
            episode_returns.append(reward)
            all_returns.append(reward)
            
            # Update portfolio value
            portfolio_values.append(portfolio_values[-1] * (1 + reward))
            
            # Update state
            state, price_rank = next_state, next_price_rank
    
    # Calculate evaluation metrics
    cum_wealth = portfolio_values[-1]
    
    # Convert to monthly returns
    monthly_returns = np.array(all_returns)
    
    # Calculate Annualized Percentage Rate (APR)
    apr = np.mean(monthly_returns) * 12
    
    # Calculate Annualized Volatility (AVOL)
    avol = np.std(monthly_returns) * np.sqrt(12)
    
    # Calculate Annualized Sharpe Ratio (ASR)
    asr = apr / avol if avol > 0 else 0
    
    # Calculate Maximum Drawdown (MDD)
    portfolio_values = np.array(portfolio_values)
    peak = np.maximum.accumulate(portfolio_values)
    drawdown = (peak - portfolio_values) / peak
    mdd = np.max(drawdown)
    
    # Calculate Calmar Ratio (CR)
    cr = apr / mdd if mdd > 0 else 0
    
    # Calculate Downside Deviation Ratio (DDR)
    downside_returns = monthly_returns[monthly_returns < 0]
    downside_dev = np.sqrt(np.mean(downside_returns ** 2)) if len(downside_returns) > 0 else 0
    ddr = apr / downside_dev if downside_dev > 0 else 0
    
    metrics = {
        'Cumulative Wealth': cum_wealth,
        'APR': apr,
        'AVOL': avol,
        'ASR': asr,
        'MDD': mdd,
        'CR': cr,
        'DDR': ddr,
        'Returns': all_returns,
        'Portfolio Values': portfolio_values
    }
    
    return metrics


def plot_portfolio_performance(alphastock_metrics, baseline_metrics=None):
    """
    Plot the performance of the AlphaStock portfolio
    
    Parameters:
    -----------
    alphastock_metrics : dict
        Metrics from AlphaStock evaluation
    baseline_metrics : dict or None
        Metrics from baseline strategies for comparison
    """
    plt.figure(figsize=(15, 10))
    
    # Plot portfolio value
    plt.subplot(2, 2, 1)
    plt.plot(alphastock_metrics['Portfolio Values'], label='AlphaStock')
    if baseline_metrics:
        for name, metrics in baseline_metrics.items():
            plt.plot(metrics['Portfolio Values'], label=name)
    plt.title('Portfolio Value')
    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    
    # Plot returns distribution
    plt.subplot(2, 2, 2)
    plt.hist(alphastock_metrics['Returns'], bins=20, alpha=0.7, label='AlphaStock')
    if baseline_metrics:
        for name, metrics in baseline_metrics.items():
            plt.hist(metrics['Returns'], bins=20, alpha=0.3, label=name)
    plt.title('Returns Distribution')
    plt.xlabel('Return')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True)
    
    # Plot metrics comparison
    plt.subplot(2, 2, 3)
    metrics = ['APR', 'ASR', 'CR', 'DDR']
    x = np.arange(len(metrics))
    width = 0.2
    
    alphastock_values = [alphastock_metrics[m] for m in metrics]
    plt.bar(x, alphastock_values, width, label='AlphaStock')
    
    if baseline_metrics:
        for i, (name, metrics_dict) in enumerate(baseline_metrics.items()):
            values = [metrics_dict[m] for m in metrics]
            plt.bar(x + width*(i+1), values, width, label=name)
    
    plt.title('Performance Metrics Comparison')
    plt.xticks(x + width/2, metrics)
    plt.legend()
    plt.grid(True)
    
    # Plot risk metrics comparison
    plt.subplot(2, 2, 4)
    risk_metrics = ['AVOL', 'MDD']
    x = np.arange(len(risk_metrics))
    
    alphastock_values = [alphastock_metrics[m] for m in risk_metrics]
    plt.bar(x, alphastock_values, width, label='AlphaStock')
    
    if baseline_metrics:
        for i, (name, metrics_dict) in enumerate(baseline_metrics.items()):
            values = [metrics_dict[m] for m in risk_metrics]
            plt.bar(x + width*(i+1), values, width, label=name)
    
    plt.title('Risk Metrics Comparison (Lower is Better)')
    plt.xticks(x + width/2, risk_metrics)
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()


def run_baseline_strategy(env, strategy_name, num_episodes=10, start_month=None):
    """
    Run a baseline investment strategy
    
    Parameters:
    -----------
    env : StockEnvironment
        The stock trading environment
    strategy_name : str
        Name of the baseline strategy ('market', 'momentum', 'reversion')
    num_episodes : int
        Number of episodes to evaluate
    start_month : int or None
        Starting month for evaluation
        
    Returns:
    --------
    dict
        Evaluation metrics
    """
    all_returns = []
    portfolio_values = [1.0]  # Initial portfolio value
    
    for episode in range(num_episodes):
        if start_month is not None:
            # Sequential evaluation
            episode_start = start_month + episode * env.episode_length
            if episode_start + env.episode_length >= env.num_months:
                break
        else:
            # Random evaluation
            episode_start = None
        
        state, price_rank = env.reset(start_month=episode_start)
        
        episode_returns = []
        done = False
        
        while not done:
            # Generate portfolio based on strategy
            if strategy_name == 'market':
                # Equal weight strategy (Buy and Hold)
                long_portfolio = np.ones(env.num_stocks) / env.num_stocks
                short_portfolio = np.zeros(env.num_stocks)
            elif strategy_name == 'momentum':
                # Momentum strategy (buy past winners, sell past losers)
                momentum_scores = np.zeros(env.num_stocks)
                
                # Use price rank as momentum indicator
                momentum_scores = 1 - price_rank[0] / env.num_stocks
                
                # Select top and bottom stocks
                long_indices = np.argsort(momentum_scores)[-env.portfolio_size:]
                short_indices = np.argsort(momentum_scores)[:env.portfolio_size]
                
                long_portfolio = np.zeros(env.num_stocks)
                short_portfolio = np.zeros(env.num_stocks)
                
                long_portfolio[long_indices] = 1 / env.portfolio_size
                short_portfolio[short_indices] = 1 / env.portfolio_size
            elif strategy_name == 'reversion':
                # Mean Reversion strategy (buy past losers, sell past winners)
                reversion_scores = np.zeros(env.num_stocks)
                
                # Use inverse price rank as reversion indicator
                reversion_scores = price_rank[0] / env.num_stocks
                
                # Select top and bottom stocks
                long_indices = np.argsort(reversion_scores)[-env.portfolio_size:]
                short_indices = np.argsort(reversion_scores)[:env.portfolio_size]
                
                long_portfolio = np.zeros(env.num_stocks)
                short_portfolio = np.zeros(env.num_stocks)
                
                long_portfolio[long_indices] = 1 / env.portfolio_size
                short_portfolio[short_indices] = 1 / env.portfolio_size
            
            # Take action
            next_state, next_price_rank, reward, done = env.step(
                long_portfolio, short_portfolio
            )
            
            # Store return
            episode_returns.append(reward)
            all_returns.append(reward)
            
            # Update portfolio value
            portfolio_values.append(portfolio_values[-1] * (1 + reward))
            
            # Update state
            state, price_rank = next_state, next_price_rank
    
    # Calculate evaluation metrics
    cum_wealth = portfolio_values[-1]
    
    # Convert to monthly returns
    monthly_returns = np.array(all_returns)
    
    # Calculate Annualized Percentage Rate (APR)
    apr = np.mean(monthly_returns) * 12
    
    # Calculate Annualized Volatility (AVOL)
    avol = np.std(monthly_returns) * np.sqrt(12)
    
    # Calculate Annualized Sharpe Ratio (ASR)
    asr = apr / avol if avol > 0 else 0
    
    # Calculate Maximum Drawdown (MDD)
    portfolio_values = np.array(portfolio_values)
    peak = np.maximum.accumulate(portfolio_values)
    drawdown = (peak - portfolio_values) / peak
    mdd = np.max(drawdown)
    
    # Calculate Calmar Ratio (CR)
    cr = apr / mdd if mdd > 0 else 0
    
    # Calculate Downside Deviation Ratio (DDR)
    downside_returns = monthly_returns[monthly_returns < 0]
    downside_dev = np.sqrt(np.mean(downside_returns ** 2)) if len(downside_returns) > 0 else 0
    ddr = apr / downside_dev if downside_dev > 0 else 0
    
    metrics = {
        'Cumulative Wealth': cum_wealth,
        'APR': apr,
        'AVOL': avol,
        'ASR': asr,
        'MDD': mdd,
        'CR': cr,
        'DDR': ddr,
        'Returns': all_returns,
        'Portfolio Values': portfolio_values
    }
    
    return metrics


def main():
    # Generate simulated stock data
    print("Generating simulated stock data...")
    prices, features = generate_simulated_stock_data(num_stocks=20, num_days=2000)
    
    # Create stock environment
    print("Creating stock environment...")
    env = StockEnvironment(prices, features, look_back=12, episode_length=12, portfolio_size=5)
    
    # Initialize AlphaStock model
    print("Initializing AlphaStock model...")
    input_dim = features.shape[2]  # Number of features
    model = AlphaStock(
        input_dim=input_dim,
        hidden_dim=64,
        attention_dim=32,
        caan_dim=32,
        look_back=12,
        use_price_prior=True
    ).to(device)
    
    # Train the model
    print("Training AlphaStock model...")
    loss_history = train_alphastock(env, model, episodes=50, lr=0.001, gamma=0.99)
    
    # Plot the training loss
    plt.figure(figsize=(10, 5))
    plt.plot(loss_history)
    plt.title('Training Loss')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.show()
    
    # Evaluate AlphaStock
    print("Evaluating AlphaStock...")
    # Use test period for evaluation
    test_start_month = int(env.num_months * 0.7)  # Start evaluation at 70% mark
    alphastock_metrics = evaluate_alphastock(env, model, num_episodes=5, start_month=test_start_month)
    
    print("\nAlphaStock Performance:")
    print(f"Cumulative Wealth: {alphastock_metrics['Cumulative Wealth']:.4f}")
    print(f"APR: {alphastock_metrics['APR']:.4f}")
    print(f"AVOL: {alphastock_metrics['AVOL']:.4f}")
    print(f"ASR: {alphastock_metrics['ASR']:.4f}")
    print(f"MDD: {alphastock_metrics['MDD']:.4f}")
    print(f"CR: {alphastock_metrics['CR']:.4f}")
    print(f"DDR: {alphastock_metrics['DDR']:.4f}")
    
    # Run baseline strategies for comparison
    print("\nRunning baseline strategies for comparison...")
    baseline_metrics = {}
    
    for strategy in ['market', 'momentum', 'reversion']:
        print(f"Evaluating {strategy} strategy...")
        baseline_metrics[strategy] = run_baseline_strategy(
            env, strategy, num_episodes=5, start_month=test_start_month)
        
        print(f"\n{strategy.capitalize()} Performance:")
        print(f"Cumulative Wealth: {baseline_metrics[strategy]['Cumulative Wealth']:.4f}")
        print(f"APR: {baseline_metrics[strategy]['APR']:.4f}")
        print(f"AVOL: {baseline_metrics[strategy]['AVOL']:.4f}")
        print(f"ASR: {baseline_metrics[strategy]['ASR']:.4f}")
        print(f"MDD: {baseline_metrics[strategy]['MDD']:.4f}")
        print(f"CR: {baseline_metrics[strategy]['CR']:.4f}")
        print(f"DDR: {baseline_metrics[strategy]['DDR']:.4f}")
    
    # Plot performance comparison
    print("\nPlotting performance comparison...")
    plot_portfolio_performance(alphastock_metrics, baseline_metrics)
    
    # Save the model
    torch.save(model.state_dict(), 'alphastock_model.pth')
    print("Model saved to 'alphastock_model.pth'")


if __name__ == "__main__":
    main()