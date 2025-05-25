import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
from tqdm import tqdm
import datetime as dt
from dateutil.relativedelta import relativedelta
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class LSTM_HA(nn.Module):
    """
    Long Short-Term Memory with History state Attention network
    """
    def __init__(self, input_dim, hidden_dim, num_layers=2, dropout=0.2):
        super(LSTM_HA, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Attention parameters
        self.attention_w = nn.Parameter(torch.Tensor(hidden_dim))
        self.attention_W1 = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.attention_W2 = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        # Initialize parameters correctly
        nn.init.uniform_(self.attention_w, -0.1, 0.1)
        nn.init.xavier_uniform_(self.attention_W1)
        nn.init.xavier_uniform_(self.attention_W2)

    def forward(self, x):
        """
        x: Tensor of shape (batch_size, seq_len, input_dim)
        """
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)  # lstm_out: (batch_size, seq_len, hidden_dim)
        
        # Get the last hidden state
        h_T = lstm_out[:, -1, :]  # h_T: (batch_size, hidden_dim)
        
        # Calculate attention weights
        batch_size, seq_len, _ = lstm_out.size()
        attention_weights = torch.zeros(batch_size, seq_len, device=x.device)
        
        for t in range(seq_len):
            h_t = lstm_out[:, t, :]  # h_t: (batch_size, hidden_dim)
            
            # Calculate attention score using matmul instead of inplace operations
            alpha_t = torch.tanh(torch.matmul(h_t, self.attention_W1) + torch.matmul(h_T, self.attention_W2))
            alpha_t = torch.matmul(alpha_t, self.attention_w)
            attention_weights[:, t] = alpha_t
        
        # Normalize attention weights
        attention_weights = F.softmax(attention_weights, dim=1)
        
        # Apply attention weights to LSTM output
        # Reshape for batch matrix multiplication
        attention_weights = attention_weights.unsqueeze(1)  # (batch_size, 1, seq_len)
        context = torch.bmm(attention_weights, lstm_out)  # (batch_size, 1, hidden_dim)
        context = context.squeeze(1)  # (batch_size, hidden_dim)
        
        # Apply layer normalization
        output = self.layer_norm(context)
        
        return output


class CAAN(nn.Module):
    """
    Cross-Asset Attention Network
    """
    def __init__(self, input_dim, attn_dim=64, dropout=0.1, num_assets=None):
        super(CAAN, self).__init__()
        
        self.input_dim = input_dim
        self.attn_dim = attn_dim
        self.num_assets = num_assets
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, attn_dim)
        
        # Query, key, value projections
        self.W_Q = nn.Linear(attn_dim, attn_dim)
        self.W_K = nn.Linear(attn_dim, attn_dim)
        self.W_V = nn.Linear(attn_dim, attn_dim)
        
        # Price rising rank prior embedding
        self.max_dist = 50  # Maximum distance to consider
        self.embedding_dim = 32
        self.rank_embedding = nn.Embedding(self.max_dist + 1, self.embedding_dim)
        self.rank_weight = nn.Sequential(
            nn.Linear(self.embedding_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1, bias=False)
        )
        
        # Feed-forward network
        self.feed_forward = nn.Sequential(
            nn.Linear(attn_dim, attn_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(attn_dim * 2, attn_dim),
            nn.LayerNorm(attn_dim)
        )
        
        # Final winner score projection
        self.score_projection = nn.Sequential(
            nn.Linear(attn_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

    def forward(self, x, price_rising_ranks=None):
        """
        x: Tensor of shape (batch_size, num_assets, input_dim)
        price_rising_ranks: Tensor of shape (batch_size, num_assets)
        """
        batch_size, num_assets, _ = x.size()
        
        # Project input to attention dimension
        x_proj = self.input_projection(x)  # (batch_size, num_assets, attn_dim)
        
        # Project inputs to queries, keys, and values
        q = self.W_Q(x_proj)  # (batch_size, num_assets, attn_dim)
        k = self.W_K(x_proj)  # (batch_size, num_assets, attn_dim)
        v = self.W_V(x_proj)  # (batch_size, num_assets, attn_dim)
        
        # Calculate raw attention scores (without inplace operations)
        attn_scores = torch.bmm(q, k.transpose(1, 2))  # (batch_size, num_assets, num_assets)
        
        # Scale by sqrt(d_k)
        attn_scores = attn_scores / (self.attn_dim ** 0.5)
        
        # Apply price rising rank prior if provided
        if price_rising_ranks is not None:
            # Calculate pairwise rank distances
            rank_dists = torch.abs(price_rising_ranks.unsqueeze(-1) - price_rising_ranks.unsqueeze(1))
            rank_dists = torch.clamp(rank_dists.long(), 0, self.max_dist)
            
            # Get embeddings for rank distances
            rank_dist_embeds = self.rank_embedding(rank_dists)  # (batch_size, num_assets, num_assets, embed_dim)
            
            # Calculate rank relation coefficients
            rank_weights = self.rank_weight(rank_dist_embeds).squeeze(-1)  # (batch_size, num_assets, num_assets)
            rank_relation = torch.sigmoid(rank_weights)
            
            # Apply rank relation to attention scores (avoiding inplace operations)
            attn_scores = attn_scores * rank_relation
        
        # Apply softmax to get attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)  # (batch_size, num_assets, num_assets)
        
        # Apply attention to values
        attn_output = torch.bmm(attn_weights, v)  # (batch_size, num_assets, attn_dim)
        
        # Apply feed-forward network
        output = self.feed_forward(attn_output)  # (batch_size, num_assets, attn_dim)
        
        # Calculate winner scores
        winner_scores = torch.sigmoid(self.score_projection(output).squeeze(-1))  # (batch_size, num_assets)
        
        return winner_scores


class AlphaStock(nn.Module):
    """
    AlphaStock model combining LSTM-HA and CAAN
    """
    def __init__(self, input_dim, hidden_dim, attn_dim, lookback_window, num_assets=None, portfolio_size=None):
        super(AlphaStock, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.attn_dim = attn_dim
        self.lookback_window = lookback_window
        self.num_assets = num_assets
        self.portfolio_size = portfolio_size
        
        # LSTM-HA component
        self.lstm_ha = LSTM_HA(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=2,
            dropout=0.2
        )
        
        # CAAN component
        self.caan = CAAN(
            input_dim=hidden_dim,
            attn_dim=attn_dim,
            dropout=0.1,
            num_assets=num_assets
        )

    def forward(self, x, price_rising_ranks=None):
        """
        x: Tensor of shape (batch_size, num_assets, seq_len, input_dim)
        price_rising_ranks: Tensor of shape (batch_size, num_assets)
        """
        batch_size, num_assets, seq_len, _ = x.size()
        
        # Process each asset through LSTM-HA
        asset_representations = []
        for i in range(num_assets):
            asset_i = x[:, i, :, :]  # (batch_size, seq_len, input_dim)
            rep_i = self.lstm_ha(asset_i)  # (batch_size, hidden_dim)
            asset_representations.append(rep_i)
        
        # Stack asset representations
        asset_representations = torch.stack(asset_representations, dim=1)  # (batch_size, num_assets, hidden_dim)
        
        # Process through CAAN
        winner_scores = self.caan(asset_representations, price_rising_ranks)  # (batch_size, num_assets)
        
        return winner_scores

    def generate_portfolios(self, winner_scores, portfolio_size=None):
        """
        Generate long and short portfolios based on winner scores
        
        winner_scores: Tensor of shape (batch_size, num_assets)
        portfolio_size: Number of assets in each portfolio
        """
        if portfolio_size is None:
            portfolio_size = self.portfolio_size
        
        batch_size, num_assets = winner_scores.size()
        
        # Sort assets by winner scores
        sorted_scores, sorted_indices = torch.sort(winner_scores, dim=1, descending=True)
        
        # Initialize portfolios
        long_portfolio = torch.zeros_like(winner_scores)
        short_portfolio = torch.zeros_like(winner_scores)
        
        # Top portfolio_size assets go to long portfolio
        for i in range(batch_size):
            top_indices = sorted_indices[i, :portfolio_size]
            top_scores = winner_scores[i, top_indices]
            
            # Create weights proportional to score differences
            # Add small epsilon to avoid division by zero
            score_diffs = top_scores - top_scores.min() + 1e-6
            normalized_weights = score_diffs / score_diffs.sum()
            
            for j, idx in enumerate(top_indices):
                long_portfolio[i, idx] = normalized_weights[j]
        
        # Bottom portfolio_size assets go to short portfolio
        for i in range(batch_size):
            bottom_indices = sorted_indices[i, -portfolio_size:]
            bottom_scores = 1 - winner_scores[i, bottom_indices]
            
            # Create weights proportional to inverse score differences
            score_diffs = bottom_scores - bottom_scores.min() + 1e-6
            normalized_weights = score_diffs / score_diffs.sum()
            
            for j, idx in enumerate(bottom_indices):
                short_portfolio[i, idx] = normalized_weights[j]
        
        return long_portfolio, short_portfolio


class AlphaStockRL:
    """
    Reinforcement Learning framework for AlphaStock
    """
    def __init__(self, model, learning_rate=0.0005, weight_decay=1e-5):
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    def compute_sharpe_ratio(self, returns):
        """
        Compute Sharpe ratio from a series of returns
        """
        mean_return = returns.mean()
        std_return = returns.std() + 1e-6  # Add small epsilon to avoid division by zero
        return mean_return / std_return
    
    def compute_sortino_ratio(self, returns):
        """
        Compute Sortino ratio (using only downside deviation)
        """
        mean_return = returns.mean()
        # Create a new tensor for downside returns to avoid inplace operations
        downside_returns = torch.where(returns < 0, returns, torch.zeros_like(returns))
        downside_deviation = torch.sqrt(torch.mean(torch.square(downside_returns)) + 1e-6)
        return mean_return / downside_deviation
    
    def train(self, states, price_rising_ranks, next_returns, market_returns=None, 
              epochs=200, threshold=0, batch_size=None, validation_data=None):
        """
        Train the model using reinforcement learning with batching and validation
        
        states: Historical states of assets (batch_size, num_assets, seq_len, input_dim)
        price_rising_ranks: Price rising ranks (batch_size, num_assets)
        next_returns: Next period returns (batch_size, num_assets)
        market_returns: Market returns for comparison (batch_size)
        validation_data: Tuple of (val_states, val_ranks, val_returns, val_market)
        """
        num_samples = states.size(0)
        
        # Default batch size is the full dataset if not specified
        if batch_size is None or batch_size >= num_samples:
            batch_size = num_samples
        
        best_sharpe = -float('inf')
        patience = 20
        patience_counter = 0
        
        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0
            total_batches = (num_samples + batch_size - 1) // batch_size
            
            # Shuffle data
            indices = torch.randperm(num_samples)
            
            # Process in batches
            for batch_idx in range(total_batches):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, num_samples)
                batch_indices = indices[start_idx:end_idx]
                
                batch_states = states[batch_indices]
                batch_ranks = price_rising_ranks[batch_indices]
                batch_returns = next_returns[batch_indices]
                batch_market = market_returns[batch_indices] if market_returns is not None else None
                
                # Forward pass to get winner scores
                winner_scores = self.model(batch_states, batch_ranks)
                
                # Generate portfolios
                long_portfolio, short_portfolio = self.model.generate_portfolios(winner_scores)
                
                # Calculate portfolio returns
                long_returns = torch.sum(long_portfolio * batch_returns, dim=1)
                short_returns = torch.sum(short_portfolio * (-batch_returns), dim=1)
                portfolio_returns = long_returns + short_returns
                
                # Calculate reward metrics
                sharpe_ratio = self.compute_sharpe_ratio(portfolio_returns)
                sortino_ratio = self.compute_sortino_ratio(portfolio_returns)
                
                # Combined reward metric
                reward = 0.7 * sharpe_ratio + 0.3 * sortino_ratio
                
                # Apply threshold if market returns provided
                if batch_market is not None:
                    market_sharpe = self.compute_sharpe_ratio(batch_market)
                    threshold_sharpe = max(threshold, market_sharpe)
                else:
                    threshold_sharpe = threshold
                
                # Loss is negative (reward - threshold)
                loss = -1.0 * (reward - threshold_sharpe)
                
                # Backward pass and optimize
                self.optimizer.zero_grad()
                loss.backward()
                # Clip gradients to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_epoch_loss = epoch_loss / total_batches
            
            # Validation
            if validation_data is not None:
                val_sharpe = self.validate(validation_data)
                
                # Early stopping
                if val_sharpe > best_sharpe:
                    best_sharpe = val_sharpe
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
            
            if (epoch + 1) % 10 == 0:
                if validation_data is not None:
                    print(f"Epoch {epoch+1}, Loss: {avg_epoch_loss:.4f}, Val Sharpe: {val_sharpe:.4f}")
                else:
                    print(f"Epoch {epoch+1}, Loss: {avg_epoch_loss:.4f}")
        
        return best_sharpe if validation_data is not None else 0
    
    def validate(self, validation_data):
        """
        Validate the model
        
        validation_data: Tuple of (val_states, val_ranks, val_returns, val_market)
        """
        self.model.eval()
        val_states, val_ranks, val_returns, val_market = validation_data
        
        with torch.no_grad():
            # Forward pass
            val_scores = self.model(val_states, val_ranks)
            
            # Generate portfolios
            val_long, val_short = self.model.generate_portfolios(val_scores)
            
            # Calculate validation returns
            val_long_returns = torch.sum(val_long * val_returns, dim=1)
            val_short_returns = torch.sum(val_short * (-val_returns), dim=1)
            val_portfolio_returns = val_long_returns + val_short_returns
            
            # Calculate Sharpe ratio
            val_sharpe = self.compute_sharpe_ratio(val_portfolio_returns)
        
        return val_sharpe
    
    def backtest(self, states, price_rising_ranks, next_returns, prices, capital=1000000, transaction_cost=0.001):
        """
        Backtest the model on historical data
        
        states: Historical states of assets (batch_size, num_assets, seq_len, input_dim)
        price_rising_ranks: Price rising ranks (batch_size, num_assets)
        next_returns: Next period returns (batch_size, num_assets)
        prices: Asset prices (batch_size, num_assets)
        transaction_cost: Transaction cost as a fraction of traded value
        """
        self.model.eval()
        
        # Forward pass to get winner scores
        with torch.no_grad():
            winner_scores = self.model(states, price_rising_ranks)
            
            # Generate portfolios
            long_portfolio, short_portfolio = self.model.generate_portfolios(winner_scores)
        
        # Convert to numpy for easier calculations
        long_portfolio_np = long_portfolio.numpy()
        short_portfolio_np = short_portfolio.numpy()
        next_returns_np = next_returns.numpy()
        prices_np = prices.numpy()
        
        # Initialize portfolio values and returns
        portfolio_values = [capital]
        portfolio_returns = []
        long_values = []
        short_values = []
        
        # Initialize tracking variables
        prev_long_positions = np.zeros_like(long_portfolio_np[0])
        prev_short_positions = np.zeros_like(short_portfolio_np[0])
        
        # Simulate trading
        for t in range(long_portfolio_np.shape[0]):
            # Calculate positions in monetary terms
            long_allocation = capital / 2
            short_allocation = capital / 2
            
            long_positions = long_portfolio_np[t] * long_allocation / prices_np[t]
            short_positions = short_portfolio_np[t] * short_allocation / prices_np[t]
            
            # Calculate transaction costs based on position changes
            if t > 0:
                # For long positions
                long_position_changes = np.abs(long_positions - prev_long_positions)
                long_tc = np.sum(long_position_changes * prices_np[t] * transaction_cost)
                
                # For short positions
                short_position_changes = np.abs(short_positions - prev_short_positions)
                short_tc = np.sum(short_position_changes * prices_np[t] * transaction_cost)
                
                # Apply transaction costs
                capital -= (long_tc + short_tc)
                
                # Recalculate positions after transaction costs
                long_allocation = capital / 2
                short_allocation = capital / 2
                long_positions = long_portfolio_np[t] * long_allocation / prices_np[t]
                short_positions = short_portfolio_np[t] * short_allocation / prices_np[t]
            
            # Update previous positions
            prev_long_positions = long_positions.copy()
            prev_short_positions = short_positions.copy()
            
            # Calculate next period values
            long_value = np.sum(long_positions * prices_np[t] * (1 + next_returns_np[t]))
            short_value = np.sum(short_positions * prices_np[t] * (1 - next_returns_np[t]))
            
            # Track values
            long_values.append(long_value)
            short_values.append(short_value)
            
            # Update portfolio value
            new_value = long_value + short_value
            portfolio_values.append(new_value)
            
            # Calculate period return
            period_return = (new_value / portfolio_values[-2]) - 1
            portfolio_returns.append(period_return)
            
            # Update capital for next period
            capital = new_value
        
        # Calculate performance metrics
        cumulative_return = (portfolio_values[-1] / portfolio_values[0]) - 1
        annual_factor = 12  # Monthly data
        annualized_return = (1 + cumulative_return) ** (annual_factor / len(portfolio_returns)) - 1
        annualized_volatility = np.std(portfolio_returns) * np.sqrt(annual_factor)
        sharpe_ratio = annualized_return / annualized_volatility if annualized_volatility > 0 else 0
        
        # Maximum drawdown calculation
        max_drawdown = 0
        peak = portfolio_values[0]
        drawdown_series = []
        
        for value in portfolio_values:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            drawdown_series.append(drawdown)
            max_drawdown = max(max_drawdown, drawdown)
        
        # Calmar Ratio
        calmar_ratio = annualized_return / max_drawdown if max_drawdown > 0 else float('inf')
        
        # Downside Deviation
        negative_returns = np.array([min(r, 0) for r in portfolio_returns])
        downside_deviation = np.sqrt(np.mean(negative_returns**2)) if len(negative_returns) > 0 else 0
        downside_deviation_ratio = annualized_return / downside_deviation if downside_deviation > 0 else float('inf')
        
        # Calculate win rate and average win/loss
        win_rate = np.mean(np.array(portfolio_returns) > 0)
        avg_win = np.mean(np.array([r for r in portfolio_returns if r > 0])) if np.any(np.array(portfolio_returns) > 0) else 0
        avg_loss = np.mean(np.array([r for r in portfolio_returns if r < 0])) if np.any(np.array(portfolio_returns) < 0) else 0
        
        return {
            'portfolio_values': portfolio_values,
            'portfolio_returns': portfolio_returns,
            'long_values': long_values,
            'short_values': short_values,
            'drawdown_series': drawdown_series,
            'cumulative_return': cumulative_return,
            'annualized_return': annualized_return,
            'annualized_volatility': annualized_volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar_ratio,
            'downside_deviation_ratio': downside_deviation_ratio,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'APR': annualized_return,
            'AVOL': annualized_volatility,
            'ASR': sharpe_ratio,
            'MDD': max_drawdown,
            'CR': calmar_ratio,
            'DDR': downside_deviation_ratio
        }


def generate_enhanced_simulated_data(num_assets=20, num_periods=100, lookback_window=12):
    """
    Generate enhanced simulated market data for testing the AlphaStock model
    with clearer market regimes and signals
    
    Parameters:
    num_assets (int): Number of assets to simulate
    num_periods (int): Number of periods to simulate
    lookback_window (int): Number of periods to look back
    
    Returns:
    pandas.DataFrame: Simulated market data
    """
    # Generate dates
    end_date = dt.datetime.now()
    start_date = end_date - relativedelta(months=num_periods + lookback_window)
    dates = pd.date_range(start=start_date, end=end_date, freq='M')
    
    # Generate tickers
    tickers = [f"STOCK_{i+1}" for i in range(num_assets)]
    
    # Create a multi-index DataFrame to store the data
    index = pd.MultiIndex.from_product([tickers, ['PX_LAST', 'VOLUME', 'PE_RATIO', 'BOOK_VAL_PER_SH', 'DVD_INDICATED_YIELD']], 
                                      names=['ticker', 'field'])
    data = pd.DataFrame(index=dates, columns=index)
    
    # Create market factors
    # Market factor - overall market movement
    market_factor = np.zeros(len(dates))
    
    # Generate market regimes
    regime_length = 24  # 24 months per regime
    num_regimes = (len(dates) + regime_length - 1) // regime_length
    
    for i in range(num_regimes):
        start_idx = i * regime_length
        end_idx = min((i + 1) * regime_length, len(dates))
        
        # Alternate between bull and bear markets
        if i % 2 == 0:  # Bull market
            trend = 0.01  # Stronger positive trend
            volatility = 0.02
        else:  # Bear market
            trend = -0.01  # Stronger negative trend
            volatility = 0.03  # Higher volatility in bear markets
        
        # Generate random market movement with the trend
        regime_movement = trend + volatility * np.random.randn(end_idx - start_idx)
        market_factor[start_idx:end_idx] = regime_movement
    
    # Create sector factors - 4 sectors
    num_sectors = 4
    sector_factors = np.zeros((len(dates), num_sectors))
    
    for s in range(num_sectors):
        # Sector specific trend and volatility
        sector_trend = 0.005 * np.random.randn()
        sector_vol = 0.02 + 0.01 * np.random.rand()
        
        # Generate sector movements
        for i in range(num_regimes):
            start_idx = i * regime_length
            end_idx = min((i + 1) * regime_length, len(dates))
            
            # Sectors perform differently in different regimes
            if i % 2 == 0:  # Bull market
                sector_regime_trend = sector_trend * (1 + 0.5 * (s % 2))  # Some sectors do better in bull markets
            else:  # Bear market
                sector_regime_trend = sector_trend * (1 - 0.5 * (s % 2))  # Some sectors do better in bear markets
            
            # Generate random sector movement with trend
            sector_movement = sector_regime_trend + sector_vol * np.random.randn(end_idx - start_idx)
            sector_factors[start_idx:end_idx, s] = sector_movement
    
    # Assign each stock to a sector
    stock_sectors = np.random.randint(0, num_sectors, num_assets)
    
    # Create stock-specific factors
    stock_qualities = np.random.uniform(0.5, 1.5, num_assets)  # Quality factor for each stock
    stock_volatilities = np.random.uniform(0.02, 0.05, num_assets)  # Base volatility for each stock
    
    # Generate data for each ticker
    for i, ticker in enumerate(tickers):
        # Get stock's sector
        sector = stock_sectors[i]
        
        # Stock-specific characteristics
        quality = stock_qualities[i]
        volatility = stock_volatilities[i]
        
        # Initial price
        price = 100 * (0.8 + 0.4 * quality)
        price_series = [price]
        
        # Generate prices
        for t in range(1, len(dates)):
            # Combine market, sector, and stock-specific factors
            market_effect = market_factor[t]
            sector_effect = sector_factors[t, sector]
            stock_specific = volatility * np.random.randn()
            
            # More realistic price change with mean reversion effect
            mean_reversion = 0.1 * (100 * quality - price) / (100 * quality)
            
            # Momentum effect - stocks that have been going up tend to continue
            momentum = 0.1 * (price / price_series[0] - 1) if t > 12 else 0
            
            # Calculate price change
            price_change = (
                0.6 * market_effect +       # Market factor
                0.3 * sector_effect +       # Sector factor 
                0.2 * stock_specific +      # Stock-specific noise
                0.1 * mean_reversion +      # Mean reversion
                0.1 * momentum +            # Momentum
                0.01 * (quality - 1)        # Quality premium
            )
            
            # Update price
            price *= (1 + price_change)
            price_series.append(price)
        
        # Add prices to DataFrame
        data[(ticker, 'PX_LAST')] = price_series
        
        # Generate other fields based on price and quality
        
        # Volume - higher for higher quality stocks and in volatile periods
        base_volume = 1000000 * quality
        volume = []
        for t in range(len(dates)):
            vol_factor = 1 + 0.5 * abs(market_factor[t])  # More volume in high market movement
            daily_vol = base_volume * vol_factor * (1 + 0.3 * np.random.randn())
            volume.append(daily_vol)
        
        data[(ticker, 'VOLUME')] = volume
        
        # PE Ratio - quality stocks have higher P/E ratios
        pe_base = 15 * quality
        pe_series = []
        for t in range(len(dates)):
            # P/E expands in bull markets, contracts in bear markets
            regime_effect = 1 + 0.2 * market_factor[t]
            pe = pe_base * regime_effect * (1 + 0.2 * np.random.randn())
            pe_series.append(pe)
        
        data[(ticker, 'PE_RATIO')] = pe_series
        
        # Book-to-Market Ratio - quality stocks have lower B/M ratios
        bm_base = 0.7 / quality
        book_values = []
        for t in range(len(dates)):
            # Book value changes slowly
            if t == 0:
                book_value = price_series[t] * bm_base
            else:
                growth = 0.01 * quality + 0.005 * np.random.randn()
                book_value = book_values[-1] * (1 + growth)
            
            book_values.append(book_value)
        
        data[(ticker, 'BOOK_VAL_PER_SH')] = book_values
        
        # Dividend Yield - quality stocks have more stable dividends
        div_base = 0.02 * quality
        div_series = []
        for t in range(len(dates)):
            # Dividends are more stable than earnings
            div = div_base * (1 + 0.1 * np.random.randn())
            div_series.append(div)
        
        data[(ticker, 'DVD_INDICATED_YIELD')] = div_series
    
    return data, tickers


def preprocess_simulated_data(data, tickers, lookback_window=12):
    """
    Preprocess simulated data for the AlphaStock model
    """
    # Identify dates
    dates = data.index.unique()
    
    # Create dictionary to store processed features
    processed_data = {
        'features': [],  # Will be shape (num_periods, num_assets, lookback_window, num_features)
        'price_rising_ranks': [],  # Will be shape (num_periods, num_assets)
        'returns': [],  # Will be shape (num_periods, num_assets)
        'prices': []  # Will be shape (num_periods, num_assets)
    }
    
    # Process data period by period
    for t in range(lookback_window, len(dates)):
        period_features = []
        period_prices = []
        period_returns = []
        
        # Get data for all tickers in the current period
        for ticker in tickers:
            # Extract features for this ticker over the lookback window
            ticker_features = []
            
            # Calculate features for each month in the lookback window
            for k in range(lookback_window):
                month_idx = t - lookback_window + k
                
                # Price rising rate (PR)
                if month_idx > 0:
                    pr = data.loc[dates[month_idx], (ticker, 'PX_LAST')] / data.loc[dates[month_idx-1], (ticker, 'PX_LAST')] - 1
                else:
                    pr = 0
                
                # Calculate volatility
                if month_idx >= 3:  # Need at least 3 months to calculate volatility
                    price_window = [data.loc[dates[i], (ticker, 'PX_LAST')] for i in range(month_idx-3, month_idx+1)]
                    returns_window = [price_window[i+1]/price_window[i] - 1 for i in range(len(price_window)-1)]
                    vol = np.std(returns_window)
                else:
                    vol = 0.02  # Default volatility
                
                # Other features
                volume = data.loc[dates[month_idx], (ticker, 'VOLUME')] / 1000000  # Normalize volume
                pe = data.loc[dates[month_idx], (ticker, 'PE_RATIO')] / 20  # Normalize P/E
                bm = data.loc[dates[month_idx], (ticker, 'BOOK_VAL_PER_SH')] / data.loc[dates[month_idx], (ticker, 'PX_LAST')]  # Book-to-Market
                div = data.loc[dates[month_idx], (ticker, 'DVD_INDICATED_YIELD')]  # Dividend yield
                
                # Store features for this month
                ticker_features.append([pr, vol, volume, pe, bm, div])
            
            # Store all features for this ticker
            period_features.append(ticker_features)
            
            # Store current price
            current_price = data.loc[dates[t], (ticker, 'PX_LAST')]
            period_prices.append(current_price)
            
            # Calculate next return
            if t < len(dates) - 1:
                next_price = data.loc[dates[t+1], (ticker, 'PX_LAST')]
                next_return = next_price / current_price - 1
            else:
                next_return = 0
            
            period_returns.append(next_return)
        
        # Calculate price rising ranks
        price_changes = [data.loc[dates[t], (ticker, 'PX_LAST')] / data.loc[dates[t-1], (ticker, 'PX_LAST')] - 1 for ticker in tickers]
        price_rising_ranks = np.argsort(np.argsort(-np.array(price_changes)))
        
        # Store data for this period
        processed_data['features'].append(period_features)
        processed_data['price_rising_ranks'].append(price_rising_ranks)
        processed_data['returns'].append(period_returns)
        processed_data['prices'].append(period_prices)
    
    # Convert lists to numpy arrays
    for key in processed_data:
        processed_data[key] = np.array(processed_data[key])
    
    # Calculate market returns (equally weighted)
    processed_data['market_returns'] = np.mean(processed_data['returns'], axis=1)
    
    return processed_data


def train_alphastock_model(processed_data, num_assets, lookback_window=12, hidden_dim=128, attn_dim=64, portfolio_size=None):
    """
    Train the AlphaStock model
    """
    # Set portfolio size if not provided
    if portfolio_size is None:
        portfolio_size = num_assets // 4
    
    # Convert numpy arrays to PyTorch tensors
    features = torch.FloatTensor(processed_data['features'])
    price_rising_ranks = torch.LongTensor(processed_data['price_rising_ranks'])
    returns = torch.FloatTensor(processed_data['returns'])
    market_returns = torch.FloatTensor(processed_data['market_returns'])
    
    # Split data into training and validation sets (80/20)
    train_size = int(0.8 * features.shape[0])
    
    train_features = features[:train_size]
    train_ranks = price_rising_ranks[:train_size]
    train_returns = returns[:train_size]
    train_market = market_returns[:train_size]
    
    val_features = features[train_size:]
    val_ranks = price_rising_ranks[train_size:]
    val_returns = returns[train_size:]
    val_market = market_returns[train_size:]
    
    # Create the AlphaStock model
    num_features = features.shape[-1]  # Number of features per asset
    model = AlphaStock(
        input_dim=num_features,
        hidden_dim=hidden_dim,
        attn_dim=attn_dim,
        lookback_window=lookback_window,
        num_assets=num_assets,
        portfolio_size=portfolio_size
    )
    
    # Create the RL framework
    rl_framework = AlphaStockRL(
        model=model,
        learning_rate=0.0005,
        weight_decay=1e-5
    )
    
    # Prepare validation data
    validation_data = (val_features, val_ranks, val_returns, val_market)
    
    # Train the model
    print("Training AlphaStock model...")
    best_val_sharpe = rl_framework.train(
        states=train_features,
        price_rising_ranks=train_ranks,
        next_returns=train_returns,
        market_returns=train_market,
        epochs=200,
        batch_size=16,
        validation_data=validation_data
    )
    
    # Final validation
    print("Final model validation...")
    val_sharpe = rl_framework.validate(validation_data)
    
    # Calculate market performance
    market_sharpe = rl_framework.compute_sharpe_ratio(val_market)
    
    print(f"Final Validation Sharpe Ratio: {val_sharpe:.4f} (Market: {market_sharpe:.4f})")
    
    return model, rl_framework


def backtest_alphastock(model, rl_framework, processed_data, tickers):
    """
    Backtest the AlphaStock model
    """
    # Convert numpy arrays to PyTorch tensors
    features = torch.FloatTensor(processed_data['features'])
    price_rising_ranks = torch.LongTensor(processed_data['price_rising_ranks'])
    returns = torch.FloatTensor(processed_data['returns'])
    prices = torch.FloatTensor(processed_data['prices'])
    
    # Calculate market performance (for comparison)
    market_returns = processed_data['market_returns']
    market_values = [1000000]  # Start with same capital as the strategy
    for r in market_returns:
        market_values.append(market_values[-1] * (1 + r))
    
    market_cumulative_return = (market_values[-1] / market_values[0]) - 1
    market_annualized_return = (1 + market_cumulative_return) ** (12 / len(market_returns)) - 1
    market_annualized_volatility = np.std(market_returns) * np.sqrt(12)
    market_sharpe_ratio = market_annualized_return / market_annualized_volatility if market_annualized_volatility > 0 else 0
    
    # Calculate maximum drawdown for market
    market_max_drawdown = 0
    market_peak = market_values[0]
    for value in market_values:
        if value > market_peak:
            market_peak = value
        drawdown = (market_peak - value) / market_peak
        market_max_drawdown = max(market_max_drawdown, drawdown)
    
    # Calculate Calmar Ratio for market
    market_calmar = market_annualized_return / market_max_drawdown if market_max_drawdown > 0 else float('inf')
    
    # Calculate downside deviation for market
    market_negative_returns = np.array([min(r, 0) for r in market_returns])
    market_downside_deviation = np.sqrt(np.mean(market_negative_returns**2)) if len(market_negative_returns) > 0 else 0
    market_ddr = market_annualized_return / market_downside_deviation if market_downside_deviation > 0 else float('inf')
    
    # Backtest
    print("Backtesting AlphaStock model...")
    backtest_results = rl_framework.backtest(
        states=features,
        price_rising_ranks=price_rising_ranks,
        next_returns=returns,
        prices=prices,
        transaction_cost=0.001  # 0.1% transaction cost
    )
    
    # Print backtest results
    print("\nPerformance Metrics:")
    print(f"{'Metric':<25} {'AlphaStock':<15} {'Market':<15}")
    print("-" * 55)
    print(f"{'Cumulative Return':<25} {backtest_results['cumulative_return']*100:.2f}% {market_cumulative_return*100:.2f}%")
    print(f"{'Annualized Return (APR)':<25} {backtest_results['APR']*100:.2f}% {market_annualized_return*100:.2f}%")
    print(f"{'Annualized Volatility (AVOL)':<25} {backtest_results['AVOL']*100:.2f}% {market_annualized_volatility*100:.2f}%")
    print(f"{'Sharpe Ratio (ASR)':<25} {backtest_results['ASR']:.4f} {market_sharpe_ratio:.4f}")
    print(f"{'Maximum Drawdown (MDD)':<25} {backtest_results['MDD']*100:.2f}% {market_max_drawdown*100:.2f}%")
    print(f"{'Calmar Ratio (CR)':<25} {backtest_results['CR']:.4f} {market_calmar:.4f}")
    print(f"{'Downside Deviation Ratio (DDR)':<25} {backtest_results['DDR']:.4f} {market_ddr:.4f}")
    print(f"{'Win Rate':<25} {backtest_results['win_rate']*100:.2f}%")
    print(f"{'Avg Win':<25} {backtest_results['avg_win']*100:.2f}%")
    print(f"{'Avg Loss':<25} {backtest_results['avg_loss']*100:.2f}%")
    
    # Plot portfolio performance
    plt.figure(figsize=(12, 6))
    plt.plot(backtest_results['portfolio_values'], label='AlphaStock')
    plt.plot(market_values, label='Market (Equal Weight)')
    plt.title('AlphaStock vs Market Performance')
    plt.xlabel('Time (Months)')
    plt.ylabel('Portfolio Value ($)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    # Plot cumulative returns
    plt.figure(figsize=(12, 6))
    cumulative_returns_alphastock = [(v / backtest_results['portfolio_values'][0]) - 1 for v in backtest_results['portfolio_values']]
    cumulative_returns_market = [(v / market_values[0]) - 1 for v in market_values]
    
    plt.plot(cumulative_returns_alphastock, label='AlphaStock')
    plt.plot(cumulative_returns_market, label='Market (Equal Weight)')
    plt.title('Cumulative Returns')
    plt.xlabel('Time (Months)')
    plt.ylabel('Cumulative Return')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    # Plot drawdown
    plt.figure(figsize=(12, 6))
    plt.plot(backtest_results['drawdown_series'], label='AlphaStock Drawdown')
    plt.axhline(y=backtest_results['MDD'], color='r', linestyle='--', label=f'Max Drawdown: {backtest_results["MDD"]*100:.2f}%')
    plt.title('AlphaStock Drawdown')
    plt.xlabel('Time (Months)')
    plt.ylabel('Drawdown')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    # Get final portfolio allocation
    with torch.no_grad():
        final_scores = model(features[-1:], price_rising_ranks[-1:])
        final_long, final_short = model.generate_portfolios(final_scores)
    
    # Print top long and short positions
    print("\nTop Long Positions:")
    long_weights = final_long[0].numpy()
    top_long_indices = np.argsort(-long_weights)[:5]
    for idx in top_long_indices:
        if long_weights[idx] > 0:
            print(f"{tickers[idx]}: {long_weights[idx]*100:.2f}%")
    
    print("\nTop Short Positions:")
    short_weights = final_short[0].numpy()
    top_short_indices = np.argsort(-short_weights)[:5]
    for idx in top_short_indices:
        if short_weights[idx] > 0:
            print(f"{tickers[idx]}: {short_weights[idx]*100:.2f}%")
    
    return backtest_results, final_long, final_short


def analyze_model_sensitivity(model, processed_data, feature_names=None):
    """
    Analyze the sensitivity of the AlphaStock model to different features
    """
    if feature_names is None:
        feature_names = ["Price Rising", "Volatility", "Volume", "PE Ratio", "Book-to-Market", "Dividend"]
    
    # Convert numpy arrays to PyTorch tensors
    features = torch.FloatTensor(processed_data['features'])
    price_rising_ranks = torch.LongTensor(processed_data['price_rising_ranks'])
    
    # Get base scores
    with torch.no_grad():
        base_scores = model(features, price_rising_ranks)
    
    # Analyze sensitivity for each feature and time step
    num_features = features.shape[-1]
    lookback_window = features.shape[-2]
    
    sensitivity = np.zeros((num_features, lookback_window))
    
    for f in range(num_features):
        for t in range(lookback_window):
            # Create perturbed features
            perturbed_features = features.clone()
            delta = 0.01  # Small perturbation
            
            # Add perturbation to the specific feature at the specific time step for all assets
            perturbed_features[:, :, t, f] += delta
            
            # Get scores with perturbed features
            with torch.no_grad():
                perturbed_scores = model(perturbed_features, price_rising_ranks)
            
            # Calculate sensitivity
            sensitivity[f, t] = torch.mean((perturbed_scores - base_scores) / delta).item()
    
    # Plot sensitivity for trading features across time
    plt.figure(figsize=(12, 6))
    for f in range(3):  # First 3 features: Price Rising, Volatility, Volume
        plt.plot(range(-lookback_window, 0), sensitivity[f, :], label=feature_names[f])
    
    plt.title('Trading Features Sensitivity Analysis')
    plt.xlabel('Months Before Trading')
    plt.ylabel('Influence on Winner Score')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    # Plot sensitivity for company features across time
    plt.figure(figsize=(12, 6))
    for f in range(3, num_features):  # Last features: PE, BM, Div
        plt.plot(range(-lookback_window, 0), sensitivity[f, :], label=feature_names[f])
    
    plt.title('Company Features Sensitivity Analysis')
    plt.xlabel('Months Before Trading')
    plt.ylabel('Influence on Winner Score')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    # Analyze feature sensitivity (average over time)
    avg_sensitivity = np.mean(sensitivity, axis=1)
    
    # Plot overall feature importance
    plt.figure(figsize=(10, 6))
    colors = ['green' if val > 0 else 'red' for val in avg_sensitivity]
    plt.bar(feature_names, avg_sensitivity, color=colors)
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.title('Overall Feature Sensitivity')
    plt.ylabel('Average Influence on Winner Score')
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.show()
    
    # Return the sensitivity analysis for further use if needed
    return sensitivity, avg_sensitivity


def main():
    # Parameters
    num_assets = 20
    num_periods = 100
    lookback_window = 12
    
    # Generate enhanced simulated data
    print("Generating enhanced simulated market data...")
    data, tickers = generate_enhanced_simulated_data(num_assets, num_periods, lookback_window)
    
    # Preprocess data
    print("Preprocessing data...")
    processed_data = preprocess_simulated_data(data, tickers, lookback_window)
    
    # Check dimensions
    print("Processed data dimensions:")
    for key, value in processed_data.items():
        if isinstance(value, np.ndarray):
            print(f"{key}: {value.shape}")
    
    # Train AlphaStock model
    model, rl_framework = train_alphastock_model(
        processed_data,
        num_assets,
        lookback_window=lookback_window,
        hidden_dim=128,
        attn_dim=64,
        portfolio_size=num_assets // 4
    )
    
    # Backtest model
    backtest_results, final_long, final_short = backtest_alphastock(model, rl_framework, processed_data, tickers)
    
    # Analyze model sensitivity
    sensitivity, avg_sensitivity = analyze_model_sensitivity(model, processed_data)
    
    # Prepare insights based on sensitivity analysis
    feature_names = ["Price Rising", "Volatility", "Volume", "PE Ratio", "Book-to-Market", "Dividend"]
    
    # Identify most influential features
    positive_features = [feature_names[i] for i in range(len(feature_names)) if avg_sensitivity[i] > 0]
    negative_features = [feature_names[i] for i in range(len(feature_names)) if avg_sensitivity[i] < 0]
    
    # Analyze time window effects for price rising
    pr_sensitivity = sensitivity[0, :]
    long_term_pr = np.mean(pr_sensitivity[:lookback_window//2])
    short_term_pr = np.mean(pr_sensitivity[lookback_window//2:])
    
    if long_term_pr > 0 and short_term_pr < 0:
        momentum_insight = "The model favors stocks with long-term growth but recent undervaluation (long-term momentum, short-term reversion)"
    elif long_term_pr > 0 and short_term_pr > 0:
        momentum_insight = "The model favors stocks with consistent growth in both long and short term (pure momentum)"
    elif long_term_pr < 0 and short_term_pr > 0:
        momentum_insight = "The model favors stocks with recent momentum but long-term underperformance (short-term momentum, long-term reversion)"
    else:
        momentum_insight = "The model favors stocks with contrarian patterns in both long and short term (pure reversion)"
    
    # Print investment recommendation
    print("\nInvestment Recommendation:")
    print("1. The AlphaStock strategy suggests buying winners and selling losers based on:")
    for feature in positive_features:
        print(f"   - High {feature}")
    for feature in negative_features:
        print(f"   - Low {feature}")
    print(f"   - {momentum_insight}")
    
    print("\n2. Current recommended portfolio:")
    print("   Buy (long position):")
    long_weights = final_long[0].numpy()
    top_long_indices = np.argsort(-long_weights)[:5]
    for idx in top_long_indices:
        if long_weights[idx] > 0:
            print(f"   - {tickers[idx]}: {long_weights[idx]*100:.2f}%")
    
    print("\n   Sell (short position):")
    short_weights = final_short[0].numpy()
    top_short_indices = np.argsort(-short_weights)[:5]
    for idx in top_short_indices:
        if short_weights[idx] > 0:
            print(f"   - {tickers[idx]}: {short_weights[idx]*100:.2f}%")
    
    print("\n3. Expected performance metrics:")
    print(f"   - Annualized Return: {backtest_results['annualized_return']*100:.2f}%")
    print(f"   - Annualized Risk (Volatility): {backtest_results['annualized_volatility']*100:.2f}%")
    print(f"   - Sharpe Ratio: {backtest_results['sharpe_ratio']:.2f}")
    print(f"   - Maximum Drawdown: {backtest_results['max_drawdown']*100:.2f}%")
    print(f"   - Win Rate: {backtest_results['win_rate']*100:.2f}%")
    
    print("\n4. Strategy insights based on feature sensitivity:")
    print(f"   - Most positive influence: {feature_names[np.argmax(avg_sensitivity)]}")
    print(f"   - Most negative influence: {feature_names[np.argmin(avg_sensitivity)]}")
    print(f"   - {momentum_insight}")


if __name__ == "__main__":
    main()