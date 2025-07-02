import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.mixture import GaussianMixture
from scipy.stats import norm

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Simulate market data
def simulate_regime_data(n_days=1000, n_stocks=5, regime_persistence=0.98):
    # Generate market regime (2-state Markov chain: bull=1, bear=0)
    regimes = np.zeros(n_days, dtype=int)
    regime_probs = np.zeros(n_days)
    
    # Initial regime (start in bull market)
    regimes[0] = 1
    regime_probs[0] = 1.0
    
    # Transition probabilities
    p_stay = regime_persistence  # Probability of staying in current regime
    
    # Generate regimes
    for i in range(1, n_days):
        if regimes[i-1] == 1:  # Bull
            regimes[i] = 1 if np.random.rand() < p_stay else 0
        else:  # Bear
            regimes[i] = 0 if np.random.rand() < p_stay else 1
        
        # Smooth regime probability (as in the paper)
        if i > 1:
            regime_probs[i] = 0.5 * (regime_probs[i-1] + (regimes[i] == 1))
        else:
            regime_probs[i] = (regimes[i] == 1)
    
    # Stock betas and parameters
    betas = 0.8 + 0.5 * np.random.rand(n_stocks)  # Betas between 0.8 and 1.3
    
    # Parameters for index returns
    bull_mean_idx = 0.001  # 0.1% daily in bull market
    bear_mean_idx = -0.001  # -0.1% daily in bear market
    bull_vol_idx = 0.01  # 1% volatility in bull market
    bear_vol_idx = 0.02  # 2% volatility in bear market
    
    # Generate index returns
    index_returns = np.zeros(n_days)
    for i in range(n_days):
        if regimes[i] == 1:  # Bull
            index_returns[i] = np.random.normal(bull_mean_idx, bull_vol_idx)
        else:  # Bear
            index_returns[i] = np.random.normal(bear_mean_idx, bear_vol_idx)
    
    # Generate stock returns (based on CAPM-like model with regimes)
    stock_returns = np.zeros((n_days, n_stocks))
    stock_regimes = np.zeros((n_days, n_stocks), dtype=int)
    stock_regime_probs = np.zeros((n_days, n_stocks))
    
    # Each stock has its own regime, but correlated with market
    for j in range(n_stocks):
        # Stock specific regime - correlated with market but can differ
        stock_regimes[:, j] = regimes.copy()
        
        # Flip some regimes randomly (20% chance)
        flip_indices = np.random.rand(n_days) < 0.2
        stock_regimes[flip_indices, j] = 1 - stock_regimes[flip_indices, j]
        
        # Calculate smoothed regime probabilities
        stock_regime_probs[0, j] = (stock_regimes[0, j] == 1)
        for i in range(1, n_days):
            if i > 1:
                stock_regime_probs[i, j] = 0.5 * (stock_regime_probs[i-1, j] + (stock_regimes[i, j] == 1))
            else:
                stock_regime_probs[i, j] = (stock_regimes[i, j] == 1)
        
        # Generate stock returns based on regime and index
        for i in range(n_days):
            beta = betas[j]
            idiosyncratic_vol = 0.015 if stock_regimes[i, j] == 1 else 0.025
            alpha = 0.0002 if stock_regimes[i, j] == 1 else -0.0003  # Small alpha
            
            # CAPM-like model with regime-dependent parameters
            stock_returns[i, j] = alpha + beta * index_returns[i] + np.random.normal(0, idiosyncratic_vol)
    
    # Calculate short-term features (63-day rolling windows as in the paper)
    window = 63
    short_term_features = np.zeros((n_days, n_stocks * 3 + 2))
    
    for i in range(window, n_days):
        # Index features
        idx_window = index_returns[i-window:i]
        short_term_features[i, 0] = idx_window.mean()  # Mean
        short_term_features[i, 1] = idx_window.std()   # Volatility
        
        # Stock features
        for j in range(n_stocks):
            stock_window = stock_returns[i-window:i, j]
            short_term_features[i, 2 + j*3] = stock_window.mean()  # Mean
            short_term_features[i, 3 + j*3] = stock_window.std()   # Volatility
            
            # Calculate beta using covariance
            cov = np.cov(stock_window, idx_window)[0, 1]
            var_idx = np.var(idx_window)
            short_term_features[i, 4 + j*3] = cov / var_idx if var_idx > 0 else 1.0  # Beta
    
    # For the first window days, use the values from day window
    for i in range(window):
        short_term_features[i] = short_term_features[window]
    
    # Create price series (starting at 100)
    index_prices = 100 * np.cumprod(1 + index_returns)
    stock_prices = 100 * np.cumprod(1 + stock_returns, axis=0)
    
    return {
        'index_returns': index_returns,
        'stock_returns': stock_returns,
        'index_prices': index_prices,
        'stock_prices': stock_prices,
        'market_regimes': regimes,
        'market_regime_probs': regime_probs,
        'stock_regimes': stock_regimes,
        'stock_regime_probs': stock_regime_probs,
        'short_term_features': short_term_features,
        'betas': betas
    }

# Define the neural network blocks as described in the paper
class MainBlock(nn.Module):
    def __init__(self, n_stocks):
        super(MainBlock, self).__init__()
        self.n_stocks = n_stocks
        
        # FNN to process index regime probability
        self.fnn = nn.Sequential(
            nn.Linear(1, 5),
            nn.GELU(),
            nn.Linear(5, 1),
            nn.Sigmoid()
        )
        
        # Learnable weight vectors for bull and bear markets
        self.theta_bull = nn.Parameter(torch.randn(n_stocks + 1))
        self.theta_bear = nn.Parameter(torch.randn(n_stocks + 1))
    
    def forward(self, index_regime_prob):
        # Process index regime probability
        omega_bull = self.fnn(index_regime_prob)
        
        # Get bull and bear weight vectors
        w_bull = torch.softmax(self.theta_bull, dim=0)[1:]  # Drop cash weight
        w_bear = torch.softmax(self.theta_bear, dim=0)[1:]  # Drop cash weight
        
        # Combine the weights
        w_1 = omega_bull * w_bull + (1 - omega_bull) * w_bear
        
        return w_1, omega_bull

class ScoreBlock(nn.Module):
    def __init__(self, n_stocks):
        super(ScoreBlock, self).__init__()
        self.n_stocks = n_stocks
        
        # FNN for processing short-term features of each asset
        self.fnn = nn.Sequential(
            nn.Linear(5, 5),
            nn.GELU(),
            nn.Linear(5, 5),
            nn.GELU(),
            nn.Linear(5, 1)
        )
        
        # Learnable weight for combining with main block output
        self.theta_1 = nn.Parameter(torch.randn(1))
    
    def forward(self, short_term_features, w_1):
        batch_size = short_term_features.shape[0]
        scores = torch.zeros(batch_size, self.n_stocks + 1, device=short_term_features.device)
        
        # Index features
        idx_mean = short_term_features[:, 0].unsqueeze(1)
        idx_vol = short_term_features[:, 1].unsqueeze(1)
        
        # Score for cash (set to 0)
        scores[:, 0] = 0
        
        # Score for each stock
        for j in range(self.n_stocks):
            # Extract features for this stock
            stock_mean = short_term_features[:, 2 + j*3].unsqueeze(1)
            stock_vol = short_term_features[:, 3 + j*3].unsqueeze(1)
            stock_beta = short_term_features[:, 4 + j*3].unsqueeze(1)
            
            # Combine features
            stock_features = torch.cat([stock_mean, stock_vol, stock_beta, idx_mean, idx_vol], dim=1)
            
            # Calculate score
            scores[:, j + 1] = self.fnn(stock_features).squeeze()
        
        # Convert scores to weights using softmax and drop cash weight
        w_sc = torch.softmax(scores, dim=1)[:, 1:]
        
        # Calculate omega_1 using sigmoid
        omega_1 = torch.sigmoid(self.theta_1)
        
        # Combine with main block output
        w_2 = (1 - omega_1) * w_sc + omega_1 * w_1
        
        return w_2, omega_1

class GateBlock(nn.Module):
    def __init__(self):
        super(GateBlock, self).__init__()
        
        # FNN for processing stock regime probability
        self.fnn = nn.Sequential(
            nn.Linear(1, 5),
            nn.GELU(),
            nn.Linear(5, 1),
            nn.Sigmoid()
        )
    
    def forward(self, stock_regime_probs, w_2):
        batch_size = stock_regime_probs.shape[0]
        n_stocks = stock_regime_probs.shape[1]
        
        # Calculate gate for each stock
        gates = torch.zeros(batch_size, n_stocks, device=stock_regime_probs.device)
        for j in range(n_stocks):
            gates[:, j] = self.fnn(stock_regime_probs[:, j].unsqueeze(1)).squeeze()
        
        # Apply gates to weight vector
        w_3 = gates * w_2
        
        return w_3, gates

class MemoryBlock(nn.Module):
    def __init__(self):
        super(MemoryBlock, self).__init__()
        
        # FNN for processing turnover
        self.fnn = nn.Sequential(
            nn.Linear(1, 5),
            nn.GELU(),
            nn.Linear(5, 1),
            nn.Sigmoid()
        )
    
    def forward(self, w_3, current_weights):
        # Calculate turnover (L1 norm of weight changes)
        turnover = torch.norm(w_3 - current_weights, p=1, dim=1, keepdim=True)
        
        # Calculate weight parameter
        omega_p = self.fnn(turnover)
        
        # Final output
        w_final = (1 - omega_p) * w_3 + omega_p * current_weights
        
        return w_final, omega_p

# Combine blocks into the full network
class EITNetwork(nn.Module):
    def __init__(self, n_stocks, transaction_cost=0.005):
        super(EITNetwork, self).__init__()
        self.n_stocks = n_stocks
        self.transaction_cost = transaction_cost
        
        # Define blocks
        self.main_block = MainBlock(n_stocks)
        self.score_block = ScoreBlock(n_stocks)
        self.gate_block = GateBlock()
        self.memory_block = MemoryBlock()
    
    def forward(self, index_regime_prob, stock_regime_probs, short_term_features, current_weights):
        # Main block
        w_1, omega_bull = self.main_block(index_regime_prob)
        
        # Score block
        w_2, omega_1 = self.score_block(short_term_features, w_1)
        
        # Gate block
        w_3, gates = self.gate_block(stock_regime_probs, w_2)
        
        # Memory block
        w_final, omega_p = self.memory_block(w_3, current_weights)
        
        return w_final, {
            'w_1': w_1,
            'omega_bull': omega_bull,
            'w_2': w_2,
            'omega_1': omega_1,
            'w_3': w_3,
            'gates': gates,
            'omega_p': omega_p
        }
    
    def calculate_cash_weight(self, w_stocks, current_weights):
        # Calculate transaction cost
        turnover = torch.sum(torch.abs(w_stocks - current_weights), dim=1)
        transaction_cost = self.transaction_cost * turnover
        
        # Cash weight
        w_cash = 1 - torch.sum(w_stocks, dim=1) - transaction_cost
        
        return w_cash

# Fix the training function
def train_eit_network(model, data, train_days, batch_size=32, learning_rate=1e-3, epochs=50, lambda_value=20):
    # Extract data
    index_returns = torch.tensor(data['index_returns'][:train_days], dtype=torch.float32)
    stock_returns = torch.tensor(data['stock_returns'][:train_days], dtype=torch.float32)
    market_regime_probs = torch.tensor(data['market_regime_probs'][:train_days], dtype=torch.float32).unsqueeze(1)
    stock_regime_probs = torch.tensor(data['stock_regime_probs'][:train_days], dtype=torch.float32)
    short_term_features = torch.tensor(data['short_term_features'][:train_days], dtype=torch.float32)
    
    n_days = train_days
    n_stocks = stock_returns.shape[1]
    
    # Prepare datasets for batch training
    indices = torch.arange(63, n_days - 6)  # Start after window and leave room for 5 steps ahead
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    model.train()
    best_loss = float('inf')
    best_state_dict = None
    
    for epoch in range(epochs):
        # Shuffle indices
        shuffled_indices = indices[torch.randperm(len(indices))]
        
        epoch_losses = []
        
        # Process in batches
        for i in range(0, len(shuffled_indices), batch_size):
            batch_indices = shuffled_indices[i:i+batch_size]
            
            # Initialize portfolio
            current_weights = torch.zeros(len(batch_indices), n_stocks, device=index_returns.device)
            
            # Calculate portfolio returns and tracking error for each day in batch
            tracking_errors = []
            excess_returns = []
            
            for t in range(5):  # Simulate 5 time steps forward
                # Get current features
                curr_indices = batch_indices + t
                idx_regime = market_regime_probs[curr_indices]
                stock_regimes = stock_regime_probs[curr_indices]
                features = short_term_features[curr_indices]
                
                # Get allocations from model
                w_stocks, _ = model(idx_regime, stock_regimes, features, current_weights)
                
                # Calculate cash weight
                w_cash = model.calculate_cash_weight(w_stocks, current_weights)
                
                # Calculate returns (next day)
                next_returns = stock_returns[curr_indices + 1]
                next_idx_return = index_returns[curr_indices + 1]
                
                # Portfolio return (including transaction costs)
                turnover = torch.sum(torch.abs(w_stocks - current_weights), dim=1)
                transaction_cost = model.transaction_cost * turnover
                portfolio_return = torch.sum(w_stocks * next_returns, dim=1) + w_cash * 0 - transaction_cost
                
                # Tracking error and excess return
                tracking_error = (portfolio_return - next_idx_return) ** 2
                excess_return = portfolio_return - next_idx_return
                
                tracking_errors.append(tracking_error)
                excess_returns.append(excess_return)
                
                # Update weights for next step (after returns are realized)
                denominator = 1 + torch.sum(w_stocks * next_returns, dim=1, keepdim=True) + w_cash.unsqueeze(1) * 0
                current_weights = w_stocks * (1 + next_returns) / denominator
            
            # Calculate EIT loss
            tracking_errors = torch.stack(tracking_errors).mean(dim=0)
            excess_returns = torch.stack(excess_returns).mean(dim=0)
            
            # EIT loss = tracking error - lambda * excess return
            loss = torch.sqrt(tracking_errors.mean()) - lambda_value * excess_returns.mean()
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_losses.append(loss.item())
        
        # Print epoch results
        avg_loss = np.mean(epoch_losses)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state_dict = model.state_dict().copy()
    
    # Load best model
    model.load_state_dict(best_state_dict)
    return model

# Backtesting function
def backtest_eit(model, data, start_day, end_day, rebalance_period=5):
    # Extract data
    index_returns = data['index_returns'][start_day:end_day]
    stock_returns = data['stock_returns'][start_day:end_day]
    market_regime_probs = data['market_regime_probs'][start_day:end_day]
    stock_regime_probs = data['stock_regime_probs'][start_day:end_day]
    short_term_features = data['short_term_features'][start_day:end_day]
    
    n_days = end_day - start_day
    n_stocks = stock_returns.shape[1]
    
    # Initialize portfolio
    current_weights = np.zeros(n_stocks)
    cash_weight = 1.0
    portfolio_value = 100.0
    index_value = 100.0
    
    # Arrays to store results
    portfolio_values = np.zeros(n_days)
    index_values = np.zeros(n_days)
    weight_history = np.zeros((n_days, n_stocks))
    cash_history = np.zeros(n_days)
    tracking_errors = np.zeros(n_days)
    excess_returns = np.zeros(n_days)
    transaction_costs = np.zeros(n_days)
    
    model.eval()
    
    with torch.no_grad():
        for t in range(n_days):
            # Rebalance every rebalance_period days
            if t % rebalance_period == 0:
                # Convert inputs to tensors
                idx_regime = torch.tensor(market_regime_probs[t], dtype=torch.float32).unsqueeze(0).unsqueeze(1)
                stock_regimes = torch.tensor(stock_regime_probs[t], dtype=torch.float32).unsqueeze(0)
                features = torch.tensor(short_term_features[t], dtype=torch.float32).unsqueeze(0)
                weights_tensor = torch.tensor(current_weights, dtype=torch.float32).unsqueeze(0)
                
                # Get allocations from model
                new_weights, _ = model(idx_regime, stock_regimes, features, weights_tensor)
                new_weights = new_weights.squeeze().numpy()
                
                # Calculate transaction costs
                turnover = np.sum(np.abs(new_weights - current_weights))
                transaction_cost = model.transaction_cost * turnover
                transaction_costs[t] = transaction_cost
                
                # Update weights
                current_weights = new_weights
                cash_weight = 1 - np.sum(current_weights) - transaction_cost
            
            # Store current weights
            weight_history[t] = current_weights
            cash_history[t] = cash_weight
            
            # Calculate daily returns
            portfolio_return = np.sum(current_weights * stock_returns[t]) + cash_weight * 0
            index_return = index_returns[t]
            
            # Update values
            portfolio_value *= (1 + portfolio_return)
            index_value *= (1 + index_return)
            
            # Store values
            portfolio_values[t] = portfolio_value
            index_values[t] = index_value
            
            # Calculate tracking error and excess return
            tracking_errors[t] = (portfolio_return - index_return) ** 2
            excess_returns[t] = portfolio_return - index_return
            
            # Update weights after returns (if not rebalancing next day)
            if (t + 1) % rebalance_period != 0 and t < n_days - 1:
                denominator = 1 + np.sum(current_weights * stock_returns[t]) + cash_weight * 0
                current_weights = current_weights * (1 + stock_returns[t]) / denominator
                cash_weight = cash_weight * (1 + 0) / denominator
    
    # Calculate performance metrics
    tracking_error = np.sqrt(np.mean(tracking_errors))
    mean_excess_return = np.mean(excess_returns)
    information_ratio = mean_excess_return / np.std(excess_returns) if np.std(excess_returns) > 0 else 0
    
    # Calculate cumulative return
    cumulative_return_portfolio = portfolio_values[-1] / portfolio_values[0] - 1
    cumulative_return_index = index_values[-1] / index_values[0] - 1
    
    # Calculate Sharpe ratio (assuming 0 risk-free rate)
    portfolio_returns = np.diff(portfolio_values) / portfolio_values[:-1]
    sharpe_ratio = np.mean(portfolio_returns) / np.std(portfolio_returns) if np.std(portfolio_returns) > 0 else 0
    
    # Calculate maximum drawdown
    peak = np.maximum.accumulate(portfolio_values)
    drawdown = (peak - portfolio_values) / peak
    max_drawdown = np.max(drawdown)
    
    # Calculate 95% CVaR
    alpha = 0.05
    portfolio_returns = np.append(np.diff(portfolio_values) / portfolio_values[:-1], 0)  # Add a dummy value
    sorted_returns = np.sort(portfolio_returns)
    var_idx = int(np.ceil(alpha * len(sorted_returns)))
    cvar = -np.mean(sorted_returns[:var_idx])
    
    # Average transaction cost per trade
    avg_transaction_cost = np.mean(transaction_costs[transaction_costs > 0])
    
    return {
        'portfolio_values': portfolio_values,
        'index_values': index_values,
        'weight_history': weight_history,
        'cash_history': cash_history,
        'tracking_error': tracking_error,
        'mean_excess_return': mean_excess_return,
        'information_ratio': information_ratio,
        'cumulative_return_portfolio': cumulative_return_portfolio,
        'cumulative_return_index': cumulative_return_index,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'cvar': cvar,
        'avg_transaction_cost': avg_transaction_cost
    }

# Define a simple re-optimization baseline for comparison
class SimpleReoptimization:
    def __init__(self, n_stocks, transaction_cost=0.005, lambda_value=20):
        self.n_stocks = n_stocks
        self.transaction_cost = transaction_cost
        self.lambda_value = lambda_value
        self.weights = np.zeros(n_stocks)
    
    def optimize(self, historical_stock_returns, historical_index_returns):
        n_days = len(historical_index_returns)
        
        # Simple optimization: find weights that minimize tracking error - lambda * excess return
        best_loss = float('inf')
        best_weights = np.zeros(self.n_stocks)
        
        # Try 1000 random weight vectors
        for _ in range(1000):
            # Generate random weights
            weights = np.random.rand(self.n_stocks)
            weights = weights / weights.sum()  # Normalize
            
            # Calculate portfolio returns
            portfolio_returns = np.sum(weights * historical_stock_returns, axis=1)
            
            # Calculate tracking error and excess return
            tracking_error = np.sqrt(np.mean((portfolio_returns - historical_index_returns) ** 2))
            excess_return = np.mean(portfolio_returns - historical_index_returns)
            
            # Calculate loss
            loss = tracking_error - self.lambda_value * excess_return
            
            if loss < best_loss:
                best_loss = loss
                best_weights = weights
        
        return best_weights
    
    def get_weights(self, historical_stock_returns, historical_index_returns, current_weights):
        # Get new weights
        new_weights = self.optimize(historical_stock_returns, historical_index_returns)
        
        # Calculate transaction costs
        turnover = np.sum(np.abs(new_weights - current_weights))
        transaction_cost = self.transaction_cost * turnover
        
        # Cash weight
        cash_weight = 1 - np.sum(new_weights) - transaction_cost
        
        return new_weights, cash_weight

# Backtest the re-optimization approach
def backtest_reoptimization(ro_model, data, start_day, end_day, lookback=252, rebalance_period=5):
    # Extract data
    index_returns = data['index_returns']
    stock_returns = data['stock_returns']
    
    n_days = end_day - start_day
    n_stocks = stock_returns.shape[1]
    
    # Initialize portfolio
    current_weights = np.zeros(n_stocks)
    cash_weight = 1.0
    portfolio_value = 100.0
    index_value = 100.0
    
    # Arrays to store results
    portfolio_values = np.zeros(n_days)
    index_values = np.zeros(n_days)
    weight_history = np.zeros((n_days, n_stocks))
    cash_history = np.zeros(n_days)
    tracking_errors = np.zeros(n_days)
    excess_returns = np.zeros(n_days)
    transaction_costs = np.zeros(n_days)
    
    for t in range(n_days):
        idx = start_day + t
        
        # Rebalance every rebalance_period days
        if t % rebalance_period == 0 and idx >= lookback:
            # Get historical data for re-optimization
            hist_start = idx - lookback
            hist_end = idx
            hist_stock_returns = stock_returns[hist_start:hist_end]
            hist_index_returns = index_returns[hist_start:hist_end]
            
            # Get new weights
            new_weights, new_cash_weight = ro_model.get_weights(
                hist_stock_returns, hist_index_returns, current_weights)
            
            # Calculate transaction costs
            turnover = np.sum(np.abs(new_weights - current_weights))
            transaction_cost = ro_model.transaction_cost * turnover
            transaction_costs[t] = transaction_cost
            
            # Update weights
            current_weights = new_weights
            cash_weight = new_cash_weight
        
        # Store current weights
        weight_history[t] = current_weights
        cash_history[t] = cash_weight
        
        # Calculate daily returns
        portfolio_return = np.sum(current_weights * stock_returns[idx]) + cash_weight * 0
        index_return = index_returns[idx]
        
        # Update values
        portfolio_value *= (1 + portfolio_return)
        index_value *= (1 + index_return)
        
        # Store values
        portfolio_values[t] = portfolio_value
        index_values[t] = index_value
        
        # Calculate tracking error and excess return
        tracking_errors[t] = (portfolio_return - index_return) ** 2
        excess_returns[t] = portfolio_return - index_return
        
        # Update weights after returns (if not rebalancing next day)
        if (t + 1) % rebalance_period != 0 and t < n_days - 1:
            denominator = 1 + np.sum(current_weights * stock_returns[idx]) + cash_weight * 0
            current_weights = current_weights * (1 + stock_returns[idx]) / denominator
            cash_weight = cash_weight * (1 + 0) / denominator
    
    # Calculate performance metrics
    tracking_error = np.sqrt(np.mean(tracking_errors))
    mean_excess_return = np.mean(excess_returns)
    information_ratio = mean_excess_return / np.std(excess_returns) if np.std(excess_returns) > 0 else 0
    
    # Calculate cumulative return
    cumulative_return_portfolio = portfolio_values[-1] / portfolio_values[0] - 1
    cumulative_return_index = index_values[-1] / index_values[0] - 1
    
    # Calculate Sharpe ratio (assuming 0 risk-free rate)
    portfolio_returns = np.diff(portfolio_values) / portfolio_values[:-1]
    sharpe_ratio = np.mean(portfolio_returns) / np.std(portfolio_returns) if np.std(portfolio_returns) > 0 else 0
    
    # Calculate maximum drawdown
    peak = np.maximum.accumulate(portfolio_values)
    drawdown = (peak - portfolio_values) / peak
    max_drawdown = np.max(drawdown)
    
    # Calculate 95% CVaR
    alpha = 0.05
    portfolio_returns = np.append(np.diff(portfolio_values) / portfolio_values[:-1], 0)  # Add a dummy value
    sorted_returns = np.sort(portfolio_returns)
    var_idx = int(np.ceil(alpha * len(sorted_returns)))
    cvar = -np.mean(sorted_returns[:var_idx])
    
    # Average transaction cost per trade
    avg_transaction_cost = np.mean(transaction_costs[transaction_costs > 0])
    
    return {
        'portfolio_values': portfolio_values,
        'index_values': index_values,
        'weight_history': weight_history,
        'cash_history': cash_history,
        'tracking_error': tracking_error,
        'mean_excess_return': mean_excess_return,
        'information_ratio': information_ratio,
        'cumulative_return_portfolio': cumulative_return_portfolio,
        'cumulative_return_index': cumulative_return_index,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'cvar': cvar,
        'avg_transaction_cost': avg_transaction_cost
    }

# Create and visualize sample data
data = simulate_regime_data(n_days=1500, n_stocks=5)

# Visualize simulated data
plt.figure(figsize=(12, 8))
plt.subplot(3, 1, 1)
plt.plot(data['index_prices'], label='Index')
plt.title('Simulated Index Prices')
plt.grid(True)

plt.subplot(3, 1, 2)
plt.plot(data['market_regime_probs'], label='Bull Market Probability')
plt.title('Simulated Market Regime Probability')
plt.grid(True)

plt.subplot(3, 1, 3)
for i in range(5):
    plt.plot(data['stock_prices'][:, i], label=f'Stock {i+1}')
plt.title('Simulated Stock Prices')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# Train the model
n_stocks = 5
train_days = 1000  # First 1000 days for training
test_days = 500     # Last 500 days for testing

model = EITNetwork(n_stocks, transaction_cost=0.005)
model = train_eit_network(
    model, 
    data, 
    train_days=train_days,
    batch_size=32,
    learning_rate=1e-3,
    epochs=30,
    lambda_value=20
)

# Test the model
eit_results = backtest_eit(
    model,
    data,
    start_day=train_days,
    end_day=train_days+test_days,
    rebalance_period=5
)

# Compare with re-optimization
ro_model = SimpleReoptimization(n_stocks, transaction_cost=0.005, lambda_value=20)
ro_results = backtest_reoptimization(
    ro_model,
    data,
    start_day=train_days,
    end_day=train_days+test_days,
    lookback=252,  # Use approximately 1 year of data
    rebalance_period=5
)

# Print performance metrics
print("\nPerformance Metrics - EIT Neural Network:")
print(f"Tracking Error: {eit_results['tracking_error']:.6f}")
print(f"Mean Excess Return: {eit_results['mean_excess_return']:.6f}")
print(f"Information Ratio: {eit_results['information_ratio']:.6f}")
print(f"Cumulative Return (Portfolio): {eit_results['cumulative_return_portfolio']:.6f}")
print(f"Cumulative Return (Index): {eit_results['cumulative_return_index']:.6f}")
print(f"Sharpe Ratio: {eit_results['sharpe_ratio']:.6f}")
print(f"Maximum Drawdown: {eit_results['max_drawdown']:.6f}")
print(f"95% CVaR: {eit_results['cvar']:.6f}")
print(f"Avg Transaction Cost: {eit_results['avg_transaction_cost']:.6f}")

print("\nPerformance Metrics - Re-optimization:")
print(f"Tracking Error: {ro_results['tracking_error']:.6f}")
print(f"Mean Excess Return: {ro_results['mean_excess_return']:.6f}")
print(f"Information Ratio: {ro_results['information_ratio']:.6f}")
print(f"Cumulative Return (Portfolio): {ro_results['cumulative_return_portfolio']:.6f}")
print(f"Cumulative Return (Index): {ro_results['cumulative_return_index']:.6f}")
print(f"Sharpe Ratio: {ro_results['sharpe_ratio']:.6f}")
print(f"Maximum Drawdown: {ro_results['max_drawdown']:.6f}")
print(f"95% CVaR: {ro_results['cvar']:.6f}")
print(f"Avg Transaction Cost: {ro_results['avg_transaction_cost']:.6f}")

# Plot results
plt.figure(figsize=(12, 10))

# Plot portfolio values
plt.subplot(3, 1, 1)
plt.plot(eit_results['portfolio_values'], label='EIT Neural Network')
plt.plot(ro_results['portfolio_values'], label='Re-optimization')
plt.plot(eit_results['index_values'], label='Index')
plt.title('Portfolio Values')
plt.grid(True)
plt.legend()

# Plot weight allocations - EIT Neural Network
plt.subplot(3, 1, 2)
for i in range(n_stocks):
    plt.plot(eit_results['weight_history'][:, i], label=f'Stock {i+1}')
plt.plot(eit_results['cash_history'], label='Cash', linestyle='--', color='black')
plt.title('EIT Neural Network - Weight Allocations')
plt.grid(True)
plt.legend()

# Plot weight allocations - Re-optimization
plt.subplot(3, 1, 3)
for i in range(n_stocks):
    plt.plot(ro_results['weight_history'][:, i], label=f'Stock {i+1}')
plt.plot(ro_results['cash_history'], label='Cash', linestyle='--', color='black')
plt.title('Re-optimization - Weight Allocations')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()