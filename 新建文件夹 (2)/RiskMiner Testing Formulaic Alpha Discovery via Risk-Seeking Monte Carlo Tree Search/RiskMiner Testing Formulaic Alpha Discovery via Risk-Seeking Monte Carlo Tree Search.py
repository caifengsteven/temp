import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
from collections import defaultdict, deque
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy.stats import pearsonr
from tqdm.notebook import tqdm
import copy
import networkx as nx
from torch.distributions import Categorical
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
random.seed(42)

class StockDataGenerator:
    """Generate synthetic stock data for testing purposes"""
    
    def __init__(self, n_stocks=50, n_days=500, price_features=True, volume_features=True):
        """
        Initialize the stock data generator
        
        Args:
            n_stocks: Number of stocks to generate
            n_days: Number of days in the time series
            price_features: Whether to include price features (open, high, low, close)
            volume_features: Whether to include volume features (volume, vwap)
        """
        self.n_stocks = n_stocks
        self.n_days = n_days
        self.price_features = price_features
        self.volume_features = volume_features
        
    def generate_data(self):
        """
        Generate synthetic stock data
        
        Returns:
            DataFrame with synthetic stock data
        """
        # Initialize data structures
        data = {}
        stock_ids = [f"stock_{i}" for i in range(self.n_stocks)]
        dates = pd.date_range(start='2020-01-01', periods=self.n_days)
        
        # Generate basic price trends
        base_prices = {}
        for stock_id in stock_ids:
            # Initial price between 10 and 100
            initial_price = np.random.uniform(10, 100)
            
            # Random drift for each stock (daily expected return)
            drift = np.random.normal(0.0002, 0.0003)  # Small positive drift on average
            
            # Random volatility for each stock
            volatility = np.random.uniform(0.01, 0.03)
            
            # Generate price series using geometric Brownian motion
            price_series = [initial_price]
            for _ in range(self.n_days-1):
                daily_return = np.random.normal(drift, volatility)
                price_series.append(price_series[-1] * (1 + daily_return))
            
            base_prices[stock_id] = price_series
        
        # Create multi-index for the DataFrame
        index = pd.MultiIndex.from_product([dates, stock_ids], names=['datetime', 'instrument'])
        
        # Initialize DataFrame with base prices
        data_dict = {}
        
        # Add price features
        if self.price_features:
            for stock_id, prices in base_prices.items():
                for i, base_price in enumerate(prices):
                    date = dates[i]
                    
                    # Add random noise to create OHLC prices
                    daily_volatility = base_price * np.random.uniform(0.005, 0.02)
                    
                    open_price = base_price * (1 + np.random.normal(0, 0.005))
                    high_price = max(open_price, base_price) * (1 + np.random.uniform(0.001, 0.015))
                    low_price = min(open_price, base_price) * (1 - np.random.uniform(0.001, 0.015))
                    close_price = base_price
                    
                    # Ensure high >= open, close, low and low <= open, close, high
                    high_price = max(high_price, open_price, close_price)
                    low_price = min(low_price, open_price, close_price)
                    
                    # Store in dictionary
                    idx = (date, stock_id)
                    if idx not in data_dict:
                        data_dict[idx] = {}
                    
                    data_dict[idx]['open'] = open_price
                    data_dict[idx]['high'] = high_price
                    data_dict[idx]['low'] = low_price
                    data_dict[idx]['close'] = close_price
        
        # Add volume features
        if self.volume_features:
            for stock_id, prices in base_prices.items():
                for i, base_price in enumerate(prices):
                    date = dates[i]
                    idx = (date, stock_id)
                    
                    # Generate volume based on price and random factors
                    # Higher volume for higher price stocks and more volatile days
                    avg_volume = base_price * 1000 * np.random.uniform(0.5, 1.5)
                    volume = int(avg_volume * (1 + np.random.normal(0, 0.3)))
                    
                    # Calculate VWAP (Volume Weighted Average Price)
                    # Typically between open and close, weighted toward higher volume times
                    if idx in data_dict:
                        open_price = data_dict[idx].get('open', base_price)
                        close_price = data_dict[idx].get('close', base_price)
                        vwap = (open_price + close_price) / 2 + np.random.normal(0, 0.003) * base_price
                    else:
                        data_dict[idx] = {}
                        vwap = base_price
                    
                    data_dict[idx]['volume'] = volume
                    data_dict[idx]['vwap'] = vwap
        
        # Calculate future returns (5-day and 10-day)
        for stock_id in stock_ids:
            for i in range(len(dates) - 10):  # Ensure we have enough future data
                current_date = dates[i]
                future_date_5 = dates[i + 5]
                future_date_10 = dates[i + 10]
                
                current_idx = (current_date, stock_id)
                future_idx_5 = (future_date_5, stock_id)
                future_idx_10 = (future_date_10, stock_id)
                
                if current_idx in data_dict and future_idx_5 in data_dict and future_idx_10 in data_dict:
                    current_price = data_dict[current_idx]['close']
                    future_price_5 = data_dict[future_idx_5]['close']
                    future_price_10 = data_dict[future_idx_10]['close']
                    
                    # Calculate returns
                    return_5d = (future_price_5 / current_price) - 1
                    return_10d = (future_price_10 / current_price) - 1
                    
                    data_dict[current_idx]['return_5d'] = return_5d
                    data_dict[current_idx]['return_10d'] = return_10d
        
        # Convert dictionary to DataFrame
        df = pd.DataFrame.from_dict(data_dict, orient='index')
        
        # Add sector information for cross-sectional operations
        sectors = {}
        num_sectors = 10
        for stock_id in stock_ids:
            sector_id = np.random.randint(0, num_sectors)
            sectors[stock_id] = f"sector_{sector_id}"
        
        # Add sector as a column
        df['sector'] = df.index.get_level_values('instrument').map(sectors)
        
        return df.reset_index()
    

    class AlphaExpression:
    """
    Class for representing and evaluating alpha expressions using Reverse Polish Notation (RPN)
    """
    
    def __init__(self):
        """Initialize the alpha expression"""
        self.PRICE_FEATURES = ['open', 'high', 'low', 'close', 'volume', 'vwap']
        self.TIME_DELTAS = [1, 5, 10, 20, 30]
        self.CONSTANTS = [-10.0, -5.0, -2.0, -1.0, -0.5, 0.5, 1.0, 2.0, 5.0, 10.0]
        
        self.UNARY_OPERATORS = {
            'Sign': lambda x: np.sign(x),
            'Abs': lambda x: np.abs(x),
            'Log': lambda x: np.log(np.abs(x) + 1e-10),
            'CSRank': self.cs_rank,
            'Std': self.std,
            'Mean': self.mean,
            'Sum': self.sum,
            'Max': self.max,
            'Min': self.min,
            'Rank': self.rank,
            'Ref': self.ref
        }
        
        self.BINARY_OPERATORS = {
            'Add': lambda x, y: x + y,
            'Sub': lambda x, y: x - y,
            'Mul': lambda x, y: x * y,
            'Div': lambda x, y: x / (y + 1e-10),
            'Greater': lambda x, y: (x > y).astype(float),
            'Less': lambda x, y: (x < y).astype(float),
            'Corr': self.corr,
            'Cov': self.cov
        }
        
        # Create token set for RPN
        self.token_set = ['BEG', 'END']
        
        # Add operands
        for feature in self.PRICE_FEATURES:
            self.token_set.append(f"${feature}")
        
        for const in self.CONSTANTS:
            self.token_set.append(str(const))
        
        for delta in self.TIME_DELTAS:
            self.token_set.append(str(delta))
        
        # Add operators
        for op in self.UNARY_OPERATORS:
            self.token_set.append(op)
        
        for op in self.BINARY_OPERATORS:
            self.token_set.append(op)
        
        # Create token to index mapping
        self.token_to_idx = {token: i for i, token in enumerate(self.token_set)}
        self.idx_to_token = {i: token for i, token in enumerate(self.token_set)}
        
        # Create action space size
        self.action_space_size = len(self.token_set)
    
    def is_valid_expression(self, tokens):
        """
        Check if the given tokens form a valid RPN expression
        
        Args:
            tokens: List of tokens in RPN format
            
        Returns:
            bool: Whether the expression is valid
        """
        if not tokens or tokens[0] != 'BEG':
            return False
        
        # Remove BEG and END tokens for validation
        if tokens[-1] == 'END':
            tokens = tokens[1:-1]
        else:
            tokens = tokens[1:]
        
        if not tokens:
            return False
        
        stack = []
        
        for token in tokens:
            if token.startswith('$') or token in [str(c) for c in self.CONSTANTS]:
                stack.append(token)  # Operand
            elif token in [str(d) for d in self.TIME_DELTAS]:
                # Time deltas are only valid after a feature and a time-series operator
                if len(stack) < 1:
                    return False
                stack.pop()  # Pop the feature or result
                stack.append("result")  # Push the result
            elif token in self.UNARY_OPERATORS:
                # Unary operators need one operand
                if len(stack) < 1:
                    return False
                stack.pop()  # Pop the operand
                stack.append("result")  # Push the result
            elif token in self.BINARY_OPERATORS:
                # Binary operators need two operands
                if len(stack) < 2:
                    return False
                stack.pop()  # Pop the second operand
                stack.pop()  # Pop the first operand
                stack.append("result")  # Push the result
            else:
                return False  # Unknown token
        
        # A valid expression should leave exactly one item on the stack
        return len(stack) == 1
    
    def evaluate(self, tokens, data):
        """
        Evaluate the alpha expression on the given data
        
        Args:
            tokens: List of tokens in RPN format
            data: DataFrame with stock data
            
        Returns:
            Series with alpha values for each stock and date
        """
        if not self.is_valid_expression(tokens):
            return None
        
        # Remove BEG and END tokens for evaluation
        if tokens[0] == 'BEG':
            tokens = tokens[1:]
        
        if tokens and tokens[-1] == 'END':
            tokens = tokens[:-1]
        
        if not tokens:
            return None
        
        # Prepare data
        df = data.copy()
        df_grouped = df.groupby(['datetime'])
        
        # Create a multi-index DataFrame for easier operations
        df_multi = df.set_index(['datetime', 'instrument'])
        
        # Initialize stack for evaluation
        stack = []
        
        for token in tokens:
            if token.startswith('$'):
                # Price feature
                feature = token[1:]  # Remove $ prefix
                if feature in self.PRICE_FEATURES:
                    stack.append(df_multi[feature])
                else:
                    return None
            elif token in [str(c) for c in self.CONSTANTS]:
                # Constant value
                const_val = float(token)
                stack.append(pd.Series([const_val] * len(df_multi), index=df_multi.index))
            elif token in [str(d) for d in self.TIME_DELTAS]:
                # Time delta for time-series operators
                if not stack:
                    return None
                    
                # The top of the stack should be an operator that needs a time delta
                # This is handled in the operator functions directly
                time_delta = int(token)
                # Keep the time delta on the stack for the next operator
                stack.append(time_delta)
            elif token in self.UNARY_OPERATORS:
                # Unary operator
                if not stack:
                    return None
                
                # Special handling for time-series operators
                if token in ['Std', 'Mean', 'Sum', 'Max', 'Min', 'Rank', 'Ref']:
                    if len(stack) < 2:
                        return None
                    
                    time_delta = stack.pop()
                    if not isinstance(time_delta, int):
                        return None
                    
                    x = stack.pop()
                    result = self.UNARY_OPERATORS[token](x, time_delta)
                else:
                    x = stack.pop()
                    result = self.UNARY_OPERATORS[token](x)
                
                stack.append(result)
            elif token in self.BINARY_OPERATORS:
                # Binary operator
                if len(stack) < 2:
                    return None
                
                # Special handling for time-series operators
                if token in ['Corr', 'Cov']:
                    if len(stack) < 3:
                        return None
                    
                    time_delta = stack.pop()
                    if not isinstance(time_delta, int):
                        return None
                    
                    y = stack.pop()
                    x = stack.pop()
                    result = self.BINARY_OPERATORS[token](x, y, time_delta)
                else:
                    y = stack.pop()
                    x = stack.pop()
                    result = self.BINARY_OPERATORS[token](x, y)
                
                stack.append(result)
        
        # The final result should be the only item on the stack
        if len(stack) != 1:
            return None
        
        result = stack[0]
        
        # Handle NaN and infinite values
        result = result.fillna(0)
        result = result.replace([np.inf, -np.inf], 0)
        
        return result
    
    # Cross-sectional rank
    def cs_rank(self, x):
        """Cross-sectional rank of values across all stocks for each date"""
        return x.groupby(level=0).rank(pct=True)
    
    # Time-series operators
    def std(self, x, window):
        """Standard deviation over a rolling window"""
        return x.groupby(level=1).rolling(window).std().droplevel(0)
    
    def mean(self, x, window):
        """Mean over a rolling window"""
        return x.groupby(level=1).rolling(window).mean().droplevel(0)
    
    def sum(self, x, window):
        """Sum over a rolling window"""
        return x.groupby(level=1).rolling(window).sum().droplevel(0)
    
    def max(self, x, window):
        """Maximum over a rolling window"""
        return x.groupby(level=1).rolling(window).max().droplevel(0)
    
    def min(self, x, window):
        """Minimum over a rolling window"""
        return x.groupby(level=1).rolling(window).min().droplevel(0)
    
    def rank(self, x, window):
        """Rank over a rolling window"""
        # For each stock, rank the current value compared to past values
        result = pd.Series(index=x.index)
        for stock in x.index.get_level_values(1).unique():
            stock_data = x.xs(stock, level=1)
            for i in range(len(stock_data)):
                if i >= window:
                    window_data = stock_data.iloc[i-window+1:i+1]
                    rank = (window_data.rank().iloc[-1] - 1) / (window - 1)  # Normalize to [0, 1]
                    result.loc[(stock_data.index[i], stock)] = rank
                else:
                    result.loc[(stock_data.index[i], stock)] = 0.5  # Default value for insufficient history
        return result
    
    def ref(self, x, days_ago):
        """Reference value from days_ago"""
        return x.groupby(level=1).shift(days_ago)
    
    def corr(self, x, y, window):
        """Correlation between x and y over a rolling window"""
        result = pd.Series(index=x.index)
        for stock in x.index.get_level_values(1).unique():
            stock_x = x.xs(stock, level=1)
            stock_y = y.xs(stock, level=1)
            
            rolling_corr = pd.Series(index=stock_x.index)
            for i in range(len(stock_x)):
                if i >= window:
                    x_window = stock_x.iloc[i-window+1:i+1]
                    y_window = stock_y.iloc[i-window+1:i+1]
                    if x_window.std() > 0 and y_window.std() > 0:
                        corr = x_window.corr(y_window)
                    else:
                        corr = 0
                    rolling_corr.iloc[i] = corr
                else:
                    rolling_corr.iloc[i] = 0
                    
            for idx, val in rolling_corr.items():
                result.loc[(idx, stock)] = val
                
        return result
    
    def cov(self, x, y, window):
        """Covariance between x and y over a rolling window"""
        result = pd.Series(index=x.index)
        for stock in x.index.get_level_values(1).unique():
            stock_x = x.xs(stock, level=1)
            stock_y = y.xs(stock, level=1)
            
            rolling_cov = pd.Series(index=stock_x.index)
            for i in range(len(stock_x)):
                if i >= window:
                    x_window = stock_x.iloc[i-window+1:i+1]
                    y_window = stock_y.iloc[i-window+1:i+1]
                    cov = x_window.cov(y_window)
                    rolling_cov.iloc[i] = cov
                else:
                    rolling_cov.iloc[i] = 0
                    
            for idx, val in rolling_cov.items():
                result.loc[(idx, stock)] = val
                
        return result
    
    def get_random_expression(self, max_length=10):
        """
        Generate a random valid alpha expression
        
        Args:
            max_length: Maximum length of the expression
            
        Returns:
            List of tokens forming a valid RPN expression
        """
        expression = ['BEG']
        stack_size = 0
        
        # Ensure we start with an operand
        operand_tokens = [f"${feature}" for feature in self.PRICE_FEATURES] + [str(c) for c in self.CONSTANTS]
        expression.append(random.choice(operand_tokens))
        stack_size += 1
        
        # Add more tokens
        length = random.randint(3, max_length)
        
        while len(expression) < length:
            if stack_size == 0:
                # Need an operand
                token = random.choice(operand_tokens)
                expression.append(token)
                stack_size += 1
            elif stack_size == 1:
                # Can add an operand, unary operator, or end
                choices = ['operand', 'unary', 'end']
                choice = random.choice(choices)
                
                if choice == 'operand':
                    token = random.choice(operand_tokens)
                    expression.append(token)
                    stack_size += 1
                elif choice == 'unary':
                    # Check if the last token is a feature that needs time delta
                    last_token = expression[-1]
                    if last_token.startswith('$'):
                        time_series_ops = ['Std', 'Mean', 'Sum', 'Max', 'Min', 'Rank', 'Ref']
                        other_ops = list(set(self.UNARY_OPERATORS.keys()) - set(time_series_ops))
                        
                        if random.random() < 0.5:  # 50% chance for time-series operator
                            token = random.choice(time_series_ops)
                            expression.append(token)
                            # Add time delta for time-series operator
                            expression.append(str(random.choice(self.TIME_DELTAS)))
                        else:
                            token = random.choice(other_ops)
                            expression.append(token)
                    else:
                        token = random.choice(list(self.UNARY_OPERATORS.keys()))
                        expression.append(token)
                        
                        # Add time delta if needed
                        if token in ['Std', 'Mean', 'Sum', 'Max', 'Min', 'Rank', 'Ref']:
                            expression.append(str(random.choice(self.TIME_DELTAS)))
                else:  # end
                    if len(expression) >= 3:  # Make sure we have at least one operation
                        expression.append('END')
                        return expression
            else:  # stack_size >= 2
                # Can add a binary operator, unary operator, or end
                choices = ['binary', 'unary', 'operand', 'end']
                choice = random.choice(choices)
                
                if choice == 'binary':
                    # Check if we need a time delta
                    token = random.choice(list(self.BINARY_OPERATORS.keys()))
                    expression.append(token)
                    stack_size -= 1  # Binary operator consumes two operands and produces one
                    
                    # Add time delta if needed
                    if token in ['Corr', 'Cov']:
                        expression.append(str(random.choice(self.TIME_DELTAS)))
                elif choice == 'unary':
                    token = random.choice(list(self.UNARY_OPERATORS.keys()))
                    expression.append(token)
                    
                    # Add time delta if needed
                    if token in ['Std', 'Mean', 'Sum', 'Max', 'Min', 'Rank', 'Ref']:
                        expression.append(str(random.choice(self.TIME_DELTAS)))
                elif choice == 'operand':
                    token = random.choice(operand_tokens)
                    expression.append(token)
                    stack_size += 1
                else:  # end
                    if len(expression) >= 3:  # Make sure we have at least one operation
                        expression.append('END')
                        return expression
        
        # Ensure the expression ends properly
        expression.append('END')
        return expression
    

def calculate_ic(alpha_values, returns):
    """
    Calculate Information Coefficient (IC)
    
    Args:
        alpha_values: Series or array of alpha values
        returns: Series or array of returns
        
    Returns:
        float: IC value
    """
    # Remove NaN values
    mask = ~(np.isnan(alpha_values) | np.isnan(returns))
    alpha_values = alpha_values[mask]
    returns = returns[mask]
    
    if len(alpha_values) == 0 or len(returns) == 0:
        return 0
    
    # Calculate Pearson correlation
    try:
        ic = pearsonr(alpha_values, returns)[0]
        return ic if not np.isnan(ic) else 0
    except:
        return 0

def calculate_rank_ic(alpha_values, returns):
    """
    Calculate Rank Information Coefficient (RankIC)
    
    Args:
        alpha_values: Series or array of alpha values
        returns: Series or array of returns
        
    Returns:
        float: RankIC value
    """
    # Remove NaN values
    mask = ~(np.isnan(alpha_values) | np.isnan(returns))
    alpha_values = alpha_values[mask]
    returns = returns[mask]
    
    if len(alpha_values) == 0 or len(returns) == 0:
        return 0
    
    # Rank the values
    alpha_ranks = pd.Series(alpha_values).rank()
    return_ranks = pd.Series(returns).rank()
    
    # Calculate Pearson correlation of ranks
    try:
        rank_ic = pearsonr(alpha_ranks, return_ranks)[0]
        return rank_ic if not np.isnan(rank_ic) else 0
    except:
        return 0

def calculate_daily_ic(alpha_values, returns, dates):
    """
    Calculate daily IC values
    
    Args:
        alpha_values: Series of alpha values
        returns: Series of returns
        dates: Series of dates
        
    Returns:
        Series: Daily IC values
    """
    daily_ic = {}
    
    for date in dates.unique():
        mask = (dates == date)
        if mask.sum() > 10:  # Ensure we have enough data points
            alpha_day = alpha_values[mask]
            returns_day = returns[mask]
            daily_ic[date] = calculate_ic(alpha_day, returns_day)
    
    return pd.Series(daily_ic)

def calculate_icir(daily_ic):
    """
    Calculate IC Information Ratio (ICIR)
    
    Args:
        daily_ic: Series of daily IC values
        
    Returns:
        float: ICIR value
    """
    if len(daily_ic) == 0:
        return 0
    
    mean_ic = np.mean(daily_ic)
    std_ic = np.std(daily_ic)
    
    if std_ic == 0:
        return 0
    
    return mean_ic / std_ic


class AlphaPool:
    """
    Class for managing a pool of alpha expressions
    """
    
    def __init__(self, max_size=10):
        """
        Initialize the alpha pool
        
        Args:
            max_size: Maximum number of alphas in the pool
        """
        self.max_size = max_size
        self.alpha_expressions = []  # List of token sequences
        self.alpha_values = []  # List of alpha values
        self.weights = []  # List of weights for each alpha
        self.alpha_evaluator = AlphaExpression()
    
    def add_alpha(self, tokens, data):
        """
        Add a new alpha to the pool
        
        Args:
            tokens: List of tokens representing the alpha expression
            data: DataFrame with stock data
            
        Returns:
            bool: Whether the alpha was added successfully
        """
        # Evaluate the new alpha
        alpha_values = self.alpha_evaluator.evaluate(tokens, data)
        
        if alpha_values is None:
            return False
        
        # If the pool is not full, add the alpha
        if len(self.alpha_expressions) < self.max_size:
            self.alpha_expressions.append(tokens)
            self.alpha_values.append(alpha_values)
            
            # Initialize weights equally if this is the first alpha
            if len(self.weights) == 0:
                self.weights = [1.0]
            else:
                # Add a small random weight for the new alpha
                self.weights.append(0.1 * np.random.randn())
                # Normalize weights
                self.weights = [w / sum(abs(w) for w in self.weights) for w in self.weights]
        else:
            # If the pool is full, compare with the alpha with the smallest absolute weight
            min_weight_idx = np.argmin([abs(w) for w in self.weights])
            
            # Update the pool
            self.alpha_expressions[min_weight_idx] = tokens
            self.alpha_values[min_weight_idx] = alpha_values
            
            # Update the weight
            self.weights[min_weight_idx] = 0.1 * np.random.randn()
            # Normalize weights
            self.weights = [w / sum(abs(w) for w in self.weights) for w in self.weights]
        
        # Optimize the weights
        self._optimize_weights(data)
        
        return True
    
    def _optimize_weights(self, data, learning_rate=0.01, n_iterations=100):
        """
        Optimize the weights using gradient descent
        
        Args:
            data: DataFrame with stock data
            learning_rate: Learning rate for gradient descent
            n_iterations: Number of iterations for gradient descent
        """
        if len(self.alpha_expressions) == 0:
            return
        
        # Extract returns
        returns_5d = data.set_index(['datetime', 'instrument'])['return_5d']
        
        # Create design matrix
        X = np.column_stack([values.values for values in self.alpha_values])
        y = returns_5d.values
        
        # Remove NaN values
        mask = ~np.isnan(y)
        X = X[mask]
        y = y[mask]
        
        if len(X) == 0 or len(y) == 0:
            return
        
        # Initialize weights if needed
        if len(self.weights) != X.shape[1]:
            self.weights = np.random.randn(X.shape[1])
            # Normalize weights
            self.weights = [w / sum(abs(w) for w in self.weights) for w in self.weights]
        
        # Gradient descent
        for _ in range(n_iterations):
            # Predict
            y_pred = X @ self.weights
            
            # Compute gradient
            gradient = -2 * X.T @ (y - y_pred) / len(y)
            
            # Update weights
            self.weights = [w - learning_rate * g for w, g in zip(self.weights, gradient)]
            
            # Normalize weights
            self.weights = [w / sum(abs(w) for w in self.weights) for w in self.weights]
    
    def get_composite_alpha(self):
        """
        Get the composite alpha values
        
        Returns:
            Series: Composite alpha values
        """
        if len(self.alpha_expressions) == 0:
            return None
        
        # Compute weighted sum of alpha values
        composite_alpha = sum(w * a for w, a in zip(self.weights, self.alpha_values))
        
        return composite_alpha
    
    def get_composite_ic(self, data):
        """
        Calculate the IC of the composite alpha
        
        Args:
            data: DataFrame with stock data
            
        Returns:
            float: IC value of the composite alpha
        """
        composite_alpha = self.get_composite_alpha()
        
        if composite_alpha is None:
            return 0
        
        # Extract returns
        returns_5d = data.set_index(['datetime', 'instrument'])['return_5d']
        
        # Calculate IC
        ic = calculate_ic(composite_alpha, returns_5d)
        
        return ic
    
    def get_composite_rank_ic(self, data):
        """
        Calculate the RankIC of the composite alpha
        
        Args:
            data: DataFrame with stock data
            
        Returns:
            float: RankIC value of the composite alpha
        """
        composite_alpha = self.get_composite_alpha()
        
        if composite_alpha is None:
            return 0
        
        # Extract returns
        returns_5d = data.set_index(['datetime', 'instrument'])['return_5d']
        
        # Calculate RankIC
        rank_ic = calculate_rank_ic(composite_alpha, returns_5d)
        
        return rank_ic
    
    def get_mutual_ic(self, tokens, data):
        """
        Calculate the mutual IC between a new alpha and the existing alphas in the pool
        
        Args:
            tokens: List of tokens representing the new alpha expression
            data: DataFrame with stock data
            
        Returns:
            float: Average mutual IC value
        """
        if len(self.alpha_expressions) == 0:
            return 0
        
        # Evaluate the new alpha
        new_alpha_values = self.alpha_evaluator.evaluate(tokens, data)
        
        if new_alpha_values is None:
            return 0
        
        # Calculate mutual IC with each alpha in the pool
        mutual_ics = []
        for alpha_values in self.alpha_values:
            mutual_ic = calculate_ic(new_alpha_values, alpha_values)
            mutual_ics.append(abs(mutual_ic))  # Use absolute value as we care about correlation magnitude
        
        # Return average mutual IC
        return np.mean(mutual_ics)
    

class RiskPolicyNetwork(nn.Module):
    """
    Neural network for the risk-seeking policy in MCTS
    """
    
    def __init__(self, input_size, hidden_size, output_size):
        """
        Initialize the policy network
        
        Args:
            input_size: Size of the input (token embedding size)
            hidden_size: Size of the hidden layer
            output_size: Size of the output (action space size)
        """
        super(RiskPolicyNetwork, self).__init__()
        
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)
            
        Returns:
            Tensor: Output tensor of shape (batch_size, output_size)
        """
        # GRU layer
        out, _ = self.gru(x)
        
        # Take the last output
        out = out[:, -1, :]
        
        # Fully connected layer
        out = self.fc(out)
        
        # Apply softmax
        out = F.softmax(out, dim=1)
        
        return out


class MCTSNode:
    """
    Node in the Monte Carlo Tree Search
    """
    
    def __init__(self, state, parent=None, action=None):
        """
        Initialize a node
        
        Args:
            state: Current state (token sequence)
            parent: Parent node
            action: Action that led to this state
        """
        self.state = state
        self.parent = parent
        self.action = action
        self.children = {}  # Maps actions to nodes
        self.visits = 0
        self.value = 0
        self.prior = {}  # Maps actions to prior probabilities
        
    def is_leaf(self):
        """Check if the node is a leaf (has no children)"""
        return len(self.children) == 0
    
    def is_terminal(self):
        """Check if the state is terminal"""
        return len(self.state) > 0 and self.state[-1] == 'END'
    
    def get_value(self):
        """Get the value of the node"""
        if self.visits == 0:
            return 0
        return self.value / self.visits


class RiskMiner:
    """
    Implementation of the RiskMiner framework for alpha discovery
    """
    
    def __init__(self, data_train, data_val, max_depth=20, num_simulations=50, c_puct=1.0, alpha_pool_size=10, 
                 quantile=0.8, lambda_mutIC=0.1, embedding_size=32, hidden_size=64):
        """
        Initialize the RiskMiner
        
        Args:
            data_train: Training data
            data_val: Validation data
            max_depth: Maximum depth of the search tree
            num_simulations: Number of MCTS simulations per search
            c_puct: Exploration constant for PUCT formula
            alpha_pool_size: Maximum size of the alpha pool
            quantile: Quantile for risk-seeking policy (higher values are more risk-seeking)
            lambda_mutIC: Weight for mutual IC in the reward function
            embedding_size: Size of the token embeddings
            hidden_size: Size of the hidden layer in the policy network
        """
        self.data_train = data_train
        self.data_val = data_val
        self.max_depth = max_depth
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.quantile = quantile
        self.lambda_mutIC = lambda_mutIC
        
        # Initialize alpha expression evaluator
        self.alpha_evaluator = AlphaExpression()
        
        # Initialize alpha pool
        self.alpha_pool = AlphaPool(max_size=alpha_pool_size)
        
        # Initialize policy network
        self.token_embedding = nn.Embedding(
            num_embeddings=self.alpha_evaluator.action_space_size,
            embedding_dim=embedding_size
        )
        
        self.policy_network = RiskPolicyNetwork(
            input_size=embedding_size,
            hidden_size=hidden_size,
            output_size=self.alpha_evaluator.action_space_size
        )
        
        self.optimizer = optim.Adam(
            list(self.token_embedding.parameters()) + 
            list(self.policy_network.parameters()),
            lr=0.001
        )
        
        # Initialize replay buffer
        self.replay_buffer = []
        self.replay_buffer_size = 1000
        
        # Initialize quantile estimate
        self.quantile_estimate = 0
        self.quantile_lr = 0.01
        
        # Token to index mapping
        self.token_to_idx = self.alpha_evaluator.token_to_idx
        self.idx_to_token = self.alpha_evaluator.idx_to_token
    
    def get_state_embedding(self, state):
        """
        Get the embedding of a state
        
        Args:
            state: List of tokens
            
        Returns:
            Tensor: Embedding of the state
        """
        if not state:
            # Empty state, return zero embedding
            return torch.zeros(1, 1, self.token_embedding.embedding_dim)
        
        # Convert tokens to indices
        indices = [self.token_to_idx.get(token, 0) for token in state]
        indices_tensor = torch.LongTensor(indices).unsqueeze(0)
        
        # Get embeddings
        embeddings = self.token_embedding(indices_tensor)
        
        return embeddings
    
    def get_policy(self, state):
        """
        Get the policy for a state
        
        Args:
            state: List of tokens
            
        Returns:
            Tensor: Policy distribution over actions
        """
        # Get state embedding
        state_embedding = self.get_state_embedding(state)
        
        # Forward pass through policy network
        policy = self.policy_network(state_embedding)
        
        return policy.squeeze(0)
    
    def select_action(self, state, available_actions=None, temperature=1.0, explore=True):
        """
        Select an action based on the policy
        
        Args:
            state: List of tokens
            available_actions: List of available actions (if None, all actions are available)
            temperature: Temperature for exploration (higher values increase exploration)
            explore: Whether to explore or take the greedy action
            
        Returns:
            str: Selected action
        """
        # Get policy
        policy = self.get_policy(state)
        
        # Apply mask for available actions
        if available_actions is not None:
            mask = torch.zeros_like(policy)
            for action in available_actions:
                action_idx = self.token_to_idx.get(action, 0)
                mask[action_idx] = 1
            policy = policy * mask
            
            # Renormalize
            if policy.sum() > 0:
                policy = policy / policy.sum()
            else:
                # If no action is available, use uniform distribution
                policy = mask / mask.sum()
        
        if explore:
            # Apply temperature
            policy = policy.pow(1 / temperature)
            
            # Renormalize
            if policy.sum() > 0:
                policy = policy / policy.sum()
            
            # Sample from the distribution
            try:
                action_idx = Categorical(policy).sample().item()
            except:
                # If sampling fails, take the most likely action
                action_idx = policy.argmax().item()
        else:
            # Take the most likely action
            action_idx = policy.argmax().item()
        
        return self.idx_to_token[action_idx]
    
    def get_legal_actions(self, state):
        """
        Get the legal actions for a state
        
        Args:
            state: List of tokens
            
        Returns:
            List: Legal actions
        """
        if not state:
            return ['BEG']
        
        if state[-1] == 'END':
            return []  # Terminal state
        
        if state[-1] == 'BEG':
            # After BEG, can only add price features or constants
            return [f"${feature}" for feature in self.alpha_evaluator.PRICE_FEATURES] + \
                   [str(c) for c in self.alpha_evaluator.CONSTANTS]
        
        # Create a temporary state to check if it's valid
        temp_state = state.copy()
        
        # Initialize set of legal actions
        legal_actions = []
        
        # Check if the current expression is valid
        valid_expression = self.alpha_evaluator.is_valid_expression(['BEG'] + temp_state)
        
        # Try each possible next token
        for token in self.alpha_evaluator.token_set:
            if token in ['BEG', 'END']:
                # BEG is only allowed at the beginning
                # END is always allowed unless the expression is not valid
                if token == 'END' and valid_expression:
                    legal_actions.append(token)
                continue
            
            # Add the token to the temporary state
            temp_state_with_token = temp_state + [token]
            
            # Check if the expression is still valid
            if self.alpha_evaluator.is_valid_expression(['BEG'] + temp_state_with_token):
                legal_actions.append(token)
        
        return legal_actions
    
    def get_intermediate_reward(self, state):
        """
        Calculate the intermediate reward for a state
        
        Args:
            state: List of tokens
            
        Returns:
            float: Reward
        """
        # Check if the expression is valid
        if not self.alpha_evaluator.is_valid_expression(['BEG'] + state):
            return 0
        
        # Evaluate the alpha expression
        alpha_values = self.alpha_evaluator.evaluate(['BEG'] + state, self.data_train)
        
        if alpha_values is None:
            return 0
        
        # Extract returns
        returns_5d = self.data_train.set_index(['datetime', 'instrument'])['return_5d']
        
        # Calculate IC
        ic = calculate_ic(alpha_values, returns_5d)
        
        # Calculate mutual IC with existing alphas in the pool
        mut_ic = self.alpha_pool.get_mutual_ic(['BEG'] + state, self.data_train)
        
        # Calculate reward
        reward = ic - self.lambda_mutIC * mut_ic
        
        return reward
    
    def get_terminal_reward(self, state):
        """
        Calculate the terminal reward for a state
        
        Args:
            state: List of tokens
            
        Returns:
            float: Reward
        """
        # Add the alpha to the pool
        added = self.alpha_pool.add_alpha(['BEG'] + state, self.data_train)
        
        if not added:
            return 0
        
        # Calculate the IC of the composite alpha
        composite_ic = self.alpha_pool.get_composite_ic(self.data_train)
        
        return composite_ic
    
    def mcts_search(self, root_state):
        """
        Perform Monte Carlo Tree Search
        
        Args:
            root_state: Root state (token sequence)
            
        Returns:
            MCTSNode: Root node of the search tree
        """
        # Initialize root node
        root = MCTSNode(root_state)
        
        # Perform simulations
        for _ in range(self.num_simulations):
            # Selection and expansion
            node, reward = self._select_and_expand(root)
            
            # Backpropagation
            self._backpropagate(node, reward)
        
        return root
    
    def _select_and_expand(self, node):
        """
        Select a leaf node and expand it
        
        Args:
            node: Root node
            
        Returns:
            Tuple: (Selected node, reward)
        """
        # Selection
        while not node.is_leaf() and not node.is_terminal():
            # Select action using PUCT formula
            action = self._select_action_puct(node)
            
            # Move to the next node
            node = node.children[action]
        
        # If the node is terminal, return it with its reward
        if node.is_terminal():
            reward = self.get_terminal_reward(node.state[1:-1])  # Remove BEG and END
            return node, reward
        
        # Expansion
        # Get legal actions
        legal_actions = self.get_legal_actions(node.state)
        
        # Get policy for the current state
        policy = self.get_policy(node.state).detach().numpy()
        
        # Create children
        for action in legal_actions:
            # Add action to state
            new_state = node.state + [action]
            
            # Create child node
            child = MCTSNode(new_state, parent=node, action=action)
            node.children[action] = child
            
            # Set prior probability
            action_idx = self.token_to_idx.get(action, 0)
            node.prior[action] = policy[action_idx]
        
        # If the node is still a leaf after expansion, return it with its reward
        if node.is_leaf():
            reward = 0
            return node, reward
        
        # Rollout
        rollout_node = self._rollout(node)
        
        # Calculate reward
        if rollout_node.is_terminal():
            reward = self.get_terminal_reward(rollout_node.state[1:-1])  # Remove BEG and END
        else:
            reward = self.get_intermediate_reward(rollout_node.state)
        
        return node, reward
    
    def _select_action_puct(self, node):
        """
        Select an action using the PUCT formula
        
        Args:
            node: Current node
            
        Returns:
            str: Selected action
        """
        # Calculate UCB scores
        ucb_scores = {}
        
        for action, child in node.children.items():
            # PUCT formula
            q_value = child.get_value()
            u_value = self.c_puct * node.prior.get(action, 0) * np.sqrt(node.visits) / (1 + child.visits)
            ucb_scores[action] = q_value + u_value
        
        # Select action with highest UCB score
        return max(ucb_scores.items(), key=lambda x: x[1])[0]
    
    def _rollout(self, node):
        """
        Perform a rollout from a node
        
        Args:
            node: Starting node
            
        Returns:
            MCTSNode: Terminal node
        """
        state = node.state
        
        # Rollout until terminal state or maximum depth
        depth = 0
        while depth < self.max_depth and (not state or state[-1] != 'END'):
            # Get legal actions
            legal_actions = self.get_legal_actions(state)
            
            if not legal_actions:
                break
            
            # Select action based on policy
            action = self.select_action(state, legal_actions, temperature=1.0, explore=True)
            
            # Update state
            state = state + [action]
            
            depth += 1
        
        # Create terminal node
        terminal_node = MCTSNode(state)
        
        return terminal_node
    
    def _backpropagate(self, node, reward):
        """
        Backpropagate the reward through the tree
        
        Args:
            node: Leaf node
            reward: Reward
        """
        # Update node statistics
        while node is not None:
            node.visits += 1
            node.value += reward
            node = node.parent
    
    def update_policy(self, batch_size=32):
        """
        Update the policy network using the replay buffer
        
        Args:
            batch_size: Batch size for training
        """
        if len(self.replay_buffer) < batch_size:
            return
        
        # Sample batch from replay buffer
        batch_indices = np.random.choice(len(self.replay_buffer), batch_size, replace=False)
        batch = [self.replay_buffer[i] for i in batch_indices]
        
        # Extract states, actions, and rewards
        states = [trajectory[0] for trajectory in batch]
        actions = [trajectory[1] for trajectory in batch]
        rewards = [trajectory[2] for trajectory in batch]
        
        # Update quantile estimate
        rewards_tensor = torch.FloatTensor(rewards)
        self.quantile_estimate += self.quantile_lr * (
            (1 - self.quantile) - (rewards_tensor <= self.quantile_estimate).float().mean()
        )
        
        # Filter trajectories based on risk level
        risk_mask = rewards_tensor > self.quantile_estimate
        
        if risk_mask.sum() == 0:
            return
        
        # Extract filtered states and actions
        filtered_states = [states[i] for i in range(len(states)) if risk_mask[i]]
        filtered_actions = [actions[i] for i in range(len(actions)) if risk_mask[i]]
        
        # Convert actions to indices
        action_indices = [self.token_to_idx.get(action, 0) for action in filtered_actions]
        action_indices_tensor = torch.LongTensor(action_indices)
        
        # Get embeddings and policy for each state
        policy_outputs = []
        for state in filtered_states:
            # Get state embedding
            state_embedding = self.get_state_embedding(state)
            
            # Forward pass through policy network
            policy = self.policy_network(state_embedding)
            
            policy_outputs.append(policy.squeeze(0))
        
        policy_outputs_tensor = torch.stack(policy_outputs)
        
        # Calculate loss
        loss = F.cross_entropy(policy_outputs_tensor, action_indices_tensor)
        
        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def train(self, num_episodes=100, update_frequency=10):
        """
        Train the RiskMiner
        
        Args:
            num_episodes: Number of episodes to train
            update_frequency: Frequency of policy updates
            
        Returns:
            Tuple: (Discovered alphas, training history)
        """
        history = {
            'episode_rewards': [],
            'composite_ic': [],
            'composite_rank_ic': []
        }
        
        for episode in tqdm(range(num_episodes), desc="Training"):
            # Initialize state
            state = []
            
            # Perform MCTS search
            root = self.mcts_search(state)
            
            # Select action with highest visit count
            action_visits = {action: child.visits for action, child in root.children.items()}
            selected_action = max(action_visits.items(), key=lambda x: x[1])[0]
            
            # Initialize trajectory
            trajectory = []
            
            # Follow the selected path
            current_node = root
            episode_reward = 0
            
            while current_node is not None and not current_node.is_terminal():
                state = current_node.state
                
                # Select action using MCTS tree
                if current_node.children:
                    action_visits = {action: child.visits for action, child in current_node.children.items()}
                    action = max(action_visits.items(), key=lambda x: x[1])[0]
                else:
                    # Get legal actions
                    legal_actions = self.get_legal_actions(state)
                    
                    if not legal_actions:
                        break
                    
                    # Select action based on policy
                    action = self.select_action(state, legal_actions, temperature=1.0, explore=True)
                
                # Calculate intermediate reward
                reward = self.get_intermediate_reward(state + [action])
                episode_reward += reward
                
                # Store transition in trajectory
                trajectory.append((state, action, reward))
                
                # Move to next node
                if action in current_node.children:
                    current_node = current_node.children[action]
                else:
                    # Create new node
                    new_state = state + [action]
                    new_node = MCTSNode(new_state, parent=current_node, action=action)
                    current_node.children[action] = new_node
                    current_node = new_node
            
            # Calculate terminal reward if the episode ended with a terminal state
            if current_node is not None and current_node.is_terminal():
                terminal_reward = self.get_terminal_reward(current_node.state[1:-1])  # Remove BEG and END
                episode_reward += terminal_reward
                
                # Store the final transition
                trajectory.append((current_node.state[:-1], 'END', terminal_reward))
            
            # Add trajectory to replay buffer
            self.replay_buffer.extend(trajectory)
            
            # Limit replay buffer size
            if len(self.replay_buffer) > self.replay_buffer_size:
                self.replay_buffer = self.replay_buffer[-self.replay_buffer_size:]
            
            # Update policy network
            if episode % update_frequency == 0:
                self.update_policy()
            
            # Store episode reward
            history['episode_rewards'].append(episode_reward)
            
            # Evaluate the composite alpha
            composite_ic = self.alpha_pool.get_composite_ic(self.data_val)
            composite_rank_ic = self.alpha_pool.get_composite_rank_ic(self.data_val)
            
            history['composite_ic'].append(composite_ic)
            history['composite_rank_ic'].append(composite_rank_ic)
        
        # Return discovered alphas and training history
        return self.alpha_pool.alpha_expressions, history


def backtest_alphas(alpha_expressions, data, alpha_evaluator, top_k=10, transaction_cost=0.001):
    """
    Backtest the alpha expressions
    
    Args:
        alpha_expressions: List of alpha expressions
        data: DataFrame with stock data
        alpha_evaluator: Alpha expression evaluator
        top_k: Number of top stocks to select
        transaction_cost: Transaction cost as a fraction of position value
        
    Returns:
        DataFrame: Backtest results
    """
    # Evaluate each alpha
    alpha_values = []
    
    for tokens in alpha_expressions:
        values = alpha_evaluator.evaluate(tokens, data)
        if values is not None:
            alpha_values.append(values)
    
    if not alpha_values:
        return None
    
    # Create a composite alpha
    composite_alpha = sum(values for values in alpha_values) / len(alpha_values)
    
    # Initialize portfolio
    dates = data['datetime'].unique()
    portfolio_values = []
    positions = {}
    cash = 1.0  # Start with $1
    
    # Backtest
    for i, date in enumerate(dates[:-5]):  # Skip last 5 days as we need future returns
        # Get current stocks and prices
        day_data = data[data['datetime'] == date]
        
        if i % 5 == 0:  # Rebalance every 5 days
            # Get alpha values for the current date
            day_alpha = composite_alpha.xs(date, level=0)
            
            # Rank stocks by alpha value
            ranked_stocks = day_alpha.sort_values(ascending=False)
            
            # Select top stocks
            top_stocks = ranked_stocks.index[:top_k].tolist()
            
            # Liquidate current positions
            for stock, shares in list(positions.items()):
                if stock in day_data['instrument'].values:
                    stock_price = day_data[day_data['instrument'] == stock]['close'].values[0]
                    cash += shares * stock_price * (1 - transaction_cost)
                    del positions[stock]
            
            # Allocate cash to top stocks
            cash_per_stock = cash / len(top_stocks)
            for stock in top_stocks:
                if stock in day_data['instrument'].values:
                    stock_price = day_data[day_data['instrument'] == stock]['close'].values[0]
                    shares = cash_per_stock / stock_price
                    positions[stock] = shares
                    cash -= cash_per_stock
        
        # Calculate portfolio value
        portfolio_value = cash
        for stock, shares in positions.items():
            if stock in day_data['instrument'].values:
                stock_price = day_data[day_data['instrument'] == stock]['close'].values[0]
                portfolio_value += shares * stock_price
        
        portfolio_values.append(portfolio_value)
    
    # Calculate returns
    returns = [0]
    for i in range(1, len(portfolio_values)):
        returns.append((portfolio_values[i] - portfolio_values[i-1]) / portfolio_values[i-1])
    
    # Calculate cumulative returns
    cumulative_returns = np.cumprod(1 + np.array(returns)) - 1
    
    # Calculate performance metrics
    sharpe_ratio = np.mean(returns[1:]) / np.std(returns[1:]) * np.sqrt(252)  # Annualized
    max_drawdown = np.max(np.maximum.accumulate(portfolio_values) - portfolio_values) / np.maximum.accumulate(portfolio_values)
    
    # Create results DataFrame
    results = pd.DataFrame({
        'date': dates[:len(portfolio_values)],
        'portfolio_value': portfolio_values,
        'return': returns,
        'cumulative_return': cumulative_returns
    })
    
    # Add performance metrics
    results.attrs['sharpe_ratio'] = sharpe_ratio
    results.attrs['max_drawdown'] = max_drawdown
    results.attrs['final_return'] = cumulative_returns[-1]
    
    return results


def main():
    """
    Main function to run the experiment
    """
    # Generate synthetic stock data
    print("Generating synthetic stock data...")
    data_generator = StockDataGenerator(n_stocks=50, n_days=600)
    data = data_generator.generate_data()
    
    # Split data into train, validation, and test sets
    train_end_date = pd.Timestamp('2020-10-01')
    val_end_date = pd.Timestamp('2021-01-01')
    
    data_train = data[data['datetime'] < train_end_date]
    data_val = data[(data['datetime'] >= train_end_date) & (data['datetime'] < val_end_date)]
    data_test = data[data['datetime'] >= val_end_date]
    
    print(f"Data split: Train={len(data_train['datetime'].unique())} days, "
          f"Validation={len(data_val['datetime'].unique())} days, "
          f"Test={len(data_test['datetime'].unique())} days")
    
    # Initialize RiskMiner
    print("Initializing RiskMiner...")
    risk_miner = RiskMiner(
        data_train=data_train,
        data_val=data_val,
        max_depth=20,
        num_simulations=50,
        c_puct=1.0,
        alpha_pool_size=10,
        quantile=0.8,
        lambda_mutIC=0.1,
        embedding_size=32,
        hidden_size=64
    )
    
    # Train RiskMiner
    print("Training RiskMiner...")
    alphas, history = risk_miner.train(num_episodes=50, update_frequency=5)
    
    # Evaluate discovered alphas
    print("Evaluating discovered alphas...")
    alpha_evaluator = AlphaExpression()
    
    # Print alpha expressions
    print("\nDiscovered Alpha Expressions:")
    for i, tokens in enumerate(alphas):
        print(f"Alpha {i+1}: {' '.join(tokens)}")
    
    # Evaluate alphas on test set
    print("\nAlpha Performance on Test Set:")
    
    # Calculate IC for each alpha
    ic_values = []
    rank_ic_values = []
    
    for tokens in alphas:
        # Evaluate alpha on test set
        alpha_values = alpha_evaluator.evaluate(tokens, data_test)
        
        if alpha_values is not None:
            # Extract returns
            returns_5d = data_test.set_index(['datetime', 'instrument'])['return_5d']
            
            # Calculate IC
            ic = calculate_ic(alpha_values, returns_5d)
            ic_values.append(ic)
            
            # Calculate RankIC
            rank_ic = calculate_rank_ic(alpha_values, returns_5d)
            rank_ic_values.append(rank_ic)
            
            print(f"Alpha: {' '.join(tokens)}")
            print(f"  IC: {ic:.4f}")
            print(f"  RankIC: {rank_ic:.4f}")
    
    # Calculate composite alpha performance
    composite_alpha = None
    
    for tokens in alphas:
        values = alpha_evaluator.evaluate(tokens, data_test)
        if values is not None:
            if composite_alpha is None:
                composite_alpha = values
            else:
                composite_alpha += values
    
    if composite_alpha is not None:
        composite_alpha /= len(alphas)
        
        # Extract returns
        returns_5d = data_test.set_index(['datetime', 'instrument'])['return_5d']
        
        # Calculate composite IC
        composite_ic = calculate_ic(composite_alpha, returns_5d)
        
        # Calculate composite RankIC
        composite_rank_ic = calculate_rank_ic(composite_alpha, returns_5d)
        
        print("\nComposite Alpha Performance:")
        print(f"  IC: {composite_ic:.4f}")
        print(f"  RankIC: {composite_rank_ic:.4f}")
    
    # Backtest alphas
    print("\nBacktesting Alphas...")
    backtest_results = backtest_alphas(
        alpha_expressions=alphas,
        data=data_test,
        alpha_evaluator=alpha_evaluator,
        top_k=10,
        transaction_cost=0.001
    )
    
    if backtest_results is not None:
        print("Backtest Results:")
        print(f"  Final Return: {backtest_results.attrs['final_return']:.4f}")
        print(f"  Sharpe Ratio: {backtest_results.attrs['sharpe_ratio']:.4f}")
        print(f"  Max Drawdown: {backtest_results.attrs['max_drawdown']:.4f}")
        
        # Plot backtest results
        plt.figure(figsize=(12, 8))
        
        # Plot cumulative return
        plt.subplot(2, 1, 1)
        plt.plot(backtest_results['date'], backtest_results['cumulative_return'])
        plt.title('Cumulative Return')
        plt.xlabel('Date')
        plt.ylabel('Return')
        plt.grid(True)
        
        # Plot portfolio value
        plt.subplot(2, 1, 2)
        plt.plot(backtest_results['date'], backtest_results['portfolio_value'])
        plt.title('Portfolio Value')
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('backtest_results.png')
        plt.show()
    
    # Plot training history
    plt.figure(figsize=(12, 8))
    
    # Plot episode rewards
    plt.subplot(3, 1, 1)
    plt.plot(history['episode_rewards'])
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.grid(True)
    
    # Plot composite IC
    plt.subplot(3, 1, 2)
    plt.plot(history['composite_ic'])
    plt.title('Composite IC')
    plt.xlabel('Episode')
    plt.ylabel('IC')
    plt.grid(True)
    
    # Plot composite RankIC
    plt.subplot(3, 1, 3)
    plt.plot(history['composite_rank_ic'])
    plt.title('Composite RankIC')
    plt.xlabel('Episode')
    plt.ylabel('RankIC')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()
    
    # Compare with random alphas
    print("\nComparing with Random Alphas...")
    
    # Generate random alphas
    random_alphas = []
    for _ in range(10):
        random_expression = alpha_evaluator.get_random_expression()
        random_alphas.append(random_expression)
    
    # Evaluate random alphas on test set
    random_ic_values = []
    random_rank_ic_values = []
    
    for tokens in random_alphas:
        # Evaluate alpha on test set
        alpha_values = alpha_evaluator.evaluate(tokens, data_test)
        
        if alpha_values is not None:
            # Extract returns
            returns_5d = data_test.set_index(['datetime', 'instrument'])['return_5d']
            
            # Calculate IC
            ic = calculate_ic(alpha_values, returns_5d)
            random_ic_values.append(ic)
            
            # Calculate RankIC
            rank_ic = calculate_rank_ic(alpha_values, returns_5d)
            random_rank_ic_values.append(rank_ic)
    
    # Calculate average IC and RankIC
    if random_ic_values:
        avg_random_ic = np.mean(random_ic_values)
        avg_random_rank_ic = np.mean(random_rank_ic_values)
        
        print("Random Alphas Performance:")
        print(f"  Average IC: {avg_random_ic:.4f}")
        print(f"  Average RankIC: {avg_random_rank_ic:.4f}")
        
        # Backtest random alphas
        random_backtest_results = backtest_alphas(
            alpha_expressions=random_alphas,
            data=data_test,
            alpha_evaluator=alpha_evaluator,
            top_k=10,
            transaction_cost=0.001
        )
        
        if random_backtest_results is not None:
            print("Random Alphas Backtest Results:")
            print(f"  Final Return: {random_backtest_results.attrs['final_return']:.4f}")
            print(f"  Sharpe Ratio: {random_backtest_results.attrs['sharpe_ratio']:.4f}")
            print(f"  Max Drawdown: {random_backtest_results.attrs['max_drawdown']:.4f}")
            
            # Plot comparison
            plt.figure(figsize=(12, 6))
            plt.plot(backtest_results['date'], backtest_results['cumulative_return'], label='RiskMiner')
            plt.plot(random_backtest_results['date'], random_backtest_results['cumulative_return'], label='Random')
            plt.title('Cumulative Return Comparison')
            plt.xlabel('Date')
            plt.ylabel('Return')
            plt.legend()
            plt.grid(True)
            plt.savefig('return_comparison.png')
            plt.show()
    
    # Compare IC and RankIC distributions
    if ic_values and random_ic_values:
        plt.figure(figsize=(12, 6))
        
        # Plot IC comparison
        plt.subplot(1, 2, 1)
        plt.boxplot([ic_values, random_ic_values], labels=['RiskMiner', 'Random'])
        plt.title('IC Comparison')
        plt.ylabel('IC')
        plt.grid(True)
        
        # Plot RankIC comparison
        plt.subplot(1, 2, 2)
        plt.boxplot([rank_ic_values, random_rank_ic_values], labels=['RiskMiner', 'Random'])
        plt.title('RankIC Comparison')
        plt.ylabel('RankIC')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('ic_comparison.png')
        plt.show()

if __name__ == "__main__":
    main()


