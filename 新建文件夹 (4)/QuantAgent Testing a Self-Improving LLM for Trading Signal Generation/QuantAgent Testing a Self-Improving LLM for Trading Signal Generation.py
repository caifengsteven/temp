import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import random
from datetime import datetime, timedelta
from sklearn.metrics import mean_squared_error
import xgboost as xgb
from scipy.stats import pearsonr
import json
import re
import os
from collections import defaultdict

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

# Create directory for storing knowledge base if it doesn't exist
if not os.path.exists('kb'):
    os.makedirs('kb')


def generate_stock_data(n_stocks=500, n_days=252, start_date='2023-01-01'):
    """
    Generate simulated stock market data.
    
    Parameters:
    -----------
    n_stocks : int
        Number of stocks to simulate
    n_days : int
        Number of trading days to simulate
    start_date : str
        Start date in 'YYYY-MM-DD' format
    
    Returns:
    --------
    DataFrame containing simulated stock data
    """
    # Generate dates
    start_date = datetime.strptime(start_date, '%Y-%m-%d')
    dates = [start_date + timedelta(days=i) for i in range(n_days)]
    # Keep only weekdays
    dates = [date for date in dates if date.weekday() < 5]
    n_days = len(dates)
    
    # Generate stock IDs
    stock_ids = [f'STOCK_{i:03d}' for i in range(n_stocks)]
    
    # Initialize data storage
    data = {
        'date': [],
        'stock_id': [],
        'open': [],
        'high': [],
        'low': [],
        'close': [],
        'pre_close': [],
        'volume': [],
        'value': [],
        'shares': [],
        'turnover': [],
        'tradenum': [],
        'vwap': [],
        'return': []
    }
    
    # Generate data for each stock
    for stock_id in stock_ids:
        # Generate price trend with random walk and some momentum
        trend = np.random.normal(0, 0.0002, n_days)
        momentum = np.random.normal(0, 0.001, n_days)
        for i in range(1, n_days):
            momentum[i] = 0.8 * momentum[i-1] + 0.2 * momentum[i]
            trend[i] = trend[i-1] + trend[i] + momentum[i]
        
        # Generate price data
        base_price = np.random.uniform(10, 100)
        price_multiplier = np.exp(trend)
        close_prices = base_price * price_multiplier
        
        # Add some volatility to high/low/open
        for i in range(n_days):
            # For each date
            data['date'].append(dates[i])
            data['stock_id'].append(stock_id)
            
            # Calculate prices
            close = close_prices[i]
            daily_vol = close * np.random.uniform(0.01, 0.03)
            high = close + np.random.uniform(0, daily_vol)
            low = close - np.random.uniform(0, daily_vol)
            open_price = low + np.random.uniform(0, high - low)
            
            # Pre-close (previous day's close)
            pre_close = close_prices[i-1] if i > 0 else close
            
            # Calculate return
            ret = (close / pre_close) - 1 if i > 0 else 0
            
            # Generate volume and related metrics
            volume = np.random.lognormal(mean=np.log(1000000), sigma=0.5)
            # Higher volume when price moves more
            volume *= (1 + 5 * abs(ret))
            
            shares = np.random.uniform(50000000, 500000000)
            turnover = volume / shares
            vwap = close * (1 + np.random.normal(0, 0.005))
            tradenum = int(volume / np.random.uniform(500, 2000))
            value = volume * vwap
            
            # Store data
            data['open'].append(open_price)
            data['high'].append(high)
            data['low'].append(low)
            data['close'].append(close)
            data['pre_close'].append(pre_close)
            data['volume'].append(volume)
            data['value'].append(value)
            data['shares'].append(shares)
            data['turnover'].append(turnover)
            data['tradenum'].append(tradenum)
            data['vwap'].append(vwap)
            data['return'].append(ret)
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    return df

# Generate simulated stock data
stock_data = generate_stock_data()

# Reshape data for easier access
def reshape_data_for_signals(df):
    """
    Reshape the data into a format suitable for signal calculation:
    Each feature has a DataFrame with dates as index and stocks as columns.
    """
    features = {}
    for feature in df.columns:
        if feature not in ['date', 'stock_id']:
            pivot = df.pivot(index='date', columns='stock_id', values=feature)
            features[feature] = pivot
    return features

# Reshape the data
features = reshape_data_for_signals(stock_data)


class Factor:
    """
    Base class for implementing trading signals/alphas.
    """
    def __init__(self):
        self.name = "BaseFactor"
        self.window_length = 1
        self.inputs = []
        
    def calc(self, data):
        """
        Calculate the trading signal.
        
        Parameters:
        -----------
        data : dict
            Dictionary containing DataFrames for each required input.
            Keys correspond to self.inputs.
        
        Returns:
        --------
        DataFrame
            Signal values for each stock and date.
        """
        raise NotImplementedError("Subclasses must implement calc method")
    
    def run(self, features):
        """
        Run the signal calculation on the provided features.
        
        Parameters:
        -----------
        features : dict
            Dictionary of DataFrames containing features data.
        
        Returns:
        --------
        DataFrame
            Signal values for each stock and date.
        """
        # Prepare data inputs
        data = {}
        for input_name in self.inputs:
            if input_name in features:
                data[input_name] = features[input_name]
            else:
                raise ValueError(f"Required input '{input_name}' not found in features")
        
        # Calculate signal
        signal = self.calc(data)
        return signal
    
def calculate_ic(signal, future_returns):
    """
    Calculate Information Coefficient (IC) - the correlation between signal and future returns.
    
    Parameters:
    -----------
    signal : DataFrame
        Signal values for each stock and date.
    future_returns : DataFrame
        Future returns for each stock and date.
    
    Returns:
    --------
    float
        Average IC across all dates.
    """
    # Align dates
    common_dates = signal.index.intersection(future_returns.index)
    if len(common_dates) == 0:
        return 0
    
    signal = signal.loc[common_dates]
    future_returns = future_returns.loc[common_dates]
    
    # Calculate IC for each date
    ics = []
    for date in common_dates:
        sig = signal.loc[date].dropna()
        ret = future_returns.loc[date].reindex(sig.index).dropna()
        
        # Ensure we have enough valid data points
        common_stocks = sig.index.intersection(ret.index)
        if len(common_stocks) < 10:  # Require at least 10 stocks for meaningful correlation
            continue
            
        # Calculate correlation
        sig = sig.loc[common_stocks]
        ret = ret.loc[common_stocks]
        
        # Skip if there's no variation in signal
        if sig.std() == 0:
            continue
            
        corr, _ = pearsonr(sig, ret)
        if not np.isnan(corr):
            ics.append(corr)
    
    # Return average IC
    return np.mean(ics) if ics else 0

def calculate_sharpe(signal, future_returns):
    """
    Calculate Sharpe ratio of a signal-weighted portfolio.
    
    Parameters:
    -----------
    signal : DataFrame
        Signal values for each stock and date.
    future_returns : DataFrame
        Future returns for each stock and date.
    
    Returns:
    --------
    float
        Sharpe ratio.
    """
    # Align dates
    common_dates = signal.index.intersection(future_returns.index)
    if len(common_dates) == 0:
        return 0
    
    signal = signal.loc[common_dates]
    future_returns = future_returns.loc[common_dates]
    
    # Calculate portfolio returns
    portfolio_returns = []
    
    for date in common_dates[:-1]:  # Skip last date as we don't have future returns
        sig = signal.loc[date].dropna()
        
        # Next date's returns
        next_date_idx = common_dates.get_loc(date) + 1
        if next_date_idx >= len(common_dates):
            continue
        next_date = common_dates[next_date_idx]
        next_ret = future_returns.loc[next_date].reindex(sig.index).dropna()
        
        # Ensure we have enough valid data points
        common_stocks = sig.index.intersection(next_ret.index)
        if len(common_stocks) < 10:
            continue
            
        # Normalize signal to create weights (zero mean, unit variance)
        weights = sig.loc[common_stocks]
        if weights.std() == 0:
            continue
        weights = (weights - weights.mean()) / weights.std()
        
        # Calculate portfolio return (weighted average of stock returns)
        port_ret = (weights * next_ret.loc[common_stocks]).sum() / abs(weights).sum()
        portfolio_returns.append(port_ret)
    
    # Calculate Sharpe ratio (annualized)
    if not portfolio_returns:
        return 0
    
    returns_series = pd.Series(portfolio_returns)
    sharpe = np.sqrt(252) * returns_series.mean() / returns_series.std() if returns_series.std() > 0 else 0
    return sharpe

def calculate_returns(signal, future_returns):
    """
    Calculate cumulative returns of a signal-weighted portfolio.
    
    Parameters:
    -----------
    signal : DataFrame
        Signal values for each stock and date.
    future_returns : DataFrame
        Future returns for each stock and date.
    
    Returns:
    --------
    float
        Cumulative return.
    """
    # Align dates
    common_dates = signal.index.intersection(future_returns.index)
    if len(common_dates) == 0:
        return 0
    
    signal = signal.loc[common_dates]
    future_returns = future_returns.loc[common_dates]
    
    # Calculate portfolio returns
    portfolio_returns = []
    
    for date in common_dates[:-1]:
        sig = signal.loc[date].dropna()
        
        # Next date's returns
        next_date_idx = common_dates.get_loc(date) + 1
        if next_date_idx >= len(common_dates):
            continue
        next_date = common_dates[next_date_idx]
        next_ret = future_returns.loc[next_date].reindex(sig.index).dropna()
        
        # Ensure we have enough valid data points
        common_stocks = sig.index.intersection(next_ret.index)
        if len(common_stocks) < 10:
            continue
            
        # Normalize signal to create weights
        weights = sig.loc[common_stocks]
        if weights.std() == 0:
            continue
        weights = (weights - weights.mean()) / weights.std()
        
        # Calculate portfolio return
        port_ret = (weights * next_ret.loc[common_stocks]).sum() / abs(weights).sum()
        portfolio_returns.append(port_ret)
    
    # Calculate cumulative return
    if not portfolio_returns:
        return 0
    
    cumulative_return = (1 + pd.Series(portfolio_returns)).prod() - 1
    return cumulative_return

def evaluate_signal(signal, future_returns):
    """
    Evaluate a trading signal's performance.
    
    Parameters:
    -----------
    signal : DataFrame
        Signal values for each stock and date.
    future_returns : DataFrame
        Future returns for each stock and date.
    
    Returns:
    --------
    dict
        Performance metrics.
    """
    # Basic validation checks
    if signal.empty:
        return {
            'ic': 0,
            'sharpe': 0,
            'return': 0,
            'valid_ratio': 0
        }
    
    # Calculate valid ratio (non-NaN values)
    valid_ratio = signal.count().sum() / (signal.shape[0] * signal.shape[1])
    
    # Calculate performance metrics
    ic = calculate_ic(signal, future_returns)
    sharpe = calculate_sharpe(signal, future_returns)
    cumulative_return = calculate_returns(signal, future_returns)
    
    return {
        'ic': ic,
        'sharpe': sharpe,
        'return': cumulative_return,
        'valid_ratio': valid_ratio
    }

def save_signal_to_kb(signal_info, kb_path='kb'):
    """
    Save a signal to the knowledge base.
    
    Parameters:
    -----------
    signal_info : dict
        Information about the signal including code, performance, and reviews.
    kb_path : str
        Path to the knowledge base directory.
    
    Returns:
    --------
    str
        Path to the saved signal file.
    """
    # Create a unique ID for the signal
    signal_id = f"{signal_info['name']}_{int(time.time())}"
    
    # Save to a JSON file
    file_path = os.path.join(kb_path, f"{signal_id}.json")
    with open(file_path, 'w') as f:
        json.dump(signal_info, f, indent=2)
    
    return file_path

def load_kb(kb_path='kb'):
    """
    Load all signals from the knowledge base.
    
    Parameters:
    -----------
    kb_path : str
        Path to the knowledge base directory.
    
    Returns:
    --------
    list
        List of signal information dictionaries.
    """
    kb = []
    
    if not os.path.exists(kb_path):
        return kb
    
    for filename in os.listdir(kb_path):
        if filename.endswith('.json'):
            file_path = os.path.join(kb_path, filename)
            try:
                with open(file_path, 'r') as f:
                    signal_info = json.load(f)
                    kb.append(signal_info)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
    
    return kb

def retrieve_from_kb(query, kb, top_k=3):
    """
    Retrieve relevant signals from the knowledge base based on a query.
    In a real implementation, this would use embedding similarity.
    For this demo, we'll use simple keyword matching.
    
    Parameters:
    -----------
    query : str
        The query to search for.
    kb : list
        The knowledge base of signals.
    top_k : int
        Number of results to return.
    
    Returns:
    --------
    list
        List of relevant signal information.
    """
    if not kb:
        return []
    
    # Simple keyword matching (in a real implementation, use embeddings)
    query_words = set(query.lower().split())
    
    results = []
    for signal in kb:
        # Search in trading idea, reviews, and code
        text_to_search = f"{signal.get('trading_idea', '')} {signal.get('reviews', '')} {signal.get('code', '')}"
        text_to_search = text_to_search.lower()
        
        # Count matching words
        match_count = sum(1 for word in query_words if word in text_to_search)
        
        # Add to results if there are matches
        if match_count > 0:
            results.append((match_count, signal))
    
    # Sort by match count (descending)
    results.sort(reverse=True, key=lambda x: x[0])
    
    # Return top_k results
    return [signal for _, signal in results[:top_k]]

def simulated_llm_writer(trading_idea, kb_hints, previous_reviews=None):
    """
    Simulate an LLM writer that generates a trading signal based on a trading idea and knowledge hints.
    
    Parameters:
    -----------
    trading_idea : str
        Description of the trading idea.
    kb_hints : list
        List of relevant signals from the knowledge base.
    previous_reviews : list
        List of previous reviews (for iterative improvement).
    
    Returns:
    --------
    str
        Generated signal code.
    """
    # For simulation purposes, we'll create a few signal templates
    templates = [
        # Moving average crossover template
        """
class MovingAverageCrossover(Factor):
    def __init__(self):
        self.name = "MovingAverageCrossover"
        self.window_length = {fast_window} + {slow_window}
        self.inputs = ["close"]
        
    def calc(self, data):
        # Calculate fast and slow moving averages
        fast_ma = data['close'].rolling(window={fast_window}).mean()
        slow_ma = data['close'].rolling(window={slow_window}).mean()
        
        # Generate crossover signal
        signal = (fast_ma > slow_ma).astype(float)
        
        # Add signal strength based on the difference between MAs
        signal = signal * (fast_ma - slow_ma) / slow_ma
        
        return signal
        """,
        
        # Momentum template
        """
class MomentumSignal(Factor):
    def __init__(self):
        self.name = "MomentumSignal"
        self.window_length = {lookback} + 1
        self.inputs = ["close"]
        
    def calc(self, data):
        # Calculate returns over lookback period
        returns = data['close'] / data['close'].shift({lookback}) - 1
        
        # Normalize returns
        normalized_returns = (returns - returns.mean()) / returns.std()
        
        return normalized_returns
        """,
        
        # Volume-based template
        """
class VolumeSignal(Factor):
    def __init__(self):
        self.name = "VolumeSignal"
        self.window_length = {lookback}
        self.inputs = ["volume", "close"]
        
    def calc(self, data):
        # Calculate volume change
        volume_change = data['volume'] / data['volume'].rolling(window={lookback}).mean()
        
        # Calculate price change
        price_change = data['close'] / data['close'].shift(1) - 1
        
        # Generate signal based on volume and price
        signal = volume_change * np.sign(price_change)
        
        return signal
        """,
        
        # Volatility-based template
        """
class VolatilitySignal(Factor):
    def __init__(self):
        self.name = "VolatilitySignal"
        self.window_length = {lookback}
        self.inputs = ["high", "low", "close"]
        
    def calc(self, data):
        # Calculate volatility (standard deviation of returns)
        returns = data['close'] / data['close'].shift(1) - 1
        volatility = returns.rolling(window={lookback}).std()
        
        # Calculate true range
        true_range = np.maximum(
            data['high'] - data['low'],
            np.maximum(
                abs(data['high'] - data['close'].shift(1)),
                abs(data['low'] - data['close'].shift(1))
            )
        )
        
        # Normalize true range
        normalized_tr = true_range / true_range.rolling(window={lookback}).mean()
        
        # Generate signal (higher volatility -> lower signal)
        signal = 1 / (volatility * normalized_tr)
        
        return signal
        """
    ]
    
    # Choose template based on the trading idea and knowledge hints
    if "moving average" in trading_idea.lower() or "crossover" in trading_idea.lower():
        template_idx = 0
        params = {
            "fast_window": random.randint(5, 20),
            "slow_window": random.randint(30, 100)
        }
    elif "momentum" in trading_idea.lower() or "trend" in trading_idea.lower():
        template_idx = 1
        params = {
            "lookback": random.randint(5, 50)
        }
    elif "volume" in trading_idea.lower():
        template_idx = 2
        params = {
            "lookback": random.randint(5, 30)
        }
    else:
        # Default to volatility
        template_idx = 3
        params = {
            "lookback": random.randint(10, 30)
        }
    
    # If we have knowledge hints, try to adapt parameters
    if kb_hints:
        # Extract parameters from similar signals
        for hint in kb_hints:
            if "window" in hint.get("code", ""):
                window_match = re.search(r'window=(\d+)', hint.get("code", ""))
                if window_match and "lookback" in params:
                    params["lookback"] = int(window_match.group(1))
            if "fast_window" in hint.get("code", ""):
                fast_match = re.search(r'fast_window=(\d+)', hint.get("code", ""))
                if fast_match and "fast_window" in params:
                    params["fast_window"] = int(fast_match.group(1))
            if "slow_window" in hint.get("code", ""):
                slow_match = re.search(r'slow_window=(\d+)', hint.get("code", ""))
                if slow_match and "slow_window" in params:
                    params["slow_window"] = int(slow_match.group(1))
    
    # If we have previous reviews, adjust the code based on feedback
    if previous_reviews:
        for review in previous_reviews:
            # Look for suggestions to adjust parameters
            if "increase" in review.lower() and "window" in review.lower():
                for param in params:
                    if "window" in param:
                        params[param] = int(params[param] * 1.5)
            elif "decrease" in review.lower() and "window" in review.lower():
                for param in params:
                    if "window" in param:
                        params[param] = max(2, int(params[param] * 0.7))
    
    # Fill in the template with parameters
    code = templates[template_idx].format(**params)
    
    return code

def simulated_llm_judge(code, trading_idea, kb_hints):
    """
    Simulate an LLM judge that evaluates a trading signal.
    
    Parameters:
    -----------
    code : str
        The signal code to evaluate.
    trading_idea : str
        Description of the trading idea.
    kb_hints : list
        List of relevant signals from the knowledge base.
    
    Returns:
    --------
    tuple
        (score, review) where score is between 0 and 10 and review is a string.
    """
    # Initialize score
    score = 5  # Default middle score
    
    # Check code structure
    if "class" not in code or "def calc" not in code:
        return 1, "Code doesn't follow the required Factor class structure."
    
    # Check for basic requirements
    if "def __init__(self)" not in code:
        score -= 2
        review = "Missing __init__ method in the Factor class."
        return score, review
    
    if "self.inputs" not in code:
        score -= 2
        review = "Missing inputs specification in the Factor class."
        return score, review
    
    # Check alignment with trading idea
    idea_keywords = set(trading_idea.lower().split())
    code_lower = code.lower()
    
    # Count matching keywords
    match_count = sum(1 for word in idea_keywords if len(word) > 3 and word in code_lower)
    idea_relevance = match_count / max(1, len(idea_keywords))
    
    # Adjust score based on idea relevance
    if idea_relevance > 0.5:
        score += 2
    elif idea_relevance > 0.3:
        score += 1
    else:
        score -= 1
    
    # Check for potential issues
    if "np.nan" in code and "fillna" not in code:
        score -= 1
    
    if "rolling" in code and "window" in code:
        score += 1
    
    # Generate review based on score
    if score >= 8:
        review = "Excellent implementation that well captures the trading idea. The signal is well-structured and considers important aspects of the market."
    elif score >= 6:
        review = "Good implementation with room for improvement. Consider adding more sophistication to the signal calculation."
    elif score >= 4:
        review = "Acceptable implementation but lacks depth. The signal may not fully capture the essence of the trading idea."
    else:
        review = "Poor implementation with significant issues. The signal needs substantial revision to be effective."
    
    # Add specific suggestions
    if "momentum" in trading_idea.lower() and "momentum" not in code_lower:
        review += " Consider incorporating momentum calculations to better align with the trading idea."
    
    if "volume" in trading_idea.lower() and "volume" not in code_lower:
        review += " The trading idea mentions volume, but the implementation doesn't utilize volume data."
    
    return min(10, max(1, score)), review

def extract_signal_class(code):
    """
    Extract the signal class from the code.
    
    Parameters:
    -----------
    code : str
        The signal code.
    
    Returns:
    --------
    str or None
        The class name if found, None otherwise.
    """
    class_match = re.search(r'class\s+(\w+)\(Factor\):', code)
    if class_match:
        return class_match.group(1)
    return None

def execute_signal_code(code, features):
    """
    Execute the signal code and return the calculated signal.
    
    Parameters:
    -----------
    code : str
        The signal code to execute.
    features : dict
        Dictionary of features data.
    
    Returns:
    --------
    tuple
        (signal, success, error_message)
    """
    try:
        # Extract the class name
        class_name = extract_signal_class(code)
        if not class_name:
            return None, False, "Could not extract class name from code."
        
        # Execute the code to define the class
        exec(code, globals())
        
        # Create an instance of the signal class
        signal_instance = eval(f"{class_name}()")
        
        # Run the signal calculation
        signal = signal_instance.run(features)
        
        return signal, True, ""
    except Exception as e:
        return None, False, str(e)
    
def inner_loop(trading_idea, kb, features, future_returns, max_iterations=3, reward_threshold=8):
    """
    Implement the inner reasoning loop to generate and refine a trading signal.
    
    Parameters:
    -----------
    trading_idea : str
        Description of the trading idea.
    kb : list
        Knowledge base of signals.
    features : dict
        Dictionary of features data.
    future_returns : DataFrame
        Future returns for evaluation.
    max_iterations : int
        Maximum number of iterations.
    reward_threshold : float
        Threshold score to stop iterations.
    
    Returns:
    --------
    dict
        Information about the generated signal.
    """
    # Initialize context buffer
    context = {
        'trading_idea': trading_idea,
        'iterations': []
    }
    
    # Iterate until max iterations or reward threshold is reached
    for t in range(max_iterations):
        print(f"Inner loop iteration {t+1}/{max_iterations}")
        
        # Step 1: Writer retrieves knowledge from KB
        kb_hints_writer = retrieve_from_kb(trading_idea, kb)
        
        # Step 2: Writer generates an answer
        previous_reviews = [iter_info.get('review', '') for iter_info in context.get('iterations', [])]
        code = simulated_llm_writer(trading_idea, kb_hints_writer, previous_reviews)
        
        # Step 3: Judge retrieves knowledge from KB
        kb_hints_judge = retrieve_from_kb(trading_idea, kb)
        
        # Step 4: Judge evaluates the answer
        score, review = simulated_llm_judge(code, trading_idea, kb_hints_judge)
        
        # Update context
        context['iterations'].append({
            'code': code,
            'kb_hints_writer': kb_hints_writer,
            'kb_hints_judge': kb_hints_judge,
            'score': score,
            'review': review
        })
        
        print(f"  Score: {score}/10")
        
        # Check if reward threshold is reached
        if score >= reward_threshold:
            print(f"  Reward threshold reached. Stopping inner loop.")
            break
    
    # Extract the final code
    final_code = context['iterations'][-1]['code']
    
    # Execute the code to get the signal
    signal, success, error_message = execute_signal_code(final_code, features)
    
    # Evaluate the signal
    performance = {}
    if success and signal is not None:
        performance = evaluate_signal(signal, future_returns)
        print(f"  Signal performance: IC={performance['ic']:.4f}, Sharpe={performance['sharpe']:.2f}")
    else:
        print(f"  Failed to execute signal: {error_message}")
    
    # Prepare the result
    signal_info = {
        'name': extract_signal_class(final_code) or "UnknownSignal",
        'trading_idea': trading_idea,
        'code': final_code,
        'performance': performance,
        'iterations': context['iterations'],
        'success': success,
        'error': error_message if not success else ""
    }
    
    return signal_info


def outer_loop(num_iterations=5, kb_path='kb'):
    """
    Implement the outer feedback loop to build and improve the knowledge base.
    
    Parameters:
    -----------
    num_iterations : int
        Number of outer loop iterations.
    kb_path : str
        Path to the knowledge base directory.
    
    Returns:
    --------
    list
        The final knowledge base.
    """
    # Generate data for testing
    stock_data = generate_stock_data()
    features = reshape_data_for_signals(stock_data)
    
    # Prepare future returns for evaluation
    future_returns = features['return'].shift(-1)  # Next day's returns
    
    # Initialize or load knowledge base
    kb = load_kb(kb_path)
    print(f"Initial knowledge base size: {len(kb)}")
    
    # Track performance over iterations
    iteration_performance = []
    
    # Sample trading ideas
    trading_ideas = [
        "Moving average crossover strategy that buys when a fast MA crosses above a slow MA",
        "Momentum strategy that buys stocks with strong recent performance",
        "Volume-based strategy that buys stocks with unusual volume spikes",
        "Mean reversion strategy that buys stocks that have fallen significantly from their recent highs",
        "Volatility breakout strategy that buys stocks with increasing volatility",
        "Trend-following strategy based on moving average and price momentum",
        "Price-volume correlation strategy that looks for divergence between price and volume",
        "Relative strength strategy comparing stocks to their sector or market",
        "Overbought/oversold strategy using oscillators like RSI",
        "Gap-filling strategy that targets stocks with recent price gaps"
    ]
    
    # Ensure we have enough trading ideas
    while len(trading_ideas) < num_iterations:
        trading_ideas.extend(trading_ideas)
    
    # Run outer loop iterations
    for i in range(num_iterations):
        print(f"\nOuter loop iteration {i+1}/{num_iterations}")
        
        # Select a trading idea
        trading_idea = trading_ideas[i % len(trading_ideas)]
        print(f"Trading idea: {trading_idea}")
        
        # Run inner loop to generate a signal
        signal_info = inner_loop(trading_idea, kb, features, future_returns)
        
        # Save signal info to track performance
        if signal_info['success']:
            iteration_performance.append({
                'iteration': i+1,
                'name': signal_info['name'],
                'ic': signal_info['performance'].get('ic', 0),
                'sharpe': signal_info['performance'].get('sharpe', 0),
                'return': signal_info['performance'].get('return', 0),
                'valid_ratio': signal_info['performance'].get('valid_ratio', 0)
            })
        
        # Add signal to knowledge base
        if signal_info['success']:
            kb_path = save_signal_to_kb(signal_info)
            print(f"Saved signal to knowledge base: {kb_path}")
            kb.append(signal_info)
        
        print(f"Current knowledge base size: {len(kb)}")
    
    # Plot performance evolution
    if iteration_performance:
        metrics = ['ic', 'sharpe', 'return', 'valid_ratio']
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics):
            values = [p[metric] for p in iteration_performance]
            axes[i].plot(range(1, len(values) + 1), values, marker='o')
            axes[i].set_title(f'Evolution of {metric}')
            axes[i].set_xlabel('Outer Loop Iteration')
            axes[i].set_ylabel(metric)
            axes[i].grid(True)
        
        plt.tight_layout()
        plt.savefig('performance_evolution.png')
        plt.close()
    
    return kb, iteration_performance

def evaluate_combined_model(kb, features, future_returns):
    """
    Evaluate a combined model using all signals in the knowledge base.
    
    Parameters:
    -----------
    kb : list
        Knowledge base of signals.
    features : dict
        Dictionary of features data.
    future_returns : DataFrame
        Future returns for evaluation.
    
    Returns:
    --------
    dict
        Performance metrics.
    """
    if not kb:
        return {
            'ic': 0,
            'sharpe': 0,
            'return': 0,
            'mse': float('inf')
        }
    
    # Calculate signals from the knowledge base
    signals = []
    
    for signal_info in kb:
        if not signal_info.get('success', False):
            continue
        
        code = signal_info['code']
        signal, success, _ = execute_signal_code(code, features)
        
        if success and signal is not None:
            signals.append(signal)
    
    if not signals:
        return {
            'ic': 0,
            'sharpe': 0,
            'return': 0,
            'mse': float('inf')
        }
    
    # Create a dataset for model training
    X_data = []
    y_data = []
    
    # Align dates
    common_dates = future_returns.index
    for signal in signals:
        common_dates = common_dates.intersection(signal.index)
    
    if len(common_dates) == 0:
        return {
            'ic': 0,
            'sharpe': 0,
            'return': 0,
            'mse': float('inf')
        }
    
    # Prepare X and y data
    for date in common_dates:
        # Skip if future returns are not available
        if date not in future_returns.index:
            continue
        
        # Get stock list
        stocks = signals[0].loc[date].index
        
        # Collect features for each stock
        for stock in stocks:
            if stock not in future_returns.loc[date].index:
                continue
                
            # Skip stocks with NaN future returns
            if pd.isna(future_returns.loc[date, stock]):
                continue
                
            # Get features for this stock
            features_vec = []
            skip = False
            for signal in signals:
                if stock not in signal.loc[date].index or pd.isna(signal.loc[date, stock]):
                    skip = True
                    break
                features_vec.append(signal.loc[date, stock])
            
            if skip:
                continue
                
            # Add to dataset
            X_data.append(features_vec)
            y_data.append(future_returns.loc[date, stock])
    
    if not X_data:
        return {
            'ic': 0,
            'sharpe': 0,
            'return': 0,
            'mse': float('inf')
        }
    
    # Convert to numpy arrays
    X = np.array(X_data)
    y = np.array(y_data)
    
    # Split data into train and test
    train_size = int(0.7 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Train a simple model (XGBoost)
    model = xgb.XGBRegressor(n_estimators=50, learning_rate=0.1, max_depth=3)
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    
    # Calculate IC
    ic, _ = pearsonr(y_test, y_pred) if len(y_test) > 2 else (0, 0)
    
    # Create a weighted portfolio based on predictions
    # (This is a simplified simulation)
    portfolio_returns = []
    for i in range(0, len(y_test), 100):  # Simulate dates (each 100 samples)
        if i + 10 > len(y_test):  # Need at least 10 stocks
            continue
            
        # Get predictions and actual returns for this "date"
        date_preds = y_pred[i:i+100]
        date_returns = y_test[i:i+100]
        
        # Create weights (normalize predictions)
        if np.std(date_preds) == 0:
            continue
        weights = (date_preds - np.mean(date_preds)) / np.std(date_preds)
        
        # Calculate portfolio return
        port_return = np.sum(weights * date_returns) / np.sum(np.abs(weights))
        portfolio_returns.append(port_return)
    
    # Calculate Sharpe ratio and cumulative return
    if portfolio_returns:
        sharpe = np.sqrt(252) * np.mean(portfolio_returns) / np.std(portfolio_returns) if np.std(portfolio_returns) > 0 else 0
        cumulative_return = (1 + np.array(portfolio_returns)).prod() - 1
    else:
        sharpe = 0
        cumulative_return = 0
    
    return {
        'ic': ic,
        'sharpe': sharpe,
        'return': cumulative_return,
        'mse': mse
    }

def main():
    """
    Main function to run the QuantAgent experiment.
    """
    print("Starting QuantAgent experiment...")
    
    # Run outer loop to build knowledge base
    kb, iteration_performance = outer_loop(num_iterations=10)
    
    # Generate data for final evaluation
    stock_data = generate_stock_data(start_date='2023-06-01')  # New data for testing
    features = reshape_data_for_signals(stock_data)
    future_returns = features['return'].shift(-1)
    
    # Evaluate individual signals
    print("\nEvaluating individual signals...")
    for i, signal_info in enumerate(kb):
        if not signal_info.get('success', False):
            continue
            
        name = signal_info['name']
        performance = signal_info['performance']
        print(f"{i+1}. {name}: IC={performance.get('ic', 0):.4f}, Sharpe={performance.get('sharpe', 0):.2f}")
    
    # Evaluate combined model performance
    print("\nEvaluating combined model performance...")
    combined_performance = evaluate_combined_model(kb, features, future_returns)
    print(f"Combined model: IC={combined_performance['ic']:.4f}, Sharpe={combined_performance['sharpe']:.2f}, MSE={combined_performance['mse']:.6f}")
    
    # Evaluate knowledge base quality progression
    if iteration_performance:
        # Calculate moving average of performance
        window_size = min(3, len(iteration_performance))
        ma_ic = []
        for i in range(len(iteration_performance) - window_size + 1):
            avg_ic = np.mean([p['ic'] for p in iteration_performance[i:i+window_size]])
            ma_ic.append(avg_ic)
        
        print("\nKnowledge base quality progression:")
        print(f"Initial average IC (first 3 signals): {ma_ic[0]:.4f}")
        print(f"Final average IC (last 3 signals): {ma_ic[-1]:.4f}")
        
        # Plot
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(ma_ic) + 1), ma_ic, marker='o')
        plt.title('Moving Average of Information Coefficient (IC)')
        plt.xlabel('Iteration')
        plt.ylabel('Average IC (3-signal window)')
        plt.grid(True)
        plt.savefig('ic_progression.png')
        plt.close()
    
    print("\nExperiment completed successfully!")
    return kb, iteration_performance, combined_performance

# Run the experiment
if __name__ == "__main__":
    kb, iteration_performance, combined_performance = main()

# Analyze results in more detail
def analyze_results(kb, iteration_performance, combined_performance):
    """
    Analyze the results of the experiment in more detail.
    """
    # Count successful signals
    successful_signals = sum(1 for signal in kb if signal.get('success', False))
    
    print(f"\nDetailed Analysis:")
    print(f"Total signals in KB: {len(kb)}")
    print(f"Successful signals: {successful_signals}")
    
    # Analyze signal diversity
    signal_types = defaultdict(int)
    for signal in kb:
        if not signal.get('success', False):
            continue
            
        # Extract signal type based on class name
        name = signal.get('name', '')
        if 'Moving' in name or 'MA' in name:
            signal_types['Moving Average'] += 1
        elif 'Momentum' in name:
            signal_types['Momentum'] += 1
        elif 'Volume' in name:
            signal_types['Volume'] += 1
        elif 'Volatility' in name:
            signal_types['Volatility'] += 1
        else:
            signal_types['Other'] += 1
    
    print("\nSignal diversity:")
    for signal_type, count in signal_types.items():
        print(f"  {signal_type}: {count} signals")
    
    # Analyze performance metrics
    if iteration_performance:
        ic_values = [p['ic'] for p in iteration_performance]
        sharpe_values = [p['sharpe'] for p in iteration_performance]
        
        print("\nPerformance metrics:")
        print(f"  Average IC: {np.mean(ic_values):.4f}")
        print(f"  Average Sharpe: {np.mean(sharpe_values):.2f}")
        print(f"  IC range: [{min(ic_values):.4f}, {max(ic_values):.4f}]")
        print(f"  Sharpe range: [{min(sharpe_values):.2f}, {max(sharpe_values):.2f}]")
    
    # Analyze combined model performance
    print("\nCombined model performance:")
    print(f"  IC: {combined_performance['ic']:.4f}")
    print(f"  Sharpe: {combined_performance['sharpe']:.2f}")
    print(f"  MSE: {combined_performance['mse']:.6f}")
    
    # Check for self-improvement trends
    if len(iteration_performance) > 3:
        first_half = iteration_performance[:len(iteration_performance)//2]
        second_half = iteration_performance[len(iteration_performance)//2:]
        
        first_half_ic = np.mean([p['ic'] for p in first_half])
        second_half_ic = np.mean([p['ic'] for p in second_half])
        
        print("\nSelf-improvement analysis:")
        print(f"  First half average IC: {first_half_ic:.4f}")
        print(f"  Second half average IC: {second_half_ic:.4f}")
        print(f"  Improvement: {(second_half_ic - first_half_ic):.4f} ({(second_half_ic/first_half_ic - 1)*100:.1f}%)")

# Run analysis
analyze_results(kb, iteration_performance, combined_performance)

def create_visualizations(kb, iteration_performance):
    """
    Create visualizations of the experiment results.
    """
    # Plot performance metrics over iterations
    if iteration_performance:
        metrics = ['ic', 'sharpe', 'return', 'valid_ratio']
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics):
            values = [p[metric] for p in iteration_performance]
            
            # Fit a trend line
            x = np.arange(len(values))
            z = np.polyfit(x, values, 1)
            p = np.poly1d(z)
            
            axes[i].plot(x + 1, values, 'o-', label='Actual')
            axes[i].plot(x + 1, p(x), 'r--', label='Trend')
            axes[i].set_title(f'Evolution of {metric}')
            axes[i].set_xlabel('Iteration')
            axes[i].set_ylabel(metric)
            axes[i].grid(True)
            axes[i].legend()
        
        plt.tight_layout()
        plt.savefig('performance_metrics.png')
        plt.close()
    
    # Plot signal type distribution
    signal_types = defaultdict(int)
    for signal in kb:
        if not signal.get('success', False):
            continue
            
        # Extract signal type based on class name
        name = signal.get('name', '')
        if 'Moving' in name or 'MA' in name:
            signal_types['Moving Average'] += 1
        elif 'Momentum' in name:
            signal_types['Momentum'] += 1
        elif 'Volume' in name:
            signal_types['Volume'] += 1
        elif 'Volatility' in name:
            signal_types['Volatility'] += 1
        else:
            signal_types['Other'] += 1
    
    if signal_types:
        plt.figure(figsize=(10, 6))
        plt.bar(signal_types.keys(), signal_types.values())
        plt.title('Distribution of Signal Types')
        plt.xlabel('Signal Type')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('signal_types.png')
        plt.close()
    
    # Plot IC distribution
    if iteration_performance:
        ic_values = [p['ic'] for p in iteration_performance]
        
        plt.figure(figsize=(10, 6))
        plt.hist(ic_values, bins=10, alpha=0.7)
        plt.axvline(np.mean(ic_values), color='r', linestyle='dashed', linewidth=1, label=f'Mean: {np.mean(ic_values):.4f}')
        plt.title('Distribution of Information Coefficient (IC)')
        plt.xlabel('IC Value')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True)
        plt.savefig('ic_distribution.png')
        plt.close()

# Create visualizations
create_visualizations(kb, iteration_performance)

