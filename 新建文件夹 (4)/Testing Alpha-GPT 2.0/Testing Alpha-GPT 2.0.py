import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb
import itertools
import random
import time
from scipy.stats import pearsonr
import json
import os
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

print("Setting up Alpha-GPT 2.0 environment...")

def generate_market_data(n_stocks=100, n_days=1000, start_date='2020-01-01'):
    """
    Generate synthetic market data for testing alpha strategies.
    
    Parameters:
    -----------
    n_stocks : int
        Number of stocks to simulate
    n_days : int
        Number of days to simulate
    start_date : str
        Start date in 'YYYY-MM-DD' format
    
    Returns:
    --------
    DataFrame with market data
    """
    # Generate dates
    dates = pd.date_range(start=start_date, periods=n_days, freq='B')
    
    # Generate stock symbols
    stock_symbols = [f'STOCK_{i:03d}' for i in range(n_stocks)]
    
    # Create empty DataFrame to store market data
    market_data = pd.DataFrame()
    
    # Generate market factor (common market movement)
    market_factor = np.cumprod(1 + np.random.normal(0.0005, 0.01, n_days))
    
    # Generate sector factors (10 sectors)
    n_sectors = 10
    sector_factors = {}
    for i in range(n_sectors):
        sector_factors[i] = np.cumprod(1 + np.random.normal(0.0002, 0.015, n_days))
    
    # Assign stocks to sectors
    stock_sectors = np.random.randint(0, n_sectors, n_stocks)
    
    # Generate individual stock data
    for i, symbol in enumerate(stock_symbols):
        # Assign sector
        sector = stock_sectors[i]
        
        # Initialize stock data
        stock_data = pd.DataFrame(index=dates)
        
        # Generate price with market, sector, and idiosyncratic components
        # Base price (starting value)
        base_price = np.random.uniform(20, 200)
        
        # Generate returns with autocorrelation
        stock_specific_returns = np.random.normal(0.0001, 0.02, n_days)
        # Add momentum effect (autocorrelation)
        for j in range(1, n_days):
            stock_specific_returns[j] = 0.2 * stock_specific_returns[j-1] + 0.8 * stock_specific_returns[j]
        
        # Combine market, sector, and idiosyncratic components
        market_weight = np.random.uniform(0.3, 0.7)
        sector_weight = np.random.uniform(0.1, 0.5)
        stock_weight = 1 - market_weight - sector_weight
        
        combined_returns = market_weight * (market_factor[1:] / market_factor[:-1] - 1) + \
                          sector_weight * (sector_factors[sector][1:] / sector_factors[sector][:-1] - 1) + \
                          stock_weight * stock_specific_returns[1:]
        
        # Generate prices
        prices = np.zeros(n_days)
        prices[0] = base_price
        for j in range(1, n_days):
            prices[j] = prices[j-1] * (1 + combined_returns[j-1])
        
        # Add price data
        stock_data['close'] = prices
        
        # Generate daily high, low, and open prices
        daily_volatility = np.random.uniform(0.01, 0.03, n_days) * prices
        stock_data['high'] = prices + daily_volatility * np.random.uniform(0.2, 1.0, n_days)
        stock_data['low'] = prices - daily_volatility * np.random.uniform(0.2, 1.0, n_days)
        stock_data['open'] = stock_data['low'] + (stock_data['high'] - stock_data['low']) * np.random.uniform(0, 1, n_days)
        
        # Generate volume
        base_volume = np.random.randint(50000, 500000)
        volume_volatility = np.random.uniform(0.1, 0.5)
        # Higher volume on days with larger price changes
        volume_multiplier = 1 + 5 * np.abs(combined_returns)
        stock_data['volume'] = base_volume * np.random.lognormal(0, volume_volatility, n_days) * volume_multiplier
        
        # Add fundamental data (quarterly, with some noise)
        pe_ratio = np.random.uniform(10, 30)
        pe_trend = np.random.uniform(-0.05, 0.05)  # Trending up or down over time
        quarterly_noise = np.random.normal(0, 1, n_days // 63 + 1)
        pe_quarterly = np.repeat(quarterly_noise, 63)[:n_days]
        stock_data['pe_ratio'] = pe_ratio * (1 + pe_trend * np.arange(n_days)/n_days) + pe_quarterly
        
        # Book value (changes quarterly)
        book_value = base_price * np.random.uniform(0.3, 0.8)
        book_growth = np.random.uniform(0, 0.1)  # Annual growth
        quarterly_growth = (1 + book_growth) ** (1/4) - 1
        quarterly_steps = np.repeat(np.arange(n_days // 63 + 1), 63)[:n_days]
        stock_data['book_value'] = book_value * (1 + quarterly_growth) ** quarterly_steps
        
        # Market cap
        shares_outstanding = np.random.randint(1000000, 100000000)
        stock_data['market_cap'] = shares_outstanding * stock_data['close']
        
        # Add sector and other metadata
        stock_data['sector'] = sector
        stock_data['symbol'] = symbol
        
        # Add to main DataFrame
        if market_data.empty:
            market_data = stock_data.copy()
        else:
            market_data = pd.concat([market_data, stock_data])
    
    # Reset index and reshape
    market_data = market_data.reset_index()
    market_data.rename(columns={'index': 'date'}, inplace=True)
    
    # Add target variable (future returns for prediction)
    # 1-day future return
    market_data['future_return_1d'] = market_data.groupby('symbol')['close'].pct_change(1).shift(-1)
    # 5-day future return
    market_data['future_return_5d'] = market_data.groupby('symbol')['close'].pct_change(5).shift(-5)
    # 20-day future return
    market_data['future_return_20d'] = market_data.groupby('symbol')['close'].pct_change(20).shift(-20)
    
    # Calculate historical returns
    market_data['return_1d'] = market_data.groupby('symbol')['close'].pct_change(1)
    market_data['return_5d'] = market_data.groupby('symbol')['close'].pct_change(5)
    market_data['return_20d'] = market_data.groupby('symbol')['close'].pct_change(20)
    
    # Create a few more technical indicators
    # Simple moving averages
    market_data['sma_5'] = market_data.groupby('symbol')['close'].transform(lambda x: x.rolling(window=5).mean())
    market_data['sma_20'] = market_data.groupby('symbol')['close'].transform(lambda x: x.rolling(window=20).mean())
    market_data['sma_50'] = market_data.groupby('symbol')['close'].transform(lambda x: x.rolling(window=50).mean())
    
    # Volume indicators
    market_data['volume_sma_5'] = market_data.groupby('symbol')['volume'].transform(lambda x: x.rolling(window=5).mean())
    market_data['volume_ratio'] = market_data['volume'] / market_data['volume_sma_5']
    
    # Volatility (20-day rolling standard deviation of returns)
    market_data['volatility_20d'] = market_data.groupby('symbol')['return_1d'].transform(lambda x: x.rolling(window=20).std())
    
    # Add date components
    market_data['day_of_week'] = market_data['date'].dt.dayofweek
    market_data['month'] = market_data['date'].dt.month
    
    return market_data

# Generate market data
print("Generating synthetic market data...")
market_data = generate_market_data()
print(f"Generated data for {market_data['symbol'].nunique()} stocks over {market_data['date'].nunique()} days")
print(market_data.head())


class AlphaBase:
    """
    Repository for storing alpha factors, their implementation, and performance.
    """
    def __init__(self, save_path='alpha_base.json'):
        self.alphas = []
        self.save_path = save_path
        self.load()
    
    def add_alpha(self, alpha_info):
        """Add a new alpha to the repository"""
        # Add a unique ID if not provided
        if 'id' not in alpha_info:
            alpha_info['id'] = len(self.alphas) + 1
        
        # Add timestamp if not provided
        if 'timestamp' not in alpha_info:
            alpha_info['timestamp'] = dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        self.alphas.append(alpha_info)
        self.save()
        return alpha_info['id']
    
    def get_alpha(self, alpha_id):
        """Get alpha by ID"""
        for alpha in self.alphas:
            if alpha['id'] == alpha_id:
                return alpha
        return None
    
    def get_alphas_by_tag(self, tag):
        """Get alphas by tag"""
        return [alpha for alpha in self.alphas if tag.lower() in [t.lower() for t in alpha.get('tags', [])]]
    
    def search_alphas(self, query):
        """Search alphas by query string"""
        query = query.lower()
        results = []
        
        for alpha in self.alphas:
            # Search in name, description, and tags
            alpha_text = (alpha.get('name', '') + ' ' + 
                         alpha.get('description', '') + ' ' + 
                         ' '.join(alpha.get('tags', [])) + ' ' +
                         alpha.get('trading_idea', '')).lower()
            
            if query in alpha_text:
                results.append(alpha)
        
        return results
    
    def save(self):
        """Save the alpha base to disk"""
        with open(self.save_path, 'w') as f:
            json.dump(self.alphas, f, indent=2)
    
    def load(self):
        """Load the alpha base from disk"""
        if os.path.exists(self.save_path):
            try:
                with open(self.save_path, 'r') as f:
                    self.alphas = json.load(f)
            except:
                self.alphas = []

class KnowledgeRepository:
    """
    Repository for storing financial knowledge, market insights, and research papers.
    """
    def __init__(self):
        # Initialize with some basic trading strategies and concepts
        self.knowledge_base = {
            "momentum": "Momentum strategies exploit the tendency of assets that have performed well to continue performing well, and assets that have performed poorly to continue performing poorly. This is based on the behavioral tendency for investors to chase returns and the under-reaction of markets to new information.",
            "mean_reversion": "Mean reversion strategies are based on the idea that prices and returns eventually move back towards the mean or average. This strategy often involves buying assets that have performed poorly and selling assets that have performed well, with the expectation that prices will revert to their historical means.",
            "value": "Value investing involves buying securities that appear underpriced according to some form of fundamental analysis. This could be based on metrics like low price-to-earnings (P/E) ratios, price-to-book (P/B) ratios, or high dividend yields.",
            "growth": "Growth investing focuses on companies that are expected to grow their earnings at an above-average rate compared to other companies. Growth investors often look for companies with strong revenue and earnings growth, even if their current valuations seem high.",
            "quality": "Quality investing focuses on companies with strong balance sheets, stable earnings, high return on equity, low debt, and effective management. Quality factors often include profitability, earnings stability, dividend growth, and strength of the balance sheet.",
            "volatility": "Volatility-based strategies can involve selling options to capture the volatility risk premium, or trading based on changes in volatility. Some strategies focus on low-volatility stocks, which have historically provided better risk-adjusted returns than high-volatility stocks.",
            "market_microstructure": "Market microstructure strategies exploit inefficiencies in the trading process, such as bid-ask spreads, market impact, and price pressure. These strategies often require high-frequency trading and are sensitive to transaction costs.",
            "trend_following": "Trend following strategies aim to capitalize on market trends by buying assets that are rising and selling assets that are falling. These strategies often use moving averages or other technical indicators to identify trends.",
            "technical_analysis": "Technical analysis uses historical price and volume data to forecast future price movements. This approach includes chart patterns, trend lines, moving averages, and other technical indicators.",
            "fundamental_analysis": "Fundamental analysis involves evaluating a security's intrinsic value by examining related economic, financial, and other qualitative and quantitative factors. Fundamental analysts study anything that can affect the security's value, from macroeconomic factors like the state of the economy and industry conditions to microeconomic factors like the effectiveness of the company's management.",
            "arbitrage": "Arbitrage strategies exploit price differences of the same or similar financial instruments on different markets or in different forms. Pure arbitrage is considered risk-free profit, but in reality, most arbitrage involves some risk."
        }
    
    def get_knowledge(self, topic):
        """Get knowledge on a specific topic"""
        return self.knowledge_base.get(topic.lower(), "No information available on this topic.")
    
    def search_knowledge(self, query):
        """Search the knowledge base for relevant information"""
        query = query.lower()
        results = {}
        
        for topic, content in self.knowledge_base.items():
            if query in topic or query in content.lower():
                results[topic] = content
        
        return results

# Initialize alpha base and knowledge repository
alpha_base = AlphaBase()
knowledge_repo = KnowledgeRepository()

print("Alpha base and knowledge repository initialized.")


class AlphaBase:
    """
    Repository for storing alpha factors, their implementation, and performance.
    """
    def __init__(self, save_path='alpha_base.json'):
        self.alphas = []
        self.save_path = save_path
        self.load()
    
    def add_alpha(self, alpha_info):
        """Add a new alpha to the repository"""
        # Add a unique ID if not provided
        if 'id' not in alpha_info:
            alpha_info['id'] = len(self.alphas) + 1
        
        # Add timestamp if not provided
        if 'timestamp' not in alpha_info:
            alpha_info['timestamp'] = dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        self.alphas.append(alpha_info)
        self.save()
        return alpha_info['id']
    
    def get_alpha(self, alpha_id):
        """Get alpha by ID"""
        for alpha in self.alphas:
            if alpha['id'] == alpha_id:
                return alpha
        return None
    
    def get_alphas_by_tag(self, tag):
        """Get alphas by tag"""
        return [alpha for alpha in self.alphas if tag.lower() in [t.lower() for t in alpha.get('tags', [])]]
    
    def search_alphas(self, query):
        """Search alphas by query string"""
        query = query.lower()
        results = []
        
        for alpha in self.alphas:
            # Search in name, description, and tags
            alpha_text = (alpha.get('name', '') + ' ' + 
                         alpha.get('description', '') + ' ' + 
                         ' '.join(alpha.get('tags', [])) + ' ' +
                         alpha.get('trading_idea', '')).lower()
            
            if query in alpha_text:
                results.append(alpha)
        
        return results
    
    def save(self):
        """Save the alpha base to disk"""
        with open(self.save_path, 'w') as f:
            json.dump(self.alphas, f, indent=2)
    
    def load(self):
        """Load the alpha base from disk"""
        if os.path.exists(self.save_path):
            try:
                with open(self.save_path, 'r') as f:
                    self.alphas = json.load(f)
            except:
                self.alphas = []

class KnowledgeRepository:
    """
    Repository for storing financial knowledge, market insights, and research papers.
    """
    def __init__(self):
        # Initialize with some basic trading strategies and concepts
        self.knowledge_base = {
            "momentum": "Momentum strategies exploit the tendency of assets that have performed well to continue performing well, and assets that have performed poorly to continue performing poorly. This is based on the behavioral tendency for investors to chase returns and the under-reaction of markets to new information.",
            "mean_reversion": "Mean reversion strategies are based on the idea that prices and returns eventually move back towards the mean or average. This strategy often involves buying assets that have performed poorly and selling assets that have performed well, with the expectation that prices will revert to their historical means.",
            "value": "Value investing involves buying securities that appear underpriced according to some form of fundamental analysis. This could be based on metrics like low price-to-earnings (P/E) ratios, price-to-book (P/B) ratios, or high dividend yields.",
            "growth": "Growth investing focuses on companies that are expected to grow their earnings at an above-average rate compared to other companies. Growth investors often look for companies with strong revenue and earnings growth, even if their current valuations seem high.",
            "quality": "Quality investing focuses on companies with strong balance sheets, stable earnings, high return on equity, low debt, and effective management. Quality factors often include profitability, earnings stability, dividend growth, and strength of the balance sheet.",
            "volatility": "Volatility-based strategies can involve selling options to capture the volatility risk premium, or trading based on changes in volatility. Some strategies focus on low-volatility stocks, which have historically provided better risk-adjusted returns than high-volatility stocks.",
            "market_microstructure": "Market microstructure strategies exploit inefficiencies in the trading process, such as bid-ask spreads, market impact, and price pressure. These strategies often require high-frequency trading and are sensitive to transaction costs.",
            "trend_following": "Trend following strategies aim to capitalize on market trends by buying assets that are rising and selling assets that are falling. These strategies often use moving averages or other technical indicators to identify trends.",
            "technical_analysis": "Technical analysis uses historical price and volume data to forecast future price movements. This approach includes chart patterns, trend lines, moving averages, and other technical indicators.",
            "fundamental_analysis": "Fundamental analysis involves evaluating a security's intrinsic value by examining related economic, financial, and other qualitative and quantitative factors. Fundamental analysts study anything that can affect the security's value, from macroeconomic factors like the state of the economy and industry conditions to microeconomic factors like the effectiveness of the company's management.",
            "arbitrage": "Arbitrage strategies exploit price differences of the same or similar financial instruments on different markets or in different forms. Pure arbitrage is considered risk-free profit, but in reality, most arbitrage involves some risk."
        }
    
    def get_knowledge(self, topic):
        """Get knowledge on a specific topic"""
        return self.knowledge_base.get(topic.lower(), "No information available on this topic.")
    
    def search_knowledge(self, query):
        """Search the knowledge base for relevant information"""
        query = query.lower()
        results = {}
        
        for topic, content in self.knowledge_base.items():
            if query in topic or query in content.lower():
                results[topic] = content
        
        return results

# Initialize alpha base and knowledge repository
alpha_base = AlphaBase()
knowledge_repo = KnowledgeRepository()

print("Alpha base and knowledge repository initialized.")

class AlphaGenerator:
    """
    Generate alpha factors based on technical analysis and market data.
    """
    def __init__(self, market_data):
        self.market_data = market_data
        self.common_windows = [5, 10, 20, 50, 100, 200]
        self.alpha_templates = {
            'momentum': self._generate_momentum_alpha,
            'mean_reversion': self._generate_mean_reversion_alpha,
            'volatility': self._generate_volatility_alpha,
            'volume': self._generate_volume_alpha,
            'moving_average': self._generate_moving_average_alpha,
            'rsi': self._generate_rsi_alpha,
            'macd': self._generate_macd_alpha,
            'bollinger': self._generate_bollinger_alpha
        }
    
    def generate_alpha(self, alpha_type, params=None):
        """
        Generate an alpha factor based on the specified type and parameters.
        
        Parameters:
        -----------
        alpha_type : str
            Type of alpha to generate (e.g., 'momentum', 'mean_reversion', etc.)
        params : dict
            Parameters for the alpha factor
            
        Returns:
        --------
        tuple : (alpha_data, alpha_info)
            alpha_data : DataFrame with the calculated alpha values
            alpha_info : dict with metadata about the alpha
        """
        if alpha_type not in self.alpha_templates:
            raise ValueError(f"Alpha type '{alpha_type}' not supported. Supported types: {list(self.alpha_templates.keys())}")
        
        # Generate default parameters if not provided
        if params is None:
            params = self._generate_default_params(alpha_type)
        
        # Generate the alpha using the appropriate template
        return self.alpha_templates[alpha_type](params)
    
    def _generate_default_params(self, alpha_type):
        """Generate default parameters for the specified alpha type"""
        if alpha_type == 'momentum':
            return {'window': random.choice(self.common_windows)}
        elif alpha_type == 'mean_reversion':
            return {'window': random.choice(self.common_windows)}
        elif alpha_type == 'volatility':
            return {'window': random.choice(self.common_windows)}
        elif alpha_type == 'volume':
            return {'window': random.choice(self.common_windows)}
        elif alpha_type == 'moving_average':
            return {
                'fast_window': random.choice([5, 10, 20]),
                'slow_window': random.choice([50, 100, 200])
            }
        elif alpha_type == 'rsi':
            return {'window': random.choice([9, 14, 21])}
        elif alpha_type == 'macd':
            return {
                'fast_window': random.choice([8, 12]),
                'slow_window': random.choice([21, 26]),
                'signal_window': random.choice([9, 14])
            }
        elif alpha_type == 'bollinger':
            return {
                'window': random.choice([20, 30]),
                'num_std': random.choice([1.5, 2.0, 2.5])
            }
        else:
            return {}
    
    def _generate_momentum_alpha(self, params):
        """Generate momentum alpha"""
        window = params.get('window', 20)
        alpha_name = f"Momentum_{window}d"
        
        # Calculate momentum as past return
        alpha_data = self.market_data.copy()
        alpha_data[alpha_name] = alpha_data.groupby('symbol')['close'].pct_change(window)
        
        # Normalize alpha values (cross-sectional)
        alpha_data[alpha_name] = alpha_data.groupby('date')[alpha_name].transform(
            lambda x: (x - x.mean()) / x.std() if x.std() != 0 else 0
        )
        
        # Create alpha metadata
        alpha_info = {
            'name': alpha_name,
            'type': 'momentum',
            'description': f"{window}-day price momentum",
            'params': params,
            'trading_idea': f"Buy stocks with strong {window}-day momentum and sell stocks with weak momentum",
            'tags': ['momentum', 'trend_following', 'technical'],
            'python_code': f"""
def calculate_momentum_{window}d(data):
    # Calculate {window}-day momentum
    momentum = data.groupby('symbol')['close'].pct_change({window})
    
    # Normalize cross-sectionally
    momentum_norm = momentum.groupby(data['date']).transform(
        lambda x: (x - x.mean()) / x.std() if x.std() != 0 else 0
    )
    
    return momentum_norm
"""
        }
        
        return alpha_data, alpha_info
    
    def _generate_mean_reversion_alpha(self, params):
        """Generate mean reversion alpha"""
        window = params.get('window', 20)
        alpha_name = f"MeanReversion_{window}d"
        
        # Calculate mean reversion as negative past return
        alpha_data = self.market_data.copy()
        alpha_data[alpha_name] = -alpha_data.groupby('symbol')['close'].pct_change(window)
        
        # Normalize alpha values (cross-sectional)
        alpha_data[alpha_name] = alpha_data.groupby('date')[alpha_name].transform(
            lambda x: (x - x.mean()) / x.std() if x.std() != 0 else 0
        )
        
        # Create alpha metadata
        alpha_info = {
            'name': alpha_name,
            'type': 'mean_reversion',
            'description': f"{window}-day mean reversion",
            'params': params,
            'trading_idea': f"Buy stocks that have underperformed over the past {window} days and sell stocks that have outperformed",
            'tags': ['mean_reversion', 'contrarian', 'technical'],
            'python_code': f"""
def calculate_mean_reversion_{window}d(data):
    # Calculate {window}-day mean reversion (negative momentum)
    mean_reversion = -data.groupby('symbol')['close'].pct_change({window})
    
    # Normalize cross-sectionally
    mean_reversion_norm = mean_reversion.groupby(data['date']).transform(
        lambda x: (x - x.mean()) / x.std() if x.std() != 0 else 0
    )
    
    return mean_reversion_norm
"""
        }
        
        return alpha_data, alpha_info
    
    def _generate_volatility_alpha(self, params):
        """Generate volatility alpha"""
        window = params.get('window', 20)
        alpha_name = f"LowVol_{window}d"
        
        # Calculate volatility as rolling standard deviation of returns
        alpha_data = self.market_data.copy()
        alpha_data['return_1d'] = alpha_data.groupby('symbol')['close'].pct_change()
        alpha_data['volatility'] = alpha_data.groupby('symbol')['return_1d'].transform(
            lambda x: x.rolling(window=window).std()
        )
        
        # Invert volatility (low volatility anomaly)
        alpha_data[alpha_name] = 1 / alpha_data['volatility']
        
        # Normalize alpha values (cross-sectional)
        alpha_data[alpha_name] = alpha_data.groupby('date')[alpha_name].transform(
            lambda x: (x - x.mean()) / x.std() if x.std() != 0 else 0
        )
        
        # Create alpha metadata
        alpha_info = {
            'name': alpha_name,
            'type': 'volatility',
            'description': f"{window}-day low volatility",
            'params': params,
            'trading_idea': f"Buy stocks with low {window}-day volatility and sell stocks with high volatility",
            'tags': ['volatility', 'low_risk', 'technical'],
            'python_code': f"""
def calculate_low_vol_{window}d(data):
    # Calculate daily returns
    daily_returns = data.groupby('symbol')['close'].pct_change()
    
    # Calculate {window}-day volatility
    volatility = daily_returns.groupby(data['symbol']).transform(
        lambda x: x.rolling(window={window}).std()
    )
    
    # Invert volatility (low volatility anomaly)
    low_vol = 1 / volatility
    
    # Normalize cross-sectionally
    low_vol_norm = low_vol.groupby(data['date']).transform(
        lambda x: (x - x.mean()) / x.std() if x.std() != 0 else 0
    )
    
    return low_vol_norm
"""
        }
        
        return alpha_data, alpha_info
    
    def _generate_volume_alpha(self, params):
        """Generate volume-based alpha"""
        window = params.get('window', 20)
        alpha_name = f"VolumeChange_{window}d"
        
        # Calculate volume change
        alpha_data = self.market_data.copy()
        alpha_data['volume_sma'] = alpha_data.groupby('symbol')['volume'].transform(
            lambda x: x.rolling(window=window).mean()
        )
        alpha_data[alpha_name] = alpha_data['volume'] / alpha_data['volume_sma'] - 1
        
        # Normalize alpha values (cross-sectional)
        alpha_data[alpha_name] = alpha_data.groupby('date')[alpha_name].transform(
            lambda x: (x - x.mean()) / x.std() if x.std() != 0 else 0
        )
        
        # Create alpha metadata
        alpha_info = {
            'name': alpha_name,
            'type': 'volume',
            'description': f"{window}-day volume change",
            'params': params,
            'trading_idea': f"Buy stocks with unusual volume increases over the past {window} days",
            'tags': ['volume', 'liquidity', 'technical'],
            'python_code': f"""
def calculate_volume_change_{window}d(data):
    # Calculate {window}-day moving average of volume
    volume_sma = data.groupby('symbol')['volume'].transform(
        lambda x: x.rolling(window={window}).mean()
    )
    
    # Calculate volume change ratio
    volume_change = data['volume'] / volume_sma - 1
    
    # Normalize cross-sectionally
    volume_change_norm = volume_change.groupby(data['date']).transform(
        lambda x: (x - x.mean()) / x.std() if x.std() != 0 else 0
    )
    
    return volume_change_norm
"""
        }
        
        return alpha_data, alpha_info
    
    def _generate_moving_average_alpha(self, params):
        """Generate moving average crossover alpha"""
        fast_window = params.get('fast_window', 10)
        slow_window = params.get('slow_window', 50)
        alpha_name = f"MACross_{fast_window}_{slow_window}"
        
        # Calculate moving averages
        alpha_data = self.market_data.copy()
        alpha_data[f'sma_{fast_window}'] = alpha_data.groupby('symbol')['close'].transform(
            lambda x: x.rolling(window=fast_window).mean()
        )
        alpha_data[f'sma_{slow_window}'] = alpha_data.groupby('symbol')['close'].transform(
            lambda x: x.rolling(window=slow_window).mean()
        )
        
        # Calculate moving average crossover
        alpha_data[alpha_name] = alpha_data[f'sma_{fast_window}'] / alpha_data[f'sma_{slow_window}'] - 1
        
        # Normalize alpha values (cross-sectional)
        alpha_data[alpha_name] = alpha_data.groupby('date')[alpha_name].transform(
            lambda x: (x - x.mean()) / x.std() if x.std() != 0 else 0
        )
        
        # Create alpha metadata
        alpha_info = {
            'name': alpha_name,
            'type': 'moving_average',
            'description': f"{fast_window}/{slow_window} moving average crossover",
            'params': params,
            'trading_idea': f"Buy stocks where the {fast_window}-day moving average is above the {slow_window}-day moving average",
            'tags': ['moving_average', 'trend_following', 'technical'],
            'python_code': f"""
def calculate_ma_cross_{fast_window}_{slow_window}(data):
    # Calculate moving averages
    sma_fast = data.groupby('symbol')['close'].transform(
        lambda x: x.rolling(window={fast_window}).mean()
    )
    sma_slow = data.groupby('symbol')['close'].transform(
        lambda x: x.rolling(window={slow_window}).mean()
    )
    
    # Calculate moving average crossover
    ma_cross = sma_fast / sma_slow - 1
    
    # Normalize cross-sectionally
    ma_cross_norm = ma_cross.groupby(data['date']).transform(
        lambda x: (x - x.mean()) / x.std() if x.std() != 0 else 0
    )
    
    return ma_cross_norm
"""
        }
        
        return alpha_data, alpha_info
    
    def _generate_rsi_alpha(self, params):
        """Generate RSI alpha"""
        window = params.get('window', 14)
        alpha_name = f"RSI_{window}"
        
        # Calculate RSI
        alpha_data = self.market_data.copy()
        alpha_data['return_1d'] = alpha_data.groupby('symbol')['close'].pct_change()
        
        # Calculate average gains and losses
        alpha_data['gain'] = alpha_data['return_1d'].clip(lower=0)
        alpha_data['loss'] = -alpha_data['return_1d'].clip(upper=0)
        
        # Calculate RSI
        alpha_data['avg_gain'] = alpha_data.groupby('symbol')['gain'].transform(
            lambda x: x.rolling(window=window).mean()
        )
        alpha_data['avg_loss'] = alpha_data.groupby('symbol')['loss'].transform(
            lambda x: x.rolling(window=window).mean()
        )
        
        # Calculate relative strength
        alpha_data['rs'] = alpha_data['avg_gain'] / alpha_data['avg_loss'].replace(0, 1e-10)
        
        # Calculate RSI
        alpha_data['rsi'] = 100 - (100 / (1 + alpha_data['rs']))
        
        # Invert RSI (overbought/oversold)
        alpha_data[alpha_name] = 50 - alpha_data['rsi']
        
        # Normalize alpha values (cross-sectional)
        alpha_data[alpha_name] = alpha_data.groupby('date')[alpha_name].transform(
            lambda x: (x - x.mean()) / x.std() if x.std() != 0 else 0
        )
        
        # Create alpha metadata
        alpha_info = {
            'name': alpha_name,
            'type': 'rsi',
            'description': f"{window}-day Relative Strength Index (mean reversion)",
            'params': params,
            'trading_idea': f"Buy stocks with low RSI (oversold) and sell stocks with high RSI (overbought)",
            'tags': ['rsi', 'mean_reversion', 'overbought_oversold', 'technical'],
            'python_code': f"""
def calculate_rsi_{window}(data):
    # Calculate daily returns
    daily_returns = data.groupby('symbol')['close'].pct_change()
    
    # Calculate gains and losses
    gains = daily_returns.clip(lower=0)
    losses = -daily_returns.clip(upper=0)
    
    # Calculate average gains and losses
    avg_gains = gains.groupby(data['symbol']).transform(
        lambda x: x.rolling(window={window}).mean()
    )
    avg_losses = losses.groupby(data['symbol']).transform(
        lambda x: x.rolling(window={window}).mean()
    )
    
    # Calculate relative strength
    rs = avg_gains / avg_losses.replace(0, 1e-10)
    
    # Calculate RSI
    rsi = 100 - (100 / (1 + rs))
    
    # Invert RSI (so that low RSI values are positive alpha)
    rsi_alpha = 50 - rsi
    
    # Normalize cross-sectionally
    rsi_norm = rsi_alpha.groupby(data['date']).transform(
        lambda x: (x - x.mean()) / x.std() if x.std() != 0 else 0
    )
    
    return rsi_norm
"""
        }
        
        return alpha_data, alpha_info
    
    def _generate_macd_alpha(self, params):
        """Generate MACD alpha"""
        fast_window = params.get('fast_window', 12)
        slow_window = params.get('slow_window', 26)
        signal_window = params.get('signal_window', 9)
        alpha_name = f"MACD_{fast_window}_{slow_window}_{signal_window}"
        
        # Calculate MACD
        alpha_data = self.market_data.copy()
        alpha_data[f'ema_{fast_window}'] = alpha_data.groupby('symbol')['close'].transform(
            lambda x: x.ewm(span=fast_window, adjust=False).mean()
        )
        alpha_data[f'ema_{slow_window}'] = alpha_data.groupby('symbol')['close'].transform(
            lambda x: x.ewm(span=slow_window, adjust=False).mean()
        )
        
        # Calculate MACD line
        alpha_data['macd_line'] = alpha_data[f'ema_{fast_window}'] - alpha_data[f'ema_{slow_window}']
        
        # Calculate signal line
        alpha_data['signal_line'] = alpha_data.groupby('symbol')['macd_line'].transform(
            lambda x: x.ewm(span=signal_window, adjust=False).mean()
        )
        
        # Calculate MACD histogram
        alpha_data[alpha_name] = alpha_data['macd_line'] - alpha_data['signal_line']
        
        # Normalize alpha values (cross-sectional)
        alpha_data[alpha_name] = alpha_data.groupby('date')[alpha_name].transform(
            lambda x: (x - x.mean()) / x.std() if x.std() != 0 else 0
        )
        
        # Create alpha metadata
        alpha_info = {
            'name': alpha_name,
            'type': 'macd',
            'description': f"MACD ({fast_window}, {slow_window}, {signal_window})",
            'params': params,
            'trading_idea': f"Buy stocks with positive MACD histogram (MACD line above signal line) and sell stocks with negative MACD histogram",
            'tags': ['macd', 'trend_following', 'momentum', 'technical'],
            'python_code': f"""
def calculate_macd_{fast_window}_{slow_window}_{signal_window}(data):
    # Calculate exponential moving averages
    ema_fast = data.groupby('symbol')['close'].transform(
        lambda x: x.ewm(span={fast_window}, adjust=False).mean()
    )
    ema_slow = data.groupby('symbol')['close'].transform(
        lambda x: x.ewm(span={slow_window}, adjust=False).mean()
    )
    
    # Calculate MACD line
    macd_line = ema_fast - ema_slow
    
    # Calculate signal line
    signal_line = macd_line.groupby(data['symbol']).transform(
        lambda x: x.ewm(span={signal_window}, adjust=False).mean()
    )
    
    # Calculate MACD histogram
    macd_hist = macd_line - signal_line
    
    # Normalize cross-sectionally
    macd_hist_norm = macd_hist.groupby(data['date']).transform(
        lambda x: (x - x.mean()) / x.std() if x.std() != 0 else 0
    )
    
    return macd_hist_norm
"""
        }
        
        return alpha_data, alpha_info
    
    def _generate_bollinger_alpha(self, params):
        """Generate Bollinger Bands alpha"""
        window = params.get('window', 20)
        num_std = params.get('num_std', 2.0)
        alpha_name = f"BollingerMeanReversion_{window}_{num_std}"
        
        # Calculate Bollinger Bands
        alpha_data = self.market_data.copy()
        alpha_data[f'sma_{window}'] = alpha_data.groupby('symbol')['close'].transform(
            lambda x: x.rolling(window=window).mean()
        )
        alpha_data[f'std_{window}'] = alpha_data.groupby('symbol')['close'].transform(
            lambda x: x.rolling(window=window).std()
        )
        
        # Calculate upper and lower bands
        alpha_data['upper_band'] = alpha_data[f'sma_{window}'] + num_std * alpha_data[f'std_{window}']
        alpha_data['lower_band'] = alpha_data[f'sma_{window}'] - num_std * alpha_data[f'std_{window}']
        
        # Calculate percent B
        alpha_data['percent_b'] = (alpha_data['close'] - alpha_data['lower_band']) / (alpha_data['upper_band'] - alpha_data['lower_band'])
        
        # Calculate mean reversion signal (reverse percent B)
        alpha_data[alpha_name] = 0.5 - alpha_data['percent_b']
        
        # Normalize alpha values (cross-sectional)
        alpha_data[alpha_name] = alpha_data.groupby('date')[alpha_name].transform(
            lambda x: (x - x.mean()) / x.std() if x.std() != 0 else 0
        )
        
        # Create alpha metadata
        alpha_info = {
            'name': alpha_name,
            'type': 'bollinger',
            'description': f"Bollinger Bands mean reversion ({window}, {num_std})",
            'params': params,
            'trading_idea': f"Buy stocks near the lower Bollinger Band and sell stocks near the upper Bollinger Band",
            'tags': ['bollinger', 'mean_reversion', 'technical'],
            'python_code': f"""
def calculate_bollinger_mean_reversion_{window}_{int(num_std)}(data):
    # Calculate moving average
    sma = data.groupby('symbol')['close'].transform(
        lambda x: x.rolling(window={window}).mean()
    )
    
    # Calculate standard deviation
    std = data.groupby('symbol')['close'].transform(
        lambda x: x.rolling(window={window}).std()
    )
    
    # Calculate upper and lower bands
    upper_band = sma + {num_std} * std
    lower_band = sma - {num_std} * std
    
    # Calculate percent B
    percent_b = (data['close'] - lower_band) / (upper_band - lower_band)
    
    # Calculate mean reversion signal (reverse percent B)
    bollinger_alpha = 0.5 - percent_b
    
    # Normalize cross-sectionally
    bollinger_alpha_norm = bollinger_alpha.groupby(data['date']).transform(
        lambda x: (x - x.mean()) / x.std() if x.std() != 0 else 0
    )
    
    return bollinger_alpha_norm
"""
        }
        
        return alpha_data, alpha_info

# Initialize alpha generator
alpha_generator = AlphaGenerator(market_data)
print("Alpha generator initialized.")


class AlphaEvaluator:
    """
    Evaluate alpha factors by calculating performance metrics and running backtests.
    """
    def __init__(self, market_data):
        self.market_data = market_data
        self.prediction_windows = [1, 5, 20]  # 1-day, 5-day, and 20-day prediction windows
    
    def evaluate_alpha(self, alpha_data, alpha_name, min_date=None, max_date=None):
        """
        Evaluate an alpha factor by calculating performance metrics.
        
        Parameters:
        -----------
        alpha_data : DataFrame
            DataFrame containing the alpha values
        alpha_name : str
            Name of the alpha factor
        min_date : datetime, optional
            Minimum date for evaluation
        max_date : datetime, optional
            Maximum date for evaluation
            
        Returns:
        --------
        dict
            Dictionary containing performance metrics
        """
        # Filter data by date if specified
        data = alpha_data.copy()
        if min_date is not None:
            data = data[data['date'] >= pd.to_datetime(min_date)]
        if max_date is not None:
            data = data[data['date'] <= pd.to_datetime(max_date)]
        
        # Drop rows with missing alpha values
        data = data.dropna(subset=[alpha_name])
        
        # Calculate performance metrics for each prediction window
        metrics = {}
        
        for window in self.prediction_windows:
            future_return_col = f'future_return_{window}d'
            
            # Calculate information coefficient (IC)
            ic_by_date = data.groupby('date').apply(
                lambda x: x[[alpha_name, future_return_col]].corr().iloc[0, 1]
                if x[alpha_name].std() > 0 and x[future_return_col].std() > 0
                else np.nan
            )
            ic = ic_by_date.mean()
            ic_std = ic_by_date.std()
            ic_t_stat = ic / (ic_std / np.sqrt(len(ic_by_date)))
            
            # Calculate returns of alpha-weighted portfolio
            data['weight'] = data.groupby('date')[alpha_name].transform(
                lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0
            )
            data['weighted_return'] = data['weight'] * data[future_return_col]
            
            # Calculate portfolio returns by date
            portfolio_returns = data.groupby('date')['weighted_return'].sum() / data.groupby('date')['weight'].apply(lambda x: np.sum(np.abs(x)))
            
            # Calculate performance metrics
            ann_factor = 252 / window
            ann_return = portfolio_returns.mean() * ann_factor
            ann_volatility = portfolio_returns.std() * np.sqrt(ann_factor)
            sharpe_ratio = ann_return / ann_volatility if ann_volatility > 0 else 0
            
            # Calculate drawdowns
            cum_returns = (1 + portfolio_returns).cumprod()
            rolling_max = cum_returns.expanding().max()
            drawdowns = (cum_returns / rolling_max) - 1
            max_drawdown = drawdowns.min()
            
            # Store metrics
            metrics[f'{window}d'] = {
                'ic': ic,
                'ic_std': ic_std,
                'ic_t_stat': ic_t_stat,
                'ann_return': ann_return,
                'ann_volatility': ann_volatility,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'win_rate': (portfolio_returns > 0).mean()
            }
        
        return metrics
    
    def run_backtest(self, alpha_data, alpha_name, prediction_window=5, min_date=None, max_date=None):
        """
        Run a backtest for an alpha factor.
        
        Parameters:
        -----------
        alpha_data : DataFrame
            DataFrame containing the alpha values
        alpha_name : str
            Name of the alpha factor
        prediction_window : int
            Prediction window in days
        min_date : datetime, optional
            Minimum date for backtest
        max_date : datetime, optional
            Maximum date for backtest
            
        Returns:
        --------
        dict
            Dictionary containing backtest results
        """
        # Filter data by date if specified
        data = alpha_data.copy()
        if min_date is not None:
            data = data[data['date'] >= pd.to_datetime(min_date)]
        if max_date is not None:
            data = data[data['date'] <= pd.to_datetime(max_date)]
        
        # Drop rows with missing alpha values
        data = data.dropna(subset=[alpha_name])
        
        # Get future return column
        future_return_col = f'future_return_{prediction_window}d'
        
        # Calculate normalized weights
        data['weight'] = data.groupby('date')[alpha_name].transform(
            lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0
        )
        
        # Calculate portfolio returns
        data['weighted_return'] = data['weight'] * data[future_return_col]
        portfolio_returns = data.groupby('date')['weighted_return'].sum() / data.groupby('date')['weight'].apply(lambda x: np.sum(np.abs(x)))
        
        # Calculate cumulative returns
        cumulative_returns = (1 + portfolio_returns).cumprod() - 1
        
        # Calculate performance metrics
        ann_factor = 252 / prediction_window
        ann_return = portfolio_returns.mean() * ann_factor
        ann_volatility = portfolio_returns.std() * np.sqrt(ann_factor)
        sharpe_ratio = ann_return / ann_volatility if ann_volatility > 0 else 0
        
        # Calculate drawdowns
        cum_returns = (1 + portfolio_returns).cumprod()
        rolling_max = cum_returns.expanding().max()
        drawdowns = (cum_returns / rolling_max) - 1
        max_drawdown = drawdowns.min()
        
        # Prepare backtest results
        backtest_results = {
            'portfolio_returns': portfolio_returns,
            'cumulative_returns': cumulative_returns,
            'ann_return': ann_return,
            'ann_volatility': ann_volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': (portfolio_returns > 0).mean()
        }
        
        return backtest_results

# Initialize alpha evaluator
alpha_evaluator = AlphaEvaluator(market_data)
print("Alpha evaluator initialized.")


class AlphaMiningAgent:
    """
    Agent for mining alpha factors based on trading ideas.
    """
    def __init__(self, market_data, alpha_generator, alpha_evaluator, alpha_base, knowledge_repo):
        self.market_data = market_data
        self.alpha_generator = alpha_generator
        self.alpha_evaluator = alpha_evaluator
        self.alpha_base = alpha_base
        self.knowledge_repo = knowledge_repo
    
    def process_trading_idea(self, trading_idea):
        """
        Process a trading idea to generate alpha factors.
        
        Parameters:
        -----------
        trading_idea : str
            Description of the trading idea
            
        Returns:
        --------
        list
            List of generated alpha factors
        """
        print(f"\nProcessing trading idea: '{trading_idea}'")
        
        # Parse the trading idea to determine relevant alpha types
        alpha_types = self._parse_trading_idea(trading_idea)
        
        print(f"Identified alpha types: {alpha_types}")
        
        # Generate alpha factors for each alpha type
        generated_alphas = []
        
        for alpha_type in alpha_types:
            # Generate multiple variations of the alpha
            for _ in range(2):  # Generate 2 variations of each alpha type
                # Generate alpha with random parameters
                alpha_data, alpha_info = self.alpha_generator.generate_alpha(alpha_type)
                
                # Add trading idea to alpha info
                alpha_info['trading_idea'] = trading_idea
                
                # Evaluate alpha
                alpha_name = alpha_info['name']
                evaluation = self.alpha_evaluator.evaluate_alpha(alpha_data, alpha_name)
                alpha_info['evaluation'] = evaluation
                
                # Run backtest
                backtest = self.alpha_evaluator.run_backtest(alpha_data, alpha_name)
                alpha_info['backtest'] = {
                    'ann_return': backtest['ann_return'],
                    'ann_volatility': backtest['ann_volatility'],
                    'sharpe_ratio': backtest['sharpe_ratio'],
                    'max_drawdown': backtest['max_drawdown'],
                    'win_rate': backtest['win_rate']
                }
                
                # Add alpha to alpha base
                alpha_id = self.alpha_base.add_alpha(alpha_info)
                
                # Add to generated alphas
                generated_alphas.append({
                    'id': alpha_id,
                    'name': alpha_name,
                    'type': alpha_info['type'],
                    'description': alpha_info['description'],
                    'sharpe_ratio': backtest['sharpe_ratio'],
                    'ic_5d': evaluation['5d']['ic']
                })
                
                print(f"Generated alpha: {alpha_name} (Sharpe: {backtest['sharpe_ratio']:.2f}, IC (5d): {evaluation['5d']['ic']:.4f})")
        
        return generated_alphas
    
    def search_and_enhance_alphas(self, query, num_variations=3):
        """
        Search for existing alphas and generate enhanced variations.
        
        Parameters:
        -----------
        query : str
            Search query
        num_variations : int
            Number of variations to generate
            
        Returns:
        --------
        tuple
            (existing_alphas, enhanced_alphas)
        """
        print(f"\nSearching for alphas matching query: '{query}'")
        
        # Search for existing alphas
        existing_alphas = self.alpha_base.search_alphas(query)
        
        print(f"Found {len(existing_alphas)} matching alphas")
        
        # Generate enhanced variations
        enhanced_alphas = []
        
        if existing_alphas:
            # Select a few alphas to enhance
            alphas_to_enhance = random.sample(existing_alphas, min(3, len(existing_alphas)))
            
            for base_alpha in alphas_to_enhance:
                alpha_type = base_alpha.get('type')
                if alpha_type in self.alpha_generator.alpha_templates:
                    # Generate variations with different parameters
                    for _ in range(num_variations):
                        # Generate a new parameter set based on the original
                        params = base_alpha.get('params', {}).copy()
                        
                        # Modify a parameter
                        if alpha_type == 'momentum' or alpha_type == 'mean_reversion' or alpha_type == 'volatility':
                            params['window'] = max(5, params.get('window', 20) + random.randint(-10, 10))
                        elif alpha_type == 'moving_average':
                            params['fast_window'] = max(2, params.get('fast_window', 10) + random.randint(-5, 5))
                            params['slow_window'] = max(params.get('fast_window', 10) + 5, params.get('slow_window', 50) + random.randint(-10, 10))
                        elif alpha_type == 'rsi':
                            params['window'] = max(2, params.get('window', 14) + random.randint(-5, 5))
                        elif alpha_type == 'macd':
                            params['fast_window'] = max(2, params.get('fast_window', 12) + random.randint(-3, 3))
                            params['slow_window'] = max(params.get('fast_window', 12) + 5, params.get('slow_window', 26) + random.randint(-5, 5))
                            params['signal_window'] = max(2, params.get('signal_window', 9) + random.randint(-3, 3))
                        elif alpha_type == 'bollinger':
                            params['window'] = max(5, params.get('window', 20) + random.randint(-10, 10))
                            params['num_std'] = max(0.5, params.get('num_std', 2.0) + random.uniform(-0.5, 0.5))
                        
                        # Generate enhanced alpha
                        alpha_data, alpha_info = self.alpha_generator.generate_alpha(alpha_type, params)
                        
                        # Add reference to base alpha
                        alpha_info['enhanced_from'] = base_alpha.get('id')
                        alpha_info['trading_idea'] = base_alpha.get('trading_idea', '')
                        
                        # Evaluate enhanced alpha
                        alpha_name = alpha_info['name']
                        evaluation = self.alpha_evaluator.evaluate_alpha(alpha_data, alpha_name)
                        alpha_info['evaluation'] = evaluation
                        
                        # Run backtest
                        backtest = self.alpha_evaluator.run_backtest(alpha_data, alpha_name)
                        alpha_info['backtest'] = {
                            'ann_return': backtest['ann_return'],
                            'ann_volatility': backtest['ann_volatility'],
                            'sharpe_ratio': backtest['sharpe_ratio'],
                            'max_drawdown': backtest['max_drawdown'],
                            'win_rate': backtest['win_rate']
                        }
                        
                        # Add alpha to alpha base
                        alpha_id = self.alpha_base.add_alpha(alpha_info)
                        
                        # Add to enhanced alphas
                        enhanced_alphas.append({
                            'id': alpha_id,
                            'name': alpha_name,
                            'type': alpha_info['type'],
                            'description': alpha_info['description'],
                            'sharpe_ratio': backtest['sharpe_ratio'],
                            'ic_5d': evaluation['5d']['ic'],
                            'enhanced_from': base_alpha.get('id')
                        })
                        
                        print(f"Enhanced alpha: {alpha_name} (Sharpe: {backtest['sharpe_ratio']:.2f}, IC (5d): {evaluation['5d']['ic']:.4f})")
        
        return existing_alphas, enhanced_alphas
    
    def _parse_trading_idea(self, trading_idea):
        """
        Parse a trading idea to determine relevant alpha types.
        
        Parameters:
        -----------
        trading_idea : str
            Description of the trading idea
            
        Returns:
        --------
        list
            List of alpha types
        """
        trading_idea = trading_idea.lower()
        
        # Define keywords for each alpha type
        alpha_keywords = {
            'momentum': ['momentum', 'trend', 'following', 'continue', 'persistence'],
            'mean_reversion': ['mean reversion', 'revert', 'reversal', 'oversold', 'overbought', 'overreaction'],
            'volatility': ['volatility', 'risk', 'stable', 'low vol', 'high vol'],
            'volume': ['volume', 'liquidity', 'trading activity'],
            'moving_average': ['moving average', 'crossover', 'ma', 'sma', 'ema'],
            'rsi': ['rsi', 'relative strength', 'oversold', 'overbought'],
            'macd': ['macd', 'moving average convergence', 'divergence'],
            'bollinger': ['bollinger', 'bands', 'deviation', 'channel']
        }
        
        # Identify alpha types based on keywords
        alpha_types = []
        for alpha_type, keywords in alpha_keywords.items():
            if any(keyword in trading_idea for keyword in keywords):
                alpha_types.append(alpha_type)
        
        # If no specific alpha types are identified, include a default set
        if not alpha_types:
            alpha_types = ['momentum', 'mean_reversion', 'moving_average']
        
        return alpha_types

# Initialize alpha mining agent
alpha_mining_agent = AlphaMiningAgent(market_data, alpha_generator, alpha_evaluator, alpha_base, knowledge_repo)
print("Alpha mining agent initialized.")


class AlphaModelingAgent:
    """
    Agent for modeling alpha factors to create predictive signals.
    """
    def __init__(self, market_data, alpha_base):
        self.market_data = market_data
        self.alpha_base = alpha_base
        self.models = {
            'linear_regression': LinearRegression,
            'ridge': Ridge,
            'lasso': Lasso,
            'random_forest': RandomForestRegressor,
            'gradient_boosting': GradientBoostingRegressor,
            'xgboost': xgb.XGBRegressor,
            'lightgbm': lgb.LGBMRegressor
        }
    
    def build_model(self, alpha_ids, model_type='xgboost', target_horizon=5, min_date=None, max_date=None):
        """
        Build a predictive model using the specified alpha factors.
        
        Parameters:
        -----------
        alpha_ids : list
            List of alpha IDs to include in the model
        model_type : str
            Type of model to build
        target_horizon : int
            Target prediction horizon in days (1, 5, or 20)
        min_date : datetime, optional
            Minimum date for training
        max_date : datetime, optional
            Maximum date for training
            
        Returns:
        --------
        dict
            Dictionary containing model information and performance metrics
        """
        print(f"\nBuilding {model_type} model with {len(alpha_ids)} alpha factors...")
        
        # Validate model type
        if model_type not in self.models:
            raise ValueError(f"Model type '{model_type}' not supported. Supported types: {list(self.models.keys())}")
        
        # Get alphas from the alpha base
        alphas = [self.alpha_base.get_alpha(alpha_id) for alpha_id in alpha_ids]
        
        # Filter out None values (invalid alpha IDs)
        alphas = [alpha for alpha in alphas if alpha is not None]
        
        if not alphas:
            raise ValueError("No valid alpha factors provided")
        
        # Prepare feature data
        feature_data = self._prepare_feature_data(alphas, min_date, max_date)
        
        # Get target column
        target_col = f'future_return_{target_horizon}d'
        
        if target_col not in feature_data.columns:
            raise ValueError(f"Target column '{target_col}' not found in data")
        
        # Split data into features and target
        X = feature_data.drop(columns=['date', 'symbol', target_col] + [f'future_return_{w}d' for w in [1, 5, 20] if f'future_return_{w}d' in feature_data.columns])
        y = feature_data[target_col]
        
        # Split data into training and test sets
        dates = feature_data['date'].unique()
        train_dates = dates[:int(0.7 * len(dates))]
        test_dates = dates[int(0.7 * len(dates)):]
        
        train_mask = feature_data['date'].isin(train_dates)
        test_mask = feature_data['date'].isin(test_dates)
        
        X_train, X_test = X[train_mask], X[test_mask]
        y_train, y_test = y[train_mask], y[test_mask]
        
        # Standardize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Create and train model
        if model_type == 'xgboost':
            model = self.models[model_type](n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
        elif model_type == 'lightgbm':
            model = self.models[model_type](n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
        elif model_type == 'random_forest':
            model = self.models[model_type](n_estimators=100, max_depth=5, random_state=42)
        elif model_type == 'gradient_boosting':
            model = self.models[model_type](n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
        elif model_type == 'ridge':
            model = self.models[model_type](alpha=1.0, random_state=42)
        elif model_type == 'lasso':
            model = self.models[model_type](alpha=0.1, random_state=42)
        else:  # linear_regression
            model = self.models[model_type]()
        
        model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_train_pred = model.predict(X_train_scaled)
        y_test_pred = model.predict(X_test_scaled)
        
        # Calculate performance metrics
        train_mse = mean_squared_error(y_train, y_train_pred)
        test_mse = mean_squared_error(y_test, y_test_pred)
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        
        # Calculate IC
        train_ic, _ = pearsonr(y_train, y_train_pred)
        test_ic, _ = pearsonr(y_test, y_test_pred)
        
        # Save predictions for backtest
        feature_data.loc[train_mask, 'pred'] = y_train_pred
        feature_data.loc[test_mask, 'pred'] = y_test_pred
        
        # Run backtest on predictions
        backtest_results = self._run_backtest(feature_data, 'pred', target_horizon)
        
        # Get feature importance (if available)
        feature_importance = self._get_feature_importance(model, X.columns)
        
        # Prepare model information
        model_info = {
            'model_type': model_type,
            'target_horizon': target_horizon,
            'num_features': len(X.columns),
            'alpha_ids': alpha_ids,
            'feature_names': list(X.columns),
            'feature_importance': feature_importance,
            'performance': {
                'train_mse': train_mse,
                'test_mse': test_mse,
                'train_r2': train_r2,
                'test_r2': test_r2,
                'train_ic': train_ic,
                'test_ic': test_ic
            },
            'backtest': {
                'train': {
                    'sharpe_ratio': backtest_results['train']['sharpe_ratio'],
                    'ann_return': backtest_results['train']['ann_return'],
                    'ann_volatility': backtest_results['train']['ann_volatility'],
                    'win_rate': backtest_results['train']['win_rate']
                },
                'test': {
                    'sharpe_ratio': backtest_results['test']['sharpe_ratio'],
                    'ann_return': backtest_results['test']['ann_return'],
                    'ann_volatility': backtest_results['test']['ann_volatility'],
                    'win_rate': backtest_results['test']['win_rate']
                }
            }
        }
        
        print(f"Model training complete:")
        print(f"  Train IC: {train_ic:.4f}, Test IC: {test_ic:.4f}")
        print(f"  Train Sharpe: {backtest_results['train']['sharpe_ratio']:.2f}, Test Sharpe: {backtest_results['test']['sharpe_ratio']:.2f}")
        
        return model_info
    
    def benchmark_models(self, alpha_ids, target_horizon=5, min_date=None, max_date=None):
        """
        Benchmark multiple model types using the specified alpha factors.
        
        Parameters:
        -----------
        alpha_ids : list
            List of alpha IDs to include in the model
        target_horizon : int
            Target prediction horizon in days (1, 5, or 20)
        min_date : datetime, optional
            Minimum date for training
        max_date : datetime, optional
            Maximum date for training
            
        Returns:
        --------
        dict
            Dictionary containing benchmark results for each model type
        """
        print(f"\nBenchmarking models with {len(alpha_ids)} alpha factors...")
        
        benchmark_results = {}
        
        for model_type in self.models.keys():
            print(f"Testing {model_type}...")
            
            try:
                model_info = self.build_model(alpha_ids, model_type, target_horizon, min_date, max_date)
                benchmark_results[model_type] = {
                    'train_ic': model_info['performance']['train_ic'],
                    'test_ic': model_info['performance']['test_ic'],
                    'train_sharpe': model_info['backtest']['train']['sharpe_ratio'],
                    'test_sharpe': model_info['backtest']['test']['sharpe_ratio']
                }
            except Exception as e:
                print(f"Error benchmarking {model_type}: {e}")
                benchmark_results[model_type] = {
                    'train_ic': None,
                    'test_ic': None,
                    'train_sharpe': None,
                    'test_sharpe': None,
                    'error': str(e)
                }
        
        # Find best model based on test IC
        best_model = max(benchmark_results.items(), key=lambda x: x[1]['test_ic'] if x[1]['test_ic'] is not None else -float('inf'))
        
        print(f"Benchmark complete. Best model: {best_model[0]} (Test IC: {best_model[1]['test_ic']:.4f})")
        
        return benchmark_results
    
    def _prepare_feature_data(self, alphas, min_date=None, max_date=None):
        """
        Prepare feature data for model training.
        
        Parameters:
        -----------
        alphas : list
            List of alpha factor information
        min_date : datetime, optional
            Minimum date for data
        max_date : datetime, optional
            Maximum date for data
            
        Returns:
        --------
        DataFrame
            DataFrame containing feature data
        """
        # Start with base market data (just the columns we need)
        base_cols = ['date', 'symbol'] + [f'future_return_{w}d' for w in [1, 5, 20]]
        feature_data = self.market_data[base_cols].copy()
        
        # Filter by date if specified
        if min_date is not None:
            feature_data = feature_data[feature_data['date'] >= pd.to_datetime(min_date)]
        if max_date is not None:
            feature_data = feature_data[feature_data['date'] <= pd.to_datetime(max_date)]
        
        # Add alpha factors as features
        for alpha in alphas:
            alpha_name = alpha['name']
            alpha_type = alpha['type']
            params = alpha['params']
            
            # Regenerate alpha values
            alpha_data, _ = self.alpha_generator.generate_alpha(alpha_type, params)
            
            # Merge alpha values into feature data
            feature_data = feature_data.merge(
                alpha_data[['date', 'symbol', alpha_name]],
                on=['date', 'symbol'],
                how='left'
            )
        
        # Drop rows with missing values
        feature_data = feature_data.dropna()
        
        return feature_data
    
    def _run_backtest(self, data, pred_col, target_horizon):
        """
        Run a backtest on model predictions.
        
        Parameters:
        -----------
        data : DataFrame
            DataFrame containing predictions
        pred_col : str
            Column name for predictions
        target_horizon : int
            Target prediction horizon in days
            
        Returns:
        --------
        dict
            Dictionary containing backtest results
        """
        # Split data into train and test sets
        dates = data['date'].unique()
        train_dates = dates[:int(0.7 * len(dates))]
        test_dates = dates[int(0.7 * len(dates)):]
        
        train_data = data[data['date'].isin(train_dates)]
        test_data = data[data['date'].isin(test_dates)]
        
        # Calculate portfolio returns for train and test sets
        train_results = self._calculate_portfolio_returns(train_data, pred_col, target_horizon)
        test_results = self._calculate_portfolio_returns(test_data, pred_col, target_horizon)
        
        return {
            'train': train_results,
            'test': test_results
        }
    
    def _calculate_portfolio_returns(self, data, pred_col, target_horizon):
        """
        Calculate portfolio returns based on predictions.
        
        Parameters:
        -----------
        data : DataFrame
            DataFrame containing predictions
        pred_col : str
            Column name for predictions
        target_horizon : int
            Target prediction horizon in days
            
        Returns:
        --------
        dict
            Dictionary containing portfolio performance metrics
        """
        # Get target return column
        target_col = f'future_return_{target_horizon}d'
        
        # Calculate portfolio weights
        data['weight'] = data.groupby('date')[pred_col].transform(
            lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0
        )
        
        # Calculate weighted returns
        data['weighted_return'] = data['weight'] * data[target_col]
        
        # Calculate portfolio returns by date
        portfolio_returns = data.groupby('date')['weighted_return'].sum() / data.groupby('date')['weight'].apply(lambda x: np.sum(np.abs(x)))
        
        # Calculate performance metrics
        ann_factor = 252 / target_horizon
        ann_return = portfolio_returns.mean() * ann_factor
        ann_volatility = portfolio_returns.std() * np.sqrt(ann_factor)
        sharpe_ratio = ann_return / ann_volatility if ann_volatility > 0 else 0
        
        return {
            'portfolio_returns': portfolio_returns,
            'ann_return': ann_return,
            'ann_volatility': ann_volatility,
            'sharpe_ratio': sharpe_ratio,
            'win_rate': (portfolio_returns > 0).mean()
        }
    
    def _get_feature_importance(self, model, feature_names):
        """
        Get feature importance from the model if available.
        
        Parameters:
        -----------
        model : object
            Trained model
        feature_names : list
            List of feature names
            
        Returns:
        --------
        dict
            Dictionary mapping feature names to importance scores
        """
        feature_importance = {}
        
        try:
            if hasattr(model, 'feature_importances_'):
                # For tree-based models
                importances = model.feature_importances_
                for name, importance in zip(feature_names, importances):
                    feature_importance[name] = float(importance)
            elif hasattr(model, 'coef_'):
                # For linear models
                importances = np.abs(model.coef_)
                for name, importance in zip(feature_names, importances):
                    feature_importance[name] = float(importance)
        except:
            pass
        
        return feature_importance

# Initialize alpha generator for the modeling agent
alpha_generator = AlphaGenerator(market_data)

# Initialize alpha modeling agent
alpha_modeling_agent = AlphaModelingAgent(market_data, alpha_base)
print("Alpha modeling agent initialized.")


class AlphaGPT:
    """
    Main interface for Alpha-GPT 2.0, integrating alpha mining and modeling agents.
    """
    def __init__(self, market_data, alpha_mining_agent, alpha_modeling_agent):
        self.market_data = market_data
        self.alpha_mining_agent = alpha_mining_agent
        self.alpha_modeling_agent = alpha_modeling_agent
        self.conversation_history = []
    
    def process_user_input(self, user_input):
        """
        Process user input and generate a response.
        
        Parameters:
        -----------
        user_input : str
            User input text
            
        Returns:
        --------
        str
            Response to the user
        """
        # Add user input to conversation history
        self.conversation_history.append({"role": "user", "content": user_input})
        
        # Analyze user input to determine intent
        intent = self._determine_intent(user_input)
        
        # Process intent
        if intent == 'generate_alphas':
            response = self._process_generate_alphas(user_input)
        elif intent == 'search_alphas':
            response = self._process_search_alphas(user_input)
        elif intent == 'build_model':
            response = self._process_build_model(user_input)
        elif intent == 'benchmark_models':
            response = self._process_benchmark_models(user_input)
        else:
            response = "I'm not sure how to help with that. You can ask me to generate alpha factors based on a trading idea, search for existing alphas, build a model with specific alphas, or benchmark different model types."
        
        # Add response to conversation history
        self.conversation_history.append({"role": "assistant", "content": response})
        
        return response
    
    def _determine_intent(self, user_input):
        """
        Determine the user's intent from their input.
        
        Parameters:
        -----------
        user_input : str
            User input text
            
        Returns:
        --------
        str
            Determined intent
        """
        user_input = user_input.lower()
        
        # Define keywords for each intent
        generate_keywords = ['generate', 'create', 'make', 'trading idea', 'alpha factor', 'alpha based on']
        search_keywords = ['search', 'find', 'look for', 'existing', 'similar']
        build_keywords = ['build', 'train', 'create model', 'combine alphas']
        benchmark_keywords = ['benchmark', 'compare', 'test models', 'evaluate models']
        
        # Check for each intent
        if any(keyword in user_input for keyword in generate_keywords):
            return 'generate_alphas'
        elif any(keyword in user_input for keyword in search_keywords):
            return 'search_alphas'
        elif any(keyword in user_input for keyword in build_keywords):
            return 'build_model'
        elif any(keyword in user_input for keyword in benchmark_keywords):
            return 'benchmark_models'
        else:
            return 'unknown'
    
    def _process_generate_alphas(self, user_input):
        """
        Process a request to generate alpha factors.
        
        Parameters:
        -----------
        user_input : str
            User input text
            
        Returns:
        --------
        str
            Response to the user
        """
        # Extract trading idea from user input
        trading_idea = user_input
        
        # Generate alphas based on trading idea
        generated_alphas = self.alpha_mining_agent.process_trading_idea(trading_idea)
        
        # Prepare response
        if generated_alphas:
            response = f"I've generated {len(generated_alphas)} alpha factors based on your trading idea:\n\n"
            
            for i, alpha in enumerate(generated_alphas):
                response += f"{i+1}. {alpha['name']} (ID: {alpha['id']})\n"
                response += f"   Description: {alpha['description']}\n"
                response += f"   Sharpe Ratio: {alpha['sharpe_ratio']:.2f}, IC (5d): {alpha['ic_5d']:.4f}\n\n"
            
            response += "You can build a model using these alphas by providing their IDs, or ask me to generate more variations."
        else:
            response = "I couldn't generate any alpha factors based on your trading idea. Please try a different idea or provide more details."
        
        return response
    
    def _process_search_alphas(self, user_input):
        """
        Process a request to search for existing alphas.
        
        Parameters:
        -----------
        user_input : str
            User input text
            
        Returns:
        --------
        str
            Response to the user
        """
        # Extract search query from user input
        query = user_input
        
        # Search for existing alphas and generate enhanced variations
        existing_alphas, enhanced_alphas = self.alpha_mining_agent.search_and_enhance_alphas(query)
        
        # Prepare response
        response = ""
        
        if existing_alphas:
            response += f"I found {len(existing_alphas)} existing alpha factors matching your query:\n\n"
            
            for i, alpha in enumerate(existing_alphas[:5]):  # Show only top 5
                response += f"{i+1}. {alpha['name']} (ID: {alpha['id']})\n"
                response += f"   Description: {alpha['description']}\n"
                response += f"   Sharpe Ratio: {alpha['backtest']['sharpe_ratio']:.2f}, IC (5d): {alpha['evaluation']['5d']['ic']:.4f}\n\n"
            
            if len(existing_alphas) > 5:
                response += f"(and {len(existing_alphas) - 5} more...)\n\n"
        else:
            response += "I couldn't find any existing alpha factors matching your query.\n\n"
        
        if enhanced_alphas:
            response += f"I've also generated {len(enhanced_alphas)} enhanced variations based on the existing alphas:\n\n"
            
            for i, alpha in enumerate(enhanced_alphas):
                response += f"{i+1}. {alpha['name']} (ID: {alpha['id']})\n"
                response += f"   Description: {alpha['description']}\n"
                response += f"   Sharpe Ratio: {alpha['sharpe_ratio']:.2f}, IC (5d): {alpha['ic_5d']:.4f}\n"
                response += f"   Enhanced from: {alpha['enhanced_from']}\n\n"
            
            response += "You can build a model using these alphas by providing their IDs."
        
        return response
    
    def _process_build_model(self, user_input):
        """
        Process a request to build a model with specific alphas.
        
        Parameters:
        -----------
        user_input : str
            User input text
            
        Returns:
        --------
        str
            Response to the user
        """
        # Extract alpha IDs from user input
        alpha_ids = self._extract_alpha_ids(user_input)
        
        if not alpha_ids:
            return "I couldn't identify any alpha IDs in your request. Please provide the IDs of the alphas you want to include in the model."
        
        # Extract model type from user input (default to xgboost)
        model_type = 'xgboost'
        for model in self.alpha_modeling_agent.models.keys():
            if model in user_input.lower():
                model_type = model
                break
        
        # Extract target horizon from user input (default to 5 days)
        target_horizon = 5
        if '1-day' in user_input or '1 day' in user_input:
            target_horizon = 1
        elif '20-day' in user_input or '20 day' in user_input:
            target_horizon = 20
        
        # Build model
        try:
            model_info = self.alpha_modeling_agent.build_model(alpha_ids, model_type, target_horizon)
            
            # Prepare response
            response = f"I've built a {model_type} model to predict {target_horizon}-day returns using {len(alpha_ids)} alpha factors.\n\n"
            
            response += "Model Performance:\n"
            response += f"- Train IC: {model_info['performance']['train_ic']:.4f}, Test IC: {model_info['performance']['test_ic']:.4f}\n"
            response += f"- Train R²: {model_info['performance']['train_r2']:.4f}, Test R²: {model_info['performance']['test_r2']:.4f}\n"
            response += f"- Train Sharpe: {model_info['backtest']['train']['sharpe_ratio']:.2f}, Test Sharpe: {model_info['backtest']['test']['sharpe_ratio']:.2f}\n"
            response += f"- Train Return: {model_info['backtest']['train']['ann_return']:.2%}, Test Return: {model_info['backtest']['test']['ann_return']:.2%}\n"
            
            # Include feature importance if available
            if model_info['feature_importance']:
                response += "\nFeature Importance (top 5):\n"
                
                # Sort features by importance
                sorted_features = sorted(
                    model_info['feature_importance'].items(),
                    key=lambda x: x[1],
                    reverse=True
                )
                
                for name, importance in sorted_features[:5]:
                    response += f"- {name}: {importance:.4f}\n"
            
            response += "\nYou can benchmark different model types with the same alpha factors to find the best model."
            
        except Exception as e:
            response = f"Error building model: {str(e)}"
        
        return response
    
    def _process_benchmark_models(self, user_input):
        """
        Process a request to benchmark different model types.
        
        Parameters:
        -----------
        user_input : str
            User input text
            
        Returns:
        --------
        str
            Response to the user
        """
        # Extract alpha IDs from user input
        alpha_ids = self._extract_alpha_ids(user_input)
        
        if not alpha_ids:
            return "I couldn't identify any alpha IDs in your request. Please provide the IDs of the alphas you want to include in the benchmark."
        
        # Extract target horizon from user input (default to 5 days)
        target_horizon = 5
        if '1-day' in user_input or '1 day' in user_input:
            target_horizon = 1
        elif '20-day' in user_input or '20 day' in user_input:
            target_horizon = 20
        
        # Run benchmark
        try:
            benchmark_results = self.alpha_modeling_agent.benchmark_models(alpha_ids, target_horizon)
            
            # Prepare response
            response = f"I've benchmarked different model types to predict {target_horizon}-day returns using {len(alpha_ids)} alpha factors.\n\n"
            
            response += "Model Comparison (Test IC / Test Sharpe):\n"
            
            # Sort models by test IC
            sorted_models = sorted(
                benchmark_results.items(),
                key=lambda x: x[1]['test_ic'] if x[1]['test_ic'] is not None else -float('inf'),
                reverse=True
            )
            
            for model_type, metrics in sorted_models:
                if metrics['test_ic'] is not None:
                    response += f"- {model_type}: {metrics['test_ic']:.4f} / {metrics['test_sharpe']:.2f}\n"
                else:
                    response += f"- {model_type}: Error - {metrics.get('error', 'Unknown error')}\n"
            
            # Recommend best model
            best_model = sorted_models[0][0]
            response += f"\nBased on the benchmark results, the {best_model} model performs best for this set of alpha factors and prediction horizon."
            
        except Exception as e:
            response = f"Error running benchmark: {str(e)}"
        
        return response
    
    def _extract_alpha_ids(self, text):
        """
        Extract alpha IDs from text.
        
        Parameters:
        -----------
        text : str
            Text to extract alpha IDs from
            
        Returns:
        --------
        list
            List of extracted alpha IDs
        """
        # Try to extract IDs in the format "ID: X" or "ID:X" or "ID X"
        import re
        
        alpha_ids = []
        
        # Look for patterns like "ID: X", "ID:X", "ID X", "alpha X", "alphas X,Y,Z"
        patterns = [
            r'ID:?\s*(\d+)',
            r'alpha\s+(\d+)',
            r'alphas\s+([\d,\s]+)'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            
            for match in matches:
                if ',' in match:
                    # Handle comma-separated list
                    for id_str in match.split(','):
                        try:
                            alpha_id = int(id_str.strip())
                            alpha_ids.append(alpha_id)
                        except:
                            pass
                else:
                    try:
                        alpha_id = int(match.strip())
                        alpha_ids.append(alpha_id)
                    except:
                        pass
        
        return alpha_ids

# Initialize the Alpha-GPT interface
alpha_gpt = AlphaGPT(market_data, alpha_mining_agent, alpha_modeling_agent)
print("Alpha-GPT 2.0 initialized and ready to process user inputs.")

def simulate_interaction():
    """
    Simulate a series of interactions with Alpha-GPT 2.0.
    """
    print("\n" + "="*80)
    print("Starting simulation of interaction with Alpha-GPT 2.0")
    print("="*80 + "\n")
    
    # Interaction 1: Generate alpha factors based on a trading idea
    user_input = "I'm interested in a momentum strategy that buys stocks with strong recent performance. Can you generate some alpha factors based on this idea?"
    print(f"User: {user_input}")
    response = alpha_gpt.process_user_input(user_input)
    print(f"Alpha-GPT 2.0: {response}\n")
    
    # Extract alpha IDs from the first response
    import re
    alpha_ids = re.findall(r'ID: (\d+)', response)
    alpha_ids = [int(id) for id in alpha_ids]
    
    # Interaction 2: Search for existing alphas
    user_input = "Search for mean reversion alphas"
    print(f"User: {user_input}")
    response = alpha_gpt.process_user_input(user_input)
    print(f"Alpha-GPT 2.0: {response}\n")
    
    # Extract more alpha IDs from the second response
    more_alpha_ids = re.findall(r'ID: (\d+)', response)
    more_alpha_ids = [int(id) for id in more_alpha_ids]
    
    # Combine alpha IDs (use up to 5)
    combined_alpha_ids = alpha_ids + more_alpha_ids
    combined_alpha_ids = combined_alpha_ids[:min(5, len(combined_alpha_ids))]
    
    # Interaction 3: Build a model with selected alphas
    user_input = f"Build a model using alpha IDs: {', '.join(map(str, combined_alpha_ids))}"
    print(f"User: {user_input}")
    response = alpha_gpt.process_user_input(user_input)
    print(f"Alpha-GPT 2.0: {response}\n")
    
    # Interaction 4: Benchmark different model types
    user_input = f"Benchmark different model types using alpha IDs: {', '.join(map(str, combined_alpha_ids))}"
    print(f"User: {user_input}")
    response = alpha_gpt.process_user_input(user_input)
    print(f"Alpha-GPT 2.0: {response}\n")
    
    # Interaction 5: Generate new alphas with a different strategy
    user_input = "Can you generate some alpha factors based on volume changes? I'm interested in strategies that exploit unusual trading volume."
    print(f"User: {user_input}")
    response = alpha_gpt.process_user_input(user_input)
    print(f"Alpha-GPT 2.0: {response}\n")
    
    print("="*80)
    print("Simulation complete")
    print("="*80 + "\n")

# Run the simulation
simulate_interaction()

def analyze_performance():
    """
    Analyze the performance of Alpha-GPT 2.0.
    """
    print("\n" + "="*80)
    print("Performance Analysis of Alpha-GPT 2.0")
    print("="*80 + "\n")
    
    # Load all alphas from the alpha base
    alphas = alpha_base.alphas
    
    if not alphas:
        print("No alphas in the alpha base to analyze.")
        return
    
    # Calculate statistics
    num_alphas = len(alphas)
    alpha_types = {}
    sharpe_ratios = []
    ics = []
    
    for alpha in alphas:
        # Count alpha types
        alpha_type = alpha.get('type', 'unknown')
        alpha_types[alpha_type] = alpha_types.get(alpha_type, 0) + 1
        
        # Collect performance metrics
        backtest = alpha.get('backtest', {})
        evaluation = alpha.get('evaluation', {}).get('5d', {})
        
        sharpe_ratio = backtest.get('sharpe_ratio')
        ic = evaluation.get('ic')
        
        if sharpe_ratio is not None:
            sharpe_ratios.append(sharpe_ratio)
        
        if ic is not None:
            ics.append(ic)
    
    # Print statistics
    print(f"Total number of alphas: {num_alphas}")
    
    print("\nAlpha Types:")
    for alpha_type, count in alpha_types.items():
        print(f"  {alpha_type}: {count} ({count/num_alphas:.1%})")
    
    if sharpe_ratios:
        print("\nSharpe Ratio Statistics:")
        print(f"  Average: {np.mean(sharpe_ratios):.2f}")
        print(f"  Median: {np.median(sharpe_ratios):.2f}")
        print(f"  Min: {min(sharpe_ratios):.2f}")
        print(f"  Max: {max(sharpe_ratios):.2f}")
        
        # Plot histogram of Sharpe ratios
        plt.figure(figsize=(10, 6))
        plt.hist(sharpe_ratios, bins=20, alpha=0.7)
        plt.axvline(np.mean(sharpe_ratios), color='red', linestyle='dashed', linewidth=1, label=f'Mean: {np.mean(sharpe_ratios):.2f}')
        plt.title('Distribution of Sharpe Ratios')
        plt.xlabel('Sharpe Ratio')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('sharpe_ratio_distribution.png')
        plt.close()
    
    if ics:
        print("\nInformation Coefficient (IC) Statistics:")
        print(f"  Average: {np.mean(ics):.4f}")
        print(f"  Median: {np.median(ics):.4f}")
        print(f"  Min: {min(ics):.4f}")
        print(f"  Max: {max(ics):.4f}")
        
        # Plot histogram of ICs
        plt.figure(figsize=(10, 6))
        plt.hist(ics, bins=20, alpha=0.7)
        plt.axvline(np.mean(ics), color='red', linestyle='dashed', linewidth=1, label=f'Mean: {np.mean(ics):.4f}')
        plt.title('Distribution of Information Coefficients (IC)')
        plt.xlabel('IC')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('ic_distribution.png')
        plt.close()
    
    # Select top performing alphas
    if sharpe_ratios:
        top_sharpe_alphas = sorted(
            [(i, alpha) for i, alpha in enumerate(alphas) if alpha.get('backtest', {}).get('sharpe_ratio') is not None],
            key=lambda x: x[1]['backtest']['sharpe_ratio'],
            reverse=True
        )[:5]
        
        print("\nTop 5 Alphas by Sharpe Ratio:")
        for i, (_, alpha) in enumerate(top_sharpe_alphas):
            print(f"  {i+1}. {alpha['name']} (ID: {alpha['id']})")
            print(f"     Type: {alpha['type']}")
            print(f"     Description: {alpha['description']}")
            print(f"     Sharpe Ratio: {alpha['backtest']['sharpe_ratio']:.2f}")
            print(f"     IC (5d): {alpha['evaluation']['5d']['ic']:.4f}")
            print()
    
    print("="*80)
    print("Performance analysis complete")
    print("="*80 + "\n")

# Run the performance analysis
analyze_performance()

