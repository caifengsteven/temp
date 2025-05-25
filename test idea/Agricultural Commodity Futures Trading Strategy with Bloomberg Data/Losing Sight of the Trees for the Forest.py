import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import logging
from scipy import stats
from sklearn.linear_model import LinearRegression
import warnings
from tqdm import tqdm
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

class DataSimulator:
    """Class to generate simulated market and stock data"""
    
    def __init__(self, n_stocks=50, n_industries=10, start_date='2018-01-01', end_date='2022-12-31'):
        """
        Initialize the data simulator
        
        Parameters:
        -----------
        n_stocks : int
            Number of stocks to simulate
        n_industries : int
            Number of industries to simulate
        start_date : str
            Start date for the simulation
        end_date : str
            End date for the simulation
        """
        self.n_stocks = n_stocks
        self.n_industries = n_industries
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.date_range = pd.date_range(start=self.start_date, end=self.end_date, freq='B')
        self.n_days = len(self.date_range)
        
        # Generate stock and industry names
        self.industry_names = [f'Industry_{i}' for i in range(n_industries)]
        self.stock_symbols = [f'STOCK_{i}' for i in range(n_stocks)]
        
        # Assign stocks to industries
        self.industry_assignments = np.random.randint(0, n_industries, n_stocks)
        self.industry_mapping = {
            self.stock_symbols[i]: self.industry_names[self.industry_assignments[i]] 
            for i in range(n_stocks)
        }
        
        # Company information
        self.company_info = pd.DataFrame({
            'Symbol': self.stock_symbols,
            'Company': [f'Company {i}' for i in range(n_stocks)],
            'Industry': [self.industry_names[ind] for ind in self.industry_assignments],
            'Sector': [f'Sector_{ind % 5}' for ind in self.industry_assignments]
        })
    
    def simulate_market_data(self, volatility=0.01, drift=0.0005):
        """
        Simulate market index data
        
        Parameters:
        -----------
        volatility : float
            Daily volatility of market returns
        drift : float
            Daily drift (expected return) of market
            
        Returns:
        --------
        DataFrame with simulated market data
        """
        # Simulate daily returns
        daily_returns = np.random.normal(drift, volatility, self.n_days)
        
        # Start at price 100
        start_price = 100
        prices = start_price * np.cumprod(1 + daily_returns)
        
        # Create DataFrame
        market_data = pd.DataFrame({
            'Open': prices * np.random.uniform(0.998, 0.999, len(prices)),
            'High': prices * np.random.uniform(1.001, 1.003, len(prices)),
            'Low': prices * np.random.uniform(0.997, 0.999, len(prices)),
            'Close': prices,
            'Adj Close': prices,
            'Volume': np.random.randint(10000000, 100000000, len(prices))
        }, index=self.date_range)
        
        self.market_returns = daily_returns
        self.market_data = market_data
        
        return market_data
    
    def simulate_stock_data(self, market_beta_range=(0.5, 1.5), stock_volatility_range=(0.01, 0.03),
                           industry_volatility=0.015):
        """
        Simulate stock price data with market, industry, and idiosyncratic components
        
        Parameters:
        -----------
        market_beta_range : tuple
            Range of market betas for stocks
        stock_volatility_range : tuple
            Range of idiosyncratic volatilities
        industry_volatility : float
            Volatility of industry factors
            
        Returns:
        --------
        Dictionary of DataFrames with stock price data
        """
        # Generate market betas for stocks
        market_betas = np.random.uniform(market_beta_range[0], market_beta_range[1], self.n_stocks)
        
        # Generate industry factors
        industry_factors = {}
        for industry in self.industry_names:
            # Industry returns are correlated with market but have own shocks
            industry_beta = np.random.uniform(0.8, 1.2)
            industry_shocks = np.random.normal(0, industry_volatility, self.n_days)
            industry_factors[industry] = industry_beta * self.market_returns + industry_shocks
        
        # Generate idiosyncratic volatilities for each stock
        stock_volatilities = np.random.uniform(stock_volatility_range[0], stock_volatility_range[1], self.n_stocks)
        
        # Simulate stock returns and prices
        stock_data = {}
        stock_returns = {}
        
        for i, symbol in enumerate(self.stock_symbols):
            # Get stock's industry
            industry = self.industry_mapping[symbol]
            
            # Components of return
            market_component = market_betas[i] * self.market_returns
            industry_component = industry_factors[industry]
            idiosyncratic_component = np.random.normal(0, stock_volatilities[i], self.n_days)
            
            # Combined return
            returns = market_component + industry_component + idiosyncratic_component
            stock_returns[symbol] = returns
            
            # Convert to price series
            start_price = np.random.uniform(20, 200)  # Random starting price
            prices = start_price * np.cumprod(1 + returns)
            
            # Create DataFrame
            data = pd.DataFrame({
                'Open': prices * np.random.uniform(0.995, 0.998, len(prices)),
                'High': prices * np.random.uniform(1.002, 1.008, len(prices)),
                'Low': prices * np.random.uniform(0.992, 0.998, len(prices)),
                'Close': prices,
                'Adj Close': prices,
                'Volume': np.random.randint(100000, 10000000, len(prices))
            }, index=self.date_range)
            
            stock_data[symbol] = data
        
        self.stock_returns = stock_returns
        self.stock_data = stock_data
        
        return stock_data
    
    def simulate_earnings_announcements(self, quarters_per_year=4, max_surprise_pct=0.2):
        """
        Simulate earnings announcement dates and surprises
        
        Parameters:
        -----------
        quarters_per_year : int
            Number of earnings announcements per year
        max_surprise_pct : float
            Maximum percentage earnings surprise
            
        Returns:
        --------
        DataFrame with earnings announcement data
        """
        announcements = []
        
        # For each stock
        for symbol in self.stock_symbols:
            # Get stock's industry
            industry = self.industry_mapping[symbol]
            
            # Calculate number of quarters in the date range
            n_years = (self.end_date - self.start_date).days / 365.25
            n_quarters = int(n_years * quarters_per_year)
            
            # Generate announcement dates for each quarter
            for q in range(n_quarters):
                # Calculate quarter start and end
                quarter_start = self.start_date + pd.DateOffset(months=3*q)
                quarter_end = quarter_start + pd.DateOffset(months=3)
                
                # Find business days in this quarter
                quarter_dates = pd.date_range(quarter_start, quarter_end, freq='B')
                
                # Select a random date in the quarter for the announcement
                if len(quarter_dates) > 0:
                    announcement_date = np.random.choice(quarter_dates)
                    
                    # Generate earnings surprise
                    eps_estimate = np.random.uniform(0.5, 2.0)
                    surprise_pct = np.random.uniform(-max_surprise_pct, max_surprise_pct)
                    eps_actual = eps_estimate * (1 + surprise_pct)
                    
                    announcements.append({
                        'symbol': symbol,
                        'date': announcement_date,
                        'quarter': q,
                        'eps_estimate': eps_estimate,
                        'eps_actual': eps_actual,
                        'surprise_percent': surprise_pct,
                        'industry': industry
                    })
        
        # Convert to DataFrame
        announcements_df = pd.DataFrame(announcements)
        
        # Ensure announcements are within the date range
        announcements_df = announcements_df[
            (announcements_df['date'] >= self.start_date) & 
            (announcements_df['date'] <= self.end_date)
        ]
        
        # Sort by date
        announcements_df = announcements_df.sort_values('date').reset_index(drop=True)
        
        self.earnings_announcements = announcements_df
        
        return announcements_df
    
    def embed_anomalies(self, pead_magnitude=0.02, momentum_factor=0.01, pairs_factor=0.02):
        """
        Embed market anomalies in the simulated stock data
        
        Parameters:
        -----------
        pead_magnitude : float
            Magnitude of post-earnings announcement drift
        momentum_factor : float
            Magnitude of momentum effect
        pairs_factor : float
            Magnitude of pairs trading opportunity
            
        Returns:
        --------
        Dictionary of DataFrames with updated stock price data
        """
        # 1. Embed post-earnings announcement drift
        for _, event in self.earnings_announcements.iterrows():
            symbol = event['symbol']
            date = event['date']
            surprise_pct = event['surprise_percent']
            
            if abs(surprise_pct) >= 0.05:  # Only significant surprises cause drift
                # Find the index of the announcement date
                if date in self.date_range:
                    date_idx = self.date_range.get_loc(date)
                    
                    # Immediate reaction
                    self.stock_returns[symbol][date_idx] += surprise_pct * pead_magnitude * 5
                    
                    # Drift over next 60 days
                    drift_length = min(60, self.n_days - date_idx - 1)
                    if drift_length > 0:
                        # Gradually decaying drift
                        for d in range(1, drift_length + 1):
                            if date_idx + d < len(self.stock_returns[symbol]):
                                drift_effect = surprise_pct * pead_magnitude * (1 - d/drift_length)
                                self.stock_returns[symbol][date_idx + d] += drift_effect
        
        # 2. Embed momentum effect
        # Calculate rolling 6-month returns for each stock
        formation_period = 126  # ~6 months
        holding_period = 21  # ~1 month
        
        # Identify winner and loser stocks in each period
        for t in range(formation_period, self.n_days, holding_period):
            # Calculate past returns
            past_returns = {}
            for symbol in self.stock_symbols:
                past_return = np.sum(self.stock_returns[symbol][t-formation_period:t])
                past_returns[symbol] = past_return
            
            # Sort stocks by past return
            sorted_stocks = sorted(past_returns.items(), key=lambda x: x[1])
            num_extreme = max(1, self.n_stocks // 10)
            loser_stocks = [s[0] for s in sorted_stocks[:num_extreme]]
            winner_stocks = [s[0] for s in sorted_stocks[-num_extreme:]]
            
            # Add momentum effect for the holding period
            for h in range(holding_period):
                if t + h < self.n_days:
                    # Winners outperform
                    for symbol in winner_stocks:
                        self.stock_returns[symbol][t + h] += momentum_factor
                    
                    # Losers underperform
                    for symbol in loser_stocks:
                        self.stock_returns[symbol][t + h] -= momentum_factor
        
        # 3. Embed pairs trading opportunities
        # Create some cointegrated pairs within industries
        formation_period = 252  # ~1 year
        
        # Group stocks by industry
        industry_stocks = {}
        for symbol, industry in self.industry_mapping.items():
            if industry not in industry_stocks:
                industry_stocks[industry] = []
            industry_stocks[industry].append(symbol)
        
        # For each industry with at least 2 stocks
        for industry, symbols in industry_stocks.items():
            if len(symbols) >= 2:
                # Create pairs
                for i in range(0, len(symbols) - 1, 2):
                    stock1 = symbols[i]
                    stock2 = symbols[i + 1]
                    
                    # Make the pair have similar returns (cointegrated)
                    common_factor = np.random.normal(0, 0.01, self.n_days)
                    
                    # Add common factor to both stocks
                    self.stock_returns[stock1] += common_factor
                    self.stock_returns[stock2] += common_factor
                    
                    # Add occasional divergence that later converges (pairs trading opportunity)
                    for t in range(formation_period, self.n_days, 60):
                        if t + 20 < self.n_days:
                            # Temporary divergence
                            divergence = np.zeros(self.n_days)
                            
                            # Stock1 outperforms temporarily
                            divergence[t:t+10] = np.linspace(0, pairs_factor, 10)
                            divergence[t+10:t+20] = np.linspace(pairs_factor, 0, 10)
                            
                            # Add divergence
                            self.stock_returns[stock1] += divergence
                            self.stock_returns[stock2] -= divergence
        
        # Recalculate price series with the embedded anomalies
        for symbol in self.stock_symbols:
            returns = self.stock_returns[symbol]
            
            # Start price from the original data
            start_price = self.stock_data[symbol]['Adj Close'][0]
            
            # Calculate new price series
            prices = start_price * np.cumprod(1 + returns)
            
            # Update DataFrame
            self.stock_data[symbol]['Adj Close'] = prices
            self.stock_data[symbol]['Close'] = prices
            self.stock_data[symbol]['Open'] = prices * np.random.uniform(0.995, 0.998, len(prices))
            self.stock_data[symbol]['High'] = prices * np.random.uniform(1.002, 1.008, len(prices))
            self.stock_data[symbol]['Low'] = prices * np.random.uniform(0.992, 0.998, len(prices))
        
        return self.stock_data


class AttentionIndexCalculator:
    """Class to calculate market-wide attention index"""
    
    def __init__(self, market_data, stock_data, industry_mapping, lookback_period=252):
        """
        Initialize the attention index calculator
        
        Parameters:
        -----------
        market_data : DataFrame
            Market index price data
        stock_data : dict
            Dictionary of stock price DataFrames
        industry_mapping : dict
            Mapping of stocks to industries
        lookback_period : int
            Number of days for rolling parameter estimation
        """
        self.market_data = market_data
        self.stock_data = stock_data
        self.industry_mapping = industry_mapping
        self.lookback_period = lookback_period
        self.logger = logging.getLogger('AttentionTrader')
    
    def calculate_attention_index(self):
        """
        Calculate the composite attention index
        
        Returns:
        --------
        tuple: (attention_index, attention_deciles)
        """
        # Calculate market returns
        market_returns = self.market_data['Adj Close'].pct_change().dropna()
        
        # Group stocks by industry
        industry_stocks = {}
        for symbol, industry in self.industry_mapping.items():
            if industry not in industry_stocks:
                industry_stocks[industry] = []
            industry_stocks[industry].append(symbol)
        
        # Calculate industry returns
        industry_returns = {}
        for industry, symbols in industry_stocks.items():
            # Calculate equal-weighted industry return
            industry_return_series = pd.Series(0, index=market_returns.index)
            count = 0
            
            for symbol in symbols:
                if symbol in self.stock_data:
                    stock_returns = self.stock_data[symbol]['Adj Close'].pct_change().dropna()
                    # Ensure index alignment
                    common_dates = stock_returns.index.intersection(industry_return_series.index)
                    if len(common_dates) > 0:
                        industry_return_series.loc[common_dates] += stock_returns.loc[common_dates]
                        count += 1
            
            if count > 0:
                industry_return_series = industry_return_series / count
                industry_returns[industry] = industry_return_series
        
        # Calculate abnormal returns for each industry
        abnormal_returns = {}
        
        for industry, returns in industry_returns.items():
            # Skip industries with too few data points
            if len(returns) <= self.lookback_period:
                continue
                
            # Align with market returns
            common_dates = returns.index.intersection(market_returns.index)
            if len(common_dates) <= self.lookback_period:
                continue
                
            industry_returns_aligned = returns.loc[common_dates]
            market_returns_aligned = market_returns.loc[common_dates]
            
            # Calculate abnormal returns using market model
            abnormal_return_series = pd.Series(index=common_dates)
            
            # Skip first lookback period
            for t in range(self.lookback_period, len(common_dates)):
                current_date = common_dates[t]
                lookback_dates = common_dates[t-self.lookback_period:t]
                
                # Fit market model
                X = market_returns_aligned.loc[lookback_dates].values.reshape(-1, 1)
                y = industry_returns_aligned.loc[lookback_dates].values
                
                model = LinearRegression().fit(X, y)
                alpha, beta = model.intercept_, model.coef_[0]
                
                # Calculate abnormal return
                expected_return = alpha + beta * market_returns_aligned.loc[current_date]
                actual_return = industry_returns_aligned.loc[current_date]
                abnormal_return = abs(actual_return - expected_return)
                
                abnormal_return_series.loc[current_date] = abnormal_return
            
            # Remove NaNs
            abnormal_returns[industry] = abnormal_return_series.dropna()
        
        # Convert to DataFrame
        abnormal_df = pd.DataFrame(abnormal_returns).dropna()
        
        # Calculate shock volatilities over rolling window
        shock_vols = abnormal_df.rolling(self.lookback_period).std().dropna()
        
        # Compute weighted composite attention index
        common_dates = abnormal_df.index.intersection(shock_vols.index)
        
        composite_index = pd.Series(index=common_dates)
        
        for date in common_dates:
            weights = 1 / shock_vols.loc[date]
            # Handle NaNs and infinite values
            weights = weights.fillna(0)
            weights = weights.replace([np.inf, -np.inf], 0)
            
            if weights.sum() > 0:
                weights = weights / weights.sum()  # Normalize weights
                composite_index.loc[date] = (abnormal_df.loc[date] * weights).sum()
        
        # Compute yearly decile ranks
        decile_ranks = pd.Series(index=composite_index.index)
        
        for year, group in composite_index.groupby(composite_index.index.year):
            valid_indices = ~group.isna()
            if valid_indices.any():
                try:
                    decile_ranks.loc[group[valid_indices].index] = pd.qcut(
                        group[valid_indices], 10, labels=False, duplicates='drop') + 1
                except Exception as e:
                    self.logger.warning(f"Couldn't compute deciles for year {year}: {e}")
                    # Use a simplified approach if qcut fails
                    sorted_values = group[valid_indices].sort_values()
                    n = len(sorted_values)
                    for i, (date, _) in enumerate(sorted_values.items()):
                        decile = int(i / n * 10) + 1
                        decile_ranks.loc[date] = decile
        
        return composite_index, decile_ranks


class BaseStrategy:
    """Base class for all attention-based strategies"""
    
    def __init__(self, attention_data, price_data):
        """
        Initialize the base strategy
        
        Parameters:
        -----------
        attention_data : Series
            Attention deciles for each date
        price_data : dict
            Dictionary of stock price DataFrames
        """
        self.attention_data = attention_data
        self.price_data = price_data
        self.positions = {}
        self.trades = []
        self.logger = logging.getLogger('AttentionTrader')
    
    def is_high_attention_day(self, date, threshold=7):
        """Check if a given date is a high attention day"""
        if date in self.attention_data.index:
            return self.attention_data.loc[date] >= threshold
        return False
    
    def is_low_attention_day(self, date, threshold=3):
        """Check if a given date is a low attention day"""
        if date in self.attention_data.index:
            return self.attention_data.loc[date] <= threshold
        return False
    
    def record_trade(self, date, symbol, direction, quantity, price, reason):
        """Record a trade for later analysis"""
        self.trades.append({
            'date': date,
            'symbol': symbol,
            'direction': direction,
            'quantity': quantity,
            'price': price,
            'reason': reason
        })
        
        # Update positions
        current_position = self.positions.get(symbol, 0)
        if direction == 'buy':
            self.positions[symbol] = current_position + quantity
        else:  # sell
            self.positions[symbol] = current_position - quantity
        
        self.logger.info(f"Trade: {date}, {symbol}, {direction}, {quantity} @ {price:.2f}, {reason}")
    
    def get_current_positions(self):
        """Get current portfolio positions"""
        return self.positions


class MomentumStrategy(BaseStrategy):
    """Momentum strategy that adjusts based on attention levels"""
    
    def __init__(self, attention_data, price_data, formation_period=126, holding_period=21):
        """
        Initialize the momentum strategy
        
        Parameters:
        -----------
        attention_data : Series
            Attention deciles for each date
        price_data : dict
            Dictionary of stock price DataFrames
        formation_period : int
            Number of days to look back for momentum calculation
        holding_period : int
            Number of days to hold momentum portfolios
        """
        super().__init__(attention_data, price_data)
        self.formation_period = formation_period
        self.holding_period = holding_period
        self.winner_stocks = []
        self.loser_stocks = []
        self.last_rebalance_date = None
    
    def generate_signals(self, current_date, max_positions=20):
        """
        Generate trading signals based on momentum and attention
        
        Parameters:
        -----------
        current_date : datetime
            Current trading date
        max_positions : int
            Maximum number of positions per side (winners/losers)
            
        Returns:
        --------
        tuple: (buy_signals, sell_signals)
        """
        # Check if we need to rebalance
        if self.last_rebalance_date is None or self._days_since_rebalance(current_date) >= self.holding_period:
            # Time to rebalance the portfolio
            self.winner_stocks, self.loser_stocks = self._select_momentum_stocks(current_date, max_positions)
            self.last_rebalance_date = current_date
        
        # Adjust position sizes based on attention level
        attention_factor = self._get_attention_factor(current_date)
        
        buy_signals = []
        sell_signals = []
        
        # If it's a high attention day, reduce momentum effect (less confident in trend continuation)
        # If it's a low attention day, increase momentum effect (more confident in trend continuation)
        
        for symbol in self.winner_stocks:
            if symbol in self.price_data and current_date in self.price_data[symbol].index:
                price = self.price_data[symbol].loc[current_date]['Adj Close']
                signal = {
                    'symbol': symbol,
                    'direction': 'buy',
                    'price': price,
                    'weight': 1.0 * attention_factor,
                    'reason': f"Winner stock, attention_factor={attention_factor:.2f}"
                }
                buy_signals.append(signal)
        
        for symbol in self.loser_stocks:
            if symbol in self.price_data and current_date in self.price_data[symbol].index:
                price = self.price_data[symbol].loc[current_date]['Adj Close']
                signal = {
                    'symbol': symbol,
                    'direction': 'sell',
                    'price': price,
                    'weight': 1.0 * attention_factor,
                    'reason': f"Loser stock, attention_factor={attention_factor:.2f}"
                }
                sell_signals.append(signal)
        
        return buy_signals, sell_signals
    
    def _days_since_rebalance(self, current_date):
        """Calculate business days since last rebalance"""
        if self.last_rebalance_date is None:
            return float('inf')
        
        # Count business days between dates
        return len(pd.date_range(start=self.last_rebalance_date, end=current_date, freq='B')) - 1
    
    def _select_momentum_stocks(self, current_date, max_positions):
        """
        Select winner and loser stocks based on past performance
        
        Parameters:
        -----------
        current_date : datetime
            Current trading date
        max_positions : int
            Maximum number of positions per side
            
        Returns:
        --------
        tuple: (winner_stocks, loser_stocks)
        """
        # Get historical data ending on current_date
        valid_symbols = []
        returns_dict = {}
        
        for symbol, data in self.price_data.items():
            if current_date in data.index:
                # Get data for formation period
                end_idx = data.index.get_loc(current_date)
                if end_idx >= self.formation_period:
                    start_idx = end_idx - self.formation_period
                    price_history = data.iloc[start_idx:end_idx+1]
                    
                    # Calculate cumulative return
                    start_price = price_history.iloc[0]['Adj Close']
                    end_price = price_history.iloc[-1]['Adj Close']
                    cumulative_return = (end_price / start_price) - 1
                    
                    returns_dict[symbol] = cumulative_return
                    valid_symbols.append(symbol)
        
        # Sort stocks by return
        sorted_returns = sorted(returns_dict.items(), key=lambda x: x[1])
        
        # Select top and bottom stocks
        num_positions = min(max_positions, len(sorted_returns) // 10)
        loser_stocks = [s[0] for s in sorted_returns[:num_positions]]
        winner_stocks = [s[0] for s in sorted_returns[-num_positions:]]
        
        return winner_stocks, loser_stocks
    
    def _get_attention_factor(self, current_date):
        """
        Calculate the attention factor for a given date
        
        Parameters:
        -----------
        current_date : datetime
            Current trading date
            
        Returns:
        --------
        float: attention factor for scaling positions
        """
        if current_date in self.attention_data.index:
            attention_decile = self.attention_data.loc[current_date]
            # Scale factor: higher attention = weaker momentum effect
            return max(0.5, 1.5 - 0.1 * attention_decile)
        return 1.0  # Default factor


class PairsStrategy(BaseStrategy):
    """Pairs trading strategy adjusted for attention levels"""
    
    def __init__(self, attention_data, price_data, industry_mapping, formation_period=252, threshold=2.0):
        """
        Initialize the pairs trading strategy
        
        Parameters:
        -----------
        attention_data : Series
            Attention deciles for each date
        price_data : dict
            Dictionary of stock price DataFrames
        industry_mapping : dict
            Mapping of stocks to industries
        formation_period : int
            Number of days to look back for pair selection
        threshold : float
            Number of standard deviations for pair divergence
        """
        super().__init__(attention_data, price_data)
        self.formation_period = formation_period
        self.threshold = threshold
        self.industry_mapping = industry_mapping
        self.pairs = []
        self.active_trades = {}  # {pair_id: {entry_date, long_symbol, short_symbol, etc.}}
        self.last_pairs_update = None
    
    def update_pairs(self, current_date, max_pairs=50):
        """
        Update the list of tradable pairs
        
        Parameters:
        -----------
        current_date : datetime
            Current trading date
        max_pairs : int
            Maximum number of pairs to select
        """
        # Only update pairs periodically (e.g., monthly)
        if (self.last_pairs_update is None or 
            (current_date - self.last_pairs_update).days > 30):
            
            self.logger.info(f"Updating pairs selection on {current_date}")
            self.pairs = self._select_pairs(current_date, max_pairs)
            self.last_pairs_update = current_date
    
    def generate_signals(self, current_date):
        """
        Generate trading signals for pairs based on divergence and attention
        
        Parameters:
        -----------
        current_date : datetime
            Current trading date
            
        Returns:
        --------
        list: trading signals
        """
        signals = []
        
        # Check for new pair trading opportunities
        for pair_id, pair_info in enumerate(self.pairs):
            # Skip if this pair is already in an active trade
            if pair_id in self.active_trades:
                continue
                
            stock1, stock2 = pair_info['stock1'], pair_info['stock2']
            
            # Check if we have price data for both stocks
            if (stock1 in self.price_data and stock2 in self.price_data and
                current_date in self.price_data[stock1].index and 
                current_date in self.price_data[stock2].index):
                
                # Calculate z-score of the pair spread
                z_score = self._calculate_pair_zscore(stock1, stock2, current_date)
                
                if abs(z_score) > self.threshold:
                    # Pair has diverged, potential trading opportunity
                    
                    # Adjust threshold based on attention
                    attention_factor = self._get_attention_factor(current_date)
                    adjusted_threshold = self.threshold * attention_factor
                    
                    if abs(z_score) > adjusted_threshold:
                        # Determine long and short positions
                        if z_score > 0:
                            # stock1 is overvalued relative to stock2
                            long_stock, short_stock = stock2, stock1
                        else:
                            # stock2 is overvalued relative to stock1
                            long_stock, short_stock = stock1, stock2
                        
                        # Record the trade entry
                        self.active_trades[pair_id] = {
                            'entry_date': current_date,
                            'long_symbol': long_stock,
                            'short_symbol': short_stock,
                            'entry_zscore': z_score,
                            'attention_level': self.attention_data.loc[current_date] if current_date in self.attention_data.index else None
                        }
                        
                        # Generate signals
                        long_price = self.price_data[long_stock].loc[current_date]['Adj Close']
                        short_price = self.price_data[short_stock].loc[current_date]['Adj Close']
                        
                        signals.append({
                            'symbol': long_stock,
                            'direction': 'buy',
                            'price': long_price,
                            'reason': f"Pairs entry: long {long_stock}, z-score={z_score:.2f}",
                            'pair_id': pair_id
                        })
                        
                        signals.append({
                            'symbol': short_stock,
                            'direction': 'sell',
                            'price': short_price,
                            'reason': f"Pairs entry: short {short_stock}, z-score={z_score:.2f}",
                            'pair_id': pair_id
                        })
        
        # Check for exit signals on existing trades
        pairs_to_exit = []
        
        for pair_id, trade_info in self.active_trades.items():
            long_stock = trade_info['long_symbol']
            short_stock = trade_info['short_symbol']
            
            # Check if we have price data for both stocks
            if (long_stock in self.price_data and short_stock in self.price_data and
                current_date in self.price_data[long_stock].index and 
                current_date in self.price_data[short_stock].index):
                
                # Calculate current z-score
                z_score = self._calculate_pair_zscore(
                    trade_info['long_symbol'], 
                    trade_info['short_symbol'], 
                    current_date
                )
                
                # Check for convergence or time-based exit (after 20 days)
                entry_date = trade_info['entry_date']
                days_in_trade = (current_date - entry_date).days
                
                if abs(z_score) < 0.5 or days_in_trade > 20:
                    # Exit the trade
                    long_price = self.price_data[long_stock].loc[current_date]['Adj Close']
                    short_price = self.price_data[short_stock].loc[current_date]['Adj Close']
                    
                    signals.append({
                        'symbol': long_stock,
                        'direction': 'sell',
                        'price': long_price,
                        'reason': f"Pairs exit: close long {long_stock}, z-score={z_score:.2f}, days={days_in_trade}",
                        'pair_id': pair_id
                    })
                    
                    signals.append({
                        'symbol': short_stock,
                        'direction': 'buy',
                        'price': short_price,
                        'reason': f"Pairs exit: close short {short_stock}, z-score={z_score:.2f}, days={days_in_trade}",
                        'pair_id': pair_id
                    })
                    
                    pairs_to_exit.append(pair_id)
        
        # Remove exited pairs from active trades
        for pair_id in pairs_to_exit:
            del self.active_trades[pair_id]
        
        return signals
    
    def _select_pairs(self, current_date, max_pairs):
        """
        Select pairs based on historical correlation
        
        Parameters:
        -----------
        current_date : datetime
            Current trading date
        max_pairs : int
            Maximum number of pairs to select
            
        Returns:
        --------
        list: selected pairs
        """
        pairs = []
        
        # Group stocks by industry
        industry_stocks = {}
        for symbol, data in self.price_data.items():
            if symbol in self.industry_mapping:
                industry = self.industry_mapping[symbol]
                if industry not in industry_stocks:
                    industry_stocks[industry] = []
                industry_stocks[industry].append(symbol)
        
        # For each industry, find highly correlated pairs
        for industry, symbols in industry_stocks.items():
            if len(symbols) < 2:
                continue
                
            # Get historical price data for formation period
            price_histories = {}
            for symbol in symbols:
                if current_date in self.price_data[symbol].index:
                    end_idx = self.price_data[symbol].index.get_loc(current_date)
                    if end_idx >= self.formation_period:
                        start_idx = end_idx - self.formation_period
                        price_history = self.price_data[symbol].iloc[start_idx:end_idx+1]['Adj Close']
                        price_histories[symbol] = price_history
            
            # Calculate correlation matrix
            if len(price_histories) >= 2:
                prices_df = pd.DataFrame(price_histories)
                correlation_matrix = prices_df.corr()
                
                # Find highly correlated pairs
                for i, symbol1 in enumerate(prices_df.columns[:-1]):
                    for j, symbol2 in enumerate(prices_df.columns[i+1:], i+1):
                        correlation = correlation_matrix.iloc[i, j]
                        
                        if correlation > 0.7:  # Only consider highly correlated pairs
                            # Calculate spread statistics
                            log_prices1 = np.log(prices_df[symbol1])
                            log_prices2 = np.log(prices_df[symbol2])
                            spread = log_prices1 - log_prices2
                            
                            # Check stationarity of spread
                            try:
                                # Calculate Augmented Dickey-Fuller test
                                from statsmodels.tsa.stattools import adfuller
                                adf_result = adfuller(spread.dropna())
                                
                                # Only add pairs that show signs of stationarity (p-value < 0.1)
                                if adf_result[1] < 0.1:
                                    pairs.append({
                                        'stock1': symbol1,
                                        'stock2': symbol2,
                                        'correlation': correlation,
                                        'adf_pvalue': adf_result[1],
                                        'industry': industry
                                    })
                            except:
                                # Skip if ADF test fails
                                pass
        
        # Sort by correlation (highest first) and select top pairs
        pairs.sort(key=lambda x: x['correlation'], reverse=True)
        return pairs[:max_pairs]
    
    def _calculate_pair_zscore(self, stock1, stock2, current_date):
        """
        Calculate z-score for a given pair on a specific date
        
        Parameters:
        -----------
        stock1 : str
            First stock symbol
        stock2 : str
            Second stock symbol
        current_date : datetime
            Current trading date
            
        Returns:
        --------
        float: z-score of the pair spread
        """
        # Get historical data for formation period
        end_idx1 = self.price_data[stock1].index.get_loc(current_date)
        end_idx2 = self.price_data[stock2].index.get_loc(current_date)
        
        if end_idx1 >= self.formation_period and end_idx2 >= self.formation_period:
            start_idx1 = end_idx1 - self.formation_period
            start_idx2 = end_idx2 - self.formation_period
            
            price_history1 = self.price_data[stock1].iloc[start_idx1:end_idx1+1]['Adj Close']
            price_history2 = self.price_data[stock2].iloc[start_idx2:end_idx2+1]['Adj Close']
            
            # Calculate log price ratio
            log_prices1 = np.log(price_history1)
            log_prices2 = np.log(price_history2)
            spread = log_prices1 - log_prices2
            
            # Calculate z-score
            mean_spread = spread.iloc[:-1].mean()  # Exclude current day
            std_spread = spread.iloc[:-1].std()
            
            if std_spread > 0:
                current_spread = spread.iloc[-1]
                z_score = (current_spread - mean_spread) / std_spread
                return z_score
        
        return 0.0
    
    def _get_attention_factor(self, current_date):
        """
        Calculate the attention factor for a given date
        
        Parameters:
        -----------
        current_date : datetime
            Current trading date
            
        Returns:
        --------
        float: attention factor for scaling positions
        """
        if current_date in self.attention_data.index:
            attention_decile = self.attention_data.loc[current_date]
            # Scale factor: higher attention = stronger pairs trading effect
            return 0.7 + 0.06 * attention_decile
        return 1.0  # Default factor


class PostEarningsAnnouncementDriftStrategy(BaseStrategy):
    """PEAD strategy adjusted for attention levels"""
    
    def __init__(self, attention_data, price_data, earnings_data, drift_period=60):
        """
        Initialize the PEAD strategy
        
        Parameters:
        -----------
        attention_data : Series
            Attention deciles for each date
        price_data : dict
            Dictionary of stock price DataFrames
        earnings_data : DataFrame
            Earnings announcement data
        drift_period : int
            Number of days to hold PEAD trades
        """
        super().__init__(attention_data, price_data)
        self.earnings_data = earnings_data
        self.drift_period = drift_period
        self.active_trades = {}  # {symbol: {entry_date, surprise_pct, etc.}}
    
    def generate_signals(self, current_date):
        """
        Generate trading signals based on earnings and attention
        
        Parameters:
        -----------
        current_date : datetime
            Current trading date
            
        Returns:
        --------
        list: trading signals
        """
        signals = []
        
        # Check for new earnings announcements
        today_earnings = self.earnings_data[self.earnings_data['date'] == current_date]
        
        for _, event in today_earnings.iterrows():
            symbol = event['symbol']
            surprise_pct = event['surprise_percent']
            
            # Only trade significant surprises (>= 5% surprise)
            if abs(surprise_pct) >= 0.05 and symbol in self.price_data:
                # Get current price
                if current_date in self.price_data[symbol].index:
                    price = self.price_data[symbol].loc[current_date]['Adj Close']
                    
                    # Adjust position size based on attention
                    attention_factor = self._get_attention_factor(current_date)
                    
                    # Determine trade direction based on surprise
                    direction = 'buy' if surprise_pct > 0 else 'sell'
                    
                    # Add to active trades
                    self.active_trades[symbol] = {
                        'entry_date': current_date,
                        'direction': direction,
                        'surprise_pct': surprise_pct,
                        'entry_price': price,
                        'attention_factor': attention_factor
                    }
                    
                    # Generate signal
                    signals.append({
                        'symbol': symbol,
                        'direction': direction,
                        'price': price,
                        'weight': abs(surprise_pct) * attention_factor,
                        'reason': f"Earnings surprise {surprise_pct:.2%}, attention_factor={attention_factor:.2f}"
                    })
        
        # Check for exit signals
        symbols_to_exit = []
        
        for symbol, trade_info in self.active_trades.items():
            entry_date = trade_info['entry_date']
            days_in_trade = (current_date - entry_date).days
            
            # Exit after drift period
            if days_in_trade >= self.drift_period and symbol in self.price_data:
                if current_date in self.price_data[symbol].index:
                    exit_price = self.price_data[symbol].loc[current_date]['Adj Close']
                    
                    # Generate exit signal (opposite of entry)
                    exit_direction = 'sell' if trade_info['direction'] == 'buy' else 'buy'
                    
                    signals.append({
                        'symbol': symbol,
                        'direction': exit_direction,
                        'price': exit_price,
                        'reason': f"PEAD exit after {days_in_trade} days"
                    })
                    
                    symbols_to_exit.append(symbol)
        
        # Remove exited symbols
        for symbol in symbols_to_exit:
            del self.active_trades[symbol]
        
        return signals
    
    def _get_attention_factor(self, current_date):
        """
        Calculate the attention factor for a given date
        
        Parameters:
        -----------
        current_date : datetime
            Current trading date
            
        Returns:
        --------
        float: attention factor for scaling positions
        """
        if current_date in self.attention_data.index:
            attention_decile = self.attention_data.loc[current_date]
            # Scale factor: higher attention = stronger PEAD effect
            return 0.5 + 0.1 * attention_decile
        return 1.0  # Default factor


class PortfolioManager:
    """Handle portfolio construction and risk management"""
    
    def __init__(self, initial_capital=1000000):
        """
        Initialize the portfolio manager
        
        Parameters:
        -----------
        initial_capital : float
            Initial portfolio capital
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.positions = {}  # {symbol: {'quantity': qty, 'price': price}}
        self.trades = []
        self.daily_values = []
        self.logger = logging.getLogger('AttentionTrader')
    
    def process_signals(self, signals, current_date, prices_data):
        """
        Process trading signals and adjust portfolio
        
        Parameters:
        -----------
        signals : list
            Trading signals from strategies
        current_date : datetime
            Current trading date
        prices_data : dict
            Dictionary of stock price DataFrames
        """
        # Group signals by symbol
        signals_by_symbol = {}
        for signal in signals:
            symbol = signal['symbol']
            if symbol not in signals_by_symbol:
                signals_by_symbol[symbol] = []
            signals_by_symbol[symbol].append(signal)
        
        # Execute trades
        for symbol, symbol_signals in signals_by_symbol.items():
            # Determine net trade direction and size
            buy_signals = [s for s in symbol_signals if s['direction'] == 'buy']
            sell_signals = [s for s in symbol_signals if s['direction'] == 'sell']
            
            net_buy_weight = sum(s.get('weight', 1.0) for s in buy_signals)
            net_sell_weight = sum(s.get('weight', 1.0) for s in sell_signals)
            
            net_weight = net_buy_weight - net_sell_weight
            
            # Calculate target position
            if net_weight != 0:
                # Maximum allocation per position (5% of portfolio)
                max_position_value = self.get_portfolio_value(current_date, prices_data) * 0.05
                
                # Scale position size by weight
                target_value = max_position_value * (abs(net_weight) / 1.0)
                
                # Calculate current position
                current_quantity = self.positions.get(symbol, {}).get('quantity', 0)
                
                # Get current price
                if symbol in prices_data and current_date in prices_data[symbol].index:
                    current_price = prices_data[symbol].loc[current_date]['Adj Close']
                    
                    # Calculate target quantity
                    target_quantity = int(target_value / current_price)
                    
                    # Adjust direction
                    if net_weight < 0:
                        target_quantity = -target_quantity
                    
                    # Calculate trade quantity
                    trade_quantity = target_quantity - current_quantity
                    
                    if trade_quantity != 0:
                        # Record trade
                        trade_direction = 'buy' if trade_quantity > 0 else 'sell'
                        trade_value = abs(trade_quantity) * current_price
                        
                        # Check if we have enough capital for buys
                        if trade_direction == 'buy' and trade_value > self.current_capital:
                            # Scale back the trade if not enough capital
                            trade_quantity = int(self.current_capital / current_price)
                            self.logger.warning(f"Reduced trade size due to capital constraints: {symbol}")
                        
                        if trade_quantity != 0:
                            # Execute the trade
                            self._execute_trade(symbol, trade_direction, abs(trade_quantity), 
                                               current_price, current_date, 
                                               " & ".join([s['reason'] for s in symbol_signals]))
        
        # Record daily portfolio value
        portfolio_value = self.get_portfolio_value(current_date, prices_data)
        self.daily_values.append({
            'date': current_date,
            'portfolio_value': portfolio_value
        })
    
    def _execute_trade(self, symbol, direction, quantity, price, date, reason):
        """
        Execute a trade by updating positions and capital
        
        Parameters:
        -----------
        symbol : str
            Stock symbol
        direction : str
            Trade direction ('buy' or 'sell')
        quantity : int
            Number of shares
        price : float
            Share price
        date : datetime
            Trade date
        reason : str
            Reason for the trade
        """
        # Update cash
        trade_value = quantity * price
        if direction == 'buy':
            self.current_capital -= trade_value
        else:  # sell
            self.current_capital += trade_value
        
        # Update positions
        if symbol not in self.positions:
            self.positions[symbol] = {'quantity': 0, 'price': 0}
        
        if direction == 'buy':
            new_qty = self.positions[symbol]['quantity'] + quantity
            # Update average price
            if new_qty > 0:
                self.positions[symbol]['price'] = (
                    (self.positions[symbol]['quantity'] * self.positions[symbol]['price'] + quantity * price) / 
                    new_qty
                )
            self.positions[symbol]['quantity'] = new_qty
        else:  # sell
            self.positions[symbol]['quantity'] -= quantity
            # If position is closed, reset price
            if self.positions[symbol]['quantity'] == 0:
                self.positions[symbol]['price'] = 0
        
        # Record the trade
        self.trades.append({
            'date': date,
            'symbol': symbol,
            'direction': direction,
            'quantity': quantity,
            'price': price,
            'value': trade_value,
            'reason': reason
        })
        
        self.logger.info(f"Executed: {date}, {symbol}, {direction}, {quantity} @ {price:.2f}, {reason}")
    
    def get_portfolio_value(self, current_date, prices_data):
        """
        Calculate current portfolio value
        
        Parameters:
        -----------
        current_date : datetime
            Current date
        prices_data : dict
            Dictionary of stock price DataFrames
            
        Returns:
        --------
        float: total portfolio value
        """
        positions_value = 0
        
        for symbol, position_info in self.positions.items():
            quantity = position_info['quantity']
            
            if quantity != 0 and symbol in prices_data and current_date in prices_data[symbol].index:
                current_price = prices_data[symbol].loc[current_date]['Adj Close']
                positions_value += quantity * current_price
        
        return self.current_capital + positions_value
    
    def get_historical_performance(self):
        """
        Get historical portfolio performance
        
        Returns:
        --------
        DataFrame: historical portfolio values and returns
        """
        if not self.daily_values:
            return pd.DataFrame()
            
        df = pd.DataFrame(self.daily_values)
        df.set_index('date', inplace=True)
        
        # Calculate returns
        df['daily_return'] = df['portfolio_value'].pct_change()
        df['cumulative_return'] = (1 + df['daily_return']).cumprod() - 1
        
        return df
    
    def get_performance_metrics(self, benchmark_returns=None):
        """
        Calculate portfolio performance metrics
        
        Parameters:
        -----------
        benchmark_returns : Series
            Benchmark returns for comparison
            
        Returns:
        --------
        dict: performance metrics
        """
        perf_data = self.get_historical_performance()
        
        if perf_data.empty:
            return {}
            
        daily_returns = perf_data['daily_return'].dropna()
        
        if len(daily_returns) < 5:
            return {}
            
        # Calculate metrics
        total_return = perf_data['cumulative_return'].iloc[-1]
        annualized_return = (1 + total_return) ** (252 / len(daily_returns)) - 1
        volatility = daily_returns.std() * (252 ** 0.5)
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
        
        # Maximum drawdown
        cumulative_returns = perf_data['cumulative_return']
        max_drawdown = 0
        peak = cumulative_returns.iloc[0]
        
        for ret in cumulative_returns:
            if ret > peak:
                peak = ret
            drawdown = (peak - ret) / (1 + peak)
            max_drawdown = max(max_drawdown, drawdown)
        
        metrics = {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'num_trades': len(self.trades)
        }
        
        # Calculate alpha and beta if benchmark provided
        if benchmark_returns is not None:
            # Align returns with benchmark
            common_dates = daily_returns.index.intersection(benchmark_returns.index)
            if len(common_dates) > 10:
                aligned_returns = daily_returns.loc[common_dates]
                aligned_benchmark = benchmark_returns.loc[common_dates]
                
                # Calculate beta
                covariance = aligned_returns.cov(aligned_benchmark)
                benchmark_variance = aligned_benchmark.var()
                beta = covariance / benchmark_variance if benchmark_variance > 0 else 0
                
                # Calculate alpha
                risk_free_rate = 0.02 / 252  # Assuming 2% annual risk-free rate
                alpha = (annualized_return - risk_free_rate) - beta * (aligned_benchmark.mean() * 252 - risk_free_rate)
                
                metrics['alpha'] = alpha
                metrics['beta'] = beta
        
        return metrics


class AttentionAllocationTradingSystem:
    """
    Comprehensive trading system that implements attention-based strategies
    """
    
    def __init__(self, initial_capital=1000000):
        """
        Initialize the trading system
        
        Parameters:
        -----------
        initial_capital : float
            Initial portfolio capital
        """
        # Setup logging
        self.logger = self._setup_logger()
        self.logger.info("Initializing Attention Allocation Trading System")
        
        # Initialize strategies
        self.strategies = {}
        
        # Initialize portfolio manager
        self.portfolio_manager = PortfolioManager(initial_capital)
        
        # Store market data
        self.market_data = None
        self.stock_data = {}
        self.attention_index = None
        self.attention_deciles = None
        
        # Data simulator
        self.data_simulator = None
    
    def _setup_logger(self):
        """Set up logging configuration"""
        logger = logging.getLogger('AttentionTrader')
        logger.setLevel(logging.INFO)
        
        # Check if handler already exists
        if not logger.handlers:
            handler = logging.FileHandler('attention_trader.log')
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
            # Also log to console
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        
        return logger
    
    def initialize_system(self, start_date, end_date=None, n_stocks=50, n_industries=10):
        """
        Initialize the trading system with simulated data
        
        Parameters:
        -----------
        start_date : str
            Start date for simulation
        end_date : str
            End date for simulation
        n_stocks : int
            Number of stocks to simulate
        n_industries : int
            Number of industries to simulate
            
        Returns:
        --------
        bool: success flag
        """
        self.logger.info(f"Initializing trading system with simulated data from {start_date} to {end_date}")
        
        # Initialize data simulator
        self.data_simulator = DataSimulator(
            n_stocks=n_stocks, 
            n_industries=n_industries,
            start_date=start_date,
            end_date=end_date
        )
        
        # Generate market data
        self.market_data = self.data_simulator.simulate_market_data()
        
        # Generate stock data
        self.stock_data = self.data_simulator.simulate_stock_data()
        
        # Generate earnings announcements
        self.earnings_data = self.data_simulator.simulate_earnings_announcements()
        
        # Embed market anomalies
        self.stock_data = self.data_simulator.embed_anomalies()
        
        # Calculate attention index
        attention_calculator = AttentionIndexCalculator(
            self.market_data,
            self.stock_data,
            self.data_simulator.industry_mapping
        )
        self.attention_index, self.attention_deciles = attention_calculator.calculate_attention_index()
        
        self.logger.info(f"Attention index calculated for {len(self.attention_deciles)} days")
        
        # Initialize strategies
        self.strategies['momentum'] = MomentumStrategy(
            attention_data=self.attention_deciles,
            price_data=self.stock_data
        )
        
        self.strategies['pairs'] = PairsStrategy(
            attention_data=self.attention_deciles,
            price_data=self.stock_data,
            industry_mapping=self.data_simulator.industry_mapping
        )
        
        self.strategies['pead'] = PostEarningsAnnouncementDriftStrategy(
            attention_data=self.attention_deciles,
            price_data=self.stock_data,
            earnings_data=self.earnings_data
        )
        
        self.logger.info("Trading system initialized successfully with simulated data")
        
        return True
    
    def run_backtest(self, start_date=None, end_date=None):
        """
        Run a backtest of the trading system
        
        Parameters:
        -----------
        start_date : str
            Start date for backtest (defaults to data start date)
        end_date : str
            End date for backtest (defaults to data end date)
            
        Returns:
        --------
        dict: performance metrics
        """
        if self.data_simulator is None:
            self.logger.error("Trading system not initialized. Call initialize_system() first.")
            return {}
        
        if start_date is None:
            start_date = self.data_simulator.start_date
        else:
            start_date = pd.to_datetime(start_date)
            
        if end_date is None:
            end_date = self.data_simulator.end_date
        else:
            end_date = pd.to_datetime(end_date)
            
        self.logger.info(f"Running backtest from {start_date} to {end_date}")
        
        # Create trading calendar
        trading_days = pd.date_range(start=start_date, end=end_date, freq='B')
        
        # Run daily trading loop
        for day in tqdm(trading_days, desc="Backtesting"):
            self.logger.info(f"Processing trading day: {day.date()}")
            
            # Check if we have price data for this day
            valid_day = False
            for symbol, data in self.stock_data.items():
                if day in data.index:
                    valid_day = True
                    break
                    
            if not valid_day:
                self.logger.warning(f"No price data available for {day.date()}, skipping day")
                continue
                
            # Update pairs (once a month)
            if 'pairs' in self.strategies:
                self.strategies['pairs'].update_pairs(day)
            
            # Collect signals from all strategies
            all_signals = []
            
            for strategy_name, strategy in self.strategies.items():
                try:
                    if strategy_name == 'momentum':
                        buy_signals, sell_signals = strategy.generate_signals(day)
                        all_signals.extend(buy_signals)
                        all_signals.extend(sell_signals)
                    elif strategy_name == 'pairs':
                        signals = strategy.generate_signals(day)
                        all_signals.extend(signals)
                    elif strategy_name == 'pead':
                        signals = strategy.generate_signals(day)
                        all_signals.extend(signals)
                except Exception as e:
                    self.logger.error(f"Error generating signals for {strategy_name}: {e}")
            
            # Process signals through portfolio manager
            if all_signals:
                self.portfolio_manager.process_signals(all_signals, day, self.stock_data)
            else:
                # Still record portfolio value for days with no trades
                portfolio_value = self.portfolio_manager.get_portfolio_value(day, self.stock_data)
                self.portfolio_manager.daily_values.append({
                    'date': day,
                    'portfolio_value': portfolio_value
                })
        
        # Calculate performance
        benchmark_returns = self.market_data['Adj Close'].pct_change()
        performance = self.portfolio_manager.get_performance_metrics(benchmark_returns)
        
        self.logger.info(f"Backtest completed with {len(self.portfolio_manager.trades)} trades")
        self.logger.info(f"Total return: {performance.get('total_return', 0):.2%}")
        
        return performance
    
    def plot_performance(self):
        """Plot portfolio performance compared to benchmark"""
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        
        perf_data = self.portfolio_manager.get_historical_performance()
        
        if perf_data.empty:
            self.logger.warning("No performance data available to plot")
            return
            
        # Get benchmark data in same date range
        benchmark_data = None
        if self.market_data is not None:
            benchmark_data = self.market_data['Adj Close']
            benchmark_data = benchmark_data[benchmark_data.index.isin(perf_data.index)]
            benchmark_returns = benchmark_data.pct_change()
            benchmark_cum_returns = (1 + benchmark_returns).cumprod() - 1
        
        # Create figure
        plt.figure(figsize=(14, 10))
        
        # Plot portfolio performance
        plt.subplot(3, 1, 1)
        plt.plot(perf_data.index, perf_data['portfolio_value'], label='Portfolio Value')
        plt.title('Portfolio Value Over Time')
        plt.ylabel('Value ($)')
        plt.grid(True)
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        
        # Plot cumulative returns comparison
        plt.subplot(3, 1, 2)
        plt.plot(perf_data.index, perf_data['cumulative_return'], label='Attention Strategy')
        
        if benchmark_data is not None:
            plt.plot(benchmark_cum_returns.index, benchmark_cum_returns, label='Market Benchmark')
            
        plt.title('Cumulative Returns')
        plt.ylabel('Return (%)')
        plt.grid(True)
        plt.legend()
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        
        # Plot attention index
        if self.attention_index is not None:
            plt.subplot(3, 1, 3)
            plt.plot(self.attention_index.index, self.attention_index)
            plt.title('Attention Index Over Time')
            plt.ylabel('Attention Index')
            plt.grid(True)
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        
        plt.tight_layout()
        plt.savefig('attention_strategy_performance.png')
        plt.show()
    
    def analyze_trades(self):
        """
        Analyze trades by strategy type and attention level
        
        Returns:
        --------
        tuple: (trades_df, strategy_perf, attention_perf)
        """
        if not self.portfolio_manager.trades:
            self.logger.warning("No trades to analyze")
            return None, None, None
            
        trades_df = pd.DataFrame(self.portfolio_manager.trades)
        
        # Extract strategy from reason
        trades_df['strategy'] = trades_df['reason'].apply(
            lambda x: 'Momentum' if 'Winner' in x or 'Loser' in x 
                    else ('Pairs' if 'pairs' in x.lower() 
                        else ('PEAD' if 'earnings' in x.lower() else 'Other'))
        )
        
        # Add attention level at trade date
        trades_df['attention_level'] = np.nan
        for i, row in trades_df.iterrows():
            if row['date'] in self.attention_deciles.index:
                trades_df.loc[i, 'attention_level'] = self.attention_deciles.loc[row['date']]
        
        # Create attention bins
        trades_df['attention_bin'] = pd.cut(
            trades_df['attention_level'], 
            bins=[0, 3.5, 6.5, 10], 
            labels=['Low', 'Medium', 'High']
        )
        
        # Calculate trade PnL
        trades_df['pnl'] = 0.0
        
        for i, row in trades_df.iterrows():
            symbol = row['symbol']
            direction = row['direction']
            quantity = row['quantity']
            entry_price = row['price']
            
            # Find exit trade
            exit_trades = trades_df[
                (trades_df['symbol'] == symbol) & 
                (trades_df['direction'] != direction) & 
                (trades_df['date'] > row['date'])
            ]
            
            if not exit_trades.empty:
                # Get the earliest exit
                exit_trade = exit_trades.iloc[0]
                exit_price = exit_trade['price']
                
                # Calculate PnL
                if direction == 'buy':
                    pnl = (exit_price - entry_price) * quantity
                else:  # sell
                    pnl = (entry_price - exit_price) * quantity
                
                trades_df.loc[i, 'pnl'] = pnl
        
        # Group by strategy and attention
        strategy_perf = trades_df.groupby(['strategy', 'attention_bin']).agg({
            'quantity': 'sum',
            'value': 'sum',
            'pnl': 'sum'
        })
        
        # Group by attention level
        attention_perf = trades_df.groupby('attention_bin').agg({
            'quantity': 'sum',
            'value': 'sum',
            'pnl': 'sum'
        })
        
        # Print analysis
        print("\nTrade Analysis by Strategy and Attention Level:")
        print(strategy_perf)
        
        print("\nTrade Analysis by Attention Level:")
        print(attention_perf)
        
        # Plot results
        import matplotlib.pyplot as plt
        
        # Plot trade count by strategy and attention
        plt.figure(figsize=(12, 10))
        
        plt.subplot(2, 1, 1)
        trades_by_strat_attention = trades_df.groupby(['strategy', 'attention_bin']).size().unstack()
        trades_by_strat_attention.plot(kind='bar', ax=plt.gca())
        plt.title('Trade Count by Strategy and Attention Level')
        plt.ylabel('Number of Trades')
        plt.grid(True)
        
        # Plot PnL by strategy and attention
        plt.subplot(2, 1, 2)
        pnl_by_strat_attention = trades_df.groupby(['strategy', 'attention_bin'])['pnl'].sum().unstack()
        pnl_by_strat_attention.plot(kind='bar', ax=plt.gca())
        plt.title('Profit/Loss by Strategy and Attention Level')
        plt.ylabel('P&L ($)')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('trades_by_strategy_attention.png')
        plt.show()
        
        return trades_df, strategy_perf, attention_perf
    
    def plot_strategy_performance(self):
        """Plot performance by strategy type"""
        if not self.portfolio_manager.trades:
            self.logger.warning("No trades to analyze")
            return
            
        trades_df = pd.DataFrame(self.portfolio_manager.trades)
        trades_df['strategy'] = trades_df['reason'].apply(
            lambda x: 'Momentum' if 'Winner' in x or 'Loser' in x 
                    else ('Pairs' if 'pairs' in x.lower() 
                        else ('PEAD' if 'earnings' in x.lower() else 'Other'))
        )
        
        # Calculate cumulative PnL by strategy
        strategies = trades_df['strategy'].unique()
        
        # Create figure
        import matplotlib.pyplot as plt
        plt.figure(figsize=(14, 8))
        
        for strategy in strategies:
            strategy_trades = trades_df[trades_df['strategy'] == strategy]
            strategy_trades = strategy_trades.sort_values('date')
            
            # Calculate PnL
            buy_trades = strategy_trades[strategy_trades['direction'] == 'buy']
            sell_trades = strategy_trades[strategy_trades['direction'] == 'sell']
            
            # Calculate daily PnL
            daily_pnl = pd.Series(0, index=self.portfolio_manager.get_historical_performance().index)
            
            for _, row in strategy_trades.iterrows():
                date = row['date']
                if date in daily_pnl.index:
                    if row['direction'] == 'buy':
                        daily_pnl.loc[date] -= row['value']
                    else:  # sell
                        daily_pnl.loc[date] += row['value']
            
            # Calculate cumulative PnL
            cumulative_pnl = daily_pnl.cumsum()
            
            # Plot
            plt.plot(cumulative_pnl.index, cumulative_pnl, label=strategy)
        
        plt.title('Cumulative P&L by Strategy')
        plt.xlabel('Date')
        plt.ylabel('Cumulative P&L ($)')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig('strategy_performance.png')
        plt.show()


# Main function to run the simulation
def main():
    # Initialize the trading system
    trading_system = AttentionAllocationTradingSystem(initial_capital=1000000)
    
    # Initialize with simulated data
    start_date = '2018-01-01'
    end_date = '2022-12-31'
    
    success = trading_system.initialize_system(
        start_date=start_date,
        end_date=end_date,
        n_stocks=50,
        n_industries=10
    )
    
    if success:
        # Run backtest
        performance = trading_system.run_backtest(start_date, end_date)
        
        # Print performance metrics
        print("\nPerformance Metrics:")
        for metric, value in performance.items():
            if 'return' in metric or 'drawdown' in metric:
                print(f"  {metric}: {value:.2%}")
            else:
                print(f"  {metric}: {value:.4f}")
        
        # Plot performance
        trading_system.plot_performance()
        
        # Analyze trades
        trading_system.analyze_trades()
        
        # Plot strategy performance
        trading_system.plot_strategy_performance()
    else:
        print("Failed to initialize trading system")

if __name__ == "__main__":
    main()