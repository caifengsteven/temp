import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
from scipy import stats
from statsmodels.tsa.stattools import adfuller
import warnings
import math
import random
from itertools import combinations
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# Constants
LOOKBACK_PERIOD = 252  # One year of trading days
FORMATION_PERIOD = 252  # One year of trading days
TRADING_PERIOD = 126   # Six months of trading days
TOP_STOCKS = 20        # Number of top target stocks to trade
BOLLINGER_WINDOW = 20  # Rolling window for Bollinger bands
BOLLINGER_K = 1        # Number of standard deviations for Bollinger bands
TRANSACTION_COST = 0.0005  # 5 bps per half-turn

# Helper function to calculate n choose k
def calc_n_choose_k(n, k):
    """Calculate n choose k (binomial coefficient)"""
    return math.factorial(n) // (math.factorial(k) * math.factorial(n - k))

class VineCopulaStrategy:
    """
    Implementation of the Statistical Arbitrage Strategy using Vine Copulas
    """
    def __init__(self, start_date, end_date, universe='SP500', 
                 n_partner_stocks=3, selection_method='extremal'):
        """
        Initialize the strategy.
        
        Parameters:
        -----------
        start_date : str
            Start date in 'YYYY-MM-DD' format
        end_date : str
            End date in 'YYYY-MM-DD' format
        universe : str
            Stock universe to use ('SP500' by default)
        n_partner_stocks : int
            Number of partner stocks to find for each target stock
        selection_method : str
            Method to use for partner selection ('extremal', 'traditional', 
            'extended', or 'geometric')
        """
        self.start_date = start_date
        self.end_date = end_date
        self.universe = universe
        self.n_partner_stocks = n_partner_stocks
        self.selection_method = selection_method
        
        # Get universe stocks
        self.universe_stocks = self._get_sample_sp500_stocks()
        print(f"Using {len(self.universe_stocks)} stocks in the simulation")
        
        # Store study periods
        self.study_periods = self._create_study_periods()
        print(f"Created {len(self.study_periods)} study periods")
        
        # Initialize results storage
        self.results = {
            'returns': [],
            'positions': [],
            'trades': [],
            'mispricing_indices': {}
        }
        
        # Generate a full dataset for all stocks across the entire period once
        # This ensures date consistency across all periods
        full_start = self.start_date
        full_end = self.end_date
        self.full_price_data = self._get_full_price_data(self.universe_stocks, full_start, full_end)
        
    def _get_sample_sp500_stocks(self):
        """Return a representative sample of S&P 500 stocks"""
        sample_stocks = [
            'AAPL US Equity', 'MSFT US Equity', 'AMZN US Equity', 'GOOGL US Equity',
            'GOOG US Equity', 'META US Equity', 'TSLA US Equity', 'NVDA US Equity',
            'UNH US Equity', 'JNJ US Equity', 'JPM US Equity', 'V US Equity',
            'PG US Equity', 'XOM US Equity', 'HD US Equity', 'CVX US Equity',
            'MA US Equity', 'BAC US Equity', 'ABBV US Equity', 'PFE US Equity',
            'AVGO US Equity', 'COST US Equity', 'DIS US Equity', 'KO US Equity',
            'CSCO US Equity', 'PEP US Equity', 'WMT US Equity', 'TMO US Equity',
            'MRK US Equity', 'LLY US Equity', 'CMCSA US Equity', 'ABT US Equity',
            'ADBE US Equity', 'MCD US Equity', 'CRM US Equity', 'NKE US Equity',
            'ACN US Equity', 'DHR US Equity', 'NEE US Equity', 'VZ US Equity',
            'TXN US Equity', 'PM US Equity', 'INTC US Equity', 'AMD US Equity',
            'UPS US Equity', 'WFC US Equity', 'BMY US Equity', 'QCOM US Equity',
            'COP US Equity', 'RTX US Equity'
        ]
        return sample_stocks
        
    def _create_study_periods(self):
        """Create study periods for the strategy"""
        start = dt.datetime.strptime(self.start_date, '%Y-%m-%d')
        end = dt.datetime.strptime(self.end_date, '%Y-%m-%d')
        
        # Calculate required length for a complete period
        required_days = 365 * 2 + 182  # init + formation + trading
        
        # Check if we have enough time for at least one period
        if (end - start).days < required_days:
            print(f"WARNING: Date range too short for a complete study period. Need at least {required_days} days.")
            print(f"Current range is {(end - start).days} days from {start} to {end}")
            return []
        
        periods = []
        current_start = start
        
        print(f"Creating study periods from {start} to {end}")
        
        while current_start + dt.timedelta(days=required_days) <= end:
            init_start = current_start
            init_end = init_start + dt.timedelta(days=365)
            formation_start = init_end
            formation_end = formation_start + dt.timedelta(days=365)
            trading_start = formation_end
            trading_end = trading_start + dt.timedelta(days=182)  # Approximately 6 months
            
            period = {
                'init_start': init_start.strftime('%Y-%m-%d'),
                'init_end': init_end.strftime('%Y-%m-%d'),
                'formation_start': formation_start.strftime('%Y-%m-%d'),
                'formation_end': formation_end.strftime('%Y-%m-%d'),
                'trading_start': trading_start.strftime('%Y-%m-%d'),
                'trading_end': trading_end.strftime('%Y-%m-%d')
            }
            
            periods.append(period)
            print(f"  Created period {len(periods)}: {init_start.strftime('%Y-%m-%d')} to {trading_end.strftime('%Y-%m-%d')}")
            
            # Shift to the next period (monthly)
            current_start += dt.timedelta(days=30)
        
        return periods
    
    def _get_full_price_data(self, tickers, start_date, end_date):
        """
        Generate full price data for all tickers for the entire period once
        to ensure date consistency across periods
        """
        # Convert dates to datetime
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        
        # Generate a date range for the full period
        full_date_range = pd.date_range(start=start, end=end, freq='B')
        
        # Generate data for all tickers
        full_price_data = {}
        for ticker in tickers:
            full_price_data[ticker] = self._generate_simulated_price_data(ticker, full_date_range)
        
        return full_price_data
    
    def get_price_data(self, tickers, start_date, end_date):
        """
        Get price data for the given tickers and date range from the full dataset
        
        Parameters:
        -----------
        tickers : list
            List of ticker symbols
        start_date : str
            Start date in 'YYYY-MM-DD' format
        end_date : str
            End date in 'YYYY-MM-DD' format
            
        Returns:
        --------
        dict of pandas.DataFrame
            Dictionary mapping ticker symbols to DataFrames with price data
        """
        # Convert dates to datetime
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        
        # Extract data for the specified period
        price_data = {}
        for ticker in tickers:
            if ticker in self.full_price_data:
                full_data = self.full_price_data[ticker]
                # Extract data for the period
                period_data = full_data.loc[start:end].copy() if start in full_data.index else full_data.head(0)
                price_data[ticker] = period_data
            else:
                # If ticker not in full data, generate new data (shouldn't happen normally)
                print(f"Warning: {ticker} not in full price data, generating new data")
                date_range = pd.date_range(start=start, end=end, freq='B')
                price_data[ticker] = self._generate_simulated_price_data(ticker, date_range)
        
        return price_data
    
    def _generate_simulated_price_data(self, ticker, date_range):
        """
        Generate simulated price data for a ticker
        
        Parameters:
        -----------
        ticker : str
            Ticker symbol
        date_range : pandas.DatetimeIndex
            Date range for which to generate data
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with simulated price data
        """
        # Use ticker name to seed random number generator for consistency
        seed_value = hash(ticker) % 10000
        np.random.seed(seed_value)
        
        # Generate a base price (between $10 and $1000)
        base_price = 10 + (990 * (seed_value % 100) / 100)
        
        # Generate a price series with random walk
        n_days = len(date_range)
        daily_returns = np.random.normal(0.0002, 0.015, n_days)  # mean slightly positive, std 1.5%
        cum_returns = np.cumprod(1 + daily_returns)
        prices = base_price * cum_returns
        
        # Generate OHLC data
        open_prices = prices
        
        # Add some intraday volatility for high and low
        intraday_vol = np.random.uniform(0.005, 0.02, n_days)
        high_prices = open_prices * (1 + intraday_vol)
        low_prices = open_prices * (1 - intraday_vol * 0.8)  # Low tends to be closer to open than high
        
        # Close prices with some random adjustment from open
        close_prices = open_prices * (1 + np.random.normal(0, 0.005, n_days))
        
        # Ensure high >= open, close and low <= open, close
        for i in range(n_days):
            high_prices[i] = max(high_prices[i], open_prices[i], close_prices[i])
            low_prices[i] = min(low_prices[i], open_prices[i], close_prices[i])
        
        # Create DataFrame
        df = pd.DataFrame({
            'PX_OPEN': open_prices,
            'PX_HIGH': high_prices,
            'PX_LOW': low_prices,
            'PX_LAST': close_prices
        }, index=date_range)
        
        return df
    
    def calculate_returns(self, price_data):
        """
        Calculate daily returns from price data
        
        Parameters:
        -----------
        price_data : dict
            Dictionary mapping ticker symbols to DataFrames with price data
            
        Returns:
        --------
        dict of pandas.DataFrame
            Dictionary mapping ticker symbols to DataFrames with daily returns
        """
        returns_data = {}
        
        for ticker, prices in price_data.items():
            if len(prices) > 1:
                # Calculate simple returns based on close prices
                returns = prices['PX_LAST'].pct_change().dropna()
                returns_data[ticker] = returns
        
        return returns_data
    
    def select_partner_stocks(self, target_stock, returns_data, method='extremal', top_preselect=50):
        """
        Find the best partner stocks for a target stock
        
        Parameters:
        -----------
        target_stock : str
            Ticker symbol of the target stock
        returns_data : dict
            Dictionary mapping ticker symbols to daily returns
        method : str
            Method to use for partner selection
        top_preselect : int
            Number of top correlated stocks to pre-select
            
        Returns:
        --------
        list
            List of partner stock tickers
        """
        if target_stock not in returns_data:
            return []
        
        target_returns = returns_data[target_stock]
        
        # First, preselect the top correlated stocks to limit computational complexity
        correlations = {}
        for ticker, returns in returns_data.items():
            if ticker == target_stock or len(returns) != len(target_returns):
                continue
                
            # Calculate Spearman's rank correlation
            common_idx = target_returns.index.intersection(returns.index)
            if len(common_idx) < 30:  # Require at least 30 common days
                continue
                
            correlation = stats.spearmanr(
                target_returns.loc[common_idx].values,
                returns.loc[common_idx].values
            )[0]
            
            correlations[ticker] = correlation
        
        # Sort by correlation (descending) and get top_preselect
        top_correlated = sorted(correlations.items(), key=lambda x: -abs(x[1]))[:top_preselect]
        preselected_tickers = [t[0] for t in top_correlated]
        
        if len(preselected_tickers) < self.n_partner_stocks:
            return []  # Not enough candidates
        
        # For each selection method, calculate dependence measure
        if method == 'traditional':
            # Traditional: Sum of pairwise Spearman correlations
            return self._traditional_selection(target_stock, preselected_tickers, returns_data)
        elif method == 'extended':
            # Extended: Multivariate version of Spearman's rho
            return self._extended_selection(target_stock, preselected_tickers, returns_data)
        elif method == 'geometric':
            # Geometric: Diagonal measure (distance from hypercube diagonal)
            return self._geometric_selection(target_stock, preselected_tickers, returns_data)
        elif method == 'extremal':
            # Extremal: Chi-square test statistic for independence
            return self._extremal_selection(target_stock, preselected_tickers, returns_data)
        else:
            raise ValueError(f"Unknown selection method: {method}")
    
    def _traditional_selection(self, target_stock, candidates, returns_data):
        """Select partner stocks using pairwise Spearman correlations"""
        best_score = -float('inf')
        best_partners = []
        
        # Get target returns
        target_returns = returns_data[target_stock]
        
        # Limit the number of candidates to consider to prevent excessive computation
        max_candidates = min(20, len(candidates))
        candidate_subset = candidates[:max_candidates]
        
        # Calculate number of combinations to check
        num_combinations = calc_n_choose_k(len(candidate_subset), self.n_partner_stocks)
        
        # If too many combinations, take a random sample
        if num_combinations > 1000:
            random.seed(hash(target_stock))
            combinations_to_check = random.sample(list(combinations(candidate_subset, self.n_partner_stocks)), 1000)
        else:
            combinations_to_check = combinations(candidate_subset, self.n_partner_stocks)
        
        for partners in combinations_to_check:
            # Calculate sum of pairwise correlations
            correlation_sum = 0
            valid = True
            
            # Check target stock with each partner
            for partner in partners:
                partner_returns = returns_data[partner]
                common_idx = target_returns.index.intersection(partner_returns.index)
                
                if len(common_idx) < 30:
                    valid = False
                    break
                    
                correlation = stats.spearmanr(
                    target_returns.loc[common_idx].values,
                    partner_returns.loc[common_idx].values
                )[0]
                
                correlation_sum += abs(correlation)
            
            if not valid:
                continue
                
            # Check correlations between partners
            for i in range(len(partners)):
                for j in range(i+1, len(partners)):
                    returns_i = returns_data[partners[i]]
                    returns_j = returns_data[partners[j]]
                    common_idx = returns_i.index.intersection(returns_j.index)
                    
                    if len(common_idx) < 30:
                        valid = False
                        break
                        
                    correlation = stats.spearmanr(
                        returns_i.loc[common_idx].values,
                        returns_j.loc[common_idx].values
                    )[0]
                    
                    correlation_sum += abs(correlation)
                
                if not valid:
                    break
            
            if not valid:
                continue
            
            # Update best partners if score is higher
            if correlation_sum > best_score:
                best_score = correlation_sum
                best_partners = partners
        
        return list(best_partners)
    
    def _extended_selection(self, target_stock, candidates, returns_data):
        """
        Select partner stocks using multivariate Spearman's rho
        
        Note: This is a simplified version using average pairwise correlation
        """
        best_score = -float('inf')
        best_partners = []
        
        # Get target returns
        target_returns = returns_data[target_stock]
        
        # Limit the number of candidates to consider to prevent excessive computation
        max_candidates = min(20, len(candidates))
        candidate_subset = candidates[:max_candidates]
        
        # Calculate number of combinations to check
        num_combinations = calc_n_choose_k(len(candidate_subset), self.n_partner_stocks)
        
        # If too many combinations, take a random sample
        if num_combinations > 1000:
            random.seed(hash(target_stock))
            combinations_to_check = random.sample(list(combinations(candidate_subset, self.n_partner_stocks)), 1000)
        else:
            combinations_to_check = combinations(candidate_subset, self.n_partner_stocks)
        
        for partners in combinations_to_check:
            # Get returns for target and partners
            stocks = [target_stock] + list(partners)
            returns_list = [returns_data[stock] for stock in stocks]
            
            # Get common index
            common_idx = returns_list[0].index
            for returns in returns_list[1:]:
                common_idx = common_idx.intersection(returns.index)
            
            if len(common_idx) < 30:
                continue
            
            # Create returns matrix
            returns_matrix = np.zeros((len(common_idx), len(stocks)))
            for i, returns in enumerate(returns_list):
                returns_matrix[:, i] = returns.loc[common_idx].values
            
            # Calculate rank correlations
            ranks = np.zeros_like(returns_matrix)
            for j in range(returns_matrix.shape[1]):
                ranks[:, j] = stats.rankdata(returns_matrix[:, j])
            
            corr_matrix = np.corrcoef(ranks, rowvar=False)
            
            # Use mean of absolute correlations as a proxy for multivariate Spearman's rho
            # Exclude diagonal elements
            np.fill_diagonal(corr_matrix, 0)
            score = np.mean(np.abs(corr_matrix))
            
            # Update best partners if score is higher
            if score > best_score:
                best_score = score
                best_partners = partners
        
        return list(best_partners)
    
    def _geometric_selection(self, target_stock, candidates, returns_data):
        """
        Select partner stocks using the geometric approach (diagonal measure)
        """
        best_score = float('inf')  # Lower is better for diagonal measure
        best_partners = []
        
        # Get target returns
        target_returns = returns_data[target_stock]
        
        # Limit the number of candidates to consider to prevent excessive computation
        max_candidates = min(20, len(candidates))
        candidate_subset = candidates[:max_candidates]
        
        # Calculate number of combinations to check
        num_combinations = calc_n_choose_k(len(candidate_subset), self.n_partner_stocks)
        
        # If too many combinations, take a random sample
        if num_combinations > 1000:
            random.seed(hash(target_stock))
            combinations_to_check = random.sample(list(combinations(candidate_subset, self.n_partner_stocks)), 1000)
        else:
            combinations_to_check = combinations(candidate_subset, self.n_partner_stocks)
        
        for partners in combinations_to_check:
            # Get returns for target and partners
            stocks = [target_stock] + list(partners)
            returns_list = [returns_data[stock] for stock in stocks]
            
            # Get common index
            common_idx = returns_list[0].index
            for returns in returns_list[1:]:
                common_idx = common_idx.intersection(returns.index)
            
            if len(common_idx) < 30:
                continue
            
            # Create returns matrix
            returns_matrix = np.zeros((len(common_idx), len(stocks)))
            for i, returns in enumerate(returns_list):
                returns_matrix[:, i] = returns.loc[common_idx].values
            
            # Transform to ranks and normalize to [0,1]
            ranks = np.zeros_like(returns_matrix)
            for j in range(returns_matrix.shape[1]):
                ranks[:, j] = stats.rankdata(returns_matrix[:, j]) / len(common_idx)
            
            # Calculate diagonal measure (distance from diagonal)
            # In 4D space, diagonal goes from (0,0,0,0) to (1,1,1,1)
            # For each point, measure Euclidean distance to this diagonal
            diagonal_dir = np.ones(len(stocks)) / np.sqrt(len(stocks))  # Unit vector along diagonal
            diagonal_dists = []
            
            for point in ranks:
                # Project point onto diagonal
                proj = np.dot(point, diagonal_dir) * diagonal_dir
                # Calculate distance from point to projection
                dist = np.linalg.norm(point - proj)
                diagonal_dists.append(dist)
            
            # Score is sum of distances (lower is better = closer to diagonal)
            score = np.sum(diagonal_dists)
            
            # Update best partners if score is lower
            if score < best_score:
                best_score = score
                best_partners = partners
        
        return list(best_partners)
    
    def _extremal_selection(self, target_stock, candidates, returns_data):
        """
        Select partner stocks using the extremal approach
        (measuring deviation from independence focusing on extreme events)
        
        This is a simplified version using chi-squared test for independence
        """
        best_score = -float('inf')  # Higher is better for extremal measure
        best_partners = []
        
        # Get target returns
        target_returns = returns_data[target_stock]
        
        # Limit the number of candidates to consider to prevent excessive computation
        max_candidates = min(20, len(candidates))
        candidate_subset = candidates[:max_candidates]
        
        # Calculate number of combinations to check
        num_combinations = calc_n_choose_k(len(candidate_subset), self.n_partner_stocks)
        
        # If too many combinations, take a random sample
        if num_combinations > 1000:
            random.seed(hash(target_stock))
            combinations_to_check = random.sample(list(combinations(candidate_subset, self.n_partner_stocks)), 1000)
        else:
            combinations_to_check = combinations(candidate_subset, self.n_partner_stocks)
        
        for partners in combinations_to_check:
            # Get returns for target and partners
            stocks = [target_stock] + list(partners)
            returns_list = [returns_data[stock] for stock in stocks]
            
            # Get common index
            common_idx = returns_list[0].index
            for returns in returns_list[1:]:
                common_idx = common_idx.intersection(returns.index)
            
            if len(common_idx) < 30:
                continue
            
            # Create returns matrix
            returns_matrix = np.zeros((len(common_idx), len(stocks)))
            for i, returns in enumerate(returns_list):
                returns_matrix[:, i] = returns.loc[common_idx].values
            
            # Transform to ranks
            ranks = np.zeros_like(returns_matrix)
            for j in range(returns_matrix.shape[1]):
                ranks[:, j] = stats.rankdata(returns_matrix[:, j])
            
            # Bin the data for chi-square test (using quartiles)
            bins = 4  # Use quartiles
            binned_data = np.zeros_like(ranks, dtype=int)
            for j in range(ranks.shape[1]):
                # Use np.digitize to bin the data
                bins_array = np.linspace(ranks[:, j].min(), ranks[:, j].max(), bins+1)
                binned_data[:, j] = np.digitize(ranks[:, j], bins_array[1:-1])
            
            # Compute chi-square test statistic for multivariate independence
            # Higher values indicate more dependence
            score = 0
            
            # We'll look at all pairs of variables
            for i in range(len(stocks)):
                for j in range(i+1, len(stocks)):
                    # Create contingency table
                    try:
                        # Count occurrences of each combination
                        combinations_count = {}
                        for row in range(len(binned_data)):
                            bin_i = binned_data[row, i]
                            bin_j = binned_data[row, j]
                            key = (bin_i, bin_j)
                            if key in combinations_count:
                                combinations_count[key] += 1
                            else:
                                combinations_count[key] = 1
                        
                        # Create expected frequencies (assuming independence)
                        # First get marginal counts
                        marginal_i = {}
                        marginal_j = {}
                        
                        for key, count in combinations_count.items():
                            bin_i, bin_j = key
                            if bin_i in marginal_i:
                                marginal_i[bin_i] += count
                            else:
                                marginal_i[bin_i] = count
                                
                            if bin_j in marginal_j:
                                marginal_j[bin_j] += count
                            else:
                                marginal_j[bin_j] = count
                        
                        # Calculate chi-square statistic
                        chi2 = 0
                        n = len(binned_data)
                        
                        for key, observed in combinations_count.items():
                            bin_i, bin_j = key
                            expected = marginal_i[bin_i] * marginal_j[bin_j] / n
                            if expected > 0:  # Avoid division by zero
                                chi2 += (observed - expected) ** 2 / expected
                        
                        score += chi2
                    except Exception as e:
                        # Skip if contingency table is problematic
                        continue
            
            # Update best partners if score is higher
            if score > best_score:
                best_score = score
                best_partners = partners
        
        return list(best_partners)
    
    def fit_model(self, target_stock, partner_stocks, returns_data, model_type='vine'):
        """
        Fit a dependence model for the quadruple (target stock + partner stocks)
        
        Parameters:
        -----------
        target_stock : str
            Ticker symbol of the target stock
        partner_stocks : list
            List of partner stock tickers
        returns_data : dict
            Dictionary mapping ticker symbols to daily returns
        model_type : str
            Type of model to fit ('vine', 'gaussian', 'student-t')
            
        Returns:
        --------
        object
            Fitted model
        """
        # Get returns for target and partners
        stocks = [target_stock] + list(partner_stocks)
        returns_list = [returns_data[stock] for stock in stocks]
        
        # Get common index
        common_idx = returns_list[0].index
        for returns in returns_list[1:]:
            common_idx = common_idx.intersection(returns.index)
        
        if len(common_idx) < 30:
            raise ValueError(f"Not enough common data points for {target_stock} and partners")
        
        # Create returns matrix
        returns_matrix = np.zeros((len(common_idx), len(stocks)))
        for i, returns in enumerate(returns_list):
            returns_matrix[:, i] = returns.loc[common_idx].values
        
        # Transform to ranks and normalize to [0,1]
        u_data = np.zeros_like(returns_matrix)
        for j in range(returns_matrix.shape[1]):
            ranks = stats.rankdata(returns_matrix[:, j])
            u_data[:, j] = (ranks - 0.5) / len(ranks)
        
        # Clip values to avoid numerical issues at the boundaries
        u_data = np.clip(u_data, 0.001, 0.999)
        
        if model_type == 'gaussian':
            # Fit a multivariate Gaussian copula
            model = self._fit_gaussian_model(u_data)
        elif model_type == 'student-t':
            # Fit a multivariate Student's t copula
            model = self._fit_student_t_model(u_data)
        elif model_type == 'vine':
            # Since vine packages are not available, use gaussian or t model based on data properties
            # In a real vine copula implementation, this would be replaced with proper vine fitting
            
            # Check for heavy tails
            has_heavy_tails = False
            for j in range(returns_matrix.shape[1]):
                kurtosis = stats.kurtosis(returns_matrix[:, j])
                if kurtosis > 1.0:  # Normal has kurtosis 0, values > 1 indicate heavier tails
                    has_heavy_tails = True
                    break
            
            if has_heavy_tails:
                # Use t-copula as approximation for vine with heavy tails
                model = self._fit_student_t_model(u_data)
            else:
                # Use Gaussian otherwise
                model = self._fit_gaussian_model(u_data)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        return model, common_idx
    
    def _fit_gaussian_model(self, u_data):
        """Fit a multivariate Gaussian model"""
        # Transform from [0,1] to standard normal
        z_data = stats.norm.ppf(u_data)
        
        # Handle edge cases (transform nans and infs)
        z_data = np.nan_to_num(z_data, nan=0, posinf=5, neginf=-5)
        
        # Calculate mean and covariance
        mean = np.zeros(z_data.shape[1])  # Assume standardized data
        cov = np.cov(z_data, rowvar=False)
        
        # Ensure covariance matrix is positive definite
        min_eig = np.min(np.linalg.eigvals(cov))
        if min_eig < 0:
            cov -= 1.1 * min_eig * np.eye(len(cov))
        
        return {'mean': mean, 'cov': cov, 'type': 'gaussian'}
    
    def _fit_student_t_model(self, u_data):
        """Fit a multivariate Student's t model"""
        # Transform from [0,1] to standard normal
        z_data = stats.norm.ppf(u_data)
        
        # Handle edge cases (transform nans and infs)
        z_data = np.nan_to_num(z_data, nan=0, posinf=5, neginf=-5)
        
        # Calculate mean and covariance
        mean = np.zeros(z_data.shape[1])  # Assume standardized data
        cov = np.cov(z_data, rowvar=False)
        
        # Ensure covariance matrix is positive definite
        min_eig = np.min(np.linalg.eigvals(cov))
        if min_eig < 0:
            cov -= 1.1 * min_eig * np.eye(len(cov))
        
        # Estimate degrees of freedom based on average kurtosis
        kurtosis_values = [stats.kurtosis(z_data[:, i]) for i in range(z_data.shape[1])]
        avg_kurtosis = np.mean(kurtosis_values)
        
        # Map kurtosis to df (higher kurtosis -> lower df)
        if avg_kurtosis > 3:
            df = 3.0
        elif avg_kurtosis > 1:
            df = 5.0
        else:
            df = 10.0
        
        return {'mean': mean, 'cov': cov, 'df': df, 'type': 'student-t'}
    
    def calculate_mispricing_index(self, model, target_stock, partner_stocks, returns_data, 
                                  formation_data_idx, model_type='vine'):
        """
        Calculate the mispricing index based on the model
        
        Parameters:
        -----------
        model : object
            Fitted model
        target_stock : str
            Ticker symbol of the target stock
        partner_stocks : list
            List of partner stock tickers
        returns_data : dict
            Dictionary mapping ticker symbols to daily returns
        formation_data_idx : pandas.DatetimeIndex
            Index used for model fitting (for rank transformation)
        model_type : str
            Type of model used ('vine', 'gaussian', 'student-t')
            
        Returns:
        --------
        pandas.Series
            Series with mispricing index values
        """
        # Get returns for target and partners
        stocks = [target_stock] + list(partner_stocks)
        
        # Get formation period returns for all stocks
        formation_returns = {}
        for stock in stocks:
            if stock in returns_data:
                formation_returns[stock] = returns_data[stock].loc[
                    returns_data[stock].index.intersection(formation_data_idx)
                ]
        
        # Verify we have formation data for all stocks
        for stock in stocks:
            if stock not in formation_returns or len(formation_returns[stock]) < 10:
                raise ValueError(f"Not enough formation data for {stock}")
            
        # Get trading period returns for all stocks (all returns data)
        trading_returns = {}
        for stock in stocks:
            if stock in returns_data:
                trading_returns[stock] = returns_data[stock]
                
        # Get common trading dates
        common_trading_idx = trading_returns[stocks[0]].index
        for stock in stocks[1:]:
            common_trading_idx = common_trading_idx.intersection(trading_returns[stock].index)
        
        # Calculate daily mispricings
        daily_mispricings = pd.Series(index=common_trading_idx, dtype=float)
        
        for date in common_trading_idx:
            try:
                # Get rank of target stock compared to its formation distribution
                target_val = trading_returns[target_stock].loc[date]
                formation_vals = formation_returns[target_stock].values
                rank = np.sum(formation_vals <= target_val) / len(formation_vals)
                rank = np.clip(rank, 0.001, 0.999)  # Clip to avoid boundary issues
                
                # Get ranks of partner stocks compared to their formation distributions
                partner_ranks = []
                for partner in partner_stocks:
                    partner_val = trading_returns[partner].loc[date]
                    partner_formation_vals = formation_returns[partner].values
                    partner_rank = np.sum(partner_formation_vals <= partner_val) / len(partner_formation_vals)
                    partner_rank = np.clip(partner_rank, 0.001, 0.999)
                    partner_ranks.append(partner_rank)
                
                # Calculate conditional probability
                if model_type == 'gaussian' or (model_type == 'vine' and model['type'] == 'gaussian'):
                    conditional_prob = self._gaussian_conditional_prob(model, rank, partner_ranks)
                elif model_type == 'student-t' or (model_type == 'vine' and model['type'] == 'student-t'):
                    conditional_prob = self._student_t_conditional_prob(model, rank, partner_ranks)
                else:
                    raise ValueError(f"Unknown model type: {model_type}")
                
                # Calculate daily mispricing
                daily_mispricings.loc[date] = conditional_prob - 0.5
            except Exception as e:
                print(f"Error calculating mispricing for {target_stock} on {date}: {e}")
                daily_mispricings.loc[date] = 0.0  # Default to 0 on error
        
        # Accumulate daily mispricings to create mispricing index
        mispricing_index = daily_mispricings.cumsum()
        
        return mispricing_index
    
    def _gaussian_conditional_prob(self, model, u1, partner_ranks):
        """Calculate conditional probability under Gaussian copula"""
        # Transform from [0,1] to standard normal
        z1 = stats.norm.ppf(u1)
        z_partners = [stats.norm.ppf(u) for u in partner_ranks]
        
        # Extract parameters
        mean = model['mean']
        cov = model['cov']
        
        # Extract relevant parts of mean and covariance
        mu1 = mean[0]
        mu2 = mean[1:]
        sigma11 = cov[0, 0]
        sigma12 = cov[0, 1:]
        sigma21 = cov[1:, 0]
        sigma22 = cov[1:, 1:]
        
        # Calculate conditional mean and variance
        try:
            # This may fail if sigma22 is not invertible
            sigma22_inv = np.linalg.inv(sigma22)
            cond_mean = mu1 + sigma12.dot(sigma22_inv).dot(np.array(z_partners) - mu2)
            cond_var = sigma11 - sigma12.dot(sigma22_inv).dot(sigma21)
            
            # Ensure variance is positive
            cond_var = max(cond_var, 1e-6)
            
            # Calculate conditional probability
            cond_prob = stats.norm.cdf(z1, loc=cond_mean, scale=np.sqrt(cond_var))
            
            # Handle edge cases
            cond_prob = max(min(cond_prob, 0.999), 0.001)
        except:
            # Fallback if calculation fails
            cond_prob = u1
        
        return cond_prob
    
    def _student_t_conditional_prob(self, model, u1, partner_ranks):
        """Calculate conditional probability under Student's t copula"""
        # This is a simplification - would need a more sophisticated approach for exact calculation
        
        # Transform from [0,1] to standard normal
        z1 = stats.norm.ppf(u1)
        z_partners = [stats.norm.ppf(u) for u in partner_ranks]
        
        # Extract parameters
        mean = model['mean']
        cov = model['cov']
        df = model['df']
        
        # Extract relevant parts of mean and covariance
        mu1 = mean[0]
        mu2 = mean[1:]
        sigma11 = cov[0, 0]
        sigma12 = cov[0, 1:]
        sigma21 = cov[1:, 0]
        sigma22 = cov[1:, 1:]
        
        # Calculate conditional mean and variance
        try:
            # This may fail if sigma22 is not invertible
            sigma22_inv = np.linalg.inv(sigma22)
            cond_mean = mu1 + sigma12.dot(sigma22_inv).dot(np.array(z_partners) - mu2)
            
            # Calculate Mahalanobis distance for scaling factor
            maha = np.dot(np.array(z_partners) - mu2, sigma22_inv.dot(np.array(z_partners) - mu2))
            
            # Adjust conditional variance based on t distribution properties
            # This is an approximation
            cond_var = (sigma11 - sigma12.dot(sigma22_inv).dot(sigma21)) * (df + maha) / (df + len(partner_ranks))
            
            # Ensure variance is positive
            cond_var = max(cond_var, 1e-6)
            
            # Calculate conditional probability using t distribution with adjusted df
            adj_df = df + len(partner_ranks)
            
            # Standardize z1 for the conditional t distribution
            std_z1 = (z1 - cond_mean) / np.sqrt(cond_var)
            
            # Calculate probability
            cond_prob = stats.t.cdf(std_z1, df=adj_df)
            
            # Handle edge cases
            cond_prob = max(min(cond_prob, 0.999), 0.001)
        except:
            # Fallback if calculation fails
            cond_prob = u1
        
        return cond_prob
    
    def select_top_stocks(self, mispricing_indices, n=20):
        """
        Select top n stocks based on ADF test statistics of mispricing indices
        
        Parameters:
        -----------
        mispricing_indices : dict
            Dictionary mapping ticker symbols to mispricing indices
        n : int
            Number of top stocks to select
            
        Returns:
        --------
        list
            List of top n stock tickers
        """
        adf_stats = {}
        
        for ticker, index in mispricing_indices.items():
            if len(index) < 30:  # Require at least 30 observations
                continue
            
            # Calculate ADF test statistic
            try:
                result = adfuller(index.values, regression='c', autolag='AIC')
                adf_stats[ticker] = result[0]  # Get test statistic
            except:
                continue
        
        # Sort by test statistic (ascending)
        sorted_stocks = sorted(adf_stats.items(), key=lambda x: x[1])
        
        # Return top n or all if fewer
        return [s[0] for s in sorted_stocks[:min(n, len(sorted_stocks))]]
    
    def generate_signals(self, mispricing_index, window=20, k=1):
        """
        Generate trading signals based on Bollinger bands
        
        Parameters:
        -----------
        mispricing_index : pandas.Series
            Mispricing index
        window : int
            Window size for calculating Bollinger bands
        k : float
            Number of standard deviations for Bollinger bands
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with columns 'signal' (-1, 0, 1) and 'exit' (True/False)
        """
        # Calculate rolling mean and standard deviation
        rolling_mean = mispricing_index.rolling(window=window).mean()
        rolling_std = mispricing_index.rolling(window=window).std()
        
        # Calculate Bollinger bands
        upper_band = rolling_mean + k * rolling_std
        lower_band = rolling_mean - k * rolling_std
        
        # Initialize signal DataFrame
        signals = pd.DataFrame(index=mispricing_index.index)
        signals['signal'] = 0  # 0: no position, 1: long, -1: short
        signals['exit'] = False
        
        # Generate signals
        for i in range(window, len(mispricing_index)):
            date = mispricing_index.index[i]
            value = mispricing_index.iloc[i]
            mean = rolling_mean.iloc[i]
            
            # Check for entry signals
            if value < lower_band.iloc[i]:
                # Cross below lower band - go long
                signals.loc[date, 'signal'] = 1
            elif value > upper_band.iloc[i]:
                # Cross above upper band - go short
                signals.loc[date, 'signal'] = -1
            
            # Check for exit signals
            if i > 0:
                prev_date = mispricing_index.index[i-1]
                prev_value = mispricing_index.iloc[i-1]
                
                # Exit when crossing mean
                if (prev_value < mean and value >= mean) or (prev_value > mean and value <= mean):
                    signals.loc[date, 'exit'] = True
        
        return signals
    
    def backtest_strategy(self, target_stock, mispricing_index, signals, price_data):
        """
        Backtest the strategy for a single stock
        
        Parameters:
        -----------
        target_stock : str
            Ticker symbol of the target stock
        mispricing_index : pandas.Series
            Mispricing index
        signals : pandas.DataFrame
            Trading signals
        price_data : dict
            Dictionary mapping ticker symbols to DataFrames with price data
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with daily returns of the strategy
        """
        # Get prices for the target stock
        prices = price_data[target_stock]['PX_LAST']
        
        # Initialize positions and trades
        positions = pd.Series(index=signals.index, data=0)
        trades = []
        current_position = 0
        entry_price = 0
        entry_date = None
        
        # Implement trading logic
        for date in signals.index:
            if date not in prices.index:
                # Skip if price is not available
                continue
            
            signal = signals.loc[date, 'signal']
            exit_signal = signals.loc[date, 'exit']
            price = prices.loc[date]
            
            # Check for exit
            if current_position != 0 and exit_signal:
                # Record the trade
                exit_price = price
                pnl = (exit_price / entry_price - 1) * current_position
                trades.append({
                    'entry_date': entry_date,
                    'exit_date': date,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'position': current_position,
                    'pnl': pnl
                })
                
                # Reset position
                current_position = 0
                entry_price = 0
                entry_date = None
            
            # Check for entry
            if current_position == 0 and signal != 0:
                # Enter new position
                current_position = signal
                entry_price = price
                entry_date = date
            
            # Record current position
            positions.loc[date] = current_position
        
        # Close any remaining position at the end
        if current_position != 0 and entry_date is not None:
            last_date = positions.index[-1]
            if last_date in prices.index:
                exit_price = prices.loc[last_date]
                pnl = (exit_price / entry_price - 1) * current_position
                trades.append({
                    'entry_date': entry_date,
                    'exit_date': last_date,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'position': current_position,
                    'pnl': pnl
                })
        
        # Calculate daily returns
        daily_returns = pd.Series(index=positions.index, data=0.0)
        
        for i in range(1, len(positions)):
            if positions.iloc[i-1] != 0:
                if positions.index[i] in prices.index and positions.index[i-1] in prices.index:
                    price_change = prices.loc[positions.index[i]] / prices.loc[positions.index[i-1]] - 1
                    daily_returns.iloc[i] = price_change * positions.iloc[i-1]
        
        # Apply transaction costs
        for trade in trades:
            entry_date = trade['entry_date']
            exit_date = trade['exit_date']
            
            # Apply costs at entry
            if entry_date in daily_returns.index:
                daily_returns.loc[entry_date] -= TRANSACTION_COST
            
            # Apply costs at exit
            if exit_date in daily_returns.index:
                daily_returns.loc[exit_date] -= TRANSACTION_COST
        
        return daily_returns, positions, trades
    
    def run_backtest(self, period, model_type='vine'):
        """
        Run backtest for a single study period
        
        Parameters:
        -----------
        period : dict
            Dictionary with study period dates
        model_type : str
            Type of model to use ('vine', 'gaussian', 'student-t')
            
        Returns:
        --------
        dict
            Dictionary with backtest results
        """
        print(f"Running backtest for period {period['init_start']} to {period['trading_end']}")
        
        try:
            # Get price data for initialization period
            init_start = period['init_start']
            init_end = period['init_end']
            
            # Use fewer stocks for testing
            universe_stocks = self.universe_stocks[:20]  # Limit to 20 for speed
            print(f"Using {len(universe_stocks)} stocks for this backtest")
            
            price_data = self.get_price_data(universe_stocks, init_start, init_end)
            
            # Calculate returns
            returns_data = self.calculate_returns(price_data)
            
            # Find partner stocks for each target stock
            partner_selections = {}
            for target_stock in universe_stocks:
                try:
                    partners = self.select_partner_stocks(
                        target_stock, returns_data, method=self.selection_method
                    )
                    if len(partners) == self.n_partner_stocks:
                        partner_selections[target_stock] = partners
                except Exception as e:
                    print(f"Error selecting partners for {target_stock}: {e}")
            
            if not partner_selections:
                print("No valid partner selections found")
                return None
                
            print(f"Found {len(partner_selections)} valid target stocks with partners")
            
            # Fit models for each target stock
            fitted_models = {}
            for target_stock, partners in partner_selections.items():
                try:
                    model, common_idx = self.fit_model(
                        target_stock, partners, returns_data, model_type=model_type
                    )
                    fitted_models[target_stock] = {
                        'model': model,
                        'partners': partners,
                        'common_idx': common_idx
                    }
                except Exception as e:
                    print(f"Error fitting model for {target_stock}: {e}")
            
            if not fitted_models:
                print("No valid models fitted")
                return None
                
            print(f"Fitted {len(fitted_models)} models")
            
            # Get price data for formation period
            formation_start = period['formation_start']
            formation_end = period['formation_end']
            
            formation_stocks = list(fitted_models.keys()) + [
                partner for model_info in fitted_models.values() 
                for partner in model_info['partners']
            ]
            formation_stocks = list(set(formation_stocks))  # Remove duplicates
            
            formation_prices = self.get_price_data(formation_stocks, formation_start, formation_end)
            formation_returns = self.calculate_returns(formation_prices)
            
            # Calculate mispricing indices for formation period
            formation_indices = {}
            for target_stock, model_info in fitted_models.items():
                try:
                    formation_index = self.calculate_mispricing_index(
                        model_info['model'], target_stock, model_info['partners'],
                        formation_returns, model_info['common_idx'], model_type=model_type
                    )
                    formation_indices[target_stock] = formation_index
                except Exception as e:
                    print(f"Error calculating formation index for {target_stock}: {e}")
            
            if not formation_indices:
                print("No valid formation indices calculated")
                return None
                
            print(f"Calculated {len(formation_indices)} formation indices")
            
            # Select top target stocks
            top_stocks = self.select_top_stocks(formation_indices, n=TOP_STOCKS)
            
            if not top_stocks:
                print("No top stocks selected")
                return None
                
            print(f"Selected {len(top_stocks)} top stocks")
            
            # Get price data for trading period
            trading_start = period['trading_start']
            trading_end = period['trading_end']
            
            trading_stocks = top_stocks + [
                partner for target in top_stocks if target in fitted_models
                for partner in fitted_models[target]['partners']
            ]
            trading_stocks = list(set(trading_stocks))  # Remove duplicates
            
            trading_prices = self.get_price_data(trading_stocks, trading_start, trading_end)
            trading_returns = self.calculate_returns(trading_prices)
            
            # Generate signals and backtest for each top stock
            portfolio_returns = None
            portfolio_positions = {}
            portfolio_trades = {}
            
            for target_stock in top_stocks:
                if target_stock not in fitted_models:
                    continue
                    
                model_info = fitted_models[target_stock]
                
                try:
                    # Calculate trading period mispricing index
                    trading_index = self.calculate_mispricing_index(
                        model_info['model'], target_stock, model_info['partners'],
                        trading_returns, model_info['common_idx'], model_type=model_type
                    )
                    
                    # Generate signals
                    signals = self.generate_signals(
                        trading_index, window=BOLLINGER_WINDOW, k=BOLLINGER_K
                    )
                    
                    # Backtest
                    stock_returns, positions, trades = self.backtest_strategy(
                        target_stock, trading_index, signals, trading_prices
                    )
                    
                    # Store results
                    if portfolio_returns is None:
                        portfolio_returns = stock_returns
                    else:
                        # Align indexes
                        combined = pd.concat([portfolio_returns, stock_returns], axis=1)
                        combined = combined.fillna(0)
                        portfolio_returns = combined.mean(axis=1)
                    
                    portfolio_positions[target_stock] = positions
                    portfolio_trades[target_stock] = trades
                    
                    # Store mispricing index
                    key = f"{target_stock}_{trading_start}_{trading_end}"
                    self.results['mispricing_indices'][key] = trading_index
                    
                except Exception as e:
                    print(f"Error backtesting {target_stock}: {e}")
            
            if portfolio_returns is None or len(portfolio_returns) == 0:
                print("No portfolio returns generated")
                return None
                
            print(f"Generated {len(portfolio_positions)} stock backtests")
            
            # Store results
            self.results['returns'].append(portfolio_returns)
            self.results['positions'].append(portfolio_positions)
            self.results['trades'].append(portfolio_trades)
            
            return {
                'returns': portfolio_returns,
                'positions': portfolio_positions,
                'trades': portfolio_trades
            }
        except Exception as e:
            print(f"Error in run_backtest: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def run_strategy(self, model_type='vine'):
        """
        Run the full strategy across all study periods
        
        Parameters:
        -----------
        model_type : str
            Type of model to use ('vine', 'gaussian', 'student-t')
            
        Returns:
        --------
        dict
            Dictionary with strategy results
        """
        # Run backtest for each study period
        for period in self.study_periods:
            result = self.run_backtest(period, model_type=model_type)
            if result is None:
                print(f"Skipping period {period['init_start']} to {period['trading_end']}")
        
        # Check if we have any results
        if not self.results['returns']:
            print("No successful backtests were completed")
            return {
                'daily_returns': pd.Series(),
                'performance': {},
                'positions': [],
                'trades': [],
                'mispricing_indices': {}
            }
        
        # Combine results from all periods
        all_returns = pd.concat(self.results['returns'])
        
        # If multiple periods overlap, average the returns
        daily_returns = all_returns.groupby(level=0).mean()
        
        # Calculate performance metrics
        performance = self.calculate_performance(daily_returns)
        
        return {
            'daily_returns': daily_returns,
            'performance': performance,
            'positions': self.results['positions'],
            'trades': self.results['trades'],
            'mispricing_indices': self.results['mispricing_indices']
        }
    
    def calculate_performance(self, returns):
        """
        Calculate performance metrics for a return series
        
        Parameters:
        -----------
        returns : pandas.Series
            Series with daily returns
            
        Returns:
        --------
        dict
            Dictionary with performance metrics
        """
        if len(returns) == 0:
            return {
                'total_return': 0,
                'annualized_return': 0,
                'annualized_vol': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'monthly_mean': 0,
                'monthly_std': 0,
                'monthly_pos': 0,
                'monthly_neg': 0,
                'var_95': 0,
                'var_99': 0
            }
            
        # Convert to monthly returns
        monthly_returns = returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
        
        # Basic metrics
        total_return = (1 + returns).prod() - 1
        annualized_return = (1 + total_return) ** (252 / len(returns)) - 1 if len(returns) > 0 else 0
        annualized_vol = returns.std() * np.sqrt(252) if len(returns) > 0 else 0
        sharpe_ratio = annualized_return / annualized_vol if annualized_vol > 0 else 0
        
        # Drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative / running_max - 1)
        max_drawdown = drawdown.min() if len(drawdown) > 0 else 0
        
        # Monthly statistics
        monthly_mean = monthly_returns.mean() if len(monthly_returns) > 0 else 0
        monthly_std = monthly_returns.std() if len(monthly_returns) > 0 else 0
        monthly_pos = (monthly_returns > 0).mean() if len(monthly_returns) > 0 else 0
        monthly_neg = (monthly_returns < 0).mean() if len(monthly_returns) > 0 else 0
        
        # Value at Risk
        var_95 = np.percentile(returns, 5) if len(returns) > 0 else 0
        var_99 = np.percentile(returns, 1) if len(returns) > 0 else 0
        
        # Create performance dictionary
        performance = {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'annualized_vol': annualized_vol,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'monthly_mean': monthly_mean,
            'monthly_std': monthly_std,
            'monthly_pos': monthly_pos,
            'monthly_neg': monthly_neg,
            'var_95': var_95,
            'var_99': var_99
        }
        
        return performance
    
    def plot_results(self, results):
        """
        Plot strategy results
        
        Parameters:
        -----------
        results : dict
            Dictionary with strategy results
        """
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Get data
        returns = results['daily_returns']
        performance = results['performance']
        
        if len(returns) == 0:
            print("No returns data to plot")
            return
            
        # Plot cumulative returns
        cumulative = (1 + returns).cumprod()
        axes[0, 0].plot(cumulative, linewidth=2)
        axes[0, 0].set_title('Cumulative Returns')
        axes[0, 0].set_ylabel('Cumulative Return')
        axes[0, 0].grid(True)
        
        # Plot drawdown
        running_max = cumulative.cummax()
        drawdown = (cumulative / running_max - 1)
        axes[0, 1].fill_between(drawdown.index, drawdown.values, 0, color='red', alpha=0.3)
        axes[0, 1].set_title('Drawdown')
        axes[0, 1].set_ylabel('Drawdown')
        axes[0, 1].grid(True)
        
        # Plot monthly returns
        monthly_returns = returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
        monthly_returns.plot(kind='bar', ax=axes[1, 0])
        axes[1, 0].set_title('Monthly Returns')
        axes[1, 0].set_ylabel('Return')
        axes[1, 0].grid(True)
        
        # Plot return distribution using distplot (works with older seaborn versions)
        try:
            # Try using histplot (newer seaborn)
            import seaborn as sns
            sns.histplot(returns, kde=True, ax=axes[1, 1])
        except AttributeError:
            # Fall back to distplot (older seaborn)
            try:
                import seaborn as sns
                sns.distplot(returns, kde=True, ax=axes[1, 1])
            except:
                # If all else fails, use matplotlib's hist
                axes[1, 1].hist(returns, bins=20, density=True, alpha=0.7)
                # Add a simple kde using gaussian_kde from scipy
                if len(returns) > 3:  # Need at least a few points for kde
                    from scipy import stats
                    try:
                        density = stats.gaussian_kde(returns)
                        x_range = np.linspace(min(returns), max(returns), 100)
                        axes[1, 1].plot(x_range, density(x_range), 'r-')
                    except:
                        pass  # Skip KDE if it fails
    
        axes[1, 1].axvline(x=0, color='red', linestyle='--')
        axes[1, 1].set_title('Return Distribution')
        axes[1, 1].set_xlabel('Daily Return')
        axes[1, 1].grid(True)
        
        # Add performance metrics
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        textstr = '\n'.join((
            f"Total Return: {performance['total_return']:.2%}",
            f"Annualized Return: {performance['annualized_return']:.2%}",
            f"Annualized Vol: {performance['annualized_vol']:.2%}",
            f"Sharpe Ratio: {performance['sharpe_ratio']:.2f}",
            f"Max Drawdown: {performance['max_drawdown']:.2%}",
            f"% Positive Months: {performance['monthly_pos']:.2%}"
        ))
        
        axes[0, 0].text(0.05, 0.05, textstr, transform=axes[0, 0].transAxes, fontsize=10,
                verticalalignment='bottom', bbox=props)
        
        # Adjust layout
        plt.tight_layout()
        plt.show()
    
    def plot_mispricing_indices(self, results, num_indices=3):
        """
        Plot sample mispricing indices
        
        Parameters:
        -----------
        results : dict
            Dictionary with strategy results
        num_indices : int
            Number of mispricing indices to plot
        """
        indices = results['mispricing_indices']
        
        if not indices:
            print("No mispricing indices available to plot.")
            return
        
        # Sample some indices
        sample_keys = list(indices.keys())[:min(num_indices, len(indices))]
        
        # Create figure
        fig, axes = plt.subplots(len(sample_keys), 1, figsize=(14, 4*len(sample_keys)))
        
        if len(sample_keys) == 1:
            axes = [axes]
        
        # Plot each index
        for i, key in enumerate(sample_keys):
            index = indices[key]
            
            # Plot mispricing index
            axes[i].plot(index, linewidth=2)
            
            # Add Bollinger bands
            rolling_mean = index.rolling(window=BOLLINGER_WINDOW).mean()
            rolling_std = index.rolling(window=BOLLINGER_WINDOW).std()
            upper_band = rolling_mean + BOLLINGER_K * rolling_std
            lower_band = rolling_mean - BOLLINGER_K * rolling_std
            
            axes[i].plot(rolling_mean, 'r--', linewidth=1)
            axes[i].plot(upper_band, 'g--', linewidth=1)
            axes[i].plot(lower_band, 'g--', linewidth=1)
            
            # Add title
            axes[i].set_title(f'Mispricing Index: {key.split("_")[0]}')
            axes[i].grid(True)
        
        # Adjust layout
        plt.tight_layout()
        plt.show()

def run_strategy_test():
    """Run a single strategy to test"""
    print("Testing Vine Copula Statistical Arbitrage Strategy")
    
    # Use a longer time period to ensure we have valid study periods
    start_date = '2015-01-01'
    end_date = '2022-01-01'
    
    print(f"Strategy period: {start_date} to {end_date}")
    
    # Create strategy instance
    strategy = VineCopulaStrategy(
        start_date=start_date,
        end_date=end_date,
        selection_method='extremal'
    )
    
    # Check if we have valid study periods
    if not strategy.study_periods:
        print("ERROR: No valid study periods created.")
        print("Let's create a manual study period for testing:")
        
        # Create a manual study period
        manual_period = {
            'init_start': '2015-01-01',
            'init_end': '2016-01-01',
            'formation_start': '2016-01-01',
            'formation_end': '2017-01-01',
            'trading_start': '2017-01-01',
            'trading_end': '2017-07-01'
        }
        
        # Add it to the strategy
        strategy.study_periods = [manual_period]
        print(f"Created 1 manual study period from {manual_period['init_start']} to {manual_period['trading_end']}")
        
    # Run a single backtest for the first period
    print("\nRunning a single backtest for the first period...")
    first_period = strategy.study_periods[0]
    backtest_result = strategy.run_backtest(first_period, model_type='student-t')
    
    if backtest_result:
        print("Backtest successful!")
        
        # Add the result to the strategy results
        strategy.results['returns'] = [backtest_result['returns']]
        strategy.results['positions'] = [backtest_result['positions']]
        strategy.results['trades'] = [backtest_result['trades']]
        
        # Create overall results
        results = {
            'daily_returns': backtest_result['returns'],
            'performance': strategy.calculate_performance(backtest_result['returns']),
            'positions': strategy.results['positions'],
            'trades': strategy.results['trades'],
            'mispricing_indices': strategy.results['mispricing_indices']
        }
        
        # Print performance metrics
        print("\nStrategy Performance:")
        for metric, value in results['performance'].items():
            if isinstance(value, float):
                print(f"{metric}: {value:.4f}")
            else:
                print(f"{metric}: {value}")
        
        # Plot results
        strategy.plot_results(results)
        
        # Plot mispricing indices
        if results['mispricing_indices']:
            strategy.plot_mispricing_indices(results)
        
        return results
    else:
        print("Backtest failed. No results to show.")
        return None
 

# Run the test
if __name__ == "__main__":
    # Run a single strategy test
    run_strategy_test()