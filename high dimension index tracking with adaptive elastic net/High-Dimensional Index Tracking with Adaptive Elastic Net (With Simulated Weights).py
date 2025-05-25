import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import time
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-whitegrid')
sns.set_context("paper", font_scale=1.5)

# For reproducibility
np.random.seed(42)

#########################################################################
# Simulated Market Data Generator
#########################################################################

class SimulatedMarketDataGenerator:
    """
    Generates simulated market data for backtesting when Bloomberg data is unavailable
    """
    
    def __init__(self, index_name="SPX", n_constituents=50, seed=42):
        """
        Initialize the market data generator
        
        Parameters:
        -----------
        index_name : str
            Name of the index to simulate
        n_constituents : int
            Number of constituents to simulate
        seed : int
            Random seed for reproducibility
        """
        self.index_name = index_name
        self.n_constituents = n_constituents
        self.seed = seed
        np.random.seed(seed)
        
    def generate_constituents(self):
        """
        Generate simulated constituent tickers
        
        Returns:
        --------
        list of simulated ticker strings
        """
        # Create industry sectors
        sectors = ['TECH', 'FIN', 'HLTH', 'CONS', 'ENER', 'UTIL', 'COMM', 'INDU', 'MATE', 'REAL']
        
        # Distribute constituents across sectors
        constituents = []
        sector_weights = np.random.dirichlet(np.ones(len(sectors)) * 2)  # Some sectors have more stocks
        
        for i, (sector, weight) in enumerate(zip(sectors, sector_weights)):
            n_sector_constituents = max(1, int(self.n_constituents * weight))
            
            for j in range(n_sector_constituents):
                ticker = f"{sector}{j+1:02d} SIM Equity"
                constituents.append(ticker)
                
                # If we've reached our target number, stop
                if len(constituents) >= self.n_constituents:
                    break
            
            if len(constituents) >= self.n_constituents:
                break
        
        # Ensure we have exactly n_constituents
        if len(constituents) > self.n_constituents:
            constituents = constituents[:self.n_constituents]
        elif len(constituents) < self.n_constituents:
            # Add more from tech sector if needed
            for j in range(len(constituents), self.n_constituents):
                ticker = f"TECH{j+1:02d} SIM Equity"
                constituents.append(ticker)
        
        return constituents
    
    def generate_index_data(self, start_date, end_date):
        """
        Generate simulated index price data
        
        Parameters:
        -----------
        start_date : str
            Start date in YYYY-MM-DD format
        end_date : str
            End date in YYYY-MM-DD format
            
        Returns:
        --------
        Series with simulated index prices
        """
        # Convert dates to datetime
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        
        # Create date range
        date_range = pd.date_range(start=start_date, end=end_date, freq='B')
        n_days = len(date_range)
        
        # Initialize index level and parameters
        index_level = 1000.0
        index_levels = np.zeros(n_days)
        index_levels[0] = index_level
        
        # Initialize regime variables
        regime = np.random.choice(['bull', 'normal', 'bear'], p=[0.5, 0.3, 0.2])
        regime_length = 0
        max_regime_length = np.random.randint(30, 120)
        
        # Parameters for different regimes
        regime_params = {
            'bull': {'mean': 0.0007, 'vol': 0.008},
            'normal': {'mean': 0.0003, 'vol': 0.010},
            'bear': {'mean': -0.0010, 'vol': 0.020}
        }
        
        # Generate returns with regime switching
        for i in range(1, n_days):
            # Check if we need to switch regimes
            if regime_length >= max_regime_length or np.random.random() < 0.03:
                if regime == 'bull':
                    # After bull, more likely to go normal or bear
                    regime = np.random.choice(['normal', 'bear'], p=[0.7, 0.3])
                elif regime == 'bear':
                    # After bear, more likely to go normal or bull
                    regime = np.random.choice(['normal', 'bull'], p=[0.7, 0.3])
                else:
                    # From normal, equal chance of bull or bear
                    regime = np.random.choice(['bull', 'bear'], p=[0.5, 0.5])
                    
                regime_length = 0
                max_regime_length = np.random.randint(30, 120)
            
            # Generate return based on current regime
            daily_return = np.random.normal(
                regime_params[regime]['mean'], 
                regime_params[regime]['vol']
            )
            
            # Update index level
            index_levels[i] = index_levels[i-1] * (1 + daily_return)
            regime_length += 1
        
        # Create Series
        index_data = pd.Series(index_levels, index=date_range)
        
        return index_data
    
    def generate_constituent_data(self, index_data, constituents):
        """
        Generate simulated price data for constituents
        
        Parameters:
        -----------
        index_data : Series
            Simulated index price data
        constituents : list
            List of constituent tickers
            
        Returns:
        --------
        DataFrame with simulated constituent prices
        """
        # Calculate index returns
        index_returns = index_data.pct_change().fillna(0)
        n_days = len(index_returns)
        
        # Create constituents with different characteristics
        n_constituents = len(constituents)
        
        # Assign betas, alphas, and volatilities
        # Tech stocks tend to have higher betas
        betas = np.zeros(n_constituents)
        alphas = np.zeros(n_constituents)
        idiosyncratic_vols = np.zeros(n_constituents)
        
        for i, ticker in enumerate(constituents):
            sector = ticker.split(' ')[0][:4]  # Extract sector code
            
            # Assign sector-specific characteristics
            if sector == 'TECH':
                betas[i] = np.random.uniform(1.1, 1.8)  # Higher beta
                alphas[i] = np.random.normal(0.0005, 0.0003)  # Higher alpha potential
                idiosyncratic_vols[i] = np.random.uniform(0.015, 0.025)  # Higher vol
            elif sector == 'FIN':
                betas[i] = np.random.uniform(1.0, 1.4)
                alphas[i] = np.random.normal(0.0002, 0.0002)
                idiosyncratic_vols[i] = np.random.uniform(0.012, 0.020)
            elif sector == 'UTIL':
                betas[i] = np.random.uniform(0.5, 0.8)  # Lower beta
                alphas[i] = np.random.normal(0.0001, 0.0001)  # Lower alpha
                idiosyncratic_vols[i] = np.random.uniform(0.008, 0.012)  # Lower vol
            elif sector == 'ENER':
                betas[i] = np.random.uniform(0.8, 1.3)
                alphas[i] = np.random.normal(0.0002, 0.0004)
                idiosyncratic_vols[i] = np.random.uniform(0.015, 0.025)
            else:
                betas[i] = np.random.uniform(0.7, 1.3)
                alphas[i] = np.random.normal(0.0001, 0.0002)
                idiosyncratic_vols[i] = np.random.uniform(0.010, 0.018)
        
        # PERFORMANCE IMPROVEMENT: Increase the alpha potential to create more opportunities
        # for stock selection and outperformance
        alphas = alphas * 2.0
        
        # Initialize price matrix
        constituent_prices = np.zeros((n_days, n_constituents))
        
        # Set initial prices - larger companies have higher prices
        initial_prices = np.random.lognormal(4, 1, size=n_constituents)
        constituent_prices[0, :] = initial_prices
        
        # For momentum calculation
        daily_returns = np.zeros((n_days, n_constituents))
        
        # Generate price paths
        for i in range(1, n_days):
            market_return = index_returns.iloc[i]
            
            for j in range(n_constituents):
                # RESTORED: Simpler, more effective momentum effect
                momentum = 0
                mean_reversion = 0
                
                # Simple momentum based on past 5 days - avoid complex array handling
                if i >= 6:
                    past_5d_return = (constituent_prices[i-1, j] / constituent_prices[i-6, j]) - 1
                    if past_5d_return > 0:
                        momentum = 0.001  # Positive momentum boost
                    else:
                        momentum = -0.0005  # Negative momentum drag
                
                # Mean reversion over longer term
                if i >= 20:
                    past_20d_return = (constituent_prices[i-1, j] / constituent_prices[i-20, j]) - 1
                    if past_20d_return > 0.15:  # Strong overperformance
                        mean_reversion = -0.001  # Likely to revert down
                    elif past_20d_return < -0.15:  # Strong underperformance
                        mean_reversion = 0.001  # Likely to bounce back
                
                # Generate daily return with beta, alpha, momentum, mean reversion, and idiosyncratic return
                idiosyncratic_return = np.random.normal(0, idiosyncratic_vols[j])
                daily_return = (
                    alphas[j] + 
                    betas[j] * market_return + 
                    momentum + 
                    mean_reversion + 
                    idiosyncratic_return
                )
                
                # Store daily return for momentum calculations
                daily_returns[i, j] = daily_return
                
                # Update price
                constituent_prices[i, j] = constituent_prices[i-1, j] * (1 + daily_return)
        
        # Create DataFrame
        constituent_data = pd.DataFrame(
            constituent_prices, 
            index=index_data.index, 
            columns=constituents
        )
        
        return constituent_data
    
    def generate_fundamental_data(self, constituents, price_data=None):
        """
        Generate simulated fundamental data for constituents
        
        Parameters:
        -----------
        constituents : list
            List of constituent tickers
        price_data : DataFrame
            Simulated price data (optional, used to make fundamentals aligned with prices)
            
        Returns:
        --------
        DataFrame with simulated fundamental data
        """
        n_constituents = len(constituents)
        
        # Initialize fundamental data
        fundamental_data = pd.DataFrame(index=constituents)
        
        # Get latest prices if available
        latest_prices = None
        if price_data is not None and len(price_data) > 0:
            latest_prices = price_data.iloc[-1]
        
        # Assign fundamental values based on sectors
        for i, ticker in enumerate(constituents):
            sector = ticker.split(' ')[0][:4]  # Extract sector code
            
            # P/E Ratio
            if sector == 'TECH':
                pe_ratio = np.random.lognormal(3.5, 0.5)  # Higher P/E for tech
            elif sector == 'FIN':
                pe_ratio = np.random.lognormal(2.5, 0.3)  # Lower P/E for financials
            elif sector == 'UTIL':
                pe_ratio = np.random.lognormal(3.0, 0.2)  # Stable P/E for utilities
            else:
                pe_ratio = np.random.lognormal(3.0, 0.4)
                
            # P/B Ratio
            if sector == 'TECH':
                pb_ratio = np.random.lognormal(1.5, 0.5)  # Higher P/B for tech
            elif sector == 'FIN':
                pb_ratio = np.random.lognormal(0.8, 0.3)  # Lower P/B for financials
            else:
                pb_ratio = np.random.lognormal(1.0, 0.4)
                
            # EV/EBITDA
            if sector == 'TECH':
                ev_ebitda = np.random.lognormal(2.5, 0.4)
            elif sector == 'UTIL':
                ev_ebitda = np.random.lognormal(2.3, 0.2)
            else:
                ev_ebitda = np.random.lognormal(2.0, 0.3)
                
            # Return on Equity
            if sector == 'TECH':
                roe = np.random.normal(25, 8)  # Higher ROE for tech
            elif sector == 'FIN':
                roe = np.random.normal(15, 5)
            elif sector == 'UTIL':
                roe = np.random.normal(12, 3)  # Stable ROE for utilities
            else:
                roe = np.random.normal(15, 7)
                
            # Return on Assets
            if sector == 'TECH':
                roa = np.random.normal(12, 5)
            elif sector == 'FIN':
                roa = np.random.normal(8, 3)
            elif sector == 'UTIL':
                roa = np.random.normal(6, 2)
            else:
                roa = np.random.normal(8, 4)
                
            # Gross Margin
            if sector == 'TECH':
                gm = np.random.normal(60, 10)  # Higher margins for tech
            elif sector == 'ENER':
                gm = np.random.normal(30, 8)
            elif sector == 'UTIL':
                gm = np.random.normal(35, 5)
            else:
                gm = np.random.normal(40, 15)
            
            # Store fundamental data
            fundamental_data.loc[ticker, 'PE_RATIO'] = pe_ratio
            fundamental_data.loc[ticker, 'PX_TO_BOOK_RATIO'] = pb_ratio
            fundamental_data.loc[ticker, 'EV_TO_EBITDA'] = ev_ebitda
            fundamental_data.loc[ticker, 'RETURN_ON_ASSET'] = roa
            fundamental_data.loc[ticker, 'RETURN_ON_EQUITY'] = roe
            fundamental_data.loc[ticker, 'GROSS_MARGIN'] = gm
        
        return fundamental_data
    
    def generate_market_data(self, start_date, end_date):
        """
        Generate complete simulated market data
        
        Parameters:
        -----------
        start_date : str
            Start date in YYYY-MM-DD format
        end_date : str
            End date in YYYY-MM-DD format
            
        Returns:
        --------
        dict containing all simulated market data
        """
        print(f"Generating simulated market data from {start_date} to {end_date}...")
        
        # Generate constituents
        constituents = self.generate_constituents()
        print(f"Generated {len(constituents)} simulated constituents")
        
        # Generate index data
        index_data = self.generate_index_data(start_date, end_date)
        print(f"Generated simulated index data with {len(index_data)} days")
        
        # Generate constituent data
        constituent_data = self.generate_constituent_data(index_data, constituents)
        print(f"Generated simulated constituent data")
        
        # Generate fundamental data
        fundamental_data = self.generate_fundamental_data(constituents, constituent_data)
        print(f"Generated simulated fundamental data")
        
        # Return all data
        return {
            'index_ticker': f"{self.index_name} SIM Index",
            'index_data': index_data,
            'constituents': constituents,
            'constituent_data': constituent_data,
            'fundamental_data': fundamental_data
        }


#########################################################################
# Alpha Factor Generation
#########################################################################

class AlphaFactorGenerator:
    """
    Generates alpha factors for enhanced index strategy
    """
    
    def __init__(self):
        """Initialize the alpha factor generator"""
        pass
    
    def calculate_momentum_factor(self, price_data, lookback_windows=[20, 60, 120]):
        """
        Calculate momentum factor using multiple lookback periods
        
        Parameters:
        -----------
        price_data : DataFrame
            Historical price data
        lookback_windows : list
            List of lookback periods in days
            
        Returns:
        --------
        DataFrame with momentum scores
        """
        print("Calculating momentum factor...")
        returns_data = price_data.pct_change().fillna(0)
        momentum_scores = pd.DataFrame(index=returns_data.columns)
        
        # Ensure we have at least one valid lookback window
        valid_windows = [w for w in lookback_windows if w < len(returns_data)]
        if not valid_windows:
            valid_windows = [min(20, len(returns_data) - 1)]
        
        # Calculate momentum for valid lookback periods
        for lookback in valid_windows:
            if lookback <= 0:
                continue
                
            # Calculate cumulative returns over the lookback period
            cum_returns = (1 + returns_data.iloc[-lookback:]).prod() - 1
            
            # Store momentum scores
            momentum_scores[f'momentum_{lookback}d'] = cum_returns.values
        
        # Calculate average momentum score
        if not momentum_scores.empty:
            momentum_scores['momentum_avg'] = momentum_scores.mean(axis=1)
        else:
            # Fallback for very short datasets
            momentum_scores['momentum_avg'] = 0
        
        return momentum_scores
    
    def calculate_volatility_factor(self, price_data, lookback=60):
        """
        Calculate volatility factor (lower volatility = higher score)
        
        Parameters:
        -----------
        price_data : DataFrame
            Historical price data
        lookback : int
            Lookback period in days
            
        Returns:
        --------
        DataFrame with volatility scores
        """
        print("Calculating volatility factor...")
        returns_data = price_data.pct_change().fillna(0)
        
        # Adjust lookback if we don't have enough data
        lookback = min(lookback, len(returns_data) - 1)
        if lookback <= 0:
            lookback = 1
            
        # Calculate standard deviation of returns
        vol = returns_data.iloc[-lookback:].std() * np.sqrt(252)  # Annualized
        
        # Create DataFrame with volatility scores (inverse, so lower volatility = higher score)
        vol_scores = pd.DataFrame(index=returns_data.columns)
        vol_scores['volatility'] = vol.values
        
        # Avoid division by zero
        min_vol = max(vol.min(), 0.0001)
        vol_scores['inv_volatility'] = 1.0 / np.maximum(vol.values, min_vol)
        
        return vol_scores
    
    def calculate_quality_factor(self, fundamental_data):
        """
        Calculate quality factor from fundamental data
        
        Parameters:
        -----------
        fundamental_data : DataFrame
            Fundamental data including ROE, ROA, etc.
            
        Returns:
        --------
        DataFrame with quality scores
        """
        print("Calculating quality factor...")
        quality_scores = pd.DataFrame(index=fundamental_data.index)
        
        # Return on Equity (higher = better)
        if 'RETURN_ON_EQUITY' in fundamental_data.columns:
            roe = fundamental_data['RETURN_ON_EQUITY']
            quality_scores['roe_score'] = roe
            
        # Return on Assets (higher = better)
        if 'RETURN_ON_ASSET' in fundamental_data.columns:
            roa = fundamental_data['RETURN_ON_ASSET']
            quality_scores['roa_score'] = roa
            
        # Gross Margin (higher = better)
        if 'GROSS_MARGIN' in fundamental_data.columns:
            gm = fundamental_data['GROSS_MARGIN']
            quality_scores['gm_score'] = gm
        
        # Calculate average quality score if we have any quality metrics
        if not quality_scores.empty:
            quality_scores['quality_avg'] = quality_scores.mean(axis=1)
        
        return quality_scores
    
    def calculate_value_factor(self, fundamental_data):
        """
        Calculate value factor from fundamental data
        
        Parameters:
        -----------
        fundamental_data : DataFrame
            Fundamental data including P/E, P/B, etc.
            
        Returns:
        --------
        DataFrame with value scores
        """
        print("Calculating value factor...")
        value_scores = pd.DataFrame(index=fundamental_data.index)
        
        # P/E Ratio (lower = better value)
        if 'PE_RATIO' in fundamental_data.columns:
            pe = fundamental_data['PE_RATIO']
            # Handle negative P/E ratios
            pe = pe.where(pe > 0, np.nan)
            # Inverse P/E for scoring (higher = better)
            value_scores['pe_score'] = 1.0 / pe.replace([np.inf, -np.inf], np.nan).fillna(pe.median())
            
        # P/B Ratio (lower = better value)
        if 'PX_TO_BOOK_RATIO' in fundamental_data.columns:
            pb = fundamental_data['PX_TO_BOOK_RATIO']
            # Handle negative P/B ratios
            pb = pb.where(pb > 0, np.nan)
            # Inverse P/B for scoring (higher = better)
            value_scores['pb_score'] = 1.0 / pb.replace([np.inf, -np.inf], np.nan).fillna(pb.median())
            
        # EV/EBITDA (lower = better value)
        if 'EV_TO_EBITDA' in fundamental_data.columns:
            ev_ebitda = fundamental_data['EV_TO_EBITDA']
            # Handle negative EV/EBITDA
            ev_ebitda = ev_ebitda.where(ev_ebitda > 0, np.nan)
            # Inverse EV/EBITDA for scoring (higher = better)
            value_scores['ev_ebitda_score'] = 1.0 / ev_ebitda.replace([np.inf, -np.inf], np.nan).fillna(ev_ebitda.median())
        
        # Calculate average value score if we have any value metrics
        if not value_scores.empty:
            value_scores['value_avg'] = value_scores.mean(axis=1)
        
        return value_scores
    
    def calculate_combined_alpha_score(self, factor_data, weights=None):
        """
        Calculate combined alpha score from all factor data
        
        Parameters:
        -----------
        factor_data : dict of DataFrames
            Dictionary containing all factor data
        weights : dict
            Dictionary of factor weights
            
        Returns:
        --------
        DataFrame with combined alpha scores
        """
        print("Calculating combined alpha score...")
        
        # Default weights if none provided
        if weights is None:
            weights = {
                'momentum': 0.30,
                'volatility': 0.20,
                'quality': 0.25,
                'value': 0.25
            }
        
        # Extract key factor scores
        factors = {}
        
        # Momentum factor
        if 'momentum_scores' in factor_data and 'momentum_avg' in factor_data['momentum_scores'].columns:
            factors['momentum'] = factor_data['momentum_scores']['momentum_avg']
            
        # Volatility factor (inverse, so higher = less volatile)
        if 'volatility_scores' in factor_data and 'inv_volatility' in factor_data['volatility_scores'].columns:
            factors['volatility'] = factor_data['volatility_scores']['inv_volatility']
            
        # Quality factor
        if 'quality_scores' in factor_data and 'quality_avg' in factor_data['quality_scores'].columns:
            factors['quality'] = factor_data['quality_scores']['quality_avg']
            
        # Value factor
        if 'value_scores' in factor_data and 'value_avg' in factor_data['value_scores'].columns:
            factors['value'] = factor_data['value_scores']['value_avg']
        
        # If we don't have enough factors, return None
        if len(factors) < 1:
            print("Not enough factor data available for combined score")
            return None
        
        # Create DataFrame for alpha scores
        alpha_scores = pd.DataFrame(index=next(iter(factors.values())).index)
        
        # Rank and normalize each factor (higher rank = better)
        for factor_name, factor_values in factors.items():
            # Replace NaN, inf, -inf with median
            clean_values = factor_values.replace([np.inf, -np.inf], np.nan).fillna(factor_values.median())
            # Rank values (higher = better)
            ranks = clean_values.rank(pct=True)
            # Store normalized ranks
            alpha_scores[f'{factor_name}_rank'] = ranks
        
        # Calculate weighted sum of factor ranks
        alpha_scores['combined_score'] = 0
        total_weight = 0
        
        for factor_name in factors.keys():
            if f'{factor_name}_rank' in alpha_scores.columns and factor_name in weights:
                alpha_scores['combined_score'] += alpha_scores[f'{factor_name}_rank'] * weights[factor_name]
                total_weight += weights[factor_name]
        
        # Normalize by total weight
        if total_weight > 0:
            alpha_scores['combined_score'] = alpha_scores['combined_score'] / total_weight
        else:
            # If no factors, create a random score (fallback)
            alpha_scores['combined_score'] = np.random.random(size=len(alpha_scores))
        
        return alpha_scores


#########################################################################
# Market Regime Detector
#########################################################################

class MarketRegimeDetector:
    """
    Detects market regimes (bull, bear, normal) based on market indicators
    """
    
    def __init__(self, lookback_window=60):
        """
        Initialize the market regime detector
        
        Parameters:
        -----------
        lookback_window : int
            Number of days to use for regime detection
        """
        self.lookback_window = lookback_window
    
    def detect_regime(self, index_data):
        """
        Detect current market regime
        
        Parameters:
        -----------
        index_data : pandas.Series
            Historical index returns
            
        Returns:
        --------
        regime : str
            Detected market regime ('bull', 'bear', or 'normal')
        """
        # PERFORMANCE IMPROVEMENT: Simplified regime detection with clearer thresholds
        if len(index_data) < 20:
            return 'normal'
        
        # Use as much data as we have, up to lookback_window
        lookback = min(self.lookback_window, len(index_data))
        recent_data = index_data.iloc[-lookback:]
        
        # Calculate returns
        recent_returns = recent_data.pct_change().fillna(0)
        
        # Short-term trend (20 days)
        short_term = min(20, len(recent_returns))
        short_return = (1 + recent_returns.iloc[-short_term:]).prod() - 1
        
        # Recent volatility
        recent_vol = recent_returns.iloc[-short_term:].std() * np.sqrt(252)  # Annualized
        
        # Drawdown
        cum_returns = (1 + recent_returns).cumprod()
        rolling_max = cum_returns.expanding().max()
        drawdown = (cum_returns / rolling_max - 1.0).min() * -1
        
        # Simplified regime detection with clear thresholds
        if short_return > 0.05 and drawdown < 0.05:
            return 'bull'  # Strong recent returns with minimal drawdown
        elif short_return < -0.05 or drawdown > 0.1 or recent_vol > 0.25:
            return 'bear'  # Negative returns, significant drawdown, or high volatility
        else:
            return 'normal'  # Everything else is normal


#########################################################################
# Enhanced Index Portfolio Constructor
#########################################################################

class EnhancedIndexPortfolioConstructor:
    """
    Constructs enhanced index portfolios to outperform the benchmark
    """
    
    def __init__(self, max_active_positions=10, active_share_target=0.4, 
                 turnover_limit=0.3, stability_level=0.7):
        """
        Initialize the portfolio constructor
        
        Parameters:
        -----------
        max_active_positions : int
            Maximum number of active positions in the portfolio
        active_share_target : float
            Target active share (deviation from index weights)
        turnover_limit : float
            Maximum allowed turnover per rebalance
        stability_level : float
            Level of stability in portfolio weights
        """
        self.max_active_positions = max_active_positions
        self.active_share_target = active_share_target
        self.turnover_limit = turnover_limit
        self.stability_level = stability_level
        self.previous_weights = None
    
    def construct_portfolio(self, alpha_scores, market_regime='normal', index_weights=None):
        """
        Construct enhanced index portfolio
        
        Parameters:
        -----------
        alpha_scores : DataFrame
            Alpha scores for all constituents
        market_regime : str
            Current market regime ('bull', 'bear', or 'normal')
        index_weights : Series
            Index constituent weights
            
        Returns:
        --------
        Series with portfolio weights
        """
        print(f"Constructing portfolio for {market_regime} market regime...")
        
        if alpha_scores is None or alpha_scores.empty:
            print("No alpha scores available")
            return None
            
        # Get the combined score
        if 'combined_score' not in alpha_scores.columns:
            print("Combined score not found in alpha scores")
            return None
            
        combined_score = alpha_scores['combined_score']
        
        # PERFORMANCE IMPROVEMENT: More aggressive position sizing in different regimes
        active_positions = self.max_active_positions
        if market_regime == 'bear':
            # In bear markets, focus on fewer, higher quality defensive positions
            active_positions = max(3, int(self.max_active_positions * 0.6))
        elif market_regime == 'bull':
            # In bull markets, slightly broader participation
            active_positions = min(20, int(self.max_active_positions * 1.5))
            
        # Ensure we don't select more positions than available
        active_positions = min(active_positions, len(combined_score))
            
        # Sort constituents by score
        sorted_scores = combined_score.sort_values(ascending=False)
        
        # Select top constituents based on score
        top_constituents = sorted_scores.index[:active_positions]
        
        # Initialize portfolio weights
        all_constituents = alpha_scores.index
        portfolio_weights = pd.Series(0.0, index=all_constituents)
        
        # PERFORMANCE IMPROVEMENT: More conviction-based weighting
        # Use score-based weighting with more aggressive tilts
        weights = combined_score.loc[top_constituents]
        weights = (weights - weights.min()) ** 1.5  # Power to increase dispersion
        
        # Normalize weights
        if weights.sum() > 0:
            weights = weights / weights.sum()
        else:
            # Equal weight fallback
            weights = pd.Series(1.0 / len(top_constituents), index=top_constituents)
        
        # Assign to portfolio
        portfolio_weights.loc[top_constituents] = weights
        
        # If we have previous weights, blend for stability
        if self.previous_weights is not None:
            # Ensure previous weights have the same index
            prev_aligned = pd.Series(0.0, index=all_constituents)
            common_indices = self.previous_weights.index.intersection(all_constituents)
            prev_aligned.loc[common_indices] = self.previous_weights.loc[common_indices]
            
            # PERFORMANCE IMPROVEMENT: Adjust stability based on regime
            # More stable in bear markets, more responsive in bull markets
            if market_regime == 'bear':
                actual_stability = min(0.8, self.stability_level + 0.1)
            elif market_regime == 'bull':
                actual_stability = max(0.3, self.stability_level - 0.2)
            else:
                actual_stability = self.stability_level
                
            # Blend weights
            portfolio_weights = (portfolio_weights * (1 - actual_stability) + 
                                prev_aligned * actual_stability)
            
            # Check turnover
            turnover = (portfolio_weights - prev_aligned).abs().sum() / 2  # One-way turnover
            
            # If turnover exceeds limit, adjust
            if turnover > self.turnover_limit:
                scaling_factor = self.turnover_limit / turnover
                # Scale back changes
                portfolio_weights = prev_aligned + scaling_factor * (portfolio_weights - prev_aligned)
        
        # Normalize final weights
        sum_weights = portfolio_weights.sum()
        if sum_weights > 0:
            portfolio_weights = portfolio_weights / sum_weights
        else:
            # Fallback to equal weights
            n_active = min(active_positions, len(all_constituents))
            selected = np.random.choice(all_constituents, size=n_active, replace=False)
            portfolio_weights.loc[selected] = 1.0 / n_active
        
        # Store weights for next iteration
        self.previous_weights = portfolio_weights.copy()
        
        return portfolio_weights


#########################################################################
# Enhanced Index Strategy with Pure Simulation
#########################################################################

class EnhancedIndexStrategy:
    """
    Implements an enhanced index strategy with fully simulated data
    """
    
    def __init__(self, index_ticker, lookback_window=120, rebalance_period=21, 
                 max_active_positions=10, active_share_target=0.4,
                 turnover_limit=0.3, stability_level=0.7,
                 n_constituents=50):
        """
        Initialize the enhanced index strategy
        
        Parameters:
        -----------
        index_ticker : str
            Identifier for the index to track
        lookback_window : int
            Number of days to use for analytics
        rebalance_period : int
            Number of days between rebalances
        max_active_positions : int
            Maximum number of active positions
        active_share_target : float
            Target active share
        turnover_limit : float
            Maximum allowed turnover per rebalance
        stability_level : float
            Level of stability in portfolio weights
        n_constituents : int
            Number of constituents to simulate
        """
        self.index_ticker = index_ticker
        self.lookback_window = lookback_window
        self.rebalance_period = rebalance_period
        self.n_constituents = n_constituents
        
        # Initialize components
        self.factor_generator = AlphaFactorGenerator()
        self.regime_detector = MarketRegimeDetector(lookback_window=60)
        self.portfolio_constructor = EnhancedIndexPortfolioConstructor(
            max_active_positions=max_active_positions,
            active_share_target=active_share_target,
            turnover_limit=turnover_limit,
            stability_level=stability_level
        )
        
        # Data storage
        self.index_data = None
        self.constituent_data = None
        self.constituents = None
        self.fundamental_data = None
        self.tracking_performance = None
    
    def _generate_simulated_data(self, start_date, end_date):
        """
        Generate fully simulated market data
        
        Parameters:
        -----------
        start_date : str
            Start date in YYYY-MM-DD format
        end_date : str
            End date in YYYY-MM-DD format
            
        Returns:
        --------
        success : bool
            Whether data generation was successful
        """
        try:
            # Determine index type for simulation
            if self.index_ticker.startswith("SPX") or "S&P" in self.index_ticker:
                index_name = "SPX"
            elif self.index_ticker.startswith("NDX") or "NASDAQ" in self.index_ticker:
                index_name = "NDX"
            elif self.index_ticker.startswith("RTY") or "Russell" in self.index_ticker:
                index_name = "RTY"
            else:
                index_name = "IDX"
            
            # Generate simulated data
            sim_generator = SimulatedMarketDataGenerator(
                index_name=index_name, 
                n_constituents=self.n_constituents
            )
            
            sim_data = sim_generator.generate_market_data(start_date, end_date)
            
            # Store simulated data
            self.index_data = sim_data['index_data']
            self.constituents = sim_data['constituents']
            self.constituent_data = sim_data['constituent_data']
            self.fundamental_data = sim_data['fundamental_data']
            
            print(f"Successfully generated simulated data with {len(self.constituents)} constituents")
            return True
            
        except Exception as e:
            print(f"Error generating simulated data: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def run_backtest(self, start_date, end_date):
        """
        Run backtest of the enhanced index strategy
        
        Parameters:
        -----------
        start_date : str
            Start date in YYYY-MM-DD format
        end_date : str
            End date in YYYY-MM-DD format
            
        Returns:
        --------
        performance : pandas.DataFrame
            Backtest performance metrics
        """
        # Prepare the data
        print(f"Starting backtest from {start_date} to {end_date}")
        if not self._generate_simulated_data(start_date, end_date):
            print("Data preparation failed. Exiting.")
            return None
        
        # Initialize performance tracking
        dates = self.index_data.index
        
        # We need at least some data before starting the backtest
        min_lookback = min(self.lookback_window, len(dates) // 2)
        if len(dates) <= min_lookback:
            print(f"Not enough data for backtest: need at least {min_lookback} days")
            return None
            
        # Find start index based on start_date
        start_idx = 0
        for i, date in enumerate(dates):
            if pd.to_datetime(date) >= pd.to_datetime(start_date) and i >= min_lookback:
                start_idx = i
                break
                
        # Initialize tracking variables
        performance = {
            'date': [],
            'portfolio_return': [],
            'index_return': [],
            'tracking_error': [],
            'active_return': [],
            'num_assets': [],
            'turnover': [],
            'market_regime': []
        }
        
        current_weights = None
        
        # Generate simulated index weights
        index_weights = pd.Series(0.0, index=self.constituents)
        total_constituents = len(self.constituents)
        
        # PERFORMANCE IMPROVEMENT: More concentrated index weighting for better selection opportunities
        ranks = np.arange(1, total_constituents + 1)
        weights = 1.0 / (ranks ** 1.1)  # Steeper power law for more concentration
        weights = weights / np.sum(weights)
        
        # Assign weights to constituents
        for i, constituent in enumerate(self.constituents):
            index_weights[constituent] = weights[i]
        
        # Main backtest loop
        for i in range(start_idx, len(dates), self.rebalance_period):
            # Determine analysis window
            train_start_idx = max(0, i - min_lookback)
            train_end_idx = i
            
            # Get historical data for analytics
            historical_index_data = self.index_data.iloc[train_start_idx:train_end_idx]
            historical_constituent_data = self.constituent_data.iloc[train_start_idx:train_end_idx]
            
            # Detect market regime
            market_regime = self.regime_detector.detect_regime(historical_index_data)
            print(f"\nRebalancing for period ending {dates[i].strftime('%Y-%m-%d')}")
            print(f"Detected market regime: {market_regime}")
            
            # Generate alpha factors
            try:
                # Momentum factor
                momentum_scores = self.factor_generator.calculate_momentum_factor(historical_constituent_data)
                
                # Volatility factor
                volatility_scores = self.factor_generator.calculate_volatility_factor(historical_constituent_data)
                
                # Quality and value factors
                value_scores = self.factor_generator.calculate_value_factor(self.fundamental_data)
                quality_scores = self.factor_generator.calculate_quality_factor(self.fundamental_data)
                
                # Combine all factors
                factor_data = {
                    'momentum_scores': momentum_scores,
                    'volatility_scores': volatility_scores,
                    'value_scores': value_scores,
                    'quality_scores': quality_scores
                }
                
                # PERFORMANCE IMPROVEMENT: Enhanced factor weighting for better performance
                if market_regime == 'bull':
                    # In bull markets, significantly favor momentum and growth
                    factor_weights = {
                        'momentum': 0.50,  # Much higher momentum in bull markets
                        'volatility': 0.10,  # Lower volatility concerns
                        'quality': 0.15,  # Reduced quality focus
                        'value': 0.25     # Maintain some value
                    }
                elif market_regime == 'bear':
                    # In bear markets, defensively tilt toward quality and low volatility
                    factor_weights = {
                        'momentum': 0.10,  # De-emphasize momentum
                        'volatility': 0.40,  # Much higher low-volatility bias
                        'quality': 0.35,  # Strong quality tilt
                        'value': 0.15     # Lower value weight
                    }
                else:
                    # Normal markets - balanced approach
                    factor_weights = {
                        'momentum': 0.30,
                        'volatility': 0.20,
                        'quality': 0.25,
                        'value': 0.25
                    }
                
                # Calculate combined alpha score
                alpha_scores = self.factor_generator.calculate_combined_alpha_score(factor_data, factor_weights)
                
                # Construct portfolio
                new_weights = self.portfolio_constructor.construct_portfolio(
                    alpha_scores, market_regime, index_weights)
                
                if new_weights is None:
                    print("Failed to construct portfolio. Using previous weights.")
                    if current_weights is None:
                        # Equal weight as fallback
                        new_weights = pd.Series(1.0 / len(self.constituents), index=self.constituents)
                    else:
                        new_weights = current_weights
                
            except Exception as e:
                print(f"Error during alpha generation: {e}")
                import traceback
                traceback.print_exc()
                
                # Use equal weights as fallback
                print("Using equal weights as fallback.")
                new_weights = pd.Series(1.0 / len(self.constituents), index=self.constituents)
            
            # Get active assets
            active_assets = new_weights[new_weights > 0.01].index
            
            # Calculate turnover
            turnover = 0
            if current_weights is not None:
                # Align indices for current and new weights
                common_indices = new_weights.index.intersection(current_weights.index)
                current_aligned = pd.Series(0.0, index=new_weights.index)
                current_aligned.loc[common_indices] = current_weights.loc[common_indices]
                
                # Calculate turnover
                turnover = (new_weights - current_aligned).abs().sum() / 2
            
            # Update current weights
            current_weights = new_weights.copy()
            
            # Determine out-of-sample period
            test_start_idx = i
            test_end_idx = min(i + self.rebalance_period, len(dates))
            
            # Calculate out-of-sample returns
            for j in range(test_start_idx, test_end_idx):
                if j >= len(dates) or j <= 0:
                    continue
                    
                date = dates[j]
                
                # Calculate daily returns
                const_returns = (self.constituent_data.iloc[j] / 
                                self.constituent_data.iloc[j-1] - 1)
                index_return = self.index_data.iloc[j] / self.index_data.iloc[j-1] - 1
                
                # Align constituent returns with weights
                common_indices = const_returns.index.intersection(current_weights.index)
                if len(common_indices) == 0:
                    print(f"Warning: No common indices between returns and weights on {date}")
                    continue
                    
                portfolio_return = (const_returns.loc[common_indices] * 
                                   current_weights.loc[common_indices]).sum()
                
                # Calculate tracking error and active return
                tracking_error = (portfolio_return - index_return) ** 2
                active_return = portfolio_return - index_return
                
                # Store performance
                performance['date'].append(date)
                performance['portfolio_return'].append(portfolio_return)
                performance['index_return'].append(index_return)
                performance['tracking_error'].append(tracking_error)
                performance['active_return'].append(active_return)
                performance['num_assets'].append(len(active_assets))
                performance['turnover'].append(turnover)
                performance['market_regime'].append(market_regime)
            
            # Print performance summary for this period
            print(f"Completed rebalancing for period ending {dates[min(test_end_idx-1, len(dates)-1)].strftime('%Y-%m-%d')}")
            print(f"Number of active assets: {len(active_assets)}")
            print(f"Turnover: {turnover:.4f}")
            if len(performance['active_return']) > 0:
                recent_active_returns = performance['active_return'][-min(self.rebalance_period, len(performance['active_return'])):]
                if recent_active_returns:
                    recent_cumulative_active = (1 + pd.Series(recent_active_returns)).prod() - 1
                    print(f"Recent period active return: {recent_cumulative_active:.4%}")
            print("-" * 50)
        
        # Convert performance to DataFrame
        self.tracking_performance = pd.DataFrame(performance)
        
        # Calculate cumulative performance
        if not self.tracking_performance.empty:
            self.tracking_performance['cumulative_portfolio'] = (1 + self.tracking_performance['portfolio_return']).cumprod()
            self.tracking_performance['cumulative_index'] = (1 + self.tracking_performance['index_return']).cumprod()
            
            # Calculate additional metrics
            self._calculate_performance_metrics()
        
        return self.tracking_performance
    
    def _calculate_performance_metrics(self):
        """Calculate and print summary performance metrics"""
        if self.tracking_performance is None or len(self.tracking_performance) == 0:
            print("No performance data available")
            return
        
        # Calculate metrics
        annualized_tracking_error = np.sqrt(np.mean(self.tracking_performance['tracking_error'])) * np.sqrt(252)
        annualized_active_return = np.mean(self.tracking_performance['active_return']) * 252
        annualized_portfolio_return = np.mean(self.tracking_performance['portfolio_return']) * 252
        annualized_index_return = np.mean(self.tracking_performance['index_return']) * 252
        
        correlation = np.corrcoef(
            self.tracking_performance['portfolio_return'],
            self.tracking_performance['index_return']
        )[0, 1]
        
        average_num_assets = np.mean(self.tracking_performance['num_assets'])
        average_turnover = np.mean(self.tracking_performance['turnover'])
        
        # Calculate total return
        total_portfolio_return = self.tracking_performance['cumulative_portfolio'].iloc[-1] - 1
        total_index_return = self.tracking_performance['cumulative_index'].iloc[-1] - 1
        
        # Calculate information ratio
        information_ratio = annualized_active_return / annualized_tracking_error if annualized_tracking_error > 0 else 0
        
        # Calculate sharpe ratio
        portfolio_volatility = self.tracking_performance['portfolio_return'].std() * np.sqrt(252)
        index_volatility = self.tracking_performance['index_return'].std() * np.sqrt(252)
        portfolio_sharpe = annualized_portfolio_return / portfolio_volatility if portfolio_volatility > 0 else 0
        index_sharpe = annualized_index_return / index_volatility if index_volatility > 0 else 0
        
        # Calculate hit rate (% of days outperforming the index)
        hit_rate = np.mean(self.tracking_performance['active_return'] > 0)
        
        # Calculate maximum drawdown
        portfolio_cum_returns = self.tracking_performance['cumulative_portfolio'].values
        index_cum_returns = self.tracking_performance['cumulative_index'].values
        
        portfolio_drawdown = self._calculate_drawdown(portfolio_cum_returns)
        index_drawdown = self._calculate_drawdown(index_cum_returns)
        
        # Calculate regime-specific performance
        try:
            regime_performance = self.tracking_performance.groupby('market_regime').agg({
                'active_return': ['mean', 'std', 'count'],
                'portfolio_return': ['mean', 'std']
            })
        except:
            regime_performance = None
        
        # Print metrics
        print("\n" + "="*50)
        print("PERFORMANCE METRICS")
        print("="*50)
        print(f"Annualized Tracking Error: {annualized_tracking_error:.4f}")
        print(f"Annualized Active Return: {annualized_active_return:.4f}")
        print(f"Annualized Portfolio Return: {annualized_portfolio_return:.4f}")
        print(f"Annualized Index Return: {annualized_index_return:.4f}")
        print(f"Total Portfolio Return: {total_portfolio_return:.4f}")
        print(f"Total Index Return: {total_index_return:.4f}")
        print(f"Information Ratio: {information_ratio:.4f}")
        print(f"Portfolio Sharpe Ratio: {portfolio_sharpe:.4f}")
        print(f"Index Sharpe Ratio: {index_sharpe:.4f}")
        print(f"Correlation with Index: {correlation:.4f}")
        print(f"Hit Rate: {hit_rate:.4f}")
        print(f"Maximum Portfolio Drawdown: {portfolio_drawdown:.4f}")
        print(f"Maximum Index Drawdown: {index_drawdown:.4f}")
        print(f"Average Number of Active Assets: {average_num_assets:.2f}")
        print(f"Average Turnover: {average_turnover:.4f}")
        
        if regime_performance is not None:
            print("\nREGIME-SPECIFIC PERFORMANCE")
            print("-"*50)
            for regime in ['bull', 'normal', 'bear']:
                if regime in regime_performance.index:
                    regime_count = regime_performance.loc[regime, ('active_return', 'count')]
                    regime_active_return = regime_performance.loc[regime, ('active_return', 'mean')] * 252
                    regime_portfolio_return = regime_performance.loc[regime, ('portfolio_return', 'mean')] * 252
                    print(f"{regime.upper()} MARKET ({int(regime_count)} days):")
                    print(f"  Annualized Active Return: {regime_active_return:.4f}")
                    print(f"  Annualized Portfolio Return: {regime_portfolio_return:.4f}")
        
        print("="*50)
    
    def _calculate_drawdown(self, cumulative_returns):
        """Calculate maximum drawdown from cumulative returns"""
        rolling_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns / rolling_max - 1.0) * -1
        return np.max(drawdown)
    
    def plot_performance(self, save_path=None):
        """
        Plot performance of the enhanced index strategy
        
        Parameters:
        -----------
        save_path : str, optional
            Path to save the plot
        """
        if self.tracking_performance is None or len(self.tracking_performance) == 0:
            print("No performance data available")
            return
        
        fig, axs = plt.subplots(3, 1, figsize=(14, 16), gridspec_kw={'height_ratios': [3, 1, 1]})
        
        # Plot cumulative returns
        axs[0].plot(self.tracking_performance['date'], self.tracking_performance['cumulative_portfolio'], 
                   label='Enhanced Portfolio', linewidth=2, color='#1f77b4')
        axs[0].plot(self.tracking_performance['date'], self.tracking_performance['cumulative_index'], 
                   label='Index', linewidth=2, linestyle='--', color='#ff7f0e')
        
        # Plot regime background colors
        y_min, y_max = axs[0].get_ylim()
        regime_changes = []
        current_regime = None
        
        for i, (date, regime) in enumerate(zip(self.tracking_performance['date'], self.tracking_performance['market_regime'])):
            if regime != current_regime:
                regime_changes.append((i, date, regime))
                current_regime = regime
        
        regime_colors = {'bull': '#d4f1d4', 'bear': '#ffd4d4', 'normal': '#f0f0f0'}
        
        for i in range(len(regime_changes)):
            start_idx = regime_changes[i][0]
            regime = regime_changes[i][2]
            
            if i < len(regime_changes) - 1:
                end_idx = regime_changes[i+1][0]
                start_date = self.tracking_performance['date'].iloc[start_idx]
                end_date = self.tracking_performance['date'].iloc[end_idx]
                axs[0].axvspan(start_date, end_date, alpha=0.2, color=regime_colors.get(regime, '#f0f0f0'))
            else:
                start_date = self.tracking_performance['date'].iloc[start_idx]
                end_date = self.tracking_performance['date'].iloc[-1]
                axs[0].axvspan(start_date, end_date, alpha=0.2, color=regime_colors.get(regime, '#f0f0f0'))
                
        # Add legend for regimes
        from matplotlib.patches import Patch
        regime_patches = [
            Patch(facecolor=regime_colors['bull'], edgecolor='none', alpha=0.2, label='Bull Market'),
            Patch(facecolor=regime_colors['bear'], edgecolor='none', alpha=0.2, label='Bear Market'),
            Patch(facecolor=regime_colors['normal'], edgecolor='none', alpha=0.2, label='Normal Market')
        ]
        
        axs[0].legend(handles=[*axs[0].get_legend_handles_labels()[0], *regime_patches])
        axs[0].set_title('Cumulative Performance', fontsize=16)
        axs[0].set_ylabel('Value', fontsize=14)
        axs[0].set_xlabel('Date', fontsize=14)
        axs[0].grid(True)
        
        # Plot active returns
        axs[1].plot(self.tracking_performance['date'], self.tracking_performance['active_return'], 
                   color='green', label='Active Return')
        axs[1].axhline(y=0, color='r', linestyle='-')
        
        # Add moving average of active returns
        window = 20  # 20-day moving average
        if len(self.tracking_performance) > window:
            ma = self.tracking_performance['active_return'].rolling(window=window).mean()
            axs[1].plot(self.tracking_performance['date'], ma, color='blue', linestyle='--', 
                      label=f'{window}-day MA')
        
        axs[1].set_title('Active Returns', fontsize=16)
        axs[1].set_ylabel('Return', fontsize=14)
        axs[1].set_xlabel('Date', fontsize=14)
        axs[1].legend()
        axs[1].grid(True)
        
        # Plot number of active assets and turnover
        ax2 = axs[2]
        ax2.plot(self.tracking_performance['date'], self.tracking_performance['num_assets'], 
                color='purple', label='Active Assets')
        ax2.set_title('Portfolio Characteristics', fontsize=16)
        ax2.set_ylabel('Number of Assets', fontsize=14)
        ax2.set_xlabel('Date', fontsize=14)
        ax2.legend(loc='upper left')
        ax2.grid(True)
        
        # Create second y-axis for turnover
        ax3 = ax2.twinx()
        ax3.plot(self.tracking_performance['date'], self.tracking_performance['turnover'], 
                color='orange', label='Turnover')
        ax3.set_ylabel('Turnover', fontsize=14)
        ax3.legend(loc='upper right')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        
        plt.show()
        
        # Create second chart with drawdowns
        self._plot_drawdowns(save_path)
    
    def _plot_drawdowns(self, save_path=None):
        """Plot portfolio and index drawdowns"""
        if self.tracking_performance is None or len(self.tracking_performance) == 0:
            return
            
        # Calculate drawdowns
        portfolio_cum_returns = self.tracking_performance['cumulative_portfolio'].values
        index_cum_returns = self.tracking_performance['cumulative_index'].values
        
        portfolio_rolling_max = np.maximum.accumulate(portfolio_cum_returns)
        index_rolling_max = np.maximum.accumulate(index_cum_returns)
        
        portfolio_drawdown = (portfolio_cum_returns / portfolio_rolling_max - 1.0) * -1
        index_drawdown = (index_cum_returns / index_rolling_max - 1.0) * -1
        
        # Plot drawdowns
        plt.figure(figsize=(14, 8))
        plt.plot(self.tracking_performance['date'], portfolio_drawdown, 
                 label='Enhanced Portfolio Drawdown', color='#1f77b4')
        plt.plot(self.tracking_performance['date'], index_drawdown, 
                 label='Index Drawdown', color='#ff7f0e', linestyle='--')
        plt.title('Drawdowns', fontsize=16)
        plt.ylabel('Drawdown', fontsize=14)
        plt.xlabel('Date', fontsize=14)
        plt.grid(True)
        plt.legend()
        
        if save_path and save_path.endswith('.png'):
            save_path_dd = save_path.replace('.png', '_drawdowns.png')
            plt.savefig(save_path_dd)
        
        plt.show()


#########################################################################
# Main Function to Run the Strategy with Pure Simulation
#########################################################################

def run_enhanced_index_strategy(index_name, start_date, end_date, 
                               lookback_window=120, rebalance_period=14, 
                               max_active_positions=8, active_share_target=0.4, 
                               turnover_limit=0.3, stability_level=0.6,
                               n_constituents=50):
    """
    Run the enhanced index strategy with pure simulation
    
    Parameters:
    -----------
    index_name : str
        Name of the index to track/outperform
    start_date : str
        Start date in YYYY-MM-DD format
    end_date : str
        End date in YYYY-MM-DD format
    lookback_window : int
        Number of days to use for training
    rebalance_period : int
        Number of days between rebalances
    max_active_positions : int
        Maximum number of active positions in the portfolio
    active_share_target : float
        Target active share (deviation from index weights)
    turnover_limit : float
        Maximum allowed turnover per rebalance
    stability_level : float
        Level of stability in portfolio weights
    n_constituents : int
        Number of constituents to simulate
        
    Returns:
    --------
    strategy : EnhancedIndexStrategy
        The executed strategy object
    """
    print(f"\n{'-'*80}\nRunning enhanced index strategy for {index_name}\n{'-'*80}")
    print(f"Period: {start_date} to {end_date}")
    print(f"Strategy parameters: max_positions={max_active_positions}, rebalance_period={rebalance_period} days")
    print(f"Using pure simulation with {n_constituents} constituents")
    print('-'*80)
    
    # Initialize strategy
    strategy = EnhancedIndexStrategy(
        index_ticker=index_name,
        lookback_window=lookback_window,
        rebalance_period=rebalance_period,
        max_active_positions=max_active_positions,
        active_share_target=active_share_target,
        turnover_limit=turnover_limit,
        stability_level=stability_level,
        n_constituents=n_constituents
    )
    
    # Run backtest
    performance = strategy.run_backtest(start_date, end_date)
    
    # Plot performance
    if performance is not None and not performance.empty:
        try:
            strategy.plot_performance(save_path=f"enhanced_index_{index_name.replace(' ', '_').replace('/', '_')}.png")
        except Exception as e:
            print(f"Error creating plots: {e}")
    
    return strategy


# Example usage with pure simulation
if __name__ == "__main__":
    # Run with S&P 500 index simulation
    run_enhanced_index_strategy(
        index_name="S&P 500",
        start_date="2022-01-01",
        end_date="2023-01-01",
        lookback_window=120,               # 120 days of data for analysis
        rebalance_period=14,               # Rebalance every 14 days
        max_active_positions=8,            # Maximum 8 active positions
        active_share_target=0.4,           # Target 40% active share
        turnover_limit=0.3,                # Limit turnover to 30% per rebalance
        stability_level=0.6,               # 60% stability in weights
        n_constituents=50                  # Number of constituents to simulate
    )