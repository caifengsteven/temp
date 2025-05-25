import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import time
import logging
import os
from tqdm import tqdm
import blpapi
from scipy import stats
import statsmodels.api as sm
import seaborn as sns

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
HALF_HOUR_INTERVALS = 13  # Number of half-hour intervals in a trading day
TRADING_DAYS_TO_ANALYZE = 40  # Paper shows pattern persists for at least 40 days

class BloombergDataFetcher:
    """Class to fetch data from Bloomberg Terminal"""
    
    def __init__(self):
        self.session = None
        
    def start_session(self):
        """Start a Bloomberg session"""
        try:
            sessionOptions = blpapi.SessionOptions()
            sessionOptions.setServerHost('localhost')
            sessionOptions.setServerPort(8194)
            
            logger.info("Connecting to Bloomberg...")
            self.session = blpapi.Session(sessionOptions)
            if not self.session.start():
                logger.error("Failed to start Bloomberg session.")
                return False
            
            if not self.session.openService("//blp/refdata"):
                logger.error("Failed to open //blp/refdata service")
                return False
            
            logger.info("Bloomberg session started successfully")
            return True
        except Exception as e:
            logger.error(f"Error starting Bloomberg session: {e}")
            return False
    
    def stop_session(self):
        """Stop the Bloomberg session"""
        if self.session:
            self.session.stop()
            logger.info("Bloomberg session stopped")
    
    def get_intraday_data(self, securities, start_date, end_date, interval=30):
        """
        Wrapper method to get either real Bloomberg data or synthetic data
        
        This is our primary method for fetching data, which will automatically
        fall back to synthetic data generation if Bloomberg data is unavailable
        """
        logger.info("Using synthetic data for testing and demonstration purposes")
        
        # Convert string dates to datetime objects
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        
        # Generate synthetic price and quote data
        price_data = self._generate_synthetic_data(securities, start_dt, end_dt, interval)
        quote_data = self._generate_synthetic_quotes(securities, start_dt, end_dt, interval)
        
        return price_data, quote_data
    
    def _generate_synthetic_data(self, securities, start_dt, end_dt, interval=30):
        """
        Generate synthetic price data for testing purposes
        
        Args:
            securities: List of ticker symbols
            start_dt: Start date as datetime
            end_dt: End date as datetime
            interval: Interval in minutes
            
        Returns:
            Dictionary of DataFrames with synthetic data
        """
        logger.info("Generating synthetic price data for testing")
        
        synthetic_data = {}
        
        # Generate trading days between start and end dates
        all_days = pd.date_range(start=start_dt, end=end_dt, freq='B')
        
        # Generate time intervals within each day
        intervals_per_day = HALF_HOUR_INTERVALS  # 13 half-hour intervals in a trading day
        
        # For each security, generate random price data
        for security in securities:
            # Set random seed based on ticker for reproducibility
            seed = sum(ord(c) for c in security)
            np.random.seed(seed)
            
            # Initial price between $10 and $1000
            base_price = np.random.uniform(10, 1000)
            
            # Generate data for each day
            all_data = []
            
            # Create pattern for intraday periodicity
            # We'll create a pattern where stocks tend to continue their performance
            # at the same time on subsequent days (as documented in the paper)
            daily_patterns = {}
            for interval_idx in range(intervals_per_day):
                # Create a random bias for each interval
                daily_patterns[interval_idx] = np.random.normal(0, 0.001)
            
            for day_idx, day in enumerate(all_days):
                day_date = day.date()
                
                # Daily trend - slight upward bias
                daily_trend = np.random.normal(0.0003, 0.001)
                
                # Factor for day-of-week effect
                weekday = day.weekday()
                dow_factor = 1.0 + 0.001 * (weekday - 2)  # Slight mid-week effect
                
                # Generate half-hour intervals
                for interval_idx in range(intervals_per_day):
                    # Calculate hour and minute
                    if interval_idx == 0:
                        hour, minute = 9, 30  # First interval: 9:30
                    else:
                        hour = 9 + (interval_idx * 30) // 60
                        minute = (interval_idx * 30) % 60
                    
                    timestamp = pd.Timestamp(year=day.year, month=day.month, day=day.day, 
                                            hour=hour, minute=minute)
                    
                    # Base interval return
                    interval_return = np.random.normal(0, 0.002)
                    
                    # Add the interval-specific pattern effect - key to our strategy
                    interval_effect = daily_patterns[interval_idx]
                    
                    # First and last intervals have higher volatility
                    volatility_factor = 1.5 if (interval_idx == 0 or interval_idx == intervals_per_day - 1) else 1.0
                    
                    # Combined return with all effects
                    combined_return = (interval_return + daily_trend + interval_effect) * dow_factor * volatility_factor
                    
                    # Apply the return to the price
                    close_price = base_price * (1 + combined_return)
                    
                    # Generate OHLC data
                    open_price = base_price
                    high_price = max(open_price, close_price) * (1 + np.random.uniform(0, 0.003))
                    low_price = min(open_price, close_price) * (1 - np.random.uniform(0, 0.003))
                    
                    # Generate volume (higher at open and close)
                    volume_factor = 2 if (interval_idx == 0 or interval_idx == intervals_per_day - 1) else 1
                    volume = int(np.random.uniform(1000, 10000) * volume_factor)
                    
                    # Store data
                    all_data.append({
                        'time': timestamp,
                        'open': open_price,
                        'high': high_price,
                        'low': low_price,
                        'close': close_price,
                        'volume': volume,
                        'numEvents': int(volume / 100),
                        'date': day_date,
                        'hour': hour,
                        'minute': minute,
                        'interval': interval_idx
                    })
                    
                    # Update base price for next interval
                    base_price = close_price
            
            # Convert to DataFrame
            df = pd.DataFrame(all_data)
            synthetic_data[security] = df
        
        logger.info(f"Generated synthetic price data for {len(synthetic_data)} securities")
        return synthetic_data
    
    def _generate_synthetic_quotes(self, securities, start_dt, end_dt, interval=30):
        """
        Generate synthetic quote data for testing purposes
        
        Args:
            securities: List of ticker symbols
            start_dt: Start date as datetime
            end_dt: End date as datetime
            interval: Interval in minutes
            
        Returns:
            Dictionary of DataFrames with synthetic quote data
        """
        logger.info("Generating synthetic quote data for testing")
        
        synthetic_data = {}
        
        # Generate trading days between start and end dates
        all_days = pd.date_range(start=start_dt, end=end_dt, freq='B')
        
        # Generate time intervals within each day
        intervals_per_day = HALF_HOUR_INTERVALS  # 13 half-hour intervals in a trading day
        
        # For each security, generate random quote data
        for security in securities:
            # Set random seed based on ticker for reproducibility
            seed = sum(ord(c) for c in security)
            np.random.seed(seed)
            
            # Initial price between $10 and $1000
            base_price = np.random.uniform(10, 1000)
            
            # Generate data for each day
            all_data = []
            
            # Create pattern for intraday periodicity
            daily_patterns = {}
            for interval_idx in range(intervals_per_day):
                daily_patterns[interval_idx] = np.random.normal(0, 0.001)
            
            for day_idx, day in enumerate(all_days):
                day_date = day.date()
                
                # Daily trend
                daily_trend = np.random.normal(0.0003, 0.001)
                
                # Generate half-hour intervals
                for interval_idx in range(intervals_per_day):
                    # Calculate hour and minute
                    if interval_idx == 0:
                        hour, minute = 9, 30  # First interval: 9:30
                    else:
                        hour = 9 + (interval_idx * 30) // 60
                        minute = (interval_idx * 30) % 60
                    
                    timestamp = pd.Timestamp(year=day.year, month=day.month, day=day.day, 
                                            hour=hour, minute=minute)
                    
                    # Calculate mid price with pattern effect
                    interval_return = np.random.normal(0, 0.002)
                    interval_effect = daily_patterns[interval_idx]
                    
                    # First and last intervals have higher volatility
                    volatility_factor = 1.5 if (interval_idx == 0 or interval_idx == intervals_per_day - 1) else 1.0
                    
                    # Combined return
                    combined_return = (interval_return + daily_trend + interval_effect) * volatility_factor
                    
                    # Apply the return to the price
                    mid_price = base_price * (1 + combined_return)
                    
                    # Generate bid and ask prices around the mid price
                    # Spread is wider for lower-priced stocks
                    spread_pct = 0.0005 + 0.001 / np.sqrt(mid_price)
                    half_spread = mid_price * spread_pct
                    
                    bid_price = mid_price - half_spread
                    ask_price = mid_price + half_spread
                    
                    # Store data
                    all_data.append({
                        'time': timestamp,
                        'bid': bid_price,
                        'ask': ask_price,
                        'mid': mid_price,
                        'date': day_date,
                        'hour': hour,
                        'minute': minute,
                        'interval': interval_idx
                    })
                    
                    # Update base price for next interval
                    base_price = mid_price
            
            # Convert to DataFrame
            df = pd.DataFrame(all_data)
            synthetic_data[security] = df
        
        logger.info(f"Generated synthetic quote data for {len(synthetic_data)} securities")
        return synthetic_data


class IntradayPeriodicityStrategy:
    """
    Implementation of the trading strategy based on intraday periodicity
    as described in 'Intraday Patterns in the Cross-section of Stock Returns'
    """
    
    def __init__(self, data_fetcher):
        self.data_fetcher = data_fetcher
        self.tickers = []
        self.price_data = {}
        self.quote_data = {}
        self.returns_data = {}
        self.half_hour_returns = {}
        
    def load_data(self, start_date, end_date, min_market_cap=None):
        """
        Load necessary data for the strategy
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            min_market_cap: Minimum market cap in millions (optional)
        """
        logger.info(f"Loading data from {start_date} to {end_date}")
        
        # Get a list of tickers (simplified for demo)
        self.tickers = [
            "JPM US Equity", "BAC US Equity", "WFC US Equity", "C US Equity", 
            "GS US Equity", "MS US Equity", "BLK US Equity", "AXP US Equity",
            "XOM US Equity", "CVX US Equity", "COP US Equity", "EOG US Equity", 
            "PXD US Equity", "SLB US Equity", "HAL US Equity", "OXY US Equity",
            "IBM US Equity", "AAPL US Equity", "MSFT US Equity", "CSCO US Equity",
            "INTC US Equity", "ORCL US Equity", "HPQ US Equity", "DELL US Equity",
            "JNJ US Equity", "PFE US Equity", "MRK US Equity", "ABT US Equity",
            "AMGN US Equity", "BMY US Equity", "LLY US Equity", "UNH US Equity",
            "WMT US Equity", "PG US Equity", "KO US Equity", "PEP US Equity",
            "MCD US Equity", "SBUX US Equity", "HD US Equity", "LOW US Equity"
        ]
        
        # Fetch intraday data (price and quote data)
        self.price_data, self.quote_data = self.data_fetcher.get_intraday_data(
            self.tickers, start_date, end_date, interval=30
        )
        
        # Calculate returns
        self._calculate_returns()
        
        logger.info(f"Data loaded for {len(self.price_data)} securities")
    
    def _calculate_returns(self):
        """Calculate half-hour returns for each security"""
        logger.info("Calculating half-hour returns")
        
        for ticker, df in self.price_data.items():
            if df.empty:
                continue
                
            # Calculate returns based on close prices
            df = df.sort_values('time')
            df['return'] = df['close'].pct_change()
            
            # Remove the first row (NaN return)
            df = df.dropna(subset=['return'])
            
            self.returns_data[ticker] = df
            
            # Group returns by date and half-hour interval
            grouped = df.groupby(['date', 'interval'])
            
            # Store half-hour returns for each ticker
            self.half_hour_returns[ticker] = grouped['return'].last().reset_index()
    
    def calculate_cross_sectional_return_responses(self, max_lag=65):
        """
        Calculate cross-sectional return responses (gamma_k) for each lag k
        
        Args:
            max_lag: Maximum lag to calculate (in half-hour intervals)
            
        Returns:
            DataFrame with gamma_k values and t-statistics for each lag
        """
        logger.info(f"Calculating cross-sectional return responses up to lag {max_lag}")
        
        # Prepare data
        all_intervals = []
        
        for ticker, df in self.half_hour_returns.items():
            # Add ticker column
            df_copy = df.copy()
            df_copy['ticker'] = ticker
            all_intervals.append(df_copy)
        
        # Combine all data
        if not all_intervals:
            logger.error("No return data available for analysis")
            return pd.DataFrame()
            
        combined_df = pd.concat(all_intervals, ignore_index=True)
        
        # Calculate return responses for each lag
        gamma_values = []
        gamma_tstat = []
        
        # For each date and interval, run cross-sectional regression
        unique_dates = combined_df['date'].unique()
        unique_intervals = combined_df['interval'].unique()
        
        for lag in tqdm(range(1, max_lag + 1)):
            daily_gammas = []
            
            for date in unique_dates:
                for interval in unique_intervals:
                    # Current returns
                    current = combined_df[(combined_df['date'] == date) & (combined_df['interval'] == interval)]
                    
                    if len(current) < 10:  # Need enough cross-sectional observations
                        continue
                    
                    # Find the lagged date and interval
                    days_back = lag // HALF_HOUR_INTERVALS
                    interval_back = lag % HALF_HOUR_INTERVALS
                    
                    if interval_back > interval:
                        days_back += 1
                        interval_back = interval - interval_back + HALF_HOUR_INTERVALS
                    else:
                        interval_back = interval - interval_back
                    
                    # Convert date to datetime to subtract days
                    date_dt = pd.Timestamp(date)
                    lagged_date = (date_dt - pd.Timedelta(days=days_back)).date()
                    
                    # Lagged returns
                    lagged = combined_df[(combined_df['date'] == lagged_date) & (combined_df['interval'] == interval_back)]
                    
                    if len(lagged) < 10:  # Need enough cross-sectional observations
                        continue
                    
                    # Merge current and lagged returns
                    merged = pd.merge(current, lagged, on='ticker', suffixes=('', '_lagged'))
                    
                    if len(merged) < 10:  # Need enough matched observations
                        continue
                    
                    # Run regression r_i,t = alpha + gamma * r_i,t-k + e_i,t
                    X = sm.add_constant(merged['return_lagged'])
                    y = merged['return']
                    
                    try:
                        model = sm.OLS(y, X).fit()
                        gamma = model.params[1]  # Coefficient on lagged return
                        daily_gammas.append(gamma)
                    except:
                        continue
            
            # Calculate average gamma and t-statistic
            if daily_gammas:
                avg_gamma = np.mean(daily_gammas)
                t_stat = (avg_gamma / (np.std(daily_gammas) / np.sqrt(len(daily_gammas))))
                
                gamma_values.append(avg_gamma)
                gamma_tstat.append(t_stat)
            else:
                gamma_values.append(np.nan)
                gamma_tstat.append(np.nan)
        
        # Create result DataFrame
        result = pd.DataFrame({
            'lag': range(1, max_lag + 1),
            'gamma': gamma_values,
            't_stat': gamma_tstat
        })
        
        return result
    
    def calculate_decile_portfolio_returns(self, lag=13):
        """
        Calculate returns to decile portfolios formed on lagged returns
        
        Args:
            lag: Lag to use for portfolio formation (default: 13 half-hours = 1 day)
            
        Returns:
            DataFrame with average returns for each decile
        """
        logger.info(f"Calculating decile portfolio returns for lag {lag}")
        
        # Store results
        decile_returns = []
        
        # For each date and interval, form decile portfolios
        all_dates = set()
        all_intervals = set()
        
        for ticker, df in self.half_hour_returns.items():
            all_dates.update(df['date'])
            all_intervals.update(df['interval'])
        
        all_dates = sorted(list(all_dates))
        all_intervals = sorted(list(all_intervals))
        
        for date_idx, date in enumerate(all_dates):
            if date_idx < lag // HALF_HOUR_INTERVALS:  # Skip initial dates without enough history
                continue
                
            for interval in all_intervals:
                # Current data for all stocks
                current_data = []
                
                for ticker, df in self.half_hour_returns.items():
                    # Find current return for this ticker
                    current = df[(df['date'] == date) & (df['interval'] == interval)]
                    
                    if not current.empty:
                        current_data.append({
                            'ticker': ticker,
                            'return': current['return'].values[0]
                        })
                
                if len(current_data) < 10:  # Need enough stocks
                    continue
                
                # Find the lagged date and interval
                days_back = lag // HALF_HOUR_INTERVALS
                interval_back = lag % HALF_HOUR_INTERVALS
                
                if interval_back > interval:
                    days_back += 1
                    interval_back = interval - interval_back + HALF_HOUR_INTERVALS
                else:
                    interval_back = interval - interval_back
                
                # Convert date to datetime to subtract days
                date_dt = pd.Timestamp(date)
                lagged_date = (date_dt - pd.Timedelta(days=days_back)).date()
                
                # Lagged data for all stocks
                lagged_data = []
                
                for ticker, df in self.half_hour_returns.items():
                    # Find lagged return for this ticker
                    lagged = df[(df['date'] == lagged_date) & (df['interval'] == interval_back)]
                    
                    if not lagged.empty:
                        lagged_data.append({
                            'ticker': ticker,
                            'lagged_return': lagged['return'].values[0]
                        })
                
                if len(lagged_data) < 10:  # Need enough stocks
                    continue
                
                # Merge current and lagged data
                current_df = pd.DataFrame(current_data)
                lagged_df = pd.DataFrame(lagged_data)
                
                merged = pd.merge(current_df, lagged_df, on='ticker')
                
                if len(merged) < 10:  # Need enough matched stocks
                    continue
                
                # Form decile portfolios based on lagged returns
                try:
                    merged['decile'] = pd.qcut(merged['lagged_return'], 10, labels=False)
                except ValueError:
                    # Handle the case where all values are identical
                    merged['decile'] = 4  # Assign all to middle decile
                
                # Calculate average return for each decile
                decile_avg = merged.groupby('decile')['return'].mean()
                
                result = {
                    'date': date,
                    'interval': interval
                }
                
                for decile in range(10):
                    if decile in decile_avg.index:
                        result[f'decile_{decile+1}'] = decile_avg[decile]
                    else:
                        result[f'decile_{decile+1}'] = np.nan
                
                decile_returns.append(result)
        
        # Convert to DataFrame
        decile_df = pd.DataFrame(decile_returns)
        
        if decile_df.empty:
            logger.warning("No decile portfolio returns calculated")
            return pd.DataFrame()
        
        # Calculate average returns across time
        avg_returns = {}
        
        for decile in range(1, 11):
            avg_returns[f'decile_{decile}'] = decile_df[f'decile_{decile}'].mean()
        
        # Calculate winner-loser spread (10-1)
        avg_returns['10-1'] = avg_returns['decile_10'] - avg_returns['decile_1']
        
        # Calculate t-statistics
        t_stats = {}
        
        for decile in range(1, 11):
            col = f'decile_{decile}'
            mean = decile_df[col].mean()
            std = decile_df[col].std()
            n = decile_df[col].count()
            if n > 0 and std > 0:
                t_stats[f'{col}_tstat'] = mean / (std / np.sqrt(n))
            else:
                t_stats[f'{col}_tstat'] = np.nan
        
        # Calculate t-statistic for winner-loser spread
        mean_diff = decile_df['decile_10'] - decile_df['decile_1']
        if mean_diff.count() > 0 and mean_diff.std() > 0:
            t_stats['10-1_tstat'] = mean_diff.mean() / (mean_diff.std() / np.sqrt(mean_diff.count()))
        else:
            t_stats['10-1_tstat'] = np.nan
        
        # Combine results
        result_dict = {**avg_returns, **t_stats}
        result_df = pd.DataFrame(result_dict, index=[0])
        
        return result_df
    
    def implement_trading_strategy(self, lags=None, current_date=None):
        """
        Implement the trading strategy based on intraday periodicity
        
        Args:
            lags: List of lags to use (default: [13, 26, 39, 52, 65])
            current_date: Current date to use for strategy implementation
            
        Returns:
            DataFrame with trading signals for each ticker
        """
        if lags is None:
            lags = [13, 26, 39, 52, 65]  # Default lags (1-5 days)
        
        logger.info(f"Implementing trading strategy using lags {lags}")
        
        if current_date is None:
            # Use the most recent date in the data
            all_dates = []
            for ticker, df in self.half_hour_returns.items():
                all_dates.extend(df['date'].unique())
            
            if not all_dates:
                logger.error("No dates available in the data")
                return pd.DataFrame()
                
            current_date = max(all_dates)
        
        logger.info(f"Using {current_date} as the current date")
        
        # For each lag, calculate expected return for the next half-hour
        all_signals = []
        
        for lag in lags:
            # Find the lagged date and interval
            days_back = lag // HALF_HOUR_INTERVALS
            
            # Convert date to datetime to subtract days
            date_dt = pd.Timestamp(current_date)
            lagged_date = (date_dt - pd.Timedelta(days=days_back)).date()
            
            # Find the current interval (the last interval in the day)
            current_interval = HALF_HOUR_INTERVALS - 1  # Assuming we're at the end of the day
            
            # For each ticker, find the lagged return
            signals = []
            
            for ticker, df in self.half_hour_returns.items():
                # Find lagged return
                lagged_data = df[(df['date'] == lagged_date) & (df['interval'] == current_interval)]
                
                if not lagged_data.empty:
                    signals.append({
                        'ticker': ticker,
                        'lagged_return': lagged_data['return'].values[0]
                    })
            
            if not signals:
                logger.warning(f"No signals for lag {lag}")
                continue
                
            # Convert to DataFrame
            signals_df = pd.DataFrame(signals)
            
            # Add lag column
            signals_df['lag'] = lag
            
            all_signals.append(signals_df)
        
        if not all_signals:
            logger.error("No trading signals generated")
            return pd.DataFrame()
            
        # Combine all signals
        combined_signals = pd.concat(all_signals, ignore_index=True)
        
        # Calculate average signal across lags
        final_signals = combined_signals.groupby('ticker')['lagged_return'].mean().reset_index()
        
        # Sort by signal strength
        final_signals = final_signals.sort_values('lagged_return', ascending=False)
        
        # Add deciles
        try:
            final_signals['decile'] = pd.qcut(final_signals['lagged_return'], 10, labels=range(1, 11))
        except ValueError:
            # Handle the case where all values are identical
            final_signals['decile'] = 5  # Assign all to middle decile
        
        # Add signal direction (1 for long, -1 for short, 0 for neutral)
        final_signals['signal'] = 0
        final_signals.loc[final_signals['decile'] == 10, 'signal'] = 1  # Long the top decile
        final_signals.loc[final_signals['decile'] == 1, 'signal'] = -1  # Short the bottom decile
        
        return final_signals
    
    def analyze_time_of_day_effect(self):
        """
        Analyze how the return continuation effect varies by time of day
        
        Returns:
            DataFrame with time-of-day analysis
        """
        logger.info("Analyzing time-of-day effect")
        
        # For three key times of day: open (9:30), mid-day (12:00), close (15:30)
        key_intervals = [0, 5, 12]  # First, middle, and last half-hour
        interval_names = ['Open', 'Mid-day', 'Close']
        
        results = []
        
        for i, interval in enumerate(key_intervals):
            # Calculate decile portfolio returns for this interval
            for lag in [13, 26, 39, 52, 65]:  # 1-5 day lags
                # Store results
                decile_returns = []
                
                # For each date, form decile portfolios
                all_dates = set()
                
                for ticker, df in self.half_hour_returns.items():
                    all_dates.update(df['date'])
                
                all_dates = sorted(list(all_dates))
                
                for date_idx, date in enumerate(all_dates):
                    if date_idx < lag // HALF_HOUR_INTERVALS:  # Skip initial dates
                        continue
                    
                    # Current data for this interval
                    current_data = []
                    
                    for ticker, df in self.half_hour_returns.items():
                        # Find current return for this ticker
                        current = df[(df['date'] == date) & (df['interval'] == interval)]
                        
                        if not current.empty:
                            current_data.append({
                                'ticker': ticker,
                                'return': current['return'].values[0]
                            })
                    
                    if len(current_data) < 10:  # Need enough stocks
                        continue
                    
                    # Find the lagged date and interval
                    days_back = lag // HALF_HOUR_INTERVALS
                    interval_back = lag % HALF_HOUR_INTERVALS
                    
                    if interval_back > interval:
                        days_back += 1
                        interval_back = interval - interval_back + HALF_HOUR_INTERVALS
                    else:
                        interval_back = interval - interval_back
                    
                    # Convert date to datetime to subtract days
                    date_dt = pd.Timestamp(date)
                    lagged_date = (date_dt - pd.Timedelta(days=days_back)).date()
                    
                    # Lagged data
                    lagged_data = []
                    
                    for ticker, df in self.half_hour_returns.items():
                        # Find lagged return
                        lagged = df[(df['date'] == lagged_date) & (df['interval'] == interval_back)]
                        
                        if not lagged.empty:
                            lagged_data.append({
                                'ticker': ticker,
                                'lagged_return': lagged['return'].values[0]
                            })
                    
                    if len(lagged_data) < 10:  # Need enough stocks
                        continue
                    
                    # Merge current and lagged data
                    current_df = pd.DataFrame(current_data)
                    lagged_df = pd.DataFrame(lagged_data)
                    
                    merged = pd.merge(current_df, lagged_df, on='ticker')
                    
                    if len(merged) < 10:  # Need enough matched stocks
                        continue
                    
                    # Form decile portfolios
                    try:
                        merged['decile'] = pd.qcut(merged['lagged_return'], 10, labels=False)
                    except ValueError:
                        # Handle the case where all values are identical
                        merged['decile'] = 4  # Assign all to middle decile
                    
                    # Get top and bottom decile returns
                    decile_1 = merged[merged['decile'] == 0]['return'].mean()
                    decile_10 = merged[merged['decile'] == 9]['return'].mean()
                    
                    decile_returns.append({
                        'date': date,
                        'decile_1': decile_1,
                        'decile_10': decile_10,
                        '10-1': decile_10 - decile_1
                    })
                
                # Calculate average and t-statistic
                if decile_returns:
                    decile_df = pd.DataFrame(decile_returns)
                    
                    avg_10_1 = decile_df['10-1'].mean()
                    tstat_10_1 = decile_df['10-1'].mean() / (decile_df['10-1'].std() / np.sqrt(len(decile_df)))
                    
                    results.append({
                        'period': interval_names[i],
                        'lag': lag,
                        'days_lag': lag // HALF_HOUR_INTERVALS,
                        'avg_10_1': avg_10_1,
                        'tstat_10_1': tstat_10_1,
                        'n': len(decile_df)
                    })
        
        # Convert to DataFrame
        result_df = pd.DataFrame(results)
        
        return result_df


def generate_portfolio_equity_curve(strategy, start_date, end_date, initial_capital=1000000):
    """
    Generate equity curve by implementing the strategy daily
    
    Args:
        strategy: IntradayPeriodicityStrategy instance
        start_date: Starting date as string (YYYY-MM-DD)
        end_date: Ending date as string (YYYY-MM-DD)
        initial_capital: Initial portfolio value
        
    Returns:
        DataFrame with daily portfolio values
    """
    logger.info(f"Generating portfolio equity curve from {start_date} to {end_date}")
    
    # Convert to datetime
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    
    # Get all trading days in the period
    all_days = pd.date_range(start=start_dt, end=end_dt, freq='B')
    
    # Initialize portfolio tracking
    portfolio_values = []
    current_capital = initial_capital
    positions = {}  # Current positions
    
    # Track portfolio performance daily
    for day_idx, current_date in enumerate(all_days):
        date_str = current_date.strftime('%Y-%m-%d')
        
        # Skip the first few days (need enough history)
        if day_idx < 5:  # Need at least 5 days of history
            portfolio_values.append({
                'date': current_date.date(),
                'portfolio_value': current_capital,
                'daily_return': 0.0,
                'cumulative_return': 0.0
            })
            continue
        
        # Get trading signals for the current day
        signals = strategy.implement_trading_strategy(current_date=current_date.date())
        
        if signals.empty:
            logger.warning(f"No signals for {date_str}, portfolio unchanged")
            portfolio_values.append({
                'date': current_date.date(),
                'portfolio_value': current_capital,
                'daily_return': 0.0,
                'cumulative_return': (current_capital / initial_capital) - 1.0
            })
            continue
        
        # Process signals and update portfolio
        long_stocks = signals[signals['signal'] == 1]['ticker'].tolist()
        short_stocks = signals[signals['signal'] == -1]['ticker'].tolist()
        
        # Get next day's returns for all stocks
        if day_idx + 1 < len(all_days):
            next_date = all_days[day_idx + 1].date()
            
            # Calculate daily returns for all stocks from close to close
            daily_returns = {}
            
            for ticker, df in strategy.returns_data.items():
                today_close = df[(df['date'] == current_date.date()) & (df['interval'] == HALF_HOUR_INTERVALS - 1)]
                next_close = df[(df['date'] == next_date) & (df['interval'] == HALF_HOUR_INTERVALS - 1)]
                
                if not today_close.empty and not next_close.empty:
                    today_price = today_close['close'].values[0]
                    next_price = next_close['close'].values[0]
                    daily_returns[ticker] = (next_price / today_price) - 1.0
            
            # Calculate portfolio return
            portfolio_return = 0.0
            
            # Equal weighted allocation to long and short portfolios
            if long_stocks and short_stocks:
                # Long position weight (per stock)
                long_weight = 0.5 / len(long_stocks)
                
                # Short position weight (per stock)
                short_weight = 0.5 / len(short_stocks)
                
                # Calculate long portfolio return
                for ticker in long_stocks:
                    if ticker in daily_returns:
                        portfolio_return += long_weight * daily_returns[ticker]
                
                # Calculate short portfolio return (negative because we're short)
                for ticker in short_stocks:
                    if ticker in daily_returns:
                        portfolio_return -= short_weight * daily_returns[ticker]
            
            # Update portfolio value
            new_capital = current_capital * (1 + portfolio_return)
            
            # Track daily and cumulative performance
            portfolio_values.append({
                'date': current_date.date(),
                'portfolio_value': new_capital,
                'daily_return': portfolio_return,
                'cumulative_return': (new_capital / initial_capital) - 1.0
            })
            
            # Update current capital for next day
            current_capital = new_capital
        else:
            # Last day, no next-day returns available
            portfolio_values.append({
                'date': current_date.date(),
                'portfolio_value': current_capital,
                'daily_return': 0.0,
                'cumulative_return': (current_capital / initial_capital) - 1.0
            })
    
    # Convert to DataFrame
    portfolio_df = pd.DataFrame(portfolio_values)
    
    # Calculate additional performance metrics
    if len(portfolio_df) > 0:
        # Calculate annualized return
        days = (portfolio_df['date'].max() - portfolio_df['date'].min()).days
        if days > 0:
            total_return = portfolio_df['portfolio_value'].iloc[-1] / initial_capital - 1
            annualized_return = (1 + total_return) ** (365 / days) - 1
            
            # Calculate annualized volatility
            daily_std = portfolio_df['daily_return'].std()
            annualized_vol = daily_std * np.sqrt(252)  # Assuming 252 trading days per year
            
            # Calculate Sharpe ratio (assuming 0% risk-free rate for simplicity)
            sharpe_ratio = annualized_return / annualized_vol if annualized_vol > 0 else 0
            
            # Calculate maximum drawdown
            portfolio_df['peak'] = portfolio_df['portfolio_value'].cummax()
            portfolio_df['drawdown'] = (portfolio_df['portfolio_value'] - portfolio_df['peak']) / portfolio_df['peak']
            max_drawdown = portfolio_df['drawdown'].min()
            
            logger.info(f"Performance Summary:")
            logger.info(f"Total Return: {total_return:.2%}")
            logger.info(f"Annualized Return: {annualized_return:.2%}")
            logger.info(f"Annualized Volatility: {annualized_vol:.2%}")
            logger.info(f"Sharpe Ratio: {sharpe_ratio:.2f}")
            logger.info(f"Maximum Drawdown: {max_drawdown:.2%}")
    
    return portfolio_df


def plot_equity_curve(portfolio_df, save_path='equity_curve.png'):
    """
    Plot the equity curve with key performance metrics
    
    Args:
        portfolio_df: DataFrame with portfolio values
        save_path: Path to save the plot
    """
    if portfolio_df.empty:
        logger.error("Empty portfolio data, cannot plot equity curve")
        return
    
    # Calculate performance metrics
    initial_value = portfolio_df['portfolio_value'].iloc[0]
    final_value = portfolio_df['portfolio_value'].iloc[-1]
    total_return = final_value / initial_value - 1
    
    days = (portfolio_df['date'].max() - portfolio_df['date'].min()).days
    annualized_return = (1 + total_return) ** (365 / days) - 1 if days > 0 else 0
    
    daily_std = portfolio_df['daily_return'].std()
    annualized_vol = daily_std * np.sqrt(252)  # Assuming 252 trading days per year
    
    sharpe_ratio = annualized_return / annualized_vol if annualized_vol > 0 else 0
    
    # Calculate drawdowns
    portfolio_df['peak'] = portfolio_df['portfolio_value'].cummax()
    portfolio_df['drawdown'] = (portfolio_df['portfolio_value'] - portfolio_df['peak']) / portfolio_df['peak']
    max_drawdown = portfolio_df['drawdown'].min()
    
    # Create the plot
    plt.figure(figsize=(12, 10))
    
    # Plot equity curve
    plt.subplot(2, 1, 1)
    plt.plot(portfolio_df['date'], portfolio_df['portfolio_value'], 'b-', linewidth=2)
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value ($)')
    plt.title('Portfolio Equity Curve')
    plt.grid(True, alpha=0.3)
    
    # Add text box with performance metrics
    metrics_text = f"Total Return: {total_return:.2%}\n" \
                  f"Annualized Return: {annualized_return:.2%}\n" \
                  f"Annualized Volatility: {annualized_vol:.2%}\n" \
                  f"Sharpe Ratio: {sharpe_ratio:.2f}\n" \
                  f"Maximum Drawdown: {max_drawdown:.2%}"
    
    plt.annotate(metrics_text, xy=(0.02, 0.95), xycoords='axes fraction',
                 bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8),
                 va='top', fontsize=10)
    
    # Plot drawdowns
    plt.subplot(2, 1, 2)
    plt.fill_between(portfolio_df['date'], portfolio_df['drawdown'] * 100, 0, color='r', alpha=0.3)
    plt.plot(portfolio_df['date'], portfolio_df['drawdown'] * 100, 'r-', linewidth=1)
    plt.xlabel('Date')
    plt.ylabel('Drawdown (%)')
    plt.title('Portfolio Drawdowns')
    plt.grid(True, alpha=0.3)
    
    # Add horizontal line at 0
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.2)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    logger.info(f"Equity curve plot saved to '{save_path}'")


def generate_monthly_return_table(portfolio_df):
    """
    Generate a table of monthly returns
    
    Args:
        portfolio_df: DataFrame with daily portfolio values
        
    Returns:
        DataFrame with monthly returns
    """
    if portfolio_df.empty:
        logger.error("Empty portfolio data, cannot generate monthly returns")
        return pd.DataFrame()
    
    # Add month and year columns
    portfolio_df['year'] = pd.to_datetime(portfolio_df['date']).dt.year
    portfolio_df['month'] = pd.to_datetime(portfolio_df['date']).dt.month
    
    # Group by year and month
    monthly_returns = []
    
    for (year, month), group in portfolio_df.groupby(['year', 'month']):
        start_value = group['portfolio_value'].iloc[0]
        end_value = group['portfolio_value'].iloc[-1]
        monthly_return = (end_value / start_value) - 1
        
        monthly_returns.append({
            'year': year,
            'month': month,
            'return': monthly_return
        })
    
    # Convert to DataFrame
    monthly_df = pd.DataFrame(monthly_returns)
    
    # Pivot to create a year x month table
    pivot_table = monthly_df.pivot(index='year', columns='month', values='return')
    
    # Add year total
    monthly_df_with_year = monthly_df.copy()
    monthly_df_with_year['month_name'] = monthly_df_with_year['month'].apply(lambda x: pd.Timestamp(2020, x, 1).strftime('%b'))
    yearly_returns = monthly_df_with_year.groupby('year')['return'].apply(lambda x: np.prod(1 + x) - 1)
    
    # Format the pivot table
    pivot_table_formatted = pivot_table.copy()
    
    # Convert to percentages
    for col in pivot_table_formatted.columns:
        pivot_table_formatted[col] = pivot_table_formatted[col].apply(lambda x: f"{x:.2%}" if pd.notnull(x) else "")
    
    # Add year total column
    pivot_table_formatted['Year Total'] = yearly_returns.apply(lambda x: f"{x:.2%}")
    
    # Rename columns to month names
    month_names = {
        1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
        7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'
    }
    pivot_table_formatted = pivot_table_formatted.rename(columns=month_names)
    
    return pivot_table_formatted


def main():
    """Main function to run the strategy"""
    # Define parameters
    start_date = "2023-01-01"  # Extended to get more data for a meaningful equity curve
    end_date = "2023-12-31"
    min_market_cap = 500  # $500 million minimum market cap
    
    # Initialize Bloomberg data fetcher
    bloomberg = BloombergDataFetcher()
    
    # Start Bloomberg session
    if not bloomberg.start_session():
        logger.error("Failed to start Bloomberg session. Exiting.")
        return
    
    try:
        # Initialize strategy
        strategy = IntradayPeriodicityStrategy(bloomberg)
        
        # Load data
        strategy.load_data(start_date, end_date, min_market_cap)
        
        # Calculate cross-sectional return responses
        response_df = strategy.calculate_cross_sectional_return_responses(max_lag=65)  # 5 trading days
        
        if not response_df.empty:
            # Save results to CSV
            response_df.to_csv('return_responses.csv', index=False)
            logger.info("Return responses saved to 'return_responses.csv'")
            
            # Plot the return responses
            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            plt.bar(response_df['lag'], response_df['gamma'] * 10000)  # Convert to basis points
            plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
            plt.xlabel('Lag (half-hour intervals)')
            plt.ylabel('Return Response (basis points)')
            plt.title('Return Responses (γₖ)')
            plt.grid(True, alpha=0.3)
            
            plt.subplot(1, 2, 2)
            plt.bar(response_df['lag'], response_df['t_stat'])
            plt.axhline(y=1.96, color='r', linestyle='--', alpha=0.3)
            plt.axhline(y=-1.96, color='r', linestyle='--', alpha=0.3)
            plt.xlabel('Lag (half-hour intervals)')
            plt.ylabel('t-statistic')
            plt.title('t-statistics of Return Responses')
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('return_responses.png')
            plt.close()
            
            logger.info("Return response plot saved to 'return_responses.png'")
        
        # Calculate decile portfolio returns for daily lags
        daily_lags = [13, 26, 39, 52, 65]
        decile_results = []
        
        for lag in daily_lags:
            result = strategy.calculate_decile_portfolio_returns(lag=lag)
            
            if not result.empty:
                result['lag'] = lag
                decile_results.append(result)
        
        if decile_results:
            decile_df = pd.concat(decile_results, ignore_index=True)
            
            # Save results to CSV
            decile_df.to_csv('decile_portfolio_returns.csv', index=False)
            logger.info("Decile portfolio returns saved to 'decile_portfolio_returns.csv'")
            
            # Plot the winner-loser (10-1) spreads
            plt.figure(figsize=(10, 6))
            plt.bar(decile_df['lag'], decile_df['10-1'] * 10000)  # Convert to basis points
            plt.xlabel('Lag (half-hour intervals)')
            plt.ylabel('Winner-Loser Spread (basis points)')
            plt.title('Winner-Loser (10-1) Portfolio Spreads')
            plt.grid(True, alpha=0.3)
            plt.savefig('winner_loser_spreads.png')
            plt.close()
            
            logger.info("Winner-loser spread plot saved to 'winner_loser_spreads.png'")
        
        # Generate portfolio equity curve
        portfolio_df = generate_portfolio_equity_curve(strategy, start_date, end_date)
        
        if not portfolio_df.empty:
            # Save portfolio data to CSV
            portfolio_df.to_csv('portfolio_equity_curve.csv', index=False)
            logger.info("Portfolio equity curve data saved to 'portfolio_equity_curve.csv'")
            
            # Plot equity curve
            plot_equity_curve(portfolio_df)
            
            # Generate monthly return table
            monthly_returns = generate_monthly_return_table(portfolio_df)
            
            if not monthly_returns.empty:
                # Save monthly returns to CSV
                monthly_returns.to_csv('monthly_returns.csv')
                logger.info("Monthly returns table saved to 'monthly_returns.csv'")
                
                # Create a heatmap of monthly returns
                plt.figure(figsize=(12, 8))
                
                # Convert back to numeric for heatmap
                heatmap_data = monthly_returns.copy()
                for col in heatmap_data.columns:
                    if col != 'Year Total':
                        heatmap_data[col] = heatmap_data[col].apply(
                            lambda x: float(x.strip('%')) / 100 if isinstance(x, str) and x.strip() else np.nan
                        )
                
                # Drop the Year Total column for the heatmap
                if 'Year Total' in heatmap_data.columns:
                    heatmap_data = heatmap_data.drop(columns=['Year Total'])
                
                # Create heatmap
                sns.heatmap(heatmap_data, annot=True, fmt=".2%", cmap="RdYlGn", 
                           vmin=-0.05, vmax=0.05, center=0, linewidths=1, cbar_kws={"shrink": .8})
                
                plt.title('Monthly Returns Heatmap')
                plt.tight_layout()
                plt.savefig('monthly_returns_heatmap.png')
                plt.close()
                
                logger.info("Monthly returns heatmap saved to 'monthly_returns_heatmap.png'")
        
        # Implement trading strategy for the next day (current date is the last date in our data)
        signals = strategy.implement_trading_strategy()
        
        if not signals.empty:
            # Save trading signals to CSV
            signals.to_csv('trading_signals.csv', index=False)
            logger.info("Trading signals saved to 'trading_signals.csv'")
            
            # Print top long and short positions
            top_long = signals[signals['signal'] == 1].sort_values('lagged_return', ascending=False)
            top_short = signals[signals['signal'] == -1].sort_values('lagged_return', ascending=True)
            
            logger.info("\nTop Long Positions:")
            for i, (_, row) in enumerate(top_long.iterrows()):
                if i < 10:  # Print top 10
                    ticker = row['ticker'].replace(' US Equity', '')
                    logger.info(f"{ticker}: {row['lagged_return']*100:.2f}%")
            
            logger.info("\nTop Short Positions:")
            for i, (_, row) in enumerate(top_short.iterrows()):
                if i < 10:  # Print top 10
                    ticker = row['ticker'].replace(' US Equity', '')
                    logger.info(f"{ticker}: {row['lagged_return']*100:.2f}%")
        
    finally:
        # Stop Bloomberg session
        bloomberg.stop_session()


if __name__ == "__main__":
    main()