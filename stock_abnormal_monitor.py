import blpapi
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import logging
import os
import argparse
import time
import re
from typing import List, Dict, Any, Optional
from scipy import stats
from tabulate import tabulate

# ANSI color codes for terminal output
RED = "\033[91m"        # Bright red
GREEN = "\033[92m"      # Bright green
YELLOW = "\033[93m"     # Bright yellow
BLUE = "\033[94m"       # Bright blue
MAGENTA = "\033[95m"    # Bright magenta
BOLD = "\033[1m"        # Bold
RESET = "\033[0m"       # Reset to default

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("volume_monitor.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('VolumeMonitor')

class VolumeMonitor:
    """
    A class to monitor and detect abnormal trading volume in stocks across markets.
    """
    
    def __init__(self, host: str = '127.0.0.1', port: int = 8194, use_sample_data: bool = False, market: str = 'HK'):
        """
        Initialize the Bloomberg connection.
        
        Args:
            host: Bloomberg server host (default: localhost)
            port: Bloomberg server port (default: 8194)
            use_sample_data: Force use of sample data even if Bloomberg is available
            market: Market to monitor (default: HK)
        """
        self.host = host
        self.port = port
        self.session = None
        self.use_sample_data = use_sample_data
        self.market = market.upper()  # Market code (HK, US, JP, etc.)
        
        # Default monitoring parameters
        self.lookback_period = 20  # Trading days to establish baseline volume
        self.detection_methods = ['zscore', 'percent_change', 'historical_percentile']
        self.zscore_threshold = 2.5
        self.percent_change_threshold = 200  # 200% increase from average
        self.percentile_threshold = 95
        
        # For storing results
        self.volume_data = {}
        self.latest_update_time = None
        
        # Market-specific currency formatting
        self.currency_symbols = {
            'HK': 'HK$',
            'US': '$',
            'JP': '¥',
            'UK': '£',
            'EU': '€',
            'CN': '¥',
            'TW': 'NT$',
            'KR': '₩',
            'SG': 'S$',
            'AU': 'A$'
        }
        
        # Get the currency symbol for the selected market
        self.currency_symbol = self.currency_symbols.get(self.market, '$')
    
    def start_session(self) -> bool:
        """
        Start a Bloomberg session.
        
        Returns:
            bool: True if successful, False otherwise
        """
        if self.use_sample_data:
            logger.info("Using sample data mode, no Bloomberg connection required")
            return True
            
        try:
            # Initialize session options
            session_options = blpapi.SessionOptions()
            session_options.setServerHost(self.host)
            session_options.setServerPort(self.port)
            
            logger.info(f"Connecting to Bloomberg at {self.host}:{self.port}")
            
            # Create a Session
            self.session = blpapi.Session(session_options)
            
            # Start the session
            if not self.session.start():
                logger.error("Failed to start Bloomberg session.")
                return False
            
            # Open the market data service
            if not self.session.openService("//blp/refdata"):
                logger.error("Failed to open //blp/refdata service")
                return False
            
            logger.info("Bloomberg session started successfully")
            return True
            
        except blpapi.Exception as e:
            logger.error(f"Bloomberg API Exception: {e}")
            return False
            
    def stop_session(self) -> None:
        """Stop the Bloomberg session."""
        if self.session and not self.use_sample_data:
            self.session.stop()
            logger.info("Bloomberg session stopped")
    
    def read_stock_list_from_file(self, filename: str) -> List[str]:
        """
        Read a list of stock tickers from a file.
        
        Args:
            filename: Path to the file containing stock tickers
            
        Returns:
            List of stock tickers
        """
        try:
            if not os.path.exists(filename):
                logger.error(f"Stock list file not found: {filename}")
                return []
                
            # Read the file
            with open(filename, 'r') as f:
                lines = f.readlines()
            
            # Process each line to extract the ticker
            tickers = []
            for line in lines:
                # Skip empty lines and comments (lines starting with #)
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                # Split by comma or whitespace if the file has additional data
                parts = re.split(r'[,\s]+', line)
                ticker = parts[0].strip()
                
                # Ensure the ticker has the appropriate suffix based on market
                if self.market == 'HK' and not ticker.endswith('HK Equity'):
                    ticker = f"{ticker} HK Equity"
                elif self.market == 'US' and not ticker.endswith('US Equity'):
                    ticker = f"{ticker} US Equity"
                elif self.market == 'JP' and not ticker.endswith('JT Equity'):
                    ticker = f"{ticker} JT Equity"
                # Add other markets as needed
                
                # Add to list if not already present
                if ticker not in tickers:
                    tickers.append(ticker)
            
            logger.info(f"Read {len(tickers)} tickers from file: {filename}")
            return tickers
            
        except Exception as e:
            logger.error(f"Error reading stock list from file: {e}")
            return []
    
    def get_stocks_to_monitor(self, stock_list_file: Optional[str] = None) -> List[str]:
        """
        Get a list of stocks to monitor.
        
        Args:
            stock_list_file: Path to file containing stock tickers (optional)
            
        Returns:
            List of stock tickers
        """
        # If a file is specified and it exists, read from file
        if stock_list_file and os.path.exists(stock_list_file):
            stocks = self.read_stock_list_from_file(stock_list_file)
            if stocks:
                return stocks
        
        # If no file is specified or the file doesn't exist or is empty, use sample data
        logger.warning(f"Using sample stock list for {self.market} market")
        return self.get_sample_stock_list()
    
    def get_sample_stock_list(self) -> List[str]:
        """
        Get a sample list of stocks for testing.
        
        Returns:
            List of sample stock tickers
        """
        if self.market == 'HK':
            # Common Hong Kong blue chips and tech stocks
            return [
                '1 HK Equity',    # CK Hutchison
                '2 HK Equity',    # CLP Holdings
                '3 HK Equity',    # Hong Kong and China Gas
                '5 HK Equity',    # HSBC Holdings
                '11 HK Equity',   # Hang Seng Bank
                '12 HK Equity',   # Henderson Land
                '16 HK Equity',   # Sun Hung Kai Properties
                '17 HK Equity',   # New World Development
                '27 HK Equity',   # Galaxy Entertainment
                '66 HK Equity',   # MTR Corporation
                '101 HK Equity',  # Hang Lung Properties
                '388 HK Equity',  # HKEX
                '700 HK Equity',  # Tencent
                '823 HK Equity',  # Link REIT
                '941 HK Equity',  # China Mobile
                '1038 HK Equity', # CKI Holdings
                '939 HK Equity',  # China Construction Bank
                '1288 HK Equity', # Agricultural Bank of China
                '1398 HK Equity', # ICBC
                '2318 HK Equity', # Ping An Insurance
                '9618 HK Equity', # JD.com
                '9988 HK Equity', # Alibaba
                '9999 HK Equity', # NetEase
                '1024 HK Equity', # Kuaishou
                '2269 HK Equity', # WuXi Biologics
                '1810 HK Equity'  # Xiaomi
            ]
        elif self.market == 'US':
            # US blue chips and tech stocks
            return [
                'AAPL US Equity',  # Apple
                'MSFT US Equity',  # Microsoft
                'AMZN US Equity',  # Amazon
                'GOOGL US Equity', # Alphabet
                'META US Equity',  # Meta
                'TSLA US Equity',  # Tesla
                'NVDA US Equity',  # NVIDIA
                'JPM US Equity',   # JPMorgan Chase
                'V US Equity',     # Visa
                'WMT US Equity',   # Walmart
                'JNJ US Equity',   # Johnson & Johnson
                'PG US Equity',    # Procter & Gamble
                'XOM US Equity',   # ExxonMobil
                'BAC US Equity',   # Bank of America
                'DIS US Equity',   # Disney
                'ADBE US Equity',  # Adobe
                'NFLX US Equity',  # Netflix
                'INTC US Equity',  # Intel
                'CSCO US Equity',  # Cisco
                'CMCSA US Equity'  # Comcast
            ]
        elif self.market == 'JP':
            # Japanese blue chips
            return [
                '7203 JT Equity',  # Toyota Motor
                '9432 JT Equity',  # NTT
                '9984 JT Equity',  # SoftBank Group
                '6758 JT Equity',  # Sony Group
                '6861 JT Equity',  # Keyence
                '7974 JT Equity',  # Nintendo
                '4502 JT Equity',  # Takeda Pharmaceutical
                '8306 JT Equity',  # Mitsubishi UFJ Financial
                '9433 JT Equity',  # KDDI
                '6501 JT Equity',  # Hitachi
                '6367 JT Equity',  # Daikin Industries
                '6954 JT Equity',  # Fanuc
                '4063 JT Equity',  # Shin-Etsu Chemical
                '8316 JT Equity',  # Sumitomo Mitsui Financial
                '9434 JT Equity',  # SoftBank Corp
                '7267 JT Equity',  # Honda Motor
                '6902 JT Equity',  # Denso
                '6098 JT Equity',  # Recruit Holdings
                '4568 JT Equity',  # Daiichi Sankyo
                '7751 JT Equity'   # Canon
            ]
        # Add more markets as needed
        else:
            logger.warning(f"No sample data for market {self.market}, using generic list")
            return [f"STOCK{i} {self.market} Equity" for i in range(1, 21)]
    
    def get_historical_prices_and_volumes(self, tickers: List[str], lookback_days: int = 60) -> pd.DataFrame:
        """
        Get historical prices and volumes for a list of securities.
        
        Args:
            tickers: List of Bloomberg tickers
            lookback_days: Number of calendar days to look back
            
        Returns:
            DataFrame with historical prices and volumes
        """
        # Return empty DataFrame with correct structure if no tickers
        if not tickers:
            logger.warning("No tickers provided for historical data")
            # Return empty DataFrame with the right columns
            return pd.DataFrame(columns=['ticker', 'date', 'price', 'volume'])
            
        if self.use_sample_data:
            return self.get_sample_historical_data(tickers, lookback_days)
        
        # Calculate start and end dates
        end_date = datetime.datetime.now()
        start_date = end_date - datetime.timedelta(days=lookback_days)
        
        try:
            refDataService = self.session.getService("//blp/refdata")
            request = refDataService.createRequest("HistoricalDataRequest")
            
            # Set date range
            request.set("startDate", start_date.strftime("%Y%m%d"))
            request.set("endDate", end_date.strftime("%Y%m%d"))
            
            # Add securities (in chunks if list is large)
            chunk_size = 50
            all_data = []
            
            for i in range(0, len(tickers), chunk_size):
                chunk_tickers = tickers[i:i+chunk_size]
                
                # Create a new request for each chunk
                chunk_request = refDataService.createRequest("HistoricalDataRequest")
                chunk_request.set("startDate", start_date.strftime("%Y%m%d"))
                chunk_request.set("endDate", end_date.strftime("%Y%m%d"))
                
                # Add securities for this chunk
                for ticker in chunk_tickers:
                    chunk_request.append("securities", ticker)
                
                # Add fields
                chunk_request.append("fields", "PX_LAST")      # Closing price
                chunk_request.append("fields", "PX_VOLUME")    # Trading volume
                
                # Send the request
                logger.info(f"Requesting historical data for {len(chunk_tickers)} tickers (chunk {i//chunk_size + 1})")
                self.session.sendRequest(chunk_request)
                
                # Process the response
                chunk_data = []
                
                while True:
                    event = self.session.nextEvent(500)
                    
                    for msg in event:
                        # Check against the message type name
                        if msg.messageType() == blpapi.Name("HistoricalDataResponse"):
                            # Check for security data
                            security_data = msg.getElement("securityData")
                            ticker = security_data.getElementAsString("security")
                            
                            # Get the field data
                            field_data = security_data.getElement("fieldData")
                            
                            # Extract values for each date
                            for i in range(field_data.numValues()):
                                date_data = field_data.getValue(i)
                                date = date_data.getElementAsDatetime("date").strftime("%Y-%m-%d")
                                
                                # Extract price and volume
                                px_last = None
                                px_volume = None
                                
                                if date_data.hasElement("PX_LAST"):
                                    px_last = date_data.getElementAsFloat("PX_LAST")
                                
                                if date_data.hasElement("PX_VOLUME"):
                                    px_volume = date_data.getElementAsFloat("PX_VOLUME")
                                
                                # Add to results
                                chunk_data.append({
                                    'ticker': ticker,
                                    'date': date,
                                    'price': px_last,
                                    'volume': px_volume
                                })
                    
                    if event.eventType() == blpapi.Event.RESPONSE:
                        break
                
                all_data.extend(chunk_data)
                time.sleep(0.1)  # Throttle requests
            
            # Convert to DataFrame
            df = pd.DataFrame(all_data)
            
            # Check if we have any data
            if df.empty:
                logger.warning("No historical data retrieved, using sample data")
                return self.get_sample_historical_data(tickers, lookback_days)
                
            df['date'] = pd.to_datetime(df['date'])
            
            # Sort by ticker and date
            df = df.sort_values(['ticker', 'date'])
            
            logger.info(f"Retrieved {len(df)} historical data points for {len(tickers)} tickers")
            return df
            
        except Exception as e:
            logger.error(f"Error retrieving historical data: {e}")
            return self.get_sample_historical_data(tickers, lookback_days)
    
    def get_sample_historical_data(self, tickers: List[str], lookback_days: int = 60) -> pd.DataFrame:
        """
        Generate sample historical price and volume data for testing.
        
        Args:
            tickers: List of tickers to generate data for
            lookback_days: Number of calendar days to look back
            
        Returns:
            DataFrame with sample historical data
        """
        logger.info(f"Generating sample historical data for {len(tickers)} tickers")
        
        # If no tickers, return empty DataFrame with the right columns
        if not tickers:
            return pd.DataFrame(columns=['ticker', 'date', 'price', 'volume'])
        
        # Generate date range (business days only)
        end_date = pd.Timestamp.now().normalize()
        start_date = end_date - pd.Timedelta(days=lookback_days)
        
        # Create business day index
        dates = pd.date_range(start=start_date, end=end_date, freq='B')
        
        all_data = []
        
        # Generate random data for each ticker
        for ticker in tickers:
            # Extract numeric part of ticker for seed - handle non-numeric tickers
            ticker_code = ticker.split()[0]
            ticker_num_str = ''.join(filter(str.isdigit, ticker_code[:10]))
            ticker_num = int(ticker_num_str) if ticker_num_str else hash(ticker_code) % 10000
            np.random.seed(ticker_num)  # Set seed based on ticker for consistent results
            
            # Base price (realistic for the market)
            if self.market == 'HK':
                base_price = np.random.uniform(5, 100)
            elif self.market == 'US':
                base_price = np.random.uniform(20, 200)
            elif self.market == 'JP':
                base_price = np.random.uniform(1000, 10000)
            else:
                base_price = np.random.uniform(10, 100)
            
            # Base volume (depends on stock popularity)
            base_volume = np.random.uniform(1e6, 1e7)
            
            # Generate price series with some trend and volatility
            price_changes = np.random.normal(0, 0.015, len(dates))  # Daily returns
            price_trend = np.cumprod(1 + price_changes) * base_price
            
            # Generate volume series with some autocorrelation
            volume_noise = np.random.lognormal(0, 0.5, len(dates))
            volume = base_volume * volume_noise
            
            # Add some volume spikes
            spike_dates = np.random.choice(range(len(dates)), size=int(len(dates)*0.1), replace=False)
            for spike_idx in spike_dates:
                volume[spike_idx] *= np.random.uniform(2, 5)
            
            # Add the data points
            for i, date in enumerate(dates):
                all_data.append({
                    'ticker': ticker,
                    'date': date,
                    'price': price_trend[i],
                    'volume': volume[i]
                })
        
        # Create DataFrame
        df = pd.DataFrame(all_data)
        
        # Ensure correct data types
        df['date'] = pd.to_datetime(df['date'])
        df['price'] = df['price'].astype(float)
        df['volume'] = df['volume'].astype(float)
        
        # Sort by ticker and date
        df = df.sort_values(['ticker', 'date'])
        
        logger.info(f"Generated {len(df)} sample data points")
        return df
    
    def detect_abnormal_volume(self, 
                              stock_list_file: Optional[str] = None,
                              lookback_period: int = 20, 
                              zscore_threshold: float = 2.5,
                              percent_change_threshold: float = 200,
                              percentile_threshold: float = 95) -> Dict[str, Any]:
        """
        Detect abnormal trading volume in stocks.
        
        Args:
            stock_list_file: Path to file containing stock tickers
            lookback_period: Trading days to establish baseline volume
            zscore_threshold: Z-score threshold for abnormal volume
            percent_change_threshold: Percent increase threshold for abnormal volume
            percentile_threshold: Historical percentile threshold for abnormal volume
            
        Returns:
            Dictionary with abnormal volume data and analysis
        """
        # Update parameters
        self.lookback_period = lookback_period
        self.zscore_threshold = zscore_threshold
        self.percent_change_threshold = percent_change_threshold
        self.percentile_threshold = percentile_threshold
        
        # Get list of stocks
        logger.info(f"Getting stock list for {self.market} market")
        stocks = self.get_stocks_to_monitor(stock_list_file)
        
        # Check if we have any stocks
        if not stocks:
            logger.warning("No stocks to analyze. Using sample data.")
            stocks = self.get_sample_stock_list()
        
        # Get historical price and volume data
        logger.info(f"Getting historical price and volume data for {len(stocks)} stocks")
        # Looking back longer than the lookback period to have enough history for percentile calculations
        history_days = lookback_period * 5  # Approx 100 trading days
        hist_data = self.get_historical_prices_and_volumes(stocks, lookback_days=history_days)
        
        # Set latest update time
        self.latest_update_time = datetime.datetime.now()
        
        # Check if we have historical data
        if hist_data.empty:
            logger.warning("No historical data available for analysis")
            # Return empty results with the current timestamp
            return {
                'Update_Time': self.latest_update_time,
                'Abnormal_Volume': [],
                'Lookback_Period': lookback_period,
                'Total_Stocks': len(stocks),
                'Abnormal_Count': 0,
                'Market': self.market
            }
        
        # Convert to a more usable format (pivot by ticker and date)
        pivot_volume = hist_data.pivot(index='date', columns='ticker', values='volume')
        pivot_price = hist_data.pivot(index='date', columns='ticker', values='price')
        
        # Get the latest date in the data
        latest_date = pivot_volume.index.max()
        
        # Calculate the lookback window start
        # Ensure we have enough data for the lookback period
        if len(pivot_volume) <= lookback_period:
            lookback_start = pivot_volume.index[0]
        else:
            lookback_start = pivot_volume.index[-lookback_period-1]  # -1 to exclude the latest day
        
        # Initialize results
        abnormal_results = []
        
        # Process each stock
        for ticker in pivot_volume.columns:
            ticker_volume = pivot_volume[ticker].dropna()
            
            if len(ticker_volume) < 5:  # Need at least 5 days of data
                continue
                
            # Get the latest volume
            latest_volume = ticker_volume.iloc[-1]
            
            # Get the lookback window volume
            lookback_volume = ticker_volume.loc[lookback_start:].iloc[:-1]  # Exclude the latest day
            
            if len(lookback_volume) < 3:  # Need at least 3 days for baseline
                continue
            
            # Calculate baseline statistics
            baseline_mean = lookback_volume.mean()
            baseline_std = lookback_volume.std()
            
            # Calculate z-score
            zscore = (latest_volume - baseline_mean) / baseline_std if baseline_std > 0 else 0
            
            # Calculate percent change from baseline mean
            percent_change = ((latest_volume / baseline_mean) - 1) * 100 if baseline_mean > 0 else 0
            
            # Calculate historical percentile
            percentile = stats.percentileofscore(ticker_volume.iloc[:-1], latest_volume)
            
            # Get current price
            current_price = pivot_price[ticker].iloc[-1] if ticker in pivot_price and len(pivot_price[ticker]) > 0 else None
            
            # Calculate price change (1-day)
            prev_price = pivot_price[ticker].iloc[-2] if ticker in pivot_price and len(pivot_price[ticker]) >= 2 else None
            price_change_1d = ((current_price / prev_price) - 1) * 100 if current_price and prev_price else None
            
            # Flag abnormal volume based on different methods
            abnormal = False
            detection_method = []
            
            if zscore > zscore_threshold:
                abnormal = True
                detection_method.append('zscore')
            
            if percent_change > percent_change_threshold:
                abnormal = True
                detection_method.append('percent_change')
            
            if percentile > percentile_threshold:
                abnormal = True
                detection_method.append('historical_percentile')
            
            if abnormal:
                abnormal_results.append({
                    'ticker': ticker,
                    'date': latest_date,
                    'volume': latest_volume,
                    'baseline_volume': baseline_mean,
                    'zscore': zscore,
                    'percent_change': percent_change,
                    'percentile': percentile,
                    'price': current_price,
                    'price_change_1d': price_change_1d,
                    'detection_method': detection_method
                })
        
        # Sort by percent change in volume
        abnormal_results = sorted(abnormal_results, key=lambda x: x['percent_change'], reverse=True)
        
        # Store the results
        self.volume_data = {
            'Update_Time': self.latest_update_time,
            'Abnormal_Volume': abnormal_results,
            'Lookback_Period': lookback_period,
            'Total_Stocks': len(stocks),
            'Abnormal_Count': len(abnormal_results),
            'Market': self.market
        }
        
        return self.volume_data
    
    def get_additional_stock_info(self, tickers: List[str]) -> pd.DataFrame:
        """
        Get additional information for stocks with abnormal volume.
        
        Args:
            tickers: List of tickers to get info for
            
        Returns:
            DataFrame with additional stock information
        """
        # Return empty DataFrame with correct structure if no tickers
        if not tickers:
            logger.warning("No tickers provided for additional info")
            return pd.DataFrame(columns=['ticker', 'NAME', 'INDUSTRY_GROUP', 'INDUSTRY_SECTOR', 
                                        'SECURITY_TYP', 'EQY_SH_OUT', 'CUR_MKT_CAP', 
                                        'REL_1D_RETURN', 'REL_5D_RETURN', 'REL_1M_RETURN', 
                                        'REL_3M_RETURN', 'BEST_TARGET_PRICE', 'BEST_ANALYST_RATING_MEAN'])
            
        if self.use_sample_data:
            return self.get_sample_stock_info(tickers)
        
        try:
            # Fields to retrieve
            fields = [
                'NAME',
                'INDUSTRY_GROUP',
                'INDUSTRY_SECTOR',
                'SECURITY_TYP',
                'EQY_SH_OUT',
                'CUR_MKT_CAP',
                'REL_1D_RETURN',
                'REL_5D_RETURN',
                'REL_1M_RETURN',
                'REL_3M_RETURN',
                'BEST_TARGET_PRICE',
                'BEST_ANALYST_RATING_MEAN'
            ]
            
            # Get the data
            refDataService = self.session.getService("//blp/refdata")
            request = refDataService.createRequest("ReferenceDataRequest")
            
            # Add securities
            for ticker in tickers:
                request.append("securities", ticker)
            
            # Add fields
            for field in fields:
                request.append("fields", field)
            
            # Send the request
            logger.info(f"Requesting additional info for {len(tickers)} stocks")
            self.session.sendRequest(request)
            
            # Process the response
            results = []
            
            while True:
                event = self.session.nextEvent(500)
                
                for msg in event:
                    # Check message type by name
                    if msg.messageType() == blpapi.Name("ReferenceDataResponse"):
                        # Check for security data
                        securities_data = msg.getElement("securityData")
                        
                        for i in range(securities_data.numValues()):
                            security_data = securities_data.getValue(i)
                            ticker = security_data.getElementAsString("security")
                            
                            # Get field data
                            data = {'ticker': ticker}
                            
                            if security_data.hasElement("fieldData"):
                                field_data = security_data.getElement("fieldData")
                                
                                # Extract each field
                                for field in fields:
                                    if field_data.hasElement(field):
                                        data[field] = field_data.getElementAsString(field)
                            
                            results.append(data)
                
                if event.eventType() == blpapi.Event.RESPONSE:
                    break
            
            # Convert to DataFrame
            df = pd.DataFrame(results)
            
            # If DataFrame is empty, use sample data
            if df.empty:
                logger.warning("No additional stock info retrieved, using sample data")
                return self.get_sample_stock_info(tickers)
            
            # Clean up the data types
            numeric_fields = ['EQY_SH_OUT', 'CUR_MKT_CAP', 'REL_1D_RETURN', 'REL_5D_RETURN', 
                             'REL_1M_RETURN', 'REL_3M_RETURN', 'BEST_TARGET_PRICE', 'BEST_ANALYST_RATING_MEAN']
            
            for field in numeric_fields:
                if field in df.columns:
                    df[field] = pd.to_numeric(df[field], errors='coerce')
            
            logger.info(f"Retrieved additional info for {len(df)} stocks")
            return df
            
        except Exception as e:
            logger.error(f"Error retrieving additional stock info: {e}")
            return self.get_sample_stock_info(tickers)
    
    def get_sample_stock_info(self, tickers: List[str]) -> pd.DataFrame:
        """
        Generate sample additional stock information for testing.
        
        Args:
            tickers: List of tickers to generate info for
            
        Returns:
            DataFrame with sample stock information
        """
        logger.info(f"Generating sample additional info for {len(tickers)} stocks")
        
        # If no tickers, return empty DataFrame with the right columns
        if not tickers:
            return pd.DataFrame(columns=['ticker', 'NAME', 'INDUSTRY_GROUP', 'INDUSTRY_SECTOR', 
                                        'SECURITY_TYP', 'EQY_SH_OUT', 'CUR_MKT_CAP', 
                                        'REL_1D_RETURN', 'REL_5D_RETURN', 'REL_1M_RETURN', 
                                        'REL_3M_RETURN', 'BEST_TARGET_PRICE', 'BEST_ANALYST_RATING_MEAN'])
        
        # Sample sectors
        sectors = [
            'Technology', 'Finance', 'Property', 'Consumer', 'Healthcare', 
            'Energy', 'Utilities', 'Telecommunications', 'Industrial', 
            'Materials', 'Real Estate', 'Consumer Discretionary', 'Consumer Staples'
        ]
        
        results = []
        
        for ticker in tickers:
            # Extract ticker code
            ticker_code = ticker.split()[0]
            
            # Use ticker to generate consistent random values
            ticker_num_str = ''.join(filter(str.isdigit, ticker_code[:10]))
            ticker_num = int(ticker_num_str) if ticker_num_str else hash(ticker_code) % 10000
            np.random.seed(ticker_num)
            
            # Generate sample data
            name = f"Company {ticker_code}"
            sector = np.random.choice(sectors)
            
            # Generate market cap (in appropriate currency)
            market_cap = np.random.uniform(1e9, 1e11)
            
            # Generate returns
            returns_1d = np.random.normal(0, 2)
            returns_5d = np.random.normal(0, 4)
            returns_1m = np.random.normal(0, 8)
            returns_3m = np.random.normal(0, 15)
            
            # Generate analyst data
            price_base = 100
            if self.market == 'JP':
                price_base = 5000  # Japanese stocks have higher nominal prices
            
            target_price = np.random.uniform(0.8, 1.2) * price_base
            analyst_rating = np.random.uniform(2.5, 4.5)
            
            # Generate data
            results.append({
                'ticker': ticker,
                'NAME': name,
                'INDUSTRY_SECTOR': sector,
                'INDUSTRY_GROUP': sector,
                'SECURITY_TYP': 'Common Stock',
                'EQY_SH_OUT': market_cap / np.random.uniform(5, 50),
                'CUR_MKT_CAP': market_cap,
                'REL_1D_RETURN': returns_1d,
                'REL_5D_RETURN': returns_5d,
                'REL_1M_RETURN': returns_1m,
                'REL_3M_RETURN': returns_3m,
                'BEST_TARGET_PRICE': target_price,
                'BEST_ANALYST_RATING_MEAN': analyst_rating
            })
        
        # Create DataFrame
        df = pd.DataFrame(results)
        
        # Ensure correct data types
        numeric_fields = ['EQY_SH_OUT', 'CUR_MKT_CAP', 'REL_1D_RETURN', 'REL_5D_RETURN', 
                         'REL_1M_RETURN', 'REL_3M_RETURN', 'BEST_TARGET_PRICE', 'BEST_ANALYST_RATING_MEAN']
        
        for field in numeric_fields:
            if field in df.columns:
                df[field] = df[field].astype(float)
        
        logger.info(f"Generated sample additional info for {len(df)} stocks")
        return df
    
    def get_latest_news(self, tickers: List[str], days_back: int = 5) -> pd.DataFrame:
        """
        Get latest news for stocks with abnormal volume.
        
        Args:
            tickers: List of tickers to get news for
            days_back: Number of days to look back for news
            
        Returns:
            DataFrame with news headlines
        """
        # Return empty DataFrame with correct structure if no tickers
        if not tickers:
            logger.warning("No tickers provided for news")
            return pd.DataFrame(columns=['ticker', 'date', 'sentiment', 'headlines'])
            
        if self.use_sample_data:
            return self.get_sample_news(tickers, days_back)
        
        try:
            # Calculate start and end dates
            end_date = datetime.datetime.now()
            start_date = end_date - datetime.timedelta(days=days_back)
            
            # Initialize results
            all_news = []
            
            # Process each ticker
            for ticker in tickers:
                # Create request for news
                refDataService = self.session.getService("//blp/refdata")
                request = refDataService.createRequest("HistoricalDataRequest")
                
                # Set date range
                request.set("startDate", start_date.strftime("%Y%m%d"))
                request.set("endDate", end_date.strftime("%Y%m%d"))
                
                # Add security
                request.append("securities", ticker)
                
                # Add fields
                request.append("fields", "NEWS_SENTIMENT_DAILY_AVG")
                request.append("fields", "NEWS_HEADLINE_DAILY")
                
                # Send the request
                logger.info(f"Requesting news for {ticker}")
                self.session.sendRequest(request)
                
                # Process the response
                while True:
                    event = self.session.nextEvent(500)
                    
                    for msg in event:
                        # Check message type by name
                        if msg.messageType() == blpapi.Name("HistoricalDataResponse"):
                            # Check for security data
                            security_data = msg.getElement("securityData")
                            ticker = security_data.getElementAsString("security")
                            
                            # Check for field data
                            field_data = security_data.getElement("fieldData")
                            
                            # Process each date
                            for i in range(field_data.numValues()):
                                date_data = field_data.getValue(i)
                                date = date_data.getElementAsDatetime("date").strftime("%Y-%m-%d")
                                
                                # Get news sentiment
                                sentiment = None
                                if date_data.hasElement("NEWS_SENTIMENT_DAILY_AVG"):
                                    sentiment = date_data.getElementAsFloat("NEWS_SENTIMENT_DAILY_AVG")
                                
                                # Get news headlines
                                headlines = []
                                if date_data.hasElement("NEWS_HEADLINE_DAILY"):
                                    news_data = date_data.getElement("NEWS_HEADLINE_DAILY")
                                    
                                    for j in range(news_data.numValues()):
                                        headline = news_data.getValue(j).getElementAsString("Headline")
                                        headlines.append(headline)
                                
                                # Add to results
                                if headlines:
                                    all_news.append({
                                        'ticker': ticker,
                                        'date': date,
                                        'sentiment': sentiment,
                                        'headlines': '\n'.join(headlines)
                                    })
                    
                    if event.eventType() == blpapi.Event.RESPONSE:
                        break
                
                time.sleep(0.1)  # Throttle requests
            
            # Convert to DataFrame
            df = pd.DataFrame(all_news)
            
            # If DataFrame is empty, use sample data
            if df.empty:
                logger.warning("No news retrieved, using sample data")
                return self.get_sample_news(tickers, days_back)
                
            df['date'] = pd.to_datetime(df['date'])
            
            # Sort by date (most recent first)
            df = df.sort_values(['ticker', 'date'], ascending=[True, False])
            
            logger.info(f"Retrieved {len(df)} news items for {len(tickers)} stocks")
            return df
            
        except Exception as e:
            logger.error(f"Error retrieving news: {e}")
            return self.get_sample_news(tickers, days_back)
    
    def get_sample_news(self, tickers: List[str], days_back: int = 5) -> pd.DataFrame:
        """
        Generate sample news headlines for testing.
        
        Args:
            tickers: List of tickers to generate news for
            days_back: Number of days to look back for news
            
        Returns:
            DataFrame with sample news
        """
        logger.info(f"Generating sample news for {len(tickers)} stocks")
        
        # If no tickers, return empty DataFrame with the right columns
        if not tickers:
            return pd.DataFrame(columns=['ticker', 'date', 'sentiment', 'headlines'])
        
        # Generate date range
        end_date = pd.Timestamp.now().normalize()
        date_range = pd.date_range(end=end_date, periods=days_back, freq='D')
        
        # Sample news headlines - generic templates
        positive_headlines = [
            "{company} reports {percent}% increase in quarterly earnings",
            "{company} announces new strategic partnership",
            "{company} expands operations in {region}",
            "Analysts upgrade {company} to 'Buy' rating",
            "{company} wins major contract in {sector} sector",
            "{company} announces share buyback program",
            "{company} dividend increase exceeds expectations",
            "{company} reports record sales",
            "New CEO appointment at {company} welcomed by investors",
            "{company} announces positive trial results"
        ]
        
        negative_headlines = [
            "{company} reports {percent}% decrease in quarterly earnings",
            "{company} announces restructuring plan and job cuts",
            "{company} faces regulatory scrutiny",
            "Analysts downgrade {company} citing competitive pressures",
            "{company} warns of margin pressure due to rising costs",
            "{company} delays product launch, cites supply chain issues",
            "{company} suspends dividend amid challenging environment",
            "{company} market share eroded by competitors",
            "CFO resignation at {company} raises governance concerns",
            "{company} facing lawsuit over business practices"
        ]
        
        neutral_headlines = [
            "{company} holds annual shareholder meeting",
            "{company} announces board changes",
            "{company} to present at industry conference",
            "{company} refinances debt with new bond issue",
            "{company} completes previously announced acquisition",
            "{company} relocates headquarters",
            "{company} releases sustainability report",
            "{company} maintains guidance for fiscal year",
            "{company} announces normal course share issuance",
            "{company} comments on industry trends at conference"
        ]
        
        # Add market-specific regions
        regions = {
            'HK': ['Greater Bay Area', 'Mainland China', 'Southeast Asia', 'Asia Pacific'],
            'US': ['West Coast', 'East Coast', 'Midwest', 'International Markets'],
            'JP': ['Tokyo', 'Osaka', 'Asia Pacific', 'Global Markets'],
            'UK': ['London', 'European Union', 'Commonwealth', 'Global Markets'],
            'CN': ['Beijing', 'Shanghai', 'Greater Bay Area', 'Belt and Road'],
            'TW': ['Taipei', 'Mainland China', 'Southeast Asia', 'Silicon Valley'],
            'KR': ['Seoul', 'Busan', 'Southeast Asia', 'Global Markets'],
            'SG': ['ASEAN', 'India', 'China', 'Global Markets']
        }
        market_regions = regions.get(self.market, ['Global', 'Regional', 'Local', 'International'])
        
        all_news = []
        
        # Generate news for each ticker
        for ticker in tickers:
            ticker_code = ticker.split()[0]
            company_name = f"Company {ticker_code}"
            
            # Use ticker to generate consistent random values
            ticker_num_str = ''.join(filter(str.isdigit, ticker_code[:10]))
            ticker_num = int(ticker_num_str) if ticker_num_str else hash(ticker_code) % 10000
            np.random.seed(ticker_num)
            
            # Generate 1-3 news items per date (more recent dates have more news)
            for i, date in enumerate(date_range):
                # More news for more recent dates
                num_news = np.random.randint(0, 4 - i // 2)
                
                for _ in range(num_news):
                    # Randomly select sentiment (slightly biased toward neutral)
                    sentiment_type = np.random.choice(['positive', 'negative', 'neutral'], p=[0.3, 0.3, 0.4])
                    
                    if sentiment_type == 'positive':
                        headline = np.random.choice(positive_headlines)
                        sentiment = np.random.uniform(0.3, 0.9)
                    elif sentiment_type == 'negative':
                        headline = np.random.choice(negative_headlines)
                        sentiment = np.random.uniform(-0.9, -0.3)
                    else:
                        headline = np.random.choice(neutral_headlines)
                        sentiment = np.random.uniform(-0.2, 0.2)
                    
                    # Format headline with company name and other placeholders
                    headline = headline.format(
                        company=company_name,
                        percent=np.random.randint(5, 35),
                        sector=np.random.choice(['technology', 'finance', 'healthcare', 'consumer', 'industrial']),
                        region=np.random.choice(market_regions)
                    )
                    
                    all_news.append({
                        'ticker': ticker,
                        'date': date,
                        'sentiment': sentiment,
                        'headlines': headline
                    })
        
        # Create DataFrame
        df = pd.DataFrame(all_news)
        
        # Sort by date (most recent first)
        df = df.sort_values(['ticker', 'date'], ascending=[True, False])
        
        logger.info(f"Generated {len(df)} sample news items")
        return df
    
    def detect_and_analyze(self,
                          stock_list_file: Optional[str] = None,
                          lookback_period: int = 20,
                          zscore_threshold: float = 2.5,
                          percent_change_threshold: float = 200,
                          percentile_threshold: float = 95) -> Dict[str, Any]:
        """
        Detect abnormal volume and perform analysis on the results.
        
        Args:
            stock_list_file: Path to file containing stock tickers
            lookback_period: Trading days to establish baseline volume
            zscore_threshold: Z-score threshold for abnormal volume
            percent_change_threshold: Percent increase threshold for abnormal volume
            percentile_threshold: Historical percentile threshold for abnormal volume
            
        Returns:
            Dictionary with detection results and analysis
        """
        # Detect abnormal volume
        volume_data = self.detect_abnormal_volume(
            stock_list_file=stock_list_file,
            lookback_period=lookback_period,
            zscore_threshold=zscore_threshold,
            percent_change_threshold=percent_change_threshold,
            percentile_threshold=percentile_threshold
        )
        
        # Extract list of tickers with abnormal volume
        abnormal_tickers = [item['ticker'] for item in volume_data.get('Abnormal_Volume', [])]
        
        if not abnormal_tickers:
            logger.warning("No abnormal volume detected")
            return volume_data
        
        # Get additional stock information
        logger.info(f"Getting additional information for {len(abnormal_tickers)} stocks with abnormal volume")
        stock_info = self.get_additional_stock_info(abnormal_tickers)
        
        # Get recent news headlines
        logger.info(f"Getting recent news for stocks with abnormal volume")
        news_data = self.get_latest_news(abnormal_tickers, days_back=5)
        
        # Merge the data
        for i, item in enumerate(volume_data['Abnormal_Volume']):
            ticker = item['ticker']
            
            # Add stock info
            if not stock_info.empty and ticker in stock_info['ticker'].values:
                stock_row = stock_info[stock_info['ticker'] == ticker].iloc[0]
                for column in stock_info.columns:
                    if column != 'ticker':
                        volume_data['Abnormal_Volume'][i][column] = stock_row[column]
            
            # Add most recent news
            if not news_data.empty and ticker in news_data['ticker'].values:
                ticker_news = news_data[news_data['ticker'] == ticker]
                if not ticker_news.empty:
                    recent_news = ticker_news.iloc[0]
                    volume_data['Abnormal_Volume'][i]['recent_news'] = recent_news['headlines']
                    volume_data['Abnormal_Volume'][i]['news_sentiment'] = recent_news['sentiment']
        
        # Additional analysis
        self.volume_data = volume_data
        
        return volume_data
    
    def print_abnormal_volume_report(self) -> None:
        """Print a report of stocks with abnormal trading volume."""
        if not self.volume_data or 'Abnormal_Volume' not in self.volume_data:
            logger.error("No volume data available. Run detect_abnormal_volume() first.")
            return
        
        market_name = self.market
        abnormal_volume = self.volume_data.get('Abnormal_Volume', [])
        if not abnormal_volume:
            logger.warning("No abnormal volume detected.")
            print("\n" + "="*120)
            print(f"ABNORMAL TRADING VOLUME REPORT FOR {market_name} MARKET - {self.latest_update_time.strftime('%Y-%m-%d %H:%M:%S')}")
            print("="*120)
            print(f"Lookback Period: {self.lookback_period} trading days")
            print(f"Detection Methods: Z-Score > {self.zscore_threshold}, Volume Change > {self.percent_change_threshold}%, Historical Percentile > {self.percentile_threshold}%")
            print(f"No stocks with abnormal volume detected out of {self.volume_data.get('Total_Stocks', 0)} stocks")
            print("="*120 + "\n")
            return
        
        print("\n" + "="*120)
        print(f"ABNORMAL TRADING VOLUME REPORT FOR {market_name} MARKET - {self.latest_update_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*120)
        print(f"Lookback Period: {self.lookback_period} trading days")
        print(f"Detection Methods: Z-Score > {self.zscore_threshold}, Volume Change > {self.percent_change_threshold}%, Historical Percentile > {self.percentile_threshold}%")
        print(f"Detected {len(abnormal_volume)} stocks with abnormal volume out of {self.volume_data.get('Total_Stocks', 0)} stocks")
        print("="*120 + "\n")
        
        # Prepare table data - main summary
        table_data = []
        
        for stock in abnormal_volume:
            ticker = stock.get('ticker', '')
            
            # Get stock name
            name = stock.get('NAME', '')
            if not name:
                name = f"Stock {ticker.split()[0]}"
            
            # Format volume
            volume = stock.get('volume', 0)
            baseline_volume = stock.get('baseline_volume', 0)
            volume_formatted = f"{volume/1e6:.2f}M" if volume >= 1e6 else f"{volume/1e3:.2f}K"
            baseline_formatted = f"{baseline_volume/1e6:.2f}M" if baseline_volume >= 1e6 else f"{baseline_volume/1e3:.2f}K"
            
            # Color-code percent change
            percent_change = stock.get('percent_change', 0)
            if percent_change > 400:
                percent_change_formatted = f"{RED}{BOLD}{percent_change:.1f}%{RESET}"
            elif percent_change > 200:
                percent_change_formatted = f"{RED}{percent_change:.1f}%{RESET}"
            else:
                percent_change_formatted = f"{percent_change:.1f}%"
            
            # Add z-score
            zscore = stock.get('zscore', 0)
            zscore_formatted = f"{zscore:.2f}"
            
            # Add price and price change
            price = stock.get('price', None)
            price_change_1d = stock.get('price_change_1d', None)
            
            if price is not None:
                price_formatted = f"{price:.2f}"
            else:
                price_formatted = "N/A"
                
            if price_change_1d is not None:
                # Color-code price change
                if price_change_1d > 5:
                    price_change_formatted = f"{GREEN}{BOLD}+{price_change_1d:.2f}%{RESET}"
                elif price_change_1d > 0:
                    price_change_formatted = f"{GREEN}+{price_change_1d:.2f}%{RESET}"
                elif price_change_1d < -5:
                    price_change_formatted = f"{RED}{BOLD}{price_change_1d:.2f}%{RESET}"
                elif price_change_1d < 0:
                    price_change_formatted = f"{RED}{price_change_1d:.2f}%{RESET}"
                else:
                    price_change_formatted = "0.00%"
            else:
                price_change_formatted = "N/A"
            
            # Add detection method
            detection_method = stock.get('detection_method', [])
            method_str = ', '.join(detection_method)
            
            # Add to table
            table_data.append([
                ticker,
                name,
                stock.get('INDUSTRY_SECTOR', 'N/A'),
                volume_formatted,
                baseline_formatted,
                percent_change_formatted,
                zscore_formatted,
                price_formatted,
                price_change_formatted,
                method_str
            ])
        
        # Print the table
        headers = ["Ticker", "Name", "Sector", "Volume", "Avg Volume", "% Change", "Z-Score", "Price", "1D Return", "Detection Method"]
        print(tabulate(table_data, headers=headers, tablefmt="psql"))
        
        # Print detailed report for top 10 volume changes
        top_10 = sorted(abnormal_volume, key=lambda x: x['percent_change'], reverse=True)[:10]
        
        if top_10:
            print("\n" + "="*120)
            print("DETAILED REPORT FOR TOP 10 VOLUME CHANGES")
            print("="*120 + "\n")
            
            for stock in top_10:
                ticker = stock.get('ticker', '')
                name = stock.get('NAME', ticker)
                
                print(f"{BOLD}{name} ({ticker}){RESET}")
                print(f"Sector: {stock.get('INDUSTRY_SECTOR', 'N/A')}")
                print(f"Market Cap: {self.format_market_cap(stock.get('CUR_MKT_CAP', 0))}")
                
                # Volume stats
                print(f"Current Volume: {stock.get('volume', 0)/1e6:.2f}M shares (Baseline: {stock.get('baseline_volume', 0)/1e6:.2f}M)")
                print(f"Volume Change: {BOLD}{stock.get('percent_change', 0):.1f}%{RESET} (Z-Score: {stock.get('zscore', 0):.2f}, Percentile: {stock.get('percentile', 0):.1f})")
                
                # Price stats
                print(f"Current Price: {stock.get('price', 0):.2f}")
                
                # Returns
                rel_1d = stock.get('REL_1D_RETURN', None)
                rel_5d = stock.get('REL_5D_RETURN', None)
                rel_1m = stock.get('REL_1M_RETURN', None)
                
                print(f"Returns: 1D: {self.format_return(rel_1d)}, 5D: {self.format_return(rel_5d)}, 1M: {self.format_return(rel_1m)}")
                
                # Analyst data
                target = stock.get('BEST_TARGET_PRICE', None)
                rating = stock.get('BEST_ANALYST_RATING_MEAN', None)
                
                if target is not None and price is not None and price > 0:
                    upside = (target / price - 1) * 100
                    print(f"Analyst Target: {target:.2f} ({self.format_return(upside)})")
                elif target is not None:
                    print(f"Analyst Target: {target:.2f}")
                
                if rating is not None:
                    rating_text = "Strong Buy" if rating >= 4.5 else "Buy" if rating >= 3.5 else "Hold" if rating >= 2.5 else "Sell" if rating >= 1.5 else "Strong Sell"
                    print(f"Analyst Rating: {rating:.1f}/5.0 ({rating_text})")
                
                # Recent news
                recent_news = stock.get('recent_news', None)
                sentiment = stock.get('news_sentiment', None)
                
                if recent_news:
                    # Handle multiline news
                    news_lines = recent_news.split('\n')
                    if sentiment is not None:
                        sentiment_text = "Positive" if sentiment > 0.2 else "Negative" if sentiment < -0.2 else "Neutral"
                        news_header = f"Recent News (Sentiment: {sentiment_text}, {sentiment:.2f}):"
                    else:
                        news_header = "Recent News:"
                    
                    print(news_header)
                    for i, line in enumerate(news_lines):
                        if i < 3:  # Limit to 3 news items
                            print(f"  - {line}")
                        elif i == 3:
                            print(f"  - ... ({len(news_lines) - 3} more news items)")
                            break
                
                print("\n" + "-"*80 + "\n")
        
        print("="*120 + "\n")
    
    def format_market_cap(self, market_cap: float) -> str:
        """
        Format market cap for display using the currency for the selected market.
        
        Args:
            market_cap: Market cap value
            
        Returns:
            Formatted market cap string
        """
        if market_cap >= 1e12:
            return f"{self.currency_symbol}{market_cap/1e12:.2f}T"
        elif market_cap >= 1e9:
            return f"{self.currency_symbol}{market_cap/1e9:.2f}B"
        elif market_cap >= 1e6:
            return f"{self.currency_symbol}{market_cap/1e6:.2f}M"
        else:
            return f"{self.currency_symbol}{market_cap:.2f}"
    
    def format_return(self, return_value: Optional[float]) -> str:
        """
        Format return percentage for display with color coding.
        
        Args:
            return_value: Return percentage or None
            
        Returns:
            Formatted return string
        """
        if return_value is None:
            return "N/A"
            
        if return_value > 10:
            return f"{GREEN}{BOLD}+{return_value:.2f}%{RESET}"
        elif return_value > 0:
            return f"{GREEN}+{return_value:.2f}%{RESET}"
        elif return_value < -10:
            return f"{RED}{BOLD}{return_value:.2f}%{RESET}"
        elif return_value < 0:
            return f"{RED}{return_value:.2f}%{RESET}"
        else:
            return "0.00%"
    
    def plot_volume_spikes(self, num_stocks: int = 5) -> None:
        """
        Plot volume charts for top stocks with abnormal volume.
        
        Args:
            num_stocks: Number of top stocks to plot
        """
        if not self.volume_data or 'Abnormal_Volume' not in self.volume_data:
            logger.error("No volume data available. Run detect_abnormal_volume() first.")
            return
        
        abnormal_volume = self.volume_data.get('Abnormal_Volume', [])
        if not abnormal_volume:
            logger.warning("No abnormal volume detected.")
            return
        
        # Get top N stocks by percent change
        top_stocks = sorted(abnormal_volume, key=lambda x: x['percent_change'], reverse=True)[:num_stocks]
        tickers = [stock['ticker'] for stock in top_stocks]
        
        # Get historical data for these stocks
        history_days = max(30, self.lookback_period * 2)  # More history for better visualization
        hist_data = self.get_historical_prices_and_volumes(tickers, lookback_days=history_days)
        
        # Plot each stock
        fig, axs = plt.subplots(num_stocks, 1, figsize=(12, 4 * num_stocks), sharex=False)
        if num_stocks == 1:
            axs = [axs]  # Convert to list for consistent indexing
        
        for i, ticker in enumerate(tickers):
            ticker_data = hist_data[hist_data['ticker'] == ticker].sort_values('date')
            
            if len(ticker_data) < 5:  # Need at least 5 data points for meaningful plot
                continue
                
            # Get stock info from the abnormal volume list
            stock_info = next((s for s in top_stocks if s['ticker'] == ticker), None)
            name = stock_info.get('NAME', ticker) if stock_info else ticker
            
            # Create figure
            ax1 = axs[i]
            ax2 = ax1.twinx()
            
            # Plot price
            ax1.plot(ticker_data['date'], ticker_data['price'], 'b-', label='Price')
            ax1.set_ylabel(f'Price ({self.currency_symbol})', color='b')
            ax1.tick_params(axis='y', labelcolor='b')
            
            # Plot volume bars
            volume_bars = ax2.bar(ticker_data['date'], ticker_data['volume'] / 1e6, 
                                 alpha=0.3, color='g', label='Volume (Million)')
            
            # Highlight the most recent bar (the abnormal one)
            if len(volume_bars) > 0:
                volume_bars[-1].set_color('r')
                volume_bars[-1].set_alpha(0.7)
            
            ax2.set_ylabel('Volume (Million Shares)', color='g')
            ax2.tick_params(axis='y', labelcolor='g')
            
            # Add a title with the abnormal volume info
            if stock_info:
                percent_change = stock_info.get('percent_change', 0)
                baseline = stock_info.get('baseline_volume', 0) / 1e6
                latest = stock_info.get('volume', 0) / 1e6
                title = f"{name} - Volume Spike: {percent_change:.1f}% (Baseline: {baseline:.2f}M, Latest: {latest:.2f}M)"
            else:
                title = f"{name} - Volume Spike"
                
            ax1.set_title(title)
            
            # Format x-axis
            ax1.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d'))
            plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
            
            # Add grid
            ax1.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def save_results_to_csv(self, output_dir: str = None) -> None:
        """
        Save abnormal volume results to CSV file.
        
        Args:
            output_dir: Directory to save output files (optional)
        """
        if not self.volume_data or 'Abnormal_Volume' not in self.volume_data:
            logger.error("No volume data available. Run detect_abnormal_volume() first.")
            return
        
        abnormal_volume = self.volume_data.get('Abnormal_Volume', [])
        if not abnormal_volume:
            logger.warning("No abnormal volume data to save.")
            return
        
        if not output_dir:
            output_dir = './'
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = self.latest_update_time.strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(output_dir, f"{self.market.lower()}_abnormal_volume_{timestamp}.csv")
        
        # Convert to DataFrame
        df = pd.DataFrame(abnormal_volume)
        
        # Save to CSV
        df.to_csv(output_file, index=False)
        logger.info(f"Saved abnormal volume data to {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Monitor and detect abnormal trading volume in stocks')
    parser.add_argument('--host', default='127.0.0.1', help='Bloomberg server host')
    parser.add_argument('--port', type=int, default=8194, help='Bloomberg server port')
    parser.add_argument('--output-dir', default='./volume_monitor', help='Directory to save output files')
    parser.add_argument('--sample', action='store_true', 
                        help='Use sample data instead of Bloomberg data')
    parser.add_argument('--market', default='HK', choices=['HK', 'US', 'JP', 'UK', 'CN', 'TW', 'KR', 'SG', 'AU', 'EU'],
                        help='Market to analyze (default: HK)')
    parser.add_argument('--stock-file', 
                        help='Path to file containing stock tickers')
    parser.add_argument('--lookback', type=int, default=20, 
                        help='Trading days to establish baseline volume')
    parser.add_argument('--zscore', type=float, default=2.5, 
                        help='Z-score threshold for abnormal volume')
    parser.add_argument('--percent', type=float, default=200, 
                        help='Percent increase threshold for abnormal volume')
    parser.add_argument('--percentile', type=float, default=95, 
                        help='Historical percentile threshold for abnormal volume')
    parser.add_argument('--plot', action='store_true', 
                        help='Plot volume charts for top abnormal volume stocks')
    parser.add_argument('--plot-count', type=int, default=5, 
                        help='Number of stocks to plot')
    
    args = parser.parse_args()
    
    # Create output directory if needed
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize volume monitor
    monitor = VolumeMonitor(
        host=args.host, 
        port=args.port, 
        use_sample_data=args.sample, 
        market=args.market
    )
    
    try:
        # Start session
        if monitor.start_session():
            # Detect abnormal volume and analyze
            logger.info(f"Detecting abnormal trading volume in {args.market} stocks...")
            monitor.detect_and_analyze(
                stock_list_file=args.stock_file,
                lookback_period=args.lookback,
                zscore_threshold=args.zscore,
                percent_change_threshold=args.percent,
                percentile_threshold=args.percentile
            )
            
            # Print report
            monitor.print_abnormal_volume_report()
            
            # Plot volume spikes if requested
            if args.plot:
                monitor.plot_volume_spikes(num_stocks=args.plot_count)
            
            # Save results
            logger.info("Saving results to CSV...")
            monitor.save_results_to_csv(output_dir=args.output_dir)
            
            logger.info("Abnormal volume detection completed.")
    
    except KeyboardInterrupt:
        logger.info("Detection interrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        import traceback
        logger.error(traceback.format_exc())
    finally:
        # Always stop the session
        monitor.stop_session()


if __name__ == "__main__":
    main()