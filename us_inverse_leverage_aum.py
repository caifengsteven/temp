import blpapi
import pandas as pd
import numpy as np
import datetime
import logging
import os
import argparse
import time
import re
from typing import List, Dict, Any, Optional
from tabulate import tabulate

# ANSI color codes for terminal output
RED = "\033[91m"        # Bright red
BOLD = "\033[1m"        # Bold
FLASH = "\033[5m"       # Flashing text (not supported in all terminals)
RESET = "\033[0m"       # Reset to default

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("us_leverage_funds_tracker.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('USLeverageFundsTracker')

class USLeverageFundsTracker:
    """
    A class to track and analyze Leveraged and Inverse ETFs in US markets.
    """
    
    def __init__(self, host: str = '127.0.0.1', port: int = 8194, use_sample_data: bool = False):
        """
        Initialize the Bloomberg connection.
        
        Args:
            host: Bloomberg server host (default: localhost)
            port: Bloomberg server port (default: 8194)
            use_sample_data: Force use of sample data even if Bloomberg is available
        """
        self.host = host
        self.port = port
        self.session = None
        self.use_sample_data = use_sample_data
        
        # Major US indices for leveraged/inverse products
        self.us_indices = [
            'SPX Index',       # S&P 500
            'NDX Index',       # NASDAQ-100
            'INDU Index',      # Dow Jones Industrial Average
            'RTY Index',       # Russell 2000
            'VIX Index',       # CBOE Volatility Index
            'SOX Index',       # PHLX Semiconductor Index
            'XAU Index',       # Philadelphia Gold/Silver Index
            'XOI Index',       # NYSE Arca Oil Index
            'MXEA Index',      # MSCI EAFE Index
            'MXEF Index'       # MSCI Emerging Markets Index
        ]
        
        # Major US issuers of leveraged/inverse products
        self.us_li_issuers = [
            'ProShares',
            'Direxion',
            'Leveraged ETFs',
            'UltraPro',
            'UltraPro Short',
            'Ultra',
            'UltraShort',
            'Daily',
            'iPath',
            'iShares',
            'Invesco',
            'Velocity Shares',
            'Credit Suisse',
            'MicroSectors',
            'BMO',
            'ETRACS',
            'GraniteShares'
        ]
        
        # Mapping of underlying names to Bloomberg codes
        self.underlying_code_map = {
            'S&P 500 Index': 'SPX Index',                  # S&P 500
            'S&P 500 VIX Index': 'VIX Index',              # CBOE Volatility Index
            'NASDAQ-100 Index': 'NDX Index',               # NASDAQ-100
            'Dow Jones Industrial Average': 'INDU Index',  # Dow Jones Industrial Average
            'Russell 2000 Index': 'RTY Index',             # Russell 2000
            'Russell 1000 Index': 'RIY Index',             # Russell 1000
            'PHLX Semiconductor Index': 'SOX Index',       # PHLX Semiconductor
            'NYSE Arca Gold Miners Index': 'GDM Index',    # NYSE Arca Gold Miners
            'S&P Biotech Select Index': 'S5BIOTR Index',   # S&P Biotech Select
            'Junior Gold Miners Index': 'GDXJ Index',      # VanEck Junior Gold Miners
            'MSCI Emerging Markets Index': 'MXEF Index',   # MSCI Emerging Markets
            'MSCI EAFE Index': 'MXEA Index',               # MSCI EAFE
            'Bloomberg Natural Gas Index': 'BCOMNG Index', # Bloomberg Natural Gas
            'Bloomberg Crude Oil Index': 'BCOMCL Index',   # Bloomberg Crude Oil
            'NYSE Arca Oil Index': 'XOI Index',            # NYSE Arca Oil
            'FANG+ Index': 'NYFANG Index',                 # NYSE FANG+
            'Financial Select Sector Index': 'S5FINL Index', # Financial Select Sector
            'Technology Select Sector Index': 'S5INFT Index', # Technology Select Sector
            'Health Care Select Sector Index': 'S5HLTH Index', # Health Care Select Sector
            'Energy Select Sector Index': 'S5ENRS Index',  # Energy Select Sector
            'S&P 500 Materials Sector Index': 'S5MATR Index', # Materials Select Sector
            'Utilities Select Sector Index': 'S5UTIL Index', # Utilities Select Sector
            'Real Estate Select Sector Index': 'S5RLST Index', # Real Estate Select Sector
            'S&P Regional Banks Index': 'S5RBNK Index',    # S&P Regional Banks
            'S&P Retail Select Index': 'S5RETL Index'      # S&P Retail Select
        }
        
        # ETF name patterns indicating leverage/inverse
        self.leverage_inverse_patterns = [
            '2X', '3X', '-1X', '-2X', '-3X',
            'BULL', 'BEAR', 'ULTRA', 'SHORT',
            'LEVERAGED', 'INVERSE', 'DAILY',
            'PRO', 'PROSHARES', 'DIREXION'
        ]
        
        # For storing results
        self.li_fund_data = {}
        self.latest_update_time = None
    
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
    
    def get_security_data(self, tickers: List[str], fields: List[str]) -> pd.DataFrame:
        """
        Get Bloomberg data for a list of securities.
        
        Args:
            tickers: List of Bloomberg tickers
            fields: List of Bloomberg fields to retrieve
            
        Returns:
            DataFrame with security data
        """
        if self.use_sample_data:
            # Generate sample data if requested
            return pd.DataFrame()
            
        if not self.session:
            logger.error("Session not started. Call start_session() first.")
            return pd.DataFrame()
            
        if not tickers:
            logger.warning("No tickers provided for data retrieval")
            return pd.DataFrame()
            
        try:
            # Get reference data service
            refDataService = self.session.getService("//blp/refdata")
            
            # For large lists, chunk the requests
            chunk_size = 100
            all_data = []
            
            for i in range(0, len(tickers), chunk_size):
                chunk = tickers[i:i + chunk_size]
                
                # Create request
                request = refDataService.createRequest("ReferenceDataRequest")
                
                # Add securities
                for ticker in chunk:
                    request.append("securities", ticker)
                
                # Add fields
                for field in fields:
                    request.append("fields", field)
                
                # Send the request
                logger.info(f"Requesting data for {len(chunk)} securities (chunk {i//chunk_size + 1})")
                self.session.sendRequest(request)
                
                # Process the response
                results = []
                
                while True:
                    event = self.session.nextEvent(500)
                    
                    for msg in event:
                        # Skip if no security data
                        if not msg.hasElement("securityData"):
                            continue
                            
                        securities_data = msg.getElement("securityData")
                        
                        for j in range(securities_data.numValues()):
                            security_data = securities_data.getValue(j)
                            ticker = security_data.getElementAsString("security")
                            
                            data = {'TICKER': ticker}
                            
                            # Check for field data
                            if security_data.hasElement("fieldData"):
                                field_data = security_data.getElement("fieldData")
                                
                                # Extract each field
                                for field in fields:
                                    if field_data.hasElement(field):
                                        # Try as string first, then as float
                                        try:
                                            data[field] = field_data.getElementAsString(field)
                                        except Exception:
                                            try:
                                                data[field] = field_data.getElementAsFloat(field)
                                            except Exception:
                                                data[field] = None
                                    else:
                                        data[field] = None
                            
                            # Check for field exceptions or errors
                            if security_data.hasElement("fieldExceptions"):
                                field_exceptions = security_data.getElement("fieldExceptions")
                                for k in range(field_exceptions.numValues()):
                                    exception = field_exceptions.getValue(k)
                                    if exception.hasElement("fieldId") and exception.hasElement("errorInfo"):
                                        field_id = exception.getElement("fieldId").getValueAsString()
                                        error_info = exception.getElement("errorInfo").getElement("message").getValueAsString()
                                        logger.debug(f"Field exception for {ticker}: {field_id} - {error_info}")
                            
                            results.append(data)
                    
                    if event.eventType() == blpapi.Event.RESPONSE:
                        # End of RESPONSE event
                        break
                
                all_data.extend(results)
                
                # Throttle requests
                if i + chunk_size < len(tickers):
                    time.sleep(0.1)
            
            # Convert to DataFrame
            df = pd.DataFrame(all_data)
            logger.info(f"Retrieved data for {len(df)} securities")
            return df
            
        except Exception as e:
            logger.error(f"Error getting security data: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return pd.DataFrame()
    
    def get_us_leveraged_inverse_funds(self) -> pd.DataFrame:
        """
        Get leveraged and inverse ETFs listed in the US using direct ticker lookup
        and additional filtering.
        
        Returns:
            DataFrame with leveraged/inverse ETF details
        """
        if self.use_sample_data:
            # Return sample data in sample mode
            return self.get_sample_li_etfs()
        
        # Known leveraged/inverse ETF tickers in US markets
        # These are commonly traded and popular leveraged/inverse products
        known_li_etfs = [
            # ProShares S&P 500
            'SSO US Equity',   # ProShares Ultra S&P 500 (2x)
            'UPRO US Equity',  # ProShares UltraPro S&P 500 (3x)
            'SDS US Equity',   # ProShares UltraShort S&P 500 (-2x)
            'SPXU US Equity',  # ProShares UltraPro Short S&P 500 (-3x)
            
            # ProShares NASDAQ-100
            'QLD US Equity',   # ProShares Ultra QQQ (2x)
            'TQQQ US Equity',  # ProShares UltraPro QQQ (3x)
            'QID US Equity',   # ProShares UltraShort QQQ (-2x)
            'SQQQ US Equity',  # ProShares UltraPro Short QQQ (-3x)
            
            # ProShares Dow Jones
            'DDM US Equity',   # ProShares Ultra Dow30 (2x)
            'UDOW US Equity',  # ProShares UltraPro Dow30 (3x)
            'DXD US Equity',   # ProShares UltraShort Dow30 (-2x)
            'SDOW US Equity',  # ProShares UltraPro Short Dow30 (-3x)
            
            # ProShares Russell 2000
            'UWM US Equity',   # ProShares Ultra Russell 2000 (2x)
            'URTY US Equity',  # ProShares UltraPro Russell 2000 (3x)
            'TWM US Equity',   # ProShares UltraShort Russell 2000 (-2x)
            'SRTY US Equity',  # ProShares UltraPro Short Russell 2000 (-3x)
            
            # ProShares Sector ETFs
            'ROM US Equity',   # ProShares Ultra Technology (2x)
            'SOXL US Equity',  # Direxion Daily Semiconductor Bull 3x
            'SOXS US Equity',  # Direxion Daily Semiconductor Bear 3x
            'FAS US Equity',   # Direxion Daily Financial Bull 3x
            'FAZ US Equity',   # Direxion Daily Financial Bear 3x
            'SPXS US Equity',  # Direxion Daily S&P 500 Bear 3x
            'SPXL US Equity',  # Direxion Daily S&P 500 Bull 3x
            
            # VIX ETFs
            'UVXY US Equity',  # ProShares Ultra VIX Short-Term Futures ETF
            'SVXY US Equity',  # ProShares Short VIX Short-Term Futures ETF
            'VIXM US Equity',  # ProShares VIX Mid-Term Futures ETF
            'VIXY US Equity',  # ProShares VIX Short-Term Futures ETF
            
            # Commodity Leveraged/Inverse ETFs
            'UCO US Equity',   # ProShares Ultra Bloomberg Crude Oil (2x)
            'SCO US Equity',   # ProShares UltraShort Bloomberg Crude Oil (-2x)
            'BOIL US Equity',  # ProShares Ultra Bloomberg Natural Gas (2x)
            'KOLD US Equity',  # ProShares UltraShort Bloomberg Natural Gas (-2x)
            'NUGT US Equity',  # Direxion Daily Gold Miners Bull 2x
            'DUST US Equity',  # Direxion Daily Gold Miners Bear 2x
            'JNUG US Equity',  # Direxion Daily Junior Gold Miners Bull 2x
            'JDST US Equity',  # Direxion Daily Junior Gold Miners Bear 2x
            
            # International Leveraged/Inverse ETFs
            'EFO US Equity',   # ProShares Ultra MSCI EAFE (2x)
            'EFU US Equity',   # ProShares UltraShort MSCI EAFE (-2x)
            'EDC US Equity',   # Direxion Daily Emerging Markets Bull 3x
            'EDZ US Equity',   # Direxion Daily Emerging Markets Bear 3x
            'YINN US Equity',  # Direxion Daily FTSE China Bull 3x
            'YANG US Equity',  # Direxion Daily FTSE China Bear 3x
            
            # More ProShares & Direxion Sector ETFs
            'ERX US Equity',   # Direxion Daily Energy Bull 2x
            'ERY US Equity',   # Direxion Daily Energy Bear 2x
            'LABU US Equity',  # Direxion Daily S&P Biotech Bull 3x
            'LABD US Equity',  # Direxion Daily S&P Biotech Bear 3x
            'TECS US Equity',  # Direxion Daily Technology Bear 3x
            'TECL US Equity',  # Direxion Daily Technology Bull 3x
            'DRN US Equity',   # Direxion Daily Real Estate Bull 3x
            'DRV US Equity',   # Direxion Daily Real Estate Bear 3x
            
            # MicroSectors
            'FNGU US Equity',  # MicroSectors FANG+ Bull 3X ETN
            'FNGD US Equity',  # MicroSectors FANG+ Bear 3X ETN
            'BULZ US Equity',  # MicroSectors Solactive FANG Bull 3X ETN
            'BERZ US Equity',  # MicroSectors Solactive FANG Bear 3X ETN
            
            # Others
            'TBTF US Equity',  # iShares Trust - 20+ Year Treasury Bond ETF
            'TMF US Equity',   # Direxion Daily 20+ Year Treasury Bull 3x
            'TMV US Equity',   # Direxion Daily 20+ Year Treasury Bear 3x
            'TYD US Equity',   # Direxion Daily 7-10 Year Treasury Bull 3x
            'TYO US Equity',   # Direxion Daily 7-10 Year Treasury Bear 3x
            'FNGG US Equity',  # Direxion Daily Select Large Caps & FANGs Bull 2X
            'SZK US Equity'    # ProShares UltraShort Consumer Goods
        ]
        
        # Systematically look for more leveraged/inverse ETFs using patterns in their names
        # Use ETF issuers known for offering leveraged/inverse products
        # These are ETF prefix/suffix patterns commonly used for leveraged products
        li_patterns = []
        for pattern in ['ULTRA', 'ULTRASHORT', 'ULTRAPRO', 'BULL', 'BEAR', '2X', '3X', 'DAILY']:
            li_patterns.append(f"{pattern}* US Equity")
        
        # Search for standard ProShares and Direxion patterns
        for prefix in ['UPRO', 'SPXU', 'TQQQ', 'SQQQ', 'UDOW', 'SDOW', 'URTY', 'SRTY', 'LABU', 'LABD', 'NUGT', 'DUST']:
            li_patterns.append(f"{prefix}* US Equity")
        
        # Known issuer prefixes
        for issuer in ['DIRX', 'PRO', 'GVIP', 'VSTO', 'DRIP', 'GUSH', 'DRN', 'DRV', 'NAIL', 'LABU', 'LABD']:
            li_patterns.append(f"{issuer}* US Equity")
        
        # Combine direct tickers and patterns for a more complete search
        search_tickers = known_li_etfs
        
        logger.info(f"Checking leveraged/inverse ETFs with {len(search_tickers)} ticker patterns")
        
        # Get data for these tickers
        fields = [
            'SECURITY_DES', 
            'SHORT_NAME', 
            'NAME', 
            'LONG_COMP_NAME',
            'SECURITY_TYP2', 
            'FUND_NET_ASSET_VAL', 
            'FUND_TOTAL_ASSETS', 
            'FUND_MKTCAP', 
            'FX_MARKET_CAP',
            'EQY_FUND_CRNCY',
            'FUND_LEVERAGE_FACTOR',
            'FUND_BENCHMARK',
            'FUND_OBJECTIVE',
            'FUND_ASSET_CLASS_FOCUS',
            'CIE_DES',
            'SECURITY_TYP'
        ]
        
        # Get direct ticker data
        etfs = self.get_security_data(search_tickers, fields)
        
        if etfs.empty:
            logger.warning("No ETF data retrieved. Using sample data.")
            return self.get_sample_li_etfs()
        
        # Filter for likely leveraged/inverse funds
        def is_leveraged_inverse(row):
            # Check security description and name
            for field in ['SECURITY_DES', 'NAME', 'SHORT_NAME', 'LONG_COMP_NAME', 'FUND_OBJECTIVE']:
                if field in row and isinstance(row[field], str):
                    text = row[field].upper()
                    
                    # Look for common keywords indicating leverage/inverse
                    for pattern in self.leverage_inverse_patterns:
                        if pattern.upper() in text:
                            return True
            
            # Check leverage factor field
            if 'FUND_LEVERAGE_FACTOR' in row and pd.notna(row['FUND_LEVERAGE_FACTOR']):
                factor = row['FUND_LEVERAGE_FACTOR']
                try:
                    factor_val = float(factor)
                    if abs(factor_val) != 1.0:  # Not standard 1x exposure
                        return True
                except:
                    # Handle non-numeric values
                    if isinstance(factor, str) and any(x in factor.upper() for x in ['2X', '3X', '-1X', '-2X', '-3X']):
                        return True
            
            # Check ticker for common patterns
            ticker = row.get('TICKER', '')
            if isinstance(ticker, str):
                ticker_prefix = ticker.split()[0]
                # Check for common leveraged ETF tickers
                for prefix in ['TQQQ', 'SQQQ', 'UPRO', 'SPXU', 'UDOW', 'SDOW', 'URTY', 'SRTY', 
                              'SOXL', 'SOXS', 'FAS', 'FAZ', 'ERX', 'ERY', 'LABU', 'LABD',
                              'NUGT', 'DUST', 'JNUG', 'JDST', 'UCO', 'SCO', 'BOIL', 'KOLD']:
                    if ticker_prefix == prefix:
                        return True
            
            return False
        
        # Apply filter
        leveraged_inverse_df = etfs[etfs.apply(is_leveraged_inverse, axis=1)].copy()
        
        if leveraged_inverse_df.empty:
            logger.warning("No leveraged/inverse ETFs identified after filtering. Using sample data.")
            return self.get_sample_li_etfs()
        
        logger.info(f"Identified {len(leveraged_inverse_df)} leveraged/inverse ETFs")
        
        # Print a few examples of what we found for debugging
        logger.info(f"Sample of identified ETFs: {leveraged_inverse_df[['TICKER', 'NAME']].head(3).to_dict('records')}")
        
        # Extract leverage type (leveraged or inverse)
        def get_leverage_type(row):
            # Check name and description
            for field in ['SECURITY_DES', 'NAME', 'SHORT_NAME', 'LONG_COMP_NAME', 'FUND_OBJECTIVE']:
                if field in row and isinstance(row[field], str):
                    text = row[field].upper()
                    
                    # Check for inverse indicators
                    if any(kw in text for kw in ['INVERSE', 'SHORT', 'BEAR', '-1X', '-2X', '-3X', 'ULTRASHORT']):
                        return 'Inverse'
                    # Check for leveraged indicators
                    if any(kw in text for kw in ['LEVERAGED', 'BULL', '2X', '3X', 'ULTRA', 'ULTRAPRO']):
                        return 'Leveraged'
            
            # Check leverage factor
            if 'FUND_LEVERAGE_FACTOR' in row and pd.notna(row['FUND_LEVERAGE_FACTOR']):
                try:
                    factor = float(row['FUND_LEVERAGE_FACTOR'])
                    if factor < 0:
                        return 'Inverse'
                    elif factor > 1:
                        return 'Leveraged'
                except:
                    pass
            
            # Check ticker suffix patterns
            ticker = row.get('TICKER', '')
            if isinstance(ticker, str):
                ticker_prefix = ticker.split()[0]
                # Common inverse ETF ticker patterns
                if ticker_prefix in ['SQQQ', 'SPXU', 'SDOW', 'SRTY', 'SOXS', 'FAZ', 'LABD', 'DUST', 'DRV', 'TMV']:
                    return 'Inverse'
                # Common leveraged ETF ticker patterns
                elif ticker_prefix in ['TQQQ', 'UPRO', 'UDOW', 'URTY', 'SOXL', 'FAS', 'LABU', 'NUGT', 'DRN', 'TMF']:
                    return 'Leveraged'
            
            return 'Unknown'
        
        # Extract underlying index
        def get_underlying_index(row):
            # Check benchmark field
            if 'FUND_BENCHMARK' in row and pd.notna(row['FUND_BENCHMARK']):
                benchmark = row['FUND_BENCHMARK']
                
                # Map common benchmark names to standardized names
                if isinstance(benchmark, str):
                    if any(x in benchmark.upper() for x in ['S&P 500', 'SPX']):
                        return 'S&P 500 Index'
                    elif any(x in benchmark.upper() for x in ['NASDAQ-100', 'NASDAQ 100', 'NDX']):
                        return 'NASDAQ-100 Index'
                    elif any(x in benchmark.upper() for x in ['DOW JONES', 'INDUSTRIAL AVERAGE', 'DJIA']):
                        return 'Dow Jones Industrial Average'
                    elif any(x in benchmark.upper() for x in ['RUSSELL 2000', 'RTY']):
                        return 'Russell 2000 Index'
                    elif any(x in benchmark.upper() for x in ['VIX', 'VOLATILITY']):
                        return 'S&P 500 VIX Index'
                    elif any(x in benchmark.upper() for x in ['SEMICONDUCTOR', 'SOX']):
                        return 'PHLX Semiconductor Index'
                    elif any(x in benchmark.upper() for x in ['MSCI EMERGING', 'MSCI EM']):
                        return 'MSCI Emerging Markets Index'
                    elif any(x in benchmark.upper() for x in ['MSCI EAFE']):
                        return 'MSCI EAFE Index'
                
                # Return the benchmark as is if no match
                return benchmark
            
            # Extract from name or description
            for field in ['SECURITY_DES', 'NAME', 'SHORT_NAME', 'LONG_COMP_NAME', 'FUND_OBJECTIVE']:
                if field in row and isinstance(row[field], str):
                    text = row[field].upper()
                    
                    # Check for major index patterns in the name
                    
                    # S&P 500
                    if 'S&P 500' in text or 'SPX' in text or 'SP500' in text:
                        if 'VIX' in text:
                            return 'S&P 500 VIX Index'
                        else:
                            return 'S&P 500 Index'
                    
                    # NASDAQ-100
                    elif 'NASDAQ-100' in text or 'NASDAQ 100' in text or 'NDX' in text or 'QQQ' in text:
                        return 'NASDAQ-100 Index'
                    
                    # Dow Jones
                    elif 'DOW JONES' in text or 'DOW 30' in text or 'DOW30' in text or 'INDU' in text or 'INDUSTRIAL AVERAGE' in text:
                        return 'Dow Jones Industrial Average'
                    
                    # Russell 2000
                    elif 'RUSSELL 2000' in text or 'RTY' in text or 'SMALL CAP' in text:
                        return 'Russell 2000 Index'
                    
                    # VIX
                    elif 'VIX' in text or 'VOLATILITY' in text:
                        return 'S&P 500 VIX Index'
                    
                    # Semiconductors
                    elif 'SEMICONDUCTOR' in text or 'SOX' in text:
                        return 'PHLX Semiconductor Index'
                    
                    # Financial Sector
                    elif 'FINANCIAL' in text:
                        return 'Financial Select Sector Index'
                    
                    # Energy Sector
                    elif 'ENERGY' in text and 'SECTOR' in text:
                        return 'Energy Select Sector Index'
                    
                    # Technology Sector
                    elif 'TECHNOLOGY' in text or 'TECH' in text:
                        return 'Technology Select Sector Index'
                    
                    # Healthcare Sector
                    elif 'HEALTH CARE' in text or 'HEALTHCARE' in text:
                        return 'Health Care Select Sector Index'
                    
                    # Real Estate
                    elif 'REAL ESTATE' in text:
                        return 'Real Estate Select Sector Index'
                    
                    # Biotech
                    elif 'BIOTECH' in text:
                        return 'S&P Biotech Select Index'
                    
                    # Gold Miners
                    elif 'GOLD MINERS' in text:
                        if 'JUNIOR' in text:
                            return 'Junior Gold Miners Index'
                        else:
                            return 'NYSE Arca Gold Miners Index'
                    
                    # Crude Oil
                    elif 'CRUDE OIL' in text:
                        return 'Bloomberg Crude Oil Index'
                    
                    # Natural Gas
                    elif 'NATURAL GAS' in text:
                        return 'Bloomberg Natural Gas Index'
                    
                    # MSCI Emerging Markets
                    elif 'MSCI EMERGING' in text or 'MSCI EM' in text:
                        return 'MSCI Emerging Markets Index'
                    
                    # MSCI EAFE
                    elif 'MSCI EAFE' in text:
                        return 'MSCI EAFE Index'
                    
                    # FANG+
                    elif 'FANG' in text:
                        return 'FANG+ Index'
            
            # Check ticker for common patterns
            ticker = row.get('TICKER', '')
            if isinstance(ticker, str):
                ticker_prefix = ticker.split()[0]
                
                # S&P 500 related ETFs
                if ticker_prefix in ['SPY', 'UPRO', 'SPXU', 'SPXL', 'SPXS', 'SSO', 'SDS']:
                    return 'S&P 500 Index'
                
                # NASDAQ-100 related ETFs
                elif ticker_prefix in ['QQQ', 'TQQQ', 'SQQQ', 'QLD', 'QID']:
                    return 'NASDAQ-100 Index'
                
                # Dow Jones related ETFs
                elif ticker_prefix in ['DIA', 'UDOW', 'SDOW', 'DDM', 'DXD']:
                    return 'Dow Jones Industrial Average'
                
                # Russell 2000 related ETFs
                elif ticker_prefix in ['IWM', 'URTY', 'SRTY', 'UWM', 'TWM']:
                    return 'Russell 2000 Index'
                
                # VIX related ETFs
                elif ticker_prefix in ['UVXY', 'TVIX', 'VIXY', 'SVXY']:
                    return 'S&P 500 VIX Index'
                
                # Semiconductor related ETFs
                elif ticker_prefix in ['SOXL', 'SOXS']:
                    return 'PHLX Semiconductor Index'
                
                # Financial sector
                elif ticker_prefix in ['FAS', 'FAZ', 'XLF']:
                    return 'Financial Select Sector Index'
                
                # Energy sector
                elif ticker_prefix in ['ERX', 'ERY']:
                    return 'Energy Select Sector Index'
                
                # Biotech sector
                elif ticker_prefix in ['LABU', 'LABD']:
                    return 'S&P Biotech Select Index'
                
                # Gold miners
                elif ticker_prefix in ['NUGT', 'DUST']:
                    return 'NYSE Arca Gold Miners Index'
                
                # Junior gold miners
                elif ticker_prefix in ['JNUG', 'JDST']:
                    return 'Junior Gold Miners Index'
                
                # Oil
                elif ticker_prefix in ['UCO', 'SCO']:
                    return 'Bloomberg Crude Oil Index'
                
                # Natural gas
                elif ticker_prefix in ['BOIL', 'KOLD']:
                    return 'Bloomberg Natural Gas Index'
                
                # FANG+
                elif ticker_prefix in ['FNGU', 'FNGD']:
                    return 'FANG+ Index'
            
            return 'Unknown'
        
        # Get Bloomberg underlying code
        def get_underlying_code(underlying_name):
            # Check if we have a direct mapping
            if underlying_name in self.underlying_code_map:
                return self.underlying_code_map[underlying_name]
                
            # If it's already in Bloomberg code format, return as is
            if ' Index' in underlying_name:
                return underlying_name
                
            # Try to extract from name
            for known_name, code in self.underlying_code_map.items():
                if known_name.replace(' Index', '').upper() in underlying_name.upper():
                    return code
                    
            # Default fallback - return the original with Index appended if needed
            if not ' Index' in underlying_name:
                return underlying_name + ' Index'
            return underlying_name
        
        # Extract leverage factor
        def get_leverage_factor(row):
            # Check leverage factor field
            if 'FUND_LEVERAGE_FACTOR' in row and pd.notna(row['FUND_LEVERAGE_FACTOR']):
                try:
                    return float(row['FUND_LEVERAGE_FACTOR'])
                except:
                    # Try to extract from string
                    if isinstance(row['FUND_LEVERAGE_FACTOR'], str):
                        matches = re.findall(r'([+-]?\d+(?:\.\d+)?)', row['FUND_LEVERAGE_FACTOR'])
                        if matches:
                            return float(matches[0])
            
            # Extract from name or description
            for field in ['SECURITY_DES', 'NAME', 'SHORT_NAME', 'LONG_COMP_NAME', 'FUND_OBJECTIVE']:
                if field in row and isinstance(row[field], str):
                    text = row[field].upper()
                    
                    # Look for patterns indicating leverage factor
                    if '3X' in text or 'TRIPLE' in text or 'ULTRAPRO' in text:
                        if 'SHORT' in text or 'INVERSE' in text or 'BEAR' in text:
                            return -3.0
                        else:
                            return 3.0
                    elif '2X' in text or 'DOUBLE' in text or 'ULTRA ' in text:
                        if 'SHORT' in text or 'INVERSE' in text or 'BEAR' in text:
                            return -2.0
                        else:
                            return 2.0
                    elif '-1X' in text or 'SHORT' in text or 'INVERSE' in text or 'BEAR' in text:
                        return -1.0
                    
                    # More general pattern
                    matches = re.findall(r'([+-]?\d+(?:\.\d+)?)[Xx]', text)
                    if matches:
                        return float(matches[0])
            
            # Determine from ticker
            ticker = row.get('TICKER', '')
            if isinstance(ticker, str):
                ticker_prefix = ticker.split()[0]
                
                # 3x Long ETFs
                if ticker_prefix in ['TQQQ', 'UPRO', 'UDOW', 'URTY', 'SOXL', 'SPXL', 'TECL', 'EDC', 'TNA', 'FNGU', 'LABU', 'TMF']:
                    return 3.0
                # 3x Short ETFs
                elif ticker_prefix in ['SQQQ', 'SPXU', 'SDOW', 'SRTY', 'SOXS', 'SPXS', 'TECS', 'EDZ', 'TZA', 'FNGD', 'LABD', 'TMV']:
                    return -3.0
                # 2x Long ETFs
                elif ticker_prefix in ['SSO', 'QLD', 'DDM', 'UWM', 'ROM', 'UYG', 'FAS', 'ERX', 'DRN', 'NUGT', 'UCO', 'BOIL']:
                    return 2.0
                # 2x Short ETFs
                elif ticker_prefix in ['SDS', 'QID', 'DXD', 'TWM', 'REW', 'SKF', 'FAZ', 'ERY', 'DRV', 'DUST', 'SCO', 'KOLD']:
                    return -2.0
            
            # Use leverage type as a fallback
            leverage_type = get_leverage_type(row)
            if leverage_type == 'Leveraged':
                return 2.0  # Most common leverage in US
            elif leverage_type == 'Inverse':
                return -1.0  # Most common inverse in US
            
            return 1.0  # Default to standard exposure
        
        # Extract AUM (Assets Under Management)
        def get_aum(row):
            # Try different fields for AUM
            for field in ['FUND_TOTAL_ASSETS', 'FUND_MKTCAP', 'FX_MARKET_CAP', 'FUND_NET_ASSET_VAL']:
                if field in row and pd.notna(row[field]):
                    try:
                        return float(row[field])
                    except:
                        pass
            
            return 0.0  # Default
        
        # Extract currency
        def get_currency(row):
            if 'EQY_FUND_CRNCY' in row and pd.notna(row['EQY_FUND_CRNCY']):
                return row['EQY_FUND_CRNCY']
            
            return 'USD'  # Default for US
        
        # Extract issuer name
        def get_issuer(row):
            # Try to get issuer from name
            name = row.get('NAME', '')
            if pd.isna(name) or not isinstance(name, str):
                name = row.get('SHORT_NAME', '')
            
            if pd.isna(name) or not isinstance(name, str):
                name = row.get('SECURITY_DES', '')
                
            if isinstance(name, str):
                name = name.upper()
                
                # Check for known US ETF issuers
                if 'PROSHARES' in name:
                    return 'ProShares'
                elif 'DIREXION' in name:
                    return 'Direxion'
                elif 'ISHARES' in name:
                    return 'iShares'
                elif 'INVESCO' in name:
                    return 'Invesco'
                elif 'VELOCITYSHARES' in name:
                    return 'VelocityShares'
                elif 'CREDIT SUISSE' in name:
                    return 'Credit Suisse'
                elif 'MICROSECTORS' in name:
                    return 'MicroSectors'
                elif 'BMO' in name:
                    return 'BMO'
                elif 'ETRACS' in name or 'UBS' in name:
                    return 'ETRACS'
                elif 'GRANITESHARES' in name:
                    return 'GraniteShares'
                
                # Extract from Ultra/UltraPro naming pattern
                if 'ULTRAPRO ' in name:
                    return 'ProShares'
                elif 'ULTRA ' in name and 'SHORT' not in name:
                    return 'ProShares'
                elif 'ULTRASHORT ' in name:
                    return 'ProShares'
                elif 'DAILY ' in name:
                    return 'Direxion'
            
            # Try to extract from ticker
            ticker = row.get('TICKER', '')
            if isinstance(ticker, str):
                ticker_prefix = ticker.split()[0]
                
                # ProShares tickers
                if ticker_prefix in ['UPRO', 'SPXU', 'TQQQ', 'SQQQ', 'UDOW', 'SDOW', 'SSO', 'SDS', 'UVXY', 'SVXY']:
                    return 'ProShares'
                
                # Direxion tickers
                elif ticker_prefix in ['SPXL', 'SPXS', 'SOXL', 'SOXS', 'FAS', 'FAZ', 'ERX', 'ERY', 'NUGT', 'DUST', 'JNUG', 'JDST']:
                    return 'Direxion'
                
                # MicroSectors tickers
                elif ticker_prefix in ['FNGU', 'FNGD', 'BULZ', 'BERZ']:
                    return 'MicroSectors'
            
            return 'Unknown'
        
        # Apply functions to extract details
        leveraged_inverse_df['ISSUER'] = leveraged_inverse_df.apply(get_issuer, axis=1)
        leveraged_inverse_df['LEVERAGE_TYPE'] = leveraged_inverse_df.apply(get_leverage_type, axis=1)
        leveraged_inverse_df['UNDERLYING'] = leveraged_inverse_df.apply(get_underlying_index, axis=1)
        leveraged_inverse_df['UNDERLYING_CODE'] = leveraged_inverse_df['UNDERLYING'].apply(get_underlying_code)
        leveraged_inverse_df['LEVERAGE_FACTOR'] = leveraged_inverse_df.apply(get_leverage_factor, axis=1)
        leveraged_inverse_df['AUM'] = leveraged_inverse_df.apply(get_aum, axis=1)
        leveraged_inverse_df['CURRENCY'] = leveraged_inverse_df.apply(get_currency, axis=1)
        
        logger.info(f"Processed {len(leveraged_inverse_df)} leveraged/inverse ETFs with details")
        return leveraged_inverse_df
    
    def get_sample_li_etfs(self) -> pd.DataFrame:
        """
        Create a sample dataset of US leveraged/inverse ETFs for testing.
        
        Returns:
            DataFrame with sample leveraged/inverse ETF data
        """
        logger.info("Creating sample leveraged/inverse ETF data for analysis")
        
        # Create sample data for US market
        data = [
            # ProShares S&P 500 ETFs
            {
                'TICKER': 'SSO US Equity',
                'NAME': 'ProShares Ultra S&P 500',
                'SECURITY_DES': 'ProShares Ultra S&P 500 (2x daily)',
                'SECURITY_TYP2': 'ETF',
                'ISSUER': 'ProShares',
                'LEVERAGE_TYPE': 'Leveraged',
                'UNDERLYING': 'S&P 500 Index',
                'UNDERLYING_CODE': 'SPX Index',
                'LEVERAGE_FACTOR': 2.0,
                'AUM': 3200000000.0,
                'CURRENCY': 'USD'
            },
            {
                'TICKER': 'UPRO US Equity',
                'NAME': 'ProShares UltraPro S&P 500',
                'SECURITY_DES': 'ProShares UltraPro S&P 500 (3x daily)',
                'SECURITY_TYP2': 'ETF',
                'ISSUER': 'ProShares',
                'LEVERAGE_TYPE': 'Leveraged',
                'UNDERLYING': 'S&P 500 Index',
                'UNDERLYING_CODE': 'SPX Index',
                'LEVERAGE_FACTOR': 3.0,
                'AUM': 2900000000.0,
                'CURRENCY': 'USD'
            },
            {
                'TICKER': 'SDS US Equity',
                'NAME': 'ProShares UltraShort S&P 500',
                'SECURITY_DES': 'ProShares UltraShort S&P 500 (-2x daily)',
                'SECURITY_TYP2': 'ETF',
                'ISSUER': 'ProShares',
                'LEVERAGE_TYPE': 'Inverse',
                'UNDERLYING': 'S&P 500 Index',
                'UNDERLYING_CODE': 'SPX Index',
                'LEVERAGE_FACTOR': -2.0,
                'AUM': 780000000.0,
                'CURRENCY': 'USD'
            },
            {
                'TICKER': 'SPXU US Equity',
                'NAME': 'ProShares UltraPro Short S&P 500',
                'SECURITY_DES': 'ProShares UltraPro Short S&P 500 (-3x daily)',
                'SECURITY_TYP2': 'ETF',
                'ISSUER': 'ProShares',
                'LEVERAGE_TYPE': 'Inverse',
                'UNDERLYING': 'S&P 500 Index',
                'UNDERLYING_CODE': 'SPX Index',
                'LEVERAGE_FACTOR': -3.0,
                'AUM': 650000000.0,
                'CURRENCY': 'USD'
            },
            
            # Direxion S&P 500 ETFs
            {
                'TICKER': 'SPXL US Equity',
                'NAME': 'Direxion Daily S&P 500 Bull 3X Shares',
                'SECURITY_DES': 'Direxion Daily S&P 500 Bull 3X Shares',
                'SECURITY_TYP2': 'ETF',
                'ISSUER': 'Direxion',
                'LEVERAGE_TYPE': 'Leveraged',
                'UNDERLYING': 'S&P 500 Index',
                'UNDERLYING_CODE': 'SPX Index',
                'LEVERAGE_FACTOR': 3.0,
                'AUM': 3100000000.0,
                'CURRENCY': 'USD'
            },
            {
                'TICKER': 'SPXS US Equity',
                'NAME': 'Direxion Daily S&P 500 Bear 3X Shares',
                'SECURITY_DES': 'Direxion Daily S&P 500 Bear 3X Shares',
                'SECURITY_TYP2': 'ETF',
                'ISSUER': 'Direxion',
                'LEVERAGE_TYPE': 'Inverse',
                'UNDERLYING': 'S&P 500 Index',
                'UNDERLYING_CODE': 'SPX Index',
                'LEVERAGE_FACTOR': -3.0,
                'AUM': 450000000.0,
                'CURRENCY': 'USD'
            },
            
            # ProShares NASDAQ-100 ETFs
            {
                'TICKER': 'QLD US Equity',
                'NAME': 'ProShares Ultra QQQ',
                'SECURITY_DES': 'ProShares Ultra QQQ (2x daily)',
                'SECURITY_TYP2': 'ETF',
                'ISSUER': 'ProShares',
                'LEVERAGE_TYPE': 'Leveraged',
                'UNDERLYING': 'NASDAQ-100 Index',
                'UNDERLYING_CODE': 'NDX Index',
                'LEVERAGE_FACTOR': 2.0,
                'AUM': 4800000000.0,
                'CURRENCY': 'USD'
            },
            {
                'TICKER': 'TQQQ US Equity',
                'NAME': 'ProShares UltraPro QQQ',
                'SECURITY_DES': 'ProShares UltraPro QQQ (3x daily)',
                'SECURITY_TYP2': 'ETF',
                'ISSUER': 'ProShares',
                'LEVERAGE_TYPE': 'Leveraged',
                'UNDERLYING': 'NASDAQ-100 Index',
                'UNDERLYING_CODE': 'NDX Index',
                'LEVERAGE_FACTOR': 3.0,
                'AUM': 16500000000.0,
                'CURRENCY': 'USD'
            },
            {
                'TICKER': 'QID US Equity',
                'NAME': 'ProShares UltraShort QQQ',
                'SECURITY_DES': 'ProShares UltraShort QQQ (-2x daily)',
                'SECURITY_TYP2': 'ETF',
                'ISSUER': 'ProShares',
                'LEVERAGE_TYPE': 'Inverse',
                'UNDERLYING': 'NASDAQ-100 Index',
                'UNDERLYING_CODE': 'NDX Index',
                'LEVERAGE_FACTOR': -2.0,
                'AUM': 320000000.0,
                'CURRENCY': 'USD'
            },
            {
                'TICKER': 'SQQQ US Equity',
                'NAME': 'ProShares UltraPro Short QQQ',
                'SECURITY_DES': 'ProShares UltraPro Short QQQ (-3x daily)',
                'SECURITY_TYP2': 'ETF',
                'ISSUER': 'ProShares',
                'LEVERAGE_TYPE': 'Inverse',
                'UNDERLYING': 'NASDAQ-100 Index',
                'UNDERLYING_CODE': 'NDX Index',
                'LEVERAGE_FACTOR': -3.0,
                'AUM': 4200000000.0,
                'CURRENCY': 'USD'
            },
            
            # ProShares Dow Jones ETFs
            {
                'TICKER': 'DDM US Equity',
                'NAME': 'ProShares Ultra Dow30',
                'SECURITY_DES': 'ProShares Ultra Dow30 (2x daily)',
                'SECURITY_TYP2': 'ETF',
                'ISSUER': 'ProShares',
                'LEVERAGE_TYPE': 'Leveraged',
                'UNDERLYING': 'Dow Jones Industrial Average',
                'UNDERLYING_CODE': 'INDU Index',
                'LEVERAGE_FACTOR': 2.0,
                'AUM': 260000000.0,
                'CURRENCY': 'USD'
            },
            {
                'TICKER': 'UDOW US Equity',
                'NAME': 'ProShares UltraPro Dow30',
                'SECURITY_DES': 'ProShares UltraPro Dow30 (3x daily)',
                'SECURITY_TYP2': 'ETF',
                'ISSUER': 'ProShares',
                'LEVERAGE_TYPE': 'Leveraged',
                'UNDERLYING': 'Dow Jones Industrial Average',
                'UNDERLYING_CODE': 'INDU Index',
                'LEVERAGE_FACTOR': 3.0,
                'AUM': 850000000.0,
                'CURRENCY': 'USD'
            },
            {
                'TICKER': 'DXD US Equity',
                'NAME': 'ProShares UltraShort Dow30',
                'SECURITY_DES': 'ProShares UltraShort Dow30 (-2x daily)',
                'SECURITY_TYP2': 'ETF',
                'ISSUER': 'ProShares',
                'LEVERAGE_TYPE': 'Inverse',
                'UNDERLYING': 'Dow Jones Industrial Average',
                'UNDERLYING_CODE': 'INDU Index',
                'LEVERAGE_FACTOR': -2.0,
                'AUM': 53000000.0,
                'CURRENCY': 'USD'
            },
            {
                'TICKER': 'SDOW US Equity',
                'NAME': 'ProShares UltraPro Short Dow30',
                'SECURITY_DES': 'ProShares UltraPro Short Dow30 (-3x daily)',
                'SECURITY_TYP2': 'ETF',
                'ISSUER': 'ProShares',
                'LEVERAGE_TYPE': 'Inverse',
                'UNDERLYING': 'Dow Jones Industrial Average',
                'UNDERLYING_CODE': 'INDU Index',
                'LEVERAGE_FACTOR': -3.0,
                'AUM': 135000000.0,
                'CURRENCY': 'USD'
            },
            
            # Sector ETFs
            {
                'TICKER': 'SOXL US Equity',
                'NAME': 'Direxion Daily Semiconductor Bull 3X Shares',
                'SECURITY_DES': 'Direxion Daily Semiconductor Bull 3X Shares',
                'SECURITY_TYP2': 'ETF',
                'ISSUER': 'Direxion',
                'LEVERAGE_TYPE': 'Leveraged',
                'UNDERLYING': 'PHLX Semiconductor Index',
                'UNDERLYING_CODE': 'SOX Index',
                'LEVERAGE_FACTOR': 3.0,
                'AUM': 5800000000.0,
                'CURRENCY': 'USD'
            },
            {
                'TICKER': 'SOXS US Equity',
                'NAME': 'Direxion Daily Semiconductor Bear 3X Shares',
                'SECURITY_DES': 'Direxion Daily Semiconductor Bear 3X Shares',
                'SECURITY_TYP2': 'ETF',
                'ISSUER': 'Direxion',
                'LEVERAGE_TYPE': 'Inverse',
                'UNDERLYING': 'PHLX Semiconductor Index',
                'UNDERLYING_CODE': 'SOX Index',
                'LEVERAGE_FACTOR': -3.0,
                'AUM': 450000000.0,
                'CURRENCY': 'USD'
            },
            {
                'TICKER': 'FAS US Equity',
                'NAME': 'Direxion Daily Financial Bull 3X Shares',
                'SECURITY_DES': 'Direxion Daily Financial Bull 3X Shares',
                'SECURITY_TYP2': 'ETF',
                'ISSUER': 'Direxion',
                'LEVERAGE_TYPE': 'Leveraged',
                'UNDERLYING': 'Financial Select Sector Index',
                'UNDERLYING_CODE': 'S5FINL Index',
                'LEVERAGE_FACTOR': 3.0,
                'AUM': 2300000000.0,
                'CURRENCY': 'USD'
            },
            {
                'TICKER': 'FAZ US Equity',
                'NAME': 'Direxion Daily Financial Bear 3X Shares',
                'SECURITY_DES': 'Direxion Daily Financial Bear 3X Shares',
                'SECURITY_TYP2': 'ETF',
                'ISSUER': 'Direxion',
                'LEVERAGE_TYPE': 'Inverse',
                'UNDERLYING': 'Financial Select Sector Index',
                'UNDERLYING_CODE': 'S5FINL Index',
                'LEVERAGE_FACTOR': -3.0,
                'AUM': 320000000.0,
                'CURRENCY': 'USD'
            },
            {
                'TICKER': 'LABU US Equity',
                'NAME': 'Direxion Daily S&P Biotech Bull 3X Shares',
                'SECURITY_DES': 'Direxion Daily S&P Biotech Bull 3X Shares',
                'SECURITY_TYP2': 'ETF',
                'ISSUER': 'Direxion',
                'LEVERAGE_TYPE': 'Leveraged',
                'UNDERLYING': 'S&P Biotech Select Index',
                'UNDERLYING_CODE': 'S5BIOTR Index',
                'LEVERAGE_FACTOR': 3.0,
                'AUM': 920000000.0,
                'CURRENCY': 'USD'
            },
            {
                'TICKER': 'LABD US Equity',
                'NAME': 'Direxion Daily S&P Biotech Bear 3X Shares',
                'SECURITY_DES': 'Direxion Daily S&P Biotech Bear 3X Shares',
                'SECURITY_TYP2': 'ETF',
                'ISSUER': 'Direxion',
                'LEVERAGE_TYPE': 'Inverse',
                'UNDERLYING': 'S&P Biotech Select Index',
                'UNDERLYING_CODE': 'S5BIOTR Index',
                'LEVERAGE_FACTOR': -3.0,
                'AUM': 68000000.0,
                'CURRENCY': 'USD'
            },
            
            # Commodity ETFs
            {
                'TICKER': 'NUGT US Equity',
                'NAME': 'Direxion Daily Gold Miners Bull 2X Shares',
                'SECURITY_DES': 'Direxion Daily Gold Miners Bull 2X Shares',
                'SECURITY_TYP2': 'ETF',
                'ISSUER': 'Direxion',
                'LEVERAGE_TYPE': 'Leveraged',
                'UNDERLYING': 'NYSE Arca Gold Miners Index',
                'UNDERLYING_CODE': 'GDM Index',
                'LEVERAGE_FACTOR': 2.0,
                'AUM': 950000000.0,
                'CURRENCY': 'USD'
            },
            {
                'TICKER': 'DUST US Equity',
                'NAME': 'Direxion Daily Gold Miners Bear 2X Shares',
                'SECURITY_DES': 'Direxion Daily Gold Miners Bear 2X Shares',
                'SECURITY_TYP2': 'ETF',
                'ISSUER': 'Direxion',
                'LEVERAGE_TYPE': 'Inverse',
                'UNDERLYING': 'NYSE Arca Gold Miners Index',
                'UNDERLYING_CODE': 'GDM Index',
                'LEVERAGE_FACTOR': -2.0,
                'AUM': 140000000.0,
                'CURRENCY': 'USD'
            },
            {
                'TICKER': 'UCO US Equity',
                'NAME': 'ProShares Ultra Bloomberg Crude Oil',
                'SECURITY_DES': 'ProShares Ultra Bloomberg Crude Oil (2x daily)',
                'SECURITY_TYP2': 'ETF',
                'ISSUER': 'ProShares',
                'LEVERAGE_TYPE': 'Leveraged',
                'UNDERLYING': 'Bloomberg Crude Oil Index',
                'UNDERLYING_CODE': 'BCOMCL Index',
                'LEVERAGE_FACTOR': 2.0,
                'AUM': 950000000.0,
                'CURRENCY': 'USD'
            },
            {
                'TICKER': 'SCO US Equity',
                'NAME': 'ProShares UltraShort Bloomberg Crude Oil',
                'SECURITY_DES': 'ProShares UltraShort Bloomberg Crude Oil (-2x daily)',
                'SECURITY_TYP2': 'ETF',
                'ISSUER': 'ProShares',
                'LEVERAGE_TYPE': 'Inverse',
                'UNDERLYING': 'Bloomberg Crude Oil Index',
                'UNDERLYING_CODE': 'BCOMCL Index',
                'LEVERAGE_FACTOR': -2.0,
                'AUM': 110000000.0,
                'CURRENCY': 'USD'
            },
            
            # VIX ETFs/ETNs
            {
                'TICKER': 'UVXY US Equity',
                'NAME': 'ProShares Ultra VIX Short-Term Futures ETF',
                'SECURITY_DES': 'ProShares Ultra VIX Short-Term Futures ETF (1.5x daily)',
                'SECURITY_TYP2': 'ETF',
                'ISSUER': 'ProShares',
                'LEVERAGE_TYPE': 'Leveraged',
                'UNDERLYING': 'S&P 500 VIX Index',
                'UNDERLYING_CODE': 'VIX Index',
                'LEVERAGE_FACTOR': 1.5,
                'AUM': 780000000.0,
                'CURRENCY': 'USD'
            },
            {
                'TICKER': 'SVXY US Equity',
                'NAME': 'ProShares Short VIX Short-Term Futures ETF',
                'SECURITY_DES': 'ProShares Short VIX Short-Term Futures ETF (-0.5x daily)',
                'SECURITY_TYP2': 'ETF',
                'ISSUER': 'ProShares',
                'LEVERAGE_TYPE': 'Inverse',
                'UNDERLYING': 'S&P 500 VIX Index',
                'UNDERLYING_CODE': 'VIX Index',
                'LEVERAGE_FACTOR': -0.5,
                'AUM': 420000000.0,
                'CURRENCY': 'USD'
            },
            
            # FANG+ ETNs
            {
                'TICKER': 'FNGU US Equity',
                'NAME': 'MicroSectors FANG+ Index 3X Leveraged ETN',
                'SECURITY_DES': 'MicroSectors FANG+ Index 3X Leveraged ETN',
                'SECURITY_TYP2': 'ETN',
                'ISSUER': 'MicroSectors',
                'LEVERAGE_TYPE': 'Leveraged',
                'UNDERLYING': 'FANG+ Index',
                'UNDERLYING_CODE': 'NYFANG Index',
                'LEVERAGE_FACTOR': 3.0,
                'AUM': 1200000000.0,
                'CURRENCY': 'USD'
            },
            {
                'TICKER': 'FNGD US Equity',
                'NAME': 'MicroSectors FANG+ Index -3X Inverse Leveraged ETN',
                'SECURITY_DES': 'MicroSectors FANG+ Index -3X Inverse Leveraged ETN',
                'SECURITY_TYP2': 'ETN',
                'ISSUER': 'MicroSectors',
                'LEVERAGE_TYPE': 'Inverse',
                'UNDERLYING': 'FANG+ Index',
                'UNDERLYING_CODE': 'NYFANG Index',
                'LEVERAGE_FACTOR': -3.0,
                'AUM': 65000000.0,
                'CURRENCY': 'USD'
            },
            
            # Unknown examples
            {
                'TICKER': 'UNKNOWN1 US Equity',
                'NAME': 'Unknown Leveraged ETF',
                'SECURITY_DES': 'Unknown Leveraged ETF Description',
                'SECURITY_TYP2': 'ETF',
                'ISSUER': 'Unknown',
                'LEVERAGE_TYPE': 'Leveraged',
                'UNDERLYING': 'Unknown',
                'UNDERLYING_CODE': 'Unknown',
                'LEVERAGE_FACTOR': 2.0,
                'AUM': 50000000.0,
                'CURRENCY': 'USD'
            },
            {
                'TICKER': 'UNKNOWN2 US Equity',
                'NAME': 'Unknown Inverse ETF',
                'SECURITY_DES': 'Unknown Inverse ETF Description',
                'SECURITY_TYP2': 'ETF',
                'ISSUER': 'Unknown',
                'LEVERAGE_TYPE': 'Inverse',
                'UNDERLYING': 'Unknown',
                'UNDERLYING_CODE': 'Unknown',
                'LEVERAGE_FACTOR': -2.0,
                'AUM': 25000000.0,
                'CURRENCY': 'USD'
            }
        ]
        
        return pd.DataFrame(data)
    
    def aggregate_by_underlying(self, etf_df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate ETF data by underlying index.
        
        Args:
            etf_df: DataFrame with ETF data
            
        Returns:
            DataFrame with aggregated data by underlying
        """
        if etf_df.empty:
            return pd.DataFrame()
        
        # Group by underlying and leverage type
        grouped = etf_df.groupby(['UNDERLYING', 'UNDERLYING_CODE', 'LEVERAGE_TYPE'])
        
        # Aggregate
        agg_data = grouped.agg({
            'TICKER': 'count',
            'AUM': 'sum'
        }).reset_index()
        
        # Rename columns
        agg_data = agg_data.rename(columns={
            'TICKER': 'NUM_FUNDS',
            'AUM': 'TOTAL_AUM'
        })
        
        # Sort by underlying and leverage type
        agg_data = agg_data.sort_values(['UNDERLYING', 'LEVERAGE_TYPE'])
        
        # Calculate percentage of total AUM
        total_aum = agg_data['TOTAL_AUM'].sum()
        agg_data['AUM_PCT'] = agg_data['TOTAL_AUM'] / total_aum * 100 if total_aum > 0 else 0
        
        return agg_data
    
    def aggregate_by_issuer(self, etf_df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate ETF data by issuer.
        
        Args:
            etf_df: DataFrame with ETF data
            
        Returns:
            DataFrame with aggregated data by issuer
        """
        if etf_df.empty:
            return pd.DataFrame()
        
        # Group by issuer
        grouped = etf_df.groupby('ISSUER')
        
        # Aggregate
        agg_data = grouped.agg({
            'TICKER': 'count',
            'AUM': 'sum'
        }).reset_index()
        
        # Rename columns
        agg_data = agg_data.rename(columns={
            'TICKER': 'NUM_FUNDS',
            'AUM': 'TOTAL_AUM'
        })
        
        # Sort by total AUM
        agg_data = agg_data.sort_values('TOTAL_AUM', ascending=False)
        
        # Calculate percentage of total AUM
        total_aum = agg_data['TOTAL_AUM'].sum()
        agg_data['AUM_PCT'] = agg_data['TOTAL_AUM'] / total_aum * 100 if total_aum > 0 else 0
        
        return agg_data
    
    def aggregate_by_leverage_factor(self, etf_df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate ETF data by leverage factor.
        
        Args:
            etf_df: DataFrame with ETF data
            
        Returns:
            DataFrame with aggregated data by leverage factor
        """
        if etf_df.empty:
            return pd.DataFrame()
        
        # Group by leverage factor
        grouped = etf_df.groupby('LEVERAGE_FACTOR')
        
        # Aggregate
        agg_data = grouped.agg({
            'TICKER': 'count',
            'AUM': 'sum'
        }).reset_index()
        
        # Rename columns
        agg_data = agg_data.rename(columns={
            'TICKER': 'NUM_FUNDS',
            'AUM': 'TOTAL_AUM'
        })
        
        # Sort by leverage factor (descending)
        agg_data = agg_data.sort_values('LEVERAGE_FACTOR', ascending=False)
        
        # Calculate percentage of total AUM
        total_aum = agg_data['TOTAL_AUM'].sum()
        agg_data['AUM_PCT'] = agg_data['TOTAL_AUM'] / total_aum * 100 if total_aum > 0 else 0
        
        return agg_data
    
    def aggregate_by_asset_class(self, etf_df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate ETF data by asset class.
        
        Args:
            etf_df: DataFrame with ETF data
            
        Returns:
            DataFrame with aggregated data by asset class
        """
        if etf_df.empty:
            return pd.DataFrame()
        
        # Function to determine asset class based on underlying name
        def get_asset_class(underlying):
            if pd.isna(underlying) or underlying == 'Unknown':
                return 'Unknown'
                
            # Broad equity indices
            if any(x in underlying for x in ['S&P 500', 'NASDAQ', 'Dow Jones', 'Russell', 'MSCI']):
                return 'Broad Equity'
            
            # Sector indices
            elif any(x in underlying for x in ['Sector', 'Financial', 'Technology', 'Health', 'Energy',
                                             'Materials', 'Utilities', 'Real Estate', 'Biotech',
                                             'Semiconductor', 'PHLX', 'Retail', 'Banks', 'FANG+']):
                return 'Sector'
            
            # Volatility
            elif 'VIX' in underlying or 'Volatility' in underlying:
                return 'Volatility'
            
            # Commodities
            elif any(x in underlying for x in ['Gold', 'Crude Oil', 'Natural Gas', 'Silver', 'Commodity']):
                return 'Commodities'
            
            # Fixed Income
            elif any(x in underlying for x in ['Treasury', 'Bond', 'High Yield', 'Interest Rate']):
                return 'Fixed Income'
            
            # International
            elif any(x in underlying for x in ['Emerging Markets', 'EAFE', 'China', 'Brazil', 'India']):
                return 'International'
            
            # Default case
            return 'Other'
        
        # Add asset class classification
        etf_df_with_class = etf_df.copy()
        etf_df_with_class['ASSET_CLASS'] = etf_df_with_class['UNDERLYING'].apply(get_asset_class)
        
        # Group by asset class and leverage type
        grouped = etf_df_with_class.groupby(['ASSET_CLASS', 'LEVERAGE_TYPE'])
        
        # Aggregate
        agg_data = grouped.agg({
            'TICKER': 'count',
            'AUM': 'sum'
        }).reset_index()
        
        # Rename columns
        agg_data = agg_data.rename(columns={
            'TICKER': 'NUM_FUNDS',
            'AUM': 'TOTAL_AUM'
        })
        
        # Sort by asset class first, then leverage type
        agg_data = agg_data.sort_values(['ASSET_CLASS', 'LEVERAGE_TYPE'])
        
        # Calculate percentage of total AUM
        total_aum = agg_data['TOTAL_AUM'].sum()
        agg_data['AUM_PCT'] = agg_data['TOTAL_AUM'] / total_aum * 100 if total_aum > 0 else 0
        
        return agg_data
    
    def track_li_funds(self) -> Dict[str, Any]:
        """
        Track leveraged and inverse funds in the US.
        
        Returns:
            Dictionary with fund data and analysis
        """
        # Find leveraged and inverse ETFs
        li_etfs = self.get_us_leveraged_inverse_funds()
        
        # Set latest update time
        self.latest_update_time = datetime.datetime.now()
        
        # Aggregate by underlying
        agg_by_underlying = self.aggregate_by_underlying(li_etfs)
        
        # Aggregate by issuer
        agg_by_issuer = self.aggregate_by_issuer(li_etfs)
        
        # Aggregate by leverage factor
        agg_by_leverage = self.aggregate_by_leverage_factor(li_etfs)
        
        # Aggregate by asset class
        agg_by_asset_class = self.aggregate_by_asset_class(li_etfs)
        
        # Store the results
        self.li_fund_data = {
            'Update_Time': self.latest_update_time,
            'ETF_List': li_etfs.to_dict('records') if not li_etfs.empty else [],
            'Agg_By_Underlying': agg_by_underlying.to_dict('records') if not agg_by_underlying.empty else [],
            'Agg_By_Issuer': agg_by_issuer.to_dict('records') if not agg_by_issuer.empty else [],
            'Agg_By_Leverage': agg_by_leverage.to_dict('records') if not agg_by_leverage.empty else [],
            'Agg_By_Asset_Class': agg_by_asset_class.to_dict('records') if not agg_by_asset_class.empty else []
        }
        
        return self.li_fund_data
    
    def format_aum(self, aum_value: float) -> str:
        """
        Format AUM value for display in USD.
        
        Args:
            aum_value: AUM value
            
        Returns:
            Formatted AUM string
        """
        if aum_value >= 1e9:
            return f"${aum_value/1e9:.2f}B"
        elif aum_value >= 1e6:
            return f"${aum_value/1e6:.2f}M"
        elif aum_value >= 1e3:
            return f"${aum_value/1e3:.2f}K"
        else:
            return f"${aum_value:.2f}"
    
    def highlight_unknown(self, value, ticker=""):
        """
        Highlight "Unknown" values in red with ticker where applicable.
        
        Args:
            value: The value to check
            ticker: Ticker to show alongside unknown value
            
        Returns:
            Formatted string with highlighting if needed
        """
        if value == 'Unknown':
            if ticker:
                return f"{RED}{BOLD}{FLASH}Unknown ({ticker}){RESET}"
            else:
                return f"{RED}{BOLD}{FLASH}Unknown{RESET}"
        return value
    
    def print_etf_list(self) -> None:
        """Print a list of leveraged and inverse ETFs."""
        if not self.li_fund_data or 'ETF_List' not in self.li_fund_data:
            logger.error("No ETF data available. Run track_li_funds() first.")
            return
        
        etf_list = self.li_fund_data.get('ETF_List', [])
        if not etf_list:
            logger.warning("No ETFs found to display.")
            return
        
        print("\n" + "="*120)
        print(f"LEVERAGED AND INVERSE ETFs IN THE US - {self.latest_update_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*120 + "\n")
        
        # Prepare table data
        table_data = []
        
        for etf in etf_list:
            ticker = etf.get('TICKER', '')
            name = etf.get('NAME', '')
            issuer = etf.get('ISSUER', '')
            underlying = etf.get('UNDERLYING', '')
            underlying_code = etf.get('UNDERLYING_CODE', '')
            leverage_type = etf.get('LEVERAGE_TYPE', '')
            leverage_factor = etf.get('LEVERAGE_FACTOR', '')
            aum = etf.get('AUM', 0)
            
            # Extract ticker code for highlighting
            ticker_code = ""
            if isinstance(ticker, str):
                ticker_match = re.search(r'^(\w+)', ticker)
                if ticker_match:
                    ticker_code = ticker_match.group(1)
            
            # Highlight unknown underlying and issuer
            highlighted_underlying = self.highlight_unknown(underlying, ticker_code)
            highlighted_code = self.highlight_unknown(underlying_code, ticker_code)
            highlighted_issuer = self.highlight_unknown(issuer, ticker_code)
            
            table_data.append([
                ticker,
                name,
                highlighted_issuer,
                highlighted_underlying,
                highlighted_code,
                leverage_type,
                leverage_factor,
                self.format_aum(aum)
            ])
        
        # Sort by issuer, then underlying, then leverage type
        try:
            table_data.sort(key=lambda x: (str(x[2]) if RED not in str(x[2]) else "ZZZZ", 
                                          str(x[3]) if RED not in str(x[3]) else "ZZZZ", 
                                          str(x[5])))
        except:
            # Fallback if sorting fails
            pass
        
        # Print the table
        headers = ["Ticker", "Name", "Issuer", "Underlying", "Bloomberg Code", "Type", "Factor", "AUM"]
        print(tabulate(table_data, headers=headers, tablefmt="psql"))
        print("\n" + "="*120 + "\n")
    
    def print_aum_by_underlying(self) -> None:
        """Print AUM aggregated by underlying index."""
        if not self.li_fund_data or 'Agg_By_Underlying' not in self.li_fund_data:
            logger.error("No aggregated data available. Run track_li_funds() first.")
            return
        
        agg_data = self.li_fund_data.get('Agg_By_Underlying', [])
        if not agg_data:
            logger.warning("No aggregated data found to display.")
            return
        
        print("\n" + "="*120)
        print(f"AUM BY UNDERLYING INDEX - {self.latest_update_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*120 + "\n")
        
        # Prepare table data
        table_data = []
        
        for row in agg_data:
            underlying = row.get('UNDERLYING', '')
            underlying_code = row.get('UNDERLYING_CODE', '')
            leverage_type = row.get('LEVERAGE_TYPE', '')
            num_funds = row.get('NUM_FUNDS', 0)
            total_aum = row.get('TOTAL_AUM', 0)
            aum_pct = row.get('AUM_PCT', 0)
            
            # Highlight unknown underlying
            highlighted_underlying = self.highlight_unknown(underlying)
            highlighted_code = self.highlight_unknown(underlying_code)
            
            table_data.append([
                highlighted_underlying,
                highlighted_code,
                leverage_type,
                num_funds,
                self.format_aum(total_aum),
                f"{aum_pct:.1f}%"
            ])
        
        # Print the table
        headers = ["Underlying", "Bloomberg Code", "Type", "# of Funds", "Total AUM", "% of Total"]
        print(tabulate(table_data, headers=headers, tablefmt="psql"))
        
        # Add a summary by underlying only (combining leveraged and inverse)
        print("\n" + "-"*120)
        print("SUMMARY BY UNDERLYING (ALL TYPES)")
        print("-"*120 + "\n")
        
        # Group by underlying
        underlying_summary = {}
        for row in agg_data:
            underlying = row.get('UNDERLYING', '')
            underlying_code = row.get('UNDERLYING_CODE', '')
            num_funds = row.get('NUM_FUNDS', 0)
            total_aum = row.get('TOTAL_AUM', 0)
            
            if underlying not in underlying_summary:
                underlying_summary[underlying] = {
                    'UNDERLYING_CODE': underlying_code,
                    'NUM_FUNDS': 0,
                    'TOTAL_AUM': 0
                }
            
            underlying_summary[underlying]['NUM_FUNDS'] += num_funds
            underlying_summary[underlying]['TOTAL_AUM'] += total_aum
        
        # Calculate percentages
        grand_total = sum(data['TOTAL_AUM'] for data in underlying_summary.values())
        
        # Prepare summary table
        summary_data = []
        for underlying, data in underlying_summary.items():
            aum_pct = data['TOTAL_AUM'] / grand_total * 100 if grand_total > 0 else 0
            
            # Highlight unknown underlying
            highlighted_underlying = self.highlight_unknown(underlying)
            highlighted_code = self.highlight_unknown(data['UNDERLYING_CODE'])
            
            summary_data.append([
                highlighted_underlying,
                highlighted_code,
                data['NUM_FUNDS'],
                self.format_aum(data['TOTAL_AUM']),
                f"{aum_pct:.1f}%"
            ])
        
        # Sort by total AUM
        try:
            summary_data.sort(key=lambda x: float(str(x[3]).replace('$', '').replace('B', 'e9').replace('M', 'e6').replace('K', 'e3')), reverse=True)
        except:
            # Fallback if sorting fails
            pass
        
        # Print the summary table
        headers = ["Underlying", "Bloomberg Code", "# of Funds", "Total AUM", "% of Total"]
        print(tabulate(summary_data, headers=headers, tablefmt="psql"))
        print("\n" + "="*120 + "\n")
    
    def print_aum_by_issuer(self) -> None:
        """Print AUM aggregated by issuer."""
        if not self.li_fund_data or 'Agg_By_Issuer' not in self.li_fund_data:
            logger.error("No aggregated data available. Run track_li_funds() first.")
            return
        
        agg_data = self.li_fund_data.get('Agg_By_Issuer', [])
        if not agg_data:
            logger.warning("No aggregated data found to display.")
            return
        
        print("\n" + "="*100)
        print(f"AUM BY ISSUER - {self.latest_update_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*100 + "\n")
        
        # Prepare table data
        table_data = []
        
        for row in agg_data:
            issuer = row.get('ISSUER', '')
            num_funds = row.get('NUM_FUNDS', 0)
            total_aum = row.get('TOTAL_AUM', 0)
            aum_pct = row.get('AUM_PCT', 0)
            
            # Highlight unknown issuer
            highlighted_issuer = self.highlight_unknown(issuer)
            
            table_data.append([
                highlighted_issuer,
                num_funds,
                self.format_aum(total_aum),
                f"{aum_pct:.1f}%"
            ])
        
        # Print the table
        headers = ["Issuer", "# of Funds", "Total AUM", "% of Total"]
        print(tabulate(table_data, headers=headers, tablefmt="psql"))
        print("\n" + "="*100 + "\n")
    
    def print_aum_by_leverage(self) -> None:
        """Print AUM aggregated by leverage factor."""
        if not self.li_fund_data or 'Agg_By_Leverage' not in self.li_fund_data:
            logger.error("No aggregated data available. Run track_li_funds() first.")
            return
        
        agg_data = self.li_fund_data.get('Agg_By_Leverage', [])
        if not agg_data:
            logger.warning("No aggregated data found to display.")
            return
        
        print("\n" + "="*100)
        print(f"AUM BY LEVERAGE FACTOR - {self.latest_update_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*100 + "\n")
        
        # Prepare table data
        table_data = []
        
        for row in agg_data:
            leverage_factor = row.get('LEVERAGE_FACTOR', 0)
            num_funds = row.get('NUM_FUNDS', 0)
            total_aum = row.get('TOTAL_AUM', 0)
            aum_pct = row.get('AUM_PCT', 0)
            
            # Format leverage factor
            if leverage_factor > 0:
                leverage_display = f"+{leverage_factor}x"
            else:
                leverage_display = f"{leverage_factor}x"
            
            table_data.append([
                leverage_display,
                num_funds,
                self.format_aum(total_aum),
                f"{aum_pct:.1f}%"
            ])
        
        # Print the table
        headers = ["Leverage Factor", "# of Funds", "Total AUM", "% of Total"]
        print(tabulate(table_data, headers=headers, tablefmt="psql"))
        print("\n" + "="*100 + "\n")
    
    def print_aum_by_asset_class(self) -> None:
        """Print AUM aggregated by asset class."""
        if not self.li_fund_data or 'Agg_By_Asset_Class' not in self.li_fund_data:
            logger.error("No aggregated data available. Run track_li_funds() first.")
            return
        
        agg_data = self.li_fund_data.get('Agg_By_Asset_Class', [])
        if not agg_data:
            logger.warning("No aggregated data found to display.")
            return
        
        print("\n" + "="*100)
        print(f"AUM BY ASSET CLASS - {self.latest_update_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*100 + "\n")
        
        # Prepare table data
        table_data = []
        
        for row in agg_data:
            asset_class = row.get('ASSET_CLASS', '')
            leverage_type = row.get('LEVERAGE_TYPE', '')
            num_funds = row.get('NUM_FUNDS', 0)
            total_aum = row.get('TOTAL_AUM', 0)
            aum_pct = row.get('AUM_PCT', 0)
            
            # Highlight unknown asset class
            if asset_class == 'Unknown':
                asset_class = self.highlight_unknown(asset_class)
            
            table_data.append([
                asset_class,
                leverage_type,
                num_funds,
                self.format_aum(total_aum),
                f"{aum_pct:.1f}%"
            ])
        
        # Print the table
        headers = ["Asset Class", "Type", "# of Funds", "Total AUM", "% of Total"]
        print(tabulate(table_data, headers=headers, tablefmt="psql"))
        
        # Add a summary by asset class only (combining leveraged and inverse)
        print("\n" + "-"*100)
        print("SUMMARY BY ASSET CLASS (ALL TYPES)")
        print("-"*100 + "\n")
        
        # Group by asset class
        class_summary = {}
        for row in agg_data:
            asset_class = row.get('ASSET_CLASS', '')
            num_funds = row.get('NUM_FUNDS', 0)
            total_aum = row.get('TOTAL_AUM', 0)
            
            if asset_class not in class_summary:
                class_summary[asset_class] = {
                    'NUM_FUNDS': 0,
                    'TOTAL_AUM': 0
                }
            
            class_summary[asset_class]['NUM_FUNDS'] += num_funds
            class_summary[asset_class]['TOTAL_AUM'] += total_aum
        
        # Calculate percentages
        grand_total = sum(data['TOTAL_AUM'] for data in class_summary.values())
        
        # Prepare summary table
        summary_data = []
        for asset_class, data in class_summary.items():
            aum_pct = data['TOTAL_AUM'] / grand_total * 100 if grand_total > 0 else 0
            
            # Highlight unknown asset class
            highlighted_class = self.highlight_unknown(asset_class) if asset_class == 'Unknown' else asset_class
            
            summary_data.append([
                highlighted_class,
                data['NUM_FUNDS'],
                self.format_aum(data['TOTAL_AUM']),
                f"{aum_pct:.1f}%"
            ])
        
        # Sort by total AUM
        try:
            summary_data.sort(key=lambda x: float(str(x[2]).replace('$', '').replace('B', 'e9').replace('M', 'e6').replace('K', 'e3')), reverse=True)
        except:
            # Fallback if sorting fails
            pass
        
        # Print the summary table
        headers = ["Asset Class", "# of Funds", "Total AUM", "% of Total"]
        print(tabulate(summary_data, headers=headers, tablefmt="psql"))
        print("\n" + "="*100 + "\n")
    
    def top_etfs_by_aum(self, n: int = 20) -> None:
        """
        Print top N ETFs by AUM.
        
        Args:
            n: Number of ETFs to display
        """
        if not self.li_fund_data or 'ETF_List' not in self.li_fund_data:
            logger.error("No ETF data available. Run track_li_funds() first.")
            return
        
        etf_list = self.li_fund_data.get('ETF_List', [])
        if not etf_list:
            logger.warning("No ETFs found to display.")
            return
        
        print("\n" + "="*120)
        print(f"TOP {n} LEVERAGED/INVERSE ETFs BY AUM - {self.latest_update_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*120 + "\n")
        
        # Convert to DataFrame if not already
        if isinstance(etf_list, list):
            etf_df = pd.DataFrame(etf_list)
        else:
            etf_df = etf_list.copy()
        
        # Sort by AUM descending
        etf_df = etf_df.sort_values('AUM', ascending=False)
        
        # Take top N
        top_etfs = etf_df.head(n)
        
        # Prepare table data
        table_data = []
        
        for _, etf in top_etfs.iterrows():
            ticker = etf.get('TICKER', '')
            name = etf.get('NAME', '')
            issuer = etf.get('ISSUER', '')
            underlying = etf.get('UNDERLYING', '')
            leverage_type = etf.get('LEVERAGE_TYPE', '')
            leverage_factor = etf.get('LEVERAGE_FACTOR', '')
            aum = etf.get('AUM', 0)
            
            # Highlight unknown values
            highlighted_underlying = self.highlight_unknown(underlying)
            highlighted_issuer = self.highlight_unknown(issuer)
            
            table_data.append([
                ticker,
                name,
                highlighted_issuer,
                highlighted_underlying,
                leverage_type,
                leverage_factor,
                self.format_aum(aum)
            ])
        
        # Print the table
        headers = ["Ticker", "Name", "Issuer", "Underlying", "Type", "Factor", "AUM"]
        print(tabulate(table_data, headers=headers, tablefmt="psql"))
        print("\n" + "="*120 + "\n")
    
    def save_results_to_csv(self, output_dir: str = None) -> None:
        """
        Save tracking results to CSV files.
        
        Args:
            output_dir: Directory to save output files (optional)
        """
        if not self.li_fund_data or 'ETF_List' not in self.li_fund_data:
            logger.error("No ETF data available. Run track_li_funds() first.")
            return
        
        if not output_dir:
            output_dir = './'
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = self.latest_update_time.strftime("%Y%m%d_%H%M%S")
        
        # Save ETF list
        etf_list = self.li_fund_data.get('ETF_List', [])
        if etf_list:
            etf_file = os.path.join(output_dir, f"us_li_etfs_{timestamp}.csv")
            
            # Convert to DataFrame if it's a list of dictionaries
            if isinstance(etf_list, list):
                etf_df = pd.DataFrame(etf_list)
            else:
                etf_df = pd.DataFrame([])
            
            if not etf_df.empty:
                # Save to CSV
                etf_df.to_csv(etf_file, index=False)
                logger.info(f"Saved ETF list to {etf_file}")
            else:
                logger.warning("No ETF data to save")
        
        # Save aggregated data by underlying
        agg_underlying = self.li_fund_data.get('Agg_By_Underlying', [])
        if agg_underlying:
            agg_file = os.path.join(output_dir, f"us_li_agg_by_underlying_{timestamp}.csv")
            
            # Convert to DataFrame if it's a list of dictionaries
            if isinstance(agg_underlying, list):
                agg_df = pd.DataFrame(agg_underlying)
            else:
                agg_df = pd.DataFrame([])
            
            if not agg_df.empty:
                # Save to CSV
                agg_df.to_csv(agg_file, index=False)
                logger.info(f"Saved aggregated data by underlying to {agg_file}")
            else:
                logger.warning("No aggregated data by underlying to save")
        
        # Save aggregated data by issuer
        agg_issuer = self.li_fund_data.get('Agg_By_Issuer', [])
        if agg_issuer:
            agg_file = os.path.join(output_dir, f"us_li_agg_by_issuer_{timestamp}.csv")
            
            # Convert to DataFrame if it's a list of dictionaries
            if isinstance(agg_issuer, list):
                agg_df = pd.DataFrame(agg_issuer)
            else:
                agg_df = pd.DataFrame([])
            
            if not agg_df.empty:
                # Save to CSV
                agg_df.to_csv(agg_file, index=False)
                logger.info(f"Saved aggregated data by issuer to {agg_file}")
            else:
                logger.warning("No aggregated data by issuer to save")
        
        # Save aggregated data by leverage factor
        agg_leverage = self.li_fund_data.get('Agg_By_Leverage', [])
        if agg_leverage:
            agg_file = os.path.join(output_dir, f"us_li_agg_by_leverage_{timestamp}.csv")
            
            # Convert to DataFrame if it's a list of dictionaries
            if isinstance(agg_leverage, list):
                agg_df = pd.DataFrame(agg_leverage)
            else:
                agg_df = pd.DataFrame([])
            
            if not agg_df.empty:
                # Save to CSV
                agg_df.to_csv(agg_file, index=False)
                logger.info(f"Saved aggregated data by leverage factor to {agg_file}")
            else:
                logger.warning("No aggregated data by leverage factor to save")
        
        # Save aggregated data by asset class
        agg_asset_class = self.li_fund_data.get('Agg_By_Asset_Class', [])
        if agg_asset_class:
            agg_file = os.path.join(output_dir, f"us_li_agg_by_asset_class_{timestamp}.csv")
            
            # Convert to DataFrame if it's a list of dictionaries
            if isinstance(agg_asset_class, list):
                agg_df = pd.DataFrame(agg_asset_class)
            else:
                agg_df = pd.DataFrame([])
            
            if not agg_df.empty:
                # Save to CSV
                agg_df.to_csv(agg_file, index=False)
                logger.info(f"Saved aggregated data by asset class to {agg_file}")
            else:
                logger.warning("No aggregated data by asset class to save")


def main():
    parser = argparse.ArgumentParser(description='Track and analyze Leveraged and Inverse ETFs in the US')
    parser.add_argument('--host', default='127.0.0.1', help='Bloomberg server host')
    parser.add_argument('--port', type=int, default=8194, help='Bloomberg server port')
    parser.add_argument('--output-dir', default='./us_li_funds', help='Directory to save output files')
    parser.add_argument('--sample', action='store_true', 
                        help='Use sample data instead of Bloomberg data')
    parser.add_argument('--top', type=int, default=20,
                        help='Number of top ETFs by AUM to display')
    
    args = parser.parse_args()
    
    # Create output directory if needed
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize tracker
    tracker = USLeverageFundsTracker(host=args.host, port=args.port, use_sample_data=args.sample)
    
    try:
        # Start session
        if tracker.start_session():
            # Track leveraged and inverse funds
            logger.info("Tracking leveraged and inverse funds in the US...")
            tracker.track_li_funds()
            
            # Print top ETFs by AUM
            tracker.top_etfs_by_aum(n=args.top)
            
            # Print ETF list
            tracker.print_etf_list()
            
            # Print AUM by underlying
            tracker.print_aum_by_underlying()
            
            # Print AUM by asset class
            tracker.print_aum_by_asset_class()
            
            # Print AUM by issuer
            tracker.print_aum_by_issuer()
            
            # Print AUM by leverage factor
            tracker.print_aum_by_leverage()
            
            # Save results
            logger.info("Saving results to CSV...")
            tracker.save_results_to_csv(output_dir=args.output_dir)
            
            logger.info("Leveraged and inverse funds tracking completed.")
    
    except KeyboardInterrupt:
        logger.info("Tracking interrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        import traceback
        logger.error(traceback.format_exc())
    finally:
        # Always stop the session
        tracker.stop_session()


if __name__ == "__main__":
    main()