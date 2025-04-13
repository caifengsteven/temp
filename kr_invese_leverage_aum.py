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
        logging.FileHandler("kr_leverage_funds_tracker.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('KRLeverageFundsTracker')

class KRLeverageFundsTracker:
    """
    A class to track and analyze Leveraged and Inverse ETFs/ETNs in Korean markets.
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
        
        # Major Korean indices for leveraged/inverse products
        self.kr_indices = [
            'KOSPI Index',    # Korea Composite Stock Price Index
            'KOSPI2 Index',   # KOSPI 200 Index
            'KOSDAQ Index',   # KOSDAQ Composite Index
            'KRX100 Index',   # KRX 100 Index
            'KRX300 Index'    # KRX 300 Index
        ]
        
        # Major Korean issuers of leveraged/inverse products
        self.kr_li_issuers = [
            'KODEX',
            'TIGER',
            'KBSTAR',
            'KOSEF',
            'ARIRANG',
            'KINDEX',
            'HANARO',
            'TIMEFOLIO',
            'SOL'
        ]
        
        # Mapping of underlying names to Bloomberg codes
        self.underlying_code_map = {
            'KOSPI Index': 'KOSPI Index',
            'KOSPI2 Index': 'KOSPI2 Index',
            'KOSDAQ Index': 'KOSDAQ Index',
            'KRX100 Index': 'KRX100 Index',
            'KRX300 Index': 'KRX300 Index',
            'MSCI Korea Index': 'MXKR Index',
            'S&P Korea Index': 'SPSKRP Index',
            'KOSPI 200 IT Index': 'K200IT Index',
            'KOSPI 200 Banks Index': 'K200BK Index',
            'KOSPI 200 Financials Index': 'K200FN Index',
            'Korea Semiconductor Index': 'KSEMIC Index',
            'KOSPI 200 Autos Index': 'K200AU Index',
            'KOSPI 200 Healthcare Index': 'K200HC Index',
            'KOSDAQ 150 Index': 'KOSDQ150 Index',
            'KOSPI 200 Energy & Chemicals Index': 'K200EC Index',
            'KOSPI 200 Consumer Discretionary Index': 'K200CD Index',
            'KOSPI 200 Consumer Staples Index': 'K200CS Index'
        }
        
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
    
    def get_kr_leveraged_inverse_funds(self) -> pd.DataFrame:
        """
        Get leveraged and inverse ETFs listed in Korea using direct ticker lookup.
        
        Returns:
            DataFrame with leveraged/inverse ETF details
        """
        if self.use_sample_data:
            # Return sample data in sample mode
            return self.get_sample_li_etfs()
        
        # Method: Use patterns for Korean leveraged/inverse ETFs
        leveraged_inverse_patterns = []
        
        # Collect known ETF patterns for major Korean ETF issuers
        # Korean ETFs use KS Equity or KQ Equity suffix, and often have numeric codes
        for issuer in self.kr_li_issuers:
            leveraged_inverse_patterns.append(f"{issuer} LEV* KS Equity")
            leveraged_inverse_patterns.append(f"{issuer} INV* KS Equity")
            leveraged_inverse_patterns.append(f"{issuer} *레버리지* KS Equity")
            leveraged_inverse_patterns.append(f"{issuer} *인버스* KS Equity")
            leveraged_inverse_patterns.append(f"{issuer} *선물* KS Equity")
            leveraged_inverse_patterns.append(f"{issuer} *2X* KS Equity")
            leveraged_inverse_patterns.append(f"{issuer} *-1X* KS Equity")
        
        # Korean ETFs often use numeric codes
        # Generate common ticker ranges for Korean leveraged/inverse ETFs
        # Common codes for KRX leveraged and inverse ETFs are in 100-300 range
        known_tickers = []
        for i in range(100, 350):
            known_tickers.append(f"{i} KS Equity")
            
        # Popular ETFs by code
        popular_kr_etfs = [
            '069500 KS Equity',  # KODEX 200
            '114800 KS Equity',  # KODEX Inverse
            '122630 KS Equity',  # KODEX Leverage
            '233740 KS Equity',  # KODEX 200 Futures Leverage
            '252670 KS Equity',  # KODEX 200 Futures Inverse
            '251340 KS Equity',  # KODEX Kosdaq150 Leverage
            '251350 KS Equity',  # KODEX Kosdaq150 Inverse
            '105010 KS Equity',  # TIGER 200 
            '123310 KS Equity',  # TIGER 200 Leverage
            '123320 KS Equity',  # TIGER 200 Inverse
            '151510 KS Equity',  # TIGER Leverage
            '214980 KS Equity',  # TIGER Inverse
            '230480 KS Equity',  # TIGER Futures Leverage
            '275980 KS Equity',  # TIGER Futures Inverse 2X
            '293180 KS Equity',  # TIGER Futures Leverage 3X
            '272560 KS Equity',  # KBSTAR 200
            '272570 KS Equity',  # KBSTAR 200 Leverage
            '272580 KS Equity',  # KBSTAR 200 Inverse
            '276990 KS Equity',  # KBSTAR 200 Leverage 3X
            '276980 KS Equity',  # KBSTAR 200 Inverse 3X
            '278530 KS Equity',  # KBSTAR 200 Futures Leverage
            '278540 KS Equity',  # KBSTAR 200 Futures Inverse
            '285010 KS Equity',  # ARIRANG 200 Leverage
            '285020 KS Equity',  # ARIRANG 200 Inverse
            '117700 KS Equity',  # KODEX Leverage KOSDAQ150
            '252400 KS Equity',  # KODEX Inverse KOSDAQ150
            '261220 KS Equity',  # KODEX 200 Leverage 2X
            '252710 KS Equity',  # KODEX 200 Inverse 2X
        ]
        
        known_tickers.extend(popular_kr_etfs)
        
        # Another approach: Scan among all securities with specific code patterns
        # Korean ETFs often have 6-digit codes
        # Scan for codes that often represent leveraged/inverse ETFs
        kodex_ranges = list(range(114800, 114900))  # KODEX Inverse range
        kodex_ranges.extend(list(range(122630, 122700)))  # KODEX Leverage range 
        kodex_ranges.extend(list(range(233740, 233800)))  # KODEX 200 Futures Leverage
        kodex_ranges.extend(list(range(252670, 252700)))  # KODEX 200 Futures Inverse
        
        tiger_ranges = list(range(123310, 123350))  # TIGER Leveraged/Inverse
        tiger_ranges.extend(list(range(230480, 230500)))  # TIGER Futures Leverage
        
        kbstar_ranges = list(range(272570, 272600))  # KBSTAR Leverage/Inverse
        kbstar_ranges.extend(list(range(276980, 277000)))  # KBSTAR 3X range
        
        # Combine all ranges
        for code in kodex_ranges + tiger_ranges + kbstar_ranges:
            known_tickers.append(f"{code} KS Equity")
        
        # Remove duplicates
        known_tickers = list(set(known_tickers))
        
        logger.info(f"Checking {len(known_tickers)} potential ETF tickers")
        
        # Get data for all these tickers
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
        
        all_etfs = self.get_security_data(known_tickers, fields)
        
        if all_etfs.empty:
            logger.warning("No ETF data retrieved. Using sample data.")
            return self.get_sample_li_etfs()
        
        # Filter for likely leveraged/inverse funds
        def is_leveraged_inverse(row):
            # Check security description and name
            for field in ['SECURITY_DES', 'NAME', 'SHORT_NAME', 'LONG_COMP_NAME', 'FUND_OBJECTIVE']:
                if field in row and isinstance(row[field], str):
                    text = row[field].upper()
                    
                    # Look for Korean and English leveraged/inverse keywords
                    if any(kw in text for kw in ['LEVERAGED', 'INVERSE', 'LEVERAGE', '레버리지', '인버스', '2X', '3X', '-1X', '-2X', '-3X']):
                        return True
                    
                    # Special terms used in Korean market
                    if '선물' in text and ('롱' in text or '숏' in text):  # Futures with long or short
                        return True
                    
                    # Sometimes they use 'bull' and 'bear' terminology
                    if any(kw in text for kw in ['BULL', 'BEAR']):
                        return True
            
            # Check leverage factor
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
            
            return False
        
        # Apply filter
        leveraged_inverse_df = all_etfs[all_etfs.apply(is_leveraged_inverse, axis=1)].copy()
        
        if leveraged_inverse_df.empty:
            logger.warning("No leveraged/inverse ETFs identified after filtering. Using sample data.")
            return self.get_sample_li_etfs()
        
        logger.info(f"Identified {len(leveraged_inverse_df)} leveraged/inverse ETFs")
        
        # Print a few examples of what we found for debugging
        logger.info(f"Sample of identified ETFs: {leveraged_inverse_df[['TICKER', 'NAME']].head(3).to_dict('records')}")
        
        # Extract leverage type (leveraged or inverse)
        def get_leverage_type(row):
            for field in ['SECURITY_DES', 'NAME', 'SHORT_NAME', 'LONG_COMP_NAME', 'FUND_OBJECTIVE']:
                if field in row and isinstance(row[field], str):
                    text = row[field].upper()
                    
                    if any(kw in text for kw in ['INVERSE', '-1X', '-2X', '-3X', '인버스', 'BEAR', '숏']):
                        return 'Inverse'
                    if any(kw in text for kw in ['LEVERAGED', 'LEVERAGE', '2X', '3X', '레버리지', 'BULL', '롱']):
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
            
            return 'Unknown'
        
        # Extract underlying index
        def get_underlying_index(row):
            # Check benchmark field
            if 'FUND_BENCHMARK' in row and pd.notna(row['FUND_BENCHMARK']):
                benchmark = row['FUND_BENCHMARK']
                
                # Map common Korean benchmark names to standardized names
                if isinstance(benchmark, str):
                    if any(x in benchmark.upper() for x in ['KOSPI 200', 'KOSPI2', 'KSP2']):
                        return 'KOSPI2 Index'
                    elif any(x in benchmark.upper() for x in ['KOSPI']):
                        return 'KOSPI Index'
                    elif any(x in benchmark.upper() for x in ['KOSDAQ', 'KQ']):
                        return 'KOSDAQ Index'
                    elif 'KRX100' in benchmark.upper():
                        return 'KRX100 Index'
                    elif 'KRX300' in benchmark.upper():
                        return 'KRX300 Index'
                    elif 'MSCI KOREA' in benchmark.upper():
                        return 'MSCI Korea Index'
                
                return benchmark
            
            # Extract from name or description
            for field in ['SECURITY_DES', 'NAME', 'SHORT_NAME', 'LONG_COMP_NAME']:
                if field in row and isinstance(row[field], str):
                    text = row[field].upper()
                    
                    if '코스피 200' in text or 'KOSPI 200' in text or 'KOSPI2' in text:
                        return 'KOSPI2 Index'
                    elif '코스피' in text and '200' not in text:
                        return 'KOSPI Index'
                    elif '코스닥' in text or 'KOSDAQ' in text:
                        return 'KOSDAQ Index'
                    elif '코스닥 150' in text or 'KOSDAQ 150' in text:
                        return 'KOSDAQ 150 Index'
                    elif 'KRX100' in text:
                        return 'KRX100 Index'
                    elif 'KRX300' in text:
                        return 'KRX300 Index'
                    elif '반도체' in text or 'SEMICONDUCTOR' in text:
                        return 'Korea Semiconductor Index'
                    elif 'IT' in text:
                        return 'KOSPI 200 IT Index'
                    
            
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
            for field in ['SECURITY_DES', 'NAME', 'SHORT_NAME', 'LONG_COMP_NAME']:
                if field in row and isinstance(row[field], str):
                    text = row[field].upper()
                    
                    # Look for patterns like "2X" or "-1X"
                    if '3X' in text:
                        return 3.0
                    elif '2X' in text:
                        return 2.0
                    elif '-3X' in text:
                        return -3.0
                    elif '-2X' in text:
                        return -2.0
                    elif '-1X' in text:
                        return -1.0
                    
                    # More general pattern
                    matches = re.findall(r'([+-]?\d+(?:\.\d+)?)[Xx]', text)
                    if matches:
                        return float(matches[0])
            
            # Default based on type
            leverage_type = get_leverage_type(row)
            if leverage_type == 'Leveraged':
                return 2.0  # Typical leverage
            elif leverage_type == 'Inverse':
                return -1.0  # Typical inverse
            
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
            
            return 'KRW'  # Default for Korea
        
        # Extract issuer name
        def get_issuer(row):
            # Try to get issuer from name
            ticker = row.get('TICKER', '')
            name = row.get('NAME', '')
            if pd.isna(name) or not isinstance(name, str):
                name = row.get('SHORT_NAME', '')
            
            if pd.isna(name) or not isinstance(name, str):
                name = row.get('SECURITY_DES', '')
                
            if isinstance(name, str):
                name = name.upper()
                
                # Check for known Korean ETF issuers
                for issuer in self.kr_li_issuers:
                    if issuer.upper() in name:
                        return issuer
                
                # Try to extract first word as issuer
                parts = name.split()
                if parts:
                    return parts[0]
            
            # Try to extract from ticker
            if isinstance(ticker, str):
                ticker_match = re.search(r'^(\d+)', ticker)
                if ticker_match:
                    code = ticker_match.group(1)
                    
                    # Map common code ranges to issuers
                    code_int = int(code)
                    if code_int >= 114800 and code_int <= 114900:
                        return 'KODEX'
                    elif code_int >= 122630 and code_int <= 122700:
                        return 'KODEX'
                    elif code_int >= 233740 and code_int <= 233800:
                        return 'KODEX'
                    elif code_int >= 123310 and code_int <= 123350:
                        return 'TIGER'
                    elif code_int >= 272570 and code_int <= 272600:
                        return 'KBSTAR'
            
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
        Create a sample dataset of Korean leveraged/inverse ETFs for testing.
        
        Returns:
            DataFrame with sample leveraged/inverse ETF data
        """
        logger.info("Creating sample leveraged/inverse ETF data for analysis")
        
        # Create sample data
        data = [
            # KOSPI 200 leveraged
            {
                'TICKER': '122630 KS Equity',
                'NAME': 'KODEX Leverage',
                'SECURITY_DES': 'KODEX 200 Leverage',
                'SECURITY_TYP2': 'ETF',
                'ISSUER': 'KODEX',
                'LEVERAGE_TYPE': 'Leveraged',
                'UNDERLYING': 'KOSPI2 Index',
                'UNDERLYING_CODE': 'KOSPI2 Index',
                'LEVERAGE_FACTOR': 2.0,
                'AUM': 1250000000000.0,
                'CURRENCY': 'KRW'
            },
            {
                'TICKER': '233740 KS Equity',
                'NAME': 'KODEX 200 Futures Leverage',
                'SECURITY_DES': 'KODEX 200 Futures Leverage ETF',
                'SECURITY_TYP2': 'ETF',
                'ISSUER': 'KODEX',
                'LEVERAGE_TYPE': 'Leveraged',
                'UNDERLYING': 'KOSPI2 Index',
                'UNDERLYING_CODE': 'KOSPI2 Index',
                'LEVERAGE_FACTOR': 2.0,
                'AUM': 950000000000.0,
                'CURRENCY': 'KRW'
            },
            {
                'TICKER': '123310 KS Equity',
                'NAME': 'TIGER 200 Leverage',
                'SECURITY_DES': 'TIGER 200 Leverage',
                'SECURITY_TYP2': 'ETF',
                'ISSUER': 'TIGER',
                'LEVERAGE_TYPE': 'Leveraged',
                'UNDERLYING': 'KOSPI2 Index',
                'UNDERLYING_CODE': 'KOSPI2 Index',
                'LEVERAGE_FACTOR': 2.0,
                'AUM': 720000000000.0,
                'CURRENCY': 'KRW'
            },
            {
                'TICKER': '272570 KS Equity',
                'NAME': 'KBSTAR 200 Leverage',
                'SECURITY_DES': 'KBSTAR 200 Leverage',
                'SECURITY_TYP2': 'ETF',
                'ISSUER': 'KBSTAR',
                'LEVERAGE_TYPE': 'Leveraged',
                'UNDERLYING': 'KOSPI2 Index',
                'UNDERLYING_CODE': 'KOSPI2 Index',
                'LEVERAGE_FACTOR': 2.0,
                'AUM': 580000000000.0,
                'CURRENCY': 'KRW'
            },
            
            # KOSPI 200 inverse
            {
                'TICKER': '114800 KS Equity',
                'NAME': 'KODEX Inverse',
                'SECURITY_DES': 'KODEX 200 Inverse',
                'SECURITY_TYP2': 'ETF',
                'ISSUER': 'KODEX',
                'LEVERAGE_TYPE': 'Inverse',
                'UNDERLYING': 'KOSPI2 Index',
                'UNDERLYING_CODE': 'KOSPI2 Index',
                'LEVERAGE_FACTOR': -1.0,
                'AUM': 880000000000.0,
                'CURRENCY': 'KRW'
            },
            {
                'TICKER': '252670 KS Equity',
                'NAME': 'KODEX 200 Futures Inverse',
                'SECURITY_DES': 'KODEX 200 Futures Inverse ETF',
                'SECURITY_TYP2': 'ETF',
                'ISSUER': 'KODEX',
                'LEVERAGE_TYPE': 'Inverse',
                'UNDERLYING': 'KOSPI2 Index',
                'UNDERLYING_CODE': 'KOSPI2 Index',
                'LEVERAGE_FACTOR': -1.0,
                'AUM': 740000000000.0,
                'CURRENCY': 'KRW'
            },
            {
                'TICKER': '123320 KS Equity',
                'NAME': 'TIGER 200 Inverse',
                'SECURITY_DES': 'TIGER 200 Inverse',
                'SECURITY_TYP2': 'ETF',
                'ISSUER': 'TIGER',
                'LEVERAGE_TYPE': 'Inverse',
                'UNDERLYING': 'KOSPI2 Index',
                'UNDERLYING_CODE': 'KOSPI2 Index',
                'LEVERAGE_FACTOR': -1.0,
                'AUM': 680000000000.0,
                'CURRENCY': 'KRW'
            },
            {
                'TICKER': '272580 KS Equity',
                'NAME': 'KBSTAR 200 Inverse',
                'SECURITY_DES': 'KBSTAR 200 Inverse',
                'SECURITY_TYP2': 'ETF',
                'ISSUER': 'KBSTAR',
                'LEVERAGE_TYPE': 'Inverse',
                'UNDERLYING': 'KOSPI2 Index',
                'UNDERLYING_CODE': 'KOSPI2 Index',
                'LEVERAGE_FACTOR': -1.0,
                'AUM': 540000000000.0,
                'CURRENCY': 'KRW'
            },
            
            # 3X ETFs
            {
                'TICKER': '293180 KS Equity',
                'NAME': 'TIGER Futures Leverage 3X',
                'SECURITY_DES': 'TIGER 200 Futures Leverage 3X ETF',
                'SECURITY_TYP2': 'ETF',
                'ISSUER': 'TIGER',
                'LEVERAGE_TYPE': 'Leveraged',
                'UNDERLYING': 'KOSPI2 Index',
                'UNDERLYING_CODE': 'KOSPI2 Index',
                'LEVERAGE_FACTOR': 3.0,
                'AUM': 425000000000.0,
                'CURRENCY': 'KRW'
            },
            {
                'TICKER': '276990 KS Equity',
                'NAME': 'KBSTAR 200 Leverage 3X',
                'SECURITY_DES': 'KBSTAR 200 Leverage 3X ETF',
                'SECURITY_TYP2': 'ETF',
                'ISSUER': 'KBSTAR',
                'LEVERAGE_TYPE': 'Leveraged',
                'UNDERLYING': 'KOSPI2 Index',
                'UNDERLYING_CODE': 'KOSPI2 Index',
                'LEVERAGE_FACTOR': 3.0,
                'AUM': 390000000000.0,
                'CURRENCY': 'KRW'
            },
            {
                'TICKER': '275980 KS Equity',
                'NAME': 'TIGER Futures Inverse 2X',
                'SECURITY_DES': 'TIGER 200 Futures Inverse 2X ETF',
                'SECURITY_TYP2': 'ETF',
                'ISSUER': 'TIGER',
                'LEVERAGE_TYPE': 'Inverse',
                'UNDERLYING': 'KOSPI2 Index',
                'UNDERLYING_CODE': 'KOSPI2 Index',
                'LEVERAGE_FACTOR': -2.0,
                'AUM': 360000000000.0,
                'CURRENCY': 'KRW'
            },
            {
                'TICKER': '276980 KS Equity',
                'NAME': 'KBSTAR 200 Inverse 3X',
                'SECURITY_DES': 'KBSTAR 200 Inverse 3X ETF',
                'SECURITY_TYP2': 'ETF',
                'ISSUER': 'KBSTAR',
                'LEVERAGE_TYPE': 'Inverse',
                'UNDERLYING': 'KOSPI2 Index',
                'UNDERLYING_CODE': 'KOSPI2 Index',
                'LEVERAGE_FACTOR': -3.0,
                'AUM': 320000000000.0,
                'CURRENCY': 'KRW'
            },
            
            # KOSDAQ ETFs
            {
                'TICKER': '251340 KS Equity',
                'NAME': 'KODEX Kosdaq150 Leverage',
                'SECURITY_DES': 'KODEX KOSDAQ150 Leverage ETF',
                'SECURITY_TYP2': 'ETF',
                'ISSUER': 'KODEX',
                'LEVERAGE_TYPE': 'Leveraged',
                'UNDERLYING': 'KOSDAQ 150 Index',
                'UNDERLYING_CODE': 'KOSDQ150 Index',
                'LEVERAGE_FACTOR': 2.0,
                'AUM': 410000000000.0,
                'CURRENCY': 'KRW'
            },
            {
                'TICKER': '251350 KS Equity',
                'NAME': 'KODEX Kosdaq150 Inverse',
                'SECURITY_DES': 'KODEX KOSDAQ150 Inverse ETF',
                'SECURITY_TYP2': 'ETF',
                'ISSUER': 'KODEX',
                'LEVERAGE_TYPE': 'Inverse',
                'UNDERLYING': 'KOSDAQ 150 Index',
                'UNDERLYING_CODE': 'KOSDQ150 Index',
                'LEVERAGE_FACTOR': -1.0,
                'AUM': 370000000000.0,
                'CURRENCY': 'KRW'
            },
            
            # Sector ETFs
            {
                'TICKER': '261060 KS Equity',
                'NAME': 'KODEX Semiconductor Leverage',
                'SECURITY_DES': 'KODEX Semiconductor Leverage ETF',
                'SECURITY_TYP2': 'ETF',
                'ISSUER': 'KODEX',
                'LEVERAGE_TYPE': 'Leveraged',
                'UNDERLYING': 'Korea Semiconductor Index',
                'UNDERLYING_CODE': 'KSEMIC Index',
                'LEVERAGE_FACTOR': 2.0,
                'AUM': 280000000000.0,
                'CURRENCY': 'KRW'
            },
            {
                'TICKER': '229200 KS Equity',
                'NAME': 'KODEX Banks Leverage',
                'SECURITY_DES': 'KODEX KOSPI Banks Leverage ETF',
                'SECURITY_TYP2': 'ETF',
                'ISSUER': 'KODEX',
                'LEVERAGE_TYPE': 'Leveraged',
                'UNDERLYING': 'KOSPI 200 Banks Index',
                'UNDERLYING_CODE': 'K200BK Index',
                'LEVERAGE_FACTOR': 2.0,
                'AUM': 195000000000.0,
                'CURRENCY': 'KRW'
            },
            
            # Unknown examples
            {
                'TICKER': '290130 KS Equity',
                'NAME': 'SOL 2X ETF',
                'SECURITY_DES': 'SOL Custom 2X ETF',
                'SECURITY_TYP2': 'ETF',
                'ISSUER': 'SOL',
                'LEVERAGE_TYPE': 'Leveraged',
                'UNDERLYING': 'Unknown',
                'UNDERLYING_CODE': 'Unknown',
                'LEVERAGE_FACTOR': 2.0,
                'AUM': 120000000000.0,
                'CURRENCY': 'KRW'
            },
            {
                'TICKER': '291890 KS Equity',
                'NAME': 'ARIRANG -2X Inverse',
                'SECURITY_DES': 'ARIRANG Custom -2X Inverse ETF',
                'SECURITY_TYP2': 'ETF',
                'ISSUER': 'ARIRANG',
                'LEVERAGE_TYPE': 'Inverse',
                'UNDERLYING': 'Unknown',
                'UNDERLYING_CODE': 'Unknown',
                'LEVERAGE_FACTOR': -2.0,
                'AUM': 95000000000.0,
                'CURRENCY': 'KRW'
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
    
    def track_li_funds(self) -> Dict[str, Any]:
        """
        Track leveraged and inverse funds in Korea.
        
        Returns:
            Dictionary with fund data and analysis
        """
        # Find leveraged and inverse ETFs
        li_etfs = self.get_kr_leveraged_inverse_funds()
        
        # Set latest update time
        self.latest_update_time = datetime.datetime.now()
        
        # Aggregate by underlying
        agg_by_underlying = self.aggregate_by_underlying(li_etfs)
        
        # Aggregate by issuer
        agg_by_issuer = self.aggregate_by_issuer(li_etfs)
        
        # Aggregate by leverage factor
        agg_by_leverage = self.aggregate_by_leverage_factor(li_etfs)
        
        # Store the results
        self.li_fund_data = {
            'Update_Time': self.latest_update_time,
            'ETF_List': li_etfs.to_dict('records') if not li_etfs.empty else [],
            'Agg_By_Underlying': agg_by_underlying.to_dict('records') if not agg_by_underlying.empty else [],
            'Agg_By_Issuer': agg_by_issuer.to_dict('records') if not agg_by_issuer.empty else [],
            'Agg_By_Leverage': agg_by_leverage.to_dict('records') if not agg_by_leverage.empty else []
        }
        
        return self.li_fund_data
    
    def format_aum(self, aum_value: float) -> str:
        """
        Format AUM value for display in KRW.
        
        Args:
            aum_value: AUM value
            
        Returns:
            Formatted AUM string
        """
        if aum_value >= 1e12:
            return f"₩{aum_value/1e12:.2f}T"
        elif aum_value >= 1e9:
            return f"₩{aum_value/1e9:.2f}B"
        elif aum_value >= 1e6:
            return f"₩{aum_value/1e6:.2f}M"
        else:
            return f"₩{aum_value:.2f}"
    
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
        print(f"LEVERAGED AND INVERSE ETFs IN KOREA - {self.latest_update_time.strftime('%Y-%m-%d %H:%M:%S')}")
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
                ticker_match = re.search(r'^(\d+)', ticker)
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
            summary_data.sort(key=lambda x: float(str(x[3]).replace('₩', '').replace('T', 'e12').replace('B', 'e9').replace('M', 'e6')), reverse=True)
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
            etf_file = os.path.join(output_dir, f"kr_li_etfs_{timestamp}.csv")
            
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
            agg_file = os.path.join(output_dir, f"kr_li_agg_by_underlying_{timestamp}.csv")
            
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
            agg_file = os.path.join(output_dir, f"kr_li_agg_by_issuer_{timestamp}.csv")
            
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
            agg_file = os.path.join(output_dir, f"kr_li_agg_by_leverage_{timestamp}.csv")
            
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


def main():
    parser = argparse.ArgumentParser(description='Track and analyze Leveraged and Inverse ETFs in Korea')
    parser.add_argument('--host', default='127.0.0.1', help='Bloomberg server host')
    parser.add_argument('--port', type=int, default=8194, help='Bloomberg server port')
    parser.add_argument('--output-dir', default='./kr_li_funds', help='Directory to save output files')
    parser.add_argument('--sample', action='store_true', 
                        help='Use sample data instead of Bloomberg data')
    
    args = parser.parse_args()
    
    # Create output directory if needed
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize tracker
    tracker = KRLeverageFundsTracker(host=args.host, port=args.port, use_sample_data=args.sample)
    
    try:
        # Start session
        if tracker.start_session():
            # Track leveraged and inverse funds
            logger.info("Tracking leveraged and inverse funds in Korea...")
            tracker.track_li_funds()
            
            # Print ETF list
            tracker.print_etf_list()
            
            # Print AUM by underlying
            tracker.print_aum_by_underlying()
            
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