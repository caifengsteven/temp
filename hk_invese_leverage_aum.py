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
        logging.FileHandler("hk_leverage_funds_tracker.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('HKLeverageFundsTracker')

class HKLeverageFundsTracker:
    """
    A class to track and analyze Leveraged and Inverse ETFs/Funds in Hong Kong markets.
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
        
        # Major Hong Kong indices for leveraged/inverse products
        self.hk_indices = [
            'HSI Index',   # Hang Seng Index
            'HSCEI Index', # Hang Seng China Enterprises Index
            'HSTECH Index' # Hang Seng TECH Index
        ]
        
        # Major issuers of leveraged/inverse products in HK
        self.hk_li_issuers = [
            'CSOP',
            'Hang Seng Investment',
            'ChinaAMC',
            'Samsung',
            'Mirae Asset',
            'Premia',
            'E Fund',
            'CICC'
        ]
        
        # Mapping of underlying names to Bloomberg codes
        self.underlying_code_map = {
            'HSI Index': 'HSI Index',
            'HSCEI Index': 'HSCEI Index',
            'HSTECH Index': 'HSTECH Index',
            'NASDAQ 100 Index': 'NDX Index',
            'S&P 500 Index': 'SPX Index',
            'CSI 300 Index': 'SHSZ300 Index',
            'FTSE China 50 Index': 'XIN9I Index',
            'CSI 500 Index': 'SHSZ500 Index',
            'MSCI China Index': 'MXCN Index',
            'Nikkei 225 Index': 'NKY Index',
            'Dow Jones Industrial Average Index': 'INDU Index',
            'FTSE China A50 Index': 'XIN9I Index',
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
    
    def get_hk_leveraged_inverse_funds(self) -> pd.DataFrame:
        """
        Get leveraged and inverse ETFs listed in Hong Kong using direct ticker lookup.
        
        Returns:
            DataFrame with leveraged/inverse ETF details
        """
        if self.use_sample_data:
            # Return sample data in sample mode
            return self.get_sample_li_etfs()
        
        # Method: Use direct tickers for known Hong Kong leveraged/inverse ETFs
        # These are from the 3-digit and 4-digit fund series in HK which typically include L&I ETFs
        known_ranges = []
        
        # 7xxx series (most leveraged/inverse ETFs in HK are in this range)
        for i in range(7200, 7600, 1):
            known_ranges.append(f"{i} HK Equity")
        
        # Add some specific tickers outside the main range
        additional_tickers = [
            '3033 HK Equity',  # CSOP HSI 2X Leveraged ETF
            '3034 HK Equity',  # CSOP HSI -1X Inverse ETF
            '2800 HK Equity',  # Tracker Fund of Hong Kong
            '2828 HK Equity',  # Hang Seng H-Share ETF
            '3088 HK Equity',  # Hang Seng TECH ETF
            '3067 HK Equity',  # Premia China ETF
        ]
        
        known_ranges.extend(additional_tickers)
        
        logger.info(f"Checking {len(known_ranges)} potential ETF tickers")
        
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
            'CIE_DES'
        ]
        
        all_etfs = self.get_security_data(known_ranges, fields)
        
        if all_etfs.empty:
            logger.warning("No ETF data retrieved. Using sample data.")
            return self.get_sample_li_etfs()
        
        # Filter for likely leveraged/inverse funds
        def is_leveraged_inverse(row):
            # Check security description and name
            for field in ['SECURITY_DES', 'NAME', 'SHORT_NAME', 'LONG_COMP_NAME', 'FUND_OBJECTIVE']:
                if field in row and isinstance(row[field], str):
                    text = row[field].upper()
                    
                    # Look for leveraged/inverse keywords
                    if any(kw in text for kw in ['LEVERAGED', 'INVERSE', 'DAILY 2X', 'DAILY -1X', 'DAILY(2X)', 'DAILY(-1X)']):
                        return True
                    if any(kw in text for kw in ['BULL', 'BEAR']):
                        return True
                    
                    # Common abbreviations in HK market
                    if ' L&I ' in text or 'L&I ETF' in text:
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
                    if isinstance(factor, str) and any(x in factor.upper() for x in ['2X', '-1X', '2.0', '-1.0']):
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
                    
                    if any(kw in text for kw in ['INVERSE', '-1X', 'DAILY -1', 'DAILY(-1X)', 'BEAR']):
                        return 'Inverse'
                    if any(kw in text for kw in ['LEVERAGED', '2X', 'DAILY 2', 'DAILY(2X)', 'BULL']):
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
                
                # Map common benchmark names to standardized names
                if isinstance(benchmark, str):
                    if 'HANG SENG INDEX' in benchmark.upper() or 'HSI' in benchmark.upper():
                        return 'HSI Index'
                    elif 'HANG SENG CHINA ENTERPRISES' in benchmark.upper() or 'HSCEI' in benchmark.upper():
                        return 'HSCEI Index'
                    elif 'HANG SENG TECH' in benchmark.upper() or 'HSTECH' in benchmark.upper():
                        return 'HSTECH Index'
                    elif 'NASDAQ 100' in benchmark.upper() or 'NDX' in benchmark.upper():
                        return 'NASDAQ 100 Index'
                    elif 'S&P 500' in benchmark.upper() or 'SPX' in benchmark.upper():
                        return 'S&P 500 Index'
                    elif 'CSI 300' in benchmark.upper() or 'SHSZ300' in benchmark.upper():
                        return 'CSI 300 Index'
                    elif 'CSI 500' in benchmark.upper() or 'SHSZ500' in benchmark.upper():
                        return 'CSI 500 Index'
                
                return benchmark
            
            # Extract from name or description
            for field in ['SECURITY_DES', 'NAME', 'SHORT_NAME', 'LONG_COMP_NAME']:
                if field in row and isinstance(row[field], str):
                    text = row[field].upper()
                    
                    if 'HSI' in text and not ('HSCEI' in text or 'HSTECH' in text):
                        return 'HSI Index'
                    elif 'HSCEI' in text or 'H-SHARE' in text or 'CHINA ENT' in text:
                        return 'HSCEI Index'
                    elif 'HSTECH' in text or 'TECH INDEX' in text:
                        return 'HSTECH Index'
                    elif 'NASDAQ 100' in text or 'NASDAQ100' in text or 'NDX' in text:
                        return 'NASDAQ 100 Index'
                    elif 'S&P 500' in text or 'SPX' in text:
                        return 'S&P 500 Index'
                    elif 'CSI 300' in text:
                        return 'CSI 300 Index'
                    elif 'CSI 500' in text:
                        return 'CSI 500 Index'
                    
                    # Try to extract other index names
                    patterns = [
                        r'ON\s+([A-Z0-9\s]+)(?:\s+INDEX)?',
                        r'([A-Z0-9\s]+)\s+DAILY\s+(?:LEVERAGED|INVERSE)',
                        r'([A-Z0-9\s]+)\s+DAILY\s+\(?[2-3]X\)?',
                        r'([A-Z0-9\s]+)\s+DAILY\s+\(?-1X\)?'
                    ]
                    
                    for pattern in patterns:
                        matches = re.findall(pattern, text)
                        if matches:
                            return matches[0].strip() + ' Index'
            
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
                    if '2X' in text or 'DAILY(2X)' in text or 'DAILY 2X' in text:
                        return 2.0
                    elif '-1X' in text or 'DAILY(-1X)' in text or 'DAILY -1X' in text:
                        return -1.0
                    elif '3X' in text:
                        return 3.0
                    elif '-2X' in text:
                        return -2.0
                    
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
            
            return 'HKD'  # Default for HK
        
        # Apply functions to extract details
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
        Create a sample dataset of leveraged/inverse ETFs for testing.
        
        Returns:
            DataFrame with sample leveraged/inverse ETF data
        """
        logger.info("Creating sample leveraged/inverse ETF data for analysis")
        
        # Create sample data
        data = [
            # HSI leveraged
            {
                'TICKER': '7200 HK Equity',
                'NAME': 'HANG SENG INV HSI LEV 2X',
                'SECURITY_DES': 'HANG SENG LEVERAGED 2X ETF',
                'SECURITY_TYP2': 'ETF',
                'LEVERAGE_TYPE': 'Leveraged',
                'UNDERLYING': 'HSI Index',
                'UNDERLYING_CODE': 'HSI Index',
                'LEVERAGE_FACTOR': 2.0,
                'AUM': 1200000000.0,
                'CURRENCY': 'HKD'
            },
            {
                'TICKER': '7226 HK Equity',
                'NAME': 'CSOP HSI DAILY LEV 2X',
                'SECURITY_DES': 'CSOP HSI DAILY LEVERAGED 2X',
                'SECURITY_TYP2': 'ETF',
                'LEVERAGE_TYPE': 'Leveraged',
                'UNDERLYING': 'HSI Index',
                'UNDERLYING_CODE': 'HSI Index',
                'LEVERAGE_FACTOR': 2.0,
                'AUM': 850000000.0,
                'CURRENCY': 'HKD'
            },
            {
                'TICKER': '7261 HK Equity',
                'NAME': 'SAMSUNG HSI DAILY LEV 2X',
                'SECURITY_DES': 'SAMSUNG HSI DAILY LEV 2X',
                'SECURITY_TYP2': 'ETF',
                'LEVERAGE_TYPE': 'Leveraged',
                'UNDERLYING': 'HSI Index',
                'UNDERLYING_CODE': 'HSI Index',
                'LEVERAGE_FACTOR': 2.0,
                'AUM': 650000000.0,
                'CURRENCY': 'HKD'
            },
            
            # HSI inverse
            {
                'TICKER': '7500 HK Equity',
                'NAME': 'HANG SENG INV HSI INV -1X',
                'SECURITY_DES': 'HANG SENG INVERSE -1X ETF',
                'SECURITY_TYP2': 'ETF',
                'LEVERAGE_TYPE': 'Inverse',
                'UNDERLYING': 'HSI Index',
                'UNDERLYING_CODE': 'HSI Index',
                'LEVERAGE_FACTOR': -1.0,
                'AUM': 950000000.0,
                'CURRENCY': 'HKD'
            },
            {
                'TICKER': '7326 HK Equity',
                'NAME': 'CSOP HSI DAILY INV -1X',
                'SECURITY_DES': 'CSOP HSI DAILY INVERSE -1X',
                'SECURITY_TYP2': 'ETF',
                'LEVERAGE_TYPE': 'Inverse',
                'UNDERLYING': 'HSI Index',
                'UNDERLYING_CODE': 'HSI Index',
                'LEVERAGE_FACTOR': -1.0,
                'AUM': 750000000.0,
                'CURRENCY': 'HKD'
            },
            {
                'TICKER': '7312 HK Equity',
                'NAME': 'SAMSUNG HSI DAILY INV -1X',
                'SECURITY_DES': 'SAMSUNG HSI DAILY INV -1X',
                'SECURITY_TYP2': 'ETF',
                'LEVERAGE_TYPE': 'Inverse',
                'UNDERLYING': 'HSI Index',
                'UNDERLYING_CODE': 'HSI Index',
                'LEVERAGE_FACTOR': -1.0,
                'AUM': 550000000.0,
                'CURRENCY': 'HKD'
            },
            
            # HSCEI leveraged
            {
                'TICKER': '7267 HK Equity',
                'NAME': 'CSOP HSCEI DAILY LEV 2X',
                'SECURITY_DES': 'CSOP HSCEI DAILY LEVERAGED 2X',
                'SECURITY_TYP2': 'ETF',
                'LEVERAGE_TYPE': 'Leveraged',
                'UNDERLYING': 'HSCEI Index',
                'UNDERLYING_CODE': 'HSCEI Index',
                'LEVERAGE_FACTOR': 2.0,
                'AUM': 620000000.0,
                'CURRENCY': 'HKD'
            },
            {
                'TICKER': '7262 HK Equity',
                'NAME': 'SAMSUNG HSCEI DAILY LEV 2X',
                'SECURITY_DES': 'SAMSUNG HSCEI DAILY LEV 2X',
                'SECURITY_TYP2': 'ETF',
                'LEVERAGE_TYPE': 'Leveraged',
                'UNDERLYING': 'HSCEI Index',
                'UNDERLYING_CODE': 'HSCEI Index',
                'LEVERAGE_FACTOR': 2.0,
                'AUM': 480000000.0,
                'CURRENCY': 'HKD'
            },
            
            # HSCEI inverse
            {
                'TICKER': '7341 HK Equity',
                'NAME': 'CSOP HSCEI DAILY INV -1X',
                'SECURITY_DES': 'CSOP HSCEI DAILY INVERSE -1X',
                'SECURITY_TYP2': 'ETF',
                'LEVERAGE_TYPE': 'Inverse',
                'UNDERLYING': 'HSCEI Index',
                'UNDERLYING_CODE': 'HSCEI Index',
                'LEVERAGE_FACTOR': -1.0,
                'AUM': 520000000.0,
                'CURRENCY': 'HKD'
            },
            {
                'TICKER': '7313 HK Equity',
                'NAME': 'SAMSUNG HSCEI DAILY INV -1X',
                'SECURITY_DES': 'SAMSUNG HSCEI DAILY INV -1X',
                'SECURITY_TYP2': 'ETF',
                'LEVERAGE_TYPE': 'Inverse',
                'UNDERLYING': 'HSCEI Index',
                'UNDERLYING_CODE': 'HSCEI Index',
                'LEVERAGE_FACTOR': -1.0,
                'AUM': 430000000.0,
                'CURRENCY': 'HKD'
            },
            
            # HSTECH leveraged
            {
                'TICKER': '7225 HK Equity',
                'NAME': 'CSOP HSTECH DAILY LEV 2X',
                'SECURITY_DES': 'CSOP HSTECH DAILY LEVERAGED 2X',
                'SECURITY_TYP2': 'ETF',
                'LEVERAGE_TYPE': 'Leveraged',
                'UNDERLYING': 'HSTECH Index',
                'UNDERLYING_CODE': 'HSTECH Index',
                'LEVERAGE_FACTOR': 2.0,
                'AUM': 880000000.0,
                'CURRENCY': 'HKD'
            },
            {
                'TICKER': '7266 HK Equity',
                'NAME': 'CHINAAMC HSTECH DAILY LEV 2X',
                'SECURITY_DES': 'CHINAAMC HSTECH DAILY LEV 2X',
                'SECURITY_TYP2': 'ETF',
                'LEVERAGE_TYPE': 'Leveraged',
                'UNDERLYING': 'HSTECH Index',
                'UNDERLYING_CODE': 'HSTECH Index',
                'LEVERAGE_FACTOR': 2.0,
                'AUM': 720000000.0,
                'CURRENCY': 'HKD'
            },
            
            # HSTECH inverse
            {
                'TICKER': '7552 HK Equity',
                'NAME': 'CSOP HSTECH DAILY INV -1X',
                'SECURITY_DES': 'CSOP HSTECH DAILY INVERSE -1X',
                'SECURITY_TYP2': 'ETF',
                'LEVERAGE_TYPE': 'Inverse',
                'UNDERLYING': 'HSTECH Index',
                'UNDERLYING_CODE': 'HSTECH Index',
                'LEVERAGE_FACTOR': -1.0,
                'AUM': 580000000.0,
                'CURRENCY': 'HKD'
            },
            {
                'TICKER': '7300 HK Equity',
                'NAME': 'CHINAAMC HSTECH DAILY INV -1X',
                'SECURITY_DES': 'CHINAAMC HSTECH DAILY INV -1X',
                'SECURITY_TYP2': 'ETF',
                'LEVERAGE_TYPE': 'Inverse',
                'UNDERLYING': 'HSTECH Index',
                'UNDERLYING_CODE': 'HSTECH Index',
                'LEVERAGE_FACTOR': -1.0,
                'AUM': 490000000.0,
                'CURRENCY': 'HKD'
            },
            
            # Other indices
            {
                'TICKER': '7280 HK Equity',
                'NAME': 'CSOP NASDAQ 100 DAILY LEV 2X',
                'SECURITY_DES': 'CSOP NASDAQ 100 DAILY LEV 2X',
                'SECURITY_TYP2': 'ETF',
                'LEVERAGE_TYPE': 'Leveraged',
                'UNDERLYING': 'NASDAQ 100 Index',
                'UNDERLYING_CODE': 'NDX Index',
                'LEVERAGE_FACTOR': 2.0,
                'AUM': 350000000.0,
                'CURRENCY': 'HKD'
            },
            {
                'TICKER': '7230 HK Equity',
                'NAME': 'CSOP NASDAQ 100 DAILY INV -1X',
                'SECURITY_DES': 'CSOP NASDAQ 100 DAILY INV -1X',
                'SECURITY_TYP2': 'ETF',
                'LEVERAGE_TYPE': 'Inverse',
                'UNDERLYING': 'NASDAQ 100 Index',
                'UNDERLYING_CODE': 'NDX Index',
                'LEVERAGE_FACTOR': -1.0,
                'AUM': 280000000.0,
                'CURRENCY': 'HKD'
            },
            {
                'TICKER': '7272 HK Equity',
                'NAME': 'CSOP S&P 500 DAILY LEV 2X',
                'SECURITY_DES': 'CSOP S&P 500 DAILY LEV 2X',
                'SECURITY_TYP2': 'ETF',
                'LEVERAGE_TYPE': 'Leveraged',
                'UNDERLYING': 'S&P 500 Index',
                'UNDERLYING_CODE': 'SPX Index',
                'LEVERAGE_FACTOR': 2.0,
                'AUM': 320000000.0,
                'CURRENCY': 'HKD'
            },
            {
                'TICKER': '7374 HK Equity',
                'NAME': 'CSOP S&P 500 DAILY INV -1X',
                'SECURITY_DES': 'CSOP S&P 500 DAILY INV -1X',
                'SECURITY_TYP2': 'ETF',
                'LEVERAGE_TYPE': 'Inverse',
                'UNDERLYING': 'S&P 500 Index',
                'UNDERLYING_CODE': 'SPX Index',
                'LEVERAGE_FACTOR': -1.0,
                'AUM': 270000000.0,
                'CURRENCY': 'HKD'
            },
            
            # Add some unknown underlying examples
            {
                'TICKER': '7299 HK Equity',
                'NAME': 'XYZ UNKNOWN DAILY LEV 2X',
                'SECURITY_DES': 'XYZ UNKNOWN DAILY LEVERAGED 2X',
                'SECURITY_TYP2': 'ETF',
                'LEVERAGE_TYPE': 'Leveraged',
                'UNDERLYING': 'Unknown',
                'UNDERLYING_CODE': 'Unknown',
                'LEVERAGE_FACTOR': 2.0,
                'AUM': 150000000.0,
                'CURRENCY': 'HKD'
            },
            {
                'TICKER': '7399 HK Equity',
                'NAME': 'ABC UNKNOWN DAILY INV -1X',
                'SECURITY_DES': 'ABC UNKNOWN DAILY INVERSE -1X',
                'SECURITY_TYP2': 'ETF',
                'LEVERAGE_TYPE': 'Inverse',
                'UNDERLYING': 'Unknown',
                'UNDERLYING_CODE': 'Unknown',
                'LEVERAGE_FACTOR': -1.0,
                'AUM': 120000000.0,
                'CURRENCY': 'HKD'
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
        agg_data['AUM_PCT'] = agg_data['TOTAL_AUM'] / total_aum * 100
        
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
        
        # Extract issuer from name
        def extract_issuer(name):
            if pd.isna(name):
                return 'Unknown'
                
            name = name.upper()
            
            if 'CSOP' in name:
                return 'CSOP'
            elif 'HANG SENG INV' in name:
                return 'Hang Seng Investment'
            elif 'CHINAAMC' in name:
                return 'ChinaAMC'
            elif 'SAMSUNG' in name:
                return 'Samsung'
            elif 'MIRAE' in name:
                return 'Mirae Asset'
            elif 'PREMIA' in name:
                return 'Premia'
            elif 'E FUND' in name:
                return 'E Fund'
            elif 'CICC' in name:
                return 'CICC'
            
            # Try to extract first word as issuer
            parts = name.split()
            if parts:
                return parts[0]
                
            return 'Unknown'
        
        # Add issuer column
        etf_df['ISSUER'] = etf_df['NAME'].apply(extract_issuer)
        
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
        agg_data['AUM_PCT'] = agg_data['TOTAL_AUM'] / total_aum * 100
        
        return agg_data
    
    def track_li_funds(self) -> Dict[str, Any]:
        """
        Track leveraged and inverse funds in Hong Kong.
        
        Returns:
            Dictionary with fund data and analysis
        """
        # Find leveraged and inverse ETFs
        li_etfs = self.get_hk_leveraged_inverse_funds()
        
        # Set latest update time
        self.latest_update_time = datetime.datetime.now()
        
        # Aggregate by underlying
        agg_by_underlying = self.aggregate_by_underlying(li_etfs)
        
        # Aggregate by issuer
        agg_by_issuer = self.aggregate_by_issuer(li_etfs)
        
        # Store the results
        self.li_fund_data = {
            'Update_Time': self.latest_update_time,
            'ETF_List': li_etfs.to_dict('records') if not li_etfs.empty else [],
            'Agg_By_Underlying': agg_by_underlying.to_dict('records') if not agg_by_underlying.empty else [],
            'Agg_By_Issuer': agg_by_issuer.to_dict('records') if not agg_by_issuer.empty else []
        }
        
        return self.li_fund_data
    
    def format_aum(self, aum_value: float) -> str:
        """
        Format AUM value for display.
        
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
        print(f"LEVERAGED AND INVERSE ETFs IN HONG KONG - {self.latest_update_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*120 + "\n")
        
        # Prepare table data
        table_data = []
        
        for etf in etf_list:
            ticker = etf.get('TICKER', '')
            name = etf.get('NAME', '')
            underlying = etf.get('UNDERLYING', '')
            underlying_code = etf.get('UNDERLYING_CODE', '')
            leverage_type = etf.get('LEVERAGE_TYPE', '')
            leverage_factor = etf.get('LEVERAGE_FACTOR', '')
            aum = etf.get('AUM', 0)
            
            # Highlight unknown underlying
            highlighted_underlying = self.highlight_unknown(underlying, ticker.split()[0] if ticker else "")
            highlighted_code = self.highlight_unknown(underlying_code, ticker.split()[0] if ticker else "")
            
            table_data.append([
                ticker,
                name,
                highlighted_underlying,
                highlighted_code,
                leverage_type,
                leverage_factor,
                self.format_aum(aum)
            ])
        
        # Sort by underlying and leverage type
        try:
            table_data.sort(key=lambda x: (str(x[2]) if RED not in str(x[2]) else "ZZZZ" + str(x[0]), str(x[4])))
        except:
            # Fallback if sorting fails
            pass
        
        # Print the table
        headers = ["Ticker", "Name", "Underlying", "Bloomberg Code", "Type", "Factor", "AUM"]
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
            summary_data.sort(key=lambda x: float(x[3].replace('$', '').replace('B', 'e9').replace('M', 'e6').replace('K', 'e3')), reverse=True)
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
            etf_file = os.path.join(output_dir, f"hk_li_etfs_{timestamp}.csv")
            
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
            agg_file = os.path.join(output_dir, f"hk_li_agg_by_underlying_{timestamp}.csv")
            
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
            agg_file = os.path.join(output_dir, f"hk_li_agg_by_issuer_{timestamp}.csv")
            
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


def main():
    parser = argparse.ArgumentParser(description='Track and analyze Leveraged and Inverse ETFs in Hong Kong')
    parser.add_argument('--host', default='127.0.0.1', help='Bloomberg server host')
    parser.add_argument('--port', type=int, default=8194, help='Bloomberg server port')
    parser.add_argument('--output-dir', default='./hk_li_funds', help='Directory to save output files')
    parser.add_argument('--sample', action='store_true', 
                        help='Use sample data instead of Bloomberg data')
    
    args = parser.parse_args()
    
    # Create output directory if needed
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize tracker
    tracker = HKLeverageFundsTracker(host=args.host, port=args.port, use_sample_data=args.sample)
    
    try:
        # Start session
        if tracker.start_session():
            # Track leveraged and inverse funds
            logger.info("Tracking leveraged and inverse funds in Hong Kong...")
            tracker.track_li_funds()
            
            # Print ETF list
            tracker.print_etf_list()
            
            # Print AUM by underlying
            tracker.print_aum_by_underlying()
            
            # Print AUM by issuer
            tracker.print_aum_by_issuer()
            
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