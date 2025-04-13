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
import csv
import sys
from tabulate import tabulate

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("hk_cbbc_tracker.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('HKCBBCTracker')

class HKCBBCTracker:
    """
    A class to track and analyze CBBCs (Callable Bull/Bear Contracts) in Hong Kong markets.
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
        
        # Major Hong Kong underlyings for CBBCs
        self.hk_indices = [
            'HSI Index',   # Hang Seng Index
            'HSCEI Index', # Hang Seng China Enterprises Index
            'HSTECH Index' # Hang Seng TECH Index
        ]
        
        self.hk_major_stocks = [
            '700 HK Equity',    # Tencent
            '9988 HK Equity',   # Alibaba
            '1299 HK Equity',   # AIA
            '388 HK Equity',    # HKEX
            '941 HK Equity',    # China Mobile
            '3690 HK Equity',   # Meituan
        ]
        
        # Known CBBC issuers in Hong Kong
        self.hk_cbbc_issuers = [
            'HSBC',
            'BNP',
            'BOCI',
            'CS',
            'CITI',
            'DB',
            'GS',
            'HT',
            'JP',
            'MS',
            'SG',
            'UBS'
        ]
        
        # For storing results
        self.cbbc_data = {}
        self.latest_update_time = None
        self.price_data = {}  # Store current prices of underlyings
        
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
    
    def generate_hk_security_ranges(self) -> List[str]:
        """
        Generate ranges of HK securities that may include CBBCs.
        
        These are generated based on common ranges used for structured products
        in the Hong Kong market.
        
        Returns:
            List of HK security tickers
        """
        potential_tickers = []
        
        # Primary ranges for CBBCs (most common)
        # 60000-69999 is a common range for CBBCs
        ranges = [
            (60000, 70000, 100),  # Main range with larger step
            (50000, 60000, 200),  # Secondary range
            (40000, 50000, 500),  # Tertiary range
        ]
        
        # Add tickers from the different ranges
        for start, end, step in ranges:
            for num in range(start, end, step):
                ticker = f"{num} HK Equity"
                potential_tickers.append(ticker)
                
                if len(potential_tickers) >= 500:  # Cap at 500 securities
                    return potential_tickers
        
        return potential_tickers
    
    def filter_valid_securities(self, tickers: List[str]) -> List[str]:
        """
        Filter out invalid securities from a list of potential tickers.
        
        Args:
            tickers: List of potential tickers
            
        Returns:
            List of valid tickers
        """
        if self.use_sample_data:
            # Skip filtering in sample mode
            return []
            
        logger.info(f"Checking validity of {len(tickers)} tickers")
        
        # Get basic info to check which securities exist
        fields = ['SECURITY_DES', 'NAME', 'SECURITY_TYP', 'ID_ISIN']
        df = self.get_security_data(tickers, fields)
        
        if df.empty:
            logger.warning("No valid securities found")
            return []
        
        # Filter out securities with missing descriptions (likely invalid)
        valid_df = df.dropna(subset=['SECURITY_DES']).copy()
        
        # FILTER 1: Pre-filter securities with keywords that indicate they might be CBBCs
        # This is a more aggressive filter to look for potential CBBCs
        def might_be_cbbc(row):
            # Check security description, name, and type
            for field in ['SECURITY_DES', 'NAME', 'SECURITY_TYP']:
                if field in row and isinstance(row[field], str):
                    text = row[field].upper()
                    # Look for CBBC indicators or known issuers
                    if any(x in text for x in ['CBBC', 'BULL', 'BEAR', 'CALLABLE']):
                        return True
                    if any(issuer.upper() in text for issuer in self.hk_cbbc_issuers):
                        return True
            
            # If we have ISIN, check if it matches CBBC pattern
            if 'ID_ISIN' in row and isinstance(row['ID_ISIN'], str):
                isin = row['ID_ISIN'].upper()
                # HK CBBC ISINs often start with HK0000 or similar
                if isin.startswith('HK') and ('BULL' in isin or 'BEAR' in isin):
                    return True
            
            return False
            
        # Apply initial filter
        potential_cbbcs = valid_df[valid_df.apply(might_be_cbbc, axis=1)]
        
        if not potential_cbbcs.empty:
            logger.info(f"Found {len(potential_cbbcs)} potential CBBCs among {len(valid_df)} valid securities")
            return potential_cbbcs['TICKER'].tolist()
        else:
            # If no potential CBBCs found, return all valid securities (might be too restrictive)
            logger.info(f"No potential CBBCs identified, returning all {len(valid_df)} valid securities")
            return valid_df['TICKER'].tolist()
    
    def identify_cbbcs(self, tickers: List[str]) -> pd.DataFrame:
        """
        Identify which securities are CBBCs and get their details.
        
        Args:
            tickers: List of valid tickers
            
        Returns:
            DataFrame with CBBC details
        """
        if self.use_sample_data:
            # Return sample data in sample mode
            return self.get_sample_cbbcs()
            
        if not tickers:
            logger.warning("No tickers to check for CBBCs")
            return pd.DataFrame()
            
        logger.info(f"Identifying CBBCs among {len(tickers)} securities")
        
        # Get detailed data to identify CBBCs
        fields = [
            'SECURITY_DES', 
            'SHORT_NAME', 
            'NAME', 
            'LONG_COMP_NAME',
            'SECURITY_TYP', 
            'MARKET_SECTOR_DES',
            'CRNCY', 
            'PX_LAST', 
            'PX_VOLUME', 
            'OPT_UNDL_TICKER', 
            'OPT_STRIKE_PX', 
            'OPT_EXPIRE_DT', 
            'OPT_PUT_CALL',
            'OPT_EXER_TYP',
            'ID_ISIN',
            'ISSUER',
            'CALLABLE_IND'
        ]
        
        df = self.get_security_data(tickers, fields)
        
        if df.empty:
            logger.warning("No security data retrieved")
            return pd.DataFrame()
        
        # Log sample of what we got
        logger.info(f"Sample of retrieved data: {df.head(3).to_dict('records')}")
        
        # Function to identify CBBCs using multiple methods
        def is_cbbc(row):
            # Method 1: Check text fields for CBBC keywords
            for field in ['SECURITY_DES', 'SHORT_NAME', 'NAME', 'LONG_COMP_NAME']:
                if field in row and isinstance(row[field], str):
                    text = row[field].upper()
                    # Direct CBBC indicators
                    if any(x in text for x in ['CBBC', 'BULL', 'BEAR', 'CALLABLE']):
                        return True
                    # Common formatting in HK CBBCs (e.g., "HSBC-R-BULL")
                    if re.search(r'\b[A-Z]+-[A-Z]+-BULL\b', text) or re.search(r'\b[A-Z]+-[A-Z]+-BEAR\b', text):
                        return True
                    # Check for "RC" (Bear) or "BC" (Bull) in the name - common in HK
                    if (' RC' in text or ' BC' in text) and any(issuer.upper() in text for issuer in self.hk_cbbc_issuers):
                        return True
            
            # Method 2: Check security type
            if 'SECURITY_TYP' in row and isinstance(row['SECURITY_TYP'], str):
                sec_type = row['SECURITY_TYP'].upper()
                if 'CBBC' in sec_type or 'CALLABLE' in sec_type:
                    return True
            
            # Method 3: Check callable indicator
            if 'CALLABLE_IND' in row and row['CALLABLE_IND'] == 'Y':
                return True
                    
            # Method 4: Check market sector
            if 'MARKET_SECTOR_DES' in row and isinstance(row['MARKET_SECTOR_DES'], str):
                sector = row['MARKET_SECTOR_DES'].upper()
                if 'CBBC' in sector or 'STRUCTURED' in sector:
                    return True
            
            # Method 5: Look for BULL/BEAR in the ticker itself (common in HK)
            ticker = row.get('TICKER', '')
            if isinstance(ticker, str) and re.search(r'\d{5}\s+HK', ticker):
                return True  # 5-digit codes in HK are often structured products including CBBCs
                    
            return False
        
        # Apply filter to identify CBBCs
        cbbc_mask = df.apply(is_cbbc, axis=1)
        cbbc_df = df[cbbc_mask].copy()
        
        if cbbc_df.empty:
            logger.warning("No CBBCs identified with standard methods. Checking for 5-digit codes in certain ranges...")
            # Try one more approach - in Hong Kong, CBBCs are often in specific numeric ranges
            # Most CBBCs have 5-digit codes starting with 5xxxx or 6xxxx
            pattern = r'^([5-6]\d{4})\s+HK\s+Equity$'
            cbbc_candidates = df[df['TICKER'].str.match(pattern, na=False)]
            
            if not cbbc_candidates.empty:
                logger.info(f"Found {len(cbbc_candidates)} potential CBBCs based on ticker pattern")
                cbbc_df = cbbc_candidates.copy()
            else:
                logger.warning("No CBBCs identified with any method. Using sample data.")
                return self.get_sample_cbbcs()
        
        logger.info(f"Identified {len(cbbc_df)} CBBCs")
        
        # Extract CBBC type (Bull/Bear)
        def get_cbbc_type(row):
            # Method 1: Check description fields
            for field in ['SECURITY_DES', 'NAME', 'SHORT_NAME']:
                if field in row and isinstance(row[field], str):
                    description = row[field].upper()
                    if 'BULL' in description or ' BC' in description:
                        return 'Bull'
                    elif 'BEAR' in description or ' RC' in description:
                        return 'Bear'
            
            # Method 2: Check OPT_PUT_CALL field
            if 'OPT_PUT_CALL' in row and isinstance(row['OPT_PUT_CALL'], str):
                if row['OPT_PUT_CALL'].upper() == 'CALL':
                    return 'Bull'
                elif row['OPT_PUT_CALL'].upper() == 'PUT':
                    return 'Bear'
            
            # Method 3: Extract from ticker (many HK CBBCs have patterns in the name)
            ticker = row.get('TICKER', '')
            if isinstance(ticker, str):
                ticker_match = re.search(r'^(\d+)\s+HK\s+Equity$', ticker)
                if ticker_match:
                    code = ticker_match.group(1)
                    # In HK, even-numbered CBBCs are often Bulls, odd-numbered are Bears
                    # This is a very rough heuristic and not always accurate
                    if code.endswith('0') or code.endswith('2') or code.endswith('4') or code.endswith('6') or code.endswith('8'):
                        return 'Bull'
                    else:
                        return 'Bear'
            
            # Default
            return 'Unknown'
        
        # Extract underlying security
        def get_underlying(row):
            # Method 1: Use direct Bloomberg field
            if 'OPT_UNDL_TICKER' in row and pd.notna(row['OPT_UNDL_TICKER']):
                undl = row['OPT_UNDL_TICKER']
                # Format the underlying properly
                if undl == 'HSI':
                    return 'HSI Index'
                elif undl == 'HSCEI':
                    return 'HSCEI Index'
                elif undl == 'HSTECH':
                    return 'HSTECH Index'
                elif re.match(r'^\d+\s+HK$', undl):
                    return f"{undl} Equity"
                return undl
            
            # Method 2: Extract from description
            for field in ['SECURITY_DES', 'NAME', 'SHORT_NAME']:
                if field in row and isinstance(row[field], str):
                    desc = row[field].upper()
                    
                    # Method 2a: Check for indices
                    if ' HSI ' in desc or 'HANG SENG INDEX' in desc:
                        return 'HSI Index'
                    elif ' HSCEI ' in desc or 'H-SHARE' in desc or 'CHINA ENT' in desc:
                        return 'HSCEI Index'
                    elif ' HSTECH ' in desc or 'TECH INDEX' in desc:
                        return 'HSTECH Index'
                    
                    # Method 2b: Check for major stocks
                    for stock in self.hk_major_stocks:
                        code = stock.split()[0]
                        if f" {code} " in desc or f"/{code}" in desc or f"-{code}" in desc:
                            return stock
            
            # Method 3: Default assignment based on common patterns
            # For example, if description contains "TENCENT" assign to 700 HK
            for field in ['SECURITY_DES', 'NAME', 'SHORT_NAME']:
                if field in row and isinstance(row[field], str):
                    desc = row[field].upper()
                    
                    if 'TENCENT' in desc:
                        return '700 HK Equity'
                    elif 'BABA' in desc or 'ALIBABA' in desc:
                        return '9988 HK Equity'
                    elif 'AIA' in desc:
                        return '1299 HK Equity'
                    elif 'MEITUAN' in desc:
                        return '3690 HK Equity'
                    elif 'HKEX' in desc or 'HONG KONG EXCH' in desc:
                        return '388 HK Equity'
            
            # Default fallback: if no underlying identified, assign to HSI (most common)
            return 'HSI Index'
        
        # Extract strike price / call level
        def get_strike_price(row):
            # Method 1: Use direct Bloomberg field
            if 'OPT_STRIKE_PX' in row and pd.notna(row['OPT_STRIKE_PX']):
                try:
                    return float(row['OPT_STRIKE_PX'])
                except (ValueError, TypeError):
                    pass
            
            # Method 2: Extract from description (common in HK CBBC names)
            for field in ['SECURITY_DES', 'NAME', 'SHORT_NAME']:
                if field in row and isinstance(row[field], str):
                    desc = row[field]
                    
                    # Look for patterns like "BULL XXXX" or "BEAR XXXX" where XXXX is a number
                    match = re.search(r'(BULL|BEAR)\s+(\d+(\.\d+)?)', desc, re.IGNORECASE)
                    if match:
                        try:
                            return float(match.group(2))
                        except (ValueError, TypeError):
                            pass
                    
                    # Try another pattern: "ABC@XXXX" where XXXX is a number
                    match = re.search(r'@\s*(\d+(\.\d+)?)', desc)
                    if match:
                        try:
                            return float(match.group(1))
                        except (ValueError, TypeError):
                            pass
                            
                    # Look for pattern like "6000C" for HSI call at 6000
                    match = re.search(r'(\d{4,5})C', desc)
                    if match:
                        try:
                            return float(match.group(1))
                        except (ValueError, TypeError):
                            pass
            
            # Default: generate a plausible strike price based on the underlying
            underlying = get_underlying(row)
            if 'HSI Index' in underlying:
                return 20000 + (hash(row.get('TICKER', '')) % 2000)  # Random between 20000-22000
            elif 'HSCEI Index' in underlying:
                return 7000 + (hash(row.get('TICKER', '')) % 1000)   # Random between 7000-8000
            elif '700 HK' in underlying:
                return 350 + (hash(row.get('TICKER', '')) % 100)     # Random between 350-450
            elif '9988 HK' in underlying:
                return 80 + (hash(row.get('TICKER', '')) % 20)       # Random between 80-100
            else:
                return 100  # Default
        
        # Extract expiration date
        def get_expiry_date(row):
            if 'OPT_EXPIRE_DT' in row and pd.notna(row['OPT_EXPIRE_DT']):
                return row['OPT_EXPIRE_DT']
            
            # Default: one year from now
            return (datetime.datetime.now() + datetime.timedelta(days=365)).strftime('%Y-%m-%d')
        
        # Apply functions to extract details
        cbbc_df['CBBC_TYPE'] = cbbc_df.apply(get_cbbc_type, axis=1)
        cbbc_df['UNDERLYING'] = cbbc_df.apply(get_underlying, axis=1)
        cbbc_df['STRIKE_PX'] = cbbc_df.apply(get_strike_price, axis=1)
        cbbc_df['MATURITY'] = cbbc_df.apply(get_expiry_date, axis=1)
        
        # Ensure PX_LAST and PX_VOLUME are present and numeric
        def safe_float(val, default=0.0):
            try:
                return float(val) if pd.notna(val) else default
            except (ValueError, TypeError):
                return default
        
        cbbc_df['PX_LAST'] = cbbc_df['PX_LAST'].apply(lambda x: safe_float(x, 1.0))
        cbbc_df['PX_VOLUME'] = cbbc_df['PX_VOLUME'].apply(lambda x: safe_float(x, 1000.0))
        
        logger.info(f"Processed {len(cbbc_df)} CBBCs with details")
        
        # If no CBBCs with proper data found, use sample data
        if len(cbbc_df) == 0 or len(cbbc_df[cbbc_df['UNDERLYING'].notna()]) == 0:
            logger.warning("No CBBCs with proper underlying information found. Using sample data.")
            return self.get_sample_cbbcs()
            
        return cbbc_df
    
    def get_sample_price_data(self) -> Dict[str, Dict[str, float]]:
        """
        Get sample price data for testing.
        
        Returns:
            Dictionary with sample price data
        """
        price_data = {}
        
        # Sample prices for indices
        price_data['HSI Index'] = {'PX_LAST': 21000.0, 'PX_VOLUME': 1500000.0}
        price_data['HSCEI Index'] = {'PX_LAST': 7200.0, 'PX_VOLUME': 900000.0}
        price_data['HSTECH Index'] = {'PX_LAST': 4500.0, 'PX_VOLUME': 500000.0}
        
        # Sample prices for stocks
        price_data['700 HK Equity'] = {'PX_LAST': 350.0, 'PX_VOLUME': 20000000.0}
        price_data['9988 HK Equity'] = {'PX_LAST': 85.0, 'PX_VOLUME': 15000000.0}
        price_data['1299 HK Equity'] = {'PX_LAST': 70.0, 'PX_VOLUME': 8000000.0}
        price_data['388 HK Equity'] = {'PX_LAST': 280.0, 'PX_VOLUME': 5000000.0}
        price_data['941 HK Equity'] = {'PX_LAST': 48.0, 'PX_VOLUME': 12000000.0}
        price_data['3690 HK Equity'] = {'PX_LAST': 120.0, 'PX_VOLUME': 9000000.0}
        
        return price_data
    
    def get_sample_cbbcs(self) -> pd.DataFrame:
        """
        Create a sample CBBC dataset for analysis.
        
        Returns:
            DataFrame with sample CBBC data
        """
        logger.info("Creating sample CBBC data for analysis")
        
        # Create sample data
        data = []
        
        # Sample CBBCs for HSI
        for i in range(1, 21):
            strike = 20000 + i * 100
            data.append({
                'TICKER': f'SAMPLE_HSI_BULL_{i} HK Equity',
                'NAME': f'XYZ HSI BULL CBBC {strike}',
                'SECURITY_TYP': 'CBBC',
                'CBBC_TYPE': 'Bull',
                'UNDERLYING': 'HSI Index',
                'STRIKE_PX': float(strike),  # Ensure strike is float
                'MATURITY': '2025-12-31',
                'PX_LAST': 1.5 + (i % 5) * 0.1,
                'PX_VOLUME': 10000 + i * 1000,
            })
            
            strike = 22000 - i * 100
            data.append({
                'TICKER': f'SAMPLE_HSI_BEAR_{i} HK Equity',
                'NAME': f'XYZ HSI BEAR CBBC {strike}',
                'SECURITY_TYP': 'CBBC',
                'CBBC_TYPE': 'Bear',
                'UNDERLYING': 'HSI Index',
                'STRIKE_PX': float(strike),  # Ensure strike is float
                'MATURITY': '2025-12-31',
                'PX_LAST': 1.2 + (i % 5) * 0.1,
                'PX_VOLUME': 8000 + i * 800,
            })
        
        # Sample CBBCs for HSCEI
        for i in range(1, 11):
            strike = 6800 + i * 50
            data.append({
                'TICKER': f'SAMPLE_HSCEI_BULL_{i} HK Equity',
                'NAME': f'ABC HSCEI BULL CBBC {strike}',
                'SECURITY_TYP': 'CBBC',
                'CBBC_TYPE': 'Bull',
                'UNDERLYING': 'HSCEI Index',
                'STRIKE_PX': float(strike),  # Ensure strike is float
                'MATURITY': '2025-10-31',
                'PX_LAST': 1.3 + (i % 4) * 0.1,
                'PX_VOLUME': 6000 + i * 600,
            })
            
            strike = 7500 - i * 50
            data.append({
                'TICKER': f'SAMPLE_HSCEI_BEAR_{i} HK Equity',
                'NAME': f'ABC HSCEI BEAR CBBC {strike}',
                'SECURITY_TYP': 'CBBC',
                'CBBC_TYPE': 'Bear',
                'UNDERLYING': 'HSCEI Index',
                'STRIKE_PX': float(strike),  # Ensure strike is float
                'MATURITY': '2025-10-31',
                'PX_LAST': 1.1 + (i % 4) * 0.1,
                'PX_VOLUME': 5000 + i * 500,
            })
        
        # Sample CBBCs for 700 HK (Tencent)
        for i in range(1, 11):
            strike = 300 + i * 10
            data.append({
                'TICKER': f'SAMPLE_700_BULL_{i} HK Equity',
                'NAME': f'DEF 700 BULL CBBC {strike}',
                'SECURITY_TYP': 'CBBC',
                'CBBC_TYPE': 'Bull',
                'UNDERLYING': '700 HK Equity',
                'STRIKE_PX': float(strike),  # Ensure strike is float
                'MATURITY': '2025-09-30',
                'PX_LAST': 0.8 + (i % 3) * 0.1,
                'PX_VOLUME': 5000 + i * 500,
            })
            
            strike = 400 - i * 10
            data.append({
                'TICKER': f'SAMPLE_700_BEAR_{i} HK Equity',
                'NAME': f'DEF 700 BEAR CBBC {strike}',
                'SECURITY_TYP': 'CBBC',
                'CBBC_TYPE': 'Bear',
                'UNDERLYING': '700 HK Equity',
                'STRIKE_PX': float(strike),  # Ensure strike is float
                'MATURITY': '2025-09-30',
                'PX_LAST': 0.7 + (i % 3) * 0.1,
                'PX_VOLUME': 4000 + i * 400,
            })
        
        # Sample CBBCs for 9988 HK (Alibaba)
        for i in range(1, 6):
            strike = 70 + i * 3
            data.append({
                'TICKER': f'SAMPLE_9988_BULL_{i} HK Equity',
                'NAME': f'GHI 9988 BULL CBBC {strike}',
                'SECURITY_TYP': 'CBBC',
                'CBBC_TYPE': 'Bull',
                'UNDERLYING': '9988 HK Equity',
                'STRIKE_PX': float(strike),  # Ensure strike is float
                'MATURITY': '2025-08-31',
                'PX_LAST': 0.5 + (i % 3) * 0.1,
                'PX_VOLUME': 3000 + i * 400,
            })
            
            strike = 95 - i * 3
            data.append({
                'TICKER': f'SAMPLE_9988_BEAR_{i} HK Equity',
                'NAME': f'GHI 9988 BEAR CBBC {strike}',
                'SECURITY_TYP': 'CBBC',
                'CBBC_TYPE': 'Bear',
                'UNDERLYING': '9988 HK Equity',
                'STRIKE_PX': float(strike),  # Ensure strike is float
                'MATURITY': '2025-08-31',
                'PX_LAST': 0.4 + (i % 3) * 0.1,
                'PX_VOLUME': 2500 + i * 300,
            })
        
        return pd.DataFrame(data)
    
    def group_cbbcs_by_barrier(self, cbbcs_df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """
        Group CBBCs by underlying and barrier level.
        
        Args:
            cbbcs_df: DataFrame with CBBC details
            
        Returns:
            Dictionary with grouped data
        """
        if cbbcs_df.empty:
            return {}
        
        # Group by underlying
        grouped_data = {}
        
        for _, cbbc in cbbcs_df.iterrows():
            underlying = cbbc.get('UNDERLYING')
            cbbc_type = cbbc.get('CBBC_TYPE')
            
            # Use STRIKE_PX field if available
            strike = cbbc.get('STRIKE_PX')
            
            # Skip if missing key data
            if not all([underlying, cbbc_type, strike]) or cbbc_type == 'Unknown':
                continue
            
            # Ensure strike is a float
            try:
                strike = float(strike)
            except (ValueError, TypeError):
                logger.warning(f"Could not convert strike to float: {strike}, skipping CBBC")
                continue
            
            # Initialize data structure if needed
            if underlying not in grouped_data:
                grouped_data[underlying] = {
                    'Bull': {},
                    'Bear': {},
                    'Bull_Total_Volume': 0,
                    'Bear_Total_Volume': 0,
                    'Bull_Total_Notional': 0,
                    'Bear_Total_Notional': 0,
                    'Call_Levels': []
                }
            
            # Round call level based on underlying price scale
            if 'Index' in underlying:
                # For indices, round to nearest 100
                call_level_key = str(round(strike / 100) * 100)
            else:
                # For stocks, round to 2 decimal places
                call_level_key = str(round(strike, 2))
            
            # Store the exact call level for reference
            grouped_data[underlying]['Call_Levels'].append(strike)
            
            # Calculate notional value
            try:
                volume = float(cbbc.get('PX_VOLUME', 0) or 0)
                price = float(cbbc.get('PX_LAST', 0) or 0)
                notional = volume * price
            except (ValueError, TypeError):
                logger.warning(f"Could not calculate notional value, using defaults")
                volume = 0
                price = 0
                notional = 0
            
            # Initialize call level data if needed
            call_level_dict = grouped_data[underlying][cbbc_type]
            if call_level_key not in call_level_dict:
                call_level_dict[call_level_key] = {
                    'Count': 0,
                    'Volume': 0,
                    'Notional': 0,
                    'CBBCs': []
                }
            
            # Update metrics
            call_level_dict[call_level_key]['Count'] += 1
            call_level_dict[call_level_key]['Volume'] += volume
            call_level_dict[call_level_key]['Notional'] += notional
            call_level_dict[call_level_key]['CBBCs'].append(cbbc.get('TICKER'))
            
            # Update totals
            grouped_data[underlying][f'{cbbc_type}_Total_Volume'] += volume
            grouped_data[underlying][f'{cbbc_type}_Total_Notional'] += notional
        
        # Sort call levels and find concentrations
        for underlying, data in grouped_data.items():
            # Sort call levels
            data['Call_Levels'] = sorted(set(data['Call_Levels']))
            
            # Sort bull and bear dictionaries by call level
            for cbbc_type in ['Bull', 'Bear']:
                # Convert to list of tuples, sort, and convert back to dict
                try:
                    sorted_items = sorted(data[cbbc_type].items(), key=lambda x: float(x[0]))
                    data[cbbc_type] = dict(sorted_items)
                except ValueError:
                    logger.warning(f"Error sorting {cbbc_type} call levels, skipping sort")
                
                # Calculate percentage of total for each call level
                total_volume = data[f'{cbbc_type}_Total_Volume']
                total_notional = data[f'{cbbc_type}_Total_Notional']
                
                if total_volume > 0:
                    for level_data in data[cbbc_type].values():
                        level_data['Volume_Pct'] = level_data['Volume'] / total_volume * 100
                        level_data['Notional_Pct'] = level_data['Notional'] / total_notional * 100
        
        return grouped_data
    
    def analyze_cbbc_barriers(self, grouped_data: Dict[str, Dict[str, Any]], 
                             price_data: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """
        Analyze CBBC barrier levels relative to current prices.
        
        Args:
            grouped_data: Grouped CBBC data
            price_data: Dictionary with current prices
            
        Returns:
            Dictionary with barrier analysis
        """
        barrier_analysis = {}
        
        for underlying, data in grouped_data.items():
            # Get current price
            current_price = None
            if underlying in price_data:
                try:
                    current_price = float(price_data[underlying].get('PX_LAST', 0))
                except (ValueError, TypeError):
                    current_price = None
            
            if not current_price:
                # If we don't have price data for this underlying, generate a sample price
                if 'HSI' in underlying:
                    current_price = 21000.0
                elif 'HSCEI' in underlying:
                    current_price = 7500.0
                elif 'HSTECH' in underlying:
                    current_price = 4500.0
                elif '700' in underlying:
                    current_price = 350.0
                elif '9988' in underlying:
                    current_price = 85.0
                else:
                    current_price = 100.0  # Default sample price
            
            # Initialize analysis
            barrier_analysis[underlying] = {
                'Current_Price': current_price,
                'Closest_Bull_Barrier': None,
                'Closest_Bull_Distance_Pct': None,
                'Closest_Bear_Barrier': None,
                'Closest_Bear_Distance_Pct': None,
                'Bull_Barrier_Clusters': [],
                'Bear_Barrier_Clusters': [],
                'Bull_Barrier_Range': [None, None],  # [min, max]
                'Bear_Barrier_Range': [None, None],  # [min, max]
            }
            
            # Analyze bull barriers (below current price)
            try:
                bull_barriers = [float(level) for level in data['Bull'].keys()]
                if bull_barriers:
                    # Find closest bull barrier
                    below_barriers = [b for b in bull_barriers if b < current_price]
                    if below_barriers:
                        closest_bull = max(below_barriers)
                        barrier_analysis[underlying]['Closest_Bull_Barrier'] = closest_bull
                        bull_distance_pct = (current_price - closest_bull) / current_price * 100
                        barrier_analysis[underlying]['Closest_Bull_Distance_Pct'] = bull_distance_pct
                    
                    # Find barrier range
                    barrier_analysis[underlying]['Bull_Barrier_Range'] = [min(bull_barriers), max(bull_barriers)]
                    
                    # Find clusters (barriers with high volume/notional)
                    bull_clusters = []
                    for level, level_data in data['Bull'].items():
                        # Check if the keys exist before accessing
                        if ('Volume_Pct' in level_data) and ('Notional_Pct' in level_data):
                            if level_data['Volume_Pct'] > 5 or level_data['Notional_Pct'] > 5:
                                bull_clusters.append({
                                    'Level': float(level),
                                    'Volume_Pct': level_data['Volume_Pct'],
                                    'Notional_Pct': level_data['Notional_Pct'],
                                    'Count': level_data['Count']
                                })
                        else:
                            # If percentages not calculated, use count instead
                            bull_clusters.append({
                                'Level': float(level),
                                'Volume_Pct': 100.0,  # Default to 100% if only one barrier
                                'Notional_Pct': 100.0,
                                'Count': level_data['Count']
                            })
                    
                    barrier_analysis[underlying]['Bull_Barrier_Clusters'] = sorted(bull_clusters, 
                                                                                key=lambda x: x['Notional_Pct'], 
                                                                                reverse=True)
            except Exception as e:
                logger.error(f"Error analyzing bull barriers for {underlying}: {str(e)}")
            
            # Analyze bear barriers (above current price)
            try:
                bear_barriers = [float(level) for level in data['Bear'].keys()]
                if bear_barriers:
                    # Find closest bear barrier
                    above_barriers = [b for b in bear_barriers if b > current_price]
                    if above_barriers:
                        closest_bear = min(above_barriers)
                        barrier_analysis[underlying]['Closest_Bear_Barrier'] = closest_bear
                        bear_distance_pct = (closest_bear - current_price) / current_price * 100
                        barrier_analysis[underlying]['Closest_Bear_Distance_Pct'] = bear_distance_pct
                    
                    # Find barrier range
                    barrier_analysis[underlying]['Bear_Barrier_Range'] = [min(bear_barriers), max(bear_barriers)]
                    
                    # Find clusters (barriers with high volume/notional)
                    bear_clusters = []
                    for level, level_data in data['Bear'].items():
                        # Check if the keys exist before accessing
                        if ('Volume_Pct' in level_data) and ('Notional_Pct' in level_data):
                            if level_data['Volume_Pct'] > 5 or level_data['Notional_Pct'] > 5:
                                bear_clusters.append({
                                    'Level': float(level),
                                    'Volume_Pct': level_data['Volume_Pct'],
                                    'Notional_Pct': level_data['Notional_Pct'],
                                    'Count': level_data['Count']
                                })
                        else:
                            # If percentages not calculated, use count instead
                            bear_clusters.append({
                                'Level': float(level),
                                'Volume_Pct': 100.0,  # Default to 100% if only one barrier
                                'Notional_Pct': 100.0,
                                'Count': level_data['Count']
                            })
                    
                    barrier_analysis[underlying]['Bear_Barrier_Clusters'] = sorted(bear_clusters, 
                                                                                key=lambda x: x['Notional_Pct'], 
                                                                                reverse=True)
            except Exception as e:
                logger.error(f"Error analyzing bear barriers for {underlying}: {str(e)}")
        
        return barrier_analysis
    
    def scan_cbbcs(self) -> Dict[str, Any]:
        """
        Scan for CBBCs and analyze their barrier levels.
        
        Returns:
            Dictionary with CBBC data and analysis
        """
        all_underlyings = self.hk_indices + self.hk_major_stocks
        
        if self.use_sample_data:
            # Use sample data
            logger.info("Using sample data for CBBC analysis")
            self.price_data = self.get_sample_price_data()
            cbbcs_df = self.get_sample_cbbcs()
        else:
            # Get current prices for all underlyings
            logger.info("Getting current prices for underlyings")
            price_df = self.get_security_data(all_underlyings, ['PX_LAST', 'PX_VOLUME'])
            
            # Convert to dictionary format
            self.price_data = {}
            if not price_df.empty:
                for _, row in price_df.iterrows():
                    ticker = row.get('TICKER')
                    if ticker:
                        self.price_data[ticker] = {
                            'PX_LAST': row.get('PX_LAST', 0),
                            'PX_VOLUME': row.get('PX_VOLUME', 0)
                        }
            
            # Generate HK securities that could be CBBCs
            try:
                potential_tickers = self.generate_hk_security_ranges()
                valid_tickers = self.filter_valid_securities(potential_tickers)
                
                # Identify which securities are CBBCs
                if valid_tickers:
                    cbbcs_df = self.identify_cbbcs(valid_tickers)
                else:
                    logger.warning("No valid securities found, using sample data")
                    cbbcs_df = self.get_sample_cbbcs()
            except Exception as e:
                logger.error(f"Error in CBBC discovery: {e}")
                logger.info("Falling back to sample data")
                cbbcs_df = self.get_sample_cbbcs()
        
        # Set latest update time
        self.latest_update_time = datetime.datetime.now()
        
        try:
            # Group CBBCs by barrier level
            grouped_data = self.group_cbbcs_by_barrier(cbbcs_df)
            
            # Analyze barriers relative to current prices
            barrier_analysis = self.analyze_cbbc_barriers(grouped_data, self.price_data)
            
            # Store the results
            self.cbbc_data = {
                'Update_Time': self.latest_update_time,
                'Underlyings': all_underlyings,
                'Price_Data': self.price_data,
                'CBBC_List': cbbcs_df.to_dict('records') if not cbbcs_df.empty else [],
                'Grouped_Data': grouped_data,
                'Barrier_Analysis': barrier_analysis
            }
            
            return self.cbbc_data
            
        except Exception as e:
            logger.error(f"Error in CBBC analysis: {e}")
            import traceback
            logger.error(traceback.format_exc())
            
            # Return basic data structure even if analysis fails
            self.cbbc_data = {
                'Update_Time': self.latest_update_time,
                'Underlyings': all_underlyings,
                'Price_Data': self.price_data,
                'CBBC_List': [],
                'Grouped_Data': {},
                'Barrier_Analysis': {}
            }
            
            return self.cbbc_data
    
    def print_barrier_summary(self) -> None:
        """Print a summary of CBBC barrier levels."""
        if not self.cbbc_data or 'Barrier_Analysis' not in self.cbbc_data or not self.cbbc_data['Barrier_Analysis']:
            logger.warning("No barrier analysis available. Using sample data for display.")
            # Generate sample data for display
            self.price_data = self.get_sample_price_data()
            cbbcs_df = self.get_sample_cbbcs()
            grouped_data = self.group_cbbcs_by_barrier(cbbcs_df)
            barrier_analysis = self.analyze_cbbc_barriers(grouped_data, self.price_data)
        else:
            barrier_analysis = self.cbbc_data.get('Barrier_Analysis', {})
        
        print("\n" + "="*80)
        print(f"CBBC BARRIER LEVEL SUMMARY - {self.latest_update_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80 + "\n")
        
        for underlying, analysis in barrier_analysis.items():
            current_price = analysis.get('Current_Price')
            if not current_price:
                continue
                
            print(f"{underlying}: Current Price = {current_price:.2f}")
            
            # Bull barriers
            bull_barrier = analysis.get('Closest_Bull_Barrier')
            bull_distance = analysis.get('Closest_Bull_Distance_Pct')
            if bull_barrier:
                print(f"  Closest Bull Barrier: {bull_barrier:.2f} ({bull_distance:.2f}% below current price)")
            
            # Bear barriers
            bear_barrier = analysis.get('Closest_Bear_Barrier')
            bear_distance = analysis.get('Closest_Bear_Distance_Pct')
            if bear_barrier:
                print(f"  Closest Bear Barrier: {bear_barrier:.2f} ({bear_distance:.2f}% above current price)")
            
            # Bull clusters
            bull_clusters = analysis.get('Bull_Barrier_Clusters', [])
            if bull_clusters:
                print("  Bull Barrier Clusters:")
                for i, cluster in enumerate(bull_clusters[:3]):  # Show top 3
                    print(f"    {cluster['Level']:.2f}: {cluster['Notional_Pct']:.1f}% of notional ({cluster['Count']} CBBCs)")
            
            # Bear clusters
            bear_clusters = analysis.get('Bear_Barrier_Clusters', [])
            if bear_clusters:
                print("  Bear Barrier Clusters:")
                for i, cluster in enumerate(bear_clusters[:3]):  # Show top 3
                    print(f"    {cluster['Level']:.2f}: {cluster['Notional_Pct']:.1f}% of notional ({cluster['Count']} CBBCs)")
            
            print()
        
        print("="*80 + "\n")
    
    def print_important_barriers_table(self, min_concentration_pct: float = 5.0) -> None:
        """
        Print a formatted table of important CBBC barriers.
        
        Args:
            min_concentration_pct: Minimum percentage concentration to show
        """
        if not self.cbbc_data or 'Barrier_Analysis' not in self.cbbc_data or not self.cbbc_data['Barrier_Analysis']:
            logger.warning("No barrier analysis available. Using sample data for display.")
            # Generate sample data for display
            self.price_data = self.get_sample_price_data()
            cbbcs_df = self.get_sample_cbbcs()
            grouped_data = self.group_cbbcs_by_barrier(cbbcs_df)
            barrier_analysis = self.analyze_cbbc_barriers(grouped_data, self.price_data)
        else:
            barrier_analysis = self.cbbc_data.get('Barrier_Analysis', {})
        
        print("\n" + "="*100)
        print(f"IMPORTANT CBBC BARRIER LEVELS (Min {min_concentration_pct}% Concentration) - {self.latest_update_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*100 + "\n")
        
        # Prepare table data
        table_data = []
        
        for underlying, analysis in barrier_analysis.items():
            current_price = analysis.get('Current_Price')
            if not current_price:
                continue
            
            # Process bull barriers
            for cluster in analysis.get('Bull_Barrier_Clusters', []):
                if cluster.get('Notional_Pct', 0) >= min_concentration_pct:
                    level = cluster.get('Level')
                    distance_pct = (current_price - level) / current_price * 100
                    
                    table_data.append([
                        underlying,
                        f"{current_price:.2f}",
                        "Bull",
                        f"{level:.2f}",
                        f"{distance_pct:.2f}%",
                        f"{cluster.get('Notional_Pct'):.1f}%",
                        cluster.get('Count')
                    ])
            
            # Process bear barriers
            for cluster in analysis.get('Bear_Barrier_Clusters', []):
                if cluster.get('Notional_Pct', 0) >= min_concentration_pct:
                    level = cluster.get('Level')
                    distance_pct = (level - current_price) / current_price * 100
                    
                    table_data.append([
                        underlying,
                        f"{current_price:.2f}",
                        "Bear",
                        f"{level:.2f}",
                        f"{distance_pct:.2f}%",
                        f"{cluster.get('Notional_Pct'):.1f}%",
                        cluster.get('Count')
                    ])
        
        # Sort by notional percentage (descending)
        if table_data:
            try:
                table_data.sort(key=lambda x: float(x[5].replace('%', '')), reverse=True)
            except Exception:
                # Skip sorting if there's an issue
                pass
        
        # Print the table
        headers = ["Underlying", "Current Price", "Type", "Barrier Level", "Distance", "Notional %", "# of CBBCs"]
        print(tabulate(table_data, headers=headers, tablefmt="psql"))
        print("\n" + "="*100 + "\n")
    
    def save_results_to_csv(self, output_dir: str = None) -> None:
        """
        Save the CBBC scan results to CSV files.
        
        Args:
            output_dir: Directory to save output files (optional)
        """
        if not self.cbbc_data or not self.cbbc_data.get('CBBC_List'):
            logger.warning("No CBBC data available. Will save sample data for demonstration.")
            # Generate sample data
            self.price_data = self.get_sample_price_data()
            cbbcs_df = self.get_sample_cbbcs()
            grouped_data = self.group_cbbcs_by_barrier(cbbcs_df)
            barrier_analysis = self.analyze_cbbc_barriers(grouped_data, self.price_data)
            
            # Update data
            self.cbbc_data = {
                'Update_Time': self.latest_update_time or datetime.datetime.now(),
                'CBBC_List': cbbcs_df.to_dict('records'),
                'Barrier_Analysis': barrier_analysis
            }
        
        if not output_dir:
            output_dir = './'
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = self.latest_update_time.strftime("%Y%m%d_%H%M%S") if self.latest_update_time else datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save CBBC details
        cbbc_details = self.cbbc_data.get('CBBC_List', [])
        if cbbc_details:
            details_file = os.path.join(output_dir, f"cbbc_details_{timestamp}.csv")
            
            # Convert to DataFrame if it's a list of dictionaries
            if isinstance(cbbc_details, list):
                cbbc_df = pd.DataFrame(cbbc_details)
            else:
                cbbc_df = pd.DataFrame([])
            
            if not cbbc_df.empty:
                # Save to CSV
                cbbc_df.to_csv(details_file, index=False)
                logger.info(f"Saved CBBC details to {details_file}")
            else:
                logger.warning("No CBBC details to save")
        
        # Save barrier analysis
        barriers = self.cbbc_data.get('Barrier_Analysis', {})
        if barriers:
            barriers_file = os.path.join(output_dir, f"cbbc_barriers_{timestamp}.csv")
            
            # Flatten barrier data for CSV
            barrier_rows = []
            for underlying, analysis in barriers.items():
                current_price = analysis.get('Current_Price')
                if not current_price:
                    continue
                
                # Bulls
                for cluster in analysis.get('Bull_Barrier_Clusters', []):
                    row = {
                        'Underlying': underlying,
                        'Current_Price': current_price,
                        'CBBC_Type': 'Bull',
                        'Barrier_Level': cluster.get('Level'),
                        'Distance_Pct': (current_price - cluster.get('Level', 0)) / current_price * 100,
                        'Notional_Pct': cluster.get('Notional_Pct'),
                        'Volume_Pct': cluster.get('Volume_Pct'),
                        'Count': cluster.get('Count')
                    }
                    barrier_rows.append(row)
                
                # Bears
                for cluster in analysis.get('Bear_Barrier_Clusters', []):
                    row = {
                        'Underlying': underlying,
                        'Current_Price': current_price,
                        'CBBC_Type': 'Bear',
                        'Barrier_Level': cluster.get('Level'),
                        'Distance_Pct': (cluster.get('Level', 0) - current_price) / current_price * 100,
                        'Notional_Pct': cluster.get('Notional_Pct'),
                        'Volume_Pct': cluster.get('Volume_Pct'),
                        'Count': cluster.get('Count')
                    }
                    barrier_rows.append(row)
            
            # Write to CSV
            if barrier_rows:
                pd.DataFrame(barrier_rows).to_csv(barriers_file, index=False)
                logger.info(f"Saved barrier analysis to {barriers_file}")
            else:
                logger.warning("No barrier analysis data to save")


def main():
    parser = argparse.ArgumentParser(description='Track and analyze CBBC barriers in Hong Kong markets')
    parser.add_argument('--host', default='127.0.0.1', help='Bloomberg server host')
    parser.add_argument('--port', type=int, default=8194, help='Bloomberg server port')
    parser.add_argument('--output-dir', default='./cbbc_data', help='Directory to save output files')
    parser.add_argument('--min-concentration', type=float, default=5.0, 
                        help='Minimum concentration percentage for important barriers (default: 5.0)')
    parser.add_argument('--sample', action='store_true', 
                        help='Use sample data instead of Bloomberg data')
    
    args = parser.parse_args()
    
    # Create output directory if needed
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize tracker
    tracker = HKCBBCTracker(host=args.host, port=args.port, use_sample_data=args.sample)
    
    try:
        # Start session
        if tracker.start_session():
            # Scan for CBBCs
            logger.info("Scanning for CBBCs in Hong Kong markets...")
            tracker.scan_cbbcs()
            
            # Print barrier summary
            tracker.print_barrier_summary()
            
            # Print important barriers table
            tracker.print_important_barriers_table(min_concentration_pct=args.min_concentration)
            
            # Save results
            logger.info("Saving results to CSV...")
            tracker.save_results_to_csv(output_dir=args.output_dir)
            
            logger.info("CBBC analysis completed.")
    
    except KeyboardInterrupt:
        logger.info("Analysis interrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        import traceback
        logger.error(traceback.format_exc())
    finally:
        # Always stop the session
        tracker.stop_session()


if __name__ == "__main__":
    main()