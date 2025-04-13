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
        logging.FileHandler("jp_leverage_funds_tracker.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('JPLeverageFundsTracker')

class JPLeverageFundsTracker:
    """
    A class to track and analyze Leveraged and Inverse ETFs in Japanese markets.
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
        
        # Major Japanese indices for leveraged/inverse products
        self.jp_indices = [
            'NKY Index',       # Nikkei 225
            'TPX Index',       # TOPIX
            'NKVI Index',      # Nikkei 225 Volatility Index
            'TPXVI Index',     # TOPIX Volatility Index
            'TPXSM Index',     # TOPIX Small
            'NKYJQ Index',     # Nikkei 225 Juniors
            'TOPXXJ Index',    # TOPIX Core 30
            'MSCI Japan'       # MSCI Japan
        ]
        
        # Major Japanese issuers of leveraged/inverse products
        self.jp_li_issuers = [
            'Nomura',
            'Daiwa',
            'Nikko',
            'Mitsubishi UFJ',
            'NEXT FUNDS',
            'MAXIS',
            'iShares',
            'SMDAM',
            'Simplex',
            'ETFS'
        ]
        
        # Mapping of underlying names to Bloomberg codes
        self.underlying_code_map = {
            'Nikkei 225 Index': 'NKY Index',           # Nikkei 225
            'Nikkei 225 VI Index': 'NKVI Index',       # Nikkei 225 Volatility Index
            'TOPIX Index': 'TPX Index',                # Tokyo Stock Price Index
            'TOPIX VI Index': 'TPXVI Index',           # TOPIX Volatility Index
            'TOPIX Core 30 Index': 'TOPXXJ Index',     # TOPIX Core 30
            'Nikkei 225 Double Inverse Index': 'NKY2I Index',  # Nikkei 225 Double Inverse
            'Nikkei 225 Leveraged Index': 'NKY2 Index',        # Nikkei 225 Leveraged (x2)
            'Nikkei 225 Inverse Index': 'NKYI Index',          # Nikkei 225 Inverse
            'TOPIX Leveraged (2x) Index': 'TPX2 Index',       # TOPIX Leveraged
            'TOPIX Inverse (-1x) Index': 'TPXI Index',         # TOPIX Inverse
            'TOPIX Double Inverse (-2x) Index': 'TPX2I Index', # TOPIX Double Inverse
            'JPX Nikkei 400 Index': 'JPNK400 Index',    # JPX Nikkei 400
            'JPX Nikkei 400 Net TR Index': 'JPNKNTR Index',  # JPX Nikkei 400 Net TR
            'MSCI Japan Index': 'MXJP Index',           # MSCI Japan
            'FTSE Japan Index': 'FTJPAN Index',        # FTSE Japan
            'JPX/S&P CAPEX & Human Capital Index': 'SPJPCAHC Index',  # JPX/S&P CAPEX & Human Capital 
            'TOPIX Banks Index': 'TPXBK Index',        # TOPIX Banks
            'Nikkei 225 Banks Index': 'NKYBNK Index',   # Nikkei 225 Banks
            'TOPIX Electric Appliances Index': 'TPXEL Index',  # TOPIX Electric Appliances
            'TOPIX Transportation Equipment Index': 'TPXTE Index'  # TOPIX Transportation Equipment
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
    
    def get_jp_leveraged_inverse_funds(self) -> pd.DataFrame:
        """
        Get leveraged and inverse ETFs listed in Japan using direct ticker lookup.
        
        Returns:
            DataFrame with leveraged/inverse ETF details
        """
        if self.use_sample_data:
            # Return sample data in sample mode
            return self.get_sample_li_etfs()
        
        # Method: Use patterns for Japanese leveraged/inverse ETFs
        # Japanese ETFs use JT Equity suffix
        
        # In Japan, most leveraged/inverse ETFs have descriptive names
        # Collect common known Japanese ETF codes and tickers
        known_tickers = []
        
        # Major Nomura ETFs (NEXT FUNDS series)
        nomura_etfs = [
            '1570 JT Equity',  # NEXT FUNDS Nikkei 225 Leveraged Index ETF
            '1571 JT Equity',  # NEXT FUNDS Nikkei 225 Inverse Index ETF
            '1357 JT Equity',  # NEXT FUNDS Nikkei 225 Double Inverse Index ETF
            '1365 JT Equity',  # NEXT FUNDS TOPIX Leveraged (2x) Index ETF
            '1366 JT Equity',  # NEXT FUNDS TOPIX Inverse (-1x) Index ETF
            '1367 JT Equity',  # NEXT FUNDS TOPIX Double Inverse (-2x) Index ETF
            '1358 JT Equity',  # NEXT FUNDS Nikkei 225 VI ETF
            '1552 JT Equity',  # NEXT FUNDS TPX Banks Index ETF
            '1368 JT Equity',  # NEXT FUNDS TOPIX-17 BANKS ETF
            '1369 JT Equity',  # NEXT FUNDS TOPIX-17 FOODS ETF
            '1386 JT Equity',  # NEXT FUNDS JPX-Nikkei 400 Leveraged 2x
            '1387 JT Equity',  # NEXT FUNDS JPX-Nikkei 400 Inverse
            '1398 JT Equity',  # NEXT FUNDS Nikkei 225 Double Inverse Index (Quarterly Settlement) ETF
            '1699 JT Equity',  # NEXT FUNDS TOPIX Double Inverse Index (Quarterly Settlement) ETF
        ]
        
        # Daiwa ETFs
        daiwa_etfs = [
            '1456 JT Equity',  # Daiwa ETF Nikkei 225 Leveraged Index
            '1457 JT Equity',  # Daiwa ETF Nikkei 225 Inverse Index
            '1458 JT Equity',  # Daiwa ETF Nikkei 225 Double Inverse Index
            '1460 JT Equity',  # Daiwa ETF TOPIX Double Inverse (-2x) Index
            '1461 JT Equity',  # Daiwa ETF TOPIX Leveraged (2x) Index
            '1462 JT Equity',  # Daiwa ETF TOPIX Inverse (-1x) Index
            '1465 JT Equity',  # Daiwa ETF Nikkei 225 Volatility Index
            '1467 JT Equity',  # Daiwa ETF TOPIX Volatility Index
            '1468 JT Equity',  # Daiwa ETF Japan JPX400 Leveraged (2x) Index
            '1469 JT Equity',  # Daiwa ETF Japan JPX400 Inverse (-1x) Index
        ]
        
        # Mitsubishi UFJ ETFs (MAXIS series)
        mitsubishi_etfs = [
            '1550 JT Equity',  # MAXIS TOPIX Leveraged Index ETF
            '1551 JT Equity',  # MAXIS TOPIX Inverse Index ETF
            '1555 JT Equity',  # MAXIS TOPIX Core 30 ETF
            '1556 JT Equity',  # MAXIS JPX-Nikkei Index 400 ETF
            '2558 JT Equity',  # MAXIS JAPANESE EQUITY ETF
            '2559 JT Equity',  # MAXIS JAPANESE EQUITY RISK CTRL ETF
        ]
        
        # Simplex ETFs
        simplex_etfs = [
            '1568 JT Equity',  # Simplex TOPIX Bull 2x (Double Bull) ETF
            '1569 JT Equity',  # Simplex TOPIX Bear 2x (Double Bear) ETF
            '1457 JT Equity',  # Simplex Nikkei 225 Bull 2x (Double Bull) ETF
            '1458 JT Equity',  # Simplex Nikkei 225 Bear 2x (Double Bear) ETF
            '2042 JT Equity',  # Simplex Nikkei 225 VI ETF
        ]
        
        # Nikko ETFs
        nikko_etfs = [
            '1579 JT Equity',  # Nikko Listed Index Fund Nikkei 225 Leveraged Index
            '1580 JT Equity',  # Nikko Listed Index Fund Nikkei 225 Inverse Index
            '1585 JT Equity',  # Nikko Listed Index Fund TOPIX Leveraged (2x) Index
            '1586 JT Equity',  # Nikko Listed Index Fund TOPIX Inverse (-1x) Index
            '1595 JT Equity',  # Nikko Listed Index Fund JPX-Nikkei Index 400 2x Leveraged
            '1596 JT Equity',  # Nikko Listed Index Fund JPX-Nikkei Index 400 Inverse
        ]
        
        # Other ETFs
        other_etfs = [
            '1678 JT Equity',  # iShares Nikkei 225 ETF
            '1680 JT Equity',  # iShares JPX-Nikkei 400 ETF
            '2521 JT Equity',  # SMT J-REITs BULL 2X
            '2522 JT Equity',  # SMT J-REITs BEAR -1X
            '2526 JT Equity',  # SMT TPX BULL 2X
            '2527 JT Equity',  # SMT TPX BEAR -1X
            '2528 JT Equity',  # SMT TPX BULL 3X
            '2529 JT Equity',  # SMT TPX BEAR -2X
        ]
        
        # Combine all ETFs
        known_tickers = nomura_etfs + daiwa_etfs + mitsubishi_etfs + simplex_etfs + nikko_etfs + other_etfs
        
        # Add systematic search for leveraged/inverse patterns in Japan market
        # Scan common ETF code ranges in Japan
        for i in range(1350, 1600):
            known_tickers.append(f"{i} JT Equity")
        
        # Higher code range that sometimes includes leveraged/inverse ETFs
        for i in range(2521, 2560):
            known_tickers.append(f"{i} JT Equity")
        
        # Remove duplicates
        all_tickers = list(set(known_tickers))
        
        logger.info(f"Checking {len(all_tickers)} potential ETF tickers")
        
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
        
        all_etfs = self.get_security_data(all_tickers, fields)
        
        if all_etfs.empty:
            logger.warning("No ETF data retrieved. Using sample data.")
            return self.get_sample_li_etfs()
        
        # Filter for likely leveraged/inverse funds
        def is_leveraged_inverse(row):
            # Check security description and name
            for field in ['SECURITY_DES', 'NAME', 'SHORT_NAME', 'LONG_COMP_NAME', 'FUND_OBJECTIVE']:
                if field in row and isinstance(row[field], str):
                    text = row[field].upper()
                    
                    # Look for Japanese and English leveraged/inverse keywords
                    # ブル = Bull, ベア = Bear, レバレッジ = Leverage, インバース = Inverse in Japanese
                    if any(kw in text for kw in [
                        'LEVERAGED', 'INVERSE', 'LEVERAGE', 'ブル', 'ベア', 'レバレッジ', 'インバース',
                        '2X', '3X', '-1X', '-2X', '-3X', 'BULL', 'BEAR', 'DOUBLE BULL', 'DOUBLE BEAR',
                        'DOUBLE INVERSE', 'DOUBLE LEVERAGE', 'ダブル', 'ダブルベア', 'ダブルブル'
                    ]):
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
            # Check name and description
            for field in ['SECURITY_DES', 'NAME', 'SHORT_NAME', 'LONG_COMP_NAME', 'FUND_OBJECTIVE']:
                if field in row and isinstance(row[field], str):
                    text = row[field].upper()
                    
                    if any(kw in text for kw in ['INVERSE', '-1X', '-2X', '-3X', 'インバース', 'BEAR', 'ベア']):
                        return 'Inverse'
                    if any(kw in text for kw in ['LEVERAGED', 'LEVERAGE', '2X', '3X', 'レバレッジ', 'BULL', 'ブル']):
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
                
                # Map common Japanese benchmark names to standardized names
                if isinstance(benchmark, str):
                    if any(x in benchmark.upper() for x in ['NIKKEI', 'NIKKEI 225', '日経225']):
                        return 'Nikkei 225 Index'
                    elif any(x in benchmark.upper() for x in ['TOPIX', 'TOKYO PRICE INDEX', '東証株価指数']):
                        return 'TOPIX Index'
                    elif any(x in benchmark.upper() for x in ['JPX', 'NIKKEI 400', 'JPX-NIKKEI 400']):
                        return 'JPX Nikkei 400 Index'
                    elif any(x in benchmark.upper() for x in ['MSCI JAPAN']):
                        return 'MSCI Japan Index'
                    elif any(x in benchmark.upper() for x in ['NIKKEI VOLATILITY', 'NIKKEI VI', 'ボラティリティ']):
                        return 'Nikkei 225 VI Index'
                    elif any(x in benchmark.upper() for x in ['TOPIX VOLATILITY', 'TOPIX VI']):
                        return 'TOPIX VI Index'
                
                return benchmark
            
            # Extract from name or description
            for field in ['SECURITY_DES', 'NAME', 'SHORT_NAME', 'LONG_COMP_NAME']:
                if field in row and isinstance(row[field], str):
                    text = row[field].upper()
                    
                    # Standard Japanese indices
                    if 'NIKKEI 225' in text and 'VI' not in text and 'VOLATILITY' not in text:
                        if 'DOUBLE INVERSE' in text or 'ダブル・インバース' in text:
                            return 'Nikkei 225 Double Inverse Index'
                        elif 'INVERSE' in text or 'インバース' in text:
                            return 'Nikkei 225 Inverse Index'
                        elif 'LEVERAGED' in text or 'レバレッジ' in text or 'DOUBLE BULL' in text or 'ダブル・ブル' in text:
                            return 'Nikkei 225 Leveraged Index'
                        else:
                            return 'Nikkei 225 Index'
                    elif 'NIKKEI VI' in text or 'NIKKEI VOLATILITY' in text or '日経平均ボラティリティー' in text:
                        return 'Nikkei 225 VI Index'
                    elif 'TOPIX' in text and 'VI' not in text and 'VOLATILITY' not in text:
                        if 'DOUBLE INVERSE' in text or '-2X' in text or 'ダブル・インバース' in text:
                            return 'TOPIX Double Inverse (-2x) Index'
                        elif 'INVERSE' in text or '-1X' in text or 'インバース' in text:
                            return 'TOPIX Inverse (-1x) Index'
                        elif 'LEVERAGED' in text or '2X' in text or 'レバレッジ' in text or 'BULL' in text or 'ブル' in text:
                            return 'TOPIX Leveraged (2x) Index'
                        else:
                            return 'TOPIX Index'
                    elif 'TOPIX VI' in text or 'TOPIX VOLATILITY' in text:
                        return 'TOPIX VI Index'
                    elif 'JPX-NIKKEI 400' in text or 'JPX NIKKEI 400' in text:
                        return 'JPX Nikkei 400 Index'
                    elif 'MSCI JAPAN' in text:
                        return 'MSCI Japan Index'
                    elif 'FTSE JAPAN' in text:
                        return 'FTSE Japan Index'
                    
                    # Sector indices
                    elif 'BANKS' in text or '銀行' in text:
                        if 'NIKKEI' in text:
                            return 'Nikkei 225 Banks Index'
                        else:
                            return 'TOPIX Banks Index'
                    elif 'J-REIT' in text:
                        return 'TSE REIT Index'
            
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
                    if 'DOUBLE INVERSE' in text or '-2X' in text or 'ダブル・インバース' in text:
                        return -2.0
                    elif 'INVERSE' in text or '-1X' in text or 'インバース' in text or 'BEAR' in text or 'ベア' in text:
                        return -1.0
                    elif 'TRIPLE BULL' in text or '3X' in text:
                        return 3.0
                    elif 'DOUBLE BULL' in text or 'DOUBLE LEVERAGED' in text or '2X' in text or 'ダブル・ブル' in text:
                        return 2.0
                    
                    # More general pattern
                    matches = re.findall(r'([+-]?\d+(?:\.\d+)?)[Xx]', text)
                    if matches:
                        return float(matches[0])
            
            # Default based on type
            leverage_type = get_leverage_type(row)
            if leverage_type == 'Leveraged':
                return 2.0  # Typical leverage in Japan
            elif leverage_type == 'Inverse':
                return -1.0  # Typical inverse in Japan
            
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
            
            return 'JPY'  # Default for Japan
        
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
                
                # Check for known Japanese ETF issuers
                if 'NEXT FUNDS' in name or 'NEXTFUNDS' in name:
                    return 'NEXT FUNDS'
                elif 'NOMURA' in name:
                    return 'Nomura'
                elif 'DAIWA' in name:
                    return 'Daiwa'
                elif 'NIKKO' in name:
                    return 'Nikko'
                elif 'MITSUBISHI' in name or 'MAXIS' in name:
                    return 'MAXIS'
                elif 'ISHARES' in name or 'BLACKROCK' in name:
                    return 'iShares'
                elif 'SIMPLEX' in name:
                    return 'Simplex'
                elif 'SMT' in name or 'SUMITOMO' in name:
                    return 'SMDAM'
                elif 'ETFS' in name:
                    return 'ETFS'
                
                # Japanese ETF issuers often include their name at beginning or have specific prefixes
                if name.startswith('MAXIS '):
                    return 'MAXIS'
                elif name.startswith('NEXT '):
                    return 'NEXT FUNDS'
                elif name.startswith('DAIWA '):
                    return 'Daiwa'
                elif name.startswith('NIKKO '):
                    return 'Nikko'
                elif name.startswith('SMT '):
                    return 'SMDAM'
                
                # Try to extract first word as issuer
                parts = name.split()
                if parts:
                    return parts[0]
            
            # Try to extract from ticker code patterns
            if isinstance(ticker, str):
                ticker_match = re.search(r'^(\d+)', ticker)
                if ticker_match:
                    code = int(ticker_match.group(1))
                    
                    # Nomura (NEXT FUNDS) typically uses 1570-1571, 1357-1358 ranges
                    if code in [1357, 1358, 1365, 1366, 1367, 1570, 1571]:
                        return 'NEXT FUNDS'
                    # Daiwa typically uses 1456-1469 range
                    elif 1456 <= code <= 1469:
                        return 'Daiwa'
                    # MAXIS typically uses 1550-1559 range
                    elif 1550 <= code <= 1559:
                        return 'MAXIS'
                    # Simplex usually in 1568-1569 range
                    elif code in [1568, 1569]:
                        return 'Simplex'
                    # Nikko typically in 1579-1596 range
                    elif 1579 <= code <= 1596:
                        return 'Nikko'
                    # SMDAM typically in 2521-2529 range
                    elif 2521 <= code <= 2529:
                        return 'SMDAM'
            
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
        Create a sample dataset of Japanese leveraged/inverse ETFs for testing.
        
        Returns:
            DataFrame with sample leveraged/inverse ETF data
        """
        logger.info("Creating sample leveraged/inverse ETF data for analysis")
        
        # Create sample data for Japanese market
        data = [
            # Nomura/NEXT FUNDS ETFs - Nikkei 225
            {
                'TICKER': '1570 JT Equity',
                'NAME': 'NEXT FUNDS Nikkei 225 Leveraged Index ETF',
                'SECURITY_DES': 'NEXT FUNDS Nikkei 225 Leveraged Index ETF',
                'SECURITY_TYP2': 'ETF',
                'ISSUER': 'NEXT FUNDS',
                'LEVERAGE_TYPE': 'Leveraged',
                'UNDERLYING': 'Nikkei 225 Leveraged Index',
                'UNDERLYING_CODE': 'NKY2 Index',
                'LEVERAGE_FACTOR': 2.0,
                'AUM': 182500000000.0,
                'CURRENCY': 'JPY'
            },
            {
                'TICKER': '1571 JT Equity',
                'NAME': 'NEXT FUNDS Nikkei 225 Inverse Index ETF',
                'SECURITY_DES': 'NEXT FUNDS Nikkei 225 Inverse Index ETF',
                'SECURITY_TYP2': 'ETF',
                'ISSUER': 'NEXT FUNDS',
                'LEVERAGE_TYPE': 'Inverse',
                'UNDERLYING': 'Nikkei 225 Inverse Index',
                'UNDERLYING_CODE': 'NKYI Index',
                'LEVERAGE_FACTOR': -1.0,
                'AUM': 124600000000.0,
                'CURRENCY': 'JPY'
            },
            {
                'TICKER': '1357 JT Equity',
                'NAME': 'NEXT FUNDS Nikkei 225 Double Inverse Index ETF',
                'SECURITY_DES': 'NEXT FUNDS Nikkei 225 Double Inverse Index ETF',
                'SECURITY_TYP2': 'ETF',
                'ISSUER': 'NEXT FUNDS',
                'LEVERAGE_TYPE': 'Inverse',
                'UNDERLYING': 'Nikkei 225 Double Inverse Index',
                'UNDERLYING_CODE': 'NKY2I Index',
                'LEVERAGE_FACTOR': -2.0,
                'AUM': 152800000000.0,
                'CURRENCY': 'JPY'
            },
            
            # Nomura/NEXT FUNDS ETFs - TOPIX
            {
                'TICKER': '1365 JT Equity',
                'NAME': 'NEXT FUNDS TOPIX Leveraged (2x) Index ETF',
                'SECURITY_DES': 'NEXT FUNDS TOPIX Leveraged (2x) Index ETF',
                'SECURITY_TYP2': 'ETF',
                'ISSUER': 'NEXT FUNDS',
                'LEVERAGE_TYPE': 'Leveraged',
                'UNDERLYING': 'TOPIX Leveraged (2x) Index',
                'UNDERLYING_CODE': 'TPX2 Index',
                'LEVERAGE_FACTOR': 2.0,
                'AUM': 69500000000.0,
                'CURRENCY': 'JPY'
            },
            {
                'TICKER': '1366 JT Equity',
                'NAME': 'NEXT FUNDS TOPIX Inverse (-1x) Index ETF',
                'SECURITY_DES': 'NEXT FUNDS TOPIX Inverse (-1x) Index ETF',
                'SECURITY_TYP2': 'ETF',
                'ISSUER': 'NEXT FUNDS',
                'LEVERAGE_TYPE': 'Inverse',
                'UNDERLYING': 'TOPIX Inverse (-1x) Index',
                'UNDERLYING_CODE': 'TPXI Index',
                'LEVERAGE_FACTOR': -1.0,
                'AUM': 32900000000.0,
                'CURRENCY': 'JPY'
            },
            {
                'TICKER': '1367 JT Equity',
                'NAME': 'NEXT FUNDS TOPIX Double Inverse (-2x) Index ETF',
                'SECURITY_DES': 'NEXT FUNDS TOPIX Double Inverse (-2x) Index ETF',
                'SECURITY_TYP2': 'ETF',
                'ISSUER': 'NEXT FUNDS',
                'LEVERAGE_TYPE': 'Inverse',
                'UNDERLYING': 'TOPIX Double Inverse (-2x) Index',
                'UNDERLYING_CODE': 'TPX2I Index',
                'LEVERAGE_FACTOR': -2.0,
                'AUM': 43800000000.0,
                'CURRENCY': 'JPY'
            },
            
            # Daiwa ETFs - Nikkei 225
            {
                'TICKER': '1456 JT Equity',
                'NAME': 'Daiwa ETF Nikkei 225 Leveraged Index',
                'SECURITY_DES': 'Daiwa ETF Nikkei 225 Leveraged Index',
                'SECURITY_TYP2': 'ETF',
                'ISSUER': 'Daiwa',
                'LEVERAGE_TYPE': 'Leveraged',
                'UNDERLYING': 'Nikkei 225 Leveraged Index',
                'UNDERLYING_CODE': 'NKY2 Index',
                'LEVERAGE_FACTOR': 2.0,
                'AUM': 97300000000.0,
                'CURRENCY': 'JPY'
            },
            {
                'TICKER': '1457 JT Equity',
                'NAME': 'Daiwa ETF Nikkei 225 Inverse Index',
                'SECURITY_DES': 'Daiwa ETF Nikkei 225 Inverse Index',
                'SECURITY_TYP2': 'ETF',
                'ISSUER': 'Daiwa',
                'LEVERAGE_TYPE': 'Inverse',
                'UNDERLYING': 'Nikkei 225 Inverse Index',
                'UNDERLYING_CODE': 'NKYI Index',
                'LEVERAGE_FACTOR': -1.0,
                'AUM': 62100000000.0,
                'CURRENCY': 'JPY'
            },
            {
                'TICKER': '1458 JT Equity',
                'NAME': 'Daiwa ETF Nikkei 225 Double Inverse Index',
                'SECURITY_DES': 'Daiwa ETF Nikkei 225 Double Inverse Index',
                'SECURITY_TYP2': 'ETF',
                'ISSUER': 'Daiwa',
                'LEVERAGE_TYPE': 'Inverse',
                'UNDERLYING': 'Nikkei 225 Double Inverse Index',
                'UNDERLYING_CODE': 'NKY2I Index',
                'LEVERAGE_FACTOR': -2.0,
                'AUM': 76200000000.0,
                'CURRENCY': 'JPY'
            },
            
            # Daiwa ETFs - TOPIX
            {
                'TICKER': '1461 JT Equity',
                'NAME': 'Daiwa ETF TOPIX Leveraged (2x) Index',
                'SECURITY_DES': 'Daiwa ETF TOPIX Leveraged (2x) Index',
                'SECURITY_TYP2': 'ETF',
                'ISSUER': 'Daiwa',
                'LEVERAGE_TYPE': 'Leveraged',
                'UNDERLYING': 'TOPIX Leveraged (2x) Index',
                'UNDERLYING_CODE': 'TPX2 Index',
                'LEVERAGE_FACTOR': 2.0,
                'AUM': 41900000000.0,
                'CURRENCY': 'JPY'
            },
            {
                'TICKER': '1462 JT Equity',
                'NAME': 'Daiwa ETF TOPIX Inverse (-1x) Index',
                'SECURITY_DES': 'Daiwa ETF TOPIX Inverse (-1x) Index',
                'SECURITY_TYP2': 'ETF',
                'ISSUER': 'Daiwa',
                'LEVERAGE_TYPE': 'Inverse',
                'UNDERLYING': 'TOPIX Inverse (-1x) Index',
                'UNDERLYING_CODE': 'TPXI Index',
                'LEVERAGE_FACTOR': -1.0,
                'AUM': 19800000000.0,
                'CURRENCY': 'JPY'
            },
            {
                'TICKER': '1460 JT Equity',
                'NAME': 'Daiwa ETF TOPIX Double Inverse (-2x) Index',
                'SECURITY_DES': 'Daiwa ETF TOPIX Double Inverse (-2x) Index',
                'SECURITY_TYP2': 'ETF',
                'ISSUER': 'Daiwa',
                'LEVERAGE_TYPE': 'Inverse',
                'UNDERLYING': 'TOPIX Double Inverse (-2x) Index',
                'UNDERLYING_CODE': 'TPX2I Index',
                'LEVERAGE_FACTOR': -2.0,
                'AUM': 36200000000.0,
                'CURRENCY': 'JPY'
            },
            
            # MAXIS ETFs (Mitsubishi UFJ)
            {
                'TICKER': '1550 JT Equity',
                'NAME': 'MAXIS TOPIX Leveraged Index ETF',
                'SECURITY_DES': 'MAXIS TOPIX Leveraged Index ETF',
                'SECURITY_TYP2': 'ETF',
                'ISSUER': 'MAXIS',
                'LEVERAGE_TYPE': 'Leveraged',
                'UNDERLYING': 'TOPIX Leveraged (2x) Index',
                'UNDERLYING_CODE': 'TPX2 Index',
                'LEVERAGE_FACTOR': 2.0,
                'AUM': 23800000000.0,
                'CURRENCY': 'JPY'
            },
            {
                'TICKER': '1551 JT Equity',
                'NAME': 'MAXIS TOPIX Inverse Index ETF',
                'SECURITY_DES': 'MAXIS TOPIX Inverse Index ETF',
                'SECURITY_TYP2': 'ETF',
                'ISSUER': 'MAXIS',
                'LEVERAGE_TYPE': 'Inverse',
                'UNDERLYING': 'TOPIX Inverse (-1x) Index',
                'UNDERLYING_CODE': 'TPXI Index',
                'LEVERAGE_FACTOR': -1.0,
                'AUM': 15700000000.0,
                'CURRENCY': 'JPY'
            },
            
            # Simplex ETFs
            {
                'TICKER': '1568 JT Equity',
                'NAME': 'Simplex TOPIX Bull 2x (Double Bull) ETF',
                'SECURITY_DES': 'Simplex TOPIX Bull 2x (Double Bull) ETF',
                'SECURITY_TYP2': 'ETF',
                'ISSUER': 'Simplex',
                'LEVERAGE_TYPE': 'Leveraged',
                'UNDERLYING': 'TOPIX Leveraged (2x) Index',
                'UNDERLYING_CODE': 'TPX2 Index',
                'LEVERAGE_FACTOR': 2.0,
                'AUM': 21600000000.0,
                'CURRENCY': 'JPY'
            },
            {
                'TICKER': '1569 JT Equity',
                'NAME': 'Simplex TOPIX Bear 2x (Double Bear) ETF',
                'SECURITY_DES': 'Simplex TOPIX Bear 2x (Double Bear) ETF',
                'SECURITY_TYP2': 'ETF',
                'ISSUER': 'Simplex',
                'LEVERAGE_TYPE': 'Inverse',
                'UNDERLYING': 'TOPIX Double Inverse (-2x) Index',
                'UNDERLYING_CODE': 'TPX2I Index',
                'LEVERAGE_FACTOR': -2.0,
                'AUM': 34700000000.0,
                'CURRENCY': 'JPY'
            },
            
            # Nikko ETFs
            {
                'TICKER': '1579 JT Equity',
                'NAME': 'Nikko Listed Index Fund Nikkei 225 Leveraged Index',
                'SECURITY_DES': 'Nikko Listed Index Fund Nikkei 225 Leveraged Index',
                'SECURITY_TYP2': 'ETF',
                'ISSUER': 'Nikko',
                'LEVERAGE_TYPE': 'Leveraged',
                'UNDERLYING': 'Nikkei 225 Leveraged Index',
                'UNDERLYING_CODE': 'NKY2 Index',
                'LEVERAGE_FACTOR': 2.0,
                'AUM': 45300000000.0,
                'CURRENCY': 'JPY'
            },
            {
                'TICKER': '1580 JT Equity',
                'NAME': 'Nikko Listed Index Fund Nikkei 225 Inverse Index',
                'SECURITY_DES': 'Nikko Listed Index Fund Nikkei 225 Inverse Index',
                'SECURITY_TYP2': 'ETF',
                'ISSUER': 'Nikko',
                'LEVERAGE_TYPE': 'Inverse',
                'UNDERLYING': 'Nikkei 225 Inverse Index',
                'UNDERLYING_CODE': 'NKYI Index',
                'LEVERAGE_FACTOR': -1.0,
                'AUM': 29800000000.0,
                'CURRENCY': 'JPY'
            },
            
            # SMDAM ETFs
            {
                'TICKER': '2526 JT Equity',
                'NAME': 'SMT TPX BULL 2X',
                'SECURITY_DES': 'SMT TOPIX Bull 2X ETF',
                'SECURITY_TYP2': 'ETF',
                'ISSUER': 'SMDAM',
                'LEVERAGE_TYPE': 'Leveraged',
                'UNDERLYING': 'TOPIX Leveraged (2x) Index',
                'UNDERLYING_CODE': 'TPX2 Index',
                'LEVERAGE_FACTOR': 2.0,
                'AUM': 19100000000.0,
                'CURRENCY': 'JPY'
            },
            {
                'TICKER': '2527 JT Equity',
                'NAME': 'SMT TPX BEAR -1X',
                'SECURITY_DES': 'SMT TOPIX Bear -1X ETF',
                'SECURITY_TYP2': 'ETF',
                'ISSUER': 'SMDAM',
                'LEVERAGE_TYPE': 'Inverse',
                'UNDERLYING': 'TOPIX Inverse (-1x) Index',
                'UNDERLYING_CODE': 'TPXI Index',
                'LEVERAGE_FACTOR': -1.0,
                'AUM': 12300000000.0,
                'CURRENCY': 'JPY'
            },
            
            # Volatility ETFs
            {
                'TICKER': '1358 JT Equity',
                'NAME': 'NEXT FUNDS Nikkei 225 VI ETF',
                'SECURITY_DES': 'NEXT FUNDS Nikkei 225 Volatility Index ETF',
                'SECURITY_TYP2': 'ETF',
                'ISSUER': 'NEXT FUNDS',
                'LEVERAGE_TYPE': 'Leveraged',
                'UNDERLYING': 'Nikkei 225 VI Index',
                'UNDERLYING_CODE': 'NKVI Index',
                'LEVERAGE_FACTOR': 1.0,  # Not really leveraged but often grouped with LI ETFs
                'AUM': 18600000000.0,
                'CURRENCY': 'JPY'
            },
            {
                'TICKER': '1467 JT Equity',
                'NAME': 'Daiwa ETF TOPIX Volatility Index',
                'SECURITY_DES': 'Daiwa ETF TOPIX Volatility Index',
                'SECURITY_TYP2': 'ETF',
                'ISSUER': 'Daiwa',
                'LEVERAGE_TYPE': 'Leveraged',
                'UNDERLYING': 'TOPIX VI Index',
                'UNDERLYING_CODE': 'TPXVI Index',
                'LEVERAGE_FACTOR': 1.0,  # Not really leveraged but often grouped with LI ETFs
                'AUM': 9800000000.0,
                'CURRENCY': 'JPY'
            },
            
            # Unknown examples
            {
                'TICKER': '9999 JT Equity',
                'NAME': 'Unknown Japanese Leveraged ETF',
                'SECURITY_DES': 'Unknown Leveraged ETF',
                'SECURITY_TYP2': 'ETF',
                'ISSUER': 'Unknown',
                'LEVERAGE_TYPE': 'Leveraged',
                'UNDERLYING': 'Unknown',
                'UNDERLYING_CODE': 'Unknown',
                'LEVERAGE_FACTOR': 2.0,
                'AUM': 8700000000.0,
                'CURRENCY': 'JPY'
            },
            {
                'TICKER': '9998 JT Equity',
                'NAME': 'Unknown Japanese Inverse ETF',
                'SECURITY_DES': 'Unknown Inverse ETF',
                'SECURITY_TYP2': 'ETF',
                'ISSUER': 'Unknown',
                'LEVERAGE_TYPE': 'Inverse',
                'UNDERLYING': 'Unknown',
                'UNDERLYING_CODE': 'Unknown',
                'LEVERAGE_FACTOR': -1.0,
                'AUM': 6400000000.0,
                'CURRENCY': 'JPY'
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
    
    def aggregate_by_index_family(self, etf_df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate ETF data by index family (Nikkei, TOPIX, etc.).
        
        Args:
            etf_df: DataFrame with ETF data
            
        Returns:
            DataFrame with aggregated data by index family
        """
        if etf_df.empty:
            return pd.DataFrame()
        
        # Function to extract index family from underlying name
        def get_index_family(underlying):
            if pd.isna(underlying) or underlying == 'Unknown':
                return 'Unknown'
                
            if 'NIKKEI' in underlying.upper() or 'NKY' in underlying.upper():
                return 'Nikkei'
            elif 'TOPIX' in underlying.upper() or 'TPX' in underlying.upper():
                return 'TOPIX'
            elif 'JPX' in underlying.upper():
                return 'JPX-Nikkei'
            elif 'REIT' in underlying.upper():
                return 'REIT'
            elif 'MSCI' in underlying.upper():
                return 'MSCI'
            elif 'FTSE' in underlying.upper():
                return 'FTSE'
            else:
                return 'Other'
        
        # Add index family classification
        etf_df_with_family = etf_df.copy()
        etf_df_with_family['INDEX_FAMILY'] = etf_df_with_family['UNDERLYING'].apply(get_index_family)
        
        # Group by index family and leverage type
        grouped = etf_df_with_family.groupby(['INDEX_FAMILY', 'LEVERAGE_TYPE'])
        
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
        
        # Sort by index family first, then leverage type
        agg_data = agg_data.sort_values(['INDEX_FAMILY', 'LEVERAGE_TYPE'])
        
        # Calculate percentage of total AUM
        total_aum = agg_data['TOTAL_AUM'].sum()
        agg_data['AUM_PCT'] = agg_data['TOTAL_AUM'] / total_aum * 100 if total_aum > 0 else 0
        
        return agg_data
    
    def track_li_funds(self) -> Dict[str, Any]:
        """
        Track leveraged and inverse funds in Japan.
        
        Returns:
            Dictionary with fund data and analysis
        """
        # Find leveraged and inverse ETFs
        li_etfs = self.get_jp_leveraged_inverse_funds()
        
        # Set latest update time
        self.latest_update_time = datetime.datetime.now()
        
        # Aggregate by underlying
        agg_by_underlying = self.aggregate_by_underlying(li_etfs)
        
        # Aggregate by issuer
        agg_by_issuer = self.aggregate_by_issuer(li_etfs)
        
        # Aggregate by leverage factor
        agg_by_leverage = self.aggregate_by_leverage_factor(li_etfs)
        
        # Aggregate by index family
        agg_by_family = self.aggregate_by_index_family(li_etfs)
        
        # Store the results
        self.li_fund_data = {
            'Update_Time': self.latest_update_time,
            'ETF_List': li_etfs.to_dict('records') if not li_etfs.empty else [],
            'Agg_By_Underlying': agg_by_underlying.to_dict('records') if not agg_by_underlying.empty else [],
            'Agg_By_Issuer': agg_by_issuer.to_dict('records') if not agg_by_issuer.empty else [],
            'Agg_By_Leverage': agg_by_leverage.to_dict('records') if not agg_by_leverage.empty else [],
            'Agg_By_Family': agg_by_family.to_dict('records') if not agg_by_family.empty else []
        }
        
        return self.li_fund_data
    
    def format_aum(self, aum_value: float) -> str:
        """
        Format AUM value for display in JPY.
        
        Args:
            aum_value: AUM value
            
        Returns:
            Formatted AUM string
        """
        if aum_value >= 1e12:
            return f"¥{aum_value/1e12:.2f}T"
        elif aum_value >= 1e9:
            return f"¥{aum_value/1e9:.2f}B"
        elif aum_value >= 1e6:
            return f"¥{aum_value/1e6:.2f}M"
        else:
            return f"¥{aum_value:.2f}"
    
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
        print(f"LEVERAGED AND INVERSE ETFs IN JAPAN - {self.latest_update_time.strftime('%Y-%m-%d %H:%M:%S')}")
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
            summary_data.sort(key=lambda x: float(str(x[3]).replace('¥', '').replace('T', 'e12').replace('B', 'e9').replace('M', 'e6')), reverse=True)
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
    
    def print_aum_by_index_family(self) -> None:
        """Print AUM aggregated by index family (Nikkei, TOPIX, etc.)."""
        if not self.li_fund_data or 'Agg_By_Family' not in self.li_fund_data:
            logger.error("No aggregated data available. Run track_li_funds() first.")
            return
        
        agg_data = self.li_fund_data.get('Agg_By_Family', [])
        if not agg_data:
            logger.warning("No aggregated data found to display.")
            return
        
        print("\n" + "="*100)
        print(f"AUM BY INDEX FAMILY - {self.latest_update_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*100 + "\n")
        
        # Prepare table data
        table_data = []
        
        for row in agg_data:
            index_family = row.get('INDEX_FAMILY', '')
            leverage_type = row.get('LEVERAGE_TYPE', '')
            num_funds = row.get('NUM_FUNDS', 0)
            total_aum = row.get('TOTAL_AUM', 0)
            aum_pct = row.get('AUM_PCT', 0)
            
            # Highlight unknown index family
            if index_family == 'Unknown':
                index_family = self.highlight_unknown(index_family)
            
            table_data.append([
                index_family,
                leverage_type,
                num_funds,
                self.format_aum(total_aum),
                f"{aum_pct:.1f}%"
            ])
        
        # Print the table
        headers = ["Index Family", "Type", "# of Funds", "Total AUM", "% of Total"]
        print(tabulate(table_data, headers=headers, tablefmt="psql"))
        
        # Add a summary by index family only (combining leveraged and inverse)
        print("\n" + "-"*100)
        print("SUMMARY BY INDEX FAMILY (ALL TYPES)")
        print("-"*100 + "\n")
        
        # Group by index family
        family_summary = {}
        for row in agg_data:
            index_family = row.get('INDEX_FAMILY', '')
            num_funds = row.get('NUM_FUNDS', 0)
            total_aum = row.get('TOTAL_AUM', 0)
            
            if index_family not in family_summary:
                family_summary[index_family] = {
                    'NUM_FUNDS': 0,
                    'TOTAL_AUM': 0
                }
            
            family_summary[index_family]['NUM_FUNDS'] += num_funds
            family_summary[index_family]['TOTAL_AUM'] += total_aum
        
        # Calculate percentages
        grand_total = sum(data['TOTAL_AUM'] for data in family_summary.values())
        
        # Prepare summary table
        summary_data = []
        for index_family, data in family_summary.items():
            aum_pct = data['TOTAL_AUM'] / grand_total * 100 if grand_total > 0 else 0
            
            # Highlight unknown index family
            highlighted_family = self.highlight_unknown(index_family) if index_family == 'Unknown' else index_family
            
            summary_data.append([
                highlighted_family,
                data['NUM_FUNDS'],
                self.format_aum(data['TOTAL_AUM']),
                f"{aum_pct:.1f}%"
            ])
        
        # Sort by total AUM
        try:
            summary_data.sort(key=lambda x: float(str(x[2]).replace('¥', '').replace('T', 'e12').replace('B', 'e9').replace('M', 'e6')), reverse=True)
        except:
            # Fallback if sorting fails
            pass
        
        # Print the summary table
        headers = ["Index Family", "# of Funds", "Total AUM", "% of Total"]
        print(tabulate(summary_data, headers=headers, tablefmt="psql"))
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
            etf_file = os.path.join(output_dir, f"jp_li_etfs_{timestamp}.csv")
            
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
            agg_file = os.path.join(output_dir, f"jp_li_agg_by_underlying_{timestamp}.csv")
            
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
            agg_file = os.path.join(output_dir, f"jp_li_agg_by_issuer_{timestamp}.csv")
            
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
            agg_file = os.path.join(output_dir, f"jp_li_agg_by_leverage_{timestamp}.csv")
            
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
        
        # Save aggregated data by index family
        agg_family = self.li_fund_data.get('Agg_By_Family', [])
        if agg_family:
            agg_file = os.path.join(output_dir, f"jp_li_agg_by_index_family_{timestamp}.csv")
            
            # Convert to DataFrame if it's a list of dictionaries
            if isinstance(agg_family, list):
                agg_df = pd.DataFrame(agg_family)
            else:
                agg_df = pd.DataFrame([])
            
            if not agg_df.empty:
                # Save to CSV
                agg_df.to_csv(agg_file, index=False)
                logger.info(f"Saved aggregated data by index family to {agg_file}")
            else:
                logger.warning("No aggregated data by index family to save")


def main():
    parser = argparse.ArgumentParser(description='Track and analyze Leveraged and Inverse ETFs in Japan')
    parser.add_argument('--host', default='127.0.0.1', help='Bloomberg server host')
    parser.add_argument('--port', type=int, default=8194, help='Bloomberg server port')
    parser.add_argument('--output-dir', default='./jp_li_funds', help='Directory to save output files')
    parser.add_argument('--sample', action='store_true', 
                        help='Use sample data instead of Bloomberg data')
    
    args = parser.parse_args()
    
    # Create output directory if needed
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize tracker
    tracker = JPLeverageFundsTracker(host=args.host, port=args.port, use_sample_data=args.sample)
    
    try:
        # Start session
        if tracker.start_session():
            # Track leveraged and inverse funds
            logger.info("Tracking leveraged and inverse funds in Japan...")
            tracker.track_li_funds()
            
            # Print ETF list
            tracker.print_etf_list()
            
            # Print AUM by underlying
            tracker.print_aum_by_underlying()
            
            # Print AUM by issuer
            tracker.print_aum_by_issuer()
            
            # Print AUM by leverage factor
            tracker.print_aum_by_leverage()
            
            # Print AUM by index family
            tracker.print_aum_by_index_family()
            
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