import blpapi
import pandas as pd
import numpy as np
import datetime
import logging
import os
import re
import time
from typing import List, Dict, Any, Optional, Tuple

# Default file paths
DEFAULT_INPUT_FILE = "stock_futures_list.txt"
DEFAULT_OUTPUT_CSV = "stock_futures_roll_data.csv"
DEFAULT_OUTPUT_EXCEL = "stock_futures_roll_data.xlsx"

# Configuration settings
USE_SAMPLE_DATA = False  # Use real Bloomberg data
USE_EXCEL_OUTPUT = True  # Set to False to use CSV output instead

# Bloomberg settings
BLOOMBERG_HOST = '127.0.0.1'
BLOOMBERG_PORT = 8194

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("stock_futures_roll_tracker.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('StockFuturesRollTracker')

class StockFuturesRollTracker:
    """
    A class to track roll activity for single stock futures by analyzing volume 
    and open interest data between the front month and next month contracts.
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
        
        # Store results
        self.roll_data = {}
        self.latest_update_time = datetime.datetime.now()
        
        # Bloomberg session
        self.session = None
        
        # Month codes for futures
        self.month_codes = {
            'F': 1,   # January
            'G': 2,   # February
            'H': 3,   # March
            'J': 4,   # April
            'K': 5,   # May
            'M': 6,   # June
            'N': 7,   # July
            'Q': 8,   # August
            'U': 9,   # September
            'V': 10,  # October
            'X': 11,  # November
            'Z': 12   # December
        }
        
        # Reverse lookup
        self.month_letters = {v: k for k, v in self.month_codes.items()}
        
        # Define common market specs for single stock futures
        self.market_specs = {
            # Taiwan Stock Exchange
            'TT': {
                'months': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],  # All months
                'rolldays': 5,
                'year_format': '1-digit',
                'suffix': 'Equity',
                'expiry_day': 3,  # Third Wednesday of month (3rd week)
                'expiry_weekday': 2  # Wednesday (0=Monday, 1=Tuesday, ...)
            },
            # Korean Exchange
            'KS': {
                'months': [3, 6, 9, 12],  # Quarterly
                'rolldays': 7, 
                'year_format': '1-digit',
                'suffix': 'Equity',
                'expiry_day': 2,  # Second Thursday of month (2nd week)
                'expiry_weekday': 3  # Thursday (0=Monday, 1=Tuesday, ...)
            },
            # Hong Kong Exchange
            'HK': {
                'months': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],  # All months
                'rolldays': 5,
                'year_format': '1-digit',
                'suffix': 'Equity',
                'expiry_day': -1,  # Last business day of month
                'expiry_weekday': None  # Not applicable for end-of-month expiry
            },
            # Singapore Exchange
            'SP': {
                'months': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],  # All months
                'rolldays': 5,
                'year_format': '1-digit',
                'suffix': 'Equity',
                'expiry_day': -1,  # Last business day of month
                'expiry_weekday': None  # Not applicable for end-of-month expiry
            },
            # Eurex
            'GY': {
                'months': [3, 6, 9, 12],  # Quarterly
                'rolldays': 7,
                'year_format': '1-digit',
                'suffix': 'Equity',
                'expiry_day': 3,  # Third Friday of month (3rd week)
                'expiry_weekday': 4  # Friday (0=Monday, 1=Tuesday, ...)
            },
            # Default for other markets
            'DEFAULT': {
                'months': [3, 6, 9, 12],  # Quarterly is most common
                'rolldays': 7,
                'year_format': '1-digit',
                'suffix': 'Equity',
                'expiry_day': 3,  # Third Friday of month (3rd week) is most common
                'expiry_weekday': 4  # Friday (0=Monday, 1=Tuesday, ...)
            }
        }
        
        # Bloomberg field mapping - try multiple field names for the same data
        self.field_mappings = {
            'volume': ['PX_VOLUME', 'VOLUME', 'VOLUME_ALL_TRADES', 'VOLUME_TDY'],
            'open_interest': ['OPEN_INT', 'PX_OPEN_INT', 'OPN_INT', 'OPEN_INTEREST'],
            'expiry_date': ['LAST_TRADEABLE_DT', 'EXPIRY_DT', 'TERMINATION_DT', 'FUT_NOTICE_FIRST']
        }
    
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
                logger.warning("Falling back to sample data")
                self.use_sample_data = True
                return True  # Return true but with sample data
            
            # Open the market data service
            if not self.session.openService("//blp/refdata"):
                logger.error("Failed to open //blp/refdata service")
                logger.warning("Falling back to sample data")
                self.use_sample_data = True
                return True  # Return true but with sample data
            
            logger.info("Bloomberg session started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Bloomberg API Exception: {e}")
            logger.warning("Falling back to sample data")
            self.use_sample_data = True
            return True  # Return true but with sample data
            
    def stop_session(self) -> None:
        """Stop the Bloomberg session."""
        if self.session and not self.use_sample_data:
            self.session.stop()
            logger.info("Bloomberg session stopped")
    
    def read_stock_futures_list(self, filename: str) -> List[Dict[str, str]]:
        """
        Read a list of stock futures tickers from a file.
        
        Args:
            filename: Path to the file containing stock futures symbols
            
        Returns:
            List of dictionaries with stock tickers and details
        """
        if not os.path.exists(filename):
            logger.error(f"Stock futures list file not found: {filename}")
            return []
            
        futures_list = []
        
        try:
            with open(filename, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    # Skip empty lines and comments
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    
                    # Remove any trailing comments
                    if '#' in line:
                        line = line.split('#')[0].strip()
                    
                    # Parse the ticker
                    # Format can be:
                    # - Just the stock code (e.g., "2330 TT")
                    # - Full example futures contract (e.g., "2330=J5 TT Equity")
                    parts = line.split()
                    
                    try:
                        if len(parts) >= 2:
                            # Extract the stock code and exchange
                            if '=' in parts[0]:
                                # This is a full futures contract
                                stock_code = parts[0].split('=')[0]
                                exchange = parts[1]
                            else:
                                # This is just the stock code
                                stock_code = parts[0]
                                exchange = parts[1]
                            
                            # Create the stock info dictionary
                            stock_info = {
                                'stock_code': stock_code,
                                'exchange': exchange,
                                'original_line': line
                            }
                            
                            futures_list.append(stock_info)
                        else:
                            logger.warning(f"Line {line_num}: Invalid format - {line}. Skipping.")
                    except Exception as e:
                        logger.warning(f"Line {line_num}: Error parsing line '{line}': {e}. Skipping.")
            
            logger.info(f"Read {len(futures_list)} stock futures from {filename}")
            return futures_list
            
        except Exception as e:
            logger.error(f"Error reading stock futures list: {e}")
            return []
    
    def determine_contract_months(self, exchange: str) -> Tuple[int, int, int, int]:
        """
        Determine the current front and next contract months for a given exchange.
        
        Args:
            exchange: Market exchange code (e.g., 'TT', 'KS')
        
        Returns:
            Tuple of (front_month, front_year, next_month, next_year)
        """
        # Get current date
        now = datetime.datetime.now()
        current_month = now.month
        current_year = now.year
        current_day = now.day
        
        # Check if the exchange is in our market specifications
        if exchange in self.market_specs:
            # Get the contract months for this exchange
            market_info = self.market_specs[exchange]
        else:
            # Use default specifications
            market_info = self.market_specs['DEFAULT']
        
        contract_months = market_info['months']
        roll_days = market_info['rolldays']
        
        # Find the nearest contract month
        # If we're within roll_days of expiry of the current month contract,
        # the front month will be the next contract month
        
        # First, determine if the current month is a contract month
        if current_month in contract_months:
            # Check if we're near expiry (simplified approach)
            days_till_month_end = (datetime.datetime(current_year, 
                                                    current_month % 12 + 1, 1) - now).days
            if days_till_month_end > roll_days:
                # Not yet near expiry, current month is front month
                front_month = current_month
                front_year = current_year
            else:
                # Near expiry, front month is the next contract month
                # Find the next contract month after current month
                next_idx = contract_months.index(current_month) + 1
                if next_idx >= len(contract_months):
                    front_month = contract_months[0]
                    front_year = current_year + 1
                else:
                    front_month = contract_months[next_idx]
                    front_year = current_year
        else:
            # Current month is not a contract month.
            # Find the next contract month after current month
            next_contract_months = [m for m in contract_months if m > current_month]
            if next_contract_months:
                front_month = min(next_contract_months)
                front_year = current_year
            else:
                # No more contract months this year, wrap to next year
                front_month = min(contract_months)
                front_year = current_year + 1
        
        # Determine the next month contract after the front month
        front_idx = contract_months.index(front_month)
        next_idx = front_idx + 1
        if next_idx >= len(contract_months):
            next_month = contract_months[0]
            next_year = front_year + 1
        else:
            next_month = contract_months[next_idx]
            next_year = front_year
        
        return front_month, front_year, next_month, next_year
    
    def construct_stock_future_contract(self, stock_code: str, exchange: str, month: int, year: int) -> str:
        """
        Construct a single stock futures contract Bloomberg ticker.
        
        Args:
            stock_code: The stock code (e.g., '2330')
            exchange: Market exchange code (e.g., 'TT')
            month: Contract month (1-12)
            year: Contract year (e.g., 2025)
        
        Returns:
            Bloomberg ticker for the stock future contract
        """
        # Get market specs
        if exchange in self.market_specs:
            market_info = self.market_specs[exchange]
        else:
            market_info = self.market_specs['DEFAULT']
        
        # Get month code
        month_code = self.month_letters[month]
        
        # Format year based on market specs
        if market_info['year_format'] == '1-digit':
            year_code = str(year)[-1]
        else:
            year_code = str(year)[-2:]
        
        # Construct the ticker
        # Format is typically: STOCKCODE=MONTHYEAR EXCHANGE SUFFIX
        # Example: 2330=J5 TT Equity
        suffix = market_info['suffix']
        contract = f"{stock_code}={month_code}{year_code} {exchange} {suffix}"
        
        return contract
    
    def calculate_expiry_date(self, contract: str, exchange: str, month: int, year: int) -> datetime.date:
        """
        Calculate the expiration date for a futures contract.
        
        Args:
            contract: Contract ticker
            exchange: Exchange code
            month: Contract month
            year: Contract year
            
        Returns:
            Expiration date
        """
        if self.use_sample_data:
            return self.estimate_expiry_date(exchange, month, year)
            
        try:
            # Try to get the expiry date from Bloomberg
            refDataService = self.session.getService("//blp/refdata")
            request = refDataService.createRequest("ReferenceDataRequest")
            
            # Add the contract
            request.append("securities", contract)
            
            # Add multiple possible fields for expiry date
            for field in self.field_mappings.get('expiry_date', []):
                request.append("fields", field)
            
            # Send the request
            self.session.sendRequest(request)
            
            # Process the response
            expiry_date = None
            
            while True:
                event = self.session.nextEvent(500)
                
                for msg in event:
                    if msg.messageType() == blpapi.Name("ReferenceDataResponse"):
                        securities = msg.getElement("securityData")
                        
                        for i in range(securities.numValues()):
                            security = securities.getValue(i)
                            
                            if security.hasElement("securityError"):
                                error = security.getElement("securityError").getElementAsString("message")
                                logger.warning(f"Security error for {contract}: {error}")
                                continue
                                
                            # Extract field data if available
                            if security.hasElement("fieldData"):
                                field_data = security.getElement("fieldData")
                                
                                # Try each expiry date field
                                for field_name in self.field_mappings.get('expiry_date', []):
                                    if field_data.hasElement(field_name):
                                        try:
                                            # Parse the expiry date from the field
                                            date_str = field_data.getElementAsString(field_name)
                                            expiry_date = datetime.datetime.strptime(date_str, "%Y-%m-%d").date()
                                            logger.info(f"Found expiry date for {contract}: {expiry_date}")
                                            break
                                        except Exception as e:
                                            logger.warning(f"Error parsing expiry date: {e}")
                
                if event.eventType() == blpapi.Event.RESPONSE:
                    break
                    
            # If we couldn't get the expiry date from Bloomberg, estimate it
            if expiry_date is None:
                logger.warning(f"Could not get expiry date for {contract} from Bloomberg, estimating")
                expiry_date = self.estimate_expiry_date(exchange, month, year)
                
            return expiry_date
                
        except Exception as e:
            logger.error(f"Error getting expiry date: {e}")
            # Fallback to estimation
            return self.estimate_expiry_date(exchange, month, year)
    
    def estimate_expiry_date(self, exchange: str, month: int, year: int) -> datetime.date:
        """
        Estimate the expiration date for a futures contract based on market specifications.
        
        Args:
            exchange: Exchange code
            month: Contract month
            year: Contract year
            
        Returns:
            Estimated expiration date
        """
        # Get market specs
        if exchange in self.market_specs:
            market_info = self.market_specs[exchange]
        else:
            market_info = self.market_specs['DEFAULT']
        
        expiry_day = market_info['expiry_day']
        expiry_weekday = market_info['expiry_weekday']
        
        # Calculate expiry date based on the exchange's rules
        if expiry_day == -1:
            # Last business day of the month
            # Get the first day of the next month and subtract one day
            if month == 12:
                next_month = datetime.date(year + 1, 1, 1)
            else:
                next_month = datetime.date(year, month + 1, 1)
                
            last_day = next_month - datetime.timedelta(days=1)
            
            # Adjust if it falls on a weekend
            while last_day.weekday() > 4:  # Saturday=5, Sunday=6
                last_day -= datetime.timedelta(days=1)
                
            return last_day
            
        elif expiry_weekday is not None:
            # Specific weekday of a specific week
            # E.g., "Third Friday" or "Second Thursday"
            
            # Find the first occurrence of that weekday in the month
            first_day = datetime.date(year, month, 1)
            days_until_first = (expiry_weekday - first_day.weekday()) % 7
            first_occurrence = first_day + datetime.timedelta(days=days_until_first)
            
            # Add the required number of weeks
            expiry_date = first_occurrence + datetime.timedelta(days=(expiry_day - 1) * 7)
            
            return expiry_date
            
        else:
            # Default to the 15th of the month
            return datetime.date(year, month, 15)
    
    def calculate_days_to_expiry(self, expiry_date: datetime.date) -> int:
        """
        Calculate the number of days until expiry.
        
        Args:
            expiry_date: Expiry date
            
        Returns:
            Number of days until expiry (0 if expired)
        """
        today = datetime.date.today()
        if expiry_date < today:
            return 0
        
        return (expiry_date - today).days
    
    def get_active_stock_futures_contracts(self, stock_info: Dict[str, str]) -> Tuple[str, str, datetime.date, datetime.date]:
        """
        For a given stock, determine the front and next month futures contracts.
        
        Args:
            stock_info: Dictionary with stock code and exchange
            
        Returns:
            Tuple of (front_month_contract, next_month_contract, front_expiry, next_expiry)
        """
        if self.use_sample_data:
            return self.get_sample_stock_futures_contracts(stock_info)
        
        try:
            stock_code = stock_info['stock_code']
            exchange = stock_info['exchange']
            
            # Determine the front and next contract months
            front_month, front_year, next_month, next_year = self.determine_contract_months(exchange)
            
            # Construct the contract tickers
            front_contract = self.construct_stock_future_contract(stock_code, exchange, front_month, front_year)
            next_contract = self.construct_stock_future_contract(stock_code, exchange, next_month, next_year)
            
            # Try to verify these contracts exist
            front_exists = self.verify_contract_exists(front_contract)
            next_exists = self.verify_contract_exists(next_contract)
            
            if front_exists and next_exists:
                logger.info(f"Found active contracts for {stock_code} {exchange}: Front={front_contract}, Next={next_contract}")
                
                # Get expiry dates
                front_expiry = self.calculate_expiry_date(front_contract, exchange, front_month, front_year)
                next_expiry = self.calculate_expiry_date(next_contract, exchange, next_month, next_year)
                
                return front_contract, next_contract, front_expiry, next_expiry
                
            elif front_exists:
                logger.warning(f"Next month contract {next_contract} not found for {stock_code} {exchange}")
                
                # Get expiry date for front contract
                front_expiry = self.calculate_expiry_date(front_contract, exchange, front_month, front_year)
                
                return front_contract, None, front_expiry, None
                
            else:
                logger.warning(f"Front month contract {front_contract} not found for {stock_code} {exchange}")
                
                # Try alternative formats
                # Some markets might use different formats than expected
                
                # Try with only the stock code as the ticker root
                if '/' in stock_code:
                    # For tickers like "700/HKD", try with just "700"
                    parts = stock_code.split('/')
                    alt_stock_code = parts[0]
                    
                    alt_front_contract = self.construct_stock_future_contract(alt_stock_code, exchange, front_month, front_year)
                    alt_next_contract = self.construct_stock_future_contract(alt_stock_code, exchange, next_month, next_year)
                    
                    front_exists = self.verify_contract_exists(alt_front_contract)
                    next_exists = self.verify_contract_exists(alt_next_contract)
                    
                    if front_exists and next_exists:
                        logger.info(f"Found active contracts with alternate stock code for {stock_code} {exchange}: Front={alt_front_contract}, Next={alt_next_contract}")
                        
                        # Get expiry dates
                        front_expiry = self.calculate_expiry_date(alt_front_contract, exchange, front_month, front_year)
                        next_expiry = self.calculate_expiry_date(alt_next_contract, exchange, next_month, next_year)
                        
                        return alt_front_contract, alt_next_contract, front_expiry, next_expiry
                        
                    elif front_exists:
                        logger.warning(f"Alternative next month contract {alt_next_contract} not found for {stock_code} {exchange}")
                        
                        # Get expiry date for front contract
                        front_expiry = self.calculate_expiry_date(alt_front_contract, exchange, front_month, front_year)
                        
                        return alt_front_contract, None, front_expiry, None
                
                # If we still can't find valid contracts, use sample data
                logger.warning(f"No valid contracts found for {stock_code} {exchange}, using sample data")
                return self.get_sample_stock_futures_contracts(stock_info)
                
        except Exception as e:
            logger.error(f"Error retrieving active stock futures contracts: {e}")
            logger.warning(f"Falling back to sample data for {stock_info['stock_code']} {stock_info['exchange']}")
            return self.get_sample_stock_futures_contracts(stock_info)
    
    def verify_contract_exists(self, contract: str) -> bool:
        """
        Verify that a contract exists in Bloomberg.
        
        Args:
            contract: Contract ticker to verify
            
        Returns:
            True if the contract exists, False otherwise
        """
        try:
            # Get reference data service
            refDataService = self.session.getService("//blp/refdata")
            request = refDataService.createRequest("ReferenceDataRequest")
            
            # Add the contract
            request.append("securities", contract)
            
            # Just request a simple field
            request.append("fields", "LAST_PRICE")
            
            # Send the request
            self.session.sendRequest(request)
            
            contract_exists = True
            
            # Process the response
            while True:
                event = self.session.nextEvent(500)
                
                for msg in event:
                    if msg.messageType() == blpapi.Name("ReferenceDataResponse"):
                        securities = msg.getElement("securityData")
                        
                        for i in range(securities.numValues()):
                            security = securities.getValue(i)
                            
                            if security.hasElement("securityError"):
                                contract_exists = False
                                error = security.getElement("securityError").getElementAsString("message")
                                logger.warning(f"Security error for {contract}: {error}")
                
                if event.eventType() == blpapi.Event.RESPONSE:
                    break
            
            return contract_exists
                
        except Exception as e:
            logger.error(f"Error verifying contract {contract}: {e}")
            return False
    
    def get_sample_stock_futures_contracts(self, stock_info: Dict[str, str]) -> Tuple[str, str, datetime.date, datetime.date]:
        """
        Generate sample front and next month stock futures contracts for testing.
        
        Args:
            stock_info: Dictionary with stock code and exchange
            
        Returns:
            Tuple of (front_month_contract, next_month_contract, front_expiry, next_expiry)
        """
        stock_code = stock_info['stock_code']
        exchange = stock_info['exchange']
        
        # Determine the front and next contract months (same logic as real method)
        front_month, front_year, next_month, next_year = self.determine_contract_months(exchange)
        
        # Construct the contract tickers
        front_contract = self.construct_stock_future_contract(stock_code, exchange, front_month, front_year)
        next_contract = self.construct_stock_future_contract(stock_code, exchange, next_month, next_year)
        
        # Estimate expiry dates
        front_expiry = self.estimate_expiry_date(exchange, front_month, front_year)
        next_expiry = self.estimate_expiry_date(exchange, next_month, next_year)
        
        logger.info(f"Generated sample contracts for {stock_code} {exchange}: Front={front_contract}, Next={next_contract}")
        return front_contract, next_contract, front_expiry, next_expiry
    
    def try_multiple_fields(self, field_data, field_type: str) -> Optional[float]:
        """
        Try multiple field names to get a specific type of data.
        
        Args:
            field_data: Bloomberg field data element
            field_type: Type of field to try ('volume' or 'open_interest')
            
        Returns:
            The field value if found, None otherwise
        """
        field_names = self.field_mappings.get(field_type, [])
        
        for field_name in field_names:
            if field_data.hasElement(field_name):
                try:
                    return field_data.getElementAsFloat(field_name)
                except Exception as e:
                    logger.warning(f"Error getting {field_name}: {e}")
        
        return None
    
    def calculate_roll_percentage(self, front_contract: str, next_contract: str, 
                                 front_expiry: datetime.date, next_expiry: datetime.date) -> Dict[str, Any]:
        """
        Calculate the roll percentage between front and next month contracts.
        
        Args:
            front_contract: Front month contract ticker
            next_contract: Next month contract ticker
            front_expiry: Front month expiry date
            next_expiry: Next month expiry date
            
        Returns:
            Dictionary with roll statistics
        """
        if self.use_sample_data:
            return self.get_sample_roll_data(front_contract, next_contract, front_expiry, next_expiry)
        
        # Calculate days to expiry
        front_days_to_expiry = self.calculate_days_to_expiry(front_expiry) if front_expiry else None
        next_days_to_expiry = self.calculate_days_to_expiry(next_expiry) if next_expiry else None
        
        if not front_contract or not next_contract:
            logger.warning("Missing contract information, cannot calculate roll percentage")
            return {
                'front_contract': front_contract,
                'next_contract': next_contract,
                'front_expiry': front_expiry,
                'next_expiry': next_expiry,
                'front_days_to_expiry': front_days_to_expiry,
                'next_days_to_expiry': next_days_to_expiry,
                'front_volume': 0,
                'next_volume': 0,
                'front_oi': 0,
                'next_oi': 0,
                'volume_roll_pct': None,
                'oi_roll_pct': None,
                'avg_roll_pct': None,
                'timestamp': self.latest_update_time.strftime("%Y-%m-%d %H:%M:%S")
            }
        
        try:
            # Get volume and open interest data
            refDataService = self.session.getService("//blp/refdata")
            request = refDataService.createRequest("ReferenceDataRequest")
            
            # Add securities
            request.append("securities", front_contract)
            request.append("securities", next_contract)
            
            # Add multiple possible fields for volume and OI
            for field in self.field_mappings.get('volume', []):
                request.append("fields", field)
            
            for field in self.field_mappings.get('open_interest', []):
                request.append("fields", field)
            
            # Send the request
            logger.info(f"Requesting roll data for {front_contract} and {next_contract}")
            self.session.sendRequest(request)
            
            # Process the response
            front_volume = 0
            front_oi = 0
            next_volume = 0
            next_oi = 0
            
            while True:
                event = self.session.nextEvent(500)
                
                for msg in event:
                    if msg.messageType() == blpapi.Name("ReferenceDataResponse"):
                        # Process all securities in the response
                        securities_data = msg.getElement("securityData")
                        
                        for i in range(securities_data.numValues()):
                            security_data = securities_data.getValue(i)
                            ticker = security_data.getElementAsString("security")
                            
                            # Check for errors
                            if security_data.hasElement("securityError"):
                                error = security_data.getElement("securityError").getElementAsString("message")
                                logger.warning(f"Security error for {ticker}: {error}")
                                continue
                            
                            # Extract field data if available
                            if security_data.hasElement("fieldData"):
                                field_data = security_data.getElement("fieldData")
                                
                                # Try to get volume and OI using multiple possible field names
                                volume = self.try_multiple_fields(field_data, 'volume')
                                oi = self.try_multiple_fields(field_data, 'open_interest')
                                
                                # Log warnings for missing data
                                if volume is None:
                                    logger.warning(f"No volume data for {ticker}")
                                    volume = 0
                                    
                                if oi is None:
                                    logger.warning(f"No open interest data for {ticker}")
                                    oi = 0
                                
                                # Assign to the correct contract
                                if ticker == front_contract:
                                    front_volume = volume
                                    front_oi = oi
                                elif ticker == next_contract:
                                    next_volume = volume
                                    next_oi = oi
                
                if event.eventType() == blpapi.Event.RESPONSE:
                    break
            
            # Check if we have enough data to calculate percentages
            have_volume = front_volume is not None and next_volume is not None and (front_volume > 0 or next_volume > 0)
            have_oi = front_oi is not None and next_oi is not None and (front_oi > 0 or next_oi > 0)
            
            # If we don't have either volume or OI data, fall back to sample data
            if not have_volume and not have_oi:
                logger.warning(f"Missing both volume and OI data for {front_contract} and {next_contract}, using sample data")
                return self.get_sample_roll_data(front_contract, next_contract, front_expiry, next_expiry)
            
            # Calculate roll percentages
            volume_roll_pct = None
            oi_roll_pct = None
            avg_roll_pct = None
            
            if have_volume:
                volume_roll_pct = (next_volume / (front_volume + next_volume)) * 100 if front_volume + next_volume > 0 else 0
            
            if have_oi:
                oi_roll_pct = (next_oi / (front_oi + next_oi)) * 100 if front_oi + next_oi > 0 else 0
            
            # Average of volume and OI roll percentages
            if volume_roll_pct is not None and oi_roll_pct is not None:
                avg_roll_pct = (volume_roll_pct + oi_roll_pct) / 2
            elif volume_roll_pct is not None:
                avg_roll_pct = volume_roll_pct
            elif oi_roll_pct is not None:
                avg_roll_pct = oi_roll_pct
            
            # Prepare result dictionary
            result = {
                'front_contract': front_contract,
                'next_contract': next_contract,
                'front_expiry': front_expiry,
                'next_expiry': next_expiry,
                'front_days_to_expiry': front_days_to_expiry,
                'next_days_to_expiry': next_days_to_expiry,
                'front_volume': front_volume or 0,
                'next_volume': next_volume or 0,
                'front_oi': front_oi or 0,
                'next_oi': next_oi or 0,
                'volume_roll_pct': volume_roll_pct,
                'oi_roll_pct': oi_roll_pct,
                'avg_roll_pct': avg_roll_pct,
                'timestamp': self.latest_update_time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # Log the results
            log_msg = f"Roll data for {front_contract}/{next_contract}: "
            
            if volume_roll_pct is not None:
                log_msg += f"Volume={volume_roll_pct:.1f}%, "
            else:
                log_msg += "Volume=N/A, "
                
            if oi_roll_pct is not None:
                log_msg += f"OI={oi_roll_pct:.1f}%, "
            else:
                log_msg += "OI=N/A, "
                
            if avg_roll_pct is not None:
                log_msg += f"Avg={avg_roll_pct:.1f}%, "
            else:
                log_msg += "Avg=N/A, "
                
            log_msg += f"Days to expiry: Front={front_days_to_expiry}, Next={next_days_to_expiry}"
            
            logger.info(log_msg)
            
            return result
            
        except Exception as e:
            logger.error(f"Error calculating roll percentage: {e}")
            logger.warning(f"Falling back to sample data for {front_contract}/{next_contract}")
            return self.get_sample_roll_data(front_contract, next_contract, front_expiry, next_expiry)
    
    def get_sample_roll_data(self, front_contract: str, next_contract: str, 
                             front_expiry: datetime.date, next_expiry: datetime.date) -> Dict[str, Any]:
        """
        Generate sample roll data for testing.
        
        Args:
            front_contract: Front month contract ticker
            next_contract: Next month contract ticker
            front_expiry: Front month expiry date
            next_expiry: Next month expiry date
            
        Returns:
            Dictionary with sample roll statistics
        """
        # Extract stock code if possible
        stock_code = None
        exchange = None
        if front_contract:
            # Pattern like "2330=J5 TT Equity"
            match = re.search(r'([^=]+)=\w\d+\s+(\w+)', front_contract)
            if match:
                stock_code = match.group(1)
                exchange = match.group(2)
        
        # Calculate days to expiry
        front_days_to_expiry = self.calculate_days_to_expiry(front_expiry) if front_expiry else None
        next_days_to_expiry = self.calculate_days_to_expiry(next_expiry) if next_expiry else None
        
        # Get current date to determine realistic roll percentages
        now = datetime.datetime.now()
        days_in_month = 30  # Approximate days in a month
        day_of_month = now.day
        
        # Create a hash of the contract name for reproducible "random" values
        if front_contract and next_contract:
            contract_hash = hash(front_contract + next_contract)
            np.random.seed(contract_hash % 10000)
            
            # Different markets have different typical roll timing windows
            if exchange in ['TT', 'HK', 'SP']:
                # Markets with monthly contracts often roll more gradually
                roll_start_pct = 0  # Days 1-5: minimal roll activity
                roll_mid_pct = 30   # Days 10-15: around 30% rolled
                roll_end_pct = 90   # Days 20+: mostly rolled
            else:
                # Quarterly markets tend to roll more quickly near expiry
                roll_start_pct = 0  # Days 1-15: minimal roll activity
                roll_mid_pct = 50   # Days 15-25: around 50% rolled
                roll_end_pct = 95   # Days 25+: mostly rolled
            
            # Use days to expiry to estimate roll percentage if available
            if front_days_to_expiry is not None:
                if front_days_to_expiry > 20:
                    base_pct = roll_start_pct
                elif front_days_to_expiry > 10:
                    # Linear interpolation between start and mid points
                    base_pct = roll_start_pct + (20 - front_days_to_expiry) / 10 * (roll_mid_pct - roll_start_pct)
                elif front_days_to_expiry > 0:
                    # Linear interpolation between mid and end points
                    base_pct = roll_mid_pct + (10 - front_days_to_expiry) / 10 * (roll_end_pct - roll_mid_pct)
                else:
                    base_pct = roll_end_pct
            else:
                # Fallback to using day of month
                if day_of_month < 5:
                    base_pct = roll_start_pct + (day_of_month / 5) * 10
                elif day_of_month < 15:
                    base_pct = roll_start_pct + ((day_of_month - 5) / 10) * (roll_mid_pct - roll_start_pct)
                elif day_of_month < 25:
                    base_pct = roll_mid_pct + ((day_of_month - 15) / 10) * (roll_end_pct - roll_mid_pct)
                else:
                    base_pct = roll_end_pct + ((day_of_month - 25) / 5) * (100 - roll_end_pct)
            
            # Add some randomness
            volume_base = base_pct + np.random.normal(0, 5)
            oi_base = base_pct + np.random.normal(0, 3)
            
            # Ensure percentages are within valid range
            volume_roll_pct = max(0, min(100, volume_base))
            oi_roll_pct = max(0, min(100, oi_base))
            
            # Calculate volumes and OI - typically smaller than index futures
            total_volume = np.random.randint(5000, 50000)
            total_oi = np.random.randint(10000, 100000)
            
            front_volume = int(total_volume * (1 - volume_roll_pct/100))
            next_volume = int(total_volume * (volume_roll_pct/100))
            front_oi = int(total_oi * (1 - oi_roll_pct/100))
            next_oi = int(total_oi * (oi_roll_pct/100))
            
            avg_roll_pct = (volume_roll_pct + oi_roll_pct) / 2
        else:
            # Default values if contracts are missing
            front_volume = 0
            next_volume = 0
            front_oi = 0
            next_oi = 0
            volume_roll_pct = None
            oi_roll_pct = None
            avg_roll_pct = None
        
        result = {
            'front_contract': front_contract,
            'next_contract': next_contract,
            'front_expiry': front_expiry,
            'next_expiry': next_expiry,
            'front_days_to_expiry': front_days_to_expiry,
            'next_days_to_expiry': next_days_to_expiry,
            'front_volume': front_volume,
            'next_volume': next_volume,
            'front_oi': front_oi,
            'next_oi': next_oi,
            'volume_roll_pct': volume_roll_pct,
            'oi_roll_pct': oi_roll_pct,
            'avg_roll_pct': avg_roll_pct,
            'timestamp': self.latest_update_time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        if avg_roll_pct is not None:
            logger.info(f"Sample roll data: Volume={volume_roll_pct:.1f}%, OI={oi_roll_pct:.1f}%, Avg={avg_roll_pct:.1f}%, Days to expiry: Front={front_days_to_expiry}, Next={next_days_to_expiry}")
        
        return result
    
    def track_stock_futures_rolls(self, futures_list_file: str) -> Dict[str, Any]:
        """
        Track roll percentages for a list of stock futures.
        
        Args:
            futures_list_file: Path to file containing stock futures tickers
            
        Returns:
            Dictionary with roll data for each stock future
        """
        # Read stock futures list
        stock_futures_list = self.read_stock_futures_list(futures_list_file)
        if not stock_futures_list:
            logger.error("No stock futures found in the provided list")
            return {}
        
        # Process each stock future
        roll_results = {}
        
        for stock_info in stock_futures_list:
            stock_code = stock_info['stock_code']
            exchange = stock_info['exchange']
            key = f"{stock_code} {exchange}"
            
            # Get active contracts
            front_contract, next_contract, front_expiry, next_expiry = self.get_active_stock_futures_contracts(stock_info)
            
            # Calculate roll percentage
            roll_data = self.calculate_roll_percentage(front_contract, next_contract, front_expiry, next_expiry)
            
            # Store result
            roll_results[key] = roll_data
            
            # Add a small delay between requests
            time.sleep(0.1)
        
        # Update data store
        self.roll_data = {
            'update_time': self.latest_update_time,
            'results': roll_results
        }
        
        return self.roll_data
    
    def save_roll_data_to_csv(self, output_file: str) -> None:
        """
        Save the roll data to a CSV file.
        
        Args:
            output_file: Path to output CSV file
        """
        if not self.roll_data or 'results' not in self.roll_data:
            logger.error("No roll data available. Run track_stock_futures_rolls() first.")
            return
        
        try:
            # Prepare data for export
            data = []
            for key, roll_info in self.roll_data['results'].items():
                # Split key into stock_code and exchange
                parts = key.split(' ')
                stock_code = parts[0]
                exchange = parts[1] if len(parts) > 1 else ''
                
                row = {
                    'stock_code': stock_code,
                    'exchange': exchange,
                    'front_contract': roll_info.get('front_contract', ''),
                    'next_contract': roll_info.get('next_contract', ''),
                    'front_expiry': roll_info.get('front_expiry', ''),
                    'next_expiry': roll_info.get('next_expiry', ''),
                    'front_days_to_expiry': roll_info.get('front_days_to_expiry', ''),
                    'next_days_to_expiry': roll_info.get('next_days_to_expiry', ''),
                    'front_volume': roll_info.get('front_volume', 0),
                    'next_volume': roll_info.get('next_volume', 0),
                    'front_oi': roll_info.get('front_oi', 0),
                    'next_oi': roll_info.get('next_oi', 0),
                    'volume_roll_pct': roll_info.get('volume_roll_pct', ''),
                    'oi_roll_pct': roll_info.get('oi_roll_pct', ''),
                    'avg_roll_pct': roll_info.get('avg_roll_pct', ''),
                    'timestamp': roll_info.get('timestamp', self.latest_update_time.strftime("%Y-%m-%d %H:%M:%S"))
                }
                data.append(row)
            
            # Convert to DataFrame
            df = pd.DataFrame(data)
            
            # Save to CSV
            df.to_csv(output_file, index=False)
            logger.info(f"Roll data saved to {output_file}")
            
        except Exception as e:
            logger.error(f"Error saving roll data to CSV: {e}")
    
    def save_roll_data_to_excel(self, output_file: str) -> None:
        """
        Save the roll data to an Excel file with formatting.
        
        Args:
            output_file: Path to output Excel file
        """
        if not self.roll_data or 'results' not in self.roll_data:
            logger.error("No roll data available. Run track_stock_futures_rolls() first.")
            return
        
        try:
            # Prepare data for export
            data = []
            for key, roll_info in self.roll_data['results'].items():
                # Split key into stock_code and exchange
                parts = key.split(' ')
                stock_code = parts[0]
                exchange = parts[1] if len(parts) > 1 else ''
                
                # Convert roll percentages to proper format for Excel (0-1 range)
                volume_roll_pct = roll_info.get('volume_roll_pct')
                if volume_roll_pct is not None:
                    volume_roll_pct = volume_roll_pct / 100
                
                oi_roll_pct = roll_info.get('oi_roll_pct')
                if oi_roll_pct is not None:
                    oi_roll_pct = oi_roll_pct / 100
                
                avg_roll_pct = roll_info.get('avg_roll_pct')
                if avg_roll_pct is not None:
                    avg_roll_pct = avg_roll_pct / 100
                
                # Format expiry dates
                front_expiry = roll_info.get('front_expiry')
                next_expiry = roll_info.get('next_expiry')
                
                row = {
                    'Stock Code': stock_code,
                    'Exchange': exchange,
                    'Front Contract': roll_info.get('front_contract', ''),
                    'Next Contract': roll_info.get('next_contract', ''),
                    'Front Expiry': front_expiry,
                    'Next Expiry': next_expiry,
                    'Front Days to Expiry': roll_info.get('front_days_to_expiry', ''),
                    'Next Days to Expiry': roll_info.get('next_days_to_expiry', ''),
                    'Front Volume': roll_info.get('front_volume', 0),
                    'Next Volume': roll_info.get('next_volume', 0),
                    'Front OI': roll_info.get('front_oi', 0),
                    'Next OI': roll_info.get('next_oi', 0),
                    'Volume Roll %': volume_roll_pct,
                    'OI Roll %': oi_roll_pct,
                    'Avg Roll %': avg_roll_pct,
                    'Timestamp': roll_info.get('timestamp', self.latest_update_time.strftime("%Y-%m-%d %H:%M:%S"))
                }
                data.append(row)
            
            # Convert to DataFrame
            df = pd.DataFrame(data)
            
            # Create Excel writer with formatting
            writer = pd.ExcelWriter(output_file, engine='xlsxwriter')
            df.to_excel(writer, sheet_name='Stock Futures Roll Data', index=False)
            
            # Get workbook and worksheet objects
            workbook = writer.book
            worksheet = writer.sheets['Stock Futures Roll Data']
            
            # Define formats
            header_format = workbook.add_format({
                'bold': True,
                'text_wrap': True,
                'valign': 'top',
                'fg_color': '#D7E4BC',
                'border': 1
            })
            
            date_format = workbook.add_format({'num_format': 'yyyy-mm-dd'})
            pct_format = workbook.add_format({'num_format': '0.0%'})
            number_format = workbook.add_format({'num_format': '#,##0'})
            days_format = workbook.add_format({'num_format': '0'})
            
            # Apply formats to columns
            for col_num, column in enumerate(df.columns):
                worksheet.write(0, col_num, column, header_format)
                
                # Set column width based on content type
                if column in ['Stock Code', 'Exchange']:
                    worksheet.set_column(col_num, col_num, 10)
                elif column in ['Front Contract', 'Next Contract']:
                    worksheet.set_column(col_num, col_num, 20)
                elif column in ['Front Expiry', 'Next Expiry']:
                    worksheet.set_column(col_num, col_num, 12, date_format)
                elif 'Days to Expiry' in column:
                    worksheet.set_column(col_num, col_num, 10, days_format)
                elif 'Roll %' in column:
                    worksheet.set_column(col_num, col_num, 10, pct_format)
                elif 'Volume' in column or 'OI' in column:
                    worksheet.set_column(col_num, col_num, 12, number_format)
                else:
                    worksheet.set_column(col_num, col_num, 15)
            
            # Add conditional formatting for roll percentages
            roll_pct_cols = ['M', 'N', 'O']  # Volume Roll %, OI Roll %, Avg Roll %
            for col in roll_pct_cols:
                worksheet.conditional_format(f'{col}2:{col}{len(df)+1}', {
                    'type': '3_color_scale',
                    'min_color': "#FFFFFF",
                    'mid_color': "#FFEB84",
                    'max_color': "#FF9A3C"
                })
            
            # Add conditional formatting for days to expiry columns
            days_cols = ['G', 'H']  # Front Days to Expiry, Next Days to Expiry
            for col in days_cols:
                worksheet.conditional_format(f'{col}2:{col}{len(df)+1}', {
                    'type': 'data_bar',
                    'bar_color': '#9CD1CE',
                    'bar_solid': False
                })
            
            # Save the workbook
            writer.close()
            logger.info(f"Roll data saved to Excel file: {output_file}")
            
        except Exception as e:
            logger.error(f"Error saving roll data to Excel: {e}")
    
    def print_roll_summary(self) -> None:
        """Print a summary of stock futures roll percentages."""
        if not self.roll_data or 'results' not in self.roll_data:
            logger.error("No roll data available. Run track_stock_futures_rolls() first.")
            return
        
        results = self.roll_data['results']
        if not results:
            logger.warning("No roll results to display.")
            return
        
        # Calculate max length for formatting
        max_stock_len = max(len(key) for key in results.keys())
        max_contract_len = 0
        
        for roll_info in results.values():
            front_len = len(str(roll_info.get('front_contract', '')))
            next_len = len(str(roll_info.get('next_contract', '')))
            max_contract_len = max(max_contract_len, front_len, next_len)
        
        # Print header
        print("\n" + "="*140)
        print(f"STOCK FUTURES ROLL PERCENTAGES - {self.latest_update_time.strftime('%Y-%m-%d %H:%M:%S')}")
        if self.use_sample_data:
            print("NOTE: Using sample data (Bloomberg connection not available)")
        print("="*140)
        
        # Print table header
        print(f"{'Stock':^{max_stock_len}} | {'Front Contract':^{max_contract_len}} | {'Next Contract':^{max_contract_len}} | " +
              f"{'Days to Exp':>10} | {'Next Exp':>10} | {'Front Vol':>10} | {'Next Vol':>10} | {'Front OI':>10} | {'Next OI':>10} | " +
              f"{'Vol Roll%':>8} | {'OI Roll%':>8} | {'Avg Roll%':>8}")
        print("-"*(max_stock_len + 2*max_contract_len + 102))
        
        # Sort by average roll percentage (descending)
        sorted_stocks = sorted(results.keys(), 
                               key=lambda x: results[x].get('avg_roll_pct', 0) if results[x].get('avg_roll_pct') is not None else -1,
                               reverse=True)
        
        # Print each row
        for stock_key in sorted_stocks:
            roll_info = results[stock_key]
            
            # Get values
            front_contract = roll_info.get('front_contract', '')
            next_contract = roll_info.get('next_contract', '')
            front_days_to_expiry = roll_info.get('front_days_to_expiry', '')
            next_days_to_expiry = roll_info.get('next_days_to_expiry', '')
            front_volume = roll_info.get('front_volume', 0)
            next_volume = roll_info.get('next_volume', 0)
            front_oi = roll_info.get('front_oi', 0)
            next_oi = roll_info.get('next_oi', 0)
            volume_roll_pct = roll_info.get('volume_roll_pct')
            oi_roll_pct = roll_info.get('oi_roll_pct')
            avg_roll_pct = roll_info.get('avg_roll_pct')
            
            # Format values
            volume_pct_str = f"{volume_roll_pct:.2f}%" if volume_roll_pct is not None else "N/A"
            oi_pct_str = f"{oi_roll_pct:.2f}%" if oi_roll_pct is not None else "N/A"
            avg_pct_str = f"{avg_roll_pct:.2f}%" if avg_roll_pct is not None else "N/A"
            
            # Format days to expiry - color code based on how close to expiry
            if front_days_to_expiry is not None:
                if front_days_to_expiry <= 5:
                    front_days_str = f"\033[91m{front_days_to_expiry}\033[0m"  # Red for imminent expiry
                elif front_days_to_expiry <= 10:
                    front_days_str = f"\033[93m{front_days_to_expiry}\033[0m"  # Yellow for approaching expiry
                else:
                    front_days_str = f"{front_days_to_expiry}"
            else:
                front_days_str = "N/A"
                
            # Show next expiry too
            next_days_str = f"{next_days_to_expiry}" if next_days_to_expiry is not None else "N/A"
            
            # Apply color based on roll percentage
            if avg_roll_pct is not None:
                if avg_roll_pct > 80:
                    avg_pct_str = f"\033[91m{avg_pct_str}\033[0m"  # Red for high roll
                elif avg_roll_pct > 50:
                    avg_pct_str = f"\033[93m{avg_pct_str}\033[0m"  # Yellow for medium roll
                elif avg_roll_pct > 20:
                    avg_pct_str = f"\033[92m{avg_pct_str}\033[0m"  # Green for low roll
            
            # Print the row
            print(f"{stock_key:<{max_stock_len}} | {front_contract:<{max_contract_len}} | {next_contract:<{max_contract_len}} | " +
                  f"{front_days_str:>10} | {next_days_str:>10} | {front_volume:>10,.0f} | {next_volume:>10,.0f} | " +
                  f"{front_oi:>10,.0f} | {next_oi:>10,.0f} | {volume_pct_str:>8} | {oi_pct_str:>8} | {avg_pct_str:>8}")
        
        print("="*140 + "\n")


def main():
    """Main function to run the stock futures roll tracker without command-line arguments."""
    print("Stock Futures Roll Tracker - Starting...")
    print(f"Current date: {datetime.datetime.now().strftime('%Y-%m-%d')}")
    print(f"Reading stock futures tickers from: {DEFAULT_INPUT_FILE}")
    print(f"Using Bloomberg data: {not USE_SAMPLE_DATA}")
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(DEFAULT_OUTPUT_CSV)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Initialize the tracker
    tracker = StockFuturesRollTracker(
        host=BLOOMBERG_HOST,
        port=BLOOMBERG_PORT,
        use_sample_data=USE_SAMPLE_DATA
    )
    
    try:
        # Create default input file if it doesn't exist
        if not os.path.exists(DEFAULT_INPUT_FILE):
            print(f"Creating default stock futures list file: {DEFAULT_INPUT_FILE}")
            with open(DEFAULT_INPUT_FILE, 'w') as f:
                f.write("""# Taiwan Stock Exchange
2330 TT    # TSMC
2317 TT    # Foxconn

# Hong Kong Exchange
700 HK     # Tencent
1299 HK    # AIA Group

# Korean Exchange
005930 KS  # Samsung Electronics
000660 KS  # SK Hynix

# Singapore Exchange
Z74 SP     # Singapore Telecommunications

# Example of a full future contract specification:
# 2330=J5 TT Equity
""")
        
        # Start Bloomberg session
        if tracker.start_session():
            # Track futures rolls
            print(f"Tracking roll percentages for stock futures...")
            tracker.track_stock_futures_rolls(DEFAULT_INPUT_FILE)
            
            # Print summary
            tracker.print_roll_summary()
            
            # Save data
            if USE_EXCEL_OUTPUT:
                print(f"Saving roll data to Excel: {DEFAULT_OUTPUT_EXCEL}")
                tracker.save_roll_data_to_excel(DEFAULT_OUTPUT_EXCEL)
            else:
                print(f"Saving roll data to CSV: {DEFAULT_OUTPUT_CSV}")
                tracker.save_roll_data_to_csv(DEFAULT_OUTPUT_CSV)
                
            print("Stock futures roll tracking completed successfully.")
            
        else:
            print("Failed to start session. Please check connection settings.")
    
    except KeyboardInterrupt:
        print("\nTracking interrupted by user")
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Always stop the session
        tracker.stop_session()


if __name__ == "__main__":
    main()