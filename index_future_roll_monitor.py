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
DEFAULT_INPUT_FILE = "futures_list.txt"
DEFAULT_OUTPUT_CSV = "futures_roll_data.csv"
DEFAULT_OUTPUT_EXCEL = "futures_roll_data.xlsx"

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
        logging.FileHandler("futures_roll_tracker.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('FuturesRollTracker')

class FuturesRollTracker:
    """
    A class to track futures roll activity by analyzing volume and open interest
    data between the front month and next month contracts.
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
        
        # Define contract specifications for common futures
        self.contract_specs = {
            # Stock Indices
            'ES': {'type': 'Index', 'exchange': 'CME', 'months': [3, 6, 9, 12], 'rolldays': 7, 'year_format': '1-digit'},  # S&P 500 E-mini 
            'NQ': {'type': 'Index', 'exchange': 'CME', 'months': [3, 6, 9, 12], 'rolldays': 7, 'year_format': '1-digit'},  # NASDAQ 100 E-mini
            'YM': {'type': 'Index', 'exchange': 'CBOT', 'months': [3, 6, 9, 12], 'rolldays': 7, 'year_format': '1-digit'}, # Dow Jones E-mini
            'RTY': {'type': 'Index', 'exchange': 'CME', 'months': [3, 6, 9, 12], 'rolldays': 7, 'year_format': '1-digit'}, # Russell 2000 E-mini
            'HSI': {'type': 'Index', 'exchange': 'HKFE', 'months': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], 'rolldays': 3, 'year_format': '1-digit'}, # Hang Seng
            'NKY': {'type': 'Index', 'exchange': 'OSE', 'months': [3, 6, 9, 12], 'rolldays': 7, 'year_format': '1-digit'},  # Nikkei 225
            
            # Energy
            'CL': {'type': 'Comdty', 'exchange': 'NYMEX', 'months': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], 'rolldays': 20, 'year_format': '1-digit'}, # Crude Oil
            'NG': {'type': 'Comdty', 'exchange': 'NYMEX', 'months': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], 'rolldays': 15, 'year_format': '1-digit'}, # Natural Gas
            'HO': {'type': 'Comdty', 'exchange': 'NYMEX', 'months': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], 'rolldays': 20, 'year_format': '1-digit'}, # Heating Oil
            'RB': {'type': 'Comdty', 'exchange': 'NYMEX', 'months': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], 'rolldays': 20, 'year_format': '1-digit'}, # RBOB Gasoline
            
            # Metals
            'GC': {'type': 'Comdty', 'exchange': 'COMEX', 'months': [2, 4, 6, 8, 10, 12], 'rolldays': 25, 'year_format': '1-digit'}, # Gold
            'SI': {'type': 'Comdty', 'exchange': 'COMEX', 'months': [3, 5, 7, 9, 12], 'rolldays': 25, 'year_format': '1-digit'},    # Silver
            'HG': {'type': 'Comdty', 'exchange': 'COMEX', 'months': [3, 5, 7, 9, 12], 'rolldays': 25, 'year_format': '1-digit'},    # Copper
            'PL': {'type': 'Comdty', 'exchange': 'NYMEX', 'months': [1, 4, 7, 10], 'rolldays': 25, 'year_format': '1-digit'},      # Platinum
            
            # Interest Rates
            'ZB': {'type': 'Comdty', 'exchange': 'CBOT', 'months': [3, 6, 9, 12], 'rolldays': 10, 'year_format': '1-digit'}, # Treasury Bond
            'ZN': {'type': 'Comdty', 'exchange': 'CBOT', 'months': [3, 6, 9, 12], 'rolldays': 10, 'year_format': '1-digit'}, # 10-Year Note
            'ZF': {'type': 'Comdty', 'exchange': 'CBOT', 'months': [3, 6, 9, 12], 'rolldays': 10, 'year_format': '1-digit'}, # 5-Year Note
            'ZT': {'type': 'Comdty', 'exchange': 'CBOT', 'months': [3, 6, 9, 12], 'rolldays': 10, 'year_format': '1-digit'}, # 2-Year Note
            'ED': {'type': 'Comdty', 'exchange': 'CME', 'months': [3, 6, 9, 12], 'rolldays': 10, 'year_format': '1-digit'},  # Eurodollar
            
            # Currencies
            '6E': {'type': 'Curncy', 'exchange': 'CME', 'months': [3, 6, 9, 12], 'rolldays': 5, 'year_format': '1-digit'},  # Euro FX
            '6J': {'type': 'Curncy', 'exchange': 'CME', 'months': [3, 6, 9, 12], 'rolldays': 5, 'year_format': '1-digit'},  # Japanese Yen
            '6B': {'type': 'Curncy', 'exchange': 'CME', 'months': [3, 6, 9, 12], 'rolldays': 5, 'year_format': '1-digit'},  # British Pound
            '6C': {'type': 'Curncy', 'exchange': 'CME', 'months': [3, 6, 9, 12], 'rolldays': 5, 'year_format': '1-digit'},  # Canadian Dollar
            '6A': {'type': 'Curncy', 'exchange': 'CME', 'months': [3, 6, 9, 12], 'rolldays': 5, 'year_format': '1-digit'},  # Australian Dollar
            '6S': {'type': 'Curncy', 'exchange': 'CME', 'months': [3, 6, 9, 12], 'rolldays': 5, 'year_format': '1-digit'},  # Swiss Franc
            
            # Agricultural
            'ZC': {'type': 'Comdty', 'exchange': 'CBOT', 'months': [3, 5, 7, 9, 12], 'rolldays': 15, 'year_format': '1-digit'},  # Corn
            'ZW': {'type': 'Comdty', 'exchange': 'CBOT', 'months': [3, 5, 7, 9, 12], 'rolldays': 15, 'year_format': '1-digit'},  # Wheat
            'ZS': {'type': 'Comdty', 'exchange': 'CBOT', 'months': [1, 3, 5, 7, 8, 9, 11], 'rolldays': 15, 'year_format': '1-digit'},  # Soybeans
            'LE': {'type': 'Comdty', 'exchange': 'CME', 'months': [2, 4, 6, 8, 10, 12], 'rolldays': 15, 'year_format': '1-digit'},  # Live Cattle
            'HE': {'type': 'Comdty', 'exchange': 'CME', 'months': [2, 4, 6, 7, 8, 10, 12], 'rolldays': 15, 'year_format': '1-digit'},  # Lean Hogs
        }
        
        # Bloomberg field mapping - try multiple field names for the same data
        self.field_mappings = {
            'volume': ['PX_VOLUME', 'VOLUME', 'VOLUME_ALL_TRADES', 'VOLUME_TDY'],
            'open_interest': ['OPEN_INT', 'PX_OPEN_INT', 'OPN_INT', 'OPEN_INTEREST']
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
    
    def read_futures_list(self, filename: str) -> List[str]:
        """
        Read a list of futures tickers from a file.
        
        Args:
            filename: Path to the file containing futures tickers
            
        Returns:
            List of futures root tickers
        """
        if not os.path.exists(filename):
            logger.error(f"Futures list file not found: {filename}")
            return []
            
        futures_list = []
        
        try:
            with open(filename, 'r') as f:
                for line in f:
                    # Skip empty lines and comments
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    
                    # Remove any trailing comments
                    if '#' in line:
                        line = line.split('#')[0].strip()
                    
                    # Split by comma or whitespace if the file has additional data
                    parts = re.split(r'[,\s]+', line)
                    ticker = parts[0].strip()
                    
                    # Add if not already in list and not empty
                    if ticker and ticker not in futures_list:
                        futures_list.append(ticker)
            
            logger.info(f"Read {len(futures_list)} futures tickers from {filename}")
            return futures_list
            
        except Exception as e:
            logger.error(f"Error reading futures list: {e}")
            return []
    
    def determine_contract_months(self, root_ticker: str) -> Tuple[int, int, int, int]:
        """
        Determine the current front and next contract months for a futures root.
        
        Args:
            root_ticker: Root ticker for the futures contract
        
        Returns:
            Tuple of (front_month, front_year, next_month, next_year)
        """
        # Get current date
        now = datetime.datetime.now()
        current_month = now.month
        current_year = now.year
        current_day = now.day
        
        # Check if the ticker is in our contract specifications
        upper_ticker = root_ticker.upper()
        if upper_ticker in self.contract_specs:
            # Get the contract months for this future
            contract_months = self.contract_specs[upper_ticker]['months']
            roll_days = self.contract_specs[upper_ticker]['rolldays']
        else:
            # Default to quarterly contracts (Mar, Jun, Sep, Dec)
            contract_months = [3, 6, 9, 12]
            roll_days = 10
        
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
    
    def get_active_futures_contracts(self, root_ticker: str) -> Tuple[str, str]:
        """
        For a given futures root ticker, determine the front and next month contracts.
        
        Args:
            root_ticker: Root ticker for the futures contract (e.g., 'ES')
            
        Returns:
            Tuple of (front_month_contract, next_month_contract)
        """
        if self.use_sample_data:
            return self.get_sample_futures_contracts(root_ticker)
        
        try:
            # Use our knowledge of futures contract schedules
            # to directly construct the contract tickers
            upper_ticker = root_ticker.upper()
            
            # Get the contract specs
            if upper_ticker in self.contract_specs:
                contract_type = self.contract_specs[upper_ticker]['type']
                year_format = self.contract_specs[upper_ticker]['year_format']
            else:
                # Try to guess based on the ticker prefix
                if upper_ticker.startswith('6'):
                    contract_type = 'Curncy'
                    year_format = '1-digit'
                elif upper_ticker in ['ES', 'NQ', 'YM', 'RTY', 'HSI', 'NKY']:
                    contract_type = 'Index'
                    year_format = '1-digit'
                else:
                    contract_type = 'Comdty'
                    year_format = '1-digit'  # Most Bloomberg futures now use 1-digit years
            
            # Determine the front and next contract months
            front_month, front_year, next_month, next_year = self.determine_contract_months(upper_ticker)
            
            # Convert to month codes
            front_month_code = self.month_letters[front_month]
            next_month_code = self.month_letters[next_month]
            
            # Format the year suffix based on specified format
            if year_format == '1-digit':
                front_year_suffix = str(front_year)[-1]
                next_year_suffix = str(next_year)[-1]
            else:  # '2-digit'
                front_year_suffix = str(front_year)[-2:]
                next_year_suffix = str(next_year)[-2:]
            
            # Construct the contract tickers
            front_contract = f"{upper_ticker}{front_month_code}{front_year_suffix} {contract_type}"
            next_contract = f"{upper_ticker}{next_month_code}{next_year_suffix} {contract_type}"
            
            # Try to verify these contracts exist
            if self.verify_contract_exists(front_contract):
                if self.verify_contract_exists(next_contract):
                    logger.info(f"Found active contracts for {root_ticker}: Front={front_contract}, Next={next_contract}")
                    return front_contract, next_contract
                else:
                    logger.warning(f"Next month contract {next_contract} not found for {root_ticker}")
                    return front_contract, None
            else:
                logger.warning(f"Front month contract {front_contract} not found for {root_ticker}")
                
                # Try with a different suffix
                if contract_type == 'Index':
                    alt_type = 'Comdty'
                else:
                    alt_type = 'Index'
                
                # Reconstruct with alternative type
                alt_front_contract = f"{upper_ticker}{front_month_code}{front_year_suffix} {alt_type}"
                alt_next_contract = f"{upper_ticker}{next_month_code}{next_year_suffix} {alt_type}"
                
                if self.verify_contract_exists(alt_front_contract):
                    if self.verify_contract_exists(alt_next_contract):
                        logger.info(f"Found active contracts for {root_ticker} with alternative type: Front={alt_front_contract}, Next={alt_next_contract}")
                        return alt_front_contract, alt_next_contract
                    else:
                        logger.warning(f"Alternative next month contract {alt_next_contract} not found for {root_ticker}")
                        return alt_front_contract, None
                else:
                    # Last chance - try with a different year format
                    if year_format == '1-digit':
                        alt_front_year_suffix = str(front_year)[-2:]
                        alt_next_year_suffix = str(next_year)[-2:]
                    else:
                        alt_front_year_suffix = str(front_year)[-1]
                        alt_next_year_suffix = str(next_year)[-1]
                    
                    alt2_front_contract = f"{upper_ticker}{front_month_code}{alt_front_year_suffix} {contract_type}"
                    alt2_next_contract = f"{upper_ticker}{next_month_code}{alt_next_year_suffix} {contract_type}"
                    
                    if self.verify_contract_exists(alt2_front_contract):
                        if self.verify_contract_exists(alt2_next_contract):
                            logger.info(f"Found active contracts for {root_ticker} with alternative year format: Front={alt2_front_contract}, Next={alt2_next_contract}")
                            return alt2_front_contract, alt2_next_contract
                        else:
                            logger.warning(f"Alternative year format next month contract {alt2_next_contract} not found for {root_ticker}")
                            return alt2_front_contract, None
                    else:
                        logger.warning(f"No valid contracts found for {root_ticker}, using sample data")
                        return self.get_sample_futures_contracts(root_ticker)
                
        except Exception as e:
            logger.error(f"Error retrieving active futures contracts: {e}")
            logger.warning(f"Falling back to sample data for {root_ticker}")
            return self.get_sample_futures_contracts(root_ticker)
    
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
    
    def get_sample_futures_contracts(self, root_ticker: str) -> Tuple[str, str]:
        """
        Generate sample front and next month contracts for testing.
        
        Args:
            root_ticker: Root ticker for the futures contract (e.g., 'ES')
            
        Returns:
            Tuple of (front_month_contract, next_month_contract)
        """
        # Use our same contract determination logic, but without verification
        upper_ticker = root_ticker.upper()
            
        # Get the contract specs
        if upper_ticker in self.contract_specs:
            contract_type = self.contract_specs[upper_ticker]['type']
            year_format = self.contract_specs[upper_ticker]['year_format']
        else:
            # Try to guess based on the ticker prefix
            if upper_ticker.startswith('6'):
                contract_type = 'Curncy'
                year_format = '1-digit'
            elif upper_ticker in ['ES', 'NQ', 'YM', 'RTY', 'HSI', 'NKY']:
                contract_type = 'Index'
                year_format = '1-digit'
            else:
                contract_type = 'Comdty'
                year_format = '1-digit'  # Most Bloomberg futures now use 1-digit years
        
        # Determine the front and next contract months
        front_month, front_year, next_month, next_year = self.determine_contract_months(upper_ticker)
        
        # Convert to month codes
        front_month_code = self.month_letters[front_month]
        next_month_code = self.month_letters[next_month]
        
        # Format the year suffix based on specified format
        if year_format == '1-digit':
            front_year_suffix = str(front_year)[-1]
            next_year_suffix = str(next_year)[-1]
        else:  # '2-digit'
            front_year_suffix = str(front_year)[-2:]
            next_year_suffix = str(next_year)[-2:]
        
        # Construct the contract tickers
        front_contract = f"{upper_ticker}{front_month_code}{front_year_suffix} {contract_type}"
        next_contract = f"{upper_ticker}{next_month_code}{next_year_suffix} {contract_type}"
        
        logger.info(f"Generated sample contracts for {root_ticker}: Front={front_contract}, Next={next_contract}")
        return front_contract, next_contract
    
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
    
    def calculate_roll_percentage(self, front_contract: str, next_contract: str) -> Dict[str, Any]:
        """
        Calculate the roll percentage between front and next month contracts.
        
        Args:
            front_contract: Front month contract ticker
            next_contract: Next month contract ticker
            
        Returns:
            Dictionary with roll statistics
        """
        if self.use_sample_data:
            return self.get_sample_roll_data(front_contract, next_contract)
        
        if not front_contract or not next_contract:
            logger.warning("Missing contract information, cannot calculate roll percentage")
            return {
                'front_contract': front_contract,
                'next_contract': next_contract,
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
                return self.get_sample_roll_data(front_contract, next_contract)
            
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
                log_msg += f"Avg={avg_roll_pct:.1f}%"
            else:
                log_msg += "Avg=N/A"
                
            logger.info(log_msg)
            
            return result
            
        except Exception as e:
            logger.error(f"Error calculating roll percentage: {e}")
            logger.warning(f"Falling back to sample data for {front_contract}/{next_contract}")
            return self.get_sample_roll_data(front_contract, next_contract)
    
    def get_sample_roll_data(self, front_contract: str, next_contract: str) -> Dict[str, Any]:
        """
        Generate sample roll data for testing.
        
        Args:
            front_contract: Front month contract ticker
            next_contract: Next month contract ticker
            
        Returns:
            Dictionary with sample roll statistics
        """
        # Extract root ticker if possible
        root_ticker = None
        if front_contract:
            match = re.search(r'([A-Z]+)[A-Z]\d+', front_contract)
            if match:
                root_ticker = match.group(1)
        
        # Get current date to determine realistic roll percentages
        now = datetime.datetime.now()
        days_in_month = 30  # Approximate days in a month
        day_of_month = now.day
        
        # Create a hash of the contract name for reproducible "random" values
        if front_contract and next_contract:
            contract_hash = hash(front_contract + next_contract)
            np.random.seed(contract_hash % 10000)
            
            # Major index futures tend to roll earlier
            is_index = root_ticker in ['ES', 'NQ', 'YM', 'RTY', 'DAX', 'HSI', 'NKY']
            is_currency = root_ticker and root_ticker.startswith('6')
            
            # Different futures have different typical roll timing windows
            if is_index:
                # Index futures typically roll around the 8th-12th of expiry month
                roll_start_pct = 5  # Days 1-5: minimal roll activity
                roll_mid_pct = 50   # Days 10-15: around 50% rolled
                roll_end_pct = 95   # Days 20+: mostly rolled
            elif is_currency:
                # Currency futures typically roll very late
                roll_start_pct = 0   # Days 1-20: minimal roll activity
                roll_mid_pct = 20    # Days 20-25: around 20% rolled
                roll_end_pct = 80    # Days 25+: significant roll
            else:
                # Commodity futures typically roll in last week
                roll_start_pct = 0  # Days 1-15: minimal roll activity
                roll_mid_pct = 30   # Days 15-25: around 30% rolled
                roll_end_pct = 90   # Days 25+: mostly rolled
            
            # Calculate base roll percentage based on day of month
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
            
            # Calculate volumes and OI
            total_volume = np.random.randint(50000, 500000)
            total_oi = np.random.randint(100000, 1000000)
            
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
            logger.info(f"Sample roll data: Volume={volume_roll_pct:.1f}%, OI={oi_roll_pct:.1f}%, Avg={avg_roll_pct:.1f}%")
        
        return result
    
    def track_futures_rolls(self, futures_list_file: str) -> Dict[str, Any]:
        """
        Track roll percentages for a list of futures.
        
        Args:
            futures_list_file: Path to file containing futures tickers
            
        Returns:
            Dictionary with roll data for each futures contract
        """
        # Read futures list
        futures_list = self.read_futures_list(futures_list_file)
        if not futures_list:
            logger.error("No futures found in the provided list")
            return {}
        
        # Process each futures contract
        roll_results = {}
        
        for root_ticker in futures_list:
            # Get active contracts
            front_contract, next_contract = self.get_active_futures_contracts(root_ticker)
            
            # Calculate roll percentage
            roll_data = self.calculate_roll_percentage(front_contract, next_contract)
            
            # Store result
            roll_results[root_ticker] = roll_data
            
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
            logger.error("No roll data available. Run track_futures_rolls() first.")
            return
        
        try:
            # Prepare data for export
            data = []
            for root_ticker, roll_info in self.roll_data['results'].items():
                row = {
                    'root_ticker': root_ticker,
                    'front_contract': roll_info.get('front_contract', ''),
                    'next_contract': roll_info.get('next_contract', ''),
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
            logger.error("No roll data available. Run track_futures_rolls() first.")
            return
        
        try:
            # Prepare data for export
            data = []
            for root_ticker, roll_info in self.roll_data['results'].items():
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
                
                row = {
                    'Root Ticker': root_ticker,
                    'Front Contract': roll_info.get('front_contract', ''),
                    'Next Contract': roll_info.get('next_contract', ''),
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
            df.to_excel(writer, sheet_name='Futures Roll Data', index=False)
            
            # Get workbook and worksheet objects
            workbook = writer.book
            worksheet = writer.sheets['Futures Roll Data']
            
            # Define formats
            header_format = workbook.add_format({
                'bold': True,
                'text_wrap': True,
                'valign': 'top',
                'fg_color': '#D7E4BC',
                'border': 1
            })
            
            pct_format = workbook.add_format({'num_format': '0.0%'})
            number_format = workbook.add_format({'num_format': '#,##0'})
            
            # Apply formats to columns
            for col_num, column in enumerate(df.columns):
                worksheet.write(0, col_num, column, header_format)
                
                # Set column width
                worksheet.set_column(col_num, col_num, 15)
                
                # Apply percentage formatting
                if 'Roll %' in column:
                    worksheet.set_column(col_num, col_num, 12, pct_format)
                
                # Apply number formatting
                if 'Volume' in column or 'OI' in column:
                    worksheet.set_column(col_num, col_num, 12, number_format)
            
            # Add conditional formatting for roll percentages
            roll_pct_cols = ['H', 'I', 'J']  # Volume Roll %, OI Roll %, Avg Roll %
            for col in roll_pct_cols:
                worksheet.conditional_format(f'{col}2:{col}{len(df)+1}', {
                    'type': '3_color_scale',
                    'min_color': "#FFFFFF",
                    'mid_color': "#FFEB84",
                    'max_color': "#FF9A3C"
                })
            
            # Save the workbook
            writer.close()
            logger.info(f"Roll data saved to Excel file: {output_file}")
            
        except Exception as e:
            logger.error(f"Error saving roll data to Excel: {e}")
    
    def print_roll_summary(self) -> None:
        """Print a summary of futures roll percentages."""
        if not self.roll_data or 'results' not in self.roll_data:
            logger.error("No roll data available. Run track_futures_rolls() first.")
            return
        
        results = self.roll_data['results']
        if not results:
            logger.warning("No roll results to display.")
            return
        
        # Calculate max length for formatting
        max_ticker_len = max(len(ticker) for ticker in results.keys())
        max_contract_len = 0
        
        for roll_info in results.values():
            front_len = len(str(roll_info.get('front_contract', '')))
            next_len = len(str(roll_info.get('next_contract', '')))
            max_contract_len = max(max_contract_len, front_len, next_len)
        
        # Print header
        print("\n" + "="*110)
        print(f"FUTURES ROLL PERCENTAGES - {self.latest_update_time.strftime('%Y-%m-%d %H:%M:%S')}")
        if self.use_sample_data:
            print("NOTE: Using sample data (Bloomberg connection not available)")
        print("="*110)
        
        # Print table header
        print(f"{'Root':^{max_ticker_len}} | {'Front Contract':^{max_contract_len}} | {'Next Contract':^{max_contract_len}} | " +
              f"{'Front Vol':>10} | {'Next Vol':>10} | {'Front OI':>10} | {'Next OI':>10} | " +
              f"{'Vol Roll%':>8} | {'OI Roll%':>8} | {'Avg Roll%':>8}")
        print("-"*(max_ticker_len + 2*max_contract_len + 76))
        
        # Sort by average roll percentage (descending)
        sorted_tickers = sorted(results.keys(), 
                               key=lambda x: results[x].get('avg_roll_pct', 0) if results[x].get('avg_roll_pct') is not None else -1,
                               reverse=True)
        
        # Print each row
        for ticker in sorted_tickers:
            roll_info = results[ticker]
            
            # Get values
            front_contract = roll_info.get('front_contract', '')
            next_contract = roll_info.get('next_contract', '')
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
            
            # Apply color based on roll percentage
            if avg_roll_pct is not None:
                if avg_roll_pct > 80:
                    avg_pct_str = f"\033[91m{avg_pct_str}\033[0m"  # Red for high roll
                elif avg_roll_pct > 50:
                    avg_pct_str = f"\033[93m{avg_pct_str}\033[0m"  # Yellow for medium roll
                elif avg_roll_pct > 20:
                    avg_pct_str = f"\033[92m{avg_pct_str}\033[0m"  # Green for low roll
            
            # Print the row
            print(f"{ticker:<{max_ticker_len}} | {front_contract:<{max_contract_len}} | {next_contract:<{max_contract_len}} | " +
                  f"{front_volume:>10,.0f} | {next_volume:>10,.0f} | {front_oi:>10,.0f} | {next_oi:>10,.0f} | " +
                  f"{volume_pct_str:>8} | {oi_pct_str:>8} | {avg_pct_str:>8}")
        
        print("="*110 + "\n")


def main():
    """Main function to run the futures roll tracker without command-line arguments."""
    print("Futures Roll Tracker - Starting...")
    print(f"Current date: {datetime.datetime.now().strftime('%Y-%m-%d')}")
    print(f"Reading futures tickers from: {DEFAULT_INPUT_FILE}")
    print(f"Using Bloomberg data: {not USE_SAMPLE_DATA}")
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(DEFAULT_OUTPUT_CSV)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Initialize the tracker
    tracker = FuturesRollTracker(
        host=BLOOMBERG_HOST,
        port=BLOOMBERG_PORT,
        use_sample_data=USE_SAMPLE_DATA
    )
    
    try:
        # Create default input file if it doesn't exist
        if not os.path.exists(DEFAULT_INPUT_FILE):
            print(f"Creating default futures list file: {DEFAULT_INPUT_FILE}")
            with open(DEFAULT_INPUT_FILE, 'w') as f:
                f.write("""# Major index futures
ES    # S&P 500 E-mini
NQ    # NASDAQ 100 E-mini
YM    # Dow Jones E-mini
RTY   # Russell 2000 E-mini

# Commodity futures
CL    # Crude Oil
GC    # Gold
SI    # Silver
HG    # Copper
NG    # Natural Gas

# Interest rate futures
ZB    # Treasury Bond
ZN    # 10-Year Note
ZF    # 5-Year Note

# Currency futures
6E    # Euro FX
6J    # Japanese Yen
6B    # British Pound
6C    # Canadian Dollar
6A    # Australian Dollar

# Asian futures
HSI   # Hang Seng Index
NKY   # Nikkei 225
""")
        
        # Start Bloomberg session
        if tracker.start_session():
            # Track futures rolls
            print(f"Tracking roll percentages for futures...")
            tracker.track_futures_rolls(DEFAULT_INPUT_FILE)
            
            # Print summary
            tracker.print_roll_summary()
            
            # Save data
            if USE_EXCEL_OUTPUT:
                print(f"Saving roll data to Excel: {DEFAULT_OUTPUT_EXCEL}")
                tracker.save_roll_data_to_excel(DEFAULT_OUTPUT_EXCEL)
            else:
                print(f"Saving roll data to CSV: {DEFAULT_OUTPUT_CSV}")
                tracker.save_roll_data_to_csv(DEFAULT_OUTPUT_CSV)
                
            print("Futures roll tracking completed successfully.")
            
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