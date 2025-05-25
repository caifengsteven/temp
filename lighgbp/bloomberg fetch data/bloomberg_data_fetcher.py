#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Bloomberg Data Fetcher
This script connects to Bloomberg, reads instruments from a file,
fetches 30-minute bar data for each instrument, and saves the data to files.
"""

import blpapi
import pandas as pd
import datetime
import os
import logging
from typing import List, Dict, Any, Optional

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Bloomberg API constants
REFDATA_SVC = "//blp/refdata"
INTRADAY_BAR_REQUEST = "IntradayBarRequest"
SESSION_STARTED = blpapi.Name("SessionStarted")
SESSION_STARTUP_FAILURE = blpapi.Name("SessionStartupFailure")
SERVICE_OPENED = blpapi.Name("ServiceOpened")
SERVICE_OPEN_FAILURE = blpapi.Name("ServiceOpenFailure")
RESPONSE_ERROR = blpapi.Name("ResponseError")
BAR_DATA = blpapi.Name("barData")
BAR_TICK_DATA = blpapi.Name("barTickData")


class BloombergDataFetcher:
    """Class to handle Bloomberg data fetching operations"""

    def __init__(self, host: str = "localhost", port: int = 8194):
        """Initialize the Bloomberg data fetcher

        Args:
            host: Bloomberg server host
            port: Bloomberg server port
        """
        self.host = host
        self.port = port
        self.session = None
        self.refdata_service = None

    def start_session(self) -> bool:
        """Start a Bloomberg API session

        Returns:
            bool: True if session started successfully, False otherwise
        """
        logger.info("Starting Bloomberg API session...")
        
        # Initialize session options
        session_options = blpapi.SessionOptions()
        session_options.setServerHost(self.host)
        session_options.setServerPort(self.port)
        
        # Create a session
        self.session = blpapi.Session(session_options)
        
        # Start the session
        if not self.session.start():
            logger.error("Failed to start session.")
            return False
        
        logger.info("Session started successfully.")
        
        # Open the reference data service
        if not self.session.openService(REFDATA_SVC):
            logger.error("Failed to open reference data service.")
            return False
        
        self.refdata_service = self.session.getService(REFDATA_SVC)
        logger.info("Reference data service opened successfully.")
        
        return True

    def stop_session(self) -> None:
        """Stop the Bloomberg API session"""
        if self.session:
            self.session.stop()
            logger.info("Session stopped.")

    def get_intraday_bars(
        self, 
        security: str, 
        event_type: str = "TRADE", 
        interval: int = 30, 
        start_date: Optional[datetime.datetime] = None,
        end_date: Optional[datetime.datetime] = None
    ) -> pd.DataFrame:
        """Get intraday bar data for a security

        Args:
            security: Bloomberg security identifier
            event_type: Type of event (TRADE, BID, ASK, etc.)
            interval: Bar interval in minutes
            start_date: Start date for data retrieval
            end_date: End date for data retrieval

        Returns:
            pd.DataFrame: DataFrame containing the bar data
        """
        logger.info(f"Fetching 30-minute bars for {security}...")
        
        # If dates not provided, use maximum range (last 140 days is the Bloomberg limit)
        if not end_date:
            end_date = datetime.datetime.now()
        if not start_date:
            # Bloomberg typically allows up to 140 days of intraday data
            start_date = end_date - datetime.timedelta(days=140)
        
        # Create the request
        request = self.refdata_service.createRequest(INTRADAY_BAR_REQUEST)
        request.set("security", security)
        request.set("eventType", event_type)
        request.set("interval", interval)  # 30-minute bars
        
        # Set the date range
        request.set("startDateTime", start_date)
        request.set("endDateTime", end_date)
        
        logger.info(f"Request period: {start_date} to {end_date}")
        
        # Send the request
        self.session.sendRequest(request)
        
        # Process the response
        bars_data = []
        
        while True:
            event = self.session.nextEvent(500)  # Timeout in milliseconds
            
            for msg in event:
                if msg.hasElement(RESPONSE_ERROR):
                    error_info = msg.getElement(RESPONSE_ERROR)
                    logger.error(f"Request failed: {error_info}")
                    return pd.DataFrame()
                
                if msg.hasElement(BAR_DATA):
                    bar_data = msg.getElement(BAR_DATA)
                    
                    if bar_data.hasElement(BAR_TICK_DATA):
                        tick_data = bar_data.getElement(BAR_TICK_DATA)
                        
                        for i in range(tick_data.numValues()):
                            bar = tick_data.getValue(i)
                            
                            time = bar.getElementAsDatetime("time")
                            open_price = bar.getElementAsFloat("open")
                            high_price = bar.getElementAsFloat("high")
                            low_price = bar.getElementAsFloat("low")
                            close_price = bar.getElementAsFloat("close")
                            volume = bar.getElementAsInteger("volume")
                            num_events = bar.getElementAsInteger("numEvents")
                            
                            bars_data.append({
                                "time": time,
                                "open": open_price,
                                "high": high_price,
                                "low": low_price,
                                "close": close_price,
                                "volume": volume,
                                "numEvents": num_events
                            })
            
            if event.eventType() == blpapi.Event.RESPONSE:
                break
        
        # Convert to DataFrame
        if bars_data:
            df = pd.DataFrame(bars_data)
            logger.info(f"Retrieved {len(df)} bars for {security}")
            return df
        else:
            logger.warning(f"No data retrieved for {security}")
            return pd.DataFrame()

    def read_instruments(self, file_path: str) -> List[str]:
        """Read instruments from a file

        Args:
            file_path: Path to the instruments file

        Returns:
            List[str]: List of instrument identifiers
        """
        try:
            with open(file_path, 'r') as f:
                # Strip whitespace and filter out empty lines
                instruments = [line.strip() for line in f if line.strip()]
            
            logger.info(f"Read {len(instruments)} instruments from {file_path}")
            return instruments
        except Exception as e:
            logger.error(f"Error reading instruments file: {e}")
            return []

    def save_data_to_csv(self, data: pd.DataFrame, security: str, output_dir: str = "output") -> str:
        """Save data to a CSV file

        Args:
            data: DataFrame containing the data
            security: Security identifier
            output_dir: Directory to save the output file

        Returns:
            str: Path to the saved file
        """
        if data.empty:
            logger.warning(f"No data to save for {security}")
            return ""
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Create a valid filename from the security identifier
        filename = security.replace(" ", "_").replace("/", "_").replace("\\", "_")
        file_path = os.path.join(output_dir, f"{filename}_30min_bars.csv")
        
        # Save to CSV
        data.to_csv(file_path, index=False)
        logger.info(f"Data saved to {file_path}")
        
        return file_path


def main():
    """Main function to run the Bloomberg data fetcher"""
    # Initialize the Bloomberg data fetcher
    fetcher = BloombergDataFetcher()
    
    try:
        # Start the Bloomberg session
        if not fetcher.start_session():
            logger.error("Failed to initialize Bloomberg session. Exiting.")
            return
        
        # Read instruments from file
        instruments = fetcher.read_instruments("instruments.txt")
        
        if not instruments:
            logger.error("No instruments found or error reading instruments file. Exiting.")
            return
        
        # Create output directory
        output_dir = "bloomberg_data"
        os.makedirs(output_dir, exist_ok=True)
        
        # Process each instrument
        for instrument in instruments:
            # Get intraday bar data
            data = fetcher.get_intraday_bars(instrument)
            
            # Save data to file
            if not data.empty:
                fetcher.save_data_to_csv(data, instrument, output_dir)
            else:
                logger.warning(f"No data retrieved for {instrument}")
        
        logger.info("Data retrieval completed successfully.")
    
    except Exception as e:
        logger.error(f"An error occurred: {e}")
    
    finally:
        # Stop the Bloomberg session
        fetcher.stop_session()


if __name__ == "__main__":
    main()
