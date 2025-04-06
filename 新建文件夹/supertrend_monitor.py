#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Supertrend Indicator Monitor
This script monitors a list of instruments from a file and checks for supertrend indicator
breakups and breakdowns every 30 minutes.
"""

import blpapi
import pandas as pd
import numpy as np
import time
import datetime
import os
import logging
from typing import List, Optional
import colorama
from colorama import Fore, Back, Style

# Initialize colorama
colorama.init()

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


class SupertrendMonitor:
    """Class to monitor supertrend indicator for a list of instruments"""

    def __init__(self, host: str = "localhost", port: int = 8194):
        """Initialize the Supertrend Monitor

        Args:
            host: Bloomberg server host
            port: Bloomberg server port
        """
        self.host = host
        self.port = port
        self.session = None
        self.refdata_service = None
        
        # Supertrend parameters
        self.atr_period = 10
        self.atr_multiplier = 3.0
        
        # Instruments list
        self.instruments = []
        
        # Previous supertrend states (to detect crossovers)
        self.prev_states = {}

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

    def read_instruments(self, file_path: str) -> bool:
        """Read instruments from a file

        Args:
            file_path: Path to the instruments file

        Returns:
            bool: True if file was read successfully, False otherwise
        """
        try:
            # Check if file exists
            if not os.path.exists(file_path):
                logger.error(f"File not found: {file_path}")
                return False
            
            # Read file
            with open(file_path, 'r') as f:
                # Strip whitespace and filter out empty lines
                self.instruments = [line.strip() for line in f if line.strip()]
            
            logger.info(f"Read {len(self.instruments)} instruments from {file_path}")
            
            # Initialize previous states
            for instrument in self.instruments:
                self.prev_states[instrument] = None
            
            # Check if any instruments were read
            if not self.instruments:
                logger.error("No instruments found in the file.")
                return False
            
            return True
        
        except Exception as e:
            logger.error(f"Error reading instruments file: {e}")
            return False

    def get_intraday_bars(
        self, 
        security: str, 
        interval: int = 30, 
        bars_count: int = 100
    ) -> pd.DataFrame:
        """Get intraday bar data for a security

        Args:
            security: Bloomberg security identifier
            interval: Bar interval in minutes
            bars_count: Number of bars to retrieve

        Returns:
            pd.DataFrame: DataFrame containing the bar data
        """
        logger.info(f"Fetching {interval}-minute bars for {security}...")
        
        # Calculate start and end dates
        end_date = datetime.datetime.now()
        # Calculate start date based on bars_count and interval
        start_date = end_date - datetime.timedelta(minutes=interval * bars_count)
        
        # Create the request
        request = self.refdata_service.createRequest(INTRADAY_BAR_REQUEST)
        request.set("security", security)
        request.set("eventType", "TRADE")
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
                            
                            bars_data.append({
                                "time": time,
                                "open": open_price,
                                "high": high_price,
                                "low": low_price,
                                "close": close_price,
                                "volume": volume
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

    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range (ATR)

        Args:
            df: DataFrame with OHLC data
            period: ATR period

        Returns:
            pd.Series: ATR values
        """
        high = df['high']
        low = df['low']
        close = df['close'].shift(1)
        
        # Calculate True Range
        tr1 = high - low
        tr2 = abs(high - close)
        tr3 = abs(low - close)
        
        tr = pd.DataFrame([tr1, tr2, tr3]).max()
        
        # Calculate ATR
        atr = tr.rolling(window=period).mean()
        
        return atr

    def calculate_supertrend(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Supertrend indicator

        Args:
            df: DataFrame with OHLC data

        Returns:
            pd.DataFrame: DataFrame with Supertrend indicator
        """
        # Make a copy of the dataframe
        df = df.copy()
        
        # Calculate ATR
        df['atr'] = self.calculate_atr(df, self.atr_period)
        
        # Calculate basic upper and lower bands
        df['basic_upper'] = (df['high'] + df['low']) / 2 + self.atr_multiplier * df['atr']
        df['basic_lower'] = (df['high'] + df['low']) / 2 - self.atr_multiplier * df['atr']
        
        # Initialize Supertrend columns
        df['supertrend_upper'] = 0.0
        df['supertrend_lower'] = 0.0
        df['supertrend'] = 0.0
        df['supertrend_direction'] = 0  # 1 for uptrend, -1 for downtrend
        
        # Calculate Supertrend
        for i in range(1, len(df)):
            # Upper band
            if df['basic_upper'].iloc[i] < df['supertrend_upper'].iloc[i-1] or df['close'].iloc[i-1] > df['supertrend_upper'].iloc[i-1]:
                df.loc[df.index[i], 'supertrend_upper'] = df['basic_upper'].iloc[i]
            else:
                df.loc[df.index[i], 'supertrend_upper'] = df['supertrend_upper'].iloc[i-1]
            
            # Lower band
            if df['basic_lower'].iloc[i] > df['supertrend_lower'].iloc[i-1] or df['close'].iloc[i-1] < df['supertrend_lower'].iloc[i-1]:
                df.loc[df.index[i], 'supertrend_lower'] = df['basic_lower'].iloc[i]
            else:
                df.loc[df.index[i], 'supertrend_lower'] = df['supertrend_lower'].iloc[i-1]
            
            # Supertrend
            if df['supertrend'].iloc[i-1] == df['supertrend_upper'].iloc[i-1] and df['close'].iloc[i] <= df['supertrend_upper'].iloc[i]:
                df.loc[df.index[i], 'supertrend'] = df['supertrend_upper'].iloc[i]
            elif df['supertrend'].iloc[i-1] == df['supertrend_upper'].iloc[i-1] and df['close'].iloc[i] > df['supertrend_upper'].iloc[i]:
                df.loc[df.index[i], 'supertrend'] = df['supertrend_lower'].iloc[i]
            elif df['supertrend'].iloc[i-1] == df['supertrend_lower'].iloc[i-1] and df['close'].iloc[i] >= df['supertrend_lower'].iloc[i]:
                df.loc[df.index[i], 'supertrend'] = df['supertrend_lower'].iloc[i]
            elif df['supertrend'].iloc[i-1] == df['supertrend_lower'].iloc[i-1] and df['close'].iloc[i] < df['supertrend_lower'].iloc[i]:
                df.loc[df.index[i], 'supertrend'] = df['supertrend_upper'].iloc[i]
            
            # Supertrend direction
            if df['close'].iloc[i] > df['supertrend'].iloc[i]:
                df.loc[df.index[i], 'supertrend_direction'] = 1  # Uptrend
            else:
                df.loc[df.index[i], 'supertrend_direction'] = -1  # Downtrend
        
        return df

    def check_supertrend_signals(self, security: str) -> Optional[str]:
        """Check for supertrend breakup/breakdown signals

        Args:
            security: Bloomberg security identifier

        Returns:
            Optional[str]: Signal type ('breakup', 'breakdown', or None)
        """
        # Get intraday bars
        df = self.get_intraday_bars(security)
        
        if df.empty:
            logger.warning(f"No data available for {security}")
            return None
        
        # Calculate Supertrend
        df = self.calculate_supertrend(df)
        
        # Get current and previous direction
        current_direction = df['supertrend_direction'].iloc[-1]
        prev_direction = df['supertrend_direction'].iloc[-2] if len(df) > 1 else None
        
        # Check if this is the first time we're checking this security
        if self.prev_states[security] is None:
            self.prev_states[security] = current_direction
            return None
        
        # Check for direction change
        if current_direction != self.prev_states[security]:
            # Update previous state
            self.prev_states[security] = current_direction
            
            if current_direction == 1:
                return "breakup"
            elif current_direction == -1:
                return "breakdown"
        
        return None

    def display_signal(self, security: str, signal: str) -> None:
        """Display supertrend signal in the terminal

        Args:
            security: Bloomberg security identifier
            signal: Signal type ('breakup' or 'breakdown')
        """
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        if signal == "breakup":
            # Flash green for breakup
            print(f"{Fore.GREEN}{Back.WHITE}{timestamp} | {security}: BREAK UP{Style.RESET_ALL}")
        elif signal == "breakdown":
            # Flash red for breakdown
            print(f"{Fore.RED}{Back.WHITE}{timestamp} | {security}: BREAK DOWN{Style.RESET_ALL}")

    def monitor_supertrend(self, interval_minutes: int = 30) -> None:
        """Monitor supertrend signals at regular intervals

        Args:
            interval_minutes: Interval between checks in minutes
        """
        logger.info(f"Starting supertrend monitoring with {len(self.instruments)} instruments")
        logger.info(f"Interval: {interval_minutes} minutes")
        logger.info(f"ATR Period: {self.atr_period}, Multiplier: {self.atr_multiplier}")
        
        print("\n" + "="*80)
        print("SUPERTREND INDICATOR MONITOR")
        print("="*80)
        print(f"{Fore.GREEN}GREEN: Breakup (Price crosses above Supertrend){Style.RESET_ALL}")
        print(f"{Fore.RED}RED: Breakdown (Price crosses below Supertrend){Style.RESET_ALL}")
        print("="*80 + "\n")
        
        try:
            while True:
                print(f"\nChecking supertrend signals at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}...")
                
                # Check each instrument
                for instrument in self.instruments:
                    signal = self.check_supertrend_signals(instrument)
                    
                    if signal:
                        self.display_signal(instrument, signal)
                
                # Calculate time until next check
                next_check = datetime.datetime.now() + datetime.timedelta(minutes=interval_minutes)
                print(f"Next check at {next_check.strftime('%Y-%m-%d %H:%M:%S')}")
                
                # Wait for next check
                time.sleep(interval_minutes * 60)
        
        except KeyboardInterrupt:
            logger.info("Monitoring stopped by user")
            print("\nMonitoring stopped. Press Enter to exit...")
        
        except Exception as e:
            logger.error(f"Error during monitoring: {e}")
            print(f"\n{Fore.RED}Error during monitoring: {e}{Style.RESET_ALL}")


def create_sample_instruments_file(file_path: str) -> None:
    """Create a sample instruments file

    Args:
        file_path: Path to the instruments file
    """
    with open(file_path, 'w') as f:
        f.write("AAPL US Equity\n")
        f.write("MSFT US Equity\n")
        f.write("AMZN US Equity\n")
        f.write("GOOGL US Equity\n")
        f.write("META US Equity\n")
    
    print(f"Sample instruments file created: {file_path}")


def main():
    """Main function to run the Supertrend Monitor"""
    # Check if instruments file exists, create sample if not
    instruments_file = "instruments.txt"
    if not os.path.exists(instruments_file):
        create_sample_instruments_file(instruments_file)
        print(f"Please edit {instruments_file} with your instruments and run the script again.")
        return
    
    # Initialize the Supertrend Monitor
    monitor = SupertrendMonitor()
    
    try:
        # Start the Bloomberg session
        if not monitor.start_session():
            logger.error("Failed to initialize Bloomberg session. Exiting.")
            return
        
        # Read instruments from file
        if not monitor.read_instruments(instruments_file):
            logger.error("Failed to read instruments from file. Exiting.")
            return
        
        # Start monitoring
        monitor.monitor_supertrend()
    
    except Exception as e:
        logger.error(f"An error occurred: {e}")
    
    finally:
        # Stop the Bloomberg session
        monitor.stop_session()


if __name__ == "__main__":
    main()
