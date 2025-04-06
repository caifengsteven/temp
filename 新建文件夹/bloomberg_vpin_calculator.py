#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Bloomberg VPIN Calculator
This script connects to Bloomberg, reads instruments from a file,
calculates VPIN (Volume-Synchronized Probability of Informed Trading) for each instrument,
and displays the results in the terminal.
"""

import blpapi
import pandas as pd
import numpy as np
import datetime
import logging
from typing import List, Optional
import math

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


class BloombergVPINCalculator:
    """Class to handle Bloomberg data fetching and VPIN calculation"""

    def __init__(self, host: str = "localhost", port: int = 8194):
        """Initialize the Bloomberg VPIN calculator

        Args:
            host: Bloomberg server host
            port: Bloomberg server port
        """
        self.host = host
        self.port = port
        self.session = None
        self.refdata_service = None

        # VPIN parameters
        self.num_buckets = 20  # Number of buckets for VPIN calculation (reduced from 50)
        self.window_size = 10  # Window size for VPIN calculation (reduced from 50)
        self.sigma_multiplier = 1.0  # Multiplier for standard deviation in bulk classification

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
        interval: int = 1,
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
        logger.info(f"Fetching intraday bars for {security}...")

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
        request.set("interval", interval)  # 1-minute bars for more granular data

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

    def bulk_classify_volume(self, df: pd.DataFrame) -> pd.DataFrame:
        """Classify volume as buy or sell using bulk volume classification

        Args:
            df: DataFrame containing price and volume data

        Returns:
            pd.DataFrame: DataFrame with buy and sell volume classifications
        """
        # Calculate price changes
        df['price_change'] = df['close'].diff()

        # Calculate standard deviation of price changes
        price_change_std = df['price_change'].std()

        # Classify volume using normal CDF
        df['buy_volume'] = df['volume'] * (0.5 + 0.5 * np.array([math.erf(x / (self.sigma_multiplier * price_change_std * math.sqrt(2))) for x in df['price_change']]))
        df['sell_volume'] = df['volume'] - df['buy_volume']

        return df

    def calculate_vpin(self, df: pd.DataFrame) -> float:
        """Calculate VPIN for a security

        Args:
            df: DataFrame containing buy and sell volume data

        Returns:
            float: VPIN value
        """
        if df.empty:
            return np.nan

        # Ensure we have buy and sell volume
        if 'buy_volume' not in df.columns or 'sell_volume' not in df.columns:
            df = self.bulk_classify_volume(df)

        # Calculate total volume
        total_volume = df['volume'].sum()

        # Calculate bucket size
        bucket_size = total_volume / self.num_buckets

        # Initialize buckets
        buckets = []
        current_bucket = {'buy_volume': 0, 'sell_volume': 0}
        remaining_volume = bucket_size

        # Fill buckets
        for _, row in df.iterrows():
            if row['volume'] <= remaining_volume:
                # Add entire bar to current bucket
                current_bucket['buy_volume'] += row['buy_volume']
                current_bucket['sell_volume'] += row['sell_volume']
                remaining_volume -= row['volume']
            else:
                # Fill current bucket and start a new one
                ratio = remaining_volume / row['volume']
                current_bucket['buy_volume'] += row['buy_volume'] * ratio
                current_bucket['sell_volume'] += row['sell_volume'] * ratio

                # Add current bucket to buckets list
                buckets.append(current_bucket)

                # Start a new bucket with remaining volume
                current_bucket = {
                    'buy_volume': row['buy_volume'] * (1 - ratio),
                    'sell_volume': row['sell_volume'] * (1 - ratio)
                }
                remaining_volume = bucket_size - (row['volume'] * (1 - ratio))

            # If bucket is full, add it to buckets list and start a new one
            if remaining_volume <= 0:
                buckets.append(current_bucket)
                current_bucket = {'buy_volume': 0, 'sell_volume': 0}
                remaining_volume = bucket_size

        # Add the last bucket if it has any volume
        if current_bucket['buy_volume'] > 0 or current_bucket['sell_volume'] > 0:
            buckets.append(current_bucket)

        # Calculate VPIN for each window
        vpin_values = []

        for i in range(len(buckets) - self.window_size + 1):
            window = buckets[i:i+self.window_size]

            # Calculate volume imbalance for each bucket in the window
            imbalances = [abs(b['buy_volume'] - b['sell_volume']) for b in window]

            # Calculate total volume in the window
            total_window_volume = sum(b['buy_volume'] + b['sell_volume'] for b in window)

            # Calculate VPIN
            vpin = sum(imbalances) / total_window_volume if total_window_volume > 0 else np.nan
            vpin_values.append(vpin)

        # Return the latest VPIN value
        return vpin_values[-1] if vpin_values else np.nan

    def calculate_and_display_vpin(self, security: str) -> float:
        """Calculate and display VPIN for a security

        Args:
            security: Bloomberg security identifier

        Returns:
            float: VPIN value
        """
        # Get intraday bar data
        df = self.get_intraday_bars(security)

        if df.empty:
            logger.warning(f"No data available for {security}")
            return np.nan

        # Classify volume
        df = self.bulk_classify_volume(df)

        # Calculate VPIN
        vpin = self.calculate_vpin(df)

        # Display VPIN
        if not np.isnan(vpin):
            logger.info(f"VPIN for {security}: {vpin:.4f}")
            print(f"VPIN for {security}: {vpin:.4f}")
        else:
            logger.warning(f"Could not calculate VPIN for {security}")
            print(f"Could not calculate VPIN for {security}: Insufficient data")

        return vpin


def main():
    """Main function to run the Bloomberg VPIN calculator"""
    # Initialize the Bloomberg VPIN calculator
    calculator = BloombergVPINCalculator()

    try:
        # Start the Bloomberg session
        if not calculator.start_session():
            logger.error("Failed to initialize Bloomberg session. Exiting.")
            return

        # Read instruments from file
        instruments = calculator.read_instruments("instruments.txt")

        if not instruments:
            logger.error("No instruments found or error reading instruments file. Exiting.")
            return

        print("\n" + "="*50)
        print("VPIN CALCULATION RESULTS")
        print("="*50)

        # Calculate VPIN for each instrument
        vpin_results = {}
        for instrument in instruments:
            vpin = calculator.calculate_and_display_vpin(instrument)
            vpin_results[instrument] = vpin

        print("\n" + "="*50)
        print("SUMMARY OF VPIN VALUES")
        print("="*50)

        # Display summary of VPIN values
        for instrument, vpin in vpin_results.items():
            if not np.isnan(vpin):
                print(f"{instrument}: {vpin:.4f}")
            else:
                print(f"{instrument}: N/A (Insufficient data)")

        print("="*50)
        logger.info("VPIN calculation completed successfully.")

    except Exception as e:
        logger.error(f"An error occurred: {e}")

    finally:
        # Stop the Bloomberg session
        calculator.stop_session()


if __name__ == "__main__":
    main()
