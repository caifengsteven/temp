#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Mock Bloomberg VPIN Calculator
This script simulates Bloomberg data for testing VPIN calculation.
"""

import pandas as pd
import numpy as np
import datetime
import logging
from typing import List
import math
import random

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MockBloombergVPINCalculator:
    """Class to simulate Bloomberg data fetching and calculate VPIN"""

    def __init__(self):
        """Initialize the Mock Bloomberg VPIN calculator"""
        # VPIN parameters
        self.num_buckets = 20  # Number of buckets for VPIN calculation (reduced from 50)
        self.window_size = 10  # Window size for VPIN calculation (reduced from 50)
        self.sigma_multiplier = 1.0  # Multiplier for standard deviation in bulk classification

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

    def generate_mock_data(self, security: str, days: int = 30) -> pd.DataFrame:
        """Generate mock price and volume data for testing

        Args:
            security: Security identifier
            days: Number of days of data to generate

        Returns:
            pd.DataFrame: DataFrame with mock price and volume data
        """
        logger.info(f"Generating mock data for {security}...")

        # Set random seed based on security name for consistent results
        random.seed(sum(ord(c) for c in security))

        # Generate timestamps (1-minute bars for the specified number of days)
        end_date = datetime.datetime.now().replace(hour=16, minute=0, second=0, microsecond=0)
        start_date = end_date - datetime.timedelta(days=days)

        # Generate timestamps for market hours (9:30 AM to 4:00 PM)
        timestamps = []
        current_date = start_date

        while current_date <= end_date:
            # Skip weekends
            if current_date.weekday() < 5:  # Monday to Friday
                market_open = current_date.replace(hour=9, minute=30, second=0, microsecond=0)
                market_close = current_date.replace(hour=16, minute=0, second=0, microsecond=0)

                current_time = market_open
                while current_time <= market_close:
                    timestamps.append(current_time)
                    current_time += datetime.timedelta(minutes=1)

            current_date += datetime.timedelta(days=1)

        # Generate price data (random walk with drift)
        n = len(timestamps)

        # Initial price between $10 and $1000
        initial_price = 10 + 990 * random.random()

        # Daily volatility between 0.5% and 2%
        daily_volatility = 0.005 + 0.015 * random.random()

        # Minute volatility
        minute_volatility = daily_volatility / math.sqrt(390)  # 390 minutes in a trading day

        # Generate returns
        returns = np.random.normal(0.0001, minute_volatility, n)  # Small positive drift

        # Calculate prices
        prices = initial_price * np.cumprod(1 + returns)

        # Generate volume data (random with some clustering)
        base_volume = 1000 + 9000 * random.random()  # Base volume between 1,000 and 10,000
        volume_volatility = 0.5 + 1.5 * random.random()  # Volume volatility

        # Generate volumes with log-normal distribution
        volumes = np.random.lognormal(math.log(base_volume), volume_volatility, n).astype(int)

        # Create DataFrame
        df = pd.DataFrame({
            'time': timestamps,
            'open': prices,
            'high': prices * (1 + 0.001 * np.random.random(n)),  # High slightly above close
            'low': prices * (1 - 0.001 * np.random.random(n)),   # Low slightly below close
            'close': prices,
            'volume': volumes,
            'numEvents': (volumes / 10).astype(int)  # Approximate number of trades
        })

        logger.info(f"Generated {len(df)} bars of mock data for {security}")
        return df

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
        """Calculate and display VPIN for a security using mock data

        Args:
            security: Security identifier

        Returns:
            float: VPIN value
        """
        # Generate mock data
        df = self.generate_mock_data(security)

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
    """Main function to run the Mock Bloomberg VPIN calculator"""
    # Initialize the Mock Bloomberg VPIN calculator
    calculator = MockBloombergVPINCalculator()

    try:
        # Read instruments from file
        instruments = calculator.read_instruments("instruments.txt")

        if not instruments:
            logger.error("No instruments found or error reading instruments file. Exiting.")
            return

        print("\n" + "="*50)
        print("VPIN CALCULATION RESULTS (MOCK DATA)")
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


if __name__ == "__main__":
    main()
