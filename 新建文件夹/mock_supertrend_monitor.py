#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Mock Supertrend Indicator Monitor
This script simulates the Supertrend Monitor without requiring a Bloomberg connection.
It reads instruments from a file and simulates supertrend indicator signals.
"""

import pandas as pd
import numpy as np
import time
import datetime
import os
import logging
import random
from typing import Optional
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


class MockSupertrendMonitor:
    """Class to simulate supertrend indicator monitoring"""

    def __init__(self):
        """Initialize the Mock Supertrend Monitor"""
        # Supertrend parameters
        self.atr_period = 10
        self.atr_multiplier = 3.0

        # Instruments list
        self.instruments = []

        # Previous supertrend states (to detect crossovers)
        self.prev_states = {}

        # Simulated price data
        self.price_data = {}

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

            # Check if any instruments were read
            if not self.instruments:
                logger.error("No instruments found in the file.")
                return False

            # Initialize previous states and price data
            for instrument in self.instruments:
                self.prev_states[instrument] = None
                try:
                    self.price_data[instrument] = self.generate_mock_price_data(instrument)
                except Exception as e:
                    logger.error(f"Error generating mock data for {instrument}: {e}")
                    return False

            return True

        except Exception as e:
            logger.error(f"Error reading instruments file: {e}")
            return False

    def generate_mock_price_data(self, security: str, bars: int = 100) -> pd.DataFrame:
        """Generate mock price data for a security

        Args:
            security: Security identifier
            bars: Number of bars to generate

        Returns:
            pd.DataFrame: DataFrame with mock OHLC data
        """
        # Set random seed based on security name for consistent results
        random.seed(sum(ord(c) for c in security))

        # Generate timestamps (30-minute bars)
        end_time = datetime.datetime.now().replace(second=0, microsecond=0)
        end_time = end_time - datetime.timedelta(minutes=end_time.minute % 30)

        timestamps = [end_time - datetime.timedelta(minutes=30 * i) for i in range(bars)]
        timestamps.reverse()  # Oldest first

        # Generate price data (random walk with drift)
        # Initial price between $50 and $500
        initial_price = 50 + 450 * random.random()

        # Daily volatility between 1% and 3%
        daily_volatility = 0.01 + 0.02 * random.random()

        # 30-minute volatility
        bar_volatility = daily_volatility / np.sqrt(13)  # ~13 30-minute bars in a trading day

        # Generate returns
        returns = np.random.normal(0.0001, bar_volatility, bars)  # Small positive drift

        # Calculate prices
        prices = initial_price * np.cumprod(1 + returns)

        # Generate OHLC data
        data = []
        for i, timestamp in enumerate(timestamps):
            close = prices[i]
            # Random intrabar volatility
            intrabar_vol = close * bar_volatility * random.uniform(0.5, 1.5)

            # Generate OHLC
            high = close + intrabar_vol * random.uniform(0, 1)
            low = close - intrabar_vol * random.uniform(0, 1)
            open_price = low + (high - low) * random.random()

            # Ensure high >= open, close and low <= open, close
            high = max(high, open_price, close)
            low = min(low, open_price, close)

            # Generate volume
            volume = int(np.random.lognormal(10, 1))

            data.append({
                'time': timestamp,
                'open': open_price,
                'high': high,
                'low': low,
                'close': close,
                'volume': volume
            })

        return pd.DataFrame(data)

    def update_mock_price_data(self, security: str) -> None:
        """Update mock price data with a new bar

        Args:
            security: Security identifier
        """
        df = self.price_data[security]

        # Get last bar
        last_bar = df.iloc[-1].copy()

        # Calculate new timestamp (30 minutes after last bar)
        new_time = last_bar['time'] + datetime.timedelta(minutes=30)

        # Calculate new prices
        last_close = last_bar['close']

        # Daily volatility between 1% and 3%
        daily_volatility = 0.01 + 0.02 * random.random()

        # 30-minute volatility
        bar_volatility = daily_volatility / np.sqrt(13)  # ~13 30-minute bars in a trading day

        # Generate return
        ret = np.random.normal(0.0001, bar_volatility)

        # Calculate new close
        new_close = last_close * (1 + ret)

        # Random intrabar volatility
        intrabar_vol = new_close * bar_volatility * random.uniform(0.5, 1.5)

        # Generate OHLC
        new_high = new_close + intrabar_vol * random.uniform(0, 1)
        new_low = new_close - intrabar_vol * random.uniform(0, 1)
        new_open = new_low + (new_high - new_low) * random.random()

        # Ensure high >= open, close and low <= open, close
        new_high = max(new_high, new_open, new_close)
        new_low = min(new_low, new_open, new_close)

        # Generate volume
        new_volume = int(np.random.lognormal(10, 1))

        # Create new bar
        new_bar = pd.DataFrame([{
            'time': new_time,
            'open': new_open,
            'high': new_high,
            'low': new_low,
            'close': new_close,
            'volume': new_volume
        }])

        # Add new bar to dataframe
        self.price_data[security] = pd.concat([df, new_bar], ignore_index=True)

        # Remove oldest bar to keep dataframe size constant
        self.price_data[security] = self.price_data[security].iloc[1:].reset_index(drop=True)

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
            security: Security identifier

        Returns:
            Optional[str]: Signal type ('breakup', 'breakdown', or None)
        """
        # Update price data with a new bar
        self.update_mock_price_data(security)

        # Get price data
        df = self.price_data[security]

        # Calculate Supertrend
        df = self.calculate_supertrend(df)

        # Get current direction
        current_direction = df['supertrend_direction'].iloc[-1]

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
            security: Security identifier
            signal: Signal type ('breakup' or 'breakdown')
        """
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if signal == "breakup":
            # Flash green for breakup
            print(f"{Fore.GREEN}{Back.WHITE}{timestamp} | {security}: BREAK UP{Style.RESET_ALL}")
        elif signal == "breakdown":
            # Flash red for breakdown
            print(f"{Fore.RED}{Back.WHITE}{timestamp} | {security}: BREAK DOWN{Style.RESET_ALL}")

    def simulate_market_events(self) -> None:
        """Simulate market events that might trigger supertrend signals"""
        # Randomly select an instrument to create a trend change
        if self.instruments and random.random() < 0.3:  # 30% chance of an event
            security = random.choice(self.instruments)

            # Get price data
            df = self.price_data[security]

            # Calculate Supertrend
            df = self.calculate_supertrend(df)

            # Get current direction
            current_direction = df['supertrend_direction'].iloc[-1]

            # Decide whether to create a trend change
            if random.random() < 0.5:  # 50% chance of trend change
                # Create a trend change
                new_direction = -current_direction  # Reverse direction

                # Get last bar
                last_bar = df.iloc[-1]

                # Get supertrend value
                supertrend_value = last_bar['supertrend']

                # Update price data to create a trend change
                if new_direction == 1:  # Change to uptrend
                    # Set close price above supertrend
                    new_close = supertrend_value * 1.02  # 2% above supertrend
                else:  # Change to downtrend
                    # Set close price below supertrend
                    new_close = supertrend_value * 0.98  # 2% below supertrend

                # Update last bar
                self.price_data[security].loc[len(df)-1, 'close'] = new_close

                # Ensure high and low are consistent
                if new_close > self.price_data[security].loc[len(df)-1, 'high']:
                    self.price_data[security].loc[len(df)-1, 'high'] = new_close
                if new_close < self.price_data[security].loc[len(df)-1, 'low']:
                    self.price_data[security].loc[len(df)-1, 'low'] = new_close

    def monitor_supertrend(self, interval_seconds: int = 5) -> None:
        """Monitor supertrend signals at regular intervals

        Args:
            interval_seconds: Interval between checks in seconds (for mock version)
        """
        logger.info(f"Starting mock supertrend monitoring with {len(self.instruments)} instruments")
        logger.info(f"Interval: {interval_seconds} seconds (accelerated for demonstration)")
        logger.info(f"ATR Period: {self.atr_period}, Multiplier: {self.atr_multiplier}")

        print("\n" + "="*80)
        print("MOCK SUPERTREND INDICATOR MONITOR")
        print("="*80)
        print(f"{Fore.GREEN}GREEN: Breakup (Price crosses above Supertrend){Style.RESET_ALL}")
        print(f"{Fore.RED}RED: Breakdown (Price crosses below Supertrend){Style.RESET_ALL}")
        print("="*80 + "\n")

        try:
            while True:
                # Simulate market events
                self.simulate_market_events()

                print(f"\nChecking supertrend signals at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}...")

                # Check each instrument
                for instrument in self.instruments:
                    signal = self.check_supertrend_signals(instrument)

                    if signal:
                        self.display_signal(instrument, signal)

                # Calculate time until next check
                next_check = datetime.datetime.now() + datetime.timedelta(seconds=interval_seconds)
                print(f"Next check at {next_check.strftime('%Y-%m-%d %H:%M:%S')}")

                # Wait for next check
                time.sleep(interval_seconds)

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
    """Main function to run the Mock Supertrend Monitor"""
    # Check if instruments file exists, create sample if not
    instruments_file = "instruments.txt"
    if not os.path.exists(instruments_file):
        create_sample_instruments_file(instruments_file)
        print(f"Please edit {instruments_file} with your instruments and run the script again.")
        return

    # Initialize the Mock Supertrend Monitor
    monitor = MockSupertrendMonitor()

    try:
        # Read instruments from file
        if not monitor.read_instruments(instruments_file):
            logger.error("Failed to read instruments from file. Exiting.")
            return

        # Start monitoring
        monitor.monitor_supertrend()

    except Exception as e:
        logger.error(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
