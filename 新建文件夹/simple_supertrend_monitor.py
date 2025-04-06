#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Simple Supertrend Indicator Monitor
A simplified version of the Supertrend Monitor for testing.
"""

import pandas as pd
import numpy as np
import time
import datetime
import os
import logging
import random
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


def read_instruments(file_path):
    """Read instruments from a file"""
    try:
        # Check if file exists
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return []
        
        # Read file
        with open(file_path, 'r') as f:
            # Strip whitespace and filter out empty lines
            instruments = [line.strip() for line in f if line.strip()]
        
        logger.info(f"Read {len(instruments)} instruments from {file_path}")
        return instruments
    
    except Exception as e:
        logger.error(f"Error reading instruments file: {e}")
        return []


def generate_mock_data(security, bars=50):
    """Generate mock price data for a security"""
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


def calculate_atr(df, period=10):
    """Calculate Average True Range (ATR)"""
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


def calculate_supertrend(df, atr_period=10, multiplier=3.0):
    """Calculate Supertrend indicator"""
    # Make a copy of the dataframe
    df = df.copy()
    
    # Calculate ATR
    df['atr'] = calculate_atr(df, atr_period)
    
    # Calculate basic upper and lower bands
    df['basic_upper'] = (df['high'] + df['low']) / 2 + multiplier * df['atr']
    df['basic_lower'] = (df['high'] + df['low']) / 2 - multiplier * df['atr']
    
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


def main():
    """Main function"""
    # Read instruments from file
    instruments = read_instruments("instruments.txt")
    
    if not instruments:
        print("No instruments found. Exiting.")
        return
    
    # Store previous states
    prev_states = {instrument: None for instrument in instruments}
    
    print("\n" + "="*80)
    print("SIMPLE SUPERTREND INDICATOR MONITOR")
    print("="*80)
    print(f"{Fore.GREEN}GREEN: Breakup (Price crosses above Supertrend){Style.RESET_ALL}")
    print(f"{Fore.RED}RED: Breakdown (Price crosses below Supertrend){Style.RESET_ALL}")
    print("="*80 + "\n")
    
    try:
        # Run for 5 iterations
        for iteration in range(5):
            print(f"\nIteration {iteration+1} at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}...")
            
            # Check each instrument
            for instrument in instruments:
                # Generate mock data
                df = generate_mock_data(instrument)
                
                # Calculate Supertrend
                df = calculate_supertrend(df)
                
                # Get current direction
                current_direction = df['supertrend_direction'].iloc[-1]
                
                # Check for direction change
                if prev_states[instrument] is not None and current_direction != prev_states[instrument]:
                    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    
                    if current_direction == 1:
                        # Breakup
                        print(f"{Fore.GREEN}{Back.WHITE}{timestamp} | {instrument}: BREAK UP{Style.RESET_ALL}")
                    else:
                        # Breakdown
                        print(f"{Fore.RED}{Back.WHITE}{timestamp} | {instrument}: BREAK DOWN{Style.RESET_ALL}")
                
                # Update previous state
                prev_states[instrument] = current_direction
            
            # Wait between iterations
            if iteration < 4:  # Don't wait after the last iteration
                print(f"Waiting 2 seconds...")
                time.sleep(2)
        
        print("\nMonitoring completed.")
    
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        print(f"\n{Fore.RED}Error: {e}{Style.RESET_ALL}")


if __name__ == "__main__":
    main()
