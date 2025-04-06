#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Mock ETF Premium/Discount Monitor
This script simulates the ETF Premium Monitor without requiring a Bloomberg connection.
It reads a CSV file with ETF and INAV tickers and simulates real-time price checks.
"""

import pandas as pd
import time
import datetime
import os
import logging
import csv
import sys
import random
from typing import List, Tuple
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


class MockETFPremiumMonitor:
    """Class to simulate ETF premium/discount monitoring"""

    def __init__(self):
        """Initialize the Mock ETF Premium Monitor"""
        # Premium/discount threshold in basis points
        self.threshold_bps = 20
        
        # Flash control
        self.flash_state = False
        
        # ETF pairs
        self.etf_pairs = []
        
        # Simulated prices
        self.simulated_prices = {}

    def read_etf_pairs_from_csv(self, file_path: str) -> bool:
        """Read ETF and INAV ticker pairs from a CSV file

        Args:
            file_path: Path to the CSV file

        Returns:
            bool: True if file was read successfully, False otherwise
        """
        try:
            # Check if file exists
            if not os.path.exists(file_path):
                logger.error(f"File not found: {file_path}")
                return False
            
            # Read CSV file
            with open(file_path, 'r') as f:
                reader = csv.reader(f)
                # Skip header row
                next(reader, None)
                
                # Read ETF pairs
                self.etf_pairs = []
                for row in reader:
                    if len(row) >= 2:
                        etf_ticker = row[0].strip()
                        inav_ticker = row[1].strip()
                        self.etf_pairs.append((etf_ticker, inav_ticker))
                        
                        # Initialize simulated prices
                        base_price = random.uniform(10, 100)
                        self.simulated_prices[etf_ticker] = base_price
                        self.simulated_prices[inav_ticker] = base_price
            
            logger.info(f"Read {len(self.etf_pairs)} ETF pairs from {file_path}")
            
            # Check if any pairs were read
            if not self.etf_pairs:
                logger.error("No ETF pairs found in the CSV file.")
                return False
            
            return True
        
        except Exception as e:
            logger.error(f"Error reading CSV file: {e}")
            return False

    def get_simulated_price(self, security: str) -> float:
        """Get simulated price for a security

        Args:
            security: Security identifier

        Returns:
            float: Simulated price
        """
        # If security not in simulated prices, add it
        if security not in self.simulated_prices:
            self.simulated_prices[security] = random.uniform(10, 100)
        
        # Get current price
        current_price = self.simulated_prices[security]
        
        # Add random variation (up to ±0.5%)
        variation = random.uniform(-0.005, 0.005)
        new_price = current_price * (1 + variation)
        
        # Update simulated price
        self.simulated_prices[security] = new_price
        
        return new_price

    def calculate_premium_discount(self, etf_price: float, inav_price: float) -> float:
        """Calculate premium/discount in basis points

        Args:
            etf_price: ETF price
            inav_price: INAV price

        Returns:
            float: Premium/discount in basis points (positive for premium, negative for discount)
        """
        if etf_price is None or inav_price is None or inav_price == 0:
            return None
        
        # Calculate premium/discount in basis points
        premium_discount_bps = (etf_price / inav_price - 1) * 10000
        
        return premium_discount_bps

    def display_premium_discount(self, etf_ticker: str, inav_ticker: str, premium_discount_bps: float) -> None:
        """Display premium/discount in the terminal with color coding

        Args:
            etf_ticker: ETF ticker
            inav_ticker: INAV ticker
            premium_discount_bps: Premium/discount in basis points
        """
        if premium_discount_bps is None:
            print(f"{Fore.YELLOW}{etf_ticker} vs {inav_ticker}: No data{Style.RESET_ALL}")
            return
        
        # Format premium/discount
        if premium_discount_bps > 0:
            premium_discount_str = f"+{premium_discount_bps:.2f} bps (Premium)"
        else:
            premium_discount_str = f"{premium_discount_bps:.2f} bps (Discount)"
        
        # Determine color and flashing based on premium/discount
        if premium_discount_bps < -self.threshold_bps:
            # Discount > threshold (GREEN, flashing)
            color = Fore.GREEN
            flash = True
        elif premium_discount_bps > self.threshold_bps:
            # Premium > threshold (RED, flashing)
            color = Fore.RED
            flash = True
        else:
            # Within threshold (BLUE, no flashing)
            color = Fore.BLUE
            flash = False
        
        # Toggle flash state
        self.flash_state = not self.flash_state
        
        # Apply flashing effect
        if flash and self.flash_state:
            # Highlight with background color for flashing effect
            bg_color = Back.WHITE
        else:
            bg_color = Back.BLACK
        
        # Display in terminal
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"{color}{bg_color}{timestamp} | {etf_ticker} vs {inav_ticker}: {premium_discount_str}{Style.RESET_ALL}")

    def simulate_market_events(self) -> None:
        """Simulate market events that cause premiums/discounts"""
        # Randomly select an ETF pair to create a premium or discount
        if self.etf_pairs and random.random() < 0.3:  # 30% chance of an event
            etf_ticker, inav_ticker = random.choice(self.etf_pairs)
            
            # Decide between premium, discount, or normal
            event_type = random.choice(["premium", "discount", "normal"])
            
            if event_type == "premium":
                # Create a premium (ETF price > INAV)
                premium_bps = random.uniform(self.threshold_bps, self.threshold_bps * 3)
                self.simulated_prices[etf_ticker] = self.simulated_prices[inav_ticker] * (1 + premium_bps / 10000)
                logger.debug(f"Simulated premium event for {etf_ticker}: +{premium_bps:.2f} bps")
            
            elif event_type == "discount":
                # Create a discount (ETF price < INAV)
                discount_bps = random.uniform(self.threshold_bps, self.threshold_bps * 3)
                self.simulated_prices[etf_ticker] = self.simulated_prices[inav_ticker] * (1 - discount_bps / 10000)
                logger.debug(f"Simulated discount event for {etf_ticker}: -{discount_bps:.2f} bps")
            
            else:
                # Reset to normal (ETF price ≈ INAV)
                self.simulated_prices[etf_ticker] = self.simulated_prices[inav_ticker] * (1 + random.uniform(-0.001, 0.001))
                logger.debug(f"Simulated normal event for {etf_ticker}")

    def monitor_etf_premiums(self, interval_seconds: int = 5) -> None:
        """Monitor ETF premiums/discounts at regular intervals

        Args:
            interval_seconds: Interval between checks in seconds
        """
        logger.info(f"Starting mock ETF premium/discount monitoring with {len(self.etf_pairs)} pairs")
        logger.info(f"Threshold: {self.threshold_bps} bps")
        logger.info(f"Interval: {interval_seconds} seconds")
        
        print("\n" + "="*80)
        print(f"MOCK ETF PREMIUM/DISCOUNT MONITOR (Threshold: {self.threshold_bps} bps)")
        print("="*80)
        print(f"{Fore.GREEN}GREEN (flashing): ETF at discount > {self.threshold_bps} bps{Style.RESET_ALL}")
        print(f"{Fore.RED}RED (flashing): ETF at premium > {self.threshold_bps} bps{Style.RESET_ALL}")
        print(f"{Fore.BLUE}BLUE: ETF within {self.threshold_bps} bps of INAV{Style.RESET_ALL}")
        print("="*80 + "\n")
        
        try:
            while True:
                # Simulate market events
                self.simulate_market_events()
                
                # Clear terminal (Windows)
                if os.name == 'nt':
                    os.system('cls')
                # Clear terminal (Unix/Linux/MacOS)
                else:
                    os.system('clear')
                
                print("\n" + "="*80)
                print(f"MOCK ETF PREMIUM/DISCOUNT MONITOR (Threshold: {self.threshold_bps} bps)")
                print(f"Last Update: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                print("="*80)
                print(f"{Fore.GREEN}GREEN (flashing): ETF at discount > {self.threshold_bps} bps{Style.RESET_ALL}")
                print(f"{Fore.RED}RED (flashing): ETF at premium > {self.threshold_bps} bps{Style.RESET_ALL}")
                print(f"{Fore.BLUE}BLUE: ETF within {self.threshold_bps} bps of INAV{Style.RESET_ALL}")
                print("="*80 + "\n")
                
                # Check each ETF pair
                for etf_ticker, inav_ticker in self.etf_pairs:
                    # Get simulated prices
                    etf_price = self.get_simulated_price(etf_ticker)
                    inav_price = self.get_simulated_price(inav_ticker)
                    
                    # Calculate premium/discount
                    premium_discount_bps = self.calculate_premium_discount(etf_price, inav_price)
                    
                    # Display in terminal
                    self.display_premium_discount(etf_ticker, inav_ticker, premium_discount_bps)
                
                # Wait for next check
                time.sleep(interval_seconds)
        
        except KeyboardInterrupt:
            logger.info("Monitoring stopped by user")
            print("\nMonitoring stopped. Press Enter to exit...")
        
        except Exception as e:
            logger.error(f"Error during monitoring: {e}")
            print(f"\n{Fore.RED}Error during monitoring: {e}{Style.RESET_ALL}")


def create_sample_csv(file_path: str) -> None:
    """Create a sample CSV file with ETF and INAV tickers

    Args:
        file_path: Path to the CSV file
    """
    with open(file_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["ETF Ticker", "INAV Ticker"])
        writer.writerow(["2823 HK Equity", "2823IV Index"])
        writer.writerow(["3067 HK Equity", "3067IV Index"])
        writer.writerow(["9834 HK Equity", "9834IV Index"])
    
    print(f"Sample CSV file created: {file_path}")


def main():
    """Main function to run the Mock ETF Premium Monitor"""
    # Check if CSV file exists, create sample if not
    csv_file = "etf_pairs.csv"
    if not os.path.exists(csv_file):
        create_sample_csv(csv_file)
        print(f"Please edit {csv_file} with your ETF pairs and run the script again.")
        return
    
    # Initialize the Mock ETF Premium Monitor
    monitor = MockETFPremiumMonitor()
    
    try:
        # Read ETF pairs from CSV
        if not monitor.read_etf_pairs_from_csv(csv_file):
            logger.error("Failed to read ETF pairs from CSV. Exiting.")
            return
        
        # Start monitoring
        monitor.monitor_etf_premiums()
    
    except Exception as e:
        logger.error(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
