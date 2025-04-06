#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ETF Premium/Discount Monitor
This script reads a CSV file with ETF and INAV tickers, checks their real-time prices from Bloomberg,
and displays the premium/discount in the terminal with color coding.
"""

import blpapi
import pandas as pd
import time
import datetime
import os
import logging
import csv
import sys
from typing import List, Tuple, Dict
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
MARKET_DATA_SVC = "//blp/mktdata"
REFDATA_SVC = "//blp/refdata"
SESSION_STARTED = blpapi.Name("SessionStarted")
SESSION_STARTUP_FAILURE = blpapi.Name("SessionStartupFailure")
SERVICE_OPENED = blpapi.Name("ServiceOpened")
SERVICE_OPEN_FAILURE = blpapi.Name("ServiceOpenFailure")
RESPONSE_ERROR = blpapi.Name("ResponseError")


class ETFPremiumMonitor:
    """Class to monitor ETF premium/discount to INAV"""

    def __init__(self, host: str = "localhost", port: int = 8194):
        """Initialize the ETF Premium Monitor

        Args:
            host: Bloomberg server host
            port: Bloomberg server port
        """
        self.host = host
        self.port = port
        self.session = None
        self.market_data_service = None
        self.refdata_service = None
        
        # Premium/discount threshold in basis points
        self.threshold_bps = 20
        
        # Flash control
        self.flash_state = False
        
        # ETF pairs
        self.etf_pairs = []

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
        
        # Open the market data service
        if not self.session.openService(MARKET_DATA_SVC):
            logger.error("Failed to open market data service.")
            return False
        
        self.market_data_service = self.session.getService(MARKET_DATA_SVC)
        logger.info("Market data service opened successfully.")
        
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
            
            logger.info(f"Read {len(self.etf_pairs)} ETF pairs from {file_path}")
            
            # Check if any pairs were read
            if not self.etf_pairs:
                logger.error("No ETF pairs found in the CSV file.")
                return False
            
            return True
        
        except Exception as e:
            logger.error(f"Error reading CSV file: {e}")
            return False

    def get_real_time_price(self, security: str) -> float:
        """Get real-time price for a security from Bloomberg

        Args:
            security: Bloomberg security identifier

        Returns:
            float: Current price of the security
        """
        try:
            # Create a request for reference data
            request = self.refdata_service.createRequest("ReferenceDataRequest")
            request.append("securities", security)
            request.append("fields", "LAST_PRICE")
            
            # Send the request
            self.session.sendRequest(request)
            
            # Process the response
            price = None
            
            while True:
                event = self.session.nextEvent(500)  # Timeout in milliseconds
                
                for msg in event:
                    if msg.hasElement("securityData"):
                        security_data = msg.getElement("securityData")
                        if security_data.hasElement("fieldData"):
                            field_data = security_data.getElement("fieldData")
                            if field_data.hasElement("LAST_PRICE"):
                                price = field_data.getElementAsFloat("LAST_PRICE")
                
                if event.eventType() == blpapi.Event.RESPONSE:
                    break
            
            if price is not None:
                logger.debug(f"Price for {security}: {price}")
                return price
            else:
                logger.warning(f"Could not retrieve price for {security}")
                return None
        
        except Exception as e:
            logger.error(f"Error getting price for {security}: {e}")
            return None

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

    def monitor_etf_premiums(self, interval_seconds: int = 60) -> None:
        """Monitor ETF premiums/discounts at regular intervals

        Args:
            interval_seconds: Interval between checks in seconds
        """
        logger.info(f"Starting ETF premium/discount monitoring with {len(self.etf_pairs)} pairs")
        logger.info(f"Threshold: {self.threshold_bps} bps")
        logger.info(f"Interval: {interval_seconds} seconds")
        
        print("\n" + "="*80)
        print(f"ETF PREMIUM/DISCOUNT MONITOR (Threshold: {self.threshold_bps} bps)")
        print("="*80)
        print(f"{Fore.GREEN}GREEN (flashing): ETF at discount > {self.threshold_bps} bps{Style.RESET_ALL}")
        print(f"{Fore.RED}RED (flashing): ETF at premium > {self.threshold_bps} bps{Style.RESET_ALL}")
        print(f"{Fore.BLUE}BLUE: ETF within {self.threshold_bps} bps of INAV{Style.RESET_ALL}")
        print("="*80 + "\n")
        
        try:
            while True:
                # Clear terminal (Windows)
                if os.name == 'nt':
                    os.system('cls')
                # Clear terminal (Unix/Linux/MacOS)
                else:
                    os.system('clear')
                
                print("\n" + "="*80)
                print(f"ETF PREMIUM/DISCOUNT MONITOR (Threshold: {self.threshold_bps} bps)")
                print(f"Last Update: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                print("="*80)
                print(f"{Fore.GREEN}GREEN (flashing): ETF at discount > {self.threshold_bps} bps{Style.RESET_ALL}")
                print(f"{Fore.RED}RED (flashing): ETF at premium > {self.threshold_bps} bps{Style.RESET_ALL}")
                print(f"{Fore.BLUE}BLUE: ETF within {self.threshold_bps} bps of INAV{Style.RESET_ALL}")
                print("="*80 + "\n")
                
                # Check each ETF pair
                for etf_ticker, inav_ticker in self.etf_pairs:
                    # Get real-time prices
                    etf_price = self.get_real_time_price(etf_ticker)
                    inav_price = self.get_real_time_price(inav_ticker)
                    
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
    """Main function to run the ETF Premium Monitor"""
    # Check if CSV file exists, create sample if not
    csv_file = "etf_pairs.csv"
    if not os.path.exists(csv_file):
        create_sample_csv(csv_file)
        print(f"Please edit {csv_file} with your ETF pairs and run the script again.")
        return
    
    # Initialize the ETF Premium Monitor
    monitor = ETFPremiumMonitor()
    
    try:
        # Start the Bloomberg session
        if not monitor.start_session():
            logger.error("Failed to initialize Bloomberg session. Exiting.")
            return
        
        # Read ETF pairs from CSV
        if not monitor.read_etf_pairs_from_csv(csv_file):
            logger.error("Failed to read ETF pairs from CSV. Exiting.")
            return
        
        # Start monitoring
        monitor.monitor_etf_premiums()
    
    except Exception as e:
        logger.error(f"An error occurred: {e}")
    
    finally:
        # Stop the Bloomberg session
        monitor.stop_session()


if __name__ == "__main__":
    main()
