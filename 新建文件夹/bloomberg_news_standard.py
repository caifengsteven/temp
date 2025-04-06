#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Bloomberg News Monitor (Standard Version)
This script monitors news for instruments from a file using standard Bloomberg services.
It can be used alongside the supertrend_monitor.py script for comprehensive market analysis.
"""

import blpapi
import datetime
import logging
import os
import time
import pandas as pd
import colorama
from colorama import Fore, Style
from typing import List, Dict, Optional

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
SESSION_STARTED = blpapi.Name("SessionStarted")
SESSION_STARTUP_FAILURE = blpapi.Name("SessionStartupFailure")
SERVICE_OPENED = blpapi.Name("ServiceOpened")
SERVICE_OPEN_FAILURE = blpapi.Name("ServiceOpenFailure")
RESPONSE_ERROR = blpapi.Name("ResponseError")


class BloombergNewsMonitor:
    """Class to monitor news for a list of instruments using standard Bloomberg services"""

    def __init__(self, host: str = "localhost", port: int = 8194):
        """Initialize the Bloomberg News Monitor

        Args:
            host: Bloomberg server host
            port: Bloomberg server port
        """
        self.host = host
        self.port = port
        self.session = None
        self.refdata_service = None
        
        # Instruments list
        self.instruments = []
        
        # Last news check time for each instrument
        self.last_check_times = {}
        
        # Keywords to highlight in news headlines
        self.highlight_keywords = [
            "upgrade", "downgrade", "beat", "miss", "raise", "cut", "dividend",
            "guidance", "outlook", "earnings", "profit", "loss", "revenue",
            "acquisition", "merger", "takeover", "buyout", "spinoff", "split",
            "lawsuit", "settlement", "investigation", "recall", "approval",
            "launch", "release", "patent", "FDA", "SEC", "CEO", "executive",
            "resign", "appoint", "layoff", "restructure", "bankruptcy"
        ]

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
            
            # Initialize last check times
            for instrument in self.instruments:
                self.last_check_times[instrument] = datetime.datetime.now() - datetime.timedelta(hours=24)
            
            # Check if any instruments were read
            if not self.instruments:
                logger.error("No instruments found in the file.")
                return False
            
            return True
        
        except Exception as e:
            logger.error(f"Error reading instruments file: {e}")
            return False

    def get_news_headlines(self, securities: List[str], max_headlines: int = 10) -> Dict[str, List[Dict]]:
        """Get news headlines for a list of securities

        Args:
            securities: List of Bloomberg security identifiers
            max_headlines: Maximum number of headlines to retrieve per security

        Returns:
            Dict[str, List[Dict]]: Dictionary mapping securities to their headlines
        """
        logger.info(f"Retrieving news for {len(securities)} securities")
        
        # Create a request for historical data
        request = self.refdata_service.createRequest("HistoricalDataRequest")
        
        # Add securities
        for security in securities:
            request.append("securities", security)
        
        # Add fields
        request.append("fields", "NEWS_HEADLINES")
        
        # Set date range (last 24 hours)
        end_date = datetime.datetime.now()
        start_date = end_date - datetime.timedelta(hours=24)
        request.set("startDate", start_date.strftime("%Y%m%d"))
        request.set("endDate", end_date.strftime("%Y%m%d"))
        
        # Set maximum data points
        request.set("maxDataPoints", max_headlines)
        
        # Send the request
        self.session.sendRequest(request)
        
        # Process the response
        headlines_by_security = {security: [] for security in securities}
        
        while True:
            event = self.session.nextEvent(500)  # Timeout in milliseconds
            
            for msg in event:
                if msg.hasElement("securityData"):
                    security_data_array = msg.getElement("securityData")
                    
                    for i in range(security_data_array.numValues()):
                        security_data = security_data_array.getValue(i)
                        security = security_data.getElementAsString("security")
                        
                        if security_data.hasElement("fieldData"):
                            field_data = security_data.getElement("fieldData")
                            
                            if field_data.hasElement("NEWS_HEADLINES"):
                                headlines_element = field_data.getElement("NEWS_HEADLINES")
                                
                                for j in range(headlines_element.numValues()):
                                    headline = headlines_element.getValue(j)
                                    
                                    # Extract headline data
                                    headline_data = {
                                        "time": headline.getElementAsDatetime("time") if headline.hasElement("time") else datetime.datetime.now(),
                                        "title": headline.getElementAsString("headline") if headline.hasElement("headline") else "",
                                        "source": headline.getElementAsString("source") if headline.hasElement("source") else "",
                                        "id": headline.getElementAsString("id") if headline.hasElement("id") else ""
                                    }
                                    
                                    # Add to headlines list for this security
                                    headlines_by_security[security].append(headline_data)
            
            if event.eventType() == blpapi.Event.RESPONSE:
                break
        
        # Filter headlines by last check time
        for security in securities:
            headlines_by_security[security] = [
                h for h in headlines_by_security[security] 
                if h["time"] > self.last_check_times[security]
            ]
            
            logger.info(f"Retrieved {len(headlines_by_security[security])} new headlines for {security}")
        
        return headlines_by_security

    def get_news_content(self, securities: List[str], max_stories: int = 5) -> Dict[str, List[Dict]]:
        """Get news content for a list of securities

        Args:
            securities: List of Bloomberg security identifiers
            max_stories: Maximum number of stories to retrieve per security

        Returns:
            Dict[str, List[Dict]]: Dictionary mapping securities to their news content
        """
        logger.info(f"Retrieving news content for {len(securities)} securities")
        
        # Create a request for reference data
        request = self.refdata_service.createRequest("ReferenceDataRequest")
        
        # Add securities
        for security in securities:
            request.append("securities", security)
        
        # Add fields
        request.append("fields", "NEWS_STORY_TEXT")
        
        # Set override for max stories
        override = request.getElement("overrides").appendElement()
        override.setElement("fieldId", "NEWS_STORY_COUNT")
        override.setElement("value", max_stories)
        
        # Send the request
        self.session.sendRequest(request)
        
        # Process the response
        stories_by_security = {security: [] for security in securities}
        
        while True:
            event = self.session.nextEvent(500)  # Timeout in milliseconds
            
            for msg in event:
                if msg.hasElement("securityData"):
                    security_data_array = msg.getElement("securityData")
                    
                    for i in range(security_data_array.numValues()):
                        security_data = security_data_array.getValue(i)
                        security = security_data.getElementAsString("security")
                        
                        if security_data.hasElement("fieldData"):
                            field_data = security_data.getElement("fieldData")
                            
                            if field_data.hasElement("NEWS_STORY_TEXT"):
                                stories_element = field_data.getElement("NEWS_STORY_TEXT")
                                
                                for j in range(stories_element.numValues()):
                                    story = stories_element.getValue(j)
                                    
                                    # Extract story data
                                    story_data = {
                                        "time": story.getElementAsDatetime("time") if story.hasElement("time") else datetime.datetime.now(),
                                        "headline": story.getElementAsString("headline") if story.hasElement("headline") else "",
                                        "source": story.getElementAsString("source") if story.hasElement("source") else "",
                                        "text": story.getElementAsString("text") if story.hasElement("text") else ""
                                    }
                                    
                                    # Add to stories list for this security
                                    stories_by_security[security].append(story_data)
            
            if event.eventType() == blpapi.Event.RESPONSE:
                break
        
        return stories_by_security

    def highlight_text(self, text: str) -> str:
        """Highlight keywords in text

        Args:
            text: Text to highlight

        Returns:
            str: Text with highlighted keywords
        """
        highlighted_text = text
        
        for keyword in self.highlight_keywords:
            # Case-insensitive replacement
            keyword_lower = keyword.lower()
            text_lower = highlighted_text.lower()
            
            start_idx = 0
            while keyword_lower in text_lower[start_idx:]:
                pos = text_lower.find(keyword_lower, start_idx)
                original_keyword = highlighted_text[pos:pos+len(keyword)]
                highlighted_text = highlighted_text[:pos] + f"{Fore.YELLOW}{original_keyword}{Style.RESET_ALL}" + highlighted_text[pos+len(keyword):]
                
                # Update indices for next search
                start_idx = pos + len(keyword) + len(f"{Fore.YELLOW}{Style.RESET_ALL}")
                text_lower = highlighted_text.lower()
        
        return highlighted_text

    def display_headlines(self, security: str, headlines: List[Dict]) -> None:
        """Display news headlines for a security

        Args:
            security: Bloomberg security identifier
            headlines: List of news headlines
        """
        if not headlines:
            return
        
        print(f"\n{Fore.CYAN}Latest news for {security}:{Style.RESET_ALL}")
        
        for i, headline in enumerate(headlines, 1):
            # Format timestamp
            timestamp = headline['time'].strftime("%Y-%m-%d %H:%M:%S")
            
            # Highlight keywords in title
            title = self.highlight_text(headline['title'])
            
            # Display headline
            print(f"{i}. [{timestamp}] {title}")
            print(f"   Source: {headline['source']}")
        
        print("")

    def save_headlines_to_csv(self, security: str, headlines: List[Dict]) -> None:
        """Save news headlines to a CSV file

        Args:
            security: Bloomberg security identifier
            headlines: List of news headlines
        """
        if not headlines:
            return
        
        # Create output directory if it doesn't exist
        output_dir = "bloomberg_news"
        os.makedirs(output_dir, exist_ok=True)
        
        # Create filename
        filename = os.path.join(output_dir, f"{security.replace(' ', '_')}_news.csv")
        
        # Convert to DataFrame and save
        df = pd.DataFrame(headlines)
        
        # Check if file exists
        if os.path.exists(filename):
            # Append to existing file
            existing_df = pd.read_csv(filename)
            combined_df = pd.concat([existing_df, df], ignore_index=True)
            # Remove duplicates based on title and time
            combined_df = combined_df.drop_duplicates(subset=['title', 'time'])
            combined_df.to_csv(filename, index=False)
        else:
            # Create new file
            df.to_csv(filename, index=False)
        
        logger.info(f"Saved {len(headlines)} headlines for {security} to {filename}")

    def monitor_news(self, interval_minutes: int = 30, max_headlines: int = 5) -> None:
        """Monitor news for instruments at regular intervals

        Args:
            interval_minutes: Interval between checks in minutes
            max_headlines: Maximum number of headlines to retrieve per instrument
        """
        logger.info(f"Starting news monitoring for {len(self.instruments)} instruments")
        logger.info(f"Interval: {interval_minutes} minutes")
        
        print("\n" + "="*80)
        print("BLOOMBERG NEWS MONITOR (STANDARD VERSION)")
        print("="*80)
        print(f"Monitoring news for {len(self.instruments)} instruments")
        print(f"Checking every {interval_minutes} minutes")
        print(f"{Fore.YELLOW}Keywords are highlighted{Style.RESET_ALL}")
        print("="*80 + "\n")
        
        try:
            while True:
                print(f"\nChecking news at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}...")
                
                # Process instruments in batches of 10 to avoid overloading the API
                batch_size = 10
                for i in range(0, len(self.instruments), batch_size):
                    batch = self.instruments[i:i+batch_size]
                    
                    # Get headlines for this batch
                    headlines_by_security = self.get_news_headlines(batch, max_headlines)
                    
                    # Display and save headlines
                    for security in batch:
                        headlines = headlines_by_security.get(security, [])
                        
                        if headlines:
                            self.display_headlines(security, headlines)
                            self.save_headlines_to_csv(security, headlines)
                        
                        # Update last check time
                        self.last_check_times[security] = datetime.datetime.now()
                
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
    """Main function to run the Bloomberg News Monitor"""
    # Check if instruments file exists, create sample if not
    instruments_file = "instruments.txt"
    if not os.path.exists(instruments_file):
        create_sample_instruments_file(instruments_file)
        print(f"Please edit {instruments_file} with your instruments and run the script again.")
        return
    
    # Initialize the Bloomberg News Monitor
    monitor = BloombergNewsMonitor()
    
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
        monitor.monitor_news()
    
    except Exception as e:
        logger.error(f"An error occurred: {e}")
    
    finally:
        # Stop the Bloomberg session
        monitor.stop_session()


if __name__ == "__main__":
    main()
