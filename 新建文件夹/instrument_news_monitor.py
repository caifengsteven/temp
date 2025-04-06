#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Instrument News Monitor
This script monitors news for instruments from a file and displays relevant headlines.
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
NEWS_API_SVC = "//blp/newsapi"
REFDATA_SVC = "//blp/refdata"
SESSION_STARTED = blpapi.Name("SessionStarted")
SESSION_STARTUP_FAILURE = blpapi.Name("SessionStartupFailure")
SERVICE_OPENED = blpapi.Name("ServiceOpened")
SERVICE_OPEN_FAILURE = blpapi.Name("ServiceOpenFailure")
RESPONSE_ERROR = blpapi.Name("ResponseError")


class InstrumentNewsMonitor:
    """Class to monitor news for a list of instruments"""

    def __init__(self, host: str = "localhost", port: int = 8194):
        """Initialize the Instrument News Monitor

        Args:
            host: Bloomberg server host
            port: Bloomberg server port
        """
        self.host = host
        self.port = port
        self.session = None
        self.news_api_service = None
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
        
        # Open the news API service
        if not self.session.openService(NEWS_API_SVC):
            logger.error("Failed to open news API service.")
            return False
        
        self.news_api_service = self.session.getService(NEWS_API_SVC)
        logger.info("News API service opened successfully.")
        
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

    def get_news_for_instrument(self, security: str, max_headlines: int = 5, since_time: Optional[datetime.datetime] = None) -> List[Dict]:
        """Get news headlines for a specific security since a given time

        Args:
            security: Bloomberg security identifier
            max_headlines: Maximum number of headlines to retrieve
            since_time: Only retrieve headlines after this time

        Returns:
            List[Dict]: List of news headlines
        """
        logger.info(f"Retrieving news for security: {security}")
        
        # Create a request for news by security
        request = self.news_api_service.createRequest("NewsBySecurityRequest")
        request.set("security", security)
        request.set("maxHeadlines", max_headlines)
        
        # Send the request
        self.session.sendRequest(request)
        
        # Process the response
        headlines = []
        
        while True:
            event = self.session.nextEvent(500)  # Timeout in milliseconds
            
            for msg in event:
                if msg.hasElement("headlines"):
                    headlines_element = msg.getElement("headlines")
                    
                    for i in range(headlines_element.numValues()):
                        headline = headlines_element.getValue(i)
                        
                        headline_time = headline.getElementAsDatetime("time")
                        
                        # Skip headlines before since_time
                        if since_time and headline_time <= since_time:
                            continue
                        
                        headline_data = {
                            "time": headline_time,
                            "title": headline.getElementAsString("title"),
                            "source": headline.getElementAsString("source") if headline.hasElement("source") else "",
                            "id": headline.getElementAsString("id")
                        }
                        
                        headlines.append(headline_data)
            
            if event.eventType() == blpapi.Event.RESPONSE:
                break
        
        logger.info(f"Retrieved {len(headlines)} headlines for security: {security}")
        return headlines

    def get_news_story(self, story_id: str) -> str:
        """Get the full text of a news story

        Args:
            story_id: News story ID

        Returns:
            str: Full text of the news story
        """
        logger.info(f"Retrieving news story: {story_id}")
        
        # Create a request for news story
        request = self.news_api_service.createRequest("NewsStoryRequest")
        request.set("id", story_id)
        
        # Send the request
        self.session.sendRequest(request)
        
        # Process the response
        story_text = ""
        
        while True:
            event = self.session.nextEvent(500)  # Timeout in milliseconds
            
            for msg in event:
                if msg.hasElement("story"):
                    story_element = msg.getElement("story")
                    story_text = story_element.getElementAsString("body")
            
            if event.eventType() == blpapi.Event.RESPONSE:
                break
        
        return story_text

    def highlight_keywords(self, text: str) -> str:
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
            title = headline['title']
            for keyword in self.highlight_keywords:
                if keyword.lower() in title.lower():
                    title = title.replace(keyword, f"{Fore.YELLOW}{keyword}{Style.RESET_ALL}")
            
            # Display headline
            print(f"{i}. [{timestamp}] {title}")
            print(f"   Source: {headline['source']} | ID: {headline['id']}")
        
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
        print("INSTRUMENT NEWS MONITOR")
        print("="*80)
        print(f"Monitoring news for {len(self.instruments)} instruments")
        print(f"Checking every {interval_minutes} minutes")
        print(f"{Fore.YELLOW}Keywords are highlighted{Style.RESET_ALL}")
        print("="*80 + "\n")
        
        try:
            while True:
                print(f"\nChecking news at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}...")
                
                # Check each instrument
                for instrument in self.instruments:
                    # Get news since last check
                    headlines = self.get_news_for_instrument(
                        instrument, 
                        max_headlines=max_headlines,
                        since_time=self.last_check_times[instrument]
                    )
                    
                    # Display headlines
                    if headlines:
                        self.display_headlines(instrument, headlines)
                        self.save_headlines_to_csv(instrument, headlines)
                    
                    # Update last check time
                    self.last_check_times[instrument] = datetime.datetime.now()
                
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
    """Main function to run the Instrument News Monitor"""
    # Check if instruments file exists, create sample if not
    instruments_file = "instruments.txt"
    if not os.path.exists(instruments_file):
        create_sample_instruments_file(instruments_file)
        print(f"Please edit {instruments_file} with your instruments and run the script again.")
        return
    
    # Initialize the Instrument News Monitor
    monitor = InstrumentNewsMonitor()
    
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
