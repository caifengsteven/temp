#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Get Latest Bloomberg News
A simple script to retrieve the latest news for instruments from a file.
"""

import blpapi
import datetime
import logging
import os
import pandas as pd
import colorama
from colorama import Fore, Style
from typing import List, Dict

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


def start_bloomberg_session(host="localhost", port=8194):
    """Start a Bloomberg API session"""
    logger.info("Starting Bloomberg API session...")
    
    # Initialize session options
    session_options = blpapi.SessionOptions()
    session_options.setServerHost(host)
    session_options.setServerPort(port)
    
    # Create a session
    session = blpapi.Session(session_options)
    
    # Start the session
    if not session.start():
        logger.error("Failed to start session.")
        return None
    
    logger.info("Session started successfully.")
    
    # Open the reference data service
    if not session.openService(REFDATA_SVC):
        logger.error("Failed to open reference data service.")
        session.stop()
        return None
    
    logger.info("Reference data service opened successfully.")
    
    return session


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


def get_latest_news(session, securities, max_headlines=10):
    """Get latest news for a list of securities"""
    logger.info(f"Retrieving news for {len(securities)} securities")
    
    # Get reference data service
    refdata_service = session.getService(REFDATA_SVC)
    
    # Create a request for reference data
    request = refdata_service.createRequest("ReferenceDataRequest")
    
    # Add securities
    for security in securities:
        request.append("securities", security)
    
    # Add fields
    request.append("fields", "NEWS_HEADLINES")
    
    # Set override for max headlines
    override = request.getElement("overrides").appendElement()
    override.setElement("fieldId", "NEWS_STORY_COUNT")
    override.setElement("value", max_headlines)
    
    # Send the request
    session.sendRequest(request)
    
    # Process the response
    headlines_by_security = {security: [] for security in securities}
    
    while True:
        event = session.nextEvent(500)  # Timeout in milliseconds
        
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
                                    "headline": headline.getElementAsString("headline") if headline.hasElement("headline") else "",
                                    "source": headline.getElementAsString("source") if headline.hasElement("source") else ""
                                }
                                
                                # Add to headlines list for this security
                                headlines_by_security[security].append(headline_data)
        
        if event.eventType() == blpapi.Event.RESPONSE:
            break
    
    return headlines_by_security


def highlight_keywords(text, keywords):
    """Highlight keywords in text"""
    highlighted_text = text
    
    for keyword in keywords:
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


def display_headlines(security, headlines, keywords):
    """Display news headlines for a security"""
    if not headlines:
        print(f"{Fore.CYAN}{security}: {Style.RESET_ALL}No recent news")
        return
    
    print(f"\n{Fore.CYAN}Latest news for {security}:{Style.RESET_ALL}")
    
    for i, headline in enumerate(headlines, 1):
        # Format timestamp
        timestamp = headline['time'].strftime("%Y-%m-%d %H:%M:%S")
        
        # Highlight keywords in headline
        title = highlight_keywords(headline['headline'], keywords)
        
        # Display headline
        print(f"{i}. [{timestamp}] {title}")
        print(f"   Source: {headline['source']}")
    
    print("")


def save_headlines_to_csv(security, headlines):
    """Save news headlines to a CSV file"""
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


def create_sample_instruments_file(file_path):
    """Create a sample instruments file"""
    with open(file_path, 'w') as f:
        f.write("AAPL US Equity\n")
        f.write("MSFT US Equity\n")
        f.write("AMZN US Equity\n")
        f.write("GOOGL US Equity\n")
        f.write("META US Equity\n")
    
    print(f"Sample instruments file created: {file_path}")


def main():
    """Main function"""
    # Important keywords to highlight
    keywords = [
        "upgrade", "downgrade", "beat", "miss", "raise", "cut", "dividend",
        "guidance", "outlook", "earnings", "profit", "loss", "revenue",
        "acquisition", "merger", "takeover", "buyout", "spinoff", "split",
        "lawsuit", "settlement", "investigation", "recall", "approval",
        "launch", "release", "patent", "FDA", "SEC", "CEO", "executive",
        "resign", "appoint", "layoff", "restructure", "bankruptcy"
    ]
    
    # Check if instruments file exists, create sample if not
    instruments_file = "instruments.txt"
    if not os.path.exists(instruments_file):
        create_sample_instruments_file(instruments_file)
        print(f"Please edit {instruments_file} with your instruments and run the script again.")
        return
    
    # Read instruments from file
    instruments = read_instruments(instruments_file)
    if not instruments:
        print("No instruments found. Exiting.")
        return
    
    # Start Bloomberg session
    session = start_bloomberg_session()
    if not session:
        print("Failed to start Bloomberg session. Exiting.")
        return
    
    try:
        print("\n" + "="*80)
        print("LATEST BLOOMBERG NEWS")
        print("="*80)
        print(f"Retrieving news for {len(instruments)} instruments")
        print(f"{Fore.YELLOW}Keywords are highlighted{Style.RESET_ALL}")
        print("="*80 + "\n")
        
        # Process instruments in batches of 10 to avoid overloading the API
        batch_size = 10
        all_headlines = {}
        
        for i in range(0, len(instruments), batch_size):
            batch = instruments[i:i+batch_size]
            print(f"Processing batch {i//batch_size + 1} of {(len(instruments) + batch_size - 1) // batch_size}...")
            
            # Get headlines for this batch
            headlines_by_security = get_latest_news(session, batch)
            all_headlines.update(headlines_by_security)
        
        # Display and save headlines
        for security in instruments:
            headlines = all_headlines.get(security, [])
            display_headlines(security, headlines, keywords)
            save_headlines_to_csv(security, headlines)
        
        print("\nDone! News headlines have been saved to the 'bloomberg_news' directory.")
    
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        print(f"\n{Fore.RED}Error: {e}{Style.RESET_ALL}")
    
    finally:
        # Stop the Bloomberg session
        session.stop()
        logger.info("Session stopped.")


if __name__ == "__main__":
    main()
