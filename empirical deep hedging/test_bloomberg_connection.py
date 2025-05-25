"""
Test Bloomberg Connection

This script tests the connection to Bloomberg API and fetches some basic data.
"""

import sys
import pandas as pd
from datetime import datetime, timedelta

# Try to import Bloomberg API libraries
try:
    import blpapi
    from xbbg import blp
    BLOOMBERG_AVAILABLE = True
    print("Bloomberg API available")
except ImportError:
    BLOOMBERG_AVAILABLE = False
    print("Bloomberg API not available. Please install the required packages.")
    print("pip install --index-url=https://bcms.bloomberg.com/pip/simple/ blpapi")
    print("pip install xbbg")
    sys.exit(1)

def test_bloomberg_connection():
    """Test connection to Bloomberg API"""
    try:
        # Initialize Bloomberg session
        session = blpapi.Session()
        
        # Start session
        if not session.start():
            print("Failed to start session.")
            return False
        
        print("Successfully connected to Bloomberg API")
        
        # Open market data service
        if not session.openService("//blp/mktdata"):
            print("Failed to open market data service")
            session.stop()
            return False
        
        print("Market data service opened")
        
        # Stop session
        session.stop()
        return True
    
    except Exception as e:
        print(f"Error connecting to Bloomberg API: {e}")
        return False

def fetch_basic_data():
    """Fetch some basic data from Bloomberg"""
    try:
        # Define tickers and fields
        tickers = ['SPY US Equity', 'AAPL US Equity', 'MSFT US Equity']
        fields = ['PX_LAST', 'VOLUME', 'NAME']
        
        # Get current data
        current_data = blp.bdp(tickers=tickers, flds=fields)
        print("\nCurrent Data:")
        print(current_data)
        
        # Get historical data for the last 5 days
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=5)).strftime('%Y-%m-%d')
        
        historical_data = blp.bdh(
            tickers=tickers,
            flds=['PX_LAST'],
            start_date=start_date,
            end_date=end_date
        )
        
        print("\nHistorical Data (Last 5 days):")
        print(historical_data)
        
        return True
    
    except Exception as e:
        print(f"Error fetching data from Bloomberg: {e}")
        return False

def main():
    """Main function"""
    print("Testing Bloomberg connection...")
    
    if test_bloomberg_connection():
        print("\nBloomberg connection test successful!")
        
        print("\nFetching basic data...")
        if fetch_basic_data():
            print("\nSuccessfully fetched data from Bloomberg!")
        else:
            print("\nFailed to fetch data from Bloomberg.")
    else:
        print("\nBloomberg connection test failed.")

if __name__ == "__main__":
    main()
