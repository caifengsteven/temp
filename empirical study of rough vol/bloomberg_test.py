"""
Bloomberg API Test Script

This script tests the connection to Bloomberg and retrieves some basic data.
"""

try:
    import pdblp
    print("pdblp module is available")
    
    # Try to connect to Bloomberg
    try:
        con = pdblp.BCon(debug=True, port=8194)
        con.start()
        print("Successfully connected to Bloomberg")
        
        # Try to get some basic data
        ticker = 'SPX Index'
        data = con.bdh(
            tickers=ticker,
            flds=['PX_LAST'],
            start_date='20230101',
            end_date='20230131'
        )
        
        print(f"Retrieved data for {ticker}:")
        print(data.head())
        
        con.stop()
        print("Bloomberg connection closed")
        
    except Exception as e:
        print(f"Error connecting to Bloomberg: {e}")
        
except ImportError:
    print("pdblp module is not available. Please install it using:")
    print("pip install pdblp")
    
    # Alternative: try to use blpapi directly
    try:
        import blpapi
        print("blpapi module is available")
    except ImportError:
        print("blpapi module is not available either")
        print("Please install Bloomberg API using:")
        print("pip install --index-url=https://bcms.bloomberg.com/pip/simple/ blpapi")
