#!/usr/bin/env python3
"""
Simple Bloomberg test to diagnose connection issues
"""

import time
from xbbg import blp

def test_multiple_tickers():
    """Test multiple tickers to see what works"""
    
    test_cases = [
        # Basic equity tickers
        'AAPL US Equity',
        'MSFT US Equity', 
        'GOOGL US Equity',
        'SPY US Equity',
        
        # Index tickers
        'SPX Index',
        'NDX Index',
        'VIX Index',
        
        # Different formats
        'ES1 Index',  # S&P 500 futures
        'NQ1 Index',  # Nasdaq futures
    ]
    
    print("Testing various Bloomberg tickers...")
    print("=" * 50)
    
    for ticker in test_cases:
        try:
            print(f"\nTesting: {ticker}")
            data = blp.bdp(tickers=ticker, flds='PX_LAST')
            
            if data.empty:
                print(f"  ❌ Empty response")
            else:
                print(f"  ✅ Success: {data}")
                return True, ticker, data
                
        except Exception as e:
            print(f"  ❌ Error: {e}")
    
    return False, None, None

def test_with_different_fields():
    """Test with different field names"""
    
    fields_to_test = [
        'PX_LAST',
        'LAST_PRICE', 
        'BID',
        'ASK',
        'VOLUME',
        'SECURITY_NAME'
    ]
    
    ticker = 'SPY US Equity'  # Most liquid ETF
    
    print(f"\nTesting different fields for {ticker}...")
    print("=" * 50)
    
    for field in fields_to_test:
        try:
            print(f"\nTesting field: {field}")
            data = blp.bdp(tickers=ticker, flds=field)
            
            if data.empty:
                print(f"  ❌ Empty response")
            else:
                print(f"  ✅ Success: {data}")
                return True, field, data
                
        except Exception as e:
            print(f"  ❌ Error: {e}")
    
    return False, None, None

def test_historical_data():
    """Test historical data which might work even if real-time doesn't"""
    
    try:
        print("\nTesting historical data...")
        print("=" * 30)
        
        # Try to get yesterday's data
        from datetime import datetime, timedelta
        yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
        
        data = blp.bdh(
            tickers='SPY US Equity',
            flds='PX_LAST', 
            start_date=yesterday,
            end_date=yesterday
        )
        
        if data.empty:
            print("❌ Empty historical data")
            return False
        else:
            print(f"✅ Historical data success: {data}")
            return True
            
    except Exception as e:
        print(f"❌ Historical data error: {e}")
        return False

def main():
    print("Bloomberg Connection Diagnostic")
    print("=" * 40)
    print("This will test various Bloomberg queries to find what works")
    
    # Test 1: Multiple tickers
    success, working_ticker, data = test_multiple_tickers()
    
    if success:
        print(f"\n🎉 Found working ticker: {working_ticker}")
        print("Bloomberg connection is working!")
        return
    
    # Test 2: Different fields
    success, working_field, data = test_with_different_fields()
    
    if success:
        print(f"\n🎉 Found working field: {working_field}")
        print("Bloomberg connection is working!")
        return
    
    # Test 3: Historical data
    if test_historical_data():
        print("\n🎉 Historical data is working!")
        print("Real-time data might have permission issues")
        return
    
    print("\n❌ All tests failed")
    print("\nPossible issues:")
    print("1. Bloomberg Terminal not fully logged in")
    print("2. No market data permissions")
    print("3. Bloomberg API not properly configured")
    print("4. Need to wait longer after Bloomberg Terminal startup")
    
    print("\nTry:")
    print("1. In Bloomberg Terminal, type: SPY <Equity> <GO>")
    print("2. Verify you can see real-time data")
    print("3. Wait 2-3 minutes after login")
    print("4. Run this script again")

if __name__ == "__main__":
    main()
