#!/usr/bin/env python3
"""
Simple Bloomberg connection test
"""

import pandas as pd
from datetime import datetime

try:
    from xbbg import blp
    print("✓ xbbg library imported successfully")
except ImportError as e:
    print(f"✗ Error importing xbbg: {e}")
    exit(1)

def test_simple_queries():
    """Test various simple Bloomberg queries"""
    
    test_cases = [
        ('AAPL US Equity', ['PX_LAST']),
        ('SPY US Equity', ['PX_LAST']),
        ('SPX Index', ['PX_LAST']),
        ('VIX Index', ['PX_LAST']),
    ]
    
    print(f"\n=== Testing Bloomberg Connection at {datetime.now()} ===")
    
    for ticker, fields in test_cases:
        try:
            print(f"\nTesting: {ticker} - {fields}")
            data = blp.bdp(tickers=ticker, flds=fields)
            print(f"  Response shape: {data.shape}")
            print(f"  Response:\n{data}")
            
            if not data.empty:
                print(f"  ✓ Success: {ticker} = {data.iloc[0, 0]}")
                return True
            else:
                print(f"  ✗ Empty response for {ticker}")
                
        except Exception as e:
            print(f"  ✗ Error with {ticker}: {e}")
    
    return False

def test_bloomberg_status():
    """Check Bloomberg service status"""
    try:
        print("\n=== Checking Bloomberg Service Status ===")
        # Try to get service status
        import blpapi
        print("✓ blpapi module available")
        
        # Try basic session
        session = blpapi.Session()
        if session.start():
            print("✓ Bloomberg session started successfully")
            session.stop()
            return True
        else:
            print("✗ Failed to start Bloomberg session")
            return False
            
    except ImportError:
        print("✗ blpapi module not available")
        return False
    except Exception as e:
        print(f"✗ Error checking Bloomberg status: {e}")
        return False

if __name__ == "__main__":
    print("Bloomberg Connection Diagnostic Tool")
    print("=" * 50)
    
    # Test basic connection
    connection_ok = test_simple_queries()
    
    if not connection_ok:
        print("\n" + "=" * 50)
        print("Connection failed. Checking Bloomberg service...")
        test_bloomberg_status()
        
        print("\nTroubleshooting steps:")
        print("1. Ensure Bloomberg Terminal is fully loaded and logged in")
        print("2. Try running a simple query in Bloomberg Terminal (e.g., SPX <Index> <GO>)")
        print("3. Wait a few minutes after login before running this script")
        print("4. Check if you have the necessary data permissions")
        print("5. Restart Bloomberg Terminal if needed")
    else:
        print("\n✓ Bloomberg connection is working!")
