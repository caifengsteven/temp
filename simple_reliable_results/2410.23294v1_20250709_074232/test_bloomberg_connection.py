#!/usr/bin/env python3
"""
Test Bloomberg connection and data retrieval
Run this script to verify Bloomberg Terminal access before running the main strategy
"""

import sys
import pandas as pd
from datetime import datetime, timedelta

def test_xbbg_import():
    """Test if xbbg can be imported"""
    try:
        from xbbg import blp
        print("✓ xbbg library imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Failed to import xbbg: {e}")
        print("  Install with: pip install xbbg blpapi --index-url=https://bcms.bloomberg.com/pip/simple/")
        return False

def test_bloomberg_connection():
    """Test basic Bloomberg connection"""
    try:
        from xbbg import blp

        # Test with a simple daily data request
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')

        print(f"Testing Bloomberg connection...")
        print(f"Requesting EURUSD data from {start_date} to {end_date}")

        data = blp.bdh(
            tickers='EURUSD Curncy',
            flds='PX_LAST',
            start_date=start_date,
            end_date=end_date
        )
        
        if data is not None and len(data) > 0:
            print(f"✓ Successfully retrieved {len(data)} daily data points")
            print(f"  Date range: {data.index.min()} to {data.index.max()}")
            # Handle multi-level columns from Bloomberg
            if isinstance(data.columns, pd.MultiIndex):
                price_col = data.iloc[:, 0]  # First column
            else:
                price_col = data['PX_LAST'] if 'PX_LAST' in data.columns else data.iloc[:, 0]
            print(f"  Price range: {price_col.min():.4f} to {price_col.max():.4f}")
            return True
        else:
            print("✗ No data returned from Bloomberg")
            return False
            
    except Exception as e:
        print(f"✗ Bloomberg connection failed: {e}")
        return False

def test_intraday_data():
    """Test intraday data retrieval"""
    try:
        from xbbg import blp

        # Test with recent intraday data (yesterday)
        yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')

        print(f"\nTesting intraday data retrieval...")
        print(f"Requesting 1-minute EURUSD data for {yesterday}")

        data = blp.bdib(
            ticker='EURUSD Curncy',
            dt=yesterday  # Use dt parameter for single day
        )
        
        if data is not None and len(data) > 0:
            print(f"✓ Successfully retrieved {len(data)} intraday data points")
            print(f"  Time range: {data.index.min()} to {data.index.max()}")
            print(f"  Columns: {list(data.columns)}")
            
            # Show sample data
            print(f"\nSample data:")
            print(data.head(3).to_string())
            return True
        else:
            print("✗ No intraday data returned from Bloomberg")
            return False
            
    except Exception as e:
        print(f"✗ Intraday data retrieval failed: {e}")
        print("  Note: Intraday data may have limited history or require special permissions")
        return False

def test_multiple_currencies():
    """Test multiple currency pairs"""
    currency_pairs = [
        'EURUSD Curncy',
        'GBPUSD Curncy', 
        'USDJPY Curncy'
    ]
    
    print(f"\nTesting multiple currency pairs...")
    
    try:
        from xbbg import blp

        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')

        results = {}

        for pair in currency_pairs:
            try:
                data = blp.bdh(
                    tickers=pair,
                    flds='PX_LAST',
                    start_date=start_date,
                    end_date=end_date
                )
                
                if data is not None and len(data) > 0:
                    results[pair] = len(data)
                    print(f"  ✓ {pair}: {len(data)} data points")
                else:
                    results[pair] = 0
                    print(f"  ✗ {pair}: No data")
                    
            except Exception as e:
                results[pair] = None
                print(f"  ✗ {pair}: Error - {e}")
        
        successful = sum(1 for v in results.values() if v and v > 0)
        print(f"\nSuccessfully retrieved data for {successful}/{len(currency_pairs)} currency pairs")
        
        return successful > 0
        
    except Exception as e:
        print(f"✗ Multiple currency test failed: {e}")
        return False

def main():
    """Run all Bloomberg tests"""
    print("Bloomberg Terminal Connection Test")
    print("=" * 50)
    
    tests = [
        ("xbbg Import", test_xbbg_import),
        ("Bloomberg Connection", test_bloomberg_connection),
        ("Intraday Data", test_intraday_data),
        ("Multiple Currencies", test_multiple_currencies)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        print("-" * 30)
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"✗ Test failed with exception: {e}")
            results[test_name] = False
    
    # Summary
    print(f"\n{'='*50}")
    print("TEST SUMMARY")
    print(f"{'='*50}")
    
    for test_name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"{test_name:20s}: {status}")
    
    total_passed = sum(results.values())
    total_tests = len(results)
    
    print(f"\nOverall: {total_passed}/{total_tests} tests passed")
    
    if total_passed == total_tests:
        print("✓ All tests passed! Bloomberg integration should work correctly.")
    elif total_passed >= 2:
        print("⚠ Some tests passed. Basic functionality available but some features may be limited.")
    else:
        print("✗ Most tests failed. Bloomberg integration may not work properly.")
        print("  Check Bloomberg Terminal connection and xbbg installation.")
    
    return total_passed >= 2

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
