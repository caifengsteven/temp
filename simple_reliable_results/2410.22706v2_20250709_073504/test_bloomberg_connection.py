"""
Test Bloomberg Connection Script
Simple script to verify Bloomberg xbbg setup before running the main GSPHAR model.
"""

import sys
from datetime import datetime, timedelta

def test_xbbg_import():
    """Test if xbbg can be imported."""
    try:
        import xbbg
        from xbbg import blp
        print("✓ xbbg imported successfully")
        print(f"  xbbg version: {getattr(xbbg, '__version__', 'unknown')}")
        return True
    except ImportError as e:
        print(f"✗ Failed to import xbbg: {e}")
        print("  Install with: pip install xbbg")
        print("  Also install dependencies: pip install ruamel.yaml pyarrow")
        return False

def test_bloomberg_connection():
    """Test basic Bloomberg connection using xbbg."""
    try:
        from xbbg import blp

        print("Testing Bloomberg connection...")

        # Test simple BDP request with timeout
        data = blp.bdp(tickers='SPX Index', flds='PX_LAST', timeout=10)

        if not data.empty:
            print("✓ Bloomberg connection successful!")
            print(f"  SPX Index last price: {data.iloc[0, 0]}")

            # Test additional fields
            extended_data = blp.bdp(
                tickers='SPX Index',
                flds=['PX_LAST', 'CHG_PCT_1D', 'VOLATILITY_30D'],
                timeout=10
            )

            if not extended_data.empty:
                print("✓ Extended field access successful!")
                for col in extended_data.columns:
                    print(f"  {col}: {extended_data.iloc[0][col]}")

            return True
        else:
            print("✗ Bloomberg returned empty data")
            print("  Check if Bloomberg Terminal is running")
            print("  Verify API permissions and ticker access")
            return False

    except Exception as e:
        print(f"✗ Bloomberg connection failed: {e}")
        print("  Make sure Bloomberg Terminal is running and API is configured")
        print("  Check if blpapi is properly installed")
        return False

def test_historical_data():
    """Test historical data retrieval using xbbg."""
    try:
        from xbbg import blp

        print("Testing historical data retrieval...")

        # Get last 20 business days of data
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')

        data = blp.bdh(
            tickers='SPX Index',
            flds=['PX_OPEN', 'PX_HIGH', 'PX_LOW', 'PX_LAST'],
            start_date=start_date,
            end_date=end_date,
            timeout=15
        )

        if not data.empty:
            print("✓ Historical data retrieval successful!")
            print(f"  Retrieved {len(data)} days of data")
            print(f"  Date range: {data.index[0]} to {data.index[-1]}")
            print(f"  Columns: {list(data.columns)}")

            # Test data quality
            if len(data) >= 10:  # At least 10 business days
                print("✓ Sufficient data volume")

                # Check for missing values
                missing_pct = data.isnull().sum().sum() / (len(data) * len(data.columns)) * 100
                print(f"  Missing data: {missing_pct:.1f}%")

                if missing_pct < 20:
                    print("✓ Good data quality")
                else:
                    print("⚠ High missing data percentage")

            return True
        else:
            print("✗ Historical data retrieval returned empty data")
            print("  Check date range and ticker validity")
            return False

    except Exception as e:
        print(f"✗ Historical data retrieval failed: {e}")
        return False

def test_multiple_tickers():
    """Test multiple ticker data retrieval using xbbg."""
    try:
        from xbbg import blp

        print("Testing multiple ticker data retrieval...")

        tickers = ['SPX Index', 'SX5E Index', 'NKY Index']

        data = blp.bdp(
            tickers=tickers,
            flds='PX_LAST',
            timeout=15
        )

        if not data.empty and len(data) == len(tickers):
            print("✓ Multiple ticker retrieval successful!")
            for ticker in tickers:
                if ticker in data.index:
                    print(f"  {ticker}: {data.loc[ticker, 'PX_LAST']}")
            return True
        else:
            print("✗ Multiple ticker retrieval failed or incomplete")
            print(f"  Expected {len(tickers)} tickers, got {len(data)}")
            if not data.empty:
                print(f"  Available tickers: {list(data.index)}")
            return False

    except Exception as e:
        print(f"✗ Multiple ticker retrieval failed: {e}")
        return False


def test_intraday_data():
    """Test intraday data retrieval using xbbg."""
    try:
        from xbbg import blp

        print("Testing intraday data retrieval...")

        # Get yesterday's date for intraday data
        yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')

        data = blp.bdib(
            ticker='SPX Index',
            dt=yesterday,
            timeout=20
        )

        if not data.empty:
            print("✓ Intraday data retrieval successful!")
            print(f"  Retrieved {len(data)} intraday bars for {yesterday}")
            print(f"  Columns: {list(data.columns)}")
            print(f"  Time range: {data.index[0]} to {data.index[-1]}")
            return True
        else:
            print("✗ Intraday data retrieval returned empty data")
            print("  This might be normal for weekends or holidays")
            return False

    except Exception as e:
        print(f"✗ Intraday data retrieval failed: {e}")
        print("  Note: Intraday data may not be available for all tickers/dates")
        return False

def main():
    """Run all Bloomberg connection tests."""
    print("="*50)
    print("Bloomberg Connection Test Suite")
    print("="*50)
    
    tests = [
        ("xbbg Import", test_xbbg_import),
        ("Bloomberg Connection", test_bloomberg_connection),
        ("Historical Data", test_historical_data),
        ("Multiple Tickers", test_multiple_tickers),
        ("Intraday Data", test_intraday_data)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        print("-" * 30)
        
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ Test failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*50)
    print("Test Summary:")
    print("="*50)
    
    passed = 0
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nPassed: {passed}/{len(results)} tests")
    
    if passed == len(results):
        print("\n✓ All tests passed! Bloomberg setup is working correctly.")
        print("  You can now run the main GSPHAR script.")
    else:
        print(f"\n✗ {len(results) - passed} test(s) failed.")
        print("  Please check Bloomberg setup and configuration.")
        print("  See bloomberg_setup_guide.md for help.")
    
    return passed == len(results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
