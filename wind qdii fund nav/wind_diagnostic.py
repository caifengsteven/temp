"""
Wind API Diagnostic Script
This script tests Wind API functionality and explores available data.
"""

import pandas as pd
from datetime import datetime, timedelta

try:
    from WindPy import w
    WIND_AVAILABLE = True
except ImportError:
    WIND_AVAILABLE = False
    print("WindPy not available.")

def test_wind_connection():
    """Test Wind connection"""
    if not WIND_AVAILABLE:
        return False
    
    try:
        result = w.start()
        print(f"Wind connection result: {result}")
        if hasattr(result, 'ErrorCode'):
            print(f"Error code: {result.ErrorCode}")
            return result.ErrorCode == 0
        return True
    except Exception as e:
        print(f"Wind connection error: {e}")
        return False

def test_simple_query():
    """Test a simple Wind query"""
    try:
        print("\nTesting simple query...")
        
        # Test with a well-known stock
        result = w.wsd("000001.SZ", "close", "2024-01-01", "2024-01-10", usedf=True)
        print(f"Simple query result type: {type(result)}")
        print(f"Result attributes: {dir(result)}")
        
        if hasattr(result, 'ErrorCode'):
            print(f"Error code: {result.ErrorCode}")
            if result.ErrorCode == 0:
                print("Simple query successful!")
                if hasattr(result, 'Data'):
                    print(f"Data type: {type(result.Data)}")
                    if isinstance(result.Data, list) and len(result.Data) > 0:
                        print(f"First data element: {result.Data[0]}")
                return True
        
        return False
        
    except Exception as e:
        print(f"Simple query error: {e}")
        return False

def test_etf_query():
    """Test ETF query"""
    try:
        print("\nTesting ETF query...")
        
        # Test with a known ETF
        etf_codes = ["510050.SH", "159915.SZ"]  # 50ETF, 创业板ETF
        
        for etf_code in etf_codes:
            print(f"\nTesting {etf_code}...")
            
            # Try different date formats
            date_formats = [
                ("2024-01-01", "2024-01-10"),
                ("20240101", "20240110")
            ]
            
            for start_date, end_date in date_formats:
                try:
                    print(f"  Date format: {start_date} to {end_date}")
                    result = w.wsd(etf_code, "close", start_date, end_date, usedf=True)
                    
                    if hasattr(result, 'ErrorCode'):
                        print(f"    Error code: {result.ErrorCode}")
                        if result.ErrorCode == 0:
                            print(f"    Success! Data type: {type(result.Data)}")
                            return True
                    else:
                        print(f"    No error code, result type: {type(result)}")
                        
                except Exception as e:
                    print(f"    Error: {e}")
        
        return False
        
    except Exception as e:
        print(f"ETF query error: {e}")
        return False

def test_fund_query():
    """Test fund query"""
    try:
        print("\nTesting fund query...")
        
        # Test with known fund codes
        fund_codes = ["000001.OF", "110022.OF"]  # 华夏成长, 易方达消费
        
        for fund_code in fund_codes:
            print(f"\nTesting {fund_code}...")
            
            try:
                result = w.wsd(fund_code, "nav", "20240101", "20240110", usedf=True)
                
                if hasattr(result, 'ErrorCode'):
                    print(f"  Error code: {result.ErrorCode}")
                    if result.ErrorCode == 0:
                        print(f"  Success! Data type: {type(result.Data)}")
                        return True
                else:
                    print(f"  No error code, result type: {type(result)}")
                    
            except Exception as e:
                print(f"  Error: {e}")
        
        return False
        
    except Exception as e:
        print(f"Fund query error: {e}")
        return False

def test_qdii_etf_query():
    """Test QDII ETF query with different approaches"""
    try:
        print("\nTesting QDII ETF query...")
        
        # Test with QDII ETF codes from the Excel file
        qdii_codes = ["513090.SH", "513130.SH", "159570.SZ"]
        
        for qdii_code in qdii_codes:
            print(f"\nTesting {qdii_code}...")
            
            # Try different field combinations
            field_combinations = [
                ["close"],
                ["nav"],
                ["pre_close"],
                ["open"],
                ["high", "low", "close"],
                ["volume"],
                ["amt"]
            ]
            
            for fields in field_combinations:
                try:
                    print(f"  Fields: {fields}")
                    result = w.wsd(qdii_code, fields, "20240101", "20240110", usedf=True)
                    
                    print(f"    Result type: {type(result)}")
                    print(f"    Has ErrorCode: {hasattr(result, 'ErrorCode')}")
                    
                    if hasattr(result, 'ErrorCode'):
                        print(f"    Error code: {result.ErrorCode}")
                        if result.ErrorCode == 0:
                            print(f"    SUCCESS! Data available")
                            if hasattr(result, 'Data'):
                                data = result.Data
                                print(f"    Data type: {type(data)}")
                                if isinstance(data, list) and len(data) > 0:
                                    print(f"    First element type: {type(data[0])}")
                                    if hasattr(data[0], 'shape'):
                                        print(f"    Data shape: {data[0].shape}")
                            return True
                    else:
                        print(f"    Error code: {result.ErrorCode}")
                        
                except Exception as e:
                    print(f"    Error: {e}")
        
        return False
        
    except Exception as e:
        print(f"QDII ETF query error: {e}")
        return False

def main():
    """Main diagnostic function"""
    print("Wind API Diagnostic Script")
    print("=" * 50)
    
    # Test connection
    print("1. Testing Wind connection...")
    if not test_wind_connection():
        print("Wind connection failed. Exiting.")
        return
    
    # Test simple query
    print("\n2. Testing simple stock query...")
    test_simple_query()
    
    # Test ETF query
    print("\n3. Testing ETF query...")
    test_etf_query()
    
    # Test fund query
    print("\n4. Testing fund query...")
    test_fund_query()
    
    # Test QDII ETF query
    print("\n5. Testing QDII ETF query...")
    test_qdii_etf_query()
    
    # Disconnect
    try:
        w.stop()
        print("\nDisconnected from Wind.")
    except:
        pass

if __name__ == "__main__":
    main()
