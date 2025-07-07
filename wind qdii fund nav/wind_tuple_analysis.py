"""
Wind API Tuple Analysis
This script analyzes the tuple structure returned by Wind API.
"""

import pandas as pd
from datetime import datetime, timedelta

try:
    from WindPy import w
    WIND_AVAILABLE = True
except ImportError:
    WIND_AVAILABLE = False
    print("WindPy not available.")

def analyze_wind_result(result, description=""):
    """Analyze Wind API result structure"""
    print(f"\n--- Analyzing {description} ---")
    print(f"Type: {type(result)}")
    print(f"Length: {len(result) if hasattr(result, '__len__') else 'N/A'}")
    
    if isinstance(result, tuple):
        for i, item in enumerate(result):
            print(f"  Item {i}: Type={type(item)}")
            if hasattr(item, 'shape'):
                print(f"    Shape: {item.shape}")
            if hasattr(item, 'columns'):
                print(f"    Columns: {list(item.columns)}")
            if hasattr(item, '__len__') and len(item) < 20:
                print(f"    Content: {item}")
            elif hasattr(item, 'head'):
                print(f"    Head:\n{item.head()}")
            else:
                print(f"    Content preview: {str(item)[:200]}...")

def test_wind_queries():
    """Test various Wind queries and analyze results"""
    if not WIND_AVAILABLE:
        return
    
    try:
        # Connect to Wind
        w.start()
        print("Connected to Wind")
        
        # Test 1: Simple stock query
        print("\n" + "="*50)
        print("TEST 1: Simple stock query")
        result = w.wsd("000001.SZ", "close", "20240101", "20240105", usedf=True)
        analyze_wind_result(result, "Simple stock query")
        
        # Test 2: ETF query
        print("\n" + "="*50)
        print("TEST 2: ETF query")
        result = w.wsd("510050.SH", "close", "20240101", "20240105", usedf=True)
        analyze_wind_result(result, "ETF query")
        
        # Test 3: QDII ETF query
        print("\n" + "="*50)
        print("TEST 3: QDII ETF query")
        result = w.wsd("513090.SH", "close", "20240101", "20240105", usedf=True)
        analyze_wind_result(result, "QDII ETF query")
        
        # Test 4: Fund query
        print("\n" + "="*50)
        print("TEST 4: Fund query")
        result = w.wsd("000001.OF", "nav", "20240101", "20240105", usedf=True)
        analyze_wind_result(result, "Fund query")
        
        # Test 5: Multiple fields
        print("\n" + "="*50)
        print("TEST 5: Multiple fields")
        result = w.wsd("513090.SH", ["close", "volume"], "20240101", "20240105", usedf=True)
        analyze_wind_result(result, "Multiple fields query")
        
        # Test 6: Without usedf
        print("\n" + "="*50)
        print("TEST 6: Without usedf")
        result = w.wsd("513090.SH", "close", "20240101", "20240105")
        analyze_wind_result(result, "Without usedf")
        
        # Test 7: WSS query
        print("\n" + "="*50)
        print("TEST 7: WSS query")
        result = w.wss("513090.SH", "sec_name", usedf=True)
        analyze_wind_result(result, "WSS query")
        
        w.stop()
        print("\nDisconnected from Wind")
        
    except Exception as e:
        print(f"Error in testing: {e}")

def create_working_nav_retriever():
    """Create a working NAV retriever based on the analysis"""
    print("\n" + "="*50)
    print("Creating working NAV retriever...")
    
    try:
        w.start()
        
        # Test with a known QDII ETF
        test_code = "513090.SH"
        print(f"Testing with {test_code}")
        
        result = w.wsd(test_code, "close", "20240101", "20240110", usedf=True)
        
        if isinstance(result, tuple) and len(result) >= 2:
            data = result[0]  # Usually the first element is the data
            error_info = result[1]  # Second element might be error info
            
            print(f"Data type: {type(data)}")
            print(f"Error info type: {type(error_info)}")
            
            if hasattr(error_info, 'ErrorCode'):
                print(f"Error code: {error_info.ErrorCode}")
                if error_info.ErrorCode == 0:
                    print("Success!")
                    if isinstance(data, pd.DataFrame):
                        print(f"DataFrame shape: {data.shape}")
                        print(f"Columns: {list(data.columns)}")
                        print("Sample data:")
                        print(data.head())
                        
                        # Save sample data
                        data['wind_code'] = test_code
                        data.to_csv("sample_wind_data.csv", encoding='utf-8-sig')
                        print("Sample data saved to sample_wind_data.csv")
            else:
                print("No ErrorCode attribute found")
                
        w.stop()
        
    except Exception as e:
        print(f"Error in working retriever: {e}")

def main():
    """Main function"""
    print("Wind API Tuple Analysis")
    print("=" * 50)
    
    test_wind_queries()
    create_working_nav_retriever()

if __name__ == "__main__":
    main()
