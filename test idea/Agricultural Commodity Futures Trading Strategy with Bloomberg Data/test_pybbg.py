import pybbg
import pandas as pd
from datetime import datetime

def test_bloomberg_connection():
    print("Testing Bloomberg connection with pybbg...")
    
    try:
        # Connect to Bloomberg
        bbg = pybbg.Pybbg()
        
        # Try to get data for a single ticker
        ticker = 'MSFT US Equity'
        start_date = '2023-01-01'
        end_date = datetime.now().strftime("%Y-%m-%d")
        fields = ['PX_LAST', 'VOLUME']
        
        print(f"📊 Retrieving data for {ticker} ({start_date} to {end_date})...")
        data = bbg.bdh(ticker, fields, start_date, end_date)
        
        if data is not None and not data.empty:
            print(f"✅ Successfully retrieved data with shape: {data.shape}")
            print("\nSample data:")
            print(data.head())
        else:
            print("⚠️ No data retrieved from Bloomberg")
        
    except Exception as e:
        import traceback
        print(f"❌ Error: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    test_bloomberg_connection()
