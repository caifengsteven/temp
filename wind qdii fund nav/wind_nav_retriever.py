"""
Wind NAV Data Retriever for QDII Funds
This script uses Wind API to get NAV data for QDII funds from the parsed Excel file.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import re

try:
    from WindPy import w
    WIND_AVAILABLE = True
except ImportError:
    WIND_AVAILABLE = False
    print("WindPy not available. Please install WindPy library.")

class WindNavRetriever:
    def __init__(self, csv_file="qdii_funds_parsed.csv"):
        self.csv_file = csv_file
        self.w = None
        self.connected = False
        
    def connect_wind(self):
        """Connect to Wind terminal"""
        if not WIND_AVAILABLE:
            print("Error: WindPy library is not installed.")
            return False
            
        try:
            self.w = w
            result = self.w.start()
            if result.ErrorCode == 0:
                self.connected = True
                print("Successfully connected to Wind terminal.")
                return True
            else:
                print(f"Failed to connect to Wind terminal. Error code: {result.ErrorCode}")
                return False
        except Exception as e:
            print(f"Error connecting to Wind: {str(e)}")
            return False
    
    def read_fund_data(self):
        """Read the parsed fund data"""
        try:
            df = pd.read_csv(self.csv_file, encoding='utf-8-sig')
            print(f"Read {len(df)} funds from {self.csv_file}")
            print(f"Columns: {list(df.columns)}")
            return df
        except Exception as e:
            print(f"Error reading fund data: {e}")
            return None
    
    def extract_wind_codes(self, df):
        """Extract Wind codes from the data"""
        wind_codes = []
        
        # The fund codes are in the '代码' column
        if '代码' in df.columns:
            fund_codes = df['代码'].tolist()
            fund_names = df['名称'].tolist() if '名称' in df.columns else [''] * len(fund_codes)
        else:
            print("No '代码' column found")
            return []
        
        for i, code in enumerate(fund_codes):
            if isinstance(code, str) and code.strip():
                # Wind codes are typically in format like 513090.SH, 159570.SZ
                wind_code = code.strip()
                wind_codes.append({
                    'wind_code': wind_code,
                    'fund_name': fund_names[i] if i < len(fund_names) else '',
                    'original_code': code
                })
        
        print(f"Extracted {len(wind_codes)} Wind codes")
        if wind_codes:
            print("Sample codes:")
            for i, item in enumerate(wind_codes[:5]):
                print(f"  {i+1}. {item['wind_code']} - {item['fund_name']}")
        
        return wind_codes
    
    def get_nav_data_from_wind(self, wind_code, start_date=None, end_date=None):
        """Get NAV data for a specific fund from Wind"""
        if not self.connected:
            print("Not connected to Wind terminal.")
            return None

        if start_date is None:
            start_date = (datetime.now() - timedelta(days=365)).strftime("%Y%m%d")
        if end_date is None:
            end_date = datetime.now().strftime("%Y%m%d")

        try:
            print(f"  Querying Wind for {wind_code}...")

            # Try different field combinations for ETFs and funds
            field_sets = [
                ["close", "volume", "amt"],  # For ETFs
                ["nav"],  # For funds
                ["close"],  # Basic price data
                ["nav", "nav_adj"]  # NAV data
            ]

            for i, fields in enumerate(field_sets):
                try:
                    print(f"    Trying field set {i+1}: {fields}")
                    result = self.w.wsd(wind_code, fields, start_date, end_date, usedf=True)

                    # Check if result is successful
                    if hasattr(result, 'ErrorCode'):
                        if result.ErrorCode == 0:
                            data = result.Data[0] if hasattr(result, 'Data') else result
                            if data is not None and not data.empty:
                                # Add fund code to the data
                                data = data.copy()
                                data['wind_code'] = wind_code
                                data['date'] = data.index
                                data = data.reset_index(drop=True)
                                print(f"    Success with field set {i+1}! Got {len(data)} records")
                                return data
                        else:
                            print(f"    Field set {i+1} failed with error code: {result.ErrorCode}")
                    else:
                        # Sometimes result doesn't have ErrorCode but has data
                        if hasattr(result, 'Data') and result.Data is not None:
                            data = result.Data[0] if isinstance(result.Data, list) else result.Data
                            if data is not None and not data.empty:
                                data = data.copy()
                                data['wind_code'] = wind_code
                                data['date'] = data.index
                                data = data.reset_index(drop=True)
                                print(f"    Success with field set {i+1}! Got {len(data)} records")
                                return data
                        elif hasattr(result, 'Data'):
                            data = result
                            if data is not None and not data.empty:
                                data = data.copy()
                                data['wind_code'] = wind_code
                                data['date'] = data.index
                                data = data.reset_index(drop=True)
                                print(f"    Success with field set {i+1}! Got {len(data)} records")
                                return data

                except Exception as field_error:
                    print(f"    Field set {i+1} error: {str(field_error)}")
                    continue

            print(f"    All field sets failed for {wind_code}")
            return None

        except Exception as e:
            print(f"    Error retrieving data for {wind_code}: {str(e)}")
            return None
    
    def get_fund_basic_info_from_wind(self, wind_codes):
        """Get basic fund information from Wind"""
        if not self.connected:
            return None

        try:
            print("Getting basic fund information from Wind...")

            # Get basic fund info
            codes = [item['wind_code'] for item in wind_codes[:5]]  # Limit for testing

            # Try different field combinations
            field_sets = [
                ["sec_name"],  # Basic name
                ["fund_fullname", "fund_setupdate"],  # Fund specific
                ["sec_name", "ipo_date"],  # General security info
            ]

            for fields in field_sets:
                try:
                    print(f"  Trying fields: {fields}")
                    result = self.w.wss(codes, fields, usedf=True)

                    if hasattr(result, 'ErrorCode') and result.ErrorCode == 0:
                        data = result.Data[0] if hasattr(result, 'Data') else result
                        if data is not None and not data.empty:
                            print(f"  Success! Got basic info for {len(data)} funds")
                            return data
                    elif hasattr(result, 'Data') and result.Data is not None:
                        data = result.Data[0] if isinstance(result.Data, list) else result.Data
                        if data is not None and not data.empty:
                            print(f"  Success! Got basic info for {len(data)} funds")
                            return data
                except Exception as e:
                    print(f"  Fields {fields} failed: {str(e)}")
                    continue

            print("  All field sets failed for basic info")
            return None

        except Exception as e:
            print(f"Error getting basic fund info: {str(e)}")
            return None
    
    def process_all_funds(self, max_funds=None):
        """Process all funds and get NAV data"""
        print("Wind NAV Data Retriever for QDII Funds")
        print("=" * 50)
        
        # Step 1: Connect to Wind
        print("Step 1: Connecting to Wind terminal...")
        if not self.connect_wind():
            return False
        
        # Step 2: Read fund data
        print("\nStep 2: Reading parsed fund data...")
        df = self.read_fund_data()
        if df is None:
            return False
        
        # Step 3: Extract Wind codes
        print("\nStep 3: Extracting Wind codes...")
        wind_codes = self.extract_wind_codes(df)
        if not wind_codes:
            print("No valid Wind codes found")
            return False
        
        # Step 4: Get basic fund info
        print(f"\nStep 4: Getting basic fund information...")
        basic_info = self.get_fund_basic_info_from_wind(wind_codes)
        if basic_info is not None:
            basic_info.to_csv("qdii_fund_basic_info_wind.csv", encoding='utf-8-sig')
            print("Basic fund info saved to qdii_fund_basic_info_wind.csv")
        
        # Step 5: Get NAV data
        print(f"\nStep 5: Getting NAV data from Wind...")
        
        if max_funds:
            wind_codes = wind_codes[:max_funds]
        
        all_nav_data = []
        total_funds = len(wind_codes)
        
        for i, fund_info in enumerate(wind_codes):
            wind_code = fund_info['wind_code']
            fund_name = fund_info['fund_name']
            
            print(f"Processing {wind_code} - {fund_name} ({i+1}/{total_funds})")
            
            nav_data = self.get_nav_data_from_wind(wind_code)
            if nav_data is not None and not nav_data.empty:
                nav_data['fund_name'] = fund_name
                all_nav_data.append(nav_data)
                print(f"    Got {len(nav_data)} records")
            else:
                print(f"    No data for {wind_code}")
        
        # Step 6: Save combined data
        print(f"\nStep 6: Saving NAV data...")
        if all_nav_data:
            combined_data = pd.concat(all_nav_data, ignore_index=True)
            
            # Save to CSV
            output_file = "qdii_nav_data_wind.csv"
            combined_data.to_csv(output_file, index=False, encoding='utf-8-sig')
            
            print(f"NAV data saved to {output_file}")
            print(f"Total records: {len(combined_data)}")
            print(f"Date range: {combined_data['date'].min()} to {combined_data['date'].max()}")
            print(f"Unique funds: {combined_data['wind_code'].nunique()}")
            
            # Show summary by fund
            print("\nSummary by fund:")
            summary = combined_data.groupby(['wind_code', 'fund_name']).agg({
                'date': ['min', 'max', 'count']
            }).round(4)
            print(summary.head(10))
            
            return True
        else:
            print("No NAV data retrieved")
            return False
    
    def disconnect_wind(self):
        """Disconnect from Wind terminal"""
        if self.connected and self.w:
            self.w.stop()
            self.connected = False
            print("Disconnected from Wind terminal.")

def main():
    """Main function"""
    retriever = WindNavRetriever()
    
    try:
        # Process first 10 funds for testing (remove max_funds for all funds)
        success = retriever.process_all_funds(max_funds=10)
        
        if success:
            print("\nWind NAV data retrieval completed successfully!")
        else:
            print("\nWind NAV data retrieval failed.")
    
    finally:
        # Always disconnect from Wind
        retriever.disconnect_wind()

if __name__ == "__main__":
    main()
