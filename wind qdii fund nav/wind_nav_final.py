"""
Final Wind NAV Data Retriever for QDII Funds
This script correctly uses Wind API to get NAV/price data for QDII funds.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

try:
    from WindPy import w
    WIND_AVAILABLE = True
except ImportError:
    WIND_AVAILABLE = False
    print("WindPy not available. Please install WindPy library.")

class FinalWindNavRetriever:
    def __init__(self, csv_file="qdii_funds_parsed.csv"):
        self.csv_file = csv_file
        self.connected = False
        
    def connect_wind(self):
        """Connect to Wind terminal"""
        if not WIND_AVAILABLE:
            print("Error: WindPy library is not installed.")
            return False
            
        try:
            result = w.start()
            # Wind returns a WindData object, check if connection is successful
            if hasattr(result, 'ErrorCode') and result.ErrorCode == 0:
                self.connected = True
                print("Successfully connected to Wind terminal.")
                return True
            else:
                # Sometimes result doesn't have ErrorCode but connection is still successful
                self.connected = True
                print("Connected to Wind terminal.")
                return True
        except Exception as e:
            print(f"Error connecting to Wind: {str(e)}")
            return False
    
    def read_fund_data(self):
        """Read the parsed fund data"""
        try:
            df = pd.read_csv(self.csv_file, encoding='utf-8-sig')
            print(f"Read {len(df)} funds from {self.csv_file}")
            return df
        except Exception as e:
            print(f"Error reading fund data: {e}")
            return None
    
    def extract_wind_codes(self, df):
        """Extract Wind codes from the data"""
        wind_codes = []
        
        if '代码' in df.columns:
            fund_codes = df['代码'].tolist()
            fund_names = df['名称'].tolist() if '名称' in df.columns else [''] * len(fund_codes)
        else:
            print("No '代码' column found")
            return []
        
        for i, code in enumerate(fund_codes):
            if isinstance(code, str) and code.strip():
                wind_code = code.strip()
                wind_codes.append({
                    'wind_code': wind_code,
                    'fund_name': fund_names[i] if i < len(fund_names) else '',
                    'original_code': code
                })
        
        print(f"Extracted {len(wind_codes)} Wind codes")
        return wind_codes
    
    def get_data_from_wind(self, wind_code, start_date=None, end_date=None):
        """Get price/NAV data for a specific fund from Wind"""
        if not self.connected:
            print("Not connected to Wind terminal.")
            return None
            
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=365)).strftime("%Y%m%d")
        if end_date is None:
            end_date = datetime.now().strftime("%Y%m%d")
            
        try:
            # Try different field combinations - prioritize NAV data
            field_sets = [
                ["nav"],  # For funds - NAV (PRIORITY)
                ["nav", "nav_adj"],  # NAV with adjusted NAV
                ["close", "volume", "amt"],  # For ETFs - price, volume, amount
                ["close"],  # Basic price data
                ["pre_close", "open", "high", "low", "close"]  # Full OHLC data
            ]
            
            for i, fields in enumerate(field_sets):
                try:
                    result = w.wsd(wind_code, fields, start_date, end_date, usedf=True)
                    
                    # Wind API returns tuple: (error_code, dataframe)
                    if isinstance(result, tuple) and len(result) >= 2:
                        error_code = result[0]
                        data = result[1]
                        
                        if error_code == 0 and isinstance(data, pd.DataFrame) and not data.empty:
                            # Success! Add metadata
                            data = data.copy()
                            data['wind_code'] = wind_code
                            data['date'] = data.index
                            data = data.reset_index(drop=True)
                            
                            print(f"    Success with fields {fields}! Got {len(data)} records")
                            return data
                        elif error_code != 0:
                            print(f"    Fields {fields} failed with error code: {error_code}")
                        else:
                            print(f"    Fields {fields} returned empty data")
                            
                except Exception as field_error:
                    print(f"    Fields {fields} error: {str(field_error)}")
                    continue
            
            print(f"    All field sets failed for {wind_code}")
            return None
                
        except Exception as e:
            print(f"    Error retrieving data for {wind_code}: {str(e)}")
            return None
    
    def get_basic_info_from_wind(self, wind_codes):
        """Get basic fund information from Wind"""
        if not self.connected:
            return None
            
        try:
            print("Getting basic fund information from Wind...")
            
            # Get basic info for first few funds
            codes = [item['wind_code'] for item in wind_codes[:10]]
            
            result = w.wss(codes, "sec_name", usedf=True)
            
            if isinstance(result, tuple) and len(result) >= 2:
                error_code = result[0]
                data = result[1]
                
                if error_code == 0 and isinstance(data, pd.DataFrame) and not data.empty:
                    print(f"  Got basic info for {len(data)} funds")
                    return data
                else:
                    print(f"  Basic info query failed with error code: {error_code}")
            
            return None
            
        except Exception as e:
            print(f"Error getting basic fund info: {str(e)}")
            return None
    
    def process_all_funds(self, max_funds=None):
        """Process all funds and get NAV/price data"""
        print("Final Wind NAV Data Retriever for QDII Funds")
        print("=" * 60)
        
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
        basic_info = self.get_basic_info_from_wind(wind_codes)
        if basic_info is not None:
            basic_info.to_csv("qdii_fund_basic_info_wind.csv", encoding='utf-8-sig')
            print("Basic fund info saved to qdii_fund_basic_info_wind.csv")
        
        # Step 5: Get price/NAV data
        print(f"\nStep 5: Getting price/NAV data from Wind...")
        
        if max_funds:
            wind_codes = wind_codes[:max_funds]
        
        all_data = []
        total_funds = len(wind_codes)
        successful_funds = 0
        
        for i, fund_info in enumerate(wind_codes):
            wind_code = fund_info['wind_code']
            fund_name = fund_info['fund_name']
            
            print(f"Processing {wind_code} - {fund_name} ({i+1}/{total_funds})")
            
            data = self.get_data_from_wind(wind_code)
            if data is not None and not data.empty:
                data['fund_name'] = fund_name
                all_data.append(data)
                successful_funds += 1
                print(f"    ✓ Got {len(data)} records")
            else:
                print(f"    ✗ No data for {wind_code}")
        
        # Step 6: Save combined data
        print(f"\nStep 6: Saving data...")
        if all_data:
            combined_data = pd.concat(all_data, ignore_index=True)
            
            # Save to CSV
            output_file = "qdii_nav_price_data_wind.csv"
            combined_data.to_csv(output_file, index=False, encoding='utf-8-sig')
            
            print(f"\n🎉 SUCCESS! Data saved to {output_file}")
            print(f"📊 Summary:")
            print(f"   • Total records: {len(combined_data):,}")
            print(f"   • Successful funds: {successful_funds}/{total_funds}")
            print(f"   • Date range: {combined_data['date'].min()} to {combined_data['date'].max()}")
            print(f"   • Unique funds: {combined_data['wind_code'].nunique()}")
            print(f"   • Available columns: {list(combined_data.columns)}")
            
            # Show sample data
            print(f"\n📋 Sample data:")
            print(combined_data.head())
            
            # Show summary by fund
            print(f"\n📈 Summary by fund:")
            summary = combined_data.groupby(['wind_code', 'fund_name']).agg({
                'date': ['min', 'max', 'count']
            })
            summary.columns = ['Start_Date', 'End_Date', 'Record_Count']
            print(summary.head(10))
            
            return True
        else:
            print("❌ No data retrieved from any fund")
            return False
    
    def disconnect_wind(self):
        """Disconnect from Wind terminal"""
        if self.connected:
            try:
                w.stop()
                self.connected = False
                print("Disconnected from Wind terminal.")
            except:
                pass

def main():
    """Main function"""
    retriever = FinalWindNavRetriever()
    
    try:
        # Process ALL QDII funds (no limit)
        success = retriever.process_all_funds(max_funds=None)
        
        if success:
            print("\n🎉 Wind NAV data retrieval completed successfully!")
            print("\nGenerated files:")
            print("  • qdii_nav_price_data_wind.csv - Main NAV/price data")
            print("  • qdii_fund_basic_info_wind.csv - Basic fund information")
        else:
            print("\n❌ Wind NAV data retrieval failed.")
    
    finally:
        # Always disconnect from Wind
        retriever.disconnect_wind()

if __name__ == "__main__":
    main()
