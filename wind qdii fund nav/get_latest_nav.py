"""
Get Latest NAV Data for QDII Funds
This script retrieves only the latest NAV values for QDII funds from Wind API.
"""

import pandas as pd
from datetime import datetime, timedelta

try:
    from WindPy import w
    WIND_AVAILABLE = True
except ImportError:
    WIND_AVAILABLE = False
    print("WindPy not available.")

class LatestNavRetriever:
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
            self.connected = True
            print("Successfully connected to Wind terminal.")
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
    
    def get_latest_nav_from_wind(self, wind_code):
        """Get latest NAV data for a specific fund from Wind"""
        if not self.connected:
            return None
            
        try:
            # Try different NAV-related fields
            nav_fields = [
                ["nav"],           # Standard NAV
                ["nav_adj"],       # Adjusted NAV
                ["unit_nav"],      # Unit NAV
                ["close"],         # Closing price (fallback for ETFs)
            ]
            
            for fields in nav_fields:
                try:
                    # Get latest data (today or most recent trading day)
                    end_date = datetime.now().strftime("%Y%m%d")
                    start_date = (datetime.now() - timedelta(days=10)).strftime("%Y%m%d")  # Last 10 days to ensure we get latest
                    
                    result = w.wsd(wind_code, fields, start_date, end_date, usedf=True)
                    
                    if isinstance(result, tuple) and len(result) >= 2:
                        error_code = result[0]
                        data = result[1]
                        
                        if error_code == 0 and isinstance(data, pd.DataFrame) and not data.empty:
                            # Get the latest (most recent) value
                            latest_data = data.iloc[-1]  # Last row is most recent
                            latest_date = data.index[-1]  # Last date
                            
                            nav_info = {
                                'wind_code': wind_code,
                                'latest_date': latest_date.strftime('%Y-%m-%d'),
                                'latest_nav': latest_data.iloc[0],  # First (and likely only) column value
                                'field_used': fields[0],
                                'data_points': len(data)
                            }
                            
                            print(f"    ✓ Latest {fields[0]}: {nav_info['latest_nav']} on {nav_info['latest_date']}")
                            return nav_info
                            
                except Exception as field_error:
                    continue
            
            print(f"    ✗ No NAV data found for {wind_code}")
            return None
                
        except Exception as e:
            print(f"    Error retrieving NAV for {wind_code}: {str(e)}")
            return None
    
    def get_latest_nav_wss(self, wind_codes):
        """Get latest NAV using WSS (cross-sectional data) - more efficient for latest values"""
        if not self.connected:
            return None

        try:
            print("Getting latest NAV using WSS method...")

            # Try different NAV fields
            nav_field_sets = [
                ["nav"],
                ["nav_adj"],
                ["unit_nav"],
                ["close"],  # Fallback for ETFs
            ]

            codes = [item['wind_code'] for item in wind_codes]

            for fields in nav_field_sets:
                try:
                    print(f"  Trying field: {fields[0]}")
                    result = w.wss(codes, fields[0], usedf=True)

                    if isinstance(result, tuple) and len(result) >= 2:
                        error_code = result[0]
                        data = result[1]

                        if error_code == 0 and isinstance(data, pd.DataFrame) and not data.empty:
                            print(f"  ✓ Success with {fields[0]}! Got data for {len(data)} funds")

                            # Add metadata
                            data['field_used'] = fields[0]
                            data['retrieval_date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

                            # Get NAV dates for each fund
                            print("  Getting NAV dates...")
                            try:
                                date_result = w.wss(codes, "nav_date", usedf=True)
                                if isinstance(date_result, tuple) and len(date_result) >= 2:
                                    date_error = date_result[0]
                                    date_data = date_result[1]
                                    if date_error == 0 and isinstance(date_data, pd.DataFrame) and not date_data.empty:
                                        data['nav_date'] = date_data.iloc[:, 0]
                                        print(f"    ✓ Got NAV dates for {len(date_data)} funds")
                                    else:
                                        data['nav_date'] = 'Unknown'
                                        print(f"    ✗ NAV dates failed with error: {date_error}")
                                else:
                                    data['nav_date'] = 'Unknown'
                            except:
                                data['nav_date'] = 'Unknown'
                                print("    ✗ NAV dates not available")

                            return data
                        else:
                            print(f"  ✗ {fields[0]} failed with error code: {error_code}")

                except Exception as e:
                    print(f"  ✗ {fields[0]} error: {str(e)}")
                    continue

            return None

        except Exception as e:
            print(f"Error in WSS method: {str(e)}")
            return None
    
    def process_latest_nav(self, max_funds=None):
        """Get latest NAV for all funds"""
        print("Latest NAV Retriever for QDII Funds")
        print("=" * 50)
        
        # Step 1: Connect to Wind
        print("Step 1: Connecting to Wind terminal...")
        if not self.connect_wind():
            return False
        
        # Step 2: Read fund data
        print("\nStep 2: Reading fund data...")
        df = self.read_fund_data()
        if df is None:
            return False
        
        # Step 3: Extract Wind codes
        print("\nStep 3: Extracting Wind codes...")
        wind_codes = []
        if '代码' in df.columns:
            for i, row in df.iterrows():
                wind_codes.append({
                    'wind_code': row['代码'],
                    'fund_name': row['名称'] if '名称' in df.columns else ''
                })
        
        if max_funds:
            wind_codes = wind_codes[:max_funds]
        
        print(f"Processing {len(wind_codes)} funds...")
        
        # Step 4: Try WSS method first (more efficient for latest data)
        print("\nStep 4: Getting latest NAV using efficient method...")
        latest_nav_data = self.get_latest_nav_wss(wind_codes)
        
        if latest_nav_data is not None:
            # Add fund names
            fund_name_map = {item['wind_code']: item['fund_name'] for item in wind_codes}
            latest_nav_data['fund_name'] = latest_nav_data.index.map(fund_name_map)
            
            # Save results
            output_file = "qdii_latest_nav.csv"
            latest_nav_data.to_csv(output_file, encoding='utf-8-sig')
            
            print(f"\n🎉 SUCCESS! Latest NAV data saved to {output_file}")
            print(f"📊 Summary:")
            print(f"   • Total funds: {len(latest_nav_data)}")
            print(f"   • Field used: {latest_nav_data['field_used'].iloc[0] if 'field_used' in latest_nav_data.columns else 'Unknown'}")
            print(f"   • Retrieved at: {latest_nav_data['retrieval_date'].iloc[0] if 'retrieval_date' in latest_nav_data.columns else 'Unknown'}")
            
            # Show sample data
            print(f"\n📋 Sample latest NAV data:")
            display_cols = [col for col in latest_nav_data.columns if col not in ['field_used', 'retrieval_date']]
            print(latest_nav_data[display_cols].head(10))
            
            return True
        else:
            # Fallback to individual queries
            print("\nStep 4b: Fallback to individual fund queries...")
            nav_results = []
            
            for i, fund_info in enumerate(wind_codes):
                wind_code = fund_info['wind_code']
                fund_name = fund_info['fund_name']
                
                print(f"Processing {wind_code} - {fund_name} ({i+1}/{len(wind_codes)})")
                
                nav_info = self.get_latest_nav_from_wind(wind_code)
                if nav_info:
                    nav_info['fund_name'] = fund_name
                    nav_results.append(nav_info)
            
            if nav_results:
                results_df = pd.DataFrame(nav_results)
                output_file = "qdii_latest_nav.csv"
                results_df.to_csv(output_file, index=False, encoding='utf-8-sig')
                
                print(f"\n🎉 SUCCESS! Latest NAV data saved to {output_file}")
                print(f"📊 Summary:")
                print(f"   • Total funds with data: {len(results_df)}")
                print(f"   • Date range: {results_df['latest_date'].min()} to {results_df['latest_date'].max()}")
                
                print(f"\n📋 Latest NAV data:")
                print(results_df[['wind_code', 'fund_name', 'latest_nav', 'latest_date']].head(10))
                
                return True
            else:
                print("❌ No NAV data retrieved")
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
    retriever = LatestNavRetriever()
    
    try:
        # Get latest NAV for ALL funds
        success = retriever.process_latest_nav(max_funds=None)
        
        if success:
            print("\n🎉 Latest NAV retrieval completed!")
            print("\nGenerated file:")
            print("  • qdii_latest_nav.csv - Latest NAV values with dates")
        else:
            print("\n❌ Latest NAV retrieval failed.")
    
    finally:
        retriever.disconnect_wind()

if __name__ == "__main__":
    main()
