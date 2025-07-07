"""
Get Underlying Index for QDII Funds
This script retrieves the underlying/tracking index for each QDII fund using Wind API.
"""

import pandas as pd
from datetime import datetime

try:
    from WindPy import w
    WIND_AVAILABLE = True
except ImportError:
    WIND_AVAILABLE = False
    print("WindPy not available.")

class QDIIIndexRetriever:
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
    
    def get_underlying_index_info(self, wind_codes):
        """Get underlying index information for QDII funds"""
        if not self.connected:
            return None
            
        try:
            print("Getting underlying index information from Wind...")
            
            # Try different field combinations for index information
            index_field_sets = [
                # Basic index fields
                ["fund_trackindexcode", "fund_trackindexname"],
                ["fund_benchmarkcode", "fund_benchmarkname"], 
                ["fund_investobj", "fund_investscope"],
                # Alternative fields
                ["fund_trackindexcode"],
                ["fund_benchmarkcode"],
                ["fund_investobj"],
                # ETF specific fields
                ["etf_trackindexcode", "etf_trackindexname"],
                ["etf_benchmarkcode", "etf_benchmarkname"],
            ]
            
            codes = [item['wind_code'] for item in wind_codes]
            
            for field_set in index_field_sets:
                try:
                    print(f"  Trying fields: {field_set}")
                    result = w.wss(codes, field_set, usedf=True)
                    
                    if isinstance(result, tuple) and len(result) >= 2:
                        error_code = result[0]
                        data = result[1]
                        
                        if error_code == 0 and isinstance(data, pd.DataFrame) and not data.empty:
                            # Check if we got meaningful data (not all NaN)
                            non_null_count = data.notna().sum().sum()
                            if non_null_count > 0:
                                print(f"  ✓ Success with {field_set}! Got data for {len(data)} funds")
                                print(f"    Non-null values: {non_null_count}")
                                
                                # Add field information
                                data['fields_used'] = str(field_set)
                                data['retrieval_date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                                
                                return data
                            else:
                                print(f"  ✗ {field_set} returned all null values")
                        else:
                            print(f"  ✗ {field_set} failed with error code: {error_code}")
                            
                except Exception as e:
                    print(f"  ✗ {field_set} error: {str(e)}")
                    continue
            
            print("  All field sets failed or returned no data")
            return None
            
        except Exception as e:
            print(f"Error getting index info: {str(e)}")
            return None
    
    def get_additional_fund_info(self, wind_codes):
        """Get additional fund information that might contain index details"""
        if not self.connected:
            return None
            
        try:
            print("Getting additional fund information...")
            
            # Try to get more comprehensive fund information
            additional_fields = [
                ["sec_name", "fund_fullname", "fund_type"],
                ["fund_investtype", "fund_investobj", "fund_investscope"],
                ["fund_setupdate", "fund_corp", "fund_manager"],
            ]
            
            codes = [item['wind_code'] for item in wind_codes]
            all_additional_data = {}
            
            for field_set in additional_fields:
                try:
                    print(f"  Trying additional fields: {field_set}")
                    result = w.wss(codes, field_set, usedf=True)
                    
                    if isinstance(result, tuple) and len(result) >= 2:
                        error_code = result[0]
                        data = result[1]
                        
                        if error_code == 0 and isinstance(data, pd.DataFrame) and not data.empty:
                            non_null_count = data.notna().sum().sum()
                            if non_null_count > 0:
                                print(f"    ✓ Got {non_null_count} non-null values")
                                for col in data.columns:
                                    all_additional_data[col] = data[col]
                            
                except Exception as e:
                    print(f"    ✗ Error: {str(e)}")
                    continue
            
            if all_additional_data:
                additional_df = pd.DataFrame(all_additional_data)
                additional_df.index = codes
                return additional_df
            
            return None
            
        except Exception as e:
            print(f"Error getting additional info: {str(e)}")
            return None
    
    def process_all_funds(self, max_funds=None):
        """Get underlying index information for all funds"""
        print("QDII Fund Underlying Index Retriever")
        print("=" * 60)
        
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
        
        # Step 4: Get underlying index information
        print(f"\nStep 4: Getting underlying index information...")
        index_data = self.get_underlying_index_info(wind_codes)
        
        # Step 5: Get additional fund information
        print(f"\nStep 5: Getting additional fund information...")
        additional_data = self.get_additional_fund_info(wind_codes)
        
        # Step 6: Combine and save data
        print(f"\nStep 6: Combining and saving data...")
        
        # Create fund name mapping
        fund_name_map = {item['wind_code']: item['fund_name'] for item in wind_codes}
        
        # Start with basic fund info
        combined_data = pd.DataFrame(index=[item['wind_code'] for item in wind_codes])
        combined_data['fund_name'] = combined_data.index.map(fund_name_map)
        
        # Add index data if available
        if index_data is not None:
            for col in index_data.columns:
                if col not in ['fields_used', 'retrieval_date']:
                    combined_data[f'index_{col}'] = index_data[col]
            combined_data['index_fields_used'] = index_data['fields_used'] if 'fields_used' in index_data.columns else 'Unknown'
        
        # Add additional data if available
        if additional_data is not None:
            for col in additional_data.columns:
                combined_data[f'fund_{col}'] = additional_data[col]
        
        # Add retrieval timestamp
        combined_data['retrieval_date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Save results
        output_file = "qdii_underlying_index_info.csv"
        combined_data.to_csv(output_file, encoding='utf-8-sig')
        
        print(f"\n🎉 SUCCESS! Index information saved to {output_file}")
        print(f"📊 Summary:")
        print(f"   • Total funds processed: {len(combined_data)}")
        
        # Show available columns
        index_cols = [col for col in combined_data.columns if col.startswith('index_')]
        fund_cols = [col for col in combined_data.columns if col.startswith('fund_')]
        
        print(f"   • Index-related columns: {len(index_cols)}")
        if index_cols:
            print(f"     {index_cols}")
        
        print(f"   • Fund-related columns: {len(fund_cols)}")
        if fund_cols:
            print(f"     {fund_cols[:5]}{'...' if len(fund_cols) > 5 else ''}")
        
        # Show sample data
        print(f"\n📋 Sample data:")
        display_cols = ['fund_name'] + index_cols[:3] + fund_cols[:2]
        sample_data = combined_data[display_cols].head(10)
        print(sample_data.to_string())
        
        # Show summary of index data availability
        if index_cols:
            print(f"\n📈 Index data availability:")
            for col in index_cols:
                non_null_count = combined_data[col].notna().sum()
                print(f"   • {col}: {non_null_count}/{len(combined_data)} funds ({non_null_count/len(combined_data)*100:.1f}%)")
        
        return True
    
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
    retriever = QDIIIndexRetriever()
    
    try:
        # Process ALL QDII funds
        success = retriever.process_all_funds(max_funds=None)
        
        if success:
            print("\n🎉 Underlying index retrieval completed!")
            print("\nGenerated file:")
            print("  • qdii_underlying_index_info.csv - Index and fund information")
        else:
            print("\n❌ Underlying index retrieval failed.")
    
    finally:
        retriever.disconnect_wind()

if __name__ == "__main__":
    main()
