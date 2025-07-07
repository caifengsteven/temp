"""
Daily QDII Fund Data Retriever
Run this script daily to get latest NAV, close prices, and underlying index info for all QDII funds.

Usage: python daily_qdii_data_retriever.py
Output: qdii_daily_data_YYYYMMDD.csv
"""

import pandas as pd
from datetime import datetime
import os

try:
    from WindPy import w
    WIND_AVAILABLE = True
except ImportError:
    WIND_AVAILABLE = False
    print("❌ WindPy not available. Please install WindPy library.")

class DailyQDIIRetriever:
    def __init__(self, fund_list_file="qdii_funds_parsed.csv"):
        self.fund_list_file = fund_list_file
        self.connected = False
        self.today = datetime.now().strftime("%Y%m%d")
        self.output_file = f"qdii_daily_data_{self.today}.csv"
        
    def connect_wind(self):
        """Connect to Wind terminal"""
        if not WIND_AVAILABLE:
            print("❌ Error: WindPy library is not installed.")
            return False
            
        try:
            w.start()
            self.connected = True
            print("✅ Successfully connected to Wind terminal.")
            return True
        except Exception as e:
            print(f"❌ Error connecting to Wind: {str(e)}")
            return False
    
    def load_fund_list(self):
        """Load QDII fund list"""
        try:
            df = pd.read_csv(self.fund_list_file, encoding='utf-8-sig')
            
            # Extract fund codes and names
            fund_list = []
            if '代码' in df.columns:
                for _, row in df.iterrows():
                    fund_list.append({
                        'wind_code': row['代码'],
                        'fund_name': row['名称'] if '名称' in df.columns else ''
                    })
            
            print(f"📋 Loaded {len(fund_list)} QDII funds from {self.fund_list_file}")
            return fund_list
            
        except Exception as e:
            print(f"❌ Error loading fund list: {str(e)}")
            return None
    
    def get_latest_nav_data(self, fund_codes):
        """Get latest NAV data for all funds"""
        if not self.connected:
            return None
            
        try:
            print("📊 Getting latest NAV data...")
            codes = [item['wind_code'] for item in fund_codes]
            
            result = w.wss(codes, "nav", usedf=True)
            
            if isinstance(result, tuple) and len(result) >= 2:
                error_code = result[0]
                data = result[1]
                
                if error_code == 0 and isinstance(data, pd.DataFrame) and not data.empty:
                    print(f"✅ Got NAV data for {len(data)} funds")
                    return data
                else:
                    print(f"❌ NAV query failed with error code: {error_code}")
            
            return None
            
        except Exception as e:
            print(f"❌ Error getting NAV data: {str(e)}")
            return None
    
    def get_latest_close_data(self, fund_codes):
        """Get latest closing prices for all funds"""
        if not self.connected:
            return None
            
        try:
            print("📊 Getting latest closing prices...")
            codes = [item['wind_code'] for item in fund_codes]
            
            result = w.wss(codes, ["close", "volume", "amt"], usedf=True)
            
            if isinstance(result, tuple) and len(result) >= 2:
                error_code = result[0]
                data = result[1]
                
                if error_code == 0 and isinstance(data, pd.DataFrame) and not data.empty:
                    print(f"✅ Got closing price data for {len(data)} funds")
                    return data
                else:
                    print(f"❌ Close price query failed with error code: {error_code}")
            
            return None
            
        except Exception as e:
            print(f"❌ Error getting closing price data: {str(e)}")
            return None
    
    def get_underlying_index_data(self, fund_codes):
        """Get underlying index information for all funds"""
        if not self.connected:
            return None
            
        try:
            print("📊 Getting underlying index data...")
            codes = [item['wind_code'] for item in fund_codes]
            
            result = w.wss(codes, ["fund_trackindexcode", "fund_trackindexname"], usedf=True)
            
            if isinstance(result, tuple) and len(result) >= 2:
                error_code = result[0]
                data = result[1]
                
                if error_code == 0 and isinstance(data, pd.DataFrame) and not data.empty:
                    print(f"✅ Got index data for {len(data)} funds")
                    return data
                else:
                    print(f"❌ Index query failed with error code: {error_code}")
            
            return None
            
        except Exception as e:
            print(f"❌ Error getting index data: {str(e)}")
            return None
    
    def combine_and_save_data(self, fund_codes, nav_data, close_data, index_data):
        """Combine all data and save to CSV"""
        try:
            print("🔄 Combining all data...")
            
            # Create base dataframe with fund information
            fund_df = pd.DataFrame(fund_codes)
            fund_df = fund_df.set_index('wind_code')
            
            # Add NAV data
            if nav_data is not None:
                fund_df['latest_nav'] = nav_data['NAV']
            else:
                fund_df['latest_nav'] = None
            
            # Add closing price data
            if close_data is not None:
                fund_df['latest_close'] = close_data['CLOSE']
                fund_df['latest_volume'] = close_data['VOLUME'] if 'VOLUME' in close_data.columns else None
                fund_df['latest_amount'] = close_data['AMT'] if 'AMT' in close_data.columns else None
            else:
                fund_df['latest_close'] = None
                fund_df['latest_volume'] = None
                fund_df['latest_amount'] = None
            
            # Add index data
            if index_data is not None:
                fund_df['index_code'] = index_data['FUND_TRACKINDEXCODE'] if 'FUND_TRACKINDEXCODE' in index_data.columns else None
                fund_df['index_name'] = index_data['FUND_TRACKINDEXNAME'] if 'FUND_TRACKINDEXNAME' in index_data.columns else None
            else:
                fund_df['index_code'] = None
                fund_df['index_name'] = None
            
            # Calculate price-NAV difference
            fund_df['price_nav_diff'] = fund_df['latest_close'] - fund_df['latest_nav']
            fund_df['price_nav_diff_pct'] = (fund_df['price_nav_diff'] / fund_df['latest_nav'] * 100).round(4)
            
            # Add metadata
            fund_df['data_date'] = self.today
            fund_df['retrieval_timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            # Reset index to include wind_code as column
            fund_df = fund_df.reset_index()
            
            # Reorder columns for better readability
            column_order = [
                'wind_code', 'fund_name', 
                'latest_nav', 'latest_close', 'price_nav_diff', 'price_nav_diff_pct',
                'latest_volume', 'latest_amount',
                'index_code', 'index_name',
                'data_date', 'retrieval_timestamp'
            ]
            
            fund_df = fund_df[column_order]
            
            # Save to CSV
            fund_df.to_csv(self.output_file, index=False, encoding='utf-8-sig')
            
            print(f"💾 Data saved to: {self.output_file}")
            return fund_df
            
        except Exception as e:
            print(f"❌ Error combining and saving data: {str(e)}")
            return None
    
    def print_summary(self, final_data):
        """Print summary of retrieved data"""
        if final_data is None:
            return
            
        print(f"\n" + "="*60)
        print(f"📊 DAILY QDII DATA SUMMARY - {self.today}")
        print("="*60)
        
        total_funds = len(final_data)
        nav_count = final_data['latest_nav'].notna().sum()
        close_count = final_data['latest_close'].notna().sum()
        index_count = final_data['index_code'].notna().sum()
        
        print(f"📈 Data Coverage:")
        print(f"   • Total funds: {total_funds}")
        print(f"   • Funds with NAV: {nav_count} ({nav_count/total_funds*100:.1f}%)")
        print(f"   • Funds with Close price: {close_count} ({close_count/total_funds*100:.1f}%)")
        print(f"   • Funds with Index info: {index_count} ({index_count/total_funds*100:.1f}%)")
        
        # Price vs NAV analysis
        both_data = final_data.dropna(subset=['latest_nav', 'latest_close'])
        if len(both_data) > 0:
            avg_diff = both_data['price_nav_diff_pct'].mean()
            print(f"\n💰 Price vs NAV Analysis ({len(both_data)} funds):")
            print(f"   • Average difference: {avg_diff:.4f}%")
            print(f"   • Max positive diff: {both_data['price_nav_diff_pct'].max():.4f}%")
            print(f"   • Max negative diff: {both_data['price_nav_diff_pct'].min():.4f}%")
        
        # Show sample data
        print(f"\n📋 Sample Data (first 5 funds):")
        sample_cols = ['wind_code', 'fund_name', 'latest_nav', 'latest_close', 'index_name']
        print(final_data[sample_cols].head().to_string(index=False))
        
        print(f"\n📁 Output file: {self.output_file}")
    
    def run_daily_update(self):
        """Main function to run daily data update"""
        print(f"🚀 DAILY QDII FUND DATA RETRIEVER")
        print(f"📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*60)
        
        # Step 1: Connect to Wind
        if not self.connect_wind():
            return False
        
        # Step 2: Load fund list
        fund_codes = self.load_fund_list()
        if not fund_codes:
            return False
        
        # Step 3: Get all data
        nav_data = self.get_latest_nav_data(fund_codes)
        close_data = self.get_latest_close_data(fund_codes)
        index_data = self.get_underlying_index_data(fund_codes)
        
        # Step 4: Combine and save
        final_data = self.combine_and_save_data(fund_codes, nav_data, close_data, index_data)
        
        # Step 5: Print summary
        self.print_summary(final_data)
        
        return final_data is not None
    
    def disconnect_wind(self):
        """Disconnect from Wind terminal"""
        if self.connected:
            try:
                w.stop()
                self.connected = False
                print("🔌 Disconnected from Wind terminal.")
            except:
                pass

def main():
    """Main function"""
    retriever = DailyQDIIRetriever()
    
    try:
        success = retriever.run_daily_update()
        
        if success:
            print(f"\n🎉 Daily data retrieval completed successfully!")
            print(f"📊 Check the output file: {retriever.output_file}")
        else:
            print(f"\n❌ Daily data retrieval failed.")
    
    finally:
        retriever.disconnect_wind()

if __name__ == "__main__":
    main()
