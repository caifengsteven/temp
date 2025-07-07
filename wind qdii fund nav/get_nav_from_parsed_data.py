"""
Get NAV Data for Parsed QDII Funds
This script reads the parsed QDII fund data and retrieves NAV information.
"""

import pandas as pd
import requests
import json
import time
from datetime import datetime, timedelta
import re

class QDIINavRetriever:
    def __init__(self, csv_file="qdii_funds_parsed.csv"):
        self.csv_file = csv_file
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
    
    def read_fund_data(self):
        """Read the parsed fund data"""
        try:
            df = pd.read_csv(self.csv_file, encoding='utf-8-sig')
            print(f"Read {len(df)} funds from {self.csv_file}")
            return df
        except Exception as e:
            print(f"Error reading fund data: {e}")
            return None
    
    def extract_fund_codes(self, df):
        """Extract fund codes from the data"""
        # The fund codes are in the '代码' column
        if '代码' in df.columns:
            fund_codes = df['代码'].tolist()
        else:
            # Try the first column if '代码' not found
            fund_codes = df.iloc[:, 0].tolist()
        
        # Clean up fund codes - remove exchange suffix for API calls
        cleaned_codes = []
        for code in fund_codes:
            if isinstance(code, str):
                # Remove .SH, .SZ suffixes
                clean_code = re.sub(r'\.(SH|SZ)$', '', code)
                if clean_code.isdigit() and len(clean_code) == 6:
                    cleaned_codes.append(clean_code)
        
        print(f"Extracted {len(cleaned_codes)} valid fund codes")
        return cleaned_codes
    
    def get_fund_nav_from_eastmoney(self, fund_code, days=365):
        """Get NAV data for a specific fund from East Money"""
        try:
            # East Money fund NAV API
            url = f"http://api.fund.eastmoney.com/f10/lsjz"
            params = {
                'fundCode': fund_code,
                'pageIndex': 1,
                'pageSize': days,
                'startDate': '',
                'endDate': ''
            }
            
            response = self.session.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                if data.get('Data') and data['Data'].get('LSJZList'):
                    nav_list = data['Data']['LSJZList']
                    
                    nav_data = []
                    for item in nav_list:
                        nav_data.append({
                            'fund_code': fund_code,
                            'date': item.get('FSRQ'),
                            'nav': float(item.get('DWJZ', 0)) if item.get('DWJZ') else None,
                            'accumulated_nav': float(item.get('LJJZ', 0)) if item.get('LJJZ') else None,
                            'daily_growth_rate': item.get('JZZZL')
                        })
                    
                    df = pd.DataFrame(nav_data)
                    if not df.empty:
                        df['date'] = pd.to_datetime(df['date'])
                        df = df.sort_values('date')
                        return df
                    
                return None
            else:
                print(f"HTTP error {response.status_code} for fund {fund_code}")
                return None
            
        except Exception as e:
            print(f"Error getting NAV data for {fund_code}: {str(e)}")
            return None
    
    def get_etf_price_from_sina(self, fund_code_with_exchange):
        """Get ETF price data from Sina Finance (alternative source)"""
        try:
            # Convert fund code format for Sina API
            if fund_code_with_exchange.endswith('.SH'):
                sina_code = 'sh' + fund_code_with_exchange.replace('.SH', '')
            elif fund_code_with_exchange.endswith('.SZ'):
                sina_code = 'sz' + fund_code_with_exchange.replace('.SZ', '')
            else:
                return None
            
            url = f"http://hq.sinajs.cn/list={sina_code}"
            response = self.session.get(url, timeout=10)
            
            if response.status_code == 200:
                content = response.text
                if 'var hq_str_' in content:
                    # Parse Sina response
                    data_part = content.split('="')[1].split('";')[0]
                    fields = data_part.split(',')
                    
                    if len(fields) > 10:
                        return {
                            'fund_code': fund_code_with_exchange.replace('.SH', '').replace('.SZ', ''),
                            'date': datetime.now().strftime('%Y-%m-%d'),
                            'current_price': float(fields[3]) if fields[3] else None,
                            'open_price': float(fields[1]) if fields[1] else None,
                            'high_price': float(fields[4]) if fields[4] else None,
                            'low_price': float(fields[5]) if fields[5] else None,
                            'volume': float(fields[8]) if fields[8] else None
                        }
            
            return None
            
        except Exception as e:
            print(f"Error getting ETF price for {fund_code_with_exchange}: {str(e)}")
            return None
    
    def get_all_nav_data(self, fund_codes, original_df, max_funds=None):
        """Get NAV data for all funds"""
        nav_data_list = []
        etf_price_list = []
        
        if max_funds:
            fund_codes = fund_codes[:max_funds]
        
        total_funds = len(fund_codes)
        print(f"Processing {total_funds} funds for NAV data...")
        
        for i, fund_code in enumerate(fund_codes):
            print(f"Processing fund {fund_code} ({i+1}/{total_funds})")
            
            # Try to get NAV data first
            nav_df = self.get_fund_nav_from_eastmoney(fund_code)
            if nav_df is not None and not nav_df.empty:
                nav_data_list.append(nav_df)
                print(f"  Got {len(nav_df)} NAV records")
            else:
                # Try to get current ETF price as alternative
                original_code = original_df[original_df['代码'].str.contains(fund_code, na=False)]['代码'].iloc[0] if len(original_df[original_df['代码'].str.contains(fund_code, na=False)]) > 0 else None
                if original_code:
                    etf_price = self.get_etf_price_from_sina(original_code)
                    if etf_price:
                        etf_price_list.append(etf_price)
                        print(f"  Got current ETF price: {etf_price['current_price']}")
                    else:
                        print(f"  No data found")
                else:
                    print(f"  No data found")
            
            # Add delay to avoid being blocked
            time.sleep(1)
        
        # Combine NAV data
        combined_nav_data = None
        if nav_data_list:
            combined_nav_data = pd.concat(nav_data_list, ignore_index=True)
        
        # Combine ETF price data
        combined_etf_data = None
        if etf_price_list:
            combined_etf_data = pd.DataFrame(etf_price_list)
        
        return combined_nav_data, combined_etf_data
    
    def save_data(self, nav_data, etf_data):
        """Save NAV and ETF data to files"""
        saved_files = []
        
        if nav_data is not None and not nav_data.empty:
            nav_file = "qdii_nav_data_complete.csv"
            nav_data.to_csv(nav_file, index=False, encoding='utf-8-sig')
            print(f"NAV data saved to {nav_file}")
            print(f"  Total NAV records: {len(nav_data)}")
            print(f"  Date range: {nav_data['date'].min()} to {nav_data['date'].max()}")
            print(f"  Unique funds with NAV data: {nav_data['fund_code'].nunique()}")
            saved_files.append(nav_file)
        
        if etf_data is not None and not etf_data.empty:
            etf_file = "qdii_etf_current_prices.csv"
            etf_data.to_csv(etf_file, index=False, encoding='utf-8-sig')
            print(f"ETF price data saved to {etf_file}")
            print(f"  Total ETF records: {len(etf_data)}")
            print(f"  Unique ETFs with price data: {etf_data['fund_code'].nunique()}")
            saved_files.append(etf_file)
        
        return saved_files
    
    def process_all_funds(self, max_funds=None):
        """Main processing function"""
        print("QDII NAV Data Retriever")
        print("=" * 50)
        
        # Step 1: Read fund data
        print("Step 1: Reading parsed fund data...")
        df = self.read_fund_data()
        if df is None:
            return False
        
        # Step 2: Extract fund codes
        print("\nStep 2: Extracting fund codes...")
        fund_codes = self.extract_fund_codes(df)
        if not fund_codes:
            print("No valid fund codes found")
            return False
        
        # Step 3: Get NAV data
        print(f"\nStep 3: Retrieving NAV data for {len(fund_codes)} funds...")
        nav_data, etf_data = self.get_all_nav_data(fund_codes, df, max_funds=max_funds)
        
        # Step 4: Save data
        print(f"\nStep 4: Saving data...")
        saved_files = self.save_data(nav_data, etf_data)
        
        if saved_files:
            print(f"\nProcessing completed! Generated files:")
            for file in saved_files:
                print(f"  - {file}")
            return True
        else:
            print("No data was saved")
            return False

def main():
    """Main function"""
    retriever = QDIINavRetriever()
    
    # Process first 20 funds for testing (remove max_funds for all funds)
    success = retriever.process_all_funds(max_funds=20)
    
    if success:
        print("\nNAV data retrieval completed successfully!")
    else:
        print("\nNAV data retrieval failed.")

if __name__ == "__main__":
    main()
