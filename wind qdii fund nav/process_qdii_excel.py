"""
Process QDII Fund Excel File and Retrieve NAV Data
This script reads the ETF_qdii.xlsx file and retrieves NAV data for the funds.
"""

import pandas as pd
import requests
import json
import time
from datetime import datetime, timedelta
import os

class QDIIExcelProcessor:
    def __init__(self, excel_file="ETF_qdii.xlsx"):
        self.excel_file = excel_file
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
    
    def read_excel_file(self):
        """Read the QDII fund list from Excel file"""
        try:
            # Try different methods to read the Excel file
            methods = [
                {'engine': 'openpyxl', 'sheet_name': 0},
                {'engine': 'openpyxl', 'sheet_name': None},
                {'engine': 'xlrd', 'sheet_name': 0},
                {'engine': None, 'sheet_name': 0}  # Let pandas choose
            ]

            for method in methods:
                try:
                    print(f"Trying to read Excel file with method: {method}")
                    if method['engine']:
                        df = pd.read_excel(self.excel_file, engine=method['engine'], sheet_name=method['sheet_name'])
                    else:
                        df = pd.read_excel(self.excel_file, sheet_name=method['sheet_name'])

                    # If sheet_name=None, we get a dict of DataFrames
                    if isinstance(df, dict):
                        print(f"Found {len(df)} sheets: {list(df.keys())}")
                        # Use the first sheet
                        sheet_name = list(df.keys())[0]
                        df = df[sheet_name]
                        print(f"Using sheet: {sheet_name}")

                    print(f"Successfully read Excel file: {self.excel_file}")
                    print(f"Shape: {df.shape}")
                    print(f"Columns: {list(df.columns)}")
                    print("\nFirst few rows:")
                    print(df.head())
                    return df

                except Exception as e:
                    print(f"Failed with method {method}: {str(e)}")
                    continue

            print("All methods failed.")
            return None

        except Exception as e:
            print(f"Error reading file: {str(e)}")
            return None
    
    def identify_fund_code_column(self, df):
        """Identify which column contains the fund codes"""
        possible_columns = ['fund_code', 'code', '基金代码', '代码', 'Fund Code', 'Code', 'symbol', 'ticker']
        
        for col in df.columns:
            col_lower = str(col).lower()
            if any(keyword in col_lower for keyword in ['code', '代码', 'symbol', 'ticker']):
                print(f"Found potential fund code column: {col}")
                return col
        
        # If no obvious column found, use the first column
        print(f"No obvious fund code column found. Using first column: {df.columns[0]}")
        return df.columns[0]
    
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
            
            response = self.session.get(url, params=params)
            
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
                else:
                    print(f"No NAV data found for fund {fund_code}")
                    return None
            else:
                print(f"HTTP error {response.status_code} for fund {fund_code}")
                return None
            
        except Exception as e:
            print(f"Error getting NAV data for {fund_code}: {str(e)}")
            return None
    
    def get_all_nav_data(self, fund_codes, max_funds=None):
        """Get NAV data for all funds"""
        nav_data_list = []
        
        if max_funds:
            fund_codes = fund_codes[:max_funds]
        
        total_funds = len(fund_codes)
        print(f"Processing {total_funds} funds for NAV data...")
        
        for i, fund_code in enumerate(fund_codes):
            print(f"Retrieving NAV data for {fund_code} ({i+1}/{total_funds})")
            
            nav_df = self.get_fund_nav_from_eastmoney(fund_code)
            if nav_df is not None and not nav_df.empty:
                nav_data_list.append(nav_df)
            
            # Add delay to avoid being blocked
            time.sleep(1)
        
        if nav_data_list:
            combined_nav_data = pd.concat(nav_data_list, ignore_index=True)
            return combined_nav_data
        else:
            return None
    
    def save_nav_data(self, nav_data, output_file="qdii_nav_data_from_excel.csv"):
        """Save NAV data to CSV file"""
        try:
            if nav_data is not None and not nav_data.empty:
                nav_data.to_csv(output_file, index=False, encoding='utf-8-sig')
                print(f"NAV data saved to {output_file}")
                print(f"Total records: {len(nav_data)}")
                print(f"Date range: {nav_data['date'].min()} to {nav_data['date'].max()}")
                print(f"Unique funds: {nav_data['fund_code'].nunique()}")
                return True
            else:
                print("No NAV data to save.")
                return False
        except Exception as e:
            print(f"Error saving NAV data: {str(e)}")
            return False
    
    def process_excel_and_get_nav(self, max_funds=None):
        """Main processing function"""
        print("QDII Excel Processor")
        print("=" * 50)
        
        # Step 1: Read Excel file
        print("Step 1: Reading Excel file...")
        df = self.read_excel_file()
        
        if df is None or df.empty:
            print("Failed to read Excel file or file is empty.")
            return False
        
        # Step 2: Identify fund code column
        print("\nStep 2: Identifying fund code column...")
        fund_code_column = self.identify_fund_code_column(df)
        
        # Step 3: Extract fund codes
        print(f"\nStep 3: Extracting fund codes from column '{fund_code_column}'...")
        fund_codes = df[fund_code_column].dropna().astype(str).tolist()
        
        # Remove any non-numeric codes (clean up)
        fund_codes = [code.strip() for code in fund_codes if code.strip().isdigit() and len(code.strip()) == 6]
        
        print(f"Found {len(fund_codes)} valid fund codes")
        if len(fund_codes) > 0:
            print(f"Sample codes: {fund_codes[:5]}")
        
        if not fund_codes:
            print("No valid fund codes found in the Excel file.")
            return False
        
        # Step 4: Get NAV data
        print(f"\nStep 4: Retrieving NAV data...")
        nav_data = self.get_all_nav_data(fund_codes, max_funds=max_funds)
        
        # Step 5: Save NAV data
        print(f"\nStep 5: Saving NAV data...")
        if nav_data is not None:
            success = self.save_nav_data(nav_data)
            return success
        else:
            print("No NAV data retrieved.")
            return False

def main():
    """Main function"""
    processor = QDIIExcelProcessor()
    
    # Process all funds (remove max_funds parameter for full processing)
    success = processor.process_excel_and_get_nav(max_funds=10)  # Limit to 10 for testing
    
    if success:
        print("\nProcessing completed successfully!")
    else:
        print("\nProcessing failed.")

if __name__ == "__main__":
    main()
