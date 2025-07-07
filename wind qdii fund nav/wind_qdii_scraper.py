"""
Wind QDII Fund Data Scraper
This script connects to Wind terminal and retrieves QDII fund data including NAV values.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sys

try:
    from WindPy import w
    WIND_AVAILABLE = True
except ImportError:
    WIND_AVAILABLE = False
    print("WindPy not available. Please install WindPy library.")

class WindQDIIFundScraper:
    def __init__(self):
        self.w = None
        self.connected = False
        
    def connect_wind(self):
        """Connect to Wind terminal"""
        if not WIND_AVAILABLE:
            print("Error: WindPy library is not installed.")
            print("Please install WindPy from Wind terminal or contact Wind support.")
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
    
    def get_qdii_fund_list(self):
        """Retrieve all QDII funds from Wind"""
        if not self.connected:
            print("Not connected to Wind terminal.")
            return None

        try:
            # Query for QDII funds using different approaches

            # Method 1: Try to get all funds and filter for QDII
            print("Trying to get all fund data...")
            result = self.w.wset("allfund", "field=wind_code,sec_name,fund_setupdate,fund_maturitydate,fund_type", usedf=True)

            if hasattr(result, 'ErrorCode') and result.ErrorCode == 0:
                all_funds = result.Data[0] if hasattr(result, 'Data') else result
                if all_funds is not None and not all_funds.empty:
                    # Filter for QDII funds
                    qdii_funds = all_funds[all_funds['sec_name'].str.contains('QDII|海外|全球|美国|欧洲|亚洲|港股|恒生', case=False, na=False)]
                    print(f"Retrieved {len(qdii_funds)} QDII funds from Wind (filtered from all funds).")
                    return qdii_funds

            # Method 2: Try sector constituent query
            print("Trying sector constituent query...")
            result = self.w.wset("sectorconstituent", "date=20241201;sectorid=1000006526000000", usedf=True)

            if hasattr(result, 'ErrorCode') and result.ErrorCode == 0:
                fund_list = result.Data[0] if hasattr(result, 'Data') else result
                if fund_list is not None and not fund_list.empty:
                    print(f"Retrieved {len(fund_list)} QDII funds from Wind (sector query).")
                    return fund_list

            # Method 3: Try fund screening
            print("Trying fund screening...")
            result = self.w.wset("fundscreener", "field=wind_code,sec_name;fund_type=混合型,股票型,债券型,货币型,QDII", usedf=True)

            if hasattr(result, 'ErrorCode') and result.ErrorCode == 0:
                all_funds = result.Data[0] if hasattr(result, 'Data') else result
                if all_funds is not None and not all_funds.empty:
                    # Filter for QDII funds
                    qdii_funds = all_funds[all_funds['sec_name'].str.contains('QDII|海外|全球|美国|欧洲|亚洲|港股|恒生', case=False, na=False)]
                    print(f"Retrieved {len(qdii_funds)} QDII funds from Wind (fund screener).")
                    return qdii_funds

            print("All Wind API methods failed. No QDII fund data retrieved.")
            return None

        except Exception as e:
            print(f"Error retrieving QDII fund list: {str(e)}")
            return None
    
    def save_fund_list_to_csv(self, fund_list, filename="qdii_fund_list.csv"):
        """Save fund list to CSV file"""
        try:
            if fund_list is not None and not fund_list.empty:
                fund_list.to_csv(filename, index=False, encoding='utf-8-sig')
                print(f"QDII fund list saved to {filename}")
                return True
            else:
                print("No fund list data to save.")
                return False
        except Exception as e:
            print(f"Error saving fund list to CSV: {str(e)}")
            return False
    
    def get_fund_nav_data(self, fund_codes, start_date=None, end_date=None):
        """Retrieve NAV data for given fund codes"""
        if not self.connected:
            print("Not connected to Wind terminal.")
            return None

        if start_date is None:
            start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")

        nav_data = {}

        try:
            for i, fund_code in enumerate(fund_codes):
                print(f"Retrieving NAV data for {fund_code} ({i+1}/{len(fund_codes)})")

                # Get NAV data
                result = self.w.wsd(fund_code, "nav", start_date, end_date, usedf=True)

                if hasattr(result, 'ErrorCode') and result.ErrorCode == 0:
                    nav_data[fund_code] = result.Data[0] if hasattr(result, 'Data') else result
                elif hasattr(result, 'Data') and result.Data is not None:
                    # Sometimes the result doesn't have ErrorCode but has data
                    nav_data[fund_code] = result.Data[0] if hasattr(result, 'Data') else result
                else:
                    error_code = getattr(result, 'ErrorCode', 'Unknown')
                    print(f"Error retrieving NAV for {fund_code}. Error code: {error_code}")
                    nav_data[fund_code] = None

        except Exception as e:
            print(f"Error retrieving NAV data: {str(e)}")
            return None

        return nav_data
    
    def save_nav_data_to_csv(self, nav_data, filename_prefix="qdii_nav_data"):
        """Save NAV data to CSV files"""
        try:
            # Save individual fund NAV data
            for fund_code, data in nav_data.items():
                if data is not None and not data.empty:
                    filename = f"{filename_prefix}_{fund_code}.csv"
                    data.to_csv(filename, encoding='utf-8-sig')
                    print(f"NAV data for {fund_code} saved to {filename}")
            
            # Create a combined NAV data file
            combined_data = []
            for fund_code, data in nav_data.items():
                if data is not None and not data.empty:
                    data_copy = data.copy()
                    data_copy['fund_code'] = fund_code
                    data_copy['date'] = data_copy.index
                    combined_data.append(data_copy)
            
            if combined_data:
                combined_df = pd.concat(combined_data, ignore_index=True)
                combined_filename = f"{filename_prefix}_combined.csv"
                combined_df.to_csv(combined_filename, index=False, encoding='utf-8-sig')
                print(f"Combined NAV data saved to {combined_filename}")
                
            return True
            
        except Exception as e:
            print(f"Error saving NAV data to CSV: {str(e)}")
            return False
    
    def disconnect_wind(self):
        """Disconnect from Wind terminal"""
        if self.connected and self.w:
            self.w.stop()
            self.connected = False
            print("Disconnected from Wind terminal.")

def main():
    """Main function to execute the QDII fund data scraping"""
    scraper = WindQDIIFundScraper()
    
    # Step 1: Connect to Wind
    print("Step 1: Connecting to Wind terminal...")
    if not scraper.connect_wind():
        print("Failed to connect to Wind terminal. Exiting.")
        return
    
    # Step 2: Get QDII fund list
    print("\nStep 2: Retrieving QDII fund list...")
    fund_list = scraper.get_qdii_fund_list()
    
    if fund_list is None or fund_list.empty:
        print("Failed to retrieve QDII fund list. Exiting.")
        scraper.disconnect_wind()
        return
    
    # Step 3: Save fund list to CSV
    print("\nStep 3: Saving fund list to CSV...")
    scraper.save_fund_list_to_csv(fund_list)
    
    # Step 4: Get NAV data for all funds
    print("\nStep 4: Retrieving NAV data for all QDII funds...")

    # Try different column names for fund codes
    fund_code_column = None
    for col in ['wind_code', 'code', 'fund_code', 'sec_code']:
        if col in fund_list.columns:
            fund_code_column = col
            break

    if fund_code_column is None:
        # Use the first column if no standard column found
        fund_code_column = fund_list.columns[0]
        print(f"Using first column '{fund_code_column}' as fund code column")

    fund_codes = fund_list[fund_code_column].tolist()

    # Limit to first 5 funds for testing (remove this limit for full data)
    fund_codes = fund_codes[:5]  # Remove this line for full data retrieval
    print(f"Processing {len(fund_codes)} funds for NAV data...")

    nav_data = scraper.get_fund_nav_data(fund_codes)
    
    # Step 5: Save NAV data to CSV
    print("\nStep 5: Saving NAV data to CSV...")
    if nav_data:
        scraper.save_nav_data_to_csv(nav_data)
    
    # Disconnect from Wind
    scraper.disconnect_wind()
    print("\nData scraping completed successfully!")

if __name__ == "__main__":
    main()
