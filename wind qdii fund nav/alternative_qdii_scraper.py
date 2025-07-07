"""
Alternative QDII Fund Data Scraper
This script scrapes QDII fund data from public sources when WindPy is not available.
"""

import pandas as pd
import requests
import json
import time
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
import re

class AlternativeQDIIFundScraper:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
    
    def get_qdii_fund_list_from_eastmoney(self):
        """Scrape QDII fund list from East Money"""
        try:
            # East Money QDII fund list API
            url = "http://fund.eastmoney.com/js/fundcode_search.js"
            response = self.session.get(url)
            
            if response.status_code == 200:
                # Extract fund data from JavaScript
                content = response.text
                # Find the array data
                start = content.find('[')
                end = content.rfind(']') + 1
                fund_data_str = content[start:end]
                
                # Parse the fund data
                fund_data = eval(fund_data_str)  # Note: eval is used here for simplicity, consider using ast.literal_eval for safety
                
                # Filter for QDII funds
                qdii_funds = []
                for fund in fund_data:
                    if len(fund) >= 4:
                        fund_code, fund_name, fund_type, fund_category = fund[0], fund[1], fund[2], fund[3]
                        # Check if it's a QDII fund
                        if 'QDII' in fund_name or 'QDII' in fund_type or 'QDII' in fund_category:
                            qdii_funds.append({
                                'fund_code': fund_code,
                                'fund_name': fund_name,
                                'fund_type': fund_type,
                                'fund_category': fund_category
                            })
                
                print(f"Found {len(qdii_funds)} QDII funds from East Money")
                return pd.DataFrame(qdii_funds)
            
        except Exception as e:
            print(f"Error scraping from East Money: {str(e)}")
        
        return None
    
    def get_qdii_fund_list_from_ttjj(self):
        """Scrape QDII fund list from TianTian Fund (ttjj.com)"""
        try:
            # TianTian Fund QDII category page
            url = "http://fund.eastmoney.com/QDII_jzzzl.html"
            response = self.session.get(url)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Find fund table
                fund_table = soup.find('table', {'id': 'oTable'})
                if fund_table:
                    funds = []
                    rows = fund_table.find('tbody').find_all('tr')
                    
                    for row in rows:
                        cells = row.find_all('td')
                        if len(cells) >= 3:
                            fund_code = cells[0].text.strip()
                            fund_name = cells[1].text.strip()
                            funds.append({
                                'fund_code': fund_code,
                                'fund_name': fund_name,
                                'source': 'ttjj'
                            })
                    
                    print(f"Found {len(funds)} QDII funds from TianTian Fund")
                    return pd.DataFrame(funds)
            
        except Exception as e:
            print(f"Error scraping from TianTian Fund: {str(e)}")
        
        return None
    
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
                            'date': item.get('FSRQ'),
                            'nav': float(item.get('DWJZ', 0)) if item.get('DWJZ') else None,
                            'accumulated_nav': float(item.get('LJJZ', 0)) if item.get('LJJZ') else None,
                            'daily_growth_rate': item.get('JZZZL')
                        })
                    
                    df = pd.DataFrame(nav_data)
                    df['date'] = pd.to_datetime(df['date'])
                    df = df.sort_values('date')
                    
                    return df
            
        except Exception as e:
            print(f"Error getting NAV data for {fund_code}: {str(e)}")
        
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
    
    def get_all_nav_data(self, fund_codes, max_funds=None):
        """Get NAV data for all funds"""
        nav_data = {}
        
        if max_funds:
            fund_codes = fund_codes[:max_funds]
        
        for i, fund_code in enumerate(fund_codes):
            print(f"Retrieving NAV data for {fund_code} ({i+1}/{len(fund_codes)})")
            
            nav_df = self.get_fund_nav_from_eastmoney(fund_code)
            nav_data[fund_code] = nav_df
            
            # Add delay to avoid being blocked
            time.sleep(1)
        
        return nav_data
    
    def save_nav_data_to_csv(self, nav_data, filename_prefix="qdii_nav_data"):
        """Save NAV data to CSV files"""
        try:
            # Save individual fund NAV data
            for fund_code, data in nav_data.items():
                if data is not None and not data.empty:
                    filename = f"{filename_prefix}_{fund_code}.csv"
                    data.to_csv(filename, index=False, encoding='utf-8-sig')
                    print(f"NAV data for {fund_code} saved to {filename}")
            
            # Create a combined NAV data file
            combined_data = []
            for fund_code, data in nav_data.items():
                if data is not None and not data.empty:
                    data_copy = data.copy()
                    data_copy['fund_code'] = fund_code
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

def main():
    """Main function to execute the alternative QDII fund data scraping"""
    scraper = AlternativeQDIIFundScraper()
    
    print("Alternative QDII Fund Data Scraper")
    print("=" * 50)
    
    # Step 1: Get QDII fund list
    print("Step 1: Retrieving QDII fund list from public sources...")
    
    # Try East Money first
    fund_list = scraper.get_qdii_fund_list_from_eastmoney()
    
    # If East Money fails, try TianTian Fund
    if fund_list is None or fund_list.empty:
        print("Trying TianTian Fund...")
        fund_list = scraper.get_qdii_fund_list_from_ttjj()
    
    if fund_list is None or fund_list.empty:
        print("Failed to retrieve QDII fund list from public sources.")
        return
    
    # Step 2: Save fund list to CSV
    print(f"\nStep 2: Saving {len(fund_list)} funds to CSV...")
    scraper.save_fund_list_to_csv(fund_list)
    
    # Step 3: Get NAV data (limit to first 10 for testing)
    print("\nStep 3: Retrieving NAV data...")
    fund_codes = fund_list['fund_code'].tolist()
    
    # Limit to first 5 funds for testing (remove this limit for full data)
    nav_data = scraper.get_all_nav_data(fund_codes, max_funds=5)
    
    # Step 4: Save NAV data
    print("\nStep 4: Saving NAV data to CSV...")
    scraper.save_nav_data_to_csv(nav_data)
    
    print("\nAlternative data scraping completed!")

if __name__ == "__main__":
    main()
