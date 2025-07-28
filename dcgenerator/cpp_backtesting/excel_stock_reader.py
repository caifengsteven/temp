#!/usr/bin/env python3
"""
Read stock codes from Excel files and generate mapped stock list for DC testing.
"""

import pandas as pd
import os
import sys

def map_stock_code(stock_code):
    """
    Map stock code according to rules:
    - If starts with 6 (e.g., 600001) -> sh600001
    - Otherwise -> szxxxxxx
    """
    stock_str = str(stock_code).strip()
    
    # Remove any existing prefixes
    if stock_str.startswith('sh') or stock_str.startswith('sz'):
        stock_str = stock_str[2:]
    
    # Ensure 6-digit format
    stock_str = stock_str.zfill(6)
    
    if stock_str.startswith('6'):
        return f"sh{stock_str}"
    else:
        return f"sz{stock_str}"

def read_excel_files():
    """
    Read all Excel files in current directory and extract stock codes from column E.
    """
    excel_files = []
    stock_codes = set()
    
    # Find all Excel files
    for file in os.listdir('.'):
        if file.endswith(('.xlsx', '.xls')):
            excel_files.append(file)
    
    if not excel_files:
        print("No Excel files found in current directory!")
        return []
    
    print(f"Found {len(excel_files)} Excel files:")
    for file in excel_files:
        print(f"  - {file}")
    
    # Read each Excel file
    for file in excel_files:
        print(f"\nReading: {file}")
        try:
            # Read Excel file
            df = pd.read_excel(file)
            
            print(f"  Columns found: {list(df.columns)}")
            print(f"  Total rows: {len(df)}")
            
            # Check if column E exists (index 4)
            if len(df.columns) > 4:
                column_e = df.iloc[:, 4]  # Column E (0-indexed, so index 4)
                print(f"  Column E name: {df.columns[4]}")
                
                # Extract stock codes
                file_codes = set()
                for value in column_e:
                    if pd.notna(value):
                        try:
                            # Convert to string and clean
                            code_str = str(value).strip()
                            if code_str and code_str != 'nan':
                                # Remove decimal points if present
                                if '.' in code_str:
                                    code_str = code_str.split('.')[0]
                                
                                # Only process if it looks like a stock code
                                if code_str.isdigit() and len(code_str) >= 6:
                                    mapped_code = map_stock_code(code_str)
                                    file_codes.add(mapped_code)
                                    stock_codes.add(mapped_code)
                        except:
                            continue
                
                print(f"  Stock codes extracted: {len(file_codes)}")
                print(f"  Sample codes: {list(file_codes)[:5]}")
                
            else:
                print(f"  Warning: File has only {len(df.columns)} columns, column E not found")
                
        except Exception as e:
            print(f"  Error reading {file}: {e}")
    
    return sorted(list(stock_codes))

def save_stock_list(stock_codes):
    """
    Save the mapped stock codes to files.
    """
    # Save to text file for C++ program
    with open('excel_stocks.txt', 'w') as f:
        for code in stock_codes:
            f.write(f"{code}\n")
    
    # Save detailed mapping for reference
    with open('stock_mapping_details.txt', 'w') as f:
        f.write("Stock Code Mapping Details\n")
        f.write("=" * 50 + "\n")
        f.write("Mapping Rules:\n")
        f.write("- Codes starting with 6 (e.g., 600001) -> sh600001\n")
        f.write("- Other codes (e.g., 000001) -> sz000001\n")
        f.write("\n")
        f.write(f"Total mapped codes: {len(stock_codes)}\n")
        f.write("\n")
        f.write("Mapped Stock Codes:\n")
        f.write("-" * 20 + "\n")
        
        sh_codes = [code for code in stock_codes if code.startswith('sh')]
        sz_codes = [code for code in stock_codes if code.startswith('sz')]
        
        f.write(f"\nShanghai Exchange (sh): {len(sh_codes)} codes\n")
        for code in sh_codes:
            f.write(f"  {code}\n")
        
        f.write(f"\nShenzhen Exchange (sz): {len(sz_codes)} codes\n")
        for code in sz_codes:
            f.write(f"  {code}\n")
    
    print(f"\nFiles saved:")
    print(f"  - excel_stocks.txt: {len(stock_codes)} stock codes for testing")
    print(f"  - stock_mapping_details.txt: Detailed mapping information")

def main():
    print("=== Excel Stock Code Reader ===")
    print("Reading stock codes from Excel files in current directory...")
    print("Looking for codes in column E and applying mapping rules.")
    print()
    
    # Read Excel files
    stock_codes = read_excel_files()
    
    if not stock_codes:
        print("No valid stock codes found!")
        return
    
    print(f"\n=== SUMMARY ===")
    print(f"Total unique stock codes found: {len(stock_codes)}")
    
    # Count by exchange
    sh_count = len([code for code in stock_codes if code.startswith('sh')])
    sz_count = len([code for code in stock_codes if code.startswith('sz')])
    
    print(f"Shanghai Exchange (sh): {sh_count}")
    print(f"Shenzhen Exchange (sz): {sz_count}")
    
    print(f"\nFirst 10 mapped codes:")
    for i, code in enumerate(stock_codes[:10]):
        print(f"  {i+1}. {code}")
    
    if len(stock_codes) > 10:
        print(f"  ... and {len(stock_codes) - 10} more")
    
    # Save to files
    save_stock_list(stock_codes)
    
    print(f"\n=== NEXT STEPS ===")
    print(f"1. Review stock_mapping_details.txt to verify mappings")
    print(f"2. Run mass testing with: ./test_excel_stocks.bat")
    print(f"3. Monitor progress with: ./3_check_progress.bat")

if __name__ == "__main__":
    main()
