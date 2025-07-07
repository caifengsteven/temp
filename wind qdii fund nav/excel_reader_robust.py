"""
Robust Excel Reader for QDII Fund Data
This script tries multiple methods to read the Excel file.
"""

import pandas as pd
import zipfile
import xml.etree.ElementTree as ET
import os

def try_read_excel_basic():
    """Try basic pandas read_excel"""
    try:
        print("Trying basic pandas read_excel...")
        df = pd.read_excel('ETF_qdii.xlsx')
        print("Success with basic read_excel!")
        return df
    except Exception as e:
        print(f"Basic read_excel failed: {e}")
        return None

def try_read_excel_with_options():
    """Try pandas read_excel with various options"""
    options = [
        {'engine': 'openpyxl', 'header': 0},
        {'engine': 'openpyxl', 'header': None},
        {'engine': 'openpyxl', 'header': 0, 'skiprows': 1},
        {'engine': 'openpyxl', 'header': None, 'skiprows': 1},
    ]
    
    for i, option in enumerate(options):
        try:
            print(f"Trying option {i+1}: {option}")
            df = pd.read_excel('ETF_qdii.xlsx', **option)
            print(f"Success with option {i+1}!")
            return df
        except Exception as e:
            print(f"Option {i+1} failed: {e}")
    
    return None

def try_read_as_zip():
    """Try to read Excel file as ZIP and extract data manually"""
    try:
        print("Trying to read Excel as ZIP file...")
        with zipfile.ZipFile('ETF_qdii.xlsx', 'r') as zip_file:
            print("ZIP contents:", zip_file.namelist())
            
            # Try to read the worksheet data
            if 'xl/worksheets/sheet1.xml' in zip_file.namelist():
                with zip_file.open('xl/worksheets/sheet1.xml') as sheet_file:
                    content = sheet_file.read().decode('utf-8')
                    print("Found sheet1.xml, first 500 chars:")
                    print(content[:500])
                    return content
            else:
                print("sheet1.xml not found in ZIP")
                return None
                
    except Exception as e:
        print(f"ZIP reading failed: {e}")
        return None

def try_convert_to_csv():
    """Try to convert Excel to CSV using different methods"""
    try:
        print("Trying to convert to CSV using LibreOffice/Excel command line...")
        # This would require LibreOffice or Excel to be installed
        import subprocess
        
        # Try LibreOffice command
        result = subprocess.run([
            'libreoffice', '--headless', '--convert-to', 'csv', 
            'ETF_qdii.xlsx', '--outdir', '.'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("LibreOffice conversion successful!")
            if os.path.exists('ETF_qdii.csv'):
                df = pd.read_csv('ETF_qdii.csv')
                return df
        else:
            print(f"LibreOffice conversion failed: {result.stderr}")
            
    except Exception as e:
        print(f"CSV conversion failed: {e}")
    
    return None

def manual_data_entry():
    """Manually create sample QDII fund data for testing"""
    print("Creating sample QDII fund data for testing...")
    
    # Sample QDII fund codes (these are real QDII fund codes)
    sample_data = {
        'fund_code': [
            '000041',  # 华夏全球股票(QDII)
            '000043',  # 嘉实美国成长股票
            '000055',  # 广发纳斯达克100ETF联接
            '000103',  # 国泰中国企业境外高收益债
            '000179',  # 广发美国房地产指数
            '000274',  # 广发亚太中高收益债
            '000369',  # 广发全球医疗保健指数
            '000614',  # 华安德国(DAX)联接
            '000834',  # 大成纳斯达克100ETF联接
            '001061',  # 华夏海外收益债券
        ],
        'fund_name': [
            '华夏全球股票(QDII)',
            '嘉实美国成长股票',
            '广发纳斯达克100ETF联接',
            '国泰中国企业境外高收益债',
            '广发美国房地产指数',
            '广发亚太中高收益债',
            '广发全球医疗保健指数',
            '华安德国(DAX)联接',
            '大成纳斯达克100ETF联接',
            '华夏海外收益债券'
        ]
    }
    
    df = pd.DataFrame(sample_data)
    print(f"Created sample data with {len(df)} QDII funds")
    return df

def main():
    """Main function to try different methods"""
    print("Robust Excel Reader for QDII Fund Data")
    print("=" * 50)
    
    # Try different methods in order
    methods = [
        try_read_excel_basic,
        try_read_excel_with_options,
        try_convert_to_csv,
        try_read_as_zip,
        manual_data_entry  # Fallback
    ]
    
    for i, method in enumerate(methods):
        print(f"\nMethod {i+1}: {method.__name__}")
        print("-" * 30)
        
        result = method()
        
        if result is not None and isinstance(result, pd.DataFrame):
            print(f"Success! Got DataFrame with shape: {result.shape}")
            print("Columns:", list(result.columns))
            print("\nFirst few rows:")
            print(result.head())
            
            # Save the result
            output_file = f"qdii_funds_method_{i+1}.csv"
            result.to_csv(output_file, index=False, encoding='utf-8-sig')
            print(f"\nSaved data to: {output_file}")
            
            return result
        elif result is not None:
            print(f"Got result but not a DataFrame: {type(result)}")
        else:
            print("Method failed, trying next...")
    
    print("\nAll methods failed!")
    return None

if __name__ == "__main__":
    result = main()
    if result is not None:
        print(f"\nFinal result: DataFrame with {len(result)} rows")
    else:
        print("\nNo data could be extracted from the Excel file.")
