"""
Excel XML Parser for QDII Fund Data
This script manually parses the Excel XML to extract fund data.
"""

import pandas as pd
import zipfile
import xml.etree.ElementTree as ET
import re

class ExcelXMLParser:
    def __init__(self, excel_file):
        self.excel_file = excel_file
        self.shared_strings = []
        
    def parse_shared_strings(self, zip_file):
        """Parse shared strings from Excel file"""
        try:
            with zip_file.open('xl/sharedStrings.xml') as strings_file:
                content = strings_file.read().decode('utf-8')
                root = ET.fromstring(content)
                
                # Extract all text values
                for si in root.findall('.//{http://schemas.openxmlformats.org/spreadsheetml/2006/main}si'):
                    text_elem = si.find('.//{http://schemas.openxmlformats.org/spreadsheetml/2006/main}t')
                    if text_elem is not None:
                        self.shared_strings.append(text_elem.text)
                    else:
                        self.shared_strings.append('')
                
                print(f"Found {len(self.shared_strings)} shared strings")
                return True
        except Exception as e:
            print(f"Error parsing shared strings: {e}")
            return False
    
    def parse_worksheet(self, zip_file):
        """Parse worksheet data"""
        try:
            with zip_file.open('xl/worksheets/sheet1.xml') as sheet_file:
                content = sheet_file.read().decode('utf-8')
                root = ET.fromstring(content)
                
                # Find all cells
                cells = {}
                for row in root.findall('.//{http://schemas.openxmlformats.org/spreadsheetml/2006/main}row'):
                    for cell in row.findall('.//{http://schemas.openxmlformats.org/spreadsheetml/2006/main}c'):
                        cell_ref = cell.get('r')  # Cell reference like A1, B2, etc.
                        cell_type = cell.get('t')  # Cell type
                        
                        value_elem = cell.find('.//{http://schemas.openxmlformats.org/spreadsheetml/2006/main}v')
                        if value_elem is not None:
                            value = value_elem.text
                            
                            # If it's a shared string, look it up
                            if cell_type == 's' and value.isdigit():
                                string_index = int(value)
                                if string_index < len(self.shared_strings):
                                    value = self.shared_strings[string_index]
                            
                            cells[cell_ref] = value
                
                print(f"Found {len(cells)} cells with data")
                return cells
                
        except Exception as e:
            print(f"Error parsing worksheet: {e}")
            return None
    
    def cells_to_dataframe(self, cells):
        """Convert cell data to DataFrame"""
        if not cells:
            return None
        
        # Parse cell references to get row and column numbers
        cell_data = {}
        for cell_ref, value in cells.items():
            # Parse cell reference like A1, B2, etc.
            match = re.match(r'([A-Z]+)(\d+)', cell_ref)
            if match:
                col_letters = match.group(1)
                row_num = int(match.group(2))
                
                # Convert column letters to number
                col_num = 0
                for char in col_letters:
                    col_num = col_num * 26 + (ord(char) - ord('A') + 1)
                
                if row_num not in cell_data:
                    cell_data[row_num] = {}
                cell_data[row_num][col_num] = value
        
        if not cell_data:
            return None
        
        # Find the range of data
        max_row = max(cell_data.keys())
        max_col = max(max(row_data.keys()) for row_data in cell_data.values())
        
        print(f"Data range: {max_row} rows, {max_col} columns")
        
        # Create DataFrame
        data = []
        for row_num in range(1, max_row + 1):
            row_data = []
            for col_num in range(1, max_col + 1):
                if row_num in cell_data and col_num in cell_data[row_num]:
                    row_data.append(cell_data[row_num][col_num])
                else:
                    row_data.append('')
            data.append(row_data)
        
        # Create column names
        columns = [f'Column_{i}' for i in range(1, max_col + 1)]
        
        df = pd.DataFrame(data, columns=columns)
        
        # Use first row as headers if it looks like headers
        if len(df) > 0:
            first_row = df.iloc[0]
            if any(isinstance(val, str) and not val.isdigit() for val in first_row):
                df.columns = first_row
                df = df.iloc[1:].reset_index(drop=True)
        
        return df
    
    def parse_excel(self):
        """Main parsing function"""
        try:
            with zipfile.ZipFile(self.excel_file, 'r') as zip_file:
                print("Parsing shared strings...")
                self.parse_shared_strings(zip_file)
                
                print("Parsing worksheet...")
                cells = self.parse_worksheet(zip_file)
                
                if cells:
                    print("Converting to DataFrame...")
                    df = self.cells_to_dataframe(cells)
                    return df
                else:
                    return None
                    
        except Exception as e:
            print(f"Error parsing Excel file: {e}")
            return None

def main():
    """Main function"""
    print("Excel XML Parser for QDII Fund Data")
    print("=" * 50)
    
    parser = ExcelXMLParser('ETF_qdii.xlsx')
    df = parser.parse_excel()
    
    if df is not None:
        print(f"\nSuccessfully parsed Excel file!")
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print("\nFirst 10 rows:")
        print(df.head(10))
        
        # Save to CSV
        output_file = "qdii_funds_parsed.csv"
        df.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"\nSaved parsed data to: {output_file}")
        
        return df
    else:
        print("Failed to parse Excel file")
        return None

if __name__ == "__main__":
    result = main()
