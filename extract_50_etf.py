import pandas as pd
import csv

def extract_etf_components(file_path):
    # Read the file to find where the component data starts
    with open(file_path, 'r', encoding='utf-8-sig') as file:
        lines = file.readlines()
    
    # Find the header row for components
    start_row = 0
    for i, line in enumerate(lines):
        if "Stock Code" in line and "Stock Name" in line:
            start_row = i
            break
    
    # Read the components data
    df = pd.read_csv(file_path, skiprows=start_row, encoding='utf-8-sig')
    
    # Clean up column names
    df.columns = df.columns.str.strip()
    
    # Convert shares to integer if possible
    if 'Shares' in df.columns:
        df['Shares'] = pd.to_numeric(df['Shares'], errors='coerce')
    
    return df

# Load the ETF components
file_path = "0050.csv"
components_df = extract_etf_components(file_path)

# Display the components
print(f"Found {len(components_df)} components in ETF 0050")
print(components_df.head(10))

# Example: Get total shares
total_shares = components_df['Shares'].sum()
print(f"Total shares across all components: {total_shares}")

# Example: Calculate percentage weight by shares
#components_df['Weight (%)'] = components_df['Shares'] / total_shares * 100

# Export to Excel if needed
components_df.to_excel("0050_components.xlsx", index=False)
print("Exported components to Excel file")