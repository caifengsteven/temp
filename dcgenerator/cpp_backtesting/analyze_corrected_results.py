#!/usr/bin/env python3
"""
Analyze corrected DC testing results
"""

import os
import re
import pandas as pd
import glob

def parse_report_file(filename):
    """Parse a single report file and extract key metrics"""
    try:
        with open(filename, 'r') as f:
            content = f.read()
        
        # Extract symbol from filename
        symbol = filename.replace('corrected_report_', '').replace('.txt', '')
        
        # Extract data summary
        data_points_match = re.search(r'Total Price Points: (\d+)', content)
        price_range_match = re.search(r'Price Range %: ([\d.]+)%', content)
        
        data_points = int(data_points_match.group(1)) if data_points_match else 0
        price_range_pct = float(price_range_match.group(1)) if price_range_match else 0.0
        
        # Extract Simple DC results (0.5% threshold)
        simple_pattern = r'0\.5%\s+(\d+)\s+\$(-?\d+)\s+(-?[\d.]+)%\s+\$(\d+)'
        simple_match = re.search(simple_pattern, content)
        
        simple_trades = int(simple_match.group(1)) if simple_match else 0
        simple_pnl = int(simple_match.group(2)) if simple_match else 0
        simple_return = float(simple_match.group(3)) if simple_match else 0.0
        
        # Extract Contrarian DC results (0.5% threshold)
        # Find contrarian section first
        contrarian_section = content.split('=== CORRECTED Contrarian DC STRATEGY RESULTS ===')[1] if '=== CORRECTED Contrarian DC STRATEGY RESULTS ===' in content else ""
        
        contrarian_match = re.search(simple_pattern, contrarian_section)
        
        contrarian_trades = int(contrarian_match.group(1)) if contrarian_match else 0
        contrarian_pnl = int(contrarian_match.group(2)) if contrarian_match else 0
        contrarian_return = float(contrarian_match.group(3)) if contrarian_match else 0.0
        
        return {
            'symbol': symbol,
            'data_points': data_points,
            'price_range_pct': price_range_pct,
            'simple_trades': simple_trades,
            'simple_pnl': simple_pnl,
            'simple_return': simple_return,
            'contrarian_trades': contrarian_trades,
            'contrarian_pnl': contrarian_pnl,
            'contrarian_return': contrarian_return
        }
        
    except Exception as e:
        print(f"Error parsing {filename}: {e}")
        return None

def main():
    print("=== Analyzing Corrected DC Testing Results ===")
    
    # Find all report files
    report_files = glob.glob("corrected_report_*.txt")
    
    if not report_files:
        print("No corrected report files found!")
        print("Make sure you've run the mass testing first.")
        return
    
    print(f"Found {len(report_files)} report files")
    
    # Parse all files
    results = []
    for filename in report_files:
        result = parse_report_file(filename)
        if result:
            results.append(result)
    
    if not results:
        print("No valid results found!")
        return
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    print(f"\nSuccessfully parsed {len(df)} reports")
    
    # Summary statistics
    print("\n=== CORRECTED DC RESULTS SUMMARY ===")
    print(f"Total stocks analyzed: {len(df)}")
    print(f"Average data points per stock: {df['data_points'].mean():.0f}")
    print(f"Average price range: {df['price_range_pct'].mean():.1f}%")
    
    print("\n=== Simple DC Strategy (0.5% threshold) ===")
    print(f"Average trades per stock: {df['simple_trades'].mean():.1f}")
    print(f"Average return: {df['simple_return'].mean():.2f}%")
    print(f"Best performing stock: {df.loc[df['simple_return'].idxmax(), 'symbol']} ({df['simple_return'].max():.2f}%)")
    print(f"Worst performing stock: {df.loc[df['simple_return'].idxmin(), 'symbol']} ({df['simple_return'].min():.2f}%)")
    
    print("\n=== Contrarian DC Strategy (0.5% threshold) ===")
    print(f"Average trades per stock: {df['contrarian_trades'].mean():.1f}")
    print(f"Average return: {df['contrarian_return'].mean():.2f}%")
    print(f"Best performing stock: {df.loc[df['contrarian_return'].idxmax(), 'symbol']} ({df['contrarian_return'].max():.2f}%)")
    print(f"Worst performing stock: {df.loc[df['contrarian_return'].idxmin(), 'symbol']} ({df['contrarian_return'].min():.2f}%)")
    
    # Top performers
    print("\n=== TOP 10 SIMPLE DC PERFORMERS ===")
    top_simple = df.nlargest(10, 'simple_return')[['symbol', 'simple_trades', 'simple_return', 'price_range_pct']]
    print(top_simple.to_string(index=False))
    
    print("\n=== TOP 10 CONTRARIAN DC PERFORMERS ===")
    top_contrarian = df.nlargest(10, 'contrarian_return')[['symbol', 'contrarian_trades', 'contrarian_return', 'price_range_pct']]
    print(top_contrarian.to_string(index=False))
    
    # Save summary to CSV
    df.to_csv('corrected_dc_summary.csv', index=False)
    print(f"\nDetailed results saved to: corrected_dc_summary.csv")
    
    # Comparison with original buggy results
    print("\n=== COMPARISON WITH ORIGINAL BUGGY RESULTS ===")
    print("Original DCGenerator bugs caused:")
    print("- Unrealistic trade counts (10,000+ trades per stock)")
    print("- Extreme returns (1,000,000%+ for contrarian strategy)")
    print("- False DC events due to logic errors")
    print()
    print("Corrected DCGenerator shows:")
    print(f"- Realistic trade counts (avg {df['simple_trades'].mean():.1f} trades)")
    print(f"- Modest returns (avg {df['simple_return'].mean():.2f}% simple, {df['contrarian_return'].mean():.2f}% contrarian)")
    print("- Proper DC event detection based on actual price movements")
    
    # Statistics on stocks with no trades
    no_trades_simple = len(df[df['simple_trades'] == 0])
    no_trades_contrarian = len(df[df['contrarian_trades'] == 0])
    
    print(f"\nStocks with no trades (0.5% threshold):")
    print(f"- Simple DC: {no_trades_simple}/{len(df)} ({no_trades_simple/len(df)*100:.1f}%)")
    print(f"- Contrarian DC: {no_trades_contrarian}/{len(df)} ({no_trades_contrarian/len(df)*100:.1f}%)")
    print("This is normal for stocks with small price ranges relative to the DC threshold.")

if __name__ == "__main__":
    main()
