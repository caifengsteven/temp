"""
Compare Closing Prices vs NAV Values
This script compares the closing prices from the historical data with the latest NAV values.
"""

import pandas as pd
import numpy as np

def load_and_compare_data():
    """Load both datasets and create comparison"""
    
    try:
        # Load historical closing price data
        print("Loading historical closing price data...")
        close_data = pd.read_csv("qdii_nav_price_data_wind.csv")
        print(f"Loaded {len(close_data)} historical records")
        
        # Get latest closing prices for each fund
        latest_close = close_data.groupby('wind_code').agg({
            'CLOSE': 'last',
            'date': 'last',
            'fund_name': 'first'
        }).reset_index()
        latest_close.columns = ['wind_code', 'latest_close_price', 'close_date', 'fund_name_close']
        
        print(f"Got latest closing prices for {len(latest_close)} funds")
        
    except Exception as e:
        print(f"Error loading closing price data: {e}")
        return None
    
    try:
        # Load latest NAV data
        print("Loading latest NAV data...")
        nav_data = pd.read_csv("qdii_latest_nav.csv")

        # Fix the wind_code column - it's in the first unnamed column
        if nav_data.columns[0] == 'Unnamed: 0' or nav_data.columns[0] == '':
            nav_data = nav_data.rename(columns={nav_data.columns[0]: 'wind_code'})
        elif 'wind_code' not in nav_data.columns:
            nav_data['wind_code'] = nav_data.index

        # Ensure wind_code is string type for merging
        nav_data['wind_code'] = nav_data['wind_code'].astype(str)
        latest_close['wind_code'] = latest_close['wind_code'].astype(str)

        print(f"Loaded NAV data for {len(nav_data)} funds")
        
    except Exception as e:
        print(f"Error loading NAV data: {e}")
        return None
    
    # Merge the datasets
    print("Merging datasets...")
    comparison = pd.merge(
        latest_close,
        nav_data[['wind_code', 'NAV', 'nav_date', 'fund_name']],
        on='wind_code',
        how='outer',
        suffixes=('_close', '_nav')
    )

    # Calculate differences
    comparison['price_nav_diff'] = comparison['latest_close_price'] - comparison['NAV']
    comparison['price_nav_diff_pct'] = (comparison['price_nav_diff'] / comparison['NAV'] * 100).round(4)

    # Use NAV fund name if close fund name is missing, or vice versa
    comparison['fund_name_final'] = comparison['fund_name'].fillna(comparison['fund_name_close'])

    # Clean up columns
    comparison = comparison[[
        'wind_code', 'fund_name_final',
        'latest_close_price', 'close_date',
        'NAV', 'nav_date',
        'price_nav_diff', 'price_nav_diff_pct'
    ]].rename(columns={'fund_name_final': 'fund_name'})
    
    return comparison

def analyze_comparison(comparison):
    """Analyze the comparison data"""
    
    print("\n" + "="*80)
    print("CLOSING PRICE vs NAV ANALYSIS")
    print("="*80)
    
    # Basic statistics
    total_funds = len(comparison)
    funds_with_both = len(comparison.dropna(subset=['latest_close_price', 'NAV']))
    funds_only_close = len(comparison[comparison['NAV'].isna() & comparison['latest_close_price'].notna()])
    funds_only_nav = len(comparison[comparison['latest_close_price'].isna() & comparison['NAV'].notna()])
    
    print(f"\n📊 SUMMARY STATISTICS:")
    print(f"   • Total funds: {total_funds}")
    print(f"   • Funds with both Close & NAV: {funds_with_both}")
    print(f"   • Funds with only Close price: {funds_only_close}")
    print(f"   • Funds with only NAV: {funds_only_nav}")
    
    # Analyze differences for funds with both values
    both_data = comparison.dropna(subset=['latest_close_price', 'NAV'])
    
    if len(both_data) > 0:
        print(f"\n📈 PRICE vs NAV DIFFERENCES (for {len(both_data)} funds with both values):")
        print(f"   • Average difference: {both_data['price_nav_diff'].mean():.4f}")
        print(f"   • Average difference %: {both_data['price_nav_diff_pct'].mean():.4f}%")
        print(f"   • Max positive difference: {both_data['price_nav_diff'].max():.4f}")
        print(f"   • Max negative difference: {both_data['price_nav_diff'].min():.4f}")
        print(f"   • Standard deviation: {both_data['price_nav_diff'].std():.4f}")
        
        # Show funds with largest differences
        print(f"\n🔝 TOP 10 LARGEST POSITIVE DIFFERENCES (Close > NAV):")
        top_positive = both_data.nlargest(10, 'price_nav_diff')[['wind_code', 'fund_name', 'latest_close_price', 'NAV', 'price_nav_diff', 'price_nav_diff_pct']]
        print(top_positive.to_string(index=False))
        
        print(f"\n🔻 TOP 10 LARGEST NEGATIVE DIFFERENCES (Close < NAV):")
        top_negative = both_data.nsmallest(10, 'price_nav_diff')[['wind_code', 'fund_name', 'latest_close_price', 'NAV', 'price_nav_diff', 'price_nav_diff_pct']]
        print(top_negative.to_string(index=False))
        
        # Show funds with smallest differences (most aligned)
        print(f"\n🎯 TOP 10 MOST ALIGNED (Smallest Absolute Differences):")
        both_data['abs_diff'] = abs(both_data['price_nav_diff'])
        most_aligned = both_data.nsmallest(10, 'abs_diff')[['wind_code', 'fund_name', 'latest_close_price', 'NAV', 'price_nav_diff', 'price_nav_diff_pct']]
        print(most_aligned.to_string(index=False))
    
    # Show date analysis
    print(f"\n📅 DATE ANALYSIS:")
    if 'close_date' in comparison.columns:
        close_dates = comparison['close_date'].dropna().unique()
        print(f"   • Close price dates: {sorted(close_dates)}")
    
    if 'nav_date' in comparison.columns:
        nav_dates = comparison['nav_date'].dropna().unique()
        print(f"   • NAV dates: {sorted(nav_dates)}")
    
    return both_data

def save_comparison_results(comparison):
    """Save comparison results to files"""
    
    # Save complete comparison
    output_file = "close_vs_nav_comparison.csv"
    comparison.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n💾 Complete comparison saved to: {output_file}")
    
    # Save summary for funds with both values
    both_data = comparison.dropna(subset=['latest_close_price', 'NAV'])
    if len(both_data) > 0:
        summary_file = "close_vs_nav_summary.csv"
        summary = both_data[['wind_code', 'fund_name', 'latest_close_price', 'NAV', 'price_nav_diff', 'price_nav_diff_pct']].copy()
        summary = summary.sort_values('price_nav_diff_pct', ascending=False)
        summary.to_csv(summary_file, index=False, encoding='utf-8-sig')
        print(f"💾 Summary comparison saved to: {summary_file}")
    
    # Create a simple latest NAV only file
    nav_only = comparison[['wind_code', 'fund_name', 'NAV', 'nav_date']].dropna(subset=['NAV'])
    nav_only_file = "qdii_latest_nav_simple.csv"
    nav_only.to_csv(nav_only_file, index=False, encoding='utf-8-sig')
    print(f"💾 Simple NAV file saved to: {nav_only_file}")
    
    return [output_file, summary_file, nav_only_file]

def main():
    """Main function"""
    print("QDII Fund: Closing Price vs NAV Comparison")
    print("=" * 60)
    
    # Load and compare data
    comparison = load_and_compare_data()
    
    if comparison is None:
        print("❌ Failed to load comparison data")
        return
    
    # Analyze the comparison
    both_data = analyze_comparison(comparison)
    
    # Save results
    saved_files = save_comparison_results(comparison)
    
    print(f"\n🎉 COMPARISON COMPLETED!")
    print(f"\n📁 Generated files:")
    for file in saved_files:
        print(f"   • {file}")
    
    print(f"\n📋 KEY INSIGHTS:")
    if len(both_data) > 0:
        avg_diff_pct = both_data['price_nav_diff_pct'].mean()
        if abs(avg_diff_pct) < 1:
            print(f"   • Close prices and NAV values are very well aligned (avg diff: {avg_diff_pct:.4f}%)")
        elif abs(avg_diff_pct) < 5:
            print(f"   • Close prices and NAV values are reasonably aligned (avg diff: {avg_diff_pct:.4f}%)")
        else:
            print(f"   • Significant differences between close prices and NAV (avg diff: {avg_diff_pct:.4f}%)")
    
    print(f"   • Use 'qdii_latest_nav_simple.csv' for clean NAV data with dates")
    print(f"   • Use 'close_vs_nav_comparison.csv' for detailed analysis")

if __name__ == "__main__":
    main()
