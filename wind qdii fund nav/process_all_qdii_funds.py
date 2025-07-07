"""
Process All QDII Funds - Complete NAV Data Retrieval
This script processes all QDII funds from the Excel file and gets their complete NAV/price data.
"""

import pandas as pd
from wind_nav_final import FinalWindNavRetriever

def main():
    """Main function to process all QDII funds"""
    print("🚀 Processing ALL QDII Funds from Excel File")
    print("=" * 60)
    
    retriever = FinalWindNavRetriever()
    
    try:
        # Process ALL funds (no limit)
        print("⚠️  This will process all 154 QDII funds. This may take 10-15 minutes.")
        user_input = input("Do you want to continue? (y/n): ")
        
        if user_input.lower() != 'y':
            print("Operation cancelled.")
            return
        
        success = retriever.process_all_funds(max_funds=None)  # No limit - process all funds
        
        if success:
            print("\n🎉 Complete QDII fund data retrieval finished!")
            
            # Load and analyze the complete data
            try:
                df = pd.read_csv("qdii_nav_price_data_wind.csv")
                
                print(f"\n📊 Final Summary:")
                print(f"   • Total records: {len(df):,}")
                print(f"   • Unique funds: {df['wind_code'].nunique()}")
                print(f"   • Date range: {df['date'].min()} to {df['date'].max()}")
                print(f"   • Data columns: {list(df.columns)}")
                
                # Fund-wise summary
                fund_summary = df.groupby(['wind_code', 'fund_name']).agg({
                    'date': ['min', 'max', 'count'],
                    'CLOSE': ['min', 'max', 'mean']
                }).round(4)
                
                fund_summary.columns = ['Start_Date', 'End_Date', 'Days', 'Min_Price', 'Max_Price', 'Avg_Price']
                fund_summary = fund_summary.reset_index()
                
                # Save fund summary
                fund_summary.to_csv("qdii_fund_summary.csv", index=False, encoding='utf-8-sig')
                print(f"\n📋 Fund summary saved to: qdii_fund_summary.csv")
                
                # Show top 10 funds by average price
                print(f"\n🔝 Top 10 funds by average price:")
                top_funds = fund_summary.nlargest(10, 'Avg_Price')[['wind_code', 'fund_name', 'Avg_Price', 'Days']]
                print(top_funds.to_string(index=False))
                
                # Show funds with most data points
                print(f"\n📈 Funds with most data points:")
                most_data = fund_summary.nlargest(10, 'Days')[['wind_code', 'fund_name', 'Days', 'Start_Date', 'End_Date']]
                print(most_data.to_string(index=False))
                
            except Exception as e:
                print(f"Error analyzing final data: {e}")
        else:
            print("\n❌ Complete QDII fund data retrieval failed.")
    
    finally:
        # Always disconnect from Wind
        retriever.disconnect_wind()

if __name__ == "__main__":
    main()
