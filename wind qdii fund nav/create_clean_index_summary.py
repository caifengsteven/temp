"""
Create Clean Index Summary
This script creates a clean summary of QDII funds with their underlying indices.
"""

import pandas as pd

def create_clean_summary():
    """Create a clean summary of QDII funds and their underlying indices"""
    
    try:
        # Load the detailed index data
        print("Loading underlying index data...")
        df = pd.read_csv("qdii_underlying_index_info.csv", encoding='utf-8-sig')
        
        # Create clean summary with essential columns
        summary = df[[
            'fund_name',
            'index_FUND_TRACKINDEXCODE', 
            'index_FUND_TRACKINDEXNAME'
        ]].copy()
        
        # Rename columns for clarity
        summary.columns = ['Fund_Name', 'Index_Code', 'Index_Name']
        
        # Add the wind code from index
        summary['Wind_Code'] = df.index
        
        # Reorder columns
        summary = summary[['Wind_Code', 'Fund_Name', 'Index_Code', 'Index_Name']]
        
        # Remove rows with missing index information
        summary_clean = summary.dropna(subset=['Index_Code', 'Index_Name'])
        
        # Sort by fund name
        summary_clean = summary_clean.sort_values('Fund_Name')
        
        # Save clean summary
        output_file = "qdii_funds_underlying_indices.csv"
        summary_clean.to_csv(output_file, index=False, encoding='utf-8-sig')
        
        print(f"\n🎉 Clean summary saved to: {output_file}")
        print(f"📊 Summary:")
        print(f"   • Total funds with index data: {len(summary_clean)}")
        print(f"   • Funds without index data: {len(summary) - len(summary_clean)}")
        
        # Show unique indices
        unique_indices = summary_clean['Index_Name'].value_counts()
        print(f"   • Unique underlying indices: {len(unique_indices)}")
        
        print(f"\n📋 Top 10 most common underlying indices:")
        print(unique_indices.head(10).to_string())
        
        print(f"\n📋 Sample data:")
        print(summary_clean.head(10).to_string(index=False))
        
        # Create index grouping
        print(f"\n📈 Creating index grouping...")
        index_groups = summary_clean.groupby(['Index_Code', 'Index_Name']).agg({
            'Fund_Name': list,
            'Wind_Code': 'count'
        }).rename(columns={'Wind_Code': 'Fund_Count'})
        
        index_groups['Fund_Names'] = index_groups['Fund_Name'].apply(lambda x: '; '.join(x))
        index_groups = index_groups[['Fund_Count', 'Fund_Names']].sort_values('Fund_Count', ascending=False)
        
        # Save index grouping
        group_file = "qdii_indices_grouping.csv"
        index_groups.to_csv(group_file, encoding='utf-8-sig')
        print(f"📊 Index grouping saved to: {group_file}")
        
        print(f"\n🔝 Top indices by number of tracking funds:")
        print(index_groups.head(10)[['Fund_Count']].to_string())
        
        return summary_clean, index_groups
        
    except Exception as e:
        print(f"Error creating clean summary: {e}")
        return None, None

def analyze_index_types(summary_clean):
    """Analyze the types of underlying indices"""
    
    if summary_clean is None:
        return
    
    print(f"\n" + "="*60)
    print("UNDERLYING INDEX ANALYSIS")
    print("="*60)
    
    # Categorize indices by type
    index_categories = {
        'Hong Kong Indices': ['恒生', 'HSTECH', 'HSHCI', 'HSIII', 'HI'],
        'US Indices': ['纳斯达克', '纳指', 'NASDAQ', 'NDXTMC'],
        'China/A-Share Indices': ['中证', 'CSI', '沪深'],
        'Global/Regional': ['MSCI', '标普', 'S&P'],
        'Sector Specific': ['科技', '医疗', '创新药', '互联网', '生物'],
    }
    
    categorized = {}
    for category, keywords in index_categories.items():
        categorized[category] = []
        for _, row in summary_clean.iterrows():
            index_name = str(row['Index_Name'])
            index_code = str(row['Index_Code'])
            if any(keyword in index_name or keyword in index_code for keyword in keywords):
                categorized[category].append(row['Fund_Name'])
    
    print(f"\n📊 INDEX CATEGORIES:")
    for category, funds in categorized.items():
        print(f"\n🔹 {category} ({len(funds)} funds):")
        for fund in funds[:5]:  # Show first 5
            print(f"   • {fund}")
        if len(funds) > 5:
            print(f"   ... and {len(funds) - 5} more")

def main():
    """Main function"""
    print("QDII Fund Underlying Index Summary Creator")
    print("=" * 60)
    
    summary_clean, index_groups = create_clean_summary()
    
    if summary_clean is not None:
        analyze_index_types(summary_clean)
        
        print(f"\n🎉 SUMMARY COMPLETED!")
        print(f"\n📁 Generated files:")
        print(f"   • qdii_funds_underlying_indices.csv - Clean fund-index mapping")
        print(f"   • qdii_indices_grouping.csv - Indices grouped by tracking funds")
        print(f"   • qdii_underlying_index_info.csv - Detailed information")
        
        print(f"\n💡 KEY INSIGHTS:")
        if len(summary_clean) > 0:
            total_funds = len(summary_clean)
            unique_indices = summary_clean['Index_Name'].nunique()
            print(f"   • {total_funds} QDII funds track {unique_indices} different indices")
            print(f"   • Average funds per index: {total_funds/unique_indices:.1f}")
            
            # Most popular index
            most_popular = summary_clean['Index_Name'].value_counts().iloc[0]
            most_popular_name = summary_clean['Index_Name'].value_counts().index[0]
            print(f"   • Most tracked index: {most_popular_name} ({most_popular} funds)")

if __name__ == "__main__":
    main()
