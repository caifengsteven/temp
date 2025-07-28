#!/usr/bin/env python3
"""
Analyze what factors determine when Simple DC vs Contrarian DC strategies work better
"""

import re
import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def parse_report_file(filename):
    """Parse a single report file and extract comprehensive metrics"""
    try:
        with open(filename, 'r') as f:
            content = f.read()
        
        # Extract symbol from filename
        if 'us_corrected_report_' in filename:
            symbol = filename.replace('us_corrected_report_', '').replace('.txt', '')
            market = 'US'
        else:
            symbol = filename.replace('corrected_report_', '').replace('.txt', '')
            market = 'China'
        
        # Extract data summary
        data_points_match = re.search(r'Total Price Points: (\d+)', content)
        price_range_match = re.search(r'Price Range: \$([0-9.]+) to \$([0-9.]+)', content)
        price_range_pct_match = re.search(r'Price Range %: ([0-9.]+)%', content)
        
        data_points = int(data_points_match.group(1)) if data_points_match else 0
        
        if price_range_match:
            min_price = float(price_range_match.group(1))
            max_price = float(price_range_match.group(2))
            price_volatility = (max_price - min_price) / min_price
        else:
            min_price = max_price = price_volatility = 0
            
        price_range_pct = float(price_range_pct_match.group(1)) if price_range_pct_match else 0.0
        
        # Extract all threshold results for Simple DC
        simple_section = content.split('=== CORRECTED Simple DC STRATEGY RESULTS ===')[1].split('===')[0]
        simple_results = {}
        
        # Look for threshold patterns (0.5%, 1.0%, etc.)
        threshold_pattern = r'([0-9.]+)%\s+(\d+)\s+\$(-?\d+)\s+(-?[0-9.]+)%\s+\$(\d+)'
        for match in re.finditer(threshold_pattern, simple_section):
            threshold = float(match.group(1))
            trades = int(match.group(2))
            pnl = int(match.group(3))
            return_pct = float(match.group(4))
            simple_results[threshold] = {
                'trades': trades,
                'pnl': pnl,
                'return': return_pct
            }
        
        # Extract all threshold results for Contrarian DC
        contrarian_section = content.split('=== CORRECTED Contrarian DC STRATEGY RESULTS ===')[1].split('===')[0]
        contrarian_results = {}
        
        for match in re.finditer(threshold_pattern, contrarian_section):
            threshold = float(match.group(1))
            trades = int(match.group(2))
            pnl = int(match.group(3))
            return_pct = float(match.group(4))
            contrarian_results[threshold] = {
                'trades': trades,
                'pnl': pnl,
                'return': return_pct
            }
        
        # Calculate additional metrics
        avg_price = (min_price + max_price) / 2 if max_price > 0 else 0
        
        # Determine which strategy works better (using 0.5% threshold)
        simple_return_05 = simple_results.get(0.5, {}).get('return', 0)
        contrarian_return_05 = contrarian_results.get(0.5, {}).get('return', 0)
        
        better_strategy = 'Simple' if simple_return_05 > contrarian_return_05 else 'Contrarian'
        performance_difference = abs(simple_return_05 - contrarian_return_05)
        
        return {
            'symbol': symbol,
            'market': market,
            'data_points': data_points,
            'min_price': min_price,
            'max_price': max_price,
            'avg_price': avg_price,
            'price_range_pct': price_range_pct,
            'price_volatility': price_volatility,
            'simple_return_05': simple_return_05,
            'contrarian_return_05': contrarian_return_05,
            'simple_trades_05': simple_results.get(0.5, {}).get('trades', 0),
            'contrarian_trades_05': contrarian_results.get(0.5, {}).get('trades', 0),
            'better_strategy': better_strategy,
            'performance_difference': performance_difference,
            'simple_results': simple_results,
            'contrarian_results': contrarian_results
        }
        
    except Exception as e:
        print(f"Error parsing {filename}: {e}")
        return None

def analyze_strategy_factors(df):
    """Analyze what factors determine strategy performance"""
    
    print("=== STRATEGY PERFORMANCE FACTOR ANALYSIS ===")
    print(f"Total symbols analyzed: {len(df)}")
    
    # Split by better performing strategy
    simple_better = df[df['better_strategy'] == 'Simple']
    contrarian_better = df[df['better_strategy'] == 'Contrarian']
    
    print(f"Simple DC works better: {len(simple_better)} symbols ({len(simple_better)/len(df)*100:.1f}%)")
    print(f"Contrarian DC works better: {len(contrarian_better)} symbols ({len(contrarian_better)/len(df)*100:.1f}%)")
    
    # Analyze key factors
    factors = ['price_range_pct', 'avg_price', 'price_volatility', 'data_points']
    
    print("\n=== FACTOR ANALYSIS ===")
    
    for factor in factors:
        print(f"\n--- {factor.upper().replace('_', ' ')} ---")
        
        simple_values = simple_better[factor]
        contrarian_values = contrarian_better[factor]
        
        print(f"When Simple DC works better:")
        print(f"  Mean: {simple_values.mean():.2f}")
        print(f"  Median: {simple_values.median():.2f}")
        print(f"  Std: {simple_values.std():.2f}")
        
        print(f"When Contrarian DC works better:")
        print(f"  Mean: {contrarian_values.mean():.2f}")
        print(f"  Median: {contrarian_values.median():.2f}")
        print(f"  Std: {contrarian_values.std():.2f}")
        
        # Statistical test
        if len(simple_values) > 0 and len(contrarian_values) > 0:
            statistic, p_value = stats.mannwhitneyu(simple_values, contrarian_values, alternative='two-sided')
            print(f"  Statistical significance (p-value): {p_value:.4f}")
            if p_value < 0.05:
                print(f"  *** SIGNIFICANT DIFFERENCE ***")
            else:
                print(f"  No significant difference")
    
    return simple_better, contrarian_better

def create_factor_analysis_plots(df, simple_better, contrarian_better):
    """Create visualization plots for factor analysis"""
    
    # Set up the plotting style
    plt.style.use('default')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('DC Strategy Performance Factor Analysis', fontsize=16)
    
    factors = [
        ('price_range_pct', 'Price Range %'),
        ('avg_price', 'Average Price ($)'),
        ('price_volatility', 'Price Volatility'),
        ('data_points', 'Data Points')
    ]
    
    for i, (factor, title) in enumerate(factors):
        ax = axes[i//2, i%2]
        
        # Create box plots
        data_to_plot = [simple_better[factor], contrarian_better[factor]]
        labels = ['Simple Better', 'Contrarian Better']
        
        box_plot = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)
        box_plot['boxes'][0].set_facecolor('lightblue')
        box_plot['boxes'][1].set_facecolor('lightcoral')
        
        ax.set_title(title)
        ax.set_ylabel(title)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('strategy_factor_analysis.png', dpi=300, bbox_inches='tight')
    print("\nFactor analysis plot saved as 'strategy_factor_analysis.png'")

def find_decision_rules(df):
    """Find decision rules for when each strategy works better"""
    
    print("\n=== DECISION RULES ANALYSIS ===")
    
    # Analyze price range thresholds
    price_ranges = [0, 5, 10, 20, 50, 100, 200]
    
    print("\nPrice Range % Analysis:")
    print("Range\t\tSimple Better\tContrarian Better\tTotal")
    print("-" * 60)
    
    for i in range(len(price_ranges)-1):
        low, high = price_ranges[i], price_ranges[i+1]
        subset = df[(df['price_range_pct'] >= low) & (df['price_range_pct'] < high)]
        
        if len(subset) > 0:
            simple_count = len(subset[subset['better_strategy'] == 'Simple'])
            contrarian_count = len(subset[subset['better_strategy'] == 'Contrarian'])
            total = len(subset)
            
            print(f"{low:3.0f}-{high:3.0f}%\t\t{simple_count:3d} ({simple_count/total*100:4.1f}%)\t{contrarian_count:3d} ({contrarian_count/total*100:4.1f}%)\t\t{total:3d}")
    
    # Analyze average price thresholds
    price_thresholds = [0, 1, 5, 10, 20, 50, 100, 1000]
    
    print("\nAverage Price Analysis:")
    print("Price Range\t\tSimple Better\tContrarian Better\tTotal")
    print("-" * 60)
    
    for i in range(len(price_thresholds)-1):
        low, high = price_thresholds[i], price_thresholds[i+1]
        subset = df[(df['avg_price'] >= low) & (df['avg_price'] < high)]
        
        if len(subset) > 0:
            simple_count = len(subset[subset['better_strategy'] == 'Simple'])
            contrarian_count = len(subset[subset['better_strategy'] == 'Contrarian'])
            total = len(subset)
            
            print(f"${low:3.0f}-${high:3.0f}\t\t{simple_count:3d} ({simple_count/total*100:4.1f}%)\t{contrarian_count:3d} ({contrarian_count/total*100:4.1f}%)\t\t{total:3d}")

def main():
    print("=== DC STRATEGY PERFORMANCE FACTOR ANALYSIS ===")
    
    # Find all report files
    china_files = glob.glob("corrected_report_*.txt")
    us_files = glob.glob("us_corrected_report_*.txt")
    
    print(f"Found {len(china_files)} China reports and {len(us_files)} US reports")
    
    if len(china_files) == 0 and len(us_files) == 0:
        print("No report files found!")
        return
    
    # Parse all files
    all_results = []
    
    for filename in china_files + us_files:
        result = parse_report_file(filename)
        if result:
            all_results.append(result)
    
    if not all_results:
        print("No valid results found!")
        return
    
    # Create DataFrame
    df = pd.DataFrame(all_results)
    
    # Remove outliers (extreme returns that might be errors)
    df = df[(df['simple_return_05'] > -100) & (df['simple_return_05'] < 1000)]
    df = df[(df['contrarian_return_05'] > -100) & (df['contrarian_return_05'] < 1000)]
    
    print(f"\nAnalyzing {len(df)} symbols after removing outliers")
    
    # Perform factor analysis
    simple_better, contrarian_better = analyze_strategy_factors(df)
    
    # Create plots
    try:
        create_factor_analysis_plots(df, simple_better, contrarian_better)
    except Exception as e:
        print(f"Could not create plots: {e}")
    
    # Find decision rules
    find_decision_rules(df)
    
    # Save detailed results
    df.to_csv('strategy_performance_analysis.csv', index=False)
    print(f"\nDetailed results saved to 'strategy_performance_analysis.csv'")
    
    # Summary insights
    print("\n=== KEY INSIGHTS ===")
    
    # Calculate correlations
    correlation_simple = df['price_range_pct'].corr(df['simple_return_05'])
    correlation_contrarian = df['price_range_pct'].corr(df['contrarian_return_05'])
    
    print(f"Price range correlation with Simple DC return: {correlation_simple:.3f}")
    print(f"Price range correlation with Contrarian DC return: {correlation_contrarian:.3f}")
    
    # Find the most predictive factor
    factors = ['price_range_pct', 'avg_price', 'price_volatility']
    best_factor = None
    best_separation = 0
    
    for factor in factors:
        simple_mean = simple_better[factor].mean()
        contrarian_mean = contrarian_better[factor].mean()
        separation = abs(simple_mean - contrarian_mean) / (simple_better[factor].std() + contrarian_better[factor].std())
        
        if separation > best_separation:
            best_separation = separation
            best_factor = factor
    
    print(f"\nMost predictive factor: {best_factor} (separation score: {best_separation:.3f})")

if __name__ == "__main__":
    main()
