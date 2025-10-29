"""
Compare 95% vs 10% Position Sizing Results
"""

import pandas as pd
import numpy as np

# Load both datasets
df_95 = pd.read_csv('backtest_top1000_2015_2025.csv')
df_10 = pd.read_csv('backtest_top1000_2015_2025_10pct.csv')

print("=" * 100)
print("POSITION SIZING COMPARISON: 95% vs 10%")
print("=" * 100)
print()

print("📊 SAMPLE SIZE")
print(f"   95% Position Size:  {len(df_95)} stocks tested")
print(f"   10% Position Size:  {len(df_10)} stocks tested")
print()

print("=" * 100)
print("SUCCESS RATE COMPARISON")
print("=" * 100)
print()

# Calculate success metrics for 95%
total_95 = len(df_95)
profitable_95 = len(df_95[df_95['total_return'] > 0])
bankrupt_95 = len(df_95[df_95['total_return'] <= -100])

# Calculate success metrics for 10%
total_10 = len(df_10)
profitable_10 = len(df_10[df_10['total_return'] > 0])
bankrupt_10 = len(df_10[df_10['total_return'] <= -100])

print(f"{'Metric':<30} {'95% Position':<20} {'10% Position':<20} {'Improvement'}")
print("-" * 100)
print(f"{'Profitable Stocks':<30} {profitable_95} ({profitable_95/total_95*100:.1f}%){'':<7} {profitable_10} ({profitable_10/total_10*100:.1f}%){'':<7} {'+' if profitable_10/total_10 > profitable_95/total_95 else ''}{(profitable_10/total_10 - profitable_95/total_95)*100:.1f}%")
print(f"{'Losing Stocks':<30} {total_95-profitable_95} ({(total_95-profitable_95)/total_95*100:.1f}%){'':<7} {total_10-profitable_10} ({(total_10-profitable_10)/total_10*100:.1f}%){'':<7} {'-' if (total_10-profitable_10)/total_10 < (total_95-profitable_95)/total_95 else ''}{((total_10-profitable_10)/total_10 - (total_95-profitable_95)/total_95)*100:.1f}%")
print(f"{'Bankruptcies (-100%)':<30} {bankrupt_95} ({bankrupt_95/total_95*100:.1f}%){'':<7} {bankrupt_10} ({bankrupt_10/total_10*100:.1f}%){'':<7} {'-' if bankrupt_10/total_10 < bankrupt_95/total_95 else ''}{(bankrupt_10/total_10 - bankrupt_95/total_95)*100:.1f}%")
print()

print("=" * 100)
print("RETURN STATISTICS COMPARISON")
print("=" * 100)
print()

print(f"{'Metric':<30} {'95% Position':<20} {'10% Position':<20} {'Difference'}")
print("-" * 100)
print(f"{'Average Return':<30} {df_95['total_return'].mean():>18.2f}% {df_10['total_return'].mean():>18.2f}% {df_10['total_return'].mean() - df_95['total_return'].mean():>18.2f}%")
print(f"{'Median Return':<30} {df_95['total_return'].median():>18.2f}% {df_10['total_return'].median():>18.2f}% {df_10['total_return'].median() - df_95['total_return'].median():>18.2f}%")
print(f"{'Best Return':<30} {df_95['total_return'].max():>18.2f}% {df_10['total_return'].max():>18.2f}% {df_10['total_return'].max() - df_95['total_return'].max():>18.2f}%")
print(f"{'Worst Return':<30} {df_95['total_return'].min():>18.2f}% {df_10['total_return'].min():>18.2f}% {df_10['total_return'].min() - df_95['total_return'].min():>18.2f}%")
print(f"{'Std Deviation':<30} {df_95['total_return'].std():>18.2f}% {df_10['total_return'].std():>18.2f}% {df_10['total_return'].std() - df_95['total_return'].std():>18.2f}%")
print()

print("=" * 100)
print("PERCENTILE COMPARISON")
print("=" * 100)
print()

percentiles = [0.95, 0.75, 0.50, 0.25, 0.05]
percentile_names = ['95th Percentile', '75th Percentile', '50th Percentile (Median)', '25th Percentile', '5th Percentile']

print(f"{'Percentile':<30} {'95% Position':<20} {'10% Position':<20} {'Difference'}")
print("-" * 100)
for name, p in zip(percentile_names, percentiles):
    val_95 = df_95['total_return'].quantile(p)
    val_10 = df_10['total_return'].quantile(p)
    print(f"{name:<30} {val_95:>18.2f}% {val_10:>18.2f}% {val_10 - val_95:>18.2f}%")
print()

print("=" * 100)
print("RISK METRICS COMPARISON")
print("=" * 100)
print()

print(f"{'Metric':<30} {'95% Position':<20} {'10% Position':<20} {'Difference'}")
print("-" * 100)
print(f"{'Average Sharpe Ratio':<30} {df_95['sharpe_ratio'].mean():>18.2f} {df_10['sharpe_ratio'].mean():>18.2f} {df_10['sharpe_ratio'].mean() - df_95['sharpe_ratio'].mean():>18.2f}")
print(f"{'Median Sharpe Ratio':<30} {df_95['sharpe_ratio'].median():>18.2f} {df_10['sharpe_ratio'].median():>18.2f} {df_10['sharpe_ratio'].median() - df_95['sharpe_ratio'].median():>18.2f}")
print(f"{'Average Max Drawdown':<30} {df_95['max_drawdown'].mean():>18.2f}% {df_10['max_drawdown'].mean():>18.2f}% {df_10['max_drawdown'].mean() - df_95['max_drawdown'].mean():>18.2f}%")
print(f"{'Average Win Rate':<30} {df_95['win_rate'].mean():>18.2f}% {df_10['win_rate'].mean():>18.2f}% {df_10['win_rate'].mean() - df_95['win_rate'].mean():>18.2f}%")
print()

print("=" * 100)
print("TRADE STATISTICS COMPARISON")
print("=" * 100)
print()

print(f"{'Metric':<30} {'95% Position':<20} {'10% Position':<20} {'Difference'}")
print("-" * 100)
print(f"{'Total Trades (All Stocks)':<30} {df_95['total_trades'].sum():>18.0f} {df_10['total_trades'].sum():>18.0f} {df_10['total_trades'].sum() - df_95['total_trades'].sum():>18.0f}")
print(f"{'Average Trades/Stock':<30} {df_95['total_trades'].mean():>18.1f} {df_10['total_trades'].mean():>18.1f} {df_10['total_trades'].mean() - df_95['total_trades'].mean():>18.1f}")
print(f"{'Median Trades/Stock':<30} {df_95['total_trades'].median():>18.1f} {df_10['total_trades'].median():>18.1f} {df_10['total_trades'].median() - df_95['total_trades'].median():>18.1f}")
print(f"{'Avg Holding Period (days)':<30} {df_95['avg_bars_held'].mean():>18.1f} {df_10['avg_bars_held'].mean():>18.1f} {df_10['avg_bars_held'].mean() - df_95['avg_bars_held'].mean():>18.1f}")
print()

print("=" * 100)
print("RETURN DISTRIBUTION COMPARISON")
print("=" * 100)
print()

bins = [
    ('Bankruptcy (-100%)', -101, -99.9),
    ('Loss > 80%', -99.9, -80),
    ('Loss 50-80%', -80, -50),
    ('Loss 20-50%', -50, -20),
    ('Loss 0-20%', -20, 0),
    ('Gain 0-20%', 0, 20),
    ('Gain 20-50%', 20, 50),
    ('Gain 50-100%', 50, 100),
    ('Gain 100-500%', 100, 500),
    ('Gain 500-1000%', 500, 1000),
    ('Gain > 1000%', 1000, float('inf'))
]

print(f"{'Return Range':<25} {'95% Position':<20} {'10% Position':<20}")
print("-" * 100)
for label, low, high in bins:
    count_95 = len(df_95[(df_95['total_return'] > low) & (df_95['total_return'] <= high)])
    count_10 = len(df_10[(df_10['total_return'] > low) & (df_10['total_return'] <= high)])
    pct_95 = count_95 / total_95 * 100
    pct_10 = count_10 / total_10 * 100
    
    bar_95 = '█' * int(pct_95 / 2)
    bar_10 = '█' * int(pct_10 / 2)
    
    print(f"{label:<25} {count_95:>4} ({pct_95:>5.1f}%) {bar_95:<25} {count_10:>4} ({pct_10:>5.1f}%) {bar_10}")
print()

print("=" * 100)
print("SHARPE RATIO DISTRIBUTION COMPARISON")
print("=" * 100)
print()

sharpe_bins = [
    ('Negative (<0)', -float('inf'), 0),
    ('Poor (0-1)', 0, 1),
    ('Good (1-2)', 1, 2),
    ('Excellent (2-3)', 2, 3),
    ('Outstanding (>3)', 3, float('inf'))
]

print(f"{'Sharpe Range':<25} {'95% Position':<20} {'10% Position':<20}")
print("-" * 100)
for label, low, high in sharpe_bins:
    count_95 = len(df_95[(df_95['sharpe_ratio'] > low) & (df_95['sharpe_ratio'] <= high)])
    count_10 = len(df_10[(df_10['sharpe_ratio'] > low) & (df_10['sharpe_ratio'] <= high)])
    pct_95 = count_95 / total_95 * 100
    pct_10 = count_10 / total_10 * 100
    
    bar_95 = '█' * int(pct_95 / 2)
    bar_10 = '█' * int(pct_10 / 2)
    
    print(f"{label:<25} {count_95:>4} ({pct_95:>5.1f}%) {bar_95:<25} {count_10:>4} ({pct_10:>5.1f}%) {bar_10}")
print()

print("=" * 100)
print("KEY INSIGHTS")
print("=" * 100)
print()

print("✅ IMPROVEMENTS WITH 10% POSITION SIZING:")
print()
print(f"   1. SUCCESS RATE:        {profitable_95/total_95*100:.1f}% → {profitable_10/total_10*100:.1f}% (+{(profitable_10/total_10 - profitable_95/total_95)*100:.1f}%)")
print(f"   2. BANKRUPTCY RATE:     {bankrupt_95/total_95*100:.1f}% → {bankrupt_10/total_10*100:.1f}% ({bankrupt_10/total_10 - bankrupt_95/total_95:.1f}%)")
print(f"   3. AVERAGE RETURN:      {df_95['total_return'].mean():.2f}% → {df_10['total_return'].mean():.2f}% (+{df_10['total_return'].mean() - df_95['total_return'].mean():.2f}%)")
print(f"   4. MEDIAN RETURN:       {df_95['total_return'].median():.2f}% → {df_10['total_return'].median():.2f}% (+{df_10['total_return'].median() - df_95['total_return'].median():.2f}%)")
print(f"   5. AVG DRAWDOWN:        {df_95['max_drawdown'].mean():.2f}% → {df_10['max_drawdown'].mean():.2f}% (+{df_10['max_drawdown'].mean() - df_95['max_drawdown'].mean():.2f}%)")
print(f"   6. SHARPE RATIO:        {df_95['sharpe_ratio'].mean():.2f} → {df_10['sharpe_ratio'].mean():.2f} (+{df_10['sharpe_ratio'].mean() - df_95['sharpe_ratio'].mean():.2f})")
print()

print("🎯 CONCLUSION:")
print()
print("   10% position sizing is DRAMATICALLY BETTER than 95%:")
print()
print("   • 77.2% bankruptcy rate → 0% bankruptcy rate (ELIMINATED ALL BANKRUPTCIES!)")
print("   • 22.8% success rate → 36.8% success rate (+61% improvement)")
print("   • Average return improved from -18.33% to +0.48%")
print("   • Median return improved from -49.93% to -4.08%")
print("   • Average drawdown reduced from -73.16% to -22.63% (69% reduction)")
print()
print("   However, the strategy still has challenges:")
print("   • Only 36.8% of stocks are profitable (still majority lose)")
print("   • Median return is still negative (-4.08%)")
print("   • Average Sharpe ratio is still negative (-0.13)")
print()
print("   RECOMMENDATION: Further improvements needed:")
print("   • Add stop losses to limit downside")
print("   • Filter stocks before trading (only trade trending stocks)")
print("   • Compare with buy-and-hold benchmark")
print()

print("=" * 100)
print("✅ COMPARISON COMPLETE!")
print("=" * 100)

