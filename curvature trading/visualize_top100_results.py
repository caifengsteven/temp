"""
Visualize Top 100 Backtest Results
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

# Read results
df = pd.read_csv('backtest_top100_2020_2023.csv')

# Create comprehensive visualization
fig = plt.figure(figsize=(20, 12))
fig.patch.set_facecolor('#000000')
gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)

# Color scheme
bg_color = '#000000'
text_color = '#ffffff'
grid_color = '#333333'
positive_color = '#00ff00'
negative_color = '#ff0000'
neutral_color = '#00d9d9'

# 1. Top 20 Performers by Return
ax1 = fig.add_subplot(gs[0, :2])
ax1.set_facecolor(bg_color)
top20 = df.nlargest(20, 'total_return')
colors = [positive_color if x > 0 else negative_color for x in top20['total_return']]
bars = ax1.barh(range(len(top20)), top20['total_return'] / 1e9, color=colors, alpha=0.7)
ax1.set_yticks(range(len(top20)))
ax1.set_yticklabels(top20['ticker'], color=text_color, fontsize=10)
ax1.set_xlabel('Total Return (Billions of %)', color=text_color, fontsize=12)
ax1.set_title('Top 20 Performers by Total Return', color=text_color, fontsize=14, fontweight='bold')
ax1.tick_params(colors=text_color)
ax1.grid(True, alpha=0.2, color=grid_color)
ax1.spines['bottom'].set_color(text_color)
ax1.spines['left'].set_color(text_color)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# Add value labels
for i, (idx, row) in enumerate(top20.iterrows()):
    value = row['total_return'] / 1e9
    ax1.text(value, i, f' {value:.1f}B', va='center', color=text_color, fontsize=8)

# 2. Sharpe Ratio Distribution
ax2 = fig.add_subplot(gs[0, 2])
ax2.set_facecolor(bg_color)
sharpe_bins = np.arange(-0.5, 4.0, 0.25)
counts, bins, patches = ax2.hist(df['sharpe_ratio'], bins=sharpe_bins, color=neutral_color, alpha=0.7, edgecolor='white')
ax2.axvline(df['sharpe_ratio'].mean(), color=positive_color, linestyle='--', linewidth=2, label=f'Mean: {df["sharpe_ratio"].mean():.2f}')
ax2.axvline(df['sharpe_ratio'].median(), color='yellow', linestyle='--', linewidth=2, label=f'Median: {df["sharpe_ratio"].median():.2f}')
ax2.set_xlabel('Sharpe Ratio', color=text_color, fontsize=12)
ax2.set_ylabel('Count', color=text_color, fontsize=12)
ax2.set_title('Sharpe Ratio Distribution', color=text_color, fontsize=14, fontweight='bold')
ax2.tick_params(colors=text_color)
ax2.grid(True, alpha=0.2, color=grid_color)
legend = ax2.legend(facecolor=bg_color, edgecolor=text_color)
for text in legend.get_texts():
    text.set_color(text_color)
ax2.spines['bottom'].set_color(text_color)
ax2.spines['left'].set_color(text_color)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

# 3. Drawdown vs Return Scatter
ax3 = fig.add_subplot(gs[1, 0])
ax3.set_facecolor(bg_color)
scatter = ax3.scatter(df['max_drawdown'], np.log10(df['total_return'] + 1), 
                     c=df['sharpe_ratio'], cmap='RdYlGn', s=100, alpha=0.6, edgecolors='white')
ax3.set_xlabel('Max Drawdown (%)', color=text_color, fontsize=12)
ax3.set_ylabel('Log10(Total Return)', color=text_color, fontsize=12)
ax3.set_title('Risk vs Return', color=text_color, fontsize=14, fontweight='bold')
ax3.tick_params(colors=text_color)
ax3.grid(True, alpha=0.2, color=grid_color)
cbar = plt.colorbar(scatter, ax=ax3)
cbar.set_label('Sharpe Ratio', color=text_color)
cbar.ax.yaxis.set_tick_params(color=text_color)
plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color=text_color)
ax3.spines['bottom'].set_color(text_color)
ax3.spines['left'].set_color(text_color)
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)

# 4. Win Rate Distribution
ax4 = fig.add_subplot(gs[1, 1])
ax4.set_facecolor(bg_color)
win_rate_bins = np.arange(0, 101, 5)
counts, bins, patches = ax4.hist(df['win_rate'], bins=win_rate_bins, color=neutral_color, alpha=0.7, edgecolor='white')
ax4.axvline(df['win_rate'].mean(), color=positive_color, linestyle='--', linewidth=2, label=f'Mean: {df["win_rate"].mean():.1f}%')
ax4.axvline(50, color=negative_color, linestyle=':', linewidth=2, label='50% (Break-even)')
ax4.set_xlabel('Win Rate (%)', color=text_color, fontsize=12)
ax4.set_ylabel('Count', color=text_color, fontsize=12)
ax4.set_title('Win Rate Distribution', color=text_color, fontsize=14, fontweight='bold')
ax4.tick_params(colors=text_color)
ax4.grid(True, alpha=0.2, color=grid_color)
legend = ax4.legend(facecolor=bg_color, edgecolor=text_color)
for text in legend.get_texts():
    text.set_color(text_color)
ax4.spines['bottom'].set_color(text_color)
ax4.spines['left'].set_color(text_color)
ax4.spines['top'].set_visible(False)
ax4.spines['right'].set_visible(False)

# 5. Trade Count Distribution
ax5 = fig.add_subplot(gs[1, 2])
ax5.set_facecolor(bg_color)
trade_bins = np.arange(0, 45, 2)
counts, bins, patches = ax5.hist(df['total_trades'], bins=trade_bins, color=neutral_color, alpha=0.7, edgecolor='white')
ax5.axvline(df['total_trades'].mean(), color=positive_color, linestyle='--', linewidth=2, label=f'Mean: {df["total_trades"].mean():.1f}')
ax5.set_xlabel('Total Trades (4 years)', color=text_color, fontsize=12)
ax5.set_ylabel('Count', color=text_color, fontsize=12)
ax5.set_title('Trade Frequency Distribution', color=text_color, fontsize=14, fontweight='bold')
ax5.tick_params(colors=text_color)
ax5.grid(True, alpha=0.2, color=grid_color)
legend = ax5.legend(facecolor=bg_color, edgecolor=text_color)
for text in legend.get_texts():
    text.set_color(text_color)
ax5.spines['bottom'].set_color(text_color)
ax5.spines['left'].set_color(text_color)
ax5.spines['top'].set_visible(False)
ax5.spines['right'].set_visible(False)

# 6. Top 20 by Sharpe Ratio
ax6 = fig.add_subplot(gs[2, :2])
ax6.set_facecolor(bg_color)
top20_sharpe = df.nlargest(20, 'sharpe_ratio')
colors_sharpe = [positive_color if x > 2.0 else neutral_color for x in top20_sharpe['sharpe_ratio']]
bars = ax6.barh(range(len(top20_sharpe)), top20_sharpe['sharpe_ratio'], color=colors_sharpe, alpha=0.7)
ax6.set_yticks(range(len(top20_sharpe)))
ax6.set_yticklabels(top20_sharpe['ticker'], color=text_color, fontsize=10)
ax6.set_xlabel('Sharpe Ratio', color=text_color, fontsize=12)
ax6.set_title('Top 20 by Sharpe Ratio (Risk-Adjusted Returns)', color=text_color, fontsize=14, fontweight='bold')
ax6.tick_params(colors=text_color)
ax6.grid(True, alpha=0.2, color=grid_color)
ax6.axvline(2.0, color='yellow', linestyle=':', linewidth=2, alpha=0.5, label='Excellent (2.0)')
ax6.axvline(3.0, color=positive_color, linestyle=':', linewidth=2, alpha=0.5, label='Outstanding (3.0)')
legend = ax6.legend(facecolor=bg_color, edgecolor=text_color)
for text in legend.get_texts():
    text.set_color(text_color)
ax6.spines['bottom'].set_color(text_color)
ax6.spines['left'].set_color(text_color)
ax6.spines['top'].set_visible(False)
ax6.spines['right'].set_visible(False)

# Add value labels
for i, (idx, row) in enumerate(top20_sharpe.iterrows()):
    value = row['sharpe_ratio']
    ax6.text(value, i, f' {value:.2f}', va='center', color=text_color, fontsize=8)

# 7. Summary Statistics Box
ax7 = fig.add_subplot(gs[2, 2])
ax7.set_facecolor(bg_color)
ax7.axis('off')

summary_text = f"""
SUMMARY STATISTICS
{'='*30}

Total Stocks:        {len(df)}
Profitable:          {len(df[df['total_return'] > 0])} ({len(df[df['total_return'] > 0])/len(df)*100:.1f}%)
Losing:              {len(df[df['total_return'] <= 0])} ({len(df[df['total_return'] <= 0])/len(df)*100:.1f}%)

RETURNS
Median Return:       {df['total_return'].median()/1e6:.1f}M%
Best Return:         {df['total_return'].max()/1e9:.1f}B%
Worst Return:        {df['total_return'].min():.1f}%

RISK METRICS
Avg Sharpe:          {df['sharpe_ratio'].mean():.2f}
Avg Drawdown:        {df['max_drawdown'].mean():.2f}%
Avg Win Rate:        {df['win_rate'].mean():.1f}%
Avg Profit Factor:   {df['profit_factor'].mean():.2f}

TRADING
Total Trades:        {df['total_trades'].sum()}
Avg Trades/Stock:    {df['total_trades'].mean():.1f}
Avg Hold Period:     {df['avg_bars_held'].mean():.1f} days

RATING: ⭐⭐⭐⭐⭐
"""

ax7.text(0.1, 0.95, summary_text, transform=ax7.transAxes, 
         fontsize=11, verticalalignment='top', fontfamily='monospace',
         color=text_color, bbox=dict(boxstyle='round', facecolor=bg_color, 
         edgecolor=positive_color, linewidth=2))

# Main title
fig.suptitle('Top 100 US Stocks Backtest Results (2020-2023)\nCurved Radius Supertrend Strategy', 
             fontsize=18, fontweight='bold', color=text_color, y=0.98)

plt.tight_layout()
plt.savefig('top100_backtest_visualization.png', dpi=150, facecolor=bg_color, edgecolor='none')
print("✅ Visualization saved to: top100_backtest_visualization.png")

# Create a second chart showing sector analysis
fig2, axes = plt.subplots(2, 2, figsize=(16, 12))
fig2.patch.set_facecolor(bg_color)

# Define sectors (simplified)
sector_map = {
    'AAPL': 'Tech', 'MSFT': 'Tech', 'GOOGL': 'Tech', 'AMZN': 'Tech', 'NVDA': 'Tech',
    'META': 'Tech', 'TSLA': 'Auto', 'JPM': 'Finance', 'V': 'Finance', 'MA': 'Finance',
    'JNJ': 'Healthcare', 'UNH': 'Healthcare', 'PG': 'Consumer', 'KO': 'Consumer',
    'XOM': 'Energy', 'CVX': 'Energy', 'WMT': 'Retail', 'HD': 'Retail', 'COST': 'Retail',
    'DIS': 'Media', 'NFLX': 'Media', 'BA': 'Aerospace', 'CAT': 'Industrial',
    'GE': 'Industrial', 'IBM': 'Tech', 'INTC': 'Tech', 'AMD': 'Tech', 'ORCL': 'Tech',
    'AXP': 'Finance', 'MS': 'Finance', 'C': 'Finance', 'WFC': 'Finance',
    'MRK': 'Healthcare', 'ABBV': 'Healthcare', 'TMO': 'Healthcare', 'ABT': 'Healthcare'
}

df['sector'] = df['ticker'].map(sector_map).fillna('Other')

# 1. Average Return by Sector
ax = axes[0, 0]
ax.set_facecolor(bg_color)
sector_returns = df.groupby('sector')['total_return'].mean().sort_values(ascending=True) / 1e9
colors = [positive_color if x > 0 else negative_color for x in sector_returns]
sector_returns.plot(kind='barh', ax=ax, color=colors, alpha=0.7)
ax.set_xlabel('Avg Return (Billions of %)', color=text_color, fontsize=12)
ax.set_title('Average Return by Sector', color=text_color, fontsize=14, fontweight='bold')
ax.tick_params(colors=text_color)
ax.grid(True, alpha=0.2, color=grid_color)
ax.spines['bottom'].set_color(text_color)
ax.spines['left'].set_color(text_color)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# 2. Average Sharpe by Sector
ax = axes[0, 1]
ax.set_facecolor(bg_color)
sector_sharpe = df.groupby('sector')['sharpe_ratio'].mean().sort_values(ascending=True)
colors = [positive_color if x > 2.0 else neutral_color for x in sector_sharpe]
sector_sharpe.plot(kind='barh', ax=ax, color=colors, alpha=0.7)
ax.set_xlabel('Avg Sharpe Ratio', color=text_color, fontsize=12)
ax.set_title('Average Sharpe Ratio by Sector', color=text_color, fontsize=14, fontweight='bold')
ax.tick_params(colors=text_color)
ax.grid(True, alpha=0.2, color=grid_color)
ax.axvline(2.0, color='yellow', linestyle=':', linewidth=2, alpha=0.5)
ax.spines['bottom'].set_color(text_color)
ax.spines['left'].set_color(text_color)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# 3. Stocks per Sector
ax = axes[1, 0]
ax.set_facecolor(bg_color)
sector_counts = df['sector'].value_counts()
colors_pie = plt.cm.Set3(range(len(sector_counts)))
wedges, texts, autotexts = ax.pie(sector_counts, labels=sector_counts.index, autopct='%1.1f%%',
                                    colors=colors_pie, startangle=90)
for text in texts:
    text.set_color(text_color)
for autotext in autotexts:
    autotext.set_color(bg_color)
    autotext.set_fontweight('bold')
ax.set_title('Stock Distribution by Sector', color=text_color, fontsize=14, fontweight='bold')

# 4. Success Rate by Sector
ax = axes[1, 1]
ax.set_facecolor(bg_color)
sector_success = df.groupby('sector').apply(lambda x: (x['total_return'] > 0).sum() / len(x) * 100).sort_values(ascending=True)
colors = [positive_color if x >= 90 else neutral_color for x in sector_success]
sector_success.plot(kind='barh', ax=ax, color=colors, alpha=0.7)
ax.set_xlabel('Success Rate (%)', color=text_color, fontsize=12)
ax.set_title('Success Rate by Sector', color=text_color, fontsize=14, fontweight='bold')
ax.tick_params(colors=text_color)
ax.grid(True, alpha=0.2, color=grid_color)
ax.axvline(90, color='yellow', linestyle=':', linewidth=2, alpha=0.5, label='90%')
legend = ax.legend(facecolor=bg_color, edgecolor=text_color)
for text in legend.get_texts():
    text.set_color(text_color)
ax.spines['bottom'].set_color(text_color)
ax.spines['left'].set_color(text_color)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

fig2.suptitle('Sector Analysis - Top 100 Stocks (2020-2023)', 
              fontsize=18, fontweight='bold', color=text_color, y=0.98)

plt.tight_layout()
plt.savefig('top100_sector_analysis.png', dpi=150, facecolor=bg_color, edgecolor='none')
print("✅ Sector analysis saved to: top100_sector_analysis.png")

print("\n📊 All visualizations complete!")

