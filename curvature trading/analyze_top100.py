"""
Analyze Top 100 US Stocks Performance from the backtest results
"""

import pandas as pd
import numpy as np

# Load the full results
df = pd.read_csv('backtest_top1000_2015_2025.csv')

# Get the top 100 stocks (first 100 from our original list)
top_100_tickers = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK.B', 'UNH', 'XOM',
    'JNJ', 'JPM', 'V', 'PG', 'MA', 'HD', 'CVX', 'MRK', 'ABBV', 'PEP',
    'COST', 'AVGO', 'KO', 'ADBE', 'WMT', 'MCD', 'CSCO', 'CRM', 'ACN', 'TMO',
    'LIN', 'ABT', 'NFLX', 'NKE', 'DHR', 'VZ', 'TXN', 'ORCL', 'PM', 'DIS',
    'CMCSA', 'INTC', 'WFC', 'AMD', 'UPS', 'NEE', 'COP', 'RTX', 'QCOM', 'HON',
    'INTU', 'UNP', 'IBM', 'LOW', 'AMGN', 'BA', 'SPGI', 'ELV', 'AMAT', 'GE',
    'CAT', 'SBUX', 'DE', 'PLD', 'BKNG', 'GILD', 'ADP', 'ADI', 'TJX', 'MDLZ',
    'CVS', 'LMT', 'SYK', 'VRTX', 'AXP', 'ISRG', 'MMC', 'CI', 'REGN', 'BLK',
    'ZTS', 'PGR', 'TMUS', 'MO', 'CB', 'SO', 'DUK', 'BSX', 'ETN', 'SCHW',
    'C', 'EOG', 'ITW', 'HCA', 'PNC', 'NOC', 'USB', 'SLB', 'MS', 'GD'
]

# Filter for top 100
top100 = df[df['ticker'].isin(top_100_tickers)].copy()

print('=' * 80)
print('TOP 100 US STOCKS BACKTEST RESULTS (2015-2025)')
print('=' * 80)
print()

# Overall statistics
total = len(top100)
profitable = len(top100[top100['total_return'] > 0])
bankrupt = len(top100[top100['total_return'] <= -100])

print(f'📊 OVERALL STATISTICS')
print(f'   Total Stocks Tested:     {total}')
print(f'   Profitable Stocks:       {profitable} ({profitable/total*100:.1f}%)')
print(f'   Losing Stocks:           {total - profitable} ({(total-profitable)/total*100:.1f}%)')
print(f'   Bankruptcies (-100%):    {bankrupt} ({bankrupt/total*100:.1f}%)')
print()

# Return statistics
print(f'📈 RETURN STATISTICS')
print(f'   Average Return:          {top100["total_return"].mean():>15.2f}%')
print(f'   Median Return:           {top100["total_return"].median():>15.2f}%')
print(f'   Best Return:             {top100["total_return"].max():>15.2f}%')
print(f'   Worst Return:            {top100["total_return"].min():>15.2f}%')
print(f'   Std Deviation:           {top100["total_return"].std():>15.2f}%')
print()

# Risk metrics
print(f'🎯 RISK METRICS')
print(f'   Average Sharpe Ratio:    {top100["sharpe_ratio"].mean():>15.2f}')
print(f'   Median Sharpe Ratio:     {top100["sharpe_ratio"].median():>15.2f}')
print(f'   Average Max Drawdown:    {top100["max_drawdown"].mean():>15.2f}%')
print(f'   Average Win Rate:        {top100["win_rate"].mean():>15.2f}%')
print(f'   Average Profit Factor:   {top100["profit_factor"].mean():>15.2f}')
print()

# Trade statistics
print(f'📊 TRADE STATISTICS')
print(f'   Total Trades (All):      {top100["total_trades"].sum():>10.0f}')
print(f'   Average Trades/Stock:    {top100["total_trades"].mean():>15.1f}')
print(f'   Median Trades/Stock:     {top100["total_trades"].median():>15.1f}')
print(f'   Average Holding Period:  {top100["avg_bars_held"].mean():>15.1f} days')
print()

print('=' * 80)
print('🏆 TOP 20 PERFORMERS (by Total Return)')
print('=' * 80)
print()
top20 = top100.nlargest(20, 'total_return')
print(f"{'Rank':<6} {'Ticker':<8} {'Return':<18} {'Sharpe':<10} {'Drawdown':<12} {'Trades'}")
print('-' * 80)
for i, row in enumerate(top20.itertuples(), 1):
    print(f'{i:<6} {row.ticker:<8} {row.total_return:>15.2f}% {row.sharpe_ratio:>9.2f} {row.max_drawdown:>10.2f}% {row.total_trades:>9.0f}')

print()
print('=' * 80)
print('📉 BOTTOM 20 PERFORMERS (by Total Return)')
print('=' * 80)
print()
bottom20 = top100.nsmallest(20, 'total_return')
print(f"{'Rank':<6} {'Ticker':<8} {'Return':<18} {'Sharpe':<10} {'Drawdown':<12} {'Trades'}")
print('-' * 80)
for i, row in enumerate(bottom20.itertuples(), 1):
    print(f'{i:<6} {row.ticker:<8} {row.total_return:>15.2f}% {row.sharpe_ratio:>9.2f} {row.max_drawdown:>10.2f}% {row.total_trades:>9.0f}')

print()
print('=' * 80)
print('🎯 TOP 20 BY SHARPE RATIO (Risk-Adjusted Returns)')
print('=' * 80)
print()
top_sharpe = top100.nlargest(20, 'sharpe_ratio')
print(f"{'Rank':<6} {'Ticker':<8} {'Sharpe':<10} {'Return':<18} {'Drawdown':<12} {'Trades'}")
print('-' * 80)
for i, row in enumerate(top_sharpe.itertuples(), 1):
    print(f'{i:<6} {row.ticker:<8} {row.sharpe_ratio:>9.2f} {row.total_return:>15.2f}% {row.max_drawdown:>10.2f}% {row.total_trades:>9.0f}')

print()
print('=' * 80)
print('📊 RETURN DISTRIBUTION')
print('=' * 80)
bins = [
    ('Loss > 80%', -float('inf'), -80),
    ('Loss 50-80%', -80, -50),
    ('Loss 20-50%', -50, -20),
    ('Loss 0-20%', -20, 0),
    ('Gain 0-50%', 0, 50),
    ('Gain 50-100%', 50, 100),
    ('Gain 100-500%', 100, 500),
    ('Gain 500-1000%', 500, 1000),
    ('Gain 1K-10K%', 1000, 10000),
    ('Gain > 10K%', 10000, float('inf'))
]

for label, low, high in bins:
    count = len(top100[(top100['total_return'] > low) & (top100['total_return'] <= high)])
    pct = count / total * 100
    bar = '█' * int(pct / 2)
    print(f'{label:<20} {count:>4} ({pct:>5.1f}%) {bar}')

print()
print('=' * 80)
print('📊 SHARPE RATIO DISTRIBUTION')
print('=' * 80)
sharpe_bins = [
    ('Negative (<0)', -float('inf'), 0),
    ('Poor (0-1)', 0, 1),
    ('Good (1-2)', 1, 2),
    ('Excellent (2-3)', 2, 3),
    ('Outstanding (>3)', 3, float('inf'))
]

for label, low, high in sharpe_bins:
    count = len(top100[(top100['sharpe_ratio'] > low) & (top100['sharpe_ratio'] <= high)])
    pct = count / total * 100
    bar = '█' * int(pct / 2)
    print(f'{label:<25} {count:>4} ({pct:>5.1f}%) {bar}')

print()
print('=' * 80)
print('✅ TOP 100 ANALYSIS COMPLETE!')
print('=' * 80)

# Save top 100 results
top100.to_csv('backtest_top100_2015_2025.csv', index=False)
print('📁 Results saved to: backtest_top100_2015_2025.csv')

