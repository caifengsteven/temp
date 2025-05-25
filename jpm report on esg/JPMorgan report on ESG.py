import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# Define simulation parameters
num_stocks = 500
num_days = 1000
start_date = datetime(2020, 1, 1)
dates = [start_date + timedelta(days=i) for i in range(num_days)]

# Create simulated ESG data components
def generate_esg_data():
    # 1. Stable ESG scores (slow-moving, updated annually)
    base_esg_scores = np.random.normal(5, 1.5, num_stocks)
    base_esg_scores = np.clip(base_esg_scores, 0, 10)  # ESG scores typically on 0-10 scale
    
    # Make these scores change slowly over time (annual updates)
    stable_esg = np.zeros((num_days, num_stocks))
    for i in range(num_stocks):
        score = base_esg_scores[i]
        for d in range(num_days):
            if d % 60 == 0 and d > 0:  # Update roughly every quarter
                # Small random adjustment
                score += np.random.normal(0, 0.2)
                score = np.clip(score, 0, 10)
            stable_esg[d, i] = score
    
    # 2. Fast-moving ESG data (newsflow/controversies)
    fast_esg = np.zeros((num_days, num_stocks))
    # Baseline fast ESG is correlated with stable ESG
    for i in range(num_stocks):
        baseline = base_esg_scores[i] * 0.7 + np.random.normal(3, 1)
        baseline = np.clip(baseline, 0, 10)
        
        # Add random walk component
        random_walk = np.zeros(num_days)
        for d in range(1, num_days):
            random_walk[d] = random_walk[d-1] + np.random.normal(0, 0.02)
        
        # Add occasional "controversy" events (negative spikes)
        controversies = np.zeros(num_days)
        if base_esg_scores[i] < 5:  # Lower ESG stocks more likely to have controversies
            num_controversies = np.random.poisson(3)  # More controversies for low ESG stocks
        else:
            num_controversies = np.random.poisson(1)  # Fewer controversies for high ESG stocks
            
        for _ in range(num_controversies):
            controversy_day = np.random.randint(0, num_days)
            severity = np.random.exponential(2) + 1
            # Controversy has immediate impact and then fades over time
            for d in range(controversy_day, min(controversy_day + 30, num_days)):
                decay = np.exp(-(d - controversy_day) / 5)
                controversies[d] -= severity * decay
        
        # Combine baseline, random walk, and controversies
        signal = baseline + random_walk + controversies
        fast_esg[:, i] = np.clip(signal, 0, 10)
    
    # Combine stable and fast ESG components
    combined_esg = 0.7 * stable_esg + 0.3 * fast_esg
    
    return stable_esg, fast_esg, combined_esg

# Generate stock returns data that's influenced by ESG factors
def generate_stock_returns(combined_esg):
    # Base market returns (common factor)
    market_returns = np.random.normal(0.0005, 0.01, num_days)
    cumulative_market = np.cumprod(1 + market_returns) - 1
    
    # Stock-specific returns
    stock_returns = np.zeros((num_days, num_stocks))
    
    for i in range(num_stocks):
        # Stock-specific volatility and beta
        volatility = np.random.uniform(0.01, 0.03)
        beta = np.random.uniform(0.7, 1.3)
        
        # ESG impact on returns (higher ESG, slightly better returns)
        avg_esg = np.mean(combined_esg[:, i])
        esg_impact = (avg_esg - 5) * 0.0001  # Small daily impact
        
        # Daily stock returns
        for d in range(num_days):
            # Market component
            market_component = beta * market_returns[d]
            
            # ESG component (with lag effect - ESG impacts future performance)
            current_esg = combined_esg[d, i]
            esg_component = esg_impact
            
            # Idiosyncratic component
            idiosyncratic = np.random.normal(0, volatility)
            
            # Total return
            stock_returns[d, i] = market_component + esg_component + idiosyncratic
    
    return stock_returns, market_returns

# Generate ESG data
stable_esg, fast_esg, combined_esg = generate_esg_data()

# Generate stock returns
stock_returns, market_returns = generate_stock_returns(combined_esg)

# Create DataFrame for easier analysis
dates_df = pd.DataFrame(index=dates)
dates_df['market_returns'] = market_returns
dates_df['market_cumulative'] = np.cumprod(1 + market_returns) - 1

# Create an ESGQ metric following JPMorgan's approach
def create_esgq_metric(stable_esg, fast_esg, combined_esg, window=60):
    esgq = np.zeros((num_days, num_stocks))
    
    for d in range(num_days):
        for i in range(num_stocks):
            # Component 1: Stable ESG score (normalized)
            stable_component = stable_esg[d, i] / 10
            
            # Component 2: Fast-moving ESG data (newsflow/controversies)
            fast_component = fast_esg[d, i] / 10
            
            # Component 3: ESG Momentum (3-month change in combined score)
            if d >= window:
                momentum = (combined_esg[d, i] - combined_esg[d-window, i]) / combined_esg[d-window, i]
                # Scale momentum to 0-1 range
                momentum = (momentum + 0.2) / 0.4  # Assuming most changes within ±20%
                momentum = np.clip(momentum, 0, 1)
            else:
                momentum = 0.5  # Neutral for initial period
            
            # JPM's ESGQ combines these components
            esgq[d, i] = 0.5 * stable_component + 0.3 * fast_component + 0.2 * momentum
    
    return esgq

# Calculate ESGQ metric
esgq = create_esgq_metric(stable_esg, fast_esg, combined_esg)

# Backtest the ESGQ strategy
def backtest_strategy(esgq, stock_returns, window=20, top_pct=0.2, rebalance_freq=20):
    portfolio_returns = np.zeros(num_days)
    holdings = np.zeros(num_stocks)
    
    for d in range(window, num_days):
        # Rebalance portfolio periodically
        if d % rebalance_freq == 0:
            # Select top stocks based on ESGQ
            prev_esgq = esgq[d-1, :]
            threshold = np.percentile(prev_esgq, 100 - top_pct * 100)
            selected = prev_esgq >= threshold
            
            # Equal weight portfolio
            holdings = np.zeros(num_stocks)
            if np.sum(selected) > 0:
                holdings[selected] = 1 / np.sum(selected)
        
        # Calculate portfolio return
        portfolio_returns[d] = np.sum(holdings * stock_returns[d, :])
    
    return portfolio_returns

# Backtest different strategies
# 1. ESGQ strategy
esgq_returns = backtest_strategy(esgq, stock_returns)

# 2. Stable ESG only strategy
stable_only_returns = backtest_strategy(stable_esg, stock_returns)

# 3. Fast ESG only strategy
fast_only_returns = backtest_strategy(fast_esg, stock_returns)

# 4. Combined but no momentum
combined_no_momentum = 0.7 * stable_esg + 0.3 * fast_esg
combined_no_momentum_returns = backtest_strategy(combined_no_momentum, stock_returns)

# Create comparison DataFrame
performance_df = dates_df.copy()
performance_df['ESGQ_returns'] = esgq_returns
performance_df['Stable_ESG_returns'] = stable_only_returns
performance_df['Fast_ESG_returns'] = fast_only_returns
performance_df['Combined_no_momentum_returns'] = combined_no_momentum_returns

# Calculate cumulative returns
for col in ['ESGQ_returns', 'Stable_ESG_returns', 'Fast_ESG_returns', 'Combined_no_momentum_returns']:
    performance_df[f'{col}_cumulative'] = np.cumprod(1 + performance_df[col]) - 1

# Calculate strategy performance metrics
def calculate_performance_metrics(returns):
    metrics = {}
    metrics['annualized_return'] = returns.mean() * 252
    metrics['annualized_volatility'] = returns.std() * np.sqrt(252)
    metrics['sharpe_ratio'] = metrics['annualized_return'] / metrics['annualized_volatility'] if metrics['annualized_volatility'] > 0 else 0
    metrics['max_drawdown'] = (1 + returns).cumprod().div((1 + returns).cumprod().cummax()).min() - 1
    metrics['hit_rate'] = (returns > 0).mean()
    metrics['t_stat'] = stats.ttest_1samp(returns, 0)[0]
    return metrics

# Calculate performance metrics for all strategies
strategies = ['ESGQ_returns', 'Stable_ESG_returns', 'Fast_ESG_returns', 'Combined_no_momentum_returns']
metrics_dict = {}

for strategy in strategies:
    metrics_dict[strategy] = calculate_performance_metrics(performance_df[strategy].iloc[20:])

# Add market performance metrics
metrics_dict['market_returns'] = calculate_performance_metrics(performance_df['market_returns'])

metrics_df = pd.DataFrame(metrics_dict).T
metrics_df = metrics_df.rename(columns={
    'annualized_return': 'Ann. Ret',
    'annualized_volatility': 'Ann. Vol.',
    'sharpe_ratio': 'Sharpe',
    'max_drawdown': 'Max DD',
    'hit_rate': 'Hit-Rate',
    't_stat': 'T-Stat'
})

# Sort by Sharpe ratio
metrics_df = metrics_df.sort_values('Sharpe', ascending=False)

# Print performance metrics instead of using display()
print("Performance Metrics:")
print(metrics_df)

# Plot cumulative returns
plt.figure(figsize=(12, 8))
plt.plot(performance_df.index, performance_df['ESGQ_returns_cumulative'], label='ESGQ Strategy')
plt.plot(performance_df.index, performance_df['Stable_ESG_returns_cumulative'], label='Stable ESG Strategy')
plt.plot(performance_df.index, performance_df['Fast_ESG_returns_cumulative'], label='Fast ESG Strategy')
plt.plot(performance_df.index, performance_df['Combined_no_momentum_returns_cumulative'], label='Combined (No Momentum)')
plt.plot(performance_df.index, performance_df['market_cumulative'], label='Market')
plt.xlabel('Date')
plt.ylabel('Cumulative Return')
plt.title('Strategy Comparison: Cumulative Returns')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot monthly returns for the best strategy
plt.figure(figsize=(12, 6))
best_strategy = metrics_df.index[0]
monthly_returns = performance_df[best_strategy].resample('M').sum()
monthly_returns.plot(kind='bar')
plt.title(f'Monthly Returns for {best_strategy.replace("_returns", "")} Strategy')
plt.ylabel('Monthly Return')
plt.xlabel('Month')
plt.grid(True, alpha=0.3)

# Plot a heatmap of the correlation between different ESG components and returns
# Create correlation data
corr_data = np.zeros((num_days, 5))
for d in range(num_days):
    corr_data[d, 0] = np.mean(stable_esg[d, :])
    corr_data[d, 1] = np.mean(fast_esg[d, :])
    corr_data[d, 2] = np.mean(esgq[d, :])
    corr_data[d, 3] = np.mean(stock_returns[d, :])
    corr_data[d, 4] = market_returns[d]

corr_df = pd.DataFrame(corr_data, columns=['Stable ESG', 'Fast ESG', 'ESGQ', 'Stock Returns', 'Market Returns'])
plt.figure(figsize=(10, 8))
sns.heatmap(corr_df.corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Heatmap')

# Plot ESG distribution by returns quintile
plt.figure(figsize=(12, 6))
# Get average returns for each stock
avg_returns = np.mean(stock_returns, axis=0)
avg_esgq = np.mean(esgq, axis=0)

# Create quintiles based on returns
quintiles = pd.qcut(avg_returns, 5, labels=False)
quintile_esg = [[] for _ in range(5)]

for i in range(len(quintiles)):
    quintile = quintiles[i]
    quintile_esg[quintile].append(avg_esgq[i])

plt.boxplot(quintile_esg)
plt.title('ESGQ Score Distribution by Return Quintile')
plt.xlabel('Return Quintile (Low to High)')
plt.ylabel('Average ESGQ Score')

plt.tight_layout()

# Test combining ESGQ with traditional factors
def create_factor_exposure(num_stocks):
    # Simulate exposure to traditional factors
    factors = {}
    
    # Value factor
    factors['value'] = np.random.normal(0, 1, num_stocks)
    
    # Momentum factor
    factors['momentum'] = np.random.normal(0, 1, num_stocks)
    
    # Quality factor
    factors['quality'] = np.random.normal(0, 1, num_stocks)
    
    # Size factor (negative = small cap)
    factors['size'] = np.random.normal(0, 1, num_stocks)
    
    # Low volatility factor
    factors['low_vol'] = np.random.normal(0, 1, num_stocks)
    
    return factors

# Create factor exposures
factor_exposures = create_factor_exposure(num_stocks)

# Backtest ESGQ combined with traditional factors
def backtest_combined_strategy(esgq, factor, stock_returns, weight_esgq=0.5, window=20, top_pct=0.2, rebalance_freq=20):
    portfolio_returns = np.zeros(num_days)
    holdings = np.zeros(num_stocks)
    
    # Standardize factor
    factor_std = (factor - np.mean(factor)) / np.std(factor)
    
    for d in range(window, num_days):
        # Rebalance portfolio periodically
        if d % rebalance_freq == 0:
            # Get ESGQ score
            prev_esgq = esgq[d-1, :]
            esgq_std = (prev_esgq - np.mean(prev_esgq)) / np.std(prev_esgq)
            
            # Combine ESGQ with factor
            combined_score = weight_esgq * esgq_std + (1 - weight_esgq) * factor_std
            
            # Select top stocks based on combined score
            threshold = np.percentile(combined_score, 100 - top_pct * 100)
            selected = combined_score >= threshold
            
            # Equal weight portfolio
            holdings = np.zeros(num_stocks)
            if np.sum(selected) > 0:
                holdings[selected] = 1 / np.sum(selected)
        
        # Calculate portfolio return
        portfolio_returns[d] = np.sum(holdings * stock_returns[d, :])
    
    return portfolio_returns

# Test ESGQ with different factors
combined_strategies = {}
for factor_name, factor_values in factor_exposures.items():
    combined_strategies[f'ESGQ_with_{factor_name}'] = backtest_combined_strategy(esgq, factor_values, stock_returns)

# Add combined strategies to performance DataFrame
for name, returns in combined_strategies.items():
    performance_df[name] = returns
    performance_df[f'{name}_cumulative'] = np.cumprod(1 + returns) - 1
    
# Calculate performance metrics for combined strategies
for name in combined_strategies.keys():
    metrics_dict[name] = calculate_performance_metrics(performance_df[name].iloc[20:])

# Create updated metrics DataFrame
metrics_df = pd.DataFrame(metrics_dict).T
metrics_df = metrics_df.rename(columns={
    'annualized_return': 'Ann. Ret',
    'annualized_volatility': 'Ann. Vol.',
    'sharpe_ratio': 'Sharpe',
    'max_drawdown': 'Max DD',
    'hit_rate': 'Hit-Rate',
    't_stat': 'T-Stat'
})

# Sort by Sharpe ratio
metrics_df = metrics_df.sort_values('Sharpe', ascending=False)

print("\nPerformance Metrics Including Combined Strategies:")
print(metrics_df.head(10))

# Plot cumulative returns for best combined strategy vs ESGQ alone
best_combined = [col for col in metrics_df.index if col.startswith('ESGQ_with_')][0]
plt.figure(figsize=(12, 8))
plt.plot(performance_df.index, performance_df['ESGQ_returns_cumulative'], label='ESGQ Strategy')
plt.plot(performance_df.index, performance_df[f'{best_combined}_cumulative'], label=f'{best_combined}')
plt.plot(performance_df.index, performance_df['market_cumulative'], label='Market')
plt.xlabel('Date')
plt.ylabel('Cumulative Return')
plt.title('ESGQ vs Best Combined Strategy: Cumulative Returns')
plt.legend()
plt.grid(True, alpha=0.3)

# Generate summary findings
print("\nKey Findings from ESG Strategy Simulation:")
print("1. ESGQ Performance:")
best_strategy = metrics_df.index[0]
worst_strategy = metrics_df.index[-1]
print(f"   - Best strategy: {best_strategy} with Sharpe ratio of {metrics_df.loc[best_strategy, 'Sharpe']:.2f}")
print(f"   - Original ESGQ Sharpe ratio: {metrics_df.loc['ESGQ_returns', 'Sharpe']:.2f}")
print(f"   - Worst strategy: {worst_strategy} with Sharpe ratio of {metrics_df.loc[worst_strategy, 'Sharpe']:.2f}")

print("\n2. Component Analysis:")
print(f"   - Stable ESG component Sharpe: {metrics_df.loc['Stable_ESG_returns', 'Sharpe']:.2f}")
print(f"   - Fast ESG component Sharpe: {metrics_df.loc['Fast_ESG_returns', 'Sharpe']:.2f}")
print(f"   - Combined without momentum Sharpe: {metrics_df.loc['Combined_no_momentum_returns', 'Sharpe']:.2f}")
print(f"   - Full ESGQ with momentum Sharpe: {metrics_df.loc['ESGQ_returns', 'Sharpe']:.2f}")

# Analyze risk reduction properties of ESGQ
market_vol = metrics_df.loc['market_returns', 'Ann. Vol.'] if 'market_returns' in metrics_df.index else performance_df['market_returns'].std() * np.sqrt(252)
esgq_vol = metrics_df.loc['ESGQ_returns', 'Ann. Vol.']
vol_reduction = (market_vol - esgq_vol) / market_vol

print("\n3. Risk Reduction:")
print(f"   - Market volatility: {market_vol:.2%}")
print(f"   - ESGQ strategy volatility: {esgq_vol:.2%}")
print(f"   - Volatility reduction: {vol_reduction:.2%}")
print(f"   - ESGQ max drawdown: {metrics_df.loc['ESGQ_returns', 'Max DD']:.2%}")

print("\n4. Factor Integration:")
best_factor = best_combined.replace('ESGQ_with_', '')
print(f"   - Best factor to combine with ESGQ: {best_factor}")
print(f"   - Sharpe improvement: {(metrics_df.loc[best_combined, 'Sharpe'] - metrics_df.loc['ESGQ_returns', 'Sharpe']):.2f} increase in Sharpe ratio")

plt.show()