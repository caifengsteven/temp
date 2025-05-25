import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import random

# Set random seed for reproducibility
np.random.seed(42)

# Simulation parameters
start_date = datetime(2007, 1, 1)
end_date = datetime(2019, 6, 30)
num_companies = 800  # Simulating companies comparable to CSI 800 index
rebalance_dates = ['04-30', '08-30', '12-30']  # Rebalance dates each year

# Generate company data
def generate_company_data(start_date, end_date, num_companies):
    # Date range for simulation
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Create empty company data structure
    companies = {}
    
    # Company sectors for diversification
    sectors = ['Technology', 'Financial', 'Consumer', 'Industrial', 'Healthcare', 'Energy', 'Materials', 'Utilities']
    
    # Generate data for each company
    for i in range(num_companies):
        company_id = f'C{i+1:04d}'
        sector = random.choice(sectors)
        
        # Base financial metrics
        base_roa = np.random.normal(0.05, 0.03)  # Return on Assets
        base_cfo = np.random.normal(0.06, 0.04)  # Cash Flow from Operations
        base_leverage = np.random.normal(0.4, 0.2)  # Long-term Debt to Assets
        base_liquidity = np.random.normal(1, 0.5)  # Current Ratio
        base_gross_margin = np.random.normal(0.3, 0.1)  # Gross Margin
        base_asset_turnover = np.random.normal(0.6, 0.2)  # Asset Turnover
        
        # Company quality - determines trend direction
        quality = np.random.normal(0, 1)
        
        # Initialize company data
        company_data = []
        
        # Generate quarterly financial data
        current_date = start_date
        while current_date <= end_date:
            quarter_end = False
            if current_date.month in [3, 6, 9, 12] and current_date.day == 31:
                quarter_end = True
            
            # Add time trend and seasonal components to metrics
            if quarter_end:
                time_trend = quality * np.random.normal(0.001, 0.0005) * (current_date - start_date).days
                seasonal = np.random.normal(0, 0.01)
                
                # Calculate metrics with trends, seasonality, and random noise
                roa = base_roa + time_trend + seasonal + np.random.normal(0, 0.01)
                roa_growth = np.random.normal(0.01, 0.02) + quality * 0.005
                cfo = base_cfo + time_trend + seasonal + np.random.normal(0, 0.015)
                leverage = base_leverage - time_trend * 0.5 + np.random.normal(0, 0.02)
                liquidity = base_liquidity + time_trend * 0.2 + np.random.normal(0, 0.1)
                shares_outstanding_growth = np.random.choice([-0.02, 0, 0.05, 0.1], p=[0.1, 0.7, 0.15, 0.05])
                gross_margin = base_gross_margin + time_trend + seasonal + np.random.normal(0, 0.01)
                gross_margin_growth = np.random.normal(0.005, 0.01) + quality * 0.003
                asset_turnover = base_asset_turnover + time_trend + seasonal + np.random.normal(0, 0.02)
                asset_turnover_growth = np.random.normal(0.005, 0.01) + quality * 0.003
                
                # Store quarterly data
                company_data.append({
                    'date': current_date,
                    'roa': roa,
                    'roa_growth': roa_growth,
                    'cfo': cfo,
                    'cfo_roa_diff': cfo - roa,
                    'leverage': leverage,
                    'leverage_change': np.random.normal(-0.01, 0.02) + quality * -0.002,
                    'liquidity': liquidity,
                    'liquidity_change': np.random.normal(0.01, 0.05) + quality * 0.003,
                    'shares_outstanding_growth': shares_outstanding_growth,
                    'gross_margin': gross_margin,
                    'gross_margin_growth': gross_margin_growth,
                    'asset_turnover': asset_turnover,
                    'asset_turnover_growth': asset_turnover_growth,
                    'sector': sector
                })
            
            # Move to next day
            current_date += timedelta(days=1)
        
        # Convert to DataFrame
        companies[company_id] = pd.DataFrame(company_data)
    
    return companies

# Generate stock price data
def generate_stock_prices(companies, start_date, end_date):
    # Date range for prices
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    all_prices = {}
    
    for company_id, financial_data in companies.items():
        # Initial price and price history
        initial_price = np.random.uniform(10, 100)
        prices = [initial_price]
        
        # Base annual return (quality companies perform better)
        quality = np.mean([np.mean(financial_data['roa']), 
                           np.mean(financial_data['cfo']), 
                           np.mean(financial_data['gross_margin'])])
        base_return = 0.05 + quality * 0.1
        
        # Generate daily prices
        for i in range(1, len(date_range)):
            # Daily return
            daily_return = np.random.normal(base_return/252, 0.015)
            
            # Adjust return based on recent financial data if available
            current_date = date_range[i]
            recent_financials = financial_data[financial_data['date'] < current_date]
            
            if not recent_financials.empty:
                last_financial = recent_financials.iloc[-1]
                
                # Financial impact on stock price
                financial_impact = (
                    last_financial['roa'] * 2 + 
                    last_financial['cfo'] * 1.5 + 
                    last_financial['gross_margin'] * 1 -
                    last_financial['leverage'] * 0.5
                )
                
                daily_return += financial_impact * 0.001
            
            # Calculate new price
            new_price = prices[-1] * (1 + daily_return)
            prices.append(new_price)
        
        # Store price data
        all_prices[company_id] = pd.Series(prices, index=date_range)
    
    # Convert to DataFrame
    price_df = pd.DataFrame(all_prices)
    return price_df

# Calculate Piotroski F-Score
def calculate_fscore(financial_data, date):
    """
    Calculate Piotroski F-Score for a company based on latest financial data before a given date.
    
    Parameters:
    - financial_data: DataFrame containing the company's financial metrics
    - date: Date for which to calculate the F-Score
    
    Returns:
    - F-Score (0-9)
    """
    # Filter for data available up to the given date
    available_data = financial_data[financial_data['date'] <= date]
    
    if available_data.empty:
        return np.nan
    
    # Get most recent financial data
    latest_data = available_data.iloc[-1]
    
    # 1. Profitability
    f1 = 1 if latest_data['roa'] > 0 else 0  # Return on Assets
    f2 = 1 if latest_data['roa_growth'] > 0 else 0  # ROA Growth
    f3 = 1 if latest_data['cfo'] > 0 else 0  # Cash Flow from Operations
    f4 = 1 if latest_data['cfo_roa_diff'] > 0 else 0  # Cash Flow > ROA
    
    # 2. Leverage/Liquidity
    f5 = 1 if latest_data['leverage_change'] < 0 else 0  # Decreased Leverage
    f6 = 1 if latest_data['liquidity_change'] > 0 else 0  # Increased Liquidity
    f7 = 1 if latest_data['shares_outstanding_growth'] <= 0 else 0  # No Dilution
    
    # 3. Operating Efficiency
    f8 = 1 if latest_data['gross_margin_growth'] > 0 else 0  # Gross Margin Improvement
    f9 = 1 if latest_data['asset_turnover_growth'] > 0 else 0  # Asset Turnover Improvement
    
    # Total F-Score
    fscore = f1 + f2 + f3 + f4 + f5 + f6 + f7 + f8 + f9
    
    return fscore

# Function to rebalance portfolio based on F-Scores
def rebalance_portfolio(companies, prices, date):
    """
    Rebalance portfolio based on F-Scores.
    - Low: F-Score <= 3
    - Mid: 4 <= F-Score <= 6
    - High: F-Score >= 7
    
    Returns dictionaries containing stocks in each category.
    """
    f_scores = {}
    
    # Calculate F-Score for each company
    for company_id, financial_data in companies.items():
        fscore = calculate_fscore(financial_data, date)
        if not np.isnan(fscore):
            f_scores[company_id] = fscore
    
    # Categorize companies based on F-Score
    low_group = {company_id: score for company_id, score in f_scores.items() if score <= 3}
    mid_group = {company_id: score for company_id, score in f_scores.items() if 4 <= score <= 6}
    high_group = {company_id: score for company_id, score in f_scores.items() if score >= 7}
    
    # Get lists of companies in each group
    low_companies = list(low_group.keys())
    mid_companies = list(mid_group.keys())
    high_companies = list(high_group.keys())
    
    return {
        'low': low_companies,
        'mid': mid_companies,
        'high': high_companies
    }

# Function to calculate portfolio performance
def calculate_portfolio_performance(prices, portfolio, start_date, end_date):
    """
    Calculate performance of portfolio between start_date and end_date.
    """
    # Extract portfolio prices
    if not portfolio:
        return None
    
    portfolio_prices = prices.loc[start_date:end_date, portfolio]
    
    # Calculate equal-weighted portfolio returns
    portfolio_returns = portfolio_prices.pct_change().mean(axis=1).dropna()
    
    # Calculate cumulative returns
    portfolio_cumulative_returns = (1 + portfolio_returns).cumprod()
    
    return {
        'returns': portfolio_returns,
        'cumulative_returns': portfolio_cumulative_returns
    }

# Function to run backtest
def run_backtest(companies, prices, start_date, end_date, rebalance_dates):
    """
    Run backtest of Piotroski F-Score strategy.
    """
    # Initialize results storage
    portfolios = {
        'low': [],
        'mid': [],
        'high': []
    }
    
    performance = {
        'low': pd.Series(),
        'mid': pd.Series(),
        'high': pd.Series()
    }
    
    # Generate rebalance dates
    all_rebalance_dates = []
    current_date = start_date
    
    while current_date <= end_date:
        for date_str in rebalance_dates:
            rebalance_date = datetime.strptime(f"{current_date.year}-{date_str}", "%Y-%m-%d")
            if start_date <= rebalance_date <= end_date:
                all_rebalance_dates.append(rebalance_date)
        current_date = datetime(current_date.year + 1, 1, 1)
    
    all_rebalance_dates.sort()
    
    # Run backtest across rebalance periods
    for i in range(len(all_rebalance_dates)):
        current_date = all_rebalance_dates[i]
        
        # Rebalance portfolio
        new_portfolios = rebalance_portfolio(companies, prices, current_date)
        
        # Store new portfolios
        for group in ['low', 'mid', 'high']:
            portfolios[group].append({
                'date': current_date,
                'stocks': new_portfolios[group],
                'count': len(new_portfolios[group])
            })
        
        # Calculate performance until next rebalance or end date
        if i < len(all_rebalance_dates) - 1:
            next_date = all_rebalance_dates[i+1]
        else:
            next_date = end_date
        
        # Calculate performance for each group
        for group in ['low', 'mid', 'high']:
            if new_portfolios[group]:  # Check if the portfolio has stocks
                period_performance = calculate_portfolio_performance(
                    prices, new_portfolios[group], current_date, next_date
                )
                
                if period_performance and not period_performance['cumulative_returns'].empty:
                    performance[group] = pd.concat([
                        performance[group],
                        period_performance['cumulative_returns']
                    ])
    
    return {
        'portfolios': portfolios,
        'performance': performance
    }

# Generate simulated data
print("Generating company financial data...")
companies = generate_company_data(start_date, end_date, num_companies)
print("Generating stock price data...")
prices = generate_stock_prices(companies, start_date, end_date)

# Run backtest
print("Running backtest...")
backtest_results = run_backtest(companies, prices, start_date, end_date, rebalance_dates)

# Plot performance
plt.figure(figsize=(12, 6))
for group in ['low', 'mid', 'high']:
    if not backtest_results['performance'][group].empty:
        plt.plot(backtest_results['performance'][group], label=f'{group.capitalize()} F-Score Group')

plt.title('Piotroski F-Score Strategy Performance')
plt.xlabel('Date')
plt.ylabel('Cumulative Return')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Calculate annual returns for each group
def calculate_annual_returns(performance_data):
    """Calculate annual returns for each portfolio group."""
    annual_returns = {}
    
    for group, returns in performance_data.items():
        if returns.empty:
            annual_returns[group] = {}
            continue
            
        # Resample to get annual returns
        annual = returns.resample('Y').last().pct_change().dropna()
        annual_returns[group] = annual.to_dict()
    
    return annual_returns

# Calculate annual returns
annual_returns = calculate_annual_returns(backtest_results['performance'])

# Display annual returns
years = sorted(set(year.year for returns in annual_returns.values() for year in returns.keys()))
annual_data = []

for year in years:
    row = {'Year': year}
    for group in ['low', 'mid', 'high']:
        # Find return for this year
        for date, ret in annual_returns[group].items():
            if date.year == year:
                row[f'{group.capitalize()} Group'] = f"{ret*100:.2f}%"
                break
        else:
            row[f'{group.capitalize()} Group'] = "N/A"
    annual_data.append(row)

annual_df = pd.DataFrame(annual_data)
print("\nAnnual Returns by F-Score Group:")
print(annual_df)

# Calculate key statistics
def calculate_portfolio_statistics(performance_data):
    """Calculate key performance statistics for each portfolio group."""
    stats = {}
    
    for group, returns in performance_data.items():
        if returns.empty:
            continue
            
        # Convert series to returns
        daily_returns = returns.pct_change().dropna()
        
        # Calculate annualized statistics
        annualized_return = ((1 + daily_returns.mean()) ** 252) - 1
        annualized_vol = daily_returns.std() * np.sqrt(252)
        sharpe_ratio = annualized_return / annualized_vol if annualized_vol != 0 else 0
        
        # Maximum drawdown
        cumulative = (1 + daily_returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative / running_max) - 1
        max_drawdown = drawdown.min()
        
        stats[group] = {
            'Annualized Return': f"{annualized_return*100:.2f}%",
            'Annualized Volatility': f"{annualized_vol*100:.2f}%",
            'Sharpe Ratio': f"{sharpe_ratio:.2f}",
            'Maximum Drawdown': f"{max_drawdown*100:.2f}%",
            'Total Return': f"{(returns.iloc[-1] - 1)*100:.2f}%"
        }
    
    return stats

# Calculate and display statistics
portfolio_stats = calculate_portfolio_statistics(backtest_results['performance'])
stats_df = pd.DataFrame(portfolio_stats).T
print("\nPortfolio Performance Statistics:")
print(stats_df)

# Analyze F-Score distribution
fscore_distribution = []
for rebalance_date in all_rebalance_dates[:1]:  # Just using the first rebalance date for analysis
    f_scores = {}
    for company_id, financial_data in companies.items():
        fscore = calculate_fscore(financial_data, rebalance_date)
        if not np.isnan(fscore):
            f_scores[company_id] = fscore
    
    # Count occurrences of each F-Score
    counts = {}
    for score in range(10):  # F-Scores range from 0 to 9
        counts[score] = sum(1 for s in f_scores.values() if s == score)
    
    fscore_distribution.append({
        'Date': rebalance_date,
        **counts
    })

# Plot F-Score distribution
fscore_df = pd.DataFrame(fscore_distribution).set_index('Date')
fscore_distribution = fscore_df.iloc[0]  # Just the first distribution

plt.figure(figsize=(10, 6))
plt.bar(fscore_distribution.index, fscore_distribution.values)
plt.title('Distribution of F-Scores Across Companies')
plt.xlabel('F-Score')
plt.ylabel('Number of Companies')
plt.xticks(range(10))
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Calculate average portfolio turnover
def calculate_turnover(portfolios):
    """Calculate average portfolio turnover between rebalances."""
    turnover = {}
    
    for group in ['low', 'mid', 'high']:
        group_portfolios = portfolios[group]
        turnover_rates = []
        
        for i in range(1, len(group_portfolios)):
            prev_stocks = set(group_portfolios[i-1]['stocks'])
            curr_stocks = set(group_portfolios[i]['stocks'])
            
            # Calculate turnover as proportion of portfolio that changed
            if prev_stocks or curr_stocks:  # Avoid division by zero
                changes = len(prev_stocks.symmetric_difference(curr_stocks))
                total = max(len(prev_stocks), len(curr_stocks))
                rate = changes / (2 * total) if total > 0 else 0
                turnover_rates.append(rate)
        
        if turnover_rates:
            turnover[group] = {
                'Average Turnover': f"{np.mean(turnover_rates)*100:.2f}%",
                'Min Turnover': f"{np.min(turnover_rates)*100:.2f}%",
                'Max Turnover': f"{np.max(turnover_rates)*100:.2f}%"
            }
        else:
            turnover[group] = {
                'Average Turnover': "N/A",
                'Min Turnover': "N/A",
                'Max Turnover': "N/A"
            }
    
    return turnover

# Calculate and display turnover
portfolio_turnover = calculate_turnover(backtest_results['portfolios'])
turnover_df = pd.DataFrame(portfolio_turnover).T
print("\nPortfolio Turnover Statistics:")
print(turnover_df)

# Calculate correlations between F-Score and subsequent returns
def fscore_return_correlation(companies, prices, rebalance_dates):
    """Calculate correlation between F-Score and subsequent 3-month returns."""
    correlations = []
    
    for rebalance_date in rebalance_dates:
        # Skip last few rebalance dates where we don't have full 3-month returns
        end_date = rebalance_date + relativedelta(months=3)
        if end_date > prices.index[-1]:
            continue
            
        # Calculate F-Scores and subsequent returns
        data = []
        for company_id, financial_data in companies.items():
            fscore = calculate_fscore(financial_data, rebalance_date)
            if np.isnan(fscore):
                continue
                
            # Calculate 3-month return
            try:
                start_price = prices.loc[rebalance_date, company_id]
                end_price = prices.loc[end_date, company_id]
                return_3m = (end_price / start_price) - 1
                
                data.append({
                    'Company': company_id,
                    'F-Score': fscore,
                    'Return_3M': return_3m
                })
            except KeyError:
                continue
        
        if data:
            df = pd.DataFrame(data)
            correlation = df['F-Score'].corr(df['Return_3M'])
            correlations.append({
                'Date': rebalance_date,
                'Correlation': correlation
            })
    
    return pd.DataFrame(correlations)

# Calculate correlations
correlations = fscore_return_correlation(companies, prices, all_rebalance_dates)

# Plot correlations over time
plt.figure(figsize=(12, 6))
plt.plot(correlations['Date'], correlations['Correlation'])
plt.title('Correlation Between F-Score and Subsequent 3-Month Returns')
plt.xlabel('Rebalance Date')
plt.ylabel('Correlation Coefficient')
plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
plt.grid(True)
plt.tight_layout()
plt.show()

print(f"\nAverage Correlation between F-Score and 3-Month Returns: {correlations['Correlation'].mean():.4f}")

# Calculate High vs Low group excess returns
def calculate_excess_returns(performance_data):
    """Calculate high minus low excess returns."""
    if 'high' not in performance_data or 'low' not in performance_data:
        return None
    
    high_returns = performance_data['high']
    low_returns = performance_data['low']
    
    # Align dates between high and low returns
    common_dates = high_returns.index.intersection(low_returns.index)
    high_aligned = high_returns.loc[common_dates]
    low_aligned = low_returns.loc[common_dates]
    
    # Calculate daily returns
    high_daily_returns = high_aligned.pct_change().dropna()
    low_daily_returns = low_aligned.pct_change().dropna()
    
    # Calculate excess returns
    common_dates = high_daily_returns.index.intersection(low_daily_returns.index)
    excess_returns = high_daily_returns.loc[common_dates] - low_daily_returns.loc[common_dates]
    
    return excess_returns

# Calculate excess returns
excess_returns = calculate_excess_returns(backtest_results['performance'])

if excess_returns is not None:
    # Plot excess returns
    plt.figure(figsize=(12, 6))
    cumulative_excess = (1 + excess_returns).cumprod()
    plt.plot(cumulative_excess)
    plt.title('Cumulative High-Low Excess Returns')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Excess Returns')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    # Calculate information ratio
    info_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(252)
    annual_excess = ((1 + excess_returns.mean()) ** 252) - 1
    
    print(f"\nHigh-Low Strategy Information Ratio: {info_ratio:.2f}")
    print(f"Annualized High-Low Excess Return: {annual_excess*100:.2f}%")