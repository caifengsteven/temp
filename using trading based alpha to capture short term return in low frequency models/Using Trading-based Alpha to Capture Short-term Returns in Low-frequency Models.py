import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import scipy.optimize as sco
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

# Set plot style - use a basic style that works across versions
plt.style.use('default')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

# Set random seed for reproducibility
np.random.seed(42)

# 1. Generate simulated market data
def generate_market_data(n_stocks=300, n_days=500, n_industries=29):
    """
    Generate simulated market data including prices, volumes, and fundamental data
    """
    # Initialize data structures
    dates = pd.date_range(start='2015-01-01', periods=n_days, freq='B')
    
    # Generate market return (common factor)
    market_return = np.random.normal(0.0005, 0.01, n_days)
    market_return = np.cumsum(market_return)
    
    # Generate industry factors (29 industries as in the paper)
    industry_returns = np.random.normal(0.0003, 0.015, (n_days, n_industries))
    industry_returns = np.cumsum(industry_returns, axis=0)
    
    # Assign industries to stocks
    stock_industries = np.random.choice(n_industries, n_stocks)
    
    # Generate style factors (10 style factors as in the paper)
    style_factors = {
        'Beta': np.random.normal(1, 0.3, n_stocks),
        'Size': np.random.lognormal(0, 1, n_stocks),
        'Non_Linear_Size': np.random.lognormal(0, 1, n_stocks),
        'BTOP': np.random.normal(0.5, 0.2, n_stocks),
        'Momentum': np.random.normal(0, 1, n_stocks),
        'Residual_Volatility': np.random.lognormal(-1, 0.5, n_stocks),
        'Liquidity': np.random.lognormal(0, 1, n_stocks),
        'Leverage': np.random.normal(0.5, 0.2, n_stocks),
        'Growth': np.random.normal(0.1, 0.05, n_stocks),
        'Earnings_Yield': np.random.normal(0.05, 0.02, n_stocks)
    }
    
    # Create price matrices
    prices = np.zeros((n_days, n_stocks))
    volumes = np.zeros((n_days, n_stocks))
    
    # Initialize prices at 100
    prices[0, :] = 100
    
    # Generate stock-specific factors (idiosyncratic returns)
    stock_specific = np.random.normal(0, 0.02, (n_days, n_stocks))
    
    # Generate fundamental data that changes quarterly
    quarters = np.ceil(np.arange(1, n_days+1) / 60).astype(int)  # Roughly quarterly
    quarterly_updates = np.max(quarters)
    
    # Fundamental data
    fundamental_data = {
        'ROE': np.random.normal(0.15, 0.10, (quarterly_updates, n_stocks)),
        'ROA': np.random.normal(0.08, 0.05, (quarterly_updates, n_stocks)),
        'Revenue_Growth': np.random.normal(0.05, 0.15, (quarterly_updates, n_stocks)),
        'Profit_Growth': np.random.normal(0.05, 0.20, (quarterly_updates, n_stocks)),
        'BP_Ratio': np.random.lognormal(-0.5, 0.5, (quarterly_updates, n_stocks)),
        'PE_TTM': np.random.lognormal(2.5, 0.5, (quarterly_updates, n_stocks)),  # log of P/E
        'Dividend_Yield': np.random.lognormal(-3, 0.8, (quarterly_updates, n_stocks))  # log of dividend yield
    }
    
    # Map fundamental data to daily values
    daily_fundamentals = {}
    for key in fundamental_data:
        daily_values = np.zeros((n_days, n_stocks))
        for d in range(n_days):
            q = quarters[d] - 1  # 0-based index
            daily_values[d, :] = fundamental_data[key][q, :]
        daily_fundamentals[key] = daily_values
    
    # Generate daily prices based on factors
    for d in range(1, n_days):
        # Market component
        market_component = market_return[d] - market_return[d-1]
        
        # Industry component
        industry_component = np.zeros(n_stocks)
        for s in range(n_stocks):
            industry_component[s] = industry_returns[d, stock_industries[s]] - industry_returns[d-1, stock_industries[s]]
        
        # Style component (simplified)
        style_component = np.zeros(n_stocks)
        for factor, exposures in style_factors.items():
            factor_return = np.random.normal(0.0002, 0.005)
            style_component += exposures * factor_return
        
        # Calculate daily returns
        daily_returns = market_component + industry_component + style_component + stock_specific[d, :]
        
        # Update prices
        prices[d, :] = prices[d-1, :] * (1 + daily_returns)
        
        # Generate volumes (correlated with volatility and returns)
        volumes[d, :] = np.exp(np.random.normal(
            10 + 0.5 * np.abs(daily_returns) + 0.2 * np.abs(stock_specific[d, :]), 
            0.5
        ))
    
    # Create DataFrame with all the data
    all_data = []
    
    for s in range(n_stocks):
        stock_data = pd.DataFrame({
            'date': dates,
            'stock_id': s,
            'industry': stock_industries[s],
            'price': prices[:, s],
            'volume': volumes[:, s],
        })
        
        # Add style factors
        for factor, values in style_factors.items():
            # Add some time variation to style factors
            factor_ts = values[s] + np.random.normal(0, 0.1, n_days) 
            stock_data[factor] = factor_ts
        
        # Add fundamental data
        for key, values in daily_fundamentals.items():
            stock_data[key] = values[:, s]
        
        all_data.append(stock_data)
    
    # Combine all stocks into single DataFrame
    market_data = pd.concat(all_data, ignore_index=True)
    
    # Add open, high, low, close prices (for technical indicators)
    market_data['open'] = market_data['price'] * (1 + np.random.normal(0, 0.005, len(market_data)))
    market_data['close'] = market_data['price'] * (1 + np.random.normal(0, 0.005, len(market_data)))
    market_data['high'] = np.maximum(market_data['open'], market_data['close']) * (1 + np.random.uniform(0, 0.01, len(market_data)))
    market_data['low'] = np.minimum(market_data['open'], market_data['close']) * (1 - np.random.uniform(0, 0.01, len(market_data)))
    
    # Calculate returns (for testing)
    market_data = market_data.sort_values(['stock_id', 'date'])
    market_data['return'] = market_data.groupby('stock_id')['close'].pct_change()
    
    # Create benchmark index (equivalent to CSI 500)
    benchmark = market_data.groupby('date')[['close']].mean()
    benchmark['return'] = benchmark['close'].pct_change()
    
    return market_data, benchmark

# Function to implement a simplified strategy that's more robust
def simplified_strategy(market_data, rebalance_freq=21, intra_rebalance_freq=3, intra_rebalance_pct=0.1):
    """
    Simplified implementation of the strategy to demonstrate the concept without complex risk modeling
    """
    # Get unique dates
    dates = sorted(market_data['date'].unique())
    
    # Initialize portfolio tracking
    portfolio = pd.DataFrame(index=dates, columns=['portfolio_value', 'benchmark_value', 'excess_return'])
    portfolio['portfolio_value'] = 1.0
    portfolio['benchmark_value'] = 1.0
    portfolio['excess_return'] = 0.0
    
    # Get unique stocks
    unique_stocks = market_data['stock_id'].unique()
    n_stocks = len(unique_stocks)
    
    # Initialize with equal weights
    current_holdings = pd.DataFrame({
        'stock_id': unique_stocks,
        'weight': np.ones(n_stocks) / n_stocks
    })
    
    # Dictionary to track ICs for fundamental factors
    factor_ics = {}
    
    # Trading days counter
    day_counter = 0
    
    # Main simulation loop
    for i, date in enumerate(dates[252:]):  # Start after sufficient history
        previous_date_idx = i - 1 + 252
        if previous_date_idx < 0:
            continue
            
        previous_date = dates[previous_date_idx]
        
        # Monthly rebalance
        if day_counter % rebalance_freq == 0:
            print(f"Monthly rebalance at {date}")
            
            # Get current date's data
            current_data = market_data[market_data['date'] == date].copy()
            
            # Calculate fundamental factors - use a subset of fundamental data columns
            fundamental_cols = ['BP_Ratio', 'PE_TTM', 'ROE', 'ROA', 'Revenue_Growth']
            fundamental_data = current_data[['stock_id'] + fundamental_cols].copy()
            
            # Handle PE_TTM (convert to earnings yield)
            fundamental_data['EPTTM'] = 1 / fundamental_data['PE_TTM'].replace(0, np.nan)
            fundamental_data['EPTTM'] = fundamental_data['EPTTM'].fillna(fundamental_data['EPTTM'].median())
            
            # Normalize factors
            for col in fundamental_data.columns:
                if col != 'stock_id':
                    mean = fundamental_data[col].mean()
                    std = fundamental_data[col].std()
                    if std > 0:
                        fundamental_data[col + '_norm'] = (fundamental_data[col] - mean) / std
                    else:
                        fundamental_data[col + '_norm'] = 0
            
            # Combine factors to create alpha score (simple average)
            norm_cols = [c for c in fundamental_data.columns if c.endswith('_norm')]
            fundamental_data['alpha_score'] = fundamental_data[norm_cols].mean(axis=1)
            
            # Rank stocks by alpha score
            fundamental_data = fundamental_data.sort_values('alpha_score', ascending=False)
            
            # Select top 100 stocks for portfolio
            top_stocks = fundamental_data.head(100)
            
            # Update holdings with equal weights
            new_holdings = pd.DataFrame({
                'stock_id': top_stocks['stock_id'].values,
                'weight': np.ones(len(top_stocks)) / len(top_stocks)
            })
            
            current_holdings = new_holdings
        
        # Intra-month rebalance using trading alphas
        elif (day_counter % rebalance_freq) % intra_rebalance_freq == 0 and day_counter % rebalance_freq != 0:
            print(f"Intra-month rebalance at {date}")
            
            # Calculate trading alphas
            # Get data for calculation
            current_idx = dates.index(date)
            start_idx = max(0, current_idx - 20)  # 20-day lookback
            lookback_dates = dates[start_idx:current_idx+1]
            
            # Get historical data
            hist_data = market_data[market_data['date'].isin(lookback_dates)].copy()
            
            # Group by stock and calculate trading alphas
            trading_alphas = []
            
            for stock_id in current_holdings['stock_id'].unique():
                stock_data = hist_data[hist_data['stock_id'] == stock_id].sort_values('date')
                
                if len(stock_data) < 10:  # Need at least 10 days of data
                    continue
                    
                # Calculate a simple momentum alpha (past 5-day return)
                last_5_days = stock_data.iloc[-5:]
                momentum = last_5_days['close'].iloc[-1] / last_5_days['close'].iloc[0] - 1
                
                # Calculate a simple mean reversion alpha (deviation from 10-day moving average)
                ma_10 = stock_data['close'].rolling(10).mean().iloc[-1]
                mean_reversion = (stock_data['close'].iloc[-1] - ma_10) / ma_10
                
                # Combine alphas
                alpha_score = momentum - mean_reversion  # momentum minus mean reversion
                
                trading_alphas.append({
                    'stock_id': stock_id,
                    'alpha_score': alpha_score
                })
            
            if not trading_alphas:
                continue
                
            trading_alphas_df = pd.DataFrame(trading_alphas)
            
            # Merge with current holdings
            holdings_with_alphas = current_holdings.merge(trading_alphas_df, on='stock_id', how='left')
            holdings_with_alphas['alpha_score'] = holdings_with_alphas['alpha_score'].fillna(0)
            
            # Sort by alpha score
            holdings_with_alphas = holdings_with_alphas.sort_values('alpha_score')
            
            # Replace bottom 10% stocks with highest alpha stocks from universe
            n_replace = max(1, int(len(holdings_with_alphas) * intra_rebalance_pct))
            stocks_to_replace = holdings_with_alphas['stock_id'].iloc[:n_replace].values
            weight_to_redistribute = holdings_with_alphas['weight'].iloc[:n_replace].sum()
            
            # Get universe stocks not in current portfolio
            universe_stocks = market_data[market_data['date'] == date]['stock_id'].unique()
            available_stocks = [s for s in universe_stocks if s not in current_holdings['stock_id'].values]
            
            # If no stocks available to add, skip rebalance
            if not available_stocks:
                continue
                
            # Calculate trading alphas for available stocks
            available_alphas = []
            for stock_id in available_stocks:
                stock_data = hist_data[hist_data['stock_id'] == stock_id].sort_values('date')
                
                if len(stock_data) < 10:  # Need at least 10 days of data
                    continue
                    
                # Calculate a simple momentum alpha
                last_5_days = stock_data.iloc[-5:]
                momentum = last_5_days['close'].iloc[-1] / last_5_days['close'].iloc[0] - 1
                
                # Calculate a simple mean reversion alpha
                ma_10 = stock_data['close'].rolling(10).mean().iloc[-1]
                mean_reversion = (stock_data['close'].iloc[-1] - ma_10) / ma_10
                
                # Combine alphas
                alpha_score = momentum - mean_reversion
                
                available_alphas.append({
                    'stock_id': stock_id,
                    'alpha_score': alpha_score
                })
            
            if not available_alphas:
                continue
                
            available_alphas_df = pd.DataFrame(available_alphas)
            available_alphas_df = available_alphas_df.sort_values('alpha_score', ascending=False)
            
            # Select top stocks to add
            stocks_to_add = available_alphas_df['stock_id'].iloc[:n_replace].values
            
            # Remove stocks being replaced
            current_holdings = current_holdings[~current_holdings['stock_id'].isin(stocks_to_replace)]
            
            # Add new stocks
            new_holdings = pd.DataFrame({
                'stock_id': stocks_to_add,
                'weight': np.ones(len(stocks_to_add)) * (weight_to_redistribute / len(stocks_to_add))
            })
            
            current_holdings = pd.concat([current_holdings, new_holdings], ignore_index=True)
        
        # Update portfolio value
        # Get returns for the day
        day_returns = market_data[(market_data['date'] == date) & 
                                 (market_data['stock_id'].isin(current_holdings['stock_id']))].copy()
        
        prev_day_returns = market_data[(market_data['date'] == previous_date) & 
                                      (market_data['stock_id'].isin(current_holdings['stock_id']))].copy()
        
        # If no overlap in stocks, use benchmark return
        if len(day_returns) == 0 or len(prev_day_returns) == 0:
            portfolio_return = 0
        else:
            # Calculate returns
            returns_data = day_returns.merge(
                prev_day_returns[['stock_id', 'close']], on='stock_id', suffixes=('', '_prev')
            )
            returns_data['daily_return'] = returns_data['close'] / returns_data['close_prev'] - 1
            
            # Merge with holdings
            holdings_returns = current_holdings.merge(returns_data[['stock_id', 'daily_return']], on='stock_id', how='left')
            holdings_returns['daily_return'] = holdings_returns['daily_return'].fillna(0)
            
            # Calculate portfolio return
            portfolio_return = (holdings_returns['weight'] * holdings_returns['daily_return']).sum()
        
        # Calculate benchmark return (average return across all stocks for the day)
        benchmark_return = market_data[market_data['date'] == date]['return'].mean()
        if pd.isna(benchmark_return):
            benchmark_return = 0
        
        # Update portfolio value
        portfolio.loc[date, 'portfolio_value'] = portfolio.loc[previous_date, 'portfolio_value'] * (1 + portfolio_return)
        portfolio.loc[date, 'benchmark_value'] = portfolio.loc[previous_date, 'benchmark_value'] * (1 + benchmark_return)
        portfolio.loc[date, 'excess_return'] = portfolio_return - benchmark_return
        
        # Increment day counter
        day_counter += 1
    
    return portfolio

# Function to create a monthly-only strategy (for comparison)
def monthly_only_strategy(market_data, rebalance_freq=21):
    """
    Implementation of a monthly-only strategy for comparison
    """
    # Get unique dates
    dates = sorted(market_data['date'].unique())
    
    # Initialize portfolio tracking
    portfolio = pd.DataFrame(index=dates, columns=['portfolio_value', 'benchmark_value', 'excess_return'])
    portfolio['portfolio_value'] = 1.0
    portfolio['benchmark_value'] = 1.0
    portfolio['excess_return'] = 0.0
    
    # Get unique stocks
    unique_stocks = market_data['stock_id'].unique()
    n_stocks = len(unique_stocks)
    
    # Initialize with equal weights
    current_holdings = pd.DataFrame({
        'stock_id': unique_stocks,
        'weight': np.ones(n_stocks) / n_stocks
    })
    
    # Trading days counter
    day_counter = 0
    
    # Main simulation loop
    for i, date in enumerate(dates[252:]):  # Start after sufficient history
        previous_date_idx = i - 1 + 252
        if previous_date_idx < 0:
            continue
            
        previous_date = dates[previous_date_idx]
        
        # Monthly rebalance
        if day_counter % rebalance_freq == 0:
            print(f"Monthly rebalance (monthly-only) at {date}")
            
            # Get current date's data
            current_data = market_data[market_data['date'] == date].copy()
            
            # Calculate fundamental factors - use a subset of fundamental data columns
            fundamental_cols = ['BP_Ratio', 'PE_TTM', 'ROE', 'ROA', 'Revenue_Growth']
            fundamental_data = current_data[['stock_id'] + fundamental_cols].copy()
            
            # Handle PE_TTM (convert to earnings yield)
            fundamental_data['EPTTM'] = 1 / fundamental_data['PE_TTM'].replace(0, np.nan)
            fundamental_data['EPTTM'] = fundamental_data['EPTTM'].fillna(fundamental_data['EPTTM'].median())
            
            # Normalize factors
            for col in fundamental_data.columns:
                if col != 'stock_id':
                    mean = fundamental_data[col].mean()
                    std = fundamental_data[col].std()
                    if std > 0:
                        fundamental_data[col + '_norm'] = (fundamental_data[col] - mean) / std
                    else:
                        fundamental_data[col + '_norm'] = 0
            
            # Combine factors to create alpha score (simple average)
            norm_cols = [c for c in fundamental_data.columns if c.endswith('_norm')]
            fundamental_data['alpha_score'] = fundamental_data[norm_cols].mean(axis=1)
            
            # Rank stocks by alpha score
            fundamental_data = fundamental_data.sort_values('alpha_score', ascending=False)
            
            # Select top 100 stocks for portfolio
            top_stocks = fundamental_data.head(100)
            
            # Update holdings with equal weights
            new_holdings = pd.DataFrame({
                'stock_id': top_stocks['stock_id'].values,
                'weight': np.ones(len(top_stocks)) / len(top_stocks)
            })
            
            current_holdings = new_holdings
        
        # Update portfolio value
        # Get returns for the day
        day_returns = market_data[(market_data['date'] == date) & 
                                 (market_data['stock_id'].isin(current_holdings['stock_id']))].copy()
        
        prev_day_returns = market_data[(market_data['date'] == previous_date) & 
                                      (market_data['stock_id'].isin(current_holdings['stock_id']))].copy()
        
        # If no overlap in stocks, use benchmark return
        if len(day_returns) == 0 or len(prev_day_returns) == 0:
            portfolio_return = 0
        else:
            # Calculate returns
            returns_data = day_returns.merge(
                prev_day_returns[['stock_id', 'close']], on='stock_id', suffixes=('', '_prev')
            )
            returns_data['daily_return'] = returns_data['close'] / returns_data['close_prev'] - 1
            
            # Merge with holdings
            holdings_returns = current_holdings.merge(returns_data[['stock_id', 'daily_return']], on='stock_id', how='left')
            holdings_returns['daily_return'] = holdings_returns['daily_return'].fillna(0)
            
            # Calculate portfolio return
            portfolio_return = (holdings_returns['weight'] * holdings_returns['daily_return']).sum()
        
        # Calculate benchmark return (average return across all stocks for the day)
        benchmark_return = market_data[market_data['date'] == date]['return'].mean()
        if pd.isna(benchmark_return):
            benchmark_return = 0
        
        # Update portfolio value
        portfolio.loc[date, 'portfolio_value'] = portfolio.loc[previous_date, 'portfolio_value'] * (1 + portfolio_return)
        portfolio.loc[date, 'benchmark_value'] = portfolio.loc[previous_date, 'benchmark_value'] * (1 + benchmark_return)
        portfolio.loc[date, 'excess_return'] = portfolio_return - benchmark_return
        
        # Increment day counter
        day_counter += 1
    
    return portfolio

# 7. Evaluate strategy performance
def evaluate_performance(portfolio):
    """
    Evaluate portfolio performance
    """
    # Calculate daily returns
    portfolio['portfolio_return'] = portfolio['portfolio_value'].pct_change()
    portfolio['benchmark_return'] = portfolio['benchmark_value'].pct_change()
    
    # Remove NaN values
    portfolio = portfolio.dropna()
    
    # Calculate annualized return and volatility
    ann_factor = 252  # Trading days in a year
    portfolio_ann_return = portfolio['portfolio_return'].mean() * ann_factor
    benchmark_ann_return = portfolio['benchmark_return'].mean() * ann_factor
    portfolio_ann_vol = portfolio['portfolio_return'].std() * np.sqrt(ann_factor)
    benchmark_ann_vol = portfolio['benchmark_return'].std() * np.sqrt(ann_factor)
    
    # Calculate tracking error
    tracking_error = portfolio['excess_return'].std() * np.sqrt(ann_factor)
    
    # Calculate information ratio
    information_ratio = (portfolio_ann_return - benchmark_ann_return) / tracking_error if tracking_error > 0 else 0
    
    # Calculate Sharpe ratio (assuming risk-free rate of 0)
    portfolio_sharpe = portfolio_ann_return / portfolio_ann_vol if portfolio_ann_vol > 0 else 0
    benchmark_sharpe = benchmark_ann_return / benchmark_ann_vol if benchmark_ann_vol > 0 else 0
    
    # Calculate max drawdown
    portfolio_cum_returns = (1 + portfolio['portfolio_return']).cumprod()
    benchmark_cum_returns = (1 + portfolio['benchmark_return']).cumprod()
    
    portfolio_drawdown = 1 - portfolio_cum_returns / portfolio_cum_returns.cummax()
    benchmark_drawdown = 1 - benchmark_cum_returns / benchmark_cum_returns.cummax()
    
    portfolio_max_drawdown = portfolio_drawdown.max()
    benchmark_max_drawdown = benchmark_drawdown.max()
    
    # Create performance summary
    performance = {
        'Annualized Return': [portfolio_ann_return, benchmark_ann_return],
        'Annualized Volatility': [portfolio_ann_vol, benchmark_ann_vol],
        'Sharpe Ratio': [portfolio_sharpe, benchmark_sharpe],
        'Max Drawdown': [portfolio_max_drawdown, benchmark_max_drawdown],
        'Tracking Error': [tracking_error, None],
        'Information Ratio': [information_ratio, None]
    }
    
    performance_df = pd.DataFrame(performance, index=['Portfolio', 'Benchmark'])
    
    return performance_df

# 8. Run the simulation and visualize results
if __name__ == "__main__":
    # Generate simulated market data
    print("Generating simulated market data...")
    market_data, benchmark = generate_market_data(n_stocks=300, n_days=500)  # Reduced dimensions for faster execution
    
    print("Market data shape:", market_data.shape)
    print("Unique stocks:", market_data['stock_id'].nunique())
    print("Date range:", market_data['date'].min(), "to", market_data['date'].max())
    
    # Run combined strategy (monthly plus intra-month trading alphas)
    print("\nRunning combined monthly + trading alpha strategy...")
    combined_portfolio = simplified_strategy(
        market_data, 
        rebalance_freq=21,
        intra_rebalance_freq=3,
        intra_rebalance_pct=0.1
    )
    
    # Run monthly-only strategy for comparison
    print("\nRunning monthly-only strategy for comparison...")
    monthly_portfolio = monthly_only_strategy(
        market_data, 
        rebalance_freq=21
    )
    
    # Evaluate performance
    combined_performance = evaluate_performance(combined_portfolio)
    monthly_performance = evaluate_performance(monthly_portfolio)
    
    print("\nCombined Strategy Performance:")
    print(combined_performance)
    
    print("\nMonthly-Only Strategy Performance:")
    print(monthly_performance)
    
    # Compare strategies
    comparison = pd.concat([
        combined_performance.loc['Portfolio'].rename('Combined Strategy'),
        monthly_performance.loc['Portfolio'].rename('Monthly-Only Strategy'),
        combined_performance.loc['Benchmark'].rename('Benchmark')
    ], axis=1).T
    
    print("\nStrategy Comparison:")
    print(comparison)
    
    # Plot portfolio values
    plt.figure(figsize=(14, 10))
    
    # Plot portfolio and benchmark values
    plt.subplot(2, 1, 1)
    plt.plot(combined_portfolio.index, combined_portfolio['portfolio_value'], label='Combined Strategy')
    plt.plot(monthly_portfolio.index, monthly_portfolio['portfolio_value'], label='Monthly-Only Strategy')
    plt.plot(combined_portfolio.index, combined_portfolio['benchmark_value'], label='Benchmark')
    plt.title('Portfolio Performance Comparison')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    plt.legend()
    plt.grid(True)
    
    # Plot excess returns
    plt.subplot(2, 1, 2)
    plt.plot(combined_portfolio.index, combined_portfolio['excess_return'].cumsum(), label='Combined Strategy Excess Return')
    plt.plot(monthly_portfolio.index, monthly_portfolio['excess_return'].cumsum(), label='Monthly-Only Excess Return')
    plt.title('Cumulative Excess Return')
    plt.xlabel('Date')
    plt.ylabel('Excess Return')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    
    # Save figures instead of showing them
    plt.savefig('strategy_comparison.png')
    print("Plots saved to 'strategy_comparison.png'")