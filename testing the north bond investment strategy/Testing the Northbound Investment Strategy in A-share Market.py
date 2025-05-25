import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from datetime import datetime, timedelta
from tqdm import tqdm
from scipy import stats

# Configure plotting
try:
    plt.style.use('seaborn')  # Use standard seaborn style for older versions
except:
    pass  # Just use default style if seaborn styles aren't available

plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.sans-serif'] = ['SimHei']  # For Chinese characters
plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore')

class NorthboundFlowAnalyzer:
    """
    A class to analyze northbound capital flows in China's A-share market
    and test the strategy described in the research paper
    """
    
    def __init__(self, start_date='2017-01-01', end_date='2021-04-30'):
        """
        Initialize the analyzer with date range
        
        Parameters:
        -----------
        start_date : str
            Start date for backtesting
        end_date : str
            End date for backtesting
        """
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        
        # Data containers
        self.stock_data = None       # Stock price and return data
        self.northbound_data = None  # Northbound flow data
        self.index_data = None       # Index data (CSI 300)
        self.fundamental_data = None # Fundamental data (ROE, etc.)
        self.stock_pool = None       # Universe of stocks
        self.factors = None          # Calculated factors
        
        # Strategy parameters
        self.rebalance_dates = None  # Monthly rebalance dates
        self.top_n = 30              # Number of stocks to select
        self.selected_stocks = {}    # Selected stocks at each rebalance date
        self.portfolio_weights = {}  # Portfolio weights at each rebalance date
        self.portfolio_returns = {}  # Portfolio returns
        
    def load_data(self):
        """
        Load or simulate the required data:
        1. Stock prices and returns
        2. Northbound holdings
        3. CSI 300 index constituents
        4. Analyst ROE forecasts
        """
        print("Generating simulated data...")
        
        # Generate date range at daily frequency
        # Make sure to start earlier to have proper lookback data
        lookback_start = self.start_date - pd.DateOffset(years=1)
        all_dates = pd.date_range(lookback_start, self.end_date, freq='B')
        
        # Generate monthly rebalance dates (month-end business days)
        all_month_ends = pd.date_range(lookback_start, self.end_date, freq='BM')
        self.rebalance_dates = pd.date_range(self.start_date, self.end_date, freq='BM')
        
        # Create a universe of 300 simulated stocks (CSI 300 constituents)
        stock_ids = [f'SH{600000+i}' if i < 150 else f'SZ{300000+i-150}' for i in range(300)]
        
        # Simulate stock prices
        print("Simulating stock prices...")
        self.stock_data = {}
        
        # Set random seed for reproducibility
        np.random.seed(42)
        
        # Market return with some randomness
        market_return = np.random.normal(0.008, 0.06, len(all_dates))
        market_return[0] = 0  # First day has no return
        
        # Cumulative market return
        market_cumret = np.cumprod(1 + market_return)
        
        # Randomly assign sector betas
        sectors = ['技术', '消费', '医药', '金融', '工业', '材料', '能源', '公用']
        sector_betas = {sector: np.random.uniform(0.8, 1.2) for sector in sectors}
        stock_sectors = {stock: np.random.choice(sectors) for stock in stock_ids}
        
        # Randomly assign stock betas (relative to sector)
        stock_betas = {stock: np.random.uniform(0.7, 1.3) for stock in stock_ids}
        
        # Simulate stock prices
        for stock in tqdm(stock_ids):
            # Base price is random between 10 and 50
            base_price = np.random.uniform(10, 50)
            
            # Get sector beta for this stock
            sector = stock_sectors[stock]
            sector_beta = sector_betas[sector]
            stock_beta = stock_betas[stock]
            
            # Generate stock-specific noise
            stock_noise = np.random.normal(0, 0.02, len(all_dates))
            
            # Generate stock return as market return * beta + noise
            stock_return = market_return * sector_beta * stock_beta + stock_noise
            stock_return[0] = 0  # First day has no return
            
            # Convert to cumulative return and then price
            stock_cumret = np.cumprod(1 + stock_return)
            price = base_price * stock_cumret
            
            # Store in stock_data
            self.stock_data[stock] = pd.DataFrame({
                'price': price,
                'return': stock_return
            }, index=all_dates)
        
        # Simulate CSI 300 index
        self.index_data = pd.DataFrame({
            'price': 1000 * market_cumret,
            'return': market_return
        }, index=all_dates)
        
        # Simulate northbound holdings
        print("Simulating northbound holdings...")
        self.northbound_data = {}
        
        # Determine stock market caps (proportional to price)
        latest_prices = {stock: self.stock_data[stock]['price'].iloc[-1] for stock in stock_ids}
        market_caps = {stock: latest_prices[stock] * np.random.uniform(1e8, 1e10) for stock in stock_ids}
        
        # Sort stocks by market cap (descending)
        sorted_by_mcap = sorted(market_caps.items(), key=lambda x: x[1], reverse=True)
        sorted_stocks = [item[0] for item in sorted_by_mcap]
        
        # Assign higher northbound preference to larger stocks and certain sectors
        preferred_sectors = ['技术', '消费', '医药']
        
        northbound_preference = {}
        for stock in stock_ids:
            # Base preference by size (rank)
            size_rank = sorted_stocks.index(stock) / len(sorted_stocks)
            size_factor = 1 - size_rank  # Higher for larger stocks
            
            # Sector preference
            sector = stock_sectors[stock]
            sector_factor = 1.5 if sector in preferred_sectors else 0.7
            
            # Combine factors
            northbound_preference[stock] = size_factor * sector_factor
        
        # Normalize preferences
        total_pref = sum(northbound_preference.values())
        northbound_preference = {k: v/total_pref for k, v in northbound_preference.items()}
        
        # Generate northbound holdings over time (gradually increasing)
        start_percentage = 0.02  # 2% at start
        end_percentage = 0.05    # 5% at end
        
        for date in all_dates:
            progress = (date - all_dates[0]).days / (all_dates[-1] - all_dates[0]).days
            overall_pct = start_percentage + progress * (end_percentage - start_percentage)
            
            date_holdings = {}
            for stock in stock_ids:
                # Base holding percentage for this stock
                hold_pct = overall_pct * northbound_preference[stock] * (1 + np.random.normal(0, 0.1))
                
                # Ensure between 0 and 10%
                hold_pct = max(0, min(0.1, hold_pct))
                
                # Calculate northbound holding value
                price = self.stock_data[stock].loc[date, 'price']
                market_cap = market_caps[stock]
                holding_value = market_cap * hold_pct
                
                date_holdings[stock] = {
                    'holding_value': holding_value,
                    'market_cap': market_cap,
                    'holding_pct': hold_pct
                }
            
            # Store in northbound_data dictionary
            self.northbound_data[date] = date_holdings
        
        # Simulate analyst forecasts for ROE for ALL months including lookback period
        print("Simulating analyst ROE forecasts...")
        self.fundamental_data = {}
        
        # Base ROE for each stock
        base_roe = {stock: np.random.uniform(0.05, 0.2) for stock in stock_ids}
        
        # Sector ROE adjustments
        sector_roe_adj = {
            '技术': 1.2,
            '消费': 1.3,
            '医药': 1.1,
            '金融': 0.9,
            '工业': 0.8,
            '材料': 0.7,
            '能源': 0.6,
            '公用': 0.5
        }
        
        # Generate monthly ROE forecasts - for ALL months including lookback period
        for date in all_month_ends:
            date_forecasts = {}
            
            for stock in stock_ids:
                sector = stock_sectors[stock]
                # Base ROE adjusted by sector
                adj_roe = base_roe[stock] * sector_roe_adj[sector]
                
                # Add some time trend and noise
                trend = 0.001 * np.random.normal(1, 0.5)  # Small trend
                noise = np.random.normal(0, 0.01)         # Random noise
                
                # Calculate forecasted ROE
                forecasted_roe = adj_roe + trend + noise
                
                # Store forecast
                date_forecasts[stock] = forecasted_roe
            
            self.fundamental_data[date] = date_forecasts
        
        # Define stock pool (CSI 300 constituents)
        self.stock_pool = stock_ids
        
        print("Data generation complete!")
    
    def calculate_factors(self):
        """
        Calculate the factors described in the research paper:
        1. Northbound static ownership factors (2)
        2. Northbound ownership change factors (3)
        3. Expected ROE improvement factor
        """
        print("Calculating factors...")
        
        # Initialize factor dictionary
        self.factors = {}
        
        # Only calculate factors from the actual start date of our backtest
        valid_rebalance_dates = [date for date in self.rebalance_dates 
                               if date >= self.start_date + pd.DateOffset(months=6)]
        
        # For each rebalance date
        for date in tqdm(valid_rebalance_dates):
            # Skip if date not in our data
            if date not in self.northbound_data:
                continue
                
            # Initialize factors for this date
            self.factors[date] = {}
                
            # Get northbound holdings data at this date
            holdings = self.northbound_data[date]
            
            # Calculate total northbound holdings on this date
            total_northbound_value = sum(stock['holding_value'] for stock in holdings.values())
            
            # Get historical dates for calculating changes
            one_month_ago = date - pd.DateOffset(months=1)
            three_months_ago = date - pd.DateOffset(months=3)
            six_months_ago = date - pd.DateOffset(months=6)
            
            # Find closest actual dates in our data
            one_month_ago = min(self.northbound_data.keys(), key=lambda x: abs(x - one_month_ago))
            three_months_ago = min(self.northbound_data.keys(), key=lambda x: abs(x - three_months_ago))
            six_months_ago = min(self.northbound_data.keys(), key=lambda x: abs(x - six_months_ago))
            
            # Get historical holdings
            one_month_holdings = self.northbound_data.get(one_month_ago, {})
            three_month_holdings = self.northbound_data.get(three_months_ago, {})
            six_month_holdings = self.northbound_data.get(six_months_ago, {})
            
            # Find closest fundamental data dates
            fund_dates = list(self.fundamental_data.keys())
            fund_date = min(fund_dates, key=lambda x: abs(x - date))
            fund_three_months_ago = min(fund_dates, key=lambda x: abs(x - three_months_ago))
            
            # Get fundamental data
            current_fund_data = self.fundamental_data.get(fund_date, {})
            three_month_fund_data = self.fundamental_data.get(fund_three_months_ago, {})
            
            # For each stock in our pool
            for stock in self.stock_pool:
                # Skip if not in holdings (shouldn't happen in our simulation)
                if stock not in holdings:
                    continue
                
                # Get current holdings data
                stock_holdings = holdings[stock]
                stock_holding_value = stock_holdings['holding_value']
                stock_market_cap = stock_holdings['market_cap']
                stock_holding_pct = stock_holdings['holding_pct']
                
                # 1. Northbound holding percentage (factor 1)
                f1 = stock_holding_pct
                
                # 2. Northbound internal holding percentage (factor 2)
                f2 = stock_holding_value / total_northbound_value if total_northbound_value > 0 else 0
                
                # Calculate 3-month average holding percentage
                three_month_dates = [date, one_month_ago, three_months_ago]
                three_month_pcts = []
                
                for d in three_month_dates:
                    if d in self.northbound_data and stock in self.northbound_data[d]:
                        if 'holding_pct' in self.northbound_data[d][stock]:
                            three_month_pcts.append(self.northbound_data[d][stock]['holding_pct'])
                
                three_month_avg = np.mean(three_month_pcts) if three_month_pcts else 0
                
                # 3. Northbound holding percentage deviation from 3-month average (factor 3)
                f3 = stock_holding_pct - three_month_avg
                
                # Get historical internal holding percentages
                one_month_internal = 0
                three_month_internal = 0
                six_month_internal = 0
                
                # Safe access to historical data
                if one_month_ago in self.northbound_data and stock in self.northbound_data[one_month_ago]:
                    one_month_total = sum(s['holding_value'] for s in self.northbound_data[one_month_ago].values())
                    one_month_internal = self.northbound_data[one_month_ago][stock]['holding_value'] / one_month_total if one_month_total > 0 else 0
                
                if three_months_ago in self.northbound_data and stock in self.northbound_data[three_months_ago]:
                    three_month_total = sum(s['holding_value'] for s in self.northbound_data[three_months_ago].values())
                    three_month_internal = self.northbound_data[three_months_ago][stock]['holding_value'] / three_month_total if three_month_total > 0 else 0
                
                if six_months_ago in self.northbound_data and stock in self.northbound_data[six_months_ago]:
                    six_month_total = sum(s['holding_value'] for s in self.northbound_data[six_months_ago].values())
                    six_month_internal = self.northbound_data[six_months_ago][stock]['holding_value'] / six_month_total if six_month_total > 0 else 0
                
                # 4. Northbound internal holding percentage 3-month change (factor 4)
                f4 = f2 - three_month_internal
                
                # 5. Northbound holding percentage 6-month change (factor 5)
                six_month_pct = 0
                if six_months_ago in self.northbound_data and stock in self.northbound_data[six_months_ago]:
                    six_month_pct = self.northbound_data[six_months_ago][stock].get('holding_pct', 0)
                f5 = stock_holding_pct - six_month_pct
                
                # 6. Expected ROE improvement factor
                f6 = 0  # Default value
                
                # Safe access to fundamental data
                if stock in current_fund_data and stock in three_month_fund_data:
                    current_roe = current_fund_data[stock]
                    three_month_roe = three_month_fund_data[stock]
                    f6 = current_roe - three_month_roe
                
                # Store factors
                self.factors[date][stock] = {
                    'northbound_pct': f1,
                    'northbound_internal_pct': f2,
                    'northbound_deviation': f3,
                    'northbound_internal_change': f4,
                    'northbound_6m_change': f5,
                    'roe_improvement': f6
                }
        
        print("Factor calculation complete!")
    
    def calculate_factor_ic(self, factor_name, dates=None):
        """
        Calculate Information Coefficient (IC) for a factor
        
        Parameters:
        -----------
        factor_name : str
            Name of the factor
        dates : list, optional
            List of dates to calculate IC for
            
        Returns:
        --------
        pd.Series
            Series of IC values indexed by date
        """
        if dates is None:
            # Use dates that have factors calculated
            dates = list(self.factors.keys())
            dates.sort()
            
            # Skip the last date as we need forward returns
            if dates:
                dates = dates[:-1]
        
        ic_values = {}
        
        for i, date in enumerate(dates):
            # Skip if date not in factors
            if date not in self.factors:
                continue
                
            # Find the next date for which we have factors
            next_dates = [d for d in self.rebalance_dates if d > date and d in self.factors]
            if not next_dates:
                continue
                
            next_date = next_dates[0]
            
            # Get factor values at current date
            factor_values = {}
            for stock in self.stock_pool:
                if stock in self.factors[date] and factor_name in self.factors[date][stock]:
                    factor_values[stock] = self.factors[date][stock][factor_name]
            
            # Get forward returns
            forward_returns = {}
            for stock in factor_values.keys():
                # Calculate return from current date to next rebalance date
                if stock in self.stock_data:
                    try:
                        current_price = self.stock_data[stock].loc[date, 'price']
                        next_price = self.stock_data[stock].loc[next_date, 'price']
                        forward_returns[stock] = (next_price / current_price) - 1
                    except:
                        # Skip if price data is not available
                        continue
            
            # Calculate rank correlation (IC)
            if len(factor_values) > 10 and len(forward_returns) > 10:
                common_stocks = set(factor_values.keys()) & set(forward_returns.keys())
                
                if len(common_stocks) > 10:
                    x = [factor_values[stock] for stock in common_stocks]
                    y = [forward_returns[stock] for stock in common_stocks]
                    
                    ic, _ = stats.spearmanr(x, y)
                    if not np.isnan(ic):
                        ic_values[date] = ic
        
        return pd.Series(ic_values)
    
    def calculate_combined_factors(self, date):
        """
        Calculate combined factors for a given date
        
        Parameters:
        -----------
        date : datetime
            Date to calculate combined factors for
            
        Returns:
        --------
        dict
            Dictionary of combined factor scores for each stock
        """
        # Skip if the date doesn't have factors
        if date not in self.factors:
            return {'static': {}, 'change': {}, 'combined': {}}
            
        # Initialize combined factor scores
        static_scores = {}
        change_scores = {}
        combined_scores = {}
        
        # For each stock in our pool
        for stock in self.stock_pool:
            if stock not in self.factors[date]:
                continue
            
            # Get factor values
            factors = self.factors[date][stock]
            
            # Calculate static factors composite (equal weight of factors 1 and 2)
            static_score = (factors['northbound_pct'] + factors['northbound_internal_pct']) / 2
            static_scores[stock] = static_score
            
            # Calculate change factors composite (equal weight of factors 3, 4, and 5)
            change_score = (factors['northbound_deviation'] + 
                           factors['northbound_internal_change'] + 
                           factors['northbound_6m_change']) / 3
            change_scores[stock] = change_score
            
            # Calculate final combined score (equal weight of static, change, and ROE improvement)
            final_score = (static_score + change_score + factors['roe_improvement']) / 3
            combined_scores[stock] = final_score
        
        # Normalize scores to [0, 1] range
        if static_scores:
            min_static = min(static_scores.values())
            max_static = max(static_scores.values())
            static_range = max_static - min_static
            if static_range > 0:
                static_scores = {k: (v - min_static) / static_range for k, v in static_scores.items()}
            
        if change_scores:
            min_change = min(change_scores.values())
            max_change = max(change_scores.values())
            change_range = max_change - min_change
            if change_range > 0:
                change_scores = {k: (v - min_change) / change_range for k, v in change_scores.items()}
            
        if combined_scores:
            min_combined = min(combined_scores.values())
            max_combined = max(combined_scores.values())
            combined_range = max_combined - min_combined
            if combined_range > 0:
                combined_scores = {k: (v - min_combined) / combined_range for k, v in combined_scores.items()}
        
        return {
            'static': static_scores,
            'change': change_scores,
            'combined': combined_scores
        }
    
    def construct_portfolio(self):
        """
        Construct portfolio based on combined factors
        """
        print("Constructing portfolio...")
        
        # Get dates for which we have factors
        factor_dates = list(self.factors.keys())
        factor_dates.sort()
        
        for i, date in enumerate(tqdm(factor_dates[:-1])):  # Skip last date
            # Calculate combined factors
            factor_scores = self.calculate_combined_factors(date)
            combined_scores = factor_scores['combined']
            
            # Skip if no scores
            if not combined_scores:
                continue
                
            # Sort stocks by combined score
            sorted_stocks = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
            
            # Select top N stocks
            n_stocks = min(self.top_n, len(sorted_stocks))
            selected = sorted_stocks[:n_stocks]
            selected_stocks = [item[0] for item in selected]
            selected_scores = [item[1] for item in selected]
            
            # Calculate weights (higher weight for higher score)
            if len(selected_stocks) > 0:
                # Min weight is 1%
                min_weight = 0.01
                
                # Normalize scores to sum to (1 - min_weight * N)
                total_score = sum(selected_scores)
                if total_score > 0:
                    remaining_weight = 1 - (min_weight * len(selected_stocks))
                    weights = [min_weight + (score / total_score) * remaining_weight for score in selected_scores]
                else:
                    # Equal weight if all scores are 0
                    weights = [1 / len(selected_stocks)] * len(selected_stocks)
                
                # Normalize weights to sum to 1
                total_weight = sum(weights)
                if total_weight > 0:
                    weights = [w / total_weight for w in weights]
                
                # Store selected stocks and weights
                self.selected_stocks[date] = selected_stocks
                self.portfolio_weights[date] = dict(zip(selected_stocks, weights))
            
        print("Portfolio construction complete!")
    
    def backtest_portfolio(self):
        """
        Backtest the portfolio and calculate performance metrics
        """
        print("Backtesting portfolio...")
        
        # Get valid dates (dates with portfolio weights)
        valid_dates = list(self.portfolio_weights.keys())
        valid_dates.sort()
        
        if not valid_dates:
            print("No valid portfolio dates to backtest")
            return {}, pd.DataFrame(), pd.DataFrame()
        
        # Initialize portfolio values and returns
        portfolio_values = pd.Series(index=self.rebalance_dates, dtype=float)
        portfolio_values.loc[valid_dates[0]] = 1.0  # Start with 1 unit of capital
        
        for i, date in enumerate(tqdm(valid_dates[:-1])):
            next_date = valid_dates[i+1]
            
            # Get portfolio composition
            weights = self.portfolio_weights[date]
            
            # Calculate portfolio return for this period
            portfolio_return = 0.0
            
            for stock, weight in weights.items():
                # Calculate stock return from current date to next date
                if stock in self.stock_data:
                    try:
                        current_price = self.stock_data[stock].loc[date, 'price']
                        next_price = self.stock_data[stock].loc[next_date, 'price']
                        stock_return = (next_price / current_price) - 1
                        portfolio_return += weight * stock_return
                    except:
                        # Skip if price data is not available
                        pass
            
            # Store portfolio return
            self.portfolio_returns[next_date] = portfolio_return
            
            # Update portfolio value
            portfolio_values.loc[next_date] = portfolio_values.loc[date] * (1 + portfolio_return)
        
        # Fill in missing values (for dates between rebalances)
        portfolio_values = portfolio_values.ffill()
        
        # Calculate benchmark (CSI 300) returns and values
        benchmark_returns = pd.Series(index=valid_dates[1:], dtype=float)
        benchmark_values = pd.Series(index=self.rebalance_dates, dtype=float)
        benchmark_values.loc[valid_dates[0]] = 1.0  # Start with 1 unit of capital
        
        for i, date in enumerate(valid_dates[:-1]):
            next_date = valid_dates[i+1]
            
            # Calculate benchmark return
            try:
                current_price = self.index_data.loc[date, 'price']
                next_price = self.index_data.loc[next_date, 'price']
                bench_return = (next_price / current_price) - 1
                
                # Store benchmark return and value
                benchmark_returns[next_date] = bench_return
                benchmark_values.loc[next_date] = benchmark_values.loc[date] * (1 + bench_return)
            except:
                # Skip if index data is not available
                pass
        
        # Fill in missing values (for dates between rebalances)
        benchmark_values = benchmark_values.ffill()
        
        # Convert portfolio returns to Series
        portfolio_returns_series = pd.Series(self.portfolio_returns)
        
        # Calculate performance metrics
        performance = {}
        
        # Convert to dataframe for analysis
        returns_df = pd.DataFrame({
            'Portfolio': portfolio_returns_series,
            'Benchmark': benchmark_returns
        })
        
        # Filter out dates with NaN values
        returns_df = returns_df.dropna()
        
        # Filter portfolio and benchmark values to match the returns dates
        values_df = pd.DataFrame({
            'Portfolio': portfolio_values,
            'Benchmark': benchmark_values
        })
        
        if len(returns_df) == 0:
            print("Not enough data to compute performance metrics")
            return {}, values_df, returns_df
        
        # Calculate cumulative returns
        performance['cumulative_return'] = {
            'Portfolio': portfolio_values.iloc[-1] / portfolio_values.iloc[0] - 1 if len(portfolio_values) > 1 else 0,
            'Benchmark': benchmark_values.iloc[-1] / benchmark_values.iloc[0] - 1 if len(benchmark_values) > 1 else 0
        }
        
        # Calculate annualized returns (assuming monthly rebalancing)
        num_years = max(1, (self.end_date - self.start_date).days / 365.25)
        performance['annualized_return'] = {
            'Portfolio': (1 + performance['cumulative_return']['Portfolio']) ** (1/num_years) - 1,
            'Benchmark': (1 + performance['cumulative_return']['Benchmark']) ** (1/num_years) - 1
        }
        
        # Calculate volatility (annualized)
        performance['volatility'] = {
            'Portfolio': returns_df['Portfolio'].std() * np.sqrt(12) if len(returns_df) > 1 else 0,
            'Benchmark': returns_df['Benchmark'].std() * np.sqrt(12) if len(returns_df) > 1 else 0
        }
        
        # Calculate Sharpe ratio (assuming risk-free rate of 0)
        performance['sharpe_ratio'] = {
            'Portfolio': performance['annualized_return']['Portfolio'] / performance['volatility']['Portfolio'] 
                         if performance['volatility']['Portfolio'] > 0 else 0,
            'Benchmark': performance['annualized_return']['Benchmark'] / performance['volatility']['Benchmark'] 
                         if performance['volatility']['Benchmark'] > 0 else 0
        }
        
        # Calculate maximum drawdown
        if len(returns_df) > 1:
            portfolio_cumret = (1 + returns_df['Portfolio']).cumprod()
            benchmark_cumret = (1 + returns_df['Benchmark']).cumprod()
            
            portfolio_peaks = portfolio_cumret.cummax()
            benchmark_peaks = benchmark_cumret.cummax()
            
            portfolio_drawdowns = (portfolio_cumret / portfolio_peaks - 1)
            benchmark_drawdowns = (benchmark_cumret / benchmark_peaks - 1)
            
            performance['max_drawdown'] = {
                'Portfolio': portfolio_drawdowns.min() if len(portfolio_drawdowns) > 0 else 0,
                'Benchmark': benchmark_drawdowns.min() if len(benchmark_drawdowns) > 0 else 0
            }
        else:
            performance['max_drawdown'] = {
                'Portfolio': 0,
                'Benchmark': 0
            }
        
        # Calculate win rate
        performance['win_rate'] = {
            'Portfolio': (returns_df['Portfolio'] > 0).mean() if len(returns_df) > 0 else 0,
            'Benchmark': (returns_df['Benchmark'] > 0).mean() if len(returns_df) > 0 else 0
        }
        
        # Calculate excess return (alpha)
        performance['excess_return'] = {
            'Total': performance['cumulative_return']['Portfolio'] - performance['cumulative_return']['Benchmark'],
            'Annualized': performance['annualized_return']['Portfolio'] - performance['annualized_return']['Benchmark']
        }
        
        # Calculate information ratio
        if len(returns_df) > 1:
            excess_returns = returns_df['Portfolio'] - returns_df['Benchmark']
            tracking_error = excess_returns.std() * np.sqrt(12)
            
            performance['information_ratio'] = {
                'Value': performance['excess_return']['Annualized'] / tracking_error if tracking_error > 0 else 0
            }
        else:
            performance['information_ratio'] = {
                'Value': 0
            }
        
        return performance, values_df, returns_df
    
    def run_analysis(self):
        """
        Run the full analysis pipeline
        """
        # Load data
        self.load_data()
        
        # Calculate factors
        self.calculate_factors()
        
        # Calculate IC for all factors
        print("Calculating factor ICs...")
        factor_ics = {}
        factor_names = [
            'northbound_pct', 
            'northbound_internal_pct', 
            'northbound_deviation',
            'northbound_internal_change', 
            'northbound_6m_change', 
            'roe_improvement'
        ]
        
        for factor in factor_names:
            factor_ics[factor] = self.calculate_factor_ic(factor)
            ic_mean = factor_ics[factor].mean() if len(factor_ics[factor]) > 0 else 0
            ic_win_rate = (factor_ics[factor] > 0).mean() if len(factor_ics[factor]) > 0 else 0
            print(f"{factor}: Avg IC = {ic_mean:.4f}, IC Win Rate = {ic_win_rate:.2%}")
        
        # Construct portfolio
        self.construct_portfolio()
        
        # Backtest portfolio
        performance, values_df, returns_df = self.backtest_portfolio()
        
        # Print performance summary
        self.print_performance(performance)
        
        # Plot results
        self.plot_results(values_df, returns_df, factor_ics)
        
        return performance, values_df, returns_df, factor_ics
    
    def print_performance(self, performance):
        """
        Print performance metrics
        
        Parameters:
        -----------
        performance : dict
            Performance metrics dictionary
        """
        print("\n" + "="*50)
        print(" Northbound Investment Strategy Performance Summary ")
        print("="*50)
        
        print("\nBacktest Period: {} to {}".format(self.start_date.strftime('%Y-%m-%d'), self.end_date.strftime('%Y-%m-%d')))
        
        # Check if performance data is available
        if not performance:
            print("\nNo performance data available.")
            return
            
        print("\nPerformance Metrics:")
        print("Cumulative Return:")
        print(f"  Portfolio: {performance['cumulative_return']['Portfolio']:.2%}")
        print(f"  Benchmark: {performance['cumulative_return']['Benchmark']:.2%}")
        print(f"  Excess: {performance['excess_return']['Total']:.2%}")
        
        print("\nAnnualized Return:")
        print(f"  Portfolio: {performance['annualized_return']['Portfolio']:.2%}")
        print(f"  Benchmark: {performance['annualized_return']['Benchmark']:.2%}")
        print(f"  Excess: {performance['excess_return']['Annualized']:.2%}")
        
        print("\nVolatility:")
        print(f"  Portfolio: {performance['volatility']['Portfolio']:.2%}")
        print(f"  Benchmark: {performance['volatility']['Benchmark']:.2%}")
        
        print("\nSharpe Ratio:")
        print(f"  Portfolio: {performance['sharpe_ratio']['Portfolio']:.2f}")
        print(f"  Benchmark: {performance['sharpe_ratio']['Benchmark']:.2f}")
        
        print("\nMaximum Drawdown:")
        print(f"  Portfolio: {performance['max_drawdown']['Portfolio']:.2%}")
        print(f"  Benchmark: {performance['max_drawdown']['Benchmark']:.2%}")
        
        print("\nWin Rate:")
        print(f"  Portfolio: {performance['win_rate']['Portfolio']:.2%}")
        print(f"  Benchmark: {performance['win_rate']['Benchmark']:.2%}")
        
        print("\nInformation Ratio:")
        print(f"  Value: {performance['information_ratio']['Value']:.2f}")
        
        print("\n" + "="*50)
    
    def plot_results(self, values_df, returns_df, factor_ics):
        """
        Plot backtest results
        
        Parameters:
        -----------
        values_df : pd.DataFrame
            Portfolio and benchmark values
        returns_df : pd.DataFrame
            Portfolio and benchmark returns
        factor_ics : dict
            Dictionary of factor ICs
        """
        # Check if there's data to plot
        if values_df.empty or returns_df.empty or all(len(ics) == 0 for ics in factor_ics.values()):
            print("Not enough data to generate plots")
            return
            
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot cumulative returns
        values_df.plot(ax=axes[0, 0], title='Cumulative Returns')
        axes[0, 0].set_ylabel('Value')
        axes[0, 0].grid(True)
        
        # Plot factor ICs
        ic_df = pd.DataFrame({k: v for k, v in factor_ics.items() if len(v) > 0})
        
        if not ic_df.empty:
            # Use a rolling mean with a smaller window if we have limited data
            window_size = min(6, len(ic_df) // 2)
            if window_size > 0:
                ic_df.rolling(window=window_size).mean().plot(ax=axes[0, 1], title=f'{window_size}-Month Rolling Average IC')
            else:
                ic_df.plot(ax=axes[0, 1], title='Factor ICs')
            axes[0, 1].set_ylabel('IC Value')
            axes[0, 1].grid(True)
        else:
            axes[0, 1].text(0.5, 0.5, 'Not enough data for IC plot', 
                     horizontalalignment='center', verticalalignment='center')
        
        # Plot monthly returns
        if not returns_df.empty:
            returns_df.plot(kind='bar', ax=axes[1, 0], title='Monthly Returns')
            axes[1, 0].set_ylabel('Return')
            axes[1, 0].grid(True)
        else:
            axes[1, 0].text(0.5, 0.5, 'Not enough data for returns plot', 
                     horizontalalignment='center', verticalalignment='center')
        
        # Plot drawdowns
        if len(returns_df) > 1:
            portfolio_cumret = (1 + returns_df['Portfolio']).cumprod()
            benchmark_cumret = (1 + returns_df['Benchmark']).cumprod()
            
            portfolio_peaks = portfolio_cumret.cummax()
            benchmark_peaks = benchmark_cumret.cummax()
            
            portfolio_drawdowns = (portfolio_cumret / portfolio_peaks - 1)
            benchmark_drawdowns = (benchmark_cumret / benchmark_peaks - 1)
            
            drawdowns_df = pd.DataFrame({
                'Portfolio': portfolio_drawdowns,
                'Benchmark': benchmark_drawdowns
            })
            
            drawdowns_df.plot(ax=axes[1, 1], title='Drawdowns')
            axes[1, 1].set_ylabel('Drawdown')
            axes[1, 1].grid(True)
        else:
            axes[1, 1].text(0.5, 0.5, 'Not enough data for drawdowns plot', 
                     horizontalalignment='center', verticalalignment='center')
        
        plt.tight_layout()
        plt.savefig('northbound_strategy_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Plot factor IC bar chart
        plt.figure(figsize=(12, 6))
        
        # Only include factors with IC data
        valid_factors = [f for f in factor_ics.keys() if len(factor_ics[f]) > 0]
        
        if valid_factors:
            ic_means = {factor: factor_ics[factor].mean() for factor in valid_factors}
            factor_names = [
                'Holding %', 
                'Internal %', 
                '3M Deviation',
                '3M Change', 
                '6M Change', 
                'ROE Improvement'
            ][:len(valid_factors)]  # Limit to number of valid factors
            
            plt.bar(factor_names, [ic_means[f] for f in valid_factors])
            plt.axhline(y=0, color='r', linestyle='-')
            plt.title('Average IC by Factor')
            plt.ylabel('Average IC')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig('northbound_factor_ic.png', dpi=300, bbox_inches='tight')
            plt.show()
        else:
            print("Not enough data for factor IC bar chart")


# Run the analysis
if __name__ == "__main__":
    # Initialize the analyzer
    analyzer = NorthboundFlowAnalyzer(start_date='2017-01-01', end_date='2021-04-30')
    
    # Run analysis
    performance, values_df, returns_df, factor_ics = analyzer.run_analysis()