import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from datetime import datetime, timedelta
import statsmodels.api as sm
from tqdm import tqdm

# Set up plotting style
plt.style.use('fivethirtyeight')
sns.set_style('whitegrid')

# Seed for reproducibility
np.random.seed(42)

class AlphaStrategyTester:
    def __init__(self):
        """Initialize the strategy tester"""
        pass
    
    def simulate_market_data(self, n_stocks=100, n_days=1000, start_date='2010-01-01'):
        """
        Simulate a market dataset with multiple stocks and their features
        
        Parameters:
        -----------
        n_stocks : int
            Number of stocks to simulate
        n_days : int
            Number of days to simulate
        start_date : str
            Start date for the simulation
            
        Returns:
        --------
        market_data : pandas.DataFrame
            Simulated market data
        """
        # Create date range
        dates = pd.date_range(start=start_date, periods=n_days)
        
        # Simulate market return with some autocorrelation
        market_return = np.zeros(n_days)
        market_return[0] = np.random.normal(0.0005, 0.01)
        
        # AR(1) process for market return
        for i in range(1, n_days):
            market_return[i] = 0.0005 + 0.1 * market_return[i-1] + np.random.normal(0, 0.01)
        
        # Convert to price series
        market_price = 100 * np.cumprod(1 + market_return)
        
        # Create a DataFrame to store all market data
        market_data = pd.DataFrame({'date': dates, 'market_return': market_return, 
                                  'market_price': market_price})
        
        # Simulate stock specific data
        stock_data = {}
        
        for stock_id in range(n_stocks):
            # Randomly assign a beta between 0.5 and 1.5
            beta = np.random.uniform(0.5, 1.5)
            
            # Size factor (market cap) - lognormal distribution
            market_cap = np.exp(np.random.normal(10, 1.5))
            
            # Specify sector (0-9, 10 sectors total)
            sector = np.random.randint(0, 10)
            
            # Stock specific return - some autocorrelation
            stock_specific_return = np.zeros(n_days)
            stock_specific_return[0] = np.random.normal(0, 0.02)
            
            # AR(1) process for stock specific return with some autocorrelation
            autocorr = np.random.uniform(-0.2, 0.2)  # Different autocorrelation for each stock
            for i in range(1, n_days):
                stock_specific_return[i] = autocorr * stock_specific_return[i-1] + np.random.normal(0, 0.02)
            
            # Generate stock return based on CAPM plus stock specific return
            stock_return = 0.0002 + beta * market_return + stock_specific_return
            
            # Convert to price series
            stock_price = 100 * np.cumprod(1 + stock_return)
            
            # Create stock data dictionary
            stock_data[f'stock_{stock_id}'] = {
                'return': stock_return,
                'price': stock_price,
                'beta': beta,
                'market_cap': market_cap,
                'sector': sector
            }
        
        # Add stock data to market_data
        for stock_id, data in stock_data.items():
            market_data[f'{stock_id}_return'] = data['return']
            market_data[f'{stock_id}_price'] = data['price']
            market_data[f'{stock_id}_beta'] = data['beta']
            market_data[f'{stock_id}_market_cap'] = data['market_cap']
            market_data[f'{stock_id}_sector'] = data['sector']
        
        # Set date as index
        market_data.set_index('date', inplace=True)
        
        return market_data, stock_data
    
    def calculate_rolling_beta(self, market_data, window=252):
        """
        Calculate rolling betas for all stocks
        
        Parameters:
        -----------
        market_data : pandas.DataFrame
            Market data with stock returns and market returns
        window : int
            Rolling window size for beta calculation
            
        Returns:
        --------
        rolling_betas : pandas.DataFrame
            Rolling betas for all stocks
        """
        rolling_betas = pd.DataFrame(index=market_data.index)
        
        # Get all stock return columns
        stock_return_cols = [col for col in market_data.columns if '_return' in col and 'market' not in col]
        
        # Calculate rolling beta for each stock
        for stock_col in tqdm(stock_return_cols, desc="Calculating rolling betas"):
            stock_id = stock_col.split('_')[0]
            
            # Rolling regression
            rolling_beta = []
            
            for i in range(window, len(market_data)):
                # Get window data
                window_data = market_data.iloc[i-window:i]
                
                # Run regression
                X = sm.add_constant(window_data['market_return'])
                y = window_data[stock_col]
                model = sm.OLS(y, X).fit()
                
                # Store beta
                rolling_beta.append(model.params[1])
            
            # Add NaN for the initial window
            rolling_beta = [np.nan] * window + rolling_beta
            
            # Add to DataFrame
            rolling_betas[f'{stock_id}_rolling_beta'] = rolling_beta
        
        return rolling_betas
    
    def beta_arbitrage_strategy(self, market_data, rolling_betas, lookback=20, n_portfolios=10):
        """
        Implement the Beta Arbitrage strategy
        
        Parameters:
        -----------
        market_data : pandas.DataFrame
            Market data with stock returns and market returns
        rolling_betas : pandas.DataFrame
            Rolling betas for all stocks
        lookback : int
            Lookback period for portfolio formation
        n_portfolios : int
            Number of portfolios to form based on beta ranking
        
        Returns:
        --------
        strategy_returns : pandas.DataFrame
            Strategy returns
        """
        # Get all stock return columns
        stock_return_cols = [col for col in market_data.columns if '_return' in col and 'market' not in col]
        
        # Initialize strategy returns
        strategy_returns = pd.DataFrame(index=market_data.index[lookback+1:])
        
        # Get all rolling beta columns
        beta_cols = rolling_betas.columns
        
        # Implement strategy
        for t in tqdm(range(lookback+1, len(market_data)), desc="Implementing Beta Arbitrage strategy"):
            # Get current date
            current_date = market_data.index[t]
            
            # Get previous date (for portfolio formation)
            prev_date = market_data.index[t-1]
            
            # Get beta values from previous date
            betas = {}
            for col in beta_cols:
                stock_id = col.split('_')[0]
                if not np.isnan(rolling_betas.loc[prev_date, col]):
                    betas[stock_id] = rolling_betas.loc[prev_date, col]
            
            # Sort stocks by beta
            sorted_betas = sorted(betas.items(), key=lambda x: x[1])
            
            # Form portfolios based on beta ranking
            portfolio_size = len(sorted_betas) // n_portfolios
            
            if portfolio_size > 0:  # Ensure we have enough stocks for portfolios
                # Low beta portfolio (first decile)
                low_beta_stocks = sorted_betas[:portfolio_size]
                
                # High beta portfolio (last decile)
                high_beta_stocks = sorted_betas[-portfolio_size:]
                
                # Calculate average beta for each portfolio
                low_beta_avg = np.mean([beta for _, beta in low_beta_stocks])
                high_beta_avg = np.mean([beta for _, beta in high_beta_stocks])
                
                # Calculate returns for each portfolio
                low_beta_return = np.mean([market_data.loc[current_date, f'{stock}_return'] for stock, _ in low_beta_stocks])
                high_beta_return = np.mean([market_data.loc[current_date, f'{stock}_return'] for stock, _ in high_beta_stocks])
                
                # Calculate BAB factor return (beta arbitrage)
                # Long low beta (with leverage to get beta=1) and short high beta (with leverage to get beta=1)
                if low_beta_avg > 0 and high_beta_avg > 0:  # Avoid division by zero
                    bab_return = (1/low_beta_avg) * low_beta_return - (1/high_beta_avg) * high_beta_return
                    
                    # Store returns
                    strategy_returns.loc[current_date, 'low_beta_return'] = low_beta_return
                    strategy_returns.loc[current_date, 'high_beta_return'] = high_beta_return
                    strategy_returns.loc[current_date, 'bab_return'] = bab_return
                    strategy_returns.loc[current_date, 'low_beta_avg'] = low_beta_avg
                    strategy_returns.loc[current_date, 'high_beta_avg'] = high_beta_avg
        
        return strategy_returns
    
    def trend_factor_strategy(self, market_data, lookback_periods=[5, 21, 63, 126, 252], n_portfolios=5):
        """
        Implement the Trend Factor strategy
        
        Parameters:
        -----------
        market_data : pandas.DataFrame
            Market data with stock returns and market returns
        lookback_periods : list
            List of lookback periods for moving averages
        n_portfolios : int
            Number of portfolios to form based on trend factor ranking
            
        Returns:
        --------
        strategy_returns : pandas.DataFrame
            Strategy returns
        """
        # Get all stock price columns
        stock_price_cols = [col for col in market_data.columns if '_price' in col and 'market' not in col]
        
        # Initialize strategy returns
        max_lookback = max(lookback_periods)
        strategy_returns = pd.DataFrame(index=market_data.index[max_lookback:])
        
        # Calculate trend factor for each stock
        for t in tqdm(range(max_lookback, len(market_data)), desc="Implementing Trend Factor strategy"):
            # Get current date
            current_date = market_data.index[t]
            
            # Calculate trend signals for each stock
            trend_signals = {}
            
            for col in stock_price_cols:
                stock_id = col.split('_')[0]
                
                # Initialize signal sum
                signal_sum = 0
                
                # Calculate moving average signals
                for period in lookback_periods:
                    # Calculate moving average
                    ma = market_data[col].iloc[t-period:t].mean()
                    
                    # Current price
                    current_price = market_data.loc[current_date, col]
                    
                    # Calculate signal (1 if price > MA, -1 otherwise)
                    signal = 1 if current_price > ma else -1
                    
                    # Add to signal sum
                    signal_sum += signal
                
                # Normalize signal to range [-1, 1]
                normalized_signal = signal_sum / len(lookback_periods)
                
                # Store trend factor
                trend_signals[stock_id] = normalized_signal
            
            # Sort stocks by trend factor
            sorted_trends = sorted(trend_signals.items(), key=lambda x: x[1])
            
            # Form portfolios based on trend factor ranking
            portfolio_size = len(sorted_trends) // n_portfolios
            
            if portfolio_size > 0:  # Ensure we have enough stocks for portfolios
                # Low trend portfolio (first quintile)
                low_trend_stocks = sorted_trends[:portfolio_size]
                
                # High trend portfolio (last quintile)
                high_trend_stocks = sorted_trends[-portfolio_size:]
                
                # Calculate returns for each portfolio
                low_trend_return = np.mean([market_data.loc[current_date, f'{stock}_return'] for stock, _ in low_trend_stocks])
                high_trend_return = np.mean([market_data.loc[current_date, f'{stock}_return'] for stock, _ in high_trend_stocks])
                
                # Calculate trend factor return (long high trend, short low trend)
                trend_factor_return = high_trend_return - low_trend_return
                
                # Store returns
                strategy_returns.loc[current_date, 'low_trend_return'] = low_trend_return
                strategy_returns.loc[current_date, 'high_trend_return'] = high_trend_return
                strategy_returns.loc[current_date, 'trend_factor_return'] = trend_factor_return
        
        return strategy_returns
    
    def analyze_strategy_performance(self, strategy_returns, strategy_name, benchmark_return=None):
        """
        Analyze the performance of a strategy
        
        Parameters:
        -----------
        strategy_returns : pandas.DataFrame
            Strategy returns
        strategy_name : str
            Name of the strategy
        benchmark_return : pandas.Series or None
            Benchmark returns for comparison
            
        Returns:
        --------
        performance_metrics : dict
            Performance metrics
        """
        # Extract main strategy return
        if strategy_name == 'Beta Arbitrage':
            strategy_return = strategy_returns['bab_return']
        elif strategy_name == 'Trend Factor':
            strategy_return = strategy_returns['trend_factor_return']
        else:
            raise ValueError("Strategy name not recognized")
        
        # Calculate cumulative returns
        cumulative_return = (1 + strategy_return).cumprod() - 1
        
        # Calculate annualized return
        annualized_return = ((1 + cumulative_return.iloc[-1]) ** (252 / len(strategy_return))) - 1
        
        # Calculate annualized volatility
        annualized_vol = strategy_return.std() * np.sqrt(252)
        
        # Calculate Sharpe ratio (assuming risk-free rate = 0)
        sharpe_ratio = annualized_return / annualized_vol
        
        # Calculate maximum drawdown
        rolling_max = cumulative_return.cummax()
        drawdown = (cumulative_return - rolling_max) / (1 + rolling_max)
        max_drawdown = drawdown.min()
        
        # Calculate hit ratio (percentage of positive returns)
        hit_ratio = (strategy_return > 0).mean()
        
        # Calculate average gain and loss
        avg_gain = strategy_return[strategy_return > 0].mean()
        avg_loss = strategy_return[strategy_return < 0].mean()
        
        # Calculate gain/loss ratio
        gain_loss_ratio = abs(avg_gain / avg_loss) if avg_loss != 0 else np.inf
        
        # Calculate monthly returns
        monthly_returns = strategy_return.resample('M').apply(lambda x: (1 + x).prod() - 1)
        
        # Calculate yearly returns
        yearly_returns = strategy_return.resample('Y').apply(lambda x: (1 + x).prod() - 1)
        
        # Output performance metrics
        print(f"\n==== {strategy_name} Strategy Performance ====")
        print(f"Annualized Return: {annualized_return:.2%}")
        print(f"Annualized Volatility: {annualized_vol:.2%}")
        print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
        print(f"Maximum Drawdown: {max_drawdown:.2%}")
        print(f"Hit Ratio: {hit_ratio:.2%}")
        print(f"Average Gain: {avg_gain:.2%}")
        print(f"Average Loss: {avg_loss:.2%}")
        print(f"Gain/Loss Ratio: {gain_loss_ratio:.2f}")
        
        # Compare with benchmark if provided
        if benchmark_return is not None:
            # Match dates
            benchmark_return = benchmark_return.loc[strategy_return.index]
            
            # Calculate benchmark cumulative return
            benchmark_cumulative_return = (1 + benchmark_return).cumprod() - 1
            
            # Calculate benchmark annualized return
            benchmark_annualized_return = ((1 + benchmark_cumulative_return.iloc[-1]) ** (252 / len(benchmark_return))) - 1
            
            # Calculate benchmark annualized volatility
            benchmark_annualized_vol = benchmark_return.std() * np.sqrt(252)
            
            # Calculate benchmark Sharpe ratio
            benchmark_sharpe_ratio = benchmark_annualized_return / benchmark_annualized_vol
            
            # Calculate information ratio
            tracking_error = (strategy_return - benchmark_return).std() * np.sqrt(252)
            information_ratio = (annualized_return - benchmark_annualized_return) / tracking_error
            
            # Calculate beta to benchmark
            cov_matrix = np.cov(strategy_return, benchmark_return)
            beta = cov_matrix[0, 1] / cov_matrix[1, 1]
            
            # Calculate alpha
            alpha = annualized_return - beta * benchmark_annualized_return
            
            print("\n==== Comparison with Benchmark ====")
            print(f"Strategy Annualized Return: {annualized_return:.2%}")
            print(f"Benchmark Annualized Return: {benchmark_annualized_return:.2%}")
            print(f"Alpha: {alpha:.2%}")
            print(f"Beta: {beta:.2f}")
            print(f"Information Ratio: {information_ratio:.2f}")
            print(f"Tracking Error: {tracking_error:.2%}")
        
        # Plot cumulative returns
        plt.figure(figsize=(12, 6))
        cumulative_return.plot(label=f'{strategy_name} Strategy')
        
        if benchmark_return is not None:
            benchmark_cumulative_return.plot(label='Benchmark', alpha=0.7)
        
        plt.title(f'{strategy_name} Strategy Cumulative Returns')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Return')
        plt.legend()
        plt.grid(True)
        plt.show()
        
        # Plot monthly returns heatmap
        monthly_returns = monthly_returns.to_frame()
        monthly_returns.index = monthly_returns.index.to_period('M')
        monthly_returns = monthly_returns.pivot_table(
            values=monthly_returns.columns[0],
            index=monthly_returns.index.year,
            columns=monthly_returns.index.month
        )
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(monthly_returns, annot=True, fmt='.1%', cmap='RdYlGn',
                   linewidths=1, center=0, cbar_kws={'label': 'Monthly Return'})
        plt.title(f'{strategy_name} Strategy Monthly Returns')
        plt.xlabel('Month')
        plt.ylabel('Year')
        plt.show()
        
        # Plot drawdown
        plt.figure(figsize=(12, 6))
        drawdown.plot(color='red', alpha=0.7)
        plt.title(f'{strategy_name} Strategy Drawdown')
        plt.xlabel('Date')
        plt.ylabel('Drawdown')
        plt.grid(True)
        plt.show()
        
        # Create performance metrics dictionary
        performance_metrics = {
            'annualized_return': annualized_return,
            'annualized_vol': annualized_vol,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'hit_ratio': hit_ratio,
            'avg_gain': avg_gain,
            'avg_loss': avg_loss,
            'gain_loss_ratio': gain_loss_ratio,
            'monthly_returns': monthly_returns,
            'cumulative_return': cumulative_return
        }
        
        if benchmark_return is not None:
            performance_metrics.update({
                'benchmark_annualized_return': benchmark_annualized_return,
                'benchmark_annualized_vol': benchmark_annualized_vol,
                'benchmark_sharpe_ratio': benchmark_sharpe_ratio,
                'benchmark_cumulative_return': benchmark_cumulative_return,
                'alpha': alpha,
                'beta': beta,
                'information_ratio': information_ratio,
                'tracking_error': tracking_error
            })
        
        return performance_metrics

# Test the strategies
strategy_tester = AlphaStrategyTester()

# Simulate market data
print("Simulating market data...")
market_data, stock_data = strategy_tester.simulate_market_data(n_stocks=50, n_days=1000)
print(f"Market data shape: {market_data.shape}")

# Calculate rolling betas
print("\nCalculating rolling betas...")
rolling_betas = strategy_tester.calculate_rolling_beta(market_data, window=126)

# Test Beta Arbitrage strategy
print("\nImplementing Beta Arbitrage strategy...")
beta_arbitrage_returns = strategy_tester.beta_arbitrage_strategy(market_data, rolling_betas, lookback=20, n_portfolios=5)

# Test Trend Factor strategy
print("\nImplementing Trend Factor strategy...")
trend_factor_returns = strategy_tester.trend_factor_strategy(market_data, lookback_periods=[5, 21, 63, 126, 252], n_portfolios=5)

# Analyze Beta Arbitrage strategy performance
print("\nAnalyzing Beta Arbitrage strategy performance...")
beta_arb_performance = strategy_tester.analyze_strategy_performance(
    beta_arbitrage_returns, 
    'Beta Arbitrage', 
    benchmark_return=market_data['market_return']
)

# Analyze Trend Factor strategy performance
print("\nAnalyzing Trend Factor strategy performance...")
trend_factor_performance = strategy_tester.analyze_strategy_performance(
    trend_factor_returns, 
    'Trend Factor', 
    benchmark_return=market_data['market_return']
)

# Compare strategies
print("\n==== Strategy Comparison ====")
print(f"Beta Arbitrage Sharpe Ratio: {beta_arb_performance['sharpe_ratio']:.2f}")
print(f"Trend Factor Sharpe Ratio: {trend_factor_performance['sharpe_ratio']:.2f}")

# Plot strategy comparison
plt.figure(figsize=(12, 6))
beta_arb_performance['cumulative_return'].plot(label='Beta Arbitrage')
trend_factor_performance['cumulative_return'].plot(label='Trend Factor')
((1 + market_data['market_return'].loc[beta_arb_performance['cumulative_return'].index]).cumprod() - 1).plot(
    label='Market', alpha=0.7
)
plt.title('Strategy Comparison: Cumulative Returns')
plt.xlabel('Date')
plt.ylabel('Cumulative Return')
plt.legend()
plt.grid(True)
plt.show()