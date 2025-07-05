import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from tqdm import tqdm
import time
import statsmodels.api as sm
from scipy.stats import pearsonr, spearmanr
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

class SentimentAnalyzer:
    """
    Class to analyze financial news sentiment and calculate sentiment scores
    """
    
    def __init__(self, use_gpt=False):
        """
        Initialize the sentiment analyzer
        
        Parameters:
        -----------
        use_gpt : bool
            Whether to use GPT for sentiment analysis (True) or simulated results (False)
        """
        self.use_gpt = use_gpt
        
    def analyze_text(self, text, date):
        """
        Analyze text to extract headlines and sentiment
        
        Parameters:
        -----------
        text : str
            The financial news text to analyze
        date : datetime
            The date of the news
            
        Returns:
        --------
        dict
            A dictionary containing date, positive count, negative count, and sentiment score
        """
        if self.use_gpt:
            # This would be the actual implementation with GPT API
            # For this simulation, we'll skip this part
            pass
        else:
            # Simulate the result of the two-stage prompt approach
            # Generate random number of positive and negative headlines
            total_headlines = 15
            positive = np.random.randint(0, total_headlines + 1)
            negative = np.random.randint(0, total_headlines - positive + 1)
            
            # Calculate sentiment score
            if positive + negative > 0:
                sentiment_score = (positive - negative) / (positive + negative)
            else:
                sentiment_score = 0
                
            return {
                'date': date,
                'positive': positive,
                'negative': negative,
                'sentiment_score': sentiment_score
            }

class MarketSimulator:
    """
    Class to simulate financial market data and news
    """
    
    def __init__(self, start_date='2010-01-01', end_date='2023-10-31'):
        """
        Initialize the market simulator
        
        Parameters:
        -----------
        start_date : str
            Start date for the simulation
        end_date : str
            End date for the simulation
        """
        self.start_date = datetime.strptime(start_date, '%Y-%m-%d')
        self.end_date = datetime.strptime(end_date, '%Y-%m-%d')
        
        # Market parameters
        self.markets = {
            'US': {'drift': 0.0001, 'volatility': 0.012, 'sentiment_impact': 0.4},
            'US_Tech': {'drift': 0.00015, 'volatility': 0.015, 'sentiment_impact': 0.5},
            'Japan': {'drift': 0.00008, 'volatility': 0.013, 'sentiment_impact': 0.3},
            'Europe': {'drift': 0.00007, 'volatility': 0.014, 'sentiment_impact': 0.35},
            'UK': {'drift': 0.00006, 'volatility': 0.011, 'sentiment_impact': 0.25},
            'Emerging': {'drift': 0.00011, 'volatility': 0.016, 'sentiment_impact': 0.45}
        }
        
        # Simulate true underlying sentiment process
        self.dates = self._generate_business_dates()
        self.true_sentiment = self._generate_sentiment()
        
        # Generate market data that's influenced by sentiment
        self.market_data = self._generate_market_data()
        
        # Generate simulated news texts
        self.news_data = self._generate_news_data()
        
    def _generate_business_dates(self):
        """Generate a list of business dates for the simulation period"""
        dates = []
        current_date = self.start_date
        
        while current_date <= self.end_date:
            # Skip weekends
            if current_date.weekday() < 5:  # 0-4 are Monday to Friday
                dates.append(current_date)
            current_date += timedelta(days=1)
            
        return dates
    
    def _generate_sentiment(self):
        """Generate a true underlying sentiment process"""
        # AR(1) process for sentiment
        n = len(self.dates)
        phi = 0.97  # high persistence in sentiment
        sigma = 0.15  # volatility of sentiment shocks
        
        sentiment = np.zeros(n)
        sentiment[0] = np.random.normal(0, sigma)
        
        for i in range(1, n):
            sentiment[i] = phi * sentiment[i-1] + np.random.normal(0, sigma)
        
        # Scale to [-1, 1]
        sentiment = np.tanh(sentiment)
        
        return sentiment
    
    def _generate_market_data(self):
        """Generate market data for all markets"""
        n = len(self.dates)
        market_data = {}
        
        # Create a common market factor that will affect all markets
        market_factor = np.zeros(n)
        market_factor[0] = np.random.normal(0, 1)
        
        for i in range(1, n):
            market_factor[i] = 0.95 * market_factor[i-1] + np.random.normal(0, 0.2)
        
        # Generate specific market data
        for market_name, params in self.markets.items():
            prices = np.zeros(n)
            prices[0] = 1000  # Starting price
            
            # Daily returns influenced by sentiment with some lag
            for i in range(1, n):
                # Short-term positive correlation with sentiment
                short_term_impact = 0
                for lag in range(1, 21):  # 1-20 day lag
                    if i - lag >= 0:
                        weight = np.exp(-0.1 * lag)  # Exponential decay of influence
                        short_term_impact += weight * self.true_sentiment[i-lag]
                
                # Long-term negative correlation (mean reversion)
                long_term_impact = 0
                for lag in range(100, 251):  # 100-250 day lag
                    if i - lag >= 0:
                        weight = np.exp(-0.01 * (lag - 100))  # Exponential decay
                        long_term_impact -= weight * self.true_sentiment[i-lag]
                
                # Combine impacts
                sentiment_impact = (short_term_impact * 0.8 + long_term_impact * 0.2) * params['sentiment_impact']
                
                # Market return with drift, common factor, sentiment impact, and idiosyncratic noise
                returns = params['drift'] + 0.4 * market_factor[i] + sentiment_impact + np.random.normal(0, params['volatility'])
                
                # Update price
                prices[i] = prices[i-1] * (1 + returns)
            
            # Store market data
            market_data[market_name] = prices
        
        return market_data
    
    def _generate_news_data(self):
        """Generate simulated news texts"""
        # For simulation, we'll just generate a placeholder
        # In a real implementation, this would contain actual news text
        return [f"Market news for {date.strftime('%Y-%m-%d')}" for date in self.dates]
    
    def get_data(self):
        """Get the simulated data"""
        data = {
            'date': self.dates,
            'true_sentiment': self.true_sentiment,
            'news': self.news_data
        }
        
        # Add market prices
        for market_name, prices in self.market_data.items():
            data[f'{market_name}_price'] = prices
        
        return pd.DataFrame(data)

class SentimentAnalysisStrategy:
    """
    Class to implement and evaluate the sentiment analysis strategy
    """
    
    def __init__(self, market_data, sentiment_analyzer):
        """
        Initialize the strategy
        
        Parameters:
        -----------
        market_data : pd.DataFrame
            The market data with prices and news
        sentiment_analyzer : SentimentAnalyzer
            The sentiment analyzer to use
        """
        self.market_data = market_data
        self.sentiment_analyzer = sentiment_analyzer
        
        # Calculate sentiment scores
        self.sentiment_scores = self._calculate_sentiment_scores()
        
        # Calculate returns for different horizons
        self.returns = self._calculate_returns()
        
        # Calculate cumulative sentiment scores
        self.cumulative_scores = self._calculate_cumulative_scores()
        
    def _calculate_sentiment_scores(self):
        """Calculate sentiment scores from news"""
        sentiment_scores = []
        
        for _, row in tqdm(self.market_data.iterrows(), total=len(self.market_data), desc="Analyzing sentiment"):
            result = self.sentiment_analyzer.analyze_text(row['news'], row['date'])
            sentiment_scores.append(result)
            
        return pd.DataFrame(sentiment_scores)
    
    def _calculate_returns(self):
        """Calculate returns for different horizons"""
        returns = {}
        
        for market_name in [col.split('_')[0] for col in self.market_data.columns if col.endswith('_price')]:
            price_col = f'{market_name}_price'
            
            # Calculate returns for different horizons
            for period in range(1, 251):  # 1 to 250 days
                returns[f'{market_name}_{period}'] = self.market_data[price_col].pct_change(period).shift(-period)
        
        return pd.DataFrame(returns)
    
    def _calculate_cumulative_scores(self):
        """Calculate cumulative sentiment scores for different periods"""
        cumulative_scores = {}
        
        # For each period, calculate the cumulative score
        for period in range(1, 251):  # 1 to 250 days
            pos_sum = self.sentiment_scores['positive'].rolling(window=period).sum()
            neg_sum = self.sentiment_scores['negative'].rolling(window=period).sum()
            
            # Avoid division by zero
            denom = pos_sum + neg_sum
            denom = denom.replace(0, np.nan)
            
            cumulative_scores[f'S{period}'] = (pos_sum - neg_sum) / denom
        
        return pd.DataFrame(cumulative_scores)
    
    def analyze_correlations(self):
        """
        Analyze correlations between sentiment scores and returns
        
        Returns:
        --------
        tuple
            Pearson and Spearman correlation matrices
        """
        # Combine data
        combined_data = pd.concat([self.sentiment_scores['sentiment_score'], 
                                  self.cumulative_scores, 
                                  self.returns], axis=1)
        
        # Remove rows with NaN values
        combined_data = combined_data.dropna()
        
        # Calculate Pearson correlations
        pearson_corr = combined_data.corr(method='pearson')
        
        # Calculate Spearman correlations
        spearman_corr = combined_data.corr(method='spearman')
        
        return pearson_corr, spearman_corr
    
    def get_best_correlations(self, corr_matrix, market_name, top_n=5):
        """
        Get the best correlations for a specific market
        
        Parameters:
        -----------
        corr_matrix : pd.DataFrame
            The correlation matrix
        market_name : str
            The name of the market
        top_n : int
            Number of top correlations to return
            
        Returns:
        --------
        pd.DataFrame
            The top correlations
        """
        # Get correlations for the specified market
        market_cols = [col for col in corr_matrix.columns if col.startswith(market_name)]
        
        # Get sentiment score columns
        sentiment_cols = [col for col in corr_matrix.columns if col.startswith('S')]
        
        # Extract the correlations
        correlations = []
        
        for score_col in sentiment_cols:
            for market_col in market_cols:
                correlations.append({
                    'sentiment_score': score_col,
                    'market_return': market_col,
                    'correlation': corr_matrix.loc[score_col, market_col]
                })
        
        # Convert to DataFrame and sort
        corr_df = pd.DataFrame(correlations)
        
        # Get top positive correlations
        top_positive = corr_df.sort_values('correlation', ascending=False).head(top_n)
        
        # Get top negative correlations
        top_negative = corr_df.sort_values('correlation').head(top_n)
        
        return top_positive, top_negative
    
    def plot_correlation_matrix(self, corr_matrix, market_name, title):
        """
        Plot the correlation matrix for a specific market
        
        Parameters:
        -----------
        corr_matrix : pd.DataFrame
            The correlation matrix
        market_name : str
            The name of the market
        title : str
            The title of the plot
        """
        # Get relevant columns
        market_cols = [col for col in corr_matrix.columns if col.startswith(market_name)]
        sentiment_cols = [col for col in corr_matrix.columns if col.startswith('S')]
        
        # Extract the sub-matrix
        sub_matrix = corr_matrix.loc[sentiment_cols, market_cols]
        
        # Create a more readable version with renamed indices
        plot_matrix = sub_matrix.copy()
        plot_matrix.index = [f'Score_{i}' for i in range(1, len(sentiment_cols) + 1)]
        plot_matrix.columns = [f'Return_{i}' for i in range(1, len(market_cols) + 1)]
        
        # Plot the correlation matrix
        plt.figure(figsize=(12, 10))
        sns.heatmap(plot_matrix, cmap='RdBu_r', vmin=-0.6, vmax=0.6, center=0, annot=False)
        plt.title(f'{title} - {market_name}')
        plt.xlabel('Market Returns (different horizons)')
        plt.ylabel('Sentiment Scores (different periods)')
        plt.tight_layout()
        plt.savefig(f'{market_name}_{title.lower().replace(" ", "_")}.png')
        plt.close()
    
    def plot_optimal_period(self, corr_matrix, market_name):
        """
        Plot the optimal period analysis
        
        Parameters:
        -----------
        corr_matrix : pd.DataFrame
            The correlation matrix
        market_name : str
            The name of the market
        """
        # Get relevant columns
        market_cols = [col for col in corr_matrix.columns if col.startswith(market_name)]
        sentiment_cols = [col for col in corr_matrix.columns if col.startswith('S')]
        
        # Extract the sub-matrix
        sub_matrix = corr_matrix.loc[sentiment_cols, market_cols]
        
        # Calculate mean correlation for each cumulative score period
        mean_correlations = []
        
        for i, score_col in enumerate(sentiment_cols):
            # Get the period from the column name (S1, S2, etc.)
            period = int(score_col[1:])
            
            # Calculate mean correlation for the first month (20 trading days)
            month_cols = market_cols[:20]
            mean_corr = sub_matrix.loc[score_col, month_cols].mean()
            
            mean_correlations.append({
                'period': period,
                'mean_correlation': mean_corr
            })
        
        # Convert to DataFrame and sort
        mean_df = pd.DataFrame(mean_correlations)
        
        # Plot
        plt.figure(figsize=(10, 6))
        plt.plot(mean_df['period'], mean_df['mean_correlation'])
        plt.title(f'Mean Correlation vs. Cumulative Score Period - {market_name}')
        plt.xlabel('Cumulative Score Period (days)')
        plt.ylabel('Mean Correlation with 1-Month Returns')
        plt.grid(True, alpha=0.3)
        
        # Find and mark the optimal period
        optimal_period = mean_df.loc[mean_df['mean_correlation'].idxmax(), 'period']
        max_corr = mean_df['mean_correlation'].max()
        
        plt.scatter([optimal_period], [max_corr], color='red', s=100, zorder=5)
        plt.annotate(f'Optimal: {optimal_period} days',
                     xy=(optimal_period, max_corr),
                     xytext=(optimal_period + 20, max_corr),
                     arrowprops=dict(facecolor='black', shrink=0.05, width=1.5),
                     fontsize=12)
        
        plt.tight_layout()
        plt.savefig(f'{market_name}_optimal_period.png')
        plt.close()
        
        return optimal_period, max_corr
    
    def run_backtesting(self, market_name, sentiment_period, return_horizon):
        """
        Run a simple backtesting of the strategy
        
        Parameters:
        -----------
        market_name : str
            The name of the market
        sentiment_period : int
            The period for cumulative sentiment score
        return_horizon : int
            The horizon for returns
            
        Returns:
        --------
        pd.DataFrame
            Backtesting results
        """
        # Prepare data
        price_col = f'{market_name}_price'
        sentiment_col = f'S{sentiment_period}'
        
        # Calculate returns for the specified horizon
        returns = self.market_data[price_col].pct_change(return_horizon).shift(-return_horizon)
        
        # Get sentiment scores
        sentiment = self.cumulative_scores[sentiment_col]
        
        # Combine data
        backtest_data = pd.DataFrame({
            'date': self.market_data['date'],
            'price': self.market_data[price_col],
            'sentiment': sentiment,
            'future_return': returns
        })
        
        # Remove rows with NaN values
        backtest_data = backtest_data.dropna()
        
        # Generate signals
        backtest_data['signal'] = np.sign(backtest_data['sentiment'])
        
        # Calculate strategy returns
        backtest_data['strategy_return'] = backtest_data['signal'] * backtest_data['future_return']
        
        # Calculate cumulative returns
        backtest_data['cum_market_return'] = (1 + backtest_data['future_return']).cumprod() - 1
        backtest_data['cum_strategy_return'] = (1 + backtest_data['strategy_return']).cumprod() - 1
        
        # Calculate performance metrics
        total_days = len(backtest_data)
        win_days = sum(backtest_data['strategy_return'] > 0)
        loss_days = sum(backtest_data['strategy_return'] < 0)
        win_rate = win_days / total_days if total_days > 0 else 0
        
        avg_win = backtest_data.loc[backtest_data['strategy_return'] > 0, 'strategy_return'].mean() if win_days > 0 else 0
        avg_loss = backtest_data.loc[backtest_data['strategy_return'] < 0, 'strategy_return'].mean() if loss_days > 0 else 0
        
        profit_factor = (win_days * avg_win) / (loss_days * abs(avg_loss)) if loss_days * abs(avg_loss) > 0 else float('inf')
        
        annual_return = ((1 + backtest_data['cum_strategy_return'].iloc[-1]) ** (252 / total_days) - 1) if total_days > 0 else 0
        annual_volatility = backtest_data['strategy_return'].std() * np.sqrt(252)
        sharpe_ratio = annual_return / annual_volatility if annual_volatility > 0 else 0
        
        # Print performance summary
        print(f"\nBacktesting Results for {market_name}")
        print(f"Sentiment Period: {sentiment_period}, Return Horizon: {return_horizon}")
        print(f"Total Days: {total_days}")
        print(f"Win Rate: {win_rate:.2%}")
        print(f"Average Win: {avg_win:.2%}")
        print(f"Average Loss: {avg_loss:.2%}")
        print(f"Profit Factor: {profit_factor:.2f}")
        print(f"Annual Return: {annual_return:.2%}")
        print(f"Annual Volatility: {annual_volatility:.2%}")
        print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
        
        # Plot cumulative returns
        plt.figure(figsize=(12, 6))
        plt.plot(backtest_data['date'], backtest_data['cum_market_return'], label='Market', color='blue')
        plt.plot(backtest_data['date'], backtest_data['cum_strategy_return'], label='Strategy', color='green')
        plt.title(f'Cumulative Returns - {market_name}')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Return')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{market_name}_backtest.png')
        plt.close()
        
        return backtest_data

# Run the simulation
print("Simulating market data...")
market_simulator = MarketSimulator(start_date='2010-01-01', end_date='2023-10-31')
market_data = market_simulator.get_data()

print(f"Generated {len(market_data)} days of market data for 6 markets")

# Initialize sentiment analyzer and strategy
print("Analyzing sentiment and calculating strategy metrics...")
sentiment_analyzer = SentimentAnalyzer(use_gpt=False)
strategy = SentimentAnalysisStrategy(market_data, sentiment_analyzer)

# Analyze correlations
pearson_corr, spearman_corr = strategy.analyze_correlations()

# Plot correlation matrices for each market
for market_name in ['US', 'US_Tech', 'Japan', 'Europe', 'UK', 'Emerging']:
    print(f"\nAnalyzing {market_name} market")
    
    # Plot Pearson correlation matrix
    strategy.plot_correlation_matrix(pearson_corr, market_name, 'Pearson Correlation')
    
    # Plot Spearman correlation matrix
    strategy.plot_correlation_matrix(spearman_corr, market_name, 'Spearman Correlation')
    
    # Find best correlations
    top_positive, top_negative = strategy.get_best_correlations(pearson_corr, market_name, top_n=5)
    
    print(f"Top positive correlations for {market_name}:")
    print(top_positive)
    
    print(f"\nTop negative correlations for {market_name}:")
    print(top_negative)
    
    # Find optimal period
    optimal_period, max_corr = strategy.plot_optimal_period(pearson_corr, market_name)
    print(f"\nOptimal period for {market_name}: {optimal_period} days (correlation: {max_corr:.4f})")
    
    # Run backtesting
    backtest_data = strategy.run_backtesting(market_name, optimal_period, 20)

# Additional analysis for robustness
# Compare average correlation patterns across markets
market_names = ['US', 'US_Tech', 'Japan', 'Europe', 'UK', 'Emerging']
avg_corr_matrix = None

for market_name in market_names:
    market_cols = [col for col in pearson_corr.columns if col.startswith(market_name)]
    sentiment_cols = [col for col in pearson_corr.columns if col.startswith('S')]
    
    sub_matrix = pearson_corr.loc[sentiment_cols, market_cols]
    
    if avg_corr_matrix is None:
        avg_corr_matrix = sub_matrix / len(market_names)
    else:
        avg_corr_matrix += sub_matrix / len(market_names)

# Plot average correlation matrix
plt.figure(figsize=(12, 10))
sns.heatmap(avg_corr_matrix, cmap='RdBu_r', vmin=-0.6, vmax=0.6, center=0, annot=False)
plt.title('Average Correlation Matrix Across All Markets')
plt.xlabel('Market Returns (different horizons)')
plt.ylabel('Sentiment Scores (different periods)')
plt.tight_layout()
plt.savefig('average_correlation_matrix.png')
plt.close()

# Calculate percentage of matrix elements that follow the common pattern
pattern_match_percentages = {}

for market_name in market_names:
    market_cols = [col for col in pearson_corr.columns if col.startswith(market_name)]
    sentiment_cols = [col for col in pearson_corr.columns if col.startswith('S')]
    
    sub_matrix = pearson_corr.loc[sentiment_cols, market_cols]
    
    # Calculate z-scores
    z_scores = (sub_matrix - avg_corr_matrix) / avg_corr_matrix.std()
    
    # Count elements with p-value < 0.01 (z-score < 2.58)
    pattern_match = (abs(z_scores) < 2.58).sum().sum()
    total_elements = sub_matrix.size
    
    pattern_match_percentages[market_name] = pattern_match / total_elements * 100

print("\nPercentage of matrix elements following the common pattern:")
for market_name, percentage in pattern_match_percentages.items():
    print(f"{market_name}: {percentage:.2f}%")

print("\nSimulation and analysis complete. Results saved as image files.")