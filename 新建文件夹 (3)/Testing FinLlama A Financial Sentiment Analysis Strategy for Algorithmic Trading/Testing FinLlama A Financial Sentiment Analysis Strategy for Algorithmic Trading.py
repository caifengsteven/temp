import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import yfinance as yf
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

class SimulatedSentimentAnalyzer:
    """
    Simulates various sentiment analysis methods with different accuracy levels
    to represent the methods mentioned in the paper.
    """
    def __init__(self, method='finllama', accuracy=None):
        """
        Initialize the sentiment analyzer.
        
        Parameters:
        -----------
        method : str
            The sentiment analysis method to simulate ('lmd', 'hiv4', 'vader', 'finbert', 'finllama')
        accuracy : float
            The accuracy level to simulate (0.0 to 1.0). If None, uses default values based on method.
        """
        self.method = method.lower()
        
        # Default accuracy levels based on the paper's implied performance
        default_accuracies = {
            'lmd': 0.65,
            'hiv4': 0.55,
            'vader': 0.60,
            'finbert': 0.70,
            'finllama': 0.80
        }
        
        self.accuracy = accuracy if accuracy is not None else default_accuracies.get(self.method, 0.5)
        
        # Initialize VADER for actual sentiment analysis if available
        try:
            self.vader = SentimentIntensityAnalyzer()
        except:
            self.vader = None
            
        # Try to load FinBERT if available
        try:
            self.finbert_tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
            self.finbert_model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
            self.finbert_pipeline = pipeline("sentiment-analysis", model=self.finbert_model, tokenizer=self.finbert_tokenizer)
        except:
            self.finbert_pipeline = None
    
    def analyze(self, text):
        """
        Analyze the sentiment of a text.
        
        Parameters:
        -----------
        text : str
            The text to analyze
            
        Returns:
        --------
        sentiment : float
            The sentiment score (-1.0 to 1.0)
        """
        # Try to use actual sentiment analysis if available
        if self.method == 'vader' and self.vader is not None:
            return self.vader.polarity_scores(text)['compound']
        
        if self.method == 'finbert' and self.finbert_pipeline is not None:
            result = self.finbert_pipeline(text)[0]
            if result['label'] == 'positive':
                return result['score']
            elif result['label'] == 'negative':
                return -result['score']
            else:
                return 0.0
        
        # If actual sentiment analysis is not available, simulate based on accuracy
        # This is a simplification and doesn't represent the actual models
        
        # Generate a random sentiment base
        base_sentiment = np.random.normal(0, 0.5)
        
        # Words that might influence the sentiment direction
        positive_words = ['increase', 'profit', 'growth', 'success', 'improve', 'positive', 'up']
        negative_words = ['decrease', 'loss', 'decline', 'fail', 'worsen', 'negative', 'down']
        
        # Count positive and negative words
        pos_count = sum(1 for word in positive_words if word in text.lower())
        neg_count = sum(1 for word in negative_words if word in text.lower())
        
        # Determine the "true" sentiment direction
        true_direction = 1 if pos_count > neg_count else -1 if neg_count > pos_count else 0
        
        # Simulate accuracy - with probability (accuracy) we get the right direction
        if np.random.rand() < self.accuracy:
            direction = true_direction
        else:
            # With probability (1-accuracy) we get a random direction
            direction = np.random.choice([-1, 0, 1])
        
        # Generate sentiment score
        if direction == 1:
            # Positive sentiment
            sentiment = np.random.uniform(0.3, 1.0)
        elif direction == -1:
            # Negative sentiment
            sentiment = np.random.uniform(-1.0, -0.3)
        else:
            # Neutral sentiment
            sentiment = np.random.uniform(-0.3, 0.3)
        
        return sentiment


def generate_simulated_news_data(companies, start_date, end_date, news_frequency=0.3):
    """
    Generate simulated financial news data.
    
    Parameters:
    -----------
    companies : list
        List of company symbols
    start_date : str
        Start date in format 'YYYY-MM-DD'
    end_date : str
        End date in format 'YYYY-MM-DD'
    news_frequency : float
        Probability of a company having news on a given day (0.0 to 1.0)
        
    Returns:
    --------
    news_df : DataFrame
        DataFrame containing simulated news data
    """
    # Convert dates to datetime
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')
    
    # Generate date range (only business days)
    date_range = []
    current = start
    while current <= end:
        if current.weekday() < 5:  # Monday to Friday
            date_range.append(current)
        current += timedelta(days=1)
    
    # Create empty list for news data
    news_data = []
    
    # Common financial phrases
    positive_phrases = [
        "reported strong earnings", "exceeded expectations", "announced new partnership",
        "launched innovative product", "expanded market share", "improved profitability",
        "raised guidance", "beat analyst estimates", "successful cost-cutting measures",
        "strong growth in revenue"
    ]
    
    negative_phrases = [
        "missed earnings expectations", "announced layoffs", "facing regulatory challenges",
        "lost market share", "lowered guidance", "reported declining sales",
        "facing increased competition", "product recall announced", "profit margins shrinking",
        "unexpected CEO resignation"
    ]
    
    neutral_phrases = [
        "announced quarterly results", "held annual meeting", "appointed new director",
        "released financial statements", "updated business strategy", "announced stock split",
        "declared regular dividend", "scheduled investor conference", "filed annual report",
        "updated market on operations"
    ]
    
    for date in date_range:
        for company in companies:
            # Determine if company has news on this day
            if np.random.rand() < news_frequency:
                # Generate random number of news articles (1-3)
                num_articles = np.random.randint(1, 4)
                
                for _ in range(num_articles):
                    # Generate sentiment with some correlation to actual stock performance
                    sentiment_direction = np.random.choice([-1, 0, 1], p=[0.3, 0.4, 0.3])
                    
                    # Select phrases based on sentiment
                    if sentiment_direction == 1:
                        phrase = np.random.choice(positive_phrases)
                        headline = f"{company} {phrase}"
                    elif sentiment_direction == -1:
                        phrase = np.random.choice(negative_phrases)
                        headline = f"{company} {phrase}"
                    else:
                        phrase = np.random.choice(neutral_phrases)
                        headline = f"{company} {phrase}"
                    
                    # Add to news data
                    news_data.append({
                        'date': date,
                        'company': company,
                        'headline': headline,
                        'true_sentiment': sentiment_direction
                    })
    
    # Convert to DataFrame
    news_df = pd.DataFrame(news_data)
    return news_df


def get_stock_data(companies, start_date, end_date, use_real_data=False):
    """
    Get historical stock data (either real or simulated).
    
    Parameters:
    -----------
    companies : list
        List of company symbols
    start_date : str
        Start date in format 'YYYY-MM-DD'
    end_date : str
        End date in format 'YYYY-MM-DD'
    use_real_data : bool
        Whether to use real data from Yahoo Finance
        
    Returns:
    --------
    stock_data : DataFrame
        DataFrame containing stock data with dates as index and companies as columns
    """
    if use_real_data:
        # Try to get real data from Yahoo Finance
        try:
            data = yf.download(companies, start=start_date, end=end_date)['Adj Close']
            # Calculate daily returns
            returns = data.pct_change().dropna()
            return returns
        except:
            print("Failed to get real data. Falling back to simulated data.")
            use_real_data = False
    
    if not use_real_data:
        # Generate simulated stock data
        # Convert dates to datetime
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')
        
        # Generate date range (only business days)
        date_range = []
        current = start
        while current <= end:
            if current.weekday() < 5:  # Monday to Friday
                date_range.append(current)
            current += timedelta(days=1)
        
        # Create DataFrame with dates as index
        returns_df = pd.DataFrame(index=date_range)
        
        # Add returns for each company
        for company in companies:
            # Generate random daily returns with some autocorrelation
            # Mean around 0.0005 (approximately 12% annual return)
            mean_return = 0.0005
            std_return = 0.015  # Daily volatility
            
            # Generate base random returns
            random_returns = np.random.normal(mean_return, std_return, len(date_range))
            
            # Add autocorrelation
            returns = np.zeros(len(date_range))
            returns[0] = random_returns[0]
            for i in range(1, len(date_range)):
                returns[i] = 0.1 * returns[i-1] + 0.9 * random_returns[i]
            
            # Add to DataFrame
            returns_df[company] = returns
        
        return returns_df


def analyze_news_sentiment(news_df, sentiment_analyzers):
    """
    Analyze sentiment of news articles using various methods.
    
    Parameters:
    -----------
    news_df : DataFrame
        DataFrame containing news data
    sentiment_analyzers : dict
        Dictionary of sentiment analyzers
        
    Returns:
    --------
    sentiment_df : DataFrame
        DataFrame containing sentiment scores for each news article and method
    """
    # Create copy of news DataFrame
    sentiment_df = news_df.copy()
    
    # Add columns for each sentiment method
    for method, analyzer in sentiment_analyzers.items():
        sentiment_df[method] = sentiment_df['headline'].apply(lambda x: analyzer.analyze(x))
    
    return sentiment_df


def aggregate_daily_sentiment(sentiment_df, companies):
    """
    Aggregate sentiment scores by company and date.
    
    Parameters:
    -----------
    sentiment_df : DataFrame
        DataFrame containing sentiment scores for each news article
    companies : list
        List of company symbols
        
    Returns:
    --------
    daily_sentiment : dict
        Dictionary of DataFrames containing daily sentiment for each method
    """
    # Get unique methods (excluding 'true_sentiment')
    methods = [col for col in sentiment_df.columns if col not in ['date', 'company', 'headline', 'true_sentiment']]
    
    # Create dictionary to store results
    daily_sentiment = {}
    
    # Aggregate for each method
    for method in methods:
        # Group by date and company, and calculate mean sentiment
        grouped = sentiment_df.groupby(['date', 'company'])[method].mean().reset_index()
        
        # Pivot to get companies as columns
        pivoted = grouped.pivot(index='date', columns='company', values=method)
        
        # Store in dictionary
        daily_sentiment[method] = pivoted
    
    return daily_sentiment


def construct_long_short_portfolio(returns_df, sentiment_df, long_short_percent=0.35):
    """
    Construct a long-short portfolio based on sentiment scores.
    
    Parameters:
    -----------
    returns_df : DataFrame
        DataFrame containing stock returns
    sentiment_df : DataFrame
        DataFrame containing daily sentiment scores
    long_short_percent : float
        Percentage of companies to go long/short on
        
    Returns:
    --------
    portfolio_returns : Series
        Series containing daily portfolio returns
    """
    # Align dates
    common_dates = sentiment_df.index.intersection(returns_df.index)
    sentiment_aligned = sentiment_df.loc[common_dates]
    returns_aligned = returns_df.loc[common_dates]
    
    # Initialize portfolio returns
    portfolio_returns = pd.Series(index=common_dates, dtype=float)
    
    # For each day, construct portfolio
    for date in common_dates:
        # Get sentiment for current day
        current_sentiment = sentiment_aligned.loc[date].dropna()
        
        if len(current_sentiment) == 0:
            # No sentiment data for this day
            portfolio_returns[date] = 0
            continue
        
        # Rank companies by sentiment
        ranked_companies = current_sentiment.sort_values(ascending=False)
        
        # Determine number of companies for long/short positions
        num_positions = int(len(ranked_companies) * long_short_percent)
        
        if num_positions == 0:
            # Not enough companies
            portfolio_returns[date] = 0
            continue
        
        # Select top companies for long position
        long_companies = ranked_companies.index[:num_positions]
        
        # Select bottom companies for short position
        short_companies = ranked_companies.index[-num_positions:]
        
        # Get returns for long and short positions
        long_returns = returns_aligned.loc[date, long_companies].mean()
        short_returns = returns_aligned.loc[date, short_companies].mean()
        
        # Calculate portfolio return (long - short)
        portfolio_returns[date] = long_returns - short_returns
    
    return portfolio_returns


def calculate_performance_metrics(returns_series):
    """
    Calculate performance metrics for a returns series.
    
    Parameters:
    -----------
    returns_series : Series
        Series containing daily returns
        
    Returns:
    --------
    metrics : dict
        Dictionary containing performance metrics
    """
    # Calculate cumulative returns
    cumulative_returns = (1 + returns_series).cumprod() - 1
    
    # Calculate log returns for annualized metrics
    log_returns = np.log(1 + returns_series)
    
    # Calculate annualized return
    annualized_return = log_returns.mean() * 252
    
    # Calculate annualized volatility
    annualized_volatility = log_returns.std() * np.sqrt(252)
    
    # Calculate Sharpe ratio (assuming 0% risk-free rate)
    sharpe_ratio = annualized_return / annualized_volatility
    
    # Calculate maximum drawdown
    cum_returns = (1 + returns_series).cumprod()
    running_max = cum_returns.cummax()
    drawdown = (cum_returns / running_max) - 1
    max_drawdown = drawdown.min()
    
    return {
        'cumulative_returns': cumulative_returns.iloc[-1],
        'annualized_return': annualized_return,
        'annualized_volatility': annualized_volatility,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'cumulative_returns_series': cumulative_returns
    }


def plot_portfolio_performance(portfolio_returns, method_names):
    """
    Plot portfolio performance.
    
    Parameters:
    -----------
    portfolio_returns : dict
        Dictionary of returns series for each method
    method_names : list
        List of method names
    """
    # Calculate cumulative returns for each method
    cumulative_returns = {}
    for method in method_names:
        cumulative_returns[method] = (1 + portfolio_returns[method]).cumprod() - 1
    
    # Convert to DataFrame for plotting
    cum_returns_df = pd.DataFrame(cumulative_returns)
    
    # Create figure with subplots
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot cumulative returns
    cum_returns_df.plot(ax=axs[0, 0])
    axs[0, 0].set_title('Cumulative Returns')
    axs[0, 0].set_ylabel('Return')
    axs[0, 0].grid(True)
    axs[0, 0].legend()
    
    # Calculate and plot 30-day moving average of returns
    ma_returns = {}
    for method in method_names:
        ma_returns[method] = portfolio_returns[method].rolling(30).mean()
    
    ma_returns_df = pd.DataFrame(ma_returns)
    ma_returns_df.plot(ax=axs[0, 1])
    axs[0, 1].set_title('30-Day Moving Average Returns')
    axs[0, 1].set_ylabel('Return')
    axs[0, 1].grid(True)
    axs[0, 1].legend()
    
    # Calculate and plot 30-day moving standard deviation of returns
    mstd_returns = {}
    for method in method_names:
        mstd_returns[method] = portfolio_returns[method].rolling(30).std()
    
    mstd_returns_df = pd.DataFrame(mstd_returns)
    mstd_returns_df.plot(ax=axs[1, 0])
    axs[1, 0].set_title('30-Day Moving Standard Deviation')
    axs[1, 0].set_ylabel('Standard Deviation')
    axs[1, 0].grid(True)
    axs[1, 0].legend()
    
    # Calculate and plot drawdowns
    drawdowns = {}
    for method in method_names:
        cum_returns = (1 + portfolio_returns[method]).cumprod()
        running_max = cum_returns.cummax()
        drawdowns[method] = (cum_returns / running_max) - 1
    
    drawdowns_df = pd.DataFrame(drawdowns)
    drawdowns_df.plot(ax=axs[1, 1])
    axs[1, 1].set_title('Drawdowns')
    axs[1, 1].set_ylabel('Drawdown')
    axs[1, 1].grid(True)
    axs[1, 1].legend()
    
    plt.tight_layout()
    plt.show()
    
    # Create a table of performance metrics
    metrics = {}
    for method in method_names:
        metrics[method] = calculate_performance_metrics(portfolio_returns[method])
    
    metrics_df = pd.DataFrame({
        method: {
            'Cumulative Returns (%)': metrics[method]['cumulative_returns'] * 100,
            'Annualized Return (%)': metrics[method]['annualized_return'] * 100,
            'Annualized Volatility (%)': metrics[method]['annualized_volatility'] * 100,
            'Sharpe Ratio': metrics[method]['sharpe_ratio'],
            'Max Drawdown (%)': metrics[method]['max_drawdown'] * 100
        } for method in method_names
    })
    
    # Add S&P 500 benchmark if available
    if 'S&P 500' in portfolio_returns:
        method_names.append('S&P 500')
        metrics['S&P 500'] = calculate_performance_metrics(portfolio_returns['S&P 500'])
        metrics_df['S&P 500'] = {
            'Cumulative Returns (%)': metrics['S&P 500']['cumulative_returns'] * 100,
            'Annualized Return (%)': metrics['S&P 500']['annualized_return'] * 100,
            'Annualized Volatility (%)': metrics['S&P 500']['annualized_volatility'] * 100,
            'Sharpe Ratio': metrics['S&P 500']['sharpe_ratio'],
            'Max Drawdown (%)': metrics['S&P 500']['max_drawdown'] * 100
        }
    
    # Sort by Sharpe Ratio (descending)
    metrics_df = metrics_df.sort_values(by='Sharpe Ratio', axis=1, ascending=False)
    
    # Plot metrics as a bar chart
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    
    # Cumulative Returns
    metrics_df.loc['Cumulative Returns (%)'].plot(kind='bar', ax=axs[0, 0])
    axs[0, 0].set_title('Cumulative Returns (%)')
    axs[0, 0].set_ylabel('Return (%)')
    axs[0, 0].grid(True)
    
    # Annualized Return
    metrics_df.loc['Annualized Return (%)'].plot(kind='bar', ax=axs[0, 1])
    axs[0, 1].set_title('Annualized Return (%)')
    axs[0, 1].set_ylabel('Return (%)')
    axs[0, 1].grid(True)
    
    # Sharpe Ratio
    metrics_df.loc['Sharpe Ratio'].plot(kind='bar', ax=axs[1, 0])
    axs[1, 0].set_title('Sharpe Ratio')
    axs[1, 0].set_ylabel('Ratio')
    axs[1, 0].grid(True)
    
    # Annualized Volatility
    metrics_df.loc['Annualized Volatility (%)'].plot(kind='bar', ax=axs[1, 1])
    axs[1, 1].set_title('Annualized Volatility (%)')
    axs[1, 1].set_ylabel('Volatility (%)')
    axs[1, 1].grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Print metrics table
    print("Performance Metrics:")
    print(metrics_df.round(2))
    
    return metrics_df


def main():
    # Define parameters
    start_date = '2015-02-01'
    end_date = '2021-06-30'
    companies = [
        'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'TSLA', 'NVDA', 'JPM', 'BAC', 'WFC',
        'C', 'GS', 'MS', 'BLK', 'PG', 'JNJ', 'KO', 'PEP', 'WMT', 'DIS', 'HD', 'UNH',
        'V', 'MA', 'PYPL', 'INTC', 'AMD', 'CSCO', 'IBM', 'ORCL', 'CRM', 'ADBE', 'MCD',
        'NKE', 'SBUX', 'BA', 'CAT', 'GE', 'MMM', 'HON', 'UNP', 'UPS', 'FDX', 'T', 'VZ',
        'CMCSA', 'NFLX', 'COST', 'TGT', 'MRK', 'PFE', 'ABT', 'BMY', 'ABBV', 'TMO', 'DHR',
        'XOM', 'CVX', 'COP', 'EOG', 'SLB', 'PXD', 'OXY'
    ]
    
    # Generate simulated news data
    print("Generating simulated news data...")
    news_df = generate_simulated_news_data(companies, start_date, end_date)
    
    # Get stock returns data (try real data first, fall back to simulated)
    print("Getting stock returns data...")
    returns_df = get_stock_data(companies, start_date, end_date, use_real_data=True)
    
    # Initialize sentiment analyzers
    print("Initializing sentiment analyzers...")
    sentiment_analyzers = {
        'lmd': SimulatedSentimentAnalyzer(method='lmd'),
        'hiv4': SimulatedSentimentAnalyzer(method='hiv4'),
        'vader': SimulatedSentimentAnalyzer(method='vader'),
        'finbert': SimulatedSentimentAnalyzer(method='finbert'),
        'finllama': SimulatedSentimentAnalyzer(method='finllama')
    }
    
    # Analyze sentiment
    print("Analyzing sentiment...")
    sentiment_df = analyze_news_sentiment(news_df, sentiment_analyzers)
    
    # Aggregate daily sentiment
    print("Aggregating daily sentiment...")
    daily_sentiment = aggregate_daily_sentiment(sentiment_df, companies)
    
    # Construct long-short portfolios
    print("Constructing long-short portfolios...")
    portfolio_returns = {}
    for method, sentiment in daily_sentiment.items():
        portfolio_returns[method] = construct_long_short_portfolio(returns_df, sentiment)
    
    # Add S&P 500 benchmark (if using real data)
    try:
        sp500 = yf.download('^GSPC', start=start_date, end=end_date)['Adj Close']
        portfolio_returns['S&P 500'] = sp500.pct_change().dropna()
    except:
        print("Failed to get S&P 500 data.")
    
    # Plot portfolio performance
    print("Plotting portfolio performance...")
    method_names = list(daily_sentiment.keys())
    metrics_df = plot_portfolio_performance(portfolio_returns, method_names)
    
    return portfolio_returns, metrics_df


if __name__ == "__main__":
    portfolio_returns, metrics_df = main()