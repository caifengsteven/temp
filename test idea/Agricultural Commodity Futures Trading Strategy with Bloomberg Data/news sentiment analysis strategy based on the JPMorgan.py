import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.feature_selection import chi2, SelectKBest
import nltk
from nltk.corpus import stopwords
import re
import string
import warnings
warnings.filterwarnings('ignore')

# Download NLTK resources
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

# Set random seed for reproducibility
np.random.seed(42)

# Parameters for simulation
num_stocks = 100
num_days = 500
start_date = datetime(2020, 1, 1)
dates = [start_date + timedelta(days=i) for i in range(num_days)]

# Function to preprocess text
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation
    text = re.sub(f'[{string.punctuation}]', ' ', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Function to handle negation in text
def handle_negation(text):
    negation_words = ['not', 'no', 'never', 'none', 'nobody', 'nothing', 'neither', 'nor', 'nowhere', 'hardly', 'barely', 'scarcely', 'doesnt', 'isnt', 'wasnt', 'shouldnt', 'wouldnt', 'couldnt', 'wont', 'cant', 'dont']
    words = text.split()
    result = []
    negate = False
    
    for word in words:
        if word in negation_words:
            negate = True
            result.append(word)
        elif negate and word not in string.punctuation:
            result.append('neg_' + word)
        else:
            result.append(word)
            negate = False
    
    return ' '.join(result)

# Generate simulated analyst reports
def generate_analyst_reports(num_reports=10000):
    # Create positive phrases typical in analyst reports
    positive_phrases = [
        "strong performance", "outperform expectations", "beat estimates", 
        "robust growth", "positive outlook", "attractive valuation", 
        "increasing market share", "excellent execution", "strong demand",
        "raising price target", "upgrade to overweight", "buy opportunity",
        "bullish outlook", "margin expansion", "impressive results",
        "high quality", "strategic acquisition", "value creation",
        "well positioned", "market leader", "competitive advantage"
    ]
    
    # Create negative phrases typical in analyst reports
    negative_phrases = [
        "weak performance", "underperform expectations", "miss estimates",
        "declining growth", "negative outlook", "unattractive valuation",
        "losing market share", "poor execution", "weak demand",
        "lowering price target", "downgrade to underweight", "sell opportunity",
        "bearish outlook", "margin contraction", "disappointing results",
        "low quality", "dilutive acquisition", "value destruction",
        "poorly positioned", "market laggard", "competitive disadvantage"
    ]
    
    # Generate text templates for reports
    templates = [
        "We {rating} {company} based on {reason1} and {reason2}. The {aspect} is {assessment}.",
        "{company} reported {quarter} results that {performance}. We {action} our {rating} rating with a price target of ${price}.",
        "We believe {company} is {position} due to {reason1}. The {aspect} {outlook}.",
        "Our analysis indicates {company} will {performance} in the next {timeframe}. {reason1} and {reason2} support our {rating} view.",
        "{company}'s {product} shows {assessment} which {impact} our {metric}. We {rating} with a {timeframe} price target of ${price}."
    ]
    
    # Generate filler content
    companies = [f"Company_{i}" for i in range(1, 101)]
    quarters = ["Q1", "Q2", "Q3", "Q4", "annual", "semi-annual", "quarterly"]
    timeframes = ["12 months", "year", "quarter", "near term", "long term", "6 months"]
    aspects = ["revenue growth", "margin", "cash flow", "earnings outlook", "market position", "product pipeline", "cost structure"]
    metrics = ["EPS estimate", "revenue forecast", "price target", "margin expectation", "EBITDA projection"]
    products = ["product line", "service offering", "platform", "solution", "technology", "division", "segment"]
    
    reports = []
    ratings = []
    
    for _ in range(num_reports):
        # Determine if this is a positive or negative report
        is_positive = np.random.random() > 0.4  # 60% positive, 40% negative
        
        if is_positive:
            rating_choices = ["overweight", "OW", "buy", "maintain overweight", "upgrade to overweight", "reiterate overweight"]
            performance_choices = ["beat expectations", "exceeded estimates", "showed strong growth", "demonstrated robust performance"]
            action_choices = ["raise", "maintain", "reiterate"]
            position_choices = ["well positioned", "a market leader", "gaining market share", "strategically advantaged"]
            outlook_choices = ["shows positive momentum", "is expected to improve", "demonstrates strength", "should outperform"]
            impact_choices = ["positively impacts", "supports", "enhances", "strengthens"]
            assessment_choices = ["strong performance", "impressive growth", "positive momentum", "excellent execution"]
            reasons = positive_phrases
            ratings.append(1)  # Positive class
        else:
            rating_choices = ["underweight", "UW", "sell", "maintain underweight", "downgrade to underweight", "reiterate underweight"]
            performance_choices = ["missed expectations", "fell short of estimates", "showed weak growth", "demonstrated poor performance"]
            action_choices = ["lower", "cut", "reduce"]
            position_choices = ["poorly positioned", "a market laggard", "losing market share", "strategically disadvantaged"]
            outlook_choices = ["shows negative momentum", "is expected to deteriorate", "demonstrates weakness", "should underperform"]
            impact_choices = ["negatively impacts", "undermines", "weakens", "reduces"]
            assessment_choices = ["weak performance", "disappointing growth", "negative momentum", "poor execution"]
            reasons = negative_phrases
            ratings.append(0)  # Negative class
        
        # Randomly select a template and fill it
        template = np.random.choice(templates)
        report = template.format(
            company=np.random.choice(companies),
            rating=np.random.choice(rating_choices),
            reason1=np.random.choice(reasons),
            reason2=np.random.choice(reasons),
            aspect=np.random.choice(aspects),
            assessment=np.random.choice(assessment_choices),
            quarter=np.random.choice(quarters),
            performance=np.random.choice(performance_choices),
            action=np.random.choice(action_choices),
            price=np.random.randint(10, 500),
            position=np.random.choice(position_choices),
            outlook=np.random.choice(outlook_choices),
            timeframe=np.random.choice(timeframes),
            product=np.random.choice(products),
            impact=np.random.choice(impact_choices),
            metric=np.random.choice(metrics)
        )
        
        reports.append(report)
    
    return pd.DataFrame({'text': reports, 'sentiment': ratings})

# Generate simulated news articles
def generate_news_articles(num_articles=5000):
    # Create positive phrases for news
    positive_phrases = [
        "shares surge", "stock rallies", "earnings beat", "raises outlook", 
        "positive results", "exceeded expectations", "strong performance", 
        "bullish forecast", "upgraded by analysts", "new partnership",
        "innovative product", "market share gains", "cost cutting success",
        "successful launch", "expanding margins", "dividend increase",
        "positive guidance", "record profits", "strong demand", "beats consensus"
    ]
    
    # Create negative phrases for news
    negative_phrases = [
        "shares plunge", "stock drops", "earnings miss", "lowers outlook", 
        "disappointing results", "fell short of expectations", "weak performance", 
        "bearish forecast", "downgraded by analysts", "ending partnership",
        "product recall", "market share loss", "cost cutting challenges",
        "delayed launch", "contracting margins", "dividend cut",
        "negative guidance", "profit warning", "weak demand", "misses consensus"
    ]
    
    # Generate text templates for news articles
    templates = [
        "{company} {news_event}: {detail}",
        "Breaking: {company} {news_event}",
        "{company} {news_event} amid {market_condition}",
        "Analysts {analyst_action} on {company} after {news_event}",
        "{company} shares {price_action} following {news_event}",
        "{company} reports {news_event}, {price_action}",
        "{sector} stocks: {company} {news_event}",
        "Market update: {company} {news_event}"
    ]
    
    companies = [f"Company_{i}" for i in range(1, 101)]
    sectors = ["Tech", "Finance", "Healthcare", "Energy", "Consumer", "Industrial", "Materials", "Utilities", "Real Estate", "Telecom"]
    market_conditions = ["market volatility", "sector rotation", "economic uncertainty", "recovery hopes", "recession fears", "inflationary pressures", "strong earnings season", "weak economic data"]
    analyst_actions = ["raise targets", "cut forecasts", "turn bullish", "express concerns", "upgrade ratings", "downgrade ratings", "remain cautious", "become optimistic"]
    
    articles = []
    sentiments = []
    stock_ids = []
    dates = []
    
    for day in range(num_days):
        # Generate between 5 and 15 news articles per day
        daily_articles = np.random.randint(5, 15)
        current_date = start_date + timedelta(days=day)
        
        for _ in range(daily_articles):
            # Select a random stock
            stock_id = np.random.randint(0, num_stocks)
            
            # Determine if this is a positive or negative article
            is_positive = np.random.random() > 0.45  # 55% positive, 45% negative
            
            if is_positive:
                news_event_choices = positive_phrases
                price_action_choices = ["jump", "surge", "rally", "rise", "climb", "advance", "gain"]
                sentiments.append(1)  # Positive class
            else:
                news_event_choices = negative_phrases
                price_action_choices = ["drop", "plunge", "fall", "decline", "slip", "retreat", "tumble"]
                sentiments.append(0)  # Negative class
            
            # Randomly select a template and fill it
            template = np.random.choice(templates)
            detail = np.random.choice(news_event_choices)
            article = template.format(
                company=f"Company_{stock_id+1}",
                news_event=np.random.choice(news_event_choices),
                detail=detail,
                market_condition=np.random.choice(market_conditions),
                analyst_action=np.random.choice(analyst_actions),
                price_action=np.random.choice(price_action_choices),
                sector=np.random.choice(sectors)
            )
            
            articles.append(article)
            stock_ids.append(stock_id)
            dates.append(current_date)
    
    return pd.DataFrame({
        'text': articles, 
        'sentiment': sentiments, 
        'stock_id': stock_ids, 
        'date': dates
    })

# Generate stock returns data
def generate_stock_returns(news_df):
    # Create a DataFrame to store stock returns
    returns_df = pd.DataFrame(index=dates, columns=[f'Company_{i+1}' for i in range(num_stocks)])
    
    # Base market return parameters
    market_mu = 0.0005  # daily mean return (about 12% annual)
    market_sigma = 0.01  # daily volatility
    
    # Generate market returns
    market_returns = np.random.normal(market_mu, market_sigma, num_days)
    
    # Stock specific parameters
    stock_betas = np.random.uniform(0.5, 1.5, num_stocks)
    stock_specific_vol = np.random.uniform(0.01, 0.03, num_stocks)
    
    # Sentiment impact parameters
    sentiment_impact_mu = 0.003  # mean impact of positive sentiment
    sentiment_impact_sigma = 0.001  # variation in impact
    
    # Dictionary to track accumulated sentiment for each stock on each day
    sentiment_tracker = {i: {date: {'count_pos': 0, 'count_neg': 0} 
                            for date in dates} 
                        for i in range(num_stocks)}
    
    # Populate sentiment tracker
    for _, row in news_df.iterrows():
        date = row['date']
        stock_id = row['stock_id']
        sentiment = row['sentiment']
        
        if sentiment == 1:
            sentiment_tracker[stock_id][date]['count_pos'] += 1
        else:
            sentiment_tracker[stock_id][date]['count_neg'] += 1
    
    # Generate stock returns
    for day in range(num_days):
        date = dates[day]
        market_return = market_returns[day]
        
        for stock_id in range(num_stocks):
            # Market component
            stock_return = stock_betas[stock_id] * market_return
            
            # Add stock-specific randomness
            stock_return += np.random.normal(0, stock_specific_vol[stock_id])
            
            # Add sentiment effect (with some delay/momentum)
            sentiment_effect = 0
            
            # Look back up to 5 days for sentiment effect with decay
            for lag in range(min(5, day+1)):
                prev_date = dates[day-lag]
                pos_count = sentiment_tracker[stock_id][prev_date]['count_pos']
                neg_count = sentiment_tracker[stock_id][prev_date]['count_neg']
                
                # Compute net sentiment effect
                if pos_count > 0 or neg_count > 0:
                    net_sentiment = (pos_count - neg_count) / (pos_count + neg_count)
                    # Apply decay based on lag
                    decay_factor = np.exp(-lag * 0.5)
                    # Add some randomness to sentiment impact
                    impact = np.random.normal(sentiment_impact_mu, sentiment_impact_sigma)
                    sentiment_effect += net_sentiment * impact * decay_factor
            
            # Add sentiment effect to return
            stock_return += sentiment_effect
            
            # Store the return
            returns_df.iloc[day, stock_id] = stock_return
    
    return returns_df

# Build the sentiment classifier pipeline
def build_sentiment_classifier(reports_df):
    # Preprocess text
    reports_df['processed_text'] = reports_df['text'].apply(preprocess_text)
    reports_df['processed_text'] = reports_df['processed_text'].apply(handle_negation)
    
    # Split into training and testing
    X_train, X_test, y_train, y_test = train_test_split(
        reports_df['processed_text'], 
        reports_df['sentiment'], 
        test_size=0.3,
        random_state=42,
        stratify=reports_df['sentiment']
    )
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    
    # First create the vectorizer to determine the number of features
    vectorizer = CountVectorizer(
        stop_words=stop_words,
        ngram_range=(1, 2),  # Unigrams and bigrams
        min_df=5             # Minimum document frequency
    )
    
    X_train_counts = vectorizer.fit_transform(X_train)
    n_features = X_train_counts.shape[1]
    
    # Determine k for SelectKBest (use all features if less than 5000)
    k = min(n_features, 5000)
    print(f"Number of features: {n_features}, using k={k} for feature selection")
    
    # Build pipeline with feature selection
    pipeline = Pipeline([
        ('vect', CountVectorizer(
            stop_words=stop_words,
            ngram_range=(1, 2),  # Unigrams and bigrams
            min_df=5             # Minimum document frequency
        )),
        ('chi2', SelectKBest(chi2, k=k)),  # Select top k features
        ('tfidf', TfidfTransformer()),
        ('clf', LogisticRegression(
            C=1.0,
            class_weight='balanced',
            max_iter=1000,
            random_state=42
        ))
    ])
    
    # Train the model
    pipeline.fit(X_train, y_train)
    
    # Evaluate on test set
    y_pred = pipeline.predict(X_test)
    
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    print("\nF1 Score:", f1_score(y_test, y_pred))
    
    # Print confusion matrix using plain matplotlib
    cm = confusion_matrix(y_test, y_pred)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['Negative', 'Positive'], rotation=45)
    plt.yticks(tick_marks, ['Negative', 'Positive'])
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
    
    return pipeline

# Apply sentiment model to news and implement trading strategy
def analyze_news_sentiment(classifier, news_df, returns_df):
    # Preprocess news text
    news_df['processed_text'] = news_df['text'].apply(preprocess_text)
    news_df['processed_text'] = news_df['processed_text'].apply(handle_negation)
    
    # Apply classifier to get predicted sentiment
    news_df['predicted_sentiment'] = classifier.predict(news_df['processed_text'])
    news_df['sentiment_score'] = classifier.predict_proba(news_df['processed_text'])[:, 1]  # Probability of positive class
    
    # Group by date and company to get weekly sentiment
    news_df['week'] = news_df['date'].dt.isocalendar().week
    news_df['year'] = news_df['date'].dt.isocalendar().year
    
    # Create a unique week identifier
    news_df['week_id'] = news_df['year'].astype(str) + '-' + news_df['week'].astype(str)
    
    # Aggregate sentiment by week and company
    weekly_sentiment = news_df.groupby(['week_id', 'stock_id'])['sentiment_score'].mean().reset_index()
    
    # Convert stock_id to company name
    weekly_sentiment['company'] = weekly_sentiment['stock_id'].apply(lambda x: f"Company_{x+1}")
    
    # Prepare to store strategy returns
    strategy_returns = pd.DataFrame(index=returns_df.index)
    
    # Add a column for market returns (equal-weighted portfolio)
    strategy_returns['market'] = returns_df.mean(axis=1)
    
    # Implement different trading strategies with various thresholds
    percentiles = [(50, 50), (60, 40), (70, 30), (80, 20), (90, 10)]
    
    for long_pctl, short_pctl in percentiles:
        strategy_name = f'ls_{long_pctl}_{short_pctl}'
        strategy_returns[strategy_name] = 0.0
    
    # Dictionary to store positions
    current_positions = {p: {} for p in [f'ls_{l}_{s}' for l, s in percentiles]}
    
    # Iterate through each week
    unique_weeks = news_df['week_id'].unique()
    
    for i, week_id in enumerate(unique_weeks):
        if i == len(unique_weeks) - 1:  # Skip the last week
            continue
            
        next_week_id = unique_weeks[i + 1]
        
        # Get sentiment for current week
        current_week = weekly_sentiment[weekly_sentiment['week_id'] == week_id]
        
        # Skip if not enough companies
        if len(current_week) < 10:
            continue
        
        # Calculate percentiles for each strategy
        for long_pctl, short_pctl in percentiles:
            strategy = f'ls_{long_pctl}_{short_pctl}'
            
            # Calculate thresholds
            high_threshold = np.percentile(current_week['sentiment_score'], 100 - long_pctl)
            low_threshold = np.percentile(current_week['sentiment_score'], short_pctl)
            
            # Select companies for long and short positions
            long_companies = current_week[current_week['sentiment_score'] >= high_threshold]['company'].tolist()
            short_companies = current_week[current_week['sentiment_score'] <= low_threshold]['company'].tolist()
            
            # Update positions for this strategy
            current_positions[strategy] = {
                'long': long_companies,
                'short': short_companies
            }
        
        # Find all dates in the next week
        next_week_dates = news_df[news_df['week_id'] == next_week_id]['date'].unique()
        
        # Apply strategy returns for each date in the next week
        for date in next_week_dates:
            if date in returns_df.index:
                for strategy, positions in current_positions.items():
                    long_pos = positions.get('long', [])
                    short_pos = positions.get('short', [])
                    
                    # Calculate strategy return for this day
                    long_return = returns_df.loc[date, long_pos].mean() if long_pos else 0
                    short_return = -returns_df.loc[date, short_pos].mean() if short_pos else 0
                    
                    if long_pos and short_pos:
                        # Equal weight to long and short sides
                        strategy_returns.loc[date, strategy] = (long_return + short_return) / 2
                    elif long_pos:
                        strategy_returns.loc[date, strategy] = long_return / 2
                    elif short_pos:
                        strategy_returns.loc[date, strategy] = short_return / 2
    
    # Calculate cumulative returns for each strategy
    cum_returns = (1 + strategy_returns).cumprod()
    
    return strategy_returns, cum_returns, news_df

# Evaluate the trading strategy
def evaluate_strategy(strategy_returns):
    # Calculate performance metrics
    perf_metrics = pd.DataFrame(index=strategy_returns.columns)
    
    # Annualized return (252 trading days)
    perf_metrics['Ann_Return'] = strategy_returns.mean() * 252
    
    # Annualized volatility
    perf_metrics['Ann_Volatility'] = strategy_returns.std() * np.sqrt(252)
    
    # Sharpe ratio (assuming 0 risk-free rate for simplicity)
    perf_metrics['Sharpe_Ratio'] = perf_metrics['Ann_Return'] / perf_metrics['Ann_Volatility']
    
    # Maximum drawdown
    cum_returns = (1 + strategy_returns).cumprod()
    running_max = cum_returns.cummax()
    drawdown = (cum_returns / running_max) - 1
    perf_metrics['Max_Drawdown'] = drawdown.min()
    
    # Win rate (percentage of positive return periods)
    perf_metrics['Win_Rate'] = (strategy_returns > 0).mean()
    
    # t-statistic
    t_stat = strategy_returns.mean() / (strategy_returns.std() / np.sqrt(len(strategy_returns)))
    perf_metrics['t_statistic'] = t_stat
    
    # Turnover (approximate based on position changes)
    perf_metrics['Turnover'] = 0.5  # Placeholder for weekly turnover
    
    # Breakeven transaction cost (bps)
    perf_metrics['Breakeven_Cost_bps'] = perf_metrics['Ann_Return'] * 10000 / perf_metrics['Turnover'] / 252
    
    return perf_metrics

# Main execution flow
def main():
    print("Generating analyst reports for training...")
    reports_df = generate_analyst_reports(num_reports=10000)
    
    print("\nTraining sentiment classifier...")
    sentiment_classifier = build_sentiment_classifier(reports_df)
    
    print("\nGenerating news articles and stock returns...")
    news_df = generate_news_articles()
    returns_df = generate_stock_returns(news_df)
    
    print("\nApplying sentiment model to news and implementing trading strategy...")
    strategy_returns, cum_returns, news_with_sentiment = analyze_news_sentiment(
        sentiment_classifier, news_df, returns_df
    )
    
    print("\nEvaluating trading strategy...")
    performance_metrics = evaluate_strategy(strategy_returns)
    print("\nPerformance Metrics:")
    print(performance_metrics)
    
    # Plot cumulative returns
    plt.figure(figsize=(12, 8))
    for col in cum_returns.columns:
        plt.plot(cum_returns.index, cum_returns[col], label=col)
    plt.title("Cumulative Returns of News Sentiment Trading Strategies")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Plot histogram of sentiment scores - using matplotlib's hist directly
    plt.figure(figsize=(10, 6))
    plt.hist(news_with_sentiment['sentiment_score'], bins=50, alpha=0.75)
    plt.title("Distribution of Sentiment Scores")
    plt.xlabel("Sentiment Score")
    plt.ylabel("Frequency")
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Plot weekly average sentiment over time
    weekly_avg_sentiment = news_with_sentiment.groupby('week_id')['sentiment_score'].mean().reset_index()
    
    plt.figure(figsize=(12, 6))
    plt.plot(range(len(weekly_avg_sentiment)), weekly_avg_sentiment['sentiment_score'])
    plt.title("Weekly Average Sentiment Score")
    plt.xlabel("Week")
    plt.ylabel("Average Sentiment Score")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Plot correlation between sentiment and returns
    plt.figure(figsize=(10, 6))
    
    # Calculate average next-day return for each sentiment score bucket
    news_with_sentiment['next_day_return'] = np.nan
    
    for idx, row in news_with_sentiment.iterrows():
        date = row['date']
        company = f"Company_{row['stock_id'] + 1}"
        
        # Get next day's date
        next_day_idx = returns_df.index.get_indexer([date], method='pad')[0] + 1
        if next_day_idx < len(returns_df):
            next_day = returns_df.index[next_day_idx]
            if company in returns_df.columns:
                news_with_sentiment.loc[idx, 'next_day_return'] = returns_df.loc[next_day, company]
    
    # Simple approach: split into 5 equal-sized bins based on sentiment
    sentiment_bins = 5
    news_with_sentiment_valid = news_with_sentiment.dropna(subset=['next_day_return'])
    bin_edges = np.linspace(
        news_with_sentiment_valid['sentiment_score'].min(),
        news_with_sentiment_valid['sentiment_score'].max(),
        sentiment_bins + 1
    )
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    avg_returns = []
    for i in range(sentiment_bins):
        mask = (news_with_sentiment_valid['sentiment_score'] >= bin_edges[i]) & \
               (news_with_sentiment_valid['sentiment_score'] < bin_edges[i+1])
        avg_returns.append(news_with_sentiment_valid.loc[mask, 'next_day_return'].mean())
    
    # Plot
    plt.bar(range(sentiment_bins), avg_returns)
    plt.title("Average Next-Day Return by Sentiment Score Bucket")
    plt.xlabel("Sentiment Score Bucket (Low to High)")
    plt.ylabel("Average Next-Day Return")
    plt.xticks(range(sentiment_bins), ['Very Negative', 'Negative', 'Neutral', 'Positive', 'Very Positive'])
    plt.grid(True, alpha=0.3)
    plt.show()

if __name__ == "__main__":
    main()