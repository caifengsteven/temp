import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import random
from datetime import datetime, timedelta

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

# Function to generate simulated forex price data
def generate_forex_data(start_date, days=120, base_price=1.2000, volatility=0.002):
    dates = [start_date + timedelta(days=i) for i in range(days)]
    prices = [base_price]
    
    for i in range(1, days):
        # Random walk with drift
        change = np.random.normal(0, volatility)
        new_price = prices[-1] * (1 + change)
        prices.append(new_price)
    
    # Create DataFrame
    df = pd.DataFrame({
        'Date': dates,
        'Price': prices
    })
    
    # Calculate daily returns
    df['Return'] = df['Price'].pct_change()
    
    # Add technical indicators
    # Simple Moving Average (50-day)
    df['MA50'] = df['Price'].rolling(window=50).mean()
    
    # RSI (14-day)
    delta = df['Price'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    return df

# Generate simulated forex data for EUR/USD
start_date = datetime(2023, 4, 1)
eur_usd_data = generate_forex_data(start_date, days=120, base_price=1.0987, volatility=0.003)

# Function to generate simulated news and social media data with sentiment
def generate_news_data(forex_data, num_news_per_day=3):
    news_data = []
    sentiment_analyzer = SentimentIntensityAnalyzer()
    
    # Sample news headlines
    positive_templates = [
        "USD strengthens amid positive economic data",
        "Federal Reserve maintains interest rates, boosting dollar",
        "Strong jobs report leads to USD rally",
        "Investors flock to USD as safe haven",
        "Dollar shows resilience in volatile market"
    ]
    
    negative_templates = [
        "USD weakens as inflation concerns grow",
        "Dollar falls after disappointing economic report",
        "Federal Reserve policy uncertainty weighs on USD",
        "Trade deficit widens, pressuring the dollar",
        "USD loses ground against major currencies"
    ]
    
    neutral_templates = [
        "Markets await Federal Reserve announcement",
        "USD trades in narrow range ahead of data release",
        "Dollar maintains position amid mixed signals",
        "Traders cautious about USD direction",
        "Currency markets show limited movement"
    ]
    
    all_templates = [positive_templates, negative_templates, neutral_templates]
    sentiment_labels = ['positive', 'negative', 'neutral']
    
    for index, row in forex_data.iterrows():
        date = row['Date']
        
        # Price change influences sentiment bias (more positive news when price is rising)
        if pd.notna(row['Return']):
            if row['Return'] > 0.001:
                sentiment_bias = [0.6, 0.2, 0.2]  # More positive news
            elif row['Return'] < -0.001:
                sentiment_bias = [0.2, 0.6, 0.2]  # More negative news
            else:
                sentiment_bias = [0.3, 0.3, 0.4]  # Balanced with slight neutral bias
        else:
            sentiment_bias = [0.33, 0.33, 0.34]  # Balanced
        
        # Generate news for this day
        for _ in range(num_news_per_day):
            # Select sentiment based on bias
            sentiment_idx = np.random.choice([0, 1, 2], p=sentiment_bias)
            templates = all_templates[sentiment_idx]
            sentiment = sentiment_labels[sentiment_idx]
            
            # Select and modify a template
            headline = random.choice(templates)
            
            # Add some random variation
            if random.random() > 0.7:
                headline = headline.replace("USD", "Dollar")
            
            # Calculate VADER score
            vader_score = sentiment_analyzer.polarity_scores(headline)
            
            news_data.append({
                'Date': date,
                'Headline': headline,
                'True_Sentiment': sentiment,
                'VADER_Score': vader_score['compound']
            })
    
    return pd.DataFrame(news_data)

# Generate simulated news data
news_data = generate_news_data(eur_usd_data)

# Prepare data for Naive Bayes model
# Use CountVectorizer to convert text to features
vectorizer = CountVectorizer(stop_words='english')
X = vectorizer.fit_transform(news_data['Headline'])
y = news_data['True_Sentiment']

# Split data for training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train Naive Bayes model
nb_model = GaussianNB()
nb_model.fit(X_train.toarray(), y_train)

# Predict on test data
y_pred = nb_model.predict(X_test.toarray())

# Calculate performance metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print(f"Naive Bayes Model Performance:")
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")

# Function to generate trading signals based on sentiment and technical indicators
def generate_trading_signals(forex_data, news_data, model, vectorizer):
    # Group news by date and calculate average sentiment
    daily_sentiment = news_data.groupby('Date').agg({
        'VADER_Score': 'mean',  # Lexicon-based
        'Headline': list  # For Naive Bayes
    }).reset_index()
    
    # Add Naive Bayes sentiment
    for idx, row in daily_sentiment.iterrows():
        headlines = row['Headline']
        features = vectorizer.transform(headlines)
        predictions = model.predict(features.toarray())
        
        # Convert predictions to numeric (-1 for negative, 0 for neutral, 1 for positive)
        sentiment_values = {'negative': -1, 'neutral': 0, 'positive': 1}
        numeric_sentiment = [sentiment_values[pred] for pred in predictions]
        
        # Calculate average sentiment
        daily_sentiment.at[idx, 'NB_Score'] = sum(numeric_sentiment) / len(numeric_sentiment)
    
    # Merge with forex data
    merged_data = pd.merge(forex_data, daily_sentiment, on='Date', how='left')
    
    # Generate trading signals
    merged_data['VADER_Signal'] = 'HOLD'
    merged_data['NB_Signal'] = 'HOLD'
    
    for idx, row in merged_data.iterrows():
        if idx < 50:  # Skip first 50 rows due to MA calculation
            continue
            
        # VADER-based signals with technical confirmation
        if pd.notna(row['VADER_Score']):
            # Buy signal: Positive sentiment + Price above MA + RSI below 70
            if (row['VADER_Score'] > 0.2 and row['Price'] > row['MA50'] and row['RSI'] < 70):
                merged_data.at[idx, 'VADER_Signal'] = 'BUY'
            # Sell signal: Negative sentiment + Price below MA + RSI above 30
            elif (row['VADER_Score'] < -0.2 and row['Price'] < row['MA50'] and row['RSI'] > 30):
                merged_data.at[idx, 'VADER_Signal'] = 'SELL'
        
        # Naive Bayes-based signals with technical confirmation
        if pd.notna(row['NB_Score']):
            # Buy signal: Positive sentiment + Price above MA + RSI below 70
            if (row['NB_Score'] > 0.5 and row['Price'] > row['MA50'] and row['RSI'] < 70):
                merged_data.at[idx, 'NB_Signal'] = 'BUY'
            # Sell signal: Negative sentiment + Price below MA + RSI above 30
            elif (row['NB_Score'] < -0.5 and row['Price'] < row['MA50'] and row['RSI'] > 30):
                merged_data.at[idx, 'NB_Signal'] = 'SELL'
    
    return merged_data

# Generate trading signals
signals_data = generate_trading_signals(eur_usd_data, news_data, nb_model, vectorizer)

# Backtest the strategy
def backtest_strategy(signals_data, strategy_column, initial_capital=10000):
    backtest_results = signals_data.copy()
    backtest_results['Position'] = 0
    
    # Set position based on signals (1 for long, -1 for short, 0 for no position)
    for idx, row in backtest_results.iterrows():
        if row[strategy_column] == 'BUY':
            backtest_results.at[idx, 'Position'] = 1
        elif row[strategy_column] == 'SELL':
            backtest_results.at[idx, 'Position'] = -1
    
    # Calculate strategy returns
    backtest_results['Strategy_Return'] = backtest_results['Position'].shift(1) * backtest_results['Return']
    
    # Calculate cumulative returns
    backtest_results['Cumulative_Market_Return'] = (1 + backtest_results['Return']).cumprod() - 1
    backtest_results['Cumulative_Strategy_Return'] = (1 + backtest_results['Strategy_Return']).cumprod() - 1
    
    # Calculate equity curve
    backtest_results['Equity_Curve'] = initial_capital * (1 + backtest_results['Cumulative_Strategy_Return'])
    
    return backtest_results

# Backtest both strategies
vader_results = backtest_strategy(signals_data, 'VADER_Signal')
nb_results = backtest_strategy(signals_data, 'NB_Signal')

# Calculate final returns
vader_final_return = vader_results['Cumulative_Strategy_Return'].iloc[-1] * 100
nb_final_return = nb_results['Cumulative_Strategy_Return'].iloc[-1] * 100

print(f"\nBacktest Results:")
print(f"Lexicon-based (VADER) Strategy Return: {vader_final_return:.2f}%")
print(f"Naive Bayes Strategy Return: {nb_final_return:.2f}%")

# Plot results
plt.figure(figsize=(14, 7))
plt.plot(vader_results['Date'], vader_results['Equity_Curve'], label='VADER Strategy')
plt.plot(nb_results['Date'], nb_results['Equity_Curve'], label='Naive Bayes Strategy')
plt.plot(vader_results['Date'], 10000 * (1 + vader_results['Cumulative_Market_Return']), label='Buy & Hold')
plt.title('Strategy Performance Comparison')
plt.xlabel('Date')
plt.ylabel('Portfolio Value ($)')
plt.legend()
plt.grid(True)
plt.tight_layout()

# Plot buy/sell signals
plt.figure(figsize=(14, 10))

# Price chart with MA
plt.subplot(2, 1, 1)
plt.plot(signals_data['Date'], signals_data['Price'], label='EUR/USD')
plt.plot(signals_data['Date'], signals_data['MA50'], label='50-day MA', alpha=0.7)

# Plot buy/sell signals
buy_signals = signals_data[signals_data['NB_Signal'] == 'BUY']
sell_signals = signals_data[signals_data['NB_Signal'] == 'SELL']

plt.scatter(buy_signals['Date'], buy_signals['Price'], marker='^', color='green', s=100, label='Buy Signal')
plt.scatter(sell_signals['Date'], sell_signals['Price'], marker='v', color='red', s=100, label='Sell Signal')

plt.title('EUR/USD Price with Naive Bayes Trading Signals')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True)

# RSI subplot
plt.subplot(2, 1, 2)
plt.plot(signals_data['Date'], signals_data['RSI'], label='RSI', color='purple')
plt.axhline(y=70, color='r', linestyle='--', alpha=0.5)
plt.axhline(y=30, color='g', linestyle='--', alpha=0.5)
plt.title('Relative Strength Index (RSI)')
plt.xlabel('Date')
plt.ylabel('RSI Value')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()