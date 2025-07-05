import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
import re
import datetime
import json
import os
from tqdm import tqdm

from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from datasets import Dataset

# For PPO fine-tuning (simulated - not actual implementation)
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from trl import create_reference_model

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)



# Define stock symbols and their industries
stocks = {
    'Technology': ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'META'],
    'Healthcare': ['JNJ', 'PFE', 'UNH', 'MRK', 'ABT'],
    'Financial': ['JPM', 'BAC', 'C', 'WFC', 'GS'],
    'Consumer': ['WMT', 'PG', 'KO', 'PEP', 'MCD'],
    'Industrial': ['GE', 'BA', 'CAT', 'MMM', 'HON']
}

# Generate synthetic stock price data
def generate_stock_data(symbol, days=500, start_date="2020-01-01"):
    start_date = datetime.datetime.strptime(start_date, "%Y-%m-%d")
    dates = [start_date + datetime.timedelta(days=i) for i in range(days)]
    dates = [date for date in dates if date.weekday() < 5]  # Only weekdays
    
    # Base price and volatility by industry
    industry = next((ind for ind, symbols in stocks.items() if symbol in symbols), None)
    
    if industry == 'Technology':
        base_price = random.uniform(100, 500)
        volatility = random.uniform(0.015, 0.025)
        trend = random.uniform(0.0002, 0.0008)
    elif industry == 'Healthcare':
        base_price = random.uniform(50, 200)
        volatility = random.uniform(0.01, 0.02)
        trend = random.uniform(0.0001, 0.0005)
    elif industry == 'Financial':
        base_price = random.uniform(30, 150)
        volatility = random.uniform(0.012, 0.022)
        trend = random.uniform(0.0001, 0.0004)
    elif industry == 'Consumer':
        base_price = random.uniform(40, 180)
        volatility = random.uniform(0.008, 0.018)
        trend = random.uniform(0.0001, 0.0004)
    else:  # Industrial
        base_price = random.uniform(35, 160)
        volatility = random.uniform(0.01, 0.02)
        trend = random.uniform(0.0001, 0.0003)
    
    # Generate price series with random walk + trend
    prices = [base_price]
    for i in range(1, len(dates)):
        # Add some occasional events
        if random.random() < 0.03:  # 3% chance of significant event
            event_impact = random.choice([-1, 1]) * random.uniform(0.03, 0.08)
        else:
            event_impact = 0
            
        # Daily price movement
        daily_return = np.random.normal(trend, volatility) + event_impact
        new_price = prices[-1] * (1 + daily_return)
        prices.append(max(new_price, 0.1 * base_price))  # Prevent prices going too low
    
    # Create DataFrame
    df = pd.DataFrame({
        'Date': dates,
        'Open': prices,
        'Symbol': symbol
    })
    
    # Add High, Low, Close
    df['High'] = df['Open'] * (1 + abs(np.random.normal(0, volatility * 0.5, len(df))))
    df['Low'] = df['Open'] * (1 - abs(np.random.normal(0, volatility * 0.5, len(df))))
    df['Close'] = df['Open'] * (1 + np.random.normal(0, volatility, len(df)))
    df['Close'] = df.apply(lambda row: min(max(row['Low'], row['Close']), row['High']), axis=1)
    
    # Add Volume
    avg_volume = random.randint(1000000, 10000000)
    df['Volume'] = np.random.poisson(avg_volume, len(df))
    
    # Add daily return
    df['Return'] = df['Close'].pct_change().fillna(0)
    
    # Add movement label (binary classification: 1 for up, 0 for down)
    df['Movement'] = (df['Return'] > 0).astype(int)
    
    return df

# Generate stock data for all symbols
all_stock_data = []
for industry, symbols in stocks.items():
    for symbol in symbols:
        all_stock_data.append(generate_stock_data(symbol))

stock_data = pd.concat(all_stock_data)
stock_data = stock_data.sort_values(['Symbol', 'Date']).reset_index(drop=True)

# Define templates for tweet generation
positive_templates = [
    "{symbol} reported better than expected earnings, beating analyst estimates.",
    "Analysts at {bank} upgraded {symbol} to a buy rating with a price target of ${price}.",
    "{symbol} announced a new product launch that's expected to boost revenue.",
    "{symbol}'s CEO expressed confidence in the company's growth prospects during the earnings call.",
    "Market sentiment for {symbol} remains strong after positive economic data.",
    "{symbol} plans to expand into new markets, potentially increasing its customer base.",
    "{symbol} announced a strategic partnership with {partner} to enhance its product offerings.",
    "Institutional investors have increased their positions in {symbol}, signaling confidence.",
    "{symbol} announced a stock buyback program worth ${amount} billion.",
    "{symbol} received regulatory approval for its new {product} product line."
]

negative_templates = [
    "{symbol} missed earnings expectations, causing concern among investors.",
    "Analysts at {bank} downgraded {symbol} to a sell rating with a price target of ${price}.",
    "{symbol} faces increasing competition in its core market segments.",
    "{symbol}'s CEO acknowledged challenges in the current business environment.",
    "Market sentiment for {symbol} weakened following disappointing industry news.",
    "{symbol} announced layoffs affecting {number} employees as part of cost-cutting measures.",
    "{symbol} is facing regulatory scrutiny over its {issue} practices.",
    "A key executive at {symbol} resigned unexpectedly, raising governance concerns.",
    "{symbol} reported supply chain issues that could impact production in the coming quarter.",
    "{symbol} lowered its full-year guidance, citing macroeconomic headwinds."
]

neutral_templates = [
    "{symbol} maintained its quarterly dividend of ${dividend} per share.",
    "Trading volume for {symbol} was in line with its 30-day average.",
    "{symbol} will report its quarterly earnings on {date}.",
    "{symbol} announced no major changes to its business strategy.",
    "Analysts have mixed views on {symbol}'s near-term prospects.",
    "{symbol} participated in the {conference} industry conference.",
    "{symbol} appointed a new board member, {name}, formerly of {company}.",
    "The {industry} sector, including {symbol}, saw flat trading today.",
    "{symbol} filed its quarterly report with the SEC.",
    "{symbol} celebrated its {number}-year anniversary since its founding."
]

# Define parameters for tweet generation
banks = ["Goldman Sachs", "JP Morgan", "Morgan Stanley", "Citi", "Bank of America", "Wells Fargo", "UBS", "Deutsche Bank"]
partners = ["Microsoft", "Amazon", "Google", "IBM", "Salesforce", "Oracle", "SAP", "Walmart", "Intel", "AMD"]
issues = ["privacy", "security", "antitrust", "labor", "environmental", "tax", "accounting", "product safety"]
conferences = ["CES", "WWDC", "Google I/O", "AWS re:Invent", "Dreamforce", "E3", "MWC", "NRF", "HIMSS", "SXSW"]

# Generate tweets based on stock movement and random events
def generate_tweets(stock_data, num_tweets_per_day=5):
    tweets_data = []
    
    for symbol in stock_data['Symbol'].unique():
        symbol_data = stock_data[stock_data['Symbol'] == symbol].copy()
        
        for i, row in symbol_data.iterrows():
            date = row['Date']
            
            # Determine how many tweets to generate for this day
            daily_tweets = random.randint(1, num_tweets_per_day)
            
            # Generate tweets based on stock movement
            movement = row['Movement']
            return_val = row['Return']
            
            # Adjust tweet sentiment based on return magnitude
            if abs(return_val) < 0.005:  # Small movement
                pos_prob, neg_prob, neut_prob = 0.2, 0.2, 0.6
            elif abs(return_val) < 0.015:  # Medium movement
                if movement == 1:  # Positive
                    pos_prob, neg_prob, neut_prob = 0.4, 0.1, 0.5
                else:  # Negative
                    pos_prob, neg_prob, neut_prob = 0.1, 0.4, 0.5
            else:  # Large movement
                if movement == 1:  # Positive
                    pos_prob, neg_prob, neut_prob = 0.6, 0.1, 0.3
                else:  # Negative
                    pos_prob, neg_prob, neut_prob = 0.1, 0.6, 0.3
            
            for _ in range(daily_tweets):
                sentiment = random.choices(
                    ['positive', 'negative', 'neutral'], 
                    weights=[pos_prob, neg_prob, neut_prob], 
                    k=1
                )[0]
                
                if sentiment == 'positive':
                    template = random.choice(positive_templates)
                elif sentiment == 'negative':
                    template = random.choice(negative_templates)
                else:
                    template = random.choice(neutral_templates)
                
                # Fill in template placeholders
                tweet = template.format(
                    symbol=symbol,
                    bank=random.choice(banks),
                    price=round(row['Close'] * random.uniform(0.8, 1.2), 2),
                    partner=random.choice(partners),
                    amount=round(random.uniform(0.5, 10), 1),
                    product=f"{symbol} {random.choice(['Pro', 'Ultra', 'Max', 'Elite', 'Premium'])}",
                    number=random.randint(100, 5000),
                    issue=random.choice(issues),
                    date=(date + datetime.timedelta(days=random.randint(1, 30))).strftime('%Y-%m-%d'),
                    dividend=round(random.uniform(0.1, 2.0), 2),
                    conference=random.choice(conferences),
                    name=f"John {random.choice(['Smith', 'Johnson', 'Williams', 'Brown', 'Jones'])}",
                    company=random.choice(list(sum(stocks.values(), []))),
                    industry=next((ind for ind, symbols in stocks.items() if symbol in symbols), None)
                )
                
                # Add some Twitter-like formatting and noise
                if random.random() < 0.3:
                    tweet = f"RT @{random.choice(['MarketWatch', 'CNBC', 'WSJ', 'Bloomberg', 'Benzinga'])}: {tweet}"
                
                if random.random() < 0.2:
                    tweet += f" #{symbol} #{next((ind for ind, symbols in stocks.items() if symbol in symbols), None)}"
                
                if random.random() < 0.1:
                    tweet += f" ${symbol}"
                
                tweets_data.append({
                    'Date': date,
                    'Symbol': symbol,
                    'Tweet': tweet,
                    'Sentiment': sentiment
                })
    
    tweets_df = pd.DataFrame(tweets_data)
    return tweets_df

# Generate tweets based on stock data
tweets_data = generate_tweets(stock_data)
tweets_data = tweets_data.sort_values(['Symbol', 'Date']).reset_index(drop=True)

# Display summary of the generated data
print(f"Generated {len(stock_data)} stock price records for {len(stock_data['Symbol'].unique())} symbols")
print(f"Generated {len(tweets_data)} tweets")

# Display example tweets for one stock
example_symbol = 'AAPL'
print(f"\nExample tweets for {example_symbol}:")
example_tweets = tweets_data[tweets_data['Symbol'] == example_symbol].head(5)
for _, row in example_tweets.iterrows():
    print(f"Date: {row['Date'].strftime('%Y-%m-%d')}, Tweet: {row['Tweet']}")

# Visualize a stock's price movement
example_stock = stock_data[stock_data['Symbol'] == example_symbol].copy()
example_stock['Date'] = pd.to_datetime(example_stock['Date'])

plt.figure(figsize=(12, 6))
plt.plot(example_stock['Date'], example_stock['Close'])
plt.title(f"{example_symbol} Stock Price")
plt.xlabel("Date")
plt.ylabel("Price ($)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("stock_price_chart.png")
plt.close()

# Prepare datasets for training and testing
def prepare_datasets(stock_data, tweets_data, train_ratio=0.8, days_input=5):
    # Merge stock data with tweets
    all_data = []
    
    for symbol in stock_data['Symbol'].unique():
        symbol_stock = stock_data[stock_data['Symbol'] == symbol].copy()
        symbol_tweets = tweets_data[tweets_data['Symbol'] == symbol].copy()
        
        # Group tweets by date
        tweets_by_date = symbol_tweets.groupby('Date')['Tweet'].apply(list).to_dict()
        
        # For each day, collect data from previous days
        for i in range(days_input, len(symbol_stock)):
            current_date = symbol_stock.iloc[i]['Date']
            
            # Get data from previous days
            prev_dates = [symbol_stock.iloc[i-j]['Date'] for j in range(1, days_input+1)]
            
            # Collect tweets from previous days
            prev_tweets = []
            for prev_date in prev_dates:
                if prev_date in tweets_by_date:
                    prev_tweets.extend(tweets_by_date[prev_date])
            
            # Skip if no tweets
            if not prev_tweets:
                continue
            
            # Get next day movement (label)
            next_day_movement = symbol_stock.iloc[i]['Movement']
            
            all_data.append({
                'Symbol': symbol,
                'Date': current_date,
                'PrevTweets': prev_tweets,
                'Movement': next_day_movement
            })
    
    # Convert to DataFrame
    dataset = pd.DataFrame(all_data)
    
    # Split into train and test
    dataset = dataset.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle
    train_size = int(len(dataset) * train_ratio)
    
    train_dataset = dataset.iloc[:train_size].reset_index(drop=True)
    test_dataset = dataset.iloc[train_size:].reset_index(drop=True)
    
    return train_dataset, test_dataset

# Prepare datasets
train_dataset, test_dataset = prepare_datasets(stock_data, tweets_data)

print(f"\nPrepared {len(train_dataset)} training samples and {len(test_dataset)} testing samples")



def summarize_tweets(tweets, symbol):
    """
    Simulate the Summarize component by extracting key facts from tweets.
    
    Args:
        tweets (list): List of tweets
        symbol (str): Stock symbol
        
    Returns:
        str: Summarized facts
    """
    # In a real implementation, this would use a LLM like GPT-3.5 or Vicuna
    # Here we'll simulate it by extracting key information
    
    facts = []
    
    # Extract factual information from tweets
    positive_count = 0
    negative_count = 0
    neutral_count = 0
    
    earnings_mentioned = False
    upgrades = []
    downgrades = []
    products = []
    partnerships = []
    
    for tweet in tweets:
        # Count sentiment (in a real implementation, this would be determined by the LLM)
        if "better than expected" in tweet.lower() or "upgraded" in tweet.lower() or "buy rating" in tweet.lower():
            positive_count += 1
        elif "missed" in tweet.lower() or "downgraded" in tweet.lower() or "sell rating" in tweet.lower():
            negative_count += 1
        else:
            neutral_count += 1
        
        # Extract specific information
        if "earnings" in tweet.lower():
            earnings_mentioned = True
            if "better than expected" in tweet.lower() or "beating" in tweet.lower():
                facts.append(f"{symbol} reported better than expected earnings.")
            elif "missed" in tweet.lower():
                facts.append(f"{symbol} missed earnings expectations.")
                
        if "upgraded" in tweet.lower():
            bank = re.search(r"Analysts at ([^,]+)", tweet)
            if bank:
                bank = bank.group(1)
                facts.append(f"{bank} upgraded {symbol}.")
                
        if "downgraded" in tweet.lower():
            bank = re.search(r"Analysts at ([^,]+)", tweet)
            if bank:
                bank = bank.group(1)
                facts.append(f"{bank} downgraded {symbol}.")
                
        if "new product" in tweet.lower() or "product line" in tweet.lower():
            facts.append(f"{symbol} announced a new product.")
            
        if "partnership" in tweet.lower():
            partner = re.search(r"partnership with ([^,]+)", tweet)
            if partner:
                partner = partner.group(1)
                facts.append(f"{symbol} announced a partnership with {partner}.")
                
        if "buyback" in tweet.lower():
            facts.append(f"{symbol} announced a stock buyback program.")
            
        if "layoffs" in tweet.lower():
            facts.append(f"{symbol} announced layoffs.")
            
        if "regulatory" in tweet.lower():
            facts.append(f"{symbol} is facing regulatory scrutiny.")
    
    # Summarize sentiment
    if positive_count > negative_count:
        facts.append(f"Overall sentiment for {symbol} is positive.")
    elif negative_count > positive_count:
        facts.append(f"Overall sentiment for {symbol} is negative.")
    else:
        facts.append(f"Overall sentiment for {symbol} is mixed.")
    
    # Remove duplicates while preserving order
    unique_facts = []
    [unique_facts.append(fact) for fact in facts if fact not in unique_facts]
    
    # Format the facts as a bullet point list
    facts_text = "- " + "\n- ".join(unique_facts)
    
    return facts_text

# Test the summarize function
example_symbol = 'AAPL'
example_tweets = tweets_data[tweets_data['Symbol'] == example_symbol].sample(10)['Tweet'].tolist()
summarized_facts = summarize_tweets(example_tweets, example_symbol)

print("\nExample Summarized Facts:")
print(summarized_facts)


def generate_explanation(summarized_facts, symbol):
    """
    Simulate the Explain component by generating a stock prediction and explanation.
    
    Args:
        summarized_facts (str): Summarized facts
        symbol (str): Stock symbol
        
    Returns:
        tuple: (price_movement, explanation)
    """
    # Count positive and negative facts
    positive_count = summarized_facts.lower().count("better than expected")
    positive_count += summarized_facts.lower().count("upgraded")
    positive_count += summarized_facts.lower().count("new product")
    positive_count += summarized_facts.lower().count("partnership")
    positive_count += summarized_facts.lower().count("buyback")
    positive_count += summarized_facts.lower().count("overall sentiment for {} is positive".format(symbol.lower()))
    
    negative_count = summarized_facts.lower().count("missed")
    negative_count += summarized_facts.lower().count("downgraded")
    negative_count += summarized_facts.lower().count("layoffs")
    negative_count += summarized_facts.lower().count("regulatory scrutiny")
    negative_count += summarized_facts.lower().count("overall sentiment for {} is negative".format(symbol.lower()))
    
    # Determine price movement
    if positive_count > negative_count:
        price_movement = "Positive"
    elif negative_count > positive_count:
        price_movement = "Negative"
    else:
        # In case of a tie, slightly bias towards positive (optimism bias in markets)
        price_movement = "Positive" if random.random() > 0.4 else "Negative"
    
    # Generate explanation based on the facts
    explanation = f"The overall sentiment for {symbol} stock is {price_movement.lower()}. "
    
    if "better than expected earnings" in summarized_facts:
        explanation += f"{symbol} reported better than expected earnings, which typically leads to positive market reaction. "
    
    if "missed earnings" in summarized_facts:
        explanation += f"{symbol} missed earnings expectations, which typically leads to negative market reaction. "
    
    if "upgraded" in summarized_facts:
        explanation += f"Analyst upgrades suggest confidence in {symbol}'s future performance. "
    
    if "downgraded" in summarized_facts:
        explanation += f"Analyst downgrades suggest concerns about {symbol}'s future performance. "
    
    if "new product" in summarized_facts:
        explanation += f"The announcement of new products indicates potential for future revenue growth. "
    
    if "partnership" in summarized_facts:
        explanation += f"Strategic partnerships can expand {symbol}'s market reach and capabilities. "
    
    if "buyback" in summarized_facts:
        explanation += f"Stock buyback programs typically indicate management's confidence in the company's value. "
    
    if "layoffs" in summarized_facts:
        explanation += f"Layoffs may signal cost-cutting measures, which can be viewed negatively by the market. "
    
    if "regulatory scrutiny" in summarized_facts:
        explanation += f"Regulatory scrutiny introduces uncertainty and potential compliance costs. "
    
    return price_movement, explanation

def self_reflect(summarized_facts, symbol, true_movement, price_movement, explanation):
    """
    Simulate the self-reflection process to improve predictions.
    
    Args:
        summarized_facts (str): Summarized facts
        symbol (str): Stock symbol
        true_movement (int): The actual price movement (1 for up, 0 for down)
        price_movement (str): The predicted price movement
        explanation (str): The explanation for the prediction
        
    Returns:
        str: Reflection on what went wrong and how to improve
    """
    # Check if prediction was correct
    predicted_correct = (price_movement == "Positive" and true_movement == 1) or \
                        (price_movement == "Negative" and true_movement == 0)
    
    if predicted_correct:
        return "The prediction was correct."
    
    # Generate reflection on why the prediction was wrong
    reflection = f"The prediction of {price_movement} for {symbol} was incorrect. "
    
    # Analyze what might have gone wrong
    if price_movement == "Positive" and true_movement == 0:
        reflection += "The analysis may have overweighted positive factors or underweighted negative factors. "
        
        # Identify potential reasons
        if "regulatory scrutiny" in summarized_facts.lower():
            reflection += "The impact of regulatory scrutiny may have been underestimated. "
        if "layoffs" in summarized_facts.lower():
            reflection += "The market may have reacted more negatively to the announced layoffs than anticipated. "
        if "missed earnings" in summarized_facts.lower():
            reflection += "Missing earnings expectations typically has a stronger negative impact than was accounted for. "
            
        reflection += "For future predictions, I should place more emphasis on negative indicators, especially those related to financial performance and regulatory issues."
        
    else:  # price_movement == "Negative" and true_movement == 1
        reflection += "The analysis may have overweighted negative factors or underweighted positive factors. "
        
        # Identify potential reasons
        if "better than expected earnings" in summarized_facts.lower():
            reflection += "The positive impact of beating earnings expectations may have been underestimated. "
        if "upgraded" in summarized_facts.lower():
            reflection += "Analyst upgrades might have a stronger positive influence than was accounted for. "
        if "new product" in summarized_facts.lower() or "partnership" in summarized_facts.lower():
            reflection += "The market may have reacted more positively to new products or partnerships than anticipated. "
            
        reflection += "For future predictions, I should place more emphasis on positive indicators, especially those related to financial performance and growth prospects."
    
    return reflection

def improve_explanation(summarized_facts, symbol, reflection):
    """
    Generate an improved explanation based on self-reflection.
    
    Args:
        summarized_facts (str): Summarized facts
        symbol (str): Stock symbol
        reflection (str): Reflection on what went wrong
        
    Returns:
        tuple: (improved_price_movement, improved_explanation)
    """
    # Determine if the original prediction was positive or negative
    original_was_positive = "The prediction of Positive" in reflection
    
    # Flip the prediction based on reflection
    improved_price_movement = "Negative" if original_was_positive else "Positive"
    
    # Generate an improved explanation that addresses the reflection
    improved_explanation = f"After careful consideration, the overall sentiment for {symbol} stock is {improved_price_movement.lower()}. "
    
    # Adjust the explanation based on the reflection
    if original_was_positive:  # Original was Positive, now Negative
        if "regulatory scrutiny" in summarized_facts.lower():
            improved_explanation += f"The regulatory scrutiny facing {symbol} introduces significant uncertainty and potential compliance costs, which is likely to weigh on investor sentiment. "
        if "layoffs" in summarized_facts.lower():
            improved_explanation += f"The announced layoffs at {symbol} suggest cost-cutting measures that could signal underlying business challenges. "
        if "missed earnings" in summarized_facts.lower():
            improved_explanation += f"Missing earnings expectations is a strong negative signal that typically leads to stock price declines. "
        
        improved_explanation += f"While there may be some positive factors for {symbol}, the negative indicators appear more significant for near-term stock performance."
    
    else:  # Original was Negative, now Positive
        if "better than expected earnings" in summarized_facts.lower():
            improved_explanation += f"{symbol}'s better-than-expected earnings demonstrate financial strength and execution capability, which typically leads to positive market reactions. "
        if "upgraded" in summarized_facts.lower():
            improved_explanation += f"Analyst upgrades for {symbol} indicate professional confidence in the company's prospects, which can positively influence investor sentiment. "
        if "new product" in summarized_facts.lower():
            improved_explanation += f"The announcement of new products suggests innovation and potential for future revenue growth. "
        if "partnership" in summarized_facts.lower():
            improved_explanation += f"Strategic partnerships can expand {symbol}'s market reach and capabilities, creating new growth opportunities. "
        
        improved_explanation += f"While there may be some challenges facing {symbol}, the positive indicators appear more significant for near-term stock performance."
    
    return improved_price_movement, improved_explanation

# Test the explanation and self-reflection process
true_movement = random.randint(0, 1)  # Randomly choose a movement for testing
price_movement, explanation = generate_explanation(summarized_facts, example_symbol)
reflection = self_reflect(summarized_facts, example_symbol, true_movement, price_movement, explanation)
improved_price_movement, improved_explanation = improve_explanation(summarized_facts, example_symbol, reflection)

print("\nInitial Prediction:")
print(f"Price Movement: {price_movement}")
print(f"Explanation: {explanation}")

print("\nSelf-Reflection:")
print(reflection)

print("\nImproved Prediction:")
print(f"Price Movement: {improved_price_movement}")
print(f"Explanation: {improved_explanation}")



class SEPModel:
    """
    Simulate the SEP model for stock predictions.
    
    In a real implementation, this would be a fine-tuned LLM.
    """
    
    def __init__(self):
        # Initialize simple heuristics based on common patterns
        self.positive_indicators = [
            "better than expected", "upgraded", "buy rating", "new product", 
            "partnership", "buyback", "positive sentiment"
        ]
        
        self.negative_indicators = [
            "missed", "downgraded", "sell rating", "layoffs", 
            "regulatory scrutiny", "negative sentiment"
        ]
        
        # Initialize weights (would be learned during PPO training)
        self.positive_weights = {
            "better than expected": 0.8,
            "upgraded": 0.6,
            "buy rating": 0.7,
            "new product": 0.5,
            "partnership": 0.4,
            "buyback": 0.5,
            "positive sentiment": 0.3
        }
        
        self.negative_weights = {
            "missed": 0.8,
            "downgraded": 0.6,
            "sell rating": 0.7,
            "layoffs": 0.5,
            "regulatory scrutiny": 0.6,
            "negative sentiment": 0.3
        }
        
        # Track examples seen during "training"
        self.training_examples = []
    
    def train(self, summarized_facts, true_movement):
        """
        Simulate training the model on examples.
        
        Args:
            summarized_facts (str): Summarized facts
            true_movement (int): The actual price movement (1 for up, 0 for down)
        """
        self.training_examples.append((summarized_facts, true_movement))
        
        # In a real implementation, this would involve PPO training
        # Here we'll just update our weights based on the example
        
        # Calculate current prediction
        positive_score, negative_score = self._calculate_scores(summarized_facts)
        predicted_movement = 1 if positive_score > negative_score else 0
        
        # If prediction was wrong, adjust weights
        if predicted_movement != true_movement:
            # Identify terms in the facts
            for term, weight in self.positive_weights.items():
                if term in summarized_facts.lower():
                    # Increase weight if term should lead to positive movement
                    if true_movement == 1:
                        self.positive_weights[term] = min(1.0, weight * 1.1)
                    # Decrease weight if term should lead to negative movement
                    else:
                        self.positive_weights[term] = max(0.1, weight * 0.9)
            
            for term, weight in self.negative_weights.items():
                if term in summarized_facts.lower():
                    # Increase weight if term should lead to negative movement
                    if true_movement == 0:
                        self.negative_weights[term] = min(1.0, weight * 1.1)
                    # Decrease weight if term should lead to positive movement
                    else:
                        self.negative_weights[term] = max(0.1, weight * 0.9)
    
    def predict(self, summarized_facts, symbol):
        """
        Generate a prediction and explanation based on summarized facts.
        
        Args:
            summarized_facts (str): Summarized facts
            symbol (str): Stock symbol
            
        Returns:
            tuple: (price_movement, explanation, confidence)
        """
        positive_score, negative_score = self._calculate_scores(summarized_facts)
        
        # Determine price movement
        if positive_score > negative_score:
            price_movement = "Positive"
            confidence = positive_score / (positive_score + negative_score)
        else:
            price_movement = "Negative"
            confidence = negative_score / (positive_score + negative_score)
        
        # Generate explanation
        explanation = self._generate_explanation(summarized_facts, symbol, price_movement, positive_score, negative_score)
        
        return price_movement, explanation, confidence
    
    def _calculate_scores(self, summarized_facts):
        """
        Calculate positive and negative scores based on the facts.
        
        Args:
            summarized_facts (str): Summarized facts
            
        Returns:
            tuple: (positive_score, negative_score)
        """
        positive_score = 0
        negative_score = 0
        
        # Calculate positive score
        for term, weight in self.positive_weights.items():
            if term in summarized_facts.lower():
                positive_score += weight
        
        # Calculate negative score
        for term, weight in self.negative_weights.items():
            if term in summarized_facts.lower():
                negative_score += weight
        
        return positive_score, negative_score
    
    def _generate_explanation(self, summarized_facts, symbol, price_movement, positive_score, negative_score):
        """
        Generate an explanation for the prediction.
        
        Args:
            summarized_facts (str): Summarized facts
            symbol (str): Stock symbol
            price_movement (str): Predicted price movement
            positive_score (float): Positive score
            negative_score (float): Negative score
            
        Returns:
            str: Explanation
        """
        explanation = f"Based on the available information, the sentiment for {symbol} stock is predicted to be {price_movement.lower()}. "
        
        # Add details about the factors considered
        positive_factors = []
        negative_factors = []
        
        for term in self.positive_weights.keys():
            if term in summarized_facts.lower():
                if term == "better than expected":
                    positive_factors.append(f"{symbol} reported better than expected earnings")
                elif term == "upgraded":
                    positive_factors.append(f"Analyst upgrades for {symbol}")
                elif term == "buy rating":
                    positive_factors.append(f"Buy ratings for {symbol}")
                elif term == "new product":
                    positive_factors.append(f"New product announcements")
                elif term == "partnership":
                    positive_factors.append(f"Strategic partnerships")
                elif term == "buyback":
                    positive_factors.append(f"Stock buyback program")
        
        for term in self.negative_weights.keys():
            if term in summarized_facts.lower():
                if term == "missed":
                    negative_factors.append(f"{symbol} missed earnings expectations")
                elif term == "downgraded":
                    negative_factors.append(f"Analyst downgrades for {symbol}")
                elif term == "sell rating":
                    negative_factors.append(f"Sell ratings for {symbol}")
                elif term == "layoffs":
                    negative_factors.append(f"Announced layoffs")
                elif term == "regulatory scrutiny":
                    negative_factors.append(f"Regulatory scrutiny")
        
        # Add details to the explanation
        if positive_factors:
            explanation += "Positive factors include: " + ", ".join(positive_factors) + ". "
        
        if negative_factors:
            explanation += "Negative factors include: " + ", ".join(negative_factors) + ". "
        
        # Explain the balance of factors
        if price_movement == "Positive":
            explanation += f"The positive factors outweigh the negative factors with a confidence of {positive_score / (positive_score + negative_score):.2f}, suggesting an upward movement for {symbol} stock."
        else:
            explanation += f"The negative factors outweigh the positive factors with a confidence of {negative_score / (positive_score + negative_score):.2f}, suggesting a downward movement for {symbol} stock."
        
        return explanation

# Create and "train" the SEP model on the training dataset
def train_sep_model(train_dataset):
    """
    Train the SEP model on the training dataset.
    
    Args:
        train_dataset (pd.DataFrame): Training dataset
        
    Returns:
        SEPModel: Trained model
    """
    model = SEPModel()
    
    for _, row in tqdm(train_dataset.iterrows(), total=len(train_dataset), desc="Training SEP model"):
        symbol = row['Symbol']
        tweets = row['PrevTweets']
        true_movement = row['Movement']
        
        # Summarize tweets
        summarized_facts = summarize_tweets(tweets, symbol)
        
        # Train the model
        model.train(summarized_facts, true_movement)
    
    return model

# Train the SEP model
sep_model = train_sep_model(train_dataset)

# Evaluate the SEP model on the test dataset
def evaluate_sep_model(model, test_dataset):
    """
    Evaluate the SEP model on the test dataset.
    
    Args:
        model (SEPModel): Trained model
        test_dataset (pd.DataFrame): Test dataset
        
    Returns:
        dict: Evaluation metrics
    """
    predictions = []
    ground_truth = []
    explanations = []
    confidences = []
    
    for _, row in tqdm(test_dataset.iterrows(), total=len(test_dataset), desc="Evaluating SEP model"):
        symbol = row['Symbol']
        tweets = row['PrevTweets']
        true_movement = row['Movement']
        
        # Summarize tweets
        summarized_facts = summarize_tweets(tweets, symbol)
        
        # Generate prediction
        price_movement, explanation, confidence = model.predict(summarized_facts, symbol)
        
        # Convert price movement to binary
        pred_movement = 1 if price_movement == "Positive" else 0
        
        predictions.append(pred_movement)
        ground_truth.append(true_movement)
        explanations.append(explanation)
        confidences.append(confidence)
    
    # Calculate metrics
    accuracy = sum(p == g for p, g in zip(predictions, ground_truth)) / len(predictions)
    
    # Calculate MCC (Matthews Correlation Coefficient)
    tp = sum(p == 1 and g == 1 for p, g in zip(predictions, ground_truth))
    tn = sum(p == 0 and g == 0 for p, g in zip(predictions, ground_truth))
    fp = sum(p == 1 and g == 0 for p, g in zip(predictions, ground_truth))
    fn = sum(p == 0 and g == 1 for p, g in zip(predictions, ground_truth))
    
    numerator = (tp * tn) - (fp * fn)
    denominator = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
    
    mcc = numerator / denominator if denominator != 0 else 0
    
    # Store results
    results = {
        'accuracy': accuracy,
        'mcc': mcc,
        'predictions': predictions,
        'ground_truth': ground_truth,
        'explanations': explanations,
        'confidences': confidences
    }
    
    return results

# Evaluate the SEP model
sep_results = evaluate_sep_model(sep_model, test_dataset)

print("\nSEP Model Evaluation:")
print(f"Accuracy: {sep_results['accuracy']:.4f}")
print(f"MCC: {sep_results['mcc']:.4f}")

# Show some example predictions with explanations
print("\nExample Predictions:")
for i in range(5):
    idx = random.randint(0, len(sep_results['predictions']) - 1)
    symbol = test_dataset.iloc[idx]['Symbol']
    pred = "Positive" if sep_results['predictions'][idx] == 1 else "Negative"
    true = "Positive" if sep_results['ground_truth'][idx] == 1 else "Negative"
    conf = sep_results['confidences'][idx]
    
    print(f"\nSymbol: {symbol}")
    print(f"Predicted: {pred} (Confidence: {conf:.2f}), Actual: {true}")
    print(f"Explanation: {sep_results['explanations'][idx]}")


# 1. VAE+Attention baseline (simplified simulation)
def vae_attention_baseline(train_dataset, test_dataset):
    """
    Simulate the VAE+Attention baseline.
    
    Args:
        train_dataset (pd.DataFrame): Training dataset
        test_dataset (pd.DataFrame): Test dataset
        
    Returns:
        dict: Evaluation metrics
    """
    # In a real implementation, this would train a VAE+Attention model
    # Here we'll simulate it with simple heuristics
    
    # Count term frequencies in training data
    term_frequencies = {}
    term_correlations = {}
    
    for _, row in train_dataset.iterrows():
        tweets = row['PrevTweets']
        movement = row['Movement']
        
        # Extract terms from tweets
        for tweet in tweets:
            words = tweet.lower().split()
            for word in words:
                if word not in term_frequencies:
                    term_frequencies[word] = 0
                    term_correlations[word] = 0
                
                term_frequencies[word] += 1
                if movement == 1:
                    term_correlations[word] += 1
                else:
                    term_correlations[word] -= 1
    
    # Calculate term weights
    term_weights = {}
    for term, freq in term_frequencies.items():
        if freq >= 5:  # Only consider terms that appear at least 5 times
            term_weights[term] = term_correlations[term] / freq
    
    # Evaluate on test data
    predictions = []
    ground_truth = []
    
    for _, row in test_dataset.iterrows():
        tweets = row['PrevTweets']
        movement = row['Movement']
        
        # Calculate score for tweets
        score = 0
        count = 0
        
        for tweet in tweets:
            words = tweet.lower().split()
            for word in words:
                if word in term_weights:
                    score += term_weights[word]
                    count += 1
        
        # Make prediction
        if count > 0:
            pred = 1 if score / count > 0 else 0
        else:
            pred = 1  # Default to positive if no known terms
        
        predictions.append(pred)
        ground_truth.append(movement)
    
    # Calculate metrics
    accuracy = sum(p == g for p, g in zip(predictions, ground_truth)) / len(predictions)
    
    # Calculate MCC
    tp = sum(p == 1 and g == 1 for p, g in zip(predictions, ground_truth))
    tn = sum(p == 0 and g == 0 for p, g in zip(predictions, ground_truth))
    fp = sum(p == 1 and g == 0 for p, g in zip(predictions, ground_truth))
    fn = sum(p == 0 and g == 1 for p, g in zip(predictions, ground_truth))
    
    numerator = (tp * tn) - (fp * fn)
    denominator = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
    
    mcc = numerator / denominator if denominator != 0 else 0
    
    results = {
        'accuracy': accuracy,
        'mcc': mcc,
        'predictions': predictions,
        'ground_truth': ground_truth
    }
    
    return results

# 2. GRU+Attention baseline (simplified simulation)
def gru_attention_baseline(train_dataset, test_dataset):
    """
    Simulate the GRU+Attention baseline.
    
    Args:
        train_dataset (pd.DataFrame): Training dataset
        test_dataset (pd.DataFrame): Test dataset
        
    Returns:
        dict: Evaluation metrics
    """
    # This is a simplified simulation - in reality, this would train a GRU+Attention model
    
    # For simulation, we'll use a similar approach to VAE+Attention but with temporal weighting
    term_frequencies = {}
    term_correlations = {}
    
    for _, row in train_dataset.iterrows():
        tweets = row['PrevTweets']
        movement = row['Movement']
        
        # Extract terms from tweets with temporal weighting
        for i, tweet in enumerate(tweets):
            # Give more weight to recent tweets
            recency_weight = 1 + (i / len(tweets))
            
            words = tweet.lower().split()
            for word in words:
                if word not in term_frequencies:
                    term_frequencies[word] = 0
                    term_correlations[word] = 0
                
                term_frequencies[word] += recency_weight
                if movement == 1:
                    term_correlations[word] += recency_weight
                else:
                    term_correlations[word] -= recency_weight
    
    # Calculate term weights
    term_weights = {}
    for term, freq in term_frequencies.items():
        if freq >= 5:  # Only consider terms that appear at least 5 times
            term_weights[term] = term_correlations[term] / freq
    
    # Evaluate on test data
    predictions = []
    ground_truth = []
    
    for _, row in test_dataset.iterrows():
        tweets = row['PrevTweets']
        movement = row['Movement']
        
        # Calculate score for tweets with temporal weighting
        score = 0
        weight_sum = 0
        
        for i, tweet in enumerate(tweets):
            recency_weight = 1 + (i / len(tweets))
            
            words = tweet.lower().split()
            tweet_score = 0
            tweet_count = 0
            
            for word in words:
                if word in term_weights:
                    tweet_score += term_weights[word]
                    tweet_count += 1
            
            if tweet_count > 0:
                score += (tweet_score / tweet_count) * recency_weight
                weight_sum += recency_weight
        
        # Make prediction
        if weight_sum > 0:
            pred = 1 if score / weight_sum > 0 else 0
        else:
            pred = 1  # Default to positive if no known terms
        
        predictions.append(pred)
        ground_truth.append(movement)
    
    # Calculate metrics
    accuracy = sum(p == g for p, g in zip(predictions, ground_truth)) / len(predictions)
    
    # Calculate MCC
    tp = sum(p == 1 and g == 1 for p, g in zip(predictions, ground_truth))
    tn = sum(p == 0 and g == 0 for p, g in zip(predictions, ground_truth))
    fp = sum(p == 1 and g == 0 for p, g in zip(predictions, ground_truth))
    fn = sum(p == 0 and g == 1 for p, g in zip(predictions, ground_truth))
    
    numerator = (tp * tn) - (fp * fn)
    denominator = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
    
    mcc = numerator / denominator if denominator != 0 else 0
    
    results = {
        'accuracy': accuracy,
        'mcc': mcc,
        'predictions': predictions,
        'ground_truth': ground_truth
    }
    
    return results

# 3. Transformer baseline (simplified simulation)
def transformer_baseline(train_dataset, test_dataset):
    """
    Simulate the Transformer baseline.
    
    Args:
        train_dataset (pd.DataFrame): Training dataset
        test_dataset (pd.DataFrame): Test dataset
        
    Returns:
        dict: Evaluation metrics
    """
    # This is a simplified simulation - in reality, this would train a Transformer model
    
    # For simulation, we'll use n-grams to capture more context
    ngram_frequencies = {}
    ngram_correlations = {}
    
    for _, row in train_dataset.iterrows():
        tweets = row['PrevTweets']
        movement = row['Movement']
        
        # Extract n-grams from tweets
        for tweet in tweets:
            words = tweet.lower().split()
            
            # Generate unigrams and bigrams
            ngrams = words + [' '.join(words[i:i+2]) for i in range(len(words)-1)]
            
            for ngram in ngrams:
                if ngram not in ngram_frequencies:
                    ngram_frequencies[ngram] = 0
                    ngram_correlations[ngram] = 0
                
                ngram_frequencies[ngram] += 1
                if movement == 1:
                    ngram_correlations[ngram] += 1
                else:
                    ngram_correlations[ngram] -= 1
    
    # Calculate n-gram weights
    ngram_weights = {}
    for ngram, freq in ngram_frequencies.items():
        if freq >= 5:  # Only consider n-grams that appear at least 5 times
            ngram_weights[ngram] = ngram_correlations[ngram] / freq
    
    # Evaluate on test data
    predictions = []
    ground_truth = []
    
    for _, row in test_dataset.iterrows():
        tweets = row['PrevTweets']
        movement = row['Movement']
        
        # Calculate score for tweets
        score = 0
        count = 0
        
        for tweet in tweets:
            words = tweet.lower().split()
            
            # Generate unigrams and bigrams
            ngrams = words + [' '.join(words[i:i+2]) for i in range(len(words)-1)]
            
            for ngram in ngrams:
                if ngram in ngram_weights:
                    score += ngram_weights[ngram]
                    count += 1
        
        # Make prediction
        if count > 0:
            pred = 1 if score / count > 0 else 0
        else:
            pred = 1  # Default to positive if no known n-grams
        
        predictions.append(pred)
        ground_truth.append(movement)
    
    # Calculate metrics
    accuracy = sum(p == g for p, g in zip(predictions, ground_truth)) / len(predictions)
    
    # Calculate MCC
    tp = sum(p == 1 and g == 1 for p, g in zip(predictions, ground_truth))
    tn = sum(p == 0 and g == 0 for p, g in zip(predictions, ground_truth))
    fp = sum(p == 1 and g == 0 for p, g in zip(predictions, ground_truth))
    fn = sum(p == 0 and g == 1 for p, g in zip(predictions, ground_truth))
    
    numerator = (tp * tn) - (fp * fn)
    denominator = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
    
    mcc = numerator / denominator if denominator != 0 else 0
    
    results = {
        'accuracy': accuracy,
        'mcc': mcc,
        'predictions': predictions,
        'ground_truth': ground_truth
    }
    
    return results

# 4. Basic LLM baseline (simulated)
def llm_baseline(test_dataset):
    """
    Simulate a basic LLM baseline.
    
    Args:
        test_dataset (pd.DataFrame): Test dataset
        
    Returns:
        dict: Evaluation metrics
    """
    # This simulates a generic LLM that can analyze sentiment but doesn't weigh mixed sentiment well
    
    predictions = []
    ground_truth = []
    explanations = []
    
    for _, row in test_dataset.iterrows():
        symbol = row['Symbol']
        tweets = row['PrevTweets']
        movement = row['Movement']
        
        # Count positive and negative sentiment in tweets
        pos_count = 0
        neg_count = 0
        
        for tweet in tweets:
            if any(term in tweet.lower() for term in ["better than expected", "upgraded", "buy rating", "new product", "partnership", "buyback"]):
                pos_count += 1
            
            if any(term in tweet.lower() for term in ["missed", "downgraded", "sell rating", "layoffs", "regulatory scrutiny"]):
                neg_count += 1
        
        # Generate prediction
        # Simulate LLM's tendency to be indecisive with mixed signals
        if abs(pos_count - neg_count) <= 2:
            # Indecisive response - count as incorrect
            pred = 0  # Default to negative for indecisive cases
            explanation = f"The sentiment for {symbol} is mixed, with both positive and negative factors present."
        else:
            # Decisive response
            pred = 1 if pos_count > neg_count else 0
            
            if pred == 1:
                explanation = f"The sentiment for {symbol} is positive, with more positive factors than negative factors."
            else:
                explanation = f"The sentiment for {symbol} is negative, with more negative factors than positive factors."
        
        predictions.append(pred)
        ground_truth.append(movement)
        explanations.append(explanation)
    
    # Calculate metrics
    accuracy = sum(p == g for p, g in zip(predictions, ground_truth)) / len(predictions)
    
    # Calculate MCC
    tp = sum(p == 1 and g == 1 for p, g in zip(predictions, ground_truth))
    tn = sum(p == 0 and g == 0 for p, g in zip(predictions, ground_truth))
    fp = sum(p == 1 and g == 0 for p, g in zip(predictions, ground_truth))
    fn = sum(p == 0 and g == 1 for p, g in zip(predictions, ground_truth))
    
    numerator = (tp * tn) - (fp * fn)
    denominator = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
    
    mcc = numerator / denominator if denominator != 0 else 0
    
    results = {
        'accuracy': accuracy,
        'mcc': mcc,
        'predictions': predictions,
        'ground_truth': ground_truth,
        'explanations': explanations
    }
    
    return results

# Run all baseline models
print("\nEvaluating baseline models...")
vae_results = vae_attention_baseline(train_dataset, test_dataset)
gru_results = gru_attention_baseline(train_dataset, test_dataset)
transformer_results = transformer_baseline(train_dataset, test_dataset)
llm_results = llm_baseline(test_dataset)

print("\nBaseline Model Evaluation:")
print(f"VAE+Attention: Accuracy = {vae_results['accuracy']:.4f}, MCC = {vae_results['mcc']:.4f}")
print(f"GRU+Attention: Accuracy = {gru_results['accuracy']:.4f}, MCC = {gru_results['mcc']:.4f}")
print(f"Transformer: Accuracy = {transformer_results['accuracy']:.4f}, MCC = {transformer_results['mcc']:.4f}")
print(f"Basic LLM: Accuracy = {llm_results['accuracy']:.4f}, MCC = {llm_results['mcc']:.4f}")
print(f"SEP (Ours): Accuracy = {sep_results['accuracy']:.4f}, MCC = {sep_results['mcc']:.4f}")

# Compare explanations between Basic LLM and SEP
print("\nComparison of Explanations:")
for i in range(3):
    idx = random.randint(0, len(sep_results['predictions']) - 1)
    symbol = test_dataset.iloc[idx]['Symbol']
    
    print(f"\nSymbol: {symbol}")
    print(f"Basic LLM Explanation: {llm_results['explanations'][idx]}")
    print(f"SEP Explanation: {sep_results['explanations'][idx]}")


# 1. VAE+Attention baseline (simplified simulation)
def vae_attention_baseline(train_dataset, test_dataset):
    """
    Simulate the VAE+Attention baseline.
    
    Args:
        train_dataset (pd.DataFrame): Training dataset
        test_dataset (pd.DataFrame): Test dataset
        
    Returns:
        dict: Evaluation metrics
    """
    # In a real implementation, this would train a VAE+Attention model
    # Here we'll simulate it with simple heuristics
    
    # Count term frequencies in training data
    term_frequencies = {}
    term_correlations = {}
    
    for _, row in train_dataset.iterrows():
        tweets = row['PrevTweets']
        movement = row['Movement']
        
        # Extract terms from tweets
        for tweet in tweets:
            words = tweet.lower().split()
            for word in words:
                if word not in term_frequencies:
                    term_frequencies[word] = 0
                    term_correlations[word] = 0
                
                term_frequencies[word] += 1
                if movement == 1:
                    term_correlations[word] += 1
                else:
                    term_correlations[word] -= 1
    
    # Calculate term weights
    term_weights = {}
    for term, freq in term_frequencies.items():
        if freq >= 5:  # Only consider terms that appear at least 5 times
            term_weights[term] = term_correlations[term] / freq
    
    # Evaluate on test data
    predictions = []
    ground_truth = []
    
    for _, row in test_dataset.iterrows():
        tweets = row['PrevTweets']
        movement = row['Movement']
        
        # Calculate score for tweets
        score = 0
        count = 0
        
        for tweet in tweets:
            words = tweet.lower().split()
            for word in words:
                if word in term_weights:
                    score += term_weights[word]
                    count += 1
        
        # Make prediction
        if count > 0:
            pred = 1 if score / count > 0 else 0
        else:
            pred = 1  # Default to positive if no known terms
        
        predictions.append(pred)
        ground_truth.append(movement)
    
    # Calculate metrics
    accuracy = sum(p == g for p, g in zip(predictions, ground_truth)) / len(predictions)
    
    # Calculate MCC
    tp = sum(p == 1 and g == 1 for p, g in zip(predictions, ground_truth))
    tn = sum(p == 0 and g == 0 for p, g in zip(predictions, ground_truth))
    fp = sum(p == 1 and g == 0 for p, g in zip(predictions, ground_truth))
    fn = sum(p == 0 and g == 1 for p, g in zip(predictions, ground_truth))
    
    numerator = (tp * tn) - (fp * fn)
    denominator = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
    
    mcc = numerator / denominator if denominator != 0 else 0
    
    results = {
        'accuracy': accuracy,
        'mcc': mcc,
        'predictions': predictions,
        'ground_truth': ground_truth
    }
    
    return results

# 2. GRU+Attention baseline (simplified simulation)
def gru_attention_baseline(train_dataset, test_dataset):
    """
    Simulate the GRU+Attention baseline.
    
    Args:
        train_dataset (pd.DataFrame): Training dataset
        test_dataset (pd.DataFrame): Test dataset
        
    Returns:
        dict: Evaluation metrics
    """
    # This is a simplified simulation - in reality, this would train a GRU+Attention model
    
    # For simulation, we'll use a similar approach to VAE+Attention but with temporal weighting
    term_frequencies = {}
    term_correlations = {}
    
    for _, row in train_dataset.iterrows():
        tweets = row['PrevTweets']
        movement = row['Movement']
        
        # Extract terms from tweets with temporal weighting
        for i, tweet in enumerate(tweets):
            # Give more weight to recent tweets
            recency_weight = 1 + (i / len(tweets))
            
            words = tweet.lower().split()
            for word in words:
                if word not in term_frequencies:
                    term_frequencies[word] = 0
                    term_correlations[word] = 0
                
                term_frequencies[word] += recency_weight
                if movement == 1:
                    term_correlations[word] += recency_weight
                else:
                    term_correlations[word] -= recency_weight
    
    # Calculate term weights
    term_weights = {}
    for term, freq in term_frequencies.items():
        if freq >= 5:  # Only consider terms that appear at least 5 times
            term_weights[term] = term_correlations[term] / freq
    
    # Evaluate on test data
    predictions = []
    ground_truth = []
    
    for _, row in test_dataset.iterrows():
        tweets = row['PrevTweets']
        movement = row['Movement']
        
        # Calculate score for tweets with temporal weighting
        score = 0
        weight_sum = 0
        
        for i, tweet in enumerate(tweets):
            recency_weight = 1 + (i / len(tweets))
            
            words = tweet.lower().split()
            tweet_score = 0
            tweet_count = 0
            
            for word in words:
                if word in term_weights:
                    tweet_score += term_weights[word]
                    tweet_count += 1
            
            if tweet_count > 0:
                score += (tweet_score / tweet_count) * recency_weight
                weight_sum += recency_weight
        
        # Make prediction
        if weight_sum > 0:
            pred = 1 if score / weight_sum > 0 else 0
        else:
            pred = 1  # Default to positive if no known terms
        
        predictions.append(pred)
        ground_truth.append(movement)
    
    # Calculate metrics
    accuracy = sum(p == g for p, g in zip(predictions, ground_truth)) / len(predictions)
    
    # Calculate MCC
    tp = sum(p == 1 and g == 1 for p, g in zip(predictions, ground_truth))
    tn = sum(p == 0 and g == 0 for p, g in zip(predictions, ground_truth))
    fp = sum(p == 1 and g == 0 for p, g in zip(predictions, ground_truth))
    fn = sum(p == 0 and g == 1 for p, g in zip(predictions, ground_truth))
    
    numerator = (tp * tn) - (fp * fn)
    denominator = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
    
    mcc = numerator / denominator if denominator != 0 else 0
    
    results = {
        'accuracy': accuracy,
        'mcc': mcc,
        'predictions': predictions,
        'ground_truth': ground_truth
    }
    
    return results

# 3. Transformer baseline (simplified simulation)
def transformer_baseline(train_dataset, test_dataset):
    """
    Simulate the Transformer baseline.
    
    Args:
        train_dataset (pd.DataFrame): Training dataset
        test_dataset (pd.DataFrame): Test dataset
        
    Returns:
        dict: Evaluation metrics
    """
    # This is a simplified simulation - in reality, this would train a Transformer model
    
    # For simulation, we'll use n-grams to capture more context
    ngram_frequencies = {}
    ngram_correlations = {}
    
    for _, row in train_dataset.iterrows():
        tweets = row['PrevTweets']
        movement = row['Movement']
        
        # Extract n-grams from tweets
        for tweet in tweets:
            words = tweet.lower().split()
            
            # Generate unigrams and bigrams
            ngrams = words + [' '.join(words[i:i+2]) for i in range(len(words)-1)]
            
            for ngram in ngrams:
                if ngram not in ngram_frequencies:
                    ngram_frequencies[ngram] = 0
                    ngram_correlations[ngram] = 0
                
                ngram_frequencies[ngram] += 1
                if movement == 1:
                    ngram_correlations[ngram] += 1
                else:
                    ngram_correlations[ngram] -= 1
    
    # Calculate n-gram weights
    ngram_weights = {}
    for ngram, freq in ngram_frequencies.items():
        if freq >= 5:  # Only consider n-grams that appear at least 5 times
            ngram_weights[ngram] = ngram_correlations[ngram] / freq
    
    # Evaluate on test data
    predictions = []
    ground_truth = []
    
    for _, row in test_dataset.iterrows():
        tweets = row['PrevTweets']
        movement = row['Movement']
        
        # Calculate score for tweets
        score = 0
        count = 0
        
        for tweet in tweets:
            words = tweet.lower().split()
            
            # Generate unigrams and bigrams
            ngrams = words + [' '.join(words[i:i+2]) for i in range(len(words)-1)]
            
            for ngram in ngrams:
                if ngram in ngram_weights:
                    score += ngram_weights[ngram]
                    count += 1
        
        # Make prediction
        if count > 0:
            pred = 1 if score / count > 0 else 0
        else:
            pred = 1  # Default to positive if no known n-grams
        
        predictions.append(pred)
        ground_truth.append(movement)
    
    # Calculate metrics
    accuracy = sum(p == g for p, g in zip(predictions, ground_truth)) / len(predictions)
    
    # Calculate MCC
    tp = sum(p == 1 and g == 1 for p, g in zip(predictions, ground_truth))
    tn = sum(p == 0 and g == 0 for p, g in zip(predictions, ground_truth))
    fp = sum(p == 1 and g == 0 for p, g in zip(predictions, ground_truth))
    fn = sum(p == 0 and g == 1 for p, g in zip(predictions, ground_truth))
    
    numerator = (tp * tn) - (fp * fn)
    denominator = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
    
    mcc = numerator / denominator if denominator != 0 else 0
    
    results = {
        'accuracy': accuracy,
        'mcc': mcc,
        'predictions': predictions,
        'ground_truth': ground_truth
    }
    
    return results

# 4. Basic LLM baseline (simulated)
def llm_baseline(test_dataset):
    """
    Simulate a basic LLM baseline.
    
    Args:
        test_dataset (pd.DataFrame): Test dataset
        
    Returns:
        dict: Evaluation metrics
    """
    # This simulates a generic LLM that can analyze sentiment but doesn't weigh mixed sentiment well
    
    predictions = []
    ground_truth = []
    explanations = []
    
    for _, row in test_dataset.iterrows():
        symbol = row['Symbol']
        tweets = row['PrevTweets']
        movement = row['Movement']
        
        # Count positive and negative sentiment in tweets
        pos_count = 0
        neg_count = 0
        
        for tweet in tweets:
            if any(term in tweet.lower() for term in ["better than expected", "upgraded", "buy rating", "new product", "partnership", "buyback"]):
                pos_count += 1
            
            if any(term in tweet.lower() for term in ["missed", "downgraded", "sell rating", "layoffs", "regulatory scrutiny"]):
                neg_count += 1
        
        # Generate prediction
        # Simulate LLM's tendency to be indecisive with mixed signals
        if abs(pos_count - neg_count) <= 2:
            # Indecisive response - count as incorrect
            pred = 0  # Default to negative for indecisive cases
            explanation = f"The sentiment for {symbol} is mixed, with both positive and negative factors present."
        else:
            # Decisive response
            pred = 1 if pos_count > neg_count else 0
            
            if pred == 1:
                explanation = f"The sentiment for {symbol} is positive, with more positive factors than negative factors."
            else:
                explanation = f"The sentiment for {symbol} is negative, with more negative factors than positive factors."
        
        predictions.append(pred)
        ground_truth.append(movement)
        explanations.append(explanation)
    
    # Calculate metrics
    accuracy = sum(p == g for p, g in zip(predictions, ground_truth)) / len(predictions)
    
    # Calculate MCC
    tp = sum(p == 1 and g == 1 for p, g in zip(predictions, ground_truth))
    tn = sum(p == 0 and g == 0 for p, g in zip(predictions, ground_truth))
    fp = sum(p == 1 and g == 0 for p, g in zip(predictions, ground_truth))
    fn = sum(p == 0 and g == 1 for p, g in zip(predictions, ground_truth))
    
    numerator = (tp * tn) - (fp * fn)
    denominator = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
    
    mcc = numerator / denominator if denominator != 0 else 0
    
    results = {
        'accuracy': accuracy,
        'mcc': mcc,
        'predictions': predictions,
        'ground_truth': ground_truth,
        'explanations': explanations
    }
    
    return results

# Run all baseline models
print("\nEvaluating baseline models...")
vae_results = vae_attention_baseline(train_dataset, test_dataset)
gru_results = gru_attention_baseline(train_dataset, test_dataset)
transformer_results = transformer_baseline(train_dataset, test_dataset)
llm_results = llm_baseline(test_dataset)

print("\nBaseline Model Evaluation:")
print(f"VAE+Attention: Accuracy = {vae_results['accuracy']:.4f}, MCC = {vae_results['mcc']:.4f}")
print(f"GRU+Attention: Accuracy = {gru_results['accuracy']:.4f}, MCC = {gru_results['mcc']:.4f}")
print(f"Transformer: Accuracy = {transformer_results['accuracy']:.4f}, MCC = {transformer_results['mcc']:.4f}")
print(f"Basic LLM: Accuracy = {llm_results['accuracy']:.4f}, MCC = {llm_results['mcc']:.4f}")
print(f"SEP (Ours): Accuracy = {sep_results['accuracy']:.4f}, MCC = {sep_results['mcc']:.4f}")

# Compare explanations between Basic LLM and SEP
print("\nComparison of Explanations:")
for i in range(3):
    idx = random.randint(0, len(sep_results['predictions']) - 1)
    symbol = test_dataset.iloc[idx]['Symbol']
    
    print(f"\nSymbol: {symbol}")
    print(f"Basic LLM Explanation: {llm_results['explanations'][idx]}")
    print(f"SEP Explanation: {sep_results['explanations'][idx]}")


class PortfolioConstructor:
    """
    Simulate the portfolio construction task using the SEP framework.
    """
    
    def __init__(self):
        # Initialize weights for different factors
        self.factor_weights = {
            "earnings": 0.3,
            "analyst_rating": 0.25,
            "new_product": 0.15,
            "partnership": 0.1,
            "market_sentiment": 0.2
        }
        
        # Initialize weights for different industries
        self.industry_weights = {
            "Technology": 1.2,
            "Healthcare": 0.9,
            "Financial": 0.8,
            "Consumer": 0.7,
            "Industrial": 0.6
        }
    
    def generate_portfolio_weights(self, positive_stocks, summarized_facts):
        """
        Generate portfolio weights for positive stocks.
        
        Args:
            positive_stocks (list): List of stocks with positive predictions
            summarized_facts (dict): Dictionary of summarized facts for each stock
            
        Returns:
            dict: Portfolio weights for each stock
        """
        if not positive_stocks:
            return {}
        
        # Calculate scores for each stock
        stock_scores = {}
        
        for symbol in positive_stocks:
            facts = summarized_facts.get(symbol, "")
            
            # Calculate score based on factors
            score = 0
            
            # Check for earnings-related information
            if "better than expected earnings" in facts.lower():
                score += self.factor_weights["earnings"] * 1.2
            elif "earnings" in facts.lower():
                score += self.factor_weights["earnings"] * 0.8
            
            # Check for analyst ratings
            if "upgraded" in facts.lower() or "buy rating" in facts.lower():
                score += self.factor_weights["analyst_rating"] * 1.2
            
            # Check for new products
            if "new product" in facts.lower():
                score += self.factor_weights["new_product"]
            
            # Check for partnerships
            if "partnership" in facts.lower():
                score += self.factor_weights["partnership"]
            
            # Check for overall sentiment
            if "positive sentiment" in facts.lower():
                score += self.factor_weights["market_sentiment"]
            
            # Adjust score based on industry
            industry = next((ind for ind, symbols in stocks.items() if symbol in symbols), None)
            score *= self.industry_weights.get(industry, 1.0)
            
            stock_scores[symbol] = max(0.01, score)  # Ensure minimum weight of 1%
        
        # Normalize scores to sum to 1
        total_score = sum(stock_scores.values())
        weights = {symbol: score / total_score for symbol, score in stock_scores.items()}
        
        return weights
    
    def generate_explanation(self, weights, summarized_facts):
        """
        Generate an explanation for the portfolio weights.
        
        Args:
            weights (dict): Portfolio weights
            summarized_facts (dict): Dictionary of summarized facts for each stock
            
        Returns:
            str: Explanation
        """
        explanation = "Portfolio weights are determined based on an analysis of each stock's fundamental and technical factors. "
        
        # Sort stocks by weight
        sorted_stocks = sorted(weights.items(), key=lambda x: x[1], reverse=True)
        
        # Explain weights for top stocks
        for symbol, weight in sorted_stocks[:3]:
            facts = summarized_facts.get(symbol, "")
            explanation += f"{symbol} is allocated {weight:.2f} of the portfolio. "
            
            # Explain factors for this stock
            reasons = []
            
            if "better than expected earnings" in facts.lower():
                reasons.append("strong earnings performance")
            
            if "upgraded" in facts.lower() or "buy rating" in facts.lower():
                reasons.append("positive analyst ratings")
            
            if "new product" in facts.lower():
                reasons.append("new product announcements")
            
            if "partnership" in facts.lower():
                reasons.append("strategic partnerships")
            
            if reasons:
                explanation += f"This is due to {', '.join(reasons)}. "
            
            # Add industry context
            industry = next((ind for ind, symbols in stocks.items() if symbol in symbols), None)
            if industry:
                explanation += f"As part of the {industry} sector, {symbol} provides exposure to {industry.lower()} trends. "
        
        explanation += "The remaining allocation is distributed among other stocks to maintain diversification while maximizing expected returns based on recent positive indicators."
        
        return explanation

def test_portfolio_construction():
    """
    Test the portfolio construction task.
    """
    # Get a set of test data points
    test_samples = test_dataset.sample(5)
    
    # Initialize portfolio constructor
    portfolio_constructor = PortfolioConstructor()
    
    # Track portfolio performance
    portfolio_performance = []
    
    # For each test point, create a portfolio and evaluate its performance
    for _, row in test_samples.iterrows():
        date = row['Date']
        symbol = row['Symbol']
        tweets = row['PrevTweets']
        
        # Get all stocks with data for this date
        date_stocks = test_dataset[test_dataset['Date'] == date]['Symbol'].unique().tolist()
        
        # Generate summarized facts for each stock
        summarized_facts = {}
        for stock in date_stocks:
            stock_tweets = test_dataset[(test_dataset['Date'] == date) & (test_dataset['Symbol'] == stock)]['PrevTweets'].values
            if len(stock_tweets) > 0:
                summarized_facts[stock] = summarize_tweets(stock_tweets[0], stock)
        
        # Predict which stocks will go up
        positive_stocks = []
        for stock in date_stocks:
            if stock in summarized_facts:
                price_movement, _, _ = sep_model.predict(summarized_facts[stock], stock)
                if price_movement == "Positive":
                    positive_stocks.append(stock)
        
        # Generate portfolio weights
        weights = portfolio_constructor.generate_portfolio_weights(positive_stocks, summarized_facts)
        
        # Generate explanation
        explanation = portfolio_constructor.generate_explanation(weights, summarized_facts)
        
        # Calculate next-day returns
        next_day = date + datetime.timedelta(days=1)
        while next_day.weekday() >= 5:  # Skip weekends
            next_day += datetime.timedelta(days=1)
        
        portfolio_return = 0
        portfolio_stocks = []
        
        for stock, weight in weights.items():
            # Get next-day return for this stock
            next_day_data = stock_data[(stock_data['Symbol'] == stock) & (stock_data['Date'] == next_day)]
            if len(next_day_data) > 0:
                stock_return = next_day_data['Return'].values[0]
                portfolio_return += weight * stock_return
                portfolio_stocks.append(stock)
        
        # Store results
        portfolio_performance.append({
            'Date': date,
            'PortfolioStocks': portfolio_stocks,
            'Weights': weights,
            'Return': portfolio_return,
            'Explanation': explanation
        })
    
    # Display results
    print("\nPortfolio Construction Results:")
    
    for result in portfolio_performance:
        print(f"\nDate: {result['Date'].strftime('%Y-%m-%d')}")
        print(f"Portfolio: {', '.join([f'{stock} ({result['Weights'][stock]:.2f})' for stock in result['PortfolioStocks']])}")
        print(f"Next-Day Return: {result['Return']:.4f}")
        print(f"Explanation: {result['Explanation']}")
    
    # Calculate overall performance
    avg_return = np.mean([result['Return'] for result in portfolio_performance])
    print(f"\nAverage Portfolio Return: {avg_return:.4f}")

# Test portfolio construction
test_portfolio_construction()