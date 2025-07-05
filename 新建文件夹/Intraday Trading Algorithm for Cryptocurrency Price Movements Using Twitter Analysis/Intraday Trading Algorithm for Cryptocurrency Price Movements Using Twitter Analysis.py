import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import tweepy
import time
import datetime
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from collections import Counter
import random
import seaborn as sns
from tqdm import tqdm

# Set random seed for reproducibility
np.random.seed(42)

class CryptoTwitterTradingStrategy:
    """
    A trading strategy that monitors cryptocurrency foundation Twitter accounts,
    analyzes tweets for specific conditions, and executes trades based on predefined criteria.
    """
    
    def __init__(self, spend_rate=0.5):
        """
        Initialize the trading strategy.
        
        Parameters:
        -----------
        spend_rate : float
            Percentage of available BTC to spend on each trade (0-1)
        """
        self.spend_rate = spend_rate
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        self.stop_words = set(stopwords.words('english'))
        
        # Beneficial word list (from paper findings)
        self.beneficial_words = [
            'announce', 'partnership', 'partnered', 'launches', 'finally', 
            'cooperation', 'binance', 'bitfinex', 'technology', 'update'
        ]
        
        # Load data structures
        self.load_crypto_mapping()
        self.load_condition_data()
        
        # Track performance
        self.portfolio = {'BTC': 1.0}  # Start with 1 BTC
        self.trade_history = []
        self.portfolio_history = []
    
    def load_crypto_mapping(self):
        """
        Load mapping between Twitter accounts and cryptocurrency symbols.
        In a real implementation, this would load from a file.
        """
        # Simulated mapping of Twitter accounts to cryptocurrency symbols
        self.crypto_mapping = {
            'ethereum': 'ETH',
            'litecoin': 'LTC',
            'cardano': 'ADA',
            'StellarOrg': 'XLM',
            'ripple': 'XRP',
            'VeChainOfficial': 'VET',
            'chainlink': 'LINK',
            'tezos': 'XTZ',
            'binance': 'BNB',
            'cosmos': 'ATOM'
        }
    
    def load_condition_data(self):
        """
        Load condition data for tweet analysis based on paper findings.
        """
        # Optimal buy and sell times based on paper findings
        self.buy_time = 1  # Buy after 1 minute (t1)
        self.sell_time = 10  # Sell after 10 minutes (t10) - peak cumulative return
        
        # Define trading conditions based on paper findings
        self.conditions = {
            'status_count': {
                'low': {'min': 0, 'max': 1500},  # Low status count (high quality)
                'medium': {'min': 1500, 'max': 2700}
            },
            'retweet': False,  # Not a retweet
            'quote': False,    # Not a quote
        }
    
    def analyze_tweet(self, tweet):
        """
        Analyze a tweet to determine if it meets the conditions for trading.
        
        Parameters:
        -----------
        tweet : dict
            Tweet data including text, user info, etc.
            
        Returns:
        --------
        dict or None
            Dictionary with analysis results if conditions are met, None otherwise
        """
        # Check if it's a retweet or quote
        if tweet['is_retweet'] or tweet['is_quote']:
            return None
        
        # Calculate sentiment
        sentiment = self.calculate_sentiment(tweet['text'])
        
        # Extract words
        words = self.extract_words(tweet['text'])
        
        # Check for beneficial words
        has_beneficial_words = any(word in self.beneficial_words for word in words)
        
        # Check status count condition
        status_count = tweet['user']['statuses_count']
        status_condition = False
        for status_range, limits in self.conditions['status_count'].items():
            if limits['min'] <= status_count <= limits['max']:
                status_condition = True
                break
        
        # Make trading decision
        if status_condition and has_beneficial_words:
            return {
                'symbol': self.crypto_mapping.get(tweet['user']['screen_name']),
                'sentiment': sentiment,
                'has_beneficial_words': has_beneficial_words,
                'status_count': status_count,
                'buy_time': self.buy_time,
                'sell_time': self.sell_time,
                'timestamp': tweet['created_at']
            }
        
        return None
    
    def calculate_sentiment(self, text):
        """
        Calculate sentiment score for a text.
        
        Parameters:
        -----------
        text : str
            Text to analyze
            
        Returns:
        --------
        float
            Sentiment score (-1 to 1)
        """
        sentiment = self.sentiment_analyzer.polarity_scores(text)
        return sentiment['compound']
    
    def extract_words(self, text):
        """
        Extract meaningful words from text by removing stopwords.
        
        Parameters:
        -----------
        text : str
            Text to process
            
        Returns:
        --------
        list
            List of meaningful words
        """
        # Remove URLs, mentions, special characters
        text = re.sub(r'http\S+', '', text)
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'[^\w\s]', '', text)
        
        # Tokenize and remove stopwords
        words = [word.lower() for word in text.split() 
                if word.lower() not in self.stop_words]
        
        return words
    
    def execute_trade(self, analysis, price_data):
        """
        Execute a trade based on the tweet analysis.
        
        Parameters:
        -----------
        analysis : dict
            Analysis results from analyze_tweet
        price_data : dict
            Price data for the cryptocurrency
            
        Returns:
        --------
        dict
            Trade details
        """
        symbol = analysis['symbol']
        if symbol not in self.crypto_mapping.values():
            return None
        
        # Get timestamps for buy and sell
        buy_timestamp = analysis['timestamp'] + datetime.timedelta(minutes=analysis['buy_time'])
        sell_timestamp = analysis['timestamp'] + datetime.timedelta(minutes=analysis['sell_time'])
        
        # Get prices at buy and sell times
        buy_price = self.get_price_at_time(price_data, symbol, buy_timestamp)
        sell_price = self.get_price_at_time(price_data, symbol, sell_timestamp)
        
        if buy_price is None or sell_price is None:
            return None
        
        # Calculate amount to buy
        btc_amount = self.portfolio.get('BTC', 0) * self.spend_rate
        coin_amount = btc_amount / buy_price
        
        # Execute buy
        if 'BTC' in self.portfolio:
            self.portfolio['BTC'] -= btc_amount
        if symbol in self.portfolio:
            self.portfolio[symbol] += coin_amount
        else:
            self.portfolio[symbol] = coin_amount
        
        # Record trade
        buy_trade = {
            'type': 'buy',
            'symbol': symbol,
            'price': buy_price,
            'amount': coin_amount,
            'btc_value': btc_amount,
            'timestamp': buy_timestamp
        }
        self.trade_history.append(buy_trade)
        
        # Execute sell after defined period
        btc_value = coin_amount * sell_price
        
        # Update portfolio
        self.portfolio[symbol] -= coin_amount
        if 'BTC' in self.portfolio:
            self.portfolio['BTC'] += btc_value
        else:
            self.portfolio['BTC'] = btc_value
        
        # Record trade
        sell_trade = {
            'type': 'sell',
            'symbol': symbol,
            'price': sell_price,
            'amount': coin_amount,
            'btc_value': btc_value,
            'timestamp': sell_timestamp,
            'profit_btc': btc_value - btc_amount,
            'profit_percent': ((btc_value / btc_amount) - 1) * 100
        }
        self.trade_history.append(sell_trade)
        
        # Return trade details
        return {
            'buy': buy_trade,
            'sell': sell_trade,
            'profit_btc': btc_value - btc_amount,
            'profit_percent': ((btc_value / btc_amount) - 1) * 100
        }
    
    def get_price_at_time(self, price_data, symbol, timestamp):
        """
        Get price of a cryptocurrency at a specific time.
        
        Parameters:
        -----------
        price_data : dict
            Price data for cryptocurrencies
        symbol : str
            Cryptocurrency symbol
        timestamp : datetime
            Timestamp to get price for
            
        Returns:
        --------
        float or None
            Price at the specified time
        """
        # In real implementation, this would query historical price data
        # For simulation, we'll use the price data passed in
        if symbol not in price_data:
            return None
        
        # Find closest timestamp
        timestamps = price_data[symbol]['timestamp']
        prices = price_data[symbol]['price']
        
        closest_idx = min(range(len(timestamps)), 
                         key=lambda i: abs(timestamps[i] - timestamp))
        
        return prices[closest_idx]
    
    def update_portfolio_history(self, timestamp):
        """
        Update portfolio history.
        
        Parameters:
        -----------
        timestamp : datetime
            Current timestamp
        """
        total_btc_value = self.get_portfolio_btc_value()
        self.portfolio_history.append({
            'timestamp': timestamp,
            'portfolio': self.portfolio.copy(),
            'btc_value': total_btc_value
        })
    
    def get_portfolio_btc_value(self, price_data=None, timestamp=None):
        """
        Calculate total portfolio value in BTC.
        
        Parameters:
        -----------
        price_data : dict, optional
            Price data for cryptocurrencies
        timestamp : datetime, optional
            Timestamp to get prices for
            
        Returns:
        --------
        float
            Total portfolio value in BTC
        """
        # For simulation with fixed prices
        if price_data is None or timestamp is None:
            btc_value = self.portfolio.get('BTC', 0)
            # In a real implementation, we would calculate the value of all coins
            # For simplicity, we'll just return the BTC amount
            return btc_value
        
        # Calculate total value in BTC
        btc_value = self.portfolio.get('BTC', 0)
        
        for symbol, amount in self.portfolio.items():
            if symbol != 'BTC' and amount > 0:
                price = self.get_price_at_time(price_data, symbol, timestamp)
                if price is not None:
                    btc_value += amount * price
        
        return btc_value
    
    def run_simulation(self, tweets, price_data, days=30):
        """
        Run a simulation of the trading strategy over a period of time.
        
        Parameters:
        -----------
        tweets : list
            List of tweets to analyze
        price_data : dict
            Price data for cryptocurrencies
        days : int, optional
            Number of days to simulate
            
        Returns:
        --------
        dict
            Simulation results
        """
        # Reset portfolio and history
        self.portfolio = {'BTC': 1.0}
        self.trade_history = []
        self.portfolio_history = []
        
        # Simulation start time
        start_time = datetime.datetime.now() - datetime.timedelta(days=days)
        end_time = datetime.datetime.now()
        
        # Filter tweets in the simulation period
        relevant_tweets = [t for t in tweets 
                          if start_time <= t['created_at'] <= end_time]
        
        # Sort tweets by timestamp
        relevant_tweets.sort(key=lambda x: x['created_at'])
        
        # Track current time and portfolio value
        current_time = start_time
        self.update_portfolio_history(current_time)
        
        # Process tweets
        for tweet in tqdm(relevant_tweets, desc="Processing tweets"):
            # Update time
            current_time = tweet['created_at']
            
            # Analyze tweet
            analysis = self.analyze_tweet(tweet)
            
            if analysis is not None:
                # Execute trade
                trade = self.execute_trade(analysis, price_data)
                
                # Update portfolio history after trade
                if trade is not None:
                    self.update_portfolio_history(trade['sell']['timestamp'])
        
        # Final portfolio update
        self.update_portfolio_history(end_time)
        
        # Calculate performance metrics
        initial_value = self.portfolio_history[0]['btc_value']
        final_value = self.portfolio_history[-1]['btc_value']
        total_return = ((final_value / initial_value) - 1) * 100
        
        # Count profitable trades
        profitable_trades = sum(1 for trade in self.trade_history 
                               if trade.get('type') == 'sell' and trade.get('profit_btc', 0) > 0)
        total_trades = sum(1 for trade in self.trade_history if trade.get('type') == 'sell')
        
        win_rate = (profitable_trades / total_trades * 100) if total_trades > 0 else 0
        
        return {
            'initial_value': initial_value,
            'final_value': final_value,
            'total_return': total_return,
            'total_trades': total_trades,
            'profitable_trades': profitable_trades,
            'win_rate': win_rate,
            'portfolio_history': self.portfolio_history,
            'trade_history': self.trade_history
        }

# Function to generate simulated cryptocurrency price data
def generate_simulated_price_data(crypto_symbols, days=30, volatile=True):
    """
    Generate simulated price data for cryptocurrencies.
    
    Parameters:
    -----------
    crypto_symbols : list
        List of cryptocurrency symbols
    days : int, optional
        Number of days of data to generate
    volatile : bool, optional
        Whether to make prices more volatile
        
    Returns:
    --------
    dict
        Simulated price data
    """
    # Calculate number of minutes
    minutes = days * 24 * 60
    
    # Base timestamp
    base_timestamp = datetime.datetime.now() - datetime.timedelta(days=days)
    
    # Generate price data for each cryptocurrency
    price_data = {}
    
    for symbol in crypto_symbols:
        # Generate timestamps at 1-minute intervals
        timestamps = [base_timestamp + datetime.timedelta(minutes=i) 
                     for i in range(minutes)]
        
        # Generate base price
        base_price = np.random.uniform(0.001, 0.1)  # Base price in BTC
        
        # Generate price movements
        if volatile:
            # More volatile for realistic crypto behavior
            volatility = np.random.uniform(0.05, 0.15)
            drift = np.random.uniform(-0.0001, 0.0002)  # Slight upward drift on average
        else:
            volatility = np.random.uniform(0.01, 0.05)
            drift = np.random.uniform(-0.00005, 0.0001)
        
        # Generate price series using Geometric Brownian Motion
        returns = np.random.normal(drift, volatility, minutes)
        
        # Add some jumps to simulate tweets impact
        num_jumps = int(minutes / (60 * 24))  # Approximately one jump per day
        jump_indices = np.random.choice(range(minutes), num_jumps, replace=False)
        
        for idx in jump_indices:
            # 70% chance of positive jump
            if np.random.random() < 0.7:
                returns[idx] += np.random.uniform(0.005, 0.03)
            else:
                returns[idx] -= np.random.uniform(0.005, 0.02)
        
        # Convert returns to prices
        prices = base_price * np.cumprod(1 + returns)
        
        # Store data
        price_data[symbol] = {
            'timestamp': timestamps,
            'price': prices.tolist()
        }
    
    return price_data

# Function to generate simulated tweets
def generate_simulated_tweets(crypto_accounts, days=30):
    """
    Generate simulated tweets from cryptocurrency foundation accounts.
    
    Parameters:
    -----------
    crypto_accounts : list
        List of cryptocurrency Twitter accounts
    days : int, optional
        Number of days of tweets to generate
        
    Returns:
    --------
    list
        Simulated tweets
    """
    # Base timestamp
    base_timestamp = datetime.datetime.now() - datetime.timedelta(days=days)
    
    # Templates for tweets
    announcement_templates = [
        "We're excited to announce our partnership with {partner}!",
        "Big news! {symbol} is now listed on {exchange}!",
        "Exciting update! We've just launched {feature}.",
        "We're thrilled to announce the release of our new {technology}.",
        "Important announcement: {symbol} network upgrade is now live!",
        "We've partnered with {partner} to bring you {benefit}.",
        "New milestone reached! {achievement}.",
        "Today we're launching our new {product}.",
        "Breaking news! {news}",
        "We're happy to announce our cooperation with {partner}."
    ]
    
    regular_templates = [
        "Check out our latest blog post on {topic}.",
        "Thanks to our amazing community for the support!",
        "Don't forget to join our AMA session tomorrow!",
        "The future of {symbol} is looking bright!",
        "Happy Friday from the {symbol} team!",
        "We're attending {conference} next week. Come say hi!",
        "Here's a quick tutorial on how to use {feature}.",
        "What's your favorite feature of {symbol}?",
        "The {symbol} community is growing every day!",
        "How are you using {symbol} today?"
    ]
    
    # Partners, exchanges, features for templates
    partners = ["Microsoft", "IBM", "Amazon", "Google", "Deloitte", 
               "PWC", "KPMG", "Accenture", "JP Morgan", "Goldman Sachs"]
    
    exchanges = ["Binance", "Coinbase", "Kraken", "Bitfinex", "Gemini", 
                "Huobi", "OKEx", "Bitstamp", "Bittrex", "KuCoin"]
    
    features = ["smart contracts", "staking", "governance voting", "cross-chain bridge", 
               "DEX integration", "layer 2 solution", "wallet update", "mobile app", 
               "API improvements", "security features"]
    
    technologies = ["blockchain", "protocol", "consensus algorithm", "scaling solution", 
                   "interoperability protocol", "privacy solution", "token standard", 
                   "smart contract platform", "DeFi protocol", "NFT marketplace"]
    
    # Generate tweets
    tweets = []
    
    for account in crypto_accounts:
        # Determine frequency (1-5 tweets per day on average)
        daily_tweets = np.random.randint(1, 6)
        
        # Status count (used for quality indicator)
        status_count = np.random.randint(100, 5000)
        
        # Followers count
        followers_count = np.random.randint(10000, 1000000)
        
        # Generate tweets for this account
        account_tweets_count = int(days * daily_tweets)
        
        for i in range(account_tweets_count):
            # Generate timestamp
            timestamp = base_timestamp + datetime.timedelta(
                days=np.random.uniform(0, days),
                hours=np.random.randint(0, 24),
                minutes=np.random.randint(0, 60)
            )
            
            # Determine if it's an announcement (20% chance)
            is_announcement = np.random.random() < 0.2
            
            if is_announcement:
                template = np.random.choice(announcement_templates)
            else:
                template = np.random.choice(regular_templates)
            
            # Fill in template
            symbol = account.upper()
            partner = np.random.choice(partners)
            exchange = np.random.choice(exchanges)
            feature = np.random.choice(features)
            technology = np.random.choice(technologies)
            topic = np.random.choice(["DeFi", "NFTs", "Web3", "Blockchain", "Crypto"])
            conference = f"Blockchain Expo {np.random.randint(2022, 2025)}"
            achievement = f"Over {np.random.randint(1, 10)}M users!"
            product = np.random.choice(["wallet", "dApp", "platform", "API"])
            news = f"Major {symbol} upgrade incoming!"
            benefit = np.random.choice(["faster transactions", "lower fees", "better security"])
            
            text = template.format(
                symbol=symbol,
                partner=partner,
                exchange=exchange,
                feature=feature,
                technology=technology,
                topic=topic,
                conference=conference,
                achievement=achievement,
                product=product,
                news=news,
                benefit=benefit
            )
            
            # Determine if it's a retweet or quote (30% chance)
            is_retweet = np.random.random() < 0.15
            is_quote = np.random.random() < 0.15 and not is_retweet
            
            # Create tweet object
            tweet = {
                'created_at': timestamp,
                'text': text,
                'is_retweet': is_retweet,
                'is_quote': is_quote,
                'user': {
                    'screen_name': account,
                    'statuses_count': status_count + i,  # Increment for each tweet
                    'followers_count': followers_count,
                    'friends_count': np.random.randint(100, 5000),
                    'favourites_count': np.random.randint(100, 10000)
                }
            }
            
            tweets.append(tweet)
    
    # Sort tweets by timestamp
    tweets.sort(key=lambda x: x['created_at'])
    
    return tweets

# Function to analyze the results of the simulation
def analyze_results(results):
    """
    Analyze and visualize the results of the trading simulation.
    
    Parameters:
    -----------
    results : dict
        Simulation results
    """
    # Print summary
    print(f"Initial Portfolio Value: {results['initial_value']:.8f} BTC")
    print(f"Final Portfolio Value: {results['final_value']:.8f} BTC")
    print(f"Total Return: {results['total_return']:.2f}%")
    print(f"Total Trades: {results['total_trades']}")
    print(f"Profitable Trades: {results['profitable_trades']}")
    print(f"Win Rate: {results['win_rate']:.2f}%")
    
    # Extract portfolio history
    timestamps = [entry['timestamp'] for entry in results['portfolio_history']]
    btc_values = [entry['btc_value'] for entry in results['portfolio_history']]
    
    # Plot portfolio value over time
    plt.figure(figsize=(12, 6))
    plt.plot(timestamps, btc_values, 'b-', linewidth=2)
    plt.title('Portfolio Value Over Time (BTC)')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value (BTC)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('portfolio_value.png')
    plt.show()
    
    # Analyze trade profitability
    if results['total_trades'] > 0:
        trade_profits = [trade['profit_percent'] 
                        for trade in results['trade_history'] 
                        if trade.get('type') == 'sell']
        
        trade_symbols = [trade['symbol'] 
                        for trade in results['trade_history'] 
                        if trade.get('type') == 'sell']
        
        # Plot trade profitability
        plt.figure(figsize=(12, 6))
        plt.bar(range(len(trade_profits)), trade_profits, color='green')
        plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
        plt.title('Profit/Loss per Trade (%)')
        plt.xlabel('Trade Number')
        plt.ylabel('Profit/Loss (%)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('trade_profitability.png')
        plt.show()
        
        # Plot profitability by cryptocurrency
        if len(set(trade_symbols)) > 1:
            symbol_profits = {}
            for i, symbol in enumerate(trade_symbols):
                if symbol not in symbol_profits:
                    symbol_profits[symbol] = []
                symbol_profits[symbol].append(trade_profits[i])
            
            avg_profits = {symbol: np.mean(profits) for symbol, profits in symbol_profits.items()}
            
            plt.figure(figsize=(12, 6))
            plt.bar(avg_profits.keys(), avg_profits.values(), color='blue')
            plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
            plt.title('Average Profit/Loss by Cryptocurrency (%)')
            plt.xlabel('Cryptocurrency')
            plt.ylabel('Average Profit/Loss (%)')
            plt.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig('profit_by_crypto.png')
            plt.show()
    
    # Analyze tweet conditions that led to trades
    trade_tweets = []
    for trade in results['trade_history']:
        if trade.get('type') == 'buy':
            # Find the tweet that triggered this trade
            for tweet in results.get('tweets_analyzed', []):
                if (tweet.get('symbol') == trade['symbol'] and 
                    abs((tweet['timestamp'] - trade['timestamp']).total_seconds()) < 120):
                    trade_tweets.append({
                        'tweet': tweet,
                        'profit': next((t['profit_percent'] for t in results['trade_history'] 
                                      if t.get('type') == 'sell' and 
                                      t.get('timestamp') == trade['timestamp'] + 
                                      datetime.timedelta(minutes=9)), None)
                    })
                    break
    
    if trade_tweets:
        # Analyze words in profitable tweets
        profitable_tweets = [entry['tweet']['text'] 
                            for entry in trade_tweets 
                            if entry['profit'] and entry['profit'] > 0]
        
        if profitable_tweets:
            # Extract words
            words = []
            for tweet in profitable_tweets:
                words.extend([word.lower() for word in re.findall(r'\b\w+\b', tweet)
                             if word.lower() not in stopwords.words('english')])
            
            # Count word frequency
            word_counts = Counter(words)
            
            # Plot top 20 words
            top_words = word_counts.most_common(20)
            
            plt.figure(figsize=(12, 6))
            plt.bar([word for word, count in top_words], 
                   [count for word, count in top_words], 
                   color='orange')
            plt.title('Most Common Words in Profitable Tweets')
            plt.xlabel('Word')
            plt.ylabel('Frequency')
            plt.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig('profitable_words.png')
            plt.show()

# Main simulation function
def run_crypto_twitter_simulation(days=30, volatile=True):
    """
    Run a simulation of the cryptocurrency Twitter trading strategy.
    
    Parameters:
    -----------
    days : int, optional
        Number of days to simulate
    volatile : bool, optional
        Whether to make prices more volatile
        
    Returns:
    --------
    dict
        Simulation results
    """
    # Define cryptocurrency accounts and symbols
    crypto_mapping = {
        'ethereum': 'ETH',
        'litecoin': 'LTC',
        'cardano': 'ADA',
        'StellarOrg': 'XLM',
        'ripple': 'XRP',
        'VeChainOfficial': 'VET',
        'chainlink': 'LINK',
        'tezos': 'XTZ',
        'binance': 'BNB',
        'cosmos': 'ATOM'
    }
    
    # Generate simulated price data
    price_data = generate_simulated_price_data(
        list(crypto_mapping.values()), days=days, volatile=volatile
    )
    
    # Generate simulated tweets
    tweets = generate_simulated_tweets(list(crypto_mapping.keys()), days=days)
    
    # Initialize trading strategy
    strategy = CryptoTwitterTradingStrategy(spend_rate=0.5)
    
    # Run simulation
    print(f"Running simulation for {days} days with {len(tweets)} tweets...")
    results = strategy.run_simulation(tweets, price_data, days=days)
    
    # Add tweets to results for analysis
    results['tweets_analyzed'] = [strategy.analyze_tweet(tweet) 
                                 for tweet in tweets 
                                 if strategy.analyze_tweet(tweet) is not None]
    
    # Analyze results
    analyze_results(results)
    
    return results

# Run the simulation
if __name__ == "__main__":
    # Run a 30-day simulation with volatile prices
    results = run_crypto_twitter_simulation(days=30, volatile=True)
    
    # Compare with a baseline simulation (no trading)
    print("\nComparing with baseline (no trading)...")
    baseline = {'initial_value': 1.0, 'final_value': 1.0, 'total_return': 0.0}
    
    if results['total_return'] > 0:
        print(f"Strategy outperformed baseline by {results['total_return']:.2f}%")
    else:
        print(f"Strategy underperformed baseline by {abs(results['total_return']):.2f}%")
    
    # Run additional analysis on the timing
    if results['trade_history']:
        buy_to_sell_times = []
        for i in range(0, len(results['trade_history']), 2):
            if i+1 < len(results['trade_history']):
                buy_time = results['trade_history'][i]['timestamp']
                sell_time = results['trade_history'][i+1]['timestamp']
                time_diff = (sell_time - buy_time).total_seconds() / 60  # in minutes
                buy_to_sell_times.append(time_diff)
        
        if buy_to_sell_times:
            print(f"\nAverage time between buy and sell: {np.mean(buy_to_sell_times):.2f} minutes")
            
            # Compare profitability by hold duration
            plt.figure(figsize=(10, 6))
            plt.scatter(buy_to_sell_times, 
                      [trade['profit_percent'] for trade in results['trade_history'] 
                       if trade.get('type') == 'sell'],
                      alpha=0.7)
            plt.title('Profit vs. Hold Duration')
            plt.xlabel('Hold Duration (minutes)')
            plt.ylabel('Profit (%)')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig('profit_vs_duration.png')
            plt.show()