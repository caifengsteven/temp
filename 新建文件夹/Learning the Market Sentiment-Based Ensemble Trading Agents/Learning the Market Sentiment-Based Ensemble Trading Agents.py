import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import random
import gym
from gym import spaces
from stable_baselines3 import DDPG, A2C, PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from datetime import datetime, timedelta
import afinn
import nltk
from nltk.tokenize import word_tokenize
import warnings
warnings.filterwarnings('ignore')

# Download NLTK resources
nltk.download('punkt', quiet=True)

class StockTradingEnv(gym.Env):
    """Custom Stock Trading Environment that follows gym interface"""
    
    metadata = {'render.modes': ['human']}
    
    def __init__(self, df, initial_balance=10000, transaction_cost_pct=0.001, window_size=10):
        super(StockTradingEnv, self).__init__()
        self.df = df
        self.initial_balance = initial_balance
        self.transaction_cost_pct = transaction_cost_pct
        self.window_size = window_size
        
        # Initialize state
        self.reset()
        
        # Action space: [0, 1] representing the percentage of balance to invest
        # 0 = sell all, 0.5 = hold, 1 = buy with all available balance
        self.action_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        
        # Observation space: [balance, shares, price_history]
        price_space = spaces.Box(low=0, high=np.inf, shape=(window_size,), dtype=np.float32)
        balance_space = spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32)
        shares_space = spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32)
        
        self.observation_space = spaces.Dict({
            'balance': balance_space,
            'shares': shares_space,
            'prices': price_space
        })
    
    def _get_observation(self):
        # Get the price history window
        end_idx = min(self.current_step + 1, len(self.df))
        start_idx = max(0, end_idx - self.window_size)
        prices = self.df['Close'].iloc[start_idx:end_idx].values
        
        # Pad with zeros if needed
        if len(prices) < self.window_size:
            prices = np.pad(prices, (self.window_size - len(prices), 0), 'constant')
        
        return {
            'balance': np.array([self.balance], dtype=np.float32),
            'shares': np.array([self.shares], dtype=np.float32),
            'prices': prices.astype(np.float32)
        }
    
    def _calculate_portfolio_value(self):
        return self.balance + self.shares * self.current_price
    
    def step(self, action):
        # Get current price
        self.current_price = self.df['Close'].iloc[self.current_step]
        previous_portfolio_value = self._calculate_portfolio_value()
        
        # Execute action (0 = sell all, 0.5 = hold, 1 = buy all)
        action_value = action[0]
        
        if action_value > 0.55:  # Buy
            # Calculate maximum shares that can be bought
            max_shares_to_buy = self.balance / (self.current_price * (1 + self.transaction_cost_pct))
            # Buy action_value percentage of max_shares
            shares_to_buy = max_shares_to_buy * (action_value - 0.5) * 2
            
            # Update balance and shares
            cost = shares_to_buy * self.current_price * (1 + self.transaction_cost_pct)
            self.balance -= cost
            self.shares += shares_to_buy
            
        elif action_value < 0.45:  # Sell
            # Calculate shares to sell
            shares_to_sell = self.shares * (0.5 - action_value) * 2
            
            # Update balance and shares
            revenue = shares_to_sell * self.current_price * (1 - self.transaction_cost_pct)
            self.balance += revenue
            self.shares -= shares_to_sell
        
        # Move to next step
        self.current_step += 1
        done = self.current_step >= len(self.df) - 1
        
        # Calculate reward (change in portfolio value)
        current_portfolio_value = self._calculate_portfolio_value()
        reward = (current_portfolio_value - previous_portfolio_value) / previous_portfolio_value
        
        # Get new observation
        obs = self._get_observation()
        
        # Store portfolio value history
        self.portfolio_value_history.append(current_portfolio_value)
        
        return obs, reward, done, {'portfolio_value': current_portfolio_value}
    
    def reset(self):
        self.balance = self.initial_balance
        self.shares = 0
        self.current_step = 0
        self.portfolio_value_history = [self.initial_balance]
        return self._get_observation()
    
    def render(self, mode='human'):
        # Implement visualization if needed
        pass

class SentimentAnalyzer:
    """Extract sentiment from financial news headlines using AFINN lexicon"""
    
    def __init__(self):
        self.afinn = afinn.Afinn(language='en')
    
    def get_sentiment(self, headline):
        """Calculate sentiment score for a single headline"""
        words = word_tokenize(headline.lower())
        total_score = 0
        count = 0
        
        for word in words:
            score = self.afinn.score(word)
            if score != 0:  # Only count words that have sentiment
                total_score += score
                count += 1
        
        if count > 0:
            return total_score / count
        else:
            return 0
    
    def get_period_sentiment(self, headlines):
        """Calculate average sentiment for a list of headlines"""
        if not headlines:
            return 0
            
        sentiments = [self.get_sentiment(headline) for headline in headlines]
        return sum(sentiments) / len(sentiments)

class SentimentEnsembleTrader:
    """Sentiment-based ensemble trading strategy"""
    
    def __init__(self, env, sentiment_threshold=2.0, alpha=0.25):
        self.env = env
        self.sentiment_threshold = sentiment_threshold  # Threshold for switching agents
        self.alpha = alpha  # Weight for Sharpe ratio in validation metric
        
        # Initialize agents
        self.agents = {
            'DDPG': DDPG('MlpPolicy', env, verbose=0),
            'A2C': A2C('MlpPolicy', env, verbose=0),
            'PPO': PPO('MlpPolicy', env, verbose=0)
        }
        
        self.current_agent_name = None
        self.current_agent = None
        self.previous_sentiment = 0
        self.sentiment_history = []
        
    def train_agents(self, total_timesteps=10000):
        """Train all agents for the specified number of timesteps"""
        for name, agent in self.agents.items():
            print(f"Training {name} agent...")
            agent.learn(total_timesteps=total_timesteps)
            print(f"{name} agent training completed")
    
    def validate_agents(self, validation_data, period_sentiments=None):
        """Validate all agents and select the best one based on the validation metric"""
        validation_env = DummyVecEnv([lambda: StockTradingEnv(validation_data)])
        
        best_score = float('-inf')
        best_agent_name = None
        
        for name, agent in self.agents.items():
            # Evaluate agent on validation data
            returns = []
            portfolio_values = []
            
            obs = validation_env.reset()
            done = False
            
            while not done:
                action, _states = agent.predict(obs)
                obs, reward, done, info = validation_env.step(action)
                returns.append(reward[0])
                portfolio_values.append(info[0]['portfolio_value'])
            
            # Calculate performance metrics
            returns = np.array(returns)
            sharpe_ratio = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
            
            # Calculate Sortino ratio (only penalizing negative returns)
            negative_returns = returns[returns < 0]
            sortino_ratio = np.mean(returns) / np.std(negative_returns) if len(negative_returns) > 0 and np.std(negative_returns) > 0 else 0
            
            # Calculate combined score
            score = self.alpha * sharpe_ratio + (1 - self.alpha) * sortino_ratio
            
            print(f"{name} Validation - Sharpe: {sharpe_ratio:.4f}, Sortino: {sortino_ratio:.4f}, Score: {score:.4f}")
            
            if score > best_score:
                best_score = score
                best_agent_name = name
        
        print(f"Best agent: {best_agent_name} with score: {best_score:.4f}")
        return best_agent_name
    
    def update_agent(self, current_sentiment, validation_data):
        """Update agent based on sentiment shift if needed"""
        # Check if sentiment has changed significantly
        sentiment_shift = abs(current_sentiment - self.previous_sentiment)
        self.sentiment_history.append(current_sentiment)
        
        if sentiment_shift > self.sentiment_threshold or self.current_agent is None:
            print(f"Significant sentiment shift detected: {sentiment_shift:.2f} > {self.sentiment_threshold}")
            # Revalidate and select best agent
            best_agent_name = self.validate_agents(validation_data)
            self.current_agent_name = best_agent_name
            self.current_agent = self.agents[best_agent_name]
            print(f"Switched to {best_agent_name} agent")
        else:
            print(f"Sentiment shift ({sentiment_shift:.2f}) below threshold, continuing with {self.current_agent_name} agent")
        
        self.previous_sentiment = current_sentiment
    
    def trade(self, obs):
        """Generate trading action using the current agent"""
        if self.current_agent is None:
            raise ValueError("No agent selected. Call update_agent first.")
        
        action, _states = self.current_agent.predict(obs)
        return action

def generate_simulated_data(start_date='2010-01-01', end_date='2019-01-01', ticker='AAPL'):
    """Generate or download stock data for backtesting"""
    try:
        # Try to download real data
        data = yf.download(ticker, start=start_date, end=end_date)
        print(f"Downloaded real data for {ticker} from {start_date} to {end_date}")
    except:
        # Generate simulated data if download fails
        print(f"Generating simulated data from {start_date} to {end_date}")
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')
        days = (end - start).days
        
        # Generate dates
        dates = [start + timedelta(days=i) for i in range(days)]
        dates = [date for date in dates if date.weekday() < 5]  # Only business days
        
        # Generate prices (random walk with drift)
        price = 100
        prices = [price]
        for _ in range(len(dates) - 1):
            change = np.random.normal(0.0005, 0.015)  # Slight positive drift
            price = price * (1 + change)
            prices.append(price)
        
        # Create DataFrame
        data = pd.DataFrame({
            'Open': prices,
            'High': [p * (1 + abs(np.random.normal(0, 0.005))) for p in prices],
            'Low': [p * (1 - abs(np.random.normal(0, 0.005))) for p in prices],
            'Close': prices,
            'Adj Close': prices,
            'Volume': [int(np.random.normal(1000000, 200000)) for _ in prices]
        }, index=dates)
    
    return data

def generate_simulated_news(start_date='2010-01-01', end_date='2019-01-01', volatility=0.8):
    """Generate simulated financial news headlines with sentiment"""
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')
    days = (end - start).days
    
    # Generate dates
    dates = [start + timedelta(days=i) for i in range(days)]
    dates = [date for date in dates if date.weekday() < 5]  # Only business days
    
    # Templates for headlines with different sentiments
    positive_templates = [
        "Markets rally as {sector} stocks surge",
        "Investor confidence increases as {company} reports strong earnings",
        "Economic outlook brightens with new {economic_indicator} data",
        "Stocks climb on positive {news_type} news",
        "{company} exceeds expectations, shares jump"
    ]
    
    negative_templates = [
        "Markets tumble amid {sector} concerns",
        "Investors flee as {company} misses projections",
        "Economic uncertainty grows with disappointing {economic_indicator}",
        "Stocks slide on troubling {news_type} news",
        "{company} struggles with declining revenue, shares drop"
    ]
    
    neutral_templates = [
        "{company} reports quarterly results",
        "Analysts review impact of {economic_indicator} on markets",
        "{sector} stocks see mixed trading day",
        "Market participants await {news_type} announcement",
        "Trading volume normal as investors assess {sector} outlook"
    ]
    
    # Fill-in variables
    sectors = ["technology", "healthcare", "finance", "energy", "retail", "manufacturing"]
    companies = ["Apple", "Amazon", "Microsoft", "Google", "JP Morgan", "Walmart", "ExxonMobil"]
    economic_indicators = ["GDP", "unemployment", "inflation", "consumer confidence", "manufacturing output"]
    news_types = ["economic", "earnings", "regulatory", "international", "Federal Reserve"]
    
    # Generate headlines with sentiment scores for each day
    daily_headlines = []
    sentiment_trend = 0  # Starting sentiment
    
    for date in dates:
        # Update sentiment trend (random walk with mean reversion)
        sentiment_trend = sentiment_trend * 0.95 + np.random.normal(0, volatility)
        
        # Generate headlines for the day based on sentiment trend
        num_headlines = random.randint(10, 20)  # Random number of headlines per day
        headlines = []
        
        for _ in range(num_headlines):
            # Determine if headline is positive, negative, or neutral based on trend
            r = np.random.normal(sentiment_trend, 1)
            
            if r > 0.5:
                template = random.choice(positive_templates)
            elif r < -0.5:
                template = random.choice(negative_templates)
            else:
                template = random.choice(neutral_templates)
            
            # Fill in template
            headline = template.format(
                sector=random.choice(sectors),
                company=random.choice(companies),
                economic_indicator=random.choice(economic_indicators),
                news_type=random.choice(news_types)
            )
            
            headlines.append(headline)
        
        daily_headlines.append({
            'date': date,
            'headlines': headlines,
            'true_sentiment': sentiment_trend  # Store the true sentiment for evaluation
        })
    
    return daily_headlines

def backtest_strategy(data, news_data, train_end_date='2016-12-31'):
    """Backtest the sentiment-based ensemble trading strategy"""
    # Split data into training and testing
    train_data = data[data.index <= train_end_date].copy()
    test_data = data[data.index > train_end_date].copy()
    
    # Split news data
    train_news = [day for day in news_data if day['date'] <= datetime.strptime(train_end_date, '%Y-%m-%d')]
    test_news = [day for day in news_data if day['date'] > datetime.strptime(train_end_date, '%Y-%m-%d')]
    
    # Create environments
    train_env = DummyVecEnv([lambda: StockTradingEnv(train_data)])
    validation_env = DummyVecEnv([lambda: StockTradingEnv(train_data.iloc[-252:])])  # Last year of training data for validation
    test_env = DummyVecEnv([lambda: StockTradingEnv(test_data)])
    
    # Initialize sentiment analyzer
    sentiment_analyzer = SentimentAnalyzer()
    
    # Initialize and train the ensemble trader
    ensemble_trader = SentimentEnsembleTrader(train_env, sentiment_threshold=2.0, alpha=0.25)
    ensemble_trader.train_agents(total_timesteps=20000)
    
    # Calculate period sentiments for test data (2-month periods)
    test_period_length = 60  # ~2 months of trading days
    test_periods = []
    
    for i in range(0, len(test_news), test_period_length):
        period_news = test_news[i:i+test_period_length]
        period_headlines = [headline for day in period_news for headline in day['headlines']]
        sentiment = sentiment_analyzer.get_period_sentiment(period_headlines)
        
        start_date = period_news[0]['date'] if period_news else None
        end_date = period_news[-1]['date'] if period_news else None
        
        test_periods.append({
            'start_date': start_date,
            'end_date': end_date,
            'sentiment': sentiment
        })
    
    # Backtest on test data
    print("Starting backtest on test data...")
    
    # Initialize metrics
    portfolio_values = []
    active_agents = []
    sentiments = []
    
    # Set up validation data
    validation_data = train_data.iloc[-252:].copy()
    
    # Initial agent selection
    initial_headlines = [headline for day in train_news[-test_period_length:] for headline in day['headlines']]
    initial_sentiment = sentiment_analyzer.get_period_sentiment(initial_headlines)
    ensemble_trader.update_agent(initial_sentiment, validation_data)
    
    # Run backtest
    obs = test_env.reset()
    done = False
    
    current_period_idx = 0
    days_in_period = 0
    
    while not done:
        # Check if we need to update the agent (new period)
        if days_in_period >= test_period_length and current_period_idx < len(test_periods) - 1:
            current_period_idx += 1
            days_in_period = 0
            
            # Update agent based on new period sentiment
            current_sentiment = test_periods[current_period_idx]['sentiment']
            print(f"\nNew period: {test_periods[current_period_idx]['start_date']} to {test_periods[current_period_idx]['end_date']}")
            print(f"Period sentiment: {current_sentiment:.2f}")
            
            ensemble_trader.update_agent(current_sentiment, validation_data)
            sentiments.append(current_sentiment)
        
        # Generate action and step environment
        action = ensemble_trader.trade(obs)
        obs, reward, done, info = test_env.step(action)
        
        # Record metrics
        portfolio_values.append(info[0]['portfolio_value'])
        active_agents.append(ensemble_trader.current_agent_name)
        
        days_in_period += 1
    
    # Calculate performance metrics
    initial_value = portfolio_values[0]
    final_value = portfolio_values[-1]
    
    daily_returns = np.diff(portfolio_values) / portfolio_values[:-1]
    
    cumulative_return = (final_value - initial_value) / initial_value
    annual_return = (1 + cumulative_return) ** (252 / len(portfolio_values)) - 1
    
    sharpe_ratio = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252)
    
    negative_returns = daily_returns[daily_returns < 0]
    sortino_ratio = np.mean(daily_returns) / np.std(negative_returns) * np.sqrt(252) if len(negative_returns) > 0 else float('inf')
    
    max_drawdown = 0
    peak = portfolio_values[0]
    for value in portfolio_values:
        if value > peak:
            peak = value
        drawdown = (peak - value) / peak
        max_drawdown = max(max_drawdown, drawdown)
    
    # Print performance metrics
    print("\nPerformance Metrics:")
    print(f"Cumulative Return: {cumulative_return:.2%}")
    print(f"Annual Return: {annual_return:.2%}")
    print(f"Maximum Drawdown: {max_drawdown:.2%}")
    print(f"Annual Volatility: {np.std(daily_returns) * np.sqrt(252):.2%}")
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    print(f"Sortino Ratio: {sortino_ratio:.2f}")
    
    # Plot results
    plt.figure(figsize=(14, 10))
    
    # Plot portfolio value
    plt.subplot(3, 1, 1)
    plt.plot(portfolio_values)
    plt.title('Portfolio Value Over Time')
    plt.xlabel('Trading Days')
    plt.ylabel('Portfolio Value ($)')
    plt.grid(True)
    
    # Plot active agent
    plt.subplot(3, 1, 2)
    agent_changes = [i for i in range(1, len(active_agents)) if active_agents[i] != active_agents[i-1]]
    agent_change_x = [0] + agent_changes + [len(active_agents)-1]
    agent_change_y = [active_agents[i] for i in agent_change_x]
    
    for i in range(len(agent_change_x)-1):
        plt.axvspan(agent_change_x[i], agent_change_x[i+1], alpha=0.2, 
                    color='green' if agent_change_y[i] == 'DDPG' else 'blue' if agent_change_y[i] == 'A2C' else 'red')
        plt.text((agent_change_x[i] + agent_change_x[i+1])/2, 0.5, agent_change_y[i], 
                 horizontalalignment='center', verticalalignment='center')
    
    plt.title('Active Agent Over Time')
    plt.xlabel('Trading Days')
    plt.yticks([])
    plt.grid(True)
    
    # Plot sentiment
    plt.subplot(3, 1, 3)
    sentiment_x = np.linspace(0, len(portfolio_values), len(sentiments))
    plt.plot(sentiment_x, sentiments)
    plt.axhline(y=ensemble_trader.sentiment_threshold, color='r', linestyle='--', label='Threshold')
    plt.axhline(y=-ensemble_trader.sentiment_threshold, color='r', linestyle='--')
    plt.title('Market Sentiment Over Time')
    plt.xlabel('Trading Days')
    plt.ylabel('Sentiment Score')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('sentiment_ensemble_results.png')
    plt.show()
    
    return {
        'portfolio_values': portfolio_values,
        'active_agents': active_agents,
        'sentiments': sentiments,
        'metrics': {
            'cumulative_return': cumulative_return,
            'annual_return': annual_return,
            'max_drawdown': max_drawdown,
            'volatility': np.std(daily_returns) * np.sqrt(252),
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio
        }
    }

def main():
    # Set random seed for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    # Define parameters
    start_date = '2010-01-01'
    train_end_date = '2016-12-31'
    end_date = '2019-01-01'
    ticker = 'AAPL'  # For real data, or will generate simulated data if download fails
    
    # Generate or download data
    stock_data = generate_simulated_data(start_date, end_date, ticker)
    news_data = generate_simulated_news(start_date, end_date)
    
    # Backtest strategy
    results = backtest_strategy(stock_data, news_data, train_end_date)
    
    # Compare with benchmark (buy and hold)
    initial_price = stock_data['Close'][stock_data.index > train_end_date].iloc[0]
    final_price = stock_data['Close'][stock_data.index > train_end_date].iloc[-1]
    benchmark_return = (final_price - initial_price) / initial_price
    
    print(f"\nBenchmark (Buy and Hold) Return: {benchmark_return:.2%}")
    print(f"Strategy Outperformance: {results['metrics']['cumulative_return'] - benchmark_return:.2%}")

if __name__ == "__main__":
    main()