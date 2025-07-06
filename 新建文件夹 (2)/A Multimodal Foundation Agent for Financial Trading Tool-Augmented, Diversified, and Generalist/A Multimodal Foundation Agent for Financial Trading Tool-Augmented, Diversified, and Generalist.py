import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import random

# Simulate data generation
def generate_stock_data(days=200, volatility=0.01, trend=0.0005, starting_price=100):
    """Generate simulated stock price data with trend and volatility"""
    prices = [starting_price]
    for _ in range(days-1):
        change = np.random.normal(trend, volatility)
        new_price = prices[-1] * (1 + change)
        prices.append(new_price)
    
    # Generate dates
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    date_range = pd.date_range(start=start_date, end=end_date, periods=days)
    
    # Create DataFrame
    df = pd.DataFrame({
        'Date': date_range,
        'Open': prices,
        'High': [p * (1 + random.uniform(0, 0.005)) for p in prices],
        'Low': [p * (1 - random.uniform(0, 0.005)) for p in prices],
        'Close': [p * (1 + random.uniform(-0.003, 0.003)) for p in prices],
        'Adj Close': [p * (1 + random.uniform(-0.002, 0.002)) for p in prices],
        'Volume': [random.randint(1000000, 5000000) for _ in range(days)]
    })
    
    return df

# Generate simulated news data
def generate_news_data(dates, asset_name):
    """Generate simulated news data for the given dates and asset"""
    positive_templates = [
        "Analysts upgrade {asset} on strong growth outlook",
        "{asset} reports better-than-expected earnings",
        "New product launch boosts {asset} market share",
        "{asset} announces expansion into new markets",
        "Investors bullish on {asset} future prospects"
    ]
    
    negative_templates = [
        "Regulatory concerns pressure {asset} stock",
        "{asset} misses revenue expectations",
        "Competition intensifies for {asset}",
        "Supply chain issues affect {asset} production",
        "Analysts downgrade {asset} citing valuation concerns"
    ]
    
    neutral_templates = [
        "{asset} maintains market position amid industry changes",
        "Investors await {asset} quarterly results",
        "{asset} announces executive changes",
        "Industry conference features {asset} presentation",
        "Market analysts mixed on {asset} outlook"
    ]
    
    news_data = []
    
    for date in dates:
        # Randomly determine number of news items for this date (0-3)
        num_news = random.randint(0, 3)
        
        for _ in range(num_news):
            sentiment = random.choice(['positive', 'negative', 'neutral'])
            
            if sentiment == 'positive':
                headline = random.choice(positive_templates).format(asset=asset_name)
                impact = random.choice(['SHORT-TERM', 'MEDIUM-TERM', 'LONG-TERM'])
            elif sentiment == 'negative':
                headline = random.choice(negative_templates).format(asset=asset_name)
                impact = random.choice(['SHORT-TERM', 'MEDIUM-TERM', 'LONG-TERM'])
            else:
                headline = random.choice(neutral_templates).format(asset=asset_name)
                impact = random.choice(['SHORT-TERM', 'MEDIUM-TERM', 'LONG-TERM'])
            
            news_data.append({
                'Date': date,
                'Headline': headline,
                'Sentiment': sentiment,
                'Impact': impact
            })
    
    return pd.DataFrame(news_data)

# Simplified classes for FinAgent components
class MarketIntelligenceModule:
    """Processes market data including news and prices"""
    
    def __init__(self):
        self.memory = []
    
    def analyze(self, price_data, news_data, current_day):
        """Analyze current day's market intelligence"""
        # Filter for current day's data
        current_price = price_data.loc[current_day]
        current_news = news_data[news_data['Date'] == price_data.loc[current_day, 'Date']]
        
        # Simple sentiment analysis
        sentiment_scores = {'positive': 1, 'negative': -1, 'neutral': 0}
        impact_weights = {'SHORT-TERM': 3, 'MEDIUM-TERM': 2, 'LONG-TERM': 1}
        
        sentiment_score = 0
        if not current_news.empty:
            for _, news in current_news.iterrows():
                sentiment_score += sentiment_scores[news['Sentiment']] * impact_weights[news['Impact']]
        
        # Price trend analysis (simplified)
        if current_day > 5:
            price_5d_change = (current_price['Close'] / price_data.loc[current_day-5, 'Close']) - 1
        else:
            price_5d_change = 0
            
        if current_day > 20:
            price_20d_change = (current_price['Close'] / price_data.loc[current_day-20, 'Close']) - 1
        else:
            price_20d_change = 0
        
        # Create market intelligence summary
        summary = {
            'date': current_price['Date'],
            'price': current_price['Close'],
            'sentiment_score': sentiment_score,
            'price_5d_change': price_5d_change,
            'price_20d_change': price_20d_change,
            'news': current_news.to_dict('records') if not current_news.empty else []
        }
        
        # Store in memory
        self.memory.append(summary)
        
        return summary
    
    def retrieve_past_intelligence(self, query_type, top_k=3):
        """Simplified retrieval of past market intelligence"""
        if not self.memory:
            return []
        
        # Sort by relevance (simplified - just returns recent items)
        sorted_memory = sorted(self.memory, key=lambda x: x['date'], reverse=True)
        
        # Return top-k items
        return sorted_memory[:min(top_k, len(sorted_memory))]


class ReflectionModule:
    """Handles low and high level reflection"""
    
    def __init__(self):
        self.low_level_memory = []
        self.high_level_memory = []
    
    def low_level_reflect(self, market_intelligence, price_data, current_day):
        """Reflect on relationship between market intelligence and price movements"""
        
        # Simple reflection logic
        short_term_prediction = "BULLISH" if market_intelligence['sentiment_score'] > 0 else "BEARISH"
        
        if market_intelligence['price_5d_change'] > 0 and market_intelligence['sentiment_score'] > 0:
            medium_term_prediction = "STRONGLY BULLISH"
        elif market_intelligence['price_5d_change'] > 0:
            medium_term_prediction = "MODERATELY BULLISH"
        elif market_intelligence['sentiment_score'] > 0:
            medium_term_prediction = "SLIGHTLY BULLISH"
        elif market_intelligence['price_5d_change'] < 0 and market_intelligence['sentiment_score'] < 0:
            medium_term_prediction = "STRONGLY BEARISH"
        elif market_intelligence['price_5d_change'] < 0:
            medium_term_prediction = "MODERATELY BEARISH"
        elif market_intelligence['sentiment_score'] < 0:
            medium_term_prediction = "SLIGHTLY BEARISH"
        else:
            medium_term_prediction = "NEUTRAL"
        
        long_term_prediction = "BULLISH" if market_intelligence['price_20d_change'] > 0 else "BEARISH"
        
        reflection = {
            'date': market_intelligence['date'],
            'short_term': short_term_prediction,
            'medium_term': medium_term_prediction,
            'long_term': long_term_prediction,
            'reasoning': f"Based on sentiment score {market_intelligence['sentiment_score']} and price changes: 5d={market_intelligence['price_5d_change']:.2%}, 20d={market_intelligence['price_20d_change']:.2%}"
        }
        
        self.low_level_memory.append(reflection)
        return reflection
    
    def high_level_reflect(self, decisions_history, market_intelligence, price_data, current_day):
        """Reflect on past trading decisions"""
        
        if not decisions_history:
            reflection = {
                'date': market_intelligence['date'],
                'assessment': "No past decisions to evaluate",
                'lessons': "Start with cautious positions"
            }
        else:
            # Evaluate most recent decision
            last_decision = decisions_history[-1]
            days_since_decision = (market_intelligence['date'] - last_decision['date']).days
            
            if days_since_decision < 5:  # Too soon to evaluate fully
                assessment = "Recent decision still unfolding"
            else:
                # Price change since decision
                price_at_decision = price_data[price_data['Date'] == last_decision['date']]['Close'].values[0]
                current_price = market_intelligence['price']
                price_change = (current_price / price_at_decision) - 1
                
                if last_decision['action'] == 'BUY' and price_change > 0:
                    assessment = f"BUY decision was correct, price increased by {price_change:.2%}"
                    lessons = "Effective recognition of bullish signals"
                elif last_decision['action'] == 'BUY' and price_change < 0:
                    assessment = f"BUY decision was incorrect, price decreased by {abs(price_change):.2%}"
                    lessons = "More careful analysis of bearish indicators needed"
                elif last_decision['action'] == 'SELL' and price_change < 0:
                    assessment = f"SELL decision was correct, price decreased by {abs(price_change):.2%}"
                    lessons = "Effective recognition of bearish signals"
                elif last_decision['action'] == 'SELL' and price_change > 0:
                    assessment = f"SELL decision was incorrect, price increased by {price_change:.2%}"
                    lessons = "More careful analysis of bullish indicators needed"
                else:  # HOLD cases
                    if abs(price_change) < 0.01:
                        assessment = "HOLD decision was appropriate, price remained stable"
                        lessons = "Effective recognition of neutral signals"
                    elif price_change > 0:
                        assessment = f"HOLD decision missed potential gains of {price_change:.2%}"
                        lessons = "More aggressive stance on bullish signals could be beneficial"
                    else:
                        assessment = f"HOLD decision prevented losses of {abs(price_change):.2%}"
                        lessons = "Effective caution in uncertain markets"
            
            reflection = {
                'date': market_intelligence['date'],
                'assessment': assessment,
                'lessons': lessons if 'lessons' in locals() else "Continue monitoring market conditions"
            }
        
        self.high_level_memory.append(reflection)
        return reflection


class DecisionMakingModule:
    """Makes trading decisions based on inputs from other modules"""
    
    def __init__(self, trader_preference="balanced"):
        self.trader_preference = trader_preference  # "aggressive", "balanced", or "conservative"
    
    def get_decision(self, market_intelligence, low_level_reflection, high_level_reflection, 
                     technical_indicators, current_position, cash):
        """Generate trading decision based on all inputs"""
        
        # Define weights based on trader preference
        if self.trader_preference == "aggressive":
            sentiment_weight = 0.4
            short_term_weight = 0.3
            medium_term_weight = 0.2
            long_term_weight = 0.1
            technical_weight = 0.5
        elif self.trader_preference == "conservative":
            sentiment_weight = 0.2
            short_term_weight = 0.1
            medium_term_weight = 0.2
            long_term_weight = 0.5
            technical_weight = 0.3
        else:  # balanced
            sentiment_weight = 0.3
            short_term_weight = 0.2
            medium_term_weight = 0.3
            long_term_weight = 0.2
            technical_weight = 0.4
        
        # Convert reflections to numeric scores
        sentiment_score = market_intelligence['sentiment_score']
        
        short_term_score = 1 if low_level_reflection['short_term'] == "BULLISH" else -1
        
        if "STRONGLY BULLISH" in low_level_reflection['medium_term']:
            medium_term_score = 2
        elif "MODERATELY BULLISH" in low_level_reflection['medium_term']:
            medium_term_score = 1
        elif "SLIGHTLY BULLISH" in low_level_reflection['medium_term']:
            medium_term_score = 0.5
        elif "NEUTRAL" in low_level_reflection['medium_term']:
            medium_term_score = 0
        elif "SLIGHTLY BEARISH" in low_level_reflection['medium_term']:
            medium_term_score = -0.5
        elif "MODERATELY BEARISH" in low_level_reflection['medium_term']:
            medium_term_score = -1
        else:  # STRONGLY BEARISH
            medium_term_score = -2
        
        long_term_score = 1 if low_level_reflection['long_term'] == "BULLISH" else -1
        
        # Technical indicators score (simplified)
        technical_score = technical_indicators['macd'] + technical_indicators['rsi'] + technical_indicators['bb']
        
        # Calculate weighted decision score
        decision_score = (
            sentiment_weight * sentiment_score +
            short_term_weight * short_term_score +
            medium_term_weight * medium_term_score +
            long_term_weight * long_term_score +
            technical_weight * technical_score
        )
        
        # Make decision based on score and current position
        threshold = 0.5  # Adjustable threshold
        
        if current_position > 0:  # Already holding
            if decision_score < -threshold:
                action = "SELL"
                reasoning = f"Bearish signals (score: {decision_score:.2f}) suggest exiting position"
            else:
                action = "HOLD"
                reasoning = f"Current signals (score: {decision_score:.2f}) support maintaining position"
        else:  # No position
            if decision_score > threshold and cash > market_intelligence['price']:
                action = "BUY"
                reasoning = f"Bullish signals (score: {decision_score:.2f}) suggest entering position"
            else:
                action = "HOLD"
                reasoning = f"Current signals (score: {decision_score:.2f}) or insufficient cash ({cash:.2f} < {market_intelligence['price']:.2f}) suggest waiting"
        
        # Create decision object
        decision = {
            'date': market_intelligence['date'],
            'action': action,
            'reasoning': reasoning,
            'decision_score': decision_score
        }
        
        return decision


class FinAgentSimulation:
    """Simplified FinAgent simulation"""
    
    def __init__(self, initial_cash=10000, trader_preference="balanced"):
        self.market_intelligence = MarketIntelligenceModule()
        self.reflection = ReflectionModule()
        self.decision_maker = DecisionMakingModule(trader_preference)
        
        self.cash = initial_cash
        self.position = 0
        self.position_value = 0
        self.decisions_history = []
        self.portfolio_history = []
    
    def calculate_technical_indicators(self, price_data, current_day):
        """Calculate simplified technical indicators"""
        
        # Simple MACD (difference between 12-day and 26-day EMA)
        if current_day >= 26:
            ema12 = price_data.loc[current_day-12+1:current_day, 'Close'].ewm(span=12).mean().iloc[-1]
            ema26 = price_data.loc[current_day-26+1:current_day, 'Close'].ewm(span=26).mean().iloc[-1]
            macd = (ema12 / ema26) - 1
        else:
            macd = 0
        
        # Simple RSI (simplified)
        if current_day >= 14:
            close_diff = price_data.loc[current_day-14+1:current_day, 'Close'].diff().dropna()
            gain = close_diff.where(close_diff > 0, 0).mean()
            loss = -close_diff.where(close_diff < 0, 0).mean()
            
            if loss == 0:
                rsi = 1  # Avoid division by zero
            else:
                rs = gain / loss
                rsi = 100 - (100 / (1 + rs))
            
            # Convert to -1 to 1 scale
            rsi_score = (rsi - 50) / 50
        else:
            rsi_score = 0
        
        # Bollinger Bands (simplified)
        if current_day >= 20:
            sma20 = price_data.loc[current_day-20+1:current_day, 'Close'].mean()
            std20 = price_data.loc[current_day-20+1:current_day, 'Close'].std()
            
            upper_band = sma20 + (2 * std20)
            lower_band = sma20 - (2 * std20)
            
            current_price = price_data.loc[current_day, 'Close']
            
            # Calculate position within bands (-1 to 1 scale)
            if current_price > upper_band:
                bb_score = 1  # Overbought
            elif current_price < lower_band:
                bb_score = -1  # Oversold
            else:
                # Position within the bands
                bb_score = 2 * (current_price - lower_band) / (upper_band - lower_band) - 1
        else:
            bb_score = 0
        
        return {
            'macd': macd,
            'rsi': rsi_score,
            'bb': bb_score
        }
    
    def run_simulation(self, price_data, news_data):
        """Run trading simulation over the price data period"""
        
        for day in range(len(price_data)):
            # Get market intelligence
            market_intelligence = self.market_intelligence.analyze(price_data, news_data, day)
            
            # Calculate technical indicators
            technical_indicators = self.calculate_technical_indicators(price_data, day)
            
            # Low-level reflection
            low_level_reflection = self.reflection.low_level_reflect(market_intelligence, price_data, day)
            
            # High-level reflection
            high_level_reflection = self.reflection.high_level_reflect(
                self.decisions_history, market_intelligence, price_data, day
            )
            
            # Make decision
            decision = self.decision_maker.get_decision(
                market_intelligence, 
                low_level_reflection,
                high_level_reflection,
                technical_indicators,
                self.position,
                self.cash
            )
            
            # Execute decision
            current_price = price_data.loc[day, 'Close']
            
            if decision['action'] == 'BUY' and self.cash >= current_price:
                shares_to_buy = self.cash // current_price
                cost = shares_to_buy * current_price
                self.cash -= cost
                self.position += shares_to_buy
                self.position_value = self.position * current_price
                
            elif decision['action'] == 'SELL' and self.position > 0:
                sale_value = self.position * current_price
                self.cash += sale_value
                self.position = 0
                self.position_value = 0
            
            # Update position value
            if self.position > 0:
                self.position_value = self.position * current_price
            
            # Save decision and portfolio value
            self.decisions_history.append(decision)
            
            portfolio_value = self.cash + self.position_value
            self.portfolio_history.append({
                'date': price_data.loc[day, 'Date'],
                'cash': self.cash,
                'position': self.position,
                'position_value': self.position_value,
                'portfolio_value': portfolio_value,
                'action': decision['action'],
                'price': current_price
            })
        
        return pd.DataFrame(self.portfolio_history)
    
    def calculate_performance(self, price_data):
        """Calculate performance metrics"""
        
        portfolio_df = pd.DataFrame(self.portfolio_history)
        
        # Calculate daily returns
        portfolio_df['daily_return'] = portfolio_df['portfolio_value'].pct_change()
        
        # Buy & Hold strategy for comparison
        initial_shares = self.cash / price_data.loc[0, 'Close']
        buy_hold_values = price_data['Close'] * initial_shares
        
        # Calculate metrics
        start_value = portfolio_df['portfolio_value'].iloc[0]
        end_value = portfolio_df['portfolio_value'].iloc[-1]
        total_return = (end_value / start_value) - 1
        
        buy_hold_return = (buy_hold_values.iloc[-1] / buy_hold_values.iloc[0]) - 1
        
        daily_returns = portfolio_df['daily_return'].dropna()
        annualized_return = (1 + total_return) ** (252 / len(portfolio_df)) - 1
        volatility = daily_returns.std() * np.sqrt(252)
        
        # Max drawdown
        portfolio_df['cumulative_max'] = portfolio_df['portfolio_value'].cummax()
        portfolio_df['drawdown'] = (portfolio_df['portfolio_value'] / portfolio_df['cumulative_max']) - 1
        max_drawdown = portfolio_df['drawdown'].min()
        
        # Sharpe ratio (simplified - using 0 as risk-free rate)
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
        
        # Summary
        performance = {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'buy_hold_return': buy_hold_return
        }
        
        return performance, portfolio_df, buy_hold_values


# Run the simulation
if __name__ == "__main__":
    # Generate simulated data
    np.random.seed(42)  # For reproducibility
    asset_name = "TechCorp"
    
    # Slight upward trend with some volatility
    price_data = generate_stock_data(days=200, volatility=0.015, trend=0.0005, starting_price=100)
    news_data = generate_news_data(price_data['Date'], asset_name)
    
    # Run simulation with different trader preferences
    strategies = {
        'Aggressive': FinAgentSimulation(initial_cash=10000, trader_preference="aggressive"),
        'Balanced': FinAgentSimulation(initial_cash=10000, trader_preference="balanced"),
        'Conservative': FinAgentSimulation(initial_cash=10000, trader_preference="conservative")
    }
    
    results = {}
    
    for name, strategy in strategies.items():
        portfolio_history = strategy.run_simulation(price_data, news_data)
        performance, portfolio_df, buy_hold_values = strategy.calculate_performance(price_data)
        
        results[name] = {
            'performance': performance,
            'portfolio_df': portfolio_df,
            'buy_hold_values': buy_hold_values
        }
    
    # Plot results
    plt.figure(figsize=(14, 8))
    
    for name, result in results.items():
        plt.plot(result['portfolio_df']['date'], result['portfolio_df']['portfolio_value'], label=name)
    
    plt.plot(price_data['Date'], results['Balanced']['buy_hold_values'], label='Buy & Hold', linestyle='--')
    
    plt.title('FinAgent Simulation: Portfolio Value Over Time')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value ($)')
    plt.legend()
    plt.grid(True)
    
    # Print performance metrics
    print("Performance Metrics:")
    print("-" * 80)
    print(f"{'Strategy':<15} {'Total Return':<15} {'Annual Return':<15} {'Sharpe Ratio':<15} {'Max Drawdown':<15}")
    print("-" * 80)
    
    for name, result in results.items():
        perf = result['performance']
        print(f"{name:<15} {perf['total_return']:.2%} {perf['annualized_return']:.2%} {perf['sharpe_ratio']:.2f} {perf['max_drawdown']:.2%}")
    
    print("-" * 80)
    print(f"Buy & Hold: {results['Balanced']['performance']['buy_hold_return']:.2%}")
    
    # Plot trading decisions for the balanced strategy
    balanced_df = results['Balanced']['portfolio_df']
    
    buy_points = balanced_df[balanced_df['action'] == 'BUY']
    sell_points = balanced_df[balanced_df['action'] == 'SELL']
    
    plt.figure(figsize=(14, 6))
    plt.plot(price_data['Date'], price_data['Close'], label='Stock Price')
    plt.scatter(buy_points['date'], buy_points['price'], color='green', marker='^', s=100, label='Buy')
    plt.scatter(sell_points['date'], sell_points['price'], color='red', marker='v', s=100, label='Sell')
    
    plt.title('Trading Decisions (Balanced Strategy)')
    plt.xlabel('Date')
    plt.ylabel('Stock Price ($)')
    plt.legend()
    plt.grid(True)
    
    plt.show()
    
    # Print a few example decisions with reasoning
    print("\nExample Trading Decisions (Balanced Strategy):")
    print("-" * 100)
    
    for i, decision in enumerate(strategies['Balanced'].decisions_history):
        if decision['action'] != 'HOLD' and i > 20:  # Skip initial decisions and only show BUY/SELL
            date = decision['date'].strftime('%Y-%m-%d')
            print(f"Date: {date}, Action: {decision['action']}, Reasoning: {decision['reasoning']}")
            
            # Show what market intelligence and reflection led to this decision
            day_index = strategies['Balanced'].portfolio_history[i]['date']
            matching_intelligence = [m for m in strategies['Balanced'].market_intelligence.memory if m['date'] == day_index]
            matching_reflection = [r for r in strategies['Balanced'].reflection.low_level_memory if r['date'] == day_index]
            
            if matching_intelligence:
                intel = matching_intelligence[0]
                print(f"  Market Intelligence: Sentiment={intel['sentiment_score']:.2f}, 5d Change={intel['price_5d_change']:.2%}")
            
            if matching_reflection:
                reflect = matching_reflection[0]
                print(f"  Reflection: Short-term={reflect['short_term']}, Medium-term={reflect['medium_term']}, Long-term={reflect['long_term']}")
            
            print("-" * 100)
            
            # Only show a few examples
            if i > 30:
                break