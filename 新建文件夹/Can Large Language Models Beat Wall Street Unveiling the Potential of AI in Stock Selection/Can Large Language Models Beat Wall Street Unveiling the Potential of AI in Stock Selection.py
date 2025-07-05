import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
import random
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

class MarketSenseAISimulation:
    """
    A simulation of the MarketSenseAI framework described in the paper.
    This implementation simulates the key components and evaluates performance
    with synthetic data.
    """
    
    def __init__(self, num_stocks=30, start_date='2022-01-01', end_date='2023-12-31', 
                 rebalance_freq='monthly', transaction_cost=0.0005):
        """
        Initialize the simulation environment
        
        Parameters:
        -----------
        num_stocks : int
            Number of stocks to simulate
        start_date, end_date : str
            Simulation period
        rebalance_freq : str
            Frequency of portfolio rebalancing ('monthly', 'quarterly')
        transaction_cost : float
            Transaction cost as a percentage of trade value
        """
        self.num_stocks = num_stocks
        self.start_date = datetime.strptime(start_date, '%Y-%m-%d')
        self.end_date = datetime.strptime(end_date, '%Y-%m-%d')
        self.rebalance_freq = rebalance_freq
        self.transaction_cost = transaction_cost
        
        # Generate dates for the simulation period
        self.dates = pd.date_range(start=self.start_date, end=self.end_date, freq='B')
        
        # Create stock identifiers
        self.stocks = [f'STOCK_{i+1}' for i in range(num_stocks)]
        
        # Dictionary to store generated data
        self.data = {
            'prices': None,
            'fundamentals': None,
            'news_sentiment': None,
            'macro': None,
            'price_dynamics': None,
            'signals': None
        }
        
        # Generate all simulated data
        self._generate_all_data()
        
    def _generate_all_data(self):
        """Generate all simulated data for the framework"""
        print("Generating simulated market data...")
        self._generate_stock_prices()
        self._generate_fundamentals()
        self._generate_news_sentiment()
        self._generate_macro_environment()
        self._calculate_price_dynamics()
        print("Simulated data generation complete.")
        
    def _generate_stock_prices(self):
        """Generate simulated stock price data"""
        # Initialize prices with random starting values
        initial_prices = np.random.uniform(50, 500, size=self.num_stocks)
        
        # Generate returns with some stocks outperforming and some underperforming
        # Add correlation structure and sector patterns
        num_days = len(self.dates)
        
        # Create correlated returns - divide stocks into sectors
        num_sectors = 5
        sector_size = self.num_stocks // num_sectors
        sectors = []
        
        for i in range(num_sectors):
            sector_stocks = list(range(i * sector_size, min((i + 1) * sector_size, self.num_stocks)))
            sectors.append(sector_stocks)
            
        # Generate market factor returns
        market_returns = np.random.normal(0.0005, 0.01, num_days)  # Slight positive drift
        
        # Generate sector-specific returns
        sector_returns = {}
        for i in range(num_sectors):
            # Some sectors perform better than others
            sector_mean = 0.0003 + 0.0004 * (i / (num_sectors - 1))
            sector_returns[i] = np.random.normal(sector_mean, 0.008, num_days)
        
        # Generate stock-specific returns
        stock_returns = np.zeros((num_days, self.num_stocks))
        
        for day in range(num_days):
            for stock_idx in range(self.num_stocks):
                # Find which sector this stock belongs to
                for sector_idx, stocks in enumerate(sectors):
                    if stock_idx in stocks:
                        # Stock return is a combination of market, sector, and idiosyncratic return
                        market_component = market_returns[day] * (0.8 + 0.4 * np.random.random())
                        sector_component = sector_returns[sector_idx][day] * (0.8 + 0.4 * np.random.random())
                        idiosyncratic = np.random.normal(0, 0.015)
                        
                        stock_returns[day, stock_idx] = market_component + sector_component + idiosyncratic
                        break
        
        # Calculate cumulative returns
        cum_returns = np.cumprod(1 + stock_returns, axis=0)
        
        # Calculate prices
        prices = np.outer(cum_returns, initial_prices)
        
        # Create DataFrame
        self.data['prices'] = pd.DataFrame(
            prices, 
            index=self.dates,
            columns=self.stocks
        )
        
        # Create a column for S&P equivalent (market index)
        self.data['prices']['S&P_EQUIVALENT'] = np.mean(prices, axis=1)
    
    def _generate_fundamentals(self):
        """Generate quarterly fundamental data for stocks"""
        # Create quarterly dates
        quarterly_dates = pd.date_range(
            start=self.start_date, 
            end=self.end_date, 
            freq='Q'
        )
        
        # Initialize DataFrames for fundamental metrics
        fundamentals = pd.DataFrame(index=quarterly_dates)
        
        # Generate fundamentals for each stock
        for stock in self.stocks:
            # Initial values
            revenue_base = np.random.uniform(1000, 10000)
            eps_base = np.random.uniform(0.5, 5)
            pe_base = np.random.uniform(10, 30)
            debt_to_equity_base = np.random.uniform(0.1, 2)
            profit_margin_base = np.random.uniform(0.05, 0.3)
            
            # Generate quarterly series with some seasonality and trends
            for quarter_idx, quarter in enumerate(quarterly_dates):
                # Add seasonal component (Q4 typically stronger)
                season_factor = 1.0 + 0.1 * (quarter.month == 12)
                
                # Add random walk component
                trend_component = 0.01 * quarter_idx
                
                # Add noise
                noise_revenue = np.random.normal(0, 0.05)
                noise_eps = np.random.normal(0, 0.1)
                noise_pe = np.random.normal(0, 1)
                noise_debt = np.random.normal(0, 0.05)
                noise_margin = np.random.normal(0, 0.01)
                
                # Calculate metrics
                revenue = revenue_base * (1 + trend_component + noise_revenue) * season_factor
                eps = eps_base * (1 + trend_component + noise_eps) * season_factor
                pe = pe_base + noise_pe
                debt_to_equity = debt_to_equity_base + noise_debt
                profit_margin = profit_margin_base + noise_margin
                
                # Store in DataFrame
                fundamentals.loc[quarter, f"{stock}_Revenue"] = revenue
                fundamentals.loc[quarter, f"{stock}_EPS"] = eps
                fundamentals.loc[quarter, f"{stock}_PE"] = pe
                fundamentals.loc[quarter, f"{stock}_DebtToEquity"] = debt_to_equity
                fundamentals.loc[quarter, f"{stock}_ProfitMargin"] = profit_margin
        
        self.data['fundamentals'] = fundamentals
    
    def _generate_news_sentiment(self):
        """Generate daily news sentiment for each stock"""
        # Create a DataFrame for news sentiment
        news_sentiment = pd.DataFrame(index=self.dates, columns=self.stocks)
        
        # Generate sentiment scores for each stock
        for stock in self.stocks:
            # Base sentiment that evolves over time
            base_sentiment = np.random.normal(0.1, 0.05)  # Slight positive bias
            
            # Generate a random walk for sentiment evolution
            sentiment_series = np.zeros(len(self.dates))
            sentiment_series[0] = base_sentiment
            
            for i in range(1, len(self.dates)):
                # Sentiment has some momentum and mean reversion
                sentiment_change = np.random.normal(0, 0.1)
                mean_reversion = 0.02 * (base_sentiment - sentiment_series[i-1])
                sentiment_series[i] = sentiment_series[i-1] + sentiment_change + mean_reversion
            
            # Add occasional news spikes (earnings, product announcements, etc.)
            num_spikes = int(len(self.dates) * 0.05)  # News on ~5% of days
            spike_indices = random.sample(range(len(self.dates)), num_spikes)
            
            for idx in spike_indices:
                # News can be positive or negative
                sentiment_series[idx] += np.random.normal(0, 0.3)
            
            # Normalize to a reasonable range
            news_sentiment[stock] = np.clip(sentiment_series, -1, 1)
        
        self.data['news_sentiment'] = news_sentiment
    
    def _generate_macro_environment(self):
        """Generate macroeconomic environment factors"""
        # Create monthly macro data
        macro_dates = pd.date_range(start=self.start_date, end=self.end_date, freq='M')
        
        # Initialize DataFrame
        macro_data = pd.DataFrame(index=macro_dates)
        
        # Generate macroeconomic indicators
        # Interest rates - usually change slowly
        interest_rates = np.zeros(len(macro_dates))
        interest_rates[0] = np.random.uniform(1, 3)
        
        for i in range(1, len(macro_dates)):
            # Interest rates change with some persistence
            change = np.random.normal(0, 0.1)
            interest_rates[i] = interest_rates[i-1] + change
        
        # Keep rates in reasonable bounds
        interest_rates = np.clip(interest_rates, 0, 8)
        
        # Inflation - correlated with interest rates but with more noise
        inflation = interest_rates + np.random.normal(0, 0.5, len(macro_dates))
        inflation = np.clip(inflation, 0, 10)
        
        # GDP growth - quarterly pattern
        gdp_growth = np.zeros(len(macro_dates))
        for i in range(len(macro_dates)):
            quarter = (macro_dates[i].month - 1) // 3
            season_factor = 0.2 * (quarter == 2)  # Q3 (summer) often stronger
            gdp_growth[i] = 2.0 + np.random.normal(0, 0.5) + season_factor
        
        # Unemployment - negatively correlated with GDP growth
        unemployment = 5.0 - 0.2 * gdp_growth + np.random.normal(0, 0.3, len(macro_dates))
        unemployment = np.clip(unemployment, 3, 10)
        
        # Consumer sentiment - affected by economic factors
        consumer_sentiment = 100 - 2*unemployment + gdp_growth - inflation + np.random.normal(0, 3, len(macro_dates))
        
        # Add to DataFrame
        macro_data['InterestRate'] = interest_rates
        macro_data['Inflation'] = inflation
        macro_data['GDPGrowth'] = gdp_growth
        macro_data['Unemployment'] = unemployment
        macro_data['ConsumerSentiment'] = consumer_sentiment
        
        self.data['macro'] = macro_data
    
    def _calculate_price_dynamics(self):
        """Calculate price dynamics indicators for stocks"""
        # Calculate on a monthly basis
        monthly_dates = pd.date_range(start=self.start_date, end=self.end_date, freq='M')
        
        # Initialize DataFrame for price dynamics
        price_dynamics = pd.DataFrame(index=monthly_dates)
        
        # For each month-end, calculate trailing metrics
        for month_end in monthly_dates:
            # Get data up to this month
            month_data = self.data['prices'].loc[:month_end]
            
            # For each stock, calculate returns, volatility, Sharpe ratio, etc.
            for stock in self.stocks:
                # Get stock prices
                stock_prices = month_data[stock]
                
                # Calculate 1, 3, 6, 12-month returns
                if len(stock_prices) >= 21:  # ~1 month of trading days
                    price_dynamics.loc[month_end, f"{stock}_Return_1M"] = (
                        stock_prices.iloc[-1] / stock_prices.iloc[-21] - 1
                    )
                
                if len(stock_prices) >= 63:  # ~3 months
                    price_dynamics.loc[month_end, f"{stock}_Return_3M"] = (
                        stock_prices.iloc[-1] / stock_prices.iloc[-63] - 1
                    )
                
                if len(stock_prices) >= 126:  # ~6 months
                    price_dynamics.loc[month_end, f"{stock}_Return_6M"] = (
                        stock_prices.iloc[-1] / stock_prices.iloc[-126] - 1
                    )
                
                if len(stock_prices) >= 252:  # ~1 year
                    price_dynamics.loc[month_end, f"{stock}_Return_12M"] = (
                        stock_prices.iloc[-1] / stock_prices.iloc[-252] - 1
                    )
                
                # Calculate volatility (21-day rolling standard deviation of returns)
                if len(stock_prices) >= 21:
                    returns = stock_prices.pct_change().iloc[-21:]
                    price_dynamics.loc[month_end, f"{stock}_Volatility"] = returns.std() * np.sqrt(252)
                
                # Calculate Sharpe ratio (assuming risk-free rate of 0 for simplicity)
                if len(stock_prices) >= 63:  # Use 3-month data for Sharpe
                    returns = stock_prices.pct_change().iloc[-63:]
                    mean_return = returns.mean() * 252
                    volatility = returns.std() * np.sqrt(252)
                    price_dynamics.loc[month_end, f"{stock}_Sharpe"] = (
                        mean_return / volatility if volatility > 0 else 0
                    )
                
                # Calculate max drawdown
                if len(stock_prices) >= 126:  # Use 6-month data for drawdown
                    prices_6m = stock_prices.iloc[-126:]
                    rolling_max = prices_6m.expanding().max()
                    drawdown = (prices_6m / rolling_max - 1)
                    price_dynamics.loc[month_end, f"{stock}_MaxDrawdown"] = drawdown.min()
        
        self.data['price_dynamics'] = price_dynamics
    
    def _summarize_news(self, stock, date, lookback_days=30):
        """
        Simulate the Progressive News Summarizer component
        
        Returns a summary of recent news sentiment for a stock
        """
        # Get sentiment data for the lookback period
        end_date = date
        start_date = end_date - timedelta(days=lookback_days)
        
        # Filter news sentiment data
        mask = (self.data['news_sentiment'].index >= start_date) & (self.data['news_sentiment'].index <= end_date)
        sentiment_period = self.data['news_sentiment'].loc[mask, stock]
        
        if len(sentiment_period) == 0:
            return {
                'avg_sentiment': 0,
                'sentiment_trend': 'neutral',
                'recent_developments': 'No significant news'
            }
        
        # Calculate average sentiment
        avg_sentiment = sentiment_period.mean()
        
        # Determine sentiment trend
        recent_sentiment = sentiment_period.iloc[-5:].mean()
        older_sentiment = sentiment_period.iloc[:-5].mean() if len(sentiment_period) > 5 else 0
        
        if recent_sentiment > older_sentiment + 0.1:
            sentiment_trend = 'improving'
        elif recent_sentiment < older_sentiment - 0.1:
            sentiment_trend = 'deteriorating'
        else:
            sentiment_trend = 'stable'
        
        # Identify significant news (spikes in sentiment)
        significant_news = sentiment_period[abs(sentiment_period) > 0.3]
        
        # Create a narrative based on sentiment
        if avg_sentiment > 0.3:
            narrative = f"Predominantly positive news coverage for {stock}"
        elif avg_sentiment < -0.3:
            narrative = f"Predominantly negative news coverage for {stock}"
        else:
            narrative = f"Mixed or neutral news coverage for {stock}"
        
        # Add details about significant news events
        if len(significant_news) > 0:
            event_count = len(significant_news)
            positive_events = sum(significant_news > 0.3)
            negative_events = sum(significant_news < -0.3)
            
            narrative += f". {event_count} significant news events: {positive_events} positive and {negative_events} negative."
        
        return {
            'avg_sentiment': avg_sentiment,
            'sentiment_trend': sentiment_trend,
            'recent_developments': narrative
        }
    
    def _summarize_fundamentals(self, stock, date):
        """
        Simulate the Fundamentals Summarizer component
        
        Returns a summary of the most recent fundamental data for a stock
        """
        # Find the most recent quarter before the given date
        mask = self.data['fundamentals'].index <= date
        if not any(mask):
            return {
                'profitability': 'unknown',
                'growth': 'unknown',
                'valuation': 'unknown',
                'financial_health': 'unknown'
            }
        
        latest_quarter = self.data['fundamentals'][mask].index[-1]
        
        # Get fundamental data
        revenue = self.data['fundamentals'].loc[latest_quarter, f"{stock}_Revenue"]
        eps = self.data['fundamentals'].loc[latest_quarter, f"{stock}_EPS"]
        pe = self.data['fundamentals'].loc[latest_quarter, f"{stock}_PE"]
        debt_to_equity = self.data['fundamentals'].loc[latest_quarter, f"{stock}_DebtToEquity"]
        profit_margin = self.data['fundamentals'].loc[latest_quarter, f"{stock}_ProfitMargin"]
        
        # If we have at least two quarters of data, calculate growth
        revenue_growth = 0
        eps_growth = 0
        
        if len(self.data['fundamentals'][mask]) >= 2:
            prev_quarter = self.data['fundamentals'][mask].index[-2]
            prev_revenue = self.data['fundamentals'].loc[prev_quarter, f"{stock}_Revenue"]
            prev_eps = self.data['fundamentals'].loc[prev_quarter, f"{stock}_EPS"]
            
            revenue_growth = (revenue / prev_revenue - 1) * 100
            eps_growth = (eps / prev_eps - 1) * 100
        
        # Categorize fundamentals
        if profit_margin > 0.2:
            profitability = 'excellent'
        elif profit_margin > 0.1:
            profitability = 'good'
        elif profit_margin > 0.05:
            profitability = 'average'
        else:
            profitability = 'poor'
            
        if revenue_growth > 15 and eps_growth > 15:
            growth = 'strong'
        elif revenue_growth > 5 and eps_growth > 5:
            growth = 'moderate'
        elif revenue_growth > 0 and eps_growth > 0:
            growth = 'slow'
        else:
            growth = 'negative'
            
        if pe < 15:
            valuation = 'undervalued'
        elif pe < 25:
            valuation = 'fairly valued'
        else:
            valuation = 'overvalued'
            
        if debt_to_equity < 0.5:
            financial_health = 'excellent'
        elif debt_to_equity < 1:
            financial_health = 'good'
        elif debt_to_equity < 1.5:
            financial_health = 'average'
        else:
            financial_health = 'concerning'
        
        return {
            'profitability': profitability,
            'growth': growth,
            'valuation': valuation,
            'financial_health': financial_health,
            'metrics': {
                'revenue': revenue,
                'eps': eps,
                'pe': pe,
                'debt_to_equity': debt_to_equity,
                'profit_margin': profit_margin,
                'revenue_growth': revenue_growth,
                'eps_growth': eps_growth
            }
        }
    
    def _summarize_price_dynamics(self, stock, date, num_similar_stocks=5):
        """
        Simulate the Stock Price Dynamics Summarizer component
        
        Returns a summary of recent price performance for a stock
        """
        # Find the most recent month-end data point before the given date
        mask = self.data['price_dynamics'].index <= date
        if not any(mask):
            return {
                'short_term_momentum': 'neutral',
                'relative_strength': 'neutral',
                'volatility_profile': 'average',
                'risk_adjusted_performance': 'average'
            }
        
        latest_month = self.data['price_dynamics'][mask].index[-1]
        
        # Get available metrics for this month
        available_metrics = [
            col for col in self.data['price_dynamics'].columns 
            if col.startswith(stock) and not pd.isna(self.data['price_dynamics'].loc[latest_month, col])
        ]
        
        if not available_metrics:
            return {
                'short_term_momentum': 'neutral',
                'relative_strength': 'neutral',
                'volatility_profile': 'average',
                'risk_adjusted_performance': 'average'
            }
        
        # Get price dynamics data for the stock
        stock_data = {}
        for metric in ['Return_1M', 'Return_3M', 'Return_6M', 'Return_12M', 'Volatility', 'Sharpe', 'MaxDrawdown']:
            metric_col = f"{stock}_{metric}"
            if metric_col in available_metrics:
                stock_data[metric] = self.data['price_dynamics'].loc[latest_month, metric_col]
            else:
                stock_data[metric] = None
        
        # Get similar stocks (in a real implementation, this would use embeddings similarity)
        # Here we'll just pick random stocks for demonstration
        similar_stocks = random.sample([s for s in self.stocks if s != stock], min(num_similar_stocks, len(self.stocks)-1))
        
        # Calculate average metrics for similar stocks
        similar_stocks_data = {}
        for metric in ['Return_1M', 'Return_3M', 'Return_6M', 'Return_12M', 'Volatility', 'Sharpe', 'MaxDrawdown']:
            values = []
            for similar_stock in similar_stocks:
                metric_col = f"{similar_stock}_{metric}"
                if metric_col in self.data['price_dynamics'].columns:
                    val = self.data['price_dynamics'].loc[latest_month, metric_col]
                    if not pd.isna(val):
                        values.append(val)
            
            similar_stocks_data[metric] = np.mean(values) if values else None
        
        # Compare stock to similar stocks
        comparisons = {}
        for metric in ['Return_1M', 'Return_3M', 'Return_6M', 'Return_12M', 'Volatility', 'Sharpe', 'MaxDrawdown']:
            if stock_data[metric] is not None and similar_stocks_data[metric] is not None:
                if metric == 'MaxDrawdown':  # For MaxDrawdown, less negative is better
                    comparisons[metric] = "better" if stock_data[metric] > similar_stocks_data[metric] else "worse"
                elif metric == 'Volatility':  # For volatility, lower is generally better
                    comparisons[metric] = "better" if stock_data[metric] < similar_stocks_data[metric] else "worse"
                else:  # For returns and Sharpe, higher is better
                    comparisons[metric] = "better" if stock_data[metric] > similar_stocks_data[metric] else "worse"
            else:
                comparisons[metric] = "unknown"
        
        # Determine short-term momentum (based on 1M and 3M returns)
        if stock_data['Return_1M'] is not None and stock_data['Return_3M'] is not None:
            recent_return = stock_data['Return_1M']
            if recent_return > 0.05:
                short_term_momentum = "strongly positive"
            elif recent_return > 0.02:
                short_term_momentum = "positive"
            elif recent_return > -0.02:
                short_term_momentum = "neutral"
            elif recent_return > -0.05:
                short_term_momentum = "negative"
            else:
                short_term_momentum = "strongly negative"
        else:
            short_term_momentum = "unknown"
        
        # Determine relative strength compared to peers
        relative_strength_score = sum(1 for metric in ['Return_1M', 'Return_3M', 'Return_6M', 'Return_12M', 'Sharpe'] 
                                     if comparisons.get(metric) == "better")
        
        if relative_strength_score >= 4:
            relative_strength = "much stronger than peers"
        elif relative_strength_score >= 3:
            relative_strength = "stronger than peers"
        elif relative_strength_score >= 2:
            relative_strength = "similar to peers"
        elif relative_strength_score >= 1:
            relative_strength = "weaker than peers"
        else:
            relative_strength = "much weaker than peers"
        
        # Determine volatility profile
        if stock_data['Volatility'] is not None:
            volatility = stock_data['Volatility']
            if volatility > 0.3:
                volatility_profile = "very high"
            elif volatility > 0.2:
                volatility_profile = "high"
            elif volatility > 0.15:
                volatility_profile = "average"
            elif volatility > 0.1:
                volatility_profile = "low"
            else:
                volatility_profile = "very low"
        else:
            volatility_profile = "unknown"
        
        # Determine risk-adjusted performance
        if stock_data['Sharpe'] is not None:
            sharpe = stock_data['Sharpe']
            if sharpe > 2:
                risk_adjusted_performance = "excellent"
            elif sharpe > 1:
                risk_adjusted_performance = "good"
            elif sharpe > 0:
                risk_adjusted_performance = "average"
            else:
                risk_adjusted_performance = "poor"
        else:
            risk_adjusted_performance = "unknown"
        
        return {
            'short_term_momentum': short_term_momentum,
            'relative_strength': relative_strength,
            'volatility_profile': volatility_profile,
            'risk_adjusted_performance': risk_adjusted_performance,
            'stock_data': stock_data,
            'similar_stocks_data': similar_stocks_data,
            'comparisons': comparisons
        }
    
    def _summarize_macro(self, date):
        """
        Simulate the Macroeconomic Environment Summary component
        
        Returns a summary of the macroeconomic environment
        """
        # Find the most recent macro data point before the given date
        mask = self.data['macro'].index <= date
        if not any(mask):
            return {
                'interest_rate_environment': 'neutral',
                'inflation_outlook': 'stable',
                'economic_growth': 'moderate',
                'market_sentiment': 'neutral'
            }
        
        latest_month = self.data['macro'][mask].index[-1]
        
        # Get macro data
        interest_rate = self.data['macro'].loc[latest_month, 'InterestRate']
        inflation = self.data['macro'].loc[latest_month, 'Inflation']
        gdp_growth = self.data['macro'].loc[latest_month, 'GDPGrowth']
        unemployment = self.data['macro'].loc[latest_month, 'Unemployment']
        consumer_sentiment = self.data['macro'].loc[latest_month, 'ConsumerSentiment']
        
        # Get trend if we have at least 3 months of data
        interest_rate_trend = 'stable'
        inflation_trend = 'stable'
        
        if len(self.data['macro'][mask]) >= 3:
            prev_months = self.data['macro'][mask].index[-3:]
            interest_rates_3m = self.data['macro'].loc[prev_months, 'InterestRate']
            inflation_3m = self.data['macro'].loc[prev_months, 'Inflation']
            
            # Calculate trends
            interest_rate_change = interest_rates_3m.iloc[-1] - interest_rates_3m.iloc[0]
            inflation_change = inflation_3m.iloc[-1] - inflation_3m.iloc[0]
            
            if interest_rate_change > 0.5:
                interest_rate_trend = 'rising sharply'
            elif interest_rate_change > 0.1:
                interest_rate_trend = 'rising'
            elif interest_rate_change < -0.5:
                interest_rate_trend = 'falling sharply'
            elif interest_rate_change < -0.1:
                interest_rate_trend = 'falling'
            
            if inflation_change > 1:
                inflation_trend = 'rising sharply'
            elif inflation_change > 0.3:
                inflation_trend = 'rising'
            elif inflation_change < -1:
                inflation_trend = 'falling sharply'
            elif inflation_change < -0.3:
                inflation_trend = 'falling'
        
        # Categorize macro environment
        if interest_rate > 5:
            interest_rate_environment = f"high interest rates ({interest_rate_trend})"
        elif interest_rate > 3:
            interest_rate_environment = f"moderate interest rates ({interest_rate_trend})"
        else:
            interest_rate_environment = f"low interest rates ({interest_rate_trend})"
            
        if inflation > 6:
            inflation_outlook = f"high inflation ({inflation_trend})"
        elif inflation > 3:
            inflation_outlook = f"moderate inflation ({inflation_trend})"
        else:
            inflation_outlook = f"low inflation ({inflation_trend})"
            
        if gdp_growth > 3:
            economic_growth = "strong"
        elif gdp_growth > 1.5:
            economic_growth = "moderate"
        elif gdp_growth > 0:
            economic_growth = "weak"
        else:
            economic_growth = "contracting"
            
        if consumer_sentiment > 110:
            market_sentiment = "very optimistic"
        elif consumer_sentiment > 100:
            market_sentiment = "optimistic"
        elif consumer_sentiment > 90:
            market_sentiment = "neutral"
        elif consumer_sentiment > 80:
            market_sentiment = "pessimistic"
        else:
            market_sentiment = "very pessimistic"
        
        return {
            'interest_rate_environment': interest_rate_environment,
            'inflation_outlook': inflation_outlook,
            'economic_growth': economic_growth,
            'market_sentiment': market_sentiment,
            'metrics': {
                'interest_rate': interest_rate,
                'inflation': inflation,
                'gdp_growth': gdp_growth,
                'unemployment': unemployment,
                'consumer_sentiment': consumer_sentiment
            }
        }
    
    def _simulate_llm_decision(self, stock, date):
        """
        Simulate the LLM-based Signal Generation component
        
        This function simulates how an LLM might make an investment decision
        based on all the summarized data inputs
        """
        # Get summaries from all components
        news_summary = self._summarize_news(stock, date)
        fundamentals_summary = self._summarize_fundamentals(stock, date)
        price_dynamics_summary = self._summarize_price_dynamics(stock, date)
        macro_summary = self._summarize_macro(date)
        
        # Initialize score components
        news_score = 0
        fundamentals_score = 0
        price_dynamics_score = 0
        macro_score = 0
        
        # Score based on news
        if news_summary['avg_sentiment'] > 0.2:
            news_score = 1
        elif news_summary['avg_sentiment'] < -0.2:
            news_score = -1
            
        if news_summary['sentiment_trend'] == 'improving':
            news_score += 0.5
        elif news_summary['sentiment_trend'] == 'deteriorating':
            news_score -= 0.5
        
        # Score based on fundamentals
        if fundamentals_summary['profitability'] in ['excellent', 'good']:
            fundamentals_score += 0.5
        elif fundamentals_summary['profitability'] == 'poor':
            fundamentals_score -= 0.5
            
        if fundamentals_summary['growth'] == 'strong':
            fundamentals_score += 0.5
        elif fundamentals_summary['growth'] == 'negative':
            fundamentals_score -= 0.5
            
        if fundamentals_summary['valuation'] == 'undervalued':
            fundamentals_score += 0.5
        elif fundamentals_summary['valuation'] == 'overvalued':
            fundamentals_score -= 0.5
            
        if fundamentals_summary['financial_health'] in ['excellent', 'good']:
            fundamentals_score += 0.5
        elif fundamentals_summary['financial_health'] == 'concerning':
            fundamentals_score -= 0.5
        
        # Score based on price dynamics
        if price_dynamics_summary['short_term_momentum'] in ['strongly positive', 'positive']:
            price_dynamics_score += 0.5
        elif price_dynamics_summary['short_term_momentum'] in ['strongly negative', 'negative']:
            price_dynamics_score -= 0.5
            
        if price_dynamics_summary['relative_strength'] in ['much stronger than peers', 'stronger than peers']:
            price_dynamics_score += 0.5
        elif price_dynamics_summary['relative_strength'] in ['weaker than peers', 'much weaker than peers']:
            price_dynamics_score -= 0.5
            
        if price_dynamics_summary['risk_adjusted_performance'] in ['excellent', 'good']:
            price_dynamics_score += 0.5
        elif price_dynamics_summary['risk_adjusted_performance'] == 'poor':
            price_dynamics_score -= 0.5
        
        # Score based on macro environment
        if macro_summary['economic_growth'] in ['strong', 'moderate']:
            macro_score += 0.25
        elif macro_summary['economic_growth'] in ['contracting']:
            macro_score -= 0.25
            
        if macro_summary['market_sentiment'] in ['very optimistic', 'optimistic']:
            macro_score += 0.25
        elif macro_summary['market_sentiment'] in ['pessimistic', 'very pessimistic']:
            macro_score -= 0.25
            
        # Interest rates and inflation have complex effects on different sectors
        # This is a simplified model
        if 'low' in macro_summary['interest_rate_environment'] and 'rising' not in macro_summary['interest_rate_environment']:
            macro_score += 0.25
        elif 'high' in macro_summary['interest_rate_environment'] and 'falling' not in macro_summary['interest_rate_environment']:
            macro_score -= 0.25
        
        # Calculate final score with weights
        # Give more weight to fundamentals and price dynamics as per the paper's findings
        final_score = (
            0.25 * news_score +
            0.3 * fundamentals_score +
            0.3 * price_dynamics_score +
            0.15 * macro_score
        )
        
        # Determine signal based on score
        if final_score > 0.75:
            signal = "BUY"
        elif final_score < -0.75:
            signal = "SELL"
        else:
            signal = "HOLD"
        
        # Create explanation
        explanation = f"""
        Investment Signal for {stock} as of {date.strftime('%Y-%m-%d')}:
        
        NEWS ANALYSIS:
        - Sentiment: {news_summary['avg_sentiment']:.2f} ({news_summary['sentiment_trend']})
        - Recent Developments: {news_summary['recent_developments']}
        
        FUNDAMENTALS ANALYSIS:
        - Profitability: {fundamentals_summary['profitability']}
        - Growth: {fundamentals_summary['growth']}
        - Valuation: {fundamentals_summary['valuation']}
        - Financial Health: {fundamentals_summary['financial_health']}
        
        PRICE DYNAMICS ANALYSIS:
        - Short-term Momentum: {price_dynamics_summary['short_term_momentum']}
        - Relative Strength: {price_dynamics_summary['relative_strength']}
        - Volatility Profile: {price_dynamics_summary['volatility_profile']}
        - Risk-adjusted Performance: {price_dynamics_summary['risk_adjusted_performance']}
        
        MACROECONOMIC ENVIRONMENT:
        - Interest Rate Environment: {macro_summary['interest_rate_environment']}
        - Inflation Outlook: {macro_summary['inflation_outlook']}
        - Economic Growth: {macro_summary['economic_growth']}
        - Market Sentiment: {macro_summary['market_sentiment']}
        
        DECISION: {signal}
        
        REASONING:
        Based on the analysis of recent news, fundamentals, price dynamics, and the macroeconomic environment,
        """
        
        # Add reasoning based on the signal
        if signal == "BUY":
            explanation += f"""
            {stock} presents a compelling investment opportunity. The company shows {fundamentals_summary['profitability']} profitability, 
            {fundamentals_summary['growth']} growth, and is {fundamentals_summary['valuation']} with {fundamentals_summary['financial_health']} financial health. 
            Recent news sentiment is {news_summary['avg_sentiment']:.2f} and {news_summary['sentiment_trend']}, while price momentum is {price_dynamics_summary['short_term_momentum']} 
            with {price_dynamics_summary['relative_strength']} performance. The current macroeconomic environment with {macro_summary['economic_growth']} economic growth 
            and {macro_summary['market_sentiment']} market sentiment supports this position.
            """
        elif signal == "SELL":
            explanation += f"""
            {stock} faces significant challenges that make it an unattractive investment at this time. The company shows {fundamentals_summary['profitability']} profitability, 
            {fundamentals_summary['growth']} growth, and is {fundamentals_summary['valuation']} with {fundamentals_summary['financial_health']} financial health. 
            Recent news sentiment is {news_summary['avg_sentiment']:.2f} and {news_summary['sentiment_trend']}, while price momentum is {price_dynamics_summary['short_term_momentum']} 
            with {price_dynamics_summary['relative_strength']} performance. The current macroeconomic environment with {macro_summary['economic_growth']} economic growth 
            and {macro_summary['market_sentiment']} market sentiment further raises concerns.
            """
        else:  # HOLD
            explanation += f"""
            {stock} presents a mixed picture that suggests maintaining current positions without increasing exposure. The company shows {fundamentals_summary['profitability']} profitability, 
            {fundamentals_summary['growth']} growth, and is {fundamentals_summary['valuation']} with {fundamentals_summary['financial_health']} financial health. 
            Recent news sentiment is {news_summary['avg_sentiment']:.2f} and {news_summary['sentiment_trend']}, while price momentum is {price_dynamics_summary['short_term_momentum']} 
            with {price_dynamics_summary['relative_strength']} performance. The current macroeconomic environment with {macro_summary['economic_growth']} economic growth 
            and {macro_summary['market_sentiment']} market sentiment suggests caution.
            """
        
        return {
            'signal': signal,
            'explanation': explanation,
            'scores': {
                'news': news_score,
                'fundamentals': fundamentals_score,
                'price_dynamics': price_dynamics_score,
                'macro': macro_score,
                'final': final_score
            }
        }
    
    def _simulate_gpt_ranking(self, buy_signals, date):
        """
        Simulate the GPT-4 ranking mechanism for buy signals
        
        This function simulates how GPT-4 might rank explanations of "buy" signals
        """
        rankings = {}
        
        for stock, signal_data in buy_signals.items():
            # Extract key factors from the explanation
            explanation = signal_data['explanation']
            scores = signal_data['scores']
            
            # Base ranking on the final score, with some noise to simulate GPT-4's judgment
            base_ranking = min(10, max(0, 5 + scores['final'] * 5))
            noise = np.random.normal(0, 0.5)  # Add some variability
            
            # Ensure ranking is between 0 and 10
            ranking = min(10, max(0, base_ranking + noise))
            
            rankings[stock] = ranking
        
        return rankings
    
    def generate_monthly_signals(self):
        """
        Generate investment signals for each month and stock
        """
        # Define rebalance dates based on frequency
        if self.rebalance_freq == 'monthly':
            rebalance_dates = pd.date_range(start=self.start_date, end=self.end_date, freq='M')
        elif self.rebalance_freq == 'quarterly':
            rebalance_dates = pd.date_range(start=self.start_date, end=self.end_date, freq='Q')
        else:
            raise ValueError(f"Unsupported rebalance frequency: {self.rebalance_freq}")
        
        # Initialize DataFrame for signals
        signals = pd.DataFrame(index=rebalance_dates, columns=self.stocks)
        explanations = {}
        
        # Generate signals for each date and stock
        for date in rebalance_dates:
            print(f"Generating signals for {date.strftime('%Y-%m-%d')}...")
            
            date_buy_signals = {}
            
            for stock in self.stocks:
                decision = self._simulate_llm_decision(stock, date)
                signals.loc[date, stock] = decision['signal']
                
                if decision['signal'] == 'BUY':
                    date_buy_signals[stock] = decision
                
                key = (date.strftime('%Y-%m-%d'), stock)
                explanations[key] = decision['explanation']
            
            # Rank BUY signals
            if date_buy_signals:
                rankings = self._simulate_gpt_ranking(date_buy_signals, date)
                
                # Store rankings in a separate attribute
                if not hasattr(self, 'rankings'):
                    self.rankings = {}
                
                self.rankings[date.strftime('%Y-%m-%d')] = rankings
        
        self.data['signals'] = signals
        self.data['explanations'] = explanations
        
        return signals
    
    def backtest_portfolios(self):
        """
        Backtest portfolios based on the generated signals
        """
        if self.data['signals'] is None:
            self.generate_monthly_signals()
        
        # Define portfolios to test
        portfolios = {
            'MS': self._backtest_ms_portfolio,
            'MS-L': self._backtest_ms_long_only,
            'MS-Top10-GPT': self._backtest_ms_top10_gpt,
            'MS-High-GPT': self._backtest_ms_high_gpt,
            'S&P-Equivalent': self._backtest_sp_equivalent
        }
        
        results = {}
        
        # Run backtests for each portfolio
        for name, backtest_func in portfolios.items():
            print(f"Backtesting {name} portfolio...")
            results[name] = backtest_func()
        
        return results
    
    def _backtest_ms_portfolio(self):
        """Backtest the MarketSenseAI portfolio with both buy and sell signals"""
        signals = self.data['signals']
        prices = self.data['prices']
        
        # Initialize portfolio
        portfolio_value = 100.0  # Start with $100
        positions = {}  # Stock -> quantity
        cash = portfolio_value
        
        # Track portfolio value over time
        portfolio_values = []
        rebalance_dates = signals.index.tolist()
        
        # For each trading day
        for date in self.dates:
            # Check if it's a rebalance date
            if date in rebalance_dates:
                # Close all positions
                for stock, quantity in list(positions.items()):
                    if stock in prices.columns:
                        cash += quantity * prices.loc[date, stock]
                positions = {}
                
                # Get new signals
                current_signals = signals.loc[date]
                
                # Count buy and sell signals
                buy_signals = [stock for stock in current_signals.index if current_signals[stock] == 'BUY']
                sell_signals = [stock for stock in current_signals.index if current_signals[stock] == 'SELL']
                
                # Allocate capital
                if buy_signals:
                    buy_allocation = cash * 0.5 / len(buy_signals) if sell_signals else cash / len(buy_signals)
                    for stock in buy_signals:
                        stock_price = prices.loc[date, stock]
                        quantity = buy_allocation / stock_price
                        positions[stock] = quantity
                        cash -= buy_allocation
                
                if sell_signals:
                    # Short selling - borrow and sell
                    sell_allocation = cash * 0.5 / len(sell_signals) if buy_signals else cash / len(sell_signals)
                    for stock in sell_signals:
                        stock_price = prices.loc[date, stock]
                        # Short positions are represented as negative quantities
                        quantity = -sell_allocation / stock_price
                        positions[stock] = quantity
                        cash += sell_allocation  # Add to cash (proceeds from short sale)
            
            # Calculate portfolio value
            port_value = cash
            for stock, quantity in positions.items():
                if stock in prices.columns:
                    port_value += quantity * prices.loc[date, stock]
            
            portfolio_values.append(port_value)
        
        # Calculate performance metrics
        returns = pd.Series(portfolio_values).pct_change().dropna()
        
        metrics = {
            'portfolio_values': pd.Series(portfolio_values, index=self.dates),
            'total_return': portfolio_values[-1] / portfolio_values[0] - 1,
            'sharpe_ratio': returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0,
            'volatility': returns.std() * np.sqrt(252),
            'max_drawdown': self._calculate_max_drawdown(portfolio_values)
        }
        
        return metrics
    
    def _backtest_ms_long_only(self):
        """Backtest the MarketSenseAI portfolio with only buy signals"""
        signals = self.data['signals']
        prices = self.data['prices']
        
        # Initialize portfolio
        portfolio_value = 100.0  # Start with $100
        positions = {}  # Stock -> quantity
        cash = portfolio_value
        
        # Track portfolio value over time
        portfolio_values = []
        rebalance_dates = signals.index.tolist()
        
        # For each trading day
        for date in self.dates:
            # Check if it's a rebalance date
            if date in rebalance_dates:
                # Close all positions
                for stock, quantity in list(positions.items()):
                    if stock in prices.columns:
                        cash += quantity * prices.loc[date, stock]
                positions = {}
                
                # Get new signals
                current_signals = signals.loc[date]
                
                # Count buy signals
                buy_signals = [stock for stock in current_signals.index if current_signals[stock] == 'BUY']
                
                # Allocate capital
                if buy_signals:
                    buy_allocation = cash / len(buy_signals)
                    for stock in buy_signals:
                        stock_price = prices.loc[date, stock]
                        quantity = buy_allocation / stock_price
                        positions[stock] = quantity
                        cash -= buy_allocation
            
            # Calculate portfolio value
            port_value = cash
            for stock, quantity in positions.items():
                if stock in prices.columns:
                    port_value += quantity * prices.loc[date, stock]
            
            portfolio_values.append(port_value)
        
        # Calculate performance metrics
        returns = pd.Series(portfolio_values).pct_change().dropna()
        
        metrics = {
            'portfolio_values': pd.Series(portfolio_values, index=self.dates),
            'total_return': portfolio_values[-1] / portfolio_values[0] - 1,
            'sharpe_ratio': returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0,
            'volatility': returns.std() * np.sqrt(252),
            'max_drawdown': self._calculate_max_drawdown(portfolio_values)
        }
        
        return metrics
    
    def _backtest_ms_top10_gpt(self):
        """Backtest the MarketSenseAI portfolio with top 10 GPT-ranked buy signals"""
        signals = self.data['signals']
        prices = self.data['prices']
        
        # Initialize portfolio
        portfolio_value = 100.0  # Start with $100
        positions = {}  # Stock -> quantity
        cash = portfolio_value
        
        # Track portfolio value over time
        portfolio_values = []
        rebalance_dates = signals.index.tolist()
        
        # For each trading day
        for date in self.dates:
            # Check if it's a rebalance date
            if date in rebalance_dates:
                # Close all positions
                for stock, quantity in list(positions.items()):
                    if stock in prices.columns:
                        cash += quantity * prices.loc[date, stock]
                positions = {}
                
                # Get new signals and rankings
                current_signals = signals.loc[date]
                date_str = date.strftime('%Y-%m-%d')
                
                if hasattr(self, 'rankings') and date_str in self.rankings:
                    # Get buy signals and their rankings
                    buy_signals = [stock for stock in current_signals.index if current_signals[stock] == 'BUY']
                    
                    if buy_signals:
                        # Get rankings for available buy signals
                        rankings = {stock: self.rankings[date_str].get(stock, 0) for stock in buy_signals}
                        
                        # Sort by ranking and take top 10
                        top_stocks = sorted(rankings.items(), key=lambda x: x[1], reverse=True)[:10]
                        top_stocks = [stock for stock, _ in top_stocks]
                        
                        # Allocate capital
                        if top_stocks:
                            buy_allocation = cash / len(top_stocks)
                            for stock in top_stocks:
                                stock_price = prices.loc[date, stock]
                                quantity = buy_allocation / stock_price
                                positions[stock] = quantity
                                cash -= buy_allocation
            
            # Calculate portfolio value
            port_value = cash
            for stock, quantity in positions.items():
                if stock in prices.columns:
                    port_value += quantity * prices.loc[date, stock]
            
            portfolio_values.append(port_value)
        
        # Calculate performance metrics
        returns = pd.Series(portfolio_values).pct_change().dropna()
        
        metrics = {
            'portfolio_values': pd.Series(portfolio_values, index=self.dates),
            'total_return': portfolio_values[-1] / portfolio_values[0] - 1,
            'sharpe_ratio': returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0,
            'volatility': returns.std() * np.sqrt(252),
            'max_drawdown': self._calculate_max_drawdown(portfolio_values)
        }
        
        return metrics
    
    def _backtest_ms_high_gpt(self):
        """Backtest the MarketSenseAI portfolio with GPT-ranked buy signals > 7"""
        signals = self.data['signals']
        prices = self.data['prices']
        
        # Initialize portfolio
        portfolio_value = 100.0  # Start with $100
        positions = {}  # Stock -> quantity
        cash = portfolio_value
        
        # Track portfolio value over time
        portfolio_values = []
        rebalance_dates = signals.index.tolist()
        
        # For each trading day
        for date in self.dates:
            # Check if it's a rebalance date
            if date in rebalance_dates:
                # Close all positions
                for stock, quantity in list(positions.items()):
                    if stock in prices.columns:
                        cash += quantity * prices.loc[date, stock]
                positions = {}
                
                # Get new signals and rankings
                current_signals = signals.loc[date]
                date_str = date.strftime('%Y-%m-%d')
                
                if hasattr(self, 'rankings') and date_str in self.rankings:
                    # Get buy signals and their rankings
                    buy_signals = [stock for stock in current_signals.index if current_signals[stock] == 'BUY']
                    
                    if buy_signals:
                        # Get rankings for available buy signals
                        rankings = {stock: self.rankings[date_str].get(stock, 0) for stock in buy_signals}
                        
                        # Filter stocks with ranking > 7
                        high_ranked_stocks = [stock for stock, rank in rankings.items() if rank > 7]
                        
                        # Allocate capital
                        if high_ranked_stocks:
                            buy_allocation = cash / len(high_ranked_stocks)
                            for stock in high_ranked_stocks:
                                stock_price = prices.loc[date, stock]
                                quantity = buy_allocation / stock_price
                                positions[stock] = quantity
                                cash -= buy_allocation
            
            # Calculate portfolio value
            port_value = cash
            for stock, quantity in positions.items():
                if stock in prices.columns:
                    port_value += quantity * prices.loc[date, stock]
            
            portfolio_values.append(port_value)
        
        # Calculate performance metrics
        returns = pd.Series(portfolio_values).pct_change().dropna()
        
        metrics = {
            'portfolio_values': pd.Series(portfolio_values, index=self.dates),
            'total_return': portfolio_values[-1] / portfolio_values[0] - 1,
            'sharpe_ratio': returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0,
            'volatility': returns.std() * np.sqrt(252),
            'max_drawdown': self._calculate_max_drawdown(portfolio_values)
        }
        
        return metrics
    
    def _backtest_sp_equivalent(self):
        """Backtest the S&P equivalent portfolio"""
        prices = self.data['prices']
        
        # Use the S&P_EQUIVALENT column as portfolio value
        sp_values = prices['S&P_EQUIVALENT'].values
        
        # Normalize to start at 100
        sp_values = sp_values * 100 / sp_values[0]
        
        # Calculate performance metrics
        returns = pd.Series(sp_values).pct_change().dropna()
        
        metrics = {
            'portfolio_values': pd.Series(sp_values, index=self.dates),
            'total_return': sp_values[-1] / sp_values[0] - 1,
            'sharpe_ratio': returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0,
            'volatility': returns.std() * np.sqrt(252),
            'max_drawdown': self._calculate_max_drawdown(sp_values)
        }
        
        return metrics
    
    def _calculate_max_drawdown(self, portfolio_values):
        """Calculate maximum drawdown from a series of portfolio values"""
        # Convert to numpy array if it's not already
        values = np.array(portfolio_values)
        
        # Calculate the running maximum
        running_max = np.maximum.accumulate(values)
        
        # Calculate drawdowns
        drawdowns = values / running_max - 1
        
        # Find the maximum drawdown
        max_drawdown = np.min(drawdowns)
        
        return max_drawdown
    
    def plot_portfolios(self, backtest_results):
        """Plot portfolio performance"""
        plt.figure(figsize=(12, 8))
        
        for name, results in backtest_results.items():
            plt.plot(results['portfolio_values'], label=f'{name} (Return: {results["total_return"]*100:.1f}%, Sharpe: {results["sharpe_ratio"]:.2f})')
        
        plt.title('Portfolio Performance Comparison')
        plt.xlabel('Date')
        plt.ylabel('Portfolio Value')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    
    def analyze_signals(self):
        """Analyze the distribution and effectiveness of signals"""
        if self.data['signals'] is None:
            self.generate_monthly_signals()
        
        signals = self.data['signals']
        
        # Count signal types
        signal_counts = {}
        for col in signals.columns:
            signal_counts[col] = signals[col].value_counts().to_dict()
        
        # Convert to DataFrame for easier analysis
        signal_counts_df = pd.DataFrame(signal_counts).T.fillna(0)
        if 'BUY' not in signal_counts_df.columns:
            signal_counts_df['BUY'] = 0
        if 'SELL' not in signal_counts_df.columns:
            signal_counts_df['SELL'] = 0
        if 'HOLD' not in signal_counts_df.columns:
            signal_counts_df['HOLD'] = 0
        
        # Calculate signal effectiveness
        effectiveness = {}
        
        # For each rebalance date
        for date in signals.index[:-1]:  # Skip the last date as we can't evaluate forward returns
            next_date_idx = signals.index.get_loc(date) + 1
            if next_date_idx < len(signals.index):
                next_date = signals.index[next_date_idx]
                
                # For each stock
                for stock in signals.columns:
                    signal = signals.loc[date, stock]
                    
                    # Calculate return until next rebalance
                    if date in self.data['prices'].index and next_date in self.data['prices'].index:
                        start_price = self.data['prices'].loc[date, stock]
                        end_price = self.data['prices'].loc[next_date, stock]
                        stock_return = end_price / start_price - 1
                        
                        # Record effectiveness
                        if signal not in effectiveness:
                            effectiveness[signal] = []
                        
                        effectiveness[signal].append(stock_return)
        
        # Calculate average returns by signal type
        avg_returns = {}
        for signal, returns in effectiveness.items():
            avg_returns[signal] = np.mean(returns) if returns else 0
        
        # Plot signal distribution
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 2, 1)
        signal_counts_df.sum().plot(kind='bar', color=['green', 'yellow', 'red'])
        plt.title('Total Count of Each Signal Type')
        plt.ylabel('Count')
        plt.grid(True, axis='y')
        
        plt.subplot(2, 2, 2)
        for signal, returns in effectiveness.items():
            if returns:
                plt.hist(returns, bins=20, alpha=0.5, label=f'{signal} (mean: {np.mean(returns):.2%})')
        plt.title('Distribution of Returns by Signal Type')
        plt.xlabel('Return')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(2, 2, 3)
        signal_types = list(avg_returns.keys())
        avg_return_values = [avg_returns[s] for s in signal_types]
        bars = plt.bar(signal_types, avg_return_values, color=['green', 'yellow', 'red'])
        plt.title('Average Return by Signal Type')
        plt.ylabel('Average Return')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2%}',
                    ha='center', va='bottom')
        plt.grid(True, axis='y')
        
        plt.subplot(2, 2, 4)
        # Calculate hit rate (positive returns for BUY, negative for SELL)
        hit_rates = {}
        for signal, returns in effectiveness.items():
            if returns:
                if signal == 'BUY':
                    hit_rates[signal] = np.mean([r > 0 for r in returns])
                elif signal == 'SELL':
                    hit_rates[signal] = np.mean([r < 0 for r in returns])
                else:  # HOLD
                    hit_rates[signal] = np.mean([abs(r) < 0.05 for r in returns])  # Arbitrary threshold
        
        signal_types = list(hit_rates.keys())
        hit_rate_values = [hit_rates[s] for s in signal_types]
        bars = plt.bar(signal_types, hit_rate_values, color=['green', 'yellow', 'red'])
        plt.title('Hit Rate by Signal Type')
        plt.ylabel('Hit Rate')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2%}',
                    ha='center', va='bottom')
        plt.grid(True, axis='y')
        
        plt.tight_layout()
        plt.show()
        
        return {
            'signal_counts': signal_counts_df,
            'avg_returns': avg_returns,
            'hit_rates': hit_rates if 'hit_rates' in locals() else None
        }
    
    def analyze_gpt_rankings(self):
        """Analyze the effectiveness of GPT rankings"""
        if not hasattr(self, 'rankings'):
            print("No GPT rankings available. Run generate_monthly_signals() first.")
            return None
        
        signals = self.data['signals']
        
        # Analyze correlation between GPT ranking and subsequent returns
        ranking_effectiveness = []
        
        # For each rebalance date
        for date in signals.index[:-1]:  # Skip the last date
            date_str = date.strftime('%Y-%m-%d')
            
            if date_str in self.rankings:
                next_date_idx = signals.index.get_loc(date) + 1
                if next_date_idx < len(signals.index):
                    next_date = signals.index[next_date_idx]
                    
                    # For each ranked stock
                    for stock, ranking in self.rankings[date_str].items():
                        # Calculate return until next rebalance
                        if date in self.data['prices'].index and next_date in self.data['prices'].index:
                            start_price = self.data['prices'].loc[date, stock]
                            end_price = self.data['prices'].loc[next_date, stock]
                            stock_return = end_price / start_price - 1
                            
                            ranking_effectiveness.append({
                                'date': date,
                                'stock': stock,
                                'ranking': ranking,
                                'return': stock_return
                            })
        
        if not ranking_effectiveness:
            print("No ranking effectiveness data available.")
            return None
        
        # Convert to DataFrame
        ranking_df = pd.DataFrame(ranking_effectiveness)
        
        # Plot ranking effectiveness
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 2, 1)
        plt.scatter(ranking_df['ranking'], ranking_df['return'], alpha=0.5)
        
        # Add trend line
        z = np.polyfit(ranking_df['ranking'], ranking_df['return'], 1)
        p = np.poly1d(z)
        plt.plot(ranking_df['ranking'], p(ranking_df['ranking']), "r--")
        
        plt.title('GPT Ranking vs. Subsequent Return')
        plt.xlabel('GPT Ranking')
        plt.ylabel('Return')
        plt.grid(True)
        
        # Calculate correlation
        corr = ranking_df[['ranking', 'return']].corr().iloc[0, 1]
        plt.text(0.05, 0.95, f'Correlation: {corr:.2f}', transform=plt.gca().transAxes)
        
        plt.subplot(2, 2, 2)
        # Group by ranking (rounded) and calculate average return
        ranking_df['ranking_rounded'] = ranking_df['ranking'].round()
        avg_return_by_ranking = ranking_df.groupby('ranking_rounded')['return'].mean()
        avg_return_by_ranking.plot(kind='bar')
        plt.title('Average Return by GPT Ranking')
        plt.xlabel('GPT Ranking (Rounded)')
        plt.ylabel('Average Return')
        plt.grid(True, axis='y')
        
        plt.subplot(2, 2, 3)
        # Calculate hit rate (positive returns) by ranking
        hit_rate_by_ranking = ranking_df.groupby('ranking_rounded')['return'].apply(lambda x: (x > 0).mean())
        hit_rate_by_ranking.plot(kind='bar')
        plt.title('Hit Rate by GPT Ranking')
        plt.xlabel('GPT Ranking (Rounded)')
        plt.ylabel('Hit Rate')
        plt.grid(True, axis='y')
        
        plt.subplot(2, 2, 4)
        # Bucket rankings into high, medium, low
        ranking_df['ranking_bucket'] = pd.cut(
            ranking_df['ranking'], 
            bins=[0, 3, 7, 10], 
            labels=['Low (0-3)', 'Medium (4-7)', 'High (8-10)']
        )
        
        bucket_stats = ranking_df.groupby('ranking_bucket')['return'].agg(['mean', 'std', 'count'])
        bucket_stats['hit_rate'] = ranking_df.groupby('ranking_bucket')['return'].apply(lambda x: (x > 0).mean())
        
        bucket_stats['mean'].plot(kind='bar')
        plt.title('Average Return by Ranking Bucket')
        plt.xlabel('GPT Ranking Bucket')
        plt.ylabel('Average Return')
        plt.grid(True, axis='y')
        
        # Add text with statistics
        for i, bucket in enumerate(bucket_stats.index):
            stats = bucket_stats.loc[bucket]
            plt.text(
                i, 
                stats['mean'] + 0.01, 
                f"Hit Rate: {stats['hit_rate']:.2f}\nCount: {stats['count']}", 
                ha='center'
            )
        
        plt.tight_layout()
        plt.show()
        
        return bucket_stats

# Run the simulation
sim = MarketSenseAISimulation(
    num_stocks=30,
    start_date='2022-01-01',
    end_date='2023-12-31',
    rebalance_freq='monthly'
)

# Generate monthly signals
signals = sim.generate_monthly_signals()

# Print a few example signals and explanations
print("\nExample Signals and Explanations:")
for date in signals.index[:2]:  # First two months
    for stock in signals.columns[:3]:  # First three stocks
        print(f"Date: {date.strftime('%Y-%m-%d')}, Stock: {stock}, Signal: {signals.loc[date, stock]}")
        key = (date.strftime('%Y-%m-%d'), stock)
        if key in sim.data['explanations']:
            explanation_lines = sim.data['explanations'][key].split('\n')
            print(f"Explanation (summary): {explanation_lines[0]}")
            print(f"Decision: {explanation_lines[-7]}")
            print("-" * 40)

# Backtest portfolios
backtest_results = sim.backtest_portfolios()

# Plot portfolio performance
sim.plot_portfolios(backtest_results)

# Analyze signals
signal_analysis = sim.analyze_signals()

# Analyze GPT rankings
ranking_analysis = sim.analyze_gpt_rankings()

# Print performance comparison
print("\nPortfolio Performance Comparison:")
performance_summary = pd.DataFrame({
    'Portfolio': list(backtest_results.keys()),
    'Total Return': [results['total_return'] for results in backtest_results.values()],
    'Sharpe Ratio': [results['sharpe_ratio'] for results in backtest_results.values()],
    'Volatility': [results['volatility'] for results in backtest_results.values()],
    'Max Drawdown': [results['max_drawdown'] for results in backtest_results.values()],
})
performance_summary.set_index('Portfolio', inplace=True)
performance_summary['Total Return'] = performance_summary['Total Return'].apply(lambda x: f"{x*100:.2f}%")
performance_summary['Sharpe Ratio'] = performance_summary['Sharpe Ratio'].apply(lambda x: f"{x:.2f}")
performance_summary['Volatility'] = performance_summary['Volatility'].apply(lambda x: f"{x*100:.2f}%")
performance_summary['Max Drawdown'] = performance_summary['Max Drawdown'].apply(lambda x: f"{x*100:.2f}%")
print(performance_summary)