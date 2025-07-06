import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from datetime import datetime, timedelta
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from transformers import TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
import torch
from datasets import Dataset
import warnings
warnings.filterwarnings('ignore')

# Download NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

class InvestmentDataSimulator:
    """Class to simulate investment research data for testing purposes"""
    
    def __init__(self):
        self.sectors = [
            'Technology', 'Healthcare', 'Consumer Discretionary', 'Financials', 
            'Energy', 'Industrials', 'Materials', 'Utilities', 'Real Estate', 
            'Consumer Staples', 'Communication Services'
        ]
        
        self.market_conditions = [
            'Inflation', 'Interest Rates', 'GDP Growth', 'Unemployment', 
            'Consumer Sentiment', 'Manufacturing Output', 'Housing Market', 
            'Trade Balance', 'Monetary Policy', 'Fiscal Policy'
        ]
        
        self.events = [
            'Interest Rate Hike', 'Interest Rate Cut', 'Oil Price Increase', 
            'Oil Price Decrease', 'Strong GDP Report', 'Weak GDP Report', 
            'Inflation Surge', 'Deflation Risks', 'Supply Chain Disruption',
            'Tech Innovation Breakthrough', 'Regulatory Changes', 'Trade War',
            'Pandemic Concerns', 'Economic Stimulus', 'Geopolitical Tensions'
        ]
        
        self.companies = {
            'Technology': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'META', 'ADBE', 'CRM', 'INTC'],
            'Healthcare': ['JNJ', 'PFE', 'UNH', 'MRK', 'ABBV', 'ABT', 'TMO', 'LLY', 'AMGN', 'BMY'],
            'Consumer Discretionary': ['AMZN', 'HD', 'MCD', 'NKE', 'SBUX', 'TGT', 'LOW', 'BKNG', 'MAR', 'TJX'],
            'Financials': ['JPM', 'BAC', 'WFC', 'C', 'GS', 'MS', 'BLK', 'AXP', 'CB', 'MMC'],
            'Energy': ['XOM', 'CVX', 'COP', 'SLB', 'EOG', 'PXD', 'PSX', 'VLO', 'KMI', 'WMB'],
            'Industrials': ['HON', 'UNP', 'UPS', 'BA', 'CAT', 'DE', 'GE', 'LMT', 'RTX', 'MMM'],
            'Materials': ['LIN', 'APD', 'ECL', 'SHW', 'FCX', 'NEM', 'NUE', 'VMC', 'MLM', 'DOW'],
            'Utilities': ['NEE', 'DUK', 'SO', 'D', 'AEP', 'XEL', 'SRE', 'ED', 'EXC', 'WEC'],
            'Real Estate': ['AMT', 'PLD', 'CCI', 'EQIX', 'PSA', 'AVB', 'EQR', 'DLR', 'SPG', 'O'],
            'Consumer Staples': ['PG', 'KO', 'PEP', 'WMT', 'COST', 'CL', 'EL', 'GIS', 'K', 'SJM'],
            'Communication Services': ['GOOGL', 'META', 'NFLX', 'CMCSA', 'VZ', 'T', 'TMUS', 'DIS', 'EA', 'ATVI']
        }
        
        # Event to sector impact mapping (positive or negative)
        self.event_sector_impact = {
            'Interest Rate Hike': {'Financials': 1, 'Real Estate': -1, 'Utilities': -1, 'Consumer Discretionary': -1},
            'Interest Rate Cut': {'Financials': -1, 'Real Estate': 1, 'Utilities': 1, 'Consumer Discretionary': 1},
            'Oil Price Increase': {'Energy': 1, 'Transportation': -1, 'Airlines': -1, 'Consumer Discretionary': -1},
            'Oil Price Decrease': {'Energy': -1, 'Transportation': 1, 'Airlines': 1, 'Consumer Discretionary': 1},
            'Strong GDP Report': {'Financials': 1, 'Consumer Discretionary': 1, 'Industrials': 1},
            'Weak GDP Report': {'Financials': -1, 'Consumer Discretionary': -1, 'Industrials': -1},
            'Inflation Surge': {'Consumer Staples': 1, 'Real Estate': 1, 'Materials': 1, 'Consumer Discretionary': -1},
            'Deflation Risks': {'Consumer Staples': -1, 'Real Estate': -1, 'Materials': -1, 'Technology': 1},
            'Supply Chain Disruption': {'Industrials': -1, 'Consumer Discretionary': -1, 'Technology': -1},
            'Tech Innovation Breakthrough': {'Technology': 1, 'Communication Services': 1},
            'Regulatory Changes': {'Financials': -1, 'Healthcare': -1, 'Technology': -1},
            'Trade War': {'Industrials': -1, 'Technology': -1, 'Materials': -1},
            'Pandemic Concerns': {'Healthcare': 1, 'Technology': 1, 'Consumer Staples': 1, 'Travel': -1},
            'Economic Stimulus': {'Consumer Discretionary': 1, 'Financials': 1, 'Industrials': 1},
            'Geopolitical Tensions': {'Defense': 1, 'Energy': 1, 'Utilities': 1, 'Consumer Discretionary': -1}
        }
        
        # Investment philosophies
        self.investment_philosophies = [
            "Value Investing", "Growth Investing", "Income Investing", 
            "Momentum Investing", "Contrarian Investing", "Index Investing"
        ]
    
    def generate_research_reports(self, num_reports=100):
        """Generate simulated investment research reports"""
        reports = []
        
        for _ in range(num_reports):
            # Choose random sector
            sector = np.random.choice(self.sectors)
            
            # Generate report details
            report_date = datetime.now() - timedelta(days=np.random.randint(1, 365))
            title = f"{sector} Sector Analysis: Opportunities and Risks"
            
            # Generate market condition analysis
            market_conditions = np.random.choice(self.market_conditions, size=3, replace=False)
            market_analysis = f"Market Conditions Analysis:\n"
            for condition in market_conditions:
                sentiment = np.random.choice(['positive', 'negative', 'neutral'])
                market_analysis += f"- {condition}: {sentiment} impact on markets\n"
            
            # Generate sector outlook
            outlook = np.random.choice(['Positive', 'Negative', 'Neutral', 'Mixed'])
            sector_analysis = f"Sector Outlook: {outlook}\n"
            
            # Select companies from sector
            selected_companies = self.companies.get(sector, [])
            if not selected_companies:
                selected_companies = np.random.choice(list(self.companies.values())[0], size=3)
            else:
                selected_companies = np.random.choice(selected_companies, 
                                                    size=min(3, len(selected_companies)), 
                                                    replace=False)
            
            # Generate company analysis
            company_analysis = "Company Analysis:\n"
            for company in selected_companies:
                sentiment = np.random.choice(['Buy', 'Sell', 'Hold'])
                target_price = round(np.random.uniform(50, 500), 2)
                company_analysis += f"- {company}: {sentiment} recommendation, Target Price: ${target_price}\n"
            
            # Generate full report
            report = f"""
            {title}
            Date: {report_date.strftime('%Y-%m-%d')}
            
            {market_analysis}
            
            {sector_analysis}
            
            {company_analysis}
            
            Investment Thesis:
            {self._generate_investment_thesis(sector, outlook, selected_companies)}
            """
            
            reports.append({
                'title': title,
                'date': report_date,
                'sector': sector,
                'content': report,
                'companies': selected_companies.tolist()
            })
        
        return pd.DataFrame(reports)
    
    def generate_market_news(self, num_news=200):
        """Generate simulated market news headlines and content"""
        news = []
        
        for _ in range(num_news):
            # Choose random event
            event = np.random.choice(self.events)
            
            # Generate news details
            news_date = datetime.now() - timedelta(days=np.random.randint(1, 180))
            
            # Get affected sectors
            affected_sectors = self.event_sector_impact.get(event, {})
            if not affected_sectors:
                affected_sectors = {np.random.choice(self.sectors): np.random.choice([-1, 1])}
            
            # Create headline
            sector_mention = np.random.choice(list(affected_sectors.keys())) if affected_sectors else np.random.choice(self.sectors)
            headline = f"{event} Expected to Impact {sector_mention} Sector"
            
            # Generate content
            content = f"""
            {headline}
            
            Date: {news_date.strftime('%Y-%m-%d')}
            
            Details:
            The recent {event} is expected to have significant implications for markets, 
            particularly in the {sector_mention} sector. Analysts are closely monitoring 
            the situation to assess potential opportunities and risks.
            
            Affected Sectors:
            """
            
            # Add affected sectors to content
            for sector, impact in affected_sectors.items():
                impact_text = "positive" if impact > 0 else "negative"
                content += f"- {sector}: {impact_text} impact\n"
            
            # Add affected companies
            affected_companies = []
            for sector in affected_sectors:
                if sector in self.companies:
                    companies = np.random.choice(self.companies[sector], 
                                                size=min(2, len(self.companies[sector])), 
                                                replace=False).tolist()
                    affected_companies.extend(companies)
            
            content += "\nAffected Companies:\n"
            for company in affected_companies:
                impact = np.random.choice(["Expected to benefit", "May face challenges"])
                content += f"- {company}: {impact}\n"
            
            news.append({
                'headline': headline,
                'date': news_date,
                'event': event,
                'content': content,
                'affected_sectors': list(affected_sectors.keys()),
                'affected_companies': affected_companies
            })
        
        return pd.DataFrame(news)
    
    def generate_stock_data(self, tickers, start_date, end_date):
        """Generate simulated stock price data for the given tickers"""
        # Create date range
        date_range = pd.date_range(start=start_date, end=end_date, freq='B')  # Business days
        
        # Initialize dataframe
        data = pd.DataFrame(index=date_range)
        
        # Add data for each ticker
        for ticker in tickers:
            # Generate random price series with some trend and volatility
            base_price = np.random.uniform(50, 200)
            trend = np.random.uniform(-0.0005, 0.0005)  # Daily trend
            volatility = np.random.uniform(0.01, 0.03)   # Daily volatility
            
            prices = [base_price]
            for i in range(1, len(date_range)):
                # Random walk with drift
                prev_price = prices[-1]
                change = trend + np.random.normal(0, volatility)
                new_price = prev_price * (1 + change)
                prices.append(max(new_price, 1.0))  # Ensure price doesn't go below 1
            
            data[ticker] = prices
        
        # Add market index
        market_base = 1000
        market_trend = 0.0002  # Slight upward bias
        market_vol = 0.01
        
        market_prices = [market_base]
        for i in range(1, len(date_range)):
            prev_price = market_prices[-1]
            change = market_trend + np.random.normal(0, market_vol)
            new_price = prev_price * (1 + change)
            market_prices.append(new_price)
        
        data['SPY'] = market_prices
        
        return data
    
    def generate_qa_pairs(self, num_pairs=50):
        """Generate question-answer pairs for instruction fine-tuning"""
        qa_pairs = []
        
        # Investment philosophy questions
        for philosophy in self.investment_philosophies:
            question = f"What is {philosophy} and how should I apply it to my portfolio?"
            answer = self._generate_investment_philosophy_answer(philosophy)
            qa_pairs.append({"question": question, "answer": answer})
        
        # Event impact questions
        for event in np.random.choice(self.events, size=min(10, len(self.events)), replace=False):
            question = f"How would {event} affect the market and which sectors would benefit or suffer?"
            answer = self._generate_event_impact_answer(event)
            qa_pairs.append({"question": question, "answer": answer})
        
        # Sector analysis questions
        for sector in np.random.choice(self.sectors, size=min(5, len(self.sectors)), replace=False):
            question = f"What is your outlook on the {sector} sector and which stocks do you recommend?"
            answer = self._generate_sector_recommendation(sector)
            qa_pairs.append({"question": question, "answer": answer})
        
        # Stock recommendation questions
        all_companies = []
        for sector_companies in self.companies.values():
            all_companies.extend(sector_companies)
        
        for company in np.random.choice(all_companies, size=min(15, len(all_companies)), replace=False):
            question = f"Should I buy, sell, or hold {company}? Please provide your analysis."
            answer = self._generate_stock_recommendation(company)
            qa_pairs.append({"question": question, "answer": answer})
        
        # Market condition questions
        for condition in np.random.choice(self.market_conditions, size=min(10, len(self.market_conditions)), replace=False):
            question = f"How does {condition} typically impact stock markets and investment strategies?"
            answer = self._generate_market_condition_answer(condition)
            qa_pairs.append({"question": question, "answer": answer})
        
        # Fill remaining with stock comparisons and portfolio questions
        while len(qa_pairs) < num_pairs:
            # Stock comparison questions
            companies = np.random.choice(all_companies, size=2, replace=False)
            question = f"Which is a better investment right now: {companies[0]} or {companies[1]}? Please explain your reasoning."
            answer = self._generate_stock_comparison(companies[0], companies[1])
            qa_pairs.append({"question": question, "answer": answer})
            
            if len(qa_pairs) >= num_pairs:
                break
                
            # Portfolio allocation questions
            risk_profile = np.random.choice(['conservative', 'moderate', 'aggressive'])
            time_horizon = np.random.choice(['short-term', 'medium-term', 'long-term'])
            question = f"How should I allocate my portfolio if I have a {risk_profile} risk tolerance and a {time_horizon} investment horizon?"
            answer = self._generate_portfolio_allocation(risk_profile, time_horizon)
            qa_pairs.append({"question": question, "answer": answer})
        
        return pd.DataFrame(qa_pairs)
    
    def _generate_investment_thesis(self, sector, outlook, companies):
        """Generate a simulated investment thesis"""
        outlook_text = {
            'Positive': "We maintain a positive outlook on the sector due to favorable market conditions and strong growth potential.",
            'Negative': "We have concerns about the sector's performance due to challenging market conditions and headwinds.",
            'Neutral': "We maintain a neutral stance on the sector, with both opportunities and challenges ahead.",
            'Mixed': "The outlook for the sector is mixed, with varying prospects across different sub-sectors and companies."
        }
        
        thesis = outlook_text.get(outlook, "")
        
        # Add company-specific insights
        thesis += " Our analysis indicates that "
        
        for i, company in enumerate(companies):
            if i > 0:
                thesis += " Meanwhile, " if i < len(companies) - 1 else " Finally, "
            
            sentiment = np.random.choice(['is well-positioned for growth', 'faces significant challenges', 'presents a compelling valuation'])
            reason = np.random.choice(['strong product pipeline', 'expanding market share', 'cost-cutting initiatives', 'regulatory challenges', 'competitive pressures'])
            
            thesis += f"{company} {sentiment} due to its {reason}."
        
        return thesis
    
    def _generate_investment_philosophy_answer(self, philosophy):
        """Generate an answer about an investment philosophy"""
        philosophies = {
            "Value Investing": """
            Value investing is an investment strategy that involves picking stocks that appear to be trading for less than their intrinsic or book value. Value investors actively seek stocks they believe the market has undervalued.
            
            To apply value investing to your portfolio:
            
            1. Focus on companies with low Price-to-Earnings (P/E) ratios, low Price-to-Book (P/B) ratios, and high dividend yields
            2. Look for strong fundamentals including solid balance sheets, stable earnings, and low debt
            3. Maintain a long-term investment horizon, as value stocks may take time to appreciate
            4. Perform thorough fundamental analysis on potential investments
            5. Be patient and disciplined, as value investing requires a contrarian mindset
            
            Top 3 value investing principles:
            1. Margin of Safety - Always buy at a significant discount to intrinsic value
            2. Mr. Market - View market volatility as an opportunity, not a guide
            3. Circle of Competence - Invest in businesses you understand
            
            Value investing was pioneered by Benjamin Graham and popularized by Warren Buffett, who has consistently outperformed the market using this approach.
            """,
            
            "Growth Investing": """
            Growth investing focuses on companies that exhibit signs of above-average growth, even if the share price appears expensive in terms of metrics like price-to-earnings ratios.
            
            To apply growth investing to your portfolio:
            
            1. Look for companies with strong year-over-year revenue and earnings growth
            2. Focus on industries and sectors with substantial growth potential
            3. Pay attention to companies with competitive advantages or market leadership
            4. Be prepared to accept higher volatility and potentially higher valuations
            5. Regularly review holdings to ensure growth stories remain intact
            
            Top 3 growth investing principles:
            1. Future Potential - Prioritize future growth prospects over current valuations
            2. Innovation Focus - Seek companies disrupting industries or creating new markets
            3. Earnings Acceleration - Look for companies with increasing rates of earnings growth
            
            Growth investing has been championed by investors like Philip Fisher and Peter Lynch, and is particularly effective in bull markets and innovation-driven economies.
            """,
            
            "Income Investing": """
            Income investing is a strategy focused on building a portfolio of assets that generates a steady stream of income through dividends, interest payments, or other distributions.
            
            To apply income investing to your portfolio:
            
            1. Focus on dividend-paying stocks with history of stable or growing dividends
            2. Consider fixed-income securities like bonds, preferred shares, and REITs
            3. Look for companies with sustainable payout ratios and strong cash flows
            4. Diversify across different income-generating assets and sectors
            5. Reinvest dividends when possible to compound returns
            
            Top 3 income investing principles:
            1. Yield Sustainability - Prioritize sustainable yields over highest current yields
            2. Income Growth - Look for companies with consistent dividend growth history
            3. Diversification - Spread investments across multiple income sources
            
            Income investing is particularly suitable for retirees or those seeking regular cash flow from their investments without needing to sell principal assets.
            """,
            
            "Momentum Investing": """
            Momentum investing is a strategy that aims to capitalize on the continuance of existing trends in the market by buying assets that have shown an upward price trend.
            
            To apply momentum investing to your portfolio:
            
            1. Look for stocks with strong price performance over the past 3-12 months
            2. Focus on relative strength compared to the broader market
            3. Implement a systematic approach with clear entry and exit rules
            4. Be prepared to actively manage positions as momentum can quickly reverse
            5. Consider both absolute momentum (a security's own past performance) and relative momentum (performance compared to peers)
            
            Top 3 momentum investing principles:
            1. Trend Following - "The trend is your friend" until it shows clear signs of reversal
            2. Disciplined Exit Strategy - Have clear rules for when to exit positions
            3. Risk Management - Use position sizing and stop-losses to manage downside risk
            
            Momentum investing tends to work best in strong bull markets but requires active management and can lead to higher turnover and potentially higher tax consequences.
            """,
            
            "Contrarian Investing": """
            Contrarian investing involves deliberately going against prevailing market trends by selling when others are buying and buying when others are selling.
            
            To apply contrarian investing to your portfolio:
            
            1. Look for assets, sectors, or markets that are currently out of favor
            2. Develop metrics to identify when an asset is truly undervalued vs. declining for fundamental reasons
            3. Maintain strong conviction and patience as contrarian positions may take time to pay off
            4. Perform thorough fundamental analysis to ensure the underlying value exists
            5. Consider sentiment indicators and extreme market movements as potential signals
            
            Top 3 contrarian investing principles:
            1. Psychological Discipline - Train yourself to act against emotional market behavior
            2. Fundamental Value - Ensure there's underlying value in your contrarian picks
            3. Patience - Be prepared for potentially long periods of underperformance before thesis plays out
            
            Contrarian investing can be psychologically challenging but has been successfully employed by investors like David Dreman and Marc Faber to identify opportunities when markets reach extremes of pessimism or optimism.
            """,
            
            "Index Investing": """
            Index investing is a passive investment strategy that seeks to replicate the performance of a specific market index, such as the S&P 500, by investing in all (or a representative sample) of the securities in that index.
            
            To apply index investing to your portfolio:
            
            1. Select index funds or ETFs that track your desired market segments
            2. Focus on funds with low expense ratios to maximize returns
            3. Consider a core-satellite approach, using index funds as the core with potential active strategies as satellites
            4. Implement dollar-cost averaging by investing regularly regardless of market conditions
            5. Rebalance periodically to maintain your desired asset allocation
            
            Top 3 index investing principles:
            1. Market Efficiency - Markets are generally efficient, making it difficult to consistently outperform
            2. Cost Minimization - Lower costs directly improve net returns
            3. Diversification - Broad market exposure reduces individual security risk
            
            Index investing has been championed by John Bogle, founder of Vanguard, and is supported by extensive academic research showing that most active managers fail to consistently outperform their benchmark indices after fees.
            """
        }
        
        return philosophies.get(philosophy, "Investment philosophy details not available.")
    
    def _generate_event_impact_answer(self, event):
        """Generate an answer about an event's impact on markets"""
        impacts = {
            'Interest Rate Hike': """
            An interest rate hike typically has the following market impacts:
            
            Stock Recommendation: JPM (Buy)
            
            Top 3 Reasons:
            1. Banks benefit from higher net interest margins during rate hikes
            2. JPM has strong balance sheet and diversified revenue streams
            3. Historically outperforms peers during tightening cycles
            
            Detailed Explanation:
            Interest rate hikes generally have a mixed impact on the market. Financials, particularly banks like JPMorgan Chase (JPM), typically benefit as higher rates increase their net interest margins. Banks can earn more on the spread between what they pay depositors and what they charge for loans.
            
            Sectors negatively impacted include:
            - Real Estate: Higher mortgage rates reduce affordability and transaction volumes
            - Utilities: These yield-sensitive sectors become less attractive as bonds offer better yields
            - Consumer Discretionary: Higher borrowing costs reduce consumer spending
            - Growth Technology: Higher discount rates reduce the present value of future earnings
            
            Companies with strong balance sheets and low debt levels are generally less affected by rate hikes. JPM is particularly well-positioned due to its diversified business model, strong risk management, and ability to benefit from increased trading volatility that often accompanies rate changes.
            
            For investors, a balanced approach is recommended with overweight positions in financials, underweight in interest-rate sensitive sectors, and a focus on companies with pricing power and low debt levels.
            """,
            
            'Oil Price Increase': """
            An oil price increase typically has the following market impacts:
            
            Stock Recommendation: CVX (Buy)
            
            Top 3 Reasons:
            1. Direct beneficiary of higher oil prices with strong production portfolio
            2. Integrated business model provides stability across price cycles
            3. Strong balance sheet and sustainable dividend even in lower price environments
            
            Detailed Explanation:
            When oil prices increase significantly, the market experiences several sector-specific impacts:
            
            Sectors positively impacted:
            - Energy: Companies like Chevron (CVX) benefit directly through higher revenues and margins
            - Energy Services: Companies providing drilling and production services see increased demand
            - Industrials related to energy: Companies manufacturing equipment for energy production
            
            Sectors negatively impacted:
            - Airlines: Fuel is their largest variable cost, pressuring margins
            - Transportation: Higher fuel costs impact shipping and logistics companies
            - Consumer Discretionary: Higher gas prices reduce disposable income for other spending
            
            Chevron (CVX) is particularly well-positioned due to its:
            - Low production costs compared to peers
            - Integrated model balancing upstream and downstream operations
            - Strong balance sheet with relatively low debt
            - History of dividend growth even through oil price cycles
            
            For portfolio strategy, consider overweighting energy stocks while being selective about companies with lower production costs and strong balance sheets. Be cautious on consumer discretionary and transportation sectors if oil prices appear to be in a sustained uptrend.
            """,
            
            'Strong GDP Report': """
            A strong GDP report typically has the following market impacts:
            
            Stock Recommendation: CAT (Buy)
            
            Top 3 Reasons:
            1. Cyclical business directly benefits from economic growth
            2. Infrastructure spending typically increases during strong economic periods
            3. Strong global presence to capture worldwide growth opportunities
            
            Detailed Explanation:
            A strong GDP report generally signals robust economic activity, which has several market implications:
            
            Sectors positively impacted:
            - Industrials: Companies like Caterpillar (CAT) benefit from increased capital expenditure
            - Financials: Banks see loan growth and fewer defaults during economic expansion
            - Consumer Discretionary: Increased consumer confidence leads to higher spending
            - Materials: Demand for raw materials increases with production and construction
            
            Sectors with limited impact:
            - Consumer Staples: These defensive sectors typically underperform during strong economic growth
            - Utilities: Considered defensive and may see less relative performance
            
            Caterpillar (CAT) is particularly well-positioned because:
            - Its equipment is essential for construction and infrastructure projects
            - Economic growth often correlates with increased mining activity
            - Government infrastructure initiatives often follow strong economic data
            - Global presence allows it to benefit from worldwide growth trends
            
            For investment strategy, consider overweighting cyclical sectors while maintaining some defensive positions as a hedge against potential economic slowdown later in the cycle. Monitor for signs of overheating or inflation that might prompt central bank tightening.
            """,
            
            'Inflation Surge': """
            An inflation surge typically has the following market impacts:
            
            Stock Recommendation: LIN (Buy)
            
            Top 3 Reasons:
            1. Strong pricing power allows passing inflation costs to customers
            2. Essential products with inelastic demand across various industries
            3. Historically outperforms during inflationary periods with stable margins
            
            Detailed Explanation:
            An inflation surge creates a complex market environment with varying impacts across sectors:
            
            Sectors that typically perform well:
            - Materials: Companies like Linde (LIN) can often pass through higher costs
            - Energy: Energy prices often drive inflation and benefit producers
            - Real Estate: Hard assets can serve as inflation hedges
            - Consumer Staples with pricing power: Can maintain margins by raising prices
            
            Sectors negatively impacted:
            - Growth Technology: Higher discount rates due to potential interest rate increases
            - Consumer Discretionary without pricing power: Margin compression
            - Fixed Income: Erodes the real value of future interest payments
            
            Linde (LIN) is particularly well-positioned because:
            - It supplies essential industrial gases with limited substitution options
            - Long-term contracts often include inflation adjustment clauses
            - Diverse customer base across healthcare, manufacturing, and food processing
            - Consolidated industry structure enables pricing discipline
            
            For portfolio positioning, focus on companies with pricing power, low labor intensity, and ownership of hard assets. Reduce exposure to companies with high P/E ratios whose future earnings will be more heavily discounted. Consider allocations to TIPS and commodities as inflation hedges.
            """,
            
            'Pandemic Concerns': """
            Pandemic concerns typically have the following market impacts:
            
            Stock Recommendation: JNJ (Buy)
            
            Top 3 Reasons:
            1. Defensive healthcare business with stable demand regardless of economic conditions
            2. Pharmaceutical division can benefit from increased healthcare spending and potential vaccine development
            3. Strong balance sheet provides stability during market uncertainty
            
            Detailed Explanation:
            Pandemic concerns create significant market volatility with clear sector divergence:
            
            Sectors that typically benefit:
            - Healthcare: Companies like Johnson & Johnson (JNJ) see sustained or increased demand
            - Technology: Remote work and digital services become essential
            - Consumer Staples: Essential products continue to be purchased regardless of conditions
            - Select Biotech: Companies involved in vaccines or treatments
            
            Sectors negatively impacted:
            - Travel and Leisure: Direct impact from mobility restrictions
            - Retail (brick and mortar): Foot traffic declines significantly
            - Energy: Reduced transportation leads to lower oil demand
            - Financials: Concerns about loan defaults and economic contraction
            
            Johnson & Johnson (JNJ) is particularly well-positioned because:
            - Diversified healthcare business spanning pharmaceuticals, medical devices, and consumer health
            - Essential products that maintain demand even during economic contraction
            - R&D capabilities that could contribute to pandemic solutions
            - AAA-rated balance sheet with significant cash reserves
            
            For portfolio strategy during pandemic concerns, increase allocation to quality companies with strong balance sheets, reduce cyclical exposure, and consider defensive sectors. Maintain liquidity for potential opportunities as markets often overreact to pandemic news.
            """
        }
        
        return impacts.get(event, f"""
        The {event} would likely have significant but varied impacts across different market sectors.
        
        Stock Recommendation: Will depend on specific sectors affected
        
        Top 3 Considerations:
        1. Evaluate direct sector exposure to the event
        2. Consider second-order effects across the supply chain
        3. Monitor policy responses that may mitigate or exacerbate impacts
        
        Detailed Explanation:
        Market reactions to this type of event typically follow a pattern where directly affected sectors see immediate price movements, followed by broader market reassessment as implications become clearer. Investors should focus on companies with strong balance sheets that can weather potential disruptions while looking for opportunities where market reactions appear excessive relative to actual business impacts.
        
        For specific investment recommendations, more analysis would be needed on the exact nature and severity of the {event} along with current market valuations and positioning.
        """)
    
    def _generate_sector_recommendation(self, sector):
        """Generate a sector recommendation and analysis"""
        sector_outlooks = {
            'Technology': """
            Technology Sector Outlook: Positive
            
            Stock Recommendation: MSFT (Buy)
            
            Top 3 Reasons:
            1. Cloud computing growth through Azure remains robust with margin expansion
            2. Recurring revenue model provides stability and visibility
            3. AI integration across product suite creates competitive advantages
            
            Detailed Explanation:
            The technology sector continues to benefit from digital transformation trends across industries. Key growth areas include cloud computing, artificial intelligence, semiconductors, and enterprise software. While valuations in parts of the sector remain elevated, companies with strong competitive positions and proven ability to monetize innovation justify premium multiples.
            
            Microsoft (MSFT) stands out within the sector due to its diversified business model spanning cloud infrastructure (Azure), productivity software (Office 365), and enterprise applications. The company's transition to subscription-based revenue has increased business predictability while expanding margins. Recent AI initiatives, particularly through OpenAI partnership, position the company at the forefront of the next computing paradigm.
            
            Other attractive stocks in the sector include:
            - NVDA: Leading position in AI chips with expanding data center footprint
            - ADBE: Dominant creative software with successful cloud transition
            
            Key risks to monitor include regulatory scrutiny, competitive pressures in cloud services, and potential margin compression as growth moderates. Overall, a selective approach focusing on companies with durable competitive advantages and reasonable valuations is recommended.
            """,
            
            'Healthcare': """
            Healthcare Sector Outlook: Positive
            
            Stock Recommendation: UNH (Buy)
            
            Top 3 Reasons:
            1. Integrated healthcare model provides multiple growth avenues
            2. Technology investments enhancing efficiency and member experience
            3. Scale advantages in negotiating power and data analytics
            
            Detailed Explanation:
            The healthcare sector benefits from consistent demand driven by demographic trends and medical advances. Current focus areas include value-based care, technological innovation, and cost containment. The sector offers defensive characteristics with growth potential, particularly as innovation accelerates.
            
            UnitedHealth Group (UNH) represents a compelling opportunity within healthcare due to its unique combination of insurance operations (UnitedHealthcare) and healthcare services (Optum). This integrated approach allows UNH to address multiple aspects of the healthcare value chain while gathering valuable data to improve outcomes and reduce costs. The company's technology investments continue to enhance member experience while expanding margins.
            
            Other attractive stocks in the sector include:
            - LLY: Strong pipeline with significant obesity and diabetes franchise
            - TMO: Essential life sciences tools benefiting from research expansion
            
            Potential risks include policy changes affecting reimbursement, pricing pressure on pharmaceuticals, and elevated valuations for high-growth biotechnology companies. A balanced approach focusing on quality companies with consistent execution and reasonable valuations is recommended.
            """,
            
            'Financials': """
            Financials Sector Outlook: Neutral
            
            Stock Recommendation: JPM (Buy)
            
            Top 3 Reasons:
            1. Industry-leading returns with strong risk management
            2. Technology investments creating operational advantages
            3. Diversified business model balancing traditional banking with investment services
            
            Detailed Explanation:
            The financial sector faces mixed conditions with net interest margin pressure offset by strong capital markets activity and wealth management growth. Banks have generally maintained healthy balance sheets with improved risk management compared to previous cycles. Fintech disruption continues to challenge traditional business models, forcing adaptation.
            
            JPMorgan Chase (JPM) stands out within the sector due to its leadership position across consumer banking, investment banking, and asset management. The bank's technology investments (~$12 billion annually) have strengthened its competitive position while improving efficiency. Strong risk management has historically allowed JPM to navigate market stress better than peers.
            
            Other attractive stocks in the sector include:
            - BLK: Asset management leader benefiting from passive investing trends
            - CME: Exchange operator with strong moat and beneficiary of volatility
            
            Key risks include net interest margin compression in a low-rate environment, potential loan losses during economic weakness, and regulatory constraints on capital return. A selective approach focusing on high-quality institutions with diversified revenue streams is recommended.
            """,
            
            'Energy': """
            Energy Sector Outlook: Neutral
            
            Stock Recommendation: CVX (Buy)
            
            Top 3 Reasons:
            1. Strong balance sheet with industry-leading dividend sustainability
            2. Diversified portfolio with quality assets across upstream and downstream
            3. Capital discipline with focus on returns over production growth
            
            Detailed Explanation:
            The energy sector continues to navigate the transition between traditional fossil fuels and renewable energy sources. Oil prices have stabilized in the mid-range, supporting cash flow for major producers while encouraging capital discipline. Long-term challenges from energy transition remain, but near-term fundamentals have improved.
            
            Chevron (CVX) represents a compelling opportunity within the sector due to its strong balance sheet, operational efficiency, and balanced portfolio approach. The company has demonstrated capital discipline by focusing on high-return projects rather than pursuing production growth at any cost. Its dividend remains well-covered even at lower oil prices.
            
            Other attractive stocks in the sector include:
            - XOM: Scale advantages with improving capital allocation
            - PSX: Refining leader with diversified operations
            
            Key risks include oil price volatility, potential policy changes accelerating energy transition, and ESG-related investment restrictions. A balanced approach focusing on companies with strong balance sheets, capital discipline, and some exposure to renewable initiatives is recommended.
            """,
            
            'Consumer Discretionary': """
            Consumer Discretionary Sector Outlook: Neutral
            
            Stock Recommendation: AMZN (Buy)
            
            Top 3 Reasons:
            1. E-commerce leadership position with expanding market share
            2. AWS cloud business provides high-margin growth driver
            3. Operational improvements enhancing profitability across segments
            
            Detailed Explanation:
            The consumer discretionary sector faces mixed conditions with inflation pressures and economic uncertainty offset by generally healthy consumer balance sheets and employment. Digital transformation continues to reshape retail while creating both winners and losers. Companies with pricing power and strong online capabilities have demonstrated resilience.
            
            Amazon (AMZN) stands out within the sector due to its dominant e-commerce platform combined with its highly profitable cloud computing business (AWS). Recent cost-cutting initiatives have improved margins while the company continues to invest in growth areas including healthcare, advertising, and logistics. The subscription-based Prime ecosystem creates customer loyalty and recurring revenue.
            
            Other attractive stocks in the sector include:
            - NKE: Brand strength with direct-to-consumer growth
            - SBUX: Premiumization strategy with strong digital engagement
            
            Key risks include consumer spending pullback during economic weakness, margin pressure from inflation, and competition for consumer wallet share. A selective approach focusing on companies with strong brands, pricing power, and digital capabilities is recommended.
            """
        }
        
        return sector_outlooks.get(sector, f"""
        {sector} Sector Outlook: Neutral
        
        Stock Recommendation: Analysis Required
        
        Top 3 Considerations:
        1. Evaluate competitive dynamics within the {sector} space
        2. Assess current valuations relative to growth prospects
        3. Consider macroeconomic factors specifically affecting this sector
        
        Detailed Explanation:
        The {sector} sector presents a mixed picture with both opportunities and challenges. Companies with differentiated products or services, operational efficiency, and prudent balance sheet management are likely to outperform peers. Investors should focus on specific business fundamentals rather than broad sector exposure.
        
        For specific stock recommendations within this sector, a detailed analysis of individual companies would be required, examining competitive positioning, management quality, valuation metrics, and growth prospects.
        """)
    
    def _generate_stock_recommendation(self, ticker):
        """Generate a stock recommendation and analysis"""
        # Determine sector for the ticker
        sector = None
        for s, companies in self.companies.items():
            if ticker in companies:
                sector = s
                break
        
        if not sector:
            sector = np.random.choice(self.sectors)
        
        # Randomly select recommendation
        recommendation = np.random.choice(['Buy', 'Hold', 'Sell'], p=[0.5, 0.3, 0.2])
        
        if recommendation == 'Buy':
            return f"""
            Stock Recommendation: {ticker} (Buy)
            
            Top 3 Reasons:
            1. {self._generate_random_positive_point(ticker, sector)}
            2. {self._generate_random_positive_point(ticker, sector)}
            3. {self._generate_random_positive_point(ticker, sector)}
            
            Detailed Explanation:
            After thorough analysis, I recommend a Buy rating for {ticker} based on favorable risk/reward characteristics and multiple growth catalysts. The company operates in the {sector} sector which currently has {np.random.choice(['favorable', 'improving', 'stable'])} dynamics.
            
            Fundamentally, {ticker} demonstrates {np.random.choice(['strong cash flow generation', 'expanding margins', 'accelerating revenue growth'])} which supports our positive outlook. The company's {np.random.choice(['management team has a strong track record', 'product pipeline looks promising', 'market position is strengthening'])}.
            
            Valuation metrics indicate the stock is {np.random.choice(['attractively valued', 'reasonably priced given growth prospects', 'trading at a discount to peers'])}. Specifically, the {np.random.choice(['P/E', 'EV/EBITDA', 'P/S'])} ratio of {np.random.choice(['15x', '8x', '2.5x'])} represents a {np.random.choice(['15%', '20%', '25%'])} discount to the sector average.
            
            Key risks to monitor include {self._generate_random_negative_point(ticker, sector)}, however we believe these concerns are more than offset by the positive factors outlined above.
            
            Based on our analysis, we establish a price target of ${np.random.randint(50, 500)} representing approximately {np.random.randint(15, 40)}% upside from current levels.
            """
        elif recommendation == 'Hold':
            return f"""
            Stock Recommendation: {ticker} (Hold)
            
            Top 3 Reasons:
            1. {self._generate_random_positive_point(ticker, sector)}
            2. {self._generate_random_negative_point(ticker, sector)}
            3. Current valuation appears to fairly reflect both opportunities and risks
            
            Detailed Explanation:
            After careful consideration, I recommend a Hold rating for {ticker} based on a balanced risk/reward profile. The company operates in the {sector} sector which currently has {np.random.choice(['mixed', 'evolving', 'competitive'])} dynamics.
            
            On the positive side, {ticker} demonstrates {np.random.choice(['solid market position', 'stable financial performance', 'reasonable growth prospects'])}. However, these strengths are offset by {np.random.choice(['increasing competition', 'margin pressures', 'slowing industry growth'])}.
            
            Valuation metrics suggest the stock is {np.random.choice(['fairly valued', 'appropriately priced given the mixed outlook', 'trading in line with historical averages'])}. The {np.random.choice(['P/E', 'EV/EBITDA', 'P/S'])} ratio of {np.random.choice(['18x', '10x', '3.5x'])} is {np.random.choice(['in line with', 'slightly above', 'slightly below'])} the sector average.
            
            We would become more positive on the stock if we see {np.random.choice(['acceleration in revenue growth', 'margin expansion', 'successful new product launches'])}, while {np.random.choice(['deterioration in market share', 'weakening balance sheet', 'execution missteps'])} would cause us to downgrade our rating.
            
            For existing shareholders, we recommend maintaining positions but would not add at current levels. For new investors, we suggest waiting for a more attractive entry point or improved risk/reward dynamics.
            """
        else:  # Sell
            return f"""
            Stock Recommendation: {ticker} (Sell)
            
            Top 3 Reasons:
            1. {self._generate_random_negative_point(ticker, sector)}
            2. {self._generate_random_negative_point(ticker, sector)}
            3. {self._generate_random_negative_point(ticker, sector)}
            
            Detailed Explanation:
            After thorough analysis, I recommend a Sell rating for {ticker} based on an unfavorable risk/reward profile and multiple concerning trends. The company operates in the {sector} sector which currently faces {np.random.choice(['significant headwinds', 'structural challenges', 'deteriorating fundamentals'])}.
            
            The primary concerns include {ticker}'s {np.random.choice(['declining market share', 'margin compression', 'weakening competitive position'])} which undermines the long-term investment thesis. The company's {np.random.choice(['management has failed to execute on strategic initiatives', 'product pipeline appears weak', 'financial position is deteriorating'])}.
            
            Valuation metrics indicate the stock is {np.random.choice(['overvalued', 'expensive relative to peers', 'not pricing in significant risks'])}. Specifically, the {np.random.choice(['P/E', 'EV/EBITDA', 'P/S'])} ratio of {np.random.choice(['25x', '12x', '4.5x'])} represents a {np.random.choice(['30%', '40%', '50%'])} premium to the sector average despite inferior growth and profitability metrics.
            
            While {self._generate_random_positive_point(ticker, sector)}, we believe this positive factor is insufficient to offset the significant concerns outlined above.
            
            Based on our analysis, we establish a price target of ${np.random.randint(20, 300)} representing approximately {np.random.randint(15, 35)}% downside from current levels.
            """
    
    def _generate_random_positive_point(self, ticker, sector):
        """Generate a random positive point about a stock"""
        positive_points = [
            f"Strong financial position with {np.random.randint(3, 15)}B in cash and low debt levels",
            f"Expanding margins due to operational efficiencies and scale advantages",
            f"Market share gains in key segments demonstrating competitive strength",
            f"Attractive valuation trading at a discount to historical averages",
            f"Robust product pipeline with potential for accelerating growth",
            f"Consistent execution with management exceeding guidance for {np.random.randint(4, 12)} consecutive quarters",
            f"Significant return of capital to shareholders through dividends and buybacks",
            f"Strategic acquisitions strengthening competitive positioning",
            f"Beneficiary of favorable industry trends including {np.random.choice(['digital transformation', 'automation', 'sustainability initiatives'])}",
            f"Expanding addressable market through new product categories"
        ]
        
        tech_points = [
            "Strong recurring revenue growth from software-as-a-service offerings",
            "Expanding AI capabilities creating competitive differentiation",
            "Cloud services growth exceeding market rates",
            "High customer retention rates demonstrating product stickiness"
        ]
        
        healthcare_points = [
            "Promising late-stage drug pipeline with multiple potential blockbusters",
            "Patent protection providing revenue visibility for key products",
            "Increasing operating margins in healthcare services segment",
            "Strategic shift toward higher-growth specialty medications"
        ]
        
        financial_points = [
            "Net interest margin expansion in rising rate environment",
            "Strong capital position exceeding regulatory requirements",
            "Growth in fee-based businesses reducing reliance on interest income",
            "Technology investments improving efficiency ratios"
        ]
        
        energy_points = [
            "Premium assets in low-cost production regions",
            "Increasing free cash flow generation at current commodity prices",
            "Disciplined capital allocation focused on returns over production growth",
            "Growing renewable energy investments positioning for energy transition"
        ]
        
        consumer_points = [
            "Strong brand equity supporting pricing power",
            "Successful e-commerce strategy with digital sales growth",
            "International expansion providing new growth avenues",
            "Innovative product launches resonating with target demographics"
        ]
        
        # Add sector-specific points
        if sector == 'Technology':
            positive_points.extend(tech_points)
        elif sector == 'Healthcare':
            positive_points.extend(healthcare_points)
        elif sector == 'Financials':
            positive_points.extend(financial_points)
        elif sector == 'Energy':
            positive_points.extend(energy_points)
        elif sector in ['Consumer Discretionary', 'Consumer Staples']:
            positive_points.extend(consumer_points)
            
        return np.random.choice(positive_points)
    
    def _generate_random_negative_point(self, ticker, sector):
        """Generate a random negative point about a stock"""
        negative_points = [
            f"Increasing competitive pressures threatening market position",
            f"Margin pressure due to rising input costs and pricing challenges",
            f"Slowing growth in core segments raising concerns about long-term trajectory",
            f"Elevated valuation leaves little room for execution missteps",
            f"Management turnover creating strategic uncertainty",
            f"High expectations embedded in current share price",
            f"Balance sheet concerns with {np.random.randint(5, 30)}B in debt and limited free cash flow",
            f"Regulatory risks including potential {np.random.choice(['antitrust action', 'pricing controls', 'environmental regulations'])}",
            f"Cyclical headwinds likely to impact near-term performance",
            f"Declining return on invested capital suggesting deteriorating business economics"
        ]
        
        tech_points = [
            "Intensifying competition in cloud services pressuring growth and margins",
            "Product cycle delays impacting revenue visibility",
            "Significant R&D investments with uncertain returns",
            "Increasing regulatory scrutiny around data privacy and market power"
        ]
        
        healthcare_points = [
            "Patent cliffs for key products without adequate replacement pipeline",
            "Pricing pressure from payers threatening margins",
            "Regulatory risks related to drug approval process",
            "Rising R&D costs with declining productivity"
        ]
        
        financial_points = [
            "Net interest margin compression in current rate environment",
            "Deteriorating loan quality metrics suggesting credit concerns",
            "Regulatory capital requirements limiting shareholder returns",
            "Fintech disruption threatening traditional business models"
        ]
        
        energy_points = [
            "High production costs relative to industry average",
            "Significant exposure to volatile commodity prices",
            "Environmental liabilities and transition risks",
            "Declining reserve replacement ratio raising long-term production concerns"
        ]
        
        consumer_points = [
            "Shifting consumer preferences away from core products",
            "Increasing private label competition pressuring market share",
            "Challenges in digital transformation relative to peers",
            "Promotional environment limiting pricing power"
        ]
        
        # Add sector-specific points
        if sector == 'Technology':
            negative_points.extend(tech_points)
        elif sector == 'Healthcare':
            negative_points.extend(healthcare_points)
        elif sector == 'Financials':
            negative_points.extend(financial_points)
        elif sector == 'Energy':
            negative_points.extend(energy_points)
        elif sector in ['Consumer Discretionary', 'Consumer Staples']:
            negative_points.extend(consumer_points)
            
        return np.random.choice(negative_points)
    
    def _generate_market_condition_answer(self, condition):
        """Generate an answer about a market condition's impact"""
        conditions = {
            'Inflation': """
            Inflation impacts stock markets and investment strategies in multiple ways:
            
            Stock Recommendation: LIN (Buy)
            
            Top 3 Reasons:
            1. Pricing power allows passing through cost increases to customers
            2. Essential industrial gases with inelastic demand across sectors
            3. Long-term contracts often include inflation adjustment clauses
            
            Detailed Explanation:
            Inflation typically affects different sectors and asset classes in varying ways:
            
            Sectors that typically perform well during inflation:
            - Materials and commodities: Companies like Linde (LIN) with pricing power can pass through higher costs
            - Energy: These companies often benefit as energy prices are themselves inflationary
            - Real Estate: Physical assets tend to appreciate with inflation
            - Select Consumer Staples: Companies with strong brands can increase prices
            
            Sectors typically challenged by inflation:
            - Growth technology: Higher discount rates reduce the present value of future earnings
            - Utilities: Regulated returns may lag inflation
            - Consumer discretionary companies without pricing power: Margin compression
            
            For investment strategies, consider:
            1. Emphasizing companies with pricing power and low labor intensity
            2. Focusing on quality companies with strong balance sheets
            3. Including some commodity exposure as an inflation hedge
            4. Considering TIPS (Treasury Inflation-Protected Securities) for fixed income allocation
            5. Being cautious with long-duration assets (growth stocks, long-term bonds)
            
            The impact of inflation depends significantly on both its level and rate of change. Moderate inflation (2-3%) is generally manageable for most businesses, while high or accelerating inflation creates more significant challenges for corporate profits and economic stability.
            """,
            
            'Interest Rates': """
            Interest rates impact stock markets and investment strategies in several important ways:
            
            Stock Recommendation: JPM (Buy)
            
            Top 3 Reasons:
            1. Net interest margin expansion during rising rate environments
            2. Strong deposit franchise provides low-cost funding
            3. Diversified business model balances interest rate sensitivity
            
            Detailed Explanation:
            Interest rates are fundamental to asset valuation and significantly impact investment strategies:
            
            Sectors that typically benefit from rising rates:
            - Financials: Banks like JPMorgan Chase (JPM) often benefit from wider lending margins
            - Insurance: Higher rates increase investment income on float
            - Value stocks: These tend to outperform growth in rising rate environments
            - Shorter-duration equities: Companies with current cash flows rather than distant earnings
            
            Sectors typically challenged by rising rates:
            - Utilities: Higher discount rates reduce the appeal of their stable dividends
            - Real Estate: Higher mortgage rates can reduce demand and increase cap rates
            - Growth Technology: Higher discount rates for future earnings
            - Consumer Durables: Higher financing costs reduce purchases
            
            For investment strategies, consider:
            1. Adjusting sector allocations based on the rate environment
            2. Shortening duration in fixed income portfolios during rising rates
            3. Focusing on companies with low debt levels or fixed-rate debt
            4. Understanding the second-order effects beyond simple sector categorizations
            5. Monitoring the yield curve shape, not just the absolute level of rates
            
            The relationship between interest rates and stock performance is complex and depends on the reason rates are changing (growth vs. inflation concerns), the speed of change, and starting levels. Gradual rate changes are typically easier for markets to absorb than sudden shifts.
            """,
            
            'GDP Growth': """
            GDP growth significantly impacts stock markets and investment strategies:
            
            Stock Recommendation: CAT (Buy)
            
            Top 3 Reasons:
            1. Highly correlated business performance with economic cycles
            2. Current valuation provides upside in expanding economy
            3. Global presence captures worldwide growth opportunities
            
            Detailed Explanation:
            GDP growth is a fundamental driver of corporate earnings and market performance:
            
            Sectors that typically benefit from strong GDP growth:
            - Industrials: Companies like Caterpillar (CAT) directly benefit from economic expansion
            - Consumer Discretionary: Consumers spend more on non-essentials during growth periods
            - Financials: Loan growth accelerates with economic activity
            - Materials: Demand increases with production and construction
            
            Sectors less sensitive to GDP growth:
            - Consumer Staples: Necessary purchases continue regardless of economic conditions
            - Utilities: Essential services with regulated returns
            - Healthcare: Medical needs persist through economic cycles
            
            For investment strategies, consider:
            1. Increasing cyclical exposure during periods of accelerating growth
            2. Focusing on companies with operating leverage that benefit from volume increases
            3. Being selective with defensive sectors which may underperform
            4. Monitoring leading economic indicators for potential inflection points
            5. Balancing growth exposure with quality factors
            
            It's important to distinguish between absolute GDP growth levels and rate of change. Markets often react more to changes in growth expectations than to absolute levels, making it essential to consider both consensus expectations and your own economic outlook when positioning portfolios.
            """,
            
            'Unemployment': """
            Unemployment levels impact stock markets and investment strategies in several important ways:
            
            Stock Recommendation: HD (Buy)
            
            Top 3 Reasons:
            1. Consumer spending resilience in strong labor markets
            2. Home improvement benefits from housing market strength
            3. Operational excellence with strong execution regardless of conditions
            
            Detailed Explanation:
            Unemployment is a key economic indicator that influences consumer spending, corporate profits, and investor sentiment:
            
            Sectors that typically benefit from low unemployment:
            - Consumer Discretionary: Companies like Home Depot (HD) benefit from higher consumer spending
            - Financials: Lower loan defaults and increased financial services activity
            - Real Estate: Stronger housing demand and rental markets
            - Retail: Increased consumer confidence and spending
            
            Sectors less sensitive to unemployment levels:
            - Consumer Staples: Necessary purchases continue regardless of employment conditions
            - Utilities: Essential services with consistent demand
            - Healthcare: Medical needs persist regardless of economic conditions
            
            For investment strategies, consider:
            1. Increasing exposure to consumer sectors during periods of improving employment
            2. Monitoring wage growth alongside unemployment for inflation implications
            3. Focusing on companies with pricing power if wage inflation accelerates
            4. Being cautious about highly leveraged consumer businesses late in employment cycles
            5. Watching for turning points in unemployment as potential market inflection signals
            
            The relationship between unemployment and markets is not always straightforward. Very low unemployment can sometimes signal late-cycle dynamics and potential wage inflation, while rapidly rising unemployment typically indicates recession conditions. The direction and rate of change often matter more than absolute levels.
            """,
            
            'Consumer Sentiment': """
            Consumer sentiment significantly impacts stock markets and investment strategies:
            
            Stock Recommendation: NKE (Buy)
            
            Top 3 Reasons:
            1. Strong brand positioning captures consumer spending when sentiment improves
            2. Premium products benefit from consumer confidence
            3. Global presence diversifies exposure to regional sentiment fluctuations
            
            Detailed Explanation:
            Consumer sentiment influences spending patterns and can be a leading indicator for economic activity:
            
            Sectors that typically benefit from strong consumer sentiment:
            - Consumer Discretionary: Companies like Nike (NKE) selling premium products
            - Retail: Both brick-and-mortar and e-commerce benefit from increased spending
            - Travel and Leisure: Consumers spend more on experiences when confident
            - Automotive: Major purchases increase with confidence
            
            Sectors less sensitive to consumer sentiment:
            - Consumer Staples: Necessary purchases continue regardless of sentiment
            - Utilities: Essential services with consistent demand
            - Healthcare: Medical needs persist through sentiment cycles
            
            For investment strategies, consider:
            1. Monitoring consumer sentiment indicators for early signs of change
            2. Distinguishing between sentiment and actual spending behavior
            3. Understanding the gap between sentiment measures and personal consumption expenditures
            4. Considering the wealth effect from housing and stock markets on sentiment
            5. Analyzing sentiment across different demographic and income groups
            
            Consumer sentiment can be volatile and sometimes disconnected from actual spending behavior. It's important to validate sentiment indicators against hard economic data like retail sales, credit card spending, and housing activity. Additionally, different consumer segments (luxury vs. value) may behave differently based on economic conditions.
            """
        }
        
        return conditions.get(condition, f"""
        {condition} impacts stock markets and investment strategies in several important ways:
        
        Stock Recommendation: Dependent on specific aspects of {condition}
        
        Top 3 Considerations:
        1. Evaluate how {condition} affects corporate profitability across sectors
        2. Consider impact on consumer behavior and spending patterns
        3. Assess potential monetary and fiscal policy responses
        
        Detailed Explanation:
        {condition} influences markets through multiple channels including corporate earnings, investor sentiment, and policy responses. The specific impact varies significantly across sectors and individual companies depending on their exposure and adaptability.
        
        For investment strategy, it's important to:
        1. Identify which sectors benefit or face headwinds from changes in {condition}
        2. Focus on companies with management teams that have successfully navigated similar conditions
        3. Consider both first-order effects and secondary consequences of {condition}
        4. Maintain diversification to manage uncertainty
        
        For specific investment recommendations, a detailed analysis of current {condition} trends, market pricing, and company-specific factors would be required.
        """)
    
    def _generate_stock_comparison(self, ticker1, ticker2):
        """Generate a comparison between two stocks"""
        # Determine sectors
        sector1 = None
        sector2 = None
        
        for s, companies in self.companies.items():
            if ticker1 in companies:
                sector1 = s
            if ticker2 in companies:
                sector2 = s
                
            if sector1 and sector2:
                break
        
        if not sector1:
            sector1 = np.random.choice(self.sectors)
        if not sector2:
            sector2 = np.random.choice(self.sectors)
        
        # Randomly select which is better
        better_ticker = np.random.choice([ticker1, ticker2])
        worse_ticker = ticker2 if better_ticker == ticker1 else ticker1
        better_sector = sector1 if better_ticker == ticker1 else sector2
        
        return f"""
        Stock Recommendation: {better_ticker} (Buy)
        
        Top 3 Reasons:
        1. {self._generate_random_positive_point(better_ticker, better_sector)}
        2. {self._generate_random_positive_point(better_ticker, better_sector)}
        3. More favorable risk/reward profile compared to {worse_ticker}
        
        Detailed Explanation:
        After conducting a comparative analysis between {ticker1} and {ticker2}, I recommend {better_ticker} as the superior investment opportunity at current valuations.
        
        {better_ticker} demonstrates stronger {np.random.choice(['revenue growth', 'margin expansion', 'return on invested capital', 'free cash flow generation'])} with {np.random.choice(['15%', '20%', '25%'])} {np.random.choice(['year-over-year growth', 'improvement', 'higher returns'])} compared to {np.random.choice(['5%', '10%', '15%'])} for {worse_ticker}.
        
        From a valuation perspective, {better_ticker} trades at a {np.random.choice(['lower', 'more attractive', 'more reasonable'])} multiple of {np.random.choice(['15x forward earnings', '8x EBITDA', '2.5x sales'])} compared to {worse_ticker}'s {np.random.choice(['18x forward earnings', '10x EBITDA', '3.5x sales'])}, despite its superior {np.random.choice(['growth profile', 'profitability metrics', 'market position'])}.
        
        While {worse_ticker} does have some advantages, including {self._generate_random_positive_point(worse_ticker, sector2 if worse_ticker == ticker2 else sector1)}, these are outweighed by {np.random.choice(['execution challenges', 'competitive pressures', 'valuation concerns'])}.
        
        Both companies face industry challenges including {np.random.choice(['increasing competition', 'regulatory scrutiny', 'shifting consumer preferences'])}, but {better_ticker} appears better positioned to navigate these headwinds due to its {np.random.choice(['stronger market position', 'more diversified business model', 'superior management execution'])}.
        
        For investors choosing between these two stocks, {better_ticker} offers a more compelling opportunity at current prices, though a diversified portfolio might include both with a higher allocation to {better_ticker}.
        """
    
    def _generate_portfolio_allocation(self, risk_profile, time_horizon):
        """Generate portfolio allocation recommendation based on risk profile and time horizon"""
        # Define allocation ranges based on risk profile and time horizon
        allocations = {
            'conservative': {
                'short-term': {'stocks': '20-30%', 'bonds': '50-60%', 'cash': '15-25%', 'alternatives': '0-5%'},
                'medium-term': {'stocks': '30-40%', 'bonds': '45-55%', 'cash': '10-15%', 'alternatives': '0-10%'},
                'long-term': {'stocks': '40-50%', 'bonds': '40-50%', 'cash': '5-10%', 'alternatives': '0-10%'}
            },
            'moderate': {
                'short-term': {'stocks': '40-50%', 'bonds': '35-45%', 'cash': '10-20%', 'alternatives': '0-10%'},
                'medium-term': {'stocks': '50-60%', 'bonds': '30-40%', 'cash': '5-10%', 'alternatives': '5-15%'},
                'long-term': {'stocks': '60-70%', 'bonds': '20-30%', 'cash': '0-10%', 'alternatives': '5-15%'}
            },
            'aggressive': {
                'short-term': {'stocks': '60-70%', 'bonds': '15-25%', 'cash': '5-15%', 'alternatives': '5-15%'},
                'medium-term': {'stocks': '70-80%', 'bonds': '10-20%', 'cash': '0-10%', 'alternatives': '5-15%'},
                'long-term': {'stocks': '80-90%', 'bonds': '0-15%', 'cash': '0-5%', 'alternatives': '5-20%'}
            }
        }
        
        selected_allocation = allocations[risk_profile][time_horizon]
        
        # Define stock sector tilts based on risk profile
        sector_tilts = {
            'conservative': "Overweight defensive sectors including Consumer Staples, Utilities, and Healthcare. Within equities, emphasize quality companies with strong balance sheets, stable earnings, and dividend history.",
            'moderate': "Balanced exposure across sectors with slight overweight to Technology, Healthcare, and Financials. Blend of growth and value styles with emphasis on quality factor.",
            'aggressive': "Overweight cyclical and growth sectors including Technology, Consumer Discretionary, and small-cap stocks. Consider emerging markets exposure for long-term growth potential."
        }
        
        # Define bond allocations based on time horizon
        bond_allocations = {
            'short-term': "Focus on short-duration bonds (1-3 years) to minimize interest rate risk. Include Treasury securities, high-quality corporate bonds, and municipal bonds if tax-advantaged income is beneficial.",
            'medium-term': "Intermediate-duration bonds (3-7 years) providing balance between yield and interest rate sensitivity. Consider core bond funds supplemented with investment-grade corporate bonds.",
            'long-term': "Broader fixed income allocation including longer-duration bonds (5-10 years) for higher yield potential. Consider international bonds for diversification and small allocations to high-yield bonds if appropriate for risk tolerance."
        }
        
        # Alternative investments suggestions
        alternative_suggestions = {
            'conservative': "Limited allocation to alternatives, focusing on lower-volatility strategies like hedged equity or multi-strategy funds.",
            'moderate': "Consider REITs, infrastructure funds, and commodities for diversification benefits.",
            'aggressive': "Explore private equity, venture capital, commodities, and other alternative strategies for enhanced returns and portfolio diversification."
        }
        
        return f"""
        For a {risk_profile} risk tolerance and {time_horizon} investment horizon, I recommend the following portfolio allocation:
        
        Stock Recommendation: Diversified portfolio with specific allocations below
        
        Top 3 Principles:
        1. Asset allocation is the primary driver of long-term returns and risk
        2. Diversification across and within asset classes reduces portfolio volatility
        3. Regular rebalancing maintains risk profile and can enhance returns
        
        Detailed Explanation:
        Based on your {risk_profile} risk profile and {time_horizon} time horizon, an appropriate portfolio allocation would be:
        
        • Stocks: {selected_allocation['stocks']}
        • Bonds: {selected_allocation['bonds']}
        • Cash: {selected_allocation['cash']}
        • Alternative Investments: {selected_allocation['alternatives']}
        
        Equity Allocation Details:
        {sector_tilts[risk_profile]}
        
        Consider a global allocation with approximately 70% domestic equities and 30% international equities to provide geographical diversification. Within international, allocate roughly 20% to developed markets and 10% to emerging markets.
        
        Fixed Income Allocation Details:
        {bond_allocations[time_horizon]}
        
        Alternative Investments Considerations:
        {alternative_suggestions[risk_profile]}
        
        Implementation Strategy:
        • Utilize low-cost index funds or ETFs as the core of your portfolio
        • Consider dollar-cost averaging when deploying new capital
        • Rebalance at least annually to maintain target allocations
        • Regularly review and adjust as your time horizon shortens or financial situation changes
        
        This allocation balances growth potential with risk management appropriate for your stated profile. As your circumstances change, the allocation should be revisited and adjusted accordingly.
        """
    

# Create data simulator
simulator = InvestmentDataSimulator()

# Generate research reports
research_reports = simulator.generate_research_reports(num_reports=100)
print(f"Generated {len(research_reports)} research reports")

# Generate market news
market_news = simulator.generate_market_news(num_news=200)
print(f"Generated {len(market_news)} market news items")

# Generate Q&A pairs for instruction fine-tuning
qa_pairs = simulator.generate_qa_pairs(num_pairs=50)
print(f"Generated {len(qa_pairs)} Q&A pairs")

# Extract unique tickers from reports and news
all_tickers = set()
for companies in research_reports['companies']:
    all_tickers.update(companies)
for companies in market_news['affected_companies']:
    all_tickers.update(companies)

all_tickers = list(all_tickers)
print(f"Collected {len(all_tickers)} unique tickers")

# Generate stock price data
start_date = '2020-01-01'
end_date = '2022-12-31'
stock_data = simulator.generate_stock_data(all_tickers, start_date, end_date)
print(f"Generated stock price data from {start_date} to {end_date} for {len(all_tickers)} tickers")

# Display examples of each dataset
print("\nExample Research Report:")
print(research_reports.iloc[0]['content'])

print("\nExample Market News:")
print(market_news.iloc[0]['content'])

print("\nExample Q&A Pair:")
print(f"Question: {qa_pairs.iloc[0]['question']}")
print(f"Answer: {qa_pairs.iloc[0]['answer']}")

print("\nStock Data Shape:", stock_data.shape)


class ModelFineTuner:
    """Class to implement fine-tuning methods described in the paper"""
    
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """Initialize the fine-tuner"""
        self.device = device
        print(f"Using device: {self.device}")
    
    def unsupervised_fine_tuning(self, reports_df, tokenizer_name="meta-llama/Llama-2-7b-chat-hf", 
                                 model_name="meta-llama/Llama-2-7b-chat-hf"):
        """
        Simulate unsupervised fine-tuning of Llama2 with LoRA
        
        Parameters:
        - reports_df: DataFrame containing research reports
        - tokenizer_name: Name of the tokenizer to use
        - model_name: Name of the model to use
        
        Returns:
        - Fine-tuned model description
        """
        print("Simulating unsupervised fine-tuning...")
        
        # In a real implementation, we would:
        # 1. Load the model and tokenizer
        # 2. Apply LoRA configuration
        # 3. Prepare the dataset
        # 4. Set up training arguments
        # 5. Train the model
        
        # For simulation purposes, we'll just show the process
        print(f"1. Would load model and tokenizer: {model_name}")
        print(f"2. Would apply LoRA with rank=4, modifying query and value matrices")
        
        # Prepare corpus text
        corpus_text = "\n\n".join(reports_df['content'].tolist())
        print(f"3. Prepared corpus with {len(corpus_text)} characters")
        
        # Tokenize (simulate)
        print("4. Would tokenize and chunk the corpus into segments of 512 tokens")
        
        # Training settings
        print("5. Would train with settings:")
        print("   - Learning rate: 2e-4")
        print("   - Batch size: 4")
        print("   - Gradient accumulation steps: 4")
        print("   - Weight decay: 0.01")
        print("   - Epochs: 3")
        
        return {
            'model_type': 'Llama2-7B with LoRA',
            'fine_tuning_type': 'Unsupervised',
            'training_data': f"{len(reports_df)} research reports",
            'training_tokens': f"~{len(corpus_text) // 4} tokens",  # rough estimate
            'description': "Fine-tuned on investment research reports for next-token prediction"
        }
    
    def supervised_fine_tuning(self, news_df, stock_data, tokenizer_name="meta-llama/Llama-2-7b-chat-hf", 
                               model_name="meta-llama/Llama-2-7b-chat-hf"):
        """
        Simulate supervised fine-tuning of Llama2 with LoRA using news headlines and stock returns
        
        Parameters:
        - news_df: DataFrame containing news headlines
        - stock_data: DataFrame containing stock price data
        - tokenizer_name: Name of the tokenizer to use
        - model_name: Name of the model to use
        
        Returns:
        - Fine-tuned model description
        """
        print("Simulating supervised fine-tuning...")
        
        # In a real implementation, we would:
        # 1. Load the model and tokenizer
        # 2. Apply LoRA configuration
        # 3. Prepare the dataset with labels
        # 4. Set up training arguments
        # 5. Train the model
        
        # For simulation purposes, we'll just show the process
        print(f"1. Would load model and tokenizer: {model_name}")
        print(f"2. Would apply LoRA with rank=4, modifying query and value matrices")
        
        # Prepare dataset with labels
        print("3. Would prepare dataset with news headlines and corresponding stock returns")
        
        # For simulation, let's calculate some statistics on our synthetic data
        headlines = news_df['headline'].tolist()
        
        # Calculate average headline length
        avg_headline_length = sum(len(h) for h in headlines) / len(headlines)
        print(f"   - Average headline length: {avg_headline_length:.1f} characters")
        
        # Count number of unique tickers
        affected_companies = news_df['affected_companies'].explode().unique()
        print(f"   - Number of unique tickers: {len(affected_companies)}")
        
        # Training settings
        print("4. Would train with settings:")
        print("   - Learning rate: 2e-4")
        print("   - Batch size: 8")
        print("   - Gradient accumulation steps: 2")
        print("   - Weight decay: 0.01")
        print("   - Epochs: 5")
        
        return {
            'model_type': 'Llama2-7B with LoRA',
            'fine_tuning_type': 'Supervised',
            'training_data': f"{len(news_df)} news headlines with stock returns",
            'unique_tickers': len(affected_companies),
            'description': "Fine-tuned to predict stock price movements from news headlines"
        }
    
    def instruction_fine_tuning(self, qa_pairs_df):
        """
        Simulate instruction fine-tuning of GPT-3.5-turbo
        
        Parameters:
        - qa_pairs_df: DataFrame containing Q&A pairs for instruction fine-tuning
        
        Returns:
        - Fine-tuned model description
        """
        print("Simulating instruction fine-tuning of GPT-3.5-turbo...")
        
        # In a real implementation, we would:
        # 1. Format the dataset for OpenAI fine-tuning
        # 2. Upload the dataset
        # 3. Create a fine-tuning job
        # 4. Monitor the job
        # 5. Use the fine-tuned model
        
        # For simulation purposes, we'll just show the process
        
        # Format the dataset
        print("1. Would format dataset for OpenAI fine-tuning")
        
        # Calculate statistics
        question_lengths = [len(q) for q in qa_pairs_df['question']]
        answer_lengths = [len(a) for a in qa_pairs_df['answer']]
        
        avg_question_length = sum(question_lengths) / len(question_lengths)
        avg_answer_length = sum(answer_lengths) / len(answer_lengths)
        
        print(f"   - Number of Q&A pairs: {len(qa_pairs_df)}")
        print(f"   - Average question length: {avg_question_length:.1f} characters")
        print(f"   - Average answer length: {avg_answer_length:.1f} characters")
        
        # Training settings
        print("2. Would fine-tune with settings:")
        print("   - Model: gpt-3.5-turbo")
        print("   - Number of epochs: 3")
        print("   - Learning rate multiplier: 0.1")
        print("   - Batch size: 4")
        
        return {
            'model_type': 'GPT-3.5-turbo',
            'fine_tuning_type': 'Instruction',
            'training_data': f"{len(qa_pairs_df)} Q&A pairs",
            'description': "Instruction fine-tuned for investment research and recommendations"
        }
    

class InvestmentStrategy:
    """Class to test investment strategies"""
    
    def __init__(self, stock_data, start_date=None, end_date=None):
        """Initialize the strategy with stock data"""
        self.stock_data = stock_data
        
        if start_date:
            self.stock_data = self.stock_data.loc[start_date:]
        if end_date:
            self.stock_data = self.stock_data.loc[:end_date]
            
        self.dates = self.stock_data.index.tolist()
    
    def generate_recommendations(self, model_type, news_df=None, event=None, 
                                risk_profile="moderate", investment_horizon="long-term"):
        """
        Generate investment recommendations based on model type
        
        Parameters:
        - model_type: Type of model to use ('baseline', 'unsupervised', 'supervised', 'instruction')
        - news_df: DataFrame containing news (for supervised model)
        - event: Specific event to analyze (for all models)
        - risk_profile: Investor risk profile (for instruction model)
        - investment_horizon: Investment time horizon (for instruction model)
        
        Returns:
        - recommendations: List of recommendations with tickers and reasons
        """
        # For simulation purposes, we'll generate recommendations based on model type
        recommendations = []
        
        if event is None:
            event = "Interest Rate Hike"  # Default event
        
        if model_type == 'baseline':
            # Baseline model (untrained Llama2) gives generic recommendations
            recommendations = [
                {'ticker': 'AAPL', 'action': 'Buy', 'confidence': 0.6, 
                 'reasons': ['Strong market position', 'Consistent growth', 'Product innovation']},
                {'ticker': 'MSFT', 'action': 'Buy', 'confidence': 0.7,
                 'reasons': ['Cloud leadership', 'Diversified revenue', 'Strong balance sheet']},
                {'ticker': 'AMZN', 'action': 'Hold', 'confidence': 0.5,
                 'reasons': ['E-commerce dominance', 'AWS growth', 'Valuation concerns']}
            ]
        
        elif model_type == 'unsupervised':
            # Unsupervised fine-tuned model provides more financially-informed recommendations
            if event == "Interest Rate Hike":
                recommendations = [
                    {'ticker': 'JPM', 'action': 'Buy', 'confidence': 0.8, 
                     'reasons': ['Net interest margin expansion', 'Strong deposit base', 'Well-capitalized']},
                    {'ticker': 'GS', 'action': 'Buy', 'confidence': 0.7,
                     'reasons': ['Trading revenue boost', 'Asset management growth', 'Valuation attractive']},
                    {'ticker': 'BLK', 'action': 'Buy', 'confidence': 0.75,
                     'reasons': ['Asset management scale', 'ETF leadership', 'Margin expansion']}
                ]
            elif event == "Oil Price Increase":
                recommendations = [
                    {'ticker': 'XOM', 'action': 'Buy', 'confidence': 0.8, 
                     'reasons': ['Upstream revenue growth', 'Integrated model', 'Dividend stability']},
                    {'ticker': 'CVX', 'action': 'Buy', 'confidence': 0.75,
                     'reasons': ['Production growth', 'Strong balance sheet', 'Capital discipline']},
                    {'ticker': 'PSX', 'action': 'Hold', 'confidence': 0.6,
                     'reasons': ['Refining margins', 'Midstream exposure', 'Potential demand concerns']}
                ]
            else:
                # Generic recommendations for other events
                recommendations = [
                    {'ticker': 'AAPL', 'action': 'Buy', 'confidence': 0.7, 
                     'reasons': ['Market leadership', 'Services growth', 'Share repurchases']},
                    {'ticker': 'JNJ', 'action': 'Buy', 'confidence': 0.65,
                     'reasons': ['Defensive characteristics', 'Healthcare innovation', 'Dividend growth']},
                    {'ticker': 'PG', 'action': 'Hold', 'confidence': 0.6,
                     'reasons': ['Consumer staples leader', 'Brand strength', 'Margin pressures']}
                ]
        
        elif model_type == 'supervised':
            # Supervised model uses correlations between news and stock movements
            if news_df is not None:
                # Find news related to the event
                event_news = news_df[news_df['event'] == event]
                
                if len(event_news) > 0:
                    # Get affected companies from news
                    affected_companies = []
                    for companies in event_news['affected_companies']:
                        affected_companies.extend(companies)
                    
                    # Count occurrences to find most mentioned companies
                    company_counts = {}
                    for company in affected_companies:
                        if company in company_counts:
                            company_counts[company] += 1
                        else:
                            company_counts[company] = 1
                    
                    # Sort by count
                    sorted_companies = sorted(company_counts.items(), key=lambda x: x[1], reverse=True)
                    
                    # Generate recommendations for top companies
                    for company, count in sorted_companies[:3]:
                        # Determine action based on event and company sector
                        sector = None
                        for s, companies in simulator.companies.items():
                            if company in companies:
                                sector = s
                                break
                        
                        if not sector:
                            sector = "Unknown"
                        
                        # Check if event is positive or negative for sector
                        event_impact = simulator.event_sector_impact.get(event, {})
                        impact = event_impact.get(sector, 0)
                        
                        action = "Buy" if impact > 0 else "Sell" if impact < 0 else "Hold"
                        confidence = min(0.5 + (count / 10) + abs(impact) * 0.1, 0.9)
                        
                        reasons = []
                        if action == "Buy":
                            reasons = [
                                f"Positive impact from {event}",
                                f"Strong position in {sector} sector",
                                "Technical indicators suggest upward momentum"
                            ]
                        elif action == "Sell":
                            reasons = [
                                f"Negative impact from {event}",
                                f"Challenges in {sector} sector",
                                "Technical indicators suggest downward momentum"
                            ]
                        else:
                            reasons = [
                                f"Mixed impact from {event}",
                                f"Neutral position in {sector} sector",
                                "Technical indicators suggest sideways momentum"
                            ]
                        
                        recommendations.append({
                            'ticker': company,
                            'action': action,
                            'confidence': confidence,
                            'reasons': reasons
                        })
                
                # If no specific news found, provide generic recommendations
                if len(recommendations) == 0:
                    recommendations = [
                        {'ticker': 'SPY', 'action': 'Hold', 'confidence': 0.5, 
                         'reasons': ['Market uncertainty', 'Awaiting clarity', 'Defensive positioning']},
                        {'ticker': 'QQQ', 'action': 'Hold', 'confidence': 0.5,
                         'reasons': ['Tech sector volatility', 'Mixed signals', 'Monitoring developments']},
                        {'ticker': 'VTV', 'action': 'Hold', 'confidence': 0.5,
                         'reasons': ['Value stock resilience', 'Dividend focus', 'Lower volatility']}
                    ]
            else:
                # No news provided, give generic recommendations
                recommendations = [
                    {'ticker': 'SPY', 'action': 'Hold', 'confidence': 0.5, 
                     'reasons': ['Market uncertainty', 'Awaiting clarity', 'Defensive positioning']},
                    {'ticker': 'QQQ', 'action': 'Hold', 'confidence': 0.5,
                     'reasons': ['Tech sector volatility', 'Mixed signals', 'Monitoring developments']},
                    {'ticker': 'VTV', 'action': 'Hold', 'confidence': 0.5,
                     'reasons': ['Value stock resilience', 'Dividend focus', 'Lower volatility']}
                ]
        
        elif model_type == 'instruction':
            # Instruction fine-tuned model provides recommendations aligned with investment preferences
            
            # Adjust recommendations based on risk profile and investment horizon
            if risk_profile == 'conservative':
                if event == "Interest Rate Hike":
                    recommendations = [
                        {'ticker': 'BRK.B', 'action': 'Buy', 'confidence': 0.8, 
                         'reasons': ['Diversified business model', 'Strong balance sheet', 'Value-oriented approach']},
                        {'ticker': 'JNJ', 'action': 'Buy', 'confidence': 0.75,
                         'reasons': ['Defensive healthcare exposure', 'Dividend aristocrat', 'Product diversification']},
                        {'ticker': 'PG', 'action': 'Buy', 'confidence': 0.7,
                         'reasons': ['Consumer staples leader', 'Pricing power', 'Dividend history']}
                    ]
                elif event == "Oil Price Increase":
                    recommendations = [
                        {'ticker': 'CVX', 'action': 'Buy', 'confidence': 0.75, 
                         'reasons': ['Integrated model', 'Dividend stability', 'Strong balance sheet']},
                        {'ticker': 'XLE', 'action': 'Buy', 'confidence': 0.7,
                         'reasons': ['Sector diversification', 'Energy exposure', 'Income potential']},
                        {'ticker': 'VDE', 'action': 'Hold', 'confidence': 0.6,
                         'reasons': ['Energy sector ETF', 'Broader exposure', 'Lower volatility than individual stocks']}
                    ]
            
            elif risk_profile == 'aggressive':
                if event == "Interest Rate Hike":
                    recommendations = [
                        {'ticker': 'GS', 'action': 'Buy', 'confidence': 0.85, 
                         'reasons': ['Trading revenue boost', 'Investment banking fees', 'Capital markets activity']},
                        {'ticker': 'MS', 'action': 'Buy', 'confidence': 0.8,
                         'reasons': ['Wealth management growth', 'Trading operations', 'Strategic acquisitions']},
                        {'ticker': 'FITB', 'action': 'Buy', 'confidence': 0.75,
                         'reasons': ['Regional bank exposure', 'Net interest margin expansion', 'Loan growth potential']}
                    ]
                elif event == "Oil Price Increase":
                    recommendations = [
                        {'ticker': 'DVN', 'action': 'Buy', 'confidence': 0.85, 
                         'reasons': ['Direct oil exposure', 'Production growth', 'Variable dividend potential']},
                        {'ticker': 'HAL', 'action': 'Buy', 'confidence': 0.8,
                         'reasons': ['Oil services recovery', 'International exposure', 'Operational leverage']},
                        {'ticker': 'MRO', 'action': 'Buy', 'confidence': 0.75,
                         'reasons': ['Upstream pure play', 'Operating leverage', 'Share repurchases']}
                    ]
            
            else:  # moderate
                if event == "Interest Rate Hike":
                    recommendations = [
                        {'ticker': 'JPM', 'action': 'Buy', 'confidence': 0.8, 
                         'reasons': ['Net interest margin expansion', 'Diversified business', 'Market leadership']},
                        {'ticker': 'BAC', 'action': 'Buy', 'confidence': 0.75,
                         'reasons': ['Consumer banking strength', 'Rate sensitivity', 'Improving efficiency']},
                        {'ticker': 'C', 'action': 'Hold', 'confidence': 0.6,
                         'reasons': ['Global exposure', 'Restructuring efforts', 'Execution risk']}
                    ]
                elif event == "Oil Price Increase":
                    recommendations = [
                        {'ticker': 'XOM', 'action': 'Buy', 'confidence': 0.8, 
                         'reasons': ['Upstream benefit', 'Integrated operations', 'Capital return focus']},
                        {'ticker': 'EOG', 'action': 'Buy', 'confidence': 0.75,
                         'reasons': ['Quality shale assets', 'Low cost producer', 'Technology implementation']},
                        {'ticker': 'VLO', 'action': 'Hold', 'confidence': 0.65,
                         'reasons': ['Refining margins', 'Crack spread dynamics', 'Demand uncertainty']}
                    ]
            
            # If event not specifically handled, provide generic recommendations
            if len(recommendations) == 0:
                if risk_profile == 'conservative':
                    recommendations = [
                        {'ticker': 'VYM', 'action': 'Buy', 'confidence': 0.7, 
                         'reasons': ['High dividend yield', 'Value orientation', 'Lower volatility']},
                        {'ticker': 'SCHD', 'action': 'Buy', 'confidence': 0.65,
                         'reasons': ['Quality dividend growers', 'Low expense ratio', 'Diversification']},
                        {'ticker': 'BND', 'action': 'Buy', 'confidence': 0.6,
                         'reasons': ['Fixed income exposure', 'Portfolio stabilizer', 'Income generation']}
                    ]
                elif risk_profile == 'aggressive':
                    recommendations = [
                        {'ticker': 'QQQ', 'action': 'Buy', 'confidence': 0.7, 
                         'reasons': ['Technology exposure', 'Growth orientation', 'Innovation focus']},
                        {'ticker': 'VUG', 'action': 'Buy', 'confidence': 0.65,
                         'reasons': ['Growth stock emphasis', 'Market leaders', 'Long-term potential']},
                        {'ticker': 'ARKK', 'action': 'Hold', 'confidence': 0.55,
                         'reasons': ['Disruptive innovation', 'High-growth potential', 'Elevated volatility']}
                    ]
                else:  # moderate
                    recommendations = [
                        {'ticker': 'VTI', 'action': 'Buy', 'confidence': 0.7, 
                         'reasons': ['Broad market exposure', 'Low cost', 'Diversification']},
                        {'ticker': 'VXUS', 'action': 'Buy', 'confidence': 0.65,
                         'reasons': ['International exposure', 'Diversification benefit', 'Valuation opportunity']},
                        {'ticker': 'BND', 'action': 'Hold', 'confidence': 0.6,
                         'reasons': ['Fixed income allocation', 'Portfolio ballast', 'Income component']}
                    ]
        
        else:
            # Unknown model type
            recommendations = [
                {'ticker': 'SPY', 'action': 'Hold', 'confidence': 0.5, 
                 'reasons': ['Model type unknown', 'Default recommendation', 'Broad market exposure']}
            ]
        
        return recommendations
    
    def backtest_strategy(self, recommendations, investment_amount=10000, 
                          start_date=None, end_date=None, rebalance_period='monthly'):
        """
        Backtest an investment strategy based on recommendations
        
        Parameters:
        - recommendations: List of recommendations with tickers and actions
        - investment_amount: Initial investment amount
        - start_date: Start date for backtest
        - end_date: End date for backtest
        - rebalance_period: How often to rebalance ('daily', 'weekly', 'monthly')
        
        Returns:
        - performance: Dictionary with performance metrics
        """
        # Filter stock data for backtest period
        backtest_data = self.stock_data.copy()
        
        if start_date:
            backtest_data = backtest_data.loc[start_date:]
        if end_date:
            backtest_data = backtest_data.loc[:end_date]
        
        # Get list of tickers from recommendations
        recommended_tickers = [rec['ticker'] for rec in recommendations]
        
        # Make sure we have data for all recommended tickers
        valid_tickers = [ticker for ticker in recommended_tickers if ticker in backtest_data.columns]
        
        if len(valid_tickers) == 0:
            print("No valid tickers found in stock data")
            return None
        
        # Create a portfolio based on recommendations
        portfolio = {}
        cash = investment_amount
        
        # Initial portfolio allocation
        for rec in recommendations:
            ticker = rec['ticker']
            action = rec['action']
            
            if ticker not in valid_tickers:
                continue
                
            if action == 'Buy':
                # Allocate equally among buy recommendations
                buy_tickers = [r['ticker'] for r in recommendations if r['action'] == 'Buy' and r['ticker'] in valid_tickers]
                allocation = cash / len(buy_tickers) if len(buy_tickers) > 0 else 0
                
                # Get initial price
                initial_price = backtest_data[ticker].iloc[0]
                
                # Calculate shares
                shares = allocation / initial_price
                
                # Update portfolio and cash
                portfolio[ticker] = shares
                cash -= allocation
        
        # Initialize performance tracking
        dates = backtest_data.index
        portfolio_values = []
        
        # Set rebalance dates
        if rebalance_period == 'daily':
            rebalance_dates = dates
        elif rebalance_period == 'weekly':
            rebalance_dates = [date for i, date in enumerate(dates) if i % 5 == 0]  # Approximate weekly
        else:  # monthly
            rebalance_dates = [date for i, date in enumerate(dates) if i % 20 == 0]  # Approximate monthly
        
        # Track portfolio value over time
        for date in dates:
            # Calculate portfolio value
            portfolio_value = cash
            
            for ticker, shares in portfolio.items():
                if ticker in backtest_data.columns:
                    price = backtest_data.loc[date, ticker]
                    portfolio_value += shares * price
            
            portfolio_values.append(portfolio_value)
            
            # Rebalance if needed
            if date in rebalance_dates and date != dates[-1]:  # Skip last date
                # Reset portfolio
                cash = portfolio_value
                portfolio = {}
                
                # Reallocate based on recommendations
                for rec in recommendations:
                    ticker = rec['ticker']
                    action = rec['action']
                    
                    if ticker not in valid_tickers:
                        continue
                        
                    if action == 'Buy':
                        # Allocate equally among buy recommendations
                        buy_tickers = [r['ticker'] for r in recommendations if r['action'] == 'Buy' and r['ticker'] in valid_tickers]
                        allocation = cash / len(buy_tickers) if len(buy_tickers) > 0 else 0
                        
                        # Get price at rebalance date
                        price = backtest_data.loc[date, ticker]
                        
                        # Calculate shares
                        shares = allocation / price
                        
                        # Update portfolio and cash
                        portfolio[ticker] = shares
                        cash -= allocation
        
        # Calculate performance metrics
        initial_value = portfolio_values[0]
        final_value = portfolio_values[-1]
        total_return = (final_value - initial_value) / initial_value
        
        # Calculate annualized return
        years = (dates[-1] - dates[0]).days / 365.25
        annualized_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
        
        # Calculate volatility
        daily_returns = [0]
        for i in range(1, len(portfolio_values)):
            daily_return = (portfolio_values[i] - portfolio_values[i-1]) / portfolio_values[i-1]
            daily_returns.append(daily_return)
        
        volatility = np.std(daily_returns) * np.sqrt(252)  # Annualized
        
        # Calculate Sharpe Ratio (assuming risk-free rate of 0.02)
        sharpe_ratio = (annualized_return - 0.02) / volatility if volatility > 0 else 0
        
        # Calculate drawdown
        max_drawdown = 0
        peak = portfolio_values[0]
        
        for value in portfolio_values:
            if value > peak:
                peak = value
            
            drawdown = (peak - value) / peak
            max_drawdown = max(max_drawdown, drawdown)
        
        # Calculate benchmark (SPY) performance
        if 'SPY' in backtest_data.columns:
            spy_start = backtest_data['SPY'].iloc[0]
            spy_end = backtest_data['SPY'].iloc[-1]
            spy_return = (spy_end - spy_start) / spy_start
            spy_annualized = (1 + spy_return) ** (1 / years) - 1 if years > 0 else 0
        else:
            spy_return = None
            spy_annualized = None
        
        return {
            'dates': dates,
            'portfolio_values': portfolio_values,
            'initial_value': initial_value,
            'final_value': final_value,
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'spy_return': spy_return,
            'spy_annualized': spy_annualized
        }
    
    def compare_strategies(self, model_types, news_df=None, event="Interest Rate Hike", 
                           risk_profiles=None, investment_horizon="long-term"):
        """
        Compare different investment strategies
        
        Parameters:
        - model_types: List of model types to compare
        - news_df: DataFrame containing news (for supervised model)
        - event: Specific event to analyze
        - risk_profiles: Dictionary mapping model types to risk profiles
        - investment_horizon: Investment time horizon
        
        Returns:
        - comparison: Dictionary with performance metrics for each strategy
        """
        comparison = {}
        
        for model_type in model_types:
            # Get risk profile for this model type
            risk_profile = "moderate"  # default
            if risk_profiles and model_type in risk_profiles:
                risk_profile = risk_profiles[model_type]
            
            # Generate recommendations
            recommendations = self.generate_recommendations(
                model_type, news_df, event, risk_profile, investment_horizon
            )
            
            # Backtest strategy
            performance = self.backtest_strategy(recommendations)
            
            # Store results
            comparison[model_type] = {
                'recommendations': recommendations,
                'performance': performance
            }
        
        return comparison
    
    def plot_comparison(self, comparison):
        """
        Plot performance comparison of different strategies
        
        Parameters:
        - comparison: Dictionary with performance metrics for each strategy
        """
        # Create figure
        plt.figure(figsize=(15, 10))
        
        # Plot portfolio values
        plt.subplot(2, 2, 1)
        
        for model_type, data in comparison.items():
            performance = data['performance']
            if performance:
                plt.plot(performance['dates'], performance['portfolio_values'], label=model_type)
        
        plt.title('Portfolio Value Over Time')
        plt.xlabel('Date')
        plt.ylabel('Value ($)')
        plt.legend()
        plt.grid(True)
        
        # Plot returns comparison
        plt.subplot(2, 2, 2)
        
        model_types = []
        annualized_returns = []
        total_returns = []
        sharpe_ratios = []
        
        for model_type, data in comparison.items():
            performance = data['performance']
            if performance:
                model_types.append(model_type)
                annualized_returns.append(performance['annualized_return'] * 100)  # Convert to percentage
                total_returns.append(performance['total_return'] * 100)  # Convert to percentage
                sharpe_ratios.append(performance['sharpe_ratio'])
        
        x = range(len(model_types))
        width = 0.3
        
        plt.bar([i - width/2 for i in x], annualized_returns, width=width, label='Annualized Return (%)')
        plt.bar([i + width/2 for i in x], total_returns, width=width, label='Total Return (%)')
        
        plt.title('Returns Comparison')
        plt.xlabel('Model Type')
        plt.ylabel('Return (%)')
        plt.xticks(x, model_types)
        plt.legend()
        plt.grid(True, axis='y')
        
        # Plot Sharpe ratio comparison
        plt.subplot(2, 2, 3)
        
        plt.bar(model_types, sharpe_ratios)
        plt.title('Sharpe Ratio Comparison')
        plt.xlabel('Model Type')
        plt.ylabel('Sharpe Ratio')
        plt.grid(True, axis='y')
        
        # Plot recommendations
        plt.subplot(2, 2, 4)
        
        for i, (model_type, data) in enumerate(comparison.items()):
            recommendations = data['recommendations']
            
            # Extract tickers and actions
            tickers = [rec['ticker'] for rec in recommendations]
            actions = [rec['action'] for rec in recommendations]
            
            # Plot as scatter
            colors = {'Buy': 'green', 'Hold': 'blue', 'Sell': 'red'}
            action_colors = [colors[action] for action in actions]
            
            plt.scatter([i] * len(tickers), range(len(tickers)), c=action_colors, s=100)
            
            # Add ticker labels
            for j, ticker in enumerate(tickers):
                plt.text(i, j, ticker, ha='center', va='center')
        
        plt.title('Recommendations by Model')
        plt.xlabel('Model Type')
        plt.ylabel('Recommendation')
        plt.yticks([])
        plt.xticks(range(len(model_types)), model_types)
        
        # Add legend for actions
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='Buy'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Hold'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Sell')
        ]
        plt.legend(handles=legend_elements, loc='best')
        
        plt.tight_layout()
        plt.savefig('strategy_comparison.png')
        plt.close()


# Initialize fine-tuner
fine_tuner = ModelFineTuner()

# Simulate unsupervised fine-tuning
unsupervised_model = fine_tuner.unsupervised_fine_tuning(research_reports)

# Simulate supervised fine-tuning
supervised_model = fine_tuner.supervised_fine_tuning(market_news, stock_data)

# Simulate instruction fine-tuning
instruction_model = fine_tuner.instruction_fine_tuning(qa_pairs)

# Initialize investment strategy
strategy = InvestmentStrategy(stock_data, start_date='2021-01-01', end_date='2022-12-31')

# Compare strategies for different events
events = ["Interest Rate Hike", "Oil Price Increase"]

for event in events:
    print(f"\nComparing strategies for event: {event}")
    
    # Define different risk profiles for different models
    risk_profiles = {
        'baseline': 'moderate',
        'unsupervised': 'moderate',
        'supervised': 'moderate',
        'instruction': 'moderate'
    }
    
    # Compare strategies
    comparison = strategy.compare_strategies(
        model_types=['baseline', 'unsupervised', 'supervised', 'instruction'],
        news_df=market_news,
        event=event,
        risk_profiles=risk_profiles,
        investment_horizon="long-term"
    )
    
    # Plot comparison
    strategy.plot_comparison(comparison)
    
    # Print performance summary
    print("\nPerformance Summary:")
    print(f"{'Model Type':<15} {'Annualized Return':<20} {'Sharpe Ratio':<15} {'Max Drawdown':<15}")
    print("-" * 65)
    
    for model_type, data in comparison.items():
        performance = data['performance']
        if performance:
            print(f"{model_type:<15} {performance['annualized_return']*100:18.2f}% {performance['sharpe_ratio']:13.2f} {performance['max_drawdown']*100:13.2f}%")
    
    # Print recommendations
    print("\nRecommendations:")
    for model_type, data in comparison.items():
        recommendations = data['recommendations']
        print(f"\n{model_type} Model Recommendations:")
        for rec in recommendations:
            print(f"  {rec['ticker']} ({rec['action']}, confidence: {rec['confidence']:.2f})")
            print(f"    Reasons: {', '.join(rec['reasons'])}")


def simulate_user_query(query, model_types=['baseline', 'unsupervised', 'supervised', 'instruction']):
    """
    Simulate how different models would respond to a user query
    
    Parameters:
    - query: User query string
    - model_types: List of model types to simulate
    
    Returns:
    - responses: Dictionary of responses from each model
    """
    responses = {}
    
    # Define some example queries and responses
    query_responses = {
        "How would rising interest rates affect my portfolio?": {
            'baseline': """
            Rising interest rates typically have mixed effects on investment portfolios. 
            
            Stocks may face pressure as higher rates increase borrowing costs and discount rates for future earnings. Growth stocks with earnings expected far in the future are particularly sensitive.
            
            Bonds typically fall in price when rates rise, with longer duration bonds experiencing larger price declines. However, higher rates eventually lead to higher income.
            
            Some sectors like Financials may benefit from rising rates as banks can increase their net interest margins.
            
            I would recommend maintaining a diversified portfolio with some allocation to value stocks, financials, and shorter-duration fixed income.
            """,
            
            'unsupervised': """
            Rising interest rates impact portfolios through multiple channels, with effects varying significantly by sector and asset class.
            
            Stock Recommendation: JPM (Buy)
            
            Top 3 Reasons:
            1. Net interest margin expansion directly benefits from rate hikes
            2. Strong deposit franchise provides low-cost funding advantage
            3. Diverse business model balances interest rate exposure
            
            Detailed Explanation:
            Interest rate increases typically pressure equity valuations through higher discount rates applied to future earnings. This effect is most pronounced for long-duration growth stocks in technology and consumer discretionary sectors, where earnings are expected far in the future.
            
            Fixed income investments generally experience price declines as rates rise, with the magnitude determined by duration. Bonds with shorter maturities are less sensitive, while long-duration bonds can experience significant price volatility.
            
            Sector impacts vary considerably:
            - Financials (particularly banks) often benefit through expanded net interest margins
            - Real estate typically faces pressure from higher mortgage costs and cap rates
            - Utilities and telecom stocks with high dividend yields become less attractive as bonds offer higher yields
            - Consumer discretionary may weaken as borrowing costs rise for both businesses and consumers
            
            For portfolio positioning, consider:
            1. Reducing duration in fixed income allocations
            2. Increasing exposure to financials, particularly quality banks
            3. Shifting from growth to value stocks
            4. Maintaining adequate cash reserves for potential opportunities
            
            The JPMorgan Chase (JPM) recommendation reflects the bank's strong positioning for a rising rate environment, with its diversified business model providing stability while its significant deposit base creates net interest margin expansion opportunity.
            """,
            
            'supervised': """
            Based on historical data correlations between interest rate increases and market performance, I can provide the following analysis:
            
            Stock Recommendation: JPM (Buy)
            
            Top 3 Reasons:
            1. Historical outperformance during Federal Reserve tightening cycles
            2. Strong correlation between net interest margin expansion and rate increases
            3. Technical indicators show positive momentum in financial sector
            
            Detailed Explanation:
            Analyzing past interest rate hiking cycles shows consistent patterns across asset classes and sectors. The data indicates bond prices typically decline (particularly longer durations), while equities show mixed performance depending on the reason for rate increases.
            
            Statistical analysis of sector performance during rate hikes shows:
            - Financial stocks have outperformed the broader market in 7 of the last 8 hiking cycles
            - Utility and real estate stocks have underperformed in 6 of the last 8 hiking cycles
            - Consumer discretionary performance varies based on economic strength
            
            Time-series analysis indicates JPM specifically has demonstrated positive price momentum during recent rate increase announcements, with an average 30-day return of 4.2% following Fed hikes.
            
            For portfolio protection, the data suggests:
            1. Reducing fixed income duration to under 5 years
            2. Increasing allocation to financial stocks by 3-5%
            3. Reducing exposure to interest-rate sensitive sectors by 2-4%
            
            This recommendation is based purely on statistical correlations from historical market data and technical indicators.
            """,
            
            'instruction': """
            Rising interest rates will impact your portfolio differently across asset classes and sectors, creating both challenges and opportunities.
            
            Stock Recommendation: JPM (Buy)
            
            Top 3 Reasons:
            1. Leading bank with significant benefit from higher interest rates
            2. Strong deposit franchise provides durable funding advantage
            3. Diversified business model reduces risk while capturing rate benefits
            
            Detailed Explanation:
            When interest rates rise, your portfolio will likely experience several effects:
            
            Fixed Income: Existing bonds will decline in value, with longer-duration bonds experiencing larger price drops. However, new bond investments will offer higher yields, eventually improving income. Consider shortening duration, emphasizing Treasury Inflation-Protected Securities (TIPS), and laddering maturities to manage this transition.
            
            Equities: The impact varies significantly by sector:
            - Financial stocks typically benefit, particularly banks like JPMorgan Chase (JPM) that earn higher net interest margins
            - Growth stocks with earnings expected far in the future often underperform as discount rates increase
            - Value stocks and companies with current cash flows tend to outperform
            - Real estate and utilities typically face pressure from higher financing costs and more attractive fixed income alternatives
            
            Cash and Cash Equivalents: Higher rates improve returns on money market funds, short-term CDs, and Treasury bills, making these more attractive for capital preservation.
            
            For your portfolio strategy, consider:
            1. Increasing allocation to financial sector with emphasis on quality institutions
            2. Reducing exposure to high-multiple growth stocks and interest-rate sensitive sectors
            3. Shortening fixed income duration while maintaining appropriate allocation
            4. Utilizing rising cash yields for liquidity needs while awaiting opportunities
            
            JPMorgan Chase (JPM) stands out as particularly well-positioned due to its industry-leading deposit base, diversified revenue streams, and strong risk management. The company has historically navigated rising rate environments effectively and should benefit from expanded margins while maintaining credit quality.
            """
        },
        
        "What sectors would benefit from increasing inflation?": {
            'baseline': """
            Several sectors typically benefit from increasing inflation:
            
            1. Energy: Oil and gas companies often perform well as energy prices are a key component of inflation.
            
            2. Materials: Companies that produce commodities like metals, chemicals, and building materials can pass higher costs to customers.
            
            3. Real Estate: Physical assets like real estate often appreciate during inflationary periods.
            
            4. Financial Services: Banks may benefit from wider net interest margins if interest rates rise in response to inflation.
            
            5. Consumer Staples: Companies with pricing power can pass increased costs to consumers.
            
            Sectors that typically underperform during inflation include technology, utilities, and consumer discretionary companies without pricing power.
            """,
            
            'unsupervised': """
            Increasing inflation creates specific sector opportunities while challenging others, based on pricing power, input costs, and asset positioning.
            
            Stock Recommendation: XOM (Buy)
            
            Top 3 Reasons:
            1. Direct beneficiary of rising energy prices, which are often key inflation drivers
            2. Integrated business model provides margin protection across value chain
            3. Strong balance sheet and capital return program enhance shareholder value
            
            Detailed Explanation:
            Inflation impacts sectors differently based on several key factors: pricing power, cost structure, debt levels, and asset composition. The following sectors typically outperform during inflationary environments:
            
            1. Energy: Companies like Exxon Mobil (XOM) directly benefit as oil and natural gas prices often drive broader inflation. The energy sector has historically provided one of the strongest inflation hedges, with average real returns of 9% during high inflation periods.
            
            2. Materials: Producers of commodities including metals, chemicals, and construction materials benefit from rising input prices. Companies with low-cost production assets generate expanding margins when selling prices increase.
            
            3. Real Estate: Physical assets with replacement costs linked to inflation tend to maintain real value. REITs with short-term leases and strong occupancy can adjust rents upward with inflation.
            
            4. Agriculture: Companies involved in food production benefit from increasing agricultural commodity prices, particularly those with owned land assets.
            
            5. Financial Services: Banks can benefit if rising inflation leads to higher interest rates, expanding net interest margins. Insurance companies with inflation-adjusted pricing models also perform well.
            
            Sectors typically challenged by inflation include:
            - Consumer Discretionary: Companies without pricing power face margin compression
            - Technology: Higher discount rates applied to future earnings affect valuations
            - Utilities: Regulated returns often lag inflation adjustments
            
            Within the energy sector, Exxon Mobil (XOM) offers superior inflation protection through its integrated model spanning exploration, production, refining, and chemicals. This vertical integration allows XOM to capture value across the energy chain regardless of where inflation impacts are strongest.
            """,
            
            'supervised': """
            Analysis of historical market data during inflationary periods reveals clear sector performance patterns that can guide investment decisions.
            
            Stock Recommendation: XOM (Buy)
            
            Top 3 Reasons:
            1. Statistical outperformance during inflation with 87% correlation to CPI increases
            2. Technical indicators show strong momentum coinciding with inflation data
            3. Price action demonstrates effective pass-through of input costs
            
            Detailed Explanation:
            Examining market returns during periods of increasing inflation (defined as CPI > 4%) over the past 50 years shows consistent sector performance patterns. Using regression analysis and correlation studies, the following sectors demonstrate statistically significant outperformance:
            
            1. Energy: Highest inflation beta at 2.3, meaning the sector typically rises 2.3% for every 1% increase in inflation. Energy commodities are themselves inflation components, creating direct exposure.
            
            2. Materials: Second highest inflation beta at 1.7, with mining and metals companies showing strongest correlations to inflation metrics.
            
            3. Real Assets: REITs and infrastructure investments demonstrate inflation betas of 1.2-1.5, with those having inflation-linked revenue adjustments performing best.
            
            4. Consumer Staples with Pricing Power: Companies with leading brands and inelastic demand show inflation betas of 0.8-1.2, successfully preserving margins.
            
            Technical analysis of Exxon Mobil (XOM) shows strong relative strength during recent inflation readings, with the stock outperforming the S&P 500 by an average of 4.2% in months with above-consensus CPI readings.
            
            The data suggests energy exposure should be increased by 3-5% above benchmark weights during persistent inflation, with XOM specifically showing the most consistent performance metrics among major integrated energy companies.
            """,
            
            'instruction': """
            Inflation creates important sector rotation opportunities by fundamentally changing relative value and competitive positioning across the market.
            
            Stock Recommendation: LIN (Buy)
            
            Top 3 Reasons:
            1. Exceptional pricing power from essential industrial gases with limited substitution options
            2. Long-term contracts with inflation adjustment clauses protect margins
            3. Diversified customer base across healthcare, manufacturing, and food processing
            
            Detailed Explanation:
            During periods of increasing inflation, sector performance diverges based on several key factors: pricing power, input cost sensitivity, balance sheet structure, and duration of cash flows.
            
            Sectors well-positioned for inflation include:
            
            1. Materials: Companies like Linde (LIN) with pricing power and essential products can pass through higher costs while maintaining or expanding margins. The industrial gas industry's consolidated structure enables pricing discipline, while long-term contracts often contain explicit inflation adjustment mechanisms.
            
            2. Energy: Both traditional and renewable energy producers benefit as energy prices are often key inflation drivers. Companies with low production costs and limited capital needs generate expanding cash flows.
            
            3. Select Financials: Banks typically benefit if inflation leads to higher interest rates, expanding net interest margins. Insurance companies with inflation-adjusted pricing models can also perform well.
            
            4. Agriculture: Farmland and agricultural producers benefit from higher food prices, particularly those with owned land assets and efficient operations.
            
            5. Infrastructure: Companies operating toll roads, transmission lines, and pipelines with inflation-linked revenue adjustments maintain real returns.
            
            Sectors challenged by inflation include:
            - Growth Technology: Higher discount rates reduce present value of distant earnings
            - Consumer Discretionary: Margin compression for companies lacking pricing power
            - Fixed-Rate Utilities: Regulated returns often lag inflation
            
            For portfolio positioning, I recommend:
            1. Emphasizing companies with pricing power and low labor intensity
            2. Focusing on short-duration equities with current cash flows rather than distant earnings
            3. Including real asset exposure through commodities, infrastructure, and REITs
            4. Considering TIPS for fixed income allocation
            
            Linde (LIN) exemplifies the ideal inflation beneficiary through its essential products, pricing power, and contractual protections, making it my highest conviction recommendation for this environment.
            """
        },
        
        "How should I position my portfolio for an economic recession?": {
            'baseline': """
            During an economic recession, consider positioning your portfolio with the following approaches:
            
            1. Increase defensive sectors: Consumer staples, utilities, and healthcare typically outperform during recessions due to inelastic demand.
            
            2. Quality focus: Companies with strong balance sheets, stable cash flows, and low debt perform better in challenging economic conditions.
            
            3. Reduce cyclical exposure: Limit positions in consumer discretionary, industrials, and materials which are more sensitive to economic cycles.
            
            4. Fixed income allocation: Increase allocation to high-quality bonds which often perform well during recessions as interest rates typically fall.
            
            5. Cash reserve: Maintain some cash to take advantage of opportunities that arise from market dislocations.
            
            The specific allocation would depend on your time horizon and risk tolerance.
            """,
            
            'unsupervised': """
            Recession positioning requires a carefully balanced defensive approach while maintaining flexibility for the eventual recovery.
            
            Stock Recommendation: JNJ (Buy)
            
            Top 3 Reasons:
            1. Defensive healthcare business with stable demand through economic cycles
            2. Diversified revenue streams across pharmaceuticals, medical devices, and consumer health
            3. Strong balance sheet with AAA credit rating provides stability in uncertain times
            
            Detailed Explanation:
            Recessions create significant challenges for investors, but historical analysis provides clear guidance on effective portfolio positioning. The optimal recession strategy balances downside protection with recovery positioning.
            
            Sector positioning should emphasize:
            
            1. Consumer Staples: Companies selling essential products experience minimal demand disruption. Focus on those with strong brands, pricing power, and conservative balance sheets.
            
            2. Healthcare: Medical spending remains relatively stable during downturns, particularly for non-discretionary treatments and medications. Johnson & Johnson (JNJ) exemplifies the ideal recession-resistant healthcare investment through its diversified business model spanning pharmaceuticals, medical devices, and consumer health products.
            
            3. Utilities: Essential services with regulated returns provide stability, though selectivity is important regarding balance sheet strength and regulatory environments.
            
            4. Quality Technology: Select technology companies with recurring revenue, strong balance sheets, and essential products/services can outperform despite the sector's general cyclicality.
            
            Sectors to reduce exposure to include:
            - Consumer Discretionary: Particularly vulnerable to spending reductions
            - Industrials: Typically experience significant earnings declines
            - Materials: Demand and pricing weakness create dual headwinds
            - Financials: Credit quality concerns and reduced activity hurt earnings
            
            Asset allocation adjustments should include:
            1. Increasing fixed income allocation with emphasis on high-quality government and corporate bonds
            2. Maintaining adequate cash reserves for both safety and opportunistic deployment
            3. Reducing overall equity exposure while focusing on quality and defense
            4. Considering hedging strategies if appropriate for portfolio size and complexity
            
            Johnson & Johnson (JNJ) provides an ideal core holding during recessionary periods due to its diverse revenue streams, exceptional balance sheet strength, and consistent performance across economic cycles.
            """,
            
            'supervised': """
            Historical market data analysis reveals distinct performance patterns during recessionary periods that can inform optimal portfolio positioning.
            
            Stock Recommendation: PG (Buy)
            
            Top 3 Reasons:
            1. Statistical outperformance during 7 of 8 previous recessions with average excess return of 14.2%
            2. Strong correlation between consumer staples sector leadership and economic contraction
            3. Technical indicators show defensive rotation already beginning in market data
            
            Detailed Explanation:
            Quantitative analysis of market performance during the 8 recessions since 1970 shows clear patterns in sector and factor performance. Using regression models controlling for market beta, I've identified statistically significant outperformance characteristics:
            
            Sector performance during recessions (average excess return vs. S&P 500):
            - Consumer Staples: +12.3%
            - Healthcare: +10.7%
            - Utilities: +8.4%
            - Communication Services: +3.2%
            - Technology: -4.6%
            - Industrials: -8.9%
            - Materials: -11.3%
            - Consumer Discretionary: -13.8%
            
            Factor performance during recessions:
            - Low volatility: +9.4%
            - Dividend yield: +7.8%
            - Quality (high ROE, low debt): +6.5%
            - Value: +2.1%
            - Momentum: -3.7%
            - Growth: -5.2%
            
            The data indicates Procter & Gamble (PG) has demonstrated remarkable consistency during economic contractions, outperforming the broader market in 7 of 8 recessions with an average excess return of 14.2%. Technical indicators including relative strength, moving average convergence/divergence, and money flow suggest defensive rotation is already beginning.
            
            For optimal positioning, market data suggests:
            1. Increasing consumer staples exposure by 5-7% above benchmark
            2. Reducing cyclical exposure by 8-10% below benchmark
            3. Shifting 15-20% of equity allocation to high-quality fixed income
            4. Maintaining 5-10% cash position for opportunistic deployment
            """,
            
            'instruction': """
            Recession portfolio positioning requires balancing defensive protection with preparedness for the eventual recovery that inevitably follows.
            
            Stock Recommendation: WMT (Buy)
            
            Top 3 Reasons:
            1. Consumer spending shifts to value retailers during economic contraction
            2. Essential product focus ensures continued demand regardless of conditions
            3. Scale advantages and operational efficiency provide competitive moat
            
            Detailed Explanation:
            Recessions create significant investment challenges but also opportunities for disciplined investors. A thoughtful recession strategy requires both defensive positioning and preparation for the eventual recovery.
            
            Immediate portfolio actions should include:
            
            1. Quality Focus: Emphasize companies with strong balance sheets, stable cash flows, and competitive advantages. These businesses can withstand economic pressure and potentially gain market share from weaker competitors.
            
            2. Sector Adjustments:
               - Increase: Consumer staples (like Walmart), healthcare, utilities, and select telecommunications
               - Reduce: Consumer discretionary, industrials, materials, and highly leveraged companies
               - Selective Approach: Technology and financials require company-specific analysis rather than sector-wide decisions
            
            3. Fixed Income Allocation: Increase high-quality bonds (Treasury and investment-grade corporate) which typically benefit from flight-to-safety and potential rate cuts.
            
            4. Cash Reserves: Maintain 5-15% cash position (depending on risk tolerance) to provide both protection and dry powder for opportunities.
            
            Walmart (WMT) represents an ideal recession-resistant investment for several reasons:
            - Consumer spending shifts to value retailers as households become more price-conscious
            - Essential product focus ensures continued store traffic regardless of economic conditions
            - Scale advantages in procurement and distribution enable market share gains during challenging periods
            - Omnichannel capabilities provide flexibility for changing consumer behaviors
            - Strong balance sheet with A credit rating provides financial stability
            
            Historical analysis shows Walmart has outperformed the S&P 500 during 7 of the last 8 recessions, with an average outperformance of 12%.
            
            Importantly, while defensive positioning is crucial, remember that markets typically begin recovering before the economy. Maintain a balanced approach that protects capital


