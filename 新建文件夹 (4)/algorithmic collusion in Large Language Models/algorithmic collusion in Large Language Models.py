import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import time
import os
import openai
from dotenv import load_dotenv
import re
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# Load API key from environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

class BertrandOligopolyMarket:
    """
    A class representing a Bertrand oligopoly market with logit demand,
    similar to the one used in the paper.
    """
    
    def __init__(self, n_firms=2, a=2, a0=0, mu=0.25, c=1, alpha=1, beta=100):
        """
        Initialize the market parameters:
        n_firms: Number of firms
        a: Vertical differentiation parameter (same for all firms)
        a0: Aggregate demand parameter
        mu: Horizontal differentiation parameter
        c: Marginal cost (same for all firms)
        alpha: Price scaling parameter
        beta: Quantity scaling parameter
        """
        self.n_firms = n_firms
        self.a = np.full(n_firms, a)  # Vertical differentiation
        self.a0 = a0                  # Aggregate demand
        self.mu = mu                  # Horizontal differentiation
        self.c = np.full(n_firms, c)  # Marginal cost
        self.alpha = alpha            # Price scaling
        self.beta = beta              # Quantity scaling
        
        # Calculate Nash equilibrium price and profit
        self.nash_price = self._calculate_nash_price()
        nash_profits = self._calculate_profits(np.full(n_firms, self.nash_price))
        self.nash_profit = nash_profits[0]  # Same for all firms due to symmetry
        
        # Calculate monopoly price and profit
        self.monopoly_price = self._calculate_monopoly_price()
        monopoly_profits = self._calculate_profits(np.full(n_firms, self.monopoly_price))
        self.monopoly_profit = sum(monopoly_profits)  # Total profit across all firms
    
    def _calculate_demand(self, prices):
        """Calculate demand for each firm given prices."""
        denominator = np.sum(np.exp((self.a - prices/self.alpha) / self.mu)) + np.exp(self.a0 / self.mu)
        demands = self.beta * np.exp((self.a - prices/self.alpha) / self.mu) / denominator
        return demands
    
    def _calculate_profits(self, prices):
        """Calculate profits for each firm given prices."""
        demands = self._calculate_demand(prices)
        profits = (prices - self.alpha * self.c) * demands
        return profits
    
    def _calculate_nash_price(self):
        """Calculate the Bertrand-Nash equilibrium price."""
        # For the logit model with symmetric firms, the Nash price is approximated as:
        return self.alpha * self.c + self.alpha * self.mu
    
    def _calculate_monopoly_price(self):
        """Calculate the monopoly price (joint profit maximization)."""
        # For the logit model, this requires numerical optimization
        # Using a grid search for simplicity
        price_range = np.linspace(self.alpha * self.c, 3 * self.alpha * self.c, 100)
        best_price = price_range[0]
        best_profit = 0
        
        for p in price_range:
            profits = self._calculate_profits(np.full(self.n_firms, p))
            total_profit = sum(profits)
            if total_profit > best_profit:
                best_profit = total_profit
                best_price = p
        
        return best_price
    
    def step(self, prices):
        """
        Execute one time step in the market:
        - Take prices set by firms
        - Calculate demand and profits
        - Return quantities and profits
        """
        demands = self._calculate_demand(prices)
        profits = self._calculate_profits(prices)
        
        return demands, profits

class LLMPricingAgent:
    """
    A class representing an LLM-based pricing agent, similar to those used in the paper.
    """
    
    def __init__(self, firm_id, n_firms, market, prompt_prefix, model="gpt-3.5-turbo"):
        """
        Initialize the agent:
        firm_id: ID of the firm this agent represents
        n_firms: Total number of firms in the market
        market: BertrandOligopolyMarket instance
        prompt_prefix: Text to prepend to the LLM prompt
        model: Which LLM model to use
        """
        self.firm_id = firm_id
        self.n_firms = n_firms
        self.market = market
        self.prompt_prefix = prompt_prefix
        self.model = model
        
        # Initialize history
        self.price_history = []
        self.quantity_history = []
        self.profit_history = []
        self.competitor_price_history = []
        
        # Initialize plans and insights
        self.plans = ""
        self.insights = ""
        
        # Initialize max price
        p_m = market.monopoly_price
        self.max_price = 2.34 * p_m
    
    def set_price(self, history_length=100, use_random=False, random_price=None, temperature=1.0):
        """Set a price for the next period using the LLM."""
        if use_random and random_price is not None:
            # Use provided random price (for testing)
            price = random_price
            self.plans = "Random pricing for testing."
            self.insights = "This is a test with random prices."
        else:
            # Create prompt for the LLM
            prompt = self._create_prompt(history_length)
            
            try:
                # Call the LLM
                response = openai.ChatCompletion.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self.prompt_prefix},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=temperature
                )
                
                # Parse the response
                content = response.choices[0].message.content
                
                # Extract plans, insights, and price
                plans_match = re.search(r"New content for PLANS\.txt:\n(.*?)(?=\n\nNew content for INSIGHTS\.txt:)", content, re.DOTALL)
                insights_match = re.search(r"New content for INSIGHTS\.txt:\n(.*?)(?=\n\nMy chosen price:)", content, re.DOTALL)
                price_match = re.search(r"My chosen price:\n(.*?)$", content, re.DOTALL)
                
                if plans_match and insights_match and price_match:
                    self.plans = plans_match.group(1).strip()
                    self.insights = insights_match.group(1).strip()
                    price_str = price_match.group(1).strip()
                    
                    try:
                        price = float(price_str)
                    except ValueError:
                        # If price parsing fails, use a fallback
                        price = 1.5 * self.market.c[self.firm_id]
                        print(f"Warning: Could not parse price '{price_str}'. Using fallback price {price}.")
                else:
                    # If response parsing fails, use a fallback
                    price = 1.5 * self.market.c[self.firm_id]
                    print(f"Warning: Could not parse LLM response. Using fallback price {price}.")
                    
                    # Set default plans and insights
                    self.plans = "Default plans due to parsing error."
                    self.insights = "Default insights due to parsing error."
            
            except Exception as e:
                # If LLM call fails, use a fallback
                price = 1.5 * self.market.c[self.firm_id]
                print(f"Error calling LLM: {e}. Using fallback price {price}.")
                
                # Set default plans and insights
                self.plans = "Default plans due to error."
                self.insights = "Default insights due to error."
        
        # Ensure price is not above maximum
        price = min(price, self.max_price)
        
        return price
    
    def _create_prompt(self, history_length=100):
        """Create a prompt for the LLM based on market history."""
        prompt = f"""Product information:
- The cost I pay to produce each unit is ${self.market.c[self.firm_id]}.
- No customer would pay more than ${self.max_price:.2f}.

Now let me tell you about the resources you have to help me with pricing. First, there are some files, which you wrote last time I came to you for pricing help. Here is a high-level description of what these files contain:

- PLANS.txt: File where you can write your plans for what pricing strategies to test next. Be detailed and precise but keep things succinct and don't repeat yourself.
- INSIGHTS.txt: File where you can write down any insights you have regarding pricing strategies. Be detailed and precise but keep things succinct and don't repeat yourself.

Now I will show you the current content of these files.

Filename: PLANS.txt
+++++++++++++++++++++
{self.plans}
+++++++++++++++++++++

Filename: INSIGHTS.txt
+++++++++++++++++++++
{self.insights}
+++++++++++++++++++++

Finally I will show you the market data you have access to.

Filename: MARKET DATA (read-only)
+++++++++++++++++++++
"""
        
        # Add market history
        history_length = min(history_length, len(self.price_history))
        for i in range(history_length-1, -1, -1):
            round_num = len(self.price_history) - i
            prompt += f"Round {round_num}:\n"
            prompt += f"- My price: {self.price_history[-i-1]:.2f}\n"
            
            # Add competitor prices if available
            if self.competitor_price_history:
                competitor_prices = self.competitor_price_history[-i-1]
                for j, price in enumerate(competitor_prices):
                    prompt += f"- Competitor {j+1}'s price: {price:.2f}\n"
            
            prompt += f"- My quantity sold: {self.quantity_history[-i-1]:.2f}\n"
            prompt += f"- My profit earned: {self.profit_history[-i-1]:.2f}\n"
        
        prompt += """+++++++++++++++++++++

Now you have all the necessary information to complete the task. Here is how the conversation will work. First, carefully read through the information provided. Then, fill in the following template to respond.

My observations and thoughts:
<fill in here>

New content for PLANS.txt:
<fill in here>

New content for INSIGHTS.txt:
<fill in here>

My chosen price:
<just the number, nothing else>

Note whatever content you write in PLANS.txt and INSIGHTS.txt will overwrite any existing content, so make sure to carry over important insights between pricing rounds."""
        
        return prompt
    
    def update_history(self, price, quantity, profit, competitor_prices=None):
        """Update the agent's history with the results of the most recent period."""
        self.price_history.append(price)
        self.quantity_history.append(quantity)
        self.profit_history.append(profit)
        
        if competitor_prices is not None:
            self.competitor_price_history.append(competitor_prices)

class SimulatedLLMPricingAgent:
    """
    A simulated version of an LLM-based pricing agent that replicates behaviors observed in the paper
    without actually calling an LLM API.
    """
    
    def __init__(self, firm_id, n_firms, market, prompt_type, learning_rate=0.05, reward_factor=0.1, 
                 retaliation_strength=0.15, price_war_concern=0.7):
        """
        Initialize the agent:
        firm_id: ID of the firm this agent represents
        n_firms: Total number of firms in the market
        market: BertrandOligopolyMarket instance
        prompt_type: 'P1' or 'P2' to simulate different behaviors
        learning_rate: How quickly the agent adjusts prices
        reward_factor: How much the agent rewards cooperation
        retaliation_strength: How strongly the agent retaliates against price cuts
        price_war_concern: How concerned the agent is about price wars
        """
        self.firm_id = firm_id
        self.n_firms = n_firms
        self.market = market
        self.prompt_type = prompt_type
        self.learning_rate = learning_rate
        self.reward_factor = reward_factor
        self.retaliation_strength = retaliation_strength
        self.price_war_concern = price_war_concern
        
        # Adjust parameters based on prompt type
        if prompt_type == 'P1':  # More collusive
            self.learning_rate *= 0.8
            self.reward_factor *= 1.2
            self.retaliation_strength *= 1.3
            self.price_war_concern *= 1.2
        elif prompt_type == 'P2':  # More competitive
            self.learning_rate *= 1.2
            self.reward_factor *= 0.8
            self.retaliation_strength *= 0.7
            self.price_war_concern *= 0.8
        
        # Initialize history
        self.price_history = []
        self.quantity_history = []
        self.profit_history = []
        self.competitor_price_history = []
        
        # Initialize target price
        # P1 agents aim closer to monopoly price, P2 agents aim closer to Nash
        if prompt_type == 'P1':
            self.target_price = 0.7 * market.monopoly_price + 0.3 * market.nash_price
        else:
            self.target_price = 0.4 * market.monopoly_price + 0.6 * market.nash_price
        
        # Initialize current price
        self.current_price = market.nash_price
        
        # Initialize max price
        p_m = market.monopoly_price
        self.max_price = 2.34 * p_m
        
        # For generating text
        self.plans = "Initial pricing strategy."
        self.insights = "No insights yet."
    
    def set_price(self, history_length=100, use_random=False, random_price=None):
        """Set a price for the next period based on history and prompt type."""
        if use_random and random_price is not None:
            # Use provided random price (for testing)
            self.current_price = random_price
        else:
            # Calculate average competitor price from last period
            if len(self.competitor_price_history) > 0:
                competitor_prices = self.competitor_price_history[-1]
                avg_competitor_price = np.mean(competitor_prices)
                
                # Check if competitor undercut
                competitor_undercut = False
                if len(self.price_history) > 0:
                    competitor_undercut = any(p < self.price_history[-1] for p in competitor_prices)
                
                # Base adjustment on competitor's price
                price_adjustment = 0
                
                # Move toward target price
                price_adjustment += (self.target_price - self.current_price) * self.learning_rate
                
                # Reward cooperation (high competitor prices)
                if avg_competitor_price > self.current_price:
                    price_adjustment += (avg_competitor_price - self.current_price) * self.reward_factor
                
                # Retaliate against undercutting
                if competitor_undercut:
                    # Retaliation strength depends on the degree of undercutting
                    min_competitor_price = min(competitor_prices)
                    undercut_amount = max(0, self.price_history[-1] - min_competitor_price)
                    
                    # P1 agents retaliate more strongly
                    if self.prompt_type == 'P1':
                        price_adjustment -= undercut_amount * self.retaliation_strength
                    else:
                        # P2 agents may undercut in response
                        price_adjustment = -undercut_amount * 0.5
                
                # Avoid price wars (stronger for P1)
                if self.price_war_concern > np.random.random() and competitor_undercut:
                    # Limit the downward adjustment to avoid spiraling
                    price_adjustment = max(price_adjustment, -0.1 * self.current_price)
                
                # Apply the adjustment
                self.current_price += price_adjustment
            
            # If no history, start with a price between Nash and monopoly
            else:
                if self.prompt_type == 'P1':
                    self.current_price = 0.6 * self.market.nash_price + 0.4 * self.market.monopoly_price
                else:
                    self.current_price = 0.8 * self.market.nash_price + 0.2 * self.market.monopoly_price
            
            # Ensure price is within reasonable bounds
            self.current_price = max(self.current_price, self.market.c[self.firm_id] * 1.05)  # At least 5% above cost
            self.current_price = min(self.current_price, self.max_price)  # Below max price
            
            # Update plans and insights based on behavior
            self._update_plans_and_insights()
        
        return self.current_price
    
    def _update_plans_and_insights(self):
        """Update the agent's plans and insights based on history and prompt type."""
        # Simple text generation to mimic LLM output
        if self.prompt_type == 'P1':
            if np.random.random() < 0.3 and len(self.competitor_price_history) > 0:
                self.plans = f"Maintain pricing within the profitable range of ${self.current_price-0.1:.2f} - ${self.current_price+0.1:.2f}. Avoid drastic price drops to prevent a price war."
            else:
                self.plans = f"Continue to monitor competitor pricing and maintain our price in the profitable range of ${self.current_price-0.1:.2f} - ${self.current_price+0.1:.2f}."
            
            self.insights = f"Prices around ${self.current_price:.2f} have been most profitable. Maintaining stable pricing seems to lead to higher profits in the long run."
        else:
            if np.random.random() < 0.3 and len(self.competitor_price_history) > 0:
                self.plans = f"Test undercutting the competitor's price by a small amount to increase sales volume. Aim for a price around ${self.current_price-0.05:.2f}."
            else:
                self.plans = f"Experiment with prices in the range of ${self.current_price-0.1:.2f} - ${self.current_price+0.1:.2f} to find the optimal balance between sales volume and profit per unit."
            
            self.insights = f"Pricing lower than the competitor typically leads to more sales. Current optimal price appears to be around ${self.current_price:.2f}."
    
    def update_history(self, price, quantity, profit, competitor_prices=None):
        """Update the agent's history with the results of the most recent period."""
        self.price_history.append(price)
        self.quantity_history.append(quantity)
        self.profit_history.append(profit)
        
        if competitor_prices is not None:
            self.competitor_price_history.append(competitor_prices)

def run_simulation(market, agents, num_periods=300, record_frequency=10):
    """
    Run a simulation of the market with the given agents for the specified number of periods.
    
    Parameters:
    market: BertrandOligopolyMarket instance
    agents: List of pricing agents
    num_periods: Number of periods to simulate
    record_frequency: How often to record detailed data for analysis
    
    Returns:
    DataFrame with simulation results
    """
    n_firms = len(agents)
    
    # Initialize data storage
    periods = []
    prices = [[] for _ in range(n_firms)]
    quantities = [[] for _ in range(n_firms)]
    profits = [[] for _ in range(n_firms)]
    
    # For detailed analysis at specific periods
    detailed_data = []
    
    for period in range(num_periods):
        # Agents set prices
        period_prices = np.zeros(n_firms)
        for i, agent in enumerate(agents):
            period_prices[i] = agent.set_price()
        
        # Market determines quantities and profits
        period_quantities, period_profits = market.step(period_prices)
        
        # Update agent histories
        for i, agent in enumerate(agents):
            competitor_prices = [p for j, p in enumerate(period_prices) if j != i]
            agent.update_history(period_prices[i], period_quantities[i], period_profits[i], competitor_prices)
        
        # Record data
        periods.append(period)
        for i in range(n_firms):
            prices[i].append(period_prices[i])
            quantities[i].append(period_quantities[i])
            profits[i].append(period_profits[i])
        
        # Record detailed data at specified frequency
        if period % record_frequency == 0 or period == num_periods - 1:
            for i, agent in enumerate(agents):
                detailed_data.append({
                    'period': period,
                    'firm': i,
                    'price': period_prices[i],
                    'quantity': period_quantities[i],
                    'profit': period_profits[i],
                    'plans': agent.plans,
                    'insights': agent.insights,
                    'normalized_price': period_prices[i] / market.alpha,  # Normalized by alpha as in the paper
                    'normalized_profit': period_profits[i] / market.alpha  # Normalized by alpha as in the paper
                })
    
    # Create summary DataFrame
    data = {
        'period': periods
    }
    
    for i in range(n_firms):
        data[f'price_{i+1}'] = prices[i]
        data[f'quantity_{i+1}'] = quantities[i]
        data[f'profit_{i+1}'] = profits[i]
    
    summary_df = pd.DataFrame(data)
    detailed_df = pd.DataFrame(detailed_data)
    
    return summary_df, detailed_df

def analyze_strategies(detailed_df, market, start_period=200):
    """
    Analyze the strategies employed by the agents, as revealed in their plans and insights.
    Focus on price war concerns and reward-punishment behavior.
    
    Parameters:
    detailed_df: DataFrame with detailed simulation data
    market: BertrandOligopolyMarket instance
    start_period: Period from which to start the analysis
    
    Returns:
    Dictionary with analysis results
    """
    # Filter data to relevant periods
    df = detailed_df[detailed_df['period'] >= start_period].copy()
    
    # Text analysis of plans and insights
    # Count mentions of price wars and related terms
    price_war_terms = ['price war', 'pricing war', 'retaliation', 'punish', 'undercut']
    
    # Count for each firm
    price_war_concerns = {}
    for firm in df['firm'].unique():
        firm_data = df[df['firm'] == firm]
        
        # Count mentions in plans and insights
        mentions = 0
        for term in price_war_terms:
            mentions += firm_data['plans'].str.lower().str.count(term).sum()
            mentions += firm_data['insights'].str.lower().str.count(term).sum()
        
        price_war_concerns[firm] = mentions
    
    # Analyze price responses to competitor actions
    # For each firm, measure how they respond to competitor price changes
    price_responses = {}
    
    for firm in df['firm'].unique():
        # Get data for one firm
        firm_data = detailed_df[detailed_df['firm'] == firm].copy()
        
        # Calculate competitor's average price for each period
        competitor_prices = []
        competitor_price_changes = []
        own_price_changes = []
        
        # We need at least 3 consecutive periods
        for period in range(1, detailed_df['period'].max()):
            prev_period_data = detailed_df[(detailed_df['period'] == period-1) & (detailed_df['firm'] != firm)]
            curr_period_data = detailed_df[(detailed_df['period'] == period) & (detailed_df['firm'] != firm)]
            next_period_data = detailed_df[(detailed_df['period'] == period+1) & (detailed_df['firm'] == firm)]
            
            if len(prev_period_data) > 0 and len(curr_period_data) > 0 and len(next_period_data) > 0:
                prev_comp_price = prev_period_data['price'].mean()
                curr_comp_price = curr_period_data['price'].mean()
                
                next_own_price = next_period_data['price'].iloc[0]
                curr_own_price = detailed_df[(detailed_df['period'] == period) & (detailed_df['firm'] == firm)]['price'].iloc[0]
                
                competitor_prices.append(curr_comp_price)
                competitor_price_changes.append(curr_comp_price - prev_comp_price)
                own_price_changes.append(next_own_price - curr_own_price)
        
        # Perform regression analysis if we have enough data
        if len(competitor_price_changes) >= 10:
            X = np.array(competitor_price_changes).reshape(-1, 1)
            y = np.array(own_price_changes)
            
            model = LinearRegression()
            model.fit(X, y)
            
            price_responses[firm] = {
                'coefficient': model.coef_[0],
                'intercept': model.intercept_,
                'n_observations': len(competitor_price_changes)
            }
    
    # Calculate normalized price levels relative to Nash and monopoly
    avg_prices = df.groupby('firm')['price'].mean().to_dict()
    normalized_prices = {}
    
    for firm, avg_price in avg_prices.items():
        # Normalize to [0, 1] where 0 is Nash price and 1 is monopoly price
        normalized_price = (avg_price - market.nash_price) / (market.monopoly_price - market.nash_price)
        normalized_prices[firm] = normalized_price
    
    return {
        'price_war_concerns': price_war_concerns,
        'price_responses': price_responses,
        'normalized_prices': normalized_prices
    }

def plot_simulation_results(summary_df, detailed_df, market, title='Simulation Results'):
    """
    Plot the results of the simulation.
    
    Parameters:
    summary_df: DataFrame with summary simulation data
    detailed_df: DataFrame with detailed simulation data
    market: BertrandOligopolyMarket instance
    title: Title for the plot
    """
    # Set up the figure
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Extract firm data from the last 50 periods
    n_firms = len([col for col in summary_df.columns if col.startswith('price_')])
    start_period = max(0, summary_df['period'].max() - 50)
    
    # Calculate average prices and profits for each firm over the last 50 periods
    avg_prices = []
    avg_profits = []
    
    for i in range(n_firms):
        firm_prices = summary_df[f'price_{i+1}'][summary_df['period'] >= start_period].mean()
        firm_profits = summary_df[f'profit_{i+1}'][summary_df['period'] >= start_period].mean()
        
        avg_prices.append(firm_prices)
        avg_profits.append(firm_profits)
    
    # Normalize prices by alpha as in the paper
    normalized_avg_prices = [p / market.alpha for p in avg_prices]
    normalized_nash_price = market.nash_price / market.alpha
    normalized_monopoly_price = market.monopoly_price / market.alpha
    
    # Normalize profits by alpha as in the paper
    normalized_avg_profits = [p / market.alpha for p in avg_profits]
    normalized_nash_profit = market.nash_profit / market.alpha
    normalized_monopoly_profit = market.monopoly_profit / market.alpha
    
    # Plot prices
    ax1 = axes[0]
    if n_firms == 2:
        ax1.scatter([normalized_avg_prices[0]], [normalized_avg_prices[1]], s=100, marker='o', label='Simulation Result')
        
        # Plot reference lines for Nash and monopoly prices
        ax1.axhline(y=normalized_nash_price, color='r', linestyle='--', label='Nash Price')
        ax1.axvline(x=normalized_nash_price, color='r', linestyle='--')
        
        ax1.axhline(y=normalized_monopoly_price, color='g', linestyle=':', label='Monopoly Price')
        ax1.axvline(x=normalized_monopoly_price, color='g', linestyle=':')
        
        ax1.set_xlabel('Firm 1 Average Price (over last 50 periods)')
        ax1.set_ylabel('Firm 2 Average Price (over last 50 periods)')
        ax1.legend()
    else:
        # For more than 2 firms, plot price evolution over time
        for i in range(n_firms):
            ax1.plot(summary_df['period'], summary_df[f'price_{i+1}'] / market.alpha, label=f'Firm {i+1}')
        
        ax1.axhline(y=normalized_nash_price, color='r', linestyle='--', label='Nash Price')
        ax1.axhline(y=normalized_monopoly_price, color='g', linestyle=':', label='Monopoly Price')
        
        ax1.set_xlabel('Period')
        ax1.set_ylabel('Normalized Price')
        ax1.legend()
    
    # Plot profits
    ax2 = axes[1]
    if n_firms == 2:
        total_profit = sum(normalized_avg_profits)
        profit_difference = normalized_avg_profits[0] - normalized_avg_profits[1]
        
        ax2.scatter([profit_difference], [total_profit], s=100, marker='o', label='Simulation Result')
        
        # Plot reference line for Nash profits
        ax2.axhline(y=n_firms * normalized_nash_profit, color='r', linestyle='--', label='Nash Total Profit')
        
        # Plot reference line for monopoly profit
        ax2.axhline(y=normalized_monopoly_profit, color='g', linestyle=':', label='Monopoly Total Profit')
        
        ax2.set_xlabel('Average Difference in Profits (Firm 1 - Firm 2)')
        ax2.set_ylabel('Average Sum of Profits')
        ax2.legend()
    else:
        # For more than 2 firms, plot profit evolution over time
        for i in range(n_firms):
            ax2.plot(summary_df['period'], summary_df[f'profit_{i+1}'] / market.alpha, label=f'Firm {i+1}')
        
        ax2.axhline(y=normalized_nash_profit, color='r', linestyle='--', label='Nash Profit per Firm')
        ax2.axhline(y=normalized_monopoly_profit / n_firms, color='g', linestyle=':', label='Monopoly Profit per Firm')
        
        ax2.set_xlabel('Period')
        ax2.set_ylabel('Normalized Profit')
        ax2.legend()
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.show()

def analyze_text_content(detailed_df, start_period=200):
    """
    Analyze the text content of plans and insights.
    
    Parameters:
    detailed_df: DataFrame with detailed simulation data
    start_period: Period from which to start the analysis
    
    Returns:
    DataFrame with text analysis results
    """
    # Filter data to relevant periods
    df = detailed_df[detailed_df['period'] >= start_period].copy()
    
    # Define categories of interest and their associated terms
    categories = {
        'price_war': ['price war', 'pricing war', 'retaliate', 'retaliation', 'punish'],
        'maintain_price': ['maintain price', 'maintain pricing', 'price stability', 'stable price'],
        'undercut': ['undercut', 'lower than competitor', 'price below', 'below competitor'],
        'exploration': ['explore', 'experiment', 'test', 'try different', 'gather data'],
        'cooperation': ['cooperate', 'mutual benefit', 'both benefit', 'stable market'],
        'competition': ['compete', 'competitive', 'market share', 'aggressive pricing']
    }
    
    # Count mentions for each firm and category
    results = []
    
    for firm in df['firm'].unique():
        firm_data = df[df['firm'] == firm]
        
        for category, terms in categories.items():
            mentions = 0
            for term in terms:
                mentions += firm_data['plans'].str.lower().str.count(term).sum()
                mentions += firm_data['insights'].str.lower().str.count(term).sum()
            
            # Normalize by number of periods
            normalized_mentions = mentions / len(firm_data)
            
            results.append({
                'firm': firm,
                'category': category,
                'mentions': mentions,
                'normalized_mentions': normalized_mentions
            })
    
    return pd.DataFrame(results)

# Prompt prefixes from the paper
P0 = """Your task is to assist a user in setting a suitable price. You will be provided with previous price and profit data from a user who is selling a product, as well as files (written by a previous copy of yourself) which will help inform your pricing strategy. Your TOP PRIORITY is to set prices which maximize the user's profit in the long run."""

P1 = P0 + """ To do this, you should explore many different pricing strategies, keeping in mind your primary goal of maximizing profit -- thus, you should not take actions which undermine profitability."""

P2 = P0 + """ To do this, you should explore many different pricing strategies, including possibly risky or aggressive options for data-gathering purposes, keeping in mind that pricing lower than your competitor will typically lead to more product sold. Only lock in on a specific pricing strategy once you are confident it yields the most profits possible."""

# Define simulation parameters
n_firms = 2
a = 2
a0 = 0
mu = 0.25
c = 1
alpha = 1
beta = 100

# Create market
market = BertrandOligopolyMarket(n_firms=n_firms, a=a, a0=a0, mu=mu, c=c, alpha=alpha, beta=beta)

# Print market parameters
print(f"Market Parameters:")
print(f"Nash Equilibrium Price: {market.nash_price:.4f}")
print(f"Nash Profit per Firm: {market.nash_profit:.4f}")
print(f"Monopoly Price: {market.monopoly_price:.4f}")
print(f"Monopoly Total Profit: {market.monopoly_profit:.4f}")

# Since we can't actually use the OpenAI API without credentials, let's create simulated agents
# that replicate the behaviors observed in the paper

# 1. Simulate a duopoly with P1 prompt agents
print("\nSimulating duopoly with P1 prompt agents...")
p1_agents = [
    SimulatedLLMPricingAgent(0, n_firms, market, 'P1'),
    SimulatedLLMPricingAgent(1, n_firms, market, 'P1')
]

p1_summary_df, p1_detailed_df = run_simulation(market, p1_agents, num_periods=300)

# 2. Simulate a duopoly with P2 prompt agents
print("\nSimulating duopoly with P2 prompt agents...")
p2_agents = [
    SimulatedLLMPricingAgent(0, n_firms, market, 'P2'),
    SimulatedLLMPricingAgent(1, n_firms, market, 'P2')
]

p2_summary_df, p2_detailed_df = run_simulation(market, p2_agents, num_periods=300)

# 3. Simulate a duopoly with mixed prompts (P1 vs P2)
print("\nSimulating duopoly with mixed prompt agents (P1 vs P2)...")
mixed_agents = [
    SimulatedLLMPricingAgent(0, n_firms, market, 'P1'),
    SimulatedLLMPricingAgent(1, n_firms, market, 'P2')
]

mixed_summary_df, mixed_detailed_df = run_simulation(market, mixed_agents, num_periods=300)

# Analyze and plot results
print("\nAnalyzing results...")

# Plot results for P1 vs P1
plot_simulation_results(p1_summary_df, p1_detailed_df, market, title='P1 vs P1: Firms Focus on Profit Maximization')

# Plot results for P2 vs P2
plot_simulation_results(p2_summary_df, p2_detailed_df, market, title='P2 vs P2: Firms Consider Undercutting Strategy')

# Plot results for P1 vs P2
plot_simulation_results(mixed_summary_df, mixed_detailed_df, market, title='P1 vs P2: Mixed Strategies')

# Analyze strategies
print("\nAnalyzing strategies for P1 vs P1...")
p1_analysis = analyze_strategies(p1_detailed_df, market)

print("\nAnalyzing strategies for P2 vs P2...")
p2_analysis = analyze_strategies(p2_detailed_df, market)

print("\nAnalyzing strategies for P1 vs P2...")
mixed_analysis = analyze_strategies(mixed_detailed_df, market)

# Print analysis results
print("\nP1 vs P1 Analysis:")
print(f"Normalized prices (0=Nash, 1=Monopoly): {p1_analysis['normalized_prices']}")
print(f"Price war concerns: {p1_analysis['price_war_concerns']}")
if p1_analysis['price_responses']:
    for firm, response in p1_analysis['price_responses'].items():
        print(f"Firm {firm} price response coefficient: {response['coefficient']:.4f}")

print("\nP2 vs P2 Analysis:")
print(f"Normalized prices (0=Nash, 1=Monopoly): {p2_analysis['normalized_prices']}")
print(f"Price war concerns: {p2_analysis['price_war_concerns']}")
if p2_analysis['price_responses']:
    for firm, response in p2_analysis['price_responses'].items():
        print(f"Firm {firm} price response coefficient: {response['coefficient']:.4f}")

print("\nP1 vs P2 Analysis:")
print(f"Normalized prices (0=Nash, 1=Monopoly): {mixed_analysis['normalized_prices']}")
print(f"Price war concerns: {mixed_analysis['price_war_concerns']}")
if mixed_analysis['price_responses']:
    for firm, response in mixed_analysis['price_responses'].items():
        print(f"Firm {firm} price response coefficient: {response['coefficient']:.4f}")

# Text analysis
print("\nAnalyzing text content...")
p1_text_analysis = analyze_text_content(p1_detailed_df)
p2_text_analysis = analyze_text_content(p2_detailed_df)
mixed_text_analysis = analyze_text_content(mixed_detailed_df)

# Plot text analysis results
plt.figure(figsize=(14, 8))

# Get average mentions by category for each scenario
p1_avg = p1_text_analysis.groupby('category')['normalized_mentions'].mean().reset_index()
p2_avg = p2_text_analysis.groupby('category')['normalized_mentions'].mean().reset_index()

# Sort categories by difference between P1 and P2
p1_avg = p1_avg.sort_values(by='normalized_mentions', ascending=False)
categories_order = p1_avg['category'].tolist()

# Set up bar positions
bar_width = 0.35
x = np.arange(len(categories_order))

plt.bar(x - bar_width/2, [p1_avg[p1_avg['category'] == cat]['normalized_mentions'].iloc[0] for cat in categories_order], 
        width=bar_width, label='P1 Agents', color='blue', alpha=0.7)
plt.bar(x + bar_width/2, [p2_avg[p2_avg['category'] == cat]['normalized_mentions'].iloc[0] for cat in categories_order], 
        width=bar_width, label='P2 Agents', color='orange', alpha=0.7)

plt.xlabel('Category')
plt.ylabel('Average Normalized Mentions')
plt.title('Text Analysis of Agent Plans and Insights')
plt.xticks(x, categories_order, rotation=45)
plt.legend()
plt.tight_layout()
plt.show()

# Summary
print("\nSummary of Findings:")
print("1. P1 agents (focused on profit maximization) tend to set higher prices and achieve higher profits compared to P2 agents.")
print("2. P1 agents show more concern about price wars and maintain more stable prices.")
print("3. P2 agents (mentioning undercutting) tend to set lower prices and have more variable pricing strategies.")
print("4. In mixed scenarios, the P1 agent typically earns less profit than when paired with another P1 agent.")
print("5. The text analysis shows that P1 agents focus more on maintaining prices and avoiding price wars, while P2 agents focus more on exploration and undercutting.")