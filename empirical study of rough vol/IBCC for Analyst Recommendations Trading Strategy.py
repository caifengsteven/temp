import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import digamma, gamma
from scipy.stats import dirichlet
from tqdm import tqdm
import seaborn as sns
from matplotlib.dates import DateFormatter, WeekdayLocator, DayLocator, MONDAY
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# Set matplotlib style - using a basic style that should work in most environments
try:
    plt.style.use('seaborn')  # Try a simpler style
except:
    pass  # If it fails, use the default style

colors = ['#E24A33', '#348ABD', '#988ED5', '#777777', '#FBC15E', '#8EBA42', '#FFB5B8']
try:
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=colors)
except:
    pass  # If it fails, use the default colors

class SimulatedMarketData:
    """Class to generate simulated stock market data"""
    
    def __init__(self, n_stocks=50, n_days=1000, n_brokers=20):
        """
        Initialize simulation parameters
        
        Parameters:
        -----------
        n_stocks: int
            Number of stocks to simulate
        n_days: int
            Number of trading days to simulate
        n_brokers: int
            Number of brokers providing recommendations
        """
        self.n_stocks = n_stocks
        self.n_days = n_days
        self.n_brokers = n_brokers
        
        # Generate dates
        self.dates = pd.date_range(start='2020-01-01', periods=n_days)
        
        # Broker skill levels (random values between 0 and 1)
        # Higher value means the broker is more skilled
        self.broker_skills = np.random.beta(2, 2, n_brokers)
        
        # Broker bias (tendency to recommend Buy vs Sell)
        # Higher value means more likely to recommend Buy
        self.broker_bias = np.random.beta(5, 3, n_brokers)  # Slightly biased towards Buy
        
        # Broker activity level (probability of making a recommendation on any day)
        # Typical brokers only cover a small subset of stocks and don't update frequently
        self.broker_activity = np.random.beta(1, 20, n_brokers)  # Most brokers are not very active
    
    def generate_price_data(self):
        """Generate stock price data with realistic properties"""
        # Initial prices
        initial_prices = np.random.uniform(10, 100, self.n_stocks)
        
        # Drift and volatility for each stock
        drifts = np.random.normal(0.0001, 0.0002, self.n_stocks)
        volatilities = np.random.uniform(0.01, 0.03, self.n_stocks)
        
        # Price data matrix
        prices = np.zeros((self.n_days, self.n_stocks))
        prices[0, :] = initial_prices
        
        # Generate prices with random walk
        for t in range(1, self.n_days):
            # Market-wide factor (affects all stocks)
            market_return = np.random.normal(0.0001, 0.01)
            
            # Stock-specific returns
            for s in range(self.n_stocks):
                # Stock return = drift + market factor + idiosyncratic component
                stock_return = drifts[s] + 0.7 * market_return + volatilities[s] * np.random.normal(0, 1)
                prices[t, s] = prices[t-1, s] * (1 + stock_return)
        
        # Convert to DataFrame
        self.price_df = pd.DataFrame(prices, index=self.dates)
        
        # Calculate returns
        self.returns_df = self.price_df.pct_change()
        
        return self.price_df
    
    def generate_recommendations(self):
        """Generate broker recommendations based on their skill and bias"""
        # Recommendation types:
        # 0: Missing (no recommendation)
        # 1: Hold
        # 2: Sell
        # 3: Buy
        
        # Initialize recommendation data
        recs = np.zeros((self.n_days, self.n_stocks, self.n_brokers), dtype=int)
        recs.fill(0)  # Default to 0 (Missing)
        
        # For each broker, stock, and day, generate recommendations
        for b in range(self.n_brokers):
            skill = self.broker_skills[b]
            bias = self.broker_bias[b]
            activity = self.broker_activity[b]
            
            # Stocks covered by this broker (each broker covers a subset of stocks)
            covered_stocks = np.random.choice(
                self.n_stocks, 
                size=max(3, int(self.n_stocks * np.random.beta(1, 5))),  # Cover between 3 and ~20% of stocks
                replace=False
            )
            
            for s in covered_stocks:
                # Days when broker makes a recommendation for this stock
                # Recommendations persist for a while and then might be updated
                recommendation_days = []
                current_day = np.random.randint(0, 20)  # Start somewhere in the first 20 days
                
                while current_day < self.n_days:
                    recommendation_days.append(current_day)
                    # Next recommendation in 30-90 days on average
                    current_day += int(np.random.exponential(60) + 1)
                
                for day in recommendation_days:
                    if day >= self.n_days:
                        continue
                        
                    # Look 60 days ahead to see if stock goes up or down
                    look_ahead_day = min(day + 60, self.n_days - 1)
                    future_return = self.price_df.iloc[look_ahead_day, s] / self.price_df.iloc[day, s] - 1
                    
                    # "Ground truth" - what a perfect analyst would recommend
                    if future_return > 0.05:  # Will go up significantly
                        true_rec = 3  # Buy
                    elif future_return < -0.05:  # Will go down significantly
                        true_rec = 2  # Sell
                    else:  # Sideways movement
                        true_rec = 1  # Hold
                    
                    # Broker's recommendation depends on their skill and bias
                    if np.random.random() < skill:
                        # Correct recommendation
                        rec = true_rec
                    else:
                        # Random recommendation
                        if true_rec == 3:  # Should be Buy
                            rec = np.random.choice([1, 2], p=[0.7, 0.3])  # More likely to say Hold than Sell
                        elif true_rec == 2:  # Should be Sell
                            rec = np.random.choice([1, 3], p=[0.5, 0.5])  # Equal chance of Hold or Buy
                        else:  # Should be Hold
                            rec = np.random.choice([2, 3], p=[0.3, 0.7])  # More likely to say Buy than Sell
                    
                    # Apply broker's bias (tendency to recommend Buy)
                    if rec != true_rec:  # Only apply bias when broker is already making a mistake
                        if np.random.random() < bias:
                            # Biased towards Buy
                            rec = 3 if np.random.random() < 0.7 else 1  # 70% Buy, 30% Hold
                    
                    # Store recommendation
                    recs[day, s, b] = rec
                    
                    # Recommendations persist for some days (typically around 30 days)
                    persistence = int(np.random.normal(30, 10))
                    for d in range(1, persistence):
                        if day + d < self.n_days:
                            recs[day + d, s, b] = rec
        
        # Convert to more usable dataframe format
        rec_data = []
        for day in range(self.n_days):
            for stock in range(self.n_stocks):
                broker_recs = recs[day, stock, :]
                rec_data.append({
                    'Date': self.dates[day],
                    'Stock': stock,
                    'Recommendations': broker_recs.tolist(),
                    'Price': self.price_df.iloc[day, stock]
                })
        
        self.recommendations_df = pd.DataFrame(rec_data)
        
        # Calculate future returns for each recommendation
        self.recommendations_df['Future_Return'] = 0.0
        for idx, row in self.recommendations_df.iterrows():
            day_idx = self.dates.get_loc(row['Date'])
            stock_idx = row['Stock']
            if day_idx + 60 < self.n_days:
                future_price = self.price_df.iloc[day_idx + 60, stock_idx]
                current_price = row['Price']
                self.recommendations_df.at[idx, 'Future_Return'] = future_price / current_price - 1
        
        # Calculate price state (Down, Flat, Up) based on future returns
        def get_price_state(ret):
            if ret < -0.05:
                return 0  # Price_Down
            elif ret > 0.05:
                return 2  # Price_Up
            else:
                return 1  # Price_Flat
        
        self.recommendations_df['Price_State'] = self.recommendations_df['Future_Return'].apply(get_price_state)
        
        return self.recommendations_df

class IBCC:
    """
    Independent Bayesian Classifier Combination (IBCC) model with variational approximation
    for analyst recommendation aggregation
    """
    
    def __init__(self, n_brokers, n_states=3, n_recommendations=4, max_iter=100, tol=1e-4):
        """
        Initialize IBCC model
        
        Parameters:
        -----------
        n_brokers: int
            Number of brokers (classifiers)
        n_states: int
            Number of price states (0=Down, 1=Flat, 2=Up)
        n_recommendations: int
            Number of possible recommendations (0=Missing, 1=Hold, 2=Sell, 3=Buy)
        max_iter: int
            Maximum number of iterations for variational inference
        tol: float
            Convergence tolerance for variational inference
        """
        self.n_brokers = n_brokers
        self.n_states = n_states
        self.n_recommendations = n_recommendations
        self.max_iter = max_iter
        self.tol = tol
        
        # Initialize hyperparameters
        # Prior for state probabilities (κ)
        self.v = np.ones(n_states)  # Flat prior
        
        # Prior for recommendation probabilities (π)
        self.alpha = np.ones((n_states, n_brokers, n_recommendations))  # Flat prior
        
        # Current parameter estimates
        self.kappa = np.ones(n_states) / n_states  # Prior probabilities for price states
        self.pi = np.ones((n_states, n_brokers, n_recommendations)) / n_recommendations  # Recommendation probabilities given state
        
        # Variational parameters
        self.v_star = self.v.copy()
        self.alpha_star = self.alpha.copy()
    
    def fit(self, data, rolling_window=False, window_size=None):
        """
        Fit the IBCC model to the data using variational inference
        
        Parameters:
        -----------
        data: DataFrame
            DataFrame with 'Recommendations' and 'Price_State' columns
        rolling_window: bool
            Whether to use a rolling window approach
        window_size: int
            Size of the rolling window (in days)
            
        Returns:
        --------
        self
        """
        if rolling_window and window_size is None:
            raise ValueError("window_size must be specified for rolling_window=True")
        
        # Extract data
        recommendations = np.array([np.array(x) for x in data['Recommendations']])
        price_states = np.array(data['Price_State'])
        
        # Initialize counts
        N_ts = np.zeros((self.n_states, self.n_brokers, self.n_recommendations))
        N_t = np.zeros(self.n_states)
        
        # Count occurrences of each combination
        for i in range(len(data)):
            state = price_states[i]
            recs = recommendations[i]
            
            N_t[state] += 1
            for b in range(self.n_brokers):
                rec = recs[b]
                N_ts[state, b, rec] += 1
        
        # Update variational parameters
        self.v_star = self.v + N_t
        self.alpha_star = self.alpha + N_ts
        
        # Run variational inference
        converged = False
        prev_elbo = -np.inf
        
        for it in range(self.max_iter):
            # Update kappa (state probabilities)
            digamma_sum_v = digamma(np.sum(self.v_star))
            for s in range(self.n_states):
                self.kappa[s] = np.exp(digamma(self.v_star[s]) - digamma_sum_v)
            
            # Update pi (recommendation probabilities given state)
            for s in range(self.n_states):
                for b in range(self.n_brokers):
                    digamma_sum_alpha = digamma(np.sum(self.alpha_star[s, b]))
                    for r in range(self.n_recommendations):
                        self.pi[s, b, r] = np.exp(digamma(self.alpha_star[s, b, r]) - digamma_sum_alpha)
            
            # Calculate ELBO (variational free energy)
            elbo = 0
            # Add contribution from kappa
            try:
                elbo += dirichlet.logpdf(self.kappa, self.v_star)
            except:
                # If dirichlet.logpdf fails, use a simpler approximation
                elbo += np.sum((self.v_star - 1) * np.log(self.kappa))
            
            # Add contribution from pi
            for s in range(self.n_states):
                for b in range(self.n_brokers):
                    try:
                        elbo += dirichlet.logpdf(self.pi[s, b], self.alpha_star[s, b])
                    except:
                        # If dirichlet.logpdf fails, use a simpler approximation
                        elbo += np.sum((self.alpha_star[s, b] - 1) * np.log(self.pi[s, b]))
            
            # Check for convergence
            if np.abs(elbo - prev_elbo) < self.tol:
                converged = True
                break
            
            prev_elbo = elbo
        
        if not converged and self.max_iter > 1:
            print(f"Warning: Variational inference did not converge after {self.max_iter} iterations")
        
        return self
    
    def predict_proba(self, recommendations):
        """
        Predict probability of each price state given broker recommendations
        
        Parameters:
        -----------
        recommendations: array-like
            Array of broker recommendations, shape (n_brokers,)
            
        Returns:
        --------
        array-like
            Probability of each price state, shape (n_states,)
        """
        # Initialize log probabilities
        log_probs = np.zeros(self.n_states)
        
        # Calculate log probabilities for each state
        for s in range(self.n_states):
            log_prob = np.log(self.kappa[s])
            
            for b in range(self.n_brokers):
                rec = recommendations[b]
                log_prob += np.log(self.pi[s, b, rec])
            
            log_probs[s] = log_prob
        
        # Normalize to get probabilities
        max_log_prob = np.max(log_probs)
        probs = np.exp(log_probs - max_log_prob)
        probs = probs / np.sum(probs)
        
        return probs
    
    def predict(self, recommendations):
        """
        Predict most likely price state given broker recommendations
        
        Parameters:
        -----------
        recommendations: array-like
            Array of broker recommendations, shape (n_brokers,)
            
        Returns:
        --------
        int
            Predicted price state
        """
        probs = self.predict_proba(recommendations)
        return np.argmax(probs)

class IBCCTrader:
    """
    Trading strategy based on IBCC model predictions
    """
    
    def __init__(self, n_brokers, c=1.0, k=1.0, lookback_window=None):
        """
        Initialize IBCCTrader
        
        Parameters:
        -----------
        n_brokers: int
            Number of brokers
        c: float
            Parameter controlling the threshold for conditional vs unconditional probabilities
        k: float
            Parameter controlling the threshold between different price states
        lookback_window: int or None
            Number of days to look back for model fitting (None = use all available data)
        """
        self.n_brokers = n_brokers
        self.c = c
        self.k = k
        self.lookback_window = lookback_window
        self.ibcc = IBCC(n_brokers)
        
    def decide_action(self, recommendations, price_state_probs, price_state_priors):
        """
        Decide trading action based on price state probabilities and priors
        
        Parameters:
        -----------
        recommendations: array-like
            Array of broker recommendations, shape (n_brokers,)
        price_state_probs: array-like
            Probability of each price state, shape (n_states,)
        price_state_priors: array-like
            Prior probability of each price state, shape (n_states,)
            
        Returns:
        --------
        int
            Trading action: -1 (Go_Short), 0 (No_Trade), 1 (Go_Long)
        """
        # Extract probabilities
        q0 = price_state_probs[0]  # Price_Down
        q1 = price_state_probs[1]  # Price_Flat
        q2 = price_state_probs[2]  # Price_Up
        
        # Extract priors
        pi0 = price_state_priors[0]  # Price_Down
        pi1 = price_state_priors[1]  # Price_Flat
        pi2 = price_state_priors[2]  # Price_Up
        
        # Calculate Broker_Flw signal
        buy_count = sum(1 for r in recommendations if r == 3)  # Count Buy recommendations
        sell_count = sum(1 for r in recommendations if r == 2)  # Count Sell recommendations
        broker_flw_signal = 1 if buy_count > sell_count else (-1 if sell_count > buy_count else 0)
        
        # Apply decision rule with parameters c and k
        if q0 > self.c * pi0 and q0 > self.k * max(q1, q2):
            ibcc_signal = -1  # Go_Short
        elif q2 > self.c * pi2 and q2 > self.k * max(q0, q1):
            ibcc_signal = 1   # Go_Long
        else:
            ibcc_signal = 0   # No_Trade
        
        # Both strategy: only trade when Broker_Flw and IBCC agree
        if ibcc_signal == broker_flw_signal:
            return ibcc_signal
        else:
            return 0  # No_Trade when signals disagree
    
    def backtest(self, data, initial_capital=10000, position_size=0.2, transaction_cost=0.001):
        """
        Backtest the trading strategy
        
        Parameters:
        -----------
        data: DataFrame
            DataFrame with 'Date', 'Stock', 'Recommendations', 'Price', and 'Price_State' columns
        initial_capital: float
            Initial capital
        position_size: float
            Fraction of capital to use per trade
        transaction_cost: float
            Transaction cost as a percentage
            
        Returns:
        --------
        DataFrame
            DataFrame with backtest results
        """
        # Group data by date and stock
        grouped = data.sort_values('Date').groupby(['Date', 'Stock'])
        
        # Initialize results
        results = []
        positions = {}  # Current positions: {stock_id: (entry_price, entry_date, position_size)}
        capital = initial_capital
        
        # Get unique dates in chronological order
        dates = np.sort(data['Date'].unique())
        
        # Calculate price state priors from all data
        price_state_counts = data['Price_State'].value_counts()
        price_state_priors = np.zeros(3)
        for s in range(3):
            price_state_priors[s] = price_state_counts.get(s, 0) / len(data)
            
        # For each date, update the model and make trading decisions
        for i, date in enumerate(dates):
            if i < 90:  # Skip the first 90 days (not enough data for model training)
                continue
                
            # Calculate start date for lookback window
            if self.lookback_window:
                lookback_start_idx = max(0, i - self.lookback_window)
                start_date = dates[lookback_start_idx]
                training_data = data[(data['Date'] >= start_date) & (data['Date'] < date)]
            else:
                training_data = data[data['Date'] < date]
            
            # Fit model on training data
            self.ibcc.fit(training_data)
            
            # Get today's data
            today_data = data[data['Date'] == date]
            
            # Evaluate current positions
            stocks_to_remove = []
            for stock_id, (entry_price, entry_date, pos_size) in positions.items():
                # Find the stock in today's data
                stock_row = today_data[today_data['Stock'] == stock_id]
                
                if len(stock_row) == 0:
                    # Stock not in today's data, keep the position
                    continue
                
                current_price = stock_row.iloc[0]['Price']
                position_value = pos_size * current_price
                
                # Calculate days held
                entry_date_idx = np.where(dates == entry_date)[0][0]
                days_held = i - entry_date_idx
                
                # Close position if held for 60 days
                if days_held >= 60:
                    if pos_size > 0:  # Long position
                        # Calculate profit/loss
                        profit = position_value - (pos_size * entry_price)
                        profit -= transaction_cost * position_value  # Sell commission
                        
                        # Update capital
                        capital += pos_size * entry_price + profit
                    else:  # Short position
                        # Calculate profit/loss
                        profit = (-pos_size) * (entry_price - current_price)
                        profit -= transaction_cost * (-pos_size) * current_price  # Buy to cover commission
                        
                        # Update capital
                        capital += (-pos_size) * entry_price + profit
                    
                    # Mark for removal
                    stocks_to_remove.append(stock_id)
            
            # Remove closed positions
            for stock_id in stocks_to_remove:
                del positions[stock_id]
            
            # Update positions value
            equity = capital
            for stock_id, (entry_price, entry_date, pos_size) in positions.items():
                stock_row = today_data[today_data['Stock'] == stock_id]
                if len(stock_row) > 0:
                    current_price = stock_row.iloc[0]['Price']
                    equity += pos_size * current_price
            
            # Make trading decisions for each stock
            for _, row in today_data.iterrows():
                stock_id = row['Stock']
                
                # Skip if already have a position in this stock
                if stock_id in positions:
                    continue
                
                # Get recommendations and predict
                recommendations = np.array(row['Recommendations'])
                price_state_probs = self.ibcc.predict_proba(recommendations)
                
                # Decide action
                action = self.decide_action(recommendations, price_state_probs, price_state_priors)
                
                # Execute action
                if action != 0:  # If not No_Trade
                    # Calculate position size in dollars
                    trade_amount = equity * position_size
                    current_price = row['Price']
                    
                    if action == 1:  # Go_Long
                        # Buy
                        shares = trade_amount / current_price
                        capital -= trade_amount
                        # Record position
                        positions[stock_id] = (current_price, date, shares)
                    else:  # Go_Short
                        # Short
                        shares = -trade_amount / current_price
                        capital -= trade_amount
                        # Record position
                        positions[stock_id] = (current_price, date, shares)
            
            # Record daily results
            positions_value = 0
            for stock_id, (entry_price, _, pos_size) in positions.items():
                stock_row = today_data[today_data['Stock'] == stock_id]
                if len(stock_row) > 0:
                    current_price = stock_row.iloc[0]['Price']
                    positions_value += pos_size * current_price
            
            # Calculate daily return
            total_value = capital + positions_value
            if i > 0:
                prev_results = [r for r in results if r['Date'] == dates[i-1]]
                if prev_results:
                    prev_value = prev_results[0]['Equity']
                    daily_return = (total_value / prev_value) - 1
                else:
                    daily_return = 0
            else:
                daily_return = 0
            
            # Record result
            results.append({
                'Date': date,
                'Capital': capital,
                'Positions_Value': positions_value,
                'Equity': total_value,
                'Daily_Return': daily_return,
                'N_Positions': len(positions)
            })
        
        # Convert results to DataFrame
        results_df = pd.DataFrame(results)
        
        # Calculate cumulative returns
        results_df['Cumulative_Return'] = (results_df['Equity'] / initial_capital) - 1
        
        # Calculate buy & hold return using price data
        market_returns = data.groupby('Date')['Price'].mean()
        market_returns = market_returns / market_returns.iloc[0] - 1
        market_returns = market_returns.reset_index()
        market_returns.columns = ['Date', 'Market_Return']
        
        # Merge market returns with results
        results_df = pd.merge(results_df, market_returns, on='Date', how='left')
        
        return results_df
            
def analyze_results(results_df):
    """
    Analyze backtest results
    
    Parameters:
    -----------
    results_df: DataFrame
        DataFrame with backtest results
        
    Returns:
    --------
    dict
        Dictionary with performance metrics
    """
    # Calculate performance metrics
    total_return = results_df['Cumulative_Return'].iloc[-1]
    
    # Annualized return
    days = (results_df['Date'].iloc[-1] - results_df['Date'].iloc[0]).days
    annual_return = ((1 + total_return) ** (252 / days)) - 1
    
    # Volatility
    daily_returns = results_df['Daily_Return']
    volatility = daily_returns.std() * np.sqrt(252)
    
    # Sharpe ratio (assuming risk-free rate = 0)
    sharpe_ratio = annual_return / volatility if volatility > 0 else 0
    
    # Maximum drawdown
    equity_curve = results_df['Equity'].values
    peak = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - peak) / peak
    max_drawdown = np.min(drawdown)
    
    # Market comparison
    market_return = results_df['Market_Return'].iloc[-1]
    
    # Return results
    return {
        'Total Return': total_return,
        'Annual Return': annual_return,
        'Volatility': volatility,
        'Sharpe Ratio': sharpe_ratio,
        'Max Drawdown': max_drawdown,
        'Market Return': market_return,
        'Alpha': total_return - market_return
    }

def plot_results(results_df):
    """
    Plot backtest results
    
    Parameters:
    -----------
    results_df: DataFrame
        DataFrame with backtest results
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 15), gridspec_kw={'height_ratios': [2, 1, 1]})
    
    # Plot equity curve
    axes[0].plot(results_df['Date'], results_df['Equity'], label='Strategy Equity')
    axes[0].set_title('Equity Curve')
    axes[0].grid(True)
    axes[0].legend()
    
    # Plot returns comparison
    axes[1].plot(results_df['Date'], results_df['Cumulative_Return'], label='Strategy Return')
    axes[1].plot(results_df['Date'], results_df['Market_Return'], label='Market Return')
    axes[1].set_title('Cumulative Returns Comparison')
    axes[1].grid(True)
    axes[1].legend()
    
    # Plot number of positions
    axes[2].plot(results_df['Date'], results_df['N_Positions'], label='Number of Positions')
    axes[2].set_title('Portfolio Size')
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.show()

# Broker_Flw trader implementation
class BrokerFlwTrader:
    def __init__(self, n_brokers):
        self.n_brokers = n_brokers
    
    def backtest(self, data, initial_capital=10000, position_size=0.2, transaction_cost=0.001):
        # Group data by date and stock
        grouped = data.sort_values('Date').groupby(['Date', 'Stock'])
        
        # Initialize results
        results = []
        positions = {}  # Current positions: {stock_id: (entry_price, entry_date, position_size)}
        capital = initial_capital
        
        # Get unique dates in chronological order
        dates = np.sort(data['Date'].unique())
        
        # For each date, make trading decisions
        for i, date in enumerate(dates):
            if i < 90:  # Skip the first 90 days
                continue
                
            # Get today's data
            today_data = data[data['Date'] == date]
            
            # Evaluate current positions
            stocks_to_remove = []
            for stock_id, (entry_price, entry_date, pos_size) in positions.items():
                # Find the stock in today's data
                stock_row = today_data[today_data['Stock'] == stock_id]
                
                if len(stock_row) == 0:
                    # Stock not in today's data, keep the position
                    continue
                
                current_price = stock_row.iloc[0]['Price']
                position_value = pos_size * current_price
                
                # Calculate days held
                entry_date_idx = np.where(dates == entry_date)[0][0]
                days_held = i - entry_date_idx
                
                # Close position if held for 60 days
                if days_held >= 60:
                    if pos_size > 0:  # Long position
                        # Calculate profit/loss
                        profit = position_value - (pos_size * entry_price)
                        profit -= transaction_cost * position_value  # Sell commission
                        
                        # Update capital
                        capital += pos_size * entry_price + profit
                    else:  # Short position
                        # Calculate profit/loss
                        profit = (-pos_size) * (entry_price - current_price)
                        profit -= transaction_cost * (-pos_size) * current_price  # Buy to cover commission
                        
                        # Update capital
                        capital += (-pos_size) * entry_price + profit
                    
                    # Mark for removal
                    stocks_to_remove.append(stock_id)
            
            # Remove closed positions
            for stock_id in stocks_to_remove:
                del positions[stock_id]
            
            # Update positions value
            equity = capital
            for stock_id, (entry_price, entry_date, pos_size) in positions.items():
                stock_row = today_data[today_data['Stock'] == stock_id]
                if len(stock_row) > 0:
                    current_price = stock_row.iloc[0]['Price']
                    equity += pos_size * current_price
            
            # Make trading decisions for each stock
            for _, row in today_data.iterrows():
                stock_id = row['Stock']
                
                # Skip if already have a position in this stock
                if stock_id in positions:
                    continue
                
                # Get recommendations
                recommendations = np.array(row['Recommendations'])
                
                # Calculate Broker_Flw signal
                buy_count = sum(1 for r in recommendations if r == 3)  # Count Buy recommendations
                sell_count = sum(1 for r in recommendations if r == 2)  # Count Sell recommendations
                action = 1 if buy_count > sell_count else (-1 if sell_count > buy_count else 0)
                
                # Execute action
                if action != 0:  # If not No_Trade
                    # Calculate position size in dollars
                    trade_amount = equity * position_size
                    current_price = row['Price']
                    
                    if action == 1:  # Go_Long
                        # Buy
                        shares = trade_amount / current_price
                        capital -= trade_amount
                        # Record position
                        positions[stock_id] = (current_price, date, shares)
                    else:  # Go_Short
                        # Short
                        shares = -trade_amount / current_price
                        capital -= trade_amount
                        # Record position
                        positions[stock_id] = (current_price, date, shares)
            
            # Record daily results
            positions_value = 0
            for stock_id, (entry_price, _, pos_size) in positions.items():
                stock_row = today_data[today_data['Stock'] == stock_id]
                if len(stock_row) > 0:
                    current_price = stock_row.iloc[0]['Price']
                    positions_value += pos_size * current_price
            
            # Calculate daily return
            total_value = capital + positions_value
            if i > 0:
                prev_results = [r for r in results if r['Date'] == dates[i-1]]
                if prev_results:
                    prev_value = prev_results[0]['Equity']
                    daily_return = (total_value / prev_value) - 1
                else:
                    daily_return = 0
            else:
                daily_return = 0
            
            # Record result
            results.append({
                'Date': date,
                'Capital': capital,
                'Positions_Value': positions_value,
                'Equity': total_value,
                'Daily_Return': daily_return,
                'N_Positions': len(positions)
            })
        
        # Convert results to DataFrame
        results_df = pd.DataFrame(results)
        
        # Calculate cumulative returns
        results_df['Cumulative_Return'] = (results_df['Equity'] / initial_capital) - 1
        
        # Calculate buy & hold return using price data
        market_returns = data.groupby('Date')['Price'].mean()
        market_returns = market_returns / market_returns.iloc[0] - 1
        market_returns = market_returns.reset_index()
        market_returns.columns = ['Date', 'Market_Return']
        
        # Merge market returns with results
        results_df = pd.merge(results_df, market_returns, on='Date', how='left')
        
        return results_df

# IBCC only trader implementation
class IBCCOnlyTrader:
    def __init__(self, n_brokers, c=1.0, k=1.0, lookback_window=None):
        self.n_brokers = n_brokers
        self.c = c
        self.k = k
        self.lookback_window = lookback_window
        self.ibcc = IBCC(n_brokers)
    
    def decide_action(self, recommendations, price_state_probs, price_state_priors):
        # Extract probabilities
        q0 = price_state_probs[0]  # Price_Down
        q1 = price_state_probs[1]  # Price_Flat
        q2 = price_state_probs[2]  # Price_Up
        
        # Extract priors
        pi0 = price_state_priors[0]  # Price_Down
        pi1 = price_state_priors[1]  # Price_Flat
        pi2 = price_state_priors[2]  # Price_Up
        
        # Apply decision rule with parameters c and k
        if q0 > self.c * pi0 and q0 > self.k * max(q1, q2):
            return -1  # Go_Short
        elif q2 > self.c * pi2 and q2 > self.k * max(q0, q1):
            return 1   # Go_Long
        else:
            return 0   # No_Trade
    
    def backtest(self, data, initial_capital=10000, position_size=0.2, transaction_cost=0.001):
        # Group data by date and stock
        grouped = data.sort_values('Date').groupby(['Date', 'Stock'])
        
        # Initialize results
        results = []
        positions = {}  # Current positions: {stock_id: (entry_price, entry_date, position_size)}
        capital = initial_capital
        
        # Get unique dates in chronological order
        dates = np.sort(data['Date'].unique())
        
        # Calculate price state priors from all data
        price_state_counts = data['Price_State'].value_counts()
        price_state_priors = np.zeros(3)
        for s in range(3):
            price_state_priors[s] = price_state_counts.get(s, 0) / len(data)
            
        # For each date, update the model and make trading decisions
        for i, date in enumerate(dates):
            if i < 90:  # Skip the first 90 days (not enough data for model training)
                continue
                
            # Calculate start date for lookback window
            if self.lookback_window:
                lookback_start_idx = max(0, i - self.lookback_window)
                start_date = dates[lookback_start_idx]
                training_data = data[(data['Date'] >= start_date) & (data['Date'] < date)]
            else:
                training_data = data[data['Date'] < date]
            
            # Fit model on training data
            self.ibcc.fit(training_data)
            
            # Get today's data
            today_data = data[data['Date'] == date]
            
            # Evaluate current positions
            stocks_to_remove = []
            for stock_id, (entry_price, entry_date, pos_size) in positions.items():
                # Find the stock in today's data
                stock_row = today_data[today_data['Stock'] == stock_id]
                
                if len(stock_row) == 0:
                    # Stock not in today's data, keep the position
                    continue
                
                current_price = stock_row.iloc[0]['Price']
                position_value = pos_size * current_price
                
                # Calculate days held
                entry_date_idx = np.where(dates == entry_date)[0][0]
                days_held = i - entry_date_idx
                
                # Close position if held for 60 days
                if days_held >= 60:
                    if pos_size > 0:  # Long position
                        # Calculate profit/loss
                        profit = position_value - (pos_size * entry_price)
                        profit -= transaction_cost * position_value  # Sell commission
                        
                        # Update capital
                        capital += pos_size * entry_price + profit
                    else:  # Short position
                        # Calculate profit/loss
                        profit = (-pos_size) * (entry_price - current_price)
                        profit -= transaction_cost * (-pos_size) * current_price  # Buy to cover commission
                        
                        # Update capital
                        capital += (-pos_size) * entry_price + profit
                    
                    # Mark for removal
                    stocks_to_remove.append(stock_id)
            
            # Remove closed positions
            for stock_id in stocks_to_remove:
                del positions[stock_id]
            
            # Update positions value
            equity = capital
            for stock_id, (entry_price, entry_date, pos_size) in positions.items():
                stock_row = today_data[today_data['Stock'] == stock_id]
                if len(stock_row) > 0:
                    current_price = stock_row.iloc[0]['Price']
                    equity += pos_size * current_price
            
            # Make trading decisions for each stock
            for _, row in today_data.iterrows():
                stock_id = row['Stock']
                
                # Skip if already have a position in this stock
                if stock_id in positions:
                    continue
                
                # Get recommendations and predict
                recommendations = np.array(row['Recommendations'])
                price_state_probs = self.ibcc.predict_proba(recommendations)
                
                # Decide action
                action = self.decide_action(recommendations, price_state_probs, price_state_priors)
                
                # Execute action
                if action != 0:  # If not No_Trade
                    # Calculate position size in dollars
                    trade_amount = equity * position_size
                    current_price = row['Price']
                    
                    if action == 1:  # Go_Long
                        # Buy
                        shares = trade_amount / current_price
                        capital -= trade_amount
                        # Record position
                        positions[stock_id] = (current_price, date, shares)
                    else:  # Go_Short
                        # Short
                        shares = -trade_amount / current_price
                        capital -= trade_amount
                        # Record position
                        positions[stock_id] = (current_price, date, shares)
            
            # Record daily results
            positions_value = 0
            for stock_id, (entry_price, _, pos_size) in positions.items():
                stock_row = today_data[today_data['Stock'] == stock_id]
                if len(stock_row) > 0:
                    current_price = stock_row.iloc[0]['Price']
                    positions_value += pos_size * current_price
            
            # Calculate daily return
            total_value = capital + positions_value
            if i > 0:
                prev_results = [r for r in results if r['Date'] == dates[i-1]]
                if prev_results:
                    prev_value = prev_results[0]['Equity']
                    daily_return = (total_value / prev_value) - 1
                else:
                    daily_return = 0
            else:
                daily_return = 0
            
            # Record result
            results.append({
                'Date': date,
                'Capital': capital,
                'Positions_Value': positions_value,
                'Equity': total_value,
                'Daily_Return': daily_return,
                'N_Positions': len(positions)
            })
        
        # Convert results to DataFrame
        results_df = pd.DataFrame(results)
        
        # Calculate cumulative returns
        results_df['Cumulative_Return'] = (results_df['Equity'] / initial_capital) - 1
        
        # Calculate buy & hold return using price data
        market_returns = data.groupby('Date')['Price'].mean()
        market_returns = market_returns / market_returns.iloc[0] - 1
        market_returns = market_returns.reset_index()
        market_returns.columns = ['Date', 'Market_Return']
        
        # Merge market returns with results
        results_df = pd.merge(results_df, market_returns, on='Date', how='left')
        
        return results_df

# Run the simulation
print("Starting simulation...")
sim = SimulatedMarketData(n_stocks=50, n_days=1000, n_brokers=20)
print("Generating price data...")
price_df = sim.generate_price_data()
print("Generating recommendations...")
recommendations_df = sim.generate_recommendations()

# Initialize traders
print("Setting up traders...")
broker_flw_trader = BrokerFlwTrader(sim.n_brokers)
ibcc_only_trader = IBCCOnlyTrader(sim.n_brokers, c=1.0, k=1.0, lookback_window=252)
both_trader = IBCCTrader(sim.n_brokers, c=1.0, k=1.0, lookback_window=252)

# Run backtests
print("Running Broker_Flw backtest...")
broker_flw_results = broker_flw_trader.backtest(recommendations_df)
print("Running IBCC Only backtest...")
ibcc_only_results = ibcc_only_trader.backtest(recommendations_df)
print("Running Both strategy backtest...")
both_results = both_trader.backtest(recommendations_df)

# Analyze results
broker_flw_metrics = analyze_results(broker_flw_results)
ibcc_only_metrics = analyze_results(ibcc_only_results)
both_metrics = analyze_results(both_results)

# Print results
print("\nBroker_Flw Strategy Performance Metrics:")
for key, value in broker_flw_metrics.items():
    print(f"{key}: {value:.4f}")

print("\nIBCC Only Strategy Performance Metrics:")
for key, value in ibcc_only_metrics.items():
    print(f"{key}: {value:.4f}")

print("\nBoth Strategy Performance Metrics:")
for key, value in both_metrics.items():
    print(f"{key}: {value:.4f}")

# Plot results
plt.figure(figsize=(12, 8))
plt.plot(broker_flw_results['Date'], broker_flw_results['Cumulative_Return'], label='Broker_Flw Strategy')
plt.plot(ibcc_only_results['Date'], ibcc_only_results['Cumulative_Return'], label='IBCC Only Strategy')
plt.plot(both_results['Date'], both_results['Cumulative_Return'], label='Both Strategy')
plt.plot(broker_flw_results['Date'], broker_flw_results['Market_Return'], label='Market')
plt.title('Cumulative Returns Comparison')
plt.xlabel('Date')
plt.ylabel('Cumulative Return')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()