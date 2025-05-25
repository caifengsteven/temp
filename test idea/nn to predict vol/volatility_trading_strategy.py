import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from datetime import datetime, timedelta
import os
from scipy.stats import norm
import matplotlib.dates as mdates
from sklearn.preprocessing import StandardScaler

# Import the volatility prediction model
from nn_to_predict_vol import (
    VolatilityNN, 
    connect_to_bloomberg, 
    get_sp500_data, 
    get_vix_data, 
    black_scholes_delta,
    black_scholes_vega,
    calculate_min_variance_delta
)

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create directories for saving results
os.makedirs("strategy_results", exist_ok=True)

class VolatilityTradingStrategy:
    """
    Trading strategy based on volatility predictions from neural network models
    """
    def __init__(self, model_3f_path='models/volatility_model_3f.pth', 
                 model_4f_path='models/volatility_model_4f.pth',
                 use_min_variance_delta=True,
                 rebalance_threshold=0.05,
                 position_limit=1.0,
                 transaction_cost=0.0005):
        """
        Initialize the trading strategy
        
        Parameters:
        model_3f_path: Path to the saved three-feature model
        model_4f_path: Path to the saved four-feature model
        use_min_variance_delta: Whether to use minimum variance delta for hedging
        rebalance_threshold: Threshold for delta change to trigger rebalancing
        position_limit: Maximum position size as a fraction of capital
        transaction_cost: Transaction cost as a fraction of trade value
        """
        self.model_3f_path = model_3f_path
        self.model_4f_path = model_4f_path
        self.use_min_variance_delta = use_min_variance_delta
        self.rebalance_threshold = rebalance_threshold
        self.position_limit = position_limit
        self.transaction_cost = transaction_cost
        
        # Load models
        self.load_models()
        
        # Initialize portfolio
        self.reset_portfolio()
    
    def load_models(self):
        """
        Load the trained neural network models
        """
        try:
            # Load three-feature model
            self.model_3f = VolatilityNN(input_size=3)
            self.model_3f.load_state_dict(torch.load(self.model_3f_path))
            self.model_3f.to(device)
            self.model_3f.eval()
            print(f"Loaded three-feature model from {self.model_3f_path}")
            
            # Load four-feature model
            self.model_4f = VolatilityNN(input_size=4)
            self.model_4f.load_state_dict(torch.load(self.model_4f_path))
            self.model_4f.to(device)
            self.model_4f.eval()
            print(f"Loaded four-feature model from {self.model_4f_path}")
            
            self.models_loaded = True
        except Exception as e:
            print(f"Error loading models: {e}")
            print("Training new models...")
            self.models_loaded = False
    
    def reset_portfolio(self):
        """
        Reset the portfolio to initial state
        """
        self.capital = 1000000  # Initial capital
        self.cash = self.capital  # Available cash
        self.options_position = 0  # Number of options contracts
        self.stock_position = 0  # Number of stock shares
        self.current_delta = 0  # Current portfolio delta
        self.target_delta = 0  # Target portfolio delta
        self.portfolio_values = []  # Track portfolio value over time
        self.trade_history = []  # Track trades
        self.daily_pnl = []  # Track daily P&L
        self.daily_returns = []  # Track daily returns
        self.rebalance_count = 0  # Count rebalancing events
        
    def predict_vol_change(self, index_return, time_to_maturity, delta, vix_level=None):
        """
        Predict implied volatility change using the trained models
        
        Parameters:
        index_return: S&P 500 index return
        time_to_maturity: Option time to maturity in years
        delta: Option delta
        vix_level: VIX level (if using four-feature model)
        
        Returns:
        Predicted implied volatility change
        """
        if not self.models_loaded:
            # Use a simple model if neural network models are not loaded
            a, b, c = -0.2329, 0.4176, -0.4892  # Default parameters from paper
            return index_return * (a + b * delta + c * delta**2) / np.sqrt(time_to_maturity)
        
        with torch.no_grad():
            if vix_level is not None:
                # Use four-feature model
                features = torch.tensor([index_return, time_to_maturity, delta, vix_level], 
                                      dtype=torch.float32).to(device)
                prediction = self.model_4f(features).item()
            else:
                # Use three-feature model
                features = torch.tensor([index_return, time_to_maturity, delta], 
                                      dtype=torch.float32).to(device)
                prediction = self.model_3f(features).item()
        
        return prediction
    
    def calculate_option_price(self, S, K, T, r, q, sigma, option_type='call'):
        """
        Calculate option price using Black-Scholes formula
        
        Parameters:
        S: Spot price
        K: Strike price
        T: Time to maturity (in years)
        r: Risk-free rate
        q: Dividend yield
        sigma: Implied volatility
        option_type: 'call' or 'put'
        
        Returns:
        Option price
        """
        d1 = (np.log(S/K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        if option_type == 'call':
            price = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:  # put
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
        
        return price
    
    def get_hedge_ratio(self, S, K, T, r, q, sigma, vix_level=None):
        """
        Calculate the hedge ratio (delta) for an option
        
        Parameters:
        S: Spot price
        K: Strike price
        T: Time to maturity (in years)
        r: Risk-free rate
        q: Dividend yield
        sigma: Implied volatility
        vix_level: VIX level (if using four-feature model)
        
        Returns:
        Hedge ratio (delta)
        """
        if self.use_min_variance_delta and self.models_loaded:
            # Use minimum variance delta from the model
            if vix_level is not None:
                # Use four-feature model
                delta = calculate_min_variance_delta(
                    self.model_4f, S, K, T, r, q, sigma, vix_level
                )
            else:
                # Use three-feature model
                delta = calculate_min_variance_delta(
                    self.model_3f, S, K, T, r, q, sigma
                )
        else:
            # Use standard Black-Scholes delta
            delta = black_scholes_delta(S, K, T, r, q, sigma)
        
        return delta
    
    def execute_trade(self, date, S, options_data, r, q, vix_level=None):
        """
        Execute trading strategy for the given date
        
        Parameters:
        date: Current date
        S: Current spot price
        options_data: DataFrame with options data
        r: Risk-free rate
        q: Dividend yield
        vix_level: VIX level (if using four-feature model)
        
        Returns:
        Updated portfolio value
        """
        # Filter options that meet our criteria
        valid_options = options_data[
            (options_data['date'] == date) & 
            (options_data['days_to_maturity'] >= 14) &
            (options_data['delta'] >= 0.3) & 
            (options_data['delta'] <= 0.7)
        ]
        
        if len(valid_options) == 0:
            # No valid options for trading
            portfolio_value = self.cash + self.stock_position * S
            self.portfolio_values.append({
                'date': date,
                'portfolio_value': portfolio_value,
                'cash': self.cash,
                'stock_value': self.stock_position * S,
                'options_value': 0,
                'delta': self.current_delta
            })
            return portfolio_value
        
        # Select an ATM option with medium-term maturity
        # In a real strategy, we might use a more sophisticated selection method
        target_option = valid_options.iloc[len(valid_options) // 2]
        
        # Extract option parameters
        K = target_option['strike']
        T = target_option['time_to_maturity']
        sigma = target_option['implied_vol']
        option_delta = target_option['delta']
        option_id = target_option['option_id']
        
        # Calculate option price
        option_price = self.calculate_option_price(S, K, T, r, q, sigma)
        
        # Calculate the hedge ratio
        hedge_ratio = self.get_hedge_ratio(S, K, T, r, q, sigma, vix_level)
        
        # Determine position size based on capital and position limit
        max_options = int((self.capital * self.position_limit) / (option_price * 100))
        
        # If we don't have a position yet, initiate one
        if self.options_position == 0:
            # Buy options
            self.options_position = max_options
            option_cost = self.options_position * option_price * 100
            
            # Calculate required stock position for delta hedging
            required_stock = -self.options_position * hedge_ratio * 100
            stock_cost = abs(required_stock - self.stock_position) * S
            
            # Apply transaction costs
            transaction_cost = (option_cost + stock_cost) * self.transaction_cost
            
            # Update cash and positions
            self.cash -= (option_cost + stock_cost + transaction_cost)
            self.stock_position = required_stock
            
            # Record the trade
            self.trade_history.append({
                'date': date,
                'action': 'INITIATE',
                'option_id': option_id,
                'option_price': option_price,
                'num_options': self.options_position,
                'stock_price': S,
                'stock_position': self.stock_position,
                'hedge_ratio': hedge_ratio,
                'transaction_cost': transaction_cost
            })
            
            # Update current delta
            self.current_delta = self.options_position * option_delta * 100 + self.stock_position
            self.target_delta = 0  # Target neutral delta
        else:
            # We already have a position, check if we need to rebalance
            current_option_delta = self.options_position * option_delta * 100
            current_stock_delta = self.stock_position
            self.current_delta = current_option_delta + current_stock_delta
            
            # Calculate the required stock position for delta hedging
            required_stock = -self.options_position * hedge_ratio * 100
            
            # Check if rebalancing is needed
            delta_change = abs(required_stock - self.stock_position) / (self.options_position * 100)
            
            if delta_change > self.rebalance_threshold:
                # Rebalance the hedge
                stock_trade = required_stock - self.stock_position
                stock_cost = abs(stock_trade) * S
                
                # Apply transaction costs
                transaction_cost = stock_cost * self.transaction_cost
                
                # Update cash and positions
                self.cash -= (stock_cost + transaction_cost)
                self.stock_position = required_stock
                
                # Record the trade
                self.trade_history.append({
                    'date': date,
                    'action': 'REBALANCE',
                    'option_id': option_id,
                    'option_price': option_price,
                    'num_options': self.options_position,
                    'stock_price': S,
                    'stock_position': self.stock_position,
                    'hedge_ratio': hedge_ratio,
                    'transaction_cost': transaction_cost
                })
                
                # Update current delta
                self.current_delta = self.options_position * option_delta * 100 + self.stock_position
                self.rebalance_count += 1
        
        # Calculate current portfolio value
        options_value = self.options_position * option_price * 100
        stock_value = self.stock_position * S
        portfolio_value = self.cash + options_value + stock_value
        
        # Record portfolio value
        self.portfolio_values.append({
            'date': date,
            'portfolio_value': portfolio_value,
            'cash': self.cash,
            'stock_value': stock_value,
            'options_value': options_value,
            'delta': self.current_delta
        })
        
        return portfolio_value
    
    def backtest(self, start_date, end_date, use_four_feature=True):
        """
        Backtest the trading strategy
        
        Parameters:
        start_date: Start date for backtesting
        end_date: End date for backtesting
        use_four_feature: Whether to use the four-feature model
        
        Returns:
        DataFrame with backtest results
        """
        print(f"Backtesting strategy from {start_date} to {end_date}")
        
        # Reset portfolio
        self.reset_portfolio()
        
        # Connect to Bloomberg (or use sample data)
        con = connect_to_bloomberg()
        
        # Get market data
        print("Fetching market data...")
        sp500_data = get_sp500_data(con, start_date, end_date)
        
        if use_four_feature:
            vix_data = get_vix_data(con, start_date, end_date)
        else:
            vix_data = None
        
        # Generate options data
        print("Generating options data...")
        options_data = self.generate_options_data(sp500_data.index, sp500_data['PX_LAST'])
        
        # Set risk-free rate and dividend yield
        r = 0.03  # Risk-free rate
        q = 0.015  # Dividend yield
        
        # Run backtest
        print("Running backtest...")
        prev_date = None
        prev_portfolio_value = self.capital
        
        for date in sp500_data.index:
            # Get current S&P 500 price
            S = sp500_data.loc[date, 'PX_LAST']
            
            # Get VIX level if using four-feature model
            if use_four_feature and vix_data is not None:
                if date in vix_data.index:
                    vix_level = vix_data.loc[date, 'VIX_LAST'] / 100  # Convert to decimal
                else:
                    vix_level = 0.15  # Default value
            else:
                vix_level = None
            
            # Execute trading strategy
            portfolio_value = self.execute_trade(date, S, options_data, r, q, vix_level)
            
            # Calculate daily P&L and return
            if prev_date is not None:
                daily_pnl = portfolio_value - prev_portfolio_value
                daily_return = daily_pnl / prev_portfolio_value
                
                self.daily_pnl.append({
                    'date': date,
                    'pnl': daily_pnl,
                    'return': daily_return
                })
                
                self.daily_returns.append(daily_return)
            
            prev_date = date
            prev_portfolio_value = portfolio_value
        
        # Calculate performance metrics
        self.calculate_performance_metrics()
        
        # Convert results to DataFrames
        portfolio_df = pd.DataFrame(self.portfolio_values)
        trades_df = pd.DataFrame(self.trade_history) if self.trade_history else pd.DataFrame()
        daily_pnl_df = pd.DataFrame(self.daily_pnl) if self.daily_pnl else pd.DataFrame()
        
        # Plot results
        self.plot_results(portfolio_df, daily_pnl_df, use_four_feature)
        
        return {
            'portfolio': portfolio_df,
            'trades': trades_df,
            'daily_pnl': daily_pnl_df,
            'performance_metrics': self.performance_metrics
        }
    
    def generate_options_data(self, dates, spot_prices):
        """
        Generate synthetic options data for backtesting
        
        Parameters:
        dates: Array of dates for the backtest
        spot_prices: Array of spot prices
        
        Returns:
        DataFrame with synthetic options data
        """
        data = []
        
        # Generate a set of options for each day
        for i, date in enumerate(dates):
            S = spot_prices[i]
            
            # Generate 20 options per day with different strikes and maturities
            for j in range(20):
                # Generate option characteristics
                delta = np.random.uniform(0.05, 0.95)
                days_to_maturity = np.random.choice([30, 60, 90, 180, 270, 365])
                time_to_maturity = days_to_maturity / 365.0
                strike = S * (1 + np.random.uniform(-0.2, 0.2))
                implied_vol = np.random.uniform(0.15, 0.35)
                
                # Create record for current day
                data.append({
                    'date': date,
                    'option_id': f"OPTION_{i}_{j}",
                    'strike': strike,
                    'days_to_maturity': days_to_maturity,
                    'time_to_maturity': time_to_maturity,
                    'implied_vol': implied_vol,
                    'delta': delta,
                    'underlying_price': S
                })
        
        # Convert to DataFrame
        return pd.DataFrame(data)
    
    def calculate_performance_metrics(self):
        """
        Calculate performance metrics for the backtest
        """
        if not self.daily_returns:
            self.performance_metrics = {
                'total_return': 0,
                'annualized_return': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'volatility': 0,
                'win_rate': 0,
                'rebalance_count': 0
            }
            return
        
        # Convert to numpy array
        returns = np.array(self.daily_returns)
        
        # Calculate metrics
        total_return = (self.portfolio_values[-1]['portfolio_value'] / self.capital) - 1
        annualized_return = (1 + total_return) ** (252 / len(returns)) - 1
        volatility = np.std(returns) * np.sqrt(252)
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
        
        # Calculate maximum drawdown
        portfolio_values = np.array([p['portfolio_value'] for p in self.portfolio_values])
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (peak - portfolio_values) / peak
        max_drawdown = np.max(drawdown)
        
        # Calculate win rate
        win_rate = np.sum(returns > 0) / len(returns)
        
        self.performance_metrics = {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'volatility': volatility,
            'win_rate': win_rate,
            'rebalance_count': self.rebalance_count
        }
    
    def plot_results(self, portfolio_df, daily_pnl_df, use_four_feature):
        """
        Plot backtest results
        
        Parameters:
        portfolio_df: DataFrame with portfolio values
        daily_pnl_df: DataFrame with daily P&L
        use_four_feature: Whether four-feature model was used
        """
        model_name = "Four-Feature" if use_four_feature else "Three-Feature"
        
        # Create figure with subplots
        fig, axs = plt.subplots(3, 1, figsize=(12, 15), sharex=True)
        
        # Plot portfolio value
        axs[0].plot(portfolio_df['date'], portfolio_df['portfolio_value'], label='Portfolio Value')
        axs[0].set_title(f'Portfolio Value - {model_name} Model')
        axs[0].set_ylabel('Value ($)')
        axs[0].grid(True)
        axs[0].legend()
        
        # Plot portfolio components
        axs[1].plot(portfolio_df['date'], portfolio_df['cash'], label='Cash')
        axs[1].plot(portfolio_df['date'], portfolio_df['stock_value'], label='Stock Value')
        axs[1].plot(portfolio_df['date'], portfolio_df['options_value'], label='Options Value')
        axs[1].set_title('Portfolio Components')
        axs[1].set_ylabel('Value ($)')
        axs[1].grid(True)
        axs[1].legend()
        
        # Plot daily returns
        if not daily_pnl_df.empty:
            axs[2].bar(daily_pnl_df['date'], daily_pnl_df['return'] * 100)
            axs[2].set_title('Daily Returns')
            axs[2].set_ylabel('Return (%)')
            axs[2].set_xlabel('Date')
            axs[2].grid(True)
        
        # Format x-axis dates
        for ax in axs:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax.xaxis.set_major_locator(mdates.MonthLocator())
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        plt.savefig(f'strategy_results/{model_name.lower().replace("-", "_")}_strategy_results.png')
        plt.close()
        
        # Plot portfolio delta
        plt.figure(figsize=(12, 6))
        plt.plot(portfolio_df['date'], portfolio_df['delta'])
        plt.title(f'Portfolio Delta - {model_name} Model')
        plt.ylabel('Delta')
        plt.xlabel('Date')
        plt.grid(True)
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'strategy_results/{model_name.lower().replace("-", "_")}_portfolio_delta.png')
        plt.close()
        
        # Print performance metrics
        print("\nPerformance Metrics:")
        print(f"Total Return: {self.performance_metrics['total_return']*100:.2f}%")
        print(f"Annualized Return: {self.performance_metrics['annualized_return']*100:.2f}%")
        print(f"Sharpe Ratio: {self.performance_metrics['sharpe_ratio']:.2f}")
        print(f"Maximum Drawdown: {self.performance_metrics['max_drawdown']*100:.2f}%")
        print(f"Volatility: {self.performance_metrics['volatility']*100:.2f}%")
        print(f"Win Rate: {self.performance_metrics['win_rate']*100:.2f}%")
        print(f"Rebalance Count: {self.performance_metrics['rebalance_count']}")

def run_strategy_comparison(start_date='2020-01-01', end_date='2020-12-31'):
    """
    Run and compare different trading strategies
    
    Parameters:
    start_date: Start date for backtesting
    end_date: End date for backtesting
    """
    print("=" * 80)
    print("Volatility Trading Strategy Comparison")
    print("=" * 80)
    
    # Create strategy instances
    strategy_3f = VolatilityTradingStrategy(
        use_min_variance_delta=True,
        rebalance_threshold=0.05
    )
    
    strategy_4f = VolatilityTradingStrategy(
        use_min_variance_delta=True,
        rebalance_threshold=0.05
    )
    
    strategy_bs = VolatilityTradingStrategy(
        use_min_variance_delta=False,
        rebalance_threshold=0.05
    )
    
    # Run backtests
    print("\nRunning backtest with three-feature model...")
    results_3f = strategy_3f.backtest(start_date, end_date, use_four_feature=False)
    
    print("\nRunning backtest with four-feature model...")
    results_4f = strategy_4f.backtest(start_date, end_date, use_four_feature=True)
    
    print("\nRunning backtest with standard Black-Scholes delta...")
    results_bs = strategy_bs.backtest(start_date, end_date, use_four_feature=False)
    
    # Compare performance metrics
    metrics_3f = results_3f['performance_metrics']
    metrics_4f = results_4f['performance_metrics']
    metrics_bs = results_bs['performance_metrics']
    
    print("\n" + "=" * 80)
    print("Performance Comparison")
    print("=" * 80)
    print(f"{'Metric':<20} {'Three-Feature':<15} {'Four-Feature':<15} {'Black-Scholes':<15}")
    print("-" * 80)
    print(f"{'Total Return':<20} {metrics_3f['total_return']*100:>14.2f}% {metrics_4f['total_return']*100:>14.2f}% {metrics_bs['total_return']*100:>14.2f}%")
    print(f"{'Annualized Return':<20} {metrics_3f['annualized_return']*100:>14.2f}% {metrics_4f['annualized_return']*100:>14.2f}% {metrics_bs['annualized_return']*100:>14.2f}%")
    print(f"{'Sharpe Ratio':<20} {metrics_3f['sharpe_ratio']:>14.2f} {metrics_4f['sharpe_ratio']:>14.2f} {metrics_bs['sharpe_ratio']:>14.2f}")
    print(f"{'Maximum Drawdown':<20} {metrics_3f['max_drawdown']*100:>14.2f}% {metrics_4f['max_drawdown']*100:>14.2f}% {metrics_bs['max_drawdown']*100:>14.2f}%")
    print(f"{'Volatility':<20} {metrics_3f['volatility']*100:>14.2f}% {metrics_4f['volatility']*100:>14.2f}% {metrics_bs['volatility']*100:>14.2f}%")
    print(f"{'Win Rate':<20} {metrics_3f['win_rate']*100:>14.2f}% {metrics_4f['win_rate']*100:>14.2f}% {metrics_bs['win_rate']*100:>14.2f}%")
    print(f"{'Rebalance Count':<20} {metrics_3f['rebalance_count']:>14d} {metrics_4f['rebalance_count']:>14d} {metrics_bs['rebalance_count']:>14d}")
    print("=" * 80)
    
    # Plot comparison of portfolio values
    plt.figure(figsize=(12, 6))
    plt.plot(results_3f['portfolio']['date'], results_3f['portfolio']['portfolio_value'], label='Three-Feature Model')
    plt.plot(results_4f['portfolio']['date'], results_4f['portfolio']['portfolio_value'], label='Four-Feature Model')
    plt.plot(results_bs['portfolio']['date'], results_bs['portfolio']['portfolio_value'], label='Black-Scholes')
    plt.title('Portfolio Value Comparison')
    plt.ylabel('Value ($)')
    plt.xlabel('Date')
    plt.grid(True)
    plt.legend()
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('strategy_results/strategy_comparison.png')
    plt.close()
    
    return {
        'three_feature': results_3f,
        'four_feature': results_4f,
        'black_scholes': results_bs
    }

if __name__ == "__main__":
    # Run strategy comparison
    results = run_strategy_comparison(start_date='2020-01-01', end_date='2020-12-31')
