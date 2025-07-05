import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class GeometricMeanMarketMaker:
    """
    Implements a Geometric Mean Market Maker (G3M) with fees.
    """
    
    def __init__(self, x_initial, y_initial, alpha=0.5, fee=0.003):
        """
        Initialize the G3M.
        
        Parameters:
        -----------
        x_initial : float
            Initial reserve of asset X (e.g., BTC)
        y_initial : float
            Initial reserve of asset Y (e.g., USD)
        alpha : float
            Weight parameter, default is 0.5 (equal weights)
        fee : float
            Transaction fee (proportional), default is 0.003 (0.3%)
        """
        self.x = x_initial
        self.y = y_initial
        self.alpha = alpha
        self.fee = fee
        self.beta = alpha / (1 - alpha)
        self.k = (x_initial ** alpha) * (y_initial ** (1 - alpha))
        
        # Track historical reserves and trades
        self.reserve_history = {
            'timestamp': [0],
            'x': [x_initial],
            'y': [y_initial],
            'internal_price': [self.beta * y_initial / x_initial],
            'bid': [(1 - fee) * self.beta * y_initial / x_initial],
            'ask': [self.beta * y_initial / (x_initial * (1 - fee))]
        }
        
        self.trade_history = {
            'timestamp': [],
            'agent': [],
            'delta_x': [],
            'delta_y': [],
            'x_after': [],
            'y_after': []
        }
        
        # Initialize IL tracking
        self.IL_history = {
            'timestamp': [0],
            'external_price': [self.beta * y_initial / x_initial],
            'LP_value': [y_initial + x_initial * self.beta * y_initial / x_initial],
            'hold_value': [y_initial + x_initial * self.beta * y_initial / x_initial],
            'IL': [0],
            'hedge_value': [0],
            'hedged_IL': [0]
        }

    def get_bid_ask(self):
        """Get current bid and ask prices."""
        S = self.beta * self.y / self.x
        bid = (1 - self.fee) * S
        ask = S / (1 - self.fee)
        return bid, ask, S
    
    def execute_trade(self, delta_x, timestamp, agent="liquidity_taker"):
        """
        Execute a trade in the G3M.
        
        Parameters:
        -----------
        delta_x : float
            Amount of asset X to trade (positive for buying Y, negative for selling Y)
        timestamp : float
            Current timestamp
        agent : str
            Type of agent executing the trade (liquidity_taker, arbitrageur)
            
        Returns:
        --------
        delta_y : float
            Amount of asset Y received/given
        """
        if delta_x > 0:  # Buying Y (selling X)
            # Calculate delta_y based on G3M equation with fee
            delta_y = -self.y * (1 - (self.x / (self.x + (1 - self.fee) * delta_x)) ** self.beta)
        else:  # Selling Y (buying X)
            # Calculate delta_y based on G3M equation with fee
            delta_y = -self.y * (1 - (self.x / (self.x + delta_x)) ** self.beta) / (1 - self.fee)
        
        # Update reserves
        self.x += delta_x
        self.y += delta_y
        
        # Record the trade
        self.trade_history['timestamp'].append(timestamp)
        self.trade_history['agent'].append(agent)
        self.trade_history['delta_x'].append(delta_x)
        self.trade_history['delta_y'].append(delta_y)
        self.trade_history['x_after'].append(self.x)
        self.trade_history['y_after'].append(self.y)
        
        # Record reserve state
        bid, ask, internal_price = self.get_bid_ask()
        self.reserve_history['timestamp'].append(timestamp)
        self.reserve_history['x'].append(self.x)
        self.reserve_history['y'].append(self.y)
        self.reserve_history['internal_price'].append(internal_price)
        self.reserve_history['bid'].append(bid)
        self.reserve_history['ask'].append(ask)
        
        return delta_y
    
    def calculate_optimal_arbitrage(self, external_price):
        """
        Calculate the optimal arbitrage trade given the external price.
        
        Parameters:
        -----------
        external_price : float
            Current price in the external market
            
        Returns:
        --------
        delta_x : float
            Optimal trade size (0 if no arbitrage opportunity)
        """
        bid, ask, internal_price = self.get_bid_ask()
        
        if external_price < bid:
            # Buy X from external market, sell in pool
            delta_x = self.x * ((bid / external_price) ** (1 - self.alpha) - 1) / (1 - self.fee)
            return delta_x
        elif external_price > ask:
            # Buy X from pool, sell in external market
            delta_x = -self.x * (1 - (external_price / ask) ** self.alpha)
            return delta_x
        else:
            # No arbitrage opportunity
            return 0
    
    def update_IL(self, external_price, timestamp, x0, y0):
        """
        Update the IL tracking.
        
        Parameters:
        -----------
        external_price : float
            Current price in the external market
        timestamp : float
            Current timestamp
        x0, y0 : float
            Initial reserves
        """
        # Calculate current value of LP
        LP_value = self.y + self.x * external_price
        
        # Calculate value of buy-and-hold strategy
        hold_value = y0 + x0 * external_price
        
        # Calculate IL
        IL = hold_value - LP_value
        
        # Record values
        self.IL_history['timestamp'].append(timestamp)
        self.IL_history['external_price'].append(external_price)
        self.IL_history['LP_value'].append(LP_value)
        self.IL_history['hold_value'].append(hold_value)
        self.IL_history['IL'].append(IL)
        
        return IL

def simulate_g3m_with_hedging(price_data, initial_x, initial_y, fee=0.003, alpha=0.5, 
                              p_trade=0.3, p_small=0.7, time_multiplier=1, seed=None):
    """
    Simulate the G3M with hedging strategy.
    
    Parameters:
    -----------
    price_data : DataFrame
        External market price data with 'timestamp' and 'price' columns
    initial_x : float
        Initial reserve of asset X
    initial_y : float
        Initial reserve of asset Y
    fee : float
        Transaction fee (proportional)
    alpha : float
        Weight parameter
    p_trade : float
        Probability of a liquidity taker trade at each time step
    p_small : float
        Probability that a liquidity taker trade is small
    time_multiplier : float
        Multiplier for time (for scaling simulation time)
    seed : int
        Random seed
        
    Returns:
    --------
    g3m : GeometricMeanMarketMaker
        The simulated G3M
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Initialize G3M
    g3m = GeometricMeanMarketMaker(initial_x, initial_y, alpha, fee)
    
    # Initial values for hedging
    x0 = initial_x
    y0 = initial_y
    hedge_value = 0
    prev_external_price = price_data.iloc[0]['price']
    prev_x = initial_x
    
    # Iterate through price data
    for i in range(1, len(price_data)):
        timestamp = price_data.iloc[i]['timestamp'] * time_multiplier
        external_price = price_data.iloc[i]['price']
        
        # Calculate hedging adjustment based on Equation (13)
        # hedge adjustment = (x0 - prev_x) * (current_price - prev_price)
        hedge_adjustment = (x0 - prev_x) * (external_price - prev_external_price)
        hedge_value += hedge_adjustment
        
        # Check for arbitrage opportunity
        delta_x_arb = g3m.calculate_optimal_arbitrage(external_price)
        
        if abs(delta_x_arb) > 1e-10:  # There is an arbitrage opportunity
            # Execute arbitrage trade
            g3m.execute_trade(delta_x_arb, timestamp, agent="arbitrageur")
        else:
            # With probability p_trade, a liquidity taker executes a trade
            if np.random.random() < p_trade:
                # Determine if it's a small or large trade
                is_small = np.random.random() < p_small
                
                # Generate random trade size
                if is_small:
                    # Small trade (stays within no-arbitrage bounds)
                    bid, ask, _ = g3m.get_bid_ask()
                    if np.random.random() < 0.5:  # Buying Y (selling X)
                        max_price = external_price * 1.01  # Slightly higher than external price
                        max_delta_x = g3m.x * 0.01  # 1% of reserves
                        delta_x = np.random.uniform(0, max_delta_x)
                    else:  # Selling Y (buying X)
                        min_price = external_price * 0.99  # Slightly lower than external price
                        max_delta_x = -g3m.x * 0.01  # 1% of reserves
                        delta_x = np.random.uniform(max_delta_x, 0)
                else:
                    # Large trade (creates arbitrage opportunity)
                    if np.random.random() < 0.5:  # Buying Y (selling X)
                        max_delta_x = g3m.x * 0.05  # 5% of reserves
                        delta_x = np.random.uniform(0, max_delta_x)
                    else:  # Selling Y (buying X)
                        max_delta_x = -g3m.x * 0.05  # 5% of reserves
                        delta_x = np.random.uniform(max_delta_x, 0)
                
                # Execute liquidity taker trade
                g3m.execute_trade(delta_x, timestamp, agent="liquidity_taker")
                
                # If it was a large trade, check for arbitrage again and execute if needed
                if not is_small:
                    delta_x_arb = g3m.calculate_optimal_arbitrage(external_price)
                    if abs(delta_x_arb) > 1e-10:
                        g3m.execute_trade(delta_x_arb, timestamp, agent="arbitrageur")
        
        # Update IL and hedging tracking
        IL = g3m.update_IL(external_price, timestamp, x0, y0)
        
        # Update hedged IL
        g3m.IL_history['hedge_value'].append(hedge_value)
        g3m.IL_history['hedged_IL'].append(IL - hedge_value)
        
        # Update for next iteration
        prev_external_price = external_price
        prev_x = g3m.x
    
    return g3m

def load_real_price_data(ticker="BTC-USD", start_date="2022-01-01", end_date="2022-02-01", interval="1h"):
    """
    Load real cryptocurrency price data.
    """
    try:
        data = yf.download(ticker, start=start_date, end=end_date, interval=interval)
        if data.empty:
            raise ValueError(f"No data found for {ticker} in the specified date range")
        
        # Create DataFrame with normalized timestamps
        price_data = pd.DataFrame({
            'timestamp': range(len(data)),
            'price': data['Close'].values
        })
        
        return price_data, data
    except Exception as e:
        print(f"Error loading data: {e}")
        return generate_simulated_price_data()

def generate_simulated_price_data(n_steps=1000, volatility=0.02, drift=0, initial_price=50000, seed=None):
    """
    Generate simulated price data following geometric Brownian motion.
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Generate price path
    returns = np.random.normal(drift, volatility, n_steps)
    price_path = initial_price * np.cumprod(1 + returns)
    
    # Create DataFrame
    price_data = pd.DataFrame({
        'timestamp': range(n_steps + 1),
        'price': np.concatenate([[initial_price], price_path])
    })
    
    return price_data

def run_experiment(fee_values, price_data, initial_x, initial_y, p_trade=0.3, p_small=0.7):
    """
    Run experiments with different fee values.
    
    Parameters:
    -----------
    fee_values : list
        List of fee values to test
    price_data : DataFrame
        External market price data
    initial_x : float
        Initial reserve of asset X
    initial_y : float
        Initial reserve of asset Y
    p_trade : float
        Probability of a liquidity taker trade
    p_small : float
        Probability that a trade is small
        
    Returns:
    --------
    results : dict
        Dictionary with results for each fee value
    """
    results = {}
    
    for fee in fee_values:
        print(f"Running simulation with fee = {fee:.4f}")
        g3m = simulate_g3m_with_hedging(price_data, initial_x, initial_y, fee, 
                                        p_trade=p_trade, p_small=p_small, seed=42)
        
        # Extract final results
        final_idx = len(g3m.IL_history['IL']) - 1
        
        results[fee] = {
            'g3m': g3m,
            'final_IL': g3m.IL_history['IL'][final_idx],
            'final_hedge_value': g3m.IL_history['hedge_value'][final_idx],
            'final_hedged_IL': g3m.IL_history['hedged_IL'][final_idx],
            'total_trades': len(g3m.trade_history['timestamp']),
            'arb_trades': g3m.trade_history['agent'].count('arbitrageur'),
            'liq_trades': g3m.trade_history['agent'].count('liquidity_taker'),
            'IL_per_trade': g3m.IL_history['IL'][final_idx] / max(1, len(g3m.trade_history['timestamp'])),
            'relative_error': (g3m.IL_history['hedged_IL'][final_idx] / 
                              (g3m.IL_history['hold_value'][final_idx])) * 100  # in percentage
        }
    
    return results

def visualize_results(results, price_data, experiment_name=""):
    """
    Visualize the results of the experiments.
    """
    # Prepare results table
    table_data = []
    for fee, result in results.items():
        table_data.append({
            'Fee (bps)': int(fee * 10000),
            'Final IL': f"{result['final_IL']:.2f}",
            'Hedge Value': f"{result['final_hedge_value']:.2f}",
            'Hedged IL': f"{result['final_hedged_IL']:.2f}",
            'Total Trades': result['total_trades'],
            'Arb Trades': result['arb_trades'],
            'Liq Trades': result['liq_trades'],
            'IL/Trade': f"{result['IL_per_trade']:.4f}",
            'Rel Error (%)': f"{result['relative_error']:.2f}"
        })
    
    results_df = pd.DataFrame(table_data)
    print(f"\nResults for {experiment_name}:")
    print(results_df)
    
    # Create visualization
    plt.figure(figsize=(20, 16))
    
    # Plot external price
    plt.subplot(3, 2, 1)
    plt.plot(price_data['timestamp'], price_data['price'])
    plt.title('External Market Price')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.grid(True)
    
    # Plot IL vs fee
    plt.subplot(3, 2, 2)
    fees = [fee * 10000 for fee in results.keys()]  # Convert to basis points
    ils = [result['final_IL'] for result in results.values()]
    hedge_values = [result['final_hedge_value'] for result in results.values()]
    hedged_ils = [result['final_hedged_IL'] for result in results.values()]
    
    plt.bar(fees, ils, alpha=0.6, label='IL')
    plt.bar(fees, hedge_values, alpha=0.6, label='Hedge Value')
    plt.bar(fees, hedged_ils, alpha=0.6, label='Hedged IL')
    plt.title('Final IL, Hedge Value, and Hedged IL vs Fee')
    plt.xlabel('Fee (basis points)')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    
    # Plot IL per trade vs fee
    plt.subplot(3, 2, 3)
    il_per_trade = [result['IL_per_trade'] for result in results.values()]
    plt.bar(fees, il_per_trade)
    plt.title('IL per Trade vs Fee')
    plt.xlabel('Fee (basis points)')
    plt.ylabel('IL per Trade')
    plt.grid(True)
    
    # Plot number of trades vs fee
    plt.subplot(3, 2, 4)
    total_trades = [result['total_trades'] for result in results.values()]
    arb_trades = [result['arb_trades'] for result in results.values()]
    liq_trades = [result['liq_trades'] for result in results.values()]
    
    plt.bar(fees, total_trades, alpha=0.6, label='Total')
    plt.bar(fees, arb_trades, alpha=0.6, label='Arbitrage')
    plt.bar(fees, liq_trades, alpha=0.6, label='Liquidity Taker')
    plt.title('Number of Trades vs Fee')
    plt.xlabel('Fee (basis points)')
    plt.ylabel('Number of Trades')
    plt.legend()
    plt.grid(True)
    
    # Plot IL and hedged IL over time for a specific fee
    selected_fee = list(results.keys())[0]  # Choose first fee for detailed plots
    g3m = results[selected_fee]['g3m']
    
    plt.subplot(3, 2, 5)
    timestamps = g3m.IL_history['timestamp']
    plt.plot(timestamps, g3m.IL_history['IL'], label='IL')
    plt.plot(timestamps, g3m.IL_history['hedge_value'], label='Hedge Value')
    plt.plot(timestamps, g3m.IL_history['hedged_IL'], label='Hedged IL')
    plt.title(f'IL, Hedge Value, and Hedged IL Over Time (Fee = {selected_fee*10000} bps)')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    
    # Plot LP value and hold value over time
    plt.subplot(3, 2, 6)
    plt.plot(timestamps, g3m.IL_history['LP_value'], label='LP Value')
    plt.plot(timestamps, g3m.IL_history['hold_value'], label='Hold Value')
    plt.title(f'LP Value and Hold Value Over Time (Fee = {selected_fee*10000} bps)')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'g3m_results_{experiment_name.replace(" ", "_")}.png')
    plt.show()

def detailed_analysis(g3m, price_data, experiment_name=""):
    """
    Perform a detailed analysis of a single G3M simulation.
    """
    # Create a figure
    plt.figure(figsize=(20, 16))
    
    # Plot reserves over time
    plt.subplot(3, 2, 1)
    timestamps = g3m.reserve_history['timestamp']
    x_reserves = g3m.reserve_history['x']
    y_reserves = g3m.reserve_history['y']
    
    plt.plot(timestamps, x_reserves, label='X Reserve')
    plt.plot(timestamps, y_reserves, label='Y Reserve')
    plt.title('Reserves Over Time')
    plt.xlabel('Time')
    plt.ylabel('Amount')
    plt.legend()
    plt.grid(True)
    
    # Plot internal price vs external price
    plt.subplot(3, 2, 2)
    internal_prices = g3m.reserve_history['internal_price']
    bid_prices = g3m.reserve_history['bid']
    ask_prices = g3m.reserve_history['ask']
    
    # Interpolate external prices to match reserve history timestamps
    external_prices = np.interp(timestamps, price_data['timestamp'], price_data['price'])
    
    plt.plot(timestamps, internal_prices, label='Internal Price')
    plt.plot(timestamps, bid_prices, label='Bid Price')
    plt.plot(timestamps, ask_prices, label='Ask Price')
    plt.plot(timestamps, external_prices, label='External Price')
    plt.title('Price Comparison')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    
    # Plot trades
    plt.subplot(3, 2, 3)
    if len(g3m.trade_history['timestamp']) > 0:
        trade_timestamps = g3m.trade_history['timestamp']
        delta_x = g3m.trade_history['delta_x']
        agents = g3m.trade_history['agent']
        
        arb_mask = [a == 'arbitrageur' for a in agents]
        liq_mask = [a == 'liquidity_taker' for a in agents]
        
        plt.scatter([trade_timestamps[i] for i in range(len(trade_timestamps)) if arb_mask[i]], 
                   [delta_x[i] for i in range(len(delta_x)) if arb_mask[i]], 
                   color='red', alpha=0.7, label='Arbitrageur')
        
        plt.scatter([trade_timestamps[i] for i in range(len(trade_timestamps)) if liq_mask[i]], 
                   [delta_x[i] for i in range(len(delta_x)) if liq_mask[i]], 
                   color='blue', alpha=0.7, label='Liquidity Taker')
        
        plt.title('Trades (delta_x)')
        plt.xlabel('Time')
        plt.ylabel('Delta X')
        plt.legend()
        plt.grid(True)
    
    # Plot IL and hedging
    plt.subplot(3, 2, 4)
    il_timestamps = g3m.IL_history['timestamp']
    il = g3m.IL_history['IL']
    hedge_value = g3m.IL_history['hedge_value']
    hedged_il = g3m.IL_history['hedged_IL']
    
    plt.plot(il_timestamps, il, label='IL')
    plt.plot(il_timestamps, hedge_value, label='Hedge Value')
    plt.plot(il_timestamps, hedged_il, label='Hedged IL')
    plt.title('IL and Hedging')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    
    # Plot LP value vs hold value
    plt.subplot(3, 2, 5)
    lp_value = g3m.IL_history['LP_value']
    hold_value = g3m.IL_history['hold_value']
    
    plt.plot(il_timestamps, lp_value, label='LP Value')
    plt.plot(il_timestamps, hold_value, label='Hold Value')
    plt.title('LP Value vs Hold Value')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    
    # Plot relative error of hedging
    plt.subplot(3, 2, 6)
    if len(il) > 0:
        relative_error = [hedged_il[i] / hold_value[i] * 100 for i in range(len(hedged_il))]
        
        plt.plot(il_timestamps, relative_error)
        plt.title('Relative Error of Hedging (%)')
        plt.xlabel('Time')
        plt.ylabel('Relative Error (%)')
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'g3m_detailed_analysis_{experiment_name.replace(" ", "_")}.png')
    plt.show()

def main():
    # Try to load real price data, or use simulated data if not available
    try:
        price_data, raw_data = load_real_price_data(
            ticker="BTC-USD", 
            start_date="2022-01-01", 
            end_date="2022-01-15", 
            interval="1h"
        )
        data_description = "BTC-USD (2022-01-01 to 2022-01-15, hourly)"
    except:
        print("Using simulated price data instead.")
        price_data = generate_simulated_price_data(n_steps=500, seed=42)
        data_description = "Simulated price data (GBM, 500 steps)"
    
    # Initial setup
    initial_price = price_data.iloc[0]['price']
    initial_x = 1.0  # 1 BTC
    initial_y = initial_price * initial_x  # USD equivalent
    fee_values = [0.0001, 0.0005, 0.001, 0.0015, 0.003]  # 1, 5, 10, 15, 30 basis points
    
    print(f"Running experiments with {data_description}")
    print(f"Initial price: ${initial_price:.2f}")
    print(f"Initial reserves: {initial_x} BTC, ${initial_y:.2f} USD")
    
    # Experiment 1: Arbitrage traders and small traders (p_trade=1, p_small=1)
    exp1_results = run_experiment(fee_values, price_data, initial_x, initial_y, p_trade=1, p_small=1)
    visualize_results(exp1_results, price_data, "Experiment 1: Arbitrage and Small Traders")
    
    # Experiment 2: Arbitrage traders and large traders (p_trade=1, p_small=0)
    exp2_results = run_experiment(fee_values, price_data, initial_x, initial_y, p_trade=1, p_small=0)
    visualize_results(exp2_results, price_data, "Experiment 2: Arbitrage and Large Traders")
    
    # Experiment 3: Arbitrage traders only (p_trade=0, p_small doesn't matter)
    exp3_results = run_experiment(fee_values, price_data, initial_x, initial_y, p_trade=0, p_small=0)
    visualize_results(exp3_results, price_data, "Experiment 3: Arbitrage Traders Only")
    
    # Detailed analysis of a specific simulation (using the first fee from Experiment 1)
    detailed_analysis(exp1_results[fee_values[0]]['g3m'], price_data, "Fee_1bp")

if __name__ == "__main__":
    main()