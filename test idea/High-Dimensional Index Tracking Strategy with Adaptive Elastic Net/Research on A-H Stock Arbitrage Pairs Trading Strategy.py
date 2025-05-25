import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings

warnings.filterwarnings('ignore')
plt.style.use('ggplot')

def generate_ah_pair(days=1000, seed=42):
    """Generate a synthetic A-H stock pair with cointegration relationship"""
    np.random.seed(seed)
    
    # Time period
    dates = pd.date_range(start='2018-01-01', periods=days)
    
    # Define the relationship parameters
    alpha = 5.0
    beta = 0.8
    
    # Generate A stock prices
    a_price = np.zeros(days)
    a_price[0] = 100.0
    for i in range(1, days):
        # Random walk with slight upward drift
        a_price[i] = a_price[i-1] * (1 + np.random.normal(0.0001, 0.01))
    
    # Generate H stock prices based on A with mean-reverting spread
    h_price = np.zeros(days)
    spread = np.zeros(days)
    
    # Initial values
    h_price[0] = alpha + beta * a_price[0] + np.random.normal(0, 3)
    spread[0] = h_price[0] - (alpha + beta * a_price[0])
    
    for i in range(1, days):
        # Target price based on relationship
        target = alpha + beta * a_price[i]
        
        # Mean reversion factor
        mean_reversion = 0.1  # Strength of pull toward equilibrium
        
        # Current spread
        curr_spread = h_price[i-1] - target
        
        # Generate new price with mean reversion and noise
        h_price[i] = target + curr_spread * (1 - mean_reversion) + np.random.normal(0, 2)
        
        # Calculate spread
        spread[i] = h_price[i] - (alpha + beta * a_price[i])
    
    # Create DataFrame
    data = pd.DataFrame({
        'a_price': a_price,
        'h_price': h_price,
        'spread': spread
    }, index=dates)
    
    return data, alpha, beta

def pairs_trading_backtest(data, train_pct=0.7, transaction_cost=0.0001):
    """
    Backtest both original and improved pairs trading strategies
    
    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame with columns 'a_price', 'h_price', and 'spread'
    train_pct : float
        Percentage of data to use for parameter calculation
    transaction_cost : float
        Transaction cost as percentage of trade value
    """
    # Split data into training and testing periods
    train_size = int(len(data) * train_pct)
    train_data = data.iloc[:train_size]
    test_data = data.iloc[train_size:]
    
    # Calculate parameters from training data
    train_spread = train_data['spread']
    mean_spread = train_spread.mean()
    std_spread = train_spread.std()
    
    # Set thresholds
    upper_threshold = mean_spread + std_spread
    lower_threshold = mean_spread - std_spread
    
    print(f"\nPairs Trading Parameters:")
    print(f"Mean spread: {mean_spread:.4f}")
    print(f"Standard deviation: {std_spread:.4f}")
    print(f"Upper threshold: {upper_threshold:.4f}")
    print(f"Lower threshold: {lower_threshold:.4f}")
    
    # Initialize backtest variables
    initial_capital = 100000
    
    # Original strategy variables
    capital_orig = initial_capital
    capital_history_orig = [initial_capital]
    trades_orig = []
    position_orig = None
    in_long_h_short_a_orig = False
    in_long_a_short_h_orig = False
    
    # Improved strategy variables
    capital_impr = initial_capital
    capital_history_impr = [initial_capital]
    trades_impr = []
    position_impr = None
    down_cross_limit_impr = False
    up_cross_limit_impr = False
    ready_to_long_h_short_a = False
    ready_to_long_a_short_h = False
    
    # Run backtest on test data
    for i in range(len(test_data)):
        current_date = test_data.index[i]
        a_price = test_data['a_price'].iloc[i]
        h_price = test_data['h_price'].iloc[i]
        spread = test_data['spread'].iloc[i]
        
        # Original strategy logic
        if spread < lower_threshold and not in_long_h_short_a_orig and not in_long_a_short_h_orig:
            # Open position: long H, short A
            position_value = capital_orig * 0.4
            h_shares = position_value / h_price
            a_shares = position_value / a_price
            
            position_orig = {
                'h_shares': h_shares,
                'a_shares': a_shares,
                'h_price': h_price,
                'a_price': a_price,
                'entry_date': current_date,
                'entry_day': i
            }
            
            in_long_h_short_a_orig = True
            trades_orig.append({
                'type': 'OPEN_LONG_H_SHORT_A',
                'date': current_date,
                'day': i,
                'a_price': a_price,
                'h_price': h_price,
                'spread': spread
            })
            
        elif spread > upper_threshold and not in_long_a_short_h_orig and not in_long_h_short_a_orig:
            # Open position: long A, short H
            position_value = capital_orig * 0.4
            a_shares = position_value / a_price
            h_shares = position_value / h_price
            
            position_orig = {
                'a_shares': a_shares,
                'h_shares': h_shares,
                'a_price': a_price,
                'h_price': h_price,
                'entry_date': current_date,
                'entry_day': i
            }
            
            in_long_a_short_h_orig = True
            trades_orig.append({
                'type': 'OPEN_LONG_A_SHORT_H',
                'date': current_date,
                'day': i,
                'a_price': a_price,
                'h_price': h_price,
                'spread': spread
            })
            
        elif spread > mean_spread and in_long_h_short_a_orig:
            # Close position: long H, short A
            h_shares = position_orig['h_shares']
            a_shares = position_orig['a_shares']
            entry_a_price = position_orig['a_price']
            entry_h_price = position_orig['h_price']
            
            # Calculate P&L
            h_pnl = (h_price - entry_h_price) * h_shares
            a_pnl = (entry_a_price - a_price) * a_shares
            
            # Calculate transaction costs
            h_cost = h_shares * h_price * transaction_cost
            a_cost = a_shares * a_price * transaction_cost
            
            # Update capital
            total_pnl = h_pnl + a_pnl - h_cost - a_cost
            capital_orig += total_pnl
            
            trades_orig.append({
                'type': 'CLOSE_LONG_H_SHORT_A',
                'date': current_date,
                'day': i,
                'a_price': a_price,
                'h_price': h_price,
                'spread': spread,
                'pnl': total_pnl,
                'entry_date': position_orig['entry_date'],
                'holding_days': (current_date - position_orig['entry_date']).days
            })
            
            in_long_h_short_a_orig = False
            position_orig = None
            
        elif spread < mean_spread and in_long_a_short_h_orig:
            # Close position: long A, short H
            a_shares = position_orig['a_shares']
            h_shares = position_orig['h_shares']
            entry_a_price = position_orig['a_price']
            entry_h_price = position_orig['h_price']
            
            # Calculate P&L
            a_pnl = (a_price - entry_a_price) * a_shares
            h_pnl = (entry_h_price - h_price) * h_shares
            
            # Calculate transaction costs
            a_cost = a_shares * a_price * transaction_cost
            h_cost = h_shares * h_price * transaction_cost
            
            # Update capital
            total_pnl = a_pnl + h_pnl - a_cost - h_cost
            capital_orig += total_pnl
            
            trades_orig.append({
                'type': 'CLOSE_LONG_A_SHORT_H',
                'date': current_date,
                'day': i,
                'a_price': a_price,
                'h_price': h_price,
                'spread': spread,
                'pnl': total_pnl,
                'entry_date': position_orig['entry_date'],
                'holding_days': (current_date - position_orig['entry_date']).days
            })
            
            in_long_a_short_h_orig = False
            position_orig = None
        
        # Record capital history
        capital_history_orig.append(capital_orig)
        
        # Improved strategy logic
        if spread < lower_threshold and not down_cross_limit_impr:
            # Mark crossing of lower threshold
            down_cross_limit_impr = True
            up_cross_limit_impr = False
            
        elif spread > upper_threshold and not up_cross_limit_impr:
            # Mark crossing of upper threshold
            up_cross_limit_impr = True
            down_cross_limit_impr = False
            
        elif down_cross_limit_impr and spread > lower_threshold and spread < mean_spread and not ready_to_long_h_short_a:
            # Setup for entry after crossing back above lower threshold
            ready_to_long_h_short_a = True
            
        elif up_cross_limit_impr and spread < upper_threshold and spread > mean_spread and not ready_to_long_a_short_h:
            # Setup for entry after crossing back below upper threshold
            ready_to_long_a_short_h = True
            
        # Execute entries for improved strategy
        if ready_to_long_h_short_a and not position_impr:
            # Open position: long H, short A
            position_value = capital_impr * 0.4
            h_shares = position_value / h_price
            a_shares = position_value / a_price
            
            position_impr = {
                'h_shares': h_shares,
                'a_shares': a_shares,
                'h_price': h_price,
                'a_price': a_price,
                'entry_date': current_date,
                'entry_day': i
            }
            
            trades_impr.append({
                'type': 'OPEN_LONG_H_SHORT_A',
                'date': current_date,
                'day': i,
                'a_price': a_price,
                'h_price': h_price,
                'spread': spread
            })
            
        elif ready_to_long_a_short_h and not position_impr:
            # Open position: long A, short H
            position_value = capital_impr * 0.4
            a_shares = position_value / a_price
            h_shares = position_value / h_price
            
            position_impr = {
                'a_shares': a_shares,
                'h_shares': h_shares,
                'a_price': a_price,
                'h_price': h_price,
                'entry_date': current_date,
                'entry_day': i
            }
            
            trades_impr.append({
                'type': 'OPEN_LONG_A_SHORT_H',
                'date': current_date,
                'day': i,
                'a_price': a_price,
                'h_price': h_price,
                'spread': spread
            })
            
        # Check for exits
        if position_impr and position_impr.get('h_shares', 0) > position_impr.get('a_shares', 0) and spread > mean_spread:
            # Close position: long H, short A
            h_shares = position_impr['h_shares']
            a_shares = position_impr['a_shares']
            entry_h_price = position_impr['h_price']
            entry_a_price = position_impr['a_price']
            
            # Calculate P&L
            h_pnl = (h_price - entry_h_price) * h_shares
            a_pnl = (entry_a_price - a_price) * a_shares
            
            # Calculate transaction costs
            h_cost = h_shares * h_price * transaction_cost
            a_cost = a_shares * a_price * transaction_cost
            
            # Update capital
            total_pnl = h_pnl + a_pnl - h_cost - a_cost
            capital_impr += total_pnl
            
            trades_impr.append({
                'type': 'CLOSE_LONG_H_SHORT_A',
                'date': current_date,
                'day': i,
                'a_price': a_price,
                'h_price': h_price,
                'spread': spread,
                'pnl': total_pnl,
                'entry_date': position_impr['entry_date'],
                'holding_days': (current_date - position_impr['entry_date']).days
            })
            
            position_impr = None
            ready_to_long_h_short_a = False
            down_cross_limit_impr = False
            
        elif position_impr and position_impr.get('a_shares', 0) > position_impr.get('h_shares', 0) and spread < mean_spread:
            # Close position: long A, short H
            a_shares = position_impr['a_shares']
            h_shares = position_impr['h_shares']
            entry_a_price = position_impr['a_price']
            entry_h_price = position_impr['h_price']
            
            # Calculate P&L
            a_pnl = (a_price - entry_a_price) * a_shares
            h_pnl = (entry_h_price - h_price) * h_shares
            
            # Calculate transaction costs
            a_cost = a_shares * a_price * transaction_cost
            h_cost = h_shares * h_price * transaction_cost
            
            # Update capital
            total_pnl = a_pnl + h_pnl - a_cost - h_cost
            capital_impr += total_pnl
            
            trades_impr.append({
                'type': 'CLOSE_LONG_A_SHORT_H',
                'date': current_date,
                'day': i,
                'a_price': a_price,
                'h_price': h_price,
                'spread': spread,
                'pnl': total_pnl,
                'entry_date': position_impr['entry_date'],
                'holding_days': (current_date - position_impr['entry_date']).days
            })
            
            position_impr = None
            ready_to_long_a_short_h = False
            up_cross_limit_impr = False
        
        # Record capital history
        capital_history_impr.append(capital_impr)
    
    # Calculate performance metrics
    test_dates = test_data.index
    
    # Original strategy
    capital_history_orig_test = capital_history_orig[-len(test_data):]
    returns_orig = np.diff(capital_history_orig_test) / capital_history_orig_test[:-1]
    annual_return_orig = np.mean(returns_orig) * 252 * 100
    
    peak_orig = np.maximum.accumulate(capital_history_orig_test)
    drawdown_orig = (capital_history_orig_test - peak_orig) / peak_orig * 100
    max_drawdown_orig = -np.min(drawdown_orig)
    
    sharpe_orig = np.sqrt(252) * np.mean(returns_orig) / np.std(returns_orig) if np.std(returns_orig) > 0 else 0
    total_return_orig = (capital_history_orig_test[-1] - initial_capital) / initial_capital * 100
    
    # Improved strategy
    capital_history_impr_test = capital_history_impr[-len(test_data):]
    returns_impr = np.diff(capital_history_impr_test) / capital_history_impr_test[:-1]
    annual_return_impr = np.mean(returns_impr) * 252 * 100
    
    peak_impr = np.maximum.accumulate(capital_history_impr_test)
    drawdown_impr = (capital_history_impr_test - peak_impr) / peak_impr * 100
    max_drawdown_impr = -np.min(drawdown_impr)
    
    sharpe_impr = np.sqrt(252) * np.mean(returns_impr) / np.std(returns_impr) if np.std(returns_impr) > 0 else 0
    total_return_impr = (capital_history_impr_test[-1] - initial_capital) / initial_capital * 100
    
    # Count completed trades
    completed_trades_orig = len([t for t in trades_orig if 'pnl' in t])
    completed_trades_impr = len([t for t in trades_impr if 'pnl' in t])
    
    # Print trade details
    print("\nOriginal Strategy Trades:")
    for i, trade in enumerate([t for t in trades_orig if 'pnl' in t]):
        print(f"Trade {i+1}: {trade['date'].date()} - {trade['type']} - PnL: ${trade['pnl']:.2f} - Duration: {trade['holding_days']} days")
    
    print("\nImproved Strategy Trades:")
    for i, trade in enumerate([t for t in trades_impr if 'pnl' in t]):
        print(f"Trade {i+1}: {trade['date'].date()} - {trade['type']} - PnL: ${trade['pnl']:.2f} - Duration: {trade['holding_days']} days")
    
    # Print performance summary
    print("\nPerformance Summary:")
    print("-" * 65)
    print(f"{'Metric':<25} {'Original Strategy':<20} {'Improved Strategy':<20}")
    print("-" * 65)
    print(f"{'Annual Return (%)':<25} {annual_return_orig:<20.2f} {annual_return_impr:<20.2f}")
    print(f"{'Max Drawdown (%)':<25} {max_drawdown_orig:<20.2f} {max_drawdown_impr:<20.2f}")
    print(f"{'Sharpe Ratio':<25} {sharpe_orig:<20.2f} {sharpe_impr:<20.2f}")
    print(f"{'Total Return (%)':<25} {total_return_orig:<20.2f} {total_return_impr:<20.2f}")
    print(f"{'Number of Trades':<25} {completed_trades_orig:<20} {completed_trades_impr:<20}")
    
    # Create visualization
    plt.figure(figsize=(15, 15))
    
    # Plot 1: Stock prices
    plt.subplot(4, 1, 1)
    plt.plot(data.index, data['a_price'], label='A Stock')
    plt.plot(data.index, data['h_price'], label='H Stock')
    plt.axvline(x=data.index[train_size], color='k', linestyle='--', label='Train/Test Split')
    plt.title('A-H Stock Prices')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    
    # Plot 2: Spread with thresholds
    plt.subplot(4, 1, 2)
    plt.plot(data.index, data['spread'], label='Spread')
    plt.axhline(y=mean_spread, color='k', linestyle='-', label='Mean')
    plt.axhline(y=upper_threshold, color='r', linestyle='--', label='Upper Threshold')
    plt.axhline(y=lower_threshold, color='b', linestyle='--', label='Lower Threshold')
    plt.axvline(x=data.index[train_size], color='k', linestyle='--')
    
    # Mark trade entry and exit points for original strategy
    for trade in trades_orig:
        if 'pnl' not in trade:  # Entry
            plt.scatter(trade['date'], trade['spread'], color='green', marker='^', s=100)
        else:  # Exit
            plt.scatter(trade['date'], trade['spread'], color='red', marker='v', s=100)
    
    plt.title('Price Spread with Trading Signals')
    plt.ylabel('Spread')
    plt.legend()
    plt.grid(True)
    
    # Plot 3: Equity curves
    plt.subplot(4, 1, 3)
    plt.plot(test_dates, capital_history_orig_test, label='Original Strategy', color='blue')
    plt.plot(test_dates, capital_history_impr_test, label='Improved Strategy', color='green')
    plt.title('Equity Curves')
    plt.ylabel('Capital ($)')
    plt.legend()
    plt.grid(True)
    
    # Plot 4: Drawdowns
    plt.subplot(4, 1, 4)
    plt.plot(test_dates, drawdown_orig, label='Original Strategy', color='blue')
    plt.plot(test_dates, drawdown_impr, label='Improved Strategy', color='green')
    plt.title('Drawdowns')
    plt.xlabel('Date')
    plt.ylabel('Drawdown (%)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return {
        'original': {
            'capital_history': capital_history_orig_test,
            'trades': trades_orig,
            'annual_return': annual_return_orig,
            'max_drawdown': max_drawdown_orig,
            'sharpe_ratio': sharpe_orig,
            'total_return': total_return_orig,
            'num_trades': completed_trades_orig
        },
        'improved': {
            'capital_history': capital_history_impr_test,
            'trades': trades_impr,
            'annual_return': annual_return_impr,
            'max_drawdown': max_drawdown_impr,
            'sharpe_ratio': sharpe_impr,
            'total_return': total_return_impr,
            'num_trades': completed_trades_impr
        }
    }

# Generate synthetic A-H stock pair data
print("Generating A-H stock pair data...")
ah_data, true_alpha, true_beta = generate_ah_pair(days=1000, seed=42)
print(f"True cointegration parameters: alpha={true_alpha:.4f}, beta={true_beta:.4f}")

# Run pairs trading backtest
print("\nRunning pairs trading backtest...")
backtest_results = pairs_trading_backtest(ah_data, train_pct=0.6)