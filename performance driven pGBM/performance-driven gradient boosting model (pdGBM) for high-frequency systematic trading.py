import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from datetime import datetime, timedelta
import random
from tqdm import tqdm

class PerformanceDrivenGBM:
    def __init__(self, n_estimators=30, learning_rate=0.1, max_depth=1, 
                 transaction_cost=0.02, alpha=0.01, h=100, loss_function="pnl"):
        """
        Initialize the pdGBM model
        
        Parameters:
        -----------
        n_estimators : int
            Number of boosting iterations
        learning_rate : float
            Learning rate for gradient boosting
        max_depth : int
            Max depth of decision tree (1 for decision stump)
        transaction_cost : float
            Transaction cost per trade
        alpha : float
            Significance threshold for trading
        h : float
            Smoothing parameter for gradient approximation
        loss_function : str
            Type of loss function: "pnl" or "sharpe"
        """
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.transaction_cost = transaction_cost
        self.alpha = alpha
        self.h = h
        self.loss_function = loss_function
        self.estimators = []
        self.F0 = None

    def _calculate_gradient_pnl(self, y, F, alpha=0.01, transaction_cost=0.02):
        """
        Calculate the negative gradient of the PnL loss function
        
        Parameters:
        -----------
        y : array-like
            Target values
        F : array-like
            Current predictions
        """
        # Use logistic function to smooth the indicator function
        def K(x, h=self.h):
            return np.exp(h*x) / (1 + np.exp(h*x))
        
        # Approximate gradient of PnL loss function
        # -[I(F > alpha)*(y - c) + I(F < -alpha)*(-y - c)]
        gradient = -(K(F - alpha) * (y - transaction_cost) - 
                    K(-alpha - F) * (y + transaction_cost))
        
        return gradient
    
    def _calculate_gradient_sharpe(self, y, F, alpha=0.01, transaction_cost=0.02):
        """
        Calculate the negative gradient of the Sharpe ratio loss function
        
        Parameters:
        -----------
        y : array-like
            Target values
        F : array-like
            Current predictions
        """
        # This is a simplified version of the Sharpe ratio gradient
        # A more accurate version would require calculating the gradient 
        # of the full Sharpe ratio formula
        def K(x, h=self.h):
            return np.exp(h*x) / (1 + np.exp(h*x))
        
        # Calculate PnL
        pnl = np.zeros_like(y)
        long_idx = F > self.alpha
        short_idx = F < -self.alpha
        pnl[long_idx] = y[long_idx] - transaction_cost
        pnl[short_idx] = -y[short_idx] - transaction_cost
        
        # Calculate mean and std of PnL
        mean_pnl = np.mean(pnl)
        std_pnl = np.std(pnl)
        
        # Approximate gradient of Sharpe ratio
        gradient = np.zeros_like(y)
        
        # Contribution to mean
        mean_grad_long = K(F - alpha) * (1/len(y)) * (1/std_pnl)
        mean_grad_short = K(-alpha - F) * (1/len(y)) * (1/std_pnl)
        
        # Contribution to std
        std_grad_long = K(F - alpha) * (1/len(y)) * (-mean_pnl/std_pnl**3) * (y - transaction_cost - mean_pnl)
        std_grad_short = K(-alpha - F) * (1/len(y)) * (-mean_pnl/std_pnl**3) * (-y - transaction_cost - mean_pnl)
        
        gradient = -(mean_grad_long + mean_grad_short + std_grad_long + std_grad_short)
        
        return gradient
    
    def fit(self, X, y):
        """
        Fit the pdGBM model
        
        Parameters:
        -----------
        X : array-like
            Feature matrix
        y : array-like
            Target values
        """
        # Initialize prediction with mean of target
        self.F0 = np.mean(y)
        F = np.ones(len(y)) * self.F0
        
        # Boosting iterations
        for m in range(self.n_estimators):
            # Calculate negative gradient
            if self.loss_function == "pnl":
                gradient = self._calculate_gradient_pnl(y, F, self.alpha, self.transaction_cost)
            else:  # sharpe
                gradient = self._calculate_gradient_sharpe(y, F, self.alpha, self.transaction_cost)
            
            # Fit base learner (decision stump)
            tree = DecisionTreeRegressor(max_depth=self.max_depth, random_state=42)
            tree.fit(X, gradient)
            
            # Update prediction
            update = tree.predict(X)
            F += self.learning_rate * update
            
            # Store the estimator
            self.estimators.append(tree)
        
        return self
    
    def predict(self, X):
        """
        Make predictions with the pdGBM model
        
        Parameters:
        -----------
        X : array-like
            Feature matrix
        
        Returns:
        --------
        array-like
            Predictions
        """
        # Start with initial prediction
        F = np.ones(len(X)) * self.F0
        
        # Add contribution from each estimator
        for m, estimator in enumerate(self.estimators):
            update = estimator.predict(X)
            F += self.learning_rate * update
        
        return F

def generate_ohlc_data(n_samples=10000, frequency='1s', volatility=0.0001, trend=0.00001, 
                      mean_reversion=0.1, jump_prob=0.001, jump_size_range=(0.001, 0.003)):
    """
    Generate simulated OHLC price data with realistic features
    
    Parameters:
    -----------
    n_samples : int
        Number of price bars to generate
    frequency : str
        Time frequency of the data
    volatility : float
        Base volatility of price changes
    trend : float
        Strength of the trend component
    mean_reversion : float
        Strength of mean reversion
    jump_prob : float
        Probability of price jumps
    jump_size_range : tuple
        Range for the size of jumps
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with OHLC price data
    """
    # Initialize price at 100
    base_price = 100
    
    # Create timestamp index
    now = pd.Timestamp.now().floor('1s')
    timestamps = [now + pd.Timedelta(frequency) * i for i in range(n_samples)]
    
    prices = []
    current_price = base_price
    trend_component = 0
    
    for i in range(n_samples):
        # Add trend component with some randomness
        trend_component = trend_component * 0.98 + np.random.normal(0, 0.0001)
        trend_effect = trend * current_price + trend_component
        
        # Mean reversion component
        reversion = mean_reversion * (base_price - current_price) / base_price
        
        # Random component (volatility)
        random_change = np.random.normal(0, volatility * current_price)
        
        # Jump component
        jump = 0
        if np.random.random() < jump_prob:
            jump_size = np.random.uniform(*jump_size_range) * current_price
            jump = jump_size * np.random.choice([-1, 1])
        
        # Combine all components
        price_change = trend_effect + reversion * current_price + random_change + jump
        current_price += price_change
        
        # Ensure price is positive
        current_price = max(current_price, 0.01)
        
        # Generate OHLC for this bar
        open_price = current_price
        high_low_range = np.random.uniform(0.0002, 0.0008) * current_price
        high_extra = np.random.uniform(0, high_low_range)
        low_extra = np.random.uniform(0, high_low_range)
        
        high_price = current_price + high_extra
        low_price = max(0.01, current_price - low_extra)
        close_price = np.random.uniform(low_price, high_price)
        current_price = close_price  # Update current price for next iteration
        
        prices.append([open_price, high_price, low_price, close_price])
    
    # Create DataFrame
    df = pd.DataFrame(prices, columns=['Open', 'High', 'Low', 'Close'], index=timestamps)
    df.index.name = 'Timestamp'
    
    return df

def calculate_technical_indicators(df):
    """
    Calculate technical indicators for the given OHLC data
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with OHLC price data
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with technical indicators
    """
    # Create copy of the dataframe
    result = df.copy()
    
    # RSI with different periods
    for period in [5, 7, 10, 14, 21]:
        # Calculate price changes
        delta = result['Close'].diff()
        
        # Calculate gains and losses
        gains = delta.copy()
        gains[gains < 0] = 0
        losses = -delta.copy()
        losses[losses < 0] = 0
        
        # Calculate average gains and losses
        avg_gains = gains.rolling(window=period).mean()
        avg_losses = losses.rolling(window=period).mean()
        
        # Calculate RS and RSI
        rs = avg_gains / avg_losses
        rsi = 100 - (100 / (1 + rs))
        
        result[f'RSI_{period}'] = rsi
    
    # Bollinger Bands with different periods
    for period in [10, 15, 20, 30, 50]:
        # Calculate simple moving average
        sma = result['Close'].rolling(window=period).mean()
        
        # Calculate standard deviation
        std = result['Close'].rolling(window=period).std()
        
        # Calculate upper and lower bands
        upper_band = sma + 2 * std
        lower_band = sma - 2 * std
        
        # Calculate Bollinger Bands indicator
        result[f'BBands_{period}'] = (result['Close'] - lower_band) / (upper_band - lower_band)
    
    # MACD with different periods
    for fast_period, slow_period in [(6, 13), (12, 26), (8, 17), (10, 20)]:
        # Calculate fast and slow EMAs
        ema_fast = result['Close'].ewm(span=fast_period, adjust=False).mean()
        ema_slow = result['Close'].ewm(span=slow_period, adjust=False).mean()
        
        # Calculate MACD line
        macd_line = ema_fast - ema_slow
        
        # Calculate signal line (9-period EMA of MACD line)
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        
        # Calculate MACD histogram
        result[f'MACD_{fast_period}_{slow_period}_9'] = macd_line - signal_line
    
    # Drop rows with NaN values
    result = result.dropna()
    
    return result

def generate_target_variable(df, horizon=10):
    """
    Generate target variable (future price change)
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with price data
    horizon : int
        Number of periods ahead for prediction
    
    Returns:
    --------
    pd.Series
        Target variable (future price change)
    """
    # Calculate future price change
    future_price = df['Close'].shift(-horizon)
    price_change = future_price - df['Close']
    
    return price_change

def run_trading_simulation(model, X_test, y_test, transaction_cost=0.02, alpha=0.01):
    """
    Run a trading simulation using the trained model
    
    Parameters:
    -----------
    model : pdGBM
        Trained pdGBM model
    X_test : array-like
        Test features
    y_test : array-like
        Test targets
    transaction_cost : float
        Transaction cost per trade
    alpha : float
        Significance threshold for trading
    
    Returns:
    --------
    dict
        Dictionary with trading results
    """
    # Make predictions
    predictions = model.predict(X_test)
    
    # Initialize results
    trades = []
    pnl = []
    positions = np.zeros(len(y_test))
    
    # Simulate trading
    for i in range(len(y_test)):
        if predictions[i] > alpha:  # Buy signal
            positions[i] = 1
            trades.append(('buy', i, y_test[i] - transaction_cost))
            pnl.append(y_test[i] - transaction_cost)
        elif predictions[i] < -alpha:  # Sell signal
            positions[i] = -1
            trades.append(('sell', i, -y_test[i] - transaction_cost))
            pnl.append(-y_test[i] - transaction_cost)
        else:  # No trade
            pnl.append(0)
    
    # Calculate performance metrics
    total_trades = np.sum(positions != 0)
    win_trades = np.sum(np.array(pnl) > 0)
    if total_trades > 0:
        win_ratio = win_trades / total_trades
    else:
        win_ratio = 0
    
    total_pnl = np.sum(pnl)
    if total_trades > 0:
        avg_pnl_per_trade = total_pnl / total_trades
    else:
        avg_pnl_per_trade = 0
    
    # Calculate Sharpe ratio (assuming daily)
    if len(pnl) > 0 and np.std(pnl) > 0:
        sharpe_ratio = np.sqrt(252) * np.mean(pnl) / np.std(pnl)
    else:
        sharpe_ratio = 0
    
    # Return results
    results = {
        'total_trades': total_trades,
        'win_ratio': win_ratio,
        'total_pnl': total_pnl,
        'avg_pnl_per_trade': avg_pnl_per_trade,
        'sharpe_ratio': sharpe_ratio,
        'trades': trades,
        'pnl': pnl,
        'positions': positions,
        'predictions': predictions
    }
    
    return results

def plot_trading_results(df, results, title="Trading Results"):
    """
    Plot trading results
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with price data
    results : dict
        Dictionary with trading results
    title : str
        Plot title
    """
    # Extract data
    prices = df['Close'].values
    positions = results['positions']
    cumulative_pnl = np.cumsum(results['pnl'])
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Plot prices and trade points
    ax1.plot(prices, color='gray', alpha=0.5, label='Price')
    
    # Find buy and sell points
    buy_points = np.where(positions == 1)[0]
    sell_points = np.where(positions == -1)[0]
    
    # Plot buy and sell points
    if len(buy_points) > 0:
        ax1.scatter(buy_points, prices[buy_points], color='green', marker='^', label='Buy')
    if len(sell_points) > 0:
        ax1.scatter(sell_points, prices[sell_points], color='red', marker='v', label='Sell')
    
    ax1.set_ylabel('Price')
    ax1.set_title(title)
    ax1.legend()
    ax1.grid(True)
    
    # Plot cumulative PnL
    ax2.plot(cumulative_pnl, color='blue', label='Cumulative PnL')
    ax2.set_ylabel('Cumulative PnL')
    ax2.set_xlabel('Time')
    ax2.legend()
    ax2.grid(True)
    
    # Add performance metrics as text
    performance_text = f"Total Trades: {results['total_trades']}\n"
    performance_text += f"Win Ratio: {results['win_ratio']:.2f}\n"
    performance_text += f"Total PnL: {results['total_pnl']:.4f}\n"
    performance_text += f"Avg PnL/Trade: {results['avg_pnl_per_trade']:.4f}\n"
    performance_text += f"Sharpe Ratio: {results['sharpe_ratio']:.2f}"
    
    plt.figtext(0.15, 0.01, performance_text, fontsize=10, 
                bbox=dict(facecolor='white', alpha=0.5))
    
    plt.tight_layout()
    plt.show()

# Main execution
if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate simulated data
    print("Generating simulated price data...")
    ohlc_data = generate_ohlc_data(n_samples=10000, frequency='1s')
    
    # Calculate technical indicators
    print("Calculating technical indicators...")
    data_with_indicators = calculate_technical_indicators(ohlc_data)
    
    # Generate target variable (10-second forward price change)
    print("Generating target variable...")
    data_with_indicators['target'] = generate_target_variable(data_with_indicators, horizon=10)
    data_with_indicators = data_with_indicators.dropna()
    
    # Split data into training and testing sets (70% train, 30% test)
    train_size = int(len(data_with_indicators) * 0.7)
    
    # Prepare features and target
    feature_columns = [col for col in data_with_indicators.columns 
                      if col.startswith(('RSI', 'BBands', 'MACD'))]
    
    X = data_with_indicators[feature_columns].values
    y = data_with_indicators['target'].values
    
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Train pdGBM model
    print("Training pdGBM model...")
    transaction_cost = 0.0002  # 0.02% of price
    alpha = 0.0001  # Significance threshold for trading
    
    pdgbm = PerformanceDrivenGBM(n_estimators=30, learning_rate=0.1, max_depth=1,
                                transaction_cost=transaction_cost, alpha=alpha, 
                                loss_function="pnl")
    
    pdgbm.fit(X_train, y_train)
    
    # Run trading simulation
    print("Running trading simulation...")
    trading_results = run_trading_simulation(pdgbm, X_test, y_test, 
                                           transaction_cost=transaction_cost, 
                                           alpha=alpha)
    
    # Print results
    print("\nTrading Results:")
    print(f"Total trades: {trading_results['total_trades']}")
    print(f"Win ratio: {trading_results['win_ratio']:.2f}")
    print(f"Total PnL: {trading_results['total_pnl']:.4f}")
    print(f"Average PnL per trade: {trading_results['avg_pnl_per_trade']:.4f}")
    print(f"Sharpe ratio: {trading_results['sharpe_ratio']:.2f}")
    
    # Plot results
    print("\nPlotting results...")
    plot_trading_results(data_with_indicators.iloc[train_size:].reset_index(drop=True), 
                        trading_results, 
                        title="pdGBM Trading Simulation Results")
    
    # Compare with benchmark models
    print("\nRunning comparison with benchmark models...")
    
    # 1. Standard GBM with L2 loss (using scikit-learn's GradientBoostingRegressor)
    from sklearn.ensemble import GradientBoostingRegressor
    
    gbm = GradientBoostingRegressor(n_estimators=30, learning_rate=0.1, max_depth=1, random_state=42)
    gbm.fit(X_train, y_train)
    
    gbm_results = run_trading_simulation(gbm, X_test, y_test,
                                       transaction_cost=transaction_cost,
                                       alpha=alpha)
    
    # 2. Linear model (LASSO)
    from sklearn.linear_model import Lasso
    
    lasso = Lasso(alpha=0.01, random_state=42)
    lasso.fit(X_train, y_train)
    
    # Create a compatible predict method
    lasso.predict_orig = lasso.predict
    lasso.predict = lambda X: lasso.predict_orig(X)
    
    lasso_results = run_trading_simulation(lasso, X_test, y_test,
                                         transaction_cost=transaction_cost,
                                         alpha=alpha)
    
    # Print comparison with simplified formatting
    print("\nModel Comparison:")
    print("Model      Total Trades    Win Ratio    Total PnL    Avg PnL/Trade    Sharpe Ratio")
    print("-" * 85)
    print(f"pdGBM      {trading_results['total_trades']}             {trading_results['win_ratio']:.2f}         {trading_results['total_pnl']:.4f}     {trading_results['avg_pnl_per_trade']:.4f}           {trading_results['sharpe_ratio']:.2f}")
    print(f"GBM L2     {gbm_results['total_trades']}             {gbm_results['win_ratio']:.2f}         {gbm_results['total_pnl']:.4f}     {gbm_results['avg_pnl_per_trade']:.4f}           {gbm_results['sharpe_ratio']:.2f}")
    print(f"LASSO      {lasso_results['total_trades']}             {lasso_results['win_ratio']:.2f}         {lasso_results['total_pnl']:.4f}     {lasso_results['avg_pnl_per_trade']:.4f}           {lasso_results['sharpe_ratio']:.2f}")
    
    # Test with different holding periods
    print("\nTesting different holding periods...")
    holding_periods = [10, 30, 60, 120, 300]
    holding_results = {}
    
    for period in holding_periods:
        print(f"Testing {period}-second holding period...")
        # Generate target variable for this holding period
        data_with_indicators[f'target_{period}'] = generate_target_variable(data_with_indicators, horizon=period)
        data_clean = data_with_indicators.dropna()
        
        # Split data
        train_size = int(len(data_clean) * 0.7)
        y_period = data_clean[f'target_{period}'].values
        y_train_period, y_test_period = y_period[:train_size], y_period[train_size:]
        
        # Training features (use same X as before but match length)
        X_train_period = data_clean[feature_columns].values[:train_size]
        X_test_period = data_clean[feature_columns].values[train_size:]
        
        # Train model
        pdgbm_period = PerformanceDrivenGBM(n_estimators=30, learning_rate=0.1, max_depth=1,
                                          transaction_cost=transaction_cost, alpha=alpha,
                                          loss_function="pnl")
        
        pdgbm_period.fit(X_train_period, y_train_period)
        
        # Run simulation
        period_results = run_trading_simulation(pdgbm_period, X_test_period, y_test_period,
                                              transaction_cost=transaction_cost,
                                              alpha=alpha)
        
        holding_results[period] = period_results
    
    # Print holding period comparison with simplified formatting
    print("\nHolding Period Comparison:")
    print("Period(s)  Total Trades    Win Ratio    Total PnL    Avg PnL/Trade    Sharpe Ratio")
    print("-" * 85)
    for period, results in holding_results.items():
        print(f"{period}         {results['total_trades']}             {results['win_ratio']:.2f}         {results['total_pnl']:.4f}     {results['avg_pnl_per_trade']:.4f}           {results['sharpe_ratio']:.2f}")
    
    # Test with different transaction costs
    print("\nTesting different transaction costs...")
    costs = [0.0001, 0.0002, 0.0003, 0.0004]
    cost_results = {}
    
    for cost in costs:
        print(f"Testing {cost:.4f} transaction cost...")
        # Train model with this cost
        pdgbm_cost = PerformanceDrivenGBM(n_estimators=30, learning_rate=0.1, max_depth=1,
                                        transaction_cost=cost, alpha=alpha,
                                        loss_function="pnl")
        
        pdgbm_cost.fit(X_train, y_train)
        
        # Run simulation
        cost_result = run_trading_simulation(pdgbm_cost, X_test, y_test,
                                           transaction_cost=cost,
                                           alpha=alpha)
        
        cost_results[cost] = cost_result
    
    # Print transaction cost comparison with simplified formatting
    print("\nTransaction Cost Comparison:")
    print("Cost       Total Trades    Win Ratio    Total PnL    Avg PnL/Trade    Sharpe Ratio")
    print("-" * 85)
    for cost, results in cost_results.items():
        print(f"{cost:.4f}     {results['total_trades']}             {results['win_ratio']:.2f}         {results['total_pnl']:.4f}     {results['avg_pnl_per_trade']:.4f}           {results['sharpe_ratio']:.2f}")