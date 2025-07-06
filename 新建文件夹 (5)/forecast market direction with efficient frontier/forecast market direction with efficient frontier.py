import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from scipy import stats
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')

class EfficientFrontierCoefficients:
    """
    Class to calculate the efficient frontier coefficients based on asset returns and covariances.
    """
    def __init__(self):
        pass
    
    def calculate_coefficients(self, returns, covariance):
        """
        Calculate the efficient frontier coefficients (A, B, C) and interpretable coefficients (rMVP, σMVP, u).
        
        Parameters:
        -----------
        returns: array-like
            Expected returns for each asset
        covariance: array-like
            Covariance matrix of asset returns
        
        Returns:
        --------
        dict: Dictionary with calculated coefficients
        """
        # Calculate inverse of covariance matrix
        V_inv = np.linalg.inv(covariance)
        
        # Calculate ones vector
        e = np.ones(len(returns))
        
        # Calculate A, B, C coefficients
        A = e.T.dot(V_inv).dot(e)
        B = returns.T.dot(V_inv).dot(e)
        C = returns.T.dot(V_inv).dot(returns)
        
        # Calculate interpretable coefficients
        r_MVP = B / A
        sigma_MVP = 1 / np.sqrt(A)
        u = np.sqrt((A * C - B**2) / A)
        
        # Calculate function of cosine similarity for u interpretation
        dot_product = np.sum(returns)
        magnitude_r = np.sqrt(np.sum(returns**2))
        magnitude_e = np.sqrt(len(returns))
        cos_similarity = dot_product / (magnitude_r * magnitude_e) if magnitude_r > 0 else 0
        cosine_function = np.sqrt(1 - cos_similarity**2)
        
        # Calculate Mahalanobis distance part of u
        mahalanobis_part = np.sqrt(returns.T.dot(V_inv).dot(returns))
        
        return {
            'A': A,
            'B': B,
            'C': C,
            'r_MVP': r_MVP,
            'sigma_MVP': sigma_MVP,
            'u': u,
            'cosine_function': cosine_function,
            'mahalanobis_part': mahalanobis_part
        }

class MarketDirectionForecaster:
    """
    Class to forecast market direction using efficient frontier coefficients.
    """
    def __init__(self, max_depth_options=[1, 2]):
        self.max_depth_options = max_depth_options
        self.model = None
        self.best_max_depth = None
    
    def train(self, X, y, sample_weights=None):
        """
        Train the CART model with cross-validation to select the best max_depth.
        
        Parameters:
        -----------
        X: DataFrame
            Feature matrix with efficient frontier coefficients
        y: Series
            Binary target variable (market direction)
        sample_weights: array-like, optional
            Sample weights for CART training
        """
        # Set up time series cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        
        best_score = 0
        best_max_depth = None
        
        # Grid search for best max_depth
        for max_depth in self.max_depth_options:
            cv_scores = []
            
            for train_idx, val_idx in tscv.split(X):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                # Get sample weights for training set if provided
                train_weights = None
                if sample_weights is not None:
                    train_weights = sample_weights.iloc[train_idx]
                
                # Train model
                tree = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
                tree.fit(X_train, y_train, sample_weight=train_weights)
                
                # Evaluate on validation set
                y_val_pred = tree.predict(X_val)
                score = accuracy_score(y_val, y_val_pred)
                cv_scores.append(score)
            
            # Calculate average score across folds
            avg_score = np.mean(cv_scores)
            
            if avg_score > best_score:
                best_score = avg_score
                best_max_depth = max_depth
        
        # Train final model with best max_depth
        self.best_max_depth = best_max_depth
        self.model = DecisionTreeClassifier(max_depth=best_max_depth, random_state=42)
        self.model.fit(X, y, sample_weight=sample_weights)
        
        return self
    
    def predict_proba(self, X):
        """
        Predict the probability of the market going up.
        
        Parameters:
        -----------
        X: DataFrame
            Feature matrix with efficient frontier coefficients
        
        Returns:
        --------
        array: Probabilities of market going up
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        
        return self.model.predict_proba(X)[:, 1]
    
    def predict(self, X):
        """
        Predict the market direction (1 for up, 0 for down).
        
        Parameters:
        -----------
        X: DataFrame
            Feature matrix with efficient frontier coefficients
        
        Returns:
        --------
        array: Predicted market directions
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        
        return self.model.predict(X)
    
    def plot_tree(self, feature_names=None, class_names=None):
        """
        Plot the decision tree.
        
        Parameters:
        -----------
        feature_names: list, optional
            Names of the features
        class_names: list, optional
            Names of the classes
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        
        plt.figure(figsize=(12, 8))
        plot_tree(self.model, filled=True, feature_names=feature_names, 
                  class_names=class_names, rounded=True)
        plt.title(f"Decision Tree (Max Depth: {self.best_max_depth})")
        plt.show()

class PortfolioOptimizer:
    """
    Class to optimize portfolio weights based on market direction forecasts.
    """
    def __init__(self, risk_free_rate=0.02/12, max_leverage=1.5, transaction_fee=0.01):
        self.risk_free_rate = risk_free_rate
        self.max_leverage = max_leverage
        self.transaction_fee = transaction_fee
    
    def calculate_beta(self, asset_returns, market_returns):
        """
        Calculate beta coefficients for CAPM.
        
        Parameters:
        -----------
        asset_returns: DataFrame
            Historical returns of assets
        market_returns: Series
            Historical returns of market
        
        Returns:
        --------
        array: Beta coefficients for each asset
        """
        betas = []
        r_squared = []
        
        for column in asset_returns.columns:
            # Perform regression
            slope, intercept, r_value, p_value, std_err = stats.linregress(market_returns, asset_returns[column])
            betas.append(slope)
            r_squared.append(r_value**2)
        
        return np.array(betas), np.array(r_squared)
    
    def conditional_expected_return(self, mu_m, sigma_m, p_up, beta, r_squared):
        """
        Calculate conditional expected returns for assets given market forecast.
        
        Parameters:
        -----------
        mu_m: float
            Historical mean return of market
        sigma_m: float
            Historical standard deviation of market returns
        p_up: float
            Probability of market going up
        beta: array
            Beta coefficients for each asset
        r_squared: array
            R-squared values from CAPM regression
        
        Returns:
        --------
        array: Conditional expected returns for each asset
        """
        # Calculate standard normal density and CDF at -mu_m/sigma_m
        z = -mu_m / sigma_m
        phi_z = stats.norm.pdf(z)
        Phi_z = stats.norm.cdf(z)
        
        # Calculate conditional expected market return using inverse Mills ratio
        market_conditional_return = mu_m + sigma_m * (2*p_up - 1) * phi_z / (p_up - (2*p_up - 1) * Phi_z)
        
        # Calculate conditional expected returns for assets using CAPM
        conditional_returns = self.risk_free_rate + beta * (market_conditional_return - self.risk_free_rate)
        
        # Weight by R-squared to account for CAPM fit
        unconditional_returns = asset_returns.mean()
        weighted_returns = r_squared * conditional_returns + (1 - r_squared) * unconditional_returns
        
        return weighted_returns
    
    def optimize_tangency_portfolio(self, returns, covariance):
        """
        Optimize the tangency portfolio.
        
        Parameters:
        -----------
        returns: array-like
            Expected returns for each asset
        covariance: array-like
            Covariance matrix of asset returns
        
        Returns:
        --------
        array: Optimal portfolio weights
        """
        from scipy.optimize import minimize
        
        # Define objective function (negative Sharpe ratio)
        def objective(weights):
            portfolio_return = np.sum(weights * returns)
            portfolio_stddev = np.sqrt(weights.T @ covariance @ weights)
            return -(portfolio_return - self.risk_free_rate) / portfolio_stddev
        
        # Define constraints
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # Sum of weights = 1
            {'type': 'ineq', 'fun': lambda x: self.max_leverage - np.sum(np.abs(x))}  # Leverage constraint
        ]
        
        # Initial guess: equal weights
        n_assets = len(returns)
        initial_weights = np.ones(n_assets) / n_assets
        
        # Optimize
        result = minimize(objective, initial_weights, method='SLSQP', constraints=constraints)
        
        return result.x

def calculate_portfolio_metrics(returns):
    """
    Calculate portfolio performance metrics.
    
    Parameters:
    -----------
    returns: Series
        Portfolio returns
    
    Returns:
    --------
    dict: Dictionary with performance metrics
    """
    # Calculate annualized metrics (assuming monthly returns)
    annual_return = (1 + returns.mean()) ** 12 - 1
    annual_volatility = returns.std() * np.sqrt(12)
    sharpe_ratio = annual_return / annual_volatility if annual_volatility > 0 else 0
    
    # Calculate maximum drawdown
    cum_returns = (1 + returns).cumprod()
    running_max = cum_returns.cummax()
    drawdown = (cum_returns / running_max - 1)
    max_drawdown = drawdown.min()
    
    return {
        'Annual Return': annual_return,
        'Annual Volatility': annual_volatility,
        'Sharpe Ratio': sharpe_ratio,
        'Max Drawdown': max_drawdown
    }

# Load data
def get_data(start_date='1999-01-01', end_date='2022-12-31'):
    """
    Load market sector ETFs data.
    
    Parameters:
    -----------
    start_date: str
        Start date for data retrieval
    end_date: str
        End date for data retrieval
    
    Returns:
    --------
    tuple: (asset_prices, sp500_prices)
    """
    # Define sector ETFs
    sector_etfs = [
        'XLK',  # Technology
        'XLV',  # Health Care
        'XLF',  # Financials
        'XLE',  # Energy
        'XLB',  # Materials
        'XLY',  # Consumer Discretionary
        'XLI',  # Industrials
        'XLU',  # Utilities
        'XLP'   # Consumer Staples
    ]
    
    # Load data
    asset_prices = yf.download(sector_etfs, start=start_date, end=end_date)['Adj Close']
    sp500_prices = yf.download('^GSPC', start=start_date, end=end_date)['Adj Close']
    
    # Forward fill missing values
    asset_prices = asset_prices.ffill()
    sp500_prices = sp500_prices.ffill()
    
    return asset_prices, sp500_prices

# Simulate the strategy with historical data
def run_backtest(asset_prices, sp500_prices, train_start='1999-01-01', test_start='2008-01-01', test_end='2022-12-31'):
    """
    Run backtest of the proposed strategy.
    
    Parameters:
    -----------
    asset_prices: DataFrame
        Historical prices of assets
    sp500_prices: Series
        Historical prices of S&P 500
    train_start: str
        Start date for training data
    test_start: str
        Start date for testing data
    test_end: str
        End date for testing data
    
    Returns:
    --------
    DataFrame: Portfolio returns and performance metrics
    """
    # Convert prices to monthly data
    monthly_asset_prices = asset_prices.resample('M').last()
    monthly_sp500_prices = sp500_prices.resample('M').last()
    
    # Calculate monthly returns
    monthly_asset_returns = monthly_asset_prices.pct_change().dropna()
    monthly_sp500_returns = monthly_sp500_prices.pct_change().dropna()
    
    # Create masks for train and test periods
    train_mask = (monthly_asset_returns.index >= train_start) & (monthly_asset_returns.index < test_start)
    test_mask = (monthly_asset_returns.index >= test_start) & (monthly_asset_returns.index <= test_end)
    
    # Define strategies to test
    strategies = {
        'CART Tangency Portfolio': run_cart_strategy,
        'Monthly Tangency Portfolio': run_tangency_portfolio_strategy,
        'Equal Weighted Portfolio': run_equal_weighted_strategy,
        'S&P 500': run_sp500_strategy
    }
    
    # Run each strategy
    results = {}
    for name, strategy_fn in strategies.items():
        print(f"Running {name} strategy...")
        strategy_returns = strategy_fn(
            monthly_asset_returns, 
            monthly_sp500_returns, 
            train_mask, 
            test_mask
        )
        results[name] = strategy_returns
    
    # Calculate performance metrics
    metrics = {}
    for name, returns in results.items():
        metrics[name] = calculate_portfolio_metrics(returns)
    
    # Create results DataFrame
    returns_df = pd.DataFrame(results)
    metrics_df = pd.DataFrame(metrics).T
    
    return returns_df, metrics_df

def run_cart_strategy(asset_returns, market_returns, train_mask, test_mask):
    """
    Implement the CART-based market direction forecast strategy.
    
    Parameters:
    -----------
    asset_returns: DataFrame
        Monthly asset returns
    market_returns: Series
        Monthly market returns
    train_mask: boolean array
        Mask for training data
    test_mask: boolean array
        Mask for testing data
    
    Returns:
    --------
    Series: Portfolio returns
    """
    # Initialize classes
    ef_calculator = EfficientFrontierCoefficients()
    forecaster = MarketDirectionForecaster()
    optimizer = PortfolioOptimizer()
    
    # Initialize results
    portfolio_returns = pd.Series(index=asset_returns[test_mask].index, dtype=float)
    current_weights = np.zeros(len(asset_returns.columns))
    
    # Run online CART forecast and portfolio optimization
    test_dates = asset_returns[test_mask].index
    for i, current_date in enumerate(test_dates):
        print(f"Processing {current_date.strftime('%Y-%m')} ({i+1}/{len(test_dates)})", end='\r')
        
        # Define in-sample data up to current date
        in_sample_mask = asset_returns.index < current_date
        in_sample_asset_returns = asset_returns[in_sample_mask]
        in_sample_market_returns = market_returns[in_sample_mask]
        
        # Calculate efficient frontier coefficients for each month
        ef_coefficients = []
        market_directions = []
        
        for month in in_sample_asset_returns.index[1:]:  # Skip first month to have previous month's data
            prev_month = in_sample_asset_returns.index[in_sample_asset_returns.index < month][-1]
            
            # Calculate monthly returns and covariance
            window_returns = in_sample_asset_returns.loc[:prev_month].iloc[-12:].mean()  # Use past 12 months
            window_cov = in_sample_asset_returns.loc[:prev_month].iloc[-12:].cov()
            
            # Calculate efficient frontier coefficients
            coefs = ef_calculator.calculate_coefficients(window_returns, window_cov)
            ef_coefficients.append([coefs['r_MVP'], coefs['sigma_MVP'], coefs['u']])
            
            # Get market direction for current month
            current_return = in_sample_market_returns.loc[month]
            market_directions.append(1 if current_return > 0 else 0)
        
        # Convert to DataFrames
        X_train = pd.DataFrame(ef_coefficients, index=in_sample_asset_returns.index[1:], 
                              columns=['r_MVP', 'sigma_MVP', 'u'])
        y_train = pd.Series(market_directions, index=in_sample_asset_returns.index[1:])
        
        # Calculate sample weights (absolute market returns)
        sample_weights = in_sample_market_returns.loc[y_train.index].abs()
        
        # Train CART model
        forecaster.train(X_train, y_train, sample_weights)
        
        # Calculate current efficient frontier coefficients
        current_window_returns = in_sample_asset_returns.iloc[-12:].mean()
        current_window_cov = in_sample_asset_returns.iloc[-12:].cov()
        current_coefs = ef_calculator.calculate_coefficients(current_window_returns, current_window_cov)
        
        # Create feature for prediction
        X_pred = pd.DataFrame([[current_coefs['r_MVP'], current_coefs['sigma_MVP'], current_coefs['u']]], 
                             columns=['r_MVP', 'sigma_MVP', 'u'])
        
        # Forecast market direction
        market_up_prob = forecaster.predict_proba(X_pred)[0]
        
        # Discretize forecast (as mentioned in the paper)
        market_forecast = 1 if market_up_prob >= 0.5 else 0
        
        # Calculate betas and R-squared
        betas, r_squared = optimizer.calculate_beta(in_sample_asset_returns, in_sample_market_returns)
        
        # Calculate market statistics
        mu_m = in_sample_market_returns.mean()
        sigma_m = in_sample_market_returns.std()
        
        # Calculate conditional expected returns
        conditional_returns = optimizer.conditional_expected_return(
            mu_m, sigma_m, market_forecast, betas, r_squared)
        
        # Optimize portfolio weights
        optimal_weights = optimizer.optimize_tangency_portfolio(
            conditional_returns, current_window_cov)
        
        # Calculate transaction costs
        if i > 0:
            transaction_cost = optimizer.transaction_fee * np.sum(np.abs(optimal_weights - current_weights))
        else:
            transaction_cost = optimizer.transaction_fee * np.sum(np.abs(optimal_weights))
        
        # Update current weights
        current_weights = optimal_weights
        
        # Calculate portfolio return for current month
        if i < len(test_dates) - 1:
            next_date = test_dates[i+1]
            month_returns = asset_returns.loc[current_date:next_date].iloc[0]
            portfolio_return = np.sum(current_weights * month_returns) - transaction_cost
            portfolio_returns.loc[current_date] = portfolio_return
    
    print("\nStrategy execution completed.")
    return portfolio_returns

def run_tangency_portfolio_strategy(asset_returns, market_returns, train_mask, test_mask):
    """
    Implement the standard tangency portfolio strategy.
    
    Parameters:
    -----------
    asset_returns: DataFrame
        Monthly asset returns
    market_returns: Series
        Monthly market returns
    train_mask: boolean array
        Mask for training data
    test_mask: boolean array
        Mask for testing data
    
    Returns:
    --------
    Series: Portfolio returns
    """
    optimizer = PortfolioOptimizer()
    
    # Initialize results
    portfolio_returns = pd.Series(index=asset_returns[test_mask].index, dtype=float)
    current_weights = np.zeros(len(asset_returns.columns))
    
    # Run monthly tangency portfolio optimization
    test_dates = asset_returns[test_mask].index
    for i, current_date in enumerate(test_dates):
        print(f"Processing {current_date.strftime('%Y-%m')} ({i+1}/{len(test_dates)})", end='\r')
        
        # Define in-sample data up to current date
        in_sample_mask = asset_returns.index < current_date
        in_sample_asset_returns = asset_returns[in_sample_mask]
        
        # Calculate expected returns and covariance
        expected_returns = in_sample_asset_returns.iloc[-12:].mean()
        covariance = in_sample_asset_returns.iloc[-12:].cov()
        
        # Optimize portfolio weights
        optimal_weights = optimizer.optimize_tangency_portfolio(expected_returns, covariance)
        
        # Calculate transaction costs
        if i > 0:
            transaction_cost = optimizer.transaction_fee * np.sum(np.abs(optimal_weights - current_weights))
        else:
            transaction_cost = optimizer.transaction_fee * np.sum(np.abs(optimal_weights))
        
        # Update current weights
        current_weights = optimal_weights
        
        # Calculate portfolio return for current month
        if i < len(test_dates) - 1:
            next_date = test_dates[i+1]
            month_returns = asset_returns.loc[current_date:next_date].iloc[0]
            portfolio_return = np.sum(current_weights * month_returns) - transaction_cost
            portfolio_returns.loc[current_date] = portfolio_return
    
    print("\nStrategy execution completed.")
    return portfolio_returns

def run_equal_weighted_strategy(asset_returns, market_returns, train_mask, test_mask):
    """
    Implement the equal-weighted portfolio strategy.
    
    Parameters:
    -----------
    asset_returns: DataFrame
        Monthly asset returns
    market_returns: Series
        Monthly market returns
    train_mask: boolean array
        Mask for training data
    test_mask: boolean array
        Mask for testing data
    
    Returns:
    --------
    Series: Portfolio returns
    """
    # Initialize results
    portfolio_returns = pd.Series(index=asset_returns[test_mask].index, dtype=float)
    
    # Calculate equal weights
    n_assets = len(asset_returns.columns)
    equal_weights = np.ones(n_assets) / n_assets
    
    # Calculate portfolio returns
    test_dates = asset_returns[test_mask].index
    for i, current_date in enumerate(test_dates):
        if i < len(test_dates) - 1:
            next_date = test_dates[i+1]
            month_returns = asset_returns.loc[current_date:next_date].iloc[0]
            portfolio_return = np.sum(equal_weights * month_returns)
            portfolio_returns.loc[current_date] = portfolio_return
    
    return portfolio_returns

def run_sp500_strategy(asset_returns, market_returns, train_mask, test_mask):
    """
    Return S&P 500 returns.
    
    Parameters:
    -----------
    asset_returns: DataFrame
        Monthly asset returns
    market_returns: Series
        Monthly market returns
    train_mask: boolean array
        Mask for training data
    test_mask: boolean array
        Mask for testing data
    
    Returns:
    --------
    Series: S&P 500 returns
    """
    return market_returns[test_mask]

def plot_results(returns_df, metrics_df):
    """
    Plot cumulative returns and metrics.
    
    Parameters:
    -----------
    returns_df: DataFrame
        Strategy returns
    metrics_df: DataFrame
        Performance metrics
    """
    # Plot cumulative returns
    plt.figure(figsize=(12, 6))
    (1 + returns_df).cumprod().plot()
    plt.title('Cumulative Returns')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Plot Sharpe ratios
    plt.figure(figsize=(10, 5))
    metrics_df['Sharpe Ratio'].sort_values().plot(kind='bar')
    plt.title('Sharpe Ratios')
    plt.ylabel('Sharpe Ratio')
    plt.grid(True, axis='y')
    plt.show()
    
    # Plot annual returns vs. volatility
    plt.figure(figsize=(10, 6))
    plt.scatter(metrics_df['Annual Volatility'], metrics_df['Annual Return'], s=100)
    
    # Add labels to each point
    for i, label in enumerate(metrics_df.index):
        plt.annotate(label, 
                    (metrics_df['Annual Volatility'].iloc[i], metrics_df['Annual Return'].iloc[i]),
                    textcoords="offset points", 
                    xytext=(0, 10), 
                    ha='center')
    
    plt.title('Risk-Return Tradeoff')
    plt.xlabel('Annual Volatility')
    plt.ylabel('Annual Return')
    plt.grid(True)
    plt.show()

# Define function to simulate data and test strategy
def simulate_and_test_strategy():
    """
    Simulate market data and test the strategy.
    """
    # Simulate asset returns with correlations
    np.random.seed(42)
    num_assets = 9  # Same as sector ETFs
    num_months = 240  # 20 years of monthly data
    
    # Create correlation matrix (positive correlations with some variation)
    correlation = np.random.uniform(0.3, 0.7, size=(num_assets, num_assets))
    correlation = (correlation + correlation.T) / 2  # Make symmetric
    np.fill_diagonal(correlation, 1)  # Diagonal is 1
    
    # Convert correlation to covariance
    volatilities = np.random.uniform(0.03, 0.06, size=num_assets)
    covariance = np.outer(volatilities, volatilities) * correlation
    
    # Simulate monthly returns
    mean_returns = np.random.uniform(0.003, 0.008, size=num_assets)  # Monthly returns between 0.3% and 0.8%
    asset_returns_data = np.random.multivariate_normal(mean_returns, covariance, size=num_months)
    
    # Simulate market returns (correlated with asset returns)
    market_beta = np.random.uniform(0.8, 1.2, size=num_assets)
    market_returns_data = np.dot(asset_returns_data, market_beta) / np.sum(market_beta) + np.random.normal(0, 0.01, num_months)
    
    # Create DataFrames
    dates = pd.date_range(start='2002-01-01', periods=num_months, freq='M')
    asset_returns = pd.DataFrame(asset_returns_data, index=dates, columns=[f'Asset_{i+1}' for i in range(num_assets)])
    market_returns = pd.Series(market_returns_data, index=dates, name='Market')
    
    # Define train and test periods
    train_mask = dates < '2012-01-01'
    test_mask = dates >= '2012-01-01'
    
    # Run the strategies
    print("Running strategies on simulated data...")
    strategies = {
        'CART Tangency Portfolio': run_cart_strategy,
        'Monthly Tangency Portfolio': run_tangency_portfolio_strategy,
        'Equal Weighted Portfolio': run_equal_weighted_strategy,
        'Market': lambda a, m, t1, t2: market_returns[test_mask]
    }
    
    # Run each strategy
    results = {}
    for name, strategy_fn in strategies.items():
        print(f"Running {name} strategy...")
        strategy_returns = strategy_fn(
            asset_returns, 
            market_returns, 
            train_mask, 
            test_mask
        )
        results[name] = strategy_returns
    
    # Calculate performance metrics
    metrics = {}
    for name, returns in results.items():
        metrics[name] = calculate_portfolio_metrics(returns)
    
    # Create results DataFrame
    returns_df = pd.DataFrame(results)
    metrics_df = pd.DataFrame(metrics).T
    
    # Plot results
    plot_results(returns_df, metrics_df)
    
    return returns_df, metrics_df

# Main execution
if __name__ == "__main__":
    print("Starting simulation...")
    
    # Option 1: Test with simulated data (faster and doesn't require internet)
    returns_df, metrics_df = simulate_and_test_strategy()
    
    # Option 2: Uncomment to test with real data (requires internet)
    # try:
    #     print("Loading sector ETF data...")
    #     asset_prices, sp500_prices = get_data()
    #     print("Running backtest...")
    #     returns_df, metrics_df = run_backtest(asset_prices, sp500_prices)
    # except Exception as e:
    #     print(f"Error with real data: {e}")
    #     print("Falling back to simulated data...")
    #     returns_df, metrics_df = simulate_and_test_strategy()
    
    print("\nPerformance Metrics:")
    print(metrics_df)