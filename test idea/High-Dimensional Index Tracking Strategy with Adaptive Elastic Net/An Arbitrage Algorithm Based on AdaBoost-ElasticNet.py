import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from math import sqrt
import warnings
warnings.filterwarnings('ignore')

class AdaBoostElasticNetArbitrage:
    """
    Implementation of the AdaBoost-ElasticNet Arbitrage algorithm as described in the paper.
    This algorithm combines K-means clustering to break industry barriers in stock selection
    and AdaBoost-enhanced ElasticNet to identify arbitrage opportunities among multiple stocks.
    """
    
    def __init__(self, n_clusters=42, pca_variance_ratio=0.85, n_boost_iterations=10, 
                 alpha=0.5, l1_ratio=0.5, random_state=42):
        """
        Initialize the AdaBoost-ElasticNet Arbitrage model
        
        Parameters:
        -----------
        n_clusters : int
            Number of clusters for K-means++
        pca_variance_ratio : float
            Ratio of variance to preserve in PCA
        n_boost_iterations : int
            Number of boosting iterations in AdaBoost
        alpha : float
            Constant that multiplies the penalty terms in ElasticNet
        l1_ratio : float
            The ElasticNet mixing parameter (0 <= l1_ratio <= 1)
        random_state : int
            Random seed for reproducibility
        """
        self.n_clusters = n_clusters
        self.pca_variance_ratio = pca_variance_ratio
        self.n_boost_iterations = n_boost_iterations
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.random_state = random_state
        
        # Initialize components
        self.pca = None
        self.kmeans = None
        self.target_stock = None
        self.clusters = None
        self.feature_stocks = None
        self.adaboost_models = None
        self.sample_weights = None
        self.model_weights = None
        
    def fit(self, financial_data, price_data, train_ratio=0.7):
        """
        Fit the model to the data
        
        Parameters:
        -----------
        financial_data : pd.DataFrame
            Financial factors (features) of stocks with stock codes as the index
        price_data : pd.DataFrame
            Historical price data with dates as the index and stock codes as columns
        train_ratio : float
            Ratio of data to use for training (0.0 to 1.0)
        """
        # Step 1: Standardize financial data
        scaler = StandardScaler()
        financial_data_scaled = pd.DataFrame(
            scaler.fit_transform(financial_data),
            index=financial_data.index,
            columns=financial_data.columns
        )
        
        # Step 2: Apply PCA to reduce dimensions
        self.pca = PCA(random_state=self.random_state)
        reduced_financial_data = self.pca.fit_transform(financial_data_scaled)
        
        # Find the number of components that explain the desired variance
        cumulative_variance = np.cumsum(self.pca.explained_variance_ratio_)
        n_components = np.argmax(cumulative_variance >= self.pca_variance_ratio) + 1
        
        # Refit PCA with the determined number of components
        self.pca = PCA(n_components=n_components, random_state=self.random_state)
        reduced_financial_data = self.pca.fit_transform(financial_data_scaled)
        
        print(f"Reduced dimensions from {financial_data.shape[1]} to {reduced_financial_data.shape[1]}")
        
        # Step 3: Apply K-means++ clustering
        self.kmeans = KMeans(n_clusters=min(self.n_clusters, len(financial_data)), 
                            init='k-means++', 
                            random_state=self.random_state)
        clusters = self.kmeans.fit_predict(reduced_financial_data)
        
        # Create a mapping from stock code to cluster
        self.clusters = pd.Series(clusters, index=financial_data.index)
        
        # Get cluster sizes
        cluster_sizes = self.clusters.value_counts()
        print(f"Cluster sizes: Min={cluster_sizes.min()}, Max={cluster_sizes.max()}, Median={cluster_sizes.median()}")
        
        # Step 4: Split price data into training and testing
        train_size = int(len(price_data) * train_ratio)
        train_data = price_data.iloc[:train_size]
        
        # Calculate correlation matrix for all stocks
        correlation_matrix = train_data.corr()
        
        # Find the best cluster for our purpose
        best_cluster = None
        best_target = None
        best_features = []
        best_corr_count = 0
        
        # Try different correlation thresholds
        for threshold in [0.8, 0.7, 0.6, 0.5]:
            print(f"Trying correlation threshold: {threshold}")
            
            # Try each cluster
            for cluster_id in cluster_sizes.index:
                stocks_in_cluster = self.clusters[self.clusters == cluster_id].index
                stocks_in_cluster = [s for s in stocks_in_cluster if s in price_data.columns]
                
                if len(stocks_in_cluster) < 3:
                    continue  # Skip clusters with too few stocks
                
                # Check each stock in the cluster as a potential target
                for potential_target in stocks_in_cluster:
                    # Get correlations with other stocks in the cluster
                    corr_with_target = correlation_matrix[potential_target].loc[stocks_in_cluster]
                    
                    # Find stocks with correlation > threshold
                    high_corr_stocks = corr_with_target[corr_with_target > threshold].index.tolist()
                    high_corr_stocks.remove(potential_target)  # Remove self
                    
                    # If we have at least 2 stocks with high correlation, this is a candidate
                    if len(high_corr_stocks) >= 2:
                        if len(high_corr_stocks) > best_corr_count:
                            best_cluster = cluster_id
                            best_target = potential_target
                            best_features = high_corr_stocks
                            best_corr_count = len(high_corr_stocks)
                
            # If we found a good cluster, break
            if best_target is not None:
                break
        
        # If we didn't find any suitable clusters with the thresholds, create an artificial one
        if best_target is None:
            print("No suitable clusters found. Creating an artificial stock grouping.")
            
            # Find the most correlated pair of stocks overall
            high_corr_pairs = []
            for i, stock1 in enumerate(price_data.columns):
                for stock2 in price_data.columns[i+1:]:
                    corr = correlation_matrix.loc[stock1, stock2]
                    if corr > 0.5:  # Lower threshold for finding any correlations
                        high_corr_pairs.append((stock1, stock2, corr))
            
            if high_corr_pairs:
                # Sort by correlation descending
                high_corr_pairs.sort(key=lambda x: x[2], reverse=True)
                
                # Get the top correlated pair
                best_target = high_corr_pairs[0][0]
                best_features = [high_corr_pairs[0][1]]
                
                # Find additional correlated stocks
                for i in range(1, min(10, len(high_corr_pairs))):
                    pair = high_corr_pairs[i]
                    if pair[0] == best_target and pair[1] not in best_features:
                        best_features.append(pair[1])
                    elif pair[1] == best_target and pair[0] not in best_features:
                        best_features.append(pair[0])
                
                print(f"Created artificial group with target {best_target} and {len(best_features)} features.")
            else:
                raise ValueError("Could not find any correlated stocks in the dataset.")
        
        self.target_stock = best_target
        self.feature_stocks = best_features
        
        print(f"Selected target stock: {self.target_stock}")
        print(f"Number of feature stocks: {len(self.feature_stocks)}")
        print(f"Feature stocks: {self.feature_stocks}")
        
        # Step 6: Train AdaBoost-ElasticNet model
        X_train = train_data[self.feature_stocks]
        y_train = train_data[self.target_stock]
        
        self.adaboost_models = []
        self.model_weights = []
        
        # Initialize sample weights
        self.sample_weights = np.ones(len(X_train)) / len(X_train)
        
        for m in range(self.n_boost_iterations):
            # Train a base ElasticNet model with current sample weights
            model = ElasticNet(
                alpha=self.alpha,
                l1_ratio=self.l1_ratio,
                random_state=self.random_state
            )
            
            # Fit model with sample weights
            model.fit(X_train, y_train, sample_weight=self.sample_weights)
            
            # Make predictions
            y_pred = model.predict(X_train)
            
            # Calculate errors
            errors = np.abs(y_train - y_pred) / y_train  # Relative errors
            
            # Calculate weighted error
            weighted_error = np.sum(self.sample_weights * errors) / np.sum(self.sample_weights)
            
            # Calculate model weight
            model_weight = weighted_error / (1 - weighted_error) if weighted_error < 1 else 1
            
            # Update sample weights
            self.sample_weights = self.sample_weights * np.power(model_weight, 1 - errors)
            
            # Normalize sample weights
            self.sample_weights = self.sample_weights / np.sum(self.sample_weights)
            
            # Save model and its weight
            self.adaboost_models.append(model)
            self.model_weights.append(np.log(1 / model_weight) if model_weight > 0 else 0)
            
            print(f"Boosting iteration {m+1}/{self.n_boost_iterations}, Weighted error: {weighted_error:.4f}")
        
        # Normalize model weights
        self.model_weights = np.array(self.model_weights)
        if np.sum(self.model_weights) > 0:
            self.model_weights = self.model_weights / np.sum(self.model_weights)
        
        return self
    
    def predict(self, X):
        """
        Make predictions using the trained AdaBoost-ElasticNet ensemble
        
        Parameters:
        -----------
        X : pd.DataFrame
            Feature data with the feature stocks as columns
            
        Returns:
        --------
        np.ndarray
            Predicted values for the target stock
        """
        if self.adaboost_models is None:
            raise ValueError("Model has not been fitted yet")
        
        # Ensure X contains the required feature stocks
        if not all(stock in X.columns for stock in self.feature_stocks):
            missing = [stock for stock in self.feature_stocks if stock not in X.columns]
            raise ValueError(f"Missing feature stocks in input data: {missing}")
        
        X = X[self.feature_stocks]
        
        # Make predictions with each base model
        predictions = np.zeros((len(X), len(self.adaboost_models)))
        
        for i, model in enumerate(self.adaboost_models):
            predictions[:, i] = model.predict(X)
        
        # Weighted combination of predictions
        final_predictions = np.dot(predictions, self.model_weights)
        
        return final_predictions
    
    def evaluate(self, X, y):
        """
        Evaluate the model performance
        
        Parameters:
        -----------
        X : pd.DataFrame
            Feature data with the feature stocks as columns
        y : pd.Series
            True values for the target stock
            
        Returns:
        --------
        dict
            Dictionary containing evaluation metrics
        """
        if len(X) == 0:
            return {'RMSE': float('nan'), 'MAPE': float('nan')}
        
        y_pred = self.predict(X)
        
        rmse = sqrt(mean_squared_error(y, y_pred))
        mape = mean_absolute_percentage_error(y, y_pred) * 100
        
        return {
            'RMSE': rmse,
            'MAPE': mape
        }
    
    def backtest(self, price_data, train_ratio=0.7, initial_capital=100000, transaction_cost=0.0001):
        """
        Backtest the strategy
        
        Parameters:
        -----------
        price_data : pd.DataFrame
            Historical price data with dates as the index and stock codes as columns
        train_ratio : float
            Ratio of data to use for training (0.0 to 1.0)
        initial_capital : float
            Initial capital for backtesting
        transaction_cost : float
            Transaction cost as a percentage of the trade amount
            
        Returns:
        --------
        dict
            Dictionary containing backtesting results
        """
        # Split data into training and testing
        train_size = int(len(price_data) * train_ratio)
        test_data = price_data.iloc[train_size:]
        
        if len(test_data) == 0:
            print("Error: No test data available.")
            return {
                'capital_history': [initial_capital],
                'dates': [price_data.index[-1]],
                'annual_return': 0.0,
                'max_drawdown': 0.0,
                'sharpe_ratio': 0.0,
                'total_return': 0.0,
                'buy_hold_capital': [initial_capital]
            }
        
        # Make sure we have the target stock and feature stocks in the test data
        if self.target_stock not in test_data.columns:
            print(f"Error: Target stock {self.target_stock} not in test data.")
            return None
        
        missing_features = [stock for stock in self.feature_stocks if stock not in test_data.columns]
        if missing_features:
            print(f"Error: Feature stocks {missing_features} not in test data.")
            return None
        
        # Initialize portfolio
        capital = initial_capital
        position = 0  # 0: no position, 1: long target stock, -1: short target stock
        capital_history = [initial_capital]
        dates = test_data.index.tolist()
        
        # Calculate daily returns for buy-and-hold strategy
        buy_hold_returns = test_data[self.target_stock].pct_change().fillna(0)
        buy_hold_capital = [initial_capital]
        
        # Track trade counts
        trade_count = 0
        
        for i in range(1, len(test_data)):
            # Get current and previous day's data
            current_day = test_data.iloc[i]
            prev_day = test_data.iloc[i-1]
            
            # Make predictions
            X_prev = prev_day.to_frame().T
            X_current = current_day.to_frame().T
            
            y_prev_pred = self.predict(X_prev)[0]
            y_current_pred = self.predict(X_current)[0]
            
            # Calculate predicted return
            pred_return = (y_current_pred - y_prev_pred) / y_prev_pred
            
            # Trading logic
            if pred_return > 0.01 and position <= 0:  # Buy signal
                # If short, close position first
                if position == -1:
                    # Calculate return from shorting
                    actual_return = (prev_day[self.target_stock] - current_day[self.target_stock]) / prev_day[self.target_stock]
                    # Apply transaction cost
                    capital = capital * (1 + actual_return - transaction_cost)
                    trade_count += 1
                
                # Go long
                position = 1
                # Apply transaction cost
                capital = capital * (1 - transaction_cost)
                trade_count += 1
                
            elif pred_return < 0 and position >= 0:  # Sell signal
                # If long, close position first
                if position == 1:
                    # Calculate return from long position
                    actual_return = (current_day[self.target_stock] - prev_day[self.target_stock]) / prev_day[self.target_stock]
                    # Apply transaction cost
                    capital = capital * (1 + actual_return - transaction_cost)
                    trade_count += 1
                
                # Go short
                position = -1
                # Apply transaction cost
                capital = capital * (1 - transaction_cost)
                trade_count += 1
            
            # If we have a position, update capital based on daily changes
            if position == 1:
                daily_return = (current_day[self.target_stock] - prev_day[self.target_stock]) / prev_day[self.target_stock]
                capital = capital * (1 + daily_return)
            elif position == -1:
                daily_return = (prev_day[self.target_stock] - current_day[self.target_stock]) / prev_day[self.target_stock]
                capital = capital * (1 + daily_return)
            
            capital_history.append(capital)
            
            # Update buy-and-hold strategy
            buy_hold_capital.append(buy_hold_capital[-1] * (1 + buy_hold_returns.iloc[i]))
        
        print(f"Total trades: {trade_count}")
        
        # Calculate strategy performance metrics
        returns = np.diff(capital_history) / np.array(capital_history)[:-1]
        
        # Calculate annualized return
        annual_return = np.mean(returns) * 252 * 100
        
        # Calculate max drawdown
        cumulative_returns = np.array(capital_history) / initial_capital
        peak = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - peak) / peak
        max_drawdown = abs(np.min(drawdown)) * 100
        
        # Calculate Sharpe ratio
        risk_free_rate = 0  # Assuming zero risk-free rate for simplicity
        sharpe_ratio = (np.mean(returns) - risk_free_rate) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
        
        # Calculate total return
        total_return = (capital_history[-1] - initial_capital) / initial_capital * 100
        
        results = {
            'capital_history': capital_history,
            'dates': dates,
            'annual_return': annual_return,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'total_return': total_return,
            'buy_hold_capital': buy_hold_capital,
            'trade_count': trade_count
        }
        
        return results
    
    def plot_backtest_results(self, results, market_index=None):
        """
        Plot the backtest results
        
        Parameters:
        -----------
        results : dict
            Results from the backtest method
        market_index : pd.Series, optional
            Market index data for comparison
        """
        if results is None or 'capital_history' not in results:
            print("No valid backtest results to plot.")
            return
            
        # Create figure
        plt.figure(figsize=(14, 8))
        
        # Convert to relative returns (starting at 100)
        strategy_returns = np.array(results['capital_history']) / results['capital_history'][0] * 100
        buy_hold_returns = np.array(results['buy_hold_capital']) / results['buy_hold_capital'][0] * 100
        
        # Plot strategy returns
        plt.plot(results['dates'], strategy_returns, 'b-', linewidth=2, label='AdaBoost-ElasticNet Strategy')
        
        # Plot buy-and-hold returns
        plt.plot(results['dates'], buy_hold_returns, 'g-', linewidth=1, label='Buy & Hold Target Stock')
        
        # Plot market index if provided
        if market_index is not None:
            market_in_range = market_index[market_index.index.isin(results['dates'])]
            if len(market_in_range) > 0:
                market_returns = market_in_range / market_in_range.iloc[0] * 100
                plt.plot(market_returns.index, market_returns.values, 'r-', linewidth=1, label='Market Index')
        
        # Add labels and title
        plt.title('AdaBoost-ElasticNet Arbitrage Strategy Backtest Results', fontsize=16)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Return (%)', fontsize=12)
        
        # Add performance metrics as text
        performance_text = (
            f"Annual Return: {results['annual_return']:.2f}%\n"
            f"Max Drawdown: {results['max_drawdown']:.2f}%\n"
            f"Sharpe Ratio: {results['sharpe_ratio']:.2f}\n"
            f"Total Return: {results['total_return']:.2f}%\n"
            f"Total Trades: {results.get('trade_count', 'N/A')}"
        )
        
        plt.text(0.02, 0.97, performance_text, transform=plt.gca().transAxes, 
                 verticalalignment='top', horizontalalignment='left',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5), fontsize=12)
        
        plt.grid(True, alpha=0.3)
        plt.legend(loc='best')
        plt.tight_layout()
        plt.show()

# Function to generate simulated data with stronger correlations
def generate_simulated_data(n_stocks=100, n_features=50, n_days=800, test_days=200, seed=42):
    """
    Generate simulated stock data for testing with stronger correlations
    
    Parameters:
    -----------
    n_stocks : int
        Number of stocks to generate
    n_features : int
        Number of financial features per stock
    n_days : int
        Number of training days
    test_days : int
        Number of test days
    seed : int
        Random seed for reproducibility
        
    Returns:
    --------
    tuple
        (financial_data, price_data, market_index)
    """
    np.random.seed(seed)
    
    # Generate financial features for stocks
    financial_data = pd.DataFrame(
        np.random.randn(n_stocks, n_features),
        index=[f'stock_{i}' for i in range(n_stocks)],
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    
    # Add some structure by creating correlated feature groups
    for i in range(0, n_features, 5):
        base_feature = np.random.randn(n_stocks)
        for j in range(min(5, n_features - i)):
            financial_data.iloc[:, i+j] = base_feature + np.random.randn(n_stocks) * 0.2
    
    # Generate dates for price data
    total_days = n_days + test_days
    dates = pd.date_range(start='2010-01-01', periods=total_days, freq='B')
    
    # Generate market index
    market_return = np.random.normal(0.0002, 0.01, total_days)
    market_index = 1000 * np.cumprod(1 + market_return)
    
    # Generate stock prices with some correlation to the market
    price_data = pd.DataFrame(index=dates)
    
    for stock in financial_data.index:
        # Create base price path
        beta = np.random.uniform(0.5, 1.5)
        stock_specific_return = np.random.normal(0.0001, 0.02, total_days)
        stock_return = beta * market_return + stock_specific_return
        stock_price = 100 * np.cumprod(1 + stock_return)
        
        # Add to price dataframe
        price_data[stock] = stock_price
    
    # Create strong correlation groups
    # Divide stocks into small groups (5-10 stocks per group)
    n_groups = 15
    stocks_per_group = n_stocks // n_groups
    stock_groups = [financial_data.index[i:i+stocks_per_group] for i in range(0, n_stocks, stocks_per_group)]
    
    # For each group, create correlated price movements
    for i, group in enumerate(stock_groups):
        # Create a common group factor
        group_factor = np.random.normal(0, 0.015, total_days)
        
        # Base stock for the group
        base_stock = group[0]
        
        # Make all stocks in the group highly correlated with the base stock
        for j, stock in enumerate(group[1:]):
            # Stronger correlation for the first few stocks in each group
            if j < 3:  # First 3 stocks have very high correlation (0.8-0.95)
                correlation = 0.8 + 0.15 * np.random.rand()
            else:  # Other stocks have moderate correlation (0.5-0.8)
                correlation = 0.5 + 0.3 * np.random.rand()
                
            # Create correlated movement with the base stock
            price_data[stock] = price_data[base_stock] * (1 + correlation * group_factor)
            
            # Add some individual noise
            individual_noise = np.random.normal(0, 0.01 * (1 - correlation), total_days)
            price_data[stock] *= np.cumprod(1 + individual_noise)
    
    market_index = pd.Series(market_index, index=dates)
    
    return financial_data, price_data, market_index

# Main execution
if __name__ == "__main__":
    # Generate simulated data with stronger correlations
    print("Generating simulated data with stronger correlations...")
    financial_data, price_data, market_index = generate_simulated_data(
        n_stocks=100, 
        n_features=50, 
        n_days=800,  # Training days
        test_days=200,  # Testing days
        seed=42
    )
    
    # Define training ratio
    train_ratio = 0.8
    
    # Split data for information display
    train_size = int(len(price_data) * train_ratio)
    train_data = price_data.iloc[:train_size]
    test_data = price_data.iloc[train_size:]
    
    print(f"Number of stocks: {financial_data.shape[0]}")
    print(f"Number of financial features: {financial_data.shape[1]}")
    print(f"Price data time range: {price_data.index[0]} to {price_data.index[-1]}")
    print(f"Training data: {len(train_data)} days")
    print(f"Testing data: {len(test_data)} days")
    
    # Initialize and train the model
    print("\nTraining AdaBoost-ElasticNet Arbitrage model...")
    model = AdaBoostElasticNetArbitrage(
        n_clusters=10,  # Reduced for the simulated data
        pca_variance_ratio=0.85,
        n_boost_iterations=10,
        alpha=0.5,
        l1_ratio=0.5,
        random_state=42
    )
    
    model.fit(financial_data, price_data, train_ratio=train_ratio)
    
    # Evaluate the model
    print("\nEvaluating model prediction performance...")
    if len(test_data) > 0:
        X_test = test_data
        y_test = test_data[model.target_stock]
        
        metrics = model.evaluate(X_test, y_test)
        print(f"Test RMSE: {metrics['RMSE']:.4f}")
        print(f"Test MAPE: {metrics['MAPE']:.2f}%")
        
        # Compare predictions with actual values
        print("\nComparing predictions with actual values...")
        y_pred = model.predict(X_test)
        
        plt.figure(figsize=(14, 6))
        plt.plot(test_data.index, y_test.values, 'b-', label='Actual')
        plt.plot(test_data.index, y_pred, 'r-', label='Predicted')
        plt.title(f'Actual vs. Predicted Price for {model.target_stock}')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    else:
        print("No test data available for evaluation.")
    
    # Backtest the strategy
    print("\nBacktesting the strategy...")
    backtest_results = model.backtest(price_data, train_ratio=train_ratio)
    
    if backtest_results is not None:
        print(f"Annual Return: {backtest_results['annual_return']:.2f}%")
        print(f"Max Drawdown: {backtest_results['max_drawdown']:.2f}%")
        print(f"Sharpe Ratio: {backtest_results['sharpe_ratio']:.2f}")
        print(f"Total Return: {backtest_results['total_return']:.2f}%")
        
        # Plot backtest results
        print("\nPlotting backtest results...")
        model.plot_backtest_results(backtest_results, market_index)
        
        # Compare with conventional methods
        print("\nComparing with conventional methods...")
        from sklearn.linear_model import LinearRegression, Ridge, Lasso
        
        if len(test_data) > 0:
            # Train OLS model
            ols_model = LinearRegression()
            ols_model.fit(train_data[model.feature_stocks], train_data[model.target_stock])
            
            # Train Ridge model
            ridge_model = Ridge(alpha=1.0)
            ridge_model.fit(train_data[model.feature_stocks], train_data[model.target_stock])
            
            # Train Lasso model
            lasso_model = Lasso(alpha=0.1)
            lasso_model.fit(train_data[model.feature_stocks], train_data[model.target_stock])
            
            # Make predictions with all models
            X_test_features = test_data[model.feature_stocks]
            ols_pred = ols_model.predict(X_test_features)
            ridge_pred = ridge_model.predict(X_test_features)
            lasso_pred = lasso_model.predict(X_test_features)
            
            # Calculate metrics
            ols_rmse = sqrt(mean_squared_error(y_test, ols_pred))
            ridge_rmse = sqrt(mean_squared_error(y_test, ridge_pred))
            lasso_rmse = sqrt(mean_squared_error(y_test, lasso_pred))
            
            ols_mape = mean_absolute_percentage_error(y_test, ols_pred) * 100
            ridge_mape = mean_absolute_percentage_error(y_test, ridge_pred) * 100
            lasso_mape = mean_absolute_percentage_error(y_test, lasso_pred) * 100
            
            # Print comparison
            print("\nPrediction Performance Comparison:")
            print(f"{'Method':<20} {'RMSE':<10} {'MAPE':<10}")
            print(f"{'-'*40}")
            print(f"{'OLS':<20} {ols_rmse:<10.4f} {ols_mape:<10.2f}%")
            print(f"{'Ridge':<20} {ridge_rmse:<10.4f} {ridge_mape:<10.2f}%")
            print(f"{'Lasso':<20} {lasso_rmse:<10.4f} {lasso_mape:<10.2f}%")
            print(f"{'AdaBoost-ElasticNet':<20} {metrics['RMSE']:<10.4f} {metrics['MAPE']:<10.2f}%")
            
            # Plot comparison of predictions
            plt.figure(figsize=(14, 8))
            plt.plot(test_data.index, y_test.values, 'k-', linewidth=2, label='Actual')
            plt.plot(test_data.index, ols_pred, 'b-', linewidth=1, label='OLS')
            plt.plot(test_data.index, ridge_pred, 'g-', linewidth=1, label='Ridge')
            plt.plot(test_data.index, lasso_pred, 'y-', linewidth=1, label='Lasso')
            plt.plot(test_data.index, y_pred, 'r-', linewidth=1, label='AdaBoost-ElasticNet')
            
            plt.title(f'Prediction Comparison for {model.target_stock}', fontsize=16)
            plt.xlabel('Date', fontsize=12)
            plt.ylabel('Price', fontsize=12)
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
        else:
            print("No test data available for comparison.")