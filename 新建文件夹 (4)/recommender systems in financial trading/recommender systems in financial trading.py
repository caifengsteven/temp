import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

class FinancialRecommenderSystem:
    """
    A recommender system for financial trading that implements the concepts
    from the paper by Alicia Vidler.
    """
    
    def __init__(self, lookback_period=252, prediction_horizon=21, n_clusters=5):
        """
        Initialize the recommender system.
        
        Parameters:
        -----------
        lookback_period : int
            Number of trading days to look back for feature calculation
        prediction_horizon : int
            Number of trading days to look ahead for return prediction
        n_clusters : int
            Number of clusters to group stocks by similarity
        """
        self.lookback_period = lookback_period
        self.prediction_horizon = prediction_horizon
        self.n_clusters = n_clusters
        
        # Initialize components
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=5)
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        self.return_predictor = RandomForestRegressor(n_estimators=100, random_state=42)
        self.conviction_model = RandomForestClassifier(n_estimators=100, random_state=42)
        
        # Track record
        self.recommendation_history = []
        
    def prepare_features(self, price_data):
        """
        Calculate features from price data.
        
        Parameters:
        -----------
        price_data : pandas.DataFrame
            DataFrame with price data for multiple stocks
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with calculated features
        """
        # Make a copy to avoid modifying the original data
        df = price_data.copy()
        
        # Create feature dataframes
        features = {}
        
        # Loop through each ticker
        for ticker in df.columns:
            ticker_data = df[ticker].dropna()
            
            if len(ticker_data) < self.lookback_period + self.prediction_horizon:
                continue
                
            # Calculate features
            ticker_features = pd.DataFrame(index=[ticker_data.index[-1]])
            
            # Price momentum features (returns over different periods)
            ticker_features['return_5d'] = ticker_data.pct_change(5).iloc[-1]
            ticker_features['return_10d'] = ticker_data.pct_change(10).iloc[-1]
            ticker_features['return_21d'] = ticker_data.pct_change(21).iloc[-1]
            ticker_features['return_63d'] = ticker_data.pct_change(63).iloc[-1]
            ticker_features['return_126d'] = ticker_data.pct_change(126).iloc[-1]
            ticker_features['return_252d'] = ticker_data.pct_change(252).iloc[-1]
            
            # Volatility features
            ticker_features['volatility_21d'] = ticker_data.pct_change().rolling(21).std().iloc[-1]
            ticker_features['volatility_63d'] = ticker_data.pct_change().rolling(63).std().iloc[-1]
            
            # Moving average indicators
            ticker_features['ma_cross_50_200'] = (ticker_data.rolling(50).mean().iloc[-1] - 
                                                ticker_data.rolling(200).mean().iloc[-1]) / ticker_data.iloc[-1]
            
            # Current price relative to historical ranges
            max_252d = ticker_data.rolling(252).max().iloc[-1]
            min_252d = ticker_data.rolling(252).min().iloc[-1]
            ticker_features['price_to_range_252d'] = (ticker_data.iloc[-1] - min_252d) / (max_252d - min_252d)
            
            # Future return (target variable)
            if len(ticker_data) >= self.lookback_period + self.prediction_horizon:
                future_price = ticker_data.shift(-self.prediction_horizon).iloc[-1]
                ticker_features['future_return'] = (future_price / ticker_data.iloc[-1]) - 1
            else:
                ticker_features['future_return'] = np.nan
            
            # Store features
            features[ticker] = ticker_features
        
        # Combine all features
        feature_df = pd.concat(features.values())
        return feature_df
    
    def cluster_stocks(self, features):
        """
        Cluster stocks based on their features.
        
        Parameters:
        -----------
        features : pandas.DataFrame
            DataFrame with calculated features
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with cluster assignments
        """
        # Drop target variable and any NA values
        X = features.drop('future_return', axis=1).dropna()
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Apply PCA for dimensionality reduction
        X_pca = self.pca.fit_transform(X_scaled)
        
        # Perform clustering
        clusters = self.kmeans.fit_predict(X_pca)
        
        # Add cluster assignments to the original data
        cluster_df = pd.DataFrame(index=X.index)
        cluster_df['cluster'] = clusters
        
        return cluster_df
    
    def train_return_predictor(self, features):
        """
        Train a model to predict future returns.
        
        Parameters:
        -----------
        features : pandas.DataFrame
            DataFrame with calculated features
            
        Returns:
        --------
        object
            Trained model
        """
        # Drop rows with missing target
        data = features.dropna()
        
        # Split features and target
        X = data.drop('future_return', axis=1)
        y = data['future_return']
        
        # Train the model
        self.return_predictor.fit(X, y)
        
        return self.return_predictor
    
    def train_conviction_model(self, features, threshold=0.05):
        """
        Train a model to predict conviction level.
        
        Parameters:
        -----------
        features : pandas.DataFrame
            DataFrame with calculated features
        threshold : float
            Threshold for binary classification of positive returns
            
        Returns:
        --------
        object
            Trained model
        """
        # Drop rows with missing target
        data = features.dropna()
        
        # Split features and target
        X = data.drop('future_return', axis=1)
        
        # Create binary target for conviction (1 if return > threshold, 0 otherwise)
        y_conviction = (data['future_return'] > threshold).astype(int)
        
        # Train the model
        self.conviction_model.fit(X, y_conviction)
        
        return self.conviction_model
    
    def generate_recommendations(self, features, clusters, top_n=10):
        """
        Generate stock recommendations with conviction levels.
        
        Parameters:
        -----------
        features : pandas.DataFrame
            DataFrame with calculated features
        clusters : pandas.DataFrame
            DataFrame with cluster assignments
        top_n : int
            Number of top recommendations to return
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with recommendations and metadata
        """
        # Drop rows with missing data
        X = features.drop('future_return', axis=1).dropna()
        
        # Predict future returns
        predicted_returns = self.return_predictor.predict(X)
        
        # Predict conviction probabilities
        conviction_probs = self.conviction_model.predict_proba(X)[:, 1]
        
        # Create recommendations dataframe
        recommendations = pd.DataFrame(index=X.index)
        recommendations['predicted_return'] = predicted_returns
        recommendations['conviction'] = conviction_probs
        
        # Add feature importance for explainability
        feature_importance = pd.Series(
            self.return_predictor.feature_importances_,
            index=X.columns
        ).sort_values(ascending=False)
        
        top_features = feature_importance.head(3).index.tolist()
        for feature in top_features:
            recommendations[f'value_{feature}'] = X[feature]
        
        # Add cluster information
        recommendations = recommendations.join(clusters)
        
        # Sort by conviction and predicted return
        recommendations['score'] = recommendations['conviction'] * recommendations['predicted_return']
        recommendations = recommendations.sort_values('score', ascending=False)
        
        # Get top N recommendations
        top_recommendations = recommendations.head(top_n)
        
        # Save to history for track record
        timestamp = datetime.now().strftime('%Y-%m-%d')
        for ticker, row in top_recommendations.iterrows():
            self.recommendation_history.append({
                'date': timestamp,
                'ticker': ticker,
                'predicted_return': row['predicted_return'],
                'conviction': row['conviction'],
                'score': row['score'],
                'cluster': row['cluster'] if 'cluster' in row else None
            })
        
        return top_recommendations
    
    def get_track_record(self, actual_returns=None):
        """
        Get the track record of past recommendations.
        
        Parameters:
        -----------
        actual_returns : pandas.DataFrame, optional
            DataFrame with actual returns for evaluation
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with recommendation track record
        """
        if not self.recommendation_history:
            return pd.DataFrame()
        
        track_record = pd.DataFrame(self.recommendation_history)
        
        # Add actual returns if provided
        if actual_returns is not None:
            for i, row in track_record.iterrows():
                ticker = row['ticker']
                date = pd.to_datetime(row['date'])
                
                if ticker in actual_returns.columns and date in actual_returns.index:
                    future_date_idx = actual_returns.index.get_indexer([date], method='pad')[0] + self.prediction_horizon
                    
                    if future_date_idx < len(actual_returns):
                        future_date = actual_returns.index[future_date_idx]
                        
                        if not np.isnan(actual_returns.loc[date, ticker]) and not np.isnan(actual_returns.loc[future_date, ticker]):
                            actual_return = (actual_returns.loc[future_date, ticker] / actual_returns.loc[date, ticker]) - 1
                            track_record.loc[i, 'actual_return'] = actual_return
                            track_record.loc[i, 'accuracy'] = 1 if (row['predicted_return'] > 0 and actual_return > 0) or \
                                                                  (row['predicted_return'] <= 0 and actual_return <= 0) else 0
        
        return track_record
    
    def get_portfolio_recommendations(self, features, risk_tolerance='medium', min_conviction=0.6):
        """
        Generate portfolio recommendations based on risk tolerance.
        
        Parameters:
        -----------
        features : pandas.DataFrame
            DataFrame with calculated features
        risk_tolerance : str
            Risk tolerance level ('low', 'medium', or 'high')
        min_conviction : float
            Minimum conviction level for recommendations
            
        Returns:
        --------
        dict
            Dictionary with portfolio recommendations and metadata
        """
        # Generate base recommendations
        clusters = self.cluster_stocks(features)
        recommendations = self.generate_recommendations(features, clusters, top_n=20)
        
        # Filter by conviction
        high_conviction = recommendations[recommendations['conviction'] >= min_conviction]
        
        # Adjust based on risk tolerance
        if risk_tolerance == 'low':
            # Lower volatility stocks with good returns
            portfolio_recs = high_conviction.sort_values('value_volatility_21d')
        elif risk_tolerance == 'high':
            # Higher expected return regardless of volatility
            portfolio_recs = high_conviction.sort_values('predicted_return', ascending=False)
        else:  # medium
            # Balance return and conviction
            portfolio_recs = high_conviction.sort_values('score', ascending=False)
        
        # Create metadata for downstream systems
        metadata = {
            'risk_tolerance': risk_tolerance,
            'min_conviction': min_conviction,
            'avg_predicted_return': portfolio_recs['predicted_return'].mean(),
            'avg_conviction': portfolio_recs['conviction'].mean(),
            'cluster_distribution': portfolio_recs['cluster'].value_counts().to_dict(),
            'top_features': self.return_predictor.feature_importances_,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        return {
            'recommendations': portfolio_recs.head(10),
            'metadata': metadata
        }
    
    def visualize_recommendations(self, recommendations):
        """
        Visualize recommendations for explainability.
        
        Parameters:
        -----------
        recommendations : pandas.DataFrame
            DataFrame with recommendations
            
        Returns:
        --------
        None
        """
        # Set up the figure
        fig = plt.figure(figsize=(15, 10))
        
        # 1. Conviction vs Expected Return scatter plot
        ax1 = plt.subplot2grid((2, 2), (0, 0))
        scatter = ax1.scatter(recommendations['predicted_return'], 
                    recommendations['conviction'], 
                    s=100, 
                    c=recommendations['cluster'] if 'cluster' in recommendations else 'blue', 
                    alpha=0.7)
        
        if 'cluster' in recommendations:
            plt.colorbar(scatter, label='Cluster')
            
        ax1.set_xlabel('Predicted Return')
        ax1.set_ylabel('Conviction Level')
        ax1.set_title('Conviction vs. Predicted Return')
        ax1.grid(True, alpha=0.3)
        
        # Add ticker labels
        for idx, row in recommendations.iterrows():
            ax1.annotate(idx, 
                        (row['predicted_return'], row['conviction']),
                        xytext=(5, 5),
                        textcoords='offset points')
        
        # 2. Feature importance
        ax2 = plt.subplot2grid((2, 2), (0, 1))
        feature_importance = pd.Series(
            self.return_predictor.feature_importances_,
            index=features.drop('future_return', axis=1).columns
        ).sort_values(ascending=False).head(10)
        
        feature_importance.plot(kind='barh', ax=ax2)
        ax2.set_title('Top 10 Feature Importance')
        ax2.set_xlabel('Importance')
        
        # 3. Cluster distribution
        if 'cluster' in recommendations:
            ax3 = plt.subplot2grid((2, 2), (1, 0))
            recommendations['cluster'].value_counts().plot(kind='pie', ax=ax3, autopct='%1.1f%%')
            ax3.set_title('Recommendation Distribution by Cluster')
            ax3.set_ylabel('')
        
        # 4. Track record
        ax4 = plt.subplot2grid((2, 2), (1, 1))
        track_record = self.get_track_record()
        
        if 'actual_return' in track_record.columns:
            track_record.sort_values('date').set_index('date')[['predicted_return', 'actual_return']].plot(ax=ax4)
            ax4.set_title('Predicted vs. Actual Returns')
        else:
            if not track_record.empty:
                track_record.sort_values('date').set_index('date')['predicted_return'].plot(ax=ax4)
                ax4.set_title('Predicted Returns Over Time')
            else:
                ax4.text(0.5, 0.5, 'No track record available yet', ha='center', va='center')
                ax4.set_title('Track Record')
        
        plt.tight_layout()
        plt.show()

class PortfolioManager:
    """
    A portfolio manager that uses recommendations from the recommender system.
    """
    
    def __init__(self, initial_capital=1000000, max_positions=20, position_size=0.05):
        """
        Initialize the portfolio manager.
        
        Parameters:
        -----------
        initial_capital : float
            Initial capital for the portfolio
        max_positions : int
            Maximum number of positions in the portfolio
        position_size : float
            Default position size as fraction of portfolio
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.max_positions = max_positions
        self.default_position_size = position_size
        
        self.positions = {}  # ticker -> {quantity, entry_price, entry_date}
        self.trades = []  # List of trade dictionaries
        self.portfolio_history = []  # List of portfolio value over time
    
    def process_recommendations(self, recommendations, current_prices, date, min_conviction=0.7, conviction_scaling=True):
        """
        Process recommendations to make trading decisions.
        
        Parameters:
        -----------
        recommendations : pandas.DataFrame
            DataFrame with stock recommendations
        current_prices : pandas.Series
            Series with current prices
        date : datetime or str
            Current date
        min_conviction : float
            Minimum conviction threshold for new positions
        conviction_scaling : bool
            Whether to scale position size by conviction
            
        Returns:
        --------
        list
            List of trade decisions
        """
        # Convert date to string if it's a datetime
        if isinstance(date, datetime):
            date = date.strftime('%Y-%m-%d')
        
        # Get high conviction recommendations
        high_conviction = recommendations[recommendations['conviction'] >= min_conviction]
        
        # Calculate available capital
        allocated_capital = sum([self.positions[ticker]['quantity'] * current_prices[ticker] 
                                 for ticker in self.positions if ticker in current_prices])
        available_capital = self.current_capital - allocated_capital
        
        # Prepare trade decisions
        trade_decisions = []
        
        # Exit positions that are no longer recommended
        for ticker in list(self.positions.keys()):
            if ticker not in high_conviction.index and ticker in current_prices:
                # Exit position
                quantity = self.positions[ticker]['quantity']
                entry_price = self.positions[ticker]['entry_price']
                exit_price = current_prices[ticker]
                
                trade = {
                    'date': date,
                    'ticker': ticker,
                    'action': 'SELL',
                    'quantity': quantity,
                    'price': exit_price,
                    'value': quantity * exit_price,
                    'return': (exit_price / entry_price) - 1,
                    'entry_date': self.positions[ticker]['entry_date'],
                    'holding_period': (pd.to_datetime(date) - pd.to_datetime(self.positions[ticker]['entry_date'])).days
                }
                
                trade_decisions.append(trade)
                self.trades.append(trade)
                
                # Update capital
                self.current_capital += quantity * exit_price
                
                # Remove from positions
                del self.positions[ticker]
        
        # Enter new positions based on recommendations
        for ticker, row in high_conviction.iterrows():
            if ticker not in self.positions and ticker in current_prices and len(self.positions) < self.max_positions:
                # Calculate position size
                if conviction_scaling:
                    position_size = self.default_position_size * row['conviction']
                else:
                    position_size = self.default_position_size
                
                # Calculate quantity based on available capital
                position_value = min(available_capital * position_size, available_capital)
                quantity = int(position_value / current_prices[ticker])
                
                if quantity > 0:
                    # Enter position
                    entry_price = current_prices[ticker]
                    
                    trade = {
                        'date': date,
                        'ticker': ticker,
                        'action': 'BUY',
                        'quantity': quantity,
                        'price': entry_price,
                        'value': quantity * entry_price,
                        'conviction': row['conviction'],
                        'predicted_return': row['predicted_return']
                    }
                    
                    trade_decisions.append(trade)
                    self.trades.append(trade)
                    
                    # Update capital and positions
                    self.current_capital -= quantity * entry_price
                    self.positions[ticker] = {
                        'quantity': quantity,
                        'entry_price': entry_price,
                        'entry_date': date
                    }
                    
                    # Update available capital
                    available_capital -= quantity * entry_price
        
        return trade_decisions
    
    def update_portfolio_value(self, current_prices, date):
        """
        Update the portfolio value based on current prices.
        
        Parameters:
        -----------
        current_prices : pandas.Series
            Series with current prices
        date : datetime or str
            Current date
            
        Returns:
        --------
        float
            Current portfolio value
        """
        # Convert date to string if it's a datetime
        if isinstance(date, datetime):
            date = date.strftime('%Y-%m-%d')
        
        # Calculate portfolio value
        stock_value = sum([self.positions[ticker]['quantity'] * current_prices[ticker] 
                          for ticker in self.positions if ticker in current_prices])
        portfolio_value = self.current_capital + stock_value
        
        # Save to history
        self.portfolio_history.append({
            'date': date,
            'cash': self.current_capital,
            'stock_value': stock_value,
            'total_value': portfolio_value,
            'return': (portfolio_value / self.initial_capital) - 1,
            'n_positions': len(self.positions)
        })
        
        return portfolio_value
    
    def get_portfolio_summary(self):
        """
        Get a summary of the current portfolio.
        
        Returns:
        --------
        dict
            Dictionary with portfolio summary
        """
        if not self.portfolio_history:
            return {
                'current_value': self.initial_capital,
                'return': 0,
                'n_positions': 0,
                'cash_ratio': 1.0
            }
        
        latest = self.portfolio_history[-1]
        
        return {
            'current_value': latest['total_value'],
            'return': latest['return'],
            'n_positions': latest['n_positions'],
            'cash_ratio': latest['cash'] / latest['total_value']
        }
    
    def plot_portfolio_performance(self, benchmark=None):
        """
        Plot the portfolio performance.
        
        Parameters:
        -----------
        benchmark : pandas.Series, optional
            Series with benchmark performance for comparison
            
        Returns:
        --------
        None
        """
        if not self.portfolio_history:
            print("No portfolio history available yet.")
            return
        
        # Convert history to DataFrame
        history_df = pd.DataFrame(self.portfolio_history)
        history_df['date'] = pd.to_datetime(history_df['date'])
        history_df = history_df.set_index('date')
        
        # Set up the figure
        fig, axes = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [2, 1]})
        
        # Plot portfolio value
        ax1 = axes[0]
        history_df['total_value'].plot(ax=ax1, label='Portfolio Value')
        
        # Add benchmark if provided
        if benchmark is not None:
            # Align benchmark to portfolio history dates
            aligned_benchmark = benchmark.reindex(history_df.index, method='pad')
            
            # Scale benchmark to start at same value as portfolio
            scaled_benchmark = aligned_benchmark / aligned_benchmark.iloc[0] * self.initial_capital
            
            scaled_benchmark.plot(ax=ax1, label='Benchmark', linestyle='--')
        
        ax1.set_title('Portfolio Performance')
        ax1.set_ylabel('Value ($)')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Plot composition
        ax2 = axes[1]
        history_df[['cash', 'stock_value']].plot(kind='area', stacked=True, ax=ax2)
        ax2.set_title('Portfolio Composition')
        ax2.set_ylabel('Value ($)')
        ax2.set_xlabel('Date')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Plot trade analysis
        if self.trades:
            self._plot_trade_analysis()
    
    def _plot_trade_analysis(self):
        """
        Plot trade analysis.
        
        Returns:
        --------
        None
        """
        # Convert trades to DataFrame
        trades_df = pd.DataFrame(self.trades)
        
        if 'return' not in trades_df.columns or trades_df[trades_df['action'] == 'SELL'].empty:
            print("No completed trades to analyze yet.")
            return
        
        # Get completed trades
        completed_trades = trades_df[trades_df['action'] == 'SELL']
        
        # Set up the figure
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Returns distribution
        ax1 = axes[0, 0]
        completed_trades['return'].hist(bins=20, ax=ax1)
        ax1.axvline(x=0, color='r', linestyle='--')
        ax1.set_title('Trade Returns Distribution')
        ax1.set_xlabel('Return')
        ax1.set_ylabel('Frequency')
        
        # 2. Returns by holding period
        ax2 = axes[0, 1]
        ax2.scatter(completed_trades['holding_period'], completed_trades['return'], alpha=0.7)
        ax2.axhline(y=0, color='r', linestyle='--')
        ax2.set_title('Returns by Holding Period')
        ax2.set_xlabel('Holding Period (days)')
        ax2.set_ylabel('Return')
        
        # 3. Cumulative return
        ax3 = axes[1, 0]
        completed_trades = completed_trades.sort_values('date')
        completed_trades['cumulative_return'] = (1 + completed_trades['return']).cumprod() - 1
        completed_trades['cumulative_return'].plot(ax=ax3)
        ax3.set_title('Cumulative Return from Completed Trades')
        ax3.set_xlabel('Trade Number')
        ax3.set_ylabel('Cumulative Return')
        
        # 4. Win/loss ratio
        ax4 = axes[1, 1]
        win_rate = (completed_trades['return'] > 0).mean()
        loss_rate = 1 - win_rate
        
        ax4.pie([win_rate, loss_rate], labels=['Win', 'Loss'], autopct='%1.1f%%', colors=['green', 'red'])
        ax4.set_title('Win/Loss Ratio')
        
        plt.tight_layout()
        plt.show()

# Function to simulate stock prices
def simulate_stock_data(n_stocks=50, n_days=1000, drift=0.0001, volatility=0.01, correlation=0.3):
    """
    Simulate stock price data with some correlation.
    
    Parameters:
    -----------
    n_stocks : int
        Number of stocks to simulate
    n_days : int
        Number of days to simulate
    drift : float
        Daily drift factor
    volatility : float
        Daily volatility
    correlation : float
        Correlation between stocks
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with simulated stock prices
    """
    # Create date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=n_days)
    dates = pd.date_range(start=start_date, end=end_date, freq='B')
    
    # Create correlation matrix
    corr_matrix = np.full((n_stocks, n_stocks), correlation)
    np.fill_diagonal(corr_matrix, 1)
    
    # Create Cholesky decomposition of correlation matrix
    L = np.linalg.cholesky(corr_matrix)
    
    # Simulate returns
    uncorrelated_returns = np.random.normal(drift, volatility, size=(len(dates), n_stocks))
    correlated_returns = uncorrelated_returns @ L.T
    
    # Convert returns to prices
    prices = np.cumprod(1 + correlated_returns, axis=0)
    
    # Scale to realistic price levels (between $10 and $200)
    min_price, max_price = 10, 200
    for i in range(n_stocks):
        prices[:, i] = min_price + (prices[:, i] - prices[0, i]) * (max_price - min_price) / prices[-1, i]
    
    # Create DataFrame
    stock_names = [f'Stock_{i+1}' for i in range(n_stocks)]
    price_df = pd.DataFrame(prices, index=dates, columns=stock_names)
    
    return price_df

# Simulate some data with sector/factor structure
def simulate_stock_data_with_sectors(n_stocks=50, n_days=1000, n_sectors=5):
    """
    Simulate stock price data with sector structure.
    
    Parameters:
    -----------
    n_stocks : int
        Number of stocks to simulate
    n_days : int
        Number of days to simulate
    n_sectors : int
        Number of sectors
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with simulated stock prices
    """
    # Create date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=n_days)
    dates = pd.date_range(start=start_date, end=end_date, freq='B')
    
    # Create sector returns
    sector_drift = np.random.uniform(0.0001, 0.0005, n_sectors)
    sector_vol = np.random.uniform(0.008, 0.015, n_sectors)
    
    sector_returns = np.zeros((len(dates), n_sectors))
    for i in range(n_sectors):
        sector_returns[:, i] = np.random.normal(sector_drift[i], sector_vol[i], len(dates))
    
    # Create market returns
    market_drift = 0.0003
    market_vol = 0.01
    market_returns = np.random.normal(market_drift, market_vol, len(dates))
    
    # Assign stocks to sectors
    stocks_per_sector = n_stocks // n_sectors
    remainder = n_stocks % n_sectors
    
    sector_assignments = []
    for i in range(n_sectors):
        if i < remainder:
            sector_assignments.extend([i] * (stocks_per_sector + 1))
        else:
            sector_assignments.extend([i] * stocks_per_sector)
    
    # Simulate stock returns with market, sector, and idiosyncratic components
    stock_returns = np.zeros((len(dates), n_stocks))
    
    for i in range(n_stocks):
        sector = sector_assignments[i]
        
        # Weights for market, sector, and idiosyncratic components
        market_weight = np.random.uniform(0.3, 0.7)
        sector_weight = np.random.uniform(0.2, 0.5)
        idio_weight = np.sqrt(1 - market_weight**2 - sector_weight**2)
        
        # Generate idiosyncratic returns
        idio_vol = np.random.uniform(0.01, 0.02)
        idio_returns = np.random.normal(0, idio_vol, len(dates))
        
        # Combine components
        stock_returns[:, i] = (market_weight * market_returns + 
                              sector_weight * sector_returns[:, sector] + 
                              idio_weight * idio_returns)
    
    # Convert returns to prices
    prices = np.cumprod(1 + stock_returns, axis=0)
    
    # Scale to realistic price levels (between $10 and $200)
    min_price, max_price = 10, 200
    for i in range(n_stocks):
        scale_factor = np.random.uniform(min_price, max_price) / prices[0, i]
        prices[:, i] *= scale_factor
    
    # Create DataFrame
    stock_names = [f'Stock_{i+1}' for i in range(n_stocks)]
    price_df = pd.DataFrame(prices, index=dates, columns=stock_names)
    
    return price_df, sector_assignments

# Function to split data into training and testing periods
def split_data(data, train_ratio=0.7):
    """
    Split data into training and testing periods.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        DataFrame with time series data
    train_ratio : float
        Ratio of data to use for training
        
    Returns:
    --------
    tuple
        (training_data, testing_data)
    """
    n = len(data)
    split_idx = int(n * train_ratio)
    
    train_data = data.iloc[:split_idx]
    test_data = data.iloc[split_idx:]
    
    return train_data, test_data

# Main function to test the strategy
def test_recommender_strategy():
    """
    Test the recommender system strategy with simulated data.
    """
    print("Simulating stock data...")
    prices, sector_assignments = simulate_stock_data_with_sectors(n_stocks=50, n_days=1000, n_sectors=5)
    
    # Create a mapping from stock names to sectors
    sector_map = {f'Stock_{i+1}': sector_assignments[i] for i in range(len(sector_assignments))}
    
    # Split data into training and testing
    train_data, test_data = split_data(prices, train_ratio=0.7)
    
    print(f"Training data: {train_data.shape}, Testing data: {test_data.shape}")
    
    # Initialize recommender system
    rs = FinancialRecommenderSystem(lookback_period=60, prediction_horizon=21, n_clusters=5)
    
    # Prepare features for training
    print("Preparing training features...")
    train_features = rs.prepare_features(train_data)
    
    # Train the models
    print("Training recommender system models...")
    rs.train_return_predictor(train_features)
    rs.train_conviction_model(train_features, threshold=0.03)
    
    # Initialize portfolio manager
    pm = PortfolioManager(initial_capital=1000000, max_positions=10, position_size=0.1)
    
    # Simulate backtesting on test data
    print("Running backtest...")
    test_dates = test_data.index[rs.lookback_period:]
    
    for i, date in enumerate(test_dates):
        if i % 21 == 0:  # Generate new recommendations every 21 days
            # Get current data
            current_data = test_data.loc[:date]
            
            # Prepare features
            features = rs.prepare_features(current_data)
            
            # Generate recommendations
            portfolio_recs = rs.get_portfolio_recommendations(features, risk_tolerance='medium', min_conviction=0.6)
            recommendations = portfolio_recs['recommendations']
            
            # Process recommendations
            current_prices = test_data.loc[date]
            trade_decisions = pm.process_recommendations(recommendations, current_prices, date)
            
            if trade_decisions:
                print(f"Date: {date.strftime('%Y-%m-%d')}, Made {len(trade_decisions)} trades")
        
        # Update portfolio value
        current_prices = test_data.loc[date]
        pm.update_portfolio_value(current_prices, date)
    
    # Get track record
    track_record = rs.get_track_record()
    
    # Calculate benchmark performance (simple average of all stocks)
    benchmark = test_data.mean(axis=1)
    benchmark = benchmark / benchmark.iloc[0] * pm.initial_capital
    
    # Plot results
    print("Plotting results...")
    pm.plot_portfolio_performance(benchmark=benchmark)
    
    # Get portfolio summary
    summary = pm.get_portfolio_summary()
    print("\nPortfolio Summary:")
    print(f"Final Value: ${summary['current_value']:.2f}")
    print(f"Total Return: {summary['return']*100:.2f}%")
    print(f"Number of Positions: {summary['n_positions']}")
    print(f"Cash Ratio: {summary['cash_ratio']*100:.2f}%")
    
    # Visualize recommendations
    latest_features = rs.prepare_features(test_data)
    latest_recommendations = rs.get_portfolio_recommendations(latest_features)['recommendations']
    rs.visualize_recommendations(latest_recommendations)
    
    # Analysis of track record
    if not track_record.empty and 'actual_return' in track_record.columns:
        print("\nRecommendation Track Record:")
        print(f"Average Predicted Return: {track_record['predicted_return'].mean()*100:.2f}%")
        print(f"Average Actual Return: {track_record['actual_return'].mean()*100:.2f}%")
        
        if 'accuracy' in track_record.columns:
            print(f"Direction Accuracy: {track_record['accuracy'].mean()*100:.2f}%")
        
        # Correlation between conviction and actual returns
        if 'conviction' in track_record.columns and 'actual_return' in track_record.columns:
            corr = track_record[['conviction', 'actual_return']].corr().iloc[0, 1]
            print(f"Correlation between Conviction and Actual Returns: {corr:.4f}")
    
    return rs, pm, prices, test_data

# Run the test
rs, pm, prices, test_data = test_recommender_strategy()

# Additional Analysis: Demonstrate explainable AI aspects of the recommender system
def analyze_explainable_ai_aspects(rs, prices, test_data):
    """
    Analyze the explainable AI aspects of the recommender system.
    
    Parameters:
    -----------
    rs : FinancialRecommenderSystem
        Trained recommender system
    prices : pandas.DataFrame
        Full price data
    test_data : pandas.DataFrame
        Test data
        
    Returns:
    --------
    None
    """
    print("\nAnalyzing Explainable AI Aspects of the Recommender System...")
    
    # Get latest data
    latest_data = test_data.iloc[-rs.lookback_period:]
    
    # Prepare features
    features = rs.prepare_features(latest_data)
    
    # Generate recommendations with different risk profiles
    low_risk_recs = rs.get_portfolio_recommendations(features, risk_tolerance='low')['recommendations']
    med_risk_recs = rs.get_portfolio_recommendations(features, risk_tolerance='medium')['recommendations']
    high_risk_recs = rs.get_portfolio_recommendations(features, risk_tolerance='high')['recommendations']
    
    # Compare recommendations across risk profiles
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot low risk recommendations
    ax1 = axes[0]
    ax1.scatter(low_risk_recs['predicted_return'], low_risk_recs['conviction'], s=100, alpha=0.7)
    for idx, row in low_risk_recs.iterrows():
        ax1.annotate(idx, (row['predicted_return'], row['conviction']), xytext=(5, 5), textcoords='offset points')
    ax1.set_title('Low Risk Recommendations')
    ax1.set_xlabel('Predicted Return')
    ax1.set_ylabel('Conviction Level')
    ax1.grid(True, alpha=0.3)
    
    # Plot medium risk recommendations
    ax2 = axes[1]
    ax2.scatter(med_risk_recs['predicted_return'], med_risk_recs['conviction'], s=100, alpha=0.7)
    for idx, row in med_risk_recs.iterrows():
        ax2.annotate(idx, (row['predicted_return'], row['conviction']), xytext=(5, 5), textcoords='offset points')
    ax2.set_title('Medium Risk Recommendations')
    ax2.set_xlabel('Predicted Return')
    ax2.set_ylabel('Conviction Level')
    ax2.grid(True, alpha=0.3)
    
    # Plot high risk recommendations
    ax3 = axes[2]
    ax3.scatter(high_risk_recs['predicted_return'], high_risk_recs['conviction'], s=100, alpha=0.7)
    for idx, row in high_risk_recs.iterrows():
        ax3.annotate(idx, (row['predicted_return'], row['conviction']), xytext=(5, 5), textcoords='offset points')
    ax3.set_title('High Risk Recommendations')
    ax3.set_xlabel('Predicted Return')
    ax3.set_ylabel('Conviction Level')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.suptitle('Comparison of Recommendations Across Risk Profiles', fontsize=16, y=1.05)
    plt.show()
    
    # Analyze feature importance for different stocks
    # Select a few stocks for demonstration
    sample_stocks = low_risk_recs.index.tolist()[:3] + high_risk_recs.index.tolist()[:3]
    
    # Get feature importance
    feature_importance = pd.Series(
        rs.return_predictor.feature_importances_,
        index=features.drop('future_return', axis=1).columns
    ).sort_values(ascending=False)
    
    # Get stock-specific feature values
    stock_features = features.loc[sample_stocks].drop('future_return', axis=1)
    
    # Normalize feature values for visualization
    scaler = StandardScaler()
    stock_features_scaled = pd.DataFrame(
        scaler.fit_transform(stock_features), 
        index=stock_features.index, 
        columns=stock_features.columns
    )
    
    # Plot feature heatmap
    plt.figure(figsize=(15, 8))
    sns.heatmap(stock_features_scaled[feature_importance.index[:10]], annot=True, cmap='coolwarm', center=0)
    plt.title('Feature Values for Sample Stocks (Top 10 Features by Importance)')
    plt.tight_layout()
    plt.show()
    
    # Demonstrate conviction analysis
    track_record = rs.get_track_record()
    
    if not track_record.empty and 'actual_return' in track_record.columns:
        # Group by conviction levels
        track_record['conviction_bin'] = pd.cut(track_record['conviction'], bins=[0, 0.6, 0.7, 0.8, 0.9, 1.0], 
                                               labels=['0.0-0.6', '0.6-0.7', '0.7-0.8', '0.8-0.9', '0.9-1.0'])
        
        # Calculate performance by conviction level
        conviction_performance = track_record.groupby('conviction_bin').agg({
            'predicted_return': 'mean',
            'actual_return': 'mean',
            'ticker': 'count'
        }).rename(columns={'ticker': 'count'})
        
        # Plot performance by conviction level
        plt.figure(figsize=(12, 6))
        conviction_performance[['predicted_return', 'actual_return']].plot(kind='bar')
        plt.title('Performance by Conviction Level')
        plt.xlabel('Conviction Level')
        plt.ylabel('Average Return')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        # Plot count by conviction level
        plt.figure(figsize=(10, 5))
        conviction_performance['count'].plot(kind='bar')
        plt.title('Number of Recommendations by Conviction Level')
        plt.xlabel('Conviction Level')
        plt.ylabel('Count')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    # Demonstrate counterfactual analysis for a selected stock
    if sample_stocks:
        selected_stock = sample_stocks[0]
        
        # Get original features
        original_features = features.loc[selected_stock].drop('future_return').copy()
        
        # Create counterfactual scenarios
        scenarios = []
        
        # Scenario 1: Increase recent returns
        scenario1 = original_features.copy()
        scenario1['return_5d'] += 0.05
        scenario1['return_10d'] += 0.03
        scenarios.append(('Increased Recent Returns', scenario1))
        
        # Scenario 2: Decrease volatility
        scenario2 = original_features.copy()
        scenario2['volatility_21d'] *= 0.7
        scenario2['volatility_63d'] *= 0.7
        scenarios.append(('Decreased Volatility', scenario2))
        
        # Scenario 3: Improve moving average indicator
        scenario3 = original_features.copy()
        scenario3['ma_cross_50_200'] += 0.02
        scenarios.append(('Improved MA Crossover', scenario3))
        
        # Calculate predictions for original and counterfactual scenarios
        predictions = []
        
        # Original prediction
        X_orig = original_features.values.reshape(1, -1)
        pred_return = rs.return_predictor.predict(X_orig)[0]
        pred_conviction = rs.conviction_model.predict_proba(X_orig)[0, 1]
        predictions.append(('Original', pred_return, pred_conviction))
        
        # Counterfactual predictions
        for name, scenario_features in scenarios:
            X_scenario = scenario_features.values.reshape(1, -1)
            scenario_return = rs.return_predictor.predict(X_scenario)[0]
            scenario_conviction = rs.conviction_model.predict_proba(X_scenario)[0, 1]
            predictions.append((name, scenario_return, scenario_conviction))
        
        # Plot counterfactual analysis
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot predicted returns
        names = [p[0] for p in predictions]
        returns = [p[1] for p in predictions]
        
        ax1.bar(names, returns)
        ax1.set_title(f'Counterfactual Analysis: Predicted Returns for {selected_stock}')
        ax1.set_ylabel('Predicted Return')
        ax1.grid(True, alpha=0.3)
        
        # Plot conviction
        convictions = [p[2] for p in predictions]
        
        ax2.bar(names, convictions)
        ax2.set_title(f'Counterfactual Analysis: Conviction Levels for {selected_stock}')
        ax2.set_ylabel('Conviction Level')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Print counterfactual analysis
        print(f"\nCounterfactual Analysis for {selected_stock}:")
        for name, pred_return, pred_conviction in predictions:
            print(f"{name}: Predicted Return = {pred_return:.4f}, Conviction = {pred_conviction:.4f}")

# Run additional analysis
analyze_explainable_ai_aspects(rs, prices, test_data)