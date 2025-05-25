import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import ndcg_score
from scipy.stats import kendalltau
import warnings
from tqdm import tqdm

# Suppress warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (12, 8)

class CrossSectionalMomentumStrategy:
    """
    Class implementing cross-sectional momentum strategy with different ranking methods:
    1. Classic momentum (Jegadeesh and Titman, 1993)
    2. MACD-based ranking (Baz et al., 2015)
    3. Regression-based ranking
    4. Learning to Rank (LTR) algorithms
    """
    
    def __init__(self, data_source='simulated', start_date='2010-01-01', end_date='2021-12-31'):
        """
        Initialize the strategy
        
        Parameters:
        -----------
        data_source : str
            Source of data ('simulated' or 'real')
        start_date : str
            Start date of backtest
        end_date : str
            End date of backtest
        """
        self.data_source = data_source
        self.start_date = start_date
        self.end_date = end_date
        
        # Data containers
        self.price_data = None
        self.returns_data = None
        self.features = None
        self.target_volatility = 0.15  # 15% target volatility as in the paper
        
        # Strategy parameters
        self.lookback_periods = {
            'short': 63,   # 3 months (21 trading days per month)
            'medium': 126,  # 6 months
            'long': 252,    # 12 months
        }
        self.holding_period = 21  # 1 month holding period
        self.n_stocks_long = 10   # Number of stocks in long portfolio
        self.n_stocks_short = 10  # Number of stocks in short portfolio
        
        # For the MACD indicators
        self.short_periods = [8, 16, 32]
        self.long_periods = [24, 48, 96]
        
        # Models
        self.models = {}
        
    def load_data(self):
        """Load or generate stock price data"""
        if self.data_source == 'simulated':
            self.generate_simulated_data()
        else:
            self.load_real_data()
            
    def generate_simulated_data(self, n_stocks=100, seed=42):
        """
        Generate simulated price data for n_stocks over n_days
        
        Parameters:
        -----------
        n_stocks : int
            Number of stocks to simulate
        seed : int
            Random seed for reproducibility
        """
        np.random.seed(seed)
        
        # Generate date range using the provided start and end dates
        start_date = pd.to_datetime(self.start_date)
        end_date = pd.to_datetime(self.end_date)
        # Generate business days (trading days) between start and end
        dates = pd.date_range(start=start_date, end=end_date, freq='B')
        n_days = len(dates)
        
        print(f"Generating data for {n_days} trading days from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        
        # Create dataframe to store prices
        self.price_data = pd.DataFrame(index=dates)
        
        # Generate market returns with some autocorrelation
        market_return = np.zeros(n_days)
        market_return[0] = np.random.normal(0.0005, 0.01)
        
        for i in range(1, n_days):
            market_return[i] = 0.0003 + 0.1 * market_return[i-1] + np.random.normal(0, 0.01)
        
        # Add some cyclicality
        market_cycle = 0.003 * np.sin(np.linspace(0, 8*np.pi, n_days))
        market_return += market_cycle
        
        # Cumulative market return
        market_price = 100 * np.cumprod(1 + market_return)
        self.price_data['MARKET'] = market_price
        
        # Generate 5 sectors with different characteristics
        sector_returns = np.zeros((5, n_days))
        sector_betas = np.random.uniform(0.7, 1.3, 5)
        
        for s in range(5):
            sector_returns[s, 0] = sector_betas[s] * market_return[0] + np.random.normal(0, 0.005)
            
            for i in range(1, n_days):
                sector_returns[s, i] = (
                    0.0001 +  # Small alpha
                    sector_betas[s] * market_return[i] +  # Beta * market
                    0.2 * sector_returns[s, i-1] +  # Autocorrelation
                    np.random.normal(0, 0.005)  # Idiosyncratic noise
                )
        
        # Assign stocks to sectors
        sectors = np.random.randint(0, 5, n_stocks)
        
        # Generate stock prices with momentum effects and sector exposure
        for i in range(n_stocks):
            ticker = f'STOCK{i+1:03d}'
            sector = sectors[i]
            
            # Stock specific parameters
            stock_beta = np.random.uniform(0.5, 1.5)  # Beta relative to sector
            momentum = np.random.uniform(-0.2, 0.2)   # Momentum parameter
            volatility = np.random.uniform(0.01, 0.03) # Stock volatility
            
            # Generate stock returns with momentum effect
            stock_return = np.zeros(n_days)
            stock_return[0] = stock_beta * sector_returns[sector, 0] + np.random.normal(0, volatility)
            
            for j in range(1, n_days):
                # Return with momentum factor
                if j > 20:  # Allow some history for momentum to work
                    past_return = np.mean(stock_return[j-20:j])
                    stock_return[j] = (
                        0.0001 +  # Small alpha
                        stock_beta * sector_returns[sector, j] +  # Sector beta
                        momentum * past_return +  # Momentum effect
                        np.random.normal(0, volatility)  # Noise
                    )
                else:
                    stock_return[j] = stock_beta * sector_returns[sector, j] + np.random.normal(0, volatility)
            
            # Create price series
            initial_price = np.random.uniform(10, 100)
            stock_price = initial_price * np.cumprod(1 + stock_return)
            
            # Add to dataframe
            self.price_data[ticker] = stock_price
        
        # Calculate returns
        self.returns_data = self.price_data.pct_change().dropna()
        
        print(f"Generated simulated price data for {n_stocks} stocks over {len(dates)} trading days")
    
    def load_real_data(self):
        """
        Load real price data from csv files or APIs
        This is a placeholder - in a real implementation you would fetch data from APIs or files
        """
        print("Loading real data is not implemented - using simulated data instead")
        self.generate_simulated_data()
    
    def calculate_features(self):
        """Calculate features for ranking models"""
        print("Calculating features...")
        
        # Initialize features dataframe
        self.features = pd.DataFrame(index=self.price_data.index)
        
        # Get stock tickers
        stock_tickers = [col for col in self.price_data.columns if col != 'MARKET']
        
        # Calculate rolling volatility for each stock (63-day window)
        volatility = {}
        for ticker in stock_tickers:
            volatility[ticker] = self.returns_data[ticker].rolling(window=63).std() * np.sqrt(252)  # Annualized
        
        # For each date, calculate features
        for date in tqdm(self.features.index[max(self.lookback_periods.values()):]):
            date_features = {}
            
            for ticker in stock_tickers:
                # Get historical data up to this date
                hist_returns = self.returns_data[ticker].loc[:date]
                
                # Feature 1-3: Raw returns over different lookback periods
                for period_name, period_length in self.lookback_periods.items():
                    if len(hist_returns) >= period_length:
                        period_return = hist_returns.iloc[-period_length:].sum()
                        date_features[(ticker, f'return_{period_name}')] = period_return
                        
                        # Volatility adjusted returns
                        if date in volatility[ticker].index:
                            vol = volatility[ticker].loc[date]
                            if not pd.isna(vol) and vol > 0:
                                date_features[(ticker, f'vol_adj_return_{period_name}')] = period_return / vol
                            else:
                                date_features[(ticker, f'vol_adj_return_{period_name}')] = np.nan
                        else:
                            date_features[(ticker, f'vol_adj_return_{period_name}')] = np.nan
                
                # Feature 4-12: MACD indicators with different parameters
                if len(hist_returns) >= 252:  # Need at least a year of data
                    for short_period in self.short_periods:
                        for long_period in self.long_periods:
                            # Calculate exponential moving averages
                            short_ema = hist_returns.iloc[-252:].ewm(span=short_period).mean()
                            long_ema = hist_returns.iloc[-252:].ewm(span=long_period).mean()
                            
                            # MACD is difference between short and long EMAs
                            macd = short_ema.iloc[-1] - long_ema.iloc[-1]
                            
                            # Normalize by volatility
                            if date in volatility[ticker].index:
                                vol = volatility[ticker].loc[date]
                                if not pd.isna(vol) and vol > 0:
                                    date_features[(ticker, f'macd_{short_period}_{long_period}')] = macd / vol
                                else:
                                    date_features[(ticker, f'macd_{short_period}_{long_period}')] = np.nan
                            else:
                                date_features[(ticker, f'macd_{short_period}_{long_period}')] = np.nan
            
            # Add features for this date
            if date_features:
                date_df = pd.DataFrame.from_dict(date_features, orient='index').T
                date_df.index = [date]
                
                # Concatenate with main features dataframe
                self.features = pd.concat([self.features, date_df])
        
        # Drop NaN rows
        self.features = self.features.dropna(how='all')
        
        print(f"Features calculated for {len(self.features)} dates")
    
    def prepare_training_data(self, train_start, train_end, feature_list=None):
        """
        Prepare training data for ranking models
        
        Parameters:
        -----------
        train_start : str
            Start date for training data
        train_end : str
            End date for training data
        feature_list : list, optional
            List of features to use. If None, use all features
            
        Returns:
        --------
        X_train : pd.DataFrame
            Features for training
        y_train : pd.DataFrame
            Target values (forward returns) for training
        """
        # Convert to datetime
        train_start = pd.to_datetime(train_start)
        train_end = pd.to_datetime(train_end)
        
        # Filter features for training period
        features_train = self.features.loc[train_start:train_end].copy()
        
        # Get stock tickers
        stock_tickers = [col for col in self.price_data.columns if col != 'MARKET']
        
        # Initialize lists to store training data
        X_data = []
        y_data = []
        
        # For each date in training period
        for date in features_train.index:
            # Get next date for forward returns
            try:
                date_loc = self.price_data.index.get_loc(date)
                next_date_idx = date_loc + self.holding_period
                if next_date_idx >= len(self.price_data.index):
                    continue
                next_date = self.price_data.index[next_date_idx]
                
                # Calculate forward returns for each stock
                forward_returns = {}
                for ticker in stock_tickers:
                    if date in self.price_data.index and next_date in self.price_data.index:
                        try:
                            if not pd.isna(self.price_data.loc[date, ticker]) and not pd.isna(self.price_data.loc[next_date, ticker]):
                                forward_return = (self.price_data.loc[next_date, ticker] / self.price_data.loc[date, ticker]) - 1
                                forward_returns[ticker] = forward_return
                        except:
                            continue
                
                # Get features for this date
                date_features = features_train.loc[date]
                
                # For each stock, extract features and target
                for ticker in stock_tickers:
                    if ticker in forward_returns:
                        # Extract features for this stock
                        stock_features = {}
                        for col in features_train.columns:
                            try:
                                # Handle multi-level column names
                                if isinstance(col, tuple) and col[0] == ticker:
                                    feature_name = col[1]
                                    if feature_list is None or feature_name in feature_list:
                                        stock_features[feature_name] = date_features[col]
                                elif ticker in col:  # Handle regular column names
                                    feature_name = col.replace(ticker + '_', '')
                                    if feature_list is None or feature_name in feature_list:
                                        stock_features[feature_name] = date_features[col]
                            except:
                                continue
                        
                        # If we have features, add to training data
                        if stock_features:
                            X_data.append(stock_features)
                            y_data.append(forward_returns[ticker])
            except:
                continue
        
        # Convert to DataFrames
        if X_data:
            X_train = pd.DataFrame(X_data)
            y_train = pd.Series(y_data)
            
            # Drop any columns with NaN values
            X_train = X_train.dropna(axis=1)
            
            # Drop any rows with NaN values
            non_nan_idx = ~X_train.isna().any(axis=1)
            X_train = X_train[non_nan_idx]
            y_train = y_train[non_nan_idx]
            
            return X_train, y_train
        else:
            return pd.DataFrame(), pd.Series()
    
    def train_models(self, train_start, train_end):
        """
        Train different ranking models
        
        Parameters:
        -----------
        train_start : str
            Start date for training data
        train_end : str
            End date for training data
        """
        print(f"Training models for period from {train_start} to {train_end}...")
        
        # Get training data
        X_train, y_train = self.prepare_training_data(train_start, train_end)
        
        if len(X_train) == 0 or len(y_train) == 0:
            print("No training data available. Make sure features are calculated and there's enough data.")
            return
        
        # Normalize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
        
        # 1. Classic momentum model - doesn't need training, it's rule-based
        
        # 2. MACD-based model - doesn't need training, it's rule-based
        
        # 3. Regression model - MLP Regressor
        print("Training MLP regression model...")
        mlp_model = MLPRegressor(
            hidden_layer_sizes=(50, 25), 
            activation='relu',
            alpha=0.0001,
            learning_rate='adaptive',
            early_stopping=True,
            validation_fraction=0.1,
            max_iter=200,
            random_state=42
        )
        mlp_model.fit(X_train_scaled, y_train)
        self.models['MLP'] = {'model': mlp_model, 'scaler': scaler, 'columns': X_train.columns}
        
        # 4. LambdaMART (using GradientBoostingRegressor as a proxy)
        print("Training LambdaMART (GBM) model...")
        gbm_model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=42
        )
        gbm_model.fit(X_train_scaled, y_train)
        self.models['LambdaMART'] = {'model': gbm_model, 'scaler': scaler, 'columns': X_train.columns}
        
        print("Models trained successfully")
    
    def calculate_classic_scores(self, date):
        """
        Calculate scores for classic momentum strategy
        
        Parameters:
        -----------
        date : datetime
            Date for which to calculate scores
            
        Returns:
        --------
        dict
            Dictionary of scores for each stock
        """
        # Get stock tickers
        stock_tickers = [col for col in self.price_data.columns if col != 'MARKET']
        
        # Get historical data up to this date
        hist_data = self.returns_data.loc[:date]
        
        # Calculate scores based on 12-month return
        scores = {}
        for ticker in stock_tickers:
            if ticker in hist_data.columns:
                # Get 12-month lookback
                lookback = min(self.lookback_periods['long'], len(hist_data))
                if lookback > 0:
                    # 12-month return as score
                    scores[ticker] = hist_data[ticker].iloc[-lookback:].sum()
        
        return scores
    
    def calculate_macd_scores(self, date):
        """
        Calculate scores for MACD-based strategy
        
        Parameters:
        -----------
        date : datetime
            Date for which to calculate scores
            
        Returns:
        --------
        dict
            Dictionary of scores for each stock
        """
        # Get stock tickers
        stock_tickers = [col for col in self.price_data.columns if col != 'MARKET']
        
        # Get historical data up to this date
        hist_data = self.returns_data.loc[:date]
        
        # Calculate 63-day volatility
        volatility = hist_data.rolling(window=63).std().loc[date]
        
        # Calculate average MACD score across different parameters
        scores = {}
        for ticker in stock_tickers:
            if ticker in hist_data.columns and ticker in volatility.index:
                macd_values = []
                
                for short_period in self.short_periods:
                    for long_period in self.long_periods:
                        # Calculate exponential moving averages
                        if len(hist_data[ticker]) >= 252:  # Need at least a year of data
                            returns = hist_data[ticker].iloc[-252:]  # Use last year of data
                            short_ema = returns.ewm(span=short_period).mean()
                            long_ema = returns.ewm(span=long_period).mean()
                            
                            # MACD is difference between short and long EMAs
                            macd = short_ema.iloc[-1] - long_ema.iloc[-1]
                            
                            # Normalize by volatility
                            vol = volatility[ticker]
                            if not pd.isna(vol) and vol > 0:
                                norm_macd = macd / vol
                                macd_values.append(norm_macd)
                
                # Average MACD values
                if macd_values:
                    scores[ticker] = np.mean(macd_values)
        
        return scores
    
    def calculate_model_scores(self, date, model_name):
        """
        Calculate scores using a trained model
        
        Parameters:
        -----------
        date : datetime
            Date for which to calculate scores
        model_name : str
            Name of the model to use
            
        Returns:
        --------
        dict
            Dictionary of scores for each stock
        """
        # Check if model exists
        if model_name not in self.models:
            print(f"Model {model_name} not found. Returning empty scores.")
            return {}
        
        # Get model and scaler
        model = self.models[model_name]['model']
        scaler = self.models[model_name]['scaler']
        columns = self.models[model_name]['columns']
        
        # Get stock tickers
        stock_tickers = [col for col in self.price_data.columns if col != 'MARKET']
        
        # Get features for this date
        if date not in self.features.index:
            return {}
        
        date_features = self.features.loc[date]
        
        # Calculate scores for each stock
        scores = {}
        X_predict = []
        tickers = []
        
        for ticker in stock_tickers:
            # Get features for this stock
            stock_features = {}
            for col in self.features.columns:
                try:
                    # Handle multi-level column names
                    if isinstance(col, tuple) and col[0] == ticker:
                        feature_name = col[1]
                        if feature_name in columns:
                            stock_features[feature_name] = date_features[col]
                    elif ticker in col:  # Handle regular column names
                        feature_name = col.replace(ticker + '_', '')
                        if feature_name in columns:
                            stock_features[feature_name] = date_features[col]
                except:
                    continue
            
            # Check if features are valid
            if len(stock_features) == len(columns) and not any(pd.isna(list(stock_features.values()))):
                # Make sure features are in the right order
                ordered_features = {col: stock_features.get(col, np.nan) for col in columns}
                X_predict.append(ordered_features)
                tickers.append(ticker)
        
        # Convert to DataFrame
        if X_predict:
            X_predict = pd.DataFrame(X_predict)
            
            # Normalize features
            X_predict_scaled = scaler.transform(X_predict)
            
            # Predict scores
            predicted_scores = model.predict(X_predict_scaled)
            
            # Create scores dictionary
            for i, ticker in enumerate(tickers):
                scores[ticker] = predicted_scores[i]
        
        return scores
    
    def backtest_strategy(self, start_date, end_date, method='classic'):
        """
        Backtest the cross-sectional momentum strategy
        
        Parameters:
        -----------
        start_date : str
            Start date for backtest
        end_date : str
            End date for backtest
        method : str
            Ranking method to use (classic, macd, MLP, LambdaMART)
            
        Returns:
        --------
        pd.Series
            Series of portfolio returns
        """
        print(f"Backtesting {method} strategy from {start_date} to {end_date}...")
        
        # Convert to datetime
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        
        # Get rebalance dates (month-end dates)
        rebalance_dates = pd.date_range(start=start_date, end=end_date, freq='M')
        
        # Filter dates to those in our data
        rebalance_dates = [date for date in rebalance_dates if date in self.price_data.index]
        
        # Initialize portfolio returns
        portfolio_returns = []
        portfolio_dates = []
        
        # For each rebalance date
        for i, date in enumerate(rebalance_dates[:-1]):
            # Calculate scores based on method
            if method == 'classic':
                scores = self.calculate_classic_scores(date)
            elif method == 'macd':
                scores = self.calculate_macd_scores(date)
            elif method in self.models:
                scores = self.calculate_model_scores(date, method)
            else:
                raise ValueError(f"Unknown method: {method}")
            
            # Skip if no scores
            if not scores:
                continue
            
            # Sort stocks by score
            sorted_stocks = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            
            # Select top and bottom stocks
            n_long = min(self.n_stocks_long, len(sorted_stocks))
            n_short = min(self.n_stocks_short, len(sorted_stocks))
            
            long_stocks = [stock for stock, _ in sorted_stocks[:n_long]]
            short_stocks = [stock for stock, _ in sorted_stocks[-n_short:]]
            
            # Skip if not enough stocks
            if len(long_stocks) == 0 or len(short_stocks) == 0:
                continue
            
            # Get next rebalance date
            next_date = rebalance_dates[i+1]
            
            # Calculate forward returns
            long_returns = []
            short_returns = []
            
            for ticker in long_stocks:
                if ticker in self.price_data.columns:
                    try:
                        fwd_return = (self.price_data.loc[next_date, ticker] / self.price_data.loc[date, ticker]) - 1
                        long_returns.append(fwd_return)
                    except:
                        continue
            
            for ticker in short_stocks:
                if ticker in self.price_data.columns:
                    try:
                        fwd_return = (self.price_data.loc[next_date, ticker] / self.price_data.loc[date, ticker]) - 1
                        short_returns.append(fwd_return)
                    except:
                        continue
            
            # Calculate portfolio return (long-short)
            if long_returns and short_returns:
                long_return = np.mean(long_returns)
                short_return = -np.mean(short_returns)  # Negative because we're shorting
                portfolio_return = (long_return + short_return) / 2  # Equal weight to long and short
                
                # Add to returns list
                portfolio_returns.append(portfolio_return)
                portfolio_dates.append(next_date)
        
        # Create Series of returns
        portfolio_returns = pd.Series(portfolio_returns, index=portfolio_dates)
        
        # Adjust to target volatility
        if len(portfolio_returns) > 0:
            realized_vol = portfolio_returns.std() * np.sqrt(12)  # Annualized
            if realized_vol > 0:
                scaling_factor = self.target_volatility / realized_vol
                portfolio_returns = portfolio_returns * scaling_factor
        
        print(f"Backtest completed with {len(portfolio_returns)} months of returns")
        return portfolio_returns
    
    def evaluate_rankings(self, method, n_periods=None):
        """
        Evaluate the ranking accuracy of a method
        
        Parameters:
        -----------
        method : str
            Ranking method to evaluate
        n_periods : int, optional
            Number of periods to evaluate. If None, use all available periods.
            
        Returns:
        --------
        dict
            Dictionary of evaluation metrics
        """
        # Get all dates with features
        feature_dates = self.features.index
        
        # Default to using half the available periods
        if n_periods is None:
            n_periods = min(20, max(1, len(feature_dates) // 2))
        
        # Make sure we have at least n_periods+1 dates
        if len(feature_dates) <= n_periods + 1:
            n_periods = max(1, len(feature_dates) - 2)
        
        # Get the last n_periods+1 dates with features
        dates = feature_dates[-n_periods-1:]
        
        # Initialize metrics
        kendall_tau_values = []
        ndcg_values = []
        
        # For each date (except the last one)
        for i in range(len(dates) - 1):
            date = dates[i]
            
            # Calculate scores based on method
            if method == 'classic':
                scores = self.calculate_classic_scores(date)
            elif method == 'macd':
                scores = self.calculate_macd_scores(date)
            elif method in self.models:
                scores = self.calculate_model_scores(date, method)
            else:
                raise ValueError(f"Unknown method: {method}")
            
            # Skip if no scores
            if not scores:
                continue
            
            # Get next date for forward returns
            next_date = dates[i+1]
            
            # Calculate actual forward returns
            actual_returns = {}
            for ticker in scores.keys():
                if ticker in self.price_data.columns:
                    try:
                        fwd_return = (self.price_data.loc[next_date, ticker] / self.price_data.loc[date, ticker]) - 1
                        actual_returns[ticker] = fwd_return
                    except:
                        continue
            
            # Skip if not enough returns
            if len(actual_returns) < 5:
                continue
            
            # Convert to lists for evaluation
            tickers = list(actual_returns.keys())
            pred_scores = np.array([scores[t] for t in tickers])
            true_scores = np.array([actual_returns[t] for t in tickers])
            
            # Calculate Kendall's Tau
            try:
                tau, _ = kendalltau(pred_scores, true_scores)
                if not pd.isna(tau):
                    kendall_tau_values.append(tau)
            except:
                pass
            
            # Calculate NDCG@10
            try:
                # For NDCG, need to sort true scores to get "ideal" ranking
                true_ranks = np.argsort(-true_scores)
                pred_ranks = np.argsort(-pred_scores)
                
                # Use scikit-learn's ndcg_score
                k = min(10, len(true_ranks))
                ndcg = ndcg_score([true_ranks[:k]], [pred_ranks[:k]], k=k)
                if not pd.isna(ndcg):
                    ndcg_values.append(ndcg)
            except:
                pass
        
        # Calculate average metrics
        avg_kendall_tau = np.mean(kendall_tau_values) if kendall_tau_values else np.nan
        avg_ndcg = np.mean(ndcg_values) if ndcg_values else np.nan
        
        return {
            'kendall_tau': avg_kendall_tau,
            'ndcg': avg_ndcg
        }
    
    def run_full_analysis(self):
        """Run full analysis of different methods"""
        # Load data
        self.load_data()
        
        # Calculate features
        self.calculate_features()
        
        # Define training and testing periods
        train_start_idx = max(self.lookback_periods.values())
        if train_start_idx < len(self.price_data.index):
            train_start = self.price_data.index[train_start_idx]
        else:
            train_start = self.price_data.index[0]
            
        # Use 2/3 of data for training, 1/3 for testing
        train_end_idx = int(len(self.price_data.index) * 2/3)
        train_end = self.price_data.index[train_end_idx]
        test_start = train_end
        test_end = self.price_data.index[-1]
        
        print(f"Training period: {train_start} to {train_end}")
        print(f"Testing period: {test_start} to {test_end}")
        
        # Train models
        self.train_models(train_start, train_end)
        
        # Backtest strategies
        results = {}
        methods = ['classic', 'macd']
        
        # Add trained models if available
        if 'MLP' in self.models:
            methods.append('MLP')
        if 'LambdaMART' in self.models:
            methods.append('LambdaMART')
        
        for method in methods:
            returns = self.backtest_strategy(test_start, test_end, method)
            results[method] = returns
        
        # Evaluate ranking accuracy
        ranking_metrics = {}
        for method in methods:
            metrics = self.evaluate_rankings(method)
            ranking_metrics[method] = metrics
        
        # Calculate performance metrics
        performance_metrics = {}
        for method, returns in results.items():
            # Skip if empty returns
            if len(returns) == 0:
                continue
                
            # Calculate metrics
            cumulative_return = (1 + returns).prod() - 1
            annualized_return = (1 + cumulative_return) ** (12 / len(returns)) - 1
            volatility = returns.std() * np.sqrt(12)  # Annualized
            sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
            
            # Calculate drawdowns
            cum_returns = (1 + returns).cumprod()
            peak = cum_returns.cummax()
            drawdowns = (cum_returns / peak) - 1
            max_drawdown = drawdowns.min()
            
            performance_metrics[method] = {
                'cumulative_return': cumulative_return,
                'annualized_return': annualized_return,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'win_rate': (returns > 0).mean()
            }
        
        # Plot results
        self.plot_results(results, performance_metrics, ranking_metrics)
        
        return results, performance_metrics, ranking_metrics
    
    def plot_results(self, returns_dict, performance_metrics, ranking_metrics):
        """
        Plot backtest results
        
        Parameters:
        -----------
        returns_dict : dict
            Dictionary of returns for each method
        performance_metrics : dict
            Dictionary of performance metrics for each method
        ranking_metrics : dict
            Dictionary of ranking metrics for each method
        """
        # Skip if no returns
        if not returns_dict or all(len(returns) == 0 for returns in returns_dict.values()):
            print("No returns to plot.")
            return
            
        # Create figure with 2 rows and 2 columns
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Cumulative returns
        ax = axes[0, 0]
        for method, returns in returns_dict.items():
            if len(returns) > 0:
                cumulative_returns = (1 + returns).cumprod()
                ax.plot(cumulative_returns.index, cumulative_returns, label=method)
        
        ax.set_title('Cumulative Returns')
        ax.set_ylabel('Value')
        ax.legend()
        ax.grid(True)
        
        # 2. Performance metrics comparison
        ax = axes[0, 1]
        
        metrics_to_plot = ['annualized_return', 'sharpe_ratio']
        methods = list(performance_metrics.keys())
        
        if methods:
            x = np.arange(len(methods))
            width = 0.35
            
            for i, metric in enumerate(metrics_to_plot):
                values = [performance_metrics[method][metric] for method in methods]
                ax.bar(x + i*width, values, width, label=metric)
            
            ax.set_title('Performance Metrics')
            ax.set_xticks(x + width/2)
            ax.set_xticklabels(methods)
            ax.legend()
            ax.grid(True)
        else:
            ax.text(0.5, 0.5, "No performance metrics available", 
                    horizontalalignment='center', verticalalignment='center')
        
        # 3. Ranking metrics comparison
        ax = axes[1, 0]
        
        metrics_to_plot = ['kendall_tau', 'ndcg']
        methods = list(ranking_metrics.keys())
        
        if methods and any(not pd.isna(ranking_metrics[m][metric]) for m in methods for metric in metrics_to_plot):
            x = np.arange(len(methods))
            width = 0.35
            
            for i, metric in enumerate(metrics_to_plot):
                values = [ranking_metrics[method][metric] if not pd.isna(ranking_metrics[method][metric]) else 0 
                          for method in methods]
                ax.bar(x + i*width, values, width, label=metric)
            
            ax.set_title('Ranking Metrics')
            ax.set_xticks(x + width/2)
            ax.set_xticklabels(methods)
            ax.legend()
            ax.grid(True)
        else:
            ax.text(0.5, 0.5, "No ranking metrics available", 
                    horizontalalignment='center', verticalalignment='center')
        
        # 4. Performance metrics table
        ax = axes[1, 1]
        ax.axis('tight')
        ax.axis('off')
        
        # Create data for table
        table_data = []
        metrics = ['annualized_return', 'volatility', 'sharpe_ratio', 'max_drawdown', 'win_rate']
        
        # Add header
        header = ['Metric'] + list(performance_metrics.keys())
        table_data.append(header)
        
        # Add rows
        for metric in metrics:
            row = [metric]
            for method in performance_metrics.keys():
                value = performance_metrics[method][metric]
                if metric in ['annualized_return', 'volatility', 'max_drawdown', 'win_rate']:
                    row.append(f'{value:.2%}')
                else:
                    row.append(f'{value:.2f}')
            table_data.append(row)
        
        # Create table
        if len(table_data) > 1:
            table = ax.table(cellText=table_data[1:], colLabels=table_data[0], loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1.2, 1.2)
            
            # Add title
            ax.set_title('Performance Metrics Summary')
        else:
            ax.text(0.5, 0.5, "No performance metrics available", 
                    horizontalalignment='center', verticalalignment='center')
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig('cross_sectional_momentum_results.png', dpi=300, bbox_inches='tight')
        plt.show()


# Run the strategy
if __name__ == "__main__":
    # Initialize strategy
    strategy = CrossSectionalMomentumStrategy(
        data_source='simulated',
        start_date='2010-01-01',
        end_date='2021-12-31'
    )
    
    # Run analysis
    results, performance_metrics, ranking_metrics = strategy.run_full_analysis()
    
    # Print summary
    print("\nPerformance Metrics Summary:")
    print("============================")
    
    for method, metrics in performance_metrics.items():
        print(f"\n{method}:")
        print(f"  Annualized Return: {metrics['annualized_return']:.2%}")
        print(f"  Volatility: {metrics['volatility']:.2%}")
        print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        print(f"  Max Drawdown: {metrics['max_drawdown']:.2%}")
        print(f"  Win Rate: {metrics['win_rate']:.2%}")
    
    print("\nRanking Metrics Summary:")
    print("=======================")
    
    for method, metrics in ranking_metrics.items():
        print(f"\n{method}:")
        print(f"  Kendall's Tau: {metrics['kendall_tau']:.4f}")
        print(f"  NDCG@10: {metrics['ndcg']:.4f}")