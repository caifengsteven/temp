import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import blpapi
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')
import logging
import os
from scipy.optimize import nnls

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BloombergDataFetcher:
    """Class to fetch data from Bloomberg"""
    
    def __init__(self, host="localhost", port=8194):
        self.host = host
        self.port = port
        self.session = None
        
    def start_session(self):
        """Start a Bloomberg API session"""
        logger.info("Starting Bloomberg API session...")
        
        session_options = blpapi.SessionOptions()
        session_options.setServerHost(self.host)
        session_options.setServerPort(self.port)
        
        # Create a session
        self.session = blpapi.Session(session_options)
        
        # Start the session
        if not self.session.start():
            logger.error("Failed to start session.")
            return False
        
        logger.info("Session started successfully.")
        return True
    
    def stop_session(self):
        """Stop the Bloomberg API session"""
        if self.session:
            self.session.stop()
            logger.info("Session stopped.")
    
    def get_historical_data(self, securities, fields, start_date, end_date, periodicity="MONTHLY"):
        """Get historical data from Bloomberg"""
        if not self.session:
            logger.error("No active session.")
            return None
        
        # Open the reference data service
        if not self.session.openService("//blp/refdata"):
            logger.error("Failed to open reference data service.")
            return None
        
        refDataService = self.session.getService("//blp/refdata")
        
        # Create the request
        request = refDataService.createRequest("HistoricalDataRequest")
        
        # Add securities
        for security in securities:
            request.append("securities", security)
        
        # Add fields
        for field in fields:
            request.append("fields", field)
        
        # Set date range
        request.set("startDate", start_date)
        request.set("endDate", end_date)
        request.set("periodicitySelection", periodicity)
        
        logger.info(f"Requesting data for {len(securities)} securities from {start_date} to {end_date}")
        
        # Send the request
        self.session.sendRequest(request)
        
        # Process the response
        data_by_security = {}
        
        while True:
            event = self.session.nextEvent(500)
            
            for msg in event:
                if msg.messageType() == blpapi.Name("HistoricalDataResponse"):
                    security_data = msg.getElement("securityData")
                    security = security_data.getElementAsString("security")
                    
                    # Check for field data
                    if security_data.hasElement("fieldData"):
                        field_data = security_data.getElement("fieldData")
                        
                        data_points = []
                        
                        # Process each data point
                        for i in range(field_data.numValues()):
                            point = field_data.getValue(i)
                            
                            # Create a dict for this data point
                            data_point = {}
                            
                            # Extract date
                            if point.hasElement("date"):
                                date = point.getElementAsDatetime("date")
                                data_point["date"] = date.strftime("%Y-%m-%d")
                            
                            # Extract fields
                            for field in fields:
                                if point.hasElement(field):
                                    data_point[field] = point.getElementAsFloat(field)
                                else:
                                    data_point[field] = None
                            
                            data_points.append(data_point)
                        
                        # Convert to DataFrame
                        if data_points:
                            df = pd.DataFrame(data_points)
                            df["date"] = pd.to_datetime(df["date"])
                            df = df.set_index("date")
                            data_by_security[security] = df
            
            # Check if we've received all events
            if event.eventType() == blpapi.Event.RESPONSE:
                break
        
        return data_by_security
    
    def get_nber_recession_data(self):
        """Get NBER recession data from FRED via Bloomberg"""
        securities = ["USREC Index"]
        fields = ["PX_LAST"]
        start_date = "19850101"
        end_date = dt.datetime.now().strftime("%Y%m%d")
        
        data = self.get_historical_data(securities, fields, start_date, end_date, "MONTHLY")
        
        if securities[0] in data:
            return data[securities[0]]
        else:
            return None

# Custom constrained regression classes
class ConstrainedRidge:
    def __init__(self, alpha=1.0, max_iter=1000, random_state=None):
        self.alpha = alpha
        self.max_iter = max_iter
        self.random_state = random_state
        self.coef_ = None
        self.intercept_ = 0
        
    def fit(self, X, y):
        # Store mean of X and y for intercept calculation
        self.X_mean_ = X.mean(axis=0)
        self.y_mean_ = y.mean()
        
        # Center X and y
        X_centered = X - self.X_mean_
        y_centered = y - self.y_mean_
        
        # Convert to numpy arrays if needed
        if isinstance(X_centered, pd.DataFrame):
            X_centered = X_centered.values
        if isinstance(y_centered, pd.Series):
            y_centered = y_centered.values
        
        # Solve non-negative least squares problem with penalty
        n_samples, n_features = X_centered.shape
        
        # Create augmented design matrix for ridge penalty
        X_augmented = np.vstack((X_centered, np.sqrt(self.alpha) * np.eye(n_features)))
        y_augmented = np.hstack((y_centered, np.zeros(n_features)))
        
        # Solve non-negative least squares problem
        self.coef_, _ = nnls(X_augmented, y_augmented)
        
        # Calculate intercept
        self.intercept_ = self.y_mean_ - np.dot(self.X_mean_, self.coef_)
        
        return self
    
    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.values
        return np.dot(X, self.coef_) + self.intercept_

class ConstrainedElasticNet:
    def __init__(self, alpha=1.0, l1_ratio=0.5, max_iter=1000, random_state=None):
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.max_iter = max_iter
        self.random_state = random_state
        self.coef_ = None
        self.intercept_ = 0
        
    def fit(self, X, y):
        # We'll use scikit-learn's Lasso with positive constraint
        # since it supports the positive parameter
        self.model = Lasso(
            alpha=self.alpha * self.l1_ratio,
            fit_intercept=True,
            max_iter=self.max_iter,
            random_state=self.random_state,
            positive=True
        )
        
        self.model.fit(X, y)
        
        # Extract coefficients and intercept
        self.coef_ = self.model.coef_
        self.intercept_ = self.model.intercept_
        
        return self
    
    def predict(self, X):
        return self.model.predict(X)

class OilPriceForecaster:
    """Class to forecast oil prices using technical indicators with economic constraints"""
    
    def __init__(self, initial_training_end="2004-12-31"):
        self.initial_training_end = initial_training_end
        self.data = None
        self.recession_data = None
        self.results = {}
    
    def load_data(self, oil_data=None, recession_data=None):
        """Load data either from provided DataFrames or fetch from Bloomberg"""
        if oil_data is not None:
            self.data = oil_data
        
        if recession_data is not None:
            self.recession_data = recession_data
        
        # If data is not provided, fetch it from Bloomberg
        if self.data is None or self.recession_data is None:
            fetcher = BloombergDataFetcher()
            
            try:
                if not fetcher.start_session():
                    logger.error("Failed to start Bloomberg session. Using sample data.")
                    self._generate_sample_data()
                    return
                
                # Fetch WTI crude oil price data
                securities = ["CL1 Comdty"]  # Front-month WTI Crude Oil futures
                fields = ["PX_LAST", "PX_VOLUME"]
                start_date = "19860101"  # Start from January 1986 as in the paper
                end_date = dt.datetime.now().strftime("%Y%m%d")
                
                oil_data = fetcher.get_historical_data(securities, fields, start_date, end_date)
                
                if securities[0] in oil_data:
                    self.data = oil_data[securities[0]].copy()
                    self.data.columns = ['price', 'volume']
                    
                    # Get CPI data to calculate real prices
                    cpi_data = fetcher.get_historical_data(["CPI Index"], ["PX_LAST"], start_date, end_date)
                    if "CPI Index" in cpi_data:
                        cpi = cpi_data["CPI Index"].copy()
                        cpi.columns = ['cpi']
                        
                        # Join the data
                        self.data = self.data.join(cpi, how='left')
                        
                        # Forward fill any missing CPI values
                        self.data['cpi'] = self.data['cpi'].fillna(method='ffill')
                        
                        # Calculate real prices
                        recent_cpi = self.data['cpi'].iloc[-1]
                        self.data['real_price'] = self.data['price'] * recent_cpi / self.data['cpi']
                    else:
                        # If CPI data not available, just use nominal prices
                        self.data['real_price'] = self.data['price']
                
                # Get NBER recession data
                self.recession_data = fetcher.get_nber_recession_data()
                
            except Exception as e:
                logger.error(f"Error fetching data: {e}")
                self._generate_sample_data()
            
            finally:
                fetcher.stop_session()
        
        # Prepare the data
        self._prepare_data()
    
    def _generate_sample_data(self):
        """Generate sample data for testing when Bloomberg is not available"""
        # Create sample date range
        index = pd.date_range(start='1986-01-01', end='2022-12-31', freq='MS')
        
        # Generate sample price data with some trend and seasonality
        np.random.seed(42)
        n = len(index)
        trend = np.linspace(20, 80, n) + np.random.normal(0, 5, n)  # Upward trend with noise
        seasonal = 5 * np.sin(np.linspace(0, 24*np.pi, n))  # Seasonal component
        noise = np.random.normal(0, 10, n)  # Random noise
        
        price = trend + seasonal + noise
        
        # Generate sample volume data
        volume = 10000 + 5000 * np.random.random(n)
        
        # Create sample dataframes
        self.data = pd.DataFrame({
            'price': price,
            'volume': volume,
            'real_price': price
        }, index=index)
        
        # Create sample recession data (approximately every 5-7 years)
        recession = np.zeros(n)
        recession[36:42] = 1  # 1989 recession
        recession[80:86] = 1  # 1993 recession
        recession[144:150] = 1  # 2001 recession
        recession[228:240] = 1  # 2008-2009 recession
        recession[362:368] = 1  # 2020 COVID recession
        
        self.recession_data = pd.DataFrame({
            'PX_LAST': recession
        }, index=index)
        
        logger.info("Generated sample data for testing")
    
    def _prepare_data(self):
        """Prepare the data for analysis"""
        # Calculate log returns
        self.data['log_price'] = np.log(self.data['real_price'])
        self.data['log_return'] = self.data['log_price'].diff()
        
        # Drop missing values
        self.data = self.data.dropna()
        
        # Generate technical indicators
        self._generate_technical_indicators()
    
    def _generate_technical_indicators(self):
        """Generate the technical indicators used in the paper"""
        # 1. Moving Average (MA) indicators
        short_mas = [1, 2, 3]
        long_mas = [9, 12]
        
        for s in short_mas:
            for l in long_mas:
                # Calculate short and long moving averages
                self.data[f'MA{s}'] = self.data['real_price'].rolling(window=s).mean()
                self.data[f'MA{l}'] = self.data['real_price'].rolling(window=l).mean()
                
                # Generate signal: 1 if short MA >= long MA, 0 otherwise
                self.data[f'MA({s},{l})'] = (self.data[f'MA{s}'] >= self.data[f'MA{l}']).astype(int)
        
        # 2. Momentum (MOM) indicators
        for j in [1, 2, 3, 6, 9, 12]:
            # Generate signal: 1 if current price >= price j months ago, 0 otherwise
            self.data[f'MOM({j})'] = (self.data['real_price'] >= self.data['real_price'].shift(j)).astype(int)
        
        # 3. On-Balance Volume (OBV) indicators
        # Calculate OBV
        self.data['price_diff'] = self.data['real_price'].diff()
        self.data['vol_direction'] = np.where(self.data['price_diff'] >= 0, 1, -1)
        self.data['vol_contribution'] = self.data['volume'] * self.data['vol_direction']
        self.data['OBV'] = self.data['vol_contribution'].cumsum()
        
        for s in short_mas:
            for l in long_mas:
                # Calculate short and long OBV moving averages
                self.data[f'OBV_MA{s}'] = self.data['OBV'].rolling(window=s).mean()
                self.data[f'OBV_MA{l}'] = self.data['OBV'].rolling(window=l).mean()
                
                # Generate signal: 1 if short OBV MA >= long OBV MA, 0 otherwise
                self.data[f'VOL({s},{l})'] = (self.data[f'OBV_MA{s}'] >= self.data[f'OBV_MA{l}']).astype(int)
        
        # Drop intermediate columns
        columns_to_drop = [f'MA{i}' for i in short_mas + long_mas] + ['price_diff', 'vol_direction', 'vol_contribution', 'OBV'] + [f'OBV_MA{i}' for i in short_mas + long_mas]
        self.data = self.data.drop(columns=columns_to_drop, errors='ignore')
        
        # List all the technical indicators (18 in total)
        self.tech_indicators = []
        
        # MA indicators (6)
        for s in short_mas:
            for l in long_mas:
                self.tech_indicators.append(f'MA({s},{l})')
        
        # MOM indicators (6)
        for j in [1, 2, 3, 6, 9, 12]:
            self.tech_indicators.append(f'MOM({j})')
        
        # VOL indicators (6)
        for s in short_mas:
            for l in long_mas:
                self.tech_indicators.append(f'VOL({s},{l})')
    
    def _cross_validate_shrinkage_params(self, X_train, y_train, model_type='Lasso', pct_validation=0.6):
        """Determine optimal shrinkage parameters through cross-validation"""
        # Split the training data into two parts
        split_idx = int(len(X_train) * pct_validation)
        X_train_cv = X_train.iloc[:split_idx]
        y_train_cv = y_train.iloc[:split_idx]
        X_val = X_train.iloc[split_idx:]
        y_val = y_train.iloc[split_idx:]
        
        # Define parameter grid based on model type
        if model_type == 'Lasso':
            param_grid = {'alpha': np.logspace(-5, 1, 20)}
            model = Lasso(max_iter=10000, random_state=42)
        elif model_type == 'Ridge':
            param_grid = {'alpha': np.logspace(-5, 1, 20)}
            model = Ridge(max_iter=10000, random_state=42)
        elif model_type == 'ElasticNet':
            param_grid = {
                'alpha': np.logspace(-5, 1, 10),
                'l1_ratio': np.linspace(0.1, 0.9, 9)
            }
            model = ElasticNet(max_iter=10000, random_state=42)
        
        # Cross-validation
        cv = TimeSeriesSplit(n_splits=5)
        grid_search = GridSearchCV(model, param_grid, cv=cv, scoring='neg_mean_squared_error')
        grid_search.fit(X_train_cv, y_train_cv)
        
        # Get best parameters
        best_params = grid_search.best_params_
        
        return best_params
    
    def _build_shrinkage_model(self, model_type, params, constrained=False):
        """Build a shrinkage regression model with given parameters"""
        if model_type == 'Lasso':
            model = Lasso(alpha=params['alpha'], max_iter=10000, random_state=42, positive=constrained)
        elif model_type == 'Ridge':
            if constrained:
                # Use our custom constrained Ridge class
                model = ConstrainedRidge(alpha=params['alpha'], max_iter=10000, random_state=42)
            else:
                model = Ridge(alpha=params['alpha'], max_iter=10000, random_state=42)
        elif model_type == 'ElasticNet':
            if constrained:
                # Use our custom constrained ElasticNet class
                model = ConstrainedElasticNet(
                    alpha=params['alpha'], 
                    l1_ratio=params.get('l1_ratio', 0.5),
                    max_iter=10000, 
                    random_state=42
                )
            else:
                model = ElasticNet(
                    alpha=params['alpha'], 
                    l1_ratio=params.get('l1_ratio', 0.5),
                    max_iter=10000, 
                    random_state=42
                )
        
        return model
    
    def forecast_oil_prices(self, methods=None, constrained_methods=None):
        """Forecast oil prices using various shrinkage methods with and without constraints"""
        # Default methods to use
        if methods is None:
            methods = ['Lasso', 'Ridge', 'ElasticNet']
            
        if constrained_methods is None:
            constrained_methods = ['Lasso-con', 'Ridge-con', 'ElasticNet-con']
        
        # Split data into training and testing periods
        train_data = self.data[self.data.index <= self.initial_training_end].copy()
        test_data = self.data[self.data.index > self.initial_training_end].copy()
        
        logger.info(f"Training data: {train_data.index.min()} to {train_data.index.max()}")
        logger.info(f"Testing data: {test_data.index.min()} to {test_data.index.max()}")
        
        # Prepare features and target
        X_initial = train_data[self.tech_indicators]
        y_initial = train_data['log_return']
        
        # Initialize results dictionaries
        forecasts = {}
        models = {}
        
        # For storing NBER recession data during the test period
        if self.recession_data is not None:
            recession_indicators = self.recession_data.loc[test_data.index].copy()
        else:
            recession_indicators = pd.DataFrame(index=test_data.index, data={'PX_LAST': np.zeros(len(test_data))})
        
        # Recursive (expanding window) forecasting
        for t in test_data.index:
            # Update training data up to time t-1
            train_data_t = self.data[self.data.index < t].copy()
            
            # Skip if not enough data
            if len(train_data_t) <= 50:
                continue
            
            # Extract features and target
            X_train = train_data_t[self.tech_indicators]
            y_train = train_data_t['log_return']
            
            # For each method
            for method in methods + constrained_methods:
                if method not in forecasts:
                    forecasts[method] = []
                    models[method] = []
                
                # Determine model type and whether it's constrained
                if '-con' in method:
                    model_type = method.split('-')[0]
                    constrained = True
                else:
                    model_type = method
                    constrained = False
                
                # Cross-validate to find optimal parameters
                best_params = self._cross_validate_shrinkage_params(X_train, y_train, model_type)
                
                # Build model
                model = self._build_shrinkage_model(model_type, best_params, constrained)
                
                # Fit model
                model.fit(X_train, y_train)
                
                # Store model coefficients
                models[method].append({
                    'date': t,
                    'coef': dict(zip(self.tech_indicators, model.coef_)),
                    'intercept': model.intercept_
                })
                
                # Make prediction
                X_t = test_data.loc[[t], self.tech_indicators]
                y_pred = model.predict(X_t)[0]
                
                # Store prediction
                forecasts[method].append({
                    'date': t,
                    'actual': test_data.loc[t, 'log_return'],
                    'forecast': y_pred
                })
        
        # Convert forecasts to DataFrames
        for method in forecasts:
            df = pd.DataFrame(forecasts[method])
            df = df.set_index('date')
            
            # Add recession indicator
            df['recession'] = recession_indicators['PX_LAST']
            
            self.results[method] = df
        
        # Convert model coefficients to DataFrames
        for method in models:
            df = pd.DataFrame(models[method])
            df = df.set_index('date')
            
            # Expand coefficients
            for indicator in self.tech_indicators:
                df[indicator] = df['coef'].apply(lambda x: x.get(indicator, 0))
            
            df = df.drop(columns=['coef'])
            
            # Store model info
            self.results[f"{method}_models"] = df
        
        # Calculate performance metrics
        self._calculate_performance_metrics()
        
        return self.results
    
    def _calculate_performance_metrics(self):
        """Calculate performance metrics for each forecasting method"""
        for method in [m for m in self.results.keys() if not m.endswith('_models')]:
            df = self.results[method]
            
            # No-change forecast (benchmark)
            df['no_change'] = 0
            
            # Calculate squared errors
            df['se_model'] = (df['actual'] - df['forecast'])**2
            df['se_benchmark'] = (df['actual'] - df['no_change'])**2
            
            # Calculate cumulative sum of squared error differences
            df['cssed'] = (df['se_benchmark'] - df['se_model']).cumsum()
            
            # Calculate MSPEs
            mspe_model = df['se_model'].mean()
            mspe_benchmark = df['se_benchmark'].mean()
            
            # Calculate out-of-sample R^2
            r2_os = 100 * (1 - mspe_model / mspe_benchmark)
            
            # Calculate metrics during recessions and expansions
            recession_df = df[df['recession'] == 1]
            expansion_df = df[df['recession'] == 0]
            
            if not recession_df.empty:
                mspe_model_rec = recession_df['se_model'].mean()
                mspe_benchmark_rec = recession_df['se_benchmark'].mean()
                r2_os_rec = 100 * (1 - mspe_model_rec / mspe_benchmark_rec)
            else:
                r2_os_rec = np.nan
            
            if not expansion_df.empty:
                mspe_model_exp = expansion_df['se_model'].mean()
                mspe_benchmark_exp = expansion_df['se_benchmark'].mean()
                r2_os_exp = 100 * (1 - mspe_model_exp / mspe_benchmark_exp)
            else:
                r2_os_exp = np.nan
            
            # Store metrics
            df.attrs['r2_os'] = r2_os
            df.attrs['r2_os_rec'] = r2_os_rec
            df.attrs['r2_os_exp'] = r2_os_exp
            
            # Update results
            self.results[method] = df
    
    def backtest_trading_strategy(self, risk_free_rate=0.02/12, risk_aversion=3, leverage=1):
        """Backtest a trading strategy based on the forecasts"""
        for method in [m for m in self.results.keys() if not m.endswith('_models')]:
            df = self.results[method].copy()
            
            # Calculate strategy returns
            # If forecast > 0, go long; if forecast < 0, go short
            df['position'] = np.sign(df['forecast'])
            df['strategy_return'] = df['position'] * df['actual']
            
            # Market timing signals
            df['timing_signal'] = np.where(df['forecast'] > 0, df['actual'], -df['actual'])
            
            # Calculate portfolio returns with leverage
            df['portfolio_return'] = df['strategy_return'] * leverage + (1 - leverage) * risk_free_rate
            
            # Calculate performance metrics
            avg_return = df['strategy_return'].mean() * 12  # Annualized
            std_dev = df['strategy_return'].std() * np.sqrt(12)  # Annualized
            sharpe = avg_return / std_dev if std_dev > 0 else 0
            
            # Calculate certainty equivalent return (CER)
            cer = df['portfolio_return'].mean() * 12 - 0.5 * risk_aversion * (df['portfolio_return'].std() * np.sqrt(12))**2
            
            # Calculate win rate
            win_rate = (df['strategy_return'] > 0).mean()
            
            # Store metrics
            df.attrs['avg_return'] = avg_return
            df.attrs['std_dev'] = std_dev
            df.attrs['sharpe'] = sharpe
            df.attrs['cer'] = cer
            df.attrs['win_rate'] = win_rate
            
            # Update results
            self.results[method] = df
    
    def plot_coefficients(self):
        """Plot the coefficients of the shrinkage methods"""
        methods = [m for m in self.results.keys() if m.endswith('_models')]
        
        # Get the most recent model coefficients for each method
        latest_coeffs = {}
        for method in methods:
            latest_coeffs[method] = self.results[method].iloc[-1][self.tech_indicators]
        
        # Plot the coefficients
        fig, axes = plt.subplots(3, 1, figsize=(12, 18))
        
        # Pairs of standard and constrained methods
        method_pairs = [
            ('Lasso_models', 'Lasso-con_models'),
            ('Ridge_models', 'Ridge-con_models'),
            ('ElasticNet_models', 'ElasticNet-con_models')
        ]
        
        for i, (std_method, con_method) in enumerate(method_pairs):
            if std_method in latest_coeffs and con_method in latest_coeffs:
                std_coef = latest_coeffs[std_method]
                con_coef = latest_coeffs[con_method]
                
                x = np.arange(len(self.tech_indicators))
                width = 0.35
                
                axes[i].bar(x - width/2, std_coef, width, label=std_method.split('_')[0])
                axes[i].bar(x + width/2, con_coef, width, label=con_method.split('_')[0])
                
                axes[i].set_xticks(x)
                axes[i].set_xticklabels(self.tech_indicators, rotation=45, ha='right')
                axes[i].set_ylabel('Coefficient Value')
                axes[i].set_title(f'{std_method.split("_")[0]} vs {con_method.split("_")[0]} Coefficients')
                axes[i].legend()
                axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig("oil_price_model_coefficients.png")
        plt.close()
    
    def plot_forecast_performance(self):
        """Plot the forecast performance of each method"""
        methods = [m for m in self.results.keys() if not m.endswith('_models')]
        
        # Plot CSSED
        plt.figure(figsize=(12, 8))
        
        for method in methods:
            df = self.results[method]
            plt.plot(df.index, df['cssed'], label=method)
        
        # Add recession shading
        if self.recession_data is not None:
            recession_data = self.recession_data.loc[self.results[methods[0]].index]
            for i in range(len(recession_data)):
                if recession_data.iloc[i]['PX_LAST'] == 1:
                    plt.axvspan(recession_data.index[i], recession_data.index[i], color='gray', alpha=0.3)
        
        plt.axhline(y=0, color='k', linestyle='--')
        plt.title('Cumulative Sum of Squared Error Differences (CSSED)')
        plt.xlabel('Date')
        plt.ylabel('CSSED')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("oil_price_forecast_cssed.png")
        plt.close()
        
        # Plot R^2_OS values
        r2_values = {method: self.results[method].attrs['r2_os'] for method in methods}
        r2_rec_values = {method: self.results[method].attrs['r2_os_rec'] for method in methods}
        r2_exp_values = {method: self.results[method].attrs['r2_os_exp'] for method in methods}
        
        plt.figure(figsize=(12, 8))
        x = np.arange(len(methods))
        width = 0.25
        
        plt.bar(x - width, [r2_values[m] for m in methods], width, label='Full Sample')
        plt.bar(x, [r2_rec_values[m] for m in methods], width, label='Recessions')
        plt.bar(x + width, [r2_exp_values[m] for m in methods], width, label='Expansions')
        
        plt.axhline(y=0, color='k', linestyle='--')
        plt.title('Out-of-Sample R^2 (R^2_OS)')
        plt.xticks(x, methods)
        plt.ylabel('R^2_OS (%)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("oil_price_forecast_r2os.png")
        plt.close()
    
    def plot_trading_performance(self):
        """Plot the trading performance of each method"""
        methods = [m for m in self.results.keys() if not m.endswith('_models')]
        
        # Plot cumulative returns
        plt.figure(figsize=(12, 8))
        
        for method in methods:
            df = self.results[method]
            plt.plot(df.index, (1 + df['strategy_return']).cumprod() - 1, label=method)
        
        # Add benchmark (buy and hold)
        benchmark_return = (1 + self.results[methods[0]]['actual']).cumprod() - 1
        plt.plot(benchmark_return.index, benchmark_return, label='Buy and Hold', linestyle='--', color='k')
        
        # Add recession shading
        if self.recession_data is not None:
            recession_data = self.recession_data.loc[self.results[methods[0]].index]
            for i in range(len(recession_data)):
                if recession_data.iloc[i]['PX_LAST'] == 1:
                    plt.axvspan(recession_data.index[i], recession_data.index[i], color='gray', alpha=0.3)
        
        plt.title('Cumulative Returns')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Return')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("oil_price_trading_returns.png")
        plt.close()
        
        # Plot CER values
        cer_values = {method: self.results[method].attrs['cer'] for method in methods}
        
        plt.figure(figsize=(12, 6))
        x = np.arange(len(methods))
        
        plt.bar(x, [cer_values[m] for m in methods])
        
        plt.title('Certainty Equivalent Return (CER)')
        plt.xticks(x, methods)
        plt.ylabel('CER')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("oil_price_trading_cer.png")
        plt.close()
        
        # Plot Sharpe ratios
        sharpe_values = {method: self.results[method].attrs['sharpe'] for method in methods}
        
        plt.figure(figsize=(12, 6))
        x = np.arange(len(methods))
        
        plt.bar(x, [sharpe_values[m] for m in methods])
        
        plt.title('Sharpe Ratio')
        plt.xticks(x, methods)
        plt.ylabel('Sharpe Ratio')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("oil_price_trading_sharpe.png")
        plt.close()

def main():
    # Initialize the forecaster
    forecaster = OilPriceForecaster(initial_training_end="2004-12-31")
    
    # Load data (this will fetch from Bloomberg if available)
    forecaster.load_data()
    
    # Forecast oil prices
    forecaster.forecast_oil_prices()
    
    # Backtest trading strategy
    forecaster.backtest_trading_strategy(risk_aversion=3, leverage=1)
    
    # Plot results
    forecaster.plot_coefficients()
    forecaster.plot_forecast_performance()
    forecaster.plot_trading_performance()
    
    # Print performance summary
    print("\n====== PERFORMANCE SUMMARY ======")
    print("\nOut-of-Sample R^2 (%):")
    for method in [m for m in forecaster.results.keys() if not m.endswith('_models')]:
        print(f"{method}: {forecaster.results[method].attrs['r2_os']:.2f}%")
    
    print("\nRecession R^2 (%):")
    for method in [m for m in forecaster.results.keys() if not m.endswith('_models')]:
        print(f"{method}: {forecaster.results[method].attrs['r2_os_rec']:.2f}%")
    
    print("\nExpansion R^2 (%):")
    for method in [m for m in forecaster.results.keys() if not m.endswith('_models')]:
        print(f"{method}: {forecaster.results[method].attrs['r2_os_exp']:.2f}%")
    
    print("\nTrading Performance:")
    for method in [m for m in forecaster.results.keys() if not m.endswith('_models')]:
        print(f"{method}:")
        print(f"  Annualized Return: {forecaster.results[method].attrs['avg_return']*100:.2f}%")
        print(f"  Sharpe Ratio: {forecaster.results[method].attrs['sharpe']:.3f}")
        print(f"  CER: {forecaster.results[method].attrs['cer']*100:.2f}%")
        print(f"  Win Rate: {forecaster.results[method].attrs['win_rate']*100:.2f}%")

if __name__ == "__main__":
    main()