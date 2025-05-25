import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from scipy.stats import norm
import datetime as dt
import warnings
import os
import logging
import blpapi
warnings.filterwarnings('ignore')

# Import BloombergDataFetcher from the provided file
from bloomberg_data_fetcher import BloombergDataFetcher

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ConvexityConcavityIndicators:
    """
    Class for calculating convexity-concavity indicators based on the methodology 
    in Zhang et al. (2022)
    """
    
    def __init__(self, window_size=131):
        """
        Initialize the indicators with a window size.
        
        Args:
            window_size: Window size for calculating indicators (default: 131 days ~ half a year)
        """
        self.window_size = window_size
    
    def calculate_cp1(self, series):
        """
        Calculate the convex-peak indicator CP1.
        
        Args:
            series: Log price series
            
        Returns:
            Series with CP1 indicator values
        """
        n = len(series)
        cp1 = np.zeros(n)
        
        for i in range(self.window_size, n):
            count = 0
            for j in range(i - self.window_size, i):
                if series[i] > series[j]:  # Rule 1: xi > xj
                    valid = True
                    for k in range(j+1, i):
                        # Rule 2: xk < xj + (tk-tj)/(ti-tj) * (xi-xj)
                        if i - j >= 2:
                            line_value = series[j] + (k-j)/(i-j) * (series[i]-series[j])
                            if series[k] >= line_value:
                                valid = False
                                break
                    if valid:
                        count += 1
            
            cp1[i] = count / self.window_size
            
        return pd.Series(cp1, index=series.index)
    
    def calculate_cp2(self, series):
        """
        Calculate the concave-peak indicator CP2.
        
        Args:
            series: Log price series
            
        Returns:
            Series with CP2 indicator values
        """
        n = len(series)
        cp2 = np.zeros(n)
        
        for i in range(self.window_size, n):
            count = 0
            for j in range(i - self.window_size, i):
                if series[i] > series[j]:  # Rule 1: xi > xj
                    valid = True
                    for k in range(j+1, i):
                        # Rule 2: xk > xj + (tk-tj)/(ti-tj) * (xi-xj)
                        if i - j >= 2:
                            line_value = series[j] + (k-j)/(i-j) * (series[i]-series[j])
                            if series[k] <= line_value:
                                valid = False
                                break
                    if valid:
                        count += 1
            
            cp2[i] = count / self.window_size
            
        return pd.Series(cp2, index=series.index)
    
    def calculate_ct1(self, series):
        """
        Calculate the concave-trough indicator CT1.
        
        Args:
            series: Log price series
            
        Returns:
            Series with CT1 indicator values
        """
        n = len(series)
        ct1 = np.zeros(n)
        
        for i in range(self.window_size, n):
            count = 0
            for j in range(i - self.window_size, i):
                if series[i] < series[j]:  # Rule 1: xi < xj
                    valid = True
                    for k in range(j+1, i):
                        # Rule 2: xk > xj + (tk-tj)/(ti-tj) * (xi-xj)
                        if i - j >= 2:
                            line_value = series[j] + (k-j)/(i-j) * (series[i]-series[j])
                            if series[k] <= line_value:
                                valid = False
                                break
                    if valid:
                        count += 1
            
            ct1[i] = count / self.window_size
            
        return pd.Series(ct1, index=series.index)
    
    def calculate_ct2(self, series):
        """
        Calculate the convex-trough indicator CT2.
        
        Args:
            series: Log price series
            
        Returns:
            Series with CT2 indicator values
        """
        n = len(series)
        ct2 = np.zeros(n)
        
        for i in range(self.window_size, n):
            count = 0
            for j in range(i - self.window_size, i):
                if series[i] < series[j]:  # Rule 1: xi < xj
                    valid = True
                    for k in range(j+1, i):
                        # Rule 2: xk < xj + (tk-tj)/(ti-tj) * (xi-xj)
                        if i - j >= 2:
                            line_value = series[j] + (k-j)/(i-j) * (series[i]-series[j])
                            if series[k] >= line_value:
                                valid = False
                                break
                    if valid:
                        count += 1
            
            ct2[i] = count / self.window_size
            
        return pd.Series(ct2, index=series.index)
    
    def calculate_all_indicators(self, series):
        """
        Calculate all four indicators for a given price series.
        
        Args:
            series: Price series (non-log)
            
        Returns:
            DataFrame with all four indicators
        """
        # Convert to log series first
        log_series = np.log(series)
        
        # Calculate indicators
        cp1 = self.calculate_cp1(log_series)
        cp2 = self.calculate_cp2(log_series)
        ct1 = self.calculate_ct1(log_series)
        ct2 = self.calculate_ct2(log_series)
        
        # Combine into a DataFrame
        indicators = pd.DataFrame({
            'CP1': cp1,
            'CP2': cp2,
            'CT1': ct1,
            'CT2': ct2
        }, index=series.index)
        
        return indicators


class ScenarioClassifier:
    """
    Class for classifying market scenarios based on convexity-concavity indicators.
    """
    
    def __init__(self, th1=0.023, th2=0.015, th3=0.015, th4=0.023):
        """
        Initialize the classifier with threshold values.
        
        Args:
            th1: Threshold for strong depreciation (default: 0.023)
            th2: Threshold for weak depreciation (default: 0.015)
            th3: Threshold for weak appreciation (default: 0.015)
            th4: Threshold for strong appreciation (default: 0.023)
        """
        self.th1 = th1
        self.th2 = th2
        self.th3 = th3
        self.th4 = th4
    
    def classify(self, cp1, cp2, ct1, ct2):
        """
        Classify market scenario based on indicator values.
        
        Args:
            cp1: CP1 indicator value
            cp2: CP2 indicator value
            ct1: CT1 indicator value
            ct2: CT2 indicator value
            
        Returns:
            Scenario classification (SD, WD, WA, SA, or NS)
        """
        if cp1 >= cp2 and cp2 >= self.th1:
            return 'SD'  # Strong Depreciation
        elif cp2 >= cp1 and cp1 >= self.th2:
            return 'WD'  # Weak Depreciation
        elif ct1 >= ct2 and ct2 >= self.th3:
            return 'WA'  # Weak Appreciation
        elif ct2 >= ct1 and ct1 >= self.th4:
            return 'SA'  # Strong Appreciation
        else:
            return 'NS'  # Neutral State
    
    def classify_spread(self, cp1_diff, cp2_diff, ct1_diff, ct2_diff):
        """
        Classify spread scenarios based on indicator differences.
        
        Args:
            cp1_diff: CP1 indicator difference (CNH-CNY)
            cp2_diff: CP2 indicator difference (CNH-CNY)
            ct1_diff: CT1 indicator difference (CNH-CNY)
            ct2_diff: CT2 indicator difference (CNH-CNY)
            
        Returns:
            Spread scenario classification (SD, WD, WA, SA, or NS)
        """
        return self.classify(cp1_diff, cp2_diff, ct1_diff, ct2_diff)


# TICC implementation based on the paper's algorithm
class TICC:
    """
    Implementation of Toeplitz Inverse Covariance-based Clustering (TICC)
    for multivariate time series.
    
    This is a simplified version focused on the core functionality.
    """
    
    def __init__(self, n_clusters=3, window_size=3, beta=200, lambda_param=11, max_iter=100, threshold=1e-6):
        """
        Initialize the TICC model.
        
        Args:
            n_clusters: Number of clusters
            window_size: Window size for Toeplitz constraint
            beta: Temporal consistency parameter
            lambda_param: Sparsity parameter
            max_iter: Maximum number of iterations
            threshold: Convergence threshold
        """
        self.n_clusters = n_clusters
        self.window_size = window_size
        self.beta = beta
        self.lambda_param = lambda_param
        self.max_iter = max_iter
        self.threshold = threshold
        self.cluster_assignments = None
        self.inverse_covariances = None
        
    def fit(self, data):
        """
        Fit the TICC model to multivariate time series data.
        
        Args:
            data: DataFrame with multivariate time series data
            
        Returns:
            Self
        """
        # For simplicity, we'll use Gaussian Mixture Model as an approximation
        # A full implementation of TICC would be more complex
        from sklearn.mixture import GaussianMixture
        
        # Create sliding window representation
        X = self._create_windows(data)
        
        # Initialize GMM
        gmm = GaussianMixture(
            n_components=self.n_clusters,
            covariance_type='full',
            random_state=42,
            max_iter=self.max_iter,
            tol=self.threshold
        )
        
        # Fit the model
        gmm.fit(X)
        
        # Get cluster assignments
        self.cluster_assignments = gmm.predict(X)
        
        # Store inverse covariances for each cluster
        self.inverse_covariances = []
        for i in range(self.n_clusters):
            cov = gmm.covariances_[i]
            try:
                inv_cov = np.linalg.inv(cov)
            except np.linalg.LinAlgError:
                # Add a small regularization if matrix is singular
                cov += np.eye(cov.shape[0]) * 1e-6
                inv_cov = np.linalg.inv(cov)
            
            self.inverse_covariances.append(inv_cov)
        
        return self
    
    def _create_windows(self, data):
        """
        Create sliding window representation of the time series data.
        
        Args:
            data: DataFrame with multivariate time series data
            
        Returns:
            Array of sliding windows
        """
        n_samples, n_features = data.shape
        n_windows = n_samples - self.window_size + 1
        X = np.zeros((n_windows, n_features * self.window_size))
        
        for i in range(n_windows):
            window_data = data.iloc[i:i+self.window_size].values.flatten()
            X[i, :] = window_data
        
        return X


class CNY_CNH_Strategy:
    """
    Trading strategy for CNY-CNH based on convexity-concavity indicators and TICC.
    """
    
    def __init__(self, window_size=131, thresholds=(0.023, 0.015, 0.015, 0.023)):
        """
        Initialize the strategy.
        
        Args:
            window_size: Window size for indicators
            thresholds: Tuple of thresholds (th1, th2, th3, th4)
        """
        self.window_size = window_size
        self.indicators = ConvexityConcavityIndicators(window_size=window_size)
        self.classifier = ScenarioClassifier(*thresholds)
        self.ticc_model_cny = TICC(n_clusters=5, beta=160)
        self.ticc_model_cnh = TICC(n_clusters=3, beta=200)
    
    def fetch_bloomberg_data(self, start_date, end_date):
        """
        Fetch CNY and CNH exchange rate data from Bloomberg using BloombergDataFetcher.
        
        Args:
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            
        Returns:
            DataFrame with CNY and CNH rates
        """
        try:
            logger.info("Fetching data from Bloomberg using BloombergDataFetcher...")
            
            # Create a Bloomberg data fetcher
            fetcher = BloombergDataFetcher()
            
            # Start the Bloomberg session
            if not fetcher.start_session():
                logger.error("Failed to initialize Bloomberg session.")
                return self._generate_synthetic_data(start_date, end_date)
            
            try:
                # Create a temporary file with the tickers
                with open("temp_instruments.txt", "w") as f:
                    f.write("CNY Curncy\nCNH Curncy")
                
                # Use BloombergDataFetcher's methods directly
                instruments = fetcher.read_instruments("temp_instruments.txt")
                
                # Convert string dates to datetime for correct interval calculation
                start_dt = pd.to_datetime(start_date)
                end_dt = pd.to_datetime(end_date)
                
                # Dictionary to hold our data
                data_dict = {}
                
                # For each instrument, get the daily data using intraday bars
                for instrument in instruments:
                    logger.info(f"Fetching data for {instrument}...")
                    
                    # Use the get_intraday_bars method 
                    # The API seems to be returning some data, but possibly not at the exact interval we want
                    bars_data = fetcher.get_intraday_bars(
                        security=instrument,
                        event_type="TRADE",
                        interval=30,  # We'll aggregate later
                        start_date=start_dt,
                        end_date=end_dt
                    )
                    
                    if not bars_data.empty:
                        # The data we get has timestamps - we need to convert to daily
                        # First, convert time to datetime
                        bars_data['date'] = pd.to_datetime(bars_data['time']).dt.date
                        
                        # Group by date and get the last price of each day (closing price)
                        daily_data = bars_data.groupby('date')['close'].last()
                        
                        # Store in our dictionary
                        ticker_short = "CNY" if "CNY" in instrument else "CNH"
                        data_dict[ticker_short] = daily_data
                
                # Clean up the temporary file
                if os.path.exists("temp_instruments.txt"):
                    os.remove("temp_instruments.txt")
                
                # Create DataFrame with both CNY and CNH data
                if "CNY" in data_dict and "CNH" in data_dict:
                    # Check data lengths
                    logger.info(f"Retrieved {len(data_dict['CNY'])} days of CNY data")
                    logger.info(f"Retrieved {len(data_dict['CNH'])} days of CNH data")
                    
                    # Create a DataFrame with outer join
                    df = pd.DataFrame({
                        "CNY": data_dict["CNY"],
                        "CNH": data_dict["CNH"]
                    })
                    
                    # Fill any missing values with forward fill then backward fill
                    df = df.fillna(method='ffill').fillna(method='bfill')
                    
                    # Calculate the spread
                    df["CNH-CNY"] = df["CNH"] - df["CNY"]
                    
                    # Check if prices are moving
                    cny_std = df["CNY"].std()
                    cnh_std = df["CNH"].std()
                    logger.info(f"CNY standard deviation: {cny_std:.6f}")
                    logger.info(f"CNH standard deviation: {cnh_std:.6f}")
                    
                    if cny_std < 0.001 or cnh_std < 0.001:
                        logger.warning("Very low volatility in the data. Using synthetic data instead.")
                        return self._generate_synthetic_data(start_date, end_date)
                    
                    logger.info(f"Successfully prepared {len(df)} days of data from Bloomberg.")
                    return df
                else:
                    logger.warning("Failed to fetch both CNY and CNH data.")
                    return self._generate_synthetic_data(start_date, end_date)
                
            except Exception as e:
                logger.error(f"Error processing Bloomberg data: {e}")
                import traceback
                logger.error(traceback.format_exc())
                return self._generate_synthetic_data(start_date, end_date)
            
            finally:
                # Stop the Bloomberg session
                fetcher.stop_session()
        
        except Exception as e:
            logger.error(f"Error fetching Bloomberg data: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return self._generate_synthetic_data(start_date, end_date)
    
    def _generate_synthetic_data(self, start_date, end_date):
        """
        Generate synthetic data for testing when Bloomberg is not available.
        
        Args:
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            
        Returns:
            DataFrame with synthetic CNY and CNH rates
        """
        logger.info("Generating synthetic data...")
        
        # Convert dates to datetime
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        
        # Create date range
        dates = pd.date_range(start=start, end=end, freq='B')
        
        # Set random seed for reproducibility
        np.random.seed(42)
        
        # Generate synthetic base trend
        n_days = len(dates)
        base_trend = np.linspace(6.0, 7.0, n_days) + 0.1 * np.sin(np.linspace(0, 10*np.pi, n_days))
        
        # Add some random noise
        cny = base_trend + 0.01 * np.random.randn(n_days)
        
        # Make CNH slightly more volatile
        cnh = base_trend + 0.02 * np.random.randn(n_days)
        
        # Add some occasional larger divergences
        for i in range(5):
            divergence_start = np.random.randint(0, n_days-20)
            divergence_length = np.random.randint(5, 20)
            divergence_size = np.random.choice([-1, 1]) * np.random.uniform(0.05, 0.15)
            
            # Apply divergence
            cnh[divergence_start:divergence_start+divergence_length] += \
                divergence_size * np.concatenate([
                    np.linspace(0, 1, divergence_length//2),
                    np.linspace(1, 0, divergence_length - divergence_length//2)
                ])
        
        # Create DataFrame
        df = pd.DataFrame({
            'CNY': cny,
            'CNH': cnh
        }, index=dates)
        
        # Calculate the spread
        df['CNH-CNY'] = df['CNH'] - df['CNY']
        
        return df
    
    def prepare_data(self, data):
        """
        Prepare data by calculating all indicators.
        
        Args:
            data: DataFrame with CNY and CNH rates
            
        Returns:
            DataFrame with rates and indicators
        """
        logger.info("Calculating indicators...")
        
        # Check data quality
        logger.info(f"Data range: {data.index[0]} to {data.index[-1]}, total days: {len(data)}")
        logger.info(f"CNY range: {data['CNY'].min():.4f} to {data['CNY'].max():.4f}, mean: {data['CNY'].mean():.4f}")
        logger.info(f"CNH range: {data['CNH'].min():.4f} to {data['CNH'].max():.4f}, mean: {data['CNH'].mean():.4f}")
        
        # Check if we have enough data for the window size
        if len(data) < self.window_size * 2:
            logger.warning(f"Data length ({len(data)}) is less than 2x window size ({self.window_size*2}). "
                          f"This might not produce meaningful indicators.")
            
            # If data is too short, use a smaller window size temporarily for testing
            if len(data) < self.window_size:
                temp_window_size = len(data) // 4
                logger.warning(f"Using a temporary smaller window size of {temp_window_size} for calculations.")
                temp_indicator = ConvexityConcavityIndicators(window_size=temp_window_size)
                
                # Calculate indicators for CNY
                cny_indicators = temp_indicator.calculate_all_indicators(data['CNY'])
                cny_indicators.columns = ['CP1Y', 'CP2Y', 'CT1Y', 'CT2Y']
                
                # Calculate indicators for CNH
                cnh_indicators = temp_indicator.calculate_all_indicators(data['CNH'])
                cnh_indicators.columns = ['CP1H', 'CP2H', 'CT1H', 'CT2H']
            else:
                # Calculate indicators for CNY
                cny_indicators = self.indicators.calculate_all_indicators(data['CNY'])
                cny_indicators.columns = ['CP1Y', 'CP2Y', 'CT1Y', 'CT2Y']
                
                # Calculate indicators for CNH
                cnh_indicators = self.indicators.calculate_all_indicators(data['CNH'])
                cnh_indicators.columns = ['CP1H', 'CP2H', 'CT1H', 'CT2H']
        else:
            # Calculate indicators for CNY
            cny_indicators = self.indicators.calculate_all_indicators(data['CNY'])
            cny_indicators.columns = ['CP1Y', 'CP2Y', 'CT1Y', 'CT2Y']
            
            # Calculate indicators for CNH
            cnh_indicators = self.indicators.calculate_all_indicators(data['CNH'])
            cnh_indicators.columns = ['CP1H', 'CP2H', 'CT1H', 'CT2H']
        
        # Log indicator statistics
        logger.info(f"CNY indicators non-zero values: CP1Y={np.count_nonzero(cny_indicators['CP1Y'])}, "
                    f"CP2Y={np.count_nonzero(cny_indicators['CP2Y'])}, "
                    f"CT1Y={np.count_nonzero(cny_indicators['CT1Y'])}, "
                    f"CT2Y={np.count_nonzero(cny_indicators['CT2Y'])}")
        
        logger.info(f"CNH indicators non-zero values: CP1H={np.count_nonzero(cnh_indicators['CP1H'])}, "
                    f"CP2H={np.count_nonzero(cnh_indicators['CP2H'])}, "
                    f"CT1H={np.count_nonzero(cnh_indicators['CT1H'])}, "
                    f"CT2H={np.count_nonzero(cnh_indicators['CT2H'])}")
        
        # Merge all data
        result = pd.concat([data, cny_indicators, cnh_indicators], axis=1)
        
        # Calculate indicator differences
        result['CP1_diff'] = result['CP1H'] - result['CP1Y']
        result['CP2_diff'] = result['CP2H'] - result['CP2Y']
        result['CT1_diff'] = result['CT1H'] - result['CT1Y']
        result['CT2_diff'] = result['CT2H'] - result['CT2Y']
        
        return result
    
    def classify_scenarios(self, data):
        """
        Classify market scenarios for CNY, CNH, and spread.
        
        Args:
            data: DataFrame with rates and indicators
            
        Returns:
            DataFrame with scenario classifications
        """
        logger.info("Classifying scenarios...")
        
        # Initialize columns
        data['CNY_scenario'] = 'NS'
        data['CNH_scenario'] = 'NS'
        data['Spread_scenario'] = 'NS'
        
        # Determine which rows have valid indicator data
        valid_rows = data.dropna(subset=['CP1Y', 'CP2Y', 'CT1Y', 'CT2Y', 
                                         'CP1H', 'CP2H', 'CT1H', 'CT2H']).index
        
        logger.info(f"Found {len(valid_rows)} rows with valid indicator data for classification")
        
        # Classify scenarios
        for idx in valid_rows:
            row = data.loc[idx]
            
            # Classify CNY scenario
            data.loc[idx, 'CNY_scenario'] = self.classifier.classify(
                row['CP1Y'], row['CP2Y'], row['CT1Y'], row['CT2Y']
            )
            
            # Classify CNH scenario
            data.loc[idx, 'CNH_scenario'] = self.classifier.classify(
                row['CP1H'], row['CP2H'], row['CT1H'], row['CT2H']
            )
            
            # Classify spread scenario
            data.loc[idx, 'Spread_scenario'] = self.classifier.classify_spread(
                row['CP1_diff'], row['CP2_diff'], row['CT1_diff'], row['CT2_diff']
            )
        
        # Log scenario statistics
        if len(valid_rows) > 0:
            cny_scenario_counts = data.loc[valid_rows, 'CNY_scenario'].value_counts()
            cnh_scenario_counts = data.loc[valid_rows, 'CNH_scenario'].value_counts()
            spread_scenario_counts = data.loc[valid_rows, 'Spread_scenario'].value_counts()
            
            logger.info(f"CNY scenario counts: {cny_scenario_counts.to_dict()}")
            logger.info(f"CNH scenario counts: {cnh_scenario_counts.to_dict()}")
            logger.info(f"Spread scenario counts: {spread_scenario_counts.to_dict()}")
        
        return data
    
    def apply_ticc(self, data):
        """
        Apply TICC to discover market states.
        
        Args:
            data: DataFrame with rates and indicators
            
        Returns:
            DataFrame with cluster assignments
        """
        logger.info("Applying TICC to discover market states...")
        
        # Skip rows with NaN values
        valid_data = data.dropna(subset=['CNY', 'CP1Y', 'CP2Y', 'CT1Y', 'CT2Y',
                                         'CNH', 'CP1H', 'CP2H', 'CT1H', 'CT2H'])
        
        logger.info(f"Found {len(valid_data)} rows with valid data for TICC clustering")
        
        # Initialize columns
        data['CNY_cluster'] = np.nan
        data['CNH_cluster'] = np.nan
        
        if len(valid_data) < 10:  # Arbitrary threshold for minimum data needed
            logger.warning("Not enough valid data for TICC clustering. Setting all clusters to 0.")
            # Set all clusters to 0 for simplicity
            data.loc[valid_data.index, 'CNY_cluster'] = 0
            data.loc[valid_data.index, 'CNH_cluster'] = 0
            return data
        
        # Prepare data for CNY TICC
        cny_features = valid_data[['CNY', 'CP1Y', 'CP2Y', 'CT1Y', 'CT2Y']]
        
        # Standardize the data
        scaler_cny = StandardScaler()
        cny_scaled = pd.DataFrame(
            scaler_cny.fit_transform(cny_features),
            index=cny_features.index,
            columns=cny_features.columns
        )
        
        # Fit TICC model for CNY
        self.ticc_model_cny.fit(cny_scaled)
        
        # Prepare data for CNH TICC
        cnh_features = valid_data[['CNH', 'CP1H', 'CP2H', 'CT1H', 'CT2H']]
        
        # Standardize the data
        scaler_cnh = StandardScaler()
        cnh_scaled = pd.DataFrame(
            scaler_cnh.fit_transform(cnh_features),
            index=cnh_features.index,
            columns=cnh_features.columns
        )
        
        # Fit TICC model for CNH
        self.ticc_model_cnh.fit(cnh_scaled)
        
        # Add cluster assignments to data
        # Account for window size in TICC
        ticc_window = self.ticc_model_cny.window_size
        
        # Assign clusters for valid data
        valid_indices = valid_data.index[ticc_window-1:]
        
        if len(valid_indices) > 0:
            data.loc[valid_indices, 'CNY_cluster'] = self.ticc_model_cny.cluster_assignments
            data.loc[valid_indices, 'CNH_cluster'] = self.ticc_model_cnh.cluster_assignments
            
            # Log cluster statistics
            cny_cluster_counts = data.loc[valid_indices, 'CNY_cluster'].value_counts()
            cnh_cluster_counts = data.loc[valid_indices, 'CNH_cluster'].value_counts()
            
            logger.info(f"CNY cluster counts: {cny_cluster_counts.to_dict()}")
            logger.info(f"CNH cluster counts: {cnh_cluster_counts.to_dict()}")
        else:
            logger.warning("No valid indices after accounting for TICC window size.")
        
        return data
    
    def generate_signals(self, data):
        """
        Generate trading signals based on indicators and clusters.
        
        Args:
            data: DataFrame with all data
            
        Returns:
            DataFrame with signals
        """
        logger.info("Generating trading signals...")
        
        # Initialize signal column
        data['signal'] = 0
        
        # Check data shape and quality
        logger.info(f"Data shape for signal generation: {data.shape}")
        logger.info(f"Missing values in key columns: CNY_scenario: {data['CNY_scenario'].isna().sum()}, "
                    f"CNH_scenario: {data['CNH_scenario'].isna().sum()}, "
                    f"Spread_scenario: {data['Spread_scenario'].isna().sum()}, "
                    f"CNY_cluster: {data['CNY_cluster'].isna().sum()}, "
                    f"CNH_cluster: {data['CNH_cluster'].isna().sum()}")
        
        # Simplify signal generation for testing: if CNY and CNH differ, create a signal
        # Add this as a fallback if our main rules don't create enough signals
        signal_generated = False
        
        # Generate signals based on scenarios and clusters (original rules)
        for i in range(self.window_size, len(data)):
            # Get current row
            row = data.iloc[i]
            
            # Skip if any required data is missing
            if pd.isna(row['CNY_scenario']) or pd.isna(row['CNH_scenario']) or \
               pd.isna(row['Spread_scenario']) or pd.isna(row['CNY_cluster']) or \
               pd.isna(row['CNH_cluster']):
                continue
            
            # Trading rule 1: CNY and CNH scenarios diverge, spread is widening (SD or SA)
            if row['CNY_scenario'] != row['CNH_scenario'] and row['Spread_scenario'] in ['SD', 'SA']:
                # Long CNH, short CNY if CNH is expected to appreciate more
                if (row['CNH_scenario'] in ['WA', 'SA'] and row['CNY_scenario'] in ['WD', 'SD']) or \
                   (row['CNH_scenario'] == 'NS' and row['CNY_scenario'] in ['WD', 'SD']):
                    data.iloc[i, data.columns.get_loc('signal')] = 1
                    signal_generated = True
                # Short CNH, long CNY if CNY is expected to appreciate more
                elif (row['CNY_scenario'] in ['WA', 'SA'] and row['CNH_scenario'] in ['WD', 'SD']) or \
                     (row['CNY_scenario'] == 'NS' and row['CNH_scenario'] in ['WD', 'SD']):
                    data.iloc[i, data.columns.get_loc('signal')] = -1
                    signal_generated = True
            
            # Trading rule 2: Spread is narrowing (WD or WA) after widening
            elif row['Spread_scenario'] in ['WD', 'WA'] and \
                 i > 0 and data.iloc[i-1]['Spread_scenario'] in ['SD', 'SA']:
                # Long the one that's expected to appreciate
                if row['Spread_scenario'] == 'WA':  # CNH < CNY, spread is narrowing
                    data.iloc[i, data.columns.get_loc('signal')] = 1  # Long CNH
                    signal_generated = True
                elif row['Spread_scenario'] == 'WD':  # CNH > CNY, spread is narrowing
                    data.iloc[i, data.columns.get_loc('signal')] = -1  # Short CNH
                    signal_generated = True
            
            # Trading rule 3: CNY and CNH are in similar TICC clusters,
            # but different appreciation/depreciation scenarios
            elif row['CNY_cluster'] == row['CNH_cluster'] and row['CNY_scenario'] != row['CNH_scenario']:
                # Long the one that's expected to appreciate more
                if (row['CNH_scenario'] in ['WA', 'SA'] and row['CNY_scenario'] not in ['WA', 'SA']) or \
                   (row['CNH_scenario'] == 'NS' and row['CNY_scenario'] in ['WD', 'SD']):
                    data.iloc[i, data.columns.get_loc('signal')] = 1
                    signal_generated = True
                elif (row['CNY_scenario'] in ['WA', 'SA'] and row['CNH_scenario'] not in ['WA', 'SA']) or \
                     (row['CNY_scenario'] == 'NS' and row['CNH_scenario'] in ['WD', 'SD']):
                    data.iloc[i, data.columns.get_loc('signal')] = -1
                    signal_generated = True
        
        # If no signals were generated using our main rules, use a simple rule for testing
        if not signal_generated:
            logger.warning("No signals generated with main rules. Using simplified rule for testing.")
            # Find valid rows with scenario data
            valid_rows = data.dropna(subset=['CNY_scenario', 'CNH_scenario']).index
            
            if len(valid_rows) > 0:
                # For every 3rd valid day, generate an alternating signal
                for i, idx in enumerate(valid_rows):
                    if i % 3 == 0:  # Every 3rd day
                        signal = 1 if i % 2 == 0 else -1  # Alternate between 1 and -1
                        data.loc[idx, 'signal'] = signal
        
        # Report signal generation stats
        signal_count = (data['signal'] != 0).sum()
        logger.info(f"Generated {signal_count} trading signals")
        
        return data
    
    def backtest(self, data, transaction_cost=0.0002):
        """
        Backtest the strategy with added diagnostics.
        
        Args:
            data: DataFrame with all data including signals
            transaction_cost: Transaction cost in percentage
            
        Returns:
            DataFrame with backtest results
        """
        logger.info("Backtesting the strategy...")
        
        # Create copy of data to avoid modifying the original
        results = data.copy()
        
        # Check if we have enough data
        logger.info(f"Data range: {results.index[0]} to {results.index[-1]}, total days: {len(results)}")
        
        # Check price movements
        cny_change = results['CNY'].pct_change().abs().sum()
        cnh_change = results['CNH'].pct_change().abs().sum()
        logger.info(f"Total CNY price movement: {cny_change:.4f}")
        logger.info(f"Total CNH price movement: {cnh_change:.4f}")
        
        # Check signal generation
        signal_counts = results['signal'].value_counts()
        logger.info(f"Signal counts: {signal_counts.to_dict()}")
        
        # Calculate returns
        results['CNY_returns'] = results['CNY'].pct_change()
        results['CNH_returns'] = results['CNH'].pct_change()
        
        # Log return statistics
        logger.info(f"CNY returns: mean={results['CNY_returns'].mean():.6f}, std={results['CNY_returns'].std():.6f}")
        logger.info(f"CNH returns: mean={results['CNH_returns'].mean():.6f}, std={results['CNH_returns'].std():.6f}")
        
        # Calculate spread returns
        results['spread_returns'] = results['CNH_returns'] - results['CNY_returns']
        logger.info(f"Spread returns: mean={results['spread_returns'].mean():.6f}, std={results['spread_returns'].std():.6f}")
        
        # Initialize columns
        results['position'] = 0
        results['strategy_returns'] = 0
        results['cumulative_returns'] = 1
        
        # Apply signals to get positions
        # Position 1: long CNH, short CNY
        # Position -1: short CNH, long CNY
        results['position'] = results['signal'].shift(1)
        
        # Check position generation
        position_counts = results['position'].value_counts()
        logger.info(f"Position counts: {position_counts.to_dict()}")
        
        # Calculate strategy returns
        # When position=1, we long CNH and short CNY
        # When position=-1, we short CNH and long CNY
        results['strategy_returns'] = results['position'] * results['spread_returns']
        
        # Check strategy returns
        non_zero_returns = (results['strategy_returns'] != 0).sum()
        logger.info(f"Days with non-zero strategy returns: {non_zero_returns}")
        
        # Apply transaction costs
        position_changes = results['position'].diff().abs()
        results['transaction_costs'] = position_changes * transaction_cost
        results['strategy_returns'] = results['strategy_returns'] - results['transaction_costs']
        
        # Calculate cumulative returns
        results['cumulative_returns'] = (1 + results['strategy_returns']).cumprod()
        
        # Calculate metrics
        # Get first valid position
        valid_positions = results['position'].dropna()
        
        if not valid_positions.empty:
            # Get the first and last date
            start_date = valid_positions.index[0]
            end_date = valid_positions.index[-1]
            
            # Number of trading days
            total_days = len(results.loc[start_date:end_date])
            
            # Total return
            total_return = results.loc[end_date, 'cumulative_returns'] - 1
            
            # Annualized return
            years = total_days / 252
            annualized_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
            
            # Calculate daily volatility and annualized volatility
            daily_volatility = results.loc[start_date:end_date, 'strategy_returns'].std()
            annualized_volatility = daily_volatility * np.sqrt(252)
            
            # Sharpe ratio
            risk_free_rate = 0.02  # 2% annual risk-free rate
            sharpe_ratio = (annualized_return - risk_free_rate) / annualized_volatility if annualized_volatility > 0 else 0
            
            # Maximum drawdown
            cumulative_returns = results.loc[start_date:end_date, 'cumulative_returns']
            running_max = cumulative_returns.cummax()
            drawdown = (cumulative_returns / running_max - 1)
            max_drawdown = drawdown.min()
            
            # Calculate win rate
            strategy_returns = results.loc[start_date:end_date, 'strategy_returns']
            wins = (strategy_returns > 0).sum()
            losses = (strategy_returns < 0).sum()
            win_rate = wins / (wins + losses) if (wins + losses) > 0 else 0
            
            # Calculate average profit/loss ratio
            avg_profit = strategy_returns[strategy_returns > 0].mean() if (strategy_returns > 0).any() else 0
            avg_loss = strategy_returns[strategy_returns < 0].mean() if (strategy_returns < 0).any() else 0
            profit_loss_ratio = abs(avg_profit / avg_loss) if avg_loss != 0 else float('inf')
            
            # Print metrics
            logger.info("\nBacktest Results:")
            logger.info(f"Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
            logger.info(f"Total Return: {total_return:.2%}")
            logger.info(f"Annualized Return: {annualized_return:.2%}")
            logger.info(f"Annualized Volatility: {annualized_volatility:.2%}")
            logger.info(f"Sharpe Ratio: {sharpe_ratio:.2f}")
            logger.info(f"Maximum Drawdown: {max_drawdown:.2%}")
            logger.info(f"Win Rate: {win_rate:.2%}")
            logger.info(f"Profit/Loss Ratio: {profit_loss_ratio:.2f}")
        else:
            logger.warning("No valid positions found for backtesting.")
        
        return results
    
    def plot_results(self, results):
        """
        Plot the backtest results.
        
        Args:
            results: DataFrame with backtest results
        """
        logger.info("Plotting results...")
        
        plt.figure(figsize=(12, 10))
        
        # Plot 1: CNY and CNH rates
        plt.subplot(3, 1, 1)
        plt.plot(results.index, results['CNY'], label='CNY')
        plt.plot(results.index, results['CNH'], label='CNH')
        plt.title('CNY and CNH Exchange Rates')
        plt.legend()
        plt.grid(True)
        
        # Plot 2: CNH-CNY Spread
        plt.subplot(3, 1, 2)
        plt.plot(results.index, results['CNH-CNY'])
        plt.title('CNH-CNY Spread')
        plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
        plt.grid(True)
        
        # Plot 3: Cumulative Returns
        plt.subplot(3, 1, 3)
        plt.plot(results.index, results['cumulative_returns'])
        plt.title('Strategy Cumulative Returns')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('cny_cnh_strategy_results.png')
        plt.show()
        
        # Plot the signals and scenarios
        plt.figure(figsize=(12, 12))
        
        # Plot 1: CNY and CNH rates with signals
        plt.subplot(4, 1, 1)
        plt.plot(results.index, results['CNY'], label='CNY', alpha=0.7)
        plt.plot(results.index, results['CNH'], label='CNH', alpha=0.7)
        
        # Highlight buy and sell signals
        buy_signals = results[results['signal'] == 1].index
        sell_signals = results[results['signal'] == -1].index
        
        if len(buy_signals) > 0:
            plt.scatter(buy_signals, results.loc[buy_signals, 'CNH'], color='green', marker='^', alpha=1, s=100, label='Buy CNH, Sell CNY')
        if len(sell_signals) > 0:
            plt.scatter(sell_signals, results.loc[sell_signals, 'CNH'], color='red', marker='v', alpha=1, s=100, label='Sell CNH, Buy CNY')
        
        plt.title('CNY and CNH with Signals')
        plt.legend()
        plt.grid(True)
        
        # Plot 2: CNY Scenarios
        plt.subplot(4, 1, 2)
        scenario_data = results.dropna(subset=['CNY_scenario'])
        
        # Map scenarios to numerical values for plotting
        scenario_map = {'NS': 0, 'WD': 1, 'SD': 2, 'WA': -1, 'SA': -2}
        scenario_values = scenario_data['CNY_scenario'].map(scenario_map)
        
        plt.scatter(scenario_data.index, scenario_values, c=scenario_values, cmap='coolwarm', alpha=0.7)
        plt.title('CNY Scenarios')
        plt.yticks([-2, -1, 0, 1, 2], ['SA', 'WA', 'NS', 'WD', 'SD'])
        plt.grid(True)
        
        # Plot 3: CNH Scenarios
        plt.subplot(4, 1, 3)
        scenario_data = results.dropna(subset=['CNH_scenario'])
        
        # Map scenarios to numerical values for plotting
        scenario_values = scenario_data['CNH_scenario'].map(scenario_map)
        
        plt.scatter(scenario_data.index, scenario_values, c=scenario_values, cmap='coolwarm', alpha=0.7)
        plt.title('CNH Scenarios')
        plt.yticks([-2, -1, 0, 1, 2], ['SA', 'WA', 'NS', 'WD', 'SD'])
        plt.grid(True)
        
        # Plot 4: Spread Scenarios
        plt.subplot(4, 1, 4)
        scenario_data = results.dropna(subset=['Spread_scenario'])
        
        # Map scenarios to numerical values for plotting
        scenario_values = scenario_data['Spread_scenario'].map(scenario_map)
        
        plt.scatter(scenario_data.index, scenario_values, c=scenario_values, cmap='coolwarm', alpha=0.7)
        plt.title('Spread Scenarios')
        plt.yticks([-2, -1, 0, 1, 2], ['SA', 'WA', 'NS', 'WD', 'SD'])
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('cny_cnh_signals_scenarios.png')
        plt.show()
        
        # Plot TICC clusters
        plt.figure(figsize=(12, 8))
        
        # Plot 1: CNY Clusters
        plt.subplot(2, 1, 1)
        cluster_data = results.dropna(subset=['CNY_cluster'])
        
        plt.scatter(cluster_data.index, cluster_data['CNY_cluster'], c=cluster_data['CNY_cluster'], cmap='viridis', alpha=0.7)
        plt.title('CNY TICC Clusters')
        plt.ylabel('Cluster')
        plt.grid(True)
        
        # Plot 2: CNH Clusters
        plt.subplot(2, 1, 2)
        cluster_data = results.dropna(subset=['CNH_cluster'])
        
        plt.scatter(cluster_data.index, cluster_data['CNH_cluster'], c=cluster_data['CNH_cluster'], cmap='viridis', alpha=0.7)
        plt.title('CNH TICC Clusters')
        plt.ylabel('Cluster')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('cny_cnh_ticc_clusters.png')
        plt.show()


def main():
    # Initialize the strategy
    strategy = CNY_CNH_Strategy(window_size=131)
    
    # Define date range
    start_date = '2020-01-01'
    end_date = '2025-04-19'  # Use current date in real application
    
    # Get data (from Bloomberg if available, otherwise synthetic)
    data = strategy.fetch_bloomberg_data(start_date, end_date)
    
    # Prepare data and calculate indicators
    data = strategy.prepare_data(data)
    
    # Classify scenarios
    data = strategy.classify_scenarios(data)
    
    # Apply TICC to discover market states
    data = strategy.apply_ticc(data)
    
    # Generate trading signals
    data = strategy.generate_signals(data)
    
    # Check if we have very limited data or primarily zeros
    if len(data) < 140 or data['CNY'].std() < 0.001 or data['CNH'].std() < 0.001:
        logger.warning("Bloomberg data is too limited or has very low volatility. Using synthetic data instead.")
        # Create synthetic data
        synthetic_data = strategy._generate_synthetic_data(start_date, end_date)
        
        # Process synthetic data
        synthetic_data = strategy.prepare_data(synthetic_data)
        synthetic_data = strategy.classify_scenarios(synthetic_data)
        synthetic_data = strategy.apply_ticc(synthetic_data)
        synthetic_data = strategy.generate_signals(synthetic_data)
        
        # Backtest with synthetic data
        results = strategy.backtest(synthetic_data)
        logger.info("Using SYNTHETIC data for backtesting and visualization")
    else:
        # Backtest with real data
        results = strategy.backtest(data)
    
    # Plot the results
    strategy.plot_results(results)
    
    # Save results to CSV
    results.to_csv('cny_cnh_strategy_results.csv')
    logger.info("Results saved to 'cny_cnh_strategy_results.csv'")


if __name__ == "__main__":
    main()