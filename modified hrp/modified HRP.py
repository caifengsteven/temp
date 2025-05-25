import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform
import blpapi
import datetime as dt
from tqdm import tqdm
import warnings
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("mhrp_portfolio.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()

warnings.filterwarnings('ignore')

class BloombergDataFetcher:
    """Class to fetch data from Bloomberg terminal."""
    
    def __init__(self):
        """Initialize Bloomberg connection."""
        self.session = None
        self.refdata_service = None
        
    def start_session(self):
        """Start Bloomberg session."""
        logger.info("Starting Bloomberg session...")
        sessionOptions = blpapi.SessionOptions()
        sessionOptions.setServerHost('localhost')
        sessionOptions.setServerPort(8194)
        
        self.session = blpapi.Session(sessionOptions)
        if not self.session.start():
            raise RuntimeError("Failed to start session.")
        
        if not self.session.openService("//blp/refdata"):
            raise RuntimeError("Failed to open //blp/refdata")
        
        self.refdata_service = self.session.getService("//blp/refdata")
        logger.info("Bloomberg session started successfully.")
    
    def stop_session(self):
        """Stop Bloomberg session."""
        if self.session:
            self.session.stop()
            logger.info("Bloomberg session stopped.")
    
    def fetch_historical_data(self, tickers, fields, start_date, end_date, 
                             period='DAILY'):
        """
        Fetch historical data from Bloomberg.
        
        Parameters:
        -----------
        tickers : list
            List of Bloomberg tickers
        fields : list
            List of Bloomberg fields (e.g., 'PX_LAST')
        start_date : str
            Start date in 'YYYYMMDD' format
        end_date : str
            End date in 'YYYYMMDD' format
        period : str
            Periodicity ('DAILY', 'WEEKLY', 'MONTHLY')
            
        Returns:
        --------
        DataFrame
            DataFrame with historical data
        """
        request = self.refdata_service.createRequest("HistoricalDataRequest")
        
        # Set request parameters
        for ticker in tickers:
            request.append("securities", ticker)
        
        for field in fields:
            request.append("fields", field)
        
        request.set("startDate", start_date)
        request.set("endDate", end_date)
        request.set("periodicitySelection", period)
        
        logger.info(f"Sending request for {len(tickers)} securities...")
        
        # Send request
        self.session.sendRequest(request)
        
        # Process response
        data = []
        
        while True:
            event = self.session.nextEvent(500)
            for msg in event:
                if msg.messageType() == "HistoricalDataResponse":
                    security_data = msg.getElement("securityData")
                    security_name = security_data.getElementAsString("security")
                    field_data = security_data.getElement("fieldData")
                    
                    for i in range(field_data.numValues()):
                        field_value = field_data.getValue(i)
                        date = field_value.getElementAsDatetime("date").strftime('%Y-%m-%d')
                        
                        row_data = {'date': date, 'security': security_name}
                        
                        for field in fields:
                            if field_value.hasElement(field):
                                row_data[field] = field_value.getElementAsFloat(field)
                            else:
                                row_data[field] = np.nan
                        
                        data.append(row_data)
            
            if event.eventType() == blpapi.Event.RESPONSE:
                break
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        if df.empty:
            logger.warning("No data received from Bloomberg.")
            return None
        
        # Convert date to datetime
        df['date'] = pd.to_datetime(df['date'])
        
        # Pivot to have tickers as columns
        pivot_df = pd.pivot_table(
            df, values=fields[0], index='date', columns='security'
        )
        
        # Ensure index is datetime
        pivot_df.index = pd.to_datetime(pivot_df.index)
        
        # Sort by date
        pivot_df.sort_index(inplace=True)
        
        logger.info(f"Received data for {pivot_df.shape[1]} securities from {pivot_df.index[0]} to {pivot_df.index[-1]}.")
        
        return pivot_df

    def fetch_risk_free_rate(self, start_date, end_date, region='US', period='DAILY'):
        """
        Fetch risk-free rate data.
        
        Parameters:
        -----------
        start_date : str
            Start date in 'YYYYMMDD' format
        end_date : str
            End date in 'YYYYMMDD' format
        region : str
            Region for risk-free rate ('US', 'EU', 'ASIA')
        period : str
            Periodicity ('DAILY', 'WEEKLY', 'MONTHLY')
            
        Returns:
        --------
        DataFrame
            DataFrame with risk-free rate data
        """
        # Select appropriate risk-free rate based on region
        if region == 'EU':
            ticker = "EUR001M Index"  # EU 1-month T-bill
        elif region == 'US':
            ticker = "US0001M Index"  # US 1-month LIBOR
        elif region == 'ASIA':
            ticker = "HIHD01M Index"  # Hong Kong 1-month HIBOR
        else:
            ticker = "US0001M Index"  # Default to US
            
        fields = ["PX_LAST"]
        
        rf_data = self.fetch_historical_data(
            [ticker], fields, start_date, end_date, period
        )
        
        if rf_data is None:
            logger.warning("No risk-free rate data received. Using zero as default.")
            # Create a DataFrame with zeros for the risk-free rate
            dates = pd.date_range(start=start_date, end=end_date, freq='D')
            rf_data = pd.DataFrame(0, index=dates, columns=[ticker])
        
        # Convert annual rate to daily/weekly/monthly
        if period == 'DAILY':
            trading_days = 252
        elif period == 'WEEKLY':
            trading_days = 52
        else:  # MONTHLY
            trading_days = 12
            
        rf_data = rf_data / 100 / trading_days  # Convert percentage to decimal and annualized to period
        
        return rf_data

    def get_index_constituents(self, index_ticker, date=None):
        """
        Fetch index constituents for a specific date
        
        Parameters:
        -----------
        index_ticker : str
            Bloomberg index ticker (e.g., "SPX Index")
        date : str
            Date in 'YYYYMMDD' format (None for current date)
            
        Returns:
        --------
        DataFrame
            DataFrame with constituent tickers and weights
        """
        request = self.refdata_service.createRequest("ReferenceDataRequest")
        request.append("securities", index_ticker)
        request.append("fields", "INDX_MWEIGHT_HIST")
        
        if date:
            overrides = request.getElement("overrides")
            override = overrides.appendElement()
            override.setElement("fieldId", "END_DATE_OVERRIDE")
            override.setElement("value", date)
        
        logger.info(f"Fetching constituents for {index_ticker}...")
        self.session.sendRequest(request)
        
        constituents = []
        weights = []
        
        try:
            while True:
                event = self.session.nextEvent(500)
                for msg in event:
                    if msg.messageType() == "ReferenceDataResponse":
                        secData = msg.getElement("securityData")
                        
                        if secData.hasElement("fieldData"):
                            fieldData = secData.getElement("fieldData")
                            
                            if fieldData.hasElement("INDX_MWEIGHT_HIST"):
                                weightsData = fieldData.getElement("INDX_MWEIGHT_HIST")
                                
                                for i in range(weightsData.numValues()):
                                    constituent = weightsData.getValue(i)
                                    
                                    if constituent.hasElement("Member Ticker and Exchange Code"):
                                        ticker = constituent.getElementAsString("Member Ticker and Exchange Code")
                                        constituents.append(ticker + " Equity")  # Add Equity suffix for Bloomberg
                                        
                                    if constituent.hasElement("Percent Weight"):
                                        weight = constituent.getElementAsFloat("Percent Weight")
                                        weights.append(weight)
                    
                if event.eventType() == blpapi.Event.RESPONSE:
                    break
        
            logger.info(f"Found {len(constituents)} constituents for {index_ticker}.")
            
            # Create a DataFrame with tickers and weights
            if constituents:
                constituent_df = pd.DataFrame({
                    'ticker': constituents,
                    'weight': weights
                })
                return constituent_df
            else:
                logger.warning(f"No constituents found for {index_ticker}. Using fallback list.")
                return self.get_fallback_constituents(index_ticker)
        
        except Exception as e:
            logger.error(f"Error fetching constituents for {index_ticker}: {e}")
            logger.info("Using fallback list of constituents...")
            return self.get_fallback_constituents(index_ticker)

    def get_sector_etfs(self):
        """
        Return a list of major sector ETFs
        
        Returns:
        --------
        DataFrame
            DataFrame with ETF tickers and equal weights
        """
        # Major sector ETFs
        sector_etfs = [
            "SPY US Equity",  # S&P 500
            "QQQ US Equity",  # Nasdaq 100
            "IWM US Equity",  # Russell 2000
            "XLF US Equity",  # Financial
            "XLE US Equity",  # Energy
            "XLK US Equity",  # Technology
            "XLV US Equity",  # Healthcare
            "XLY US Equity",  # Consumer Discretionary
            "XLP US Equity",  # Consumer Staples
            "XLU US Equity",  # Utilities
            "XLI US Equity",  # Industrials
            "XLB US Equity",  # Materials
            "XLRE US Equity", # Real Estate
            "XLC US Equity",  # Communication Services
            "GLD US Equity",  # Gold
            "TLT US Equity",  # Long-Term Treasury
            "IEF US Equity",  # Intermediate Treasury
            "SHY US Equity",  # Short-Term Treasury
            "LQD US Equity",  # Corporate Bonds
            "HYG US Equity",  # High Yield Bonds
        ]
        
        # Create DataFrame with equal weights
        weights = [1.0/len(sector_etfs)] * len(sector_etfs)
        etf_df = pd.DataFrame({
            'ticker': sector_etfs,
            'weight': weights
        })
        
        logger.info(f"Using {len(sector_etfs)} major ETFs.")
        
        return etf_df

    def get_global_etfs(self):
        """
        Return a list of global ETFs representing different asset classes
        
        Returns:
        --------
        DataFrame
            DataFrame with ETF tickers and equal weights
        """
        # Global ETFs representing various asset classes
        global_etfs = [
            # US Equity
            "SPY US Equity",  # S&P 500
            "QQQ US Equity",  # Nasdaq 100
            "IWM US Equity",  # Russell 2000
            "MDY US Equity",  # S&P MidCap 400
            
            # International Equity
            "EFA US Equity",  # MSCI EAFE (Developed Markets)
            "EEM US Equity",  # MSCI Emerging Markets
            "VGK US Equity",  # FTSE Europe
            "EWJ US Equity",  # MSCI Japan
            "FXI US Equity",  # FTSE China 50
            
            # Fixed Income
            "AGG US Equity",  # Barclays Aggregate Bond
            "TLT US Equity",  # 20+ Year Treasury
            "IEF US Equity",  # 7-10 Year Treasury
            "LQD US Equity",  # Investment Grade Corporate
            "HYG US Equity",  # High Yield Corporate
            "EMB US Equity",  # Emerging Markets Bonds
            
            # Commodities
            "GLD US Equity",  # Gold
            "SLV US Equity",  # Silver
            "USO US Equity",  # Oil
            "DBC US Equity",  # Commodities Basket
            
            # Real Estate
            "VNQ US Equity",  # US REITs
            "REM US Equity",  # Mortgage REITs
            
            # Alternative
            "VXX US Equity",  # Volatility
        ]
        
        # Create DataFrame with equal weights
        weights = [1.0/len(global_etfs)] * len(global_etfs)
        etf_df = pd.DataFrame({
            'ticker': global_etfs,
            'weight': weights
        })
        
        logger.info(f"Using {len(global_etfs)} global ETFs.")
        
        return etf_df
    
    def get_crypto_tickers(self):
        """
        Return a list of major cryptocurrency tickers in Bloomberg format
        
        Returns:
        --------
        DataFrame
            DataFrame with cryptocurrency tickers and equal weights
        """
        # Major cryptocurrencies in Bloomberg format
        crypto_tickers = [
            "XBTUSD Curncy",  # Bitcoin/USD
            "XETUSD Curncy",  # Ethereum/USD
            "XRPUSD Curncy",  # Ripple/USD
            "XLCUSD Curncy",  # Litecoin/USD
            "XBNUSD Curncy",  # Binance Coin/USD
            "XADUSD Curncy",  # Cardano/USD
            "XDOUSD Curncy",  # Dogecoin/USD
            "XTZUSD Curncy",  # Tezos/USD
            "SOLUSD Curncy",  # Solana/USD
            "MATUSD Curncy",  # Polygon/USD
            "LINKUSD Curncy", # Chainlink/USD
            "UNIUSD Curncy",  # Uniswap/USD
            "AVAXUSD Curncy", # Avalanche/USD
            "DOTUSD Curncy",  # Polkadot/USD
        ]
        
        # Create DataFrame with equal weights
        weights = [1.0/len(crypto_tickers)] * len(crypto_tickers)
        crypto_df = pd.DataFrame({
            'ticker': crypto_tickers,
            'weight': weights
        })
        
        logger.info(f"Using {len(crypto_tickers)} cryptocurrencies.")
        
        return crypto_df

    def get_major_fx_pairs(self):
        """
        Return a list of major FX currency pairs in Bloomberg format
        
        Returns:
        --------
        DataFrame
            DataFrame with FX tickers and equal weights
        """
        # Major FX pairs in Bloomberg format
        fx_tickers = [
            "EURUSD Curncy",  # EUR/USD
            "USDJPY Curncy",  # USD/JPY
            "GBPUSD Curncy",  # GBP/USD
            "AUDUSD Curncy",  # AUD/USD
            "USDCAD Curncy",  # USD/CAD
            "USDCHF Curncy",  # USD/CHF
            "NZDUSD Curncy",  # NZD/USD
            "EURJPY Curncy",  # EUR/JPY
            "EURGBP Curncy",  # EUR/GBP
            "EURCHF Curncy",  # EUR/CHF
            "GBPJPY Curncy",  # GBP/JPY
            "CADJPY Curncy",  # CAD/JPY
            "AUDJPY Curncy",  # AUD/JPY
            "EURAUD Curncy",  # EUR/AUD
            "EURNZD Curncy",  # EUR/NZD
        ]
        
        # Create DataFrame with equal weights
        weights = [1.0/len(fx_tickers)] * len(fx_tickers)
        fx_df = pd.DataFrame({
            'ticker': fx_tickers,
            'weight': weights
        })
        
        logger.info(f"Using {len(fx_tickers)} FX pairs.")
        
        return fx_df

    def get_fallback_constituents(self, index_ticker):
        """
        Return a fallback list of assets based on the requested index
        
        Parameters:
        -----------
        index_ticker : str
            Bloomberg index ticker
            
        Returns:
        --------
        DataFrame
            DataFrame with tickers and weights
        """
        # Based on the requested index, return different sets of assets
        if "SPX" in index_ticker or "SP500" in index_ticker:
            # Top S&P 500 stocks by market cap
            major_tickers = [
                "AAPL US Equity",  # Apple
                "MSFT US Equity",  # Microsoft
                "AMZN US Equity",  # Amazon
                "NVDA US Equity",  # NVIDIA
                "GOOGL US Equity", # Alphabet A
                "GOOG US Equity",  # Alphabet C
                "META US Equity",  # Meta Platforms
                "BRK/B US Equity", # Berkshire Hathaway B
                "TSLA US Equity",  # Tesla
                "JPM US Equity",   # JPMorgan Chase
                "JNJ US Equity",   # Johnson & Johnson
                "V US Equity",     # Visa
                "UNH US Equity",   # UnitedHealth Group
                "PG US Equity",    # Procter & Gamble
                "MA US Equity",    # Mastercard
                "HD US Equity",    # Home Depot
                "AVGO US Equity",  # Broadcom
                "MRK US Equity",   # Merck & Co.
                "CVX US Equity",   # Chevron
                "KO US Equity",    # Coca-Cola
            ]
            logger.info(f"Using {len(major_tickers)} S&P 500 stocks.")
        elif "ETF" in index_ticker or "SECTOR" in index_ticker:
            return self.get_sector_etfs()
        elif "GLOBAL" in index_ticker:
            return self.get_global_etfs()
        elif "CRYPTO" in index_ticker:
            return self.get_crypto_tickers()
        elif "FX" in index_ticker:
            return self.get_major_fx_pairs()
        else:
            # Default to major ETFs
            return self.get_sector_etfs()
        
        # Create DataFrame with equal weights if not returned already
        weights = [1.0/len(major_tickers)] * len(major_tickers)
        constituent_df = pd.DataFrame({
            'ticker': major_tickers,
            'weight': weights
        })
        
        return constituent_df


class ModifiedHRP:
    """
    Implementation of the Modified Hierarchical Risk Parity approach
    based on Molyboga (2020)
    """
    
    def __init__(self, tickers=None):
        """
        Initialize the Modified Hierarchical Risk Parity object
        
        Parameters:
        -----------
        tickers : list
            List of tickers to include in the portfolio
        """
        self.tickers = tickers
        self.prices = None
        self.returns = None
        self.cov_matrix = None
        self.corr_matrix = None
        self.dist_matrix = None
        self.link_matrix = None
        self.sort_ix = None
        self.hrp_weights = None
        self.hrpc_weights = None
        self.hrpce_weights = None
        self.mhrp_weights = None
        self.target_vol = 0.12  # 12% annualized volatility target
        
        # Initialize Bloomberg connection
        self.bloomberg = None
    
    def connect_to_bloomberg(self):
        """
        Connect to Bloomberg API
        
        Returns:
        --------
        bool
            True if connection successful, False otherwise
        """
        try:
            self.bloomberg = BloombergDataFetcher()
            self.bloomberg.start_session()
            return True
        except Exception as e:
            logger.error(f"Could not connect to Bloomberg: {e}")
            return False
    
    def fetch_data(self, start_date, end_date, field='PX_LAST', period='WEEKLY'):
        """
        Fetch data from Bloomberg
        
        Parameters:
        -----------
        start_date : str
            Start date in format 'YYYY-MM-DD'
        end_date : str
            End date in format 'YYYY-MM-DD'
        field : str
            Bloomberg field to fetch
        period : str
            Periodicity ('DAILY', 'WEEKLY', 'MONTHLY')
            
        Returns:
        --------
        DataFrame of prices
        """
        if self.bloomberg is None:
            success = self.connect_to_bloomberg()
            if not success:
                logger.warning("Bloomberg connection not available. Using sample data.")
                return self._generate_sample_data(start_date, end_date)
        
        if self.tickers is None or len(self.tickers) == 0:
            raise ValueError("No tickers specified")
        
        logger.info(f"Fetching data for {len(self.tickers)} tickers...")
        
        # Format dates for Bloomberg query
        start_date_fmt = start_date.replace('-', '')
        end_date_fmt = end_date.replace('-', '')
        
        try:
            # Fetch data from Bloomberg
            prices = self.bloomberg.fetch_historical_data(
                tickers=self.tickers,
                fields=[field],
                start_date=start_date_fmt,
                end_date=end_date_fmt,
                period=period
            )
            
            if prices is None or prices.empty:
                raise ValueError("No valid data retrieved from Bloomberg")
                
            # Store the price data and tickers
            self.prices = prices
            self.tickers = list(prices.columns)
            
            logger.info(f"Successfully fetched data for {len(self.tickers)} tickers")
            
            return prices
            
        except Exception as e:
            logger.error(f"Error fetching data from Bloomberg: {e}")
            return self._generate_sample_data(start_date, end_date)
    
    def _generate_sample_data(self, start_date, end_date):
        """
        Generate sample data for testing
        
        Parameters:
        -----------
        start_date : str
            Start date in format 'YYYY-MM-DD'
        end_date : str
            End date in format 'YYYY-MM-DD'
            
        Returns:
        --------
        DataFrame of prices
        """
        if self.tickers is None or len(self.tickers) == 0:
            # Create sample tickers for ETFs
            self.tickers = [f'ETF_{i}' for i in range(20)]
        
        # Convert dates to datetime
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        
        # Generate date range
        date_range = pd.date_range(start=start_date, end=end_date, freq='W-FRI')  # Weekly data
        
        # Set up parameters for simulation
        n_assets = len(self.tickers)
        n_periods = len(date_range)
        
        # Generate correlation matrix with cluster structure
        # This simulates similar asset classes that might be correlated
        np.random.seed(42)  # For reproducibility
        
        # Base correlation matrix (low correlation between assets)
        corr_matrix = np.eye(n_assets) * 0.7 + np.ones((n_assets, n_assets)) * 0.1
        np.fill_diagonal(corr_matrix, 1.0)
        
        # Add cluster structure (higher correlation within clusters)
        cluster_size = n_assets // 4  # Divide into 4 clusters (e.g., equity, fixed income, commodities, alternatives)
        for i in range(0, n_assets, cluster_size):
            end_idx = min(i + cluster_size, n_assets)
            cluster_indices = range(i, end_idx)
            for idx1 in cluster_indices:
                for idx2 in cluster_indices:
                    if idx1 != idx2:
                        corr_matrix[idx1, idx2] = 0.5 + np.random.rand() * 0.3  # Higher correlation within cluster
        
        # Ensure correlation matrix is symmetric and positive definite
        corr_matrix = (corr_matrix + corr_matrix.T) / 2
        min_eig = np.min(np.linalg.eigvalsh(corr_matrix))
        if min_eig < 0:
            # Add a small positive value to the diagonal
            corr_matrix += (-min_eig + 1e-8) * np.eye(n_assets)
        
        # Generate volatilities for each asset (typical range for ETFs)
        vols = np.random.uniform(0.10, 0.30, n_assets)
        
        # Create covariance matrix
        cov_matrix = np.outer(vols, vols) * corr_matrix
        
        # Generate weekly returns
        weekly_returns = np.random.multivariate_normal(
            mean=np.zeros(n_assets) + 0.001,  # Small positive drift
            cov=cov_matrix / 52,  # Weekly covariance
            size=n_periods
        )
        
        # Convert returns to prices starting at 100
        prices = 100 * np.cumprod(1 + weekly_returns, axis=0)
        
        # Create DataFrame
        prices_df = pd.DataFrame(
            prices, 
            index=date_range, 
            columns=self.tickers
        )
        
        # Store the price data
        self.prices = prices_df
        
        logger.info(f"Generated sample data for {n_assets} assets from {start_date} to {end_date}")
        return prices_df
    
    def calculate_returns(self):
        """
        Calculate returns from price data
        
        Returns:
        --------
        DataFrame of returns
        """
        if self.prices is None:
            raise ValueError("No price data available. Call fetch_data() first.")
        
        # Calculate returns
        self.returns = self.prices.pct_change().dropna()
        
        return self.returns
    
    def _exponentially_weighted_covariance_with_shrinkage(self, half_life=21):
        """
        Calculate exponentially weighted covariance matrix with Ledoit-Wolf shrinkage
        
        Parameters:
        -----------
        half_life : int
            Half-life for exponential weighting in days
            
        Returns:
        --------
        tuple: (covariance matrix, correlation matrix)
        """
        if self.returns is None:
            self.calculate_returns()
            
        # Calculate decay factor
        decay_factor = 0.5 ** (1 / half_life)
        
        # Calculate weights for exponential weighting
        n_periods = len(self.returns)
        weights = np.array([decay_factor ** i for i in range(n_periods)])
        weights = weights[::-1]  # Reverse to give more weight to recent observations
        weights = weights / weights.sum()  # Normalize
        
        # Calculate weighted mean returns
        weighted_mean = np.zeros(len(self.returns.columns))
        for i, w in enumerate(weights):
            weighted_mean += w * self.returns.iloc[i].values
            
        # Calculate weighted covariance matrix
        exp_weighted_cov = np.zeros((len(self.returns.columns), len(self.returns.columns)))
        for i, w in enumerate(weights):
            deviation = self.returns.iloc[i].values - weighted_mean
            exp_weighted_cov += w * np.outer(deviation, deviation)
        
        # Manual implementation of Ledoit-Wolf shrinkage
        # Step 1: Estimate sample covariance matrix
        sample_cov = exp_weighted_cov
        
        # Step 2: Compute the target matrix (identity scaled by average variance)
        avg_var = np.mean(np.diag(sample_cov))
        target = np.eye(len(sample_cov)) * avg_var
        
        # Step 3: Estimate optimal shrinkage intensity
        # This is a simplified approach - we use a fixed shrinkage parameter
        # In practice, this would be estimated from the data
        shrinkage = 0.2  # Fixed shrinkage parameter as used in Molyboga (2020)
        
        # Step 4: Combine the sample covariance and target
        shrunk_cov = (1 - shrinkage) * sample_cov + shrinkage * target
        
        # Calculate correlation matrix
        std_devs = np.sqrt(np.diag(shrunk_cov))
        corr_matrix = shrunk_cov / np.outer(std_devs, std_devs)
        
        # Ensure correlation matrix is valid
        corr_matrix = np.clip(corr_matrix, -1, 1)
        np.fill_diagonal(corr_matrix, 1.0)
        
        # Convert to DataFrames
        cov_df = pd.DataFrame(shrunk_cov, index=self.returns.columns, columns=self.returns.columns)
        corr_df = pd.DataFrame(corr_matrix, index=self.returns.columns, columns=self.returns.columns)
        
        return cov_df, corr_df
        
    def get_clusters(self, correlation_matrix=None, method='single'):
        """
        Perform hierarchical clustering on the correlation matrix
        
        Parameters:
        -----------
        correlation_matrix : DataFrame, optional
            Correlation matrix to use (if None, uses sample correlation)
        method : str
            Linkage method to use
            
        Returns:
        --------
        tuple: (linkage matrix, sorted indices)
        """
        if correlation_matrix is None:
            if self.returns is None:
                self.calculate_returns()
            correlation_matrix = self.returns.corr()
        
        # Convert correlation to distance
        distance = np.sqrt((1 - correlation_matrix) / 2)
        
        # Convert distance matrix to condensed form
        dist_condensed = squareform(distance)
        
        # Perform hierarchical clustering
        link = linkage(dist_condensed, method=method)
        
        # Store for later use
        self.corr_matrix = correlation_matrix
        self.dist_matrix = distance
        self.link_matrix = link
        
        return link, correlation_matrix.index
    
    def plot_dendrogram(self, link=None, labels=None, filename='dendrogram.png'):
        """
        Plot dendrogram of the hierarchical clustering
        
        Parameters:
        -----------
        link : ndarray, optional
            Linkage matrix (if None, uses stored linkage matrix)
        labels : list, optional
            Labels for the dendrogram (if None, uses tickers)
        filename : str
            Filename to save the plot
        """
        if link is None:
            if self.link_matrix is None:
                self.get_clusters()
            link = self.link_matrix
            
        if labels is None:
            labels = self.tickers
            
        plt.figure(figsize=(10, 6))
        dendrogram(link, labels=labels, leaf_rotation=90)
        plt.title('Hierarchical Clustering Dendrogram')
        plt.xlabel('Assets')
        plt.ylabel('Distance')
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()
        
    def quasi_diagonalize(self, link=None, labels=None):
        """
        Quasi-diagonalize the correlation matrix using hierarchical clustering
        
        Parameters:
        -----------
        link : ndarray, optional
            Linkage matrix (if None, uses stored linkage matrix)
        labels : list, optional
            Labels for the dendrogram (if None, uses tickers)
            
        Returns:
        --------
        list: Sorted indices
        """
        if link is None:
            if self.link_matrix is None:
                self.get_clusters()
            link = self.link_matrix
            
        if labels is None:
            if self.corr_matrix is None:
                self.get_clusters()
            labels = self.corr_matrix.index
            
        # Get sort order from linkage matrix
        sort_ix = self._get_quasi_diag(link)
        sorted_labels = [labels[i] for i in sort_ix]
        
        # Store sorted indices
        self.sort_ix = sort_ix
        
        return sorted_labels
    
    def _get_quasi_diag(self, link):
        """
        Helper method for quasi-diagonalization
        
        Parameters:
        -----------
        link : ndarray
            Linkage matrix
            
        Returns:
        --------
        list: Sorted indices
        """
        # Sort clustered items by distance
        n = len(link) + 1
        sorted_ix = []  # Initialize with empty list
        
        # Recursive function to get ordering
        def get_children(id):
            if id < n:  # Original item
                return [id]
            else:  # Cluster
                # Find the cluster's children
                idx = id - n
                left = int(link[idx, 0])
                right = int(link[idx, 1])
                return get_children(left) + get_children(right)
        
        # Start with the root cluster
        sorted_ix = get_children(2*n - 2)
        return sorted_ix
    
    def recursive_bisection(self, cov, sort_ix=None, inv_var=True):
        """
        Perform recursive bisection to allocate weights
        
        Parameters:
        -----------
        cov : DataFrame
            Covariance matrix
        sort_ix : list, optional
            Sorted indices from quasi-diagonalization
        inv_var : bool
            If True, use inverse variance allocation (original HRP)
            If False, use equal volatility allocation (HRPCE)
            
        Returns:
        --------
        Series: Portfolio weights
        """
        if sort_ix is None:
            if self.sort_ix is None:
                self.quasi_diagonalize()
            sort_ix = self.sort_ix
            
        # Initialize weights
        w = pd.Series(1, index=cov.index)
        
        # Start recursive bisection
        clusters = [sort_ix]
        while len(clusters) > 0:
            cluster = clusters.pop(0)  # Get the first cluster
            
            # If cluster has more than one item, split it
            if len(cluster) > 1:
                # Find the middle point
                mid_point = len(cluster) // 2
                
                # Split into left and right clusters
                left_cluster = cluster[:mid_point]
                right_cluster = cluster[mid_point:]
                
                # Add new clusters to the list
                clusters.append(left_cluster)
                clusters.append(right_cluster)
                
                # Get indices for left and right clusters
                left_indices = [cov.index[i] for i in left_cluster]
                right_indices = [cov.index[i] for i in right_cluster]
                
                # Calculate risk measures for allocation
                if inv_var:  # Original HRP - Inverse Variance Allocation
                    left_risk = np.sum(np.diag(cov.loc[left_indices, left_indices].values))
                    right_risk = np.sum(np.diag(cov.loc[right_indices, right_indices].values))
                else:  # HRPCE - Equal Volatility Allocation
                    left_risk = np.sum(np.sqrt(np.diag(cov.loc[left_indices, left_indices].values)))
                    right_risk = np.sum(np.sqrt(np.diag(cov.loc[right_indices, right_indices].values)))
                
                # Calculate alpha (risk-based weight factor)
                alpha_left = 1 - left_risk / (left_risk + right_risk)
                alpha_right = 1 - right_risk / (left_risk + right_risk)
                
                # Update weights
                w[left_indices] *= alpha_left
                w[right_indices] *= alpha_right
                
        # Normalize weights to sum to 1
        w = w / w.sum()
        
        return w
    
    def calculate_portfolio_volatility(self, weights, cov_matrix=None):
        """
        Calculate portfolio volatility
        
        Parameters:
        -----------
        weights : Series
            Portfolio weights
        cov_matrix : DataFrame, optional
            Covariance matrix (if None, uses sample covariance)
            
        Returns:
        --------
        float: Annualized portfolio volatility
        """
        if cov_matrix is None:
            if self.returns is None:
                self.calculate_returns()
            cov_matrix = self.returns.cov()
            
        weights_array = np.array(weights)
        portfolio_variance = weights_array.T @ cov_matrix.values @ weights_array
        portfolio_volatility = np.sqrt(portfolio_variance) * np.sqrt(52)  # Annualized (assuming weekly data)
        
        return portfolio_volatility
    
    def apply_volatility_targeting(self, weights, cov_matrix=None, target_vol=None):
        """
        Apply volatility targeting to portfolio weights
        
        Parameters:
        -----------
        weights : Series
            Portfolio weights
        cov_matrix : DataFrame, optional
            Covariance matrix (if None, uses sample covariance)
        target_vol : float, optional
            Target annualized volatility (if None, uses stored target)
            
        Returns:
        --------
        Series: Scaled portfolio weights
        """
        if target_vol is None:
            target_vol = self.target_vol
            
        # Calculate current portfolio volatility
        current_vol = self.calculate_portfolio_volatility(weights, cov_matrix)
        
        # Calculate scaling factor
        scaling_factor = target_vol / current_vol
        
        # Scale weights
        scaled_weights = weights * scaling_factor
        
        return scaled_weights
    
    def calculate_all_portfolios(self):
        """
        Calculate weights for all portfolio approaches
        
        Returns:
        --------
        dict: Dictionary of portfolio weights
        """
        if self.returns is None:
            self.calculate_returns()
            
        # Original HRP approach - Sample covariance and inverse variance allocation
        sample_cov = self.returns.cov()
        sample_corr = self.returns.corr()
        
        # Get clusters using sample correlation
        self.get_clusters(sample_corr)
        self.quasi_diagonalize()
        
        # Original HRP weights (sample covariance, inverse variance allocation)
        self.hrp_weights = self.recursive_bisection(sample_cov, inv_var=True)
        
        # Enhanced covariance estimation
        exp_cov, exp_corr = self._exponentially_weighted_covariance_with_shrinkage()
        
        # Get clusters using exponential weighted correlation
        self.get_clusters(exp_corr)
        self.quasi_diagonalize()
        
        # HRPC weights (enhanced covariance, inverse variance allocation)
        self.hrpc_weights = self.recursive_bisection(exp_cov, inv_var=True)
        
        # HRPCE weights (enhanced covariance, equal volatility allocation)
        self.hrpce_weights = self.recursive_bisection(exp_cov, inv_var=False)
        
        # MHRP weights (enhanced covariance, equal volatility allocation, with volatility targeting)
        self.mhrp_weights = self.apply_volatility_targeting(self.hrpce_weights, exp_cov)
        
        # Return all portfolio weights
        return {
            'HRP': self.hrp_weights,
            'HRPC': self.hrpc_weights,
            'HRPCE': self.hrpce_weights,
            'MHRP': self.mhrp_weights
        }
    
    def plot_portfolio_weights(self, portfolio_weights=None, filename='portfolio_weights.png'):
        """
        Plot portfolio weights for different approaches
        
        Parameters:
        -----------
        portfolio_weights : dict, optional
            Dictionary of portfolio weights (if None, calculates weights)
        filename : str
            Filename to save the plot
        """
        if portfolio_weights is None:
            portfolio_weights = self.calculate_all_portfolios()
            
        # Convert to DataFrame for plotting
        weights_df = pd.DataFrame(portfolio_weights)
        
        # Sort by HRP weights for better visualization
        weights_df = weights_df.sort_values('HRP', ascending=False)
        
        # Plot
        plt.figure(figsize=(12, 8))
        weights_df.plot(kind='bar', figsize=(14, 7))
        plt.title('Portfolio Weights Comparison')
        plt.xlabel('Assets')
        plt.ylabel('Weight')
        plt.legend(title='Method')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()
        
        # Also save weights to CSV
        weights_df.to_csv('portfolio_weights.csv')
        
        return weights_df
    
    def calculate_portfolio_statistics(self, portfolio_weights=None, returns=None):
        """
        Calculate portfolio statistics for different approaches
        
        Parameters:
        -----------
        portfolio_weights : dict, optional
            Dictionary of portfolio weights (if None, calculates weights)
        returns : DataFrame, optional
            Returns data (if None, uses stored returns)
            
        Returns:
        --------
        DataFrame: Portfolio statistics
        """
        if portfolio_weights is None:
            portfolio_weights = self.calculate_all_portfolios()
            
        if returns is None:
            if self.returns is None:
                self.calculate_returns()
            returns = self.returns
            
        # Calculate portfolio returns
        portfolio_returns = {}
        for method, weights in portfolio_weights.items():
            # Align weights with returns
            common_assets = returns.columns.intersection(weights.index)
            aligned_weights = weights.loc[common_assets]
            aligned_weights = aligned_weights / aligned_weights.sum()  # Renormalize
            
            # Calculate portfolio return series
            portfolio_returns[method] = returns[common_assets].dot(aligned_weights)
        
        # Convert to DataFrame
        portfolio_returns_df = pd.DataFrame(portfolio_returns)
        
        # Calculate statistics
        stats = pd.DataFrame({
            method: {
                'Annual Return': returns_series.mean() * 52,
                'Annual Volatility': returns_series.std() * np.sqrt(52),
                'Sharpe Ratio': (returns_series.mean() * 52) / (returns_series.std() * np.sqrt(52)),
                'Max Drawdown': (returns_series.cumsum() - returns_series.cumsum().cummax()).min(),
                'Skewness': returns_series.skew(),
                'Kurtosis': returns_series.kurtosis(),
                'Positive Periods (%)': (returns_series > 0).mean() * 100,
                'Negative Periods (%)': (returns_series < 0).mean() * 100,
                'Volatility Targeting Applied': method == 'MHRP'
            } for method, returns_series in portfolio_returns.items()
        })
        
        # Save statistics to CSV
        stats.to_csv('portfolio_statistics.csv')
        
        return stats, portfolio_returns_df
    
    def plot_portfolio_performance(self, portfolio_returns):
        """
        Plot portfolio performance
        
        Parameters:
        -----------
        portfolio_returns : DataFrame
            Portfolio returns for different approaches
        """
        # Calculate cumulative returns
        cumulative_returns = (1 + portfolio_returns).cumprod()
        
        # Plot cumulative returns
        plt.figure(figsize=(12, 8))
        cumulative_returns.plot(figsize=(14, 7))
        plt.title('Cumulative Portfolio Returns')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Return')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(title='Method')
        plt.tight_layout()
        plt.savefig('cumulative_returns.png')
        plt.close()
        
        # Plot drawdowns
        drawdowns = cumulative_returns / cumulative_returns.cummax() - 1
        
        plt.figure(figsize=(12, 8))
        drawdowns.plot(figsize=(14, 7))
        plt.title('Portfolio Drawdowns')
        plt.xlabel('Date')
        plt.ylabel('Drawdown')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(title='Method')
        plt.tight_layout()
        plt.savefig('drawdowns.png')
        plt.close()
        
        # Plot monthly returns box plot
        plt.figure(figsize=(12, 8))
        portfolio_returns.boxplot(figsize=(14, 7))
        plt.title('Period Returns Distribution')
        plt.xlabel('Method')
        plt.ylabel('Period Return')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig('return_distribution.png')
        plt.close()
        
        # Save cumulative returns to CSV
        cumulative_returns.to_csv('cumulative_returns.csv')
        
    def backtest_portfolios(self, window_size=52, rebalance_freq=4):
        """
        Backtest all portfolio approaches with rolling window
        
        Parameters:
        -----------
        window_size : int
            Size of the rolling window in periods
        rebalance_freq : int
            Rebalancing frequency in periods
            
        Returns:
        --------
        tuple: (portfolio statistics, portfolio returns)
        """
        if self.returns is None:
            self.calculate_returns()
            
        # Initialize DataFrames to store results
        portfolio_returns = {
            'HRP': pd.Series(index=self.returns.index[window_size:]),
            'HRPC': pd.Series(index=self.returns.index[window_size:]),
            'HRPCE': pd.Series(index=self.returns.index[window_size:]),
            'MHRP': pd.Series(index=self.returns.index[window_size:])
        }
        
        logger.info(f"Running backtest with window size {window_size} and rebalance frequency {rebalance_freq}")
        
        # Loop through time periods
        for i in tqdm(range(window_size, len(self.returns), rebalance_freq), desc="Backtesting"):
            # Get training window
            train_returns = self.returns.iloc[i-window_size:i]
            
            # Skip if training window is too small
            if len(train_returns) < window_size / 2:
                logger.warning(f"Not enough data at period {i}. Skipping.")
                continue
                
            # Create a new MHRP object for each period to avoid data leakage
            mhrp = ModifiedHRP(tickers=train_returns.columns.tolist())
            mhrp.returns = train_returns
            
            # Calculate portfolio weights
            weights = mhrp.calculate_all_portfolios()
            
            # Get test window
            end_idx = min(i + rebalance_freq, len(self.returns))
            test_returns = self.returns.iloc[i:end_idx]
            
            if len(test_returns) == 0:
                logger.warning(f"No test data at period {i}. Skipping.")
                continue
            
            # Calculate portfolio returns for each method
            for method, method_weights in weights.items():
                # Align weights with returns
                common_assets = test_returns.columns.intersection(method_weights.index)
                
                if len(common_assets) == 0:
                    logger.warning(f"No common assets for {method} at period {i}. Skipping.")
                    continue
                    
                aligned_weights = method_weights.loc[common_assets]
                aligned_weights = aligned_weights / aligned_weights.sum()  # Renormalize
                
                # Calculate portfolio return series
                period_returns = test_returns[common_assets].dot(aligned_weights)
                
                # Store in results
                portfolio_returns[method].iloc[i-window_size:i-window_size+(end_idx-i)] = period_returns.values
        
        # Convert to DataFrame
        portfolio_returns_df = pd.DataFrame(portfolio_returns).dropna(how='all')
        
        # Calculate statistics
        stats = pd.DataFrame({
            method: {
                'Annual Return': returns_series.mean() * 52,
                'Annual Volatility': returns_series.std() * np.sqrt(52),
                'Sharpe Ratio': (returns_series.mean() * 52) / (returns_series.std() * np.sqrt(52)),
                'Max Drawdown': (returns_series.cumsum() - returns_series.cumsum().cummax()).min(),
                'Skewness': returns_series.skew(),
                'Kurtosis': returns_series.kurtosis(),
                'Positive Periods (%)': (returns_series > 0).mean() * 100,
                'Negative Periods (%)': (returns_series < 0).mean() * 100
            } for method, returns_series in portfolio_returns_df.items() if not returns_series.isna().all()
        })
        
        # Save backtest results to CSV
        portfolio_returns_df.to_csv('backtest_returns.csv')
        stats.to_csv('backtest_statistics.csv')
        
        # Log stats
        logger.info("Backtest Statistics:")
        for method in stats.columns:
            logger.info(f"\n{method} Results:")
            for metric, value in stats[method].items():
                logger.info(f"{metric}: {value:.4f}")
        
        return stats, portfolio_returns_df


def main():
    """Main function to run the MHRP portfolio optimization."""
    
    # Define parameters
    start_date = '20150101'  # Start date in YYYYMMDD format
    end_date = '20230101'    # End date in YYYYMMDD format
    period = 'WEEKLY'        # Data frequency
    
    # Asset selections to choose from:
    # 1. Major sector ETFs
    # 2. Global ETFs (multi-asset)
    # 3. S&P 500 components
    # 4. Cryptocurrencies
    # 5. FX pairs
    asset_type = "SECTOR"  # Change this to your preferred asset type: "SECTOR", "GLOBAL", "SPX", "CRYPTO", "FX"
    
    # Initialize Bloomberg connection and get tickers
    try:
        logger.info("Connecting to Bloomberg...")
        bloomberg = BloombergDataFetcher()
        bloomberg.start_session()
        
        # Get appropriate tickers based on asset type
        logger.info(f"Getting {asset_type} assets...")
        if asset_type == "SECTOR":
            assets = bloomberg.get_sector_etfs()
        elif asset_type == "GLOBAL":
            assets = bloomberg.get_global_etfs()
        elif asset_type == "CRYPTO":
            assets = bloomberg.get_crypto_tickers()
        elif asset_type == "FX":
            assets = bloomberg.get_major_fx_pairs()
        else:  # SPX
            assets = bloomberg.get_fallback_constituents("SPX Index")
        
        tickers = assets['ticker'].tolist()
        
        logger.info(f"Running MHRP strategy with {len(tickers)} {asset_type} assets")
        
        # Initialize MHRP
        mhrp = ModifiedHRP(tickers=tickers)
        
        # Fetch historical data
        logger.info(f"Fetching historical data from {start_date} to {end_date}...")
        mhrp.bloomberg = bloomberg  # Share Bloomberg connection
        prices = mhrp.fetch_data(
            start_date=start_date, 
            end_date=end_date,
            period=period
        )
        
        # Calculate returns
        returns = mhrp.calculate_returns()
        
        # Plot dendrogram to visualize clusters
        logger.info("Generating hierarchical clustering dendrogram...")
        exp_cov, exp_corr = mhrp._exponentially_weighted_covariance_with_shrinkage()
        link, labels = mhrp.get_clusters(exp_corr, method='complete')  # Use 'complete' linkage for better clusters
        mhrp.plot_dendrogram(link, labels, filename=f'{asset_type}_dendrogram.png')
        
        # Calculate all portfolio weights
        logger.info("Calculating portfolio weights...")
        portfolio_weights = mhrp.calculate_all_portfolios()
        
        # Plot weights
        logger.info("Plotting portfolio weights...")
        weights_df = mhrp.plot_portfolio_weights(portfolio_weights, filename=f'{asset_type}_portfolio_weights.png')
        
        # Calculate and display portfolio statistics
        logger.info("Calculating portfolio statistics...")
        stats, portfolio_returns_df = mhrp.calculate_portfolio_statistics(portfolio_weights)
        print("\nPortfolio Statistics:")
        print(stats)
        
        # Plot performance
        logger.info("Plotting performance metrics...")
        mhrp.plot_portfolio_performance(portfolio_returns_df)
        
        # Run backtest
        logger.info("Running backtest...")
        backtest_stats, backtest_returns = mhrp.backtest_portfolios(
            window_size=52,  # One year of data
            rebalance_freq=4  # Quarterly rebalancing
        )
        
        # Plot backtest results
        logger.info("Plotting backtest results...")
        mhrp.plot_portfolio_performance(backtest_returns)
        
        print("\nBacktest Statistics:")
        print(backtest_stats)
        
        logger.info("Analysis complete! Results saved to CSV files and plots.")
        
    except Exception as e:
        logger.error(f"Error in main: {e}")
        import traceback
        logger.error(traceback.format_exc())
        
    finally:
        # Close Bloomberg session
        if 'bloomberg' in locals():
            bloomberg.stop_session()


if __name__ == "__main__":
    main()