import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.optimize import minimize
from datetime import datetime, timedelta
import os
import blpapi
import logging
from typing import List, Dict, Any, Optional

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Bloomberg API constants
REFDATA_SVC = "//blp/refdata"
HIST_DATA_REQUEST = "HistoricalDataRequest"
SESSION_STARTED = blpapi.Name("SessionStarted")
SESSION_STARTUP_FAILURE = blpapi.Name("SessionStartupFailure")
SERVICE_OPENED = blpapi.Name("ServiceOpened")
SERVICE_OPEN_FAILURE = blpapi.Name("ServiceOpenFailure")
RESPONSE_ERROR = blpapi.Name("ResponseError")
SECURITY_DATA = blpapi.Name("securityData")
FIELD_DATA = blpapi.Name("fieldData")

class BloombergDataFetcher:
    """Class to handle Bloomberg data fetching operations"""

    def __init__(self, host: str = "localhost", port: int = 8194):
        """Initialize the Bloomberg data fetcher

        Args:
            host: Bloomberg server host
            port: Bloomberg server port
        """
        self.host = host
        self.port = port
        self.session = None
        self.refdata_service = None

    def start_session(self) -> bool:
        """Start a Bloomberg API session

        Returns:
            bool: True if session started successfully, False otherwise
        """
        logger.info("Starting Bloomberg API session...")
        
        # Initialize session options
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
        
        # Open the reference data service
        if not self.session.openService(REFDATA_SVC):
            logger.error("Failed to open reference data service.")
            return False
        
        self.refdata_service = self.session.getService(REFDATA_SVC)
        logger.info("Reference data service opened successfully.")
        
        return True

    def stop_session(self) -> None:
        """Stop the Bloomberg API session"""
        if self.session:
            self.session.stop()
            logger.info("Session stopped.")

    def get_historical_data(
        self, 
        securities: List[str], 
        fields: List[str], 
        start_date: str,
        end_date: str,
        periodicity: str = "DAILY"
    ) -> Dict[str, pd.DataFrame]:
        """Get historical data for securities

        Args:
            securities: List of Bloomberg security identifiers
            fields: List of Bloomberg fields to retrieve
            start_date: Start date for data retrieval (YYYY-MM-DD)
            end_date: End date for data retrieval (YYYY-MM-DD)
            periodicity: Data frequency (DAILY, WEEKLY, MONTHLY, etc.)

        Returns:
            Dict[str, pd.DataFrame]: Dictionary of DataFrames containing historical data for each security
        """
        logger.info(f"Fetching historical data for {len(securities)} securities...")
        
        # Create the request
        request = self.refdata_service.createRequest(HIST_DATA_REQUEST)
        
        # Add securities to the request
        for security in securities:
            request.append("securities", security)
        
        # Add fields to the request
        for field in fields:
            request.append("fields", field)
        
        # Set the date range
        request.set("startDate", start_date)
        request.set("endDate", end_date)
        request.set("periodicitySelection", periodicity)
        
        logger.info(f"Request period: {start_date} to {end_date}, Periodicity: {periodicity}")
        
        # Send the request
        self.session.sendRequest(request)
        
        # Process the response
        all_data = {}
        
        while True:
            event = self.session.nextEvent(500)  # Timeout in milliseconds
            
            for msg in event:
                if msg.hasElement(RESPONSE_ERROR):
                    error_info = msg.getElement(RESPONSE_ERROR)
                    logger.error(f"Request failed: {error_info}")
                    return {}
                
                if msg.hasElement(SECURITY_DATA):
                    security_data_array = msg.getElement(SECURITY_DATA)
                    
                    for i in range(security_data_array.numValues()):
                        security_data = security_data_array.getValue(i)
                        security = security_data.getElementAsString("security")
                        
                        if security_data.hasElement("securityError"):
                            error_info = security_data.getElement("securityError")
                            logger.error(f"Error for {security}: {error_info}")
                            continue
                        
                        if security_data.hasElement(FIELD_DATA):
                            field_data = security_data.getElement(FIELD_DATA)
                            data_list = []
                            
                            for j in range(field_data.numValues()):
                                field_values = field_data.getValue(j)
                                data_dict = {"date": field_values.getElementAsDatetime("date")}
                                
                                for field in fields:
                                    if field_values.hasElement(field):
                                        try:
                                            data_dict[field] = field_values.getElementAsFloat(field)
                                        except:
                                            data_dict[field] = field_values.getElementAsString(field)
                                
                                data_list.append(data_dict)
                            
                            if data_list:
                                df = pd.DataFrame(data_list)
                                df.set_index("date", inplace=True)
                                all_data[security] = df
                                logger.info(f"Retrieved {len(df)} data points for {security}")
                            else:
                                logger.warning(f"No data retrieved for {security}")
            
            if event.eventType() == blpapi.Event.RESPONSE:
                break
        
        return all_data

# Generate synthetic data for testing without Bloomberg
def generate_synthetic_data(tickers, start_date, end_date, seed=42):
    """
    Generate synthetic price data for testing when Bloomberg is not available
    
    Parameters:
    tickers (list): List of ticker symbols
    start_date (str): Start date in 'YYYY-MM-DD' format
    end_date (str): End date in 'YYYY-MM-DD' format
    seed (int): Random seed for reproducibility
    
    Returns:
    pd.DataFrame: Synthetic price data
    """
    print("Generating synthetic data for testing...")
    
    # Convert dates to datetime
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    
    # Create date range
    dates = pd.date_range(start=start, end=end, freq='B')
    
    # Initialize DataFrame
    data = pd.DataFrame(index=dates)
    
    # Set random seed for reproducibility
    np.random.seed(seed)
    
    # Generate synthetic prices for each ticker
    for ticker in tickers:
        # Initial price
        price = 100
        
        # Add autocorrelation and market factors to create more realistic data
        # with different asset class behavior and market regimes
        if 'Index' in ticker:
            vol = 0.012  # Higher vol for indices
            drift = 0.0003
            autocorr = 0.05
            market_factor = 0.7  # Higher market exposure
        elif 'Comdty' in ticker and not any(m in ticker for m in ['GCA', 'SIA', 'PLA', 'PAA']):
            vol = 0.018  # Higher vol for commodities
            drift = 0.0002
            autocorr = 0.08
            market_factor = 0.3
        elif any(m in ticker for m in ['GCA', 'SIA', 'PLA', 'PAA']):  # Precious metals
            vol = 0.014
            drift = 0.0002
            autocorr = 0.1
            market_factor = -0.1  # Negative correlation to market
        elif 'Curncy' in ticker:
            vol = 0.008  # Lower vol for currencies
            drift = 0.0
            autocorr = 0.15
            market_factor = 0.0  # Low market exposure
        else:
            vol = 0.01
            drift = 0.0001
            autocorr = 0.05
            market_factor = 0.2
        
        # Generate common market factor
        market = np.random.normal(0.0005, 0.01, len(dates))
        market_cumulative = np.cumsum(market)
        
        # Add some crisis periods to the market factor
        # 2000-2002 dot-com bust
        dotcom_start = np.where(dates >= pd.Timestamp('2000-03-01'))[0][0]
        dotcom_end = np.where(dates <= pd.Timestamp('2002-10-01'))[0][-1]
        market[dotcom_start:dotcom_end] = market[dotcom_start:dotcom_end] - 0.001
        
        # 2008 financial crisis
        crisis_start = np.where(dates >= pd.Timestamp('2007-10-01'))[0][0]
        crisis_end = np.where(dates <= pd.Timestamp('2009-03-01'))[0][-1]
        market[crisis_start:crisis_end] = market[crisis_start:crisis_end] - 0.002
        
        # 2011 European debt crisis
        euro_start = np.where(dates >= pd.Timestamp('2011-05-01'))[0][0]
        euro_end = np.where(dates <= pd.Timestamp('2011-11-01'))[0][-1]
        market[euro_start:euro_end] = market[euro_start:euro_end] - 0.001
        
        # 2015-2016 market correction
        corr_start = np.where(dates >= pd.Timestamp('2015-08-01'))[0][0]
        corr_end = np.where(dates <= pd.Timestamp('2016-02-01'))[0][-1]
        market[corr_start:corr_end] = market[corr_start:corr_end] - 0.001
        
        # Generate price series
        prices = [price]
        returns = [0]
        
        for i in range(1, len(dates)):
            # Add autocorrelation and market factor
            momentum = autocorr * returns[-1]
            market_influence = market_factor * market[i]
            
            # Random return with drift, momentum and market influence
            ret = np.random.normal(drift, vol) + momentum + market_influence
            returns.append(ret)
            price = price * (1 + ret)
            prices.append(price)
        
        data[ticker] = prices
    
    print(f"Generated synthetic data for {len(tickers)} tickers from {start_date} to {end_date}")
    return data

# Define the asset classes and tickers
def get_tickers():
    """Define the futures tickers by asset class"""
    # This follows the list of 59 instruments from Appendix D of the paper
    tickers = {
        'Bonds': [
            'DUA Comdty',  # SCHATZ
            'OEA Comdty',  # BOBL
            'RXA Comdty',  # BUND
            'GA Comdty',   # LONG GILT
            'YMA Comdty',  # AUSTRALIA 3Y
            'XMA Comdty',  # AUSTRALIA 10Y
            'JBA Comdty',  # JGB
            'CNA Comdty',  # CANADA 10Y
            'TUA Comdty',  # US-TREASURY 2Y
            'FVA Comdty',  # US-TREASURY 5Y
            'TYA Comdty',  # US-TREASURY 10Y
            'USA Comdty',  # US-TREASURY 30Y
        ],
        'STIR': [
            'ERA Comdty',  # EURIBOR
            'EDA Comdty',  # EURODOLLAR
            'ESA Comdty',  # EUROSWISS
            'LA Comdty',   # SHORT STERLING
        ],
        'Equity': [
            'CFA Index',   # CAC 40
            'GXA Index',   # DAX
            'ZA Index',    # FTSE 100
            'IBA Index',   # IBEX
            'ESA Index',   # S&P 500
            'NQA Index',   # NASDAQ 100
            'PTA Index',   # S&P TORONTO
            'NXA Index',   # NIKKEI 225
            'XPA Index',   # SPI 200
            'AIA Index',   # FTSE JSE40
            'HIA Index',   # HANG SENG
            'TWA Index',   # MSCI TAIWAN
            'IHA Index',   # NIFTY
            'QZA Index',   # SGX MSCI INDEX
            'VGA Index',   # DJ EURO STOXX 50
        ],
        'Commodities': [
            # Metals
            'HGA Comdty',  # COPPER
            'GCA Comdty',  # GOLD
            'SIA Comdty',  # SILVER
            'PLA Comdty',  # PLATINUM
            'PAA Comdty',  # PALLADIUM
            # Energy
            'COA Comdty',  # BRENT CRUDE
            'CLA Comdty',  # WTI CRUDE
            'HOA Comdty',  # HEATING OIL
            'QSA Comdty',  # GAS OIL
            'NGA Comdty',  # NATURAL GAS
            # Agriculture
            'CTA Comdty',  # COTTON
            'LHA Comdty',  # LEAN HOGS
            'LCA Comdty',  # LIVE CATTLE
            'FCA Comdty',  # FEEDER CATTLE
            'CCA Comdty',  # COCOA
            'KCA Comdty',  # COFFEE
            'SBA Comdty',  # SUGAR
            'CA Comdty',   # CORN
            'SA Comdty',   # SOYBEAN
            'WA Comdty',   # WHEAT
            'SMA Comdty',  # SOYBEAN MEAL
        ],
        'FX': [
            'ADA Curncy',  # AUDUSD
            'CDA Curncy',  # CADUSD
            'SFA Curncy',  # CHFUSD
            'ECA Curncy',  # EURUSD
            'BPA Curncy',  # GBPUSD
            'JYA Curncy',  # JPYUSD
            'PEA Curncy',  # MXPUSD
        ]
    }
    
    # Flatten the dictionary into a list
    all_tickers = []
    for asset_class, asset_tickers in tickers.items():
        all_tickers.extend(asset_tickers)
    
    return tickers, all_tickers

# Calculate EWMA (Exponential Weighted Moving Average)
def calculate_ewma(series, window, adjust=False):
    """Calculate exponential weighted moving average"""
    ewma = series.ewm(span=window, adjust=adjust).mean()
    return ewma

# Calculate EWMA volatility (RiskMetrics methodology)
def calculate_ewma_vol(series, window=20, adjust=False):
    """Calculate exponential weighted moving average volatility"""
    # RiskMetrics methodology uses lambda = 0.94 which corresponds to a window of about 20 days
    vol = np.sqrt(series.ewm(span=window, adjust=adjust).var())
    # Ensure minimum volatility to avoid division by zero
    min_vol = 1e-6 * np.abs(series.mean()) if not np.isnan(series.mean()) else 1e-6
    return np.maximum(vol, min_vol)

# Calculate the trend signal as described in the paper
def calculate_trend_signal(prices, short_window, long_window, vol_window=20):
    """
    Calculate the trend signal as described in the paper with improved stability:
    1. Calculate short and long moving averages
    2. Calculate the difference (DMA)
    3. Normalize by the volatility of DMA
    """
    # Calculate short and long moving averages
    short_ma = calculate_ewma(prices, short_window)
    long_ma = calculate_ewma(prices, long_window)
    
    # Calculate difference
    dma = short_ma - long_ma
    
    # Calculate volatility of DMA
    dma_vol = calculate_ewma_vol(dma, vol_window)
    
    # Ensure volatility is not too small
    dma_vol = np.maximum(dma_vol, 1e-6 * prices.mean())
    
    # Normalize
    signal = dma / dma_vol
    
    # Clip to reasonable values
    signal = np.clip(signal, -5, 5)
    
    return signal

# Calculate risk-adjusted returns
def calculate_risk_adjusted_returns(returns, vol_window=60):
    """Calculate risk-adjusted returns by dividing by rolling volatility"""
    volatility = returns.rolling(window=vol_window, min_periods=30).std() * np.sqrt(252)  # Annualize
    
    # Ensure minimum volatility to avoid division by zero
    min_vol = 0.01  # 1% minimum annualized volatility
    volatility = np.maximum(volatility, min_vol)
    
    risk_adjusted_returns = returns / volatility.shift(1)  # Use previous volatility estimate
    return risk_adjusted_returns

# Logistic function to map signal to position
def logistic_function(x, steepness=1.0, max_value=1.0):
    """Map signal to position using logistic function with better stability"""
    # Clip input to reasonable range
    x = np.clip(x, -10, 10)
    return max_value * (2 / (1 + np.exp(-steepness * x)) - 1)

# Calculate optimal weights for a pair of instruments
def optimal_weights(si, sj, rho):
    """
    Calculate optimal weights for a pair of instruments based on their signals and correlation
    as derived in the paper (Equations 1 and 2) with improved numerical stability
    """
    # Handle edge cases
    if np.abs(rho) >= 0.999:  # Prevent division by zero
        rho = 0.999 * np.sign(rho)
    
    # Clip signals to prevent extreme values
    si = np.clip(si, -5, 5)
    sj = np.clip(sj, -5, 5)
    
    # Calculate denominator
    denom = (1 - rho**2) * (si**2 - 2*rho*si*sj + sj**2)
    
    # Handle numerical issues
    if np.abs(denom) < 1e-6:
        wi = 0.5 * np.sign(si) if si != 0 else 0.5
        wj = 0.5 * np.sign(sj) if sj != 0 else 0.5
    else:
        # Calculate weights
        wi = (si - rho*sj) / np.sqrt(denom)
        wj = (sj - rho*si) / np.sqrt(denom)
    
    # Ensure weights are not extreme
    max_weight = 3.0
    if np.abs(wi) > max_weight or np.abs(wj) > max_weight:
        scaling = max_weight / max(np.abs(wi), np.abs(wj))
        wi *= scaling
        wj *= scaling
    
    return wi, wj

# Decompose weights into absolute and relative components
def decompose_weights(wi, wj, si, sj, rho):
    """
    Decompose the optimal weights into absolute and relative components
    as derived in the paper (Appendix B)
    """
    # Handle edge case
    if np.abs(rho) >= 0.999:
        rho = 0.999 * np.sign(rho)
    
    # Clip signals to prevent extreme values
    si = np.clip(si, -5, 5)
    sj = np.clip(sj, -5, 5)
    
    # Calculate gamma (from Appendix B)
    denom = (1 - rho**2) * (si**2 - 2*rho*si*sj + sj**2)
    if np.abs(denom) < 1e-6:
        gamma = 1
    else:
        gamma = 1 / np.sqrt(denom)
    
    # Limit gamma to prevent numerical issues
    gamma = np.minimum(gamma, 10.0)
    
    # Calculate absolute and relative components (Equations B7-B10)
    wi_abs = gamma * (1 - np.abs(rho)) * si
    wj_abs = gamma * (1 - np.abs(rho)) * sj
    
    wi_rel = gamma * np.abs(rho) * (si - np.sign(rho)*sj)
    wj_rel = gamma * np.abs(rho) * (sj - np.sign(rho)*si)
    
    # Ensure components are not extreme
    max_comp = 3.0
    for comp in [wi_abs, wj_abs, wi_rel, wj_rel]:
        if np.abs(comp) > max_comp:
            comp = max_comp * np.sign(comp)
    
    return wi_abs, wj_abs, wi_rel, wj_rel

# Calculate the signal for the pair with optimal weights
def optimal_pair_signal(si, sj, rho):
    """
    Calculate the signal for the pair with optimal weights
    as derived in the paper (Equation 3) with improved numerical stability
    """
    # Handle edge case
    if np.abs(rho) >= 0.999:
        rho = 0.999 * np.sign(rho)
    
    # Clip signals
    si = np.clip(si, -5, 5)
    sj = np.clip(sj, -5, 5)
    
    # Calculate signal
    num = si**2 - 2*rho*si*sj + sj**2
    denom = 1 - rho**2
    
    # Handle extremely small denominator
    if denom < 1e-6:
        denom = 1e-6
    
    signal = np.sqrt(num / denom)
    
    # Clip to reasonable values
    signal = np.clip(signal, 0, 10)
    
    return signal

# Calculate exposures for absolute trend strategy
def absolute_trend_strategy(signals, volatilities, capital=1000000, target_vol=0.1, weights=None):
    """
    Calculate instrument exposures for the absolute trend strategy
    as described in the paper (Section "Portfolio Construction")
    """
    if weights is None:
        weights = np.ones(len(signals), dtype=np.float64) / len(signals)
    
    # Ensure minimum volatility
    volatilities = np.maximum(volatilities, 0.001)
    
    # Calculate unit risk position
    unit_risk = weights * capital * target_vol / volatilities
    
    # Map signal to leverage factor (-1 to 1)
    leverage = np.array([logistic_function(s) for s in signals], dtype=np.float64)
    
    # Calculate exposures
    exposures = leverage * unit_risk
    
    # Limit leverage
    total_exposure = np.sum(np.abs(exposures))
    max_leverage = 2 * capital  # Maximum 2x leverage (more conservative)
    
    if total_exposure > max_leverage:
        scaling_factor = max_leverage / total_exposure
        exposures = exposures * scaling_factor
    
    return exposures

# Calculate exposures for relative trend strategy
def relative_trend_strategy(signals, correlations, volatilities, capital=1000000, target_vol=0.1, weights=None):
    """
    Calculate instrument exposures for the relative trend strategy
    as described in the paper (Section "Portfolio Construction")
    """
    n = len(signals)
    if weights is None:
        # Equal weight for all pairs
        pair_weights = np.ones((n, n), dtype=np.float64) / (n * (n-1))
        np.fill_diagonal(pair_weights, 0)
    else:
        # Construct pair weights from instrument weights
        pair_weights = np.outer(weights, weights)
        np.fill_diagonal(pair_weights, 0)
        pair_sum = pair_weights.sum()
        if pair_sum > 0:
            pair_weights = pair_weights / pair_sum
    
    # Ensure minimum volatility
    volatilities = np.maximum(volatilities, 0.001)
    
    # Ensure correlation matrix is valid
    correlations = np.clip(correlations, -0.999, 0.999)
    
    # Initialize exposure matrix
    exposures = np.zeros(n, dtype=np.float64)
    
    # Loop through all pairs
    for i in range(n):
        for j in range(i+1, n):
            # Calculate volatility of the pair
            pair_vol = np.sqrt(volatilities[i]**2 + volatilities[j]**2 - 2*correlations[i,j]*volatilities[i]*volatilities[j])
            
            # Ensure minimum pair volatility
            pair_vol = max(pair_vol, 0.001)
            
            # Calculate the signal for the pair (using cross-sectional signal)
            pair_signal = (signals[i] - signals[j]) / pair_vol
            
            # Clip signal to reasonable range
            pair_signal = np.clip(pair_signal, -5, 5)
            
            # Map signal to leverage factor (-1 to 1)
            leverage = logistic_function(pair_signal)
            
            # Calculate unit risk position for the pair
            unit_risk_pair = pair_weights[i,j] * capital * target_vol / pair_vol
            
            # Add exposures with safety check
            i_exposure = leverage * unit_risk_pair / volatilities[i]
            j_exposure = -leverage * unit_risk_pair / volatilities[j]
            
            # Limit individual exposure changes to prevent extreme values
            max_exposure_per_pair = capital * 0.1  # Limit to 10% of capital per pair
            if abs(i_exposure) > max_exposure_per_pair or abs(j_exposure) > max_exposure_per_pair:
                scale = max_exposure_per_pair / max(abs(i_exposure), abs(j_exposure))
                i_exposure *= scale
                j_exposure *= scale
            
            exposures[i] += i_exposure
            exposures[j] += j_exposure
    
    # Limit total leverage
    total_exposure = np.sum(np.abs(exposures))
    max_leverage = 2 * capital  # Maximum 2x leverage (more conservative)
    
    if total_exposure > max_leverage:
        scaling_factor = max_leverage / total_exposure
        exposures = exposures * scaling_factor
    
    return exposures

# Calculate exposures for dynamic trend strategy
def dynamic_trend_strategy(signals, correlations, volatilities, capital=1000000, target_vol=0.1, weights=None):
    """
    Calculate instrument exposures for the dynamic trend strategy
    as described in the paper (Section "Portfolio Construction")
    """
    n = len(signals)
    if weights is None:
        # Equal weight for all pairs
        pair_weights = np.ones((n, n), dtype=np.float64) / (n * (n-1))
        np.fill_diagonal(pair_weights, 0)
    else:
        # Construct pair weights from instrument weights
        pair_weights = np.outer(weights, weights)
        np.fill_diagonal(pair_weights, 0)
        pair_sum = pair_weights.sum()
        if pair_sum > 0:
            pair_weights = pair_weights / pair_sum
    
    # Ensure minimum volatility
    volatilities = np.maximum(volatilities, 0.001)
    
    # Ensure correlation matrix is valid
    correlations = np.clip(correlations, -0.999, 0.999)
    
    # Initialize exposure matrix
    exposures = np.zeros(n, dtype=np.float64)
    
    # Loop through all pairs
    for i in range(n):
        for j in range(i+1, n):
            try:
                # Calculate optimal weights for the pair
                wi, wj = optimal_weights(signals[i], signals[j], correlations[i,j])
                
                # Calculate the signal for the pair with optimal weights
                pair_signal = optimal_pair_signal(signals[i], signals[j], correlations[i,j])
                
                # Map signal to leverage factor (-1 to 1)
                leverage = logistic_function(pair_signal)
                
                # Calculate volatility of the optimal weighted pair
                pair_vol = np.sqrt(wi**2 + wj**2 + 2*wi*wj*correlations[i,j])
                
                # Ensure minimum pair volatility
                pair_vol = max(pair_vol, 0.001)
                
                # Calculate unit risk position for the pair
                unit_risk_pair = pair_weights[i,j] * capital * target_vol / pair_vol
                
                # Add exposures with safety check
                i_exposure = leverage * unit_risk_pair * wi
                j_exposure = leverage * unit_risk_pair * wj
                
                # Limit individual exposure changes to prevent extreme values
                max_exposure_per_pair = capital * 0.1  # Limit to 10% of capital per pair
                if abs(i_exposure) > max_exposure_per_pair or abs(j_exposure) > max_exposure_per_pair:
                    scale = max_exposure_per_pair / max(abs(i_exposure), abs(j_exposure))
                    i_exposure *= scale
                    j_exposure *= scale
                
                exposures[i] += i_exposure
                exposures[j] += j_exposure
            except Exception as e:
                logger.warning(f"Error in dynamic strategy pair {i},{j}: {e}")
                continue
    
    # Limit total leverage
    total_exposure = np.sum(np.abs(exposures))
    max_leverage = 2 * capital  # Maximum 2x leverage (more conservative)
    
    if total_exposure > max_leverage:
        scaling_factor = max_leverage / total_exposure
        exposures = exposures * scaling_factor
    
    return exposures

# Calculate allocation to absolute and relative trends
def calculate_allocation(signals, correlations, weights=None):
    """
    Calculate the allocation to absolute and relative trends
    as described in the paper (Appendix C)
    """
    n = len(signals)
    if weights is None:
        # Equal weight for all pairs
        pair_weights = np.ones((n, n), dtype=np.float64) / (n * (n-1))
        np.fill_diagonal(pair_weights, 0)
    else:
        # Construct pair weights from instrument weights
        pair_weights = np.outer(weights, weights)
        np.fill_diagonal(pair_weights, 0)
        pair_sum = pair_weights.sum()
        if pair_sum > 0:
            pair_weights = pair_weights / pair_sum
    
    # Ensure correlation matrix is valid
    correlations = np.clip(correlations, -0.999, 0.999)
    
    # Clip signals
    signals = np.clip(signals, -5, 5)
    
    # Initialize absolute and relative weight contributions
    absolute_contribution = 0
    relative_contribution = 0
    
    # Loop through all pairs
    for i in range(n):
        for j in range(i+1, n):
            try:
                # Calculate gamma (as in Appendix C)
                denom = (1 - correlations[i,j]**2) * (signals[i]**2 - 2*correlations[i,j]*signals[i]*signals[j] + signals[j]**2)
                if np.abs(denom) < 1e-6:
                    gamma = 1
                else:
                    gamma = 1 / np.sqrt(denom)
                
                # Limit gamma to prevent numerical issues
                gamma = np.minimum(gamma, 10.0)
                
                # Calculate absolute contribution (Equation C5)
                abs_contrib = (1 - np.abs(correlations[i,j])) * (np.abs(signals[i]) + np.abs(signals[j])) * \
                            np.sqrt(1 - correlations[i,j]**2)
                
                # Calculate relative contribution (Equation C6)
                rel_contrib = 2 * np.abs(correlations[i,j]) * np.abs(signals[i] - np.sign(correlations[i,j])*signals[j]) * \
                            np.sqrt(1 - correlations[i,j]**2)
                
                # Clip contributions to reasonable values
                abs_contrib = np.clip(abs_contrib, 0, 10)
                rel_contrib = np.clip(rel_contrib, 0, 10)
                
                # Weight by pair weight
                absolute_contribution += pair_weights[i,j] * abs_contrib
                relative_contribution += pair_weights[i,j] * rel_contrib
            except Exception as e:
                logger.warning(f"Error in allocation calculation for pair {i},{j}: {e}")
                continue
    
    # Normalize with safety check
    total = absolute_contribution + relative_contribution
    if total > 1e-6:
        absolute_allocation = absolute_contribution / total
        relative_allocation = relative_contribution / total
    else:
        absolute_allocation = 0.8  # Default based on the paper
        relative_allocation = 0.2
    
    return absolute_allocation, relative_allocation

# Backtest strategy
def backtest_strategy(prices, short_window, long_window, strategy_type, transaction_cost=0, capital=1000000, target_vol=0.1, max_leverage=2.0):
    """
    Backtest a trend following strategy with improved numerical stability
    
    Parameters:
    prices (pd.DataFrame): Price data
    short_window (int): Short moving average window
    long_window (int): Long moving average window
    strategy_type (str): 'absolute', 'relative', 'dynamic', or 'mix'
    transaction_cost (float): Transaction cost in basis points (e.g., 20 = 0.2 basis points = 0.002%)
    capital (float): Initial capital
    target_vol (float): Target annualized volatility
    max_leverage (float): Maximum leverage allowed
    
    Returns:
    pd.DataFrame: Strategy performance metrics
    """
    # Calculate returns
    returns = prices.pct_change().dropna()
    
    # Calculate risk-adjusted returns
    risk_adj_returns = calculate_risk_adjusted_returns(returns)
    
    # Calculate trend signals
    signals = pd.DataFrame(index=prices.index, columns=prices.columns)
    for col in prices.columns:
        signals[col] = calculate_trend_signal(prices[col], short_window, long_window)
    
    # Calculate rolling correlation matrix
    correlation_matrices = {}
    vol_window = 60  # Use 60-day window for correlation estimation
    
    for date in returns.index[vol_window:]:
        # Get the window of returns
        window_returns = returns.loc[:date].tail(vol_window)
        
        # Calculate correlation matrix
        correlation_matrices[date] = window_returns.corr().values
    
    # Calculate rolling volatility
    volatilities = returns.rolling(window=vol_window, min_periods=30).std() * np.sqrt(252)  # Annualize
    
    # Initialize portfolios (use float64 for better precision)
    portfolio_values = pd.Series(index=returns.index, data=float(capital), dtype=np.float64)
    exposures = pd.DataFrame(index=returns.index, columns=prices.columns, data=0.0, dtype=np.float64)
    absolute_allocation = pd.Series(index=returns.index, data=np.nan, dtype=np.float64)
    relative_allocation = pd.Series(index=returns.index, data=np.nan, dtype=np.float64)
    
    # Asset class weights (as mentioned in the paper)
    asset_class_weights = {
        'Bonds': 0.3,
        'STIR': 0.0,
        'Equity': 0.3,
        'Commodities': 0.3,
        'FX': 0.1
    }
    
    # Create instrument weights
    instrument_weights = {}
    tickers_dict, _ = get_tickers()
    for asset_class, weight in asset_class_weights.items():
        if asset_class in tickers_dict:
            tickers_in_class = [ticker for ticker in tickers_dict[asset_class] if ticker in prices.columns]
            if tickers_in_class:
                for ticker in tickers_in_class:
                    instrument_weights[ticker] = weight / len(tickers_in_class)
    
    # Convert to numpy array in the same order as prices.columns
    weights = np.array([instrument_weights.get(col, 0) for col in prices.columns], dtype=np.float64)
    weights = weights / weights.sum()  # Normalize
    
    # Start from vol_window to have enough data for correlation estimation
    start_idx = max(vol_window, long_window)
    
    # Loop through time
    for t in range(start_idx, len(returns)):
        date = returns.index[t]
        prev_date = returns.index[t-1]
        
        # Get signals, volatilities, and correlation matrix
        current_signals = signals.iloc[t].values
        current_vols = volatilities.iloc[t-1].values  # Use previous volatility estimate
        
        # Handle missing values
        current_vols = np.where(np.isnan(current_vols), np.nanmean(current_vols), current_vols)
        current_vols = np.where(current_vols <= 0.001, 0.01, current_vols)  # Ensure positive volatility with reasonable minimum
        
        # Clip signals to prevent extreme values
        current_signals = np.clip(current_signals, -5, 5)
        
        # Get correlation matrix (if available, otherwise use identity)
        if date in correlation_matrices:
            current_corr = correlation_matrices[date]
        else:
            current_corr = np.eye(len(current_signals))
        
        # Ensure correlation matrix is valid
        current_corr = np.clip(current_corr, -0.999, 0.999)
        
        # Make sure the portfolio value is valid
        if not np.isfinite(portfolio_values[prev_date]) or portfolio_values[prev_date] <= 0:
            logger.warning(f"Invalid portfolio value at {prev_date}: {portfolio_values[prev_date]}. Resetting to initial capital.")
            portfolio_values[prev_date] = capital
        
        # Calculate exposures based on strategy type
        try:
            if strategy_type == 'absolute':
                current_exposures = absolute_trend_strategy(current_signals, current_vols, 
                                                          portfolio_values[prev_date], target_vol, weights)
            elif strategy_type == 'relative':
                current_exposures = relative_trend_strategy(current_signals, current_corr, current_vols, 
                                                          portfolio_values[prev_date], target_vol, weights)
            elif strategy_type == 'dynamic':
                current_exposures = dynamic_trend_strategy(current_signals, current_corr, current_vols, 
                                                         portfolio_values[prev_date], target_vol, weights)
                
                # Calculate allocation to absolute and relative trends
                abs_alloc, rel_alloc = calculate_allocation(current_signals, current_corr, weights)
                absolute_allocation[date] = abs_alloc
                relative_allocation[date] = rel_alloc
            elif strategy_type == 'mix':
                # 50/50 mix of absolute and relative strategies
                abs_exposures = absolute_trend_strategy(current_signals, current_vols, 
                                                     portfolio_values[prev_date] * 0.5, target_vol, weights)
                rel_exposures = relative_trend_strategy(current_signals, current_corr, current_vols, 
                                                     portfolio_values[prev_date] * 0.5, target_vol, weights)
                current_exposures = abs_exposures + rel_exposures
            else:
                raise ValueError(f"Unknown strategy type: {strategy_type}")
        except Exception as e:
            logger.error(f"Error calculating exposures at {date}: {e}")
            if t > start_idx:
                current_exposures = exposures.iloc[t-1].values
            else:
                current_exposures = np.zeros(len(prices.columns))
        
        # Limit leverage to prevent extreme values
        total_exposure = np.sum(np.abs(current_exposures))
        if total_exposure > portfolio_values[prev_date] * max_leverage:
            scaling_factor = portfolio_values[prev_date] * max_leverage / total_exposure
            current_exposures = current_exposures * scaling_factor
            logger.warning(f"Reducing leverage at {date} from {total_exposure/portfolio_values[prev_date]:.2f}x to {max_leverage:.2f}x")
        
        # Handle numerical stability - clip exposures to reasonable values
        current_exposures = np.clip(current_exposures, -portfolio_values[prev_date] * max_leverage, portfolio_values[prev_date] * max_leverage)
        
        # Store exposures
        exposures.iloc[t] = current_exposures
        
        # Calculate transaction costs (only for t > start_idx)
        if t > start_idx:
            prev_exposures = exposures.iloc[t-1].values
            exposure_changes = np.abs(current_exposures - prev_exposures)
            
            # Calculate transaction costs in basis points 
            # Convert transaction_cost from basis points to percentage (1bp = 0.01%)
            tx_cost_pct = transaction_cost / 10000  # Convert from basis points to decimal
            tx_costs = np.sum(exposure_changes * tx_cost_pct) / portfolio_values[prev_date]
        else:
            tx_costs = 0
        
        # Calculate portfolio return
        returns_slice = returns.iloc[t].values
        
        # Replace any extreme returns or NaN values - compatible with older NumPy versions
        returns_slice = np.nan_to_num(returns_slice)
        returns_slice[~np.isfinite(returns_slice)] = 0.0  # Handle inf and -inf
        returns_slice = np.clip(returns_slice, -0.5, 0.5)  # Limit to +/- 50% returns
        
        # Calculate portfolio return with safety checks
        try:
            portfolio_return = np.sum(current_exposures * returns_slice) / portfolio_values[prev_date] - tx_costs
            
            # Limit maximum daily loss/gain to prevent extreme values
            portfolio_return = np.clip(portfolio_return, -0.5, 0.5)
            
            # Update portfolio value
            new_value = portfolio_values[prev_date] * (1 + portfolio_return)
            
            # Ensure the new value is finite and positive
            if np.isfinite(new_value) and new_value > 0:
                portfolio_values[date] = new_value
            else:
                logger.warning(f"Invalid portfolio update at {date}: {new_value}. Using previous value.")
                portfolio_values[date] = portfolio_values[prev_date]
        except Exception as e:
            logger.error(f"Error calculating portfolio return at {date}: {e}")
            portfolio_values[date] = portfolio_values[prev_date]
    
    # Calculate strategy returns
    strategy_returns = portfolio_values.pct_change().dropna()
    
    # Remove any extreme or invalid returns - compatible with older NumPy versions
    strategy_returns = strategy_returns.replace([np.inf, -np.inf], np.nan)
    strategy_returns = strategy_returns.dropna()
    strategy_returns = strategy_returns.clip(-0.5, 0.5)  # Limit to +/- 50% daily returns
    
    # Calculate performance metrics
    cumulative_returns = (1 + strategy_returns).cumprod()
    drawdowns = cumulative_returns / cumulative_returns.cummax() - 1
    
    # Calculate annual statistics
    annual_stats = {}
    years = sorted(set(strategy_returns.index.year))
    for year in years:
        year_returns = strategy_returns[strategy_returns.index.year == year]
        if not year_returns.empty:
            annual_return = (1 + year_returns).prod() - 1
            annual_vol = year_returns.std() * np.sqrt(252)
            annual_sharpe = annual_return / annual_vol if annual_vol > 0 else 0
            
            year_cum_returns = (1 + year_returns).cumprod()
            year_drawdown = (year_cum_returns / year_cum_returns.cummax() - 1).min()
            
            annual_stats[year] = {
                'return': annual_return,
                'volatility': annual_vol,
                'sharpe': annual_sharpe,
                'max_drawdown': year_drawdown
            }
    
    # Calculate overall statistics
    overall_return = (1 + strategy_returns).prod() - 1
    overall_vol = strategy_returns.std() * np.sqrt(252)
    overall_sharpe = overall_return / overall_vol if overall_vol > 0 else 0
    max_drawdown = drawdowns.min()
    
    # Calculate skewness
    daily_skew = strategy_returns.skew()
    
    # Group returns by month
    monthly_returns = strategy_returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
    monthly_skew = monthly_returns.skew()
    
    result = {
        'returns': strategy_returns,
        'cumulative_returns': cumulative_returns,
        'drawdowns': drawdowns,
        'annual_stats': annual_stats,
        'overall_return': overall_return,
        'overall_volatility': overall_vol,
        'overall_sharpe': overall_sharpe,
        'max_drawdown': max_drawdown,
        'max_drawdown_to_vol': max_drawdown / overall_vol,
        'daily_skewness': daily_skew,
        'monthly_skewness': monthly_skew,
        'absolute_allocation': absolute_allocation,
        'relative_allocation': relative_allocation
    }
    
    return result

# Compare strategies
def compare_strategies(prices, short_window, long_window, transaction_cost=0, capital=1000000, target_vol=0.1):
    """
    Compare different trend following strategies
    
    Parameters:
    prices (pd.DataFrame): Price data
    short_window (int): Short moving average window
    long_window (int): Long moving average window
    transaction_cost (float): Transaction cost in basis points
    capital (float): Initial capital
    target_vol (float): Target annualized volatility
    
    Returns:
    dict: Results for each strategy
    """
    strategies = ['absolute', 'relative', 'dynamic', 'mix']
    results = {}
    
    for strategy in strategies:
        print(f"Backtesting {strategy} strategy...")
        results[strategy] = backtest_strategy(prices, short_window, long_window, strategy, 
                                               transaction_cost, capital, target_vol)
    
    # Calculate strategy correlations
    returns_df = pd.DataFrame({s: results[s]['returns'] for s in strategies})
    correlations = returns_df.corr()
    
    # Print performance summary
    print("\nStrategy Performance Summary:")
    print("-----------------------------")
    print(f"Parameters: S={short_window}, L={long_window}, Transaction Cost=${transaction_cost}")
    print("\nOverall Sharpe Ratios:")
    for strategy in strategies:
        t_stat = results[strategy]['overall_sharpe'] * np.sqrt(len(results[strategy]['returns']))
        print(f"{strategy.capitalize()}: {results[strategy]['overall_sharpe']:.2f} (t-stat: {t_stat:.2f})")
    
    print("\nMaximum Drawdown to Volatility Ratios:")
    for strategy in strategies:
        print(f"{strategy.capitalize()}: {results[strategy]['max_drawdown_to_vol']:.2f}")
    
    print("\nSkewness (Daily/Monthly):")
    for strategy in strategies:
        print(f"{strategy.capitalize()}: {results[strategy]['daily_skewness']:.2f} / {results[strategy]['monthly_skewness']:.2f}")
    
    print("\nStrategy Correlations:")
    print(correlations)
    
    # Create output directory if it doesn't exist
    output_dir = os.path.join(os.getcwd(), 'strategy_output')
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot cumulative returns
    plt.figure(figsize=(12, 6))
    for strategy in strategies:
        plt.plot(results[strategy]['cumulative_returns'], label=strategy.capitalize())
    plt.title('Cumulative Returns')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'cumulative_returns_S{short_window}_L{long_window}_TC{transaction_cost}.png'))
    
    # Plot drawdowns
    plt.figure(figsize=(12, 6))
    for strategy in strategies:
        plt.plot(results[strategy]['drawdowns'], label=strategy.capitalize())
    plt.title('Drawdowns')
    plt.xlabel('Date')
    plt.ylabel('Drawdown')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'drawdowns_S{short_window}_L{long_window}_TC{transaction_cost}.png'))
    
    # Plot allocation to absolute and relative trends (only for dynamic strategy)
    if 'dynamic' in results:
        plt.figure(figsize=(12, 6))
        plt.plot(results['dynamic']['absolute_allocation'], label='Absolute Trend')
        plt.plot(results['dynamic']['relative_allocation'], label='Relative Trend')
        plt.title('Dynamic Strategy: Allocation to Absolute and Relative Trends')
        plt.xlabel('Date')
        plt.ylabel('Allocation')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'allocation_S{short_window}_L{long_window}_TC{transaction_cost}.png'))
    
    # Save performance metrics to CSV
    metrics = []
    for strategy in strategies:
        strategy_metrics = {
            'Strategy': strategy.capitalize(),
            'Sharpe Ratio': results[strategy]['overall_sharpe'],
            'Max Drawdown to Vol': results[strategy]['max_drawdown_to_vol'],
            'Daily Skewness': results[strategy]['daily_skewness'],
            'Monthly Skewness': results[strategy]['monthly_skewness'],
            'Total Return': results[strategy]['overall_return'],
            'Volatility': results[strategy]['overall_volatility'],
            'Max Drawdown': results[strategy]['max_drawdown']
        }
        metrics.append(strategy_metrics)
    
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(os.path.join(output_dir, f'metrics_S{short_window}_L{long_window}_TC{transaction_cost}.csv'), index=False)
    
    return results

def main():
    # Parameters
    start_date = '2000-01-01'
    end_date = '2018-12-31'
    
    # Define the "magic pairs" as mentioned in the paper
    signal_parameters = [
        (8, 24),    # S=8, L=24
        (16, 48),   # S=16, L=48
        (32, 96)    # S=32, L=96
    ]
    
    # Transaction costs in basis points - the paper likely measures in basis points
    # 0 = no transaction costs
    # 20 = 0.2 basis points = 0.002% per dollar traded
    transaction_costs = [0, 20]  # basis points
    
    # Get tickers
    _, all_tickers = get_tickers()
    
    try:
        # Initialize the Bloomberg data fetcher
        fetcher = BloombergDataFetcher()
        
        # Try to connect to Bloomberg
        use_synthetic_data = True
        if fetcher.start_session():
            logger.info("Successfully connected to Bloomberg API")
            
            try:
                # Fetch historical data
                logger.info(f"Fetching data for {len(all_tickers)} instruments...")
                
                # Fetch data in batches to avoid request size issues
                batch_size = 20
                all_data = {}
                
                for i in range(0, len(all_tickers), batch_size):
                    batch_tickers = all_tickers[i:i+batch_size]
                    logger.info(f"Fetching batch {i//batch_size + 1} of {(len(all_tickers) + batch_size - 1)//batch_size}")
                    
                    batch_data = fetcher.get_historical_data(
                        batch_tickers,
                        ["PX_LAST"],
                        start_date,
                        end_date,
                        "DAILY"
                    )
                    
                    # Add batch data to all data
                    all_data.update(batch_data)
                
                # Check if we got any data
                if all_data:
                    logger.info(f"Successfully fetched data for {len(all_data)} out of {len(all_tickers)} instruments")
                    
                    # Convert to DataFrame
                    price_data = pd.DataFrame({ticker: data["PX_LAST"] for ticker, data in all_data.items()})
                    
                    # Drop columns with too many missing values
                    missing_pct = price_data.isnull().mean()
                    valid_columns = missing_pct[missing_pct < 0.3].index
                    price_data = price_data[valid_columns]
                    
                    # Forward fill missing values
                    price_data = price_data.ffill()
                    
                    logger.info(f"Analysis will be performed on {len(valid_columns)} instruments")
                    use_synthetic_data = False
                else:
                    logger.warning("No data fetched from Bloomberg. Using synthetic data instead.")
            except Exception as e:
                logger.error(f"Error fetching data from Bloomberg: {e}")
                logger.info("Using synthetic data instead")
            
            # Stop the Bloomberg session
            fetcher.stop_session()
        else:
            logger.warning("Failed to connect to Bloomberg. Using synthetic data instead.")
    except Exception as e:
        logger.error(f"Error initializing Bloomberg fetcher: {e}")
        logger.info("Using synthetic data instead")
    
    # If we couldn't get Bloomberg data, use synthetic data
    if use_synthetic_data:
        logger.info("Generating synthetic data for testing the strategies")
        price_data = generate_synthetic_data(all_tickers, start_date, end_date)
    
    # Test all parameter combinations
    for short_window, long_window in signal_parameters:
        for tx_cost in transaction_costs:
            logger.info(f"\n=== Testing with S={short_window}, L={long_window}, Transaction Cost=${tx_cost} ===")
            results = compare_strategies(price_data, short_window, long_window, 
                                         transaction_cost=tx_cost)
    
    # Final message about output location
    output_dir = os.path.join(os.getcwd(), 'strategy_output')
    logger.info(f"\nOutput charts and performance metrics saved to: {output_dir}")
    plt.show()

if __name__ == "__main__":
    main()