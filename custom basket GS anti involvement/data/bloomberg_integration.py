"""
Bloomberg Integration Module for China Anti-Involution Basket

This module provides comprehensive Bloomberg Terminal integration for real-time
data retrieval, historical analysis, and basket validation.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union
import logging
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import blpapi
    BLOOMBERG_AVAILABLE = True
except ImportError:
    BLOOMBERG_AVAILABLE = False
    warnings.warn("Bloomberg API not available. Using mock data for demonstration.")

class BloombergDataProvider:
    """Bloomberg Terminal data provider for basket construction and monitoring"""
    
    def __init__(self, host="localhost", port=8194):
        """
        Initialize Bloomberg data provider
        
        Args:
            host: Bloomberg API host (default: localhost)
            port: Bloomberg API port (default: 8194)
        """
        self.host = host
        self.port = port
        self.session = None
        self.ref_data_service = None
        self.hist_data_service = None
        
        # Bloomberg field mappings
        self.field_mappings = self._define_field_mappings()
        
        if BLOOMBERG_AVAILABLE:
            self._initialize_connection()
        else:
            logger.warning("Bloomberg API not available. Using mock data provider.")
    
    def _define_field_mappings(self) -> Dict[str, str]:
        """
        Define Bloomberg field mappings for basket data
        
        Returns:
            Dictionary mapping internal field names to Bloomberg fields
        """
        return {
            # Market data
            'market_cap_usd': 'CUR_MKT_CAP',
            'price': 'PX_LAST',
            'volume_20d': 'VOLUME_AVG_20D',
            'volatility_260d': 'VOLATILITY_260D',
            'beta': 'BETA_RAW_OVERRIDABLE',
            'shares_outstanding': 'EQY_SH_OUT',
            
            # Financial data
            'revenue': 'SALES_REV_TURN',
            'revenue_growth_3y': 'SALES_GROWTH_3YR_AVG',
            'profit_margin': 'PROF_MARGIN',
            'roe': 'RETURN_ON_EQUITY',
            'debt_to_equity': 'BS_DEBT_TO_EQTY',
            'total_assets': 'BS_TOT_ASSET',
            'free_cash_flow': 'CF_FREE_CASH_FLOW',
            
            # Company information
            'company_name': 'NAME',
            'company_name_english': 'NAME_ENGLISH',
            'country': 'CNTRY_OF_DOMICILE',
            'exchange': 'EXCH_CODE',
            'currency': 'CRNCY',
            'sector_gics': 'GICS_SECTOR_NAME',
            'industry_gics': 'GICS_INDUSTRY_NAME',
            
            # Ownership and governance
            'ownership_type': 'OWNERSHIP_TYPE',  # Custom field
            'state_ownership_pct': 'STATE_OWNERSHIP_PCT',  # Custom field
            'institutional_ownership': 'INSTITUTIONAL_OWNERSHIP_PCT',
            
            # ESG and sustainability
            'esg_score': 'ESG_DISCLOSURE_SCORE',
            'carbon_intensity': 'CARBON_INTENSITY_SCOPE_1_2',
            'environmental_score': 'ENVIRONMENTAL_PILLAR_SCORE',
            'social_score': 'SOCIAL_PILLAR_SCORE',
            'governance_score': 'GOVERNANCE_PILLAR_SCORE',
            
            # Analyst data
            'analyst_count': 'NUM_EST_ANALYSTS',
            'consensus_rating': 'BEST_ANALYST_RECS_MEAN',
            'price_target': 'BEST_TARGET_PRICE',
            'eps_estimate': 'BEST_EPS_EST',
            
            # Anti-involution specific (custom calculations)
            'market_share_domestic': 'MARKET_SHARE_DOMESTIC',  # Custom field
            'pricing_power_score': 'PRICING_POWER_SCORE',  # Custom calculation
            'consolidation_benefit_score': 'CONSOLIDATION_BENEFIT_SCORE',  # Custom
            'innovation_score': 'INNOVATION_SCORE',  # Custom calculation
            'government_relationship_score': 'GOVERNMENT_RELATIONSHIP_SCORE'  # Custom
        }
    
    def _initialize_connection(self):
        """Initialize Bloomberg API connection"""
        if not BLOOMBERG_AVAILABLE:
            return
            
        try:
            # Create session options
            session_options = blpapi.SessionOptions()
            session_options.setServerHost(self.host)
            session_options.setServerPort(self.port)
            
            # Create and start session
            self.session = blpapi.Session(session_options)
            
            if not self.session.start():
                raise Exception("Failed to start Bloomberg session")
            
            # Open reference data service
            if not self.session.openService("//blp/refdata"):
                raise Exception("Failed to open Bloomberg reference data service")
            
            self.ref_data_service = self.session.getService("//blp/refdata")
            
            # Open historical data service
            if not self.session.openService("//blp/refdata"):
                self.hist_data_service = self.session.getService("//blp/refdata")
            
            logger.info("Bloomberg connection established successfully")
            
        except Exception as e:
            logger.error(f"Bloomberg connection failed: {e}")
            self.session = None
            raise
    
    def get_reference_data(self, securities: List[str], fields: List[str]) -> pd.DataFrame:
        """
        Get reference data for securities
        
        Args:
            securities: List of Bloomberg tickers
            fields: List of Bloomberg field names
            
        Returns:
            DataFrame with reference data
        """
        if not BLOOMBERG_AVAILABLE or self.session is None:
            return self._get_mock_reference_data(securities, fields)
        
        try:
            # Create request
            request = self.ref_data_service.createRequest("ReferenceDataRequest")
            
            # Add securities
            for security in securities:
                request.getElement("securities").appendValue(security)
            
            # Add fields
            for field in fields:
                request.getElement("fields").appendValue(field)
            
            # Send request
            self.session.sendRequest(request)
            
            # Process response
            data = []
            while True:
                event = self.session.nextEvent(500)
                
                if event.eventType() == blpapi.Event.RESPONSE:
                    for msg in event:
                        security_data = msg.getElement("securityData")
                        
                        for i in range(security_data.numValues()):
                            security = security_data.getValue(i)
                            ticker = security.getElement("security").getValue()
                            
                            field_data = {"ticker": ticker}
                            
                            if security.hasElement("fieldData"):
                                fields_element = security.getElement("fieldData")
                                
                                for field in fields:
                                    if fields_element.hasElement(field):
                                        try:
                                            value = fields_element.getElement(field).getValue()
                                            field_data[field] = value
                                        except:
                                            field_data[field] = None
                                    else:
                                        field_data[field] = None
                            
                            # Handle field errors
                            if security.hasElement("fieldExceptions"):
                                exceptions = security.getElement("fieldExceptions")
                                for j in range(exceptions.numValues()):
                                    exception = exceptions.getValue(j)
                                    field_id = exception.getElement("fieldId").getValue()
                                    error_info = exception.getElement("errorInfo")
                                    logger.warning(f"Field error for {ticker}.{field_id}: {error_info}")
                            
                            data.append(field_data)
                    break
            
            return pd.DataFrame(data)
            
        except Exception as e:
            logger.error(f"Error retrieving reference data: {e}")
            return self._get_mock_reference_data(securities, fields)
    
    def get_historical_data(self, 
                          securities: List[str], 
                          fields: List[str],
                          start_date: datetime,
                          end_date: datetime,
                          frequency: str = "DAILY") -> pd.DataFrame:
        """
        Get historical data for securities
        
        Args:
            securities: List of Bloomberg tickers
            fields: List of Bloomberg field names
            start_date: Start date for historical data
            end_date: End date for historical data
            frequency: Data frequency (DAILY, WEEKLY, MONTHLY)
            
        Returns:
            DataFrame with historical data
        """
        if not BLOOMBERG_AVAILABLE or self.session is None:
            return self._get_mock_historical_data(securities, fields, start_date, end_date)
        
        try:
            # Create request
            request = self.ref_data_service.createRequest("HistoricalDataRequest")
            
            # Add securities
            for security in securities:
                request.getElement("securities").appendValue(security)
            
            # Add fields
            for field in fields:
                request.getElement("fields").appendValue(field)
            
            # Set date range
            request.set("startDate", start_date.strftime("%Y%m%d"))
            request.set("endDate", end_date.strftime("%Y%m%d"))
            request.set("periodicitySelection", frequency)
            
            # Send request
            self.session.sendRequest(request)
            
            # Process response
            data = []
            while True:
                event = self.session.nextEvent(500)
                
                if event.eventType() == blpapi.Event.RESPONSE:
                    for msg in event:
                        security_data = msg.getElement("securityData")
                        ticker = security_data.getElement("security").getValue()
                        
                        field_data_array = security_data.getElement("fieldData")
                        
                        for i in range(field_data_array.numValues()):
                            field_data = field_data_array.getValue(i)
                            date = field_data.getElement("date").getValue()
                            
                            row_data = {"ticker": ticker, "date": date}
                            
                            for field in fields:
                                if field_data.hasElement(field):
                                    try:
                                        value = field_data.getElement(field).getValue()
                                        row_data[field] = value
                                    except:
                                        row_data[field] = None
                                else:
                                    row_data[field] = None
                            
                            data.append(row_data)
                    break
            
            df = pd.DataFrame(data)
            if not df.empty:
                df['date'] = pd.to_datetime(df['date'])
            
            return df
            
        except Exception as e:
            logger.error(f"Error retrieving historical data: {e}")
            return self._get_mock_historical_data(securities, fields, start_date, end_date)
    
    def get_basket_data(self, tickers: List[str]) -> pd.DataFrame:
        """
        Get comprehensive basket data for stock selection and analysis
        
        Args:
            tickers: List of stock tickers
            
        Returns:
            DataFrame with comprehensive stock data
        """
        # Define required fields for basket construction
        required_fields = [
            'CUR_MKT_CAP', 'PX_LAST', 'VOLUME_AVG_20D', 'VOLATILITY_260D',
            'BETA_RAW_OVERRIDABLE', 'SALES_REV_TURN', 'SALES_GROWTH_3YR_AVG',
            'PROF_MARGIN', 'RETURN_ON_EQUITY', 'BS_DEBT_TO_EQTY',
            'NAME', 'NAME_ENGLISH', 'CNTRY_OF_DOMICILE', 'EXCH_CODE',
            'GICS_SECTOR_NAME', 'GICS_INDUSTRY_NAME', 'ESG_DISCLOSURE_SCORE',
            'NUM_EST_ANALYSTS', 'BEST_ANALYST_RECS_MEAN'
        ]
        
        # Get reference data
        data = self.get_reference_data(tickers, required_fields)
        
        # Add custom calculated fields
        if not data.empty:
            data = self._add_custom_fields(data)
        
        return data
    
    def _add_custom_fields(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add custom calculated fields for anti-involution analysis
        
        Args:
            data: DataFrame with Bloomberg data
            
        Returns:
            DataFrame with additional custom fields
        """
        # Market cap category
        data['market_cap_category'] = data['CUR_MKT_CAP'].apply(
            lambda x: 'Large Cap' if x >= 10e9 else 'Mid Cap' if x >= 2e9 else 'Small Cap'
        )
        
        # Liquidity score (based on volume)
        data['liquidity_score'] = np.minimum(100, data['VOLUME_AVG_20D'] / 1e6)
        
        # Financial health score
        data['financial_health_score'] = self._calculate_financial_health_score(data)
        
        # Anti-involution alignment scores (placeholder - would be calculated based on sector analysis)
        data['pricing_power_score'] = np.random.uniform(30, 95, len(data))  # Placeholder
        data['consolidation_benefit_score'] = np.random.uniform(40, 95, len(data))  # Placeholder
        data['innovation_score'] = np.random.uniform(20, 95, len(data))  # Placeholder
        data['government_relationship_score'] = np.random.uniform(50, 95, len(data))  # Placeholder
        data['market_share_domestic'] = np.random.uniform(0.02, 0.6, len(data))  # Placeholder
        
        return data
    
    def _calculate_financial_health_score(self, data: pd.DataFrame) -> pd.Series:
        """Calculate financial health score"""
        # Normalize metrics to 0-100 scale
        profit_score = np.clip((data['PROF_MARGIN'] + 0.1) * 500, 0, 100)
        roe_score = np.clip(data['RETURN_ON_EQUITY'] * 500, 0, 100)
        leverage_score = np.clip(100 - (data['BS_DEBT_TO_EQTY'] * 25), 0, 100)
        
        # Weighted average
        financial_health = (profit_score * 0.4 + roe_score * 0.3 + leverage_score * 0.3)
        
        return financial_health.fillna(50)  # Default score for missing data
    
    def _get_mock_reference_data(self, securities: List[str], fields: List[str]) -> pd.DataFrame:
        """Generate mock reference data for testing"""
        logger.info("Using mock Bloomberg data")
        
        # Load from CSV if available, otherwise generate random data
        try:
            mock_data = pd.read_csv('data/stock_universe.csv')
            # Filter for requested securities
            mock_data = mock_data[mock_data['ticker'].isin(securities)]
            return mock_data
        except:
            # Generate random mock data
            data = []
            for security in securities:
                row = {'ticker': security}
                for field in fields:
                    if 'MKT_CAP' in field:
                        row[field] = np.random.uniform(1e9, 200e9)
                    elif 'PRICE' in field:
                        row[field] = np.random.uniform(10, 500)
                    elif 'VOLUME' in field:
                        row[field] = np.random.uniform(1e6, 200e6)
                    elif 'VOLATILITY' in field:
                        row[field] = np.random.uniform(0.2, 0.8)
                    elif 'BETA' in field:
                        row[field] = np.random.uniform(0.5, 2.0)
                    else:
                        row[field] = np.random.uniform(0, 100)
                data.append(row)
            
            return pd.DataFrame(data)
    
    def _get_mock_historical_data(self, securities: List[str], fields: List[str], 
                                start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Generate mock historical data for testing"""
        logger.info("Using mock historical Bloomberg data")
        
        date_range = pd.date_range(start_date, end_date, freq='D')
        data = []
        
        for security in securities:
            for date in date_range:
                row = {'ticker': security, 'date': date}
                for field in fields:
                    if 'PX_LAST' in field:
                        row[field] = 100 + np.random.normal(0, 5)  # Random walk around 100
                    elif 'VOLUME' in field:
                        row[field] = np.random.uniform(1e6, 10e6)
                    else:
                        row[field] = np.random.uniform(0, 100)
                data.append(row)
        
        return pd.DataFrame(data)
    
    def validate_basket_composition(self, basket_composition: Dict) -> Dict:
        """
        Validate basket composition against Bloomberg data
        
        Args:
            basket_composition: Basket composition dictionary
            
        Returns:
            Validation results
        """
        validation_results = {
            'valid': True,
            'warnings': [],
            'errors': []
        }
        
        # Extract tickers from basket
        tickers = [stock.ticker for stock, weight in basket_composition['stocks']]
        
        # Get current market data
        current_data = self.get_basket_data(tickers)
        
        # Validate liquidity
        for ticker in tickers:
            ticker_data = current_data[current_data['ticker'] == ticker]
            if not ticker_data.empty:
                volume = ticker_data['VOLUME_AVG_20D'].iloc[0]
                if volume < 5e6:  # $5M minimum daily volume
                    validation_results['warnings'].append(
                        f"{ticker}: Low liquidity (${volume/1e6:.1f}M daily volume)"
                    )
        
        # Validate market cap requirements
        for ticker in tickers:
            ticker_data = current_data[current_data['ticker'] == ticker]
            if not ticker_data.empty:
                market_cap = ticker_data['CUR_MKT_CAP'].iloc[0]
                if market_cap < 1e9:  # $1B minimum market cap
                    validation_results['errors'].append(
                        f"{ticker}: Market cap too small (${market_cap/1e9:.1f}B)"
                    )
        
        # Check for data availability
        missing_data_tickers = set(tickers) - set(current_data['ticker'].unique())
        for ticker in missing_data_tickers:
            validation_results['errors'].append(f"{ticker}: No Bloomberg data available")
        
        if validation_results['errors']:
            validation_results['valid'] = False
        
        return validation_results
    
    def disconnect(self):
        """Close Bloomberg connection"""
        if self.session:
            self.session.stop()
            logger.info("Bloomberg connection closed")

# Example usage and testing
if __name__ == "__main__":
    # Initialize Bloomberg provider
    bloomberg = BloombergDataProvider()
    
    # Test with sample tickers
    test_tickers = ["002594 CH Equity", "TSLA US Equity", "700 HK Equity"]
    test_fields = ["CUR_MKT_CAP", "PX_LAST", "NAME"]
    
    # Get reference data
    ref_data = bloomberg.get_reference_data(test_tickers, test_fields)
    print("Reference Data:")
    print(ref_data)
    
    # Get comprehensive basket data
    basket_data = bloomberg.get_basket_data(test_tickers)
    print("\nBasket Data:")
    print(basket_data.head())
    
    # Disconnect
    bloomberg.disconnect()
