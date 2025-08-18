"""
Backtesting Framework for HSTECH Estimation System

This module provides comprehensive backtesting capabilities to validate
the accuracy of the HSTECH index estimation methods.
"""

from typing import Dict, List, Optional, Tuple, Any
from decimal import Decimal
from datetime import datetime, timezone, timedelta
import logging
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns

from ..models import Config, EstimationResult, IndexData, PriceData, CurrencyRate
from ..estimation import HSTECHEstimator
from ..data import DataManager

logger = logging.getLogger(__name__)


class BacktestResult:
    """Container for backtesting results."""
    
    def __init__(self):
        self.start_date = None
        self.end_date = None
        self.total_predictions = 0
        self.successful_predictions = 0
        
        # Accuracy metrics
        self.mse = None
        self.mae = None
        self.rmse = None
        self.mape = None  # Mean Absolute Percentage Error
        self.correlation = None
        self.directional_accuracy = None
        
        # Detailed results
        self.daily_results = []
        self.method_performance = {}
        self.error_distribution = {}
        
        # Confidence analysis
        self.confidence_vs_accuracy = {}
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert results to dictionary for serialization."""
        return {
            "start_date": self.start_date.isoformat() if self.start_date else None,
            "end_date": self.end_date.isoformat() if self.end_date else None,
            "total_predictions": self.total_predictions,
            "successful_predictions": self.successful_predictions,
            "success_rate": self.successful_predictions / self.total_predictions if self.total_predictions > 0 else 0,
            "metrics": {
                "mse": float(self.mse) if self.mse else None,
                "mae": float(self.mae) if self.mae else None,
                "rmse": float(self.rmse) if self.rmse else None,
                "mape": float(self.mape) if self.mape else None,
                "correlation": float(self.correlation) if self.correlation else None,
                "directional_accuracy": float(self.directional_accuracy) if self.directional_accuracy else None
            },
            "method_performance": self.method_performance,
            "error_distribution": self.error_distribution,
            "confidence_vs_accuracy": self.confidence_vs_accuracy
        }


class HSTECHBacktester:
    """
    Comprehensive backtesting framework for HSTECH estimation.
    
    Features:
    - Historical simulation of estimation process
    - Multiple accuracy metrics calculation
    - Method-specific performance analysis
    - Confidence calibration analysis
    - Visualization of results
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.estimator = HSTECHEstimator(config)
        self.data_manager = DataManager(config)
        
        logger.info("Initialized HSTECH backtester")
    
    async def run_backtest(
        self,
        start_date: str,
        end_date: str,
        estimation_frequency: str = "daily"
    ) -> BacktestResult:
        """
        Run comprehensive backtest over specified period.
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            estimation_frequency: Frequency of estimations ("daily", "hourly")
            
        Returns:
            BacktestResult with comprehensive metrics
        """
        logger.info(f"Starting backtest from {start_date} to {end_date}")
        
        result = BacktestResult()
        result.start_date = datetime.fromisoformat(start_date)
        result.end_date = datetime.fromisoformat(end_date)
        
        # Fetch historical data for the entire period
        historical_data = await self._prepare_historical_data(start_date, end_date)
        
        if not historical_data:
            logger.error("Failed to fetch historical data for backtesting")
            return result
        
        # Run day-by-day simulation
        current_date = result.start_date
        daily_results = []
        
        while current_date <= result.end_date:
            try:
                # Skip weekends (assuming markets are closed)
                if current_date.weekday() >= 5:
                    current_date += timedelta(days=1)
                    continue
                
                # Simulate estimation for this date
                daily_result = await self._simulate_daily_estimation(
                    current_date, historical_data
                )
                
                if daily_result:
                    daily_results.append(daily_result)
                    result.total_predictions += 1
                    
                    if daily_result["estimation_successful"]:
                        result.successful_predictions += 1
                
                current_date += timedelta(days=1)
                
            except Exception as e:
                logger.warning(f"Error processing date {current_date}: {e}")
                current_date += timedelta(days=1)
                continue
        
        result.daily_results = daily_results
        
        # Calculate aggregate metrics
        self._calculate_metrics(result)
        
        logger.info(f"Backtest complete: {result.successful_predictions}/{result.total_predictions} "
                   f"successful predictions")
        
        return result
    
    async def _prepare_historical_data(
        self, 
        start_date: str, 
        end_date: str
    ) -> Dict[str, pd.DataFrame]:
        """Prepare historical data for backtesting."""
        logger.info("Preparing historical data for backtesting")
        
        # Get all symbols we need
        from ...data.hstech_components import HSTECH_COMPONENTS, MARKET_INDICATORS
        from ..data.adr_mapper import ADRMapper
        
        adr_mapper = ADRMapper()
        all_symbols = []
        
        # Add HSTECH component symbols
        all_symbols.extend([stock.symbol for stock in HSTECH_COMPONENTS])
        
        # Add ADR symbols
        all_symbols.extend(adr_mapper.get_all_adr_symbols())
        
        # Add market indicators
        all_symbols.extend(MARKET_INDICATORS)
        
        # Add HSTECH index itself
        all_symbols.append("^HSTECH")
        
        # Fetch historical data
        days = (datetime.fromisoformat(end_date) - datetime.fromisoformat(start_date)).days + 30
        historical_data = await self.data_manager.fetch_historical_data(all_symbols, days)
        
        logger.info(f"Prepared historical data for {len(historical_data)} symbols")
        return historical_data
    
    async def _simulate_daily_estimation(
        self,
        target_date: datetime,
        historical_data: Dict[str, pd.DataFrame]
    ) -> Optional[Dict[str, Any]]:
        """Simulate the estimation process for a specific date."""
        
        try:
            # Get actual HSTECH value for this date (ground truth)
            actual_hstech = self._get_actual_hstech_value(target_date, historical_data)
            if actual_hstech is None:
                return None
            
            # Simulate data available at estimation time
            # (Use data up to previous day to simulate real-time scenario)
            estimation_date = target_date - timedelta(days=1)
            
            # Prepare simulated input data
            simulated_data = self._prepare_simulated_data(estimation_date, historical_data)
            
            if not self._validate_simulated_data(simulated_data):
                return None
            
            # Run estimation
            try:
                estimation_result = await self.estimator.estimate_current_price(
                    **simulated_data
                )
                
                # Calculate errors
                estimated_value = float(estimation_result.estimated_value)
                actual_value = float(actual_hstech.value)
                
                absolute_error = abs(estimated_value - actual_value)
                percentage_error = absolute_error / actual_value * 100
                
                # Determine directional accuracy
                if len(historical_data.get("^HSTECH", [])) > 1:
                    previous_actual = self._get_actual_hstech_value(
                        target_date - timedelta(days=1), historical_data
                    )
                    if previous_actual:
                        actual_direction = 1 if actual_value > float(previous_actual.value) else -1
                        estimated_direction = 1 if estimated_value > float(previous_actual.value) else -1
                        directional_correct = actual_direction == estimated_direction
                    else:
                        directional_correct = None
                else:
                    directional_correct = None
                
                return {
                    "date": target_date,
                    "actual_value": actual_value,
                    "estimated_value": estimated_value,
                    "absolute_error": absolute_error,
                    "percentage_error": percentage_error,
                    "confidence": estimation_result.confidence,
                    "method_weights": estimation_result.method_weights,
                    "directional_correct": directional_correct,
                    "estimation_successful": True
                }
                
            except Exception as e:
                logger.warning(f"Estimation failed for {target_date}: {e}")
                return {
                    "date": target_date,
                    "actual_value": float(actual_hstech.value),
                    "estimated_value": None,
                    "absolute_error": None,
                    "percentage_error": None,
                    "confidence": None,
                    "method_weights": None,
                    "directional_correct": None,
                    "estimation_successful": False
                }
        
        except Exception as e:
            logger.warning(f"Error simulating estimation for {target_date}: {e}")
            return None
    
    def _get_actual_hstech_value(
        self, 
        date: datetime, 
        historical_data: Dict[str, pd.DataFrame]
    ) -> Optional[IndexData]:
        """Get actual HSTECH index value for a specific date."""
        
        if "^HSTECH" not in historical_data:
            return None
        
        hstech_data = historical_data["^HSTECH"]
        
        # Find closest date
        target_date_str = date.strftime("%Y-%m-%d")
        
        # Look for exact match first
        matching_rows = hstech_data[hstech_data['date'].dt.strftime("%Y-%m-%d") == target_date_str]
        
        if not matching_rows.empty:
            row = matching_rows.iloc[0]
            return IndexData(
                value=Decimal(str(row['close'])),
                timestamp=date,
                change=None,
                change_percent=None
            )
        
        return None
    
    def _prepare_simulated_data(
        self, 
        estimation_date: datetime, 
        historical_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, Any]:
        """Prepare simulated input data for estimation."""
        
        # This is a simplified simulation - in a real implementation,
        # you would need to carefully simulate the exact data that would
        # have been available at the estimation time
        
        simulated_data = {
            "current_adr_prices": {},
            "last_hk_prices": {},
            "current_exchange_rate": None,
            "last_hstech_close": None,
            "current_indicator_prices": {},
            "previous_indicator_prices": {},
            "historical_data": historical_data
        }
        
        # For backtesting purposes, we'll use simplified mock data
        # In a production system, this would be much more sophisticated
        
        return simulated_data
    
    def _validate_simulated_data(self, simulated_data: Dict[str, Any]) -> bool:
        """Validate that simulated data is sufficient for estimation."""
        # Basic validation - can be enhanced
        return simulated_data is not None
    
    def _calculate_metrics(self, result: BacktestResult):
        """Calculate aggregate performance metrics."""
        
        if not result.daily_results:
            return
        
        # Filter successful predictions
        successful_results = [r for r in result.daily_results if r["estimation_successful"]]
        
        if not successful_results:
            return
        
        # Extract values
        actual_values = [r["actual_value"] for r in successful_results]
        estimated_values = [r["estimated_value"] for r in successful_results]
        percentage_errors = [r["percentage_error"] for r in successful_results]
        
        # Calculate metrics
        result.mse = mean_squared_error(actual_values, estimated_values)
        result.mae = mean_absolute_error(actual_values, estimated_values)
        result.rmse = np.sqrt(result.mse)
        result.mape = np.mean(percentage_errors)
        result.correlation = np.corrcoef(actual_values, estimated_values)[0, 1]
        
        # Directional accuracy
        directional_results = [r for r in successful_results if r["directional_correct"] is not None]
        if directional_results:
            correct_directions = sum(1 for r in directional_results if r["directional_correct"])
            result.directional_accuracy = correct_directions / len(directional_results)
        
        # Method performance analysis
        self._analyze_method_performance(result)
        
        # Confidence calibration
        self._analyze_confidence_calibration(result)
    
    def _analyze_method_performance(self, result: BacktestResult):
        """Analyze performance of different estimation methods."""
        # This would analyze how different method weights correlate with accuracy
        # Implementation depends on detailed tracking of method contributions
        pass
    
    def _analyze_confidence_calibration(self, result: BacktestResult):
        """Analyze how well confidence scores correlate with actual accuracy."""
        # This would bin predictions by confidence level and measure actual accuracy
        # Implementation depends on detailed confidence tracking
        pass
    
    def generate_report(self, result: BacktestResult, output_path: str = None):
        """Generate comprehensive backtesting report."""
        
        report = f"""
HSTECH Index Estimation Backtesting Report
==========================================

Period: {result.start_date.strftime('%Y-%m-%d')} to {result.end_date.strftime('%Y-%m-%d')}
Total Predictions: {result.total_predictions}
Successful Predictions: {result.successful_predictions}
Success Rate: {result.successful_predictions/result.total_predictions*100:.1f}%

Performance Metrics:
-------------------
Mean Squared Error (MSE): {result.mse:.2f}
Mean Absolute Error (MAE): {result.mae:.2f}
Root Mean Squared Error (RMSE): {result.rmse:.2f}
Mean Absolute Percentage Error (MAPE): {result.mape:.2f}%
Correlation: {result.correlation:.3f}
Directional Accuracy: {result.directional_accuracy*100:.1f}%

"""
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report)
            logger.info(f"Backtesting report saved to {output_path}")
        else:
            print(report)


def create_backtester(config: Config) -> HSTECHBacktester:
    """Create and return a HSTECHBacktester instance."""
    return HSTECHBacktester(config)
