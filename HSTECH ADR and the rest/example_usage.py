#!/usr/bin/env python3
"""
HSTECH Index Estimation System - Example Usage

This script demonstrates how to use the HSTECH estimation system to estimate
the index price using US market data when Hong Kong markets are closed.
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime, timezone

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.models import Config
from src.estimation import HSTECHEstimator
from src.data import DataManager
from src.utils import setup_logging, MarketHoursChecker
from src.backtesting import create_performance_summary, print_performance_summary


async def main():
    """Main example function."""
    
    # Load configuration
    try:
        config = Config.from_yaml("config/config.yaml")
        print("✓ Loaded configuration from config/config.yaml")
    except FileNotFoundError:
        print("⚠ Configuration file not found, using defaults")
        config = Config()
    
    # Setup logging
    logger = setup_logging(config.logging)
    logger.info("Starting HSTECH estimation example")
    
    print("\n" + "="*60)
    print("HSTECH INDEX ESTIMATION SYSTEM - EXAMPLE USAGE")
    print("="*60)
    
    # Initialize components
    print("\n1. Initializing system components...")
    
    estimator = HSTECHEstimator(config)
    data_manager = DataManager(config)
    market_checker = MarketHoursChecker(config.market_hours)
    
    print("✓ Estimator initialized")
    print("✓ Data manager initialized")
    print("✓ Market hours checker initialized")
    
    # Check market status
    print("\n2. Checking market status...")
    market_status = market_checker.get_market_status_summary()
    
    print(f"   Hong Kong market open: {market_status['hong_kong_market_open']}")
    print(f"   US market open: {market_status['us_market_open']}")
    print(f"   Should run estimation: {market_status['should_run_estimation']}")
    print(f"   Reason: {market_status['estimation_reason']}")
    
    # Fetch current data
    print("\n3. Fetching market data...")
    
    try:
        data_success = await data_manager.fetch_all_current_data()
        
        if data_success:
            print("✓ Successfully fetched market data")
            
            # Get data quality report
            quality_report = data_manager.get_data_quality_report()
            print(f"   Data quality: {quality_report['overall_quality']}")
            print(f"   ADR coverage: {quality_report['adr_coverage']['coverage_percent']:.1f}%")
            print(f"   HK coverage: {quality_report['hk_coverage']['coverage_percent']:.1f}%")
            print(f"   Indicator coverage: {quality_report['indicator_coverage']['coverage_percent']:.1f}%")
            
        else:
            print("⚠ Failed to fetch some market data")
            
    except Exception as e:
        print(f"✗ Error fetching data: {e}")
        return
    
    # Run estimation if appropriate
    if market_status['should_run_estimation'] and data_success:
        print("\n4. Running HSTECH estimation...")
        
        try:
            # Get estimation data
            estimation_data = await data_manager.get_estimation_data()
            
            # Fetch historical data for covariance modeling
            print("   Fetching historical data for covariance modeling...")
            from data.hstech_components import HSTECH_COMPONENTS
            symbols = [stock.symbol for stock in HSTECH_COMPONENTS[:10]]  # Limit for demo
            historical_data = await data_manager.fetch_historical_data(symbols, days=60)
            
            # Run estimation
            result = await estimator.estimate_current_price(
                current_adr_prices=estimation_data[0],
                last_hk_prices=estimation_data[1],
                current_exchange_rate=estimation_data[2],
                last_hstech_close=estimation_data[3],
                current_indicator_prices=estimation_data[4],
                previous_indicator_prices=estimation_data[5],
                historical_data=historical_data
            )
            
            print("✓ Estimation completed successfully!")
            print(f"\n   ESTIMATED HSTECH INDEX: {result.estimated_value:.2f}")
            print(f"   Confidence: {result.confidence:.1%}")
            print(f"   Timestamp: {result.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}")
            
            print(f"\n   Method Weights:")
            for method, weight in result.method_weights.items():
                print(f"     {method}: {weight:.1%}")
            
            print(f"\n   Top Component Contributions:")
            sorted_contributions = sorted(
                result.component_contributions.items(),
                key=lambda x: abs(x[1]),
                reverse=True
            )
            for symbol, contribution in sorted_contributions[:5]:
                print(f"     {symbol}: {contribution:+.4f}")
            
        except Exception as e:
            print(f"✗ Estimation failed: {e}")
            logger.error(f"Estimation error: {e}", exc_info=True)
    
    else:
        print("\n4. Skipping estimation (not appropriate time or insufficient data)")
    
    # Show estimator capabilities
    print("\n5. System capabilities summary...")
    summary = estimator.get_estimation_summary()
    
    print(f"   Total HSTECH components: {summary['total_hstech_components']}")
    print(f"   ADR-mapped components: {summary['adr_coverage']['adr_component_count']}")
    print(f"   ADR weight coverage: {summary['adr_coverage']['adr_weight_coverage']:.1%}")
    print(f"   Market indicators: {len(summary['market_indicators'])}")
    print(f"   Covariance lookback: {summary['covariance_lookback_days']} days")
    
    # Demonstrate backtesting (simplified)
    print("\n6. Backtesting demonstration...")
    print("   (This would normally run a full historical backtest)")
    print("   For demo purposes, showing metrics calculation example:")
    
    # Example metrics calculation
    actual_values = [10000, 10100, 9950, 10200, 10150]
    predicted_values = [10050, 10080, 9980, 10180, 10120]
    
    demo_summary = create_performance_summary(actual_values, predicted_values)
    print_performance_summary(demo_summary)
    
    print("\n" + "="*60)
    print("EXAMPLE COMPLETED SUCCESSFULLY")
    print("="*60)
    
    print("\nNext steps:")
    print("1. Configure your API keys in config/config.yaml")
    print("2. Adjust estimation weights based on your requirements")
    print("3. Run backtesting to validate performance")
    print("4. Set up automated scheduling for regular estimations")
    print("5. Integrate with your trading or monitoring systems")


def run_simple_estimation():
    """Simple synchronous example for quick testing."""
    
    print("HSTECH Estimation System - Simple Example")
    print("-" * 40)
    
    # Create estimator with default config
    config = Config()
    estimator = HSTECHEstimator(config)
    
    print(f"✓ Estimator created with {len(estimator.adr_estimator.adr_components)} ADR components")
    
    # Show ADR coverage
    coverage_stats = estimator.adr_estimator.get_adr_coverage_stats()
    print(f"✓ ADR coverage: {coverage_stats['adr_weight_coverage']:.1%} of index weight")
    
    # Show market indicators
    indicators = estimator.enhanced_estimator.indicator_symbols
    print(f"✓ Market indicators: {', '.join(indicators)}")
    
    print("\nTo run full estimation, use: python example_usage.py")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--simple":
        run_simple_estimation()
    else:
        try:
            asyncio.run(main())
        except KeyboardInterrupt:
            print("\n\nExample interrupted by user")
        except Exception as e:
            print(f"\nExample failed with error: {e}")
            sys.exit(1)
