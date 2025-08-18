#!/usr/bin/env python3
"""
HSTECH Index Estimation System - Bloomberg Integration Example

This script demonstrates how to use the HSTECH estimation system with Bloomberg data.
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime, timezone

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.models import Config
from src.estimation import HSTECHEstimator
from src.data import DataManager, BloombergFetcher
from src.utils import setup_logging, MarketHoursChecker


async def test_bloomberg_connection():
    """Test Bloomberg connection and data availability."""
    
    print("\n" + "="*60)
    print("BLOOMBERG CONNECTION TEST")
    print("="*60)
    
    # Load configuration
    try:
        config = Config.from_yaml("config/config.yaml")
        print("✓ Loaded configuration")
    except FileNotFoundError:
        print("⚠ Configuration file not found, using defaults")
        config = Config()
    
    # Initialize Bloomberg fetcher
    bloomberg_fetcher = BloombergFetcher(config)
    
    print(f"\nBloomberg Status:")
    print(f"  Terminal Available: {bloomberg_fetcher.terminal_available}")
    print(f"  Cloud API Key: {'✓' if bloomberg_fetcher.bloomberg_api_key else '✗'}")
    print(f"  Cloud URL: {bloomberg_fetcher.bloomberg_cloud_url or 'Not configured'}")
    
    # Test ADR price fetching
    print(f"\nTesting ADR Price Fetching...")
    test_symbols = ["TCEHY", "BABA", "JD"]
    
    try:
        adr_prices = await bloomberg_fetcher.fetch_adr_prices(test_symbols)
        
        if adr_prices:
            print(f"✓ Successfully fetched {len(adr_prices)} ADR prices:")
            for symbol, price_data in adr_prices.items():
                print(f"  {symbol}: {price_data.price} {price_data.currency} (from {price_data.source})")
        else:
            print("⚠ No ADR prices fetched")
            
    except Exception as e:
        print(f"✗ ADR price fetch failed: {e}")
    
    # Test HK stock fetching
    print(f"\nTesting HK Stock Fetching...")
    hk_symbols = ["0700.HK", "9988.HK", "3690.HK"]
    
    try:
        hk_prices = await bloomberg_fetcher.fetch_hk_prices(hk_symbols)
        
        if hk_prices:
            print(f"✓ Successfully fetched {len(hk_prices)} HK prices:")
            for symbol, price_data in hk_prices.items():
                print(f"  {symbol}: {price_data.price} {price_data.currency} (from {price_data.source})")
        else:
            print("⚠ No HK prices fetched")
            
    except Exception as e:
        print(f"✗ HK price fetch failed: {e}")
    
    # Test currency rate
    print(f"\nTesting Currency Rate Fetching...")
    
    try:
        currency_rate = await bloomberg_fetcher.fetch_currency_rate("USD", "HKD")
        print(f"✓ USD/HKD rate: {currency_rate.rate:.4f} (from {currency_rate.source})")
        
    except Exception as e:
        print(f"✗ Currency rate fetch failed: {e}")
    
    # Test HSTECH index
    print(f"\nTesting HSTECH Index Fetching...")
    
    try:
        hstech_data = await bloomberg_fetcher.fetch_hstech_index()
        
        if hstech_data:
            print(f"✓ HSTECH Index: {hstech_data.value:.2f}")
            if hstech_data.change:
                print(f"  Change: {hstech_data.change:+.2f} ({hstech_data.change_percent:+.2f}%)")
        else:
            print("⚠ No HSTECH index data fetched")
            
    except Exception as e:
        print(f"✗ HSTECH index fetch failed: {e}")


async def run_bloomberg_estimation():
    """Run full HSTECH estimation using Bloomberg data."""
    
    print("\n" + "="*60)
    print("HSTECH ESTIMATION WITH BLOOMBERG DATA")
    print("="*60)
    
    # Load configuration
    try:
        config = Config.from_yaml("config/config.yaml")
        print("✓ Loaded configuration from config/config.yaml")
    except FileNotFoundError:
        print("⚠ Configuration file not found, using defaults")
        config = Config()
    
    # Setup logging
    logger = setup_logging(config.logging)
    logger.info("Starting Bloomberg-based HSTECH estimation")
    
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
    print("\n3. Fetching market data from Bloomberg...")
    
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
            
            # Show data sources
            print(f"\n   Data Sources Used:")
            for symbol, price_data in list(data_manager.current_adr_prices.items())[:3]:
                print(f"     {symbol}: {price_data.source}")
            
        else:
            print("⚠ Failed to fetch some market data")
            
    except Exception as e:
        print(f"✗ Error fetching data: {e}")
        return
    
    # Run estimation
    print("\n4. Running HSTECH estimation...")
    
    try:
        # Get estimation data
        estimation_data = await data_manager.get_estimation_data()
        
        # Fetch historical data for covariance modeling
        print("   Fetching historical data from Bloomberg...")
        from data.hstech_components import HSTECH_COMPONENTS
        symbols = [stock.symbol for stock in HSTECH_COMPONENTS[:10]]  # Limit for demo
        historical_data = await data_manager.fetch_historical_data(symbols, days=60)
        
        print(f"   Historical data fetched for {len(historical_data)} symbols")
        
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
        print(f"\n   📈 ESTIMATED HSTECH INDEX: {result.estimated_value:.2f}")
        print(f"   🎯 Confidence: {result.confidence:.1%}")
        print(f"   ⏰ Timestamp: {result.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}")
        
        print(f"\n   ⚖️  Method Weights:")
        for method, weight in result.method_weights.items():
            print(f"     {method}: {weight:.1%}")
        
        print(f"\n   📊 Top Component Contributions:")
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
    
    print("\n" + "="*60)
    print("BLOOMBERG ESTIMATION COMPLETED")
    print("="*60)


async def main():
    """Main function to run Bloomberg integration examples."""
    
    print("HSTECH Index Estimation System - Bloomberg Integration")
    print("=" * 60)
    
    # Test Bloomberg connection first
    await test_bloomberg_connection()
    
    # Run full estimation
    await run_bloomberg_estimation()
    
    print("\nBloomberg Integration Notes:")
    print("1. Ensure Bloomberg Terminal is running (for BLPAPI)")
    print("2. Configure Bloomberg Cloud API credentials in config.yaml")
    print("3. Bloomberg provides more reliable and comprehensive data")
    print("4. Fallback to Yahoo Finance/Alpha Vantage if Bloomberg unavailable")
    print("5. Bloomberg supports both real-time and historical data")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nBloomberg example interrupted by user")
    except Exception as e:
        print(f"\nBloomberg example failed with error: {e}")
        sys.exit(1)
