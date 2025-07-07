"""
Main QDII Fund Data Scraper
This script attempts to use Wind API first, then falls back to alternative sources.
"""

import sys
import os
from datetime import datetime

def check_wind_availability():
    """Check if WindPy is available"""
    try:
        from WindPy import w
        return True
    except ImportError:
        return False

def run_wind_scraper():
    """Run the Wind-based scraper"""
    try:
        from wind_qdii_scraper import main as wind_main
        print("Using Wind API for data retrieval...")
        wind_main()
        return True
    except Exception as e:
        print(f"Error running Wind scraper: {str(e)}")
        return False

def run_alternative_scraper():
    """Run the alternative scraper"""
    try:
        from alternative_qdii_scraper import main as alt_main
        print("Using alternative public sources for data retrieval...")
        alt_main()
        return True
    except Exception as e:
        print(f"Error running alternative scraper: {str(e)}")
        return False

def main():
    """Main function that orchestrates the data scraping"""
    print("QDII Fund Data Scraper")
    print("=" * 50)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Check if Wind is available
    wind_available = check_wind_availability()
    
    if wind_available:
        print("WindPy detected. Attempting to use Wind API...")
        success = run_wind_scraper()

        if not success:
            print("\nWind API failed. Falling back to alternative sources...")
            success = run_alternative_scraper()
    else:
        print("WindPy not available. Using alternative public sources...")
        success = run_alternative_scraper()

    # If Wind failed, always try alternative as backup
    if not success or wind_available:
        print("\nAlso running alternative scraper for comparison...")
        alt_success = run_alternative_scraper()
        success = success or alt_success
    
    if success:
        print(f"\nData scraping completed successfully at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("\nGenerated files:")
        
        # List generated CSV files
        csv_files = [f for f in os.listdir('.') if f.endswith('.csv')]
        for csv_file in csv_files:
            file_size = os.path.getsize(csv_file)
            print(f"  - {csv_file} ({file_size} bytes)")
    else:
        print("\nData scraping failed. Please check the error messages above.")

if __name__ == "__main__":
    main()
