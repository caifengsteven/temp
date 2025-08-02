#!/usr/bin/env python3
"""
Robust SPX Volatility Surface Retrieval with retry logic
"""

import pandas as pd
import numpy as np
from datetime import datetime
import json
import time
import sys

try:
    from xbbg import blp
    print("✓ xbbg library imported successfully")
except ImportError as e:
    print(f"✗ Error importing xbbg: {e}")
    sys.exit(1)

def wait_for_bloomberg_connection(max_retries=5, wait_seconds=10):
    """Wait for Bloomberg connection with retries"""
    print("Waiting for Bloomberg connection...")
    
    for attempt in range(max_retries):
        try:
            print(f"Attempt {attempt + 1}/{max_retries}")
            
            # Try multiple tickers to find one that works
            test_tickers = ['SPY US Equity', 'AAPL US Equity', 'MSFT US Equity']
            
            for ticker in test_tickers:
                try:
                    data = blp.bdp(tickers=ticker, flds=['PX_LAST'])
                    if not data.empty and len(data.columns) > 0:
                        price = data.iloc[0, 0]
                        print(f"✓ Connection successful! {ticker} = {price}")
                        return True
                except:
                    continue
            
            if attempt < max_retries - 1:
                print(f"No data received. Waiting {wait_seconds} seconds...")
                time.sleep(wait_seconds)
            
        except Exception as e:
            print(f"Connection attempt failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(wait_seconds)
    
    return False

def get_spx_current_level():
    """Get current SPX level with multiple ticker attempts"""
    spx_tickers = ['SPX Index', 'SPY US Equity']
    
    for ticker in spx_tickers:
        try:
            data = blp.bdp(tickers=ticker, flds=['PX_LAST'])
            if not data.empty:
                level = data.iloc[0, 0]
                print(f"Current {ticker}: {level}")
                if ticker == 'SPY US Equity':
                    # SPY is roughly 1/10th of SPX, so multiply by 10 for approximation
                    return level * 10
                return level
        except Exception as e:
            print(f"Failed to get {ticker}: {e}")
    
    return None

def get_vix_level():
    """Get current VIX level"""
    try:
        data = blp.bdp(tickers='VIX Index', flds=['PX_LAST'])
        if not data.empty:
            vix = data.iloc[0, 0]
            print(f"Current VIX: {vix}")
            return vix
    except Exception as e:
        print(f"Failed to get VIX: {e}")
    return None

def get_sample_options_data():
    """Try to get some sample options data"""
    try:
        print("Attempting to retrieve sample options data...")
        
        # Try to get some SPY options (more liquid than SPX)
        spy_data = blp.bdp(tickers='SPY US Equity', flds=['PX_LAST'])
        if spy_data.empty:
            print("Cannot get SPY price, skipping options")
            return None
        
        spy_price = spy_data.iloc[0, 0]
        print(f"SPY current price: {spy_price}")
        
        # Try to construct some option tickers
        # Format: SPY US MM/DD/YY C/P[Strike] Equity
        from datetime import datetime, timedelta
        
        # Get next Friday (typical option expiry)
        today = datetime.now()
        days_ahead = 4 - today.weekday()  # Friday is 4
        if days_ahead <= 0:
            days_ahead += 7
        next_friday = today + timedelta(days=days_ahead)
        
        # Try a few different expiry formats
        expiry_formats = [
            next_friday.strftime("%m/%d/%y"),
            next_friday.strftime("%m/%d/%Y"),
        ]
        
        # Try strikes around current price
        strikes = [int(spy_price), int(spy_price) + 5, int(spy_price) - 5]
        
        for expiry in expiry_formats:
            for strike in strikes:
                for opt_type in ['C', 'P']:
                    ticker = f'SPY US {expiry} {opt_type}{strike} Equity'
                    try:
                        print(f"Trying: {ticker}")
                        data = blp.bdp(tickers=ticker, flds=['PX_LAST', 'OPT_IMPLIED_VOLATILITY_MID'])
                        if not data.empty:
                            print(f"✓ Success with {ticker}")
                            print(data)
                            return data
                    except Exception as e:
                        print(f"Failed {ticker}: {e}")
        
        return None
        
    except Exception as e:
        print(f"Error in sample options: {e}")
        return None

def create_sample_volatility_surface():
    """Create a sample volatility surface with available data"""
    print("\n=== Creating Sample Volatility Surface ===")
    
    # Get current market data
    spx_level = get_spx_current_level()
    vix_level = get_vix_level()
    
    if spx_level is None:
        print("Cannot proceed without SPX level")
        return None
    
    # Create a basic surface structure
    surface_data = {
        'timestamp': datetime.now().isoformat(),
        'spx_level': spx_level,
        'vix_level': vix_level,
        'data_source': 'Bloomberg Terminal via xbbg',
        'notes': 'Sample surface - actual options data may require different approach'
    }
    
    # Try to get some actual options data
    options_data = get_sample_options_data()
    if options_data is not None:
        surface_data['sample_options'] = options_data.to_dict()
    
    return surface_data

def save_surface_data(surface_data, filename_prefix="spx_surface_sample"):
    """Save the surface data to files"""
    if surface_data is None:
        print("No data to save")
        return
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save as JSON
    json_filename = f"{filename_prefix}_{timestamp}.json"
    with open(json_filename, 'w') as f:
        json.dump(surface_data, f, indent=2, default=str)
    print(f"Surface data saved to: {json_filename}")
    
    # Save summary
    summary = {
        'file': json_filename,
        'timestamp': surface_data.get('timestamp'),
        'spx_level': surface_data.get('spx_level'),
        'vix_level': surface_data.get('vix_level'),
        'has_options_data': 'sample_options' in surface_data
    }
    
    summary_filename = f"{filename_prefix}_summary_{timestamp}.txt"
    with open(summary_filename, 'w') as f:
        f.write("SPX Volatility Surface Summary\n")
        f.write("=" * 30 + "\n")
        for key, value in summary.items():
            f.write(f"{key}: {value}\n")
    
    print(f"Summary saved to: {summary_filename}")
    return json_filename, summary_filename

def main():
    """Main function"""
    print("=== SPX Volatility Surface Retrieval (Robust Version) ===")
    print(f"Started at: {datetime.now()}")
    
    # Wait for Bloomberg connection
    if not wait_for_bloomberg_connection():
        print("❌ Could not establish Bloomberg connection")
        print("\nPlease ensure:")
        print("1. Bloomberg Terminal is fully loaded and logged in")
        print("2. You have market data permissions")
        print("3. Try running a simple query in Bloomberg Terminal first")
        return
    
    print("✅ Bloomberg connection established")
    
    # Create surface data
    surface_data = create_sample_volatility_surface()
    
    # Save data
    files = save_surface_data(surface_data)
    
    if files:
        print(f"\n✅ Success! Files created:")
        print(f"   Data: {files[0]}")
        print(f"   Summary: {files[1]}")
    else:
        print("❌ Failed to save data")

if __name__ == "__main__":
    main()
