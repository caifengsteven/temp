#!/usr/bin/env python3
# Using Python 3.10 environment
"""
SPX Index Volatility Surface Retrieval using xbbg
This script retrieves implied volatility data for SPX options and constructs a volatility surface.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
import sys

try:
    from xbbg import blp
    print("xbbg library imported successfully")
except ImportError as e:
    print(f"Error importing xbbg: {e}")
    print("Please ensure Bloomberg Terminal is running and xbbg is properly installed")
    sys.exit(1)

def test_bloomberg_connection():
    """Test Bloomberg connection with a simple query"""
    try:
        print("Testing Bloomberg connection...")
        print("Please ensure Bloomberg Terminal is running and logged in.")

        # Test with a simple equity query
        test_data = blp.bdp(tickers='SPX Index', flds=['PX_LAST'])
        print(f"Raw response: {test_data}")
        print(f"Response shape: {test_data.shape}")
        print(f"Response columns: {test_data.columns.tolist()}")

        if test_data.empty:
            print("Bloomberg returned empty data. Please check:")
            print("1. Bloomberg Terminal is running")
            print("2. Bloomberg Terminal is logged in")
            print("3. You have access to SPX Index data")
            return False

        spx_level = test_data.iloc[0, 0]
        print(f"Bloomberg connection successful. SPX current level: {spx_level}")
        return True
    except Exception as e:
        print(f"Bloomberg connection failed: {e}")
        print("Please ensure:")
        print("1. Bloomberg Terminal is running")
        print("2. Bloomberg Terminal is logged in")
        print("3. Bloomberg API is properly configured")
        return False

def get_spx_option_chain():
    """Retrieve SPX option chain"""
    try:
        print("Retrieving SPX option chain...")
        chain = blp.bds(tickers='SPX Index', flds='OPT_CHAIN')
        print(f"Retrieved {len(chain)} option contracts")
        return chain
    except Exception as e:
        print(f"Error retrieving option chain: {e}")
        return None

def get_spx_volatility_data(option_tickers, max_options=100):
    """Retrieve implied volatility for SPX options"""
    try:
        print(f"Retrieving implied volatility for {min(len(option_tickers), max_options)} options...")
        
        # Limit the number of options to avoid overwhelming the API
        limited_tickers = option_tickers[:max_options]
        
        # Fields for implied volatility
        vol_fields = [
            'OPT_IMPLIED_VOLATILITY_MID',
            'OPT_IMPLIED_VOLATILITY_BID', 
            'OPT_IMPLIED_VOLATILITY_ASK',
            'OPT_STRIKE_PX',
            'OPT_EXPIRE_DT',
            'OPT_PUT_CALL',
            'PX_LAST'
        ]
        
        vol_data = blp.bdp(tickers=limited_tickers, flds=vol_fields)
        print(f"Successfully retrieved volatility data for {len(vol_data)} options")
        return vol_data
        
    except Exception as e:
        print(f"Error retrieving volatility data: {e}")
        return None

def structure_volatility_surface(vol_data):
    """Structure the volatility data into a surface format"""
    try:
        print("Structuring volatility surface...")
        
        # Clean and prepare the data
        df = vol_data.copy()
        df = df.dropna(subset=['OPT_IMPLIED_VOLATILITY_MID'])
        
        # Convert expiration dates to datetime
        df['OPT_EXPIRE_DT'] = pd.to_datetime(df['OPT_EXPIRE_DT'])
        
        # Calculate days to expiration
        today = datetime.now()
        df['days_to_expiry'] = (df['OPT_EXPIRE_DT'] - today).dt.days
        
        # Filter out expired options
        df = df[df['days_to_expiry'] > 0]
        
        # Create moneyness (strike/spot ratio)
        current_spx = blp.bdp(tickers='SPX Index', flds=['PX_LAST']).iloc[0, 0]
        df['moneyness'] = df['OPT_STRIKE_PX'] / current_spx
        
        # Separate calls and puts
        calls = df[df['OPT_PUT_CALL'] == 'C'].copy()
        puts = df[df['OPT_PUT_CALL'] == 'P'].copy()
        
        print(f"Structured surface with {len(calls)} calls and {len(puts)} puts")
        print(f"Current SPX level: {current_spx}")
        print(f"Strike range: {df['OPT_STRIKE_PX'].min()} - {df['OPT_STRIKE_PX'].max()}")
        print(f"Expiry range: {df['days_to_expiry'].min()} - {df['days_to_expiry'].max()} days")
        
        return df, calls, puts, current_spx
        
    except Exception as e:
        print(f"Error structuring volatility surface: {e}")
        return None, None, None, None

def save_volatility_data(df, calls, puts, current_spx, filename_prefix="spx_vol_surface"):
    """Save volatility surface data to files"""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save complete dataset
        csv_filename = f"{filename_prefix}_{timestamp}.csv"
        df.to_csv(csv_filename, index=True)
        print(f"Complete volatility data saved to: {csv_filename}")
        
        # Save calls and puts separately
        calls_filename = f"{filename_prefix}_calls_{timestamp}.csv"
        puts_filename = f"{filename_prefix}_puts_{timestamp}.csv"
        calls.to_csv(calls_filename, index=True)
        puts.to_csv(puts_filename, index=True)
        print(f"Calls data saved to: {calls_filename}")
        print(f"Puts data saved to: {puts_filename}")
        
        # Create summary statistics
        summary = {
            'timestamp': timestamp,
            'current_spx_level': float(current_spx),
            'total_options': len(df),
            'calls_count': len(calls),
            'puts_count': len(puts),
            'strike_range': {
                'min': float(df['OPT_STRIKE_PX'].min()),
                'max': float(df['OPT_STRIKE_PX'].max())
            },
            'expiry_range_days': {
                'min': int(df['days_to_expiry'].min()),
                'max': int(df['days_to_expiry'].max())
            },
            'volatility_stats': {
                'min_vol': float(df['OPT_IMPLIED_VOLATILITY_MID'].min()),
                'max_vol': float(df['OPT_IMPLIED_VOLATILITY_MID'].max()),
                'mean_vol': float(df['OPT_IMPLIED_VOLATILITY_MID'].mean())
            }
        }
        
        # Save summary as JSON
        summary_filename = f"{filename_prefix}_summary_{timestamp}.json"
        with open(summary_filename, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"Summary saved to: {summary_filename}")
        
        return csv_filename, summary_filename
        
    except Exception as e:
        print(f"Error saving data: {e}")
        return None, None

def main():
    """Main function to retrieve and save SPX volatility surface"""
    print("=== SPX Volatility Surface Retrieval ===")
    print(f"Started at: {datetime.now()}")
    
    # Test Bloomberg connection
    if not test_bloomberg_connection():
        print("Cannot proceed without Bloomberg connection")
        return
    
    # Get option chain
    chain = get_spx_option_chain()
    if chain is None or len(chain) == 0:
        print("Failed to retrieve option chain")
        return
    
    # Extract option tickers
    option_tickers = chain['opt_chain'].tolist() if 'opt_chain' in chain.columns else chain.index.tolist()
    print(f"Found {len(option_tickers)} option contracts")
    
    # Get volatility data
    vol_data = get_spx_volatility_data(option_tickers, max_options=200)
    if vol_data is None:
        print("Failed to retrieve volatility data")
        return
    
    # Structure the data
    df, calls, puts, current_spx = structure_volatility_surface(vol_data)
    if df is None:
        print("Failed to structure volatility surface")
        return
    
    # Save the data
    csv_file, summary_file = save_volatility_data(df, calls, puts, current_spx)
    if csv_file:
        print(f"\n=== Success! ===")
        print(f"Volatility surface data saved successfully")
        print(f"Main file: {csv_file}")
        print(f"Summary: {summary_file}")
    else:
        print("Failed to save data")

if __name__ == "__main__":
    main()
