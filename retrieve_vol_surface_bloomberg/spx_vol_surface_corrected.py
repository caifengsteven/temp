#!/usr/bin/env python3
"""
SPX Index Volatility Surface Retrieval using xbbg (Corrected Version)
Based on proper xbbg documentation: https://xbbg.readthedocs.io/en/latest/
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import sys

try:
    from xbbg import blp
    print("✓ xbbg library imported successfully")
except ImportError as e:
    print(f"✗ Error importing xbbg: {e}")
    print("Please ensure Bloomberg Terminal is running and xbbg is properly installed")
    sys.exit(1)

def test_bloomberg_connection():
    """Test Bloomberg connection with a simple query using correct xbbg syntax"""
    try:
        print("Testing Bloomberg connection...")
        
        # Test with SPX Index using correct xbbg syntax
        test_data = blp.bdp(tickers='SPX Index', flds='PX_LAST')
        print(f"Response: {test_data}")
        
        if test_data.empty:
            print("Bloomberg returned empty data")
            return False, None
            
        spx_level = test_data.iloc[0, 0]
        print(f"✓ Bloomberg connection successful. SPX current level: {spx_level}")
        return True, spx_level
        
    except Exception as e:
        print(f"✗ Bloomberg connection failed: {e}")
        return False, None

def get_spx_option_chain():
    """Retrieve SPX option chain using BDS"""
    try:
        print("Retrieving SPX option chain...")
        
        # Use BDS to get option chain
        chain_data = blp.bds(tickers='SPX Index', flds='OPT_CHAIN')
        print(f"Retrieved option chain with shape: {chain_data.shape}")
        print(f"Sample data:\n{chain_data.head()}")
        
        return chain_data
        
    except Exception as e:
        print(f"Error retrieving option chain: {e}")
        return None

def get_volatility_data_for_options(option_tickers, max_options=50):
    """Get implied volatility data for option tickers"""
    try:
        print(f"Retrieving volatility data for {min(len(option_tickers), max_options)} options...")
        
        # Limit number of options to avoid overwhelming the API
        limited_tickers = option_tickers[:max_options]
        
        # Fields for volatility data
        vol_fields = [
            'PX_LAST',
            'OPT_IMPLIED_VOLATILITY_MID',
            'OPT_IMPLIED_VOLATILITY_BID',
            'OPT_IMPLIED_VOLATILITY_ASK',
            'OPT_STRIKE_PX',
            'OPT_EXPIRE_DT',
            'OPT_PUT_CALL'
        ]
        
        # Use BDP to get current data for multiple tickers
        vol_data = blp.bdp(tickers=limited_tickers, flds=vol_fields)
        print(f"Retrieved volatility data with shape: {vol_data.shape}")
        print(f"Sample data:\n{vol_data.head()}")
        
        return vol_data
        
    except Exception as e:
        print(f"Error retrieving volatility data: {e}")
        return None

def get_vix_data():
    """Get current VIX level"""
    try:
        vix_data = blp.bdp(tickers='VIX Index', flds='PX_LAST')
        if not vix_data.empty:
            vix_level = vix_data.iloc[0, 0]
            print(f"Current VIX level: {vix_level}")
            return vix_level
    except Exception as e:
        print(f"Error getting VIX: {e}")
    return None

def create_volatility_surface(vol_data, spx_level):
    """Structure volatility data into surface format"""
    try:
        print("Structuring volatility surface...")
        
        if vol_data is None or vol_data.empty:
            print("No volatility data to structure")
            return None
        
        # Clean the data
        df = vol_data.copy()
        
        # Remove rows without implied volatility
        df = df.dropna(subset=['OPT_IMPLIED_VOLATILITY_MID'])
        
        if df.empty:
            print("No valid volatility data after cleaning")
            return None
        
        # Convert expiration dates
        if 'OPT_EXPIRE_DT' in df.columns:
            df['OPT_EXPIRE_DT'] = pd.to_datetime(df['OPT_EXPIRE_DT'])
            
            # Calculate days to expiration
            today = datetime.now()
            df['days_to_expiry'] = (df['OPT_EXPIRE_DT'] - today).dt.days
            
            # Filter out expired options
            df = df[df['days_to_expiry'] > 0]
        
        # Calculate moneyness if we have strike data
        if 'OPT_STRIKE_PX' in df.columns and spx_level:
            df['moneyness'] = df['OPT_STRIKE_PX'] / spx_level
        
        # Separate calls and puts if we have that data
        calls = puts = None
        if 'OPT_PUT_CALL' in df.columns:
            calls = df[df['OPT_PUT_CALL'] == 'C'].copy()
            puts = df[df['OPT_PUT_CALL'] == 'P'].copy()
            print(f"Structured surface: {len(calls)} calls, {len(puts)} puts")
        
        print(f"Final surface data shape: {df.shape}")
        return df, calls, puts
        
    except Exception as e:
        print(f"Error structuring surface: {e}")
        return None

def save_volatility_surface(surface_data, spx_level, vix_level, calls=None, puts=None):
    """Save volatility surface to files"""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save main surface data
        if surface_data is not None:
            csv_filename = f"spx_vol_surface_{timestamp}.csv"
            surface_data.to_csv(csv_filename, index=True)
            print(f"✓ Volatility surface saved to: {csv_filename}")
        
        # Save calls and puts separately if available
        if calls is not None and not calls.empty:
            calls_filename = f"spx_vol_calls_{timestamp}.csv"
            calls.to_csv(calls_filename, index=True)
            print(f"✓ Calls data saved to: {calls_filename}")
        
        if puts is not None and not puts.empty:
            puts_filename = f"spx_vol_puts_{timestamp}.csv"
            puts.to_csv(puts_filename, index=True)
            print(f"✓ Puts data saved to: {puts_filename}")
        
        # Create summary
        summary = {
            'timestamp': timestamp,
            'spx_level': float(spx_level) if spx_level else None,
            'vix_level': float(vix_level) if vix_level else None,
            'total_options': len(surface_data) if surface_data is not None else 0,
            'calls_count': len(calls) if calls is not None else 0,
            'puts_count': len(puts) if puts is not None else 0,
        }
        
        if surface_data is not None and not surface_data.empty:
            if 'OPT_IMPLIED_VOLATILITY_MID' in surface_data.columns:
                vol_col = surface_data['OPT_IMPLIED_VOLATILITY_MID']
                summary['volatility_stats'] = {
                    'min_vol': float(vol_col.min()),
                    'max_vol': float(vol_col.max()),
                    'mean_vol': float(vol_col.mean())
                }
        
        # Save summary
        summary_filename = f"spx_vol_summary_{timestamp}.json"
        with open(summary_filename, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"✓ Summary saved to: {summary_filename}")
        
        return summary_filename
        
    except Exception as e:
        print(f"Error saving data: {e}")
        return None

def main():
    """Main function to retrieve SPX volatility surface"""
    print("=== SPX Volatility Surface Retrieval (Corrected) ===")
    print(f"Started at: {datetime.now()}")
    
    # Test Bloomberg connection
    connection_ok, spx_level = test_bloomberg_connection()
    if not connection_ok:
        print("❌ Cannot proceed without Bloomberg connection")
        print("\nPlease ensure:")
        print("1. Bloomberg Terminal is running and logged in")
        print("2. You have market data permissions")
        return
    
    # Get VIX level
    vix_level = get_vix_data()
    
    # Get option chain
    chain_data = get_spx_option_chain()
    if chain_data is None or chain_data.empty:
        print("❌ Could not retrieve option chain")
        return
    
    # Extract option tickers from chain
    if 'opt_chain' in chain_data.columns:
        option_tickers = chain_data['opt_chain'].tolist()
    else:
        # If the structure is different, try to get tickers from index or other columns
        option_tickers = chain_data.index.tolist() if not chain_data.empty else []
    
    if not option_tickers:
        print("❌ No option tickers found in chain data")
        return
    
    print(f"Found {len(option_tickers)} option contracts")
    
    # Get volatility data
    vol_data = get_volatility_data_for_options(option_tickers, max_options=100)
    if vol_data is None:
        print("❌ Could not retrieve volatility data")
        return
    
    # Structure the surface
    surface_result = create_volatility_surface(vol_data, spx_level)
    if surface_result is None:
        print("❌ Could not structure volatility surface")
        return
    
    # Unpack results
    if len(surface_result) == 3:
        surface_data, calls, puts = surface_result
    else:
        surface_data = surface_result
        calls = puts = None
    
    # Save the data
    summary_file = save_volatility_surface(surface_data, spx_level, vix_level, calls, puts)
    
    if summary_file:
        print(f"\n✅ Success! SPX volatility surface retrieved and saved")
        print(f"Summary file: {summary_file}")
    else:
        print("❌ Failed to save volatility surface")

if __name__ == "__main__":
    main()
