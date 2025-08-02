#!/usr/bin/env python3
"""
Complete SPX Volatility Surface Solution
- Includes correct Bloomberg code using xbbg
- Provides sample/mock data when Bloomberg is not available
- Comprehensive error handling and troubleshooting
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import sys

# Try to import xbbg
try:
    from xbbg import blp
    BLOOMBERG_AVAILABLE = True
    print("✓ xbbg library imported successfully")
except ImportError as e:
    BLOOMBERG_AVAILABLE = False
    print(f"⚠️  xbbg not available: {e}")
    print("Will create sample volatility surface instead")

def test_bloomberg_connection():
    """Test Bloomberg connection"""
    if not BLOOMBERG_AVAILABLE:
        return False, None
    
    try:
        print("Testing Bloomberg connection...")
        
        # Test multiple tickers to find one that works
        test_tickers = ['SPY US Equity', 'SPX Index', 'VIX Index']
        
        for ticker in test_tickers:
            try:
                data = blp.bdp(tickers=ticker, flds='PX_LAST')
                if not data.empty:
                    price = data.iloc[0, 0]
                    print(f"✓ Bloomberg working: {ticker} = {price}")
                    return True, price if 'SPX' in ticker else price * 10  # Approximate SPX from SPY
            except:
                continue
        
        print("❌ Bloomberg connection failed - all test queries returned empty")
        return False, None
        
    except Exception as e:
        print(f"❌ Bloomberg error: {e}")
        return False, None

def get_bloomberg_volatility_surface():
    """Get real volatility surface from Bloomberg"""
    try:
        print("Retrieving SPX volatility surface from Bloomberg...")
        
        # Get current SPX level
        spx_data = blp.bdp(tickers='SPX Index', flds='PX_LAST')
        spx_level = spx_data.iloc[0, 0] if not spx_data.empty else None
        
        # Get VIX
        vix_data = blp.bdp(tickers='VIX Index', flds='PX_LAST')
        vix_level = vix_data.iloc[0, 0] if not vix_data.empty else None
        
        # Get SPX option chain
        chain_data = blp.bds(tickers='SPX Index', flds='OPT_CHAIN')
        
        if chain_data.empty:
            print("No option chain data available")
            return None
        
        # Get option tickers
        if 'opt_chain' in chain_data.columns:
            option_tickers = chain_data['opt_chain'].tolist()[:100]  # Limit to 100
        else:
            option_tickers = chain_data.index.tolist()[:100]
        
        # Get volatility data for options
        vol_fields = [
            'PX_LAST', 'OPT_IMPLIED_VOLATILITY_MID', 'OPT_IMPLIED_VOLATILITY_BID',
            'OPT_IMPLIED_VOLATILITY_ASK', 'OPT_STRIKE_PX', 'OPT_EXPIRE_DT', 'OPT_PUT_CALL'
        ]
        
        vol_data = blp.bdp(tickers=option_tickers, flds=vol_fields)
        
        return {
            'spx_level': spx_level,
            'vix_level': vix_level,
            'volatility_data': vol_data,
            'source': 'Bloomberg Terminal'
        }
        
    except Exception as e:
        print(f"Error getting Bloomberg data: {e}")
        return None

def create_sample_volatility_surface():
    """Create a realistic sample volatility surface for demonstration"""
    print("Creating sample SPX volatility surface...")
    
    # Current market approximations (as of typical market conditions)
    spx_level = 4500  # Approximate SPX level
    vix_level = 18.5  # Approximate VIX level
    
    # Create strikes around current level
    strikes = np.arange(4200, 4800, 25)  # Strikes from 4200 to 4800 in 25-point increments
    
    # Create expiration dates (next 6 monthly expirations)
    today = datetime.now()
    expirations = []
    for i in range(1, 7):
        # Third Friday of each month (approximate)
        exp_month = today.month + i
        exp_year = today.year
        if exp_month > 12:
            exp_month -= 12
            exp_year += 1
        
        # Approximate third Friday
        exp_date = datetime(exp_year, exp_month, 15)
        # Adjust to Friday
        exp_date = exp_date + timedelta(days=(4 - exp_date.weekday()) % 7)
        expirations.append(exp_date)
    
    # Create volatility surface data
    surface_data = []
    
    for exp_date in expirations:
        days_to_exp = (exp_date - today).days
        time_to_exp = days_to_exp / 365.0
        
        for strike in strikes:
            moneyness = strike / spx_level
            
            # Create both calls and puts
            for option_type in ['C', 'P']:
                # Simple volatility smile model
                # Higher vol for OTM options, lower for ATM
                atm_vol = vix_level / 100.0  # Convert VIX to decimal
                
                # Add smile effect
                smile_factor = 0.5 * abs(moneyness - 1.0)  # Smile increases away from ATM
                term_structure = 0.02 * np.sqrt(time_to_exp)  # Term structure effect
                
                implied_vol = atm_vol + smile_factor + term_structure
                
                # Add some randomness for realism
                implied_vol += np.random.normal(0, 0.01)
                implied_vol = max(0.05, implied_vol)  # Minimum 5% vol
                
                # Estimate option price using simple Black-Scholes approximation
                if option_type == 'C':
                    intrinsic = max(0, spx_level - strike)
                else:
                    intrinsic = max(0, strike - spx_level)
                
                time_value = implied_vol * spx_level * np.sqrt(time_to_exp) * 0.4
                option_price = intrinsic + time_value
                
                surface_data.append({
                    'ticker': f'SPX {exp_date.strftime("%m/%d/%y")} {option_type}{int(strike)} Index',
                    'strike': strike,
                    'expiration': exp_date,
                    'option_type': option_type,
                    'days_to_expiry': days_to_exp,
                    'moneyness': moneyness,
                    'implied_vol_mid': implied_vol,
                    'implied_vol_bid': implied_vol - 0.005,
                    'implied_vol_ask': implied_vol + 0.005,
                    'option_price': option_price,
                    'time_to_expiry': time_to_exp
                })
    
    df = pd.DataFrame(surface_data)
    
    return {
        'spx_level': spx_level,
        'vix_level': vix_level,
        'volatility_data': df,
        'source': 'Sample/Mock Data'
    }

def save_volatility_surface(surface_dict):
    """Save volatility surface to files"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Extract data
    spx_level = surface_dict['spx_level']
    vix_level = surface_dict['vix_level']
    vol_data = surface_dict['volatility_data']
    source = surface_dict['source']
    
    # Save main data
    csv_filename = f"spx_volatility_surface_{timestamp}.csv"
    vol_data.to_csv(csv_filename, index=False)
    print(f"✓ Volatility surface saved to: {csv_filename}")
    
    # Separate calls and puts
    if 'option_type' in vol_data.columns:
        calls = vol_data[vol_data['option_type'] == 'C']
        puts = vol_data[vol_data['option_type'] == 'P']
        
        calls_filename = f"spx_calls_{timestamp}.csv"
        puts_filename = f"spx_puts_{timestamp}.csv"
        
        calls.to_csv(calls_filename, index=False)
        puts.to_csv(puts_filename, index=False)
        
        print(f"✓ Calls saved to: {calls_filename}")
        print(f"✓ Puts saved to: {puts_filename}")
    
    # Create summary
    summary = {
        'timestamp': timestamp,
        'data_source': source,
        'spx_level': float(spx_level) if spx_level else None,
        'vix_level': float(vix_level) if vix_level else None,
        'total_options': len(vol_data),
        'unique_strikes': len(vol_data['strike'].unique()) if 'strike' in vol_data.columns else 0,
        'unique_expirations': len(vol_data['expiration'].unique()) if 'expiration' in vol_data.columns else 0,
    }
    
    if 'implied_vol_mid' in vol_data.columns:
        vol_stats = vol_data['implied_vol_mid'].describe()
        summary['volatility_statistics'] = {
            'min': float(vol_stats['min']),
            'max': float(vol_stats['max']),
            'mean': float(vol_stats['mean']),
            'std': float(vol_stats['std'])
        }
    
    # Save summary
    summary_filename = f"spx_volatility_summary_{timestamp}.json"
    with open(summary_filename, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"✓ Summary saved to: {summary_filename}")
    
    return csv_filename, summary_filename

def main():
    """Main function"""
    print("=== SPX Volatility Surface Retrieval ===")
    print(f"Started at: {datetime.now()}")
    print()
    
    # Try Bloomberg first
    bloomberg_working, spx_level = test_bloomberg_connection()
    
    if bloomberg_working:
        print("Attempting to retrieve real Bloomberg data...")
        surface_dict = get_bloomberg_volatility_surface()
        
        if surface_dict is None:
            print("Bloomberg data retrieval failed, creating sample data...")
            surface_dict = create_sample_volatility_surface()
    else:
        print("Bloomberg not available, creating sample volatility surface...")
        surface_dict = create_sample_volatility_surface()
    
    # Save the data
    csv_file, summary_file = save_volatility_surface(surface_dict)
    
    print(f"\n✅ Success!")
    print(f"Data source: {surface_dict['source']}")
    print(f"SPX Level: {surface_dict['spx_level']}")
    print(f"VIX Level: {surface_dict['vix_level']}")
    print(f"Main file: {csv_file}")
    print(f"Summary: {summary_file}")
    
    if surface_dict['source'] != 'Bloomberg Terminal':
        print(f"\n📋 Bloomberg Troubleshooting:")
        print("1. Ensure Bloomberg Terminal is running and logged in")
        print("2. Test in Terminal: SPX <Index> <GO>")
        print("3. Verify market data permissions")
        print("4. Wait 2-3 minutes after login before running script")
        print("5. Install Bloomberg API: pip install blpapi --index-url=https://bcms.bloomberg.com/pip/simple/")

if __name__ == "__main__":
    main()
