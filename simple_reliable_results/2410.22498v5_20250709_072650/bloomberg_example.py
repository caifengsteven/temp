#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bloomberg Data Integration Example for VIX Stochastic Volatility Model
Demonstrates proper usage of xbbg library following the official documentation
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import Bloomberg API
try:
    from xbbg import blp
    BLOOMBERG_AVAILABLE = True
    print("✓ Bloomberg xbbg library imported successfully")
except ImportError:
    BLOOMBERG_AVAILABLE = False
    print("✗ xbbg library not available")
    print("Install with: pip install xbbg")
    exit(1)

def test_bloomberg_connection():
    """Test basic Bloomberg connectivity."""
    print("\n" + "="*50)
    print("TESTING BLOOMBERG CONNECTION")
    print("="*50)
    
    try:
        # Test with SPX Index - most reliable ticker
        print("Testing connection with SPX Index...")
        test_data = blp.bdp(tickers='SPX Index', flds=['PX_LAST', 'NAME'])
        
        if test_data is not None and not test_data.empty:
            price = test_data.iloc[0, 0]
            name = test_data.iloc[0, 1]
            print(f"✓ Connection successful!")
            print(f"  {name}: {price:.2f}")
            return True
        else:
            print("✗ No data returned")
            return False
            
    except Exception as e:
        print(f"✗ Connection failed: {e}")
        print("\nTroubleshooting:")
        print("1. Ensure Bloomberg Terminal is running")
        print("2. Check Bloomberg API permissions")
        print("3. Verify network connectivity")
        return False

def get_vix_data(start_date='2020-01-01'):
    """Load VIX data using xbbg."""
    print(f"\nLoading VIX data from {start_date}...")
    
    try:
        # Load VIX historical data
        vix_data = blp.bdh(
            tickers='VIX Index',
            flds=['PX_LAST', 'PX_HIGH', 'PX_LOW'],
            start_date=start_date,
            end_date=datetime.now().strftime('%Y-%m-%d')
        )
        
        if vix_data is not None and not vix_data.empty:
            print(f"✓ Loaded {len(vix_data)} VIX observations")
            print(f"  Date range: {vix_data.index[0].strftime('%Y-%m-%d')} to {vix_data.index[-1].strftime('%Y-%m-%d')}")
            print(f"  VIX range: {vix_data.iloc[:, 0].min():.2f} - {vix_data.iloc[:, 0].max():.2f}")
            
            # Clean column names
            vix_data.columns = ['VIX', 'VIX_High', 'VIX_Low']
            vix_data['log_VIX'] = np.log(vix_data['VIX'])
            
            return vix_data
        else:
            print("✗ No VIX data returned")
            return None
            
    except Exception as e:
        print(f"✗ Error loading VIX data: {e}")
        return None

def get_corporate_bond_data(start_date='2020-01-01'):
    """Load corporate bond data using xbbg."""
    print(f"\nLoading corporate bond data from {start_date}...")
    
    # Corporate bond tickers
    bond_tickers = [
        'HYG US Equity',   # iShares High Yield Corporate Bond ETF
        'LQD US Equity',   # iShares Investment Grade Corporate Bond ETF
        'VCIT US Equity',  # Vanguard Intermediate-Term Corporate Bond ETF
    ]
    
    try:
        # Load bond price data
        bond_data = blp.bdh(
            tickers=bond_tickers,
            flds=['PX_LAST'],
            start_date=start_date,
            end_date=datetime.now().strftime('%Y-%m-%d')
        )
        
        if bond_data is not None and not bond_data.empty:
            print(f"✓ Loaded {len(bond_data)} bond price observations")
            
            # Calculate returns
            bond_returns = bond_data.pct_change().dropna()
            print(f"✓ Calculated {len(bond_returns)} return observations")
            
            return {
                'prices': bond_data,
                'returns': bond_returns
            }
        else:
            print("✗ No bond data returned")
            return None
            
    except Exception as e:
        print(f"✗ Error loading bond data: {e}")
        return None

def get_spread_data(start_date='2020-01-01'):
    """Load corporate bond spread data using xbbg."""
    print(f"\nLoading spread data from {start_date}...")
    
    # Spread tickers
    spread_tickers = [
        'LUACOAS Index',   # US Credit OAS
        'C0A0 Index',      # US Investment Grade Corporate OAS
        'H0A0 Index',      # US High Yield Corporate OAS
    ]
    
    try:
        spread_data = blp.bdh(
            tickers=spread_tickers,
            flds=['PX_LAST'],
            start_date=start_date,
            end_date=datetime.now().strftime('%Y-%m-%d')
        )
        
        if spread_data is not None and not spread_data.empty:
            print(f"✓ Loaded {len(spread_data)} spread observations")
            
            # Clean column names
            spread_data.columns = ['Credit_OAS', 'IG_OAS', 'HY_OAS']
            
            return spread_data
        else:
            print("✗ No spread data returned")
            return None
            
    except Exception as e:
        print(f"✗ Error loading spread data: {e}")
        return None

def get_current_market_snapshot():
    """Get current market data snapshot."""
    print("\n" + "="*50)
    print("CURRENT MARKET SNAPSHOT")
    print("="*50)
    
    tickers = [
        'VIX Index',
        'SPX Index', 
        'HYG US Equity',
        'LQD US Equity',
        'LUACOAS Index',
    ]
    
    try:
        current_data = blp.bdp(
            tickers=tickers,
            flds=['PX_LAST', 'CHG_PCT_1D', 'NAME']
        )
        
        if current_data is not None and not current_data.empty:
            print(f"{'Ticker':<15} {'Name':<30} {'Price':<10} {'Change %':<10}")
            print("-" * 70)
            
            for ticker in tickers:
                if ticker in current_data.index:
                    name = current_data.loc[ticker, 'NAME'][:25] + "..." if len(str(current_data.loc[ticker, 'NAME'])) > 25 else current_data.loc[ticker, 'NAME']
                    price = current_data.loc[ticker, 'PX_LAST']
                    change = current_data.loc[ticker, 'CHG_PCT_1D']
                    print(f"{ticker:<15} {name:<30} {price:<10.2f} {change:<10.2f}")
            
            return current_data
        else:
            print("✗ No current data returned")
            return None
            
    except Exception as e:
        print(f"✗ Error getting current data: {e}")
        return None

def create_data_visualization(vix_data, bond_data, spread_data):
    """Create visualization of Bloomberg data."""
    print("\nCreating data visualization...")
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    
    # Plot VIX
    if vix_data is not None:
        axes[0].plot(vix_data.index, vix_data['VIX'], 'red', alpha=0.8, linewidth=1)
        axes[0].set_title('VIX Index (Bloomberg Data)', fontsize=14)
        axes[0].set_ylabel('VIX Level')
        axes[0].grid(True, alpha=0.3)
    
    # Plot Bond ETF prices
    if bond_data is not None:
        for col in bond_data['prices'].columns:
            axes[1].plot(bond_data['prices'].index, bond_data['prices'][col], 
                        alpha=0.8, linewidth=1, label=col[0])
        axes[1].set_title('Corporate Bond ETF Prices (Bloomberg Data)', fontsize=14)
        axes[1].set_ylabel('Price')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    
    # Plot Spreads
    if spread_data is not None:
        for col in spread_data.columns:
            axes[2].plot(spread_data.index, spread_data[col], 
                        alpha=0.8, linewidth=1, label=col)
        axes[2].set_title('Corporate Bond Spreads (Bloomberg Data)', fontsize=14)
        axes[2].set_ylabel('Spread (bps)')
        axes[2].set_xlabel('Date')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('bloomberg_data_visualization.png', dpi=300, bbox_inches='tight')
    print("✓ Saved bloomberg_data_visualization.png")
    
    return fig

def main():
    """Main function demonstrating Bloomberg data integration."""
    print("VIX STOCHASTIC VOLATILITY MODEL")
    print("Bloomberg Data Integration Example")
    print("Using xbbg library following official documentation")
    print("="*60)
    
    # Test connection
    if not test_bloomberg_connection():
        print("\nCannot proceed without Bloomberg connection.")
        return False
    
    # Get current market snapshot
    current_data = get_current_market_snapshot()
    
    # Load historical data
    start_date = (datetime.now() - timedelta(days=2*365)).strftime('%Y-%m-%d')
    
    vix_data = get_vix_data(start_date)
    bond_data = get_corporate_bond_data(start_date)
    spread_data = get_spread_data(start_date)
    
    # Create visualization
    if any([vix_data is not None, bond_data is not None, spread_data is not None]):
        fig = create_data_visualization(vix_data, bond_data, spread_data)
    
    # Summary
    print("\n" + "="*60)
    print("BLOOMBERG DATA INTEGRATION SUMMARY")
    print("="*60)
    print(f"✓ Connection test: {'Passed' if current_data is not None else 'Failed'}")
    print(f"✓ VIX data: {'Loaded' if vix_data is not None else 'Failed'}")
    print(f"✓ Bond data: {'Loaded' if bond_data is not None else 'Failed'}")
    print(f"✓ Spread data: {'Loaded' if spread_data is not None else 'Failed'}")
    
    if all([vix_data is not None, bond_data is not None]):
        print(f"\n✓ Ready for VIX stochastic volatility model analysis")
        print(f"  VIX observations: {len(vix_data)}")
        print(f"  Bond observations: {len(bond_data['returns'])}")
        
        return {
            'vix_data': vix_data,
            'bond_data': bond_data,
            'spread_data': spread_data,
            'current_data': current_data
        }
    else:
        print(f"\n⚠ Insufficient data for model analysis")
        return None

if __name__ == "__main__":
    if BLOOMBERG_AVAILABLE:
        result = main()
        if result:
            print("\n✓ Bloomberg integration example completed successfully!")
            plt.show()
        else:
            print("\n✗ Bloomberg integration example failed")
    else:
        print("Please install xbbg library first: pip install xbbg")
