"""
Test Curved Radius Supertrend with Real Data from NAS Database
"""

import pandas as pd
import numpy as np
import pymysql
import matplotlib.pyplot as plt
from datetime import datetime
from test_curved_supertrend import CurvedRadiusSupertrend

# ============================================================================
# Database Functions
# ============================================================================

def connect_to_nas():
    """Connect to NAS MySQL database"""
    config = {
        'host': '192.168.50.230',
        'port': 3306,
        'user': 'root',
        'password': '352471Cf!1',
        'database': 'us_stock_sip_min_aggs'
    }
    
    try:
        connection = pymysql.connect(**config)
        print(f"✓ Connected to NAS database: {config['database']}")
        return connection
    except Exception as e:
        print(f"✗ Error connecting to database: {e}")
        return None

def get_stock_data(connection, table_name='200309', ticker='A', limit=500):
    """
    Fetch stock data from NAS database
    
    Parameters:
    -----------
    connection : pymysql.Connection
        Database connection
    table_name : str
        Table name (format: YYYYMM)
    ticker : str
        Stock ticker symbol
    limit : int
        Number of records to fetch
    """
    try:
        query = f"""
        SELECT window_start, open, high, low, close, volume, transactions
        FROM `{table_name}`
        WHERE ticker = %s
        ORDER BY window_start ASC
        LIMIT %s
        """
        
        df = pd.read_sql(query, connection, params=(ticker, limit))
        
        if len(df) > 0:
            # Convert window_start from nanoseconds to datetime
            df['datetime'] = pd.to_datetime(df['window_start'], unit='ns')
            
            # Convert Decimal to float
            for col in ['open', 'high', 'low', 'close']:
                df[col] = df[col].astype(float)
            
            print(f"✓ Fetched {len(df)} records for {ticker} from table {table_name}")
            print(f"  Date range: {df['datetime'].min()} to {df['datetime'].max()}")
            print(f"  Price range: ${df['low'].min():.2f} - ${df['high'].max():.2f}")
            
            return df
        else:
            print(f"✗ No data found for {ticker} in table {table_name}")
            return None
            
    except Exception as e:
        print(f"✗ Error fetching data: {e}")
        return None

def list_available_tickers(connection, table_name='200309', limit=20):
    """List available tickers in a table"""
    try:
        query = f"""
        SELECT DISTINCT ticker, COUNT(*) as record_count
        FROM `{table_name}`
        GROUP BY ticker
        ORDER BY record_count DESC
        LIMIT %s
        """
        
        cursor = connection.cursor()
        cursor.execute(query, (limit,))
        results = cursor.fetchall()
        
        print(f"\n✓ Top {limit} tickers in table {table_name}:")
        for i, (ticker, count) in enumerate(results, 1):
            print(f"  {i:2d}. {ticker:6s} - {count:,} records")
        
        cursor.close()
        return [row[0] for row in results]
        
    except Exception as e:
        print(f"✗ Error listing tickers: {e}")
        return []

# ============================================================================
# Analysis Functions
# ============================================================================

def analyze_with_curved_supertrend(df, radius_strength=0.15, atr_length=14, 
                                   atr_mult=2.0, smoothness=5):
    """
    Analyze stock data with Curved Radius Supertrend
    
    Parameters:
    -----------
    df : pd.DataFrame
        Stock data with columns: open, high, low, close
    radius_strength : float
        Radius strength parameter (0.01-0.3)
    atr_length : int
        ATR calculation period
    atr_mult : float
        ATR multiplier
    smoothness : int
        Smoothing factor
    """
    print("\n" + "=" * 80)
    print("CURVED RADIUS SUPERTREND ANALYSIS")
    print("=" * 80)
    
    # Create indicator
    indicator = CurvedRadiusSupertrend(
        atr_length=atr_length,
        atr_mult=atr_mult,
        radius_strength=radius_strength,
        smoothness=smoothness
    )
    
    # Calculate
    result = indicator.calculate(
        df['high'].values,
        df['low'].values,
        df['close'].values
    )
    
    # Add results to dataframe
    df['curved_band'] = result['curved_band']
    df['direction'] = result['direction']
    df['buy_signal'] = result['buy_signals']
    df['sell_signal'] = result['sell_signals']
    df['outer_band'] = result['outer_band']
    
    # Print statistics
    print(f"\nIndicator Parameters:")
    print(f"  ATR Length: {atr_length}")
    print(f"  ATR Multiplier: {atr_mult}")
    print(f"  Radius Strength: {radius_strength}")
    print(f"  Smoothness: {smoothness}")
    
    print(f"\nSignal Statistics:")
    print(f"  Total Buy Signals: {result['buy_signals'].sum()}")
    print(f"  Total Sell Signals: {result['sell_signals'].sum()}")
    
    # Calculate trend statistics
    uptrend_bars = (result['direction'] == 1).sum()
    downtrend_bars = (result['direction'] == -1).sum()
    print(f"\nTrend Statistics:")
    print(f"  Uptrend bars: {uptrend_bars} ({uptrend_bars/len(df)*100:.1f}%)")
    print(f"  Downtrend bars: {downtrend_bars} ({downtrend_bars/len(df)*100:.1f}%)")
    
    # Show signals
    if result['buy_signals'].sum() > 0:
        print(f"\nBuy Signals:")
        buy_indices = np.where(result['buy_signals'])[0]
        for idx in buy_indices[:5]:  # Show first 5
            if idx < len(df):
                print(f"  {df.iloc[idx]['datetime']}: ${df.iloc[idx]['close']:.2f}")
        if len(buy_indices) > 5:
            print(f"  ... and {len(buy_indices) - 5} more")
    
    if result['sell_signals'].sum() > 0:
        print(f"\nSell Signals:")
        sell_indices = np.where(result['sell_signals'])[0]
        for idx in sell_indices[:5]:  # Show first 5
            if idx < len(df):
                print(f"  {df.iloc[idx]['datetime']}: ${df.iloc[idx]['close']:.2f}")
        if len(sell_indices) > 5:
            print(f"  ... and {len(sell_indices) - 5} more")
    
    return df, result

def plot_analysis(df, ticker, table_name):
    """Plot the analysis results"""
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 12), sharex=True)
    
    # Plot 1: Price and Curved Band
    ax1.plot(df.index, df['close'], label='Close', color='black', linewidth=1.5, alpha=0.8)
    ax1.fill_between(df.index, df['low'], df['high'], alpha=0.1, color='gray', label='High-Low Range')
    
    # Color the curved band based on direction
    for i in range(1, len(df)):
        color = 'green' if df.iloc[i]['direction'] == 1 else 'red'
        ax1.plot([i-1, i], 
                [df.iloc[i-1]['curved_band'], df.iloc[i]['curved_band']], 
                color=color, linewidth=3, alpha=0.8)
    
    # Plot outer band
    ax1.plot(df.index, df['outer_band'], label='Outer Band', 
            color='blue', linewidth=1, alpha=0.3, linestyle='--')
    
    # Plot signals
    buy_idx = df[df['buy_signal']].index
    sell_idx = df[df['sell_signal']].index
    
    ax1.scatter(buy_idx, df.loc[buy_idx, 'close'], marker='^', color='green', 
               s=150, label='Buy Signal', zorder=5, edgecolors='darkgreen', linewidths=2)
    ax1.scatter(sell_idx, df.loc[sell_idx, 'close'], marker='v', color='red', 
               s=150, label='Sell Signal', zorder=5, edgecolors='darkred', linewidths=2)
    
    ax1.set_ylabel('Price ($)', fontsize=12)
    ax1.set_title(f'Curved Radius Supertrend - {ticker} ({table_name})', fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Volume
    colors = ['green' if df.iloc[i]['direction'] == 1 else 'red' for i in range(len(df))]
    ax2.bar(df.index, df['volume'], color=colors, alpha=0.5)
    ax2.set_ylabel('Volume', fontsize=12)
    ax2.set_title('Volume (colored by trend)', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Direction
    ax3.fill_between(df.index, 0, df['direction'], 
                     where=(df['direction'] > 0), color='green', alpha=0.4, label='Uptrend')
    ax3.fill_between(df.index, 0, df['direction'], 
                     where=(df['direction'] < 0), color='red', alpha=0.4, label='Downtrend')
    ax3.set_ylabel('Trend Direction', fontsize=12)
    ax3.set_xlabel('Bar Index', fontsize=12)
    ax3.set_title('Trend Direction', fontsize=12)
    ax3.legend(loc='best', fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    filename = f'curved_supertrend_{ticker}_{table_name}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"\n✓ Chart saved to: {filename}")
    plt.show()

# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("CURVED RADIUS SUPERTREND - REAL DATA TEST")
    print("=" * 80)
    
    # Connect to database
    connection = connect_to_nas()
    
    if connection:
        try:
            # List available tickers
            tickers = list_available_tickers(connection, table_name='200309', limit=20)
            
            # Fetch data for a ticker (using first available ticker)
            if tickers:
                ticker = tickers[0]  # Use the ticker with most data
                df = get_stock_data(connection, table_name='200309', ticker=ticker, limit=500)
                
                if df is not None and len(df) > 50:
                    # Analyze with Curved Radius Supertrend
                    # Using 1-minute data, so use scalping parameters (0.08-0.12)
                    df_analyzed, result = analyze_with_curved_supertrend(
                        df, 
                        radius_strength=0.10,  # Scalping setting for 1-min data
                        atr_length=14,
                        atr_mult=2.0,
                        smoothness=5
                    )
                    
                    # Plot results
                    plot_analysis(df_analyzed, ticker, '200309')
                else:
                    print("✗ Insufficient data for analysis")
            else:
                print("✗ No tickers found")
                
        finally:
            connection.close()
            print("\n✓ Database connection closed")
    
    print("\n" + "=" * 80)
    print("TEST COMPLETED")
    print("=" * 80)

