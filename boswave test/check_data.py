"""
Check QQQ data to see what's wrong with candlesticks
"""

import pandas as pd
import pymysql

def connect_to_nas():
    """Connect to NAS database"""
    config = {
        'host': '192.168.50.230',
        'port': 3306,
        'user': 'root',
        'password': '352471Cf!1',
        'database': 'us_stock_sip_min_aggs'
    }
    return pymysql.connect(**config)

def check_qqq_data():
    """Check QQQ data structure"""
    
    print("\n" + "="*80)
    print("CHECKING QQQ DATA")
    print("="*80 + "\n")
    
    connection = connect_to_nas()
    
    # Get sample data
    query = """
    SELECT window_start, open, high, low, close, volume
    FROM `200309`
    WHERE ticker = 'QQQ'
    ORDER BY window_start ASC
    LIMIT 20
    """
    
    df = pd.read_sql(query, connection)
    connection.close()
    
    # Convert timestamp
    df['datetime'] = pd.to_datetime(df['window_start'], unit='ns')
    
    print("First 20 bars of QQQ data:")
    print("="*80)
    print(f"{'#':<4} {'DateTime':<20} {'Open':<12} {'High':<12} {'Low':<12} {'Close':<12} {'Volume':<10}")
    print("-"*80)
    
    for i, row in df.iterrows():
        print(f"{i:<4} {str(row['datetime']):<20} "
              f"{row['open']:<12} {row['high']:<12} {row['low']:<12} "
              f"{row['close']:<12} {row['volume']:<10}")
    
    print("\n" + "="*80)
    print("DATA ANALYSIS")
    print("="*80 + "\n")
    
    # Convert to float
    for col in ['open', 'high', 'low', 'close']:
        df[col] = df[col].astype(float)
    
    print(f"Data types:")
    print(df.dtypes)
    
    print(f"\nPrice statistics:")
    print(f"  Open:  min={df['open'].min():.2f}, max={df['open'].max():.2f}")
    print(f"  High:  min={df['high'].min():.2f}, max={df['high'].max():.2f}")
    print(f"  Low:   min={df['low'].min():.2f}, max={df['low'].max():.2f}")
    print(f"  Close: min={df['close'].min():.2f}, max={df['close'].max():.2f}")
    
    print(f"\nChecking for data issues:")
    
    # Check if high >= low
    invalid_hl = df[df['high'] < df['low']]
    if len(invalid_hl) > 0:
        print(f"  ❌ Found {len(invalid_hl)} bars where High < Low!")
        print(invalid_hl)
    else:
        print(f"  ✓ All bars have High >= Low")
    
    # Check if high >= open, close
    invalid_h = df[(df['high'] < df['open']) | (df['high'] < df['close'])]
    if len(invalid_h) > 0:
        print(f"  ❌ Found {len(invalid_h)} bars where High < Open or Close!")
        print(invalid_h)
    else:
        print(f"  ✓ All bars have High >= Open and Close")
    
    # Check if low <= open, close
    invalid_l = df[(df['low'] > df['open']) | (df['low'] > df['close'])]
    if len(invalid_l) > 0:
        print(f"  ❌ Found {len(invalid_l)} bars where Low > Open or Close!")
        print(invalid_l)
    else:
        print(f"  ✓ All bars have Low <= Open and Close")
    
    # Check for zero or negative prices
    zero_prices = df[(df['open'] <= 0) | (df['high'] <= 0) | (df['low'] <= 0) | (df['close'] <= 0)]
    if len(zero_prices) > 0:
        print(f"  ❌ Found {len(zero_prices)} bars with zero or negative prices!")
        print(zero_prices)
    else:
        print(f"  ✓ All prices are positive")
    
    # Check for unrealistic price ranges
    print(f"\nPrice range analysis:")
    for i, row in df.head(10).iterrows():
        range_pct = ((row['high'] - row['low']) / row['close']) * 100
        print(f"  Bar {i}: Range = ${row['high'] - row['low']:.4f} ({range_pct:.2f}% of close)")
    
    print("\n" + "="*80 + "\n")
    
    return df

if __name__ == "__main__":
    df = check_qqq_data()

