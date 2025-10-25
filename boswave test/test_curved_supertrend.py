"""
Test script for:
1. NAS database connection
2. Curved Radius Supertrend indicator implementation
"""

import pandas as pd
import numpy as np
import pymysql
from pymysql import Error
import matplotlib.pyplot as plt
from datetime import datetime

# ============================================================================
# Database Connection Test
# ============================================================================

def test_nas_connection():
    """Test connection to NAS MySQL database"""
    print("=" * 80)
    print("Testing NAS Database Connection")
    print("=" * 80)
    
    connection = None
    try:
        # Connection parameters
        config = {
            'host': '192.168.50.230',
            'port': 3306,
            'user': 'root',
            'password': '352471Cf!1',
            'database': 'us_stock_sip_min_aggs'
        }
        
        print(f"\nConnecting to: {config['host']}:{config['port']}")
        print(f"Database: {config['database']}")
        print(f"User: {config['user']}")

        # Establish connection
        connection = pymysql.connect(**config)
        
        if connection.open:
            db_info = connection.get_server_info()
            print(f"\n✓ Successfully connected to MySQL Server version {db_info}")
            
            cursor = connection.cursor()
            cursor.execute("SELECT DATABASE();")
            record = cursor.fetchone()
            print(f"✓ Connected to database: {record[0]}")
            
            # List tables
            cursor.execute("SHOW TABLES;")
            tables = cursor.fetchall()
            print(f"\n✓ Found {len(tables)} tables in database:")
            for i, table in enumerate(tables[:10], 1):  # Show first 10 tables
                print(f"  {i}. {table[0]}")
            if len(tables) > 10:
                print(f"  ... and {len(tables) - 10} more tables")
            
            # Get sample data from first table if available
            if tables:
                first_table = tables[0][0]
                print(f"\n✓ Sample data from table '{first_table}':")
                cursor.execute(f"SELECT * FROM `{first_table}` LIMIT 5;")
                columns = [desc[0] for desc in cursor.description]
                print(f"  Columns: {columns}")
                
                rows = cursor.fetchall()
                for row in rows[:3]:  # Show first 3 rows
                    print(f"  {row}")
            
            cursor.close()
            return connection, True
            
    except Error as e:
        print(f"\n✗ Error connecting to MySQL: {e}")
        return None, False
    
    finally:
        if connection and connection.open:
            connection.close()
            print("\n✓ Connection closed")

# ============================================================================
# Curved Radius Supertrend Implementation
# ============================================================================

class CurvedRadiusSupertrend:
    """
    Curved Radius Supertrend [BOSWaves] - Python Implementation
    
    This indicator upgrades the classic Supertrend by using dynamic curvature
    instead of rigid ATR bands, creating smoother trend flows.
    """
    
    def __init__(self, atr_length=14, atr_mult=2.0, radius_strength=0.2, smoothness=5):
        """
        Initialize the Curved Radius Supertrend indicator
        
        Parameters:
        -----------
        atr_length : int
            Number of bars for ATR calculation (default: 14)
        atr_mult : float
            ATR multiplier for band distance (default: 2.0)
        radius_strength : float
            Controls curve acceleration (0.01-0.3, default: 0.2)
            Recommended by timeframe:
            - 1-5min: 0.08-0.12
            - 15min: 0.12-0.15
            - 1H: 0.15-0.18
            - 4H: 0.18-0.22
            - Daily: 0.20-0.25
            - Weekly: 0.25-0.30
        smoothness : int
            Smoothing applied to curved band (default: 5)
        """
        self.atr_length = atr_length
        self.atr_mult = atr_mult
        self.radius_strength = radius_strength
        self.smoothness = smoothness
    
    def calculate_atr(self, high, low, close):
        """Calculate Average True Range"""
        df = pd.DataFrame({'high': high, 'low': low, 'close': close})
        
        # True Range calculation
        df['h-l'] = df['high'] - df['low']
        df['h-pc'] = abs(df['high'] - df['close'].shift(1))
        df['l-pc'] = abs(df['low'] - df['close'].shift(1))
        df['tr'] = df[['h-l', 'h-pc', 'l-pc']].max(axis=1)
        
        # ATR is the moving average of True Range
        atr = df['tr'].rolling(window=self.atr_length).mean()
        
        return atr
    
    def calculate(self, high, low, close):
        """
        Calculate Curved Radius Supertrend

        Parameters:
        -----------
        high, low, close : array-like
            Price data

        Returns:
        --------
        dict with keys:
            - curved_band: The curved supertrend line
            - direction: 1 for uptrend, -1 for downtrend
            - buy_signals: Boolean array for buy signals
            - sell_signals: Boolean array for sell signals
            - outer_band: Outer envelope band
        """
        # Convert to pandas Series for easier handling
        high = pd.Series(high) if not isinstance(high, pd.Series) else high
        low = pd.Series(low) if not isinstance(low, pd.Series) else low
        close = pd.Series(close) if not isinstance(close, pd.Series) else close

        n = len(close)

        # Calculate ATR
        atr = self.calculate_atr(high, low, close)

        # Calculate HL2 (median price)
        hl2 = (high + low) / 2
        
        # Initialize arrays
        supertrend = np.full(n, np.nan)
        direction = np.zeros(n, dtype=int)
        curved_band = np.full(n, np.nan)
        
        # Variables for curve calculation
        anchor_price = np.nan
        anchor_bar = 0
        velocity = 0.0
        bar_count = 0
        
        # Calculate basic supertrend bands
        upper_band = hl2 + (self.atr_mult * atr)
        lower_band = hl2 - (self.atr_mult * atr)
        
        for i in range(n):
            if i == 0 or np.isnan(atr.iloc[i]):
                supertrend[i] = lower_band.iloc[i] if not np.isnan(lower_band.iloc[i]) else np.nan
                direction[i] = 1
                continue
            
            prev_supertrend = supertrend[i-1]
            prev_direction = direction[i-1]
            
            # Standard supertrend logic
            if prev_direction == 1:
                if close.iloc[i] < prev_supertrend:
                    supertrend[i] = upper_band.iloc[i]
                else:
                    supertrend[i] = max(lower_band.iloc[i], prev_supertrend)
            else:
                if close.iloc[i] > prev_supertrend:
                    supertrend[i] = lower_band.iloc[i]
                else:
                    supertrend[i] = min(upper_band.iloc[i], prev_supertrend)
            
            # Determine direction
            if close.iloc[i] < supertrend[i]:
                direction[i] = -1
            elif close.iloc[i] > supertrend[i]:
                direction[i] = 1
            else:
                direction[i] = prev_direction
            
            # Detect trend change
            trend_changed = direction[i] != prev_direction
            
            if trend_changed:
                anchor_price = supertrend[i]
                anchor_bar = i
                velocity = 0.0
                bar_count = 0
            
            # Increment bar counter
            bar_count += 1
            
            # Calculate curved offset using acceleration (parabolic curve)
            if not np.isnan(anchor_price):
                # Acceleration increases with each bar (quadratic growth)
                velocity = velocity + (self.radius_strength * bar_count)
                
                # Apply velocity in direction of trend
                if direction[i] == 1:
                    # Uptrend - curve upward with acceleration
                    supertrend[i] = anchor_price + velocity
                else:
                    # Downtrend - curve downward with acceleration
                    supertrend[i] = anchor_price - velocity
        
        # Apply smoothing to create flowing curves
        curved_band = pd.Series(supertrend).rolling(window=self.smoothness, min_periods=1).mean().values
        
        # Calculate outer band
        outer_band = np.where(direction == 1, 
                             curved_band + atr.values, 
                             curved_band - atr.values)
        
        # Detect signals
        buy_signals = np.zeros(n, dtype=bool)
        sell_signals = np.zeros(n, dtype=bool)
        
        for i in range(1, n):
            if direction[i] == 1 and direction[i-1] == -1:
                buy_signals[i] = True
            elif direction[i] == -1 and direction[i-1] == 1:
                sell_signals[i] = True
        
        return {
            'curved_band': curved_band,
            'direction': direction,
            'buy_signals': buy_signals,
            'sell_signals': sell_signals,
            'outer_band': outer_band,
            'atr': atr.values
        }

# ============================================================================
# Testing Functions
# ============================================================================

def test_with_sample_data():
    """Test the indicator with generated sample data"""
    print("\n" + "=" * 80)
    print("Testing Curved Radius Supertrend with Sample Data")
    print("=" * 80)
    
    # Generate sample price data (simulated trend)
    np.random.seed(42)
    n = 200
    
    # Create a trending price series
    trend = np.linspace(100, 120, n)
    noise = np.random.normal(0, 2, n)
    close = trend + noise
    
    # Add some volatility
    high = close + np.abs(np.random.normal(0, 1, n))
    low = close - np.abs(np.random.normal(0, 1, n))
    
    # Create indicator
    indicator = CurvedRadiusSupertrend(
        atr_length=14,
        atr_mult=2.0,
        radius_strength=0.15,  # 1H timeframe setting
        smoothness=5
    )
    
    # Calculate
    result = indicator.calculate(high, low, close)
    
    print(f"\n✓ Calculated Curved Radius Supertrend for {n} bars")
    print(f"  Buy signals: {result['buy_signals'].sum()}")
    print(f"  Sell signals: {result['sell_signals'].sum()}")
    
    # Plot results
    plot_results(close, high, low, result)
    
    return result

def plot_results(close, high, low, result):
    """Plot the indicator results"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    
    # Plot 1: Price and Curved Band
    ax1.plot(close, label='Close', color='black', linewidth=1, alpha=0.7)
    ax1.plot(high, label='High', color='gray', linewidth=0.5, alpha=0.3)
    ax1.plot(low, label='Low', color='gray', linewidth=0.5, alpha=0.3)
    
    # Color the curved band based on direction
    curved_band = result['curved_band']
    direction = result['direction']
    
    for i in range(1, len(curved_band)):
        color = 'green' if direction[i] == 1 else 'red'
        ax1.plot([i-1, i], [curved_band[i-1], curved_band[i]], 
                color=color, linewidth=2, alpha=0.8)
    
    # Plot outer band
    ax1.plot(result['outer_band'], label='Outer Band', 
            color='blue', linewidth=1, alpha=0.3, linestyle='--')
    
    # Plot signals
    buy_idx = np.where(result['buy_signals'])[0]
    sell_idx = np.where(result['sell_signals'])[0]
    
    ax1.scatter(buy_idx, close[buy_idx], marker='^', color='green', 
               s=100, label='Buy Signal', zorder=5)
    ax1.scatter(sell_idx, close[sell_idx], marker='v', color='red', 
               s=100, label='Sell Signal', zorder=5)
    
    ax1.set_ylabel('Price')
    ax1.set_title('Curved Radius Supertrend [BOSWaves]')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Direction
    ax2.fill_between(range(len(direction)), 0, direction, 
                     where=(direction > 0), color='green', alpha=0.3, label='Uptrend')
    ax2.fill_between(range(len(direction)), 0, direction, 
                     where=(direction < 0), color='red', alpha=0.3, label='Downtrend')
    ax2.set_ylabel('Trend Direction')
    ax2.set_xlabel('Bar Index')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('curved_supertrend_test.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Chart saved to: curved_supertrend_test.png")
    plt.show()

# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("CURVED RADIUS SUPERTREND - TEST SUITE")
    print("=" * 80)
    
    # Test 1: Database Connection
    connection, success = test_nas_connection()
    
    # Test 2: Indicator with sample data
    test_with_sample_data()
    
    # Close database connection if open
    if connection and connection.open:
        connection.close()
        print("\n✓ Database connection closed")
    
    print("\n" + "=" * 80)
    print("ALL TESTS COMPLETED")
    print("=" * 80)

