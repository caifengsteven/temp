"""
Exact Replication of Curved Radius Supertrend Pine Script
Line-by-line translation from Pine Script to Python
"""

import pandas as pd
import numpy as np
import pymysql
import matplotlib.pyplot as plt
from datetime import datetime

# ============================================================================
# Exact Pine Script Replication
# ============================================================================

class CurvedRadiusSupertrendExact:
    """
    Exact replication of the Pine Script indicator
    //@version=6
    indicator("Curved Radius Supertrend [BOSWaves]", overlay=true)
    """
    
    def __init__(self, atr_length=14, atr_mult=2.0, radius_strength=0.2, smoothness=5):
        self.atr_length = atr_length
        self.atr_mult = atr_mult
        self.radius_strength = radius_strength
        self.smoothness = smoothness
    
    def calculate(self, high, low, close):
        """
        Exact translation of Pine Script logic
        """
        # Convert to pandas Series
        high = pd.Series(high, dtype=float).reset_index(drop=True)
        low = pd.Series(low, dtype=float).reset_index(drop=True)
        close = pd.Series(close, dtype=float).reset_index(drop=True)
        
        n = len(close)
        
        # ============================================================================
        # Supertrend Calculation (from Pine Script)
        # ============================================================================
        
        # atr = ta.atr(atrLength)
        atr = self.calculate_atr(high, low, close)
        
        # src = hl2
        src = (high + low) / 2
        
        # upperBand = src + (atrMult * atr)
        # lowerBand = src - (atrMult * atr)
        upperBand = src + (self.atr_mult * atr)
        lowerBand = src - (self.atr_mult * atr)
        
        # var float supertrend = na
        # var int direction = 1
        supertrend = np.full(n, np.nan)
        direction = np.ones(n, dtype=int)
        
        # Variables for curve calculation
        anchorPrice = np.nan
        anchorBar = 0
        velocity = 0.0
        barCount = 0
        
        # Initialize
        # if na(supertrend)
        #     supertrend := lowerBand
        #     direction := 1
        supertrend[0] = lowerBand.iloc[0]
        direction[0] = 1
        
        # Main loop
        for i in range(1, n):
            prevSupertrend = supertrend[i-1]
            prevDirection = direction[i-1]
            
            # Standard supertrend logic
            # if direction == 1
            #     supertrend := close < prevSupertrend ? upperBand : math.max(lowerBand, prevSupertrend)
            # else
            #     supertrend := close > prevSupertrend ? lowerBand : math.min(upperBand, prevSupertrend)
            
            if prevDirection == 1:
                if close.iloc[i] < prevSupertrend:
                    supertrend[i] = upperBand.iloc[i]
                else:
                    supertrend[i] = max(lowerBand.iloc[i], prevSupertrend)
            else:
                if close.iloc[i] > prevSupertrend:
                    supertrend[i] = lowerBand.iloc[i]
                else:
                    supertrend[i] = min(upperBand.iloc[i], prevSupertrend)
            
            # int prevDirection = direction
            # if close < supertrend
            #     direction := -1
            # if close > supertrend
            #     direction := 1
            
            if close.iloc[i] < supertrend[i]:
                direction[i] = -1
            elif close.iloc[i] > supertrend[i]:
                direction[i] = 1
            else:
                direction[i] = prevDirection
        
        # ============================================================================
        # Curved Radius Implementation (from Pine Script)
        # ============================================================================
        
        # Reset variables for curve calculation
        anchorPrice = np.nan
        anchorBar = 0
        velocity = 0.0
        barCount = 0
        
        curvedSupertrend = supertrend.copy()
        
        for i in range(1, n):
            prevDirection = direction[i-1]
            
            # Detect trend change - set new anchor
            # bool trendChanged = direction != prevDirection
            trendChanged = (direction[i] != prevDirection)
            
            # if trendChanged
            #     anchorPrice := supertrend
            #     anchorBar := bar_index
            #     velocity := 0.0
            #     barCount := 0
            
            if trendChanged:
                anchorPrice = supertrend[i]
                anchorBar = i
                velocity = 0.0
                barCount = 0
            
            # Increment bar counter
            # barCount := barCount + 1
            barCount = barCount + 1
            
            # Calculate curved offset using acceleration creating a parabolic curve
            # if not na(anchorPrice)
            #     // Acceleration increases with each bar (quadratic growth)
            #     velocity := velocity + (radiusStrength * barCount)
            #     
            #     // Apply velocity in direction of trend
            #     if direction == 1
            #         // Uptrend - curve upward with acceleration
            #         supertrend := anchorPrice + velocity
            #     else
            #         // Downtrend - curve downward with acceleration  
            #         supertrend := anchorPrice - velocity
            
            if not np.isnan(anchorPrice):
                # Acceleration increases with each bar (quadratic growth)
                velocity = velocity + (self.radius_strength * barCount)
                
                # Apply velocity in direction of trend
                if direction[i] == 1:
                    # Uptrend - curve upward with acceleration
                    curvedSupertrend[i] = anchorPrice + velocity
                else:
                    # Downtrend - curve downward with acceleration
                    curvedSupertrend[i] = anchorPrice - velocity
        
        # Apply smoothing to create flowing curves
        # curvedBand = ta.sma(supertrend, smoothness)
        curvedBand = pd.Series(curvedSupertrend).rolling(window=self.smoothness, min_periods=1).mean().values
        
        # ============================================================================
        # Signals (from Pine Script)
        # ============================================================================
        
        # buySignal = trendChanged and direction == 1
        # sellSignal = trendChanged and direction == -1
        
        buySignal = np.zeros(n, dtype=bool)
        sellSignal = np.zeros(n, dtype=bool)
        
        for i in range(1, n):
            trendChanged = (direction[i] != direction[i-1])
            
            if trendChanged and direction[i] == 1:
                buySignal[i] = True
            elif trendChanged and direction[i] == -1:
                sellSignal[i] = True
        
        # Calculate outer band
        # outerBand = direction == 1 ? curvedBand + atr : curvedBand - atr
        outerBand = np.where(direction == 1, curvedBand + atr.values, curvedBand - atr.values)
        
        return {
            'curved_band': curvedBand,
            'direction': direction,
            'buy_signals': buySignal,
            'sell_signals': sellSignal,
            'outer_band': outerBand,
            'atr': atr.values
        }
    
    def calculate_atr(self, high, low, close):
        """Calculate ATR exactly as Pine Script ta.atr()"""
        df = pd.DataFrame({'high': high, 'low': low, 'close': close})
        
        # True Range
        df['h-l'] = df['high'] - df['low']
        df['h-pc'] = abs(df['high'] - df['close'].shift(1))
        df['l-pc'] = abs(df['low'] - df['close'].shift(1))
        df['tr'] = df[['h-l', 'h-pc', 'l-pc']].max(axis=1)
        
        # ATR = RMA (Running Moving Average) in Pine Script
        # In Pine Script, ta.atr uses RMA, not SMA
        # RMA is exponential moving average with alpha = 1/length
        atr = df['tr'].ewm(alpha=1/self.atr_length, adjust=False).mean()
        
        return atr

# ============================================================================
# Backtesting Engine
# ============================================================================

class BacktestEngine:
    """Simple backtesting engine with COMPOUNDING"""

    def __init__(self, initial_capital=10000, commission=0.001):
        self.initial_capital = initial_capital
        self.commission = commission

    def run(self, df, signals):
        """Run backtest - only enter position once per signal with COMPOUNDING"""
        n = len(df)
        equity = np.full(n, self.initial_capital, dtype=float)
        position = 0  # 1 = long, -1 = short, 0 = flat
        entry_price = 0.0
        shares = 0.0
        trades = []

        for i in range(n):
            if i > 0:
                equity[i] = equity[i-1]

            # Buy signal - enter long, close short if exists
            if signals['buy_signals'][i]:
                # Close short position
                if position == -1:
                    # P&L from short position (using current equity)
                    pnl_pct = (entry_price - df.iloc[i]['close']) / entry_price
                    pnl = pnl_pct * equity[i-1]
                    commission_cost = equity[i-1] * self.commission * 2
                    net_pnl = pnl - commission_cost

                    equity[i] = equity[i-1] + net_pnl

                    trades.append({
                        'type': 'short',
                        'entry_price': entry_price,
                        'exit_price': df.iloc[i]['close'],
                        'pnl': net_pnl,
                        'pnl_pct': pnl_pct * 100
                    })

                # Enter long
                position = 1
                entry_price = df.iloc[i]['close']
                shares = equity[i] / entry_price

            # Sell signal - enter short, close long if exists
            elif signals['sell_signals'][i]:
                # Close long position
                if position == 1:
                    # P&L from long position (using current equity)
                    pnl_pct = (df.iloc[i]['close'] - entry_price) / entry_price
                    pnl = pnl_pct * equity[i-1]
                    commission_cost = equity[i-1] * self.commission * 2
                    net_pnl = pnl - commission_cost

                    equity[i] = equity[i-1] + net_pnl

                    trades.append({
                        'type': 'long',
                        'entry_price': entry_price,
                        'exit_price': df.iloc[i]['close'],
                        'pnl': net_pnl,
                        'pnl_pct': pnl_pct * 100
                    })

                # Enter short
                position = -1
                entry_price = df.iloc[i]['close']
                shares = equity[i] / entry_price

            # Update unrealized P&L
            elif position != 0:
                if position == 1:
                    unrealized_pct = (df.iloc[i]['close'] - entry_price) / entry_price
                else:
                    unrealized_pct = (entry_price - df.iloc[i]['close']) / entry_price

                unrealized = unrealized_pct * equity[i-1]
                equity[i] = equity[i-1] + unrealized
        
        # Calculate metrics
        trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
        
        if len(trades_df) > 0:
            wins = trades_df[trades_df['pnl'] > 0]
            losses = trades_df[trades_df['pnl'] < 0]
            
            metrics = {
                'initial_capital': self.initial_capital,
                'final_capital': equity[-1],
                'total_return': equity[-1] - self.initial_capital,
                'total_return_pct': (equity[-1] - self.initial_capital) / self.initial_capital * 100,
                'total_trades': len(trades_df),
                'winning_trades': len(wins),
                'losing_trades': len(losses),
                'win_rate': len(wins) / len(trades_df) * 100,
                'avg_win': wins['pnl'].mean() if len(wins) > 0 else 0,
                'avg_loss': losses['pnl'].mean() if len(losses) > 0 else 0,
                'profit_factor': abs(wins['pnl'].sum() / losses['pnl'].sum()) if len(losses) > 0 and losses['pnl'].sum() != 0 else 0,
            }
        else:
            metrics = {
                'initial_capital': self.initial_capital,
                'final_capital': equity[-1],
                'total_return': 0,
                'total_return_pct': 0,
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'profit_factor': 0,
            }
        
        return {
            'equity': equity,
            'trades': trades_df,
            'metrics': metrics
        }

# ============================================================================
# Database and Utility Functions
# ============================================================================

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

def get_stock_data(connection, table_name, ticker, limit):
    """Fetch stock data from database"""
    query = f"""
    SELECT window_start, open, high, low, close, volume
    FROM `{table_name}`
    WHERE ticker = %s
    ORDER BY window_start ASC
    LIMIT %s
    """

    df = pd.read_sql(query, connection, params=(ticker, limit))
    df['datetime'] = pd.to_datetime(df['window_start'], unit='ns')

    for col in ['open', 'high', 'low', 'close']:
        df[col] = df[col].astype(float)

    return df

def print_report(results, ticker, params):
    """Print backtest report"""
    metrics = results['metrics']

    print("\n" + "=" * 80)
    print("BACKTEST REPORT - EXACT PINE SCRIPT REPLICATION")
    print("=" * 80)

    print(f"\nSymbol: {ticker}")
    print(f"Parameters:")
    print(f"  ATR Length: {params['atr_length']}")
    print(f"  ATR Multiplier: {params['atr_mult']}")
    print(f"  Radius Strength: {params['radius_strength']}")
    print(f"  Smoothness: {params['smoothness']}")

    print(f"\n{'PERFORMANCE':^80}")
    print("-" * 80)
    print(f"Initial Capital:     ${metrics['initial_capital']:>12,.2f}")
    print(f"Final Capital:       ${metrics['final_capital']:>12,.2f}")
    print(f"Total Return:        ${metrics['total_return']:>12,.2f}")
    print(f"Return %:            {metrics['total_return_pct']:>12.2f}%")

    print(f"\n{'TRADES':^80}")
    print("-" * 80)
    print(f"Total Trades:        {metrics['total_trades']:>12}")
    print(f"Winning Trades:      {metrics['winning_trades']:>12}")
    print(f"Losing Trades:       {metrics['losing_trades']:>12}")
    print(f"Win Rate:            {metrics['win_rate']:>12.2f}%")
    print(f"Average Win:         ${metrics['avg_win']:>12,.2f}")
    print(f"Average Loss:        ${metrics['avg_loss']:>12,.2f}")
    print(f"Profit Factor:       {metrics['profit_factor']:>12.2f}")

    # Show sample trades
    if len(results['trades']) > 0:
        print(f"\n{'SAMPLE TRADES (First 10)':^80}")
        print("-" * 80)
        for i, trade in results['trades'].head(10).iterrows():
            print(f"{trade['type']:<6} Entry: ${trade['entry_price']:.2f}, "
                  f"Exit: ${trade['exit_price']:.2f}, "
                  f"P&L: ${trade['pnl']:.2f} ({trade['pnl_pct']:.2f}%)")

        if len(results['trades']) > 10:
            print(f"... and {len(results['trades']) - 10} more trades")

    print("\n" + "=" * 80)

def plot_results(df, signals, results, ticker, params):
    """Plot backtest results"""
    fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)

    # Plot 1: Price and Curved Band
    ax1 = axes[0]
    ax1.plot(df.index, df['close'], label='Close', color='black', linewidth=1, alpha=0.7)

    # Color curved band by direction
    curved_band = signals['curved_band']
    direction = signals['direction']

    for i in range(1, len(curved_band)):
        color = 'green' if direction[i] == 1 else 'red'
        ax1.plot([i-1, i], [curved_band[i-1], curved_band[i]],
                color=color, linewidth=2.5, alpha=0.8)

    # Plot signals
    buy_idx = np.where(signals['buy_signals'])[0]
    sell_idx = np.where(signals['sell_signals'])[0]

    ax1.scatter(buy_idx, df.iloc[buy_idx]['close'], marker='^',
               color='green', s=150, label='Buy', zorder=5, edgecolors='darkgreen', linewidths=2)
    ax1.scatter(sell_idx, df.iloc[sell_idx]['close'], marker='v',
               color='red', s=150, label='Sell', zorder=5, edgecolors='darkred', linewidths=2)

    ax1.set_ylabel('Price ($)', fontsize=12)
    ax1.set_title(f'Curved Radius Supertrend - {ticker} (RS={params["radius_strength"]})',
                 fontsize=14, fontweight='bold')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Equity Curve
    ax2 = axes[1]
    equity = results['equity']
    ax2.plot(equity, label='Equity', color='blue', linewidth=2)
    ax2.axhline(y=results['metrics']['initial_capital'], color='gray',
               linestyle='--', alpha=0.5, label='Initial Capital')
    ax2.fill_between(range(len(equity)), results['metrics']['initial_capital'], equity,
                     where=(equity >= results['metrics']['initial_capital']),
                     color='green', alpha=0.2)
    ax2.fill_between(range(len(equity)), results['metrics']['initial_capital'], equity,
                     where=(equity < results['metrics']['initial_capital']),
                     color='red', alpha=0.2)
    ax2.set_ylabel('Equity ($)', fontsize=12)
    ax2.set_title('Equity Curve', fontsize=12, fontweight='bold')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)

    # Plot 3: Trade P&L
    ax3 = axes[2]
    if len(results['trades']) > 0:
        pnl_values = results['trades']['pnl'].values
        colors = ['green' if x > 0 else 'red' for x in pnl_values]
        ax3.bar(range(len(pnl_values)), pnl_values, color=colors, alpha=0.6)
        ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax3.set_ylabel('P&L ($)', fontsize=12)
        ax3.set_xlabel('Trade Number', fontsize=12)
        ax3.set_title('Trade P&L', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    filename = f'exact_pine_{ticker}_RS{params["radius_strength"]}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"\n✓ Chart saved: {filename}")
    plt.show()

# ============================================================================
# Main Testing Function
# ============================================================================

def run_backtest(ticker='QQQ', table_name='200309', limit=2000,
                atr_length=14, atr_mult=2.0, radius_strength=0.2, smoothness=5):
    """Run complete backtest"""

    print("\n" + "=" * 80)
    print("CURVED RADIUS SUPERTREND - EXACT PINE SCRIPT REPLICATION")
    print("=" * 80)

    # Fetch data
    print(f"\n[1/4] Fetching data for {ticker}...")
    connection = connect_to_nas()
    df = get_stock_data(connection, table_name, ticker, limit)
    connection.close()

    print(f"✓ Fetched {len(df)} bars")
    print(f"  Date range: {df['datetime'].min()} to {df['datetime'].max()}")
    print(f"  Price range: ${df['low'].min():.2f} - ${df['high'].max():.2f}")

    # Calculate indicator
    print(f"\n[2/4] Calculating indicator...")
    indicator = CurvedRadiusSupertrendExact(
        atr_length=atr_length,
        atr_mult=atr_mult,
        radius_strength=radius_strength,
        smoothness=smoothness
    )

    signals = indicator.calculate(df['high'].values, df['low'].values, df['close'].values)

    print(f"✓ Indicator calculated")
    print(f"  Buy signals: {signals['buy_signals'].sum()}")
    print(f"  Sell signals: {signals['sell_signals'].sum()}")

    # Run backtest
    print(f"\n[3/4] Running backtest...")
    backtester = BacktestEngine(initial_capital=10000, commission=0.001)
    results = backtester.run(df, signals)

    print(f"✓ Backtest complete")
    print(f"  Total trades: {len(results['trades'])}")

    # Print report
    print(f"\n[4/4] Generating report...")
    params = {
        'atr_length': atr_length,
        'atr_mult': atr_mult,
        'radius_strength': radius_strength,
        'smoothness': smoothness
    }

    print_report(results, ticker, params)
    plot_results(df, signals, results, ticker, params)

    return results

# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    # Test with different radius strengths as recommended in the Pine Script
    print("\n" + "=" * 80)
    print("TESTING DIFFERENT RADIUS STRENGTH VALUES")
    print("=" * 80)

    # For 1-minute data, Pine Script recommends 0.08-0.12 for scalping
    test_params = [
        {'radius_strength': 0.08, 'desc': '1-5min (Scalping) - Lower bound'},
        {'radius_strength': 0.10, 'desc': '1-5min (Scalping) - Mid'},
        {'radius_strength': 0.12, 'desc': '1-5min (Scalping) - Upper bound'},
        {'radius_strength': 0.15, 'desc': '15min (Intraday)'},
        {'radius_strength': 0.20, 'desc': 'Default'},
    ]

    all_results = []

    for params in test_params:
        print(f"\n\n{'='*80}")
        print(f"Testing: {params['desc']} (Radius Strength = {params['radius_strength']})")
        print(f"{'='*80}")

        results = run_backtest(
            ticker='QQQ',
            table_name='200309',
            limit=2000,
            atr_length=14,
            atr_mult=2.0,
            radius_strength=params['radius_strength'],
            smoothness=5
        )

        all_results.append({
            'radius_strength': params['radius_strength'],
            'description': params['desc'],
            **results['metrics']
        })

    # Summary comparison
    print("\n\n" + "=" * 80)
    print("PARAMETER COMPARISON SUMMARY")
    print("=" * 80)

    summary_df = pd.DataFrame(all_results)
    print(f"\n{'RS':<6} {'Description':<30} {'Return %':<12} {'Trades':<10} {'Win Rate':<12}")
    print("-" * 80)

    for _, row in summary_df.iterrows():
        print(f"{row['radius_strength']:<6.2f} {row['description']:<30} "
              f"{row['total_return_pct']:<12.2f} {row['total_trades']:<10} "
              f"{row['win_rate']:<12.1f}%")

    print("\n" + "=" * 80)
    print("TESTING COMPLETE")
    print("=" * 80)


