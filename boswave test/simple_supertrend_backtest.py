"""
Simple Traditional Supertrend Backtest
Using standard Supertrend logic without the curved modifications
to establish a baseline for comparison
"""

import pandas as pd
import numpy as np
import pymysql
import matplotlib.pyplot as plt
from datetime import datetime

# ============================================================================
# Traditional Supertrend Indicator
# ============================================================================

class TraditionalSupertrend:
    """Traditional Supertrend indicator"""
    
    def __init__(self, atr_length=14, atr_mult=3.0):
        self.atr_length = atr_length
        self.atr_mult = atr_mult
    
    def calculate_atr(self, high, low, close):
        """Calculate Average True Range"""
        df = pd.DataFrame({'high': high, 'low': low, 'close': close})
        
        df['h-l'] = df['high'] - df['low']
        df['h-pc'] = abs(df['high'] - df['close'].shift(1))
        df['l-pc'] = abs(df['low'] - df['close'].shift(1))
        df['tr'] = df[['h-l', 'h-pc', 'l-pc']].max(axis=1)
        
        atr = df['tr'].rolling(window=self.atr_length).mean()
        
        return atr
    
    def calculate(self, high, low, close):
        """Calculate traditional Supertrend"""
        high = pd.Series(high) if not isinstance(high, pd.Series) else high
        low = pd.Series(low) if not isinstance(low, pd.Series) else low
        close = pd.Series(close) if not isinstance(close, pd.Series) else close
        
        n = len(close)
        atr = self.calculate_atr(high, low, close)
        hl2 = (high + low) / 2
        
        # Initialize
        supertrend = np.full(n, np.nan)
        direction = np.ones(n, dtype=int)
        
        upper_band = hl2 + (self.atr_mult * atr)
        lower_band = hl2 - (self.atr_mult * atr)
        
        # Calculate Supertrend
        for i in range(n):
            if i == 0:
                supertrend[i] = lower_band.iloc[i]
                direction[i] = 1
                continue
            
            # Update bands with trailing logic
            if lower_band.iloc[i] > supertrend[i-1] or close.iloc[i-1] < supertrend[i-1]:
                final_lower = lower_band.iloc[i]
            else:
                final_lower = supertrend[i-1]
            
            if upper_band.iloc[i] < supertrend[i-1] or close.iloc[i-1] > supertrend[i-1]:
                final_upper = upper_band.iloc[i]
            else:
                final_upper = supertrend[i-1]
            
            # Determine trend
            if close.iloc[i] <= final_upper:
                supertrend[i] = final_upper
                direction[i] = -1
            else:
                supertrend[i] = final_lower
                direction[i] = 1
        
        # Generate signals (only once per trend change)
        buy_signals = np.zeros(n, dtype=bool)
        sell_signals = np.zeros(n, dtype=bool)
        
        for i in range(1, n):
            if direction[i] == 1 and direction[i-1] == -1:
                buy_signals[i] = True
            elif direction[i] == -1 and direction[i-1] == 1:
                sell_signals[i] = True
        
        return {
            'supertrend': supertrend,
            'direction': direction,
            'buy_signals': buy_signals,
            'sell_signals': sell_signals,
            'atr': atr.values
        }

# ============================================================================
# Backtesting Engine (Simplified)
# ============================================================================

class SimpleBacktester:
    """Simple backtesting engine"""
    
    def __init__(self, initial_capital=10000, commission=0.001):
        self.initial_capital = initial_capital
        self.commission = commission
    
    def run(self, df, signals):
        """Run backtest"""
        n = len(df)
        equity = np.full(n, self.initial_capital)
        position = 0  # 1 = long, -1 = short, 0 = flat
        entry_price = 0
        trades = []
        
        for i in range(n):
            if i > 0:
                equity[i] = equity[i-1]
            
            # Buy signal
            if signals['buy_signals'][i]:
                # Close short if exists
                if position == -1:
                    pnl = (entry_price - df.iloc[i]['close']) / entry_price * equity[i-1]
                    commission_cost = equity[i-1] * self.commission * 2
                    equity[i] = equity[i-1] + pnl - commission_cost
                    
                    trades.append({
                        'type': 'short',
                        'entry_price': entry_price,
                        'exit_price': df.iloc[i]['close'],
                        'pnl': pnl - commission_cost
                    })
                
                # Open long
                position = 1
                entry_price = df.iloc[i]['close']
            
            # Sell signal
            elif signals['sell_signals'][i]:
                # Close long if exists
                if position == 1:
                    pnl = (df.iloc[i]['close'] - entry_price) / entry_price * equity[i-1]
                    commission_cost = equity[i-1] * self.commission * 2
                    equity[i] = equity[i-1] + pnl - commission_cost
                    
                    trades.append({
                        'type': 'long',
                        'entry_price': entry_price,
                        'exit_price': df.iloc[i]['close'],
                        'pnl': pnl - commission_cost
                    })
                
                # Open short
                position = -1
                entry_price = df.iloc[i]['close']
            
            # Update equity with unrealized P&L
            elif position != 0:
                if position == 1:
                    unrealized = (df.iloc[i]['close'] - entry_price) / entry_price * equity[i-1]
                else:
                    unrealized = (entry_price - df.iloc[i]['close']) / entry_price * equity[i-1]
                equity[i] = equity[i-1] + unrealized
        
        # Calculate metrics
        trades_df = pd.DataFrame(trades)
        
        if len(trades_df) > 0:
            wins = trades_df[trades_df['pnl'] > 0]
            losses = trades_df[trades_df['pnl'] < 0]
            
            metrics = {
                'total_return': equity[-1] - self.initial_capital,
                'total_return_pct': (equity[-1] - self.initial_capital) / self.initial_capital * 100,
                'total_trades': len(trades_df),
                'winning_trades': len(wins),
                'losing_trades': len(losses),
                'win_rate': len(wins) / len(trades_df) * 100 if len(trades_df) > 0 else 0,
                'avg_win': wins['pnl'].mean() if len(wins) > 0 else 0,
                'avg_loss': losses['pnl'].mean() if len(losses) > 0 else 0,
            }
        else:
            metrics = {
                'total_return': 0,
                'total_return_pct': 0,
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0,
                'avg_win': 0,
                'avg_loss': 0,
            }
        
        return {
            'equity': equity,
            'trades': trades_df,
            'metrics': metrics
        }

# ============================================================================
# Database and Testing
# ============================================================================

def connect_to_nas():
    """Connect to NAS"""
    config = {
        'host': '192.168.50.230',
        'port': 3306,
        'user': 'root',
        'password': '352471Cf!1',
        'database': 'us_stock_sip_min_aggs'
    }
    
    return pymysql.connect(**config)

def get_stock_data(connection, table_name, ticker, limit):
    """Fetch stock data"""
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

def run_simple_backtest(ticker='QQQ', table_name='200309', limit=2000, 
                       atr_length=14, atr_mult=3.0):
    """Run simple backtest"""
    print("\n" + "=" * 80)
    print("TRADITIONAL SUPERTREND BACKTEST")
    print("=" * 80)
    
    # Fetch data
    print(f"\nFetching {ticker} data...")
    connection = connect_to_nas()
    df = get_stock_data(connection, table_name, ticker, limit)
    connection.close()
    
    print(f"✓ Fetched {len(df)} bars")
    print(f"  Date range: {df['datetime'].min()} to {df['datetime'].max()}")
    
    # Calculate indicator
    print(f"\nCalculating Supertrend (ATR Length={atr_length}, Multiplier={atr_mult})...")
    indicator = TraditionalSupertrend(atr_length=atr_length, atr_mult=atr_mult)
    signals = indicator.calculate(df['high'].values, df['low'].values, df['close'].values)
    
    print(f"✓ Buy signals: {signals['buy_signals'].sum()}")
    print(f"✓ Sell signals: {signals['sell_signals'].sum()}")
    
    # Run backtest
    print(f"\nRunning backtest...")
    backtester = SimpleBacktester(initial_capital=10000, commission=0.001)
    results = backtester.run(df, signals)
    
    # Print results
    metrics = results['metrics']
    print("\n" + "=" * 80)
    print("BACKTEST RESULTS")
    print("=" * 80)
    print(f"\nInitial Capital:     ${10000:,.2f}")
    print(f"Final Capital:       ${results['equity'][-1]:,.2f}")
    print(f"Total Return:        ${metrics['total_return']:,.2f} ({metrics['total_return_pct']:.2f}%)")
    print(f"\nTotal Trades:        {metrics['total_trades']}")
    print(f"Winning Trades:      {metrics['winning_trades']} ({metrics['win_rate']:.1f}%)")
    print(f"Losing Trades:       {metrics['losing_trades']}")
    print(f"Average Win:         ${metrics['avg_win']:,.2f}")
    print(f"Average Loss:        ${metrics['avg_loss']:,.2f}")
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    
    # Price and signals
    ax1.plot(df.index, df['close'], label='Close', color='black', linewidth=1)
    ax1.plot(df.index, signals['supertrend'], label='Supertrend', color='blue', linewidth=2)
    
    buy_idx = np.where(signals['buy_signals'])[0]
    sell_idx = np.where(signals['sell_signals'])[0]
    
    ax1.scatter(buy_idx, df.iloc[buy_idx]['close'], marker='^', color='green', s=100, label='Buy', zorder=5)
    ax1.scatter(sell_idx, df.iloc[sell_idx]['close'], marker='v', color='red', s=100, label='Sell', zorder=5)
    
    ax1.set_ylabel('Price ($)')
    ax1.set_title(f'Traditional Supertrend - {ticker}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Equity curve
    ax2.plot(results['equity'], label='Equity', color='blue', linewidth=2)
    ax2.axhline(y=10000, color='gray', linestyle='--', alpha=0.5)
    ax2.set_ylabel('Equity ($)')
    ax2.set_xlabel('Bar Index')
    ax2.set_title('Equity Curve')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'simple_backtest_{ticker}_{table_name}.png', dpi=150)
    print(f"\n✓ Chart saved to: simple_backtest_{ticker}_{table_name}.png")
    plt.show()
    
    return results

if __name__ == "__main__":
    # Test with different ATR multipliers
    for atr_mult in [2.0, 2.5, 3.0, 3.5, 4.0]:
        print(f"\n\n{'='*80}")
        print(f"Testing ATR Multiplier = {atr_mult}")
        print(f"{'='*80}")
        results = run_simple_backtest(ticker='QQQ', table_name='200309', limit=2000, 
                                     atr_length=14, atr_mult=atr_mult)
        print(f"\nResult: {results['metrics']['total_return_pct']:.2f}% return, "
              f"{results['metrics']['total_trades']} trades, "
              f"{results['metrics']['win_rate']:.1f}% win rate")

