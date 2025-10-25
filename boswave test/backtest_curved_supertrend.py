"""
Backtesting Engine for Curved Radius Supertrend
- Fixed signal generation (only once per trend change)
- Complete position tracking
- Performance metrics and reporting
"""

import pandas as pd
import numpy as np
import pymysql
import matplotlib.pyplot as plt
from datetime import datetime
from test_curved_supertrend import CurvedRadiusSupertrend

# ============================================================================
# Enhanced Indicator with Fixed Signal Logic
# ============================================================================

class CurvedRadiusSupertrendFixed(CurvedRadiusSupertrend):
    """
    Enhanced version with proper signal generation:
    - Signals only trigger ONCE when entering a new position
    - No repeated signals during the same trend
    """
    
    def calculate(self, high, low, close):
        """Calculate with fixed signal logic"""
        # Get base calculation from parent class
        result = super().calculate(high, low, close)
        
        # Fix signals: only trigger once per trend change, not continuously
        direction = result['direction']
        buy_signals = np.zeros(len(direction), dtype=bool)
        sell_signals = np.zeros(len(direction), dtype=bool)
        
        in_position = False
        position_type = None  # 'long' or 'short'
        
        for i in range(1, len(direction)):
            current_trend = 'up' if direction[i] == 1 else 'down'
            prev_trend = 'up' if direction[i-1] == 1 else 'down'
            
            # Trend changed from down to up -> BUY signal (only if not already long)
            if current_trend == 'up' and prev_trend == 'down':
                if not in_position or position_type != 'long':
                    buy_signals[i] = True
                    in_position = True
                    position_type = 'long'
            
            # Trend changed from up to down -> SELL signal (only if not already short)
            elif current_trend == 'down' and prev_trend == 'up':
                if not in_position or position_type != 'short':
                    sell_signals[i] = True
                    in_position = True
                    position_type = 'short'
        
        result['buy_signals'] = buy_signals
        result['sell_signals'] = sell_signals
        
        return result

# ============================================================================
# Backtesting Engine
# ============================================================================

class BacktestEngine:
    """
    Comprehensive backtesting engine for trading strategies
    """
    
    def __init__(self, initial_capital=10000, commission=0.001, slippage=0.0005):
        """
        Initialize backtesting engine
        
        Parameters:
        -----------
        initial_capital : float
            Starting capital in dollars
        commission : float
            Commission rate (0.001 = 0.1%)
        slippage : float
            Slippage rate (0.0005 = 0.05%)
        """
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        
    def run_backtest(self, df, signals):
        """
        Run backtest on price data with signals
        
        Parameters:
        -----------
        df : pd.DataFrame
            Price data with columns: open, high, low, close
        signals : dict
            Dictionary with 'buy_signals' and 'sell_signals' boolean arrays
        
        Returns:
        --------
        dict with backtest results
        """
        n = len(df)
        
        # Initialize tracking arrays
        position = np.zeros(n)  # 1 = long, -1 = short, 0 = no position
        entry_price = np.zeros(n)
        exit_price = np.zeros(n)
        pnl = np.zeros(n)
        equity = np.full(n, self.initial_capital)
        
        # Trade tracking
        trades = []
        current_position = 0
        current_entry_price = 0
        current_entry_bar = 0
        
        for i in range(n):
            # Copy previous equity
            if i > 0:
                equity[i] = equity[i-1]
            
            # Check for BUY signal
            if signals['buy_signals'][i]:
                # Close short position if exists
                if current_position == -1:
                    exit_px = df.iloc[i]['close'] * (1 + self.slippage)
                    trade_pnl = (current_entry_price - exit_px) * (equity[i-1] / current_entry_price)
                    commission_cost = equity[i-1] * self.commission * 2  # Entry + exit
                    net_pnl = trade_pnl - commission_cost
                    
                    equity[i] = equity[i-1] + net_pnl
                    pnl[i] = net_pnl
                    exit_price[i] = exit_px
                    
                    # Record trade
                    trades.append({
                        'type': 'short',
                        'entry_bar': current_entry_bar,
                        'exit_bar': i,
                        'entry_price': current_entry_price,
                        'exit_price': exit_px,
                        'pnl': net_pnl,
                        'return_pct': (net_pnl / equity[i-1]) * 100,
                        'bars_held': i - current_entry_bar
                    })
                
                # Open long position
                current_position = 1
                current_entry_price = df.iloc[i]['close'] * (1 + self.slippage)
                current_entry_bar = i
                entry_price[i] = current_entry_price
                position[i] = 1
            
            # Check for SELL signal
            elif signals['sell_signals'][i]:
                # Close long position if exists
                if current_position == 1:
                    exit_px = df.iloc[i]['close'] * (1 - self.slippage)
                    trade_pnl = (exit_px - current_entry_price) * (equity[i-1] / current_entry_price)
                    commission_cost = equity[i-1] * self.commission * 2  # Entry + exit
                    net_pnl = trade_pnl - commission_cost
                    
                    equity[i] = equity[i-1] + net_pnl
                    pnl[i] = net_pnl
                    exit_price[i] = exit_px
                    
                    # Record trade
                    trades.append({
                        'type': 'long',
                        'entry_bar': current_entry_bar,
                        'exit_bar': i,
                        'entry_price': current_entry_price,
                        'exit_price': exit_px,
                        'pnl': net_pnl,
                        'return_pct': (net_pnl / equity[i-1]) * 100,
                        'bars_held': i - current_entry_bar
                    })
                
                # Open short position
                current_position = -1
                current_entry_price = df.iloc[i]['close'] * (1 - self.slippage)
                current_entry_bar = i
                entry_price[i] = current_entry_price
                position[i] = -1
            
            else:
                # Maintain current position
                position[i] = current_position
                
                # Update unrealized P&L for open position
                if current_position == 1:  # Long
                    unrealized_pnl = (df.iloc[i]['close'] - current_entry_price) * (equity[i-1] / current_entry_price)
                    equity[i] = equity[i-1] + unrealized_pnl
                elif current_position == -1:  # Short
                    unrealized_pnl = (current_entry_price - df.iloc[i]['close']) * (equity[i-1] / current_entry_price)
                    equity[i] = equity[i-1] + unrealized_pnl
        
        # Close any remaining position at the end
        if current_position != 0:
            exit_px = df.iloc[-1]['close']
            if current_position == 1:
                trade_pnl = (exit_px - current_entry_price) * (equity[-2] / current_entry_price)
            else:
                trade_pnl = (current_entry_price - exit_px) * (equity[-2] / current_entry_price)
            
            commission_cost = equity[-2] * self.commission * 2
            net_pnl = trade_pnl - commission_cost
            equity[-1] = equity[-2] + net_pnl
            
            trades.append({
                'type': 'long' if current_position == 1 else 'short',
                'entry_bar': current_entry_bar,
                'exit_bar': n-1,
                'entry_price': current_entry_price,
                'exit_price': exit_px,
                'pnl': net_pnl,
                'return_pct': (net_pnl / equity[-2]) * 100,
                'bars_held': n - 1 - current_entry_bar
            })
        
        # Calculate performance metrics
        trades_df = pd.DataFrame(trades)
        metrics = self.calculate_metrics(equity, trades_df)
        
        return {
            'equity': equity,
            'position': position,
            'pnl': pnl,
            'trades': trades_df,
            'metrics': metrics
        }
    
    def calculate_metrics(self, equity, trades_df):
        """Calculate performance metrics"""
        if len(trades_df) == 0:
            return {
                'total_return': 0,
                'total_return_pct': 0,
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'profit_factor': 0,
                'max_drawdown': 0,
                'max_drawdown_pct': 0,
                'sharpe_ratio': 0
            }
        
        # Basic metrics
        total_return = equity[-1] - self.initial_capital
        total_return_pct = (total_return / self.initial_capital) * 100
        
        # Trade statistics
        winning_trades = trades_df[trades_df['pnl'] > 0]
        losing_trades = trades_df[trades_df['pnl'] < 0]
        
        total_trades = len(trades_df)
        num_wins = len(winning_trades)
        num_losses = len(losing_trades)
        win_rate = (num_wins / total_trades * 100) if total_trades > 0 else 0
        
        avg_win = winning_trades['pnl'].mean() if num_wins > 0 else 0
        avg_loss = losing_trades['pnl'].mean() if num_losses > 0 else 0
        
        total_wins = winning_trades['pnl'].sum() if num_wins > 0 else 0
        total_losses = abs(losing_trades['pnl'].sum()) if num_losses > 0 else 0
        profit_factor = (total_wins / total_losses) if total_losses > 0 else float('inf')
        
        # Drawdown
        peak = np.maximum.accumulate(equity)
        drawdown = peak - equity
        max_drawdown = drawdown.max()
        max_drawdown_pct = (max_drawdown / peak[drawdown.argmax()] * 100) if peak[drawdown.argmax()] > 0 else 0
        
        # Sharpe ratio (annualized, assuming 252 trading days)
        returns = np.diff(equity) / equity[:-1]
        sharpe_ratio = (returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0
        
        return {
            'total_return': total_return,
            'total_return_pct': total_return_pct,
            'total_trades': total_trades,
            'winning_trades': num_wins,
            'losing_trades': num_losses,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'max_drawdown': max_drawdown,
            'max_drawdown_pct': max_drawdown_pct,
            'sharpe_ratio': sharpe_ratio,
            'avg_bars_held': trades_df['bars_held'].mean()
        }

# ============================================================================
# Database and Data Functions
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
        return connection
    except Exception as e:
        print(f"✗ Error connecting to database: {e}")
        return None

def get_stock_data(connection, table_name='200309', ticker='QQQ', limit=2000):
    """Fetch stock data from NAS database"""
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
            df['datetime'] = pd.to_datetime(df['window_start'], unit='ns')
            for col in ['open', 'high', 'low', 'close']:
                df[col] = df[col].astype(float)
            
            return df
        else:
            return None
            
    except Exception as e:
        print(f"✗ Error fetching data: {e}")
        return None

# ============================================================================
# Reporting and Visualization
# ============================================================================

def print_backtest_report(results, ticker, table_name, params):
    """Print comprehensive backtest report"""
    metrics = results['metrics']
    trades_df = results['trades']

    print("\n" + "=" * 80)
    print("BACKTEST REPORT")
    print("=" * 80)

    print(f"\nSymbol: {ticker}")
    print(f"Table: {table_name}")
    print(f"Initial Capital: ${results['equity'][0]:,.2f}")
    print(f"Final Capital: ${results['equity'][-1]:,.2f}")

    print(f"\nIndicator Parameters:")
    print(f"  ATR Length: {params['atr_length']}")
    print(f"  ATR Multiplier: {params['atr_mult']}")
    print(f"  Radius Strength: {params['radius_strength']}")
    print(f"  Smoothness: {params['smoothness']}")

    print(f"\n{'PERFORMANCE METRICS':^80}")
    print("-" * 80)
    print(f"Total Return:        ${metrics['total_return']:>12,.2f}  ({metrics['total_return_pct']:>6.2f}%)")
    print(f"Max Drawdown:        ${metrics['max_drawdown']:>12,.2f}  ({metrics['max_drawdown_pct']:>6.2f}%)")
    print(f"Sharpe Ratio:        {metrics['sharpe_ratio']:>12.2f}")

    print(f"\n{'TRADE STATISTICS':^80}")
    print("-" * 80)
    print(f"Total Trades:        {metrics['total_trades']:>12}")
    print(f"Winning Trades:      {metrics['winning_trades']:>12}  ({metrics['win_rate']:>6.2f}%)")
    print(f"Losing Trades:       {metrics['losing_trades']:>12}")
    print(f"Average Win:         ${metrics['avg_win']:>12,.2f}")
    print(f"Average Loss:        ${metrics['avg_loss']:>12,.2f}")
    print(f"Profit Factor:       {metrics['profit_factor']:>12.2f}")
    print(f"Avg Bars Held:       {metrics['avg_bars_held']:>12.1f}")

    # Show sample trades
    if len(trades_df) > 0:
        print(f"\n{'SAMPLE TRADES (First 10)':^80}")
        print("-" * 80)
        print(f"{'Type':<8} {'Entry Bar':<12} {'Exit Bar':<12} {'Entry $':<12} {'Exit $':<12} {'P&L $':<12} {'Return %':<10}")
        print("-" * 80)

        for i, trade in trades_df.head(10).iterrows():
            print(f"{trade['type']:<8} {trade['entry_bar']:<12} {trade['exit_bar']:<12} "
                  f"${trade['entry_price']:<11.2f} ${trade['exit_price']:<11.2f} "
                  f"${trade['pnl']:<11.2f} {trade['return_pct']:<9.2f}%")

        if len(trades_df) > 10:
            print(f"\n... and {len(trades_df) - 10} more trades")

    print("\n" + "=" * 80)

def plot_backtest_results(df, results, ticker, table_name):
    """Plot comprehensive backtest results"""
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(4, 2, hspace=0.3, wspace=0.3)

    # Plot 1: Price and Signals
    ax1 = fig.add_subplot(gs[0:2, :])
    ax1.plot(df.index, df['close'], label='Close Price', color='black', linewidth=1, alpha=0.7)

    # Plot entry/exit points
    trades_df = results['trades']
    for _, trade in trades_df.iterrows():
        entry_idx = trade['entry_bar']
        exit_idx = trade['exit_bar']

        if trade['type'] == 'long':
            ax1.scatter(entry_idx, trade['entry_price'], marker='^', color='green', s=100, zorder=5)
            ax1.scatter(exit_idx, trade['exit_price'], marker='v', color='red', s=100, zorder=5)
            ax1.plot([entry_idx, exit_idx], [trade['entry_price'], trade['exit_price']],
                    'g--' if trade['pnl'] > 0 else 'r--', alpha=0.3, linewidth=1)
        else:  # short
            ax1.scatter(entry_idx, trade['entry_price'], marker='v', color='red', s=100, zorder=5)
            ax1.scatter(exit_idx, trade['exit_price'], marker='^', color='green', s=100, zorder=5)
            ax1.plot([entry_idx, exit_idx], [trade['entry_price'], trade['exit_price']],
                    'g--' if trade['pnl'] > 0 else 'r--', alpha=0.3, linewidth=1)

    ax1.set_ylabel('Price ($)', fontsize=12)
    ax1.set_title(f'Backtest Results - {ticker} ({table_name})', fontsize=14, fontweight='bold')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Equity Curve
    ax2 = fig.add_subplot(gs[2, :])
    equity = results['equity']
    ax2.plot(equity, label='Equity', color='blue', linewidth=2)
    ax2.axhline(y=equity[0], color='gray', linestyle='--', alpha=0.5, label='Initial Capital')
    ax2.fill_between(range(len(equity)), equity[0], equity, where=(equity >= equity[0]),
                     color='green', alpha=0.2, label='Profit')
    ax2.fill_between(range(len(equity)), equity[0], equity, where=(equity < equity[0]),
                     color='red', alpha=0.2, label='Loss')
    ax2.set_ylabel('Equity ($)', fontsize=12)
    ax2.set_xlabel('Bar Index', fontsize=12)
    ax2.set_title('Equity Curve', fontsize=12, fontweight='bold')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)

    # Plot 3: Trade P&L Distribution
    ax3 = fig.add_subplot(gs[3, 0])
    if len(trades_df) > 0:
        pnl_values = trades_df['pnl'].values
        colors = ['green' if x > 0 else 'red' for x in pnl_values]
        ax3.bar(range(len(pnl_values)), pnl_values, color=colors, alpha=0.6)
        ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax3.set_ylabel('P&L ($)', fontsize=10)
        ax3.set_xlabel('Trade Number', fontsize=10)
        ax3.set_title('Trade P&L Distribution', fontsize=11, fontweight='bold')
        ax3.grid(True, alpha=0.3)

    # Plot 4: Drawdown
    ax4 = fig.add_subplot(gs[3, 1])
    peak = np.maximum.accumulate(equity)
    drawdown = (peak - equity) / peak * 100
    ax4.fill_between(range(len(drawdown)), 0, drawdown, color='red', alpha=0.3)
    ax4.plot(drawdown, color='darkred', linewidth=1)
    ax4.set_ylabel('Drawdown (%)', fontsize=10)
    ax4.set_xlabel('Bar Index', fontsize=10)
    ax4.set_title('Drawdown', fontsize=11, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.invert_yaxis()

    plt.tight_layout()
    filename = f'backtest_{ticker}_{table_name}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"\n✓ Backtest chart saved to: {filename}")
    plt.show()

# ============================================================================
# Main Execution
# ============================================================================

def run_full_backtest(ticker='QQQ', table_name='200309', limit=2000,
                     radius_strength=0.10, atr_length=14, atr_mult=2.0, smoothness=5,
                     initial_capital=10000, commission=0.001, slippage=0.0005):
    """
    Run complete backtest with data fetching, indicator calculation, and reporting
    """
    print("\n" + "=" * 80)
    print("CURVED RADIUS SUPERTREND - BACKTEST ENGINE")
    print("=" * 80)

    # Connect to database
    print("\n[1/5] Connecting to NAS database...")
    connection = connect_to_nas()
    if not connection:
        print("✗ Failed to connect to database")
        return None
    print("✓ Connected successfully")

    # Fetch data
    print(f"\n[2/5] Fetching data for {ticker} from table {table_name}...")
    df = get_stock_data(connection, table_name, ticker, limit)
    connection.close()

    if df is None or len(df) < 50:
        print("✗ Insufficient data")
        return None

    print(f"✓ Fetched {len(df)} bars")
    print(f"  Date range: {df['datetime'].min()} to {df['datetime'].max()}")
    print(f"  Price range: ${df['low'].min():.2f} - ${df['high'].max():.2f}")

    # Calculate indicator
    print(f"\n[3/5] Calculating Curved Radius Supertrend...")
    indicator = CurvedRadiusSupertrendFixed(
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
    print(f"\n[4/5] Running backtest...")
    engine = BacktestEngine(
        initial_capital=initial_capital,
        commission=commission,
        slippage=slippage
    )

    results = engine.run_backtest(df, signals)
    print(f"✓ Backtest complete")
    print(f"  Total trades executed: {len(results['trades'])}")

    # Generate report
    print(f"\n[5/5] Generating report...")
    params = {
        'atr_length': atr_length,
        'atr_mult': atr_mult,
        'radius_strength': radius_strength,
        'smoothness': smoothness
    }

    print_backtest_report(results, ticker, table_name, params)
    plot_backtest_results(df, results, ticker, table_name)

    return results

if __name__ == "__main__":
    # Run backtest with default parameters
    results = run_full_backtest(
        ticker='QQQ',
        table_name='200309',
        limit=2000,
        radius_strength=0.10,  # Scalping setting for 1-min data
        atr_length=14,
        atr_mult=2.0,
        smoothness=5,
        initial_capital=10000,
        commission=0.001,  # 0.1%
        slippage=0.0005    # 0.05%
    )


