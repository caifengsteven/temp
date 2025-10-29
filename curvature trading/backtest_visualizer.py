"""
Visualization tools for backtest results

This module provides comprehensive visualization for backtesting results.
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
from typing import Dict
from matplotlib.gridspec import GridSpec


def plot_backtest_results(results: Dict, ticker: str = None, save_path: str = None):
    """
    Create comprehensive visualization of backtest results
    
    Parameters:
    -----------
    results : Dict
        Results dictionary from BacktestEngine
    ticker : str
        Stock ticker (for title)
    save_path : str
        Path to save the figure
    """
    data = results['data']
    equity_curve = results['equity_curve']
    trades = results['trades']
    stats = results['statistics']
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(4, 2, figure=fig, hspace=0.3, wspace=0.25)
    
    # Title
    title = f"Backtest Results"
    if ticker:
        title += f" - {ticker}"
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # 1. Price chart with indicator and trades
    ax1 = fig.add_subplot(gs[0:2, :])
    
    # Plot price
    ax1.plot(data['date'], data['close'], label='Close Price', 
            color='black', linewidth=1.5, alpha=0.7)
    
    # Plot curved bands
    ax1.plot(data['date'], data['curved_upper'], label='Upper Band',
            color='red', linewidth=1.5, alpha=0.5, linestyle='--')
    ax1.plot(data['date'], data['curved_lower'], label='Lower Band',
            color='green', linewidth=1.5, alpha=0.5, linestyle='--')
    
    # Plot trend line
    uptrend_mask = data['direction'] == 1
    downtrend_mask = data['direction'] == -1
    
    trend_up = data['trend_line'].copy()
    trend_up[downtrend_mask] = np.nan
    ax1.plot(data['date'], trend_up, color='green', linewidth=2.5, 
            label='Uptrend', alpha=0.8)
    
    trend_down = data['trend_line'].copy()
    trend_down[uptrend_mask] = np.nan
    ax1.plot(data['date'], trend_down, color='red', linewidth=2.5,
            label='Downtrend', alpha=0.8)
    
    # Mark trades
    for trade in trades:
        entry_date = trade.entry_date
        exit_date = trade.exit_date
        
        # Find prices
        entry_price = trade.entry_price
        exit_price = trade.exit_price
        
        if trade.direction == 'LONG':
            # Buy signal
            ax1.scatter(entry_date, entry_price, color='green', marker='^',
                       s=150, zorder=5, edgecolors='white', linewidths=2)
            # Sell signal
            if exit_date:
                ax1.scatter(exit_date, exit_price, color='red', marker='v',
                           s=150, zorder=5, edgecolors='white', linewidths=2)
        else:  # SHORT
            # Short entry
            ax1.scatter(entry_date, entry_price, color='red', marker='v',
                       s=150, zorder=5, edgecolors='white', linewidths=2)
            # Short exit
            if exit_date:
                ax1.scatter(exit_date, exit_price, color='green', marker='^',
                           s=150, zorder=5, edgecolors='white', linewidths=2)
    
    ax1.set_title('Price Chart with Curved Radius Supertrend', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Date', fontsize=10)
    ax1.set_ylabel('Price ($)', fontsize=10)
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
    
    # 2. Equity curve
    ax2 = fig.add_subplot(gs[2, :])
    
    ax2.plot(equity_curve['date'], equity_curve['equity'], 
            color='blue', linewidth=2, label='Portfolio Equity')
    ax2.axhline(y=stats['final_equity'], color='green', linestyle='--', 
               alpha=0.5, label=f"Final: ${stats['final_equity']:,.0f}")
    ax2.axhline(y=100000, color='gray', linestyle='--', 
               alpha=0.5, label='Initial Capital')
    
    ax2.set_title('Equity Curve', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Date', fontsize=10)
    ax2.set_ylabel('Equity ($)', fontsize=10)
    ax2.legend(loc='best', fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
    
    # 3. Drawdown
    ax3 = fig.add_subplot(gs[3, 0])
    
    equity_curve['cummax'] = equity_curve['equity'].cummax()
    equity_curve['drawdown'] = (equity_curve['equity'] - equity_curve['cummax']) / equity_curve['cummax'] * 100
    
    ax3.fill_between(equity_curve['date'], equity_curve['drawdown'], 0,
                     color='red', alpha=0.3)
    ax3.plot(equity_curve['date'], equity_curve['drawdown'],
            color='red', linewidth=1.5)
    
    ax3.set_title('Drawdown', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Date', fontsize=10)
    ax3.set_ylabel('Drawdown (%)', fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)
    
    # 4. Trade distribution
    ax4 = fig.add_subplot(gs[3, 1])
    
    returns = [t.return_pct for t in trades]
    
    ax4.hist(returns, bins=30, color='steelblue', alpha=0.7, edgecolor='black')
    ax4.axvline(x=0, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax4.axvline(x=np.mean(returns), color='green', linestyle='--', 
               linewidth=2, alpha=0.7, label=f'Mean: {np.mean(returns):.2f}%')
    
    ax4.set_title('Trade Return Distribution', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Return (%)', fontsize=10)
    ax4.set_ylabel('Frequency', fontsize=10)
    ax4.legend(loc='best', fontsize=9)
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Add statistics text box
    stats_text = f"""
    Total Trades: {stats['total_trades']}
    Win Rate: {stats['win_rate']:.1f}%
    Total Return: {stats['total_return_pct']:.2f}%
    Sharpe Ratio: {stats['sharpe_ratio']:.2f}
    Max Drawdown: {stats['max_drawdown_pct']:.2f}%
    Profit Factor: {stats['profit_factor']:.2f}
    """
    
    fig.text(0.02, 0.02, stats_text, fontsize=9, family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Save or show
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    return fig


def plot_parameter_optimization(results_list: list, param_name: str, save_path: str = None):
    """
    Plot parameter optimization results
    
    Parameters:
    -----------
    results_list : list
        List of backtest results with different parameters
    param_name : str
        Name of the parameter being optimized
    save_path : str
        Path to save the figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Parameter Optimization: {param_name}', fontsize=14, fontweight='bold')
    
    # Extract data
    param_values = [r['params'][param_name] for r in results_list]
    returns = [r['total_return_pct'] for r in results_list]
    sharpe = [r['sharpe_ratio'] for r in results_list]
    max_dd = [r['max_drawdown_pct'] for r in results_list]
    win_rates = [r['win_rate'] for r in results_list]
    
    # Plot 1: Return vs Parameter
    axes[0, 0].plot(param_values, returns, 'o-', color='blue', linewidth=2, markersize=8)
    axes[0, 0].set_xlabel(param_name, fontsize=10)
    axes[0, 0].set_ylabel('Total Return (%)', fontsize=10)
    axes[0, 0].set_title('Total Return', fontsize=11, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Sharpe vs Parameter
    axes[0, 1].plot(param_values, sharpe, 'o-', color='green', linewidth=2, markersize=8)
    axes[0, 1].set_xlabel(param_name, fontsize=10)
    axes[0, 1].set_ylabel('Sharpe Ratio', fontsize=10)
    axes[0, 1].set_title('Sharpe Ratio', fontsize=11, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Max Drawdown vs Parameter
    axes[1, 0].plot(param_values, max_dd, 'o-', color='red', linewidth=2, markersize=8)
    axes[1, 0].set_xlabel(param_name, fontsize=10)
    axes[1, 0].set_ylabel('Max Drawdown (%)', fontsize=10)
    axes[1, 0].set_title('Maximum Drawdown', fontsize=11, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Win Rate vs Parameter
    axes[1, 1].plot(param_values, win_rates, 'o-', color='purple', linewidth=2, markersize=8)
    axes[1, 1].set_xlabel(param_name, fontsize=10)
    axes[1, 1].set_ylabel('Win Rate (%)', fontsize=10)
    axes[1, 1].set_title('Win Rate', fontsize=11, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    return fig


def plot_multi_stock_comparison(all_results: dict, save_path: str = None):
    """
    Compare backtest results across multiple stocks
    
    Parameters:
    -----------
    all_results : dict
        Dictionary with ticker as key and results as value
    save_path : str
        Path to save the figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Multi-Stock Comparison', fontsize=14, fontweight='bold')
    
    tickers = list(all_results.keys())
    
    # Extract statistics
    returns = [all_results[t]['statistics']['total_return_pct'] for t in tickers]
    sharpe = [all_results[t]['statistics']['sharpe_ratio'] for t in tickers]
    max_dd = [all_results[t]['statistics']['max_drawdown_pct'] for t in tickers]
    win_rates = [all_results[t]['statistics']['win_rate'] for t in tickers]
    
    # Plot 1: Total Returns
    axes[0, 0].bar(tickers, returns, color='steelblue', alpha=0.7, edgecolor='black')
    axes[0, 0].set_ylabel('Total Return (%)', fontsize=10)
    axes[0, 0].set_title('Total Returns', fontsize=11, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    axes[0, 0].axhline(y=0, color='red', linestyle='--', alpha=0.5)
    
    # Plot 2: Sharpe Ratios
    axes[0, 1].bar(tickers, sharpe, color='green', alpha=0.7, edgecolor='black')
    axes[0, 1].set_ylabel('Sharpe Ratio', fontsize=10)
    axes[0, 1].set_title('Sharpe Ratios', fontsize=11, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Max Drawdowns
    axes[1, 0].bar(tickers, max_dd, color='red', alpha=0.7, edgecolor='black')
    axes[1, 0].set_ylabel('Max Drawdown (%)', fontsize=10)
    axes[1, 0].set_title('Maximum Drawdowns', fontsize=11, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Win Rates
    axes[1, 1].bar(tickers, win_rates, color='purple', alpha=0.7, edgecolor='black')
    axes[1, 1].set_ylabel('Win Rate (%)', fontsize=10)
    axes[1, 1].set_title('Win Rates', fontsize=11, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    axes[1, 1].axhline(y=50, color='gray', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    return fig

