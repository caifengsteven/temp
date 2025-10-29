"""
Backtesting Engine for Curved Radius Supertrend Strategy

This module provides a comprehensive backtesting framework for the
Curved Radius Supertrend indicator.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from curved_radius_supertrend import CurvedRadiusSupertrend


class Trade:
    """Represents a single trade"""
    
    def __init__(self, entry_date, entry_price, direction, size=1.0):
        self.entry_date = entry_date
        self.entry_price = entry_price
        self.direction = direction  # 'LONG' or 'SHORT'
        self.size = size
        self.exit_date = None
        self.exit_price = None
        self.pnl = 0.0
        self.return_pct = 0.0
        self.bars_held = 0
    
    def close(self, exit_date, exit_price):
        """Close the trade"""
        self.exit_date = exit_date
        self.exit_price = exit_price
        
        if self.direction == 'LONG':
            self.pnl = (exit_price - self.entry_price) * self.size
            self.return_pct = (exit_price / self.entry_price - 1) * 100
        else:  # SHORT
            self.pnl = (self.entry_price - exit_price) * self.size
            self.return_pct = (self.entry_price / exit_price - 1) * 100
    
    def __repr__(self):
        status = "OPEN" if self.exit_date is None else "CLOSED"
        return f"Trade({self.direction}, {status}, Entry: {self.entry_price:.2f}, Exit: {self.exit_price:.2f if self.exit_price else 'N/A'}, PnL: {self.pnl:.2f})"


class BacktestEngine:
    """
    Backtesting engine for Curved Radius Supertrend strategy
    """
    
    def __init__(
        self,
        initial_capital: float = 100000.0,
        commission: float = 0.001,  # 0.1% per trade
        slippage: float = 0.0005,   # 0.05% slippage
        position_size: float = 1.0,  # Fraction of capital per trade
        allow_short: bool = False
    ):
        """
        Initialize backtesting engine
        
        Parameters:
        -----------
        initial_capital : float
            Starting capital
        commission : float
            Commission rate (as decimal, e.g., 0.001 = 0.1%)
        slippage : float
            Slippage rate (as decimal)
        position_size : float
            Fraction of capital to use per trade (0.0 to 1.0)
        allow_short : bool
            Whether to allow short positions
        """
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.position_size = position_size
        self.allow_short = allow_short
        
        # Results storage
        self.trades = []
        self.equity_curve = []
        self.current_position = None
        self.cash = initial_capital
        self.equity = initial_capital
    
    def calculate_position_size(self, price: float) -> float:
        """Calculate number of shares to trade"""
        available_capital = self.cash * self.position_size
        shares = available_capital / price
        return shares
    
    def apply_costs(self, price: float, direction: str) -> float:
        """Apply commission and slippage to price"""
        if direction == 'BUY':
            # Pay commission and slippage when buying
            adjusted_price = price * (1 + self.commission + self.slippage)
        else:  # SELL
            # Pay commission and slippage when selling
            adjusted_price = price * (1 - self.commission - self.slippage)
        
        return adjusted_price
    
    def run_backtest(
        self,
        data: pd.DataFrame,
        indicator_params: Dict = None
    ) -> Dict:
        """
        Run backtest on historical data
        
        Parameters:
        -----------
        data : pd.DataFrame
            DataFrame with columns: date, open, high, low, close, volume
        indicator_params : Dict
            Parameters for CurvedRadiusSupertrend indicator
            
        Returns:
        --------
        Dict : Backtest results and statistics
        """
        # Reset state
        self.trades = []
        self.equity_curve = []
        self.current_position = None
        self.cash = self.initial_capital
        self.equity = self.initial_capital
        
        # Default indicator parameters
        if indicator_params is None:
            indicator_params = {
                'atr_period': 10,
                'atr_multiplier': 3.0,
                'radius_strength': 0.5,
                'smoothness': 3
            }
        
        # Calculate indicator
        print("Calculating Curved Radius Supertrend...")
        indicator = CurvedRadiusSupertrend(**indicator_params)
        
        result = indicator.calculate(
            data['high'].values,
            data['low'].values,
            data['close'].values
        )
        
        # Add indicator results to data
        data = data.copy()
        data['direction'] = result['direction'].values
        data['trend_line'] = result['trend_line'].values
        data['curved_upper'] = result['curved_upper'].values
        data['curved_lower'] = result['curved_lower'].values
        
        # Run through each bar
        print("Running backtest...")
        bankruptcy = False

        for i in range(1, len(data)):
            current_bar = data.iloc[i]
            prev_bar = data.iloc[i-1]

            date = current_bar['date']
            price = current_bar['close']

            # Calculate current equity FIRST (before any trades)
            if self.current_position is not None:
                if self.current_position.direction == 'LONG':
                    position_value = self.current_position.size * price
                    self.equity = self.cash + position_value
                else:  # SHORT
                    # For short: equity = cash + (entry_price - current_price) * size
                    pnl = (self.current_position.entry_price - price) * self.current_position.size
                    self.equity = self.cash + pnl
            else:
                self.equity = self.cash

            # BANKRUPTCY CHECK - Stop if equity <= 0
            if self.equity <= 0:
                bankruptcy = True
                # Close position at current price if any
                if self.current_position is not None:
                    exit_price = self.apply_costs(price, 'SELL')
                    self.current_position.close(date, exit_price)
                    self.current_position.bars_held = i - data[data['date'] == self.current_position.entry_date].index[0]
                    self.trades.append(self.current_position)
                    self.current_position = None

                # Set equity to 0
                self.equity = 0
                self.cash = 0

                # Record final equity
                self.equity_curve.append({
                    'date': date,
                    'equity': 0,
                    'cash': 0,
                    'position': 0
                })
                break  # Stop backtest

            # Detect trend change
            trend_changed = current_bar['direction'] != prev_bar['direction']

            if trend_changed:
                # Close existing position if any
                if self.current_position is not None:
                    exit_price = self.apply_costs(price, 'SELL')
                    self.current_position.close(date, exit_price)
                    self.current_position.bars_held = i - data[data['date'] == self.current_position.entry_date].index[0]

                    # Update cash based on P&L
                    if self.current_position.direction == 'LONG':
                        # Long: cash back = size * exit_price
                        self.cash += self.current_position.size * exit_price
                    else:  # SHORT
                        # Short: P&L = (entry_price - exit_price) * size
                        pnl = (self.current_position.entry_price - exit_price) * self.current_position.size
                        self.cash += pnl

                    self.trades.append(self.current_position)
                    self.current_position = None

                    # Update equity after closing
                    self.equity = self.cash

                    # Check for bankruptcy after closing position
                    if self.equity <= 0:
                        bankruptcy = True
                        self.equity = 0
                        self.cash = 0
                        self.equity_curve.append({
                            'date': date,
                            'equity': 0,
                            'cash': 0,
                            'position': 0
                        })
                        break

                # Open new position (only if we have positive equity)
                if self.equity > 0:
                    if current_bar['direction'] == 1:  # Uptrend - go LONG
                        entry_price = self.apply_costs(price, 'BUY')
                        size = self.calculate_position_size(entry_price)

                        if size > 0 and (size * entry_price) <= self.cash:
                            self.current_position = Trade(date, entry_price, 'LONG', size)
                            self.cash -= size * entry_price

                    elif current_bar['direction'] == -1 and self.allow_short:  # Downtrend - go SHORT
                        entry_price = self.apply_costs(price, 'SELL')
                        size = self.calculate_position_size(entry_price)

                        if size > 0:
                            self.current_position = Trade(date, entry_price, 'SHORT', size)
                            # For short, no cash needed upfront (simplified model)

            # Recalculate equity after potential trades
            if self.current_position is not None:
                if self.current_position.direction == 'LONG':
                    position_value = self.current_position.size * price
                    self.equity = self.cash + position_value
                else:  # SHORT
                    pnl = (self.current_position.entry_price - price) * self.current_position.size
                    self.equity = self.cash + pnl
            else:
                self.equity = self.cash

            # Record equity
            self.equity_curve.append({
                'date': date,
                'equity': max(0, self.equity),  # Never record negative equity
                'cash': self.cash,
                'position': 1 if self.current_position is not None else 0
            })

        # Close any remaining position (only if not bankrupt)
        if not bankruptcy and self.current_position is not None:
            final_bar = data.iloc[-1]
            exit_price = self.apply_costs(final_bar['close'], 'SELL')
            self.current_position.close(final_bar['date'], exit_price)
            self.current_position.bars_held = len(data) - 1 - data[data['date'] == self.current_position.entry_date].index[0]

            # Update cash based on P&L
            if self.current_position.direction == 'LONG':
                self.cash += self.current_position.size * exit_price
            else:  # SHORT
                pnl = (self.current_position.entry_price - exit_price) * self.current_position.size
                self.cash += pnl

            self.trades.append(self.current_position)
            self.equity = self.cash

        # Calculate statistics
        stats = self.calculate_statistics(data)

        # Add bankruptcy flag to stats
        stats['bankruptcy'] = bankruptcy

        return {
            'trades': self.trades,
            'equity_curve': pd.DataFrame(self.equity_curve),
            'statistics': stats,
            'data': data,
            'indicator_params': indicator_params
        }
    
    def calculate_statistics(self, data: pd.DataFrame) -> Dict:
        """Calculate backtest statistics"""

        # Calculate final equity
        final_equity = self.equity_curve[-1]['equity'] if self.equity_curve else self.initial_capital
        total_return = (final_equity / self.initial_capital - 1) * 100

        if len(self.trades) == 0:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0.0,
                'total_pnl': 0.0,
                'avg_pnl_per_trade': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'profit_factor': 0.0,
                'total_return_pct': total_return,
                'final_equity': final_equity,
                'sharpe_ratio': 0.0,
                'max_drawdown_pct': 0.0,
                'avg_bars_held': 0.0,
                'start_date': data['date'].iloc[0] if len(data) > 0 else None,
                'end_date': data['date'].iloc[-1] if len(data) > 0 else None,
                'error': 'No trades executed'
            }
        
        # Basic statistics
        total_trades = len(self.trades)
        winning_trades = [t for t in self.trades if t.pnl > 0]
        losing_trades = [t for t in self.trades if t.pnl <= 0]
        
        win_rate = len(winning_trades) / total_trades * 100 if total_trades > 0 else 0
        
        total_pnl = sum(t.pnl for t in self.trades)
        avg_pnl = total_pnl / total_trades if total_trades > 0 else 0
        
        avg_win = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t.pnl for t in losing_trades]) if losing_trades else 0
        
        profit_factor = abs(sum(t.pnl for t in winning_trades) / sum(t.pnl for t in losing_trades)) if losing_trades and sum(t.pnl for t in losing_trades) != 0 else float('inf')
        
        # Returns
        final_equity = self.equity_curve[-1]['equity'] if self.equity_curve else self.initial_capital
        total_return = (final_equity / self.initial_capital - 1) * 100
        
        # Equity curve analysis
        equity_df = pd.DataFrame(self.equity_curve)
        equity_df['returns'] = equity_df['equity'].pct_change()
        
        # Sharpe ratio (annualized, assuming 252 trading days)
        if len(equity_df) > 1:
            sharpe_ratio = np.sqrt(252) * equity_df['returns'].mean() / equity_df['returns'].std() if equity_df['returns'].std() > 0 else 0
        else:
            sharpe_ratio = 0
        
        # Maximum drawdown
        equity_df['cummax'] = equity_df['equity'].cummax()
        equity_df['drawdown'] = (equity_df['equity'] - equity_df['cummax']) / equity_df['cummax'] * 100
        max_drawdown = equity_df['drawdown'].min()
        
        # Average bars held
        avg_bars_held = np.mean([t.bars_held for t in self.trades]) if self.trades else 0
        
        return {
            'total_trades': total_trades,
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'avg_pnl_per_trade': avg_pnl,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'total_return_pct': total_return,
            'final_equity': final_equity,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown_pct': max_drawdown,
            'avg_bars_held': avg_bars_held,
            'start_date': data['date'].iloc[0],
            'end_date': data['date'].iloc[-1]
        }

