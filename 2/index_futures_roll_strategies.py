"""
Index Futures Roll Strategy Backtester
=====================================
This module implements and tests various index futures roll trading strategies
using Bloomberg-style data.

Strategies Implemented:
1. Liquidity-Based Strategy - Roll when back month OI exceeds front month OI
2. Fair Value Strategy - Roll based on deviation from theoretical fair value
3. Momentum Strategy - Roll based on spread momentum signals

Author: Matrix Agent
Version: 1.0.0
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
import json
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass, field

# Set style for visualizations
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

warnings.filterwarnings('ignore')


@dataclass
class StrategyResult:
    """Results from a single strategy backtest"""
    strategy_name: str
    total_pnl: float
    avg_roll_cost: float
    max_drawdown: float
    sharpe_ratio: float
    win_rate: float
    num_rolls: int
    roll_dates: List[datetime]
    roll_costs: List[float]
    cumulative_pnl: pd.Series = field(default_factory=pd.Series)
    

class BloombergDataSimulator:
    """
    Simulates Bloomberg-style index futures data for backtesting.
    In production, this would connect to Bloomberg API via blpapi.
    """
    
    def __init__(self, config: dict):
        self.config = config
        self.index_futures = config['index_futures'][0]
        self.risk_free_rate = config['backtest_parameters']['risk_free_rate']
        
    def generate_futures_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Generate realistic index futures data with calendar spreads."""
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        
        # Generate daily business days
        dates = pd.date_range(start, end, freq='B')
        
        # Generate underlying index price (S&P 500-like)
        np.random.seed(42)
        index_returns = np.random.normal(0.0004, 0.01, len(dates))
        index_prices = 3000 * np.exp(np.cumsum(index_returns))
        
        # Generate all contracts and their active periods
        contracts = self._generate_contract_timeline(start, end)
        
        print(f"Generated {len(contracts)} contracts")
        
        # Build comprehensive dataset - generate data for ALL contracts on each date
        data_records = []
        
        # Get unique dates
        unique_dates = sorted(dates.unique())
        
        for i, date in enumerate(unique_dates):
            idx_price = index_prices[i]
            
            # For each date, get all active contracts (near and far)
            active_contracts = [c for c in contracts if c['start_date'] <= date <= c['end_date']]
            
            # Sort by expiration (nearest first)
            active_contracts = sorted(active_contracts, key=lambda x: x['expiration'])
            
            # Take up to 2 nearest contracts
            for rank, contract in enumerate(active_contracts[:3]):  # Take top 3 for more data
                days_to_exp = (contract['expiration'] - date).days
                
                if days_to_exp < 0:
                    continue
                    
                # Calculate fair value (cost of carry)
                days_fraction = days_to_exp / 365
                fair_value = idx_price * (1 + self.risk_free_rate * days_fraction)
                
                # Add market noise
                if days_to_exp < 30:
                    volatility_factor = 1.2
                else:
                    volatility_factor = 1.0
                
                market_noise = np.random.normal(0, 0.5) * volatility_factor
                futures_price = fair_value + market_noise
                
                # Generate volume and open interest
                if days_to_exp < 7:
                    base_oi = 500000 * (days_to_exp / 30 + 0.3)
                elif days_to_exp < 60:
                    base_oi = 800000 + np.random.normal(0, 50000)
                else:
                    days_from_start = (date - contract['start_date']).days
                    base_oi = max(100000, min(600000, 100000 + days_from_start * 8000))
                
                volume = int(base_oi * np.random.uniform(0.05, 0.15))
                
                record = {
                    'date': date,
                    'ticker': contract['ticker'],
                    'contract_month': contract['contract_month'],
                    'expiration': contract['expiration'],
                    'days_to_expiration': days_to_exp,
                    'close': round(futures_price, 2),
                    'volume': max(1000, volume),
                    'open_interest': int(max(10000, base_oi)),
                    'implied_rate': self.risk_free_rate + np.random.normal(0, 0.002),
                    'contract_rank': rank + 1  # 1 = nearest, 2 = second nearest, etc.
                }
                data_records.append(record)
        
        df = pd.DataFrame(data_records)
        
        if len(df) == 0:
            print("ERROR: No data generated!")
            return df
            
        df = df.sort_values(['date', 'days_to_expiration']).reset_index(drop=True)
        
        print(f"Generated {len(df)} data points")
        print(f"Date range: {df['date'].min()} to {df['date'].max()}")
        print(f"Unique contracts: {df['ticker'].nunique()}")
        
        # Debug: Check contracts per date
        contracts_per_date = df.groupby('date').size()
        print(f"Contracts per date: min={contracts_per_date.min()}, max={contracts_per_date.max()}, mean={contracts_per_date.mean():.1f}")
        
        return df
    
    def _generate_contract_timeline(self, start_date: datetime, end_date: datetime) -> List[Dict]:
        """Generate quarterly futures contract timeline with proper overlap"""
        contracts = []
        
        # Generate quarterly contracts for 3 years
        for year in range(start_date.year - 1, end_date.year + 2):
            for month in [3, 6, 9, 12]:
                exp_date = datetime(year, month, third_friday(month, year))
                
                # Contract starts 1 year before expiration and ends 1 day before expiration
                start = exp_date - timedelta(days=365)
                end = exp_date - timedelta(days=1)
                
                # Only include contracts that overlap with our date range
                if end >= start_date and start <= end_date:
                    contracts.append({
                        'ticker': f"ES{self._month_code(month)}{str(year)[-1]}",
                        'contract_month': f"{self._month_code(month)}{str(year)[-1]}",
                        'expiration': exp_date,
                        'start_date': start,
                        'end_date': end
                    })
        
        return contracts
    
    def _month_code(self, month: int) -> str:
        """Convert month number to futures month code"""
        codes = {1: 'F', 2: 'G', 3: 'H', 4: 'J', 5: 'K', 6: 'M',
                 7: 'N', 8: 'Q', 9: 'U', 10: 'V', 11: 'X', 12: 'Z'}
        return codes.get(month, 'Z')


def third_friday(month: int, year: int) -> int:
    """Calculate the third Friday of a given month"""
    from calendar import monthcalendar, weekday
    cal = monthcalendar(year, month)
    fridays = []
    for week in cal:
        for day in week:
            if day != 0 and weekday(year, month, day) == 4:
                fridays.append(day)
    return fridays[2]


class LiquidityRollStrategy:
    """Liquidity-Based Roll Strategy - Roll when back month OI exceeds front month"""
    
    def __init__(self, name: str = "Liquidity-Based"):
        self.name = name
        
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate roll signals based on open interest crossover."""
        # Get near (rank 1) and far (rank 2) contracts for each date
        near_data = data[data['contract_rank'] == 1][['date', 'open_interest', 'close', 'days_to_expiration']].copy()
        near_data.columns = ['date', 'near_oi', 'near_price', 'near_dte']
        
        far_data = data[data['contract_rank'] == 2][['date', 'open_interest', 'close', 'days_to_expiration']].copy()
        far_data.columns = ['date', 'far_oi', 'far_price', 'far_dte']
        
        # Merge near and far data
        signals = near_data.merge(far_data, on='date')
        
        if len(signals) == 0:
            print("Warning: No near/far contract pairs found")
            return pd.DataFrame()
        
        # Calculate OI ratio
        signals['oi_ratio'] = signals['far_oi'] / signals['near_oi']
        
        # Roll when far OI exceeds near OI (liquidity shift)
        signals['roll_signal'] = 0
        
        # Also trigger roll if near contract is close to expiration
        signals.loc[signals['near_dte'] <= 3, 'roll_signal'] = 1
        
        # Also trigger when OI ratio crosses above 1 (liquidity shift)
        signals.loc[signals['oi_ratio'] > 1.0, 'roll_signal'] = 1
        
        return signals
    
    def execute_roll(self, data: pd.DataFrame, roll_date: pd.Timestamp) -> float:
        """Calculate the roll cost for a specific date"""
        date_data = data[data['date'] == roll_date].copy()
        
        if len(date_data) < 2:
            return np.nan
            
        date_data = date_data.sort_values('days_to_expiration')
        near = date_data.iloc[0]
        far = date_data.iloc[1]
        
        # Roll cost = Far price - Near price (positive = cost)
        roll_cost = far['close'] - near['close']
        
        return roll_cost


class FairValueRollStrategy:
    """Fair Value Roll Strategy - Roll based on deviation from theoretical fair value"""
    
    def __init__(self, name: str = "Fair Value", deviation_threshold: float = 0.5):
        self.name = name
        self.deviation_threshold = deviation_threshold
        
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate roll signals based on fair value deviation"""
        # Get near (rank 1) and far (rank 2) contracts
        near_data = data[data['contract_rank'] == 1][['date', 'close', 'days_to_expiration', 'implied_rate']].copy()
        near_data.columns = ['date', 'near_price', 'near_dte', 'near_rate']
        
        far_data = data[data['contract_rank'] == 2][['date', 'close', 'days_to_expiration']].copy()
        far_data.columns = ['date', 'far_price', 'far_dte']
        
        signals = near_data.merge(far_data, on='date')
        
        if len(signals) == 0:
            print("Warning: No near/far contract pairs found")
            return pd.DataFrame()
        
        # Market spread
        signals['market_spread'] = signals['far_price'] - signals['near_price']
        
        # Theoretical fair value spread
        days_diff = signals['far_dte'] - signals['near_dte']
        signals['fair_value_spread'] = signals['near_price'] * signals['near_rate'] * (days_diff / 365)
        
        # Deviation from fair value
        signals['deviation'] = signals['market_spread'] - signals['fair_value_spread']
        
        # Z-score of deviation
        signals['deviation_ma'] = signals['deviation'].rolling(window=5).mean()
        signals['deviation_std'] = signals['deviation'].rolling(window=5).std()
        signals['deviation_zscore'] = (signals['deviation'] - signals['deviation_ma']) / signals['deviation_std']
        
        # Roll when deviation is negative (spread cheap) or near expiration
        signals['roll_signal'] = 0
        signals.loc[signals['deviation_zscore'] < -self.deviation_threshold, 'roll_signal'] = 1
        signals.loc[signals['near_dte'] <= 3, 'roll_signal'] = 1
        
        return signals
    
    def execute_roll(self, data: pd.DataFrame, roll_date: pd.Timestamp) -> float:
        """Calculate the roll cost for a specific date"""
        date_data = data[data['date'] == roll_date].copy()
        
        if len(date_data) < 2:
            return np.nan
            
        date_data = date_data.sort_values('days_to_expiration')
        near = date_data.iloc[0]
        far = date_data.iloc[1]
        
        roll_cost = far['close'] - near['close']
        
        return roll_cost


class MomentumRollStrategy:
    """Momentum Roll Strategy - Roll based on calendar spread momentum"""
    
    def __init__(self, name: str = "Momentum", lookback: int = 5):
        self.name = name
        self.lookback = lookback
        
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate roll signals based on momentum"""
        # Get near and far contracts
        near_data = data[data['contract_rank'] == 1][['date', 'close', 'days_to_expiration']].copy()
        near_data.columns = ['date', 'near_price', 'near_dte']
        
        far_data = data[data['contract_rank'] == 2][['date', 'close']].copy()
        far_data.columns = ['date', 'far_price']
        
        signals = near_data.merge(far_data, on='date')
        
        if len(signals) == 0:
            print("Warning: No near/far contract pairs found")
            return pd.DataFrame()
        
        # Calculate spread
        signals['spread'] = signals['far_price'] - signals['near_price']
        
        # Moving average of spread
        signals['spread_ma'] = signals['spread'].rolling(window=self.lookback).mean()
        
        # Momentum signal
        signals['momentum'] = signals['spread'] - signals['spread_ma']
        
        # Roll when spread is below MA (narrowing) or near expiration
        signals['roll_signal'] = 0
        signals.loc[signals['spread'] < signals['spread_ma'], 'roll_signal'] = 1
        signals.loc[signals['near_dte'] <= 3, 'roll_signal'] = 1
        
        return signals
    
    def execute_roll(self, data: pd.DataFrame, roll_date: pd.Timestamp) -> float:
        """Calculate the roll cost for a specific date"""
        date_data = data[data['date'] == roll_date].copy()
        
        if len(date_data) < 2:
            return np.nan
            
        date_data = date_data.sort_values('days_to_expiration')
        near = date_data.iloc[0]
        far = date_data.iloc[1]
        
        roll_cost = far['close'] - near['close']
        
        return roll_cost


class BenchmarkStrategy:
    """Benchmark: Always roll on the last trading day before expiration"""
    
    def __init__(self, name: str = "Benchmark (Expiry Roll)"):
        self.name = name
        
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate roll signals - always roll on last day"""
        near_data = data[data['contract_rank'] == 1][['date', 'days_to_expiration', 'close']].copy()
        near_data.columns = ['date', 'near_dte', 'near_price']
        
        # Roll on last day before expiration
        near_data['roll_signal'] = (near_data['near_dte'] <= 1).astype(int)
        
        return near_data
    
    def execute_roll(self, data: pd.DataFrame, roll_date: pd.Timestamp) -> float:
        """Calculate the roll cost for a specific date"""
        date_data = data[data['date'] == roll_date].copy()
        
        if len(date_data) < 2:
            return np.nan
            
        date_data = date_data.sort_values('days_to_expiration')
        near = date_data.iloc[0]
        far = date_data.iloc[1]
        
        roll_cost = far['close'] - near['close']
        
        return roll_cost


class BacktestEngine:
    """Backtest Engine for Futures Roll Strategies"""
    
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.results = {}
        
    def run_strategy(self, strategy, strategy_name: str) -> StrategyResult:
        """Run backtest for a single strategy"""
        print(f"\n{'='*60}")
        print(f"Running Backtest: {strategy_name}")
        print(f"{'='*60}")
        
        # Generate signals
        signals = strategy.generate_signals(self.data)
        
        if len(signals) == 0:
            print(f"No signals generated for {strategy_name}")
            return None
        
        # Get roll dates where roll_signal == 1
        roll_signals = signals[signals['roll_signal'] == 1]
        
        if len(roll_signals) == 0:
            print(f"No roll signals for {strategy_name}")
            return None
        
        roll_dates = roll_signals['date'].unique()
        roll_dates = sorted(roll_dates)
        
        print(f"Number of roll candidates: {len(roll_dates)}")
        
        # Calculate roll costs
        roll_costs = []
        valid_roll_dates = []
        
        for roll_date in roll_dates:
            cost = strategy.execute_roll(self.data, roll_date)
            if not np.isnan(cost):
                roll_costs.append(cost)
                valid_roll_dates.append(roll_date)
        
        print(f"Valid roll costs: {len(roll_costs)}")
        
        if len(roll_costs) == 0:
            print(f"No valid roll costs for {strategy_name}")
            return None
        
        # Calculate statistics
        total_pnl = -sum(roll_costs)  # Negative because roll cost is a cost
        avg_roll_cost = np.mean(roll_costs)
        
        # Calculate cumulative P&L
        cumulative_pnl = pd.Series(-np.array(roll_costs)).cumsum()
        cumulative_pnl.index = range(len(roll_costs))
        
        # Calculate drawdown
        running_max = cumulative_pnl.cummax()
        drawdown = cumulative_pnl - running_max
        max_drawdown = drawdown.min()
        
        # Calculate Sharpe ratio on P&L (returns)
        # This is the CORRECT way: calculate returns from P&L changes
        pnl_series = pd.Series(roll_costs)
        
        # Calculate period-over-period changes in cumulative P&L
        if len(cumulative_pnl) > 1:
            period_returns = cumulative_pnl.diff().fillna(0)
            # Annualized return (quarterly rolls = 4 periods per year)
            annual_return = period_returns.mean() * 4
            # Annualized volatility
            annual_vol = period_returns.std() * np.sqrt(4)
            
            if annual_vol > 0:
                sharpe_ratio = (annual_return - 0.02) / annual_vol
            else:
                sharpe_ratio = 0
        else:
            sharpe_ratio = 0
        
        # Win rate (negative roll cost = win, meaning roll was free or you got paid)
        # In practice, for futures rolls, "winning" means getting paid to roll (backwardation)
        win_rate = sum(1 for c in roll_costs if c < 0) / len(roll_costs)
        
        # Additional metrics
        roll_cost_std = np.std(roll_costs)
        
        print(f"Total P&L: ${total_pnl:.2f}")
        print(f"Average Roll Cost: ${avg_roll_cost:.2f} ± ${roll_cost_std:.2f}")
        print(f"Max Drawdown: ${max_drawdown:.2f}")
        print(f"Sharpe Ratio: {sharpe_ratio:.3f}")
        print(f"Win Rate: {win_rate:.1%}")
        
        result = StrategyResult(
            strategy_name=strategy_name,
            total_pnl=total_pnl,
            avg_roll_cost=avg_roll_cost,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe_ratio,
            win_rate=win_rate,
            num_rolls=len(roll_costs),
            roll_dates=valid_roll_dates,
            roll_costs=roll_costs,
            cumulative_pnl=cumulative_pnl
        )
        
        self.results[strategy_name] = result
        
        return result
    
    def compare_strategies(self) -> pd.DataFrame:
        """Compare all strategies and return summary"""
        if not self.results:
            return pd.DataFrame()
        
        summary_data = []
        for name, result in self.results.items():
            summary_data.append({
                'Strategy': name,
                'Total P&L ($)': round(result.total_pnl, 2),
                'Avg Roll Cost ($)': round(result.avg_roll_cost, 2),
                'Max Drawdown ($)': round(result.max_drawdown, 2),
                'Sharpe Ratio': round(result.sharpe_ratio, 3),
                'Win Rate (%)': round(result.win_rate * 100, 1),
                'Num Rolls': result.num_rolls
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.sort_values('Total P&L ($)', ascending=False)
        
        return summary_df


class ResultsVisualizer:
    """Generate visualizations for strategy results"""
    
    def __init__(self, results: Dict[str, StrategyResult]):
        self.results = results
        self.colors = {
            'Liquidity-Based': '#2563eb',
            'Fair Value': '#10b981', 
            'Momentum': '#f59e0b',
            'Benchmark (Expiry Roll)': '#6b7280'
        }
        
    def plot_cumulative_pnl(self, save_path: str = None):
        """Plot cumulative P&L for all strategies"""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for name, result in self.results.items():
            color = self.colors.get(name, '#333333')
            ax.plot(result.cumulative_pnl, label=name, color=color, linewidth=2, marker='o', markersize=4)
        
        ax.set_xlabel('Roll Number', fontsize=12)
        ax.set_ylabel('Cumulative P&L ($)', fontsize=12)
        ax.set_title('Cumulative P&L Comparison: Index Futures Roll Strategies', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Chart saved to: {save_path}")
        
        plt.show()
        
    def plot_roll_cost_distribution(self, save_path: str = None):
        """Plot distribution of roll costs"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        for idx, (name, result) in enumerate(self.results.items()):
            if idx >= 4:
                break
            ax = axes[idx]
            color = self.colors.get(name, '#333333')
            
            ax.hist(result.roll_costs, bins=15, color=color, alpha=0.7, edgecolor='black')
            ax.axvline(x=0, color='red', linestyle='--', linewidth=2)
            ax.set_xlabel('Roll Cost ($)', fontsize=10)
            ax.set_ylabel('Frequency', fontsize=10)
            ax.set_title(f'{name}', fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Chart saved to: {save_path}")
        
        plt.show()
        
    def plot_performance_comparison(self, summary_df: pd.DataFrame, save_path: str = None):
        """Plot performance comparison bar chart"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Total P&L
        ax1 = axes[0, 0]
        colors = [self.colors.get(s, '#333333') for s in summary_df['Strategy']]
        bars1 = ax1.barh(summary_df['Strategy'], summary_df['Total P&L ($)'], color=colors, alpha=0.8)
        ax1.set_xlabel('Total P&L ($)', fontsize=11)
        ax1.set_title('Total P&L Comparison', fontsize=12, fontweight='bold')
        ax1.axvline(x=0, color='black', linestyle='--', alpha=0.5)
        for bar, val in zip(bars1, summary_df['Total P&L ($)']):
            ax1.text(val + 5, bar.get_y() + bar.get_height()/2, f'${val:.0f}', va='center', fontsize=9)
        
        # Sharpe Ratio
        ax2 = axes[0, 1]
        bars2 = ax2.barh(summary_df['Strategy'], summary_df['Sharpe Ratio'], color=colors, alpha=0.8)
        ax2.set_xlabel('Sharpe Ratio', fontsize=11)
        ax2.set_title('Risk-Adjusted Returns (Sharpe Ratio)', fontsize=12, fontweight='bold')
        ax2.axvline(x=0, color='black', linestyle='--', alpha=0.5)
        for bar, val in zip(bars2, summary_df['Sharpe Ratio']):
            ax2.text(val + 0.02, bar.get_y() + bar.get_height()/2, f'{val:.2f}', va='center', fontsize=9)
        
        # Win Rate
        ax3 = axes[1, 0]
        bars3 = ax3.barh(summary_df['Strategy'], summary_df['Win Rate (%)'], color=colors, alpha=0.8)
        ax3.set_xlabel('Win Rate (%)', fontsize=11)
        ax3.set_title('Win Rate (Negative Roll Costs)', fontsize=12, fontweight='bold')
        for bar, val in zip(bars3, summary_df['Win Rate (%)']):
            ax3.text(val + 1, bar.get_y() + bar.get_height()/2, f'{val:.0f}%', va='center', fontsize=9)
        
        # Average Roll Cost
        ax4 = axes[1, 1]
        bars4 = ax4.barh(summary_df['Strategy'], summary_df['Avg Roll Cost ($)'], color=colors, alpha=0.8)
        ax4.set_xlabel('Avg Roll Cost ($)', fontsize=11)
        ax4.set_title('Average Roll Cost', fontsize=12, fontweight='bold')
        for bar, val in zip(bars4, summary_df['Avg Roll Cost ($)']):
            ax4.text(val + 0.5, bar.get_y() + bar.get_height()/2, f'${val:.2f}', va='center', fontsize=9)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Chart saved to: {save_path}")
        
        plt.show()


def main():
    """Main execution function"""
    print("="*80)
    print("INDEX FUTURES ROLL STRATEGY BACKTESTER")
    print("="*80)
    
    # Load configuration
    with open('C:\\Users\\caife\\.minimax-agent\\projects\\2\\config.json', 'r') as f:
        config = json.load(f)
    
    print(f"\nConfiguration loaded:")
    print(f"  Index: {config['index_futures'][0]['name']}")
    print(f"  Period: {config['backtest_parameters']['start_date']} to {config['backtest_parameters']['end_date']}")
    print(f"  Risk-free rate: {config['backtest_parameters']['risk_free_rate']:.2%}")
    
    # Generate simulated Bloomberg data
    print("\n" + "="*60)
    print("GENERATING SIMULATED BLOOMBERG DATA")
    print("="*60)
    
    simulator = BloombergDataSimulator(config)
    futures_data = simulator.generate_futures_data(
        config['backtest_parameters']['start_date'],
        config['backtest_parameters']['end_date']
    )
    
    if len(futures_data) == 0:
        print("ERROR: No data generated!")
        return None, None
    
    # Initialize backtest engine
    backtest = BacktestEngine(futures_data)
    
    # Run strategies
    strategies = [
        (LiquidityRollStrategy(), "Liquidity-Based"),
        (FairValueRollStrategy(deviation_threshold=0.5), "Fair Value"),
        (MomentumRollStrategy(lookback=5), "Momentum"),
        (BenchmarkStrategy(), "Benchmark (Expiry Roll)")
    ]
    
    for strategy, name in strategies:
        backtest.run_strategy(strategy, name)
    
    # Compare strategies
    print("\n" + "="*60)
    print("STRATEGY COMPARISON SUMMARY")
    print("="*60)
    
    summary = backtest.compare_strategies()
    if len(summary) > 0:
        print(summary.to_string(index=False))
    else:
        print("No strategies completed successfully!")
        return None, None
    
    # Determine best strategy
    print("\n" + "="*60)
    print("BEST STRATEGY ANALYSIS")
    print("="*60)
    
    if len(summary) > 0:
        # For roll strategies, we want to MINIMIZE costs (less negative = better)
        # Best = least amount lost = highest Total P&L (least negative)
        best_by_pnl = summary.loc[summary['Total P&L ($)'].idxmax(), 'Strategy']
        
        # Calculate cost savings vs benchmark
        benchmark_pnl = summary[summary['Strategy'] == 'Benchmark (Expiry Roll)']['Total P&L ($)'].values
        if len(benchmark_pnl) > 0:
            benchmark_cost = benchmark_pnl[0]
            summary['Cost Savings vs Benchmark ($)'] = benchmark_cost - summary['Total P&L ($)']
            best_cost_savings = summary.loc[summary['Cost Savings vs Benchmark ($)'].idxmax(), 'Strategy']
        else:
            best_cost_savings = "N/A"
        
        print(f"Best by Total P&L (least loss): {best_by_pnl}")
        print(f"Best Cost Savings vs Benchmark: {best_cost_savings}")
        
        print("\n" + "="*60)
        print("KEY INSIGHT: Interpreting Roll Strategy Performance")
        print("="*60)
        print("""
For futures roll strategies:
- In contango (normal market): You ALWAYS pay to roll (roll cost > 0)
- The goal is to MINIMIZE the roll cost, not make profit
- "Winning" = Getting paid to roll (backwardation) or paying less than benchmark
- "Losing" = Paying more than benchmark to roll

The strategies all show losses because we're in a simulated contango market.
What matters is: Which strategy minimizes costs vs the benchmark?
""")
    
    # Generate visualizations
    print("\n" + "="*60)
    print("GENERATING VISUALIZATIONS")
    print("="*60)
    
    visualizer = ResultsVisualizer(backtest.results)
    
    visualizer.plot_cumulative_pnl('C:\\Users\\caife\\.minimax-agent\\projects\\2\\cumulative_pnl.png')
    visualizer.plot_roll_cost_distribution('C:\\Users\\caife\\.minimax-agent\\projects\\2\\roll_cost_distribution.png')
    visualizer.plot_performance_comparison(summary, 'C:\\Users\\caife\\.minimax-agent\\projects\\2\\performance_comparison.png')
    
    # Save results to CSV
    summary.to_csv('C:\\Users\\caife\\.minimax-agent\\projects\\2\\strategy_comparison.csv', index=False)
    print("\nResults saved to strategy_comparison.csv")
    
    # Final recommendation
    print("\n" + "="*80)
    print("FINAL RECOMMENDATION")
    print("="*80)
    
    if len(summary) > 0:
        # Calculate composite score
        min_pnl = summary['Total P&L ($)'].min()
        max_pnl = summary['Total P&L ($)'].max()
        min_sharpe = summary['Sharpe Ratio'].min()
        max_sharpe = summary['Sharpe Ratio'].max()
        min_wr = summary['Win Rate (%)'].min()
        max_wr = summary['Win Rate (%)'].max()
        
        summary['PNL_Score'] = (summary['Total P&L ($)'] - min_pnl) / (max_pnl - min_pnl + 0.001)
        summary['Sharpe_Score'] = (summary['Sharpe Ratio'] - min_sharpe) / (max_sharpe - min_sharpe + 0.001)
        summary['WinRate_Score'] = (summary['Win Rate (%)'] - min_wr) / (max_wr - min_wr + 0.001)
        summary['Composite_Score'] = (summary['PNL_Score'] + summary['Sharpe_Score'] + summary['WinRate_Score']) / 3
        
        best_overall = summary.loc[summary['Composite_Score'].idxmax(), 'Strategy']
        
        print(f"\nBased on composite scoring (P&L + Sharpe Ratio + Win Rate):")
        print(f"  BEST STRATEGY: {best_overall}")
        print(f"\nThis strategy provides the best risk-adjusted returns for index futures rolling.")
    
    return backtest.results, summary


if __name__ == "__main__":
    results, summary = main()
