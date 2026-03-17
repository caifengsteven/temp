import pandas as pd
import numpy as np
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    print("Installing matplotlib and seaborn...")
    import subprocess
    subprocess.run(['pip', 'install', 'matplotlib', 'seaborn'], check=True)
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True

class EquityCurveAnalyzer:
    def __init__(self):
        """
        Equity curve analysis and visualization
        """
        self.colors = {
            'Rank-Based': '#1f77b4',
            'Mean-Variance': '#ff7f0e', 
            'Maximum Sharpe': '#2ca02c',
            'Risk Parity': '#d62728',
            'Minimum Variance': '#9467bd',
            'Black-Litterman': '#8c564b',
            'Factor-Based': '#e377c2'
        }
        
        # Create output directories
        for dir_name in ['./equity_curves', './performance_charts']:
            if not os.path.exists(dir_name):
                os.makedirs(dir_name)
    
    def load_portfolio_data(self):
        """
        Load portfolio evaluation data
        """
        try:
            evaluation_df = pd.read_csv('./portfolio_results/portfolio_evaluation.csv')
            print(f"Loaded portfolio data: {len(evaluation_df)} records")
            return evaluation_df
        except FileNotFoundError:
            print("Portfolio evaluation data not found. Please run portfolio_optimization.py first.")
            return None
    
    def calculate_equity_curves(self, evaluation_df):
        """
        Calculate cumulative equity curves for each method
        """
        print("Calculating equity curves...")
        
        # Sort by date to ensure proper time series
        evaluation_df['date'] = pd.to_datetime(evaluation_df['date'])
        evaluation_df = evaluation_df.sort_values('date')
        
        equity_curves = {}
        
        for method in evaluation_df['method'].unique():
            method_data = evaluation_df[evaluation_df['method'] == method].copy()
            method_data = method_data.sort_values('date')
            
            # Calculate cumulative returns
            returns = method_data['portfolio_return'].values
            dates = method_data['date'].values
            
            # Start with $100,000 initial capital
            initial_capital = 100000
            equity_curve = [initial_capital]
            
            for ret in returns:
                new_value = equity_curve[-1] * (1 + ret)
                equity_curve.append(new_value)
            
            # Create dates for equity curve (including start date)
            start_date = dates[0] - pd.DateOffset(months=1)
            equity_dates = [start_date] + list(dates)
            
            equity_curves[method] = {
                'dates': equity_dates,
                'equity': equity_curve,
                'returns': [0] + list(returns),
                'final_value': equity_curve[-1],
                'total_return': (equity_curve[-1] / initial_capital - 1) * 100
            }
        
        return equity_curves
    
    def create_equity_curve_chart(self, equity_curves):
        """
        Create comprehensive equity curve visualization
        """
        if not PLOTTING_AVAILABLE:
            print("Plotting not available")
            return
        
        print("Creating equity curve charts...")
        
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Portfolio Performance Analysis', fontsize=16, fontweight='bold')
        
        # 1. Main Equity Curves
        ax1 = axes[0, 0]
        for method, data in equity_curves.items():
            ax1.plot(data['dates'], data['equity'], 
                    label=f"{method} (${data['final_value']:,.0f})", 
                    linewidth=2.5, color=self.colors.get(method, 'gray'))
        
        ax1.set_title('Equity Curves - All Methods', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
        # 2. Returns Distribution
        ax2 = axes[0, 1]
        returns_data = []
        methods = []
        
        for method, data in equity_curves.items():
            returns_data.extend(data['returns'][1:])  # Skip initial 0 return
            methods.extend([method] * (len(data['returns']) - 1))
        
        returns_df = pd.DataFrame({'Method': methods, 'Returns': returns_data})
        
        # Box plot of returns
        box_plot = ax2.boxplot([returns_df[returns_df['Method'] == method]['Returns'].values 
                               for method in equity_curves.keys()], 
                              labels=list(equity_curves.keys()), patch_artist=True)
        
        # Color the boxes
        for patch, method in zip(box_plot['boxes'], equity_curves.keys()):
            patch.set_facecolor(self.colors.get(method, 'gray'))
            patch.set_alpha(0.7)
        
        ax2.set_title('Returns Distribution by Method', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Monthly Returns')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1%}'))
        
        # 3. Drawdown Analysis
        ax3 = axes[1, 0]
        for method, data in equity_curves.items():
            equity = np.array(data['equity'])
            running_max = np.maximum.accumulate(equity)
            drawdown = (equity - running_max) / running_max * 100
            
            ax3.fill_between(data['dates'], drawdown, 0, 
                           alpha=0.6, color=self.colors.get(method, 'gray'), 
                           label=f"{method} (Max: {np.min(drawdown):.1f}%)")
        
        ax3.set_title('Drawdown Analysis', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Date')
        ax3.set_ylabel('Drawdown (%)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Performance Metrics Comparison
        ax4 = axes[1, 1]
        
        # Calculate key metrics
        metrics_data = []
        for method, data in equity_curves.items():
            returns = np.array(data['returns'][1:])
            
            total_return = data['total_return']
            volatility = np.std(returns) * np.sqrt(12) * 100  # Annualized %
            sharpe = (np.mean(returns) * 12 - 0.03) / (np.std(returns) * np.sqrt(12))
            
            metrics_data.append({
                'Method': method,
                'Total Return (%)': total_return,
                'Volatility (%)': volatility,
                'Sharpe Ratio': sharpe
            })
        
        metrics_df = pd.DataFrame(metrics_data)
        
        # Create grouped bar chart
        x = np.arange(len(metrics_df))
        width = 0.25
        
        bars1 = ax4.bar(x - width, metrics_df['Total Return (%)'], width, 
                       label='Total Return (%)', alpha=0.8, color='skyblue')
        bars2 = ax4.bar(x, metrics_df['Volatility (%)'], width, 
                       label='Volatility (%)', alpha=0.8, color='lightcoral')
        bars3 = ax4.bar(x + width, metrics_df['Sharpe Ratio'] * 10, width, 
                       label='Sharpe Ratio (×10)', alpha=0.8, color='lightgreen')
        
        ax4.set_title('Performance Metrics Comparison', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Portfolio Method')
        ax4.set_ylabel('Value')
        ax4.set_xticks(x)
        ax4.set_xticklabels(metrics_df['Method'], rotation=45, ha='right')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                ax4.annotate(f'{height:.1f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),  # 3 points vertical offset
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.savefig('./equity_curves/comprehensive_equity_analysis.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        return metrics_df
    
    def create_individual_equity_curves(self, equity_curves):
        """
        Create individual equity curve charts for each method
        """
        if not PLOTTING_AVAILABLE:
            return
        
        print("Creating individual equity curve charts...")
        
        for method, data in equity_curves.items():
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
            
            # Equity curve
            ax1.plot(data['dates'], data['equity'], 
                    linewidth=3, color=self.colors.get(method, 'blue'))
            ax1.fill_between(data['dates'], data['equity'], 100000, 
                           alpha=0.3, color=self.colors.get(method, 'blue'))
            
            ax1.set_title(f'{method} - Equity Curve', fontsize=16, fontweight='bold')
            ax1.set_ylabel('Portfolio Value ($)')
            ax1.grid(True, alpha=0.3)
            ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
            
            # Add performance text
            final_value = data['final_value']
            total_return = data['total_return']
            ax1.text(0.02, 0.98, f'Final Value: ${final_value:,.0f}\nTotal Return: {total_return:.1f}%', 
                    transform=ax1.transAxes, fontsize=12, fontweight='bold',
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # Monthly returns
            monthly_returns = data['returns'][1:]  # Skip initial 0
            ax2.bar(data['dates'][1:], [r * 100 for r in monthly_returns], 
                   color=[self.colors.get(method, 'blue') if r >= 0 else 'red' for r in monthly_returns],
                   alpha=0.7)
            
            ax2.set_title(f'{method} - Monthly Returns', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Date')
            ax2.set_ylabel('Monthly Return (%)')
            ax2.grid(True, alpha=0.3)
            ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
            
            plt.tight_layout()
            plt.savefig(f'./equity_curves/{method.replace(" ", "_").replace("-", "_")}_equity_curve.png', 
                       dpi=300, bbox_inches='tight')
            plt.close()
    
    def create_performance_summary_table(self, equity_curves):
        """
        Create detailed performance summary table
        """
        print("Creating performance summary table...")
        
        summary_data = []
        
        for method, data in equity_curves.items():
            returns = np.array(data['returns'][1:])
            
            # Calculate comprehensive metrics
            initial_capital = 100000
            final_value = data['final_value']
            total_return = (final_value / initial_capital - 1) * 100
            
            # Annualized metrics
            num_periods = len(returns)
            annualized_return = ((final_value / initial_capital) ** (12 / num_periods) - 1) * 100
            volatility = np.std(returns) * np.sqrt(12) * 100
            
            # Risk metrics
            sharpe_ratio = (annualized_return / 100 - 0.03) / (volatility / 100)
            
            # Drawdown
            equity = np.array(data['equity'])
            running_max = np.maximum.accumulate(equity)
            drawdowns = (equity - running_max) / running_max
            max_drawdown = np.min(drawdowns) * 100
            
            # Win rate
            win_rate = np.mean(returns > 0) * 100
            
            # Best and worst months
            best_month = np.max(returns) * 100
            worst_month = np.min(returns) * 100
            
            summary_data.append({
                'Method': method,
                'Initial Capital': f'${initial_capital:,}',
                'Final Value': f'${final_value:,.0f}',
                'Total Return (%)': f'{total_return:.1f}%',
                'Annualized Return (%)': f'{annualized_return:.1f}%',
                'Volatility (%)': f'{volatility:.1f}%',
                'Sharpe Ratio': f'{sharpe_ratio:.2f}',
                'Max Drawdown (%)': f'{max_drawdown:.1f}%',
                'Win Rate (%)': f'{win_rate:.1f}%',
                'Best Month (%)': f'{best_month:.1f}%',
                'Worst Month (%)': f'{worst_month:.1f}%',
                'Number of Periods': num_periods
            })
        
        summary_df = pd.DataFrame(summary_data)
        
        # Save to CSV
        summary_df.to_csv('./equity_curves/performance_summary_table.csv', index=False)
        
        # Print formatted table
        print("\n" + "="*120)
        print("COMPREHENSIVE PORTFOLIO PERFORMANCE SUMMARY")
        print("="*120)
        print(summary_df.to_string(index=False))
        
        return summary_df

def main():
    """
    Main equity curve analysis function
    """
    print("EQUITY CURVE ANALYSIS")
    print("=" * 50)
    
    # Initialize analyzer
    analyzer = EquityCurveAnalyzer()
    
    # Load portfolio data
    evaluation_df = analyzer.load_portfolio_data()
    if evaluation_df is None:
        return
    
    # Calculate equity curves
    equity_curves = analyzer.calculate_equity_curves(evaluation_df)
    
    # Create comprehensive charts
    metrics_df = analyzer.create_equity_curve_chart(equity_curves)
    
    # Create individual charts
    analyzer.create_individual_equity_curves(equity_curves)
    
    # Create performance summary
    summary_df = analyzer.create_performance_summary_table(equity_curves)
    
    print("\n" + "="*80)
    print("EQUITY CURVE ANALYSIS COMPLETED!")
    print("="*80)
    print("Charts saved to:")
    print("- ./equity_curves/comprehensive_equity_analysis.png")
    print("- ./equity_curves/[Method]_equity_curve.png (individual charts)")
    print("- ./equity_curves/performance_summary_table.csv")
    
    # Show best performing method
    best_method = summary_df.loc[summary_df['Total Return (%)'].str.rstrip('%').astype(float).idxmax()]
    print(f"\nBest Performing Method: {best_method['Method']}")
    print(f"Final Value: {best_method['Final Value']}")
    print(f"Total Return: {best_method['Total Return (%)']}")

if __name__ == "__main__":
    main()
