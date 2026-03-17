import pandas as pd
import numpy as np
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def load_and_analyze_equity_curves():
    """
    Load portfolio data and calculate equity curves
    """
    try:
        evaluation_df = pd.read_csv('./portfolio_results/portfolio_evaluation.csv')
        print(f"Loaded portfolio data: {len(evaluation_df)} records")
    except FileNotFoundError:
        print("Portfolio evaluation data not found. Please run portfolio_optimization.py first.")
        return None
    
    # Sort by date
    evaluation_df['date'] = pd.to_datetime(evaluation_df['date'])
    evaluation_df = evaluation_df.sort_values('date')
    
    print("\nCalculating equity curves...")
    
    # Initial capital
    initial_capital = 100000
    
    equity_results = {}
    
    for method in evaluation_df['method'].unique():
        method_data = evaluation_df[evaluation_df['method'] == method].copy()
        method_data = method_data.sort_values('date')
        
        # Calculate cumulative performance
        returns = method_data['portfolio_return'].values
        dates = method_data['date'].values
        
        # Calculate equity curve
        equity_values = [initial_capital]
        for ret in returns:
            new_value = equity_values[-1] * (1 + ret)
            equity_values.append(new_value)
        
        final_value = equity_values[-1]
        total_return = (final_value / initial_capital - 1) * 100
        
        # Calculate metrics
        annualized_return = ((final_value / initial_capital) ** (12 / len(returns)) - 1) * 100
        volatility = np.std(returns) * np.sqrt(12) * 100
        sharpe_ratio = (annualized_return / 100 - 0.03) / (volatility / 100) if volatility > 0 else 0
        
        # Drawdown calculation
        running_max = np.maximum.accumulate(equity_values)
        drawdowns = [(eq - rm) / rm for eq, rm in zip(equity_values, running_max)]
        max_drawdown = min(drawdowns) * 100
        
        # Win rate
        win_rate = np.mean(np.array(returns) > 0) * 100
        
        equity_results[method] = {
            'dates': list(dates),
            'returns': list(returns),
            'equity_values': equity_values,
            'final_value': final_value,
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'best_month': max(returns) * 100,
            'worst_month': min(returns) * 100
        }
    
    return equity_results

def create_equity_table(equity_results):
    """
    Create detailed equity curve table
    """
    print("\n" + "="*120)
    print("PORTFOLIO EQUITY CURVE ANALYSIS")
    print("="*120)
    
    # Create summary table
    summary_data = []
    
    for method, data in equity_results.items():
        summary_data.append({
            'Method': method,
            'Initial Capital': f'${100000:,}',
            'Final Value': f'${data["final_value"]:,.0f}',
            'Total Return': f'{data["total_return"]:.1f}%',
            'Annualized Return': f'{data["annualized_return"]:.1f}%',
            'Volatility': f'{data["volatility"]:.1f}%',
            'Sharpe Ratio': f'{data["sharpe_ratio"]:.2f}',
            'Max Drawdown': f'{data["max_drawdown"]:.1f}%',
            'Win Rate': f'{data["win_rate"]:.1f}%',
            'Best Month': f'{data["best_month"]:.1f}%',
            'Worst Month': f'{data["worst_month"]:.1f}%'
        })
    
    summary_df = pd.DataFrame(summary_data)
    
    # Sort by total return
    summary_df['Total Return Numeric'] = summary_df['Total Return'].str.rstrip('%').astype(float)
    summary_df = summary_df.sort_values('Total Return Numeric', ascending=False)
    summary_df = summary_df.drop('Total Return Numeric', axis=1)
    
    print(summary_df.to_string(index=False))
    
    # Save to file
    summary_df.to_csv('./equity_curves/equity_performance_summary.csv', index=False)
    
    return summary_df

def create_monthly_performance_table(equity_results):
    """
    Create month-by-month performance table
    """
    print("\n" + "="*100)
    print("MONTHLY PERFORMANCE BREAKDOWN")
    print("="*100)
    
    # Get all unique dates
    all_dates = set()
    for data in equity_results.values():
        all_dates.update(data['dates'])
    
    all_dates = sorted(list(all_dates))
    
    # Create monthly performance table
    monthly_data = []
    
    for date in all_dates:
        row = {'Date': date.strftime('%Y-%m')}
        
        for method, data in equity_results.items():
            if date in data['dates']:
                idx = data['dates'].index(date)
                return_pct = data['returns'][idx] * 100
                row[method] = f'{return_pct:.2f}%'
            else:
                row[method] = 'N/A'
        
        monthly_data.append(row)
    
    monthly_df = pd.DataFrame(monthly_data)
    print(monthly_df.to_string(index=False))
    
    # Save to file
    monthly_df.to_csv('./equity_curves/monthly_performance.csv', index=False)
    
    return monthly_df

def create_equity_progression_table(equity_results):
    """
    Show equity progression over time
    """
    print("\n" + "="*100)
    print("EQUITY PROGRESSION (Portfolio Values)")
    print("="*100)
    
    # Get all unique dates
    all_dates = set()
    for data in equity_results.values():
        all_dates.update(data['dates'])
    
    all_dates = sorted(list(all_dates))
    
    # Create equity progression table
    equity_data = []
    
    # Add initial values
    initial_row = {'Date': 'Initial'}
    for method in equity_results.keys():
        initial_row[method] = '$100,000'
    equity_data.append(initial_row)
    
    # Add monthly values
    for date in all_dates:
        row = {'Date': date.strftime('%Y-%m')}
        
        for method, data in equity_results.items():
            if date in data['dates']:
                idx = data['dates'].index(date)
                equity_value = data['equity_values'][idx + 1]  # +1 because equity_values includes initial
                row[method] = f'${equity_value:,.0f}'
            else:
                row[method] = 'N/A'
        
        equity_data.append(row)
    
    equity_df = pd.DataFrame(equity_data)
    print(equity_df.to_string(index=False))
    
    # Save to file
    equity_df.to_csv('./equity_curves/equity_progression.csv', index=False)
    
    return equity_df

def main():
    """
    Main analysis function
    """
    print("SIMPLE EQUITY CURVE ANALYSIS")
    print("=" * 50)
    
    # Create output directory
    if not os.path.exists('./equity_curves'):
        os.makedirs('./equity_curves')
    
    # Load and analyze data
    equity_results = load_and_analyze_equity_curves()
    
    if equity_results is None:
        return
    
    # Create analysis tables
    summary_df = create_equity_table(equity_results)
    monthly_df = create_monthly_performance_table(equity_results)
    equity_df = create_equity_progression_table(equity_results)
    
    # Find best performer
    best_method = max(equity_results.keys(), 
                     key=lambda x: equity_results[x]['total_return'])
    
    best_data = equity_results[best_method]
    
    print("\n" + "="*80)
    print("BEST PERFORMING PORTFOLIO")
    print("="*80)
    print(f"Method: {best_method}")
    print(f"Final Value: ${best_data['final_value']:,.0f}")
    print(f"Total Return: {best_data['total_return']:.1f}%")
    print(f"Annualized Return: {best_data['annualized_return']:.1f}%")
    print(f"Sharpe Ratio: {best_data['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {best_data['max_drawdown']:.1f}%")
    print(f"Win Rate: {best_data['win_rate']:.1f}%")
    
    print("\n" + "="*80)
    print("FILES GENERATED:")
    print("="*80)
    print("- ./equity_curves/equity_performance_summary.csv")
    print("- ./equity_curves/monthly_performance.csv") 
    print("- ./equity_curves/equity_progression.csv")
    print("- ./equity_curves/comprehensive_equity_analysis.png (if matplotlib available)")
    
    print("\n" + "="*80)
    print("EQUITY CURVE ANALYSIS COMPLETED!")
    print("="*80)

if __name__ == "__main__":
    main()
