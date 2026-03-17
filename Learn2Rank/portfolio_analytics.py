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

class PortfolioAnalytics:
    def __init__(self):
        """
        Advanced portfolio analytics and visualization
        """
        self.risk_free_rate = 0.03
        
        # Create output directories
        for dir_name in ['./portfolio_plots', './portfolio_analytics']:
            if not os.path.exists(dir_name):
                os.makedirs(dir_name)
    
    def calculate_portfolio_metrics(self, returns_series):
        """
        Calculate comprehensive portfolio performance metrics
        """
        returns = np.array(returns_series)
        
        # Basic metrics
        total_return = np.prod(1 + returns) - 1
        annualized_return = (1 + total_return) ** (12 / len(returns)) - 1
        volatility = np.std(returns) * np.sqrt(12)
        sharpe_ratio = (annualized_return - self.risk_free_rate) / volatility
        
        # Risk metrics
        downside_returns = returns[returns < 0]
        downside_deviation = np.std(downside_returns) * np.sqrt(12) if len(downside_returns) > 0 else 0
        sortino_ratio = (annualized_return - self.risk_free_rate) / downside_deviation if downside_deviation > 0 else np.inf
        
        # Drawdown metrics
        cumulative_returns = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = (cumulative_returns - running_max) / running_max
        max_drawdown = np.min(drawdowns)
        
        # Win rate
        win_rate = np.mean(returns > 0)
        
        # Value at Risk (95% confidence)
        var_95 = np.percentile(returns, 5)
        
        # Calmar ratio
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else np.inf
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'var_95': var_95,
            'calmar_ratio': calmar_ratio,
            'downside_deviation': downside_deviation
        }
    
    def create_performance_comparison(self, evaluation_df):
        """
        Create comprehensive performance comparison
        """
        print("Creating performance comparison...")
        
        # Calculate metrics for each method
        performance_metrics = []
        
        for method in evaluation_df['method'].unique():
            method_data = evaluation_df[evaluation_df['method'] == method]
            returns = method_data['portfolio_return'].values
            
            metrics = self.calculate_portfolio_metrics(returns)
            metrics['method'] = method
            metrics['num_periods'] = len(returns)
            
            performance_metrics.append(metrics)
        
        metrics_df = pd.DataFrame(performance_metrics)
        
        # Save metrics
        metrics_df.to_csv('./portfolio_analytics/performance_metrics.csv', index=False)
        
        # Print comparison table
        print("\n" + "="*100)
        print("COMPREHENSIVE PORTFOLIO PERFORMANCE COMPARISON")
        print("="*100)
        
        display_cols = ['method', 'annualized_return', 'volatility', 'sharpe_ratio', 
                       'sortino_ratio', 'max_drawdown', 'win_rate', 'calmar_ratio']
        
        display_df = metrics_df[display_cols].round(4)
        print(display_df.to_string(index=False))
        
        return metrics_df
    
    def create_risk_return_analysis(self, metrics_df):
        """
        Create risk-return analysis and plots
        """
        if not PLOTTING_AVAILABLE:
            print("Plotting not available - skipping visualizations")
            return
        
        # Risk-Return scatter plot
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        plt.scatter(metrics_df['volatility'], metrics_df['annualized_return'], 
                   s=100, alpha=0.7, c=metrics_df['sharpe_ratio'], cmap='viridis')
        plt.colorbar(label='Sharpe Ratio')
        plt.xlabel('Volatility (Annualized)')
        plt.ylabel('Return (Annualized)')
        plt.title('Risk-Return Profile')
        
        # Add method labels
        for i, row in metrics_df.iterrows():
            plt.annotate(row['method'], (row['volatility'], row['annualized_return']),
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        # Sharpe ratio comparison
        plt.subplot(2, 2, 2)
        plt.bar(metrics_df['method'], metrics_df['sharpe_ratio'])
        plt.title('Sharpe Ratio Comparison')
        plt.xticks(rotation=45)
        plt.ylabel('Sharpe Ratio')
        
        # Maximum drawdown comparison
        plt.subplot(2, 2, 3)
        plt.bar(metrics_df['method'], metrics_df['max_drawdown'])
        plt.title('Maximum Drawdown Comparison')
        plt.xticks(rotation=45)
        plt.ylabel('Max Drawdown')
        
        # Win rate comparison
        plt.subplot(2, 2, 4)
        plt.bar(metrics_df['method'], metrics_df['win_rate'])
        plt.title('Win Rate Comparison')
        plt.xticks(rotation=45)
        plt.ylabel('Win Rate')
        
        plt.tight_layout()
        plt.savefig('./portfolio_plots/risk_return_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Risk-return analysis saved to './portfolio_plots/risk_return_analysis.png'")
    
    def create_portfolio_composition_analysis(self, composition_df):
        """
        Analyze portfolio composition patterns
        """
        print("Analyzing portfolio composition...")
        
        # Top holdings by method
        top_holdings_by_method = {}
        
        for method in composition_df['method'].unique():
            method_data = composition_df[composition_df['method'] == method]
            top_holdings = method_data.groupby('ticker')['weight'].agg(['mean', 'count', 'std']).round(4)
            top_holdings = top_holdings.sort_values('mean', ascending=False).head(10)
            top_holdings_by_method[method] = top_holdings
        
        # Save top holdings analysis
        with open('./portfolio_analytics/top_holdings_analysis.txt', 'w') as f:
            f.write("TOP HOLDINGS ANALYSIS BY METHOD\n")
            f.write("="*50 + "\n\n")
            
            for method, holdings in top_holdings_by_method.items():
                f.write(f"{method}:\n")
                f.write("-" * 30 + "\n")
                f.write(holdings.to_string())
                f.write("\n\n")
        
        # Concentration analysis
        concentration_analysis = composition_df.groupby(['method', 'date']).apply(
            lambda x: (x['weight'] ** 2).sum()  # Herfindahl index
        ).reset_index(name='concentration')
        
        concentration_summary = concentration_analysis.groupby('method')['concentration'].agg(['mean', 'std']).round(4)
        
        print("\nPortfolio Concentration Analysis (Herfindahl Index):")
        print("-" * 60)
        print(concentration_summary)
        
        return top_holdings_by_method, concentration_summary
    
    def create_performance_attribution(self, evaluation_df, composition_df):
        """
        Performance attribution analysis
        """
        print("Creating performance attribution analysis...")
        
        # Merge evaluation and composition data
        merged_data = evaluation_df.merge(
            composition_df.groupby(['date', 'method']).agg({
                'weight': ['count', 'std'],
                'ticker': lambda x: len(x.unique())
            }).round(4),
            on=['date', 'method']
        )
        
        # Analyze relationship between portfolio characteristics and performance
        attribution_results = {}
        
        for method in evaluation_df['method'].unique():
            method_data = merged_data[merged_data['method'] == method]
            
            if len(method_data) > 1:
                # Correlation between concentration and performance
                concentration_corr = np.corrcoef(
                    method_data['weight_concentration'], 
                    method_data['portfolio_return']
                )[0, 1]
                
                # Correlation between number of stocks and performance
                num_stocks_corr = np.corrcoef(
                    method_data['num_stocks'], 
                    method_data['portfolio_return']
                )[0, 1]
                
                attribution_results[method] = {
                    'concentration_correlation': concentration_corr,
                    'num_stocks_correlation': num_stocks_corr,
                    'avg_return': method_data['portfolio_return'].mean(),
                    'return_volatility': method_data['portfolio_return'].std()
                }
        
        attribution_df = pd.DataFrame(attribution_results).T
        attribution_df.to_csv('./portfolio_analytics/performance_attribution.csv')
        
        print("\nPerformance Attribution Analysis:")
        print("-" * 50)
        print(attribution_df.round(4))
        
        return attribution_df
    
    def generate_executive_summary(self, metrics_df, evaluation_df):
        """
        Generate executive summary report
        """
        print("Generating executive summary...")
        
        # Find best performing method
        best_sharpe = metrics_df.loc[metrics_df['sharpe_ratio'].idxmax()]
        best_return = metrics_df.loc[metrics_df['annualized_return'].idxmax()]
        best_risk_adj = metrics_df.loc[metrics_df['calmar_ratio'].idxmax()]
        
        # Overall statistics
        total_periods = len(evaluation_df['date'].unique())
        total_methods = len(evaluation_df['method'].unique())
        
        summary_report = f"""
PORTFOLIO OPTIMIZATION EXECUTIVE SUMMARY
{'='*60}

ANALYSIS OVERVIEW:
- Total Time Periods Analyzed: {total_periods}
- Portfolio Optimization Methods: {total_methods}
- Analysis Period: {evaluation_df['date'].min()} to {evaluation_df['date'].max()}

TOP PERFORMING STRATEGIES:

1. HIGHEST SHARPE RATIO:
   Method: {best_sharpe['method']}
   Sharpe Ratio: {best_sharpe['sharpe_ratio']:.4f}
   Annualized Return: {best_sharpe['annualized_return']:.2%}
   Volatility: {best_sharpe['volatility']:.2%}
   Max Drawdown: {best_sharpe['max_drawdown']:.2%}

2. HIGHEST RETURN:
   Method: {best_return['method']}
   Annualized Return: {best_return['annualized_return']:.2%}
   Sharpe Ratio: {best_return['sharpe_ratio']:.4f}
   Volatility: {best_return['volatility']:.2%}

3. BEST RISK-ADJUSTED (Calmar Ratio):
   Method: {best_risk_adj['method']}
   Calmar Ratio: {best_risk_adj['calmar_ratio']:.4f}
   Annualized Return: {best_risk_adj['annualized_return']:.2%}
   Max Drawdown: {best_risk_adj['max_drawdown']:.2%}

OVERALL INSIGHTS:
- Average Win Rate Across Methods: {metrics_df['win_rate'].mean():.2%}
- Best Overall Win Rate: {metrics_df['win_rate'].max():.2%} ({metrics_df.loc[metrics_df['win_rate'].idxmax(), 'method']})
- Lowest Volatility: {metrics_df['volatility'].min():.2%} ({metrics_df.loc[metrics_df['volatility'].idxmin(), 'method']})
- Highest Volatility: {metrics_df['volatility'].max():.2%} ({metrics_df.loc[metrics_df['volatility'].idxmax(), 'method']})

RECOMMENDATIONS:
1. For Risk-Averse Investors: Consider {metrics_df.loc[metrics_df['volatility'].idxmin(), 'method']} (lowest volatility)
2. For Return-Focused Investors: Consider {best_return['method']} (highest returns)
3. For Balanced Approach: Consider {best_sharpe['method']} (best risk-adjusted returns)

Files Generated:
- Performance Metrics: ./portfolio_analytics/performance_metrics.csv
- Portfolio Compositions: ./portfolio_results/portfolio_compositions.csv
- Risk-Return Analysis: ./portfolio_plots/risk_return_analysis.png
- Top Holdings Analysis: ./portfolio_analytics/top_holdings_analysis.txt
"""
        
        # Save summary report
        with open('./portfolio_analytics/executive_summary.txt', 'w') as f:
            f.write(summary_report)
        
        print(summary_report)
        
        return summary_report

def main():
    """
    Main analytics function
    """
    print("PORTFOLIO ANALYTICS AND REPORTING")
    print("=" * 50)
    
    # Initialize analytics
    analytics = PortfolioAnalytics()
    
    # Load evaluation data
    try:
        evaluation_df = pd.read_csv('./portfolio_results/portfolio_evaluation.csv')
        composition_df = pd.read_csv('./portfolio_results/portfolio_compositions.csv')
    except FileNotFoundError:
        print("Portfolio evaluation data not found. Please run portfolio_optimization.py first.")
        return
    
    # Create comprehensive analysis
    metrics_df = analytics.create_performance_comparison(evaluation_df)
    analytics.create_risk_return_analysis(metrics_df)
    
    top_holdings, concentration = analytics.create_portfolio_composition_analysis(composition_df)
    attribution = analytics.create_performance_attribution(evaluation_df, composition_df)
    
    # Generate executive summary
    summary = analytics.generate_executive_summary(metrics_df, evaluation_df)
    
    print("\n" + "="*60)
    print("PORTFOLIO ANALYTICS COMPLETED!")
    print("="*60)
    print("All analysis files saved to ./portfolio_analytics/")

if __name__ == "__main__":
    main()
