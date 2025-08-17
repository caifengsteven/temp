#!/usr/bin/env python3
"""
Simple LSTM-BEKK Demo

A simplified demonstration that focuses on the key components without training.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
from datetime import datetime

# Import key components
from lstm_bekk_trading.data import DataManager, DataProcessor
from lstm_bekk_trading.strategies import GMVPortfolio
from lstm_bekk_trading.risk import PerformanceMetrics
from lstm_bekk_trading.backtesting import BenchmarkModels


def create_demo_data(n_assets=5, n_periods=252):
    """Create demonstration data with realistic return characteristics."""
    np.random.seed(42)

    # Create date index
    dates = pd.date_range(start='2023-01-01', periods=n_periods, freq='D')

    # Realistic parameters for daily returns
    annual_vol = 0.20  # 20% annual volatility
    daily_vol = annual_vol / np.sqrt(252)  # Convert to daily
    annual_return = 0.08  # 8% annual return
    daily_return = annual_return / 252  # Convert to daily

    # Create correlation structure
    correlation_matrix = np.eye(n_assets)
    for i in range(n_assets):
        for j in range(i+1, n_assets):
            corr = np.random.uniform(0.2, 0.6)  # Realistic correlations
            correlation_matrix[i, j] = corr
            correlation_matrix[j, i] = corr

    # Generate returns with volatility clustering
    returns = []
    volatility_state = 1.0  # Initial volatility state

    for t in range(n_periods):
        # GARCH-like volatility clustering
        volatility_state = 0.95 * volatility_state + 0.05 + 0.1 * np.random.normal()**2
        current_vol = daily_vol * np.sqrt(volatility_state)

        # Generate correlated returns
        independent_returns = np.random.normal(daily_return, current_vol, n_assets)

        # Apply correlation structure
        L = np.linalg.cholesky(correlation_matrix)
        correlated_returns = L @ independent_returns

        returns.append(correlated_returns)

    # Create DataFrame
    asset_names = [f'Asset_{i+1}' for i in range(n_assets)]
    returns_df = pd.DataFrame(returns, index=dates, columns=asset_names)

    # Already in decimal form, scale to percentage for display (as in LSTM-BEKK paper)
    returns_df = returns_df * 100

    return returns_df


def demonstrate_data_processing():
    """Demonstrate data processing capabilities."""
    print("\n" + "="*50)
    print("DATA PROCESSING DEMONSTRATION")
    print("="*50)
    
    # Create demo data
    returns_data = create_demo_data()
    print(f"Generated {len(returns_data)} periods for {len(returns_data.columns)} assets")
    
    # Initialize data processor
    processor = DataProcessor()
    
    # Clean data
    cleaned_data = processor.clean_data(returns_data)
    print(f"Data cleaning: {len(returns_data)} -> {len(cleaned_data)} periods")
    
    # Calculate statistics
    stats = processor.get_data_statistics(cleaned_data)
    print(f"Mean returns: {stats['mean_returns'].mean():.4f}")
    print(f"Average volatility: {stats['volatility'].mean():.4f}")
    print(f"Average correlation: {stats['correlation_matrix'].values[np.triu_indices_from(stats['correlation_matrix'].values, k=1)].mean():.3f}")
    
    return cleaned_data


def demonstrate_portfolio_optimization(returns_data):
    """Demonstrate portfolio optimization."""
    print("\n" + "="*50)
    print("PORTFOLIO OPTIMIZATION DEMONSTRATION")
    print("="*50)
    
    # Initialize GMV optimizer
    gmv_optimizer = GMVPortfolio()
    
    # Calculate sample covariance matrix
    cov_matrix = returns_data.cov().values
    print(f"Covariance matrix shape: {cov_matrix.shape}")
    
    # Optimize portfolio
    try:
        weights = gmv_optimizer.optimize_weights(cov_matrix)
        print(f"GMV weights: {weights}")
        print(f"Weights sum: {weights.sum():.6f}")
        
        # Calculate portfolio metrics
        expected_returns = returns_data.mean().values
        metrics = gmv_optimizer.calculate_portfolio_metrics(
            weights, expected_returns, cov_matrix
        )
        
        print(f"Expected return: {metrics['expected_return']:.4f}")
        print(f"Portfolio volatility: {metrics['volatility']:.4f}")
        print(f"Sharpe ratio: {metrics['sharpe_ratio']:.3f}")
        print(f"Effective assets: {metrics['effective_assets']:.1f}")
        
        return weights, metrics
        
    except Exception as e:
        print(f"Optimization failed: {e}")
        # Return equal weights as fallback
        n_assets = len(returns_data.columns)
        weights = np.ones(n_assets) / n_assets
        return weights, {}


def demonstrate_risk_analysis(returns_data, weights):
    """Demonstrate risk analysis."""
    print("\n" + "="*50)
    print("RISK ANALYSIS DEMONSTRATION")
    print("="*50)
    
    # Calculate portfolio returns (convert percentage returns back to decimal)
    portfolio_returns = (returns_data / 100 * weights).sum(axis=1)
    
    # Initialize performance metrics
    perf_metrics = PerformanceMetrics()
    
    # Calculate comprehensive metrics
    metrics = perf_metrics.calculate_comprehensive_metrics(portfolio_returns)
    
    print(f"Total return: {metrics['total_return']:.2%}")
    print(f"Annualized return: {metrics['annualized_return']:.2%}")
    print(f"Annualized volatility: {metrics['annualized_volatility']:.2%}")
    print(f"Sharpe ratio: {metrics['sharpe_ratio']:.3f}")
    print(f"Max drawdown: {metrics['max_drawdown']:.2%}")
    print(f"VaR 5%: {metrics['var_5']:.4f}")
    print(f"ES 5%: {metrics['es_5']:.4f}")
    
    return portfolio_returns, metrics


def demonstrate_benchmark_comparison(returns_data):
    """Demonstrate benchmark model comparison."""
    print("\n" + "="*50)
    print("BENCHMARK COMPARISON DEMONSTRATION")
    print("="*50)
    
    # Initialize benchmark models
    benchmark_models = BenchmarkModels()
    
    # Get benchmark results
    benchmarks = benchmark_models.get_all_benchmarks(returns_data)
    
    # Compare performance
    perf_metrics = PerformanceMetrics()
    comparison_data = []
    
    for name, result in benchmarks.items():
        if 'returns' in result and len(result['returns']) > 0:
            try:
                returns = result['returns']
                # Convert percentage returns to decimal for proper calculation
                if returns.abs().mean() > 1:  # Likely percentage returns
                    returns = returns / 100
                metrics = perf_metrics.calculate_returns_metrics(returns)
                
                comparison_data.append({
                    'Strategy': name,
                    'Total Return': metrics['total_return'],
                    'Annualized Return': metrics['annualized_return'],
                    'Volatility': metrics['annualized_volatility'],
                    'Sharpe Ratio': metrics['sharpe_ratio']
                })
            except Exception as e:
                print(f"Error calculating metrics for {name}: {e}")
    
    # Display comparison
    if comparison_data:
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('Sharpe Ratio', ascending=False)
        
        print("\nBenchmark Performance Comparison:")
        print("-" * 80)
        for _, row in comparison_df.iterrows():
            print(f"{row['Strategy']:20s} | "
                  f"Return: {row['Annualized Return']:7.2%} | "
                  f"Vol: {row['Volatility']:6.2%} | "
                  f"Sharpe: {row['Sharpe Ratio']:6.3f}")
    
    return benchmarks


def create_visualizations(returns_data, portfolio_returns, weights):
    """Create demonstration visualizations."""
    print("\n" + "="*50)
    print("CREATING VISUALIZATIONS")
    print("="*50)
    
    try:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Cumulative returns (portfolio_returns already in decimal)
        cumulative_returns = (1 + portfolio_returns).cumprod()
        axes[0, 0].plot(cumulative_returns.index, cumulative_returns.values)
        axes[0, 0].set_title('Portfolio Cumulative Returns')
        axes[0, 0].set_ylabel('Cumulative Return')
        axes[0, 0].grid(True)
        
        # 2. Portfolio weights
        axes[0, 1].bar(returns_data.columns, weights)
        axes[0, 1].set_title('Portfolio Weights')
        axes[0, 1].set_ylabel('Weight')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. Return distribution (convert to percentage for display)
        axes[1, 0].hist(portfolio_returns * 100, bins=30, alpha=0.7, edgecolor='black')
        axes[1, 0].set_title('Portfolio Return Distribution')
        axes[1, 0].set_xlabel('Daily Return (%)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Correlation heatmap
        corr_matrix = returns_data.corr()
        im = axes[1, 1].imshow(corr_matrix, cmap='RdBu', vmin=-1, vmax=1)
        axes[1, 1].set_title('Asset Correlation Matrix')
        axes[1, 1].set_xticks(range(len(corr_matrix.columns)))
        axes[1, 1].set_yticks(range(len(corr_matrix.columns)))
        axes[1, 1].set_xticklabels(corr_matrix.columns, rotation=45)
        axes[1, 1].set_yticklabels(corr_matrix.columns)
        
        # Add colorbar
        plt.colorbar(im, ax=axes[1, 1])
        
        plt.tight_layout()
        plt.savefig('lstm_bekk_simple_demo.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        print("Visualization saved as 'lstm_bekk_simple_demo.png'")
        
    except Exception as e:
        print(f"Visualization error: {e}")


def main():
    """Main demonstration function."""
    print("="*60)
    print("LSTM-BEKK TRADING SYSTEM - SIMPLE DEMONSTRATION")
    print("="*60)
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    try:
        # 1. Data processing
        returns_data = demonstrate_data_processing()
        
        # 2. Portfolio optimization
        weights, opt_metrics = demonstrate_portfolio_optimization(returns_data)
        
        # 3. Risk analysis
        portfolio_returns, risk_metrics = demonstrate_risk_analysis(returns_data, weights)
        
        # 4. Benchmark comparison
        benchmarks = demonstrate_benchmark_comparison(returns_data)
        
        # 5. Visualizations
        create_visualizations(returns_data, portfolio_returns, weights)
        
        print("\n" + "="*60)
        print("DEMONSTRATION COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("\nKey Results:")
        print(f"- Portfolio Sharpe Ratio: {risk_metrics.get('sharpe_ratio', 0):.3f}")
        print(f"- Max Drawdown: {risk_metrics.get('max_drawdown', 0):.2%}")
        print(f"- Number of benchmarks tested: {len(benchmarks)}")
        print("\nThis demonstrates the core functionality of the LSTM-BEKK system.")
        print("For full model training and advanced features, see the main.py script.")
        
    except Exception as e:
        print(f"Demonstration failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
