import pandas as pd
import numpy as np
import torch
import mysql.connector
from datetime import datetime, timedelta
import os
from tqdm import tqdm
import warnings
import copy
from scipy.optimize import minimize
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    print("Matplotlib/Seaborn not available. Plots will be skipped.")
warnings.filterwarnings('ignore')

# Import the existing algorithms
from algs.ranknet import RankNet, pairwise_data
from algs.listnet import ListNet, ListMLE_loss, ndcg
import xgboost as xgb

class PortfolioOptimizer:
    def __init__(self, db_host='localhost', db_user='root', db_password='352471Cf', db_name='yuqerdata'):
        """
        Portfolio optimization system on top of Learn2Rank predictions
        """
        self.db_host = db_host
        self.db_user = db_user
        self.db_password = db_password
        self.db_name = db_name
        self.connection = None
        
        # Portfolio parameters
        self.max_position_size = 0.05  # Max 5% per stock
        self.min_position_size = 0.01  # Min 1% per stock
        self.max_stocks = 50  # Maximum number of stocks in portfolio
        self.risk_free_rate = 0.03  # 3% annual risk-free rate
        
        # Create output directories
        for dir_name in ['./portfolio_results', './portfolio_plots']:
            if not os.path.exists(dir_name):
                os.makedirs(dir_name)
    
    def connect_database(self):
        """Connect to MySQL database"""
        try:
            self.connection = mysql.connector.connect(
                host=self.db_host,
                user=self.db_user,
                password=self.db_password,
                database=self.db_name
            )
            print(f"Successfully connected to database: {self.db_name}")
            return True
        except mysql.connector.Error as err:
            print(f"Error connecting to database: {err}")
            return False
    
    def load_ranking_predictions(self, predictions_folder='./comprehensive_predictions'):
        """
        Load ranking predictions from Learn2Rank models
        """
        print("Loading ranking predictions...")
        
        all_predictions = {}
        
        # Load LambdaMART predictions (best performing)
        lambda_files = [f for f in os.listdir(predictions_folder) if f.startswith('LambdaMART')]
        
        for file in lambda_files:
            # Extract date from filename
            date_str = file.split('_')[1]
            
            # Load predictions
            df = pd.read_csv(os.path.join(predictions_folder, file))
            
            # Sort by prediction score
            df_sorted = df.sort_values('pred', ascending=False)
            
            all_predictions[date_str] = {
                'predictions': df_sorted,
                'top_stocks': df_sorted.head(100),  # Top 100 for portfolio selection
                'date': date_str
            }
        
        print(f"Loaded predictions for {len(all_predictions)} periods")
        return all_predictions
    
    def calculate_risk_metrics(self, returns_data, lookback_window=60):
        """
        Calculate risk metrics for portfolio optimization
        """
        # Calculate covariance matrix
        cov_matrix = returns_data.cov() * 252  # Annualized
        
        # Calculate volatilities
        volatilities = returns_data.std() * np.sqrt(252)  # Annualized
        
        # Calculate correlation matrix
        corr_matrix = returns_data.corr()
        
        return {
            'covariance_matrix': cov_matrix,
            'volatilities': volatilities,
            'correlation_matrix': corr_matrix
        }
    
    def mean_variance_optimization(self, expected_returns, cov_matrix, risk_aversion=1.0):
        """
        Mean-variance optimization (Markowitz)
        """
        n_assets = len(expected_returns)
        
        # Objective function: maximize utility = expected_return - 0.5 * risk_aversion * variance
        def objective(weights):
            portfolio_return = np.dot(weights, expected_returns)
            portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
            return -(portfolio_return - 0.5 * risk_aversion * portfolio_variance)
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0},  # Weights sum to 1
        ]
        
        # Bounds for each weight
        bounds = [(self.min_position_size, self.max_position_size) for _ in range(n_assets)]
        
        # Initial guess
        x0 = np.array([1.0 / n_assets] * n_assets)
        
        # Optimize
        result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
        
        if result.success:
            return result.x
        else:
            print("Optimization failed, using equal weights")
            return np.array([1.0 / n_assets] * n_assets)
    
    def risk_parity_optimization(self, cov_matrix):
        """
        Risk parity optimization - equal risk contribution
        """
        n_assets = len(cov_matrix)
        
        def risk_budget_objective(weights):
            portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            marginal_contrib = np.dot(cov_matrix, weights) / portfolio_vol
            contrib = weights * marginal_contrib
            target_contrib = portfolio_vol / n_assets
            return np.sum((contrib - target_contrib) ** 2)
        
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}]
        bounds = [(self.min_position_size, self.max_position_size) for _ in range(n_assets)]
        x0 = np.array([1.0 / n_assets] * n_assets)
        
        result = minimize(risk_budget_objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
        
        if result.success:
            return result.x
        else:
            return np.array([1.0 / n_assets] * n_assets)
    
    def maximum_sharpe_optimization(self, expected_returns, cov_matrix):
        """
        Maximum Sharpe ratio optimization
        """
        n_assets = len(expected_returns)
        
        def negative_sharpe(weights):
            portfolio_return = np.dot(weights, expected_returns)
            portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            return -(portfolio_return - self.risk_free_rate) / portfolio_vol
        
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}]
        bounds = [(self.min_position_size, self.max_position_size) for _ in range(n_assets)]
        x0 = np.array([1.0 / n_assets] * n_assets)
        
        result = minimize(negative_sharpe, x0, method='SLSQP', bounds=bounds, constraints=constraints)
        
        if result.success:
            return result.x
        else:
            return np.array([1.0 / n_assets] * n_assets)
    
    def rank_based_optimization(self, predictions, top_n=30):
        """
        Simple rank-based portfolio construction
        """
        # Select top N stocks
        top_stocks = predictions.head(top_n)

        # Weight by prediction score (normalized)
        scores = top_stocks['pred'].values
        scores_normalized = (scores - scores.min()) / (scores.max() - scores.min())

        # Apply softmax for smoother weights
        exp_scores = np.exp(scores_normalized * 2)  # Temperature = 2
        weights = exp_scores / exp_scores.sum()

        # Apply position size constraints
        weights = np.clip(weights, self.min_position_size, self.max_position_size)
        weights = weights / weights.sum()  # Renormalize

        return top_stocks['ticker'].values, weights

    def factor_based_optimization(self, predictions, factor_columns, top_n=30):
        """
        Factor-based portfolio optimization using factor exposures
        """
        top_stocks = predictions.head(top_n)

        # Extract factor exposures
        factor_data = top_stocks[factor_columns].fillna(0)

        # Normalize factors
        factor_data_norm = (factor_data - factor_data.mean()) / factor_data.std()

        # Calculate factor scores (simple equal-weighted combination)
        factor_score = factor_data_norm.mean(axis=1)

        # Weight by factor score
        scores = factor_score.values
        scores_normalized = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)

        # Apply softmax
        exp_scores = np.exp(scores_normalized * 1.5)
        weights = exp_scores / exp_scores.sum()

        # Apply constraints
        weights = np.clip(weights, self.min_position_size, self.max_position_size)
        weights = weights / weights.sum()

        return top_stocks['ticker'].values, weights

    def minimum_variance_optimization(self, cov_matrix):
        """
        Minimum variance portfolio optimization
        """
        n_assets = len(cov_matrix)

        def objective(weights):
            return np.dot(weights.T, np.dot(cov_matrix, weights))

        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}]
        bounds = [(self.min_position_size, self.max_position_size) for _ in range(n_assets)]
        x0 = np.array([1.0 / n_assets] * n_assets)

        result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)

        if result.success:
            return result.x
        else:
            return np.array([1.0 / n_assets] * n_assets)

    def black_litterman_optimization(self, expected_returns, cov_matrix, market_caps=None):
        """
        Black-Litterman portfolio optimization
        """
        n_assets = len(expected_returns)

        # If no market caps provided, use equal weights as market portfolio
        if market_caps is None:
            market_weights = np.array([1.0 / n_assets] * n_assets)
        else:
            market_weights = market_caps / market_caps.sum()

        # Risk aversion parameter (estimated from market portfolio)
        market_variance = np.dot(market_weights.T, np.dot(cov_matrix, market_weights))
        market_return = np.dot(market_weights, expected_returns)
        risk_aversion = market_return / market_variance

        # Implied equilibrium returns
        pi = risk_aversion * np.dot(cov_matrix, market_weights)

        # Black-Litterman with views (using predicted returns as views)
        tau = 0.025  # Scaling factor
        P = np.eye(n_assets)  # View matrix (views on all assets)
        Q = expected_returns  # View returns (our predictions)
        omega = np.diag(np.diag(tau * cov_matrix))  # View uncertainty

        # Black-Litterman formula
        M1 = np.linalg.inv(tau * cov_matrix)
        M2 = np.dot(P.T, np.dot(np.linalg.inv(omega), P))
        M3 = np.dot(np.linalg.inv(tau * cov_matrix), pi)
        M4 = np.dot(P.T, np.dot(np.linalg.inv(omega), Q))

        mu_bl = np.dot(np.linalg.inv(M1 + M2), M3 + M4)
        cov_bl = np.linalg.inv(M1 + M2)

        # Optimize with Black-Litterman inputs
        return self.mean_variance_optimization(mu_bl, cov_bl, risk_aversion)
    
    def build_portfolios(self, predictions_data, method='all'):
        """
        Build portfolios using different optimization methods
        """
        print("Building optimized portfolios...")
        
        portfolio_results = {}
        
        for date, data in tqdm(predictions_data.items(), desc="Building portfolios"):
            predictions = data['top_stocks']
            
            # Get synthetic returns for optimization (in practice, use historical returns)
            expected_returns = predictions['future_return'].values[:30]  # Top 30 stocks
            tickers = predictions['ticker'].values[:30]
            
            # Create synthetic covariance matrix (in practice, calculate from historical data)
            n_stocks = len(expected_returns)
            correlation = 0.3  # Assume 30% correlation between stocks
            volatilities = np.abs(expected_returns) * 2 + 0.1  # Synthetic volatilities
            
            cov_matrix = np.outer(volatilities, volatilities) * correlation
            np.fill_diagonal(cov_matrix, volatilities ** 2)
            
            portfolios = {}
            
            # 1. Rank-based portfolio
            rank_tickers, rank_weights = self.rank_based_optimization(predictions)
            portfolios['rank_based'] = {
                'tickers': rank_tickers,
                'weights': rank_weights,
                'method': 'Rank-Based'
            }
            
            # 2. Mean-variance optimization
            if method in ['all', 'mean_variance']:
                mv_weights = self.mean_variance_optimization(expected_returns, cov_matrix)
                portfolios['mean_variance'] = {
                    'tickers': tickers,
                    'weights': mv_weights,
                    'method': 'Mean-Variance'
                }
            
            # 3. Maximum Sharpe ratio
            if method in ['all', 'max_sharpe']:
                sharpe_weights = self.maximum_sharpe_optimization(expected_returns, cov_matrix)
                portfolios['max_sharpe'] = {
                    'tickers': tickers,
                    'weights': sharpe_weights,
                    'method': 'Maximum Sharpe'
                }
            
            # 4. Risk parity
            if method in ['all', 'risk_parity']:
                rp_weights = self.risk_parity_optimization(cov_matrix)
                portfolios['risk_parity'] = {
                    'tickers': tickers,
                    'weights': rp_weights,
                    'method': 'Risk Parity'
                }

            # 5. Minimum variance
            if method in ['all', 'min_variance']:
                mv_weights = self.minimum_variance_optimization(cov_matrix)
                portfolios['min_variance'] = {
                    'tickers': tickers,
                    'weights': mv_weights,
                    'method': 'Minimum Variance'
                }

            # 6. Black-Litterman
            if method in ['all', 'black_litterman']:
                bl_weights = self.black_litterman_optimization(expected_returns, cov_matrix)
                portfolios['black_litterman'] = {
                    'tickers': tickers,
                    'weights': bl_weights,
                    'method': 'Black-Litterman'
                }

            # 7. Factor-based optimization (if factor data available)
            if method in ['all', 'factor_based']:
                # Use key factors for factor-based optimization
                key_factors = ['ROE', 'PE', 'PB', 'NetProfitGrowRate', 'RSI']
                available_factors = [f for f in key_factors if f in predictions.columns]

                if available_factors:
                    factor_tickers, factor_weights = self.factor_based_optimization(
                        predictions, available_factors
                    )
                    portfolios['factor_based'] = {
                        'tickers': factor_tickers,
                        'weights': factor_weights,
                        'method': 'Factor-Based'
                    }

            portfolio_results[date] = portfolios
        
        return portfolio_results
    
    def evaluate_portfolios(self, portfolio_results, predictions_data):
        """
        Evaluate portfolio performance
        """
        print("Evaluating portfolio performance...")
        
        evaluation_results = []
        
        for date, portfolios in portfolio_results.items():
            actual_returns = predictions_data[date]['predictions']
            
            for method, portfolio in portfolios.items():
                tickers = portfolio['tickers']
                weights = portfolio['weights']
                
                # Calculate portfolio return
                portfolio_return = 0
                total_weight = 0
                
                for i, ticker in enumerate(tickers):
                    if i < len(weights):
                        # Find actual return for this ticker
                        ticker_data = actual_returns[actual_returns['ticker'] == ticker]
                        if not ticker_data.empty:
                            actual_return = ticker_data['future_return'].iloc[0]
                            portfolio_return += weights[i] * actual_return
                            total_weight += weights[i]
                
                # Normalize if needed
                if total_weight > 0:
                    portfolio_return = portfolio_return / total_weight
                
                # Calculate portfolio metrics
                result = {
                    'date': date,
                    'method': portfolio['method'],
                    'portfolio_return': portfolio_return,
                    'num_stocks': len(tickers),
                    'max_weight': np.max(weights),
                    'min_weight': np.min(weights),
                    'weight_concentration': np.sum(weights ** 2),  # Herfindahl index
                    'top_5_stocks': list(tickers[:5]),
                    'top_5_weights': list(weights[:5])
                }
                
                evaluation_results.append(result)
        
        return pd.DataFrame(evaluation_results)
    
    def create_portfolio_reports(self, evaluation_df, portfolio_results):
        """
        Create comprehensive portfolio reports
        """
        print("Creating portfolio reports...")
        
        # Performance summary by method
        performance_summary = evaluation_df.groupby('method').agg({
            'portfolio_return': ['mean', 'std', 'min', 'max'],
            'num_stocks': 'mean',
            'weight_concentration': 'mean'
        }).round(4)
        
        print("\n" + "="*80)
        print("PORTFOLIO OPTIMIZATION RESULTS SUMMARY")
        print("="*80)
        print(performance_summary)
        
        # Calculate Sharpe ratios
        sharpe_ratios = evaluation_df.groupby('method')['portfolio_return'].agg(['mean', 'std'])
        sharpe_ratios['sharpe_ratio'] = (sharpe_ratios['mean'] - self.risk_free_rate/12) / sharpe_ratios['std']
        
        print(f"\nSharpe Ratios by Method:")
        print("-" * 40)
        for method, row in sharpe_ratios.iterrows():
            print(f"{method:<20}: {row['sharpe_ratio']:.4f}")
        
        # Win rates
        win_rates = evaluation_df.groupby('method')['portfolio_return'].apply(lambda x: (x > 0).mean())
        print(f"\nWin Rates by Method:")
        print("-" * 40)
        for method, win_rate in win_rates.items():
            print(f"{method:<20}: {win_rate:.2%}")
        
        # Save detailed results
        evaluation_df.to_csv('./portfolio_results/portfolio_evaluation.csv', index=False)
        performance_summary.to_csv('./portfolio_results/performance_summary.csv')
        
        # Create portfolio composition report
        self.create_portfolio_composition_report(portfolio_results)
        
        return performance_summary, sharpe_ratios, win_rates
    
    def create_portfolio_composition_report(self, portfolio_results):
        """
        Create detailed portfolio composition reports
        """
        composition_data = []
        
        for date, portfolios in portfolio_results.items():
            for method, portfolio in portfolios.items():
                tickers = portfolio['tickers']
                weights = portfolio['weights']
                
                for i, (ticker, weight) in enumerate(zip(tickers, weights)):
                    composition_data.append({
                        'date': date,
                        'method': portfolio['method'],
                        'ticker': ticker,
                        'weight': weight,
                        'rank': i + 1
                    })
        
        composition_df = pd.DataFrame(composition_data)
        composition_df.to_csv('./portfolio_results/portfolio_compositions.csv', index=False)
        
        # Top holdings analysis
        top_holdings = composition_df.groupby(['method', 'ticker'])['weight'].agg(['mean', 'count', 'std']).round(4)
        top_holdings = top_holdings.sort_values('mean', ascending=False)
        
        print(f"\nTop Holdings Across All Periods:")
        print("-" * 60)
        print(top_holdings.head(20))
        
        return composition_df

def main():
    """
    Main portfolio optimization function
    """
    print("PORTFOLIO OPTIMIZATION ON LEARN2RANK PREDICTIONS")
    print("=" * 60)
    
    # Initialize optimizer
    optimizer = PortfolioOptimizer()
    
    # Connect to database
    if not optimizer.connect_database():
        print("Failed to connect to database")
        return
    
    # Load ranking predictions
    predictions_data = optimizer.load_ranking_predictions()
    
    if not predictions_data:
        print("No prediction data found. Please run comprehensive_yuqer_learn2rank.py first.")
        return
    
    # Build portfolios using different optimization methods
    portfolio_results = optimizer.build_portfolios(predictions_data, method='all')
    
    # Evaluate portfolio performance
    evaluation_df = optimizer.evaluate_portfolios(portfolio_results, predictions_data)
    
    # Create comprehensive reports
    performance_summary, sharpe_ratios, win_rates = optimizer.create_portfolio_reports(evaluation_df, portfolio_results)
    
    # Find best performing method
    best_method = sharpe_ratios.loc[sharpe_ratios['sharpe_ratio'].idxmax()]
    
    print(f"\n" + "="*80)
    print("PORTFOLIO OPTIMIZATION COMPLETED SUCCESSFULLY!")
    print("="*80)
    print(f"Best performing method: {best_method.name}")
    print(f"Sharpe ratio: {best_method['sharpe_ratio']:.4f}")
    print(f"Average return: {best_method['mean']:.4f}")
    print(f"Volatility: {best_method['std']:.4f}")
    
    print(f"\nResults saved to:")
    print(f"- ./portfolio_results/portfolio_evaluation.csv")
    print(f"- ./portfolio_results/performance_summary.csv") 
    print(f"- ./portfolio_results/portfolio_compositions.csv")
    
    # Close connection
    if optimizer.connection:
        optimizer.connection.close()
        print("\nDatabase connection closed")

if __name__ == "__main__":
    main()
