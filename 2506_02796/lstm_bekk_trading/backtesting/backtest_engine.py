"""
Backtesting Engine

Comprehensive backtesting framework for LSTM-BEKK trading strategies.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
import logging
from datetime import datetime, timedelta
from tqdm import tqdm

from ..models import LSTMBEKKModel
from ..strategies import TradingStrategies
from ..risk import RiskManager, PerformanceMetrics
from .benchmark_models import BenchmarkModels


class BacktestEngine:
    """
    Comprehensive backtesting engine for LSTM-BEKK trading system.
    
    Implements walk-forward analysis, out-of-sample testing, and performance comparison.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize backtesting engine.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.backtest_config = self.config.get('backtesting', {})
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.risk_manager = RiskManager(config)
        self.performance_metrics = PerformanceMetrics()
        self.benchmark_models = BenchmarkModels(config)
        
        # Backtesting parameters
        self.walk_forward = self.backtest_config.get('walk_forward', True)
        self.refit_frequency = self.backtest_config.get('refit_frequency', 63)  # days
        self.initial_capital = self.config.get('trading', {}).get('portfolio', {}).get('initial_capital', 1000000)
        
        # Results storage
        self.backtest_results = {}
        self.performance_history = []
        
    def run_backtest(self, data_manager, 
                    start_date: Optional[str] = None,
                    end_date: Optional[str] = None,
                    universe: str = "sp500_sample") -> Dict:
        """
        Run comprehensive backtest.
        
        Args:
            data_manager: DataManager instance with loaded data
            start_date: Backtest start date
            end_date: Backtest end date
            universe: Stock universe to use
            
        Returns:
            Comprehensive backtest results
        """
        self.logger.info("Starting comprehensive backtest")
        
        # Load data
        if not hasattr(data_manager, 'demeaned_returns') or data_manager.demeaned_returns is None:
            data_manager.load_data(universe, start_date, end_date)
        
        returns_data = data_manager.get_returns(demeaned=True)
        
        # Split data for backtesting
        train_data = data_manager.get_train_data()
        val_data = data_manager.get_validation_data()
        test_data = data_manager.get_test_data()
        
        self.logger.info(f"Backtest data: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")
        
        # Run LSTM-BEKK strategy
        lstm_bekk_results = self._run_lstm_bekk_strategy(
            train_data, val_data, test_data, returns_data
        )
        
        # Run benchmark strategies
        benchmark_results = self._run_benchmark_strategies(test_data)
        
        # Combine results
        all_results = {
            'lstm_bekk': lstm_bekk_results,
            **benchmark_results
        }
        
        # Performance comparison
        performance_comparison = self._compare_strategies(all_results)
        
        # Risk analysis
        risk_analysis = self._analyze_risk(all_results)
        
        # Compile final results
        final_results = {
            'strategy_results': all_results,
            'performance_comparison': performance_comparison,
            'risk_analysis': risk_analysis,
            'backtest_config': {
                'start_date': test_data.index[0],
                'end_date': test_data.index[-1],
                'universe': universe,
                'n_assets': len(test_data.columns),
                'n_periods': len(test_data)
            }
        }
        
        self.backtest_results = final_results
        self.logger.info("Backtest completed successfully")
        
        return final_results
    
    def _run_lstm_bekk_strategy(self, train_data: pd.DataFrame,
                               val_data: pd.DataFrame,
                               test_data: pd.DataFrame,
                               full_returns: pd.DataFrame) -> Dict:
        """Run LSTM-BEKK strategy backtest."""
        self.logger.info("Running LSTM-BEKK strategy backtest")
        
        n_assets = len(train_data.columns)
        
        # Initialize model and strategies
        model = LSTMBEKKModel(n_assets, self.config)
        trading_strategies = TradingStrategies(self.config)
        
        # Train model
        self.logger.info("Training LSTM-BEKK model")
        training_history = model.fit(train_data, val_data, verbose=True)
        
        # Walk-forward backtesting on test data
        portfolio_returns = []
        portfolio_weights = []
        covariance_forecasts = []
        
        refit_dates = []
        current_model = model
        
        for i in tqdm(range(len(test_data)), desc="LSTM-BEKK Backtest"):
            current_date = test_data.index[i]
            
            # Refit model periodically if walk-forward is enabled
            if self.walk_forward and i > 0 and i % self.refit_frequency == 0:
                self.logger.info(f"Refitting model at {current_date}")
                
                # Expand training window
                expanded_train_end = test_data.index[i-1]
                expanded_train_data = full_returns.loc[:expanded_train_end].tail(len(train_data))
                
                # Refit model
                current_model = LSTMBEKKModel(n_assets, self.config)
                current_model.fit(expanded_train_data, verbose=False)
                refit_dates.append(current_date)
            
            # Get historical data up to current date
            historical_data = full_returns.loc[:current_date].tail(252)  # Last year
            
            # Generate covariance forecast
            try:
                cov_forecast = current_model.predict_covariance(historical_data, steps=1)[0]
                covariance_forecasts.append(cov_forecast)
                
                # Generate trading signals
                model_outputs = {
                    'covariance_forecast': cov_forecast,
                    'volatility_forecast': np.sqrt(np.diag(cov_forecast)),
                    'correlation_forecast': cov_forecast / np.outer(np.sqrt(np.diag(cov_forecast)), 
                                                                   np.sqrt(np.diag(cov_forecast)))
                }
                
                signals = trading_strategies.generate_trading_signals(
                    model_outputs, historical_data, current_date
                )
                
                weights = signals.get('final_weights')
                if weights is None:
                    weights = np.ones(n_assets) / n_assets  # Equal weights fallback
                
            except Exception as e:
                self.logger.warning(f"Error generating signals at {current_date}: {e}")
                weights = np.ones(n_assets) / n_assets
                cov_forecast = np.eye(n_assets) * 0.01
                covariance_forecasts.append(cov_forecast)
            
            portfolio_weights.append(weights)
            
            # Calculate portfolio return
            if i < len(test_data) - 1:  # Not the last period
                next_returns = test_data.iloc[i + 1]
                portfolio_return = np.sum(weights * next_returns)
                portfolio_returns.append(portfolio_return)
        
        portfolio_returns = pd.Series(portfolio_returns, index=test_data.index[:-1])
        
        return {
            'name': 'LSTM-BEKK',
            'returns': portfolio_returns,
            'weights': np.array(portfolio_weights),
            'covariance_forecasts': covariance_forecasts,
            'training_history': training_history,
            'refit_dates': refit_dates,
            'description': 'LSTM-BEKK model with dynamic rebalancing'
        }
    
    def _run_benchmark_strategies(self, test_data: pd.DataFrame) -> Dict:
        """Run benchmark strategies."""
        self.logger.info("Running benchmark strategies")
        
        # Get benchmark results
        benchmarks = self.benchmark_models.get_all_benchmarks(test_data)
        
        # Align returns to test period
        for name, benchmark in benchmarks.items():
            if isinstance(benchmark['returns'], pd.Series):
                # Align with test data
                aligned_returns = benchmark['returns'].reindex(test_data.index[:-1]).fillna(0)
                benchmark['returns'] = aligned_returns
        
        return benchmarks
    
    def _compare_strategies(self, all_results: Dict) -> pd.DataFrame:
        """Compare strategy performance."""
        self.logger.info("Comparing strategy performance")
        
        strategy_returns = {}
        for name, result in all_results.items():
            if 'returns' in result and len(result['returns']) > 0:
                strategy_returns[name] = result['returns']
        
        # Calculate comprehensive metrics for each strategy
        comparison_data = []
        
        for strategy_name, returns in strategy_returns.items():
            try:
                metrics = self.performance_metrics.calculate_comprehensive_metrics(returns)
                
                comparison_data.append({
                    'Strategy': strategy_name,
                    'Total Return': metrics['total_return'],
                    'Annualized Return': metrics['annualized_return'],
                    'Annualized Volatility': metrics['annualized_volatility'],
                    'Sharpe Ratio': metrics['sharpe_ratio'],
                    'Max Drawdown': metrics['max_drawdown'],
                    'Calmar Ratio': metrics['calmar_ratio'],
                    'VaR 5%': metrics.get('var_5', np.nan),
                    'ES 5%': metrics.get('es_5', np.nan),
                    'Skewness': metrics['skewness'],
                    'Kurtosis': metrics['kurtosis']
                })
                
            except Exception as e:
                self.logger.error(f"Error calculating metrics for {strategy_name}: {e}")
        
        comparison_df = pd.DataFrame(comparison_data).set_index('Strategy')
        
        # Rank strategies
        comparison_df['Sharpe Rank'] = comparison_df['Sharpe Ratio'].rank(ascending=False)
        comparison_df['Return Rank'] = comparison_df['Annualized Return'].rank(ascending=False)
        comparison_df['Risk Rank'] = comparison_df['Annualized Volatility'].rank(ascending=True)
        
        return comparison_df
    
    def _analyze_risk(self, all_results: Dict) -> Dict:
        """Analyze risk characteristics of strategies."""
        risk_analysis = {}
        
        for strategy_name, result in all_results.items():
            if 'returns' in result and len(result['returns']) > 0:
                returns = result['returns']
                
                try:
                    # Basic risk metrics
                    risk_metrics = {
                        'volatility': returns.std() * np.sqrt(252),
                        'downside_volatility': returns[returns < 0].std() * np.sqrt(252),
                        'var_5': -np.percentile(returns, 5),
                        'var_1': -np.percentile(returns, 1),
                        'max_daily_loss': returns.min(),
                        'max_daily_gain': returns.max(),
                        'positive_days': (returns > 0).mean(),
                        'tail_ratio': len(returns[returns < -np.percentile(returns, 5)]) / len(returns)
                    }
                    
                    risk_analysis[strategy_name] = risk_metrics
                    
                except Exception as e:
                    self.logger.error(f"Error in risk analysis for {strategy_name}: {e}")
        
        return risk_analysis
    
    def generate_backtest_report(self) -> Dict:
        """Generate comprehensive backtest report."""
        if not self.backtest_results:
            raise ValueError("No backtest results available. Run backtest first.")
        
        report = {
            'executive_summary': self._generate_executive_summary(),
            'detailed_results': self.backtest_results,
            'recommendations': self._generate_recommendations(),
            'report_date': datetime.now()
        }
        
        return report
    
    def _generate_executive_summary(self) -> Dict:
        """Generate executive summary of backtest results."""
        comparison = self.backtest_results['performance_comparison']
        
        # Find best performing strategy
        best_sharpe = comparison['Sharpe Ratio'].idxmax()
        best_return = comparison['Annualized Return'].idxmax()
        lowest_risk = comparison['Annualized Volatility'].idxmin()
        
        # LSTM-BEKK performance
        lstm_bekk_metrics = comparison.loc['LSTM-BEKK'] if 'LSTM-BEKK' in comparison.index else None
        
        summary = {
            'best_sharpe_strategy': best_sharpe,
            'best_return_strategy': best_return,
            'lowest_risk_strategy': lowest_risk,
            'lstm_bekk_performance': lstm_bekk_metrics.to_dict() if lstm_bekk_metrics is not None else None,
            'total_strategies_tested': len(comparison),
            'backtest_period': f"{self.backtest_results['backtest_config']['start_date']} to {self.backtest_results['backtest_config']['end_date']}"
        }
        
        return summary
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on backtest results."""
        recommendations = []
        
        comparison = self.backtest_results['performance_comparison']
        
        if 'LSTM-BEKK' in comparison.index:
            lstm_bekk_rank = comparison.loc['LSTM-BEKK', 'Sharpe Rank']
            
            if lstm_bekk_rank <= 2:
                recommendations.append("LSTM-BEKK shows strong performance and is recommended for implementation")
            elif lstm_bekk_rank <= len(comparison) / 2:
                recommendations.append("LSTM-BEKK shows moderate performance - consider parameter tuning")
            else:
                recommendations.append("LSTM-BEKK underperforms - review model specification and data quality")
        
        # Risk recommendations
        risk_analysis = self.backtest_results['risk_analysis']
        if 'LSTM-BEKK' in risk_analysis:
            lstm_risk = risk_analysis['LSTM-BEKK']
            if lstm_risk['volatility'] > 0.20:
                recommendations.append("Consider reducing portfolio volatility through position sizing")
            if lstm_risk['var_5'] > 0.05:
                recommendations.append("High tail risk detected - implement additional risk controls")
        
        return recommendations
