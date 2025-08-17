"""
Risk Manager

Central risk management system for the LSTM-BEKK trading framework.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
import logging
from datetime import datetime, timedelta

from .var_models import VaRCalculator, ExpectedShortfall, RiskDecomposition
from .performance_metrics import PerformanceMetrics


class RiskManager:
    """
    Central risk management system.
    
    Monitors portfolio risk, enforces limits, and provides risk reporting.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize risk manager.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.risk_config = self.config.get('trading', {}).get('risk', {})
        self.logger = logging.getLogger(__name__)
        
        # Risk limits
        self.max_portfolio_volatility = self.risk_config.get('max_portfolio_volatility', 0.15)
        self.var_confidence = self.risk_config.get('var_confidence', 0.05)
        self.es_confidence = self.risk_config.get('es_confidence', 0.01)
        self.max_drawdown_limit = self.risk_config.get('max_drawdown_limit', 0.20)
        
        # Initialize components
        self.var_calculator = VaRCalculator(self.var_confidence)
        self.es_calculator = ExpectedShortfall(self.es_confidence)
        self.risk_decomposition = RiskDecomposition()
        self.performance_metrics = PerformanceMetrics()
        
        # Risk monitoring state
        self.risk_alerts = []
        self.risk_history = []
        
    def assess_portfolio_risk(self, weights: np.ndarray,
                            covariance_matrix: np.ndarray,
                            returns_history: pd.DataFrame,
                            current_date: datetime) -> Dict:
        """
        Comprehensive portfolio risk assessment.
        
        Args:
            weights: Current portfolio weights
            covariance_matrix: Covariance matrix forecast
            returns_history: Historical returns
            current_date: Current date
            
        Returns:
            Risk assessment report
        """
        risk_report = {
            'date': current_date,
            'portfolio_volatility': None,
            'var_estimates': {},
            'es_estimates': {},
            'risk_decomposition': {},
            'risk_alerts': [],
            'risk_limits_breached': False
        }
        
        try:
            # Calculate portfolio volatility
            portfolio_variance = weights.T @ covariance_matrix @ weights
            portfolio_volatility = np.sqrt(portfolio_variance * 252)  # Annualized
            risk_report['portfolio_volatility'] = portfolio_volatility
            
            # Check volatility limit
            if portfolio_volatility > self.max_portfolio_volatility:
                alert = {
                    'type': 'volatility_breach',
                    'message': f"Portfolio volatility {portfolio_volatility:.3f} exceeds limit {self.max_portfolio_volatility:.3f}",
                    'severity': 'high'
                }
                risk_report['risk_alerts'].append(alert)
                risk_report['risk_limits_breached'] = True
            
            # Calculate portfolio returns
            portfolio_returns = (returns_history * weights).sum(axis=1)
            
            # VaR calculations
            var_parametric = self.var_calculator.parametric_var(
                portfolio_returns.values, covariance_matrix, weights
            )
            var_historical = self.var_calculator.historical_var(portfolio_returns.values)
            var_monte_carlo = self.var_calculator.monte_carlo_var(
                portfolio_returns.mean(), covariance_matrix, weights
            )
            
            risk_report['var_estimates'] = {
                'parametric': var_parametric,
                'historical': var_historical,
                'monte_carlo': var_monte_carlo
            }
            
            # Expected Shortfall calculations
            es_parametric = self.es_calculator.parametric_es(
                portfolio_returns.values, covariance_matrix, weights
            )
            es_historical = self.es_calculator.historical_es(portfolio_returns.values)
            
            risk_report['es_estimates'] = {
                'parametric': es_parametric,
                'historical': es_historical
            }
            
            # Risk decomposition
            component_analysis = self.risk_decomposition.component_var(
                weights, covariance_matrix, self.var_confidence
            )
            risk_report['risk_decomposition'] = component_analysis
            
            # Drawdown check
            if len(portfolio_returns) > 1:
                drawdown_metrics = self.performance_metrics.calculate_drawdown_metrics(portfolio_returns)
                current_drawdown = abs(drawdown_metrics['drawdown_series'].iloc[-1])
                
                if current_drawdown > self.max_drawdown_limit:
                    alert = {
                        'type': 'drawdown_breach',
                        'message': f"Current drawdown {current_drawdown:.3f} exceeds limit {self.max_drawdown_limit:.3f}",
                        'severity': 'high'
                    }
                    risk_report['risk_alerts'].append(alert)
                    risk_report['risk_limits_breached'] = True
            
        except Exception as e:
            self.logger.error(f"Error in portfolio risk assessment: {e}")
            risk_report['error'] = str(e)
        
        # Store risk history
        self.risk_history.append(risk_report)
        
        # Store alerts
        self.risk_alerts.extend(risk_report['risk_alerts'])
        
        return risk_report
    
    def check_position_limits(self, weights: np.ndarray,
                            asset_names: List[str]) -> Dict:
        """
        Check individual position limits.
        
        Args:
            weights: Portfolio weights
            asset_names: Asset names
            
        Returns:
            Position limit check results
        """
        portfolio_config = self.config.get('trading', {}).get('portfolio', {})
        max_position = portfolio_config.get('max_position_size', 0.1)
        min_position = portfolio_config.get('min_position_size', 0.01)
        
        violations = []
        
        for i, (weight, asset) in enumerate(zip(weights, asset_names)):
            if abs(weight) > max_position:
                violations.append({
                    'asset': asset,
                    'weight': weight,
                    'limit': max_position,
                    'type': 'max_position_exceeded'
                })
            elif abs(weight) < min_position and abs(weight) > 1e-6:
                violations.append({
                    'asset': asset,
                    'weight': weight,
                    'limit': min_position,
                    'type': 'min_position_violated'
                })
        
        return {
            'violations': violations,
            'compliant': len(violations) == 0,
            'max_weight': np.max(np.abs(weights)),
            'concentration_ratio': np.sum(weights ** 2)  # Herfindahl index
        }
    
    def stress_test_portfolio(self, weights: np.ndarray,
                            covariance_matrix: np.ndarray,
                            stress_scenarios: Optional[Dict] = None) -> Dict:
        """
        Perform stress testing on the portfolio.
        
        Args:
            weights: Portfolio weights
            covariance_matrix: Base covariance matrix
            stress_scenarios: Custom stress scenarios
            
        Returns:
            Stress test results
        """
        if stress_scenarios is None:
            stress_scenarios = self._get_default_stress_scenarios()
        
        stress_results = {}
        
        for scenario_name, scenario in stress_scenarios.items():
            try:
                # Apply stress to covariance matrix
                stressed_cov = self._apply_stress_scenario(covariance_matrix, scenario)
                
                # Calculate stressed portfolio metrics
                stressed_variance = weights.T @ stressed_cov @ weights
                stressed_volatility = np.sqrt(stressed_variance * 252)
                
                # Calculate stressed VaR
                stressed_var = self.var_calculator.parametric_var(
                    np.array([0]), stressed_cov, weights  # Assume zero expected return for stress test
                )
                
                stress_results[scenario_name] = {
                    'stressed_volatility': stressed_volatility,
                    'stressed_var': stressed_var,
                    'volatility_change': stressed_volatility / np.sqrt(weights.T @ covariance_matrix @ weights * 252) - 1,
                    'scenario_description': scenario.get('description', '')
                }
                
            except Exception as e:
                self.logger.error(f"Error in stress scenario {scenario_name}: {e}")
                stress_results[scenario_name] = {'error': str(e)}
        
        return stress_results
    
    def _get_default_stress_scenarios(self) -> Dict:
        """Get default stress testing scenarios."""
        return {
            'market_crash': {
                'description': '2008-style market crash',
                'volatility_multiplier': 2.0,
                'correlation_increase': 0.3
            },
            'correlation_breakdown': {
                'description': 'Correlation breakdown scenario',
                'volatility_multiplier': 1.5,
                'correlation_multiplier': 0.5
            },
            'high_volatility': {
                'description': 'High volatility regime',
                'volatility_multiplier': 1.8,
                'correlation_increase': 0.1
            }
        }
    
    def _apply_stress_scenario(self, covariance_matrix: np.ndarray,
                             scenario: Dict) -> np.ndarray:
        """Apply stress scenario to covariance matrix."""
        stressed_cov = covariance_matrix.copy()
        
        # Extract volatilities and correlations
        volatilities = np.sqrt(np.diag(covariance_matrix))
        correlations = covariance_matrix / np.outer(volatilities, volatilities)
        
        # Apply volatility stress
        vol_multiplier = scenario.get('volatility_multiplier', 1.0)
        stressed_volatilities = volatilities * vol_multiplier
        
        # Apply correlation stress
        if 'correlation_increase' in scenario:
            corr_increase = scenario['correlation_increase']
            stressed_correlations = correlations + corr_increase * (1 - correlations)
            np.fill_diagonal(stressed_correlations, 1.0)
        elif 'correlation_multiplier' in scenario:
            corr_multiplier = scenario['correlation_multiplier']
            stressed_correlations = correlations * corr_multiplier
            np.fill_diagonal(stressed_correlations, 1.0)
        else:
            stressed_correlations = correlations
        
        # Reconstruct covariance matrix
        stressed_cov = np.outer(stressed_volatilities, stressed_volatilities) * stressed_correlations
        
        return stressed_cov
    
    def generate_risk_report(self, portfolio_returns: pd.Series,
                           benchmark_returns: Optional[pd.Series] = None) -> Dict:
        """
        Generate comprehensive risk report.
        
        Args:
            portfolio_returns: Portfolio return series
            benchmark_returns: Benchmark return series
            
        Returns:
            Comprehensive risk report
        """
        report = {
            'report_date': datetime.now(),
            'performance_metrics': {},
            'risk_alerts_summary': {},
            'recent_risk_assessments': [],
            'recommendations': []
        }
        
        # Performance metrics
        report['performance_metrics'] = self.performance_metrics.calculate_comprehensive_metrics(
            portfolio_returns, benchmark_returns
        )
        
        # Risk alerts summary
        recent_alerts = [alert for alert in self.risk_alerts if alert.get('date', datetime.now()) > datetime.now() - timedelta(days=30)]
        report['risk_alerts_summary'] = {
            'total_alerts': len(self.risk_alerts),
            'recent_alerts': len(recent_alerts),
            'high_severity_alerts': len([a for a in recent_alerts if a.get('severity') == 'high'])
        }
        
        # Recent risk assessments
        report['recent_risk_assessments'] = self.risk_history[-10:] if len(self.risk_history) > 10 else self.risk_history
        
        # Generate recommendations
        report['recommendations'] = self._generate_risk_recommendations(report)
        
        return report
    
    def _generate_risk_recommendations(self, risk_report: Dict) -> List[str]:
        """Generate risk management recommendations."""
        recommendations = []
        
        # Check performance metrics
        metrics = risk_report['performance_metrics']
        
        if metrics.get('sharpe_ratio', 0) < 0.5:
            recommendations.append("Consider improving risk-adjusted returns (Sharpe ratio < 0.5)")
        
        if abs(metrics.get('max_drawdown', 0)) > 0.15:
            recommendations.append("High maximum drawdown detected - consider reducing portfolio risk")
        
        if metrics.get('annualized_volatility', 0) > self.max_portfolio_volatility:
            recommendations.append("Portfolio volatility exceeds target - consider rebalancing")
        
        # Check recent alerts
        alerts_summary = risk_report['risk_alerts_summary']
        if alerts_summary.get('high_severity_alerts', 0) > 0:
            recommendations.append("Address high-severity risk alerts immediately")
        
        return recommendations
